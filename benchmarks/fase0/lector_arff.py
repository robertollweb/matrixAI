# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Un lector de ARFF que aguanta lo que OpenML sirve de verdad.

**CUATRO de los CUARENTA datasets del protocolo no se podían leer** —el 10 %—
y nadie lo sabía porque la pasada exploratoria del 101-C3 solo usa doce, y
ninguno de los cuatro está entre ellos. Medido el 2026-09-12:

  · `pendigits`   — `ValueError:  8 value not in ('0',…,'9')`
  · `diamonds`    — `ValueError: 'Very Good' value not in ('Fair','Good',…)`
  · `house_prices_nominal` — `ValueError: 'Wd Sdng' value not in (…)`
  · `us_crime`    — `NotImplementedError: String attributes not supported`

Cuando 101-C5 llegue a ellos fallarán, y la regla de cierre cuenta un fallo
como **dataset perdido para ese motor**: cuatro datasets perdidos para los
siete motores a la vez, por un problema de formato que no tiene nada que ver
con la calidad de ningún modelo.

EL FICHERO NO SE TOCA. Se normaliza una copia EN MEMORIA, así que el sha256
que el protocolo registra sigue valiendo — es la huella del dato original, y
cambiarla para poder leerlo sería romper justo lo que la hace útil.

LAS TRES COSAS QUE SE NORMALIZAN, y ninguna cambia un valor:

1. **Espacios de alineación en los datos.** `pendigits` escribe `, 8` para que
   las columnas queden cuadradas a la vista, y su declaración dice `{0,…,9}`
   sin espacios. `scipy` compara literal y no encuentra `" 8"`. Se recortan
   los espacios de los bordes de cada campo.

2. **Comillas y espacios en la declaración nominal.** `diamonds` declara
   `{Fair, Good, Ideal, Premium, 'Very Good'}` y sus datos traen `Very Good`
   sin comillas. Se quitan las comillas de la declaración y se recortan sus
   espacios, para que los dos lados digan lo mismo.

3. **Atributos `STRING`.** `scipy` no los soporta y no hay forma de
   convertirlos sin inventar un vocabulario. En `us_crime` son el nombre del
   municipio: un IDENTIFICADOR, no un predictor — y además una fuga en
   potencia, porque identifica la fila. Se declaran como el que son y se
   EXCLUYEN, con su nombre en el informe: no se convierten en categóricas en
   silencio, que sería meterle al modelo una columna de identificadores.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ATRIBUTO = re.compile(r"^\s*@attribute\s+(?P<nombre>'[^']*'|\"[^\"]*\"|\S+)\s+(?P<tipo>.+?)\s*$",
                       re.IGNORECASE)


@dataclass
class ArffLeido:
    """Lo leído, y QUÉ hubo que hacer para leerlo.

    `normalizaciones` y `columnas_excluidas` no son adorno: un fichero que se
    ha tenido que retocar para entrar no es el mismo que entró tal cual, y
    quien lea los números tiene derecho a saber cuál de los dos está mirando.
    """

    filas: list[dict[str, Any]]
    objetivo: str
    normalizaciones: list[str] = field(default_factory=list)
    columnas_excluidas: list[str] = field(default_factory=list)


def _sin_comillas(texto: str) -> str:
    t = texto.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    return t


def normalizar_texto_arff(texto: str) -> tuple[str, list[str], list[str]]:
    """Devuelve `(texto normalizado, qué se normalizó, columnas excluidas)`."""
    notas: list[str] = []
    excluidas: list[str] = []
    lineas = texto.splitlines()
    cabecera: list[str] = []
    indice_de_columna = 0
    fuera: list[int] = []
    i = 0
    for i, linea in enumerate(lineas):
        if linea.lower().startswith("@data"):
            break
        m = _ATRIBUTO.match(linea)
        if m is None:
            cabecera.append(linea)
            continue
        nombre, tipo = _sin_comillas(m.group("nombre")), m.group("tipo").strip()
        if tipo.lower() == "string":
            # Se EXCLUYE, no se convierte: una columna de texto libre es un
            # identificador o una nota, y meterla como categórica le daría al
            # modelo una columna que identifica la fila.
            fuera.append(indice_de_columna)
            excluidas.append(nombre)
            indice_de_columna += 1
            continue
        if tipo.startswith("{") and tipo.endswith("}"):
            valores = [_sin_comillas(v) for v in tipo[1:-1].split(",")]
            limpio = "{" + ",".join(valores) + "}"
            if limpio != tipo:
                notas.append(f"declaración nominal de {nombre!r}: comillas y espacios recortados")
            cabecera.append(f"@attribute '{nombre}' {limpio}")
        else:
            cabecera.append(f"@attribute '{nombre}' {tipo}")
        indice_de_columna += 1

    cuerpo: list[str] = []
    recorto = False
    for linea in lineas[i + 1:]:
        if not linea.strip() or linea.lstrip().startswith("%"):
            continue
        campos = [c.strip() for c in linea.split(",")]
        if len(campos) != indice_de_columna:
            cuerpo.append(linea)          # fila rara: se deja y que hable scipy
            continue
        if any(c != d for c, d in zip(campos, linea.split(","))):
            recorto = True
        campos = [_sin_comillas(c) if c[:1] in "'\"" else c for c in campos]
        cuerpo.append(",".join(c for j, c in enumerate(campos) if j not in fuera))
    if recorto:
        notas.append("espacios de alineación recortados en los datos")
    if excluidas:
        notas.append(f"columnas STRING excluidas: {', '.join(excluidas)}")
    return "\n".join(cabecera + ["@data"] + cuerpo) + "\n", notas, excluidas


def cargar(ruta: Path, objetivo_declarado: str | None = None) -> ArffLeido:
    """Lee el ARFF, normalizando solo si hace falta.

    Se INTENTA primero tal cual: si el fichero entra sin retoques, no se toca
    nada y `normalizaciones` sale vacía. Normalizar siempre convertiría en
    invisible la diferencia entre un fichero limpio y uno que hubo que
    arreglar.
    """
    from scipy.io import arff as _arff

    crudo = ruta.read_text(encoding="utf-8", errors="replace")
    notas: list[str] = []
    excluidas: list[str] = []
    try:
        datos, meta = _arff.loadarff(io.StringIO(crudo))
    except Exception:  # noqa: BLE001 — el motivo da igual: se reintenta normalizado
        normalizado, notas, excluidas = normalizar_texto_arff(crudo)
        datos, meta = _arff.loadarff(io.StringIO(normalizado))

    nombres = meta.names()
    tipos = dict(zip(nombres, meta.types()))
    objetivo = objetivo_declarado if objetivo_declarado in nombres else nombres[-1]
    filas: list[dict[str, Any]] = []
    for i, registro in enumerate(datos):
        fila: dict[str, Any] = {"row_id": f"{ruta.stem}-{i}"}
        for nombre in nombres:
            valor = registro[nombre]
            if tipos[nombre] == "numeric":
                fila[nombre] = None if valor != valor else float(valor)
            else:
                texto = valor.decode() if isinstance(valor, (bytes, bytearray)) else str(valor)
                fila[nombre] = None if texto in ("", "?") else texto
        filas.append(fila)
    return ArffLeido(filas=filas, objetivo=objetivo, normalizaciones=notas,
                     columnas_excluidas=excluidas)
