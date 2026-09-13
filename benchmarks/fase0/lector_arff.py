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

Y DOS COSAS QUE SE DECLARAN, que no son normalización del texto:

4. **El `?` del estándar ARFF es un FALTANTE.** Es el marcador de ausencia del
   formato, y `scipy` lo entrega tal cual —la cadena `"?"`— en las columnas
   nominales (en las numéricas ya sale `NaN`). Quien lo reciba como texto se
   lo mete al modelo como una CATEGORÍA más. Medido el 2026-09-13: `sick`
   tiene 150 de sus 3.772 filas con `sex == "?"`, y por ese camino ni se
   imputaban ni activaban el indicador de faltante en NINGUNO de los cuatro
   motores. Aquí se convierte en `None`, que es lo que `preparacion` entiende.

5. **La columna IDENTIFICADORA que el catálogo dice que sobra**, derivada de
   lo ya declarado y no de un umbral inventado aquí. Ver el bloque largo antes
   de `cargar`.
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

    `motivos_de_exclusion` dice POR QUÉ se fue cada columna, con el número que
    lo decidió. Una exclusión silenciosa es un dato que desaparece sin rastro;
    una exclusión que solo dice el nombre obliga a quien audita a reconstruir
    el criterio desde el código. `avisos` es lo que NO se pudo decidir: un
    descuadre entre lo que el catálogo declara y lo que el fichero trae se
    cuenta en voz alta en vez de resolverse a ojo.
    """

    filas: list[dict[str, Any]]
    objetivo: str
    normalizaciones: list[str] = field(default_factory=list)
    columnas_excluidas: list[str] = field(default_factory=list)
    motivos_de_exclusion: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


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


# --- La columna IDENTIFICADORA que el catálogo dice que sobra ---------------
#
# CINCO de los cuarenta traen una columna más de las que el catálogo declara:
# `splice` (+1), `house_prices_nominal` (+1), `Allstate_Claims_Severity` (+1),
# `house_sales` (+1) y `us_crime` (+1). Medido el 2026-09-13 comparando
# `n_columnas` del protocolo registrado con los `@attribute` de cada ARFF: los
# otros treinta y cinco cuadran exactamente.
#
# La de más es SIEMPRE el identificador de fila, que OpenML no cuenta como
# columna. Metido al modelo es una FUGA: `splice` le daría `Instance_name`,
# **3.178 valores distintos para 3.190 filas**.
#
# POR QUÉ NO SE EXCLUYE «LO QUE TENGA MUCHOS NIVELES». Porque eso tiraría una
# categórica legítima de alta cardinalidad, y en el protocolo hay una de
# verdad: `KDDCup09_appetency.Var200`, con **15.415 niveles**. Los dos números,
# como proporción de las filas del dataset:
#
#     splice.Instance_name        3.178 / 3.190  = 0,9962   <- identificador
#     KDDCup09_appetency.Var200  15.415 / 50.000 = 0,3083   <- categórica real
#     Amazon_employee_access.RESOURCE  7.518 / 32.769 = 0,2294
#     okcupid-stem.speaks              7.019 / 50.789 = 0,1382
#
# No se rozan: lo que distingue al identificador no es tener muchos niveles,
# es tener CASI TANTOS VALORES DISTINTOS COMO FILAS.
#
# Y AUN ASÍ NO SE ELIGE UN UMBRAL AQUÍ, por dos razones:
#
# 1. El número ya está declarado en el núcleo. `dataset_analysis` decide desde
#    el contrato 71 qué es un identificador (`_IDENTIFIER_UNIQUE_RATIO`, la
#    densidad de tramo del id secuencial, y el NOMBRE para el id esparcido),
#    con los dos lados medidos. Escribir aquí un segundo criterio sería el
#    defecto de siempre: dos sitios declarando lo mismo acaban divergiendo.
#    Se le PREGUNTA al núcleo, por su API pública.
#
# 2. CUÁNTAS sobran tampoco se adivina: lo dice el catálogo. `n_columnas` del
#    protocolo registrado menos los `@attribute` que el fichero trae da el
#    sobrante EXACTO, y solo se excluye si el núcleo señala exactamente esas.
#    Si los dos no coinciden no se elige el que convenga: se deja todo y se
#    dice en voz alta (`avisos`).
#
# `us_crime` es el caso que enseña que las dos señales son independientes: su
# columna de más es `communityname`, y su `unique_ratio` es 0,9168 — el núcleo
# NO la llama identificador. Se va igual, por STRING, y después de eso su
# cuenta ya cuadra con el catálogo y no sobra nada.


def _texto_para_el_nucleo(valor: Any) -> str:
    """El valor como lo escribiría un CSV, que es lo que el núcleo analiza.

    `None` sale como celda vacía —el núcleo la cuenta como nula, igual que
    `"?"`—, y un `float` sale con su `str`: `1.0` le sigue pareciendo entero
    (`float.is_integer()`), que es lo que hace falta para que un `Id` guardado
    por `scipy` en un `float64` no deje de parecer el entero que el ARFF
    declaró.
    """
    return "" if valor is None else str(valor)


def columnas_que_el_nucleo_llama_identificador(
        filas: list[dict[str, Any]], columnas: list[str]) -> dict[str, dict[str, Any]]:
    """Le pregunta a `dataset_analysis` cuáles de `columnas` son un id.

    Devuelve `{columna: lo que el núcleo midió}` — el número con el que lo
    afirmó, no solo el veredicto.

    EL PRE-FILTRO NO ES UN CRITERIO NUEVO, y por eso no puede cambiar ninguna
    respuesta: el núcleo exige `unique_ratio >= _IDENTIFIER_UNIQUE_RATIO` como
    condición NECESARIA en sus dos ramas de identificador, así que una columna
    que no llega a ese ratio jamás sale `identifier`. Se usa su propia
    constante, importada y no copiada, y sirve solo para no convertir en CSV
    las 132 columnas de `Allstate_Claims_Severity` (188.318 filas) cuando la
    pregunta es por una sola.
    """
    import csv as _csv
    import io as _io

    from matrixai.training.dataset_analysis import (_IDENTIFIER_MIN_ROWS,
                                                    _IDENTIFIER_UNIQUE_RATIO,
                                                    analyze_dataset_csv)

    candidatas: list[str] = []
    for columna in columnas:
        presentes = [f.get(columna) for f in filas]
        presentes = [v for v in presentes if v is not None]
        if len(presentes) < _IDENTIFIER_MIN_ROWS:
            continue
        if len(set(presentes)) / len(presentes) >= _IDENTIFIER_UNIQUE_RATIO:
            candidatas.append(columna)
    if not candidatas:
        return {}

    buffer = _io.StringIO()
    escritor = _csv.writer(buffer)
    escritor.writerow(candidatas)
    for fila in filas:
        escritor.writerow([_texto_para_el_nucleo(fila.get(c)) for c in candidatas])
    analisis = analyze_dataset_csv(buffer.getvalue())
    return {c: analisis["columns"][c] for c in candidatas
            if analisis["columns"][c].get("type") == "identifier"}


def _identificadores_que_sobran(
        filas: list[dict[str, Any]], columnas: list[str], objetivo: str,
        n_columnas_declaradas: int | None) -> tuple[list[str], dict[str, str], list[str]]:
    """Las columnas identificadoras que el CATÁLOGO dice que sobran.

    Devuelve `(a excluir, motivo por columna, avisos)`. Sin `n_columnas_
    declaradas` no se excluye nada: el criterio entero se apoya en que alguien
    haya declarado cuántas columnas tiene el dataset, y sin esa declaración
    esto sería exactamente la heurística que no se quiere.
    """
    if n_columnas_declaradas is None:
        return [], {}, []
    sobrante = len(columnas) - n_columnas_declaradas
    if sobrante == 0:
        return [], {}, []
    if sobrante < 0:
        return [], {}, [
            f"el fichero trae {len(columnas)} columnas y el catálogo declara "
            f"{n_columnas_declaradas}: FALTAN {-sobrante}, no sobran. No se excluye nada."]

    medidas = columnas_que_el_nucleo_llama_identificador(
        filas, [c for c in columnas if c != objetivo])
    if len(medidas) != sobrante:
        return [], {}, [
            f"el catálogo declara {n_columnas_declaradas} columnas y el fichero trae "
            f"{len(columnas)}: sobran {sobrante}, pero el núcleo llama identificador a "
            f"{len(medidas)} ({', '.join(sorted(medidas)) or 'ninguna'}). No se excluye "
            f"nada: cuál sobra no se adivina."]

    motivos = {
        columna: (
            f"identificador de fila: el catálogo declara {n_columnas_declaradas} columnas "
            f"y el ARFF trae {len(columnas)}; el núcleo la clasifica `identifier` con "
            f"{medida.get('cardinality')} valores distintos en {len(filas)} filas "
            f"(unique_ratio={medida.get('unique_ratio')})")
        for columna, medida in medidas.items()}
    return sorted(medidas), motivos, []


def cargar(ruta: Path, objetivo_declarado: str | None = None,
           n_columnas_declaradas: int | None = None) -> ArffLeido:
    """Lee el ARFF, normalizando solo si hace falta.

    Se INTENTA primero tal cual: si el fichero entra sin retoques, no se toca
    nada y `normalizaciones` sale vacía. Normalizar siempre convertiría en
    invisible la diferencia entre un fichero limpio y uno que hubo que
    arreglar.

    `n_columnas_declaradas` es el `n_columnas` que el protocolo registró para
    este dataset. Con él —y SOLO con él— se puede excluir la columna
    identificadora que OpenML no cuenta; ver el bloque de arriba. Sin él el
    lector no inventa nada y devuelve todas las columnas que el fichero trae.
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

    motivos = {c: "atributo STRING: `scipy` no lo soporta y convertirlo en "
                  "categórica sería meterle al modelo texto libre que identifica la fila"
               for c in excluidas}
    sobran, motivos_id, avisos = _identificadores_que_sobran(
        filas, list(nombres), objetivo, n_columnas_declaradas)
    for columna in sobran:
        for fila in filas:
            fila.pop(columna, None)
    # NO va a `normalizaciones` a proposito: eso declara lo que hubo que
    # hacerle al TEXTO para que `scipy` lo leyera, y esta exclusion ocurre
    # despues de leerlo. Mezclarlas volveria `normalizaciones` un cajon y
    # dejaria de poder distinguirse un fichero que hubo que retocar de uno que
    # entro limpio. Se declara en `columnas_excluidas` con su motivo.
    excluidas = list(excluidas) + sobran
    motivos.update(motivos_id)
    return ArffLeido(filas=filas, objetivo=objetivo, normalizaciones=notas,
                     columnas_excluidas=excluidas, motivos_de_exclusion=motivos,
                     avisos=avisos)
