# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C2 — el decimal español (`12,5`, `1.234,56`) de un CSV con `;`.

Una sola regla, pura (texto -> decisión), que usa `normalize_csv_delimiter` y por tanto TODO lo que
entra por `normalize_csv_text`: el análisis, el entrenamiento, la verificación, el estudio y el lote.
Nada de esto vive en el backend: allí solo se llama a `normalize_csv_text` (dos sitios declarando lo
mismo acaban divergiendo).

LA REGLA (contrato 150, C2)
  * Solo en un CSV que YA es `;` (la señal conservadora de `normalize_csv_delimiter`). Un CSV con `,`
    como separador no se toca nunca: en inglés `1,234` son miles.
  * Por COLUMNA y TODO O NADA. Una columna se convierte si TODAS sus celdas no vacías son enteros o
    decimales con coma y AL MENOS UNA lleva coma. Se quita el punto de millar y la coma pasa a punto.
  * Una celda que no encaja deja la columna ENTERA como estaba, y se dice por qué.
  * Una columna con `1.234` sin coma es AMBIGUA (¿1,234 o 1234?): no se toca, con aviso.

QUÉ ES UNA CELDA DE CADA CLASE (sin espacios de los bordes; las vacías y las marcas de ausencia del
análisis —`NA`, `n/a`, `null`, `nan`, `none`, `-`, `?`— no cuentan, igual que en `analyze_dataset_csv`)
  * ``E`` entero: `12`, `-3`, `0`. Con ceros a la izquierda (`007`) NO es un entero: es un código.
  * ``D`` decimal con coma: `12,5`, `-3,25`, `1.234,56`, `0,5`. Con ceros a la izquierda en la parte
    entera (`01,02`, `00,5`) NO es un decimal: es un código que se parece a uno.
  * ``M`` miles sin coma: `1.234`, `12.345.678`. Ambiguo.
  * ``P`` decimal con punto: `12.5`, `3.14159`.
  * ``PCT`` porcentaje con coma: `12,5 %`. El símbolo lo hace texto: no se convierte (y nada se calcula).
  * ``X`` todo lo demás.

El texto de las declaraciones lo redacta el núcleo (invariante 9 del contrato): la interfaz solo lo enseña.
Cada declaración es `{tipo, columna, cuantas, detalle, texto: {es, en}}`, la misma forma que las de
`matrixai.tablas` (C1) para que el recibo (C4) las junte sin traducirlas.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["ClasificadorDeColumnas", "convertir_celda", "convertir_celda_de_una_columna_decidida",
           "TIPOS_DE_DECLARACION"]

_ENTERO = r"(?:0|[1-9]\d*)"
_CON_MILES = r"[1-9]\d{0,2}(?:\.\d{3})+"

_RE_E = re.compile(rf"^[+-]?{_ENTERO}$")
_RE_D = re.compile(rf"^[+-]?(?:{_ENTERO}|{_CON_MILES}),\d+$")
_RE_M = re.compile(rf"^[+-]?{_CON_MILES}$")
_RE_P = re.compile(r"^[+-]?\d+\.\d+$")
_RE_PCT = re.compile(rf"^[+-]?(?:{_ENTERO}|{_CON_MILES}),\d+\s*%$")

TIPOS_DE_DECLARACION = ("decimal_coma_convertido", "decimal_coma_no_convertido")


_CLASES = ("E", "D", "M", "P", "PCT", "X")
_E, _D, _M, _P, _PCT, _X = range(6)


def _clase(celda: str) -> int:
    """`_E`, `_D`, `_M`, `_P`, `_PCT` o `_X` (ver el docstring del módulo). La celda ya viene sin espacios."""
    if "," in celda:
        if _RE_D.match(celda):
            return _D
        if _RE_PCT.match(celda):
            return _PCT
        return _X
    if "." in celda:
        if not _RE_P.match(celda):
            return _X
        return _M if _RE_M.match(celda) else _P
    return _E if _RE_E.match(celda) else _X


def convertir_celda(celda: str) -> str:
    """`'  1.234,5 '` -> `'  1234.5 '` (los espacios de los bordes se conservan). Solo se llama sobre celdas
    de clase ``D``; una celda de otra clase se devuelve como llegó."""
    nucleo = celda.strip()
    if _clase(nucleo) != _D:
        return celda
    nuevo = nucleo.replace(".", "").replace(",", ".")
    inicio = celda.index(nucleo) if nucleo else 0
    return celda[:inicio] + nuevo + celda[inicio + len(nucleo):]


def convertir_celda_de_una_columna_decidida(celda: str) -> str:
    """Lo mismo que `convertir_celda`, para una celda de una columna que `ClasificadorDeColumnas` YA decidió convertir: sus
    celdas con dato son todas `E` o `D`, así que una con `,` es un decimal con coma y no hace falta volver a comprobarlo con
    la regex (medido: la comprobación doble costaba más que la lectura del CSV en una tabla de 51.000 × 100)."""
    if "," not in celda:
        return celda
    nucleo = celda.strip()
    nuevo = nucleo.replace(".", "").replace(",", ".")
    if nucleo == celda:
        return nuevo
    inicio = celda.index(nucleo)
    return celda[:inicio] + nuevo + celda[inicio + len(nucleo):]


def _p(n: int, uno: str, varios: str) -> str:
    """`uno` si n == 1; si no, `varios` con `{n}` puesto (ningún texto dice «1 celdas»)."""
    return uno if n == 1 else varios.replace("{n}", str(n))


class ClasificadorDeColumnas:
    """Cuenta, por columna, las celdas de cada clase en UNA pasada (sin guardar las filas) y decide.

    Uso: `ver(fila)` por cada fila de DATOS (la cabecera se pasa a `__init__`), y luego `a_convertir()`
    (los índices de columna que se convierten) y `declaraciones()` (lo que se hizo y lo que se dejó).

    COSTE. Medido con 51.000 × 100 (30 MB): la clasificación celda a celda con una regex por celda costaba
    9 veces lo que el resto de la normalización. Por eso `ver` descarta primero por la forma más barata (un
    entero sin `,` ni `.` no pasa por ninguna regex, un texto que empieza por letra tampoco) y solo las celdas
    con `,` o `.` llegan a `_clase`.
    """

    def __init__(self, cabecera: list[str]) -> None:
        from matrixai.training.dataset_analysis import _NULL_TOKENS  # noqa: PLC0415
        self._ausentes = _NULL_TOKENS
        self._largo_maximo_de_una_marca = max(len(t) for t in _NULL_TOKENS)
        self._cabecera = [str(c) for c in cabecera]
        # Por columna: sus seis contadores (o `None` si no ha visto ninguna celda con dato), y si está «muerta».
        self._cuentas: list[list[int] | None] = [None] * len(self._cabecera)
        # Una columna con una celda que no es número ya no se convertirá NUNCA (todo o nada): desde ahí solo se mira si
        # una celda tiene forma de decimal con coma (`D`) o de porcentaje (`PCT`) para poder decirlo; lo demás no se cuenta.
        self._muertas: list[bool] = [False] * len(self._cabecera)
        # Primer ejemplo de cada clase, por columna: (texto de la celda, nº de fila de datos, 1-based).
        self._ejemplos: list[dict[int, tuple[str, int]]] = [{} for _ in self._cabecera]
        self._fila = 0

    def _ensanchar(self, n: int) -> None:
        falta = n - len(self._cuentas)
        self._cuentas += [None] * falta
        self._muertas += [False] * falta
        self._ejemplos += [{} for _ in range(falta)]
        self._cabecera += [f"columna {len(self._cabecera) + k + 1}" for k in range(falta)]

    def _contar(self, i: int, clase: int, nucleo: str) -> None:
        c = self._cuentas[i]
        if c is None:
            c = self._cuentas[i] = [0, 0, 0, 0, 0, 0]
        if c[clase] == 0:
            self._ejemplos[i][clase] = (nucleo, self._fila)
        c[clase] += 1
        if clase == _X:
            self._muertas[i] = True

    def ver(self, fila: list[str]) -> None:
        self._fila += 1
        if len(fila) > len(self._cuentas):
            self._ensanchar(len(fila))
        muertas = self._muertas
        ausentes = self._ausentes
        largo = self._largo_maximo_de_una_marca
        for i, celda in enumerate(fila):
            if muertas[i]:
                if "," in celda:
                    nucleo = celda.strip()
                    clase = _clase(nucleo)
                    if clase == _D or clase == _PCT:
                        self._contar(i, clase, nucleo)
                continue
            nucleo = celda.strip()
            if not nucleo:
                continue
            if "," in nucleo or "." in nucleo:
                clase = _clase(nucleo)
            elif nucleo.isdigit() and nucleo.isascii():
                clase = _E if (nucleo[0] != "0" or len(nucleo) == 1) else _X
            elif len(nucleo) <= largo and nucleo.lower() in ausentes:
                continue
            elif nucleo[0].isalpha():
                clase = _X
            else:
                clase = _clase(nucleo)
            self._contar(i, clase, nucleo)

    def _cuenta(self, i: int) -> dict[str, int]:
        return {c: n for c, n in zip(_CLASES, self._cuentas[i] or ()) if n}

    def _nombre(self, i: int) -> str:
        return self._cabecera[i]

    def _se_convierte(self, i: int) -> bool:
        c = self._cuenta(i)
        return c.get("D", 0) > 0 and all(clase in ("D", "E") for clase in c)

    def a_convertir(self) -> frozenset[int]:
        return frozenset(i for i in range(len(self._cuentas)) if self._se_convierte(i))

    def declaraciones(self) -> list[dict[str, Any]]:
        salida: list[dict[str, Any]] = []
        for i in range(len(self._cuentas)):
            c = self._cuenta(i)
            if not c:
                continue
            ej = self._ejemplos[i]
            nombre = self._nombre(i)
            if self._se_convierte(i):
                de = ej[_D][0]
                salida.append(_declarar(
                    "decimal_coma_convertido", nombre, c["D"],
                    {"ejemplo": {"de": de, "a": convertir_celda(de)}, "enteros_sin_cambio": c.get("E", 0)}))
                continue
            parecen_decimales = c.get("D", 0) + c.get("PCT", 0)
            ambigua_sola = (c.get("M", 0) > 0 and c.get("D", 0) == 0 and c.get("PCT", 0) == 0
                            and all(clase in ("E", "M") for clase in c))
            if parecen_decimales == 0 and not ambigua_sola:
                continue
            # El motivo: lo primero que impide convertir, en este orden (el más fácil de arreglar a mano primero).
            if c.get("X", 0):
                motivo, culpable = "no_numerica", _X
            elif c.get("P", 0):
                motivo, culpable = "mezcla_con_punto", _P
            elif c.get("M", 0):
                motivo, culpable = "ambigua", _M
            else:
                motivo, culpable = "porcentaje", _PCT
            texto_ej, fila_ej = ej[culpable]
            cuantas = {"ambigua": c.get("M", 0), "porcentaje": c.get("PCT", 0)}.get(motivo, parecen_decimales)
            detalle: dict[str, Any] = {
                "motivo": motivo,
                "ejemplo": {"valor": texto_ej, "fila": fila_ej},
            }
            if _D in ej:
                detalle["ejemplo_decimal"] = ej[_D][0]
            salida.append(_declarar("decimal_coma_no_convertido", nombre, cuantas, detalle))
        return salida


def _declarar(tipo: str, columna: str, cuantas: int, detalle: dict[str, Any]) -> dict[str, Any]:
    return {"tipo": tipo, "columna": columna, "cuantas": cuantas, "detalle": detalle,
            "texto": {"es": _texto(tipo, columna, cuantas, detalle, "es"),
                      "en": _texto(tipo, columna, cuantas, detalle, "en")}}


def _texto(tipo: str, col: str, n: int, d: dict[str, Any], idioma: str) -> str:
    es = idioma == "es"
    if tipo == "decimal_coma_convertido":
        ej = d["ejemplo"]
        if es:
            return _p(n, f"En «{col}», 1 celda con coma decimal pasó a punto decimal («{ej['de']}» → «{ej['a']}»).",
                      f"En «{col}», " + "{n} celdas con coma decimal pasaron a punto decimal (p. ej. «" + ej["de"] + "» → «" + ej["a"] + "»).")
        return _p(n, f"In «{col}», 1 cell with a decimal comma was changed to a decimal point («{ej['de']}» → «{ej['a']}»).",
                  f"In «{col}», " + "{n} cells with a decimal comma were changed to a decimal point (e.g. «" + ej["de"] + "» → «" + ej["a"] + "»).")
    motivo = d["motivo"]
    v, fila = d["ejemplo"]["valor"], d["ejemplo"]["fila"]
    if motivo == "no_numerica":
        if es:
            return (f"«{col}» tiene celdas con forma de decimal con coma, pero también «{v}» (fila {fila} de datos), "
                    "que no es un número: la columna se dejó como estaba.")
        return (f"«{col}» has cells shaped like a decimal comma, but also «{v}» (data row {fila}), "
                "which is not a number: the column was left as it was.")
    if motivo == "mezcla_con_punto":
        if es:
            return (f"«{col}» mezcla decimales con coma y con punto («{v}», fila {fila} de datos): "
                    "no se sabe cuál manda y la columna se dejó como estaba.")
        return (f"«{col}» mixes decimal commas and decimal points («{v}», data row {fila}): "
                "there is no telling which one is meant and the column was left as it was.")
    if motivo == "ambigua":
        if es:
            return _p(n, f"«{col}» tiene 1 celda como «{v}» (fila {fila} de datos) que puede ser un decimal o un número con miles: no se tocó.",
                      f"«{col}» tiene " + "{n}" + f" celdas como «{v}» (la primera, fila {fila} de datos) que pueden ser decimales o números con miles: no se tocó la columna.")
        return _p(n, f"«{col}» has 1 cell like «{v}» (data row {fila}) that could be a decimal or a number with thousands: it was not touched.",
                  f"«{col}» has " + "{n}" + f" cells like «{v}» (the first one, data row {fila}) that could be decimals or numbers with thousands: the column was not touched.")
    # porcentaje
    if es:
        return _p(n, f"«{col}» tiene 1 porcentaje con coma decimal («{v}»): el símbolo % lo hace texto y no se convirtió.",
                  f"«{col}» tiene " + "{n}" + f" porcentajes con coma decimal (p. ej. «{v}»): el símbolo % los hace texto y no se convirtieron.")
    return _p(n, f"«{col}» has 1 percentage with a decimal comma («{v}»): the % symbol makes it text and it was not converted.",
              f"«{col}» has " + "{n}" + f" percentages with a decimal comma (e.g. «{v}»): the % symbol makes them text and they were not converted.")
