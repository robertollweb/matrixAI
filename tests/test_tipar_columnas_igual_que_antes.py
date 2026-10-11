"""`tipar_columnas_numericas` más barata (10-10) decide EXACTAMENTE lo mismo que la versión anterior.

Medido en el arranque de APSFailure (76.000 × 171): 19,5 → 12,9 s. La versión nueva lee cada columna una vez y quita los
vacíos sobre el conjunto de valores distintos; esta prueba la compara con una COPIA LITERAL de la anterior (abajo) en las
formas de datos que trae la gente: números como texto (con espacios, signos, notación científica, «nan», «inf»), vacíos y
`None`, columnas ya tipadas (int, float, bool), mixtas texto/número, todo vacío, una sola fila, columnas ausentes."""
from __future__ import annotations

import copy
import math
import random

import pytest

from matrixai.training.preparacion import tipar_columnas_numericas


def _anterior(filas, columnas):
    """La versión de antes del 10-10, COPIA LITERAL (núcleo `d92ca07`)."""
    tocadas: list[str] = []
    for columna in columnas:
        no_vacios = [f.get(columna) for f in filas]
        no_vacios = [v for v in no_vacios if v not in (None, "")]
        if not no_vacios:
            continue
        try:
            tipados = {v: float(v) for v in set(no_vacios)}
        except (TypeError, ValueError):
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in no_vacios):
            continue
        for fila in filas:
            valor = fila.get(columna)
            if valor not in (None, ""):
                fila[columna] = tipados[valor]
        tocadas.append(columna)
    return tuple(tocadas)


_TEXTOS = ["0", "1", "1.5", "-2", " 3 ", "1e3", "nan", "NaN", "inf", "-inf", "0.0", "007", "+4"]
_RAROS = ["a", "Sí", "no", "1,5", "x1", "?"]


def _celda(rng, forma):
    if rng.random() < 0.15:
        return rng.choice([None, ""])
    if forma == "texto_numerico":
        return rng.choice(_TEXTOS)
    if forma == "mixta":
        return rng.choice(_TEXTOS + _RAROS)
    if forma == "categorica":
        return rng.choice(_RAROS)
    if forma == "ya_float":
        return rng.choice([0.0, 1.5, -2.0, float("nan")])
    if forma == "ya_int":
        return rng.choice([0, 1, 7])
    if forma == "bool":
        return rng.choice([True, False])
    if forma == "int_y_texto":
        return rng.choice([1, 2, "3"])
    raise AssertionError(forma)


_FORMAS = ["texto_numerico", "mixta", "categorica", "ya_float", "ya_int", "bool", "int_y_texto", "vacia"]


def _filas(semilla, n):
    rng = random.Random(semilla)
    filas = []
    for _ in range(n):
        fila = {}
        for forma in _FORMAS:
            fila[forma] = rng.choice([None, ""]) if forma == "vacia" else _celda(rng, forma)
        if rng.random() < 0.1:
            del fila["texto_numerico"]          # columna ausente en algunas filas
        filas.append(fila)
    return filas


def _iguales(a, b):
    """Igualdad de filas con NaN (NaN != NaN): por tipo y valor, NaN con NaN."""
    if len(a) != len(b):
        return False
    for fa, fb in zip(a, b):
        if fa.keys() != fb.keys():
            return False
        for k in fa:
            x, y = fa[k], fb[k]
            if type(x) is not type(y):
                return False
            if isinstance(x, float) and math.isnan(x):
                if not math.isnan(y):
                    return False
            elif x != y:
                return False
    return True


@pytest.mark.parametrize("semilla,n", [(0, 1), (1, 2), (2, 50), (3, 500), (4, 3000)])
def test_decide_y_escribe_exactamente_lo_mismo_que_la_version_anterior(semilla, n):
    filas = _filas(semilla, n)
    columnas = _FORMAS + ["no_existe"]
    viejas, nuevas = copy.deepcopy(filas), copy.deepcopy(filas)
    assert tipar_columnas_numericas(nuevas, columnas) == _anterior(viejas, columnas)
    assert _iguales(nuevas, viejas)


def test_el_texto_numerico_se_convierte_y_lo_ya_tipado_no_se_declara_tocado():
    filas = [{"x": "1", "y": 2.0, "z": "a"}, {"x": "", "y": 3.0, "z": "1"}, {"x": " 2.5 ", "y": None, "z": ""}]
    assert tipar_columnas_numericas(filas, ["x", "y", "z"]) == ("x",)
    assert [f["x"] for f in filas] == [1.0, "", 2.5]
    assert [f["z"] for f in filas] == ["a", "1", ""]
