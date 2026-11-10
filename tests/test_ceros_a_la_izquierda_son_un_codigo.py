"""Decisión A de Roberto (09-11, tarjeta `ceros-a-la-izquierda-en-el-estudio`): «08001» no es 8001.

El análisis ya decía «categórica» para una columna con códigos postales y el estudio la convertía en número antes de entrenar
(`tipar_columnas_numericas`: «08001» → 8001.0, «00012» → 12.0). Una sola regla (`codigos_con_ceros`) la usan el análisis, el
proyecto y el tipado del estudio. Aquí: la regla y sus bordes, el tipado con y sin el parámetro, y la coherencia con el análisis."""
from __future__ import annotations

import copy
import csv
import io

import pytest

from matrixai.training.codigos_con_ceros import codigos_con_ceros_de, es_un_codigo_con_ceros
from matrixai.training.dataset_analysis import _has_significant_leading_zero, analyze_dataset_csv
from matrixai.training.preparacion import (ajustar_preparacion, limite_de_codigo_con_ceros, tipar_columnas_numericas,
                                           transformar_fila)

SON_CODIGO = ["08001", "00012", "007", "0012", "00", "012", " 007 "]
NO_SON_CODIGO = ["0", "0.5", "0,5", "-0.3", "0e3", "-08", "+07", "8001", "10", "70", "0.05", "00.5", "0x1F", "", "abc", "0a",
                 "0²", None, 7, 0, 0.5, True]


@pytest.mark.parametrize("valor", SON_CODIGO)
def test_son_un_codigo(valor):
    assert es_un_codigo_con_ceros(valor)


@pytest.mark.parametrize("valor", NO_SON_CODIGO)
def test_no_son_un_codigo(valor):
    assert not es_un_codigo_con_ceros(valor)


def test_el_analisis_usa_la_misma_regla():
    for v in SON_CODIGO + [x for x in NO_SON_CODIGO if isinstance(x, str)]:
        assert _has_significant_leading_zero(v.strip()) == es_un_codigo_con_ceros(v), repr(v)


def test_codigos_de_una_lista_recortados_sin_repetir_en_orden():
    assert codigos_con_ceros_de(["08001", " 08001", "5", "007", None, 3]) == ["08001", "007"]


def _filas(col, valores):
    return [{col: v, "n": str(i)} for i, v in enumerate(valores)]


def test_un_codigo_postal_se_queda_como_texto_y_se_declara():
    filas = _filas("cp", ["08001", "28013", "00012", "08001", "41001", "46001"] * 5)
    antes = copy.deepcopy(filas)
    codigos: dict = {}
    tocadas = tipar_columnas_numericas(filas, ["cp", "n"], conservar_ceros_a_la_izquierda=True, codigos_con_ceros=codigos)
    assert tocadas == ("n",)                                              # «cp» NO se tocó
    assert [f["cp"] for f in filas] == [f["cp"] for f in antes]            # con sus ceros
    assert all(isinstance(f["cp"], str) for f in filas)
    assert codigos == {"cp": {"celdas": 15, "ejemplos": ["08001", "00012"]}}
    assert isinstance(filas[0]["n"], float)


def test_con_ceros_solo_en_algun_valor_la_columna_entera_es_un_codigo():
    """Una columna donde el cero se perdió en unas filas y no en otras: la regla del análisis, la columna entera."""
    filas = _filas("c", ["8001", "28013", "007", "41001"])
    codigos: dict = {}
    assert tipar_columnas_numericas(filas, ["c"], conservar_ceros_a_la_izquierda=True, codigos_con_ceros=codigos) == ()
    assert [f["c"] for f in filas] == ["8001", "28013", "007", "41001"]
    assert codigos["c"]["celdas"] == 1 and codigos["c"]["ejemplos"] == ["007"]


@pytest.mark.parametrize("valores", [
    ["0", "1", "2", "3"],                      # el cero solo
    ["0", "0.5", "1.25", "3"],
    ["-0.3", "0.5", "-8", "12"],
    ["0.5", "0.05", "0.25", "10"],
    ["0e3", "1e3", "2"],
    ["-08", "-07", "12"],                       # con signo: numérica, como en el análisis
    ["10", "70", "100", "205"],                 # ceros al final, no a la izquierda
])
def test_las_numericas_de_verdad_siguen_siendo_numericas(valores):
    filas = _filas("x", valores)
    codigos: dict = {}
    assert tipar_columnas_numericas(filas, ["x"], conservar_ceros_a_la_izquierda=True, codigos_con_ceros=codigos) == ("x",)
    assert codigos == {}
    assert all(isinstance(f["x"], float) for f in filas)


def test_el_decimal_espanol_ya_convertido_sigue_numerico():
    """«0,5» llega aquí ya como «0.5» (`decimal_es` lo convierte al leer el CSV con `;`): sigue numérica."""
    filas = _filas("x", ["0.5", "1.25", "0.75", "12.5"])
    assert tipar_columnas_numericas(filas, ["x"], conservar_ceros_a_la_izquierda=True) == ("x",)


def test_nif_sin_letra_y_producto_de_cuatro_cifras_son_codigos():
    nif = _filas("nif", ["00123456", "12345678", "05555555", "00000001"] * 3)
    prod = _filas("prod", ["0012", "0013", "0100", "0012", "0200"] * 3)
    for filas, col in ((nif, "nif"), (prod, "prod")):
        assert tipar_columnas_numericas(filas, [col], conservar_ceros_a_la_izquierda=True) == ()
        assert all(isinstance(f[col], str) for f in filas)


def test_un_na_entre_codigos_no_cambia_la_decision_y_el_na_no_se_toca():
    filas = _filas("cp", ["08001", "NA", "28013", "", None, "00012"])
    codigos: dict = {}
    leidos: dict = {}
    assert tipar_columnas_numericas(filas, ["cp"], leidos_como_faltante=leidos, conservar_ceros_a_la_izquierda=True,
                                    codigos_con_ceros=codigos) == ()
    assert leidos == {}                              # no se declaró «NA leído como faltante»: la columna es un código
    assert codigos["cp"]["celdas"] == 2 and codigos["cp"]["ejemplos"] == ["08001", "00012"]
    assert [f["cp"] for f in filas] == ["08001", "NA", "28013", "", None, "00012"]


def test_una_columna_de_texto_con_un_cero_delante_no_se_declara_codigo():
    """«A12» y «007»: era texto de todos modos. Declarar «se lee como código» sería decir algo que no pasó."""
    filas = _filas("t", ["A12", "007", "B3", "C9"])
    codigos: dict = {}
    assert tipar_columnas_numericas(filas, ["t"], conservar_ceros_a_la_izquierda=True, codigos_con_ceros=codigos) == ()
    assert codigos == {}


def test_por_omision_no_cambia_nada_la_fase_0_y_las_fechas_miden_con_lo_de_siempre():
    """Sin el parámetro, «08001» sigue siendo 8001.0: es lo que tipa `benchmarks/fase0` (protocolo sellado)."""
    filas = _filas("cp", ["08001", "28013", "00012"])
    assert tipar_columnas_numericas(filas, ["cp"]) == ("cp",)
    assert [f["cp"] for f in filas] == [8001.0, 28013.0, 12.0]


def test_el_codigo_llega_a_la_preparacion_como_categoria_y_se_transforma_con_sus_ceros():
    filas = [{"cp": ["08001", "28013", "00012", "41001"][i % 4], "x": str(i), "y": "a" if i % 2 else "b"} for i in range(40)]
    tipar_columnas_numericas(filas, ["cp", "x"], conservar_ceros_a_la_izquierda=True)
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["cp", "x"], admite_categoricas=True, admite_faltantes=True)
    assert politica.columna("cp").tipo != "numerica"
    assert politica.columna("x").tipo == "numerica"
    fila = transformar_fila({"cp": "00012", "x": "3"}, politica)
    assert fila["cp"] == "00012"                     # la categoría con sus ceros (el motor la codifica), no un 12 numérico


def test_coherencia_con_el_analisis_el_cp_es_categorico_y_el_estudio_lo_trata_como_tal():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["cp", "n", "y"])
    for i in range(60):
        w.writerow([["08001", "28013", "00012", "41001"][i % 4], i * 3 + 1000, "a" if i % 2 else "b"])
    analisis = analyze_dataset_csv(buf.getvalue())
    tipo = {c["name"]: c["type"] for c in analisis["columns"]} if isinstance(analisis.get("columns"), list) else \
        {k: v["type"] for k, v in analisis["columns"].items()}
    assert tipo["cp"] == "categorical"
    filas = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert tipar_columnas_numericas(filas, ["cp", "n"], conservar_ceros_a_la_izquierda=True) == ("n",)


def test_la_frase_del_nucleo_en_los_dos_idiomas():
    d = limite_de_codigo_con_ceros("cp", ["08001", "00012"], 15).a_json()
    assert d["clave"] == "codigo_con_ceros_a_la_izquierda" and d["campo"] == "cp"
    assert d["medida"] == {"celdas": 15, "ejemplos": ["08001", "00012"]}
    assert "cp lleva ceros a la izquierda en 15 celdas («08001», «00012»)" in d["motivo"]["es"]
    assert "cp has leading zeros in 15 cells («08001», «00012»)" in d["motivo"]["en"]
