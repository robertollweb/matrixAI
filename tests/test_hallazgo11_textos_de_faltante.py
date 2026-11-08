"""Hallazgo 11 (2026-11-08): una columna numérica con «NA» (o «?», «#N/A»…) escrito pasaba a CATEGÓRICA en la
preparación —cientos de categorías, y el valor numérico se perdía— mientras el análisis la leía numérica.
Ahora hay UNA lista (`textos_de_faltante`), y esos textos son faltantes DECLARADOS solo en una columna que, sin ellos,
es numérica; una columna de texto con «NA» (Norteamérica) no se toca."""
from __future__ import annotations

import csv
import io
import random

import pytest

from matrixai.training import dataset_analysis
from matrixai.training.dataset_analysis import analyze_dataset_csv
from matrixai.training.dataset_project import generate_project_from_dataset
from matrixai.training.particion_por_diseno_tiempo import Rechazo, convertir_tiempos
from matrixai.training.preparacion import (
    CATEGORIA_FALTANTE, PoliticaDePreparacion, ajustar_preparacion, limite_de_textos_leidos_como_faltante,
    tipar_columnas_numericas, transformar_fila)
from matrixai.training.lo_visto import comprobar_lo_visto
from matrixai.training.textos_de_faltante import (
    TEXTOS_DE_FALTANTE, TEXTOS_DE_FALTANTE_GENERALES, TEXTOS_SOLO_EN_NUMERICA, es_texto_de_faltante)

TEXTOS = ["NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "-", "?", "", "   ", "#N/A", "."]


def _filas(texto, fraccion, n=200, seed=3):
    r = random.Random(seed)
    return [{"x": texto if r.random() < fraccion else f"{r.uniform(0, 100):.2f}", "z": f"{r.random():.3f}",
             "y": r.choice(["a", "b"])} for _ in range(n)]


def _csv(filas):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
    return out.getvalue()


@pytest.mark.parametrize("texto", TEXTOS)
@pytest.mark.parametrize("fraccion", [0.01, 0.30, 0.90])
def test_una_columna_numerica_con_el_texto_de_faltante_sigue_numerica_en_la_preparacion_y_en_el_analisis(texto, fraccion):
    filas = _filas(texto, fraccion)
    # Reparación (I4): también «#N/A» y «.»: antes el análisis decía `identifier` (al 1 %) o `categorical` de una columna que
    # el estudio usaba como número.
    analisis = analyze_dataset_csv(_csv(filas))["columns"]["x"]
    assert analisis["type"] == "number", (texto, analisis)
    tipar_columnas_numericas(filas, ["x", "z"])
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["x", "z"], admite_categoricas=True, admite_faltantes=True)
    p = politica.columna("x")
    assert p.tipo == "numerica" and p.categorias_conocidas == (), p          # antes: categorica, con ~395 categorías
    assert p.mediana is not None and p.minimo is not None and p.maximo is not None
    faltan = sum(1 for f in filas if f["x"] is None or f["x"] != f["x"] or f["x"] == "")
    assert p.proporcion_faltante == pytest.approx(faltan / len(filas))


def test_tipar_pone_none_en_los_textos_de_faltante_y_los_declara():
    filas = [{"x": "1.5"}, {"x": "NA"}, {"x": "n/a"}, {"x": "#N/A"}, {"x": " ? "}, {"x": ""}, {"x": "7"}]
    leidos: dict = {}
    assert tipar_columnas_numericas(filas, ["x"], leidos_como_faltante=leidos) == ("x",)
    assert [f["x"] for f in filas] == [1.5, None, None, None, None, "", 7.0]
    assert leidos == {"x": {"textos": ["#N/A", "?", "NA", "n/a"], "celdas": 4,                  # lo vacío no se declara: ya era faltante
                            "por_texto": {"#N/A": 1, "?": 1, "NA": 1, "n/a": 1}}}


def test_la_declaracion_no_repite_un_texto_y_dice_cuantas_celdas_tiene_cada_uno():
    """Auditoría, M1: «NA», « NA» y «NA » se declaraban `['NA', 'NA', 'NA']`. Ahora UN texto con su cuenta."""
    filas = [{"x": v} for v in ["NA", " NA", "NA ", "1", "2", "?", "3"]]
    leidos: dict = {}
    tipar_columnas_numericas(filas, ["x"], leidos_como_faltante=leidos)
    assert leidos["x"] == {"textos": ["?", "NA"], "celdas": 4, "por_texto": {"?": 1, "NA": 3}}
    limite = limite_de_textos_leidos_como_faltante("x", leidos["x"]["por_texto"])
    assert limite.clave == "textos_leidos_como_faltante" and limite.campo == "x"
    assert "«?» (1), «NA» (3)" in limite.motivo["es"] and "«?» (1), «NA» (3)" in limite.motivo["en"]
    assert "4 de sus celdas" in limite.motivo["es"] and "4 of its cells" in limite.motivo["en"]
    # con un solo texto, sin la cuenta repetida
    solo = limite_de_textos_leidos_como_faltante("x", {"NA": 3})
    assert "3 de sus celdas dicen «NA»:" in solo.motivo["es"] and "(3)" not in solo.motivo["es"]


def test_una_columna_de_texto_con_NA_como_categoria_legitima_no_se_toca():
    regiones = ["NA", "EU", "AS", "NA", "EU", "SA"] * 10
    filas = [{"region": r, "y": "a" if i % 2 else "b"} for i, r in enumerate(regiones)]
    antes = [dict(f) for f in filas]
    leidos: dict = {}
    assert tipar_columnas_numericas(filas, ["region"], leidos_como_faltante=leidos) == ()
    assert filas == antes and leidos == {}
    p = ajustar_preparacion(filas, objetivo="y", columnas=["region"], admite_categoricas=True, admite_faltantes=True).columna("region")
    assert p.tipo == "categorica" and "NA" in p.categorias_conocidas and p.proporcion_faltante == 0.0
    # y al transformar: «NA» es una categoría conocida, no un faltante
    pol = ajustar_preparacion(filas, objetivo="y", columnas=["region"], admite_categoricas=True, admite_faltantes=True)
    assert transformar_fila({"region": "NA"}, pol)["region"] == "NA"


def test_una_columna_con_otro_texto_dentro_sigue_categorica():
    filas = [{"x": v, "y": "a"} for v in ["1", "2", "NA", "abc", "4"] * 4]
    assert tipar_columnas_numericas(filas, ["x"]) == ()
    assert [f["x"] for f in filas][:5] == ["1", "2", "NA", "abc", "4"]


def test_una_columna_solo_de_textos_de_faltante_no_se_inventa_numerica():
    filas = [{"x": "NA", "y": "a"} for _ in range(10)]
    assert tipar_columnas_numericas(filas, ["x"]) == ()


def test_ajustar_con_valores_ya_tipados_y_un_NA_suelto_declara_el_limite():
    filas = [{"x": float(i), "y": "a"} for i in range(50)] + [{"x": "NA", "y": "b"}, {"x": "?", "y": "b"}]
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["x"], admite_categoricas=True, admite_faltantes=False)
    p = politica.columna("x")
    assert p.tipo == "numerica" and p.proporcion_faltante == pytest.approx(2 / 52)
    (limite,) = [l for l in politica.limites if l.clave == "textos_leidos_como_faltante"]
    assert limite.campo == "x" and limite.medida == {"textos": ["?", "NA"], "celdas": 2, "por_texto": {"?": 1, "NA": 1}}
    assert "«?» (1), «NA» (1)" in limite.motivo["es"] and "«?» (1), «NA» (1)" in limite.motivo["en"]
    assert "faltantes" in limite.motivo["es"] and "missing" in limite.motivo["en"]


def test_transformar_fila_lee_el_texto_como_faltante_igual_que_el_vacio():
    filas = _filas("NA", 0.3)
    tipar_columnas_numericas(filas, ["x"])
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["x"], admite_categoricas=True, admite_faltantes=False)
    politica = PoliticaDePreparacion.desde_json(politica.a_json())       # ida y vuelta por el JSON, como en el paquete
    vacio = transformar_fila({"x": ""}, politica)
    for texto in TEXTOS:
        if texto.lower() == "nan":
            continue          # «nan» ya se leía como NaN con su indicador a 0: NO cambia (ver abajo)
        assert transformar_fila({"x": texto}, politica) == vacio, texto
    nan = transformar_fila({"x": "nan"}, politica)
    assert nan["x"] != nan["x"] and nan["x__faltante"] == 0.0                # como antes del hallazgo 11
    assert vacio["x__faltante"] == 1.0 and vacio["x"] == politica.columna("x").mediana
    assert transformar_fila({"x": "3.5"}, politica)["x__faltante"] == 0.0


def test_la_receta_de_un_estudio_ya_entrenado_manda():
    """Una política ajustada ANTES (x categórica, con «NA» entre sus categorías) sigue leyendo «NA» como categoría."""
    vieja = PoliticaDePreparacion.desde_json({
        "columnas": [{"columna": "x", "tipo": "categorica", "admite_nativo": True, "proporcion_faltante": 0.0,
                      "mediana": None, "minimo": None, "maximo": None,
                      "categorias_conocidas": ["1.5", "NA"], "categoria_de_referencia": "NA"}],
        "filas_excluidas_sin_objetivo": 0, "filas_de_train_efectivas": 10, "limites": []})
    assert transformar_fila({"x": "NA"}, vieja)["x"] == "NA"
    assert transformar_fila({"x": ""}, vieja)["x"] == CATEGORIA_FALTANTE


def test_el_uso_de_la_fila_y_lo_visto_no_marcan_un_texto_de_faltante_como_valor_raro():
    filas = _filas("NA", 0.3)
    tipar_columnas_numericas(filas, ["x"])
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["x"], admite_categoricas=True, admite_faltantes=False)
    assert comprobar_lo_visto({"x": "NA"}, politica).a_json()["dentro_de_lo_visto"] is True       # faltaba al entrenar


def test_hay_una_sola_lista_y_el_analisis_general_sigue_como_antes():
    assert dataset_analysis._NULL_TOKENS == {"", "na", "n/a", "null", "nan", "none", "-", "?"}       # la de ANTES del hallazgo 11
    assert dataset_analysis._NULL_TOKENS == set(TEXTOS_DE_FALTANTE_GENERALES)
    assert TEXTOS_DE_FALTANTE == TEXTOS_DE_FALTANTE_GENERALES | TEXTOS_SOLO_EN_NUMERICA == TEXTOS_DE_FALTANTE_GENERALES | {"#n/a", "."}
    assert all(es_texto_de_faltante(t) for t in TEXTOS if t.lower() != "nan") and not es_texto_de_faltante("nan")
    assert not es_texto_de_faltante(5) and not es_texto_de_faltante(None) and not es_texto_de_faltante("Norteamérica")


def test_una_categorica_con_punto_y_un_objetivo_con_NA_de_excel_se_comportan_como_antes():
    """`.` y `#N/A` son faltante SOLO en una columna numérica: en texto y en el objetivo siguen siendo un valor."""
    filas = [{"cat": c, "x": f"{i}.5", "y": y} for i, (c, y) in enumerate(
        zip(["a", ".", "b", ".", "#N/A"] * 8, ["si", "no", "#N/A", "si", "no"] * 8))]
    a = analyze_dataset_csv(_csv(filas))["columns"]
    assert a["cat"]["null_count"] == 0 and a["cat"]["type"] == "categorical" and a["cat"]["cardinality"] == 4
    assert a["y"]["null_count"] == 0 and a["y"]["cardinality"] == 3             # «#N/A» es una clase más, como antes
    # y la preparación: una categórica con «.» y «#N/A» las conserva como categorías
    p = ajustar_preparacion(filas, objetivo="y", columnas=["cat"], admite_categoricas=True, admite_faltantes=True).columna("cat")
    assert p.tipo == "categorica" and {".", "#N/A"} <= set(p.categorias_conocidas) and p.proporcion_faltante == 0.0
    assert tipar_columnas_numericas([dict(f) for f in filas], ["cat"]) == ()
    # el objetivo: el criterio de «sin dato» del análisis no lo cuenta
    assert dataset_analysis._is_null("#N/A") is False and dataset_analysis._is_null(".") is False



# ---------------------------------------------------------------------------------------------------------------------
# Reparación tras la auditoría (I4): la misma enfermedad en los vecinos
# ---------------------------------------------------------------------------------------------------------------------

def _fechas(texto, n=150):
    r = random.Random(3)
    return [{"f": texto if i % 5 == 0 else f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}", "y": r.choice("ab")} for i in range(n)]


@pytest.mark.parametrize("texto", ["NA", "#N/A", " n/a ", ".", "?"])
def test_una_fecha_con_un_texto_de_faltante_sigue_siendo_fecha_en_el_analisis_y_en_la_preparacion(texto):
    """I4 (a): el análisis decía `date` y la preparación la hacía CATEGÓRICA (82 categorías). Ahora fecha en los dos, con sus
    variables faltantes en esas celdas y la declaración dicha."""
    filas = _fechas(texto)
    a = analyze_dataset_csv(_csv(filas))["columns"]["f"]
    assert a["type"] == "date" and a["null_count"] == 30, a
    tipar_columnas_numericas(filas, ["f"])
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["f"], admite_categoricas=True, admite_faltantes=True,
                                   con_fechas=True)
    p = politica.columna("f")
    assert p.tipo == "fecha" and p.formato_de_fecha == "%Y-%m-%d" and p.categorias_conocidas == (), p
    assert p.proporcion_faltante == pytest.approx(30 / 150)
    (limite,) = [l for l in politica.limites if l.clave == "textos_leidos_como_faltante"]
    assert limite.campo == "f" and limite.medida["celdas"] == 30 and limite.medida["textos"] == [texto.strip()]
    fila = transformar_fila({"f": texto}, politica)
    assert fila == transformar_fila({"f": ""}, politica) and fila["f__anio__faltante"] == 1.0


def test_una_columna_de_texto_que_no_es_fecha_con_NA_no_se_vuelve_fecha():
    filas = [{"f": v, "y": "a"} for v in ["NA", "Madrid", "2024-01-01", "NA"] * 10]
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["f"], admite_categoricas=True, admite_faltantes=True,
                                   con_fechas=True)
    assert politica.columna("f").tipo == "categorica" and "NA" in politica.columna("f").categorias_conocidas


def test_la_particion_temporal_sigue_rechazando_un_NA_en_la_columna_del_tiempo():
    """La detección de fechas SOLO lee los textos de faltante si se lo piden (la preparación): la partición temporal convierte
    cada valor con el formato y un «NA» ahí sigue siendo un rechazo dicho, no un `ValueError` de `strptime`."""
    valores = {i: ("NA" if i % 7 == 0 else f"2024-01-{1 + i % 28:02d}") for i in range(40)}
    with pytest.raises(Rechazo):
        convertir_tiempos(valores, "f")


@pytest.mark.parametrize("texto", ["#N/A", "."])
@pytest.mark.parametrize("fraccion", [0.01, 0.30])
def test_el_analisis_lee_el_faltante_de_excel_y_de_sas_en_una_numerica_y_no_la_llama_identificador(texto, fraccion):
    """I4 (b): con «#N/A» o «.» el análisis decía `identifier` (al 1 %) o `categorical` y la preparación numérica."""
    filas = [{"x": texto if i % int(1 / fraccion) == 0 else f"{random.Random(i).uniform(0, 100):.2f}", "y": "ab"[i % 2]}
             for i in range(300)]
    a = analyze_dataset_csv(_csv(filas))["columns"]["x"]
    celdas = sum(1 for f in filas if f["x"] == texto)
    assert a["type"] == "number" and a["null_count"] == celdas, a
    assert a["textos_leidos_como_faltante"] == {texto: celdas}
    tipar_columnas_numericas(filas, ["x"])
    p = ajustar_preparacion(filas, objetivo="y", columnas=["x"], admite_categoricas=True, admite_faltantes=True).columna("x")
    assert p.tipo == "numerica" and p.proporcion_faltante == pytest.approx(celdas / 300)       # los DOS dicen lo mismo


def test_el_analisis_con_la_ausencia_declarada_no_aplica_la_regla():
    filas = [{"x": "#N/A" if i % 10 == 0 else str(i), "y": "ab"[i % 2]} for i in range(100)]
    a = analyze_dataset_csv(_csv(filas), tokens_de_ausencia={""})["columns"]["x"]
    assert a["type"] != "number" and "textos_leidos_como_faltante" not in a


def test_un_codigo_con_cero_delante_y_un_punto_no_se_vuelve_numerico():
    filas = [{"cp": v, "y": "ab"[i % 2]} for i, v in enumerate(["08001", "39001", ".", "28080"] * 10)]
    a = analyze_dataset_csv(_csv(filas))["columns"]["cp"]
    assert a["type"] != "number" and a["null_count"] == 0 and "textos_leidos_como_faltante" not in a


@pytest.mark.parametrize("texto", ["#N/A", "."])
def test_generate_project_from_dataset_imputa_el_faltante_de_excel_en_una_numerica(texto):
    """Antes de la reparación, con el análisis ya alineado, fallaba con «must be numeric, got '#N/A'»: el camino clásico
    leía la celda con `_is_null`, sin la regla. Ahora la imputa con la mediana y lo DECLARA (`imputed_numeric`)."""
    r = random.Random(1)
    filas = [{"x": texto if i % 10 == 0 else f"{r.uniform(0, 100):.2f}", "z": f"{r.uniform(0, 10):.1f}",
              "y": r.choice(["a", "b"])} for i in range(300)]
    g = generate_project_from_dataset(_csv(filas), "y")
    assert g["accepted"] is True
    assert g["provenance"]["missing_values"]["imputed_numeric"]["x"]["cells"] == 30
    preparadas = list(csv.DictReader(io.StringIO(g["csv_text"])))
    assert not any(v == texto for fila in preparadas for v in fila.values())
    assert sum(1 for fila in preparadas if fila["x_faltante"] == "1") == 30
