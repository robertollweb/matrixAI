"""Hallazgo 11, M3 (decisión A de Roberto, 09-11, tarjeta `h11-objetivo-con-na-de-excel`): en un OBJETIVO numérico —una
regresión— cualquier texto de faltante de LA lista (`textos_de_faltante`: «NA», «#N/A», «-», «.», « na »…) es una fila SIN
objetivo: se quita y se declara, como ya pasaba con «NA». Antes «#N/A» de Excel paraba el estudio («no es un número») y
`generate_project_from_dataset` fallaba en la validación del CSV preparado («must be numeric, got '#N/A'»).

El otro lado sigue igual: en una CLASIFICACIÓN o en un texto, «#N/A» y «.» son una clase/un valor (y «NA», como siempre,
lo que diga el criterio de la confirmación)."""
from __future__ import annotations

import csv
import io
import random

import pytest

from matrixai.training.dataset_project import generate_project_from_dataset, prepare_dataset_from_provenance
from matrixai.training.preparacion import limite_de_textos_del_objetivo_sin_dato
from matrixai.training.preparacion_textos import MOTIVOS, huecos_de
from matrixai.training.textos_de_faltante import TEXTOS_DE_FALTANTE, sin_dato_en_un_objetivo_numerico

#: Las formas que trae la gente en un objetivo numérico: Excel, SAS/Stata, mayúsculas, espacios.
FORMAS_SIN_DATO = ["NA", "#N/A", "-", ".", " na ", "N/A", "#n/a", "NULL", "?", "None", "nan", "NaN", "", "   "]


def _csv_regresion(faltas, n=160, seed=3):
    r = random.Random(seed)
    o = io.StringIO()
    w = csv.writer(o)
    w.writerow(["x", "z", "y"])
    for i in range(n):
        x = r.uniform(0, 10)
        y = faltas[(i // 8) % len(faltas)] if i % 8 == 0 else f"{2 * x + r.random():.2f}"
        w.writerow([f"{x:.2f}", f"{r.random():.2f}", y])
    return o.getvalue()


@pytest.mark.parametrize("forma", FORMAS_SIN_DATO)
def test_cada_forma_de_sin_dato_es_una_fila_sin_objetivo_numerico(forma):
    assert sin_dato_en_un_objetivo_numerico(forma) is True


def test_la_regla_del_objetivo_es_LA_lista_sin_una_nueva():
    """Una sola lista: todo texto de `TEXTOS_DE_FALTANTE` (y sus formas en mayúsculas y con espacios) es «sin dato» en un
    objetivo numérico, y nada fuera de ella lo es. Si mañana la lista gana un texto, el objetivo lo gana solo."""
    for t in TEXTOS_DE_FALTANTE:
        assert sin_dato_en_un_objetivo_numerico(t) and sin_dato_en_un_objetivo_numerico(f"  {t.upper()} "), t
    for valor in ("12.5", "0", "-3", "1e3", "abc", "n/d", "Sí", "NO", "Norteamérica", "#VALUE!", 0, 7.5, False):
        assert sin_dato_en_un_objetivo_numerico(valor) is False, valor
    assert sin_dato_en_un_objetivo_numerico(None) and sin_dato_en_un_objetivo_numerico(float("nan"))


def test_la_declaracion_dice_que_textos_y_cuantas_en_los_dos_idiomas():
    valores = ["1.5", "#N/A", "NA", " na ", "-", "", None, "2", "#N/A", "."]
    lim = limite_de_textos_del_objetivo_sin_dato("y", valores)
    assert lim is not None and lim.clave == "textos_del_objetivo_sin_dato" and lim.campo == "y"
    # «NA» y « na » son UN texto (la forma normalizada); la vacía y el `None` no son un texto y no se nombran.
    assert lim.medida["por_texto"] == {"#N/A": 2, "-": 1, ".": 1, "NA": 2} and lim.medida["celdas"] == 6
    es, en = lim.motivo["es"], lim.motivo["en"]
    assert "y es el objetivo" in es and "6 de sus celdas" in es and "«#N/A» (2)" in es and "filas sin objetivo" in es
    assert "y is the target" in en and "6 of its cells" in en and "«#N/A» (2)" in en and "rows without a target" in en
    assert huecos_de(MOTIVOS["textos_del_objetivo_sin_dato"]["es"]) == huecos_de(MOTIVOS["textos_del_objetivo_sin_dato"]["en"])


def test_sin_textos_no_hay_declaracion():
    assert limite_de_textos_del_objetivo_sin_dato("y", ["1", "2", "", None]) is None


@pytest.mark.parametrize("faltas", [["#N/A", "NA", "-"], [" na ", ".", "N/A", "#n/a", "NULL", "?"]])
def test_generar_un_proyecto_de_regresion_quita_las_filas_sin_objetivo_y_las_cuenta(faltas):
    """Antes: `DatasetProjectError` («must be numeric, got '#N/A'»). Ahora las 20 filas (una de cada ocho) se quitan, como
    las de «NA», y lo dice `rows_dropped_null_target`. El CSV preparado no lleva ninguna."""
    res = generate_project_from_dataset(_csv_regresion(faltas), "y", unidad_de_observacion="una fila")
    assert res["provenance"]["task"] == "regression"
    assert res["provenance"]["rows_dropped_null_target"] == 20
    filas = list(csv.DictReader(io.StringIO(res["csv_text"])))
    assert len(filas) == 140
    objetivo = [k for k in filas[0] if k not in ("x", "z") and not k.startswith(("x_", "z_"))][-1]
    assert all(float(f[objetivo]) == float(f[objetivo]) for f in filas)


def test_repreparar_el_mismo_csv_quita_las_mismas_filas_y_da_el_mismo_csv():
    """La re-preparación (reimportar el dataset de un modelo) lee el objetivo con la MISMA regla que al generar."""
    raw = _csv_regresion(["#N/A", "NA", "-", "."])
    res = generate_project_from_dataset(raw, "y", unidad_de_observacion="una fila")
    re_prep = prepare_dataset_from_provenance(raw, res["provenance"])
    assert re_prep.csv_text == res["csv_text"]
    assert re_prep.provenance["rows_dropped_null_target"] == 20


def test_el_otro_lado_una_clasificacion_conserva_su_NA_de_excel_como_clase():
    """Dígitos 0/1 con «#N/A» de Excel en una clasificación, y un texto con tildes y mayúsculas («Sí», «NO», «#N/A»):
    «#N/A» sigue siendo una clase más (no se quita ninguna fila por él), como antes de M3."""
    r = random.Random(5)
    digitos = "x,y\n" + "".join(f"{x:.2f},{'#N/A' if i % 9 == 0 else ('1' if x > 5 else '0')}\n"
                                for i, x in enumerate(r.uniform(0, 10) for _ in range(160)))
    res = generate_project_from_dataset(digitos, "y", unidad_de_observacion="una fila")
    assert res["provenance"]["task"] == "classification" and res["provenance"]["rows_dropped_null_target"] == 0
    texto = "x,y\n" + "".join(f"{x:.2f},{['Sí', 'NO', '#N/A'][i % 3]}\n" for i, x in enumerate(r.uniform(0, 10) for _ in range(90)))
    res = generate_project_from_dataset(texto, "y", unidad_de_observacion="una fila")
    assert res["provenance"]["task"] == "classification" and res["provenance"]["rows_dropped_null_target"] == 0


def test_m1_con_la_ausencia_declarada_manda_la_declaracion_y_no_se_toca_el_objetivo():
    """Menor m1 de la re-auditoría final: la guarda de `_objetivo_numerico_sin_textos_de_faltante` («con la ausencia
    DECLARADA no se toca nada») no la sostenía ninguna prueba. Sin ella, «?» declarado (un ARFF) pasa a vacío, que con
    tokens declarados ya NO es ausencia, y la regresión se rompe («target … is empty»)."""
    r = random.Random(2)
    filas = ["x1,y"] + [f"{round(r.uniform(0, 10), 2)},{'?' if i % 10 == 3 else round(r.uniform(1, 50), 2)}"
                        for i in range(100)]
    texto = "\n".join(filas) + "\n"
    res = generate_project_from_dataset(texto, "y", tokens_de_ausencia={"?"})
    p = res["provenance"]
    assert p["preparation_spec"]["task"] == "regression"
    assert p.get("rows_dropped_null_target") == 10
    assert prepare_dataset_from_provenance(texto, p).csv_text == res["csv_text"]


def test_m2_el_objetivo_corregido_a_integer_lee_el_n_a_de_excel_como_fila_sin_objetivo():
    """Menor m2: la regla mira el tipo CORREGIDO del objetivo (`column_type_overrides`), no solo el inferido. Un 0/1 con
    «#N/A» (la clásica deja corregirlo a `integer`) es una regresión con esas filas quitadas; con solo el tipo inferido
    volvía el error de antes («no tiene un rango numérico calculable»)."""
    r = random.Random(5)
    filas = ["x1,y"] + [f"{round(r.uniform(0, 10), 2)},{'#N/A' if i % 7 == 3 else i % 2}" for i in range(120)]
    texto = "\n".join(filas) + "\n"
    res = generate_project_from_dataset(texto, "y", column_type_overrides={"y": "integer"})
    p = res["provenance"]
    assert p["preparation_spec"]["task"] == "regression"
    assert p.get("rows_dropped_null_target") == 17
    assert prepare_dataset_from_provenance(texto, p).csv_text == res["csv_text"]
