# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los rangos de normalización de «crear desde datos» salen SOLO de las filas de
ENTRENAMIENTO (corte «rangos de train» del núcleo, A1–A8).

Antes, `generate_project_from_dataset` sacaba el rango de cada columna numérica
(y el del objetivo de regresión) de TODAS las filas: la validación, que elige la
mejor época y publica la cifra, ya había influido en la normalización. Aquí se
ata lo que el corte promete:

* T1/T1b: extremos solo en las filas de validación NO cambian rangos, prompt,
  `.mxai`, `.mxtrain`, CSV de train ni el modelo; con objetivos nulos
  intercalados, la partición es la de las filas CON objetivo, no la del índice
  crudo del fichero.
* T2: con un `split` con tramo de prueba, los extremos de la prueba tampoco
  cuentan.
* T5: la clásica reenvía TODAS las propuestas del CSV entero; un extremo igual a
  su propuesta es una aceptación (y se queda el de train), uno distinto es una
  corrección (y gana). También en el temporal (retardos y objetivo desplazado).
* T6: el objetivo NO se recorta a [0,1] al entrenar con `recortar_objetivo=False`;
  las entradas sí.
* T7: los bordes (sin valores en train, constante en train, degenerada,
  identificador reconsiderado, tipo corregido a número).
* T8: la partición temporal y la legada coinciden en su `train` (por eso los
  rangos del temporal son los correctos aunque el SPLIT se reescriba después).
* T9: la guardia que falla cerrada si el entrenador partiría distinto.
* T3: PARIDAD. Si la validación (y lo que no es train) no trae extremos, todo es
  idéntico a calcular los rangos con el CSV entero —que es lo que hacía el código
  de antes—: rangos, prompt, `.mxai`, `.mxtrain`, CSV preparado y `params_best`.
  (La captura byte a byte contra el código de `52b4bbc`, que no puede vivir en
  una prueba, la repiten las auditorías con su guion `paridad.py`.)
* I1: la elección de época con el objetivo sin recortar (ver su sección).

T4 (ida y vuelta por el Studio) y T5 por el endpoint viven en el backend del
Studio (`test_rangos_de_train_studio.py`).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import time

import pytest

from matrixai.playground import (
    _columna_de_salida, _get_job_status, _normalize_csv_with_ranges, _submit_training_job,
    _training_jobs)
from matrixai.playground_api import analyze_dataset_csv
from matrixai.training import dense_generator
from matrixai.training.dataset_project import (
    DatasetProjectError, generate_project_from_dataset,
    generate_temporal_project_from_dataset)
from matrixai.training.parser import MatrixAITrainingParseError, parse_split_line
from matrixai.training.particion import (
    LINEA_SPLIT_POR_DEFECTO, particion_legada, particion_para, particion_temporal)


# --------------------------------------------------------------------------- datos

def _csv(filas, cols):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(cols)
    for f in filas:
        w.writerow(["" if f[c] is None else f[c] for c in cols])
    return out.getvalue()


def _filas(n=100, seed=1):
    """x1, x2 -> y. Los extremos de cada columna están PLANTADOS dentro de train
    (filas 30 y 40), así que el resto de filas, validación incluida, cae dentro
    del rango de train salvo lo que cada prueba ponga a propósito."""
    r = random.Random(seed)
    filas = []
    for _ in range(n):
        x1, x2 = round(r.uniform(2, 8), 3), round(r.uniform(1, 4), 3)
        filas.append({"x1": x1, "x2": x2, "y": round(3 * x1 + 2 * x2 + r.gauss(0, .3), 3)})
    filas[30].update(x1=0.0, x2=0.5, y=0.0)
    filas[40].update(x1=10.0, x2=5.0, y=50.0)
    return filas


COLS = ["x1", "x2", "y"]


def _con(filas, celdas):
    """Copia de `filas` con `{(fila, columna): valor}` puesto."""
    filas = copy.deepcopy(filas)
    for (i, c), v in celdas.items():
        filas[i][c] = v
    return filas


def _propuestas(filas, cols=COLS):
    """Lo que el NÚCLEO propondría mirando solo `filas` (se le pregunta a su
    análisis, no se copia la fórmula)."""
    an = analyze_dataset_csv(_csv(filas, cols))["columns"]
    return {c: tuple(float(v) for v in an[c]["proposed_range"]) for c in cols}


def _gen(filas, cols=COLS, objetivo="y", **kw):
    return generate_project_from_dataset(_csv(filas, cols), objetivo, **kw)


def _rangos(proj):
    return {c: tuple(r) for c, r in proj["field_ranges"].items()}


def _entrenar(proj, *, recortar_objetivo=True, epocas=4):
    tr = proj.get("target_range")
    sub = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"], epochs_override=epocas,
        field_ranges=proj.get("field_ranges"), target_range=tuple(tr) if tr else None,
        seed=42, recortar_objetivo=recortar_objetivo)
    assert sub.get("ok"), sub
    st = {}
    for _ in range(900):
        st = _get_job_status(sub["job_id"])
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st.get("error")
    st["_job_id"] = sub["job_id"]
    return st


def _lineas_de_train(proj, n_train):
    return proj["csv_text"].splitlines()[: 1 + n_train]


# --------------------------------------------------------------------------- T1

def test_T1_extremos_solo_en_la_validacion_no_cambian_nada_REGRESION():
    base = _filas()
    a = _gen(_con(base, {(85, "x1"): 1000.0, (86, "x2"): -500.0, (87, "y"): 5000.0}))
    b = _gen(_con(base, {(85, "x1"): 9000.0, (86, "x2"): -900.0, (87, "y"): 90000.0}))
    assert _rangos(a) == _rangos(b)
    assert a["target_range"] == b["target_range"]
    assert a["provenance"]["synthesized_prompt"] == b["provenance"]["synthesized_prompt"]
    assert a["mxai"] == b["mxai"]
    assert a["training_text"] == b["training_text"]
    n_train = a["provenance"]["range_fit"]["n_rows_fit"]
    assert n_train == 80
    assert _lineas_de_train(a, n_train) == _lineas_de_train(b, n_train)
    # y son los de train: los que el núcleo propondría mirando SOLO las 80 primeras
    esperado = _propuestas(base[:80])
    assert _rangos(a) == {c: esperado[c] for c in ("x1", "x2")}
    assert tuple(a["target_range"]) == esperado["y"]
    assert a["field_ranges"]["x1"][1] < 20           # no 1000 ni 9000
    # el MODELO es el mismo: los dos extremos de las entradas superan el `hi` de
    # train y se recortan los dos a 1,0 (y el objetivo, con el recorte de siempre)
    assert _entrenar(a)["params_best"] == _entrenar(b)["params_best"]


def test_T1_extremos_del_objetivo_solo_en_validacion_misma_traza_de_train_loss():
    base = _filas()
    a = _gen(_con(base, {(85, "y"): 400.0}))
    b = _gen(_con(base, {(85, "y"): 900.0}))
    assert a["target_range"] == b["target_range"]
    ta = _entrenar(a, recortar_objetivo=False)
    tb = _entrenar(b, recortar_objetivo=False)
    la = [e["train_loss"] for e in ta["epochs"]]
    lb = [e["train_loss"] for e in tb["epochs"]]
    k = min(len(la), len(lb))
    assert k >= 3
    assert la[:k] == lb[:k]                           # el train no ve la validación
    # La fila 85 (400 / 900) está fuera del rango del objetivo de train: es
    # INALCANZABLE, así que la ELECCIÓN de época no la mira (I1) y las dos
    # pérdidas de validación coinciden...
    assert [e["validation_loss"] for e in ta["epochs"][:k]] == \
           [e["validation_loss"] for e in tb["epochs"][:k]]
    assert ta["epoch_selection"] == {"rule": "target_within_range", "validation_rows": 19, "of": 20}
    # ...y el objetivo SIN recortar sí llega a lo que se MIDE: la cifra es distinta
    assert ta["mae"] != tb["mae"] and ta["r2"] != tb["r2"]


def test_T1_paridad_si_la_validacion_no_se_sale_los_rangos_son_los_del_csv_entero():
    filas = _filas()                                   # validación dentro del rango de train
    proj = _gen(filas)
    entero = _propuestas(filas)
    assert _rangos(proj) == {c: entero[c] for c in ("x1", "x2")}
    assert tuple(proj["target_range"]) == entero["y"]


def test_T1b_objetivos_nulos_intercalados_la_particion_es_la_de_las_filas_CON_objetivo():
    base = _filas()
    nulos = {(i, "y"): None for i in range(10)}        # 90 filas con objetivo
    # train = int(90 * 0.8) = 72 filas CON objetivo = filas crudas 10..81.
    # La fila cruda 81 es de TRAIN por índice preparado (y de validación por
    # índice crudo: 81 >= 80); la 82 es la primera de validación.
    filas = _con(base, {**nulos, (81, "x1"): 500.0, (82, "x1"): 9000.0})
    proj = _gen(filas)
    assert proj["provenance"]["range_fit"]["n_rows_with_target"] == 90
    assert proj["provenance"]["range_fit"]["n_rows_fit"] == 72
    esperado = _propuestas([f for f in filas[10:82]])
    assert proj["field_ranges"]["x1"][1] == esperado["x1"][1]
    assert 500 < proj["field_ranges"]["x1"][1] < 600     # incluye el 500, no el 9000


def test_T1b_el_tamano_que_ve_el_generador_es_el_de_las_filas_CON_objetivo():
    """`dataset_rows` (lo que dimensiona la red y redacta el aviso de «dataset
    pequeño») cuenta las filas que van a ENTRENAR —las que tienen objetivo, el
    mismo `filas_con_objetivo` de la partición—, no las del fichero (contrato
    64 C2). Ninguna prueba lo ataba: contar las del fichero salía verde
    (auditoría P2c). 30 filas con objetivo y 3.000 sin él: el aviso dice 30."""
    r = random.Random(1)
    cols = [f"x{i}" for i in range(6)] + ["y"]
    filas = []
    for i in range(3030):
        f = {f"x{j}": round(r.uniform(0, 10), 3) for j in range(6)}
        f["y"] = round(sum(f.values()) + r.gauss(0, .3), 3) if i < 30 else None
        filas.append(f)
    proj = _gen(filas, cols)
    assert proj["provenance"]["range_fit"]["n_rows_with_target"] == 30
    avisos = json.dumps(proj.get("pipeline_stages"), ensure_ascii=False)
    assert "El dataset (30 filas) es pequeño" in avisos
    assert "3.030" not in avisos and "3030" not in avisos


# --------------------------------------------------------------------------- T2

SPLIT_CON_PRUEBA = "SPLIT train=0.6 validation=0.2 test=0.2 seed=7 protocol=2"


def test_T2_con_tramo_de_prueba_los_extremos_de_la_prueba_no_cuentan():
    base = _filas()
    p = particion_para(100, parse_split_line(SPLIT_CON_PRUEBA))
    assert len(p.test) == 20 and len(p.train) == 60
    fila_prueba_a, fila_prueba_b = p.test[0], p.test[1]
    fila_val = p.validation[0]
    a = _gen(_con(base, {(fila_prueba_a, "x1"): 1000.0, (fila_prueba_b, "y"): 5000.0,
                         (fila_val, "x2"): 700.0}), split=SPLIT_CON_PRUEBA)
    b = _gen(_con(base, {(fila_prueba_a, "x1"): 9000.0, (fila_prueba_b, "y"): 90000.0,
                         (fila_val, "x2"): 800.0}), split=SPLIT_CON_PRUEBA)
    assert _rangos(a) == _rangos(b) and a["target_range"] == b["target_range"]
    assert a["training_text"].count(SPLIT_CON_PRUEBA) == 1
    assert SPLIT_CON_PRUEBA in a["training_artifacts"]["training_text"]
    rf = a["provenance"]["range_fit"]
    assert rf["split"] == SPLIT_CON_PRUEBA and rf["partition"]["n_test"] == 20
    # rangos = los de las filas de train de ESA partición (barajada con la semilla 7)
    esperado = _propuestas([base[i] for i in p.train])
    assert _rangos(a) == {c: esperado[c] for c in ("x1", "x2")}
    assert tuple(a["target_range"]) == esperado["y"]
    # y el modelo no los ve (validación con el mismo valor recortado a 1,0 en las dos variantes)
    assert _entrenar(a)["params_best"] == _entrenar(b)["params_best"]


def test_T2_un_split_invalido_se_rechaza_con_el_parser_del_mxtrain():
    with pytest.raises(DatasetProjectError, match="split no válido"):
        _gen(_filas(), split="SPLIT train=0.9 validation=0.9")
    with pytest.raises(DatasetProjectError, match="split no válido"):
        _gen(_filas(), split="SPLIT train=0.6 validation=0.2 test=0.2")   # test sin protocol=2


def test_parse_split_line_es_el_parser_del_mxtrain_y_la_constante_es_la_que_se_escribe():
    from matrixai.training.parser import parse_training_text
    spec = parse_split_line(SPLIT_CON_PRUEBA)
    assert (spec.train, spec.validation, spec.test, spec.seed, spec.protocol) == \
        (0.6, 0.2, 0.2, 7, "2")
    with pytest.raises(MatrixAITrainingParseError):
        parse_split_line("SPLIT train=1.5 validation=0.2")
    proj = _gen(_filas())
    assert LINEA_SPLIT_POR_DEFECTO in proj["training_text"]
    assert parse_training_text(proj["training_text"]).dataset.split == \
        parse_split_line(LINEA_SPLIT_POR_DEFECTO)


# --------------------------------------------------------------------------- T3

def _plantar(filas, cols, i_lo, i_hi):
    """Los extremos de cada columna numérica, en dos filas de TRAIN."""
    for c in cols:
        vals = [f[c] for f in filas if isinstance(f.get(c), (int, float))]
        es_int = all(isinstance(v, int) for v in vals)
        filas[i_lo][c] = (min(vals) - 1) if es_int else round(min(vals) - 1.5, 3)
        filas[i_hi][c] = (max(vals) + 1) if es_int else round(max(vals) + 1.5, 3)
    return filas


def _escenarios_t3():
    r = random.Random(11)
    reg = [{"a": round(r.uniform(0, 10), 3), "b": round(r.uniform(-5, 5), 3), "n": r.randint(1, 50)}
           for _ in range(120)]
    for f in reg:
        f["y"] = round(2 * f["a"] - f["b"] + 0.1 * f["n"] + r.gauss(0, .5), 3)
    clas = []
    for _ in range(150):
        a, b, cat = round(r.uniform(0, 1), 4), round(r.uniform(100, 900), 2), r.choice(["rojo", "verde", "azul"])
        clas.append({"a": a, "b": b, "cat": cat, "bo": r.choice(["true", "false"]),
                     "y": "si" if a + b / 900 + (0.3 if cat == "rojo" else 0) > 1.0 else "no"})
    ciudades = [f"c{i:02d}x" for i in range(30)]
    comp = []
    for i in range(300):
        x = round(r.uniform(0, 20), 3)
        c = ciudades[i % 30]
        comp.append({"ciudad": c, "x": x, "y": round(x * 1.5 + ciudades.index(c) * 0.2 + r.gauss(0, 1), 3)})
    nulos = []
    for i in range(130):
        a, k = round(r.uniform(0, 3), 3), r.randint(0, 20)
        nulos.append({"a": a, "k": k, "y": round(a * 3 + k + r.gauss(0, .3), 2)})
    _plantar(nulos, ["a", "k", "y"], 1, 5)
    for i, f in enumerate(nulos):
        if i % 11 == 3:
            f["a"] = None                      # hueco en una entrada (imputación)
        if i % 9 == 4:
            f["y"] = None                      # objetivo nulo intercalado
    entero = [{"a": round(r.uniform(0, 100), 1), "flag": r.choice(["yes", "no"])} for _ in range(110)]
    for f in entero:
        f["y"] = int(f["a"] // 7)
    return {
        "regresion": (_plantar(reg, ["a", "b", "n", "y"], 3, 7), ["a", "b", "n", "y"], "y"),
        "clasificacion": (_plantar(clas, ["a", "b"], 2, 9), ["a", "b", "cat", "bo", "y"], "y"),
        "compuesta": (_plantar(comp, ["x", "y"], 4, 8), ["ciudad", "x", "y"], "y"),
        "nulos": (nulos, ["a", "k", "y"], "y"),
        "objetivo_entero": (_plantar(entero, ["a", "y"], 2, 6), ["a", "flag", "y"], "y"),
    }


@pytest.mark.parametrize("nombre", ["regresion", "clasificacion", "compuesta", "nulos", "objetivo_entero"])
def test_T3_sin_extremos_fuera_de_train_todo_es_como_con_el_csv_entero(nombre):
    filas, cols, objetivo = _escenarios_t3()[nombre]
    texto = _csv(filas, cols)
    an = analyze_dataset_csv(texto)["columns"]
    entero = {c: tuple(float(v) for v in i["proposed_range"]) for c, i in an.items()
              if i.get("type") in ("number", "integer") and i.get("proposed_range") and not i.get("constant")}
    nuevo = generate_project_from_dataset(texto, objetivo)
    # lo que hacía el código de antes: los rangos del CSV ENTERO, que aquí se
    # imponen como correcciones (sin el flag, ganan siempre)
    viejo = generate_project_from_dataset(texto, objetivo, column_range_overrides=entero)
    assert viejo["provenance"]["range_fit"]["user_declared"] == sorted(entero)
    assert nuevo["field_ranges"] == viejo["field_ranges"] and nuevo["target_range"] == viejo["target_range"]
    for clave in ("mxai", "training_text", "csv_text"):
        assert nuevo[clave] == viejo[clave], clave
    assert nuevo["provenance"]["synthesized_prompt"] == viejo["provenance"]["synthesized_prompt"]
    assert sorted(nuevo["provenance"]["range_fit"]["from_train"]) == sorted(entero)
    # y el MODELO, entrenado como el Studio
    a, b = _entrenar(nuevo, recortar_objetivo=False, epocas=3), _entrenar(viejo, recortar_objetivo=False, epocas=3)
    assert a["params_best"] == b["params_best"] and a["epochs"] == b["epochs"]


def test_T3_temporal_sin_extremos_fuera_de_train_los_rangos_son_los_del_csv_entero():
    filas = _serie(140)
    _plantar(filas, ["v", "x"], 10, 15)
    texto = _csv(filas, ["fecha", "v", "x"])
    crudo = _propuestas(filas, ["v", "x"])
    proj = generate_temporal_project_from_dataset(texto, "v", **TEMPORAL)
    assert _rangos(proj) == {"v": crudo["v"], "x": crudo["x"], "v_lag1": crudo["v"], "v_lag2": crudo["v"]}
    assert tuple(proj["target_range"]) == crudo["v"]
    viejo = generate_temporal_project_from_dataset(
        texto, "v", **TEMPORAL,
        column_range_overrides={"v": crudo["v"], "x": crudo["x"], "v_lag1": crudo["v"], "v_lag2": crudo["v"]})
    for clave in ("mxai", "training_text", "csv_text"):
        assert proj[clave] == viejo[clave], clave


# --------------------------------------------------------------------------- T5

def _reenviar_todo(filas, cols=COLS):
    """Lo que hace la clásica: TODAS las propuestas del CSV ENTERO como override."""
    return _propuestas(filas, cols)


def test_T5_las_propuestas_reenviadas_sin_tocar_son_aceptacion_y_salen_los_de_train():
    base = _filas()
    filas = _con(base, {(85, "x1"): 1000.0, (87, "y"): 5000.0})
    reenviado = _reenviar_todo(filas)                        # las del CSV ENTERO
    assert reenviado["x1"][1] > 1000
    sin = _gen(filas)                                        # lo que sale con los de train
    con = _gen(filas, column_range_overrides=reenviado, rangos_reenviados_del_analisis=True)
    assert _rangos(con) == _rangos(sin) and con["target_range"] == sin["target_range"]
    rf = con["provenance"]["range_fit"]
    assert rf["accepted_proposal"] == ["x1", "x2", "y"] and rf["user_declared"] == []
    assert rf["from_train"] == []


def test_T5_sin_el_flag_el_override_gana_siempre_invariante_8():
    filas = _con(_filas(), {(85, "x1"): 1000.0})
    reenviado = _reenviar_todo(filas)
    proj = _gen(filas, column_range_overrides=reenviado)
    assert _rangos(proj)["x1"] == reenviado["x1"]
    assert proj["provenance"]["range_fit"]["user_declared"] == ["x1", "x2", "y"]


def test_T5_una_correccion_se_conserva_y_una_edicion_a_medias_conserva_lo_editado():
    filas = _con(_filas(), {(85, "x1"): 1000.0})
    reenviado = _reenviar_todo(filas)
    train = _propuestas(filas[:80])
    ov = dict(reenviado)
    ov["x2"] = (-100.0, 100.0)                               # corregida a mano: gana
    ov["x1"] = (-5.0, reenviado["x1"][1])                    # a medias: lo editado + hi = propuesta
    proj = _gen(filas, column_range_overrides=ov, rangos_reenviados_del_analisis=True)
    assert tuple(proj["field_ranges"]["x2"]) == (-100.0, 100.0)
    assert tuple(proj["field_ranges"]["x1"]) == (-5.0, train["x1"][1])
    rf = proj["provenance"]["range_fit"]
    assert "x2" in rf["user_declared"] and "x2" not in rf["accepted_proposal"]
    assert "x1" in rf["user_declared"] and "x1" in rf["accepted_proposal"]
    assert tuple(proj["target_range"]) == train["y"]


def test_T5_un_extremo_que_degeneraria_el_par_se_usa_tal_como_llego():
    filas = _con(_filas(), {(85, "x1"): 1000.0})
    reenviado = _reenviar_todo(filas)
    train_hi = _propuestas(filas[:80])["x1"][1]
    # lo editado (lo) por ENCIMA del hi de train: aceptar el hi de train daría lo >= hi
    ov = {**reenviado, "x1": (train_hi + 50.0, reenviado["x1"][1])}
    proj = _gen(filas, column_range_overrides=ov, rangos_reenviados_del_analisis=True)
    assert tuple(proj["field_ranges"]["x1"]) == (train_hi + 50.0, reenviado["x1"][1])


def _serie(n=100, seed=3):
    r = random.Random(seed)
    filas = [{"fecha": f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}",
              "v": round(r.uniform(2, 8), 3), "x": round(r.uniform(1, 4), 3)} for i in range(n)]
    filas[30].update(v=0.0, x=0.5)
    filas[40].update(v=10.0, x=5.0)
    return filas


TEMPORAL = dict(temporal_column="fecha", horizon=1, lag_window_columns=["v"], lag_window_size=2)


def test_T5_temporal_overrides_crudos_y_retardos_con_la_propuesta_de_su_origen():
    # Extremos solo en la validación: la fila 90 (dentro del pipeline) y la 99, la
    # ÚLTIMA, que el pipeline descarta (no tiene objetivo futuro) — ahí el
    # análisis del CSV crudo ve 500/900 y el del CSV post-pipeline no, así que
    # comparar contra el análisis interno (en vez del crudo) NO daría lo mismo.
    filas = _con(_serie(), {(90, "v"): 500.0, (90, "x"): 900.0, (99, "v"): 600.0, (99, "x"): 950.0})
    cols = ["fecha", "v", "x"]
    crudo = _propuestas(filas, ["v", "x"])
    ov = {"v": crudo["v"], "x": crudo["x"],
          "v_lag1": crudo["v"], "v_lag2": crudo["v"]}              # como la clásica
    sin = generate_temporal_project_from_dataset(_csv(filas, cols), "v", **TEMPORAL)
    con = generate_temporal_project_from_dataset(
        _csv(filas, cols), "v", **TEMPORAL, column_range_overrides=ov,
        rangos_reenviados_del_analisis=True)
    assert _rangos(con) == _rangos(sin) and con["target_range"] == sin["target_range"]
    assert max(r[1] for r in _rangos(con).values()) < 100 and con["target_range"][1] < 100
    # sin el flag, los reenviados del CSV entero ganan (el comportamiento de antes)
    sin_flag = generate_temporal_project_from_dataset(
        _csv(filas, cols), "v", **TEMPORAL, column_range_overrides=ov)
    assert sin_flag["field_ranges"]["v"][1] > 500 and sin_flag["field_ranges"]["x"][1] > 900
    rf = con["provenance"]["range_fit"]
    assert rf["split"].startswith("SPLIT") and "mode=temporal" in rf["split"]
    assert rf["partition"]["mode"] == "temporal"
    assert {"v", "x", "v_lag1", "v_lag2"} <= set(rf["accepted_proposal"])


def test_T5_temporal_el_objetivo_desplazado_se_compara_con_la_propuesta_de_su_columna_cruda():
    """La clásica le reenvía al objetivo DESPLAZADO (`v_target_h1`) la propuesta
    de su columna cruda (`v`), y es con ESA con la que hay que compararlo para
    saber si es una aceptación. Con el máximo de `v` en la PRIMERA fila —que no
    es objetivo de nadie: el desplazamiento la deja sin futuro—, la propuesta del
    `v` crudo (que lo ve) y la del objetivo desplazado (que no) difieren; sin el
    mapeo, lo reenviado contaría como corrección del usuario y el objetivo se
    quedaría con el [-100, 1100] del CSV crudo (auditoría P3g, sabotaje verde)."""
    filas = _con(_serie(), {(0, "v"): 1000.0, (30, "v"): 0.0, (90, "v"): 500.0})
    cols = ["fecha", "v", "x"]
    crudo = _propuestas(filas, ["v", "x"])
    assert crudo["v"] == (-100.0, 1100.0)
    ov = {"v": crudo["v"], "x": crudo["x"], "v_lag1": crudo["v"], "v_lag2": crudo["v"]}
    sin = generate_temporal_project_from_dataset(_csv(filas, cols), "v", **TEMPORAL)
    con = generate_temporal_project_from_dataset(
        _csv(filas, cols), "v", **TEMPORAL, column_range_overrides=ov,
        rangos_reenviados_del_analisis=True)
    assert con["target_range"] == sin["target_range"]
    assert con["target_range"][1] < 100                      # no el 1100 del v crudo
    rf = con["provenance"]["range_fit"]
    assert "v_target_h1" in rf["accepted_proposal"] and "v_target_h1" not in rf["user_declared"]


# --------------------------------------------------------------------------- T6

def test_T6_el_objetivo_no_se_recorta_y_las_entradas_si():
    rangos = {"x1": (0.0, 10.0), "y": (0.0, 100.0)}
    texto = "x1,y\n20,200\n-5,-50\n5,50\n"
    sin = list(csv.DictReader(io.StringIO(
        _normalize_csv_with_ranges(texto, rangos, sin_recorte=frozenset({"y"})))))
    assert [float(f["x1"]) for f in sin] == [1.0, 0.0, 0.5]          # entradas: recortadas
    assert [float(f["y"]) for f in sin] == [2.0, -0.5, 0.5]          # objetivo: tal cual
    con = list(csv.DictReader(io.StringIO(_normalize_csv_with_ranges(texto, rangos))))
    assert [float(f["y"]) for f in con] == [1.0, 0.0, 0.5]           # lo de siempre


def test_T6_entrenar_sin_recortar_el_objetivo_cambia_la_perdida_de_validacion_y_se_declara():
    base = _filas()
    a = _gen(_con(base, {(85, "y"): 400.0}))
    b = _gen(_con(base, {(85, "y"): 900.0}))
    # recortado (lo de siempre): las dos validaciones coinciden
    ra, rb = _entrenar(a, epocas=3), _entrenar(b, epocas=3)
    assert [e["validation_loss"] for e in ra["epochs"]] == [e["validation_loss"] for e in rb["epochs"]]
    assert "target_clipped" not in _training_jobs[ra["_job_id"]]["run_provenance"]
    assert ra["mae"] == rb["mae"]                                     # y la cifra también: verdad recortada
    assert "epoch_selection" not in ra
    # sin recortar: la verdad de validación es la verdad EN LA CIFRA (mae/r2 con
    # el 400 y el 900 de verdad), y la elección de época no mira lo inalcanzable
    sa, sb = _entrenar(a, recortar_objetivo=False, epocas=3), _entrenar(b, recortar_objetivo=False, epocas=3)
    assert sa["mae"] != sb["mae"]
    assert [e["validation_loss"] for e in sa["epochs"]] == [e["validation_loss"] for e in sb["epochs"]]
    assert _training_jobs[sa["_job_id"]]["run_provenance"]["target_clipped"] is False
    # y LO QUE SE ENTRENÓ lleva el objetivo sin recortar: el CSV que recibió el
    # entrenador (su huella), comparado con el que habría recibido recortando
    salida = _columna_de_salida(a["mxai"])
    rangos = {**a["field_ranges"], salida: tuple(a["target_range"])}
    sin = _normalize_csv_with_ranges(a["csv_text"], rangos, sin_recorte=frozenset({salida}))
    con = _normalize_csv_with_ranges(a["csv_text"], rangos)
    assert sin != con
    assert _training_jobs[sa["_job_id"]]["trained_csv_sha256"] == hashlib.sha256(sin.encode()).hexdigest()
    assert _training_jobs[ra["_job_id"]]["trained_csv_sha256"] == hashlib.sha256(con.encode()).hexdigest()


def test_T6_el_caso_ordenado_sin_dominio_declarado_da_la_cifra_honrada():
    """Kelvin ascendente SIN declarar el dominio de `centigrados`: la validación
    está fuera de lo visto al entrenar (en la entrada, que se recorta, y en el
    objetivo, que no). Antes el rango de todas las filas lo tapaba."""
    filas = [{"c": c, "k": round(c + 273.15, 2)} for c in range(100)]
    ascendente = _gen(filas, ["c", "k"], "k")
    barajado = random.Random(5).sample(filas, len(filas))
    revuelto = _gen(barajado, ["c", "k"], "k")
    # el rango de `c` ya no es el del CSV entero ([-10, 109]) sino el de las 80 primeras filas
    assert ascendente["field_ranges"]["c"][1] < 100 and revuelto["field_ranges"]["c"][1] >= 100
    ra = _entrenar(ascendente, recortar_objetivo=False, epocas=50)["r2"]
    rr = _entrenar(revuelto, recortar_objetivo=False, epocas=50)["r2"]
    assert ra < rr                                                  # extrapolar cuesta, y ahora se ve


# --------------------------------------------------------------------------- I1
#
# LA ELECCIÓN DE ÉPOCA Y LA CIFRA SON DOS PREGUNTAS (auditoría I1, 03-10). Con el
# objetivo sin recortar (A8), la mejor época y la parada temprana se eligen SOLO
# con las filas de validación cuyo objetivo cae dentro de su rango —las que el
# modelo puede alcanzar—, y `mae`/`rmse`/`r2` se miden con TODAS, sin recortar.
# Medido antes de este arreglo, Kelvin ascendente por torch: época 1 en vez de
# la 50 y 1,6 K de error DENTRO del dominio en vez de 0,008.

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _kelvin(orden):
    return [{"centigrados": c, "k": round(c + 273.15, 2)} for c in orden]


def _entrenar_con(proj, backend, monkeypatch, *, recortar_objetivo, seed=42, epocas=50):
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", backend)
    tr = proj.get("target_range")
    sub = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"], epochs_override=epocas,
        field_ranges=proj.get("field_ranges"), target_range=tuple(tr) if tr else None,
        seed=seed, recortar_objetivo=recortar_objetivo)
    assert sub.get("ok"), sub
    st = {}
    for _ in range(1500):
        st = _get_job_status(sub["job_id"])
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert st["status"] == "done", st.get("error")
    assert st["backend"] == backend, st.get("backend")
    st["_job_id"] = sub["job_id"]
    return st


def _predice_kelvin(proj, params_best, centigrados):
    """Lo que predice el modelo entrenado, en kelvin, como al predecir: la entrada
    normalizada con su rango y recortada a [0, 1], la salida desnormalizada."""
    from matrixai.forward.dense_forward import dense_forward
    from matrixai.parameters.store import ParameterSet
    from matrixai.parser import parse_text
    net = parse_text(proj["mxai"]).networks[0]
    ps = ParameterSet.from_dict(params_best)
    lo, hi = proj["field_ranges"]["centigrados"]
    tlo, thi = proj["target_range"]
    return [tlo + dense_forward(net, ps, [min(1.0, max(0.0, (c - lo) / (hi - lo)))])[0] * (thi - tlo)
            for c in centigrados]


def _mae(pred, verdad):
    return sum(abs(p - v) for p, v in zip(pred, verdad)) / len(verdad)


TRAIN_K, VAL_K = list(range(80)), list(range(80, 100))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch no instalado")
@pytest.mark.parametrize("seed", [42, 1])
def test_I1_torch_kelvin_ordenado_elige_la_epoca_de_siempre_y_publica_la_cifra_honrada(monkeypatch, seed):
    proj = _gen(_kelvin(range(100)), ["centigrados", "k"], "k")
    sin = _entrenar_con(proj, "torch", monkeypatch, recortar_objetivo=False, seed=seed)
    con = _entrenar_con(proj, "torch", monkeypatch, recortar_objetivo=True, seed=seed)
    # LA ELECCIÓN: la misma época y el mismo modelo que con el objetivo recortado
    # (sin la regla, con la verdad sin recortar, salía la época 1)
    assert sin["best_epoch"] == con["best_epoch"] == 50
    assert sin["params_best"] == con["params_best"]
    # y el modelo acierta DENTRO del dominio (las filas de train): centésimas de K
    assert _mae(_predice_kelvin(proj, sin["params_best"], TRAIN_K), [c + 273.15 for c in TRAIN_K]) < 0.05
    # se eligió con las 7 filas alcanzables (80..86 °C) de las 20 de validación
    assert sin["epoch_selection"] == {"rule": "target_within_range", "validation_rows": 7, "of": 20}
    # LA CIFRA: la de la verdad SIN recortar, calculada a mano con el mismo modelo
    pred_val = _predice_kelvin(proj, sin["params_best"], VAL_K)
    assert sin["mae"] == pytest.approx(_mae(pred_val, [c + 273.15 for c in VAL_K]), abs=1e-3)
    assert sin["mae"] > 3.0 and sin["r2"] < 0.5 < 0.99 < con["r2"]      # extrapolar no se tapa


@pytest.mark.parametrize("seed", [0, 9])
def test_I1_stdlib_kelvin_ordenado_elige_con_las_filas_alcanzables(monkeypatch, seed):
    """Por stdlib (SGD de lote 1) RECORTAR la verdad de las filas inalcanzables
    tampoco vale: con la semilla 0 elegía la época 9 (0,41 K de error en dominio).
    Y sin regla ninguna, con la 9, la época 1 (1,18 K). Excluyéndolas: ~0 K."""
    proj = _gen(_kelvin(range(100)), ["centigrados", "k"], "k")
    st = _entrenar_con(proj, "stdlib", monkeypatch, recortar_objetivo=False, seed=seed)
    assert _mae(_predice_kelvin(proj, st["params_best"], TRAIN_K), [c + 273.15 for c in TRAIN_K]) < 0.05
    assert st["epoch_selection"] == {"rule": "target_within_range", "validation_rows": 7, "of": 20}


@pytest.mark.parametrize("backend", ["stdlib", pytest.param(
    "torch", marks=pytest.mark.skipif(not _HAS_TORCH, reason="torch no instalado"))])
def test_I1_datos_barajados_la_eleccion_no_cambia_nada(monkeypatch, backend):
    """Si ninguna fila de validación se sale del rango del objetivo, elegir con
    las alcanzables es elegir con todas: lo mismo que antes, byte a byte."""
    proj = _gen(random.Random(5).sample(_kelvin(range(100)), 100), ["centigrados", "k"], "k")
    sin = _entrenar_con(proj, backend, monkeypatch, recortar_objetivo=False, epocas=20)
    con = _entrenar_con(proj, backend, monkeypatch, recortar_objetivo=True, epocas=20)
    assert sin["epoch_selection"] == {"rule": "target_within_range", "validation_rows": 20, "of": 20}
    assert sin["params_best"] == con["params_best"] and sin["epochs"] == con["epochs"]
    assert (sin["r2"], sin["mae"], sin["best_epoch"]) == (con["r2"], con["mae"], con["best_epoch"])


def test_I1_si_ninguna_fila_de_validacion_es_alcanzable_se_elige_recortando(monkeypatch):
    filas = [{"x": i, "y": round(i / 10, 2) if i < 80 else 100 + i} for i in range(100)]
    proj = _gen(filas, ["x", "y"], "y")
    assert proj["target_range"][1] < 100
    sin = _entrenar_con(proj, "stdlib", monkeypatch, recortar_objetivo=False, epocas=6)
    con = _entrenar_con(proj, "stdlib", monkeypatch, recortar_objetivo=True, epocas=6)
    assert sin["epoch_selection"] == {"rule": "target_clipped", "validation_rows": 20, "of": 20}
    assert sin["params_best"] == con["params_best"] and sin["best_epoch"] == con["best_epoch"]
    assert sin["mae"] > con["mae"]                       # la cifra, con la verdad sin recortar


def _con_ciudad(filas):
    """Las mismas filas con una categórica de 30 niveles: el generador hace una
    red COMPUESTA (embedding), que es otro camino de entrenamiento."""
    r = random.Random(4)
    ciudades = [f"c{i:02d}x" for i in range(30)]
    filas = copy.deepcopy(filas)
    for i, f in enumerate(filas):
        f["ciudad"] = ciudades[i % 30] if i < 30 else r.choice(ciudades)
    return filas


@pytest.mark.parametrize("backend", ["stdlib", pytest.param(
    "torch", marks=pytest.mark.skipif(not _HAS_TORCH, reason="torch no instalado"))])
def test_I1_la_red_compuesta_elige_igual(monkeypatch, backend):
    base = _con_ciudad(_filas())
    cols = ["ciudad", "x1", "x2", "y"]
    a = _gen(_con(base, {(85, "y"): 400.0}), cols)
    b = _gen(_con(base, {(85, "y"): 900.0}), cols)
    assert "EMBEDDING ciudad_emb" in a["mxai"]
    sa = _entrenar_con(a, backend, monkeypatch, recortar_objetivo=False, epocas=3)
    sb = _entrenar_con(b, backend, monkeypatch, recortar_objetivo=False, epocas=3)
    assert sa.get("network_kind") == "composite_network"
    assert sa["epoch_selection"] == {"rule": "target_within_range", "validation_rows": 19, "of": 20}
    assert [e["validation_loss"] for e in sa["epochs"]] == [e["validation_loss"] for e in sb["epochs"]]
    assert sa["params_best"] == sb["params_best"]
    assert sa["mae"] != sb["mae"]
    # y sin nada inalcanzable, igual que recortando (por torch, la validación
    # explícita sustituye al 80/20 interno del trainer: las mismas filas)
    c = _gen(base, cols)
    sc = _entrenar_con(c, backend, monkeypatch, recortar_objetivo=False, epocas=3)
    rc = _entrenar_con(c, backend, monkeypatch, recortar_objetivo=True, epocas=3)
    assert sc["epoch_selection"]["validation_rows"] == 20
    assert sc["params_best"] == rc["params_best"] and sc["epochs"] == rc["epochs"]


@pytest.mark.skipif(not _HAS_TORCH, reason="torch no instalado")
def test_I1_el_transformer_elige_igual():
    from matrixai.playground import _run_playground_training
    from matrixai.training.transformer_generator import TransformerNetworkGenerator
    gen = TransformerNetworkGenerator().generate("resenas: Text[16]\nOUTPUT puntuacion: Scalar")
    muestras = [
        ("me encanta este producto", 0.9), ("terrible experiencia", 0.1),
        ("bastante bueno en general", 0.7), ("no lo recomiendo nunca", 0.2),
        ("calidad excelente de verdad", 0.95), ("una decepcion total", 0.05),
        ("cumple lo que promete bien", 0.75), ("muy malo no comprar", 0.15),
    ] * 2
    def _csv_con(ultimo):
        filas = muestras[:-1] + [(muestras[-1][0], ultimo)]          # la fila 15 es de validación
        return "resenas,predicted_value\n" + "".join(f"{t},{v}\n" for t, v in filas)
    r = {v: _run_playground_training(gen.mxai_text, gen.training_text, _csv_con(v), epochs_override=4,
                                     target_range=(0.0, 1.0), recortar_objetivo=False)
         for v in (5.0, 9.0)}
    assert r[5.0]["ok"] and r[9.0]["ok"], (r[5.0].get("error"), r[9.0].get("error"))
    assert r[5.0]["epoch_selection"] == {"rule": "target_within_range", "validation_rows": 3, "of": 4}
    assert [e["validation_loss"] for e in r[5.0]["epochs"]] == [e["validation_loss"] for e in r[9.0]["epochs"]]
    assert r[5.0]["mae"] != r[9.0]["mae"]


# --------------------------------------------------------------------------- T7

def _con_columna(base, nombre, valores):
    filas = copy.deepcopy(base)
    for f, v in zip(filas, valores):
        f[nombre] = v
    return filas


def test_T7_columna_sin_ningun_valor_en_train_usa_el_csv_entero_y_lo_declara():
    base = _filas()
    valores = [None] * 80 + [round(1 + 0.1 * i, 2) for i in range(20)]
    proj = _gen(_con_columna(base, "x3", valores), COLS[:2] + ["x3", "y"])
    rf = proj["provenance"]["range_fit"]
    assert rf["fallback_full_csv"] == {"x3": "no_values_in_train"}
    assert "x3" not in rf["from_train"]
    csv_entero = analyze_dataset_csv(_csv(_con_columna(base, "x3", valores), ["x3"]))
    assert tuple(proj["field_ranges"]["x3"]) == tuple(csv_entero["columns"]["x3"]["proposed_range"])
    avisos = json.dumps(proj.get("pipeline_stages"), ensure_ascii=False)
    assert "x3" in avisos and "ningún valor en las filas de entrenamiento" in avisos


def test_T7_columna_constante_en_train_pero_no_en_el_csv_se_declara_y_avisa():
    base = _filas()
    valores = [5.0] * 80 + [round(5 + 0.5 * i, 2) for i in range(1, 21)]
    proj = _gen(_con_columna(base, "x3", valores), COLS[:2] + ["x3", "y"])
    rf = proj["provenance"]["range_fit"]
    assert rf["constant_in_train"] == ["x3"]
    assert tuple(proj["field_ranges"]["x3"]) == (4.0, 6.0)          # ±1 de `_propose_margin`
    assert "constantes en las filas" in json.dumps(proj.get("pipeline_stages"), ensure_ascii=False) \
        or "constante(s) en las filas" in json.dumps(proj.get("pipeline_stages"), ensure_ascii=False)
    en = _gen(_con_columna(base, "x3", valores), COLS[:2] + ["x3", "y"], locale="en")
    assert "constant in the training rows" in json.dumps(en.get("pipeline_stages"))


def test_T7_rango_que_degenera_al_redondear_usa_el_csv_entero_y_lo_declara():
    base = _filas()
    valores = [1.0, 1.00000001] * 40 + [round(1 + 0.1 * i, 2) for i in range(20)]
    filas = _con_columna(base, "x3", valores)
    proj = _gen(filas, COLS[:2] + ["x3", "y"])
    rf = proj["provenance"]["range_fit"]
    assert rf["fallback_full_csv"] == {"x3": "degenerate_range"}
    assert proj["field_ranges"]["x3"][0] < proj["field_ranges"]["x3"][1]
    assert proj["field_ranges"]["x3"][1] > 2.5                      # el del CSV entero


def test_T7_identificador_reconsiderado_toma_el_rango_de_train():
    filas = [{"centigrados": c, "k": round(c + 273.15, 2)} for c in range(100)]
    proj = _gen(filas, ["centigrados", "k"], "k")
    assert any("reconsidered_identifier_as_feature:centigrados" in op
               for op in proj["provenance"]["operations"])
    assert tuple(proj["field_ranges"]["centigrados"]) == (-8.0, 87.0)       # 0..79 ±10 %, enteros
    assert "centigrados" in proj["provenance"]["range_fit"]["from_train"]


def test_T7_tipo_corregido_a_numero_toma_el_rango_de_train_con_ese_tipo():
    filas = [{"centigrados": c, "k": round(c + 273.15, 2)} for c in range(100)]
    proj = _gen(filas, ["centigrados", "k"], "k",
                column_type_overrides={"centigrados": "number"})
    assert tuple(proj["field_ranges"]["centigrados"]) == (-7.9, 86.9)
    assert "centigrados" in proj["provenance"]["range_fit"]["from_train"]


# --------------------------------------------------------------------------- T8

def test_T8_la_particion_temporal_y_la_legada_coinciden_en_su_train():
    for n in range(0, 5001):
        legada, temporal = particion_legada(n).train, particion_temporal(n, 0.8).train
        assert len(legada) == len(temporal), n
        if n <= 300:
            assert legada == temporal, n


def test_T8_generacion_temporal_con_extremos_en_el_ultimo_20_por_ciento():
    filas = _con(_serie(), {(95, "v"): 500.0, (96, "x"): 900.0})
    proj = generate_temporal_project_from_dataset(_csv(filas, ["fecha", "v", "x"]), "v", **TEMPORAL)
    assert max(r[1] for r in _rangos(proj).values()) < 100
    assert proj["target_range"][1] < 100
    assert "mode=temporal" in proj["training_text"]


# --------------------------------------------------------------------------- T9

def test_T9_si_el_generador_escribe_otro_SPLIT_que_el_de_los_rangos_no_se_genera(monkeypatch):
    monkeypatch.setattr(dense_generator, "LINEA_SPLIT_POR_DEFECTO",
                        "SPLIT train=0.6 validation=0.2 test=0.2 seed=7 protocol=2")
    with pytest.raises(DatasetProjectError, match="partición"):
        _gen(_filas())
    # pasando el `split` que de verdad se va a usar, sí
    proj = _gen(_filas(), split="SPLIT train=0.6 validation=0.2 test=0.2 seed=7 protocol=2")
    assert proj["provenance"]["range_fit"]["partition"]["protocol"] == "2"


def test_T9_temporal_si_el_SPLIT_por_omision_cambia_la_reescritura_temporal_no_genera(monkeypatch):
    """La guardia del envoltorio temporal (`_forzar_split_temporal_en_proyecto`):
    los rangos se ajustan con el SPLIT de la generación y después el SPLIT se
    reescribe a `mode=temporal`. Hoy sus `train` coinciden (T8), así que la
    guardia no salta; pero si el SPLIT por omisión cambiara de ratio —la legada
    lo IGNORA (0,8 fijo) y el temporal lo honra—, los rangos se ajustarían con
    el 80 % y el entrenador partiría por el 70 %: habrían visto filas que él
    reserva. Aquí se provoca ese cambio y se exige que NO se genere (auditoría
    P6b: la recomprobación era inalcanzable y ninguna prueba la ataba)."""
    from matrixai.training import dataset_project as dp
    otro_ratio = "SPLIT train=0.7 validation=0.3 seed=42"
    monkeypatch.setattr(dense_generator, "LINEA_SPLIT_POR_DEFECTO", otro_ratio)
    monkeypatch.setattr(dp, "LINEA_SPLIT_POR_DEFECTO", otro_ratio)
    # la generación sin temporal SÍ sale (su partición es la de sus rangos)...
    assert _gen(_filas())["provenance"]["range_fit"]["split"] == otro_ratio
    # ...y la temporal no, porque el entrenador partiría distinto
    with pytest.raises(DatasetProjectError, match="partición"):
        generate_temporal_project_from_dataset(_csv(_serie(), ["fecha", "v", "x"]), "v", **TEMPORAL)


def test_por_omision_el_objetivo_SE_SIGUE_recortando_la_densa_de_los_motores_lo_hereda():
    """El Studio y las rutas del núcleo piden `recortar_objetivo=False` (A8); quien no lo pide —la densa de los
    motores (120-C4), que normaliza con sus rangos de train y compara `params_best` con objetivos de validación
    distintos— hereda el comportamiento de antes. Esta prueba lo fija AQUÍ: hasta hoy solo lo cazaba una prueba de
    motores, en otro repositorio (sabotaje del supervisor, 03-10: el valor por omisión a False salía verde)."""
    import inspect
    from matrixai import playground
    for funcion in (playground._submit_training_job, playground._run_playground_training):
        assert inspect.signature(funcion).parameters["recortar_objetivo"].default is True, funcion.__name__


def test_I1_ejemplos_para_elegir_devuelve_lo_que_dice_en_los_tres_casos():
    """La regla, mirada en lo que DEVUELVE y no solo en la época que acaba saliendo: la prueba de
    extremo a extremo de arriba solo muerde si la época cambia, y con pocas épocas recortar o no
    recortar pueden coincidir (sabotaje del supervisor, 03-10: «sin alcanzables, no recortar» salió
    VERDE en todo el fichero)."""
    from matrixai.training.dense_trainer import ejemplos_para_elegir
    fuera = [([0.5], [1.4]), ([0.6], [-0.2])]
    sel, decl = ejemplos_para_elegir(fuera, (0.0, 1.0))
    assert [y for _, y in sel] == [[1.0], [0.0]]                  # sin alcanzables: recortadas
    assert decl == {"rule": "target_clipped", "validation_rows": 2, "of": 2}
    mezcla = [([0.1], [0.3]), ([0.9], [1.7]), ([0.2], [1.0])]
    sel, decl = ejemplos_para_elegir(mezcla, (0.0, 1.0))
    assert sel == [mezcla[0], mezcla[2]]                          # solo las alcanzables, sin tocar
    assert decl == {"rule": "target_within_range", "validation_rows": 2, "of": 3}
    sel, decl = ejemplos_para_elegir(mezcla, None)
    assert sel is mezcla and decl is None                         # sin rango: lo de siempre
