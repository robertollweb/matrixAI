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

Los T3 (paridad bit a bit con el código de antes), T4 (ida y vuelta por el
Studio) y T5 por el endpoint viven en el backend del Studio.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import random
import time

import pytest

from matrixai.playground import (
    _get_job_status, _normalize_csv_with_ranges, _submit_training_job, _training_jobs)
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
    # y el objetivo SIN recortar sí llega a la validación: pierde distinto
    assert [e["validation_loss"] for e in ta["epochs"][:k]] != \
           [e["validation_loss"] for e in tb["epochs"][:k]]


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
    # sin recortar: la verdad de validación es la verdad
    sa, sb = _entrenar(a, recortar_objetivo=False, epocas=3), _entrenar(b, recortar_objetivo=False, epocas=3)
    assert [e["validation_loss"] for e in sa["epochs"]] != [e["validation_loss"] for e in sb["epochs"]]
    assert _training_jobs[sa["_job_id"]]["run_provenance"]["target_clipped"] is False


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


def test_por_omision_el_objetivo_SE_SIGUE_recortando_la_densa_de_los_motores_lo_hereda():
    """El Studio y las rutas del núcleo piden `recortar_objetivo=False` (A8); quien no lo pide —la densa de los
    motores (120-C4), que normaliza con sus rangos de train y compara `params_best` con objetivos de validación
    distintos— hereda el comportamiento de antes. Esta prueba lo fija AQUÍ: hasta hoy solo lo cazaba una prueba de
    motores, en otro repositorio (sabotaje del supervisor, 03-10: el valor por omisión a False salía verde)."""
    import inspect
    from matrixai import playground
    for funcion in (playground._submit_training_job, playground._run_playground_training):
        assert inspect.signature(funcion).parameters["recortar_objetivo"].default is True, funcion.__name__
