"""CONTRATO 136: la verificación decide con su catálogo QUÉ compara y con qué MARGEN.

De la 3.ª pasada del 134 (N4 y N5). R3 se fiaba del paquete en las dos cosas que deciden si
compara —el `comparable` y las tolerancias que trae—, y un inventario se puede volver a sellar
sin firma: con `comparable: false` o `tolerance_abs: 1e9` en las cifras alteradas, R3 daba PASS,
`fully_checked: true` y código 0. Y un modelo FUNCTION honesto del Studio daba FAIL, porque la
verificación leía la validación (`metrics.json`) y el producto publica el evaluador sobre el
dataset entero.

1. la regla es UNA (`reproduce.comparabilidad_de_metrica`) y la usan el export y R3;
2. `comparable: false` en un paquete cuyo núcleo lo considera comparable → se compara → FAIL;
3. otro margen declarado → no comparable CON SU MOTIVO (es/en), INCOMPARABLE, nunca PASS;
4. un paquete honesto da el mismo veredicto que con el 134;
5. FUNCTION, con la misma función y las mismas filas que el producto: PASS, nunca FAIL; y lo
   que no se puede medir, INCOMPARABLE.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest. Los rangos son de DOMINIO, nunca [0, 1].
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from matrixai.export.reproduce import comparabilidad_de_metrica
from matrixai.export.verify import verify_package

import test_c134_r3_con_los_datos_del_producto as base  # noqa: E402  (helpers de paquetes)

_SHA = "a" * 64


def _entrada(**kw):
    e = {"name": "mae", "value": 3.0, "split": "full", "dataset_sha256": _SHA,
         "direction": "lower_is_better"}
    e.update(kw)
    return e


# ── 1. la regla ─────────────────────────────────────────────────────────────────────────────────

def test_la_regla_rellena_el_margen_del_catalogo_y_acepta_el_que_coincide():
    d = comparabilidad_de_metrica(_entrada())
    assert d["comparable"] and d["margen_ajeno"] is None
    assert d["entry"]["tolerance_abs"] == 0.0
    assert d["entry"]["tolerance_scope"] == "same_environment_same_seed"
    # Lo declarado que ES el del catálogo no estorba, con o sin alcance.
    assert comparabilidad_de_metrica(_entrada(tolerance_abs=0.0))["comparable"]
    assert comparabilidad_de_metrica(_entrada(
        tolerance_abs=0.0, tolerance_scope="same_environment_same_seed"))["comparable"]


@pytest.mark.parametrize("otro", [
    {"tolerance_abs": 1e9}, {"tolerance_abs": 0.05}, {"tolerance_rel": 0.5},
    {"tolerance_abs": False}, {"tolerance_abs": "0.0"},
    {"tolerance_abs": 0.0, "tolerance_scope": "cualquier_sitio"},
])
def test_otro_margen_declarado_deja_la_metrica_no_comparable_y_lo_dice(otro):
    d = comparabilidad_de_metrica(_entrada(**otro))
    assert d["comparable"] is False
    assert d["margen_ajeno"] is not None and "tolerance_abs" in d["margen_ajeno"]


def test_una_metrica_que_el_catalogo_no_conoce_sigue_como_hoy():
    # Sin dirección no se compara; con la dirección y su margen declarados, como antes.
    sin = comparabilidad_de_metrica(_entrada(name="kappa", direction=None))
    assert sin["comparable"] is False and "direction" in sin["faltan"]
    con = comparabilidad_de_metrica(_entrada(name="kappa", tolerance_abs=0.01))
    assert con["comparable"] is True and con["margen_ajeno"] is None


def test_el_export_y_el_verificador_usan_la_MISMA_funcion():
    """Una sola regla: si se cambia, cambian los dos. Un doble que lo niega todo lo prueba."""
    from matrixai.export import reproduce as r
    nula = {"entry": {}, "comparable": False, "faltan": ["x"], "margen_ajeno": None}
    with mock.patch.object(r, "comparabilidad_de_metrica", lambda e: dict(nula, entry=dict(e))):
        m = r._normalize_metrics([_entrada()])[0]
        assert m["comparable"] is False
        inf = base_r3(_manifiesto([_entrada()]), {"mae": 3.0})
    assert inf["status"] == "INCOMPARABLE", inf


# ── 2. R3 ───────────────────────────────────────────────────────────────────────────────────────

def _manifiesto(metricas):
    from matrixai.export.reproduce import _normalize_metrics
    return {"metrics": _normalize_metrics(metricas)}


def base_r3(manifiesto, obtenidas, *, locale="es", **extra):
    from matrixai.export.verify import _verificar_r3
    # El entorno que la tolerancia pide es el de aquí.
    from matrixai.export.reproduce import build_environment
    manifiesto = dict(manifiesto, environment={
        "environment_sha256": build_environment().get("environment_sha256")})
    return _verificar_r3(manifiesto, {"status": "PASS", "metrics": obtenidas, **extra}, locale=locale)


def _con(m, **kw):
    """Lo que hace quien vuelve a sellar el inventario: toca el manifiesto ya exportado."""
    m = json.loads(json.dumps(m))
    for x in m["metrics"]:
        x.update(kw)
    return m


def test_la_mentira_gorda_con_comparable_false_da_FAIL():
    honesto = _manifiesto([_entrada(value=316.7), _entrada(name="r2", value=0.9, direction=None)])
    mentira = _con(honesto, comparable=False)
    assert all(x["comparable"] is False for x in mentira["metrics"])
    # El núcleo la considera comparable: se compara con lo que se obtuvo, y no cuadra.
    r3 = base_r3(mentira, {"mae": 0.18, "r2": -0.23})
    assert r3["status"] == "FAIL", r3
    assert sorted(m["metric"] for m in r3["metrics"]) == ["mae", "r2"]


@pytest.mark.parametrize("locale,frase", [
    ("es", "declara un margen que no es el medido por el núcleo (mae)"),
    ("en", "declares a margin that is not the one measured by the core (mae)"),
])
def test_con_tolerance_abs_1e9_es_INCOMPARABLE_con_su_motivo_y_nunca_PASS(locale, frase):
    m = _manifiesto([_entrada(value=316.7), _entrada(name="rmse", value=9.0, direction=None)])
    m = _con(m, tolerance_abs=1e9)
    r3 = base_r3(m, {"mae": 0.18, "rmse": 0.2}, locale=locale)
    assert r3["status"] == "INCOMPARABLE", r3
    assert "mae" in r3["margin_not_catalog"]
    assert r3["status"] != "PASS"
    one = base_r3(_con(_manifiesto([_entrada(value=316.7)]), tolerance_abs=1e9), {"mae": 0.18},
                  locale=locale)
    assert one["status"] == "INCOMPARABLE" and frase in one["reason"], one


def test_el_margen_ajeno_de_una_metrica_no_tapa_a_las_otras_y_un_FAIL_sigue_mandando():
    m = _manifiesto([_entrada(name="accuracy", value=0.9, direction=None, split="validation"),
                     _entrada(value=3.0)])
    m["metrics"][0]["tolerance_abs"] = 0.5      # accuracy con otro margen
    # mae cuadra → no basta para PASS: la otra no se contrastó
    r3 = base_r3(m, {"accuracy": 0.9, "mae": 3.0})
    assert r3["status"] == "INCOMPARABLE" and r3["checked"] == ["mae"], r3
    assert r3["margin_not_catalog"] == ["accuracy"]
    # mae NO cuadra → FAIL manda, y lo ajeno se cita
    r3 = base_r3(m, {"accuracy": 0.9, "mae": 3.5})
    assert r3["status"] == "FAIL" and r3["margin_not_catalog"] == ["accuracy"], r3


def test_sin_margen_en_el_paquete_manda_el_del_catalogo_CON_SU_ALCANCE():
    """Un paquete al que se le quitó el margen (o uno viejo) se compara con el del catálogo, y el
    alcance del catálogo se exige: en otro entorno es INCOMPARABLE, no un FAIL contra un margen que
    allí no se midió."""
    m = _con(_manifiesto([_entrada(value=3.0)]), tolerance_abs=None, tolerance_rel=None)
    for x in m["metrics"]:
        x.pop("tolerance_scope", None)
    # En el entorno del paquete: se compara con 0,0.
    assert base_r3(m, {"mae": 3.0})["status"] == "PASS"
    assert base_r3(m, {"mae": 3.5})["status"] == "FAIL"
    # En otro: INCOMPARABLE, con el motivo del entorno.
    from matrixai.export.verify import _verificar_r3
    otro = dict(m, environment={"environment_sha256": "f" * 64})
    r3 = _verificar_r3(otro, {"status": "PASS", "metrics": {"mae": 3.5}}, locale="en")
    assert r3["status"] == "INCOMPARABLE", r3


def test_un_paquete_honesto_da_el_mismo_veredicto_que_con_el_134():
    m = _manifiesto([_entrada(value=3.0), _entrada(name="r2", value=0.5, direction=None)])
    r3 = base_r3(m, {"mae": 3.0, "r2": 0.5})
    assert r3["status"] == "PASS" and sorted(r3["checked"]) == ["mae", "r2"] and \
        "margin_not_catalog" not in r3, r3
    assert base_r3(m, {"mae": 3.1, "r2": 0.5})["status"] == "FAIL"


# ── 3. por el camino del producto, con el paquete re-sellado ────────────────────────────────────

def test_regresion_con_rangos_de_dominio_la_mentira_con_margen_o_comparable_no_da_PASS():
    job = base._job_regresion("stdlib")

    def gorda_y(**kw):
        def tocar(m):
            base._mentira_gorda(m)
            for x in m["metrics"]:
                if x["name"] in ("mae", "rmse", "r2"):
                    x.update(kw)
        return tocar

    for kw, esperado in (({"comparable": False}, "FAIL"),
                         ({"tolerance_abs": 1e9}, "INCOMPARABLE")):
        p = base._paquete(job, base._MXAI_REG, base._MXTRAIN_REG, base._RECETA_REG,
                          tocar=gorda_y(**kw))
        inf = verify_package(p, run_training=True)
        r3 = inf["stages"]["R3"]
        assert r3["status"] == esperado, (kw, r3)
        assert inf["exit_code"] != 0 and inf["fully_checked"] is False, inf


# ── 4. FUNCTION (N4): la misma función y las mismas filas que el producto ───────────────────────

_MXAI_FN = """PROJECT CelsiusToKelvin

VECTOR Reading[1]
  celsius: Scalar[-50, 150]
END

PARAM W1 Vector[1]
END

PARAM b1 Scalar
END

FUNCTION PredictedKelvinModel
  predicted_kelvin: Scalar = linear(W1 * Reading + b1)
END

GRAPH
  Reading -> PredictedKelvinModel
END
"""
_MXTRAIN_FN = """MODEL model.mxai

DATASET KelvinData
  SOURCE csv("kelvin.csv")
  INPUT Reading FROM COLUMNS [celsius]
  TARGET predicted_kelvin: Scalar[200, 450]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END

LOSS KelvinLoss
  TYPE mse
  PREDICTION predicted_kelvin
  TARGET predicted_kelvin
END

OPTIMIZER KelvinOptimizer
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE W1, b1
END

RUN
  EPOCHS 60
END
"""
_RECETA_FN = "kelvin = celsius + 273.15\n"
_RANGOS_FN = {"celsius": (-50.0, 150.0)}
_JOB_FN: dict = {}


def _job_function() -> dict:
    if "j" not in _JOB_FN:
        csv = base._generar(_MXAI_FN, _MXTRAIN_FN, _RANGOS_FN, _RECETA_FN, filas=120,
                            target_range=(200.0, 450.0))
        _JOB_FN["j"] = base._entrenar_por_el_producto(
            _MXAI_FN, _MXTRAIN_FN, csv, _RECETA_FN, motor="stdlib", rangos=_RANGOS_FN,
            target_range=(200.0, 450.0), recortar=False)
    return _JOB_FN["j"]


def _paquete_function(*, como: str, alterar: str | None = None) -> Path:
    """`como='studio'`: las cifras del evaluador, `split: full` (lo que publica el Studio).
    `como='cli'`: las del `metrics.json` del entrenador, `split: validation` (lo que publica la CLI)."""
    job = _job_function()
    if como == "studio":
        return base._paquete(job, _MXAI_FN, _MXTRAIN_FN, _RECETA_FN, alterar=alterar)

    def a_validacion(m):
        disco = job["result"]["metrics"]           # el `metrics.json` que escribió el entrenador
        for x in m["metrics"]:
            if x["name"] in ("mae", "r2") and isinstance(disco.get(x["name"]), (int, float)):
                x["value"] = disco[x["name"]] + (0.05 if x["name"] == alterar else 0.0)
                x["split"] = "validation"
    return base._paquete(job, _MXAI_FN, _MXTRAIN_FN, _RECETA_FN, tocar=a_validacion)


def test_FUNCTION_publicado_por_el_Studio_da_PASS_y_no_FAIL():
    res = _job_function()["result"]
    assert res["metric_splits"]["mae"] == "full"
    assert res["mae"] > 100, res["mae"]            # en la escala del dominio (200..450), no 0..1
    inf = verify_package(_paquete_function(como="studio"), run_training=True)
    assert inf["stages"]["training"]["status"] == "PASS", inf["stages"]["training"]
    r3 = inf["stages"]["R3"]
    assert r3["status"] == "PASS", r3
    assert {"mae", "rmse", "r2"} <= set(r3["checked"]), r3
    obtenidas = inf["stages"]["training"]["metrics_by_split"]["full"]
    assert {n: obtenidas[n] for n in ("mae", "rmse", "r2")} == {n: res[n] for n in ("mae", "rmse", "r2")}


def test_FUNCTION_publicado_por_la_CLI_sigue_dando_el_mismo_veredicto():
    """La CLI publica el `metrics.json` (validación): el verificador compara con SU partición."""
    inf = verify_package(_paquete_function(como="cli"), run_training=True)
    r3 = inf["stages"]["R3"]
    assert r3["status"] == "PASS", r3
    assert {"mae", "r2"} <= set(r3["checked"]), r3


@pytest.mark.parametrize("como", ["studio", "cli"])
def test_FUNCTION_con_una_cifra_alterada_da_FAIL(como):
    r3 = verify_package(_paquete_function(como=como, alterar="mae"),
                        run_training=True)["stages"]["R3"]
    assert r3["status"] == "FAIL" and [m["metric"] for m in r3["metrics"]] == ["mae"], r3


def test_FUNCTION_si_no_se_pudo_medir_como_el_producto_es_INCOMPARABLE_nunca_FAIL():
    from matrixai.export import verify as v
    with mock.patch.object(v, "_metricas_de_function_como_el_producto", lambda *a, **k: {}):
        inf = verify_package(_paquete_function(como="studio"), run_training=True)
    r3 = inf["stages"]["R3"]
    assert r3["status"] == "INCOMPARABLE", r3
    assert {"mae", "rmse", "r2"} <= set(r3["not_reported"]), r3


# ── 5. AUDITORÍA DEL 136: con UNA clave tocada en la cifra alterada, R3 ya no da PASS ───────────
#
# Medido por la auditoría (`~/.agentes/136/auditoria.md`, H1-H3): con `split: null`,
# `dataset_sha256: null`, `direction: null` o `tolerance_abs: null` en las cifras alteradas, el
# paquete re-sellado salía R3 PASS, `fully_checked: true` y código 0, comparando solo las dos
# pérdidas (el 134 daba FAIL). La regla repone lo que el catálogo sabe (H2, H3) y R3 NOMBRA toda
# cifra publicada que no compara, y entonces no dice PASS (H1).

def _con_en(m, nombres, **kw):
    """Como `_con`, pero solo en las cifras `nombres`: las que mienten."""
    m = json.loads(json.dumps(m))
    for x in m["metrics"]:
        if x["name"] in nombres:
            x.update(kw)
    return m


def _tres():
    """mae y rmse (las que mienten) y una pérdida que cuadra, como en la regresión del Studio."""
    return _manifiesto([_entrada(value=316.7), _entrada(name="rmse", value=400.0, direction=None),
                        _entrada(name="final_train_loss", value=0.4, split="train", direction=None)])


_LA_MENTIRA = {"mae": 31.67, "rmse": 40.0, "final_train_loss": 0.4}


@pytest.mark.parametrize("clave", ["split", "dataset_sha256"])
@pytest.mark.parametrize("locale,frase", [
    ("es", "el paquete publica cifras que no se pueden comparar"),
    ("en", "the package publishes figures that cannot be compared"),
])
def test_una_cifra_que_la_regla_no_compara_se_NOMBRA_y_R3_no_da_PASS(clave, locale, frase):
    m = _con_en(_tres(), ("mae", "rmse"), **{clave: None})
    r3 = base_r3(m, dict(_LA_MENTIRA), locale=locale)
    assert r3["status"] == "INCOMPARABLE", r3
    assert r3["checked"] == ["final_train_loss"], r3
    assert r3["not_compared"] == [{"name": "mae", "missing": [clave]},
                                  {"name": "rmse", "missing": [clave]}], r3
    assert frase in r3["reason"] and f"mae ({clave})" in r3["reason"], r3["reason"]


def test_con_cifras_sin_comparar_un_FAIL_sigue_mandando_y_las_cita():
    m = _con_en(_tres(), ("mae", "rmse"), split=None)
    r3 = base_r3(m, dict(_LA_MENTIRA, final_train_loss=0.9))
    assert r3["status"] == "FAIL", r3
    assert [x["metric"] for x in r3["metrics"]] == ["final_train_loss"], r3
    assert [x["name"] for x in r3["not_compared"]] == ["mae", "rmse"], r3


def test_tambien_se_nombran_un_nombre_fuera_del_catalogo_sin_direccion_y_lo_que_no_es_una_metrica():
    """La fila «nombre desconocido + `direction: null`» de la auditoría daba PASS."""
    m = _tres()
    for x in m["metrics"]:
        if x["name"] in ("mae", "rmse"):
            x["name"] += "_ajustada"
            x["direction"] = None
    m["metrics"].append("mae: 31.67")
    r3 = base_r3(m, dict(_LA_MENTIRA))
    assert r3["status"] == "INCOMPARABLE", r3
    assert r3["checked"] == ["final_train_loss"], r3
    assert r3["not_compared"] == [{"name": "mae_ajustada", "missing": ["direction"]},
                                  {"name": "rmse_ajustada", "missing": ["direction"]},
                                  {"name": "metrics[3]", "missing": ["object"]}], r3


def test_direction_null_en_una_cifra_del_catalogo_se_repone_y_se_compara():
    """H2: la dirección es del NOMBRE; la regla la repone como el export, y la cifra se compara."""
    m = _con(_manifiesto([_entrada(value=316.7)]), direction=None)
    d = comparabilidad_de_metrica(m["metrics"][0])
    assert d["comparable"] and d["entry"]["direction"] == "lower_is_better", d
    assert base_r3(m, {"mae": 31.67})["status"] == "FAIL"
    assert base_r3(m, {"mae": 316.7})["status"] == "PASS"


def test_sin_tolerancia_y_con_el_alcance_del_catalogo_manda_el_margen_del_catalogo():
    """H3: lo natural es quitar SOLO `tolerance_abs`; el alcance que trae todo paquete honesto se
    queda. Antes la cifra se quedaba sin margen y desaparecía en silencio."""
    m = _con(_manifiesto([_entrada(value=316.7)]), tolerance_abs=None)
    assert m["metrics"][0]["tolerance_scope"] == "same_environment_same_seed"
    d = comparabilidad_de_metrica(m["metrics"][0])
    assert d["comparable"] and d["entry"]["tolerance_abs"] == 0.0, d
    assert base_r3(m, {"mae": 31.67})["status"] == "FAIL"
    assert base_r3(m, {"mae": 316.7})["status"] == "PASS"


@pytest.mark.parametrize("locale,frase", [
    ("es", "declara un margen que no es el medido por el núcleo (mae)"),
    ("en", "declares a margin that is not the one measured by the core (mae)"),
])
def test_sin_tolerancia_y_con_un_alcance_ajeno_es_margen_ajeno_y_el_motivo_lo_dice(locale, frase):
    """H3, el otro lado: un alcance que el catálogo no mide, SIN tolerancia, es un margen ajeno (con
    `tolerance_abs: 0.0` ya lo era). Y en un INCOMPARABLE MIXTO -otra cifra sí cuadra- el motivo
    lo dice en la prosa, no solo en `margin_not_catalog` (M3, A4)."""
    m = _manifiesto([_entrada(value=316.7), _entrada(name="r2", value=0.5, direction=None)])
    m = _con_en(m, ("mae",), tolerance_abs=None, tolerance_scope="any_environment")
    assert comparabilidad_de_metrica(m["metrics"][0])["margen_ajeno"] is not None
    r3 = base_r3(m, {"mae": 31.67, "r2": 0.5}, locale=locale)
    assert r3["status"] == "INCOMPARABLE", r3
    assert r3["checked"] == ["r2"] and r3["margin_not_catalog"] == ["mae"], r3
    assert frase in r3["reason"], r3["reason"]


@pytest.mark.parametrize("raw", [
    _entrada(direction=None),
    _entrada(name="accuracy", value=0.8, direction=None, tolerance_scope="same_environment_same_seed"),
    _entrada(tolerance_scope="any_environment"),
    _entrada(tolerance_abs=0.0, tolerance_scope="any_environment"),
    _entrada(split=None),
    _entrada(dataset_sha256=None),
    _entrada(name="kappa", direction=None, tolerance_abs=0.05),
    _entrada(name="kappa", tolerance_abs=0.05),
], ids=["sin_direction", "sin_tolerancia_con_su_alcance", "alcance_ajeno", "alcance_ajeno_con_0",
        "sin_split", "sin_digest", "kappa_sin_direction", "kappa_completa"])
def test_el_export_y_R3_deciden_LO_MISMO_de_cada_cifra(raw):
    """«La MISMA regla» (§2.1): lo que el export escribe en `comparable` es lo que R3 decide, tanto
    sobre lo que el llamante mandó como sobre lo ya escrito. Antes el export reponía la dirección
    ANTES de llamar a la regla y R3 no (H2): `sin_direction` salía comparable al exportar y no al
    verificar."""
    from matrixai.export.reproduce import _normalize_metrics
    escrita = _normalize_metrics([raw])[0]
    assert comparabilidad_de_metrica(raw)["comparable"] is escrita["comparable"], escrita
    assert comparabilidad_de_metrica(escrita)["comparable"] is escrita["comparable"], escrita


@pytest.mark.parametrize("tolerancia", [{"tolerance_abs": 0.0}, {}], ids=["con_0", "sin_tolerancia"])
def test_el_export_no_pisa_el_alcance_que_declaro_el_llamante(tolerancia):
    """M3, A5: el alcance del llamante tiene que llegar a la regla. Si no, uno ajeno con
    `tolerance_abs: 0.0` saldría comparable y con el alcance del catálogo, pisando el suyo."""
    from matrixai.export.reproduce import _normalize_metrics
    escrita = _normalize_metrics([_entrada(tolerance_scope="matriz de entornos v1", **tolerancia)])[0]
    assert escrita["tolerance_scope"] == "matriz de entornos v1", escrita
    assert escrita["comparable"] is False, escrita


@pytest.mark.parametrize("locale,frase", [
    ("es", "publica cifras medidas sobre otro dataset que el suyo (mae, rmse)"),
    ("en", "publishes figures measured on a dataset other than its own (mae, rmse)"),
])
def test_una_cifra_medida_sobre_OTRO_dataset_no_se_compara_y_se_nombra(locale, frase):
    """M6 (deuda anterior al 136): la cifra dice sobre qué datos se midió, y R3 no lo miraba. Si no
    es ninguno de los dos digests del paquete (el crudo o el preparado: el producto declara uno u
    otro), no se compara: ni PASS con un valor honesto re-etiquetado, ni FAIL contra un
    reentrenamiento sobre otros datos. Con el crudo o el preparado, se compara como siempre."""
    m = _tres()
    m["artifacts"] = {"dataset": {"sha256": "c" * 64, "sha256_prepared": _SHA}}
    honestas = {"mae": 316.7, "rmse": 400.0, "final_train_loss": 0.4}
    crudo = _con_en(m, ("rmse",), dataset_sha256="c" * 64)
    assert base_r3(crudo, dict(honestas))["status"] == "PASS"
    assert base_r3(crudo, dict(_LA_MENTIRA))["status"] == "FAIL"
    ajeno = _con_en(m, ("mae", "rmse"), dataset_sha256="d" * 64)
    for obtenidas in (honestas, _LA_MENTIRA):
        r3 = base_r3(ajeno, dict(obtenidas), locale=locale)
        assert r3["status"] == "INCOMPARABLE", r3
        assert r3["other_dataset"] == ["mae", "rmse"] and r3["checked"] == ["final_train_loss"], r3
        assert frase in r3["reason"], r3["reason"]


# Por el camino del producto: el paquete del Studio (su captura, sus cifras y particiones, con
# rangos de dominio), la mentira gorda en las cifras de la tarea y UNA clave tocada, re-sellado.
_UNA_CLAVE = [
    ({"split": None}, "INCOMPARABLE"),
    ({"dataset_sha256": None}, "INCOMPARABLE"),
    ({"direction": None}, "FAIL"),
    ({"tolerance_abs": None}, "FAIL"),
    ({"tolerance_abs": None, "tolerance_scope": "any_environment"}, "INCOMPARABLE"),
    ({"dataset_sha256": "c" * 64}, "INCOMPARABLE"),                       # M6
]
_MIENTEN = {"reg": ("mae", "rmse", "r2"), "clasif": ("accuracy", "macro_f1")}


def _mentira_de(tarea, **kw):
    def tocar(m):
        for x in m["metrics"]:
            if x["name"] not in _MIENTEN[tarea]:
                continue
            if x["name"] in ("mae", "rmse"):
                x["value"] = x["value"] / 10
            else:
                x["value"] = 0.9999 if x["name"] == "r2" else 0.99
            x.update(kw)
    return tocar


@pytest.mark.parametrize("tarea,motor", [("reg", "stdlib"), ("reg", "torch"), ("clasif", "torch")])
def test_por_el_camino_del_producto_UNA_clave_en_la_cifra_alterada_no_da_PASS(tarea, motor):
    if tarea == "reg":
        job, textos = base._job_regresion(motor), (base._MXAI_REG, base._MXTRAIN_REG, base._RECETA_REG)
    else:
        job, textos = base._job_clasificacion(motor), (base._MXAI, base._MXTRAIN, base._RECETA)
    for kw, esperado in _UNA_CLAVE:
        inf = verify_package(base._paquete(job, *textos, tocar=_mentira_de(tarea, **kw)),
                             run_training=True)
        r3 = inf["stages"]["R3"]
        assert r3["status"] == esperado, (kw, r3)
        assert inf["fully_checked"] is False, (kw, inf)
        assert inf["exit_code"] == (2 if esperado == "FAIL" else 3), (kw, inf["exit_code"])
        if esperado == "FAIL":
            assert sorted(x["metric"] for x in r3["metrics"]) == sorted(_MIENTEN[tarea]), (kw, r3)
        else:
            nombradas = ({x["name"] for x in r3.get("not_compared") or []}
                         | set(r3.get("margin_not_catalog") or []) | set(r3.get("other_dataset") or []))
            assert nombradas == set(_MIENTEN[tarea]), (kw, r3)
            assert not set(r3["checked"]) & set(_MIENTEN[tarea]), (kw, r3)
