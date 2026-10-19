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
