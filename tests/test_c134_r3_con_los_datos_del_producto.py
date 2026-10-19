"""CONTRATO 134, AUDITORÍA H1 (BLOQUEANTE): el reentrenamiento prepara los datos COMO EL PRODUCTO.

Medido el 19-10 por el camino real del Studio (`_studio_train_start` → bundle → taller limpio
→ otro proceso): con rangos de dominio (`credit_score` [300, 850], `age` [18, 90]) R3 daba FAIL
a un paquete honesto, con stdlib y con torch —exactitud 0,583 publicada y 0,083 reentrenada,
pérdida 0,98 frente a 17,7—, porque la verificación entrenaba el CSV regenerado TAL CUAL y el
producto lo había normalizado antes. Los `[0, 1]` de las pruebas anteriores son la identidad y
no lo veían.

Aquí los paquetes salen de la entrada real del producto (`playground._submit_training_job`, la
que usa el Studio), con SU captura y SUS cifras, y con las formas de datos de verdad:

1. rangos de dominio → R3 PASS con stdlib y con torch, y una cifra alterada → FAIL;
2. el CSV preparado que no cuadra con su `sha256_prepared`, o un paquete con rangos que no lo
   declara → INCOMPARABLE con su motivo, nunca FAIL;
3. una regresión con `target_range` y el objetivo SIN recortar (el Studio entrena así), con
   objetivos fuera del rango: la regla de elegir la época se aplica al reentrenar → PASS;
4. R1 con el rango del objetivo: una regresión cuyo objetivo generó el rango se regenera
   (antes R1 daba FAIL), y una generada sin él también.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from matrixai.export.verify import verify_package

torch = pytest.importorskip("torch")

# ── clasificación con rangos de dominio ─────────────────────────────────────────────────────────

_MXAI = (
    "PROJECT LoanProject\n\n"
    "VECTOR LoanApplicant[2]\n  credit_score: Scalar\n  age: Scalar\nEND\n\n"
    "NETWORK LoanRiskClassifier\n  INPUT LoanApplicant\n"
    "  LAYER Dense units=16 activation=relu\n"
    "  LAYER Dense units=3 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[low, mid, high]\nEND\n\n"
    "GRAPH\n  LoanApplicant -> LoanRiskClassifier\nEND\n"
)
_MXTRAIN = (
    "MODEL model.mxai\n\nDATASET LoanDataset\n"
    "  SOURCE csv(\"loan.train.csv\")\n"
    "  INPUT LoanApplicant FROM COLUMNS [credit_score, age]\n"
    "  TARGET predicted_class: Label[low, mid, high]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS LoanLoss\n  TYPE cross_entropy\n  PREDICTION LoanRiskClassifier\n"
    "  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER LoanOptimizer\n  TYPE sgd\n  LEARNING_RATE 0.05\n"
    "  UPDATE LoanRiskClassifier.*\nEND\n\nRUN\n  EPOCHS 4\nEND\n"
)
#: La de `camino.py` de la auditoría: umbrales en la escala del dominio.
_RECETA = "high: age > 60\nlow: credit_score < 400\nDEFAULT: mid"
_RANGOS = {"credit_score": (300.0, 850.0), "age": (18.0, 90.0)}

# ── regresión con el rango de la salida ─────────────────────────────────────────────────────────

_MXAI_REG = (
    "PROJECT Casa\n\nVECTOR Vivienda[2]\n  superficie_m2: Scalar\n  antiguedad: Scalar\nEND\n\n"
    "NETWORK Precio\n  INPUT Vivienda\n  LAYER Dense units=16 activation=relu\n"
    "  LAYER Dense units=1 activation=linear\n  OUTPUT precio_eur: Scalar\nEND\n\n"
    "GRAPH\n  Vivienda -> Precio\nEND\n"
)
_MXTRAIN_REG = (
    "MODEL model.mxai\n\nDATASET Casas\n  SOURCE csv(\"casas.train.csv\")\n"
    "  INPUT Vivienda FROM COLUMNS [superficie_m2, antiguedad]\n  TARGET precio_eur: Scalar\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE mse\n  PREDICTION Precio\n  TARGET precio_eur\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.01\n  UPDATE Precio.*\nEND\n\nRUN\n  EPOCHS 12\nEND\n"
)
_RANGOS_REG = {"superficie_m2": (30.0, 400.0), "antiguedad": (0.0, 90.0)}
_RECETA_REG = "y = 2000*superficie_m2 - 500*antiguedad + 60000\nNOISE: 0.05"
_RANGO_SALIDA = (60000.0, 900000.0)

_VERIFICADA = {"verified": True, "code": "regenera_el_dataset"}


def _generar(mxai: str, mxtrain: str, rangos: dict, receta: str, filas: int = 60,
             target_range: tuple[float, float] | None = None) -> str:
    from matrixai.playground_api import generate_synthetic_dataset
    g = generate_synthetic_dataset(mxai, mxtrain, filas, 4242, "coherent", False,
                                   field_ranges_override=rangos, recipe_text=receta,
                                   **({"target_range": target_range} if target_range else {}))
    assert g.get("ok"), g
    return g["csv_text"]


def _entrenar_por_el_producto(mxai: str, mxtrain: str, csv: str, receta: str, *, motor: str,
                              rangos: dict, target_range=None, recortar: bool = True) -> dict:
    """El trabajo del producto, por SU entrada (`_submit_training_job`, la del Studio)."""
    from matrixai import playground as pg
    with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": motor}):
        r = pg._submit_training_job(
            mxai, mxtrain, csv, None, field_ranges=rangos, seed=7, target_range=target_range,
            recipe_text=receta, dataset_seed=4242, recipe_verification=_VERIFICADA,
            dataset_mode="coherent", recortar_objetivo=recortar)
        assert r.get("ok"), r
        job = pg._training_jobs[r["job_id"]]
        tope = time.monotonic() + 300
        while job["status"] == "running" and time.monotonic() < tope:
            time.sleep(0.05)
    assert job["status"] == "done", (job["status"], job.get("error"))
    assert job["result"]["backend"] == motor, job["result"].get("backend")
    return job


def _paquete(job: dict, mxai: str, mxtrain: str, receta: str, *, alterar: str | None = None,
             tocar=None) -> Path:
    """El paquete como lo exporta el Studio: SU captura, SUS cifras y sus particiones, con el
    digest del CSV PREPARADO en cada métrica. `tocar(manifiesto)` lo edita y se re-sella."""
    from matrixai.export.reproduce import añadir_inventario_de_ficheros, write_reproduce_manifest
    from matrixai.training.metric_identity import DIRECCION
    res, cap = job["result"], job["run_provenance"]
    d = Path(tempfile.mkdtemp(prefix="c134d-"))
    (d / "model.mxai").write_text(mxai, encoding="utf-8")
    (d / "training.mxtrain").write_text(mxtrain, encoding="utf-8")
    (d / "recipe.txt").write_text(receta, encoding="utf-8")
    metricas = []
    for nombre in DIRECCION:
        valor = res.get(nombre)
        if not isinstance(valor, (int, float)) or isinstance(valor, bool):
            continue
        m = {"name": nombre, "value": valor + (0.05 if nombre == alterar else 0.0),
             "dataset_sha256": cap["dataset_sha256_prepared"]}
        if (res.get("metric_splits") or {}).get(nombre):
            m["split"] = res["metric_splits"][nombre]
        metricas.append(m)
    write_reproduce_manifest(
        d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
        dataset_sha256=cap["dataset_sha256_raw"], dataset_rows=cap["dataset_rows"],
        generation={"seeds": cap["seeds"]}, metrics=metricas, run_provenance=cap,
        weights_source="trained")
    if tocar is not None:
        m = json.loads((d / "reproduce.json").read_text(encoding="utf-8"))
        tocar(m)
        (d / "reproduce.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    añadir_inventario_de_ficheros(d)
    return d


_JOBS: dict[str, dict] = {}


def _job_clasificacion(motor: str) -> dict:
    if motor not in _JOBS:
        csv = _generar(_MXAI, _MXTRAIN, _RANGOS, _RECETA)
        _JOBS[motor] = _entrenar_por_el_producto(_MXAI, _MXTRAIN, csv, _RECETA, motor=motor,
                                                 rangos=_RANGOS)
    return _JOBS[motor]


# ── 1. rangos de dominio ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("motor", ["stdlib", "torch"])
def test_con_rangos_de_dominio_R3_da_PASS_a_un_paquete_honesto(motor):
    job = _job_clasificacion(motor)
    cap = job["run_provenance"]
    # El dato que hace falta: el preparado NO es el crudo (no son la identidad).
    assert cap["dataset_sha256_prepared"] != cap["dataset_sha256_raw"]
    inf = verify_package(_paquete(job, _MXAI, _MXTRAIN, _RECETA), run_training=True)
    assert inf["stages"]["R1"]["status"] == "PASS", inf["stages"]["R1"]
    t, r3 = inf["stages"]["training"], inf["stages"]["R3"]
    assert t["status"] == "PASS", t
    assert t["not_applied_from_capture"] == [], t
    # LO QUE SE ROMPÍA: con el CSV sin preparar, esto era FAIL (0,583 publicada, 0,083 obtenida).
    assert r3["status"] == "PASS", r3
    assert t["applied_from_capture"]["dataset_sha256_prepared"] == cap["dataset_sha256_prepared"]
    assert sorted(r3["checked"]) == sorted(["accuracy", "macro_f1", "final_train_loss",
                                            "best_validation_loss"]), r3


@pytest.mark.parametrize("motor", ["stdlib", "torch"])
def test_con_rangos_de_dominio_una_cifra_alterada_da_FAIL(motor):
    """El control de que R3 compara de verdad con los datos preparados."""
    inf = verify_package(_paquete(_job_clasificacion(motor), _MXAI, _MXTRAIN, _RECETA,
                                  alterar="accuracy"), run_training=True)
    r3 = inf["stages"]["R3"]
    assert r3["status"] == "FAIL", r3
    assert [m["metric"] for m in r3["metrics"]] == ["accuracy"]


# ── 2. el digest del preparado manda ───────────────────────────────────────────────────────────

def _otro_digest(m):
    m["artifacts"]["dataset"]["sha256_prepared"] = "0" * 64


def _sin_digest(m):
    m["artifacts"]["dataset"]["sha256_prepared"] = None


@pytest.mark.parametrize("motor", ["stdlib", "torch"])
@pytest.mark.parametrize("tocar,frase_es,frase_en", [
    (_otro_digest, "no cuadra con el sha256 del CSV", "does not match the sha256 of the CSV"),
    (_sin_digest, "no declara el sha256 del CSV preparado", "does not declare the sha256 of the prepared CSV"),
])
def test_un_preparado_que_no_cuadra_o_no_se_declara_es_INCOMPARABLE_y_nunca_FAIL(
        motor, tocar, frase_es, frase_en):
    """Con una cifra ALTERADA: si se comparara, saldría FAIL. No se compara."""
    paquete = _paquete(_job_clasificacion(motor), _MXAI, _MXTRAIN, _RECETA, alterar="accuracy",
                       tocar=tocar)
    for locale, frase in (("es", frase_es), ("en", frase_en)):
        inf = verify_package(paquete, run_training=True, locale=locale)
        assert inf["stages"]["manifest"]["status"] == "PASS", inf["stages"]["manifest"]
        t, r3 = inf["stages"]["training"], inf["stages"]["R3"]
        assert t["status"] == "PASS", t
        assert "dataset_sha256_prepared" in t["not_applied_from_capture"], t
        assert frase in t["not_applied_reasons"]["dataset_sha256_prepared"]
        assert r3["status"] == "INCOMPARABLE", r3
        assert frase in r3["reason"]


# ── 3. la regresión con el objetivo sin recortar ───────────────────────────────────────────────

_JOBS_REG: dict[str, dict] = {}


def _job_regresion_fuera(motor: str) -> dict:
    """El Studio entrena con `recortar_objetivo=False` (endpoints, A8). Con el rango de la salida
    por debajo del máximo de los datos, una parte del objetivo queda FUERA de [0, 1] y la época se
    elige solo con las filas alcanzables."""
    if motor not in _JOBS_REG:
        csv = _generar(_MXAI_REG, _MXTRAIN_REG, _RANGOS_REG, _RECETA_REG, filas=80)
        ys = sorted(float(linea.split(",")[-1]) for linea in csv.splitlines()[1:])
        rango = (ys[0], ys[int(len(ys) * 0.75)])
        _JOBS_REG[motor] = _entrenar_por_el_producto(
            _MXAI_REG, _MXTRAIN_REG, csv, _RECETA_REG, motor=motor, rangos=_RANGOS_REG,
            target_range=rango, recortar=False)
    return _JOBS_REG[motor]


@pytest.mark.parametrize("motor", ["stdlib", "torch"])
def test_una_regresion_con_el_objetivo_sin_recortar_se_reentrena_como_el_producto(motor):
    job = _job_regresion_fuera(motor)
    res = job["result"]
    # Que la regla de la época se aplicó DE VERDAD en el producto: sin esto la prueba no mide nada.
    eleccion = res.get("epoch_selection") or {}
    assert eleccion.get("validation_rows", 0) < eleccion.get("of", 0), eleccion
    assert job["run_provenance"].get("target_clipped") is False
    inf = verify_package(_paquete(job, _MXAI_REG, _MXTRAIN_REG, _RECETA_REG), run_training=True)
    t, r3 = inf["stages"]["training"], inf["stages"]["R3"]
    assert inf["stages"]["R1"]["status"] == "PASS", inf["stages"]["R1"]
    assert t["status"] == "PASS", t
    assert t["not_applied_from_capture"] == [], t
    assert r3["status"] == "PASS", r3
    assert {"final_train_loss", "best_validation_loss"} <= set(r3["checked"]), r3
    if motor == "torch":
        assert {"mae", "rmse", "r2"} <= set(r3["checked"]), r3


def test_leer_mal_si_el_objetivo_se_recorto_no_da_FAIL_lo_caza_el_digest():
    """Si el paquete dijera «recortado» habiéndose entrenado sin recortar, el preparado sería
    otro: el digest no cuadra y R3 no compara, aunque una cifra esté alterada."""
    def recortado(m):
        m["generation"].pop("target_clipped", None)
    paquete = _paquete(_job_regresion_fuera("torch"), _MXAI_REG, _MXTRAIN_REG, _RECETA_REG,
                       alterar="mae", tocar=recortado)
    inf = verify_package(paquete, run_training=True)
    assert inf["stages"]["R3"]["status"] == "INCOMPARABLE", inf["stages"]["R3"]
    assert "dataset_sha256_prepared" in inf["stages"]["training"]["not_applied_from_capture"]


# ── 4. R1 con el rango del objetivo ────────────────────────────────────────────────────────────

#: Una receta que NO es una fórmula de regresión: el objetivo lo saca el generador, y con
#: `target_range` lo saca DENTRO de él (05-10). Medido el 19-10: con y sin el rango, dos CSV.
_RECETA_SIN_FORMULA = "regenerable\n"


@pytest.mark.parametrize("con_rango_al_generar", [True, False])
def test_R1_regenera_una_regresion_con_o_sin_el_rango_del_objetivo(con_rango_al_generar):
    """Fases y el Workbench generan Y entrenan con el rango de la salida; la clásica genera sin
    él y entrena con él (su código no se toca). El paquete solo declara el de entrenar: R1 prueba
    los dos y decide el sha256 completo."""
    rango_entrenar = _RANGO_SALIDA if con_rango_al_generar else (-1.5, 1.5)
    csv = _generar(_MXAI_REG, _MXTRAIN_REG, _RANGOS_REG, _RECETA_SIN_FORMULA, filas=60,
                   target_range=_RANGO_SALIDA if con_rango_al_generar else None)
    sin = _generar(_MXAI_REG, _MXTRAIN_REG, _RANGOS_REG, _RECETA_SIN_FORMULA, filas=60)
    con = _generar(_MXAI_REG, _MXTRAIN_REG, _RANGOS_REG, _RECETA_SIN_FORMULA, filas=60,
                   target_range=rango_entrenar)
    assert hashlib.sha256(sin.encode()).hexdigest() != hashlib.sha256(con.encode()).hexdigest(), \
        "con y sin el rango salen el mismo CSV: la prueba no mide nada"
    job = _entrenar_por_el_producto(_MXAI_REG, _MXTRAIN_REG, csv, _RECETA_SIN_FORMULA,
                                    motor="stdlib", rangos=_RANGOS_REG, target_range=rango_entrenar,
                                    recortar=False)
    inf = verify_package(_paquete(job, _MXAI_REG, _MXTRAIN_REG, _RECETA_SIN_FORMULA),
                         run_training=False)
    r1 = inf["stages"]["R1"]
    assert r1["status"] == "PASS", r1
    assert r1["target_range_applied"] is con_rango_al_generar, r1
