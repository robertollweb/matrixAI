# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 59 C1 — normalización del target: un modelo de regresión desde
CSV debe APRENDER de verdad, sea cual sea la escala del target. Antes de
este corte, un target en escala de dominio (p.ej. 273-372 Kelvin) hacía
explotar el MSE con los defaults de entrenamiento y la red colapsaba a
predecir la media (ver 59_REGRESION_QUE_APRENDE_CONTRACT.md, caso real de
Roberto: centigrados -> Kelvin, R²=-0.0001 antes del fix).

Test de cierre del contrato (C1): el CSV centigrados/Kelvin de 100 filas
alcanza R² >= 0.99 con los defaults de entrenamiento — sin tocar
arquitectura, learning rate ni épocas."""
from __future__ import annotations

import time
import unittest

from matrixai.playground import _get_job_status, _submit_training_job
from matrixai.training.dataset_project import generate_project_from_dataset


def _kelvin_csv(n: int = 100) -> str:
    """Caso real de Roberto: centigrados -> Kelvin, relación lineal exacta
    (y = x + 273.15). `centigrados` es un entero 0..n-1 casi-único, así que
    dispara la heurística de identificador (hallazgo de C2, no de C1) — los
    overrides son el escape manual ya existente, no parte de este fix."""
    lines = ["centigrados,prediccionKelvin"]
    for c in range(n):
        lines.append(f"{c},{c + 273.15}")
    return "\n".join(lines) + "\n"


def _kelvin_overrides() -> dict:
    return {
        "column_type_overrides": {"centigrados": "number"},
        "column_range_overrides": {"centigrados": (0.0, 99.0)},
    }


def _kelvin_with_categorical_csv(n: int = 100) -> str:
    """Mismo caso Kelvin + una categórica de 15 valores (> _ONEHOT_MAX=12)
    para forzar la ruta composite/embedding — el fix de C1 debe cubrir esa
    ruta también, no solo la densa plana."""
    cats = [f"dev{i}" for i in range(15)]
    lines = ["centigrados,dispositivo,prediccionKelvin"]
    for c in range(n):
        lines.append(f"{c}.5,{cats[c % 15]},{round(c + 0.5 + 273.15, 2)}")
    return "\n".join(lines) + "\n"


def _classification_csv(n: int = 60) -> str:
    lines = ["a,b,y"]
    for i in range(n):
        a, b = (i * 0.37) % 10, (i * 0.53) % 10
        lines.append(f"{a:.4f},{b:.4f},{'pos' if a + b > 10 else 'neg'}")
    return "\n".join(lines) + "\n"


def _submit_and_wait(proj: dict, **kwargs) -> dict:
    submitted = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"],
        field_ranges=proj.get("field_ranges"), **kwargs,
    )
    assert submitted.get("ok"), submitted.get("error")
    job_id = submitted["job_id"]
    status: dict = {}
    for _ in range(300):
        status = _get_job_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert status["status"] == "done", status.get("error")
    return status


def _submit_and_wait_allowing_error(proj: dict, **kwargs) -> dict:
    """Como `_submit_and_wait` pero SIN exigir que el job termine bien.

    Solo la usa el lado «sin normalizar el target», donde el desenlace
    esperado es precisamente que el entrenamiento se vaya al garete — y
    eso a veces es un colapso con números y a veces un desbordamiento
    numérico que deja el job en `error`. Ver el docstring de
    `test_composite_route_avoids_catastrophic_collapse_with_normalized_target`
    para los nueve casos medidos. El lado normalizado sigue usando
    `_submit_and_wait`, que sí exige `done`.
    """
    submitted = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"],
        field_ranges=proj.get("field_ranges"), **kwargs,
    )
    assert submitted.get("ok"), submitted.get("error")
    status: dict = {}
    for _ in range(300):
        status = _get_job_status(submitted["job_id"])
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert status["status"] in ("done", "error"), status
    return status


class TestKelvinRegressionLearns(unittest.TestCase):
    """El test de cierre literal del contrato: R² >= 0.99 para una relación
    lineal perfecta, con los defaults de entrenamiento (sin tocar lr/épocas/
    arquitectura — eso es C3, no C1)."""

    def test_kelvin_csv_reaches_r2_at_least_0_99(self):
        proj = generate_project_from_dataset(
            _kelvin_csv(), "prediccionKelvin", **_kelvin_overrides(),
        )
        self.assertIsNotNone(proj.get("target_range"))
        status = _submit_and_wait(proj, target_range=tuple(proj["target_range"]))
        self.assertEqual(status["task_kind"], "regression")
        self.assertFalse(status.get("model_collapsed"), "el modelo no debería colapsar a predecir la media")
        self.assertGreaterEqual(status["r2"], 0.99, f"R²={status['r2']} — el modelo no aprendió la relación lineal")

    def test_target_range_echoed_in_job_result(self):
        """Procedencia auditable (decisión A del contrato): el rango usado
        para normalizar viaja en el resultado del job, no se recalcula
        ad-hoc en otro punto (p.ej. `_studio_infer`)."""
        proj = generate_project_from_dataset(
            _kelvin_csv(), "prediccionKelvin", **_kelvin_overrides(),
        )
        tr = tuple(proj["target_range"])
        status = _submit_and_wait(proj, target_range=tr)
        self.assertEqual(tuple(status["target_range"]), tr)

    def test_mae_rmse_are_in_domain_scale_not_normalized_space(self):
        """Si MAE/RMSE se quedaran en espacio normalizado [0,1] en vez de
        reescalarse a la unidad real, saldrían minúsculos (¡el rango del
        target es ~119 K!) — un MAE de 0.0004 "parece" una cifra sólida pero
        es la unidad equivocada (justo el problema que 58.1 quería evitar).
        Un modelo que aprendió bien esta relación lineal debe tener un MAE
        de una fracción de grado Kelvin, nunca micro-unidades ni cientos."""
        proj = generate_project_from_dataset(
            _kelvin_csv(), "prediccionKelvin", **_kelvin_overrides(),
        )
        status = _submit_and_wait(proj, target_range=tuple(proj["target_range"]))
        self.assertGreater(status["mae"], 0.001, "MAE sospechosamente pequeño — ¿se quedó en espacio normalizado?")
        self.assertLess(status["mae"], 5.0, "MAE demasiado grande para una relación lineal exacta")

    def test_composite_route_avoids_catastrophic_collapse_with_normalized_target(self):
        """Mismo caso Kelvin pero con una categórica de alta cardinalidad
        que fuerza la ruta composite/embedding (`supervision_source` =
        composite_generator) — el fix de C1 se enhebró también en
        evaluate_composite_network/evaluate_composite_network_torch, no
        solo en la ruta densa plana.

        Con pocas filas y una categórica de 15 valores (60 parámetros de
        embedding), la ruta composite no converge tan limpio como la densa
        plana — eso es una cuestión de capacidad/arquitectura (C3, no C1).
        Lo que SÍ es responsabilidad de C1: que normalizar el target evite
        el desastre. Comparación antes/después medida:
        SIN target_range R²=-37.3, MAE=26.4 K, model_collapsed=True;
        CON target_range R²=-0.03..-0.12 (según semilla), MAE~4 K,
        model_collapsed=False. Se comprueban ambos lados con el MISMO
        proyecto para que la comparación sea real, no dos runs sueltos.

        **EL LADO «SIN» NO TIENE UN ÚNICO DESENLACE, y dar por hecho que sí
        hacía esta prueba dependiente del sorteo.** Medido el 2026-09-14
        sobre nueve cardinalidades de la misma categórica (13..30), con el
        MISMO código: el entrenamiento sin normalizar acaba **unas veces en
        colapso** (R²=-3.976, -1,3e121) y **otras en desbordamiento
        numérico** (el job sale `error` con `(34, 'Numerical result out of
        range')`) — 5 de 9 cardinalidades desbordaban ya ANTES de tocar
        nada. Que con 15 valores saliera colapso y no desbordamiento era el
        sorteo de la inicialización, no una propiedad: bastó que el
        vocabulario ganara UN valor (el código reservado de «categoría
        nunca vista», 101-C5) para que ese mismo caso pasara a desbordar y
        la prueba se pusiera roja sin que el producto hubiera empeorado —
        el lado que sí importa, el normalizado, pasó de R²=-0,39 / MAE=5,26
        a R²=0,63 / MAE=2,38 en ese punto.

        Las dos son la MISMA afirmación de C1 («sin normalizar, el
        entrenamiento se va al garete»), así que se aceptan las dos y se
        exige el lado normalizado con números propios, no relativos a un
        desenlace que puede no existir.
        """
        proj = generate_project_from_dataset(
            _kelvin_with_categorical_csv(), "prediccionKelvin",
        )
        self.assertEqual(proj.get("supervision_source"), "composite_generator")
        self.assertIsNotNone(proj.get("target_range"))

        status_without = _submit_and_wait_allowing_error(proj, epochs_override=100)
        se_fue_al_garete = (
            status_without["status"] == "error"
            or status_without.get("model_collapsed")
            # R² negativo de tres cifras: no es «converge peor», es no aprender.
            or (status_without.get("r2") is not None and status_without["r2"] < -100.0)
        )
        self.assertTrue(
            se_fue_al_garete,
            f"se esperaba colapso, desbordamiento o R² catastrófico SIN "
            f"normalizar el target; salió {status_without}",
        )

        status_with = _submit_and_wait(
            proj, target_range=tuple(proj["target_range"]), epochs_override=100,
        )
        self.assertEqual(status_with["task_kind"], "regression")
        self.assertFalse(status_with.get("model_collapsed"), "no debería colapsar con el target normalizado")
        # Números PROPIOS, no relativos al lado que puede no dar ninguno.
        # Medidos sobre 13..30 categorías, antes y después del código
        # reservado: R² entre -0,64 y 0,74 y MAE entre 2,3 y 5,4 K. Las
        # cotas dejan holgura de sobra a la semilla y siguen a años luz del
        # lado sin normalizar (R² de tres cifras negativas, MAE 26,4 K).
        self.assertGreater(status_with["r2"], -2.0,
                           f"R²={status_with['r2']} — el target normalizado tendría que aprender algo")
        self.assertLess(status_with["mae"], 15.0,
                        f"MAE={status_with['mae']} K — demasiado para el target normalizado")
        # Y la otra mitad: un MAE de micro-unidades sería el bug de 58.1
        # (la métrica quedándose en espacio normalizado), no un acierto.
        self.assertGreater(status_with["mae"], 0.001, "MAE sospechosamente pequeño")
        if status_without["status"] == "done":
            # Cuando el lado sin normalizar SÍ devuelve números, la
            # comparación directa se sigue exigiendo: es la más fuerte.
            self.assertGreater(
                status_with["r2"], status_without["r2"] + 5.0,
                "el fix debería mejorar R² de forma drástica, aunque no llegue a converger del todo",
            )
            self.assertLess(status_with["mae"], status_without["mae"] / 2,
                             "el MAE debería reducirse claramente al normalizar el target")


class TestRetrocompatWithoutTargetRange(unittest.TestCase):
    """Decisión B del contrato: `target_range` ausente (caller viejo, o un
    `.mxai` de antes de este contrato) debe seguir funcionando exactamente
    como hoy — nunca un error nuevo por no pasarlo."""

    def test_submitting_without_target_range_does_not_error(self):
        proj = generate_project_from_dataset(
            _kelvin_csv(n=30), "prediccionKelvin", **_kelvin_overrides(),
        )
        status = _submit_and_wait(proj)  # sin target_range
        self.assertEqual(status["task_kind"], "regression")
        self.assertIsNone(status.get("target_range"))

    def test_classification_never_receives_a_target_range(self):
        proj = generate_project_from_dataset(_classification_csv(), "y")
        self.assertIsNone(proj.get("target_range"))
        status = _submit_and_wait(proj, target_range=None)
        self.assertEqual(status["task_kind"], "classification")
        self.assertIsNone(status.get("target_range"))
        self.assertIsNotNone(status.get("accuracy"))
        self.assertGreaterEqual(status["accuracy"], 0.0)
        self.assertLessEqual(status["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
