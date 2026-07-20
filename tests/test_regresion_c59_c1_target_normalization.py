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
        el colapso catastrófico. Comparación antes/después medida:
        SIN target_range R²=-37.3, MAE=26.4 K, model_collapsed=True;
        CON target_range R²=-0.03..-0.12 (según semilla), MAE~4 K,
        model_collapsed=False. Se comprueban ambos lados con el MISMO
        proyecto para que la comparación sea real, no dos runs sueltos."""
        proj = generate_project_from_dataset(
            _kelvin_with_categorical_csv(), "prediccionKelvin",
        )
        self.assertEqual(proj.get("supervision_source"), "composite_generator")
        self.assertIsNotNone(proj.get("target_range"))

        status_without = _submit_and_wait(proj, epochs_override=100)
        self.assertTrue(status_without.get("model_collapsed"), "se esperaba colapso SIN normalizar el target")

        status_with = _submit_and_wait(
            proj, target_range=tuple(proj["target_range"]), epochs_override=100,
        )
        self.assertEqual(status_with["task_kind"], "regression")
        self.assertFalse(status_with.get("model_collapsed"), "no debería colapsar con el target normalizado")
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
