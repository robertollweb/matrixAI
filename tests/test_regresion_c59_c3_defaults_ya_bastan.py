# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 59 C3 — defaults de regresión (decisión E): "se ajustan SOLO
hasta donde haga falta para que el caso canónico aprenda... o si basta con
C1". Medido, no asumido: con el target normalizado (C1), el caso canónico
(centigrados -> Kelvin, un solo feature, relación lineal) alcanza R² >= 0.99
con los defaults YA existentes — SGD lr=0.01, 50 épocas, arquitectura densa
32->16->1 ReLU, split secuencial (la ruta densa de NETWORK,
`DenseSupervisedTrainer`, es deliberadamente secuencial desde
BIBLIOTECA_PROYECTOS_INTELIGENTES C3, invariante de ESE corte — nunca
tocado aquí).

Conclusión de C3: NINGÚN default cambia. Este fichero fija esa conclusión
con evidencia — robustez frente a la semilla de inicialización de pesos Y
frente al orden de las filas del CSV (ninguno de los dos debería importar
para una relación lineal simple; si alguna vez dejara de ser cierto, estos
tests lo detectan antes que un usuario real)."""
from __future__ import annotations

import random
import time
import unittest

from matrixai.playground import _get_job_status, _submit_training_job
from matrixai.training.dataset_project import generate_project_from_dataset


def _kelvin_rows(order: list[int]) -> str:
    lines = ["centigrados,prediccionKelvin"]
    for c in order:
        lines.append(f"{c},{c + 273.15}")
    return "\n".join(lines) + "\n"


def _train_with_seed(csv_text: str, seed: int) -> dict:
    proj = generate_project_from_dataset(csv_text, "prediccionKelvin")
    submitted = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"],
        field_ranges=proj.get("field_ranges"), target_range=tuple(proj["target_range"]),
        seed=seed,
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


class TestCurrentDefaultsAlreadyLearnTheCanonicalCase(unittest.TestCase):
    """Con C1 aplicado, ningún ajuste de shuffle/lr/épocas/arquitectura es
    necesario — se mide con varias semillas de inicialización de pesos Y
    varios órdenes de fila, no una sola corrida con suerte."""

    def test_ascending_order_multiple_seeds(self):
        csv_text = _kelvin_rows(list(range(100)))
        for seed in (42, 1, 7):
            status = _train_with_seed(csv_text, seed)
            self.assertGreaterEqual(
                status["r2"], 0.99, f"seed={seed}: R²={status['r2']} — un default dejó de bastar",
            )
            self.assertFalse(status.get("model_collapsed"), f"seed={seed}")

    def test_fully_shuffled_file_order(self):
        """La ruta densa de NETWORK (`DenseSupervisedTrainer`) parte el
        split de forma SECUENCIAL (primeras filas = train, últimas =
        validación) por diseño de un corte anterior — este test confirma
        que, para el caso canónico, el orden del fichero no importa: tanto
        ascendente (test anterior) como aleatorio dan R² >= 0.99."""
        rnd = random.Random(99)
        order = list(range(100))
        rnd.shuffle(order)
        status = _train_with_seed(_kelvin_rows(order), seed=42)
        self.assertGreaterEqual(status["r2"], 0.99)
        self.assertFalse(status.get("model_collapsed"))


if __name__ == "__main__":
    unittest.main()
