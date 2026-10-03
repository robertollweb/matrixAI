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
frente al orden de las filas del CSV.

CORTE «RANGOS DE TRAIN» (03-10): se entrena COMO EL STUDIO (`/train-start`:
objetivo sin recortar, `recortar_objetivo=False`), y el orden del fichero SÍ
importa cuando el dominio no se declara. Los rangos de normalización salen solo
de las filas de train, y en el Kelvin ascendente sin dominio declarado la
validación (80..99 °C) cae fuera de lo visto al entrenar: es extrapolación, y la
cifra honrada lo dice (medido como el Studio: R² 0,98925 con la semilla 1 por
stdlib, 0,02 por torch; lo ata `test_T6_el_caso_ordenado_sin_dominio_declarado_
da_la_cifra_honrada` en `test_rangos_de_train_nucleo.py`). Antes de ese corte
«el orden no importa» era cierto solo porque el rango veía la validación. Así
que el caso canónico ascendente DECLARA su dominio (`centigrados` 0..99, como
`test_regresion_c59_c1`), y el barajado no lo necesita: los dos con los
defaults, que es lo que este fichero afirma."""
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


#: El dominio del caso canónico, declarado como lo declara `test_regresion_c59_c1`.
_DOMINIO_KELVIN = {
    "column_type_overrides": {"centigrados": "number"},
    "column_range_overrides": {"centigrados": (0.0, 99.0)},
}


def _train_with_seed(csv_text: str, seed: int, **overrides) -> dict:
    proj = generate_project_from_dataset(csv_text, "prediccionKelvin", **overrides)
    submitted = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"],
        field_ranges=proj.get("field_ranges"), target_range=tuple(proj["target_range"]),
        seed=seed,
        # Como entrena el Studio: el objetivo NO se recorta (A8).
        recortar_objetivo=False,
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
        """Ascendente con su dominio declarado. Medido como el Studio: R² 0,9999 /
        0,9977 / 1,0000 por stdlib y 1,0 / 1,0 / 0,9998 por torch."""
        csv_text = _kelvin_rows(list(range(100)))
        for seed in (42, 1, 7):
            status = _train_with_seed(csv_text, seed, **_DOMINIO_KELVIN)
            self.assertGreaterEqual(
                status["r2"], 0.99, f"seed={seed}: R²={status['r2']} — un default dejó de bastar",
            )
            self.assertFalse(status.get("model_collapsed"), f"seed={seed}")

    def test_fully_shuffled_file_order(self):
        """La ruta densa de NETWORK (`DenseSupervisedTrainer`) parte el
        split de forma SECUENCIAL (primeras filas = train, últimas =
        validación) por diseño de un corte anterior. Barajado, la validación
        cae dentro de lo visto al entrenar y el caso canónico aprende SIN
        declarar el dominio (medido como el Studio: R² 1,0 por stdlib y por
        torch). Ascendente sin dominio no: ver el docstring del módulo."""
        rnd = random.Random(99)
        order = list(range(100))
        rnd.shuffle(order)
        status = _train_with_seed(_kelvin_rows(order), seed=42)
        self.assertGreaterEqual(status["r2"], 0.99)
        self.assertFalse(status.get("model_collapsed"))


if __name__ == "__main__":
    unittest.main()
