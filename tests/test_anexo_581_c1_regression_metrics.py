# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Anexo 58.1 C1 — métricas veraces por tarea: un `0.0`/`[]`/`{}` de la tarea
que NO aplica (accuracy/macro_f1/confusion_matrix/labels/per_label en
regresión; mae/rmse/r2 en clasificación) nunca debe llegar al resultado de
entrenamiento — debe ser `None`, para que el Studio pueda distinguir "no
aplica" de "rendimiento nulo"."""
from __future__ import annotations

import time
import unittest

from matrixai.playground import _eval_report_from_dense_result, _get_job_status, _submit_training_job
from matrixai.training.dataset_project import generate_project_from_dataset
from matrixai.training.dense_evaluator import DenseEvaluationResult
from matrixai.training.spec import EvaluationResult


def _regression_csv(n: int = 60) -> str:
    lines = ["a,b,y"]
    for i in range(n):
        a, b = (i * 0.37) % 10, (i * 0.53) % 10
        lines.append(f"{a:.4f},{b:.4f},{(a + b) / 2:.4f}")
    return "\n".join(lines) + "\n"


def _classification_csv(n: int = 60) -> str:
    lines = ["a,b,y"]
    for i in range(n):
        a, b = (i * 0.37) % 10, (i * 0.53) % 10
        lines.append(f"{a:.4f},{b:.4f},{'pos' if a + b > 10 else 'neg'}")
    return "\n".join(lines) + "\n"


def _train_and_wait(csv_text: str, target: str = "y") -> dict:
    proj = generate_project_from_dataset(csv_text, target)
    submitted = _submit_training_job(proj["mxai"], proj["training_text"], proj["csv_text"], epochs_override=2)
    assert submitted.get("ok"), submitted.get("error")
    job_id = submitted["job_id"]
    status: dict = {}
    for _ in range(200):
        status = _get_job_status(job_id)
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert status["status"] == "done", status.get("error")
    return status


class TestEvaluationResultToDictDiscriminatesByTask(unittest.TestCase):
    """Fix de raíz: `EvaluationResult.to_dict()` (spec.py) antes SIEMPRE
    incluía accuracy/labels/confusion_matrix/per_label/macro_precision/
    macro_recall/macro_f1, incluso en regresión (con sus defaults 0.0/[]/{}).
    Espejo de la disciminación que `DenseEvaluationResult.to_dict()` ya
    aplicaba correctamente."""

    def _regression_result(self) -> EvaluationResult:
        return EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s", parameter_set_id="p",
            dataset="d", dataset_fingerprint="f", dataset_schema={}, rows=10, loss=0.01,
            accuracy=0.0, labels=[], confusion_matrix={}, per_label={},
            macro_precision=0.0, macro_recall=0.0, macro_f1=0.0,
            mae=0.05, rmse=0.1, r2=0.99,
        )

    def _classification_result(self) -> EvaluationResult:
        return EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s", parameter_set_id="p",
            dataset="d", dataset_fingerprint="f", dataset_schema={}, rows=10, loss=0.5,
            accuracy=0.9, labels=["a", "b"], confusion_matrix={"a": {"a": 1}}, per_label={"a": {}},
            macro_precision=0.9, macro_recall=0.9, macro_f1=0.9,
        )

    def test_regression_omits_classification_keys_entirely(self):
        d = self._regression_result().to_dict()
        for key in ("accuracy", "labels", "confusion_matrix", "per_label",
                    "macro_precision", "macro_recall", "macro_f1"):
            self.assertNotIn(key, d, f"{key!r} should be absent for regression, not 0.0/[]/{{}}")

    def test_regression_still_carries_its_own_metrics(self):
        d = self._regression_result().to_dict()
        self.assertAlmostEqual(d["mae"], 0.05)
        self.assertAlmostEqual(d["rmse"], 0.1)
        self.assertAlmostEqual(d["r2"], 0.99)

    def test_classification_omits_regression_keys(self):
        d = self._classification_result().to_dict()
        for key in ("mae", "rmse", "r2"):
            self.assertNotIn(key, d)

    def test_classification_still_carries_its_own_metrics(self):
        d = self._classification_result().to_dict()
        self.assertAlmostEqual(d["accuracy"], 0.9)
        self.assertAlmostEqual(d["macro_f1"], 0.9)
        self.assertEqual(d["labels"], ["a", "b"])

    def test_binary_classification_without_named_labels_is_not_mistaken_for_regression(self):
        """Hallazgo real descubierto al escribir este anexo: `is_regression()`
        heredado usaba `not self.labels` — una clasificación BINARIA con
        salida `Probability` (sigmoid) y SIN bloque `LABELS` (nombres de
        clase) también tiene `labels=[]`, así que se confundía con
        regresión. Reproducido con `test_m3_metrics.py::
        test_binary_classification_has_metrics` (que empezó a fallar al
        discriminar `to_dict()`, exponiendo el bug preexistente). Fix:
        `loss_fn` explícito, con `not labels` solo como fallback para
        quien no lo pase."""
        binary_no_labels = EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s", parameter_set_id="p",
            dataset="d", dataset_fingerprint="f", dataset_schema={}, rows=10, loss=0.5,
            accuracy=0.8, labels=[], confusion_matrix={}, per_label={},
            macro_precision=0.0, macro_recall=0.0, macro_f1=0.7,
            loss_fn="binary_cross_entropy",
        )
        self.assertFalse(binary_no_labels.is_regression())
        d = binary_no_labels.to_dict()
        self.assertIn("accuracy", d)
        self.assertIn("macro_f1", d)
        self.assertNotIn("mae", d)

    def test_is_regression_without_loss_fn_falls_back_to_legacy_label_heuristic(self):
        """Retrocompat: un caller que NO pase `loss_fn` (código/tests
        anteriores a este corte) conserva el comportamiento previo exacto."""
        legacy = EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s", parameter_set_id="p",
            dataset="d", dataset_fingerprint="f", dataset_schema={}, rows=10, loss=0.01,
            accuracy=0.0, labels=[], confusion_matrix={}, per_label={},
            macro_precision=0.0, macro_recall=0.0, macro_f1=0.0,
        )
        self.assertTrue(legacy.is_regression())


class TestEvalReportFromDenseResultDiscriminatesByTask(unittest.TestCase):
    """`_eval_report_from_dense_result` (playground.py, ruta torch/GPU M14)
    reconstruye el dict con acceso DIRECTO a los atributos del dataclass
    (nunca `.to_dict()`), así que no heredaba la discriminación — devolvía
    accuracy/macro_f1 en 0.0 incluso en regresión, y mae/rmse/r2 en 0.0
    incluso en clasificación."""

    def test_regression_reports_none_for_classification_keys(self):
        result = DenseEvaluationResult(rows=10, loss=0.5, loss_fn="mse", mae=0.1, rmse=0.2, r2=0.9)
        report = _eval_report_from_dense_result(result, None)
        self.assertIsNone(report["accuracy"])
        self.assertIsNone(report["macro_f1"])
        self.assertIsNone(report["confusion_matrix"])
        self.assertIsNone(report["labels"])
        self.assertIsNone(report["per_label"])
        self.assertEqual(report["mae"], 0.1)
        self.assertEqual(report["rmse"], 0.2)
        self.assertEqual(report["r2"], 0.9)

    def test_classification_reports_none_for_regression_keys(self):
        result = DenseEvaluationResult(
            rows=10, loss=0.5, loss_fn="cross_entropy", accuracy=0.8, macro_f1=0.75,
            labels=["a", "b"], confusion_matrix={"a": {"a": 5}},
            precision={"a": 0.8}, recall={"a": 0.8}, f1={"a": 0.8},
        )
        report = _eval_report_from_dense_result(result, ["a", "b"])
        self.assertIsNone(report["mae"])
        self.assertIsNone(report["rmse"])
        self.assertIsNone(report["r2"])
        self.assertEqual(report["accuracy"], 0.8)
        self.assertEqual(report["macro_f1"], 0.75)
        self.assertIn("a", report["per_label"])


class TestStudioTrainingResultTruthfulMetricsEndToEnd(unittest.TestCase):
    """Reproducción exacta del hallazgo original: entrenar una plantilla de
    regresión real (vía el mismo camino que usa el Studio,
    `generate_project_from_dataset` → `_submit_training_job` →
    `_get_job_status`) y confirmar que `macro_f1`/`accuracy` NUNCA son 0.0
    para una tarea que no aplica, y que `mae`/`rmse`/`r2` SÍ llegan
    poblados (antes siempre `None` para un modelo NETWORK denso, porque se
    leían del artefacto "metrics.json" que solo escribe el entrenador
    lineal/FUNCTION legado — `DenseSupervisedTrainer` nunca lo produce)."""

    def test_regression_job_never_reports_zero_macro_f1_or_accuracy(self):
        status = _train_and_wait(_regression_csv())
        self.assertEqual(status["task_kind"], "regression")
        self.assertIsNone(status.get("accuracy"))
        self.assertIsNone(status.get("macro_f1"))
        self.assertIsNone(status.get("confusion_matrix"))
        self.assertIsNone(status.get("labels"))
        self.assertIsNone(status.get("per_label"))

    def test_regression_job_reports_real_mae_rmse_r2(self):
        status = _train_and_wait(_regression_csv())
        self.assertIsNotNone(status.get("mae"))
        self.assertIsNotNone(status.get("rmse"))
        self.assertIsNotNone(status.get("r2"))
        self.assertGreaterEqual(status["mae"], 0.0)
        self.assertGreaterEqual(status["rmse"], 0.0)

    def test_classification_job_reports_none_for_regression_metrics(self):
        status = _train_and_wait(_classification_csv())
        self.assertEqual(status["task_kind"], "classification")
        self.assertIsNone(status.get("mae"))
        self.assertIsNone(status.get("rmse"))
        self.assertIsNone(status.get("r2"))

    def test_classification_job_accuracy_unchanged_by_this_fix(self):
        """Regresión de no alterar lo que ya funcionaba: `accuracy` de
        clasificación sigue viniendo de la MISMA fuente de siempre
        (TrainingRunResult, no evaluation_report) — solo se discrimina a
        `None` en regresión, nunca se cambia de fuente para clasificación."""
        status = _train_and_wait(_classification_csv())
        self.assertIsNotNone(status.get("accuracy"))
        self.assertGreaterEqual(status["accuracy"], 0.0)
        self.assertLessEqual(status["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
