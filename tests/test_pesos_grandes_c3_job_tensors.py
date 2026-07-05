# SPDX-License-Identifier: AGPL-3.0-only
"""PESOS_GRANDES C3 — el job y el resultado llevan tensores, no dicts.

Ver 48_PESOS_GRANDES_CONTRATO.md. Por debajo del umbral, `params_best` es el
dict de valores de siempre (retro-compat). Por encima, `params_best` es un
MARCADOR (sin valores) y los tensores viajan en `best_state_dict` en la memoria
del job; `_get_job_status` nunca los expone (no serializables). Cualquier
consumidor que haga `ParameterSet.from_dict(params_best)` sobre un grande recibe
un error CLARO (no un KeyError).
"""
from __future__ import annotations

import csv
import io
import json
import random
import time
import unittest
from importlib import util
from unittest.mock import patch

_HAS_TORCH = util.find_spec("torch") is not None

MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""

TRAIN = """MODEL P.mxai
DATASET D
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [a, b]
  TARGET predicted_class: Label[A, B]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END
LOSS L
  TYPE cross_entropy
  PREDICTION Net
  TARGET predicted_class
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE Net.*
END
RUN
  EPOCHS 2
END
"""


def _csv() -> str:
    rng = random.Random(0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["a", "b", "predicted_class"])
    for _ in range(40):
        a, b = rng.random(), rng.random()
        w.writerow([a, b, "B" if a > 0.5 else "A"])
    return buf.getvalue()


class MarkerAndFromDictTest(unittest.TestCase):
    """El marcador y su guard en ParameterSet.from_dict — no necesitan torch."""

    def test_marker_roundtrips_and_is_detected(self) -> None:
        from matrixai.parameters.store import (
            build_torch_state_marker, is_torch_state_marker,
        )
        marker = build_torch_state_marker(4_000_000_000)
        self.assertTrue(is_torch_state_marker(marker))
        self.assertEqual(marker["param_count"], 4_000_000_000)
        self.assertFalse(marker["materialized"])
        # un dict de ParameterSet normal NO es un marcador
        self.assertFalse(is_torch_state_marker({"parameter_set_id": "x", "parameters": {}}))
        self.assertFalse(is_torch_state_marker(None))

    def test_from_dict_on_marker_raises_actionable_error(self) -> None:
        from matrixai.parameters.store import ParameterSet, build_torch_state_marker
        with self.assertRaises(ValueError) as ctx:
            ParameterSet.from_dict(build_torch_state_marker(50_000_000))
        msg = str(ctx.exception)
        self.assertIn("marcador", msg)
        self.assertIn("C4", msg)  # apunta al corte que lo persiste

    def test_from_dict_on_real_params_still_works(self) -> None:
        from matrixai.parameters.store import ParameterSet
        data = {
            "parameter_set_id": "ps1", "model_hash": "m", "parameter_schema_hash": "s",
            "source": "torch", "parameters": {"W": {"values": [[1.0]]}}, "metrics": {},
        }
        ps = ParameterSet.from_dict(data)  # no debe lanzar
        self.assertEqual(ps.parameter_set_id, "ps1")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class JobCarriesTensorsTest(unittest.TestCase):
    def _train(self, threshold: str):
        from matrixai.playground import _run_playground_dense_training
        with patch.dict("os.environ", {
            "MATRIXAI_TRAIN_BACKEND": "torch",
            "MATRIXAI_TORCH_NATIVE_MIN_PARAMS": threshold,
        }):
            return _run_playground_dense_training(MXAI, TRAIN, _csv(), epochs_override=3)

    def test_small_model_keeps_values_dict_retrocompat(self) -> None:
        from matrixai.parameters.store import is_torch_state_marker
        res = self._train("1000000")  # umbral alto → materializa
        self.assertTrue(res["ok"], res.get("error"))
        self.assertTrue(res["materialized"])
        self.assertFalse(is_torch_state_marker(res["params_best"]))
        self.assertIn("parameters", res["params_best"])   # dict de valores real
        self.assertIsNone(res["best_state_dict"])

    def test_large_model_carries_marker_and_tensors_not_lists(self) -> None:
        from matrixai.parameters.store import is_torch_state_marker
        res = self._train("1")  # umbral 1 → cualquier red es "grande"
        self.assertTrue(res["ok"], res.get("error"))
        self.assertFalse(res["materialized"])
        self.assertTrue(is_torch_state_marker(res["params_best"]))
        # el marcador NO tiene listas de valores
        self.assertNotIn("parameters", res["params_best"])
        # los tensores están, en memoria, como state_dict (no listas)
        self.assertIsNotNone(res["best_state_dict"])
        import torch
        for v in res["best_state_dict"].values():
            self.assertIsInstance(v, torch.Tensor)

    def test_large_model_trains_by_torch_not_stdlib_fallback(self) -> None:
        import torch
        res = self._train("1")
        # el bug de la frontera C2 era caer a stdlib; C3 lo cierra de verdad
        self.assertEqual(res["backend"], "cuda" if torch.cuda.is_available() else "cpu")

    def test_large_model_result_without_tensors_is_json_serializable(self) -> None:
        res = self._train("1")
        payload = {k: v for k, v in res.items() if k != "best_state_dict"}
        json.dumps(payload)  # no debe lanzar (marcador + métricas, sin tensores)

    def test_collapse_probed_by_torch_for_large_model(self) -> None:
        # el probe M7 corrió por torch (state_dict) durante el entrenamiento →
        # model_collapsed presente sin haber materializado nada
        res = self._train("1")
        self.assertIn("model_collapsed", res)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class JobStatusStripsTensorsTest(unittest.TestCase):
    """`_get_job_status` nunca expone `best_state_dict` (torch tensors no
    serializables) — el flujo async completo debe dar un status JSON-safe."""

    def test_async_job_status_has_marker_and_no_tensors(self) -> None:
        from matrixai.playground import _submit_training_job, _get_job_status
        from matrixai.parameters.store import is_torch_state_marker
        with patch.dict("os.environ", {
            "MATRIXAI_TRAIN_BACKEND": "torch",
            "MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1",
        }):
            job = _submit_training_job(MXAI, TRAIN, _csv(), epochs_override=3)
            job_id = job["job_id"]
            for _ in range(240):
                st = _get_job_status(job_id)
                if st["status"] != "running":
                    break
                time.sleep(0.05)
        self.assertEqual(st["status"], "done", st)
        self.assertNotIn("best_state_dict", st)          # tensores fuera del status
        self.assertTrue(is_torch_state_marker(st["params_best"]))
        json.dumps(st)                                    # status entero es JSON-safe


if __name__ == "__main__":
    unittest.main()
