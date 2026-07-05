# SPDX-License-Identifier: AGPL-3.0-only
"""PESOS_GRANDES C2 — el trainer no convierte: state_dict + eval + probe en torch.

Ver 48_PESOS_GRANDES_CONTRATO.md. Por debajo del umbral (`torch_native_min_params`),
comportamiento IDÉNTICO a siempre (`best_params` con valores). Por encima, el
trainer devuelve `best_state_dict` (tensores CPU) y `best_params=None` — la
conversión O(#params) a listas Python (`materialize_parameter_set`, antes
`torch_module_to_parameter_set`) se salta. `evaluate_dense_network_torch` y
`probe_collapse_torch` consumen `state_dict` directamente, sin ParameterSet.
"""
from __future__ import annotations

import random
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


def _setup():
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(MXAI)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    ps = build_network_parameter_set(net, rl, program_hash(prog), seed=1)
    return net, ps


def _signal_examples(n=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        rows.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    return rows


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ThresholdBothWaysTest(unittest.TestCase):
    """Umbral respetado en ambos sentidos (bajado por env para no entrenar una
    red real de 50M+ params en la suite de tests, mismo patrón que otros
    cortes de este contrato)."""

    def test_below_threshold_materializes_like_always(self) -> None:
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _setup()
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1000000"}):
            res = train_dense_network_torch(
                net, ps, _signal_examples(), "cross_entropy",
                lr=0.5, epochs=3, device="cpu", seed=1,
            )
        self.assertTrue(res["materialized"])
        self.assertIsNotNone(res["best_params"])
        self.assertIsNone(res["best_state_dict"])
        # forma exactamente igual a la de siempre (retro-compat, invariante 1)
        self.assertTrue(hasattr(res["best_params"], "to_dict"))

    def test_above_threshold_returns_state_dict_not_params(self) -> None:
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _setup()
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1"}):
            res = train_dense_network_torch(
                net, ps, _signal_examples(), "cross_entropy",
                lr=0.5, epochs=3, device="cpu", seed=1,
            )
        self.assertFalse(res["materialized"])
        self.assertIsNone(res["best_params"])
        self.assertIsNotNone(res["best_state_dict"])
        self.assertEqual(set(res["best_state_dict"]), {"Net.W1", "Net.b1", "Net.W2", "Net.b2"})

    def test_explicit_materialize_overrides_the_threshold(self) -> None:
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _setup()
        # umbral bajo (forzaría state_dict) pero materialize=True explícito gana
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1"}):
            res = train_dense_network_torch(
                net, ps, _signal_examples(), "cross_entropy",
                lr=0.5, epochs=3, device="cpu", seed=1, materialize=True,
            )
        self.assertTrue(res["materialized"])
        self.assertIsNotNone(res["best_params"])

    def test_invalid_env_threshold_falls_back_to_default(self) -> None:
        from matrixai.resources import torch_native_min_params, DEFAULT_TORCH_NATIVE_MIN_PARAMS
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "not-a-number"}):
            self.assertEqual(torch_native_min_params(), DEFAULT_TORCH_NATIVE_MIN_PARAMS)
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "-5"}):
            self.assertEqual(torch_native_min_params(), DEFAULT_TORCH_NATIVE_MIN_PARAMS)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ProbeAndEvalFromStateDictTest(unittest.TestCase):
    """`evaluate_dense_network_torch`/`probe_collapse_torch` con `state_dict`
    directo (sin ParameterSet) dan el MISMO resultado que con la ParameterSet
    materializada de siempre — paridad, no una vía alternativa con drift."""

    def _train_both_ways(self):
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _setup()
        examples = _signal_examples()
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1000000"}):
            materialized = train_dense_network_torch(
                net, ps, examples, "cross_entropy", lr=0.5, epochs=5, device="cpu", seed=1,
            )
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1"}):
            raw = train_dense_network_torch(
                net, ps, examples, "cross_entropy", lr=0.5, epochs=5, device="cpu", seed=1,
            )
        return net, examples, materialized, raw

    def test_eval_from_state_dict_matches_eval_from_parameter_set(self) -> None:
        from matrixai.training.dense_torch_trainer import evaluate_dense_network_torch
        net, examples, materialized, raw = self._train_both_ways()
        via_params = evaluate_dense_network_torch(
            net, materialized["best_params"], examples, "cross_entropy", device="cpu",
        )
        via_state = evaluate_dense_network_torch(
            net, None, examples, "cross_entropy", device="cpu",
            state_dict=raw["best_state_dict"],
        )
        self.assertAlmostEqual(via_params.loss, via_state.loss, places=5)
        self.assertEqual(via_params.accuracy, via_state.accuracy)

    def test_probe_from_state_dict_matches_probe_from_parameter_set(self) -> None:
        from matrixai.training.dense_torch_trainer import probe_collapse_torch
        net, examples, materialized, raw = self._train_both_ways()
        via_params = probe_collapse_torch(net, materialized["best_params"], 2, device="cpu")
        via_state = probe_collapse_torch(
            net, None, 2, device="cpu", state_dict=raw["best_state_dict"],
        )
        self.assertIsNotNone(via_params)
        self.assertIsNotNone(via_state)
        self.assertEqual(via_params["collapsed"], via_state["collapsed"])

    def test_probe_torch_matches_probe_stdlib_on_small_model(self) -> None:
        """Paridad explícita del contrato: el probe torch (nuevo camino, via
        state_dict) y el probe stdlib (_probe_model_collapse, runtime Python)
        dan el MISMO veredicto sobre el MISMO modelo entrenado."""
        from matrixai.training.dense_torch_trainer import (
            train_dense_network_torch, probe_collapse_torch,
        )
        from matrixai.playground import _probe_model_collapse

        net, ps = _setup()
        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1"}):
            res = train_dense_network_torch(
                net, ps, _signal_examples(), "cross_entropy",
                lr=0.5, epochs=5, device="cpu", seed=1,
            )
        torch_probe = probe_collapse_torch(
            net, None, 2, device="cpu", state_dict=res["best_state_dict"],
        )
        # el probe stdlib necesita una ParameterSet-con-valores (mismo dict que
        # produciría el formato json, C4) — materializamos AQUÍ solo para el
        # test de paridad, no como parte del camino de producción.
        from matrixai.forward.dense_torch import materialize_parameter_set
        materialized_ps = materialize_parameter_set(net, res["best_state_dict"], ps)
        stdlib_probe = _probe_model_collapse(MXAI, materialized_ps.to_dict())
        self.assertIsNotNone(torch_probe)
        self.assertIsNotNone(stdlib_probe)
        self.assertEqual(torch_probe["collapsed"], stdlib_probe["collapsed"])

    def test_missing_both_parameter_set_and_state_dict_raises_clean_error(self) -> None:
        from matrixai.training.dense_torch_trainer import (
            evaluate_dense_network_torch, DenseTorchTrainError,
        )
        net, _ = _setup()
        with self.assertRaises(DenseTorchTrainError):
            evaluate_dense_network_torch(net, None, _signal_examples(), "cross_entropy")

    def test_probe_missing_both_returns_none_not_crash(self) -> None:
        from matrixai.training.dense_torch_trainer import probe_collapse_torch
        net, _ = _setup()
        self.assertIsNone(probe_collapse_torch(net, None, 2, device="cpu"))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class CancelStillFreesResourcesTest(unittest.TestCase):
    """Cancelar sigue funcionando igual con el camino state_dict (por encima
    del umbral): la cancelación ocurre DENTRO del bucle, antes de decidir
    materialize — el `finally` (limpieza de VRAM) no cambia con este corte."""

    def test_epoch_callback_can_cancel_above_threshold(self) -> None:
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _setup()

        class _Stop(Exception):
            pass

        def cb(entry):
            if entry["epoch"] >= 3:
                raise _Stop()

        with patch.dict("os.environ", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "1"}):
            with self.assertRaises(_Stop):
                train_dense_network_torch(
                    net, ps, _signal_examples(), "cross_entropy",
                    lr=0.5, epochs=80, device="cpu", seed=1, epoch_callback=cb,
                )


class DenseModuleStateDictHelpersTest(unittest.TestCase):
    """torch_module_to_parameter_set sigue dando EXACTAMENTE el mismo resultado
    tras el refactor (reusa dense_module_to_state_dict + materialize_parameter_set
    internamente) — cero cambio de comportamiento para callers existentes."""

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_torch_module_to_parameter_set_unchanged_after_refactor(self) -> None:
        from matrixai.forward.dense_torch import (
            dense_network_to_torch_module, torch_module_to_parameter_set,
            dense_module_to_state_dict, materialize_parameter_set,
        )
        net, ps = _setup()
        module = dense_network_to_torch_module(net, ps)
        direct = torch_module_to_parameter_set(net, module, ps)
        via_helpers = materialize_parameter_set(
            net, dense_module_to_state_dict(net, module), ps
        )
        self.assertEqual(direct.to_dict(), via_helpers.to_dict())


if __name__ == "__main__":
    unittest.main()
