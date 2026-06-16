"""P14 Corte 4: backend_runtime block in training_trace.json and evaluation_report.json."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.typed.mxai"
_EMAIL_TRAIN = _BASE / "examples" / "email-agent.supervised.mxtrain"


# ---------------------------------------------------------------------------
# EvaluationResult.backend_runtime field contract
# ---------------------------------------------------------------------------

class TestEvaluationResultBackendRuntimeField(unittest.TestCase):
    def test_field_exists_with_empty_default(self):
        from matrixai.training.spec import EvaluationResult
        import dataclasses
        names = {f.name for f in dataclasses.fields(EvaluationResult)}
        self.assertIn("backend_runtime", names)

    def test_to_dict_omits_backend_runtime_when_empty(self):
        from matrixai.training.spec import EvaluationResult
        result = EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s",
            parameter_set_id="p", dataset="d", dataset_fingerprint="f",
            dataset_schema={}, rows=10, loss=0.5, accuracy=0.9,
            labels=["a", "b"], confusion_matrix={}, per_label={},
            macro_precision=0.9, macro_recall=0.9, macro_f1=0.9,
        )
        d = result.to_dict()
        self.assertNotIn("backend_runtime", d)

    def test_to_dict_includes_backend_runtime_when_set(self):
        from matrixai.training.spec import EvaluationResult
        result = EvaluationResult(
            model="m", model_hash="h", parameter_schema_hash="s",
            parameter_set_id="p", dataset="d", dataset_fingerprint="f",
            dataset_schema={}, rows=10, loss=0.5, accuracy=0.9,
            labels=["a", "b"], confusion_matrix={}, per_label={},
            macro_precision=0.9, macro_recall=0.9, macro_f1=0.9,
            backend_runtime={"target": "torch", "device": "cpu"},
        )
        d = result.to_dict()
        self.assertIn("backend_runtime", d)
        self.assertEqual(d["backend_runtime"]["target"], "torch")
        self.assertEqual(d["backend_runtime"]["device"], "cpu")


# ---------------------------------------------------------------------------
# _build_backend_runtime helper
# ---------------------------------------------------------------------------

class TestBuildBackendRuntime(unittest.TestCase):
    def _make_training(self, device="cpu"):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.spec import BackendSpec
        spec = parse_training_file(_EMAIL_TRAIN)
        return dataclasses.replace(spec, backend=BackendSpec(target="torch", device=device))

    def test_returns_dict_with_required_keys(self):
        from matrixai.training.torch_trainer import _build_backend_runtime
        training = self._make_training()
        rt = _build_backend_runtime("cpu", training)
        for key in ("target", "device", "torch_version", "device_name"):
            self.assertIn(key, rt)

    def test_target_is_torch(self):
        from matrixai.training.torch_trainer import _build_backend_runtime
        training = self._make_training()
        rt = _build_backend_runtime("cpu", training)
        self.assertEqual(rt["target"], "torch")

    def test_device_reflects_argument(self):
        from matrixai.training.torch_trainer import _build_backend_runtime
        training = self._make_training()
        rt = _build_backend_runtime("cpu", training)
        self.assertEqual(rt["device"], "cpu")

    def test_seed_key_present(self):
        from matrixai.training.torch_trainer import _build_backend_runtime
        training = self._make_training()
        rt = _build_backend_runtime("cpu", training)
        self.assertIn("seed", rt)

    def test_cuda_version_absent_on_cpu_only_machine(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if torch_device_info().get("cuda_available"):
            self.skipTest("CUDA available — cuda_version key expected")
        from matrixai.training.torch_trainer import _build_backend_runtime
        training = self._make_training()
        rt = _build_backend_runtime("cpu", training)
        self.assertNotIn("cuda_version", rt)

    def test_cuda_version_present_when_cuda_available(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if not torch_device_info().get("cuda_available"):
            self.skipTest("CUDA not available")
        from matrixai.training.torch_trainer import _build_backend_runtime
        training = self._make_training("cuda")
        rt = _build_backend_runtime("cuda", training)
        self.assertIn("cuda_version", rt)


# ---------------------------------------------------------------------------
# training_trace.json receives backend_runtime after torch train
# ---------------------------------------------------------------------------

class TestTrainingTraceBackendRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.spec import BackendSpec
        from matrixai.training.torch_trainer import TorchSupervisedTrainer
        spec = parse_training_file(_EMAIL_TRAIN)
        spec = dataclasses.replace(spec, backend=BackendSpec(target="torch", device="cpu"))
        cls._tmpdir = tempfile.TemporaryDirectory()
        result = TorchSupervisedTrainer().train(
            spec,
            output_dir=cls._tmpdir.name + "/run",
            base_path=_BASE,
            training_path=_EMAIL_TRAIN,
        )
        trace_path = Path(result.artifacts["training_trace"])
        cls._trace = json.loads(trace_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_training_trace_has_backend_runtime_key(self):
        self.assertIn("backend_runtime", self._trace)

    def test_backend_runtime_target_is_torch(self):
        self.assertEqual(self._trace["backend_runtime"]["target"], "torch")

    def test_backend_runtime_device_is_cpu(self):
        self.assertEqual(self._trace["backend_runtime"]["device"], "cpu")

    def test_backend_runtime_has_torch_version(self):
        self.assertIn("torch_version", self._trace["backend_runtime"])
        self.assertIsNotNone(self._trace["backend_runtime"]["torch_version"])

    def test_backend_runtime_has_device_name_key(self):
        self.assertIn("device_name", self._trace["backend_runtime"])

    def test_backend_runtime_has_seed_key(self):
        self.assertIn("seed", self._trace["backend_runtime"])


# ---------------------------------------------------------------------------
# stdlib training_trace does NOT get backend_runtime (form unchanged)
# ---------------------------------------------------------------------------

class TestStdlibTrainingTraceNoBackendRuntime(unittest.TestCase):
    def test_stdlib_trace_has_no_backend_runtime(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.trainer import SupervisedTrainer
        spec = parse_training_file(_EMAIL_TRAIN)
        self.assertIsNone(spec.backend)
        tmpdir = tempfile.TemporaryDirectory()
        try:
            result = SupervisedTrainer().train(
                spec,
                output_dir=tmpdir.name + "/run",
                base_path=_BASE,
                training_path=_EMAIL_TRAIN,
            )
            trace_path = Path(result.artifacts["training_trace"])
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertNotIn("backend_runtime", trace)
        finally:
            tmpdir.cleanup()


# ---------------------------------------------------------------------------
# EvaluationResult carries backend_runtime for torch evaluator
# ---------------------------------------------------------------------------

class TestTorchEvaluatorBackendRuntime(unittest.TestCase):
    @classmethod
    def _params(cls):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set
        return build_initial_parameter_set(parse_file(_EMAIL_MXAI))

    def test_torch_evaluator_sets_backend_runtime(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.spec import BackendSpec
        from matrixai.training.torch_evaluator import TorchSupervisedEvaluator
        spec = parse_training_file(_EMAIL_TRAIN)
        spec = dataclasses.replace(spec, backend=BackendSpec(target="torch", device="cpu"))
        result = TorchSupervisedEvaluator().evaluate(
            spec, parameter_set=self._params(), base_path=_BASE
        )
        self.assertTrue(result.backend_runtime)
        self.assertEqual(result.backend_runtime["target"], "torch")
        self.assertEqual(result.backend_runtime["device"], "cpu")

    def test_torch_evaluator_backend_runtime_in_to_dict(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.spec import BackendSpec
        from matrixai.training.torch_evaluator import TorchSupervisedEvaluator
        spec = parse_training_file(_EMAIL_TRAIN)
        spec = dataclasses.replace(spec, backend=BackendSpec(target="torch", device="cpu"))
        result = TorchSupervisedEvaluator().evaluate(
            spec, parameter_set=self._params(), base_path=_BASE
        )
        d = result.to_dict()
        self.assertIn("backend_runtime", d)
        self.assertEqual(d["backend_runtime"]["target"], "torch")

    def test_stdlib_evaluator_no_backend_runtime(self):
        from matrixai.training.parser import parse_training_file
        from matrixai.training.trainer import SupervisedEvaluator
        spec = parse_training_file(_EMAIL_TRAIN)
        result = SupervisedEvaluator().evaluate(
            spec, parameter_set=self._params(), base_path=_BASE
        )
        d = result.to_dict()
        self.assertNotIn("backend_runtime", d)


if __name__ == "__main__":
    unittest.main()
