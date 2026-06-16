"""P14 Corte 3: Device propagation to the torch tensor backend."""
from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_MXAI = _BASE / "examples" / "email-agent.typed.mxai"
_EMAIL_TRAIN = _BASE / "examples" / "email-agent.supervised.mxtrain"

# ---------------------------------------------------------------------------
# TorchForwardRunner._validate_config device gating
# ---------------------------------------------------------------------------

class TestTorchForwardRunnerDeviceValidation(unittest.TestCase):
    def test_cpu_device_always_valid(self):
        from matrixai.compiler.torch_forward import TorchForwardRunner
        # Should not raise
        runner = TorchForwardRunner(device="cpu")
        runner._validate_config()

    def test_float64_dtype_raises(self):
        from matrixai.compiler.torch_forward import TorchForwardError, TorchForwardRunner
        runner = TorchForwardRunner(device="cpu", dtype="float64")
        with self.assertRaises(TorchForwardError):
            runner._validate_config()

    def test_cuda_unavailable_raises_forward_error(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if "cuda" in torch_device_info()["available_devices"]:
            self.skipTest("CUDA available — skip unavailable error test")
        from matrixai.compiler.torch_forward import TorchForwardError, TorchForwardRunner
        runner = TorchForwardRunner(device="cuda")
        with self.assertRaises(TorchForwardError) as ctx:
            runner._validate_config()
        self.assertIn("cuda", str(ctx.exception).lower())
        self.assertIn("not available", str(ctx.exception).lower())

    def test_mps_unavailable_raises_forward_error(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if "mps" in torch_device_info()["available_devices"]:
            self.skipTest("MPS available — skip unavailable error test")
        from matrixai.compiler.torch_forward import TorchForwardError, TorchForwardRunner
        runner = TorchForwardRunner(device="mps")
        with self.assertRaises(TorchForwardError) as ctx:
            runner._validate_config()
        self.assertIn("mps", str(ctx.exception).lower())

    def test_error_message_contains_available_devices(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if "cuda" in torch_device_info()["available_devices"]:
            self.skipTest("CUDA available — skip unavailable error test")
        from matrixai.compiler.torch_forward import TorchForwardError, TorchForwardRunner
        runner = TorchForwardRunner(device="cuda")
        with self.assertRaises(TorchForwardError) as ctx:
            runner._validate_config()
        # Error message should mention what IS available
        self.assertIn("cpu", str(ctx.exception).lower())

    def test_old_p5_error_message_no_longer_present(self):
        """P5 hard-coded error is gone; only dtype and availability checks remain."""
        from matrixai.compiler.torch_forward import TorchForwardRunner
        import inspect
        src = inspect.getsource(TorchForwardRunner._validate_config)
        self.assertNotIn("P5 torch forward supports only device", src)


# ---------------------------------------------------------------------------
# TorchForwardRunner.run() parameter tensor device movement
# ---------------------------------------------------------------------------

class TestTorchForwardRunnerCpuRun(unittest.TestCase):
    """Smoke test: forward run on cpu still works after Corte 3 changes."""

    def test_run_cpu_produces_result(self):
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parser import parse_file
        program = parse_file(_EMAIL_MXAI)
        result = TorchForwardRunner(device="cpu").run(
            program, {"Email": {"urgency": 0.7, "formality": 0.8, "urgency_score": 0.6}}
        )
        self.assertEqual(result["target"], "torch")
        self.assertIn("state", result)


# ---------------------------------------------------------------------------
# _batch_tensors device parameter
# ---------------------------------------------------------------------------

class TestBatchTensorsDeviceParameter(unittest.TestCase):
    def _make_batch(self):
        from matrixai.training.data import MatrixAIBatch
        return MatrixAIBatch(
            inputs={"Email": [[0.7, 0.8, 0.6], [0.3, 0.4, 0.5]]},
            targets={"label": ["support", "sales"]},
            metadata={},
        )

    def test_batch_tensors_cpu_default(self):
        from matrixai.training.torch_trainer import _batch_tensors
        import importlib
        torch = importlib.import_module("torch")
        batch = self._make_batch()
        inp, tgt = _batch_tensors(batch, "Email", "label", "softmax_cross_entropy", ["support", "sales"], torch)
        self.assertEqual(str(inp.device), "cpu")
        self.assertEqual(str(tgt.device), "cpu")

    def test_batch_tensors_cpu_explicit(self):
        from matrixai.training.torch_trainer import _batch_tensors
        import importlib
        torch = importlib.import_module("torch")
        batch = self._make_batch()
        inp, tgt = _batch_tensors(batch, "Email", "label", "softmax_cross_entropy", ["support", "sales"], torch, "cpu")
        self.assertEqual(str(inp.device), "cpu")

    def test_batch_tensors_binary_objective_cpu(self):
        from matrixai.training.torch_trainer import _batch_tensors
        import importlib
        torch = importlib.import_module("torch")
        batch = self._make_batch()
        inp, tgt = _batch_tensors(batch, "Email", "label", "binary_cross_entropy", ["support", "sales"], torch, "cpu")
        self.assertEqual(inp.dtype, torch.float32)
        self.assertEqual(tgt.dtype, torch.float32)


# ---------------------------------------------------------------------------
# TorchSupervisedTrainer reads device from training.backend
# ---------------------------------------------------------------------------

class TestTorchSupervisedTrainerDeviceReading(unittest.TestCase):
    def test_device_read_from_backend_cpu(self):
        """Training with explicit BackendSpec cpu completes."""
        from matrixai.training.parser import parse_training_file
        from matrixai.training.spec import BackendSpec
        from matrixai.training.torch_trainer import TorchSupervisedTrainer
        import tempfile
        spec = parse_training_file(_EMAIL_TRAIN)
        spec = dataclasses.replace(spec, backend=BackendSpec(target="torch", device="cpu"))
        self.assertEqual(spec.backend.device, "cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = TorchSupervisedTrainer().train(
                spec,
                output_dir=tmpdir + "/run",
                base_path=_BASE,
                training_path=_EMAIL_TRAIN,
            )
        self.assertIsNotNone(result)

    def test_device_defaults_to_cpu_when_no_backend(self):
        """If training.backend is None, device defaults to cpu — no AttributeError."""
        from matrixai.training.parser import parse_training_file
        from matrixai.training.torch_trainer import TorchSupervisedTrainer
        import tempfile
        spec = parse_training_file(_EMAIL_TRAIN)
        self.assertIsNone(spec.backend)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = TorchSupervisedTrainer().train(
                spec,
                output_dir=tmpdir + "/run",
                base_path=_BASE,
                training_path=_EMAIL_TRAIN,
            )
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# CLI injects device into training spec via dataclasses.replace
# ---------------------------------------------------------------------------

class TestCLIDeviceInjection(unittest.TestCase):
    def test_cmd_train_injects_backend_spec_into_training(self):
        """After _cmd_train runs _validate_device and parses training, it injects BackendSpec."""
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, "-m", "matrixai", "train",
             "examples/email-agent.typed.mxai",
             "--training", str(_EMAIL_TRAIN),
             "--output", "/tmp/_p14_corte3_test_cpu_run",
             "--backend", "torch",
             "--device", "cpu"],
            cwd=_BASE, capture_output=True, text=True,
        )
        # Should succeed (exit 0)
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("Training OK", r.stdout)

    def test_cmd_train_cuda_unavailable_exits_before_training(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        if "cuda" in torch_device_info()["available_devices"]:
            self.skipTest("CUDA available")
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, "-m", "matrixai", "train",
             "examples/email-agent.typed.mxai",
             "--training", str(_EMAIL_TRAIN),
             "--output", "/tmp/_p14_corte3_test_cuda",
             "--backend", "torch",
             "--device", "cuda"],
            cwd=_BASE, capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 1)
        combined = r.stdout + r.stderr
        self.assertIn("cuda", combined.lower())

    def test_backend_metadata_reflects_device(self):
        """torch_backend_metadata(device=X) returns dict with device=X."""
        from matrixai.compiler import torch_backend_metadata
        meta = torch_backend_metadata(device="cpu")
        self.assertEqual(meta["device"], "cpu")
        meta_cuda = torch_backend_metadata(device="cuda")
        self.assertEqual(meta_cuda["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
