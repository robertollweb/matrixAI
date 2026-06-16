"""P14 Corte 2: BackendSpec, BACKEND block in .mxtrain, and --device CLI flag tests."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_EMAIL_TRAIN = _BASE / "examples" / "email-agent.supervised.mxtrain"

_MINIMAL_MXTRAIN = """\
MODEL examples/email-agent.typed.mxai

DATASET D
  SOURCE csv("examples/email-agent.train.csv")
  INPUT Email FROM COLUMNS [urgency]
  TARGET label: Label[support, sales]
END

LOSS L
  TYPE CrossEntropy
  PREDICTION logits
  TARGET label
END

OPTIMIZER O
  TYPE SGD
  LEARNING_RATE 0.01
  UPDATE *
END
"""


# ---------------------------------------------------------------------------
# BackendSpec dataclass contract
# ---------------------------------------------------------------------------

class TestBackendSpecContract(unittest.TestCase):
    def test_importable_from_training(self):
        from matrixai.training import BackendSpec
        self.assertIsNotNone(BackendSpec)

    def test_importable_from_spec(self):
        from matrixai.training.spec import BackendSpec
        self.assertIsNotNone(BackendSpec)

    def test_defaults_are_stdlib_cpu(self):
        from matrixai.training import BackendSpec
        b = BackendSpec()
        self.assertEqual(b.target, "stdlib")
        self.assertEqual(b.device, "cpu")

    def test_to_dict_has_target_and_device(self):
        from matrixai.training import BackendSpec
        d = BackendSpec(target="torch", device="cpu").to_dict()
        self.assertEqual(d["target"], "torch")
        self.assertEqual(d["device"], "cpu")

    def test_is_frozen(self):
        from matrixai.training import BackendSpec
        b = BackendSpec()
        with self.assertRaises((AttributeError, TypeError)):
            b.device = "cuda"  # type: ignore[misc]

    def test_invalid_target_raises(self):
        from matrixai.training import BackendSpec
        with self.assertRaises(ValueError):
            BackendSpec(target="jax")

    def test_invalid_device_raises(self):
        from matrixai.training import BackendSpec
        with self.assertRaises(ValueError):
            BackendSpec(device="gpu")

    def test_valid_combinations(self):
        from matrixai.training import BackendSpec
        valid = [
            ("stdlib", "cpu"),
            ("torch", "cpu"),
            ("torch", "cuda"),
            ("torch", "mps"),
        ]
        for target, device in valid:
            b = BackendSpec(target=target, device=device)
            self.assertEqual(b.target, target)
            self.assertEqual(b.device, device)

    def test_stdlib_non_cpu_raises(self):
        from matrixai.training import BackendSpec
        for device in ("cuda", "mps"):
            with self.assertRaises(ValueError):
                BackendSpec(target="stdlib", device=device)


# ---------------------------------------------------------------------------
# TrainingSpec backward compat: backend is None when block absent
# ---------------------------------------------------------------------------

class TestTrainingSpecBackendField(unittest.TestCase):
    def test_backend_none_when_block_absent(self):
        from matrixai.training.parser import parse_training_text
        spec = parse_training_text(_MINIMAL_MXTRAIN)
        self.assertIsNone(spec.backend)

    def test_to_dict_has_no_backend_key_when_none(self):
        from matrixai.training.parser import parse_training_text
        d = parse_training_text(_MINIMAL_MXTRAIN).to_dict()
        self.assertNotIn("backend", d)

    def test_real_mxtrain_files_have_no_backend(self):
        from matrixai.training.parser import parse_training_file
        spec = parse_training_file(_EMAIL_TRAIN)
        self.assertIsNone(spec.backend)


# ---------------------------------------------------------------------------
# BACKEND block parser
# ---------------------------------------------------------------------------

class TestBackendBlockParser(unittest.TestCase):
    def _parse(self, backend_block: str):
        from matrixai.training.parser import parse_training_text
        return parse_training_text(_MINIMAL_MXTRAIN + "\n" + backend_block)

    def test_backend_target_torch_device_cpu(self):
        spec = self._parse("BACKEND\n  TARGET torch\n  DEVICE cpu\nEND\n")
        self.assertIsNotNone(spec.backend)
        self.assertEqual(spec.backend.target, "torch")  # type: ignore[union-attr]
        self.assertEqual(spec.backend.device, "cpu")    # type: ignore[union-attr]

    def test_backend_target_torch_device_cuda(self):
        spec = self._parse("BACKEND\n  TARGET torch\n  DEVICE cuda\nEND\n")
        self.assertEqual(spec.backend.target, "torch")  # type: ignore[union-attr]
        self.assertEqual(spec.backend.device, "cuda")   # type: ignore[union-attr]

    def test_backend_target_torch_device_mps(self):
        spec = self._parse("BACKEND\n  TARGET torch\n  DEVICE mps\nEND\n")
        self.assertEqual(spec.backend.device, "mps")    # type: ignore[union-attr]

    def test_backend_target_stdlib(self):
        spec = self._parse("BACKEND\n  TARGET stdlib\n  DEVICE cpu\nEND\n")
        self.assertEqual(spec.backend.target, "stdlib")  # type: ignore[union-attr]

    def test_backend_default_device_cpu_when_only_target(self):
        spec = self._parse("BACKEND\n  TARGET torch\nEND\n")
        self.assertEqual(spec.backend.device, "cpu")  # type: ignore[union-attr]

    def test_backend_default_target_stdlib_when_only_device(self):
        spec = self._parse("BACKEND\n  DEVICE cpu\nEND\n")
        self.assertEqual(spec.backend.target, "stdlib")  # type: ignore[union-attr]

    def test_backend_in_to_dict(self):
        spec = self._parse("BACKEND\n  TARGET torch\n  DEVICE cpu\nEND\n")
        d = spec.to_dict()
        self.assertIn("backend", d)
        self.assertEqual(d["backend"]["target"], "torch")
        self.assertEqual(d["backend"]["device"], "cpu")

    def test_invalid_target_raises_parse_error(self):
        from matrixai.training.parser import MatrixAITrainingParseError
        with self.assertRaises(MatrixAITrainingParseError):
            self._parse("BACKEND\n  TARGET jax\nEND\n")

    def test_invalid_device_raises_parse_error(self):
        from matrixai.training.parser import MatrixAITrainingParseError
        with self.assertRaises(MatrixAITrainingParseError):
            self._parse("BACKEND\n  TARGET torch\n  DEVICE gpu\nEND\n")

    def test_unknown_backend_line_raises_parse_error(self):
        from matrixai.training.parser import MatrixAITrainingParseError
        with self.assertRaises(MatrixAITrainingParseError):
            self._parse("BACKEND\n  OPTIMIZER sgd\nEND\n")


# ---------------------------------------------------------------------------
# CLI --device flag: validation logic
# ---------------------------------------------------------------------------

class TestCLIDeviceValidation(unittest.TestCase):
    def _run(self, *args: str) -> tuple[int, str]:
        r = subprocess.run(
            [sys.executable, "-m", "matrixai", *args],
            cwd=_BASE, capture_output=True, text=True,
        )
        return r.returncode, r.stdout + r.stderr

    def test_train_help_shows_device(self):
        rc, out = self._run("train", "--help")
        self.assertEqual(rc, 0)
        self.assertIn("--device", out)

    def test_evaluate_help_shows_device(self):
        rc, out = self._run("evaluate", "--help")
        self.assertEqual(rc, 0)
        self.assertIn("--device", out)

    def test_device_cuda_without_torch_backend_exits_1(self):
        # Passing --device cuda with --backend stdlib must fail before touching disk
        rc, out = self._run(
            "train",
            "examples/email-agent.typed.mxai",
            "--training", str(_EMAIL_TRAIN),
            "--output", "/tmp/_p14_test_should_not_exist",
            "--backend", "stdlib",
            "--device", "cuda",
        )
        self.assertEqual(rc, 1)
        self.assertIn("cuda", out.lower())

    def test_device_mps_without_torch_backend_exits_1(self):
        rc, out = self._run(
            "train",
            "examples/email-agent.typed.mxai",
            "--training", str(_EMAIL_TRAIN),
            "--output", "/tmp/_p14_test_should_not_exist",
            "--backend", "stdlib",
            "--device", "mps",
        )
        self.assertEqual(rc, 1)

    def test_device_cpu_with_stdlib_backend_is_default_and_valid(self):
        # cpu + stdlib is always valid — argparse should accept it
        rc, out = self._run("train", "--help")
        self.assertEqual(rc, 0)

    @unittest.skipIf(
        __import__("matrixai.parameters.tensor_bridge", fromlist=["torch_device_info"])
        .torch_device_info()
        .get("cuda_available"),
        "CUDA is available — skip the 'not available' error test",
    )
    def test_device_cuda_unavailable_exits_1(self):
        """When cuda hardware is absent, --device cuda must fail with exit 1."""
        rc, out = self._run(
            "train",
            "examples/email-agent.typed.mxai",
            "--training", str(_EMAIL_TRAIN),
            "--output", "/tmp/_p14_test_should_not_exist",
            "--backend", "torch",
            "--device", "cuda",
        )
        self.assertEqual(rc, 1)
        self.assertIn("cuda", out.lower())
        self.assertIn("not available", out.lower())

    @unittest.skipIf(
        __import__("matrixai.parameters.tensor_bridge", fromlist=["torch_device_info"])
        .torch_device_info()
        .get("mps_available"),
        "MPS is available — skip the 'not available' error test",
    )
    def test_device_mps_unavailable_exits_1(self):
        rc, out = self._run(
            "train",
            "examples/email-agent.typed.mxai",
            "--training", str(_EMAIL_TRAIN),
            "--output", "/tmp/_p14_test_should_not_exist",
            "--backend", "torch",
            "--device", "mps",
        )
        self.assertEqual(rc, 1)
        self.assertIn("mps", out.lower())


if __name__ == "__main__":
    unittest.main()
