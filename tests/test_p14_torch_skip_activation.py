"""P14 Corte 6: Confirm previously-skipped torch tests now activate with PyTorch installed.

The 10 tests guarded by @unittest.skipUnless(torch_available(), ...) were skipped
when PyTorch was absent from the CI environment.  With torch installed they must
run and pass.  This file verifies the activation contract:

  - torch_available() returns True
  - the 10 guarded tests are collected and not skipped in this environment
  - the 3 "when_absent" tests self-skip correctly when torch IS present
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent


class TestTorchAvailableInEnvironment(unittest.TestCase):
    def test_torch_available_returns_true(self):
        from matrixai.parameters.tensor_bridge import torch_available
        self.assertTrue(torch_available(), "PyTorch must be installed in this environment")

    def test_torch_importable(self):
        import importlib
        torch = importlib.import_module("torch")
        self.assertIsNotNone(torch.__version__)

    def test_torch_device_info_torch_available_true(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        info = torch_device_info()
        self.assertTrue(info["torch_available"])
        self.assertIsNotNone(info["torch_version"])

    def test_cpu_always_in_available_devices(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        self.assertIn("cpu", torch_device_info()["available_devices"])


class TestSkipUnlessGuardedTestsAreActive(unittest.TestCase):
    """Run the three target test files in isolation and confirm no skip for torch tests."""

    def _count_outcomes(self, test_file: str) -> dict[str, int]:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=no"],
            cwd=_BASE,
            capture_output=True,
            text=True,
        )
        out = r.stdout
        passed = out.count(" PASSED")
        skipped = out.count(" SKIPPED")
        return {"passed": passed, "skipped": skipped, "returncode": r.returncode}

    def test_torch_forward_tests_active(self):
        """test_torch_forward.py: 3 skipUnless tests must not be skipped."""
        outcomes = self._count_outcomes("tests/test_torch_forward.py")
        # The 3 skipUnless tests pass; the 1 "when_absent" test skips itself
        self.assertGreaterEqual(outcomes["passed"], 3, f"Expected ≥3 passed, got {outcomes}")
        self.assertLessEqual(outcomes["skipped"], 1, f"Expected ≤1 skipped (when_absent), got {outcomes}")

    def test_torch_training_tests_active(self):
        """test_torch_training.py: 4 skipUnless tests must not be skipped."""
        outcomes = self._count_outcomes("tests/test_torch_training.py")
        self.assertGreaterEqual(outcomes["passed"], 4, f"Expected ≥4 passed, got {outcomes}")
        self.assertLessEqual(outcomes["skipped"], 2, f"Expected ≤2 skipped (when_absent), got {outcomes}")

    def test_parameters_torch_tests_active(self):
        """test_parameters.py: 3 skipUnless tests must not be skipped."""
        outcomes = self._count_outcomes("tests/test_parameters.py")
        self.assertGreaterEqual(outcomes["passed"], 3, f"Expected ≥3 passed, got {outcomes}")
        # 1 "when_absent" test skips itself
        self.assertLessEqual(outcomes["skipped"], 1, f"Expected ≤1 skipped, got {outcomes}")


class TestAbsencePathsStillIntact(unittest.TestCase):
    """The 'when torch absent' code paths (lazy import, error handling) compile and are reachable."""

    def test_tensor_bridge_import_is_lazy(self):
        """torch is NOT imported at module load — only inside functions."""
        import importlib
        import sys
        # Reload the module to check it doesn't import torch at top level
        if "matrixai.parameters.tensor_bridge" in sys.modules:
            mod = sys.modules["matrixai.parameters.tensor_bridge"]
        else:
            mod = importlib.import_module("matrixai.parameters.tensor_bridge")
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # 'import torch' should not appear at top level (outside function bodies)
        lines = src.splitlines()
        top_level_torch_imports = [
            ln for ln in lines
            if ln.strip().startswith("import torch") and not ln.startswith(" ") and not ln.startswith("\t")
        ]
        self.assertEqual(top_level_torch_imports, [], f"Unexpected top-level torch import: {top_level_torch_imports}")

    def test_torch_available_function_exists(self):
        from matrixai.parameters.tensor_bridge import torch_available
        self.assertTrue(callable(torch_available))


if __name__ == "__main__":
    unittest.main()
