"""P14 Corte 1: hardware device detection helper tests.

torch_device_info() must:
- Be safe to call without PyTorch installed (lazy import).
- Always include 'cpu' in available_devices.
- Report torch_version, cuda_version, device_name correctly for the environment.
- Report cuda_available / mps_available as booleans.
- Be importable from matrixai.parameters.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestTorchDeviceInfoContract(unittest.TestCase):
    """Core contract: torch_device_info() always returns a well-formed dict."""

    def _info(self):
        from matrixai.parameters import torch_device_info
        return torch_device_info()

    def test_returns_dict(self):
        self.assertIsInstance(self._info(), dict)

    def test_has_torch_available_key(self):
        info = self._info()
        self.assertIn("torch_available", info)
        self.assertIsInstance(info["torch_available"], bool)

    def test_has_torch_version_key(self):
        info = self._info()
        self.assertIn("torch_version", info)

    def test_has_available_devices_list(self):
        info = self._info()
        self.assertIn("available_devices", info)
        self.assertIsInstance(info["available_devices"], list)

    def test_cpu_always_in_available_devices(self):
        self.assertIn("cpu", self._info()["available_devices"])

    def test_has_cuda_available_bool(self):
        info = self._info()
        self.assertIn("cuda_available", info)
        self.assertIsInstance(info["cuda_available"], bool)

    def test_has_mps_available_bool(self):
        info = self._info()
        self.assertIn("mps_available", info)
        self.assertIsInstance(info["mps_available"], bool)

    def test_has_cuda_version_key(self):
        self.assertIn("cuda_version", self._info())

    def test_has_device_name_key(self):
        self.assertIn("device_name", self._info())

    def test_importable_from_parameters(self):
        from matrixai.parameters import torch_device_info
        self.assertTrue(callable(torch_device_info))

    def test_importable_from_tensor_bridge(self):
        from matrixai.parameters.tensor_bridge import torch_device_info
        self.assertTrue(callable(torch_device_info))


class TestTorchDeviceInfoWithTorchInstalled(unittest.TestCase):
    """Tests for the current environment where torch IS installed."""

    @classmethod
    def setUpClass(cls):
        from matrixai.parameters import torch_available, torch_device_info
        cls.available = torch_available()
        cls.info = torch_device_info()

    def test_torch_available_is_true(self):
        self.assertTrue(self.available)

    def test_torch_version_is_string(self):
        self.assertIsInstance(self.info["torch_version"], str)
        self.assertGreater(len(self.info["torch_version"]), 0)

    def test_torch_available_true_in_info(self):
        self.assertTrue(self.info["torch_available"])

    def test_cpu_in_devices(self):
        self.assertIn("cpu", self.info["available_devices"])

    def test_cuda_absent_when_no_hardware(self):
        # In this CPU-only environment, cuda must not appear in devices
        # and cuda_available must be False.
        if self.info["cuda_available"]:
            self.assertIn("cuda", self.info["available_devices"])
        else:
            self.assertNotIn("cuda", self.info["available_devices"])
            self.assertIsNone(self.info["cuda_version"])
            self.assertIsNone(self.info["device_name"])

    def test_mps_absent_when_no_hardware(self):
        if self.info["mps_available"]:
            self.assertIn("mps", self.info["available_devices"])
        else:
            self.assertNotIn("mps", self.info["available_devices"])

    def test_available_devices_has_no_duplicates(self):
        devices = self.info["available_devices"]
        self.assertEqual(len(devices), len(set(devices)))

    def test_available_devices_only_known_values(self):
        for d in self.info["available_devices"]:
            self.assertIn(d, {"cpu", "cuda", "mps"})


class TestTorchDeviceInfoCudaPresent(unittest.TestCase):
    """Tests that run only when CUDA is available."""

    @classmethod
    def setUpClass(cls):
        from matrixai.parameters import torch_device_info
        cls.info = torch_device_info()

    @unittest.skipUnless(
        __import__("matrixai.parameters.tensor_bridge", fromlist=["torch_device_info"])
        .torch_device_info()
        .get("cuda_available"),
        "CUDA not available in this environment",
    )
    def test_cuda_in_devices(self):
        self.assertIn("cuda", self.info["available_devices"])

    @unittest.skipUnless(
        __import__("matrixai.parameters.tensor_bridge", fromlist=["torch_device_info"])
        .torch_device_info()
        .get("cuda_available"),
        "CUDA not available in this environment",
    )
    def test_cuda_version_is_string(self):
        self.assertIsInstance(self.info["cuda_version"], str)

    @unittest.skipUnless(
        __import__("matrixai.parameters.tensor_bridge", fromlist=["torch_device_info"])
        .torch_device_info()
        .get("cuda_available"),
        "CUDA not available in this environment",
    )
    def test_device_name_is_string(self):
        self.assertIsInstance(self.info["device_name"], str)
        self.assertGreater(len(self.info["device_name"]), 0)


class TestTorchDeviceInfoMpsPresent(unittest.TestCase):
    """Tests that run only when MPS is available."""

    @classmethod
    def setUpClass(cls):
        from matrixai.parameters import torch_device_info
        cls.info = torch_device_info()

    @unittest.skipUnless(
        __import__("matrixai.parameters.tensor_bridge", fromlist=["torch_device_info"])
        .torch_device_info()
        .get("mps_available"),
        "MPS not available in this environment",
    )
    def test_mps_in_devices(self):
        self.assertIn("mps", self.info["available_devices"])


class TestTorchDeviceInfoWithoutTorch(unittest.TestCase):
    """Simulate environment where torch is not installed."""

    def _info_without_torch(self):
        from matrixai.parameters import tensor_bridge
        with patch.object(tensor_bridge, "torch_available", return_value=False):
            return tensor_bridge.torch_device_info()

    def test_returns_dict_when_torch_absent(self):
        self.assertIsInstance(self._info_without_torch(), dict)

    def test_torch_available_false_when_absent(self):
        self.assertFalse(self._info_without_torch()["torch_available"])

    def test_cpu_still_in_devices_when_torch_absent(self):
        self.assertIn("cpu", self._info_without_torch()["available_devices"])

    def test_torch_version_none_when_absent(self):
        self.assertIsNone(self._info_without_torch()["torch_version"])

    def test_cuda_available_false_when_torch_absent(self):
        self.assertFalse(self._info_without_torch()["cuda_available"])

    def test_mps_available_false_when_torch_absent(self):
        self.assertFalse(self._info_without_torch()["mps_available"])


if __name__ == "__main__":
    unittest.main()
