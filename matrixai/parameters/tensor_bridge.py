# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module, util
from typing import Any, Mapping

from matrixai.parameters.store import ParameterSet


class TensorParameterBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class TensorParameterBridge:
    device: str = "cpu"
    dtype: str = "float32"

    def to_torch_tensors(self, parameter_set: ParameterSet) -> dict[str, Any]:
        self._validate_config()
        errors = validate_parameter_set_for_torch(parameter_set)
        if errors:
            raise TensorParameterBridgeError("; ".join(errors))
        torch = _import_torch()
        dtype = _torch_dtype(torch, self.dtype)

        tensors: dict[str, Any] = {}
        for name, parameter in parameter_set.parameters.items():
            tensor = torch.tensor(parameter.get("values"), dtype=dtype, device=self.device)
            expected_shape = list(parameter.get("shape", []))
            actual_shape = list(tensor.shape)
            if actual_shape != expected_shape:
                raise TensorParameterBridgeError(
                    f"Parameter {name} expected tensor shape {expected_shape}, got {actual_shape}"
                )
            tensors[name] = tensor
            function = parameter.get("function")
            if function:
                tensors[f"{function}.{name}"] = tensor
        return tensors

    def from_torch_tensors(
        self,
        template: ParameterSet,
        tensors: Mapping[str, Any],
        *,
        parameter_set_id: str | None = None,
        source: str = "torch",
        metrics: dict[str, Any] | None = None,
    ) -> ParameterSet:
        self._validate_config()
        torch = _import_torch()
        data = template.to_dict()
        if parameter_set_id is not None:
            data["parameter_set_id"] = parameter_set_id
        data["source"] = source
        if metrics is not None:
            data["metrics"] = deepcopy(metrics)

        errors: list[str] = []
        for name, parameter in data["parameters"].items():
            tensor = _lookup_tensor(name, parameter, tensors)
            if tensor is None:
                errors.append(f"Missing tensor for parameter {name}")
                continue
            errors.extend(_torch_tensor_errors(name, tensor, parameter, torch, self.dtype))
            if errors:
                continue
            parameter["values"] = _tensor_to_json_value(tensor)

        if errors:
            raise TensorParameterBridgeError("; ".join(errors))
        return ParameterSet.from_dict(data)

    def _validate_config(self) -> None:
        if self.device != "cpu":
            raise TensorParameterBridgeError("P5 tensor bridge supports only device='cpu'")
        if self.dtype != "float32":
            raise TensorParameterBridgeError("P5 tensor bridge supports only dtype='float32'")


def parameter_set_to_torch_tensors(parameter_set: ParameterSet) -> dict[str, Any]:
    return TensorParameterBridge().to_torch_tensors(parameter_set)


def torch_tensors_to_parameter_set(
    template: ParameterSet,
    tensors: Mapping[str, Any],
    *,
    parameter_set_id: str | None = None,
    source: str = "torch",
    metrics: dict[str, Any] | None = None,
) -> ParameterSet:
    return TensorParameterBridge().from_torch_tensors(
        template,
        tensors,
        parameter_set_id=parameter_set_id,
        source=source,
        metrics=metrics,
    )


def validate_parameter_set_for_torch(parameter_set: ParameterSet) -> list[str]:
    errors: list[str] = []
    for name, parameter in parameter_set.parameters.items():
        expected_dtype = parameter.get("dtype", "float32")
        if expected_dtype != "float32":
            errors.append(f"Parameter {name} expected dtype float32, got {expected_dtype}")
        expected_shape = list(parameter.get("shape", []))
        try:
            actual_shape = _value_shape(parameter.get("values"))
        except ValueError as exc:
            errors.append(f"Parameter {name} invalid: {exc}")
            continue
        if actual_shape != expected_shape:
            errors.append(f"Parameter {name} expected values shape {expected_shape}, got {actual_shape}")
    return errors


def torch_available() -> bool:
    return util.find_spec("torch") is not None


def enable_tf32_if_cuda(device: str) -> bool:
    """M15(c) — habilita TF32 (tensor cores) en matmul/cudnn cuando se entrena en CUDA.

    En Ampere+ (A100, RTX 30/40/Ada), TF32 multiplica el throughput de las matmul fp32
    ~5-8x con pérdida de precisión despreciable para entrenamiento (mantisa de 10 bits).
    SÓLO afecta a CUDA: el camino CPU y el stdlib determinista quedan intactos, así que no
    toca la reproducibilidad/auditoría del suelo determinista. El camino torch/GPU es el
    "techo de velocidad", no la fuente de verdad. Desactivable con MATRIXAI_GPU_TF32=0.

    Devuelve True si lo activó. Idempotente y barato (flags globales de torch).
    """
    import os
    if os.environ.get("MATRIXAI_GPU_TF32", "1") == "0":
        return False
    if not str(device).startswith("cuda") or not torch_available():
        return False
    try:
        torch = import_module("torch")
        if not torch.cuda.is_available():
            return False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception:  # noqa: BLE001
        return False


def torch_device_info() -> dict[str, Any]:
    """Report available hardware devices and PyTorch version. Safe to call without torch installed."""
    if not torch_available():
        return {
            "torch_available": False,
            "torch_version": None,
            "available_devices": ["cpu"],
            "cuda_available": False,
            "cuda_version": None,
            "device_name": None,
            "mps_available": False,
        }
    try:
        torch = import_module("torch")
    except Exception:  # noqa: BLE001
        return {
            "torch_available": False,
            "torch_version": None,
            "available_devices": ["cpu"],
            "cuda_available": False,
            "cuda_version": None,
            "device_name": None,
            "mps_available": False,
        }
    devices: list[str] = ["cpu"]
    cuda_available = bool(torch.cuda.is_available())
    cuda_version: str | None = None
    device_name: str | None = None
    if cuda_available:
        devices.append("cuda")
        cuda_version = getattr(torch.version, "cuda", None)
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            device_name = None
    mps_available = bool(
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if mps_available:
        devices.append("mps")
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "available_devices": devices,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "mps_available": mps_available,
    }


def _import_torch():
    if not torch_available():
        raise TensorParameterBridgeError(
            "TensorParameterBridge requires optional dependency PyTorch to materialize tensors"
        )
    try:
        return import_module("torch")
    except Exception as exc:  # noqa: BLE001
        raise TensorParameterBridgeError(f"Unable to import optional dependency PyTorch: {exc}") from exc


def _torch_dtype(torch, dtype: str):
    if dtype == "float32":
        return torch.float32
    raise TensorParameterBridgeError(f"Unsupported torch bridge dtype {dtype!r}")


def _lookup_tensor(name: str, parameter: dict[str, Any], tensors: Mapping[str, Any]) -> Any | None:
    if name in tensors:
        return tensors[name]
    function = parameter.get("function")
    if function:
        qualified_name = f"{function}.{name}"
        if qualified_name in tensors:
            return tensors[qualified_name]
    return None


def _torch_tensor_errors(name: str, tensor: Any, parameter: dict[str, Any], torch, dtype: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(tensor, torch.Tensor):
        return [f"Parameter {name} must be a torch.Tensor"]
    expected_shape = list(parameter.get("shape", []))
    actual_shape = list(tensor.shape)
    if actual_shape != expected_shape:
        errors.append(f"Parameter {name} expected tensor shape {expected_shape}, got {actual_shape}")
    expected_dtype = _torch_dtype(torch, dtype)
    if tensor.dtype != expected_dtype:
        errors.append(f"Parameter {name} expected tensor dtype {expected_dtype}, got {tensor.dtype}")
    if tensor.device.type != "cpu":
        errors.append(f"Parameter {name} expected CPU tensor, got device {tensor.device}")
    return errors


def _tensor_to_json_value(tensor: Any) -> Any:
    return tensor.detach().cpu().tolist()


def _value_shape(value: Any) -> list[int]:
    if isinstance(value, list):
        if not value:
            return [0]
        first_shape = _value_shape(value[0])
        for item in value[1:]:
            if _value_shape(item) != first_shape:
                raise ValueError("Parameter contains ragged values")
        return [len(value)] + first_shape
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter contains non-numeric value {value!r}") from exc
    return []