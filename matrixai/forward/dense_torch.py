# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P18 C9 — Materialización opcional de NetworkSpec como torch.nn.Module."""
from __future__ import annotations

from typing import Any

from matrixai.parameters.store import ParameterSet
from matrixai.parameters.tensor_bridge import torch_available


class DenseTorchError(ValueError):
    pass


_ACTIVATION_MODULES = {
    "relu": "ReLU",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
}


def dense_network_to_torch_module(network: Any, parameter_set: ParameterSet) -> Any:
    """Build a torch.nn.Module from a NetworkSpec and load weights from a ParameterSet.

    Returns a DenseNetworkModule (subclass of nn.Module).
    Raises DenseTorchError if torch is not installed.
    """
    if not torch_available():
        raise DenseTorchError("PyTorch is not installed — C9 torch materialisation requires torch")
    import torch
    import torch.nn as nn

    linears = nn.ModuleList()
    activations: list[str] = []

    for layer in network.layers:
        w_key = f"{network.name}.W{layer.index}"
        b_key = f"{network.name}.b{layer.index}"
        if w_key not in parameter_set.parameters:
            raise DenseTorchError(f"Missing parameter {w_key!r} in ParameterSet")
        if b_key not in parameter_set.parameters:
            raise DenseTorchError(f"Missing parameter {b_key!r} in ParameterSet")

        W = parameter_set.parameters[w_key].get("values")
        b = parameter_set.parameters[b_key].get("values")
        # M15(a): las dimensiones salen del campo `shape` para poder construir el módulo
        # SIN valores (plantilla de estructura) y dejar el init nativo de nn.Linear.
        shape = parameter_set.parameters[w_key].get("shape")
        if shape and len(shape) == 2:
            out_features, in_features = int(shape[0]), int(shape[1])
        elif W is not None:
            out_features, in_features = len(W), len(W[0])
        else:
            raise DenseTorchError(f"Cannot determine shape for {w_key!r} (no shape, no values)")

        linear = nn.Linear(in_features, out_features)
        # Si la plantilla trae pesos, se cargan; si vienen None (plantilla de estructura),
        # se deja el init nativo de torch (Kaiming) — más rápido y se entrena igual.
        if W is not None and b is not None:
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(W, dtype=torch.float32))
                linear.bias.copy_(torch.tensor(b, dtype=torch.float32))

        linears.append(linear)
        activations.append(layer.activation)

    return _DenseNetworkModule(linears, activations)


def dense_torch_forward(module: Any, input_vector: list[float]) -> list[float]:
    """Run a forward pass through a DenseNetworkModule. Returns a Python list."""
    import torch
    with torch.no_grad():
        x = torch.tensor(input_vector, dtype=torch.float32)
        output = module(x)
        return output.tolist()


def torch_module_to_parameter_set(
    network: Any,
    module: Any,
    template: ParameterSet,
) -> ParameterSet:
    """Extract weights from a DenseNetworkModule back into a ParameterSet."""
    import torch
    new_params: dict[str, Any] = {}
    for key, param in template.parameters.items():
        new_params[key] = dict(param)

    for i, linear in enumerate(module.linears):
        layer = network.layers[i]
        w_key = f"{network.name}.W{layer.index}"
        b_key = f"{network.name}.b{layer.index}"
        with torch.no_grad():
            new_params[w_key] = {
                **template.parameters[w_key],
                "values": linear.weight.detach().tolist(),
            }
            new_params[b_key] = {
                **template.parameters[b_key],
                "values": linear.bias.detach().tolist(),
            }

    return ParameterSet(
        parameter_set_id=template.parameter_set_id,
        model_hash=template.model_hash,
        parameter_schema_hash=template.parameter_schema_hash,
        source="torch",
        parameters=new_params,
        metrics=template.metrics,
    )


# ---------------------------------------------------------------------------
# Internal module
# ---------------------------------------------------------------------------

class _DenseNetworkModule:
    """Wrapper that acts as a torch.nn.Module for a dense network."""

    def __init__(self, linears: Any, activations: list[str]) -> None:
        import torch.nn as nn

        # Make this a proper nn.Module by inheriting dynamically
        # We store linears as an nn.ModuleList so parameters() works
        self._linears = linears
        self._activations = activations
        self._torch_module = _build_torch_module(linears, activations)

    def __call__(self, x: Any) -> Any:
        return self._torch_module(x)

    def forward(self, x: Any) -> Any:
        return self._torch_module(x)

    def parameters(self) -> Any:
        return self._torch_module.parameters()

    def named_parameters(self) -> Any:
        return self._torch_module.named_parameters()

    def eval(self) -> "_DenseNetworkModule":
        self._torch_module.eval()
        return self

    def train(self, mode: bool = True) -> "_DenseNetworkModule":
        self._torch_module.train(mode)
        return self

    @property
    def linears(self) -> Any:
        return self._linears


def _build_torch_module(linears: Any, activations: list[str]) -> Any:
    import torch.nn as nn

    layers: list[Any] = []
    for linear, activation in zip(linears, activations):
        layers.append(linear)
        if activation == "softmax":
            layers.append(nn.Softmax(dim=-1))
        elif activation in _ACTIVATION_MODULES:
            layers.append(getattr(nn, _ACTIVATION_MODULES[activation])())
        # "linear" → no activation module

    return nn.Sequential(*layers)
