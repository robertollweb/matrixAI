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


def dense_torch_forward(module: Any, input_vector: list[float], device: str = "cpu") -> list[float]:
    """Run a forward pass through a DenseNetworkModule. Returns a Python list.

    PESOS_GRANDES C7a: `device` (default "cpu", retro-compat) debe coincidir
    con el dispositivo del módulo — si `module` vive en CUDA y el input se
    crea en CPU, el forward revienta por mismatch de dispositivo."""
    import torch
    with torch.no_grad():
        x = torch.tensor(input_vector, dtype=torch.float32, device=device)
        output = module(x)
        return output.tolist()


def dense_module_to_state_dict(network: Any, module: Any) -> dict[str, Any]:
    """PESOS_GRANDES C2 — pesos entrenados como tensores CPU, NUNCA como listas
    Python. Claves iguales a las de la `ParameterSet` (`{network}.W{i}`/`b{i}`)
    para que `dense_network_to_torch_module_from_state` y
    `materialize_parameter_set` puedan consumirlas sin volver a tocar el módulo
    torch. `.detach().cpu().clone()` es una copia de tensor (vectorizada, C a
    C) — cero iteración Python sobre valores individuales."""
    state: dict[str, Any] = {}
    for i, linear in enumerate(module.linears):
        layer = network.layers[i]
        state[f"{network.name}.W{layer.index}"] = linear.weight.detach().cpu().clone()
        state[f"{network.name}.b{layer.index}"] = linear.bias.detach().cpu().clone()
    return state


def dense_network_to_torch_module_from_state(
    network: Any,
    state: dict[str, Any],
    device: str = "cpu",
) -> Any:
    """PESOS_GRANDES C2 — reconstruye el módulo directamente desde tensores
    entrenados (`dense_module_to_state_dict`), SIN pasar por una `ParameterSet`
    con valores. Las dimensiones salen del propio tensor (`W.shape`), no de un
    `template` — el state_dict ya es la fuente de verdad de la arquitectura
    entrenada. Usado por `evaluate_dense_network_torch`/`probe_collapse_torch`
    cuando el trainer no materializó (modelo grande, ver `estimate_model_resources`)."""
    if not torch_available():
        raise DenseTorchError("PyTorch is not installed — requires torch")
    import torch
    import torch.nn as nn

    linears = nn.ModuleList()
    activations: list[str] = []
    for layer in network.layers:
        w_key = f"{network.name}.W{layer.index}"
        b_key = f"{network.name}.b{layer.index}"
        if w_key not in state or b_key not in state:
            raise DenseTorchError(f"Missing tensor {w_key!r}/{b_key!r} in state")
        W, b = state[w_key], state[b_key]
        out_features, in_features = int(W.shape[0]), int(W.shape[1])
        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(W.to(device))
            linear.bias.copy_(b.to(device))
        linears.append(linear)
        activations.append(layer.activation)

    module = _DenseNetworkModule(linears, activations)
    module._torch_module.to(device)
    return module


def build_parameter_template_for_state(program: Any) -> tuple[Any, ParameterSet]:
    """PESOS_GRANDES C4 — reconstruye `(network, template)` para un programa
    dense_network ya parseado, listos para `materialize_parameter_set` o para
    sacar `model_hash`/`parameter_schema_hash` al escribir un `.mxw`.

    Reusa el mismo camino que el trainer (`check_network_types` +
    `build_network_parameter_set(..., with_values=False)`) para no duplicar la
    lógica de resolución de shapes en cada caller (Studio save-as-json,
    save-as-binary, y en C5 el load). `with_values=False` es una plantilla de
    estructura — el seed no genera valores, así que un valor fijo (0) es
    correcto: los VALORES reales siempre vienen de `state` (los tensores
    entrenados), nunca de esta plantilla."""
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash

    net = program.networks[0]
    vector_map = {v.name: v for v in program.vectors}
    type_result = check_network_types(net, vector_map)
    resolved_layers = type_result.resolved_layers if type_result.resolved_layers else net.layers
    template = build_network_parameter_set(
        net, resolved_layers, program_hash(program), seed=0, with_values=False,
    )
    return net, template


def materialize_parameter_set(
    network: Any,
    state: dict[str, Any],
    template: ParameterSet,
) -> ParameterSet:
    """PESOS_GRANDES C2 — la ÚNICA puerta al `tolist()` (O(#params) en Python).

    Llamarla es una decisión explícita del caller (C4: el usuario eligió
    `weights_format=json`), nunca un paso automático del trainer/eval/probe.
    `state` son tensores CPU (`dense_module_to_state_dict`); el resultado es
    una `ParameterSet` completa, idéntica en forma a la que producía
    `torch_module_to_parameter_set` (que ahora reusa esta función)."""
    new_params: dict[str, Any] = {key: dict(param) for key, param in template.parameters.items()}
    for key, tensor in state.items():
        new_params[key] = {**template.parameters[key], "values": tensor.tolist()}
    return ParameterSet(
        parameter_set_id=template.parameter_set_id,
        model_hash=template.model_hash,
        parameter_schema_hash=template.parameter_schema_hash,
        source="torch",
        parameters=new_params,
        metrics=template.metrics,
    )


def torch_module_to_parameter_set(
    network: Any,
    module: Any,
    template: ParameterSet,
) -> ParameterSet:
    """Extract weights from a DenseNetworkModule back into a ParameterSet.

    PESOS_GRANDES C2: reusa `dense_module_to_state_dict` + `materialize_parameter_set`
    (mismo resultado que antes — cero cambio de comportamiento para callers
    existentes; el `tolist()` sigue pasando por una única función explícita)."""
    state = dense_module_to_state_dict(network, module)
    return materialize_parameter_set(network, state, template)


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
