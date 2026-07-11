# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P19 C9 — Materialización opcional de composite_network como torch.nn.Module."""
from __future__ import annotations

from typing import Any

from matrixai.ir.schema import get_interleaved_body
from matrixai.parameters.store import ParameterSet
from matrixai.parameters.tensor_bridge import torch_available


COMPOSITE_TORCH_FORWARD_ATOL: float = 1e-4  # contract: torch forward ≈ stdlib with atol=1e-4


class CompositeTorchError(ValueError):
    pass


def composite_network_to_torch_module(
    network: Any,
    parameter_set: ParameterSet,
) -> Any:
    """Build a torch.nn.Module from a composite NetworkSpec and load weights.

    Raises CompositeTorchError if torch is not installed or a required
    parameter is missing from parameter_set.
    """
    if not torch_available():
        raise CompositeTorchError(
            "PyTorch is not installed — C9 torch materialisation requires torch"
        )
    # TRANSFORMER C2 (auditoría): this builder does not know the transformer
    # block yet — building the module anyway would silently DROP the block
    # (a fail-open worse than an error). The torch path lands in C4.
    if getattr(network, "transformer_blocks", []):
        raise CompositeTorchError(
            f"composite_network_to_torch_module: NETWORK {network.name} contains a "
            f"BLOCK TRANSFORMER — the torch module for it lands in TRANSFORMER_BLOQUE "
            f"C4; building the module now would silently omit the block"
        )
    import torch
    import torch.nn as nn

    # --- Embedding modules ---
    embedding_modules: dict[str, Any] = {}
    for emb in getattr(network, "embeddings", []):
        table_key = f"{network.name}.{emb.name}.table"
        if table_key not in parameter_set.parameters:
            raise CompositeTorchError(f"Missing embedding table parameter: {table_key!r}")
        # M15(f): dims desde shape; copiar la tabla solo si la plantilla trae valores
        # (with_values=False → init nativo de nn.Embedding).
        table = parameter_set.parameters[table_key].get("values")
        shape = parameter_set.parameters[table_key].get("shape")
        vocab, dim = (int(shape[0]), int(shape[1])) if shape else (len(table), len(table[0]))
        emb_mod = nn.Embedding(vocab, dim)
        if table is not None:
            with torch.no_grad():
                emb_mod.weight.data = torch.tensor(table, dtype=torch.float32)
        embedding_modules[emb.name] = emb_mod

    # --- Body items (interleaved top_layers and blocks) ---
    body_items = get_interleaved_body(network)

    # --- Per-layer modules ---
    # Keys use "__" to separate path segments so nn.ModuleDict accepts them.
    layer_modules: dict[str, Any] = {}
    for _, kind, spec in body_items:
        if kind == "layer":
            _build_layer_module(layer_modules, spec, network.name, f"L{spec.index}", parameter_set)
        else:
            block = spec
            block_prefix = f"{network.name}.{block.name}"
            for layer in block.layers:
                key = f"{block.name}__L{layer.index}"
                _build_layer_module(layer_modules, layer, block_prefix, f"L{layer.index}", parameter_set, dict_key=key)

    # --- Structural specs for forward ---
    embed_specs = [(emb.name, emb.source) for emb in getattr(network, "embeddings", [])]
    concat_specs = [(c.name, list(c.sources)) for c in getattr(network, "concats", [])]

    return _CompositeNetworkModule(
        embedding_modules=embedding_modules,
        layer_modules=layer_modules,
        embed_specs=embed_specs,
        concat_specs=concat_specs,
        body_items=body_items,
    )


def composite_torch_forward(
    module: Any,
    input_data: dict[str, Any],
    training: bool = False,
) -> list[float]:
    """Run a forward pass through a _CompositeNetworkModule. Returns a Python list."""
    import torch
    module_mode = module.training
    if training:
        module.train()
    else:
        module.eval()
    with torch.no_grad():
        result = module.forward_with_dict(input_data)
    if training != module_mode:
        module.train(module_mode)
    return result.tolist()


def composite_torch_forward_batch(
    module: Any,
    batch: list[dict[str, Any]],
    training: bool = False,
) -> list[list[float]]:
    """M15(e) — batched forward over a list of input dicts → list of output rows.

    Same semantics as calling composite_torch_forward per sample, but in a single
    batched pass (one kernel per layer instead of one per sample). Used by the torch
    composite evaluator. Returns a Python list of lists.
    """
    import torch
    if not batch:
        return []
    module_mode = module.training
    if training:
        module.train()
    else:
        module.eval()
    with torch.no_grad():
        result = module.forward_batch(batch)
    if training != module_mode:
        module.train(module_mode)
    return result.detach().cpu().tolist()


def torch_module_to_composite_parameter_set(
    network: Any,
    module: Any,
    template: ParameterSet,
) -> ParameterSet:
    """Extract weights from a _CompositeNetworkModule back into a ParameterSet."""
    new_params: dict[str, Any] = {k: dict(v) for k, v in template.parameters.items()}

    for emb in getattr(network, "embeddings", []):
        table_key = f"{network.name}.{emb.name}.table"
        if table_key in new_params:
            emb_mod = module.embeddings[emb.name]
            new_params[table_key] = {
                **template.parameters[table_key],
                "values": emb_mod.weight.detach().tolist(),
            }

    body_items = get_interleaved_body(network)

    for _, kind, spec in body_items:
        if kind == "layer":
            _extract_layer_weights(new_params, spec, network.name, f"L{spec.index}", module, template)
        else:
            block = spec
            for layer in block.layers:
                key = f"{block.name}__L{layer.index}"
                _extract_layer_weights(new_params, layer, f"{network.name}.{block.name}", f"L{layer.index}", module, template, dict_key=key)

    return ParameterSet(
        parameter_set_id=template.parameter_set_id,
        model_hash=template.model_hash,
        parameter_schema_hash=template.parameter_schema_hash,
        source="torch",
        parameters=new_params,
        metrics=template.metrics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_layer_module(
    layer_modules: dict[str, Any],
    layer: Any,
    param_prefix: str,
    param_key: str,
    parameter_set: ParameterSet,
    dict_key: str | None = None,
) -> None:
    """Add an nn.Module for a single layer into layer_modules dict."""
    import torch
    import torch.nn as nn

    lt = layer.layer_type
    key = dict_key if dict_key is not None else param_key

    if lt == "Dense":
        w_path = f"{param_prefix}.{param_key}.W"
        b_path = f"{param_prefix}.{param_key}.b"
        if w_path not in parameter_set.parameters:
            raise CompositeTorchError(f"Missing parameter {w_path!r}")
        if b_path not in parameter_set.parameters:
            raise CompositeTorchError(f"Missing parameter {b_path!r}")
        # M15(f): dims desde shape; copiar pesos solo si la plantilla los trae
        # (with_values=False → init nativo de nn.Linear, sembrado por torch.manual_seed).
        W = parameter_set.parameters[w_path].get("values")
        b = parameter_set.parameters[b_path].get("values")
        w_shape = parameter_set.parameters[w_path].get("shape")
        out_f, in_f = (int(w_shape[0]), int(w_shape[1])) if w_shape else (len(W), len(W[0]))
        linear = nn.Linear(in_f, out_f)
        if W is not None and b is not None:
            with torch.no_grad():
                linear.weight.data = torch.tensor(W, dtype=torch.float32)
                linear.bias.data = torch.tensor(b, dtype=torch.float32)
        layer_modules[key] = linear

    elif lt == "LayerNorm":
        g_path = f"{param_prefix}.{param_key}.gamma"
        b_path = f"{param_prefix}.{param_key}.beta"
        if g_path not in parameter_set.parameters:
            raise CompositeTorchError(f"Missing parameter {g_path!r}")
        # M15(f): dims desde shape; copiar gamma/beta solo si la plantilla los trae.
        gamma = parameter_set.parameters[g_path].get("values")
        beta = parameter_set.parameters[b_path].get("values")
        g_shape = parameter_set.parameters[g_path].get("shape")
        features = int(g_shape[0]) if g_shape else len(gamma)
        ln = nn.LayerNorm(features, eps=1e-5, elementwise_affine=True)
        if gamma is not None and beta is not None:
            with torch.no_grad():
                ln.weight.data = torch.tensor(gamma, dtype=torch.float32)
                ln.bias.data = torch.tensor(beta, dtype=torch.float32)
        layer_modules[key] = ln

    elif lt == "Dropout":
        rate = getattr(layer, "rate", 0.0)
        layer_modules[key] = nn.Dropout(p=rate)


def _extract_layer_weights(
    new_params: dict[str, Any],
    layer: Any,
    param_prefix: str,
    param_key: str,
    module: Any,
    template: ParameterSet,
    dict_key: str | None = None,
) -> None:
    lt = layer.layer_type
    key = dict_key if dict_key is not None else param_key
    if key not in module.sublayers:
        return

    if lt == "Dense":
        w_path = f"{param_prefix}.{param_key}.W"
        b_path = f"{param_prefix}.{param_key}.b"
        linear = module.sublayers[key]
        if w_path in template.parameters:
            new_params[w_path] = {**template.parameters[w_path], "values": linear.weight.detach().tolist()}
        if b_path in template.parameters:
            new_params[b_path] = {**template.parameters[b_path], "values": linear.bias.detach().tolist()}

    elif lt == "LayerNorm":
        g_path = f"{param_prefix}.{param_key}.gamma"
        b_path = f"{param_prefix}.{param_key}.beta"
        ln = module.sublayers[key]
        if g_path in template.parameters:
            new_params[g_path] = {**template.parameters[g_path], "values": ln.weight.detach().tolist()}
        if b_path in template.parameters:
            new_params[b_path] = {**template.parameters[b_path], "values": ln.bias.detach().tolist()}


def _apply_torch_activation(z: Any, name: str) -> Any:
    import torch
    import torch.nn.functional as F
    if name == "relu":
        return F.relu(z)
    elif name == "sigmoid":
        return torch.sigmoid(z)
    elif name == "tanh":
        return torch.tanh(z)
    elif name == "softmax":
        return F.softmax(z, dim=-1)
    elif name == "gelu":
        return F.gelu(z)
    elif name == "linear" or name is None:
        return z
    return z


# ---------------------------------------------------------------------------
# nn.Module implementation
# ---------------------------------------------------------------------------

class _CompositeNetworkModule:
    """Wraps composite network logic as a torch.nn.Module-compatible object."""

    def __init__(
        self,
        embedding_modules: dict[str, Any],
        layer_modules: dict[str, Any],
        embed_specs: list[tuple[str, str]],
        concat_specs: list[tuple[str, list[str]]],
        body_items: list[tuple[int, str, Any]],
    ) -> None:
        import torch.nn as nn

        # Register via proper nn.Module so parameters() works
        self._module = _TorchCompositeModule(embedding_modules, layer_modules)
        self._embed_specs = embed_specs
        self._concat_specs = concat_specs
        self._body_items = body_items

    @property
    def embeddings(self) -> Any:
        return self._module.embeddings

    @property
    def sublayers(self) -> Any:
        return self._module.sublayers

    @property
    def training(self) -> bool:
        return self._module.training

    def train(self, mode: bool = True) -> "_CompositeNetworkModule":
        self._module.train(mode)
        return self

    def eval(self) -> "_CompositeNetworkModule":
        self._module.eval()
        return self

    def to(self, device: Any) -> "_CompositeNetworkModule":
        self._module.to(device)
        return self

    def parameters(self) -> Any:
        return self._module.parameters()

    def named_parameters(self) -> Any:
        return self._module.named_parameters()

    def forward_with_dict(self, input_data: dict[str, Any]) -> Any:
        import torch

        # Device-aware: place input tensors on the same device as the module's
        # parameters so GPU (CUDA) training/inference works, not only CPU.
        try:
            device = next(self._module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # 1. Convert input fields to tensors
        named: dict[str, Any] = {}
        for name, value in input_data.items():
            if isinstance(value, (int, float)):
                named[name] = torch.tensor([float(value)], dtype=torch.float32, device=device)
            else:
                named[name] = torch.tensor([float(v) for v in value], dtype=torch.float32, device=device)

        # 2. Embedding lookups
        for emb_name, source in self._embed_specs:
            idx = named[source].long().squeeze(0)
            named[emb_name] = self._module.embeddings[emb_name](idx)

        # 3. Concats
        for concat_name, sources in self._concat_specs:
            parts = [named[s] for s in sources]
            named[concat_name] = torch.cat(parts, dim=-1)

        # 4. Initial current vector
        emb_names = {e[0] for e in self._embed_specs}
        if self._concat_specs:
            current = named[self._concat_specs[-1][0]]
        else:
            parts = [named[k] for k in input_data if k not in emb_names]
            current = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        # 5. Body items in textual order
        for _, kind, spec in self._body_items:
            if kind == "layer":
                key = f"L{spec.index}"
                current = self._apply_layer(spec, current, key)
            else:
                block = spec
                residual_from = getattr(block, "residual_from", "")
                if residual_from == "PREVIOUS":
                    skip = current
                elif residual_from:
                    skip = named.get(residual_from)
                else:
                    skip = None
                for layer in block.layers:
                    key = f"{block.name}__L{layer.index}"
                    current = self._apply_layer(layer, current, key)
                if skip is not None:
                    current = current + skip

        return current

    def forward_batch(self, batch: list[dict[str, Any]]) -> Any:
        """M15(e) — batched forward over a list of input dicts → tensor (batch, out).

        Mirrors forward_with_dict but carries a leading batch dimension through the
        whole body. nn.Linear / LayerNorm / activations operate on the last dim, so
        the body loop and _apply_layer work unchanged with the extra batch axis →
        the result is IDÉNTICO al per-muestra (LayerNorm normaliza por fila igual).
        Mueve los matmuls a un solo kernel por capa en vez de uno por muestra.
        """
        import torch

        try:
            device = next(self._module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # 1. Stack each input field across the batch → (n, width) per field.
        keys = list(batch[0].keys())
        named: dict[str, Any] = {}
        for name in keys:
            col = []
            for row in batch:
                v = row[name]
                col.append([float(v)] if isinstance(v, (int, float)) else [float(x) for x in v])
            named[name] = torch.tensor(col, dtype=torch.float32, device=device)  # (n, width)

        return self.forward_named_batch(named, keys)

    def forward_named_batch(self, named_inputs: dict[str, Any], keys: list[str] | None = None) -> Any:
        """Batched forward from already-materialized input tensors.

        Training can preload the full dataset on the target device once and pass
        indexed tensor views here. This keeps the expensive Python dict/list ->
        tensor conversion out of the epoch loop while preserving forward_batch
        semantics for evaluation and compatibility.
        """
        import torch

        keys = list(keys or named_inputs.keys())
        named: dict[str, Any] = dict(named_inputs)

        # 2. Embedding lookups (batched): idx (n,) → (n, dim)
        for emb_name, source in self._embed_specs:
            idx = named[source].long().squeeze(-1)
            named[emb_name] = self._module.embeddings[emb_name](idx)

        # 3. Concats over the feature dim
        for concat_name, sources in self._concat_specs:
            named[concat_name] = torch.cat([named[s] for s in sources], dim=-1)

        # 4. Initial current vector (n, total_features)
        emb_names = {e[0] for e in self._embed_specs}
        if self._concat_specs:
            current = named[self._concat_specs[-1][0]]
        else:
            parts = [named[k] for k in keys if k not in emb_names]
            current = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        # 5. Body items in textual order (batch dim flows through unchanged).
        for _, kind, spec in self._body_items:
            if kind == "layer":
                current = self._apply_layer(spec, current, f"L{spec.index}")
            else:
                block = spec
                residual_from = getattr(block, "residual_from", "")
                if residual_from == "PREVIOUS":
                    skip = current
                elif residual_from:
                    skip = named.get(residual_from)
                else:
                    skip = None
                for layer in block.layers:
                    current = self._apply_layer(layer, current, f"{block.name}__L{layer.index}")
                if skip is not None:
                    current = current + skip

        return current

    def _apply_layer(self, layer: Any, x: Any, key: str) -> Any:
        lt = layer.layer_type
        if lt == "Dense":
            z = self._module.sublayers[key](x)
            return _apply_torch_activation(z, getattr(layer, "activation", "linear"))
        elif lt == "LayerNorm":
            return self._module.sublayers[key](x)
        elif lt == "Dropout":
            if key in self._module.sublayers:
                return self._module.sublayers[key](x)
            return x
        elif lt == "Activation":
            return _apply_torch_activation(x, getattr(layer, "activation_kind", "relu"))
        else:
            return x  # Pool, Reshape — pass-through


class _TorchCompositeModule:
    """Actual nn.Module that holds all registered submodules."""

    def __init__(self, embedding_modules: dict, layer_modules: dict) -> None:
        import torch.nn as nn

        class _Inner(nn.Module):
            pass

        self._inner = _Inner()
        # Register embeddings
        emb_dict = {k: v for k, v in embedding_modules.items()}
        self._inner.embeddings = nn.ModuleDict(emb_dict)
        # Register sublayers — keys with dots are not allowed in ModuleDict,
        # but keys using "__" (already sanitized) are fine.
        self._inner.sublayers = nn.ModuleDict(layer_modules)

    @property
    def embeddings(self) -> Any:
        return self._inner.embeddings

    @property
    def sublayers(self) -> Any:
        return self._inner.sublayers

    @property
    def training(self) -> bool:
        return self._inner.training

    def train(self, mode: bool = True) -> "_TorchCompositeModule":
        self._inner.train(mode)
        return self

    def eval(self) -> "_TorchCompositeModule":
        self._inner.eval()
        return self

    def to(self, device: Any) -> "_TorchCompositeModule":
        self._inner.to(device)
        return self

    def parameters(self) -> Any:
        return self._inner.parameters()

    def named_parameters(self) -> Any:
        return self._inner.named_parameters()
