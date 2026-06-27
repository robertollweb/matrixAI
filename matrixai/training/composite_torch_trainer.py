# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""GPU-C2 — torch training loop for composite_network (P19).

Mirrors the dense torch trainer (GPU-C1) for composite architectures: embeddings,
concat, residual blocks, LayerNorm, Dropout. Training uses the same batched forward
as evaluation (module.forward_batch) — one kernel per layer per batch, same as the
dense path. Batch sizing follows effective_batch_size: CUDA uses 16384 (or
MATRIXAI_GPU_BATCH) by default, CPU respects the spec or 2048.

Determinista (stdlib) = suelo; este camino torch = techo de velocidad con GPU.
"""
from __future__ import annotations

from typing import Any, Callable

from matrixai.parameters.store import ParameterSet
from matrixai.parameters.tensor_bridge import torch_available, enable_tf32_if_cuda
from matrixai.forward.composite_torch import (
    composite_network_to_torch_module,
    torch_module_to_composite_parameter_set,
)
from matrixai.training.dense_torch_trainer import _loss_on_probabilities, effective_batch_size


class CompositeTorchTrainError(ValueError):
    pass


def train_composite_network_torch(
    network: Any,
    parameter_set: ParameterSet,
    examples: list[tuple[dict[str, Any], list[float]]],
    loss_fn: str,
    *,
    lr: float = 0.01,
    epochs: int = 50,
    early_stop: tuple[int, str] | None = None,
    device: str = "cpu",
    seed: int = 42,
    batch_size: int | None = None,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train a composite_network via torch autograd with batched forward.

    Uses module.forward_batch() (one kernel per layer per batch) — same path as
    evaluation but WITH gradient tracking. Batch sizing mirrors the dense trainer:
    CUDA uses max(MATRIXAI_GPU_BATCH|16384, spec), CPU respects spec or 2048.

    examples: list of (input_dict {field: value}, target list). Embedding-source
    fields must carry the integer index. Returns the same shape as the dense torch
    trainer: {best_params, epochs, best_val_loss, best_epoch, train_loss, backend,
    device, effective_batch_size}.
    cancel_check: called after each batch; raise to abort (mismo contrato que el denso).
    """
    if not torch_available():
        raise CompositeTorchTrainError("PyTorch is not installed — GPU/torch training requires torch")
    import torch

    if not examples:
        raise CompositeTorchTrainError("no training examples")

    torch.manual_seed(seed)
    enable_tf32_if_cuda(device)  # M15(c): tensor cores en A100/Ada
    module = composite_network_to_torch_module(network, parameter_set)
    module.to(device)

    split = max(1, int(len(examples) * 0.8)) if len(examples) > 1 else len(examples)
    train_ex, val_ex = examples[:split], (examples[split:] or examples[:split])

    input_keys = list(examples[0][0].keys())

    def _materialize(rows: list[tuple[dict[str, Any], list[float]]]) -> tuple[dict[str, Any], Any]:
        named: dict[str, Any] = {}
        for name in input_keys:
            col = []
            for row, _ in rows:
                value = row[name]
                col.append([float(value)] if isinstance(value, (int, float)) else [float(v) for v in value])
            named[name] = torch.tensor(col, dtype=torch.float32, device=device)
        targets = torch.tensor([target for _, target in rows], dtype=torch.float32, device=device)
        return named, targets

    train_named, train_targets = _materialize(train_ex)
    val_named, val_targets = _materialize(val_ex)

    bs = effective_batch_size(device, batch_size, len(train_ex))
    optimizer = torch.optim.SGD(module.parameters(), lr=lr)

    # M15 — mejor estado como tensores torch (clon barato); la conversión cara a
    # ParameterSet se hace una sola vez al final, no en cada época que mejora.
    def _snapshot() -> list[Any]:
        return [p.detach().clone() for p in module.parameters()]

    def _slice_named(named: dict[str, Any], index: Any) -> dict[str, Any]:
        return {name: tensor[index] for name, tensor in named.items()}

    def _batch_loss(named: dict[str, Any], targets: Any, index: Any, training: bool) -> Any:
        """Forward + loss sobre tensores ya cargados en el device."""
        module.train(training)
        batch_named = _slice_named(named, index)
        batch_targets = targets[index]
        out = module.forward_named_batch(batch_named, input_keys)
        return _loss_on_probabilities(out, batch_targets, loss_fn, torch)

    if str(device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:  # noqa: BLE001
            pass

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = _snapshot()
    epoch_trace: list[dict[str, Any]] = []
    no_improve = 0
    patience = early_stop[0] if early_stop else None
    train_loss_val = 0.0

    try:
        for epoch in range(1, epochs + 1):
            rng = torch.Generator().manual_seed(seed + epoch)
            perm = torch.randperm(len(train_ex), generator=rng).to(train_targets.device)
            epoch_loss = 0.0
            for start in range(0, len(train_ex), bs):
                idx = perm[start: start + bs]
                optimizer.zero_grad()
                loss = _batch_loss(train_named, train_targets, idx, training=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * int(idx.numel())
                if cancel_check is not None:
                    cancel_check()
            train_loss_val = epoch_loss / max(1, len(train_ex))

            # Validación batched bajo no_grad
            module.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for vstart in range(0, len(val_ex), bs):
                    vend = min(vstart + bs, len(val_ex))
                    vslice = slice(vstart, vend)
                    val_loss_sum += float(_batch_loss(val_named, val_targets, vslice, training=False).detach()) * (vend - vstart)
            val_loss = val_loss_sum / max(1, len(val_ex))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = _snapshot()
                no_improve = 0
            else:
                no_improve += 1

            entry = {"epoch": epoch, "loss": train_loss_val, "val_loss": val_loss}
            epoch_trace.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)

            if patience is not None and no_improve >= patience:
                break

        # Restaurar el mejor estado y convertir a ParameterSet una sola vez (M15).
        with torch.no_grad():
            for p, s in zip(module.parameters(), best_state):
                p.copy_(s)
        best_params = torch_module_to_composite_parameter_set(network, module, parameter_set)

        peak_vram_gb = 0.0
        if str(device).startswith("cuda"):
            try:
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
            except Exception:  # noqa: BLE001
                pass

        return {
            "best_params": best_params,
            "epochs": epoch_trace,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_loss": train_loss_val,
            "backend": "torch",
            "device": device,
            "effective_batch_size": bs,
            "peak_vram_gb": round(peak_vram_gb, 2),
        }
    finally:
        # M18: liberar la VRAM en este frame (retorno normal y cancelación). La traza de
        # la excepción de cancel retiene estas locales aguas arriba; borrarlas aquí permite
        # recuperar la memoria de inmediato al pulsar Stop (idéntico al trainer denso).
        try:
            del module, optimizer, best_state, train_named, train_targets, val_named, val_targets
        except Exception:  # noqa: BLE001
            pass
        if str(device).startswith("cuda"):
            try:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def evaluate_composite_network_torch(
    network: Any,
    parameter_set: ParameterSet,
    examples: list[tuple[dict[str, Any], list[float]]],
    loss_fn: str,
    *,
    labels: list[str] | None = None,
    device: str = "cpu",
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """GPU-C6/M14+M15(e) — evaluación de un composite_network con forward BATCHED en torch/GPU.

    Mismas métricas que `evaluate_composite_network` (reusa `result_from_predictions`),
    pero el forward corre en batches por el módulo torch en `device` (M15e:
    `composite_torch_forward_batch`, un kernel por capa en vez de uno por muestra) en vez
    de `composite_forward` (Python) fila a fila. Troceado en bloques de 4096 para no
    reventar la VRAM con datasets grandes (igual que el denso).
    """
    from matrixai.training.dense_evaluator import result_from_predictions
    from matrixai.forward.composite_torch import (
        composite_network_to_torch_module,
        composite_torch_forward_batch,
    )

    if not torch_available():
        raise CompositeTorchTrainError("PyTorch is not installed — torch evaluation requires torch")
    if not examples:
        raise ValueError("examples must be non-empty")

    enable_tf32_if_cuda(device)  # M15(c)
    module = composite_network_to_torch_module(network, parameter_set)
    module.to(device)

    _EVAL_CHUNK = 4096
    predictions: list[Any] = []
    try:
        for start in range(0, len(examples), _EVAL_CHUNK):
            chunk = examples[start : start + _EVAL_CHUNK]
            predictions.extend(composite_torch_forward_batch(module, [x for x, _ in chunk], False))
            if cancel_check is not None:
                cancel_check()
        targets = [t for _, t in examples]
        return result_from_predictions(predictions, targets, loss_fn, labels)
    finally:
        # M18: liberar el módulo GPU en este frame (igual que el denso) para que una
        # cancelación entre chunks no deje la VRAM ocupada vía la traza de la excepción.
        try:
            del module
        except Exception:  # noqa: BLE001
            pass
        if str(device).startswith("cuda"):
            try:
                import gc as _gc
                import torch as _torch  # esta función no liga `torch` en su scope
                _gc.collect()
                _torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
