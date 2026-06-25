# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""GPU-C2 — torch training loop for composite_network (P19).

Mirrors the dense torch trainer (GPU-C1) for composite architectures: embeddings,
concat, residual blocks, LayerNorm, Dropout. The composite torch module processes one
sample at a time (`forward_with_dict`), so training is per-sample SGD (like the stdlib
composite trainer) — correct on CPU and CUDA. Batched composite forward (for full GPU
throughput) is a future optimisation; the dense path (C1) is already batched.

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
from matrixai.training.dense_torch_trainer import _loss_on_probabilities


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
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train a composite_network via torch autograd (per-sample SGD).

    examples: list of (input_dict {field: value}, target list). Embedding-source
    fields must carry the integer index. Returns the same shape as the dense torch
    trainer: {best_params, epochs, best_val_loss, best_epoch, train_loss, backend, device}.
    cancel_check: called periodically; raise to abort (mismo contrato que el denso, M18).
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

    optimizer = torch.optim.SGD(module.parameters(), lr=lr)

    def _sample_loss(inp: dict[str, Any], tgt: list[float]) -> Any:
        out = module.forward_with_dict(inp)                      # (out,)
        target = torch.tensor([tgt], dtype=torch.float32, device=device)  # (1, out)
        return _loss_on_probabilities(out.unsqueeze(0), target, loss_fn, torch)

    # M15 — mejor estado como tensores torch (clon barato); la conversión cara a
    # ParameterSet se hace una sola vez al final, no en cada época que mejora.
    def _snapshot() -> list[Any]:
        return [p.detach().clone() for p in module.parameters()]

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = _snapshot()
    epoch_trace: list[dict[str, Any]] = []
    no_improve = 0
    patience = early_stop[0] if early_stop else None
    train_loss_val = 0.0
    order = list(range(len(train_ex)))

    try:
        for epoch in range(1, epochs + 1):
            module.train(True)
            rng = torch.Generator().manual_seed(seed + epoch)
            perm = torch.randperm(len(train_ex), generator=rng).tolist()
            epoch_loss = 0.0
            for _i, j in enumerate(perm):
                inp, tgt = train_ex[j]
                optimizer.zero_grad()
                loss = _sample_loss(inp, tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                # M18: cancelación intra-época (cada 64 muestras; el backward domina,
                # el coste del check es ínfimo) para que Stop no espere a fin de época.
                if cancel_check is not None and (_i & 63) == 0:
                    cancel_check()
            train_loss_val = epoch_loss / max(1, len(train_ex))

            module.eval()
            with torch.no_grad():
                v = sum(float(_sample_loss(inp, tgt).detach()) for inp, tgt in val_ex)
                val_loss = v / max(1, len(val_ex))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = _snapshot()           # clon de tensores (barato)
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

        return {
            "best_params": best_params,
            "epochs": epoch_trace,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_loss": train_loss_val,
            "backend": "torch",
            "device": device,
        }
    finally:
        # M18: liberar la VRAM en este frame (retorno normal y cancelación). La traza de
        # la excepción de cancel retiene estas locales aguas arriba; borrarlas aquí permite
        # recuperar la memoria de inmediato al pulsar Stop (idéntico al trainer denso).
        try:
            del module, optimizer, best_state
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
