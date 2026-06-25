# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""GPU-C1 — torch training loop for dense_network (multi-layer).

The core already materialises a NetworkSpec as an autograd-capable torch module
(`dense_network_to_torch_module`) and round-trips it back to a ParameterSet
(`torch_module_to_parameter_set`); what was missing was a training loop over those
modules. This provides it: optimizer over `module.parameters()`, loss computed on the
module's probability output (same semantics as the stdlib trainer's `compute_loss`),
val-loss save_best and early stopping. Runs on CPU or CUDA via `device`.

Determinista (stdlib) sigue siendo el suelo; este camino torch es el techo de
velocidad cuando hay GPU. La selección de backend y el fallback viven en el Studio (C3).
"""
from __future__ import annotations

from typing import Any, Callable

from matrixai.parameters.store import ParameterSet
from matrixai.parameters.tensor_bridge import torch_available, enable_tf32_if_cuda
from matrixai.forward.dense_torch import (
    dense_network_to_torch_module,
    torch_module_to_parameter_set,
)

_EPS = 1e-9

_GPU_DEFAULT_BATCH = 16384


def effective_batch_size(device: str, batch_size: int | None, n_train: int) -> int:
    """Batch efectivo, según el dispositivo (M15/M12). Función pura y testeable.

    - CUDA: el batch debe ser GRANDE para llenar la VRAM y mantener el throughput. El
      training text autogenerado trae `BATCH size=8` (pensado para el demo CPU stdlib);
      honrarlo en GPU la deja al ralentí. Así que se IGNORA un batch pequeño del spec y se
      usa `max(MATRIXAI_GPU_BATCH|16384, spec)` — un usuario que pide MÁS, manda.
    - CPU: respeta el batch del spec (o 2048). Lotes pequeños van bien en CPU.
    En ambos casos se capa por `n_train` (nunca exceder el nº de ejemplos de entrenamiento).
    """
    import os
    if str(device).startswith("cuda"):
        raw = os.environ.get("MATRIXAI_GPU_BATCH")
        try:
            gpu_default = int(raw) if raw else _GPU_DEFAULT_BATCH
        except ValueError:
            gpu_default = _GPU_DEFAULT_BATCH
        if gpu_default <= 0:  # 0/negativo/inválido → default (no desactiva el batch)
            gpu_default = _GPU_DEFAULT_BATCH
        return max(1, min(n_train, max(gpu_default, batch_size or 0)))
    return max(1, min(n_train, batch_size if batch_size is not None else 2048))


class DenseTorchTrainError(ValueError):
    pass


def _loss_on_probabilities(out: Any, target: Any, loss_fn: str, torch: Any) -> Any:
    """Loss over the network's probability output — matches stdlib compute_loss."""
    if loss_fn == "mse":
        return torch.nn.functional.mse_loss(out, target)
    if loss_fn == "binary_cross_entropy":
        out = out.clamp(_EPS, 1.0 - _EPS)
        return torch.nn.functional.binary_cross_entropy(out, target)
    if loss_fn == "cross_entropy":
        # out are softmax probabilities, target is one-hot → -Σ y·log(p)
        return -(torch.log(out.clamp_min(_EPS)) * target).sum(dim=1).mean()
    raise DenseTorchTrainError(f"Unknown loss function {loss_fn!r}")


def train_dense_network_torch(
    network: Any,
    parameter_set: ParameterSet,
    examples: list[tuple[list[float], list[float]]],
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
    """Train a dense_network via torch autograd.

    examples: list of (input_vector, target) as produced by `_examples_to_xy`.
    Returns: {best_params (ParameterSet), epochs (trace), best_val_loss, best_epoch,
    backend, device, train_loss}.
    cancel_check: called after each batch; raise _TrainingCancelled-compatible exception to abort.
    """
    if not torch_available():
        raise DenseTorchTrainError("PyTorch is not installed — GPU/torch training requires torch")
    import torch

    if not examples:
        raise DenseTorchTrainError("no training examples")

    torch.manual_seed(seed)
    enable_tf32_if_cuda(device)  # M15(c): tensor cores en A100/Ada (~5-8x matmul fp32)
    module = dense_network_to_torch_module(network, parameter_set)
    module._torch_module.to(device)

    # 80/20 split (mirrors the stdlib dense trainer)
    split = max(1, int(len(examples) * 0.8)) if len(examples) > 1 else len(examples)
    train_ex, val_ex = examples[:split], examples[split:]

    def _to_tensors(rows: list[tuple[list[float], list[float]]]):
        xs = torch.tensor([x for x, _ in rows], dtype=torch.float32, device=device)
        ys = torch.tensor([y for _, y in rows], dtype=torch.float32, device=device)
        return xs, ys

    train_x, train_y = _to_tensors(train_ex)
    val_x, val_y = _to_tensors(val_ex) if val_ex else (train_x, train_y)

    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    # Batch efectivo (M15/M12): en CUDA ignora el `BATCH size=8` autogenerado y usa un
    # batch grande (MATRIXAI_GPU_BATCH|16384) para llenar la GPU; en CPU respeta el spec.
    # Lógica pura y testeable en `effective_batch_size`. Si una red enorme da OOM en una
    # GPU pequeña (T4 16GB), baja MATRIXAI_GPU_BATCH.
    bs = effective_batch_size(device, batch_size, len(train_ex))

    # M15 — guardar el mejor estado como tensores torch (clon barato en device); la
    # conversión cara torch→ParameterSet (O(params) en Python) se hace UNA sola vez al
    # final, no en cada época que mejora.
    def _snapshot() -> list[Any]:
        return [p.detach().clone() for p in module.parameters()]

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = _snapshot()
    epoch_trace: list[dict[str, Any]] = []
    no_improve = 0
    patience = early_stop[0] if early_stop else None
    train_loss_val = 0.0

    try:
        for epoch in range(1, epochs + 1):
            module.train(True)
            perm = torch.randperm(train_x.shape[0])
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, train_x.shape[0], bs):
                idx = perm[i:i + bs]
                xb, yb = train_x[idx], train_y[idx]
                optimizer.zero_grad()
                out = module(xb)
                loss = _loss_on_probabilities(out, yb, loss_fn, torch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
                if cancel_check is not None:
                    cancel_check()
            train_loss_val = epoch_loss / max(1, n_batches)

            module.eval()
            with torch.no_grad():
                val_out = module(val_x)
                val_loss = float(_loss_on_probabilities(val_out, val_y, loss_fn, torch).detach())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = _snapshot()           # clon de tensores (barato), no conversión
                no_improve = 0
            else:
                no_improve += 1

            entry = {"epoch": epoch, "loss": train_loss_val, "val_loss": val_loss}
            epoch_trace.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)  # may raise to cancel (watchdog/cancel)

            if patience is not None and no_improve >= patience:
                break

        # Restaurar el mejor estado y convertir a ParameterSet una sola vez (M15).
        with torch.no_grad():
            for p, s in zip(module.parameters(), best_state):
                p.copy_(s)
        best_params = torch_module_to_parameter_set(network, module, parameter_set)

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
            "batch_size": bs,  # batch efectivo usado (auditable; en CUDA puede subir el del spec)
            "peak_vram_gb": round(peak_vram_gb, 2),
        }
    finally:
        # Liberar la VRAM en el sitio donde se reservó, tanto en retorno normal como en
        # CANCELACIÓN. Clave: al cancelar, la traza de la excepción mantiene vivas estas
        # locales (módulo + datos + clon del mejor estado = los GB en GPU) mientras
        # propaga; sin borrarlas aquí, el empty_cache() de aguas arriba no puede devolver
        # la memoria y la GPU "se queda" ocupada tras Stop. Borrarlas en este frame las
        # saca de la traza y permite recuperar la VRAM de inmediato.
        try:
            del module, optimizer, train_x, train_y, val_x, val_y, best_state
        except Exception:  # noqa: BLE001
            pass
        if str(device).startswith("cuda"):
            try:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def evaluate_dense_network_torch(
    network: Any,
    parameter_set: ParameterSet,
    examples: list[tuple[list[float], list[float]]],
    loss_fn: str,
    *,
    labels: list[str] | None = None,
    device: str = "cpu",
    cancel_check: Callable[[], None] | None = None,
) -> Any:
    """GPU-C6/M14 — evaluación de un dense_network con forward BATCHED en torch/GPU.

    Mismas métricas que `evaluate_dense_network` (reusa `result_from_predictions`),
    pero el forward corre en un solo batch sobre `device` en vez de fila a fila en
    Python. Con datasets grandes + redes anchas esto pasa de minutos/horas a ms y evita
    que un entrenamiento por GPU se quede colgado evaluando en CPU.
    """
    from matrixai.training.dense_evaluator import result_from_predictions

    if not torch_available():
        raise DenseTorchTrainError("PyTorch is not installed — torch evaluation requires torch")
    import torch

    if not examples:
        raise ValueError("examples must be non-empty")

    enable_tf32_if_cuda(device)  # M15(c)
    module = dense_network_to_torch_module(network, parameter_set)
    module._torch_module.to(device)
    module.eval()

    # Chunks of 4096 rows: safe even when row-limits are removed (avoids VRAM OOM
    # with very large datasets; each chunk is independent, result is equivalent).
    _EVAL_CHUNK = 4096
    predictions: list[Any] = []
    try:
        with torch.no_grad():
            for start in range(0, len(examples), _EVAL_CHUNK):
                chunk = examples[start : start + _EVAL_CHUNK]
                xs = torch.tensor([x for x, _ in chunk], dtype=torch.float32, device=device)
                out = module(xs)
                predictions.extend(out.detach().cpu().tolist())
                if cancel_check is not None:
                    cancel_check()

        targets = [t for _, t in examples]
        return result_from_predictions(predictions, targets, loss_fn, labels)
    finally:
        # Igual que en el trainer: liberar el módulo GPU en este frame para que una
        # cancelación entre chunks no deje la VRAM ocupada vía la traza de la excepción.
        try:
            del module
        except Exception:  # noqa: BLE001
            pass
        if str(device).startswith("cuda"):
            try:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def probe_collapse_torch(
    network: Any,
    parameter_set: ParameterSet,
    input_dim: int,
    *,
    device: str = "cpu",
    tol: float = 1e-4,
) -> dict[str, Any] | None:
    """M7 en torch/GPU — prueba de colapso (predictor constante) por el módulo torch.

    4 probes (ceros, unos, 2 aleatorios sembrados) en un solo batch sobre `device`;
    True si la salida es ~constante. Reemplaza el `_probe_model_collapse` por runtime
    Python (O(params) × 4 forwards → minutos y GBs de RAM con redes grandes); aquí es
    instantáneo. Best-effort: None si torch no está o falla."""
    if not torch_available() or input_dim <= 0:
        return None
    import torch  # noqa: PLC0415
    import random  # noqa: PLC0415
    rng = random.Random(7)
    rows = [
        [0.0] * input_dim,
        [1.0] * input_dim,
        [round(rng.random(), 6) for _ in range(input_dim)],
        [round(rng.random(), 6) for _ in range(input_dim)],
    ]
    module = dense_network_to_torch_module(network, parameter_set)
    module._torch_module.to(device)
    module.eval()
    try:
        with torch.no_grad():
            out = module(torch.tensor(rows, dtype=torch.float32, device=device)).detach().cpu().tolist()
        first = out[0]
        collapsed = all(
            abs(v - first[i]) <= tol for o in out[1:] for i, v in enumerate(o)
        )
        res: dict[str, Any] = {"collapsed": collapsed}
        if collapsed:
            res["constant_output"] = [round(v, 6) for v in first]
        return res
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            del module
        except Exception:  # noqa: BLE001
            pass
        if str(device).startswith("cuda"):
            try:
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
