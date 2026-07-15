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
    type_result: Any = None,
    optimizer: str | None = None,
    pad_id: int | None = None,
    materialize: bool | None = None,
    output_name: str = "",
    initial_state_dict: dict[str, Any] | None = None,
    validation_examples: list[tuple[dict[str, Any], list[float]]] | None = None,
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

    TRANSFORMER C4: para redes con BLOCK TRANSFORMER, pasar `type_result`
    (check_composite_network_types) — el módulo se construye vía la entrada
    común, que delega en el builder del transformer. Las muestras llevan UNA
    clave (la SEQUENCE) con la lista [L] de token ids; `pad_id` deriva la
    máscara de padding (None = todo real, filas pre-tokenizadas a L fija).
    `optimizer`: "sgd" | "adam" — default adam si hay transformer (los
    transformers reales no convergen bien con SGD plano, aviso de P11), sgd si
    no (comportamiento previo intacto). `materialize` sigue la puerta
    PESOS_GRANDES del trainer denso: None = materializar solo por debajo de
    torch_native_min_params(); por encima devuelve best_state_dict (tensores
    CPU, claves del ParameterSet) y best_params=None.
    """
    if not torch_available():
        raise CompositeTorchTrainError("PyTorch is not installed — GPU/torch training requires torch")
    import torch

    if not examples:
        raise CompositeTorchTrainError("no training examples")

    is_transformer = bool(getattr(network, "transformer_blocks", []))
    if is_transformer and type_result is None:
        raise CompositeTorchTrainError(
            f"NETWORK {network.name} contains a BLOCK TRANSFORMER — pass "
            f"type_result= (from check_composite_network_types) to train it"
        )

    torch.manual_seed(seed)
    enable_tf32_if_cuda(device)  # M15(c): tensor cores en A100/Ada
    # output_name participa en el hash de esquema (residual-2 de la re-auditoría
    # C3): debe llegar hasta el builder o un set construido con output_name
    # fallaría aquí con un mismatch engañoso.
    module = composite_network_to_torch_module(
        network, parameter_set, type_result, output_name
    )
    # Auditoría C4 [MEDIA-1]: reanudar desde un best_state_dict PESOS_GRANDES
    # (espejo del initial_state_dict del trainer denso).
    if initial_state_dict is not None:
        if not is_transformer:
            raise CompositeTorchTrainError(
                "initial_state_dict is only supported for transformer networks (C4)"
            )
        from matrixai.forward.transformer_torch import _load_state_into_module
        _load_state_into_module(module, initial_state_dict)
    module.to(device)

    # Auditoría C4 ronda 2 [MEDIA-2]: particiones EXPLÍCITAS del caller (el
    # adapter CLI aplica el DATASET SPLIT declarado) — con validation_examples,
    # `examples` es TODO train y no se re-parte por posición.
    if validation_examples is not None:
        if not validation_examples:
            raise CompositeTorchTrainError("validation_examples must be non-empty")
        train_ex, val_ex = examples, validation_examples
    else:
        split = max(1, int(len(examples) * 0.8)) if len(examples) > 1 else len(examples)
        train_ex, val_ex = examples[:split], (examples[split:] or examples[:split])

    input_keys = list(examples[0][0].keys())
    if is_transformer:
        if len(input_keys) != 1:
            raise CompositeTorchTrainError(
                f"transformer training expects exactly one input key (the "
                f"SEQUENCE), got {input_keys}"
            )
        seq_key = input_keys[0]

    def _materialize(rows: list[tuple[dict[str, Any], list[float]]]) -> tuple[dict[str, Any], Any]:
        named: dict[str, Any] = {}
        if is_transformer:
            ids = torch.tensor(
                [[int(v) for v in row[seq_key]] for row, _ in rows],
                dtype=torch.long, device=device,
            )
            named[seq_key] = ids
            named["__mask__"] = (
                ids != pad_id if pad_id is not None
                else torch.ones_like(ids, dtype=torch.bool)
            )
        else:
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
    opt_name = optimizer or ("adam" if is_transformer else "sgd")
    if opt_name == "adam":
        optim = torch.optim.Adam(module.parameters(), lr=lr)
    elif opt_name == "sgd":
        optim = torch.optim.SGD(module.parameters(), lr=lr)
    else:
        raise CompositeTorchTrainError(
            f"unsupported optimizer {opt_name!r} — supported: sgd, adam"
        )

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
        if is_transformer:
            out = module.forward_batch(
                batch_named[seq_key], masks=batch_named["__mask__"]
            )
        else:
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
    # Auditoría C4 ronda 2 [MEDIA-1] — línea base de época 0 al REANUDAR
    # (espejo del trainer denso): sin ella, best_val_loss arranca en inf y la
    # época 1 siempre "gana" aunque haya EMPEORADO los pesos de partida (lr
    # alto, pocas épocas). Con la base, si ninguna época mejora el punto de
    # partida se devuelven los pesos iniciales intactos (best_epoch=0) —
    # reanudar no pierde el estado previo. Solo warm-start: init fresco intacto.
    if initial_state_dict is not None:
        module.eval()
        with torch.no_grad():
            baseline_sum = 0.0
            for vstart in range(0, len(val_ex), bs):
                vend = min(vstart + bs, len(val_ex))
                vslice = slice(vstart, vend)
                baseline_sum += float(
                    _batch_loss(val_named, val_targets, vslice, training=False).detach()
                ) * (vend - vstart)
        best_val_loss = baseline_sum / max(1, len(val_ex))
        best_epoch = 0
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
                optim.zero_grad()
                loss = _batch_loss(train_named, train_targets, idx, training=True)
                loss.backward()
                optim.step()
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
        # TRANSFORMER C4 — puerta PESOS_GRANDES (espejo del trainer denso):
        # param_count con .numel() (O(#tensores)); por encima del umbral NO se
        # materializan listas Python — el resultado lleva best_state_dict
        # (tensores CPU, claves del ParameterSet) y best_params=None.
        from matrixai.resources import torch_native_min_params
        param_count = sum(p.numel() for p in module.parameters())
        should_materialize = materialize if materialize is not None else (
            param_count < torch_native_min_params()
        )
        best_state_dict = None
        if should_materialize:
            best_params = torch_module_to_composite_parameter_set(network, module, parameter_set)
        else:
            best_params = None
            if is_transformer:
                from matrixai.forward.composite_torch import (
                    transformer_module_to_state_dict,
                )
                best_state_dict = transformer_module_to_state_dict(module)
            else:
                raise CompositeTorchTrainError(
                    f"composite network {network.name} exceeds "
                    f"torch_native_min_params() but has no state_dict path — "
                    f"only transformer networks support it (C4)"
                )

        peak_vram_gb = 0.0
        if str(device).startswith("cuda"):
            try:
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
            except Exception:  # noqa: BLE001
                pass

        return {
            "best_params": best_params,
            "best_state_dict": best_state_dict,
            "param_count": param_count,
            "optimizer": opt_name,
            "n_train": len(train_ex),
            "n_val": len(val_ex),
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
            del module, optim, best_state, train_named, train_targets, val_named, val_targets
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
    parameter_set: ParameterSet | None,
    examples: list[tuple[dict[str, Any], list[float]]],
    loss_fn: str,
    *,
    labels: list[str] | None = None,
    device: str = "cpu",
    cancel_check: Callable[[], None] | None = None,
    type_result: Any = None,
    pad_id: int | None = None,
    output_name: str = "",
    state_dict: dict[str, Any] | None = None,
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

    is_transformer = bool(getattr(network, "transformer_blocks", []))
    if is_transformer and type_result is None:
        raise CompositeTorchTrainError(
            f"NETWORK {network.name} contains a BLOCK TRANSFORMER — pass "
            f"type_result= (from check_composite_network_types) to evaluate it"
        )

    enable_tf32_if_cuda(device)  # M15(c)
    # Auditoría C4 [MEDIA-1]: evaluar directamente desde un best_state_dict
    # PESOS_GRANDES (best_params=None) — espejo del evaluador denso. Con
    # state_dict, parameter_set puede ser None.
    if state_dict is not None:
        if not is_transformer:
            raise CompositeTorchTrainError(
                "state_dict evaluation is only supported for transformer networks (C4)"
            )
        from matrixai.forward.transformer_torch import (
            transformer_network_to_torch_module_from_state,
        )
        module = transformer_network_to_torch_module_from_state(
            network, type_result, state_dict, output_name=output_name,
        )
    elif parameter_set is None:
        raise CompositeTorchTrainError(
            "evaluate requires a parameter_set or a state_dict"
        )
    else:
        module = composite_network_to_torch_module(
            network, parameter_set, type_result, output_name
        )
    module.to(device)

    _EVAL_CHUNK = 4096
    predictions: list[Any] = []
    try:
        if is_transformer:
            from matrixai.forward.transformer_torch import (
                transformer_torch_forward_batch,
            )
            seq_key = next(iter(examples[0][0].keys()))
        for start in range(0, len(examples), _EVAL_CHUNK):
            chunk = examples[start : start + _EVAL_CHUNK]
            if is_transformer:
                rows = [[int(v) for v in x[seq_key]] for x, _ in chunk]
                predictions.extend(
                    transformer_torch_forward_batch(module, rows, pad_id=pad_id)
                )
            else:
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


def probe_collapse_transformer_torch(
    network: Any,
    type_result: Any,
    seq_length: int,
    vocab_size: int,
    *,
    device: str = "cpu",
    tol: float = 1e-4,
    parameter_set: ParameterSet | None = None,
    state_dict: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """M7 para BLOCK TRANSFORMER — mismo probe de colapso (predictor constante)
    que `probe_collapse_torch` (dense), adaptado a una entrada SEQUENCE: 4
    secuencias de ids sintéticas (ceros, máximo, 2 aleatorias sembradas) en
    vez de 4 vectores de floats — el resto del contrato es idéntico (4
    forwards en un batch, instantáneo, `collapsed=True` si la salida no
    varía con la entrada).

    Autoauditoría C4/C5 — SECUENCIAS_PRODUCTO: `_probe_model_collapse`
    (`playground.py`, M7 original) mira exclusivamente `program.vectors`,
    así que devuelve `None` para CUALQUIER modelo Text (su INPUT es una
    SEQUENCE, nunca tiene VECTOR) — el aviso "Modelo colapsado" de la UI
    (con el botón "Reintentar con otra inicialización", M8-A3) nunca se
    disparaba para un transformer, ni siquiera cuando el modelo colapsó de
    verdad a un predictor constante (mismo síntoma que un accuracy ~50% /
    F1 macro ~33% en clasificación binaria — la firma matemática de "predice
    SIEMPRE la misma clase"). Mismo patrón que `probe_collapse_torch`: se
    adjunta al resultado del entrenamiento (torch, instantáneo) en vez de
    reintentar por un runtime Python que ni siquiera existe para BLOCK
    TRANSFORMER (invariante 6 del contrato A).

    Best-effort: `None` si torch no está disponible o si no hay pesos
    (ni `parameter_set` ni `state_dict`) que probar."""
    if not torch_available() or seq_length <= 0 or vocab_size <= 1:
        return None
    if parameter_set is None and state_dict is None:
        return None
    import torch  # noqa: PLC0415
    import random  # noqa: PLC0415
    from matrixai.forward.transformer_torch import (
        transformer_network_to_torch_module,
        transformer_network_to_torch_module_from_state,
        transformer_torch_forward_batch,
    )

    rng = random.Random(7)
    rows = [
        [0] * seq_length,
        [vocab_size - 1] * seq_length,
        [rng.randrange(vocab_size) for _ in range(seq_length)],
        [rng.randrange(vocab_size) for _ in range(seq_length)],
    ]
    try:
        module = (
            transformer_network_to_torch_module_from_state(network, type_result, state_dict)
            if state_dict is not None
            else transformer_network_to_torch_module(network, type_result, parameter_set)
        )
        module.to(device)
        outputs = transformer_torch_forward_batch(module, rows)
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            del module
        except Exception:  # noqa: BLE001
            pass

    first = outputs[0]
    if any(len(o) != len(first) for o in outputs[1:]):
        return None
    collapsed = all(
        abs(value - first[i]) <= tol
        for probe_out in outputs[1:]
        for i, value in enumerate(probe_out)
    )
    result: dict[str, Any] = {"collapsed": collapsed}
    if collapsed:
        result["constant_output"] = [round(v, 6) for v in first]
    return result
