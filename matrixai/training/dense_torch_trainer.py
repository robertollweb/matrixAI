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

import time
from typing import Any, Callable

from matrixai.parameters.store import ParameterSet
from matrixai.parameters.tensor_bridge import torch_available, enable_tf32_if_cuda
from matrixai.forward.dense_torch import (
    dense_network_to_torch_module,
    dense_network_to_torch_module_from_state,
    dense_module_to_state_dict,
    torch_module_to_parameter_set,
)
from matrixai.resources import torch_native_min_params
from matrixai.training.laboratorio import CapturaSpec

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


def _capas_a_listas(estado: list[Any]) -> list[dict[str, Any]]:
    """[W1, b1, W2, b2, …] (tensores) → [{"W": [[…]], "b": […]}, …], una sola vez
    y solo de las épocas que van al bloque. float32 → float de Python es exacto."""
    return [{"W": estado[i].detach().cpu().tolist(), "b": estado[i + 1].detach().cpu().tolist()}
            for i in range(0, len(estado), 2)]


def _bloque_laboratorio(network, module, capturar, plan, best_state, best_epoch,
                        estados, ultima, hist, parciales, val_ex, loss_fn,
                        param_count, torch, cada_k=1, n_lotes=0,
                        fallos=None) -> dict[str, Any]:
    """133-C2 — arma el bloque `laboratorio` con lo que el bucle fue guardando.

    `val_ex` es la validación ENTERA y preparada (de ella salen las filas); `fallos`,
    los `omitido` de una captura que se apagó a medias (ver el bucle)."""
    from matrixai.training import laboratorio as lab

    finales = _capas_a_listas(best_state) if plan["pesos_finales"] else None
    muestra = None
    if plan["n_muestra"] > 0 and finales is not None and estados is not None:
        disp = dict(estados)
        disp[best_epoch] = best_state
        elegidas = lab.elegir_muestra(
            disp, 1 if 1 in disp else min(disp), best_epoch,
            ultima if ultima is not None else best_epoch, plan["n_muestra"])
        # Época 0 = los pesos de PARTIDA (arranque en caliente sin mejora): se dice.
        muestra = [(e, (m + ["partida"]) if e == 0 else m, _capas_a_listas(disp[e]))
                   for e, m in elegidas]
    extra = list(fallos or [])
    grad = None
    if plan["gradientes"] and hist:
        guardadas, om = lab.muestrear_gradientes(hist, best_epoch, plan["n_epocas_grad"])
        if om is not None:
            extra.append(om)
        valores = torch.stack([h for _e, h in guardadas]).cpu().tolist()  # [épocas][tensores]
        epocas = [e for e, _h in guardadas]
        n_capas = len(module.linears)
        grad = {
            "definicion": lab.DEFINICION_GRADIENTE,
            "epocas": epocas,
            "epocas_medidas": len(hist),
            "cada_k_lotes": cada_k,
            "lotes_por_epoca": n_lotes,
            "parciales": [e for e in parciales if e in set(epocas)],
            "por_capa": [
                {"capa": network.layers[i].index,
                 "W": [fila[2 * i] for fila in valores],
                 "b": [fila[2 * i + 1] for fila in valores]}
                for i in range(n_capas)],
        }
    filas = lab.filas_de_validacion(val_ex, plan["n_filas"], loss_fn, capturar)
    forma = lab.capas_de(
        network, [(lin.out_features, lin.in_features) for lin in module.linears])
    return lab.armar_bloque(
        entrenador="torch", network=network, plan=plan, finales=finales,
        epoca_final=best_epoch, muestra=muestra, grad=grad, filas=filas,
        spec=capturar, n_params=param_count, forma=forma, extra_omitido=extra,
        parciales=list(parciales), epocas_corridas=ultima or 0)


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
    materialize: bool | None = None,
    initial_state_dict: dict[str, Any] | None = None,
    optimizer: str | None = None,
    validation_examples: list[tuple[list[float], list[float]]] | None = None,
    plazo: float | None = None,
    weight_decay: float = 0.0,
    schedule: str | None = None,
    capturar: CapturaSpec | None = None,
    validacion_completa: list[tuple[list[float], list[float]]] | None = None,
) -> dict[str, Any]:
    """Train a dense_network via torch autograd.

    examples: list of (input_vector, target) as produced by `_examples_to_xy`.
    Returns: {best_params (ParameterSet | None), best_state_dict (dict[str, Tensor]
    | None — CPU), epochs (trace), best_val_loss, best_epoch, backend, device,
    train_loss, materialized (bool)}.
    cancel_check: called after each batch; raise _TrainingCancelled-compatible exception to abort.

    **`plazo`: PARAR A TIEMPO Y QUEDARSE CON LO APRENDIDO** (2026-09-22). Un instante de
    `time.monotonic()`; `None` = sin plazo, como siempre. Se mira DESPUÉS DE CADA LOTE
    —con lote 8, una época de un conjunto grande son miles de pasos—, y al pasarse se
    sale del lote, se evalúa esa época A MEDIAS igual que una entera, se queda el MEJOR
    estado visto y se devuelve `parado_por_plazo: True`. No es `cancel_check`: cancelar
    tira el trabajo; esto lo entrega. Motivo, medido en Fase 0: la densa no paraba sola y
    la mataban desde fuera a 643-646 s de un tope de 630, sin devolver ni la mejor época
    que ya tenía; en el producto eso es «el motor falló» donde debía ser «entrené lo que
    cupo». Con un plazo que no se alcanza, el entrenamiento es IDÉNTICO al de sin plazo
    (misma semilla, mismos pesos): solo se compara un reloj. La época a medias lleva
    `"parcial": True` en su traza, y SOLO si quedaban tandas de esa época al vencer el plazo
    (131-C5): si venció en la última, la época está completa y no se marca (se para igual).

    PESOS_GRANDES C5: `initial_state_dict` (tensores CPU, mismas claves que
    `dense_module_to_state_dict`/`.mxw`) reanuda el entrenamiento desde esos
    pesos en vez de la inicialización nativa de `nn.Linear` — mismo patrón que
    ya usa `evaluate_dense_network_torch`/`probe_collapse_torch` para
    reconstruir el módulo desde tensores (`dense_network_to_torch_module_from_
    state`), nunca desde una `ParameterSet` con valores (ningún `.tolist()` de
    por medio, ni para modelos grandes ni pequeños). Si se omite, se
    inicializa desde `parameter_set` como siempre (plantilla vacía → init
    nativo, o valores → los de siempre).

    PESOS_GRANDES C2: por defecto (`materialize=None`), el propio trainer decide
    según `torch_native_min_params()` — por debajo del umbral, comportamiento
    IDÉNTICO a siempre (`best_params` con valores, `best_state_dict=None`); por
    encima, la conversión O(#params) a listas Python se SALTA: `best_params=None`
    y `best_state_dict` lleva los tensores CPU (`dense_module_to_state_dict`,
    copia vectorizada, no iteración Python). `materialize` explícito fuerza una
    rama u otra (tests; o un caller que ya sabe qué formato quiere).

    Nota de alcance (C2): esta función y `evaluate_dense_network_torch`/
    `probe_collapse_torch` YA soportan el camino sin materializar, testeado en
    aislamiento. Conectarlo al job/resultado vivo del Studio (`params_best` en
    el dict del job, hoy siempre un `ParameterSet.to_dict()`) es C3 ("el job
    lleva tensores, no dicts") — hasta que C3 cierre, el caller de producción
    (`playground._dense_torch_train_result`) sigue sin pasar `materialize=False`
    y por tanto entrena exactamente igual que hoy para cualquier tamaño.

    CONTRATO 118-C3b: `weight_decay` (regularización L2, `torch.optim`) se
    aplica al optimizador que se acabe instanciando (`sgd`/`adam`/`adamw`,
    los tres lo admiten) — por omisión 0.0, byte-idéntico a antes. `schedule`
    admite únicamente `"cosine"` (`torch.optim.lr_scheduler.CosineAnnealingLR`,
    `T_max=epochs` — las épocas MÁXIMAS pedidas, no las que de verdad se
    ejecuten si la parada temprana corta antes) y avanza UN PASO POR ÉPOCA,
    incluida la última si es parcial por `plazo`. `None` (por omisión) no
    crea ningún scheduler — comportamiento previo intacto.

    CONTRATO 133-C2 (el laboratorio): `capturar` (un `CapturaSpec`, `None` por
    omisión) pide guardar lo MEDIDO de este entrenamiento, para poder enseñarlo:
    las normas del gradiente por capa y por época, los pesos y sesgos finales (los
    de la MEJOR época: los mismos `best_params` de «Probar una fila»), los pesos de
    una muestra de épocas (con 3 o más huecos, siempre la primera, la mejor y la última; con
    menos, la mejor y la última) y ~20 filas de
    validación ya preparadas. Va en `resultado["laboratorio"]`; lo que no cupo en
    los topes, en `laboratorio["omitido"]`. Con `capturar=None` NO se crea ningún
    acumulador ni se clona nada, no se toca el generador aleatorio y el resultado
    no lleva la clave: el entrenamiento es byte a byte el de antes. Con captura
    tampoco cambia: solo se LEE el gradiente (`p.grad`, sin modificarlo) y se
    clonan pesos. La excepción son los topes de RELOJ (`plazo`, o un vigilante de
    fuera): la captura añade tiempo y puede mover el lote en el que corta el reloj.
    La captura nunca tumba el entrenamiento: si la medida del gradiente o una copia
    de pesos falla a mitad, esa parte se apaga, se declara en `omitido` (texto fijo y
    el tipo de la excepción en `error`) y el entrenamiento sigue igual.
    `validacion_completa` (solo con `capturar`): la validación ENTERA y preparada de la
    que salen las filas del bloque, cuando `validation_examples` es solo el
    subconjunto con el que se ELIGE la época (regresión con el objetivo sin recortar:
    las filas alcanzables). Sin ella, las filas salen de la validación del bucle.
    """
    if not torch_available():
        raise DenseTorchTrainError("PyTorch is not installed — GPU/torch training requires torch")
    import torch

    if not examples:
        raise DenseTorchTrainError("no training examples")

    torch.manual_seed(seed)  # sigue haciendo falta: siembra también el shuffle de cada época (randperm)
    enable_tf32_if_cuda(device)  # M15(c): tensor cores en A100/Ada (~5-8x matmul fp32)
    module = (
        dense_network_to_torch_module_from_state(network, initial_state_dict, device)
        if initial_state_dict is not None
        else dense_network_to_torch_module(network, parameter_set)
    )
    module._torch_module.to(device)

    # Auditoría BIBLIOTECA_PROYECTOS_INTELIGENTES C3 [ALTA]: particiones
    # EXPLÍCITAS del caller (playground.py aplica el SPLIT mode=temporal|
    # random declarado, mismo patrón que `train_composite_network_torch`)
    # — sin `validation_examples`, 80/20 fijo de siempre (mirrors the
    # stdlib dense trainer), byte-idéntico.
    if validation_examples is not None:
        if not validation_examples:
            raise DenseTorchTrainError("validation_examples must be non-empty")
        train_ex, val_ex = examples, validation_examples
    else:
        split = max(1, int(len(examples) * 0.8)) if len(examples) > 1 else len(examples)
        train_ex, val_ex = examples[:split], examples[split:]

    def _to_tensors(rows: list[tuple[list[float], list[float]]]):
        xs = torch.tensor([x for x, _ in rows], dtype=torch.float32, device=device)
        ys = torch.tensor([y for _, y in rows], dtype=torch.float32, device=device)
        return xs, ys

    train_x, train_y = _to_tensors(train_ex)
    val_x, val_y = _to_tensors(val_ex) if val_ex else (train_x, train_y)

    # Auditoría C4 [ALTA-1]: honrar el optimizador declarado; desconocido →
    # error, nunca sustitución silenciosa. Default sgd (comportamiento previo).
    # CONTRATO 118-C3b: adamw + weight_decay (en los tres, no solo adamw).
    opt_name = optimizer or "sgd"
    if opt_name == "adamw":
        optim_ = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optim_ = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optim_ = torch.optim.SGD(module.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"unsupported optimizer {opt_name!r} — supported: sgd, adam, adamw"
        )
    optimizer = optim_
    # CONTRATO 118-C3b: `T_max` son las épocas MÁXIMAS pedidas (`epochs`, el
    # parámetro de esta función) — el programa no se re-escala si la parada
    # temprana corta antes (mismo criterio que `torch_trainer.py`).
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if schedule == "cosine" else None
    )
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

    # 133-C2: la captura. Todo lo de este bloque cuelga de `capturar`; con None estas
    # variables se quedan vacías y el bucle no ejecuta ninguna rama nueva.
    cap_plan = None
    cap_params: list[Any] = []
    cap_acc = None
    cap_hist: list[Any] = []
    cap_parciales: list[int] = []
    cap_estados: dict[int, list[Any]] = {}
    cap_programadas: frozenset[int] = frozenset()
    cap_cada = 1
    cap_n = 0
    cap_n_lotes = 0
    cap_fallos: list[dict[str, Any]] = []
    cap_val = val_ex
    if capturar is not None:
        from matrixai.training import laboratorio as _lab
        cap_params = list(module.parameters())
        _n_params = sum(p.numel() for p in cap_params)
        cap_val = validacion_completa if validacion_completa else val_ex
        cap_plan = _lab.planear(
            capturar, _n_params, len(module.linears), epochs,
            len(cap_val), len(train_ex[0][0]) + len(train_ex[0][1]))
        if cap_plan["gradientes"]:
            cap_acc = torch.zeros(len(cap_params), device=device)
        cap_programadas = _lab.epocas_programadas(epochs, cap_plan["n_muestra"])
        cap_n_lotes = -(-train_x.shape[0] // bs)
        cap_cada = _lab.lotes_cada(cap_n_lotes, capturar)
        _norma_lote = getattr(torch, "_foreach_norm", None)
    # PESOS_GRANDES C5 audit (MEDIA): al REANUDAR, los pesos de partida son un
    # candidato real a "mejor estado" (a diferencia del init aleatorio, que
    # nunca vale la pena conservar) — se evalúan como línea base "época 0".
    # Sin esto, `best_val_loss` arranca en inf y la época 1 siempre "gana"
    # aunque haya EMPEORADO los pesos (lr alto, dataset malo, pocas épocas):
    # el resultado devolvía pesos destruidos como best y el auto-guardado del
    # Studio machacaba el snapshot bueno con ellos. Con la línea base, si
    # ninguna época mejora el punto de partida, se devuelven los pesos de
    # partida intactos (`best_epoch=0`) — "reentrenar no pierde los pesos
    # previos". Solo para warm-start: el camino de init fresco no cambia.
    if initial_state_dict is not None:
        module.eval()
        with torch.no_grad():
            init_out = module(val_x)
            best_val_loss = float(_loss_on_probabilities(init_out, val_y, loss_fn, torch).detach())
        best_epoch = 0
    epoch_trace: list[dict[str, Any]] = []
    no_improve = 0
    patience = early_stop[0] if early_stop else None
    train_loss_val = 0.0
    parado_por_plazo = False

    try:
        for epoch in range(1, epochs + 1):
            module.train(True)
            perm = torch.randperm(train_x.shape[0])
            epoch_loss = 0.0
            n_batches = 0
            epoca_a_medias = False
            for i in range(0, train_x.shape[0], bs):
                idx = perm[i:i + bs]
                xb, yb = train_x[idx], train_y[idx]
                optimizer.zero_grad()
                out = module(xb)
                loss = _loss_on_probabilities(out, yb, loss_fn, torch)
                loss.backward()
                if cap_acc is not None and n_batches % cap_cada == 0:
                    # 133-C2: norma L2 del gradiente de cada tensor, ANTES de optimizer.step
                    # (así es previa al weight_decay, que torch suma dentro del paso). Se
                    # acumula en el dispositivo, sin `.item()` por lote. Uno de cada
                    # `cap_cada` lotes (el primero siempre): ver `laboratorio.lotes_cada`.
                    # En su propio `try`: si falla, se apaga la captura de gradientes, se
                    # declara y el entrenamiento sigue (los pesos no se tocan aquí).
                    try:
                        _g = [p.grad for p in cap_params]
                        cap_acc += torch.stack(
                            _norma_lote(_g) if _norma_lote is not None else [x.norm() for x in _g])
                        cap_n += 1
                    except Exception as _exc:  # noqa: BLE001
                        cap_acc = None
                        cap_hist = []
                        cap_fallos.append(_lab.omitido_por_fallo(
                            "grad_norm", epochs, _lab.FALLO_GRADIENTE, _exc, epoca=epoch))
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
                if cancel_check is not None:
                    cancel_check()
                if plazo is not None and time.monotonic() >= plazo:
                    parado_por_plazo = True
                    # 131-C5: a medias SOLO si quedaban tandas. Si el plazo venció en la
                    # ÚLTIMA, la época está completa aunque se pare aquí.
                    epoca_a_medias = (i + bs) < train_x.shape[0]
                    break
            train_loss_val = epoch_loss / max(1, n_batches)
            # 118-C3b: un paso por ÉPOCA (no por lote) — al menos un
            # `optimizer.step()` ya corrió en este bucle (el `plazo` se
            # comprueba DESPUÉS de él), así que esto vale también para la
            # última época si terminó a medias por plazo.
            if scheduler is not None:
                scheduler.step()

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

            if capturar is not None:
                if cap_acc is not None:
                    try:
                        cap_hist.append((epoch, cap_acc / max(1, cap_n)))
                        cap_acc.zero_()
                        cap_n = 0
                    except Exception as _exc:  # noqa: BLE001
                        cap_acc = None
                        cap_hist = []
                        cap_fallos.append(_lab.omitido_por_fallo(
                            "grad_norm", epochs, _lab.FALLO_GRADIENTE, _exc, epoca=epoch))
                if epoca_a_medias:
                    cap_parciales.append(epoch)
                if epoch in cap_programadas and cap_estados is not None:
                    try:
                        cap_estados[epoch] = _snapshot()
                    except Exception as _exc:  # noqa: BLE001
                        cap_estados = None
                        cap_fallos.append(_lab.omitido_por_fallo(
                            "pesos_por_epoca", cap_plan["n_muestra"],
                            _lab.FALLO_PESOS_POR_EPOCA, _exc, epoca=epoch))

            entry = {"epoch": epoch, "loss": train_loss_val, "val_loss": val_loss}
            if epoca_a_medias:
                entry["parcial"] = True
            epoch_trace.append(entry)
            if epoch_callback is not None:
                epoch_callback(entry)  # may raise to cancel (watchdog/cancel)

            if parado_por_plazo:
                break
            if patience is not None and no_improve >= patience:
                break

        # 133-C2: el estado de la ÚLTIMA época, antes de restaurar el mejor. Solo si
        # hay muestra que guardar: sin ella, esa copia entera de la red no la usa nadie.
        cap_ultima = epoch_trace[-1]["epoch"] if epoch_trace else None
        if (capturar is not None and cap_plan["n_muestra"] > 0 and cap_estados is not None
                and cap_ultima is not None and cap_ultima not in cap_estados):
            try:
                cap_estados[cap_ultima] = _snapshot()
            except Exception as _exc:  # noqa: BLE001
                cap_estados = None
                cap_fallos.append(_lab.omitido_por_fallo(
                    "pesos_por_epoca", cap_plan["n_muestra"],
                    _lab.FALLO_PESOS_POR_EPOCA, _exc, epoca=cap_ultima))

        # Restaurar el mejor estado (clon barato en device, sin conversión).
        with torch.no_grad():
            for p, s in zip(module.parameters(), best_state):
                p.copy_(s)

        # PESOS_GRANDES C2: decidir DESPUÉS de entrenar, con el módulo real ya
        # en memoria — param_count es una suma de .numel() (torch, O(#tensores),
        # instantáneo), nunca una iteración Python sobre valores.
        param_count = sum(p.numel() for p in module.parameters())
        should_materialize = materialize if materialize is not None else (
            param_count < torch_native_min_params()
        )
        if should_materialize:
            best_params = torch_module_to_parameter_set(network, module, parameter_set)
            best_state_dict = None
        else:
            best_params = None
            best_state_dict = dense_module_to_state_dict(network, module)

        peak_vram_gb = 0.0
        if str(device).startswith("cuda"):
            try:
                peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
            except Exception:  # noqa: BLE001
                pass

        laboratorio = None
        if capturar is not None:
            try:
                laboratorio = _bloque_laboratorio(
                    network, module, capturar, cap_plan, best_state, best_epoch,
                    cap_estados, cap_ultima, cap_hist, cap_parciales, cap_val, loss_fn,
                    param_count, torch, cada_k=cap_cada, n_lotes=cap_n_lotes,
                    fallos=cap_fallos)
            except Exception as exc:  # noqa: BLE001 — la captura nunca tumba el entrenamiento
                laboratorio = _lab.bloque_por_fallo(exc, "torch")

        resultado = {
            "best_params": best_params,
            "best_state_dict": best_state_dict,
            "materialized": should_materialize,
            "param_count": param_count,
            "epochs": epoch_trace,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_loss": train_loss_val,
            "backend": "torch",
            "device": device,
            "batch_size": bs,  # batch efectivo usado (auditable; en CUDA puede subir el del spec)
            # Las filas con las que se entrena de verdad, para poder
            # decir cuántos PASOS tuvo cada época. Sin esto habría que
            # deducirlas fuera, y eso ya salió mal otras veces.
            "train_rows": len(train_ex),
            "peak_vram_gb": round(peak_vram_gb, 2),
            # Ver `plazo` en el docstring: True = se paró por reloj y esto es la
            # mejor época vista hasta ahí, no un entrenamiento terminado.
            "parado_por_plazo": parado_por_plazo,
        }
        if laboratorio is not None:
            resultado["laboratorio"] = laboratorio
        return resultado
    finally:
        # Liberar la VRAM en el sitio donde se reservó, tanto en retorno normal como en
        # CANCELACIÓN. Clave: al cancelar, la traza de la excepción mantiene vivas estas
        # locales (módulo + datos + clon del mejor estado = los GB en GPU) mientras
        # propaga; sin borrarlas aquí, el empty_cache() de aguas arriba no puede devolver
        # la memoria y la GPU "se queda" ocupada tras Stop. Borrarlas en este frame las
        # saca de la traza y permite recuperar la VRAM de inmediato.
        try:
            del module, optimizer, train_x, train_y, val_x, val_y, best_state
            if cap_estados is not None:
                cap_estados.clear()
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
    parameter_set: ParameterSet | None,
    examples: list[tuple[list[float], list[float]]],
    loss_fn: str,
    *,
    labels: list[str] | None = None,
    device: str = "cpu",
    cancel_check: Callable[[], None] | None = None,
    state_dict: dict[str, Any] | None = None,
    target_scale: float | None = None,
) -> Any:
    """GPU-C6/M14 — evaluación de un dense_network con forward BATCHED en torch/GPU.

    Mismas métricas que `evaluate_dense_network` (reusa `result_from_predictions`),
    pero el forward corre en un solo batch sobre `device` en vez de fila a fila en
    Python. Con datasets grandes + redes anchas esto pasa de minutos/horas a ms y evita
    que un entrenamiento por GPU se quede colgado evaluando en CPU.

    PESOS_GRANDES C2: acepta el MÓDULO ya entrenado directamente vía `state_dict`
    (`dense_module_to_state_dict`/`train_dense_network_torch(...).best_state_dict`)
    — sin pasar por una `ParameterSet` con valores (round-trip lista→torch que
    era inútil incluso en modelos pequeños). Si se da `state_dict`, `parameter_set`
    se ignora (puede ser `None`); si no, se usa `parameter_set` como hasta ahora.
    """
    from matrixai.training.dense_evaluator import result_from_predictions

    if not torch_available():
        raise DenseTorchTrainError("PyTorch is not installed — torch evaluation requires torch")
    import torch

    if not examples:
        raise ValueError("examples must be non-empty")
    if state_dict is None and parameter_set is None:
        raise DenseTorchTrainError("evaluate_dense_network_torch requires parameter_set or state_dict")

    enable_tf32_if_cuda(device)  # M15(c)
    module = (
        dense_network_to_torch_module_from_state(network, state_dict, device)
        if state_dict is not None
        else dense_network_to_torch_module(network, parameter_set)
    )
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
        return result_from_predictions(predictions, targets, loss_fn, labels, target_scale)
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
    parameter_set: ParameterSet | None,
    input_dim: int,
    *,
    device: str = "cpu",
    tol: float = 1e-4,
    state_dict: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """M7 en torch/GPU — prueba de colapso (predictor constante) por el módulo torch.

    4 probes (ceros, unos, 2 aleatorios sembrados) en un solo batch sobre `device`;
    True si la salida es ~constante. Reemplaza el `_probe_model_collapse` por runtime
    Python (O(params) × 4 forwards → minutos y GBs de RAM con redes grandes); aquí es
    instantáneo. Best-effort: None si torch no está o falla.

    PESOS_GRANDES C2: acepta `state_dict` (tensores CPU del entrenamiento) como
    alternativa a `parameter_set` — mismo patrón que `evaluate_dense_network_torch`,
    sin depender de una `ParameterSet` con valores materializados."""
    if not torch_available() or input_dim <= 0:
        return None
    if state_dict is None and parameter_set is None:
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
    module = (
        dense_network_to_torch_module_from_state(network, state_dict, device)
        if state_dict is not None
        else dense_network_to_torch_module(network, parameter_set)
    )
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
