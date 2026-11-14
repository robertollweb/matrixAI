# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""CONTRATO 167-C1 — la retropropagación PASO A PASO de UN peso (mini-ejecución).

Reproduce los primeros lotes de la primera época del entrenador torch denso (misma
semilla, mismo orden de operaciones sobre el generador, mismo SGD) y, para un peso
elegido de una capa Dense, anota en cada lote: valor antes, gradiente, tasa, gradiente
efectivo (`g + decaimiento·w`), valor después calculado a mano (`w − tasa × g_ef`) y el
valor que deja el `optimizer.step()` REAL. La paridad entre los dos es la comprobación
del invariante 2: si no cuadra, `ok` es False con motivo y no se dibuja.

No toca `dense_torch_trainer.py` ni el generador de nadie más: es una ejecución aparte.
Solo SGD (D1 = A): con Adam/AdamW el paso depende de momentos y se declara `ok: False`.

SOLO EN CPU, y eso es una condición, no un detalle (auditoría 167, B1): en CUDA el entrenador
real usa otro lote (`effective_batch_size("cuda", 8, 160)` = 160, no 8) y otra aritmética
(TF32), así que esta repetición NO es la de una ejecución en GPU. Quien llama solo la lanza si
la ejecución real entrenó en CPU, y además compara `primera_epoca` (lote efectivo, filas de
entrenamiento, lotes y pérdida media de la época 1, calculados igual que el entrenador) con lo
que la ejecución real DEJÓ ESCRITO: si algo no coincide, no se enseña nada. Por eso la
mini-ejecución recorre la PRIMERA ÉPOCA ENTERA (anota solo los primeros `n_pasos` lotes): la
pérdida de la época 1 del entrenador es la media de todos sus lotes.
La paridad en GPU (repetir en el dispositivo de la ejecución) NO está hecha: es otro corte.
"""
from __future__ import annotations

import math
from typing import Any

TOPE_PASOS = 20                 # D3: los 20 primeros lotes de la primera época
TOLERANCIA_ABS = 1e-7           # paridad con optimizer.step(): |a mano − torch| <= ABS + REL·|w|
TOLERANCIA_REL = 1e-6           # (float32: un ulp de un peso ~1 es 6e-8)

MOTIVO_SOLO_SGD = {
    "es": "Con Adam y AdamW el paso depende de los momentos, no es «gradiente por tasa»: solo se enseña con SGD.",
    "en": "With Adam and AdamW the step depends on the moments, it is not «gradient times rate»: only shown with SGD.",
}
MOTIVO_NO_FINITO = {
    "es": "La pérdida o el gradiente no son finitos en este lote: el paso no se puede enseñar.",
    "en": "The loss or the gradient is not finite on this batch: the step cannot be shown.",
}
MOTIVO_PARIDAD = {
    "es": "El valor calculado a mano no coincide con el que deja el optimizador: no se dibuja.",
    "en": "The hand-computed value does not match what the optimizer leaves: it is not drawn.",
}
MOTIVO_PESO = {
    "es": "Ese peso no existe en la red (capa, fila o columna fuera de rango).",
    "en": "That weight does not exist in the network (layer, row or column out of range).",
}
MOTIVO_SIN_TORCH = {
    "es": "PyTorch no está instalado: no hay mini-ejecución.",
    "en": "PyTorch is not installed: there is no mini-run.",
}


def huella_de_los_pesos(valores: Any) -> str:
    """Huella de una lista de pesos (W1, b1, W2, b2… aplanados, en ese orden) como float32: la de la repetición y la de los
    pesos que la ejecución real guardó en su bloque del laboratorio se calculan con ESTA función, a los dos lados. Un NaN
    vale siempre lo mismo (sus bits de carga no dicen nada)."""
    import hashlib
    from array import array
    nan = float("nan")
    datos = array("f", (nan if v != v else v for v in valores))
    return hashlib.sha256(datos.tobytes()).hexdigest()


def _fallo(motivo: dict[str, str], **extra: Any) -> dict[str, Any]:
    return {"ok": False, "motivo": dict(motivo), "pasos": [], **extra}


def paso_de_un_peso(
    network: Any,
    parameter_set: Any,
    examples: list[tuple[list[float], list[float]]],
    loss_fn: str,
    *,
    capa: int,
    fila: int,
    columna: int,
    lr: float = 0.01,
    seed: int = 42,
    batch_size: int | None = None,
    validation_examples: list[tuple[list[float], list[float]]] | None = None,
    initial_state_dict: dict[str, Any] | None = None,
    optimizer: str | None = None,
    weight_decay: float = 0.0,
    n_pasos: int = TOPE_PASOS,
) -> dict[str, Any]:
    """`capa` es 1-based (como `Net.W{capa}`); `fila`/`columna` 0-based de W (salidas × entradas).

    Los argumentos son los del entrenador (mismos nombres) para que quien llama pase lo
    mismo que pasó al estudio. Devuelve un dict serializable; ver el módulo.
    """
    from matrixai.forward.dense_torch import (
        dense_network_to_torch_module, dense_network_to_torch_module_from_state)
    from matrixai.training.dense_torch_trainer import (
        _loss_on_probabilities, effective_batch_size)
    from matrixai.training.dense_torch_trainer import torch_available

    opt_name = optimizer or "sgd"
    if opt_name != "sgd":
        return _fallo(MOTIVO_SOLO_SGD, optimizador=opt_name)
    if not torch_available():
        return _fallo(MOTIVO_SIN_TORCH)
    if not examples:
        raise ValueError("no training examples")
    import torch

    n_pasos_pedido = int(n_pasos)
    n_pasos = max(1, min(n_pasos_pedido, TOPE_PASOS))

    # MISMO ORDEN que train_dense_network_torch: sembrar, construir el módulo (puede
    # consumir el generador con el init nativo), partir, tensores, optimizador, randperm.
    torch.manual_seed(seed)
    module = (
        dense_network_to_torch_module_from_state(network, initial_state_dict, "cpu")
        if initial_state_dict is not None
        else dense_network_to_torch_module(network, parameter_set)
    )
    module._torch_module.to("cpu")
    n_capas = len(module.linears)
    if not (1 <= capa <= n_capas):
        return _fallo(MOTIVO_PESO)
    peso = module.linears[capa - 1].weight
    if not (0 <= fila < peso.shape[0] and 0 <= columna < peso.shape[1]):
        return _fallo(MOTIVO_PESO)

    if validation_examples is not None:
        train_ex = examples
    else:
        split = max(1, int(len(examples) * 0.8)) if len(examples) > 1 else len(examples)
        train_ex = examples[:split]
    train_x = torch.tensor([x for x, _ in train_ex], dtype=torch.float32)
    train_y = torch.tensor([y for _, y in train_ex], dtype=torch.float32)
    optim_ = torch.optim.SGD(module.parameters(), lr=lr, weight_decay=weight_decay)
    bs = effective_batch_size("cpu", batch_size, len(train_ex))
    n_lotes_epoca = -(-train_x.shape[0] // bs)
    # Los pasos que se enseñan son los que HAY: con 1 o 4 lotes en la primera época, 1 o 4
    # (auditoría 167, M2: decía 20). `pasos_pedidos` sigue diciendo lo que se pidió.
    n_pasos = min(n_pasos, n_lotes_epoca)

    module.train(True)
    perm = torch.randperm(train_x.shape[0])
    pasos: list[dict[str, Any]] = []
    motivo: dict[str, str] | None = None
    max_err = 0.0
    # La época 1 ENTERA, con la misma cuenta de la pérdida que el entrenador (`epoch_loss +=
    # float(loss.detach())` lote a lote, y la media al final): es lo que se compara con lo que
    # dejó escrito la ejecución real. Tras un paso no finito se deja de anotar, pero la época
    # sigue como en el entrenador (que tampoco para).
    perdida_epoca = 0.0
    n_lotes = 0
    for k, i in enumerate(range(0, train_x.shape[0], bs)):
        idx = perm[i:i + bs]
        xb, yb = train_x[idx], train_y[idx]
        optim_.zero_grad()
        loss = _loss_on_probabilities(module(xb), yb, loss_fn, torch)
        loss.backward()
        perdida = float(loss.detach())
        anotar = k < n_pasos and motivo is None
        a_mano = None
        if anotar:
            w0 = peso[fila, columna].detach().clone()
            g = peso.grad[fila, columna].detach().clone()
            if not (bool(torch.isfinite(g)) and math.isfinite(perdida)):
                pasos.append({"paso": k + 1, "peso_antes": float(w0), "gradiente": None,
                              "perdida": None, "no_finito": True})
                motivo = MOTIVO_NO_FINITO
                anotar = False
            else:
                g_ef = g + weight_decay * w0               # lo que SGD usa con decaimiento
                a_mano = w0 - lr * g_ef                    # w ← w − tasa × gradiente efectivo
        optim_.step()
        perdida_epoca += perdida
        n_lotes += 1
        if anotar and a_mano is not None:
            real = peso[fila, columna].detach().clone()
            err = abs(float(a_mano) - float(real))
            max_err = max(max_err, err)
            pasos.append({
                "paso": k + 1,
                "filas_del_lote": int(idx.numel()),
                "perdida": perdida,
                "peso_antes": float(w0),
                "gradiente": float(g),
                "decaimiento": float(weight_decay),
                "gradiente_efectivo": float(g_ef),
                "tasa": float(lr),
                "paso_aplicado": float(lr * g_ef),
                "peso_despues": float(a_mano),
                "peso_despues_torch": float(real),
                "error": err,
                "cuadra": err <= TOLERANCIA_ABS + TOLERANCIA_REL * abs(float(real)),
            })
    paridad_ok = motivo is None and bool(pasos) and all(p["cuadra"] for p in pasos)
    if motivo is None and not paridad_ok:
        motivo = MOTIVO_PARIDAD
    return {
        "ok": paridad_ok,
        "motivo": dict(motivo) if motivo else None,
        "optimizador": "sgd",
        "capa": capa, "fila": fila, "columna": columna,
        "tasa": float(lr), "decaimiento": float(weight_decay),
        "tope_pasos": TOPE_PASOS, "pasos_pedidos": n_pasos_pedido,
        "pasos_aplicados": n_pasos, "lotes_por_epoca": n_lotes_epoca,
        "tamano_de_lote": bs,
        "paridad": {"tolerancia_abs": TOLERANCIA_ABS, "tolerancia_rel": TOLERANCIA_REL,
                    "error_maximo": max_err},
        # Lo que la ejecución real también deja escrito (`effort` y `epochs[0]` de su resultado):
        # el cinturón de quien llama. `perdida` es la media de la época 1, sumada como el
        # entrenador (no finita si el entrenamiento divergió en ella: se compara igual).
        # `huella_de_los_pesos`: TODOS los pesos al acabar la época 1, para compararlos con los que el bloque del
        # laboratorio guardó de esa época (cuando los guardó).
        "primera_epoca": {"tamano_de_lote": bs, "filas_de_entrenamiento": len(train_ex),
                          "lotes": n_lotes, "perdida": perdida_epoca / max(1, n_lotes),
                          "huella_de_los_pesos": huella_de_los_pesos(
                              v for p in module.parameters() for v in p.detach().flatten().tolist())},
        "pasos": pasos,
    }
