# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C1 — estimar recursos ANTES de entrenar.

Antes de lanzar un entrenamiento, el usuario decide con qué formato guardar los
pesos (ver `48_PESOS_GRANDES_CONTRATO.md`, decisión 1): `json` (portable, coste
O(params) en CPU/RAM/disco) o `binary` (sidecar `.mxw`, minutos, ~4 bytes/valor).
Esta estimación es la información con la que elige — nunca bloquea (invariante 6),
solo informa, y sus tasas son constantes MEDIDAS (ver el diagnóstico del contrato,
2026-07-03), no números mágicos dispersos.

`param_count` se deriva de `BackendContractAnalyzer` — reusa el manifest ya
existente (shapes por tensor), nunca materializa los valores: es O(#tensores),
no O(#params). El mismo manifest, filtrado por `role == "bias"`, da el ancho de
salida de cada capa (una bias por capa densa) para la estimación de activaciones.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_GIB = 1024 ** 3
BYTES_PER_FLOAT32 = 4

# PESOS_GRANDES decisión 1: umbral que fija el DEFAULT del formato de
# persistencia (json por debajo, binario por encima — C4) y, desde C2, si el
# trainer torch materializa la ParameterSet-con-valores o devuelve un
# state_dict de tensores. Configurable (la "máquina del usuario decide"),
# nunca hardcodeado dos veces.
DEFAULT_TORCH_NATIVE_MIN_PARAMS = 50_000_000
_TORCH_NATIVE_MIN_PARAMS_ENV = "MATRIXAI_TORCH_NATIVE_MIN_PARAMS"


def torch_native_min_params() -> int:
    """Umbral de parámetros (env `MATRIXAI_TORCH_NATIVE_MIN_PARAMS`, default 50M).

    Un valor inválido o <=0 cae al default (nunca desactiva el umbral por accidente
    con un typo — mismo espíritu que los overrides de `matrixai.limits`)."""
    raw = os.environ.get(_TORCH_NATIVE_MIN_PARAMS_ENV)
    if raw is None:
        return DEFAULT_TORCH_NATIVE_MIN_PARAMS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TORCH_NATIVE_MIN_PARAMS
    return value if value > 0 else DEFAULT_TORCH_NATIVE_MIN_PARAMS


# PESOS_GRANDES C6: a diferencia de `torch_native_min_params` (una decisión de
# producto, ajustable), este es un límite FÍSICO del formato protobuf que usa
# ONNX (guarda los pesos EN LÍNEA en el mensaje; protobuf rechaza serializar
# mensajes >2 GiB) — no es una política, así que no tiene override por env.
# Un valor algo por debajo de los 2 GiB reales de protobuf (2**31-1 bytes) deja
# margen para el resto del grafo (nodos, nombres, metadata), pequeño frente al
# peso de los tensores en un modelo grande.
ONNX_PROTOBUF_LIMIT_GIB = 1.9


# Tasas MEDIDAS 2026-07-03 (ver "Diagnóstico" en 48_PESOS_GRANDES_CONTRATO.md):
# listas Python (tolist()) y json.dumps sobre listas de floats, en esta máquina
# (≈ CPU de Colab). Constantes nombradas, no mágicas.
PYTHON_LIST_BYTES_PER_VALUE = 32.0
JSON_BYTES_PER_VALUE = 20.3
TOLIST_VALUES_PER_SECOND = 11_000_000.0
JSON_DUMP_VALUES_PER_SECOND = 1_900_000.0
# Velocidad de escritura a disco asumida para el sidecar binario — conservadora
# (SSD lento / red lenta tipo Drive); documentada y nombrada, no oculta en una
# fórmula. Ajustable si hace falta (no hay env override: es una estimación, no
# un límite operativo).
ASSUMED_DISK_WRITE_BYTES_PER_SECOND = 100_000_000.0  # 100 MB/s
# Margen sobre pesos+gradientes+activaciones para overhead de CUDA (fragmentación,
# buffers temporales del kernel, contexto). No es una medición fina; es un margen
# de seguridad documentado para que la estimación avise ANTES de un OOM, no después.
VRAM_MARGIN_FACTOR = 1.2
# Batch por defecto cuando no se conoce el nº de filas del dataset (estimación
# "a ciegas", antes de generar/subir datos). En CPU y CUDA respectivamente,
# alineado con los defaults de `dense_torch_trainer.effective_batch_size`.
_DEFAULT_BATCH_CPU = 2048
_DEFAULT_BATCH_CUDA = 16384


@dataclass(frozen=True)
class ResourceEstimate:
    """Estimación orientativa (invariante 6) de recursos para un modelo dado."""

    param_count: int
    weights_gib: float
    vram_train_gib: float
    effective_batch: int
    json_ram_gib: float
    json_disk_gib: float
    json_time_seconds: float
    binary_ram_gib: float
    binary_disk_gib: float
    binary_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_count": self.param_count,
            "weights_gib": round(self.weights_gib, 4),
            "vram_train_gib": round(self.vram_train_gib, 4),
            "effective_batch": self.effective_batch,
            "json": {
                "ram_gib": round(self.json_ram_gib, 4),
                "disk_gib": round(self.json_disk_gib, 4),
                "time_seconds": round(self.json_time_seconds, 2),
            },
            "binary": {
                "ram_gib": round(self.binary_ram_gib, 4),
                "disk_gib": round(self.binary_disk_gib, 4),
                "time_seconds": round(self.binary_time_seconds, 2),
            },
            # Invariante 6: la estimación es orientativa, nunca bloquea.
            "orientative": True,
        }


def _manifest_shapes_and_widths(program: Any) -> tuple[list[list[int]], list[int]]:
    """Shapes de cada tensor entrenable + anchos de salida por capa (bias.shape[0]).

    Vía `BackendContractAnalyzer`: el manifest ya existe para cualquier programa
    (dense_network, composite_network, o el mundo LAYER/FUNCTION de P1-P11) y es
    O(#tensores) — nunca materializa los `initial_value` (ver fix 9502f56).

    Auditoría: un tensor entrenable ESCALAR tiene `shape=[]` en el manifest (no
    ausencia de shape) — p.ej. el bias de `sigmoid_linear`/`linear_regression`
    (ver `test_training_contract.py`, `b1` con `shape=[]`). `shape is None` (la
    clave "shape" ni aparece) es lo único que significa "sin información de
    shape, no contar". Confundir `[]` con `None` (`if not shape`) descontaba
    cada escalar entrenable del `param_count`.
    """
    from matrixai.compiler.backend_contract import BackendContractAnalyzer

    report = BackendContractAnalyzer().analyze(program)
    manifest = report.to_dict().get("parameter_manifest") or []
    shapes: list[list[int]] = []
    widths: list[int] = []
    for entry in manifest:
        shape = entry.get("shape")
        if shape is None:
            continue
        shapes.append(list(shape))
        if entry.get("role") == "bias" and shape:
            widths.append(int(shape[0]))
    return shapes, widths


def _param_count(shapes: list[list[int]]) -> int:
    total = 0
    for shape in shapes:
        prod = 1
        for d in shape:
            prod *= int(d)
        total += prod
    return total


def _batch_from_training_text(training_text: str) -> int | None:
    """El BATCH size REAL declarado en el `.mxtrain`, si lo hay.

    Auditoría: la tarjeta del Studio solo pasaba `rows`, nunca el batch — un
    `.mxtrain` con `BATCH size=65536` explícito quedaba invisible para la
    estimación de VRAM (que entonces usaba el default del dispositivo, pudiendo
    subestimarla y no disparar el aviso). Reusa el parser del core (una sola
    fuente de verdad del BATCH declarado, no una regex duplicada en el frontend).
    """
    if not training_text:
        return None
    try:
        from matrixai.training.parser import parse_training_text
        spec = parse_training_text(training_text)
    except Exception:  # noqa: BLE001
        return None
    if spec.dataset and spec.dataset.batch:
        return spec.dataset.batch.size
    return None


def _resolve_effective_batch(device: str, batch: int | None, rows: int) -> int:
    """Batch efectivo, con la MISMA regla que aplicará el entrenamiento real.

    Límite conocido (no bloqueante, señalado en la auditoría C1 2026-07-03):
    en CUDA, `effective_batch_size` IGNORA un batch pequeño del `.mxtrain`
    (`BATCH size=8` autogenerado) y sube al default de GPU — pero solo cuando
    conoce `rows` (necesita `n_train` para capar). Si se llama a
    `estimate_model_resources` con `training_text` pero SIN `rows` en CUDA,
    este helper cae a la rama de abajo y devuelve el batch del `.mxtrain` TAL
    CUAL (p.ej. 8), subestimando el batch — y por tanto la VRAM — que el
    entrenamiento real usaría (16384 por defecto). La tarjeta del Studio
    siempre pasa `rows` (el tamaño del CSV generado/subido), así que el flujo
    de producto no lo sufre; un caller directo (CLI, script, futuro C2-C6) que
    omita `rows` en CUDA debe saber que la estimación es más optimista de lo
    real. Corregir esto requeriría un `n_train` "desconocido pero grande" o
    aplicar la regla de CUDA sin cap — se deja así a propósito hasta que haya
    un caso de uso real que lo necesite.
    """
    if rows and rows > 0:
        from matrixai.training.dense_torch_trainer import effective_batch_size
        return effective_batch_size(device, batch, rows)
    if batch and batch > 0:
        return batch
    return _DEFAULT_BATCH_CUDA if str(device).startswith("cuda") else _DEFAULT_BATCH_CPU


def estimate_model_resources(
    program: Any,
    *,
    rows: int = 0,
    batch: int | None = None,
    training_text: str | None = None,
    device: str = "cpu",
) -> ResourceEstimate:
    """Estimación de recursos ANTES de entrenar (PESOS_GRANDES C1).

    `program` es un `MatrixAIProgram` ya parseado (el `.mxai` generado). `rows` es
    el tamaño del dataset (0 si aún no se conoce — la estimación de VRAM usa el
    batch por defecto del dispositivo en ese caso); `batch` es un tamaño de lote
    explícito si el usuario ya lo fijó (gana sobre lo declarado en `.mxtrain`);
    `training_text` es el `.mxtrain` generado — si `batch` no se pasa, se deriva
    de su `BATCH size=...` real (auditoría: antes se ignoraba y un BATCH grande
    explícito podía subestimar la VRAM); `device` es "cpu" o "cuda" (o "cuda:N").

    Nota de alcance (corte C1, decisión al implementar): el borrador del contrato
    proponía además aceptar `shapes` crudas y un parámetro `epochs`. `epochs` no
    afecta a ninguna de las magnitudes que este corte estima (memoria/disco/tiempo
    de guardado; el tiempo de ENTRENAMIENTO se extrapola en vivo tras la 1ª época,
    no aquí) — incluirlo sin uso sería un parámetro muerto, así que se omite.
    El input alternativo por `shapes` crudas no tenía un caso de uso real (siempre
    hay un `program` parseado en el momento de estimar) y se deja fuera; añadir si
    aparece un caso de uso concreto.
    """
    shapes, widths = _manifest_shapes_and_widths(program)
    param_count = _param_count(shapes)

    weights_bytes = param_count * BYTES_PER_FLOAT32
    weights_gib = weights_bytes / _GIB

    if batch is None and training_text:
        batch = _batch_from_training_text(training_text)
    eff_batch = _resolve_effective_batch(device, batch, rows)

    # VRAM de entrenamiento: pesos + gradientes (SGD sin momentum — verificado en
    # dense_torch_trainer.py, sin estados de optimizador que dupliquen esto de
    # nuevo) + activaciones (batch x anchos, formula literal del contrato) + margen.
    activations_bytes = eff_batch * sum(widths) * BYTES_PER_FLOAT32
    vram_train_bytes = (weights_bytes * 2 + activations_bytes) * VRAM_MARGIN_FACTOR
    vram_train_gib = vram_train_bytes / _GIB

    # Formato JSON (hoy): tolist() + json.dumps, tasas medidas.
    json_ram_bytes = param_count * PYTHON_LIST_BYTES_PER_VALUE
    json_disk_bytes = param_count * JSON_BYTES_PER_VALUE
    json_time_seconds = (
        param_count / TOLIST_VALUES_PER_SECOND
        + param_count / JSON_DUMP_VALUES_PER_SECOND
    )

    # Formato binario (.mxw, C4): una copia CPU fp32 + escritura a disco.
    binary_ram_bytes = param_count * BYTES_PER_FLOAT32
    binary_disk_bytes = param_count * BYTES_PER_FLOAT32
    binary_time_seconds = binary_disk_bytes / ASSUMED_DISK_WRITE_BYTES_PER_SECOND

    return ResourceEstimate(
        param_count=param_count,
        weights_gib=weights_gib,
        vram_train_gib=vram_train_gib,
        effective_batch=eff_batch,
        json_ram_gib=json_ram_bytes / _GIB,
        json_disk_gib=json_disk_bytes / _GIB,
        json_time_seconds=json_time_seconds,
        binary_ram_gib=binary_ram_bytes / _GIB,
        binary_disk_gib=binary_disk_bytes / _GIB,
        binary_time_seconds=binary_time_seconds,
    )
