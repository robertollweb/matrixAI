# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""PESOS_GRANDES C4 — formato binario propio `.mxw` para pesos entrenados.

Decisión 4 del contrato (`48_PESOS_GRANDES_CONTRATO.md`): NO usamos `torch.save`
(pickle — cargar un fichero ajeno puede ejecutar código arbitrario, mal encaje
con la filosofía de firma/verificación de P21/P22). `.mxw` es un formato propio:

    [4 bytes]  magic "MXW1"
    [8 bytes]  longitud de la cabecera (uint64 little-endian)
    [N bytes]  cabecera JSON: {version, model_hash, parameter_schema_hash,
               tensors: [{path, shape, dtype, offset, nbytes}, ...],
               content_hash (sha256 del cuerpo), total_bytes}
    [...]      cuerpo: blobs raw float32 little-endian, uno por tensor, en el
               orden de `tensors`, sin separadores (los offsets ya lo dan todo)

Escribible/verificable con stdlib puro (`struct`, `json`, `hashlib`) — la
cabecera y el hash se pueden inspeccionar sin numpy ni torch. Cargar los
VALORES para entrenar/inferir sí requiere numpy/torch (`frombuffer`), como el
resto del core cuando hay tensores de por medio.

Escritura atómica (invariante 3): se escribe a `<path>.tmp` y se hace
`os.replace` al final — un proceso que muere a mitad nunca deja un `.mxw` a
medias donde estaba el bueno.

Nada de esto itera valores en Python: `tensor.numpy().tobytes()` y
`hashlib.update()` son operaciones vectorizadas (C), no bucles sobre floats —
mismo espíritu que el resto de PESOS_GRANDES (nunca O(#params) en Python puro).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

_MAGIC = b"MXW1"
_FORMAT_VERSION = 1
_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB — streaming copy header->file, no todo en RAM


class MxwError(ValueError):
    """Fichero `.mxw` ausente, con magic inválido, truncado o con el hash de
    contenido alterado (tamper) — siempre un error explícito, nunca pesos
    silenciosamente incorrectos."""


def write_mxw(
    path: str | Path,
    state: dict[str, Any],
    *,
    model_hash: str,
    parameter_schema_hash: str,
) -> dict[str, Any]:
    """Escribe `state` (dict `{path: tensor}`, p.ej. de
    `dense_module_to_state_dict`) como `.mxw` en `path`, atómicamente.

    Devuelve la cabecera escrita (metadata) — el caller la persiste en el
    snapshot JSON del modelo (para validar sin releer el `.mxw` entero).
    """
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    body_tmp = path.with_name(path.name + ".body.tmp")

    tensors_meta: list[dict[str, Any]] = []
    offset = 0
    hasher = hashlib.sha256()

    try:
        # Cuerpo primero: necesitamos offsets y el hash de contenido ANTES de
        # poder escribir la cabecera (que va delante en el fichero final).
        with open(body_tmp, "wb") as body:
            for name, tensor in state.items():
                arr = tensor.detach()
                if arr.device.type != "cpu":
                    arr = arr.cpu()
                if str(arr.dtype) != "torch.float32":
                    arr = arr.float()
                arr = arr.contiguous()
                raw = arr.numpy().tobytes()
                body.write(raw)
                hasher.update(raw)
                tensors_meta.append({
                    "path": name,
                    "shape": list(arr.shape),
                    "dtype": "float32",
                    "offset": offset,
                    "nbytes": len(raw),
                })
                offset += len(raw)

        header = {
            "version": _FORMAT_VERSION,
            "model_hash": model_hash,
            "parameter_schema_hash": parameter_schema_hash,
            "tensors": tensors_meta,
            "content_hash": hasher.hexdigest(),
            "total_bytes": offset,
        }
        header_bytes = json.dumps(header).encode("utf-8")

        with open(tmp_path, "wb") as out, open(body_tmp, "rb") as body:
            out.write(_MAGIC)
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)
            while True:
                chunk = body.read(_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)

        os.replace(tmp_path, path)  # atómico en el mismo filesystem
        return header
    finally:
        for p in (tmp_path, body_tmp):
            try:
                p.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass


def _read_header(f: Any, path: Path) -> tuple[dict[str, Any], int]:
    magic = f.read(4)
    if magic != _MAGIC:
        raise MxwError(f"{path}: no es un fichero .mxw válido (magic incorrecto)")
    len_bytes = f.read(8)
    if len(len_bytes) != 8:
        raise MxwError(f"{path}: fichero truncado (falta la longitud de cabecera)")
    (header_len,) = struct.unpack("<Q", len_bytes)
    raw_header = f.read(header_len)
    if len(raw_header) != header_len:
        raise MxwError(f"{path}: fichero truncado (cabecera incompleta)")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MxwError(f"{path}: cabecera .mxw corrupta ({exc})") from exc
    body_start = 4 + 8 + header_len
    return header, body_start


def read_mxw_header(path: str | Path) -> dict[str, Any]:
    """Lee SOLO la cabecera (metadata) — barato, sin traer los blobs a RAM.
    Útil para listar/validar un modelo sin cargar sus pesos."""
    path = Path(path)
    if not path.exists():
        raise MxwError(f"{path}: fichero .mxw no encontrado")
    with open(path, "rb") as f:
        header, _ = _read_header(f, path)
    return header


def read_mxw_header_and_body_start(path: str | Path) -> tuple[dict[str, Any], int]:
    """Como `read_mxw_header` pero devuelve también el offset (bytes) donde
    empieza el cuerpo — necesario para `stream_mxw_tensor` (leer un tensor
    concreto sin traer el cuerpo entero a RAM). PESOS_GRANDES C7 auditoría:
    la base del export ONNX external-data por STREAMING (un modelo de 15 GiB
    no cabe en RAM ni debe copiarse dos veces)."""
    path = Path(path)
    if not path.exists():
        raise MxwError(f"{path}: fichero .mxw no encontrado")
    with open(path, "rb") as f:
        return _read_header(f, path)


def validate_mxw_tensor_meta(meta: dict[str, Any], path: Any = "") -> tuple[str, int, int, list[int]]:
    """Valida la coherencia interna de la metadata de UN tensor de la cabecera
    (`shape`/`offset`/`nbytes`, mismos chequeos que hace `read_mxw` antes de
    rehidratar) y devuelve `(name, offset, nbytes, shape)`. La cabecera NO está
    cubierta por `content_hash`, así que un `.mxw` manipulado a mano podría
    traer metadata incoherente — este chequeo la ataja con `MxwError` explícito
    tanto en el camino que materializa (`read_mxw`) como en el que solo
    streamea bytes (`stream_mxw_tensor`)."""
    name = meta.get("path")
    try:
        offset = int(meta["offset"])
        nbytes = int(meta["nbytes"])
        shape = [int(d) for d in meta["shape"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise MxwError(f"{path}: metadata de tensor {name!r} inválida en la cabecera ({exc})") from exc
    if offset < 0 or nbytes < 0:
        raise MxwError(f"{path}: tensor {name!r} con offset/nbytes negativos ({offset}/{nbytes})")
    if nbytes % 4 != 0:
        raise MxwError(
            f"{path}: tensor {name!r} con nbytes={nbytes} no es múltiplo de 4 (float32) — cabecera corrupta."
        )
    expected_elems = 1
    for d in shape:
        if d < 0:
            raise MxwError(f"{path}: tensor {name!r} con dimensión negativa {d} en shape")
        expected_elems *= d
    if expected_elems * 4 != nbytes:
        raise MxwError(
            f"{path}: tensor {name!r} incoherente — shape {shape} implica "
            f"{expected_elems * 4} bytes pero la cabecera declara nbytes={nbytes}."
        )
    return name, offset, nbytes, shape


def stream_mxw_tensor(f: Any, body_start: int, meta: dict[str, Any], out: Any,
                      chunk_bytes: int = _CHUNK_BYTES) -> int:
    """Copia los bytes de UN tensor del `.mxw` (fichero abierto `f`, con el
    cuerpo empezando en `body_start`) a `out` (file-like binario) por chunks,
    sin traer el tensor entero a RAM. Devuelve los bytes copiados.

    PESOS_GRANDES C7 auditoría: el corazón del export external-data por
    streaming — `f` puede ser el `.mxw` de 15 GiB y `out` un entry de un zip
    o el `.onnx.data`; en ningún momento hay más de `chunk_bytes` en memoria
    (a diferencia de `read_mxw`, que trae el cuerpo entero + copias)."""
    name, offset, nbytes, _shape = validate_mxw_tensor_meta(meta)
    f.seek(body_start + offset)
    remaining = nbytes
    while remaining:
        buf = f.read(min(chunk_bytes, remaining))
        if not buf:
            raise MxwError(f"tensor {name!r} truncado en el cuerpo del fichero .mxw")
        out.write(buf)
        remaining -= len(buf)
    return nbytes


def read_mxw(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Lee un `.mxw` completo → `dict[str, torch.Tensor]` (CPU, float32).

    Con `verify=True` (default) recalcula el hash de contenido y lo compara
    contra la cabecera — un sidecar corrupto o manipulado falla con
    `MxwError` explícito en vez de servir pesos incorrectos en silencio.
    """
    import numpy as np
    import torch

    path = Path(path)
    if not path.exists():
        raise MxwError(f"{path}: fichero .mxw no encontrado")
    with open(path, "rb") as f:
        header, _ = _read_header(f, path)
        body = f.read()

    expected_bytes = header.get("total_bytes")
    if expected_bytes is not None and len(body) != expected_bytes:
        raise MxwError(
            f"{path}: tamaño del cuerpo no coincide con la cabecera "
            f"({len(body)} bytes leídos, {expected_bytes} esperados) — "
            "fichero truncado o corrupto."
        )
    if verify:
        actual_hash = hashlib.sha256(body).hexdigest()
        expected_hash = header.get("content_hash")
        if actual_hash != expected_hash:
            raise MxwError(
                f"{path}: el hash de contenido no coincide (esperado "
                f"{expected_hash}, calculado {actual_hash}) — el fichero .mxw "
                "ha sido modificado o está corrompido."
            )

    result: dict[str, Any] = {}
    for meta in header.get("tensors", []):
        # PESOS_GRANDES C4 audit (reauditoría Opus, BAJA): la cabecera NO está
        # cubierta por `content_hash`, así que una cabecera con
        # `shape`/`offset`/`nbytes` incoherentes puede pasar los chequeos de
        # arriba y luego reventar en `frombuffer`/`reshape` con un `ValueError`
        # pelado que se escaparía del `except MxwError` de los callers → 500.
        # `validate_mxw_tensor_meta` la ataja con `MxwError` explícito (misma
        # validación que usa el camino de streaming, C7 auditoría).
        name, offset, nbytes, shape = validate_mxw_tensor_meta(meta, path)
        if offset + nbytes > len(body):
            raise MxwError(
                f"{path}: tensor {name!r} apunta fuera del cuerpo del fichero "
                f"(offset={offset}, nbytes={nbytes}, cuerpo={len(body)} bytes) — cabecera corrupta."
            )
        chunk = body[offset: offset + nbytes]
        if len(chunk) != nbytes:
            raise MxwError(
                f"{path}: tensor {name!r} truncado en el cuerpo del fichero"
            )
        try:
            arr = np.frombuffer(chunk, dtype=np.float32).reshape(shape)
            result[name] = torch.from_numpy(arr.copy())  # copy: tensor propio, no alias de `body`
        except (ValueError, TypeError) as exc:
            raise MxwError(
                f"{path}: no se pudo rehidratar el tensor {name!r} ({exc}) — cabecera corrupta."
            ) from exc
    return result
