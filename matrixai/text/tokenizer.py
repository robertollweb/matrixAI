# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""SECUENCIAS_PRODUCTO C1 — tokenizador byte-level propio (`byte_v1`).

Decisión 1 del contrato (`52_SECUENCIAS_PRODUCTO_CONTRACT.md`): cero
dependencias, determinista, sin entrenamiento y sin fichero de vocabulario
que distribuir — un texto UTF-8 se codifica byte a byte (vocab base 256,
0-255) más tres ids especiales: `PAD=256` (relleno), `UNK=257` (reservado
para vocabularios futuros — nunca lo emite `encode`, todo byte 0-255 es
válido por construcción de UTF-8) y `CLS=258` (token de clasificación
opcional, para BLOCK TRANSFORMER POOL cls). BPE/WordPiece quedan fuera de
alcance (contrato futuro), igual que en P11.
"""
from __future__ import annotations

from typing import Any


class ByteTokenizer:
    """Tokenizador byte-level determinista de longitud fija `length`.

    `encode`/`decode` son inversas entre sí para cualquier texto cuya
    codificación UTF-8 quepa en `length` bytes (menos uno si `add_cls`);
    textos más largos se truncan (pérdida de información, no de bytes
    parciales: `decode` nunca revienta con una secuencia UTF-8 cortada a
    medias, la reemplaza).
    """

    PAD = 256
    UNK = 257
    CLS = 258
    BASE_VOCAB = 256
    VOCAB_SIZE = 259

    def __init__(self, length: int) -> None:
        if not isinstance(length, int) or length < 1:
            raise ValueError(f"ByteTokenizer length must be a positive integer, got {length!r}")
        self.length = length

    def encode(self, text: str, *, add_cls: bool = False) -> list[int]:
        """UTF-8 → bytes (cada byte es su propio id, 0-255), truncado a
        `length` (a `length - 1` si `add_cls`, dejando sitio al CLS inicial),
        relleno con `PAD` hasta `length`."""
        raw = list(text.encode("utf-8"))
        if add_cls:
            ids = [self.CLS] + raw[: self.length - 1]
        else:
            ids = raw[: self.length]
        if len(ids) < self.length:
            ids = ids + [self.PAD] * (self.length - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Best-effort: descarta PAD/CLS/UNK (y cualquier id fuera de rango)
        y decodifica el resto como UTF-8, sustituyendo bytes inválidos
        (`errors="replace"`) en vez de lanzar — una secuencia multibyte
        cortada por el truncado de `encode` nunca debe reventar `decode`."""
        raw = bytes(i for i in ids if 0 <= i < self.BASE_VOCAB)
        return raw.decode("utf-8", errors="replace")

    def config(self) -> dict[str, Any]:
        """Metadata serializable del tokenizador — viaja en `field_seq`/
        `inference_spec.json` (invariante 2 del contrato: mismo texto →
        mismos ids en Studio, CLI y predict.py exportado)."""
        return {
            "kind": "byte_v1",
            "length": self.length,
            "vocab_size": self.VOCAB_SIZE,
            "pad": self.PAD,
            "cls": self.CLS,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ByteTokenizer:
        kind = config.get("kind")
        if kind != "byte_v1":
            raise ValueError(f"Unknown tokenizer kind {kind!r}; expected 'byte_v1'")
        return cls(int(config["length"]))
