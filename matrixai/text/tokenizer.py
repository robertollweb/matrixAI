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
    _REQUIRED_CONFIG_KEYS = ("kind", "length", "vocab_size", "pad", "cls")

    def __init__(self, length: int) -> None:
        # auditoría C1 [BAJA]: `bool` hereda de `int` — `ByteTokenizer(True)`
        # pasaría `isinstance(length, int)` y crearía un tokenizador de
        # longitud 1 por accidente. `type(length) is int` lo excluye.
        if type(length) is not int or length < 1:
            raise ValueError(f"ByteTokenizer length must be a positive integer, got {length!r}")
        self.length = length

    def encode(self, text: str, *, add_cls: bool = False) -> list[int]:
        """UTF-8 → bytes (cada byte es su propio id, 0-255), truncado a
        `length` (a `length - 1` si `add_cls`, dejando sitio al CLS inicial),
        relleno con `PAD` hasta `length`."""
        # auditoría C1 [BAJA]: entrada no-str daba un AttributeError críptico
        # (bytes/None/int no tienen `.encode`); `add_cls` no-bool (p.ej. una
        # cadena "false", truthy) colaba un CLS que el caller no pidió.
        if not isinstance(text, str):
            raise TypeError(f"ByteTokenizer.encode expects str, got {type(text).__name__}")
        if type(add_cls) is not bool:
            raise TypeError(f"add_cls must be bool, got {type(add_cls).__name__}")
        # auditoría C1 [BAJA]: truncar el `bytes` ANTES de convertir a list
        # — `list(text.encode(...))` materializaba O(len(texto)) enteros
        # aunque `length` capa el resultado a unas pocas decenas.
        raw = text.encode("utf-8")
        limit = self.length - 1 if add_cls else self.length
        ids = list(raw[:limit])
        if add_cls:
            ids = [self.CLS] + ids
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
        """Reconstruye el tokenizador desde `config()` — validación ESTRICTA
        (auditoría C1 [MEDIA]): `inference_spec.json`/`field_seq` son entrada
        EXTERNA (invariante 2 del contrato: "si el spec y el modelo no
        casan, el export falla con razón visible" — nunca en silencio, y
        menos aún reconstruyendo un tokenizador DISTINTO al declarado). Antes
        `int(config["length"])` normalizaba `3.9`/`"4"` sin avisar y
        `vocab_size`/`pad`/`cls` ni se miraban."""
        if not isinstance(config, dict):
            raise ValueError(f"tokenizer config must be an object, got {type(config).__name__}")
        missing = [k for k in cls._REQUIRED_CONFIG_KEYS if k not in config]
        if missing:
            raise ValueError(f"tokenizer config missing required keys: {missing}")
        kind = config["kind"]
        if kind != "byte_v1":
            raise ValueError(f"Unknown tokenizer kind {kind!r}; expected 'byte_v1'")
        length = config["length"]
        if type(length) is not int or length < 1:
            raise ValueError(f"tokenizer config length must be a positive int, got {length!r}")
        if config["vocab_size"] != cls.VOCAB_SIZE:
            raise ValueError(
                f"tokenizer config vocab_size must be {cls.VOCAB_SIZE}, got {config['vocab_size']!r}"
            )
        if config["pad"] != cls.PAD:
            raise ValueError(f"tokenizer config pad must be {cls.PAD}, got {config['pad']!r}")
        if config["cls"] != cls.CLS:
            raise ValueError(f"tokenizer config cls must be {cls.CLS}, got {config['cls']!r}")
        return cls(length)
