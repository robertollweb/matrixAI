# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""SECUENCIAS_PRODUCTO C1 — tokenizador byte-level (`byte_v1`).

Contrato 52 §C1: UTF-8 → bytes deterministas, truncado/padding a `length`
fijo, CLS opcional, config serializable estable.
"""
from __future__ import annotations

import pytest

from matrixai.text.tokenizer import ByteTokenizer


class TestEncodeBasics:
    def test_ascii_roundtrip(self):
        tok = ByteTokenizer(length=16)
        ids = tok.encode("hello")
        assert len(ids) == 16
        assert ids[:5] == list(b"hello")
        assert tok.decode(ids) == "hello"

    def test_accented_multibyte_roundtrip(self):
        tok = ByteTokenizer(length=32)
        text = "mañana está lloviendo"
        ids = tok.encode(text)
        assert len(ids) == 32
        assert tok.decode(ids) == text

    def test_emoji_roundtrip(self):
        tok = ByteTokenizer(length=16)
        text = "genial 🎉🔥"
        ids = tok.encode(text)
        assert len(ids) == 16
        assert tok.decode(ids) == text

    def test_empty_string(self):
        tok = ByteTokenizer(length=8)
        ids = tok.encode("")
        assert ids == [ByteTokenizer.PAD] * 8
        assert tok.decode(ids) == ""

    def test_padding_fills_with_pad_token(self):
        tok = ByteTokenizer(length=10)
        ids = tok.encode("ab")
        assert ids == [ord("a"), ord("b")] + [ByteTokenizer.PAD] * 8

    def test_truncation_exact_length(self):
        tok = ByteTokenizer(length=5)
        ids = tok.encode("abcde")
        assert ids == list(b"abcde")

    def test_truncation_longer_text(self):
        tok = ByteTokenizer(length=5)
        ids = tok.encode("abcdefgh")
        assert ids == list(b"abcde")
        assert len(ids) == 5

    def test_truncation_mid_multibyte_char_decodes_without_crashing(self):
        """Un carácter multibyte cortado a la mitad por el truncado no debe
        reventar decode (best-effort, errors="replace")."""
        tok = ByteTokenizer(length=3)  # "mañana" -> b'm\xc3\xb1...' — corta el 'ñ' a medias
        ids = tok.encode("mañana")
        assert len(ids) == 3
        decoded = tok.decode(ids)  # no debe lanzar
        assert isinstance(decoded, str)

    def test_determinism(self):
        tok = ByteTokenizer(length=16)
        text = "determinista, siempre igual"
        assert tok.encode(text) == tok.encode(text)
        assert ByteTokenizer(length=16).encode(text) == tok.encode(text)


class TestClsToken:
    def test_add_cls_prepends_token(self):
        tok = ByteTokenizer(length=8)
        ids = tok.encode("abc", add_cls=True)
        assert ids[0] == ByteTokenizer.CLS
        assert ids[1:4] == list(b"abc")
        assert len(ids) == 8

    def test_add_cls_reserves_one_slot_for_truncation(self):
        tok = ByteTokenizer(length=4)
        ids = tok.encode("abcdef", add_cls=True)
        assert ids == [ByteTokenizer.CLS] + list(b"abc")
        assert len(ids) == 4

    def test_default_no_cls(self):
        tok = ByteTokenizer(length=4)
        ids = tok.encode("ab")
        assert ByteTokenizer.CLS not in ids

    def test_decode_strips_cls_and_pad(self):
        tok = ByteTokenizer(length=8)
        ids = tok.encode("hi", add_cls=True)
        assert tok.decode(ids) == "hi"


class TestConfig:
    def test_config_shape_is_stable(self):
        tok = ByteTokenizer(length=64)
        assert tok.config() == {
            "kind": "byte_v1",
            "length": 64,
            "vocab_size": 259,
            "pad": 256,
            "cls": 258,
        }

    def test_vocab_size_is_256_plus_specials(self):
        assert ByteTokenizer.VOCAB_SIZE == ByteTokenizer.BASE_VOCAB + 3
        assert ByteTokenizer.PAD == 256
        assert ByteTokenizer.UNK == 257
        assert ByteTokenizer.CLS == 258

    def test_unk_never_emitted_by_encode(self):
        """UNK está reservado para vocabularios futuros — todo byte 0-255 es
        válido por construcción de UTF-8, así que encode() nunca lo emite."""
        tok = ByteTokenizer(length=32)
        samples = ["ascii", "ñáéíóú", "🎉🔥💯", "\x00\x01\x02", "混合 text"]
        for text in samples:
            assert ByteTokenizer.UNK not in tok.encode(text)

    def test_from_config_roundtrip(self):
        tok = ByteTokenizer(length=48)
        rebuilt = ByteTokenizer.from_config(tok.config())
        assert rebuilt.length == tok.length
        assert rebuilt.encode("same config, same ids") == tok.encode("same config, same ids")

    def test_from_config_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="byte_v1"):
            ByteTokenizer.from_config({
                "kind": "bpe_v1", "length": 32,
                "vocab_size": 259, "pad": 256, "cls": 258,
            })

    # -- auditoría C1 [MEDIA]: from_config estricto, sin coerción -----------

    def test_from_config_rejects_missing_keys(self):
        for missing in ByteTokenizer._REQUIRED_CONFIG_KEYS:
            full = {"kind": "byte_v1", "length": 16, "vocab_size": 259, "pad": 256, "cls": 258}
            del full[missing]
            with pytest.raises(ValueError, match="missing required keys"):
                ByteTokenizer.from_config(full)

    def test_from_config_rejects_non_dict(self):
        with pytest.raises(ValueError, match="object"):
            ByteTokenizer.from_config(["kind", "byte_v1"])  # type: ignore[arg-type]

    def test_from_config_rejects_contradictory_values(self):
        """Repro del auditor: vocab_size/pad/cls que no casan con byte_v1
        se aceptaban y se normalizaban en silencio a la config canónica."""
        with pytest.raises(ValueError, match="vocab_size must be 259"):
            ByteTokenizer.from_config({
                "kind": "byte_v1", "length": 4,
                "vocab_size": 999, "pad": 0, "cls": 1,
            })

    def test_from_config_rejects_wrong_pad(self):
        with pytest.raises(ValueError, match="pad must be 256"):
            ByteTokenizer.from_config({
                "kind": "byte_v1", "length": 4,
                "vocab_size": 259, "pad": 0, "cls": 258,
            })

    def test_from_config_rejects_wrong_cls(self):
        with pytest.raises(ValueError, match="cls must be 258"):
            ByteTokenizer.from_config({
                "kind": "byte_v1", "length": 4,
                "vocab_size": 259, "pad": 256, "cls": 1,
            })

    @pytest.mark.parametrize("bad_length", [3.9, "4", None, [4], 4.0])
    def test_from_config_rejects_non_int_length_without_coercion(self, bad_length):
        """Repro del auditor: int(config["length"]) normalizaba 3.9->3,
        "4"->4 en silencio — ahora el tipo debe ser EXACTAMENTE int."""
        with pytest.raises(ValueError, match="length must be a positive int"):
            ByteTokenizer.from_config({
                "kind": "byte_v1", "length": bad_length,
                "vocab_size": 259, "pad": 256, "cls": 258,
            })

    def test_from_config_rejects_bool_length(self):
        """bool hereda de int — True/False no son longitudes válidas."""
        with pytest.raises(ValueError, match="length must be a positive int"):
            ByteTokenizer.from_config({
                "kind": "byte_v1", "length": True,
                "vocab_size": 259, "pad": 256, "cls": 258,
            })

    def test_from_config_rejects_non_positive_length(self):
        with pytest.raises(ValueError, match="length must be a positive int"):
            ByteTokenizer.from_config({
                "kind": "byte_v1", "length": 0,
                "vocab_size": 259, "pad": 256, "cls": 258,
            })


class TestValidation:
    @pytest.mark.parametrize("bad_length", [0, -1, -100])
    def test_rejects_non_positive_length(self, bad_length):
        with pytest.raises(ValueError, match="positive"):
            ByteTokenizer(length=bad_length)

    def test_rejects_non_integer_length(self):
        with pytest.raises(ValueError, match="positive"):
            ByteTokenizer(length=3.5)  # type: ignore[arg-type]

    def test_decode_ignores_out_of_range_ids(self):
        tok = ByteTokenizer(length=8)
        assert tok.decode([ord("a"), 999, -1, ByteTokenizer.PAD]) == "a"

    # -- auditoría C1 [BAJA] --------------------------------------------

    def test_rejects_bool_length(self):
        """bool hereda de int — ByteTokenizer(True) no debe colarse como
        longitud 1."""
        with pytest.raises(ValueError, match="positive integer"):
            ByteTokenizer(length=True)  # type: ignore[arg-type]

    def test_encode_rejects_non_str_input(self):
        tok = ByteTokenizer(length=8)
        for bad in (b"ab", None, 5, ["a", "b"]):
            with pytest.raises(TypeError, match="expects str"):
                tok.encode(bad)  # type: ignore[arg-type]

    def test_encode_rejects_non_bool_add_cls(self):
        """Una cadena truthy como add_cls="false" no debe colar un CLS que
        el caller no pidió explícitamente."""
        tok = ByteTokenizer(length=8)
        with pytest.raises(TypeError, match="add_cls must be bool"):
            tok.encode("ab", add_cls="false")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="add_cls must be bool"):
            tok.encode("ab", add_cls=1)  # type: ignore[arg-type]

    def test_encode_does_not_materialize_full_input_before_truncating(self):
        """auditoría C1 [BAJA]: encode debe truncar el bytes ANTES de
        convertir a list — un texto enorme con length pequeño no debe
        construir una lista proporcional al texto completo."""
        tok = ByteTokenizer(length=8)
        huge = "x" * 10_000_000
        ids = tok.encode(huge)  # debe ser rápido y devolver solo `length` ids
        assert len(ids) == 8
        assert ids == [ord("x")] * 8
