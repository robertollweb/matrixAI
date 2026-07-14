# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""SECUENCIAS_PRODUCTO C3 — dataset sintético de texto + upload.

Contrato 52 §C3: `_generate_synthetic_dataset` para un modelo Text produce un
CSV con la COLUMNA DE TEXTO CRUDO (nunca ids — invariante 1), con 3 orígenes
de etiqueta (decisión 5): `synthetic_random` (sin señal, declarado como tal),
`synthetic_template` (señal determinista, "el suelo sin LLM") y
`synthetic_llm_examples` (LLM redacta N ejemplos por clase, con fallback a
template si falla/no valida — nunca a random). El boundary de tokenización
vive en `CSVTextDataAdapter`/`_resolve_transformer_dataset`: el CSV guarda
texto, el trainer ve ids. Upload valida columna de texto (vacíos, codificación,
truncado) y ecoa `field_seq` (invariante 2/3 de GEN).
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from matrixai.text.tokenizer import ByteTokenizer
from matrixai.training.data import CSVDataAdapter, CSVTextDataAdapter
from matrixai.training.parser import parse_training_text
from matrixai.training.synthetic_text import (
    generate_random_examples,
    generate_template_examples,
    generate_text_examples,
    llm_text_examples,
    parse_llm_text_examples,
    validate_llm_text_examples,
    _signal_token,
)
from matrixai.training.transformer_generator import TransformerNetworkGenerator
from matrixai.training.transformer_trainer import _resolve_transformer_dataset
from matrixai.training.verifier import TrainingVerifier


_PROMPT = "resenas: Text[16]\nOUTPUT clase: ProbabilityMap[NEG, POS]"
_LABELS = ["NEG", "POS"]


def _gen():
    return TransformerNetworkGenerator().generate(_PROMPT)


class _FakeProvider:
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, system, user):  # noqa: ARG002
        return self._text


_LLM_GOOD_TEXT = (
    "NEG: que mal producto, se rompio enseguida\n"
    "NEG: pesimo servicio, no lo recomiendo\n"
    "POS: excelente calidad, muy contento\n"
    "POS: llego rapido y funciona genial\n"
)


# ---------------------------------------------------------------------------
# synthetic_text.py — los 3 orígenes de etiqueta
# ---------------------------------------------------------------------------

class TestGenerateRandomExamples:
    def test_produces_n_examples(self):
        rows = generate_random_examples(10, _LABELS, seed=1)
        assert len(rows) == 10
        assert all(isinstance(t, str) and t for t, _ in rows)
        assert all(l in _LABELS for _, l in rows)

    def test_deterministic_for_same_seed(self):
        assert generate_random_examples(20, _LABELS, seed=7) == generate_random_examples(20, _LABELS, seed=7)

    def test_different_seed_differs(self):
        a = generate_random_examples(30, _LABELS, seed=1)
        b = generate_random_examples(30, _LABELS, seed=2)
        assert a != b


class TestSignalToken:
    def test_stable_per_class(self):
        assert _signal_token("POS") == _signal_token("POS")

    def test_distinct_between_classes(self):
        assert _signal_token("POS") != _signal_token("NEG")

    def test_not_hash_based(self):
        """No debe depender de hash() — no determinista entre procesos por
        PYTHONHASHSEED. Verificamos indirectamente: el token es puramente
        función del identificador normalizado (mismo resultado en este
        proceso sin ninguna variable de entorno de por medio) y no contiene
        dígitos grandes típicos de un hash."""
        token = _signal_token("Producto Estrella")
        assert token == "senal_producto_estrella"


class TestGenerateTemplateExamples:
    def test_round_robin_class_coverage(self):
        rows = generate_template_examples(9, ["A", "B", "C"], seed=1)
        counts = {l: 0 for l in ["A", "B", "C"]}
        for _, label in rows:
            counts[label] += 1
        assert counts == {"A": 3, "B": 3, "C": 3}

    def test_each_row_contains_its_class_signal_token(self):
        rows = generate_template_examples(6, _LABELS, seed=3)
        for text, label in rows:
            assert _signal_token(label) in text

    def test_deterministic_for_same_seed(self):
        assert generate_template_examples(12, _LABELS, seed=5) == generate_template_examples(12, _LABELS, seed=5)


class TestParseLlmTextExamples:
    def test_parses_matching_lines(self):
        by_label = parse_llm_text_examples(_LLM_GOOD_TEXT, _LABELS)
        assert len(by_label["NEG"]) == 2
        assert len(by_label["POS"]) == 2

    def test_case_and_whitespace_insensitive_label_match(self):
        by_label = parse_llm_text_examples("  pos :  genial esto\n", _LABELS)
        assert by_label["POS"] == ["genial esto"]

    def test_lines_with_unknown_label_are_dropped(self):
        by_label = parse_llm_text_examples("NEUTRAL: no es una clase declarada\n", _LABELS)
        assert by_label == {"NEG": [], "POS": []}

    def test_non_matching_lines_ignored(self):
        raw = "Aqui tienes los ejemplos:\n1. algo\nNEG: mal producto\n"
        by_label = parse_llm_text_examples(raw, _LABELS)
        assert by_label["NEG"] == ["mal producto"]

    def test_duplicate_text_not_repeated(self):
        raw = "NEG: mal producto\nNEG: mal producto\n"
        by_label = parse_llm_text_examples(raw, _LABELS)
        assert by_label["NEG"] == ["mal producto"]


class TestValidateLlmTextExamples:
    def test_sufficient_examples_pass(self):
        by_label = {"NEG": ["a", "b"], "POS": ["c", "d"]}
        assert validate_llm_text_examples(by_label, _LABELS) == []

    def test_insufficient_examples_reported(self):
        by_label = {"NEG": ["a"], "POS": ["c", "d"]}
        problems = validate_llm_text_examples(by_label, _LABELS)
        assert problems
        assert "NEG" in problems[0]

    def test_missing_class_entirely_reported(self):
        by_label = {"NEG": [], "POS": ["c", "d"]}
        problems = validate_llm_text_examples(by_label, _LABELS)
        assert "NEG" in problems[0]


class TestLlmTextExamples:
    def test_provider_success_parses(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            return_value=_FakeProvider(_LLM_GOOD_TEXT),
        ):
            by_label = llm_text_examples("clasificar reseñas", _LABELS, n_per_class=2)
        assert by_label is not None
        assert len(by_label["NEG"]) == 2
        assert len(by_label["POS"]) == 2

    def test_provider_error_returns_none(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            side_effect=RuntimeError("no key"),
        ):
            assert llm_text_examples("x", _LABELS, n_per_class=2) is None

    def test_garbage_response_parses_to_empty_pools(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            return_value=_FakeProvider("I cannot help with that."),
        ):
            by_label = llm_text_examples("x", _LABELS, n_per_class=2)
        assert by_label == {"NEG": [], "POS": []}


class TestGenerateTextExamplesDispatch:
    """Punto de entrada único de los 3 orígenes (decisión 5)."""

    def test_random_mode_returns_synthetic_random(self):
        rows, origin = generate_text_examples("random", "clasificar", 10, _LABELS, seed=1)
        assert origin == "synthetic_random"
        assert len(rows) == 10

    def test_coherent_without_llm_returns_synthetic_template(self):
        rows, origin = generate_text_examples("coherent", "clasificar", 10, _LABELS, seed=1, use_llm=False)
        assert origin == "synthetic_template"
        assert len(rows) == 10

    def test_coherent_with_llm_success_returns_synthetic_llm_examples(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            return_value=_FakeProvider(_LLM_GOOD_TEXT),
        ):
            rows, origin = generate_text_examples("coherent", "clasificar", 8, _LABELS, seed=1, use_llm=True)
        assert origin == "synthetic_llm_examples"
        assert len(rows) == 8

    def test_coherent_with_llm_provider_error_falls_back_to_template_never_random(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            side_effect=RuntimeError("no key"),
        ):
            rows, origin = generate_text_examples("coherent", "clasificar", 8, _LABELS, seed=1, use_llm=True)
        assert origin == "synthetic_template"
        for text, label in rows:
            assert _signal_token(label) in text

    def test_coherent_with_llm_insufficient_examples_falls_back_to_template(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            return_value=_FakeProvider("NEG: solo un ejemplo\n"),
        ):
            rows, origin = generate_text_examples("coherent", "clasificar", 8, _LABELS, seed=1, use_llm=True)
        assert origin == "synthetic_template"

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError):
            generate_text_examples("random", "x", 5, [], seed=1)


# ---------------------------------------------------------------------------
# Boundary de tokenización: el CSV guarda texto, el trainer ve ids
# ---------------------------------------------------------------------------

class TestCSVTextDataAdapter:
    def test_encodes_text_column_to_token_ids(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("resenas,clase\nhola,POS\nadios,NEG\n", encoding="utf-8")
        tokenizer = ByteTokenizer(8)
        adapter = CSVTextDataAdapter(csv_path, "Resenas", "resenas", "clase", tokenizer, _LABELS)
        examples = adapter.examples()
        assert len(examples) == 2
        expected = [float(b) for b in tokenizer.encode("hola")]
        assert examples[0].vector == expected
        assert all(v == float(int(v)) for v in examples[0].vector)

    def test_schema_reflects_single_text_column(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("resenas,clase\nhola,POS\n", encoding="utf-8")
        tokenizer = ByteTokenizer(8)
        adapter = CSVTextDataAdapter(csv_path, "Resenas", "resenas", "clase", tokenizer, _LABELS)
        schema = adapter.schema().to_dict()
        assert schema["input_columns"] == ["resenas"]
        assert schema["input_vector"] == "Resenas"

    def test_csv_data_adapter_still_encodes_floats_unaffected(self, tmp_path: Path):
        """Regresión: refactorizar `_encode_row` como hook no debe tocar el
        comportamiento numérico existente de CSVDataAdapter."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("t0,t1,clase\n1,2,POS\n3,4,NEG\n", encoding="utf-8")
        adapter = CSVDataAdapter(csv_path, "Seq", ["t0", "t1"], "clase", _LABELS)
        examples = adapter.examples()
        assert examples[0].vector == [1.0, 2.0]
        assert examples[1].vector == [3.0, 4.0]


class TestVerifierAcceptsRawTextSequence:
    def test_one_raw_text_column_ok(self, tmp_path: Path):
        gen = _gen()
        (tmp_path / f"{gen.network_name}Project.mxai").write_text(gen.mxai_text)
        (tmp_path / f"{gen.network_name.lower()}.train.csv").write_text(gen.dataset_template_text)
        training = parse_training_text(gen.training_text)
        report = TrainingVerifier().verify(training, base_path=tmp_path)
        assert report.ok, report.errors

    def test_legacy_pre_tokenized_columns_still_ok(self, tmp_path: Path):
        gen = _gen()
        legacy_training = gen.training_text.replace(
            "FROM COLUMNS [resenas]",
            "FROM COLUMNS [" + ", ".join(f"t{i}" for i in range(16)) + "]",
        )
        (tmp_path / f"{gen.network_name}Project.mxai").write_text(gen.mxai_text)
        header = ",".join([f"t{i}" for i in range(16)] + ["predicted_class"])
        row = ",".join(["0"] * 16 + ["neg"])
        (tmp_path / f"{gen.network_name.lower()}.train.csv").write_text(f"{header}\n{row}\n{row}\n")
        training = parse_training_text(legacy_training)
        report = TrainingVerifier().verify(training, base_path=tmp_path)
        assert report.ok, report.errors

    def test_wrong_column_count_rejected(self, tmp_path: Path):
        gen = _gen()
        bad_training = gen.training_text.replace(
            "FROM COLUMNS [resenas]", "FROM COLUMNS [resenas, extra]",
        )
        (tmp_path / f"{gen.network_name}Project.mxai").write_text(gen.mxai_text)
        (tmp_path / f"{gen.network_name.lower()}.train.csv").write_text(
            "resenas,extra,predicted_class\nhola,x,neg\n"
        )
        training = parse_training_text(bad_training)
        report = TrainingVerifier().verify(training, base_path=tmp_path)
        assert not report.ok
        assert any("raw-text column" in e and "pre-tokenized" in e for e in report.errors)


class TestResolveTransformerDatasetBoundary:
    def _write(self, tmp_path: Path, gen, training_text: str, csv_text: str) -> None:
        (tmp_path / f"{gen.network_name}Project.mxai").write_text(gen.mxai_text)
        (tmp_path / f"{gen.network_name.lower()}.train.csv").write_text(csv_text)
        (tmp_path / "training_text").write_text(training_text)  # not read by parser, just for debugging

    def test_raw_text_column_tokenized_at_load(self, tmp_path: Path):
        gen = _gen()
        self._write(
            tmp_path, gen, gen.training_text,
            "resenas,predicted_class\nme encanta este producto,pos\nque mal servicio,neg\n",
        )
        training = parse_training_text(gen.training_text)
        loaded = _resolve_transformer_dataset(training, tmp_path)
        seq_name = loaded["seq"].name
        first_vector, first_label = loaded["examples"][0]
        assert len(first_vector[seq_name]) == 16
        assert all(isinstance(v, int) and 0 <= v <= 258 for v in first_vector[seq_name])
        tokenizer = ByteTokenizer(16)
        assert first_vector[seq_name] == tokenizer.encode("me encanta este producto")

    def test_legacy_pretokenized_columns_still_load(self, tmp_path: Path):
        gen = _gen()
        legacy_training = gen.training_text.replace(
            "FROM COLUMNS [resenas]",
            "FROM COLUMNS [" + ", ".join(f"t{i}" for i in range(16)) + "]",
        )
        header = ",".join([f"t{i}" for i in range(16)] + ["predicted_class"])
        row = ",".join([str(i) for i in range(16)] + ["neg"])
        self._write(tmp_path, gen, legacy_training, f"{header}\n{row}\n{row}\n")
        training = parse_training_text(legacy_training)
        loaded = _resolve_transformer_dataset(training, tmp_path)
        seq_name = loaded["seq"].name
        first_vector, _ = loaded["examples"][0]
        assert first_vector[seq_name] == list(range(16))

    def test_wrong_column_count_raises(self, tmp_path: Path):
        gen = _gen()
        bad_training = gen.training_text.replace(
            "FROM COLUMNS [resenas]", "FROM COLUMNS [resenas, extra]",
        )
        self._write(tmp_path, gen, bad_training, "resenas,extra,predicted_class\nhola,x,neg\n")
        training = parse_training_text(bad_training)
        with pytest.raises(ValueError, match="raw-text column"):
            _resolve_transformer_dataset(training, tmp_path)


# ---------------------------------------------------------------------------
# playground.py — dataset sintético de texto (dispatch + metadata + límites)
# ---------------------------------------------------------------------------

class TestPlaygroundSyntheticTextDataset:
    def _generate(self, rows=20, seed=1, mode="random", use_llm=False, prompt=_PROMPT):
        from matrixai.playground import _generate_synthetic_dataset

        gen = TransformerNetworkGenerator().generate(prompt)
        return _generate_synthetic_dataset(gen.mxai_text, gen.training_text, rows, seed, mode, use_llm=use_llm)

    def test_random_mode_csv_has_one_text_column_and_signal_warning(self):
        r = self._generate(mode="random")
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_random"
        assert r["columns"] == ["resenas", "predicted_class"]
        assert "signal_warning" in r
        rows = list(csv.reader(io.StringIO(r["csv_text"])))
        assert rows[0] == ["resenas", "predicted_class"]
        assert len(rows) - 1 == r["rows"]

    def test_coherent_mode_no_llm_uses_template_origin(self):
        r = self._generate(mode="coherent", use_llm=False)
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_template"
        assert "signal_warning" not in r

    def test_coherent_mode_llm_success_uses_llm_origin(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            return_value=_FakeProvider(_LLM_GOOD_TEXT),
        ):
            r = self._generate(rows=8, mode="coherent", use_llm=True)
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_llm_examples"

    def test_coherent_mode_llm_failure_falls_back_to_template_not_random(self):
        with patch(
            "matrixai.agents.llm_proposal.ChatCompletionsLLMProposalProvider.from_env",
            side_effect=RuntimeError("no key"),
        ):
            r = self._generate(mode="coherent", use_llm=True)
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_template"

    def test_field_types_and_field_seq_echoed(self):
        r = self._generate(mode="random")
        assert r["field_types"] == {"resenas": "text"}
        assert r["field_seq"] == {"resenas": {"length": 16, "tokenizer": "byte_v1"}}

    def test_rows_capped_by_profile_produces_warning(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_ROWS", "5")
        r = self._generate(rows=50, mode="random")
        assert r["ok"], r.get("error")
        assert r["rows"] == 5
        assert "rows_capped_warning" in r
        assert "5" in r["rows_capped_warning"]

    def test_rows_within_profile_no_warning(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_ROWS", "1000")
        r = self._generate(rows=10, mode="random")
        assert r["ok"], r.get("error")
        assert "rows_capped_warning" not in r

    def test_multi_column_input_rejected(self):
        from matrixai.playground import _generate_synthetic_dataset

        gen = _gen()
        bad_training = gen.training_text.replace(
            "FROM COLUMNS [resenas]", "FROM COLUMNS [resenas, extra]",
        )
        r = _generate_synthetic_dataset(gen.mxai_text, bad_training, 10, 1, "random")
        assert r["ok"] is False
        assert "una columna" in r["error"]

    def test_sequence_not_found_rejected(self):
        from matrixai.playground import _generate_synthetic_dataset

        gen = _gen()
        bad_training = gen.training_text.replace("INPUT Resenas FROM COLUMNS", "INPUT Bogus FROM COLUMNS")
        r = _generate_synthetic_dataset(gen.mxai_text, bad_training, 10, 1, "random")
        assert r["ok"] is False
        assert "Bogus" in r["error"]

    def test_regression_target_rejected_no_labels(self):
        from matrixai.playground import _generate_synthetic_dataset

        gen = TransformerNetworkGenerator().generate("resenas: Text[16]\nOUTPUT puntuacion: Scalar")
        r = _generate_synthetic_dataset(gen.mxai_text, gen.training_text, 10, 1, "random")
        assert r["ok"] is False
        assert "clasificación" in r["error"]

    def test_tabular_prompt_regression_unaffected(self):
        """Un modelo sin SEQUENCE sigue la rama tabular existente, no la de texto."""
        from matrixai.playground import _generate_synthetic_dataset

        mxai = """PROJECT House

VECTOR Home[2]
  size: Score
  rooms: Score
END

FUNCTION Price
  value: Scalar = size + rooms
END

GRAPH
  Home -> Price
END

AUDIT
  EXPLAIN Home -> Price
END
"""
        training = """MODEL HouseProject.mxai

DATASET HouseTrainingSet
  SOURCE csv("house.train.csv")
  INPUT Home FROM COLUMNS [size, rooms]
  TARGET predicted_value: Scalar
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END

LOSS HouseLoss
  TYPE mse
  PREDICTION Price
  TARGET predicted_value
END

OPTIMIZER HouseOptimizer
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Price.*
END

RUN
  EPOCHS 10
END
"""
        r = _generate_synthetic_dataset(mxai, training, 10, 1, "random")
        assert r["ok"], r.get("error")
        assert "field_seq" not in r or not r.get("field_seq")


class TestCsvTemplateText:
    def test_text_model_template_uses_realistic_placeholder_not_numbers(self):
        from matrixai.playground import _csv_template

        gen = _gen()
        r = _csv_template(gen.mxai_text, gen.training_text)
        assert r["ok"], r.get("error")
        assert r["input_columns"] == ["resenas"]
        lines = r["template_csv"].splitlines()
        assert lines[0] == "resenas,predicted_class"
        assert "0.5" not in lines[1]
        assert lines[1].split(",")[0]


class TestValidateTextColumn:
    def test_valid_rows_no_errors_no_warnings(self):
        from matrixai.playground import _validate_text_column

        csv_text = "resenas,clase\nme encanta,pos\nque mal,neg\n"
        errors, warnings = _validate_text_column(csv_text, "resenas", seq_length=64)
        assert errors == []
        assert warnings == []

    def test_empty_row_is_error(self):
        from matrixai.playground import _validate_text_column

        csv_text = "resenas,clase\n,pos\nmal,neg\n"
        errors, warnings = _validate_text_column(csv_text, "resenas", seq_length=64)
        assert errors
        assert "2" in errors[0]

    def test_replacement_char_is_warning(self):
        from matrixai.playground import _validate_text_column

        csv_text = "resenas,clase\ntexto�roto,pos\n"
        errors, warnings = _validate_text_column(csv_text, "resenas", seq_length=64)
        assert errors == []
        assert warnings and "reemplazo" in warnings[0]

    def test_row_exceeding_length_is_warning_not_error(self):
        from matrixai.playground import _validate_text_column

        long_text = "x" * 100
        csv_text = f"resenas,clase\n{long_text},pos\n"
        errors, warnings = _validate_text_column(csv_text, "resenas", seq_length=16)
        assert errors == []
        assert warnings and "16" in warnings[0]


class TestValidateTrainingCsvTextIntegration:
    def test_valid_csv_ok_and_echoes_field_seq(self):
        from matrixai.playground import _validate_training_csv

        gen = _gen()
        csv_text = "resenas,predicted_class\nme encanta este producto,pos\nque mal servicio,neg\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_text)
        assert r["ok"], r
        assert r["field_seq"] == {"resenas": {"length": 16, "tokenizer": "byte_v1"}}
        assert r["rows"] == 2

    def test_missing_text_column_rejected(self):
        from matrixai.playground import _validate_training_csv

        gen = _gen()
        csv_text = "otra_columna,predicted_class\nhola,pos\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_text)
        assert r["ok"] is False
        assert "resenas" in r["missing_columns"]

    def test_empty_text_row_rejected(self):
        from matrixai.playground import _validate_training_csv

        gen = _gen()
        csv_text = "resenas,predicted_class\n,pos\nmal servicio,neg\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_text)
        assert r["ok"] is False
        assert any("vacías" in e for e in r["errors"])

    def test_row_exceeding_length_warns_but_passes(self):
        from matrixai.playground import _validate_training_csv

        gen = _gen()
        long_text = "x" * 100
        csv_text = f"resenas,predicted_class\n{long_text},pos\nbreve,neg\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_text)
        assert r["ok"], r
        assert any("16" in w for w in r["warnings"])

    def test_row_count_exceeding_profile_rejected(self, monkeypatch):
        from matrixai.playground import _validate_training_csv

        monkeypatch.setenv("MATRIXAI_MAX_ROWS", "1")
        gen = _gen()
        csv_text = "resenas,predicted_class\nuno,pos\ndos,neg\n"
        r = _validate_training_csv(gen.mxai_text, gen.training_text, csv_text)
        assert r["ok"] is False
        assert "2" in r["error"]

    def test_field_ranges_argument_is_noop_for_text_model(self):
        """Un caller que pase field_ranges por error no debe corromper la
        columna de texto (invariante: no hay rangos de dominio para Text)."""
        from matrixai.playground import _validate_training_csv

        gen = _gen()
        csv_text = "resenas,predicted_class\nme encanta este producto,pos\nque mal servicio,neg\n"
        r = _validate_training_csv(
            gen.mxai_text, gen.training_text, csv_text, field_ranges={"resenas": (0.0, 1.0)},
        )
        assert r["ok"], r
        assert r["rows"] == 2
