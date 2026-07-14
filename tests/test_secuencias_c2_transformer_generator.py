# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""SECUENCIAS_PRODUCTO C2 — `Text` en el prompt → modelo transformer.

Contrato 52 §C2: `parse_field_specs` gana el kind `text` (`Text`, `Text[L]`);
`analyze_playground_request` enruta un prompt con UN campo Text al
`TransformerNetworkGenerator` (SEQUENCE + EMBEDDING + BLOCK TRANSFORMER +
POOL mean + cabeza), con `architecture_decision.kind == "transformer"` y
`source == "prompt_types"`. Mezcla Text+tabular o varios Text → error
accionable (decisión 3). Tabular puro y series temporales no cambian.
"""
from __future__ import annotations

import pytest

from matrixai.generation.prompt_field_specs import parse_field_specs
from matrixai.parser.parser import parse_text
from matrixai.playground import analyze_playground_request
from matrixai.training.transformer_generator import (
    TransformerNetworkGenerator,
    TransformerNetworkGeneratorError,
)
from matrixai.types import check_program_types


def _stage_warnings(result) -> list[str]:
    return [w for s in result.get("pipeline_stages", []) for w in (s.get("warnings") or [])]


# ---------------------------------------------------------------------------
# parse_field_specs: kind "text"
# ---------------------------------------------------------------------------

class TestFieldSpecsText:
    def test_bare_text_defaults_length_none(self):
        r = parse_field_specs("resenas: Text")
        assert r.fields == [
            f for f in r.fields if f.name == "resenas" and f.kind == "text" and f.length is None
        ]

    def test_text_with_bracket_length(self):
        r = parse_field_specs("resenas: Text[128]")
        f = r.by_name()["resenas"]
        assert f.kind == "text"
        assert f.length == 128

    def test_texto_spanish_alias(self):
        r = parse_field_specs("cuerpo: Texto[32]")
        f = r.by_name()["cuerpo"]
        assert f.kind == "text" and f.length == 32

    def test_case_insensitive(self):
        r = parse_field_specs("resenas: TEXT[16]")
        assert r.by_name()["resenas"].length == 16

    def test_invalid_length_falls_back_to_none_with_warning(self):
        r = parse_field_specs("resenas: Text[abc]")
        f = r.by_name()["resenas"]
        assert f.kind == "text" and f.length is None
        assert r.warnings

    def test_non_positive_length_falls_back_with_warning(self):
        r = parse_field_specs("resenas: Text[0]")
        f = r.by_name()["resenas"]
        assert f.length is None
        assert r.warnings

    def test_text_alongside_other_field_declarations_both_parsed(self):
        """parse_field_specs en sí no valida la mezcla — eso es responsabilidad
        del generador (decisión 3); el parser reporta ambos campos tal cual."""
        r = parse_field_specs("resenas: Text\nedad: Scalar")
        kinds = {f.name: f.kind for f in r.fields}
        assert kinds == {"resenas": "text", "edad": "scalar"}

    def test_duplicate_text_lengths_are_a_structured_conflict(self):
        r = parse_field_specs("resenas: Text[64]\nresenas: Text[128]")
        assert r.by_name()["resenas"].length == 64  # legacy first-wins view
        assert r.declares_kind("text") is True
        assert len(r.conflicts) == 1
        assert r.conflicts[0].first.length == 64
        assert r.conflicts[0].duplicate.length == 128
        assert any("length=64" in w and "length=128" in w for w in r.warnings)

    def test_identical_duplicate_text_declaration_is_harmless(self):
        r = parse_field_specs("resenas: Text[64]\nresenas: TEXT[64]")
        assert r.by_name()["resenas"].length == 64
        assert r.conflicts == []


# ---------------------------------------------------------------------------
# TransformerNetworkGenerator
# ---------------------------------------------------------------------------

class TestGeneratorMinimalPrompt:
    def _generate_and_typecheck(self, prompt: str):
        gen = TransformerNetworkGenerator().generate(prompt)
        prog = parse_text(gen.mxai_text)
        tr = check_program_types(prog)
        return gen, prog, tr

    def test_spanish_minimal_prompt(self):
        gen, prog, tr = self._generate_and_typecheck(
            "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert tr.ok, tr.errors
        assert len(prog.sequences) == 1
        assert prog.networks and prog.networks[0].transformer_blocks
        assert gen.field_name == "resenas"

    def test_english_minimal_prompt(self):
        gen, prog, tr = self._generate_and_typecheck(
            "reviews: Text\nOUTPUT class: ProbabilityMap[NEG, POS]"
        )
        assert tr.ok, tr.errors
        assert prog.networks[0].transformer_blocks
        assert gen.field_name == "reviews"

    def test_length_from_prompt_respected(self):
        gen, prog, tr = self._generate_and_typecheck("resenas: Text[128]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert tr.ok, tr.errors
        assert gen.length == 128
        assert prog.sequences[0].length == 128

    def test_defaults_match_decision_6(self):
        gen, _, tr = self._generate_and_typecheck("resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert tr.ok
        assert (gen.length, gen.dim, gen.layers, gen.heads) == (64, 64, 2, 4)

    def test_vocab_size_is_byte_tokenizer_vocab(self):
        gen, prog, _ = self._generate_and_typecheck("resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert prog.sequences[0].vocab_size == 259

    def test_pool_mean_and_single_output_layer(self):
        gen, _, _ = self._generate_and_typecheck("resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert "POOL mean" in gen.mxai_text

    def test_regression_task(self):
        gen, prog, tr = self._generate_and_typecheck("resenas: Text\nOUTPUT puntuacion: Scalar")
        assert tr.ok, tr.errors
        assert gen.output_activation == "linear"


class TestGeneratorErrors:
    def test_mixing_text_and_tabular_raises(self):
        with pytest.raises(TransformerNetworkGeneratorError, match="Mezclar Text"):
            TransformerNetworkGenerator().generate(
                "resenas: Text\nedad: Scalar\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_multiple_text_fields_raises(self):
        with pytest.raises(TransformerNetworkGeneratorError, match="un campo Text"):
            TransformerNetworkGenerator().generate(
                "titulo: Text\ncuerpo: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_no_text_field_raises(self):
        with pytest.raises(TransformerNetworkGeneratorError, match="Text field"):
            TransformerNetworkGenerator().generate("edad: Scalar\nOUTPUT clase: ProbabilityMap[NEG, POS]")

    def test_empty_prompt_raises(self):
        with pytest.raises(TransformerNetworkGeneratorError):
            TransformerNetworkGenerator().generate("   ")

    def test_mixing_with_bare_untyped_tabular_field_raises(self):
        """Auditoría C2 [MEDIA]: campos tabulares SIN tipo declarado ("campos:
        edad, ingreso") no pasaban por parse_field_specs y se ignoraban en
        silencio — deben detectarse igual que un campo tipado."""
        with pytest.raises(TransformerNetworkGeneratorError, match="Mezclar Text"):
            TransformerNetworkGenerator().generate(
                "resenas: Text\ncampos: edad, ingreso\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_mixing_with_one_bare_untyped_tabular_field_raises(self):
        """Una lista legacy de cardinalidad uno también es una mezcla; el
        extractor denso conserva mínimo dos por defecto, pero Text la valida
        con mínimo uno porque ya existe una declaración explícita inequívoca."""
        with pytest.raises(TransformerNetworkGeneratorError, match="Mezclar Text"):
            TransformerNetworkGenerator().generate(
                "resenas: Text\ncampos: edad\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_duplicate_conflicting_declaration_text_first_raises(self):
        """Auditoría C2 [MEDIA]: 'resenas: Text' seguido de 'resenas: Scalar'
        generaba el transformer en silencio, ignorando la contradicción."""
        with pytest.raises(TransformerNetworkGeneratorError, match="contradictorias"):
            TransformerNetworkGenerator().generate(
                "resenas: Text\nresenas: Scalar\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_duplicate_conflicting_declaration_scalar_first_raises_informatively(self):
        """Mismo conflicto en el orden inverso: antes daba el mensaje genérico
        "no Text field" sin explicar la contradicción; ahora la menciona."""
        with pytest.raises(TransformerNetworkGeneratorError, match="contradice"):
            TransformerNetworkGenerator().generate(
                "resenas: Scalar\nresenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_duplicate_text_with_different_lengths_raises(self):
        with pytest.raises(TransformerNetworkGeneratorError, match="contradictorias"):
            TransformerNetworkGenerator().generate(
                "resenas: Text[64]\nresenas: Text[128]\n"
                "OUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_length_exceeding_limit_raises(self):
        """Auditoría C2 [ALTA]: Text[L] sin tope permitía agotar memoria (L
        arbitrario materializa columnas t0..t{L-1} varias veces: mxai,
        training_text, dataset_template_text) y la atención escala O(L²)."""
        with pytest.raises(TransformerNetworkGeneratorError, match="límite de longitud"):
            TransformerNetworkGenerator().generate(
                "resenas: Text[1000000000]\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_length_within_default_profile_limit_ok(self):
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text[400]\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert gen.length == 400

    def test_length_limit_configurable_via_profile(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "avanzado")
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text[4000]\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert gen.length == 4000

    def test_length_limit_hard_in_hosted(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_HOSTED", "1")
        monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "avanzado")  # ignorado en hosted
        with pytest.raises(TransformerNetworkGeneratorError, match="límite de longitud"):
            TransformerNetworkGenerator().generate(
                "resenas: Text[4000]\nOUTPUT clase: ProbabilityMap[NEG, POS]"
            )

    def test_self_referencing_variables_list_does_not_false_positive(self):
        """"variables: resenas" (repite el propio nombre del campo Text) no
        debe interpretarse como un campo tabular adicional."""
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text\nvariables: resenas\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert gen.field_name == "resenas"


class TestGeneratorM12M17:
    def test_depth_from_prompt_sets_layers(self):
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text\n5 capas\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert gen.layers == 5

    def test_width_from_prompt_sets_dim(self):
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text\n128 unidades\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert gen.dim == 128

    def test_heads_stays_a_divisor_of_dim_when_width_not_divisible(self):
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text\n50 unidades\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        )
        assert gen.dim == 50
        assert gen.dim % gen.heads == 0
        prog = parse_text(gen.mxai_text)
        tr = check_program_types(prog)
        assert tr.ok, tr.errors


class TestGeneratorFieldMetadata:
    def test_field_types_marks_text(self):
        gen = TransformerNetworkGenerator().generate("resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert gen.field_types == {"resenas": "text"}

    def test_field_seq_matches_contract_shape(self):
        gen = TransformerNetworkGenerator().generate("resenas: Text[32]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert gen.field_seq == {"resenas": {"length": 32, "tokenizer": "byte_v1"}}

    def test_is_transformer_flag(self):
        gen = TransformerNetworkGenerator().generate("resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert gen.is_transformer is True


class TestGeneratorInvariantOne:
    """GEN invariante 1 (reusado por el contrato 52): el prompt explícito gana
    — el LLM/caller no puede des-declarar el campo Text."""

    def test_input_fields_conflicting_is_ignored_with_warning(self):
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
            input_fields=["otra_cosa"],
        )
        assert gen.field_name == "resenas"
        assert any("invariante 1" in w for w in gen.warnings)

    def test_input_fields_matching_the_text_field_is_silent(self):
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
            input_fields=["resenas"],
        )
        assert not any("invariante 1" in w for w in gen.warnings)


class TestGeneratorTrainingArtifacts:
    def test_training_text_parses_and_verifies_with_matching_csv(self, tmp_path):
        from matrixai.training import parse_training_text
        from matrixai.training.verifier import TrainingVerifier

        gen = TransformerNetworkGenerator().generate("resenas: Text[8]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        (tmp_path / f"{gen.network_name}Project.mxai").write_text(gen.mxai_text)
        (tmp_path / f"{gen.network_name.lower()}.train.csv").write_text(gen.dataset_template_text)
        training = parse_training_text(gen.training_text)
        report = TrainingVerifier().verify(training, base_path=tmp_path)
        assert report.ok, report.errors

    def test_training_text_uses_adam(self):
        gen = TransformerNetworkGenerator().generate("resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert "TYPE adam" in gen.training_text

    def test_dataset_template_has_one_raw_text_column(self):
        """SECUENCIAS_PRODUCTO C3: UNA columna de texto crudo (nombrada como
        el propio campo Text) + el target — nunca t0..t{L-1} (`length` solo
        afecta la tokenización en train, no la forma del CSV)."""
        gen = TransformerNetworkGenerator().generate("resenas: Text[8]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        header = gen.dataset_template_text.splitlines()[0].split(",")
        assert header == ["resenas", "predicted_class"]

    def test_training_text_has_one_raw_text_column(self):
        gen = TransformerNetworkGenerator().generate("resenas: Text[8]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        assert "FROM COLUMNS [resenas]" in gen.training_text
        assert "t0" not in gen.training_text


# ---------------------------------------------------------------------------
# analyze_playground_request — dispatch, metadata, regression
# ---------------------------------------------------------------------------

class TestPlaygroundDispatch:
    def test_text_prompt_routes_to_transformer_generator(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar reseñas de producto\nresenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        assert r["architecture_decision"] == {
            "kind": "transformer", "source": "prompt_types", "rationale": "",
        }
        assert r["supervision_source"] == "transformer_generator"
        assert r["field_types"] == {"resenas": "text"}
        assert r["field_seq"] == {"resenas": {"length": 64, "tokenizer": "byte_v1"}}

    def test_exact_contract_minimal_prompt_routes_to_transformer(self):
        """Auditoría C2 [ALTA]: el prompt mínimo LITERAL del contrato (§C2) no
        lleva ningún verbo de intención neural ("clasificar"/"predecir") —
        _is_neural_prompt() a secas lo enviaba a PromptSupervisor. El campo
        Text debe bastar por sí solo para entrar al generador."""
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        assert r["supervision_source"] == "transformer_generator"
        assert r["architecture_decision"]["kind"] == "transformer"
        assert r["field_types"] == {"resenas": "text"}
        assert "BLOCK enc TRANSFORMER" in r.get("mxai", "")
        assert "ACTION" not in r.get("mxai", "")  # no PromptSupervisor fallback

    def test_ok_true_for_a_valid_transformer_prompt(self):
        """Auditoría C2 [MEDIA]: backend_contract.ok=False por diseño para
        BLOCK TRANSFORMER (solo el runner interactivo de un ejemplo no lo
        soporta — training/export sí) no debe declarar rechazado un modelo
        válido, typechequeado y entrenable recién generado."""
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        assert r["ok"] is True
        assert r["accepted"] is True
        checks = {c["name"]: c["ok"] for c in r["checks"]}
        assert checks["backend_contract"] is True
        stage = next(s for s in r["pipeline_stages"] if s["name"] == "backend_contract")
        assert stage["status"] == "warning"  # visible, no oculto — pero no "fail"
        assert stage["errors"] == []
        assert any("BLOCK TRANSFORMER" in w for w in stage["warnings"])

    @pytest.mark.parametrize("prefix", ["", "Clasificar reseñas\n"])
    def test_scalar_first_text_conflict_never_degrades_to_other_router(self, prefix):
        """La declaración Text duplicada debe poseer el routing aunque no sea
        la first-wins efectiva; el generador reporta la contradicción tanto sin
        verbo neuronal como con uno, nunca PromptSupervisor/dense."""
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": prefix + "resenas: Scalar\nresenas: Text\n"
                      "OUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        assert r["ok"] is False
        assert r["accepted"] is False
        assert "contradice" in (r.get("error") or "")
        assert not r.get("mxai")

    def test_backend_parameter_errors_remain_blocking_in_mxai_mode(self):
        """El soft-pass solo aplica a capacidades separadas del transformer;
        un contrato PARAM incoherente nunca puede producir ok=True con errores."""
        bad_mxai = """PROJECT BadExplicitParams

PARAM W1 Tensor[2, 2]
  TRAINABLE true
  INIT zeros
END

VECTOR Email[2]
  urgency: Score
  sender_trust: Score
END

FUNCTION Classifier
  C: ProbabilityMap = softmax(W1 * Email + b1)
END

GRAPH
  Email -> Classifier
END

AUDIT
  EXPLAIN Email -> Classifier
END
"""
        r = analyze_playground_request({"mode": "mxai", "mxai_text": bad_mxai})
        check = next(c for c in r["checks"] if c["name"] == "backend_contract")
        assert check["ok"] is False
        assert check["errors"]
        assert r["ok"] is False
        assert r["accepted"] is False

    def test_text_prompt_parses_and_typechecks(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar reseñas de producto\nresenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        checks = {c["name"]: c["ok"] for c in r["checks"]}
        assert checks["parser"] is True
        assert checks["verifier_agent"] is True
        assert checks["typecheck"] is True

    def test_mixing_error_surfaces_accountable_message_not_fallback(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar reseñas de producto\nresenas: Text\nedad: Scalar\n"
                      "OUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        assert r["ok"] is False
        assert "Mezclar Text" in (r.get("error") or "")
        # No cae al fallback de PromptSupervisor (que generaría un workflow ACTION)
        assert "ACTION" not in r.get("mxai", "")

    def test_transformer_warning_suppressed_when_text_used(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar reseñas de producto con un transformer\nresenas: Text\n"
                      "OUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        warnings = _stage_warnings(r) + [str(r.get("error") or "")]
        assert not any("transformer" in w and "no soportad" in w for w in warnings)

    def test_transformer_generator_stage_labeled_correctly(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar reseñas de producto\nresenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        names = {s.get("name"): s.get("label") for s in r.get("pipeline_stages", [])}
        assert names.get("transformer_generator") == "Transformer Network Generator"


class TestRegressionTabularAndSequence:
    """Decisión 4 + retro-compat (invariante 5): prompts sin Text no cambian."""

    def test_plain_tabular_prompt_unaffected(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Predecir el precio de una casa segun superficie y habitaciones",
        })
        assert r["ok"] is True
        assert r["architecture_decision"]["kind"] == "dense"
        assert r.get("field_seq") == {}

    def test_composite_categorical_prompt_unaffected(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar clientes por segmento y riesgo\n"
                      "segmento: Categorical[A,B,C,D,E,F,G,H,I,J,K,L,M,N]\n"
                      "OUTPUT clase: ProbabilityMap[BAJO, ALTO]",
        })
        assert r["ok"] is True
        assert r["architecture_decision"]["kind"] == "composite"

    def test_time_series_warning_unaffected(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Predecir una serie temporal de ventas mensuales",
        })
        warnings = _stage_warnings(r)
        assert any("Secuencias/series temporales" in w for w in warnings)
        assert r["architecture_decision"]["kind"] != "transformer"

    def test_sequence_wording_with_text_field_does_not_warn(self):
        """Auditoría C2 [BAJA]: un prompt Text que ADEMÁS mencione "secuencia"/
        "serie temporal" en prosa SÍ genera el transformer (decisión 4) — el
        aviso de "no soportado, se genera tabular" no debe aparecer, sería
        contradictorio con el propio resultado."""
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "analizar la secuencia de reseñas\nresenas: Text\n"
                      "OUTPUT clase: ProbabilityMap[NEG, POS]",
        })
        assert r["architecture_decision"]["kind"] == "transformer"
        warnings = _stage_warnings(r)
        assert not any("Secuencias/series temporales" in w for w in warnings)
