"""GENERACIÓN tipos del prompt — Corte 1: parser compartido de FieldSpec.

Cubre la "Gramática del prompt" del contrato GENERACION_TIPOS_PROMPT_CONTRACT.md:
ES/EN, acentos, espacios, duplicados, rango invertido/no numérico, Scalar[0,1] ==
Scalar en [0,1], separadores, prosa ruidosa, retrocompat, y que la salida
(ProbabilityMap) NO se confunde con un campo.
"""
from __future__ import annotations

import unittest

from matrixai.generation import parse_field_specs, FieldSpec


def _by_name(prompt: str):
    return parse_field_specs(prompt).by_name()


class ScalarRangeGrammarTest(unittest.TestCase):
    def test_scalar_en_bracket_es(self):
        f = _by_name("edad: Scalar en [18, 95]")["edad"]
        self.assertEqual((f.kind, f.range), ("scalar", (18.0, 95.0)))

    def test_scalar_in_bracket_en(self):
        f = _by_name("age: Scalar in [0, 1]")["age"]
        self.assertEqual(f.range, (0.0, 1.0))

    def test_scalar_bracket_no_word_equals_scalar_en(self):
        # Scalar[0,1] == Scalar en [0,1] == Scalar [0, 1]
        a = _by_name("x: Scalar[0,1]")["x"]
        b = _by_name("x: Scalar en [0, 1]")["x"]
        c = _by_name("x: Scalar [0,1]")["x"]
        self.assertEqual(a.range, (0.0, 1.0))
        self.assertEqual(a.range, b.range)
        self.assertEqual(a.range, c.range)

    def test_scalar_without_range(self):
        f = _by_name("x: Scalar")["x"]
        self.assertEqual((f.kind, f.range), ("scalar", None))

    def test_inverted_range_dropped_with_warning(self):
        r = parse_field_specs("t: Scalar en [95, 18]")
        self.assertIsNone(r.by_name()["t"].range)
        self.assertTrue(any("invertido" in w or "degenerado" in w for w in r.warnings))

    def test_degenerate_range_min_equals_max_dropped(self):
        self.assertIsNone(_by_name("t: Scalar [5, 5]")["t"].range)

    def test_non_numeric_range_dropped_with_warning(self):
        r = parse_field_specs("t: Scalar en [low, high]")
        self.assertIsNone(r.by_name()["t"].range)
        self.assertTrue(r.warnings)

    def test_float_bounds(self):
        self.assertEqual(_by_name("k: Scalar en [0.3, 15.0]")["k"].range, (0.3, 15.0))

    def test_integer_is_scalar_with_flag(self):
        f = _by_name("edad: Integer[0, 120]")["edad"]
        self.assertEqual((f.kind, f.integer, f.range), ("scalar", True, (0.0, 120.0)))


class CategoricalGrammarTest(unittest.TestCase):
    def test_basic_values(self):
        f = _by_name("color: Categorical[red, green, blue]")["color"]
        self.assertEqual((f.kind, f.values), ("categorical", ("red", "green", "blue")))

    def test_accents_and_spaces_preserved(self):
        f = _by_name("esp: Categorical[CARDIOLOGÍA, UCI, Médico de Familia]")["esp"]
        self.assertEqual(f.values, ("CARDIOLOGÍA", "UCI", "Médico de Familia"))

    def test_duplicates_collapse_preserving_order(self):
        f = _by_name("g: Categorical[A, B, A, C, B]")["g"]
        self.assertEqual(f.values, ("A", "B", "C"))

    def test_single_value_downgraded_to_scalar_with_warning(self):
        r = parse_field_specs("t: Categorical[SOLO]")
        self.assertEqual(r.by_name()["t"].kind, "scalar")
        self.assertTrue(any("2 valores" in w or "menos de 2" in w for w in r.warnings))

    def test_case_insensitive_type(self):
        self.assertEqual(_by_name("g: CATEGORICAL[A, B]")["g"].kind, "categorical")
        self.assertEqual(_by_name("g: categórica[A, B]")["g"].kind, "categorical")


class BooleanGrammarTest(unittest.TestCase):
    def test_boolean_variants(self):
        for decl in ("f: Boolean", "f: bool", "f: booleano", "f: BOOLEAN"):
            self.assertEqual(_by_name(decl)["f"].kind, "boolean", decl)


class SeparatorsAndBlockTest(unittest.TestCase):
    def test_multiline_block(self):
        prompt = (
            "FEATURES:\n"
            "  edad: Scalar en [18, 95]\n"
            "  segmento: Categorical[PARTICULAR, PYME, PREMIUM]\n"
            "  tiene_hipoteca: Boolean\n"
        )
        specs = _by_name(prompt)
        self.assertEqual(set(specs), {"edad", "segmento", "tiene_hipoteca"})
        self.assertEqual(specs["segmento"].kind, "categorical")

    def test_comma_and_semicolon_separated_same_line(self):
        specs = _by_name("a: Scalar [0,1], b: Boolean; c: Categorical[X, Y]")
        self.assertEqual(set(specs), {"a", "b", "c"})
        self.assertEqual(specs["c"].values, ("X", "Y"))

    def test_prose_header_before_fields(self):
        # a noisy header with a ':' before the real declarations must not swallow them
        prompt = "FEATURES NUMÉRICAS (2), normalizables: edad: Scalar en [0, 120], imc: Scalar en [10, 70]"
        specs = _by_name(prompt)
        self.assertIn("edad", specs)
        self.assertIn("imc", specs)
        self.assertEqual(specs["edad"].range, (0.0, 120.0))


class RetrocompatAndScopeTest(unittest.TestCase):
    def test_bare_names_not_returned(self):
        # untyped field lists are handled by the legacy extractor, not here
        self.assertEqual(parse_field_specs("FEATURES: edad, imc, tension").fields, [])

    def test_output_probabilitymap_not_a_field(self):
        specs = _by_name(
            "estado: ProbabilityMap[OK, DESGASTE, FALLO]\n"
            "temperatura: Scalar en [0, 150]"
        )
        self.assertIn("temperatura", specs)
        self.assertNotIn("estado", specs)

    def test_metadata_lines_not_fields(self):
        # PROYECTO:/DOMINIO:/MODO: are not field types → ignored
        specs = _by_name(
            "PROYECTO: EstadoMaquina\nDOMINIO: industria\nMODO: multiclase\n"
            "temperatura: Scalar en [0, 150]"
        )
        self.assertEqual(set(specs), {"temperatura"})

    def test_accented_field_name_sanitized(self):
        f = _by_name("categoría: Categorical[A, B]")
        # name is ascii-folded + lowercased
        self.assertIn("categoria", f)

    def test_name_with_leading_accent_not_truncated(self):
        # regression: finditer used to start capture mid-token ("área_clinica"→"rea_clinica")
        f = _by_name("área_clinica: Scalar[0, 1]")
        self.assertIn("area_clinica", f)
        self.assertEqual(f["area_clinica"].range, (0.0, 1.0))

    def test_name_with_hyphen_not_truncated(self):
        f = _by_name("saldo-medio: Scalar[0, 1]")
        self.assertIn("saldo_medio", f)
        self.assertNotIn("medio", f)

    def test_name_with_spaces_and_accent(self):
        f = _by_name("área clínica: Scalar[0, 1]")
        self.assertIn("area_clinica", f)
        self.assertNotIn("clinica", f)

    def test_empty_prompt(self):
        self.assertEqual(parse_field_specs("").fields, [])
        self.assertEqual(parse_field_specs(None).fields, [])  # type: ignore[arg-type]

    def test_duplicate_field_name_first_wins(self):
        specs = _by_name("x: Scalar en [0, 1]\nx: Boolean")
        self.assertEqual(specs["x"].kind, "scalar")


class RealPromptTest(unittest.TestCase):
    def test_banca_prompt(self):
        prompt = (
            "PROYECTO: AbandonoClientesBanca\nMODO: clasificación binaria.\n"
            "FEATURES:\n"
            "  edad: Scalar en [18, 95]\n"
            "  saldo_medio: Scalar en [0, 500000]\n"
            "  segmento: Categorical[PARTICULAR, PYME, PREMIUM, BANCA_PRIVADA]\n"
            "  tiene_hipoteca: Boolean\n"
            "SALIDA: resultado: ProbabilityMap[PERMANECE, ABANDONA]\n"
        )
        specs = _by_name(prompt)
        self.assertEqual(set(specs), {"edad", "saldo_medio", "segmento", "tiene_hipoteca"})
        self.assertEqual(specs["segmento"].values,
                         ("PARTICULAR", "PYME", "PREMIUM", "BANCA_PRIVADA"))
        self.assertEqual(specs["tiene_hipoteca"].kind, "boolean")
        self.assertEqual(specs["edad"].range, (18.0, 95.0))
