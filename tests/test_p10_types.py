"""P10 Cut 1 — Structured types: Tensor, Sequence, Embedding[vocab,dim], List, Map."""
from __future__ import annotations

import unittest

from matrixai.types import (
    TypeSpec,
    embedding_dims,
    parse_type_spec,
    tensor_shape,
    validate_value_against_type,
)


class TensorTypeTest(unittest.TestCase):
    def test_tensor_1d(self) -> None:
        t = parse_type_spec("Tensor[8]")
        self.assertEqual(t.name, "Tensor")
        self.assertEqual(t.parameters["shape"], [8])

    def test_tensor_2d(self) -> None:
        t = parse_type_spec("Tensor[4, 8]")
        self.assertEqual(t.parameters["shape"], [4, 8])

    def test_tensor_3d(self) -> None:
        t = parse_type_spec("Tensor[2, 4, 8]")
        self.assertEqual(t.parameters["shape"], [2, 4, 8])

    def test_tensor_shape_helper(self) -> None:
        t = parse_type_spec("Tensor[4, 8]")
        self.assertEqual(tensor_shape(t), (4, 8))

    def test_tensor_shape_1d_helper(self) -> None:
        t = parse_type_spec("Tensor[16]")
        self.assertEqual(tensor_shape(t), (16,))

    def test_tensor_bare_has_no_shape(self) -> None:
        t = parse_type_spec("Tensor")
        self.assertIsNone(tensor_shape(t))


class EmbeddingTypeTest(unittest.TestCase):
    def test_embedding_single_dim(self) -> None:
        t = parse_type_spec("Embedding[1536]")
        self.assertEqual(t.name, "Embedding")
        self.assertEqual(t.parameters["dim"], 1536)
        self.assertNotIn("vocab", t.parameters)

    def test_embedding_vocab_dim(self) -> None:
        t = parse_type_spec("Embedding[512, 64]")
        self.assertEqual(t.parameters["vocab"], 512)
        self.assertEqual(t.parameters["dim"], 64)

    def test_embedding_dims_helper(self) -> None:
        t = parse_type_spec("Embedding[512, 64]")
        self.assertEqual(embedding_dims(t), (512, 64))

    def test_embedding_dims_single_returns_none(self) -> None:
        t = parse_type_spec("Embedding[64]")
        self.assertIsNone(embedding_dims(t))

    def test_tensor_shape_for_embedding_vocab_dim(self) -> None:
        t = parse_type_spec("Embedding[512, 64]")
        self.assertEqual(tensor_shape(t), (512, 64))

    def test_tensor_shape_for_embedding_single_dim(self) -> None:
        t = parse_type_spec("Embedding[64]")
        self.assertEqual(tensor_shape(t), (64,))


class SequenceTypeTest(unittest.TestCase):
    def test_sequence_tensor_1d(self) -> None:
        t = parse_type_spec("Sequence[Tensor[4], 128]")
        self.assertEqual(t.name, "Sequence")
        self.assertEqual(t.parameters["length"], 128)
        element = TypeSpec.from_dict(t.parameters["element_type"])
        self.assertIsNotNone(element)
        self.assertEqual(element.name, "Tensor")  # type: ignore[union-attr]
        self.assertEqual(element.parameters["shape"], [4])  # type: ignore[union-attr]

    def test_sequence_tensor_2d(self) -> None:
        # Nested brackets: Tensor[4,8] must not be split at the inner comma
        t = parse_type_spec("Sequence[Tensor[4, 8], 32]")
        self.assertEqual(t.parameters["length"], 32)
        element = TypeSpec.from_dict(t.parameters["element_type"])
        self.assertIsNotNone(element)
        self.assertEqual(element.parameters["shape"], [4, 8])  # type: ignore[union-attr]

    def test_sequence_scalar(self) -> None:
        t = parse_type_spec("Sequence[Scalar, 64]")
        self.assertEqual(t.parameters["length"], 64)
        element = TypeSpec.from_dict(t.parameters["element_type"])
        self.assertIsNotNone(element)
        self.assertEqual(element.name, "Scalar")  # type: ignore[union-attr]


class ListTypeTest(unittest.TestCase):
    def test_list_probability(self) -> None:
        t = parse_type_spec("List[Probability]")
        self.assertEqual(t.name, "List")
        element = TypeSpec.from_dict(t.parameters["element_type"])
        self.assertIsNotNone(element)
        self.assertEqual(element.name, "Probability")  # type: ignore[union-attr]

    def test_list_tensor(self) -> None:
        t = parse_type_spec("List[Tensor[8]]")
        element = TypeSpec.from_dict(t.parameters["element_type"])
        self.assertIsNotNone(element)
        self.assertEqual(element.parameters["shape"], [8])  # type: ignore[union-attr]

    def test_list_bare_still_works(self) -> None:
        t = parse_type_spec("List")
        self.assertEqual(t.name, "List")
        self.assertNotIn("element_type", t.parameters)


class MapTypeTest(unittest.TestCase):
    def test_map_string_score(self) -> None:
        t = parse_type_spec("Map[String, Score]")
        self.assertEqual(t.name, "Map")
        k = TypeSpec.from_dict(t.parameters["key_type"])
        v = TypeSpec.from_dict(t.parameters["value_type"])
        self.assertIsNotNone(k)
        self.assertIsNotNone(v)
        self.assertEqual(k.name, "String")  # type: ignore[union-attr]
        self.assertEqual(v.name, "Score")  # type: ignore[union-attr]

    def test_map_bare_still_works(self) -> None:
        t = parse_type_spec("Map")
        self.assertEqual(t.name, "Map")
        self.assertNotIn("key_type", t.parameters)


class ValidateStructuredTypesTest(unittest.TestCase):
    def test_list_validates_elements(self) -> None:
        t = parse_type_spec("List[Probability]")
        errors = validate_value_against_type("probs", [0.1, 0.9, 0.5], t)
        self.assertEqual(errors, [])

    def test_list_rejects_non_list(self) -> None:
        t = parse_type_spec("List[Probability]")
        errors = validate_value_against_type("probs", 0.5, t)
        self.assertTrue(len(errors) > 0)

    def test_sequence_max_length(self) -> None:
        t = parse_type_spec("Sequence[Scalar, 4]")
        errors = validate_value_against_type("seq", [1.0, 2.0, 3.0, 4.0, 5.0], t)
        self.assertTrue(len(errors) > 0)

    def test_sequence_within_length_ok(self) -> None:
        t = parse_type_spec("Sequence[Scalar, 8]")
        errors = validate_value_against_type("seq", [1.0, 2.0], t)
        self.assertEqual(errors, [])

    def test_map_validates_dict(self) -> None:
        t = parse_type_spec("Map[String, Scalar]")
        errors = validate_value_against_type("m", {"a": 1.0}, t)
        self.assertEqual(errors, [])

    def test_map_rejects_non_dict(self) -> None:
        t = parse_type_spec("Map[String, Scalar]")
        errors = validate_value_against_type("m", [1, 2], t)
        self.assertTrue(len(errors) > 0)


class BackwardCompatTest(unittest.TestCase):
    """Existing P1-P5 type annotations must still parse correctly."""

    def test_probability(self) -> None:
        t = parse_type_spec("Probability")
        self.assertEqual(t.name, "Probability")
        self.assertIsNotNone(t.range)

    def test_score_with_range(self) -> None:
        t = parse_type_spec("Score[0, 10]")
        self.assertEqual(t.name, "Score")
        self.assertIsNotNone(t.range)
        self.assertEqual(t.range.maximum, 10.0)  # type: ignore[union-attr]

    def test_embedding_single_dim_unchanged(self) -> None:
        t = parse_type_spec("Embedding[1536]")
        self.assertEqual(t.parameters["dim"], 1536)
        self.assertNotIn("shape", t.parameters)

    def test_tensor_2d_unchanged(self) -> None:
        t = parse_type_spec("Tensor[3, 8]")
        self.assertEqual(t.parameters["shape"], [3, 8])

    def test_vector(self) -> None:
        t = parse_type_spec("Vector[10]")
        self.assertEqual(t.parameters["dim"], 10)

    def test_label_args(self) -> None:
        t = parse_type_spec("Label[A, B, C]")
        self.assertEqual(t.parameters["args"], ["A", "B", "C"])

    def test_record_bare(self) -> None:
        t = parse_type_spec("Record")
        self.assertEqual(t.name, "Record")


class TypeCheckerParamShapeTest(unittest.TestCase):
    """check_program_types warns on Tensor PARAM without declared shape."""

    def _make_program(self, param_type: str) -> object:
        from matrixai.parser import parse_text
        return parse_text(
            f"PROJECT Test\n"
            f"VECTOR Input[1]\n"
            f"  x : Scalar\n"
            f"END\n"
            f"PARAM W {param_type}\n"
            f"  TRAINABLE true\n"
            f"END\n"
            f"FUNCTION F\n"
            f"  result = softmax(W * Input + W)\n"
            f"END\n"
            f"GRAPH\n"
            f"  Input -> F\n"
            f"END\n"
        )

    def test_tensor_with_shape_no_warning(self) -> None:
        from matrixai.types import check_program_types
        program = self._make_program("Tensor[4, 8]")
        result = check_program_types(program)
        shape_warnings = [w for w in result.warnings if "without declared shape" in w]
        self.assertEqual(shape_warnings, [])

    def test_tensor_bare_emits_warning(self) -> None:
        from matrixai.types import check_program_types
        program = self._make_program("Tensor")
        result = check_program_types(program)
        shape_warnings = [w for w in result.warnings if "without declared shape" in w]
        self.assertTrue(len(shape_warnings) > 0)


if __name__ == "__main__":
    unittest.main()
