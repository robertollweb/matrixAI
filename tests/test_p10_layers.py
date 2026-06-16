"""P10 Cut 2 — LAYER blocks with inline PARAMs and call_layer expressions."""
from __future__ import annotations

import unittest

from matrixai.ir.schema import LayerSpec
from matrixai.parser import parse_text
from matrixai.types import parse_type_spec


_ATTENTION_PROGRAM = """\
PROJECT AttentionTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
LAYER Attention(Tensor[64]) -> Tensor[64]
  PARAM Wq Tensor[64, 64]
  PARAM Wk Tensor[64, 64]
  PARAM Wv Tensor[64, 64]
END
FUNCTION F
  result = call_layer(Attention, Input)
END
GRAPH
  Input -> F
END
"""

_SIMPLE_LAYER_PROGRAM = """\
PROJECT SimpleLayer
VECTOR Input[1]
  x : Scalar
END
LAYER Linear
  PARAM W Tensor[8, 4]
  PARAM b Tensor[8]
END
FUNCTION F
  result = call_layer(Linear, Input)
END
GRAPH
  Input -> F
END
"""

_BARE_LAYER_PROGRAM = """\
PROJECT BareLayer
VECTOR Input[1]
  x : Scalar
END
LAYER Proj
END
FUNCTION F
  result = softmax(W * Input + b)
END
PARAM W Tensor[4, 1]
  TRAINABLE true
END
PARAM b Tensor[4]
  TRAINABLE true
END
GRAPH
  Input -> F
END
"""


class LayerParseTest(unittest.TestCase):
    def test_attention_layer_parsed(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        self.assertEqual(len(program.layers), 1)
        layer = program.layers[0]
        self.assertIsInstance(layer, LayerSpec)
        self.assertEqual(layer.name, "Attention")

    def test_attention_layer_has_three_params(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        layer = program.layers[0]
        self.assertEqual(len(layer.params), 3)
        names = [p.name for p in layer.params]
        self.assertIn("Wq", names)
        self.assertIn("Wk", names)
        self.assertIn("Wv", names)

    def test_attention_layer_param_shapes(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        layer = program.layers[0]
        for p in layer.params:
            self.assertEqual(p.type_spec.parameters["shape"], [64, 64])

    def test_attention_layer_input_type(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        layer = program.layers[0]
        self.assertIsNotNone(layer.input_type)
        self.assertEqual(layer.input_type.name, "Tensor")  # type: ignore[union-attr]
        self.assertEqual(layer.input_type.parameters["shape"], [64])  # type: ignore[union-attr]

    def test_attention_layer_output_type(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        layer = program.layers[0]
        self.assertIsNotNone(layer.output_type)
        self.assertEqual(layer.output_type.name, "Tensor")  # type: ignore[union-attr]
        self.assertEqual(layer.output_type.parameters["shape"], [64])  # type: ignore[union-attr]

    def test_layer_params_trainable_by_default(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        layer = program.layers[0]
        for p in layer.params:
            self.assertTrue(p.trainable)

    def test_simple_layer_parsed(self) -> None:
        program = parse_text(_SIMPLE_LAYER_PROGRAM)
        self.assertEqual(len(program.layers), 1)
        layer = program.layers[0]
        self.assertEqual(layer.name, "Linear")
        self.assertEqual(len(layer.params), 2)

    def test_bare_layer_no_params(self) -> None:
        program = parse_text(_BARE_LAYER_PROGRAM)
        self.assertEqual(len(program.layers), 1)
        layer = program.layers[0]
        self.assertEqual(layer.name, "Proj")
        self.assertEqual(len(layer.params), 0)
        self.assertIsNone(layer.input_type)
        self.assertIsNone(layer.output_type)

    def test_no_layers_by_default(self) -> None:
        program = parse_text(
            "PROJECT Empty\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "FUNCTION F\n  result = x\nEND\nGRAPH\n  V -> F\nEND\n"
        )
        self.assertEqual(program.layers, [])

    def test_layer_inline_param_trainable_false(self) -> None:
        text = (
            "PROJECT T\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "LAYER L\n  PARAM W Tensor[4, 4] TRAINABLE false\nEND\n"
            "FUNCTION F\n  result = call_layer(L, V)\nEND\nGRAPH\n  V -> F\nEND\n"
        )
        program = parse_text(text)
        layer = program.layers[0]
        self.assertFalse(layer.params[0].trainable)

    def test_layer_inline_param_init(self) -> None:
        text = (
            "PROJECT T\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "LAYER L\n  PARAM W Tensor[4, 4] INIT zeros\nEND\n"
            "FUNCTION F\n  result = call_layer(L, V)\nEND\nGRAPH\n  V -> F\nEND\n"
        )
        program = parse_text(text)
        layer = program.layers[0]
        self.assertEqual(layer.params[0].initializer, "zeros")


class CallLayerExpressionTest(unittest.TestCase):
    def test_call_layer_kind(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.kind, "layer_call")

    def test_call_layer_layer_name(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.parameters["layer"], "Attention")

    def test_call_layer_input(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.parameters["input"], "Input")
        self.assertIn("Input", fn.semantic.inputs)

    def test_call_layer_simple_linear(self) -> None:
        program = parse_text(_SIMPLE_LAYER_PROGRAM)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.kind, "layer_call")
        self.assertEqual(fn.semantic.parameters["layer"], "Linear")


class LayerToDictTest(unittest.TestCase):
    def test_layers_appear_in_to_dict(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        d = program.to_dict()
        self.assertIn("layers", d)
        self.assertEqual(len(d["layers"]), 1)

    def test_layer_dict_has_name(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        d = program.to_dict()
        self.assertEqual(d["layers"][0]["name"], "Attention")

    def test_layer_dict_has_parameters(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        d = program.to_dict()
        params = d["layers"][0]["parameters"]
        self.assertEqual(len(params), 3)

    def test_layer_dict_has_types(self) -> None:
        program = parse_text(_ATTENTION_PROGRAM)
        d = program.to_dict()
        self.assertIn("input_type", d["layers"][0])
        self.assertIn("output_type", d["layers"][0])

    def test_no_layers_key_when_empty(self) -> None:
        program = parse_text(_BARE_LAYER_PROGRAM)
        # Bare PROJ layer with no params but it still exists
        d = program.to_dict()
        self.assertIn("layers", d)

    def test_no_layers_key_when_truly_empty(self) -> None:
        program = parse_text(
            "PROJECT Empty\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "FUNCTION F\n  result = x\nEND\nGRAPH\n  V -> F\nEND\n"
        )
        d = program.to_dict()
        self.assertNotIn("layers", d)


class BackwardCompatLayerTest(unittest.TestCase):
    """Programs without LAYER blocks must parse identically to before."""

    def test_existing_program_unchanged(self) -> None:
        from matrixai.parser import parse_text as pt
        program = pt(
            "PROJECT Email\n"
            "VECTOR Input[2]\n  urgency : Score[0, 10]\n  length : Scalar\nEND\n"
            "PARAM W Tensor[2, 2]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * Input + W)\nEND\n"
            "GRAPH\n  Input -> F\nEND\n"
        )
        self.assertEqual(program.layers, [])
        self.assertEqual(len(program.parameters), 1)
        self.assertEqual(len(program.functions), 1)


if __name__ == "__main__":
    unittest.main()
