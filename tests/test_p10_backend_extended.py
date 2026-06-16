"""P10 Cut 6 — BackendContractAnalyzer extended: layer_manifest with shapes and param counts."""
from __future__ import annotations

import unittest

from matrixai.compiler import BackendContractAnalyzer
from matrixai.parser import parse_text


_ATTENTION_PROGRAM = """\
PROJECT AttentionTest
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
LAYER Attention(Tensor[4]) -> Tensor[4]
  PARAM Wq Tensor[4, 4]
  PARAM Wk Tensor[4, 4]
  PARAM Wv Tensor[4, 4]
END
LAYER Linear
  PARAM W Tensor[8, 4]
  PARAM b Tensor[8]
END
FUNCTION F
  result = call_layer(Attention, Input)
END
GRAPH
  Input -> F
END
"""

_NO_LAYER_PROGRAM = """\
PROJECT Flat
VECTOR Input[2]
  x : Scalar
  y : Scalar
END
PARAM W Tensor[3, 2]
  TRAINABLE true
END
PARAM b Tensor[3]
  TRAINABLE true
END
FUNCTION F
  result = softmax(W * Input + b)
END
GRAPH
  Input -> F
END
"""


class LayerManifestTest(unittest.TestCase):
    def _report(self, text: str):
        return BackendContractAnalyzer().analyze(parse_text(text))

    def test_layer_manifest_present(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        self.assertIsInstance(report.layer_manifest, list)

    def test_layer_manifest_count(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        self.assertEqual(len(report.layer_manifest), 2)

    def test_attention_layer_in_manifest(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        names = [e["layer"] for e in report.layer_manifest]
        self.assertIn("Attention", names)

    def test_linear_layer_in_manifest(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        names = [e["layer"] for e in report.layer_manifest]
        self.assertIn("Linear", names)

    def test_attention_layer_param_count(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        self.assertEqual(attn["trainable_param_count"], 3)

    def test_linear_layer_param_count(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        linear = next(e for e in report.layer_manifest if e["layer"] == "Linear")
        self.assertEqual(linear["trainable_param_count"], 2)

    def test_layer_param_paths(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        paths = [p["path"] for p in attn["parameters"]]
        self.assertIn("Attention.Wq", paths)
        self.assertIn("Attention.Wk", paths)
        self.assertIn("Attention.Wv", paths)

    def test_layer_param_shapes(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        for p in attn["parameters"]:
            self.assertEqual(p["shape"], [4, 4])

    def test_layer_has_input_type(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        self.assertIn("input_type", attn)
        self.assertEqual(attn["input_type"]["name"], "Tensor")

    def test_layer_has_output_type(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        self.assertIn("output_type", attn)

    def test_empty_layer_manifest_for_flat_program(self) -> None:
        report = self._report(_NO_LAYER_PROGRAM)
        self.assertEqual(report.layer_manifest, [])

    def test_layer_manifest_dtype(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        for p in attn["parameters"]:
            self.assertEqual(p["dtype"], "float32")

    def test_layer_manifest_trainable_flag(self) -> None:
        report = self._report(_ATTENTION_PROGRAM)
        attn = next(e for e in report.layer_manifest if e["layer"] == "Attention")
        for p in attn["parameters"]:
            self.assertTrue(p["trainable"])

    def test_non_trainable_param_excluded(self) -> None:
        text = (
            "PROJECT T\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "LAYER L\n  PARAM W Tensor[4, 4]\n  PARAM frozen Tensor[4, 4] TRAINABLE false\nEND\n"
            "FUNCTION F\n  result = call_layer(L, V)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        report = BackendContractAnalyzer().analyze(parse_text(text))
        layer_entry = report.layer_manifest[0]
        self.assertEqual(layer_entry["trainable_param_count"], 1)
        paths = [p["path"] for p in layer_entry["parameters"]]
        self.assertIn("L.W", paths)
        self.assertNotIn("L.frozen", paths)


if __name__ == "__main__":
    unittest.main()
