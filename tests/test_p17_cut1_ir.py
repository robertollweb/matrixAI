from __future__ import annotations

import unittest

from matrixai.compiler.backend_contract import BackendContractAnalyzer
from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
from matrixai.parser import parse_text
from matrixai.types import semantic_kind_output_type


CELSIUS_KELVIN_MXAI = """\
PROJECT CelsiusToKelvin

VECTOR Reading[1]
  celsius: Scalar
END

PARAM W1 Vector[1]
END

PARAM b1 Scalar
END

FUNCTION KelvinPrediction
  predicted_kelvin: Scalar = linear(W1 * Reading + b1)
END

GRAPH
  Reading -> KelvinPrediction
END
"""

MULTI_FEATURE_MXAI = """\
PROJECT HousePrice

VECTOR Features[3]
  sqm: Scalar[20, 500]
  rooms: Scalar[1, 10]
  floor: Scalar[0, 30]
END

PARAM W1 Vector[3]
END

PARAM b1 Scalar
END

FUNCTION PricePrediction
  predicted_price: Scalar = linear(W1 * Features + b1)
END

GRAPH
  Features -> PricePrediction
END
"""


class TestParserLinearRegression(unittest.TestCase):

    def test_parse_linear_produces_linear_regression_kind(self):
        program = parse_text(CELSIUS_KELVIN_MXAI)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.kind, "linear_regression")

    def test_parse_linear_inputs_correct(self):
        program = parse_text(CELSIUS_KELVIN_MXAI)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.inputs, ["Reading"])

    def test_parse_linear_parameters_correct(self):
        program = parse_text(CELSIUS_KELVIN_MXAI)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.parameters["weights"], "W1")
        self.assertEqual(fn.semantic.parameters["bias"], "b1")

    def test_parse_linear_output_name(self):
        program = parse_text(CELSIUS_KELVIN_MXAI)
        fn = program.functions[0]
        self.assertEqual(fn.output, "predicted_kelvin")

    def test_parse_linear_multi_feature(self):
        program = parse_text(MULTI_FEATURE_MXAI)
        fn = program.functions[0]
        self.assertEqual(fn.semantic.kind, "linear_regression")
        self.assertEqual(fn.semantic.inputs, ["Features"])


class TestSemanticKindOutputType(unittest.TestCase):

    def test_linear_regression_returns_scalar(self):
        from matrixai.types import SCALAR
        result = semantic_kind_output_type("linear_regression")
        self.assertEqual(result.name, SCALAR.name)

    def test_sigmoid_linear_still_probability(self):
        result = semantic_kind_output_type("sigmoid_linear")
        self.assertEqual(result.name, "Probability")

    def test_softmax_linear_still_probability_map(self):
        result = semantic_kind_output_type("softmax_linear")
        self.assertIn("Map", result.name)


class TestBackendContractLinearRegression(unittest.TestCase):

    def _analyze(self, mxai: str) -> object:
        program = parse_text(mxai)
        return BackendContractAnalyzer().analyze(program)

    def test_backend_contract_ok(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        self.assertTrue(report.ok, msg=str(report.parameter_errors))

    def test_trainable_parameters_found(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        names = {p.name for p in report.trainable_parameters}
        self.assertIn("W1", names)
        self.assertIn("b1", names)

    def test_weights_shape_matches_input_dim(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        w = next(p for p in report.trainable_parameters if p.name == "W1")
        self.assertEqual(w.shape, (1,))

    def test_bias_shape_is_scalar(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        b = next(p for p in report.trainable_parameters if p.name == "b1")
        self.assertEqual(b.shape, ())

    def test_weights_role(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        w = next(p for p in report.trainable_parameters if p.name == "W1")
        self.assertEqual(w.role, "weights")

    def test_bias_role(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        b = next(p for p in report.trainable_parameters if p.name == "b1")
        self.assertEqual(b.role, "bias")

    def test_function_output_shape_scalar(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        shapes = report.tensor_shapes
        self.assertIn("KelvinPrediction", shapes)
        self.assertEqual(shapes["KelvinPrediction"], [])

    def test_multi_feature_weights_shape(self):
        report = self._analyze(MULTI_FEATURE_MXAI)
        w = next(p for p in report.trainable_parameters if p.name == "W1")
        self.assertEqual(w.shape, (3,))

    def test_function_node_differentiable(self):
        report = self._analyze(CELSIUS_KELVIN_MXAI)
        fn_node = next(n for n in report.nodes if n.node == "KelvinPrediction")
        self.assertTrue(fn_node.differentiable)
        self.assertTrue(fn_node.supported)

    def test_torch_backend_gates_regression_for_p17_1(self):
        program = parse_text(CELSIUS_KELVIN_MXAI)
        report = BackendContractAnalyzer(target="torch").analyze(program)
        fn_node = next(n for n in report.nodes if n.node == "KelvinPrediction")
        self.assertFalse(fn_node.supported)
        self.assertFalse(fn_node.differentiable)
        self.assertIn("P17.1", fn_node.reason)


class TestDifferentiablePythonCompiler(unittest.TestCase):

    def _compile_and_run(self, mxai: str, inputs: dict, params: dict) -> dict:
        program = parse_text(mxai)
        source = DifferentiablePythonCompiler().compile(program)
        ns: dict = {}
        exec(compile(source, "<generated>", "exec"), ns)
        return ns["run"](inputs, params)

    def test_compile_succeeds(self):
        program = parse_text(CELSIUS_KELVIN_MXAI)
        source = DifferentiablePythonCompiler().compile(program)
        self.assertIn("_parameterized_linear_regression", source)

    def test_run_with_known_weights_computes_kelvin(self):
        # W1=1.0, b1=273.15 → kelvin = celsius + 273.15
        result = self._compile_and_run(
            CELSIUS_KELVIN_MXAI,
            {"Reading": {"celsius": 0.0}},
            {"W1": [1.0], "b1": 273.15},
        )
        predicted = result["state"]["predicted_kelvin"]
        self.assertAlmostEqual(float(predicted), 273.15, places=4)

    def test_run_100_celsius(self):
        result = self._compile_and_run(
            CELSIUS_KELVIN_MXAI,
            {"Reading": {"celsius": 100.0}},
            {"W1": [1.0], "b1": 273.15},
        )
        predicted = result["state"]["predicted_kelvin"]
        self.assertAlmostEqual(float(predicted), 373.15, places=4)

    def test_run_negative_celsius(self):
        result = self._compile_and_run(
            CELSIUS_KELVIN_MXAI,
            {"Reading": {"celsius": -273.15}},
            {"W1": [1.0], "b1": 273.15},
        )
        predicted = result["state"]["predicted_kelvin"]
        self.assertAlmostEqual(float(predicted), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
