"""P10 Cut 8 — Non-regression: P1-P9 programs parse, validate and execute unchanged."""
from __future__ import annotations

import unittest


class P1LanguageRegressionTest(unittest.TestCase):
    def test_email_program_parses(self) -> None:
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT EmailAgent\n"
            "VECTOR Input[3]\n"
            "  urgency : Score[0, 10]\n"
            "  length : Scalar\n"
            "  spam_score : Probability\n"
            "END\n"
            "PARAM W Tensor[3, 3]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION RouteEmail\n"
            "  result = softmax(W * Input + b)\n"
            "END\n"
            "GRAPH\n  Input -> RouteEmail\nEND\n"
        )
        self.assertEqual(program.project, "EmailAgent")
        self.assertEqual(len(program.vectors), 1)
        self.assertEqual(len(program.functions), 1)
        self.assertEqual(len(program.layers), 0)

    def test_fall_risk_program_parses(self) -> None:
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT FallRisk\n"
            "VECTOR Patient[2]\n"
            "  age : Score[0, 100]\n"
            "  mobility : Probability\n"
            "END\n"
            "PARAM W Tensor[1, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Scalar\n  TRAINABLE true\nEND\n"
            "FUNCTION AssessRisk\n"
            "  result = sigmoid(W * Patient + b)\n"
            "END\n"
            "GRAPH\n  Patient -> AssessRisk\nEND\n"
        )
        self.assertEqual(program.project, "FallRisk")


class P2TypeSystemRegressionTest(unittest.TestCase):
    def test_all_p2_types_parse(self) -> None:
        from matrixai.types import parse_type_spec
        for spec in [
            "Probability", "Score[0, 10]", "Risk", "Confidence",
            "Label[A, B, C]", "Vector[10]", "Scalar", "Embedding[1536]",
        ]:
            t = parse_type_spec(spec)
            self.assertIsNotNone(t, spec)

    def test_probability_range(self) -> None:
        from matrixai.types import parse_type_spec
        t = parse_type_spec("Probability")
        self.assertIsNotNone(t.range)
        self.assertEqual(t.range.minimum, 0.0)  # type: ignore[union-attr]
        self.assertEqual(t.range.maximum, 1.0)  # type: ignore[union-attr]

    def test_check_program_types_no_errors(self) -> None:
        from matrixai.parser import parse_text
        from matrixai.types import check_program_types
        program = parse_text(
            "PROJECT T\nVECTOR V[1]\n  x : Scalar\nEND\n"
            "PARAM W Tensor[3, 1]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        result = check_program_types(program)
        self.assertEqual(result.errors, [])


class P3BackendContractRegressionTest(unittest.TestCase):
    def test_backend_contract_ok_for_softmax_linear(self) -> None:
        from matrixai.compiler import BackendContractAnalyzer
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        report = BackendContractAnalyzer().analyze(program)
        self.assertTrue(report.ok)

    def test_parameter_manifest_has_path(self) -> None:
        from matrixai.compiler import BackendContractAnalyzer
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        report = BackendContractAnalyzer().analyze(program)
        for m in report.parameter_manifest:
            self.assertIn("path", m)


class P4TrainingRegressionTest(unittest.TestCase):
    def test_build_initial_parameter_set(self) -> None:
        from matrixai.parameters.store import build_initial_parameter_set
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        ps = build_initial_parameter_set(program)
        self.assertIn("W", ps.parameters)
        self.assertIn("b", ps.parameters)

    def test_validate_parameter_set_ok(self) -> None:
        from matrixai.parameters.store import build_initial_parameter_set, validate_parameter_set
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        ps = build_initial_parameter_set(program)
        result = validate_parameter_set(program, ps)
        self.assertTrue(result.ok, result.errors)


class P1P5DifferentiablePythonRegressionTest(unittest.TestCase):
    def test_differentiable_python_compiles(self) -> None:
        import types
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Tensor[3, 2]\n  TRAINABLE true\nEND\n"
            "PARAM b Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = softmax(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        source = DifferentiablePythonCompiler().compile(program)
        module = types.ModuleType("_reg")
        exec(source, module.__dict__)  # noqa: S102
        result = module.run({"V": {"a": 1.0, "b": 2.0}})
        self.assertIn("F", result["state"])

    def test_sigmoid_linear_compiles(self) -> None:
        import types
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n  a : Scalar\n  b : Scalar\nEND\n"
            "PARAM W Vector[2]\n  TRAINABLE true\nEND\n"
            "PARAM b Scalar\n  TRAINABLE true\nEND\n"
            "FUNCTION F\n  result = sigmoid(W * V + b)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        source = DifferentiablePythonCompiler().compile(program)
        module = types.ModuleType("_reg2")
        exec(source, module.__dict__)  # noqa: S102
        result = module.run({"V": {"a": 1.0, "b": 0.5}})
        self.assertIn("F", result["state"])


class P10NewFeaturesCoexistTest(unittest.TestCase):
    """P10 features (layers, structured types) coexist with P1-P9 features."""

    def test_program_with_layers_and_flat_params(self) -> None:
        from matrixai.compiler import BackendContractAnalyzer
        from matrixai.parser import parse_text
        program = parse_text(
            "PROJECT Mixed\n"
            "VECTOR Input[2]\n  x : Scalar\n  y : Scalar\nEND\n"
            "LAYER Proj\n  PARAM W Tensor[4, 2]\nEND\n"
            "PARAM W2 Tensor[3, 4]\n  TRAINABLE true\nEND\n"
            "PARAM b2 Tensor[3]\n  TRAINABLE true\nEND\n"
            "FUNCTION LayerF\n  result = call_layer(Proj, Input)\nEND\n"
            "FUNCTION ClassF\n  result = softmax(W2 * LayerF + b2)\nEND\n"
            "GRAPH\n  Input -> LayerF -> ClassF\nEND\n"
        )
        self.assertEqual(len(program.layers), 1)
        self.assertEqual(len(program.parameters), 2)
        report = BackendContractAnalyzer().analyze(program)
        # LayerF uses layer_call (in _CONTINUOUS_KINDS), ClassF uses softmax_linear
        fn_names = {n.node: n for n in report.nodes if n.node_type == "function"}
        self.assertTrue(fn_names["LayerF"].supported)
        self.assertTrue(fn_names["ClassF"].supported)

    def test_structured_types_in_vector_fields(self) -> None:
        from matrixai.parser import parse_text
        from matrixai.types import parse_type_spec
        program = parse_text(
            "PROJECT T\nVECTOR V[2]\n"
            "  embedding : Embedding[64]\n"
            "  score : Score[0, 1]\n"
            "END\n"
            "FUNCTION F\n  result = relu(V)\nEND\n"
            "GRAPH\n  V -> F\nEND\n"
        )
        self.assertEqual(len(program.vectors), 1)
        e_type = program.vectors[0].field_types["embedding"]
        self.assertEqual(e_type.name, "Embedding")
        self.assertEqual(e_type.parameters["dim"], 64)


if __name__ == "__main__":
    unittest.main()
