from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from matrixai.compiler import PythonBackendCompiler
from matrixai.core import check_mx_types, parse
from matrixai.parser import parse_file, parse_text
from matrixai.runtime import MatrixAIRuntime
from matrixai.types import check_program_types, parse_type_spec


ROOT = Path(__file__).resolve().parents[1]


class MatrixAITypeSystemTest(unittest.TestCase):
    def _load_generated_module(self, path: Path):
        spec = importlib.util.spec_from_file_location("compiled_typed_program", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_parse_base_ai_and_structured_types(self) -> None:
        probability = parse_type_spec("Probability")
        score = parse_type_spec("Score[0, 10]")
        embedding = parse_type_spec("Embedding[1536]")
        tensor = parse_type_spec("Tensor[3, 8]")

        self.assertEqual(probability.name, "Probability")
        self.assertTrue(probability.range.contains(0.5))
        self.assertFalse(probability.range.contains(1.2))
        self.assertEqual(score.range.maximum, 10.0)
        self.assertEqual(embedding.parameters["dim"], 1536)
        self.assertEqual(tensor.parameters["shape"], [3, 8])

    def test_mx_parser_accepts_typed_params_and_return(self) -> None:
        stmt = parse("score(x: Record) -> Score = normalize(relevance(x))")[0]

        self.assertEqual(stmt.param_types["x"].name, "Record")
        self.assertEqual(stmt.return_type.name, "Score")
        self.assertEqual(stmt.to_dict()["return_type"]["name"], "Score")

    def test_mx_typecheck_rejects_incompatible_return(self) -> None:
        stmts = parse("choice(scores: ProbabilityMap) -> Probability = argmax(scores)")

        result = check_mx_types(stmts)

        self.assertFalse(result.ok)
        self.assertTrue(any("declares Probability" in error for error in result.errors))

    def test_mx_typecheck_accepts_score_pipeline(self) -> None:
        stmts = parse(
            "score(x: Record) -> Score = normalize(relevance(x))\n"
            "utility(x: Record) -> Score = score(x) - 0.2 * cost(x)"
        )

        result = check_mx_types(stmts)

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.symbols["score"].name, "Score")

    def test_mxai_parser_accepts_typed_fields_and_outputs(self) -> None:
        program = parse_text(
            """PROJECT TypedScores

VECTOR X[2]
  raw: Score[0, 10]
  confidence: Probability
END

FUNCTION Normed
  normed: Probability = scale(raw, 0, 10)
END

GRAPH
  X -> Normed -> Alert
END

ACTION Alert
  WHEN Normed > 0.8
  POLICY simulate_only
  CALL simulated.alert.send
END

AUDIT
  EXPLAIN X -> Normed -> Alert
END
"""
        )

        self.assertEqual(program.vectors[0].field_types["raw"].name, "Score")
        self.assertEqual(program.functions[0].output_type.name, "Probability")
        self.assertEqual(program.to_dict()["vectors"][0]["field_types"]["raw"]["name"], "Score")

    def test_program_typecheck_rejects_function_output_mismatch(self) -> None:
        program = parse_text(
            """PROJECT BadTypedChoice

VECTOR Email[2]
  urgency: Probability
  confidence: Probability
END

FUNCTION Scores
  scores = softmax(W1 * Email + b1)
END

FUNCTION Choice
  choice: Probability = argmax(scores)
END

GRAPH
  Email -> Scores -> Choice
END

AUDIT
  EXPLAIN Email -> Scores -> Choice
END
"""
        )

        result = check_program_types(program)

        self.assertFalse(result.ok)
        self.assertTrue(any("declares Probability" in error for error in result.errors))

    def test_runtime_rejects_typed_vector_field_out_of_range(self) -> None:
        program = parse_text(
            """PROJECT TypedRuntime

VECTOR X[1]
  confidence: Probability
END

FUNCTION Activation
  activation: Probability = sigmoid(20 * (confidence - 0.5))
END

GRAPH
  X -> Activation -> Alert
END

ACTION Alert
  WHEN Activation > 0.8
  POLICY simulate_only
  CALL simulated.alert.send
END

AUDIT
  EXPLAIN X -> Activation -> Alert
END
"""
        )

        with self.assertRaisesRegex(ValueError, "outside Probability range"):
            MatrixAIRuntime().run(program, {"X": {"confidence": 1.5}})

    def test_compiled_python_rejects_typed_vector_field_out_of_range(self) -> None:
        program = parse_text(
            """PROJECT TypedCompiled

VECTOR X[1]
  confidence: Probability
END

FUNCTION Activation
  activation: Probability = sigmoid(20 * (confidence - 0.5))
END

GRAPH
  X -> Activation -> Alert
END

ACTION Alert
  WHEN Activation > 0.8
  POLICY simulate_only
  CALL simulated.alert.send
END

AUDIT
  EXPLAIN X -> Activation -> Alert
END
"""
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "typed_compiled.py"
            output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
            module = self._load_generated_module(output_path)

        with self.assertRaisesRegex(ValueError, "outside Probability range"):
            module.run({"X": {"confidence": 1.5}})

    def test_typed_domain_examples_typecheck(self) -> None:
        cases = [
            ("email-agent.typed.mxai", "Email", "Classifier", "ProbabilityMap"),
            ("fall-risk.typed.mxai", "Patient", "RiskModel", "Risk"),
            ("pharmacy-dispense.typed.mxai", "Order", "PriorityModel", "Risk"),
        ]

        for filename, vector_name, function_name, output_type in cases:
            with self.subTest(filename=filename):
                program = parse_file(ROOT / "examples" / filename)
                result = check_program_types(program)

                self.assertTrue(result.ok, result.errors)
                self.assertTrue(program.vectors[0].field_types)
                self.assertEqual(program.vectors[0].name, vector_name)
                self.assertEqual(result.symbols[function_name].name, output_type)
                self.assertEqual(result.symbols[program.actions[0].name].name, "ActionResult")

    def test_typed_domain_examples_runtime_matches_compiled(self) -> None:
        cases = [
            ("email-agent.typed.mxai", "email-sample.json"),
            ("fall-risk.typed.mxai", "fall-risk-sample.json"),
            ("pharmacy-dispense.typed.mxai", "pharmacy-dispense-sample.json"),
        ]

        for program_file, input_file in cases:
            with self.subTest(program_file=program_file):
                program = parse_file(ROOT / "examples" / program_file)
                input_data = json.loads((ROOT / "examples" / input_file).read_text(encoding="utf-8"))

                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_path = Path(tmp_dir) / "compiled_typed_domain.py"
                    output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
                    module = self._load_generated_module(output_path)

                generated = module.run(input_data)
                runtime = MatrixAIRuntime().run(program, input_data)

                self.assertEqual(generated["actions"], runtime["actions"])
                self.assertEqual(len(generated["trace"]), len(runtime["trace"]))

    def test_typed_domain_example_rejects_out_of_range_field(self) -> None:
        program = parse_file(ROOT / "examples" / "email-agent.typed.mxai")
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))
        input_data["Email"]["urgency"] = 1.5

        with self.assertRaisesRegex(ValueError, "outside Score range"):
            MatrixAIRuntime().run(program, input_data)


if __name__ == "__main__":
    unittest.main()
