from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from matrixai.compiler import PythonBackendCompiler
from matrixai.core import Evaluator, parse
from matrixai.functions import build_default_registry
from matrixai.parser import parse_file


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "tests" / "snapshots" / "p1"


def _snapshot(name: str):
    return json.loads((SNAPSHOT_DIR / name).read_text(encoding="utf-8"))


class MatrixAISnapshotTest(unittest.TestCase):
    maxDiff = None

    def test_p1_mx_trace_json_snapshot(self) -> None:
        source = (ROOT / "examples" / "p1_demo.mx").read_text(encoding="utf-8")
        stmts = parse(source)
        evaluator = Evaluator(build_default_registry())
        evaluator.define_all(stmts)
        env = {"relevance": 0.9, "coherence": 0.8, "cost": 0.1}

        results = []
        for stmt in stmts:
            value, trace = evaluator.eval_definition(stmt.name, env)
            results.append({"name": stmt.name, "value": value, "trace": trace.to_dict()})

        self.assertEqual(results, _snapshot("mx_trace_p1_demo.json"))

    def test_mxai_ir_json_snapshot(self) -> None:
        program = parse_file(str(ROOT / "examples" / "email-agent.mxai"))

        self.assertEqual(program.to_dict(), _snapshot("mxai_ir_email_agent.json"))

    def test_compiled_python_output_snapshot(self) -> None:
        program = parse_file(str(ROOT / "examples" / "email-agent.mxai"))
        input_data = json.loads((ROOT / "examples" / "email-sample.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "compiled_email_agent.py"
            output_path.write_text(PythonBackendCompiler().compile(program), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("compiled_email_agent", output_path)
            if spec is None or spec.loader is None:
                raise AssertionError("Could not load compiled snapshot module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        self.assertEqual(module.run(input_data), _snapshot("compiled_python_email_agent_output.json"))


if __name__ == "__main__":
    unittest.main()
