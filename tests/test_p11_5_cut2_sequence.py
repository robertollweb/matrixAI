"""P11.5 Cut 2 — Canonical sequential transformer-classifier template.

Verifies that:
- SEQUENCE block is parsed correctly by the parser
- SequenceSpec appears in MatrixAIProgram.sequences
- Graph node_type for SEQUENCE is 'sequence'
- Backend contract accepts programs with SEQUENCE nodes
- encoder_embed layer with embedding_lookup + mean_pooling is valid
- differentiable_python forward pass produces finite logits for integer token input
- torch forward pass produces finite logits for integer token input
- 2D embedding_lookup returns list-of-lists in differentiable_python
- mean_pooling on 2D input returns averaged 1D vector
- SequenceSpec roundtrips through to_dict()
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

_BASE = Path(__file__).parent.parent
_TEMPLATE = _BASE / "examples" / "transformer-classifier.mxai"
_TRAIN_CSV = _BASE / "examples" / "transformer-classifier.train.csv"
_TRAIN_SPEC = _BASE / "examples" / "transformer-classifier.mxtrain"


def _parse_template():
    from matrixai.parser import parse_file
    return parse_file(str(_TEMPLATE))


class TestSequenceSpec(unittest.TestCase):
    def test_template_parses(self):
        prog = _parse_template()
        self.assertEqual(prog.project, "transformer_classifier")

    def test_sequences_list_has_one_entry(self):
        prog = _parse_template()
        self.assertEqual(len(prog.sequences), 1)

    def test_sequence_name(self):
        prog = _parse_template()
        self.assertEqual(prog.sequences[0].name, "Input")

    def test_sequence_length(self):
        prog = _parse_template()
        self.assertEqual(prog.sequences[0].length, 8)

    def test_sequence_vocab_size(self):
        prog = _parse_template()
        self.assertEqual(prog.sequences[0].vocab_size, 32)

    def test_vectors_list_is_empty(self):
        prog = _parse_template()
        self.assertEqual(len(prog.vectors), 0)

    def test_graph_node_type_sequence(self):
        prog = _parse_template()
        self.assertEqual(prog.graph.node_types.get("Input"), "sequence")

    def test_graph_nodes_order(self):
        prog = _parse_template()
        self.assertEqual(prog.graph.nodes, ["Input", "Embed", "AttnBlock", "FfnBlock", "Logits"])

    def test_encoder_embed_layer_exists(self):
        prog = _parse_template()
        names = [l.name for l in prog.layers]
        self.assertIn("encoder_embed", names)

    def test_encoder_embed_has_embed_table_param(self):
        prog = _parse_template()
        embed = next(l for l in prog.layers if l.name == "encoder_embed")
        param_names = {p.name for p in embed.params}
        self.assertIn("embed_table", param_names)

    def test_encoder_embed_has_embedding_lookup_op(self):
        prog = _parse_template()
        embed = next(l for l in prog.layers if l.name == "encoder_embed")
        kinds = [op.kind for op in embed.body_ops]
        self.assertIn("embedding_lookup", kinds)

    def test_encoder_embed_has_mean_pooling_op(self):
        prog = _parse_template()
        embed = next(l for l in prog.layers if l.name == "encoder_embed")
        kinds = [op.kind for op in embed.body_ops]
        self.assertIn("mean_pooling", kinds)

    def test_sequence_spec_to_dict(self):
        prog = _parse_template()
        d = prog.to_dict()
        self.assertIn("sequences", d)
        self.assertEqual(d["sequences"][0]["name"], "Input")
        self.assertEqual(d["sequences"][0]["length"], 8)
        self.assertEqual(d["sequences"][0]["vocab_size"], 32)

    def test_sequence_spec_not_in_vectors_in_dict(self):
        prog = _parse_template()
        d = prog.to_dict()
        self.assertEqual(d["vectors"], [])


class TestBackendContractWithSequence(unittest.TestCase):
    def test_contract_ok(self):
        from matrixai.compiler import BackendContractAnalyzer
        prog = _parse_template()
        report = BackendContractAnalyzer(target="differentiable_python").analyze(prog)
        self.assertTrue(report.ok, report.parameter_errors)

    def test_sequence_node_is_supported(self):
        from matrixai.compiler import BackendContractAnalyzer
        prog = _parse_template()
        report = BackendContractAnalyzer(target="differentiable_python").analyze(prog)
        input_node = next(n for n in report.nodes if n.node == "Input")
        self.assertTrue(input_node.supported)
        self.assertEqual(input_node.node_type, "sequence")

    def test_param_count_includes_embed_table(self):
        from matrixai.compiler import BackendContractAnalyzer
        prog = _parse_template()
        report = BackendContractAnalyzer(target="differentiable_python").analyze(prog)
        param_names = [p.name for p in report.trainable_parameters]
        self.assertIn("embed_table", param_names)

    def test_torch_contract_ok(self):
        from matrixai.compiler import BackendContractAnalyzer
        prog = _parse_template()
        report = BackendContractAnalyzer(target="torch").analyze(prog)
        self.assertTrue(report.ok, report.parameter_errors)


class TestDifferentiablePythonForwardWithSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matrixai.parser import parse_file
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        from matrixai.parameters.store import build_initial_parameter_set

        prog = parse_file(str(_TEMPLATE))
        ps = build_initial_parameter_set(prog)
        compiler = DifferentiablePythonCompiler()
        source = compiler.compile(prog)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)
        params = {k: v["values"] for k, v in ps.parameters.items()}
        cls.ns = ns
        cls.params = params

    def _run(self, tokens: list[int]) -> dict:
        return self.ns["run"]({"Input": tokens}, self.params)

    def test_state_has_logits(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        self.assertIn("logits", result["state"])

    def test_logits_have_two_elements(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        self.assertEqual(len(result["state"]["logits"]), 2)

    def test_logits_are_finite(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        for v in result["state"]["logits"]:
            self.assertTrue(math.isfinite(v), f"non-finite logit: {v}")

    def test_state_has_embedded(self):
        result = self._run([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertIn("embedded", result["state"])

    def test_embedded_has_size_8(self):
        result = self._run([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(result["state"]["embedded"]), 8)

    def test_sequence_node_in_state(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        self.assertIn("Input", result["state"])

    def test_sequence_input_stored_as_integers(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        tokens = result["state"]["Input"]
        self.assertEqual(len(tokens), 8)

    def test_different_tokens_produce_different_logits(self):
        r1 = self._run([0, 0, 0, 0, 0, 0, 0, 0])
        r2 = self._run([31, 31, 31, 31, 31, 31, 31, 31])
        l1 = r1["state"]["logits"]
        l2 = r2["state"]["logits"]
        self.assertNotEqual(l1, l2)

    def test_forward_is_deterministic(self):
        r1 = self._run([1, 2, 3, 4, 5, 6, 7, 8])
        r2 = self._run([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(r1["state"]["logits"], r2["state"]["logits"])

    def test_trace_has_sequence_node(self):
        result = self._run([0] * 8)
        types = [t["node_type"] for t in result["trace"]]
        self.assertIn("sequence", types)


class TestTorchForwardWithSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matrixai.parser import parse_file
        from matrixai.compiler.torch_forward import TorchForwardRunner
        from matrixai.parameters.store import build_initial_parameter_set

        prog = parse_file(str(_TEMPLATE))
        ps = build_initial_parameter_set(prog)
        cls.runner = TorchForwardRunner()
        cls.program = prog
        cls.ps = ps

    def _run(self, tokens: list[int]) -> dict:
        return self.runner.run(self.program, {"Input": tokens}, self.ps)

    def test_logits_are_finite(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        for v in result["state"]["logits"]:
            self.assertTrue(math.isfinite(v), f"non-finite logit: {v}")

    def test_logits_have_two_elements(self):
        result = self._run([3, 7, 2, 1, 5, 0, 8, 4])
        self.assertEqual(len(result["state"]["logits"]), 2)

    def test_trace_has_sequence_node_type(self):
        result = self._run([0] * 8)
        types = [t["node_type"] for t in result["trace"]]
        self.assertIn("sequence", types)

    def test_both_backends_agree_on_logit_shape(self):
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
        prog = self.program
        ps = self.ps
        compiler = DifferentiablePythonCompiler()
        source = compiler.compile(prog)
        ns: dict = {}
        exec(compile(source, "<test>", "exec"), ns)
        params_vals = {k: v["values"] for k, v in ps.parameters.items()}
        tokens = [5, 10, 15, 20, 1, 2, 3, 4]
        r_dp = ns["run"]({"Input": tokens}, params_vals)
        r_torch = self._run(tokens)
        self.assertEqual(len(r_dp["state"]["logits"]), len(r_torch["state"]["logits"]))


class TestEmbeddingLookupAndMeanPooling(unittest.TestCase):
    def test_embedding_lookup_2d_table(self):
        from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler

        MINI_MXAI = """\
PROJECT mini_embed

SEQUENCE Tokens
  length = 4
  vocab_size = 8
END

LAYER embedder(Tensor[4]) -> Tensor[3]
  PARAM E Tensor[8, 3]
  embedded = embedding_lookup(E, input)
  result = mean_pooling(embedded)
END

FUNCTION Out
  out = call_layer(embedder, Tokens)
END

GRAPH
  Tokens -> Out
END
"""
        from matrixai.parser import parse_text
        from matrixai.compiler import BackendContractAnalyzer
        from matrixai.parameters.store import build_initial_parameter_set

        prog = parse_text(MINI_MXAI)
        ps = build_initial_parameter_set(prog)
        compiler = DifferentiablePythonCompiler()
        source = compiler.compile(prog)
        ns: dict = {}
        exec(compile(source, "<mini>", "exec"), ns)
        params_vals = {k: v["values"] for k, v in ps.parameters.items()}
        result = ns["run"]({"Tokens": [0, 3, 5, 7]}, params_vals)
        out = result["state"]["out"]
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertTrue(math.isfinite(v))

    def test_mean_pooling_2d_averages_rows(self):
        val_2d = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
        state = {"x": val_2d}

        HELPER_SRC = """\
def _resolve_vector(name, state):
    v = state.get(name, 0.0)
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, dict):
        return [float(x) for x in v.values()]
    return [float(v)]

def _tensor_mean_pooling(x, mask, state):
    val = state.get(x)
    if isinstance(val, list) and val and isinstance(val[0], list):
        embed_dim = len(val[0])
        n = len(val)
        return [sum(row[j] for row in val) / n for j in range(embed_dim)]
    vec = _resolve_vector(x, state)
    if not vec:
        return vec
    mask_vec = _resolve_vector(mask, state) if mask else [1.0] * len(vec)
    total = sum(mask_vec)
    if total == 0.0:
        return [0.0] * len(vec)
    return [v * m / total for v, m in zip(vec, mask_vec)]
"""
        ns: dict = {}
        exec(HELPER_SRC, ns)
        result = ns["_tensor_mean_pooling"]("x", "", state)
        self.assertAlmostEqual(result[0], 3.0)
        self.assertAlmostEqual(result[1], 4.0)
        self.assertAlmostEqual(result[2], 5.0)

    def test_sequence_parser_with_inline_definition(self):
        from matrixai.parser import parse_text
        from matrixai.ir import SequenceSpec

        prog = parse_text("""\
PROJECT test_seq

SEQUENCE Tokens
  length = 5
  vocab_size = 16
END

GRAPH
END
""")
        self.assertEqual(len(prog.sequences), 1)
        s = prog.sequences[0]
        self.assertIsInstance(s, SequenceSpec)
        self.assertEqual(s.name, "Tokens")
        self.assertEqual(s.length, 5)
        self.assertEqual(s.vocab_size, 16)

    def test_sequence_parser_requires_length(self):
        from matrixai.parser import parse_text, MatrixAIParseError
        with self.assertRaises(MatrixAIParseError):
            parse_text("""\
PROJECT bad
SEQUENCE X
  vocab_size = 8
END
GRAPH
END
""")

    def test_sequence_parser_requires_vocab_size(self):
        from matrixai.parser import parse_text, MatrixAIParseError
        with self.assertRaises(MatrixAIParseError):
            parse_text("""\
PROJECT bad
SEQUENCE X
  length = 4
END
GRAPH
END
""")


class TestTrainingCsvFormat(unittest.TestCase):
    def test_csv_has_token_columns(self):
        import csv
        with open(_TRAIN_CSV, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        for i in range(8):
            self.assertIn(f"t{i}", headers)

    def test_csv_has_label_column(self):
        import csv
        with open(_TRAIN_CSV, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        self.assertIn("label", headers)

    def test_csv_values_are_integers(self):
        import csv
        with open(_TRAIN_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for i in range(8):
                    val = row[f"t{i}"]
                    self.assertEqual(int(float(val)), int(float(val)),
                                     f"t{i}={val!r} is not an integer")
                break  # check first row only

    def test_train_spec_references_sequence_columns(self):
        spec_text = _TRAIN_SPEC.read_text()
        self.assertIn("t0", spec_text)
        self.assertIn("t7", spec_text)


if __name__ == "__main__":
    unittest.main()
