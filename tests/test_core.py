"""Tests for matrixai.core (lexer, parser, AST, evaluator, graph, trace)
and matrixai.functions (math_ops, transforms, scoring).

Coverage:
  - Lexer: numbers, identifiers, operators, comments, errors
  - Parser: literals, variables, binary ops with precedence, unary minus,
            function calls, nested calls, assignments with/without params
  - AST nodes: eval, to_dict, __str__
  - Evaluator: numbers, variables, binary ops, user-defined functions,
               registry functions, nested calls, dotted vars, env pass-through
  - Trace: steps recorded, output correct
  - Graph: nodes and edges from assignment
  - Functions — math_ops: add, sub, mul, div, pow, sqrt, abs, min, max, mean, sum
  - Functions — transforms: normalize, clip, scale, softmax, sigmoid
  - Functions — scoring: relevance, coherence, confidence, cost, argmax, topk, rank
  - CLI eval command: basic run, --trace, --json, --graph flags
  - End-to-end: p1_demo.mx example (score=0.86, utility=0.84)
"""
from __future__ import annotations

import json
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from matrixai.core import (
    AssignNode,
    BinaryOpNode,
    CallNode,
    EvalTrace,
    EvaluationError,
    Evaluator,
    FunctionRegistry,
    LexError,
    NumberNode,
    ParseError,
    VarNode,
    ast_to_graph,
    graph_to_text,
    parse,
    tokenize,
)
from matrixai.functions import build_default_registry
from matrixai.functions.math_ops import abs_, add, div, max_, mean, min_, mul, pow_, sqrt, sub, sum_
from matrixai.functions.scoring import argmax, coherence, confidence, cost, rank, relevance, topk
from matrixai.functions.transforms import clip, normalize, scale, sigmoid, softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval(extra_fns: dict | None = None) -> Evaluator:
    reg = build_default_registry()
    if extra_fns:
        reg.register_many(extra_fns)
    return Evaluator(reg)


# ===========================================================================
# Lexer
# ===========================================================================

class TestLexer(unittest.TestCase):
    def test_number_integer(self):
        toks = tokenize("42")
        self.assertEqual(toks[0].kind, "NUMBER")
        self.assertEqual(toks[0].value, "42")

    def test_number_float(self):
        toks = tokenize("0.6")
        self.assertEqual(toks[0].kind, "NUMBER")
        self.assertEqual(toks[0].value, "0.6")

    def test_number_scientific(self):
        toks = tokenize("1e-3")
        self.assertEqual(toks[0].kind, "NUMBER")

    def test_ident(self):
        toks = tokenize("relevance")
        self.assertEqual(toks[0].kind, "IDENT")
        self.assertEqual(toks[0].value, "relevance")

    def test_dotted_ident(self):
        toks = tokenize("Confidence.max")
        self.assertEqual(toks[0].kind, "IDENT")
        self.assertEqual(toks[0].value, "Confidence.max")

    def test_operators(self):
        toks = tokenize("+ - * /")
        ops = [t.value for t in toks if t.kind == "OP"]
        self.assertEqual(ops, ["+", "-", "*", "/"])

    def test_parens_comma_equals(self):
        kinds = {t.kind for t in tokenize("(x, y) =")}
        self.assertIn("LPAREN", kinds)
        self.assertIn("RPAREN", kinds)
        self.assertIn("COMMA", kinds)
        self.assertIn("EQUALS", kinds)

    def test_comment_ignored(self):
        toks = tokenize("x # this is a comment\n= 1")
        kinds = [t.kind for t in toks if t.kind != "EOF"]
        self.assertNotIn("SKIP", kinds)

    def test_eof_present(self):
        toks = tokenize("")
        self.assertEqual(toks[-1].kind, "EOF")

    def test_unexpected_char_raises(self):
        with self.assertRaises(LexError):
            tokenize("x @ y")


# ===========================================================================
# Parser
# ===========================================================================

class TestParser(unittest.TestCase):
    def test_literal(self):
        stmts = parse("x = 3.14")
        self.assertIsInstance(stmts[0].expr, NumberNode)
        self.assertAlmostEqual(stmts[0].expr.value, 3.14)

    def test_variable(self):
        stmts = parse("result = x")
        self.assertIsInstance(stmts[0].expr, VarNode)
        self.assertEqual(stmts[0].expr.name, "x")

    def test_assignment_no_params(self):
        stmts = parse("decision = x")
        self.assertEqual(stmts[0].name, "decision")
        self.assertEqual(stmts[0].params, [])

    def test_assignment_with_params(self):
        stmts = parse("score(x) = x")
        self.assertEqual(stmts[0].name, "score")
        self.assertEqual(stmts[0].params, ["x"])

    def test_assignment_two_params(self):
        stmts = parse("cosine(a, b) = a")
        self.assertEqual(stmts[0].params, ["a", "b"])

    def test_binop_add(self):
        stmts = parse("r = a + b")
        node = stmts[0].expr
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, "+")

    def test_precedence_mul_before_add(self):
        stmts = parse("r = 2 + 3 * 4")
        # should be 2 + (3*4), so top node is +
        node = stmts[0].expr
        self.assertEqual(node.op, "+")
        self.assertIsInstance(node.right, BinaryOpNode)
        self.assertEqual(node.right.op, "*")

    def test_parentheses_override_precedence(self):
        stmts = parse("r = (2 + 3) * 4")
        node = stmts[0].expr
        self.assertEqual(node.op, "*")
        self.assertIsInstance(node.left, BinaryOpNode)
        self.assertEqual(node.left.op, "+")

    def test_unary_minus(self):
        stmts = parse("r = -1")
        node = stmts[0].expr
        self.assertIsInstance(node, BinaryOpNode)
        self.assertEqual(node.op, "*")
        self.assertAlmostEqual(node.left.value, -1.0)

    def test_function_call_no_args(self):
        stmts = parse("r = f()")
        node = stmts[0].expr
        self.assertIsInstance(node, CallNode)
        self.assertEqual(node.name, "f")
        self.assertEqual(node.args, [])

    def test_function_call_one_arg(self):
        stmts = parse("r = relevance(x)")
        node = stmts[0].expr
        self.assertIsInstance(node, CallNode)
        self.assertEqual(node.name, "relevance")
        self.assertEqual(len(node.args), 1)

    def test_function_call_nested(self):
        stmts = parse("r = normalize(sigmoid(x))")
        outer = stmts[0].expr
        self.assertEqual(outer.name, "normalize")
        inner = outer.args[0]
        self.assertEqual(inner.name, "sigmoid")

    def test_multiple_statements(self):
        src = "score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)\nutility(x) = score(x) - 0.2 * cost(x)"
        stmts = parse(src)
        self.assertEqual(len(stmts), 2)
        self.assertEqual(stmts[0].name, "score")
        self.assertEqual(stmts[1].name, "utility")

    def test_bad_syntax_raises(self):
        with self.assertRaises(ParseError):
            parse("= x")

    def test_to_dict(self):
        stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")
        d = stmts[0].to_dict()
        self.assertEqual(d["type"], "assign")
        self.assertEqual(d["name"], "score")
        self.assertEqual(d["params"], ["x"])
        self.assertIn("expr", d)

    def test_str(self):
        stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")
        s = str(stmts[0])
        self.assertIn("score", s)
        self.assertIn("=", s)


# ===========================================================================
# AST nodes — eval
# ===========================================================================

class TestAstNodes(unittest.TestCase):
    def test_number_node_eval(self):
        self.assertAlmostEqual(NumberNode(3.14).eval({}), 3.14)

    def test_var_node_eval(self):
        self.assertAlmostEqual(VarNode("x").eval({"x": 0.5}), 0.5)

    def test_var_node_dotted(self):
        self.assertAlmostEqual(
            VarNode("C.max").eval({"C": {"max": 0.9, "min": 0.1}}), 0.9
        )

    def test_var_node_missing_raises(self):
        with self.assertRaises(KeyError):
            VarNode("missing").eval({})

    def test_binop_add(self):
        n = BinaryOpNode("+", NumberNode(1.0), NumberNode(2.0))
        self.assertAlmostEqual(n.eval({}), 3.0)

    def test_binop_mul(self):
        n = BinaryOpNode("*", NumberNode(0.6), NumberNode(0.9))
        self.assertAlmostEqual(n.eval({}), 0.54)

    def test_binop_div_by_zero(self):
        n = BinaryOpNode("/", NumberNode(1.0), NumberNode(0.0))
        with self.assertRaises(ZeroDivisionError):
            n.eval({})

    def test_number_to_dict(self):
        d = NumberNode(1.5).to_dict()
        self.assertEqual(d, {"type": "number", "value": 1.5})

    def test_var_to_dict(self):
        d = VarNode("x").to_dict()
        self.assertEqual(d, {"type": "var", "name": "x"})

    def test_binop_to_dict(self):
        d = BinaryOpNode("+", NumberNode(1.0), NumberNode(2.0)).to_dict()
        self.assertEqual(d["type"], "add")
        self.assertIn("left", d)
        self.assertIn("right", d)

    def test_assign_to_dict(self):
        stmts = parse("f(x) = x")
        d = stmts[0].to_dict()
        self.assertEqual(d["type"], "assign")
        self.assertEqual(d["params"], ["x"])


# ===========================================================================
# Evaluator
# ===========================================================================

class TestEvaluator(unittest.TestCase):
    def _ev(self):
        return _make_eval()

    def test_literal(self):
        ev = self._ev()
        ev.define_all(parse("k = 42"))
        val, _ = ev.eval_definition("k", {})
        self.assertAlmostEqual(val, 42.0)

    def test_variable_from_env(self):
        ev = self._ev()
        ev.define_all(parse("r = x"))
        val, _ = ev.eval_definition("r", {"x": 0.7})
        self.assertAlmostEqual(val, 0.7)

    def test_binop(self):
        ev = self._ev()
        ev.define_all(parse("r = a + b"))
        val, _ = ev.eval_definition("r", {"a": 1.0, "b": 2.0})
        self.assertAlmostEqual(val, 3.0)

    def test_precedence_in_eval(self):
        ev = self._ev()
        ev.define_all(parse("r = 2 + 3 * 4"))
        val, _ = ev.eval_definition("r", {})
        self.assertAlmostEqual(val, 14.0)

    def test_registry_function(self):
        ev = self._ev()
        ev.define_all(parse("r = normalize(1.5)"))
        val, _ = ev.eval_definition("r", {})
        self.assertAlmostEqual(val, 1.0)  # clipped

    def test_user_defined_function_call(self):
        ev = self._ev()
        ev.define_all(parse("double(x) = 2 * x\nresult(x) = double(x)"))
        val, _ = ev.eval_definition("result", {"x": 5.0})
        self.assertAlmostEqual(val, 10.0)

    def test_score_example(self):
        """score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)"""
        ev = self._ev()
        ev.define_all(parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)"))
        candidate = {"relevance": 0.9, "coherence": 0.8}
        val, _ = ev.eval_definition("score", candidate)
        # 0.6*0.9 + 0.4*0.8 = 0.54 + 0.32 = 0.86
        self.assertAlmostEqual(val, 0.86, places=6)

    def test_utility_example(self):
        """utility(x) = score(x) - 0.2 * cost(x)  → depends on score"""
        ev = self._ev()
        ev.define_all(parse(
            "score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)\n"
            "utility(x) = score(x) - 0.2 * cost(x)"
        ))
        candidate = {"relevance": 0.9, "coherence": 0.8, "cost": 0.1}
        val, _ = ev.eval_definition("utility", candidate)
        # score=0.86, 0.86 - 0.2*0.1 = 0.84
        self.assertAlmostEqual(val, 0.84, places=6)

    def test_call_method(self):
        ev = self._ev()
        ev.define_all(parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)"))
        candidate = {"relevance": 0.9, "coherence": 0.8}
        val, trace = ev.call("score", [candidate])
        self.assertAlmostEqual(val, 0.86, places=6)
        self.assertIsInstance(trace, EvalTrace)

    def test_undefined_raises(self):
        ev = self._ev()
        with self.assertRaises(EvaluationError):
            ev.eval_definition("nonexistent", {})

    def test_missing_registry_fn_raises(self):
        reg = FunctionRegistry()  # empty registry
        ev = Evaluator(reg)
        ev.define_all(parse("r = unknown_fn(x)"))
        with self.assertRaises(KeyError):
            ev.eval_definition("r", {"x": 1.0})

    def test_nested_calls(self):
        ev = self._ev()
        ev.define_all(parse("r = normalize(sigmoid(0))"))
        val, _ = ev.eval_definition("r", {})
        # sigmoid(0) = 0.5, normalize(0.5) = 0.5
        self.assertAlmostEqual(val, 0.5, places=6)

    def test_unary_minus_eval(self):
        ev = self._ev()
        ev.define_all(parse("r = -x"))
        val, _ = ev.eval_definition("r", {"x": 3.0})
        self.assertAlmostEqual(val, -3.0)


# ===========================================================================
# Trace
# ===========================================================================

class TestTrace(unittest.TestCase):
    def test_output_recorded(self):
        ev = _make_eval()
        ev.define_all(parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)"))
        val, trace = ev.eval_definition("score", {"relevance": 0.9, "coherence": 0.8})
        self.assertAlmostEqual(trace.output, val)

    def test_steps_nonempty(self):
        ev = _make_eval()
        ev.define_all(parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)"))
        _, trace = ev.eval_definition("score", {"relevance": 0.9, "coherence": 0.8})
        self.assertGreater(len(trace.steps), 0)

    def test_trace_contains_function_calls(self):
        ev = _make_eval()
        ev.define_all(parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)"))
        _, trace = ev.eval_definition("score", {"relevance": 0.9, "coherence": 0.8})
        ops = [s.op for s in trace.steps]
        self.assertIn("relevance", ops)
        self.assertIn("coherence", ops)

    def test_trace_to_dict(self):
        ev = _make_eval()
        ev.define_all(parse("r = normalize(0.5)"))
        _, trace = ev.eval_definition("r", {})
        d = trace.to_dict()
        self.assertIn("node", d)
        self.assertIn("expression", d)
        self.assertIn("steps", d)
        self.assertIn("output", d)

    def test_trace_node_name(self):
        ev = _make_eval()
        ev.define_all(parse("myFunc(x) = x"))
        _, trace = ev.eval_definition("myFunc", {"x": 1.0})
        self.assertEqual(trace.node, "myFunc")


# ===========================================================================
# Graph
# ===========================================================================

class TestGraph(unittest.TestCase):
    def _score_graph(self):
        stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")
        return ast_to_graph(stmts[0])

    def test_has_nodes_and_edges(self):
        g = self._score_graph()
        self.assertIn("nodes", g)
        self.assertIn("edges", g)

    def test_has_input_node(self):
        g = self._score_graph()
        types = [n["type"] for n in g["nodes"]]
        self.assertIn("input", types)

    def test_has_output_node(self):
        g = self._score_graph()
        types = [n["type"] for n in g["nodes"]]
        self.assertIn("output", types)

    def test_has_call_nodes(self):
        g = self._score_graph()
        call_names = [n["function"] for n in g["nodes"] if n["type"] == "call"]
        self.assertIn("relevance", call_names)
        self.assertIn("coherence", call_names)

    def test_edges_reference_valid_nodes(self):
        g = self._score_graph()
        node_ids = {n["id"] for n in g["nodes"]}
        for src, dst in g["edges"]:
            self.assertIn(src, node_ids, f"Edge source {src!r} not in nodes")
            self.assertIn(dst, node_ids, f"Edge target {dst!r} not in nodes")

    def test_no_params_graph(self):
        stmts = parse("decision = x")
        g = ast_to_graph(stmts[0])
        types = [n["type"] for n in g["nodes"]]
        self.assertNotIn("input", types)

    def test_graph_to_text(self):
        g = self._score_graph()
        text = graph_to_text(g)
        self.assertIn("Nodes:", text)
        self.assertIn("Edges:", text)


# ===========================================================================
# Functions — math_ops
# ===========================================================================

class TestMathOps(unittest.TestCase):
    def test_add(self):
        self.assertAlmostEqual(add(1.0, 2.0), 3.0)

    def test_sub(self):
        self.assertAlmostEqual(sub(5.0, 3.0), 2.0)

    def test_mul(self):
        self.assertAlmostEqual(mul(2.0, 3.0), 6.0)

    def test_div(self):
        self.assertAlmostEqual(div(6.0, 2.0), 3.0)

    def test_div_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            div(1.0, 0.0)

    def test_pow(self):
        self.assertAlmostEqual(pow_(2.0, 3.0), 8.0)

    def test_sqrt(self):
        self.assertAlmostEqual(sqrt(4.0), 2.0)

    def test_abs_negative(self):
        self.assertAlmostEqual(abs_(-3.5), 3.5)

    def test_min_varargs(self):
        self.assertAlmostEqual(min_(3.0, 1.0, 2.0), 1.0)

    def test_max_varargs(self):
        self.assertAlmostEqual(max_(1.0, 9.0, 5.0), 9.0)

    def test_mean(self):
        self.assertAlmostEqual(mean(1.0, 2.0, 3.0), 2.0)

    def test_sum(self):
        self.assertAlmostEqual(sum_(1.0, 2.0, 3.0), 6.0)


# ===========================================================================
# Functions — transforms
# ===========================================================================

class TestTransforms(unittest.TestCase):
    def test_normalize_clips_high(self):
        self.assertAlmostEqual(normalize(1.5), 1.0)

    def test_normalize_clips_low(self):
        self.assertAlmostEqual(normalize(-0.5), 0.0)

    def test_normalize_passthrough(self):
        self.assertAlmostEqual(normalize(0.7), 0.7)

    def test_clip(self):
        self.assertAlmostEqual(clip(5.0, 0.0, 3.0), 3.0)

    def test_scale(self):
        # 50 in [0,100] → 0.5 in [0,1]
        self.assertAlmostEqual(scale(50.0, 0.0, 100.0), 0.5)

    def test_scale_zero_span(self):
        self.assertAlmostEqual(scale(5.0, 5.0, 5.0), 0.0)

    def test_softmax_sums_to_one(self):
        result = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(result), 1.0)

    def test_softmax_monotone(self):
        result = softmax([1.0, 2.0, 3.0])
        self.assertLess(result[0], result[1])
        self.assertLess(result[1], result[2])

    def test_sigmoid_at_zero(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_sigmoid_large_positive(self):
        self.assertGreater(sigmoid(100.0), 0.999)

    def test_sigmoid_large_negative(self):
        self.assertLess(sigmoid(-100.0), 0.001)


# ===========================================================================
# Functions — scoring
# ===========================================================================

class TestScoring(unittest.TestCase):
    _cand = {"relevance": 0.9, "coherence": 0.8, "cost": 0.1}

    def test_relevance_from_dict(self):
        self.assertAlmostEqual(relevance(self._cand), 0.9)

    def test_coherence_from_dict(self):
        self.assertAlmostEqual(coherence(self._cand), 0.8)

    def test_confidence_missing_returns_zero(self):
        self.assertAlmostEqual(confidence(self._cand), 0.0)

    def test_cost_from_dict(self):
        self.assertAlmostEqual(cost(self._cand), 0.1)

    def test_relevance_from_scalar(self):
        self.assertAlmostEqual(relevance(0.7), 0.7)

    def test_argmax(self):
        items = [0.3, 0.9, 0.5]
        self.assertAlmostEqual(argmax(items), 0.9)

    def test_topk(self):
        items = [0.3, 0.9, 0.5, 0.7]
        result = topk(items, lambda x: x, 2)
        self.assertEqual(result, [0.9, 0.7])

    def test_rank(self):
        items = [0.3, 0.9, 0.5]
        result = rank(items, lambda x: x)
        self.assertEqual(result, [0.9, 0.5, 0.3])


# ===========================================================================
# End-to-end: p1_demo.mx
# ===========================================================================

class TestEndToEnd(unittest.TestCase):
    def test_score_and_utility(self):
        """Full pipeline: parse p1_demo.mx → eval → expected values."""
        demo_path = Path(__file__).parent.parent / "examples" / "p1_demo.mx"
        if not demo_path.exists():
            self.skipTest("examples/p1_demo.mx not found")

        source = demo_path.read_text(encoding="utf-8")
        stmts = parse(source)
        ev = _make_eval()
        ev.define_all(stmts)

        candidate = {"relevance": 0.9, "coherence": 0.8, "cost": 0.1}
        score_val, _ = ev.eval_definition("score", candidate)
        utility_val, _ = ev.eval_definition("utility", candidate)

        self.assertAlmostEqual(score_val, 0.86, places=6)
        self.assertAlmostEqual(utility_val, 0.84, places=6)

    def test_trace_score(self):
        stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")
        ev = _make_eval()
        ev.define_all(stmts)
        candidate = {"relevance": 0.9, "coherence": 0.8}
        val, trace = ev.eval_definition("score", candidate)
        self.assertAlmostEqual(val, 0.86, places=6)
        self.assertEqual(trace.node, "score")
        ops = [s.op for s in trace.steps]
        self.assertIn("relevance", ops)
        self.assertIn("coherence", ops)

    def test_graph_score(self):
        stmts = parse("score(x) = 0.6 * relevance(x) + 0.4 * coherence(x)")
        g = ast_to_graph(stmts[0])
        call_names = {n["function"] for n in g["nodes"] if n["type"] == "call"}
        self.assertEqual(call_names, {"relevance", "coherence"})
        out_names = [n["name"] for n in g["nodes"] if n["type"] == "output"]
        self.assertEqual(out_names, ["score"])


# ===========================================================================
# CLI — eval command
# ===========================================================================

class TestCliEval(unittest.TestCase):
    _demo = str(Path(__file__).parent.parent / "examples" / "p1_demo.mx")
    _input = '{"relevance": 0.9, "coherence": 0.8, "cost": 0.1}'

    def _run(self, *argv) -> tuple[int, str]:
        import sys
        from io import StringIO
        from matrixai.cli import main
        captured = StringIO()
        with patch("sys.stdout", captured), patch("sys.argv", ["matrixai", *argv]):
            code = main()
        return code, captured.getvalue()

    def test_eval_basic(self):
        if not Path(self._demo).exists():
            self.skipTest("p1_demo.mx not found")
        code, out = self._run("eval", self._demo, "--input", self._input)
        self.assertEqual(code, 0)
        self.assertIn("score", out)
        self.assertIn("utility", out)

    def test_eval_json(self):
        if not Path(self._demo).exists():
            self.skipTest("p1_demo.mx not found")
        code, out = self._run("eval", self._demo, "--input", self._input, "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIsInstance(data, list)
        names = [r["name"] for r in data]
        self.assertIn("score", names)
        self.assertIn("utility", names)
        score_entry = next(r for r in data if r["name"] == "score")
        self.assertAlmostEqual(score_entry["value"], 0.86, places=5)

    def test_eval_trace(self):
        if not Path(self._demo).exists():
            self.skipTest("p1_demo.mx not found")
        code, out = self._run(
            "eval", self._demo, "--input", self._input, "--json", "--trace"
        )
        self.assertEqual(code, 0)
        data = json.loads(out)
        score_entry = next(r for r in data if r["name"] == "score")
        self.assertIn("trace", score_entry)
        self.assertIn("steps", score_entry["trace"])

    def test_eval_specific_call(self):
        if not Path(self._demo).exists():
            self.skipTest("p1_demo.mx not found")
        code, out = self._run(
            "eval", self._demo, "--input", self._input, "--call", "score", "--json"
        )
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "score")


if __name__ == "__main__":
    unittest.main()
