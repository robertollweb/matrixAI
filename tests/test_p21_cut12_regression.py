"""P21 C12 — Regression: P1–P20 programs unchanged + canonical example."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram, NetworkSpec
from matrixai.parser.parser import parse_text
from matrixai.registry import (
    ModelRegistry,
    RegistryEntry,
    compute_entry_hash,
    compute_composite_model_hash,
)
from matrixai.actions.trace import ActionTrace, sign_action_trace, verify_action_trace

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version=_MATRIXAI_VERSION,
)


def _make_entry(name: str, version: str = "v1", **overrides) -> RegistryEntry:
    kw = {**_BASE, **overrides, "name": name, "version": version}
    hash_kw = {k: v for k, v in kw.items() if k in {
        "name", "version", "model_hash", "parameter_schema_hash",
        "parameter_set_id", "training_trace_hash", "evaluation_report_hash", "matrixai_version"
    }}
    eh = compute_entry_hash(**hash_kw)
    return RegistryEntry(
        name=name, version=version, entry_hash=eh,
        model_hash=kw["model_hash"],
        parameter_schema_hash=kw["parameter_schema_hash"],
        parameter_set_id=kw["parameter_set_id"],
        input_type=kw.get("input_type", {"kind": "VECTOR"}),
        output_type=kw.get("output_type", {"kind": "Tensor", "shape": [128]}),
        metrics=kw.get("metrics", {"accuracy": 0.91}),
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


# ── existing models without imports unchanged ─────────────────────────────────

_CLASSIC_MXAI = """
PROJECT EmailAgent

VECTOR Email[3]
  urgency
  sender_trust
  topic_support
END

FUNCTION Classifier
  C = softmax(W1 * Email + b1)
END

DISTRIBUTION Confidence
  Confidence ~ Categorical(C)
END

GRAPH
  Email -> Classifier
  Classifier -> Confidence
END
"""

_DENSE_MXAI = """
PROJECT HousePrice

VECTOR House[3]
  size_m2: Scalar
  rooms: Scalar
  age: Scalar
END

NETWORK PriceNet
  INPUT House
  LAYER Dense units=8 activation=relu
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT price: Scalar
END

GRAPH
  House -> PriceNet
END
"""


def test_existing_models_without_imports_unchanged():
    """Programs without IMPORT produce identical structure as before P21."""
    prog = parse_text(_CLASSIC_MXAI)
    assert prog.project == "EmailAgent"
    assert prog.imports == []
    assert len(prog.functions) == 1
    assert prog.functions[0].name == "Classifier"


def test_program_without_imports_has_empty_imports_list():
    prog = parse_text(_DENSE_MXAI)
    assert prog.imports == []
    assert prog.networks[0].name == "PriceNet"


def test_to_dict_excludes_imports_when_empty():
    prog = parse_text(_CLASSIC_MXAI)
    d = prog.to_dict()
    # imports key absent or empty list — either is acceptable
    assert d.get("imports", []) == []


# ── P18 dense networks can be registered ─────────────────────────────────────

def test_p18_dense_networks_can_be_registered(tmp_path):
    """A P18-style entry (no blockers, evaluation_report_hash set) pushes successfully."""
    reg = ModelRegistry(tmp_path / "reg")
    entry = _make_entry("price_net")
    reg.push(entry)
    retrieved = reg.get("price_net", "v1")
    assert retrieved.name == "price_net"
    assert retrieved.entry_hash == entry.entry_hash


def test_p18_registry_entry_survives_round_trip(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    entry = _make_entry("dense_v18", metrics={"loss": 0.03, "mse": 0.12})
    reg.push(entry)
    got = reg.get("dense_v18", "v1")
    assert got.metrics["mse"] == pytest.approx(0.12)


# ── P17 regression models can be registered ──────────────────────────────────

def test_p17_regression_models_can_be_registered(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    entry = _make_entry(
        "linear_regressor",
        input_type={"kind": "VECTOR", "size": 5},
        output_type={"kind": "Scalar"},
        metrics={"mse": 0.05},
    )
    reg.push(entry)
    got = reg.get("linear_regressor", "v1")
    assert got.output_type["kind"] == "Scalar"


def test_p17_multiple_versions_coexist(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("linear", version="v1"))
    reg.push(_make_entry("linear", version="v2", model_hash="sha256:" + "e" * 64))
    v1 = reg.get("linear", "v1")
    v2 = reg.get("linear", "v2")
    assert v1.entry_hash != v2.entry_hash


# ── suite passes without signing key ─────────────────────────────────────────

def test_suite_passes_without_signing_key(tmp_path, monkeypatch):
    """Registry push/get works with no signing key in env."""
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    entry = _make_entry("unsigned_model")
    reg.push(entry)
    got = reg.get("unsigned_model", "v1")
    assert got.name == "unsigned_model"


def test_verify_without_signing_key_passes(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("model_no_key"))
    ok = reg.verify("model_no_key", "v1")
    assert ok


# ── torch remains optional ────────────────────────────────────────────────────

def test_torch_remains_optional():
    """Core registry operations must not import torch."""
    import sys
    torch_was_loaded = "torch" in sys.modules
    from matrixai.registry import ModelRegistry, RegistryEntry, compute_entry_hash  # noqa
    if not torch_was_loaded:
        assert "torch" not in sys.modules, "registry import pulled in torch unexpectedly"


def test_ir_schema_does_not_require_torch():
    """IR schema (including ImportSpec) must be importable without torch."""
    import sys
    torch_was_loaded = "torch" in sys.modules
    from matrixai.ir.schema import ImportSpec, MatrixAIProgram  # noqa
    if not torch_was_loaded:
        assert "torch" not in sys.modules


# ── P20 action traces remain valid for non-composite models ──────────────────

def _make_trace(model_hash: str, hmac_sig: str | None = None) -> ActionTrace:
    return ActionTrace(
        report_id="r001",
        model_hash=model_hash,
        parameter_set_id="ps_001",
        action_contract_hash="sha256:" + "0" * 64,
        input_hash="sha256:" + "1" * 64,
        executed_at="2026-06-01T12:00:00+00:00",
        executor_kind="simulate_only",
        ok=True,
        response_summary="ok",
        error=None,
        latency_ms=0.0,
        hmac_signature=hmac_sig,
    )


def test_p20_action_traces_remain_valid_for_non_composite_models():
    """ActionTrace created with a plain model_hash (non-composite) still signs/verifies."""
    plain_hash = "sha256:" + "a" * 64
    trace = _make_trace(plain_hash)
    key = "b" * 64
    sig = sign_action_trace(trace, key)
    signed = _make_trace(plain_hash, hmac_sig=sig)
    assert verify_action_trace(signed, key) is True


def test_p20_trace_model_hash_field_unchanged():
    """Existing ActionTrace.model_hash field accepts any sha256 value."""
    trace = _make_trace("sha256:" + "f" * 64)
    assert trace.model_hash.startswith("sha256:")


# ── canonical example: text-routing-pipeline.mxai ────────────────────────────

_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_canonical_example_file_exists():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    assert example.exists(), f"Expected example at {example}"


def test_canonical_example_parses_successfully():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    assert prog.project == "TextRoutingPipeline"


def test_canonical_example_has_two_imports():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    assert len(prog.imports) == 2


def test_canonical_example_import_modes():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    frozen = next(i for i in prog.imports if i.registry_name == "text_encoder")
    trainable = next(i for i in prog.imports if i.registry_name == "sentiment_classifier")
    assert frozen.mode == "FROZEN"
    assert trainable.mode == "TRAINABLE"


def test_canonical_example_import_aliases():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    aliases = {i.alias for i in prog.imports}
    assert "TextEncoder" in aliases
    assert "SentimentClassifier" in aliases


def test_canonical_example_has_router_network():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    assert any(n.name == "Router" for n in prog.networks)


def test_canonical_example_graph_has_edges():
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    assert len(prog.graph.edges) > 0


def test_canonical_example_composite_hash(tmp_path):
    """composite_model_hash can be computed once the imports are in the registry."""
    example = _EXAMPLES_DIR / "text-routing-pipeline.mxai"
    prog = parse_text(example.read_text())
    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("text_encoder", model_hash="sha256:" + "aa" * 32))
    reg.push(_make_entry("sentiment_classifier", model_hash="sha256:" + "bb" * 32))
    composite_hash = compute_composite_model_hash(prog, reg)
    assert composite_hash.startswith("sha256:")
    assert len(composite_hash) == len("sha256:") + 64
