"""P21 C8 — Composite forward pass and training scaffold."""
from __future__ import annotations

import json

import pytest

from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram, VectorSpec
from matrixai.parameters import ParameterSet
from matrixai.registry import ModelRegistry, RegistryEntry, compute_entry_hash
from matrixai.training.composite_trainer import (
    CompositeForwardResult,
    composite_forward,
    composite_training_step,
    validate_composite_trainability,
)

# ── real-artifact helpers ─────────────────────────────────────────────────────

_SCORER_MODEL = """\
PROJECT Scorer
VECTOR Input[1]
  x: Score
END
FUNCTION Score
  R: Risk = sigmoid(W1 * Input + b1)
END
GRAPH
  Input -> Score
END
"""

_EVAL_JSON = json.dumps({"accuracy": 0.9, "loss": 0.1})


def _write_run_dir(tmp: "Path", model_src: str = _SCORER_MODEL, w: float = 1.0) -> "Path":
    """Write minimal run-dir artifacts suitable for push_run_dir."""
    from pathlib import Path
    run_dir = tmp / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "model_snapshot.mxai").write_text(model_src)
    ps = ParameterSet(
        parameter_set_id="t1",
        model_hash="sha256:" + "a" * 64,
        parameter_schema_hash="sha256:" + "b" * 64,
        parameters={
            "Score.W1": {"values": [w], "shape": [1]},
            "Score.b1": {"values": [0.0], "shape": [1]},
        },
    )
    (run_dir / "params.best.json").write_text(json.dumps(ps.to_dict()))
    (run_dir / "evaluation_report.json").write_text(_EVAL_JSON)
    return run_dir


def _prog_with_frozen(frozen_alias: str, registry_name: str = "scorer") -> MatrixAIProgram:
    """Composite program: Input → <frozen> → (no trainable)."""
    return MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias=frozen_alias, registry_name=registry_name, version="v1", mode="FROZEN")],
        graph=GraphSpec(nodes=["Input", frozen_alias], edges=[("Input", frozen_alias)]),
        vectors=[VectorSpec(name="Input", size=1, fields=["x"])],
    )

# ── helpers ───────────────────────────────────────────────────────────────────

_BASE = dict(
    model_hash="sha256:" + "a" * 64,
    parameter_schema_hash="sha256:" + "b" * 64,
    parameter_set_id="ps_001",
    training_trace_hash="sha256:" + "c" * 64,
    evaluation_report_hash="sha256:" + "d" * 64,
    matrixai_version="0.21.0",
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
        input_type={"kind": "VECTOR"},
        output_type={"kind": "Tensor", "shape": [4]},
        metrics={},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


def _ps(*, extra_params: dict | None = None) -> ParameterSet:
    params = {"Router.W1": {"values": [1.0, 2.0], "shape": [2]}}
    if extra_params:
        params.update(extra_params)
    return ParameterSet(
        parameter_set_id="ps_test",
        model_hash="m_hash",
        parameter_schema_hash="s_hash",
        parameters=params,
    )


def _prog(frozen_alias: str, trainable_alias: str | None = None) -> MatrixAIProgram:
    imports = [ImportSpec(alias=frozen_alias, registry_name="enc", version="v1", mode="FROZEN")]
    if trainable_alias:
        imports.append(ImportSpec(alias=trainable_alias, registry_name="router", version="v1", mode="TRAINABLE"))
    nodes = ["Input", frozen_alias] + ([trainable_alias] if trainable_alias else [])
    edges = [("Input", frozen_alias)]
    if trainable_alias:
        edges.append((frozen_alias, trainable_alias))
    return MatrixAIProgram(
        project="Test",
        imports=imports,
        graph=GraphSpec(nodes=nodes, edges=edges),
        vectors=[VectorSpec(name="Input", size=1, fields=["x"])],
    )


# ── composite forward ─────────────────────────────────────────────────────────

def test_composite_forward_loads_frozen_params_from_registry(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog("Enc")
    ps = _ps()
    result = composite_forward(program, ps, input_data=[1.0], registry=reg)
    assert "Enc" in result.frozen_params_loaded
    assert result.frozen_params_loaded["Enc"]["registry_name"] == "enc"


def test_composite_forward_combines_frozen_and_trainable(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _prog("Enc", "Router")
    ps = _ps()
    result = composite_forward(program, ps, input_data=[1.0], registry=reg)
    assert "Enc" in result.frozen_params_loaded
    assert "Router.W1" in result.trainable_params_used


def test_composite_forward_returns_result_type(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog("Enc")
    ps = _ps()
    result = composite_forward(program, ps, [1.0], registry=reg)
    assert isinstance(result, CompositeForwardResult)


def test_composite_forward_includes_composite_model_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog("Enc")
    result = composite_forward(program, _ps(), [1.0], registry=reg)
    assert result.composite_model_hash.startswith("sha256:")


# ── training ──────────────────────────────────────────────────────────────────

def test_training_updates_only_trainable_components(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _prog("Enc", "Router")
    original_ps = _ps()
    new_ps, step = composite_training_step(program, original_ps, [1.0], [0.0], reg)
    assert "Router.W1" in step.gradient_keys
    assert "Enc" in step.frozen_unchanged
    assert "Enc" not in step.gradient_keys


def test_training_does_not_modify_registry_entries(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry("enc")
    reg.push(entry)
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _prog("Enc", "Router")
    ps = _ps()
    composite_training_step(program, ps, [1.0], [0.0], reg)
    # Registry entry unchanged
    fetched = reg.get("enc", "v1")
    assert fetched.entry_hash == entry.entry_hash


def test_training_gradient_flows_only_through_trainable(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    reg.push(_make_entry("router", model_hash="sha256:" + "f" * 64))
    program = _prog("Enc", "Router")
    ps = _ps()
    new_ps, step = composite_training_step(program, ps, [1.0], [0.0], reg)
    # Only trainable params updated
    for key in step.gradient_keys:
        assert key.split(".")[0] != "Enc"  # frozen, no gradient


def test_training_with_all_frozen_components_fails_validation(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))
    program = _prog("Enc")  # only frozen import, no trainable params in ps
    ps = ParameterSet(
        parameter_set_id="ps",
        model_hash="h",
        parameter_schema_hash="s",
        parameters={},  # empty - no trainable params
    )
    with pytest.raises(ValueError, match="trainable"):
        composite_training_step(program, ps, [1.0], [0.0], reg)


def test_composite_evaluation_uses_composite_model_hash(tmp_path):
    reg = ModelRegistry(tmp_path)
    entry = _make_entry("enc")
    reg.push(entry)
    program = _prog("Enc")
    result = composite_forward(program, _ps(), [1.0], registry=reg)
    # The composite hash must include the entry_hash of "enc"
    assert result.composite_model_hash != ""
    assert result.composite_model_hash.startswith("sha256:")


# ── validate_composite_trainability ──────────────────────────────────────────

def test_validate_composite_trainability_true_when_trainable_params(tmp_path):
    program = _prog("Enc", "Router")
    ps = _ps()  # has Router.W1
    assert validate_composite_trainability(program, ps) is True


def test_validate_composite_trainability_false_when_all_frozen(tmp_path):
    program = _prog("Enc")  # only frozen
    ps = ParameterSet(
        parameter_set_id="p", model_hash="h", parameter_schema_hash="s", parameters={}
    )
    assert validate_composite_trainability(program, ps) is False


# ── real frozen-execute tests ─────────────────────────────────────────────────

def test_frozen_execute_produces_non_identity_output(tmp_path):
    """Frozen component must execute real model, not return input unchanged."""
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _write_run_dir(tmp_path / "scorer")
    reg.push_run_dir(run_dir, name="scorer", version="v1")

    program = _prog_with_frozen("Scorer")
    ps = ParameterSet(
        parameter_set_id="ps", model_hash="h", parameter_schema_hash="s", parameters={}
    )
    result = composite_forward(program, ps, input_data=[0.5], registry=reg)

    scorer_out = result.outputs.get("Scorer")
    # sigmoid(1.0 * 0.5 + 0.0) ≈ 0.622 — definitely not [0.5]
    assert scorer_out is not None
    assert scorer_out != [0.5], "passthrough returned — frozen component did not execute"
    assert isinstance(scorer_out, float)
    assert 0.0 < scorer_out < 1.0


def test_frozen_execute_output_varies_with_weight(tmp_path):
    """Different weights → different outputs from frozen component."""
    reg1 = ModelRegistry(tmp_path / "reg1")
    reg2 = ModelRegistry(tmp_path / "reg2")
    run_dir1 = _write_run_dir(tmp_path / "r1", w=0.1)
    run_dir2 = _write_run_dir(tmp_path / "r2", w=5.0)
    reg1.push_run_dir(run_dir1, name="scorer", version="v1")
    reg2.push_run_dir(run_dir2, name="scorer", version="v1")

    ps = ParameterSet(parameter_set_id="p", model_hash="h", parameter_schema_hash="s", parameters={})
    program = _prog_with_frozen("Scorer")
    out1 = composite_forward(program, ps, input_data=[0.8], registry=reg1).outputs["Scorer"]
    out2 = composite_forward(program, ps, input_data=[0.8], registry=reg2).outputs["Scorer"]

    assert out1 != out2, "weight change did not affect frozen component output"


def test_frozen_execute_falls_back_to_identity_when_no_artifacts(tmp_path):
    """Mock registry entry (no files) must gracefully fall back to passthrough."""
    reg = ModelRegistry(tmp_path)
    reg.push(_make_entry("enc"))   # metadata-only, no model.mxai / params.json

    program = _prog("Enc")
    ps = _ps()
    result = composite_forward(program, ps, input_data=[0.7], registry=reg)

    # Fallback: output equals input
    assert result.outputs.get("Enc") == [0.7]


def test_frozen_execute_scalar_input_mapped_correctly(tmp_path):
    """Scalar input_val is wrapped into the frozen model's vector field."""
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _write_run_dir(tmp_path / "s")
    reg.push_run_dir(run_dir, name="scorer", version="v1")

    program = _prog_with_frozen("Scorer")
    ps = ParameterSet(parameter_set_id="p", model_hash="h", parameter_schema_hash="s", parameters={})
    # Pass a bare scalar — _map_input_to_vector should wrap it
    result = composite_forward(program, ps, input_data=0.5, registry=reg)
    out = result.outputs.get("Scorer")
    assert isinstance(out, float)
    assert 0.0 < out < 1.0


def test_frozen_execute_dict_input_passed_directly(tmp_path):
    """Dict input_val is forwarded as-is to MatrixAIRuntime."""
    reg = ModelRegistry(tmp_path / "reg")
    run_dir = _write_run_dir(tmp_path / "s")
    reg.push_run_dir(run_dir, name="scorer", version="v1")

    program = _prog_with_frozen("Scorer")
    ps = ParameterSet(parameter_set_id="p", model_hash="h", parameter_schema_hash="s", parameters={})
    result = composite_forward(program, ps, input_data={"x": 0.5}, registry=reg)
    out = result.outputs.get("Scorer")
    assert isinstance(out, float)
    assert 0.0 < out < 1.0
