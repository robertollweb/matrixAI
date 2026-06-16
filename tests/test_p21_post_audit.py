"""P21 post-audit fixes: composite_trainer hash, node types, push_run_dir, CLI typecheck."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from matrixai.ir.schema import GraphSpec, ImportSpec, MatrixAIProgram
from matrixai.parser.parser import parse_text
from matrixai.registry import ModelRegistry, RegistryEntry, compute_entry_hash
from matrixai.registry.composite_hash import compute_composite_model_hash

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
        output_type={"kind": "Tensor", "shape": [128]},
        metrics={},
        matrixai_version=kw["matrixai_version"],
        created_at="2026-06-01T12:00:00+00:00",
        training_dataset_fingerprint="",
        interpretability_level="full",
        training_trace_hash=kw["training_trace_hash"],
        evaluation_report_hash=kw["evaluation_report_hash"],
    )


def _make_run_dir(tmp_path: Path, *, with_model: bool = True,
                  with_params: bool = True, with_trace: bool = True) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.91}))
    if with_model:
        (run_dir / "model.mxai").write_text("PROJECT Dummy\n")
    if with_params:
        (run_dir / "params.json").write_text(json.dumps({
            "parameter_set_id": "ps_test_001",
            "model_hash": "sha256:" + "a" * 64,
            "parameter_schema_hash": "sha256:" + "b" * 64,
            "parameters": {},
            "metrics": {"accuracy": 0.91},
        }))
    if with_trace:
        (run_dir / "training_trace.json").write_text(json.dumps({"epochs": 10}))
    return run_dir


# ── Fix 1: composite_trainer uses canonical hash ──────────────────────────────

def test_composite_forward_hash_matches_canonical(tmp_path):
    """composite_forward.composite_model_hash must equal compute_composite_model_hash."""
    from matrixai.training.composite_trainer import composite_forward
    from matrixai.parameters import ParameterSet

    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("enc"))

    program = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="FROZEN")],
        graph=GraphSpec(nodes=["Input", "Enc"], edges=[("Input", "Enc")]),
    )
    ps = ParameterSet(
        parameter_set_id="ps_001",
        model_hash="sha256:" + "a" * 64,
        parameter_schema_hash="sha256:" + "b" * 64,
        parameters={},
    )
    result = composite_forward(program, ps, {"x": 1.0}, reg)
    canonical = compute_composite_model_hash(program, reg)
    assert result.composite_model_hash == canonical


def test_composite_forward_hash_changes_with_local_graph(tmp_path):
    """composite_forward reflects local GRAPH changes in its composite_model_hash."""
    from matrixai.training.composite_trainer import composite_forward
    from matrixai.parameters import ParameterSet

    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("enc"))
    imp = ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="FROZEN")

    ps = ParameterSet(
        parameter_set_id="ps_001",
        model_hash="sha256:" + "a" * 64,
        parameter_schema_hash="sha256:" + "b" * 64,
        parameters={},
    )
    prog_a = MatrixAIProgram(
        project="Test", imports=[imp],
        graph=GraphSpec(nodes=["Input", "Enc"], edges=[("Input", "Enc")]),
    )
    prog_b = MatrixAIProgram(
        project="Test2", imports=[imp],
        graph=GraphSpec(nodes=["Input", "Enc"], edges=[("Input", "Enc")]),
    )
    h_a = composite_forward(prog_a, ps, {}, reg).composite_model_hash
    h_b = composite_forward(prog_b, ps, {}, reg).composite_model_hash
    assert h_a != h_b


# ── Fix 2: _attach_node_types marks import aliases as composite_model ─────────

def test_parser_marks_import_alias_as_composite_model():
    src = """
PROJECT Test

IMPORT Enc FROM registry text_encoder@v1 FROZEN

VECTOR Input[1]
  x
END

GRAPH
  Input -> Enc
END
"""
    prog = parse_text(src)
    assert prog.graph.node_types.get("Enc") == "composite_model"


def test_parser_marks_multiple_import_aliases():
    src = """
PROJECT Test

IMPORT Enc FROM registry text_encoder@v1 FROZEN
IMPORT Sent FROM registry sentiment@v1 TRAINABLE

GRAPH
  Enc -> Sent
END
"""
    prog = parse_text(src)
    assert prog.graph.node_types.get("Enc") == "composite_model"
    assert prog.graph.node_types.get("Sent") == "composite_model"


def test_parser_non_import_nodes_unaffected():
    src = """
PROJECT Test

IMPORT Enc FROM registry text_encoder@v1 FROZEN

VECTOR Input[1]
  x
END

GRAPH
  Input -> Enc
END
"""
    prog = parse_text(src)
    assert prog.graph.node_types.get("Input") == "vector"


# ── Fix 4: ModelRegistry.push_run_dir ─────────────────────────────────────────

def test_push_run_dir_succeeds_with_all_artifacts(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reg = ModelRegistry(tmp_path / "reg")
    entry = reg.push_run_dir(run_dir, "my_model", "v1")
    assert entry.name == "my_model"
    assert entry.version == "v1"
    assert entry.entry_hash.startswith("sha256:")


def test_push_run_dir_copies_artifacts_to_registry(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reg = ModelRegistry(tmp_path / "reg")
    reg.push_run_dir(run_dir, "my_model", "v1")
    entry_dir = reg.layout.entry_dir("my_model", "v1")
    assert (entry_dir / "evaluation_report.json").exists()
    assert (entry_dir / "model.mxai").exists()
    assert (entry_dir / "params.json").exists()
    assert (entry_dir / "training_trace.json").exists()


def test_push_run_dir_fails_without_evaluation_report(tmp_path):
    from matrixai.registry.model_registry import ModelRegistryError
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reg = ModelRegistry(tmp_path / "reg")
    with pytest.raises(ModelRegistryError, match="evaluation_report"):
        reg.push_run_dir(run_dir, "my_model", "v1")


def test_push_run_dir_reads_parameter_set_id_from_params(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reg = ModelRegistry(tmp_path / "reg")
    entry = reg.push_run_dir(run_dir, "my_model", "v1")
    assert entry.parameter_set_id == "ps_test_001"


def test_push_run_dir_entry_is_retrievable(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reg = ModelRegistry(tmp_path / "reg")
    pushed = reg.push_run_dir(run_dir, "my_model", "v1")
    retrieved = reg.get("my_model", "v1")
    assert retrieved.entry_hash == pushed.entry_hash


def test_push_run_dir_works_without_optional_artifacts(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_model=False, with_params=False, with_trace=False)
    reg = ModelRegistry(tmp_path / "reg")
    entry = reg.push_run_dir(run_dir, "minimal", "v1")
    assert entry.name == "minimal"


def test_push_run_dir_is_append_only(tmp_path):
    from matrixai.registry.model_registry import DuplicateEntryError
    run_dir = _make_run_dir(tmp_path)
    reg = ModelRegistry(tmp_path / "reg")
    reg.push_run_dir(run_dir, "my_model", "v1")
    with pytest.raises(DuplicateEntryError):
        reg.push_run_dir(run_dir, "my_model", "v1")


def test_push_run_dir_prefers_params_best_json(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "params.best.json").write_text(json.dumps({
        "parameter_set_id": "ps_best_001",
        "model_hash": "sha256:" + "a" * 64,
        "parameter_schema_hash": "sha256:" + "b" * 64,
        "parameters": {},
        "metrics": {},
    }))
    reg = ModelRegistry(tmp_path / "reg")
    entry = reg.push_run_dir(run_dir, "my_model", "v1")
    assert entry.parameter_set_id == "ps_best_001"


# ── Fix 4: CLI registry push real ────────────────────────────────────────────

def test_cli_registry_push_registers_entry(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reg_path = str(tmp_path / "reg")
    result = subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "registry", "push",
         str(run_dir), "--name", "cli_model", "--version", "v1",
         "--registry-path", reg_path],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "cli_model" in result.stdout

    # Entry is retrievable after push
    reg = ModelRegistry(reg_path)
    entry = reg.get("cli_model", "v1")
    assert entry.name == "cli_model"


def test_cli_registry_push_fails_without_evaluation_report(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "registry", "push",
         str(run_dir), "--name", "x", "--version", "v1",
         "--registry-path", str(tmp_path / "reg")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


# ── Fix 3: CLI typecheck registry-aware ──────────────────────────────────────

def test_cli_typecheck_accepts_registry_path_flag(tmp_path):
    """--registry-path flag on typecheck must be accepted without error."""
    mxai_path = tmp_path / "test.mxai"
    mxai_path.write_text("PROJECT Simple\n\nVECTOR V[1]\n  x\nEND\n\nGRAPH\n  V -> V\nEND\n")
    result = subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "typecheck",
         str(mxai_path), "--registry-path", str(tmp_path / "reg")],
        capture_output=True, text=True,
    )
    # Should not fail just because the flag is present (no imports → no registry lookup)
    assert result.returncode in (0, 1)
    assert "unrecognized" not in result.stderr.lower()


# ── Fix A: check_composite_program_types rejects mutable tags ─────────────────

def test_composite_typecheck_rejects_mutable_tag_by_default(tmp_path):
    """@latest import rejected without allow_mutable_tags=True."""
    from matrixai.types import check_composite_program_types, TypeCompatibilityError

    reg = ModelRegistry(tmp_path / "reg")
    prog = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="latest", mode="FROZEN")],
        graph=GraphSpec(nodes=[], edges=[]),
    )
    with pytest.raises(TypeCompatibilityError) as exc_info:
        check_composite_program_types(prog, reg)
    # The per-import error is in exc.value.errors, not in the top-level message.
    assert any("mutable" in e for e in exc_info.value.errors)


def test_composite_typecheck_accepts_mutable_tag_with_flag(tmp_path):
    """@latest import accepted when allow_mutable_tags=True (registry lookup may fail but not due to policy)."""
    from matrixai.registry.model_registry import EntryNotFoundError
    from matrixai.types import check_composite_program_types

    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("enc"))
    reg.tag("enc", "v1", "latest")

    prog = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="latest", mode="FROZEN")],
        graph=GraphSpec(nodes=[], edges=[]),
    )
    # Should not raise due to policy; may warn about mutable tag
    result = check_composite_program_types(prog, reg, allow_mutable_tags=True)
    assert result.ok


def test_composite_typecheck_pinned_version_not_affected(tmp_path):
    """Pinned version (v1) passes through without policy rejection."""
    from matrixai.types import check_composite_program_types

    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("enc"))
    prog = MatrixAIProgram(
        project="Test",
        imports=[ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="FROZEN")],
        graph=GraphSpec(nodes=[], edges=[]),
    )
    result = check_composite_program_types(prog, reg)
    assert result.ok


def test_cli_typecheck_mutable_import_rejected_without_flag(tmp_path):
    """CLI typecheck with @latest import fails unless --allow-mutable-imports passed."""
    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("enc"))
    reg.tag("enc", "v1", "latest")

    mxai = tmp_path / "prog.mxai"
    mxai.write_text(
        "PROJECT T\n\nIMPORT Enc FROM registry enc@latest FROZEN\n\nGRAPH\n  Enc -> Enc\nEND\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "typecheck", str(mxai),
         "--registry-path", str(tmp_path / "reg")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_cli_typecheck_mutable_import_accepted_with_flag(tmp_path):
    """CLI typecheck with @latest import succeeds when --allow-mutable-imports is passed."""
    reg = ModelRegistry(tmp_path / "reg")
    reg.push(_make_entry("enc"))
    reg.tag("enc", "v1", "latest")

    mxai = tmp_path / "prog.mxai"
    # Graph has no cross-component edges, so no type mismatch — only tag policy matters.
    mxai.write_text(
        "PROJECT T\n\nIMPORT Enc FROM registry enc@latest FROZEN\n\nGRAPH\nEND\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "typecheck", str(mxai),
         "--registry-path", str(tmp_path / "reg"),
         "--allow-mutable-imports"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# ── Fix B: pull copies all artifacts ──────────────────────────────────────────

def test_pull_copies_all_artifacts(tmp_path):
    """pull() transfers model.mxai, params.json, training_trace.json, evaluation_report.json."""
    run_dir = _make_run_dir(tmp_path)
    src = ModelRegistry(tmp_path / "src")
    dst = ModelRegistry(tmp_path / "dst")
    src.push_run_dir(run_dir, "mymodel", "v1")
    src.pull("mymodel", "v1", dst)

    entry_dir = dst.layout.entry_dir("mymodel", "v1")
    assert (entry_dir / "evaluation_report.json").exists()
    assert (entry_dir / "model.mxai").exists()
    assert (entry_dir / "params.json").exists()
    assert (entry_dir / "training_trace.json").exists()


def test_pull_preserves_artifact_content(tmp_path):
    """Pulled artifacts have the same content as the originals."""
    run_dir = _make_run_dir(tmp_path)
    src = ModelRegistry(tmp_path / "src")
    dst = ModelRegistry(tmp_path / "dst")
    src.push_run_dir(run_dir, "mymodel", "v1")
    src.pull("mymodel", "v1", dst)

    src_eval = (src.layout.entry_dir("mymodel", "v1") / "evaluation_report.json").read_text()
    dst_eval = (dst.layout.entry_dir("mymodel", "v1") / "evaluation_report.json").read_text()
    assert src_eval == dst_eval


def test_pull_entry_is_verifiable_in_target(tmp_path):
    """Pulled entry verifies correctly in the target registry."""
    run_dir = _make_run_dir(tmp_path)
    src = ModelRegistry(tmp_path / "src")
    dst = ModelRegistry(tmp_path / "dst")
    src.push_run_dir(run_dir, "mymodel", "v1")
    src.pull("mymodel", "v1", dst)
    assert dst.verify("mymodel", "v1") is True
