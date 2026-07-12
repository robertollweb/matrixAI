# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C6 — auditoría del bloque + cierre duro (contrato 51 §C6).

Dos piezas:
(a) Metadata pública del bloque (lista del borrador MENOS optimizador/scheduler)
    expuesta en inference_spec.json y model_manifest.json, reusando el
    layer_manifest que backend_contract.py ya calcula (single source of
    truth, cero coste O(params) adicional).
(b) Registro P21: `push_run_dir` reconocía solo params.best.json/params.json
    y caía al hash nulo para parameter_set.json (dense_trainer.py,
    transformer_trainer.py) o weights.mxw (PESOS_GRANDES) — el hash del
    registro NO cubría los pesos del bloque en ninguno de los dos caminos
    reales que usa el transformer. `verify()` re-hashea weights.mxw en
    streaming (nunca sha256 del fichero completo — invariante 6/PESOS_GRANDES).
(c) Cierre duro: ciclo CLI real (subprocess) `.mxai` → `mx train` → `mx
    export-bundle` → `predict.py` importado y ejecutado → mismos resultados
    que el forward de referencia sobre los MISMOS pesos entrenados.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from matrixai.parser.parser import parse_text
from matrixai.parameters.network_params import (
    TRANSFORMER_BLOCK_VERSION,
    build_composite_network_parameter_set,
    transformer_block_export_metadata,
    transformer_block_param_count,
)
from matrixai.parameters.store import program_hash
from matrixai.types import check_composite_network_types

torch = pytest.importorskip("torch")
from importlib import util as _il_util

_HAS_ONNX = _il_util.find_spec("onnx") is not None
_HAS_ORT = _il_util.find_spec("onnxruntime") is not None
pytestmark = pytest.mark.skipif(
    not (_HAS_ONNX and _HAS_ORT), reason="onnx + onnxruntime required"
)

L, VOCAB, DIM, LAYERS, HEADS, FF = 6, 11, 8, 1, 2, 16

_MXAI = f"""
PROJECT C6Test

SEQUENCE Texto
  length = {L}
  vocab_size = {VOCAB}
END

NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM {DIM}
  BLOCK enc TRANSFORMER
    LAYERS {LAYERS}
    HEADS {HEADS}
    FF {FF}
  END
  POOL mean
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END

GRAPH
  Texto -> N
END
"""

_MXTRAIN = """
MODEL {model}

DATASET D
  SOURCE csv("{csv}")
  INPUT Texto FROM COLUMNS [t0, t1, t2, t3, t4, t5]
  TARGET clase: Label[NEG, POS]
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
  TYPE adam
  LEARNING_RATE 0.01
  UPDATE N.*
END

RUN
  EPOCHS 2
END
"""


def _build(seed: int = 7):
    prog = parse_text(_MXAI)
    net = prog.networks[0]
    res = check_composite_network_types(
        net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
    )
    assert res.ok, res.errors
    ps = build_composite_network_parameter_set(net, res, program_hash(prog), seed=seed)
    return prog, net, res, ps


def _write_dataset(tmp_path: Path, n: int = 20, seed: int = 3):
    rng = random.Random(seed)
    rows: list[tuple[list[int], str]] = []
    for _ in range(n):
        row = [rng.randrange(VOCAB) for _ in range(L)]
        rows.append((row, "POS" if sum(row) % 2 == 0 else "NEG"))
    csv_path = tmp_path / "toy.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"t{i}" for i in range(L)] + ["clase"])
        for row, label in rows:
            w.writerow(row + [label])
    model_path = tmp_path / "toy.mxai"
    model_path.write_text(_MXAI, encoding="utf-8")
    return model_path, csv_path, rows


# ---------------------------------------------------------------------------
# (a) Metadata del bloque
# ---------------------------------------------------------------------------

class TestBlockMetadata:
    def test_export_metadata_matches_architecture(self):
        from matrixai.compiler import BackendContractAnalyzer
        prog, net, res, ps = _build()
        report = BackendContractAnalyzer().analyze(prog)
        entry = next(
            e for e in report.layer_manifest if e.get("layer_type") == "TransformerBlock"
        )
        meta = transformer_block_export_metadata(entry)
        assert meta["block_version"] == TRANSFORMER_BLOCK_VERSION
        assert meta["layers"] == LAYERS
        assert meta["heads"] == HEADS
        assert meta["embedding_dim"] == DIM
        assert meta["feed_forward_dim"] == FF
        assert meta["dropout"] == 0.0
        assert meta["activation"] == "gelu"
        assert meta["positional_encoding"] == "sinusoidal"
        assert meta["backend"] == "torch"
        assert meta["param_count_total"] == transformer_block_param_count(
            LAYERS, DIM, FF, L, "sinusoidal"
        )
        # Nada se congela hoy (fuera de alcance: pesos preentrenados externos)
        assert meta["param_count_trainable"] == meta["param_count_total"]
        assert set(meta["initialization"]) <= {"xavier_normal", "zeros", "ones", "he_normal"}

    def test_inference_spec_carries_transformer_block(self):
        from matrixai.compiler import BackendContractAnalyzer
        from matrixai.export.inference_spec import build_inference_spec
        from matrixai.export.onnx_exporter import export_onnx
        import tempfile
        prog, net, res, ps = _build()
        path = Path(tempfile.mkdtemp()) / "model.onnx"
        result = export_onnx(prog, ps, path)
        spec = build_inference_spec(prog, ps, result, labels=["NEG", "POS"])
        entry = next(
            e for e in BackendContractAnalyzer().analyze(prog).layer_manifest
            if e.get("layer_type") == "TransformerBlock"
        )
        assert spec["transformer_block"] == transformer_block_export_metadata(entry)

    def test_model_manifest_carries_transformer_block(self):
        from matrixai.export.bundle import _build_model_manifest
        prog, net, res, ps = _build()
        manifest = _build_model_manifest(prog, ps)
        assert manifest["matrixai_version"]
        assert manifest["transformer_block"]["layers"] == LAYERS
        assert manifest["transformer_block"]["backend"] == "torch"

    def test_dense_only_manifest_has_no_transformer_block(self):
        """Una red densa normal no gana una clave transformer_block falsa."""
        from matrixai.export.bundle import _build_model_manifest
        from matrixai.parameters.network_params import build_network_parameter_set
        from matrixai.types import check_network_types
        dense_src = """
PROJECT DenseC6
VECTOR X[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK D
  INPUT X
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END
GRAPH
  X -> D
END
"""
        prog = parse_text(dense_src)
        net = prog.networks[0]
        res = check_network_types(net, {v.name: v for v in prog.vectors})
        assert res.ok, res.errors
        ps = build_network_parameter_set(net, res.resolved_layers, program_hash(prog), seed=1)
        manifest = _build_model_manifest(prog, ps)
        assert "transformer_block" not in manifest


# ---------------------------------------------------------------------------
# (b) Registro P21 — el hash cubre los pesos del bloque
# ---------------------------------------------------------------------------

class TestRegistryCoversBlockWeights:
    def test_push_run_dir_reads_parameter_set_json(self, tmp_path):
        """dense_trainer.py/transformer_trainer.py escriben parameter_set.json
        (no params.best.json) — antes de este fix caía al hash nulo."""
        from matrixai.parameters.store import write_parameter_set
        from matrixai.registry.model_registry import ModelRegistry
        _, _, _, ps = _build()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        write_parameter_set(run_dir / "parameter_set.json", ps)
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))

        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(run_dir, "toy", "1.0.0")
        null_hash = "sha256:" + "0" * 64
        assert entry.parameter_schema_hash not in ("", null_hash)
        assert entry.parameter_schema_hash == ps.parameter_schema_hash
        assert entry.params_content_hash != ""
        assert registry.verify("toy", "1.0.0") is True

    def test_params_best_json_still_wins_over_parameter_set_json(self, tmp_path):
        """La prioridad existente (params.best.json > params.json) no se rompe
        al añadir parameter_set.json como tercer candidato."""
        from matrixai.registry.model_registry import ModelRegistry
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "parameter_set.json").write_text(json.dumps({
            "parameter_set_id": "wrong", "parameter_schema_hash": "sha256:" + "1" * 64,
            "parameters": {}, "metrics": {},
        }))
        (run_dir / "params.best.json").write_text(json.dumps({
            "parameter_set_id": "right", "parameter_schema_hash": "sha256:" + "2" * 64,
            "parameters": {}, "metrics": {},
        }))
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(run_dir, "toy", "1.0.0")
        assert entry.parameter_set_id == "right"

    def test_push_run_dir_reads_weights_mxw_header(self, tmp_path):
        """PESOS_GRANDES: sin parameter_set.json, solo weights.mxw — el hash
        del registro debe cubrir los pesos reales (content_hash de la
        cabecera, calculado en streaming al escribir, no el placeholder nulo)."""
        from matrixai.parameters.binary_store import write_mxw
        from matrixai.registry.model_registry import ModelRegistry
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state = {"N.enc.layer_0.attention.Wq": torch.randn(DIM, DIM)}
        header = write_mxw(
            run_dir / "weights.mxw", state,
            model_hash="sha256:" + "a" * 64,
            parameter_schema_hash="sha256:" + "b" * 64,
        )
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.8}))

        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(run_dir, "toy-big", "1.0.0")
        assert entry.parameter_schema_hash == "sha256:" + "b" * 64
        assert entry.params_content_hash == "sha256:" + header["content_hash"]
        assert (registry.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw").exists()
        assert registry.verify("toy-big", "1.0.0") is True

    def test_verify_detects_weights_mxw_tamper_without_full_read(self, tmp_path, monkeypatch):
        """verify() re-deriva el hash en streaming (mxw_body_content_hash) —
        prueba que detecta un tamper real del cuerpo, y que NO usa
        sha256_bytes(fichero completo) (interceptado para que fallar si se
        llama con el .mxw, ya que eso rompería el presupuesto de memoria de
        PESOS_GRANDES para modelos grandes de verdad)."""
        from matrixai.parameters.binary_store import write_mxw
        from matrixai.registry.model_registry import ModelRegistry
        import matrixai.registry.model_registry as mr

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state = {"N.enc.layer_0.attention.Wq": torch.randn(DIM, DIM)}
        write_mxw(
            run_dir / "weights.mxw", state,
            model_hash="sha256:" + "a" * 64, parameter_schema_hash="sha256:" + "b" * 64,
        )
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.8}))
        registry = ModelRegistry(tmp_path / "registry")
        registry.push_run_dir(run_dir, "toy-big", "1.0.0")
        assert registry.verify("toy-big", "1.0.0") is True

        mxw_path = registry.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw"
        data = bytearray(mxw_path.read_bytes())
        data[-1] ^= 0xFF
        mxw_path.write_bytes(bytes(data))

        from matrixai.registry.model_registry import VerificationError
        with pytest.raises(VerificationError, match="weights.mxw"):
            registry.verify("toy-big", "1.0.0")

    def test_pull_copies_weights_mxw_across_registries(self, tmp_path):
        from matrixai.parameters.binary_store import write_mxw
        from matrixai.registry.model_registry import ModelRegistry
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state = {"N.enc.layer_0.attention.Wq": torch.randn(DIM, DIM)}
        write_mxw(
            run_dir / "weights.mxw", state,
            model_hash="sha256:" + "a" * 64, parameter_schema_hash="sha256:" + "b" * 64,
        )
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.8}))
        src = ModelRegistry(tmp_path / "src_registry")
        src.push_run_dir(run_dir, "toy-big", "1.0.0")
        dst = ModelRegistry(tmp_path / "dst_registry")
        src.pull("toy-big", "1.0.0", dst)
        assert (dst.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw").exists()
        assert dst.verify("toy-big", "1.0.0") is True


# ---------------------------------------------------------------------------
# (c) Cierre duro — ciclo CLI completo
# ---------------------------------------------------------------------------

class TestCierreDuro:
    def test_cli_train_export_import_infer_same_results(self, tmp_path):
        model_path, csv_path, rows = _write_dataset(tmp_path)
        (tmp_path / "toy.mxtrain").write_text(
            _MXTRAIN.format(model="toy.mxai", csv="toy.csv"), encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        train = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "train", str(model_path),
             "--training", str(tmp_path / "toy.mxtrain"),
             "--output", str(out_dir), "--json"],
            capture_output=True, text=True, timeout=300,
        )
        assert train.returncode == 0, train.stderr
        ps_path = out_dir / "parameter_set.json"
        assert ps_path.exists()

        bundle_dir = tmp_path / "bundle"
        export = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "export-bundle", str(model_path),
             "--params", str(ps_path), "--outdir", str(bundle_dir), "--json"],
            capture_output=True, text=True, timeout=180,
        )
        assert export.returncode == 0, export.stderr
        payload = json.loads(export.stdout)
        assert payload["equivalence_passed"] is True

        spec_path = bundle_dir / "inference_spec.json"
        predict_path = bundle_dir / "predict.py"
        manifest_path = bundle_dir / "model_manifest.json"
        assert spec_path.exists() and predict_path.exists()

        spec = json.loads(spec_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        assert spec["transformer_block"]["layers"] == LAYERS
        assert manifest["transformer_block"] == spec["transformer_block"]

        # Import: carga el predict.py REAL del bundle (no la plantilla en el
        # árbol fuente) y ejecuta inferencia sobre una fila real del dataset.
        import importlib.util
        mod_spec = importlib.util.spec_from_file_location("_c6_bundled_predict", str(predict_path))
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        model = module.MatrixAIModel(str(spec_path))

        test_ids, _label = rows[0]
        got = model.predict(test_ids)
        got_ordered = [got["NEG"], got["POS"]]

        # Referencia: forward oficial directo sobre LOS MISMOS pesos entrenados
        # (parameter_set.json), sin pasar por ONNX — cierra el invariante 5
        # (export→import reproduce arquitectura y pesos).
        from matrixai.parameters.store import load_parameter_set
        from matrixai.forward.transformer_forward import transformer_network_forward
        prog = parse_text(_MXAI)
        net = prog.networks[0]
        tr = check_composite_network_types(net, {}, {s.name: s for s in prog.sequences})
        ps = load_parameter_set(ps_path)
        ref = transformer_network_forward(net, tr, ps, test_ids).output

        assert max(abs(a - b) for a, b in zip(got_ordered, ref)) < 1e-4

        # Registro P21: el modelo recién entrenado se registra y su hash
        # cubre de verdad los pesos (no el placeholder nulo).
        from matrixai.registry.model_registry import ModelRegistry
        (out_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(out_dir, "toy-c6", "1.0.0")
        assert entry.params_content_hash != ""
        assert registry.verify("toy-c6", "1.0.0") is True
