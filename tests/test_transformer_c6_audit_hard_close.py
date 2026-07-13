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
# Auditoría C6 ronda 2 [BAJA]: skip GRANULAR — las pruebas de registro y
# evaluate no necesitan onnx/ort; solo las de export lo exigen.
requires_onnx = pytest.mark.skipif(
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

    @requires_onnx
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
        (no params.best.json) — antes de este fix caía al hash nulo. Ronda 2
        [ALTA-4]: el .mxai del run es OBLIGATORIO en este camino."""
        from matrixai.parameters.store import write_parameter_set
        from matrixai.registry.model_registry import ModelRegistry
        _, _, _, ps = _build()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        write_parameter_set(run_dir / "parameter_set.json", ps)
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))
        (run_dir / "model.mxai").write_text(_MXAI)

        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(run_dir, "toy", "1.0.0")
        null_hash = "sha256:" + "0" * 64
        assert entry.parameter_schema_hash not in ("", null_hash)
        assert entry.parameter_schema_hash == ps.parameter_schema_hash
        assert entry.params_content_hash != ""
        assert registry.verify("toy", "1.0.0") is True

    def test_push_run_dir_modern_entry_without_mxai_rejected(self, tmp_path):
        """Ronda 2 [ALTA-4]: sin .mxai el cross-check pesos↔modelo no puede
        hacerse — antes se saltaba en silencio, ahora es error."""
        from matrixai.parameters.store import write_parameter_set
        from matrixai.registry.model_registry import ModelRegistry, ModelRegistryError
        _, _, _, ps = _build()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        write_parameter_set(run_dir / "parameter_set.json", ps)
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(ModelRegistryError, match="requires the trained model"):
            registry.push_run_dir(run_dir, "toy", "1.0.0")

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

    def _mxw_run_dir(self, tmp_path):
        """Run dir moderno completo: weights.mxw del modelo _MXAI + model.mxai
        + evaluation_report (ronda 2 [ALTA-4]: el .mxai es obligatorio)."""
        from matrixai.parameters.binary_store import write_mxw
        prog = parse_text(_MXAI)
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        state = {"N.enc.layer_0.attention.Wq": torch.randn(DIM, DIM)}
        header = write_mxw(
            run_dir / "weights.mxw", state,
            model_hash=program_hash(prog),
            parameter_schema_hash="sha256:" + "b" * 64,
        )
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.8}))
        (run_dir / "model.mxai").write_text(_MXAI)
        return run_dir, header

    def test_push_run_dir_reads_weights_mxw_header(self, tmp_path):
        """PESOS_GRANDES: sin parameter_set.json, solo weights.mxw — el hash
        del registro cubre el fichero COMPLETO (ronda 2 [ALTA-3]: la cabecera
        manda sobre la interpretación del cuerpo, también se hashea)."""
        from matrixai.parameters.binary_store import mxw_file_content_hash
        from matrixai.registry.model_registry import ModelRegistry
        run_dir, header = self._mxw_run_dir(tmp_path)

        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(run_dir, "toy-big", "1.0.0")
        assert entry.parameter_schema_hash == "sha256:" + "b" * 64
        assert entry.params_content_hash == (
            "sha256:" + mxw_file_content_hash(run_dir / "weights.mxw")
        )
        # y NO es el hash solo-cuerpo de la cabecera
        assert entry.params_content_hash != "sha256:" + header["content_hash"]
        assert (registry.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw").exists()
        assert registry.verify("toy-big", "1.0.0") is True

    def test_verify_detects_weights_mxw_body_tamper(self, tmp_path):
        """verify() re-deriva el hash en streaming — tamper del cuerpo detectado."""
        from matrixai.registry.model_registry import ModelRegistry, VerificationError
        run_dir, _ = self._mxw_run_dir(tmp_path)
        registry = ModelRegistry(tmp_path / "registry")
        registry.push_run_dir(run_dir, "toy-big", "1.0.0")
        assert registry.verify("toy-big", "1.0.0") is True

        mxw_path = registry.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw"
        data = bytearray(mxw_path.read_bytes())
        data[-1] ^= 0xFF
        mxw_path.write_bytes(bytes(data))
        with pytest.raises(VerificationError, match="weights.mxw"):
            registry.verify("toy-big", "1.0.0")

    def test_verify_detects_weights_mxw_header_tamper(self, tmp_path):
        """Ronda 2 [ALTA-3] (repro de Roberto): renombrar un path en la
        CABECERA (attention.Wq → attention.WX) manteniendo el cuerpo intacto
        pasaba verify() — la cabecera determina cómo se interpretan los
        bytes, así que el hash del registro cubre el fichero entero."""
        from matrixai.registry.model_registry import ModelRegistry, VerificationError
        run_dir, _ = self._mxw_run_dir(tmp_path)
        registry = ModelRegistry(tmp_path / "registry")
        registry.push_run_dir(run_dir, "toy-big", "1.0.0")

        mxw_path = registry.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw"
        data = mxw_path.read_bytes()
        tampered = data.replace(b"attention.Wq", b"attention.WX", 1)
        assert tampered != data
        mxw_path.write_bytes(tampered)
        with pytest.raises(VerificationError, match="weights.mxw"):
            registry.verify("toy-big", "1.0.0")

    def test_verify_detects_missing_artifacts(self, tmp_path):
        """Ronda 2 [ALTA-2] (repro de Roberto): borrar un artefacto con hash
        declarado dejaba verify() en verde — ahora falla cerrado, para CADA
        artefacto custodiado."""
        from matrixai.registry.model_registry import ModelRegistry, VerificationError
        for missing, pattern in [
            ("weights.mxw", "neither params.json nor weights.mxw"),
            ("model.mxai", "model.mxai is declared"),
            ("evaluation_report.json", "evaluation_report.json is declared"),
        ]:
            run_dir, _ = self._mxw_run_dir(tmp_path / missing.replace(".", "_"))
            registry = ModelRegistry(tmp_path / f"registry_{missing}")
            registry.push_run_dir(run_dir, "toy-big", "1.0.0")
            assert registry.verify("toy-big", "1.0.0") is True
            (registry.layout.entry_dir("toy-big", "1.0.0") / missing).unlink()
            with pytest.raises(VerificationError, match=pattern):
                registry.verify("toy-big", "1.0.0")

    def test_pull_copies_weights_mxw_across_registries(self, tmp_path):
        from matrixai.registry.model_registry import ModelRegistry
        run_dir, _ = self._mxw_run_dir(tmp_path)
        src = ModelRegistry(tmp_path / "src_registry")
        src.push_run_dir(run_dir, "toy-big", "1.0.0")
        dst = ModelRegistry(tmp_path / "dst_registry")
        src.pull("toy-big", "1.0.0", dst)
        assert (dst.layout.entry_dir("toy-big", "1.0.0") / "weights.mxw").exists()
        assert dst.verify("toy-big", "1.0.0") is True


# ---------------------------------------------------------------------------
# (c) Cierre duro — ciclo CLI completo
# ---------------------------------------------------------------------------

@requires_onnx
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

        # Tolerancia DOCUMENTADA C5 (ONNX float32 vs referencia float64):
        # atol 1e-5, medida real ~5e-8 — el cierre duro usa la misma, no
        # una más laxa (auditoría C6 [BAJA-2]).
        assert max(abs(a - b) for a, b in zip(got_ordered, ref)) < 1e-5

        # Registro P21 vía el ciclo REAL (auditoría C6 [ALTA-1]): el
        # evaluation_report lo produce `mx evaluate` de verdad — antes se
        # escribía a mano en este test, maquillando que el CLI no tenía
        # evaluador transformer y producía un report vacío en verde.
        evaluate = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "evaluate", str(model_path),
             "--training", str(tmp_path / "toy.mxtrain"),
             "--params", str(ps_path),
             "--output", str(out_dir / "evaluation_report.json"), "--json"],
            capture_output=True, text=True, timeout=180,
        )
        assert evaluate.returncode == 0, evaluate.stderr
        report = json.loads((out_dir / "evaluation_report.json").read_text())
        assert report["rows"] == 20  # real, no el 0 del fail-open denso

        from matrixai.registry.model_registry import ModelRegistry
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(out_dir, "toy-c6", "1.0.0")
        assert entry.params_content_hash != ""
        assert registry.verify("toy-c6", "1.0.0") is True


# ---------------------------------------------------------------------------
# Auditoría C6 (2026-07-12) — evaluate CLI, WASM mask, validadores, cross-check
# ---------------------------------------------------------------------------

class TestC6AuditFixes:
    def _trained(self, tmp_path):
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
        return model_path, tmp_path / "toy.mxtrain", out_dir / "parameter_set.json", rows

    def test_evaluator_produces_real_metrics(self, tmp_path):
        """[ALTA-1] El evaluador dedicado evalúa filas REALES (no 0 en verde)."""
        from matrixai.parameters.store import load_parameter_set
        from matrixai.training import parse_training_text
        from matrixai.training.transformer_trainer import TransformerSupervisedEvaluator
        model_path, mxtrain, ps_path, rows = self._trained(tmp_path)
        training = parse_training_text(mxtrain.read_text())
        result = TransformerSupervisedEvaluator().evaluate(
            training, load_parameter_set(ps_path), base_path=tmp_path,
        )
        assert result.rows == len(rows) == 20
        assert result.labels == ["NEG", "POS"]
        assert result.loss > 0.0
        assert result.confusion_matrix  # métricas de clasificación reales
        assert result.to_dict()["parameter_schema_hash"]

    def test_cli_evaluate_rejects_stdlib_and_defaults_to_transformer(self, tmp_path):
        """[ALTA-1/2] `mx evaluate` real: default evalúa, stdlib explícito error."""
        model_path, mxtrain, ps_path, _ = self._trained(tmp_path)
        ok = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "evaluate", str(model_path),
             "--training", str(mxtrain), "--params", str(ps_path), "--json"],
            capture_output=True, text=True, timeout=180,
        )
        assert ok.returncode == 0, ok.stderr
        payload = json.loads(ok.stdout)
        assert payload["rows"] == 20
        bad = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "evaluate", str(model_path),
             "--training", str(mxtrain), "--params", str(ps_path),
             "--backend", "stdlib"],
            capture_output=True, text=True, timeout=120,
        )
        assert bad.returncode == 1
        assert "requires --backend torch" in bad.stderr

    @requires_onnx
    def test_wasm_predict_js_feeds_the_mask(self, tmp_path):
        """[ALTA-3] El predict.js del WASM alimenta AMBAS entradas del grafo."""
        from matrixai.export.wasm_exporter import export_wasm
        prog = parse_text(_MXAI)
        net = prog.networks[0]
        res = check_composite_network_types(
            net, {}, {s.name: s for s in prog.sequences})
        ps = build_composite_network_parameter_set(
            net, res, program_hash(prog), seed=3)
        result = export_wasm(prog, ps, output_dir=str(tmp_path / "wasm"))
        js = (tmp_path / "wasm" / "predict.js").read_text()
        assert "Texto_mask" in js
        assert "maskData" in js and "Float32Array.from(mask)" in js
        manifest = json.loads((tmp_path / "wasm" / "wasm_manifest.json").read_text())
        assert manifest["mask_input"] == "Texto_mask"
        # El grafo de verdad exige la mask: alimentar solo los ids FALLA
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(tmp_path / "wasm" / "model.onnx"))
        assert {i.name for i in sess.get_inputs()} == {"Texto", "Texto_mask"}
        with pytest.raises(Exception, match="Texto_mask"):
            sess.run(None, {"Texto": np.array([[1, 2, 3, 4, 5, 0]], dtype=np.int64)})

    @requires_onnx
    def test_wasm_dense_manifest_mask_is_none(self, tmp_path):
        """El resto de modelos no gana una mask falsa en el manifest WASM."""
        from matrixai.export.wasm_exporter import export_wasm
        from matrixai.parameters.network_params import build_network_parameter_set
        from matrixai.types import check_network_types
        dense_src = """
PROJECT DenseWasm
VECTOR X[2]
  a: Scalar
  b: Scalar
END
NETWORK D
  INPUT X
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
        ps = build_network_parameter_set(net, res.resolved_layers, program_hash(prog), seed=1)
        export_wasm(prog, ps, output_dir=str(tmp_path / "wasm"))
        manifest = json.loads((tmp_path / "wasm" / "wasm_manifest.json").read_text())
        assert manifest["mask_input"] is None
        assert "maskData" not in (tmp_path / "wasm" / "predict.js").read_text()

    def test_cli_validate_parameters_accepts_transformer(self, tmp_path):
        """[MEDIA-1] `mx validate-parameters` valida un set transformer válido."""
        model_path, mxtrain, ps_path, _ = self._trained(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "validate-parameters",
             str(model_path), "--params", str(ps_path), "--json"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["ok"] is True
        assert payload["errors"] == []

    def test_push_run_dir_rejects_foreign_mxai(self, tmp_path):
        """[MEDIA-2] parameter_set.json de OTRO modelo junto a un .mxai → error."""
        from matrixai.parameters.store import write_parameter_set
        from matrixai.registry.model_registry import ModelRegistry, ModelRegistryError
        _, _, _, ps = _build()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        write_parameter_set(run_dir / "parameter_set.json", ps)
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))
        # Un .mxai que NO es el modelo de esos pesos
        (run_dir / "model.mxai").write_text(
            "PROJECT Otro\nVECTOR X[1]\n  a: Scalar\nEND\n"
            "NETWORK D\n  INPUT X\n  LAYER Dense units=2 activation=softmax\n"
            "  OUTPUT clase: ProbabilityMap[NEG, POS]\nEND\nGRAPH\n  X -> D\nEND\n"
        )
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(ModelRegistryError, match="different model"):
            registry.push_run_dir(run_dir, "evil", "1.0.0")

    def test_push_run_dir_matching_mxai_passes(self, tmp_path):
        """[MEDIA-2] El caso legítimo (mismo modelo) registra y verifica."""
        from matrixai.parameters.store import write_parameter_set
        from matrixai.registry.model_registry import ModelRegistry
        prog, net, res, ps = _build()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        write_parameter_set(run_dir / "parameter_set.json", ps)
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.9}))
        (run_dir / "model.mxai").write_text(_MXAI)
        registry = ModelRegistry(tmp_path / "registry")
        entry = registry.push_run_dir(run_dir, "toy", "1.0.0")
        assert registry.verify("toy", "1.0.0") is True

    def test_push_run_dir_mxw_foreign_model_hash_rejected(self, tmp_path):
        """[MEDIA-2] weights.mxw con model_hash ajeno al .mxai del run → error."""
        from matrixai.parameters.binary_store import write_mxw
        from matrixai.registry.model_registry import ModelRegistry, ModelRegistryError
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        state = {"N.enc.layer_0.attention.Wq": torch.randn(DIM, DIM)}
        write_mxw(
            run_dir / "weights.mxw", state,
            model_hash="mxai_" + "0" * 16,  # NO es el hash de _MXAI
            parameter_schema_hash="sha256:" + "b" * 64,
        )
        (run_dir / "evaluation_report.json").write_text(json.dumps({"accuracy": 0.8}))
        (run_dir / "model.mxai").write_text(_MXAI)
        registry = ModelRegistry(tmp_path / "registry")
        with pytest.raises(ModelRegistryError, match="different model"):
            registry.push_run_dir(run_dir, "evil-big", "1.0.0")

    @requires_onnx
    def test_onnx_embeds_transformer_block_metadata(self, tmp_path):
        """[MEDIA-3] La metadata del bloque viaja EN el .onnx exportado."""
        from matrixai.export.onnx_exporter import export_onnx
        import onnxruntime as ort
        prog, net, res, ps = _build()
        path = tmp_path / "model.onnx"
        export_onnx(prog, ps, path)
        meta = ort.InferenceSession(str(path)).get_modelmeta().custom_metadata_map
        block = json.loads(meta["matrixai_transformer_block"])
        assert block["layers"] == LAYERS
        assert block["heads"] == HEADS
        assert block["backend"] == "torch"
        # y coincide con lo que declaran el spec y el manifest (fuente única)
        from matrixai.parameters.network_params import (
            transformer_block_export_metadata_for_program,
        )
        assert block == transformer_block_export_metadata_for_program(prog)

    def test_metadata_helper_returns_none_for_dense(self):
        from matrixai.parameters.network_params import (
            transformer_block_export_metadata_for_program,
        )
        dense_src = """
PROJECT DenseMeta
VECTOR X[1]
  a: Scalar
END
NETWORK D
  INPUT X
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END
GRAPH
  X -> D
END
"""
        assert transformer_block_export_metadata_for_program(parse_text(dense_src)) is None


# ---------------------------------------------------------------------------
# Auditoría C6 ronda 2 (Roberto, 2026-07-12) — ciclo .mxw por CLI, P21
# fail-closed, JS ejecutado de verdad
# ---------------------------------------------------------------------------

_HAS_NODE = __import__("shutil").which("node") is not None


class TestC6AuditRound2:
    def _trained_mxw(self, tmp_path):
        """Entrena con el umbral PESOS_GRANDES forzado → weights.mxw real."""
        import os
        model_path, csv_path, rows = _write_dataset(tmp_path)
        (tmp_path / "toy.mxtrain").write_text(
            _MXTRAIN.format(model="toy.mxai", csv="toy.csv"), encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        env = dict(os.environ, MATRIXAI_TORCH_NATIVE_MIN_PARAMS="10")
        train = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "train", str(model_path),
             "--training", str(tmp_path / "toy.mxtrain"),
             "--output", str(out_dir), "--json"],
            capture_output=True, text=True, timeout=300, env=env,
        )
        assert train.returncode == 0, train.stderr
        assert (out_dir / "weights.mxw").exists()
        assert (out_dir / "model.mxai").exists()  # [ALTA-4] el trainer lo copia
        assert not (out_dir / "parameter_set.json").exists()
        return model_path, tmp_path / "toy.mxtrain", out_dir

    def test_mxw_cli_cycle_evaluate_export_registry(self, tmp_path):
        """[ALTA-1] Ciclo CLI completo con weights.mxw: evaluate → export-onnx
        → export-bundle → registry push/verify — antes moría en UTF-8."""
        model_path, mxtrain, out_dir = self._trained_mxw(tmp_path)
        mxw = out_dir / "weights.mxw"

        ev = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "evaluate", str(model_path),
             "--training", str(mxtrain), "--params", str(mxw),
             "--output", str(out_dir / "evaluation_report.json"), "--json"],
            capture_output=True, text=True, timeout=180,
        )
        assert ev.returncode == 0, ev.stderr
        payload = json.loads(ev.stdout)
        assert payload["rows"] == 20
        assert payload["parameter_set_id"] == "torch_state"

        if not (_HAS_ONNX and _HAS_ORT):
            pytest.skip("onnx + onnxruntime required for the export half")
        onnx_out = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "export-onnx", str(model_path),
             "--params", str(mxw), "--output", str(out_dir / "model.onnx")],
            capture_output=True, text=True, timeout=180,
        )
        assert onnx_out.returncode == 0, onnx_out.stderr

        bundle = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "export-bundle", str(model_path),
             "--params", str(mxw), "--outdir", str(tmp_path / "bundle"), "--json"],
            capture_output=True, text=True, timeout=180,
        )
        assert bundle.returncode == 0, bundle.stderr
        bpayload = json.loads(bundle.stdout)
        assert bpayload["equivalence_skipped_reason"]  # registrado, no silencioso
        assert (tmp_path / "bundle" / "predict.py").exists()

        # ONNX del bundle == ONNX del export directo, bit a bit en la salida
        import numpy as np
        import onnxruntime as ort
        ids = np.array([[1, 2, 3, 4, 5, 0]], dtype=np.int64)
        mask = np.ones((1, L), dtype=np.float32)
        a = ort.InferenceSession(str(out_dir / "model.onnx")).run(
            None, {"Texto": ids, "Texto_mask": mask})[0]
        b = ort.InferenceSession(str(tmp_path / "bundle" / "model.onnx")).run(
            None, {"Texto": ids, "Texto_mask": mask})[0]
        assert float(abs(a - b).max()) == 0.0

        # Registro P21 del run .mxw completo
        push = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "registry", "push", str(out_dir),
             "--name", "toy-big", "--version", "v1",
             "--registry-path", str(tmp_path / "reg")],
            capture_output=True, text=True, timeout=120,
        )
        assert push.returncode == 0, push.stderr
        verify = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "registry", "verify", "toy-big@v1",
             "--registry-path", str(tmp_path / "reg")],
            capture_output=True, text=True, timeout=120,
        )
        assert verify.returncode == 0, verify.stderr

    def test_mxw_evaluate_matches_json_evaluate(self, tmp_path):
        """Los MISMOS pesos evaluados por el camino .mxw y por el camino JSON
        dan métricas idénticas (mismo seed de entrenamiento, CPU)."""
        import os
        model_path, csv_path, rows = _write_dataset(tmp_path)
        (tmp_path / "toy.mxtrain").write_text(
            _MXTRAIN.format(model="toy.mxai", csv="toy.csv"), encoding="utf-8",
        )
        results = {}
        for tag, env_extra in [("json", {}), ("mxw", {"MATRIXAI_TORCH_NATIVE_MIN_PARAMS": "10"})]:
            out_dir = tmp_path / f"out_{tag}"
            env = dict(os.environ, **env_extra)
            train = subprocess.run(
                [sys.executable, "-m", "matrixai.cli", "train", str(model_path),
                 "--training", str(tmp_path / "toy.mxtrain"),
                 "--output", str(out_dir), "--json"],
                capture_output=True, text=True, timeout=300, env=env,
            )
            assert train.returncode == 0, train.stderr
            params = out_dir / ("parameter_set.json" if tag == "json" else "weights.mxw")
            assert params.exists()
            ev = subprocess.run(
                [sys.executable, "-m", "matrixai.cli", "evaluate", str(model_path),
                 "--training", str(tmp_path / "toy.mxtrain"),
                 "--params", str(params), "--json"],
                capture_output=True, text=True, timeout=180,
            )
            assert ev.returncode == 0, ev.stderr
            results[tag] = json.loads(ev.stdout)
        assert results["json"]["rows"] == results["mxw"]["rows"] == 20
        assert abs(results["json"]["loss"] - results["mxw"]["loss"]) < 1e-6
        assert results["json"]["accuracy"] == results["mxw"]["accuracy"]

    def test_mxw_evaluate_foreign_model_rejected(self, tmp_path):
        """El header del .mxw declara su modelo — evaluarlo contra otro .mxai
        es error, no coincidencia de shapes."""
        model_path, mxtrain, out_dir = self._trained_mxw(tmp_path)
        # Cambia el modelo (FF distinto → mismo error aunque los shapes casen o no)
        other = _MXAI.replace(f"FF {FF}", f"FF {FF * 2}")
        model_path.write_text(other, encoding="utf-8")
        ev = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "evaluate", str(model_path),
             "--training", str(mxtrain), "--params", str(out_dir / "weights.mxw")],
            capture_output=True, text=True, timeout=180,
        )
        assert ev.returncode == 1
        assert "different model" in ev.stderr

    def test_mxw_export_wasm_rejected_actionably(self, tmp_path):
        """[ALTA-1] export-wasm con .mxw → error accionable, no UTF-8."""
        model_path, mxtrain, out_dir = self._trained_mxw(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "export-wasm", str(model_path),
             "--params", str(out_dir / "weights.mxw"), "--outdir", str(tmp_path / "w")],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 1
        assert ".mxw" in proc.stderr and "export-bundle" in proc.stderr
        assert "UnicodeDecodeError" not in proc.stderr

    @requires_onnx
    def test_onnx_embeds_matrixai_version(self, tmp_path):
        """[MEDIA-1] La versión del compilador viaja EN el .onnx autónomo."""
        from matrixai.export.onnx_exporter import export_onnx
        import matrixai as _mx
        import onnxruntime as ort
        prog, net, res, ps = _build()
        path = tmp_path / "model.onnx"
        export_onnx(prog, ps, path)
        meta = ort.InferenceSession(str(path)).get_modelmeta().custom_metadata_map
        assert meta.get("matrixai_version") == _mx.__version__

    @requires_onnx
    @pytest.mark.skipif(not _HAS_NODE, reason="node required to execute predict.js")
    def test_wasm_predict_js_validations_run_for_real(self, tmp_path):
        """[MEDIA-2] Ejecuta el predict.js REAL con node (ort stubeado):
        las validaciones semánticas rechazan máscara todo-ceros (antes: NaN
        silencioso), longitudes malas, ids fuera de rango y mask[0]=0 con
        POOL cls; una llamada válida construye ambos feeds."""
        from matrixai.export.wasm_exporter import export_wasm
        for pool, expect_cls in [("mean", False), ("cls", True)]:
            src = _MXAI.replace("POOL mean", f"POOL {pool}")
            prog = parse_text(src)
            net = prog.networks[0]
            res = check_composite_network_types(
                net, {}, {s.name: s for s in prog.sequences})
            ps = build_composite_network_parameter_set(
                net, res, program_hash(prog), seed=3)
            outdir = tmp_path / f"wasm_{pool}"
            export_wasm(prog, ps, output_dir=str(outdir))
            harness = f"""
global.ort = {{
  InferenceSession: {{ create: async () => ({{
    run: async (feeds) => {{
      if (!feeds['Texto'] || !feeds['Texto_mask']) throw new Error('missing feed');
      return {{ 'N_L3_out': {{ data: [0.5, 0.5] }} }};
    }}
  }}) }},
  Tensor: class {{ constructor(t, d, s) {{ this.type = t; this.data = d; this.dims = s; }} }},
}};
const {{ predict }} = require('{outdir.as_posix()}/predict.js');
async function expectThrow(fn, tag) {{
  try {{ await fn(); console.log('NOTHROW:' + tag); }}
  catch (e) {{ console.log('THREW:' + tag + ':' + e.message); }}
}}
(async () => {{
  const ok = await predict([1,2,3,4,5,0]);
  console.log('OK:' + JSON.stringify(ok));
  await expectThrow(() => predict([1,2,3,4,5,0], [0,0,0,0,0,0]), 'zeromask');
  await expectThrow(() => predict([1,2,3], undefined), 'shortids');
  await expectThrow(() => predict([1,2,3,4,5,99]), 'vocab');
  await expectThrow(() => predict([1,2,3,4,5,0], [1,1,2,1,1,1]), 'nonbinary');
  await expectThrow(() => predict([1,2,3,4,5,0], [0,1,1,1,1,1]), 'clsmask');
}})();
"""
            script = tmp_path / f"harness_{pool}.js"
            script.write_text(harness, encoding="utf-8")
            proc = subprocess.run(
                ["node", str(script)], capture_output=True, text=True, timeout=60,
            )
            assert proc.returncode == 0, proc.stderr
            out = proc.stdout
            assert "OK:[0.5,0.5]" in out
            assert "THREW:zeromask" in out
            assert "THREW:shortids" in out
            assert "THREW:vocab" in out
            assert "THREW:nonbinary" in out
            if expect_cls:
                assert "THREW:clsmask" in out
            else:
                assert "NOTHROW:clsmask" in out  # mean no exige mask[0]
