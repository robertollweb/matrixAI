# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C5 — export ONNX + inference_spec + estimador.

Contrato §C5: ONNX == torch == stdlib en shapes mini (patrón P15, validador
oficial de equivalencia con rama transformer); spec de entrada
{"encoding": "token_ids", "length": L, "vocab_size": V}; el estimador de
PESOS_GRANDES cuenta los params del bloque con la fórmula de C2 y un modelo
grande (12×768 ≈ 108M) supera el umbral.

Tolerancia documentada ONNX↔stdlib/torch: atol 1e-5 (grafo float32 vs
referencia float64; medida real ~5e-8 en mini).
"""
from __future__ import annotations

import tempfile
from importlib import util
from pathlib import Path

import pytest

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None
_HAS_TORCH = util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_ONNX and _HAS_ORT), reason="onnx + onnxruntime required"
)

from matrixai.parameters.network_params import (
    build_composite_network_parameter_set,
    transformer_block_param_count,
)
from matrixai.parameters.store import program_hash
from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types

ATOL_ONNX = 1e-5  # documentada (float32 vs float64); medida real ~5e-8


def _mxai(
    *, length: int = 6, vocab: int = 11, dim: int = 8, layers: int = 2,
    heads: int = 2, ff: int = 16, pos: str = "sinusoidal", pool: str = "mean",
    activation: str = "gelu",
) -> str:
    return f"""
PROJECT C5Test

SEQUENCE Texto
  length = {length}
  vocab_size = {vocab}
END

NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM {dim}
  BLOCK enc TRANSFORMER
    LAYERS {layers}
    HEADS {heads}
    FF {ff}
    ACTIVATION {activation}
    POS {pos}
  END
  POOL {pool}
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END

GRAPH
  Texto -> N
END
"""


def _build(src: str, seed: int = 7):
    prog = parse_text(src)
    net = prog.networks[0]
    res = check_composite_network_types(
        net, {v.name: v for v in prog.vectors}, {s.name: s for s in prog.sequences}
    )
    assert res.ok, res.errors
    ps = build_composite_network_parameter_set(net, res, program_hash(prog), seed=seed)
    return prog, net, res, ps


def _export(prog, ps):
    from matrixai.export.onnx_exporter import export_onnx
    path = Path(tempfile.mkdtemp()) / "model.onnx"
    result = export_onnx(prog, ps, path)
    return path, result


def _ort_run(path, ids, mask=None):
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    length = len(ids)
    mask_f = [1.0] * length if mask is None else [1.0 if m else 0.0 for m in mask]
    out = sess.run(None, {
        sess.get_inputs()[0].name: np.array([ids], dtype=np.int64),
        sess.get_inputs()[1].name: np.array([mask_f], dtype=np.float32),
    })
    return out[0][0].tolist()


IDS = [1, 2, 3, 4, 5, 6]
MASK3 = [True, True, True, False, False, False]


# ---------------------------------------------------------------------------
# Equivalencia ONNX == stdlib == torch (las tres vías del contrato)
# ---------------------------------------------------------------------------

class TestEquivalenciaTresVias:
    @pytest.mark.parametrize("variant", [
        {},                                  # base: mean/sinusoidal/gelu
        {"pool": "cls"},
        {"pos": "learned"},
        {"activation": "relu"},
        {"heads": 1},
        {"heads": 4},
    ])
    def test_onnx_equals_stdlib(self, variant):
        from matrixai.forward.transformer_forward import transformer_network_forward
        prog, net, res, ps = _build(_mxai(**variant))
        path, _ = _export(prog, ps)
        for ids, mask in ((IDS, None), ([1, 2, 3, 0, 0, 0], MASK3)):
            onnx_out = _ort_run(path, ids, mask)
            std = transformer_network_forward(net, res, ps, ids, mask=mask)
            assert max(abs(a - b) for a, b in zip(std.output, onnx_out)) < ATOL_ONNX

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
    def test_onnx_equals_torch(self):
        import torch
        from matrixai.forward.transformer_torch import (
            transformer_network_to_torch_module,
            transformer_torch_forward_batch,
        )
        prog, net, res, ps = _build(_mxai())
        path, _ = _export(prog, ps)
        module = transformer_network_to_torch_module(net, res, ps, dtype=torch.float64)
        for ids, mask in ((IDS, None), ([1, 2, 3, 0, 0, 0], MASK3)):
            onnx_out = _ort_run(path, ids, mask)
            tor = transformer_torch_forward_batch(
                module, [ids], masks=[mask] if mask else None
            )[0]
            assert max(abs(a - b) for a, b in zip(tor, onnx_out)) < ATOL_ONNX

    def test_official_validator_transformer_branch(self):
        """El validador OFICIAL (patrón P15) con la rama transformer: ids y
        máscaras aleatorias (mitad con padding), 20 muestras."""
        from matrixai.export.equivalence import validate_onnx_equivalence
        prog, net, res, ps = _build(_mxai())
        path, _ = _export(prog, ps)
        eq = validate_onnx_equivalence(prog, ps, path, n_samples=20)
        assert eq.passed, (eq.max_abs_diff, eq.max_rel_diff)
        assert eq.n_outputs_per_sample == 2
        assert eq.max_abs_diff < ATOL_ONNX

    def test_padding_content_irrelevant_in_onnx(self):
        """Invariante 1c también en el grafo exportado: cambiar el contenido
        de las posiciones enmascaradas no cambia la salida."""
        prog, net, res, ps = _build(_mxai())
        path, _ = _export(prog, ps)
        a = _ort_run(path, [1, 2, 3, 0, 0, 0], MASK3)
        b = _ort_run(path, [1, 2, 3, 9, 7, 5], MASK3)
        assert a == b

    def test_batch_export_matches_single_rows(self):
        import numpy as np
        import onnxruntime as ort
        prog, net, res, ps = _build(_mxai())
        path, _ = _export(prog, ps)
        sess = ort.InferenceSession(str(path))
        rows = [IDS, [6, 5, 4, 3, 2, 1], [1, 1, 2, 2, 3, 3]]
        batched = sess.run(None, {
            "Texto": np.array(rows, dtype=np.int64),
            "Texto_mask": np.ones((3, 6), dtype=np.float32),
        })[0].tolist()
        singles = [_ort_run(path, r) for r in rows]
        for b, s in zip(batched, singles):
            assert max(abs(x - y) for x, y in zip(b, s)) < 1e-6


# ---------------------------------------------------------------------------
# inference_spec — token_ids (literal del contrato)
# ---------------------------------------------------------------------------

class TestInferenceSpec:
    def _spec(self, **kwargs):
        from matrixai.export.inference_spec import build_inference_spec
        prog, net, res, ps = _build(_mxai(**kwargs))
        path, result = _export(prog, ps)
        return build_inference_spec(
            prog, ps, result, labels=["NEG", "POS"],
        ), prog

    def test_token_ids_encoding(self):
        spec, prog = self._spec()
        entry = spec["input"]["Texto"]
        assert entry == {"encoding": "token_ids", "length": 6, "vocab_size": 11}
        assert spec["input_order"] == ["Texto"]
        assert spec["input_name"] == spec["onnx_input"] == "Texto"
        assert spec["mask_input"] == "Texto_mask"
        assert spec["input_shape"] == spec["mask_shape"] == [-1, 6]
        assert spec["model_hash"] and spec["parameter_schema_hash"]

    def test_cls_note_present(self):
        spec, _ = self._spec(pool="cls")
        assert "position 0" in spec.get("notes", "")

    def test_example_input_validated(self):
        from matrixai.export.inference_spec import (
            InferenceSpecError,
            build_inference_spec,
        )
        prog, net, res, ps = _build(_mxai())
        path, result = _export(prog, ps)
        # ejemplo válido pasa
        build_inference_spec(
            prog, ps, result, labels=["NEG", "POS"],
            example_input={"Texto": IDS},
        )
        # longitud errónea y vocab fuera de rango fallan en el export
        with pytest.raises(InferenceSpecError, match="list of 6 token ids"):
            build_inference_spec(
                prog, ps, result, example_input={"Texto": [1, 2, 3]},
            )
        with pytest.raises(InferenceSpecError, match="outside"):
            build_inference_spec(
                prog, ps, result, example_input={"Texto": [1, 2, 3, 4, 5, 99]},
            )


# ---------------------------------------------------------------------------
# Estimador PESOS_GRANDES — la fórmula de C2 cuenta el bloque
# ---------------------------------------------------------------------------

class TestEstimador:
    def test_param_count_includes_block_formula(self):
        from matrixai.resources import estimate_model_resources
        prog, net, res, ps = _build(_mxai())
        est = estimate_model_resources(prog)
        expected_block = transformer_block_param_count(2, 8, 16)
        expected_total = expected_block + 11 * 8 + (4 * 8 + 4) + (2 * 4 + 2)
        assert est.param_count == expected_total

    def test_large_transformer_crosses_threshold(self):
        """El 12×768 del contrato (~108M params con embedding) supera el umbral
        de PESOS_GRANDES — el estimador avisa ANTES de entrenar sin materializar
        un solo peso (manifest O(#tensores))."""
        from matrixai.resources import estimate_model_resources, torch_native_min_params
        src = _mxai(length=64, vocab=30000, dim=768, layers=12, heads=12, ff=3072)
        prog = parse_text(src)
        est = estimate_model_resources(prog)
        expected_block = transformer_block_param_count(12, 768, 3072)
        assert est.param_count > expected_block  # + embedding + cabeza
        assert est.param_count > torch_native_min_params()
        assert est.exceeds_native_threshold is True
        assert est.to_dict()["exceeds_native_threshold"] is True
        assert est.weights_gib > 0.3  # ~108M float32 ≈ 0.4 GiB


# ---------------------------------------------------------------------------
# Capacidad de export y validaciones del exporter
# ---------------------------------------------------------------------------

class TestCapacidadExport:
    def test_backend_export_supported_after_c5(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        prog, *_ = _build(_mxai())
        report = BackendContractAnalyzer().analyze(prog)
        node = next(n for n in report.nodes if n.node == "N")
        assert node.export_ok is True          # C5
        assert node.training_ok is True        # C4
        assert node.forward_ok is False        # runner sin rama NETWORK
        assert node.supported is False         # agregado conservador

    def test_export_requires_materialized_params(self):
        from matrixai.export.onnx_exporter import OnnxExportError, export_onnx
        prog, net, res, _ = _build(_mxai())
        template = build_composite_network_parameter_set(
            net, res, program_hash(prog), seed=7, with_values=False,
        )
        path = Path(tempfile.mkdtemp()) / "m.onnx"
        # El validador de export corta ANTES con el listado completo de valores
        # ausentes (fail-closed correcto — el guard del pipeline es la 2ª capa).
        with pytest.raises(OnnxExportError, match="validation failed|invalid values"):
            export_onnx(prog, template, path)

    def test_export_rejects_dirty_typecheck(self):
        from matrixai.export.onnx_exporter import OnnxExportError, export_onnx
        src = _mxai(heads=3)  # 3 no divide 8
        prog = parse_text(src)
        net = prog.networks[0]
        _, _, res_ok, ps = _build(_mxai(heads=2))
        path = Path(tempfile.mkdtemp()) / "m.onnx"
        with pytest.raises(OnnxExportError, match="typecheck|hash"):
            export_onnx(prog, ps, path)

    def test_exported_metadata_hashes(self):
        prog, net, res, ps = _build(_mxai())
        path, result = _export(prog, ps)
        assert result.model_hash == ps.model_hash
        assert result.parameter_schema_hash == ps.parameter_schema_hash
        assert result.input_name == "Texto"
        assert result.input_shape == [-1, 6]
        assert result.output_shape == [-1, 2]


# ---------------------------------------------------------------------------
# Auditoría C5 — bundle público, activaciones intercaladas y PESOS_GRANDES
# ---------------------------------------------------------------------------

class TestC5AuditFixes:
    @pytest.mark.parametrize("where,kind", [
        ("pre", "softmax"),   # [B,L,D]: softmax debe ir sobre D, no sobre L
        ("post", "gelu"),     # tras POOL: GELU exacta, nunca Identity
    ])
    def test_interleaved_activation_onnx_parity(self, where, kind):
        from matrixai.forward.transformer_forward import transformer_network_forward
        src = _mxai()
        layer = f"  LAYER Activation kind={kind}"
        if where == "pre":
            src = src.replace("  BLOCK enc TRANSFORMER", layer + "\n  BLOCK enc TRANSFORMER")
        else:
            src = src.replace(
                "  LAYER Dense units=4 activation=relu",
                layer + "\n  LAYER Dense units=4 activation=relu",
            )
        prog, net, res, ps = _build(src)
        path, _ = _export(prog, ps)
        ids = [1, 2, 3, 0, 0, 0]
        got = _ort_run(path, ids, MASK3)
        expected = transformer_network_forward(net, res, ps, ids, mask=MASK3).output
        assert max(abs(a - b) for a, b in zip(got, expected)) < ATOL_ONNX

    def test_edge_bundle_predict_py_is_sequence_usable(self, tmp_path):
        import importlib.util
        import json
        from matrixai.export.bundle import create_edge_bundle
        from matrixai.parameters.store import write_parameter_set

        src = _mxai()
        prog, _net, _res, ps = _build(src)
        mxai = tmp_path / "model.mxai"
        params = tmp_path / "params.json"
        bundle = tmp_path / "bundle"
        mxai.write_text(src, encoding="utf-8")
        write_parameter_set(params, ps)
        result = create_edge_bundle(
            prog, ps, mxai, params, bundle, labels=["NEG", "POS"],
        )
        assert "model_manifest.json" in result.files
        assert json.loads((bundle / "example_input.json").read_text()) == {"Texto": [0] * 6}

        module_spec = importlib.util.spec_from_file_location("_c5_predict", bundle / "predict.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = module.MatrixAIModel(bundle / "inference_spec.json")
        direct = model.predict(IDS)
        explicit = model.predict({"Texto": IDS, "Texto_mask": [1, 1, 1, 1, 1, 1]})
        assert direct == explicit
        assert set(direct) == {"NEG", "POS"}

    def test_predict_rejects_invalid_masks(self, tmp_path):
        import importlib.util
        from matrixai.export.bundle import create_edge_bundle
        from matrixai.parameters.store import write_parameter_set

        src = _mxai(pool="cls")
        prog, _net, _res, ps = _build(src)
        mxai = tmp_path / "model.mxai"
        params = tmp_path / "params.json"
        bundle = tmp_path / "bundle"
        mxai.write_text(src, encoding="utf-8")
        write_parameter_set(params, ps)
        create_edge_bundle(
            prog, ps, mxai, params, bundle, labels=["NEG", "POS"], validate=False,
        )
        module_spec = importlib.util.spec_from_file_location("_c5_predict_mask", bundle / "predict.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = module.MatrixAIModel(bundle / "inference_spec.json")
        with pytest.raises(module.MatrixAIModelError, match="at least one real token"):
            model.predict({"Texto": IDS, "Texto_mask": [0] * 6})
        with pytest.raises(module.MatrixAIModelError, match=r"mask\[0\]"):
            model.predict({"Texto": IDS, "Texto_mask": [0, 1, 1, 1, 1, 1]})

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
    def test_state_dict_transformer_export_parity(self, tmp_path):
        import torch
        from matrixai.export.onnx_exporter import export_onnx

        prog, _net, _res, ps = _build(_mxai())
        state = {
            path: torch.tensor(entry["values"], dtype=torch.float32)
            for path, entry in ps.parameters.items()
        }
        path = tmp_path / "state.onnx"
        export_onnx(
            prog, None, path, state_dict=state,
            model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
        )
        expected_path, _ = _export(prog, ps)
        assert _ort_run(path, IDS, MASK3) == _ort_run(expected_path, IDS, MASK3)

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
    def test_state_dict_transformer_export_validates_schema_and_shapes(self, tmp_path):
        import torch
        from matrixai.export.onnx_exporter import OnnxExportError, export_onnx

        prog, _net, _res, ps = _build(_mxai())
        state = {
            path: torch.tensor(entry["values"], dtype=torch.float32)
            for path, entry in ps.parameters.items()
        }
        with pytest.raises(OnnxExportError, match="parameter_schema_hash"):
            export_onnx(
                prog, None, tmp_path / "schema.onnx", state_dict=state,
                model_hash=ps.model_hash, parameter_schema_hash="wrong",
            )
        broken = dict(state)
        broken["N.enc.layer_0.attention.Wq"] = torch.zeros(9, 8)
        with pytest.raises(OnnxExportError, match="shape"):
            export_onnx(
                prog, None, tmp_path / "shape.onnx", state_dict=broken,
                model_hash=ps.model_hash,
                parameter_schema_hash=ps.parameter_schema_hash,
            )

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
    def test_mxw_transformer_external_graph_roundtrip(self, tmp_path):
        import torch
        from matrixai.export.onnx_exporter import export_onnx_graph_external
        from matrixai.parameters.binary_store import (
            read_mxw_header_and_body_start,
            stream_mxw_tensor,
            write_mxw,
        )

        prog, _net, _res, ps = _build(_mxai())
        state = {
            path: torch.tensor(entry["values"], dtype=torch.float32)
            for path, entry in ps.parameters.items()
        }
        mxw = tmp_path / "weights.mxw"
        write_mxw(
            mxw, state, model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
        )
        header, body_start = read_mxw_header_and_body_start(mxw)
        onnx_path = tmp_path / "model.onnx"
        result, ordered = export_onnx_graph_external(
            prog, header, onnx_path,
            model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
        )
        with open(mxw, "rb") as source, open(tmp_path / "model.onnx.data", "wb") as data:
            for meta in ordered:
                stream_mxw_tensor(source, body_start, meta, data)
        assert result.external_data is True
        assert {m["path"] for m in ordered} == set(state)
        expected_path, _ = _export(prog, ps)
        assert _ort_run(onnx_path, IDS, MASK3) == _ort_run(expected_path, IDS, MASK3)

        # The public bundle dispatcher must use the same generic graph path;
        # before this audit it unconditionally called the dense-only exporter.
        from matrixai.export.bundle import create_edge_bundle
        template = build_composite_network_parameter_set(
            prog.networks[0],
            check_composite_network_types(prog.networks[0], {}, {"Texto": prog.sequences[0]}),
            program_hash(prog), with_values=False,
        )
        mxai = tmp_path / "model.mxai"
        mxai.write_text(_mxai(), encoding="utf-8")
        bundle_result = create_edge_bundle(
            prog, template, mxai, None, tmp_path / "bundle",
            mxw_path=mxw, mxw_header=header,
            model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
            labels=["NEG", "POS"], validate=False,
        )
        assert bundle_result.export_result.external_data is True
        assert bundle_result.external_data_layout
        assert "model.onnx.data" in bundle_result.files
