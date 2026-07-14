# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""SECUENCIAS_PRODUCTO C5 — export descargable con tokenizador embebido.

Contrato §C5: `create_edge_bundle`/inference_spec aceptan `field_seq` (misma
procedencia que field_ranges/types/categories — invariante 4) y elevan la
entrada SEQUENCE a `{"encoding": "text", "tokenizer": {config byte_v1},
"length": L}`; predict.py tokeniza con un `_ByteTokenizer` EMBEBIDO (stdlib,
cero dependencias nuevas) y acepta texto crudo (invariante 1). Sin
`field_seq` el comportamiento es BYTE A BYTE el de TRANSFORMER_BLOQUE C5
(retrocompatible — `test_transformer_c5_onnx_export.py` sigue intacto).

Cierre duro del contrato: ciclo completo de usuario (prompt → generar →
dataset → entrenar → export → importar predict.py → predecir con texto
crudo) reproduce `expected_output.json`, igual que `test_gen_c6_roundtrip.py`
para modelos tabulares.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

import pytest

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None
_HAS_TORCH = util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_ONNX and _HAS_ORT), reason="onnx + onnxruntime required"
)

from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.parameters.store import program_hash
from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types


def _mxai(
    *, length: int = 8, vocab: int = 259, dim: int = 8, layers: int = 1,
    heads: int = 2, ff: int = 16, pool: str = "mean",
) -> str:
    return f"""
PROJECT C5TextTest

SEQUENCE Resenas
  length = {length}
  vocab_size = {vocab}
END

NETWORK N
  INPUT Resenas
  EMBEDDING tok FROM Resenas DIM {dim}
  BLOCK enc TRANSFORMER
    LAYERS {layers}
    HEADS {heads}
    FF {ff}
    ACTIVATION gelu
    POS sinusoidal
  END
  POOL {pool}
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END

GRAPH
  Resenas -> N
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


# ---------------------------------------------------------------------------
# inference_spec — encoding "text" (aditivo, gateado por field_seq explícito)
# ---------------------------------------------------------------------------

class TestInferenceSpecText:
    FIELD_SEQ = {"resenas": {"length": 8, "tokenizer": "byte_v1"}}

    def _spec(self, field_seq=None, **kwargs):
        from matrixai.export.inference_spec import build_inference_spec
        prog, net, res, ps = _build(_mxai(**kwargs))
        path, result = _export(prog, ps)
        return build_inference_spec(
            prog, ps, result, labels=["NEG", "POS"], field_seq=field_seq,
        ), prog

    def test_text_encoding_with_field_seq(self):
        spec, _ = self._spec(field_seq=self.FIELD_SEQ)
        entry = spec["input"]["Resenas"]
        assert entry["encoding"] == "text"
        assert entry["length"] == 8
        assert entry["tokenizer"] == {
            "kind": "byte_v1", "length": 8, "vocab_size": 259,
            "pad": 256, "cls": 258, "add_cls": False,
        }
        assert spec["input_order"] == ["Resenas"]
        assert spec["mask_input"] == "Resenas_mask"

    def test_add_cls_true_when_pool_cls(self):
        spec, _ = self._spec(field_seq=self.FIELD_SEQ, pool="cls")
        assert spec["input"]["Resenas"]["tokenizer"]["add_cls"] is True

    def test_add_cls_false_when_pool_mean(self):
        spec, _ = self._spec(field_seq=self.FIELD_SEQ, pool="mean")
        assert spec["input"]["Resenas"]["tokenizer"]["add_cls"] is False

    def test_without_field_seq_stays_token_ids_retrocompat(self):
        """Sin field_seq, byte a byte el contrato A (TRANSFORMER_BLOQUE C5)."""
        spec, _ = self._spec(field_seq=None)
        assert spec["input"]["Resenas"] == {
            "encoding": "token_ids", "length": 8, "vocab_size": 259,
        }

    def test_length_mismatch_raises(self):
        from matrixai.export.inference_spec import InferenceSpecError
        with pytest.raises(InferenceSpecError, match="does not describe this model"):
            self._spec(field_seq={"resenas": {"length": 99, "tokenizer": "byte_v1"}})

    def test_example_input_text_validated(self):
        from matrixai.export.inference_spec import build_inference_spec, InferenceSpecError
        prog, net, res, ps = _build(_mxai())
        path, result = _export(prog, ps)
        # texto no vacío pasa
        build_inference_spec(
            prog, ps, result, labels=["NEG", "POS"], field_seq=self.FIELD_SEQ,
            example_input={"Resenas": "hola"},
        )
        with pytest.raises(InferenceSpecError, match="non-empty string"):
            build_inference_spec(
                prog, ps, result, field_seq=self.FIELD_SEQ,
                example_input={"Resenas": ""},
            )
        with pytest.raises(InferenceSpecError, match="non-empty string"):
            build_inference_spec(
                prog, ps, result, field_seq=self.FIELD_SEQ,
                example_input={"Resenas": [1, 2, 3]},
            )

    def test_build_example_input_text_placeholder(self):
        from matrixai.export.inference_spec import build_example_input
        spec, _ = self._spec(field_seq=self.FIELD_SEQ)
        assert build_example_input(spec) == {"Resenas": "texto de ejemplo"}

    def test_build_example_input_legacy_ids_unchanged(self):
        from matrixai.export.inference_spec import build_example_input
        spec, _ = self._spec(field_seq=None)
        assert build_example_input(spec) == {"Resenas": [0] * 8}


# ---------------------------------------------------------------------------
# predict.py embebido — _ByteTokenizer replica matrixai.text.tokenizer BYTE A BYTE
# ---------------------------------------------------------------------------

class TestEmbeddedByteTokenizerMatchesCore:
    _TEXTS = [
        "", "hola", "el producto llegó roto", "emoji 😀🚀 test",
        "x" * 200,  # fuerza truncado
    ]

    @pytest.mark.parametrize("length", [1, 6, 16, 64])
    @pytest.mark.parametrize("add_cls", [False, True])
    def test_matches_core_tokenizer_across_texts(self, length, add_cls):
        from matrixai.text.tokenizer import ByteTokenizer
        from matrixai.export.predict_template import _ByteTokenizer

        core = ByteTokenizer(length)
        cfg = core.config()
        cfg["add_cls"] = add_cls
        embedded = _ByteTokenizer(cfg)
        for text in self._TEXTS:
            assert embedded.encode(text) == core.encode(text, add_cls=add_cls), text

    def test_rejects_unknown_kind(self):
        from matrixai.export.predict_template import _ByteTokenizer, MatrixAIModelError
        with pytest.raises(MatrixAIModelError, match="byte_v1"):
            _ByteTokenizer({"kind": "bpe", "length": 8, "pad": 256, "cls": 258})

    def test_rejects_non_string_input(self):
        from matrixai.export.predict_template import _ByteTokenizer, MatrixAIModelError
        tok = _ByteTokenizer({"kind": "byte_v1", "length": 8, "pad": 256, "cls": 258})
        with pytest.raises(MatrixAIModelError, match="string"):
            tok.encode(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bundle end-to-end: field_seq -> inference_spec.json -> predict.py (texto crudo)
# ---------------------------------------------------------------------------

class TestBundleEndToEndText(unittest.TestCase):
    FIELD_SEQ = {"resenas": {"length": 8, "tokenizer": "byte_v1"}}

    def _bundle(self):
        from matrixai.export import create_edge_bundle
        from matrixai.parameters import write_parameter_set
        prog, net, res, ps = _build(_mxai())
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(_mxai(), encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)
        result = create_edge_bundle(
            prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
            outdir=str(td / "b"), validate=True,
            field_seq=self.FIELD_SEQ, labels=["NEG", "POS"],
        )
        return result, Path(result.bundle_dir)

    def test_bundle_produces_text_inference_spec(self):
        result, bd = self._bundle()
        self.assertIsNone(result.inference_spec_skipped_reason)
        spec = json.loads((bd / "inference_spec.json").read_text())
        self.assertEqual(spec["input"]["Resenas"]["encoding"], "text")
        example = json.loads((bd / "example_input.json").read_text())
        self.assertEqual(example, {"Resenas": "texto de ejemplo"})

    def test_predict_py_reproduces_expected_output(self):
        result, bd = self._bundle()
        m = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        example = json.loads((bd / "example_input.json").read_text())
        expected = json.loads((bd / "expected_output.json").read_text())
        got = model.predict(example)
        for k, v in expected.items():
            self.assertAlmostEqual(got[k], v, places=4)

    def test_predict_accepts_bare_string_and_dict_forms(self):
        result, bd = self._bundle()
        m = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        out_str = model.predict("el producto llegó roto")
        out_dict = model.predict({"Resenas": "el producto llegó roto"})
        self.assertEqual(set(out_str), {"NEG", "POS"})
        self.assertAlmostEqual(out_str["NEG"], out_dict["NEG"], places=9)
        self.assertAlmostEqual(sum(out_str.values()), 1.0, places=5)

    def test_predict_batch_texts(self):
        result, bd = self._bundle()
        m = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        outs = model.predict_batch(["buen producto", "muy mal servicio"])
        self.assertEqual(len(outs), 2)
        for out in outs:
            self.assertAlmostEqual(sum(out.values()), 1.0, places=5)

    def test_predict_rejects_non_text_record(self):
        result, bd = self._bundle()
        m = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        with self.assertRaises(mod.MatrixAIModelError):
            model.predict([1, 2, 3])  # legacy token_ids shape, not accepted here
        with self.assertRaises(mod.MatrixAIModelError):
            model.predict({"Resenas": 123})


# ---------------------------------------------------------------------------
# Cierre duro del contrato: ciclo COMPLETO de usuario, texto crudo de punta a
# punta — prompt -> generar -> dataset (plantillas) -> entrenar -> exportar ->
# importar predict.py -> predecir con texto humano.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_ONNX and _HAS_ORT and _HAS_TORCH, "onnx/onnxruntime/torch required")
class CierreDuroTextRoundTripTest(unittest.TestCase):
    PROMPT = "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]"

    def test_user_roundtrip_prompt_to_prediction_with_raw_text(self):
        from matrixai.playground import analyze_playground_request, _generate_synthetic_dataset, _run_playground_training
        from matrixai.parameters.store import ParameterSet
        from matrixai.parameters import write_parameter_set
        from matrixai.export import create_edge_bundle

        # 1. generar — un campo Text del prompt basta por sí solo (C2)
        res = analyze_playground_request({"mode": "prompt", "prompt": self.PROMPT, "use_llm": False})
        self.assertTrue(res.get("ok"), res.get("error"))
        mxai, training = res["mxai"], res["training_text"]
        field_seq = res["field_seq"]
        self.assertIn("resenas", field_seq)
        self.assertEqual(field_seq["resenas"]["tokenizer"], "byte_v1")

        # 2. dataset sintético de plantillas — señal real, texto crudo en el CSV
        ds = _generate_synthetic_dataset(mxai, training, 24, 7, "coherent", use_llm=False)
        self.assertTrue(ds.get("ok"), ds.get("error"))
        self.assertEqual(ds["field_seq"], field_seq)

        # 3. entrenar (breve) — mismo dispatch network_call que dense/composite
        tr = _run_playground_training(mxai, training, ds["csv_text"], epochs_override=3)
        self.assertTrue(tr.get("ok"), tr.get("error"))
        self.assertEqual(tr["network_kind"], "composite_network")
        ps = ParameterSet.from_dict(tr["params_best"])

        # 4. exportar Edge bundle con field_seq — texto crudo de punta a punta
        prog = parse_text(mxai)
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(mxai, encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)
        result = create_edge_bundle(
            prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
            outdir=str(td / "b"), validate=True,
            field_seq=field_seq, labels=["NEG", "POS"],
        )
        self.assertIsNone(result.inference_spec_skipped_reason)
        bd = Path(result.bundle_dir)
        spec = json.loads((bd / "inference_spec.json").read_text())
        seq_name = spec["input_order"][0]
        self.assertEqual(spec["input"][seq_name]["encoding"], "text")

        # 5. importar predict.py — standalone, sin MatrixAI
        m = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        # expected_output.json se reproduce con example_input.json
        example = json.loads((bd / "example_input.json").read_text())
        expected = json.loads((bd / "expected_output.json").read_text())
        got = model.predict(example)
        for k, v in expected.items():
            self.assertAlmostEqual(got[k], v, places=4)

        # 6. predecir con texto humano nuevo — invariante 1, nunca ids
        out = model.predict({seq_name: "el producto llegó roto"})
        self.assertEqual(set(out), {"NEG", "POS"})
        self.assertAlmostEqual(sum(out.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
