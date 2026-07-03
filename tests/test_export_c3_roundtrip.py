"""EXPORT Modelo descargable — Corte 3: garantía de corrección (round-trip).

Cierre DURO del contrato. Demuestra, a NIVEL DE USUARIO, que el paquete descargado
predice lo mismo que el Studio sobre el MISMO registro CRUDO. Cubre escalar con rango,
one-hot y embedding.

Para cada codificación se comprueba, de forma NO circular:
  1. Vector: la codificación de predict.py de un registro crudo coincide con la
     normalización del Studio calculada de forma independiente — escalares con la función
     real `_normalize_csv_with_ranges`, categóricas por definición (one-hot=1 / índice).
  2. Extremo a extremo: ONNX(encode_predict(crudo)) ≈ runtime_de_referencia(norm_studio(crudo))
     dentro de atol 1e-5 (combina con la equivalencia ONNX↔runtime ya probada en P15/M2v2).

Distinto del equivalence ONNX existente, que compara con vectores YA normalizados: aquí el
punto de partida es el registro crudo y se ejerce todo el pipeline de normalización.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from importlib import util
from pathlib import Path

import unittest

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None

_ATOL = 1e-5


def _studio_scalar(value, lo, hi) -> float:
    """Authoritative Studio scalar normalization (the real training-time function)."""
    from matrixai.playground import _normalize_csv_with_ranges
    csv = f"v\n{value}\n"
    return float(_normalize_csv_with_ranges(csv, {"v": (lo, hi)}).strip().splitlines()[1])


def _load_predict(bundle_dir: Path):
    """Import the bundle's own predict.py (as a downloaded consumer would).

    Returns (model, module): the module's own MatrixAIModelError is a distinct class
    from the core's, so callers must use the one from the loaded module.
    """
    spec = util.spec_from_file_location("predict_dl", str(bundle_dir / "predict.py"))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MatrixAIModel(str(bundle_dir / "inference_spec.json"))
    return model, module


def _onnx_raw(model, vector):
    import numpy as np
    x = np.array([vector], dtype=np.float32)
    return list(model.session.run(None, {model.input_name: x})[0][0])


@unittest.skipUnless(_HAS_ONNX and _HAS_ORT, "onnx + onnxruntime required")
class _RoundTripBase(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)

    def _bundle(self, mxai, ps, **meta):
        from matrixai.parameters import write_parameter_set
        from matrixai.export import create_edge_bundle
        params = self.td / "params.json"
        write_parameter_set(str(params), ps)
        mxai_path = self.td / "model.mxai"
        mxai_path.write_text(mxai, encoding="utf-8")
        result = create_edge_bundle(
            self.prog, ps, mxai_path=str(mxai_path), params_path=str(params),
            outdir=str(self.td / "bundle"), validate=False, **meta,
        )
        self.assertIsNone(result.inference_spec_skipped_reason)
        model, module = _load_predict(Path(result.bundle_dir))
        self.predict_error = module.MatrixAIModelError
        return model

    def _assert_roundtrip(self, model, raw, v_studio, ref_raw):
        import numpy as np
        v_predict = model._encode(raw)[0]
        # 1. normalization equivalence (non-circular: independent studio normalization)
        self.assertTrue(
            np.allclose(v_predict, v_studio, atol=_ATOL),
            f"predict.py encoding {v_predict} != studio normalization {v_studio}",
        )
        # 2. end-to-end (raw onnx): onnx(predict encoding) ≈ reference runtime(studio norm)
        pred_raw = _onnx_raw(model, v_predict)
        self.assertTrue(
            np.allclose(pred_raw, ref_raw, atol=_ATOL),
            f"predict.py output {pred_raw} != studio runtime output {ref_raw}",
        )
        # 3. user-level: the PUBLIC predict() (decoded/labelled) matches the reference
        # output decoded the same way. This is the literal "downloaded model predicts
        # what the Studio predicts" guarantee.
        expected = model._decode(np.asarray(ref_raw))
        self._assert_decoded_close(model.predict(raw), expected)

    def _assert_decoded_close(self, actual, expected):
        import numpy as np
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            self.assertTrue(np.allclose([actual[k] for k in expected],
                                        [expected[k] for k in expected], atol=_ATOL),
                            f"decoded {actual} != {expected}")
        else:
            self.assertTrue(np.allclose(actual, expected, atol=_ATOL),
                            f"decoded {actual} != {expected}")


class ScalarRangeRoundTripTest(_RoundTripBase):
    MXAI = """PROJECT ScalarRange
VECTOR Patient[3]
  edad: Scalar
  imc: Scalar
  tension: Scalar
END
NETWORK Net
  INPUT Patient
  LAYER Dense units=6 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[BAJO, ALTO]
END
GRAPH
  Patient -> Net
END
"""
    RANGES = {"edad": (0, 120), "imc": (10, 70), "tension": (0, 250)}

    def test_roundtrip(self):
        import numpy as np
        from matrixai.parser import parse_text
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export.equivalence import _run_numpy_dense
        self.prog = parse_text(self.MXAI)
        ps = build_initial_parameter_set(self.prog)
        model = self._bundle(self.MXAI, ps, field_ranges=self.RANGES)

        raw = {"edad": 60, "imc": 40, "tension": 125}
        order = self.prog.vectors[0].fields
        v_studio = [_studio_scalar(raw[c], *self.RANGES[c]) for c in order]
        ref_raw = _run_numpy_dense(self.prog.networks[0], ps, np.array(v_studio, dtype=np.float32), np)
        self._assert_roundtrip(model, raw, v_studio, ref_raw)
        # decode is a labelled classification
        self.assertEqual(set(model.predict(raw)), {"BAJO", "ALTO"})

    def test_out_of_range_clipped_consistently(self):
        import numpy as np
        from matrixai.parser import parse_text
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export.equivalence import _run_numpy_dense
        self.prog = parse_text(self.MXAI)
        ps = build_initial_parameter_set(self.prog)
        model = self._bundle(self.MXAI, ps, field_ranges=self.RANGES)
        # edad 999 is far above the range -> both sides clip to 1.0
        raw = {"edad": 999, "imc": 40, "tension": 125}
        order = self.prog.vectors[0].fields
        v_studio = [_studio_scalar(raw[c], *self.RANGES[c]) for c in order]
        self.assertEqual(v_studio[0], 1.0)
        ref_raw = _run_numpy_dense(self.prog.networks[0], ps, np.array(v_studio, dtype=np.float32), np)
        self._assert_roundtrip(model, raw, v_studio, ref_raw)


class OneHotRoundTripTest(_RoundTripBase):
    def _run_case(self, group, categories, raw_category):
        """Build a one-hot dense model over `categories`, round-trip `raw_category`."""
        import numpy as np
        from matrixai.parser import parse_text
        from matrixai.parameters import build_initial_parameter_set
        from matrixai.export.equivalence import _run_numpy_dense
        from matrixai.training.categorical import _build_group_names

        cols = _build_group_names(group, categories)
        col_lines = "\n".join(f"  {c}: Scalar" for c in cols)
        mxai = f"""PROJECT OneHot
VECTOR Sample[{len(cols) + 1}]
{col_lines}
  size: Scalar
END
NETWORK Net
  INPUT Sample
  LAYER Dense units=6 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[NO, SI]
END
GRAPH
  Sample -> Net
END
"""
        self.prog = parse_text(mxai)
        ps = build_initial_parameter_set(self.prog)
        model = self._bundle(mxai, ps, field_ranges={"size": (0, 100)},
                             field_categories={group: categories})

        raw = {group: raw_category, "size": 50}
        order = self.prog.vectors[0].fields
        chosen_col = cols[categories.index(raw_category)]
        v_studio = [0.0] * len(order)
        v_studio[order.index(chosen_col)] = 1.0
        v_studio[order.index("size")] = _studio_scalar(50, 0, 100)
        ref_raw = _run_numpy_dense(self.prog.networks[0], ps, np.array(v_studio, dtype=np.float32), np)
        self._assert_roundtrip(model, raw, v_studio, ref_raw)
        return model

    def test_roundtrip(self):
        model = self._run_case("color", ["red", "green", "blue"], "green")
        # unknown category must fail loudly, not silently misencode
        with self.assertRaises(self.predict_error):
            model.predict({"color": "violet", "size": 50})

    def test_roundtrip_accents_and_spaces(self):
        # raw value with accents/spaces -> exact sanitized column -> ONNX, all the way.
        self._run_case("especialidad",
                       ["CARDIOLOGÍA", "UCI", "Médico de Familia"],
                       "Médico de Familia")


class EmbeddingRoundTripTest(_RoundTripBase):
    def _build_model(self, *, human_vocab):
        """Composite EMBEDDING model. With human_vocab the spec uses a vocab list;
        without it, vocab_size only (the consumer must send an integer index)."""
        from matrixai.parser import parse_text
        from matrixai.parameters.store import program_hash
        from matrixai.parameters.network_params import build_composite_network_parameter_set
        from matrixai.types import check_composite_network_types
        from matrixai.training.composite_generator import CompositeNetworkGenerator

        # GEN C5: vocab must exceed _ONEHOT_MAX (12) for a caller-declared
        # categorical to become an embedding (below it is one-hot territory).
        gen = CompositeNetworkGenerator().generate(
            "Clasificar bajo medio alto con bloques residuales",
            categorical_fields={"categoria": 16}, force_residual=True,
            input_fields=["categoria", "precio"],
        )
        mxai = gen.mxai_text
        self.prog = parse_text(mxai)
        self.net = self.prog.networks[0]
        # sanity: this is genuinely an embedding model
        self.assertEqual(self.prog.vectors[0].field_types["categoria"].name, "Integer")
        self.assertTrue(any(e.source == "categoria" for e in self.net.embeddings))

        vmap = {v.name: v for v in self.prog.vectors}
        tr = check_composite_network_types(self.net, vmap)
        self.ps = build_composite_network_parameter_set(
            self.net, tr, model_hash_str=program_hash(self.prog), seed=11)
        meta = {"field_ranges": {"precio": (0, 1000)}}
        if human_vocab is not None:
            meta["field_categories"] = {"categoria": human_vocab}
        return self._bundle(mxai, self.ps, **meta)

    def _ref(self, v_studio):
        import numpy as np
        from matrixai.export.equivalence import _run_composite_forward
        return _run_composite_forward(self.net, self.ps, self.prog.vectors[0],
                                      np.array(v_studio, dtype=np.float32))

    def _v_studio(self, categoria_index, precio):
        order = self.prog.vectors[0].fields
        v = [0.0] * len(order)
        v[order.index("categoria")] = float(categoria_index)
        v[order.index("precio")] = _studio_scalar(precio, 0, 1000)
        return v

    def test_roundtrip_human_vocab(self):
        vocab = [f"c{i}" for i in range(16)]
        model = self._build_model(human_vocab=vocab)
        raw = {"categoria": "c5", "precio": 500}
        v_studio = self._v_studio(vocab.index("c5"), 500)
        self._assert_roundtrip(model, raw, v_studio, self._ref(v_studio))
        with self.assertRaises(self.predict_error):
            model.predict({"categoria": "nope", "precio": 500})

    def test_roundtrip_vocab_size_only(self):
        # no human vocab -> spec carries vocab_size; consumer sends the integer index.
        model = self._build_model(human_vocab=None)
        spec_field = json.loads((self.td / "bundle" / "inference_spec.json").read_text())["fields"]["categoria"]
        self.assertIn("vocab_size", spec_field)
        self.assertNotIn("vocab", spec_field)
        raw = {"categoria": 5, "precio": 500}
        v_studio = self._v_studio(5, 500)
        self._assert_roundtrip(model, raw, v_studio, self._ref(v_studio))
        # out-of-range index must fail loudly
        with self.assertRaises(self.predict_error):
            model.predict({"categoria": 99, "precio": 500})
