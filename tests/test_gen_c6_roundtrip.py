"""GENERACIÓN tipos del prompt — C6: round-trip de nivel de usuario + invariante 6.

Ciclo completo SIN editor de esquema: prompt con tipos declarados → generar →
dataset sintético (la metadata del generador viaja) → entrenar (breve) → export
Edge bundle → importar `predict.py` → predecir con VALORES HUMANOS ("TORNO",
"si", 65). Cuatro arquetipos: one-hot (C2), embedding alta-card (C5), boolean/
integer/rango (C3) y binario etiquetado (C4).

También el invariante 6 (consistencia .mxai ↔ metadata antes de empaquetar):
las contradicciones (grupo one-hot parcial/ausente, vocab ≠ tabla de embedding,
categórica que quedó escalar) fallan el spec con razón visible en vez de
producir un bundle que predice mal o miente al consumidor.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from importlib import util
from pathlib import Path

from matrixai.parser import parse_text
from matrixai.parameters import build_initial_parameter_set, write_parameter_set
from matrixai.parameters.store import ParameterSet
from matrixai.playground import (
    analyze_playground_request,
    _generate_synthetic_dataset,
    _normalize_csv_with_ranges,
    _run_playground_training,
)


def _onnx_available() -> bool:
    return util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime required")
class _UserRoundTripBase(unittest.TestCase):
    """Drive the exact flow a Studio user gets, via the core public surface."""

    def _roundtrip(self, prompt: str, labels_expected: list[str] | None = None):
        # 1. generar — the type metadata comes from the generator (invariante 4)
        res = analyze_playground_request({"mode": "prompt", "prompt": prompt, "use_llm": False})
        self.assertTrue(res.get("ok"), res.get("error"))
        mxai, training = res["mxai"], res["training_text"]
        fc, fr, ft = res["field_categories"], res["field_ranges"], res["field_types"]

        # 2. dataset sintético — the generator metadata travels (no schema editor)
        ds = _generate_synthetic_dataset(
            mxai, training, 60, 7, "random",
            field_ranges_override=fr, field_types=ft, field_categories=fc,
        )
        self.assertTrue(ds.get("ok"), ds.get("error"))

        # 3. entrenar (breve) — CSV is domain-scale; normalize at the boundary
        csv_norm = _normalize_csv_with_ranges(
            ds["csv_text"], {k: tuple(v) for k, v in ds["field_ranges"].items()}
        )
        tr = _run_playground_training(mxai, training, csv_norm, epochs_override=3)
        self.assertTrue(tr.get("ok"), tr.get("error"))
        ps = ParameterSet.from_dict(tr["params_best"])

        # 4. exportar Edge bundle con la MISMA metadata
        from matrixai.export import create_edge_bundle
        prog = parse_text(mxai)
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(mxai, encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)
        result = create_edge_bundle(
            prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
            outdir=str(td / "b"), validate=False,
            field_ranges=fr, field_types=ft, field_categories=fc,
            labels=labels_expected,
        )
        self.assertIsNone(result.inference_spec_skipped_reason)

        # 5. importar el bundle — predict.py standalone, sin MatrixAI
        bd = Path(result.bundle_dir)
        m = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(m)
        m.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        # expected_output.json se reproduce con example_input.json
        example = json.loads((bd / "example_input.json").read_text())
        expected = json.loads((bd / "expected_output.json").read_text())
        got = model.predict(example)
        if isinstance(expected, dict):
            for k, v in expected.items():
                self.assertAlmostEqual(got[k], v, places=4)
        return model, bd, mod


class OneHotUserRoundTripTest(_UserRoundTripBase):
    def test_onehot_prompt_to_prediction_with_human_values(self):
        model, bd, mod = self._roundtrip(
            "clasificar estado de maquinas\nFEATURES:\n"
            "  temperatura: Scalar en [0, 150]\n"
            "  vibracion: Scalar en [0, 50]\n"
            "  tipo_maquina: Categorical[TORNO, FRESADORA, PRENSA]\n"
            "SALIDA: estado: ProbabilityMap[OK, FALLO]",
            labels_expected=["ok", "fallo"],
        )
        out = model.predict({"temperatura": 90, "vibracion": 12, "tipo_maquina": "TORNO"})
        self.assertEqual(set(out), {"ok", "fallo"})
        self.assertAlmostEqual(sum(out.values()), 1.0, places=5)
        with self.assertRaises(mod.MatrixAIModelError):
            model.predict({"temperatura": 90, "vibracion": 12, "tipo_maquina": "SIERRA"})


class EmbeddingUserRoundTripTest(_UserRoundTripBase):
    def test_highcard_prompt_routes_to_embedding_and_accepts_names(self):
        values = ", ".join(f"M{i}" for i in range(15))
        model, bd, _mod = self._roundtrip(
            "clasificar maquinas\nFEATURES:\n"
            "  temperatura: Scalar en [0, 150]\n"
            f"  modelo: Categorical[{values}]\n"
            "SALIDA: estado: ProbabilityMap[OK, FALLO]",
            labels_expected=["ok", "fallo"],
        )
        spec = json.loads((bd / "inference_spec.json").read_text())
        # GEN C5: high-card prompt categorical → embedding con vocab HUMANO
        self.assertEqual(spec["fields"]["modelo"]["encoding"], "embedding_index")
        self.assertEqual(spec["fields"]["modelo"]["vocab"], [f"M{i}" for i in range(15)])
        out = model.predict({"temperatura": 70, "modelo": "M7"})
        self.assertAlmostEqual(sum(out.values()), 1.0, places=5)


class BooleanIntegerUserRoundTripTest(_UserRoundTripBase):
    def test_boolean_and_range_prompt_accepts_human_values(self):
        model, bd, mod = self._roundtrip(
            "clasificacion binaria de abandono\nFEATURES:\n"
            "  edad: Scalar en [18, 95]\n"
            "  saldo: Scalar en [0, 500000]\n"
            "  tiene_hipoteca: Boolean\n"
            "  cod_region: Integer[0, 10]\n",
        )
        out = model.predict({"edad": 56, "saldo": 12000, "tiene_hipoteca": "si",
                             "cod_region": 3})
        self.assertIsInstance(out, float)  # sigmoide binario retrocompat
        with self.assertRaises(mod.MatrixAIModelError):
            model.predict({"edad": 56, "saldo": 12000, "tiene_hipoteca": "si",
                           "cod_region": 3.7})  # integer rechaza fraccionario


class LabeledBinaryUserRoundTripTest(_UserRoundTripBase):
    def test_probabilitymap_two_labels_gives_named_distribution(self):
        model, bd, _mod = self._roundtrip(
            "clasificar abandono de clientes con red densa\nFEATURES:\n"
            "  edad: Scalar en [18, 95]\n"
            "  saldo: Scalar en [0, 500000]\n"
            "SALIDA: resultado: ProbabilityMap[PERMANECE, ABANDONA]",
            labels_expected=["permanece", "abandona"],
        )
        spec = json.loads((bd / "inference_spec.json").read_text())
        self.assertEqual(spec["output"]["kind"], "classification")
        self.assertEqual(spec["output"]["labels"], ["permanece", "abandona"])
        out = model.predict({"edad": 44, "saldo": 90000})
        self.assertEqual(set(out), {"permanece", "abandona"})
        self.assertAlmostEqual(sum(out.values()), 1.0, places=5)


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime required")
class Invariant6ConsistencyTest(unittest.TestCase):
    """Invariante 6: metadata que contradice el .mxai falla el spec con razón
    visible (el bundle sigue construyéndose, pero nunca miente al consumidor)."""

    def _bundle(self, mxai: str, **meta):
        from matrixai.export import create_edge_bundle
        prog = parse_text(mxai)
        if any(getattr(n, "kind", "") == "composite_network" for n in prog.networks):
            # the generic builder only handles dense/function programs
            from matrixai.types import check_composite_network_types
            from matrixai.parameters.network_params import build_composite_network_parameter_set
            from matrixai.parameters.store import program_hash
            net = next(n for n in prog.networks
                       if getattr(n, "kind", "") == "composite_network")
            vmap = {v.name: v for v in prog.vectors}
            ps = build_composite_network_parameter_set(
                net, check_composite_network_types(net, vmap),
                model_hash_str=program_hash(prog))
        else:
            ps = build_initial_parameter_set(prog)
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(mxai, encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            return create_edge_bundle(
                prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
                outdir=str(td / "b"), validate=False, **meta,
            )

    def _onehot_mxai(self):
        from matrixai.training.dense_generator import DenseNetworkGenerator
        return DenseNetworkGenerator().generate(
            "clasificar\nFEATURES:\n  temp: Scalar en [0, 100]\n"
            "  tipo: Categorical[A, B, C]\nSALIDA: y: ProbabilityMap[OK, KO]"
        ).mxai_text

    def test_partial_onehot_group_fails_the_spec(self):
        # metadata declares 4 values but the model has 3 one-hot columns
        r = self._bundle(self._onehot_mxai(),
                         field_categories={"tipo": ["A", "B", "C", "D"]},
                         labels=["ok", "ko"])
        self.assertIsNotNone(r.inference_spec_skipped_reason)
        self.assertIn("tipo", r.inference_spec_skipped_reason)

    def test_categories_for_scalar_field_fail_the_spec(self):
        # 'temp' exists but is a plain scalar — declaring categories contradicts it
        r = self._bundle(self._onehot_mxai(),
                         field_categories={"tipo": ["A", "B", "C"],
                                           "temp": ["BAJO", "ALTO"]},
                         labels=["ok", "ko"])
        self.assertIsNotNone(r.inference_spec_skipped_reason)
        self.assertIn("temp", r.inference_spec_skipped_reason)

    def test_unknown_categorical_fails_the_spec(self):
        r = self._bundle(self._onehot_mxai(),
                         field_categories={"tipo": ["A", "B", "C"],
                                           "fantasma": ["X", "Y"]},
                         labels=["ok", "ko"])
        self.assertIsNotNone(r.inference_spec_skipped_reason)
        self.assertIn("fantasma", r.inference_spec_skipped_reason)

    def test_embedding_vocab_size_mismatch_fails_the_spec(self):
        from matrixai.training.composite_generator import CompositeNetworkGenerator
        gen = CompositeNetworkGenerator().generate(
            "clasificar\nFEATURES:\n  precio: Scalar en [0, 100]\n"
            "  categoria: Categorical[A, B, C]\nSALIDA: y: ProbabilityMap[OK, KO]",
        )
        # embedding table has 3 rows; metadata claims 5 values
        r = self._bundle(gen.mxai_text,
                         field_categories={"categoria": ["A", "B", "C", "D", "E"]},
                         labels=["ok", "ko"])
        self.assertIsNotNone(r.inference_spec_skipped_reason)
        self.assertIn("categoria", r.inference_spec_skipped_reason)

    def test_consistent_metadata_still_builds_usable_spec(self):
        r = self._bundle(self._onehot_mxai(),
                         field_categories={"tipo": ["A", "B", "C"]},
                         field_ranges={"temp": (0, 100)},
                         labels=["ok", "ko"])
        self.assertIsNone(r.inference_spec_skipped_reason)


class PreExpandedDatasetGroupsTest(unittest.TestCase):
    """GEN C6: el endpoint de dataset reconstruye los grupos one-hot de un modelo
    YA expandido en generación (C2) — el sampler emite exactamente un 1 por grupo
    y la metadata se ecoa (antes: floats aleatorios y metadata perdida)."""

    def test_sampler_respects_pre_expanded_groups(self):
        import csv as csvmod
        import io
        from matrixai.training.dense_generator import DenseNetworkGenerator
        r = DenseNetworkGenerator().generate(
            "clasificar\nFEATURES:\n  temp: Scalar en [0, 100]\n"
            "  tipo: Categorical[TORNO, FRESADORA, PRENSA]\nSALIDA: y: ProbabilityMap[OK, KO]"
        )
        ds = _generate_synthetic_dataset(
            r.mxai_text, r.training_text, 30, 7, "random",
            field_ranges_override=r.field_ranges, field_types=r.field_types,
            field_categories=r.field_categories,
        )
        self.assertTrue(ds.get("ok"), ds.get("error"))
        self.assertEqual(ds["one_hot_groups"],
                         {"tipo": ["tipo__torno", "tipo__fresadora", "tipo__prensa"]})
        self.assertEqual(ds["field_categories"], {"tipo": ["TORNO", "FRESADORA", "PRENSA"]})
        rows = list(csvmod.DictReader(io.StringIO(ds["csv_text"])))
        cols = ["tipo__torno", "tipo__fresadora", "tipo__prensa"]
        for row in rows:
            self.assertEqual(sum(float(row[c]) for c in cols), 1.0)


class EmbeddingDatasetRangeIsolationTest(unittest.TestCase):
    """GEN C6 audit: embedding source columns are integer lookup indices.

    They must not be treated as rangeable scalars by user overrides or LLM range
    suggestions; otherwise dataset generation can emit 54 for an Integer[0,14]
    embedding source, and range-normalization can hide the corruption before
    training.
    """

    def _embedding_gen(self):
        from matrixai.training.composite_generator import CompositeNetworkGenerator
        values = ", ".join(f"M{i}" for i in range(15))
        return CompositeNetworkGenerator().generate(
            "clasificar\nFEATURES:\n  precio: Scalar\n"
            f"  modelo: Categorical[{values}]\nSALIDA: y: ProbabilityMap[OK, KO]"
        )

    def test_user_range_for_embedding_source_is_ignored(self):
        import csv as csvmod
        import io
        gen = self._embedding_gen()
        ds = _generate_synthetic_dataset(
            gen.mxai_text, gen.training_text, 20, 7, "random",
            field_categories=gen.field_categories,
            field_ranges_override={"modelo": (0, 100)},
        )
        self.assertTrue(ds.get("ok"), ds.get("error"))
        self.assertNotIn("modelo", ds["field_ranges"])
        rows = list(csvmod.DictReader(io.StringIO(ds["csv_text"])))
        vals = [float(r["modelo"]) for r in rows]
        self.assertTrue(all(v.is_integer() and 0 <= v <= 14 for v in vals), vals)

    def test_llm_ranges_do_not_include_embedding_sources(self):
        import csv as csvmod
        import io
        from unittest.mock import patch
        gen = self._embedding_gen()
        seen: list[str] = []

        def fake_ranges(fields, context=""):
            seen.extend(fields)
            return {"precio": (0, 100), "modelo": (0, 100)}

        with patch("matrixai.playground._detect_llm_mode",
                   return_value={"active": True, "model": "x", "provider": "test"}), \
             patch("matrixai.playground._llm_field_ranges", side_effect=fake_ranges):
            ds = _generate_synthetic_dataset(
                gen.mxai_text, gen.training_text, 20, 7, "random", use_llm=True,
                field_categories=gen.field_categories,
            )
        self.assertTrue(ds.get("ok"), ds.get("error"))
        self.assertIn("precio", seen)
        self.assertNotIn("modelo", seen)
        self.assertEqual(ds["field_ranges"], {"precio": [0, 100]})
        rows = list(csvmod.DictReader(io.StringIO(ds["csv_text"])))
        vals = [float(r["modelo"]) for r in rows]
        self.assertTrue(all(v.is_integer() and 0 <= v <= 14 for v in vals), vals)


class BooleanRangeIsolationTest(unittest.TestCase):
    """GEN C6 audit: boolean columns are 0/1 flags — never rangeable.

    A user/LLM range on a prompt-declared Boolean poisoned the echoed
    field_ranges: training normalization squashed the flag (1 → 0.01 with
    [0,100]) and the export rejected the spec (invariante 6) — the C6 promise
    broke exactly when the LLM was active. Booleans keep their type annotation
    and stay out of the range machinery entirely.
    """

    PROMPT = (
        "clasificacion binaria de abandono\nFEATURES:\n"
        "  edad: Scalar en [18, 95]\n"
        "  tiene_hipoteca: Boolean\n"
    )

    def _generated(self):
        res = analyze_playground_request(
            {"mode": "prompt", "prompt": self.PROMPT, "use_llm": False})
        self.assertTrue(res.get("ok"), res.get("error"))
        return res

    def test_llm_ranges_do_not_include_booleans(self):
        import csv as csvmod
        import io
        from unittest.mock import patch
        res = self._generated()
        seen: list[str] = []

        def fake_ranges(fields, context=""):
            seen.extend(fields)
            return {f: (0.0, 100.0) for f in fields}

        with patch("matrixai.playground._detect_llm_mode",
                   return_value={"active": True, "model": "x", "provider": "test"}), \
             patch("matrixai.playground._llm_field_ranges", side_effect=fake_ranges):
            ds = _generate_synthetic_dataset(
                res["mxai"], res["training_text"], 20, 7, "random", use_llm=True,
                field_ranges_override=res["field_ranges"],
                field_types=res["field_types"],
                field_categories=res["field_categories"],
            )
        self.assertTrue(ds.get("ok"), ds.get("error"))
        self.assertNotIn("tiene_hipoteca", seen)
        self.assertNotIn("tiene_hipoteca", ds["field_ranges"])
        # the type annotation still travels (it is metadata, not a range)
        self.assertEqual(ds["field_types"], {"tiene_hipoteca": "boolean"})
        rows = list(csvmod.DictReader(io.StringIO(ds["csv_text"])))
        self.assertTrue(all(r["tiene_hipoteca"] in {"0", "0.0", "1", "1.0"} for r in rows))
        # end-to-end: the echoed metadata still yields a usable spec (before the
        # fix, the boolean range made the export drop it with invariante 6)
        if _onnx_available():
            from matrixai.export import create_edge_bundle
            prog = parse_text(res["mxai"])
            ps = build_initial_parameter_set(prog)
            td = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, str(td), True)
            (td / "m.mxai").write_text(res["mxai"], encoding="utf-8")
            write_parameter_set(str(td / "p.json"), ps)
            r = create_edge_bundle(
                prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
                outdir=str(td / "b"), validate=False,
                field_ranges={k: tuple(v) for k, v in ds["field_ranges"].items()},
                field_types=ds["field_types"],
                field_categories=ds["field_categories"],
            )
            self.assertIsNone(r.inference_spec_skipped_reason)

    def test_user_range_for_boolean_is_ignored(self):
        res = self._generated()
        fr = dict(res["field_ranges"])
        fr["tiene_hipoteca"] = (0.0, 100.0)
        ds = _generate_synthetic_dataset(
            res["mxai"], res["training_text"], 20, 7, "random",
            field_ranges_override=fr,
            field_types=res["field_types"],
            field_categories=res["field_categories"],
        )
        self.assertTrue(ds.get("ok"), ds.get("error"))
        self.assertNotIn("tiene_hipoteca", ds["field_ranges"])
        self.assertIn("edad", ds["field_ranges"])


class ArchKindReflectsEmittedNetworkTest(unittest.TestCase):
    """GEN C6 (deuda de Colab): architecture_decision.kind refleja la red
    REALMENTE emitida, no la intención de ruteo — un ruteo composite que acaba
    en densa pura ya no se rotula 'composite (embeddings)'."""

    def test_composite_routing_that_emits_dense_is_labelled_dense(self):
        from unittest.mock import patch
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={"labels": ["A", "B"],
                                 "categorical_fields": {"zona": 30}}):
            # 'zona' es huérfana (el LLM no dio input_fields) → el dispatch la
            # descarta → el composite emite una red DENSA pura.
            res = analyze_playground_request(
                {"mode": "prompt", "prompt": "clasificar riesgo de clientes", "use_llm": True}
            )
        self.assertTrue(res.get("ok"), res.get("error"))
        self.assertNotIn("EMBEDDING", res["mxai"])
        self.assertEqual(res["architecture_decision"]["kind"], "dense")

    def test_prompt_highcard_labels_composite(self):
        values = ", ".join(f"V{i}" for i in range(15))
        res = analyze_playground_request({
            "mode": "prompt",
            "prompt": f"clasificar\nFEATURES:\n  x: Scalar en [0, 1]\n"
                      f"  modelo: Categorical[{values}]\nSALIDA: y: ProbabilityMap[A, B]",
            "use_llm": False,
        })
        self.assertIn("EMBEDDING", res["mxai"])
        self.assertEqual(res["architecture_decision"]["kind"], "composite")
        self.assertEqual(res["architecture_decision"]["source"], "prompt_types")
