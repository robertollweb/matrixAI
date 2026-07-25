# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 61 C5 — paridad de normalización entre `/api/train` (síncrono) y
`/api/train-start` (asíncrono).

Antes de este corte, `/api/train` -> `_run_playground_training` NO recibía
`field_ranges`: un CSV en escala de dominio (presión ~1000 junto a una señal
~1) llegaba CRUDO al trainer y la red colapsaba a una sola clase, entrenando
en un espacio distinto al que usa la inferencia (`predict.py` normaliza con los
rangos). `/api/train-start` sí normalizaba desde M5. El fallo se reprodujo por
la ruta síncrona; la ruta del Studio (`/api/train-start`) nunca lo tuvo.

Estos tests NO se apoyan en `accuracy > baseline` ni SOLO en `model_collapsed`
(que sondea salida constante en espacio [0,1] y puede dar False para un modelo
entrenado en dominio que aun así predice una única clase — ver contrato §pega).
Exigen, como manda C5: ambas clases predichas, matriz de confusión NO
degenerada y macro-F1 muy por encima del predictor constante.
"""
from __future__ import annotations

import io
import json
import time
import unittest
from http.client import HTTPMessage
from unittest.mock import patch

from matrixai.playground import (
    _compose_normalize_ranges,
    _get_job_status,
    _handler_class,
    _normalize_input_with_ranges,
    _playground_run_with_params,
    _run_playground_training,
    _submit_training_job,
)
from matrixai.training.dataset_project import generate_project_from_dataset


def _kelvin_project():
    """Regresión C→K (relación lineal exacta) con rango de target de dominio."""
    csv = "centigrados,prediccionKelvin\n" + "\n".join(
        f"{c},{c + 273.15}" for c in range(100)) + "\n"
    proj = generate_project_from_dataset(
        csv, "prediccionKelvin",
        column_type_overrides={"centigrados": "number"},
        column_range_overrides={"centigrados": (0.0, 99.0)},
    )
    mxai = proj["mxai"]["text"] if isinstance(proj["mxai"], dict) else proj["mxai"]
    fr = {k: tuple(v) for k, v in (proj.get("field_ranges") or {}).items()}
    return proj, mxai, fr, tuple(proj["target_range"])


def _disparate_scale_csv(n: int = 90) -> str:
    """Clasificación con escalas MUY dispares: `pressure`/`volume` en cientos
    (sin señal) y `signal` en [0,1] (determina la etiqueta). Sin normalizar,
    las features grandes dominan gradientes/activaciones y el modelo no puede
    aprender de `signal` -> colapsa a una clase. Con `field_ranges` todo entra
    en [0,1] y la relación es trivial de aprender."""
    lines = ["pressure,volume,signal,label"]
    for i in range(n):
        pressure = 1000 + (i % 40)
        volume = 500 + ((i * 3) % 60)
        signal = ((i * 7) % 100) / 100.0
        label = "pos" if signal >= 0.5 else "neg"
        lines.append(f"{pressure}.5,{volume}.2,{signal:.2f},{label}")
    return "\n".join(lines) + "\n"


def _project():
    proj = generate_project_from_dataset(_disparate_scale_csv(), "label")
    mxai = proj["mxai"]["text"] if isinstance(proj["mxai"], dict) else proj["mxai"]
    fr = {k: tuple(v) for k, v in (proj.get("field_ranges") or {}).items()}
    return proj, mxai, fr


def _pred_totals(confusion: dict) -> dict:
    """Suma por CLASE PREDICHA — un total en cero = clase nunca predicha."""
    totals: dict = {}
    for _true, row in confusion.items():
        for pred, count in row.items():
            totals[pred] = totals.get(pred, 0) + count
    return totals


class TestApiTrainNormalizationParity(unittest.TestCase):
    """El fix conductual: la ruta síncrona con `field_ranges` aprende ambas
    clases; sin ellos, degenera (control negativo que prueba que el fix hace
    falta y no es cosmético)."""

    def test_domain_scale_csv_learns_both_classes_with_ranges(self):
        proj, mxai, fr = _project()
        # las 3 features llevan rango; ninguna llega cruda al trainer
        self.assertEqual(set(fr), {"pressure", "volume", "signal"})
        res = _run_playground_training(
            mxai, proj["training_text"], proj["csv_text"],
            epochs_override=15, field_ranges=fr,
        )
        self.assertTrue(res.get("ok"), res.get("error"))
        cm = res.get("confusion_matrix") or res.get("confusion")
        totals = _pred_totals(cm)
        self.assertEqual(len(totals), 2, f"deberían predecirse 2 clases, cm={cm}")
        self.assertTrue(all(v > 0 for v in totals.values()),
                        f"alguna clase nunca se predice (colapso): cm={cm}")
        self.assertGreater(res["macro_f1"], 0.75,
                           f"macro-F1={res['macro_f1']} — debería superar de largo al predictor constante")

    def test_normalizing_learns_both_classes_and_never_hurts(self):
        """Comparativo robusto (no un control negativo frágil): con el MISMO
        proyecto, normalizar con `field_ranges` produce un resultado NO
        degenerado (ambas clases predichas) y con macro-F1 >= al de entrenar en
        crudo. NO se fija una magnitud de mejora ni se afirma que el trainer en
        crudo DEBA colapsar: una mejora legítima del optimizador podría aprender
        aun sin normalizar, y este test debe seguir pasando en ese caso. Lo que
        el contrato garantiza es que normalizar nunca empeora y que, con rangos,
        el modelo aprende de verdad."""
        proj, mxai, fr = _project()
        without = _run_playground_training(
            mxai, proj["training_text"], proj["csv_text"], epochs_override=15,
        )
        with_ranges = _run_playground_training(
            mxai, proj["training_text"], proj["csv_text"],
            epochs_override=15, field_ranges=fr,
        )
        self.assertTrue(without.get("ok") and with_ranges.get("ok"))
        totals = _pred_totals(with_ranges.get("confusion_matrix") or with_ranges.get("confusion"))
        self.assertTrue(len(totals) == 2 and all(v > 0 for v in totals.values()),
                        "con rangos el modelo debe predecir ambas clases")
        self.assertGreater(with_ranges["macro_f1"], 0.75,
                           "con rangos el modelo debe superar de largo al predictor constante")
        self.assertGreaterEqual(
            round(with_ranges["macro_f1"], 6), round(without["macro_f1"], 6),
            "normalizar nunca debería reducir el macro-F1",
        )


class TestApiTrainRouteThreadsFieldRanges(unittest.TestCase):
    """Wiring del endpoint: `/api/train` enhebra `field_ranges` a
    `_run_playground_training`, igual que `/api/train-start` (invariante 10).
    No entrena de verdad — captura los kwargs para probar solo el cableado."""

    def _handler(self, path: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        handler = object.__new__(_handler_class(None))
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.path = path
        handler.command = "POST"
        handler.request_version = "HTTP/1.1"
        handler.headers = HTTPMessage()
        handler.headers["Content-Length"] = str(len(body))
        handler.close_connection = True
        return handler

    def test_train_route_passes_coerced_field_ranges(self):
        captured: dict = {}

        def _fake_train(mxai_text, training_text, csv_text, epochs_override=None,
                        field_ranges=None, target_range=None):
            captured["field_ranges"] = field_ranges
            return {"ok": True}

        payload = {
            "mxai_text": "x", "training_text": "y", "csv_text": "z",
            "field_ranges": {"pressure": [996.6, 1043.4], "signal": [-0.1, 1.09]},
        }
        handler = self._handler("/api/train", payload)
        with patch("matrixai.playground._run_playground_training", _fake_train), \
                patch.object(handler, "_send_json", lambda *a, **k: None):
            handler.do_POST()
        self.assertEqual(
            captured.get("field_ranges"),
            {"pressure": (996.6, 1043.4), "signal": (-0.1, 1.09)},
            "el route debe coaccionar y pasar field_ranges (paridad con train-start)",
        )

    def test_train_route_without_field_ranges_passes_none(self):
        """Retrocompat: un caller viejo sin `field_ranges` sigue funcionando
        (None, nunca un error nuevo)."""
        captured: dict = {"field_ranges": "SENTINEL"}

        def _fake_train(mxai_text, training_text, csv_text, epochs_override=None,
                        field_ranges=None, target_range=None):
            captured["field_ranges"] = field_ranges
            return {"ok": True}

        handler = self._handler(
            "/api/train", {"mxai_text": "x", "training_text": "y", "csv_text": "z"})
        with patch("matrixai.playground._run_playground_training", _fake_train), \
                patch.object(handler, "_send_json", lambda *a, **k: None):
            handler.do_POST()
        self.assertIsNone(captured.get("field_ranges"))


class TestComposeNormalizeRanges(unittest.TestCase):
    """Frontera única compartida por async y síncrono: compone features + la
    columna REAL del target (nunca asumida) y es best-effort ante mxai roto."""

    def test_feature_ranges_pass_through(self):
        out = _compose_normalize_ranges("", {"a": (0.0, 10.0)}, None)
        self.assertEqual(out, {"a": (0.0, 10.0)})

    def test_target_range_maps_to_real_output_column(self):
        # un proyecto de regresión real da el nombre de salida verdadero
        proj = generate_project_from_dataset(
            "c,k\n" + "\n".join(f"{i},{i + 273.15}" for i in range(40)) + "\n",
            "k",
            column_type_overrides={"c": "number"},
            column_range_overrides={"c": (0.0, 39.0)},
        )
        mxai = proj["mxai"]["text"] if isinstance(proj["mxai"], dict) else proj["mxai"]
        tr = tuple(proj["target_range"])
        out = _compose_normalize_ranges(mxai, {"c": (0.0, 39.0)}, tr)
        self.assertIn("c", out)
        # la columna de salida (p.ej. "predicted_value") recibe el target_range
        extra = [k for k in out if k != "c"]
        self.assertEqual(len(extra), 1, f"debería añadir 1 columna de salida, out={out}")
        self.assertEqual(out[extra[0]], tr)

    def test_invalid_mxai_still_returns_feature_ranges(self):
        out = _compose_normalize_ranges("::not a program::", {"a": (0.0, 1.0)}, (5.0, 9.0))
        self.assertEqual(out, {"a": (0.0, 1.0)}, "mxai inválido no debe perder los rangos de features")


def _handler(path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    handler = object.__new__(_handler_class(None))
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.path = path
    handler.command = "POST"
    handler.request_version = "HTTP/1.1"
    handler.headers = HTTPMessage()
    handler.headers["Content-Length"] = str(len(body))
    handler.close_connection = True
    return handler


def _post(path: str, payload: dict) -> dict:
    """Ejercita el handler HTTP real (do_POST) y captura el JSON respondido."""
    captured: dict = {}
    handler = _handler(path, payload)
    handler._send_json = lambda body, status=200: captured.update(body=body, status=status)  # type: ignore[method-assign]
    handler.do_POST()
    return captured


class TestTargetRangeThroughHttpTrain(unittest.TestCase):
    """ALTA 2 (auditoría de Roberto): `target_range` por la ruta HTTP síncrona
    `/api/train`. Antes: el route no extraía `target_range` y
    `_run_playground_training` no lo pasaba a los trainers → MAE en espacio
    normalizado y `target_range=None`."""

    def test_train_route_denormalizes_regression_metrics(self):
        proj, mxai, fr, tr = _kelvin_project()
        out = _post("/api/train", {
            "mxai_text": mxai, "training_text": proj["training_text"],
            "csv_text": proj["csv_text"], "epochs_override": 60,
            "field_ranges": {k: list(v) for k, v in fr.items()},
            "target_range": list(tr),
        })
        self.assertEqual(out["status"], 200, out.get("body"))
        body = out["body"]
        self.assertEqual(body.get("task_kind"), "regression")
        self.assertEqual(tuple(body["target_range"]), tr, "target_range debe reflejarse en la respuesta")
        self.assertGreater(body["r2"], 0.99, f"r2={body.get('r2')}")
        # MAE en Kelvin real (rango ~119 K), no en espacio normalizado [0,1]
        self.assertGreater(body["mae"], 0.001, "MAE sospechosamente pequeño — ¿quedó normalizado?")
        self.assertLess(body["mae"], 5.0, "MAE demasiado grande para una relación lineal exacta")

    def test_train_route_without_target_range_leaves_it_none(self):
        proj, mxai, fr, _tr = _kelvin_project()
        out = _post("/api/train", {
            "mxai_text": mxai, "training_text": proj["training_text"],
            "csv_text": proj["csv_text"], "epochs_override": 30,
            "field_ranges": {k: list(v) for k, v in fr.items()},
        })
        self.assertEqual(out["status"], 200, out.get("body"))
        self.assertIsNone(out["body"].get("target_range"))


class TestRunWithParamsCoherence(unittest.TestCase):
    """ALTA 1 (auditoría de Roberto): coherencia train/serve en
    `/api/run-with-params`. Con `field_ranges` un input de dominio se normaliza
    igual que al entrenar; sin ellos se ejecuta crudo (incoherente)."""

    @classmethod
    def setUpClass(cls):
        proj, mxai, fr = _project()
        res = _run_playground_training(
            mxai, proj["training_text"], proj["csv_text"],
            epochs_override=15, field_ranges=fr)
        cls.mxai, cls.fr = mxai, fr
        cls.params_json = json.dumps(res["params_best"])
        cls.domain_input = json.dumps({"pressure": 1020.5, "volume": 530.2, "signal": 0.2})

    @staticmethod
    def _state(result: dict) -> dict:
        # el endpoint devuelve result["result"] como repr de dict (str) — quirk
        # preexistente; se parsea de vuelta para inspeccionarlo.
        import ast
        return ast.literal_eval(result["result"])["state"]

    def test_domain_input_with_ranges_equals_manual_normalization(self):
        with_ranges = _playground_run_with_params(
            self.mxai, self.params_json, self.domain_input, field_ranges=self.fr)
        # normalización a mano de la misma entrada (control)
        raw = {"pressure": 1020.5, "volume": 530.2, "signal": 0.2}
        manual = json.dumps({k: round(min(1.0, max(0.0, (v - self.fr[k][0]) / (self.fr[k][1] - self.fr[k][0]))), 6)
                             for k, v in raw.items()})
        manual_run = _playground_run_with_params(self.mxai, self.params_json, manual)
        self.assertTrue(with_ranges.get("ok") and manual_run.get("ok"))
        self.assertEqual(self._state(with_ranges)["predicted_class"],
                         self._state(manual_run)["predicted_class"],
                         "normalizar en el endpoint debe igualar al input normalizado a mano")

    def test_domain_input_without_ranges_is_incoherent(self):
        raw_run = _playground_run_with_params(self.mxai, self.params_json, self.domain_input)
        with_ranges = _playground_run_with_params(
            self.mxai, self.params_json, self.domain_input, field_ranges=self.fr)
        self.assertNotEqual(self._state(raw_run)["predicted_class"],
                            self._state(with_ranges)["predicted_class"],
                            "ejecutar dominio en crudo NO debe coincidir con la versión normalizada")

    def test_run_with_params_route_threads_field_ranges(self):
        out = _post("/api/run-with-params", {
            "mxai_text": self.mxai, "params_json": self.params_json,
            "input_json": self.domain_input,
            "field_ranges": {k: list(v) for k, v in self.fr.items()},
        })
        self.assertEqual(out["status"], 200, out.get("body"))
        # el input del trace debe estar normalizado a [0,1], no en dominio
        norm_input = self._state(out["body"])["Input"]
        self.assertTrue(all(0.0 <= x <= 1.0 for x in norm_input),
                        f"el route no normalizó el input de dominio: {norm_input}")


class TestNormalizeInputHelper(unittest.TestCase):
    def test_nested_and_flat_forms(self):
        fr = {"a": (0.0, 10.0), "b": (100.0, 200.0)}
        nested = _normalize_input_with_ranges({"V": {"a": 5.0, "b": 150.0}}, fr)
        self.assertEqual(nested, {"V": {"a": 0.5, "b": 0.5}})
        flat = _normalize_input_with_ranges({"a": 2.5}, fr)
        self.assertEqual(flat, {"a": 0.25})

    def test_fields_without_range_and_clamping(self):
        fr = {"a": (0.0, 10.0)}
        out = _normalize_input_with_ranges({"a": 20.0, "boolean_col": 1}, fr)
        self.assertEqual(out["a"], 1.0, "fuera de rango se recorta a 1.0")
        self.assertEqual(out["boolean_col"], 1, "columnas sin rango se dejan intactas")

    def test_no_ranges_is_identity(self):
        data = {"a": 5.0, "b": 3}
        self.assertEqual(_normalize_input_with_ranges(data, None), data)


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None


def _parse_labels(mxai: str) -> list[str]:
    import re
    m = re.search(r"ProbabilityMap\[([^\]]+)\]", mxai)
    return [s.strip() for s in m.group(1).split(",")] if m else []


@unittest.skipUnless(_onnx_available(), "onnx/onnxruntime required")
class TestFromDataExportPredictCoherence(unittest.TestCase):
    """Residuo C5: E2E del flujo desde-DATOS (`generate_project_from_dataset`) →
    entrenar (frontera C61) → export bundle → `predict.py`. Cierra la coherencia
    train/serve/export: `predict.py` (que normaliza el input de dominio con los
    `field_ranges` del `inference_spec`) debe dar la MISMA predicción que
    `/api/run-with-params` con esos mismos rangos, para una entrada de dominio."""

    def test_predict_py_matches_run_with_params_on_domain_input(self):
        import ast
        import shutil
        import tempfile
        from importlib import util
        from pathlib import Path
        from matrixai.parser import parse_text
        from matrixai.parameters import write_parameter_set
        from matrixai.parameters.store import ParameterSet
        from matrixai.export import create_edge_bundle

        proj, mxai, fr = _project()
        labels = _parse_labels(mxai)
        self.assertEqual(labels, ["neg", "pos"])
        tr = _run_playground_training(
            mxai, proj["training_text"], proj["csv_text"],
            epochs_override=15, field_ranges=fr)
        ps = ParameterSet.from_dict(tr["params_best"])

        prog = parse_text(mxai)
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(td), True)
        (td / "m.mxai").write_text(mxai, encoding="utf-8")
        write_parameter_set(str(td / "p.json"), ps)
        result = create_edge_bundle(
            prog, ps, mxai_path=str(td / "m.mxai"), params_path=str(td / "p.json"),
            outdir=str(td / "b"), validate=False, field_ranges=fr, labels=labels)
        self.assertIsNone(result.inference_spec_skipped_reason, "el bundle desde-datos debe ser autousable")

        bd = Path(result.bundle_dir)
        spec = util.spec_from_file_location(f"pred_{id(self)}", str(bd / "predict.py"))
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.MatrixAIModel(str(bd / "inference_spec.json"))

        # entrada de DOMINIO — predict.py la normaliza vía el inference_spec
        domain = {"pressure": 1020.5, "volume": 530.2, "signal": 0.2}
        got = model.predict(domain)  # {"neg": p0, "pos": p1}

        # la misma entrada de dominio por run-with-params + field_ranges
        rwp = _playground_run_with_params(
            mxai, json.dumps(tr["params_best"]), json.dumps(domain), field_ranges=fr)
        probs = ast.literal_eval(rwp["result"])["state"]["predicted_class"]
        self.assertAlmostEqual(got["neg"], probs[0], places=4)
        self.assertAlmostEqual(got["pos"], probs[1], places=4)
        self.assertAlmostEqual(got["neg"] + got["pos"], 1.0, places=5)


def _mar_rows(n: int = 60) -> str:
    lines = ["fecha,altura_ola,temperatura"]
    for d in range(1, n + 1):
        lines.append(f"2024-{(d - 1) // 28 + 1:02d}-{(d - 1) % 28 + 1:02d},{2.0 + d * 0.1:.2f},{15.0 + d * 0.05:.2f}")
    return "\n".join(lines) + "\n"


class TestTemporalFromDataNormalization(unittest.TestCase):
    """Residuo C5: flujo temporal desde-datos. Invariante 2 — los `field_ranges`
    usan los nombres FINALES tras el lag (`altura_ola_lag1/2`), no claves
    huérfanas — y la frontera C61 normaliza feature y target de la regresión
    temporal sin colapsar."""

    def test_temporal_project_trains_with_lag_aligned_ranges(self):
        from matrixai.training.dataset_project import generate_temporal_project_from_dataset
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2)
        self.assertTrue(res.get("ok"), res.get("error"))
        mxai = res["mxai"]["text"] if isinstance(res["mxai"], dict) else res["mxai"]
        fr = {k: tuple(v) for k, v in (res.get("field_ranges") or {}).items()}
        # invariante 2: rangos con los nombres finales tras el lag, no huérfanos
        self.assertIn("altura_ola_lag1", fr)
        self.assertIn("altura_ola_lag2", fr)
        self.assertNotIn("fecha", fr, "la columna temporal cruda nunca es feature/rango")
        tr = res.get("target_range")
        out = _run_playground_training(
            mxai, res["training_text"], res["csv_text"], epochs_override=40,
            field_ranges=fr, target_range=tuple(tr) if tr else None)
        self.assertTrue(out.get("ok"), out.get("error"))
        self.assertFalse(out.get("model_collapsed"), "la regresión temporal no debería colapsar")
        if tr is not None:
            self.assertEqual(tuple(out["target_range"]), tuple(tr))


class TestValidateCsvAlwaysReportsError(unittest.TestCase):
    """Auditoría reload→reentrenar: cuando el verificador rechaza el CSV (p.ej.
    valores de target `0/1` que no coinciden con las etiquetas `class_0/class_1`
    tras reimportar el CSV crudo), `_validate_training_csv` debe rellenar `error`
    con un resumen accionable, no dejarlo en None (la UI mostraba "error al
    cargar" sin causa)."""

    def test_target_value_mismatch_populates_error_summary(self):
        from matrixai.playground import _validate_training_csv
        proj, mxai, fr = _project()  # target real es `predicted_class` (class_0/1)
        # CSV con la columna renombrada pero valores SIN mapear (0/1 crudos)
        import csv as _csv
        rows = list(_csv.DictReader(io.StringIO(proj["csv_text"])))
        header = list(rows[0].keys())
        lines = [",".join(header)]
        for i, r in enumerate(rows[:20]):
            r = dict(r)
            r["predicted_class"] = str(i % 2)  # 0/1 en vez de class_0/class_1
            lines.append(",".join(str(r[c]) for c in header))
        bad_csv = "\n".join(lines) + "\n"
        res = _validate_training_csv(mxai, proj["training_text"], bad_csv, field_ranges=fr)
        self.assertFalse(res.get("ok"))
        self.assertIsNotNone(res.get("error"), "error no debe ser None cuando falla el verificador")
        self.assertIn("must be one of", res["error"])
        self.assertTrue(res.get("errors"), "el detalle completo sigue en errors")


class TestSyncAsyncParity(unittest.TestCase):
    """Invariante 10: `/api/train` (síncrono) y `/api/train-start` (asíncrono)
    con la misma semilla y `field_ranges` producen el MISMO resultado."""

    def test_same_seed_same_result(self):
        proj, mxai, fr = _project()
        sync = _run_playground_training(
            mxai, proj["training_text"], proj["csv_text"],
            epochs_override=12, field_ranges=fr)
        submitted = _submit_training_job(
            mxai, proj["training_text"], proj["csv_text"],
            epochs_override=12, field_ranges=fr, seed=42)
        self.assertTrue(submitted.get("ok"), submitted)
        status: dict = {}
        for _ in range(400):
            status = _get_job_status(submitted["job_id"])
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.2)
        self.assertEqual(status["status"], "done", status)
        self.assertAlmostEqual(sync["accuracy"], status["accuracy"], places=6,
                               msg="sync y async deben coincidir con la misma semilla")
        self.assertEqual(sync.get("confusion_matrix") or sync.get("confusion"),
                         status.get("confusion_matrix") or status.get("confusion"))


if __name__ == "__main__":
    unittest.main()
