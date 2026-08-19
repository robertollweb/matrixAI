"""Contrato 82 — Corte C1: EL PAQUETE REPRODUCIBLE.

Lo que este corte tenía que arreglar, medido antes de tocar nada:

  1. El paquete NO llevaba el `.mxtrain`. Con `model.mxai` + `params.best.json`
     + `model.onnx` se puede PREDECIR y no se puede REENTRENAR.
  2. No había manifiesto de reproducción. Reproducir necesita cinco cosas
     —`.mxai`, `.mxtrain`, receta, filas y semilla— y viajaban dos.
  3. El entorno no se declaraba. Medido: `export/bundle.py` escribe
     `_REQUIREMENTS = "numpy>=1.24\\nonnxruntime>=1.16\\n"`, que sirve para
     INFERIR y no para reproducir un número — esta máquina tiene numpy 2.4.4 y
     onnxruntime 1.26.0, ambas dentro del rango y ninguna igual al mínimo.

Y los invariantes del §6 que se comprueban aquí:

  §6.2  un modelo SIN receta lo dice (`reproducible: false` + motivo), no se le
        fabrica ninguna.
  §6.5  la versión del core y el motor viajan siempre.
  §6.6  R1 se decide con el SHA-256 COMPLETO, no con el fingerprint. Medido: la
        huella del core es `"data_" + sha256(...)[:16]` — 64 bits, identificador
        visual y NO prueba de integridad.
  §6.7  esto demuestra COHERENCIA, no AUTENTICIDAD, y el manifiesto lo dice.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib import util
from pathlib import Path

_HAS_ONNX = util.find_spec("onnx") is not None
_HAS_ORT = util.find_spec("onnxruntime") is not None

_BASE = Path(__file__).parent.parent
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.mxai"
_FALL_RISK_MXTRAIN = _BASE / "examples" / "fall-risk.supervised.mxtrain"

# Una receta cualquiera en el idioma del contrato 80. El bundle la copia tal
# cual; lo que se prueba aquí es que VIAJA y que su digest es el del fichero.
_RECIPE = "high: age > 75 OR previous_falls > 1\nlow: age < 60\nDEFAULT: low\n"

# Un sha256 COMPLETO de mentira (64 hex) para el dataset esperado: el contrato
# pide el del CSV entero, no el fichero.
_DATASET_SHA = "0e5b47e75c81ff09" * 4


class ReproduceManifestUnitTest(unittest.TestCase):
    """El manifiesto por su cuenta: no necesita onnx ni un bundle real."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)
        shutil.copy2(_FALL_RISK_MXAI, self.td / "model.mxai")

    def _full(self, **over):
        """Un paquete con las CINCO cosas: modelo, entrenamiento, receta, filas
        y semilla."""
        from matrixai.export import build_reproduce_manifest
        kwargs = dict(
            training_filename="model.mxtrain",
            recipe_filename="data_recipe.txt",
            dataset_sha256=_DATASET_SHA,
            dataset_rows=300,
            generation={"mode": "coherent", "seeds": {"dataset": 42, "init": 42}},
        )
        kwargs.update(over)
        # Los ficheros se ponen SOLO si el caso los pide: si el paquete no
        # lleva receta, la receta no está en el directorio — que es justo la
        # situación que se quiere probar, no un `filename=None` sobre un
        # fichero que sí existe.
        if kwargs["training_filename"]:
            shutil.copy2(_FALL_RISK_MXTRAIN, self.td / kwargs["training_filename"])
        if kwargs["recipe_filename"]:
            (self.td / kwargs["recipe_filename"]).write_text(_RECIPE, encoding="utf-8")
        return build_reproduce_manifest(self.td, **kwargs)

    # ── §5-C1 · la forma del manifiesto ────────────────────────────────

    def test_artifacts_carry_path_media_type_and_full_sha256(self):
        m = self._full()
        for kind, filename, media in (
            ("model", "model.mxai", "text/mxai"),
            ("training", "model.mxtrain", "text/mxtrain"),
            ("recipe", "data_recipe.txt", "text/plain"),
        ):
            art = m["artifacts"][kind]
            self.assertEqual(art["path"], filename, kind)
            self.assertEqual(art["media_type"], media, kind)
            # El digest es el del FICHERO REAL del paquete, no lo que dijo
            # quien exporta: un manifiesto que repite lo que le cuentan no
            # verifica nada.
            esperado = hashlib.sha256((self.td / filename).read_bytes()).hexdigest()
            self.assertEqual(art["sha256"], esperado, kind)
            self.assertEqual(len(art["sha256"]), 64, kind)

    def test_dataset_travels_as_expectation_not_as_file(self):
        # §6.4: el paquete no incrusta las filas — empaqueta la regla que las
        # produce — así que el dataset va por su valor ESPERADO.
        m = self._full()
        self.assertEqual(m["artifacts"]["dataset"], {"sha256": _DATASET_SHA, "rows": 300})
        self.assertNotIn("path", m["artifacts"]["dataset"])
        self.assertFalse((self.td / "dataset.csv").exists())

    def test_dataset_sha256_is_the_full_digest_not_the_short_fingerprint(self):
        # §6.6, y el dato que lo motiva: la huella del core son 16 hex.
        from matrixai.training.data import dataset_fingerprint
        huella = dataset_fingerprint(self.td / "model.mxai")
        self.assertTrue(huella.startswith("data_"))
        self.assertEqual(len(huella) - len("data_"), 16)
        m = self._full()
        self.assertEqual(len(m["artifacts"]["dataset"]["sha256"]), 64)

    # ── §5-C1 · `generation` lleva TODOS los parámetros efectivos ──────

    def test_generation_carries_the_versions_the_core_owns(self):
        from matrixai.training.dataset_manifest import (
            CSV_SERIALIZATION_VERSION, SYNTHETIC_GENERATOR_VERSION,
        )
        from matrixai.training.domain_rules import RECIPE_FORMAT_VERSION
        g = self._full()["generation"]
        self.assertEqual(g["generator_version"], SYNTHETIC_GENERATOR_VERSION)
        self.assertEqual(g["recipe_format_version"], RECIPE_FORMAT_VERSION)
        self.assertEqual(g["csv_serialization_version"], CSV_SERIALIZATION_VERSION)

    def test_caller_cannot_forge_the_core_owned_versions(self):
        # El core sabe qué generador tiene; dejar que el llamante declare otro
        # sería firmar una afirmación que no podemos sostener.
        from matrixai.training.dataset_manifest import SYNTHETIC_GENERATOR_VERSION
        g = self._full(generation={
            "seeds": {"dataset": 1},
            "generator_version": "matrixai.synthetic.v999",
        })["generation"]
        self.assertEqual(g["generator_version"], SYNTHETIC_GENERATOR_VERSION)

    def test_generation_keeps_the_effective_parameters(self):
        g = self._full(generation={
            "mode": "coherent",
            "seeds": {"dataset": 42, "init": 7},
            "field_ranges": {"age": [40, 95]},
            "field_types": {"age": "integer"},
            "field_categories": {"ward": ["a", "b"]},
            "one_hot_groups": {"ward": ["ward_a", "ward_b"]},
            "excluded_identifiers": ["patient_id"],
            "backend": "torch",
            "device": "cuda",
            "deterministic_options": {"torch_deterministic": True},
        })["generation"]
        self.assertEqual(g["mode"], "coherent")
        self.assertEqual(g["field_ranges"], {"age": [40, 95]})
        self.assertEqual(g["field_types"], {"age": "integer"})
        self.assertEqual(g["field_categories"], {"ward": ["a", "b"]})
        self.assertEqual(g["one_hot_groups"], {"ward": ["ward_a", "ward_b"]})
        self.assertEqual(g["excluded_identifiers"], ["patient_id"])
        self.assertEqual(g["backend"], "torch")
        self.assertEqual(g["device"], "cuda")
        self.assertEqual(g["deterministic_options"], {"torch_deterministic": True})

    def test_rows_live_in_one_place_only(self):
        # Están en `artifacts.dataset.rows`. Repetirlas en `generation` sería
        # el segundo sitio declarando lo mismo, y acabarían divergiendo.
        m = self._full()
        self.assertEqual(m["artifacts"]["dataset"]["rows"], 300)
        self.assertNotIn("rows", m["generation"])

    # ── §5-C1 · LAS TRES SEMILLAS, POR SEPARADO ───────────────────────

    def test_the_three_seeds_travel_separately(self):
        g = self._full(generation={"seeds": {"dataset": 42, "init": 7}})["generation"]
        self.assertEqual(set(g["seeds"]), {"dataset", "split", "init"})
        self.assertEqual(g["seeds"]["dataset"], 42)
        self.assertEqual(g["seeds"]["init"], 7)

    def test_split_seed_is_read_from_the_mxtrain_that_actually_trained(self):
        # Medido: `examples/fall-risk.supervised.mxtrain` declara
        # `SPLIT train=0.75 validation=0.25 seed=7`. No hay override por línea
        # de comandos para el split seed (sí para --backend/--device), así que
        # el `.mxtrain` ES la fuente y no hace falta que el llamante lo repita.
        g = self._full(generation={"seeds": {"dataset": 42}})["generation"]
        self.assertEqual(g["seeds"]["split"], 7)

    def test_an_unknown_seed_stays_null_it_is_never_invented(self):
        # Una semilla ausente NO es un cero, y tampoco es el 42 por defecto
        # del entrenador (`DenseSupervisedTrainer.train(seed: int = 42)`):
        # inventarla convertiría un paquete irreproducible en uno que parece
        # reproducible y falla al comprobarlo.
        m = self._full(training_filename=None, generation={"seeds": {"dataset": 42}})
        seeds = m["generation"]["seeds"]
        self.assertIsNone(seeds["init"])
        self.assertIsNone(seeds["split"])
        self.assertNotEqual(seeds["init"], 0)
        self.assertNotEqual(seeds["init"], 42)

    # ── §5-C1 · el entorno va CERRADO ─────────────────────────────────

    def test_environment_is_closed_exact_versions_no_ranges(self):
        from importlib import metadata
        env = self._full()["environment"]
        self.assertEqual(env["matrixai_version"], __import__("matrixai").__version__)
        self.assertTrue(env["python"]["version"])
        self.assertTrue(env["platform"]["machine"])
        # El contraste que da sentido al corte: el `requirements.txt` del
        # bundle declara RANGOS y aquí no puede haber ninguno.
        from matrixai.export.bundle import _REQUIREMENTS
        self.assertIn(">=", _REQUIREMENTS)
        textos = [v for v in env["packages"].values() if v is not None]
        self.assertTrue(textos, "el entorno no declaró ningún paquete")
        for value in textos:
            for operador in (">=", "<=", "~=", "*", ">", "<"):
                self.assertNotIn(operador, value)
        # Y son las versiones REALMENTE instaladas, no una lista escrita a mano.
        for name, declared in env["packages"].items():
            try:
                real = metadata.version(name)
            except Exception:  # noqa: BLE001
                real = None
            self.assertEqual(declared, real, name)

    def test_environment_has_its_own_digest(self):
        from matrixai.export import canonical_json
        env = self._full()["environment"]
        sin_digest = {k: v for k, v in env.items() if k != "environment_sha256"}
        self.assertEqual(
            env["environment_sha256"],
            hashlib.sha256(canonical_json(sin_digest).encode("utf-8")).hexdigest(),
        )

    # ── §5-C1 · `manifest_sha256` ─────────────────────────────────────

    def test_manifest_digest_excludes_itself_and_verifies(self):
        from matrixai.export import manifest_digest, verify_manifest_digest
        m = self._full()
        self.assertEqual(len(m["manifest_sha256"]), 64)
        self.assertTrue(verify_manifest_digest(m))
        sin_campo = {k: v for k, v in m.items() if k != "manifest_sha256"}
        self.assertEqual(m["manifest_sha256"], manifest_digest(sin_campo))

    def test_tampering_any_field_breaks_the_manifest_digest(self):
        # Un banco de pruebas sin dientes no vale: se sabotea a propósito.
        from matrixai.export import verify_manifest_digest
        for clave, valor in (
            ("reproducible", False),
            ("schema_version", "9.9"),
            ("metrics", [{"name": "accuracy", "value": 0.99}]),
        ):
            with self.subTest(clave=clave):
                m = self._full()
                m[clave] = valor
                self.assertFalse(verify_manifest_digest(m))
        # Y también un cambio ENTERRADO en un sub-objeto.
        m = self._full()
        m["artifacts"]["recipe"]["sha256"] = "0" * 64
        self.assertFalse(verify_manifest_digest(m))

    def test_the_canonicalization_travels_so_the_digest_can_be_recomputed(self):
        # Un digest que nadie sabe recomputar no verifica nada.
        m = self._full()
        self.assertIn("sort_keys", m["manifest_canonicalization"])

    # ── §6.2 · un modelo sin receta LO DICE ───────────────────────────

    def test_a_model_without_a_recipe_says_so_instead_of_keeping_quiet(self):
        m = self._full(recipe_filename=None)
        self.assertFalse(m["reproducible"])
        self.assertIn("recipe", m["missing"])
        self.assertIn("no data recipe", m["reproducible_reason"])
        # Y no se le fabrica ninguna: el hueco se declara vacío.
        self.assertIsNone(m["artifacts"]["recipe"])
        self.assertFalse((self.td / "data_recipe.txt").exists())

    def test_missing_pieces_are_named_one_by_one(self):
        m = self._full(training_filename=None, recipe_filename=None,
                       dataset_sha256=None, dataset_rows=None,
                       generation={"seeds": {}})
        self.assertEqual(
            set(m["missing"]),
            {"training", "recipe", "dataset_sha256", "dataset_rows", "seed_dataset"},
        )

    def test_a_package_with_the_five_pieces_is_reproducible(self):
        m = self._full()
        self.assertTrue(m["reproducible"])
        self.assertEqual(m["missing"], [])
        # Un `null` es la respuesta: no hay motivo porque no falta nada.
        self.assertIsNone(m["reproducible_reason"])

    def test_missing_mxtrain_alone_blocks_reproduction(self):
        # Sin el contrato de entrenamiento se puede predecir y no reentrenar.
        m = self._full(training_filename=None)
        self.assertFalse(m["reproducible"])
        self.assertEqual(m["missing"], ["training"])
        self.assertIsNone(m["artifacts"]["training"])

    # ── §6.7 · coherencia, NO autenticidad ────────────────────────────

    def test_the_manifest_does_not_promise_authenticity(self):
        m = self._full()
        self.assertIn("NOT prove", m["claim"])
        self.assertIn("authenticity", m["claim"].lower())

    # ── §5 bis · las métricas, estructuradas ──────────────────────────

    def test_metrics_keep_the_fields_that_make_them_comparable(self):
        m = self._full(metrics=[{
            "name": "accuracy", "value": 0.6166666666666667, "split": "validation",
            "dataset_sha256": _DATASET_SHA, "tolerance_abs": 0.0,
        }])
        metric = m["metrics"][0]
        # Los campos ausentes quedan VISIBLES en `null`: una tolerancia que no
        # está se ve; una clave que no está se pasa por alto.
        for campo in ("name", "value", "split", "dataset_sha256", "evaluator",
                      "evaluator_version", "aggregation", "direction",
                      "tolerance_abs", "tolerance_rel"):
            self.assertIn(campo, metric, campo)
        self.assertEqual(metric["split"], "validation")
        self.assertIsNone(metric["evaluator"])
        self.assertEqual(metric["tolerance_abs"], 0.0)

    def test_a_metric_without_a_value_is_rejected(self):
        from matrixai.export import ReproduceManifestError
        with self.assertRaises(ReproduceManifestError):
            self._full(metrics=[{"name": "accuracy"}])
        with self.assertRaises(ReproduceManifestError):
            self._full(metrics=[{"value": 0.9}])

    def test_no_metrics_is_an_empty_list_not_a_missing_key(self):
        self.assertEqual(self._full()["metrics"], [])


@unittest.skipUnless(_HAS_ONNX and _HAS_ORT, "onnx/onnxruntime not installed")
class BundleShipsTheReproduciblePackageTest(unittest.TestCase):
    """El paquete de verdad: `create_edge_bundle` de punta a punta."""

    def setUp(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        self.prog = parse_file(_FALL_RISK_MXAI)
        self.ps = build_initial_parameter_set(self.prog)
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)
        self.params = self.td / "params.json"
        write_parameter_set(str(self.params), self.ps)

    def _bundle(self, name="b", **over):
        from matrixai.export import create_edge_bundle
        kwargs = dict(mxai_path=str(_FALL_RISK_MXAI), params_path=str(self.params),
                      outdir=str(self.td / name), validate=False)
        kwargs.update(over)
        return create_edge_bundle(self.prog, self.ps, **kwargs)

    def test_bundle_ships_the_mxtrain_byte_for_byte(self):
        r = self._bundle(mxtrain_path=str(_FALL_RISK_MXTRAIN))
        shipped = Path(r.bundle_dir) / "model.mxtrain"
        self.assertIn("model.mxtrain", r.files)
        self.assertEqual(shipped.read_bytes(), _FALL_RISK_MXTRAIN.read_bytes())

    def test_the_mxtrain_enters_with_a_fixed_name(self):
        # El nombre del fichero de origen es del proyecto de quien exporta; no
        # tiene por qué viajar dentro del paquete (igual que `model.mxai`).
        origen = self.td / "mi-proyecto-privado.mxtrain"
        shutil.copy2(_FALL_RISK_MXTRAIN, origen)
        r = self._bundle(name="b_nombre", mxtrain_path=str(origen))
        self.assertIn("model.mxtrain", r.files)
        self.assertNotIn("mi-proyecto-privado.mxtrain", r.files)

    def test_reproduce_json_is_always_there_even_with_nothing_to_declare(self):
        # Un paquete que calla no dice «esto no se puede reproducir»: no dice
        # nada. §6.2 lo pide explícito.
        r = self._bundle(name="b_pelado")
        self.assertIn("reproduce.json", r.files)
        m = json.loads((Path(r.bundle_dir) / "reproduce.json").read_text())
        self.assertFalse(m["reproducible"])
        self.assertIn("recipe", m["missing"])
        self.assertIn("training", m["missing"])

    def test_the_full_package_is_declared_reproducible(self):
        r = self._bundle(
            name="b_full",
            mxtrain_path=str(_FALL_RISK_MXTRAIN),
            data_recipe=_RECIPE,
            dataset_sha256=_DATASET_SHA,
            dataset_rows=300,
            generation={"mode": "coherent", "seeds": {"dataset": 42, "init": 42},
                        "backend": "stdlib", "device": "cpu"},
        )
        bd = Path(r.bundle_dir)
        self.assertEqual(
            {"model.mxai", "model.mxtrain", "data_recipe.txt", "reproduce.json"}
            - set(r.files), set())
        m = json.loads((bd / "reproduce.json").read_text())
        self.assertTrue(m["reproducible"], m["reproducible_reason"])
        # El resultado devuelve el manifiesto entero para que el llamante no
        # tenga que releer el fichero ni decidir por su cuenta qué falta.
        self.assertEqual(r.reproduce, m)
        self.assertEqual(r.to_dict()["reproduce"], m)

    def test_digests_are_taken_from_the_files_not_echoed_from_the_caller(self):
        r = self._bundle(name="b_dig", mxtrain_path=str(_FALL_RISK_MXTRAIN),
                         data_recipe=_RECIPE)
        bd = Path(r.bundle_dir)
        m = json.loads((bd / "reproduce.json").read_text())
        for kind, filename in (("model", "model.mxai"),
                               ("training", "model.mxtrain"),
                               ("recipe", "data_recipe.txt")):
            self.assertEqual(
                m["artifacts"][kind]["sha256"],
                hashlib.sha256((bd / filename).read_bytes()).hexdigest(), kind)

    def test_touching_the_recipe_after_packaging_breaks_its_digest(self):
        # Sabotaje a propósito: cambiar UNA línea de la receta tiene que dejar
        # de casar con lo declarado (§C2 construirá su `FAIL` sobre esto).
        r = self._bundle(name="b_sab", mxtrain_path=str(_FALL_RISK_MXTRAIN),
                         data_recipe=_RECIPE)
        bd = Path(r.bundle_dir)
        m = json.loads((bd / "reproduce.json").read_text())
        antes = m["artifacts"]["recipe"]["sha256"]
        (bd / "data_recipe.txt").write_text(_RECIPE.replace("75", "70"), encoding="utf-8")
        despues = hashlib.sha256((bd / "data_recipe.txt").read_bytes()).hexdigest()
        self.assertNotEqual(antes, despues)

    def test_the_readme_lists_the_real_files_and_declares_the_state(self):
        con = Path(self._bundle(
            name="b_readme_si", mxtrain_path=str(_FALL_RISK_MXTRAIN),
            data_recipe=_RECIPE, dataset_sha256=_DATASET_SHA, dataset_rows=300,
            generation={"seeds": {"dataset": 42}},
        ).bundle_dir) / "README.md"
        texto = con.read_text()
        self.assertIn("`model.mxtrain`", texto)
        self.assertIn("`data_recipe.txt`", texto)
        self.assertIn("`reproduce.json`", texto)
        self.assertIn("**reproducible**", texto)

        sin = Path(self._bundle(name="b_readme_no").bundle_dir) / "README.md"
        texto = sin.read_text()
        # Lista los ficheros REALES: sin `.mxtrain` no lo anuncia.
        self.assertNotIn("`model.mxtrain`", texto)
        self.assertNotIn("`data_recipe.txt`", texto)
        self.assertIn("`reproduce.json`", texto)
        self.assertIn("no data recipe", texto)

    def test_the_bundle_manifest_verifies_itself(self):
        from matrixai.export import verify_manifest_digest
        r = self._bundle(name="b_verify", mxtrain_path=str(_FALL_RISK_MXTRAIN),
                         data_recipe=_RECIPE)
        m = json.loads((Path(r.bundle_dir) / "reproduce.json").read_text())
        self.assertTrue(verify_manifest_digest(m))


@unittest.skipUnless(_HAS_ONNX and _HAS_ORT, "onnx/onnxruntime not installed")
class ExportBundleCliTest(unittest.TestCase):
    """El cableado: que el API exista y el llamante no lo use ya ha pasado
    CATORCE veces en este producto. Aquí se conduce por la línea de comandos."""

    def setUp(self):
        from matrixai.parser import parse_file
        from matrixai.parameters import build_initial_parameter_set, write_parameter_set
        prog = parse_file(_FALL_RISK_MXAI)
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)
        self.params = self.td / "params.json"
        write_parameter_set(str(self.params), build_initial_parameter_set(prog))
        (self.td / "receta.txt").write_text(_RECIPE, encoding="utf-8")

    def _run(self, *extra, outdir="cli"):
        return subprocess.run(
            [sys.executable, "-m", "matrixai.cli", "export-bundle", str(_FALL_RISK_MXAI),
             "--params", str(self.params), "--outdir", str(self.td / outdir),
             "--no-validate", "--force", *extra],
            capture_output=True, text=True, timeout=300, cwd=str(_BASE),
        )

    def test_cli_ships_training_recipe_and_reproduce_metadata(self):
        sidecar = self.td / "repro.json"
        sidecar.write_text(json.dumps({
            "dataset_sha256": _DATASET_SHA,
            "dataset_rows": 300,
            "generation": {"mode": "coherent", "seeds": {"dataset": 42, "init": 42},
                           "backend": "stdlib", "device": "cpu"},
            "metrics": [{"name": "accuracy", "value": 0.6166666666666667,
                         "split": "validation", "dataset_sha256": _DATASET_SHA}],
        }), encoding="utf-8")
        out = self._run("--training", str(_FALL_RISK_MXTRAIN),
                        "--data-recipe", str(self.td / "receta.txt"),
                        "--reproduce-metadata", str(sidecar))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Reproducible: yes", out.stdout)
        m = json.loads((self.td / "cli" / "reproduce.json").read_text())
        self.assertTrue(m["reproducible"])
        self.assertEqual(m["generation"]["seeds"], {"dataset": 42, "split": 7, "init": 42})
        self.assertEqual(m["metrics"][0]["split"], "validation")

    def test_cli_says_out_loud_when_the_package_is_not_reproducible(self):
        out = self._run(outdir="cli_no")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Reproducible: no", out.stdout)
        self.assertIn("no data recipe", out.stdout)

    def test_cli_refuses_the_short_fingerprint_as_dataset_proof(self):
        # §6.6: `data_49219efba673b8d0` es un identificador visual de 64 bits,
        # no una prueba de integridad. Colarlo donde el contrato pide sha256
        # completo daría un paquete que parece verificable y no lo es.
        sidecar = self.td / "corto.json"
        sidecar.write_text(json.dumps({"dataset_sha256": "data_49219efba673b8d0"}),
                           encoding="utf-8")
        out = self._run("--reproduce-metadata", str(sidecar), outdir="cli_corto")
        self.assertEqual(out.returncode, 1)
        self.assertIn("64 lowercase hex", out.stderr)
        self.assertFalse((self.td / "cli_corto").exists())

    def test_cli_refuses_a_non_integer_seed(self):
        sidecar = self.td / "semilla.json"
        sidecar.write_text(json.dumps({"generation": {"seeds": {"dataset": "42"}}}),
                           encoding="utf-8")
        out = self._run("--reproduce-metadata", str(sidecar), outdir="cli_semilla")
        self.assertEqual(out.returncode, 1)
        self.assertIn("integer or null", out.stderr)


if __name__ == "__main__":
    unittest.main()
