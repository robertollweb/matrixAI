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


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _captura(*, recipe_text=_RECIPE, dataset_sha256=_DATASET_SHA,
             dataset_rows=300, generation=None, **over) -> dict:
    """La CAPTURA que el core guarda en el run al entrenar (82-C2).

    Es la única fuente de lo que el manifiesto afirma. Se compone aquí desde
    los mismos ficheros de ejemplo que se empaquetan, así que el caso honesto
    —captura y paquete de acuerdo— es el que sale por defecto y los casos
    adversariales son los que se desvían de él a propósito.

    Ojo con `generation`: sus parámetros efectivos son del RUN, así que viajan
    DENTRO de la captura. Lo que un test quiera probar como «lo que manda la
    pantalla» se pasa aparte, por el payload de `build_reproduce_manifest`.
    """
    gen = dict(generation) if isinstance(generation, dict) else {}
    semillas = gen.pop("seeds", None)
    semillas = dict(semillas) if isinstance(semillas, dict) else {}
    contrato = _FALL_RISK_MXTRAIN.read_text(encoding="utf-8")
    cap = {
        "schema_version": "1.0",
        "mxai_sha256": _sha(_FALL_RISK_MXAI.read_text(encoding="utf-8")),
        "mxtrain_sha256": _sha(contrato),
        "mxtrain_text": contrato,
        "recipe_sha256": _sha(recipe_text) if recipe_text else None,
        "recipe_text": recipe_text,
        "dataset_sha256_raw": dataset_sha256,
        "dataset_sha256_prepared": None,
        "dataset_rows": dataset_rows,
        "seeds": {"dataset": semillas.get("dataset"), "split": semillas.get("split"),
                  "init": semillas.get("init")},
        "backend": gen.pop("backend", None),
        "device": gen.pop("device", None),
        "generator_version": None,
        "csv_serialization_version": None,
    }
    cap.update(gen)
    cap.update(over)
    return cap


class ReproduceManifestUnitTest(unittest.TestCase):
    """El manifiesto por su cuenta: no necesita onnx ni un bundle real."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)
        shutil.copy2(_FALL_RISK_MXAI, self.td / "model.mxai")

    def _full(self, *, provenance="auto", **over):
        """Un paquete con las CINCO cosas —modelo, entrenamiento, receta, filas
        y semilla— Y la captura que las respalda (82-C2).

        Las cinco ya no bastan: desde el 82-C2 el manifiesto solo afirma lo que
        salga de la captura del run, así que el caso «todo bien» la lleva. Con
        `provenance=None` se prueba justo lo contrario: las cinco piezas y nada
        que ate el paquete a los pesos que lleva dentro.

        Los valores del caso viajan por LAS DOS vías —captura y payload— y
        coherentes entre sí, que es el caso honesto: así un test que cambie
        solo el payload mide contradicción, y uno que cambie el valor mide
        forma.
        """
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
        if provenance == "auto":
            provenance = _captura(
                recipe_text=_RECIPE if kwargs["recipe_filename"] else None,
                dataset_sha256=kwargs["dataset_sha256"],
                dataset_rows=kwargs["dataset_rows"],
                generation=kwargs["generation"],
            )
        return build_reproduce_manifest(self.td, run_provenance=provenance, **kwargs)

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
        # Los DOS digests, porque son dos preguntas distintas y NO coinciden
        # (medido el 2026-08-19): el CRUDO es contra el que se compara un
        # dataset regenerado —R1, «byte a byte»— y el PREPARADO es el del
        # fichero con el que la red entrenó, que es el que ata estos pesos.
        self.assertEqual(m["artifacts"]["dataset"],
                         {"sha256": _DATASET_SHA, "sha256_prepared": None, "rows": 300})
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
        # El core sabe qué generador tiene; dejar que QUIEN EXPORTA declare
        # otro sería firmar una afirmación que no podemos sostener. Y desde el
        # 82-C2 tampoco se descarta en silencio: se declara el conflicto y el
        # paquete deja de ser reproducible, porque hay dos versiones del mismo
        # dato y una es falsa.
        from matrixai.training.dataset_manifest import SYNTHETIC_GENERATOR_VERSION
        m = self._full(
            provenance=_captura(generation={"seeds": {"dataset": 1}}),
            generation={"seeds": {"dataset": 1},
                        "generator_version": "matrixai.synthetic.v999"},
        )
        self.assertEqual(m["generation"]["generator_version"],
                         SYNTHETIC_GENERATOR_VERSION)
        self.assertIn("generation.generator_version",
                      [c["field"] for c in m["conflicts"]])
        self.assertFalse(m["reproducible"])

    def test_the_captured_generator_version_is_the_one_that_ran(self):
        # Al revés que el anterior, y no es lo mismo: la captura NO es «el
        # llamante». Si el run se capturó con otro generador y este core se ha
        # actualizado desde entonces, manda lo que CORRIÓ —regenerar aquí no
        # daría lo mismo— y la diferencia se declara. Una nota vieja miente
        # igual que un dato falso.
        m = self._full(provenance=_captura(
            generation={"seeds": {"dataset": 42, "init": 42}},
            generator_version="matrixai.synthetic.v1-antiguo"))
        self.assertEqual(m["generation"]["generator_version"],
                         "matrixai.synthetic.v1-antiguo")
        conflicto = [c for c in m["conflicts"]
                     if c["field"] == "generation.generator_version"]
        self.assertEqual(len(conflicto), 1)
        self.assertEqual(conflicto[0]["source"], "running_core")
        self.assertFalse(m["reproducible"])

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
        self.assertNotEqual(seeds["init"], 0)
        self.assertNotEqual(seeds["init"], 42)
        # La del SPLIT sí se sabe aunque el `.mxtrain` no viaje: la captura
        # lleva el contrato ENTERO, así que lo que falta es el artefacto, no
        # el dato. Sin captura y sin fichero no hay de dónde sacarla, y se
        # queda en `null` — no en el 42 del entrenador.
        self.assertEqual(seeds["split"], 7)
        sin = self._full(training_filename=None, provenance=None)
        self.assertIsNone(sin["generation"]["seeds"]["split"])

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
        gen = {"mode": "coherent", "seeds": {"dataset": 42, "init": 42},
               "backend": "stdlib", "device": "cpu"}
        r = self._bundle(
            name="b_full",
            mxtrain_path=str(_FALL_RISK_MXTRAIN),
            data_recipe=_RECIPE,
            dataset_sha256=_DATASET_SHA,
            dataset_rows=300,
            generation=gen,
            # 82-C2: sin la captura del run el paquete no puede demostrar su
            # relación con los pesos que lleva, y el cableado hasta aquí es
            # justo lo que hay que probar — que el API la acepte y el llamante
            # no la pase ya ha pasado CATORCE veces en este producto.
            run_provenance=_captura(dataset_sha256=_DATASET_SHA, dataset_rows=300,
                                    generation=gen),
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
            run_provenance=_captura(generation={"seeds": {"dataset": 42}}),
        ).bundle_dir) / "README.md"
        texto = con.read_text()
        self.assertIn("`model.mxtrain`", texto)
        self.assertIn("`data_recipe.txt`", texto)
        self.assertIn("`reproduce.json`", texto)
        self.assertIn("**reproducible**", texto)
        # Y DE DÓNDE sale lo que afirma: «reproducible» sin eso se leía como
        # «alguien rellenó los campos al exportar y salieron bien».
        self.assertIn("capture the core recorded while training", texto)

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
        # El `.mxtrain` y la receta SÍ viajan, que es lo que este test cubría.
        self.assertIn("model.mxtrain", out.stdout)
        self.assertIn("data_recipe.txt", out.stdout)
        m = json.loads((self.td / "cli" / "reproduce.json").read_text())

        # Y AQUÍ ESTÁ EL CAMBIO DEL 82-C2, medido conduciendo por la línea de
        # comandos. Un sidecar escrito a mano NO es una captura del run: nadie
        # ha entrenado nada al escribirlo, así que el paquete no puede
        # demostrar su relación con los pesos que lleva y lo dice en voz alta
        # en vez de declararse reproducible.
        self.assertIn("Reproducible: no", out.stdout)
        self.assertIn("authoritative run capture", out.stdout)
        self.assertFalse(m["reproducible"])
        self.assertEqual(m["provenance"]["source"], "export_payload_only")
        self.assertIn("run_provenance", m["missing"])

        # Lo que mandó el sidecar no se publica como cierto — pero tampoco se
        # tira en silencio: viaja aparte y con su nombre, para que quien
        # exporta vea que llegó y por qué no se firmó.
        sin_verificar = m["provenance"]["unverified_payload"]
        self.assertEqual(sin_verificar["dataset_sha256"], _DATASET_SHA)
        self.assertEqual(sin_verificar["dataset_rows"], 300)
        self.assertEqual(sin_verificar["generation"]["seeds"]["dataset"], 42)

        # La del SPLIT es la excepción, y por un motivo: NO sale del sidecar,
        # sale del `.mxtrain` que viaja en el paquete y cuyo digest está en
        # este mismo manifiesto. Quien lo recibe puede rederivarla.
        self.assertEqual(m["generation"]["seeds"],
                         {"dataset": None, "split": 7, "init": None})
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

    def test_cli_refuses_a_metric_that_cannot_be_compared_with_anything(self):
        # El sidecar del CLI validaba `dataset_sha256`, las filas y las
        # semillas por su cuenta y NO miraba dentro de `metrics`: pasaba la
        # lista tal cual. Ahora la regla es UNA y vive en el core, así que
        # llega hasta aquí conduciendo por la línea de comandos — que el API
        # tenga la regla y el llamante no la use ya ha pasado CATORCE veces.
        #
        # Lo del `outdir` está MEDIDO, no supuesto: el bundle se construye en
        # un `TemporaryDirectory` (`bundle.py:185`) y solo aparece al final,
        # así que un sidecar mal escrito no deja nada a medias en disco.
        sidecar = self.td / "metrica.json"
        sidecar.write_text(json.dumps({
            "metrics": [{"name": "accuracy", "value": "muy alta"}],
        }), encoding="utf-8")
        out = self._run("--reproduce-metadata", str(sidecar), outdir="cli_metrica")
        self.assertEqual(out.returncode, 1)
        self.assertIn("must be a number", out.stderr)
        self.assertFalse((self.td / "cli_metrica").exists())

    def test_cli_refuses_a_non_integer_seed(self):
        sidecar = self.td / "semilla.json"
        sidecar.write_text(json.dumps({"generation": {"seeds": {"dataset": "42"}}}),
                           encoding="utf-8")
        out = self._run("--reproduce-metadata", str(sidecar), outdir="cli_semilla")
        self.assertEqual(out.returncode, 1)
        self.assertIn("integer or null", out.stderr)


class ElManifiestoNoPuedeMentirTest(unittest.TestCase):
    """82-C1 · lo que el manifiesto NO puede aceptar (medido el 2026-08-19).

    El hallazgo, ejecutado contra el core antes de arreglar nada:

        build_reproduce_manifest(..., dataset_sha256="not-a-sha",
                                 dataset_rows=-4,
                                 generation={"seeds": {"dataset": "42"}})
        -> reproducible: True | missing: []

    El paquete PROMETÍA reproducirse con un digest que no es un digest, filas
    negativas y una semilla que es texto: la «falsa sensación de garantía» que
    el propio contrato prohíbe (§6.7). El predicado miraba PRESENCIA y no
    validez, y la única validación estricta vivía en dos llamantes —el CLI y
    el backend del Studio—, así que cualquier otro la esquivaba.
    """

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)
        shutil.copy2(_FALL_RISK_MXAI, self.td / "model.mxai")
        shutil.copy2(_FALL_RISK_MXTRAIN, self.td / "model.mxtrain")
        (self.td / "data_recipe.txt").write_text(_RECIPE, encoding="utf-8")

    def _build(self, *, provenance="auto", **over):
        """El caso honesto: captura y payload de acuerdo.

        Los valores del caso van por LAS DOS vías a propósito. Lo que se prueba
        en esta clase es qué formas NO puede aceptar el manifiesto, y una forma
        imposible tiene que cortarse venga de la pantalla o venga de la
        captura: si solo se probara una, la otra sería el hueco.
        """
        from matrixai.export import build_reproduce_manifest
        kwargs = dict(
            training_filename="model.mxtrain",
            recipe_filename="data_recipe.txt",
            dataset_sha256=_DATASET_SHA,
            dataset_rows=300,
            generation={"seeds": {"dataset": 42}},
        )
        kwargs.update(over)
        if provenance == "auto":
            provenance = _captura(dataset_sha256=kwargs["dataset_sha256"],
                                  dataset_rows=kwargs["dataset_rows"],
                                  generation=kwargs["generation"])
        return build_reproduce_manifest(self.td, run_provenance=provenance, **kwargs)

    def _rechaza(self, **over):
        """Comprueba que se rechaza Y devuelve el mensaje, para poder mirarlo.

        Se comprueba el mensaje además del tipo: una excepción que no dice qué
        campo está mal obliga a quien exporta a adivinar cuál de los diez es.
        """
        from matrixai.export import ReproduceManifestError
        with self.assertRaises(ReproduceManifestError) as ctx:
            self._build(**over)
        return str(ctx.exception)

    # ── EL HALLAZGO, entero ────────────────────────────────────────────

    def test_the_measured_finding_no_longer_produces_a_reproducible_package(self):
        mensaje = self._rechaza(dataset_sha256="not-a-sha", dataset_rows=-4,
                                generation={"seeds": {"dataset": "42"}})
        self.assertIn("dataset_sha256", mensaje)

    # ── dataset_sha256 (§6.6) ──────────────────────────────────────────

    def test_a_dataset_sha256_that_is_not_a_sha256_is_refused(self):
        for basura in ("not-a-sha", "", "  ", "0" * 63, "0" * 65,
                       "A" * 64,          # mayúsculas: otro texto, otro digest
                       "z" * 64,          # 64 caracteres que no son hex
                       12345, True, ["0" * 64]):
            with self.subTest(basura=basura):
                self.assertIn("64 lowercase hex", self._rechaza(dataset_sha256=basura))

    def test_the_short_fingerprint_is_refused_where_the_contract_asks_for_proof(self):
        # §6.6: `data_49219efba673b8d0` son 64 BITS. Es el identificador que
        # enseña el producto y no una prueba de integridad; colarlo aquí daría
        # un paquete que parece verificable y no lo es.
        self.assertIn("data_...", self._rechaza(dataset_sha256="data_49219efba673b8d0"))

    def test_a_real_digest_still_goes_through(self):
        # Un banco con dientes tiene que morder solo lo malo: si rechazara
        # también lo bueno, el arreglo sería una regresión disfrazada.
        m = self._build(dataset_sha256=hashlib.sha256(b"csv").hexdigest())
        self.assertTrue(m["reproducible"])

    # ── dataset_rows ───────────────────────────────────────────────────

    def test_row_counts_that_no_dataset_can_have_are_refused(self):
        for basura in (0, -4, -1, "300", 300.0, None if False else 3.5, [300]):
            with self.subTest(basura=basura):
                self.assertIn("positive integer", self._rechaza(dataset_rows=basura))

    def test_true_is_not_a_row_count(self):
        # `isinstance(True, int)` es True en Python: sin `type(...) is int` el
        # paquete habría publicado «1 fila» y su manifiesto lo habría firmado.
        self.assertIn("positive integer", self._rechaza(dataset_rows=True))
        self.assertEqual(self._build(dataset_rows=1)["artifacts"]["dataset"]["rows"], 1)

    # ── las semillas ───────────────────────────────────────────────────

    def test_a_seed_that_is_not_an_integer_is_refused(self):
        for basura in ("42", 42.0, True, False, [42], {"v": 42}):
            for cual in ("dataset", "split", "init"):
                with self.subTest(basura=basura, semilla=cual):
                    mensaje = self._rechaza(generation={"seeds": {cual: basura}})
                    self.assertIn("integer or null", mensaje)
                    self.assertIn(cual, mensaje)

    def test_a_null_seed_is_still_a_legitimate_answer(self):
        # Ausente NO es inválido (§6.2): se declara `null` y el hueco se ve en
        # `missing`. Lo que no vale es una semilla que no es una semilla.
        m = self._build(generation={"seeds": {"dataset": 42, "init": None}})
        self.assertIsNone(m["generation"]["seeds"]["init"])
        self.assertTrue(m["reproducible"])

    # ── rutas: el manifiesto no puede apuntar fuera del paquete ────────

    def test_an_artifact_path_cannot_escape_the_package(self):
        for ruta in ("../../etc/passwd", "sub/model.mxtrain", "/etc/passwd", ".."):
            with self.subTest(ruta=ruta):
                self.assertIn("plain filename", self._rechaza(training_filename=ruta))

    # ── el resto de `generation` ───────────────────────────────────────

    def test_generation_shapes_that_cannot_be_used_to_regenerate_are_refused(self):
        casos = [
            ({"mode": "coherent-plus"}, "generation.mode"),
            ({"backend": "tensorflow"}, "generation.backend"),
            ({"device": ""}, "generation.device"),
            ({"field_ranges": {"age": [40]}}, "field_ranges"),
            ({"field_ranges": {"age": ["a", "b"]}}, "field_ranges"),
            ({"field_ranges": {"age": [95, 40]}}, "min > max"),
            ({"field_types": {"age": 7}}, "field_types"),
            ({"field_categories": {"ward": "a"}}, "field_categories"),
            ({"excluded_identifiers": "patient_id"}, "excluded_identifiers"),
            ({"deterministic_options": "yes"}, "deterministic_options"),
            ({"seeds": "42"}, "generation.seeds"),
        ]
        for extra, esperado in casos:
            with self.subTest(extra=extra):
                gen = {"seeds": {"dataset": 42}}
                gen.update(extra)
                self.assertIn(esperado, self._rechaza(generation=gen))

    def test_the_effective_parameters_that_are_well_formed_still_travel(self):
        m = self._build(generation={
            "seeds": {"dataset": 42, "init": 7}, "mode": "coherent",
            "backend": "torch", "device": "cuda:0",
            "field_ranges": {"age": [40, 95]},
            "field_categories": {"ward": ["a", "b"]},
            "excluded_identifiers": ["patient_id"],
        })
        self.assertEqual(m["generation"]["device"], "cuda:0")
        self.assertEqual(m["generation"]["field_ranges"], {"age": [40, 95]})

    # ── §5 bis · las métricas ──────────────────────────────────────────

    def test_a_metric_with_an_impossible_shape_is_refused(self):
        casos = [
            ({"name": "accuracy", "value": "muy alta"}, "must be a number"),
            ({"name": "accuracy", "value": True}, "must be a number"),
            ({"name": "accuracy", "value": float("nan")}, "finite"),
            ({"name": "accuracy", "value": float("inf")}, "finite"),
            ({"name": 7, "value": 0.9}, "non-empty 'name'"),
            ({"name": "  ", "value": 0.9}, "non-empty 'name'"),
            ({"name": "a", "value": 0.9, "split": "no-existe"}, "split"),
            ({"name": "a", "value": 0.9, "direction": "hacia-arriba"}, "direction"),
            ({"name": "a", "value": 0.9, "aggregation": ""}, "aggregation"),
            ({"name": "a", "value": 0.9, "evaluator_version": {"a": 1}}, "evaluator_version"),
            ({"name": "a", "value": 0.9, "tolerance_abs": -5}, ">= 0.0"),
            ({"name": "a", "value": 0.9, "tolerance_abs": "un poco"}, "must be a number"),
            ({"name": "a", "value": 0.9, "tolerance_rel": -0.1}, ">= 0.0"),
        ]
        for metrica, esperado in casos:
            with self.subTest(metrica=metrica):
                self.assertIn(esperado, self._rechaza(metrics=[metrica]))

    def test_a_metric_cannot_carry_the_short_fingerprint_either(self):
        # El CLI ya exigía 64 hex para el `dataset_sha256` del manifiesto y NO
        # lo exigía para el de DENTRO de cada métrica, que es justo donde el
        # §5 bis dice «sobre QUÉ datos se midió». Por ahí se colaba.
        mensaje = self._rechaza(metrics=[{
            "name": "accuracy", "value": 0.9,
            "dataset_sha256": "data_49219efba673b8d0",
        }])
        self.assertIn("64 lowercase hex", mensaje)

    def test_two_values_for_the_same_metric_are_a_contradiction_not_two_metrics(self):
        mensaje = self._rechaza(metrics=[
            {"name": "accuracy", "value": 0.9, "split": "validation"},
            {"name": "accuracy", "value": 0.1, "split": "validation"},
        ])
        self.assertIn("two different values", mensaje)
        # Y la misma métrica en particiones DISTINTAS sí son dos métricas.
        m = self._build(metrics=[
            {"name": "accuracy", "value": 0.9, "split": "train"},
            {"name": "accuracy", "value": 0.6, "split": "validation"},
        ])
        self.assertEqual(len(m["metrics"]), 2)

    def test_an_incomplete_metric_is_declared_incomplete_not_smuggled_as_complete(self):
        # §5 bis pide diez campos y hoy el producto no tiene tres: medido,
        # `evaluator_version` no existe en el core y las tolerancias salen de
        # repetir en la matriz de entornos, que no se ha hecho. Rechazarlas
        # dejaría el paquete sin métricas y rellenarlas sería fabricar lo que
        # el core no ha dicho: se declaran.
        m = self._build(metrics=[{"name": "macro_f1", "value": 0.551227}])
        metrica = m["metrics"][0]
        self.assertFalse(metrica["comparable"])
        for campo in ("split", "dataset_sha256", "evaluator", "evaluator_version",
                      "aggregation", "direction", "tolerance_abs", "tolerance_rel"):
            self.assertIn(campo, metrica["incomplete"], campo)
        self.assertNotIn("name", metrica["incomplete"])
        self.assertNotIn("value", metrica["incomplete"])

    def test_a_metric_is_comparable_only_when_r3_could_actually_use_it(self):
        completa = {
            "name": "accuracy", "value": 0.6166666666666667, "split": "validation",
            "dataset_sha256": _DATASET_SHA, "direction": "higher_is_better",
            "tolerance_abs": 0.0,
        }
        self.assertTrue(self._build(metrics=[completa])["metrics"][0]["comparable"])
        for quitar in ("split", "dataset_sha256", "direction"):
            with self.subTest(quitar=quitar):
                recortada = {k: v for k, v in completa.items() if k != quitar}
                m = self._build(metrics=[recortada])
                self.assertFalse(m["metrics"][0]["comparable"])
                self.assertIn(quitar, m["metrics"][0]["incomplete"])
        # Sin NINGUNA tolerancia no hay umbral, y §5 bis prohíbe inventarlo.
        sin_tolerancia = {k: v for k, v in completa.items() if k != "tolerance_abs"}
        self.assertFalse(self._build(metrics=[sin_tolerancia])["metrics"][0]["comparable"])
        # Con la relativa en vez de la absoluta, sí.
        sin_tolerancia["tolerance_rel"] = 0.001
        self.assertTrue(self._build(metrics=[sin_tolerancia])["metrics"][0]["comparable"])

    def test_the_caller_cannot_declare_its_own_metric_comparable(self):
        m = self._build(metrics=[{"name": "a", "value": 0.9, "comparable": True,
                                  "incomplete": []}])
        self.assertFalse(m["metrics"][0]["comparable"])
        self.assertTrue(m["metrics"][0]["incomplete"])

    # ── el predicado depende de la VALIDEZ, y reparte por etapas ───────

    def test_reproducible_no_longer_rests_on_mere_presence(self):
        # Antes bastaba con que las cinco claves no fueran `None`. Ahora nada
        # inválido llega al predicado, así que «hay dataset_sha256» significa
        # «hay un sha256 que sirve».
        from matrixai.export import ReproduceManifestError
        for over in ({"dataset_sha256": "not-a-sha"}, {"dataset_rows": -4},
                     {"generation": {"seeds": {"dataset": "42"}}}):
            with self.subTest(over=over):
                with self.assertRaises(ReproduceManifestError):
                    self._build(**over)

    def test_what_only_r3_needs_is_declared_instead_of_kept_quiet(self):
        # El paquete que el Studio produce HOY: receta, contrato, huella,
        # filas y semilla de dataset. Es reproducible (R1 + reentrenar) y NO
        # se puede contrastar: falta la semilla de init, el motor y las
        # métricas. Eso NO lo pone en falso — se dice aparte.
        m = self._build()
        self.assertTrue(m["reproducible"])
        self.assertEqual(m["missing"], [])
        r3 = m["verifiable"]["r3"]
        self.assertFalse(r3["possible"])
        self.assertEqual(set(r3["missing"]), {"seed_init", "backend", "device", "metrics"})
        self.assertIn("weight-initialisation seed", r3["reason"])
        # Y el motivo dice POR QUÉ el motor importa: el contrato 60 existió
        # porque torch y stdlib no daban lo mismo.
        self.assertIn("torch", r3["reason"])

    def test_the_stages_speak_the_same_vocabulary_as_the_verifier(self):
        m = self._build()
        self.assertEqual(set(m["verifiable"]), {"manifest", "r1", "training", "r3"})
        # `manifest` siempre se puede intentar: el manifiesto y su digest
        # viajan siempre.
        self.assertTrue(m["verifiable"]["manifest"]["possible"])
        self.assertTrue(m["verifiable"]["r1"]["possible"])

    def test_r1_needs_the_mxtrain_too_because_the_generator_asks_for_it(self):
        # Medido: `_generate_synthetic_dataset(mxai_text, training_text, ...)`
        # recibe el contrato como argumento OBLIGATORIO — sin `.mxtrain` no
        # hay ni siquiera dataset que regenerar, no solo no hay reentrenamiento.
        m = self._build(training_filename=None)
        self.assertFalse(m["verifiable"]["r1"]["possible"])
        self.assertIn("training", m["verifiable"]["r1"]["missing"])

    def test_a_fully_wired_package_can_run_the_four_stages(self):
        m = self._build(
            generation={"seeds": {"dataset": 42, "init": 7}, "backend": "stdlib",
                        "device": "cpu"},
            metrics=[{"name": "accuracy", "value": 0.61, "split": "validation",
                      "dataset_sha256": _DATASET_SHA,
                      "direction": "higher_is_better", "tolerance_abs": 0.0}],
        )
        for etapa in ("manifest", "r1", "training", "r3"):
            self.assertTrue(m["verifiable"][etapa]["possible"], etapa)
            self.assertIsNone(m["verifiable"][etapa]["reason"], etapa)

    # ── §6.7 · el manifiesto no se contradice a sí mismo ───────────────

    def test_the_claim_does_not_say_reproducible_when_it_is_not(self):
        m = self._build(recipe_filename=None, dataset_sha256=None,
                        dataset_rows=None, generation={"seeds": {}})
        self.assertFalse(m["reproducible"])
        # El texto era FIJO: decía «proves ... reproducible» tres líneas por
        # encima de su propio «Not reproducible: …», dentro del mismo fichero.
        self.assertIn("does NOT prove the model can be reproduced", m["claim"])
        self.assertNotIn("and reproducible", m["claim"])
        # Lo que sigue siendo cierto en los dos casos: coherencia sí (§6.7),
        # autenticidad no.
        self.assertIn("internally consistent", m["claim"])
        self.assertIn("authenticity", m["claim"].lower())

    def test_the_claim_says_reproducible_when_it_is(self):
        m = self._build()
        self.assertIn("internally consistent and reproducible", m["claim"])
        self.assertIn("authenticity", m["claim"].lower())

    def test_the_manifest_is_valid_json_and_its_digest_covers_the_new_blocks(self):
        from matrixai.export import verify_manifest_digest
        m = self._build(metrics=[{"name": "a", "value": 0.9}])
        # `NaN`/`Infinity` no son JSON: si se colaran, el fichero no se podría
        # releer y su `manifest_sha256` estaría firmando algo ilegible.
        texto = json.dumps(m, allow_nan=False)
        self.assertIn("verifiable", json.loads(texto))
        self.assertTrue(verify_manifest_digest(m))
        m["verifiable"]["r3"]["possible"] = True
        self.assertFalse(verify_manifest_digest(m))


class LaFichaNoTranquilizaDeMasTest(unittest.TestCase):
    """El README del paquete, en los dos casos (§6.2 y §6.3)."""

    def _readme(self, reproduce):
        from matrixai.export.bundle import _build_readme
        import types
        # Se llama a la función que redacta, no al bundle entero: lo que se
        # prueba aquí es el TEXTO, y montar onnx para eso sería probar otra
        # cosa. El bundle real ya está cubierto más arriba.
        from matrixai.parser import parse_file
        program = parse_file(_FALL_RISK_MXAI)
        export_result = types.SimpleNamespace(
            opset_version=17, external_data=False,
            model_hash="mxai_0", parameter_schema_hash="params_0",
            parameter_set_id="ps_0", output_name="Risk", output_shape=[1, 3],
            input_shape=[1, 5], skipped_functions=[],
        )
        eq_result = None
        return _build_readme(program, export_result, eq_result,
                             inference_spec={"fields": []}, example_input=None,
                             has_params_json=True, external_data=False,
                             smoke_test_skipped=True, has_mxtrain=True,
                             has_recipe=True, reproduce=reproduce)

    def test_a_reproducible_package_says_what_it_still_cannot_check(self):
        # Media verdad tranquilizadora: «reproducible» sin decir que R3 no se
        # puede correr deja a quien descarga creyendo que las métricas están
        # comprobadas.
        reproduce = {
            "reproducible": True,
            "verifiable": {"r3": {"possible": False, "missing": ["metrics"],
                                  "reason": "Cannot contrast the published metrics "
                                            "against their tolerance: no published "
                                            "metric carries what a comparison needs."}},
        }
        texto = self._readme(reproduce)
        self.assertIn("**reproducible**", texto)
        self.assertIn("Cannot contrast the published metrics", texto)

    def test_a_package_that_can_be_checked_end_to_end_adds_no_warning(self):
        reproduce = {"reproducible": True,
                     "verifiable": {"r3": {"possible": True, "missing": [], "reason": None}}}
        texto = self._readme(reproduce)
        self.assertIn("**reproducible**", texto)
        self.assertNotIn("Cannot contrast", texto)

    def test_the_files_table_does_not_promise_what_the_file_may_not_carry(self):
        # Un dibujo —y una tabla— afirman por omisión: en un paquete sin
        # receta, `reproduce.json` no lleva «lo que hace falta para rehacer
        # el modelo», lleva lo que FALTA.
        texto = self._readme({"reproducible": False, "reproducible_reason": "Not reproducible: x."})
        self.assertIn("Whether this model can be rebuilt", texto)
        self.assertNotIn("What it takes to rebuild this model", texto)


class LaCapturaMandaTest(unittest.TestCase):
    """82-C2 · el manifiesto se construye SOLO desde la captura del run.

    EL HALLAZGO, medido con sondas el 2026-08-19 y ejecutado contra el core:
    el paquete se armaba con lo que mandaba la PANTALLA en el momento de
    exportar. Cambiando BATCH, EPOCHS, receta, filas, digest y semilla, los
    valores viajaban al paquete, `conflicts` no existía y el manifiesto se
    declaraba `reproducible: true` igual — un `reproduce.json` podía describir
    OTRO dataset y OTRO contrato que los pesos que iban a su lado.

    `reproducible` dependía de CINCO PRESENCIAS (`.mxtrain`, receta, digest,
    filas y semilla): estando las cinco, daba igual de dónde vinieran.

    Las tres reglas que se prueban aquí:

      1. `reproduce.json` se construye SOLO desde la captura.
      2. Lo que llegue por el payload sirve para DETECTAR CONFLICTO, nunca
         para rellenar. Si discrepan, manda la captura.
      3. Sin captura no hay `reproducible: true`, con un motivo que lo diga:
         el paquete no puede demostrar su relación con los pesos que lleva.
    """

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.td), True)
        shutil.copy2(_FALL_RISK_MXAI, self.td / "model.mxai")
        shutil.copy2(_FALL_RISK_MXTRAIN, self.td / "model.mxtrain")
        (self.td / "data_recipe.txt").write_text(_RECIPE, encoding="utf-8")

    def _build(self, *, provenance="auto", **over):
        from matrixai.export import build_reproduce_manifest
        if provenance == "auto":
            provenance = _captura(generation={"seeds": {"dataset": 42}})
        return build_reproduce_manifest(
            self.td, training_filename="model.mxtrain",
            recipe_filename="data_recipe.txt", run_provenance=provenance, **over)

    # ── REGLA 3 · sin captura no hay promesa ──────────────────────────

    def test_the_five_pieces_are_no_longer_enough(self):
        # Todo presente y bien formado, y nada que ate el paquete a sus pesos.
        m = self._build(provenance=None, dataset_sha256=_DATASET_SHA,
                        dataset_rows=300, generation={"seeds": {"dataset": 42}})
        self.assertFalse(m["reproducible"])
        self.assertEqual(m["missing"][0], "run_provenance")
        self.assertIn("cannot demonstrate its relation to the weights it carries",
                      m["reproducible_reason"])

    def test_without_a_capture_nothing_about_the_run_is_published(self):
        # Ni relleno «razonable» ni ceros: lo que dijo la pantalla no es una
        # respuesta sobre lo que PASÓ, así que no se publica como tal.
        m = self._build(provenance=None, dataset_sha256=_DATASET_SHA,
                        dataset_rows=300,
                        generation={"seeds": {"dataset": 42, "init": 7},
                                    "backend": "torch", "device": "cuda"})
        self.assertIsNone(m["artifacts"]["dataset"])
        self.assertIsNone(m["generation"]["seeds"]["dataset"])
        self.assertIsNone(m["generation"]["seeds"]["init"])
        self.assertIsNone(m["generation"]["backend"])
        self.assertIsNone(m["generation"]["device"])
        # Y NO se pierde: viaja aparte, declarado como lo que es.
        sin = m["provenance"]["unverified_payload"]
        self.assertEqual(sin["dataset_sha256"], _DATASET_SHA)
        self.assertEqual(sin["generation"]["backend"], "torch")
        self.assertEqual(m["provenance"]["source"], "export_payload_only")
        self.assertFalse(m["provenance"]["run_capture"]["present"])

    def test_a_package_without_a_capture_cannot_run_any_stage(self):
        m = self._build(provenance=None)
        for etapa in ("r1", "training", "r3"):
            self.assertFalse(m["verifiable"][etapa]["possible"], etapa)
            self.assertIn("run_provenance", m["verifiable"][etapa]["missing"], etapa)
        # `manifest` sí: el manifiesto y su digest viajan siempre.
        self.assertTrue(m["verifiable"]["manifest"]["possible"])

    # ── REGLA 1 · el manifiesto sale de la captura ────────────────────

    def test_the_manifest_states_what_the_run_captured(self):
        m = self._build(provenance=_captura(
            dataset_sha256="c" * 64, dataset_rows=1234,
            generation={"seeds": {"dataset": 5, "init": 6}, "backend": "torch",
                        "device": "cuda:0", "mode": "coherent"},
            dataset_sha256_prepared="d" * 64))
        self.assertEqual(m["artifacts"]["dataset"]["sha256"], "c" * 64)
        self.assertEqual(m["artifacts"]["dataset"]["sha256_prepared"], "d" * 64)
        self.assertEqual(m["artifacts"]["dataset"]["rows"], 1234)
        self.assertEqual(m["generation"]["seeds"]["dataset"], 5)
        self.assertEqual(m["generation"]["seeds"]["init"], 6)
        self.assertEqual(m["generation"]["backend"], "torch")
        self.assertEqual(m["generation"]["device"], "cuda:0")
        self.assertEqual(m["generation"]["mode"], "coherent")
        self.assertEqual(m["provenance"]["source"], "run_capture")

    def test_the_capture_is_named_by_a_digest_not_copied_into_the_package(self):
        # `mxtrain_text` y `recipe_text` ya viajan como ficheros con su propio
        # digest: repetirlos aquí sería el segundo sitio declarando lo mismo.
        # Pero el manifiesto sí dice DE QUÉ captura sale, y con un nombre que
        # no se puede falsificar.
        cap = _captura(generation={"seeds": {"dataset": 42}})
        m = self._build(provenance=cap)
        firma = m["provenance"]["run_capture"]
        self.assertTrue(firma["present"])
        self.assertEqual(firma["schema_version"], "1.0")
        self.assertEqual(len(firma["sha256"]), 64)
        self.assertNotIn(cap["mxtrain_text"], json.dumps(m))
        # Y cambia con la captura: dos runs distintos no firman igual.
        otra = self._build(provenance=_captura(
            generation={"seeds": {"dataset": 43}}))
        self.assertNotEqual(firma["sha256"], otra["provenance"]["run_capture"]["sha256"])

    def test_the_split_seed_comes_from_the_captured_contract(self):
        # El contrato de la CAPTURA, no el fichero del paquete: si el que
        # viaja fuese otro ya sería un conflicto, y la semilla del reparto
        # tiene que salir del que entrenó de verdad.
        self.assertEqual(self._build()["generation"]["seeds"]["split"], 7)

    # ── REGLA 2 · el payload contrasta, no rellena ────────────────────

    def test_a_payload_that_says_something_else_is_declared_and_loses(self):
        m = self._build(
            dataset_sha256="c" * 64, dataset_rows=999,
            generation={"seeds": {"dataset": 7}, "backend": "torch"},
            provenance=_captura(dataset_sha256=_DATASET_SHA, dataset_rows=300,
                                generation={"seeds": {"dataset": 42},
                                            "backend": "stdlib"}))
        campos = {c["field"]: c for c in m["conflicts"]}
        self.assertEqual(
            set(campos),
            {"artifacts.dataset.sha256", "artifacts.dataset.rows",
             "generation.seeds.dataset", "generation.backend"})
        # MANDA LA CAPTURA, y los dos valores se declaran para que quien
        # recibe el paquete no tenga que adivinar cuál es cuál.
        self.assertEqual(m["artifacts"]["dataset"]["rows"], 300)
        self.assertEqual(m["generation"]["seeds"]["dataset"], 42)
        self.assertEqual(m["generation"]["backend"], "stdlib")
        self.assertEqual(campos["generation.seeds.dataset"]["captured"], 42)
        self.assertEqual(campos["generation.seeds.dataset"]["received"], 7)
        self.assertEqual(campos["generation.backend"]["source"], "export_payload")
        # UN CONFLICTO DECLARADO NO PUEDE DAR `reproducible: true`.
        self.assertFalse(m["reproducible"])
        self.assertEqual(m["missing"], [])
        self.assertIn("the capture is what counts", m["reproducible_reason"])

    def test_a_conflict_blocks_every_stage_not_just_its_own(self):
        m = self._build(dataset_rows=999)
        for etapa in ("r1", "training", "r3"):
            self.assertFalse(m["verifiable"][etapa]["possible"], etapa)
            self.assertIn("artifacts.dataset.rows",
                          m["verifiable"][etapa]["conflicts"], etapa)
            self.assertIn("two versions of the same value",
                          m["verifiable"][etapa]["reason"], etapa)

    def test_the_prepared_digest_is_not_a_contradiction(self):
        # MEDIDO el 2026-08-19: lo que el producto manda hoy por el payload
        # (`trained_csv_sha256`) es el digest del CSV PREPARADO, no el del
        # crudo. Tratarlo como conflicto haría saltar la alarma en todos los
        # paquetes reales — y una alarma que salta siempre no avisa de nada.
        m = self._build(dataset_sha256="d" * 64,
                        provenance=_captura(dataset_sha256=_DATASET_SHA,
                                            dataset_sha256_prepared="d" * 64,
                                            generation={"seeds": {"dataset": 42}}))
        self.assertEqual(m["conflicts"], [])
        self.assertTrue(m["reproducible"])

    def test_what_the_capture_does_not_declare_is_ignored_not_invented(self):
        # La captura no sabe de este parámetro: no hay con qué discrepar, y
        # rellenarlo con lo que diga la pantalla es justo lo que se prohíbe.
        # Se declara ignorado para que se VEA que llegó y no se publicó.
        m = self._build(generation={"seeds": {"dataset": 42, "init": 9},
                                    "field_ranges": {"age": [40, 95]}})
        self.assertIsNone(m["generation"]["seeds"]["init"])
        self.assertIsNone(m["generation"]["field_ranges"])
        self.assertEqual(m["provenance"]["ignored_payload_fields"],
                         ["generation.field_ranges", "generation.seeds.init"])
        self.assertEqual(m["conflicts"], [])
        # Ignorar no es contradecir: el paquete sigue siendo reproducible si
        # la captura trae lo que hace falta.
        self.assertTrue(m["reproducible"])

    # ── LOS ARTEFACTOS SON LOS DEL RUN ────────────────────────────────

    def test_an_artifact_that_is_not_the_one_that_trained_is_declared(self):
        (self.td / "model.mxtrain").write_text("MODEL otro.mxai\n", encoding="utf-8")
        m = self._build()
        conflicto = [c for c in m["conflicts"]
                     if c["field"] == "artifacts.training.sha256"]
        self.assertEqual(len(conflicto), 1)
        self.assertEqual(conflicto[0]["source"], "bundle_file")
        self.assertFalse(m["artifacts"]["training"]["matches_capture"])
        self.assertFalse(m["reproducible"])

    def test_a_recipe_attached_at_export_time_is_not_the_run_s_recipe(self):
        # El caso feo: un modelo entrenado con datos REALES —sin receta que
        # compartir— al que se le engancha una receta al empaquetar. El
        # paquete diría que el dataset se puede regenerar, y el run nunca tuvo
        # ninguno que regenerar.
        m = self._build(provenance=_captura(recipe_text=None,
                                            generation={"seeds": {"dataset": 42}}))
        conflicto = [c for c in m["conflicts"]
                     if c["field"] == "artifacts.recipe.sha256"]
        self.assertEqual(len(conflicto), 1)
        self.assertIsNone(conflicto[0]["captured"])
        self.assertFalse(m["reproducible"])

    def test_a_trailing_newline_is_not_a_different_artifact(self):
        # MEDIDO: la captura digiere el `.mxai` ya `.strip()`eado
        # (`playground.py:3398`) y el bundle copia el fichero BYTE A BYTE, así
        # que un salto de línea final daría dos digests distintos del MISMO
        # programa. Declararlo conflicto sería una falsa alarma en todos los
        # paquetes reales. Un aserto que falla puede estar mal EL ASERTO.
        texto = (self.td / "model.mxai").read_text(encoding="utf-8")
        m = self._build(provenance=_captura(
            mxai_sha256=_sha(texto.strip()),
            generation={"seeds": {"dataset": 42}}))
        self.assertTrue(m["artifacts"]["model"]["matches_capture"])
        self.assertEqual(m["conflicts"], [])

    def test_without_a_capture_matching_is_null_not_false(self):
        # Un valor ausente no es un cero: sin captura no es que el artefacto
        # NO case, es que no hay con qué compararlo.
        m = self._build(provenance=None)
        self.assertIsNone(m["artifacts"]["model"]["matches_capture"])

    # ── LA CAPTURA TAMPOCO PUEDE TENER CUALQUIER FORMA ────────────────

    def test_an_impossible_capture_is_refused_where_it_can_still_be_fixed(self):
        from matrixai.export import ReproduceManifestError
        casos = [
            ({"schema_version": "9.9"}, "schema_version"),
            ({"schema_version": None}, "schema_version"),
            ({"mxai_sha256": "not-a-sha"}, "mxai_sha256"),
            ({"mxtrain_sha256": "a" * 63}, "mxtrain_sha256"),
            ({"mxtrain_text": None}, "mxtrain_text"),
            ({"mxtrain_text": "   "}, "mxtrain_text"),
            ({"dataset_sha256_raw": "data_49219efba673b8d0"}, "dataset_sha256_raw"),
            ({"dataset_rows": 0}, "dataset_rows"),
            ({"dataset_rows": True}, "dataset_rows"),
            ({"seeds": {"dataset": "42"}}, "seeds"),
            ({"seeds": {"init": 42.0}}, "seeds"),
            ({"backend": "tensorflow"}, "backend"),
            ({"recipe_text": None, "recipe_sha256": "a" * 64},
             "recipe_sha256 and recipe_text"),
            ({"recipe_sha256": None}, "recipe_sha256 and recipe_text"),
        ]
        for over, esperado in casos:
            with self.subTest(over=over):
                with self.assertRaises(ReproduceManifestError) as ctx:
                    self._build(provenance=_captura(**over))
                self.assertIn(esperado, str(ctx.exception))

    def test_a_capture_that_is_not_json_never_reaches_the_package(self):
        # Se escribiría con el paquete ya medio hecho, y su `manifest_sha256`
        # estaría firmando un fichero que nadie puede releer.
        from matrixai.export import ReproduceManifestError
        with self.assertRaises(ReproduceManifestError):
            self._build(provenance=_captura(deterministic_options={"x": {1, 2}}))

    # ── EL TEXTO NO PUEDE TRANQUILIZAR DE MÁS ─────────────────────────

    def test_the_claim_says_where_what_it_states_comes_from(self):
        m = self._build()
        self.assertTrue(m["reproducible"])
        self.assertIn("internally consistent and reproducible", m["claim"])
        self.assertIn("matches the authoritative capture", m["claim"])

    def test_the_claim_of_a_package_without_a_capture_says_the_whole_thing(self):
        # «No reproducible» y «no hay con qué demostrar de qué run sale esto»
        # no son lo mismo, y el segundo hay que decirlo entero: lo que se
        # pierde no es una comprobación, es la relación del paquete con los
        # pesos que lleva dentro.
        m = self._build(provenance=None)
        self.assertIn("cannot tie this package to the weights it carries",
                      m["claim"])
        self.assertNotIn("and reproducible", m["claim"])
        self.assertIn("authenticity", m["claim"].lower())

    def test_the_claim_does_not_blame_the_capture_when_there_is_one(self):
        # Un paquete CON captura a la que le falta una pieza no puede decir
        # que no hay captura: sería declarar lo que no pasó.
        m = self._build(provenance=_captura(dataset_sha256=None,
                                            generation={"seeds": {"dataset": 42}}))
        self.assertFalse(m["reproducible"])
        self.assertIn("comes from the capture", m["claim"])
        self.assertNotIn("cannot tie this package", m["claim"])

    def test_the_digest_covers_the_new_blocks(self):
        from matrixai.export import verify_manifest_digest
        m = self._build(dataset_rows=999)
        self.assertTrue(verify_manifest_digest(m))
        json.dumps(m, allow_nan=False)
        # Borrar el conflicto para que parezca limpio rompe el digest.
        m["conflicts"] = []
        self.assertFalse(verify_manifest_digest(m))

    def test_the_readme_of_a_contradicted_package_prints_the_reason(self):
        from matrixai.export.bundle import _build_readme
        import types
        from matrixai.parser import parse_file
        export_result = types.SimpleNamespace(
            opset_version=17, external_data=False, model_hash="mxai_0",
            parameter_schema_hash="params_0", parameter_set_id="ps_0",
            output_name="Risk", output_shape=[1, 3], input_shape=[1, 5],
            skipped_functions=[])
        m = self._build(dataset_rows=999)
        texto = _build_readme(parse_file(_FALL_RISK_MXAI), export_result, None,
                              inference_spec={"fields": []}, example_input=None,
                              has_params_json=True, external_data=False,
                              smoke_test_skipped=True, has_mxtrain=True,
                              has_recipe=True, reproduce=m)
        self.assertNotIn("**reproducible**", texto)
        self.assertIn("the capture is what counts", texto)

    # ── §5 bis · R3 sigue sin poder contrastarse, y se dice ───────────

    def test_a_metric_declares_what_it_lacks_and_r3_stays_unverifiable(self):
        # Hoy el producto no tiene `evaluator_version` (medido con grep) ni
        # tolerancias —salen de repetir en la matriz de entornos, que no se ha
        # hecho—. Rechazar la métrica dejaría el paquete sin métricas;
        # rellenarla sería fabricar lo que el core no ha dicho. Así que cada
        # una declara qué le falta y R3 se queda en no verificable.
        m = self._build(metrics=[{"name": "accuracy", "value": 0.61,
                                  "split": "validation"}])
        metrica = m["metrics"][0]
        self.assertIn("tolerance_abs", metrica["incomplete"])
        self.assertIn("evaluator_version", metrica["incomplete"])
        self.assertIn("direction", metrica["incomplete"])
        self.assertFalse(metrica["comparable"])
        self.assertIsNone(metrica["tolerance_abs"])
        self.assertIsNone(metrica["tolerance_rel"])
        self.assertFalse(m["verifiable"]["r3"]["possible"])
        self.assertIn("metrics", m["verifiable"]["r3"]["missing"])
        # Y no lo pone en falso: reproducir y contrastar son cosas distintas.
        self.assertTrue(m["reproducible"])

    def test_a_metric_measured_on_another_dataset_is_a_contradiction(self):
        # Una métrica dice sobre QUÉ datos se midió. Si ese digest no es el
        # del run, R3 estaría contrastando dos mediciones distintas.
        m = self._build(metrics=[{"name": "accuracy", "value": 0.61,
                                  "split": "validation", "dataset_sha256": "e" * 64,
                                  "direction": "higher_is_better",
                                  "tolerance_abs": 0.0}])
        self.assertIn("metrics[0].dataset_sha256",
                      [c["field"] for c in m["conflicts"]])
        self.assertFalse(m["reproducible"])

    def test_a_metric_measured_on_the_run_s_dataset_goes_through(self):
        # Un banco con dientes muerde solo lo malo.
        m = self._build(metrics=[{"name": "accuracy", "value": 0.61,
                                  "split": "validation",
                                  "dataset_sha256": _DATASET_SHA,
                                  "direction": "higher_is_better",
                                  "tolerance_abs": 0.0}])
        self.assertEqual(m["conflicts"], [])
        self.assertTrue(m["metrics"][0]["comparable"])
        self.assertTrue(m["reproducible"])


if __name__ == "__main__":
    unittest.main()
