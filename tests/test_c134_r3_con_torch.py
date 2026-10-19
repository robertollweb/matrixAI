"""CONTRATO 134 — R3 CON TORCH: un modelo entrenado con torch se puede contrastar.

Medido el 19-10 por `endpoints._studio_verify`: el camino de torch no declaraba
`metric_splits` (R3 salía INCOMPARABLE: ninguna métrica traía `split`) y, peor,
la verificación reentrenaba SIEMPRE sin torch. Con solo lo primero arreglado, R3
habría comparado la exactitud de torch (0,333) con la de stdlib (0,917) y dado
FAIL a un paquete honesto. Las dos piezas van juntas, y estas pruebas las atan:

1. el resultado de torch DECLARA sus particiones;
2. la verificación reentrena con torch cuando el paquete lo declara;
3. sin torch instalado —ocultado DE VERDAD, en otro proceso— es INCOMPARABLE
   con su motivo, y NUNCA FAIL, ni siquiera con una cifra alterada;
4. un paquete stdlib sigue como hoy.

Los paquetes se construyen como el producto: el dataset sale de su receta, y
las cifras publicadas las mide el propio entrenador de torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("torch")

from matrixai.export.verify import verify_package  # noqa: E402

_MXAI = (
    "PROJECT LoanProject\n\n"
    "VECTOR LoanApplicant[2]\n  credit_score: Scalar\n  age: Scalar\nEND\n\n"
    "NETWORK LoanRiskClassifier\n  INPUT LoanApplicant\n"
    "  LAYER Dense units=16 activation=relu\n"
    "  LAYER Dense units=3 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[low, mid, high]\nEND\n\n"
    "GRAPH\n  LoanApplicant -> LoanRiskClassifier\nEND\n"
)
_MXTRAIN = (
    "MODEL model.mxai\n\nDATASET LoanDataset\n"
    "  SOURCE csv(\"loan.train.csv\")\n"
    "  INPUT LoanApplicant FROM COLUMNS [credit_score, age]\n"
    "  TARGET predicted_class: Label[low, mid, high]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS LoanLoss\n  TYPE cross_entropy\n  PREDICTION LoanRiskClassifier\n"
    "  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER LoanOptimizer\n  TYPE sgd\n  LEARNING_RATE 0.05\n"
    "  UPDATE LoanRiskClassifier.*\nEND\n\nRUN\n  EPOCHS 4\nEND\n"
)
_RECETA = "high: age > 0.5\nlow: credit_score < 0.2\nDEFAULT: mid"
_RANGOS = {"credit_score": [0, 1], "age": [0, 1]}
_SEMILLA_INIT = 7
_METRICAS = ("accuracy", "macro_f1", "final_train_loss", "best_validation_loss")

_CSV: list[str] = []


def _csv() -> str:
    if not _CSV:
        from matrixai.playground_api import generate_synthetic_dataset
        r = generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 4242, "coherent", False,
                                       recipe_text=_RECETA)
        assert r.get("ok"), r
        _CSV.append(r["csv_text"])
    return _CSV[0]


def _entrenar_como_el_producto(motor: str) -> dict:
    """El resultado que el producto da para este modelo, con `motor`."""
    from matrixai.playground import _run_playground_dense_training
    with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": motor}):
        r = _run_playground_dense_training(_MXAI, _MXTRAIN, _csv(), 4, seed=_SEMILLA_INIT)
    assert r.get("ok"), r
    assert r.get("backend") == ("torch" if motor == "torch" else "stdlib"), r.get("backend")
    return r


def _paquete(motor: str, *, alterar: str | None = None, dispositivo: str = "cpu") -> Path:
    """Un paquete como el que exporta el producto: cifras medidas por SU entrenador."""
    from matrixai.export.reproduce import añadir_inventario_de_ficheros, write_reproduce_manifest
    res = _entrenar_como_el_producto(motor)
    csv = _csv()
    sha = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    d = Path(tempfile.mkdtemp(prefix="c134-"))
    (d / "model.mxai").write_text(_MXAI, encoding="utf-8")
    (d / "training.mxtrain").write_text(_MXTRAIN, encoding="utf-8")
    (d / "recipe.txt").write_text(_RECETA, encoding="utf-8")
    particiones = res.get("metric_splits") or {}
    metricas = []
    for nombre in _METRICAS:
        valor = res[nombre]
        if nombre == alterar:
            valor = valor + 0.05
        m = {"name": nombre, "value": valor, "dataset_sha256": sha}
        if particiones.get(nombre):
            m["split"] = particiones[nombre]
        metricas.append(m)
    captura = {
        "schema_version": "1.2",
        "mxai_sha256": hashlib.sha256(_MXAI.strip().encode()).hexdigest(),
        "mxtrain_sha256": hashlib.sha256(_MXTRAIN.encode()).hexdigest(),
        "mxtrain_text": _MXTRAIN,
        "recipe_sha256": hashlib.sha256(_RECETA.encode()).hexdigest(),
        "recipe_text": _RECETA,
        "seeds": {"dataset": 4242, "split": 42, "init": _SEMILLA_INIT},
        "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
        "dataset_rows": 60, "dataset_rows_used": 60,
        "epochs_effective": 4, "epochs_ran": 4, "warm_start": False,
        "mode": "coherent", "field_ranges": _RANGOS,
        "backend": res["backend"], "device": dispositivo if motor == "torch" else "cpu",
        "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
    }
    write_reproduce_manifest(
        d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
        dataset_sha256=sha, dataset_rows=60,
        generation={"seeds": {"dataset": 4242, "split": 42, "init": _SEMILLA_INIT}},
        metrics=metricas, run_provenance=captura, weights_source="trained")
    añadir_inventario_de_ficheros(d)
    return d


class ElResultadoDeTorchDeclaraSusParticionesTest(unittest.TestCase):
    def test_el_resultado_de_torch_declara_sobre_que_filas_mide_cada_cifra(self):
        r = _entrenar_como_el_producto("torch")
        self.assertEqual(r["metric_splits"], {
            "accuracy": "validation", "macro_f1": "validation",
            # Clasificación: lo que no se midió es None, no una partición inventada.
            "mae": None, "rmse": None, "r2": None,
            "final_train_loss": "train", "best_validation_loss": "validation"})

    def test_en_regresion_mae_rmse_r2_son_de_VALIDACION_no_full(self):
        """Distinto del camino stdlib, que los mide sobre todo el dataset: en
        torch se evalúan sobre `val_ex`. Declarar `full` compararía contra filas
        que no son las medidas."""
        from matrixai.playground import _run_playground_dense_training
        mxai = ("PROJECT P\n\nVECTOR V[2]\n  a: Scalar\n  b: Scalar\nEND\n\nNETWORK N\n"
                "  INPUT V\n  LAYER Dense units=8 activation=relu\n"
                "  LAYER Dense units=1 activation=linear\n  OUTPUT c: Scalar\nEND\n\n"
                "GRAPH\n  V -> N\nEND\n")
        mxtrain = _MXTRAIN.replace("LoanApplicant", "V").replace(
            "[credit_score, age]", "[a, b]").replace(
            "predicted_class: Label[low, mid, high]", "c: Scalar").replace(
            "cross_entropy", "mse").replace("LoanRiskClassifier", "N").replace(
            "TARGET predicted_class", "TARGET c")
        csv = "a,b,c\n" + "".join(f"{i / 40},{(i * 7 % 40) / 40},{i / 80}\n" for i in range(40))
        with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "torch"}):
            r = _run_playground_dense_training(mxai, mxtrain, csv, 3, seed=1)
        self.assertTrue(r.get("ok"), r)
        self.assertEqual(r["backend"], "torch")
        self.assertEqual(r["metric_splits"], {
            "accuracy": None, "macro_f1": None,
            "mae": "validation", "rmse": "validation", "r2": "validation",
            "final_train_loss": "train", "best_validation_loss": "validation"})


class LaVerificacionReentrenaConElEntrenadorDelPaqueteTest(unittest.TestCase):
    def test_la_verificacion_reentrena_con_torch_cuando_el_paquete_lo_declara(self):
        from matrixai.export import verify as v
        import matrixai.playground as pg
        paquete = _paquete("torch")
        real = pg._dense_torch_train_result
        llamadas = []

        def espia(*a, **k):
            llamadas.append(a[5])  # la semilla
            return real(*a, **k)

        with mock.patch.object(pg, "_dense_torch_train_result", espia), \
             mock.patch("matrixai.training.dense_trainer.DenseSupervisedTrainer.train",
                        side_effect=AssertionError("reentrenó con stdlib")):
            inf = verify_package(paquete, run_training=True)
        self.assertEqual(llamadas, [_SEMILLA_INIT], "no usó el camino de torch con la semilla")
        t = inf["stages"]["training"]
        self.assertEqual(t["status"], "PASS", t.get("reason"))
        self.assertEqual(t["not_applied_from_capture"], [])
        self.assertEqual(t["applied_from_capture"]["device"], "cpu")
        self.assertEqual(inf["stages"]["R3"]["status"], "PASS", inf["stages"]["R3"])
        self.assertEqual(sorted(inf["stages"]["R3"]["checked"]), sorted(_METRICAS))
        self.assertTrue(hasattr(v, "_reentrenar_con_torch"))

    def test_con_una_cifra_publicada_alterada_R3_da_FAIL(self):
        """El control de que compara de verdad: con el MISMO camino y una cifra
        tocada a mano, FAIL; si esto diera PASS, el PASS de arriba no diría nada."""
        for tocada in ("accuracy", "final_train_loss"):
            with self.subTest(tocada=tocada):
                inf = verify_package(_paquete("torch", alterar=tocada), run_training=True)
                r3 = inf["stages"]["R3"]
                self.assertEqual(r3["status"], "FAIL", r3)
                self.assertEqual([m["metric"] for m in r3["metrics"]], [tocada])

    def test_pedir_CUDA_sin_tenerla_no_se_disfraza_de_CPU(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest("aquí hay CUDA")
        inf = verify_package(_paquete("torch", dispositivo="cuda"), run_training=True)
        self.assertEqual(inf["stages"]["training"]["not_applied_from_capture"], ["device"])
        self.assertEqual(inf["stages"]["R3"]["status"], "INCOMPARABLE")

    def test_un_paquete_stdlib_sigue_como_hoy(self):
        import matrixai.playground as pg
        paquete = _paquete("stdlib")
        with mock.patch.object(pg, "_dense_torch_train_result",
                               side_effect=AssertionError("un paquete stdlib no usa torch")):
            inf = verify_package(paquete, run_training=True)
        t = inf["stages"]["training"]
        self.assertEqual(t["status"], "PASS", t.get("reason"))
        self.assertEqual(t["not_applied_from_capture"], [])
        self.assertNotIn("not_applied_reasons", t)
        self.assertEqual(inf["stages"]["R3"]["status"], "PASS", inf["stages"]["R3"])


#: Un proceso donde `import torch` FALLA de verdad (un buscador en `sys.meta_path`),
#: no un núcleo que supone que no está: es lo que vería otra máquina.
_SIN_TORCH = textwrap.dedent("""
    import importlib.abc, json, sys
    class _Sin(importlib.abc.MetaPathFinder):
        def find_spec(self, nombre, ruta=None, destino=None):
            if nombre == "torch" or nombre.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch' (oculto por la prueba)")
    sys.meta_path.insert(0, _Sin())
    try:
        import torch
        print(json.dumps({"torch_oculto": False})); sys.exit(0)
    except ModuleNotFoundError:
        pass
    from matrixai.export.verify import verify_package
    inf = verify_package(sys.argv[1], run_training=True, locale=sys.argv[2])
    print(json.dumps({"torch_oculto": True, "stages": inf["stages"], "exit": inf["exit_code"]},
                     default=str))
""")


class SinTorchEsINCOMPARABLEnuncaFAILTest(unittest.TestCase):
    def _sin_torch(self, paquete: Path, locale: str = "es") -> dict:
        p = subprocess.run([sys.executable, "-c", _SIN_TORCH, str(paquete), locale],
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertTrue(out["torch_oculto"], "el torch no quedó oculto: la prueba no mide nada")
        return out

    def test_sin_torch_instalado_es_INCOMPARABLE_con_su_motivo_y_nunca_FAIL(self):
        for alterada in (None, "accuracy"):
            with self.subTest(cifra_alterada=alterada):
                out = self._sin_torch(_paquete("torch", alterar=alterada))
                t, r3 = out["stages"]["training"], out["stages"]["R3"]
                self.assertEqual(t["status"], "PASS", t.get("reason"))
                self.assertEqual(t["not_applied_from_capture"], ["backend"])
                self.assertIn("torch", t["not_applied_reasons"]["backend"])
                self.assertEqual(r3["status"], "INCOMPARABLE", r3)
                self.assertIn("backend", r3["reason"])
                self.assertIn("no está instalado", r3["reason"])

    def test_el_motivo_sale_tambien_en_ingles(self):
        out = self._sin_torch(_paquete("torch", alterar="accuracy"), "en")
        r3 = out["stages"]["R3"]
        self.assertEqual(r3["status"], "INCOMPARABLE", r3)
        self.assertIn("not installed here", r3["reason"])
        self.assertIn("not installed here",
                      out["stages"]["training"]["not_applied_reasons"]["backend"])

    def test_un_paquete_stdlib_sin_torch_sigue_dando_PASS(self):
        out = self._sin_torch(_paquete("stdlib"))
        self.assertEqual(out["stages"]["R3"]["status"], "PASS", out["stages"]["R3"])


#: El caso REAL de la auditoría (H3): torch NO INSTALADO, no escondido. `python3 -s -E`
#: deja fuera el `site-packages` del usuario —donde vive torch en esta máquina— y las
#: variables de entorno; el árbol entra por `sys.path`. Así torch tampoco está en la
#: huella del entorno, y salta antes la guarda de «otro entorno» (lo que no veía la
#: prueba con `sys.meta_path`, que deja sus metadatos).
_SIN_TORCH_DE_VERDAD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    try:
        import torch
        print(json.dumps({"sin_torch": False})); sys.exit(0)
    except ImportError:
        pass
    import matrixai
    assert matrixai.__file__.startswith(sys.argv[1]), matrixai.__file__
    if sys.argv[4] == "cli":
        from matrixai.cli import main
        sys.argv = ["matrixai", "verify", sys.argv[2], "--retrain", "--locale", sys.argv[3]]
        main()
        sys.exit(0)
    from matrixai.export.verify import verify_package
    inf = verify_package(sys.argv[2], run_training=True, locale=sys.argv[3])
    print(json.dumps({"sin_torch": True, "stages": inf["stages"]}, default=str))
""")

_ARBOL = str(Path(__file__).resolve().parents[1])


class SinTorchDeVerdadElMotivoLlegaALaPersonaTest(unittest.TestCase):
    """AUDITORÍA DEL 134, H3: medido con `python3 -s -E`, R3 salía INCOMPARABLE por «otro
    entorno» y «se entrenó con torch y aquí no hay» solo estaba en `--json`."""

    def _correr(self, paquete: Path, locale: str, modo: str) -> str:
        p = subprocess.run([sys.executable, "-s", "-E", "-c", _SIN_TORCH_DE_VERDAD, _ARBOL,
                            str(paquete), locale, modo],
                           capture_output=True, text=True, timeout=600, cwd="/", env={})
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        if '"sin_torch": false' in p.stdout:
            self.skipTest("aquí torch está instalado fuera del site del usuario: no se puede medir")
        return p.stdout

    def test_R3_da_el_porque_del_backend_aunque_salte_antes_la_guarda_del_entorno(self):
        paquete = _paquete("torch", alterar="accuracy")
        for locale, frase in (("es", "no está instalado"), ("en", "not installed here")):
            with self.subTest(locale=locale):
                out = json.loads(self._correr(paquete, locale, "json").strip().splitlines()[-1])
                t, r3 = out["stages"]["training"], out["stages"]["R3"]
                self.assertEqual(t["not_applied_from_capture"], ["backend"], t)
                # Con una cifra ALTERADA, y aun así nunca FAIL.
                self.assertEqual(r3["status"], "INCOMPARABLE", r3)
                self.assertIn(frase, r3["reason"])
                self.assertEqual(r3.get("not_applied"), ["backend"], r3)

    def test_la_salida_humana_de_la_CLI_ensena_el_motivo_del_backend(self):
        texto = self._correr(_paquete("torch"), "es", "cli")
        lineas = texto.splitlines()
        self.assertTrue(any(l.startswith("training") and "PASS" in l for l in lineas), texto)
        # La línea propia del porqué, debajo de `training` (no basta con que lo diga R3).
        self.assertTrue(any(l.strip().startswith("· backend:") and "no está instalado" in l
                            for l in lineas), texto)


class LasGuardasDelCaminoDeTorchTienenPruebaTest(unittest.TestCase):
    """AUDITORÍA DEL 134, H4: dos sabotajes de la auditoría salían VERDES."""

    def test_un_dataset_partido_en_train_y_eval_da_NOT_RUN_con_su_motivo(self):
        """El camino de torch lee UN csv; el nombre `-synthetic-train.csv` es la
        convención de `generate-dataset` (CLI), y ahí no se entrena con la mitad."""
        global _MXTRAIN
        original = _MXTRAIN
        try:
            _MXTRAIN = original.replace("loan.train.csv", "loan-synthetic-train.csv")
            self.assertIn("loan-synthetic-train.csv", _MXTRAIN)
            paquete = _paquete("torch")
        finally:
            _MXTRAIN = original
        for locale, frase in (("es", "se parte en train y eval"), ("en", "split into train and eval")):
            with self.subTest(locale=locale):
                inf = verify_package(paquete, run_training=True, locale=locale)
                t, r3 = inf["stages"]["training"], inf["stages"]["R3"]
                self.assertEqual(t["status"], "INCOMPARABLE", t)
                self.assertIn(frase, t["reason"])
                self.assertEqual(r3["status"], "NOT_RUN", r3)
                self.assertIn(frase, r3["reason"])

    def test_torch_declarado_en_una_red_compuesta_dice_que_no_hay_camino_de_torch(self):
        """Una red compuesta (aquí, con `LayerNorm`) entrenada con torch: el
        reentrenamiento no tiene un camino de torch que repetir, y el motivo llega
        aunque el entrenador estándar no pueda con ella."""
        global _MXAI
        original = _MXAI
        from matrixai.playground import _network_is_composite, _run_playground_composite_training
        try:
            _MXAI = original.replace("  LAYER Dense units=3 activation=softmax\n",
                                     "  LAYER LayerNorm\n  LAYER Dense units=3 activation=softmax\n")
            self.assertTrue(_network_is_composite(_MXAI))
            with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "torch"}):
                res = _run_playground_composite_training(_MXAI, _MXTRAIN, _csv(), 4, seed=_SEMILLA_INIT)
            self.assertTrue(res.get("ok"), res)
            self.assertEqual(res.get("backend"), "torch")
            with mock.patch(f"{__name__}._entrenar_como_el_producto", lambda motor: res):
                paquete = _paquete("torch")
        finally:
            _MXAI = original
        for locale, frase in (("es", "no tiene un reentrenamiento de torch"),
                              ("en", "no torch retraining that can be repeated")):
            with self.subTest(locale=locale):
                inf = verify_package(paquete, run_training=True, locale=locale)
                t, r3 = inf["stages"]["training"], inf["stages"]["R3"]
                self.assertIn(frase, t["not_applied_reasons"]["backend"], t)
                self.assertIn(frase, t["reason"])
                self.assertNotEqual(r3["status"], "FAIL", r3)
                self.assertIn(frase, r3["reason"])


class CUDASinMedirNoSeComparaTest(unittest.TestCase):
    def test_reentrenar_en_CUDA_deja_device_sin_aplicar_con_su_motivo(self):
        """AUDITORÍA DEL 134, H2: la tolerancia 0,0 de torch se midió en CPU, no en CUDA.
        Aquí no hay GPU: se finge que CUDA existe y el entrenamiento corre en CPU (lo que
        se prueba es la DECISIÓN, no CUDA). Con una cifra alterada, nunca FAIL."""
        from matrixai.export import verify as v
        import matrixai.playground as pg
        real = pg._dense_torch_train_result
        dispositivos = []

        def en_cpu(mxai, training, spec, csv, dispositivo, *a, **k):
            dispositivos.append(dispositivo)
            return real(mxai, training, spec, csv, "cpu", *a, **k)

        paquete = _paquete("torch", alterar="accuracy", dispositivo="cuda")
        with mock.patch.object(v, "_dispositivo_de_torch", lambda declarado: "cuda"), \
             mock.patch.object(pg, "_dense_torch_train_result", en_cpu):
            inf = verify_package(paquete, run_training=True)
        self.assertEqual(dispositivos, ["cuda"], "no reentrenó en la máquina de la captura")
        t, r3 = inf["stages"]["training"], inf["stages"]["R3"]
        self.assertEqual(t["status"], "PASS", t)
        self.assertIn("device", t["not_applied_from_capture"])
        self.assertNotIn("device", t["applied_from_capture"])
        self.assertIn("CUDA", t["not_applied_reasons"]["device"])
        self.assertEqual(r3["status"], "INCOMPARABLE", r3)
        self.assertIn("no está medida", r3["reason"])


class LasCifrasDeUnaSolaFilaNoSonDeValidacionTest(unittest.TestCase):
    def test_con_una_fila_la_validacion_es_la_de_entrenamiento_y_no_se_declara(self):
        """AUDITORÍA DEL 134, H6: con una sola fila `val_ex` es la propia fila de
        entrenamiento; esas cifras no se declaran `validation`."""
        from matrixai.playground import _run_playground_dense_training
        una = "\n".join(_csv().splitlines()[:2]) + "\n"
        with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "torch"}):
            r = _run_playground_dense_training(_MXAI, _MXTRAIN, una, 2, seed=_SEMILLA_INIT)
        self.assertTrue(r.get("ok"), r)
        self.assertEqual(r["backend"], "torch")
        self.assertEqual(r["metric_splits"], {
            "accuracy": None, "macro_f1": None, "mae": None, "rmse": None, "r2": None,
            "final_train_loss": "train", "best_validation_loss": None})


class LasMetricasDelReentrenamientoSonLasDelCatalogoTest(unittest.TestCase):
    def test_lo_que_recoge_el_reentrenamiento_de_torch_es_el_catalogo(self):
        """AUDITORÍA DEL 134, H7: una sola lista. El catálogo (`metric_identity.DIRECCION`)
        da la tolerancia y sin ella R3 no compara, así que lo que el reentrenamiento
        recoge es el catálogo, no una copia: una métrica nueva en él entra sola."""
        from matrixai.training import metric_identity
        paquete = _paquete("torch")
        con_una_mas = {**metric_identity.DIRECCION, "best_epoch": "higher_is_better"}
        with mock.patch.object(metric_identity, "DIRECCION", con_una_mas):
            inf = verify_package(paquete, run_training=True)
        medidas = inf["stages"]["training"]["metrics"]
        self.assertIn("best_epoch", medidas)
        self.assertEqual(set(medidas), {"accuracy", "macro_f1", "final_train_loss",
                                        "best_validation_loss", "best_epoch"})


class LaToleranciaDeTorchEstaMedidaTest(unittest.TestCase):
    def test_torch_en_CPU_con_semilla_repite_BIT_A_BIT(self):
        """MEDIDO el 19-10 (contrato 134 §2.4): el mismo paquete reentrenado en 12
        configuraciones —procesos distintos, 1/2/3/8 hilos, hilo principal y hilo
        aparte; modelo chico, de 6 entradas y 30 épocas, y de regresión— dio las
        MISMAS cifras hasta el último dígito. Por eso la tolerancia de torch es la
        del catálogo (0,0, alcance «mismo entorno, misma semilla»), no una
        inventada. Esta prueba guarda la medida: si un día torch deja de repetir,
        R3 daría FAIL a paquetes honestos y la tolerancia habría que medirla otra
        vez, no suponerla."""
        a, b = (_entrenar_como_el_producto("torch") for _ in range(2))
        for nombre in _METRICAS + ("best_epoch",):
            self.assertEqual(repr(a[nombre]), repr(b[nombre]), nombre)
        from matrixai.training.metric_identity import tolerancia_medida
        self.assertEqual(tolerancia_medida("accuracy")["tolerance_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
