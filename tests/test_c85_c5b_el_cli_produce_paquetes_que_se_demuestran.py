"""EL CAMINO DE QUIEN HACE `pip install` YA PRODUCE UN PAQUETE QUE SE DEMUESTRA.

Medido el 2026-08-25, antes de esto: un paquete exportado desde el CLI salía
**`Reproducible: no`** con CINCO motivos —no consta que los pesos vengan de un
entrenamiento, no viaja captura, no se sabe el sha256 del dataset, ni sus
filas, ni la semilla—. O sea que la galería del 85-C5 («reprodúcelo tú») no se
podía cumplir por el único camino que tiene alguien de fuera.

Y no era por falta de datos: `generate-dataset` sabe la semilla, el modo y las
filas; `train` sabe que entrenó. **Faltaba que alguien lo escribiera.**

Después: `manifest PASS · R1 PASS · training PASS · R3 PASS`.

Las tres cosas que costaron, y que están fijadas aquí:

1. **El fichero de entrenamiento es un TRAMO de la generación**, no la
   generación entera (`generate-dataset` reparte después de generar). Declarar
   su digest hacía que R1 regenerara una cosa y comparara con otra: **FAIL
   sobre un paquete honesto**.
2. **Sin el TOTAL de filas no se puede comprobar la receta**: regenerar «tantas
   como tiene el train» produce otro dataset.
3. **El dispositivo no se impone: se compara.** Imponerlo sería prometer un
   entorno que no se controla; comprobar que coincide es un hecho.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.export.reproduce import ReproduceManifestError, capturar_run
from matrixai.export.verify import _configuracion_del_run


class LaCapturaSeConstruyeYSeValidaTest(unittest.TestCase):
    def _captura(self, **extra):
        return capturar_run(mxai_text="NETWORK N", mxtrain_text="MODEL m.mxai", **extra)

    def test_lleva_la_identidad_del_run(self):
        c = self._captura()
        self.assertEqual(len(c["mxai_sha256"]), 64)
        self.assertEqual(len(c["mxtrain_sha256"]), 64)
        self.assertEqual(c["mxtrain_text"], "MODEL m.mxai")

    def test_una_semilla_que_no_se_sabe_NO_viaja_como_nula(self):
        """Un `None` dentro de `seeds` parece una semilla declarada."""
        c = self._captura(seeds={"split": 42, "dataset": None, "init": None})
        self.assertEqual(c["seeds"], {"split": 42})

    def test_sin_ninguna_semilla_no_se_escribe_la_clave(self):
        self.assertNotIn("seeds", self._captura(seeds={"dataset": None}))

    def test_el_modo_viaja_porque_sin_el_R1_no_compara(self):
        self.assertEqual(self._captura(mode="coherent")["mode"], "coherent")

    def test_warm_start_false_es_una_AFIRMACION(self):
        self.assertIs(self._captura(warm_start=False)["warm_start"], False)

    def test_la_receta_viaja_con_su_digest_o_no_viaja(self):
        c = self._captura(recipe_text="alto: a > 1")
        self.assertEqual(len(c["recipe_sha256"]), 64)
        self.assertNotIn("recipe_sha256", self._captura())

    def test_una_captura_IMPOSIBLE_se_corta_al_construirla(self):
        """El mismo validador que la leerá después. Si esto no pasa por ahí, no
        vale de nada haberla escrito."""
        with self.assertRaises(ReproduceManifestError):
            capturar_run(mxai_text="N", mxtrain_text="M",
                         recipe_verification={"verified": "puede", "code": "x"})


class ElDispositivoSeComparaNoSeImponeTest(unittest.TestCase):
    def _manifiesto(self, device):
        return {"generation": {"device": device, "seeds": {"init": 7}}}

    def test_si_COINCIDE_queda_aplicado(self):
        aplicadas, sin_aplicar, _, _ = _configuracion_del_run(
            self._manifiesto("cpu"), object(), admite_semilla=False, maquina="cpu")
        self.assertEqual(aplicadas.get("device"), "cpu")
        self.assertNotIn("device", sin_aplicar)

    def test_si_NO_coincide_sigue_sin_aplicarse(self):
        """Y es la verdad: sus métricas no describen la misma ejecución."""
        _, sin_aplicar, _, _ = _configuracion_del_run(
            self._manifiesto("cuda"), object(), admite_semilla=False, maquina="cpu")
        self.assertIn("device", sin_aplicar)

    def test_si_NO_SE_SABE_no_se_afirma_nada(self):
        aplicadas, sin_aplicar, _, _ = _configuracion_del_run(
            self._manifiesto("cpu"), object(), admite_semilla=False, maquina=None)
        self.assertNotIn("device", aplicadas)
        self.assertIn("device", sin_aplicar)


class LoQueElCLI_LeeDelManifiestoTest(unittest.TestCase):
    """`_generacion_declarada` saca semilla, filas TOTALES y modo de donde los
    escribió `generate-dataset`, y sin el total no se puede comprobar nada."""

    def _args(self, **kw):
        return type("Args", (), kw)()

    def test_del_manifiesto_salen_los_tres(self):
        from matrixai.cli import _generacion_declarada
        with TemporaryDirectory() as d:
            ruta = Path(d) / "m.json"
            ruta.write_text(json.dumps(
                {"generator": {"seed": 7, "rows": 400, "mode": "coherent"}}),
                encoding="utf-8")
            self.assertEqual(
                _generacion_declarada(self._args(dataset_manifest=str(ruta),
                                                 dataset_seed=None)),
                (7, 400, "coherent"))

    def test_sin_manifiesto_queda_la_semilla_suelta_y_NINGUN_total(self):
        from matrixai.cli import _generacion_declarada
        semilla, filas, _ = _generacion_declarada(
            self._args(dataset_manifest=None, dataset_seed=99))
        self.assertEqual(semilla, 99)
        self.assertIsNone(filas)

    def test_un_manifiesto_ilegible_no_tumba_el_entrenamiento(self):
        from matrixai.cli import _generacion_declarada
        with TemporaryDirectory() as d:
            ruta = Path(d) / "m.json"
            ruta.write_text("{no es json", encoding="utf-8")
            self.assertEqual(
                _generacion_declarada(self._args(dataset_manifest=str(ruta),
                                                 dataset_seed=5)),
                (5, None, "coherent"))


if __name__ == "__main__":
    unittest.main()


class LaExactitudSEESCRIBE_noSoloSeImprimeTest(unittest.TestCase):
    """Medido el 2026-08-25: el CLI enseñaba `Accuracy: 0.968750` por pantalla y
    **no constaba en ningún fichero**, así que no podía viajar al paquete ni
    contrastarse en R3. Estaba calculada a cuatro líneas del sitio donde se
    escribe la traza."""

    def _entrenar(self, d: Path):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        from matrixai.training.parser import parse_training_text
        mxai = (
            "PROJECT P\n\nVECTOR E[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
            "NETWORK N\n  INPUT E\n  LAYER Dense units=4 activation=relu\n"
            "  LAYER Dense units=2 activation=softmax\n"
            "  OUTPUT predicted_class: ProbabilityMap[si, no]\nEND\n\n"
            "GRAPH\n  E -> N\nEND\n")
        mxtrain = (
            "MODEL p.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
            "  INPUT E FROM COLUMNS [a, b]\n"
            "  TARGET predicted_class: Label[si, no]\n"
            "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=4\nEND\n\n"
            "LOSS L\n  TYPE cross_entropy\n  PREDICTION N\n  TARGET predicted_class\nEND\n\n"
            "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.5\n  UPDATE N.*\nEND\n\n"
            "RUN\n  EPOCHS 3\nEND\n")
        (d / "p.mxai").write_text(mxai, encoding="utf-8")
        (d / "p.mxtrain").write_text(mxtrain, encoding="utf-8")
        filas = ["a,b,predicted_class"]
        for i in range(40):
            x = i / 40
            filas.append(f"{x},{1 - x},{'si' if x > 0.5 else 'no'}")
        (d / "d.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
        spec = parse_training_text(mxtrain)
        return DenseSupervisedTrainer().train(
            spec, output_dir=str(d / "run"), base_path=d,
            training_path=d / "p.mxtrain")

    def test_la_traza_guarda_la_exactitud_que_el_CLI_imprime(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            resultado = self._entrenar(d)
            traza = json.loads((Path(resultado.output_dir) / "training_trace.json")
                               .read_text(encoding="utf-8"))
        self.assertIsNotNone(traza.get("accuracy"))
        self.assertAlmostEqual(traza["accuracy"], resultado.accuracy, places=9)

    def test_y_tambien_las_metricas_de_clasificacion_que_ya_calculaba(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            resultado = self._entrenar(d)
            traza = json.loads((Path(resultado.output_dir) / "training_trace.json")
                               .read_text(encoding="utf-8"))
        # `validation_metrics` es lo que la pantalla y el export saben leer.
        self.assertIn("macro_f1", traza.get("validation_metrics") or {})


class LasDireccionesDeLasMetricasTest(unittest.TestCase):
    """Una dirección al revés hace que R3 **apruebe un modelo que empeoró**: si
    la exactitud se declarara `lower_is_better`, bajar de 0,96 a 0,50 se leería
    como una mejora dentro de tolerancia."""

    def test_mas_exactitud_es_mejor_y_menos_perdida_tambien(self):
        import inspect

        from matrixai import cli
        fuente = inspect.getsource(cli._cmd_export_bundle)
        self.assertIn('"name": "accuracy"', fuente)
        # La exactitud, hacia arriba; la pérdida, hacia abajo. Se comprueba que
        # cada una lleva la suya y no la contraria.
        bloque_exactitud = fuente.split('"name": "accuracy"')[1].split("}")[0]
        self.assertIn("higher_is_better", bloque_exactitud)
        bloque_perdida = fuente.split('"name": "best_validation_loss"')[1].split("}")[0]
        self.assertIn("lower_is_better", bloque_perdida)
