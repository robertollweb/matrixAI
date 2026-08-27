"""R3: LO QUE EL REENTRENAMIENTO APLICA DE LA CAPTURA, Y LO QUE NO (2026-08-24).

Medido conduciendo el producto en cuanto la cadena entera pudo correr por
primera vez: R3 salía **FAIL** acusando al paquete —«al menos una métrica se
sale de su tolerancia», con la tolerancia en 0,0— mientras el informe decía

    applied_from_capture: {}      not_applied_from_capture: []

o sea, «no apliqué nada» y «no dejé nada sin aplicar» a la vez. Una lista vacía
ahí afirma por omisión: se lee como que se aplicó todo.

Debajo había TRES defectos en una función de veinticinco líneas:

1. leía `manifiesto["run_provenance"]`, **que no existe** en el manifiesto (la
   raíz tiene `provenance`, y `provenance.run_capture` es solo un resumen con
   `present/schema_version/sha256`), así que devolvía `({}, [])` siempre;
2. se llamaba **después** de entrenar, así que aplicar épocas o semilla no
   podía cambiar un run que ya había corrido;
3. buscaba `seeds["training"]`, y las semillas publicadas son `dataset`,
   `split` e `init`.

Y al arreglarlos salió un cuarto: `spec.epochs` **no existe** —las épocas viven
en `spec.run.epochs` y las dos dataclases son `frozen`—, así que la rama de las
épocas tampoco entraba nunca.
"""
from __future__ import annotations

import unittest

from matrixai.export.verify import _configuracion_del_run, _spec_con_epocas
from matrixai.training.parser import parse_training_text

_MXTRAIN = (
    "MODEL m.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT E FROM COLUMNS [a, b]\n"
    "  TARGET predicted_class: Label[alto, bajo]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION N\n  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE N.*\nEND\n\n"
    "RUN\n  EPOCHS 5\nEND\n"
)


def _spec():
    return parse_training_text(_MXTRAIN)


def _manifiesto(**generacion):
    base = {"seeds": {"dataset": 1, "split": 42, "init": 7},
            "epochs_effective": 60, "device": "cpu", "warm_start": False}
    base.update(generacion)
    return {"generation": base}


class LaConfiguracionSaleDeGENERATIONTest(unittest.TestCase):
    def test_la_semilla_de_INICIALIZACION_se_aplica_por_el_entrenador(self):
        aplicadas, sin_aplicar, del_run, _spec_nuevo = _configuracion_del_run(
            _manifiesto(), _spec(), admite_semilla=True)
        # No se toca el contrato: la semilla de init es un argumento de
        # `train(..., seed=)`, y por eso viaja aparte.
        self.assertEqual(del_run, {"seed": 7})
        self.assertEqual(aplicadas["seeds.init"], 7)
        self.assertNotIn("seeds", sin_aplicar)

    def test_si_el_entrenador_NO_admite_semilla_se_DICE(self):
        """Los modelos FUNCTION entrenan sin semilla. Callarlo dejaría a R3
        comparando contra otra inicialización y llamándolo veredicto."""
        aplicadas, sin_aplicar, del_run, _ = _configuracion_del_run(
            _manifiesto(), _spec(), admite_semilla=False)
        self.assertEqual(del_run, {})
        self.assertIn("seeds.init", sin_aplicar)

    def test_las_epocas_del_RUN_ganan_a_las_del_contrato(self):
        spec = _spec()
        self.assertEqual(spec.run.epochs, 5)      # lo que dice el `.mxtrain`
        aplicadas, _, _, nuevo = _configuracion_del_run(
            _manifiesto(), spec, admite_semilla=True)
        self.assertEqual(aplicadas["epochs_effective"], 60)
        self.assertEqual(nuevo.run.epochs, 60)
        # Y el original no se toca: son dataclases congeladas.
        self.assertEqual(spec.run.epochs, 5)

    def test_warm_start_FALSE_esta_aplicado_porque_reentrenar_es_eso(self):
        aplicadas, sin_aplicar, _, _ = _configuracion_del_run(
            _manifiesto(), _spec(), admite_semilla=True)
        self.assertIs(aplicadas["warm_start"], False)
        self.assertNotIn("warm_start", sin_aplicar)

    def test_warm_start_CON_pesos_no_se_puede_aplicar_y_se_dice(self):
        m = _manifiesto(warm_start={"sha256": "a" * 64})
        _, sin_aplicar, _, _ = _configuracion_del_run(m, _spec(), admite_semilla=True)
        self.assertIn("warm_start", sin_aplicar)

    def test_el_dispositivo_NO_se_impone_y_se_declara(self):
        """Forzar el entorno de quien verifica sería prometer algo que no se
        controla. Lo honesto es decir que no se aplicó."""
        _, sin_aplicar, _, _ = _configuracion_del_run(
            _manifiesto(), _spec(), admite_semilla=True)
        self.assertIn("device", sin_aplicar)

    def test_un_manifiesto_sin_GENERATION_no_inventa_nada(self):
        aplicadas, sin_aplicar, del_run, _ = _configuracion_del_run(
            {}, _spec(), admite_semilla=True)
        self.assertEqual((aplicadas, sin_aplicar, del_run), ({}, [], {}))


class LasEpocasSeAplicanConstruyendoTest(unittest.TestCase):
    def test_sin_bloque_RUN_no_hay_donde_aplicarlas(self):
        spec = _spec()
        object.__setattr__(spec, "run", None)
        self.assertIsNone(_spec_con_epocas(spec, 10))

    def test_un_valor_que_no_es_un_entero_positivo_no_se_aplica(self):
        for malo in ("diez", None, 0, -3):
            with self.subTest(malo=malo):
                self.assertIsNone(_spec_con_epocas(_spec(), malo))


if __name__ == "__main__":
    unittest.main()


class NoAplicaNoEsLoMismoQueNoSePudoTest(unittest.TestCase):
    """Medido el 2026-08-25 montando la galería: el paquete del Kelvin se
    quedaba en `R3 INCOMPARABLE` **para siempre** porque el entrenador de los
    modelos FUNCTION no acepta `seed`. Pero su inicialización **no depende de
    ninguna**: `build_initial_parameter_set` devuelve `W1 = [0.05]` y `b1 = 0.0`
    dos veces seguidas.

    O sea que el reentrenamiento SÍ reproduce esa inicialización, y llamarlo
    «sin aplicar» era declarar una limitación que no existe. Lo declara el
    entrenador (`inicializacion_determinista`), no lo adivina el verificador.
    """

    def test_si_la_inicializacion_es_determinista_la_semilla_NO_bloquea(self):
        aplicadas, sin_aplicar, del_run, _ = _configuracion_del_run(
            _manifiesto(), _spec(), admite_semilla=False,
            inicializacion_determinista=True)
        self.assertNotIn("seeds.init", sin_aplicar)
        self.assertIn("seeds.init", aplicadas)
        self.assertIn("no aplica", str(aplicadas["seeds.init"]))
        # Y no se le pasa nada al entrenador: no la acepta.
        self.assertEqual(del_run, {})

    def test_si_NO_es_determinista_sigue_siendo_una_limitacion(self):
        _, sin_aplicar, _, _ = _configuracion_del_run(
            _manifiesto(), _spec(), admite_semilla=False,
            inicializacion_determinista=False)
        self.assertIn("seeds.init", sin_aplicar)

    def test_el_entrenador_FUNCTION_lo_declara(self):
        from matrixai.training.trainer import SupervisedTrainer
        self.assertTrue(SupervisedTrainer.inicializacion_determinista)

    def test_y_su_inicializacion_lo_CUMPLE(self):
        """La declaración sin la medición sería una promesa. Aquí se comprueba
        que de verdad arranca siempre igual."""
        from matrixai.parameters.store import build_initial_parameter_set
        from matrixai.parser.parser import parse_text

        programa = parse_text(
            "PROJECT P\n\nVECTOR E[1]\n  x: Scalar\nEND\n\n"
            "PARAM W1 Vector[1]\nEND\n\nPARAM b1 Scalar\nEND\n\n"
            "FUNCTION M\n  y: Scalar = linear(W1 * E + b1)\nEND\n\n"
            "GRAPH\n  E -> M\nEND\n")
        a = build_initial_parameter_set(programa, parameter_set_id="a")
        b = build_initial_parameter_set(programa, parameter_set_id="b")
        self.assertEqual(a.parameters["W1"]["values"], b.parameters["W1"]["values"])
        self.assertEqual(a.parameters["b1"]["values"], b.parameters["b1"]["values"])


class LasMetricasQueQUEDAN_EN_DISCO_seLeenTest(unittest.TestCase):
    """R3 comparaba solo lo que el `TrainingRunResult` devuelve, y el camino
    FUNCTION deja `r2` y `mae` en `metrics.json` — las mismas que publica un
    paquete de regresión. Resultado: «ninguna de las métricas comparables la
    reportó el reentrenamiento», con el fichero delante. Es la tercera vez hoy
    que el dato estaba en disco y no lo leía nadie."""

    def _dir(self, contenido: str | None):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        if contenido is not None:
            (d / "metrics.json").write_text(contenido, encoding="utf-8")
        return d

    def test_lee_los_numeros(self):
        from matrixai.export.verify import _metricas_del_taller
        d = self._dir('{"r2": 1.0, "mae": 6.5e-17, "epochs": 25}')
        self.assertEqual(_metricas_del_taller(d),
                         {"r2": 1.0, "mae": 6.5e-17, "epochs": 25})

    def test_lo_que_NO_es_numero_no_entra(self):
        """Meterlo obligaría a quien compara a distinguirlo después."""
        from matrixai.export.verify import _metricas_del_taller
        d = self._dir('{"r2": 1.0, "labels": ["a", "b"], "note": "x"}')
        self.assertEqual(_metricas_del_taller(d), {"r2": 1.0})

    def test_sin_fichero_no_inventa_nada(self):
        from matrixai.export.verify import _metricas_del_taller
        self.assertEqual(_metricas_del_taller(self._dir(None)), {})

    def test_un_fichero_ilegible_no_tumba_la_verificacion(self):
        from matrixai.export.verify import _metricas_del_taller
        self.assertEqual(_metricas_del_taller(self._dir("{roto")), {})
