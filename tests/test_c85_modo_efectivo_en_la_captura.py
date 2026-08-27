"""EL MODO DE GENERACIÓN, EFECTIVO Y EN LA CAPTURA (hallazgo del 2026-08-24).

Medido conduciendo el producto antes de tocar nada: un paquete exportado por el
Studio salía con `generation.mode: null`, y entonces

    manifest PASS · R1 INCOMPARABLE · training NOT_RUN · R3 NOT_RUN

porque el verificador no puede regenerar sin saber con qué modo se generó. Y
como `training` es INCOMPARABLE si R1 no pasó, y R3 depende de training, **las
tres etapas que demuestran que el paquete se rehace eran inalcanzables**: de un
`matrixai verify` sobre un paquete del producto solo se podía comprobar la
integridad del manifiesto.

El campo existía en el core desde el 82 —`_RUN_PROVENANCE_GENERATION_KEYS` lista
`mode` como el PRIMER parámetro efectivo que la captura declara— y la captura
nunca lo traía. Otra vez el cableado.

Y hay una trampa que esta prueba fija, porque la primera versión del arreglo la
tenía: **el modo que viaja tiene que ser el EFECTIVO, no el pedido**. Un
«coherente» sin reglas de dominio degrada a aleatorio (opción A), así que
declarar el pedido haría que el paquete dijera que se generó de una forma en la
que no se generó — y R1 regeneraría otra cosa.
"""
from __future__ import annotations

import time
import unittest

from matrixai.playground import _submit_training_job
from matrixai.playground_api import generate_synthetic_dataset

_MXAI = (
    "PROJECT Riesgo\n\nVECTOR Entrada[2]\n  edad: Scalar\n  ingresos: Scalar\nEND\n\n"
    "NETWORK Clasificador\n  INPUT Entrada\n"
    "  LAYER Dense units=8 activation=relu\n"
    "  LAYER Dense units=2 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
    "GRAPH\n  Entrada -> Clasificador\nEND\n"
)
_MXTRAIN = (
    "MODEL riesgo.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT Entrada FROM COLUMNS [edad, ingresos]\n"
    "  TARGET predicted_class: Label[alto, bajo]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION Clasificador\n  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE Clasificador.*\nEND\n\n"
    "RUN\n  EPOCHS 3\nEND\n"
)
_RECETA = "alto: edad > 0.6\nDEFAULT: bajo"


class ElGeneradorPublicaElModoEfectivoTest(unittest.TestCase):
    def test_con_receta_el_efectivo_es_el_pedido(self):
        r = generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 7, "coherent", False,
                                       recipe_text=_RECETA)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["mode"], "coherent")
        self.assertEqual(r["effective_mode"], "coherent")
        self.assertEqual(r["label_origin"], "synthetic_domain")

    def test_SIN_receta_el_pedido_dice_coherente_y_el_efectivo_dice_ALEATORIO(self):
        """La trampa, y por eso esta prueba es la importante del fichero.

        `mode` sigue diciendo «coherent» —es lo que se pidió, y hay pantallas
        que lo enseñan—, pero las etiquetas son aleatorias. Lo que puede viajar
        a un manifiesto es lo segundo.
        """
        r = generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 7, "coherent", False)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["mode"], "coherent")
        self.assertEqual(r["effective_mode"], "random")
        self.assertEqual(r["label_origin"], "synthetic_random")

    def test_aleatorio_es_aleatorio_por_los_dos_lados(self):
        r = generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 7, "random", False)
        self.assertEqual((r["mode"], r["effective_mode"]), ("random", "random"))


class LaCapturaLlevaElModoTest(unittest.TestCase):
    def _captura(self, **extra):
        datos = generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 7, "coherent", False,
                                           recipe_text=_RECETA)
        r = _submit_training_job(_MXAI, _MXTRAIN, datos["csv_text"], 2,
                                 recipe_text=_RECETA, dataset_seed=7, **extra)
        self.assertTrue(r.get("ok"), r)
        from matrixai.playground import _training_jobs
        # Se ESPERA a que termine: solo hay un entrenamiento a la vez, y la
        # segunda prueba del fichero se encontraba «Ya hay un entrenamiento en
        # curso» —un fallo de la prueba, no del producto—.
        for _ in range(600):
            if _training_jobs[r["job_id"]]["status"] != "running":
                break
            time.sleep(0.05)
        return _training_jobs[r["job_id"]]["run_provenance"]

    def test_el_modo_declarado_viaja_a_la_captura_con_el_nombre_QUE_EL_CORE_LEE(self):
        """`mode`, y no otro nombre.

        `export/reproduce.py` publica como parámetro efectivo toda clave de la
        captura que no sea identidad ni digesto, **con su propio nombre**. Un
        `dataset_mode` habría salido en el manifiesto como `dataset_mode` y
        `generation.mode` habría seguido siendo `null` — medido antes de
        corregirlo, y es justo el fallo que esta prueba impide que vuelva.
        """
        captura = self._captura(dataset_mode="coherent")
        self.assertEqual(captura["mode"], "coherent")

    def test_sin_declararlo_la_captura_dice_que_NO_CONSTA(self):
        """`None`, no un valor por defecto: el entrenador no generó los datos y
        no puede saberlo. «No consta» no es «aleatorio»."""
        captura = self._captura()
        self.assertIsNone(captura["mode"])


if __name__ == "__main__":
    unittest.main()
