"""UNA DIVERGENCIA SE CUENTA, NO SE VUELCA (medido el 2026-08-24).

Contra el paquete **publicado** 1.6.0, con el Kelvin del propio repositorio y el
`.mxtrain` que escribe `matrixai generate-training`:

    OverflowError: (34, 'Numerical result out of range')
      ... y su traza de pila, dentro del cálculo de métricas

Es lo primero que le pasa a quien prueba una regresión desde el CLI, y es el
peor mensaje posible: no dice qué pasó, ni en qué época, ni qué hacer. El
PR1-C4 de este proyecto dice que cada error frecuente nombra su corrección.

Y hay una lección dentro: **no basta con mirar si los pesos son finitos.**
Cuando reventó de verdad los pesos seguían siendo números —enormes, pero
finitos— y lo que desbordaba era el CUADRADO del error. La guarda comprueba lo
que de verdad se va a calcular.
"""
from __future__ import annotations

import math
import unittest

from matrixai.errors import error_training_diverged
from matrixai.training.trainer import _comprobar_que_no_diverge


class _Ejemplo:
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label


class LaGuardaCazaLaDivergenciaTest(unittest.TestCase):
    def test_unos_pesos_sanos_no_levantan_nada(self):
        ejemplos = [_Ejemplo([1.0], 2.0), _Ejemplo([2.0], 4.0)]
        _comprobar_que_no_diverge([2.0], 0.0, 3, ejemplos)   # no levanta

    def test_pesos_INFINITOS(self):
        with self.assertRaises(ValueError) as e:
            _comprobar_que_no_diverge([math.inf], 0.0, 5, [_Ejemplo([1.0], 1.0)])
        self.assertIn("diverged at epoch 5", str(e.exception))

    def test_PESOS_FINITOS_pero_el_cuadrado_del_error_desborda(self):
        """El caso REAL, y el que la primera versión de la guarda no cazaba."""
        ejemplos = [_Ejemplo([1e200], 1.0)]
        with self.assertRaises(ValueError) as e:
            _comprobar_que_no_diverge([1e200], 0.0, 7, ejemplos)
        self.assertIn("diverged at epoch 7", str(e.exception))

    def test_el_mensaje_dice_QUE_HACER_y_no_promete_que_funcione(self):
        texto = error_training_diverged(7, "173.15 to 373.15")
        self.assertIn("LEARNING_RATE", texto)          # la corrección concreta
        self.assertIn("Scalar[0, 1000]", texto)        # la segunda
        self.assertIn("173.15 to 373.15", texto)       # la escala medida
        self.assertIn("None of the three is guaranteed", texto)
        # Y no manda a leer la traza de pila, que es lo que salía antes.
        self.assertNotIn("Traceback", texto)

    def test_sin_escala_medible_el_mensaje_no_se_la_inventa(self):
        texto = error_training_diverged(2)
        self.assertNotIn("Your targets run from", texto)
        self.assertIn("diverged at epoch 2", texto)

    def test_una_etiqueta_que_no_es_un_numero_no_tumba_la_guarda(self):
        """La guarda no puede convertir un dato raro en un fallo de
        entrenamiento: si no puede mirar, se calla."""
        _comprobar_que_no_diverge([1.0], 0.0, 1, [_Ejemplo([1.0], "alto")])


if __name__ == "__main__":
    unittest.main()


class LaGuardaESTA_CABLEADA_alEntrenamientoTest(unittest.TestCase):
    """LA AUDITORÍA DEL CICLO LA CAZÓ (2026-08-25): al revertir la llamada a
    `_comprobar_que_no_diverge` **la suite se quedaba verde**.

    Las pruebas de arriba cubren la FUNCIÓN y ninguna cubría que el entrenador
    la LLAME — que es el hueco que este proyecto lleva quince veces apuntado: el
    core sabe hacerlo y el llamante no lo usa. Un arreglo que se puede quitar
    sin que nada se ponga rojo no está cubierto.
    """

    def test_entrenar_algo_que_diverge_levanta_el_error_ACCIONABLE(self):
        import tempfile
        from pathlib import Path as _Path

        from matrixai.training.parser import parse_training_text
        from matrixai.training.trainer import SupervisedTrainer

        mxai = (
            "PROJECT D\n\nVECTOR E[1]\n  x: Scalar\nEND\n\n"
            "PARAM W1 Vector[1]\nEND\n\nPARAM b1 Scalar\nEND\n\n"
            "FUNCTION M\n  y: Scalar = linear(W1 * E + b1)\nEND\n\n"
            "GRAPH\n  E -> M\nEND\n")
        # Sin rango declarado en ninguna parte: no se normaliza, y con una tasa
        # enorme sobre valores grandes esto revienta — que es justo el caso que
        # el mensaje accionable vino a explicar.
        mxtrain = (
            "MODEL d.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
            "  INPUT E FROM COLUMNS [x]\n  TARGET y: Scalar\n"
            "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=4\nEND\n\n"
            "LOSS L\n  TYPE mse\n  PREDICTION y\n  TARGET y\nEND\n\n"
            "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 5.0\n  UPDATE W1, b1\nEND\n\n"
            "RUN\n  EPOCHS 50\nEND\n")
        with tempfile.TemporaryDirectory() as tmp:
            d = _Path(tmp)
            (d / "d.mxai").write_text(mxai, encoding="utf-8")
            (d / "d.mxtrain").write_text(mxtrain, encoding="utf-8")
            filas = ["x,y"] + [f"{i * 100},{i * 100 + 273.15}" for i in range(40)]
            (d / "d.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
            spec = parse_training_text(mxtrain)
            with self.assertRaises(ValueError) as e:
                SupervisedTrainer().train(spec, output_dir=str(d / "run"), base_path=d,
                                          training_path=d / "d.mxtrain")
        # Y lo que se levanta es EL MENSAJE, no un OverflowError pelado.
        self.assertIn("diverged at epoch", str(e.exception))
        self.assertIn("LEARNING_RATE", str(e.exception))
