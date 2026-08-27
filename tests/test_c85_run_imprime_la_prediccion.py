"""`matrixai run` IMPRIME LA PREDICCIÓN (medido el 2026-08-24).

Contra el paquete **publicado** 1.6.0: `run` sobre el ejemplo Kelvin del propio
repositorio —entrenado a `val loss 0.000013`— imprimía el proyecto, las
acciones y la narrativa de auditoría, y **ni un número**. La predicción solo
salía con `--json`. Está en el código, no era una impresión: `_print_run_report`
tocaba `project`, `actions` y `audit`, y nunca `state`.

O sea que **el comando que predice no enseñaba la predicción** para cualquier
modelo sin acciones discretas — todo clasificador o regresor puro, que es lo
que hace el 99 % de quien llega.

Y una segunda decisión, que es la misma que la 1.4.2 ya tomó para el paquete:
**un clasificador nombra sus clases**. Un `[2.6e-38, 1.0]` obliga a ir al
`.mxai` a ver qué posición es qué clase, y ahí es donde uno se equivoca.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from matrixai.cli import _print_run_report, _salidas_declaradas
from matrixai.parser import parse_text

_REGRESION = (
    "PROJECT K\n\nVECTOR Reading[1]\n  celsius: Scalar\nEND\n\n"
    "PARAM W1 Vector[1]\nEND\n\nPARAM b1 Scalar\nEND\n\n"
    "FUNCTION KelvinPrediction\n"
    "  predicted_kelvin: Scalar = linear(W1 * Reading + b1)\nEND\n\n"
    "GRAPH\n  Reading -> KelvinPrediction\nEND\n"
)
_CLASIFICADOR = (
    "PROJECT C\n\nVECTOR Entrada[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
    "NETWORK Clasificador\n  INPUT Entrada\n"
    "  LAYER Dense units=4 activation=relu\n"
    "  LAYER Dense units=2 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
    "GRAPH\n  Entrada -> Clasificador\nEND\n"
)


def _salida(program, state, actions=()):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_run_report(getattr(program, "project", "P"),
                          {"state": state, "actions": list(actions), "audit": "—"},
                          program)
    return buf.getvalue()


class LaPrediccionSeImprimeTest(unittest.TestCase):
    def test_una_regresion_ENSEÑA_SU_NUMERO(self):
        program = parse_text(_REGRESION)
        salida = _salida(program, {"celsius": 25.0, "Reading": [25.0],
                                   "KelvinPrediction": 298.147, "predicted_kelvin": 298.147})
        self.assertIn("predicted_kelvin: 298.147", salida)

    def test_un_clasificador_NOMBRA_SUS_CLASES(self):
        program = parse_text(_CLASIFICADOR)
        salida = _salida(program, {"predicted_class": [0.0, 1.0]})
        self.assertIn("alto=0.0000", salida)
        self.assertIn("bajo=1.0000", salida)
        # Y no el vector crudo, que es lo que obligaba a ir al `.mxai`.
        self.assertNotIn("[0.0, 1.0]", salida)

    def test_NO_se_vuelca_el_estado_entero(self):
        """El estado trae también las entradas y los nodos intermedios.
        Volcarlo convertiría una predicción en un listado."""
        program = parse_text(_REGRESION)
        salida = _salida(program, {"celsius": 25.0, "Reading": [25.0],
                                   "KelvinPrediction": 298.147, "predicted_kelvin": 298.147})
        self.assertNotIn("celsius:", salida)
        self.assertNotIn("Reading:", salida)

    def test_sin_programa_no_revienta_y_no_inventa_nada(self):
        """`_print_run_report` se llama desde más de un sitio: sin el programa
        delante se comporta como antes en vez de fallar."""
        salida = _salida(None, {"predicted_kelvin": 1.0})
        self.assertIn("Project:", salida)
        self.assertNotIn("predicted_kelvin", salida)

    def test_una_etiqueta_de_mas_no_se_empareja_a_ciegas(self):
        """Si las clases declaradas y los valores no cuadran, se enseña el
        valor tal cual: emparejar a ciegas pondría un nombre encima del número
        equivocado, que es peor que no ponerlo."""
        program = parse_text(_CLASIFICADOR)
        salida = _salida(program, {"predicted_class": [0.1, 0.2, 0.7]})
        self.assertIn("predicted_class: [0.1, 0.2, 0.7]", salida)


class LasSalidasSeLeenDelIRTest(unittest.TestCase):
    def test_de_una_funcion_y_de_una_red(self):
        self.assertEqual(_salidas_declaradas(parse_text(_REGRESION)),
                         [("predicted_kelvin", [])])
        self.assertEqual(_salidas_declaradas(parse_text(_CLASIFICADOR)),
                         [("predicted_class", ["alto", "bajo"])])


if __name__ == "__main__":
    unittest.main()
