"""83-C4 — la serie, alimentada con un `DriftReport` DE VERDAD.

Probar la función no es probar el producto. Estas pruebas no fabrican el
informe: lo produce `DriftDetector` de P22 a partir de una política real,
y por ahí salieron **dos defectos que las pruebas con fixture no podían
ver** (medidos el 2026-08-20):

1. **`serie_de_deriva` leía `results` como una LISTA** y el core lo
   produce como un **diccionario** feature → resultado. Con datos de
   verdad la serie salía **vacía**: y un histórico vacío se lee como «no
   hay deriva», que es justo lo que esa función existe para no decir.
2. **`DriftReport` no tiene `to_dict`**, así que el `dict(informe)` de
   respaldo del backend levantaba `TypeError: not iterable`. El endpoint
   habría contestado 500 la primera vez que un modelo tuviera política de
   verdad. No se veía porque las pruebas nunca pasaban de la rama «este
   modelo no declara política».

Las dos son la misma lección: un fixture describe la forma que alguien
supuso, no la que el core produce.
"""

import dataclasses
import random
import unittest
from pathlib import Path

from matrixai.continual.drift import DriftDetector, DriftReport
from matrixai.continual.parser import parse_mxcontinual
from matrixai.continual.supervision import serie_de_deriva

_FUENTE = Path(__file__).resolve().parent / "test_p22_cut3_drift_detector.py"


def _politica_real():
    texto = _FUENTE.read_text(encoding="utf-8")
    ini = texto.index('_POLICY_SRC = """') + len('_POLICY_SRC = """')
    return parse_mxcontinual(texto[ini:texto.index('"""', ini)])


def _informe_real(*, desplazado: bool):
    """Un `DriftReport` del detector de P22, no uno escrito a mano."""
    random.seed(7)
    features = ("severity", "hour_of_day", "sender_domain", "score", "size")
    referencia = {f: [random.gauss(0.0, 1.0) for _ in range(200)] for f in features}
    media = 3.0 if desplazado else 0.0
    produccion = {f: [random.gauss(media, 1.0) for _ in range(200)] for f in features}
    return DriftDetector(_politica_real()).run_check(referencia, produccion)


class ElInformeDelCoreSeSERIALIZATest(unittest.TestCase):
    def test_un_DriftReport_no_se_puede_convertir_con_dict(self):
        """La suposición que rompía el endpoint, fijada para que se vea."""
        informe = _informe_real(desplazado=True)
        self.assertIsInstance(informe, DriftReport)
        self.assertFalse(hasattr(informe, "to_dict"))
        with self.assertRaises(TypeError):
            dict(informe)

    def test_con_asdict_SI_y_results_es_un_DICCIONARIO(self):
        datos = dataclasses.asdict(_informe_real(desplazado=True))
        self.assertIsInstance(datos["results"], dict)
        self.assertIn("severity", datos["results"])


class LaSerieLEELoQueElCoreESCRIBETest(unittest.TestCase):
    def _serie(self, *, desplazado):
        return serie_de_deriva([dataclasses.asdict(_informe_real(desplazado=desplazado))])

    def test_una_medicion_de_verdad_NO_sale_como_una_serie_vacia(self):
        serie = self._serie(desplazado=True)
        self.assertFalse(serie["never_measured"])
        self.assertEqual(len(serie["points"]), 1)

    def test_las_cinco_features_de_la_politica_aparecen(self):
        serie = self._serie(desplazado=True)
        self.assertEqual(
            sorted(serie["features"]),
            sorted(["severity", "hour_of_day", "sender_domain", "score", "size"]))

    def test_el_veredicto_es_EL_DEL_CORE_no_uno_deducido_aqui(self):
        informe = _informe_real(desplazado=True)
        serie = serie_de_deriva([dataclasses.asdict(informe)])
        self.assertEqual(serie["points"][0]["drift_detected"], informe.drift_detected)
        for nombre, resultado in informe.results.items():
            self.assertEqual(serie["points"][0]["features"][nombre]["drift_detected"],
                             resultado.drift_detected)

    def test_sin_desplazamiento_tambien_se_lee_y_no_es_lo_mismo_que_no_medir(self):
        serie = self._serie(desplazado=False)
        self.assertFalse(serie["never_measured"])
        self.assertEqual(len(serie["points"]), 1)


if __name__ == "__main__":
    unittest.main()
