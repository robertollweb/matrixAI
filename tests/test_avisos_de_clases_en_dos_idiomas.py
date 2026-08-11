"""Lo que redacta el CORE se traduce en el core.

Contrato 75, medido el 2026-08-11 preparando capturas para la web: con
la interfaz en inglés y `locale=en`, la pantalla salía a medias en
español porque estos avisos estaban escritos en castellano fijo. El
rationale del LLM sí venía en inglés — el hueco era del core, no del
modelo.

Es la regla del proyecto: **lo que redacta el core se traduce en el
core**, no al pintarlo. Traducirlo en la interfaz significaría tener dos
versiones de lo que el producto dice sobre su propia decisión.

Ojo con lo que ESTA prueba no cubre: el generador COMPUESTO no recibe
idioma todavía, así que sigue con el valor por defecto. No es una
regresión —hace lo mismo que antes— pero tampoco está resuelto, y
callarlo sería peor que anotarlo.
"""

from __future__ import annotations

import re
import unittest

from matrixai.training.dense_generator import DenseNetworkGenerator, resolve_task_and_labels

#: Palabras FUNCIONALES: no se pueden evitar escribiendo en castellano, y
#: por eso detectan español mucho mejor que una lista de términos
#: escogidos a mano.
_ESPANOL = re.compile(r"\b(el|la|los|las|del|con|que|para|una|sin|hay|asi|aqui)\b", re.I)
_INGLES = re.compile(r"\b(the|with|there|and|for|are|from|your)\b", re.I)


class AvisosDeClasesEnDosIdiomas(unittest.TestCase):
    def setUp(self) -> None:
        self.dg = DenseNetworkGenerator()

    def _avisos(self, prompt: str, labels, locale: str) -> list[str]:
        _task, _labels, warnings = resolve_task_and_labels(self.dg, prompt, labels, locale)
        return warnings

    def test_una_sola_clase_avisa_en_ingles_cuando_se_pide_ingles(self) -> None:
        avisos = self._avisos("classify emails", ["spam"], "en")
        self.assertTrue(avisos, "sin aviso no hay nada que traducir: el caso ha dejado de dispararse")
        texto = " ".join(avisos)
        self.assertRegex(texto, _INGLES)
        self.assertNotRegex(texto, _ESPANOL)

    def test_y_en_espanol_cuando_se_pide_espanol(self) -> None:
        avisos = self._avisos("clasificar correos", ["spam"], "es")
        self.assertTrue(avisos)
        self.assertRegex(" ".join(avisos), _ESPANOL)

    def test_el_idioma_por_defecto_sigue_siendo_espanol(self) -> None:
        """Sin pedir idioma, lo de siempre.

        El generador compuesto llama sin idioma, así que este defecto
        es lo que lo mantiene igual que antes.
        """
        avisos = self._avisos("clasificar correos", ["spam"], "es")
        sin_pedir = resolve_task_and_labels(self.dg, "clasificar correos", ["spam"])[2]
        self.assertEqual(avisos, sin_pedir)

    def test_los_dos_idiomas_dicen_lo_MISMO(self) -> None:
        """Traducir no es decir otra cosa.

        Los dos tienen que nombrar el número de clases y la salida
        (`ProbabilityMap`): si una versión se queda sin la instrucción
        de cómo arreglarlo, la mitad que lee ese idioma se queda sin
        saber qué hacer.
        """
        es = " ".join(self._avisos("clasificar correos", ["spam"], "es"))
        en = " ".join(self._avisos("classify emails", ["spam"], "en"))
        for texto in (es, en):
            self.assertIn("ProbabilityMap", texto)
            self.assertIn("spam", texto)


if __name__ == "__main__":
    unittest.main()
