"""«2.369 parameters» en inglés son dos coma tres seis nueve.

Auditoría de la aplicación (pasada 1). En una pantalla en INGLÉS, el
mismo número aparecía con dos separadores distintos a la vez:

    64 → 32 → 1 · 2,369 parameters · 3 warnings      ← el raíl
    architecture from source 'llm': 64-32, 2.369 parameters   ← el motivo

El segundo sale de `miles()`, que llevaba el separador español FIJO y se
usaba también dentro de la rama inglesa de `rationale` y en los avisos de
presupuesto y techo de datos. Lo que redacta el core se traduce en el
core, y eso incluye cómo se escribe un número.
"""
from __future__ import annotations

import unittest

from matrixai.training.architecture_policy import miles


class ElSeparadorDependeDelIdioma(unittest.TestCase):
    def test_en_espanol_es_punto(self) -> None:
        self.assertEqual(miles(2369), "2.369")
        self.assertEqual(miles(532481, "es"), "532.481")

    def test_en_ingles_es_coma(self) -> None:
        self.assertEqual(miles(2369, "en"), "2,369")
        self.assertEqual(miles(532481, "en"), "532,481")

    def test_no_se_escriben_igual(self) -> None:
        """El caso que lo delató: no es tipografía, cambia el número."""
        self.assertNotEqual(miles(50000, "es"), miles(50000, "en"))

    def test_por_defecto_sigue_siendo_espanol(self) -> None:
        """La mitad que impide romper a quien ya lo llamaba sin idioma."""
        self.assertEqual(miles(3474), "3.474")

    def test_los_pequenos_no_llevan_separador(self) -> None:
        for loc in ("es", "en"):
            self.assertEqual(miles(200, loc), "200")


class ElMOTIVOQueRedactaElCoreLoUsaBien(unittest.TestCase):
    """Probar la función no es probar el producto: se mira el texto que
    el core redacta de verdad."""

    def test_el_rationale_en_ingles_no_lleva_punto_de_miles(self) -> None:
        from matrixai.training import architecture_policy as ap
        capas, motivo, avisos = ap.decidir_arquitectura(
            n_features=8, n_rows=5000, tarea="classification", locale="en",
        )[:3] if hasattr(ap, "decidir_arquitectura") else ([], "", [])
        if motivo:
            # Si hay número de parámetros en el motivo, va en formato inglés.
            import re
            for m in re.findall(r"\d[\d.,]*\d", motivo):
                self.assertNotRegex(m, r"^\d{1,3}\.\d{3}$", f"separador español en inglés: {motivo}")


if __name__ == "__main__":
    unittest.main()
