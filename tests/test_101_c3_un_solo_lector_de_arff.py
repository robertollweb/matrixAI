# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3 — de ARFF se lee por UN sitio, y esta prueba lo impide olvidar.

POR QUÉ EXISTE, y es una historia de tres copias.

`benchmarks/fase0/` llegó a tener **tres** implementaciones de «leer un ARFF»:
la del lector (`lector_arff.py`), la de la pasada exploratoria, y la de la
calibración. Nadie las copió con mala intención: cada una nació porque hacía
falta leer un fichero y ya había un ejemplo al lado.

Y divergieron, que es lo que pasa siempre. Lo que cada copia se dejó:

  · el marcador de ausencia del estándar ARFF es `"?"`, no la cadena vacía, y
    dos de las tres lo dejaban llegar al núcleo **como una categoría más** —
    ni imputado ni con indicador, y eso vale para los cuatro motores;
  · el objetivo **no es el último atributo** en cinco de los cuarenta datasets
    del protocolo, y dos copias usaban `nombres[-1]`: habrían entrenado contra
    otra columna;
  · la columna identificadora que el catálogo dice que sobra —3.178 valores
    distintos para 3.190 filas en uno de ellos— entraba como predictora, que
    es una fuga;
  · y las normalizaciones sin las cuales cuatro de los cuarenta no se pueden
    ni leer.

Ninguna de esas cuatro cosas se arregla «en el lector» si el llamante tiene su
propia copia. Por eso lo que se prueba aquí no es que el lector funcione —eso
ya está probado— sino que **no haya nadie leyendo por su cuenta**.

Es una prueba estática y lo declara: mira el texto de los módulos. No puede
cazar una copia escrita de otra manera, y ese límite está dicho en vez de
sugerido. Lo que sí caza es el caso real, que es alguien llamando a
`scipy.io.arff` directamente porque le venía de paso.
"""
from __future__ import annotations

import unittest
from pathlib import Path

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"

#: El ÚNICO módulo con permiso para abrir un ARFF. Si algún día hay otro, se
#: añade aquí **a propósito** y el commit explica por qué dos lectores.
_LECTOR_UNICO = "lector_arff.py"


def _modulos_de_fase0() -> list[Path]:
    return sorted(p for p in _FASE0.glob("*.py") if p.name != "__init__.py")


class UnSoloLectorDeArffTest(unittest.TestCase):

    def test_solo_el_lector_unico_abre_un_ARFF(self):
        """`scipy.io.arff` se llama desde UN sitio. Cualquier otro módulo que
        lo llame está leyendo por su cuenta, y lo que se dejará por el camino
        no se sabrá hasta que muerda."""
        culpables = []
        for modulo in _modulos_de_fase0():
            if modulo.name == _LECTOR_UNICO:
                continue
            texto = modulo.read_text(encoding="utf-8")
            # Se buscan las dos formas de llegar: el import y la llamada.
            sin_comentarios = "\n".join(
                l for l in texto.splitlines() if not l.lstrip().startswith("#"))
            if "arff.loadarff" in sin_comentarios or "from scipy.io import arff" in sin_comentarios:
                culpables.append(modulo.name)
        self.assertEqual(
            culpables, [],
            f"{culpables} abre(n) un ARFF por su cuenta en vez de usar "
            f"`{_LECTOR_UNICO}`. Tres copias ya divergieron una vez: se "
            "dejaron el `\"?\"` como faltante, el objetivo declarado, la "
            "exclusión del identificador y las normalizaciones")

    def test_el_lector_unico_SI_lo_abre(self):
        """La otra mitad, sin la cual lo de arriba lo pasaría un directorio
        donde nadie lee ARFF ninguno — por ejemplo si el lector se renombra y
        la constante se queda atrás."""
        texto = (_FASE0 / _LECTOR_UNICO).read_text(encoding="utf-8")
        self.assertIn("arff", texto,
                      f"{_LECTOR_UNICO} ya no abre ARFF: o se renombró el "
                      "lector, o esta prueba está vigilando un fichero que ya "
                      "no hace lo que dice su nombre")

    def test_quien_necesita_leer_lo_pide_al_UNICO_sitio_que_lo_hace(self):
        """Y el caso concreto que costó las tres copias: la calibración tenía
        la suya con los tres defectos a la vez, y ahora importa la de la
        pasada. Si alguien la vuelve a escribir en local, esto cae."""
        texto = (_FASE0 / "calibracion_101_c3.py").read_text(encoding="utf-8")
        self.assertIn("from pasada_exploratoria_101_c3 import cargar_arff", texto,
                      "la calibración ha dejado de delegar: si vuelve a leer "
                      "por su cuenta, vuelve a medir tiempos entrenando "
                      "contra la columna equivocada en cinco datasets")
        self.assertNotIn("def cargar_arff", texto,
                         "la calibración ha vuelto a definir su propio "
                         "`cargar_arff`: esa es la cuarta copia")


if __name__ == "__main__":
    unittest.main()
