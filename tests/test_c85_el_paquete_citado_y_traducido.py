"""LO QUE EL PAQUETE DICE DE SÍ MISMO: citado, y dicho otra vez en castellano.

DECISIÓN DE ROBERTO (2026-08-26), la salida intermedia de las tres que había
sobre la mesa:

* **dejarlo en inglés** era coherente con el 85-C2b —lo que el paquete escribió
  se cita, no se traduce, porque esos bytes los cubre su `manifest_sha256`— pero
  deja un párrafo largo en otro idioma en mitad de la pantalla;
* **traducirlo** arregla eso y le pone al paquete en la boca algo que no dijo;
* **las dos, diciendo cuál es cuál**, es la única que no miente en ninguna
  dirección. Cuesta más, y es la que se hizo.

Y LA PARTE FINA: esto **no traduce la frase, vuelve a componer el hecho**. El
manifiesto declara `missing: ["recipe", "seed_dataset"]` —una lista de claves,
no prosa— y de ahí salen tanto el inglés del paquete como el castellano de
`reproduce_textos`. Traducir la cadena sería adivinar; componer desde las mismas
claves es decir lo mismo otra vez.
"""
from __future__ import annotations

import unittest

from matrixai.export.reproduce import _MISSING_REASONS
from matrixai.export.reproduce_textos import (
    CLAVES,
    claves_sin_traducir,
    motivo_no_reproducible,
    motivos_que_faltan,
)


class LasDosListasDicenLO_MISMOTest(unittest.TestCase):

    def test_ni_una_clave_sin_pareja(self):
        """Dos diccionarios uno al lado del otro invitan a editar uno y olvidar
        el otro. Esta prueba es la que hace que eso se note."""
        self.assertEqual(set(_MISSING_REASONS), set(CLAVES),
                         "hay claves que solo saben decirse en un idioma")

    def test_y_ninguna_se_queda_en_blanco(self):
        for clave, texto in CLAVES.items():
            with self.subTest(clave=clave):
                self.assertTrue(texto.strip(), f"{clave} no dice nada")
                # Un barrido de palabras FUNCIONALES: no se pueden evitar
                # escribiendo en castellano, y son lo que distingue una frase
                # traducida de una copiada del inglés.
                self.assertTrue(
                    any(f" {p} " in f" {texto} " for p in
                        ("el", "la", "de", "que", "con", "no", "se", "los", "un")),
                    f"{clave} no parece estar en castellano: {texto[:60]}")


class NO_SE_INVENTA_LoQueNoSabeDecirTest(unittest.TestCase):

    def test_una_clave_desconocida_se_ENUMERA(self):
        self.assertEqual(claves_sin_traducir(["recipe", "algo_nuevo"]), ["algo_nuevo"])

    def test_y_no_entra_en_la_frase(self):
        partes = motivos_que_faltan(["recipe", "algo_nuevo"])
        self.assertEqual(len(partes), 1)
        self.assertNotIn("algo_nuevo", " ".join(partes))

    def test_si_no_sabe_decir_NINGUNA_devuelve_None(self):
        """`None` y no una frase vacía: quien lo pinte tiene que distinguir «no
        hay nada que traducir» de «hay algo y no lo sé decir»."""
        self.assertIsNone(motivo_no_reproducible(["algo_nuevo", "otro"]))

    def test_con_las_que_sabe_compone_la_frase_entera(self):
        frase = motivo_no_reproducible(["recipe", "seed_dataset"])
        self.assertIsNotNone(frase)
        self.assertTrue(frase.startswith("No es reproducible: "))
        self.assertTrue(frase.endswith("."))
        self.assertIn("receta", frase)
        self.assertIn("semilla", frase)

    def test_respeta_el_ORDEN_en_que_llegan(self):
        una = motivo_no_reproducible(["recipe", "seed_dataset"])
        otra = motivo_no_reproducible(["seed_dataset", "recipe"])
        self.assertNotEqual(una, otra)


class NO_ES_UNA_TRADUCCION_DE_LA_FRASETest(unittest.TestCase):
    """Se compone desde `missing`, no desde el `reason` en inglés.

    Es lo que permite decir que el paquete no escribió esto: no se ha tocado ni
    leído su prosa.
    """

    def test_la_funcion_no_recibe_la_frase_inglesa(self):
        import inspect

        firma = inspect.signature(motivo_no_reproducible)
        self.assertEqual(list(firma.parameters), ["faltan"],
                         "si recibiera el `reason` sería una traducción, y no lo es")
