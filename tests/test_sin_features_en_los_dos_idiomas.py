# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""«Ninguna columna es utilizable» también se dice en inglés.

POR QUÉ IMPORTA MÁS QUE OTROS MENSAJES: estos tres son de los pocos de
este módulo que ve alguien de FUERA. Es lo que recibe quien sube su
primer CSV y no sale nada — el momento exacto en el que decide si el
producto le sirve. Roberto puso como P0 «que una persona externa pueda
llegar de cero a un modelo»; un rechazo en un idioma que no lee es
fricción justo ahí.

Medido el 2026-08-14: con `locale='en'` los tres salían en español.

CÓMO SE DETECTA EL ESPAÑOL: por palabras **funcionales** (el, la, de,
con, que…), no por una lista de términos escogida a mano. Las
funcionales no se pueden evitar escribiendo en castellano, y eso es lo
que las hace servir para un barrido — una lista de palabras elegidas la
esquiva cualquier frase nueva.
"""
from __future__ import annotations

import unittest

from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
)

# OJO CON «no»: es funcional en español Y en inglés («No column can be
# used…»), así que como detector de español da un falso positivo. La
# primera versión de esta lista la incluía y ponía en rojo un mensaje
# inglés perfecto — un aserto que falla puede estar mal EL ASERTO. Las
# que quedan no existen en inglés.
_FUNCIONALES_ES = (" el ", " la ", " las ", " los ", " de ", " del ", " con ",
                   " que ", " para ", " una ", " son ", " todo ")


def _csv_solo_texto(n: int = 60) -> str:
    """Un CSV cuya única entrada es texto libre: hoy no hay modelo posible."""
    buenas, malas = ("excellent", "great"), ("horrible", "awful")
    filas = ["comment,rating"]
    for i in range(n):
        palabra = buenas[i % 2] if i % 2 == 0 else malas[i % 2]
        filas.append(f"the service was really {palabra} on visit {i},"
                     f"{'good' if i % 2 == 0 else 'bad'}")
    return "\n".join(filas) + "\n"


def _csv_todo_constante(n: int = 40) -> str:
    """Dos columnas con un único valor: no hay nada que aprender de ellas."""
    filas = ["a,b,y"] + [f"1,2,{i % 2}" for i in range(n)]
    return "\n".join(filas) + "\n"


def _error(csv: str, target: str, locale: str) -> str:
    try:
        generate_project_from_dataset(csv, target, locale=locale)
    except DatasetProjectError as exc:
        return str(exc)
    raise AssertionError("se esperaba un DatasetProjectError y no lo hubo")


class LosRechazosSinFeaturesHablanElIdiomaPedido(unittest.TestCase):

    def test_los_dos_csv_de_prueba_provocan_de_verdad_su_rechazo(self):
        """UN FIXTURE DESCRIBE ALGO QUE EXISTE.

        Si estos CSV dejaran de disparar el rechazo —porque el detector
        de texto cambia, o porque las constantes se aceptan—, las
        pruebas de abajo pasarían sin comprobar ningún idioma: no hay
        español porque no hay frase. `_error` ya falla si no salta la
        excepción, y esto lo deja dicho.
        """
        self.assertIn("comment", _error(_csv_solo_texto(), "rating", "es"))
        self.assertIn("'a'", _error(_csv_todo_constante(), "y", "es"))

    def test_en_ingles_no_se_cuela_una_palabra_espanola(self):
        for etiqueta, csv, target in (("texto", _csv_solo_texto(), "rating"),
                                      ("constantes", _csv_todo_constante(), "y")):
            with self.subTest(caso=etiqueta):
                mensaje = f" {_error(csv, target, 'en').lower()} "
                coladas = [p.strip() for p in _FUNCIONALES_ES if p in mensaje]
                self.assertEqual(coladas, [], msg=f"quedan palabras españolas: {coladas}")

    def test_en_ingles_sigue_diciendo_LO_MISMO(self):
        """Traducir no puede perder lo que hacía útil el mensaje.

        El de texto vale por tres cosas: nombra la columna, dice que el
        camino desde datos no hace modelos de texto, y **dice por dónde
        sí** (`«comment: Text»`). Un mensaje traducido que se deje la
        salida es media verdad tranquilizadora en otro idioma.
        """
        mensaje = _error(_csv_solo_texto(), "rating", "en")
        self.assertIn("comment", mensaje)
        self.assertIn("TEXT", mensaje)
        self.assertIn("comment: Text", mensaje)

    def test_sin_pedir_idioma_sigue_saliendo_EXACTAMENTE_lo_de_antes(self):
        """Arreglar un sesgo puede crear el contrario.

        El español es el de por defecto y lo usa casi todo el mundo hoy:
        si `locale` se colara mal —o alguien pusiera 'en' de base— esto
        se pondría rojo antes de llegar a nadie.
        """
        por_defecto = _error(_csv_solo_texto(), "rating", "es")
        self.assertIn("Ninguna columna es utilizable como feature", por_defecto)
        self.assertIn("comment: Text", por_defecto)

    def test_un_idioma_que_no_existe_cae_al_espanol_y_no_revienta(self):
        mensaje = _error(_csv_solo_texto(), "rating", "pt")
        self.assertIn("Ninguna columna es utilizable", mensaje)


if __name__ == "__main__":
    unittest.main()
