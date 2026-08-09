"""CONTRATO 71 — un identificador no es una característica.

Hallazgo del barrido de los 15 modelos: un prompt SIN objetivo construye
un modelo igualmente e inventa ocho columnas, y la primera es
`customer_id`. Reproducido contra el backend real: «analizar los datos de
clientes de una empresa de telefonía» devuelve `customer_id, age, gender,
plan_type, …` y NO avisaba de nada.

Un identificador como entrada deja al modelo memorizar quién es cada fila
en vez de aprender de sus rasgos: acierta en los datos que ha visto y no
sirve para nadie nuevo.

Por el NOMBRE y no por los datos, a propósito: cuando el modelo nace de
un prompt no hay CSV que analizar. `dataset_analysis` sí mira los valores
cuando los hay, y esa vía no se toca.
"""

import unittest

from matrixai.training.dense_generator import (
    DenseNetworkGenerator,
    parece_identificador,
    resolve_prompt_fields,
)


class TestPareceIdentificador(unittest.TestCase):
    def test_los_que_lo_son(self):
        for nombre in ("id", "customer_id", "id_cliente", "uuid", "guid",
                       "dni", "nif", "nie", "ssn", "matricula", "serial",
                       "identificador", "identifier", "Customer_ID"):
            self.assertTrue(parece_identificador(nombre), nombre)

    def test_los_que_NO_lo_son(self):
        """La lista es corta a propósito: cada entrada de más es un aviso
        falso sobre una columna legítima, y un aviso que salta cuando no
        toca enseña a ignorarlos."""
        for nombre in ("edad", "price", "valid", "idea", "rapidez", "orden",
                       "numero_de_hijos", "identidad_visual"):
            self.assertFalse(parece_identificador(nombre), nombre)

    def test_un_codigo_postal_no_es_un_identificador(self):
        """Es una categórica de dominio con significado —el propio core la
        trata como EMBEDDING—, así que avisar sería estorbar."""
        self.assertFalse(parece_identificador("codigo_postal"))
        self.assertFalse(parece_identificador("zip_code"))


class TestAvisoAlResolverCampos(unittest.TestCase):
    def setUp(self):
        self.dg = DenseNetworkGenerator()

    def _avisos(self, campos, locale="es"):
        _, _, _, _, w = resolve_prompt_fields(self.dg, "clasificar clientes", campos, locale)
        return w

    def test_avisa_del_identificador(self):
        avisos = self._avisos(["customer_id", "edad"])
        self.assertTrue(any("customer_id" in a and "identificador" in a for a in avisos), avisos)

    def test_no_avisa_cuando_no_hay_ninguno(self):
        avisos = self._avisos(["edad", "ingresos"])
        self.assertFalse(any("identificador" in a for a in avisos), avisos)

    def test_el_aviso_va_en_el_idioma_pedido(self):
        avisos = self._avisos(["customer_id"], "en")
        self.assertTrue(any("looks like an identifier" in a for a in avisos), avisos)
        self.assertFalse(any("parece un identificador" in a for a in avisos), avisos)

    def test_se_avisa_pero_NO_se_quita(self):
        """Quitar una columna que alguien ha pedido, sin decirlo, es
        decidir por su cuenta sobre sus datos. Y un identificador puede
        llevar información de verdad (un código de producto que agrupa
        familias). Quien lo sabe, decide."""
        campos, _, _, _, _ = resolve_prompt_fields(self.dg, "clasificar clientes", ["customer_id", "edad"])
        self.assertIn("customer_id", campos)

    def test_avisa_de_cada_uno(self):
        avisos = self._avisos(["customer_id", "order_id", "edad"])
        cuantos = sum(1 for a in avisos if "identificador" in a)
        self.assertEqual(cuantos, 2, avisos)


if __name__ == "__main__":
    unittest.main()
