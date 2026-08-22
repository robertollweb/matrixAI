"""CONTRATO 81-C3 — transferir las salidas de un nodo a las entradas del siguiente.

Lo único que quedaba del runtime. Y es donde se cuelan los fallos que
después nadie explica: un nodo que recibe algo que no es lo que su
predecesor produjo da un resultado plausible y equivocado.

Reglas que este fichero fija:

* **Una entrada que falta NO se rellena.** Ni con `None`, ni con vacío:
  el pipeline para y dice qué nodo esperaba qué.
* **Se comprueba el digest** de lo que se transporta cuando el productor
  lo declaró: si cambió por el camino, el consumidor no debe verlo.
* **Un nodo no lee salidas de nodos que aún no han corrido**, aunque
  estén en el grafo: leerlas daría el valor de una ejecución anterior.
"""

import unittest

from matrixai.pipelines.runtime import (
    EntradaQueFalta,
    SalidaAlterada,
    entradas_para,
)


class LoQueUnNodoRECIBETest(unittest.TestCase):
    def test_toma_la_salida_de_aquel_del_que_depende(self):
        salidas = {"a": {"value": 42, "digest": None}}
        self.assertEqual(entradas_para("b", ["a"], salidas), {"a": 42})

    def test_varias_dependencias_llegan_TODAS(self):
        salidas = {"a": {"value": 1, "digest": None}, "b": {"value": 2, "digest": None}}
        self.assertEqual(entradas_para("c", ["a", "b"], salidas), {"a": 1, "b": 2})

    def test_sin_dependencias_recibe_un_diccionario_vacio(self):
        self.assertEqual(entradas_para("a", [], {}), {})


class LoQueFALTANoSeRellenaTest(unittest.TestCase):
    def test_una_dependencia_que_no_ha_producido_PARA_el_pipeline(self):
        """Rellenar con `None` haría que el nodo corriera con un hueco y
        devolviera algo plausible y equivocado."""
        with self.assertRaises(EntradaQueFalta) as caja:
            entradas_para("b", ["a"], {})
        mensaje = str(caja.exception)
        # Dice QUIÉN esperaba QUÉ: sin eso hay que ir al grafo a deducirlo.
        self.assertIn("b", mensaje)
        self.assertIn("a", mensaje)

    def test_un_nodo_que_corrio_pero_NO_produjo_tambien_para(self):
        """Haber corrido no es haber producido. Un `None` guardado como
        salida es un hueco, y pasarlo adelante lo esconde."""
        with self.assertRaises(EntradaQueFalta):
            entradas_para("b", ["a"], {"a": {"value": None, "digest": None}})


class LoQueCAMBIAPorElCaminoTest(unittest.TestCase):
    def test_si_el_digest_no_cuadra_NO_se_entrega(self):
        salidas = {"a": {"value": "hola", "digest": "sha256:" + "0" * 64}}
        with self.assertRaises(SalidaAlterada):
            entradas_para("b", ["a"], salidas, comprobar_digests=True)

    def test_cuando_cuadra_se_entrega(self):
        import hashlib
        valor = "hola"
        digest = "sha256:" + hashlib.sha256(valor.encode()).hexdigest()
        salidas = {"a": {"value": valor, "digest": digest}}
        self.assertEqual(entradas_para("b", ["a"], salidas, comprobar_digests=True),
                         {"a": "hola"})

    def test_sin_digest_declarado_NO_se_inventa_una_comprobacion(self):
        """Un productor que no declaró digest no es un productor que
        mienta: se entrega y se sabe que eso no se comprobó."""
        salidas = {"a": {"value": "hola", "digest": None}}
        self.assertEqual(entradas_para("b", ["a"], salidas, comprobar_digests=True),
                         {"a": "hola"})


if __name__ == "__main__":
    unittest.main()
