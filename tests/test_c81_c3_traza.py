"""CONTRATO 81-C3 — la traza unificada de una ejecución.

Es el corazón del corte: lo que el runtime registra es lo que después
sostiene el recibo. Una traza incompleta produce un recibo que afirma más
de lo que vio.

Lo que el contrato exige y aquí se mide:

- identificador **raíz** de ejecución, y cada nodo **relacionado** con él;
- inicio y final de cada nodo, con el modelo y el runtime usados;
- las **políticas evaluadas** y el resultado de las validaciones;
- digests de entradas y salidas cuando corresponda;
- **impedir que un nodo no autorizado ejecute un efecto externo**;
- **`dry-run`**: simular sin efectos;
- y la regla de 13.2 bis: **un reintento no repite un efecto no
  idempotente por su cuenta**.
"""

import unittest

from matrixai.pipelines.trace import (
    EfectoNoAutorizado,
    ReintentoNoSeguro,
    TrazaDeEjecucion,
)


class LaRaizATaTodoTest(unittest.TestCase):
    def test_cada_nodo_referencia_la_raiz(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="text_encoder@v1")
        t.cerrar_nodo("s1", ok=True)
        nodo = t.a_dict()["nodes"][0]
        self.assertEqual(nodo["root_id"], "run-1")

    def test_un_nodo_sin_cerrar_se_VE(self):
        """Un nodo abierto y no cerrado es una ejecución que se cortó. Si
        la traza lo diera por terminado, el recibo diría que fue bien."""
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1")
        d = t.a_dict()
        self.assertFalse(d["complete"])
        self.assertEqual(d["nodes"][0]["status"], "open")

    def test_registra_inicio_final_modelo_y_runtime(self):
        t = TrazaDeEjecucion(root_id="run-1", runtime="matrixai/1.5.0")
        t.abrir_nodo("s1", modelo="m@v1")
        t.cerrar_nodo("s1", ok=True)
        nodo = t.a_dict()["nodes"][0]
        for campo in ("started_at", "ended_at", "model", "runtime"):
            self.assertTrue(nodo[campo], f"falta {campo}")


class LoQueSEREGISTRADeCadaNodoTest(unittest.TestCase):
    def test_las_politicas_evaluadas_viajan_con_su_regla(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1")
        t.anotar_politica("s1", {"decision": "allow", "rule_id": "R3",
                                 "explain": "p@1.0 → allow (R3)"})
        t.cerrar_nodo("s1", ok=True)
        politicas = t.a_dict()["nodes"][0]["policies"]
        self.assertEqual(politicas[0]["rule_id"], "R3")

    def test_los_digests_de_entrada_y_salida_se_anotan(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1", input_digest="sha256:" + "a" * 64)
        t.cerrar_nodo("s1", ok=True, output_digest="sha256:" + "b" * 64)
        nodo = t.a_dict()["nodes"][0]
        self.assertTrue(nodo["input_digest"].startswith("sha256:"))
        self.assertTrue(nodo["output_digest"].startswith("sha256:"))

    def test_un_error_se_propaga_DECLARADO(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1")
        t.cerrar_nodo("s1", ok=False, error="el modelo no respondió")
        nodo = t.a_dict()["nodes"][0]
        self.assertEqual(nodo["status"], "failed")
        self.assertIn("no respondió", nodo["error"])


class LosEfectosEXTERNOSSeControlanTest(unittest.TestCase):
    def test_un_nodo_NO_autorizado_no_puede_ejecutar_un_efecto(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1")  # sin `efectos=True`
        with self.assertRaises(EfectoNoAutorizado):
            t.registrar_efecto("s1", "envió un correo")

    def test_uno_autorizado_SI_y_queda_registrado(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1", efectos=True)
        t.registrar_efecto("s1", "envió un correo")
        self.assertEqual(t.a_dict()["nodes"][0]["effects"], ["envió un correo"])

    def test_en_DRY_RUN_no_se_ejecuta_ningun_efecto(self):
        """Simular con efectos reales no es simular."""
        t = TrazaDeEjecucion(root_id="run-1", dry_run=True)
        t.abrir_nodo("s1", modelo="m@v1", efectos=True)
        with self.assertRaises(EfectoNoAutorizado):
            t.registrar_efecto("s1", "envió un correo")
        self.assertTrue(t.a_dict()["dry_run"])


class ReintentosYEfectosNoIdempotentesTest(unittest.TestCase):
    """13.2 bis: el runtime NO puede garantizar unilateralmente que un
    reintento no duplique un efecto no idempotente. Así que no lo finge:
    se niega, y quien sepa que su operación es segura lo declara."""

    def test_reintentar_algo_NO_idempotente_se_NIEGA(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1", efectos=True, idempotente=False)
        t.cerrar_nodo("s1", ok=False, error="timeout")
        with self.assertRaises(ReintentoNoSeguro):
            t.reintentar("s1")

    def test_reintentar_algo_IDEMPOTENTE_se_permite_y_se_cuenta(self):
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1", idempotente=True)
        t.cerrar_nodo("s1", ok=False, error="timeout")
        t.reintentar("s1")
        self.assertEqual(t.a_dict()["nodes"][0]["attempts"], 2)

    def test_por_DEFECTO_no_es_idempotente(self):
        """El lado prudente: dar por idempotente lo que no se ha declarado
        duplicaría cobros, correos y escrituras."""
        t = TrazaDeEjecucion(root_id="run-1")
        t.abrir_nodo("s1", modelo="m@v1")
        t.cerrar_nodo("s1", ok=False, error="x")
        with self.assertRaises(ReintentoNoSeguro):
            t.reintentar("s1")


if __name__ == "__main__":
    unittest.main()
