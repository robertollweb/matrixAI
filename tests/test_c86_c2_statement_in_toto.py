"""86-C2 — EL MISMO RECIBO, EN EL SOBRE QUE LEE EL RESTO DEL MUNDO.

El nativo (`application/vnd.matrixai.receipt+json`) es correcto y no se retira,
pero **no lo entiende ninguna herramienta de las que ya tiene un comprador**.
Aquí el mismo recibo viaja además como un **Statement de in-toto**, que es lo
que leen `cosign`, los motores de políticas y los ingestores de atestaciones.

Lo que NO cambia, y va escrito en el módulo: la firma sigue siendo HMAC y sigue
demostrando **consistencia, no autenticidad**. Un sobre estándar con firma
simétrica es *legible* ahí fuera, no *confiable* ahí fuera.

El detalle que hace que un formato «casi estándar» no lo sea: en in-toto el
algoritmo es la CLAVE del mapa de digests (`{"sha256": "abc…"}`), así que el
prefijo `sha256:` que usa el recibo **se quita**. Dejarlo dentro del valor
produce un digest que ningún verificador reconoce.
"""
from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.pipelines.receipt import (  # noqa: E402
    PAYLOAD_TYPE,
    PAYLOAD_TYPE_IN_TOTO,
    PREDICATE_TYPE,
    ReciboInvalido,
    firmar_recibo,
    nivel_del_recibo,
    recibo_del_sobre,
    sobre_in_toto,
    sujetos_del_recibo,
    verificar_recibo,
)
from test_c81_recibos import _recibo  # noqa: E402

_CLAVE = b"clave-de-pruebas"


def _statement(sobre) -> dict:
    return json.loads(base64.b64decode(sobre["payload"]))


class ElSobreEsUnStatementDeVerdadTest(unittest.TestCase):
    def test_lleva_el_tipo_y_el_payload_type_estandar(self):
        s = sobre_in_toto(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertEqual(s["payloadType"], PAYLOAD_TYPE_IN_TOTO)
        self.assertEqual(_statement(s)["_type"], "https://in-toto.io/Statement/v1")

    def test_el_predicado_es_NUESTRO_y_va_versionado_por_URL(self):
        """Un `predicateType` en un dominio ajeno es una promesa que no
        podemos sostener."""
        d = _statement(sobre_in_toto(_recibo(), clave=_CLAVE, key_id="k1"))
        self.assertEqual(d["predicateType"], PREDICATE_TYPE)
        self.assertTrue(PREDICATE_TYPE.startswith("https://matrixaistudio.org/"))

    def test_el_recibo_entero_viaja_como_predicado(self):
        d = _statement(sobre_in_toto(_recibo(), clave=_CLAVE, key_id="k1"))
        self.assertEqual(d["predicate"]["receipt_id"], "r-1")
        self.assertIn("evidence", d["predicate"])

    def test_los_digests_van_SIN_el_prefijo_del_algoritmo(self):
        """En in-toto el algoritmo es la clave del mapa. Con el prefijo dentro
        del valor, ningún verificador reconoce el digest."""
        for sujeto in sujetos_del_recibo(_recibo()):
            valor = sujeto["digest"]["sha256"]
            self.assertNotIn(":", valor)
            self.assertEqual(len(valor), 64)

    def test_los_sujetos_son_el_pipeline_y_sus_modelos(self):
        nombres = [s["name"] for s in sujetos_del_recibo(_recibo())]
        self.assertIn("pipeline:clasificar@1.0.0", nombres)
        self.assertIn("text_encoder@v1", nombres)

    def test_un_recibo_que_no_cumple_14_2_NO_se_envuelve(self):
        """Sacarlo en un sobre estándar lo haría parecer más serio, no serlo."""
        with self.assertRaises(ReciboInvalido):
            sobre_in_toto({"schema_version": "1.0", "receipt_id": "x"},
                          clave=_CLAVE, key_id="k1")


class LosDosSobresValenLoMismoTest(unittest.TestCase):
    def test_el_de_in_toto_verifica_igual_que_el_nativo(self):
        r = _recibo()
        self.assertTrue(verificar_recibo(firmar_recibo(r, clave=_CLAVE, key_id="k1"),
                                         clave=_CLAVE)["ok"])
        self.assertTrue(verificar_recibo(sobre_in_toto(r, clave=_CLAVE, key_id="k1"),
                                         clave=_CLAVE)["ok"])

    def test_una_clave_que_no_es_no_verifica_ninguno(self):
        r = _recibo()
        self.assertFalse(verificar_recibo(sobre_in_toto(r, clave=_CLAVE, key_id="k1"),
                                          clave=b"otra")["ok"])

    def test_el_PAE_ata_el_tipo_al_contenido(self):
        """La firma de un sobre no vale en el otro aunque el recibo sea el
        mismo: si valiera, una firma buena se podría reutilizar cambiando el
        tipo del sobre."""
        r = _recibo()
        nativo = firmar_recibo(r, clave=_CLAVE, key_id="k1")
        intoto = sobre_in_toto(r, clave=_CLAVE, key_id="k1")
        self.assertNotEqual(nativo["signatures"][0]["sig"], intoto["signatures"][0]["sig"])
        # Y cambiarle el tipo a un sobre bueno lo invalida.
        falsificado = dict(nativo, payloadType=PAYLOAD_TYPE_IN_TOTO)
        self.assertFalse(verificar_recibo(falsificado, clave=_CLAVE)["ok"])

    def test_el_NIVEL_es_el_mismo_por_los_dos_caminos(self):
        """La evidencia vive en el `predicate`. Sin leerla ahí, un A2 en sobre
        estándar se leería como A1 — y el sobre nuevo valdría menos que el
        viejo por un detalle de forma."""
        r = _recibo(evidence={"trace_root": "sha256:" + "4" * 64,
                              "package_sha256": "a" * 64})
        self.assertEqual(nivel_del_recibo(firmar_recibo(r, clave=_CLAVE, key_id="k1")), "A2")
        self.assertEqual(nivel_del_recibo(sobre_in_toto(r, clave=_CLAVE, key_id="k1")), "A2")


class LoQueNoSeInterpretaAMediasTest(unittest.TestCase):
    def test_un_Statement_de_OTRO_predicado_no_se_lee_como_nuestro(self):
        s = sobre_in_toto(_recibo(), clave=_CLAVE, key_id="k1")
        d = _statement(s)
        d["predicateType"] = "https://slsa.dev/provenance/v1"
        s["payload"] = base64.b64encode(json.dumps(d).encode()).decode()
        recibo, motivo = recibo_del_sobre(s)
        self.assertIsNone(recibo)
        self.assertIn("no es el de un recibo de MatrixAI", motivo)

    def test_un_sobre_nativo_sigue_dando_su_recibo(self):
        recibo, motivo = recibo_del_sobre(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"))
        self.assertIsNone(motivo)
        self.assertEqual(recibo["receipt_id"], "r-1")

    def test_y_el_tipo_nativo_no_ha_cambiado(self):
        """El de siempre no se retira: quien ya lo lea sigue leyéndolo."""
        self.assertEqual(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")["payloadType"],
                         PAYLOAD_TYPE)


if __name__ == "__main__":
    unittest.main()
