"""Ola 2 (30-10-2026) — el nivel de un recibo NO promete más que su verificación.

El inventario de la Ola 2 midió (sonda 5) que `nivel_del_recibo` daba **A1** a un
sobre con una firma basura (`keyid: "inventada", sig: "AAAA"`) mientras
`verificar_recibo` la rechazaba: miraba que HUBIERA firmas, no que VALIERAN.

La regla que se prueba aquí:

* con la clave, el nivel es el de `verificar_recibo` (el MISMO criterio) o A0;
* sin la clave «no se pudo comprobar» NO es A1: es A0, y lo que sería si la firma
  valiera se dice aparte (`nivel_si_la_firma_vale`, `assurance_if_signature_holds`);
* la firma buena sigue dando su A1 / A2 (el otro lado: un nivel que dijera A0 a
  todo pasaría las pruebas de arriba).
"""

import base64
import copy
import inspect
import json
import unittest

from matrixai.pipelines import receipt as R
from matrixai.pipelines.receipt import (
    firmar_recibo,
    nivel_del_recibo,
    nivel_si_la_firma_vale,
    sobre_in_toto,
    verificar_recibo,
)
from matrixai.pipelines.verifier import verificar_sobre

from test_c81_recibos import _recibo

_CLAVE = b"0" * 32
_OTRA = b"9" * 32
_EVIDENCIA = {"trace_root": "sha256:" + "4" * 64, "package_sha256": "a" * 64}


def _basura(sobre):
    """El sobre de la sonda 5: la firma es de mentira."""
    malo = copy.deepcopy(sobre)
    malo["signatures"] = [{"keyid": "inventada", "sig": "AAAA"}]
    return malo


def _con_el_payload_tocado(sobre):
    malo = copy.deepcopy(sobre)
    cuerpo = json.loads(base64.b64decode(malo["payload"]))
    cuerpo["receipt_id"] = "otro"
    malo["payload"] = base64.b64encode(json.dumps(cuerpo, sort_keys=True).encode()).decode()
    return malo


class LaFirmaBasuraNoDaNivelTest(unittest.TestCase):
    def setUp(self):
        self.bueno = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.basura = _basura(self.bueno)

    def test_la_sonda_5__verificar_la_rechaza_y_el_nivel_ya_no_dice_A1(self):
        self.assertFalse(verificar_recibo(self.basura, clave=_CLAVE)["ok"])
        self.assertEqual(nivel_del_recibo(self.basura, clave=_CLAVE), "A0")

    def test_ni_sin_clave(self):
        self.assertEqual(nivel_del_recibo(self.basura), "A0")

    def test_firmado_con_OTRA_clave_tampoco(self):
        self.assertEqual(nivel_del_recibo(self.bueno, clave=_OTRA), "A0")

    def test_el_payload_tocado_despues_de_firmar_tampoco(self):
        tocado = _con_el_payload_tocado(self.bueno)
        self.assertFalse(verificar_recibo(tocado, clave=_CLAVE)["ok"])
        self.assertEqual(nivel_del_recibo(tocado, clave=_CLAVE), "A0")

    def test_una_firma_basura_sobre_un_recibo_con_evidencia_no_compra_A2(self):
        sobre = _basura(firmar_recibo(_recibo(evidence=_EVIDENCIA), clave=_CLAVE, key_id="k1"))
        self.assertEqual(nivel_del_recibo(sobre, clave=_CLAVE), "A0")

    def test_el_statement_in_toto_con_firma_basura_tampoco(self):
        sobre = _basura(sobre_in_toto(_recibo(), clave=_CLAVE, key_id="k1"))
        self.assertEqual(nivel_del_recibo(sobre, clave=_CLAVE), "A0")


class LaFirmaBuenaSigueDandoSuNivelTest(unittest.TestCase):
    """El otro lado: arreglar un sesgo puede crear el contrario."""

    def test_A1(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(sobre, clave=_CLAVE), "A1")

    def test_A2_por_la_evidencia(self):
        sobre = firmar_recibo(_recibo(evidence=_EVIDENCIA), clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(sobre, clave=_CLAVE), "A2")

    def test_el_statement_in_toto_bueno_da_lo_mismo_que_el_nativo(self):
        r = _recibo(evidence=_EVIDENCIA)
        self.assertEqual(nivel_del_recibo(sobre_in_toto(r, clave=_CLAVE, key_id="k1"), clave=_CLAVE),
                         nivel_del_recibo(firmar_recibo(r, clave=_CLAVE, key_id="k1"), clave=_CLAVE))

    def test_un_recibo_sin_firmar_sigue_siendo_A0_con_o_sin_clave(self):
        self.assertEqual(nivel_del_recibo(_recibo()), "A0")
        self.assertEqual(nivel_del_recibo(_recibo(), clave=_CLAVE), "A0")


class SinClaveNoSePudoComprobarNoEsA1Test(unittest.TestCase):
    def setUp(self):
        self.sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")

    def test_una_firma_buena_sin_clave_es_A0(self):
        self.assertEqual(nivel_del_recibo(self.sobre), "A0")

    def test_lo_que_sostendria_si_valiera_se_dice_APARTE_y_con_su_nombre(self):
        self.assertEqual(nivel_si_la_firma_vale(self.sobre), "A1")
        # y con una firma basura, la misma función dice A1: por eso NO es una garantía
        # y el nivel que se enseña a alguien no la usa.
        self.assertEqual(nivel_si_la_firma_vale(_basura(self.sobre)), "A1")

    def test_el_verificador_sin_clave_dice_A0_y_el_A1_condicional(self):
        r = verificar_sobre(self.sobre, clave=None)
        self.assertEqual(r["assurance_level"], "A0")
        self.assertEqual(r["assurance_if_signature_holds"], "A1")
        self.assertFalse(r["signature_checked"])
        self.assertTrue(r["assurance_is_claimed_not_checked"])

    def test_el_verificador_sin_clave_con_firma_basura_no_dice_A1_tampoco(self):
        r = verificar_sobre(_basura(self.sobre), clave=None)
        self.assertEqual(r["assurance_level"], "A0")

    def test_el_verificador_con_la_clave_buena_dice_A1_y_sin_condicional(self):
        r = verificar_sobre(self.sobre, clave=_CLAVE)
        self.assertEqual(r["assurance_level"], "A1")
        self.assertIsNone(r["assurance_if_signature_holds"])

    def test_el_verificador_con_firma_basura_y_clave_dice_A0(self):
        r = verificar_sobre(_basura(self.sobre), clave=_CLAVE)
        self.assertFalse(r["ok"])
        self.assertEqual(r["assurance_level"], "A0")


class ElNivelUsaElCriterioDeVerificarReciboTest(unittest.TestCase):
    """Reutilizar, no copiar: dos sitios decidiendo qué firma vale acabarían divergiendo."""

    def test_el_nivel_llama_a_verificar_recibo(self):
        self.assertIn("verificar_recibo(", inspect.getsource(R.nivel_del_recibo))

    def test_nivel_distinto_de_A0_si_y_solo_si_verificar_recibo_dice_ok(self):
        bueno = firmar_recibo(_recibo(evidence=_EVIDENCIA), clave=_CLAVE, key_id="k1")
        casos = [bueno, _basura(bueno), _con_el_payload_tocado(bueno)]
        for clave in (_CLAVE, _OTRA):
            for sobre in casos:
                with self.subTest(clave=clave, firma=sobre["signatures"][0]["sig"][:6]):
                    ok = verificar_recibo(sobre, clave=clave)["ok"]
                    self.assertEqual(nivel_del_recibo(sobre, clave=clave) != "A0", ok)


if __name__ == "__main__":
    unittest.main()
