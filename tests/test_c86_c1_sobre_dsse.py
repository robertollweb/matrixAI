"""CONTRATO 86-C1 — el sobre, conforme a DSSE de verdad.

Medido el 2026-08-24 sobre `matrixai/pipelines/receipt.py`: `firmar_recibo`
guardaba `"payload": payload.decode("utf-8")` —el JSON canonicalizado **en
claro**— y la especificación de DSSE manda el payload **en base64**. El PAE
(`_pae`) ya estaba bien: se calcula sobre los BYTES CRUDOS. O sea que la
parte difícil estaba bien y la fácil rompía la compatibilidad.

Lo que este fichero fija, y por qué cada cosa:

1. **El sobre emitido lleva el payload en base64**, y decodifica a los
   mismos bytes que se firmaron. Sin esto no hay conformidad ninguna.
2. **El PAE sigue sobre los bytes crudos.** Es la mitad que ya estaba
   bien, y «arreglar algo puede romper lo de al lado»: si alguien
   «corrigiera» el PAE para que también fuera sobre el base64, el sobre
   dejaría de verificarse en cualquier implementación ajena y las pruebas
   de arriba seguirían verdes.
3. **Una implementación AJENA de DSSE lo acepta.** Escrita aquí desde la
   especificación, sin importar nada de `receipt.py` salvo el tipo: si
   verificara con nuestro código, comprobaría que somos consistentes con
   nosotros mismos, que es justo lo que no hace falta demostrar.
4. **Los sobres del formato anterior se RECHAZAN con su motivo.**
   Decisión de Roberto (2026-08-24): fuera no hay ni un recibo emitido, así
   que no se escribe código de compatibilidad. Y el motivo tiene que decir
   QUÉ pasó y QUÉ hacer —volver a emitirlo—, no «sobre inválido»: un
   mensaje genérico manda a buscar una corrupción que no existe.
5. **Un base64 válido con el contenido manipulado no verifica.** El
   base64 no es una firma; si esto pasara, el cambio habría convertido el
   sobre en un envoltorio decorativo.
"""

import base64
import binascii
import hashlib
import hmac
import json
import unittest

from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.receipt import (
    PAYLOAD_TYPE,
    firmar_recibo,
    nivel_del_recibo,
    payload_del_sobre,
    verificar_recibo,
)
from matrixai.pipelines.verifier import (
    comparar_recibos,
    inspeccionar_recibo,
    verificar_sobre,
)

_CLAVE = b"0" * 32


def _recibo(**cambios):
    """El MISMO recibo completo de §14.2 que usan las pruebas del 81.

    Un fixture a medias describe un recibo que el producto no produce, y
    mientras describa eso ninguna prueba puede ver el hueco (H2 del
    refutador, 2026-08-20).
    """
    base = {
        "schema_version": "1.0",
        "receipt_id": "r-1",
        "event_type": "pipeline_execution",
        "created_at": "2026-08-20T10:00:00Z",
        "subject": {"purpose": "clasificar tickets", "actor": "prueba"},
        "pipeline": {
            "pipeline_id": "clasificar", "pipeline_version": "1.0.0",
            "pipeline_digest": "sha256:" + "1" * 64,
            "executed_path_digest": "sha256:" + "2" * 64,
            "nodes": [{"node_id": "s1", "model_ref": "text_encoder@v1"}],
        },
        "models": [{"model_id": "text_encoder", "version": "v1",
                    "digest": "sha256:" + "3" * 64}],
        "input": {"entry_nodes": ["s1"], "raw_data_included": False},
        "checks": {"policy_results": []},
        "output": {"outcome": "completed", "output_digests": {}},
        "steps": [{"id": "s1", "model": "text_encoder@v1",
                   "entry_hash": "sha256:" + "a" * 64}],
        "evidence": {"trace_root": "sha256:" + "4" * 64},
    }
    base.update(cambios)
    return base


def _sobre_del_formato_anterior(recibo=None):
    """El sobre EXACTO que emitía el código hasta el 2026-08-24.

    Payload en claro y firma buena sobre esos mismos bytes: es lo que
    tendría en la mano quien guardara un recibo de antes del cambio. Se
    reconstruye aquí —y no se guarda un fichero de ejemplo— porque lo que
    se prueba es el RECHAZO, y un sobre viejo con la firma rota probaría
    otra cosa.
    """
    crudo = jcs_bytes(recibo if recibo is not None else _recibo())
    pae = (b"DSSEv1 " + str(len(PAYLOAD_TYPE)).encode() + b" "
           + PAYLOAD_TYPE.encode() + b" "
           + str(len(crudo)).encode() + b" " + crudo)
    firma = base64.b64encode(hmac.new(_CLAVE, pae, hashlib.sha256).digest())
    return {
        "payloadType": PAYLOAD_TYPE,
        "payload": crudo.decode("utf-8"),
        "signatures": [{"keyid": "k1", "sig": firma.decode("ascii")}],
    }


class ElSobreVaEnBASE64Test(unittest.TestCase):
    def test_el_payload_emitido_es_base64_estandar(self):
        """Estricto (`validate=True`): un base64 «casi» no lo lee nadie."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        crudo = base64.b64decode(sobre["payload"], validate=True)
        # Y ANCLA POSITIVA: sobre una cadena vacía el `b64decode` también
        # pasa, y esta prueba quedaría comprobando nada.
        self.assertTrue(sobre["payload"])
        self.assertTrue(crudo.startswith(b"{"))

    def test_y_NO_va_en_claro(self):
        """El defecto medido, dicho al revés: si volviera, esto se cae."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertFalse(sobre["payload"].lstrip().startswith("{"))

    def test_decodifica_a_los_bytes_JCS_que_se_firmaron(self):
        """No basta con que sea base64: tiene que ser el base64 DE LO
        FIRMADO. Un sobre con base64 de otra cosa sería conforme y falso."""
        recibo = _recibo()
        sobre = firmar_recibo(recibo, clave=_CLAVE, key_id="k1")
        self.assertEqual(base64.b64decode(sobre["payload"], validate=True),
                         jcs_bytes(recibo))

    def test_el_RESTO_del_sobre_no_cambia(self):
        """C1 toca el transporte del payload y nada más: la firma sigue
        siendo HMAC y `payloadType`/`keyid`/`sig` siguen donde estaban."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertEqual(sorted(sobre), ["payload", "payloadType", "signatures"])
        self.assertEqual(sobre["payloadType"], PAYLOAD_TYPE)
        self.assertEqual(sobre["signatures"][0]["keyid"], "k1")
        self.assertTrue(sobre["signatures"][0]["sig"])


class ElPAESigueSobreLosBYTESCRUDOSTest(unittest.TestCase):
    """La mitad que YA estaba bien, con una prueba que la sujeta.

    Sin esto, «hacerlo todo base64» —PAE incluido— pasaría las pruebas de
    arriba y dejaría un sobre que ninguna implementación ajena verifica.
    """

    def test_la_firma_cuadra_con_el_PAE_de_los_bytes_DECODIFICADOS(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        crudo = base64.b64decode(sobre["payload"], validate=True)
        pae = (b"DSSEv1 " + str(len(PAYLOAD_TYPE)).encode() + b" "
               + PAYLOAD_TYPE.encode() + b" "
               + str(len(crudo)).encode() + b" " + crudo)
        esperada = base64.b64encode(
            hmac.new(_CLAVE, pae, hashlib.sha256).digest()).decode("ascii")
        self.assertEqual(sobre["signatures"][0]["sig"], esperada)

    def test_se_verifica_sobre_LO_QUE_SALE_DEL_BASE64_no_sobre_un_recompuesto(self):
        """P23-R-0016, dicho en los términos del 86-C1.

        La invariante era «se verifica sobre los bytes que el sobre
        lleva»; ahora esos bytes salen de decodificar el base64, y
        decodificar NO es recomponer. Se prueba con un payload que **no
        es JCS** —`json.dumps`, con sus espacios— firmado sobre sus
        propios bytes: si el verificador recanonicalizara el objeto, esta
        firma buena saldría mala. Con un payload ya canónico la prueba no
        vería nada, porque recanonicalizarlo lo deja igual.
        """
        cuerpo = json.dumps(_recibo(), indent=2).encode("utf-8")
        self.assertNotEqual(cuerpo, jcs_bytes(_recibo()),
                            "si fueran iguales esta prueba no mide nada")
        pae = (b"DSSEv1 " + str(len(PAYLOAD_TYPE)).encode() + b" "
               + PAYLOAD_TYPE.encode() + b" "
               + str(len(cuerpo)).encode() + b" " + cuerpo)
        sobre = {
            "payloadType": PAYLOAD_TYPE,
            "payload": base64.b64encode(cuerpo).decode("ascii"),
            "signatures": [{"keyid": "k1", "sig": base64.b64encode(
                hmac.new(_CLAVE, pae, hashlib.sha256).digest()).decode("ascii")}],
        }
        self.assertTrue(verificar_recibo(sobre, clave=_CLAVE)["ok"])

    def test_y_NO_con_el_PAE_del_base64(self):
        """El otro lado, que es el que se rompería al «arreglar» el PAE."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        b64 = sobre["payload"].encode("ascii")
        pae = (b"DSSEv1 " + str(len(PAYLOAD_TYPE)).encode() + b" "
               + PAYLOAD_TYPE.encode() + b" "
               + str(len(b64)).encode() + b" " + b64)
        equivocada = base64.b64encode(
            hmac.new(_CLAVE, pae, hashlib.sha256).digest()).decode("ascii")
        self.assertNotEqual(sobre["signatures"][0]["sig"], equivocada)


class UnaImplementacionAJENADeDSSELoAceptaTest(unittest.TestCase):
    """Escrita desde la especificación, sin llamar a `receipt.py`.

    Verificar nuestro sobre con nuestro verificador demuestra que somos
    consistentes con nosotros mismos, que es lo único que no hace falta
    demostrar. Este verificador solo conoce DSSE: `payload` en base64,
    PAE `DSSEv1 SP len(type) SP type SP len(body) SP body` sobre los bytes
    decodificados, y `sig` en base64.
    """

    @staticmethod
    def _verificador_ajeno(sobre, clave):
        cuerpo = base64.b64decode(sobre["payload"], validate=True)
        tipo = sobre["payloadType"].encode("utf-8")
        pae = b"DSSEv1 " + b" ".join([
            str(len(sobre["payloadType"])).encode(), tipo,
            str(len(cuerpo)).encode(), cuerpo,
        ])
        esperada = hmac.new(clave, pae, hashlib.sha256).digest()
        for firma in sobre["signatures"]:
            if hmac.compare_digest(base64.b64decode(firma["sig"]), esperada):
                return cuerpo
        raise AssertionError("la firma no cuadra")

    def test_lo_verifica_y_saca_el_recibo(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        cuerpo = self._verificador_ajeno(sobre, _CLAVE)
        self.assertEqual(json.loads(cuerpo)["receipt_id"], "r-1")

    def test_y_ESE_MISMO_verificador_no_podia_con_el_formato_anterior(self):
        """La medida que justifica el corte: el sobre de antes ni siquiera
        llegaba a comprobar la firma — se caía al decodificar."""
        with self.assertRaises((binascii.Error, ValueError)):
            self._verificador_ajeno(_sobre_del_formato_anterior(), _CLAVE)


class ElFormatoANTERIORSeRechazaConSuMOTIVOTest(unittest.TestCase):
    """Decisión de Roberto (2026-08-24): se rechazan, sin compatibilidad.

    Y **diciéndolo**: la invariante 3 del contrato 86 es «nada se retira
    sin decirlo», y un «sobre inválido» genérico sobre un recibo que se
    emitió perfectamente bien hace tres días es media verdad.
    """

    @staticmethod
    def _dice_lo_que_pasa_y_que_hacer(caso, texto):
        texto = (texto or "").lower()
        caso.assertIn("base64", texto, "no dice qué pasó")
        caso.assertIn("formato anterior", texto, "no dice que es el de antes")
        caso.assertIn("volver a emitir", texto, "no dice qué hacer")

    def test_verificar_recibo_lo_rechaza_diciendolo(self):
        r = verificar_recibo(_sobre_del_formato_anterior(), clave=_CLAVE)
        self.assertFalse(r["ok"])
        self._dice_lo_que_pasa_y_que_hacer(self, r["reason"])
        # Y NO lo llama firma mala: la firma es buena, y mandar a mirar la
        # clave sería mandar a buscar la avería donde no está.
        self.assertNotIn("la firma no corresponde", r["reason"])

    def test_el_verificador_del_81_C4_tambien(self):
        r = verificar_sobre(_sobre_del_formato_anterior(), clave=_CLAVE)
        self.assertFalse(r["ok"])
        self._dice_lo_que_pasa_y_que_hacer(self, " ".join(r["problems"]))

    def test_y_no_sale_con_medio_informe(self):
        """Quien encadene esto en un guion no puede encontrarse un informe
        con la mitad de las claves según por dónde haya salido."""
        viejo = verificar_sobre(_sobre_del_formato_anterior(), clave=_CLAVE)
        bueno = verificar_sobre(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"),
                                clave=_CLAVE)
        self.assertEqual(sorted(viejo), sorted(bueno))

    def test_inspect_lo_rechaza_diciendolo(self):
        r = inspeccionar_recibo(_sobre_del_formato_anterior())
        self.assertFalse(r["ok"])
        self._dice_lo_que_pasa_y_que_hacer(self, r["reason"])

    def test_compare_lo_rechaza_diciendolo(self):
        r = comparar_recibos(_sobre_del_formato_anterior(),
                             firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"))
        self.assertFalse(r["identical"])
        self._dice_lo_que_pasa_y_que_hacer(self, " ".join(r["differences"]))

    def test_el_NIVEL_de_un_sobre_ilegible_es_A0(self):
        """Antes caía en el `except` de `json.loads` y salía **A1** —«lleva
        firma»—. Un sobre que este código no lee no se presenta como
        firmado: se presenta como lo que es."""
        self.assertEqual(nivel_del_recibo(_sobre_del_formato_anterior()), "A0")

    def test_el_CLI_lo_rechaza_diciendolo_y_con_codigo_2(self):
        """Por la puerta por la que entra un usuario: `matrixai receipt
        verify`. Probar la función no es probar el producto."""
        import contextlib
        import io
        import sys as _sys
        import tempfile
        from pathlib import Path

        from matrixai.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "viejo.mxreceipt"
            ruta.write_text(json.dumps(_sobre_del_formato_anterior()),
                            encoding="utf-8")
            salida = io.StringIO()
            viejo_argv = _sys.argv
            _sys.argv = ["matrixai", "receipt", "verify", str(ruta),
                         "--key", _CLAVE.hex()]
            try:
                with contextlib.redirect_stdout(salida):
                    codigo = main()
            finally:
                _sys.argv = viejo_argv
        self.assertEqual(codigo, 2)
        self._dice_lo_que_pasa_y_que_hacer(self, salida.getvalue())

    def test_lo_que_NO_es_base64_NI_el_formato_anterior_se_dice_aparte(self):
        """Declarar lo que PASÓ: acusar de «formato anterior» a un sobre
        que trae basura sería inventarse un diagnóstico."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        sobre["payload"] = "no-es-base64!!"
        crudo, motivo = payload_del_sobre(sobre)
        self.assertIsNone(crudo)
        self.assertIn("base64", motivo.lower())
        # NO afirma que sea del formato anterior: lo plantea como
        # condición («si se emitió antes de…»), porque no se sabe.
        self.assertIn("Si el recibo se emitió antes", motivo)
        self.assertNotEqual(
            motivo, payload_del_sobre(_sobre_del_formato_anterior())[1],
            "un payload ilegible y uno del formato anterior no son el "
            "mismo diagnóstico")
        # Pero SÍ apunta a la posibilidad, que es lo útil.
        self.assertIn("volver a emitirlo", motivo)

    def test_un_payload_en_claro_NUNCA_pasa_por_base64_valido(self):
        """La detección es segura por construcción, no por suerte: `{` no
        está en el alfabeto de base64. Se comprueba sobre recibos con
        contenidos distintos, no sobre uno solo."""
        for receipt_id in ("r-1", "r-2", "abcd", "0", "x" * 40):
            with self.subTest(receipt_id=receipt_id):
                sobre = _sobre_del_formato_anterior(_recibo(receipt_id=receipt_id))
                crudo, motivo = payload_del_sobre(sobre)
                self.assertIsNone(crudo)
                self.assertIn("formato anterior", motivo)


class ElBASE64NoEsUnaFIRMATest(unittest.TestCase):
    """Un sobre bien codificado y con el contenido cambiado NO verifica.

    Si esto pasara, el cambio habría convertido el sobre en un envoltorio
    decorativo: conforme a DSSE por fuera y sin garantía por dentro.
    """

    def test_cambiar_el_contenido_y_recodificar_no_cuela(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        crudo = base64.b64decode(sobre["payload"], validate=True)
        self.assertIn(b"clasificar tickets", crudo)
        sobre["payload"] = base64.b64encode(
            crudo.replace(b"clasificar tickets", b"clasificar NOMINAS")).decode("ascii")
        r = verificar_recibo(sobre, clave=_CLAVE)
        self.assertFalse(r["ok"])
        self.assertIn("firma", r["reason"].lower())

    def test_y_el_verificador_del_81_C4_tampoco_lo_aprueba(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        crudo = base64.b64decode(sobre["payload"], validate=True)
        sobre["payload"] = base64.b64encode(
            crudo.replace(b'"outcome":"completed"',
                          b'"outcome":"COMPLETADO"')).decode("ascii")
        r = verificar_sobre(sobre, clave=_CLAVE)
        self.assertFalse(r["ok"])
        self.assertTrue(r["signature_checked"])
        self.assertFalse(r["signature_valid"])

    def test_cambiar_el_payloadType_tampoco(self):
        """Lo que el PAE existe para impedir, con el payload ya en base64:
        una firma buena no debe valer para otro tipo."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        sobre["payloadType"] = "application/vnd.in-toto+json"
        self.assertFalse(verificar_recibo(sobre, clave=_CLAVE)["ok"])

    def test_un_sobre_INTACTO_si_verifica(self):
        """El otro lado: sin esto, un verificador que dijera «no» a todo
        pasaría las tres pruebas de arriba."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertTrue(verificar_recibo(sobre, clave=_CLAVE)["ok"])
        self.assertTrue(verificar_sobre(sobre, clave=_CLAVE)["ok"])
        self.assertEqual(nivel_del_recibo(sobre), "A1")


if __name__ == "__main__":
    unittest.main()
