"""CONTRATO 81-C4 — el verificador de recibos, independiente del runtime.

`matrixai receipt inspect|verify|compare`. El contrato le exige, literal:

- validar el esquema · canonicalizar · comprobar firma e identidad de la
  clave · comprobar digests · **mostrar qué garantías se han verificado**
  y **qué elementos NO han podido verificarse** · rechazar campos
  críticos ambiguos · **no presentar A1 como A2** · **no presentar
  trazabilidad como prueba de corrección conceptual**.

Las tres últimas son prohibiciones, y son las que más pruebas llevan
aquí: un verificador que exagera lo que ha comprobado es peor que no
tenerlo, porque alguien decidirá confiando en él.
"""

import json
import unittest

from matrixai.pipelines.receipt import firmar_recibo
from matrixai.pipelines.verifier import inspeccionar_recibo, verificar_sobre, comparar_recibos

_CLAVE = b"0" * 32


def _recibo(**cambios):
    """Un recibo COMPLETO según §14.2, que es lo único que el producto
    emite y lo único que `firmar_recibo` acepta.

    Estaba a medias —cinco campos— y pasaba, porque ni el emisor ni el
    verificador exigían las secciones de §14.2 (H2 del refutador,
    2026-08-20). Un fixture a medias describe un recibo que el producto
    no produce, y mientras describió eso ninguna prueba pudo ver el
    hueco. Cada prueba sigue variando SOLO la parte de la que habla.
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
            # §14.2 bis / P23-R-0015: el camino que DE VERDAD se recorrió.
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


class DiceQUEHaVerificadoYQUENoTest(unittest.TestCase):
    def test_lista_las_garantias_comprobadas(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=_CLAVE)
        self.assertIn("signature", r["verified"])
        self.assertIn("schema", r["verified"])

    def test_y_lista_lo_que_NO_ha_podido_verificar(self):
        """Un verificador que solo dice lo que comprobó deja creer que
        comprobó todo. Lo que no pudo mirar también se enseña."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=_CLAVE)
        # No hay raíces de confianza decididas todavía: la identidad de la
        # clave NO se puede verificar, y se dice.
        self.assertIn("key_identity", r["unverified"])
        self.assertTrue(r["unverified"]["key_identity"])

    def test_sin_clave_NO_finge_haber_comprobado_la_firma(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=None)
        self.assertNotIn("signature", r["verified"])
        self.assertIn("signature", r["unverified"])


class LasTresProhibicionesTest(unittest.TestCase):
    def test_NO_presenta_A1_como_A2(self):
        """Un recibo firmado sin evidencias reproducibles es A1. Decir A2
        prometería una reproducción que nadie puede hacer."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=_CLAVE)
        self.assertEqual(r["assurance_level"], "A1")
        self.assertNotEqual(r["assurance_level"], "A2")

    def test_ignora_el_nivel_que_el_recibo_se_ATRIBUYE(self):
        sobre = firmar_recibo(_recibo(assurance_level="A4"), clave=_CLAVE, key_id="k1")
        self.assertEqual(verificar_sobre(sobre, clave=_CLAVE)["assurance_level"], "A1")

    def test_NO_presenta_trazabilidad_como_correccion(self):
        """Lo dice el contrato con esas palabras. Que un pipeline esté
        trazado no significa que su resultado sea correcto, y confundirlo
        haría que un recibo verde avalara una decisión equivocada."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=_CLAVE)
        aviso = " ".join(r["disclaimers"]).lower()
        self.assertIn("no prueba", aviso)
        self.assertTrue(any("correc" in d.lower() for d in r["disclaimers"]))


class CamposCRITICOSAmbiguosTest(unittest.TestCase):
    def test_dos_pasos_con_el_MISMO_id_se_rechazan(self):
        """Si dos pasos comparten id, «el paso s1» no identifica a
        ninguno: cualquier afirmación sobre él es ambigua."""
        malo = _recibo(steps=[{"id": "s1", "model": "a@v1"},
                              {"id": "s1", "model": "b@v1"}])
        sobre = firmar_recibo(malo, clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=_CLAVE)
        self.assertFalse(r["ok"])
        self.assertIn("s1", json.dumps(r))

    def test_un_digest_que_NO_es_un_digest_se_rechaza(self):
        malo = _recibo(steps=[{"id": "s1", "model": "a@v1", "entry_hash": "sha256:cafe"}])
        sobre = firmar_recibo(malo, clave=_CLAVE, key_id="k1")
        self.assertFalse(verificar_sobre(sobre, clave=_CLAVE)["ok"])


class InspeccionarYCompararTest(unittest.TestCase):
    def test_inspeccionar_NO_verifica_y_lo_dice(self):
        """`inspect` enseña el contenido; no comprueba nada. Si no lo
        dijera, alguien leería un `inspect` limpio como un `verify` bueno."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = inspeccionar_recibo(sobre)
        self.assertEqual(r["receipt_id"], "r-1")
        self.assertTrue(r["note"])
        self.assertIn("no verifica", r["note"].lower())

    def test_comparar_dice_EN_QUE_se_diferencian(self):
        uno = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        otro = firmar_recibo(_recibo(receipt_id="r-2"), clave=_CLAVE, key_id="k1")
        r = comparar_recibos(uno, otro)
        self.assertFalse(r["identical"])
        self.assertIn("receipt_id", json.dumps(r["differences"]))

    def test_comparar_dos_iguales_lo_dice(self):
        uno = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertTrue(comparar_recibos(uno, dict(uno))["identical"])


if __name__ == "__main__":
    unittest.main()


class ElNivelNoSeLeeComoUnAPROBADOTest(unittest.TestCase):
    """Sin clave, `assurance: A1` significa «el recibo lleva firma», no
    «la firma es buena». Sin decirlo, se lee como un aprobado."""

    def test_sin_clave_avisa_de_que_el_nivel_es_DECLARADO(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=None)
        self.assertTrue(r["assurance_is_claimed_not_checked"])

    def test_con_la_firma_comprobada_ya_no_avisa(self):
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=_CLAVE)
        self.assertFalse(r["assurance_is_claimed_not_checked"])


class ElVERIFICADORNoAprueba_LoQueNoHaMiradoTest(unittest.TestCase):
    """H2 y H6 del refutador (2026-08-20).

    Aquí se prueba el lado de la LECTURA, que es el que importa: un
    recibo forjado a mano no pasa por `firmar_recibo`, así que lo que lo
    detenga tiene que estar en el verificador. *Validar en la escritura no
    valida la lectura.*
    """

    @staticmethod
    def _forjado(recibo):
        """Un sobre DSSE con firma buena, sin pasar por el emisor."""
        import base64
        import hashlib
        import hmac
        import json

        from matrixai.pipelines.receipt import PAYLOAD_TYPE

        payload = json.dumps(recibo)
        pae = (b"DSSEv1 " + str(len(PAYLOAD_TYPE)).encode() + b" "
               + PAYLOAD_TYPE.encode() + b" "
               + str(len(payload.encode())).encode() + b" " + payload.encode())
        firma = hmac.new(_CLAVE, pae, hashlib.sha256).digest()
        return {"payloadType": PAYLOAD_TYPE, "payload": payload,
                "signatures": [{"keyid": "k1", "sig": base64.b64encode(firma).decode()}]}

    def test_un_recibo_sin_las_secciones_de_14_2_NO_sale_verificado(self):
        sobre = self._forjado({"schema_version": "1.0", "receipt_id": "m",
                               "steps": [{"id": "n0"}],
                               "evidence": {"package_sha256": True}})
        r = verificar_sobre(sobre, clave=_CLAVE)
        self.assertFalse(r["ok"])
        self.assertNotIn("schema", r["verified"])
        # La firma SÍ es buena: lo que falla es el contenido, y se
        # distinguen. Decir «firma mala» aquí mandaría a buscar la avería
        # donde no está.
        self.assertIn("signature", r["verified"])
        self.assertIn("schema", r["unverified"])

    def test_y_NOMBRA_lo_que_falta(self):
        sobre = self._forjado({"schema_version": "1.0", "receipt_id": "m",
                               "steps": [{"id": "n0"}], "evidence": {}})
        problemas = " ".join(verificar_sobre(sobre, clave=_CLAVE)["problems"])
        for seccion in ("event_type", "created_at", "subject", "output"):
            self.assertIn(seccion, problemas)

    def test_una_firma_COMPROBADA_Y_RECHAZADA_no_se_llama_no_comprobada(self):
        """H6: `"signature" not in verificado` metía en el mismo saco «no
        había clave» y «la firma es mala», y el CLI imprimía «la firma no
        se ha comprobado» sobre una firma que sí se comprobó."""
        sobre = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        r = verificar_sobre(sobre, clave=b"9" * 32)
        self.assertTrue(r["signature_checked"])
        self.assertFalse(r["signature_valid"])
        self.assertFalse(r["assurance_is_claimed_not_checked"])

    def test_sin_clave_SI_es_no_comprobada(self):
        r = verificar_sobre(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"), clave=None)
        self.assertFalse(r["signature_checked"])
        self.assertTrue(r["assurance_is_claimed_not_checked"])

    def test_sin_clave_NO_esta_COMPROBADO_del_todo(self):
        """Y de ahí sale el `rc=3`: quien encadene `receipt verify &&
        desplegar` estaba tratando un recibo no comprobado como
        comprobado, que es el bloqueante que el 82 registró para
        `verify` y aquí seguía sin arreglar."""
        r = verificar_sobre(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"), clave=None)
        self.assertFalse(r["fully_checked"])
        self.assertIn("signature", r["unchecked"])

    def test_con_la_clave_y_todo_bien_SI_esta_comprobado_del_todo(self):
        """El otro lado: si `fully_checked` no pudiera ser cierto nunca,
        el código `0` no se alcanzaría y no distinguiría nada.

        `key_identity` NO cuenta: está sin decidir a propósito (§0.1) y es
        una limitación permanente, no un «esta vez no se pudo».
        """
        r = verificar_sobre(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"), clave=_CLAVE)
        self.assertTrue(r["fully_checked"])
        self.assertEqual(r["unchecked"], [])
        self.assertEqual(r["standing_limitation"], "key_identity")

    def test_inspect_enseña_la_fecha_QUE_EL_EMISOR_ESCRIBE(self):
        """Leía `produced_at`, un nombre que no escribe nadie: la fecha
        salía en blanco en todos los recibos del producto. Lo tapaba el
        fixture, que describía un recibo que no existe."""
        r = inspeccionar_recibo(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"))
        self.assertEqual(r["created_at"], "2026-08-20T10:00:00Z")


class LosTRESCodigosDeSalidaTest(unittest.TestCase):
    """H6 del refutador (2026-08-20): `receipt verify` devolvía `0 if ok
    else 2`, así que un recibo verificado SIN CLAVE —firma sin comprobar,
    identidad sin comprobar— salía con el mismo `0` que uno íntegro y
    firmado.

    El 82 registró exactamente ese bloqueante para `matrixai verify`
    («quien encadene `verify && desplegar` trataba un paquete no
    verificado como verificado») y aquí seguía sin arreglar, un módulo más
    allá. Se usan LOS MISMOS códigos, importados de allí: dos escalas
    distintas para lo mismo acabarían divergiendo.
    """

    @staticmethod
    def _rc(informe):
        from matrixai.cli import _salida_de_receipt

        return _salida_de_receipt(informe)

    def _informe(self, clave):
        return verificar_sobre(firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1"),
                               clave=clave)

    def test_todo_comprobado_y_bien_es_0(self):
        self.assertEqual(self._rc(self._informe(_CLAVE)), 0)

    def test_algo_FALLA_es_2(self):
        self.assertEqual(self._rc(self._informe(b"9" * 32)), 2)

    def test_NO_SE_PUDO_COMPROBAR_es_3_y_no_0(self):
        """Ni un aprobado ni una acusación."""
        self.assertEqual(self._rc(self._informe(None)), 3)

    def test_los_tres_son_DISTINTOS(self):
        """Sin esto, la prueba de arriba la pasaría un código que siempre
        devuelve lo mismo."""
        codigos = {self._rc(self._informe(_CLAVE)),
                   self._rc(self._informe(b"9" * 32)),
                   self._rc(self._informe(None))}
        self.assertEqual(len(codigos), 3)

    def test_son_LOS_MISMOS_que_los_de_matrixai_verify(self):
        from matrixai.export.verify import SALIDAS

        self.assertEqual(
            (SALIDAS["ok"], SALIDAS["fail"], SALIDAS["incomparable"]), (0, 2, 3))
