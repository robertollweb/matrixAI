"""CONTRATO 81 — el recibo `.mxreceipt` y sus niveles A0-A4.

Decisión de Roberto (2026-08-20): **DSSE sobre bytes JSON canonicalizados
con JCS**. El contrato exige A1 y la base de A2; A3 y A4 quedan
reservados.

| Nivel | Qué afirma |
|---|---|
| **A0** | hay recibo estructurado, **sin firmar** — solo desarrollo |
| **A1** | firmado: integridad, procedencia y relación con artefactos |
| **A2** | además, vinculado a evidencias reproducibles |
| **A3/A4** | reservados: atestación del entorno y prueba criptográfica |

> **P23-R-0016.** La firma se verifica sobre los **bytes originales**. El
> verificador NO parsea y vuelve a serializar antes de comprobarla.

Y la mitad que Roberto dejó abierta a propósito —**quién es de fiar**:
raíces, rotación, revocación— no está: por eso un recibo firmado aquí
demuestra **consistencia**, no autenticidad, y el nivel lo dice.
"""

import unittest

from matrixai.pipelines.receipt import (
    NIVELES,
    ReciboInvalido,
    firmar_recibo,
    nivel_del_recibo,
    verificar_recibo,
)

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


class ElNIVELDiceLoQueElReciboSOSTIENETest(unittest.TestCase):
    def test_sin_firma_es_A0_y_se_dice(self):
        """A0 es «solo desarrollo». Callarlo dejaría que un recibo sin
        firmar pasara por uno firmado en cualquier informe."""
        self.assertEqual(nivel_del_recibo(_recibo()), "A0")

    def test_firmado_es_A1(self):
        firmado = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A1")

    def test_con_evidencias_reproducibles_es_A2(self):
        firmado = firmar_recibo(
            _recibo(evidence={"reproduce": {"package_sha256": "b" * 64}}),
            clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A2")

    def test_una_RAIZ_DE_TRAZA_no_basta_para_A2(self):
        """Encontrado al emitir recibos desde el motor (81-C3): TODO
        recibo llevaría `evidence.trace_root` —apuntarse a sí mismo— y
        con la regla anterior eso solo ya lo declaraba A2. A2 pide
        evidencia que PUEDA REPRODUCIRSE; una traza dice qué pasó, no
        permite volver a hacerlo."""
        firmado = firmar_recibo(
            _recibo(evidence={"trace_root": "run-1", "runtime": "matrixai"}),
            clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A1")

    def test_lo_que_NO_se_reconoce_como_reproducible_tampoco_sube_el_nivel(self):
        """Fallo cerrado: un campo inventado dentro de `evidence` no
        convierte un recibo en reproducible por el hecho de estar ahí."""
        firmado = firmar_recibo(
            _recibo(evidence={"parece_una_evidencia": {"algo": 1}}),
            clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A1")

    def test_A3_y_A4_estan_RESERVADOS_y_no_se_alcanzan(self):
        """Reservado significa que nadie puede declararlos todavía. Si un
        recibo pudiera decir A4 sin prueba criptográfica, el nivel dejaría
        de significar nada."""
        firmado = firmar_recibo(_recibo(assurance_level="A4"), clave=_CLAVE, key_id="k1")
        self.assertIn(nivel_del_recibo(firmado), ("A1", "A2"))
        self.assertEqual(NIVELES[-2:], ("A3", "A4"))


class LaFirmaSeVerificaSobreLosBYTESTest(unittest.TestCase):
    def test_un_recibo_intacto_verifica(self):
        firmado = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertTrue(verificar_recibo(firmado, clave=_CLAVE)["ok"])

    def test_cambiar_UNA_letra_lo_invalida(self):
        firmado = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        firmado["payload"] = firmado["payload"].replace("clasificar", "otra_cosa")
        r = verificar_recibo(firmado, clave=_CLAVE)
        self.assertFalse(r["ok"])
        self.assertIn("firma", r["reason"].lower())

    def test_otra_clave_no_lo_valida(self):
        firmado = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertFalse(verificar_recibo(firmado, clave=b"1" * 32)["ok"])

    def test_se_verifica_sobre_el_PAYLOAD_guardado_no_sobre_uno_recompuesto(self):
        """P23-R-0016. Si el verificador reserializara, dos serializadores
        distintos invalidarían una firma buena — o peor, validarían una
        mala."""
        import inspect
        from matrixai.pipelines import receipt
        fuente = inspect.getsource(receipt.verificar_recibo)
        # ANCLA POSITIVA (auditoría 2ª pasada): sobre una fuente vacía el
        # `assertNotIn` pasa, y la prueba quedaría comprobando nada.
        self.assertIn("def verificar_recibo", fuente)
        self.assertNotIn("jcs_bytes(", fuente,
                         "el verificador NO debe recanonicalizar el payload")


class UnReciboMalFormADONoSeFirmATest(unittest.TestCase):
    def test_sin_esquema_no_se_firma(self):
        with self.assertRaises(ReciboInvalido):
            firmar_recibo({"receipt_id": "r"}, clave=_CLAVE, key_id="k1")

    def test_sin_pasos_no_hay_nada_que_atestiguar(self):
        with self.assertRaises(ReciboInvalido):
            firmar_recibo(_recibo(steps=[]), clave=_CLAVE, key_id="k1")

    def test_la_firma_dice_QUE_CLAVE_la_hizo(self):
        """Sin `key_id`, verificar exige adivinar con cuál se firmó."""
        firmado = firmar_recibo(_recibo(), clave=_CLAVE, key_id="k1")
        self.assertEqual(firmado["signatures"][0]["keyid"], "k1")


if __name__ == "__main__":
    unittest.main()


class LoQueElREFUTADOREncontroTest(unittest.TestCase):
    """H2 (2026-08-20): el emisor firmaba —y el verificador aprobaba— un
    recibo con diez de las doce secciones de §14.2 ausentes, y le daba A2
    porque `evidence.package_sha256` era el booleano `True`.

    Es la avería que este contrato dice haber aprendido —«validar en la
    escritura no valida la lectura»— con el agravante de que **tampoco
    validaba la escritura**.
    """

    def test_un_recibo_al_que_le_falta_medio_14_2_NO_se_firma(self):
        minimo = {"schema_version": "1.0", "receipt_id": "minimo-001",
                  "steps": [{"id": "n0"}], "evidence": {"package_sha256": True}}
        with self.assertRaises(ReciboInvalido) as caja:
            firmar_recibo(minimo, clave=_CLAVE, key_id="k1")
        # Y DICE QUÉ FALTA, una por una: «no cumple §14.2» a secas manda a
        # leerse el contrato para averiguar cuál.
        dicho = str(caja.exception)
        for seccion in ("event_type", "created_at", "subject", "output"):
            self.assertIn(seccion, dicho)

    def test_sin_el_CAMINO_EJECUTADO_tampoco(self):
        """P23-R-0015 lo declara **DEBE**: sin él, dos recorridos distintos
        del mismo grafo producen recibos idénticos."""
        for campo in ("pipeline_digest", "executed_path_digest", "nodes"):
            with self.subTest(campo=campo):
                grafo = dict(_recibo()["pipeline"])
                del grafo[campo]
                with self.assertRaises(ReciboInvalido) as caja:
                    firmar_recibo(_recibo(pipeline=grafo), clave=_CLAVE, key_id="k1")
                self.assertIn(campo, str(caja.exception))

    def test_un_event_type_INVENTADO_no_se_libra_de_las_secciones(self):
        """Fail-closed: si un tipo desconocido pasara, bastaría
        inventárselo para que no se exigiera nada."""
        with self.assertRaises(ReciboInvalido) as caja:
            firmar_recibo(_recibo(event_type="lo_que_sea"), clave=_CLAVE, key_id="k1")
        self.assertIn("desconocido", str(caja.exception))

    def test_el_BOOLEANO_no_compra_un_A2(self):
        """El nivel lo elegía quien escribía el recibo, que es justo lo
        que `nivel_del_recibo` existe para impedir."""
        firmado = firmar_recibo(_recibo(evidence={"package_sha256": True}),
                                clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A1")

    def test_ni_una_cadena_que_no_es_un_DIGEST(self):
        firmado = firmar_recibo(_recibo(evidence={"package_sha256": "si"}),
                                clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A1")

    def test_ni_una_referencia_VACIA(self):
        """Un `{}` es tan poco reproducible como la ausencia."""
        firmado = firmar_recibo(_recibo(evidence={"replay_reference": {}}),
                                clave=_CLAVE, key_id="k1")
        self.assertEqual(nivel_del_recibo(firmado), "A1")

    def test_pero_una_evidencia_QUE_SE_PUEDE_SEGUIR_sigue_dando_A2(self):
        """El otro lado, que es el que se rompe al apretar: arreglar un
        sesgo puede crear el contrario."""
        for evidencia in ({"package_sha256": "sha256:" + "e" * 64},
                          {"package_sha256": "e" * 64},
                          {"replay_reference": "rep_local.json"},
                          {"reproduce": {"package_sha256": "e" * 64}}):
            with self.subTest(evidencia=evidencia):
                firmado = firmar_recibo(_recibo(evidence=evidencia),
                                        clave=_CLAVE, key_id="k1")
                self.assertEqual(nivel_del_recibo(firmado), "A2")
