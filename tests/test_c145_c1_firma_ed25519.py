"""145-C1 — la firma ASIMÉTRICA Ed25519 del recibo (Ola 2, idea 30).

Decisiones de Roberto (31-10-2026) que sostienen estas pruebas:

* **D4** Ed25519 como SEGUNDA firma DSSE junto a la HMAC; `cryptography` es un extra
  opcional (`[firma]`); la HMAC se mantiene y **el nivel A no cambia por la firma**.
* **D11** vocabulario cerrado «firma válida / no coincide / no se pudo comprobar». Nunca
  «verificado», «auténtico» ni «certificado» a secas.

Lo que NO se promete (y por eso hay pruebas de lo que no sube): una firma válida no dice quién
es el firmante, ni que el modelo sea correcto, ni la hora; no hay revocación.
"""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import unittest
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

from cryptography.hazmat.primitives import serialization  # noqa: E402  (la prueba SÍ la necesita)
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from matrixai.pipelines import firma_asimetrica as A  # noqa: E402
from matrixai.pipelines import receipt as R  # noqa: E402
from matrixai.pipelines.receipt import (  # noqa: E402
    PAYLOAD_TYPE,
    ReciboInvalido,
    firmar_recibo,
    nivel_del_recibo,
    nivel_si_la_firma_vale,
    sobre_in_toto,
    verificar_recibo,
)
from matrixai.pipelines.verifier import verificar_sobre  # noqa: E402

from test_c81_recibos import _recibo  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
_HMAC = b"0" * 32
_EVIDENCIA = {"trace_root": "sha256:" + "4" * 64, "package_sha256": "a" * 64}

#: Lo que D11 prohíbe como afirmación suelta. `autenticidad` NO casa (otra palabra, y es la que
#: el producto usa para decir que NO se demuestra).
_PROHIBIDAS = re.compile(
    r"verificad[oa]s?|aut[eé]ntic[oa]s?|certificad[oa]s?|\bverified\b|\bauthentic\b|\bcertified\b",
    re.IGNORECASE)

_PAR = A.generar_par()  # (privada_pem, publica_pem): uno por proceso basta, no se compara entre pruebas
_OTRO_PAR = A.generar_par()


def _doble(recibo=None, **kw):
    """Un sobre con LAS DOS firmas: HMAC + Ed25519."""
    return firmar_recibo(recibo or _recibo(), clave=_HMAC, key_id="k1",
                         clave_privada=kw.get("privada", _PAR[0]))


def _solo_ed(recibo=None):
    return firmar_recibo(recibo or _recibo(), clave_privada=_PAR[0])


def _estado(sobre, publica=_PAR[1], **kw):
    return verificar_recibo(sobre, clave_publica=publica, **kw)["firmas"]["ed25519"]["estado"]


def _tocar_payload(sobre):
    malo = copy.deepcopy(sobre)
    cuerpo = json.loads(base64.b64decode(malo["payload"]))
    cuerpo["receipt_id"] = "otro"
    malo["payload"] = base64.b64encode(json.dumps(cuerpo, sort_keys=True).encode()).decode()
    return malo


# ───────────────────────────────────────────────────────────────────────── ida y vuelta, y el formato

class IdaYVueltaTest(unittest.TestCase):
    def test_firma_y_comprueba_el_recibo_nativo_y_el_in_toto(self):
        for sobre in (_doble(), sobre_in_toto(_recibo(), clave=_HMAC, key_id="k1",
                                              clave_privada=_PAR[0])):
            r = verificar_recibo(sobre, clave=_HMAC, clave_publica=_PAR[1])
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["firmas"]["hmac"]["estado"], "valida")
            self.assertEqual(r["firmas"]["ed25519"]["estado"], "valida")
            self.assertEqual(r["firmas"]["ed25519"]["huella"],
                             A.huella_de_clave_publica(_PAR[1]))

    def test_con_solo_la_publica_basta(self):
        """El sentido de la asimétrica: quien comprueba NO tiene el secreto HMAC."""
        self.assertTrue(verificar_recibo(_doble(), clave_publica=_PAR[1])["ok"])
        self.assertTrue(verificar_recibo(_solo_ed(), clave_publica=_PAR[1])["ok"])

    def test_el_sobre_con_dos_firmas_tiene_la_forma_documentada(self):
        sobre = _doble()
        hmac_, ed = sobre["signatures"]
        # la HMAC, EXACTAMENTE como antes del 145 (no se cambia el sobre que ya hay por ahí)
        self.assertEqual(set(hmac_), {"keyid", "sig"})
        self.assertEqual(hmac_["keyid"], "k1")
        # la Ed25519: algoritmo EXPLÍCITO y keyid = huella de la pública
        self.assertEqual(set(ed), {"keyid", "alg", "sig"})
        self.assertEqual(ed["alg"], "ed25519")
        self.assertEqual(ed["keyid"], A.huella_de_clave_publica(_PAR[1]))
        self.assertRegex(ed["keyid"], r"\A[0-9a-f]{64}\Z")
        self.assertEqual(len(base64.b64decode(ed["sig"], validate=True)), 64)

    def test_las_dos_firman_el_MISMO_PAE_y_el_payload_no_se_recanonicaliza(self):
        """No re-canonicalizar: el payload del sobre con Ed25519 es el MISMO que el del sobre solo-HMAC,
        y la firma se comprueba, con `cryptography` a pelo, sobre PAE(payloadType, bytes del payload)."""
        recibo = _recibo()
        solo_hmac = firmar_recibo(recibo, clave=_HMAC, key_id="k1")
        doble = _doble(recibo)
        self.assertEqual(doble["payload"], solo_hmac["payload"])
        self.assertEqual(doble["signatures"][0], solo_hmac["signatures"][0])
        crudo = base64.b64decode(doble["payload"])
        pae = b"DSSEv1 %d %s %d %s" % (len(PAYLOAD_TYPE), PAYLOAD_TYPE.encode(), len(crudo), crudo)
        publica = serialization.load_pem_public_key(_PAR[1].encode())
        publica.verify(base64.b64decode(doble["signatures"][1]["sig"]), pae)  # no lanza

    def test_la_HMAC_sin_clave_privada_sale_igual_que_siempre(self):
        sobre = firmar_recibo(_recibo(), clave=_HMAC, key_id="k1")
        self.assertEqual(len(sobre["signatures"]), 1)
        self.assertNotIn("alg", sobre["signatures"][0])

    def test_firmar_sin_ninguna_clave_o_la_HMAC_sin_keyid_se_corta_antes(self):
        with self.assertRaises(ReciboInvalido):
            firmar_recibo(_recibo())
        with self.assertRaises(ReciboInvalido):
            firmar_recibo(_recibo(), clave=_HMAC, key_id="  ", clave_privada=_PAR[0])


class LaHuellaEsUnFormatoEstableTest(unittest.TestCase):
    """Vector 1 de la RFC 8032 (§7.1): ancla el ALGORITMO (no Ed25519ph, no otra curva) y la HUELLA."""
    SEMILLA = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    PUBLICA = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    FIRMA = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e3970"
        "1cf9b46bd25bf5f0595bbe24655141438e7a100b")

    def test_la_huella_es_el_sha256_de_los_32_bytes_crudos_en_hex_minuscula(self):
        esperada = hashlib.sha256(self.PUBLICA).hexdigest()
        # el valor fijado a mano: si cambia el formato, esto se pone rojo
        self.assertEqual(esperada, "21fe31dfa154a261626bf854046fd2271b7bed4b6abe45aa58877ef47f9721b9")
        self.assertEqual(A.huella_de_clave_publica(self.PUBLICA), esperada)
        # y da igual cómo llegue la misma clave: crudos, hex, base64, PEM, privada
        privada = ed25519.Ed25519PrivateKey.from_private_bytes(self.SEMILLA)
        pem = A.clave_publica_pem(privada)
        for origen in (self.PUBLICA.hex(), base64.b64encode(self.PUBLICA).decode(), pem,
                       pem.encode(), privada, privada.public_key()):
            self.assertEqual(A.huella_de_clave_publica(origen), esperada, origen)

    def test_la_firma_del_vector_de_la_RFC_se_comprueba_y_la_nuestra_coincide_byte_a_byte(self):
        self.assertTrue(A.comprobar(self.PUBLICA, b"", base64.b64encode(self.FIRMA).decode()))
        self.assertEqual(base64.b64decode(A.firmar(self.SEMILLA, b"")), self.FIRMA)  # determinista
        self.assertFalse(A.comprobar(self.PUBLICA, b"x", base64.b64encode(self.FIRMA).decode()))

    def test_un_par_nuevo_cada_vez_y_la_privada_no_se_parece_a_la_publica(self):
        p1, q1 = A.generar_par()
        p2, q2 = A.generar_par()
        self.assertNotEqual(p1, p2)
        self.assertNotEqual(A.huella_de_clave_publica(q1), A.huella_de_clave_publica(q2))
        self.assertIn(b"BEGIN PRIVATE KEY", p1)
        self.assertNotIn("PRIVATE", q1)


# ───────────────────────────────────────────────────────────────────────── los tres estados

class LosTresEstadosTest(unittest.TestCase):
    def test_payload_tocado__no_coincide(self):
        sobre = _tocar_payload(_doble())
        self.assertEqual(_estado(sobre), "no_coincide")
        self.assertFalse(verificar_recibo(sobre, clave_publica=_PAR[1])["ok"])

    def test_payloadType_cambiado__no_coincide(self):
        sobre = _doble()
        sobre["payloadType"] = "application/vnd.matrixai.policy+json"
        self.assertEqual(_estado(sobre), "no_coincide")
        # y también el del in-toto: un tipo cambiado a mano no vale en otro sitio
        intoto = sobre_in_toto(_recibo(), clave_privada=_PAR[0])
        intoto["payloadType"] = PAYLOAD_TYPE
        self.assertEqual(_estado(intoto), "no_coincide")

    def test_firma_basura__no_coincide(self):
        for basura in ("AAAA", "no es base64!!", base64.b64encode(os.urandom(64)).decode(),
                       base64.b64encode(os.urandom(63)).decode(), "", None, 7):
            sobre = _doble()
            sobre["signatures"][1]["sig"] = basura
            self.assertEqual(_estado(sobre), "no_coincide", basura)

    def test_una_firma_tocada_en_un_bit__no_coincide(self):
        sobre = _doble()
        sig = bytearray(base64.b64decode(sobre["signatures"][1]["sig"]))
        sig[0] ^= 1
        sobre["signatures"][1]["sig"] = base64.b64encode(bytes(sig)).decode()
        self.assertEqual(_estado(sobre), "no_coincide")

    def test_otra_clave__no_coincide_y_el_motivo_dice_las_dos_huellas(self):
        r = verificar_recibo(_doble(), clave_publica=_OTRO_PAR[1])
        self.assertEqual(r["firmas"]["ed25519"]["estado"], "no_coincide")
        self.assertIn(A.huella_de_clave_publica(_PAR[1]), r["firmas"]["ed25519"]["motivo"])
        self.assertIn(A.huella_de_clave_publica(_OTRO_PAR[1]), r["firmas"]["ed25519"]["motivo"])
        self.assertFalse(r["ok"])

    def test_sin_clave_publica__no_se_pudo_comprobar_NUNCA_valida(self):
        for sobre in (_doble(), _solo_ed()):
            r = verificar_recibo(sobre)
            self.assertFalse(r["ok"])
            self.assertEqual(r["firmas"]["ed25519"]["estado"], "no_se_pudo_comprobar")
            r2 = verificar_recibo(sobre, clave=_HMAC)  # con la HMAC tampoco se da por comprobada la otra
            self.assertEqual(r2["firmas"]["ed25519"]["estado"], "no_se_pudo_comprobar")

    def test_una_clave_publica_ilegible__no_se_pudo_comprobar_no_es_una_acusacion(self):
        r = verificar_recibo(_doble(), clave_publica="esto no es una clave")
        self.assertEqual(r["firmas"]["ed25519"]["estado"], "no_se_pudo_comprobar")
        self.assertFalse(r["ok"])

    def test_un_sobre_sin_firma_Ed25519_no_cumple_una_pregunta_por_ella(self):
        r = verificar_recibo(firmar_recibo(_recibo(), clave=_HMAC, key_id="k1"),
                             clave_publica=_PAR[1])
        self.assertEqual(r["firmas"]["ed25519"]["estado"], "no_se_pudo_comprobar")
        self.assertFalse(r["ok"])

    def test_una_Ed25519_SIN_su_alg_no_cuenta_como_Ed25519(self):
        """El algoritmo va explícito y es lo que decide cómo se lee la firma."""
        sobre = _doble()
        del sobre["signatures"][1]["alg"]
        self.assertEqual(_estado(sobre), "no_se_pudo_comprobar")
        sobre["signatures"][1]["alg"] = "ed448"
        self.assertEqual(_estado(sobre), "no_se_pudo_comprobar")

    def test_el_keyid_es_una_etiqueta_que_nadie_firma__no_decide_ni_se_repite(self):
        sobre = _doble()
        sobre["signatures"][1]["keyid"] = "f" * 64   # mentira
        r = verificar_recibo(sobre, clave_publica=_PAR[1])
        self.assertTrue(r["ok"])  # la firma SÍ vale con esa clave...
        self.assertEqual(r["firmas"]["ed25519"]["huella"], A.huella_de_clave_publica(_PAR[1]))
        self.assertNotIn("f" * 64, json.dumps(r))   # ...y no se repite la etiqueta falsa

    def test_con_dos_Ed25519_basta_que_una_valga(self):
        sobre = _doble()
        sobre["signatures"].append({"keyid": "x", "alg": "ed25519", "sig": "AAAA"})
        self.assertEqual(_estado(sobre), "valida")

    def test_el_estado_es_de_la_ed_y_no_de_la_hmac(self):
        """La HMAC mala no vuelve mala la Ed25519 ni al revés: cada una dice lo suyo."""
        sobre = _doble()
        sobre["signatures"][0]["sig"] = "AAAA"
        r = verificar_recibo(sobre, clave=_HMAC, clave_publica=_PAR[1])
        self.assertEqual(r["firmas"]["hmac"]["estado"], "no_coincide")
        self.assertEqual(r["firmas"]["ed25519"]["estado"], "valida")
        self.assertFalse(r["ok"])   # se pidieron las dos, y una no cuadra


# ───────────────────────────────────────────────────────────────────────── el nivel NO sube

class ElNivelANoSubePorLaEd25519Test(unittest.TestCase):
    def test_el_nivel_es_el_MISMO_con_y_sin_la_segunda_firma(self):
        for evidencia in (None, _EVIDENCIA):
            r = _recibo()
            if evidencia:
                r["evidence"] = evidencia
            solo = firmar_recibo(r, clave=_HMAC, key_id="k1")
            doble = firmar_recibo(r, clave=_HMAC, key_id="k1", clave_privada=_PAR[0])
            self.assertEqual(nivel_del_recibo(doble, clave=_HMAC), nivel_del_recibo(solo, clave=_HMAC))
            self.assertEqual(nivel_del_recibo(doble), nivel_del_recibo(solo))
            self.assertEqual(nivel_si_la_firma_vale(doble), nivel_si_la_firma_vale(solo))
            self.assertEqual(verificar_sobre(doble, clave=_HMAC, clave_publica=_PAR[1])["assurance_level"],
                             verificar_sobre(solo, clave=_HMAC)["assurance_level"])

    def test_un_sobre_con_SOLO_Ed25519_se_queda_en_A0_aunque_la_firma_valga(self):
        sobre = _solo_ed()
        self.assertTrue(verificar_recibo(sobre, clave_publica=_PAR[1])["ok"])
        self.assertEqual(nivel_del_recibo(sobre), "A0")
        self.assertEqual(nivel_del_recibo(sobre, clave=_HMAC), "A0")
        self.assertEqual(nivel_si_la_firma_vale(sobre), "A0")
        informe = verificar_sobre(sobre, clave_publica=_PAR[1])
        self.assertEqual(informe["assurance_level"], "A0")
        self.assertEqual(informe["signature_ed25519"]["estado"], "valida")

    def test_con_solo_la_publica_el_nivel_del_doble_sigue_siendo_A0(self):
        """Sin la clave HMAC «no se pudo comprobar» la HMAC, y eso es A0 (cabo del 30-10)."""
        informe = verificar_sobre(_doble(), clave_publica=_PAR[1])
        self.assertEqual(informe["assurance_level"], "A0")
        self.assertEqual(informe["assurance_if_signature_holds"], "A1")

    def test_una_firma_de_un_algoritmo_desconocido_tampoco_sube_el_nivel(self):
        sobre = _solo_ed()
        sobre["signatures"][0]["alg"] = "dilithium9"
        self.assertEqual(nivel_si_la_firma_vale(sobre), "A0")

    def test_una_firma_Ed25519_basura_no_da_A1_con_HMAC_buena_ausente(self):
        sobre = _doble()
        sobre["signatures"] = [{"keyid": "x", "alg": "ed25519", "sig": "AAAA"}]
        self.assertEqual(nivel_del_recibo(sobre, clave=_HMAC), "A0")


# ───────────────────────────────────────────────────────────────────────── el informe del verificador

class ElVerificadorTest(unittest.TestCase):
    def test_valida_basta_para_dar_por_comprobado_y_la_otra_queda_a_la_vista(self):
        informe = verificar_sobre(_doble(), clave_publica=_PAR[1])
        self.assertIn("signature_ed25519", informe["verified"])
        self.assertIn("signature", informe["unverified"])      # la HMAC se dice, no se calla
        self.assertTrue(informe["fully_checked"])
        self.assertEqual(informe["unchecked"], [])
        self.assertEqual(informe["problems"], [])

    def test_no_coincide_es_un_problema_y_nunca_fully_checked(self):
        informe = verificar_sobre(_tocar_payload(_doble()), clave=None, clave_publica=_PAR[1])
        self.assertTrue(informe["problems"])
        self.assertFalse(informe["fully_checked"])
        self.assertEqual(informe["signature_ed25519"]["estado"], "no_coincide")

    def test_con_la_HMAC_buena_pero_la_Ed25519_PEDIDA_y_ausente_no_es_un_aprobado(self):
        """El que quita la Ed25519 del sobre no consigue un `fully_checked` por tener la HMAC buena
        cuando quien comprueba PIDIÓ la Ed25519."""
        sobre = firmar_recibo(_recibo(), clave=_HMAC, key_id="k1")
        informe = verificar_sobre(sobre, clave=_HMAC, clave_publica=_PAR[1])
        self.assertFalse(informe["fully_checked"])
        self.assertIn("signature_ed25519", informe["unchecked"])

    def test_si_no_se_pidio_la_Ed25519_y_la_HMAC_vale_el_informe_de_siempre(self):
        informe = verificar_sobre(_doble(), clave=_HMAC)
        self.assertTrue(informe["fully_checked"])
        self.assertEqual(informe["signature_ed25519"]["estado"], "no_se_pudo_comprobar")
        self.assertIn("signature_ed25519", informe["unverified"])
        self.assertEqual(informe["assurance_level"], "A1")

    def test_un_sobre_sin_Ed25519_y_sin_pedirla_no_tiene_entrada(self):
        informe = verificar_sobre(firmar_recibo(_recibo(), clave=_HMAC, key_id="k1"), clave=_HMAC)
        self.assertIsNone(informe["signature_ed25519"])
        self.assertNotIn("signature_ed25519", informe["unverified"])

    def test_el_informe_de_un_sobre_ilegible_lleva_el_mismo_campo(self):
        self.assertIn("signature_ed25519", verificar_sobre({"payload": "%%%"}, clave=None))


# ───────────────────────────────────────────────────────────────────────── sin cryptography

_BLOQUEO = textwrap.dedent('''
    import importlib.abc, sys
    class _SinCrypto(importlib.abc.MetaPathFinder):
        def find_spec(self, nombre, ruta=None, destino=None):
            if nombre == "cryptography" or nombre.startswith("cryptography."):
                raise ImportError("cryptography bloqueada por la prueba")
    sys.meta_path.insert(0, _SinCrypto())
''')


class SinCryptographyTest(unittest.TestCase):
    """`cryptography` es un EXTRA. Sin ella, nada se rompe salvo lo que la necesita, y lo dice."""

    def _sin_cryptography(self):
        pisados = {n: None for n in list(sys.modules) if n == "cryptography" or n.startswith("cryptography.")}
        pisados.update({"cryptography": None, "cryptography.exceptions": None,
                        "cryptography.hazmat.primitives": None,
                        "cryptography.hazmat.primitives.asymmetric": None})
        return mock.patch.dict(sys.modules, pisados)

    def test_firmar_con_Ed25519_da_un_error_claro(self):
        with self._sin_cryptography():
            self.assertFalse(A.disponible())
            with self.assertRaises(A.FirmaNoDisponible) as cm:
                firmar_recibo(_recibo(), clave=_HMAC, key_id="k1", clave_privada=_PAR[0])
            self.assertIn("cryptography", str(cm.exception))
            self.assertIn("matrixai[firma]", str(cm.exception))
            with self.assertRaises(A.FirmaNoDisponible):
                A.generar_par()

    def test_la_HMAC_sigue_firmando_y_comprobando(self):
        sobre = _doble()   # con Ed25519, hecho CON cryptography
        with self._sin_cryptography():
            solo = firmar_recibo(_recibo(), clave=_HMAC, key_id="k1")
            self.assertTrue(verificar_recibo(solo, clave=_HMAC)["ok"])
            self.assertTrue(verificar_recibo(sobre, clave=_HMAC)["ok"])   # un sobre doble, sin pública
            self.assertEqual(nivel_del_recibo(sobre, clave=_HMAC), "A1")
            self.assertTrue(verificar_sobre(sobre, clave=_HMAC)["fully_checked"])

    def test_comprobar_la_Ed25519_dice_no_se_pudo_comprobar_falta_cryptography(self):
        sobre = _doble()
        with self._sin_cryptography():
            r = verificar_recibo(sobre, clave_publica=_PAR[1])
            est = r["firmas"]["ed25519"]
            self.assertEqual(est["estado"], "no_se_pudo_comprobar")
            self.assertIn("falta cryptography", est["motivo"])
            self.assertFalse(r["ok"])
            informe = verificar_sobre(sobre, clave=_HMAC, clave_publica=_PAR[1])
            self.assertFalse(informe["fully_checked"])   # se pidió, y no se pudo

    def test_en_un_proceso_de_verdad_sin_la_biblioteca_el_paquete_importa_y_firma_con_HMAC(self):
        codigo = _BLOQUEO + textwrap.dedent('''
            import sys
            sys.path.insert(0, "tests")
            from matrixai.pipelines import receipt as R
            from test_c81_recibos import _recibo
            s = R.firmar_recibo(_recibo(), clave=b"k", key_id="k1")
            assert R.verificar_recibo(s, clave=b"k")["ok"]
            assert "cryptography" not in sys.modules, "importar el recibo no debe cargar cryptography"
            try:
                R.firmar_recibo(_recibo(), clave_privada=b"x" * 32)
            except R._asim.FirmaNoDisponible as exc:
                print("ERROR-CLARO:", exc)
        ''')
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=RAIZ)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("ERROR-CLARO: falta cryptography", r.stdout)


# ───────────────────────────────────────────────────────────────────────── la privada no se escapa

class LaPrivadaNoSaleTest(unittest.TestCase):
    def _rastros(self, pem: bytes) -> list[str]:
        privada = serialization.load_pem_private_key(pem, password=None)
        semilla = privada.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                        serialization.NoEncryption())
        cuerpo = b"".join(pem.strip().splitlines()[1:-1]).decode()
        return [semilla.hex(), base64.b64encode(semilla).decode(), cuerpo, "PRIVATE KEY",
                base64.urlsafe_b64encode(semilla).decode().rstrip("=")]

    def test_ni_el_sobre_ni_el_informe_ni_el_log_llevan_la_privada(self):
        pem = _PAR[0]
        rastros = self._rastros(pem)
        with self.assertLogs(level=logging.DEBUG) as logs:
            logging.getLogger("matrixai").debug("sonda")   # para que assertLogs no falle si no hay nada
            sobre = firmar_recibo(_recibo(), clave=_HMAC, key_id="k1", clave_privada=pem)
            intoto = sobre_in_toto(_recibo(), clave_privada=pem)
            informe = verificar_sobre(sobre, clave=_HMAC, clave_publica=_PAR[1])
            crudo = verificar_recibo(sobre, clave=_HMAC, clave_publica=_PAR[1])
        todo = json.dumps([sobre, intoto, informe, crudo]) + "\n".join(logs.output)
        for rastro in rastros:
            self.assertNotIn(rastro, todo)

    def test_un_error_de_clave_no_repite_el_material(self):
        pem = _PAR[0]
        truncada = pem[:-40]   # un PEM roto
        for malo in (truncada, b"-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n", b"x" * 31):
            with self.assertRaises(A.ClaveInvalida) as cm:
                firmar_recibo(_recibo(), clave_privada=malo)
            mensaje = str(cm.exception) + repr(cm.exception)
            self.assertNotIn(malo.decode(errors="replace").strip(), mensaje)
            for rastro in self._rastros(pem):
                self.assertNotIn(rastro, mensaje)

    def test_una_clave_PUBLICA_no_sirve_para_firmar_y_una_privada_de_otro_tipo_tampoco(self):
        with self.assertRaises(A.ClaveInvalida):
            firmar_recibo(_recibo(), clave_privada=_PAR[1])
        rsa_pem = subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt",
                                  "rsa_keygen_bits:2048"], capture_output=True) if shutil.which("openssl") else None
        if rsa_pem is not None and rsa_pem.returncode == 0:
            with self.assertRaises(A.ClaveInvalida):
                firmar_recibo(_recibo(), clave_privada=rsa_pem.stdout)

    def test_pubkey_acepta_una_privada_y_solo_enseña_la_publica(self):
        publica = A.cargar_clave_publica(_PAR[0])
        self.assertEqual(A.clave_publica_pem(publica), _PAR[1])


# ───────────────────────────────────────────────────────────────────────── D11: el vocabulario

def _constantes_de_texto(arbol: ast.AST, *, sin_docstrings=True) -> list[str]:
    docstrings = set()
    if sin_docstrings:
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                cuerpo = nodo.body
                if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                    docstrings.add(id(cuerpo[0].value))
    return [n.value for n in ast.walk(arbol)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


class ElVocabularioDeD11Test(unittest.TestCase):
    def test_el_barrido_tiene_dientes(self):
        for malo in ("Firma verificada", "firma auténtica", "Certificado", "verified", "Authentic key",
                     "certified"):
            self.assertTrue(_PROHIBIDAS.search(malo), malo)
        for bueno in ("firma válida", "no se pudo comprobar", "no autenticidad", "valid signature",
                      "could not be checked", "verificar", "verifies"):
            self.assertFalse(_PROHIBIDAS.search(bueno), bueno)

    def test_el_vocabulario_es_exactamente_tres_frases_por_idioma_sin_palabras_prohibidas(self):
        self.assertEqual(set(A.ESTADOS), {"valida", "no_coincide", "no_se_pudo_comprobar"})
        for idioma in ("es", "en"):
            self.assertEqual(set(A.VOCABULARIO[idioma]), set(A.ESTADOS))
            for estado in A.ESTADOS:
                texto = A.texto_del_estado(estado, idioma)
                self.assertFalse(_PROHIBIDAS.search(texto), texto)
        self.assertEqual(A.texto_del_estado("valida", "es"), "firma válida")
        self.assertEqual(A.texto_del_estado("no_se_pudo_comprobar", "es"), "no se pudo comprobar")

    def test_los_textos_del_codigo_nuevo_no_dicen_verificado_autentico_ni_certificado(self):
        """Estático: TODAS las constantes de texto (docstrings aparte) del módulo nuevo y de las
        funciones nuevas de `receipt.py` y de la CLI."""
        fuente = (RAIZ / "matrixai/pipelines/firma_asimetrica.py").read_text(encoding="utf-8")
        textos = _constantes_de_texto(ast.parse(fuente))
        self.assertGreater(len(textos), 10)
        receipt = ast.parse((RAIZ / "matrixai/pipelines/receipt.py").read_text(encoding="utf-8"))
        nuevas = {"_firmas_del_sobre", "_firma_pedida", "verificar_recibo", "_comprobar_ed25519",
                  "firmar_recibo", "sobre_in_toto"}
        for nodo in ast.walk(receipt):
            if isinstance(nodo, ast.FunctionDef) and nodo.name in nuevas:
                textos += _constantes_de_texto(nodo)
            if isinstance(nodo, ast.Assign) and any(getattr(t, "id", "") in ("_NOTA_HMAC", "_NOTA_ED25519")
                                                     for t in nodo.targets):
                textos += _constantes_de_texto(nodo)
        cli = ast.parse((RAIZ / "matrixai/cli.py").read_text(encoding="utf-8"))
        for nodo in ast.walk(cli):
            if isinstance(nodo, ast.FunctionDef) and nodo.name in ("_linea_ed25519", "_cmd_receipt_pubkey"):
                textos += _constantes_de_texto(nodo)
        for texto in textos:
            self.assertFalse(_PROHIBIDAS.search(texto), texto)

    def test_los_avisos_que_viajan_en_cada_informe_tampoco(self):
        from matrixai.pipelines.verifier import _AVISOS
        for aviso in _AVISOS:
            self.assertFalse(_PROHIBIDAS.search(aviso), aviso)

    def test_los_motivos_y_notas_que_salen_en_cada_escenario_tampoco(self):
        escenarios = [
            verificar_recibo(_doble(), clave=_HMAC, clave_publica=_PAR[1]),
            verificar_recibo(_tocar_payload(_doble()), clave=_HMAC, clave_publica=_PAR[1]),
            verificar_recibo(_doble(), clave_publica=_OTRO_PAR[1]),
            verificar_recibo(_doble()),
            verificar_recibo(_doble(), clave_publica="ni idea"),
            verificar_recibo(firmar_recibo(_recibo(), clave=_HMAC, key_id="k"), clave_publica=_PAR[1]),
            verificar_recibo({"payload": "%%%"}, clave_publica=_PAR[1]),
        ]
        for informe in escenarios:
            for texto in re.findall(r'"(?:motivo|reason|note)": "([^"]*)"', json.dumps(informe, ensure_ascii=False)):
                self.assertFalse(_PROHIBIDAS.search(texto), texto)
        valida = json.dumps(escenarios[0], ensure_ascii=False)
        self.assertIn("firma válida", valida)           # y la nota no promete lo que no es
        self.assertIn("no hay revocación", valida)


# ───────────────────────────────────────────────────────────────────────── la CLI

def _cli(*argv, entorno=None):
    env = dict(os.environ, **(entorno or {}))
    return subprocess.run([sys.executable, "-m", "matrixai", *argv], capture_output=True, text=True,
                          cwd=RAIZ, env=env)


def _cli_sin_cryptography(*argv):
    codigo = _BLOQUEO + "import sys\nfrom matrixai.cli import main\nsys.argv = ['matrixai'] + sys.argv[1:]\nsys.exit(main())\n"
    return subprocess.run([sys.executable, "-c", codigo, *argv], capture_output=True, text=True, cwd=RAIZ)


def _bloque_ed25519(salida: str, etiqueta: str) -> str:
    lineas = salida.splitlines()
    i = next(n for n, l in enumerate(lineas) if l.startswith(etiqueta))
    bloque = [lineas[i]]
    for l in lineas[i + 1:]:
        if not l.startswith("  "):
            break
        bloque.append(l)
    return "\n".join(bloque)


class LaCLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = TemporaryDirectory()
        d = Path(cls._tmp.name)
        cls.d = d
        cls.privada = d / "priv.pem"
        cls.privada.write_bytes(_PAR[0])
        cls.publica = d / "pub.pem"
        cls.publica.write_text(_PAR[1])
        cls.otra = d / "otra.pem"
        cls.otra.write_text(_OTRO_PAR[1])
        cls.sobre = d / "r.json"
        cls.sobre.write_text(json.dumps(_doble()))
        cls.tocado = d / "tocado.json"
        cls.tocado.write_text(json.dumps(_tocar_payload(_doble())))
        cls.solo_hmac = d / "hmac.json"
        cls.solo_hmac.write_text(json.dumps(firmar_recibo(_recibo(), clave=_HMAC, key_id="k1")))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_pubkey_da_la_huella_y_la_publica_de_la_privada_o_de_la_publica(self):
        huella = A.huella_de_clave_publica(_PAR[1])
        for fichero in (self.privada, self.publica):
            r = _cli("receipt", "pubkey", str(fichero))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"fingerprint: {huella}", r.stdout)
            self.assertIn(_PAR[1].strip(), r.stdout)
            self.assertNotIn("PRIVATE", r.stdout + r.stderr)
        j = json.loads(_cli("receipt", "--json", "pubkey", str(self.privada)).stdout)
        self.assertEqual(j["fingerprint"], huella)
        self.assertEqual(j["algorithm"], "ed25519")

    def test_pubkey_no_imprime_nunca_la_privada(self):
        r = _cli("receipt", "pubkey", str(self.privada))
        semilla = serialization.load_pem_private_key(_PAR[0], None).private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        for rastro in (semilla.hex(), base64.b64encode(semilla).decode(),
                       b"".join(_PAR[0].strip().splitlines()[1:-1]).decode()):
            self.assertNotIn(rastro, r.stdout + r.stderr)

    def test_pubkey_con_un_fichero_que_no_es_una_clave_o_que_no_existe_sale_con_1_y_lo_dice(self):
        malo = self.d / "malo.txt"
        malo.write_text("hola")
        for f in (malo, self.d / "no-existe.pem"):
            r = _cli("receipt", "pubkey", str(f))
            self.assertEqual(r.returncode, 1)
            self.assertIn("receipt:", r.stderr)
            self.assertEqual(r.stdout, "")

    def test_verify_valida__0_y_dice_firma_valida_en_los_dos_idiomas(self):
        r = _cli("receipt", "verify", str(self.sobre), "--pubkey", str(self.publica), "--locale", "es")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        bloque = _bloque_ed25519(r.stdout, "Firma Ed25519")
        self.assertIn("firma válida", bloque)
        self.assertIn(A.huella_de_clave_publica(_PAR[1]), bloque)
        self.assertIn("assurance: A0", r.stdout)         # y el nivel NO sube
        e = _cli("receipt", "verify", str(self.sobre), "--pubkey", str(self.publica), "--locale", "en")
        self.assertIn("valid signature", _bloque_ed25519(e.stdout, "Ed25519 signature"))

    def test_verify_no_coincide__2(self):
        for sobre, pub in ((self.tocado, self.publica), (self.sobre, self.otra)):
            r = _cli("receipt", "verify", str(sobre), "--pubkey", str(pub), "--locale", "es")
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("la firma no coincide", _bloque_ed25519(r.stdout, "Firma Ed25519"))

    def test_verify_sin_pubkey__no_se_pudo_comprobar__3_nunca_valida(self):
        r = _cli("receipt", "verify", str(self.sobre), "--locale", "es")
        self.assertEqual(r.returncode, 3, r.stdout)
        bloque = _bloque_ed25519(r.stdout, "Firma Ed25519")
        self.assertIn("no se pudo comprobar", bloque)
        self.assertNotIn("firma válida", bloque)

    def test_verify_pidiendo_la_Ed25519_de_un_sobre_que_ya_no_la_lleva__3(self):
        r = _cli("receipt", "verify", str(self.solo_hmac), "--key", _HMAC.hex(), "--pubkey",
                 str(self.publica), "--locale", "es")
        self.assertEqual(r.returncode, 3, r.stdout)

    def test_verify_con_la_HMAC_sola_sigue_dando_lo_de_siempre(self):
        r = _cli("receipt", "verify", str(self.solo_hmac), "--key", _HMAC.hex())
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertNotIn("Ed25519", r.stdout)
        self.assertIn("assurance: A1", r.stdout)

    def test_verify_con_una_pubkey_que_no_se_puede_leer__1_y_con_una_basura__3(self):
        r = _cli("receipt", "verify", str(self.sobre), "--pubkey", str(self.d / "no.pem"))
        self.assertEqual(r.returncode, 1)
        basura = self.d / "basura.pem"
        basura.write_text("no soy una clave")
        r = _cli("receipt", "verify", str(self.sobre), "--pubkey", str(basura), "--locale", "es")
        self.assertEqual(r.returncode, 3)
        self.assertNotIn("firma válida", _bloque_ed25519(r.stdout, "Firma Ed25519"))

    def test_verify_json_lleva_el_estado_cerrado(self):
        j = json.loads(_cli("receipt", "--json", "verify", str(self.sobre), "--pubkey",
                            str(self.publica)).stdout)
        self.assertEqual(j["signature_ed25519"]["estado"], "valida")
        self.assertEqual(j["assurance_level"], "A0")

    def test_sin_cryptography_la_CLI_lo_dice_y_la_HMAC_sigue(self):
        r = _cli_sin_cryptography("receipt", "pubkey", str(self.publica))
        self.assertEqual(r.returncode, 1)
        self.assertIn("falta cryptography", r.stderr)
        r = _cli_sin_cryptography("receipt", "verify", str(self.sobre), "--pubkey", str(self.publica),
                                  "--locale", "es")
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("falta cryptography", r.stdout)
        self.assertIn("no se pudo comprobar", r.stdout)
        r = _cli_sin_cryptography("receipt", "verify", str(self.solo_hmac), "--key", _HMAC.hex())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_los_textos_de_la_CLI_barridos_es_en_no_dicen_verificado_autentico_ni_certificado(self):
        salidas = []
        for idioma, etiqueta in (("es", "Firma Ed25519"), ("en", "Ed25519 signature")):
            for sobre, pub in ((self.sobre, self.publica), (self.tocado, self.publica),
                               (self.sobre, self.otra), (self.sobre, None)):
                args = ["receipt", "verify", str(sobre), "--locale", idioma]
                if pub:
                    args += ["--pubkey", str(pub)]
                salidas.append(_bloque_ed25519(_cli(*args).stdout, etiqueta))
        salidas.append(_cli("receipt", "pubkey", str(self.privada)).stdout)
        salidas.append(_cli("receipt", "pubkey", str(self.d / "no.pem")).stderr)
        salidas.append(_cli_sin_cryptography("receipt", "pubkey", str(self.publica)).stderr)
        self.assertEqual(len(salidas), 11)
        for texto in salidas:
            self.assertTrue(texto.strip())
            self.assertFalse(_PROHIBIDAS.search(texto), texto)


_HAS_ONNX = util.find_spec("onnxruntime") is not None and util.find_spec("onnx") is not None


@unittest.skipUnless(_HAS_ONNX, "onnx + onnxruntime required")
class AttestFirmaConEd25519Test(unittest.TestCase):
    def _attest(self, d: Path, *extra: str):
        from test_c86_integrado_en_el_producto import _datos_del_sigmoide
        from test_c87_c2_attest import _sigmoide
        return subprocess.run(
            [sys.executable, "-m", "matrixai", "attest", str(_sigmoide(d)), "--data",
             str(_datos_del_sigmoide(d)), "-o", str(d / "r.json"), *extra],
            capture_output=True, text=True, cwd=RAIZ)

    def test_con_key_y_ed25519_key_el_recibo_lleva_las_dos_y_el_nivel_es_el_de_siempre(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "k.pem").write_bytes(_PAR[0])
            r = self._attest(d, "--key", "aabb", "--in-toto", "--ed25519-key", str(d / "k.pem"))
            self.assertEqual(r.returncode, 0, r.stderr)
            sobre = json.loads((d / "r.json").read_text())
            intoto = json.loads((d / "r.in_toto.json").read_text())
            for s in (sobre, intoto):
                self.assertEqual([f.get("alg") for f in s["signatures"]], [None, "ed25519"])
                self.assertTrue(verificar_recibo(s, clave=bytes.fromhex("aabb"), clave_publica=_PAR[1])["ok"])
            self.assertIn("assurance: A1", r.stderr)
            self.assertNotIn("PRIVATE", r.stdout + r.stderr + (d / "r.json").read_text())

    def test_solo_con_ed25519_key_se_queda_en_A0_y_lo_dice_sin_decir_que_no_esta_firmado(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "k.pem").write_bytes(_PAR[0])
            r = self._attest(d, "--ed25519-key", str(d / "k.pem"))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("assurance: A0", r.stderr)
            self.assertIn("Ed25519 signature", r.stderr)
            self.assertNotIn("UNSIGNED", r.stderr)
            self.assertTrue(verificar_recibo(json.loads((d / "r.json").read_text()),
                                             clave_publica=_PAR[1])["ok"])

    def test_una_clave_que_no_vale_corta_antes_de_escribir_nada(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "k.pem").write_text("basura")
            r = self._attest(d, "--ed25519-key", str(d / "k.pem"))
            self.assertEqual(r.returncode, 2)
            self.assertFalse((d / "r.json").exists())


# ───────────────────────────────────────────────────────────────────────── paridad con WebCrypto

_NODE = shutil.which("node") or ("/usr/bin/node" if Path("/usr/bin/node").exists() else None)


def _node(entrada: dict) -> dict:
    """Corre `tests/c145_c1_webcrypto.mjs`. SIN node la prueba FALLA, salvo `MATRIXAI_SIN_NODE=1`:
    una paridad que se salta en silencio deja de vigilar el único sitio donde el JS de la web vive."""
    if _NODE is None:
        raise AssertionError("falta node: la paridad con WebCrypto no se puede comprobar "
                             "(MATRIXAI_SIN_NODE=1 para saltarla a propósito)")
    r = subprocess.run([_NODE, str(RAIZ / "tests/c145_c1_webcrypto.mjs")], input=json.dumps(entrada),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-600:]
    return json.loads(r.stdout)


@unittest.skipIf(os.environ.get("MATRIXAI_SIN_NODE") == "1" and _NODE is None, "MATRIXAI_SIN_NODE=1")
class ParidadConWebCryptoTest(unittest.TestCase):
    """La sonda 2 del inventario, hecha prueba: lo que firma Python lo comprueba `crypto.subtle`
    (que es lo que usará `/verificar`, C4) y al revés, con la clave en crudo y en SPKI/PEM."""

    def test_node_comprueba_lo_que_firma_Python_los_mismos_casos_que_Python(self):
        sobre = _doble()
        crudo = base64.b64decode(sobre["payload"])
        pae = R._pae(PAYLOAD_TYPE, crudo)
        sig = base64.b64decode(sobre["signatures"][1]["sig"])
        tocada = bytes([sig[0] ^ 1]) + sig[1:]
        casos = {
            "ok": (pae, sig),
            "payload_tocado": (R._pae(PAYLOAD_TYPE, crudo.replace(b"receipt_id", b"receipt_Id")), sig),
            "tipo_cambiado": (R._pae("application/vnd.matrixai.policy+json", crudo), sig),
            "firma_tocada": (pae, tocada),
        }
        publica = A.cargar_clave_publica(_PAR[1])
        raw = publica.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        b64 = lambda x: base64.b64encode(x).decode()  # noqa: E731
        salida = _node({"modo": "verificar", "raw": b64(raw), "pem": _PAR[1],
                        "casos": [{"nombre": n, "msg": b64(m), "sig": b64(s)} for n, (m, s) in casos.items()]})
        # la HUELLA también coincide: SHA-256 de los bytes crudos, calculado por WebCrypto
        self.assertEqual(salida["huella"], A.huella_de_clave_publica(_PAR[1]))
        for nombre, (mensaje, firma) in casos.items():
            en_python = A.comprobar(publica, mensaje, b64(firma))
            self.assertEqual(salida["resultados"][nombre], {"raw": en_python, "spki": en_python}, nombre)
        self.assertTrue(salida["resultados"]["ok"]["raw"])
        for nombre in ("payload_tocado", "tipo_cambiado", "firma_tocada"):
            self.assertFalse(salida["resultados"][nombre]["raw"], nombre)

    def test_Python_comprueba_lo_que_firma_WebCrypto_con_el_PAE_de_un_recibo_real(self):
        sobre = _doble()
        pae = R._pae(PAYLOAD_TYPE, base64.b64decode(sobre["payload"]))
        firmado = _node({"modo": "firmar", "msg": base64.b64encode(pae).decode()})
        self.assertTrue(A.comprobar(base64.b64decode(firmado["raw"]), pae, firmado["sig"]))
        # y un sobre ENTERO armado con la firma de Node se comprueba con nuestra API
        de_node = copy.deepcopy(sobre)
        de_node["signatures"] = [{"keyid": A.huella_de_clave_publica(base64.b64decode(firmado["raw"])),
                                  "alg": "ed25519", "sig": firmado["sig"]}]
        self.assertEqual(_estado(de_node, publica=base64.b64decode(firmado["raw"])), "valida")
        self.assertEqual(_estado(de_node), "no_coincide")   # con la clave de OTRA instalación, no


if __name__ == "__main__":
    unittest.main()
