"""86-C4 — FIRMAR CON SIGSTORE: lo que aporta, lo que no, y dónde no se puede.

La firma de hoy es HMAC: demuestra **consistencia** —el recibo no se ha tocado—
pero no **autenticidad**, porque las dos partes comparten la clave. Sigstore
responde a la otra pregunta: de quién es esto.

Lo que se fija aquí, y es lo que evita que esto se lea como más de lo que es:

1. **Sigstore NO sube el nivel del recibo.** A0–A4 hablan de lo que se
   COMPROBÓ, no de la fuerza de la firma. Un recibo firmado con Sigstore y sin
   evidencia reproducible sigue siendo A1.
2. **La biblioteca y la identidad son dos problemas distintos**, con dos
   soluciones distintas: juntarlos en un «no se puede» mandaría a instalar algo
   a quien ya lo tiene.
3. **Lo que no se puede firmar no devuelve un bundle a medias**: levanta el
   error con su motivo.

LO QUE ESTE FICHERO NO PRUEBA, y se dice: **la firma real**. El modo sin claves
necesita red y una identidad de verdad, y este servidor no las tiene. Lo probado
es todo lo que rodea a esa llamada.
"""
from __future__ import annotations

import unittest
from unittest import mock

from matrixai.pipelines.receipt import firmar_recibo, nivel_del_recibo
from matrixai.pipelines.sigstore_firma import (
    SigstoreNoDisponible,
    estado_de_sigstore,
    firmar_con_sigstore,
)


class DiceQueFaltaYPorQueTest(unittest.TestCase):
    def test_sin_biblioteca_y_sin_identidad_lo_dice_TODO(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch.dict("sys.modules", {"sigstore": None}):
            estado = estado_de_sigstore()
        self.assertFalse(estado["can_sign"])
        self.assertIn("sigstore", estado["reason"])
        self.assertIn("identidad OIDC", estado["reason"])

    def test_con_biblioteca_pero_sin_identidad_NO_manda_a_instalar_nada(self):
        """Mandar a instalar algo a quien ya lo tiene es hacerle perder el
        tiempo buscando donde no es."""
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch.dict("sys.modules", {"sigstore": mock.MagicMock()}):
            estado = estado_de_sigstore()
        self.assertFalse(estado["can_sign"])
        self.assertIn("identidad OIDC", estado["reason"])
        self.assertNotIn("pip install", estado["reason"])

    def test_con_identidad_pero_sin_biblioteca_manda_a_instalarla(self):
        with mock.patch.dict("os.environ", {"SIGSTORE_IDENTITY_TOKEN": "x"}, clear=True), \
             mock.patch.dict("sys.modules", {"sigstore": None}):
            estado = estado_de_sigstore()
        self.assertIn("pip install sigstore", estado["reason"])
        self.assertNotIn("identidad OIDC", estado["reason"])

    def test_con_las_dos_cosas_dice_que_SI_y_sin_motivo(self):
        with mock.patch.dict("os.environ", {"SIGSTORE_IDENTITY_TOKEN": "x"}, clear=True), \
             mock.patch.dict("sys.modules", {"sigstore": mock.MagicMock()}):
            estado = estado_de_sigstore()
        self.assertTrue(estado["can_sign"])
        self.assertIsNone(estado["reason"])


class NoDevuelveUnBundleAMediasTest(unittest.TestCase):
    def test_sin_nada_levanta_su_motivo(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch.dict("sys.modules", {"sigstore": None}):
            with self.assertRaises(SigstoreNoDisponible) as e:
                firmar_con_sigstore(b"lo que sea")
        self.assertIn("sigstore", str(e.exception))

    def test_si_la_firma_revienta_se_DICE_y_no_se_devuelve_nada(self):
        """«Una firma que no se hizo no se presenta como hecha»."""
        falso = mock.MagicMock()
        falso.SigningContext.production.side_effect = RuntimeError("sin red")
        with mock.patch.dict("os.environ", {"SIGSTORE_IDENTITY_TOKEN": "x"}, clear=True), \
             mock.patch.dict("sys.modules", {
                 "sigstore": mock.MagicMock(), "sigstore.sign": falso,
                 "sigstore.oidc": mock.MagicMock()}):
            with self.assertRaises(SigstoreNoDisponible) as e:
                firmar_con_sigstore(b"payload")
        self.assertIn("no pudo firmar", str(e.exception))
        self.assertIn("sin red", str(e.exception))


class SigstoreNoSubeElNivelTest(unittest.TestCase):
    """Lo más importante de este corte: los niveles hablan de lo que se
    COMPROBÓ, no de la fuerza de la firma. Si firmar mejor subiera el nivel,
    A2 dejaría de significar «hay evidencia reproducible»."""

    def _recibo(self, **cambios):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from test_c81_recibos import _recibo
        return _recibo(**cambios)

    def test_un_recibo_sin_evidencia_reproducible_sigue_siendo_A1(self):
        sobre = firmar_recibo(self._recibo(), clave=b"k", key_id="k1")
        self.assertEqual(nivel_del_recibo(sobre), "A1")

    def test_y_con_ella_es_A2_por_LA_EVIDENCIA_no_por_la_firma(self):
        sobre = firmar_recibo(
            self._recibo(evidence={"trace_root": "sha256:" + "4" * 64,
                                   "package_sha256": "a" * 64}),
            clave=b"k", key_id="k1")
        self.assertEqual(nivel_del_recibo(sobre), "A2")

    def test_el_modulo_lo_deja_ESCRITO(self):
        """Un límite que solo vive en la cabeza de quien lo escribió se pierde
        en la siguiente lectura."""
        from matrixai.pipelines import sigstore_firma
        self.assertIn("No sube el nivel del recibo", sigstore_firma.__doc__)


if __name__ == "__main__":
    unittest.main()
