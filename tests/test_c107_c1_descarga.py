# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — descarga verificada de los pesos de un proveedor de embeddings.

Del corte: «Los pesos no van en la imagen: proveedor descargable con licencia
aceptada (102-C5)». Lo que se prueba aquí es que descargar no es traerse un
fichero: es traerlo del sitio permitido, comprobar que es EXACTAMENTE el que
el catálogo fija, y no dejar nunca a medias algo que la siguiente ejecución
pueda dar por bueno.
"""
from __future__ import annotations

import hashlib
import io
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.text.embeddings.descarga import (
    HOSTS_PERMITIDOS,
    SUFIJOS_PERMITIDOS,
    DescargaError,
    FicheroDelProveedor,
    descargar_fichero,
    digest_de,
    estado_local,
    git_blob_sha1_de,
    host_permitido,
    sha256_de,
    url_de,
    verificar_fichero,
)

REV = "bf8b056651a2c21b8d2565580b8569da283cab23"


def _fichero(datos: bytes, ruta: str = "onnx/model.onnx", algoritmo: str = "sha256") -> FicheroDelProveedor:
    digest = (
        hashlib.sha256(datos).hexdigest()
        if algoritmo == "sha256"
        else hashlib.sha1(b"blob %d\0" % len(datos) + datos).hexdigest()
    )
    return FicheroDelProveedor(ruta=ruta, tamano_bytes=len(datos), digest=digest, algoritmo=algoritmo)


class _Respuesta(io.BytesIO):
    """Una respuesta HTTP de mentira, con `status` y `headers`."""

    def __init__(self, datos: bytes, status: int = 200, headers: dict | None = None) -> None:
        super().__init__(datos)
        self.status = status
        self.headers = headers or {}

    def getcode(self) -> int:
        return self.status


class TestAllowlistDeHosts(unittest.TestCase):
    """El allowlist compara el sufijo CON el punto delante a propósito.

    El comentario del módulo dice que por eso `evilhf.co` no cuela; sin esta
    prueba, esa afirmación no la sostiene nada y el día que alguien
    «simplifique» el `endswith` nadie se entera.
    """

    def test_los_hosts_exactos_pasan(self):
        for h in HOSTS_PERMITIDOS:
            self.assertTrue(host_permitido(h), h)

    def test_el_cdn_regional_pasa_porque_es_subdominio(self):
        # Medido el 2026-09-14: pedir un fichero a huggingface.co redirige a
        # este host. Una lista de nombres exactos habría caducado ahí.
        self.assertTrue(host_permitido("us.aws.cdn.hf.co"))
        self.assertTrue(host_permitido("cas-bridge.xethub.hf.co"))

    def test_un_dominio_que_solo_TERMINA_parecido_no_pasa(self):
        for h in ("evilhf.co", "nothuggingface.co", "hf.co.malo.com", "malohf.co", ""):
            self.assertFalse(host_permitido(h), h)

    def test_el_sufijo_lleva_punto(self):
        for s in SUFIJOS_PERMITIDOS:
            self.assertTrue(s.startswith("."), s)


class TestPorQueNoSeUsaSecureFetch(unittest.TestCase):
    """El módulo explica por qué NO reutiliza `matrixai.training.secure_fetch`:
    porque aquel acumula el cuerpo entero en memoria y aquí los ficheros son
    de cientos de MB. Una línea que explica por qué no hace lo obvio necesita
    una prueba con su nombre, o el siguiente que pase la «unifica»."""

    def test_el_cuerpo_se_escribe_a_disco_por_trozos_no_de_una_vez(self):
        datos = b"x" * (3 * (1 << 20) + 7)
        f = _fichero(datos)
        lecturas: list[int] = []

        class Contadora(_Respuesta):
            def read(self, n=-1):  # noqa: D102
                lecturas.append(n)
                return super().read(n)

        with TemporaryDirectory() as tmp:
            destino = Path(tmp) / "onnx" / "model.onnx"
            descargar_fichero("https://huggingface.co/x", destino, f, abridor=lambda req: Contadora(datos))
            self.assertTrue(destino.is_file())
        self.assertGreater(len(lecturas), 1, "se leyó de una vez: eso es justo lo que el módulo dice que no hace")
        self.assertTrue(all(n != -1 for n in lecturas), f"alguna lectura fue sin tope: {lecturas}")

    def test_secure_fetch_sigue_devolviendo_el_cuerpo_entero(self):
        # La otra mitad: la afirmación es sobre `secure_fetch`, así que si
        # `secure_fetch` cambiara y empezara a volcar a disco, este motivo
        # dejaría de ser cierto y habría que revisarlo.
        from matrixai.training import secure_fetch

        campos = secure_fetch.SecureFetchResult.__dataclass_fields__
        self.assertIn("body", campos)
        # `from __future__ import annotations` deja el tipo como texto.
        self.assertEqual(str(campos["body"].type).replace("'", ""), "bytes")


class TestRedireccionesValidadasAMano(unittest.TestCase):
    def test_una_redireccion_a_host_prohibido_se_corta(self):
        datos = b"payload"
        f = _fichero(datos)
        llamadas: list[str] = []

        def abridor(req):
            llamadas.append(req.full_url)
            raise urllib.error.HTTPError(
                req.full_url, 302, "Found", {"Location": "https://evil.example.com/x"}, None
            )

        with TemporaryDirectory() as tmp:
            with self.assertRaises(DescargaError) as ctx:
                descargar_fichero("https://huggingface.co/a", Path(tmp) / "a", f, abridor=abridor)
        self.assertIn("host no permitido", str(ctx.exception))
        self.assertEqual(len(llamadas), 1, "siguió el salto antes de mirarlo")

    def test_una_redireccion_al_cdn_permitido_se_sigue(self):
        datos = b"payload" * 100
        f = _fichero(datos)
        vistas: list[str] = []

        def abridor(req):
            vistas.append(req.full_url)
            if "huggingface.co" in req.full_url:
                raise urllib.error.HTTPError(
                    req.full_url, 302, "Found", {"Location": "https://us.aws.cdn.hf.co/blob"}, None
                )
            return _Respuesta(datos)

        with TemporaryDirectory() as tmp:
            destino = Path(tmp) / "m.onnx"
            descargar_fichero("https://huggingface.co/a", destino, f, abridor=abridor)
            self.assertEqual(destino.read_bytes(), datos)
        self.assertEqual(len(vistas), 2)

    def test_el_abridor_por_defecto_no_sigue_redirecciones_solo(self):
        """Si `urlopen` las siguiera, el allowlist por salto sería decorativo."""
        from matrixai.text.embeddings.descarga import _SinRedirecciones

        manejador = _SinRedirecciones()
        self.assertIsNone(manejador.redirect_request(None, None, 302, "Found", {}, "https://evil.example.com"))
        self.assertTrue(issubclass(_SinRedirecciones, urllib.request.HTTPRedirectHandler))


class TestVerificacionDeDigest(unittest.TestCase):
    def test_sha256_y_git_blob_sha1_son_los_que_publica_el_origen(self):
        datos = b"hola"
        with TemporaryDirectory() as tmp:
            r = Path(tmp) / "f"
            r.write_bytes(datos)
            self.assertEqual(sha256_de(r), hashlib.sha256(datos).hexdigest())
            self.assertEqual(git_blob_sha1_de(r), hashlib.sha1(b"blob 4\0hola").hexdigest())
            self.assertEqual(digest_de(r, "sha256"), sha256_de(r))
            self.assertEqual(digest_de(r, "git-blob-sha1"), git_blob_sha1_de(r))

    def test_un_fichero_cambiado_en_un_solo_bit_se_rechaza(self):
        datos = b"a" * 500
        f = _fichero(datos, ruta="vocab.txt", algoritmo="git-blob-sha1")
        with TemporaryDirectory() as tmp:
            r = Path(tmp) / "vocab.txt"
            r.write_bytes(datos)
            verificar_fichero(r, f)
            r.write_bytes(b"a" * 499 + b"b")
            with self.assertRaises(DescargaError) as ctx:
                verificar_fichero(r, f)
        self.assertIn("NO es el que fija el catálogo", str(ctx.exception))

    def test_un_fichero_a_medias_dice_que_esta_a_medias(self):
        datos = b"z" * 1000
        f = _fichero(datos)
        with TemporaryDirectory() as tmp:
            r = Path(tmp) / "m"
            r.write_bytes(datos[:400])
            with self.assertRaises(DescargaError) as ctx:
                verificar_fichero(r, f)
        self.assertIn("descarga incompleta", str(ctx.exception))

    def test_un_digest_que_no_cuadra_NO_deja_el_fichero_puesto(self):
        """Si quedara, la siguiente ejecución lo daría por bueno sin mirar."""
        datos = b"bueno" * 100
        f = _fichero(datos)
        with TemporaryDirectory() as tmp:
            destino = Path(tmp) / "m.onnx"
            with self.assertRaises(DescargaError):
                descargar_fichero(
                    "https://huggingface.co/a", destino, f, abridor=lambda req: _Respuesta(b"otra cosa")
                )
            self.assertFalse(destino.exists(), "quedó el fichero malo")
            self.assertFalse(list(Path(tmp).glob("*.parcial")), "quedó el parcial")

    def test_estado_local_dice_el_motivo_no_un_booleano(self):
        datos = b"q" * 300
        f = _fichero(datos)
        with TemporaryDirectory() as tmp:
            self.assertEqual(estado_local(Path(tmp), [f]), {"onnx/model.onnx": "falta"})
            (Path(tmp) / "onnx").mkdir()
            (Path(tmp) / "onnx" / "model.onnx").write_bytes(datos)
            self.assertEqual(estado_local(Path(tmp), [f]), {"onnx/model.onnx": "ok"})


class TestLaRevisionEsUnCheckpointConcreto(unittest.TestCase):
    def test_una_rama_no_vale_como_revision(self):
        for mala in ("main", "bf8b056", "", "BF8B056651A2C21B8D2565580B8569DA283CAB23", "z" * 40):
            with self.assertRaises(DescargaError, msg=mala):
                url_de("minishlab/potion-base-8M", mala, "onnx/model.onnx")

    def test_la_url_lleva_la_revision_no_main(self):
        u = url_de("minishlab/potion-base-8M", REV, "onnx/model.onnx")
        self.assertIn(REV, u)
        self.assertNotIn("/main/", u)
        self.assertTrue(u.startswith("https://huggingface.co/"))


class TestUnFicheroSinDigestNoSeDescarga(unittest.TestCase):
    def test_sin_digest_no_se_construye(self):
        with self.assertRaises(ValueError):
            FicheroDelProveedor(ruta="a", tamano_bytes=1, digest="", algoritmo="sha256")

    def test_algoritmo_desconocido_no_se_construye(self):
        with self.assertRaises(ValueError):
            FicheroDelProveedor(ruta="a", tamano_bytes=1, digest="ab", algoritmo="md5")


if __name__ == "__main__":
    unittest.main()
