# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — fetch seguro compartido.

Cero red real: un `_FakeOpener` sustituye `urllib.request.build_opener`
por completo (mismo patrón de "transport" inyectable que
`test_llm_multiprovider.py` usa para `ChatCompletionsLLMProposalProvider`).
"""
from __future__ import annotations

import io
import socket
import unittest.mock
import urllib.error

import pytest

from matrixai.training.secure_fetch import (
    SecureFetchError,
    _PinnedHTTPSConnection,
    _resolve_and_validate,
    secure_fetch,
)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/csv"):
        self._buf = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    """`responses` es una lista de callables `url -> _FakeResponse` o
    excepciones a lanzar, consumida en orden — un valor por llamada a
    `.open()`, para poder simular una cadena de redirecciones."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.requested_urls: list[str] = []

    def open(self, url: str, *, timeout: float):
        self.requested_urls.append(url)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


_ALLOWED = frozenset({"api.example.com"})


class TestHappyPath:
    def test_returns_body_and_metadata(self):
        opener = _FakeOpener([_FakeResponse(b"a,b\n1,2\n")])
        result = secure_fetch("https://api.example.com/data.csv", allowed_hosts=_ALLOWED, opener=opener)
        assert result.body == b"a,b\n1,2\n"
        assert result.status == 200
        assert result.content_type == "text/csv"
        assert result.url == "https://api.example.com/data.csv"


class TestSchemeAndHostAllowlist:
    def test_rejects_non_https_scheme(self):
        opener = _FakeOpener([])
        with pytest.raises(SecureFetchError, match="solo https"):
            secure_fetch("http://api.example.com/data.csv", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == []

    def test_rejects_host_outside_allowlist(self):
        opener = _FakeOpener([])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://evil.example.com/data.csv", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == []

    def test_rejects_file_scheme(self):
        """SSRF vía esquema local — cero llamadas al opener."""
        opener = _FakeOpener([])
        with pytest.raises(SecureFetchError, match="solo https"):
            secure_fetch("file:///etc/passwd", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == []


class TestOversize:
    def test_cuts_the_stream_and_fails_clean_without_buffering_everything(self):
        big_body = b"x" * 1000
        opener = _FakeOpener([_FakeResponse(big_body)])
        with pytest.raises(SecureFetchError, match="supera el límite"):
            secure_fetch(
                "https://api.example.com/big.csv", allowed_hosts=_ALLOWED,
                opener=opener, max_bytes=100, chunk_size=32,
            )

    def test_exactly_at_the_limit_succeeds(self):
        body = b"x" * 100
        opener = _FakeOpener([_FakeResponse(body)])
        result = secure_fetch(
            "https://api.example.com/exact.csv", allowed_hosts=_ALLOWED,
            opener=opener, max_bytes=100, chunk_size=32,
        )
        assert len(result.body) == 100


class TestTimeout:
    def test_timeout_during_connect_raises_actionable_error(self):
        opener = _FakeOpener([TimeoutError("timed out")])
        with pytest.raises(SecureFetchError, match="Tiempo de espera agotado"):
            secure_fetch("https://api.example.com/slow.csv", allowed_hosts=_ALLOWED, opener=opener, timeout=1.0)

    def test_timeout_during_read_raises_actionable_error(self):
        class _TimeoutOnRead:
            status = 200
            headers = {"Content-Type": "text/csv"}

            def read(self, n: int = -1) -> bytes:
                raise TimeoutError("read timed out")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        opener = _FakeOpener([_TimeoutOnRead()])
        with pytest.raises(SecureFetchError, match="Tiempo de espera agotado leyendo"):
            secure_fetch("https://api.example.com/slow.csv", allowed_hosts=_ALLOWED, opener=opener)


class TestReadFailures:
    def test_connection_reset_during_read_is_wrapped(self):
        """Auditoría 2026-07-17 (ronda 2) [MEDIA]: solo TimeoutError estaba
        envuelto durante la lectura — un ConnectionResetError (u otro
        OSError/http.client.HTTPException) se escapaba sin traducir."""
        class _ResetOnRead:
            status = 200
            headers = {"Content-Type": "text/csv"}

            def read(self, n: int = -1) -> bytes:
                raise ConnectionResetError("peer reset")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        opener = _FakeOpener([_ResetOnRead()])
        with pytest.raises(SecureFetchError, match="Fallo de red leyendo"):
            secure_fetch("https://api.example.com/x", allowed_hosts=_ALLOWED, opener=opener)

    def test_incomplete_read_during_read_is_wrapped(self):
        import http.client

        class _IncompleteOnRead:
            status = 200
            headers = {"Content-Type": "text/csv"}

            def read(self, n: int = -1) -> bytes:
                raise http.client.IncompleteRead(b"partial")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        opener = _FakeOpener([_IncompleteOnRead()])
        with pytest.raises(SecureFetchError, match="Fallo de red leyendo"):
            secure_fetch("https://api.example.com/x", allowed_hosts=_ALLOWED, opener=opener)


class TestHttpErrors:
    def test_non_redirect_http_error_is_actionable(self):
        exc = urllib.error.HTTPError("https://api.example.com/x", 404, "Not Found", {}, None)
        opener = _FakeOpener([exc])
        with pytest.raises(SecureFetchError, match="HTTP 404"):
            secure_fetch("https://api.example.com/x", allowed_hosts=_ALLOWED, opener=opener)

    def test_connection_failure_is_actionable(self):
        exc = urllib.error.URLError("connection refused")
        opener = _FakeOpener([exc])
        with pytest.raises(SecureFetchError, match="No se pudo contactar"):
            secure_fetch("https://api.example.com/x", allowed_hosts=_ALLOWED, opener=opener)


class TestRedirects:
    def test_follows_a_redirect_within_the_allowlist(self):
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found",
            {"Location": "https://api.example.com/new"}, None,
        )
        opener = _FakeOpener([redirect, _FakeResponse(b"ok")])
        result = secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)
        assert result.body == b"ok"
        assert result.url == "https://api.example.com/new"
        assert opener.requested_urls == ["https://api.example.com/old", "https://api.example.com/new"]

    def test_rejects_a_redirect_outside_the_allowlist(self):
        """El caso central de la auditoría de seguridad: un proveedor
        legítimo no puede convertirse en un vector SSRF vía una
        redirección a un host distinto."""
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found",
            {"Location": "https://evil.example.com/steal"}, None,
        )
        opener = _FakeOpener([redirect])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)
        # Nunca se llega a pedir la URL maliciosa.
        assert opener.requested_urls == ["https://api.example.com/old"]

    def test_rejects_a_redirect_to_a_non_https_scheme(self):
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found",
            {"Location": "http://api.example.com/downgrade"}, None,
        )
        opener = _FakeOpener([redirect])
        with pytest.raises(SecureFetchError, match="solo https"):
            secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)

    def test_redirect_without_location_header_is_actionable(self):
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found", {}, None,
        )
        opener = _FakeOpener([redirect])
        with pytest.raises(SecureFetchError, match="sin cabecera Location"):
            secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)

    def test_too_many_redirects_fails_clean(self):
        def _redirect(n: int):
            return urllib.error.HTTPError(
                f"https://api.example.com/{n}", 302, "Found",
                {"Location": f"https://api.example.com/{n + 1}"}, None,
            )
        opener = _FakeOpener([_redirect(i) for i in range(10)])
        with pytest.raises(SecureFetchError, match="Demasiadas redirecciones"):
            secure_fetch("https://api.example.com/0", allowed_hosts=_ALLOWED, opener=opener, max_redirects=3)


class TestSSRFInternalIPsAsLiteralHost:
    """Auditoría C8 (ronda 1) — un literal IP usado DIRECTAMENTE como host
    en la URL (o en un redirect) nunca está en `allowed_hosts` (que solo
    contiene NOMBRES de dominio), así que cae en `Host no permitido`.
    Reauditoría C8 (ronda 2) [MEDIA]: esto NO cubre DNS rebinding — un
    HOST PERMITIDO (`archive-api.open-meteo.com`) que resuelve a una IP
    interna en el momento de conectar pasaba `_validate_url` limpio,
    porque esa función solo compara texto, nunca resuelve DNS. Ver
    `TestDNSRebindingProtection` más abajo para la protección real
    (resolver + validar la IP antes de conectar)."""

    def test_rejects_a_direct_request_to_a_loopback_ip(self):
        opener = _FakeOpener([])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://127.0.0.1/admin", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == []

    def test_rejects_a_direct_request_to_the_cloud_metadata_ip(self):
        """169.254.169.254 — el objetivo clásico de SSRF contra metadata
        de instancia en nube (AWS/GCP/Azure)."""
        opener = _FakeOpener([])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://169.254.169.254/latest/meta-data/", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == []

    def test_rejects_a_direct_request_to_a_private_range_ip(self):
        opener = _FakeOpener([])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://10.0.0.5/internal", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == []

    def test_rejects_a_redirect_to_a_loopback_ip(self):
        """El caso realmente peligroso: un proveedor externo confiable
        redirige (por config maliciosa, compromiso, o bug) hacia una IP
        interna en vez de a otro dominio — misma defensa que un host
        de dominio ajeno, verificada explícitamente con una IP."""
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found",
            {"Location": "https://127.0.0.1:8080/steal"}, None,
        )
        opener = _FakeOpener([redirect])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == ["https://api.example.com/old"]

    def test_rejects_a_redirect_to_the_cloud_metadata_ip(self):
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found",
            {"Location": "https://169.254.169.254/latest/meta-data/iam/"}, None,
        )
        opener = _FakeOpener([redirect])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == ["https://api.example.com/old"]

    def test_rejects_a_redirect_to_ipv6_loopback(self):
        redirect = urllib.error.HTTPError(
            "https://api.example.com/old", 302, "Found",
            {"Location": "https://[::1]/steal"}, None,
        )
        opener = _FakeOpener([redirect])
        with pytest.raises(SecureFetchError, match="Host no permitido"):
            secure_fetch("https://api.example.com/old", allowed_hosts=_ALLOWED, opener=opener)
        assert opener.requested_urls == ["https://api.example.com/old"]


class TestDNSRebindingProtection:
    """Reauditoría C8 (ronda 2) [MEDIA]: `_validate_url` solo compara el
    NOMBRE textual del host contra el allowlist — un host PERMITIDO que
    resuelve (ahora, o tras un cambio de DNS) a una IP interna pasaba
    limpio, porque nada resolvía DNS antes de conectar. La protección
    real vive en `_resolve_and_validate` (usada por `_PinnedHTTPSConnection.
    connect()`, la ÚNICA conexión real que hace el opener por defecto) —
    resuelve UNA vez, valida TODAS las IPs candidatas, y conecta a la
    MISMA dirección ya validada, sin una segunda resolución (que sería la
    ventana TOCTOU de un ataque de DNS rebinding)."""

    def test_resolve_and_validate_rejects_a_private_ip(self):
        with unittest.mock.patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443)),
        ]):
            with pytest.raises(SecureFetchError, match="no pública"):
                _resolve_and_validate("archive-api.open-meteo.com", 443)

    def test_resolve_and_validate_rejects_loopback(self):
        with unittest.mock.patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]):
            with pytest.raises(SecureFetchError, match="no pública"):
                _resolve_and_validate("archive-api.open-meteo.com", 443)

    def test_resolve_and_validate_rejects_the_cloud_metadata_ip(self):
        with unittest.mock.patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443)),
        ]):
            with pytest.raises(SecureFetchError, match="no pública"):
                _resolve_and_validate("archive-api.open-meteo.com", 443)

    def test_resolve_and_validate_rejects_ipv6_loopback(self):
        with unittest.mock.patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
        ]):
            with pytest.raises(SecureFetchError, match="no pública"):
                _resolve_and_validate("archive-api.open-meteo.com", 443)

    def test_resolve_and_validate_rejects_if_any_candidate_is_private(self):
        """Un atacante con control PARCIAL de DNS podría intercalar una IP
        pública "señuelo" junto a la interna real — se rechaza si
        CUALQUIERA de las candidatas no es pública, no solo la primera."""
        with unittest.mock.patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443)),
        ]):
            with pytest.raises(SecureFetchError, match="no pública"):
                _resolve_and_validate("archive-api.open-meteo.com", 443)

    def test_resolve_and_validate_accepts_a_public_ip(self):
        family, sockaddr = None, None
        with unittest.mock.patch("socket.getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        ]):
            family, sockaddr = _resolve_and_validate("archive-api.open-meteo.com", 443)
        assert sockaddr == ("8.8.8.8", 443)

    def test_resolve_and_validate_wraps_a_dns_failure(self):
        with unittest.mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("no address")):
            with pytest.raises(SecureFetchError, match="No se pudo resolver"):
                _resolve_and_validate("archive-api.open-meteo.com", 443)

    def test_pinned_connection_connects_to_the_pre_validated_address_only(self):
        """Verifica que `_PinnedHTTPSConnection.connect()` conecta al
        SOCKADDR devuelto por `_resolve_and_validate` — NUNCA vuelve a
        resolver DNS por su cuenta (que sería la ventana TOCTOU real)."""
        conn = _PinnedHTTPSConnection("archive-api.open-meteo.com", 443, timeout=5.0)
        fake_sock = unittest.mock.MagicMock()
        fake_wrapped = unittest.mock.MagicMock()
        conn._context = unittest.mock.MagicMock()
        conn._context.wrap_socket.return_value = fake_wrapped

        with unittest.mock.patch(
            "matrixai.training.secure_fetch._resolve_and_validate",
            return_value=(socket.AF_INET, ("203.0.113.5", 443)),
        ) as mock_resolve, unittest.mock.patch("socket.socket", return_value=fake_sock):
            conn.connect()

        mock_resolve.assert_called_once_with("archive-api.open-meteo.com", 443)
        fake_sock.connect.assert_called_once_with(("203.0.113.5", 443))
        conn._context.wrap_socket.assert_called_once_with(
            fake_sock, server_hostname="archive-api.open-meteo.com",
        )
        assert conn.sock is fake_wrapped

    def test_pinned_connection_closes_the_socket_on_connect_failure(self):
        conn = _PinnedHTTPSConnection("archive-api.open-meteo.com", 443, timeout=5.0)
        fake_sock = unittest.mock.MagicMock()
        fake_sock.connect.side_effect = OSError("connection refused")

        with unittest.mock.patch(
            "matrixai.training.secure_fetch._resolve_and_validate",
            return_value=(socket.AF_INET, ("203.0.113.5", 443)),
        ), unittest.mock.patch("socket.socket", return_value=fake_sock):
            with pytest.raises(OSError):
                conn.connect()
        fake_sock.close.assert_called_once()


class TestRateLimit:
    """Auditoría C8 — matriz de errores §29, "límite de peticiones"
    (rate limit). Comparte el camino genérico de `HTTPError` no-3xx
    (ya cubierto por `test_non_redirect_http_error_is_actionable`), pero
    se nombra explícitamente para que la categoría quede verificada."""

    def test_http_429_is_actionable(self):
        exc = urllib.error.HTTPError("https://api.example.com/x", 429, "Too Many Requests", {}, None)
        opener = _FakeOpener([exc])
        with pytest.raises(SecureFetchError, match="HTTP 429"):
            secure_fetch("https://api.example.com/x", allowed_hosts=_ALLOWED, opener=opener)


class TestNoCredentials:
    """Auditoría C8: "credenciales incorrectas" de la matriz §29 es N/A
    para v1 — los 3 proveedores son APIs públicas sin clave (docstring
    del módulo). Este test lo hace verificable: `secure_fetch` nunca
    añade cabeceras de autenticación por su cuenta — si algún día un
    proveedor CON credenciales se añadiera, tendría que pasarlas él
    mismo (fuera de este módulo), nunca inyectadas aquí en silencio."""

    def test_never_sends_an_authorization_header(self):
        captured_headers: dict = {}

        class _CapturingOpener:
            def open(self_inner, url, *, timeout):
                captured_headers["url"] = url
                return _FakeResponse(b"ok")

        secure_fetch("https://api.example.com/x", allowed_hosts=_ALLOWED, opener=_CapturingOpener())
        # `secure_fetch` llama a `opener.open(url, timeout=...)` — nunca
        # construye una `Request` con cabeceras propias, así que no hay
        # ningún camino por el que una credencial pueda colarse aquí.
        assert captured_headers["url"] == "https://api.example.com/x"


if __name__ == "__main__":
    pytest.main([__file__])
