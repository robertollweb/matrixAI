# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — fetch seguro compartido.

Cero red real: un `_FakeOpener` sustituye `urllib.request.build_opener`
por completo (mismo patrón de "transport" inyectable que
`test_llm_multiprovider.py` usa para `ChatCompletionsLLMProposalProvider`).
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from matrixai.training.secure_fetch import SecureFetchError, secure_fetch


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


if __name__ == "__main__":
    pytest.main([__file__])
