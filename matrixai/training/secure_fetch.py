# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — fetch seguro compartido.

Punto ÚNICO de salida a la red para cualquier `DataProvider` (invariante
7: fallo externo limpio; auditoría de seguridad completa en C8). Reglas
fijas, no configurables por quien llama:

  - Solo HTTPS — nunca http/file/ftp/data (cierra esquemas que una config
    de usuario pudiera colar).
  - El host de la URL debe estar en `allowed_hosts`, un allowlist FIJO
    por proveedor — la config del usuario (parámetros de la descarga)
    solo puede influir en la query string, NUNCA en el host: así un
    proveedor nunca puede convertirse en un vector de SSRF por muy
    maliciosa que sea la config.
  - Redirecciones seguidas MANUALMENTE (nunca automáticas vía
    `HTTPRedirectHandler`): cada salto se valida contra el MISMO
    allowlist antes de seguirlo; se corta a `max_redirects`.
  - Reauditoría C8 [MEDIA]: el allowlist de arriba solo compara el
    NOMBRE textual del host — no bastaba contra DNS rebinding (un host
    permitido que en el momento de conectar resuelve a una IP interna).
    `_PinnedHTTPSConnection` resuelve DNS y valida que TODAS las IPs
    candidatas sean públicas ANTES de conectar, y conecta a la MISMA
    dirección ya validada (nunca una segunda resolución en el connect()
    real de `http.client`, que sería la ventana TOCTOU) — el SNI/Host
    sigue siendo el nombre original, así que la validación del
    certificado TLS no se ve afectada.
  - El cuerpo se lee en TROZOS (nunca `.read()` de una vez) — se corta en
    cuanto se supera `max_bytes`, sin cargar el exceso en memoria.
  - Sin credenciales en v1 — ningún header de autenticación se admite
    aquí; los proveedores v1 son APIs públicas sin clave.

Reutiliza el patrón de "transport" inyectable ya usado en
`matrixai/agents/llm_proposal.py` (`_urlopen_json_transport`): el
`opener` es sustituible por uno mockeado en tests — cero red real en CI.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class SecureFetchError(Exception):
    """Cualquier fallo de fetch seguro — mensaje siempre accionable."""


@dataclass(frozen=True)
class SecureFetchResult:
    url: str
    status: int
    body: bytes
    content_type: str | None


class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
    """Bloquea el seguimiento automático de redirecciones — `secure_fetch`
    las sigue a mano, validando cada salto contra el allowlist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D102
        return None


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """IPs internas/no-públicas — loopback, RFC1918/enlace-local, metadata
    de nube (169.254.169.254 cae en `is_link_local`), multicast, reservado,
    sin especificar. Rechazo conservador: cualquiera de estos categoriza
    la IP como "no pública" (no intenta distinguir casos legítimos)."""
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def _resolve_and_validate(host: str, port: int) -> tuple[int, tuple]:
    """Resuelve DNS UNA sola vez y valida que TODAS las direcciones
    candidatas sean públicas — si CUALQUIERA no lo es, se rechaza el host
    entero (un atacante con control parcial de DNS podría intercalar una
    IP pública "señuelo" junto a una interna real). Devuelve la familia +
    sockaddr de la PRIMERA candidata para que `_PinnedHTTPSConnection`
    conecte exactamente a esa dirección — nunca a una resuelta de nuevo."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecureFetchError(f"No se pudo resolver el host {host!r}: {exc}") from exc
    if not infos:
        raise SecureFetchError(f"El host {host!r} no resolvió a ninguna dirección.")
    for family, _socktype, _proto, _canonname, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_disallowed_ip(ip):
            raise SecureFetchError(
                f"El host {host!r} resuelve a una IP no pública ({sockaddr[0]}) "
                "— rechazado (protección SSRF / DNS rebinding)."
            )
    family, _socktype, _proto, _canonname, sockaddr = infos[0]
    return family, sockaddr


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """`http.client.HTTPSConnection.connect()` de base resuelve DNS DE
    NUEVO al conectar — la ventana TOCTOU exacta de un ataque DNS
    rebinding contra un host allowlisted (válido en el primer chequeo,
    reapuntado a una IP interna en el segundo). Esta subclase resuelve y
    valida UNA vez (`_resolve_and_validate`) y conecta a ESA dirección
    exacta; el SNI (`server_hostname=self.host`) sigue siendo el nombre
    original, así que la verificación del certificado TLS es idéntica a
    la de una conexión normal."""

    def connect(self) -> None:
        family, sockaddr = _resolve_and_validate(self.host, self.port)
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(sockaddr)
        except OSError:
            sock.close()
            raise
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """Mismo `HTTPSHandler` de siempre, pero usando `_PinnedHTTPSConnection`
    en vez de `http.client.HTTPSConnection` — la única diferencia es CÓMO
    se resuelve/conecta, la verificación de certificado (`self._context`)
    es exactamente la misma."""

    def https_open(self, req):  # noqa: ANN001, D102
        return self.do_open(_PinnedHTTPSConnection, req, context=self._context)


_DEFAULT_OPENER = urllib.request.build_opener(_NoAutoRedirect, _PinnedHTTPSHandler)
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


def secure_fetch(
    url: str,
    *,
    allowed_hosts: frozenset[str],
    timeout: float = 15.0,
    max_bytes: int = 20_000_000,
    max_redirects: int = 5,
    chunk_size: int = 65_536,
    opener: Any = None,
) -> SecureFetchResult:
    """GET seguro de `url`. Lanza `SecureFetchError` (nunca deja un estado
    a medias — invariante 7) ante: esquema no https, host fuera de
    `allowed_hosts` (en la URL inicial O en cualquier redirección),
    demasiadas redirecciones, timeout, error HTTP no-3xx, o respuesta que
    supera `max_bytes`."""
    opener = opener or _DEFAULT_OPENER
    current_url = url
    for _ in range(max_redirects + 1):
        _validate_url(current_url, allowed_hosts)
        try:
            response = opener.open(current_url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in _REDIRECT_CODES:
                location = exc.headers.get("Location") if exc.headers else None
                if not location:
                    raise SecureFetchError(
                        f"Redirección {exc.code} sin cabecera Location en {current_url!r}."
                    ) from exc
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            raise SecureFetchError(
                f"HTTP {exc.code} al pedir {current_url!r}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise SecureFetchError(
                f"Tiempo de espera agotado pidiendo {current_url!r} (timeout={timeout}s)."
            ) from exc
        except urllib.error.URLError as exc:
            raise SecureFetchError(
                f"No se pudo contactar {current_url!r}: {exc.reason}"
            ) from exc

        try:
            with response:
                body = _read_capped(response, max_bytes, chunk_size, current_url)
                status = getattr(response, "status", 200) or 200
                headers = getattr(response, "headers", None)
                content_type = headers.get("Content-Type") if headers else None
                return SecureFetchResult(
                    url=current_url, status=status, body=body, content_type=content_type,
                )
        except TimeoutError as exc:
            raise SecureFetchError(
                f"Tiempo de espera agotado leyendo {current_url!r} (timeout={timeout}s)."
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            # Auditoría 2026-07-17 [MEDIA]: solo TimeoutError estaba envuelto
            # aquí — un ConnectionResetError/IncompleteRead/otro OSError de
            # bajo nivel durante la LECTURA (no en el open()) se escapaba sin
            # traducir y podía terminar como un 500 genérico en el caller,
            # en vez de un fallo limpio y accionable (invariante 7).
            # TimeoutError es un OSError desde Python 3.10 pero se atrapa
            # antes con su propio mensaje más específico.
            raise SecureFetchError(
                f"Fallo de red leyendo {current_url!r}: {exc}"
            ) from exc

    raise SecureFetchError(f"Demasiadas redirecciones (> {max_redirects}) partiendo de {url!r}.")


def _validate_url(url: str, allowed_hosts: frozenset[str]) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise SecureFetchError(f"Esquema no permitido {parts.scheme!r} en {url!r} — solo https.")
    host = (parts.hostname or "").lower()
    if host not in allowed_hosts:
        raise SecureFetchError(
            f"Host no permitido {host!r} en {url!r} — fuera del allowlist del proveedor."
        )


def _read_capped(response: Any, max_bytes: int, chunk_size: int, url: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise SecureFetchError(
                f"Respuesta de {url!r} supera el límite de {max_bytes} bytes — descarga cortada."
            )
        chunks.append(chunk)
    return b"".join(chunks)
