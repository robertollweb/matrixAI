# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `stooq`.

Stooq (https://stooq.com): histórico de cotizaciones (acciones, índices)
en CSV directo, sin clave. Licencia verificada (ver memoria del
proyecto): uso personal/educativo, nunca comercial.

**Riesgo externo verificado 2026-07-17, no un bug de este código**:
`stooq.com` (y `stooq.pl`) gatean HOY toda petición no-navegador tras un
reto JS de prueba-de-trabajo ("This site requires JavaScript to verify
your browser") — ni siquiera con un User-Agent de navegador responden con
el CSV. Un fetch de servidor (`urllib`, sin motor JS) NUNCA puede pasar
ese reto. El código de este módulo es correcto y su comprobación de
esquema (`_EXPECTED_HEADER`) YA detecta y rechaza esa página HTML como
"esquema inesperado" — el fallo queda limpio (invariante 7), no en
silencio. Pendiente de decisión de Roberto (documentado, no resuelto
aquí): sustituir la fuente bursátil o aceptar que `stooq` no funciona en
la práctica hasta que cambien su política anti-bot.
"""
from __future__ import annotations

import csv
import io
import urllib.parse
from datetime import date, datetime, timezone
from typing import Any

from matrixai.training.data_provider import (
    DataProviderError,
    DownloadEstimate,
    DownloadResult,
    LicenseInfo,
    ProviderMetadata,
    require_license_accepted,
)
from matrixai.training.secure_fetch import SecureFetchError, secure_fetch

_HOST = "stooq.com"
_ALLOWED_HOSTS = frozenset({_HOST})
_EXPECTED_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume"]
_VALID_INTERVALS = frozenset({"d", "w", "m"})
_MAX_BYTES = 20_000_000
_TIMEOUT = 20.0


class StooqProvider:
    provider_id = "stooq"

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_id=self.provider_id,
            display_name="Stooq",
            description="Histórico de cotizaciones bursátiles públicas por símbolo.",
            requires_network=True,
        )

    def get_license_info(self) -> LicenseInfo:
        return LicenseInfo(
            name="Stooq (uso personal/educativo)",
            url="https://stooq.com/",
            summary="Datos gratuitos para uso personal y educativo — no para uso comercial.",
            requires_attribution=False,
            commercial_use_allowed=False,
        )

    def check_availability(self) -> bool:
        try:
            secure_fetch(
                self._build_url({
                    "symbol": "aapl.us", "start_date": "2024-01-02",
                    "end_date": "2024-01-03", "interval": "d",
                }),
                allowed_hosts=_ALLOWED_HOSTS, timeout=_TIMEOUT, max_bytes=_MAX_BYTES,
            )
            return True
        except SecureFetchError:
            return False

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        symbol = config.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            errors.append("symbol debe ser un texto no vacío (p.ej. 'aapl.us').")
        interval = config.get("interval", "d")
        if interval not in _VALID_INTERVALS:
            errors.append(f"interval debe ser uno de {sorted(_VALID_INTERVALS)} (recibido {interval!r}).")
        parsed_start = self._parse_date(config.get("start_date"), "start_date", errors)
        parsed_end = self._parse_date(config.get("end_date"), "end_date", errors)
        if parsed_start and parsed_end and parsed_start > parsed_end:
            errors.append("start_date debe ser anterior o igual a end_date.")
        return errors

    def _parse_date(self, value: Any, field: str, errors: list[str]) -> date | None:
        if not isinstance(value, str):
            errors.append(f"{field} debe ser una fecha 'YYYY-MM-DD'.")
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            errors.append(f"{field} {value!r} no es una fecha ISO válida.")
            return None

    def estimate_download(self, config: dict[str, Any]) -> DownloadEstimate:
        errors = self.validate_config(config)
        if errors:
            raise DataProviderError("Config inválida: " + "; ".join(errors))
        start = date.fromisoformat(config["start_date"])
        end = date.fromisoformat(config["end_date"])
        days = (end - start).days + 1
        return DownloadEstimate(
            estimated_rows=days, estimated_bytes=days * 40,
            notes=(
                f"{days} día(s) naturales — estimación aproximada; fines de "
                "semana y festivos sin cotización se excluyen al descargar."
            ),
        )

    def download(self, config: dict[str, Any], *, license_accepted: bool) -> DownloadResult:
        require_license_accepted(license_accepted, self.provider_id)
        errors = self.validate_config(config)
        if errors:
            raise DataProviderError("Config inválida: " + "; ".join(errors))

        url = self._build_url(config)
        try:
            fetched = secure_fetch(url, allowed_hosts=_ALLOWED_HOSTS, timeout=_TIMEOUT, max_bytes=_MAX_BYTES)
        except SecureFetchError as exc:
            raise DataProviderError(f"Stooq: {exc}") from exc

        csv_text, columns, n_rows = self._to_canonical_csv(fetched.body, config)
        return DownloadResult(
            csv_text=csv_text, rows=n_rows, columns=columns,
            source_url=fetched.url, fetched_at=_utcnow_iso(),
            license_info=self.get_license_info(),
            provenance_extra={"symbol": config["symbol"], "interval": config.get("interval", "d")},
        )

    def _build_url(self, config: dict[str, Any]) -> str:
        params = {
            "s": config["symbol"],
            "d1": config["start_date"].replace("-", ""),
            "d2": config["end_date"].replace("-", ""),
            "i": config.get("interval", "d"),
        }
        return f"https://{_HOST}/q/d/l/?{urllib.parse.urlencode(params)}"

    def _to_canonical_csv(self, body: bytes, config: dict[str, Any]) -> tuple[str, list[str], int]:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataProviderError(f"Stooq devolvió una respuesta no-UTF8: {exc}") from exc
        rows = list(csv.reader(io.StringIO(text)))
        if not rows or rows[0] != _EXPECTED_HEADER:
            # Cubre tanto un símbolo inexistente (Stooq responde "No data")
            # como el reto anti-bot JS (ver docstring del módulo) — ambos
            # son "esquema inesperado", nunca se intenta adivinar el CSV.
            raise DataProviderError(
                f"Stooq devolvió una respuesta con forma inesperada para "
                f"{config['symbol']!r}: {text[:200]!r}"
            )
        data_rows = rows[1:]
        if not data_rows:
            raise DataProviderError(
                f"Stooq no devolvió ninguna cotización para {config['symbol']!r} en ese rango."
            )
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(_EXPECTED_HEADER)
        writer.writerows(data_rows)
        return buf.getvalue(), list(_EXPECTED_HEADER), len(data_rows)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
