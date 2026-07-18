# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `stooq` (**NO
REGISTRADO por defecto** — ver `data_provider.get_default_registry`).

Stooq (https://stooq.com): histórico de cotizaciones (acciones, índices)
en CSV directo, sin clave. Licencia verificada (ver memoria del
proyecto): uso personal/educativo, nunca comercial.

**Riesgo externo verificado 2026-07-17, no un bug de este código**:
`stooq.com` (y `stooq.pl`) gatean HOY toda petición no-navegador tras un
reto JS de prueba-de-trabajo ("This site requires JavaScript to verify
your browser") — ni siquiera con un User-Agent de navegador responden con
el CSV. Un fetch de servidor (`urllib`, sin motor JS) NUNCA puede pasar
ese reto (confirmado también con `curl` real, no solo con este código).
Reintentado deliberadamente NO ejecutando un motor JS/Playwright/
Selenium para resolver el reto — ampliaría la superficie de ataque,
consumiría recursos y podría romperse sin aviso, para un problema que no
es "fetch inseguro" sino "el proveedor no quiere tráfico de servidor".

Reauditoría 2026-07-17 (ronda 2) [ALTA]: por eso `stooq` se RETIRA del
registro por defecto (`get_default_registry()` ya no lo incluye — ver
`provider_ecb_fx.py` como sustituto de fuente de series temporales
financieras). La clase se conserva aquí, completa y con sus propios
bugs corregidos (validación de fila a fila, `check_availability` que
valida la FORMA del cuerpo, no solo la ausencia de excepción), por si
Roberto decide reactivarla si Stooq cambia su política anti-bot — pero
NADIE debe registrarla en producción mientras el reto siga bloqueando
toda petición de servidor.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime, timezone
import urllib.parse
from typing import Any

from matrixai.training.data_provider import (
    DataProviderError,
    DownloadEstimate,
    DownloadResult,
    LicenseAcceptance,
    LicenseInfo,
    ProviderMetadata,
    require_valid_acceptance,
)
from matrixai.training.secure_fetch import SecureFetchError, secure_fetch

_HOST = "stooq.com"
_ALLOWED_HOSTS = frozenset({_HOST})
_EXPECTED_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume"]
_NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
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
            summary_i18n={
                "en": "Free data for personal and educational use — not for commercial use.",
            },
        )

    def check_availability(self) -> bool:
        # Auditoría 2026-07-17 (ronda 2) [ALTA]: comprobar solo que
        # `secure_fetch` no lance NO basta — el reto anti-bot responde
        # HTTP 200 con HTML (verificado con curl real), así que "sin
        # excepción" declaraba `True` incluso viendo esa página. Se valida
        # la FORMA del cuerpo con el mismo camino que usa `download`.
        probe_config = {"symbol": "aapl.us", "start_date": "2024-01-02", "end_date": "2024-01-03"}
        try:
            fetched = secure_fetch(
                self._build_url({**probe_config, "interval": "d"}),
                allowed_hosts=_ALLOWED_HOSTS, timeout=_TIMEOUT, max_bytes=_MAX_BYTES,
            )
            self._to_canonical_csv(fetched.body, probe_config)
            return True
        except (SecureFetchError, DataProviderError):
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
            notes_i18n={
                "en": (
                    f"{days} calendar day(s) — approximate estimate; weekends and "
                    "market holidays are excluded when downloading."
                ),
            },
        )

    def download(self, config: dict[str, Any], *, license_acceptance: LicenseAcceptance | None) -> DownloadResult:
        require_valid_acceptance(license_acceptance, self)
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
            provenance_extra={
                "symbol": config["symbol"], "interval": config.get("interval", "d"),
                "license_acceptance": license_acceptance.to_dict(),
            },
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

        # Auditoría 2026-07-17 (ronda 2) [MEDIA]: solo se validaba la
        # CABECERA — una fila con menos campos, fecha ilegible, un valor
        # OHLCV no numérico o vacío pasaba tal cual al CSV "canónico"
        # entregado a C1, que asumiría datos limpios.
        start = date.fromisoformat(config["start_date"]) if "start_date" in config else None
        end = date.fromisoformat(config["end_date"]) if "end_date" in config else None
        seen_dates: set[str] = set()
        for row_num, row in enumerate(data_rows, start=2):  # fila 1 es la cabecera
            if len(row) != len(_EXPECTED_HEADER):
                raise DataProviderError(
                    f"Stooq: la fila {row_num} tiene {len(row)} campos, se esperaban "
                    f"{len(_EXPECTED_HEADER)} ({','.join(_EXPECTED_HEADER)})."
                )
            row_date_raw = row[0]
            try:
                row_date = date.fromisoformat(row_date_raw)
            except ValueError as exc:
                raise DataProviderError(
                    f"Stooq: la fila {row_num} tiene una fecha inválida {row_date_raw!r}."
                ) from exc
            if start is not None and end is not None and not (start <= row_date <= end):
                raise DataProviderError(
                    f"Stooq: la fila {row_num} tiene fecha {row_date_raw!r}, fuera del "
                    f"rango solicitado [{config['start_date']}, {config['end_date']}]."
                )
            if row_date_raw in seen_dates:
                # Reauditoría 2026-07-17 (ronda 3) [MEDIA]: mismo criterio
                # que el fix en ecb_fx — una fecha repetida no debe
                # producir dos filas en el CSV "canónico".
                raise DataProviderError(f"Stooq: la fila {row_num} repite la fecha {row_date_raw!r}.")
            seen_dates.add(row_date_raw)
            for col_name, value in zip(_NUMERIC_COLUMNS, row[1:]):
                if not value.strip():
                    raise DataProviderError(f"Stooq: la fila {row_num} tiene {col_name!r} vacío.")
                try:
                    numeric_value = float(value)
                except ValueError as exc:
                    raise DataProviderError(
                        f"Stooq: la fila {row_num} tiene {col_name!r}={value!r} no numérico."
                    ) from exc
                if not math.isfinite(numeric_value):
                    # Reauditoría 2026-07-17 (ronda 3) [MEDIA]: float("nan")/
                    # float("inf") NO lanzan ValueError — `float(value)` por
                    # sí solo aceptaba "NaN"/"Infinity" como una cotización real.
                    raise DataProviderError(
                        f"Stooq: la fila {row_num} tiene {col_name!r}={value!r} no finito (NaN/infinito)."
                    )

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(_EXPECTED_HEADER)
        writer.writerows(data_rows)
        return buf.getvalue(), list(_EXPECTED_HEADER), len(data_rows)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
