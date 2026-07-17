# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `ecb_fx`.

Banco Central Europeo (https://data.ecb.europa.eu): tipos de cambio de
referencia diarios EUR→divisa, API SDMX pública, sin clave, CSV directo
— añadido en la reauditoría 2026-07-17 como sustituto de `stooq`
(bloqueado hoy por un reto anti-bot JS, ver `provider_stooq.py`).
Verificado con `curl` real durante este mismo corte: responde HTTPS,
CSV, sin necesitar motor JS.

Licencia verificada (`curl` real a la página de disclaimer/copyright del
ECB, 2026-07-17): reutilización libre — incluido uso comercial — con
atribución obligatoria (citar al ECB como fuente; si el dato se
modifica, decirlo explícitamente). Ver `get_license_info`.

Formato CSV real (`format=csvdata&detail=dataonly`):
`KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,
OBS_VALUE` — se parsea por NOMBRE de columna (`csv.DictReader`), nunca
por posición, para no depender de que el ECB mantenga el orden exacto.
"""
from __future__ import annotations

import csv
import io
import re
import urllib.parse
from datetime import date, datetime, timezone
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

_HOST = "data-api.ecb.europa.eu"
_ALLOWED_HOSTS = frozenset({_HOST})
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_MAX_DAYS = 366 * 20  # el ECB publica desde 1999 — 20 años es generoso sin ser ilimitado
_MAX_BYTES = 20_000_000
_TIMEOUT = 20.0
_REQUIRED_COLUMNS = ("TIME_PERIOD", "OBS_VALUE")


class EcbFxProvider:
    provider_id = "ecb_fx"

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_id=self.provider_id,
            display_name="BCE — tipos de cambio",
            description="Tipos de cambio de referencia diarios EUR→divisa (Banco Central Europeo).",
            requires_network=True,
        )

    def get_license_info(self) -> LicenseInfo:
        return LicenseInfo(
            name="European Central Bank — Disclaimer & copyright",
            url="https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
            summary=(
                "Reutilización libre (incluido uso comercial) con atribución obligatoria: "
                "hay que citar al ECB como fuente y declarar explícitamente cualquier "
                "modificación de los datos (p.ej. ajuste estacional)."
            ),
            requires_attribution=True,
            commercial_use_allowed=True,
        )

    def check_availability(self) -> bool:
        probe_config = {"currency": "USD", "start_date": "2024-01-02", "end_date": "2024-01-03"}
        try:
            fetched = secure_fetch(
                self._build_url(probe_config), allowed_hosts=_ALLOWED_HOSTS,
                timeout=_TIMEOUT, max_bytes=_MAX_BYTES,
            )
            self._to_canonical_csv(fetched.body, probe_config)
            return True
        except (SecureFetchError, DataProviderError):
            return False

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        currency = config.get("currency")
        if not isinstance(currency, str) or not _CURRENCY_RE.match(currency):
            errors.append("currency debe ser un código ISO-4217 de 3 letras mayúsculas (p.ej. 'USD').")
        elif currency == "EUR":
            errors.append("currency no puede ser 'EUR' (la serie ya es EUR→divisa).")
        parsed_start = self._parse_date(config.get("start_date"), "start_date", errors)
        parsed_end = self._parse_date(config.get("end_date"), "end_date", errors)
        if parsed_start and parsed_end:
            if parsed_start > parsed_end:
                errors.append("start_date debe ser anterior o igual a end_date.")
            elif (parsed_end - parsed_start).days + 1 > _MAX_DAYS:
                errors.append(f"El rango start_date/end_date no puede superar {_MAX_DAYS} días.")
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
            estimated_rows=days, estimated_bytes=days * 60,
            notes=f"{days} día(s) naturales — estimación aproximada; fines de semana sin cotización se excluyen.",
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
            raise DataProviderError(f"BCE: {exc}") from exc

        csv_text, columns, n_rows = self._to_canonical_csv(fetched.body, config)
        return DownloadResult(
            csv_text=csv_text, rows=n_rows, columns=columns,
            source_url=fetched.url, fetched_at=_utcnow_iso(),
            license_info=self.get_license_info(),
            provenance_extra={
                "currency": config["currency"],
                "license_acceptance": license_acceptance.to_dict(),
            },
        )

    def _build_url(self, config: dict[str, Any]) -> str:
        # Serie de referencia diaria EUR→divisa (EXR/D.{ccy}.EUR.SP00.A) —
        # ver documentación SDMX del ECB (data.ecb.europa.eu/help/api/overview).
        params = {
            "startPeriod": config["start_date"], "endPeriod": config["end_date"],
            "format": "csvdata", "detail": "dataonly",
        }
        return f"https://{_HOST}/service/data/EXR/D.{config['currency']}.EUR.SP00.A?{urllib.parse.urlencode(params)}"

    def _to_canonical_csv(self, body: bytes, config: dict[str, Any]) -> tuple[str, list[str], int]:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataProviderError(f"BCE devolvió una respuesta no-UTF8: {exc}") from exc
        if not text.strip():
            raise DataProviderError(
                f"El BCE no devolvió ninguna cotización para {config.get('currency')!r} en ese rango."
            )
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
        missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise DataProviderError(
                f"BCE devolvió un CSV sin las columnas esperadas {missing} — esquema inesperado "
                f"(columnas recibidas: {fieldnames})."
            )

        start = date.fromisoformat(config["start_date"]) if "start_date" in config else None
        end = date.fromisoformat(config["end_date"]) if "end_date" in config else None
        canonical_rows: list[tuple[str, str]] = []
        for row_num, row in enumerate(reader, start=2):  # fila 1 es la cabecera
            raw_date = row.get("TIME_PERIOD")
            raw_value = row.get("OBS_VALUE")
            if not raw_date or not raw_value:
                raise DataProviderError(f"BCE: la fila {row_num} tiene TIME_PERIOD/OBS_VALUE vacío.")
            try:
                row_date = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise DataProviderError(f"BCE: la fila {row_num} tiene TIME_PERIOD inválido {raw_date!r}.") from exc
            if start is not None and end is not None and not (start <= row_date <= end):
                raise DataProviderError(
                    f"BCE: la fila {row_num} tiene fecha {raw_date!r}, fuera del rango "
                    f"solicitado [{config['start_date']}, {config['end_date']}]."
                )
            try:
                float(raw_value)
            except ValueError as exc:
                raise DataProviderError(f"BCE: la fila {row_num} tiene OBS_VALUE={raw_value!r} no numérico.") from exc
            canonical_rows.append((raw_date, raw_value))

        if not canonical_rows:
            raise DataProviderError(
                f"El BCE no devolvió ninguna cotización para {config.get('currency')!r} en ese rango."
            )

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["Date", "ExchangeRate"])
        writer.writerows(canonical_rows)
        return buf.getvalue(), ["Date", "ExchangeRate"], len(canonical_rows)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
