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

Reauditoría 2026-07-17 (ronda 3) [MEDIA]: `currency` validaba solo la
FORMA (3 letras mayúsculas) — cualquier código sintácticamente válido
pero inexistente (p.ej. "ZZZ") pasaba `validate_config`/`estimate_
download` y solo fallaba al descargar de verdad. `_KNOWN_CURRENCIES` es
la lista REAL y COMPLETA de divisas de la serie EXR diaria, obtenida en
vivo con `curl` contra el propio catálogo del BCE (`GET .../EXR/
D..EUR.SP00.A?format=csvdata&detail=serieskeysonly`, 2026-07-17) — no
una lista ISO-4217 genérica ni inventada. Incluye divisas de
preadopción del euro ya discontinuadas (CYP, EEK, GRD, LTL, LVL, MTL,
SIT, SKK) porque siguen presentes en la serie HISTÓRICA (invariante 5:
reproducibilidad de un proyecto antiguo que las use).
"""
from __future__ import annotations

import csv
import io
import math
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
# Divisas REALES de la serie EXR/D..EUR.SP00.A — capturadas en vivo
# (curl) contra data-api.ecb.europa.eu el 2026-07-17, ver docstring.
_KNOWN_CURRENCIES = frozenset({
    "ARS", "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CYP", "CZK", "DKK",
    "DZD", "EEK", "GBP", "GRD", "HKD", "HRK", "HUF", "IDR", "ILS", "INR",
    "ISK", "JPY", "KRW", "LTL", "LVL", "MAD", "MTL", "MXN", "MYR", "NOK",
    "NZD", "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "SIT", "SKK", "THB",
    "TRY", "TWD", "USD", "ZAR",
})
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
        # Reauditoría C8 [ALTA]: el resumen omitía una obligación real del
        # disclaimer oficial del BCE (verificado con `curl` real,
        # 2026-07-18) — si el resultado se INCORPORA en documentos
        # VENDIDOS, hay que informar a los compradores, antes de pagar y
        # cada vez que accedan, de que la información original está
        # disponible gratis en el sitio del BCE. Añadida al resumen.
        return LicenseInfo(
            name="European Central Bank — Disclaimer & copyright",
            url="https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
            summary=(
                "Reutilización libre (incluido uso comercial) con atribución obligatoria: "
                "hay que citar al ECB como fuente y declarar explícitamente cualquier "
                "modificación de los datos (p.ej. ajuste estacional). Si el resultado se "
                "VENDE (en cualquier formato), hay que informar a los compradores, antes de "
                "pagar y cada vez que accedan, de que la información original es gratuita "
                "en el sitio del BCE."
            ),
            requires_attribution=True,
            commercial_use_allowed=True,
            summary_i18n={
                "en": (
                    "Free reuse (including commercial use) with mandatory attribution: "
                    "cite the ECB as the source and explicitly disclose any data modifications. "
                    "If the result is SOLD (in any format), buyers must be informed, before "
                    "paying and every time they access it, that the original information is "
                    "free on the ECB's website."
                ),
            },
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
        if not isinstance(currency, str):
            errors.append("currency debe ser un texto (código de divisa).")
        elif currency == "EUR":
            errors.append("currency no puede ser 'EUR' (la serie ya es EUR→divisa).")
        elif currency not in _KNOWN_CURRENCIES:
            errors.append(
                f"currency {currency!r} no es una divisa publicada por la serie EXR diaria "
                f"del BCE. Divisas disponibles: {sorted(_KNOWN_CURRENCIES)}."
            )
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
            notes_i18n={
                "en": f"{days} calendar day(s) — approximate estimate; weekends without rates are excluded.",
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
        seen_dates: set[str] = set()
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
            if raw_date in seen_dates:
                # Reauditoría 2026-07-17 (ronda 3) [MEDIA]: una serie con
                # una fecha repetida (dato revisado/duplicado del BCE) se
                # colaba dos veces en el CSV "canónico" — una sola fecha
                # debe producir una sola fila.
                raise DataProviderError(f"BCE: la fila {row_num} repite la fecha {raw_date!r}.")
            seen_dates.add(raw_date)
            try:
                numeric_value = float(raw_value)
            except ValueError as exc:
                raise DataProviderError(f"BCE: la fila {row_num} tiene OBS_VALUE={raw_value!r} no numérico.") from exc
            if not math.isfinite(numeric_value):
                # Reauditoría 2026-07-17 (ronda 3) [MEDIA]: float("nan")/
                # float("inf") NO lanzan ValueError en Python — `float(raw_
                # value)` por sí solo aceptaba "NaN"/"Infinity" como si
                # fueran una cotización real.
                raise DataProviderError(
                    f"BCE: la fila {row_num} tiene OBS_VALUE={raw_value!r} no finito (NaN/infinito)."
                )
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
