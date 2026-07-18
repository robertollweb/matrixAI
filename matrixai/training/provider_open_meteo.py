# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `open_meteo`.

Open-Meteo (https://open-meteo.com): API pública, sin clave, JSON. Dos
datasets cubiertos — el contrato los exige a los dos porque el ejemplo
canónico del estado de la mar necesita `marine`:
  - `archive`: histórico meteorológico diario por lat/lon
    (`archive-api.open-meteo.com/v1/archive`).
  - `marine`: histórico marino horario (altura de ola, periodo...) por
    lat/lon (`marine-api.open-meteo.com/v1/marine`).

Licencia verificada (ver memoria del proyecto): CC-BY 4.0, uso NO
comercial — declarada en `get_license_info`; el gate de licencia
(`require_valid_acceptance`) es quien la hace bloqueante, no este
módulo.

El host de cada endpoint es FIJO (constante de módulo) — la config del
usuario (lat/lon/fechas/variables) solo entra en la query string, nunca
en el host (ver `secure_fetch` sobre por qué esto importa para SSRF).
"""
from __future__ import annotations

import csv
import io
import json
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

_ARCHIVE_HOST = "archive-api.open-meteo.com"
_MARINE_HOST = "marine-api.open-meteo.com"
_ALLOWED_HOSTS = frozenset({_ARCHIVE_HOST, _MARINE_HOST})
_DATASETS = frozenset({"archive", "marine"})
_MAX_BYTES = 20_000_000
_TIMEOUT = 20.0
_MAX_VARIABLES = 20
_MAX_DAYS = 366 * 5  # 5 años — generoso para el ejemplo canónico (histórico largo), acota el caso patológico


class OpenMeteoProvider:
    provider_id = "open_meteo"

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_id=self.provider_id,
            display_name="Open-Meteo",
            description="Histórico meteorológico y marino público, por coordenadas.",
            requires_network=True,
        )

    def get_license_info(self) -> LicenseInfo:
        return LicenseInfo(
            name="Open-Meteo (CC-BY 4.0, uso no comercial)",
            url="https://open-meteo.com/en/license",
            summary=(
                "Gratuito para uso NO comercial con atribución obligatoria "
                "(CC-BY 4.0) — el uso comercial requiere una licencia de pago aparte."
            ),
            requires_attribution=True,
            commercial_use_allowed=False,
            summary_i18n={
                "en": (
                    "Free for non-commercial use with mandatory attribution "
                    "(CC-BY 4.0); commercial use requires a separate paid license."
                ),
            },
        )

    def check_availability(self) -> bool:
        # Auditoría 2026-07-17 (ronda 2) [ALTA, mismo patrón que motivó el
        # fix en Stooq]: comprobar solo que `secure_fetch` no lance no basta
        # — un 200 con un cuerpo de forma inesperada (mantenimiento,
        # cambio de esquema) declararía "disponible" sin serlo de verdad.
        # Se valida el CUERPO con el mismo camino que usa `download`.
        probe_config = {"dataset": "archive", "variables": ["temperature_2m_max"]}
        try:
            fetched = secure_fetch(
                f"https://{_ARCHIVE_HOST}/v1/archive?latitude=0&longitude=0"
                "&start_date=2024-01-01&end_date=2024-01-01&daily=temperature_2m_max",
                allowed_hosts=_ALLOWED_HOSTS, timeout=_TIMEOUT, max_bytes=_MAX_BYTES,
            )
            self._to_canonical_csv(fetched.body, probe_config)
            return True
        except (SecureFetchError, DataProviderError):
            return False

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        dataset = config.get("dataset")
        if dataset not in _DATASETS:
            errors.append(f"dataset debe ser uno de {sorted(_DATASETS)} (recibido {dataset!r}).")
        lat, lon = config.get("latitude"), config.get("longitude")
        if not isinstance(lat, (int, float)) or isinstance(lat, bool) or not (-90 <= lat <= 90):
            errors.append("latitude debe ser un número entre -90 y 90.")
        if not isinstance(lon, (int, float)) or isinstance(lon, bool) or not (-180 <= lon <= 180):
            errors.append("longitude debe ser un número entre -180 y 180.")
        parsed_start = self._parse_date(config.get("start_date"), "start_date", errors)
        parsed_end = self._parse_date(config.get("end_date"), "end_date", errors)
        if parsed_start and parsed_end:
            if parsed_start > parsed_end:
                errors.append("start_date debe ser anterior o igual a end_date.")
            elif (parsed_end - parsed_start).days + 1 > _MAX_DAYS:
                # Auditoría 2026-07-17 (ronda 2) [MEDIA]: sin cota, un rango
                # de fechas arbitrariamente largo (o un typo de año) pide un
                # JSON potencialmente enorme — max_bytes de secure_fetch ya
                # lo corta, pero es mejor rechazarlo ANTES de la petición
                # con un mensaje específico.
                errors.append(f"El rango start_date/end_date no puede superar {_MAX_DAYS} días.")
        variables = config.get("variables")
        if (not isinstance(variables, list) or not variables
                or not all(isinstance(v, str) and v.strip() for v in variables)):
            errors.append("variables debe ser una lista no vacía de nombres de variable.")
        elif len(variables) > _MAX_VARIABLES:
            errors.append(f"variables no puede tener más de {_MAX_VARIABLES} entradas.")
        elif len(set(variables)) != len(variables):
            # Auditoría 2026-07-17 (ronda 2) [MEDIA]: una variable repetida
            # pasaba limpio y generaba DOS columnas CSV con el mismo nombre
            # (`temperature_2m_max,temperature_2m_max`) — C1 no puede
            # consumir una cabecera con nombres duplicados.
            errors.append("variables tiene nombres duplicados.")
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
        variables = config["variables"]
        # Estimación sin red: marine/archive devuelven 1 fila/día (archive) u
        # 1 fila/hora (marine) — se muestra la unidad real en las notas para
        # no prometer una cifra de filas que luego no cuadre.
        rows_per_day = 24 if config["dataset"] == "marine" else 1
        estimated_rows = days * rows_per_day
        return DownloadEstimate(
            estimated_rows=estimated_rows, estimated_bytes=estimated_rows * len(variables) * 12,
            notes=(
                f"{days} día(s) × {len(variables)} variable(s) "
                f"({'horario' if config['dataset'] == 'marine' else 'diario'}), estimación aproximada."
            ),
            notes_i18n={
                "en": (
                    f"{days} day(s) × {len(variables)} variable(s) "
                    f"({'hourly' if config['dataset'] == 'marine' else 'daily'}), approximate estimate."
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
            raise DataProviderError(f"Open-Meteo: {exc}") from exc

        csv_text, columns, n_rows = self._to_canonical_csv(fetched.body, config)
        return DownloadResult(
            csv_text=csv_text, rows=n_rows, columns=columns,
            source_url=fetched.url, fetched_at=_utcnow_iso(),
            license_info=self.get_license_info(),
            provenance_extra={
                "dataset": config["dataset"],
                "latitude": config["latitude"],
                "longitude": config["longitude"],
                "license_acceptance": license_acceptance.to_dict(),
            },
        )

    def _build_url(self, config: dict[str, Any]) -> str:
        host = _ARCHIVE_HOST if config["dataset"] == "archive" else _MARINE_HOST
        granularity = "daily" if config["dataset"] == "archive" else "hourly"
        params = {
            "latitude": config["latitude"], "longitude": config["longitude"],
            "start_date": config["start_date"], "end_date": config["end_date"],
            granularity: ",".join(config["variables"]),
            "timezone": "UTC",
        }
        return f"https://{host}/v1/{config['dataset']}?{urllib.parse.urlencode(params)}"

    def _to_canonical_csv(self, body: bytes, config: dict[str, Any]) -> tuple[str, list[str], int]:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DataProviderError(f"Open-Meteo devolvió una respuesta no-JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise DataProviderError("Open-Meteo devolvió un JSON con forma inesperada (no es un objeto).")
        if payload.get("error"):
            raise DataProviderError(f"Open-Meteo rechazó la petición: {payload.get('reason', payload)}")

        granularity = "daily" if config["dataset"] == "archive" else "hourly"
        block = payload.get(granularity)
        if not isinstance(block, dict) or "time" not in block:
            raise DataProviderError(
                f"Open-Meteo devolvió un JSON sin la clave {granularity!r} esperada — esquema inesperado."
            )
        times = block["time"]
        if not isinstance(times, list):
            raise DataProviderError("Open-Meteo: 'time' no es una lista — esquema inesperado.")
        if not times:
            # Auditoría C8 [MEDIA — matriz de errores §29, "dataset vacío"]:
            # una combinación de fecha/ubicación/dataset sin cobertura
            # devuelve `time: []` con HTTP 200 — sin este check, el CSV
            # canónico salía con solo cabecera y el fallo se detectaba mucho
            # más tarde (analyze_dataset_csv, fuera del try/except de
            # generate_project_from_template) con un mensaje genérico que no
            # dice que el proveedor es quien no tiene datos. Mismo criterio
            # que ya aplicaba `provider_ecb_fx.py`.
            raise DataProviderError(
                f"Open-Meteo no devolvió ningún registro para dataset={config['dataset']!r}, "
                f"lat={config['latitude']!r}, lon={config['longitude']!r}, rango "
                f"{config['start_date']}..{config['end_date']} — prueba otra ubicación o rango de fechas."
            )
        variables = config["variables"]
        for var in variables:
            if var not in block:
                raise DataProviderError(f"Open-Meteo no devolvió la variable {var!r} solicitada.")
            if not isinstance(block[var], list) or len(block[var]) != len(times):
                raise DataProviderError(
                    f"Open-Meteo: la variable {var!r} tiene una longitud distinta de 'time' — esquema inesperado."
                )

        columns = ["time", *variables]
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        for i in range(len(times)):
            row = [times[i], *(block[var][i] for var in variables)]
            writer.writerow(["" if v is None else v for v in row])
        return buf.getvalue(), columns, len(times)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
