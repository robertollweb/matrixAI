# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `open_meteo`.

Cero red real: `secure_fetch` se sustituye por completo vía
`unittest.mock.patch`. Los fixtures JSON de `archive`/`error`/`marine`
son respuestas REALES grabadas de la API pública el 2026-07-17 (curl
directo, ver contrato — recortadas a un rango corto de días)."""
from __future__ import annotations

import json
import unittest.mock

import pytest

from matrixai.training.data_provider import DataProviderError
from matrixai.training.provider_open_meteo import OpenMeteoProvider
from matrixai.training.secure_fetch import SecureFetchError, SecureFetchResult

# Respuesta REAL grabada: GET archive-api.open-meteo.com/v1/archive
# ?latitude=52.52&longitude=13.41&start_date=2024-01-01&end_date=2024-01-03
# &daily=temperature_2m_max,temperature_2m_min&timezone=UTC (2026-07-17).
_REAL_ARCHIVE_JSON = {
    "latitude": 52.54833, "longitude": 13.407822, "generationtime_ms": 3.67,
    "utc_offset_seconds": 0, "timezone": "GMT", "timezone_abbreviation": "GMT",
    "elevation": 38.0,
    "daily_units": {"time": "iso8601", "temperature_2m_max": "°C", "temperature_2m_min": "°C"},
    "daily": {
        "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "temperature_2m_max": [7.4, 7.9, 10.6],
        "temperature_2m_min": [3.4, 2.5, 7.0],
    },
}

# Respuesta REAL grabada: GET marine-api.open-meteo.com/v1/marine
# ?latitude=53.55&longitude=8.0&start_date=2024-01-01&end_date=2024-01-01
# &hourly=wave_height&timezone=UTC (2026-07-17) — recortada a 3 horas
# (la real trae 48, con algún `null` genuino de la API en horas sin dato).
_REAL_MARINE_JSON = {
    "latitude": 53.541664, "longitude": 7.9583435, "generationtime_ms": 1.2,
    "utc_offset_seconds": 0, "timezone": "GMT", "timezone_abbreviation": "GMT",
    "elevation": 1.0,
    "hourly_units": {"time": "iso8601", "wave_height": "m"},
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
        "wave_height": [None, None, None],
    },
}

# Respuesta REAL grabada: GET archive-api.open-meteo.com/v1/archive
# ?latitude=999&longitude=13.41&... (latitud fuera de rango, 2026-07-17).
_REAL_ERROR_JSON = {
    "reason": "Latitude must be in range of -90 to 90°. Given: 999.0.",
    "error": True,
}


def _archive_config(**overrides):
    base = {
        "dataset": "archive", "latitude": 52.52, "longitude": 13.41,
        "start_date": "2024-01-01", "end_date": "2024-01-03",
        "variables": ["temperature_2m_max", "temperature_2m_min"],
    }
    base.update(overrides)
    return base


def _marine_config(**overrides):
    base = {
        "dataset": "marine", "latitude": 53.55, "longitude": 8.0,
        "start_date": "2024-01-01", "end_date": "2024-01-01",
        "variables": ["wave_height"],
    }
    base.update(overrides)
    return base


def _fetch_result(payload: dict, url: str = "https://archive-api.open-meteo.com/v1/archive?...") -> SecureFetchResult:
    return SecureFetchResult(
        url=url, status=200, body=json.dumps(payload).encode("utf-8"), content_type="application/json",
    )


def _patched(**kwargs):
    return unittest.mock.patch("matrixai.training.provider_open_meteo.secure_fetch", **kwargs)


class TestValidateConfig:
    def test_valid_archive_config_has_no_errors(self):
        assert OpenMeteoProvider().validate_config(_archive_config()) == []

    def test_invalid_dataset_is_rejected(self):
        errors = OpenMeteoProvider().validate_config(_archive_config(dataset="bogus"))
        assert any("dataset" in e for e in errors)

    def test_latitude_out_of_range_is_rejected(self):
        errors = OpenMeteoProvider().validate_config(_archive_config(latitude=999))
        assert any("latitude" in e for e in errors)

    def test_start_after_end_is_rejected(self):
        errors = OpenMeteoProvider().validate_config(
            _archive_config(start_date="2024-06-01", end_date="2024-01-01")
        )
        assert any("start_date" in e for e in errors)

    def test_empty_variables_is_rejected(self):
        errors = OpenMeteoProvider().validate_config(_archive_config(variables=[]))
        assert any("variables" in e for e in errors)


class TestDownloadArchiveWithRealFixture:
    def test_returns_canonical_csv_matching_real_response(self):
        with _patched(return_value=_fetch_result(_REAL_ARCHIVE_JSON)) as mock_fetch:
            result = OpenMeteoProvider().download(_archive_config(), license_accepted=True)
        assert mock_fetch.call_count == 1
        assert result.rows == 3
        assert result.columns == ["time", "temperature_2m_max", "temperature_2m_min"]
        lines = result.csv_text.splitlines()
        assert lines[0] == "time,temperature_2m_max,temperature_2m_min"
        assert lines[1] == "2024-01-01,7.4,3.4"
        assert lines[3] == "2024-01-03,10.6,7.0"

    def test_license_info_matches_verified_open_meteo_terms(self):
        info = OpenMeteoProvider().get_license_info()
        assert info.commercial_use_allowed is False
        assert info.requires_attribution is True


class TestDownloadMarineWithRealFixture:
    def test_returns_canonical_csv_and_preserves_real_nulls_as_empty(self):
        with _patched(return_value=_fetch_result(_REAL_MARINE_JSON)):
            result = OpenMeteoProvider().download(_marine_config(), license_accepted=True)
        assert result.rows == 3
        lines = result.csv_text.splitlines()
        assert lines[0] == "time,wave_height"
        # La API real devolvió null genuino en esas horas (noche, sin boya) —
        # se preserva como campo vacío, nunca se inventa un 0.0.
        assert lines[1] == "2024-01-01T00:00,"


class TestApiErrorResponse:
    def test_real_error_payload_raises_actionable_error(self):
        with _patched(return_value=_fetch_result(_REAL_ERROR_JSON)):
            with pytest.raises(DataProviderError, match="Latitude must be in range"):
                OpenMeteoProvider().download(_archive_config(latitude=52.52), license_accepted=True)


class TestUnexpectedSchema:
    def test_non_json_body_is_rejected(self):
        bad = SecureFetchResult(url="https://archive-api.open-meteo.com/x", status=200, body=b"<html>nope</html>", content_type="text/html")
        with _patched(return_value=bad):
            with pytest.raises(DataProviderError, match="no-JSON"):
                OpenMeteoProvider().download(_archive_config(), license_accepted=True)

    def test_missing_daily_key_is_rejected(self):
        bad = _fetch_result({"latitude": 1, "longitude": 1})
        with _patched(return_value=bad):
            with pytest.raises(DataProviderError, match="esquema inesperado"):
                OpenMeteoProvider().download(_archive_config(), license_accepted=True)

    def test_missing_requested_variable_is_rejected(self):
        payload = {"daily": {"time": ["2024-01-01"], "temperature_2m_max": [7.4]}}
        with _patched(return_value=_fetch_result(payload)):
            with pytest.raises(DataProviderError, match="temperature_2m_min"):
                OpenMeteoProvider().download(_archive_config(), license_accepted=True)

    def test_variable_length_mismatch_is_rejected(self):
        payload = {"daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_max": [7.4],
            "temperature_2m_min": [3.4, 2.5],
        }}
        with _patched(return_value=_fetch_result(payload)):
            with pytest.raises(DataProviderError, match="longitud distinta"):
                OpenMeteoProvider().download(_archive_config(), license_accepted=True)


class TestSecureFetchFailurePropagates:
    def test_secure_fetch_error_becomes_data_provider_error(self):
        with _patched(side_effect=SecureFetchError("timeout")):
            with pytest.raises(DataProviderError, match="Open-Meteo"):
                OpenMeteoProvider().download(_archive_config(), license_accepted=True)


class TestLicenseGate:
    def test_download_without_acceptance_makes_zero_requests(self):
        with _patched() as mock_fetch:
            with pytest.raises(DataProviderError, match="exige aceptar su licencia"):
                OpenMeteoProvider().download(_archive_config(), license_accepted=False)
            mock_fetch.assert_not_called()


class TestHostAllowlistWiring:
    def test_archive_dataset_uses_archive_host(self):
        with _patched(return_value=_fetch_result(_REAL_ARCHIVE_JSON)) as mock_fetch:
            OpenMeteoProvider().download(_archive_config(), license_accepted=True)
        called_url = mock_fetch.call_args.args[0]
        assert called_url.startswith("https://archive-api.open-meteo.com/v1/archive?")

    def test_marine_dataset_uses_marine_host(self):
        with _patched(return_value=_fetch_result(_REAL_MARINE_JSON)) as mock_fetch:
            OpenMeteoProvider().download(_marine_config(), license_accepted=True)
        called_url = mock_fetch.call_args.args[0]
        assert called_url.startswith("https://marine-api.open-meteo.com/v1/marine?")

    def test_allowed_hosts_passed_to_secure_fetch_are_the_fixed_pair(self):
        with _patched(return_value=_fetch_result(_REAL_ARCHIVE_JSON)) as mock_fetch:
            OpenMeteoProvider().download(_archive_config(), license_accepted=True)
        allowed = mock_fetch.call_args.kwargs["allowed_hosts"]
        assert allowed == frozenset({"archive-api.open-meteo.com", "marine-api.open-meteo.com"})


class TestEstimate:
    def test_estimate_does_not_touch_the_network(self):
        with _patched() as mock_fetch:
            estimate = OpenMeteoProvider().estimate_download(_archive_config())
        mock_fetch.assert_not_called()
        assert estimate.estimated_rows == 3

    def test_marine_estimate_uses_hourly_granularity(self):
        estimate = OpenMeteoProvider().estimate_download(_marine_config())
        assert estimate.estimated_rows == 24  # 1 día × 24 horas


class TestAvailability:
    def test_check_availability_true_on_success(self):
        with _patched(return_value=_fetch_result(_REAL_ARCHIVE_JSON)):
            assert OpenMeteoProvider().check_availability() is True

    def test_check_availability_false_on_failure(self):
        with _patched(side_effect=SecureFetchError("down")):
            assert OpenMeteoProvider().check_availability() is False
