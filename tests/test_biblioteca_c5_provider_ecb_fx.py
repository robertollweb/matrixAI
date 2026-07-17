# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 (reauditoría ronda 2) — proveedor
`ecb_fx`, sustituto de `stooq` (bloqueado por un reto anti-bot JS real).

Cero red real en los tests: `secure_fetch` se sustituye por completo vía
`unittest.mock.patch`. `_REAL_CSV`/`_REAL_ERROR_BODY` son respuestas
REALES grabadas por `curl` directo contra `data-api.ecb.europa.eu`
2026-07-17 (ver docstring de `provider_ecb_fx.py`)."""
from __future__ import annotations

import unittest.mock

import pytest

from matrixai.training.data_provider import DataProviderError, LicenseAcceptanceStore
from matrixai.training.provider_ecb_fx import EcbFxProvider
from matrixai.training.secure_fetch import SecureFetchError, SecureFetchResult

# Respuesta REAL grabada: GET data-api.ecb.europa.eu/service/data/EXR/
# D.USD.EUR.SP00.A?startPeriod=2024-01-01&endPeriod=2024-01-10&format=
# csvdata&detail=dataonly (2026-07-17).
_REAL_CSV = (
    "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.0956\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-03,1.0919\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-04,1.0953\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-05,1.0921\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-08,1.0946\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-09,1.094\n"
    "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-10,1.0946\n"
)


def _config(**overrides):
    base = {"currency": "USD", "start_date": "2024-01-01", "end_date": "2024-01-10"}
    base.update(overrides)
    return base


def _fetch_result(body: str, content_type: str = "text/csv") -> SecureFetchResult:
    return SecureFetchResult(
        url="https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?...",
        status=200, body=body.encode("utf-8"), content_type=content_type,
    )


def _patched(**kwargs):
    return unittest.mock.patch("matrixai.training.provider_ecb_fx.secure_fetch", **kwargs)


def _accepted():
    provider = EcbFxProvider()
    store = LicenseAcceptanceStore()
    return provider, store.record(provider, actor="test")


class TestValidateConfig:
    def test_valid_config_has_no_errors(self):
        assert EcbFxProvider().validate_config(_config()) == []

    def test_lowercase_currency_is_rejected(self):
        errors = EcbFxProvider().validate_config(_config(currency="usd"))
        assert any("currency" in e for e in errors)

    def test_wrong_length_currency_is_rejected(self):
        errors = EcbFxProvider().validate_config(_config(currency="US"))
        assert any("currency" in e for e in errors)

    def test_eur_is_rejected(self):
        errors = EcbFxProvider().validate_config(_config(currency="EUR"))
        assert any("EUR" in e for e in errors)

    def test_start_after_end_is_rejected(self):
        errors = EcbFxProvider().validate_config(_config(start_date="2024-06-01", end_date="2024-01-01"))
        assert any("start_date" in e for e in errors)

    def test_range_over_max_days_is_rejected(self):
        errors = EcbFxProvider().validate_config(_config(start_date="2000-01-01", end_date="2024-01-01"))
        assert any("días" in e for e in errors)

    def test_well_formed_but_nonexistent_currency_is_rejected(self):
        """Reauditoría 2026-07-17 (ronda 3) [MEDIA]: 'ZZZ' tiene la FORMA
        correcta (3 letras mayúsculas) pero no es una divisa real de la
        serie EXR — antes solo se validaba la forma, así que pasaba
        validate_config()/estimate_download() y solo fallaba al
        descargar de verdad."""
        errors = EcbFxProvider().validate_config(_config(currency="ZZZ"))
        assert any("ZZZ" in e for e in errors)

    def test_all_known_currencies_are_accepted(self):
        for currency in [
            "ARS", "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CYP", "CZK", "DKK",
            "DZD", "EEK", "GBP", "GRD", "HKD", "HRK", "HUF", "IDR", "ILS", "INR",
            "ISK", "JPY", "KRW", "LTL", "LVL", "MAD", "MTL", "MXN", "MYR", "NOK",
            "NZD", "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "SIT", "SKK", "THB",
            "TRY", "TWD", "USD", "ZAR",
        ]:
            assert EcbFxProvider().validate_config(_config(currency=currency)) == []


class TestDownloadWithRealFixture:
    def test_returns_canonical_csv_matching_real_response(self):
        provider, acceptance = _accepted()
        with _patched(return_value=_fetch_result(_REAL_CSV)) as mock_fetch:
            result = provider.download(_config(), license_acceptance=acceptance)
        assert mock_fetch.call_count == 1
        assert result.rows == 7
        assert result.columns == ["Date", "ExchangeRate"]
        lines = result.csv_text.splitlines()
        assert lines[0] == "Date,ExchangeRate"
        assert lines[1] == "2024-01-02,1.0956"
        assert lines[-1] == "2024-01-10,1.0946"

    def test_license_info_allows_commercial_use_with_attribution(self):
        info = EcbFxProvider().get_license_info()
        assert info.commercial_use_allowed is True
        assert info.requires_attribution is True


class TestUnexpectedSchema:
    def test_empty_body_is_rejected(self):
        provider, acceptance = _accepted()
        with _patched(return_value=_fetch_result("")):
            with pytest.raises(DataProviderError, match="ninguna cotización"):
                provider.download(_config(), license_acceptance=acceptance)

    def test_missing_required_columns_is_rejected(self):
        provider, acceptance = _accepted()
        bad = "KEY,FREQ\nEXR.D.USD.EUR.SP00.A,D\n"
        with _patched(return_value=_fetch_result(bad)):
            with pytest.raises(DataProviderError, match="esquema inesperado"):
                provider.download(_config(), license_acceptance=acceptance)

    def test_non_numeric_obs_value_is_rejected(self):
        provider, acceptance = _accepted()
        bad = (
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,not_a_number\n"
        )
        with _patched(return_value=_fetch_result(bad)):
            with pytest.raises(DataProviderError, match="no numérico"):
                provider.download(_config(), license_acceptance=acceptance)

    def test_nan_obs_value_is_rejected(self):
        """Reauditoría 2026-07-17 (ronda 3) [MEDIA]: float("nan") NO
        lanza ValueError en Python — un float(raw_value) desnudo aceptaba
        "NaN" como si fuera una cotización real."""
        provider, acceptance = _accepted()
        bad = (
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,NaN\n"
        )
        with _patched(return_value=_fetch_result(bad)):
            with pytest.raises(DataProviderError, match="no finito"):
                provider.download(_config(), license_acceptance=acceptance)

    def test_infinite_obs_value_is_rejected(self):
        provider, acceptance = _accepted()
        bad = (
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,inf\n"
        )
        with _patched(return_value=_fetch_result(bad)):
            with pytest.raises(DataProviderError, match="no finito"):
                provider.download(_config(), license_acceptance=acceptance)

    def test_duplicate_date_is_rejected(self):
        provider, acceptance = _accepted()
        bad = (
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.1\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.2\n"
        )
        with _patched(return_value=_fetch_result(bad)):
            with pytest.raises(DataProviderError, match="repite la fecha"):
                provider.download(_config(), license_acceptance=acceptance)

    def test_row_date_outside_requested_range_is_rejected(self):
        provider, acceptance = _accepted()
        bad = (
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2030-01-02,1.1\n"
        )
        with _patched(return_value=_fetch_result(bad)):
            with pytest.raises(DataProviderError, match="fuera del rango"):
                provider.download(_config(), license_acceptance=acceptance)


class TestSecureFetchFailurePropagates:
    def test_secure_fetch_error_becomes_data_provider_error(self):
        provider, acceptance = _accepted()
        with _patched(side_effect=SecureFetchError("boom")):
            with pytest.raises(DataProviderError, match="BCE"):
                provider.download(_config(), license_acceptance=acceptance)


class TestLicenseGate:
    def test_download_without_acceptance_makes_zero_requests(self):
        with _patched() as mock_fetch:
            with pytest.raises(DataProviderError):
                EcbFxProvider().download(_config(), license_acceptance=None)
            mock_fetch.assert_not_called()


class TestHostAllowlist:
    def test_allowed_host_is_the_fixed_ecb_data_api(self):
        provider, acceptance = _accepted()
        with _patched(return_value=_fetch_result(_REAL_CSV)) as mock_fetch:
            provider.download(_config(), license_acceptance=acceptance)
        assert mock_fetch.call_args.kwargs["allowed_hosts"] == frozenset({"data-api.ecb.europa.eu"})

    def test_url_is_built_from_the_fixed_series_path(self):
        provider, acceptance = _accepted()
        with _patched(return_value=_fetch_result(_REAL_CSV)) as mock_fetch:
            provider.download(_config(), license_acceptance=acceptance)
        called_url = mock_fetch.call_args.args[0]
        assert called_url.startswith("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?")


class TestEstimate:
    def test_estimate_does_not_touch_the_network(self):
        with _patched() as mock_fetch:
            estimate = EcbFxProvider().estimate_download(_config())
        mock_fetch.assert_not_called()
        assert estimate.estimated_rows == 10


class TestAvailability:
    def test_check_availability_true_on_success(self):
        # check_availability() prueba el rango 2024-01-02..03 — el CSV debe
        # coincidir con ESE rango (la validación fila-a-fila rechaza fechas
        # fuera del rango pedido, ver TestUnexpectedSchema).
        probe_csv = (
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.0956\n"
            "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-03,1.0919\n"
        )
        with _patched(return_value=_fetch_result(probe_csv)):
            assert EcbFxProvider().check_availability() is True

    def test_check_availability_false_on_failure(self):
        with _patched(side_effect=SecureFetchError("down")):
            assert EcbFxProvider().check_availability() is False

    def test_check_availability_false_when_body_shape_is_wrong(self):
        """Mismo criterio que el fix de Stooq: un 200 con cuerpo de forma
        inesperada no debe declararse disponible."""
        with _patched(return_value=_fetch_result("<html>not csv</html>", content_type="text/html")):
            assert EcbFxProvider().check_availability() is False
