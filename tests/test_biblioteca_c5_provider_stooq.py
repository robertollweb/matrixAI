# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `stooq`.

Cero red real: `secure_fetch` se sustituye por completo vía
`unittest.mock.patch`.

`_REAL_ANTIBOT_HTML` es una respuesta REAL grabada (curl directo a
`stooq.com/q/d/l/?s=aapl.us&...`, 2026-07-17, ver docstring de
`provider_stooq.py`): stooq.com gatea HOY toda petición no-navegador tras
un reto JS de prueba-de-trabajo — este test confirma que el proveedor la
rechaza limpiamente como esquema inesperado, no que la descarga funcione
de verdad contra el servicio real (bloqueada externamente, no un bug de
este código). `_TYPICAL_CSV` es una reconstrucción del formato documentado
de Stooq (`Date,Open,High,Low,Close,Volume`), no una captura en vivo."""
from __future__ import annotations

import unittest.mock

import pytest

from matrixai.training.data_provider import DataProviderError
from matrixai.training.provider_stooq import StooqProvider
from matrixai.training.secure_fetch import SecureFetchError, SecureFetchResult

_TYPICAL_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,187.15,188.44,183.89,184.25,82488700\n"
    "2024-01-03,184.22,185.88,183.43,184.05,58414500\n"
)

# Respuesta REAL grabada, 2026-07-17 (recortada): stooq.com responde con un
# reto JS de prueba-de-trabajo a CUALQUIER petición sin motor JS, incluso
# con User-Agent de navegador — ver docstring del módulo del proveedor.
_REAL_ANTIBOT_HTML = (
    '<!DOCTYPE html><html><head><meta charset="utf-8">'
    '<meta name="robots" content="noindex,nofollow"></head><body>'
    '<noscript>This site requires JavaScript to verify your browser. '
    'Please enable JavaScript and reload.</noscript>'
    '<script nonce="0q-ErOb2HlIeeDfyt_mS8g">(async()=>{...})();</script>'
    "</body></html>"
)


def _config(**overrides):
    base = {"symbol": "aapl.us", "start_date": "2024-01-02", "end_date": "2024-01-03", "interval": "d"}
    base.update(overrides)
    return base


def _fetch_result(body: str, content_type: str = "text/csv") -> SecureFetchResult:
    return SecureFetchResult(
        url="https://stooq.com/q/d/l/?...", status=200,
        body=body.encode("utf-8"), content_type=content_type,
    )


def _patched(**kwargs):
    return unittest.mock.patch("matrixai.training.provider_stooq.secure_fetch", **kwargs)


class TestValidateConfig:
    def test_valid_config_has_no_errors(self):
        assert StooqProvider().validate_config(_config()) == []

    def test_empty_symbol_is_rejected(self):
        errors = StooqProvider().validate_config(_config(symbol=""))
        assert any("symbol" in e for e in errors)

    def test_invalid_interval_is_rejected(self):
        errors = StooqProvider().validate_config(_config(interval="y"))
        assert any("interval" in e for e in errors)

    def test_start_after_end_is_rejected(self):
        errors = StooqProvider().validate_config(_config(start_date="2024-06-01", end_date="2024-01-01"))
        assert any("start_date" in e for e in errors)


class TestDownloadHappyPath:
    def test_returns_canonical_csv(self):
        with _patched(return_value=_fetch_result(_TYPICAL_CSV)) as mock_fetch:
            result = StooqProvider().download(_config(), license_accepted=True)
        assert mock_fetch.call_count == 1
        assert result.rows == 2
        assert result.columns == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert "2024-01-02,187.15" in result.csv_text

    def test_license_info_is_never_commercial(self):
        info = StooqProvider().get_license_info()
        assert info.commercial_use_allowed is False


class TestUnexpectedSchema:
    def test_real_antibot_challenge_page_is_rejected_cleanly(self):
        """El caso central descubierto en este corte: stooq.com devuelve
        un reto JS en vez de CSV — el proveedor debe fallar limpio
        (invariante 7), nunca intentar parsear HTML como CSV."""
        with _patched(return_value=_fetch_result(_REAL_ANTIBOT_HTML, content_type="text/html")):
            with pytest.raises(DataProviderError, match="forma inesperada"):
                StooqProvider().download(_config(), license_accepted=True)

    def test_no_data_response_for_unknown_symbol_is_rejected(self):
        with _patched(return_value=_fetch_result("No data\n")):
            with pytest.raises(DataProviderError, match="forma inesperada"):
                StooqProvider().download(_config(symbol="thisisnotarealsymbolxyz"), license_accepted=True)

    def test_header_only_no_rows_is_rejected(self):
        with _patched(return_value=_fetch_result("Date,Open,High,Low,Close,Volume\n")):
            with pytest.raises(DataProviderError, match="ninguna cotización"):
                StooqProvider().download(_config(), license_accepted=True)

    def test_empty_body_is_rejected(self):
        with _patched(return_value=_fetch_result("")):
            with pytest.raises(DataProviderError, match="forma inesperada"):
                StooqProvider().download(_config(), license_accepted=True)


class TestSecureFetchFailurePropagates:
    def test_secure_fetch_error_becomes_data_provider_error(self):
        with _patched(side_effect=SecureFetchError("timeout")):
            with pytest.raises(DataProviderError, match="Stooq"):
                StooqProvider().download(_config(), license_accepted=True)


class TestLicenseGate:
    def test_download_without_acceptance_makes_zero_requests(self):
        with _patched() as mock_fetch:
            with pytest.raises(DataProviderError, match="exige aceptar su licencia"):
                StooqProvider().download(_config(), license_accepted=False)
            mock_fetch.assert_not_called()


class TestHostAllowlist:
    def test_allowed_hosts_is_stooq_com_only(self):
        with _patched(return_value=_fetch_result(_TYPICAL_CSV)) as mock_fetch:
            StooqProvider().download(_config(), license_accepted=True)
        allowed = mock_fetch.call_args.kwargs["allowed_hosts"]
        assert allowed == frozenset({"stooq.com"})


class TestEstimate:
    def test_estimate_does_not_touch_the_network(self):
        with _patched() as mock_fetch:
            estimate = StooqProvider().estimate_download(_config())
        mock_fetch.assert_not_called()
        assert estimate.estimated_rows == 2


class TestAvailability:
    def test_check_availability_reflects_the_real_antibot_gate(self):
        """Documenta el estado real descubierto: HOY `check_availability`
        devuelve False contra el servicio real (ver TestUnexpectedSchema)
        — aquí solo se verifica el mecanismo con un mock que simula ese
        mismo resultado."""
        with _patched(side_effect=SecureFetchError("blocked")):
            assert StooqProvider().check_availability() is False

    def test_check_availability_true_when_the_service_responds_normally(self):
        with _patched(return_value=_fetch_result(_TYPICAL_CSV)):
            assert StooqProvider().check_availability() is True
