# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `synthetic_local`."""
from __future__ import annotations

import csv
import io

import pytest

from matrixai.training.data_provider import DataProviderError, LicenseAcceptanceStore
from matrixai.training.provider_synthetic_local import SyntheticLocalProvider


def _config(**overrides):
    base = {
        "seed": 42,
        "rows": 10,
        "columns": [
            {"name": "edad", "type": "integer", "range": [18, 90]},
            {"name": "temperatura", "type": "number", "range": [10.0, 30.0]},
            {"name": "activo", "type": "boolean"},
            {"name": "region", "type": "categorical", "categories": ["norte", "sur"]},
            {"name": "fecha", "type": "date", "date_start": "2024-01-01", "date_step_days": 1},
        ],
    }
    base.update(overrides)
    return base


def _accepted():
    provider = SyntheticLocalProvider()
    store = LicenseAcceptanceStore()
    return provider, store.record(provider, actor="test")


class TestValidateConfig:
    def test_valid_config_has_no_errors(self):
        assert SyntheticLocalProvider().validate_config(_config()) == []

    def test_missing_seed_is_rejected(self):
        config = _config()
        del config["seed"]
        errors = SyntheticLocalProvider().validate_config(config)
        assert any("seed" in e for e in errors)

    def test_zero_rows_is_rejected(self):
        errors = SyntheticLocalProvider().validate_config(_config(rows=0))
        assert any("rows" in e for e in errors)

    def test_unknown_column_type_is_rejected(self):
        errors = SyntheticLocalProvider().validate_config(
            _config(columns=[{"name": "x", "type": "vector3"}])
        )
        assert any("type" in e for e in errors)

    def test_duplicate_column_names_are_rejected(self):
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "x", "type": "number", "range": [0, 1]},
            {"name": "x", "type": "number", "range": [0, 1]},
        ]))
        assert any("duplicado" in e for e in errors)

    def test_categorical_needs_at_least_two_categories(self):
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "c", "type": "categorical", "categories": ["solo_una"]},
        ]))
        assert any("categories" in e for e in errors)

    def test_invalid_range_min_gte_max_is_rejected(self):
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "x", "type": "number", "range": [10, 5]},
        ]))
        assert any("range" in e for e in errors)


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 (ronda 2) [ALTA] — validate_config() aceptaba
# configuraciones que generaban datos incorrectos o un fallo sin envolver.
# ---------------------------------------------------------------------------

class TestIntegerRangeMustHaveIntegerBounds:
    def test_non_integer_bounds_for_integer_type_are_rejected(self):
        """[0.9, 1.1] pasaba validate_config() y luego randint(int(0.9),
        int(1.1)) == randint(0, 1) truncaba en silencio a un rango
        DISTINTO del declarado — 0 queda fuera de [0.9, 1.1]."""
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "x", "type": "integer", "range": [0.9, 1.1]},
        ]))
        assert any("enteros" in e for e in errors)

    def test_integer_bounds_that_are_integer_valued_floats_are_accepted(self):
        # 18.0/90.0 son floats pero de VALOR entero — no deben rechazarse.
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "x", "type": "integer", "range": [18.0, 90.0]},
        ]))
        assert errors == []


class TestNonFiniteRangeIsRejected:
    def test_nan_range_is_rejected(self):
        """isinstance(nan, float) es True y `nan >= hi` siempre es False,
        así que [nan, 5] "pasaba" la comprobación min<max sin ser un
        rango real."""
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "x", "type": "number", "range": [float("nan"), 5]},
        ]))
        assert any("finito" in e for e in errors)

    def test_infinite_range_is_rejected(self):
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "x", "type": "number", "range": [0, float("inf")]},
        ]))
        assert any("finito" in e for e in errors)


class TestDateStepDaysIsValidated:
    def test_non_integer_date_step_days_is_rejected(self):
        """"bad" pasaba validate_config() limpio y explotaba con
        ValueError sin envolver dentro de _sample() — un 500, no un
        error accionable."""
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "d", "type": "date", "date_start": "2024-01-01", "date_step_days": "bad"},
        ]))
        assert any("date_step_days" in e for e in errors)

    def test_zero_or_negative_date_step_days_is_rejected(self):
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "d", "type": "date", "date_start": "2024-01-01", "date_step_days": 0},
        ]))
        assert any("date_step_days" in e for e in errors)

    def test_download_never_raises_an_unwrapped_value_error(self):
        provider, acceptance = _accepted()
        with pytest.raises(DataProviderError, match="date_step_days"):
            provider.download(_config(columns=[
                {"name": "d", "type": "date", "date_start": "2024-01-01", "date_step_days": "bad"},
            ]), license_acceptance=acceptance)


class TestDuplicateCategoriesAreRejected:
    def test_duplicate_categories_are_rejected(self):
        """["a","a"] cumplía "al menos 2 textos" pero produce una columna
        CONSTANTE (rng.choice siempre devuelve "a") — no aporta señal."""
        errors = SyntheticLocalProvider().validate_config(_config(columns=[
            {"name": "c", "type": "categorical", "categories": ["a", "a"]},
        ]))
        assert any("duplicados" in e for e in errors)


class TestDeterminism:
    def test_same_seed_produces_byte_identical_csv(self):
        provider, acceptance = _accepted()
        r1 = provider.download(_config(), license_acceptance=acceptance)
        r2 = provider.download(_config(), license_acceptance=acceptance)
        assert r1.csv_text == r2.csv_text

    def test_different_seed_produces_different_csv(self):
        provider, acceptance = _accepted()
        r1 = provider.download(_config(seed=1), license_acceptance=acceptance)
        r2 = provider.download(_config(seed=2), license_acceptance=acceptance)
        assert r1.csv_text != r2.csv_text


class TestDownloadShape:
    def test_returns_the_declared_number_of_rows(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(rows=25), license_acceptance=acceptance)
        assert result.rows == 25
        rows = list(csv.reader(io.StringIO(result.csv_text)))
        assert len(rows) - 1 == 25  # cabecera + 25 filas

    def test_columns_match_declared_names_and_order(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(), license_acceptance=acceptance)
        assert result.columns == ["edad", "temperatura", "activo", "region", "fecha"]

    def test_integer_range_is_respected(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(rows=200), license_acceptance=acceptance)
        reader = csv.DictReader(io.StringIO(result.csv_text))
        for row in reader:
            assert 18 <= int(row["edad"]) <= 90

    def test_categorical_values_come_from_declared_categories(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(rows=50), license_acceptance=acceptance)
        reader = csv.DictReader(io.StringIO(result.csv_text))
        values = {row["region"] for row in reader}
        assert values <= {"norte", "sur"}

    def test_date_column_is_sequential_from_start(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(rows=3), license_acceptance=acceptance)
        reader = csv.DictReader(io.StringIO(result.csv_text))
        dates = [row["fecha"] for row in reader]
        assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_source_url_is_none_no_network_involved(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(), license_acceptance=acceptance)
        assert result.source_url is None

    def test_provenance_extra_carries_the_acceptance_receipt(self):
        provider, acceptance = _accepted()
        result = provider.download(_config(), license_acceptance=acceptance)
        assert result.provenance_extra["license_acceptance"] == acceptance.to_dict()


class TestLicenseGate:
    def test_download_without_acceptance_raises_and_generates_nothing(self):
        with pytest.raises(DataProviderError, match="exige un recibo"):
            SyntheticLocalProvider().download(_config(), license_acceptance=None)


class TestEstimate:
    def test_estimate_matches_declared_rows(self):
        estimate = SyntheticLocalProvider().estimate_download(_config(rows=100))
        assert estimate.estimated_rows == 100

    def test_estimate_rejects_invalid_config(self):
        with pytest.raises(DataProviderError):
            SyntheticLocalProvider().estimate_download(_config(rows=-1))


class TestAvailabilityAndMetadata:
    def test_always_available_no_network_dependency(self):
        assert SyntheticLocalProvider().check_availability() is True

    def test_metadata_declares_no_network(self):
        assert SyntheticLocalProvider().get_metadata().requires_network is False
