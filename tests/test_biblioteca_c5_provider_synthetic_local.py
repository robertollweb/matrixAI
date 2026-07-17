# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `synthetic_local`."""
from __future__ import annotations

import csv
import io

import pytest

from matrixai.training.data_provider import DataProviderError
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


class TestDeterminism:
    def test_same_seed_produces_byte_identical_csv(self):
        provider = SyntheticLocalProvider()
        r1 = provider.download(_config(), license_accepted=True)
        r2 = provider.download(_config(), license_accepted=True)
        assert r1.csv_text == r2.csv_text

    def test_different_seed_produces_different_csv(self):
        provider = SyntheticLocalProvider()
        r1 = provider.download(_config(seed=1), license_accepted=True)
        r2 = provider.download(_config(seed=2), license_accepted=True)
        assert r1.csv_text != r2.csv_text


class TestDownloadShape:
    def test_returns_the_declared_number_of_rows(self):
        result = SyntheticLocalProvider().download(_config(rows=25), license_accepted=True)
        assert result.rows == 25
        rows = list(csv.reader(io.StringIO(result.csv_text)))
        assert len(rows) - 1 == 25  # cabecera + 25 filas

    def test_columns_match_declared_names_and_order(self):
        result = SyntheticLocalProvider().download(_config(), license_accepted=True)
        assert result.columns == ["edad", "temperatura", "activo", "region", "fecha"]

    def test_integer_range_is_respected(self):
        result = SyntheticLocalProvider().download(_config(rows=200), license_accepted=True)
        reader = csv.DictReader(io.StringIO(result.csv_text))
        for row in reader:
            assert 18 <= int(row["edad"]) <= 90

    def test_categorical_values_come_from_declared_categories(self):
        result = SyntheticLocalProvider().download(_config(rows=50), license_accepted=True)
        reader = csv.DictReader(io.StringIO(result.csv_text))
        values = {row["region"] for row in reader}
        assert values <= {"norte", "sur"}

    def test_date_column_is_sequential_from_start(self):
        result = SyntheticLocalProvider().download(_config(rows=3), license_accepted=True)
        reader = csv.DictReader(io.StringIO(result.csv_text))
        dates = [row["fecha"] for row in reader]
        assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_source_url_is_none_no_network_involved(self):
        result = SyntheticLocalProvider().download(_config(), license_accepted=True)
        assert result.source_url is None


class TestLicenseGate:
    def test_download_without_acceptance_raises_and_generates_nothing(self):
        with pytest.raises(DataProviderError, match="exige aceptar su licencia"):
            SyntheticLocalProvider().download(_config(), license_accepted=False)


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
