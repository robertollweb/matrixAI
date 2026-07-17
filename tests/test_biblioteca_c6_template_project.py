# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C6 — flujo B: plantilla → proyecto.

`generate_project_from_template`: envoltorio delgado sobre C5 (descarga)
+ C3 (pipeline opcional) + C2 (generación, cero caminos paralelos).
Cubre el caso sintético E2E (determinista, sin red), el caso temporal
(pipeline con shift_target, target ya en nombre transformado), paridad
con la llamada directa a C2, defensa anti-fuga, y errores accionables de
cada etapa."""
from __future__ import annotations

import pytest

from matrixai.training.data_provider import LicenseAcceptanceStore, get_default_registry
from matrixai.training.dataset_project import generate_project_from_dataset
from matrixai.training.template_project import TemplateProjectError, generate_project_from_template


def _tabular_template(**overrides):
    base = {
        "id": "clasificacion_sintetica",
        "version": "1.0.0",
        "state": "published",
        "category": "tabular",
        "difficulty": "principiante",
        "provider_id": "synthetic_local",
        "requires_network": False,
        "license": {
            "name": "Datos sintéticos propios", "url": "", "summary": "Sin restricciones.",
            "requires_attribution": False, "commercial_use_allowed": True,
        },
        "i18n": {
            "es": {"name": "Clasificación sintética", "description": "Ejemplo tabular.", "limitations": "Ninguna."},
            "en": {"name": "Synthetic classification", "description": "Tabular example.", "limitations": "None."},
        },
        "provider_config": {
            "seed": 42, "rows": 100,
            "columns": [
                {"name": "x1", "type": "number", "range": [0, 1]},
                {"name": "x2", "type": "number", "range": [0, 1]},
                {"name": "y", "type": "categorical", "categories": ["a", "b"]},
            ],
        },
        "target_column": "y",
    }
    base.update(overrides)
    return base


def _temporal_template(**overrides):
    base = {
        "id": "serie_sintetica",
        "version": "1.0.0",
        "state": "published",
        "category": "series_temporales",
        "difficulty": "intermedio",
        "provider_id": "synthetic_local",
        "requires_network": False,
        "license": {
            "name": "Datos sintéticos propios", "url": "", "summary": "Sin restricciones.",
            "requires_attribution": False, "commercial_use_allowed": True,
        },
        "i18n": {
            "es": {"name": "Serie sintética", "description": "Ejemplo temporal.", "limitations": "Ninguna."},
            "en": {"name": "Synthetic series", "description": "Temporal example.", "limitations": "None."},
        },
        "provider_config": {
            "seed": 7, "rows": 30,
            "columns": [
                {"name": "fecha", "type": "date", "date_start": "2024-01-01", "date_step_days": 1},
                {"name": "valor", "type": "number", "range": [0, 100]},
            ],
        },
        "pipeline_operations": [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "valor", "horizon": 1},
            {"op": "missing_values", "strategy": "drop"},
        ],
        "target_column": "valor_target_h1",
    }
    base.update(overrides)
    return base


def _accepted(provider_id: str = "synthetic_local"):
    provider = get_default_registry().get(provider_id)
    store = LicenseAcceptanceStore()
    return store.record(provider, actor="test")


class TestSyntheticTemplateE2E:
    def test_generates_a_full_project(self):
        result = generate_project_from_template(_tabular_template(), license_acceptance=_accepted())
        assert result["ok"]
        assert "x1: Scalar" in result["mxai"]
        assert "OUTPUT predicted_class" in result["mxai"] or "predicted_class" in result["mxai"]

    def test_provenance_records_template_and_provider_info(self):
        result = generate_project_from_template(_tabular_template(), license_acceptance=_accepted())
        prov = result["provenance"]
        assert prov["source"] == "template"
        assert prov["template_id"] == "clasificacion_sintetica"
        assert prov["template_version"] == "1.0.0"
        assert prov["provider_id"] == "synthetic_local"
        assert prov["provider_download"]["rows"] == 100
        assert prov["provider_download"]["source_url"] is None
        assert prov["provider_download"]["license_acceptance"]["provider_id"] == "synthetic_local"

    def test_deterministic_same_seed_produces_same_project(self):
        r1 = generate_project_from_template(_tabular_template(), license_acceptance=_accepted())
        r2 = generate_project_from_template(_tabular_template(), license_acceptance=_accepted())
        assert r1["mxai"] == r2["mxai"]


class TestParityWithDirectC2Call:
    def test_same_schema_via_template_or_direct_c2_produces_the_same_mxai(self):
        """El generador de C6 no debe reimplementar NADA de C2 — mismo
        CSV + mismo esquema declarado → mismo mxai, venga por donde venga."""
        template = _tabular_template()
        from matrixai.training.provider_synthetic_local import SyntheticLocalProvider
        raw_csv = SyntheticLocalProvider().download(
            template["provider_config"], license_acceptance=_accepted(),
        ).csv_text

        direct = generate_project_from_dataset(raw_csv, template["target_column"])
        via_template = generate_project_from_template(template, license_acceptance=_accepted())
        assert direct["mxai"] == via_template["mxai"]


class TestTemporalTemplateWithPipeline:
    def test_generates_a_temporal_project(self):
        result = generate_project_from_template(_temporal_template(), license_acceptance=_accepted())
        assert result["ok"]
        assert "valor: Scalar" in result["mxai"]  # sigue siendo feature tras el shift
        assert "fecha" not in result["mxai"]  # la columna temporal cruda nunca es feature

    def test_provenance_operations_include_pipeline_steps_first(self):
        result = generate_project_from_template(_temporal_template(), license_acceptance=_accepted())
        ops = result["provenance"]["operations"]
        assert ops[:3] == ["sort_temporal", "shift_target", "missing_values"]

    def test_rows_dropped_for_the_horizon_edge(self):
        result = generate_project_from_template(_temporal_template(), license_acceptance=_accepted())
        # 30 filas - 1 (sin futuro para horizon=1) = 29
        assert len(result["csv_text"].splitlines()) - 1 == 29

    def test_empty_pipeline_result_raises_actionable_error(self):
        template = _temporal_template(provider_config={
            "seed": 1, "rows": 1,
            "columns": [
                {"name": "fecha", "type": "date", "date_start": "2024-01-01"},
                {"name": "valor", "type": "number", "range": [0, 1]},
            ],
        })
        with pytest.raises(TemplateProjectError, match="ninguna fila"):
            generate_project_from_template(template, license_acceptance=_accepted())


class TestAntiLeakageDefenseInDepth:
    def test_a_lag_smaller_than_the_horizon_leaves_a_residual_leak(self):
        """Un uso indebido de la plantilla debe detectarse igual que en
        C4 — defensa en profundidad, no confiar solo en que la plantilla
        esté "bien escrita". shift_target(horizon=3) + lag_window(k=1)
        sobre la propia columna desplazada deja un desplazamiento neto
        de +2 (ni +3 ni 0) — sigue siendo fuga."""
        template = _temporal_template(
            pipeline_operations=[
                {"op": "sort_temporal", "column": "fecha"},
                {"op": "shift_target", "column": "valor", "horizon": 3},
                {"op": "lag_window", "columns": ["valor_target_h3"], "window": 1},
                {"op": "missing_values", "strategy": "drop"},
            ],
            target_column="valor_target_h3",
        )
        with pytest.raises(TemplateProjectError, match="Fuga temporal"):
            generate_project_from_template(template, license_acceptance=_accepted())

    def test_a_lag_exactly_matching_the_horizon_cancels_out_no_leak(self):
        """Caso simétrico (ya conocido de la reauditoría de C3): un lag
        de exactamente el mismo tamaño que el horizonte CANCELA el
        desplazamiento a 0 — es matemáticamente el valor presente bajo
        otro nombre, no una fuga."""
        template = _temporal_template(
            pipeline_operations=[
                {"op": "sort_temporal", "column": "fecha"},
                {"op": "shift_target", "column": "valor", "horizon": 1},
                {"op": "lag_window", "columns": ["valor_target_h1"], "window": 1},
                {"op": "missing_values", "strategy": "drop"},
            ],
            target_column="valor_target_h1",
        )
        result = generate_project_from_template(template, license_acceptance=_accepted())
        assert result["ok"]


class TestDisabledTemplate:
    def test_disabled_template_is_rejected(self):
        template = _tabular_template(state="disabled")
        with pytest.raises(TemplateProjectError, match="deshabilitada"):
            generate_project_from_template(template, license_acceptance=_accepted())


class TestInvalidTemplate:
    def test_malformed_template_is_rejected_before_touching_the_provider(self):
        template = _tabular_template()
        del template["target_column"]
        with pytest.raises(TemplateProjectError, match="Plantilla inválida"):
            generate_project_from_template(template, license_acceptance=_accepted())


class TestLicenseGate:
    def test_download_without_acceptance_is_rejected(self):
        with pytest.raises(TemplateProjectError, match="Descarga fallida"):
            generate_project_from_template(_tabular_template(), license_acceptance=None)


class TestTargetGenerationFailurePropagates:
    def test_unknown_target_column_is_a_template_project_error(self):
        template = _tabular_template(target_column="no_existe")
        with pytest.raises(TemplateProjectError, match="Generación"):
            generate_project_from_template(template, license_acceptance=_accepted())
