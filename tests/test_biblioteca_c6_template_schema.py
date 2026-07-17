# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C6 — validador de plantillas."""
from __future__ import annotations

from matrixai.training.template_schema import validate_template


def _template(**overrides):
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
                {"name": "y", "type": "categorical", "categories": ["a", "b"]},
            ],
        },
        "target_column": "y",
    }
    base.update(overrides)
    return base


class TestValidTemplate:
    def test_valid_template_has_no_errors(self):
        assert validate_template(_template()) == []

    def test_optional_fields_can_be_present(self):
        errors = validate_template(_template(
            pipeline_operations=[{"op": "drop_duplicates"}],
            column_type_overrides={"x1": "number"},
            column_range_overrides={"x1": [0, 1]},
            column_category_overrides={"y": ["a", "b"]},
        ))
        assert errors == []


class TestNotADict:
    def test_non_dict_input_is_rejected(self):
        assert validate_template("not a dict") == ["La plantilla debe ser un objeto JSON."]
        assert validate_template([1, 2, 3]) != []
        assert validate_template(None) != []


class TestUnknownFields:
    def test_unknown_top_level_field_is_rejected(self):
        errors = validate_template(_template(unexpected_field="x"))
        assert any("desconocidos" in e for e in errors)


class TestMissingFields:
    def test_missing_required_field_is_rejected(self):
        template = _template()
        del template["target_column"]
        errors = validate_template(template)
        assert any("obligatorios ausentes" in e for e in errors)


class TestId:
    def test_uppercase_id_is_rejected(self):
        errors = validate_template(_template(id="Marina_Ola"))
        assert any("id" in e for e in errors)

    def test_id_starting_with_digit_is_rejected(self):
        errors = validate_template(_template(id="1abc"))
        assert any("id" in e for e in errors)

    def test_id_with_hyphen_is_rejected(self):
        errors = validate_template(_template(id="marina-ola"))
        assert any("id" in e for e in errors)


class TestVersion:
    def test_non_semver_version_is_rejected(self):
        errors = validate_template(_template(version="1.0"))
        assert any("version" in e for e in errors)

    def test_v_prefixed_version_is_rejected(self):
        errors = validate_template(_template(version="v1.0.0"))
        assert any("version" in e for e in errors)


class TestState:
    def test_invalid_state_is_rejected(self):
        errors = validate_template(_template(state="beta"))
        assert any("state" in e for e in errors)

    def test_disabled_state_is_a_valid_shape(self):
        # "disabled" es una forma válida — la exclusión de listado/creación
        # de proyecto es una decisión del catálogo/generador, no del schema.
        assert validate_template(_template(state="disabled")) == []


class TestDifficulty:
    def test_invalid_difficulty_is_rejected(self):
        errors = validate_template(_template(difficulty="experto"))
        assert any("difficulty" in e for e in errors)


class TestProviderId:
    def test_unknown_provider_is_rejected(self):
        errors = validate_template(_template(provider_id="no_existe"))
        assert any("provider_id" in e for e in errors)

    def test_real_provider_is_accepted(self):
        errors = validate_template(_template(provider_id="open_meteo", requires_network=True, provider_config={
            "dataset": "archive", "latitude": 1, "longitude": 1,
            "start_date": "2024-01-01", "end_date": "2024-01-02", "variables": ["temperature_2m_max"],
        }))
        assert errors == []


class TestRequiresNetworkConsistency:
    def test_mismatched_requires_network_is_rejected(self):
        """synthetic_local no usa red — declarar requires_network=True
        para él es una mentira que rompería el badge offline/red de la UI."""
        errors = validate_template(_template(requires_network=True))
        assert any("requires_network" in e for e in errors)

    def test_non_boolean_requires_network_is_rejected(self):
        errors = validate_template(_template(requires_network="yes"))
        assert any("requires_network" in e for e in errors)


class TestProviderConfig:
    def test_provider_config_is_validated_against_the_real_provider(self):
        """El provider_config se valida de verdad contra
        provider.validate_config — no solo "es un objeto"."""
        errors = validate_template(_template(provider_config={"seed": 1, "rows": -5, "columns": []}))
        assert errors != []

    def test_non_dict_provider_config_is_rejected(self):
        errors = validate_template(_template(provider_config="not a dict"))
        assert any("provider_config" in e for e in errors)


class TestLicense:
    def test_missing_license_field_is_rejected(self):
        template = _template()
        del template["license"]["url"]
        errors = validate_template(template)
        assert any("license" in e for e in errors)

    def test_unknown_license_field_is_rejected(self):
        template = _template()
        template["license"]["extra"] = "x"
        errors = validate_template(template)
        assert any("license" in e for e in errors)

    def test_empty_license_name_is_rejected(self):
        template = _template()
        template["license"]["name"] = ""
        errors = validate_template(template)
        assert any("license.name" in e for e in errors)

    def test_license_url_may_be_empty_string(self):
        # Un dataset propio/sin fuente externa puede no tener URL de licencia.
        errors = validate_template(_template())
        assert errors == []


class TestI18n:
    def test_missing_locale_is_rejected(self):
        template = _template()
        del template["i18n"]["en"]
        errors = validate_template(template)
        assert any("i18n" in e for e in errors)

    def test_extra_locale_is_rejected(self):
        template = _template()
        template["i18n"]["fr"] = template["i18n"]["es"]
        errors = validate_template(template)
        assert any("i18n" in e for e in errors)

    def test_missing_i18n_field_is_rejected(self):
        template = _template()
        del template["i18n"]["en"]["limitations"]
        errors = validate_template(template)
        assert any("i18n.en" in e for e in errors)

    def test_empty_i18n_text_is_rejected(self):
        template = _template()
        template["i18n"]["en"]["description"] = "   "
        errors = validate_template(template)
        assert any("i18n.en.description" in e for e in errors)


class TestPipelineOperations:
    def test_unknown_op_is_rejected(self):
        errors = validate_template(_template(pipeline_operations=[{"op": "delete_everything"}]))
        assert any("desconocido" in e for e in errors)

    def test_unknown_param_is_rejected(self):
        errors = validate_template(_template(
            pipeline_operations=[{"op": "sort_temporal", "column": "fecha", "bogus": 1}]
        ))
        assert any("parámetros desconocidos" in e for e in errors)

    def test_valid_ops_are_accepted(self):
        errors = validate_template(_template(pipeline_operations=[
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "y", "horizon": 1},
            {"op": "missing_values", "strategy": "drop"},
        ]))
        assert errors == []

    def test_non_list_pipeline_operations_is_rejected(self):
        errors = validate_template(_template(pipeline_operations={"op": "sort_temporal"}))
        assert any("pipeline_operations" in e for e in errors)


class TestOverridesShape:
    def test_non_dict_type_overrides_is_rejected(self):
        errors = validate_template(_template(column_type_overrides=["number"]))
        assert any("column_type_overrides" in e for e in errors)

    def test_invalid_range_overrides_shape_is_rejected(self):
        errors = validate_template(_template(column_range_overrides={"x1": [0]}))
        assert any("column_range_overrides" in e for e in errors)

    def test_invalid_category_overrides_shape_is_rejected(self):
        errors = validate_template(_template(column_category_overrides={"y": "a,b"}))
        assert any("column_category_overrides" in e for e in errors)
