# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C6 — schema de plantilla (flujo B).

Una plantilla es un JSON que declara, de antemano, TODO lo que hace
falta para generar un proyecto sin que el usuario suba nada: de qué
proveedor de datos (C5) viene el CSV, qué pipeline (C3) se le aplica, y
el esquema FINAL (target + tipos/rangos/categorías) — a diferencia del
flujo A (C1-C4), donde el esquema se INFIERE de un CSV real subido, aquí
se DECLARA porque la plantilla ya conoce su propia fuente de datos.

`validate_template` es SOLO validación de forma (nunca toca la red ni
genera nada) — el generador de proyecto real es
`matrixai.training.template_project.generate_project_from_template`."""
from __future__ import annotations

import re
from typing import Any

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_VALID_STATES = frozenset({"draft", "published", "deprecated", "disabled"})
_VALID_DIFFICULTIES = frozenset({"principiante", "intermedio", "avanzado"})
_REQUIRED_LOCALES = frozenset({"es", "en"})
_REQUIRED_I18N_FIELDS = frozenset({"name", "description", "limitations"})
_REQUIRED_LICENSE_FIELDS = frozenset({
    "name", "url", "summary", "requires_attribution", "commercial_use_allowed",
})

_REQUIRED_TOP_LEVEL = frozenset({
    "id", "version", "state", "category", "difficulty", "provider_id",
    "requires_network", "license", "i18n", "provider_config", "target_column",
})
_OPTIONAL_TOP_LEVEL = frozenset({
    "pipeline_operations", "column_type_overrides", "column_range_overrides",
    "column_category_overrides",
})
_ALLOWED_TOP_LEVEL = _REQUIRED_TOP_LEVEL | _OPTIONAL_TOP_LEVEL


class TemplateValidationError(ValueError):
    """Plantilla con forma inválida — mensaje siempre accionable
    (invariante 7)."""


def validate_template(data: Any) -> list[str]:
    """Errores accionables (lista vacía si `data` es una plantilla
    válida). Nunca toca la red — valida solo la FORMA del JSON, incluida
    la existencia real del `provider_id` declarado (el registro de
    proveedores, `get_default_registry()`, es puro Python/sin red)."""
    if not isinstance(data, dict):
        return ["La plantilla debe ser un objeto JSON."]

    errors: list[str] = []
    unknown = set(data) - _ALLOWED_TOP_LEVEL
    if unknown:
        errors.append(f"Campos desconocidos: {sorted(unknown)}.")
    missing = _REQUIRED_TOP_LEVEL - set(data)
    if missing:
        errors.append(f"Campos obligatorios ausentes: {sorted(missing)}.")

    errors.extend(_validate_id(data.get("id")))
    errors.extend(_validate_version(data.get("version")))
    errors.extend(_validate_state(data.get("state")))
    errors.extend(_validate_non_empty_str("category", data.get("category")))
    errors.extend(_validate_difficulty(data.get("difficulty")))
    errors.extend(_validate_non_empty_str("target_column", data.get("target_column")))
    errors.extend(_validate_license(data.get("license")))
    errors.extend(_validate_i18n(data.get("i18n")))

    provider = None
    if "provider_id" in data:
        provider, provider_errors = _validate_provider_id(data.get("provider_id"))
        errors.extend(provider_errors)

    if "requires_network" in data:
        requires_network = data.get("requires_network")
        if not isinstance(requires_network, bool):
            errors.append("requires_network debe ser un booleano.")
        elif provider is not None and requires_network != provider.get_metadata().requires_network:
            errors.append(
                f"requires_network={requires_network!r} no coincide con el proveedor "
                f"{data.get('provider_id')!r} (requires_network="
                f"{provider.get_metadata().requires_network!r})."
            )

    if "provider_config" in data:
        provider_config = data.get("provider_config")
        if not isinstance(provider_config, dict):
            errors.append("provider_config debe ser un objeto.")
        elif provider is not None:
            config_errors = provider.validate_config(provider_config)
            errors.extend(f"provider_config: {e}" for e in config_errors)

    if "pipeline_operations" in data:
        errors.extend(_validate_pipeline_operations(data.get("pipeline_operations")))

    for key in ("column_type_overrides",):
        if key in data and not _is_str_to_str_dict(data.get(key)):
            errors.append(f"{key} debe ser un objeto {{columna: tipo}} de texto a texto.")
    if "column_range_overrides" in data and not _is_valid_range_overrides(data.get("column_range_overrides")):
        errors.append("column_range_overrides debe ser un objeto {columna: [min, max]} numérico.")
    if "column_category_overrides" in data and not _is_valid_category_overrides(data.get("column_category_overrides")):
        errors.append("column_category_overrides debe ser un objeto {columna: [valores de texto]}.")

    return errors


def _validate_id(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str) or not _ID_RE.match(value):
        return ["id debe ser texto en minúsculas/dígitos/guion bajo, empezando por una letra (p.ej. 'marina_ola')."]
    return []


def _validate_version(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str) or not _SEMVER_RE.match(value):
        return ["version debe ser semver 'MAJOR.MINOR.PATCH' (p.ej. '1.0.0')."]
    return []


def _validate_state(value: Any) -> list[str]:
    if value is None:
        return []
    if value not in _VALID_STATES:
        return [f"state debe ser uno de {sorted(_VALID_STATES)} (recibido {value!r})."]
    return []


def _validate_difficulty(value: Any) -> list[str]:
    if value is None:
        return []
    if value not in _VALID_DIFFICULTIES:
        return [f"difficulty debe ser uno de {sorted(_VALID_DIFFICULTIES)} (recibido {value!r})."]
    return []


def _validate_non_empty_str(field: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str) or not value.strip():
        return [f"{field} debe ser un texto no vacío."]
    return []


def _validate_license(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["license debe ser un objeto."]
    errors: list[str] = []
    unknown = set(value) - _REQUIRED_LICENSE_FIELDS
    if unknown:
        errors.append(f"license: campos desconocidos {sorted(unknown)}.")
    missing = _REQUIRED_LICENSE_FIELDS - set(value)
    if missing:
        errors.append(f"license: campos obligatorios ausentes {sorted(missing)}.")
        return errors
    for field in ("name", "summary"):
        if not isinstance(value[field], str) or not value[field].strip():
            errors.append(f"license.{field} debe ser un texto no vacío.")
    if not isinstance(value["url"], str):
        errors.append("license.url debe ser un texto (puede ser vacío si no aplica).")
    for field in ("requires_attribution", "commercial_use_allowed"):
        if not isinstance(value[field], bool):
            errors.append(f"license.{field} debe ser un booleano.")
    return errors


def _validate_i18n(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["i18n debe ser un objeto."]
    errors: list[str] = []
    if set(value) != _REQUIRED_LOCALES:
        errors.append(f"i18n debe declarar exactamente los locales {sorted(_REQUIRED_LOCALES)} (recibido {sorted(value)}).")
        return errors
    for locale, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != _REQUIRED_I18N_FIELDS:
            errors.append(f"i18n.{locale} debe tener exactamente los campos {sorted(_REQUIRED_I18N_FIELDS)}.")
            continue
        for field, text in entry.items():
            if not isinstance(text, str) or not text.strip():
                errors.append(f"i18n.{locale}.{field} debe ser un texto no vacío.")
    return errors


def _validate_provider_id(value: Any):
    if not isinstance(value, str) or not value.strip():
        return None, ["provider_id debe ser un texto no vacío."]
    from matrixai.training.data_provider import DataProviderError, get_default_registry
    try:
        provider = get_default_registry().get(value)
    except DataProviderError as exc:
        return None, [f"provider_id: {exc}"]
    return provider, []


def _validate_pipeline_operations(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["pipeline_operations debe ser una lista de operaciones."]
    from matrixai.training.dataset_pipeline import _OP_PARAM_KEYS  # noqa: PLC0415 — vocabulario cerrado, reusado tal cual

    errors: list[str] = []
    for i, op in enumerate(value):
        if not isinstance(op, dict) or "op" not in op:
            errors.append(f"pipeline_operations[{i}] debe ser un objeto con clave 'op'.")
            continue
        op_name = op.get("op")
        if op_name not in _OP_PARAM_KEYS:
            errors.append(
                f"pipeline_operations[{i}].op {op_name!r} desconocido — vocabulario cerrado: "
                f"{sorted(_OP_PARAM_KEYS)}."
            )
            continue
        allowed_params = _OP_PARAM_KEYS[op_name] | {"op"}
        unknown_params = set(op) - allowed_params
        if unknown_params:
            errors.append(f"pipeline_operations[{i}] ({op_name}): parámetros desconocidos {sorted(unknown_params)}.")
    return errors


def _is_str_to_str_dict(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _is_valid_range_overrides(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for k, v in value.items():
        if not isinstance(k, str):
            return False
        if (not isinstance(v, (list, tuple)) or len(v) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
            return False
    return True


def _is_valid_category_overrides(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for k, v in value.items():
        if not isinstance(k, str):
            return False
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return False
    return True
