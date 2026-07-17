# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C6 — flujo B: plantilla → proyecto.

Envoltorio DELGADO reutilizando C5 (`DataProvider.download`) + C3
(`run_pipeline`) + C2 (`generate_project_from_dataset`) — cero caminos
paralelos, mismo criterio que el envoltorio temporal de C4.

Diferencia deliberada con C4: el esquema de una plantilla es DECLARADO,
no inferido — quien escribe el JSON de la plantilla ya conoce su propia
fuente de datos, así que `target_column`/`column_type_overrides`/etc.
usan directamente los nombres de columna FINALES (ya transformados por
`pipeline_operations`, si los hay) — a diferencia del editor de esquema
del SPA (C4), que solo ve nombres crudos y necesitaba propagar
correcciones del target crudo al target desplazado. Una plantilla no
necesita esa propagación: quien la escribe puede declarar el override
bajo el nombre correcto directamente."""
from __future__ import annotations

from typing import Any

from matrixai.training.data_provider import DataProviderError, LicenseAcceptance
from matrixai.training.template_schema import validate_template


class TemplateProjectError(Exception):
    """Cualquier fallo generando un proyecto desde una plantilla —
    mensaje siempre accionable (invariante 7)."""


def generate_project_from_template(
    template: dict[str, Any], *, license_acceptance: LicenseAcceptance | None,
) -> dict[str, Any]:
    """Genera un proyecto MatrixAI completo A PARTIR de una plantilla.

    1. Valida la FORMA de la plantilla (`validate_template`) — nunca
       genera nada a partir de un JSON malformado, aunque el catálogo ya
       la haya validado al cargarla (defensa en profundidad: esta
       función es la única que de verdad ejecuta la plantilla).
    2. Descarga el CSV declarado vía el proveedor (C5) —
       `license_acceptance` se propaga tal cual a `provider.download`,
       que exige un recibo válido (invariante 2) antes de tocar la red.
    3. Si la plantilla declara `pipeline_operations`, las aplica (C3) —
       defensa en profundidad con `check_anti_leakage` tras generar,
       igual que C4.
    4. Entrega el CSV (transformado o crudo) + el esquema DECLARADO a
       `generate_project_from_dataset` (C2) — el resto del flujo
       (prompt tipado, generación, validación) es EXACTAMENTE el mismo
       que los flujos A y B/C4 (invariante 4: "proyecto normal").

    Devuelve el MISMO shape que `generate_project_from_dataset`, con la
    procedencia extendida: `source="template"`, `template_id`,
    `template_version`, `provider_id`, `provider_download` (metadata de
    la descarga real: url, fecha, filas, recibo de licencia), y las
    operaciones del pipeline (si las hay) antepuestas a `operations`.

    Lanza `TemplateProjectError` si la plantilla es inválida, está
    deshabilitada (`state="disabled"`), la descarga falla, el pipeline
    falla, se detecta fuga temporal, o la generación (C2) falla."""
    errors = validate_template(template)
    if errors:
        raise TemplateProjectError("Plantilla inválida: " + "; ".join(errors))
    if template["state"] == "disabled":
        raise TemplateProjectError(
            f"La plantilla {template['id']!r} está deshabilitada — no se puede generar un proyecto desde ella."
        )

    from matrixai.training.data_provider import get_default_registry

    provider = get_default_registry().get(template["provider_id"])
    try:
        download = provider.download(template["provider_config"], license_acceptance=license_acceptance)
    except DataProviderError as exc:
        raise TemplateProjectError(f"Descarga fallida: {exc}") from exc

    pipeline_operations = template.get("pipeline_operations") or []
    pipeline_result = None
    prepared_csv = download.csv_text
    if pipeline_operations:
        from matrixai.training.dataset_pipeline import PipelineError, run_pipeline
        from matrixai.training.dataset_project import _read_rows, _rows_to_csv_text

        rows = _read_rows(download.csv_text)
        try:
            pipeline_result = run_pipeline(rows, pipeline_operations)
        except PipelineError as exc:
            raise TemplateProjectError(f"Pipeline: {exc}") from exc
        if not pipeline_result.rows:
            raise TemplateProjectError(
                "El pipeline de la plantilla no deja ninguna fila — revisa los parámetros "
                "(ventana/horizonte demasiado grandes para el dataset descargado)."
            )
        prepared_csv = _rows_to_csv_text(pipeline_result.rows)

    from matrixai.training.dataset_project import DatasetProjectError, generate_project_from_dataset

    try:
        result = generate_project_from_dataset(
            prepared_csv, template["target_column"],
            column_type_overrides=template.get("column_type_overrides") or None,
            column_range_overrides=_as_range_tuples(template.get("column_range_overrides")),
            column_category_overrides=template.get("column_category_overrides") or None,
        )
    except DatasetProjectError as exc:
        raise TemplateProjectError(f"Generación: {exc}") from exc

    if pipeline_result is not None:
        from matrixai.training.dataset_pipeline import check_anti_leakage

        feature_columns = list(result["provenance"]["feature_name_map"].keys())
        leaks = check_anti_leakage(pipeline_result, feature_columns)
        if leaks:
            raise TemplateProjectError("Fuga temporal: " + "; ".join(leaks))

    prov = result["provenance"]
    prov["source"] = "template"
    prov["template_id"] = template["id"]
    prov["template_version"] = template["version"]
    prov["provider_id"] = template["provider_id"]
    prov["provider_download"] = {
        "source_url": download.source_url,
        "fetched_at": download.fetched_at,
        "rows": download.rows,
        "license_acceptance": license_acceptance.to_dict() if license_acceptance else None,
    }
    if pipeline_result is not None:
        prov["operations"] = [s.operation for s in pipeline_result.steps] + prov["operations"]

    return result


def _as_range_tuples(overrides: dict[str, Any] | None) -> dict[str, tuple[float, float]] | None:
    if not overrides:
        return None
    return {k: (float(v[0]), float(v[1])) for k, v in overrides.items()}
