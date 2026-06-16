# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Project scaffolding: matrixai init <project_name> --template <template_name>."""

import shutil
import sys
from pathlib import Path
from string import Template


class ScaffoldError(Exception):
    pass


def get_templates_dir() -> Path:
    """Return the templates directory path."""
    return Path(__file__).parent / "templates"


def list_templates() -> list[str]:
    """List available templates."""
    templates_dir = get_templates_dir()
    if not templates_dir.exists():
        return []
    return [d.name for d in templates_dir.iterdir() if d.is_dir()]


def scaffold_project(
    project_name: str,
    template_name: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Generate a new project from a template.

    Args:
        project_name: Name of the new project (used in filenames)
        template_name: Name of the template to use (e.g., 'classification')
        output_dir: Directory where the project will be created (default: cwd)

    Returns:
        Path to the newly created project directory

    Raises:
        ScaffoldError: If template not found or project already exists
    """
    from matrixai.errors import error_init_template_not_found, error_init_project_exists, error_permission_denied

    templates_dir = get_templates_dir()
    template_dir = templates_dir / template_name

    if not template_dir.exists():
        available = list_templates()
        raise ScaffoldError(error_init_template_not_found(template_name, available))

    requested_project = Path(project_name)
    resolved_project_name = requested_project.name
    if not resolved_project_name:
        raise ScaffoldError("Error: Project name must include a directory name")

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)

    project_dir = requested_project if requested_project.is_absolute() else output_dir / requested_project

    if project_dir.exists():
        raise ScaffoldError(error_init_project_exists(resolved_project_name, str(project_dir)))

    # Check write permissions
    try:
        project_dir.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ScaffoldError(error_permission_denied(str(output_dir))) from None

    # Copy template to project directory
    shutil.copytree(template_dir, project_dir)

    # Process templates: rename files and substitute $project_name in content
    for file_path in list(project_dir.rglob("*")):
        if file_path.is_file():
            # Rename files with {{project_name}}
            if "{{project_name}}" in file_path.name:
                new_name = file_path.name.replace("{{project_name}}", resolved_project_name)
                new_path = file_path.parent / new_name
                file_path = file_path.rename(new_path)

            # Substitute content in .mxai, .mxtrain, .md, and .csv files
            if file_path.suffix in (".mxai", ".mxtrain", ".md", ".csv"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    template = Template(content)
                    substituted = template.safe_substitute(project_name=resolved_project_name)
                    file_path.write_text(substituted, encoding="utf-8")
                except Exception as e:
                    raise ScaffoldError(
                        f"Error processing template file {file_path}: {e}"
                    ) from e

    return project_dir
