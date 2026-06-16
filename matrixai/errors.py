# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""User-facing error messages with actionable suggestions."""


def error_init_project_exists(project_name: str, project_dir: str) -> str:
    """Project directory already exists."""
    return f"""\
Error: Project '{project_name}' already exists at {project_dir}

Options:
  1. Use a different project name:
     python3 -m matrixai init my-project-2 --template classification

  2. Remove the existing project:
     rm -rf {project_dir}
     python3 -m matrixai init {project_name} --template classification
"""


def error_init_template_not_found(template_name: str, available: list[str]) -> str:
    """Template not found."""
    available_list = "\n    ".join(f"- {t}" for t in available) if available else "(no templates found)"
    return f"""\
Error: Template '{template_name}' not found

Available templates:
    {available_list}

Usage:
  python3 -m matrixai init my-project --template classification
"""


def error_file_not_found(filepath: str, context: str) -> str:
    """File not found."""
    return f"""\
Error: File not found: {filepath}

{context}

Check:
  1. Is the file path correct?
  2. Are you in the right directory? (pwd)
  3. Does the file exist? (ls -la {filepath})
"""


def error_mxai_parse_error(filepath: str, parse_error: str) -> str:
    """Failed to parse .mxai file."""
    return f"""\
Error: Failed to parse {filepath}

{parse_error}

Check:
  1. Syntax: is the .mxai file valid MatrixAI syntax?
  2. Indentation: are NETWORK/LAYER/END blocks indented correctly?
  3. Example: see templates/classification/ for a valid example

  python3 -m matrixai parse {filepath}  # for detailed error
"""


def error_train_missing_files() -> str:
    """Missing training files."""
    return """\
Error: Missing required files for training

You need:
  1. A .mxai file (model architecture) — e.g., my-model.mxai
  2. A .mxtrain file (training config) — e.g., my-model.mxtrain
  3. A dataset CSV file — e.g., dataset/train.csv

Quick fix: create a new project with template:
  python3 -m matrixai init my-project --template classification
  cd my-project
  python3 -m matrixai train my-project.mxai --training my-project.mxtrain

Or see QUICKSTART.md for a full walkthrough.
"""


def error_params_not_found(params_path: str) -> str:
    """Trained parameters file not found."""
    return f"""\
Error: Parameters file not found: {params_path}

This file is created when you train a model:
  python3 -m matrixai train model.mxai --training model.mxtrain --output runs/v1

After training, use the output:
  python3 -m matrixai run model.mxai --params runs/v1/params.best.json --input input/sample.json

Did you train the model yet? (check if runs/v1/ exists)
"""


def error_serve_model_not_found(model_path: str) -> str:
    """Model file required by the HTTP server was not found."""
    return f"""\
Error: Model file not found: {model_path}

The HTTP server needs an existing .mxai project file.

Quick fix:
  python3 -m matrixai init my-first-classifier --template classification
  python3 -m matrixai train my-first-classifier/my-first-classifier.mxai --training my-first-classifier/my-first-classifier.mxtrain --output my-first-classifier/runs/v1
  python3 -m matrixai serve my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/v1/params.best.json --api-key dev-secret
"""


def error_serve_params_not_found(params_path: str) -> str:
    """Parameters file required by the HTTP server was not found."""
    return f"""\
Error: Parameters file not found: {params_path}

The HTTP server needs trained parameters from a previous train run.

Quick fix:
  python3 -m matrixai train my-first-classifier/my-first-classifier.mxai --training my-first-classifier/my-first-classifier.mxtrain --output my-first-classifier/runs/v1
  python3 -m matrixai serve my-first-classifier/my-first-classifier.mxai --params my-first-classifier/runs/v1/params.best.json --api-key dev-secret
"""


def error_invalid_input_format(input_str: str, expected_format: str) -> str:
    """Invalid input format."""
    return f"""\
Error: Invalid input format: {input_str}

Expected JSON format:
  {expected_format}

Example:
  python3 -m matrixai run model.mxai --params params.json --input input/sample.json

Check:
  python3 -m json.tool input/sample.json
"""


def error_network_not_found() -> str:
    """Network definition not found in .mxai."""
    return """\
Error: No NETWORK block found in .mxai file

A valid .mxai must have:
  NETWORK my_model
    INPUT
      field_name: Type
    END

    LAYER ...

    OUTPUT
      output_name: Type
    END
  END

See QUICKSTART.md or templates/classification/ for examples.
"""


def error_permission_denied(path: str) -> str:
    """Permission denied writing to directory."""
    return f"""\
Error: Permission denied writing to {path}

Try:
  1. Check directory permissions:
     ls -ld $(dirname {path})

  2. Write to a different location:
     python3 -m matrixai init my-project --output-dir ~/my-projects

  3. Use sudo if necessary:
     sudo python3 -m matrixai init my-project --output-dir {path}
"""


def error_server_port_in_use(port: int) -> str:
    """Server port already in use."""
    return f"""\
Error: Port {port} is already in use

Options:
  1. Stop the process using that port:
     lsof -i :{port}
     kill -9 <PID>

  2. Use a different port:
     python3 -m matrixai serve model.mxai --params params.json --port {port + 1}

  3. Try a different port range (8001-8010)
"""
