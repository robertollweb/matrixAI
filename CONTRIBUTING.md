# Contributing to MatrixAI

Thank you for your interest in contributing to MatrixAI. This document explains
how to propose changes, the expected code style, and the review process.

## Code of Conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior
through the channel described there.

## Contributor License Agreement (CLA)

MatrixAI core is licensed under the GNU Affero General Public License version 3
(AGPL-3.0-only). Before your first contribution can be merged, you must sign the
[Contributor License Agreement](CLA.md).

The signing process is automated: when you open your first pull request, a bot
will ask you to confirm acceptance by posting a comment. You only need to do
this once. The CLA preserves the maintainer's ability to license the project
under more than one license.

## Ways to Contribute

- **Report a bug**: open an issue using the bug report template.
- **Request a feature**: open an issue using the feature request template.
- **Ask a question**: open an issue using the question template.
- **Submit code**: open a pull request following the steps below.

## Development Setup

```bash
git clone https://github.com/robertollweb/matrixAI.git
cd matrixAI
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the Tests

The full suite must pass before a pull request can be merged:

```bash
python3 -m pytest -q
```

Some tests are skipped automatically when optional dependencies (such as
PyTorch) are not installed. That is expected.

## Code Style

- Target Python 3.10+.
- Follow PEP 8. Keep functions small and focused.
- Every source file must start with the license header:

  ```python
  # SPDX-License-Identifier: AGPL-3.0-only
  # Copyright (C) 2026 Roberto Llamosas Conde
  ```

- Prefer clear names over comments. Add comments only where intent is not
  obvious from the code.
- Do not introduce references to specific external providers by brand name.
  When describing compatibility with an external protocol or API format, use
  neutral, technical terms.

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your change with accompanying tests.
3. Ensure `python3 -m pytest -q` passes locally.
4. Sign the CLA when prompted on your pull request.
5. Open the pull request with a clear description of the change and the
   motivation behind it.
6. A maintainer will review your contribution. Address review feedback by
   pushing additional commits to the same branch.

## Reporting Security Issues

Do not open public issues for security vulnerabilities. Follow the process in
[SECURITY.md](SECURITY.md).
