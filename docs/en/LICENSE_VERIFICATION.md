# License Verification

Status: verified for PR5-C1.

MatrixAI uses the GNU Affero General Public License version 3 (AGPL v3) for the core repository. This document records the verification performed before distribution work continues.

## Sources Checked

- GNU canonical AGPL v3 text: https://www.gnu.org/licenses/agpl-3.0.txt
- GNU AGPL v3 reference page: https://www.gnu.org/licenses/agpl-3.0.en.html
- Choose a License summary: https://choosealicense.com/licenses/agpl-3.0/
- SPDX identifier reference: https://spdx.org/licenses/AGPL-3.0-only.html

## Local Checks

- `LICENSE` exists at the repository root.
- `LICENSE` matches the GNU canonical AGPL v3 text byte-for-byte.
- SHA-256 of local `LICENSE`: `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
- `pyproject.toml` includes `license = { file = "LICENSE" }`.
- `pyproject.toml` includes the AGPL v3 classifier.
- `pyproject.toml` includes `license-files = ["LICENSE"]`.
- Main source entrypoints include `SPDX-License-Identifier: AGPL-3.0-only`.

## Interpretation

Choose a License describes GNU AGPLv3 as a strong copyleft license whose network-use condition requires source availability for modified versions served over a network. This matches the PR5 goal: keep the MatrixAI core free and auditable, including SaaS-style use of modified versions.

## Limits

This verification confirms that the **core** repository contains the intended license text and metadata. It covers the core and nothing else.

It does not cover MatrixAI Studio, which is a separate, proprietary product: its terms are in its own `LICENSE`, shipped in the root of the Studio package next to `TERCEROS.md`, and summarised in [BUSINESS_MODEL.md](BUSINESS_MODEL.md). Receiving the Studio still includes receiving this core under AGPL-3.0, with every right the AGPL grants over it.

This is not legal advice. What the AGPL means inside your own organisation — whether you may embed, extend, or redistribute something that carries an AGPL core — is a question for your own counsel.
