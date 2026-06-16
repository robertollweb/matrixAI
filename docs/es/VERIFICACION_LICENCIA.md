# Verificacion de licencia

Estado: verificada para PR5-C1.

MatrixAI usa la GNU Affero General Public License version 3 (AGPL v3) para el repositorio core. Este documento registra la verificacion realizada antes de continuar con el trabajo de distribucion.

## Fuentes comprobadas

- Texto canonico GNU AGPL v3: https://www.gnu.org/licenses/agpl-3.0.txt
- Pagina de referencia GNU AGPL v3: https://www.gnu.org/licenses/agpl-3.0.en.html
- Resumen de Choose a License: https://choosealicense.com/licenses/agpl-3.0/
- Referencia de identificador SPDX: https://spdx.org/licenses/AGPL-3.0-only.html

## Comprobaciones locales

- `LICENSE` existe en la raiz del repositorio.
- `LICENSE` coincide byte a byte con el texto canonico GNU AGPL v3.
- SHA-256 del `LICENSE` local: `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`.
- `pyproject.toml` incluye `license = { file = "LICENSE" }`.
- `pyproject.toml` incluye el clasificador AGPL v3.
- `pyproject.toml` incluye `license-files = ["LICENSE"]`.
- Los puntos de entrada principales del codigo fuente incluyen `SPDX-License-Identifier: AGPL-3.0-only`.

## Interpretacion

Choose a License describe GNU AGPLv3 como una licencia copyleft fuerte cuya condicion de uso en red exige disponibilidad del codigo fuente para versiones modificadas servidas por red. Esto encaja con el objetivo de PR5: mantener libre y auditable el core de MatrixAI, incluido el uso tipo SaaS de versiones modificadas.

## Limites

Esta verificacion confirma que el repositorio contiene el texto y los metadatos de licencia previstos. No es asesoramiento legal. Las implicaciones comerciales, la licencia futura del Studio, los acuerdos de contribucion y la gestion fiscal de donativos o productos de pago deben revisarse con profesionales cualificados antes del lanzamiento.
