# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — proveedor `synthetic_local`.

Determinista por seed, SIN red — first-class (no un "modo de prueba"):
útil para plantillas offline (C7 "clasificación tabular") y para
cualquier demo/test que necesite un CSV canónico real sin depender de
una API externa. Genera columnas DECLARADAS explícitamente en la config
— nunca infiere nada, la inferencia es cosa de C1 sobre el CSV ya
producido (separación de responsabilidades, invariante 8)."""
from __future__ import annotations

import csv
import io
import math
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any

from matrixai.training.data_provider import (
    DataProviderError,
    DownloadEstimate,
    DownloadResult,
    LicenseAcceptance,
    LicenseInfo,
    ProviderMetadata,
    require_valid_acceptance,
)

_VALID_TYPES = frozenset({"number", "integer", "boolean", "categorical", "date"})
_MAX_ROWS = 1_000_000
_MAX_COLUMNS = 200


class SyntheticLocalProvider:
    provider_id = "synthetic_local"

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata(
            provider_id=self.provider_id,
            display_name="Datos sintéticos (local)",
            description=(
                "CSV determinista generado localmente a partir de una "
                "declaración de columnas — sin red, sin restricciones de licencia."
            ),
            requires_network=False,
        )

    def get_license_info(self) -> LicenseInfo:
        return LicenseInfo(
            name="Datos sintéticos propios",
            url="",
            summary=(
                "Datos generados localmente, sin origen externo — sin "
                "restricciones de licencia ni atribución."
            ),
            requires_attribution=False,
            commercial_use_allowed=True,
            summary_i18n={
                "en": (
                    "Locally generated data with no external source — no "
                    "license restrictions or attribution requirements."
                ),
            },
        )

    def check_availability(self) -> bool:
        return True  # nunca depende de nada externo

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        seed = config.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            errors.append("seed debe ser un entero.")
        rows = config.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            errors.append("rows debe ser un entero positivo.")
        elif rows > _MAX_ROWS:
            errors.append(f"rows no puede superar {_MAX_ROWS}.")
        columns = config.get("columns")
        if not isinstance(columns, list) or not columns:
            errors.append("columns debe ser una lista no vacía de definiciones de columna.")
            return errors
        if len(columns) > _MAX_COLUMNS:
            errors.append(f"columns no puede tener más de {_MAX_COLUMNS} columnas.")
        seen_names: set[str] = set()
        for i, col in enumerate(columns):
            errors.extend(self._validate_column(col, i, seen_names))
        return errors

    def _validate_column(self, col: Any, index: int, seen_names: set[str]) -> list[str]:
        if not isinstance(col, dict):
            return [f"columns[{index}] debe ser un objeto."]
        errors: list[str] = []
        name = col.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"columns[{index}].name debe ser un texto no vacío.")
        elif name in seen_names:
            errors.append(f"columns[{index}].name {name!r} está duplicado.")
        else:
            seen_names.add(name)
        col_type = col.get("type")
        if col_type not in _VALID_TYPES:
            errors.append(
                f"columns[{index}].type debe ser uno de {sorted(_VALID_TYPES)} (recibido {col_type!r})."
            )
            return errors
        if col_type in ("number", "integer"):
            errors.extend(self._validate_range(col.get("range"), index, col_type))
        elif col_type == "categorical":
            cats = col.get("categories")
            if (not isinstance(cats, list) or len(cats) < 2
                    or not all(isinstance(c, str) and c.strip() for c in cats)):
                errors.append(
                    f"columns[{index}].categories debe ser una lista de al menos 2 textos no vacíos."
                )
            elif len(set(cats)) != len(cats):
                # Auditoría 2026-07-17 (ronda 2) [ALTA]: ["a","a"] pasaba
                # "al menos 2 textos" pero produce una columna CONSTANTE
                # (rng.choice siempre devuelve "a") — no aporta ninguna
                # señal, contradice "categórica" de verdad.
                errors.append(f"columns[{index}].categories tiene valores duplicados.")
        elif col_type == "date":
            start = col.get("date_start")
            if not isinstance(start, str):
                errors.append(f"columns[{index}].date_start debe ser una fecha 'YYYY-MM-DD'.")
            else:
                try:
                    date.fromisoformat(start)
                except ValueError:
                    errors.append(f"columns[{index}].date_start {start!r} no es una fecha ISO válida.")
            # Auditoría 2026-07-17 (ronda 2) [ALTA]: date_step_days nunca se
            # validaba — un valor no convertible a entero (p.ej. "bad")
            # pasaba validate_config() limpio y explotaba con ValueError sin
            # envolver dentro de _sample(), un HTTP 500 en vez de un error
            # accionable (invariante 7).
            step = col.get("date_step_days", 1)
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                errors.append(f"columns[{index}].date_step_days debe ser un entero positivo (recibido {step!r}).")
        return errors

    def _validate_range(self, rng: Any, index: int, col_type: str) -> list[str]:
        if (not isinstance(rng, (list, tuple)) or len(rng) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng)):
            return [f"columns[{index}].range debe ser [min, max] numérico."]
        lo, hi = float(rng[0]), float(rng[1])
        # Auditoría 2026-07-17 (ronda 2) [ALTA]: NaN/infinito pasaban el
        # isinstance() de arriba (son floats válidos en Python) y `nan >=
        # hi` siempre es False, así que un rango [nan, 5] "pasaba" la
        # comprobación min<max sin ser un rango real.
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return [f"columns[{index}].range debe ser un rango finito (nada de NaN/infinito)."]
        if lo >= hi:
            return [f"columns[{index}].range debe ser [min, max] con min < max."]
        if col_type == "integer" and not (lo.is_integer() and hi.is_integer()):
            # Auditoría 2026-07-17 (ronda 2) [ALTA]: un rango [0.9, 1.1]
            # para type="integer" pasaba la validación y luego randint(int
            # (0.9), int(1.1)) == randint(0, 1) truncaba en silencio a un
            # rango DISTINTO del declarado (0 queda fuera de [0.9, 1.1]).
            return [
                f"columns[{index}].range para type='integer' debe tener límites "
                f"enteros (recibido [{rng[0]!r}, {rng[1]!r}])."
            ]
        return []

    def estimate_download(self, config: dict[str, Any]) -> DownloadEstimate:
        errors = self.validate_config(config)
        if errors:
            raise DataProviderError("Config inválida: " + "; ".join(errors))
        rows = config["rows"]
        columns = config["columns"]
        # Estimación aproximada (~8 bytes/valor + separadores) — la generación
        # real es instantánea y determinista, no hace falta muestrear de verdad.
        estimated_bytes = rows * len(columns) * 8
        return DownloadEstimate(
            estimated_rows=rows, estimated_bytes=estimated_bytes,
            notes="Estimación aproximada — la generación real es instantánea (sin red).",
            notes_i18n={
                "en": "Approximate estimate — actual generation is instantaneous (offline).",
            },
        )

    def download(self, config: dict[str, Any], *, license_acceptance: LicenseAcceptance | None) -> DownloadResult:
        require_valid_acceptance(license_acceptance, self)
        errors = self.validate_config(config)
        if errors:
            raise DataProviderError("Config inválida: " + "; ".join(errors))

        rng = random.Random(config["seed"])
        columns = config["columns"]
        names = [c["name"] for c in columns]
        rows: list[dict[str, str]] = []
        for i in range(config["rows"]):
            rows.append({col["name"]: self._sample(col, rng, i) for col in columns})

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

        return DownloadResult(
            csv_text=buf.getvalue(), rows=len(rows), columns=names,
            source_url=None, fetched_at=_utcnow_iso(),
            license_info=self.get_license_info(),
            provenance_extra={
                "seed": config["seed"],
                "license_acceptance": license_acceptance.to_dict(),
            },
        )

    def _sample(self, col: dict[str, Any], rng: random.Random, row_index: int) -> str:
        col_type = col["type"]
        if col_type == "number":
            lo, hi = col["range"]
            return str(round(rng.uniform(float(lo), float(hi)), 4))
        if col_type == "integer":
            lo, hi = col["range"]
            return str(rng.randint(int(lo), int(hi)))
        if col_type == "boolean":
            return str(rng.randint(0, 1))
        if col_type == "categorical":
            return rng.choice(col["categories"])
        if col_type == "date":
            start = date.fromisoformat(col["date_start"])
            step = int(col.get("date_step_days", 1))
            return (start + timedelta(days=row_index * step)).isoformat()
        raise DataProviderError(f"Tipo de columna {col_type!r} no soportado.")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
