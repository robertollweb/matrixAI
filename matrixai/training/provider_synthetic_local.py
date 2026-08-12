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

#: `linear` es una columna DERIVADA de otra ya declarada:
#: `valor = escala · origen + desplazamiento (+ ruido)`.
#:
#: Existe porque sin ella este proveedor solo sabe generar columnas
#: INDEPENDIENTES, y un objetivo independiente de sus entradas es ruido:
#: no hay nada que aprender y la plantilla que lo use enseña a un modelo
#: a no aprender nada. Con esto se puede publicar un caso donde la
#: relación existe de verdad y se ve que el entrenamiento la encuentra —
#: el de Kelvin (`K = °C + 273,15`), que es exactamente eso.
#:
#: Se declara con parámetros —origen, escala, desplazamiento, ruido— y
#: NO con una fórmula de texto: una fórmula obligaría a evaluar cadenas
#: de la plantilla, que es abrir un intérprete a un fichero de datos.
#: Con estos cuatro números se cubre la familia lineal entera, y lo que
#: no sea lineal se trae en un CSV.

#: `threshold` es la ETIQUETA derivada: corta una columna numérica en
#: tramos y les pone nombre. Es la otra mitad de `linear`, y por el mismo
#: motivo: sin ella, la columna objetivo de una plantilla de
#: CLASIFICACIÓN se sortea aparte y no guarda ninguna relación con las
#: entradas. Se puede publicar «predice si esta máquina va a fallar» y
#: entrenar un modelo que no puede acertar más que echándolo a suertes —
#: y eso es justo lo que la web no debe enseñar.
#: `seasonal` es una columna con MEMORIA: sigue una onda suave a lo
#: largo de las filas —`amplitud · sin(2π·(fila+fase)/periodo) +
#: desplazamiento (+ ruido)`— en vez de sortearse fila a fila.
#:
#: Sin ella no se puede publicar una serie temporal honesta. Una
#: auditoría externa lo midió en la demo (2026-08-12): «Consumo eléctrico
#: de mañana» y «Serie temporal sintética» prometían una relación
#: temporal y sacaban **R² NEGATIVO** —peor que contestar siempre la
#: media— porque cada día se muestreaba independiente del anterior, así
#: que mirar hoy no dice nada de mañana. Los retardos y el
#: desplazamiento del objetivo estaban bien puestos; lo que faltaba era
#: que hubiera algo que aprender.
#:
#: Determinista por el ÍNDICE de fila, así que el CSV sigue siendo el
#: mismo con la misma semilla.
_VALID_TYPES = frozenset({
    "number", "integer", "boolean", "categorical", "date", "linear", "threshold",
    "seasonal",
})
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
        if col_type == "seasonal":
            errors.extend(self._validate_seasonal(col, index))
        elif col_type == "linear":
            errors.extend(self._validate_linear(col, index, seen_names))
        elif col_type == "threshold":
            errors.extend(self._validate_threshold(col, index, seen_names))
        elif col_type in ("number", "integer"):
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

    def _validate_linear(self, col: dict[str, Any], index: int, seen_names: set[str]) -> list[str]:
        """Una derivada solo puede mirar HACIA ATRÁS.

        `seen_names` trae las columnas ya declaradas, y el origen tiene
        que estar entre ellas: si se admitiera una de más adelante habría
        que resolver un orden —y con dos que se apunten la una a la otra,
        un ciclo—. Exigir que ya esté declarada lo hace imposible por
        construcción, y de paso el CSV se genera de una pasada.
        """
        errors: list[str] = []
        origen = col.get("from")
        if not isinstance(origen, str) or not origen.strip():
            errors.append(f"columns[{index}].from debe ser el nombre de otra columna.")
        elif origen == col.get("name"):
            errors.append(f"columns[{index}].from no puede ser la propia columna.")
        elif origen not in seen_names:
            errors.append(
                f"columns[{index}].from {origen!r} tiene que estar DECLARADA ANTES en columns."
            )
        for campo, obligatorio in (("scale", True), ("offset", True), ("noise", False)):
            valor = col.get(campo)
            if valor is None and not obligatorio:
                continue
            if not isinstance(valor, (int, float)) or isinstance(valor, bool):
                errors.append(f"columns[{index}].{campo} debe ser un número.")
            elif campo == "noise" and valor < 0:
                errors.append(f"columns[{index}].noise no puede ser negativo.")
            elif not math.isfinite(float(valor)):
                errors.append(f"columns[{index}].{campo} tiene que ser finito.")
        return errors

    def _validate_threshold(self, col: dict[str, Any], index: int, seen_names: set[str]) -> list[str]:
        """Cortes CRECIENTES y una etiqueta más que cortes.

        Con los cortes desordenados, un tramo se queda vacío y su
        etiqueta no sale nunca: la plantilla declararía una clase que el
        dataset no contiene, y el propio verificador del core la
        rechazaría después —con un error que no señala aquí—.
        """
        errors: list[str] = []
        origen = col.get("from")
        if not isinstance(origen, str) or not origen.strip():
            errors.append(f"columns[{index}].from debe ser el nombre de otra columna.")
        elif origen not in seen_names:
            errors.append(
                f"columns[{index}].from {origen!r} tiene que estar DECLARADA ANTES en columns."
            )
        cortes = col.get("cuts")
        if (not isinstance(cortes, list) or not cortes
                or not all(isinstance(c, (int, float)) and not isinstance(c, bool)
                           and math.isfinite(float(c)) for c in cortes)):
            errors.append(f"columns[{index}].cuts debe ser una lista no vacía de números finitos.")
        elif list(cortes) != sorted(cortes) or len(set(cortes)) != len(cortes):
            errors.append(f"columns[{index}].cuts tiene que ir en orden creciente y sin repetidos.")
        etiquetas = col.get("labels")
        if (not isinstance(etiquetas, list)
                or not all(isinstance(e, str) and e.strip() for e in etiquetas)):
            errors.append(f"columns[{index}].labels debe ser una lista de textos no vacíos.")
        elif len(set(etiquetas)) != len(etiquetas):
            errors.append(f"columns[{index}].labels tiene valores duplicados.")
        elif isinstance(cortes, list) and len(etiquetas) != len(cortes) + 1:
            errors.append(
                f"columns[{index}].labels tiene {len(etiquetas)} etiquetas para {len(cortes)} "
                "cortes: hacen falta exactamente una más que cortes."
            )
        return errors

    def _validate_seasonal(self, col: dict[str, Any], index: int) -> list[str]:
        """Periodo positivo y amplitud finita.

        Un periodo de 0 o negativo divide por cero o recorre la onda al
        revés; una amplitud de 0 da una columna CONSTANTE, que no aporta
        ninguna señal — el mismo criterio que ya se exige a las
        categóricas.
        """
        errors: list[str] = []
        periodo = col.get("period")
        if not isinstance(periodo, (int, float)) or isinstance(periodo, bool) or periodo <= 0:
            errors.append(f"columns[{index}].period debe ser un número positivo.")
        for campo, obligatorio in (("amplitude", True), ("offset", True), ("noise", False)):
            valor = col.get(campo)
            if valor is None and not obligatorio:
                continue
            if not isinstance(valor, (int, float)) or isinstance(valor, bool) or not math.isfinite(float(valor)):
                errors.append(f"columns[{index}].{campo} debe ser un número finito.")
            elif campo == "amplitude" and float(valor) == 0.0:
                errors.append(f"columns[{index}].amplitude no puede ser 0: daría una columna constante.")
            elif campo == "noise" and float(valor) < 0:
                errors.append(f"columns[{index}].noise no puede ser negativo.")
        return errors

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
            # Fila a fila y columna a columna, no por comprensión: una
            # columna `linear` necesita el valor que YA se ha generado
            # para su origen en ESTA misma fila.
            fila: dict[str, str] = {}
            for col in columns:
                fila[col["name"]] = self._sample(col, rng, i, fila)
            rows.append(fila)

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

    def _sample(
        self,
        col: dict[str, Any],
        rng: random.Random,
        row_index: int,
        fila: dict[str, str] | None = None,
    ) -> str:
        col_type = col["type"]
        if col_type == "seasonal":
            periodo = float(col["period"])
            valor = (float(col["amplitude"])
                     * math.sin(2 * math.pi * (row_index + float(col.get("phase") or 0.0)) / periodo)
                     + float(col["offset"]))
            ruido = float(col.get("noise") or 0.0)
            if ruido > 0:
                valor += rng.gauss(0.0, ruido)
            return str(round(valor, 4))
        if col_type == "threshold":
            origen = (fila or {}).get(col["from"])
            if origen is None:
                raise DataProviderError(
                    f"columns[{col['name']!r}].from {col['from']!r} no está en la fila; "
                    "tiene que declararse antes."
                )
            valor = float(origen)
            tramo = 0
            for corte in col["cuts"]:
                if valor < float(corte):
                    break
                tramo += 1
            return str(col["labels"][tramo])
        if col_type == "linear":
            origen = (fila or {}).get(col["from"])
            if origen is None:
                # No debería llegar aquí: `validate_config` ya exige que
                # el origen esté declarado antes. Se dice en vez de
                # escribir una celda vacía que luego se leería como un
                # dato que falta.
                raise DataProviderError(
                    f"columns[{col['name']!r}].from {col['from']!r} no está en la fila; "
                    "tiene que declararse antes."
                )
            valor = float(col["scale"]) * float(origen) + float(col["offset"])
            ruido = float(col.get("noise") or 0.0)
            if ruido > 0:
                valor += rng.gauss(0.0, ruido)
            return str(round(valor, 4))
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
