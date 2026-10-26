# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El `reproduce.json` de un ESTUDIO — contrato 138-C2a.

POR QUÉ EXISTE. `ficha_tripod`, `huecos_probast`/`mapa_probast` y `ml_bom` leen
`reproduce.json`, y el paquete que exporta un estudio del Studio
(`matrixai_engines.paquete_de_estudio`) no lo llevaba: con cualquier estudio
real las tres contestaban «no reproduce.json». Medido el 2026-10-25 con
`exportar_estudio` sobre un estudio de verdad.

QUÉ HACE. Compone el manifiesto con **lo que el estudio YA midió y declaró** y
con nada más. Lo que el estudio no sabe se OMITE —no se rellena con un cero ni
con un valor «razonable»—, y las tres funciones de arriba lo dirán «no
disponible» por su cuenta. Lo que no está aquí y podría parecer que sí:

* `artifacts.model` / `training` / `recipe`: un estudio no tiene `.mxai`, ni
  `.mxtrain`, ni receta de datos. Van a `null`, que es «esto no viaja aquí».
* `generation.mode`: el modo (`random`/`coherent`) es del generador sintético, y
  un estudio entrena con el CSV de quien lo trae.
* `backend` / `device`: su vocabulario es el del camino clásico (`stdlib`,
  `torch`); el motor ganador de un estudio no cabe en él.

`reproducible` es SIEMPRE `false`, con su motivo en castellano e inglés: un
estudio no se reproduce por el camino clásico de `matrixai export` /
`matrixai verify`. Afirmar lo contrario sería lo que este producto persigue.

EL ESQUEMA es el de `reproduce.py` (1.1): los mismos nombres de campo, el mismo
`build_environment()`, el mismo digest. No se inventa uno nuevo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from matrixai.export.reproduce import (
    MANIFEST_CANONICALIZATION,
    REPRODUCE_MANIFEST_FILENAME,
    REPRODUCE_SCHEMA_VERSION,
    ReproduceManifestError,
    _normalize_metrics,
    _require_full_sha256,
    _require_int,
    _require_positive_int,
    _require_str_list,
    _require_text,
    build_environment,
    build_problem_block,
    manifest_digest,
)

__all__ = ["manifiesto_de_estudio", "escribir_manifiesto_de_estudio",
           "MOTIVO_NO_REPRODUCIBLE"]

#: Por qué un estudio NO se declara reproducible. Bilingüe: la ficha TRIPOD lo
#: enseña en el idioma que se le pida, y el resto del core (BOM, CLI) lee la
#: versión inglesa de `reproducible_reason`, como con el camino clásico.
MOTIVO_NO_REPRODUCIBLE: dict[str, str] = {
    "es": ("este paquete sale de un estudio de selección de modelos (validación "
           "cruzada entre varios motores), no de un único entrenamiento: no lleva "
           "`.mxai`, `.mxtrain` ni receta de datos, y no se reproduce con "
           "`matrixai verify`. Lo que registra es lo que el estudio midió y declaró."),
    "en": ("this package comes from a model-selection study (cross-validation "
           "across several engines), not from a single training run: it carries no "
           "`.mxai`, `.mxtrain` or data recipe, and it cannot be reproduced with "
           "`matrixai verify`. What it records is what the study measured and "
           "declared."),
}

#: Las particiones que un estudio mide y el vocabulario cerrado de `reproduce.py`
#: (`_METRIC_SPLITS`) conoce: la prueba reservada y la estimación de la selección
#: (validación cruzada sobre el train).
_PARTICIONES = ("test", "validation")


def _metricas(metricas: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not metricas:
        return []
    for i, m in enumerate(metricas):
        if not isinstance(m, Mapping):
            raise ReproduceManifestError(f"metricas[{i}] must be an object")
        if m.get("split") not in _PARTICIONES:
            raise ReproduceManifestError(
                f"metricas[{i}].split must be one of {_PARTICIONES}, got {m.get('split')!r}: "
                f"a figure without the partition it was measured on is just a number")
    return _normalize_metrics([dict(m) for m in metricas])


def manifiesto_de_estudio(
    *,
    filas: int | None = None,
    huella_del_dataset: str | None = None,
    tipos_de_campo: Mapping[str, str] | None = None,
    columnas_excluidas: Sequence[str] | None = None,
    semilla_del_reparto: int | None = None,
    metricas: Sequence[Mapping[str, Any]] | None = None,
    problema: Any | None = None,
    entorno: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """El manifiesto de un estudio, con SOLO lo que el estudio sabe.

    Todo es opcional y **ausente es ausente**: ningún argumento `None` produce un
    campo con valor inventado.

    * `filas`: cuántas filas entrenaron (positivo).
    * `huella_del_dataset`: sha256 COMPLETO del CSV tal como lo recibió el estudio.
    * `tipos_de_campo`: `{columna: tipo}` de las entradas del modelo.
    * `columnas_excluidas`: las columnas que se quedaron fuera (identificador de
      fila, texto libre…). `[]` significa «el estudio sabe que no hubo ninguna»;
      `None`, «no consta».
    * `semilla_del_reparto`: la semilla con que se repartió train/test/pliegues.
    * `metricas`: `[{name, value, split: test|validation, n_observations?}]`.
    * `problema`: el `ProblemSpec` (o su `a_json()`), para el orden de las clases
      y la clase positiva.
    * `entorno`: por omisión, `build_environment()` del proceso que exporta.
    """
    if filas is not None:
        _require_positive_int(filas, "filas")
    if huella_del_dataset is not None:
        _require_full_sha256(huella_del_dataset, "huella_del_dataset")
    if semilla_del_reparto is not None:
        _require_int(semilla_del_reparto, "semilla_del_reparto")

    field_types: dict[str, str] | None = None
    if tipos_de_campo:
        field_types = {}
        for campo, tipo in tipos_de_campo.items():
            _require_text(campo, "tipos_de_campo (nombre)")
            field_types[str(campo)] = _require_text(tipo, f"tipos_de_campo[{campo!r}]")

    excluidas: list[str] | None = None
    if columnas_excluidas is not None:
        if isinstance(columnas_excluidas, (str, bytes)):
            # Un texto no es una lista de columnas: `list("row_id")` serían cinco "columnas" de una letra.
            raise ReproduceManifestError(
                f"columnas_excluidas must be a list of strings, got {columnas_excluidas!r}")
        excluidas = sorted(set(_require_str_list(list(columnas_excluidas),
                                                 "columnas_excluidas")))

    dataset: dict[str, Any] | None = None
    if filas is not None or huella_del_dataset is not None:
        dataset = {}
        if huella_del_dataset is not None:
            dataset["sha256"] = huella_del_dataset
        if filas is not None:
            dataset["rows"] = filas

    generation: dict[str, Any] = {}
    if field_types is not None:
        generation["field_types"] = field_types
    if excluidas is not None:
        generation["excluded_identifiers"] = excluidas
    if semilla_del_reparto is not None:
        generation["seeds"] = {"split": semilla_del_reparto}

    manifiesto: dict[str, Any] = {
        "schema_version": REPRODUCE_SCHEMA_VERSION,
        "claim": (
            "This manifest records what a model-selection study measured and "
            "declared. It does NOT prove the model can be reproduced (see "
            "`reproducible_reason`), nor authorship or authenticity, and it is not "
            "a certification."),
        "reproducible": False,
        "reproducible_reason": MOTIVO_NO_REPRODUCIBLE["en"],
        "reproducible_reason_i18n": dict(MOTIVO_NO_REPRODUCIBLE),
        "artifacts": {"model": None, "training": None, "recipe": None,
                      "dataset": dataset},
        "problem": build_problem_block(problema),
        "generation": generation,
        "environment": dict(entorno) if entorno is not None else build_environment(),
        "metrics": _metricas(metricas),
        "manifest_canonicalization": MANIFEST_CANONICALIZATION,
    }
    manifiesto["manifest_sha256"] = manifest_digest(manifiesto)
    return manifiesto


def escribir_manifiesto_de_estudio(bundle_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Escribe `reproduce.json` en el paquete del estudio y devuelve el manifiesto.

    Mismo formato de fichero que el camino clásico (`indent=2`, ASCII).
    """
    manifiesto = manifiesto_de_estudio(**kwargs)
    ruta = Path(bundle_dir) / REPRODUCE_MANIFEST_FILENAME
    ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifiesto
