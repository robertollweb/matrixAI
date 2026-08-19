# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Contrato 82-C1 — el manifiesto `reproduce.json` del paquete exportado.

Hasta hoy el bundle decía **con qué modelo se predice**; lo que no decía es
cómo volver a fabricarlo. Reproducir necesita CINCO cosas —`.mxai`,
`.mxtrain`, receta, filas y semilla— y al paquete llegaban dos
(`model.mxai` y `data_recipe.txt`).

Este módulo escribe la pieza que faltaba: un manifiesto versionado donde
cada artefacto va **por referencia con su digest**, para que verificar y
cargar sean la misma operación (§5-C1 del contrato).

Tres decisiones que no son de estilo:

* **El sha256 va COMPLETO** (§6.6), y no es solo por la longitud. La
  huella que enseña el producto es `"data_" + sha256(...)[:16]` —64 bits,
  cómodos de leer y no una prueba—, pero además **no mide lo mismo**:
  medido el 2026-08-19, el camino tabular la calcula sobre el JSON
  canónico de las FILAS (`InMemoryDataAdapter.fingerprint`,
  `training/data.py:234`), así que **el mismo contenido serializado con
  `;` en vez de `,` da la MISMA huella y distinto sha256**. La huella
  responde «los mismos datos»; el sha256, «el mismo fichero». R1 es «byte
  a byte», así que aquí manda el sha256 del CSV, entero.

  *(Ojo al leerlo: en ese payload, `rows` son las filas ENTERAS, no su
  número. Confundirlo lleva a creer que la huella es de metadatos y que
  no distingue contenidos — y sí los distingue.)*
* **Un modelo sin receta LO DICE** (§6.2), con `reproducible: false` y su
  motivo. No se le fabrica una receta: los modelos entrenados con datos
  reales —el caso del hospital— no tienen ninguna que compartir, y fingir
  que sí es peor que no poder reproducirlos.
* **El entorno va CERRADO**. Medido el 2026-08-19: el `requirements.txt`
  del bundle es `numpy>=1.24` / `onnxruntime>=1.16`, que sirve para
  INFERIR y no para reproducir un número —esta máquina tiene numpy 2.4.4 y
  onnxruntime 1.26.0, ambas dentro del rango y ninguna igual al mínimo—.
  Aquí van versiones exactas, plataforma y un digest del bloque entero.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from matrixai.export.inference_spec import _matrixai_version
from matrixai.training.dataset_manifest import (
    CSV_SERIALIZATION_VERSION,
    SYNTHETIC_GENERATOR_VERSION,
)
from matrixai.training.domain_rules import RECIPE_FORMAT_VERSION

#: Versión del formato de ESTE manifiesto (§5-C1 del contrato lo fija en "1.0").
#: Un consumidor que no la reconozca debe negarse a interpretarlo, no adivinar.
REPRODUCE_SCHEMA_VERSION = "1.0"

REPRODUCE_MANIFEST_FILENAME = "reproduce.json"

#: Nombre del `.mxtrain` DENTRO del paquete. Fijo a propósito: el nombre del
#: fichero de origen es del proyecto de quien exporta y no debe filtrarse al
#: paquete (el `.mxai` ya viaja como `model.mxai` por el mismo motivo).
TRAINING_ARTIFACT_NAME = "model.mxtrain"

#: Los `media_type` son los del §5-C1, literales.
_MEDIA_TYPES = {
    "model": "text/mxai",
    "training": "text/mxtrain",
    "recipe": "text/plain",
}

#: Cómo se canonicaliza el manifiesto antes de digerirlo. Viaja DENTRO del
#: manifiesto porque un digest que nadie sabe recomputar no verifica nada:
#: quien recibe el paquete tiene que poder reproducir el mismo byte a byte
#: sin leerse este fichero.
MANIFEST_CANONICALIZATION = "json:sort_keys=true,separators=(',',':'),ensure_ascii=true,utf-8"

#: Los paquetes cuyo número de versión CAMBIA EL RESULTADO, no la lista
#: completa de lo instalado. Son los que deciden la aritmética: numpy y torch
#: hacen las cuentas, onnx/onnxruntime deciden el grafo exportado y lo que
#: devuelve `predict.py`. El contrato 60 existió precisamente porque torch y
#: stdlib no daban lo mismo. Un paquete ausente se declara `null` —no se
#: omite—: «torch no estaba instalado» es un dato que explica el determinismo,
#: y un valor ausente no es un cero.
_ENVIRONMENT_PACKAGES = ("numpy", "onnx", "onnxruntime", "torch")

#: Campos de una métrica publicada (§5 bis). Un número suelto no se puede
#: comparar: sin `split` no se sabe si es de entrenamiento o de validación
#: —eso ya costó un hallazgo en este producto— y sin `dataset_sha256` no se
#: sabe sobre QUÉ datos se midió.
_METRIC_FIELDS = (
    "name", "value", "split", "dataset_sha256", "evaluator", "evaluator_version",
    "aggregation", "direction", "tolerance_abs", "tolerance_rel",
)

#: Claves de `generation` que las escribe el CORE y NO el llamante. El core
#: sabe qué versión de generador tiene; dejar que el llamante declare otra
#: sería firmar una afirmación que no podemos sostener.
_CORE_OWNED_GENERATION_KEYS = (
    "generator_version", "recipe_format_version", "csv_serialization_version",
)


class ReproduceManifestError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------

def canonical_json(payload: Any) -> str:
    """Forma canónica de un objeto JSON, la MISMA que digiere `manifest_digest`.

    Mismo criterio que `_fingerprint_payload` en `matrixai/training/data.py`
    (`sort_keys` + separadores compactos + ascii): dos sitios canonicalizando
    distinto darían dos digests distintos del mismo contenido.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """sha256 COMPLETO (64 hex) de un texto. Sin recortar: ver §6.6."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """sha256 COMPLETO de un fichero, leído por trozos.

    Por trozos y no de un `read_bytes()`: por aquí pasa el `.mxai` de un
    modelo grande, y traérselo entero a RAM es justo lo que el camino de
    pesos grandes evita.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Digest del manifiesto **sin** su campo `manifest_sha256` (§5-C1).

    Se excluye la clave entera, no se pone en blanco: un `""` seguiría
    formando parte del texto digerido y quien verificara tendría que adivinar
    con qué valor de relleno se calculó.
    """
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verify_manifest_digest(manifest: dict[str, Any]) -> bool:
    """¿El `manifest_sha256` que trae el manifiesto cuadra con su contenido?"""
    declared = manifest.get("manifest_sha256")
    if not isinstance(declared, str) or not declared:
        return False
    return declared == manifest_digest(manifest)


# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

def _package_version(name: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 — PackageNotFoundError y cualquier metadata rota
        return None


def build_environment() -> dict[str, Any]:
    """El entorno efectivo, CERRADO: versiones exactas, plataforma y su digest.

    Medido en esta máquina el 2026-08-19: matrixai 1.5.0, CPython 3.12.3,
    Linux-6.8.0-137-generic-x86_64-with-glibc2.39, numpy 2.4.4, onnx 1.21.0,
    onnxruntime 1.26.0, torch 2.11.0+cpu.

    `environment_sha256` resume el bloque entero para que comparar dos
    entornos sea UNA comparación y no diez; los campos siguen ahí para poder
    decir en qué se diferencian.
    """
    env: dict[str, Any] = {
        "matrixai_version": _matrixai_version(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            # `sys.version` trae además el compilador y la fecha del build:
            # dos CPython 3.12.3 compilados distinto no son el mismo entorno.
            "sys_version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {name: _package_version(name) for name in _ENVIRONMENT_PACKAGES},
    }
    env["environment_sha256"] = hashlib.sha256(
        canonical_json(env).encode("utf-8")
    ).hexdigest()
    return env


# ---------------------------------------------------------------------------
# Bloque `generation`
# ---------------------------------------------------------------------------

def _split_seed_from_training(training_text: str) -> tuple[int | None, bool]:
    """La semilla del SPLIT, leída del `.mxtrain` que de verdad entrenó.

    Devuelve `(semilla, se_pudo_leer)`. La distinción importa: `None` con
    `True` significa «el SPLIT no declara semilla» —legítimo, y obligatorio
    en `mode=temporal`, que el parser rechaza con semilla— mientras que
    `None` con `False` significa «no se pudo leer el `.mxtrain`».

    Se lee del fichero y no del llamante a propósito: el split seed no tiene
    override por línea de comandos (medido en `cli.py`: solo `--backend` y
    `--device` lo tienen), así que el `.mxtrain` ES la fuente. Dos sitios
    declarando lo mismo acaban divergiendo.
    """
    try:
        from matrixai.training.parser import parse_training_text
        spec = parse_training_text(training_text)
    except Exception:  # noqa: BLE001 — un .mxtrain que no parsea no invalida el paquete
        return None, False
    split = getattr(spec.dataset, "split", None)
    if split is None:
        return None, True
    return getattr(split, "seed", None), True


def build_generation_block(
    caller: dict[str, Any] | None,
    *,
    training_text: str | None = None,
) -> dict[str, Any]:
    """Normaliza el bloque `generation`: TODOS los parámetros efectivos.

    Receta, filas y semilla no bastan (§5-C1): un valor por defecto que
    cambie entre versiones cambia el dataset sin que nadie toque nada. Por
    eso viajan también las versiones del generador, del formato de receta y
    de la serialización del CSV, el modo, los rangos/tipos/categorías/
    identificadores, las TRES semillas por separado y el backend/dispositivo.

    Lo que el llamante no sepa se queda en `null`. No se rellena con un cero
    ni con un valor «razonable»: una semilla inventada convertiría un paquete
    irreproducible en uno que parece reproducible y falla al comprobarlo.

    `rows` NO va aquí: vive en `artifacts.dataset.rows` (§5-C1). Repetirlo
    sería el segundo sitio declarando lo mismo.
    """
    caller = dict(caller or {})
    seeds_in = caller.pop("seeds", None) or {}
    if not isinstance(seeds_in, dict):
        raise ReproduceManifestError("generation.seeds must be an object")

    split_seed = seeds_in.get("split")
    if split_seed is None and training_text is not None:
        parsed_seed, ok = _split_seed_from_training(training_text)
        if ok:
            split_seed = parsed_seed

    block: dict[str, Any] = {
        # Estas tres las pone el core, pase lo que pase (ver
        # `_CORE_OWNED_GENERATION_KEYS`).
        "generator_version": SYNTHETIC_GENERATOR_VERSION,
        "recipe_format_version": RECIPE_FORMAT_VERSION,
        "csv_serialization_version": CSV_SERIALIZATION_VERSION,
        "mode": caller.pop("mode", None),
        "field_ranges": caller.pop("field_ranges", None),
        "field_types": caller.pop("field_types", None),
        "field_categories": caller.pop("field_categories", None),
        "one_hot_groups": caller.pop("one_hot_groups", None),
        "excluded_identifiers": caller.pop("excluded_identifiers", None),
        "seeds": {
            # La del generador de datos (`/api/generate-dataset`).
            "dataset": seeds_in.get("dataset"),
            # La del reparto train/validación (bloque SPLIT del `.mxtrain`).
            "split": split_seed,
            # La de inicialización de pesos (`DenseSupervisedTrainer.train`,
            # medido: `seed: int = 42` por defecto). No se asume ese 42: si
            # el llamante no lo declara, no sabemos cuál corrió de verdad.
            "init": seeds_in.get("init"),
        },
        "backend": caller.pop("backend", None),
        "device": caller.pop("device", None),
        "deterministic_options": caller.pop("deterministic_options", None),
    }
    # Lo que el llamante añada de más se conserva: descartarlo en silencio
    # perdería un parámetro efectivo que este core aún no conoce por nombre.
    for key in _CORE_OWNED_GENERATION_KEYS:
        caller.pop(key, None)
    for key, value in caller.items():
        block.setdefault(str(key), value)
    return block


# ---------------------------------------------------------------------------
# Métricas (§5 bis)
# ---------------------------------------------------------------------------

def _normalize_metrics(metrics: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if metrics is None:
        return []
    if not isinstance(metrics, list):
        raise ReproduceManifestError("metrics must be a list of objects")
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(metrics):
        if not isinstance(raw, dict):
            raise ReproduceManifestError(f"metrics[{i}] must be an object")
        if not str(raw.get("name") or "").strip():
            raise ReproduceManifestError(f"metrics[{i}] requires a non-empty 'name'")
        if raw.get("value") is None:
            # Una métrica sin valor no es una métrica; y un `null` aquí no es
            # «cero», es que no se midió — así que se rechaza en vez de
            # publicarla como si fuera un número.
            raise ReproduceManifestError(f"metrics[{i}] requires a 'value'")
        # Los campos que falten quedan a `null` y VISIBLES: una tolerancia
        # ausente se ve, una clave que no está se pasa por alto.
        entry = {field: raw.get(field) for field in _METRIC_FIELDS}
        for key, value in raw.items():
            if key not in entry:
                entry[str(key)] = value
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# El manifiesto
# ---------------------------------------------------------------------------

def build_reproduce_manifest(
    bundle_dir: str | Path,
    *,
    model_filename: str = "model.mxai",
    training_filename: str | None = None,
    recipe_filename: str | None = None,
    dataset_sha256: str | None = None,
    dataset_rows: int | None = None,
    generation: dict[str, Any] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construye el `reproduce.json` de un bundle YA escrito en `bundle_dir`.

    Los digests se calculan sobre los ficheros REALES del paquete, no sobre lo
    que el llamante diga que puso: un manifiesto que repite lo que le cuentan
    no verifica nada.
    """
    bundle_dir = Path(bundle_dir)

    def _artifact(kind: str, filename: str | None) -> dict[str, Any] | None:
        if not filename:
            return None
        path = bundle_dir / filename
        if not path.is_file():
            return None
        return {
            "path": filename,
            "media_type": _MEDIA_TYPES[kind],
            "sha256": sha256_file(path),
        }

    model = _artifact("model", model_filename)
    training = _artifact("training", training_filename)
    recipe = _artifact("recipe", recipe_filename)

    training_text: str | None = None
    if training is not None:
        training_text = (bundle_dir / training["path"]).read_text(encoding="utf-8")

    generation_block = build_generation_block(generation, training_text=training_text)

    # El dataset va por su valor ESPERADO, no como fichero: §6.4 —el paquete
    # no incrusta datos cuya licencia prohíba redistribuirlos— y aquí es
    # gratis, porque se empaqueta la regla que los produce.
    dataset: dict[str, Any] | None = None
    if dataset_sha256 is not None or dataset_rows is not None:
        dataset = {"sha256": dataset_sha256, "rows": dataset_rows}

    # ── Qué falta para poder reproducir (§3: son CINCO cosas) ──
    # `model.mxai` siempre está. Del resto se declara EXACTAMENTE lo que
    # falta, con un código estable por hueco: el motivo en prosa es para
    # quien lee y los códigos son para quien traduce o automatiza.
    missing: list[str] = []
    if training is None:
        missing.append("training")
    if recipe is None:
        missing.append("recipe")
    if not dataset_sha256:
        missing.append("dataset_sha256")
    if dataset_rows is None:
        missing.append("dataset_rows")
    if generation_block["seeds"]["dataset"] is None:
        missing.append("seed_dataset")

    manifest: dict[str, Any] = {
        "schema_version": REPRODUCE_SCHEMA_VERSION,
        # §6.7: esto demuestra COHERENCIA, no AUTENTICIDAD. Quien modifique
        # receta, manifiesto y métricas de forma coherente obtiene un paquete
        # coherente. Las firmas son del contrato 81, y prometerlas aquí sería
        # la «falsa sensación de garantía» que ese contrato identifica.
        "claim": (
            "This manifest proves the package is internally consistent and "
            "reproducible. It does NOT prove authorship or authenticity: "
            "signatures are out of scope here."
        ),
        "reproducible": not missing,
        "reproducible_reason": None if not missing else _missing_reason(missing),
        "missing": missing,
        "artifacts": {
            "model": model,
            "training": training,
            "recipe": recipe,
            "dataset": dataset,
        },
        "generation": generation_block,
        "environment": build_environment(),
        "metrics": _normalize_metrics(metrics),
    }
    manifest["manifest_canonicalization"] = MANIFEST_CANONICALIZATION
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


_MISSING_REASONS = {
    "training": (
        "no .mxtrain travels in this package, so the model cannot be retrained"
    ),
    "recipe": (
        "this model has no data recipe — it was trained on real data that is not "
        "redistributable, and no recipe was invented for it"
    ),
    "dataset_sha256": (
        "the expected dataset sha256 is unknown, so a regenerated dataset cannot "
        "be compared against anything"
    ),
    "dataset_rows": "the expected row count is unknown",
    "seed_dataset": "the dataset generation seed is unknown",
}


def _missing_reason(missing: list[str]) -> str:
    """El motivo en prosa, listando lo que falta y por qué eso impide reproducir.

    En inglés como el resto de lo que se descarga (README, `predict.py`): este
    texto lo lee quien recibe el paquete, que puede estar en cualquier sitio.
    Quien quiera enseñarlo traducido tiene los códigos de `missing`.
    """
    parts = [_MISSING_REASONS.get(item, item) for item in missing]
    return "Not reproducible: " + "; ".join(parts) + "."


def write_reproduce_manifest(bundle_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Escribe `reproduce.json` dentro del bundle y devuelve el manifiesto.

    Se escribe SIEMPRE, incluso cuando no se puede reproducir: un paquete que
    calla no dice «no se puede», dice nada. §6.2 lo pide explícito.
    """
    manifest = build_reproduce_manifest(bundle_dir, **kwargs)
    (Path(bundle_dir) / REPRODUCE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return manifest
