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

Cuatro decisiones que no son de estilo:

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
* **El manifiesto no puede sellar lo que no sabe.** Medido el 2026-08-19
  exportando un job antes de que terminara de entrenar: la respuesta HTTP
  decía `weights_source: untrained` y `reproduce.json` decía
  `reproducible: true`, porque el manifiesto no sabía NADA de los pesos y el
  aviso vivía solo en esa respuesta —que no acompaña al ZIP—. Ahora el estado
  de los pesos entra como dato (`weights_source`), viaja dentro del paquete
  (`weights.source`) y **«no consta» no es «entrenado»**: sin él no hay
  `reproducible: true`. Lo mismo con lo que el `.mxtrain` no puede decir —las
  épocas que de verdad corrieron, los rangos que decidieron el CSV preparado,
  las filas que se usaron y si el run arrancó de unos pesos que ya existían—:
  se publica lo que el run capturó, y lo que se contradiga se declara.
* **El entorno va CERRADO**. Medido el 2026-08-19: el `requirements.txt`
  del bundle es `numpy>=1.24` / `onnxruntime>=1.16`, que sirve para
  INFERIR y no para reproducir un número —esta máquina tiene numpy 2.4.4 y
  onnxruntime 1.26.0, ambas dentro del rango y ninguna igual al mínimo—.
  Aquí van versiones exactas, plataforma y un digest del bloque entero.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

from matrixai.export.inference_spec import _matrixai_version
from matrixai.training.dataset_manifest import (
    CSV_SERIALIZATION_VERSION,
    SYNTHETIC_GENERATOR_VERSION,
)
from matrixai.training.domain_rules import RECIPE_FORMAT_VERSION
from matrixai.training.metric_identity import (
    identidad_de_metrica,
    tolerancia_medida,
)
from matrixai.export.terceros import (
    CLAVE_EN_EL_MANIFIESTO as EMBEDDING_PROVIDER_KEY,
    ComponenteDeTercerosIncompleto,
    bloque_para_el_manifiesto,
)

#: Versión del formato de ESTE manifiesto (§5-C1 del contrato lo fija en "1.0").
#: Un consumidor que no la reconozca debe negarse a interpretarlo, no adivinar.
#: 1.1 (2026-09-05, 103-C1) es ADITIVA sobre 1.0: el mismo manifiesto más un
#: bloque `problem` con el problema confirmado —objetivo, tarea, **el orden de
#: las clases** y la clase positiva—. `verify` ya acepta 1.0, 1.1 y 1.2, así que
#: un paquete nuevo no se vuelve ilegible; y un manifiesto 1.0 no lleva
#: `problem`, que es distinto de llevarlo vacío.
#:
#: **107-C2 añade `embedding_provider` y NO sube la versión, a sabiendas.** El
#: precedente (1.0 → 1.1 por `problem`) diría que sí, pero subirla aquí no
#: compra lo que parece: `verify` ya acepta 1.0, 1.1 y 1.2, así que un
#: verificador viejo no rechazaría un paquete con texto por la versión —lo
#: rechaza por el bloque, que es donde se comprueba (`_verificar_terceros`)—.
#: Y sí cuesta: la lista de versiones es vocabulario compartido con el Studio.
#: Queda declarado como decisión pendiente, no como olvido, y hay una prueba a
#: su nombre de que la puerta cierra igual con 1.1.
REPRODUCE_SCHEMA_VERSION = "1.1"

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


#: Vocabularios CERRADOS, tomados de lo que el producto acepta de verdad
#: —no inventados aquí—:
#:   * `mode`: `_generate_synthetic_dataset` (playground.py:561) coerce
#:     cualquier otra cosa a "random". Un manifiesto que declare
#:     "coherent-plus" estaría prometiendo un modo que no existe.
#:   * `backend`: `_VALID_TARGETS` de `training/spec.py:154`.
#:   * `split`: los que el producto sabe medir. "full" es el dataset ENTERO,
#:     que es sobre lo que mide `evaluation_report` — medido el 2026-08-19:
#:     `macro_f1` sale de `validation_metrics` o de `evaluation_report`, y son
#:     particiones distintas. Sin nombre, las dos cifras se confunden, que es
#:     justo el hallazgo que ya costó este producto (§5 bis).
_GENERATION_MODES = ("random", "coherent")
_TRAINING_BACKENDS = ("stdlib", "torch")
_METRIC_SPLITS = ("train", "validation", "test", "full")
_METRIC_DIRECTIONS = ("higher_is_better", "lower_is_better")

#: De dónde salen LOS PESOS QUE VIAJAN EN ESTE PAQUETE. Vocabulario del
#: producto, no inventado aquí: `endpoints.py:5471` lo calcula como
#: `"trained" if has_weights else "untrained"` y la interfaz lo lee con ese
#: mismo par (`studio/src/api/client.ts:1102`).
#:
#: No es un dato del RUN y por eso no sale de la captura: la captura se compone
#: al EMPEZAR a entrenar y no puede testificar sobre los bytes que alguien
#: empaquetó después. Lo declara quien empaqueta, que es el único que los ha
#: visto.
_WEIGHTS_SOURCES = ("trained", "untrained")

#: Qué necesita una métrica para que R3 pueda CONTRASTARLA (no para publicarla).
#: Sin `split` no es la misma cifra; sin `dataset_sha256` no se sabe sobre QUÉ
#: datos se midió; sin `direction` no se puede decir si desviarse es mejor o
#: peor; y sin tolerancia no hay umbral — y §5 bis prohíbe fabricarlo, porque
#: sale de repetir en la matriz de entornos, no de dos ejecuciones.
#: `evaluator`, `evaluator_version` y `aggregation` NO entran aquí: dicen QUIÉN
#: midió, no cómo comparar. Faltan igual y se declaran en `incomplete`.
_METRIC_R3_FIELDS = ("split", "dataset_sha256", "direction")

#: Claves de una métrica que decide el CORE, no quien exporta: si el llamante
#: manda `comparable: true` sobre una métrica sin tolerancia, gana el core.
_CORE_OWNED_METRIC_KEYS = ("comparable", "incomplete")


#: Versiones del bloque `run_provenance` que ESTE módulo sabe interpretar.
#: La escribe el core al entrenar (`playground.py::RUN_PROVENANCE_SCHEMA_VERSION`,
#: hoy "1.0"). No se importa de allí a propósito: `playground` importa `export`,
#: y el import al revés cerraría el círculo. Una versión desconocida se RECHAZA
#: en vez de interpretarse a medias — adivinar qué significa un campo nuevo es
#: justo lo que el `schema_version` existe para evitar.
#: "1.1" (2026-08-19) es ADITIVA sobre "1.0" y la escribe el core desde este
#: corte: los mismos campos obligatorios más `epochs_effective`, `epochs_ran`,
#: `dataset_rows_used`, `field_ranges`, `target_range` y `warm_start`. Se
#: aceptan LAS DOS porque las dos existen en disco —los modelos guardados antes
#: de hoy llevan capturas "1.0"—, y la versión es justo lo que permite
#: distinguir AUSENTE de FALSO: una captura "1.0" no dice que el run partiera
#: de cero, dice que no lo sabe, y eso no se rellena con un `false`.
#: 1.2 añade `recipe_verification`: si la receta y la semilla que declara la
#: captura REGENERAN el dataset, o no se pudo comprobar. Antes de esto una
#: captura comprobada y una sin comprobar producían bloques `provenance`
#: idénticos, y el veredicto vivía solo en la respuesta HTTP de train-start —
#: que no acompaña al paquete, el mismo patrón que este fichero ya reprocha.
_RUN_PROVENANCE_SCHEMA_VERSIONS = ("1.0", "1.1", "1.2")

#: Lo que la captura declara y el manifiesto publica como parámetro efectivo.
#: Son las mismas claves que `build_generation_block` normaliza: la captura es
#: la fuente autoritativa de todas ellas, así que se validan con la MISMA
#: función y no con una segunda copia de las reglas.
_RUN_PROVENANCE_GENERATION_KEYS = (
    "mode", "field_ranges", "field_types", "field_categories", "one_hot_groups",
    "excluded_identifiers", "deterministic_options",
    # Los de la captura "1.1". Los nombres NO se eligen aquí: son los que
    # `playground.py` escribe, leídos el 2026-08-19.
    #   * `epochs_effective`: las que el spec se configuró a correr YA con el
    #     tope aplicado (`_apply_epoch_cap`). El tope del operador
    #     (`MATRIXAI_MAX_EPOCHS`, que traen los perfiles) recorta sin tocar el
    #     contrato, así que el paquete puede llevar un `EPOCHS 50` y unos pesos
    #     de 3 épocas.
    #   * `epochs_ran`: las que CORRIERON. No sustituye a la anterior: medido
    #     con `EARLY_STOP patience=1`, `effective` 50 y `ran` 20 sin que ningún
    #     tope tocara nada — y eso SÍ se reproduce, porque el early stop lo
    #     declara el propio contrato.
    #   * `target_range`: la escala del objetivo. Decide con qué datos se
    #     entrenó de verdad (`_normalize_csv_with_ranges`), igual que
    #     `field_ranges`: sin él no se rehace el CSV preparado.
    #   * `warm_start`: de qué pesos PARTIÓ el run. `false` = de la
    #     inicialización; un objeto = de unos pesos que ya existían, con su
    #     huella; `null` = llegaron pesos y el run no llegó a decir si los usó.
    #     No está en el `.mxai`, ni en el `.mxtrain`, ni en las semillas: con
    #     torch se midió loss 1.102227 desde cero y 1.094253 reanudado, con la
    #     MISMA captura byte a byte.
    "epochs_effective", "epochs_ran", "target_range", "warm_start",
)

#: Claves de la captura que NO son parámetros de generación (identidad del run,
#: digests y textos). Todo lo demás que traiga se trata como parámetro efectivo
#: y viaja a `generation`: descartar en silencio un parámetro que este core aún
#: no conoce por nombre perdería justo lo que hace falta para regenerar.
_RUN_PROVENANCE_CORE_KEYS = (
    "schema_version", "mxai_sha256", "mxtrain_sha256", "mxtrain_text",
    "recipe_sha256", "recipe_text", "dataset_sha256_raw",
    "dataset_sha256_prepared", "dataset_rows", "seeds", "backend", "device",
    "generator_version", "csv_serialization_version", "recipe_verification",
    # `dataset_rows_used` va con el dataset (`artifacts.dataset.rows_used`), no
    # con los parámetros de generación: `rows` son las filas del CSV CRUDO —lo
    # que hay que regenerar para R1— y `rows_used` las que el entrenamiento
    # consumió del preparado. Dos preguntas distintas, como los dos digests del
    # dataset, y por eso no se funden en una.
    "dataset_rows_used",
    # `weights_source` no se publica desde aquí (lo declara quien empaqueta,
    # ver `_WEIGHTS_SOURCES`), pero se reconoce como clave de la captura para
    # que no se cuele en `generation` como si fuera un parámetro del run.
    "weights_source",
)

# `\Z` y NO `$`: en Python `$` casa TAMBIÉN antes de un salto de línea
# final, así que `"0"*64 + "\n"` —65 caracteres— pasaba por un sha256
# válido y viajaba al manifiesto con su `\n` dentro. R1 compara digests
# completos, y ese valor no puede igualar a ninguno calculado jamás: el
# paquete prometía una comparación imposible. Lo destapó una auditoría
# adversarial el 2026-08-19, y el banco del supervisor tampoco lo veía
# porque probaba `"0"*65` (rechazado) y no `"0"*64 + "\n"`.
_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")


class ReproduceManifestError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validación de forma (82-C1, bloqueante medido el 2026-08-19)
# ---------------------------------------------------------------------------
#
# POR QUÉ ESTO VIVE EN EL CORE Y NO EN EL LLAMANTE. Medido antes de tocar nada:
#
#     build_reproduce_manifest(..., dataset_sha256="not-a-sha", dataset_rows=-4,
#                              generation={"seeds": {"dataset": "42"}})
#     -> reproducible: True | missing: []
#
# El paquete PROMETÍA reproducirse con un digest que no es un digest, filas
# negativas y una semilla que es texto. La única validación estricta que
# existía estaba en dos llamantes —`cli.py::_load_reproduce_metadata` y
# `endpoints.py::_reproduccion_para_el_bundle`—, así que cualquier tercero la
# esquivaba. Y las dos ya estaban divergiendo (el CLI exigía 64 hex y el core
# no), que es el segundo sitio declarando lo mismo de siempre.
#
# POR QUÉ EXCEPCIÓN Y NO `reproducible: false`. Son dos situaciones distintas
# y mezclarlas pierde información:
#
#   * Un valor AUSENTE es una respuesta legítima (§6.2): el modelo del
#     hospital no tiene receta que compartir. Eso sigue siendo
#     `reproducible: false` con su motivo, y el paquete se escribe igual.
#   * Un valor IMPOSIBLE no es una respuesta: es un fallo de cableado de quien
#     exporta. Publicarlo como «no reproducible» sería declarar algo FALSO
#     sobre el modelo —que no se puede rehacer— cuando lo cierto es que el
#     dato llegó mal y probablemente sí se puede. Y el paquete saldría por la
#     puerta con el fallo dentro, en un campo que ya nadie vuelve a mirar.
#
# Quien recibe el paquete queda mejor protegido con la excepción: un paquete
# que no llega a existir no puede mentirle, y quien exporta se entera en el
# sitio donde todavía se puede arreglar. Es la misma decisión que ya tomaba
# `_studio_model_save` al rechazar un `job_architecture_mismatch` en vez de
# guardar el modelo con una nota.


def _require_int(value: Any, what: str) -> int:
    """Un entero DE VERDAD.

    `type(...) is not int` y no `isinstance`: en Python `True` es un `int` para
    `isinstance` (medido: `isinstance(True, int) is True`), y `True` no es una
    semilla. Un `42.0` tampoco: viene de un JSON que perdió el tipo.
    """
    if type(value) is not int:
        raise ReproduceManifestError(f"{what} must be an integer or null, got {value!r}")
    return value


def _require_positive_int(value: Any, what: str) -> int:
    if type(value) is not int or value < 1:
        raise ReproduceManifestError(f"{what} must be a positive integer, got {value!r}")
    return value


def _require_full_sha256(value: Any, what: str) -> str:
    """64 hex minúsculas, §6.6.

    La huella que enseña el producto es `"data_" + sha256(...)[:16]`: 64 bits,
    cómoda de leer y NO una prueba de integridad. Aceptarla aquí colaría un
    identificador donde el contrato pide una prueba.
    """
    if not isinstance(value, str) or not _SHA256_HEX.match(value):
        raise ReproduceManifestError(
            f"{what} must be the FULL sha256 (64 lowercase hex chars), "
            f"not the short 'data_...' fingerprint, got {value!r}"
        )
    return value


def _require_real(value: Any, what: str, *, minimum: float | None = None) -> float:
    """Un número real y FINITO.

    Lo de finito no es teoría: `json.dumps` escribe `NaN` e `Infinity`, que no
    son JSON válido — un manifiesto con eso dentro no lo puede leer quien lo
    recibe, y encima su `manifest_sha256` cubriría un fichero ilegible.
    """
    if type(value) not in (int, float):
        raise ReproduceManifestError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ReproduceManifestError(f"{what} must be a finite number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ReproduceManifestError(f"{what} must be >= {minimum}, got {value!r}")
    return value


def _require_text(value: Any, what: str, *, choices: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReproduceManifestError(f"{what} must be a non-empty string, got {value!r}")
    if choices is not None and value not in choices:
        raise ReproduceManifestError(
            f"{what} must be one of {list(choices)}, got {value!r}"
        )
    return value


def _require_plain_filename(value: Any, what: str) -> str:
    """Un nombre de fichero DENTRO del paquete, sin ruta.

    Un `../../etc/passwd` en `artifacts.*.path` haría que el manifiesto
    apuntara fuera del paquete, y quien lo verifique (C2) lo seguiría. El
    sandbox completo es del contrato 81, pero escribir aquí una ruta que sale
    del paquete es una forma imposible y se rechaza en origen.

    **AUDITORÍA 1ª pasada (2026-08-20): esta función estaba bien y no
    bastaba.** El párrafo de arriba decía «quien lo verifique lo seguiría»
    —y era literalmente cierto: `verify` lo seguía—. Rechazar aquí protege
    contra construir un paquete raro **con esta función**; no protege de
    nada contra un paquete que llega de fuera, porque quien lo fabrica no
    pasa por aquí. La mitad que de verdad protege es la del verificador
    (`_ruta_fuera_del_paquete`), y faltaba.

    La lección, que vale más que el arreglo: **validar en la escritura no
    valida la lectura**. Cuando el dato viene de fuera, el que tiene que
    comprobar es quien lo lee.
    """
    if not isinstance(value, str) or not value.strip():
        raise ReproduceManifestError(f"{what} must be a non-empty filename, got {value!r}")
    if value != Path(value).name or value in (".", ".."):
        raise ReproduceManifestError(
            f"{what} must be a plain filename inside the package "
            f"(no directories, no '..'), got {value!r}"
        )
    return value


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReproduceManifestError(f"{what} must be an object, got {value!r}")
    return value


def _require_str_list(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ReproduceManifestError(f"{what} must be a list of strings, got {value!r}")
    return value


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


#: El manifiesto no puede llevar su propio digest dentro, así que se
#: excluye del inventario. Su integridad la cubre `manifest_sha256`.
_FUERA_DEL_INVENTARIO = ("reproduce.json",)


def añadir_inventario_de_ficheros(bundle_dir: str | Path) -> dict[str, Any]:
    """Registra el digest de **TODO lo que viaja**, no solo de lo declarado.

    REFUTACIÓN (2026-08-20) [BLOQUEANTE]: `reproduce.json` declaraba
    cuatro artefactos —`model`, `training`, `recipe`, `dataset`— y el
    paquete llevaba **catorce ficheros**. Medido: sustituyendo
    `model.onnx`, `predict.py` y `params.best.json`, `matrixai verify`
    contestaba **`manifest PASS`**, idéntico al del paquete honesto. Y el
    Space del C4 ejecuta ese `predict.py` justo después de decir que el
    paquete está íntegro.

    El README del propio paquete prometía «the digest of each artifact
    that travels», y era falso. *Se comprobaba lo que el manifiesto
    DECLARA, no lo que el paquete LLEVA* — media limpieza, que es el
    criterio con el que este mismo contrato rechazó el `continue` mudo de
    los artefactos sin `sha256`.

    Se llama **al final** de armar el paquete, cuando ya está todo dentro:
    el `space/` y el `README.md` se escriben después del manifiesto, así
    que hacerlo antes dejaría fuera justo lo que se ejecuta.
    """
    raiz = Path(bundle_dir)
    manifiesto_path = raiz / "reproduce.json"
    if not manifiesto_path.is_file():
        return {}
    manifest = json.loads(manifiesto_path.read_text(encoding="utf-8"))

    ficheros: dict[str, str] = {}
    for fichero in sorted(raiz.rglob("*")):
        if not fichero.is_file() or fichero.is_symlink():
            continue
        relativa = fichero.relative_to(raiz).as_posix()
        if relativa in _FUERA_DEL_INVENTARIO:
            continue
        ficheros[relativa] = hashlib.sha256(fichero.read_bytes()).hexdigest()

    manifest["files"] = ficheros
    manifest["files_covered"] = len(ficheros)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    manifiesto_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


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


def _require_range_pair(valor: Any, what: str) -> None:
    """Un `[min, max]` que se pueda USAR al rehacer los datos.

    Vive suelto porque lo piden `field_ranges` (uno por campo) y
    `target_range` (uno para el objetivo), y las reglas son las mismas: dos
    números finitos y en orden. Escribirlas dos veces es como acaban
    divergiendo — este producto ya lleva catorce huecos de cableado.
    """
    if (not isinstance(valor, (list, tuple)) or len(valor) != 2
            or not all(type(x) in (int, float) and math.isfinite(x) for x in valor)):
        raise ReproduceManifestError(
            f"{what} must be a [min, max] pair of finite numbers, got {valor!r}")
    if valor[0] > valor[1]:
        # Un rango del revés no recorta nada: ni el generador sabría sacar un
        # valor de él ni el normalizador reescalar con él, y el paquete diría
        # que sí.
        raise ReproduceManifestError(f"{what} has min > max: {valor!r}")


def _epochs_from_training(training_text: str | None) -> int | None:
    """Las épocas que DECLARA el `.mxtrain`, leídas del contrato.

    Mismo criterio que `_split_seed_from_training`, y por el mismo motivo: es
    un dato del ARTEFACTO —quien recibe el paquete puede rederivarlo del
    fichero que lleva al lado, cuyo digest está en este mismo manifiesto—, así
    que la captura no lo copia y aquí no se pide que nadie lo declare. Dos
    sitios declarando lo mismo acaban divergiendo.

    Se lee para poder CONTRASTARLO con las épocas que el run se configuró a
    correr. Medido el 2026-08-19: `_apply_epoch_cap` (playground.py:1430)
    recorta con el tope del operador (`MATRIXAI_MAX_EPOCHS`, que traen los
    perfiles) o con el override del cliente, y el `.mxtrain` sigue diciendo
    `EPOCHS 50` mientras corrieron 3. Quien reentrenara con el contrato del
    paquete correría 50 y no llegaría a estos pesos.
    """
    if training_text is None:
        return None
    try:
        from matrixai.training.parser import parse_training_text
        spec = parse_training_text(training_text)
    except Exception:  # noqa: BLE001 — un .mxtrain que no parsea no invalida el paquete
        return None
    run = getattr(spec, "run", None)
    epocas = getattr(run, "epochs", None) if run is not None else None
    return epocas if type(epocas) is int else None


def _validate_generation_shapes(caller: dict[str, Any]) -> None:
    """La forma de los parámetros efectivos de `generation` (§5-C1).

    No se exige que estén —lo que no se sepa va `null` y se ve—, pero lo que
    llegue tiene que poder USARSE al regenerar. Los nombres del manifiesto no
    son los del generador (medido: `field_ranges` es `field_ranges_override`,
    `excluded_identifiers` es `field_identifiers`), así que quien reproduzca
    los va a traducir: si la forma no cuadra, falla ahí y no aquí.

    `one_hot_groups` NO se le exige nada más que la forma porque en el
    generador es una SALIDA, no una entrada (playground.py:853): viaja para
    poder comparar, no para reinyectarlo.

    `mode` no entra en los requisitos de R1 a propósito. Medido el 2026-08-19
    con `fall-risk` + receta, 30 filas, semilla 42: `random` y `coherent`
    dieron el MISMO csv (sha256 `551b392465a377c2…` los dos). No hay medida
    que diga que cambia el resultado, así que exigirlo declararía
    irreproducibles paquetes que sí lo son, por suposición.
    """
    if "mode" in caller and caller["mode"] is not None:
        _require_text(caller["mode"], "generation.mode", choices=_GENERATION_MODES)
    if "backend" in caller and caller["backend"] is not None:
        _require_text(caller["backend"], "generation.backend", choices=_TRAINING_BACKENDS)
    if "device" in caller and caller["device"] is not None:
        # `device` no lleva vocabulario cerrado: el producto escribe "cpu",
        # "cuda" y también "cuda:0". Cerrarlo aquí rechazaría un dispositivo
        # real por no estar en una lista escrita a mano.
        _require_text(caller["device"], "generation.device")
    if "field_ranges" in caller and caller["field_ranges"] is not None:
        rangos = _require_mapping(caller["field_ranges"], "generation.field_ranges")
        for campo, valor in rangos.items():
            _require_range_pair(valor, f"generation.field_ranges[{campo!r}]")
    if "target_range" in caller and caller["target_range"] is not None:
        # LOS RANGOS DECIDEN CON QUÉ DATOS SE ENTRENÓ, no cómo se dibujan.
        # Medido el 2026-08-19: el MISMO CSV crudo con y sin rangos da un
        # `prepared` distinto y una pérdida distinta, porque
        # `_normalize_csv_with_ranges` reescribe las columnas. Sin ellos,
        # rehacer R1 con lo publicado no casa.
        _require_range_pair(caller["target_range"], "generation.target_range")
    if "epochs_effective" in caller and caller["epochs_effective"] is not None:
        # Las que el run se CONFIGURÓ a correr, ya con el tope. Cero épocas no
        # es un entrenamiento y `True` no es un número de épocas (en Python
        # `isinstance(True, int)` es True), que es lo que corta este require.
        _require_positive_int(caller["epochs_effective"], "generation.epochs_effective")
    if "epochs_ran" in caller and caller["epochs_ran"] is not None:
        # `0` SÍ es un hecho aquí —un run cancelado antes de terminar la
        # primera época—, así que se pide entero NO NEGATIVO y no positivo:
        # confundirlo con un valor imposible tiraría una captura honesta.
        _require_int(caller["epochs_ran"], "generation.epochs_ran")
        if caller["epochs_ran"] < 0:
            raise ReproduceManifestError(
                f"generation.epochs_ran must be >= 0, got {caller['epochs_ran']!r}")
        # Y no se puede correr MÁS de lo que el run se configuró a correr:
        # `epochs_effective` es el tope ya aplicado y `epochs_ran` lo que cupo
        # dentro. El core no lo produce —`epochs_ran` es `len(job["epochs"])`
        # de ESE run, no un acumulado—, así que un `ran > effective` solo llega
        # de una captura escrita a mano, y sin este corte salía
        # `reproducible: true` (medido 2026-08-19). Se nombran las DOS cifras:
        # «imposible» a secas obliga a adivinar cuál de ellas está mal.
        efectivas = caller.get("epochs_effective")
        if type(efectivas) is int and caller["epochs_ran"] > efectivas:
            raise ReproduceManifestError(
                f"generation.epochs_ran ({caller['epochs_ran']}) cannot exceed "
                f"generation.epochs_effective ({efectivas}): the run cannot have "
                f"executed more epochs than it was configured to run")
    if "warm_start" in caller and caller["warm_start"] is not None:
        # DE QUÉ PESOS PARTIÓ, con la forma que escribe el core: `false` (de la
        # inicialización), un objeto con la huella de los pesos de partida, o
        # `null` (llegaron pesos y el run no llegó a decir si los usó). `true`
        # a secas NO vale: diría que hubo warm start sin decir de qué pesos, y
        # eso no se puede contrastar con nada.
        warm = caller["warm_start"]
        if warm is True or not isinstance(warm, (bool, dict)):
            raise ReproduceManifestError(
                "generation.warm_start must be false (started from the "
                "initialisation) or an object identifying the weights the run "
                f"started from, got {warm!r}")
        if isinstance(warm, dict):
            if warm.get("sha256") is not None:
                _require_full_sha256(warm["sha256"], "generation.warm_start.sha256")
            for clave in ("tensors", "params"):
                if warm.get(clave) is not None:
                    _require_positive_int(warm[clave], f"generation.warm_start.{clave}")
    if "field_types" in caller and caller["field_types"] is not None:
        tipos = _require_mapping(caller["field_types"], "generation.field_types")
        for campo, valor in tipos.items():
            _require_text(valor, f"generation.field_types[{campo!r}]")
    for clave in ("field_categories", "one_hot_groups"):
        if clave in caller and caller[clave] is not None:
            grupos = _require_mapping(caller[clave], f"generation.{clave}")
            for campo, valor in grupos.items():
                _require_str_list(valor, f"generation.{clave}[{campo!r}]")
    if "excluded_identifiers" in caller and caller["excluded_identifiers"] is not None:
        _require_str_list(caller["excluded_identifiers"], "generation.excluded_identifiers")
    if ("deterministic_options" in caller
            and caller["deterministic_options"] is not None):
        _require_mapping(caller["deterministic_options"], "generation.deterministic_options")


def validate_generation_payload(
    caller: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Qué es un `generation` POSIBLE. Devuelve `(resto, semillas)`.

    Vive suelta porque hay DOS sitios que lo necesitan y solo puede haber una
    definición: el bloque que se construye desde la captura, y el `generation`
    que manda quien exporta —que ya no rellena nada (82-C2) pero se sigue
    validando igual—.

    Medido el 2026-08-19, y por eso está aquí: al pasar a construir el bloque
    SOLO desde la captura, el `generation` del payload dejó de tocar esta
    función y un `seeds: {"dataset": "42"}` pasó de excepción a colarse sin
    mirar. Salía `reproducible: false` por otro motivo, así que el banco
    adversarial no lo veía; el CLI, que valida el sidecar llamando a este
    core, sí lo habría dejado pasar.

    Un valor AUSENTE es una respuesta legítima («no hubo override») y un valor
    IMPOSIBLE es un fallo de cableado de quien exporta: lo primero viaja como
    `null`, lo segundo se corta aquí.
    """
    if caller is not None and not isinstance(caller, dict):
        raise ReproduceManifestError(f"generation must be an object, got {caller!r}")
    caller = dict(caller or {})
    seeds_in = caller.pop("seeds", None) or {}
    _require_mapping(seeds_in, "generation.seeds")

    # LAS SEMILLAS, VALIDADAS. Medido antes de tocar nada: un `"42"` de texto
    # entraba tal cual y el paquete salía `reproducible: true`. Quien lo
    # recibiera intentaría sembrar con una cadena y obtendría otro dataset —o
    # un error— con el manifiesto diciéndole que debería salir el mismo.
    for nombre, valor in seeds_in.items():
        if valor is not None:
            _require_int(valor, f"generation.seeds[{nombre!r}]")

    # El resto de parámetros efectivos: se valida la FORMA, no se exige su
    # presencia. Un `null` aquí sigue siendo una respuesta legítima; lo que no
    # vale es un rango que no es un rango.
    _validate_generation_shapes(caller)
    return caller, seeds_in


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
    caller, seeds_in = validate_generation_payload(caller)

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
        # LAS ÉPOCAS, TRES CIFRAS Y NINGUNA SUSTITUYE A OTRA. `declared` sale
        # del `.mxtrain` que viaja (la captura no lo copia: se rederiva del
        # artefacto), `effective` es lo que el run se configuró a correr ya con
        # el tope, y `ran` lo que corrió. Resumirlas en una es justo lo que
        # dejaba pasar un paquete con «EPOCHS 50» en el contrato y unos pesos
        # de 3 épocas.
        "epochs_declared": _epochs_from_training(training_text),
        "epochs_effective": caller.pop("epochs_effective", None),
        "epochs_ran": caller.pop("epochs_ran", None),
        "field_ranges": caller.pop("field_ranges", None),
        # Los rangos del objetivo van con los de las entradas: los dos deciden
        # el CSV preparado con el que la red entrenó.
        "target_range": caller.pop("target_range", None),
        # De dónde arrancaron los pesos: `false` = de la inicialización, un
        # objeto = de unos pesos que ya existían (con su huella), `null` = no
        # consta. Y ausente NO es `false`: una captura "1.0" no lo declaraba.
        "warm_start": caller.pop("warm_start", None),
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
# El problema confirmado (103-C1)
# ---------------------------------------------------------------------------

#: Lo que del problema confirmado viaja en el paquete. No va entero el
#: `ProblemSpec`: los predictores, su disponibilidad y las restricciones son del
#: estudio, y lo que hace falta para LEER lo que este modelo emite es qué
#: predice, con qué tarea, en qué ORDEN están sus clases y cuál es la positiva.
_PROBLEM_KEYS = ("target", "task", "classes", "positive_label",
                 "observation_unit", "prediction_time", "horizon", "intended_use")


def build_problem_block(problem: Any) -> dict[str, Any] | None:
    """El problema confirmado que viaja con el paquete, o `None` si no viaja.

    POR QUÉ ESTÁ AQUÍ, que es el criterio de cierre del 103-C1: **la clase
    positiva y el orden de las clases viajan al manifiesto**. Un vector de
    probabilidades sin el orden de sus clases no se puede leer —cuál es la
    columna 0 es justo lo que hay que saber— y una binaria sin clase positiva
    deja sin definir sensibilidad, VPP y el umbral. Hasta aquí eso vivía en el
    `.mxai` y en el `inference_spec`, y el manifiesto —que es lo que alguien
    lee para saber qué es este paquete— no lo decía.

    Acepta un `ProblemSpec` del 104-C0 o el mapa que produce su `a_json()`.
    Ausente es `None` y se ve: un paquete que no declara su problema no es un
    paquete cuyo problema esté vacío.

    Fail-closed con motivo, como el resto de este fichero: una clase positiva
    que no está entre las clases, o una regresión con clases, no es una
    respuesta —es un fallo de cableado de quien empaqueta— y se corta aquí,
    donde todavía se puede arreglar.
    """
    if problem is None:
        return None
    if hasattr(problem, "a_json"):
        crudo = problem.a_json()
        # `a_json()` de un `ProblemSpec` viene en sobre (`schema`,
        # `schema_version`, cuerpo). Se toma el cuerpo: el sobre es del 104 y
        # aquí manda el `schema_version` del manifiesto.
        cuerpo = {k: v for k, v in crudo.items() if k in _PROBLEM_KEYS}
    else:
        cuerpo = _require_mapping(problem, "problem")

    desconocidas = sorted(set(cuerpo) - set(_PROBLEM_KEYS))
    if desconocidas:
        raise ReproduceManifestError(
            f"problem carries unknown keys {desconocidas}: half interpreting a "
            f"field is what the schema version exists to prevent")

    # El vocabulario de tareas es el del 104-C0 y se COMPARTE. Una segunda lista
    # aquí acabaría divergiendo el día que se añada una tarea.
    from matrixai.estudio.vocabulario import (  # noqa: PLC0415
        TAREAS,
        TAREAS_DE_CLASIFICACION,
    )

    target = _require_text(cuerpo.get("target"), "problem.target")
    task = _require_text(cuerpo.get("task"), "problem.task", choices=TAREAS)
    clases_crudas = cuerpo.get("classes")
    positiva = cuerpo.get("positive_label")

    clases: list[str] | None = None
    if task in TAREAS_DE_CLASIFICACION:
        if clases_crudas is None:
            raise ReproduceManifestError(
                "problem.classes is required for a classification: without the "
                "ORDERED classes there is no telling which probability column "
                "belongs to which class")
        clases = _require_str_list(clases_crudas, "problem.classes")
        if len(clases) < 2:
            raise ReproduceManifestError(
                f"problem.classes must carry two or more classes, got {clases!r}")
        if len(set(clases)) != len(clases):
            raise ReproduceManifestError(
                f"problem.classes must not repeat a class, got {clases!r}")
        if task == "binary_classification":
            if positiva is None:
                raise ReproduceManifestError(
                    "problem.positive_label is required for a binary classification: "
                    "without it sensitivity, PPV and the threshold are undefined")
            _require_text(positiva, "problem.positive_label")
        if positiva is not None and positiva not in clases:
            raise ReproduceManifestError(
                f"problem.positive_label {positiva!r} is not among problem.classes "
                f"{clases!r}")
    else:
        if clases_crudas is not None:
            raise ReproduceManifestError(
                f"problem.classes does not belong to a {task}, got {clases_crudas!r}: "
                f"if they are classes, the task is not regression")
        if positiva is not None:
            raise ReproduceManifestError(
                f"problem.positive_label does not belong to a {task}, got {positiva!r}")

    horizonte = cuerpo.get("horizon")
    if horizonte is not None:
        horizonte = _require_mapping(horizonte, "problem.horizon")
        if sorted(horizonte) != ["magnitud", "unidad"]:
            raise ReproduceManifestError(
                f"problem.horizon must be {{magnitud, unidad}}, got {sorted(horizonte)}")
        _require_real(horizonte["magnitud"], "problem.horizon.magnitud", minimum=0.0)
        _require_text(horizonte["unidad"], "problem.horizon.unidad")
        horizonte = dict(horizonte)

    momento = cuerpo.get("prediction_time")
    if momento is not None:
        _require_text(momento, "problem.prediction_time")
    if horizonte is not None and momento is None:
        # La misma regla que el `ProblemSpec`, y no se relaja al publicar: un
        # horizonte sin ancla no dice desde cuándo se cuenta.
        raise ReproduceManifestError(
            "problem.horizon travels without problem.prediction_time: a horizon "
            "without an anchor does not say when it starts counting")
    for nombre in ("observation_unit", "intended_use"):
        if cuerpo.get(nombre) is not None:
            _require_text(cuerpo[nombre], f"problem.{nombre}")

    return {
        "target": target,
        "task": task,
        # EL ORDEN ES EL DATO. `classes` no es un conjunto: es la
        # correspondencia entre columna de probabilidades y clase.
        "classes": clases,
        "positive_label": positiva,
        "observation_unit": cuerpo.get("observation_unit"),
        "prediction_time": momento,
        "horizon": horizonte,
        "intended_use": cuerpo.get("intended_use"),
    }


# ---------------------------------------------------------------------------
# Métricas (§5 bis)
# ---------------------------------------------------------------------------

def _normalize_metrics(metrics: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Valida las métricas del §5 bis y declara cuáles puede usar R3.

    DOS COSAS DISTINTAS, y mezclarlas fue el defecto medido el 2026-08-19:

    * **Forma imposible → excepción.** Entraban sin mirar `value: "muy alta"`,
      `tolerance_abs: -5`, `direction: "hacia-arriba"` y
      `dataset_sha256: "data_49219efba673b8d0"` (¡la huella corta que el §6.6
      prohíbe expresamente!). Una métrica así no se puede contrastar con nada:
      publicarla es afirmar que se midió algo que no se midió.
    * **Forma INCOMPLETA → se declara, no se rechaza ni se disimula.** El
      §5 bis pide diez campos y hoy el producto no tiene tres de ellos
      (`evaluator_version` no existe en el core —medido con grep— y las dos
      tolerancias salen de repetir en la matriz de entornos, que no se ha
      hecho). Rechazarlas dejaría el paquete sin métricas; rellenarlas sería
      fabricar lo que el core no ha dicho. Así que cada métrica dice qué le
      falta (`incomplete`) y si R3 puede contrastarla (`comparable`).
    """
    if metrics is None:
        return []
    if not isinstance(metrics, list):
        raise ReproduceManifestError("metrics must be a list of objects")
    out: list[dict[str, Any]] = []
    vistas: set[tuple[str, Any]] = set()
    for i, raw in enumerate(metrics):
        if not isinstance(raw, dict):
            raise ReproduceManifestError(f"metrics[{i}] must be an object")
        nombre = raw.get("name")
        if not isinstance(nombre, str) or not nombre.strip():
            raise ReproduceManifestError(f"metrics[{i}] requires a non-empty 'name'")
        if raw.get("value") is None:
            # Una métrica sin valor no es una métrica; y un `null` aquí no es
            # «cero», es que no se midió — así que se rechaza en vez de
            # publicarla como si fuera un número.
            raise ReproduceManifestError(f"metrics[{i}] requires a 'value'")
        # Un `True` pasaría por número (`isinstance(True, int)`) y se
        # publicaría como una exactitud de 1.0.
        _require_real(raw["value"], f"metrics[{i}].value")

        if raw.get("split") is not None:
            _require_text(raw["split"], f"metrics[{i}].split", choices=_METRIC_SPLITS)
        if raw.get("dataset_sha256") is not None:
            # §6.6 otra vez, y aquí se colaba: el CLI exige 64 hex para el
            # `dataset_sha256` del manifiesto y NO lo exigía para el de dentro
            # de cada métrica, que es donde el §5 bis dice «sobre QUÉ datos se
            # midió».
            _require_full_sha256(raw["dataset_sha256"], f"metrics[{i}].dataset_sha256")
        if raw.get("direction") is not None:
            _require_text(raw["direction"], f"metrics[{i}].direction",
                          choices=_METRIC_DIRECTIONS)
        for campo in ("evaluator", "evaluator_version", "aggregation"):
            if raw.get(campo) is not None:
                _require_text(raw[campo], f"metrics[{i}].{campo}")
        for campo in ("tolerance_abs", "tolerance_rel"):
            if raw.get(campo) is not None:
                # Una tolerancia negativa no es «más estricta»: es un umbral
                # que ninguna diferencia puede cumplir, así que convertiría
                # cualquier reproducción exacta en una discrepancia.
                _require_real(raw[campo], f"metrics[{i}].{campo}", minimum=0.0)

        clave = (nombre, raw.get("split"))
        if clave in vistas:
            # Dos `accuracy` de validación con valores distintos no son dos
            # métricas: son una contradicción, y quien verifique tendría que
            # elegir cuál cree.
            raise ReproduceManifestError(
                f"metrics[{i}] repeats name={nombre!r} split={raw.get('split')!r}; "
                f"a package cannot publish two different values for the same metric"
            )
        vistas.add(clave)

        # Los campos que falten quedan a `null` y VISIBLES: una tolerancia
        # ausente se ve, una clave que no está se pasa por alto.
        entry = {field: raw.get(field) for field in _METRIC_FIELDS}

        # `direction` y `aggregation` NO son del run: son del NOMBRE de la
        # métrica, y el core las sabe. Que las pusiera quien exporta sería
        # un segundo sitio declarando lo mismo — y sin `direction` R3 no
        # puede comparar, porque «se ha movido 0,03» no dice si mejoró o
        # empeoró. Lo que declare el llamante MANDA: esto rellena, no pisa.
        # Y una métrica que el catálogo no conoce se queda sin ellas y lo
        # dice en `incomplete`, en vez de deducirlas por el sufijo.
        identidad = identidad_de_metrica(nombre)
        if identidad:
            for campo in ("direction", "aggregation"):
                if entry.get(campo) is None and identidad.get(campo) is not None:
                    entry[campo] = identidad[campo]

        # LA TOLERANCIA, medida y CON SU ALCANCE (decisión de Roberto,
        # 2026-08-20). Las dos cosas juntas o ninguna: una tolerancia sin
        # alcance haría que un `PASS` de R3 se leyera como «reproduce igual
        # en cualquier sitio» cuando lo medido es «reproduce igual AQUÍ».
        #
        # Solo si el llamante no declaró NINGUNA: quien mide su propia
        # repetibilidad sabe más que este catálogo, y pisarle la suya sería
        # decidir por él.
        if entry.get("tolerance_abs") is None and entry.get("tolerance_rel") is None:
            medida = tolerancia_medida(nombre)
            if medida and not entry.get("tolerance_scope"):
                entry["tolerance_abs"] = medida["tolerance_abs"]
                entry["tolerance_scope"] = medida["tolerance_scope"]
        for key, value in raw.items():
            if key not in entry and key not in _CORE_OWNED_METRIC_KEYS:
                entry[str(key)] = value

        faltan = [c for c in _METRIC_FIELDS if entry.get(c) is None]
        if entry["tolerance_abs"] is None and entry["tolerance_rel"] is None:
            faltan_r3 = [c for c in _METRIC_R3_FIELDS if entry.get(c) is None]
            faltan_r3.append("tolerance_abs|tolerance_rel")
        else:
            faltan_r3 = [c for c in _METRIC_R3_FIELDS if entry.get(c) is None]
        entry["incomplete"] = faltan
        entry["comparable"] = not faltan_r3
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# LA CAPTURA DEL RUN (`run_provenance`) — 82-C2
# ---------------------------------------------------------------------------
#
# POR QUÉ EXISTE. Hasta hoy el manifiesto se construía con lo que mandaba la
# PANTALLA en el momento de exportar. Medido con sondas el 2026-08-19:
# cambiando BATCH, EPOCHS, receta, filas, digest y semilla, el paquete salía
# con esos valores dentro y `reproducible: true` igual — un `reproduce.json`
# podía describir OTRO dataset y OTRO contrato que los pesos que viajaban a su
# lado, y nada en el fichero lo delataba.
#
# La captura es la única fuente autoritativa: la escribe el core AL ENTRENAR
# (`playground.py`, `job["run_provenance"]`), con datos que salen todos del
# mismo run. De ahí las tres reglas de este corte:
#
#   1. `reproduce.json` se construye SOLO desde la captura.
#   2. Lo que llegue por el payload del export sirve para DETECTAR CONFLICTO y
#      declararlo, NUNCA para rellenar. Si discrepan, manda la captura.
#   3. Sin captura no hay `reproducible: true`, y el motivo lo dice entero: el
#      paquete no puede demostrar su relación con los pesos que lleva.
#
# La 3 es la que cierra el hallazgo que quedaba abierto. Antes `reproducible`
# dependía de CINCO PRESENCIAS (`.mxtrain`, receta, digest, filas y semilla):
# estando las cinco, daba igual de dónde vinieran.


def capturar_run(
    *,
    mxai_text: str,
    mxtrain_text: str,
    dataset_csv: str | None = None,
    dataset_rows: int | None = None,
    seeds: dict[str, Any] | None = None,
    backend: str | None = None,
    device: str | None = None,
    recipe_text: str | None = None,
    warm_start: Any = None,
    recipe_verification: dict[str, Any] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """La captura de un run, construida por quien ENTRENA.

    POR QUÉ EXISTE (medido el 2026-08-25). Un paquete exportado desde el CLI
    salía **`Reproducible: no`** con cinco motivos —no consta que los pesos
    vengan de un entrenamiento, no viaja captura, no se sabe el sha256 del
    dataset, ni sus filas, ni la semilla—, o sea que **el camino de quien hace
    `pip install matrixai-core` no puede producir un paquete que se demuestre**.
    Y no era por falta de datos: el CLI genera el dataset (sabe su semilla, sus
    filas y su digest) y lo entrena (sabe que los pesos son entrenados). Lo que
    faltaba era que alguien lo escribiera.

    El validador de esta misma casa (`_normalize_run_provenance`) ya declara qué
    es una captura válida; esto lo CONSTRUYE, para que no haya un segundo sitio
    decidiendo su forma. Lo que no se sepa **se deja fuera**: un valor ausente
    es una respuesta legítima y un valor inventado no.
    """
    captura: dict[str, Any] = {
        "schema_version": _RUN_PROVENANCE_SCHEMA_VERSIONS[-1],
        "mxai_sha256": hashlib.sha256(mxai_text.strip().encode("utf-8")).hexdigest(),
        "mxtrain_sha256": hashlib.sha256(mxtrain_text.encode("utf-8")).hexdigest(),
        "mxtrain_text": mxtrain_text,
    }
    if dataset_csv is not None:
        captura["dataset_sha256_raw"] = hashlib.sha256(
            dataset_csv.encode("utf-8")).hexdigest()
    if isinstance(dataset_rows, int) and dataset_rows > 0:
        captura["dataset_rows"] = dataset_rows
    if seeds:
        # Solo las semillas que de verdad se saben: un `None` aquí dentro es
        # peor que no traer la clave, porque parece una semilla declarada.
        limpias = {k: v for k, v in seeds.items() if isinstance(v, int)}
        if limpias:
            captura["seeds"] = limpias
    if backend:
        captura["backend"] = str(backend)
    if device:
        captura["device"] = str(device)
    if recipe_text is not None and recipe_text.strip():
        captura["recipe_text"] = recipe_text
        captura["recipe_sha256"] = hashlib.sha256(
            recipe_text.encode("utf-8")).hexdigest()
    if mode:
        # CON QUÉ MODO se generó. Sin él, R1 se niega a comparar —«guessing one
        # would rebuild a different one»— y con razón: el mismo texto y la
        # misma semilla dan otro dataset en modo aleatorio.
        captura["mode"] = str(mode)
    if warm_start is not None:
        # DE QUÉ PESOS PARTIÓ. `false` es una AFIRMACIÓN —arrancó de la
        # inicialización—, y no decirlo deja al paquete sin poder descartar que
        # el modelo venga de otro cuyos pesos iniciales no viajan.
        captura["warm_start"] = warm_start
    if recipe_verification is not None:
        # ¿Se COMPROBÓ que esa receta y esa semilla regeneran el dataset? Quien
        # entrena es quien puede comprobarlo, y el core publica el veredicto
        # tal cual. Sin esto, la receta viaja como una promesa.
        captura["recipe_verification"] = recipe_verification
    # La captura se valida con el MISMO validador que la leerá después: si esto
    # no pasa por ahí, no vale de nada haberla escrito.
    _normalize_run_provenance(captura)
    return captura


def _normalize_run_provenance(raw: Any) -> dict[str, Any] | None:
    """Valida la captura del run y la devuelve normalizada (o `None` si no hay).

    Mismo criterio que el resto del módulo, y por el mismo motivo: un valor
    AUSENTE es una respuesta legítima —el run puede no saber el motor todavía,
    o no haber tenido receta— y un valor IMPOSIBLE es un fallo de cableado de
    quien entrena, que se corta aquí y no viaja dentro de un paquete.

    Lo que se exige SIEMPRE es la identidad del run: los digests del `.mxai` y
    del `.mxtrain` y el texto del contrato. Sin eso no hay captura que valga —
    es justo lo que ata el manifiesto a unos pesos concretos—, y aceptar una a
    medias devolvería el problema que este corte cierra.
    """
    if raw is None:
        return None
    cap = dict(_require_mapping(raw, "run_provenance"))

    version = cap.get("schema_version")
    if version not in _RUN_PROVENANCE_SCHEMA_VERSIONS:
        raise ReproduceManifestError(
            f"run_provenance.schema_version must be one of "
            f"{list(_RUN_PROVENANCE_SCHEMA_VERSIONS)}, got {version!r}; a capture "
            f"this core does not know how to read is not interpreted half-way"
        )

    for clave in ("mxai_sha256", "mxtrain_sha256"):
        _require_full_sha256(cap.get(clave), f"run_provenance.{clave}")
    _require_text(cap.get("mxtrain_text"), "run_provenance.mxtrain_text")

    # ¿Se COMPROBÓ que esa receta y esa semilla regeneran el dataset? El
    # vocabulario del código no se define aquí: es el de quien comprueba (el
    # backend usa `regenera_el_dataset`, `no_regenera_el_dataset`,
    # `no_se_ha_podido_comprobar`). El core decide con el BOOLEANO y publica el
    # código tal cual — dos sitios declarando el mismo vocabulario acabarían
    # divergiendo, y este producto ya lleva dieciséis huecos de cableado.
    verificacion = cap.get("recipe_verification")
    if verificacion is not None:
        v = _require_mapping(verificacion, "run_provenance.recipe_verification")
        if type(v.get("verified")) is not bool:
            raise ReproduceManifestError(
                "run_provenance.recipe_verification.verified must be a boolean "
                f"(true/false), got {v.get('verified')!r}: a verdict that is not "
                "yes or no cannot sustain a reproducibility claim")
        if v.get("code") is not None:
            _require_text(v["code"], "run_provenance.recipe_verification.code")
        if v["verified"] and cap.get("recipe_text") is None:
            raise ReproduceManifestError(
                "run_provenance.recipe_verification says the recipe was verified, "
                "but the capture carries no recipe_text: there is nothing that "
                "could have been verified")

    # Receta: o las dos cosas o ninguna. Un digest sin texto no se puede
    # recomputar y un texto sin digest no se puede contrastar; publicar media
    # receta sería media verdad tranquilizadora.
    receta_sha, receta_txt = cap.get("recipe_sha256"), cap.get("recipe_text")
    if receta_sha is not None:
        _require_full_sha256(receta_sha, "run_provenance.recipe_sha256")
    if receta_txt is not None:
        _require_text(receta_txt, "run_provenance.recipe_text")
    if (receta_sha is None) != (receta_txt is None):
        raise ReproduceManifestError(
            "run_provenance must carry recipe_sha256 and recipe_text together "
            "or neither: a digest without its text cannot be recomputed, and a "
            "text without its digest cannot be contrasted"
        )

    for clave in ("dataset_sha256_raw", "dataset_sha256_prepared"):
        if cap.get(clave) is not None:
            _require_full_sha256(cap[clave], f"run_provenance.{clave}")
    if cap.get("dataset_rows") is not None:
        _require_positive_int(cap["dataset_rows"], "run_provenance.dataset_rows")
    # Las filas que el entrenamiento CONSUMIÓ, aparte de las del fichero crudo.
    # Medido el 2026-08-19: cinco líneas en blanco daban 305 para unos pesos
    # entrenados con 300, y regenerar 305 no da ese dataset.
    if cap.get("dataset_rows_used") is not None:
        _require_positive_int(cap["dataset_rows_used"],
                              "run_provenance.dataset_rows_used")
    # Si la captura llega a declarar el estado de los pesos, se valida con el
    # MISMO vocabulario que usa quien empaqueta. No decide nada por su cuenta
    # (ver `_WEIGHTS_SOURCES`): sirve para contrastar y declarar la
    # contradicción si el paquete dice otra cosa.
    if cap.get("weights_source") is not None:
        _require_text(cap["weights_source"], "run_provenance.weights_source",
                      choices=_WEIGHTS_SOURCES)

    semillas = dict(_require_mapping(cap.get("seeds") or {}, "run_provenance.seeds"))
    for nombre, valor in semillas.items():
        if valor is not None:
            _require_int(valor, f"run_provenance.seeds[{nombre!r}]")
    cap["seeds"] = semillas

    if cap.get("backend") is not None:
        _require_text(cap["backend"], "run_provenance.backend",
                      choices=_TRAINING_BACKENDS)
    if cap.get("device") is not None:
        _require_text(cap["device"], "run_provenance.device")
    for clave in ("generator_version", "csv_serialization_version"):
        if cap.get(clave) is not None:
            _require_text(cap[clave], f"run_provenance.{clave}")

    # Los parámetros efectivos que traiga se validan con la MISMA función que
    # los del payload: dos sitios declarando qué es un rango válido acabarían
    # divergiendo, y este producto ya lleva catorce huecos de cableado.
    _validate_generation_shapes(
        {k: v for k, v in cap.items() if k not in _RUN_PROVENANCE_CORE_KEYS})

    # Una captura que no es JSON no se puede digerir ni escribir: fallaría al
    # volcar el manifiesto, ya con el paquete medio hecho.
    try:
        canonical_json(cap)
    except (TypeError, ValueError) as exc:
        raise ReproduceManifestError(
            f"run_provenance must be JSON-serialisable: {exc}") from exc
    return cap


def _capture_digest(cap: dict[str, Any]) -> str:
    """Digest de la captura ENTERA, para nombrarla sin copiarla.

    El manifiesto no repite `mxtrain_text` ni `recipe_text` —ya viajan como
    ficheros con su propio digest, y repetirlos sería el segundo sitio
    declarando lo mismo—, pero sí dice de QUÉ captura salió, y eso necesita un
    nombre que no se pueda falsificar.
    """
    return hashlib.sha256(canonical_json(cap).encode("utf-8")).hexdigest()


def _file_digest_variants(path: Path) -> set[str]:
    """Los digests con los que un MISMO artefacto puede presentarse.

    MEDIDO el 2026-08-19, y por eso esto no es una comparación cruda:

      * la captura digiere el `.mxai` ya `.strip()`eado
        (`playground.py:3398`), mientras que el bundle copia el fichero BYTE A
        BYTE (`test_bundle_ships_the_mxtrain_byte_for_byte`);
      * la receta se escribe en el paquete como `receta.strip() + "\n"`
        (`bundle.py`), y la captura guarda el texto tal como llegó.

    O sea que un salto de línea final —que no cambia el programa ni la receta—
    daría dos digests distintos. Declararlo «conflicto» sería una falsa alarma
    en TODOS los paquetes reales, y una alarma que salta siempre no avisa de
    nada. Se comparan las dos formas: el fichero tal cual y su texto sin
    espacios de borde. Un artefacto de verdad distinto no casa con ninguna.
    """
    variantes = {sha256_file(path)}
    try:
        variantes.add(sha256_text(path.read_text(encoding="utf-8").strip()))
    except (OSError, UnicodeDecodeError):
        # Un artefacto binario no tiene «texto sin espacios»: se queda con su
        # digest de bytes, que es el que corresponde.
        pass
    return variantes


def _text_digest_variants(text: str | None) -> set[str]:
    """Lo mismo por el otro lado: los digests del texto que guardó la captura."""
    if text is None:
        return set()
    return {sha256_text(text), sha256_text(text.strip())}


def _conflict(field: str, source: str, captured: Any, received: Any) -> dict[str, Any]:
    """Un conflicto declarado: el campo, de dónde salió el otro valor, y los dos.

    Se declaran LOS DOS valores a propósito. Decir solo «no coinciden» obliga a
    quien recibe el paquete a adivinar cuál es cuál, y el que manda —el
    capturado— es justo el que no está en ninguna otra parte del fichero.
    """
    return {"field": field, "source": source, "captured": captured, "received": received}


#: De dónde vino el valor que contradice a la captura. No es decorativo: un
#: fichero del paquete que no casa con la captura y una pantalla que manda otro
#: número se arreglan en sitios distintos.
#: `training_contract` es el `.mxtrain` (el que la captura guardó, que es el
#: mismo que viaja salvo que ya haya un conflicto de digest declarado sobre él):
#: lo que ese contrato DECLARA frente a lo que el run hizo de verdad.
_CONFLICT_SOURCES = ("bundle_file", "export_payload", "running_core",
                     "training_contract")

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
    run_provenance: dict[str, Any] | None = None,
    weights_source: str | None = None,
    problem: Any | None = None,
    embedding_provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye el `reproduce.json` de un bundle YA escrito en `bundle_dir`.

    `run_provenance` es LA CAPTURA que el core guardó en el run al entrenar, y
    es la única fuente de lo que el manifiesto AFIRMA: dataset, semillas,
    motor, máquina y versiones salen de ahí y de ningún otro sitio.

    El resto de argumentos (`dataset_sha256`, `dataset_rows`, `generation`) son
    lo que manda quien exporta. Se validan igual —un valor imposible es un
    fallo de cableado y se corta aquí— pero NO rellenan nada: sirven para
    detectar que la pantalla dice una cosa y el run dijo otra, y entonces el
    manifiesto lo declara en `conflicts` y manda la captura.

    `weights_source` es el ESTADO DE LOS PESOS que van en este paquete
    (`"trained"` o `"untrained"`), y lo declara quien empaqueta porque es el
    único que los ha visto: la captura se compone al empezar a entrenar y no
    puede testificar sobre unos bytes elegidos después. Sin él no se puede
    decir `reproducible: true` — «no consta» no es «entrenado»—, y lo que se
    declare viaja DENTRO del paquete, en `weights.source`: hasta hoy el aviso
    vivía solo en la respuesta HTTP del export, que no acompaña al ZIP.

    Los digests de los artefactos se calculan sobre los ficheros REALES del
    paquete, no sobre lo que el llamante diga que puso: un manifiesto que
    repite lo que le cuentan no verifica nada.
    """
    bundle_dir = Path(bundle_dir)

    # ── VALIDACIÓN, ANTES DE PROMETER NADA ────────────────────────────────
    # Aquí se corta lo imposible venga de donde venga. A partir de esta línea,
    # «hay un dataset_sha256» ya significa «hay un sha256 que sirve» — y eso
    # vale tanto para lo que manda la pantalla como para lo que guardó el run.
    if dataset_sha256 is not None:
        _require_full_sha256(dataset_sha256, "dataset_sha256")
    if dataset_rows is not None:
        # Cero filas no es un dataset y las negativas no existen. Y ojo con el
        # `type is int`: `dataset_rows=True` habría publicado «1 fila».
        _require_positive_int(dataset_rows, "dataset_rows")
    if weights_source is not None:
        # Un estado que no está en el vocabulario no es «desconocido»: es un
        # fallo de cableado de quien empaqueta, y se corta aquí — publicarlo
        # como `null` lo confundiría con «no consta», que es otra cosa.
        _require_text(weights_source, "weights_source", choices=_WEIGHTS_SOURCES)
    for etiqueta, valor in (("model_filename", model_filename),
                            ("training_filename", training_filename),
                            ("recipe_filename", recipe_filename)):
        if valor is not None:
            _require_plain_filename(valor, etiqueta)

    # El `generation` del payload se valida AUNQUE no rellene nada. Que ya no
    # se publique no lo convierte en inofensivo: quien exporta se merece
    # enterarse de que su semilla es una cadena en el sitio donde todavía se
    # puede arreglar, y el CLI valida su sidecar llamando aquí (ensayo en seco
    # sobre un directorio vacío, `cli.py::_load_reproduce_metadata`).
    validate_generation_payload(generation)
    # EL COMPONENTE DE TERCEROS (107-C2), también ANTES de escribir nada. Un
    # embedding preentrenado es opaco y el expediente lo escribe así; si la
    # declaración llega a medias, el paquete se para aquí —donde todavía se
    # puede arreglar— en vez de viajar con medio componente, que se lee como
    # si el proveedor estuviera declarado. `None` es «este paquete no usa
    # ninguno», que no es lo mismo que «no consta»: sin columnas de texto no
    # hay proveedor que declarar.
    if embedding_provider is not None:
        try:
            embedding_provider = bloque_para_el_manifiesto(embedding_provider)
        except ComponenteDeTercerosIncompleto as exc:
            raise ReproduceManifestError(
                f"embedding_provider incompleto: falta o no vale {exc.faltan}"
            ) from None
    # El problema se valida ANTES de escribir nada, igual que todo lo de arriba:
    # un `positive_label` que no está entre las clases es un fallo de cableado y
    # se corta donde todavía se puede arreglar.
    problem_block = build_problem_block(problem)

    capture = _normalize_run_provenance(run_provenance)

    def _artifact(kind: str, filename: str | None) -> dict[str, Any] | None:
        if not filename:
            return None
        path = bundle_dir / filename
        if not path.is_file():
            return None
        # UN FICHERO VACÍO NO ES UN ARTEFACTO (2026-08-19).
        #
        # Se comprobaba `is_file()` y nada más, así que un `.mxtrain` de
        # CERO BYTES viajaba en el paquete y el manifiesto decía
        # `reproducible: yes`. Medido con el CLI: reentrenar con esos
        # bytes exactos responde «mxai_text y training_text son
        # obligatorios» (`playground.py`), o sea que el paquete prometía
        # algo que su propio contenido no permite. Lo destapó una
        # auditoría adversarial.
        #
        # Se mira que tenga contenido REAL —no solo espacios—: un fichero
        # con un salto de línea tampoco es un contrato de entrenamiento.
        try:
            if not path.read_text(encoding="utf-8", errors="replace").strip():
                return None
        except OSError:
            return None
        return {
            "path": filename,
            "media_type": _MEDIA_TYPES[kind],
            "sha256": sha256_file(path),
        }

    model = _artifact("model", model_filename)
    training = _artifact("training", training_filename)
    recipe = _artifact("recipe", recipe_filename)

    conflicts: list[dict[str, Any]] = []
    ignorados: list[str] = []

    # ── ¿SON ESTOS LOS ARTEFACTOS DEL RUN? ────────────────────────────────
    # Esta es la comprobación que da sentido a todo lo demás: el paquete lleva
    # unos pesos y un manifiesto, y sin esto nada ata lo uno a lo otro. Un
    # `.mxai` distinto del que entrenó, o una receta que el run no usó, se
    # declaran aquí y dejan el paquete en `reproducible: false`.
    for kind, art, sha_cap, txt_cap in (
        ("model", model, (capture or {}).get("mxai_sha256"), None),
        ("training", training, (capture or {}).get("mxtrain_sha256"),
         (capture or {}).get("mxtrain_text")),
        ("recipe", recipe, (capture or {}).get("recipe_sha256"),
         (capture or {}).get("recipe_text")),
    ):
        if art is None:
            continue
        if capture is None:
            # `null` y no `false`: sin captura no es que NO case, es que no hay
            # con qué comparar. Un valor ausente no es un cero.
            art["matches_capture"] = None
            continue
        esperados = ({sha_cap} if sha_cap else set()) | _text_digest_variants(txt_cap)
        casa = bool(esperados & _file_digest_variants(bundle_dir / art["path"]))
        art["matches_capture"] = casa
        if not casa:
            # `captured=None` con un fichero presente NO es un descuido: dice
            # que el run no tuvo ese artefacto y el paquete lo lleva igual
            # —una receta añadida al exportar sobre un modelo entrenado con
            # datos reales—, que es de las peores mentiras que puede contar.
            conflicts.append(_conflict(f"artifacts.{kind}.sha256", "bundle_file",
                                       sha_cap, art["sha256"]))

    training_text: str | None = None
    if training is not None:
        training_text = (bundle_dir / training["path"]).read_text(encoding="utf-8")

    # ── `generation`: SOLO desde la captura ───────────────────────────────
    # Sin captura no se publica un solo parámetro efectivo del run: lo que
    # mandó la pantalla no es una respuesta sobre lo que PASÓ. Se conserva
    # aparte, en `provenance.unverified_payload`, para no perderlo en silencio.
    if capture is not None:
        fuente_gen = {k: v for k, v in capture.items()
                      if k not in _RUN_PROVENANCE_CORE_KEYS}
        fuente_gen["seeds"] = dict(capture["seeds"])
        fuente_gen["backend"] = capture.get("backend")
        fuente_gen["device"] = capture.get("device")
        # El contrato de la CAPTURA, no el fichero del paquete: si el `.mxtrain`
        # que viaja fuese otro, ya está declarado como conflicto arriba, y la
        # semilla del split tiene que salir del que entrenó de verdad.
        texto_contrato = capture["mxtrain_text"]
    else:
        fuente_gen = None
        # La semilla del SPLIT sí se lee del `.mxtrain` que viaja: no es un
        # dato del payload, es un dato del ARTEFACTO, y su digest está en este
        # mismo manifiesto — quien recibe el paquete puede rederivarla.
        texto_contrato = training_text

    generation_block = build_generation_block(fuente_gen, training_text=texto_contrato)

    # Las versiones que deciden el resultado las escribe el core... pero la que
    # importa es la que CORRIÓ, no la que este proceso tiene hoy. Si el run se
    # capturó con otro generador y el core se ha actualizado desde entonces,
    # regenerar aquí no da lo mismo: manda la captura y la diferencia se
    # declara. Una nota vieja miente igual que un dato falso.
    if capture is not None:
        for clave in ("generator_version", "csv_serialization_version"):
            capturado = capture.get(clave)
            if capturado is None:
                continue
            del_core = generation_block[clave]
            generation_block[clave] = capturado
            if capturado != del_core:
                conflicts.append(
                    _conflict(f"generation.{clave}", "running_core", capturado, del_core))

    # ── LAS ÉPOCAS QUE DICE EL CONTRATO Y LAS QUE CORRIERON ───────────────
    # Medido el 2026-08-19: `_apply_epoch_cap` recorta con el tope del operador
    # (`MATRIXAI_MAX_EPOCHS`, que traen los perfiles) o con el override del
    # cliente, y el `.mxtrain` del paquete sigue diciendo `EPOCHS 50` mientras
    # corrieron 1, 4 o 50. Quien reentrene con ese contrato correrá 50 y no
    # llegará a estos pesos: el paquete lleva DOS versiones de la misma cifra y
    # no puede prometer ninguna. La captura manda —es lo que PASÓ— y la
    # diferencia se declara en vez de callarse.
    if capture is not None:
        _declaradas = generation_block.get("epochs_declared")
        _efectivas = generation_block.get("epochs_effective")
        if (_declaradas is not None and _efectivas is not None
                and _declaradas != _efectivas):
            conflicts.append(_conflict("generation.epochs", "training_contract",
                                       _efectivas, _declaradas))

    # ── El dataset, por su valor ESPERADO ─────────────────────────────────
    # §6.4: el paquete no incrusta datos cuya licencia prohíba redistribuirlos,
    # y aquí es gratis porque se empaqueta la regla que los produce.
    #
    # Van los DOS digests porque son dos preguntas distintas (medido el
    # 2026-08-19, y no coinciden): el CRUDO es el del CSV tal como llega, que
    # es contra el que se compara un dataset regenerado (R1, «byte a byte»); el
    # PREPARADO es el del fichero con el que la red entrenó de verdad, después
    # de `_normalize_external_csv` y `_normalize_csv_with_ranges`, y es el que
    # ata estos pesos.
    dataset: dict[str, Any] | None = None
    esperados_ds: set[str] = set()
    if capture is not None:
        esperados_ds = {d for d in (capture.get("dataset_sha256_raw"),
                                    capture.get("dataset_sha256_prepared")) if d}
        if (esperados_ds or capture.get("dataset_rows") is not None
                or capture.get("dataset_rows_used") is not None):
            dataset = {
                "sha256": capture.get("dataset_sha256_raw"),
                "sha256_prepared": capture.get("dataset_sha256_prepared"),
                # DOS RECUENTOS, como los dos digests y por lo mismo: `rows`
                # son las filas del CSV CRUDO —lo que hay que regenerar para
                # R1— y `rows_used` las que el entrenamiento consumió del
                # preparado. Medido el 2026-08-19: hoy coinciden en todos los
                # casos probados, y publicarlos por separado hace VISIBLE el
                # día que dejen de coincidir en vez de taparlo. `null` en
                # `rows_used` dice «esta captura no lo contó», no «las mismas».
                "rows": capture.get("dataset_rows"),
                "rows_used": capture.get("dataset_rows_used"),
            }

    metrics_block = _normalize_metrics(metrics)

    # ── LO QUE MANDÓ LA PANTALLA: contrastar, nunca rellenar ──────────────
    if capture is not None:
        if dataset_sha256 is not None:
            if not esperados_ds:
                ignorados.append("dataset_sha256")
            elif dataset_sha256 not in esperados_ds:
                # Se acepta el crudo O el preparado. No es laxitud: medido el
                # 2026-08-19, lo que el producto manda hoy por el payload
                # (`trained_csv_sha256`) es el PREPARADO, así que exigir el
                # crudo declararía un conflicto en todos los paquetes reales —
                # y una alarma que salta siempre no avisa de nada. Cualquier
                # otro valor es otro dataset.
                conflicts.append(_conflict("artifacts.dataset.sha256", "export_payload",
                                           capture.get("dataset_sha256_raw"),
                                           dataset_sha256))
        if dataset_rows is not None:
            # Se acepta el recuento del CRUDO o el del PREPARADO, por el mismo
            # motivo medido que con los dos digests: quien exporta manda una de
            # las dos cifras según de dónde la saque, y declarar conflicto por
            # la otra haría saltar la alarma en paquetes honestos. Cualquier
            # tercera cifra es otro dataset.
            esperadas = {n for n in (capture.get("dataset_rows"),
                                     capture.get("dataset_rows_used")) if n is not None}
            if not esperadas:
                ignorados.append("dataset_rows")
            elif dataset_rows not in esperadas:
                conflicts.append(_conflict("artifacts.dataset.rows", "export_payload",
                                           capture.get("dataset_rows"), dataset_rows))

        payload_gen = dict(generation or {})
        for nombre, valor in dict(payload_gen.pop("seeds", None) or {}).items():
            if valor is None:
                continue
            capturado = generation_block["seeds"].get(nombre)
            if capturado is None:
                ignorados.append(f"generation.seeds.{nombre}")
            elif capturado != valor:
                conflicts.append(_conflict(f"generation.seeds.{nombre}",
                                           "export_payload", capturado, valor))
        for clave, valor in payload_gen.items():
            if valor is None:
                continue
            capturado = generation_block.get(clave)
            if capturado is None:
                # La captura no declara este parámetro: no hay con qué
                # discrepar, y rellenarlo con lo que diga la pantalla es
                # justo lo que este corte prohíbe. Se declara ignorado para
                # que se VEA que llegó y no se publicó.
                ignorados.append(f"generation.{clave}")
            elif capturado != valor:
                conflicts.append(_conflict(f"generation.{clave}", "export_payload",
                                           capturado, valor))

        # Una métrica dice sobre QUÉ datos se midió. Si ese digest no es el del
        # run, la cifra es de otra medición y compararla con esta no significa
        # nada — R3 estaría contrastando dos cosas distintas.
        if esperados_ds:
            for i, metrica in enumerate(metrics_block):
                medida_en = metrica.get("dataset_sha256")
                if medida_en is not None and medida_en not in esperados_ds:
                    conflicts.append(
                        _conflict(f"metrics[{i}].dataset_sha256", "export_payload",
                                  capture.get("dataset_sha256_raw"), medida_en))

    # ── ¿SON ESTOS PESOS EL RESULTADO DE ESE ENTRENAMIENTO? ───────────────
    #
    # EL HALLAZGO QUE CIERRA ESTE CORTE. Medido el 2026-08-19 exportando un job
    # ANTES de que terminara de entrenar: la respuesta HTTP decía
    # `weights_source: untrained` y `reproduce.json` decía `reproducible: true`
    # sin un motivo en contra — porque el manifiesto no sabía nada de los
    # pesos—. El aviso vivía solo en la respuesta HTTP, que no acompaña al ZIP:
    # quien recibiera el paquete leía un modelo que se declara reproducible y
    # predice ruido de la inicialización aleatoria.
    #
    # La captura NO decide esto y por eso no basta con ella: se compone al
    # EMPEZAR el run y no puede testificar sobre unos bytes que se eligieron
    # después (por eso mismo `run_provenance` sobrevivía al borrado de los
    # pesos). Lo declara quien empaqueta... con un matiz: si la captura llega a
    # declararlo, solo puede DESMENTIR, nunca avalar — un run que dice que no
    # produjo pesos entrenados no se arregla porque quien exporta diga que sí,
    # y la contradicción se declara aparte.
    capturado_ws = capture.get("weights_source") if capture is not None else None
    if (capturado_ws is not None and weights_source is not None
            and capturado_ws != weights_source):
        conflicts.append(_conflict("weights.source", "export_payload",
                                   capturado_ws, weights_source))
    estado_pesos = "untrained" if capturado_ws == "untrained" else weights_source

    # Lo que impide llamar a esto «el modelo entrenado», con su código propio.
    # Va aparte de lo que falta para R1 porque no es lo mismo: el dataset se
    # puede regenerar y el modelo se puede reentrenar igual —esas etapas siguen
    # siendo posibles—; lo que no se puede es presentar ESTOS pesos como el
    # resultado del run que el manifiesto describe.
    weights_missing: list[str] = []
    if estado_pesos is None:
        # «No consta» NO es «entrenado». Es el hueco que hay que decir entero.
        weights_missing.append("weights_source")
    elif estado_pesos != "trained":
        weights_missing.append("weights_untrained")

    # DE DÓNDE ARRANCARON LOS PESOS (warm start). Medido el 2026-08-19 con
    # torch: entrenar desde cero dio loss 1.102227 y reanudar sobre un modelo
    # guardado 1.094253, con la MISMA captura byte a byte — el `.mxtrain`, las
    # semillas y los datos no determinan el resultado si se partió de otros
    # pesos, y esos pesos iniciales no viajan en el paquete.
    #
    # Tres respuestas y tres códigos, porque se arreglan en sitios distintos:
    #   * la captura NO LO DECLARA (una "1.0", de antes de que el core lo
    #     supiera): no consta, y no consta no es «partió de cero»;
    #   * `null`: llegaron pesos y el run no llegó a decir si los usó — no se
    #     puede distinguir de uno que sí, y los dos no dan los mismos números;
    #   * un OBJETO: partió de unos pesos que no viajan en el paquete.
    # `false` es la única que no bloquea, y es una afirmación del core, no un
    # hueco: medido, el camino stdlib ignora unos pesos ofrecidos y ese run sí
    # partió de la inicialización — declararlo irreproducible sería la mentira
    # del otro lado.
    if capture is not None:
        if "warm_start" not in capture:
            weights_missing.append("warm_start_unknown")
        else:
            _warm = capture["warm_start"]
            if _warm is None:
                weights_missing.append("warm_start_undecided")
            elif _warm is not False:
                weights_missing.append("warm_start")

    # ── Qué falta, y PARA QUÉ ETAPA falta (§5-C1 y §5-C2) ─────────────────
    # `model.mxai` siempre está. Del resto se declara EXACTAMENTE lo que
    # falta, con un código estable por hueco: el motivo en prosa es para
    # quien lee y los códigos son para quien traduce o automatiza.
    #
    # Y se reparte por etapas, con los nombres de C2 —`R1`, `training`,
    # `R3`— para que el manifiesto y el verificador hablen el mismo idioma.
    #
    # `run_provenance` va PRIMERO cuando falta porque no es un hueco más: sin
    # la captura, todo lo que hay debajo lo dijo quien exportó, y podría
    # describir otro dataset y otro contrato que los pesos que viajan al lado.
    #
    # `R1` = regenerar el dataset y comprobar su sha256. Medido: el generador
    # (`playground.py:535::_generate_synthetic_dataset`) recibe `training_text`
    # como argumento OBLIGATORIO, así que sin `.mxtrain` tampoco hay R1 — no
    # es solo cosa de reentrenar.
    r1_missing: list[str] = []
    if capture is None:
        r1_missing.append("run_provenance")
    if training is None:
        r1_missing.append("training")
    if recipe is None:
        r1_missing.append("recipe")
    if not (dataset and dataset.get("sha256")):
        r1_missing.append("dataset_sha256")
    if not (dataset and dataset.get("rows")):
        r1_missing.append("dataset_rows")
    if generation_block["seeds"]["dataset"] is None:
        r1_missing.append("seed_dataset")
    # Y que alguien haya COMPROBADO que esa receta con esa semilla regenera el
    # dataset. R1 es exactamente eso, así que sin la comprobación R1 no es una
    # promesa demostrada sino una declaración del cliente: medido el
    # 2026-08-20, el core acepta por HTTP una receta que no regenera el CSV y
    # el manifiesto salía `reproducible: true`. Solo se exige si HAY receta:
    # sin ella no hay nada que comprobar y un modelo entrenado con datos
    # propios es un caso honesto.
    if recipe is not None:
        _verif = (capture or {}).get("recipe_verification")
        if not (isinstance(_verif, dict) and _verif.get("verified") is True):
            r1_missing.append("recipe_verification")

    # `training` = el reentrenamiento llega a término. Necesita el `.mxtrain` y
    # los datos, y los datos salen de R1: por eso su lista es la misma. No se
    # inventa un hueco propio para que parezca que aporta algo.
    training_missing = list(r1_missing)

    # `R3` = contrastar las métricas publicadas. Pide todo lo anterior MÁS:
    #   * la semilla de inicialización —sin ella el reentrenamiento parte de
    #     otros pesos y las cifras no tienen por qué coincidir—,
    #   * el motor y el dispositivo (§6.5): el contrato 60 existió porque
    #     torch y stdlib NO daban lo mismo, así que «mismas métricas» sin
    #     declarar el motor no se puede afirmar de nadie,
    #   * y al menos una métrica que se pueda contrastar (§5 bis).
    # Los pesos van PRIMERO: sin ellos no hay nada que contrastar, se pueda o
    # no reentrenar. R1 y `training` no los llevan a propósito —regenerar el
    # dataset y reentrenar siguen siendo posibles con unos pesos sin entrenar
    # dentro del paquete—; lo que no se puede es comparar métricas contra un
    # modelo que no salió de este run.
    r3_missing = list(weights_missing) + list(training_missing)
    if generation_block["seeds"]["init"] is None:
        r3_missing.append("seed_init")
    if generation_block.get("backend") is None:
        r3_missing.append("backend")
    if generation_block.get("device") is None:
        r3_missing.append("device")
    if not any(m.get("comparable") for m in metrics_block):
        r3_missing.append("metrics")

    # `reproducible` sigue significando lo mismo que en §3 —las CINCO cosas—
    # más la captura que las respalda y unos pesos que sean los del run. Lo que
    # solo hace falta para R3 NO lo pone en falso; se declara aparte, que es lo
    # contrario de callarlo.
    #
    # El estado de los pesos va DELANTE de todo lo demás en el motivo: si lo
    # que viaja no es el modelo entrenado, lo primero que hay que leer no es
    # que falte una semilla.
    missing = list(weights_missing) + list(training_missing)

    # UN CONFLICTO DECLARADO NO PUEDE DAR `reproducible: true`. No falta nada:
    # sobra: hay dos versiones de lo mismo y una de ellas es falsa. Y bloquea
    # las tres etapas, no solo la suya, porque un paquete que se contradice no
    # se puede contrastar por ninguna parte — quien verificara sabría que algo
    # no cuadra, pero no cuál de las dos versiones estaba mirando.
    campos_en_conflicto = [c["field"] for c in conflicts]

    verifiable = {
        # `manifest` no depende del contenido: el manifiesto y su digest
        # viajan siempre, así que esta etapa siempre se puede intentar.
        "manifest": {"possible": True, "missing": [], "conflicts": [], "reason": None},
        "r1": _stage(
            "r1", r1_missing, campos_en_conflicto,
            "regenerate the dataset from the recipe and compare its full sha256"),
        "training": _stage(
            "training", training_missing, campos_en_conflicto,
            "retrain the model to completion"),
        "r3": _stage(
            "r3", r3_missing, campos_en_conflicto,
            "contrast the published metrics against their tolerance"),
    }

    reproducible = not missing and not conflicts

    # Lo que llegó por el payload y no se publicó. Va entero y con su nombre:
    # tirarlo en silencio perdería la pista de por qué el paquete no dice lo
    # que quien exportó creía que iba a decir.
    sin_verificar: dict[str, Any] | None = None
    if capture is None:
        sin_verificar = {
            "dataset_sha256": dataset_sha256,
            "dataset_rows": dataset_rows,
            "generation": generation or None,
        }
        if not any(v is not None for v in sin_verificar.values()):
            sin_verificar = None

    provenance = {
        # De dónde salió lo que este manifiesto afirma. Es la primera pregunta
        # que hay que poder contestar mirando el fichero.
        "source": "run_capture" if capture is not None else "export_payload_only",
        "run_capture": {
            "present": capture is not None,
            "schema_version": capture.get("schema_version") if capture else None,
            # Nombra la captura sin copiarla: `mxtrain_text` y `recipe_text` ya
            # viajan como ficheros con su digest, y repetirlos aquí sería el
            # segundo sitio declarando lo mismo.
            "sha256": _capture_digest(capture) if capture else None,
        },
        # El veredicto viaja DENTRO del paquete, que es lo que este corte
        # cierra: `null` cuando la captura no lo declara — «no consta» no es
        # «comprobado».
        "recipe_verification": (capture or {}).get("recipe_verification"),
        "unverified_payload": sin_verificar,
        "ignored_payload_fields": sorted(set(ignorados)),
    }

    # LO PRIMERO QUE SE LEE. El `claim` es la frase que resume el fichero, y
    # con unos pesos que no son los del run lo que hay que decir antes que nada
    # no es que falte una semilla: es que este modelo no ha aprendido nada.
    # Media verdad tranquilizadora es peor que callar.
    aviso_pesos = ""
    if "weights_untrained" in weights_missing:
        aviso_pesos = (
            "WARNING: the weights in this package are random initialisation, not "
            "the result of a training run — this model predicts nothing that was "
            "learned. "
        )
    elif "weights_source" in weights_missing:
        aviso_pesos = (
            "WARNING: this package does not state whether the weights it carries "
            "come from a training run, and 'not stated' is not 'trained'. "
        )

    manifest: dict[str, Any] = {
        "schema_version": REPRODUCE_SCHEMA_VERSION,
        # §6.7: esto demuestra COHERENCIA, no AUTENTICIDAD. Quien modifique
        # receta, manifiesto y métricas de forma coherente obtiene un paquete
        # coherente. Las firmas son del contrato 81, y prometerlas aquí sería
        # la «falsa sensación de garantía» que ese contrato identifica.
        #
        # Y SE RAMIFICA EN TRES. El texto era fijo y luego fueron dos, pero
        # «no reproducible» y «no hay con qué demostrar de qué run sale esto»
        # no son lo mismo, y el segundo es el que hay que decir entero: lo que
        # se pierde no es una comprobación, es la relación del paquete con los
        # pesos que lleva dentro.
        "claim": aviso_pesos + (
            (
                "This manifest proves the package is internally consistent and "
                "reproducible: every artifact needed to rebuild this model "
                "travels here with its digest, and each one matches the "
                "authoritative capture the core recorded while training it."
                if reproducible else
                "This manifest proves the package is internally consistent: the "
                "artifacts that do travel here match their declared digests, and "
                "what it declares about the run comes from the capture the core "
                "recorded while training. It does NOT prove the model can be "
                "reproduced — see `reproducible_reason`."
                if capture is not None else
                "This manifest proves the package is internally consistent: the "
                "artifacts that do travel here match their declared digests. It "
                "does NOT prove the model can be reproduced, and it cannot tie "
                "this package to the weights it carries: no authoritative run "
                "capture was recorded while training, so anything else stated "
                "about the run came from whoever exported it — see "
                "`reproducible_reason`."
            )
            + " It does NOT prove authorship or authenticity: signatures are "
              "out of scope here."
        ),
        "reproducible": reproducible,
        "reproducible_reason": (
            None if reproducible else _not_reproducible_reason(missing, conflicts)),
        "missing": missing,
        # Lo que sobra, no lo que falta: dos versiones del mismo dato, cuál
        # mandó y cuál se descartó.
        "conflicts": conflicts,
        "provenance": provenance,
        "verifiable": verifiable,
        "artifacts": {
            "model": model,
            "training": training,
            "recipe": recipe,
            "dataset": dataset,
        },
        # EL ESTADO DE LOS PESOS, DENTRO DEL PAQUETE. Hasta hoy solo existía en
        # la respuesta HTTP del export —`weights_source: untrained`—, que no
        # acompaña al ZIP: quien recibía el fichero no tenía cómo saber que los
        # pesos eran aleatorios, y `export_manifest.json` tampoco lo llevaba.
        # `null` significa «no consta», que no es «entrenado»; el motivo entero
        # está en `reproducible_reason`.
        "weights": {"source": estado_pesos},
        # QUÉ PREDICE ESTE MODELO, con el orden de sus clases y su clase
        # positiva (103-C1). `null` cuando nadie lo confirmó: un valor ausente
        # no es un cero, y un paquete que no declara su problema no es un
        # paquete cuyo problema esté vacío.
        "problem": problem_block,
        # EL COMPONENTE DE TERCEROS (107-C2). Va SIEMPRE, con `null` cuando no
        # hay ninguno: un paquete que calla no dice «no uso pesos ajenos», dice
        # nada — y un embedding preentrenado es lo más opaco que puede llevar
        # dentro. Lo que hay aquí sale del catálogo MEDIDO de 107-C1
        # (`matrixai.text.embeddings.declaracion`), nunca de números tecleados.
        EMBEDDING_PROVIDER_KEY: embedding_provider,
        "generation": generation_block,
        "environment": build_environment(),
        "metrics": metrics_block,
    }
    manifest["manifest_canonicalization"] = MANIFEST_CANONICALIZATION
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


_MISSING_REASONS = {
    # EL HUECO QUE CIERRA EL HALLAZGO. Y se dice entero: no es «falta un
    # campo», es que el paquete no puede demostrar de qué entrenamiento sale.
    "run_provenance": (
        "no authoritative run capture travels with this model, so the package "
        "cannot demonstrate its relation to the weights it carries: everything "
        "else declared here was supplied by whoever exported it and may describe "
        "a different dataset or a different training contract"
    ),
    # Los pesos: no es que falte una pieza para rehacer el modelo, es que el
    # que viaja no es el modelo.
    "weights_source": (
        "this package does not state whether the weights it carries come from a "
        "training run, and 'not stated' is not 'trained': a package whose weights "
        "may be random initialisation cannot be presented as the model described "
        "here"
    ),
    "weights_untrained": (
        "the weights in this package are random initialisation, not the result of "
        "the training described here — exporting before the run finished, or after "
        "the weights were cleared, produces exactly this: the model can still be "
        "rebuilt by retraining, but the one that travels is not it"
    ),
    "warm_start": (
        "this run started from weights that already existed (warm start) and those "
        "initial weights do not travel here, so retraining from the contract, the "
        "data and the seeds in this package does not have to land on these numbers "
        "— measured with torch: 1.102227 from scratch against 1.094253 resumed, "
        "with the same capture byte for byte"
    ),
    "warm_start_undecided": (
        "this run was given weights to start from and never got to state whether "
        "the trainer used them, so it cannot be told apart from a run that started "
        "from scratch — and those two do not produce the same numbers"
    ),
    "warm_start_unknown": (
        "the run capture does not say where the weights started from, so nothing "
        "here rules out that this model was retrained on top of another one whose "
        "initial weights do not travel in the package"
    ),
    "training": (
        "no .mxtrain travels in this package, so the model cannot be retrained"
    ),
    # POR QUÉ NO HAY RECETA NO LO SABE EL CORE (2ª auditoría externa del
    # 2026-08-25, hallazgo 2 residual). Esto decía «it was trained on real data
    # that is NOT REDISTRIBUTABLE», que es un hecho sobre la licencia de unos
    # datos que este código no ha visto. En la galería quedó a la vista: el caso
    # de la lluvia publica su CSV para descargar mientras su propio paquete
    # decía que no se puede redistribuir. Lo que sí consta es lo que falta —la
    # receta— y lo que eso impide.
    "recipe": (
        "this model has no data recipe, so its dataset cannot be regenerated and "
        "compared; why there is none is not recorded in the package"
    ),
    "dataset_sha256": (
        "the expected dataset sha256 is unknown, so a regenerated dataset cannot "
        "be compared against anything"
    ),
    "dataset_rows": "the expected row count is unknown",
    "seed_dataset": "the dataset generation seed is unknown",
    # Los tres siguientes NO impiden reproducir: impiden CONTRASTAR (R3).
    "seed_init": (
        "the weight-initialisation seed is unknown, so a retrained model does "
        "not have to land on the same numbers"
    ),
    "backend": (
        "the training backend is not declared, and stdlib and torch do not "
        "produce the same numbers"
    ),
    "device": "the training device is not declared",
    "metrics": (
        "no published metric carries what a comparison needs (split, the "
        "dataset sha256 it was measured on, a direction and a tolerance)"
    ),
}


def _stage(name: str, missing: list[str], conflicts: list[str],
           what: str) -> dict[str, Any]:
    """Una etapa de `matrixai verify` (§5-C2), tal como la deja este paquete.

    `possible` NO es `PASS`: dice si la etapa se puede siquiera INTENTAR con
    lo que viaja aquí. Quién gana o pierde lo dice C2 al ejecutarla; este
    manifiesto solo declara con qué se cuenta, para que un `NOT_RUN` allí
    tenga su motivo escrito aquí y no haya que adivinarlo.

    Los conflictos bloquean igual que lo que falta, y por un motivo distinto:
    no es que no haya con qué intentarlo, es que hay DOS versiones del mismo
    dato. Intentar la etapa con una de ellas mediría algo que no es lo que el
    paquete dice ser.
    """
    if not missing and not conflicts:
        return {"possible": True, "missing": [], "conflicts": [], "reason": None}
    partes = [_MISSING_REASONS.get(item, item) for item in missing]
    if conflicts:
        # Ni «el export» ni «la pantalla»: la contradicción puede venir de un
        # fichero del paquete, de lo que mandó quien exporta, de este core, o
        # del propio contrato de entrenamiento (las épocas que declara frente a
        # las que corrieron). Decir siempre «el export» mandaría a arreglarlo
        # al sitio equivocado.
        partes.append(
            "the package carries two versions of " + ", ".join(conflicts)
            + " (see `conflicts` for where each one came from), and a package "
              "that states two versions of the same value cannot be contrasted "
              "against either"
        )
    return {
        "possible": False,
        "missing": missing,
        "conflicts": conflicts,
        "reason": f"Cannot {what}: " + "; ".join(partes) + ".",
    }


def _not_reproducible_reason(missing: list[str],
                             conflicts: list[dict[str, Any]]) -> str:
    """El motivo entero: lo que falta Y lo que se contradice.

    Los dos en la misma frase a propósito. Separarlos dejaría un paquete con
    todo presente y un conflicto dentro diciendo «Not reproducible: » y nada
    más — media verdad, que en este fichero es peor que callar.
    """
    partes = [_MISSING_REASONS.get(item, item) for item in missing]
    for c in conflicts:
        partes.append(
            f"{c['field']} was captured as {c['captured']!r} and the "
            f"{c['source']} says {c['received']!r}; the capture is what counts, "
            f"and a package that carries both cannot promise either"
        )
    return "Not reproducible: " + "; ".join(partes) + "."


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
