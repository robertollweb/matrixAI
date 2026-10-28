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

EL BLOQUE `study` (138, reparación de la auditoría de C1–C3) es lo que un estudio sabe y el camino clásico no:
con qué MODO corrió (`rapido`/`completo`, el vocabulario de `MODOS_DE_ESTUDIO`), qué motores dejó fuera ese modo, y
CÓMO se repartieron las filas. Va SIEMPRE —vacío si no sabe nada—, porque su presencia es lo que dice a la ficha y a
PROBAST que esto es un estudio: aquí `artifacts.dataset.rows` son las filas del CONJUNTO (las de la huella), no las que
entrenaron. En un estudio el modelo entregado NO entrena con todas: hay una prueba reservada que no ve (y, dentro del
desarrollo, una cola interna con la que decide cuándo parar). La auditoría lo cazó: la ficha decía «Filas que
entrenaron: 360» con 72 de ellas reservadas.

139-C2 (Ola 2) — QUÉ MODELO ES EL ENTREGADO Y CÓMO SE CALIBRÓ. Desde el 139-C1 el modelo entregado se reentrena con TODO
el desarrollo (menos su cola de parada temprana) y su recalibración y su umbral salen de las predicciones del ganador
fuera de pliegue —desde el 139-C4 también una red ganadora, con las épocas fijadas de su ajuste de selección—; cuando no se
puede (el texto, una red sin épocas que fijar o que agota su techo de seguridad, un pliegue sin modelo, el ganador fuera de
pliegue peor que la referencia…) se entrega el del pliegue 0, como antes, y el estudio dice por qué. El bloque `study` lo declara, y lo que el estudio no dijo SE OMITE (un estudio guardado antes
del 139 no trae nada de esto, y la ficha dice «no disponible»: ni «pliegue 0» ni una cifra deducida):

* `study.rows.early_stopping_tail` — las filas de esa cola (no entrenan; a un árbol le dicen cuándo parar, a una red con
  épocas fijas ya no le deciden nada: ver `study.network_training`). Solo si el estudio la contó y no es cero.
* `study.delivered_engine` — el MOTOR del modelo entregado (`candidate_engine` del sobre de selección; la deuda del 138:
  el expediente no decía qué motor era).
* `study.calibration_and_threshold` — `method` (`fuera_de_pliegue` | `pliegue_0`, `METODOS_DE_CALIBRACION`), `reason`
  (`{es, en}`, solo si cayó al pliegue 0 y el estudio dijo por qué) y las cuentas que el estudio dio (`folds`,
  `out_of_fold_rows`, `rows_for_calibration`, `rows_for_threshold`). En un reparto `temporal` (140, D4; reparación del I2 de
  la auditoría del C2/C3), además: `temporal_assumption` (`{es, en}`: el modelo final predice un futuro que no vio y la
  calibración y el umbral se eligieron con el pasado), `drift_between_halves` (la deriva MEDIDA entre la mitad anterior de cada
  validación, que calibra, y la posterior, que elige el umbral: `measure` `prevalence` | `target_mean`, las dos cifras, `z`,
  `z_threshold`, las filas de cada mitad y `drift`; NO mide la deriva del desarrollo al futuro) y
  `conformal_without_exchangeability` (`{es, en}`: la cobertura nominal del intervalo conforme no es una garantía sin
  intercambiabilidad; la que cuenta es la medida en la prueba reservada).
* `study.split` (140-C2, Ola 2: validación por grupos y por tiempo) — CÓMO se repartieron las filas del examen, tal cual lo
  declara el sobre del estudio (`reparto`): `design` (`iid` | `groups` | `temporal`), `unit_column` (la columna que identifica
  al sujeto, solo en `groups`), `time_column` (solo en `temporal`), `gap` (`{magnitude, unit}`: el hueco entre lo que entrena
  y lo que se mide) y `gap_origin` (`declared_horizon`: el horizonte del problema, el valor por omisión; `requested`: lo pidió
  la persona), `out_of_fold_coverage` (`development_rows`, `rows_without_validation`: las filas de desarrollo sin predicción
  fuera de pliegue —el primer bloque de una ventana temporal—) y `limits` (lo que el reparto declaró, con su frase en los dos
  idiomas). Lo que el estudio no dijo se omite: un estudio anterior al 140 no trae el bloque. Y `target_in_development_and_test`
  (decisión de Roberto del 28-10): la proporción de cada clase (o la media y la desviación típica) del objetivo en el desarrollo y
  en la prueba reservada, el mayor cambio y si `changes_a_lot` con su `criterion`; solo informa.
* `study.network_training` (reparación de la re-auditoría del 139, I1/I2) — SOLO si el entregado es una red reentrenada con
  épocas fijas (`calibracion_y_umbral.entrenamiento_de_la_red` del sobre): `fixed_epochs` y de dónde salen
  (`best_epoch_of_selection`, `completed_epochs_of_selection`: la mejor época de la selección acotada a las completas),
  `epochs_run_in_selection`, `selection_stopped_by_time_limit` y `selection_time_limit_seconds` (si el plazo cortó la selección, el NÚMERO de épocas
  lo fijó ese reloj), `epochs_run_in_final`, `delivers` (`last_epoch`), `safety_ceiling_seconds` y
  `early_stopping_tail_decides` (`false`: con épocas fijas la cola no decide nada). Lo que el sobre no dijo se omite.

El `manifest_sha256` cubre el bloque entero, como el resto del manifiesto: cambiar el método o el motor cambia el digest.

`reproducible` es SIEMPRE `false`, con su motivo en castellano e inglés: un
estudio no se reproduce por el camino clásico de `matrixai export` /
`matrixai verify`. Afirmar lo contrario sería lo que este producto persigue.

EL ESQUEMA es el de `reproduce.py` (1.1): los mismos nombres de campo, el mismo
`build_environment()`, el mismo digest. No se inventa uno nuevo.
"""

from __future__ import annotations

import json
import math
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
           "MOTIVO_NO_REPRODUCIBLE", "METODOS_DE_CALIBRACION", "DISENOS_DEL_REPARTO"]

#: Cómo pudo repartirse un estudio (140-C2): el vocabulario de `reparto.diseno` del sobre de selección del Studio. Cerrado: un
#: diseño que no esté aquí no se publica (`groups_and_time` no lo mide el Studio todavía).
DISENOS_DEL_REPARTO: tuple[str, ...] = ("iid", "groups", "temporal")

#: `reparto.gap_origen` del sobre → `study.split.gap_origin`, en inglés.
_ORIGEN_DEL_HUECO = {"horizonte_declarado": "declared_horizon", "pedido": "requested"}

#: Cómo pudo calibrarse el modelo entregado y elegirse su umbral (139-C1): el vocabulario del sobre de selección del
#: Studio (`calibracion_y_umbral.metodo`). Cerrado: un método que no esté aquí no se publica.
METODOS_DE_CALIBRACION: tuple[str, ...] = ("fuera_de_pliegue", "pliegue_0")

#: Las cuentas que el estudio da de su calibración, en el sobre (clave del Studio) y en el manifiesto (clave inglesa).
_CUENTAS_DE_LA_CALIBRACION: tuple[tuple[str, str], ...] = (
    ("pliegues", "folds"),
    ("filas_fuera_de_pliegue", "out_of_fold_rows"),
    ("filas_para_calibrar", "rows_for_calibration"),
    ("filas_para_el_umbral", "rows_for_threshold"),
)


#: `entrenamiento_de_la_red` del sobre (clave del Studio) → `study.network_training` (clave inglesa), con su tipo.
_ENTRENAMIENTO_DE_LA_RED: tuple[tuple[str, str, str], ...] = (
    ("epocas_fijadas", "fixed_epochs", "entero"),
    ("mejor_epoca_de_la_seleccion", "best_epoch_of_selection", "entero"),
    ("epocas_completas_de_la_seleccion", "completed_epochs_of_selection", "entero"),
    ("parado_por_plazo_en_la_seleccion", "selection_stopped_by_time_limit", "booleano"),
    ("epocas_ejecutadas_de_la_seleccion", "epochs_run_in_selection", "entero"),
    ("plazo_de_la_seleccion_segundos", "selection_time_limit_seconds", "segundos"),
    ("epocas_ejecutadas_en_el_final", "epochs_run_in_final", "entero"),
    ("se_entrega", "delivers", "entrega"),
    ("techo_de_seguridad_segundos", "safety_ceiling_seconds", "segundos"),
    ("la_cola_decide_algo", "early_stopping_tail_decides", "booleano"),
)
_LO_QUE_SE_ENTREGA = {"ultima_epoca": "last_epoch"}


def _entrenamiento_de_la_red(declarado: Mapping[str, Any]) -> dict[str, Any]:
    """El bloque `study.network_training`, validado y SIN rellenar lo que el estudio no dijo (`None` = no consta: se omite)."""
    if not isinstance(declarado, Mapping):
        raise ReproduceManifestError(
            f"calibracion_y_umbral.entrenamiento_de_la_red must be an object, got {declarado!r}")
    if type(declarado.get("epocas_fijadas")) is not int or declarado["epocas_fijadas"] < 1:
        raise ReproduceManifestError(
            "calibracion_y_umbral.entrenamiento_de_la_red.epocas_fijadas must be a positive integer, got "
            f"{declarado.get('epocas_fijadas')!r}: a network block without its fixed epochs says nothing")
    bloque: dict[str, Any] = {}
    for clave_del_sobre, clave, tipo in _ENTRENAMIENTO_DE_LA_RED:
        valor = declarado.get(clave_del_sobre)
        if valor is None:
            continue
        nombre = f"calibracion_y_umbral.entrenamiento_de_la_red.{clave_del_sobre}"
        if tipo == "entero" and (type(valor) is not int or valor < 0):
            raise ReproduceManifestError(f"{nombre} must be a non-negative integer, got {valor!r}")
        if tipo == "booleano" and not isinstance(valor, bool):
            raise ReproduceManifestError(f"{nombre} must be a boolean, got {valor!r}")
        if tipo == "segundos":
            if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not valor > 0:
                raise ReproduceManifestError(f"{nombre} must be a positive number of seconds, got {valor!r}")
            valor = float(valor)
        if tipo == "entrega":
            if valor not in _LO_QUE_SE_ENTREGA:
                raise ReproduceManifestError(f"{nombre} must be one of {tuple(_LO_QUE_SE_ENTREGA)}, got {valor!r}")
            valor = _LO_QUE_SE_ENTREGA[valor]
        bloque[clave] = valor
    return bloque


def _reparto_del_estudio(declarado: Mapping[str, Any]) -> dict[str, Any]:
    """El bloque `study.split` (140-C2), validado y SIN rellenar lo que el estudio no dijo (`None` = no consta: se omite)."""
    from matrixai.estudio.vocabulario import UNIDADES_DE_TIEMPO  # noqa: PLC0415

    if not isinstance(declarado, Mapping):
        raise ReproduceManifestError(f"reparto_del_estudio must be an object, got {declarado!r}")
    bloque: dict[str, Any] = {
        "design": _require_text(declarado.get("diseno"), "reparto_del_estudio.diseno", choices=DISENOS_DEL_REPARTO)}
    for clave_del_sobre, clave in (("unit_id_field", "unit_column"), ("time_column", "time_column")):
        valor = declarado.get(clave_del_sobre)
        if valor is not None:
            bloque[clave] = _require_text(valor, f"reparto_del_estudio.{clave_del_sobre}")
    gap = declarado.get("gap")
    if gap is not None:
        # I1 de la auditoría del C2/C3 (28-10): un hueco 0 es VÁLIDO («sin hueco» es una petición legítima: la pantalla y el
        # `Horizonte` del núcleo admiten ≥ 0), y aquí se exigía > 0: un estudio `completed` con hueco 0 no se podía exportar
        # (`ReproduceManifestError` dentro de `exportar_estudio`). Lo que sigue sin valer: negativo, NaN, infinito, un booleano.
        if (not isinstance(gap, Mapping) or isinstance(gap.get("magnitud"), bool)
                or not isinstance(gap.get("magnitud"), (int, float)) or not math.isfinite(gap["magnitud"])
                or not gap["magnitud"] >= 0 or gap.get("unidad") not in UNIDADES_DE_TIEMPO):
            raise ReproduceManifestError(
                f"reparto_del_estudio.gap must be {{magnitud >= 0, unidad in {tuple(UNIDADES_DE_TIEMPO)}}}, got {gap!r}")
        bloque["gap"] = {"magnitude": gap["magnitud"], "unit": gap["unidad"]}
        origen = declarado.get("gap_origen")
        if origen is not None:
            if origen not in _ORIGEN_DEL_HUECO:
                raise ReproduceManifestError(
                    f"reparto_del_estudio.gap_origen must be one of {tuple(_ORIGEN_DEL_HUECO)}, got {origen!r}")
            bloque["gap_origin"] = _ORIGEN_DEL_HUECO[origen]
    cobertura = declarado.get("cobertura_fuera_de_pliegue")
    if cobertura is not None:
        if (not isinstance(cobertura, Mapping) or type(cobertura.get("filas_de_desarrollo")) is not int
                or type(cobertura.get("filas_sin_validacion")) is not int or cobertura["filas_de_desarrollo"] < 1
                or not 0 <= cobertura["filas_sin_validacion"] <= cobertura["filas_de_desarrollo"]):
            raise ReproduceManifestError(
                f"reparto_del_estudio.cobertura_fuera_de_pliegue must be {{filas_de_desarrollo >= 1, "
                f"0 <= filas_sin_validacion <= filas_de_desarrollo}}, got {cobertura!r}")
        bloque["out_of_fold_coverage"] = {"development_rows": cobertura["filas_de_desarrollo"],
                                          "rows_without_validation": cobertura["filas_sin_validacion"]}
    limites = declarado.get("limites")
    if limites:
        if not isinstance(limites, (list, tuple)):
            raise ReproduceManifestError(f"reparto_del_estudio.limites must be a list, got {limites!r}")
        salida = []
        for i, limite in enumerate(limites):
            motivo = limite.get("motivo") if isinstance(limite, Mapping) else None
            if (not isinstance(limite, Mapping) or not isinstance(motivo, Mapping) or set(motivo) != {"es", "en"}):
                raise ReproduceManifestError(
                    f"reparto_del_estudio.limites[{i}] must be {{clave, motivo: {{es, en}}, medida?}}, got {limite!r}")
            salida.append({"key": _require_text(limite.get("clave"), f"reparto_del_estudio.limites[{i}].clave"),
                           "reason": {idioma: _require_text(motivo[idioma], f"reparto_del_estudio.limites[{i}].motivo.{idioma}")
                                      for idioma in ("es", "en")},
                           **({"measure": dict(limite["medida"])} if isinstance(limite.get("medida"), Mapping)
                              and limite["medida"] else {})})
        bloque["limits"] = salida
    objetivo = declarado.get("objetivo_en_desarrollo_y_prueba")
    if objetivo is not None:
        bloque["target_in_development_and_test"] = _objetivo_en_desarrollo_y_prueba(objetivo)
    return bloque


def _numero_finito(valor: Any, nombre: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
        raise ReproduceManifestError(f"{nombre} must be a finite number, got {valor!r}")
    return float(valor)


def _filas_positivas(valor: Any, nombre: str) -> int:
    if type(valor) is not int or valor < 1:
        raise ReproduceManifestError(f"{nombre} must be a positive integer, got {valor!r}")
    return valor


def _objetivo_en_desarrollo_y_prueba(declarado: Mapping[str, Any]) -> dict[str, Any]:
    """`reparto.objetivo_en_desarrollo_y_prueba` (decisión de Roberto del 28-10: «decir la proporción del objetivo, o su media
    en regresión, en el desarrollo y en la prueba, y avisar si cambia mucho; solo informa»), validado, en inglés."""
    nombre = "reparto_del_estudio.objetivo_en_desarrollo_y_prueba"
    if not isinstance(declarado, Mapping) or declarado.get("tarea") not in ("clasificacion", "regresion"):
        raise ReproduceManifestError(f"{nombre} must be an object with tarea clasificacion|regresion, got {declarado!r}")
    if not isinstance(declarado.get("cambia_mucho"), bool):
        raise ReproduceManifestError(f"{nombre}.cambia_mucho must be a boolean, got {declarado.get('cambia_mucho')!r}")
    salida: dict[str, Any] = {"task": "classification" if declarado["tarea"] == "clasificacion" else "regression",
                              "changes_a_lot": declarado["cambia_mucho"]}
    for lado, side in (("desarrollo", "development"), ("prueba", "test")):
        datos = declarado.get(lado)
        if not isinstance(datos, Mapping):
            raise ReproduceManifestError(f"{nombre}.{lado} must be an object, got {datos!r}")
        bloque: dict[str, Any] = {"rows": _filas_positivas(datos.get("filas"), f"{nombre}.{lado}.filas")}
        if declarado["tarea"] == "clasificacion":
            proporciones = datos.get("proporciones")
            if not isinstance(proporciones, Mapping) or not proporciones:
                raise ReproduceManifestError(f"{nombre}.{lado}.proporciones must be a non-empty object")
            bloque["shares"] = {str(c): _numero_finito(p, f"{nombre}.{lado}.proporciones[{c!r}]")
                                for c, p in proporciones.items()}
        else:
            bloque["mean"] = _numero_finito(datos.get("media"), f"{nombre}.{lado}.media")
            bloque["standard_deviation"] = _numero_finito(datos.get("desviacion_tipica"), f"{nombre}.{lado}.desviacion_tipica")
        salida[side] = bloque
    if declarado["tarea"] == "clasificacion":
        mayor = declarado.get("mayor_cambio")
        if not isinstance(mayor, Mapping):
            raise ReproduceManifestError(f"{nombre}.mayor_cambio must be an object, got {mayor!r}")
        salida["largest_change"] = {"class": _require_text(mayor.get("clase"), f"{nombre}.mayor_cambio.clase"),
                                    "points": _numero_finito(mayor.get("puntos"), f"{nombre}.mayor_cambio.puntos"),
                                    "z": _numero_finito(mayor.get("z"), f"{nombre}.mayor_cambio.z")}
    else:
        cambio = declarado.get("cambio")
        if not isinstance(cambio, Mapping):
            raise ReproduceManifestError(f"{nombre}.cambio must be an object, got {cambio!r}")
        if cambio.get("desarrollo_sin_variacion") is True:
            # R5 de la re-auditoría del C2/C3 (28-10): el objetivo NO VARÍA en el desarrollo. Sin desviación con que medir el
            # cambio, `desviaciones` es `null` (no un cero: un cero diría «no cambió»), y `z` también lo es si la prueba es
            # constante. Solo se admite si la desviación del desarrollo es de verdad 0.
            if salida["development"]["standard_deviation"] != 0:
                raise ReproduceManifestError(f"{nombre}.cambio.desarrollo_sin_variacion needs a development standard deviation "
                                             f"of 0, got {salida['development']['standard_deviation']!r}")
            if cambio.get("desviaciones") is not None:
                raise ReproduceManifestError(f"{nombre}.cambio.desviaciones must be null when the development does not vary")
            z = cambio.get("z")
            salida["change"] = {"standard_deviations": None,
                                "z": None if z is None else _numero_finito(z, f"{nombre}.cambio.z"),
                                "development_without_variation": True}
        else:
            salida["change"] = {"standard_deviations": _numero_finito(cambio.get("desviaciones"),
                                                                      f"{nombre}.cambio.desviaciones"),
                                "z": _numero_finito(cambio.get("z"), f"{nombre}.cambio.z")}
    criterio = declarado.get("criterio")
    if isinstance(criterio, Mapping):
        salida["criterion"] = {({"puntos": "points", "desviaciones": "standard_deviations"}.get(k, k)):
                               _numero_finito(v, f"{nombre}.criterio.{k}") for k, v in criterio.items()}
    return salida


def _calibracion_y_umbral(declarada: Mapping[str, Any]) -> dict[str, Any]:
    """El bloque `study.calibration_and_threshold`, validado y SIN rellenar lo que el estudio no dijo."""
    if not isinstance(declarada, Mapping):
        raise ReproduceManifestError(
            f"calibracion_y_umbral must be an object, got {declarada!r}")
    bloque: dict[str, Any] = {
        "method": _require_text(declarada.get("metodo"), "calibracion_y_umbral.metodo",
                                choices=METODOS_DE_CALIBRACION)}
    motivo = declarada.get("motivo")
    if motivo is not None:
        if not isinstance(motivo, Mapping) or set(motivo) != {"es", "en"}:
            raise ReproduceManifestError(
                f"calibracion_y_umbral.motivo must be an object with exactly «es» and «en», got {motivo!r}")
        bloque["reason"] = {idioma: _require_text(motivo[idioma], f"calibracion_y_umbral.motivo.{idioma}")
                            for idioma in ("es", "en")}
    for clave_del_sobre, clave in _CUENTAS_DE_LA_CALIBRACION:
        valor = declarada.get(clave_del_sobre)
        if valor is None:
            continue
        if type(valor) is not int or valor < 0:
            raise ReproduceManifestError(
                f"calibracion_y_umbral.{clave_del_sobre} must be a non-negative integer, got {valor!r}")
        bloque[clave] = valor
    # I2 de la auditoría del C2/C3 (28-10; D4 de Roberto: «el sobre/ficha dicen que el umbral y la prevalencia pueden derivar con
    # el tiempo» y «la conforme se publica con … la nota de que no hay intercambiabilidad»). El sobre ya los escribía y la
    # ficha no los leía. Solo en un reparto `temporal`; ausentes, se omiten (un estudio `iid` no cambia ni un byte).
    for clave_del_sobre, clave in (("supuesto_temporal", "temporal_assumption"),
                                   ("conforme_sin_intercambiabilidad", "conformal_without_exchangeability")):
        texto = declarada.get(clave_del_sobre)
        if texto is None:
            continue
        if not isinstance(texto, Mapping) or set(texto) != {"es", "en"}:
            raise ReproduceManifestError(
                f"calibracion_y_umbral.{clave_del_sobre} must be an object with exactly «es» and «en», got {texto!r}")
        bloque[clave] = {idioma: _require_text(texto[idioma], f"calibracion_y_umbral.{clave_del_sobre}.{idioma}")
                         for idioma in ("es", "en")}
    deriva = declarada.get("deriva")
    if deriva is not None:
        bloque["drift_between_halves"] = _deriva_entre_las_mitades(deriva)
    return bloque


#: `calibracion_y_umbral.deriva` del sobre (clave del Studio) → `study.calibration_and_threshold.drift_between_halves`.
_DERIVA = (("medida", "measure"), ("en_la_mitad_que_calibra", "in_the_calibration_half"),
           ("en_la_mitad_del_umbral", "in_the_threshold_half"), ("z", "z"), ("umbral_de_z", "z_threshold"),
           ("filas_que_calibran", "calibration_rows"), ("filas_del_umbral", "threshold_rows"), ("hay_deriva", "drift"))
_MEDIDAS_DE_LA_DERIVA = {"prevalencia": "prevalence", "media_del_objetivo": "target_mean"}


def _deriva_entre_las_mitades(deriva: Mapping[str, Any]) -> dict[str, Any]:
    """La deriva MEDIDA entre la mitad anterior de cada validación (calibra) y la posterior (elige el umbral) de un reparto
    `temporal` (140-C2, D4). Mide eso y nada más: NO es la deriva del desarrollo al futuro (la prueba reservada), y la ficha
    lo dice. Todo lo que trae se valida; no se rellena nada."""
    if not isinstance(deriva, Mapping):
        raise ReproduceManifestError(f"calibracion_y_umbral.deriva must be an object, got {deriva!r}")
    salida: dict[str, Any] = {}
    for clave_del_sobre, clave in _DERIVA:
        valor = deriva.get(clave_del_sobre)
        nombre = f"calibracion_y_umbral.deriva.{clave_del_sobre}"
        if clave_del_sobre == "medida":
            salida[clave] = _MEDIDAS_DE_LA_DERIVA[_require_text(valor, nombre, choices=tuple(_MEDIDAS_DE_LA_DERIVA))]
        elif clave_del_sobre == "hay_deriva":
            if not isinstance(valor, bool):
                raise ReproduceManifestError(f"{nombre} must be a boolean, got {valor!r}")
            salida[clave] = valor
        elif clave_del_sobre in ("filas_que_calibran", "filas_del_umbral"):
            if type(valor) is not int or valor < 0:
                raise ReproduceManifestError(f"{nombre} must be a non-negative integer, got {valor!r}")
            salida[clave] = valor
        else:
            if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
                raise ReproduceManifestError(f"{nombre} must be a finite number, got {valor!r}")
            salida[clave] = float(valor)
    return salida

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
    filas_de_desarrollo: int | None = None,
    filas_de_prueba_reservada: int | None = None,
    filas_que_entrenaron_el_modelo: int | None = None,
    filas_de_cola_de_parada_temprana: int | None = None,
    motor_del_modelo_entregado: str | None = None,
    calibracion_y_umbral: Mapping[str, Any] | None = None,
    modo: str | None = None,
    motores_fuera_por_el_modo: Sequence[str] | None = None,
    reparto_del_estudio: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """El manifiesto de un estudio, con SOLO lo que el estudio sabe.

    Todo es opcional y **ausente es ausente**: ningún argumento `None` produce un
    campo con valor inventado.

    * `filas`: cuántas filas tiene el conjunto con que trabajó el estudio (positivo): desarrollo Y prueba reservada.
      NO son las que entrenaron el modelo (esas van en `filas_que_entrenaron_el_modelo`).
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
    * `filas_de_desarrollo` / `filas_de_prueba_reservada`: cómo repartió el estudio sus filas (las de la validación
      cruzada y la selección / las que se apartaron antes y solo midieron al ganador). Contadas por el estudio, nunca
      deducidas aquí.
    * `filas_que_entrenaron_el_modelo`: con cuántas filas se ajustó el modelo ENTREGADO (no puede pasar de las de
      desarrollo).
    * `filas_de_cola_de_parada_temprana` (139-C2): las filas de la cola interna de parada temprana del modelo entregado
      (no entrenan; con una red de épocas fijas tampoco deciden nada). Con `filas_que_entrenaron_el_modelo` no pueden pasar de las de desarrollo.
    * `motor_del_modelo_entregado` (139-C2): el motor del modelo entregado (`candidate_engine` de la selección).
    * `calibracion_y_umbral` (139-C2): `{metodo, motivo?, pliegues?, filas_fuera_de_pliegue?, filas_para_calibrar?,
      filas_para_el_umbral?, entrenamiento_de_la_red?}` tal cual lo declara el sobre del estudio; `metodo` en
      `METODOS_DE_CALIBRACION`. Con `entrenamiento_de_la_red` (139-C4, una red reentrenada con épocas fijas) el manifiesto
      lleva además `study.network_training`.
    * `modo`: `rapido` o `completo` (`MODOS_DE_ESTUDIO`), el modo con que corrió.
    * `motores_fuera_por_el_modo`: los motores que no compitieron POR EL MODO (no por fallar). Solo con `modo`.
    * `reparto_del_estudio` (140-C2): `{diseno, unit_id_field, time_column, gap: {magnitud, unidad}, gap_origen,
      cobertura_fuera_de_pliegue: {filas_de_desarrollo, filas_sin_validacion}, limites: [{clave, motivo: {es, en}, medida}]}`
      tal cual lo declara el sobre del estudio (`reparto`); `diseno` en `DISENOS_DEL_REPARTO`. Va en `study.split`.
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

    for valor, nombre in ((filas_de_desarrollo, "filas_de_desarrollo"),
                          (filas_de_prueba_reservada, "filas_de_prueba_reservada"),
                          (filas_que_entrenaron_el_modelo, "filas_que_entrenaron_el_modelo"),
                          (filas_de_cola_de_parada_temprana, "filas_de_cola_de_parada_temprana")):
        if valor is not None:
            _require_positive_int(valor, nombre)
    if (filas_que_entrenaron_el_modelo is not None and filas_de_desarrollo is not None
            and filas_que_entrenaron_el_modelo > filas_de_desarrollo):
        raise ReproduceManifestError(
            f"filas_que_entrenaron_el_modelo ({filas_que_entrenaron_el_modelo}) cannot exceed "
            f"filas_de_desarrollo ({filas_de_desarrollo}): the delivered model trains inside development")
    if (filas_de_cola_de_parada_temprana is not None and filas_de_desarrollo is not None
            and (filas_que_entrenaron_el_modelo or 0) + filas_de_cola_de_parada_temprana > filas_de_desarrollo):
        raise ReproduceManifestError(
            f"filas_que_entrenaron_el_modelo ({filas_que_entrenaron_el_modelo}) plus the early-stopping tail "
            f"({filas_de_cola_de_parada_temprana}) cannot exceed filas_de_desarrollo ({filas_de_desarrollo})")
    if motor_del_modelo_entregado is not None:
        _require_text(motor_del_modelo_entregado, "motor_del_modelo_entregado")
    if filas is not None:
        for valor, nombre in ((filas_de_desarrollo, "filas_de_desarrollo"),
                              (filas_de_prueba_reservada, "filas_de_prueba_reservada")):
            if valor is not None and valor > filas:
                raise ReproduceManifestError(
                    f"{nombre} ({valor}) cannot exceed the dataset rows ({filas})")
    if modo is not None:
        from matrixai.estudio.competicion import MODOS_DE_ESTUDIO  # noqa: PLC0415
        _require_text(modo, "modo", choices=MODOS_DE_ESTUDIO)
    fuera_por_el_modo: list[str] | None = None
    if motores_fuera_por_el_modo is not None:
        if modo is None:
            raise ReproduceManifestError(
                "motores_fuera_por_el_modo without modo: an engine left out «by the mode» needs the mode")
        if isinstance(motores_fuera_por_el_modo, (str, bytes)):
            raise ReproduceManifestError(
                f"motores_fuera_por_el_modo must be a list of strings, got {motores_fuera_por_el_modo!r}")
        fuera_por_el_modo = sorted(set(_require_str_list(list(motores_fuera_por_el_modo),
                                                         "motores_fuera_por_el_modo")))

    estudio: dict[str, Any] = {}
    if modo is not None:
        estudio["mode"] = modo
    if fuera_por_el_modo:
        estudio["engines_left_out_by_mode"] = fuera_por_el_modo
    reparto = {clave: valor for clave, valor in (
        ("development", filas_de_desarrollo), ("holdout_test", filas_de_prueba_reservada),
        ("trained_the_delivered_model", filas_que_entrenaron_el_modelo),
        ("early_stopping_tail", filas_de_cola_de_parada_temprana)) if valor is not None}
    if reparto:
        estudio["rows"] = reparto
    if motor_del_modelo_entregado is not None:
        estudio["delivered_engine"] = motor_del_modelo_entregado
    if reparto_del_estudio is not None:
        estudio["split"] = _reparto_del_estudio(reparto_del_estudio)
    if calibracion_y_umbral is not None:
        estudio["calibration_and_threshold"] = _calibracion_y_umbral(calibracion_y_umbral)
        if calibracion_y_umbral.get("entrenamiento_de_la_red") is not None:
            estudio["network_training"] = _entrenamiento_de_la_red(calibracion_y_umbral["entrenamiento_de_la_red"])

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
        # Siempre, aunque vacío: su presencia dice «esto es un estudio» (ver el docstring del módulo).
        "study": estudio,
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
