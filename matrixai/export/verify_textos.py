"""CONTRATO 85-C2b — los motivos de `verify`, en el idioma que se pida.

Todo lo que `matrixai/export/verify.py` REDACTA vive aquí, en los dos
idiomas y uno al lado del otro. Está aparte de `verify.py` a propósito:
así el verificador sigue leyéndose como lo que comprueba, y las dos
versiones de cada frase se ven juntas — que es la única manera de que
una traducción no se quede vieja sin que nadie lo note.

**Lo que NO se traduce, y no por descuido:**

* Los `status` (`PASS`, `FAIL`, `NOT_RUN`, `INCOMPARABLE`) y los nombres
  de etapa (`manifest`, `R1`, `training`, `R3`): son VALORES, y quien los
  lee por programa depende de ellos. Traducirlos rompería a todo el que
  encadene `verify && desplegar`.
* Lo que un motivo INTERPOLA: una ruta, una huella, un número, el nombre
  de un campo del manifiesto (`recipe`, `seed`, `dataset_sha256`) o el
  nombre de una opción del CLI (`--retrain`). Un dato no cambia de idioma.
* El texto que viene de FUERA —el mensaje de una excepción del sistema,
  el motivo que el propio paquete escribió en su `reproduce.json`—. Eso
  se CITA, y por eso viaja entre «comillas» en español y entre "comillas"
  en inglés: quien lee tiene que poder ver dónde acaba lo que dice el
  verificador y dónde empieza lo que dice otro.

El patrón es el de la casa (`_marcos` en `matrixai/playground.py`):
diccionarios por idioma y `str(locale or "es").strip().lower()`, con
respaldo al español. Ni una librería de i18n: dos idiomas y frases que
las escribe el core no piden más maquinaria.
"""

from __future__ import annotations

from typing import Any

__all__ = ["IDIOMAS", "IDIOMA_POR_DEFECTO", "idioma", "motivo"]

#: Los idiomas que este fichero sabe redactar.
IDIOMAS = ("es", "en")

#: El mismo que el resto del core (`playground.py`, medido): español.
IDIOMA_POR_DEFECTO = "es"


_MOTIVOS: dict[str, dict[str, str]] = {
    "es": {
        # -- el manifiesto, antes de mirarlo -------------------------------
        "sin_reproduce": "el paquete no lleva ningún {fichero}",
        "reproduce_ilegible": "{fichero} no es un JSON legible: «{error}»",
        "reproduce_no_es_objeto": "{fichero} no es un objeto",
        # -- rutas de artefacto (van en `problem`) --------------------------
        "p_ruta_vacia": "la ruta está vacía o lleva espacios alrededor",
        "p_ruta_absoluta": ("la ruta es absoluta, y un paquete solo describe sus "
                            "propios ficheros"),
        "p_ruta_escapa": "la ruta se sale del paquete con '..'",
        "p_ruta_irresoluble": "la ruta no se puede resolver: «{error}»",
        "p_ruta_fuera": "la ruta apunta fuera del paquete (¿enlace simbólico?)",
        # -- artefactos e inventario (van en `problem`) ----------------------
        "p_declarado_pero_ausente": ("el manifiesto lo declara y no viaja dentro "
                                     "del paquete"),
        "p_sha256_no_cuadra": "el sha256 no cuadra",
        "p_entrada_invalida": ("la entrada del artefacto no es ni un objeto ni una "
                               "ausencia declarada (null)"),
        "p_sin_sha256": ("el artefacto trae un fichero y ningún sha256 con el que "
                         "comprobarlo"),
        "p_sin_ruta_ni_sha256": "el artefacto no declara ni ruta ni sha256",
        "p_enlace_simbolico": ("el artefacto es un enlace simbólico; un artefacto "
                               "de integridad tiene que ser un fichero normal"),
        "p_ausente": "falta",
        # -- etapa `manifest` -------------------------------------------------
        "m_sin_digest": "el manifiesto no declara ningún manifest_sha256",
        "m_digest_no_cuadra": "el manifiesto no cuadra con su propio manifest_sha256",
        "m_sin_artefactos": "el manifiesto no declara ningún artefacto",
        "m_esquema_desconocido": (
            "el manifiesto declara schema_version {version} y este verificador lee "
            "{conocidos}: un formato que no conoce no se interpreta a medias"),
        "m_artefacto_no_cuadra": ("al menos un artefacto no cuadra con el sha256 "
                                  "que declara"),
        "m_fichero_no_cuadra": ("al menos un fichero del paquete no cuadra con el "
                                "sha256 que declara"),
        "m_ficheros_sin_cubrir": ("el paquete lleva ficheros que el manifiesto no "
                                  "cubre, así que su integridad no se puede "
                                  "comprobar"),
        # -- componente de terceros (107-C2) ------------------------------------
        "tp_texto_sin_proveedor": (
            "{fichero} dice que este modelo recibe columnas calculadas por un "
            "proveedor de embeddings, y el manifiesto no declara ninguno: no consta "
            "qué pesos de terceros produjeron esos números"),
        "tp_proveedor_incompleto": (
            "el paquete declara un proveedor de embeddings y su declaración está a "
            "medias; falta o no vale: {faltan}"),
        "tp_spec_sin_proveedor": (
            "{fichero} no dice con qué proveedor se calcularon las columnas de texto, "
            "así que fuera del paquete no se puede reproducir el mismo vector"),
        "tp_no_cuadra": (
            "el manifiesto y {fichero} declaran proveedores de embeddings que no "
            "cuadran en {claves}: con otro tokenizer o con otro truncado el vector es "
            "otro, aunque los pesos sean los mismos"),
        # -- etapa `R1` --------------------------------------------------------
        "r1_no_posible_segun_el_paquete": ("el propio paquete declara que R1 no es "
                                           "posible: «{cita}»"),
        "r1_no_posible": "R1 no es posible con este paquete",
        "r1_faltan_datos": ("no se puede regenerar el dataset ni comparar su "
                            "sha256: falta {faltan}"),
        "r1_sin_modelo": "el paquete no lleva ningún modelo desde el que regenerar",
        "r1_sin_mxtrain": ("el paquete no lleva el .mxtrain, y el generador lo "
                           "necesita"),
        "r1_sin_filas": "el paquete no declara cuántas filas hay que generar",
        "r1_artefactos_ilegibles": ("los artefactos del paquete no se pueden leer: "
                                    "«{error}»"),
        "r1_sin_modo": ("el paquete no declara con qué modo de generación se hizo "
                        "su dataset, y adivinar uno rehará otro distinto"),
        "r1_no_regenerable": "el dataset no se ha podido regenerar: «{error}»",
        "r1_sin_csv": "el generador no ha devuelto ningún CSV",
        "r1_digest_distinto": ("el dataset regenerado no tiene el sha256 que el "
                               "paquete declara"),
        # -- etapa `training` ---------------------------------------------------
        "t_no_pedido": "nadie pidió reentrenar (usa --retrain)",
        "t_sin_dataset": ("el dataset no se ha podido regenerar, así que no hay "
                          "con qué entrenar"),
        "t_dataset_no_guardado": "el dataset regenerado no se guardó para entrenar",
        "t_sin_mxtrain": ("el paquete no lleva ningún .mxtrain con el que "
                          "reentrenar"),
        "t_no_arranca": "el reentrenamiento no ha podido correr: «{error}»",
        "t_nota": ("el entrenamiento llegó a término; cuadrar con las métricas "
                   "publicadas es R3"),
        # -- etapa `R3` ----------------------------------------------------------
        "r3_sin_reentrenamiento": ("las métricas solo se contrastan contra un "
                                   "entrenamiento nuevo (usa --retrain)"),
        "r3_training_no_paso": "el entrenamiento no llegó a PASS",
        "r3_reentrenamiento_incompleto": (
            "el reentrenamiento no llegó a término, así que no hay valor nuevo con "
            "el que contrastar: {porque}"),
        "r3_sin_metricas": "el paquete no publica ninguna métrica que contrastar",
        "r3_metricas_incompletas": ("ninguna métrica publicada lleva lo que hace "
                                    "falta para compararla"),
        "r3_reentrenamiento_sin_metricas": ("el reentrenamiento no ha dado métricas "
                                            "que contrastar"),
        "r3_configuracion_sin_aplicar": (
            "el reentrenamiento no ha podido aplicar parte de la configuración "
            "capturada ({claves}), así que sus métricas no describen la misma "
            "ejecución y una diferencia tampoco probaría que el paquete miente"),
        "r3_ninguna_comparable": ("el reentrenamiento no ha informado de ninguna de "
                                  "las métricas comparables"),
        "r3_fuera_de_tolerancia": "al menos una métrica se sale de su tolerancia",
        # -- alcance de la tolerancia ---------------------------------------------
        "tol_alcance_desconocido": (
            "el paquete declara alcances de tolerancia que este verificador no "
            "conoce ({alcances}), así que tampoco puede saber si aplican aquí"),
        "tol_sin_digest_de_entorno": (
            "el paquete declara una tolerancia con alcance de entorno y ninguna "
            "huella de entorno, así que tampoco hay manera de saber si este es el "
            "entorno donde se midió"),
        "tol_otro_entorno": (
            "la tolerancia se midió para el entorno propio del paquete "
            "({declarado}…) y este es distinto ({actual}…), así que una diferencia "
            "aquí tampoco probaría que el paquete miente"),
        # -- el informe entero ------------------------------------------------------
        "sin_manifiesto_que_comparar": "sin manifiesto contra el que comparar",
        "sin_integridad": ("la comprobación de integridad del paquete no ha pasado, "
                           "así que ya no hay de qué fiarse"),
    },
    "en": {
        "sin_reproduce": "the package carries no {fichero}",
        "reproduce_ilegible": "{fichero} is not readable JSON: \"{error}\"",
        "reproduce_no_es_objeto": "{fichero} is not an object",
        "p_ruta_vacia": "path is empty or padded with spaces",
        "p_ruta_absoluta": ("path is absolute, and a package only describes its own "
                            "files"),
        "p_ruta_escapa": "path escapes the package with '..'",
        "p_ruta_irresoluble": "path cannot be resolved: \"{error}\"",
        "p_ruta_fuera": "path resolves outside the package (symlink?)",
        "p_declarado_pero_ausente": ("declared in the manifest but missing from the "
                                     "package"),
        "p_sha256_no_cuadra": "sha256 mismatch",
        "p_entrada_invalida": ("the artifact entry is neither an object nor a "
                               "declared absence (null)"),
        "p_sin_sha256": ("the artifact ships a file with no sha256 to check it "
                         "against"),
        "p_sin_ruta_ni_sha256": "the artifact declares neither a path nor a sha256",
        "p_enlace_simbolico": ("the artifact is a symlink; an integrity artifact "
                               "must be a regular file"),
        "p_ausente": "missing",
        "m_sin_digest": "the manifest declares no manifest_sha256",
        "m_digest_no_cuadra": "the manifest does not match its own manifest_sha256",
        "m_sin_artefactos": "the manifest declares no artifacts",
        "m_esquema_desconocido": (
            "the manifest declares schema_version {version} and this verifier reads "
            "{conocidos}: a format it does not know is not interpreted halfway"),
        "m_artefacto_no_cuadra": ("at least one artifact does not match its declared "
                                  "sha256"),
        "m_fichero_no_cuadra": ("at least one file in the package does not match its "
                                "declared sha256"),
        "m_ficheros_sin_cubrir": ("the package ships files the manifest does not "
                                  "cover, so their integrity cannot be checked"),
        # -- third-party component (107-C2) --------------------------------------
        "tp_texto_sin_proveedor": (
            "{fichero} states this model takes columns computed by an embedding "
            "provider, and the manifest declares none: which third-party weights "
            "produced those numbers is not recorded"),
        "tp_proveedor_incompleto": (
            "the package declares an embedding provider and its declaration is "
            "incomplete; missing or invalid: {faltan}"),
        "tp_spec_sin_proveedor": (
            "{fichero} does not say which provider computed the text columns, so the "
            "same vector cannot be reproduced outside the package"),
        "tp_no_cuadra": (
            "the manifest and {fichero} declare embedding providers that disagree on "
            "{claves}: with another tokenizer or another truncation the vector is a "
            "different one, even with the same weights"),
        "r1_no_posible_segun_el_paquete": ("the package itself declares R1 is not "
                                           "possible: \"{cita}\""),
        "r1_no_posible": "R1 is not possible for this package",
        "r1_faltan_datos": ("cannot regenerate the dataset and compare its sha256: "
                            "{faltan} unknown"),
        "r1_sin_modelo": "the package carries no model to regenerate from",
        "r1_sin_mxtrain": ("the package carries no .mxtrain, which the generator "
                           "requires"),
        "r1_sin_filas": "the package does not declare how many rows to generate",
        "r1_artefactos_ilegibles": "the package artifacts cannot be read: \"{error}\"",
        "r1_sin_modo": ("the package does not declare which generation mode produced "
                        "its dataset, and guessing one would rebuild a different one"),
        "r1_no_regenerable": "the dataset could not be regenerated: \"{error}\"",
        "r1_sin_csv": "the generator returned no CSV",
        "r1_digest_distinto": ("the regenerated dataset does not have the sha256 the "
                               "package declares"),
        "t_no_pedido": "retraining was not requested (use --retrain)",
        "t_sin_dataset": ("the dataset could not be regenerated, so there is nothing "
                          "to train on"),
        "t_dataset_no_guardado": "the regenerated dataset was not kept for training",
        "t_sin_mxtrain": "the package carries no .mxtrain to retrain from",
        "t_no_arranca": "retraining could not run: \"{error}\"",
        "t_nota": "training completed; matching the published metrics is R3",
        "r3_sin_reentrenamiento": ("metrics can only be contrasted against a fresh "
                                   "training run (use --retrain)"),
        "r3_training_no_paso": "training did not pass",
        "r3_reentrenamiento_incompleto": (
            "the retraining run did not complete, so there is no fresh value to "
            "contrast: {porque}"),
        "r3_sin_metricas": "the package publishes no metrics to contrast",
        "r3_metricas_incompletas": "no published metric carries what a comparison needs",
        "r3_reentrenamiento_sin_metricas": ("the retraining did not report metrics to "
                                            "contrast"),
        "r3_configuracion_sin_aplicar": (
            "the retraining could not apply part of the captured configuration "
            "({claves}), so its metrics do not describe the same run and a "
            "difference would not prove the package wrong"),
        "r3_ninguna_comparable": ("none of the comparable metrics was reported by the "
                                  "retraining"),
        "r3_fuera_de_tolerancia": "at least one metric falls outside its tolerance",
        "tol_alcance_desconocido": (
            "the package declares tolerance scopes this verifier does not know "
            "({alcances}), so it cannot tell whether they apply here"),
        "tol_sin_digest_de_entorno": (
            "the package declares an environment-scoped tolerance but no environment "
            "digest, so there is no way to tell whether this is the environment it "
            "was measured in"),
        "tol_otro_entorno": (
            "the tolerance was measured for the package's own environment "
            "({declarado}…) and this one is different ({actual}…), so a difference "
            "here would not prove the package wrong"),
        "sin_manifiesto_que_comparar": "no manifest to compare against",
        "sin_integridad": ("the package integrity check did not pass, so nothing else "
                           "can be trusted"),
    },
}


# Que los dos idiomas digan LAS MISMAS COSAS se comprueba al importar, no
# al pintar: una clave que falte en uno de los dos saldría en producción
# como una frase en el otro idioma —media pantalla traducida, que es
# justo el defecto que este corte existe para quitar— y nadie lo vería
# hasta tener delante ese caso concreto, que suele ser el raro.
_SOLO_ES = sorted(set(_MOTIVOS["es"]) - set(_MOTIVOS["en"]))
_SOLO_EN = sorted(set(_MOTIVOS["en"]) - set(_MOTIVOS["es"]))
if _SOLO_ES or _SOLO_EN:  # pragma: no cover — no puede pasar con el fichero sano
    raise RuntimeError(
        "los motivos de verify no dicen lo mismo en los dos idiomas: "
        f"solo en es={_SOLO_ES}, solo en en={_SOLO_EN}")


def idioma(locale: Any) -> str:
    """El idioma pedido, normalizado; español si no se reconoce.

    Mismo criterio que `_marcos` en `playground.py`, y a propósito: dos
    sitios normalizando distinto acabarían dando dos idiomas para la
    misma pantalla.
    """
    pedido = str(locale or IDIOMA_POR_DEFECTO).strip().lower()
    return pedido if pedido in _MOTIVOS else IDIOMA_POR_DEFECTO


def motivo(clave: str, locale: Any = IDIOMA_POR_DEFECTO, **datos: Any) -> str:
    """La frase `clave` en el idioma pedido, con sus datos dentro.

    Una clave desconocida NO devuelve una cadena vacía: revienta. Un
    motivo vacío se pinta como «esta etapa no trae motivo», que es una
    afirmación falsa sobre el paquete de otro.
    """
    plantilla = _MOTIVOS[idioma(locale)][clave]
    return plantilla.format(**datos) if datos else plantilla
