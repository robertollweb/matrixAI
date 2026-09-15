# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C2 — el componente de terceros dentro del expediente.

Aquí NO se compone la declaración: eso lo hace quien EJECUTA el proveedor, a
partir del catálogo medido de 107-C1 (`matrixai.text.embeddings.declaracion`
en el núcleo, `ProveedorDeTexto.declaracion()` en `matrixai-engines`). Lo que
vive aquí es lo otro: **comprobarla y traducirla** a los tres sitios donde el
paquete la enseña —el manifiesto, el BOM y la `inference_spec`—.

Tres decisiones que no son de estilo:

* **La comprobación devuelve NOMBRES DE CAMPO, no prosa.** `verify` redacta en
  dos idiomas (85-C2b) y un dato no cambia de idioma: lo que falta se nombra
  (`licencia.spdx`, `digest_del_tokenizer`) y la frase la pone quien pinta.
* **Una declaración a medias no se enseña a medias.** Un BOM con un componente
  de terceros al que le falta la licencia se lee como si el proveedor estuviera
  declarado, y eso es peor que no enumerarlo: media verdad tranquilizadora. O
  está entero o `lo_que_falta()` lo dice con sus campos.
* **El digest del tokenizer NO viaja en `hashes` de CycloneDX cuando su
  algoritmo es `git-blob-sha1`.** El origen publica `sha256` para los ficheros
  grandes (LFS) y el `oid` de git para los pequeños, y ese `oid` es el sha1 de
  `blob <n>\\0` + contenido: **no es el sha1 del fichero**. La lista `hash-alg`
  de CycloneDX solo admite `SHA-1`, así que meterlo ahí sería afirmar que un
  `sha1sum` del `tokenizer.json` da ese número, y no lo da. Viaja como
  propiedad, con su algoritmo escrito al lado.

Lo que este módulo NO puede cerrar de 107-C2 y por qué: el recibo (106-C2) y la
tarjeta de uso (106-C4) viven en `matrixai-engines`
(`recibo_de_estudio.py`, `tarjeta_de_uso.py`), fuera de este paquete. Consumen
el MISMO diccionario que se comprueba aquí; cablearlos es trabajo de ese
repositorio y queda declarado, no fingido.
"""
from __future__ import annotations

import copy
import json
from typing import Any

__all__ = [
    "CLAVE_EN_EL_MANIFIESTO",
    "CLAVE_EN_LA_SPEC",
    "ComponenteDeTercerosIncompleto",
    "ENCODING_DE_TEXTO",
    "bloque_para_el_manifiesto",
    "bloque_para_la_spec",
    "componente_para_el_bom",
    "exigir_completo",
    "la_spec_usa_un_proveedor",
    "las_tres_cosas",
    "validar",
]

#: Dónde se guarda la declaración dentro de `reproduce.json`.
CLAVE_EN_EL_MANIFIESTO = "embedding_provider"

#: Y dentro de `inference_spec.json`. Son dos sitios a propósito y no es una
#: copia: el manifiesto es el EXPEDIENTE (lo que se audita) y la spec es el
#: CONTRATO DE EJECUCIÓN que `predict.py` cumple fuera (106-C5 / 107-C4). Lo
#: que impide que diverjan es que `verify` compara las tres cosas del
#: invariante 6 entre los dos y falla si no cuadran.
CLAVE_EN_LA_SPEC = "embedding_provider"

#: Cómo se llama, en la `inference_spec`, una columna que un proveedor de
#: terceros convierte en números. Es el marcador por el que `verify` sabe que
#: este paquete LLEVA TEXTO: sin él, «un paquete con texto sin
#: `embedding_provider` no valida» dependería de que quien exporta se acordara
#: de declararlo, que es justo lo que no se puede suponer.
ENCODING_DE_TEXTO = "text_embedding"

#: Los orígenes admitidos de la longitud efectiva (107, invariante 6.3). Es
#: vocabulario cerrado: un número cuya procedencia no se sabe no sirve para
#: prometer un vector.
ORIGENES_DE_LA_LONGITUD = ("libreria_del_proveedor", "paquete")


class ComponenteDeTercerosIncompleto(ValueError):
    """La declaración del proveedor no está entera. Lleva los campos dentro."""

    def __init__(self, faltan: list[str]) -> None:
        self.faltan = list(faltan)
        super().__init__(
            "la declaración del componente de terceros no está completa; "
            f"falta o no vale: {', '.join(self.faltan)}"
        )


def _texto(valor: Any) -> bool:
    return isinstance(valor, str) and bool(valor.strip())


def _entero_positivo(valor: Any) -> bool:
    # `type(valor) is int` y no `isinstance`: `True` es un `int` para Python y
    # una dimensión de 1 publicada por un booleano sería un número inventado.
    return type(valor) is int and valor > 0


def _digest(bloque: Any, faltan: list[str], prefijo: str) -> None:
    if not isinstance(bloque, dict):
        faltan.append(prefijo)
        return
    for clave, valido in (("ruta", _texto), ("digest", _texto),
                          ("algoritmo", _texto), ("tamano_bytes", _entero_positivo)):
        if not valido(bloque.get(clave)):
            faltan.append(f"{prefijo}.{clave}")
    algoritmo = bloque.get("algoritmo")
    if _texto(algoritmo) and algoritmo not in ("sha256", "git-blob-sha1"):
        faltan.append(f"{prefijo}.algoritmo")


def validar(componente: Any) -> list[str]:
    """Qué le falta a esta declaración para ser un componente de terceros.

    Devuelve la lista de campos vacía cuando está entera. No redacta: los
    nombres son los del propio diccionario, para que `verify` los interpole en
    el idioma que le pidan y para que quien lea el informe sepa qué mirar.

    Lo que se exige sale del criterio de 107-C2 —id, versión, origen, licencia,
    digest de pesos, digest de tokenizer, dimensión, idiomas, longitud máxima,
    normalización, `opaco: true` con su frase y `datos_de_ajuste: ninguno`— y
    de los invariantes 4, 5 y 6 del contrato.
    """
    faltan: list[str] = []
    if not isinstance(componente, dict):
        return ["embedding_provider"]

    if componente.get("componente") != CLAVE_EN_EL_MANIFIESTO:
        faltan.append("componente")
    for clave in ("id", "familia", "repo", "ejecutado_por"):
        if not _texto(componente.get(clave)):
            faltan.append(clave)

    # LA VERSIÓN de un modelo de terceros es su CHECKPOINT (102, invariante 7;
    # 107, invariante 5): la licencia se fija por checkpoint concreto y `main`
    # no es un checkpoint. Por eso se exige el commit entero y no una etiqueta.
    revision = componente.get("revision")
    if not (_texto(revision) and len(revision) == 40
            and all(c in "0123456789abcdef" for c in revision)):
        faltan.append("revision")

    licencia = componente.get("licencia")
    if not isinstance(licencia, dict):
        faltan.append("licencia")
    else:
        for clave in ("spdx", "fuente", "leida_el", "tal_como_lo_declara_el_origen"):
            if not _texto(licencia.get(clave)):
                faltan.append(f"licencia.{clave}")
        # `compatible` no es informativo: una licencia incompatible descalifica
        # al proveedor. Declararla y seguir sería «media verdad».
        if licencia.get("compatible") is not True:
            faltan.append("licencia.compatible")

    # ACEPTAR NO ELIMINA RESTRICCIONES (107, invariante 5), pero no haber
    # aceptado nada es otra cosa: el paquete no puede decir de dónde salió el
    # permiso para traerse esos pesos.
    aceptacion = componente.get("aceptacion_de_licencia")
    if not isinstance(aceptacion, dict):
        faltan.append("aceptacion_de_licencia")
    else:
        for clave in ("spdx", "aceptada_el", "por"):
            if not _texto(aceptacion.get(clave)):
                faltan.append(f"aceptacion_de_licencia.{clave}")
        if (isinstance(licencia, dict) and _texto(licencia.get("spdx"))
                and aceptacion.get("spdx") != licencia.get("spdx")):
            # Aceptar OTRA licencia no es aceptar esta.
            faltan.append("aceptacion_de_licencia.spdx")

    # LAS TRES COSAS del invariante 6, reescrito el 2026-09-14 con lo medido:
    # con los dos digests el vector TODAVÍA no está prometido.
    _digest(componente.get("digest_de_los_pesos"), faltan, "digest_de_los_pesos")
    _digest(componente.get("digest_del_tokenizer"), faltan, "digest_del_tokenizer")
    longitud = componente.get("longitud_maxima")
    if not isinstance(longitud, dict):
        faltan.append("longitud_maxima")
    else:
        if not _entero_positivo(longitud.get("tokens")):
            faltan.append("longitud_maxima.tokens")
        if longitud.get("origen") not in ORIGENES_DE_LA_LONGITUD:
            faltan.append("longitud_maxima.origen")
        for clave in ("fuente", "por_que", "fijada_en"):
            if not _texto(longitud.get(clave)):
                faltan.append(f"longitud_maxima.{clave}")

    if not _entero_positivo(componente.get("dimension")):
        faltan.append("dimension")
    if not _texto(componente.get("pooling")):
        faltan.append("pooling")
    if not isinstance(componente.get("normaliza"), bool):
        faltan.append("normaliza")

    idiomas = componente.get("idiomas_medidos")
    if not isinstance(idiomas, dict) or not idiomas:
        faltan.append("idiomas_medidos")
    else:
        for idioma, fila in sorted(idiomas.items()):
            if not isinstance(fila, dict) or not _texto(fila.get("estado")):
                faltan.append(f"idiomas_medidos.{idioma}.estado")

    # OPACO, Y DICHO (107, invariante 4). `opaco: true` sin la frase es un
    # booleano que nadie lee; la frase sin el booleano no se puede filtrar.
    if componente.get("opaco") is not True:
        faltan.append("opaco")
    if not _texto(componente.get("opacidad")):
        faltan.append("opacidad")

    # «mientras no se ajuste nada» (107-C2). Cualquier otro valor describe un
    # modelo que ya no es el del catálogo, y este expediente no lo cubre.
    if componente.get("datos_de_ajuste") != "ninguno":
        faltan.append("datos_de_ajuste")

    # Los pesos NO van en la imagen (102-C5). Un `true` aquí contradice el
    # paquete entero, y un ausente deja sin decir dónde están.
    if componente.get("pesos_en_la_imagen") is not False:
        faltan.append("pesos_en_la_imagen")

    return faltan


def exigir_completo(componente: Any) -> dict[str, Any]:
    """La declaración, o un fallo con los campos que le faltan."""
    faltan = validar(componente)
    if faltan:
        raise ComponenteDeTercerosIncompleto(faltan)
    return componente


def las_tres_cosas(componente: dict[str, Any]) -> dict[str, Any]:
    """Lo que hace falta para que el vector esté prometido (invariante 6).

    Tres y no dos, y está medido: `model2vec.StaticModel.encode()` recorta a
    512 tokens por el valor por omisión de un parámetro de su librería que no
    aparece en ningún fichero del paquete, y sobre un texto de 8.008 tokens el
    coseno contra la propia librería es 0,658 con el corte y 1,000 sin él.
    """
    return {
        "digest_de_los_pesos": str((componente.get("digest_de_los_pesos") or {}).get("digest")),
        "digest_del_tokenizer": str((componente.get("digest_del_tokenizer") or {}).get("digest")),
        "longitud_maxima_tokens": (componente.get("longitud_maxima") or {}).get("tokens"),
    }


def bloque_para_el_manifiesto(componente: Any) -> dict[str, Any]:
    """La declaración tal como viaja en `reproduce.json`, comprobada entera."""
    exigir_completo(componente)
    # Copia: el manifiesto se sella con su digest y no puede compartir objetos
    # mutables con quien lo construyó.
    return copy.deepcopy(componente)


def bloque_para_la_spec(componente: Any) -> dict[str, Any]:
    """Lo que `inference_spec.json` necesita para reproducir el MISMO vector.

    Es un subconjunto, no un resumen: aquí va lo que `predict.py` comprueba al
    cargar (107-C4) y nada más. Lo que se audita —licencia, cobertura de
    idioma, opacidad— vive en el manifiesto y en el BOM.
    """
    exigir_completo(componente)
    longitud = componente["longitud_maxima"]
    return {
        "id": componente["id"],
        "repo": componente["repo"],
        "revision": componente["revision"],
        "fichero_del_modelo": componente["digest_de_los_pesos"]["ruta"],
        "digest_de_los_pesos": dict(componente["digest_de_los_pesos"]),
        "digest_del_tokenizer": dict(componente["digest_del_tokenizer"]),
        "longitud_maxima": {
            "tokens": longitud["tokens"],
            "origen": longitud["origen"],
            "fuente": longitud["fuente"],
        },
        "dimension": componente["dimension"],
        "pooling": componente["pooling"],
        "normaliza": componente["normaliza"],
        "sin_el_proveedor_no_se_predice": (
            "este modelo recibe columnas calculadas por un proveedor de embeddings "
            "de terceros; sin ese proveedor, con esa revisión y esos digests, no se "
            "predice — nunca se sustituye por otro"
        ),
    }


def la_spec_usa_un_proveedor(spec: Any) -> bool:
    """¿Este paquete lleva texto servido por un proveedor de terceros?

    Se mira por las DOS vías, y no es redundancia: el bloque lo escribe quien
    exporta, y los campos los escribe el modelo. Un paquete al que alguien
    borrara el bloque para que no se le exigiera la declaración sigue teniendo
    sus columnas de texto, y al revés.
    """
    if not isinstance(spec, dict):
        return False
    if isinstance(spec.get(CLAVE_EN_LA_SPEC), dict):
        return True
    for entrada in (spec.get("fields") or {}).values():
        if isinstance(entrada, dict) and entrada.get("encoding") == ENCODING_DE_TEXTO:
            return True
    for entrada in (spec.get("input") or {}).values():
        if isinstance(entrada, dict) and entrada.get("encoding") == ENCODING_DE_TEXTO:
            return True
    return False


def _propiedad(nombre: str, valor: Any) -> dict[str, str] | None:
    """Una propiedad de CycloneDX, o nada. Mismo criterio que `bom.py`: un
    ausente escrito como `"None"` es la peor clase de dato, porque parece uno."""
    if valor in (None, "", [], {}):
        return None
    if isinstance(valor, (dict, list)):
        valor = json.dumps(valor, sort_keys=True, ensure_ascii=False)
    if isinstance(valor, bool):
        valor = "true" if valor else "false"
    return {"name": nombre, "value": str(valor)}


def referencia_en_el_bom(componente: dict[str, Any]) -> str:
    """El `bom-ref` del proveedor. Sale del id y del checkpoint: dos revisiones
    del mismo modelo son dos componentes distintos y no pueden compartir ref."""
    return f"embedding-provider:{componente['id']}@{componente['revision'][:12]}"


def componente_para_el_bom(componente: Any) -> dict[str, Any]:
    """El proveedor como componente `machine-learning-model` de CycloneDX 1.6.

    Con `licenses` y `externalReferences`, que es literalmente lo que pide el
    criterio de 107-C2, y con la opacidad donde la busca quien lee un BOM:
    `modelCard.considerations.technicalLimitations`.
    """
    exigir_completo(componente)
    pesos = componente["digest_de_los_pesos"]
    tokenizer = componente["digest_del_tokenizer"]
    longitud = componente["longitud_maxima"]

    propiedades = [
        _propiedad("matrixai:embedding.provider_id", componente["id"]),
        _propiedad("matrixai:embedding.family", componente["familia"]),
        _propiedad("matrixai:embedding.executed_by", componente["ejecutado_por"]),
        _propiedad("matrixai:embedding.dimension", componente["dimension"]),
        _propiedad("matrixai:embedding.pooling", componente["pooling"]),
        _propiedad("matrixai:embedding.normalized", componente["normaliza"]),
        # EL DIGEST DEL TOKENIZER, AQUÍ Y NO EN `hashes`. El tokenizer es parte
        # del modelo (107, invariante 6) y su digest tiene que viajar; lo que no
        # puede es viajar disfrazado. Cuando el origen publica el `oid` de git
        # —el sha1 de `blob <n>\0` + contenido— llamarlo `SHA-1` en `hashes`
        # afirmaría que un `sha1sum` del fichero da ese número, y no lo da.
        _propiedad("matrixai:embedding.tokenizer.path", tokenizer["ruta"]),
        _propiedad("matrixai:embedding.tokenizer.digest", tokenizer["digest"]),
        _propiedad("matrixai:embedding.tokenizer.digest_algorithm", tokenizer["algoritmo"]),
        _propiedad("matrixai:embedding.weights.path", pesos["ruta"]),
        _propiedad("matrixai:embedding.weights.digest_algorithm", pesos["algoritmo"]),
        _propiedad("matrixai:embedding.weights.bytes", pesos["tamano_bytes"]),
        # Las TRES cosas del invariante 6 juntas: los dos digests de arriba y
        # el truncado, con su procedencia. Un límite heredado del valor por
        # omisión de una librería cambia el día que cambie la librería.
        _propiedad("matrixai:embedding.max_tokens", longitud["tokens"]),
        _propiedad("matrixai:embedding.max_tokens.origin", longitud["origen"]),
        _propiedad("matrixai:embedding.max_tokens.source", longitud["fuente"]),
        _propiedad("matrixai:embedding.max_tokens.fixed_in", longitud["fijada_en"]),
        _propiedad("matrixai:embedding.opaque", componente["opaco"]),
        _propiedad("matrixai:embedding.fine_tuning_data", componente["datos_de_ajuste"]),
        _propiedad("matrixai:embedding.weights_in_image", componente["pesos_en_la_imagen"]),
        _propiedad("matrixai:embedding.license.spdx", componente["licencia"]["spdx"]),
        _propiedad("matrixai:embedding.license.read_at", componente["licencia"]["leida_el"]),
        _propiedad("matrixai:embedding.license.accepted_at",
                   componente["aceptacion_de_licencia"]["aceptada_el"]),
        _propiedad("matrixai:embedding.license.accepted_by",
                   componente["aceptacion_de_licencia"]["por"]),
    ]
    for idioma, fila in sorted((componente["idiomas_medidos"] or {}).items()):
        propiedades.append(_propiedad(f"matrixai:embedding.language.{idioma}.status",
                                      fila.get("estado")))
        propiedades.append(_propiedad(f"matrixai:embedding.language.{idioma}.auc",
                                      fila.get("auc_coherencia_por_documento")))
    for i, cosa in enumerate(componente.get("lo_que_no_se_sabe") or []):
        propiedades.append(_propiedad(f"matrixai:embedding.unknown.{i}", cosa))

    limitaciones = [componente["opacidad"]]
    for idioma, fila in sorted((componente["idiomas_medidos"] or {}).items()):
        if fila.get("estado") != "cubierto":
            limitaciones.append(
                f"language {idioma}: {fila.get('estado')} — "
                f"{(fila.get('limitacion') or {}).get('frase') or fila.get('frase') or ''}".strip())
    limitaciones.append(
        f"text longer than {longitud['tokens']} tokens is truncated by this provider "
        f"({longitud['origen']}: {longitud['fuente']})")

    referencias = [
        {"type": "distribution",
         "url": f"https://huggingface.co/{componente['repo']}/tree/{componente['revision']}"},
        {"type": "license", "url": componente["licencia"]["fuente"]},
    ]
    ficha = ((componente.get("idiomas_segun_su_autor") or {}).get("fuente"))
    if _texto(ficha):
        referencias.append({"type": "model-card", "url": ficha})

    salida: dict[str, Any] = {
        "type": "machine-learning-model",
        "bom-ref": referencia_en_el_bom(componente),
        "name": componente["id"],
        # LA VERSIÓN ES EL CHECKPOINT, entero. Un `1.0` inventado aquí sería
        # exactamente lo que el invariante 5 prohíbe: la licencia y los pesos
        # se fijan por checkpoint concreto.
        "version": componente["revision"],
        "description": (
            f"third-party text embedding provider ({componente['familia']}), run by "
            f"{componente['ejecutado_por']}; weights are NOT audited"),
        "supplier": {"name": str(componente["repo"]).split("/")[0]},
        "licenses": [{"license": {"id": componente["licencia"]["spdx"]}}],
        "externalReferences": referencias,
        "hashes": [{"alg": "SHA-256", "content": pesos["digest"]}]
        if pesos["algoritmo"] == "sha256" else [],
        "properties": [p for p in propiedades if p is not None],
        "modelCard": {"considerations": {"technicalLimitations": limitaciones}},
    }
    if not salida["hashes"]:
        # Un `hashes: []` no dice nada y el esquema exige al menos uno cuando
        # está; sin sha256 de los pesos el digest viaja como propiedad, con su
        # algoritmo dicho, en vez de callarse.
        del salida["hashes"]
        salida["properties"].append(
            {"name": "matrixai:embedding.weights.digest", "value": pesos["digest"]})
    return salida
