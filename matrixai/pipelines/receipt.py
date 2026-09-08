"""El recibo `.mxreceipt` y su firma — contrato 81 (C1/C4).

Decisión de Roberto (2026-08-20): **DSSE sobre bytes JSON canonicalizados
con JCS**.

**Lo que un recibo firmado aquí demuestra hoy es CONSISTENCIA, no
autenticidad.** La otra mitad de la decisión —quién es de fiar: raíces de
confianza, identidad y rotación de claves, revocación, sellado de tiempo,
verificación offline— quedó abierta a propósito. Decirlo importa: es la
misma distinción que el contrato 82 tiene escrita en su §6.7, y sin ella
un recibo firmado se leería como una garantía que nadie ha respaldado.

Los niveles del §A del contrato:

* **A0** — hay recibo estructurado, **sin firmar**. Solo desarrollo.
* **A1** — firmado: integridad, procedencia y relación con artefactos.
* **A2** — además, atado a evidencias reproducibles.
* **A3/A4** — **reservados**: atestación del entorno y prueba
  criptográfica de la ejecución. No se pueden declarar todavía, y por eso
  no se alcanzan por mucho que un recibo los pida: un nivel que cualquiera
  puede escribirse a sí mismo no significa nada.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
from typing import Any

from matrixai.pipelines.canonical import jcs_bytes

__all__ = ["NIVELES", "PAYLOAD_TYPE_IN_TOTO", "PREDICATE_TYPE", "ReciboInvalido",
           "firmar_recibo", "nivel_del_recibo",
           "payload_del_sobre", "problemas_de_esquema", "recibo_del_sobre",
           "sobre_in_toto",
           "sujetos_del_recibo", "verificar_recibo"]

#: Los cinco niveles, en orden. A3 y A4 están reservados: existen para
#: poder decir que NO se alcanzan.
NIVELES = ("A0", "A1", "A2", "A3", "A4")

#: DSSE: el tipo del payload viaja dentro, con su versión, para que quien
#: verifique sepa qué está mirando sin adivinarlo.
PAYLOAD_TYPE = "application/vnd.matrixai.receipt+json;version=1.0"

_ESQUEMAS = ("1.0",)

#: EL SOBRE QUE SÍ LEE EL RESTO DEL MUNDO (86-C2).
#:
#: El nativo (`PAYLOAD_TYPE`) no se retira: lo emitimos también, y es el que
#: verifica `matrixai receipt`. Pero un `application/vnd.matrixai.receipt+json`
#: no lo entiende **ninguna** herramienta de las que ya tiene un comprador, así
#: que el mismo recibo viaja además como un **Statement de in-toto**, que es el
#: sobre que leen `cosign`, los verificadores de políticas y los ingestores de
#: atestaciones.
#:
#: Lo que NO cambia por esto: la firma sigue siendo HMAC y sigue demostrando
#: CONSISTENCIA, no autenticidad. Un sobre estándar con una firma simétrica es
#: legible por ahí fuera, no confiable por ahí fuera — y decir lo contrario
#: sería la media verdad de siempre.
PAYLOAD_TYPE_IN_TOTO = "application/vnd.in-toto+json"

#: El tipo de predicado, versionado por URL y en un dominio que es NUESTRO. Un
#: `predicateType` inventado en un dominio ajeno es una promesa que no podemos
#: sostener.
PREDICATE_TYPE = "https://matrixaistudio.org/attestations/receipt/v1"

_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"

#: EL SOBRE ANTERIOR AL 2026-08-24 NO SE LEE, y se dice por qué.
#:
#: Hasta el 86-C1 este módulo guardaba `"payload": payload.decode("utf-8")`
#: —el JSON canonicalizado EN CLARO—, y DSSE pide el payload **en base64**.
#: Decisión de Roberto (2026-08-24): **se rechazan**, sin código de
#: compatibilidad, porque fuera no hay ni un recibo emitido y este es el
#: momento más barato de la vida de un formato. Y se rechazan DICIÉNDOLO:
#: un «sobre inválido» genérico manda a buscar la avería donde no está, y
#: leerlos en silencio sería una compatibilidad callada — la invariante 3
#: del contrato 86 dice que nada se retira sin decirlo.
_FORMATO_ANTERIOR = (
    "el `payload` de este sobre va EN CLARO, y DSSE lo pide en base64: es "
    "el formato anterior al 2026-08-24 (contrato 86-C1). No se lee ni se "
    "convierte a propósito —una compatibilidad callada haría pasar por "
    "conforme algo que ninguna herramienta ajena de DSSE acepta—: hay que "
    "VOLVER A EMITIR el recibo con esta versión")

#: Y lo que no es base64 ni se parece al formato anterior tampoco se
#: interpreta a medias: se dice qué se miró y qué salió.
_NO_ES_BASE64 = (
    "el `payload` de este sobre no es base64 estándar, que es lo que DSSE "
    "pide ({error}). Si el recibo se emitió antes del 2026-08-24 es del "
    "formato anterior y hay que volver a emitirlo; si viene de otra "
    "herramienta, tiene que codificar el payload en base64")


def payload_del_sobre(sobre: Any) -> tuple[bytes | None, str | None]:
    """Los BYTES que el sobre lleva firmados, o el motivo de que no.

    Vive AQUÍ y no en cada lector porque son varios los que abren un
    sobre —`verificar_recibo`, `nivel_del_recibo`, el verificador del
    81-C4 y el CLI— y dos sitios declarando qué es un payload válido
    acabarían divergiendo: un día uno leería lo que el otro rechaza.

    La detección del formato anterior es segura por construcción: el
    payload en claro de un recibo empieza por `{`, y `{` no está en el
    alfabeto de base64, así que **nunca** puede decodificarse por error
    como base64 válido.
    """
    if not isinstance(sobre, dict):
        return None, "el sobre no es un objeto"
    payload = sobre.get("payload")
    if not isinstance(payload, str) or not payload:
        return None, "el sobre no lleva payload"
    try:
        return base64.b64decode(payload, validate=True), None
    except (binascii.Error, ValueError) as exc:
        if payload.lstrip().startswith("{"):
            return None, _FORMATO_ANTERIOR
        return None, _NO_ES_BASE64.format(error=exc)


#: Lo que hace que una evidencia sea REPRODUCIBLE, y no solo evidencia.
#: A2 pide (§14.5) estar atado a «un entorno, paquete o conjunto de
#: evidencias que PUEDE REPRODUCIRSE», y una raíz de traza no es eso: dice
#: qué pasó, no permite volver a hacerlo. La lista es CERRADA a propósito
#: —lo que no se reconoce no sube el nivel—: al revés, cualquier recibo
#: que llevara un `evidence` con algo dentro se declararía A2 por el mero
#: hecho de apuntar a su propia traza.
_EVIDENCIA_REPRODUCIBLE = ("replay_reference", "reproduce", "package",
                           "package_sha256")

#: Un sha256 en hexadecimal, con o sin el prefijo del algoritmo.
_DIGEST_RE = re.compile(r"^(sha256:)?[0-9a-f]{64}$")

#: LO QUE §14.2 PIDE EN TODO RECIBO.
#:
#: H2 del refutador (2026-08-20): el verificador decía `verified: schema`
#: sobre un recibo con **diez de las doce secciones ausentes**, incluidas
#: `pipeline_digest` y `executed_path_digest` que P23-R-0015 declara
#: **DEBE**. Miraba tres cosas —`payloadType`, `schema_version` y que
#: `steps` no estuviera vacío— y llamaba a eso «comprobar el esquema». Y
#: el emisor tampoco las exigía: era la avería que este contrato dice
#: haber aprendido —«validar en la escritura no valida la lectura»— con
#: el agravante de que **tampoco validaba la escritura**.
#:
#: La lista vive AQUÍ, junto al emisor, y el verificador la importa: si
#: cada uno tuviera la suya, un día el verificador aceptaría lo que el
#: emisor no produce, o al revés.
_SECCIONES_COMUNES = ("schema_version", "receipt_id", "event_type",
                      "created_at", "subject", "steps", "output", "evidence")

#: Y lo que pide cada tipo de acontecimiento. Un `replay` no tiene
#: `models` ni `input` —reproduce un paquete, no ejecuta un grafo—, y
#: exigírselos sería rechazar recibos honestos.
_SECCIONES_POR_TIPO = {
    "pipeline_execution": ("pipeline", "models", "input", "checks"),
    "replay": ("package", "environment"),
    # 87-C2 — la atestación de una EVALUACIÓN. No es una ejecución de pipeline
    # (no hay grafo) ni un replay (no se reproduce un paquete): se corrió un
    # modelo sobre unos datos y se midió. Lo que la hace valer es lo que exige:
    # QUÉ modelo (con su digest), SOBRE QUÉ datos (con el suyo) y QUÉ salió.
    # Sin las tres, un recibo de evaluación es un número con firma.
    "evaluation_attestation": ("models", "dataset", "metrics"),
    # 106-C2 — la ejecución de un ESTUDIO (matrixai_engines: candidatos,
    # selección, calibrador/umbral, evaluación final) — no es un grafo de
    # pipeline ni una atestación de una sola evaluación: hubo una
    # COMPETICIÓN entre candidatos con una decisión al final, y eso es
    # justo lo que "declarar A2 a mano no lo concede" pide poder
    # distinguir de "solo se puede recalcular la métrica".
    "study_execution": ("dataset", "protocolo", "candidatos", "seleccion",
                        "pipeline_final", "calibrador_umbral", "entorno"),
}

#: §14.2 bis / P23-R-0015: «El recibo **DEBE** comprometer el camino
#: ejecutado». Sin `executed_path_digest`, dos ejecuciones que recorren
#: ramas distintas del mismo grafo producen recibos idénticos.
_CAMINO_EJECUTADO = ("pipeline_digest", "executed_path_digest", "nodes")


def problemas_de_esquema(recibo: Any) -> list[str]:
    """Lo que le falta a un recibo para ser un recibo (§14.2), o nada.

    Se usa en los DOS extremos —`firmar_recibo` no firma lo que esto
    rechaza, y el verificador no dice `verified: schema` sin pasarlo—
    porque una comprobación que solo vive en un lado deja el otro abierto.
    """
    if not isinstance(recibo, dict):
        return ["un recibo es un objeto"]

    problemas: list[str] = []
    for seccion in _SECCIONES_COMUNES:
        if recibo.get(seccion) in (None, "", [], {}):
            problemas.append(f"falta {seccion!r}, que §14.2 pide en todo recibo")

    tipo = recibo.get("event_type")
    esperadas = _SECCIONES_POR_TIPO.get(tipo) if isinstance(tipo, str) else None
    if esperadas is None:
        # Fail-closed: un tipo desconocido no se interpreta a medias. Si
        # se aceptara, bastaría inventarse un `event_type` para que no se
        # le exigiera ninguna sección.
        problemas.append(
            f"event_type {tipo!r} desconocido; este código conoce "
            f"{sorted(_SECCIONES_POR_TIPO)}. Un tipo que no se reconoce no "
            "se verifica a medias")
    else:
        for seccion in esperadas:
            if recibo.get(seccion) in (None, "", [], {}):
                problemas.append(
                    f"falta {seccion!r}, que §14.2 pide en un recibo "
                    f"de tipo {tipo!r}")

    if tipo == "pipeline_execution":
        grafo = recibo.get("pipeline")
        if isinstance(grafo, dict):
            for campo in _CAMINO_EJECUTADO:
                if grafo.get(campo) in (None, "", [], {}):
                    problemas.append(
                        f"falta pipeline.{campo}, que P23-R-0015 declara DEBE "
                        "(§14.2 bis): sin el camino ejecutado, dos recorridos "
                        "distintos del mismo grafo dan recibos idénticos")
    return problemas


def _evidencia_reproducible(evidencia: Any) -> bool:
    """¿La evidencia se puede SEGUIR, o solo está presente?

    H2, segunda mitad: esto era `any(evidencia.get(k) for k in …)`, así
    que `{"package_sha256": True}` —el booleano— daba **A2**. La lista
    cerrada era de CLAVES, no de contenido, y el nivel lo seguía
    eligiendo quien escribía el recibo, que es justo lo que
    `nivel_del_recibo` existe para impedir.

    Se pide algo que alguien pueda ir a buscar: un digest bien formado
    donde la clave dice digest, una referencia no vacía en otro caso. Un
    booleano no es ninguna de las dos cosas.
    """
    if not isinstance(evidencia, dict):
        return False
    for clave in _EVIDENCIA_REPRODUCIBLE:
        valor = evidencia.get(clave)
        if clave.endswith("_sha256"):
            if isinstance(valor, str) and _DIGEST_RE.match(valor.strip().lower()):
                return True
            continue
        if isinstance(valor, str) and valor.strip():
            return True
        # Un objeto vale si lleva algo dentro que se pueda seguir: un
        # `{}` es tan poco reproducible como la ausencia.
        if isinstance(valor, dict) and any(
                isinstance(v, str) and v.strip() for v in valor.values()):
            return True
    return False


class ReciboInvalido(ValueError):
    """El recibo no se puede firmar tal como llega.

    Se corta ANTES de firmar: firmar un recibo mal formado produce algo
    que verifica correctamente y no atestigua nada, que es peor que no
    tener recibo.
    """


def _validar(recibo: Any) -> dict[str, Any]:
    if not isinstance(recibo, dict):
        raise ReciboInvalido("un recibo es un objeto")
    if recibo.get("schema_version") not in _ESQUEMAS:
        raise ReciboInvalido(
            f"schema_version {recibo.get('schema_version')!r} desconocida; "
            f"este código lee {list(_ESQUEMAS)}")
    if not str(recibo.get("receipt_id") or "").strip():
        raise ReciboInvalido("el recibo necesita un identificador")
    pasos = recibo.get("steps")
    if not isinstance(pasos, list) or not pasos:
        raise ReciboInvalido(
            "un recibo sin pasos no atestigua nada: no dice qué se ejecutó")
    # Y LAS SECCIONES DE §14.2, aquí también. Firmar un recibo al que le
    # faltan diez de las doce secciones produce algo que verifica
    # perfectamente y no atestigua nada — peor que no tener recibo,
    # porque lleva una firma buena delante.
    faltan = problemas_de_esquema(recibo)
    if faltan:
        raise ReciboInvalido(
            "el recibo no cumple §14.2 y firmarlo lo haría pasar por "
            "completo: " + "; ".join(faltan))
    return dict(recibo)


def _firma(payload: bytes, clave: bytes) -> str:
    """HMAC-SHA256 sobre el sobre PAE de DSSE.

    HMAC y no asimétrico **a propósito y por ahora**: la parte asimétrica
    exige decidir raíces de confianza y rotación, que es justo la mitad
    que quedó abierta. Con HMAC el recibo demuestra que no lo han tocado
    entre quien lo emitió y quien lo lee — consistencia—, y el día que se
    decida la confianza se cambia aquí sin tocar el resto.
    """
    return base64.b64encode(hmac.new(clave, payload, hashlib.sha256).digest()).decode()


def _pae(tipo: str, payload: bytes) -> bytes:
    """Pre-Authentication Encoding de DSSE.

    Ata el TIPO al contenido: sin él, el mismo payload firmado como
    «recibo» valdría como «política», y un atacante podría reutilizar una
    firma buena en otro sitio.
    """
    return b"DSSEv1 " + b" ".join([
        str(len(tipo)).encode(), tipo.encode(),
        str(len(payload)).encode(), payload,
    ])


def recibo_del_sobre(sobre: Any) -> tuple[dict[str, Any] | None, str | None]:
    """El RECIBO que hay dentro de un sobre, sea del tipo que sea (86-C2).

    Un sobre nativo lleva el recibo directamente; uno de in-toto lo lleva como
    `predicate` dentro de un Statement. Quien quiera leerlo —el nivel, el CLI,
    quien encadene esto en un guion— no tiene por qué saber en cuál está, y
    **dos sitios decidiendo dónde vive el recibo acabarían divergiendo**.
    """
    crudo, motivo = payload_del_sobre(sobre)
    if crudo is None:
        return None, motivo
    import json
    try:
        contenido = json.loads(crudo)
    except (TypeError, ValueError) as exc:
        return None, f"el payload del sobre no es JSON ({exc})"
    if not isinstance(contenido, dict):
        return None, "el payload del sobre no es un objeto"
    if contenido.get("_type") == _STATEMENT_TYPE:
        # Un Statement de OTRO predicado no se interpreta como si fuera
        # nuestro: decir que no se reconoce es la respuesta correcta.
        if contenido.get("predicateType") != PREDICATE_TYPE:
            return None, (
                f"el sobre lleva un Statement de in-toto con predicateType "
                f"{contenido.get('predicateType')!r}, que no es el de un recibo "
                f"de MatrixAI ({PREDICATE_TYPE})")
        predicado = contenido.get("predicate")
        if not isinstance(predicado, dict):
            return None, "el Statement no lleva un `predicate` que sea un objeto"
        return predicado, None
    return contenido, None


def sujetos_del_recibo(recibo: dict[str, Any]) -> list[dict[str, Any]]:
    """Los ARTEFACTOS sobre los que habla el recibo, en forma de in-toto.

    Un Statement dice «esto que afirmo va DE estos artefactos», y los artefactos
    de un recibo son los modelos que se ejecutaron y el pipeline que los
    encadenó. Sus digests ya viajan dentro; aquí solo se les da la forma que
    espera quien lo va a leer.

    **El prefijo `sha256:` se quita**: en in-toto el algoritmo es la CLAVE del
    mapa (`{"sha256": "abc…"}`), así que dejarlo dentro del valor produce un
    digest que ningún verificador reconoce. Es exactamente la clase de detalle
    por el que un formato «casi estándar» no lo es.
    """
    sujetos: list[dict[str, Any]] = []
    pipeline = recibo.get("pipeline") if isinstance(recibo.get("pipeline"), dict) else {}
    digest_pipeline = str(pipeline.get("pipeline_digest") or "")
    if digest_pipeline:
        nombre = f"pipeline:{pipeline.get('pipeline_id')}@{pipeline.get('pipeline_version')}"
        sujetos.append({"name": nombre,
                        "digest": {"sha256": digest_pipeline.removeprefix("sha256:")}})
    for modelo in recibo.get("models") or []:
        if not isinstance(modelo, dict):
            continue
        digest = str(modelo.get("digest") or "")
        if not digest:
            continue
        sujetos.append({
            "name": f"{modelo.get('model_id')}@{modelo.get('version')}",
            "digest": {"sha256": digest.removeprefix("sha256:")},
        })
    return sujetos


def sobre_in_toto(recibo: dict[str, Any], *, clave: bytes, key_id: str) -> dict[str, Any]:
    """El MISMO recibo, en un sobre que lee el resto del mundo (86-C2).

    No sustituye a `firmar_recibo`: se emiten los dos y se dice cuál es cuál.
    El payload va en base64 y el PAE ata el tipo al contenido, igual que en el
    nativo — lo único que cambia es qué hay dentro y cómo se llama.

    **Un recibo que no pasa §14.2 no se envuelve.** Sacarlo en un sobre
    estándar lo haría parecer más serio, no serlo.
    """
    limpio = _validar(recibo)
    if not str(key_id or "").strip():
        raise ReciboInvalido(
            "la firma tiene que decir con QUÉ clave se hizo: sin `keyid`, "
            "verificar exige adivinarlo")
    declaracion = {
        "_type": _STATEMENT_TYPE,
        "subject": sujetos_del_recibo(limpio),
        "predicateType": PREDICATE_TYPE,
        "predicate": limpio,
    }
    payload = jcs_bytes(declaracion)
    return {
        "payloadType": PAYLOAD_TYPE_IN_TOTO,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [{"keyid": key_id.strip(),
                        "sig": _firma(_pae(PAYLOAD_TYPE_IN_TOTO, payload), clave)}],
    }


def firmar_recibo(recibo: dict[str, Any], *, clave: bytes, key_id: str) -> dict[str, Any]:
    """Devuelve el sobre DSSE con el payload canonicalizado y su firma."""
    limpio = _validar(recibo)
    if not str(key_id or "").strip():
        raise ReciboInvalido(
            "la firma tiene que decir con QUÉ clave se hizo: sin `keyid`, "
            "verificar exige adivinarlo")
    payload = jcs_bytes(limpio)
    return {
        "payloadType": PAYLOAD_TYPE,
        # EN BASE64, como manda DSSE (86-C1). Lo que se firma —y lo que
        # `_pae` recibe— siguen siendo los BYTES CRUDOS: base64 es el
        # transporte del sobre, no lo firmado.
        #
        # Y son los bytes TAL CUAL se firmaron, solo que codificados:
        # reparsear y volver a serializar es donde se cuelan las
        # diferencias (P23-R-0016), y por eso el verificador decodifica
        # esto en vez de recanonicalizar el objeto.
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [{"keyid": key_id.strip(), "sig": _firma(_pae(PAYLOAD_TYPE, payload), clave)}],
    }


def verificar_recibo(sobre: Any, *, clave: bytes) -> dict[str, Any]:
    """¿La firma cuadra con los bytes que el sobre lleva?

    **Se verifica sobre el payload GUARDADO**, no sobre uno recompuesto a
    partir del objeto: reserializar aquí haría que dos serializadores
    distintos invalidaran una firma buena, o peor, validaran una mala.
    Desde el 86-C1 eso significa **decodificar el base64 del sobre y
    verificar sobre ESOS bytes** — decodificar no es recomponer: los
    bytes que salen son exactamente los que entraron.
    """
    crudo, motivo = payload_del_sobre(sobre)
    if crudo is None:
        return {"ok": False, "reason": motivo}
    firmas = sobre.get("signatures")
    if not isinstance(firmas, list) or not firmas:
        return {"ok": False, "reason": "el sobre no lleva firma"}

    tipo = str(sobre.get("payloadType") or "")
    esperada = _firma(_pae(tipo, crudo), clave)
    for firma in firmas:
        if isinstance(firma, dict) and hmac.compare_digest(str(firma.get("sig") or ""), esperada):
            return {"ok": True, "keyid": firma.get("keyid"),
                    "note": "demuestra CONSISTENCIA, no autenticidad: las raíces "
                            "de confianza siguen sin decidirse"}
    return {"ok": False, "reason": "la firma no corresponde a estos bytes"}


def nivel_del_recibo(recibo_o_sobre: Any) -> str:
    """Qué nivel A0-A4 sostiene de verdad — no el que se declare.

    Un nivel que cualquiera puede escribirse a sí mismo no significa
    nada: aquí se **deduce de lo que el recibo lleva**, y `assurance_level`
    del propio documento se ignora a propósito.
    """
    if not isinstance(recibo_o_sobre, dict):
        return "A0"
    # UN RECIBO FIRMADO CON SIGSTORE ESTÁ FIRMADO (2ª auditoría externa del
    # 2026-08-25, hallazgo 4 residual). El envoltorio de Sigstore no lleva
    # `signatures` —lleva su bundle—, así que caía por el `if` de abajo y salía
    # **A0 / UNSIGNED** justo después de firmar bien: el CLI recomendaba
    # «pass --key to sign it» sobre un recibo ya firmado. Sigstore sigue **sin
    # subir el nivel** (eso es del 86-C4, y hay prueba): aquí solo se deja de
    # decir que no está firmado cuando lo está.
    bundle = recibo_o_sobre.get("sigstore")
    if isinstance(bundle, dict) and bundle.get("bundle") and isinstance(
            recibo_o_sobre.get("payload"), dict):
        contenido = recibo_o_sobre["payload"]
        return "A2" if _evidencia_reproducible(contenido.get("evidence")) else "A1"

    firmado = bool(recibo_o_sobre.get("signatures")) and bool(recibo_o_sobre.get("payload"))
    if not firmado:
        return "A0"
    import json
    # UN SOBRE QUE NO SE PUEDE ABRIR NO SOSTIENE NADA: A0.
    #
    # Antes del 86-C1 esto leía el payload en claro, y un sobre ilegible
    # caía en el `except` de abajo y salía **A1** —«lleva firma»—. Con el
    # formato anterior rechazado eso sería justo la media verdad que la
    # invariante 3 prohíbe: un sobre que este código no lee no se
    # presenta como firmado, se presenta como lo que es.
    crudo, _motivo = payload_del_sobre(recibo_o_sobre)
    if crudo is None:
        return "A0"
    try:
        json.loads(crudo)
    except (TypeError, ValueError):
        # Base64 bien formado pero el contenido no es JSON: el sobre SÍ
        # es un sobre y lleva firma, aunque no se pueda leer qué firma.
        return "A1"
    # Y el recibo puede venir DENTRO de un Statement de in-toto (86-C2): sin
    # esto, un sobre estándar perdía su nivel —la evidencia vive en el
    # `predicate`— y un A2 se leía como A1.
    contenido, _ = recibo_del_sobre(recibo_o_sobre)
    evidencia = contenido.get("evidence") if isinstance(contenido, dict) else None
    if _evidencia_reproducible(evidencia):
        return "A2"
    return "A1"
