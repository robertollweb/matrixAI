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
import hashlib
import hmac
import re
from typing import Any

from matrixai.pipelines.canonical import jcs_bytes

__all__ = ["NIVELES", "ReciboInvalido", "firmar_recibo", "nivel_del_recibo",
           "problemas_de_esquema",
           "verificar_recibo"]

#: Los cinco niveles, en orden. A3 y A4 están reservados: existen para
#: poder decir que NO se alcanzan.
NIVELES = ("A0", "A1", "A2", "A3", "A4")

#: DSSE: el tipo del payload viaja dentro, con su versión, para que quien
#: verifique sepa qué está mirando sin adivinarlo.
PAYLOAD_TYPE = "application/vnd.matrixai.receipt+json;version=1.0"

_ESQUEMAS = ("1.0",)


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
        # El payload se guarda TAL CUAL se firmó. Es lo que se verifica
        # después: reparsear y volver a serializar es donde se cuelan las
        # diferencias (P23-R-0016).
        "payload": payload.decode("utf-8"),
        "signatures": [{"keyid": key_id.strip(), "sig": _firma(_pae(PAYLOAD_TYPE, payload), clave)}],
    }


def verificar_recibo(sobre: Any, *, clave: bytes) -> dict[str, Any]:
    """¿La firma cuadra con los bytes que el sobre lleva?

    **Se verifica sobre el payload GUARDADO**, no sobre uno recompuesto a
    partir del objeto: reserializar aquí haría que dos serializadores
    distintos invalidaran una firma buena, o peor, validaran una mala.
    """
    if not isinstance(sobre, dict):
        return {"ok": False, "reason": "el sobre no es un objeto"}
    payload = sobre.get("payload")
    firmas = sobre.get("signatures")
    if not isinstance(payload, str) or not isinstance(firmas, list) or not firmas:
        return {"ok": False, "reason": "el sobre no lleva payload y firma"}

    tipo = str(sobre.get("payloadType") or "")
    crudo = payload.encode("utf-8")
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
    firmado = bool(recibo_o_sobre.get("signatures")) and bool(recibo_o_sobre.get("payload"))
    if not firmado:
        return "A0"
    import json
    try:
        contenido = json.loads(recibo_o_sobre["payload"])
    except (TypeError, ValueError):
        return "A1"
    evidencia = contenido.get("evidence") if isinstance(contenido, dict) else None
    if _evidencia_reproducible(evidencia):
        return "A2"
    return "A1"
