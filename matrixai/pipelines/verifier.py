"""Verificador de recibos, INDEPENDIENTE del runtime — contrato 81-C4.

Independiente a propósito: quien recibe un recibo no tiene por qué
ejecutar el motor que lo produjo, y un verificador que vive dentro del
runtime no puede comprobar nada sobre él.

Las tres cosas que el contrato le **prohíbe**, y que son las que más
importan porque alguien decidirá confiando en esto:

* **No presentar A1 como A2.** Un recibo firmado sin evidencias
  reproducibles es A1; decir A2 prometería una reproducción que nadie
  puede hacer.
* **No presentar trazabilidad como prueba de corrección conceptual.** Que
  un pipeline esté trazado no significa que su resultado sea correcto, y
  confundirlo haría que un recibo verde avalara una decisión equivocada.
* **Rechazar campos críticos ambiguos.** Si dos pasos comparten `id`, «el
  paso s1» no identifica a ninguno, y cualquier afirmación sobre él es
  ambigua.

Y una regla de forma: **se enseña lo verificado Y lo NO verificado**. Un
verificador que solo lista lo que comprobó deja creer que comprobó todo.
"""

from __future__ import annotations

import json
import re
from typing import Any

from matrixai.pipelines.receipt import (
    nivel_del_recibo,
    payload_del_sobre,
    verificar_recibo,
)

__all__ = ["inspeccionar_recibo", "verificar_sobre", "comparar_recibos"]

_SHA256 = re.compile(r"\Asha256:[0-9a-f]{64}\Z")

#: Lo que este verificador NUNCA afirma. Viajan en cada respuesta: un
#: aviso que hay que ir a buscar al manual no avisa.
_AVISOS = (
    "Un recibo verificado prueba que estos bytes no han cambiado y que "
    "quien tenía la clave los firmó. NO prueba que la decisión del "
    "pipeline sea correcta: la trazabilidad no es corrección conceptual.",
    "Mientras no se decidan las raíces de confianza, una firma válida "
    "demuestra CONSISTENCIA, no autenticidad: nadie ha respaldado que esa "
    "clave sea de quien dice ser.",
)


def _payload(sobre: Any) -> tuple[dict[str, Any] | None, str | None]:
    """El recibo que el sobre lleva dentro, o EL MOTIVO de que no.

    Devuelve el motivo y no solo `None` porque quien lea esto necesita
    saber si el sobre está roto, si es del formato anterior al 86-C1 o si
    lo que hay dentro no es un recibo: «el sobre no lleva un recibo
    legible», a secas, manda a mirar el fichero entero.

    **El base64 lo decodifica `payload_del_sobre`, del emisor** (86-C1):
    si el verificador tuviera su propia idea de qué es un payload válido,
    un día leería lo que el emisor no produce, o al revés.
    """
    crudo, motivo = payload_del_sobre(sobre)
    if crudo is None:
        return None, motivo
    try:
        datos = json.loads(crudo)
    except ValueError as exc:
        return None, (f"el payload del sobre no es JSON ({exc}): se "
                      "decodificó el base64 y lo de dentro no es un recibo")
    if not isinstance(datos, dict):
        return None, "el payload del sobre no es un objeto: un recibo lo es"
    return datos, None


def _esquema_invalido(sobre: dict[str, Any], recibo: dict[str, Any]) -> list[str]:
    """Los problemas de esquema del sobre y del recibo, o lista vacía.

    Se comprueba **lo mismo que exige el emisor** (`receipt.py`), y a
    propósito importándolo de allí: si el verificador tuviera su propia
    lista de versiones aceptables, un día aceptaría lo que el emisor no
    produce, o al revés — dos sitios declarando el mismo contrato.
    """
    from matrixai.pipelines.receipt import (
        PAYLOAD_TYPE,
        _ESQUEMAS,
        problemas_de_esquema,
    )

    problemas: list[str] = []
    tipo = sobre.get("payloadType")
    if tipo != PAYLOAD_TYPE:
        # DSSE ata el tipo al contenido justamente para que una firma
        # buena no valga en otro sitio. Aceptar cualquier tipo tiraría esa
        # garantía por la borda sin que nadie lo notara.
        problemas.append(
            f"payloadType {tipo!r} no es el de un recibo ({PAYLOAD_TYPE!r}): "
            "el tipo va firmado con el contenido y aceptar otro haría "
            "válida aquí una firma hecha para otra cosa")
    version = recibo.get("schema_version")
    if version not in _ESQUEMAS:
        problemas.append(
            f"schema_version {version!r} desconocida; este verificador lee "
            f"{list(_ESQUEMAS)}. Una versión que no se conoce no se "
            "interpreta a medias")
    if not isinstance(recibo.get("steps"), list) or not recibo.get("steps"):
        problemas.append(
            "un recibo sin pasos no atestigua nada: no dice qué se ejecutó")
    # Y LAS SECCIONES DE §14.2 — la MISMA lista que exige el emisor.
    #
    # H2 del refutador (2026-08-20): esto miraba tres cosas y llamaba a
    # eso «comprobar el esquema», así que un recibo sin `created_at`, sin
    # `subject`, sin `pipeline` —y por tanto sin `pipeline_digest` ni
    # `executed_path_digest`—, sin `models`, sin `input`, sin `checks` y
    # sin `output` salía con `verified: schema`. Un recibo firmado que
    # dice que su esquema está comprobado y no lo está es peor que uno
    # sin firmar.
    problemas.extend(problemas_de_esquema(recibo))
    return problemas


def _ambiguedades(recibo: dict[str, Any]) -> list[str]:
    """Campos críticos que no identifican lo que dicen identificar."""
    problemas: list[str] = []
    pasos = recibo.get("steps") or []
    vistos: set[str] = set()
    for paso in pasos:
        if not isinstance(paso, dict):
            problemas.append("hay un paso que no es un objeto")
            continue
        pid = str(paso.get("id") or "")
        if not pid:
            problemas.append("hay un paso sin id: no se puede hablar de él")
        elif pid in vistos:
            problemas.append(
                f"el id de paso {pid!r} está repetido: «el paso {pid}» no "
                "identifica a ninguno de los dos")
        vistos.add(pid)
        digest = paso.get("entry_hash")
        if digest is not None and not _SHA256.match(str(digest)):
            problemas.append(
                f"el paso {pid!r} declara un entry_hash que no es un sha256 "
                f"completo: {digest!r}")
    return problemas


def inspeccionar_recibo(sobre: Any) -> dict[str, Any]:
    """Enseña lo que el recibo DICE. No comprueba nada, y lo advierte.

    Sin ese aviso, alguien leería un `inspect` limpio como un `verify`
    bueno — y son cosas distintas.
    """
    recibo, motivo = _payload(sobre)
    if recibo is None:
        return {"ok": False, "reason": motivo,
                "note": "inspect NO verifica nada: solo enseña el contenido"}
    return {
        "ok": True,
        "receipt_id": recibo.get("receipt_id"),
        "pipeline": recibo.get("pipeline"),
        "steps": len(recibo.get("steps") or []),
        # `created_at`, que es como se llama en §14.2 y lo que el emisor
        # escribe. Esto leía `produced_at`, un nombre que NADIE escribe:
        # `inspect` enseñaba la fecha en blanco de todos los recibos del
        # producto. Salió al exigir las secciones de §14.2, porque los
        # fixtures que lo tapaban describían un recibo que no existe.
        "created_at": recibo.get("created_at"),
        "keyids": [f.get("keyid") for f in (sobre.get("signatures") or [])
                   if isinstance(f, dict)],
        "note": "inspect NO verifica nada: enseña lo que el recibo dice. "
                "Para comprobarlo, `receipt verify`.",
    }


def verificar_sobre(sobre: Any, *, clave: bytes | None) -> dict[str, Any]:
    """Verifica lo que se pueda y **declara lo que no**."""
    verificado: list[str] = []
    sin_verificar: dict[str, str] = {}
    problemas: list[str] = []

    recibo, motivo = _payload(sobre)
    if recibo is None:
        # Y CON SU MOTIVO. Un sobre del formato anterior al 86-C1 se
        # rechaza diciendo que lo es y que hay que volver a emitirlo: si
        # saliera «el sobre no lleva un recibo legible» a secas, quien lo
        # reciba se pondría a buscar una corrupción que no existe.
        return {"ok": False, "verified": [], "assurance_level": "A0",
                "unverified": {"schema": motivo},
                "problems": [motivo],
                "disclaimers": list(_AVISOS),
                # Los mismos campos que la salida normal: quien encadene
                # esto en un guion no puede encontrarse un informe con la
                # mitad de las claves según por dónde salga.
                "assurance_is_claimed_not_checked": False,
                "signature_checked": False,
                "signature_valid": False,
                "fully_checked": False,
                "unchecked": ["schema", "signature"],
                "standing_limitation": "key_identity"}

    # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: aquí se añadía "schema"
    # a lo VERIFICADO por el mero hecho de que el JSON se pudiera
    # decodificar. Medido: un sobre con `payloadType: "application/evil"` y
    # `schema_version: "999"` salía `ok: true`, nivel A1, `verified:
    # ["schema"]`. El verificador **afirmaba haber comprobado lo que no
    # había mirado**, que es justo lo único que este componente existe para
    # no hacer.
    #
    # Y es el MISMO defecto que el de las rutas del 82: `firmar_recibo` sí
    # rechaza una `schema_version` desconocida, así que un emisor honesto
    # no puede producirla — pero el verificador lee cosas que vienen de
    # FUERA. *Validar en la escritura no valida la lectura.*
    problemas.extend(_esquema_invalido(sobre, recibo))
    if not problemas:
        verificado.append("schema")
    else:
        sin_verificar["schema"] = "el recibo no cumple el esquema que declara"

    problemas.extend(_ambiguedades(recibo))

    # LAS TRES COSAS QUE LE PUEDEN PASAR A UNA FIRMA, separadas.
    #
    # H6 del refutador (2026-08-20): «no había clave» y «la firma es
    # mala» iban al mismo saco, así que el CLI imprimía «la firma no se
    # ha comprobado» sobre una firma que SÍ se comprobó y se rechazó. Es
    # el mismo colapso que este contrato dice haber corregido en el
    # `rc=3 → mismatch` del recibo de reproducción, un módulo más allá.
    firma_comprobada = clave is not None
    if clave is None:
        # NO se finge haber comprobado la firma: sin clave no se puede, y
        # decirlo es la diferencia entre un informe y una impresión.
        sin_verificar["signature"] = (
            "no se aportó clave, así que la firma no se ha comprobado")
    else:
        resultado = verificar_recibo(sobre, clave=clave)
        if resultado.get("ok"):
            verificado.append("signature")
        else:
            problemas.append(str(resultado.get("reason") or "firma inválida"))

    # La identidad de la clave NO se puede comprobar todavía: las raíces
    # de confianza están sin decidir (§0.1, mitad abierta a propósito).
    sin_verificar["key_identity"] = (
        "no hay raíces de confianza decididas: no se puede comprobar que la "
        "clave sea de quien dice ser")

    nivel = nivel_del_recibo(sobre) if not problemas else "A0"
    return {
        "ok": not problemas,
        "verified": verificado,
        "unverified": sin_verificar,
        "problems": problemas,
        # DEDUCIDO, nunca el que el recibo se atribuya.
        "assurance_level": nivel,
        # Y el nivel dice lo que el recibo SOSTIENE, no lo que se ha
        # comprobado en esta pasada: sin clave, un A1 significa «lleva
        # firma», no «la firma es buena». Sin esta línea, un `assurance:
        # A1` con la firma sin verificar se lee como un aprobado.
        "assurance_is_claimed_not_checked": not firma_comprobada,
        # Y QUÉ le pasó a la firma, sin colapsar: «no se pidió» no es
        # «falló», y quien encadene esto en un script necesita
        # distinguirlas para no tratar un recibo sin comprobar como uno
        # comprobado.
        "signature_checked": firma_comprobada,
        "signature_valid": "signature" in verificado,
        # LO QUE NO SE PUDO COMPROBAR Y SÍ SE PODÍA, a la vista. Un
        # informe con esto no vacío no es un aprobado: es un «no lo sé».
        #
        # `key_identity` NO cuenta aquí, y no por comodidad: las raíces de
        # confianza están sin decidir a propósito (§0.1, mitad abierta), o
        # sea que no es «esta vez no se pudo» sino una limitación
        # permanente y declarada. Metiéndola, `fully_checked` sería
        # siempre falso y el código de salida `0` no se alcanzaría nunca —
        # un código que no puede ocurrir no distingue nada.
        "fully_checked": "signature" in verificado and not problemas,
        "unchecked": sorted(k for k in sin_verificar if k != "key_identity"),
        # Y la limitación permanente, dicha aparte para que no se
        # confunda con un fallo de esta pasada.
        "standing_limitation": "key_identity",
        "disclaimers": list(_AVISOS),
    }


def _payload_o_recibo(objeto: Any) -> tuple[dict[str, Any] | None, str | None]:
    """El recibo, venga dentro de un sobre o pelado.

    Un recibo sin firmar **no tiene sobre**, y es exactamente el caso que ve
    quien no ha configurado clave. Exigir sobre aquí dejaba fuera de la
    comparación justo a quien más la necesita.
    """
    if isinstance(objeto, dict) and "payload" not in objeto and "schema_version" in objeto:
        return dict(objeto), None
    return _payload(objeto)


def comparar_recibos(uno: Any, otro: Any) -> dict[str, Any]:
    """En qué se diferencian dos recibos.

    Compara el CONTENIDO, no los bytes del sobre: dos recibos iguales
    firmados con claves distintas siguen diciendo lo mismo, y decir que
    difieren mandaría a buscar una diferencia que no existe.
    """
    # ACEPTA UN SOBRE **O** UN RECIBO PELADO (2026-08-25). La pantalla del
    # 85-C3 tiene el recibo, no siempre su sobre —un recibo sin firmar no lo
    # tiene—, y sin esto comparar desde la interfaz era imposible. Se resuelve
    # AQUÍ, en el único sitio que decide qué es comparar dos recibos: hacerlo
    # en el backend habría sido un segundo sitio con su propia idea.
    (a, motivo_a), (b, motivo_b) = _payload_o_recibo(uno), _payload_o_recibo(otro)
    if a is None or b is None:
        return {"identical": False,
                "differences": [m for m in (motivo_a, motivo_b) if m]}
    diferencias: list[dict[str, Any]] = []
    for clave in sorted(set(a) | set(b)):
        if a.get(clave) != b.get(clave):
            diferencias.append({"field": clave, "a": a.get(clave), "b": b.get(clave)})
    return {"identical": not diferencias, "differences": diferencias}
