"""CONTRATO 83-C4 — supervisar una propuesta de refinamiento.

El `RefinementAgent` puede proponer; **decidir es un acto humano** y
queda registrado con quién y cuándo (§5.2 del contrato 83, misma línea
que el 81 defiende en su §26.10).

Aquí viven las dos piezas que no son de pantalla:

* el **diff** de la propuesta, calculado UNA vez y en el core —§5.1: no
  se recalcula lo que el core ya dice, y dos sitios calculando lo mismo
  acaban dando dos respuestas—;
* el **registro de la decisión**, que exige motivo para rechazar porque
  el contrato pide que *«el motivo se guarde»*: aceptar un rechazo mudo
  dejaría constancia de que alguien dijo que no y de nada más.
"""

from __future__ import annotations

import difflib
from typing import Any

__all__ = ["DecisionInvalida", "diff_de_la_propuesta", "registrar_decision",
           "serie_de_deriva"]


class DecisionInvalida(ValueError):
    """La decisión no se puede registrar tal como llega.

    Es un error y no un `False` silencioso a propósito: una decisión que
    no se registra bien es una decisión que después nadie puede explicar,
    y el corte existe justamente para dejar constancia.
    """


def diff_de_la_propuesta(original: str, propuesto: str) -> list[dict[str, Any]]:
    """Las líneas del cambio, marcadas, para que la pantalla las pinte.

    Devuelve **todas** las líneas —incluidas las que no cambian— porque un
    diff que solo enseña lo modificado obliga a abrir el original al lado
    para entenderlo.

    Y cuando la propuesta es IGUAL al original devuelve las líneas con
    marca `" "`, no una lista vacía: una lista vacía se lee como «no hay
    propuesta», que es otra cosa muy distinta.
    """
    izquierda = (original or "").splitlines(keepends=True)
    derecha = (propuesto or "").splitlines(keepends=True)
    salida: list[dict[str, Any]] = []
    for linea in difflib.ndiff(izquierda, derecha):
        marca, texto = linea[:1], linea[2:]
        if marca == "?":
            # `ndiff` intercala líneas de «pistas» que no son contenido:
            # colarlas haría que el texto reconstruido no fuera el real.
            continue
        salida.append({"mark": marca, "text": texto})
    return salida


def registrar_decision(
    refinement_id: str,
    *,
    aceptada: bool,
    motivo: str | None,
    quien: str,
    cuando: str,
) -> dict[str, Any]:
    """Deja constancia de la decisión sobre una propuesta.

    **Rechazar exige motivo.** Aceptar es seguir el camino que el sistema
    propuso; rechazar es apartarse de él, y eso es lo que alguien tendrá
    que poder explicar dentro de seis meses.

    **Y siempre hay un quién.** Un registro anónimo no cumple «queda
    registrado con quién y cuándo», así que no se acepta a medias.
    """
    if not (refinement_id or "").strip():
        raise DecisionInvalida("una decisión sin la propuesta que decide no es una decisión")
    if not (quien or "").strip():
        raise DecisionInvalida(
            "la decisión tiene que decir QUIÉN la tomó: promover o descartar "
            "es un acto humano y queda registrado")
    if not (cuando or "").strip():
        raise DecisionInvalida("la decisión tiene que decir CUÁNDO se tomó")
    limpio = (motivo or "").strip()
    if not aceptada and not limpio:
        raise DecisionInvalida(
            "rechazar una propuesta exige un motivo, y se guarda: sin él "
            "solo consta que alguien dijo que no, que es casi no dejar constancia")
    return {
        "refinement_id": refinement_id.strip(),
        "accepted": bool(aceptada),
        "reason": limpio or None,
        "who": quien.strip(),
        "when": cuando.strip(),
    }


def serie_de_deriva(informes: list[dict[str, Any]]) -> dict[str, Any]:
    """La deriva que P22 calcula, ordenada en el tiempo para pintarla.

    **No recalcula nada** (§5.1): `drift_detected` es el veredicto del
    core y se copia tal cual. Si aquí se dedujera del valor y el umbral,
    habría dos sitios decidiendo lo mismo y acabarían discrepando —
    exactamente lo que la invariante prohíbe.

    **Y lo que no se midió no se dibuja como si se hubiera medido.** Un
    dibujo afirma por omisión: una serie con un hueco pintado a cero
    diría que ese día no hubo deriva, cuando lo que hubo fue que no se
    miró. Las mediciones saltadas viajan con `value: None`, su
    `skipped: true` y su motivo, para que la pantalla pueda enseñar el
    hueco COMO hueco.
    """
    puntos: list[dict[str, Any]] = []
    features: list[str] = []

    for informe in informes:
        if not isinstance(informe, dict):
            continue
        por_feature: dict[str, Any] = {}
        # `results` es un DICCIONARIO feature → resultado, que es lo que
        # produce el core (`DriftReport.results: dict[str, FeatureDriftResult]`,
        # medido el 2026-08-20 con un detector real). Esto leía una LISTA, así
        # que la serie salía vacía por muchas mediciones que hubiera — y un
        # histórico vacío se lee como «no hay deriva», que es justo lo que el
        # resto de esta función existe para no decir.
        resultados = informe.get("results")
        if not isinstance(resultados, dict):
            continue
        for nombre, r in resultados.items():
            if not isinstance(r, dict) or not nombre:
                continue
            nombre = str(nombre)
            if nombre not in features:
                features.append(nombre)
            saltada = bool(r.get("skipped")) or not r.get("enough_samples", True)
            por_feature[nombre] = {
                # `None` cuando no se midió, y NUNCA 0.0: un cero es un
                # dato («no hubo deriva») y esto es la ausencia de dato.
                "value": None if saltada else r.get("observed_value"),
                "threshold": r.get("threshold"),
                "method": r.get("method"),
                # El veredicto es del CORE, copiado, no deducido.
                "drift_detected": bool(r.get("drift_detected")),
                "skipped": saltada,
                "skip_reason": r.get("skip_reason") or (
                    "not enough samples" if not r.get("enough_samples", True) else None),
                "samples_used": r.get("samples_used"),
            }
        puntos.append({
            "checked_at": informe.get("checked_at"),
            "drift_detected": bool(informe.get("drift_detected")),
            "enough_samples": bool(informe.get("enough_samples", True)),
            "total_production_samples": informe.get("total_production_samples"),
            "features": por_feature,
        })

    # En orden por fecha: una serie desordenada se pinta como una gráfica
    # que va y viene en el tiempo, y se lee como ruido del modelo.
    puntos.sort(key=lambda p: str(p.get("checked_at") or ""))
    return {
        "points": puntos,
        "features": features,
        # Distinguir «nunca se ha medido» de «se midió y no hubo deriva»:
        # una gráfica en blanco se lee como lo segundo.
        "never_measured": not puntos,
    }
