# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Firmar un recibo con Sigstore — contrato 86-C4, y con su límite delante.

QUÉ APORTA. La firma de hoy es **HMAC**: demuestra que el recibo no se ha tocado
entre quien lo emitió y quien lo lee —consistencia— pero **no quién lo emitió**,
porque las dos partes comparten la misma clave. Sigstore firma con una identidad
verificable y deja constancia pública, así que responde a la otra pregunta:
**de quién es esto**.

QUÉ NO APORTA, y va escrito aquí para que nadie lo deduzca al revés:

* **No sube el nivel del recibo.** Los niveles A0–A4 hablan de lo que se
  COMPROBÓ —evidencia reproducible, atestación del entorno—, no de la fuerza de
  la firma. Un recibo firmado con Sigstore y sin evidencia reproducible sigue
  siendo A1. Hay prueba de eso.
* **No decide raíces de confianza.** Quién es de fiar sigue sin decidirse
  (§ del 81); esto añade una firma verificable, no una política de confianza.

DÓNDE NO SE PUEDE, y por eso se comprueba antes de prometer nada: el modo sin
claves necesita **una identidad OIDC y red**. Una instalación descargable en la
máquina de alguien, sin navegador ni token, **no las tiene** — y ahí este camino
dice que no se puede y por qué, en vez de fallar a medias.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["SigstoreNoDisponible", "estado_de_sigstore", "firmar_con_sigstore"]

#: Las variables que llevan una identidad OIDC ya obtenida (CI, un flujo previo).
#: Si no hay ninguna, el modo sin claves tendría que abrir un navegador — y en
#: una instalación descargable no lo hay.
_VARIABLES_DE_IDENTIDAD = ("SIGSTORE_IDENTITY_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN")


class SigstoreNoDisponible(RuntimeError):
    """No se puede firmar con Sigstore aquí, y se dice exactamente por qué."""


def estado_de_sigstore() -> dict[str, Any]:
    """Qué hay y qué falta para poder firmar. Se puede consultar sin firmar.

    Devuelve los dos hechos por separado —la biblioteca y la identidad— porque
    **son dos problemas distintos con dos soluciones distintas**, y juntarlos en
    un «no se puede» mandaría a instalar algo a quien ya lo tiene.
    """
    try:
        import sigstore  # noqa: F401,PLC0415
        biblioteca = True
    except ImportError:
        biblioteca = False
    identidad = any(os.environ.get(v) for v in _VARIABLES_DE_IDENTIDAD)
    return {
        "library": biblioteca,
        "identity": identidad,
        "can_sign": biblioteca and identidad,
        "reason": None if (biblioteca and identidad) else _motivo(biblioteca, identidad),
    }


def _motivo(biblioteca: bool, identidad: bool) -> str:
    if not biblioteca and not identidad:
        return ("falta la biblioteca `sigstore` (`pip install sigstore`) y no hay "
                "identidad OIDC en el entorno: el modo sin claves necesita las dos")
    if not biblioteca:
        return "falta la biblioteca `sigstore` (`pip install sigstore`)"
    return ("no hay identidad OIDC en el entorno "
            f"({' o '.join(_VARIABLES_DE_IDENTIDAD)}): el modo sin claves la pide, y "
            "en una instalación sin navegador ni token no se puede obtener aquí")


def firmar_con_sigstore(payload: bytes, *, identidad: str | None = None) -> dict[str, Any]:
    """Firma los bytes y devuelve el bundle de Sigstore.

    **No se envuelve nada a medias**: si falta la biblioteca o la identidad, se
    levanta `SigstoreNoDisponible` con el motivo, en vez de devolver un bundle
    vacío que parecería una firma.
    """
    estado = estado_de_sigstore()
    if not estado["can_sign"] and identidad is None:
        raise SigstoreNoDisponible(estado["reason"])

    try:
        from sigstore.oidc import IdentityToken  # noqa: PLC0415
        from sigstore.sign import SigningContext  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover — cubierto por `estado_de_sigstore`
        raise SigstoreNoDisponible(_motivo(False, estado["identity"])) from exc

    crudo = identidad or next(
        (os.environ[v] for v in _VARIABLES_DE_IDENTIDAD if os.environ.get(v)), None)
    if not crudo:
        raise SigstoreNoDisponible(_motivo(True, False))

    # ESTE CAMINO NO SE HA EJERCITADO AQUÍ, y se dice en vez de suponerlo: el
    # flujo real necesita red y una identidad de verdad. Lo que sí está probado
    # es todo lo anterior —cuándo se puede y cuándo no— y que un fallo aquí
    # levanta `SigstoreNoDisponible` con su motivo, no un bundle a medias.
    try:
        contexto = SigningContext.production()
        with contexto.signer(IdentityToken(crudo)) as firmante:
            resultado = firmante.sign_artifact(payload)
        return {"bundle": resultado.to_json(), "mode": "sigstore-keyless"}
    except SigstoreNoDisponible:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SigstoreNoDisponible(
            f"Sigstore no pudo firmar ({type(exc).__name__}: {exc}). No se devuelve "
            "un bundle a medias: una firma que no se hizo no se presenta como hecha"
        ) from exc
