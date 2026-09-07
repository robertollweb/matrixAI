# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C5 — el esquema versionado completo: «problema confirmado, errores,
sospechas, evidencia insuficiente, medidas/umbrales, contexto, decisiones,
políticas y SplitPlan» (texto literal). COMPONE lo que 103-C1/C2/C3/C4 ya
producen — no repite ningún campo:

  - problema confirmado -> `ProblemSpec` (103-C1, `objetivo.py`)
  - errores             -> `Diagnostico.bloqueos` (103-C2)
  - sospechas           -> `Diagnostico.sospechas` (103-C2)
  - evidencia insuficiente -> `Diagnostico.limites` (103-C2, invariante 4:
    «limita la conclusión o deja métricas indefinidas» — el propio
    `Limite` YA es eso, no un cuarto tipo de hallazgo)
  - medidas/umbrales    -> ya dentro de `Sospecha.medida`/`Limite.medida`
  - políticas           -> `PoliticaDePreparacion` (103-C3)
  - SplitPlan           -> `PropuestaDeParticion.plan` (103-C4, 104-C0)
  - contexto/decisiones -> `AceptacionDeSospecha` (NUEVO en este corte,
    ver abajo) — lo único que 103-C2 dejó explícitamente pendiente:
    «la aceptación misma es del 103-C5, aquí solo se detecta y se mide»
    (docstring literal de `Sospecha`).

ACEPTAR UNA SOSPECHA, CON RECIBO AUDITABLE — mismo patrón que
`LicenseAcceptance` (`data_provider.py`), no un booleano efímero: el
digest fija los términos EXACTOS de la sospecha aceptada (clave, campo,
motivo, medida). Si el diagnóstico se repite y la MISMA sospecha sale con
una medida distinta (el dato cambió, o se corrigió algo), el digest deja
de coincidir y la aceptación anterior deja de cubrirla — «una aceptación
de sospecha permanece visible... pero eso no es lo mismo que decir que
sigue vigente para un diagnóstico que ya no es el que se aceptó».

TRES ESTADOS DE DISEÑO, NUNCA COLAPSADOS (mismo principio que las tres
formas de hallazgo). `bloqueado`: hay al menos un error estructural —
"un error estructural conserva bloqueo" (texto literal), ninguna
aceptación lo levanta. `exploratorio`: sin bloqueos, pero queda alguna
sospecha sin aceptar — "un estudio exploratorio no se presenta como
validado" (texto literal). `confirmado`: sin bloqueos y todas las
sospechas (si las hay) aceptadas con recibo vigente.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from matrixai.estudio.esquemas import SplitPlan
from matrixai.training.diagnostico import Diagnostico, Sospecha
from matrixai.training.diagnostico_de_riesgo_textos import motivo
from matrixai.training.objetivo import Confirmacion
from matrixai.training.preparacion import PoliticaDePreparacion

__all__ = [
    "ESTADOS_DE_DISENO", "VERSION_ESQUEMA", "AceptacionDeSospecha",
    "DiagnosticoDeRiesgo", "ErrorDeAceptacion", "RegistroDeAceptaciones",
    "aceptar_sospecha", "sospechas_pendientes",
]

VERSION_ESQUEMA = "1.0"

#: Nunca colapsados -- ver el docstring del módulo.
ESTADOS_DE_DISENO = ("bloqueado", "exploratorio", "confirmado")


class ErrorDeAceptacion(Exception):
    def __init__(self, clave: str, **campos: object) -> None:
        textos = motivo(clave, **campos)
        super().__init__(textos["en"])
        self.clave = clave
        self.es = textos["es"]
        self.en = textos["en"]


def _digest_de_sospecha(sospecha: Sospecha) -> str:
    """Fija los TÉRMINOS EXACTOS de lo aceptado — mismo mecanismo que
    `_license_digest` en `data_provider.py`: si la sospecha cambia
    (medida distinta, motivo distinto), el digest ya no coincide."""
    crudo = json.dumps(sospecha.a_json(), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(crudo).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AceptacionDeSospecha:
    """Recibo AUDITABLE de aceptar una sospecha -- nunca un booleano
    efímero. `sospecha_digest` es lo que decide si esta aceptación sigue
    cubriendo la sospecha actual (ver `sospechas_pendientes`)."""

    aceptacion_id: str
    sospecha_clave: str
    sospecha_campo: str | None
    sospecha_digest: str
    contexto: str
    actor: str
    accepted_at: str

    def a_json(self) -> dict[str, Any]:
        return {
            "aceptacion_id": self.aceptacion_id, "sospecha_clave": self.sospecha_clave,
            "sospecha_campo": self.sospecha_campo, "sospecha_digest": self.sospecha_digest,
            "contexto": self.contexto, "actor": self.actor, "accepted_at": self.accepted_at,
        }


def aceptar_sospecha(sospecha: Sospecha, *, contexto: str, actor: str) -> AceptacionDeSospecha:
    """Registra la aceptación de UNA sospecha -- exige contexto y autor,
    nunca una aceptación anónima o sin motivo (invariante 3 del 103:
    «exige contexto»)."""
    if not actor or not actor.strip():
        raise ErrorDeAceptacion("aceptacion_sin_actor")
    if not contexto or not contexto.strip():
        raise ErrorDeAceptacion("aceptacion_sin_contexto", campo=sospecha.campo or sospecha.clave)
    return AceptacionDeSospecha(
        aceptacion_id=str(uuid.uuid4()), sospecha_clave=sospecha.clave,
        sospecha_campo=sospecha.campo, sospecha_digest=_digest_de_sospecha(sospecha),
        contexto=contexto.strip(), actor=actor.strip(), accepted_at=_utcnow_iso())


def sospechas_pendientes(diagnostico: Diagnostico,
                         aceptaciones: tuple[AceptacionDeSospecha, ...]) -> tuple[Sospecha, ...]:
    """Las sospechas del diagnóstico SIN una aceptación vigente -- por
    digest exacto, no por clave: dos sospechas con la misma clave pero
    medida distinta son términos distintos, cada una necesita su propio
    recibo."""
    digests_aceptados = {a.sospecha_digest for a in aceptaciones}
    return tuple(s for s in diagnostico.sospechas if _digest_de_sospecha(s) not in digests_aceptados)


class RegistroDeAceptaciones:
    """v1: en memoria, vive mientras el proceso esté arriba -- mismo
    alcance declarado que `LicenseAcceptanceStore` (persistir entre
    reinicios es extensión de infraestructura, no de este corte)."""

    def __init__(self) -> None:
        self._aceptaciones: list[AceptacionDeSospecha] = []

    def aceptar(self, sospecha: Sospecha, *, contexto: str, actor: str) -> AceptacionDeSospecha:
        aceptacion = aceptar_sospecha(sospecha, contexto=contexto, actor=actor)
        self._aceptaciones.append(aceptacion)
        return aceptacion

    def todas(self) -> tuple[AceptacionDeSospecha, ...]:
        return tuple(self._aceptaciones)


@dataclass(frozen=True)
class DiagnosticoDeRiesgo:
    """El esquema versionado completo -- composición, nunca duplicación
    (ver el docstring del módulo para el mapeo campo a campo).

    `confirmacion` (103-C1), NO `ProblemSpec` a secas: el objetivo puede
    estar bloqueado (caso 1 del 71 -- `precio` a partir de `precio`) y
    entonces no hay `ProblemSpec` que envolver todavía. `diagnostico`
    (103-C2) es `None` en ese caso -- el diagnóstico de DATOS nunca
    arranca sobre un objetivo sin confirmar, no tendría target sobre el
    que medir nada."""

    confirmacion: Confirmacion
    diagnostico: Diagnostico | None = None
    preparacion: PoliticaDePreparacion | None = None
    particion: SplitPlan | None = None
    aceptaciones: tuple[AceptacionDeSospecha, ...] = ()
    version: str = VERSION_ESQUEMA

    @property
    def estado(self) -> str:
        if not self.confirmacion.confirmado:
            return "bloqueado"
        if self.diagnostico is not None and self.diagnostico.impide_recomendacion:
            return "bloqueado"
        if self.diagnostico is not None and sospechas_pendientes(self.diagnostico, self.aceptaciones):
            return "exploratorio"
        return "confirmado"

    @property
    def puede_continuar(self) -> bool:
        """Solo `bloqueado` impide seguir -- `exploratorio` SÍ deja
        continuar (texto literal: «puede continuar solo en el estado
        permitido», y el estado permitido para una sospecha sin aceptar
        es justo exploratorio, no bloqueado)."""
        return self.estado != "bloqueado"

    def a_json(self) -> dict[str, Any]:
        return {
            "version": self.version, "estado": self.estado,
            "confirmacion": self.confirmacion.a_json(),
            "diagnostico": self.diagnostico.a_json() if self.diagnostico is not None else None,
            "preparacion": self.preparacion.a_json() if self.preparacion is not None else None,
            "particion": self.particion.a_json() if self.particion is not None else None,
            "aceptaciones": [a.a_json() for a in self.aceptaciones],
        }
