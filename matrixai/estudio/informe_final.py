# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C6 — evaluación final y reportes: `evaluacion.json`, el documento que
junta lo que ya midieron los cortes anteriores — nunca vuelve a calcular
nada, solo lo referencia y lo empaqueta con el mismo sobre versionado
(`schema`/`schema_version`) que usa cada esquema de 104-C0.

COMPOSICIÓN, NO DUPLICACIÓN. `InformeDeEvaluacion` no repite `pipeline_
digest`/`split_plan_digest`/`evaluated_role`/`evidence` — los tiene ya el
`EvaluationResult` (104-C0/104-C5) que envuelve, y copiarlos aquí sería la
misma decisión declarada dos veces, divergiendo con el tiempo. «Distinguir
selección interna, prueba final y cohorte externa» (texto literal del
corte) ya lo hace `evaluated_role` (`development`/`selection`/`calibration`
para lo interno, `test` para la prueba final, `external_test` para la
cohorte externa) — no hace falta un vocabulario nuevo que diga lo mismo con
otras palabras.

QUÉ SE INCORPORA, Y CON QUÉ TIPO. `intervalos` son objetos `Intervalo`
(105-C2) de verdad — tiene `desde_json()` propio, se reconstruye. `curva_
de_fiabilidad`, `recalibracion` (105-C3) y cada entrada de `comparaciones`
(105-C5) se guardan como el `dict` que ya produce su propio `a_json()`: esos
tres tipos nunca tuvieron un `desde_json()` en sus cortes (no lo pedía su
criterio de terminado), y construir uno aquí solo para este informe sería
inventar una capacidad a un tipo ajeno en vez de declarar, con precedente
igual que `FittedPipelineSpec.calibrator: dict[str, Any]`, que esos campos
viajan opacos.

NINGÚN CAMPO DE EQUIDAD O APLICABILIDAD CLÍNICA. El criterio de terminado
lo dice explícito: «no afirmar equidad o aplicabilidad clínica por disponer
de subgrupos». La manera estructural de cumplirlo es no darle a este
esquema NINGÚN campo donde esa afirmación pudiera escribirse — `CLAVES` es
cerrado y no incluye nada parecido; quien alimente una tarjeta/recibo o un
TRIPOD lo hace leyendo los NÚMEROS de aquí (métricas, intervalos, curva de
fiabilidad) y redactando su propia conclusión fuera de este documento, no
rellenando un campo que este informe nunca ofrece.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import (
    EvaluationResult,
    RegistroDeMigraciones,
    _abrir,
    _saca,
    _secuencia,
    _sobre,
)
from matrixai.estudio.incertidumbre import Intervalo
from matrixai.estudio.validacion import digest_canonico, exigir_texto, solo_estas_claves

__all__ = ["InformeDeEvaluacion"]


@dataclass(frozen=True)
class InformeDeEvaluacion:
    """`evaluacion.json`. Referencia el pipeline y la partición exactos a
    través de `evaluacion` (`EvaluationResult`, ya validado por su propio
    esquema) e incorpora lo que 105-C2/C3/C5 hayan medido sobre ese mismo
    pipeline — nada de esto se calcula aquí."""

    ESQUEMA: ClassVar[str] = "matrixai.estudio.evaluacion_final_report"
    CLAVES: ClassVar[tuple[str, ...]] = (
        "informe_id", "evaluacion", "intervalos", "curva_de_fiabilidad",
        "recalibracion", "comparaciones")

    informe_id: str
    evaluacion: EvaluationResult
    intervalos: tuple[Intervalo, ...] = ()
    curva_de_fiabilidad: dict[str, Any] | None = None
    recalibracion: dict[str, Any] | None = None
    comparaciones: tuple[dict[str, Any], ...] = ()
    migracion: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.informe_id, "informe_id")
        if not isinstance(self.evaluacion, EvaluationResult):
            raise EsquemaInvalido("no_es_mapa", campo="evaluacion", valor=repr(self.evaluacion))
        object.__setattr__(self, "intervalos", tuple(self.intervalos))
        for intervalo in self.intervalos:
            if not isinstance(intervalo, Intervalo):
                raise EsquemaInvalido("no_es_mapa", campo="intervalos", valor=repr(intervalo))
        object.__setattr__(self, "comparaciones", tuple(self.comparaciones))
        for comparacion in self.comparaciones:
            if not isinstance(comparacion, dict):
                raise EsquemaInvalido("no_es_mapa", campo="comparaciones", valor=repr(comparacion))
        if self.curva_de_fiabilidad is not None and not isinstance(self.curva_de_fiabilidad, dict):
            raise EsquemaInvalido("no_es_mapa", campo="curva_de_fiabilidad",
                                  valor=repr(self.curva_de_fiabilidad))
        if self.recalibracion is not None and not isinstance(self.recalibracion, dict):
            raise EsquemaInvalido("no_es_mapa", campo="recalibracion", valor=repr(self.recalibracion))

    def _cuerpo(self) -> dict[str, Any]:
        return {
            "informe_id": self.informe_id,
            "evaluacion": self.evaluacion.a_json(),
            "intervalos": [i.a_json() for i in self.intervalos],
            "curva_de_fiabilidad": dict(self.curva_de_fiabilidad) if self.curva_de_fiabilidad else None,
            "recalibracion": dict(self.recalibracion) if self.recalibracion else None,
            "comparaciones": [dict(c) for c in self.comparaciones],
        }

    def a_json(self) -> dict[str, Any]:
        return _sobre(self.ESQUEMA, self._cuerpo(), self.migracion)

    def digest(self) -> str:
        return digest_canonico(_sobre(self.ESQUEMA, self._cuerpo(), None))

    @classmethod
    def desde_json(cls, payload: Any, *,
                   migraciones: "RegistroDeMigraciones | None" = None) -> "InformeDeEvaluacion":
        cuerpo, marca = _abrir(cls.ESQUEMA, payload, migraciones)
        solo_estas_claves(cuerpo, cls.CLAVES, cls.ESQUEMA)
        return cls(
            informe_id=_saca(cuerpo, "informe_id", cls.ESQUEMA),
            evaluacion=EvaluationResult.desde_json(_saca(cuerpo, "evaluacion", cls.ESQUEMA)),
            intervalos=tuple(Intervalo.desde_json(i) for i in
                             _secuencia(cuerpo.get("intervalos"), "intervalos")),
            curva_de_fiabilidad=cuerpo.get("curva_de_fiabilidad"),
            recalibracion=cuerpo.get("recalibracion"),
            comparaciones=tuple(_secuencia(cuerpo.get("comparaciones"), "comparaciones")),
            migracion=marca)
