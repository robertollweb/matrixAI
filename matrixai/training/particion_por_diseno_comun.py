# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140-C1 — lo que comparten las piezas de la partición por grupos y por tiempo.

Dos cosas pequeñas, en un módulo aparte para que `particion_por_diseno_tiempo`
y `particion_por_diseno_grupos` no se importen entre sí:

- `Rechazo`: una razón por la que el reparto NO se puede hacer. Las funciones
  puras la lanzan; `proponer_particion` la convierte en un `Bloqueo` con sus
  números. Nunca llega al llamante como una excepción cruda de Python.
- `BloqueoConNumeros`: un `Bloqueo` (mismo contrato: `clave`, `motivo`,
  `campo`) que además lleva la `medida` —los números del rechazo— como la
  llevan los `Limite`. Es una subclase para no tocar `Bloqueo` (compartido con
  el 103-C1): quien solo lee `clave`/`motivo` lo ve igual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from matrixai.training.objetivo import Bloqueo
from matrixai.training.particion_por_diseno_textos import motivo

__all__ = ["BloqueoConNumeros", "Rechazo", "clave_de_clase", "nombres_de_clases"]


@dataclass(frozen=True)
class BloqueoConNumeros(Bloqueo):
    medida: dict[str, Any] = field(default_factory=dict)

    def a_json(self) -> dict[str, Any]:
        return {**super().a_json(), "medida": dict(self.medida)}


class Rechazo(Exception):
    """El reparto pedido no se puede hacer, y por qué (con números)."""

    def __init__(self, clave: str, rol: str | None, medida: dict[str, Any] | None = None,
                 **textos: object) -> None:
        """`rol` es el ROL del campo (`unit_id_field`, `time_column`…), como
        en los demás `Bloqueo`/`Limite` de este módulo; si el texto debe nombrar
        la COLUMNA del CSV, va en `textos` como `campo=`."""
        super().__init__(clave)
        self.clave = clave
        self.campo = rol
        self.medida = dict(medida or {})
        self.textos = textos

    def bloqueo(self) -> BloqueoConNumeros:
        return BloqueoConNumeros(clave=self.clave, campo=self.campo,
                                 motivo=motivo(self.clave, **{"campo": self.campo, **self.textos}),
                                 medida=self.medida)


def clave_de_clase(clase: Any) -> str:
    """Cómo se nombra una clase en un mensaje y en una medida (JSON no admite
    claves que no sean texto)."""
    return str(clase)


def nombres_de_clases(clases: Any) -> dict[Any, str]:
    """El nombre de cada clase en los mensajes y medidas de un mismo reparto: `clave_de_clase`, salvo que dos clases
    DISTINTAS se escriban igual (`1` y `"1"`), y entonces las dos van con `repr` (`1` y `'1'`).

    Re-auditoría (M-c, 28-10): las dos se juntaban en las medidas (`{'0': 9, '1': 2}`, una clave perdida) y el motivo
    decía «la clase 1 solo tiene 0 filas en desarrollo» habiendo filas de `1`. Desde un CSV no pasa (todo llega como
    texto); lo mínimo honesto es no escribir dos clases con el mismo nombre. No se juntan: son dos clases."""
    clases = list(clases)
    por_texto: dict[str, int] = {}
    for c in clases:
        por_texto[clave_de_clase(c)] = por_texto.get(clave_de_clase(c), 0) + 1
    return {c: (repr(c) if por_texto[clave_de_clase(c)] > 1 else clave_de_clase(c)) for c in clases}
