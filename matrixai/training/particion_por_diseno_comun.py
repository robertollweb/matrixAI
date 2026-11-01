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

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from matrixai.training.diagnostico import Limite
from matrixai.training.objetivo import Bloqueo
from matrixai.training.particion_por_diseno_textos import motivo

__all__ = ["BloqueoConNumeros", "Rechazo", "clave_de_clase", "escritura_normalizada", "limite_de_sujetos_que_se_escriben_igual",
           "nombres_de_clases"]


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


# ---------------------------------------------------------------------------
# Auditoría I5 del 140b: «7» y «007» del mismo paciente
# ---------------------------------------------------------------------------

#: Cuántos grupos de valores se citan en la frase (la medida los lleva todos los que cuenta).
_EJEMPLOS_DE_ESCRITURA = 3


def escritura_normalizada(valor: str) -> str:
    """Cómo se escribe un identificador sin lo que suele cambiar al copiarlo de un sitio a otro: los ceros a la izquierda de
    cada número (`007` → `7`, `P-007` → `P-7`; Excel los quita al juntar hojas), los espacios y las mayúsculas. Solo sirve para
    AVISAR: el reparto sigue usando el valor tal cual (no se mapean los datos de nadie)."""
    sin_espacios = re.sub(r"\s+", "", str(valor)).casefold()
    return re.sub(r"\d+", lambda m: str(int(m.group())), sin_espacios)


def limite_de_sujetos_que_se_escriben_igual(unidad_por_id: Mapping[str, str], campo: str) -> Limite | None:
    """El aviso (un `Limite` con sus números) de que dos o más valores DISTINTOS de la columna del sujeto se escriben igual al
    normalizarlos (`escritura_normalizada`), o `None`. Si son el mismo sujeto escrito de dos maneras, el reparto los trata como
    dos y pueden caer uno en el desarrollo y otro en la prueba: justo lo que `groups` y `groups_and_time` prometen que no pasa.
    No los unifica: lo dice, con cuántos valores y cuántas filas, y la persona decide."""
    filas_de: dict[str, int] = {}
    for unidad in unidad_por_id.values():
        filas_de[unidad] = filas_de.get(unidad, 0) + 1
    por_escritura: dict[str, list[str]] = {}
    for unidad in filas_de:
        por_escritura.setdefault(escritura_normalizada(unidad), []).append(unidad)
    grupos = sorted((sorted(v) for v in por_escritura.values() if len(v) > 1), key=lambda g: (-sum(filas_de[u] for u in g), g))
    if not grupos:
        return None
    filas = sum(filas_de[u] for g in grupos for u in g)
    ejemplos = "; ".join(" = ".join(f"«{u}»" for u in g) + f" ({sum(filas_de[u] for u in g)} filas)"
                         for g in grupos[:_EJEMPLOS_DE_ESCRITURA])
    ejemplos_en = "; ".join(" = ".join(f"“{u}”" for u in g) + f" ({sum(filas_de[u] for u in g)} rows)"
                            for g in grupos[:_EJEMPLOS_DE_ESCRITURA])
    textos = motivo("sujetos_que_se_escriben_igual", campo=campo, valor=len(grupos), filas=filas,
                    valores=sum(len(g) for g in grupos), ejemplos="{ejemplos}")
    return Limite(clave="sujetos_que_se_escriben_igual", campo="unit_id_field",
                  motivo={"es": textos["es"].replace("{ejemplos}", ejemplos),
                          "en": textos["en"].replace("{ejemplos}", ejemplos_en)},
                  medida={"columna": campo, "grupos": len(grupos), "valores": sum(len(g) for g in grupos), "filas": filas,
                          "ejemplos": [{"valores": g, "filas": sum(filas_de[u] for u in g)} for g in grupos[:10]]})
