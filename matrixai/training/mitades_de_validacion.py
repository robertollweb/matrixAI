# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140-C1 — las dos mitades de una `valida`: calibrar / elegir el umbral.

**Para qué.** El reentrenamiento con todo el desarrollo (139) puntúa cada
pliegue con un modelo que no lo vio y parte las filas de cada `valida` en dos:
con una se calibra y con la otra se elige el umbral. Hoy esa partición es
POSICIONAL (`len(ids) // 2`, `reentrenar_con_todo._mitades_fuera_de_pliegue`).
Eso vale para `iid` —la `valida` sale barajada— pero con otros diseños:

- en `groups`, una unidad puede quedar partida entre las dos mitades: sin fuga
  hacia la prueba, pero no es el aislamiento que el diseño promete;
- en `temporal`, la mitad posicional no es cronológica si la `valida` no viene
  ordenada, y el umbral se elegiría con datos ANTERIORES a los de calibrar.

**Qué hace.** Una función pura, `partir_valida_en_mitades`, sin cablear (el
backend la usará en el C2):

- `iid`: EXACTAMENTE las mitades posicionales de hoy (`ids[:n//2]`, `ids[n//2:]`).
- `groups`: reparte las UNIDADES —nunca las filas— en dos mitades de tamaño
  parecido, con las clases repartidas (la misma pieza voraz del reparto de los
  pliegues). Una sola unidad no se puede partir: se corta por posición y se
  DECLARA (`Limite`).
- `temporal`: la parte ANTERIOR calibra y la POSTERIOR elige el umbral, con el
  corte entre dos instantes distintos (filas del mismo instante, a un lado).
  Si todo es un solo instante, posicional y declarado.

Devuelve ids e ÍNDICES (posiciones en la lista recibida) de cada mitad: quien
llama puede cortar una `Muestra` por posición sin volver a buscar ids.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from matrixai.training.diagnostico import Limite
from matrixai.training.particion_por_diseno_comun import clave_de_clase
from matrixai.training.particion_por_diseno_grupos import repartir_cubos
from matrixai.training.particion_por_diseno_textos import motivo

__all__ = ["DISENOS_CON_MITADES", "MitadesDeValida", "partir_valida_en_mitades"]

DISENOS_CON_MITADES = ("iid", "groups", "temporal")


@dataclass(frozen=True)
class MitadesDeValida:
    para_calibrar: tuple[str, ...]
    para_el_umbral: tuple[str, ...]
    #: Posiciones, dentro de los `ids` recibidos, de cada mitad.
    indices_para_calibrar: tuple[int, ...]
    indices_para_el_umbral: tuple[int, ...]
    limites: tuple[Limite, ...] = ()


def _resultado(ids: Sequence[str], calibrar: Sequence[int], umbral: Sequence[int],
               limites: Sequence[Limite] = ()) -> MitadesDeValida:
    return MitadesDeValida(tuple(ids[i] for i in calibrar), tuple(ids[i] for i in umbral),
                           tuple(calibrar), tuple(umbral), tuple(limites))


def partir_valida_en_mitades(ids: Sequence[str], *, diseno: str,
                             unidad_por_id: Mapping[str, str] | None = None,
                             tiempo_por_id: Mapping[str, float] | None = None,
                             clase_por_id: Mapping[str, Any] | None = None,
                             semilla: int = 0) -> MitadesDeValida:
    """Las filas `ids` de UNA `valida` en dos mitades. `diseno` en
    `DISENOS_CON_MITADES`; `groups_and_time` no está soportado (140b)."""
    n = len(ids)
    if diseno == "iid":
        mitad = n // 2
        return _resultado(ids, range(mitad), range(mitad, n))
    if diseno == "temporal":
        if tiempo_por_id is None:
            raise ValueError("el diseño temporal necesita tiempo_por_id")
        return _mitades_cronologicas(ids, tiempo_por_id)
    if diseno == "groups":
        if unidad_por_id is None:
            raise ValueError("el diseño groups necesita unidad_por_id")
        return _mitades_por_unidad(ids, unidad_por_id, clase_por_id, semilla)
    raise ValueError(f"el diseño {diseno!r} no tiene mitades de validación (soportados: {DISENOS_CON_MITADES})")


def _mitades_cronologicas(ids: Sequence[str], tiempo: Mapping[str, float]) -> MitadesDeValida:
    n = len(ids)
    orden = sorted(range(n), key=lambda i: (tiempo[ids[i]], ids[i]))
    # cortes posibles: entre dos instantes distintos
    cortes = [c for c in range(1, n) if tiempo[ids[orden[c - 1]]] != tiempo[ids[orden[c]]]]
    limites: list[Limite] = []
    if cortes:
        corte = min(cortes, key=lambda c: (abs(c - n / 2), c))
    else:
        corte = n // 2
        if n > 1:
            limites.append(Limite(
                clave="valida_con_un_solo_instante", campo="time_column",
                motivo=motivo("valida_con_un_solo_instante", campo="time_column"),
                medida={"filas": n}))
    return _resultado(ids, orden[:corte], orden[corte:], limites)


def _mitades_por_unidad(ids: Sequence[str], unidad_por_id: Mapping[str, str],
                        clase_por_id: Mapping[str, Any] | None, semilla: int) -> MitadesDeValida:
    n = len(ids)
    posiciones: dict[str, list[int]] = {}
    for i, obs in enumerate(ids):
        posiciones.setdefault(str(unidad_por_id[obs]), []).append(i)
    if len(posiciones) < 2:
        mitad = n // 2
        limites = (Limite(clave="valida_con_una_sola_unidad", campo="unit_id_field",
                          motivo=motivo("valida_con_una_sola_unidad", campo="unit_id_field"),
                          medida={"filas": n, "unidades": len(posiciones)}),) if n > 1 else ()
        return _resultado(ids, range(mitad), range(mitad, n), limites)

    clase_de = (lambda obs: clase_por_id[obs]) if clase_por_id is not None else (lambda obs: 0)
    cuentas: dict[str, dict[Any, int]] = {}
    for unidad, pos in posiciones.items():
        c: dict[Any, int] = {}
        for i in pos:
            c[clase_de(ids[i])] = c.get(clase_de(ids[i]), 0) + 1
        cuentas[unidad] = c
    clases = sorted({k for c in cuentas.values() for k in c}, key=str)
    total = {k: sum(c.get(k, 0) for c in cuentas.values()) for k in clases}
    objetivos = [{k: total[k] / 2 for k in clases} for _ in range(2)]
    mitades = repartir_cubos(cuentas, clases, objetivos, random.Random(f"{semilla}:mitades"))
    if mitades is None:       # no hay manera de que las dos tengan todas las clases: se reparte igualmente y se declara
        mitades = repartir_cubos(cuentas, clases, objetivos, random.Random(f"{semilla}:mitades"), cubrir=False)
    if not mitades[0] or not mitades[1]:          # con ≥ 2 unidades, nunca una mitad vacía
        origen, destino = (1, 0) if not mitades[0] else (0, 1)
        pequena = min(mitades[origen], key=lambda u: (sum(cuentas[u].values()), u))
        mitades[origen].remove(pequena)
        mitades[destino].append(pequena)
    calibrar = sorted(i for u in mitades[0] for i in posiciones[u])
    umbral = sorted(i for u in mitades[1] for i in posiciones[u])

    limites: list[Limite] = []
    if clase_por_id is not None and len(clases) > 1:
        for k in clases:
            en_cada = (sum(1 for i in calibrar if clase_por_id[ids[i]] == k),
                       sum(1 for i in umbral if clase_por_id[ids[i]] == k))
            if 0 in en_cada:
                limites.append(Limite(
                    clave="mitad_sin_una_clase", campo="unit_id_field",
                    motivo=motivo("mitad_sin_una_clase", clase=clave_de_clase(k),
                                  detalle=f"{en_cada[0]} / {en_cada[1]}"),
                    medida={"clase": clave_de_clase(k), "filas_para_calibrar": en_cada[0],
                            "filas_para_el_umbral": en_cada[1]}))
    return _resultado(ids, calibrar, umbral, limites)
