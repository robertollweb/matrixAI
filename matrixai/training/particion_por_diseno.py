# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C4 — particiones por diseño: proponer un `SplitPlan` (104-C0) —
desarrollo / prueba / cohorte externa, seguro por unidad y por cronología —
y, DENTRO de desarrollo, los pliegues de validación cruzada para búsqueda y
calibración.

FORMA YA EXISTENTE, REUTILIZADA. `SplitPlan` (104-C0) ya valida el
vocabulario de diseño (`iid`/`temporal`/`groups`/`groups_and_time`) y ya
hace cumplir el aislamiento de unidades ENTRE ROLES (`_comprobar_
aislamiento_de_unidades`: ninguna unidad aparece en dos roles a la vez).
Este corte no repite esa comprobación — construye el GENERADOR: el
algoritmo que decide qué fila va en qué rol y qué pliegue, de forma que el
`SplitPlan` resultante pase esa comprobación porque el reparto ya la
respeta, no porque se le pida después.

LA CRONOLOGÍA, QUE `SplitPlan` NO PUEDE VERIFICAR POR SU CUENTA (texto
literal del criterio: «se verifica intersección de conjuntos y cronología,
no solo hashes diferentes»). `SplitPlan` es un esquema — no lleva los
valores de fecha de cada fila, solo la asignación de rol. Comprobar que
"prueba" es CRONOLÓGICAMENTE POSTERIOR a "desarrollo" (no solo un conjunto
disjunto de ids) es trabajo de este módulo, que sí tiene las fechas.

UNIDAD, NO FILA, ES LA UNIDAD DE TIEMPO EN `groups_and_time`. Con datos
agrupados, la fecha de referencia de una unidad es su observación MÁS
TARDÍA — las unidades se ordenan por esa fecha y se reparten enteras
(nunca partidas) a un lado u otro del corte. Esto no promete cero
solapamiento fila a fila entre desarrollo y prueba (una unidad con un
rango de fechas amplio puede tener filas tempranas del lado "tarde"); es
una definición declarada de cronología a nivel de unidad, no una en la que
no se puede confiar — la alternativa (partir una unidad por fecha de fila)
rompería el aislamiento de unidades, que es el invariante más fuerte de
los dos.

REDUCIR PLIEGUES, NUNCA ROMPER UN GRUPO (texto literal). Si se piden más
pliegues de los que hay grupos —o eventos de la clase minoritaria— para
sostenerlos, se reducen con un `Limite` declarado; con menos de 2 grupos
en desarrollo, el diseño es un `Bloqueo`: no hay con qué separar
desarrollo de una prueba que generalice a unidades nuevas.

FUERA DE ESTE CORTE, CON SU MOTIVO. La validación cruzada temporal
"rodante" (rolling-origin/expanding window) no se implementa: `temporal` y
`groups_and_time` producen UN solo corte desarrollo/prueba, sin rotación
de pliegues dentro de desarrollo — mezclar tiempo con k-fold shuffling
sería fingir independencia donde el diseño declara que no la hay
(105-C2, invariante 5, el mismo principio aplicado aquí). La
estratificación de `groups` por clase (equilibrar la clase minoritaria
entre pliegues de grupos, no solo el número de filas) tampoco: el criterio
la declara condicional («solo cuando sea compatible»), y una heurística
combinatoria a medias sería peor que no tenerla.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from matrixai.estudio import Horizonte, SplitPlan
from matrixai.estudio.validacion import digest_canonico
from matrixai.training.diagnostico import Bloqueo, Limite
from matrixai.training.particion_por_diseno_textos import motivo

__all__ = [
    "MINIMO_GRUPOS", "AsignacionDePliegues", "Pliegue", "PropuestaDeParticion",
    "proponer_particion",
]

#: Con menos de 2 grupos en desarrollo no hay con qué separar desarrollo de
#: una prueba que generalice a unidades nuevas.
MINIMO_GRUPOS = 2

_SEGUNDOS_POR_UNIDAD = {
    "seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0,
    "weeks": 604800.0, "months": 2629800.0, "years": 31557600.0,
}


def _segundos_de(horizonte: Horizonte) -> float:
    return horizonte.magnitud * _SEGUNDOS_POR_UNIDAD[horizonte.unidad]


def _como_epoch(valor: Any) -> float:
    if isinstance(valor, datetime):
        return valor.timestamp()
    if isinstance(valor, date):
        return datetime(valor.year, valor.month, valor.day).timestamp()
    return float(valor)


@dataclass(frozen=True)
class Pliegue:
    """Un pliegue de validación cruzada DENTRO de desarrollo — nunca toca
    `test` ni `external_test`."""

    repeticion: int
    pliegue: int
    entrena: tuple[str, ...]
    valida: tuple[str, ...]

    def a_json(self) -> dict[str, Any]:
        return {"repeticion": self.repeticion, "pliegue": self.pliegue,
                "entrena": list(self.entrena), "valida": list(self.valida)}


@dataclass(frozen=True)
class AsignacionDePliegues:
    folds: int
    repeats: int
    seed: int
    pliegues: tuple[Pliegue, ...]

    def pliegue_de(self, repeticion: int, pliegue: int) -> Pliegue | None:
        return next((p for p in self.pliegues
                    if p.repeticion == repeticion and p.pliegue == pliegue), None)

    def a_json(self) -> dict[str, Any]:
        return {"folds": self.folds, "repeats": self.repeats, "seed": self.seed,
                "pliegues": [p.a_json() for p in self.pliegues]}

    def digest(self) -> str:
        return digest_canonico(self.a_json())


@dataclass(frozen=True)
class PropuestaDeParticion:
    plan: SplitPlan | None
    pliegues: AsignacionDePliegues | None
    bloqueos: tuple[Bloqueo, ...] = ()
    limites: tuple[Limite, ...] = ()

    @property
    def es_viable(self) -> bool:
        return not self.bloqueos and self.plan is not None


def _reparto_kfold(ids: Sequence[str], folds: int, azar: random.Random) -> list[list[str]]:
    barajados = list(ids)
    azar.shuffle(barajados)
    cubos: list[list[str]] = [[] for _ in range(folds)]
    for i, obs in enumerate(barajados):
        cubos[i % folds].append(obs)
    return cubos


def _pliegues_desde_cubos(cubos: list[list[str]], repeticion: int) -> list[Pliegue]:
    todas = [obs for cubo in cubos for obs in cubo]
    resultado = []
    for i, valida in enumerate(cubos):
        entrena = tuple(o for o in todas if o not in set(valida))
        resultado.append(Pliegue(repeticion=repeticion, pliegue=i,
                                 entrena=entrena, valida=tuple(valida)))
    return resultado


def _kfold_iid(ids: Sequence[str], etiqueta_por_id: Mapping[str, Any] | None,
               folds: int, repeats: int, seed: int) -> tuple[tuple[Pliegue, ...], tuple[Limite, ...]]:
    limites: list[Limite] = []
    if etiqueta_por_id:
        conteos: dict[Any, int] = {}
        for obs in ids:
            conteos[etiqueta_por_id[obs]] = conteos.get(etiqueta_por_id[obs], 0) + 1
        if len(conteos) < 2:
            limites.append(Limite(clave="una_sola_clase_no_estratifica", campo="objetivo",
                                  motivo=motivo("una_sola_clase_no_estratifica", campo="objetivo"),
                                  medida={"clases": len(conteos)}))
            etiqueta_por_id = None
        else:
            minoria = min(conteos.values())
            if minoria < folds:
                nuevos_folds = max(2, minoria)
                limites.append(Limite(
                    clave="pliegues_reducidos_por_eventos", campo="objetivo",
                    motivo=motivo("pliegues_reducidos_por_eventos", opciones=folds, valor=nuevos_folds),
                    medida={"pliegues_pedidos": folds, "pliegues_usados": nuevos_folds,
                            "eventos_clase_minoritaria": minoria}))
                folds = nuevos_folds

    pliegues: list[Pliegue] = []
    for r in range(repeats):
        azar = random.Random(f"{seed}:iid:{r}")
        if etiqueta_por_id:
            por_clase: dict[Any, list[str]] = {}
            for obs in ids:
                por_clase.setdefault(etiqueta_por_id[obs], []).append(obs)
            cubos: list[list[str]] = [[] for _ in range(folds)]
            # reparto estratificado: cada clase se baraja y se reparte round-robin
            for clase in sorted(por_clase, key=str):
                miembros = list(por_clase[clase])
                azar.shuffle(miembros)
                for i, obs in enumerate(miembros):
                    cubos[i % folds].append(obs)
            # 101-C4, hallazgo real (2026-09-08/09): el reparto de arriba
            # equilibra las clases ENTRE pliegues (el objetivo de
            # estratificar) pero deja el orden DENTRO de cada pliegue
            # agrupado por clase -- todas las de la primera clase
            # procesada, luego todas las de la segunda. Un llamante que
            # reparte `pliegue.valida` por POSICIÓN en vez de barajar antes
            # de partir (calibración/selección en `informe_101_c4.py` y en
            # `estudio_job.py`) hereda ese sesgo entero: medido con "adult"
            # real, la primera mitad posicional de un pliegue.valida de
            # 7.815 filas dio 3.907/3.907 de una sola clase, cero de la
            # otra. Barajar cada cubo, con la MISMA `azar` (determinista,
            # sembrada), deshace el orden agrupado sin tocar qué filas
            # caen en qué pliegue -- el equilibrio entre pliegues no cambia.
            for cubo in cubos:
                azar.shuffle(cubo)
        else:
            cubos = _reparto_kfold(ids, folds, azar)
        pliegues.extend(_pliegues_desde_cubos(cubos, r))
    return tuple(pliegues), tuple(limites)


def _kfold_groups(ids: Sequence[str], unidad_por_id: Mapping[str, str],
                  folds: int, repeats: int, seed: int) -> tuple[tuple[Pliegue, ...], tuple[Limite, ...]]:
    filas_por_unidad: dict[str, list[str]] = {}
    for obs in ids:
        filas_por_unidad.setdefault(unidad_por_id[obs], []).append(obs)
    grupos = sorted(filas_por_unidad)
    limites: list[Limite] = []
    if len(grupos) < folds:
        nuevos_folds = max(2, len(grupos))
        limites.append(Limite(
            clave="pliegues_reducidos_por_grupos", campo="unit_id_field",
            motivo=motivo("pliegues_reducidos_por_grupos", opciones=folds, valor=nuevos_folds),
            medida={"pliegues_pedidos": folds, "pliegues_usados": nuevos_folds, "grupos": len(grupos)}))
        folds = nuevos_folds

    pliegues: list[Pliegue] = []
    for r in range(repeats):
        azar = random.Random(f"{seed}:groups:{r}")
        cubos_de_grupos = _reparto_kfold(grupos, folds, azar)
        cubos_de_filas = [[obs for grupo in cubo for obs in filas_por_unidad[grupo]]
                          for cubo in cubos_de_grupos]
        pliegues.extend(_pliegues_desde_cubos(cubos_de_filas, r))
    return tuple(pliegues), tuple(limites)


def proponer_particion(filas: Sequence[Mapping[str, Any]], *, plan_id: str,
                       observation_id_field: str, split_type: str, seed: int,
                       test_fraction: float = 0.2, folds: int = 5, repeats: int = 1,
                       unit_id_field: str | None = None, time_column: str | None = None,
                       gap: Horizonte | None = None, objetivo: str | None = None,
                       ids_cohorte_externa: Sequence[str] = ()) -> PropuestaDeParticion:
    """`split_type` en `("iid", "temporal", "groups", "groups_and_time")`.
    `ids_cohorte_externa` se retira ANTES de repartir nada más — esas filas
    reciben el rol `external_test` y no entran en ningún pliegue."""
    todos = {str(f[observation_id_field]): f for f in filas}
    externos = set(str(i) for i in ids_cohorte_externa)
    internos = {obs: fila for obs, fila in todos.items() if obs not in externos}

    unidad_por_id = ({obs: str(fila[unit_id_field]) for obs, fila in internos.items()}
                     if unit_id_field else None)
    tiempo_por_id = ({obs: _como_epoch(fila[time_column]) for obs, fila in internos.items()}
                     if time_column else None)
    etiqueta_por_id = ({obs: fila[objetivo] for obs, fila in internos.items()}
                       if objetivo else None)

    if split_type == "groups":
        n_grupos = len(set(unidad_por_id.values())) if unidad_por_id else 0
        if n_grupos < MINIMO_GRUPOS:
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(
                Bloqueo(clave="grupos_insuficientes", campo="unit_id_field",
                       motivo=motivo("grupos_insuficientes", campo="unit_id_field", valor=n_grupos)),))
        return _proponer_groups(plan_id, internos, externos, unidad_por_id, observation_id_field,
                                test_fraction, seed, folds, repeats, etiqueta_por_id, unit_id_field)

    if split_type == "iid":
        return _proponer_iid(plan_id, internos, externos, observation_id_field,
                             test_fraction, seed, folds, repeats, etiqueta_por_id)

    if split_type == "temporal":
        return _proponer_temporal(plan_id, internos, externos, tiempo_por_id, observation_id_field,
                                  test_fraction, gap, time_column)

    if split_type == "groups_and_time":
        n_grupos = len(set(unidad_por_id.values())) if unidad_por_id else 0
        if n_grupos < MINIMO_GRUPOS:
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(
                Bloqueo(clave="grupos_insuficientes", campo="unit_id_field",
                       motivo=motivo("grupos_insuficientes", campo="unit_id_field", valor=n_grupos)),))
        return _proponer_groups_and_time(plan_id, internos, externos, unidad_por_id, tiempo_por_id,
                                         observation_id_field, test_fraction, gap,
                                         unit_id_field, time_column)

    raise ValueError(f"split_type desconocido: {split_type!r}")


def _plan_con_roles(plan_id: str, split_type: str, roles: dict[str, str],
                    observation_id_field: str, *, unit_id_field: str | None = None,
                    units: dict[str, str] | None = None, time_column: str | None = None,
                    gap: Horizonte | None = None, folds: int | None = None,
                    repeats: int | None = None, seed: int | None = None) -> SplitPlan:
    return SplitPlan(plan_id=plan_id, split_type=split_type, assignments=roles,
                     observation_id_field=observation_id_field, units=units,
                     unit_id_field=unit_id_field, time_column=time_column, gap=gap,
                     folds=folds, repeats=repeats, seed=seed)


def _proponer_iid(plan_id, internos, externos, observation_id_field, test_fraction, seed,
                  folds, repeats, etiqueta_por_id):
    ids = sorted(internos)
    azar = random.Random(f"{seed}:corte")
    if etiqueta_por_id:
        por_clase: dict[Any, list[str]] = {}
        for obs in ids:
            por_clase.setdefault(etiqueta_por_id[obs], []).append(obs)
        prueba: list[str] = []
        for clase in sorted(por_clase, key=str):
            miembros = list(por_clase[clase])
            azar.shuffle(miembros)
            corte = round(len(miembros) * test_fraction)
            prueba.extend(miembros[:corte])
    else:
        barajados = list(ids)
        azar.shuffle(barajados)
        corte = round(len(barajados) * test_fraction)
        prueba = barajados[:corte]
    prueba_set = set(prueba)
    roles = {obs: ("test" if obs in prueba_set else "development") for obs in ids}
    roles.update({obs: "external_test" for obs in externos})

    desarrollo = [obs for obs in ids if obs not in prueba_set]
    pliegues_tupla, limites = _kfold_iid(desarrollo, etiqueta_por_id, folds, repeats, seed)
    plan = _plan_con_roles(plan_id, "iid", roles, observation_id_field,
                           folds=folds, repeats=repeats, seed=seed)
    asignacion = AsignacionDePliegues(folds=pliegues_tupla[-1].pliegue + 1 if pliegues_tupla else folds,
                                      repeats=repeats, seed=seed, pliegues=pliegues_tupla)
    return PropuestaDeParticion(plan=plan, pliegues=asignacion, limites=limites)


def _proponer_groups(plan_id, internos, externos, unidad_por_id, observation_id_field,
                     test_fraction, seed, folds, repeats, etiqueta_por_id, unit_id_field):
    filas_por_unidad: dict[str, list[str]] = {}
    for obs in internos:
        filas_por_unidad.setdefault(unidad_por_id[obs], []).append(obs)
    grupos = sorted(filas_por_unidad)
    azar = random.Random(f"{seed}:corte")
    azar.shuffle(grupos)

    n_total = len(internos)
    objetivo_prueba = round(n_total * test_fraction)
    prueba: list[str] = []
    grupos_prueba: list[str] = []
    for grupo in grupos:
        if len(prueba) >= objetivo_prueba:
            break
        prueba.extend(filas_por_unidad[grupo])
        grupos_prueba.append(grupo)
    prueba_set = set(prueba)

    roles = {obs: ("test" if obs in prueba_set else "development") for obs in internos}
    roles.update({obs: "external_test" for obs in externos})
    units = dict(unidad_por_id)

    desarrollo = [obs for obs in internos if obs not in prueba_set]
    pliegues_tupla, limites = _kfold_groups(desarrollo, unidad_por_id, folds, repeats, seed)
    plan = _plan_con_roles(plan_id, "groups", roles, observation_id_field,
                           unit_id_field=unit_id_field, units=units,
                           folds=folds, repeats=repeats, seed=seed)
    asignacion = AsignacionDePliegues(folds=pliegues_tupla[-1].pliegue + 1 if pliegues_tupla else folds,
                                      repeats=repeats, seed=seed, pliegues=pliegues_tupla)
    return PropuestaDeParticion(plan=plan, pliegues=asignacion, limites=limites)


def _proponer_temporal(plan_id, internos, externos, tiempo_por_id, observation_id_field,
                       test_fraction, gap, time_column):
    ids = sorted(internos, key=lambda o: (tiempo_por_id[o], o))
    n = len(ids)
    corte_idx = n - round(n * test_fraction)
    tiempo_de_corte = tiempo_por_id[ids[corte_idx]] if corte_idx < n else tiempo_por_id[ids[-1]]
    medio_gap = _segundos_de(gap) / 2.0 if gap is not None else 0.0

    desarrollo = [o for o in ids if tiempo_por_id[o] < tiempo_de_corte - medio_gap]
    prueba = [o for o in ids if tiempo_por_id[o] >= tiempo_de_corte + medio_gap]

    if not desarrollo or not prueba:
        return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(
            Bloqueo(clave="cronologia_no_se_pudo_separar", campo="time_column",
                   motivo=motivo("cronologia_no_se_pudo_separar")),))

    assert max(tiempo_por_id[o] for o in desarrollo) <= min(tiempo_por_id[o] for o in prueba)

    roles = {obs: "development" for obs in desarrollo}
    roles.update({obs: "test" for obs in prueba})
    roles.update({obs: "external_test" for obs in externos})
    plan = _plan_con_roles(plan_id, "temporal", roles, observation_id_field,
                           time_column=time_column, gap=gap)
    return PropuestaDeParticion(plan=plan, pliegues=None)


def _proponer_groups_and_time(plan_id, internos, externos, unidad_por_id, tiempo_por_id,
                              observation_id_field, test_fraction, gap,
                              unit_id_field, time_column):
    filas_por_unidad: dict[str, list[str]] = {}
    for obs in internos:
        filas_por_unidad.setdefault(unidad_por_id[obs], []).append(obs)
    referencia_por_unidad = {u: max(tiempo_por_id[o] for o in obs_de_u)
                             for u, obs_de_u in filas_por_unidad.items()}
    grupos = sorted(filas_por_unidad, key=lambda u: (referencia_por_unidad[u], u))

    n_total = len(internos)
    objetivo_prueba = round(n_total * test_fraction)
    medio_gap = _segundos_de(gap) / 2.0 if gap is not None else 0.0

    filas_por_grupo_acum = 0
    corte_grupo_idx = len(grupos)
    for i in range(len(grupos) - 1, -1, -1):
        filas_por_grupo_acum += len(filas_por_unidad[grupos[i]])
        if filas_por_grupo_acum >= objetivo_prueba:
            corte_grupo_idx = i
            break
    tiempo_de_corte = (referencia_por_unidad[grupos[corte_grupo_idx]]
                       if corte_grupo_idx < len(grupos) else referencia_por_unidad[grupos[-1]])

    desarrollo_grupos = [u for u in grupos if referencia_por_unidad[u] < tiempo_de_corte - medio_gap]
    prueba_grupos = [u for u in grupos if referencia_por_unidad[u] >= tiempo_de_corte + medio_gap]

    if not desarrollo_grupos or not prueba_grupos:
        return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(
            Bloqueo(clave="cronologia_no_se_pudo_separar", campo="time_column",
                   motivo=motivo("cronologia_no_se_pudo_separar")),))

    desarrollo = [o for u in desarrollo_grupos for o in filas_por_unidad[u]]
    prueba = [o for u in prueba_grupos for o in filas_por_unidad[u]]
    roles = {obs: "development" for obs in desarrollo}
    roles.update({obs: "test" for obs in prueba})
    roles.update({obs: "external_test" for obs in externos})
    # las unidades embargadas por el hueco no tienen rol: `units` solo puede
    # declarar observaciones que SÍ están en `roles`, o SplitPlan las rechaza.
    units = {obs: unidad for obs, unidad in unidad_por_id.items() if obs in roles}
    plan = _plan_con_roles(plan_id, "groups_and_time", roles, observation_id_field,
                           unit_id_field=unit_id_field, units=units,
                           time_column=time_column, gap=gap)
    return PropuestaDeParticion(plan=plan, pliegues=None)
