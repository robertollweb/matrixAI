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

140-C1 — GRUPOS Y TIEMPO DE VERDAD. Lo que el 103-C4 dejó fuera, con su motivo,
entra ahora (contrato 140, decisiones D1 y D2 de Roberto, 27-10-2026):

- `temporal` da PLIEGUES: ventana que se expande con hueco
  (`particion_por_diseno_tiempo`). `entrena` sale ordenado por tiempo, las filas
  del mismo instante nunca quedan a los dos lados de un corte, el pliegue 0 es el
  más reciente y `repeats` se fija a 1 (declarado). El primer bloque solo entrena:
  la propuesta dice cuántas filas de desarrollo no tienen validación fuera de
  pliegue (`cobertura.filas_sin_validacion`).
- `groups` reparte la CLASE entre grupos, sin partir ninguno
  (`particion_por_diseno_grupos`), y con la guarda de D2: «repartir la clase
  entre grupos; si no cabe, bajar el número de pliegues hasta 2 y decirlo; si ni
  así, rechazar el estudio con los números». Lo mismo para la prueba reservada y
  para `temporal`: nunca una sola clase en silencio.
- El tiempo llega como lo trae la gente: texto del CSV (con `fechas.py`; las
  fechas ambiguas se RECHAZAN), `datetime` o número, y el `gap` como el `dict` del
  JSON o un `Horizonte`.
- `iid` NO cambia (bit a bit, con prueba que lo compara con el código anterior)
  y `groups_and_time` tampoco: sigue dando un solo corte sin pliegues, fuera de
  este corte (140b) porque su definición no sirve para paneles (medido).
- Sin objetivo de clasificación (regresión) o con una sola clase, `groups` reparte
  como siempre (por grupos barajados) y lo declara si había una sola clase.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from matrixai.estudio import Horizonte, SplitPlan
from matrixai.estudio.validacion import digest_canonico
from matrixai.training.diagnostico import Bloqueo, Limite
from matrixai.training.dataset_analysis import _is_null
from matrixai.training.particion_por_diseno_comun import BloqueoConNumeros, Rechazo, nombres_de_clases
from matrixai.training.particion_por_diseno_grupos import (
    _limite_de_fraccion, fuera_del_margen, repartir_grupos_estratificado)
from matrixai.training.particion_por_diseno_textos import motivo
from matrixai.training.particion_por_diseno_tiempo import (
    convertir_gap, convertir_tiempos, elegir_ventana, segundos_de)

__all__ = [
    "MINIMO_GRUPOS", "AsignacionDePliegues", "CoberturaDePliegues", "Pliegue",
    "PropuestaDeParticion", "proponer_particion",
]

#: Con menos de 2 grupos en desarrollo no hay con qué separar desarrollo de
#: una prueba que generalice a unidades nuevas.
MINIMO_GRUPOS = 2
#: `groups` sin guardas de clase: la prueba no se lleva tantos grupos que a desarrollo le queden menos que estos
#: (con 1 no hay dos pliegues que no compartan sujeto). Auditoría I5.
_MINIMO_GRUPOS_EN_DESARROLLO = 2

_segundos_de = segundos_de


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
class CoberturaDePliegues:
    """Qué filas de desarrollo están en alguna `valida` (140-C1).

    En `iid` y `groups` las `valida` de una repetición cubren TODO el
    desarrollo. En `temporal` NO: el primer bloque solo entrena, así que sus
    filas no tienen predicción fuera de pliegue. El backend (C2) lo declara
    como `cobertura_fuera_de_pliegue`."""

    filas_de_desarrollo: int
    #: Los ids de desarrollo que no están en ninguna `valida` (la primera repetición).
    ids_sin_validacion: tuple[str, ...]
    #: Cuántas filas quitó el hueco de `entrena`, por pliegue (vacío si no hay hueco o no es `temporal`).
    quitadas_por_el_hueco: tuple[int, ...] = ()

    @property
    def filas_sin_validacion(self) -> int:
        return len(self.ids_sin_validacion)

    @property
    def filas_con_validacion(self) -> int:
        return self.filas_de_desarrollo - len(self.ids_sin_validacion)

    def a_json(self) -> dict[str, Any]:
        return {"filas_de_desarrollo": self.filas_de_desarrollo,
                "filas_con_validacion": self.filas_con_validacion,
                "filas_sin_validacion": self.filas_sin_validacion,
                "quitadas_por_el_hueco": list(self.quitadas_por_el_hueco)}


@dataclass(frozen=True)
class PropuestaDeParticion:
    plan: SplitPlan | None
    pliegues: AsignacionDePliegues | None
    bloqueos: tuple[Bloqueo, ...] = ()
    limites: tuple[Limite, ...] = ()
    #: 140-C1. `None` si no hay pliegues (`groups_and_time`) o la propuesta no es viable.
    cobertura: CoberturaDePliegues | None = None
    #: 140-C2. SOLO en `temporal`: el instante (segundos) de cada fila de desarrollo y de prueba, tal como el generador lo
    #: leyó (formato de fecha detectado una vez, sobre todo el conjunto). El backend lo necesita para ordenar por tiempo lo que
    #: reparte DESPUÉS (el ajuste final con todo el desarrollo, las mitades cronológicas) sin volver a leer la columna: releerla
    #: sobre otras filas o ya tipada podría darle otro formato. `None` en los demás diseños. No cuenta en la igualdad ni en `repr`.
    tiempo_por_id: dict[str, float] | None = field(default=None, compare=False, repr=False)

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
        # El conjunto se construye UNA vez por pliegue. Estaba DENTRO de la
        # condición, así que se reconstruía para cada fila: cuadrático, 80 s
        # de los 180 de una confirmación de 51.000 filas (medido el 2026-09-26).
        de_validacion = set(valida)
        entrena = tuple(o for o in todas if o not in de_validacion)
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

    # `.get`: una fila SIN la columna de unidad (un `DictReader` con la fila corta) no es un `KeyError` del núcleo;
    # llega como `None`, que es una unidad faltante y la para `unidad_vacia` (re-auditoría, IMPORTANTE 3).
    unidad_por_id = ({obs: str(fila.get(unit_id_field)) for obs, fila in internos.items()}
                     if unit_id_field else None)
    # La cohorte externa también lleva su unidad en el plan (`SplitPlan` agrupado exige una por observación).
    unidad_de_los_externos = ({obs: str(todos[obs].get(unit_id_field)) for obs in externos if obs in todos}
                              if unit_id_field else {})
    tiempo_por_id = None
    es_fecha = True
    if split_type in ("temporal", "groups_and_time"):
        # 140-C1: el tiempo como lo trae la gente (texto del CSV, datetime, número) y el hueco como el dict
        # del JSON. Lo que no se entiende es un Bloqueo con sus números, no una excepción de Python.
        if not time_column:
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(
                Bloqueo(clave="falta_columna_de_tiempo", campo="time_column",
                        motivo=motivo("falta_columna_de_tiempo", valor=split_type)),))
        try:
            gap = convertir_gap(gap)
            convertido = convertir_tiempos({obs: fila.get(time_column) for obs, fila in internos.items()},
                                           time_column)
        except Rechazo as rechazo:
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(rechazo.bloqueo(),))
        tiempo_por_id, es_fecha = convertido.valores, convertido.es_fecha
        if (split_type == "temporal" and gap is not None and not es_fecha and gap.magnitud > 0
                and gap.unidad != "seconds"):
            # Auditoría I4: un hueco en días o años sobre una columna que se lee como NÚMERO (`2000`, `20240131`)
            # se aplicaba como segundos y el corte salía con un `cronologia_no_se_pudo_separar` que no decía por
            # qué. Se decide ANTES de cortar. `AAAAMMDD` sigue siendo un número (lo decide `fechas.py`, que leen
            # también la confirmación y la preparación); sin hueco, un tiempo numérico ordena como siempre.
            ejemplos = sorted({str(fila.get(time_column)) for fila in internos.values()})[:3]
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
                clave="hueco_en_tiempo_numerico", campo="gap",
                motivo=motivo("hueco_en_tiempo_numerico", campo=time_column, valor=f"{gap.magnitud:g} {gap.unidad}",
                              ejemplos=", ".join(ejemplos)),
                medida={"columna": time_column, "magnitud_del_hueco": gap.magnitud, "unidad_del_hueco": gap.unidad,
                        "ejemplos": ejemplos}),))
    etiqueta_por_id = ({obs: fila[objetivo] for obs, fila in internos.items()}
                       if objetivo else None)

    if split_type in ("groups", "groups_and_time") and unit_id_field:
        # Auditoría M1: una unidad vacía llegaba a `SplitPlan` y salía un `EsquemaInvalido` («units.0 must be a
        # non-empty string»). Una fila sin sujeto no se puede poner a un lado ni a otro sin adivinar de quién es.
        # Re-auditoría (IMPORTANTE 3): «vacía» es lo que el núcleo ya llama faltante (`_is_null`: `NA`, `N/A`, `nan`,
        # `NULL`, `None`, `?`, `-`, la celda que no está); sin esto «N/A» era un sujeto más, y podía ser el enorme.
        vacias = sum(1 for obs in todos if _unidad_falta(todos[obs].get(unit_id_field)))
        if vacias:
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
                clave="unidad_vacia", campo="unit_id_field",
                motivo=motivo("unidad_vacia", campo=unit_id_field, valor=vacias, n=len(todos)),
                medida={"columna": unit_id_field, "filas_sin_unidad": vacias, "filas": len(todos)}),))

    if split_type == "groups":
        compartidas = sorted(set(unidad_de_los_externos.values()) & set(unidad_por_id.values())) if unidad_por_id else []
        if compartidas:
            # Auditoría M1: el mismo sujeto en la cohorte externa y dentro dejaba a `SplitPlan` lanzar un
            # `EsquemaInvalido`. La cohorte externa mide a sujetos que el modelo no ha visto: si comparte uno, no lo es.
            return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
                clave="cohorte_externa_comparte_unidad", campo="unit_id_field",
                motivo=motivo("cohorte_externa_comparte_unidad", campo=unit_id_field, valor=len(compartidas),
                              ejemplos=", ".join(compartidas[:3])),
                medida={"columna": unit_id_field, "unidades_compartidas": len(compartidas),
                        "ejemplos": compartidas[:3]}),))
        n_grupos = len(set(unidad_por_id.values())) if unidad_por_id else 0
        if n_grupos < MINIMO_GRUPOS:
            return _grupos_insuficientes(n_grupos, len(internos), unit_id_field)
        return _proponer_groups(plan_id, internos, externos, unidad_por_id, observation_id_field,
                                test_fraction, seed, folds, repeats, etiqueta_por_id, unit_id_field,
                                unidad_de_los_externos, columna_objetivo=objetivo)

    if split_type == "iid":
        return _proponer_iid(plan_id, internos, externos, observation_id_field,
                             test_fraction, seed, folds, repeats, etiqueta_por_id)

    if split_type == "temporal":
        return _proponer_temporal(plan_id, internos, externos, tiempo_por_id, observation_id_field,
                                  test_fraction, gap, time_column, folds=folds, repeats=repeats,
                                  seed=seed, etiqueta_por_id=etiqueta_por_id, es_fecha=es_fecha,
                                  columna_objetivo=objetivo)

    if split_type == "groups_and_time":
        n_grupos = len(set(unidad_por_id.values())) if unidad_por_id else 0
        if n_grupos < MINIMO_GRUPOS:
            return _grupos_insuficientes(n_grupos, len(internos), unit_id_field)
        return _proponer_groups_and_time(plan_id, internos, externos, unidad_por_id, tiempo_por_id,
                                         observation_id_field, test_fraction, gap,
                                         unit_id_field, time_column)

    raise ValueError(f"split_type desconocido: {split_type!r}")


def _unidad_falta(valor: Any) -> bool:
    """¿Esta celda de la columna de unidad dice «no hay sujeto»? Lo decide `_is_null` (la misma lista con la que el
    núcleo da por faltante el objetivo), sobre el texto: `None` (la celda no está) y `NaN` también."""
    return valor is None or _is_null(str(valor))


def _grupos_insuficientes(n_grupos: int, filas: int, unit_id_field: str | None) -> PropuestaDeParticion:
    """Re-auditoría (M-b): nombraba el ROL («unit_id_field tiene 1 grupo(s)») y no llevaba números. Aquí el núcleo
    tiene la columna; `mitades_de_validacion` también la recibe (`columna_de_unidad`), pero su llamante del backend
    aún no se la pasa: eso es cableado del C2."""
    columna = unit_id_field or "unit_id_field"
    return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
        clave="grupos_insuficientes", campo="unit_id_field",
        motivo=motivo("grupos_insuficientes", campo=columna, valor=n_grupos),
        medida={"columna": columna, "grupos": n_grupos, "grupos_necesarios": MINIMO_GRUPOS, "filas": filas}),))


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
    if not prueba:
        return _sin_prueba(test_fraction, len(ids))
    prueba_set = set(prueba)
    roles = {obs: ("test" if obs in prueba_set else "development") for obs in ids}
    roles.update({obs: "external_test" for obs in externos})

    desarrollo = [obs for obs in ids if obs not in prueba_set]
    pliegues_tupla, limites = _kfold_iid(desarrollo, etiqueta_por_id, folds, repeats, seed)
    plan = _plan_con_roles(plan_id, "iid", roles, observation_id_field,
                           folds=folds, repeats=repeats, seed=seed)
    asignacion = AsignacionDePliegues(folds=pliegues_tupla[-1].pliegue + 1 if pliegues_tupla else folds,
                                      repeats=repeats, seed=seed, pliegues=pliegues_tupla)
    return PropuestaDeParticion(plan=plan, pliegues=asignacion, limites=limites,
                                cobertura=_cobertura(desarrollo, asignacion))


def _sin_prueba(test_fraction: float, filas: int) -> PropuestaDeParticion:
    """Auditoría M1: una prueba reservada vacía (`test_fraction` 0, o tan pequeña que no llega a una fila) llegaba a
    `SplitPlan` y salía un `EsquemaInvalido` que hablaba de `evaluacion_anidada`."""
    return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
        clave="sin_prueba_reservada", campo="test_fraction",
        motivo=motivo("sin_prueba_reservada", opciones=f"{test_fraction:g}", n=filas),
        medida={"fraccion_pedida": test_fraction, "filas": filas}),))


def _cobertura(desarrollo: Sequence[str], asignacion: AsignacionDePliegues,
               quitadas: Sequence[int] = ()) -> CoberturaDePliegues:
    primera = min((p.repeticion for p in asignacion.pliegues), default=0)
    con_validacion = {o for p in asignacion.pliegues if p.repeticion == primera for o in p.valida}
    return CoberturaDePliegues(
        filas_de_desarrollo=len(desarrollo),
        ids_sin_validacion=tuple(sorted(o for o in desarrollo if o not in con_validacion)),
        quitadas_por_el_hueco=tuple(quitadas))


def _proponer_groups(plan_id, internos, externos, unidad_por_id, observation_id_field,
                     test_fraction, seed, folds, repeats, etiqueta_por_id, unit_id_field,
                     unidad_de_los_externos=None, columna_objetivo=None):
    filas_por_unidad: dict[str, list[str]] = {}
    for obs in internos:
        filas_por_unidad.setdefault(unidad_por_id[obs], []).append(obs)
    if test_fraction <= 0:
        return _sin_prueba(test_fraction, len(internos))
    limites_previos: tuple[Limite, ...] = ()
    if etiqueta_por_id:
        clases = {etiqueta_por_id[obs] for obs in internos}
        if len(clases) >= 2:
            return _proponer_groups_estratificado(plan_id, internos, externos, unidad_por_id, filas_por_unidad,
                                                  observation_id_field, test_fraction, seed, folds, repeats,
                                                  etiqueta_por_id, unit_id_field, unidad_de_los_externos or {})
        limites_previos = (Limite(clave="una_sola_clase_sin_guardas", campo="objetivo",
                                  motivo=motivo("una_sola_clase_sin_guardas", campo=columna_objetivo or "objetivo"),
                                  medida={"clases": len(clases)}),)
    grupos = sorted(filas_por_unidad)
    azar = random.Random(f"{seed}:corte")
    azar.shuffle(grupos)

    n_total = len(internos)
    objetivo_prueba = round(n_total * test_fraction)
    if objetivo_prueba == 0:
        return _sin_prueba(test_fraction, n_total)
    minimo_de_grupos = 1 + _MINIMO_GRUPOS_EN_DESARROLLO
    if len(grupos) < minimo_de_grupos:
        # Auditoría I5 (heredado de la 139): con 2 grupos la prueba se lleva uno y a desarrollo le queda uno (o
        # ninguno): pliegues vacíos y una propuesta «viable» que no valida nada.
        pedidos = max(2, folds)
        return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
            clave="grupos_insuficientes_para_prueba_y_pliegues", campo="unit_id_field",
            motivo=motivo("grupos_insuficientes_para_prueba_y_pliegues", campo=unit_id_field, valor=len(grupos),
                          opciones=pedidos, partes=pedidos + 1, minimo=minimo_de_grupos),
            medida={"grupos": len(grupos), "pliegues_pedidos": pedidos,
                    "grupos_necesarios_con_los_pedidos": pedidos + 1,
                    "grupos_necesarios_con_2_pliegues": minimo_de_grupos}),))
    prueba: list[str] = []
    grupos_prueba: list[str] = []
    recortada = False
    for grupo in grupos:
        if len(prueba) >= objetivo_prueba:
            break
        if len(grupos) - len(grupos_prueba) <= _MINIMO_GRUPOS_EN_DESARROLLO:
            # Auditoría I5: el grupo siguiente dejaría a desarrollo con menos de 2 grupos (pliegues vacíos). Solo
            # pasa donde la 139 daba un reparto inservible; lo que funcionaba sale bit a bit igual.
            recortada = True
            break
        if fuera_del_margen(len(prueba) + len(filas_por_unidad[grupo]), n_total, test_fraction, solo_por_arriba=True):
            # Supervisor, 27-10 (la auditoría lo vio en la ruta estratificada; aquí, en regresión, seguía): un
            # sujeto enorme no entra en la prueba si la desborda más del margen; se prueba con los siguientes.
            # El margen es el de la fracción PEDIDA, no el de las filas redondeadas (re-auditoría, M-a): con
            # `round(n·f)·1,5`, n=1254 y f=0,1 daban un tope de 187,5 y la prueba de 188 filas de la 139 (dentro del
            # margen, 188,1) pasaba a 146. Así solo cambia lo que la 139 dejaba fuera del margen.
            continue
        prueba.extend(filas_por_unidad[grupo])
        grupos_prueba.append(grupo)
    if not prueba and not recortada:
        # Ningún grupo cabe en el margen: el más pequeño (y el `Limite` de abajo lo dice).
        menor = min(enumerate(grupos), key=lambda par: (len(filas_por_unidad[par[1]]), par[0]))[1]
        prueba.extend(filas_por_unidad[menor])
        grupos_prueba.append(menor)
    limites_previos += _limite_de_fraccion({g: {"filas": len(filas_por_unidad[g])} for g in grupos},
                                           grupos_prueba, test_fraction, unit_id_field)
    prueba_set = set(prueba)
    if recortada:
        limites_previos += (Limite(
            clave="prueba_recortada_para_dejar_grupos", campo="unit_id_field",
            motivo=motivo("prueba_recortada_para_dejar_grupos", campo=unit_id_field, valor=len(prueba),
                          n=objetivo_prueba, partes=len(grupos), minimo=_MINIMO_GRUPOS_EN_DESARROLLO),
            medida={"grupos": len(grupos), "grupos_de_prueba": len(grupos_prueba),
                    "grupos_en_desarrollo": len(grupos) - len(grupos_prueba), "filas_de_prueba": len(prueba),
                    "filas_de_prueba_pedidas": objetivo_prueba, "filas": n_total}),)

    roles = {obs: ("test" if obs in prueba_set else "development") for obs in internos}
    roles.update({obs: "external_test" for obs in externos})
    units = {**unidad_por_id, **(unidad_de_los_externos or {})}

    desarrollo = [obs for obs in internos if obs not in prueba_set]
    pliegues_tupla, limites = _kfold_groups(desarrollo, unidad_por_id, folds, repeats, seed)
    plan = _plan_con_roles(plan_id, "groups", roles, observation_id_field,
                           unit_id_field=unit_id_field, units=units,
                           folds=folds, repeats=repeats, seed=seed)
    asignacion = AsignacionDePliegues(folds=pliegues_tupla[-1].pliegue + 1 if pliegues_tupla else folds,
                                      repeats=repeats, seed=seed, pliegues=pliegues_tupla)
    return PropuestaDeParticion(plan=plan, pliegues=asignacion, limites=limites_previos + limites,
                                cobertura=_cobertura(desarrollo, asignacion))


def _proponer_groups_estratificado(plan_id, internos, externos, unidad_por_id, filas_por_unidad,
                                   observation_id_field, test_fraction, seed, folds, repeats,
                                   etiqueta_por_id, unit_id_field, unidad_de_los_externos):
    """`groups` con clasificación: la prueba y cada pliegue llevan TODAS las clases, sin partir
    ninguna unidad (140-C1, D2)."""
    try:
        reparto = repartir_grupos_estratificado(
            filas_por_unidad, etiqueta_por_id, folds=folds, test_fraction=test_fraction,
            seed=seed, repeats=repeats, campo=unit_id_field)
    except Rechazo as rechazo:
        return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(rechazo.bloqueo(),))
    prueba_set = {obs for grupo in reparto.grupos_de_prueba for obs in filas_por_unidad[grupo]}
    roles = {obs: ("test" if obs in prueba_set else "development") for obs in internos}
    roles.update({obs: "external_test" for obs in externos})
    desarrollo = [obs for obs in internos if obs not in prueba_set]

    pliegues: list[Pliegue] = []
    for r, cubos in enumerate(reparto.cubos_por_repeticion):
        cubos_de_filas = [[obs for grupo in cubo for obs in filas_por_unidad[grupo]] for cubo in cubos]
        pliegues.extend(_pliegues_desde_cubos(cubos_de_filas, r))
    plan = _plan_con_roles(plan_id, "groups", roles, observation_id_field,
                           unit_id_field=unit_id_field, units={**unidad_por_id, **unidad_de_los_externos},
                           folds=reparto.pliegues_usados, repeats=repeats, seed=seed)
    asignacion = AsignacionDePliegues(folds=reparto.pliegues_usados, repeats=repeats, seed=seed,
                                      pliegues=tuple(pliegues))
    return PropuestaDeParticion(plan=plan, pliegues=asignacion, limites=reparto.limites,
                                cobertura=_cobertura(desarrollo, asignacion))


def _proponer_temporal(plan_id, internos, externos, tiempo_por_id, observation_id_field,
                       test_fraction, gap, time_column, *, folds=5, repeats=1, seed=0,
                       etiqueta_por_id=None, es_fecha=True, columna_objetivo=None):
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

    limites: list[Limite] = []
    clase_por_id = None
    clases: list[Any] = []
    if etiqueta_por_id:
        clases = sorted({etiqueta_por_id[o] for o in internos}, key=str)
        if len(clases) >= 2:
            # D2: la prueba reservada (lo más reciente) no puede quedarse sin una clase en silencio.
            en_prueba = {etiqueta_por_id[o] for o in prueba}
            ausentes = [c for c in clases if c not in en_prueba]
            if ausentes:
                rara = ausentes[0]
                nombre = nombres_de_clases(clases)
                return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(BloqueoConNumeros(
                    clave="prueba_sin_una_clase", campo="objetivo",
                    motivo=motivo("prueba_sin_una_clase", clase=nombre[rara], filas=len(prueba)),
                    medida={"clase": nombre[rara], "filas_de_prueba": len(prueba),
                            "clases_que_faltan": [nombre[c] for c in ausentes],
                            "filas_de_prueba_por_clase": {nombre[c]: sum(1 for o in prueba if etiqueta_por_id[o] == c)
                                                          for c in clases}}),))
            clase_por_id = {o: etiqueta_por_id[o] for o in desarrollo}
        else:
            limites.append(Limite(clave="una_sola_clase_sin_guardas", campo="objetivo",
                                  motivo=motivo("una_sola_clase_sin_guardas", campo=columna_objetivo or "objetivo"),
                                  medida={"clases": len(clases)}))
    if gap is not None and not es_fecha:
        limites.append(Limite(clave="tiempo_numerico_sin_unidad", campo="time_column",
                              motivo=motivo("tiempo_numerico_sin_unidad", campo=time_column),
                              medida={"hueco_segundos": _segundos_de(gap)}))
    if repeats > 1:
        limites.append(Limite(clave="repeticiones_fijadas_a_1", campo="repeats",
                              motivo=motivo("repeticiones_fijadas_a_1", opciones=repeats, valor=1),
                              medida={"repeticiones_pedidas": repeats, "repeticiones_usadas": 1}))
    try:
        ventana, limites_de_la_ventana = elegir_ventana(
            desarrollo, tiempo_por_id, folds=folds, gap=gap, clase_por_id=clase_por_id,
            clases=clases if clase_por_id is not None else None, campo_de_tiempo=time_column,
            tiempo_original={o: internos[o].get(time_column) for o in desarrollo})
    except Rechazo as rechazo:
        return PropuestaDeParticion(plan=None, pliegues=None, bloqueos=(rechazo.bloqueo(),))

    roles = {obs: "development" for obs in desarrollo}
    roles.update({obs: "test" for obs in prueba})
    roles.update({obs: "external_test" for obs in externos})
    # La ventana no baraja nada: la semilla no decide el reparto. Va en el plan igualmente (como en los
    # demás diseños): quien lee `plan.seed` no debe encontrarse un `None`.
    plan = _plan_con_roles(plan_id, "temporal", roles, observation_id_field,
                           time_column=time_column, gap=gap, folds=ventana.folds, repeats=1, seed=seed)
    pliegues = tuple(Pliegue(repeticion=0, pliegue=p, entrena=entrena, valida=valida)
                     for p, (entrena, valida) in enumerate(ventana.pliegues))
    asignacion = AsignacionDePliegues(folds=ventana.folds, repeats=1, seed=seed, pliegues=pliegues)
    return PropuestaDeParticion(plan=plan, pliegues=asignacion,
                                limites=tuple(limites) + limites_de_la_ventana,
                                cobertura=_cobertura(desarrollo, asignacion, ventana.quitadas_por_el_hueco),
                                tiempo_por_id={o: tiempo_por_id[o] for o in (*desarrollo, *prueba)})


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
