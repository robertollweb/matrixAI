# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b-C1 — «por sujeto y por el tiempo» ESTRICTO: ningún sujeto a los dos lados y la prueba, posterior.

**Por qué existe.** El `groups_and_time` publicado (103-C4) pone sujetos ENTEROS a un lado u otro del corte,
ordenados por su observación más tardía. Medido en el 140b-C0: en Parkinson el 90 % de las filas de la prueba son
ANTERIORES al final del desarrollo (78 % con un calendario escalonado), y en un panel (Avocado, lluvia: todas las
unidades acaban el mismo día) no es viable. Prueba «sujetos nuevos», no «hacia delante».

Este módulo es la variante «B1» del C0, **en un módulo aparte y ADITIVO (D1)**: `proponer_particion(split_type=
"groups_and_time")` sigue dando el reparto publicado BIT A BIT (una prueba lo afirma con los datos de Parkinson); quien
quiera el estricto llama a `proponer_particion_estricta`.

**Qué es el reparto.**

- Un CORTE en el tiempo `c`: el cuantil `q` de las filas. Con hueco `g`, el desarrollo es lo anterior a `c − g/2` y la
  prueba, lo posterior a `c + g/2`.
- Los sujetos de PRUEBA se escogen entre los que tienen filas posteriores al corte (los «elegibles»): uno que solo tiene
  visitas antiguas aportaría cero filas a la prueba (medido: 18 de 42 pacientes, con un calendario escalonado). Un
  sujeto de prueba solo juega con sus filas posteriores; los demás, solo con las anteriores. **Ningún sujeto queda en los
  dos lados.** El resto de filas (las anteriores de los sujetos de prueba, las posteriores de los demás, las del hueco)
  NO tienen rol, y el resultado DECLARA cuántas son y qué porcentaje (en Parkinson, con la omisión, el 42 %).
- **Mínimos de la prueba (D2, medidos en el C0): rechazo duro con sus números.** ≥ 5 sujetos, ≥ 50 filas y cada clase en
  ≥ 2 sujetos de prueba. Con menos, la nota es del azar de quién cayó en la prueba.
- **Corte automático (D2).** Por omisión `s` = 0,3 (la cuota de sujetos de prueba, sobre TODOS los sujetos) y `q` = 0,7.
  Si esa prueba no cumple los mínimos, la escalera relaja lo justo —más sujetos de prueba (`s` + 0,1 hasta 0,5) y un
  corte más temprano (`q` − 0,1 hasta 0,4)—, **en orden de lo menos a lo más relajado (entre iguales, antes más sujetos)
  y se queda con el PRIMERO que cumple**. La escalera del C0 buscaba «el mayor desarrollo que cumple los mínimos» y dejaba pruebas de 8 pacientes
  (Parkinson 132 filas, el mínimo que pasaba): aquí lo que se busca es una prueba SUFICIENTE, no el máximo desarrollo.
  `automatico=False` prueba solo lo pedido.
- **Pliegues (D3) = F2q del C0.** Pliegues por sujeto × el último 30 % de las FILAS del desarrollo: cada pliegue valida
  a los sujetos de su cubo solo en ese tramo y entrena con los demás sujetos, antes del tramo (y a más de `gap` de él). El
  tramo se mide en filas y no en tiempo (F2 por *span* dejó validaciones vacías con calendarios largos). Se estratifica
  con el asignador REAL de `groups` (`_cubrir`/`repartir_cubos`), contando por mitad de tiempo: la prueba, con las filas
  posteriores al corte; los pliegues, con las filas del tramo. Si el desarrollo tiene menos de 300 filas, o ni con 2
  pliegues el tramo deja validaciones suficientes (≥ 30 filas, entrenamientos de ≥ 100 y todas las clases), se cae a
  F1 —pliegues por sujeto, sin tiempo— y se DICE.
- **Intervalo (D4).** El método existente de bootstrap por unidad (`cluster_bootstrap_by_unit`, el de `groups`) sobre
  las filas de la prueba. `incertidumbre.py` no se toca. El resultado declara el método, los sujetos de prueba y que el
  intervalo es **condicional a esta ventana de tiempo**: no incluye cambios entre periodos (con deriva compartida la
  cobertura marginal se hunde, medido en el C0). Con pocos sujetos de prueba el percentil infracubre; la cobertura NO se
  redacta aquí: se deja el número de sujetos de prueba para que quien la enseñe lo use.

**Lo que NO hace.** No quita de `repartir_cubos` su tope de nodos: en conjuntos enormes un «no se ha encontrado»
del cubrimiento no prueba que no exista (como en `groups`). No elige el corte por fecha puesta por la persona (D2,
«avanzado»). La cola de parada temprana y las mitades de `valida` son del backend (C2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from matrixai.estudio import Horizonte
from matrixai.training.diagnostico import Bloqueo, Limite
from matrixai.training.particion_por_diseno import (
    AsignacionDePliegues, Pliegue, PropuestaDeParticion, _cobertura, _grupos_insuficientes,
    _kfold_groups, _pliegues_desde_cubos, _plan_con_roles, _sin_prueba, _unidad_falta)
from matrixai.training.particion_por_diseno_comun import BloqueoConNumeros, Rechazo, nombres_de_clases
from matrixai.training.particion_por_diseno_grupos import _cubrir, repartir_cubos, repartir_grupos_estratificado
from matrixai.training.particion_por_diseno_textos import motivo
from matrixai.training.particion_por_diseno_tiempo import convertir_gap, convertir_tiempos, segundos_de

__all__ = [
    "CUANTIL_DEL_CORTE_POR_OMISION", "DOMINANCIA_DE_UNA_UNIDAD", "FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO",
    "FRACCION_DE_LA_VENTANA", "FRACCION_DE_SUJETOS_POR_OMISION", "IntervaloDeclarado",
    "MINIMO_FILAS_DE_ENTRENAMIENTO", "MINIMO_FILAS_DE_PRUEBA", "MINIMO_FILAS_DE_VALIDACION",
    "MINIMO_UNIDADES_DE_PRUEBA", "MINIMO_UNIDADES_POR_CLASE", "PropuestaEstricta", "RepartoEstricto",
    "escalera_del_corte", "proponer_particion_estricta", "repartir_estricto",
]

#: Los mínimos de la prueba (D2). Medidos en el 140b-C0 (750 celdas sintéticas + 5 conjuntos reales): los tres tienen
#: dientes y el reparto automático los cumple en los reales (0 rechazos en 20/20 semillas).
MINIMO_UNIDADES_DE_PRUEBA = 5
MINIMO_FILAS_DE_PRUEBA = 50
MINIMO_UNIDADES_POR_CLASE = 2

#: La omisión (C0, P4): la regla del protocolo —cumplimiento ≥ 0,90 en cada forma de referencia y ≥ 150 filas de
#: prueba— dio s = 0,3 y q = 0,7 (segunda: 0,3 y 0,6).
FRACCION_DE_SUJETOS_POR_OMISION = 0.3
CUANTIL_DEL_CORTE_POR_OMISION = 0.7
_PASO_DE_LA_ESCALERA = 0.1
_FRACCION_MAXIMA_DE_SUJETOS = 0.5
_CUANTIL_MINIMO = 0.4
_PELDANOS_DE_SUJETOS = 3

#: Pliegues (D3, F2q del C0): el tramo de validación es el último 30 % de las FILAS del desarrollo, y la frontera de
#: viabilidad con la que F2q cae a F1 («desarrollo de < ≈ 300–400 filas»).
FRACCION_DE_LA_VENTANA = 0.3
FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO = 300
MINIMO_FILAS_DE_VALIDACION = 30
MINIMO_FILAS_DE_ENTRENAMIENTO = 100
_MINIMO_PLIEGUES = 2
_MINIMO_UNIDADES_DE_DESARROLLO = 2

#: Un sujeto con más de esta parte de las filas de la prueba la domina: el bootstrap por unidad depende casi de él.
DOMINANCIA_DE_UNA_UNIDAD = 0.5


# ---------------------------------------------------------------------------
# Lo que el reparto declara
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntervaloDeclarado:
    """Qué intervalo corresponde a esta prueba (D4), sin redactar su cobertura.

    El método es el de `groups` de `incertidumbre.py` (`diseno_de_remuestreo` es el valor que hay que pasarle a
    `intervalo(diseno=…)`): `groups_and_time` sigue fuera de `DISENOS_SOPORTADOS` a propósito. `unidades_de_prueba` es
    el número que decide cuánto se parece el intervalo a su nivel nominal (medido en el C0: se infracubre con pocos)."""

    metodo: str
    diseno_de_remuestreo: str
    unidad_de_remuestreo: str
    #: Siempre `True`: el intervalo dice cuánto varía la nota por elegir otros sujetos EN ESTA VENTANA, no cuánto
    #: rendirá en el futuro (con deriva compartida entre periodos, la cobertura marginal se hunde).
    condicional_a_la_ventana: bool
    unidades_de_prueba: int
    ventana_de_prueba: tuple[float, float]

    def a_json(self) -> dict[str, Any]:
        return {"metodo": self.metodo, "diseno_de_remuestreo": self.diseno_de_remuestreo,
                "unidad_de_remuestreo": self.unidad_de_remuestreo,
                "condicional_a_la_ventana": self.condicional_a_la_ventana,
                "unidades_de_prueba": self.unidades_de_prueba,
                "ventana_de_prueba": list(self.ventana_de_prueba)}


@dataclass(frozen=True)
class RepartoEstricto:
    """El reparto con todo lo que hay que declarar de él."""

    desarrollo: tuple[str, ...]
    prueba: tuple[str, ...]
    #: Las filas sin rol: ni desarrollo ni prueba (ver la cabecera).
    descartadas: tuple[str, ...]
    unidades_de_desarrollo: tuple[str, ...]
    unidades_de_prueba: tuple[str, ...]
    #: El corte, en segundos (o en la unidad del número, si el tiempo es numérico), y cómo se lee.
    corte: float
    corte_legible: str
    hueco_segundos: float
    #: Lo pedido y lo usado: si difieren, la escalera relajó el corte (y hay un `Limite`).
    fraccion_de_sujetos_pedida: float
    cuantil_pedido: float
    fraccion_de_sujetos: float
    cuantil: float
    #: Cuántos cortes se probaron hasta dar con este (1 = el pedido).
    cortes_probados: int
    filas_descartadas: int
    porcentaje_descartado: float
    #: `"por_sujeto_y_ultimo_tramo"` (F2q) o `"por_sujeto"` (F1, declarado por un `Limite`).
    tipo_de_pliegues: str
    pliegues: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    #: El inicio del tramo de validación de F2q (mismos segundos que el corte); `None` en F1.
    inicio_del_tramo: float | None
    quitadas_por_el_hueco: tuple[int, ...]
    limites: tuple[Limite, ...]
    intervalo: IntervaloDeclarado

    def a_json(self) -> dict[str, Any]:
        """La declaración (sin las listas de filas): lo que el manifiesto y la ficha tienen que decir."""
        return {
            "modo": "grupos_y_tiempo_estricto",
            "corte": self.corte, "corte_legible": self.corte_legible, "hueco_segundos": self.hueco_segundos,
            "fraccion_de_sujetos_pedida": self.fraccion_de_sujetos_pedida, "cuantil_pedido": self.cuantil_pedido,
            "fraccion_de_sujetos": self.fraccion_de_sujetos, "cuantil": self.cuantil,
            "cortes_probados": self.cortes_probados,
            "filas_de_desarrollo": len(self.desarrollo), "filas_de_prueba": len(self.prueba),
            "filas_descartadas": self.filas_descartadas, "porcentaje_descartado": self.porcentaje_descartado,
            "unidades_de_desarrollo": len(self.unidades_de_desarrollo),
            "unidades_de_prueba": list(self.unidades_de_prueba),
            "tipo_de_pliegues": self.tipo_de_pliegues, "pliegues": len(self.pliegues),
            "inicio_del_tramo": self.inicio_del_tramo,
            "intervalo": self.intervalo.a_json(),
        }


@dataclass(frozen=True)
class PropuestaEstricta(PropuestaDeParticion):
    """Una `PropuestaDeParticion` (el mismo contrato: plan, pliegues, bloqueos, límites, cobertura) más el reparto
    declarado. `reparto` es `None` si la propuesta no es viable. No cuenta en la igualdad ni en `repr`."""

    reparto: RepartoEstricto | None = field(default=None, compare=False, repr=False)


# ---------------------------------------------------------------------------
# Piezas pequeñas
# ---------------------------------------------------------------------------

def _cuantil(ordenados: Sequence[float], q: float) -> float:
    """El cuantil `q` con interpolación lineal (el `np.quantile` por omisión; el núcleo no depende de numpy)."""
    posicion = (len(ordenados) - 1) * q
    bajo = math.floor(posicion)
    alto = min(bajo + 1, len(ordenados) - 1)
    return ordenados[bajo] + (ordenados[alto] - ordenados[bajo]) * (posicion - bajo)


def _numero(x: float) -> str:
    return f"{x:g}"


def _porcentaje(x: float) -> str:
    return f"{100 * x:g}"


def escalera_del_corte(fraccion: float, cuantil: float, automatico: bool = True) -> list[tuple[float, float]]:
    """Los cortes que se prueban, EN ORDEN. El primero es el pedido; después, de lo menos a lo más relajado (más
    sujetos de prueba y/o un corte más temprano). **No ordena por desarrollo**: el primero que cumple los mínimos gana."""
    if not automatico:
        return [(fraccion, cuantil)]
    sujetos = [round(fraccion + _PASO_DE_LA_ESCALERA * i, 4) for i in range(_PELDANOS_DE_SUJETOS)]
    sujetos = [s for i, s in enumerate(sujetos) if i == 0 or s <= _FRACCION_MAXIMA_DE_SUJETOS + 1e-9]
    cuantiles = [cuantil]
    i = 1
    while round(cuantil - _PASO_DE_LA_ESCALERA * i, 4) >= _CUANTIL_MINIMO - 1e-9:
        cuantiles.append(round(cuantil - _PASO_DE_LA_ESCALERA * i, 4))
        i += 1
    parejas = [(i_s, i_q) for i_s in range(len(sujetos)) for i_q in range(len(cuantiles))]
    # Entre dos peldaños igual de relajados, antes el de MÁS sujetos: la cobertura del intervalo mejora con el número de
    # sujetos de prueba, no con el de filas (C0), y un corte más temprano descarta más.
    parejas.sort(key=lambda p: (p[0] + p[1], p[1], p[0]))
    return [(sujetos[i_s], cuantiles[i_q]) for i_s, i_q in parejas]


def _legible(segundos: float, formato: str | None, es_fecha: bool) -> str:
    """El corte como lo escribiría la persona: una fecha (si el tiempo se leyó de un texto con formato) o el número."""
    if formato is not None:
        momento = datetime(1970, 1, 1) + timedelta(seconds=segundos)
        con_hora = any(d in formato for d in ("%H", "%I", "%M", "%S", "%T", "%X", "%c"))
        return momento.isoformat(sep=" ", timespec="seconds") if con_hora else momento.date().isoformat()
    return _numero(segundos)


# ---------------------------------------------------------------------------
# Un corte: la prueba y el desarrollo
# ---------------------------------------------------------------------------

@dataclass
class _Intento:
    fraccion: float
    cuantil: float
    corte: float
    elegibles: int
    unidades_de_prueba: tuple[str, ...] = ()
    prueba: tuple[str, ...] = ()
    desarrollo: tuple[str, ...] = ()
    #: Qué mínimo no cumple (`None` si cumple todo): unidades, filas, clase, sin_elegibles, desarrollo_sin_clase,
    #: desarrollo_vacio.
    fallo: str | None = None
    #: Para el mensaje: por clase, cuántos sujetos de prueba (o, si no hubo reparto, cuántos elegibles) la traen.
    sujetos_por_clase: dict[Any, int] = field(default_factory=dict)
    clase_que_falta: Any = None
    unidades_de_desarrollo: int = 0

    def calidad(self) -> tuple[int, int]:
        return (len(self.unidades_de_prueba), len(self.prueba))


def _cuenta_de_clases(filas: Sequence[str], clase_de) -> dict[Any, int]:
    cuenta: dict[Any, int] = {}
    for o in filas:
        cuenta[clase_de(o)] = cuenta.get(clase_de(o), 0) + 1
    return cuenta


def _probar_un_corte(filas_de: Mapping[str, Sequence[str]], tiempo: Mapping[str, float], ordenados: Sequence[float],
                     clase_de, clases: Sequence[Any], hueco_s: float, fraccion: float, cuantil: float,
                     seed: int) -> _Intento:
    """Un corte `(s, q)`: elige los sujetos de prueba entre los elegibles y mira los mínimos."""
    corte = _cuantil(ordenados, cuantil)
    medio = hueco_s / 2.0
    despues = corte + medio
    antes = corte - medio
    posteriores = {u: [o for o in filas if tiempo[o] >= despues] for u, filas in filas_de.items()}
    posteriores = {u: filas for u, filas in posteriores.items() if filas}
    intento = _Intento(fraccion, cuantil, corte, len(posteriores))
    if not posteriores:
        intento.fallo = "sin_elegibles"
        return intento

    n_prueba = max(1, round(len(filas_de) * fraccion))
    clasifica = clases != [None]
    if n_prueba >= len(posteriores):
        elegidos = sorted(posteriores)
    else:
        # El asignador REAL de `groups`, con las cuentas de la mitad POSTERIOR al corte (lo que un sujeto aporta si es de
        # prueba). Primero se cubren las clases DOS veces (dos cubos disjuntos con todas las clases → cada clase en ≥ 2
        # sujetos de prueba), después se reparte el resto hacia la parte de filas que le toca a la prueba.
        cuentas = {u: _cuenta_de_clases(filas, clase_de) for u, filas in posteriores.items()}
        total = {c: sum(cuentas[u].get(c, 0) for u in cuentas) for c in clases}
        parte = n_prueba / len(posteriores)
        azar = random.Random(f"{seed}:estricto:prueba:{fraccion}:{cuantil}")
        capacidad = sum(total.values()) * parte
        cubierta = _cubrir(cuentas, clases, MINIMO_UNIDADES_POR_CLASE, azar,
                           capacidades=[capacidad / MINIMO_UNIDADES_POR_CLASE] * MINIMO_UNIDADES_POR_CLASE)
        if cubierta is None:
            intento.fallo = "clase"
            intento.sujetos_por_clase = {c: sum(1 for u in cuentas if cuentas[u].get(c)) for c in clases}
            intento.clase_que_falta = min(clases, key=lambda c: (intento.sujetos_por_clase[c], str(c)))
            return intento
        cubos = repartir_cubos(
            cuentas, clases, [{c: total[c] * parte for c in clases}, {c: total[c] * (1 - parte) for c in clases}],
            azar, cubrir=False, previos=[[u for cubo in cubierta for u in cubo], []])
        elegidos = sorted(cubos[0]) if cubos else []

    prueba = [o for u in elegidos for o in posteriores[u]]
    intento.unidades_de_prueba = tuple(elegidos)
    intento.prueba = tuple(sorted(prueba, key=lambda o: (tiempo[o], o)))
    if len(elegidos) < MINIMO_UNIDADES_DE_PRUEBA:
        intento.fallo = "unidades"
        return intento
    if len(prueba) < MINIMO_FILAS_DE_PRUEBA:
        intento.fallo = "filas"
        return intento
    if clasifica:
        por_clase = {c: len({u for u in elegidos if any(clase_de(o) == c for o in posteriores[u])}) for c in clases}
        intento.sujetos_por_clase = por_clase
        if min(por_clase.values()) < MINIMO_UNIDADES_POR_CLASE:
            intento.fallo = "clase"
            intento.clase_que_falta = min(clases, key=lambda c: (por_clase[c], str(c)))
            return intento

    en_la_prueba = set(elegidos)
    desarrollo = [o for u, filas in filas_de.items() if u not in en_la_prueba for o in filas if tiempo[o] < antes]
    intento.desarrollo = tuple(sorted(desarrollo, key=lambda o: (tiempo[o], o)))
    unidades = {u for u, filas in filas_de.items() if u not in en_la_prueba and any(tiempo[o] < antes for o in filas)}
    intento.unidades_de_desarrollo = len(unidades)
    if len(unidades) < _MINIMO_UNIDADES_DE_DESARROLLO:
        intento.fallo = "desarrollo_vacio"
        return intento
    if clasifica:
        presentes = _cuenta_de_clases(desarrollo, clase_de)
        ausentes = [c for c in clases if c not in presentes]
        if ausentes:
            intento.fallo = "desarrollo_sin_clase"
            intento.clase_que_falta = ausentes[0]
    return intento


# ---------------------------------------------------------------------------
# Los pliegues
# ---------------------------------------------------------------------------

def _pliegues_con_tiempo(desarrollo: Sequence[str], unidad_de: Mapping[str, str], tiempo: Mapping[str, float],
                         clase_de, clases: Sequence[Any], hueco_s: float, pedidos: int, seed: int,
                         ) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], int, list[int], float] | str:
    """F2q. Devuelve `(pliegues, usados, quitadas por el hueco, inicio del tramo)` o el motivo por el que no cabe
    (`"pocas_filas"` o `"ventana"`)."""
    if len(desarrollo) < FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO:
        return "pocas_filas"
    inicio = _cuantil(sorted(tiempo[o] for o in desarrollo), 1 - FRACCION_DE_LA_VENTANA)
    de_cada_unidad: dict[str, list[str]] = {}
    for o in desarrollo:
        de_cada_unidad.setdefault(unidad_de[o], []).append(o)
    en_el_tramo = {u: [o for o in filas if tiempo[o] >= inicio] for u, filas in de_cada_unidad.items()}
    con_tramo = {u: filas for u, filas in en_el_tramo.items() if filas}
    cuentas = {u: _cuenta_de_clases(filas, clase_de) for u, filas in con_tramo.items()}
    total = {c: sum(cuentas[u].get(c, 0) for u in cuentas) for c in clases}
    for k in range(max(_MINIMO_PLIEGUES, pedidos), _MINIMO_PLIEGUES - 1, -1):
        if len(con_tramo) < k:
            continue
        azar = random.Random(f"{seed}:estricto:pliegues:{k}")
        cubos = repartir_cubos(cuentas, clases, [{c: total[c] / k for c in clases} for _ in range(k)], azar)
        if cubos is None:
            continue
        pliegues: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        quitadas: list[int] = []
        for cubo in sorted(cubos, key=lambda c: min(c) if c else ""):
            propias = set(cubo)
            valida = sorted((o for u in cubo for o in con_tramo[u]), key=lambda o: (tiempo[o], o))
            otras = [o for u, filas in de_cada_unidad.items() if u not in propias for o in filas if tiempo[o] < inicio]
            entrena = sorted((o for o in otras if inicio - tiempo[o] > hueco_s), key=lambda o: (tiempo[o], o))
            quitadas.append(len(otras) - len(entrena))
            pliegues.append((tuple(entrena), tuple(valida)))
        if _son_viables(pliegues, clase_de, clases):
            return pliegues, k, quitadas, inicio
    return "ventana"


def _son_viables(pliegues, clase_de, clases: Sequence[Any]) -> bool:
    for entrena, valida in pliegues:
        if len(valida) < MINIMO_FILAS_DE_VALIDACION or len(entrena) < MINIMO_FILAS_DE_ENTRENAMIENTO:
            return False
        if clases != [None] and any(len(_cuenta_de_clases(lado, clase_de)) < len(clases) for lado in (entrena, valida)):
            return False
    return True


def _pliegues_por_sujeto(desarrollo: Sequence[str], unidad_de: Mapping[str, str], clase_de, clases: Sequence[Any],
                         pedidos: int, seed: int, columna_de_unidad: str,
                         ) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], int, tuple[Limite, ...]]:
    """F1: pliegues por sujeto, sin mirar el tiempo. Estratificados con el asignador de `groups` si hay clases."""
    de_cada_unidad: dict[str, list[str]] = {}
    for o in desarrollo:
        de_cada_unidad.setdefault(unidad_de[o], []).append(o)
    if clases != [None]:
        reparto = repartir_grupos_estratificado(
            de_cada_unidad, {o: clase_de(o) for o in desarrollo}, folds=pedidos, test_fraction=0.0, seed=seed,
            repeats=1, campo=columna_de_unidad)
        cubos_de_filas = [[o for u in cubo for o in de_cada_unidad[u]] for cubo in reparto.cubos_por_repeticion[0]]
        hechos = _pliegues_desde_cubos(cubos_de_filas, 0)
        usados, limites = reparto.pliegues_usados, reparto.limites
    else:
        hechos_, limites = _kfold_groups(list(desarrollo), unidad_de, pedidos, 1, seed)
        hechos = list(hechos_)
        usados = hechos[-1].pliegue + 1 if hechos else pedidos
    return [(tuple(p.entrena), tuple(p.valida)) for p in hechos], usados, limites


# ---------------------------------------------------------------------------
# El reparto
# ---------------------------------------------------------------------------

def repartir_estricto(ids: Sequence[str], unidad_por_id: Mapping[str, str], tiempo_por_id: Mapping[str, float], *,
                      etiqueta_por_id: Mapping[str, Any] | None = None, gap: Horizonte | None = None,
                      fraccion_de_sujetos: float = FRACCION_DE_SUJETOS_POR_OMISION,
                      cuantil_del_corte: float = CUANTIL_DEL_CORTE_POR_OMISION, automatico: bool = True,
                      folds: int = 5, seed: int = 0, columna_de_unidad: str = "unit_id_field",
                      columna_de_tiempo: str = "time_column", tiempo_original: Mapping[str, Any] | None = None,
                      formato_de_fecha: str | None = None, es_fecha: bool = True) -> RepartoEstricto:
    """El reparto estricto de `ids` (internos, sin la cohorte externa). Lanza `Rechazo` con los números si no se
    puede (mínimos de la prueba, desarrollo sin clase, pliegues sin reparto). Determinista: no depende del orden de
    `ids`."""
    if not 0 < fraccion_de_sujetos < 1:
        raise ValueError(f"fraccion_de_sujetos debe estar en (0, 1): {fraccion_de_sujetos!r}")
    if not 0 < cuantil_del_corte < 1:
        raise ValueError(f"cuantil_del_corte debe estar en (0, 1): {cuantil_del_corte!r}")
    ids = sorted(ids)
    hueco_s = segundos_de(gap) if gap is not None else 0.0
    filas_de: dict[str, list[str]] = {}
    for o in ids:
        filas_de.setdefault(unidad_por_id[o], []).append(o)
    ordenados = sorted(tiempo_por_id[o] for o in ids)

    limites: list[Limite] = []
    clase_de = lambda o: None    # noqa: E731  (sin clases que repartir, todas las filas son de «una clase»: se cuentan filas)
    clases: list[Any] = [None]
    nombre: dict[Any, str] = {}
    if etiqueta_por_id:
        todas = sorted({etiqueta_por_id[o] for o in ids}, key=str)
        if len(todas) >= 2:
            clases, nombre = todas, nombres_de_clases(todas)
            clase_de = lambda o: etiqueta_por_id[o]    # noqa: E731
        else:
            limites.append(Limite(clave="una_sola_clase_sin_guardas", campo="objetivo",
                                  motivo=motivo("una_sola_clase_sin_guardas", campo="objetivo"),
                                  medida={"clases": len(todas)}))

    escalera = escalera_del_corte(fraccion_de_sujetos, cuantil_del_corte, automatico)
    intentos: list[_Intento] = []
    elegido: _Intento | None = None
    for fraccion, cuantil in escalera:
        intento = _probar_un_corte(filas_de, tiempo_por_id, ordenados, clase_de, clases, hueco_s, fraccion, cuantil, seed)
        intentos.append(intento)
        if intento.fallo is None:
            elegido = intento
            break
    if elegido is None:
        raise _rechazo(intentos, ids, clases, nombre, hueco_s, columna_de_unidad, columna_de_tiempo, tiempo_original,
                       tiempo_por_id, formato_de_fecha, es_fecha)
    return _armar(elegido, intentos, ids, filas_de, unidad_por_id, tiempo_por_id, clase_de, clases, nombre, hueco_s,
                  limites, fraccion_de_sujetos, cuantil_del_corte, folds, seed, columna_de_unidad,
                  formato_de_fecha, es_fecha)


def _armar(elegido: _Intento, intentos: Sequence[_Intento], ids, filas_de, unidad_por_id, tiempo, clase_de, clases,
           nombre, hueco_s, limites: list[Limite], fraccion_pedida, cuantil_pedido, folds, seed, columna_de_unidad,
           formato, es_fecha) -> RepartoEstricto:
    desarrollo, prueba = elegido.desarrollo, elegido.prueba
    con_rol = set(desarrollo) | set(prueba)
    descartadas = tuple(o for o in ids if o not in con_rol)
    porcentaje = round(100 * len(descartadas) / len(ids), 1)
    unidades_de_desarrollo = tuple(sorted({unidad_por_id[o] for o in desarrollo}))

    limites.append(Limite(
        clave="estricto_filas_descartadas", campo="time_column",
        motivo=motivo("estricto_filas_descartadas", valor=len(descartadas), n=len(ids), porcentaje=f"{porcentaje:g}"),
        medida={"filas_descartadas": len(descartadas), "filas": len(ids), "porcentaje_descartado": porcentaje,
                "filas_de_desarrollo": len(desarrollo), "filas_de_prueba": len(prueba)}))
    if (elegido.fraccion, elegido.cuantil) != (fraccion_pedida, cuantil_pedido):
        limites.append(Limite(
            clave="estricto_corte_relajado", campo="test_fraction",
            motivo=motivo("estricto_corte_relajado", pedida=_porcentaje(fraccion_pedida),
                          cuantil_pedido=_numero(cuantil_pedido), usada=_porcentaje(elegido.fraccion),
                          cuantil_usado=_numero(elegido.cuantil)),
            medida={"fraccion_pedida": fraccion_pedida, "cuantil_pedido": cuantil_pedido,
                    "fraccion_usada": elegido.fraccion, "cuantil_usado": elegido.cuantil,
                    "cortes_probados": len(intentos)}))
    por_unidad = _cuenta_de_clases(prueba, lambda o: unidad_por_id[o])
    dominante = max(por_unidad, key=lambda u: (por_unidad[u], u))
    if por_unidad[dominante] / len(prueba) > DOMINANCIA_DE_UNA_UNIDAD:
        limites.append(Limite(
            clave="estricto_unidad_dominante_en_la_prueba", campo="unit_id_field",
            motivo=motivo("estricto_unidad_dominante_en_la_prueba", dominante=dominante, unidad=columna_de_unidad,
                          porcentaje=f"{100 * por_unidad[dominante] / len(prueba):.1f}"),
            medida={"unidad": dominante, "filas_de_la_unidad": por_unidad[dominante], "filas_de_prueba": len(prueba),
                    "umbral": DOMINANCIA_DE_UNA_UNIDAD}))

    unidad_de = {o: unidad_por_id[o] for o in desarrollo}
    hecho = _pliegues_con_tiempo(desarrollo, unidad_de, tiempo, clase_de, clases, hueco_s, folds, seed)
    if isinstance(hecho, str):
        pliegues, usados, limites_f1 = _pliegues_por_sujeto(desarrollo, unidad_de, clase_de, clases, folds, seed,
                                                            columna_de_unidad)
        pliegues = [(tuple(sorted(e, key=lambda o: (tiempo[o], o))), tuple(sorted(v, key=lambda o: (tiempo[o], o))))
                    for e, v in pliegues]
        clave = ("estricto_pliegues_por_sujeto_por_pocas_filas" if hecho == "pocas_filas"
                 else "estricto_pliegues_por_sujeto_por_ventana_vacia")
        textos = ({"filas": len(desarrollo), "minimo": FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO} if hecho == "pocas_filas"
                  else {"fraccion": _porcentaje(FRACCION_DE_LA_VENTANA), "pliegues": _MINIMO_PLIEGUES,
                        "minimo": MINIMO_FILAS_DE_VALIDACION})
        limites.append(Limite(clave=clave, campo="folds", motivo=motivo(clave, **textos),
                              medida={"filas_de_desarrollo": len(desarrollo),
                                      "filas_minimas_con_tiempo": FILAS_MINIMAS_PARA_PLIEGUES_CON_TIEMPO,
                                      "fraccion_de_la_ventana": FRACCION_DE_LA_VENTANA,
                                      "minimo_de_filas_de_validacion": MINIMO_FILAS_DE_VALIDACION}))
        limites.extend(limites_f1)
        tipo, inicio, quitadas = "por_sujeto", None, ()
    else:
        pliegues, usados, quitadas_, inicio = hecho
        quitadas = tuple(quitadas_)
        if usados < max(_MINIMO_PLIEGUES, folds):
            limites.append(Limite(
                clave="estricto_pliegues_reducidos", campo="folds",
                motivo=motivo("estricto_pliegues_reducidos", pedidos=max(_MINIMO_PLIEGUES, folds), usados=usados,
                              minimo=MINIMO_FILAS_DE_VALIDACION),
                medida={"pliegues_pedidos": max(_MINIMO_PLIEGUES, folds), "pliegues_usados": usados}))
        tipo = "por_sujeto_y_ultimo_tramo"

    desde, hasta = tiempo[prueba[0]], tiempo[prueba[-1]]
    intervalo = IntervaloDeclarado(
        metodo="cluster_bootstrap_by_unit", diseno_de_remuestreo="groups", unidad_de_remuestreo="unit",
        condicional_a_la_ventana=True, unidades_de_prueba=len(elegido.unidades_de_prueba),
        ventana_de_prueba=(desde, hasta))
    return RepartoEstricto(
        desarrollo=desarrollo, prueba=prueba, descartadas=descartadas, unidades_de_desarrollo=unidades_de_desarrollo,
        unidades_de_prueba=elegido.unidades_de_prueba, corte=elegido.corte,
        corte_legible=_legible(elegido.corte, formato, es_fecha), hueco_segundos=hueco_s,
        fraccion_de_sujetos_pedida=fraccion_pedida, cuantil_pedido=cuantil_pedido,
        fraccion_de_sujetos=elegido.fraccion, cuantil=elegido.cuantil, cortes_probados=len(intentos),
        filas_descartadas=len(descartadas), porcentaje_descartado=porcentaje, tipo_de_pliegues=tipo,
        pliegues=tuple(pliegues), inicio_del_tramo=inicio, quitadas_por_el_hueco=tuple(quitadas),
        limites=tuple(limites), intervalo=intervalo)


def _rechazo(intentos: Sequence[_Intento], ids, clases, nombre, hueco_s, columna_de_unidad, columna_de_tiempo,
             tiempo_original, tiempo, formato, es_fecha) -> Rechazo:
    """El rechazo con los números del MEJOR intento (el de más sujetos y filas de prueba)."""
    mejor = max(intentos, key=lambda i: (i.calidad(), -intentos.index(i)))
    base = {"unidad": columna_de_unidad, "columna_de_tiempo": columna_de_tiempo, "cortes_probados": len(intentos),
            "fraccion_de_sujetos": mejor.fraccion, "cuantil": mejor.cuantil, "corte": mejor.corte,
            "hueco_segundos": hueco_s, "sujetos_elegibles": mejor.elegibles}
    n = len(intentos)
    if mejor.fallo == "sin_elegibles":
        ultimo = max(ids, key=lambda o: (tiempo[o], o))
        como_venia = str(tiempo_original[ultimo]) if tiempo_original is not None else _numero(tiempo[ultimo])
        hueco = f"{_numero(hueco_s)} s" if hueco_s else "0"
        return Rechazo("estricto_sin_sujetos_con_filas_posteriores", "time_column",
                       {**base, "ultimo_tiempo": como_venia}, tiempo=columna_de_tiempo,
                       corte=_legible(mejor.corte, formato, es_fecha), hueco=hueco, unidad=columna_de_unidad,
                       ultimo=como_venia)
    if mejor.fallo == "unidades":
        return Rechazo("estricto_prueba_con_pocas_unidades", "test_fraction",
                       {**base, "unidades_de_prueba": len(mejor.unidades_de_prueba),
                        "filas_de_prueba": len(mejor.prueba), "minimo": MINIMO_UNIDADES_DE_PRUEBA},
                       valor=len(mejor.unidades_de_prueba), unidad=columna_de_unidad,
                       minimo=MINIMO_UNIDADES_DE_PRUEBA, n=n, filas=len(mejor.prueba), elegibles=mejor.elegibles)
    if mejor.fallo == "filas":
        return Rechazo("estricto_prueba_con_pocas_filas", "test_fraction",
                       {**base, "unidades_de_prueba": len(mejor.unidades_de_prueba),
                        "filas_de_prueba": len(mejor.prueba), "minimo": MINIMO_FILAS_DE_PRUEBA},
                       valor=len(mejor.prueba), partes=len(mejor.unidades_de_prueba), unidad=columna_de_unidad,
                       minimo=MINIMO_FILAS_DE_PRUEBA, n=n)
    if mejor.fallo == "clase":
        clase = mejor.clase_que_falta
        return Rechazo("estricto_clase_en_pocos_sujetos_de_prueba", "objetivo",
                       {**base, "clase": nombre[clase], "minimo": MINIMO_UNIDADES_POR_CLASE,
                        "sujetos_de_prueba_por_clase": {nombre[c]: v for c, v in mejor.sujetos_por_clase.items()},
                        "unidades_de_prueba": len(mejor.unidades_de_prueba)},
                       clase=nombre[clase], valor=mejor.sujetos_por_clase.get(clase, 0), unidad=columna_de_unidad,
                       minimo=MINIMO_UNIDADES_POR_CLASE, partes=mejor.sujetos_por_clase.get(clase, 0), n=n)
    if mejor.fallo == "desarrollo_sin_clase":
        clase = mejor.clase_que_falta
        return Rechazo("estricto_desarrollo_sin_clase", "objetivo",
                       {**base, "clase": nombre[clase], "filas_de_desarrollo": len(mejor.desarrollo)},
                       clase=nombre[clase], filas=len(mejor.desarrollo))
    return Rechazo("estricto_desarrollo_vacio", "time_column",
                   {**base, "filas_de_desarrollo": len(mejor.desarrollo),
                    "unidades_de_desarrollo": mejor.unidades_de_desarrollo, "minimo": _MINIMO_UNIDADES_DE_DESARROLLO},
                   filas=len(mejor.desarrollo), valor=mejor.unidades_de_desarrollo, unidad=columna_de_unidad,
                   minimo=_MINIMO_UNIDADES_DE_DESARROLLO)


# ---------------------------------------------------------------------------
# La propuesta
# ---------------------------------------------------------------------------

def proponer_particion_estricta(filas: Sequence[Mapping[str, Any]], *, plan_id: str, observation_id_field: str,
                                seed: int, unit_id_field: str | None, time_column: str | None,
                                objetivo: str | None = None,
                                fraccion_de_sujetos: float = FRACCION_DE_SUJETOS_POR_OMISION,
                                cuantil_del_corte: float = CUANTIL_DEL_CORTE_POR_OMISION, automatico: bool = True,
                                gap: Horizonte | Mapping[str, Any] | None = None, folds: int = 5, repeats: int = 1,
                                ids_cohorte_externa: Sequence[str] = ()) -> PropuestaEstricta:
    """`proponer_particion(split_type="groups_and_time")` en su forma ESTRICTA. Misma entrada que la publicada
    (`filas`, `plan_id`, `unit_id_field`, `time_column`, `gap`, `objetivo`, `ids_cohorte_externa`), con la cuota de
    sujetos de prueba (`fraccion_de_sujetos`, que NO es la fracción de filas de `test_fraction`) y el cuantil del corte.

    El resultado es una `PropuestaDeParticion` con pliegues (`pliegues`, `cobertura`) y su `reparto` declarado. Todo lo
    que no se puede es un `Bloqueo` con sus números, nunca una excepción de Python."""
    todos = {str(f[observation_id_field]): f for f in filas}
    externos = set(str(i) for i in ids_cohorte_externa)
    internos = {obs: fila for obs, fila in todos.items() if obs not in externos}

    def sin(*bloqueos: Bloqueo) -> PropuestaEstricta:
        return PropuestaEstricta(plan=None, pliegues=None, bloqueos=tuple(bloqueos))

    if not time_column:
        return sin(Bloqueo(clave="falta_columna_de_tiempo", campo="time_column",
                           motivo=motivo("falta_columna_de_tiempo", valor="groups_and_time")))
    try:
        gap = convertir_gap(gap)
        convertido = convertir_tiempos({obs: fila.get(time_column) for obs, fila in internos.items()}, time_column)
    except Rechazo as rechazo:
        return sin(rechazo.bloqueo())
    if gap is not None and not convertido.es_fecha and gap.magnitud > 0 and gap.unidad != "seconds":
        # Lo mismo que `temporal` (auditoría I4 del 140): un hueco en días sobre una columna que se lee como NÚMERO se
        # aplicaría como segundos. Aquí también, y se decide ANTES de cortar.
        ejemplos = sorted({str(fila.get(time_column)) for fila in internos.values()})[:3]
        return sin(BloqueoConNumeros(
            clave="hueco_en_tiempo_numerico", campo="gap",
            motivo=motivo("hueco_en_tiempo_numerico", campo=time_column, valor=f"{gap.magnitud:g} {gap.unidad}",
                          ejemplos=", ".join(ejemplos)),
            medida={"columna": time_column, "magnitud_del_hueco": gap.magnitud, "unidad_del_hueco": gap.unidad,
                    "ejemplos": ejemplos}))
    if not unit_id_field:
        return _como_estricta(_grupos_insuficientes(0, len(internos), unit_id_field))
    vacias = sum(1 for obs in todos if _unidad_falta(todos[obs].get(unit_id_field)))
    if vacias:
        return sin(BloqueoConNumeros(
            clave="unidad_vacia", campo="unit_id_field",
            motivo=motivo("unidad_vacia", campo=unit_id_field, valor=vacias, n=len(todos)),
            medida={"columna": unit_id_field, "filas_sin_unidad": vacias, "filas": len(todos)}))
    unidad_por_id = {obs: str(fila.get(unit_id_field)) for obs, fila in internos.items()}
    unidad_de_los_externos = {obs: str(todos[obs].get(unit_id_field)) for obs in externos if obs in todos}
    compartidas = sorted(set(unidad_de_los_externos.values()) & set(unidad_por_id.values()))
    if compartidas:
        return sin(BloqueoConNumeros(
            clave="cohorte_externa_comparte_unidad", campo="unit_id_field",
            motivo=motivo("cohorte_externa_comparte_unidad", campo=unit_id_field, valor=len(compartidas),
                          ejemplos=", ".join(compartidas[:3])),
            medida={"columna": unit_id_field, "unidades_compartidas": len(compartidas),
                    "ejemplos": compartidas[:3]}))
    n_grupos = len(set(unidad_por_id.values()))
    if n_grupos < 2:
        return _como_estricta(_grupos_insuficientes(n_grupos, len(internos), unit_id_field))
    if fraccion_de_sujetos <= 0:
        return _como_estricta(_sin_prueba(fraccion_de_sujetos, len(internos)))

    etiqueta_por_id = {obs: fila[objetivo] for obs, fila in internos.items()} if objetivo else None
    try:
        reparto = repartir_estricto(
            sorted(internos), unidad_por_id, convertido.valores, etiqueta_por_id=etiqueta_por_id, gap=gap,
            fraccion_de_sujetos=fraccion_de_sujetos, cuantil_del_corte=cuantil_del_corte, automatico=automatico,
            folds=folds, seed=seed, columna_de_unidad=unit_id_field, columna_de_tiempo=time_column,
            tiempo_original={obs: fila.get(time_column) for obs, fila in internos.items()},
            formato_de_fecha=convertido.formato, es_fecha=convertido.es_fecha)
    except Rechazo as rechazo:
        return sin(rechazo.bloqueo())

    limites = list(reparto.limites)
    if gap is not None and not convertido.es_fecha:
        limites.append(Limite(clave="tiempo_numerico_sin_unidad", campo="time_column",
                              motivo=motivo("tiempo_numerico_sin_unidad", campo=time_column),
                              medida={"hueco_segundos": segundos_de(gap)}))
    if repeats > 1:
        limites.append(Limite(clave="repeticiones_fijadas_a_1", campo="repeats",
                              motivo=motivo("repeticiones_fijadas_a_1", opciones=repeats, valor=1),
                              medida={"repeticiones_pedidas": repeats, "repeticiones_usadas": 1}))

    roles = {obs: "development" for obs in reparto.desarrollo}
    roles.update({obs: "test" for obs in reparto.prueba})
    roles.update({obs: "external_test" for obs in externos})
    # Las filas descartadas no tienen rol; `units` solo declara las que SÍ están en `roles` (como en la publicada).
    units = {obs: unidad for obs, unidad in {**unidad_por_id, **unidad_de_los_externos}.items() if obs in roles}
    plan = _plan_con_roles(plan_id, "groups_and_time", roles, observation_id_field, unit_id_field=unit_id_field,
                           units=units, time_column=time_column, gap=gap, folds=len(reparto.pliegues), repeats=1,
                           seed=seed)
    pliegues = tuple(Pliegue(repeticion=0, pliegue=p, entrena=entrena, valida=valida)
                     for p, (entrena, valida) in enumerate(reparto.pliegues))
    asignacion = AsignacionDePliegues(folds=len(pliegues), repeats=1, seed=seed, pliegues=pliegues)
    return PropuestaEstricta(
        plan=plan, pliegues=asignacion, limites=tuple(limites),
        cobertura=_cobertura(reparto.desarrollo, asignacion, reparto.quitadas_por_el_hueco),
        tiempo_por_id={o: convertido.valores[o] for o in (*reparto.desarrollo, *reparto.prueba)},
        reparto=reparto)


def _como_estricta(propuesta: PropuestaDeParticion) -> PropuestaEstricta:
    return PropuestaEstricta(plan=propuesta.plan, pliegues=propuesta.pliegues, bloqueos=propuesta.bloqueos,
                             limites=propuesta.limites, cobertura=propuesta.cobertura)
