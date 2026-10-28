# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140-C1 — `groups` estratificado por clase, respetando la unidad.

**Qué resuelve.** Hasta aquí `groups` separaba por sujeto pero no miraba la
clase: la prueba eran grupos barajados hasta el 20 % de las filas y los pliegues,
grupos barajados. Medido el 27-10 (8 sujetos × 10 filas, los positivos en 2
sujetos): la prueba reservada salía con UNA sola clase en las 4 semillas, y
varios pliegues de validación sin un positivo, sin declarar nada.
`StratifiedGroupKFold` de sklearn tampoco lo evita (3 de 5 `valida` sin
positivos y ningún aviso), así que la comprobación se hace aquí.

**Cómo.** Reparto determinista (Python puro: el núcleo tiene `dependencies = []`;
sklearn solo es oráculo en las pruebas):

- Cada grupo es una unidad indivisible con su vector de filas por clase. Se
  reparten en «cubos» (la prueba y los pliegues) con un objetivo por cubo y clase.
- **Primero se CUBREN las clases, con una búsqueda exacta** (`_cubrir`): cada cubo
  recibe, para cada clase, al menos un grupo que la traiga. Un voraz no basta —medido
  el 28-10: en 125 de 137 casos de tres clases en los que el voraz fallaba, el
  reparto existía (comprobado por fuerza bruta)—, así que es una búsqueda con
  vuelta atrás (primero la pareja cubo/clase con menos grupos posibles). En el
  caso de dos clases nunca vuelve atrás. Para conjuntos grandes la ramificación se
  acota (8 alternativas por paso): ahí un «no se ha encontrado» ya no es una
  prueba de que no exista.
- **Después se reparte el resto**, de los grupos grandes a los pequeños (con un
  pequeño desorden sembrado, para que otra semilla dé otro reparto), cada grupo al
  cubo donde MENOS crece la varianza entre cubos del llenado relativo (filas de la
  clase / objetivo del cubo).
- La prueba se aparta ANTES que los pliegues y sin tocar los grupos que cubren los
  pliegues: así a desarrollo le queda un reparto que cubre las `K` partes, en
  todas las repeticiones.

**Las guardas (D2 del contrato 140, literal de Roberto): «repartir la clase entre
grupos; si no cabe, bajar el número de pliegues hasta 2 y decirlo; si ni así,
rechazar el estudio con los números».** Se prueba con `K` pliegues; si el
reparto no deja TODAS las clases en la prueba y en cada `valida`, con `K−1`…
hasta 2. Cada reducción es un `Limite` con las cifras; si ni con 2 cabe, un
`Rechazo` con cuántos grupos tienen la clase y cuántos hacen falta. Nunca una
prueba con una sola clase en silencio.

**Lo que NO cambia**: sin objetivo de clasificación (regresión) o con una sola
clase, esta pieza no se usa y el reparto es el de siempre (`_kfold_groups`).
"""

from __future__ import annotations

import heapq
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from matrixai.training.diagnostico import Limite
from matrixai.training.particion_por_diseno_comun import Rechazo, nombres_de_clases
from matrixai.training.particion_por_diseno_textos import motivo

__all__ = ["MARGEN_RELATIVO_DE_LA_PRUEBA", "MINIMO_PLIEGUES", "ReparteGrupos", "fuera_del_margen",
           "repartir_cubos", "repartir_grupos_estratificado"]

MINIMO_PLIEGUES = 2
#: Cuánto puede apartarse la fracción de prueba obtenida de la pedida, en relativo, sin declararlo (auditoría I1):
#: con 0,2 pedido, de 0,1 a 0,3. Los grupos no se parten, así que alguna desviación es inevitable; más que esto se dice.
MARGEN_RELATIVO_DE_LA_PRUEBA = 0.5


def fuera_del_margen(filas_de_la_parte: int, filas: int, fraccion: float, *, solo_por_arriba: bool = False) -> bool:
    """¿Se aparta `filas_de_la_parte / filas` de `fraccion` más que el margen? LA definición del margen: la usan el
    `Limite` que lo declara y la ruta sin estratificar al decidir si un grupo cabe (re-auditoría, M-a: un tope
    redondeado, `round(n·f)·1,5`, cambiaba repartos que la 139 ya dejaba dentro del margen). `solo_por_arriba`:
    quedarse corto no cuenta (un grupo que aún no llega a lo pedido «cabe»)."""
    desvio = filas_de_la_parte / filas - fraccion
    return (desvio if solo_por_arriba else abs(desvio)) > MARGEN_RELATIVO_DE_LA_PRUEBA * fraccion


def _con_margen(capacidad: float) -> float:
    """Las filas que puede llevar un cubo sin salirse del margen por arriba (`capacidad` son las que le tocan)."""
    return capacidad * (1 + MARGEN_RELATIVO_DE_LA_PRUEBA)
#: Cuánto se desordena el tamaño de cada grupo al ordenarlos de grande a pequeño.
_DESORDEN = 0.25


class _Tablero:
    """Los cubos, con lo que llevan, y el coste de poner un grupo en uno.

    El coste es el aumento de la VARIANZA, entre cubos, del llenado relativo de
    cada clase (filas de la clase en el cubo / filas de la clase que le tocan).
    Con el llenado relativo, un cubo del 20 % y uno del 80 % son comparables: el
    reparto no empuja todo al grande."""

    def __init__(self, objetivos: Sequence[Mapping[Any, float]]) -> None:
        self.n = len(objetivos)
        self.objetivos = [dict(o) for o in objetivos]
        self.filas = [defaultdict(float) for _ in objetivos]
        self.suma = defaultdict(float)
        self.suma_cuadrados = defaultdict(float)
        self.grupos: list[list[str]] = [[] for _ in objetivos]
        self.tiene: list[set[Any]] = [set() for _ in objetivos]

    def coste(self, cubo: int, cuenta: Mapping[Any, int]) -> float:
        total = 0.0
        for clase, n in cuenta.items():
            objetivo = self.objetivos[cubo].get(clase, 0.0) or 1e-9
            llenado, paso = self.filas[cubo][clase] / objetivo, n / objetivo
            s1, s2 = self.suma[clase], self.suma_cuadrados[clase]
            antes = s2 / self.n - (s1 / self.n) ** 2
            s1n, s2n = s1 + paso, s2 + 2 * llenado * paso + paso * paso
            total += (s2n / self.n - (s1n / self.n) ** 2) - antes
        return total

    def poner(self, cubo: int, grupo: str, cuenta: Mapping[Any, int]) -> None:
        for clase, n in cuenta.items():
            objetivo = self.objetivos[cubo].get(clase, 0.0) or 1e-9
            llenado, paso = self.filas[cubo][clase] / objetivo, n / objetivo
            self.suma[clase] += paso
            self.suma_cuadrados[clase] += 2 * llenado * paso + paso * paso
            self.filas[cubo][clase] += n
            self.tiene[cubo].add(clase)
        self.grupos[cubo].append(grupo)


_NODOS_MAXIMOS = 5000
_RAMAS_EN_GRANDE = 8
_GRANDE = 64


def _cubrir(cuentas: Mapping[str, Mapping[Any, int]], clases: Sequence[Any], n_cubos: int,
            azar: random.Random, excluidos: frozenset[str] = frozenset(),
            capacidades: Sequence[float] | None = None) -> list[list[str]] | None:
    """Un grupo por pareja (cubo, clase) pendiente, de modo que CADA cubo tenga TODAS las clases;
    `None` si no se encuentra. Búsqueda con vuelta atrás, la pareja con menos grupos libres primero.

    `capacidades` son las filas que le tocan a cada cubo. **Entre los grupos que traen la clase pendiente se
    prefiere primero el que NO desborda el cubo** (auditoría I1, 27-10): sin esto se elegía el que traía más
    clases nuevas, que en datos longitudinales es el sujeto más grande —el único con las dos clases—, y la
    prueba, que se cubre la primera, salía del 60 % pedida al 20 %. El orden solo cambia qué se prueba antes:
    la vuelta atrás encuentra un reparto igual si existe.

    **Desbordar es pasarse del MARGEN, no de la capacidad pelada** (re-auditoría, IMPORTANTE 1, 28-10): con la
    capacidad pelada, cualquier grupo que pasara de la parte justa por una fila contaba como peor que uno diminuto,
    y en el caso 90 de la re-auditoría (90 filas, 20 % pedido) la prueba salía con 4 filas (4,4 %) donde el C1 la
    dejaba en 21 (23 %, dentro del margen). Es el mismo margen que la ruta
    sin estratificar (`MARGEN_RELATIVO_DE_LA_PRUEBA`; `_con_margen` es `fuera_del_margen` por arriba, en filas)."""
    nombres = [g for g in sorted(cuentas) if g not in excluidos]
    filas_de = {g: sum(cuentas[g].values()) for g in nombres}
    lleno = [0.0] * n_cubos
    tamano = {g: sum(cuentas[g].values()) * (1.0 + _DESORDEN * azar.random()) for g in nombres}
    con_clase = {c: [g for g in nombres if cuentas[g].get(c)] for c in clases}
    libres = {c: len(con_clase[c]) for c in clases}
    asignado: dict[str, int] = {}
    cubierto: list[set[Any]] = [set() for _ in range(n_cubos)]
    nodos = [0]

    def resolver() -> bool:
        pendientes = [(b, c) for b in range(n_cubos) for c in clases if c not in cubierto[b]]
        if not pendientes:
            return True
        nodos[0] += 1
        if nodos[0] > _NODOS_MAXIMOS:
            return False
        b, c = min(pendientes, key=lambda bc: (libres[bc[1]], str(bc[1]), bc[0]))
        if libres[c] == 0:
            return False
        candidatos = [g for g in con_clase[c] if g not in asignado]
        nuevas = lambda g: sum(1 for k in cuentas[g] if k not in cubierto[b])   # noqa: E731
        desborda = (lambda g: max(0.0, lleno[b] + filas_de[g] - _con_margen(capacidades[b]))) if capacidades is not None \
            else (lambda g: 0.0)                                                # noqa: E731
        clave = lambda g: (desborda(g), -nuevas(g), tamano[g], g)               # noqa: E731
        orden = sorted(candidatos, key=clave) if len(candidatos) <= _GRANDE \
            else heapq.nsmallest(_RAMAS_EN_GRANDE, candidatos, key=clave)
        for g in orden:
            anadidas = [k for k in cuentas[g] if k not in cubierto[b]]
            asignado[g] = b
            lleno[b] += filas_de[g]
            cubierto[b].update(anadidas)
            for k in cuentas[g]:
                libres[k] -= 1
            if resolver():
                return True
            for k in cuentas[g]:
                libres[k] += 1
            cubierto[b].difference_update(anadidas)
            lleno[b] -= filas_de[g]
            del asignado[g]
        return False

    if not resolver():
        return None
    cubos: list[list[str]] = [[] for _ in range(n_cubos)]
    for g in nombres:
        if g in asignado:
            cubos[asignado[g]].append(g)
    return cubos


def repartir_cubos(cuentas: Mapping[str, Mapping[Any, int]], clases: Sequence[Any],
                   objetivos: Sequence[Mapping[Any, float]], azar: random.Random, *,
                   cubrir: bool = True, previos: Sequence[Sequence[str]] | None = None,
                   ) -> list[list[str]] | None:
    """Reparte los grupos (`{grupo: {clase: filas}}`) en `len(objetivos)` cubos.

    Con `cubrir`, cada cubo lleva todas las clases o devuelve `None` (no se encontró un reparto que
    lo consiga). `previos` son grupos ya colocados en cada cubo (los que cubren, buscados antes)."""
    tablero = _Tablero(objetivos)
    libres = set(cuentas)
    if previos is None:
        previos = (_cubrir(cuentas, clases, tablero.n, azar, capacidades=[sum(o.values()) for o in objetivos])
                   if cubrir else [[] for _ in objetivos])
        if previos is None:
            return None
    for cubo, grupos in enumerate(previos):
        for g in grupos:
            tablero.poner(cubo, g, cuentas[g])
            libres.discard(g)
    clave_de_orden = {g: -sum(cuentas[g].values()) * (1.0 + _DESORDEN * azar.random()) for g in sorted(cuentas)}
    for grupo in sorted(libres, key=lambda g: (clave_de_orden[g], g)):
        orden = list(range(tablero.n))
        azar.shuffle(orden)
        cubo = min(orden, key=lambda b: tablero.coste(b, cuentas[grupo]))      # empata el primero
        tablero.poner(cubo, grupo, cuentas[grupo])
    return tablero.grupos


# ---------------------------------------------------------------------------
# El reparto de la propuesta: prueba reservada + K pliegues
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReparteGrupos:
    pliegues_usados: int
    #: Los grupos de la prueba reservada (vacío si `test_fraction` es 0).
    grupos_de_prueba: tuple[str, ...]
    #: Por repetición, los cubos de GRUPOS (uno por pliegue) de desarrollo.
    cubos_por_repeticion: tuple[tuple[tuple[str, ...], ...], ...]
    limites: tuple[Limite, ...]


def _cuentas(filas_por_unidad: Mapping[str, Sequence[str]], clase_por_id: Mapping[str, Any]
             ) -> dict[str, dict[Any, int]]:
    cuentas: dict[str, dict[Any, int]] = {}
    for unidad, filas in filas_por_unidad.items():
        c: dict[Any, int] = {}
        for obs in filas:
            c[clase_por_id[obs]] = c.get(clase_por_id[obs], 0) + 1
        cuentas[unidad] = c
    return cuentas


def _hay_todas(grupos: Sequence[str], cuentas: Mapping[str, Mapping[Any, int]], clases: Sequence[Any]) -> bool:
    presentes = {c for g in grupos for c in cuentas[g]}
    return all(c in presentes for c in clases)


def _intentar(cuentas: Mapping[str, Mapping[Any, int]], clases: Sequence[Any], k: int,
              test_fraction: float, seed: int, repeats: int) -> ReparteGrupos | None:
    total = {c: sum(cuentas[g].get(c, 0) for g in cuentas) for c in clases}
    grupos_de_prueba: list[str] = []
    if test_fraction > 0:
        # Un reparto que cubre la prueba y los `k` pliegues A LA VEZ: la prueba se lleva sus grupos y NO
        # toca los de los pliegues, así que a desarrollo le queda con qué cubrirlos en cada repetición.
        filas = sum(total.values())
        cubierta = _cubrir(cuentas, clases, k + 1, random.Random(f"{seed}:corte"),
                           capacidades=[filas * test_fraction] + [filas * (1 - test_fraction) / k] * k)
        if cubierta is None:
            return None
        grupos_de_prueba = _llenar_la_prueba(cuentas, clases, total, test_fraction, cubierta,
                                             random.Random(f"{seed}:corte:resto"))
    en_prueba = set(grupos_de_prueba)
    desarrollo = {g: cuentas[g] for g in cuentas if g not in en_prueba}
    total_dev = {c: sum(desarrollo[g].get(c, 0) for g in desarrollo) for c in clases}
    por_repeticion = []
    for r in range(repeats):
        azar = random.Random(f"{seed}:groups:{r}")
        cubos = repartir_cubos(desarrollo, clases, [{c: total_dev[c] / k for c in clases} for _ in range(k)], azar)
        if cubos is None:
            return None
        # el orden de los grupos dentro de un pliegue no debe llevar el de la asignación (de grandes a pequeños)
        for cubo in cubos:
            azar.shuffle(cubo)
        por_repeticion.append(tuple(tuple(c) for c in cubos))
    return ReparteGrupos(k, tuple(grupos_de_prueba), tuple(por_repeticion), ())


def _llenar_la_prueba(cuentas, clases, total, test_fraction, cubierta, azar) -> list[str]:
    """Los grupos de la prueba: los que cubren (cubo 0 de `cubierta`) y los que hagan falta hasta su
    parte, sin tocar los que cubren los pliegues."""
    tablero = _Tablero([{c: total[c] * test_fraction for c in clases},
                        {c: total[c] * (1 - test_fraction) for c in clases}])
    colocados = set()
    for destino, grupos in ((0, cubierta[0]), (1, [g for cubo in cubierta[1:] for g in cubo])):
        for g in grupos:
            tablero.poner(destino, g, cuentas[g])
            colocados.add(g)
    clave_de_orden = {g: -sum(cuentas[g].values()) * (1.0 + _DESORDEN * azar.random()) for g in sorted(cuentas)}
    for grupo in sorted((g for g in cuentas if g not in colocados), key=lambda g: (clave_de_orden[g], g)):
        a_la_prueba = tablero.coste(0, cuentas[grupo]) < tablero.coste(1, cuentas[grupo])
        tablero.poner(0 if a_la_prueba else 1, grupo, cuentas[grupo])
    return tablero.grupos[0]


def _limite_de_fraccion(cuentas: Mapping[str, Mapping[Any, int]], grupos_de_prueba: Sequence[str],
                        test_fraction: float, campo: str) -> tuple[Limite, ...]:
    """Un `Limite` si la prueba se aparta de la fracción pedida más del margen (auditoría I1)."""
    if test_fraction <= 0 or not grupos_de_prueba:
        return ()
    filas = sum(sum(c.values()) for c in cuentas.values())
    en_prueba = sum(sum(cuentas[g].values()) for g in grupos_de_prueba)
    obtenida = en_prueba / filas
    if not fuera_del_margen(en_prueba, filas, test_fraction):
        return ()
    mayor = max(grupos_de_prueba, key=lambda g: (sum(cuentas[g].values()), g))
    medida = {"fraccion_pedida": test_fraction, "fraccion_obtenida": round(obtenida, 4),
              "filas_de_prueba": en_prueba, "filas": filas, "grupos_de_prueba": len(grupos_de_prueba),
              "mayor_grupo_de_prueba": mayor, "filas_del_mayor_grupo_de_prueba": sum(cuentas[mayor].values()),
              "margen_relativo": MARGEN_RELATIVO_DE_LA_PRUEBA}
    return (Limite(clave="fraccion_de_prueba_desviada", campo="unit_id_field",
                   motivo=motivo("fraccion_de_prueba_desviada", campo=campo,
                                 opciones=f"{100 * test_fraction:g}", valor=f"{100 * obtenida:.1f}",
                                 n=en_prueba, partes=filas, grupo=mayor,
                                 minimo=sum(cuentas[mayor].values())),
                   medida=medida),)


def repartir_grupos_estratificado(filas_por_unidad: Mapping[str, Sequence[str]],
                                  clase_por_id: Mapping[str, Any], *, folds: int,
                                  test_fraction: float, seed: int, repeats: int,
                                  campo: str = "unit_id_field") -> ReparteGrupos:
    """La prueba y los pliegues de `groups`, con todas las clases en cada uno.

    `clase_por_id` cubre todas las filas de `filas_por_unidad`, y trae al menos
    dos clases (con una sola, el llamante usa el reparto de siempre). Lanza
    `Rechazo` si ni con 2 pliegues cabe."""
    cuentas = _cuentas(filas_por_unidad, clase_por_id)
    clases = sorted({c for cuenta in cuentas.values() for c in cuenta}, key=str)
    grupos_con = {c: sum(1 for cuenta in cuentas.values() if cuenta.get(c)) for c in clases}
    nombre = nombres_de_clases(clases)      # `1` y `"1"` no se escriben igual (re-auditoría, M-c)
    partes_extra = 1 if test_fraction > 0 else 0
    pedidos = max(MINIMO_PLIEGUES, folds)

    def la_rara() -> Any:
        return min(clases, key=lambda c: (grupos_con[c], str(c)))

    ultimo_motivo: tuple[str, Any] | None = None
    for k in range(pedidos, MINIMO_PLIEGUES - 1, -1):
        faltan = [c for c in clases if grupos_con[c] < k + partes_extra]
        resultado = None if faltan else _intentar(cuentas, clases, k, test_fraction, seed, repeats)
        if resultado is None:
            ultimo_motivo = ("conteo", min(faltan, key=lambda c: (grupos_con[c], str(c)))) if faltan \
                else ("reparto", la_rara())
            continue
        desvio = _limite_de_fraccion(cuentas, resultado.grupos_de_prueba, test_fraction, campo)
        if k == pedidos:
            return ReparteGrupos(resultado.pliegues_usados, resultado.grupos_de_prueba,
                                 resultado.cubos_por_repeticion, desvio)
        causa, clase = ultimo_motivo   # lo que impidió k + 1
        medida = {"pliegues_pedidos": pedidos, "pliegues_usados": k, "clase": nombre[clase],
                  "grupos_con_la_clase": grupos_con[clase], "grupos_necesarios": pedidos + partes_extra,
                  "grupos_por_clase": {nombre[c]: grupos_con[c] for c in clases}}
        clave = ("pliegues_reducidos_por_clase_en_grupos" if causa == "conteo"
                 else "pliegues_reducidos_por_reparto_de_clases_en_grupos")
        limite = Limite(clave=clave, campo="unit_id_field",
                        motivo=motivo(clave, campo=campo, opciones=pedidos, valor=k, clase=nombre[clase],
                                      n=grupos_con[clase], partes=pedidos + partes_extra),
                        medida=medida)
        return ReparteGrupos(resultado.pliegues_usados, resultado.grupos_de_prueba,
                             resultado.cubos_por_repeticion, (limite,) + desvio)

    clase = la_rara()
    minimo = MINIMO_PLIEGUES + partes_extra
    medida = {"pliegues_pedidos": pedidos, "clase": nombre[clase],
              "grupos_con_la_clase": grupos_con[clase], "grupos_necesarios_con_los_pedidos": pedidos + partes_extra,
              "grupos_necesarios_con_2_pliegues": minimo,
              "grupos_por_clase": {nombre[c]: grupos_con[c] for c in clases}}
    if grupos_con[clase] < minimo:
        raise Rechazo("clase_sin_grupos_suficientes", "unit_id_field", medida, campo=campo, clase=nombre[clase],
                      valor=grupos_con[clase], opciones=pedidos, partes=pedidos + partes_extra, minimo=minimo)
    raise Rechazo("clases_sin_reparto_por_grupos", "unit_id_field", medida, campo=campo, clase=nombre[clase],
                  valor=grupos_con[clase])
