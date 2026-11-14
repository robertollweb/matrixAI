# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2b — el banco de la CODIFICACIÓN DE CATEGÓRICAS de «Filas a revisar» (auditoría 165, I4 e I5). Pieza pura (sin E/S de medida): la usan
`medir_c165_c2b.py` y las pruebas. Regla: `REGLA_C2B.md` (sellada en `analiza_c165_c2b.py`). NUNCA es una cifra de producto.

Reutiliza del banco del C2 (`banco_c165`, sellado, que NO se toca) la carga de los conjuntos, la exclusión de columnas del producto, la siembra
`extremo` y las métricas; añade aquí lo nuevo: los conjuntos DERIVADOS con categorías, las 6 variantes de codificación, las siembras de «categoría
rara» con nombres neutros (y la del C2 para comparar), la sensibilidad a RENOMBRAR, la referencia de la semilla y la inflación por la vacía.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np

import banco_c165 as banco
from banco_c165 import _an, es_ausente, es_numero

ORDENES = ("alfabetico", "frecuencia")
VACIAS = ("menos_uno", "nivel_propio", "centro")
HOY = {"orden": "alfabetico", "vacia": "menos_uno"}
VARIANTES = [{"orden": o, "vacia": v} for o in ORDENES for v in VACIAS]
TIPOS_RARA = ("rara_medio", "rara_azar", "rara_final")
TIPOS = TIPOS_RARA + ("extremo",)
TIPOS_DE_DECISION = ("rara_medio", "rara_azar", "extremo")
TASAS = banco.TASAS
REPETICIONES = banco.REPETICIONES
CONJUNTO_REAL = "dresses-sales"
DERIVADOS = ("breast-cancer-wisconsin-diagnostic", "cardiotocography", "climate-model-simulation-crashes", "pc1", "us_crime")
NOMBRES = (CONJUNTO_REAL,) + DERIVADOS
CORTES = (0.05, 0.20, 0.45, 0.75, 0.93)
FRACCION_VACIA = 0.03
#: 40 municipios con tildes, mayúsculas y minúsculas: nombres «de la gente», y ninguno empieza por `!` ni por `zzzz_`.
MUNICIPIOS = ("Ávila", "Málaga", "Logroño", "Santander", "Cádiz", "León", "Úbeda", "Écija", "Zaragoza", "Bilbao", "Oviedo", "Huelva",
              "Jaén", "Lugo", "Ourense", "Soria", "Teruel", "Burgos", "Palencia", "Segovia", "Cuenca", "Toledo", "Murcia", "Almería",
              "Córdoba", "Sevilla", "Granada", "Badajoz", "Cáceres", "Girona", "Lleida", "Tarragona", "Castellón", "Alicante", "Elche",
              "Gijón", "Vigo", "Ferrol", "Pamplona", "Vitoria")


def nombre_de_variante(v: dict) -> str:
    return f"{v['orden']}|{v['vacia']}"


def es_hoy(v: dict) -> bool:
    return all(v[k] == HOY[k] for k in HOY)


def k_del_producto(n: int) -> int:
    return min(50, max(1, math.ceil(0.05 * n)))


# ---------------------------------------------------------------- conjuntos
def _columnas_a_categorizar(filas: list[dict]) -> list[str]:
    nums = sorted(banco._numericas(filas))
    if len(nums) < 3:
        raise ValueError("menos de 3 numericas completas con >= 10 valores distintos")
    return [nums[0], nums[len(nums) // 2], nums[-1]]


def derivar(nombre: str, filas: list[dict]) -> tuple[list[dict], dict]:
    """El conjunto DERIVADO: tres numéricas pasan a categóricas de seis niveles con nombres de municipio barajados y un 3 % de vacías."""
    nuevas = [dict(f) for f in filas]
    detalle: dict[str, Any] = {"columnas": {}}
    for c in _columnas_a_categorizar(filas):
        rng = random.Random(banco._semilla("c2b-derivar", nombre, c))
        x = np.array([float(f[c]) for f in filas])
        cortes = np.quantile(x, CORTES)
        nivel = np.searchsorted(cortes, x, side="right")          # 0..5
        nombres = list(MUNICIPIOS)
        rng.shuffle(nombres)
        nombres = nombres[:len(CORTES) + 1]
        vacias = set(rng.sample(range(len(filas)), max(1, round(FRACCION_VACIA * len(filas)))))
        for i, f in enumerate(nuevas):
            f[c] = "" if i in vacias else nombres[int(nivel[i])]
        detalle["columnas"][c] = {"niveles": nombres, "vacias": len(vacias)}
    return nuevas, detalle


def cargar(nombre: str) -> tuple[list[dict], dict]:
    """(filas, detalle): el real tal cual (sin objetivo) o el derivado."""
    filas = banco.cargar(nombre)
    if nombre == CONJUNTO_REAL:
        return filas, {"derivado": False}
    nuevas, det = derivar(nombre, filas)
    return nuevas, {"derivado": True, **det}


# ---------------------------------------------------------------- codificación
def _codigos(txt: list[str | None], variante: dict) -> np.ndarray:
    """Los códigos ordinales de una categórica (`None` = vacía) con la variante: el orden de los niveles y qué se hace con la vacía."""
    if variante["vacia"] == "nivel_propio":
        txt = ["" if t is None else t for t in txt]                # la vacía es un nivel más, con nombre «»
    aparicion = {k: i for i, k in enumerate(dict.fromkeys(t for t in txt if t is not None))}
    cont: dict[str, int] = {}
    for t in txt:
        if t is not None:
            cont[t] = cont.get(t, 0) + 1
    if variante["orden"] == "alfabetico":
        ordenados = sorted(cont)
    else:                                                          # a igual frecuencia, la primera aparición: no depende del nombre
        ordenados = sorted(cont, key=lambda k: (-cont[k], aparicion[k]))
    cod = {k: float(i) for i, k in enumerate(ordenados)}
    relleno = -1.0
    if variante["vacia"] == "centro":
        con_dato = [cod[t] for t in txt if t is not None]
        relleno = float(np.median(con_dato)) if con_dato else -1.0
    return np.array([cod[t] if t is not None else relleno for t in txt], float)


def matriz(filas: list[dict], variante: dict) -> np.ndarray:
    usadas, tipos = banco._columnas_usadas(filas)
    cols = []
    for c in usadas:
        if tipos[c] == "numerica":
            v = np.array([float(f[c]) if es_numero(f.get(c)) else np.nan for f in filas])
            cols.append(np.where(np.isnan(v), float(np.nanmedian(v)), v))
        else:
            cols.append(_codigos([None if es_ausente(f.get(c)) else str(f.get(c)) for f in filas], variante))
    return np.column_stack(cols)


def ordenar(filas: list[dict], variante: dict, semilla: int = 0) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    X = matriz(filas, variante)
    m = IsolationForest(n_estimators=200, max_samples=len(X), random_state=semilla, n_jobs=1).fit(X)
    p = -m.score_samples(X)
    return np.lexsort((np.arange(len(p)), -p))


def control_hoy_igual_al_producto(filas: list[dict]) -> bool:
    res = _an.puntuar_filas(filas, None, 0, n_semillas=2)
    return [int(i) for i in ordenar(filas, HOY)] == [e["fila"] - 1 for e in res["filas_a_revisar"]]


# ---------------------------------------------------------------- siembras
def categoricas(filas: list[dict]) -> list[str]:
    """Las categóricas USADAS por el producto con 2 a 30 niveles, por nombre."""
    usadas, tipos = banco._columnas_usadas(filas)
    out = []
    for c in usadas:
        if tipos[c] != "numerica":
            niveles = {f.get(c) for f in filas if not es_ausente(f.get(c))}
            if 2 <= len(niveles) <= 30:
                out.append(c)
    return sorted(out)


def posicion_alfabetica(nombre: str, niveles: Sequence[str]) -> float:
    orden = sorted(set(niveles) | {nombre})
    return orden.index(nombre) / max(1, len(orden) - 1)


def sembrar(filas: list[dict], tipo: str, tasa: float, repeticion: int, nombre: str = "") -> tuple[list[dict], list[int], dict]:
    if tipo == "extremo":
        return banco.sembrar(filas, "extremo", tasa, repeticion, nombre)
    if tipo not in TIPOS_RARA:
        raise ValueError(tipo)
    rng = random.Random(banco._semilla("c2b", nombre, tipo, tasa, repeticion))
    n = len(filas)
    ns = banco.n_sembradas(n, tasa)
    elegidas = sorted(rng.sample(range(n), ns))
    cats = categoricas(filas)
    if not cats:
        raise ValueError("sin categoricas")
    c = rng.choice(cats)
    niveles = sorted({str(f[c]) for f in filas if not es_ausente(f.get(c))})
    nuevas = [dict(f) for f in filas]
    posiciones = []
    for i in elegidas:
        if tipo == "rara_final":
            nuevo = f"valor_nuevo_{i}"
        elif tipo == "rara_medio":
            nuevo = f"{niveles[(len(niveles) - 1) // 2]}~{i:05d}"
        else:
            nuevo = f"{rng.choice(niveles)}~{i:05d}"
        nuevas[i][c] = nuevo
        posiciones.append(posicion_alfabetica(nuevo, niveles))
    return nuevas, elegidas, {"tipo": tipo, "n_sembradas": ns, "columna": c, "posicion_media": float(np.mean(posiciones))}


# ---------------------------------------------------------------- renombrar, semilla, vacías
def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def renombres(filas: list[dict]) -> list[dict]:
    """Los renombres de la regla: en las dos primeras categóricas, el nivel más y el menos frecuente, a `!`+nombre y a `zzzz_`+nombre."""
    out = []
    for c in categoricas(filas)[:2]:
        cont: dict[str, int] = {}
        for f in filas:
            v = f.get(c)
            if not es_ausente(v):
                cont[str(v)] = cont.get(str(v), 0) + 1
        aparicion = {k: i for i, k in enumerate(dict.fromkeys(str(f[c]) for f in filas if not es_ausente(f.get(c))))}
        por_frec = sorted(cont, key=lambda k: (-cont[k], aparicion[k]))
        menos = sorted(cont, key=lambda k: (cont[k], aparicion[k]))[0]
        for nivel in dict.fromkeys([por_frec[0], menos]):
            for prefijo in ("!", "zzzz_"):
                out.append({"columna": c, "nivel": nivel, "nuevo": prefijo + nivel})
    return out


def renombrar(filas: list[dict], r: dict) -> list[dict]:
    return [{**f, r["columna"]: r["nuevo"]} if str(f.get(r["columna"])) == r["nivel"] else dict(f) for f in filas]


def inflacion_por_la_vacia(filas: list[dict], orden: Sequence[int]) -> float | None:
    """(proporción de filas con alguna categórica usada vacía entre las k primeras) / (esa proporción en todo el conjunto)."""
    usadas, tipos = banco._columnas_usadas(filas)
    cats = [c for c in usadas if tipos[c] != "numerica"]
    con_vacia = [any(es_ausente(f.get(c)) for c in cats) for f in filas]
    base = sum(con_vacia) / len(filas)
    if base == 0:
        return None
    k = k_del_producto(len(filas))
    return (sum(con_vacia[int(i)] for i in orden[:k]) / k) / base


def metricas(orden: Sequence[int], sembradas: Sequence[int]) -> dict:
    return banco.metricas(orden, sembradas)
