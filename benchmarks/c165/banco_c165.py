# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2 — el BANCO INTERNO de «Filas a revisar»: anomalías sembradas por nosotros y recuperación medida. Pieza pura (sin E/S de medida):
la usan `medir_c165_c2.py` y las pruebas. NUNCA es una cifra de producto (ver `REGLA_C2.md`).

  * `sembrar(...)` sustituye filas por anomalías de UN tipo (`extremo`, `combinacion`, `rara`), con semilla, y devuelve QUÉ filas se sembraron.
  * `ordenar(...)` da el orden de revisión (0 = la más rara) de una VARIANTE de defectos: escalado × categóricas × max_samples.
    La variante «hoy» tiene que reproducir `matrixai_engines.anomalias.puntuar_filas` (control).
  * `metricas(...)` da recuperación@k, precisión@k y AP de lo sembrado.
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

from matrixai_engines import anomalias as _an  # noqa: E402
from matrixai_engines.tipos_de_columna import es_ausente, es_numero, tipificar  # noqa: E402

TIPOS = ("extremo", "combinacion", "rara")
TASAS = (0.01, 0.05)
REPETICIONES = (0, 1, 2)
ESCALADOS = ("ninguno", "cuantiles")
CATEGORICAS = ("frecuencia", "ordinal", "onehot")
MAX_SAMPLES = ("auto", "todas")
HOY = {"escalado": "ninguno", "categoricas": "frecuencia", "max_samples": "auto"}
VARIANTES = [{"escalado": e, "categoricas": c, "max_samples": m} for e in ESCALADOS for c in CATEGORICAS for m in MAX_SAMPLES]
MIN_SEMBRADAS = 3
CORR_MIN, CORR_COPIA = 0.5, 0.999

#: (nombre, csv, columna objetivo que se QUITA): el catálogo del Studio.
CONJUNTOS = (("breast-cancer-wisconsin-diagnostic", "breast-cancer-wisconsin-diagnostic.csv", "diagnostico"),
             ("cardiotocography", "cardiotocography.csv", "estado_fetal"),
             ("climate-model-simulation-crashes", "climate-model-simulation-crashes.csv", "outcome"),
             ("dresses-sales", "dresses-sales.csv", "Class"),
             ("pc1", "pc1.csv", "defects"),
             ("us_crime", "us_crime.csv", "ViolentCrimesPerPop"))
DIR_DATOS = _RAIZ.parent / "matrixaistudio" / "studio-backend" / "matrixai_studio" / "ejemplos_medidos"


def nombre_de_variante(v: dict) -> str:
    return f"{v['escalado']}|{v['categoricas']}|{v['max_samples']}"


def es_hoy(v: dict) -> bool:
    return all(v[k] == HOY[k] for k in HOY)


def sha256_fichero(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar(nombre: str) -> list[dict]:
    """Las filas del conjunto SIN la columna objetivo (el flujo no tiene objetivo)."""
    for n, csv_, objetivo in CONJUNTOS:
        if n == nombre:
            filas = _an.leer_csv(str(DIR_DATOS / csv_))
            return [{k: v for k, v in f.items() if k != objetivo} for f in filas]
    raise KeyError(nombre)


def _semilla(*partes: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, partes)).encode()).hexdigest()[:8], 16)


def _numericas(filas: list[dict]) -> dict[str, np.ndarray]:
    """Columnas numéricas COMPLETAS (todo presente y numérico) con ≥ 10 valores distintos y no constantes: de ellas se siembra."""
    out = {}
    for c in filas[0]:
        vs = [_an._decimal(f.get(c)) for f in filas]
        if all(es_numero(v) for v in vs):
            x = np.array([float(v) for v in vs])
            if len(np.unique(x)) >= 10:
                out[c] = x
    return out


def _categoricas(filas: list[dict]) -> list[str]:
    tipos = tipificar(filas, list(filas[0])).tipos
    out = []
    for c, t in tipos.items():
        niveles = {f.get(c) for f in filas if not es_ausente(f.get(c))}
        if t == "categorica" and 2 <= len(niveles) <= 30:
            out.append(c)
    return out


def pareja_mas_correlacionada(filas: list[dict]) -> tuple[str, str, float] | None:
    nums = _numericas(filas)
    cols = list(nums)[:60]
    if len(cols) < 2:
        return None
    M = np.column_stack([nums[c] for c in cols])
    with np.errstate(all="ignore"):
        R = np.corrcoef(M, rowvar=False)
    mejor = None
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = R[i, j]
            if not np.isfinite(r) or abs(r) >= CORR_COPIA or abs(r) < CORR_MIN:
                continue
            if mejor is None or abs(r) > abs(mejor[2]):
                mejor = (cols[i], cols[j], float(r))
    return mejor


def aplicabilidad(filas: list[dict]) -> dict[str, str | None]:
    """Por tipo: None si aplica, o el MOTIVO por el que este conjunto no lo tiene."""
    return {"extremo": None if _numericas(filas) else "sin columnas numericas completas con >= 10 valores distintos",
            "combinacion": None if pareja_mas_correlacionada(filas) else f"ninguna pareja numerica con {CORR_MIN} <= |r| < {CORR_COPIA}",
            "rara": None if _categoricas(filas) else "sin columnas categoricas (2 a 30 niveles)"}


def n_sembradas(n: int, tasa: float) -> int:
    return max(MIN_SEMBRADAS, round(tasa * n))


def sembrar(filas: list[dict], tipo: str, tasa: float, repeticion: int, nombre: str = "") -> tuple[list[dict], list[int], dict]:
    """Copia de `filas` con `n_sembradas` filas SUSTITUIDAS por anomalías de `tipo`. Devuelve (filas nuevas, índices sembrados 0-based
    ordenados, detalle de lo hecho). Determinista en (nombre, tipo, tasa, repeticion)."""
    if tipo not in TIPOS:
        raise ValueError(tipo)
    rng = random.Random(_semilla(nombre, tipo, tasa, repeticion))
    n = len(filas)
    ns = n_sembradas(n, tasa)
    elegidas = sorted(rng.sample(range(n), ns))
    nuevas = [dict(f) for f in filas]
    detalle: dict[str, Any] = {"tipo": tipo, "n_sembradas": ns}
    if tipo == "extremo":
        nums = _numericas(filas)
        if not nums:
            raise ValueError("sin numericas")
        cols = sorted(nums)
        usadas = []
        for i in elegidas:
            c = rng.choice(cols)
            x = nums[c]
            nuevas[i][c] = repr(float(x.max() + 0.5 * (x.max() - x.min())))
            usadas.append(c)
        detalle["columnas"] = usadas
    elif tipo == "combinacion":
        par = pareja_mas_correlacionada(filas)
        if par is None:
            raise ValueError("sin pareja")
        a, b, r = par
        nums = _numericas(filas)
        xa, xb = nums[a], nums[b]
        detalle.update(pareja=[a, b], r=r)
        for i in elegidas:
            qa = rng.uniform(85, 95)
            qb = rng.uniform(5, 15) if r > 0 else rng.uniform(85, 95)
            nuevas[i][a] = repr(float(np.percentile(xa, qa)))
            nuevas[i][b] = repr(float(np.percentile(xb, qb)))
    else:
        cats = _categoricas(filas)
        if not cats:
            raise ValueError("sin categoricas")
        c = rng.choice(sorted(cats))
        detalle["columna"] = c
        for i in elegidas:
            nuevas[i][c] = f"valor_nuevo_{i}"
    return nuevas, elegidas, detalle


# ---------------------------------------------------------------- el detector, con variantes de defectos
def _columnas_usadas(filas: list[dict]) -> tuple[list[str], dict[str, str]]:
    """LA MISMA exclusión que `puntuar_filas` (reutiliza sus funciones; no la copia)."""
    filas = [{c: _an._decimal(v) for c, v in f.items()} for f in filas]
    pedidas = list(filas[0])
    n = len(filas)
    t_prev = tipificar(filas, pedidas).tipos
    cand = [c for c in pedidas if not _an._excluir(c, [f.get(c) for f in filas], t_prev[c], n)]
    if not cand:
        raise ValueError("ninguna columna util")
    return cand, dict(tipificar(filas, cand).tipos)


def _rango_medio(x: np.ndarray) -> np.ndarray:
    s = np.sort(x)
    izq, der = np.searchsorted(s, x, "left"), np.searchsorted(s, x, "right")
    return ((izq + der - 1) / 2.0) / max(1, len(x) - 1)


def matriz(filas: list[dict], variante: dict) -> np.ndarray:
    filas = [{c: _an._decimal(v) for c, v in f.items()} for f in filas]
    usadas, tipos = _columnas_usadas(filas)
    cols = []
    for c in usadas:
        if tipos[c] == "numerica":
            v = np.array([float(f[c]) if es_numero(f.get(c)) else np.nan for f in filas])
            cols.append(np.where(np.isnan(v), float(np.nanmedian(v)), v))
            continue
        txt = [None if es_ausente(f.get(c)) else str(f.get(c)) for f in filas]
        niveles = sorted({t for t in txt if t is not None})
        if variante["categoricas"] == "frecuencia":
            cont = {k: sum(1 for t in txt if t == k) for k in niveles}
            tot = sum(cont.values())
            cols.append(np.array([cont[t] / tot if t is not None else 1.0 for t in txt]))
        elif variante["categoricas"] == "ordinal":
            cod = {k: i for i, k in enumerate(niveles)}
            cols.append(np.array([float(cod[t]) if t is not None else -1.0 for t in txt]))
        else:
            for k in niveles:
                cols.append(np.array([1.0 if t == k else 0.0 for t in txt]))
    X = np.column_stack(cols)
    if variante["escalado"] == "cuantiles":
        X = np.column_stack([_rango_medio(X[:, j]) for j in range(X.shape[1])])
    return X


def ordenar(filas: list[dict], variante: dict, semilla: int = 0, *, contamination: Any = None) -> np.ndarray:
    """Índices de fila (0-based) del más raro al menos raro (desempate por índice). Mismo detector que el producto."""
    from sklearn.ensemble import IsolationForest
    X = matriz(filas, variante)
    ms = "auto" if variante["max_samples"] == "auto" else len(X)
    kw = {} if contamination is None else {"contamination": contamination}
    m = IsolationForest(n_estimators=200, max_samples=ms, random_state=semilla, n_jobs=1, **kw).fit(X)
    p = -m.score_samples(X)
    return np.lexsort((np.arange(len(p)), -p))


def puntuaciones(filas: list[dict], variante: dict, semilla: int = 0, *, contamination: Any = None) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    X = matriz(filas, variante)
    kw = {} if contamination is None else {"contamination": contamination}
    m = IsolationForest(n_estimators=200, max_samples="auto" if variante["max_samples"] == "auto" else len(X), random_state=semilla,
                        n_jobs=1, **kw).fit(X)
    return -m.score_samples(X)


def metricas(orden: Sequence[int], sembradas: Sequence[int]) -> dict:
    """recuperación@k y precisión@k para k = ns/2, ns, 2ns (⌈·⌉, mínimo 1) y AP de lo sembrado sobre todo el orden."""
    s = set(int(i) for i in sembradas)
    ns = len(s)
    esta = np.array([int(i) in s for i in orden])
    acum = np.cumsum(esta)
    out: dict[str, Any] = {"n_sembradas": ns}
    for etiqueta, k in (("medio", max(1, math.ceil(ns / 2))), ("uno", ns), ("doble", min(len(orden), 2 * ns))):
        hits = int(acum[k - 1])
        out[f"k_{etiqueta}"] = k
        out[f"recuperacion_{etiqueta}"] = hits / ns
        out[f"precision_{etiqueta}"] = hits / k
    pos = np.nonzero(esta)[0]
    out["ap"] = float(np.mean([(i + 1) / (p + 1) for i, p in enumerate(pos)])) if len(pos) else 0.0
    return out


def control_hoy_igual_al_producto(filas: list[dict]) -> bool:
    """La variante «hoy» da el MISMO orden que `puntuar_filas` del producto (semilla 0)."""
    res = _an.puntuar_filas(filas, None, 0, n_semillas=2)
    producto = [e["fila"] - 1 for e in res["filas_a_revisar"]]
    return [int(i) for i in ordenar(filas, HOY_VARIANTE)] == producto


def control_contamination_invariante(filas: list[dict]) -> bool:
    a = puntuaciones(filas, HOY_VARIANTE, 0)
    b = puntuaciones(filas, HOY_VARIANTE, 0, contamination=0.05)
    return bool(np.array_equal(a, b))


HOY_VARIANTE = dict(HOY)
