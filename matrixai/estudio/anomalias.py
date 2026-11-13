# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""«Filas a revisar» (contrato 165, C1): un CSV SIN columna objetivo -> lista ORDENADA de filas a revisar.

Solo núcleo, sin interfaz ni recibo. Qué promete y qué NO:

  * Una puntuación es un ORDEN (rango), no una probabilidad ni un veredicto. «Rara» = rara respecto a ESTE CSV, no «mala».
  * NINGUNA cifra de acierto: sin etiquetas no existe exactitud, sensibilidad ni «n anomalías». La salida no trae esas
    claves y `no_afirma` lo dice. Lo que sí trae: estabilidad medida (misma lista con otras semillas).
  * No pasa por `dataset_project` (que descarta las filas sin objetivo): aquí no hay objetivo y ninguna fila se pierde en
    silencio; las ignoradas (sin ningún dato en las columnas usadas) se declaran con su número.
  * Tipos: `matrixai_engines.tipos_de_columna.tipificar` (un solo sitio). Excluidas y DICHAS: identificadores, constantes,
    casi vacías, y categóricas de cardinalidad casi igual al número de filas (dominan el aislamiento: «rara = única»).
  * Detector: IsolationForest de scikit-learn, sin `contamination` (no se estima ni se inventa). Categóricas por
    frecuencia; faltantes numéricos por la mediana (declarado). Nada se corrige ni se borra: se señala.
  * «Columnas a la cola» por fila: DESCRIPTIVO (valor en la cola de su distribución o categoría rara), no la razón del algoritmo.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from typing import Any, Sequence

import numpy as np

from matrixai_engines.tipos_de_columna import es_ausente, es_numero, tipificar

VERSION_DEL_FORMATO = 1
DETECTOR = "IsolationForest"
N_SEMILLAS = 5
MIN_FILAS_FIABLES = 200
UMBRAL_VACIA = 0.95          # proporción de ausentes a partir de la cual una columna no informa
UMBRAL_UNICA = 0.9           # distintos / presentes: a partir de aquí una categórica identifica filas
COLA_NUMERICA = 0.025        # percentil de cada extremo
COLA_CATEGORIA = 0.01        # frecuencia relativa por debajo de la cual una categoría es rara
_COMA = re.compile(r"^\s*[-+]?\d{1,3}(\.\d{3})*,\d+\s*$|^\s*[-+]?\d+,\d+\s*$")

NO_AFIRMA = [
    "No hay etiquetas: no se calcula ni se afirma exactitud, sensibilidad, AUROC ni «porcentaje de anomalías detectadas».",
    "La puntuación es un ORDEN (rango), no una probabilidad ni un veredicto: «rara» significa rara respecto a este CSV.",
    "Una fila a revisar no es un error ni un fraude: no se borra ni se corrige, se señala.",
    "El número de filas realmente anómalas se desconoce; no se estima ninguna proporción (no hay `contamination`).",
    "Las columnas a la cola son una descripción de qué valores son extremos o raros, no la razón interna del detector.",
    "La estabilidad mide solo cuánto cambia la lista con la semilla, no si la lista es correcta.",
]


def leer_csv(ruta: str, *, delimitador: str | None = None) -> list[dict[str, str]]:
    """Filas como dicts de texto, tal cual (sin interpretar). Detecta `;` o `,` si no se dice."""
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        muestra = f.read(8192)
        f.seek(0)
        if delimitador is None:
            delimitador = ";" if muestra.split("\n", 1)[0].count(";") > muestra.split("\n", 1)[0].count(",") else ","
        return [dict(r) for r in csv.DictReader(f, delimiter=delimitador)]


def _decimal(valor: Any) -> Any:
    """`'1,5'` -> `'1.5'` (coma decimal), solo si la celda ES eso; lo demás, intacto."""
    if isinstance(valor, str) and _COMA.match(valor):
        return valor.strip().replace(".", "").replace(",", ".")
    return valor


def _a_float(valor: Any) -> float | None:
    return float(valor) if es_numero(valor) else None


def _excluir(col: str, valores: list[Any], tipo: str, n: int) -> str | None:
    presentes = [v for v in valores if not es_ausente(v)]
    if not presentes:
        return "vacia: ningun valor presente"
    if (n - len(presentes)) / n >= UMBRAL_VACIA:
        return f"casi vacia: {n - len(presentes)} de {n} filas sin dato"
    distintos = {str(v) for v in presentes}
    if len(distintos) <= 1:
        return "constante: un solo valor, no distingue filas"
    if tipo == "categorica" and len(presentes) >= 5 and len(distintos) / len(presentes) >= UMBRAL_UNICA:
        return (f"identificador o casi unica: {len(distintos)} valores distintos en {len(presentes)} filas "
                f"(dominaria el aislamiento: «rara» seria «unica»)")
    if tipo == "numerica" and len(presentes) >= 5:
        nums = [float(v) for v in presentes]
        if all(x == int(x) for x in nums) and len(distintos) == len(nums):
            ordenados = sorted(nums)
            if ordenados[-1] - ordenados[0] <= 1.5 * len(nums):
                return "identificador: enteros todos distintos y consecutivos (un contador de fila)"
    return None


def _matriz(filas: list[dict], usadas: list[str], tipos: dict[str, str]):
    n = len(filas)
    cols, frecs = [], {}
    for c in usadas:
        if tipos[c] == "numerica":
            v = np.array([_a_float(f.get(c)) if _a_float(f.get(c)) is not None else np.nan for f in filas], float)
            med = float(np.nanmedian(v))
            cols.append(np.where(np.isnan(v), med, v))
        else:
            txt = [None if es_ausente(f.get(c)) else str(f.get(c)) for f in filas]
            cont: dict[str, int] = {}
            for t in txt:
                if t is not None:
                    cont[t] = cont.get(t, 0) + 1
            total = sum(cont.values())
            frecs[c] = {k: v / total for k, v in cont.items()}
            cols.append(np.array([frecs[c].get(t, 0.0) if t is not None else 1.0 for t in txt], float))
    return np.column_stack(cols), frecs


def _puntuar(X: np.ndarray, semilla: int) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=200, random_state=semilla, n_jobs=1).fit(X)
    return -m.score_samples(X)          # mayor = mas raro


def _rangos(p: np.ndarray) -> np.ndarray:
    """Posicion 1..n de cada fila, 1 = la mas rara; desempate por indice de fila (determinista)."""
    orden = np.lexsort((np.arange(len(p)), -p))
    r = np.empty(len(p), int)
    r[orden] = np.arange(1, len(p) + 1)
    return r


def _a_la_cola(fila: dict, usadas, tipos, filas_num, frecs) -> list[dict]:
    out = []
    for c in usadas:
        v = fila.get(c)
        if es_ausente(v):
            continue
        if tipos[c] == "numerica":
            x = float(v)
            col = filas_num[c]          # ya ORDENADA (ver `puntuar_filas`): búsqueda binaria, no un barrido por fila
            n_col = len(col)
            le = float(np.searchsorted(col, x, side="right")) / n_col
            ge = float(n_col - np.searchsorted(col, x, side="left")) / n_col
            if le <= COLA_NUMERICA and x < col[-1]:
                out.append({"columna": c, "valor": str(v), "motivo": "cola baja", "percentil": round(100 * le, 2)})
            elif ge <= COLA_NUMERICA and x > col[0]:
                out.append({"columna": c, "valor": str(v), "motivo": "cola alta", "percentil": round(100 * (1 - ge), 2)})
        else:
            f = frecs[c].get(str(v), 0.0)
            if f < COLA_CATEGORIA:
                out.append({"columna": c, "valor": str(v), "motivo": "categoria rara", "frecuencia": round(f, 6)})
    return out


def puntuar_filas(filas: Sequence[dict], columnas: Sequence[str] | None = None, semilla: int = 0, *,
                  k: int | None = None, n_semillas: int = N_SEMILLAS) -> dict:
    """Lista ORDENADA de filas a revisar. `filas`: dicts (valores crudos o texto de CSV). `columnas`: las que se miran
    (por omisión todas). `k`: tamaño del corte con que se mide la estabilidad (por omisión max(1, 5 % de las filas), tope 50).
    Devuelve un dict JSON-serializable; no contiene ninguna cifra de acierto."""
    if n_semillas < 2:
        raise ValueError("la estabilidad necesita al menos 2 semillas")
    filas = [{c: _decimal(v) for c, v in f.items()} for f in filas]
    n_total = len(filas)
    if n_total < 2:
        raise ValueError("hacen falta al menos 2 filas")
    pedidas = list(columnas) if columnas is not None else list(filas[0].keys())
    faltan = [c for c in pedidas if c not in filas[0]]
    if faltan:
        raise ValueError(f"columnas que no existen en el CSV: {faltan}")
    excluidas, candidatas = [], []
    # tipo provisional solo para decidir exclusiones; el definitivo sale de `tipificar` sobre las que quedan
    t_prev = tipificar(filas, pedidas).tipos
    for c in pedidas:
        motivo = _excluir(c, [f.get(c) for f in filas], t_prev[c], n_total)
        if motivo:
            excluidas.append({"columna": c, "motivo": motivo})
        else:
            candidatas.append(c)
    if not candidatas:
        raise ValueError("ninguna columna util: todas excluidas (ver motivos): " + "; ".join(
            f"{e['columna']}: {e['motivo']}" for e in excluidas))
    tip = tipificar(filas, candidatas)
    tipos = dict(tip.tipos)
    usadas = list(candidatas)
    # filas sin ningun dato en las columnas usadas: no se pueden puntuar; se declaran, no se pierden en silencio
    con_dato = [i for i, f in enumerate(filas) if any(not es_ausente(f.get(c)) for c in usadas)]
    _con_dato = set(con_dato)  # una vez: dentro de la comprensión era cuadrático (51.000 filas)
    ignoradas = [i + 1 for i in range(n_total) if i not in _con_dato]
    base = [filas[i] for i in con_dato]
    X, frecs = _matriz(base, usadas, tipos)
    n = len(base)
    kk = k if k is not None else min(50, max(1, math.ceil(0.05 * n)))
    kk = min(kk, n)
    semillas = [semilla + i for i in range(n_semillas)]
    puntos = [_puntuar(X, s) for s in semillas]
    p = puntos[0]
    r = _rangos(p)
    tops = [set(np.lexsort((np.arange(n), -q))[:kk].tolist()) for q in puntos]
    pares = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(len(tops)) for j in range(i + 1, len(tops))]
    solape_min, solape_medio = min(pares), sum(pares) / len(pares)
    num = {c: np.sort(X[:, j]) for j, c in enumerate(usadas) if tipos[c] == "numerica"}
    orden = np.lexsort((np.arange(n), -p))
    lista = []
    for pos in orden:
        fila = base[int(pos)]
        lista.append({
            "rango": int(r[pos]),
            "fila": con_dato[int(pos)] + 1,
            "percentil": round(100.0 * (1 - (int(r[pos]) - 1) / n), 2),
            "puntuacion": round(float(p[pos]), 6),
            "columnas_a_la_cola": _a_la_cola(fila, usadas, tipos, num, frecs),
        })
    avisos = []
    if n < MIN_FILAS_FIABLES:
        avisos.append(f"Solo {n} filas: con menos de {MIN_FILAS_FIABLES} el detector no es fiable.")
    if solape_medio < 0.5:
        avisos.append("Esta lista cambia con la semilla: no te fies del orden fino.")
    if ignoradas:
        avisos.append(f"{len(ignoradas)} filas sin ningun dato en las columnas usadas no se puntuaron.")
    return {
        "formato": VERSION_DEL_FORMATO,
        "tipo": "filas_a_revisar",
        "etiquetas": "ninguna",
        "detector": {"nombre": DETECTOR, "semilla": semilla, "n_estimadores": 200, "contamination": None,
                     "codificacion": "numericas tal cual (faltante = mediana); categoricas por frecuencia relativa"},
        "datos": {"filas": n_total, "filas_puntuadas": n, "filas_ignoradas": ignoradas},
        "columnas_usadas": [{"columna": c, "tipo": tipos[c]} for c in usadas],
        "columnas_excluidas": excluidas,
        "estabilidad": {"semillas": semillas, "k": kk, "solape_medio": round(solape_medio, 4),
                        "solape_minimo": round(solape_min, 4),
                        "lectura": "Jaccard de las k primeras filas entre pares de semillas"},
        "avisos": avisos,
        "filas_a_revisar": lista,
        "no_afirma": list(NO_AFIRMA),
    }


def _main(argv: list[str]) -> int:
    import argparse
    a = argparse.ArgumentParser(prog="python -m matrixai.estudio.anomalias", description="Filas a revisar (sin etiquetas)")
    a.add_argument("csv")
    a.add_argument("--columnas", help="lista separada por comas; por omision, todas")
    a.add_argument("--semilla", type=int, default=0)
    a.add_argument("--k", type=int, default=None)
    a.add_argument("--salida", help="fichero JSON; por omision, stdout")
    o = a.parse_args(argv)
    cols = [c.strip() for c in o.columnas.split(",")] if o.columnas else None
    res = puntuar_filas(leer_csv(o.csv), cols, o.semilla, k=o.k)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if o.salida:
        with open(o.salida, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
