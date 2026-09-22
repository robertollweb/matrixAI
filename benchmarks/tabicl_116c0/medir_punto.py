# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C0 — UN punto de la rejilla, dentro del contenedor y en un proceso nuevo.

Imprime una línea JSON con los tiempos y la memoria pico del PROCESO (`VmHWM` de
`/proc/self/status`): leída tras ajustar (el pico hasta ahí) y al final (el pico
total, que incluye predecir). No `memory.peak` del cgroup: con el núcleo de esta
máquina no se pone a cero entre puntos.

Sin red: los pesos se montan en `/pesos` (`HF_HOME`) y `HF_HUB_OFFLINE=1`.

    python3 medir_punto.py --fuente sintetico --filas 1000 --columnas 32 --lotes 1,32,256,todas
"""
from __future__ import annotations

import argparse
import json
import random
import time


def vmhwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("VmHWM:"):
                return int(linea.split()[1]) / 1024.0
    raise RuntimeError("sin VmHWM en /proc/self/status")


def datos(fuente: str, filas: int, columnas: int, n_test: int, semilla: int = 0):
    """Matrices numéricas `X` e `y`. `sintetico`: binaria, gaussianas con una frontera
    lineal; `optdigits`/`adult`: submuestreo DETERMINISTA de filas y columnas del ARFF
    montado en /datos (las categóricas de adult, codificadas por orden de aparición)."""
    import numpy as np
    rng = np.random.default_rng(semilla)
    if fuente == "sintetico":
        X = rng.normal(size=(filas + n_test, columnas))
        w = rng.normal(size=columnas)
        y = (X @ w + rng.normal(scale=0.5, size=filas + n_test) > 0).astype(int)
        return X[:filas], y[:filas], X[filas:], y[filas:]
    from scipy.io import arff
    ruta = {"optdigits": "/datos/28.arff", "adult": "/datos/1590.arff"}[fuente]
    crudo, meta = arff.loadarff(ruta)
    nombres = meta.names()
    objetivo = nombres[-1]
    predictores = [n for n in nombres if n != objetivo][:columnas]
    codigos: dict[str, dict] = {}

    def num(nombre, valor):
        if isinstance(valor, (bytes, bytearray)):
            texto = valor.decode()
            return float(codigos.setdefault(nombre, {}).setdefault(texto, len(codigos[nombre])))
        return float(valor)
    X = np.array([[num(n, r[n]) for n in predictores] for r in crudo])
    # Las etiquetas REALES: optdigits es multiclase (10 clases), y fue en multiclase
    # donde el piloto del 112 tumbó el contenedor; binarizarlo no mediría eso.
    etiquetas = [r[objetivo] for r in crudo]
    clases = {e: i for i, e in enumerate(sorted(set(etiquetas)))}
    y = np.array([clases[e] for e in etiquetas])
    orden = rng.permutation(len(X))
    X, y = X[orden], y[orden]
    return X[:filas], y[:filas], X[filas:filas + n_test], y[filas:filas + n_test]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fuente", required=True)
    p.add_argument("--filas", type=int, required=True)
    p.add_argument("--columnas", type=int, required=True)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--lotes", default="1,32,256,todas")
    a = p.parse_args()
    t0 = time.perf_counter()
    from tabicl import TabICLClassifier
    t_import = time.perf_counter() - t0
    X, y, Xt, _ = datos(a.fuente, a.filas, a.columnas, a.n_test)
    modelo = TabICLClassifier()
    t1 = time.perf_counter()
    modelo.fit(X, y)
    t_ajuste = time.perf_counter() - t1
    pico_ajuste = vmhwm_mb()
    t2 = time.perf_counter()
    modelo.predict_proba(Xt[:1])
    frio = time.perf_counter() - t2
    ms_por_fila = {}
    for lote in a.lotes.split(","):
        tam = len(Xt) if lote == "todas" else int(lote)
        # Con lote 1 se miden 64 filas, no las 512: el número es por fila, y
        # 512 llamadas de una fila no dirían nada más, solo tardarían más.
        n = min(len(Xt), 64) if tam == 1 else len(Xt)
        t = time.perf_counter()
        for i in range(0, n, tam):
            modelo.predict_proba(Xt[i:i + tam])
        ms_por_fila[lote] = round(1000 * (time.perf_counter() - t) / n, 3)
    print(json.dumps({"fuente": a.fuente, "filas": a.filas, "columnas": a.columnas,
                      "columnas_reales": int(X.shape[1]), "segundos_import": round(t_import, 2),
                      "segundos_ajuste": round(t_ajuste, 2), "primera_prediccion_s": round(frio, 3),
                      "ms_por_fila": ms_por_fila, "vmhwm_tras_ajuste_mb": round(pico_ajuste, 1),
                      "vmhwm_total_mb": round(vmhwm_mb(), 1)}), flush=True)


if __name__ == "__main__":
    main()
