# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C4 — UN punto de la rejilla, en un proceso nuevo, en la GPU si la hay.

Es el `medir_punto.py` de 116-C0 (CPU) con cuatro cosas más, las que el C4 necesita saber para
decidir si TabICL cabe en un estudio del paquete GPU (decisión de Roberto del 27-09):

- el DISPOSITIVO (`--dispositivo auto`: la GPU si torch la ve), y cuál se usó de verdad;
- la memoria de la GPU (`torch.cuda.max_memory_allocated`), además del pico de RAM del proceso;
- las TRES tareas del estudio (binaria, multiclase de 10 clases, regresión), con datos sintéticos
  y deterministas — el piloto del 112 tumbó el contenedor justo en multiclase;
- lo que cuesta un INTENTO de estudio: importar, ajustar y predecir una validación de un cuarto
  de las filas. En CPU (116-C2) fueron ~72 s, y la búsqueda entera de un estudio tiene 140 s para
  todos los motores: por eso hay que medirlo aquí.

Imprime UNA línea JSON. Sin red: los pesos, en `--pesos` (un directorio plano con los `.ckpt`).

    python3 medir_punto.py --tarea binaria --filas 2000 --columnas 32 --pesos /pesos
"""
from __future__ import annotations

import argparse
import json
import time

T_INICIO = time.perf_counter()

FICHEROS = {"clasificacion": "tabicl-classifier-v2-20260212.ckpt",
            "regresion": "tabicl-regressor-v2-20260212.ckpt"}


def vmhwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("VmHWM:"):
                return int(linea.split()[1]) / 1024.0
    raise RuntimeError("sin VmHWM en /proc/self/status")


def datos(tarea: str, filas: int, columnas: int, n_test: int, semilla: int = 0):
    """`X`, `y` de contexto y de prueba. Sintéticos y deterministas: una frontera lineal con
    ruido (binaria), el argmax de 10 puntuaciones lineales (multiclase) o una combinación lineal
    con ruido (regresión)."""
    import numpy as np  # noqa: PLC0415
    rng = np.random.default_rng(semilla)
    n = filas + n_test
    X = rng.normal(size=(n, columnas))
    if tarea == "binaria":
        y = (X @ rng.normal(size=columnas) + rng.normal(scale=0.5, size=n) > 0).astype(int)
    elif tarea == "multiclase":
        y = (X @ rng.normal(size=(columnas, 10)) + rng.normal(scale=0.5, size=(n, 10))).argmax(1)
    elif tarea == "regresion":
        y = X @ rng.normal(size=columnas) + rng.normal(scale=0.5, size=n)
    else:
        raise ValueError(f"tarea desconocida: {tarea}")
    return X[:filas], y[:filas], X[filas:], y[filas:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tarea", required=True, choices=("binaria", "multiclase", "regresion"))
    p.add_argument("--filas", type=int, required=True)
    p.add_argument("--columnas", type=int, required=True)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--lotes", default="1,32,256,todas")
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--pesos", default="/pesos")
    a = p.parse_args()

    t0 = time.perf_counter()
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from tabicl import TabICLClassifier, TabICLRegressor  # noqa: PLC0415
    t_import = time.perf_counter() - t0

    hay_gpu = torch.cuda.is_available()
    if a.dispositivo == "cuda" and not hay_gpu:
        raise SystemExit("se pidió --dispositivo cuda y torch no ve ninguna GPU")
    dispositivo = "cuda" if (a.dispositivo == "cuda" or (a.dispositivo == "auto" and hay_gpu)) else "cpu"
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()

    X, y, Xt, _ = datos(a.tarea, a.filas, a.columnas, a.n_test)
    clase, fichero = ((TabICLRegressor, FICHEROS["regresion"]) if a.tarea == "regresion"
                      else (TabICLClassifier, FICHEROS["clasificacion"]))
    modelo = clase(model_path=f"{a.pesos}/{fichero}", checkpoint_version=fichero,
                   allow_auto_download=False, device=dispositivo, random_state=0)
    predecir = modelo.predict if a.tarea == "regresion" else modelo.predict_proba

    t1 = time.perf_counter()
    modelo.fit(X, y)
    t_ajuste = time.perf_counter() - t1
    pico_ajuste = vmhwm_mb()

    t2 = time.perf_counter()
    predecir(Xt[:1])
    frio = time.perf_counter() - t2

    # LO QUE COSTARÍA UN INTENTO DEL ESTUDIO: predecir una validación de un cuarto de las filas
    # (el reparto del estudio deja ~20-25 % para validar). Se toma de la prueba, repitiendo
    # filas si hacen falta más de las que hay: aquí importa el tiempo, no la métrica.
    n_validacion = max(1, a.filas // 4)
    Xv = np.resize(Xt, (n_validacion, Xt.shape[1]))
    t3 = time.perf_counter()
    predecir(Xv)
    t_validacion = time.perf_counter() - t3

    ms_por_fila = {}
    for lote in a.lotes.split(","):
        tam = len(Xt) if lote == "todas" else int(lote)
        # Con lote 1 se miden 16 filas, no las 512: el número es por fila. (Eran 64 en 116-C0;
        # medido el 27-09 en CPU, cada fila suelta cuesta ~3,2 s y 64 eran 3 minutos por punto.)
        n = min(len(Xt), 16) if tam == 1 else len(Xt)
        t = time.perf_counter()
        for i in range(0, n, tam):
            predecir(Xt[i:i + tam])
        ms_por_fila[lote] = round(1000 * (time.perf_counter() - t) / n, 3)

    salida = {
        "tarea": a.tarea, "filas": a.filas, "columnas": a.columnas,
        "dispositivo_pedido": a.dispositivo, "dispositivo": dispositivo,
        "gpu": torch.cuda.get_device_name(0) if dispositivo == "cuda" else None,
        "segundos_import": round(t_import, 2), "segundos_ajuste": round(t_ajuste, 2),
        "primera_prediccion_s": round(frio, 3),
        "segundos_validacion": round(t_validacion, 2), "n_validacion": n_validacion,
        # Desde que arrancó ESTE guion (sin contar el arranque del intérprete, que mide
        # `medir_todo.py` por fuera): importar + ajustar + predecir la validación.
        "segundos_de_un_intento": round(t_import + t_ajuste + t_validacion, 2),
        "ms_por_fila": ms_por_fila,
        "vmhwm_tras_ajuste_mb": round(pico_ajuste, 1), "vmhwm_total_mb": round(vmhwm_mb(), 1),
        "vram_pico_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                         if dispositivo == "cuda" else None),
        "segundos_desde_el_inicio_del_guion": round(time.perf_counter() - T_INICIO, 2),
    }
    print(json.dumps(salida), flush=True)


if __name__ == "__main__":
    main()
