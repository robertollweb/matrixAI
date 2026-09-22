# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C0 — recorre la rejilla registrada en el contrato 116, un contenedor por punto.

Protocolo (contrato 116, C0, registrado el 2026-09-22 antes de medir):
- `--memory=6g --memory-swap=6g --cpus=5`, UNA medición a la vez, carga ≤ 4 y ningún
  Chromium vivo; sin red (`--network none`): los pesos, montados.
- filas de contexto {500, 1.000, 2.000, 4.000, 8.000} × columnas {8, 32, 64};
  optdigits (64 columnas, multiclase: el que tumbó el piloto), adult (14 columnas: se
  mide con 8 y con todas) y una rejilla sintética.
- si un punto no cabe (el techo lo mata: código 137), los de MÁS filas con las mismas
  columnas y la misma fuente no se lanzan, y se anotan «no lanzado, por el anterior».

Antes, UNA vez y con red: `bash preparar.sh` (construye la imagen y baja los pesos).

    cd /home/deployer/matrixAI && python3 benchmarks/tabicl_116c0/orquestar.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

AQUI = Path(__file__).resolve().parent
IMAGEN = "medicion-tabicl-116c0:2.2.0"
PESOS = Path.home() / "tabicl_116c0_pesos"
DATOS = Path.home() / "fase0_openml_datos" / "arff"
RUTA_SALIDA = AQUI / "resultado_116c0.json"
FILAS = (500, 1000, 2000, 4000, 8000)
REJILLA = {"optdigits": (8, 32, 64), "adult": (8, 14), "sintetico": (8, 32, 64)}
#: Filas que tiene cada fuente (`None`: sintética, las que se pidan). Un punto que pide
#: más de las que hay (contexto + prueba) no se lanza y se dice (auditoría del 22-09:
#: optdigits tiene 5.620 y la rejilla le pedía 8.000).
FILAS_DISPONIBLES = {"optdigits": 5620, "adult": 48842, "sintetico": None}
N_TEST = 512
CARGA_MAXIMA = 4.0
TOPE_POR_PUNTO_S = 3600


def puntos() -> list[tuple[str, int, int]]:
    return [(f, c, n) for f, cols in REJILLA.items() for c in cols for n in FILAS]


def recorrer(medir: Callable[[str, int, int], dict],
             guardar: Callable[[list[dict]], None] | None = None) -> list[dict]:
    """La rejilla con la regla de parada: tras un «no cabe», las filas mayores de la
    MISMA fuente y columnas no se lanzan. `guardar` se llama tras CADA punto: si algo
    corta la medición a medias, lo medido hasta ahí queda escrito."""
    registros = []
    caidos: set[tuple[str, int]] = set()
    for fuente, columnas, filas in puntos():
        base = {"fuente": fuente, "columnas": columnas, "filas": filas}
        disponibles = FILAS_DISPONIBLES.get(fuente)
        if disponibles is not None and filas + N_TEST > disponibles:
            registros.append({**base, "estado": "no_hay_tantas_filas",
                              "filas_disponibles": disponibles})
        elif (fuente, columnas) in caidos:
            registros.append({**base, "estado": "no_lanzado_por_el_anterior"})
        else:
            r = medir(fuente, columnas, filas)
            registros.append({**base, **r})
            if r["estado"] == "no_cabe":
                caidos.add((fuente, columnas))
        if guardar is not None:
            guardar(registros)
    return registros


def _carga() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def _hay_chromium() -> bool:
    salida = subprocess.run(["docker", "ps", "--format", "{{.Image}}"], capture_output=True,
                            text=True).stdout
    return "playwright" in salida


def medir_en_docker(fuente: str, columnas: int, filas: int) -> dict:
    while _carga() > CARGA_MAXIMA or _hay_chromium():
        time.sleep(30)
    nombre = f"medicion-116c0-{fuente}-{columnas}-{filas}"
    orden = ["docker", "run", "--rm", "--name", nombre, "--network", "none",
             "--memory=6g", "--memory-swap=6g",
             "--cpus=5", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
             "-e", "HF_HOME=/pesos", "-e", "HF_HUB_OFFLINE=1",
             "-v", f"{PESOS}:/pesos:ro", "-v", f"{DATOS}:/datos:ro", "-v", f"{AQUI}:/arnes:ro",
             IMAGEN, "python3", "/arnes/medir_punto.py", "--fuente", fuente,
             "--filas", str(filas), "--columnas", str(columnas)]
    inicio = time.perf_counter()
    try:
        r = subprocess.run(orden, capture_output=True, text=True, timeout=TOPE_POR_PUNTO_S)
    except subprocess.TimeoutExpired:
        # Matar el CONTENEDOR, no solo el cliente de docker: si no, sigue vivo
        # comiéndose la memoria de la máquina (auditoría del 22-09).
        subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)
        return {"estado": "tope_agotado", "tope_s": TOPE_POR_PUNTO_S}
    pared = round(time.perf_counter() - inicio, 1)
    if r.returncode == 137:
        return {"estado": "no_cabe", "codigo": 137, "pared_s": pared}
    if r.returncode != 0:
        return {"estado": "fallo", "codigo": r.returncode, "pared_s": pared,
                "error": r.stderr[-1500:]}
    return {"estado": "medido", "pared_s": pared, **json.loads(r.stdout.strip().splitlines()[-1])}


def main() -> None:
    creado = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def guardar(registros: list[dict]) -> None:
        RUTA_SALIDA.write_text(json.dumps({
            "creado": creado, "imagen": IMAGEN,
            "techo": "--memory=6g --memory-swap=6g --cpus=5 --network none",
            "completa": len(registros) == len(puntos()), "registros": registros},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    recorrer(medir_en_docker, guardar)
    print(RUTA_SALIDA)


if __name__ == "__main__":
    main()
