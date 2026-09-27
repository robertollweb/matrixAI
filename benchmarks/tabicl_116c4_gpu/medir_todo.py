# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C4 — la rejilla ENTERA, dentro de UN contenedor y con un proceso nuevo por punto.

Para el PC con GPU de Roberto (decisión del 27-09: TabICL primero en el paquete GPU, medido en
una GPU de verdad antes de meterlo en el estudio). No hace falta nada más que Docker: este guion
corre DENTRO de la imagen y lanza `medir_punto.py` una vez por punto, cada uno en un proceso
nuevo — su memoria pico (RAM y GPU) es la suya, no la acumulada de los anteriores.

- Rejilla: tareas {binaria, multiclase, regresión} × columnas {8, 32, 64, 128} × filas de
  contexto {1.000, 2.000, 4.000, 8.000, 16.000, 32.000}. En CPU el techo decidido es 2.000 × 32;
  lo que se quiere saber es hasta dónde llega una GPU, y cuánto cuesta un intento.
- Misma regla que 116-C0: si un punto no termina (fallo, memoria o su tope de tiempo), los de
  MÁS filas con la misma tarea y columnas no se lanzan, y se anotan «no lanzado, por el anterior».
- Se guarda tras CADA punto, y al relanzar se reaprovecha lo ya medido: se puede cortar y seguir.

    python3 medir_todo.py --salida /salida/resultado_116c4_gpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

AQUI = Path(__file__).resolve().parent
TAREAS = ("binaria", "multiclase", "regresion")
COLUMNAS = (8, 32, 64, 128)
FILAS = (1000, 2000, 4000, 8000, 16000, 32000)
REJILLA_DE_PRUEBA = {"tareas": TAREAS, "columnas": (8,), "filas": (200,)}
TOPE_POR_PUNTO_S = 900


def puntos(tareas=TAREAS, columnas=COLUMNAS, filas=FILAS) -> list[tuple[str, int, int]]:
    """En orden: por tarea y columnas, de MENOS a MÁS filas (la regla de saltar lo mayor
    necesita ese orden)."""
    return [(t, c, f) for t in tareas for c in columnas for f in sorted(filas)]


def clave(p: dict | tuple) -> tuple[str, int, int]:
    return (p["tarea"], p["columnas"], p["filas"]) if isinstance(p, dict) else p


def recorrer(lista: list[tuple[str, int, int]], medir: Callable[[str, int, int], dict],
             previos: list[dict] | None = None,
             guardar: Callable[[list[dict]], None] | None = None) -> list[dict]:
    """Mide cada punto (o reaprovecha el ya MEDIDO de `previos`); tras uno que no termina,
    los mayores de su misma tarea y columnas no se lanzan."""
    ya = {clave(p): p for p in (previos or []) if p.get("estado") == "medido"}
    hechos: list[dict] = []
    roto: set[tuple[str, int]] = set()
    for tarea, columnas, filas in lista:
        if (tarea, columnas) in roto:
            hechos.append({"tarea": tarea, "columnas": columnas, "filas": filas,
                           "estado": "no_lanzado", "motivo": "no lanzado, por el anterior"})
        elif (tarea, columnas, filas) in ya:
            hechos.append(ya[(tarea, columnas, filas)])
        else:
            r = medir(tarea, columnas, filas)
            hechos.append(r)
            if r.get("estado") != "medido":
                roto.add((tarea, columnas))
        if guardar:
            guardar(hechos)
    return hechos


def medir_en_un_proceso(pesos: str, dispositivo: str,
                        tope_s: float = TOPE_POR_PUNTO_S) -> Callable[[str, int, int], dict]:
    def medir(tarea: str, columnas: int, filas: int) -> dict:
        base = {"tarea": tarea, "columnas": columnas, "filas": filas}
        orden = [sys.executable, str(AQUI / "medir_punto.py"), "--tarea", tarea,
                 "--filas", str(filas), "--columnas", str(columnas),
                 "--dispositivo", dispositivo, "--pesos", pesos]
        t = time.perf_counter()
        try:
            r = subprocess.run(orden, capture_output=True, text=True, timeout=tope_s)
        except subprocess.TimeoutExpired:
            return {**base, "estado": "tope", "motivo": f"pasó de {tope_s:.0f} s"}
        segundos = round(time.perf_counter() - t, 2)
        lineas = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode != 0 or not lineas:
            return {**base, "estado": "fallo", "codigo": r.returncode, "segundos_proceso": segundos,
                    "motivo": (r.stderr or r.stdout)[-800:]}
        # `segundos_proceso` es el proceso ENTERO, barrido de lotes incluido (el de una fila solo
        # ya son ~50 s en CPU): NO es lo que cuesta un intento del estudio. Ese es
        # `segundos_de_un_intento` (importar + ajustar + predecir la validación). Medido el 27-09:
        # confundirlos hacía leer 131 s donde el intento costaba 8,4.
        return {**base, **json.loads(lineas[-1]), "estado": "medido", "segundos_proceso": segundos}
    return medir


def maquina() -> dict:
    import torch  # noqa: PLC0415
    datos = {"python": platform.python_version(), "torch": torch.__version__,
             "cuda_de_torch": torch.version.cuda, "hay_gpu": torch.cuda.is_available(),
             "cpus": os.cpu_count()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        datos.update({"gpu": props.name, "vram_total_mb": round(props.total_memory / 2**20),
                      "n_gpus": torch.cuda.device_count()})
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            total = next(l for l in f if l.startswith("MemTotal:"))
        datos["ram_total_mb"] = round(int(total.split()[1]) / 1024)
    except (OSError, StopIteration):
        pass
    try:
        from importlib.metadata import version  # noqa: PLC0415 — `tabicl` no trae `__version__`
        datos["tabicl"] = version("tabicl")
    except Exception:  # noqa: BLE001 — sin el paquete no hay versión que decir
        datos["tabicl"] = None
    # Lo instalado DE VERDAD en la imagen (pip freeze al construirla), no lo que dice el Dockerfile.
    versiones = AQUI / "versiones_de_la_imagen.txt"
    if versiones.exists():
        datos["versiones_de_la_imagen"] = versiones.read_text(encoding="utf-8").splitlines()
    return datos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--salida", default="/salida/resultado_116c4_gpu.json")
    p.add_argument("--dispositivo", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--pesos", default="/pesos")
    p.add_argument("--prueba", action="store_true", help="rejilla mínima, para ver que el kit anda")
    a = p.parse_args()
    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    previos = json.loads(salida.read_text())["puntos"] if salida.exists() else []
    lista = puntos(**REJILLA_DE_PRUEBA) if a.prueba else puntos()
    # SIN GPU, LA REJILLA ENTERA NO: en CPU un punto pequeño ya cuesta minutos (medido el 27-09)
    # y la rejilla entera serían días. Casi siempre es un `--gpus all` olvidado o un driver sin
    # configurar, y es mejor decirlo en el primer segundo que tras una noche en CPU.
    if not a.prueba and a.dispositivo == "auto":
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            raise SystemExit("No veo ninguna GPU. ¿Falta `--gpus all` en `docker run`, o el driver "
                             "de NVIDIA / el toolkit de contenedores? (Para forzar la CPU a propósito: "
                             "--dispositivo cpu; para probar el kit: --prueba.)")
    cabecera = {"kit": "116-C4 (medición en GPU)", "creado": datetime.now(timezone.utc).isoformat(),
                "maquina": maquina(), "tope_por_punto_s": TOPE_POR_PUNTO_S,
                "rejilla": "prueba" if a.prueba else "completa", "dispositivo_pedido": a.dispositivo}
    print(json.dumps(cabecera["maquina"], ensure_ascii=False), flush=True)

    def guardar(hechos: list[dict]) -> None:
        temporal = salida.with_suffix(".parcial")
        temporal.write_text(json.dumps({**cabecera, "puntos": hechos}, ensure_ascii=False, indent=1))
        temporal.replace(salida)
        ultimo = hechos[-1]
        print(f"[{len(hechos)}/{len(lista)}] {ultimo['tarea']} {ultimo['columnas']} col × "
              f"{ultimo['filas']} filas: {ultimo['estado']}"
              + (f", un intento {ultimo.get('segundos_de_un_intento')} s (proceso entero, con el "
                 f"barrido de lotes: {ultimo.get('segundos_proceso')} s)" if ultimo["estado"] == "medido" else ""),
              flush=True)

    recorrer(lista, medir_en_un_proceso(a.pesos, a.dispositivo), previos, guardar)
    print(f"hecho: {salida}", flush=True)


if __name__ == "__main__":
    main()
