#!/usr/bin/env python3
"""128-C5 — corre DENTRO del contenedor del Studio (lo lanza `lanzar.*` con `docker exec -d`): cada ~2 s escribe una
linea JSON con la hora, la RAM del contenedor (cgroup) y, si hay `nvidia-smi`, la VRAM usada y el uso de la GPU.
`medir_c5.py` toma los picos de cada estudio de ahi. Solo biblioteca estandar. Se para solo cuando se borra el contenedor."""
import json
import shutil
import subprocess
import sys
import time

CGROUP = ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")


def ram_mb():
    for ruta in CGROUP:
        try:
            with open(ruta) as f:
                return round(int(f.read().strip()) / 2 ** 20, 1)
        except (OSError, ValueError):
            continue
    return None


def gpu():
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        mem, uso = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return mem, uso
    except Exception:                              # noqa: BLE001 — una muestra perdida no para nada
        return None, None


def main(destino, cada=2.0):
    with open(destino, "a", buffering=1) as f:
        while True:
            vram, uso = gpu()
            f.write(json.dumps({"t": round(time.time(), 2), "ram_mb": ram_mb(), "vram_mb": vram, "gpu_util": uso}) + "\n")
            time.sleep(cada)


if __name__ == "__main__":
    main(sys.argv[1])
