#!/usr/bin/env python3
"""128 · C3b — la TABLA para recalibrar la guarda de memoria de TabM: por conjunto, lo que la guarda ve (filas × columnas de
entrada), lo que se midió (el pico del cgroup con TabM, el del HIJO de TabM y el del servidor en sus intentos, y el pico
sin TabM) y lo que decidía la guarda de hoy. La regla está escrita en el contrato (09-10) ANTES de medir:

- **CABE** si el pico del cgroup con TabM no pasa del 90 % del techo y ningún proceso murió por memoria (`oom_kill` = 0).
- La guarda de hoy: `(1 + 0,5 × M celdas) × 1,15` GB frente al techo (`memoria_del_estudio.memoria_estimada_del_estudio_gb`,
  copiada aquí a propósito y comprobada contra la del Studio en la prueba: si cambia allí, la prueba lo dice).

    python3 tabla_c3b_128.py c3b_128_tabm_forzado_6g.json c3b_128_tabm_columnas_6g.json --sin-tabm c4_128_rcpu.json

Entrada: registros de la sonda en modo `--tabm-forzado` (`<conjunto>:tabm_forzado`), que traen `guarda_de_verdad` (las
filas y columnas que vio la guarda), `memory_events` y el resumen por fase; y, aparte, una pasada SIN TabM del instrumento
del 120 (`memoria_pico_mb` por conjunto) para la columna «sin TabM». No decide nada: enseña.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FRACCION_QUE_CABE = 0.90
MOTOR_TABM = "matrixai.dense.tabm_cpu"
GB_BASE, GB_POR_MILLON, MARGEN = 1.0, 0.5, 1.15          # la fórmula de hoy (ver la cabecera)
MB_POR_GB = 1024.0


def guarda_de_hoy_gb(filas: int, columnas: int) -> float:
    return (GB_BASE + GB_POR_MILLON * filas * columnas / 1_000_000) * MARGEN


def techo_mb(memoria: str) -> float:
    """«6g» → 6.144 MB; «1150m» → 1.150 MB (como `docker --memory`, en MiB)."""
    m = memoria.strip().lower()
    if m.endswith("g"):
        return float(m[:-1]) * 1024.0
    if m.endswith("m"):
        return float(m[:-1])
    raise ValueError(f"memoria sin unidad conocida: {memoria!r}")


def _pico_de_tabm(resumen: dict) -> tuple[float | None, float | None]:
    """(pico del hijo, pico del servidor) en las fases de TabM: sus intentos de la búsqueda (`intento:<tabm>`) y, si TabM
    entrenó el reajuste, el reajuste no se puede atribuir a un motor desde fuera: no entra aquí (se ve en el total)."""
    fases = (resumen or {}).get("por_fase") or {}
    f = fases.get(f"intento:{MOTOR_TABM}")
    if not f:
        return None, None
    return f.get("pico_hijo_mb"), f.get("pico_servidor_mb")


def fila(nombre: str, rec: dict, sin_tabm: dict | None) -> dict:
    guarda = (rec.get("guarda_de_verdad") or [{}])[0]
    filas, columnas = guarda.get("filas"), guarda.get("columnas_de_entrada")
    techo = techo_mb(rec.get("memoria_del_contenedor") or "6g")
    pico = rec.get("memoria_pico_mb")
    oom = (rec.get("memory_events") or {}).get("oom_kill")
    hijo, servidor = _pico_de_tabm(rec.get("resumen_de_memoria") or {})
    cabe = (rec.get("estado") == "completed" and pico is not None and pico <= FRACCION_QUE_CABE * techo and oom == 0)
    hoy = guarda_de_hoy_gb(filas, columnas) if filas and columnas else None
    return {"nombre": nombre, "filas": filas, "columnas": columnas,
            "celdas_M": round(filas * columnas / 1e6, 2) if filas and columnas else None,
            "estado": rec.get("estado"), "intentos_de_tabm": rec.get("intentos_de_tabm"), "campeon": rec.get("campeon"),
            "pico_con_tabm_mb": pico, "pico_sin_tabm_mb": (sin_tabm or {}).get("memoria_pico_mb"),
            "pico_hijo_tabm_mb": hijo, "pico_servidor_en_tabm_mb": servidor, "oom_kill": oom, "techo_mb": techo,
            "cabe": cabe, "guarda_de_hoy_gb": round(hoy, 2) if hoy is not None else None,
            "guarda_de_hoy_deja_entrar": (hoy * MB_POR_GB <= techo) if hoy is not None else None}


def tabla(docs: list[dict], sin_tabm_doc: dict | None) -> list[dict]:
    sin = (sin_tabm_doc or {}).get("conjuntos") or {}
    out = []
    for d in docs:
        for clave, rec in (d.get("conjuntos") or {}).items():
            if not clave.endswith(":tabm_forzado"):
                continue
            nombre = clave.split(":")[0]
            out.append(fila(nombre, rec, sin.get(nombre)))
    return sorted(out, key=lambda f: (f["columnas"] or 0, f["filas"] or 0))


def _fmt(x):
    return "—" if x is None else (f"{x:,.0f}".replace(",", ".") if isinstance(x, float) else str(x))


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sondas", nargs="+")
    ap.add_argument("--sin-tabm", default=None, help="una pasada SIN TabM del instrumento del 120 (c4_128_rcpu.json)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    docs = [json.loads(Path(p).read_text()) for p in a.sondas]
    t = tabla(docs, json.loads(Path(a.sin_tabm).read_text()) if a.sin_tabm else None)
    print(f"{'conjunto':28} {'filas':>8} {'col.':>5} {'M cel.':>7} {'estado':10} {'con TabM':>9} {'sin TabM':>9} "
          f"{'hijo TabM':>9} {'servidor':>9} {'oom':>4} {'¿cabe?':7} {'hoy GB':>7} {'hoy deja':8}")
    for f in t:
        print(f"{f['nombre'][:28]:28} {_fmt(f['filas']):>8} {_fmt(f['columnas']):>5} {('—' if f['celdas_M'] is None else f"{f['celdas_M']:.2f}"):>7} "
              f"{str(f['estado'])[:10]:10} {_fmt(f['pico_con_tabm_mb']):>9} {_fmt(f['pico_sin_tabm_mb']):>9} "
              f"{_fmt(f['pico_hijo_tabm_mb']):>9} {_fmt(f['pico_servidor_en_tabm_mb']):>9} {_fmt(f['oom_kill']):>4} "
              f"{'sí' if f['cabe'] else 'NO':7} {_fmt(f['guarda_de_hoy_gb']):>7} "
              f"{'entrar' if f['guarda_de_hoy_deja_entrar'] else 'fuera':8}")
    if a.json:
        Path(a.json).write_text(json.dumps(t, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
