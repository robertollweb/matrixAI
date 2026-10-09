#!/usr/bin/env python3
"""128 · C4 — el control de los conjuntos donde TabM (o los árboles) paran por RELOJ: ¿el C4 cae dentro de la dispersión
de la v2.12? La regla está escrita en el contrato (09-10) ANTES del control:

    python3 rango_c4_128.py --v212 referencia_128_pcpu_v212.json control_tabm_v212_1.json control_tabm_v212_2.json \
                            --c4 c4_128_pcpu.json control_tabm_c4_1.json control_tabm_c4_2.json --solo house_16H,mfeat-factors

Por conjunto, la cifra del test (la de `veredicto_120_c3p.cifra_de_test`: el campeón, el test independiente, ×100) de cada
pasada de la v2.12 y de cada una del C4:
- **dentro**: el rango del C4 [mín, máx] solapa el de la v2.12 (o cae dentro): es la dispersión del producto, no el
  contrato.
- **empeora**: TODAS las del C4 quedan por debajo de la PEOR de la v2.12. Se busca por qué antes de seguir.
- **mejora**: todas las del C4 por encima de la mejor de la v2.12 (se dice; no es lo que se buscaba aquí).
- Un conjunto que no completa en alguna pasada lo dice aparte: dejar de terminar no es dispersión.

Código de salida: 0 si ninguno empeora ni deja de terminar, 1 si alguno.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def _cargar_por_ruta(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v3p = _cargar_por_ruta("veredicto_120_c3p", AQUI / "veredicto_120_c3p.py")


def cifras(docs: list[dict], nombre: str) -> list[dict]:
    out = []
    for d in docs:
        rec = (d.get("conjuntos") or {}).get(nombre)
        if rec is None:
            continue
        out.append({"campeon": rec.get("campeon"), "completa": v3p.completa(rec),
                    "test": v3p.cifra_de_test(rec) if v3p.completa(rec) else None})
    return out


def clasificar(v212: list[dict], c4: list[dict]) -> dict:
    if not v212 or not c4:
        return {"clase": "sin_pasadas", "v212": v212, "c4": c4}
    if not all(x["completa"] for x in v212 + c4):
        return {"clase": "no_completa_en_alguna", "v212": v212, "c4": c4}
    a = [x["test"] for x in v212]
    b = [x["test"] for x in c4]
    if max(b) < min(a):
        clase = "empeora"
    elif min(b) > max(a):
        clase = "mejora"
    else:
        clase = "dentro"
    return {"clase": clase, "v212": v212, "c4": c4, "rango_v212": [min(a), max(a)], "rango_c4": [min(b), max(b)]}


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v212", nargs="+", required=True)
    ap.add_argument("--c4", nargs="+", required=True)
    ap.add_argument("--solo", required=True, help="conjuntos separados por comas")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    v212 = [json.loads(Path(p).read_text()) for p in a.v212]
    c4 = [json.loads(Path(p).read_text()) for p in a.c4]
    res, malo = {}, False
    for n in [x for x in a.solo.split(",") if x]:
        r = clasificar(cifras(v212, n), cifras(c4, n))
        res[n] = r
        malo |= r["clase"] in ("empeora", "no_completa_en_alguna", "sin_pasadas")
        fmt = lambda xs: ", ".join(f"{x['test']:.4f} ({str(x['campeon']).split('.')[-1]})" if x["test"] is not None
                                   else f"— ({x['campeon']})" for x in xs)
        print(f"{n:28} {r['clase']:22} v2.12: {fmt(r['v212'])}  |  C4: {fmt(r['c4'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("VEREDICTO:", "alguno EMPEORA o no completa" if malo else "dentro de la dispersión de la v2.12 (o mejor)")
    return 1 if malo else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
