#!/usr/bin/env python3
"""128 · C4 — ¿la diferencia la pone el CÓDIGO o el RELOJ? Intento a intento, entre dos pasadas (sus `_estados/`).

Los motores que paran por reloj (los árboles con la política por tamaño del 120-C6, `paro: reloj`; TabM con su plazo de
entrenamiento, `parado_por_plazo`) dan otro resultado si el reloj les deja otra cantidad de trabajo, aunque el código sea
el mismo. Por eso, por cada intento (mismo conjunto, mismo candidato) se compara:

- **la cantidad que pone el reloj**: los árboles entrenados (`politica_por_tamano.arboles_entrenados`) o las épocas de
  TabM (`entrenamiento_efectivo.epocas_ejecutadas`, y si la última quedó a medias);
- **el valor** de la métrica de selección.

**Con la misma cantidad, el valor tiene que ser IDÉNTICO**: si no, lo cambió algo que no es el reloj (el código, los datos,
la semilla) — eso es una VIOLACIÓN y no se explica con la carga. Con distinta cantidad, la diferencia se apunta como «del
reloj». Y se resume si la pasada nueva hace sistemáticamente MENOS trabajo (más lenta) que la de referencia: la media de
árboles/épocas de los intentos que paran por reloj, en las dos.

    python3 reloj_c4_128.py referencia_128_pcpu_v212_estados c4_128_pcpu_estados

Código de salida: 0 sin violaciones, 1 con alguna.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXACTO = 0.0


def cantidad_del_reloj(entrada: dict) -> tuple[str, object, bool] | None:
    """`(tipo, cantidad, paro_por_reloj)` del intento, o None si no es de los que el reloj puede cortar."""
    ce = entrada.get("config_efectiva") or {}
    ent = ce.get("entrenamiento_efectivo")
    if isinstance(ent, dict) and "epocas_ejecutadas" in ent:
        return ("epocas", (ent.get("epocas_ejecutadas"), bool(ent.get("ultima_epoca_parcial"))),
                bool(ent.get("parado_por_plazo")))
    pol = (ce.get("hiperparametros") or {}).get("politica_por_tamano")
    if isinstance(pol, dict) and "arboles_entrenados" in pol:
        return ("arboles", pol.get("arboles_entrenados"), pol.get("paro") == "reloj")
    return None


def _valor(entrada: dict):
    return (entrada.get("metrica_de_seleccion") or {}).get("value")


def _entradas(ruta: Path) -> dict[str, dict]:
    d = json.loads(ruta.read_text())
    return {e["candidate"]: e for e in d.get("leaderboard") or [] if e.get("candidate")}


def comparar_conjunto(ref: dict[str, dict], nuevo: dict[str, dict]) -> dict:
    out = {"pares": 0, "misma_cantidad": 0, "misma_cantidad_mismo_valor": 0, "violaciones": [], "del_reloj": [],
           "sin_cantidad_distinto_valor": [], "cantidad_media": {}}
    sumas: dict[str, list] = {}
    for cand in sorted(set(ref) & set(nuevo)):
        a, b = ref[cand], nuevo[cand]
        if a.get("estado") != "completed" or b.get("estado") != "completed":
            continue
        out["pares"] += 1
        ca, cb = cantidad_del_reloj(a), cantidad_del_reloj(b)
        va, vb = _valor(a), _valor(b)
        if ca is None or cb is None:
            if va != vb:
                out["sin_cantidad_distinto_valor"].append({"candidato": cand, "ref": va, "nuevo": vb})
            continue
        if ca[2] or cb[2]:                                   # solo los que el reloj cortó en alguna de las dos
            s = sumas.setdefault(ca[0], [0.0, 0.0, 0])
            s[0] += ca[1][0] if ca[0] == "epocas" else ca[1]
            s[1] += cb[1][0] if cb[0] == "epocas" else cb[1]
            s[2] += 1
        if ca[1] == cb[1]:
            out["misma_cantidad"] += 1
            if va == vb:
                out["misma_cantidad_mismo_valor"] += 1
            else:
                out["violaciones"].append({"candidato": cand, "cantidad": ca[1], "ref": va, "nuevo": vb})
        elif va != vb:
            out["del_reloj"].append({"candidato": cand, "cantidad_ref": ca[1], "cantidad_nuevo": cb[1], "ref": va,
                                     "nuevo": vb})
    out["cantidad_media"] = {k: {"ref": round(s[0] / s[2], 2), "nuevo": round(s[1] / s[2], 2), "intentos": s[2]}
                             for k, s in sumas.items()}
    return out


def comparar(dir_ref: Path, dir_nuevo: Path) -> dict:
    res = {}
    for f in sorted(dir_ref.glob("*.json")):
        g = dir_nuevo / f.name
        if g.exists():
            res[f.stem] = comparar_conjunto(_entradas(f), _entradas(g))
    return res


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("estados_ref")
    ap.add_argument("estados_nuevo")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = comparar(Path(a.estados_ref), Path(a.estados_nuevo))
    violaciones = 0
    for n, r in res.items():
        violaciones += len(r["violaciones"]) + len(r["sin_cantidad_distinto_valor"])
        medias = " ".join(f"{k} {v['ref']}→{v['nuevo']} (n={v['intentos']})" for k, v in r["cantidad_media"].items())
        print(f"{n[:34]:34} pares {r['pares']:3}  misma cantidad {r['misma_cantidad']:3} (iguales "
              f"{r['misma_cantidad_mismo_valor']:3})  VIOLACIONES {len(r['violaciones']) + len(r['sin_cantidad_distinto_valor'])}"
              f"  del reloj {len(r['del_reloj']):2}  {medias}")
        for v in r["violaciones"] + r["sin_cantidad_distinto_valor"]:
            print(f"    VIOLACIÓN {v}")
    print(f"TOTAL violaciones (mismo trabajo, distinto valor): {violaciones}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 1 if violaciones else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
