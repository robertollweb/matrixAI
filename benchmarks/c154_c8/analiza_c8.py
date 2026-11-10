"""Aplica REGLA_DE_DECISION.md a los c8_*.json de una carpeta. Uso: analiza_c8.py DIR"""
import json, statistics, sys
from pathlib import Path

T95 = {4: 2.776, 1: 12.706, 2: 4.303, 3: 3.182}
SOLO_TEXTO, LIN, SEL = "sklearn.texto_tfidf", "sklearn.texto_y_tabla_lineal", "lightgbm.texto_y_tabla"


def ic(d):
    n = len(d); m = statistics.fmean(d)
    h = T95.get(n - 1, 2.0) * statistics.stdev(d) / n ** 0.5 if n > 1 else float("nan")
    return m, m - h, m + h


def main(dir_):
    out = {}
    for f in sorted(Path(dir_).glob("c8_*.json")):
        r = json.load(open(f)); c = r["estudios"]["completo"]; reg = r["tarea"] == "regression"
        met = c["metrica"]; med = {m: v for m, v in c["seleccion_media"].items() if v is not None}
        pm = lambda m: c["intentos_por_motor"][m]["valores"]
        mejor = (min if reg else max)
        H = mejor([LIN, SEL], key=lambda m: med[m]) if LIN in med and SEL in med else None
        cand_B = [m for m in (SOLO_TEXTO, r.get("x_estrella")) if m in med]
        B = mejor(cand_B, key=lambda m: med[m]) if cand_B else None
        d = {"campeon_completo": c["campeon"], "metrica": met, "H": H, "B": B, "x_estrella": r.get("x_estrella"),
             "seleccion": med, "fuera": c["fuera"], "columnas": c["columnas_politica"]}
        if H and B:
            dif = [((b - h) / b if reg else h - b) for h, b in zip(pm(H), pm(B))]
            m, lo, hi = ic(dif)
            test = lambda e: (r["estudios"].get(f"solo-{e}") or {}).get("prueba_reservada_del_campeon", {}).get(met)
            tH, tB = test(H), test(B)
            mt = None if tH is None or tB is None else ((tB - tH) / tB if reg else tH - tB)
            umbral = 0.02 if reg else 0.010
            v = "GANA" if (m >= umbral and lo > 0 and (mt or 0) > 0) else "PIERDE" if (m <= -umbral and hi < 0) else "EMPATE"
            d.update(mejora_seleccion=m, ic95=[lo, hi], mejora_prueba=mt, prueba={H: tH, B: tB}, veredicto=v)
        d["prueba_por_candidato"] = {k[5:]: v.get("prueba_reservada_del_campeon") for k, v in r["estudios"].items() if k.startswith("solo-")}
        ref = [statistics.median([x for x in c["intentos_por_motor"][m]["wall_s"] if x]) for m in ("lightgbm", "sklearn.hgb") if m in c["intentos_por_motor"] and c["intentos_por_motor"][m]["wall_s"]]
        d["mediana_wall_s"] = {m: statistics.median([x for x in v["wall_s"] if x]) for m, v in c["intentos_por_motor"].items() if any(v["wall_s"])}
        if ref:
            d["razon_de_tiempo_a_la_referencia"] = {m: d["mediana_wall_s"][m] / statistics.fmean(ref) for m in (LIN, SEL) if m in d["mediana_wall_s"]}
        d["segundos_completo"] = c["segundos_del_estudio"]
        out[r["conjunto"]] = d
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    v = {k: x.get("veredicto") for k, x in out.items()}
    print("VEREDICTOS", v, "-> gana en", sum(1 for x in v.values() if x == "GANA"), "de", len(v), "; pierde en", [k for k, x in v.items() if x == "PIERDE"])


if __name__ == "__main__":
    main(sys.argv[1])
