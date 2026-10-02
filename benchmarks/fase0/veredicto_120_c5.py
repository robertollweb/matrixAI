#!/usr/bin/env python3
"""120-C5 · el VEREDICTO de las dos correcciones de TabM (enmienda 3 del protocolo 119 v4): TabM 1.2.0 frente a la
1.1.0 CONGELADA (`resultado_pasada_119_c3.json`) en los 12 conjuntos no sellados con columnas categóricas.

    python3 veredicto_120_c5.py resultado_pasada_119_c3.json <resultado de la pasada --solo con la 1.2.0>

Lo que fija la enmienda 3 y aplica este fichero (escrito ANTES de tener los datos):

- **La cifra de un conjunto**: la media, sobre los intentos (repetición, pliegue) COMPLETOS EN LOS DOS, de la métrica
  de cierre de la Fase 0 (`metrica_de_cierre_por_dataset`: AUROC, accuracy o R²), ×100. Emparejado: mismas
  particiones y semillas, así que el mismo (repetición, pliegue) es el mismo problema.
- **Deja de completar**: un intento completo con la 1.1.0 que con la 1.2.0 no lo está (o le falta la métrica).
- **Se adopta la 1.2.0** si, en los conjuntos que DECIDEN, ninguno baja ≥ 2 puntos, ninguno deja de completar (enmienda
  4: dejar de completar es peor que bajar 2) y no bajan (≥ 1) más de los que suben (≥ 1); |diferencia| < 1 no cuenta.
  Los umbrales son los de `veredicto_120` (un solo sitio).
- **No deciden** (enmienda 4) los conjuntos donde la 1.1.0 congelada paró por PLAZO en TODOS sus intentos: ahí la
  diferencia mezcla la corrección con la carga de la máquina (la congelada corría 2 procesos; esta, uno). Se miden y se
  informan. Sale del propio dato congelado (`entrenamiento_efectivo.parado_por_plazo`), no de una lista.
- La pasada nueva tiene que ser de la 1.2.0 y medir EXACTAMENTE los 12; si no, no es esta medida (PARO).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("veredicto_120", AQUI / "veredicto_120.py")
_v120 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v120)
UMBRAL, BAJA_QUE_CORTA = _v120.UMBRAL, _v120.BAJA_QUE_CORTA

MOTOR = "matrixai.dense.tabm_cpu"
#: Los 12 de la enmienda 3 (no sellados con categóricas, leídos en sus ARFF).
CONJUNTOS = ("dresses-sales", "Moneyball", "house_prices_nominal", "Internet-Advertisements", "sick",
             "PhishingWebsites", "Amazon_employee_access", "KDDCup09_appetency", "okcupid-stem", "connect-4", "kick",
             "Allstate_Claims_Severity")


class NoEsEstaMedida(Exception):
    """La pasada nueva no es la que registró la enmienda 3."""


def _por_intento(resultado: dict) -> dict[str, dict[tuple[int, int], float | None]]:
    """conjunto → {(repetición, pliegue): métrica de cierre, o None si el intento no completó con ella}."""
    cierre = resultado["metrica_de_cierre_por_dataset"]
    out: dict[str, dict[tuple[int, int], float | None]] = {}
    for x in resultado["resultados"]:
        if x.get("motor") != MOTOR or x.get("dataset") not in CONJUNTOS:
            continue
        valor = (x.get("metricas") or {}).get(cierre[x["dataset"]]) if x.get("estado") == "completed" else None
        out.setdefault(x["dataset"], {})[(int(x["repeticion"]), int(x["pliegue"]))] = (
            float(valor) if isinstance(valor, (int, float)) else None)
    return out


def versiones(resultado: dict) -> set[str]:
    """Las `engine_version` de los intentos de TabM de los 12 (cada intento guarda la suya), sin el sufijo de torch."""
    return {str(x.get("engine_version") or "").split("+")[0] for x in resultado["resultados"]
            if x.get("motor") == MOTOR and x.get("dataset") in CONJUNTOS}


def por_plazo_en_la_congelada(viejo: dict) -> dict[str, tuple[int, int]]:
    """conjunto → (intentos que pararon por plazo, intentos) en la 1.1.0 congelada."""
    out: dict[str, list[int]] = {}
    for x in viejo["resultados"]:
        if x.get("motor") == MOTOR and x.get("dataset") in CONJUNTOS:
            c = out.setdefault(x["dataset"], [0, 0])
            c[0] += bool((x.get("entrenamiento_efectivo") or {}).get("parado_por_plazo"))
            c[1] += 1
    return {n: (a, b) for n, (a, b) in out.items()}


def comparar(viejo: dict, nuevo: dict) -> list[dict]:
    """Una fila por conjunto de los 12. La congelada tiene que ser TODA de la 1.1.0 y la nueva TODA de la 1.2.0."""
    if versiones(viejo) != {"1.1.0"} or versiones(nuevo) != {"1.2.0"}:
        raise NoEsEstaMedida(f"versiones del motor: congelada {sorted(versiones(viejo))}, "
                             f"nueva {sorted(versiones(nuevo))} (tienen que ser 1.1.0 y 1.2.0)")
    a, b = _por_intento(viejo), _por_intento(nuevo)
    plazo = por_plazo_en_la_congelada(viejo)
    if set(b) != set(CONJUNTOS):
        raise NoEsEstaMedida(f"la pasada nueva mide {sorted(b)}, no los 12 de la enmienda 3")
    filas = []
    for n in CONJUNTOS:
        va, vb = a.get(n, {}), b[n]
        p_n, n_n = plazo.get(n, (0, 0))
        decide = not (n_n and p_n == n_n)
        dejan = sorted(k for k, v in va.items() if v is not None and vb.get(k) is None)
        comunes = sorted(k for k in va if va[k] is not None and vb.get(k) is not None)
        if dejan:
            filas.append({"nombre": n, "clase": "deja_de_completar", "intentos_que_dejan": dejan,
                          "diferencia": None, "n_comunes": len(comunes), "decide": decide,
                          "plazo_en_la_congelada": [p_n, n_n]})
            continue
        if not comunes:
            filas.append({"nombre": n, "clase": "ninguno_completa", "diferencia": None, "n_comunes": 0,
                          "decide": decide, "plazo_en_la_congelada": [p_n, n_n]})
            continue
        ma = 100.0 * sum(va[k] for k in comunes) / len(comunes)
        mb = 100.0 * sum(vb[k] for k in comunes) / len(comunes)
        d = mb - ma
        clase = ("baja_2" if d <= -BAJA_QUE_CORTA else "baja" if d <= -UMBRAL else "sube" if d >= UMBRAL else "igual")
        filas.append({"nombre": n, "clase": clase, "v1_1_0": ma, "v1_2_0": mb, "diferencia": d,
                      "n_comunes": len(comunes), "decide": decide, "plazo_en_la_congelada": [p_n, n_n]})
    return filas


def veredicto(filas: list[dict]) -> dict:
    """La regla, sobre los conjuntos que deciden; los demás, informados aparte."""
    cl = Counter(f["clase"] for f in filas if f.get("decide", True))
    suben, bajan = cl["sube"], cl["baja"] + cl["baja_2"]
    adopta = not cl["baja_2"] and not cl["deja_de_completar"] and bajan <= suben
    return {"suben": suben, "bajan": bajan, "bajan_2": cl["baja_2"], "dejan_de_completar": cl["deja_de_completar"],
            "iguales": cl["igual"], "ninguno_completa": cl["ninguno_completa"], "adopta_la_1_2_0": adopta,
            "deciden": [f["nombre"] for f in filas if f.get("decide", True)],
            "no_deciden_por_plazo": {f["nombre"]: f["clase"] for f in filas if not f.get("decide", True)}}


def _main(argv: list[str]) -> int:
    viejo = json.loads(Path(argv[1]).read_text())
    nuevo = json.loads(Path(argv[2]).read_text())
    filas = comparar(viejo, nuevo)
    for f in filas:
        d = "" if f["diferencia"] is None else f"{f['diferencia']:+.2f}"
        print(f"{f['nombre']:28} {f['clase']:18} {d:>8}  ({f['n_comunes']} intentos comunes)"
              f"{'' if f['decide'] else '  [no decide: plazo en la congelada]'}")
    print(json.dumps(veredicto(filas), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
