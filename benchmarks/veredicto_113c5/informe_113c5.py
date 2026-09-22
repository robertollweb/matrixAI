# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""113-C5 — el veredicto de la pasada v2, con los números delante de Roberto.

Lee los DOS artefactos (101-C5, la densa vieja con el protocolo v1; y 113-C4, la
densa nueva con el v2) y compone lo que pide el contrato 113, C5:
- la regla de cierre para TODOS los motores, v1 frente a v2 (es relativa al mejor:
  mover uno mueve a los demás);
- la densa nueva frente a la vieja, en total, por tarea y por tamaño;
- dónde gana y dónde pierde, conjunto a conjunto (solo los que están en los dos);
- el tiempo por intento de la densa (mediana de `wall_s` de los completados);
- qué les pasa a los tres aprobados, y el margen de catboost.

No recalcula la regla: la LEE de `alcance_y_veredicto`, que es lo que cada pasada
calculó con su propio protocolo. Un segundo cálculo aquí sería un segundo sitio
declarando lo mismo.

    cd /home/deployer/matrixAI && python3 benchmarks/veredicto_113c5/informe_113c5.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

FASE0 = Path(__file__).resolve().parents[1] / "fase0"
V1 = FASE0 / "pasada_amplia_101_c5_resultado.json"
V2 = FASE0 / "pasada_v2_113_resultado.json"
DENSA = "matrixai.dense.torch_cpu"
APROBADOS = ("lightgbm", "sklearn.hgb", "catboost")


def por_motor(artefacto: dict) -> dict:
    return artefacto["alcance_y_veredicto"]["por_motor"]


def resumen_de_la_regla(artefacto: dict) -> dict[str, dict]:
    return {m: {"cumplidos": d["cumplidos"], "datasets": d["datasets"], "fraccion": d["fraccion"],
                "cumple_la_regla": d["cumple_la_regla"],
                "puede_perder": d.get("datasets_que_puede_perder_sin_incumplir"),
                "le_faltan": d.get("datasets_que_le_faltan_para_cumplir")}
            for m, d in por_motor(artefacto).items()}


def atributos_por_dataset(artefacto: dict) -> dict[str, dict]:
    atributos = {}
    for r in artefacto["resultados"]:
        atributos.setdefault(r["dataset"], {"tarea": r["tarea"], "cubo": r["cubo"],
                                            "sellado": r["sellado"]})
    return atributos


def cumple_por_dataset(artefacto: dict, motor: str) -> dict[str, bool]:
    return {d["dataset"]: bool(d["cumple"]) for d in por_motor(artefacto)[motor]["detalle"]}


def desglose(artefacto: dict, motor: str, clave: str) -> dict[str, str]:
    atributos = atributos_por_dataset(artefacto)
    cuenta: dict[str, list[bool]] = {}
    for dataset, cumple in cumple_por_dataset(artefacto, motor).items():
        grupo = atributos.get(dataset, {}).get(clave, "¿?")
        cuenta.setdefault(str(grupo), []).append(cumple)
    return {g: f"{sum(v)}/{len(v)}" for g, v in sorted(cuenta.items())}


def cambios(v1: dict, v2: dict, motor: str) -> dict[str, list[str]]:
    a, b = cumple_por_dataset(v1, motor), cumple_por_dataset(v2, motor)
    comunes = sorted(set(a) & set(b))
    return {"gana": [d for d in comunes if b[d] and not a[d]],
            "pierde": [d for d in comunes if a[d] and not b[d]],
            "solo_en_v1": sorted(set(a) - set(b)), "solo_en_v2": sorted(set(b) - set(a))}


def mediana_de_segundos(artefacto: dict, motor: str) -> float | None:
    tiempos = [r["wall_s"] for r in artefacto["resultados"]
               if r["motor"] == motor and r["estado"] == "completed" and r.get("wall_s") is not None]
    return round(statistics.median(tiempos), 2) if tiempos else None


def informe(v1: dict, v2: dict) -> dict:
    return {
        "v2_terminada": v2.get("parcial") is False,
        "regla": {"v1": resumen_de_la_regla(v1), "v2": resumen_de_la_regla(v2)},
        "densa": {
            "por_tarea": {"v1": desglose(v1, DENSA, "tarea"), "v2": desglose(v2, DENSA, "tarea")},
            "por_tamano": {"v1": desglose(v1, DENSA, "cubo"), "v2": desglose(v2, DENSA, "cubo")},
            "cambios": cambios(v1, v2, DENSA),
            "segundos_por_intento_mediana": {"v1": mediana_de_segundos(v1, DENSA),
                                             "v2": mediana_de_segundos(v2, DENSA)}},
        "aprobados": {m: {"v1": resumen_de_la_regla(v1)[m], "v2": resumen_de_la_regla(v2)[m],
                          "cambios": cambios(v1, v2, m)} for m in APROBADOS},
    }


def main() -> dict:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    resultado = informe(v1, v2)
    if not resultado["v2_terminada"]:
        print("AVISO: la pasada v2 NO ha terminado; esto es un ensayo sobre un artefacto parcial",
              file=sys.stderr)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return resultado


if __name__ == "__main__":
    main()
