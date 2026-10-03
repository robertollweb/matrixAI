#!/usr/bin/env python3
"""120 · C3′ — el VEREDICTO de TabM como un motor MÁS (con torch) contra R1-GPU, y cuándo PARAR. Un solo sitio: lo
usa el guion de la medida (`referencia_120_r0.py --c3p --contra R1-GPU`) para cortar, y quien lea el resultado.

    python3 veredicto_120_c3p.py referencia_120_r1gpu.json resultado_120_c3p.json

Lo que fija (enmienda 7 del protocolo 120, registrada ANTES del código y de medir):

- **La cifra de un conjunto** es la del CAMPEÓN en el TEST (`seleccion.evaluacion_final`, rol `test`, mirada una vez
  y después de elegir), en puntos ×100 con la métrica de cierre de la tarea: AUROC en binarias, accuracy en
  multiclase, R² en regresión (el estudio la evalúa en el test: `_metricas_a_evaluar`). NO la media de selección:
  C3′ AÑADE un candidato, y el máximo de las medias sobre más candidatos nunca baja — esa cifra diría «mejora» por
  construcción. A diferencia de `veredicto_120.py` (C2), aquí el test es la cifra, y es la única vez que se fija.
- **El suelo de ruido**: R1-GPU mide los 4 de humo DOS veces. La mayor |diferencia| entre las dos es el suelo (0 si
  son idénticas); una diferencia por debajo de él no cuenta, y la paridad lo admite. Si un conjunto completa en una
  repetición y no en la otra, el instrumento no es repetible: `Incomparable`.
- **La paridad**: un conjunto cuyo campeón en C3′ NO es TabM entrenó lo mismo que en R1-GPU (los demás motores tienen
  sus segundos de hoy), así que su campeón es el mismo motor y su cifra la de R1-GPU dentro del suelo. Si no, PARO del
  instrumento: no es un resultado.
- **Presencia**: en R1-GPU la densa compite (la imagen trae torch) y TabM no; en C3′ compiten las dos. Si no, PARO:
  no se estaría midiendo lo registrado.
- **La regla 4** (global, no por tamaño: C3′ no depende del tamaño): MEJORA si más suben (≥ umbral) que bajan
  (≥ umbral), ninguno baja ≥ 2 y nadie deja de completar. El umbral es max(1 punto, suelo); la bajada que corta, max(2,
  suelo).
- **La regla 9**: la pasada se corta en el primer conjunto que baja ≥ 2 o deja de completar, o cuando los que bajan
  superan a los que suben más todos los que faltan. Lo no medido no cuenta como pérdida.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

UMBRAL = 1.0
BAJA_QUE_CORTA = 2.0
EXACTO = 1e-9
MOTOR_TABM = "matrixai.dense.tabm_cpu"
MOTOR_DENSA = "matrixai.dense.torch_cpu"
METRICA_DE_CIERRE = {"binary_classification": "auroc", "multiclass_classification": "accuracy",
                     "regression": "r2"}


class Incomparable(Exception):
    """R1-GPU y C3′ no midieron lo mismo (sin cifra de test, otra tarea, un control que no se repite): PARO."""


def completa(rec: dict | None) -> bool:
    return bool(rec) and rec.get("estado") == "completed" and bool(rec.get("campeon"))


def cifra_de_test(rec: dict) -> float | None:
    """La cifra del campeón en el test, ×100; None si el conjunto no completa. Un estudio completado SIN su
    evaluación del test, o sin la métrica de cierre de su tarea, es un instrumento roto: `Incomparable`."""
    if not completa(rec):
        return None
    nombre = rec.get("nombre")
    metrica = METRICA_DE_CIERRE.get(rec.get("tarea"))
    if metrica is None:
        raise Incomparable(f"{nombre}: tarea {rec.get('tarea')!r} sin métrica de cierre registrada")
    test = rec.get("test") or {}
    if test.get("rol") != "test" or test.get("evidencia") != "independent_test":
        raise Incomparable(f"{nombre}: completó sin una evaluación del test independiente: {str(test)[:200]}")
    valor = (test.get("metricas") or {}).get(metrica)
    if valor is None:
        raise Incomparable(f"{nombre}: el test no trae {metrica}: {sorted((test.get('metricas') or {}))}")
    return 100.0 * float(valor)


def suelo_de_ruido(primera: dict[str, dict], segunda: dict[str, dict]) -> float:
    """La mayor |diferencia| de la cifra de test entre las dos pasadas de los mismos conjuntos (los 4 de humo de
    R1-GPU), con el mismo campeón exigido; 0.0 si son idénticas. Lanza `Incomparable` si los conjuntos no son los
    mismos, si uno completa en una y no en la otra, o si cambia el motor campeón."""
    if set(primera) != set(segunda) or not primera:
        raise Incomparable(f"el control de repetición no tiene los mismos conjuntos: {sorted(primera)} / {sorted(segunda)}")
    peor = 0.0
    for n in sorted(primera):
        a, b = primera[n], segunda[n]
        if completa(a) != completa(b):
            raise Incomparable(f"{n}: completa en una repetición y no en la otra ({a.get('estado')} / {b.get('estado')})")
        if not completa(a):
            continue
        if a["campeon"] != b["campeon"]:
            raise Incomparable(f"{n}: el campeón cambia al repetir ({a['campeon']} / {b['campeon']})")
        d = abs(cifra_de_test(a) - cifra_de_test(b))
        peor = max(peor, d if d >= EXACTO else 0.0)
    return peor


def motores_con_intentos(rec: dict) -> set[str]:
    return {x.get("motor") for x in rec.get("intentos") or [] if x.get("motor")}


def presencia(rec: dict, *, c3p: bool, con_densa: bool = True) -> str | None:
    """El motivo para PARAR o None. Solo se exige en un estudio que completó (uno que no, ya cuenta como pérdida).
    `con_densa`: C3′ (paquete con GPU) la quiere compitiendo; C3″ (enmienda 8, paquete CPU) la quiere FUERA."""
    if not completa(rec):
        return None
    motores = motores_con_intentos(rec)
    n = rec.get("nombre")
    if con_densa and MOTOR_DENSA not in motores:
        return f"{n}: la densa no tiene intentos — la imagen no trae torch: no es la condición registrada"
    if not con_densa and MOTOR_DENSA in motores:
        return f"{n}: la densa anterior tiene intentos — en el paquete CPU (enmienda 8) no compite"
    if c3p and MOTOR_TABM not in motores:
        return f"{n}: C3′ sin intentos de TabM — no se está midiendo C3′"
    if not c3p and MOTOR_TABM in motores:
        return f"{n}: R1-GPU con intentos de TabM — no es la referencia registrada"
    return None


def comparar(r1: dict | None, c3: dict, suelo: float) -> dict:
    """Un conjunto: R1-GPU frente a C3′. `clase` ∈ sube | baja | baja_2 | igual | deja_de_completar |
    empieza_a_completar | ninguno_completa."""
    nombre = c3.get("nombre")
    umbral, corta_en = max(UMBRAL, suelo), max(BAJA_QUE_CORTA, suelo)
    c1, cc = completa(r1), completa(c3)
    base = {"nombre": nombre, "campeon_r1": (r1 or {}).get("campeon"), "campeon_c3p": c3.get("campeon"),
            "pared_r1_s": (r1 or {}).get("pared_estudio_s"), "pared_c3p_s": c3.get("pared_estudio_s")}
    if c1 and cc:
        if r1.get("tarea") != c3.get("tarea"):
            raise Incomparable(f"{nombre}: tarea {r1.get('tarea')!r} en R1-GPU y {c3.get('tarea')!r} en C3′")
        p1, p2 = cifra_de_test(r1), cifra_de_test(c3)
        d = p2 - p1
        clase = ("baja_2" if d <= -corta_en else "baja" if d <= -umbral else "sube" if d >= umbral else "igual")
        return {**base, "clase": clase, "r1": p1, "c3p": p2, "diferencia": d}
    clase = "deja_de_completar" if c1 else "empieza_a_completar" if cc else "ninguno_completa"
    return {**base, "clase": clase, "r1": cifra_de_test(r1) if c1 else None,
            "c3p": cifra_de_test(c3) if cc else None, "diferencia": None}


def paridad(comparacion: dict, suelo: float) -> str | None:
    """El motivo para PARAR o None. Sin TabM de campeón, mismo campeón que R1-GPU y la misma cifra (dentro del
    suelo; exacta si el suelo es 0)."""
    if comparacion["clase"] in ("ninguno_completa", "deja_de_completar", "empieza_a_completar"):
        return None
    if comparacion["campeon_c3p"] == MOTOR_TABM:
        return None
    n = comparacion["nombre"]
    if comparacion["campeon_c3p"] != comparacion["campeon_r1"]:
        return (f"{n}: sin TabM de campeón, el campeón cambia ({comparacion['campeon_r1']} → "
                f"{comparacion['campeon_c3p']}): los demás motores no entrenaron lo mismo")
    if abs(comparacion["diferencia"]) > max(suelo, EXACTO):
        return (f"{n}: sin TabM de campeón, la cifra no es la de R1-GPU ({comparacion['diferencia']:+.6f} puntos; "
                f"suelo {suelo:g})")
    return None


def corta(comparaciones: list[dict], faltan: int) -> str | None:
    """La regla 9 tras el último conjunto medido."""
    ult = comparaciones[-1]
    if ult["clase"] == "baja_2":
        return f"{ult['nombre']} baja {ult['diferencia']:.2f} puntos"
    if ult["clase"] == "deja_de_completar":
        return f"{ult['nombre']} deja de completar"
    cl = Counter(c["clase"] for c in comparaciones)
    suben = cl["sube"] + cl["empieza_a_completar"]
    bajan = cl["baja"] + cl["baja_2"]
    if bajan > suben + faltan:
        return f"bajan {bajan} y, aunque subieran los {faltan} que faltan, solo subirían {suben + faltan}"
    return None


def veredicto(comparaciones: list[dict]) -> dict:
    cl = Counter(c["clase"] for c in comparaciones)
    suben = cl["sube"] + cl["empieza_a_completar"]
    bajan = cl["baja"] + cl["baja_2"]
    mejora = bool(comparaciones) and suben > bajan and not cl["baja_2"] and not cl["deja_de_completar"]
    return {"conjuntos": [c["nombre"] for c in comparaciones], "suben": suben, "bajan": bajan,
            "bajan_2": cl["baja_2"], "dejan_de_completar": cl["deja_de_completar"], "iguales": cl["igual"],
            "ninguno_completa": cl["ninguno_completa"],
            "tabm_campeon_en": [c["nombre"] for c in comparaciones if c["campeon_c3p"] == MOTOR_TABM],
            "mejora": mejora}


def _main(argv: list[str]) -> int:
    r1 = json.loads(Path(argv[1]).read_text())
    c3 = json.loads(Path(argv[2]).read_text())
    suelo = r1.get("suelo_de_ruido")
    if suelo is None:
        print(f"{argv[1]} no trae suelo_de_ruido: no es una referencia R1-GPU"); return 2
    comps = [comparar(r1["conjuntos"].get(n), rec, suelo) for n, rec in c3["conjuntos"].items()]
    for c in comps:
        dif = "" if c["diferencia"] is None else f"{c['diferencia']:+.2f}"
        print(f"{c['nombre']:34} {c['clase']:20} {dif:>7}  {c['campeon_r1']} → {c['campeon_c3p']}")
    print(json.dumps(veredicto(comps), ensure_ascii=False, indent=1))
    if c3.get("corte"):
        print("CORTADA:", c3["corte"])
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
