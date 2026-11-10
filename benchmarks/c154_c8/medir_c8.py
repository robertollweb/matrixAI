"""154-C8 · criterio 8 POR EL ESTUDIO REAL (no con el arnés de C1/C2): estudios completos sobre C (PLACSP), D (BOE) y ropa
con el texto declarado + columnas. Protocolo y regla de decisión: REGLA_DE_DECISION.md (escrita ANTES de medir).

Por conjunto:
  1. UN estudio COMPLETO (todos los candidatos que compiten de verdad, protocolo por omisión del estudio: 5 pliegues x 1,
     20 % de prueba reservada, semilla 0). Registra quién gana, la métrica de selección de cada candidato (media y por pliegue),
     lo que el estudio hizo con cada columna (tipos tras `tipar_columnas_numericas`), el reparto de la búsqueda por motor y el
     tiempo de cada intento (frío: un hijo por intento).
  2. UN estudio por candidato de interés, SOLO él + el baseline de reserva (que el estudio exige): H_lin, H_sel, T (solo texto)
     y X* (el mejor de solo tabla del estudio completo). Así cada uno pasa por el camino entero del estudio (reajuste con todo
     el desarrollo, recalibración, umbral) y se mide en LA PRUEBA RESERVADA, que el estudio completo solo mide para el campeón.
     Mismo reparto de filas (misma semilla, mismos datos), distinto reloj (más reserva por motor): se declara.
Uso: medir_c8.py [--conjuntos D,ropa,C] [--filas N] [--folds K] [--salida DIR] [--tope-estudio S] [--datos DIR]
El código lo fija PYTHONPATH (lo pone ~/encolar.sh); se imprime de dónde sale cada paquete."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

DATOS = Path("/home/deployer/.agentes/ola3-10/medida/datos")
SEMILLA = 107031
HIBRIDO_LIN, HIBRIDO_SEL = "sklearn.texto_y_tabla_lineal", "lightgbm.texto_y_tabla"
SOLO_TEXTO = "sklearn.texto_tfidf"
RESERVA = "baseline"
SOLO_TABLA = ("lightgbm", "sklearn.hgb", "matrixai.dense.torch_cpu", "matrixai.dense.tabm_cpu")
CONJ = {
    "D": dict(csv="boe/tarea_d.csv", tarea="binary_classification", pos="agotada", idioma="es", objeto="un documento",
              cols=("departamento", "rango", "ambito", "anio_publicacion")),
    "ropa": dict(csv="ropa/ropa.csv", tarea="binary_classification", pos="0", idioma="en", objeto="una reseña",
                 cols=("Clothing_ID", "Age", "Positive_Feedback_Count", "Division_Name", "Department_Name", "Class_Name")),
    "C": dict(csv="placsp/tarea_c.csv", tarea="regression", pos=None, idioma="es", objeto="una licitación",
              cols=("tipo_code", "procedimiento_code", "cpv_division", "organo")),
}


def leer(nombre: str, datos: Path, filas: int | None):
    cfg = CONJ[nombre]
    with (datos / cfg["csv"]).open(encoding="utf-8", newline="") as f:
        todas = list(csv.DictReader(f))
    if filas and len(todas) > filas:
        todas = random.Random(SEMILLA).sample(todas, filas)
    campos = ["row_id", "texto", *cfg["cols"], "target"]
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=campos, extrasaction="ignore")
    w.writeheader()
    w.writerows(todas)
    payload = {"csv_text": b.getvalue(), "objetivo": "target", "tarea": cfg["tarea"],
               "entradas": ["texto", *cfg["cols"]], "unidad_de_observacion": cfg["objeto"],
               "observation_id_field": "row_id", "columnas_de_texto": {"texto": cfg["idioma"]}}
    if cfg["pos"] is not None:
        payload["clase_positiva"] = cfg["pos"]
    return cfg, payload, len(todas)


def _metricas(lista) -> dict:
    return {m["metric_id"]: m["value"] for m in (lista or [])}


def correr_estudio(ej, payload: dict, etiqueta: str, tope: float, solo: tuple[str, ...] | None) -> dict:
    """Un estudio en proceso. `solo`: los motores que SE QUEDAN (más la reserva); `None` = todos los de siempre."""
    original = ej._motores_del_estudio
    if solo is not None:
        quedan = set(solo) | {RESERVA}

        def acotado(*a, **k):
            permitidos, motores = original(*a, **k)
            falta = quedan - set(motores)
            assert not falta, f"el estudio no ofrece {sorted(falta)}: ofrece {sorted(motores)}"
            return (tuple(m for m in permitidos if m in quedan), {m: v for m, v in motores.items() if m in quedan})
        ej._motores_del_estudio = acotado
    t0 = time.time()
    try:
        r = ej.iniciar_estudio(dict(payload), owner="c154-c8")
        if not r.get("ok") or (r.get("estudio") or {}).get("estado") == "failed":
            return {"etiqueta": etiqueta, "estado": "no_arranco", "rechazo": json.dumps(r, default=str, ensure_ascii=False)[:600]}
        job = r["estudio"]["job_id"]
        while True:
            e = ej.estudio_status(job, "c154-c8")
            if e["estado"] in ("completed", "failed", "cancelled"):
                break
            if time.time() - t0 > tope:
                ej.cancelar_estudio(job, "c154-c8")
                time.sleep(5)
                e = ej.estudio_status(job, "c154-c8")
                e["estado"] = "cancelado_por_tope"
                break
            time.sleep(2)
    finally:
        ej._motores_del_estudio = original
    segundos = round(time.time() - t0, 1)
    s = e.get("seleccion") or {}
    out = {"etiqueta": etiqueta, "job": job, "estado": e["estado"], "segundos_del_estudio": segundos,
           "error": e.get("error"), "campeon": s.get("candidate_engine"),
           "metrica_de_seleccion": (s.get("recomputo_de_seleccion") or {}).get("metric_id_calidad"),
           "prueba_reservada_del_campeon": _metricas((s.get("evaluacion_final") or {}).get("metrics")),
           "filas_del_reparto": s.get("filas_del_reparto"),
           "reparto_de_la_busqueda_segundos": s.get("reparto_de_la_busqueda_segundos"),
           "fuera": [{"motor": m.get("motor"), "clave": m.get("clave"), "motivo": ((m.get("motivo_del_motor") or {}).get("es") or "")[:200]}
                     for m in (s.get("motores_que_no_puntuaron") or [])],
           "fuera_por_plazo": e.get("motores_fuera_por_plazo"), "fuera_por_memoria": e.get("motores_fuera_por_memoria"),
           "columnas_politica": [{k: c.get(k) for k in c if k in ("columna", "tipo", "rol", "imputacion", "codificacion", "accion")}
                                 for c in (s.get("politica_preparacion") or {}).get("columnas", [])],
           "lectura_del_modelo": {k: (s.get("lectura_del_modelo") or {}).get(k) for k in ("modo", "texto", "tabla")},
           "entrenamiento_de_los_reajustes": s.get("entrenamiento_de_los_reajustes"),
           "latencia_ms_por_fila": (s.get("latencia") or {}).get("latency_ms_per_row")}
    # la media de la selección por motor (la cifra con la que el estudio decide) y, por intento, su pliegue y su reloj
    out["seleccion_media"] = {m["motor"]: m["valor"] for m in (s.get("media_de_la_seleccion") or {}).get("motores", [])}
    metrica = out["metrica_de_seleccion"]
    por_motor: dict[str, dict] = {}
    for x in e.get("leaderboard") or []:
        motor = x.get("engine") or x.get("motor") or x.get("candidate")
        d = por_motor.setdefault(motor, {"valores": [], "wall_s": [], "cpu_s": [], "ram_mb": [], "estados": []})
        m = x.get("metrica_de_seleccion") or {}
        d["valores"].append(m.get("value"))
        d["wall_s"].append(x.get("wall_seconds"))
        d["cpu_s"].append(x.get("cpu_seconds"))
        d["ram_mb"].append(x.get("ram_mb"))
        d["estados"].append(x.get("estado"))
    out["intentos_por_motor"] = por_motor
    out["_claves_de_una_entrada"] = sorted((e.get("leaderboard") or [{}])[0].keys())
    out["metrica"] = metrica
    return out


def ic95(dif: list[float]) -> tuple[float, float, float]:
    """media y límites del IC95 (t de Student, n-1 g.l.) de las diferencias por pliegue."""
    n = len(dif)
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    m = statistics.fmean(dif)
    h = t * statistics.stdev(dif) / n ** 0.5 if n > 1 else float("nan")
    return m, m - h, m + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjuntos", default="D,ropa,C")
    ap.add_argument("--filas", type=int, default=0, help="prueba en pequeño: muestra de N filas (0 = todas)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tope-estudio", type=float, default=3600.0)
    ap.add_argument("--datos", default=str(DATOS))
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    os.environ["MATRIXAI_ESTUDIOS_DIR"] = str(salida / "estudios")
    import matrixai
    import matrixai_engines
    import matrixai_studio.estudio_job as ej
    print("CÓDIGO", matrixai.__file__, matrixai_engines.__file__, ej.__file__, flush=True)
    (salida / "procedencia.txt").write_text(f"{matrixai.__file__}\n{matrixai_engines.__file__}\n{ej.__file__}\nPYTHONPATH={os.environ.get('PYTHONPATH')}\n")
    for nombre in a.conjuntos.split(","):
        destino = salida / f"c8_{nombre}.json"
        if destino.exists():
            print(nombre, "ya medido, se salta", flush=True)
            continue
        cfg, payload, n = leer(nombre, Path(a.datos), a.filas or None)
        payload["folds"], payload["repeats"] = a.folds, 1
        res = {"conjunto": nombre, "filas_dadas_al_estudio": n, "folds": a.folds, "tarea": cfg["tarea"], "estudios": {}}
        print(f"=== {nombre}: {n} filas, estudio completo", flush=True)
        completo = correr_estudio(ej, payload, f"{nombre}-completo", a.tope_estudio, None)
        res["estudios"]["completo"] = completo
        print(nombre, "completo", completo["estado"], completo.get("campeon"), completo.get("seleccion_media"), completo["segundos_del_estudio"], flush=True)
        destino.with_suffix(".parcial.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        if completo["estado"] == "completed":
            medias = {m: v for m, v in completo["seleccion_media"].items() if v is not None}   # None = no puntuó (ver «fuera»)
            menor_es_mejor = cfg["tarea"] == "regression"
            tablas = [m for m in SOLO_TABLA if m in medias]
            mejor_tabla = (min if menor_es_mejor else max)(tablas, key=lambda m: medias[m]) if tablas else None
            res["x_estrella"] = mejor_tabla
            solos = [HIBRIDO_LIN, HIBRIDO_SEL, SOLO_TEXTO] + ([mejor_tabla] if mejor_tabla else [])
            for motor in solos:
                if motor not in medias:
                    res["estudios"][f"solo-{motor}"] = {"estado": "no_compitio_en_el_completo"}
                    continue
                est = correr_estudio(ej, payload, f"{nombre}-solo-{motor}", a.tope_estudio, (motor,))
                res["estudios"][f"solo-{motor}"] = est
                print(nombre, "solo", motor, est["estado"], est.get("campeon"), est.get("prueba_reservada_del_campeon"), est["segundos_del_estudio"], flush=True)
                shutil.rmtree(salida / "estudios" / str(est.get("job")), ignore_errors=True)
        shutil.rmtree(salida / "estudios" / str(completo.get("job")), ignore_errors=True)
        destino.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        destino.with_suffix(".parcial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
