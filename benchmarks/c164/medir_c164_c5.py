"""164-C5 · MEDIDA FINAL del pronóstico temporal POR EL ESTUDIO REAL del Studio (`estudio_job.iniciar_estudio` con `payload["forecast"]`,
el camino de la persona): lluvia, Avocado y un ejemplo NUEVO (NYC Uber pickups, CC0). Plan y rejilla: ~/.agentes/164/c5/PLAN.md.

Qué responde, por estudio: ¿el campeón gana a la persistencia y a la estacional en la prueba reservada, por cuánto (mejora de MAE y MASE),
en cuántos pliegues y con qué cobertura del intervalo (la MEDIDA en la prueba, junto a la nominal)? Todo se LEE del sobre del estudio; este
guion no recalcula ni reimplementa nada de él. Resultado DESCRIPTIVO: el contrato 164 no fija umbral de «cumple».

Uso: medir_c164_c5.py --salida DIR [--conjuntos lluvia,avocado,uber_h1,uber_h6] [--dias N] [--ciudades N] [--folds K] [--modo completo|rapido]
                       [--tope-estudio S] [--max-series N]
`--modo rapido` y `--max-series` son SOLO para la sonda. El código lo fija PYTHONPATH (lo pone ~/encolar.sh); se imprime de dónde sale.
Reanuda: un conjunto con su JSON ya escrito se salta."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
LLUVIA = Path("/home/deployer/lluvia_MUNDIAL.csv")
DATOS_114 = Path("/home/deployer/fechas_114c5_datos")
DUENO = "c164-c5"

#: La configuración de cada estudio (la rejilla mínima del PLAN). `unidad`: lo que la persona escribiría como «qué es una fila».
CONJUNTOS = {
    "lluvia": dict(objetivo="rain", unidad="una ciudad y un día", tiempo="dia", serie="ciudad", horizon=1, lags=7, season=7,
                   lag_columns=["temp_mean", "pressure", "humidity", "cloud_cover"], extra_inputs=[], datos="lluvia"),
    "avocado": dict(objetivo="AveragePrice", unidad="una región, un tipo y una semana", tiempo="Date", serie="serie", horizon=1,
                    lags=4, season=52, lag_columns=["Total_Volume"], extra_inputs=[], datos="avocado"),
    "uber_h1": dict(objetivo="pickups", unidad="un distrito y una hora", tiempo="pickup_dt", serie="borough", horizon=1, lags=24,
                    season=24, lag_columns=["temp"], extra_inputs=["hday"], datos="uber"),
    "uber_h6": dict(objetivo="pickups", unidad="un distrito y una hora", tiempo="pickup_dt", serie="borough", horizon=6, lags=24,
                    season=24, lag_columns=["temp"], extra_inputs=["hday"], datos="uber"),
}


# ---------------------------------------------------------------------------------------------------------------- datos (puro)
def a_csv(filas: list[dict], campos: list[str]) -> str:
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=campos, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(filas)
    return b.getvalue()


def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for trozo in iter(lambda: fh.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def cargar_lluvia(ruta: Path = LLUVIA, dias: int = 730, ciudades: int | None = None) -> tuple[list[dict], dict]:
    """Como el 164-C1: las ciudades se identifican por (lat, lon), se queda con los ÚLTIMOS `dias` días de cada una y solo con las que
    tienen `dias` o más; el calendario se reconstruye (el CSV no trae fecha): día d = 2020-01-01 + d, igual en todas."""
    ult: dict = defaultdict(lambda: deque(maxlen=dias))
    with ruta.open(newline="") as fh:
        r = csv.reader(fh)
        cab = next(r)
        ilat, ilon = cab.index("lat"), cab.index("lon")
        for row in r:
            ult[(row[ilat], row[ilon])].append(row)
    cols = ["rain", "temp_mean", "pressure", "humidity", "cloud_cover"]
    filas, vistas, cortas = [], 0, 0
    for k, rows in sorted(ult.items()):
        if len(rows) != dias:
            cortas += 1
            continue
        if ciudades is not None and vistas >= ciudades:
            break
        for d, row in enumerate(rows):
            f = {c: row[cab.index(c)] for c in cols}
            f["ciudad"] = f"C{vistas}"
            f["dia"] = (date(2020, 1, 1) + timedelta(days=d)).isoformat()
            filas.append(f)
        vistas += 1
    return filas, {"ciudades": vistas, "ciudades_con_menos_dias_descartadas": cortas, "dias_por_ciudad": dias,
                   "calendario": "reconstruido: 2020-01-01 + posición (el CSV no trae fecha)"}


def cargar_avocado(ruta: Path = DATOS_114 / "43820.arff", max_series: int | None = None) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(RAIZ / "benchmarks" / "fechas_114c5"))
    from medir_114c5 import leer_arff  # noqa: PLC0415  (el lector de ARFF del 114-C5: una sola copia)
    n, _t = leer_arff(ruta)
    filas = [{"AveragePrice": str(r["AveragePrice"]), "Total_Volume": str(r["Total_Volume"]), "Date": r["Date"],
              "serie": f'{r["region"]}|{r["type"]}'} for r in n]
    if max_series is not None:
        quedan = sorted({f["serie"] for f in filas})[:max_series]
        filas = [f for f in filas if f["serie"] in quedan]
    return filas, {"series": len({f["serie"] for f in filas}), "serie": "region|type"}


def cargar_uber(ruta: Path = DATOS_114 / "43746.arff", max_series: int | None = None) -> tuple[list[dict], dict]:
    """NYC Uber pickups (OpenML 43746, CC0). Se quitan las filas SIN distrito: no tienen serie, y su calendario es irregular."""
    sys.path.insert(0, str(RAIZ / "benchmarks" / "fechas_114c5"))
    from medir_114c5 import leer_arff  # noqa: PLC0415
    n, _t = leer_arff(ruta)
    sin = sum(1 for r in n if not r.get("borough"))
    filas = [{"pickup_dt": r["pickup_dt"], "borough": r["borough"], "pickups": str(int(r["pickups"])), "temp": str(r["temp"]),
              "hday": r["hday"]} for r in n if r.get("borough")]
    if max_series is not None:
        quedan = sorted({f["borough"] for f in filas})[:max_series]
        filas = [f for f in filas if f["borough"] in quedan]
    return filas, {"series": len({f["borough"] for f in filas}), "filas_sin_distrito_quitadas": sin}


def datos_de(nombre: str, a) -> tuple[list[dict], dict, Path]:
    d = CONJUNTOS[nombre]["datos"]
    if d == "lluvia":
        f, i = cargar_lluvia(dias=a.dias, ciudades=a.ciudades if a.ciudades else (a.max_series or None))
        return f, i, LLUVIA
    if d == "avocado":
        f, i = cargar_avocado(max_series=a.max_series or None)
        return f, i, DATOS_114 / "43820.arff"
    f, i = cargar_uber(max_series=a.max_series or None)
    return f, i, DATOS_114 / "43746.arff"


def payload_de(nombre: str, filas: list[dict], folds: int, modo: str) -> dict:
    c = CONJUNTOS[nombre]
    campos = [c["tiempo"], c["serie"], c["objetivo"], *c["lag_columns"], *c["extra_inputs"]]
    forecast = {"time_column": c["tiempo"], "series_column": c["serie"], "horizon": c["horizon"], "lags": c["lags"],
                "lag_columns": list(c["lag_columns"]), "season": c["season"]}
    if c["extra_inputs"]:
        forecast["extra_inputs"] = list(c["extra_inputs"])
    p = {"csv_text": a_csv(filas, campos), "objetivo": c["objetivo"], "folds": folds, "repeats": 1,
         "unidad_de_observacion": c["unidad"], "forecast": forecast}
    if modo != "completo":
        p["modo"] = modo
    return p


# ------------------------------------------------------------------------------------------------- lectura del sobre (puro)
def _n(x):
    return None if x is None else float(x)


def resumen_del_sobre(sobre: dict, final: dict) -> dict:
    """Lo que se publica de UN estudio terminado, LEÍDO del sobre (nada se recalcula). Lo ausente sale `None`, nunca 0."""
    p = (sobre.get("pronostico") or final.get("seleccion", {}).get("pronostico") or {})
    nota = p.get("nota") or {}
    prueba = nota.get("prueba")
    conf = sobre.get("conforme") or {}
    cal = conf.get("calibracion") or {}
    cob = conf.get("cobertura_en_test")
    pliegues = [{"pliegue": q.get("pliegue"), "n": q.get("n"),
                 "mejora_sobre_persistencia": _n(q.get("mejora_sobre_persistencia")),
                 "mejora_sobre_estacional": _n(q.get("mejora_sobre_estacional"))} for q in nota.get("pliegues", [])]
    return {
        "campeon": (final.get("seleccion") or {}).get("candidate_engine"),
        "decision": ((final.get("seleccion") or {}).get("decision") or {}).get("outcome"),
        "horizonte_pasos": p.get("horizonte_pasos"), "retardos": p.get("retardos"), "estacion": p.get("estacion"),
        "hueco": p.get("hueco"), "filas": p.get("filas"),
        "reparto": {k: ((final.get("seleccion") or {}).get("reparto") or {}).get(k) for k in ("diseno", "gap", "gap_origen")},
        "prueba": None if prueba is None else {
            "n": prueba.get("n"),
            "mae_modelo": _n((prueba.get("modelo") or {}).get("mae")),
            "mae_persistencia": _n((prueba.get("persistencia") or {}).get("mae")),
            "mae_estacional": _n((prueba.get("estacional") or {}).get("mae")),
            "mejora_mae_sobre_persistencia": _n(prueba.get("mejora_mae_sobre_persistencia")),
            "mejora_mae_sobre_estacional": _n(prueba.get("mejora_mae_sobre_estacional")),
            "mase": _n(prueba.get("mase"))},
        "escala_de_la_mase": _n(nota.get("escala_de_la_mase")),
        "pliegues": pliegues, "veredicto_por_pliegue": nota.get("veredicto_por_pliegue"),
        "intervalo": {"disponible": bool(cal) and cal.get("cuantil") is not None, "cobertura_nominal": (None if cal.get("alfa") is None else round(1.0 - float(cal["alfa"]), 10)),
                      "n_calibracion": cal.get("n_calibracion"), "cuantil": cal.get("cuantil"), "cobertura_medida_en_la_prueba": cob,
                      "sin_intercambiabilidad": conf.get("sin_intercambiabilidad"), "motivo_si_no": conf.get("motivo")},
        "limites": [l.get("clave") for l in p.get("limites", []) if isinstance(l, dict)],
        "motivo_de_la_nota": nota.get("motivo"),
    }


def lectura_en_una_linea(r: dict) -> str:
    """Para el log: una frase con lo medido (sin veredicto de cumple)."""
    t = r.get("prueba")
    if not t:
        return "sin prueba medida"
    v = r.get("veredicto_por_pliegue") or {}
    def f(x):
        return "n/d" if x is None else f"{100 * x:+.1f}%"
    return (f"prueba n={t['n']}: mejora s/persistencia {f(t['mejora_mae_sobre_persistencia'])}, s/estacional "
            f"{f(t['mejora_mae_sobre_estacional'])}, MASE {t['mase'] if t['mase'] is None else round(t['mase'], 3)}; pliegues que gana a "
            f"persistencia {((v.get('persistencia') or {}).get('gana'))}/{((v.get('persistencia') or {}).get('medidos'))}")


# ------------------------------------------------------------------------------------------------------------ el estudio real
def correr_estudio(ej, payload: dict, tope: float) -> dict:
    t0 = time.time()
    r = ej.iniciar_estudio(dict(payload), owner=DUENO)
    estudio = r.get("estudio") or {}
    if not r.get("ok") or estudio.get("estado") == "failed" or "job_id" not in estudio:
        return {"estado": "no_arranco", "rechazo": json.dumps(r, default=str, ensure_ascii=False)[:800]}
    job = estudio["job_id"]
    while True:
        e = ej.estudio_status(job, DUENO)
        if e.get("estado") in ("completed", "failed", "cancelled"):
            break
        if time.time() - t0 > tope:
            ej.cancelar_estudio(job, DUENO)
            time.sleep(5)
            e = ej.estudio_status(job, DUENO)
            e["estado"] = "cancelado_por_tope"
            break
        time.sleep(2)
    sobre = ej.cargar_seleccion(ej._jobs[job]["directorio_checkpoint"]) if job in ej._jobs else None
    return {"estado": e.get("estado"), "job": job, "segundos": round(time.time() - t0, 1), "error": e.get("error"),
            "final": e, "sobre": sobre or {}}


def procedencia() -> dict:
    import matrixai
    import matrixai_engines
    import matrixai_studio.estudio_job as ej

    def sha(modulo_file: str) -> str | None:
        d = Path(modulo_file).resolve()
        for q in [d, *d.parents]:
            if (q / ".git").exists():
                try:
                    return subprocess.run(["git", "-C", str(q), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
                except OSError:
                    return None
        return None
    return {"nucleo": matrixai.__file__, "motores": matrixai_engines.__file__, "backend": ej.__file__,
            "commits": {"nucleo": sha(matrixai.__file__), "motores": sha(matrixai_engines.__file__), "studio": sha(ej.__file__)},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "pythonpath": os.environ.get("PYTHONPATH")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--conjuntos", default="lluvia,avocado,uber_h1,uber_h6")
    ap.add_argument("--dias", type=int, default=730, help="lluvia: últimos N días por ciudad")
    ap.add_argument("--ciudades", type=int, default=0, help="lluvia: solo las N primeras (0 = las 56)")
    ap.add_argument("--max-series", type=int, default=0, help="SOLO sonda: recorta a las N primeras series (0 = todas)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--modo", default="completo", choices=["completo", "rapido"])
    ap.add_argument("--tope-estudio", type=float, default=3600.0)
    a = ap.parse_args(argv)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    os.environ["MATRIXAI_ESTUDIOS_DIR"] = str(salida / "estudios")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import matrixai_studio.estudio_job as ej
    proc = procedencia()
    print("CÓDIGO", proc["nucleo"], proc["motores"], proc["backend"], proc["commits"], flush=True)
    (salida / "procedencia.json").write_text(json.dumps(proc, indent=1))
    for nombre in a.conjuntos.split(","):
        destino = salida / f"c164_c5_{nombre}.json"
        if destino.exists():
            print(nombre, "ya medido, se salta", flush=True)
            continue
        filas, info, ruta = datos_de(nombre, a)
        payload = payload_de(nombre, filas, a.folds, a.modo)
        print(f"=== {nombre}: {len(filas)} filas, modo {a.modo}, folds {a.folds}, h={CONJUNTOS[nombre]['horizon']}", flush=True)
        r = correr_estudio(ej, payload, a.tope_estudio)
        res = {"conjunto": nombre, "config": CONJUNTOS[nombre], "datos": {**info, "filas_dadas_al_estudio": len(filas),
               "fichero": str(ruta), "sha256_del_fichero": sha256_de(ruta),
               "sha256_del_csv_dado": hashlib.sha256(payload["csv_text"].encode()).hexdigest()},
               "parametros": {"folds": a.folds, "repeats": 1, "modo": a.modo, "tope_estudio_s": a.tope_estudio},
               "estado": r["estado"], "segundos": r.get("segundos"), "error": r.get("error"), "rechazo": r.get("rechazo"),
               "ru_maxrss_hijos_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
               "ru_maxrss_proceso_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
        if r["estado"] == "completed":
            res["resultado"] = resumen_del_sobre(r["sobre"], r["final"])
            print(nombre, r["estado"], r["segundos"], "s", res["resultado"]["campeon"], lectura_en_una_linea(res["resultado"]), flush=True)
        else:
            print(nombre, r["estado"], r.get("segundos"), r.get("error") or r.get("rechazo"), flush=True)
        destino.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        import shutil
        shutil.rmtree(salida / "estudios" / str(r.get("job")), ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
