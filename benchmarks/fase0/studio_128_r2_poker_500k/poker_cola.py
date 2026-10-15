"""128-R2, regla (2): poker-hand-500k en CPU (2 CPU, 6 GB, como el servidor; el brazo A del kit 6: TabM en CPU, fuera por tiempo) termina CON
CAMPEÓN y con cifra de test, y los árboles completan sus pliegues. Un ESTUDIO ENTERO por el camino real (`iniciar_estudio` en este proceso,
el hilo del estudio de verdad, un proceso por intento), para la COLA NOCTURNA (`~/encolar.sh`). Estimado 30-45 min.

Qué escribe (en `./resultados-128-r2/`, bajo el directorio de la orden: la cola lo copia a `~/cola-nocturna/resultados/<nombre>/`):
  poker-hand-500k.estado.json   el estado ENTERO del estudio (`estudio_status`)
  veredicto.json / veredicto.txt  la regla (2), punto por punto
Y todo por la salida estándar (el registro de la cola).

Uso (lo pone `encolar.sh`, con PYTHONPATH a los tres worktrees): taskset -c 6,7 python3 poker_cola.py
Guarda `if __name__ == "__main__"`: cada intento es un `multiprocessing.spawn` que re-importa este guion."""
import gzip, hashlib, json, os, shutil, sys, tempfile, time
from pathlib import Path

CSV_GZ = Path("/home/deployer/.agentes/128-r2/datos/poker-hand-500k.csv.gz")
SHA256_GZ = "1a2316ecf90c8c12dbd6054287aa56465e53497074fd0ebb6905f7af9ba7797b"      # el del manifiesto del kit 6
TOPE_S = 4800.0        # el estudio entero; la cola pone el suyo por encima (5400)


def main():
    # `--humo N`: SOLO para probar el guion (las N primeras filas, un estudio de segundos); no mide nada de la regla (2).
    humo = int(sys.argv[sys.argv.index("--humo") + 1]) if "--humo" in sys.argv else None
    salida = Path("resultados-128-r2-humo" if humo else "resultados-128-r2")
    salida.mkdir(exist_ok=True)
    assert hashlib.sha256(CSV_GZ.read_bytes()).hexdigest() == SHA256_GZ, "el CSV no es el del manifiesto del kit 6"
    from matrixai_studio import estudio_job
    import matrixai, matrixai_engines
    esperado = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p.endswith("studio-backend")]
    print("estudio_job:", estudio_job.__file__, "| núcleo:", matrixai.__file__, "| motores:", matrixai_engines.__file__,
          "| afinidad", sorted(os.sched_getaffinity(0)), "| PYTHONPATH studio:", esperado, flush=True)
    assert esperado and estudio_job.__file__.startswith(esperado[0]), "el estudio NO sale del worktree fijado por la cola"
    print("presupuesto por tamaño:", estudio_job._UMBRAL_DE_CELDAS_GRANDE, estudio_job._SEGUNDOS_POR_CELDA_GRANDE,
          "| factor de clases (multiclase, 10):", estudio_job._factor_de_clases("multiclass_classification", 10), flush=True)

    texto = gzip.decompress(CSV_GZ.read_bytes()).decode("utf-8")
    if humo:
        texto = "\n".join(texto.split("\n")[:humo + 1]) + "\n"
    cabecera = texto.split("\n", 1)[0].split(",")
    cuerpo = {"csv_text": texto, "objetivo": "Class", "tarea": "multiclass_classification",
              "entradas": [c for c in cabecera if c != "Class"], "unidad_de_observacion": "una fila"}
    tmp = Path(tempfile.mkdtemp(prefix="128r2-"))
    owner = "128-r2"
    t0 = time.time()
    try:
        r = estudio_job.iniciar_estudio(cuerpo, owner=owner, directorio_de_estudios=tmp)
        job_id = (r.get("estudio") or {}).get("job_id")
        print("iniciar:", {k: v for k, v in (r.get("estudio") or {}).items() if k != "motivo"}, flush=True)
        assert job_id, r
        ultimo = None
        while True:
            est = estudio_job.estudio_status(job_id, owner)
            visto = (est.get("estado"), est.get("n_intentos_completados"))
            if visto != ultimo:
                print(f"[{time.time() - t0:7.0f}s] {visto[0]} intentos completados {visto[1]}/{est.get('n_intentos_total')}", flush=True)
                ultimo = visto
            if est.get("estado") in ("completed", "failed", "cancelled"):
                break
            if time.time() - t0 > TOPE_S:
                print("TOPE del guion: se cancela", flush=True)
                estudio_job.cancelar_estudio(job_id, owner=owner)
                time.sleep(20)
                est = estudio_job.estudio_status(job_id, owner)
                break
            time.sleep(5)
        pared = time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (salida / "poker-hand-500k.estado.json").write_text(json.dumps(est, ensure_ascii=False, indent=1, default=str))
    sel = est.get("seleccion") or {}
    campeon = sel.get("candidate_engine")
    ef = sel.get("evaluacion_final")
    test = ({m.get("metric_id"): m.get("value") for m in (ef.get("metrics") or []) if isinstance(m, dict)}
            if isinstance(ef, dict) else None)
    intentos = {}
    for fila in est.get("leaderboard") or []:
        intentos.setdefault(fila["engine"], []).append(
            {"candidate": fila.get("candidate"), "estado": fila.get("estado"), "wall_seconds": fila.get("wall_seconds"),
             "valor": (fila.get("metrica_de_seleccion") or {}).get("value"),
             "motivo": (fila.get("motivo") or {}).get("es") if isinstance(fila.get("motivo"), dict) else fila.get("motivo")})
    declarada = est.get("presupuesto_segun_el_tamano")
    # la regla (2), punto por punto
    arboles = {m: intentos.get(m, []) for m in ("lightgbm", "sklearn.hgb")}
    pliegues_completos = {m: sum(1 for x in v if x["estado"] == "completed") for m, v in arboles.items()}
    puntos = {
        "estudio_completed": est.get("estado") == "completed",
        "hay_campeon": campeon is not None,
        "hay_cifra_de_test": bool(test) and any(v is not None for v in test.values()),
        "lightgbm_5_de_5": pliegues_completos["lightgbm"] == 5,
        "hgb_5_de_5": pliegues_completos["sklearn.hgb"] == 5,
        "declara_grande_con_clases": bool(declarada) and declarada.get("factor_de_clases") == 10
                                     and declarada.get("coste_de_los_arboles") == 50_000_000,
    }
    veredicto = {"cumple_la_regla_2": all(puntos.values()), "puntos": puntos, "campeon": campeon, "test": test,
                 "pared_s": round(pared, 1), "segundos_reales": est.get("segundos_reales"),
                 "presupuesto_total_segundos": est.get("presupuesto_total_segundos"),
                 "pliegues_completos_de_los_arboles": pliegues_completos,
                 "intentos_por_motor": intentos, "presupuesto_segun_el_tamano": declarada,
                 "motores_fuera_por_plazo": est.get("motores_fuera_por_plazo"),
                 "motores_fuera_por_memoria": est.get("motores_fuera_por_memoria"),
                 "motores_que_no_puntuaron": sel.get("motores_que_no_puntuaron"),
                 "estado": est.get("estado"), "error": est.get("error"), "error_motivo": est.get("error_motivo"),
                 "loadavg_final": open("/proc/loadavg").read().split()[:3]}
    (salida / "veredicto.json").write_text(json.dumps(veredicto, ensure_ascii=False, indent=1, default=str))
    lineas = [f"REGLA (2) DEL 128-R2: {'CUMPLE' if veredicto['cumple_la_regla_2'] else 'NO CUMPLE'}"]
    lineas += [f"  {'ok ' if v else 'MAL'} {k}" for k, v in puntos.items()]
    lineas += [f"  campeón: {campeon} | test: {test} | pared {pared:.0f} s | segundos_reales {est.get('segundos_reales')}"]
    for m, v in intentos.items():
        lineas.append(f"  {m}: " + ", ".join(f"{x['estado']}({x['wall_seconds'] if x['wall_seconds'] is None else round(x['wall_seconds'], 1)}s)" for x in v))
    (salida / "veredicto.txt").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas), flush=True)


if __name__ == "__main__":
    main()
