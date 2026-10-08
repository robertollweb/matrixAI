#!/usr/bin/env python3
"""128 · C0 — dónde se va la MEMORIA y el TIEMPO de un estudio con un conjunto grande, SIN cambiar nada del producto.

Instrumento: la imagen del paquete CPU publicada (`matrixai-studio:v2.12`), un estudio de verdad por HTTP (como
R0/R1 del 120) y, DENTRO del contenedor, un muestreador (`docker exec`) que cada ~1 s lee `/proc/<pid>/status`
(VmRSS, VmHWM) y `/proc/<pid>/cmdline` de CADA proceso (el servidor y cada hijo `multiprocessing.spawn`) y la memoria
del cgroup. NO se usa `tracemalloc` (no ve las arenas de pymalloc: lección del 117-C4). La fase del estudio sale de
lo que dice el estado por HTTP (`estado`, `n_intentos_completados`, `n_intentos_total`) y de cuándo vuelve el POST
de `/estudio/iniciar` (la preparación y el diagnóstico ocurren DENTRO de ese POST, y no se pueden separar desde fuera).

    python3 sonda_128_c0.py --control                      # el instrumento ve una reserva CONOCIDA (±10 %), por proceso
    python3 sonda_128_c0.py --humo                         # dresses-sales, minutos: fases y procesos
    python3 sonda_128_c0.py --solo APSFailure,KDDCup09_appetency --memoria 9g     # la pasada de memoria
    python3 sonda_128_c0.py --solo Allstate_Claims_Severity                       # UN intento por motor, SIN tope
    python3 sonda_128_c0.py                                # los tres, en ese orden (APS, KDDCup09, Allstate)

El intento de Allstate se mide sin cambiar el producto: dentro del contenedor, un guion (`INTENTO_SIN_TOPE`) llama a
`estudio_job.iniciar_estudio` con el mismo cuerpo que el HTTP, sustituye EN MEMORIA (no en disco) `ejecutar_estudio`
por una medida del primer pliegue —un intento de cada motor— y por cada uno mide: construir el intento en el padre
(preparación), el pickle de lo que viaja al hijo, `ejecutar_intento` directo (sin hijo) y `ejecutar_intento_aislado`
(el camino real) con el tope puesto en 100.000 s. Del reparto salen el ajuste (`recursos.wall_seconds`) y «lo demás».

Solo LEE los repos (no escribe, ni deja __pycache__). Los hijos `spawn` re-ejecutan el módulo: todo va tras `__main__`.
"""
import argparse, atexit, csv, gzip, hashlib, importlib.util, json, os, re, signal, socket, subprocess, sys, threading, time
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
NOMBRE_CONTENEDOR = "sonda-128-c0"
IMAGEN = "matrixai-studio:v2.12"
MEMORIA = "9g"
SALIDA = AQUI / "sonda_128_c0.json"
TOPE_ESTUDIO_S = 3600          # por conjunto (sondeo); pasado esto se registra «plazo» y se cancela
CONJUNTOS_DEL_C0 = ("APSFailure", "KDDCup09_appetency", "Allstate_Claims_Severity")
HUMO = ("dresses-sales",)
ARCHIVO_PARAR = "/tmp/parar_128_c0"
ARCHIVO_MUESTRAS = "/tmp/muestras_128_c0.jsonl"
MARCA_CONTROL = "RESERVA_128_C0"
#: Un proceso que RESERVA 1 GB de memoria virtual SIN tocarla (`mmap` anónimo): su RSS es casi cero y su VmSize pasa
#: del giga. Sin él, el control no distinguía leer la residente (VmRSS) de leer la virtual (VmSize): un `bytearray`
#: tocado entero tiene las dos casi iguales (sabotaje del supervisor, 08-10: VmSize en vez de VmRSS, control VERDE).
MARCA_VIRTUAL = "VIRTUAL_128_C0"


def _cargar_por_ruta(nombre, ruta):
    """Por RUTA y sin tocar `sys.path` (como hace el 120): importar esto no cambia qué importa el resto."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


r0 = _cargar_por_ruta("referencia_120_r0", AQUI / "referencia_120_r0.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================ el muestreador (corre DENTRO del contenedor)
MUESTREADOR = r'''
# Muestreador de procesos del contenedor 128-C0. Solo la biblioteca estándar. Una línea JSON por tick.
import json, os, sys, time

UNA_VEZ_Y_SALIR = "--una-vez" in sys.argv


def _kb(texto):
    try:
        return int(texto.split()[0])
    except Exception:
        return None


def leer_proceso(pid):
    """{pid, ppid, rss_mb, hwm_mb, cmd} de un proceso, o None si ya no existe o no se deja leer."""
    try:
        with open(f"/proc/{pid}/status") as f:
            campos = {}
            for linea in f:
                k, _, v = linea.partition(":")
                if k in ("PPid", "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
                    campos[k] = v.strip()
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, ValueError):
        return None
    rss = _kb(campos.get("VmRSS", ""))
    if rss is None:                       # hilo del kernel o zombi: sin RSS
        return None
    mb = lambda k: None if _kb(campos.get(k, "")) is None else round(_kb(campos[k]) / 1024, 1)
    return {"pid": int(pid), "ppid": int(campos.get("PPid", "0") or 0), "rss_mb": round(rss / 1024, 1),
            "hwm_mb": mb("VmHWM"), "anon_mb": mb("RssAnon"), "file_mb": mb("RssFile"), "shmem_mb": mb("RssShmem"),
            "cmd": cmd[:300]}


def _cgroup():
    out = {}
    for ruta, nombre in (("/sys/fs/cgroup/memory.current", "cgroup_mb"), ("/sys/fs/cgroup/memory.peak", "cgroup_pico_mb"),
                         ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "cgroup_mb"),
                         ("/sys/fs/cgroup/memory/memory.max_usage_in_bytes", "cgroup_pico_mb")):
        if nombre in out:
            continue
        try:
            with open(ruta) as f:
                out[nombre] = round(int(f.read().strip()) / 2**20, 1)
        except (OSError, ValueError):
            pass
    return out


def muestrear_una_vez(excluir=None):
    """Una muestra: {t, procesos: [...], cgroup_mb, cgroup_pico_mb}. `excluir`: pids que no cuentan (este muestreador)."""
    excluir = {os.getpid()} if excluir is None else excluir
    procesos = []
    for nombre in os.listdir("/proc"):
        if nombre.isdigit() and int(nombre) not in excluir:
            p = leer_proceso(nombre)
            if p is not None:
                procesos.append(p)
    m = {"t": time.time(), "procesos": sorted(procesos, key=lambda p: p["pid"])}
    m.update(_cgroup())
    return m


def main():
    destino = sys.argv[1]
    parar = sys.argv[2]
    intervalo = 1.0
    siguiente = time.time()
    with open(destino, "a") as f:
        while not os.path.exists(parar):
            f.write(json.dumps(muestrear_una_vez(), separators=(",", ":")) + "\n")
            f.flush()
            siguiente += intervalo
            time.sleep(max(0.0, siguiente - time.time()))


if __name__ == "__main__":
    if UNA_VEZ_Y_SALIR:
        print(json.dumps(muestrear_una_vez()))
    else:
        main()
'''

# ============================================================================ el intento SIN tope (corre DENTRO del contenedor)
INTENTO_SIN_TOPE = r'''
# 128-C0: un intento de cada motor con el tope puesto en 100.000 s, midiendo cada parte. No cambia el producto en disco:
# sustituye `ejecutar_estudio` EN MEMORIA, en este proceso, antes de llamar a `iniciar_estudio` con el cuerpo del HTTP.
import dataclasses, json, pickle, sys, time


def main():
    cuerpo = json.load(open(sys.argv[1]))
    salida = sys.argv[2]
    import matrixai_studio.estudio_job as ej
    from matrixai_engines.harness import ejecutar_intento
    from matrixai_engines.subproceso import ejecutar_intento_aislado, MARGEN_POR_DEFECTO_SEGUNDOS
    R = {"eventos": [], "motores": [], "margen_del_producto_s": MARGEN_POR_DEFECTO_SEGUNDOS}

    def ev(e):
        R["eventos"].append([time.time(), e])

    def recursos(intento):
        r = intento.recursos or {}
        return {"estado": intento.estado, "ajuste_wall_s": r.get("wall_seconds"), "ajuste_cpu_s": r.get("cpu_seconds")}

    def medir(estudio, motores, intentos, *, fase, directorio_checkpoint, cancelado=lambda: False,
              aislar_por_proceso=False):
        claves = list(intentos._claves)
        n = sum(1 for c in claves if c[:2] == claves[0][:2])          # los intentos del primer pliegue: uno por motor
        it = iter(intentos)
        for i in range(n):
            ev(f"construir {i}")
            t0 = time.perf_counter()
            d = next(it)                                                # preparación + particiones, en el padre
            prep = time.perf_counter() - t0
            m = {"motor": d.motor_id, "tope_del_producto_s": d.presupuesto.wall_seconds, "hilos": d.presupuesto.hilos,
                 "filas_train": len(d.train.row_ids), "filas_validation": len(d.validation.row_ids) if d.validation else 0,
                 "filas_test": len(d.test.row_ids), "columnas_tras_preparar": len(d.train.features[0]),
                 "preparacion_en_el_padre_s": round(prep, 2)}
            motor = motores[d.motor_id]
            sin_tope = dataclasses.replace(d.presupuesto, wall_seconds=100000.0)
            ev(f"pickle {i}")
            t0 = time.perf_counter()
            blob = pickle.dumps((motor, d.train, d.validation, d.test, d.spec), protocol=pickle.HIGHEST_PROTOCOL)
            m["pickle_s"] = round(time.perf_counter() - t0, 2)
            m["pickle_mb"] = round(len(blob) / 2**20, 1)
            del blob
            comun = dict(candidate=d.candidate_id(), split_plan_digest=d.split_plan_digest, dataset=d.dataset,
                         pliegue=d.pliegue, repeticion=d.repeticion)
            ev(f"directo {i}")
            t0 = time.perf_counter()
            a = ejecutar_intento(motor, d.train, d.validation, d.test, d.spec, sin_tope, **comun)
            m["directo_total_s"] = round(time.perf_counter() - t0, 2)
            m["directo"] = recursos(a)
            ev(f"aislado {i}")
            t0 = time.perf_counter()
            b = ejecutar_intento_aislado(motor, d.train, d.validation, d.test, d.spec, sin_tope, **comun)
            m["aislado_total_s"] = round(time.perf_counter() - t0, 2)
            m["aislado"] = recursos(b)
            ev(f"fin {i}")
            tope = m["tope_del_producto_s"]
            m["tope_con_margen_s"] = None if tope is None else round(tope + R["margen_del_producto_s"], 1)
            m["cabe_en_el_tope"] = None if tope is None else m["aislado_total_s"] <= tope + R["margen_del_producto_s"]
            if m["aislado"]["ajuste_wall_s"] is not None:
                m["aislado_menos_ajuste_s"] = round(m["aislado_total_s"] - m["aislado"]["ajuste_wall_s"], 2)
            R["motores"].append(m)
            json.dump(R, open(salida, "w"), indent=1)
            del d
        ev("medido")
        json.dump(R, open(salida, "w"), indent=1)
        return ej.Leaderboard(estudio_id=estudio.estudio_id)

    ej.ejecutar_estudio = medir
    ej._reajustar_en_un_proceso_aparte = lambda *a, **k: None
    ev("iniciar")
    t0 = time.perf_counter()
    r = ej.iniciar_estudio(cuerpo)
    R["iniciar_estudio_s"] = round(time.perf_counter() - t0, 2)   # lo que tarda hasta devolver el job (preparación + diagnóstico)
    R["respuesta"] = {k: v for k, v in (r.get("estudio") or r).items() if k in ("estado", "job_id", "clave", "error")}
    ev("iniciado")
    job = ej._jobs.get((r.get("estudio") or {}).get("job_id"))
    if job is None:
        json.dump(R, open(salida, "w"), indent=1)
        sys.exit(2)
    job["hilo"].join()
    ev("hilo terminado")
    json.dump(R, open(salida, "w"), indent=1)


if __name__ == "__main__":
    main()
'''


# ============================================================================ lo que no necesita Docker: roles, fases, resumen
def rol_de(cmd):
    """El papel de un proceso del contenedor según su línea de órdenes (`/proc/<pid>/cmdline` con espacios)."""
    c = (cmd or "").strip()
    if not c:
        return "otro"
    primero = os.path.basename(c.split()[0])
    if "muestreador_128_c0" in c:
        return "sonda"
    if MARCA_VIRTUAL in c:
        return "control_virtual"
    if MARCA_CONTROL in c:
        return "control"
    if "intento_sin_tope_128_c0" in c:
        return "guion_del_intento"
    if primero in ("docker-init", "tini") or "docker-init" in c:
        return "init"
    if primero == "runc":
        return "exec_de_docker"
    if "multiprocessing.resource_tracker" in c or "resource_tracker" in c:
        return "resource_tracker"
    if "multiprocessing.spawn" in c or "spawn_main" in c:
        return "hijo_spawn"
    if primero in ("sh", "dash", "bash"):
        return "sh"
    if "matrixai_studio" in c:
        return "servidor"
    return "otro"


def fase_en(eventos, t):
    """La etiqueta del último evento con tiempo <= t (`eventos`: [(t, etiqueta)] ordenados); «antes» si no hay."""
    et = "antes"
    for te, e in eventos:
        if te <= t:
            et = e
        else:
            break
    return et


_RX_INTENTO = re.compile(r"^intento (\d+)/(\d+)$")
_RX_PARTE = re.compile(r"^(construir|pickle|directo|aislado) (\d+)$")


def grupo_de(etiqueta, motores_por_intento=None):
    """La fase agrupada: «intento k/N» pasa a «intento:<motor>» si se sabe el motor del intento k (inferido del orden
    del leaderboard final); si no, «intento»."""
    p = _RX_PARTE.match(etiqueta)             # el guion del intento sin tope: «directo 2» -> «directo:<motor>»
    if p:
        motor = (motores_por_intento or {}).get(int(p.group(2)))
        return f"{p.group(1)}:{motor}" if motor else p.group(1)
    m = _RX_INTENTO.match(etiqueta)
    if not m:
        return etiqueta
    motor = (motores_por_intento or {}).get(int(m.group(1)) - 1)
    return f"intento:{motor}" if motor else "intento"


def resumir(muestras, eventos, motores_por_intento=None, rol=rol_de):
    """De la serie (lista de {t, procesos:[{pid,ppid,rss_mb,hwm_mb,cmd}], cgroup_mb?}) y los eventos de fase al resumen:
    el pico por proceso, por rol y por fase, el pico total y cuánto pesa cada hijo frente al servidor.

    La suma de RSS de los procesos cuenta dos veces lo compartido (páginas de una biblioteca, copy-on-write): el pico
    del CGROUP (`cgroup_mb`) es lo que mira el OOM-killer, y se da aparte."""
    eventos = sorted(eventos, key=lambda e: e[0])
    por_pid, por_fase, pico_total, pico_cgroup = {}, {}, None, None
    for m in muestras:
        fase = fase_en(eventos, m["t"])
        grupo = grupo_de(fase, motores_por_intento)
        procs = [dict(p, rol=rol(p.get("cmd", ""))) for p in m["procesos"]]
        procs = [p for p in procs if p["rol"] not in ("sonda",)]
        total = round(sum(p["rss_mb"] for p in procs), 1)
        srv = max([p["rss_mb"] for p in procs if p["rol"] == "servidor"] or [0.0])
        if pico_total is None or total > pico_total["rss_total_mb"]:
            pico_total = {"rss_total_mb": total, "t": m["t"], "fase": fase, "grupo": grupo,
                          "procesos": {str(p["pid"]): [p["rol"], p["rss_mb"]] for p in procs}}
        cg = m.get("cgroup_mb")
        if cg is not None and (pico_cgroup is None or cg > pico_cgroup["cgroup_mb"]):
            pico_cgroup = {"cgroup_mb": cg, "t": m["t"], "fase": fase, "grupo": grupo}
        g = por_fase.setdefault(grupo, {"muestras": 0, "t_ini": m["t"], "t_fin": m["t"], "pico_total_mb": 0.0,
                                        "pico_servidor_mb": 0.0, "pico_hijo_mb": 0.0, "pico_cgroup_mb": None})
        g["muestras"] += 1
        g["t_fin"] = m["t"]
        g["pico_total_mb"] = max(g["pico_total_mb"], total)
        g["pico_servidor_mb"] = max(g["pico_servidor_mb"], srv)
        g["pico_hijo_mb"] = max([g["pico_hijo_mb"]] + [p["rss_mb"] for p in procs if p["rol"] == "hijo_spawn"])
        if cg is not None:
            g["pico_cgroup_mb"] = cg if g["pico_cgroup_mb"] is None else max(g["pico_cgroup_mb"], cg)
        for p in procs:
            q = por_pid.setdefault(p["pid"], {"pid": p["pid"], "ppid": p["ppid"], "rol": p["rol"], "cmd": p["cmd"][:160],
                                              "primera_t": m["t"], "ultima_t": m["t"], "pico_rss_mb": 0.0,
                                              "pico_hwm_mb": None, "fase_del_pico": None, "servidor_en_el_pico_mb": 0.0,
                                              "grupo_al_nacer": grupo})
            q["ultima_t"] = m["t"]
            if p["rss_mb"] >= q["pico_rss_mb"]:
                q["pico_rss_mb"], q["fase_del_pico"], q["servidor_en_el_pico_mb"] = p["rss_mb"], fase, srv
            if p.get("hwm_mb") is not None:
                q["pico_hwm_mb"] = p["hwm_mb"] if q["pico_hwm_mb"] is None else max(q["pico_hwm_mb"], p["hwm_mb"])
    pico_servidor = max([q["pico_rss_mb"] for q in por_pid.values() if q["rol"] == "servidor"] or [0.0])
    for q in por_pid.values():
        q["vive_s"] = round(q["ultima_t"] - q["primera_t"], 1)
        if q["rol"] == "hijo_spawn":
            q["sobre_pico_del_servidor"] = round(q["pico_rss_mb"] / pico_servidor, 3) if pico_servidor else None
            q["sobre_servidor_en_ese_instante"] = (round(q["pico_rss_mb"] / q["servidor_en_el_pico_mb"], 3)
                                                  if q["servidor_en_el_pico_mb"] else None)
    por_rol = {}
    for q in por_pid.values():
        r = por_rol.setdefault(q["rol"], {"procesos_distintos": 0, "pico_rss_mb": 0.0})
        r["procesos_distintos"] += 1
        r["pico_rss_mb"] = max(r["pico_rss_mb"], q["pico_rss_mb"])
    hijos = [q for q in por_pid.values() if q["rol"] == "hijo_spawn"]
    for g in por_fase.values():
        g["duracion_s"] = round(g["t_fin"] - g["t_ini"], 1)
        for k in ("t_ini", "t_fin"):
            del g[k]
    return {"muestras": len(muestras), "pico_total": pico_total, "pico_cgroup": pico_cgroup,
            "pico_servidor_mb": pico_servidor, "por_rol": por_rol, "por_fase": por_fase,
            "hijo_mayor": (max(hijos, key=lambda q: q["pico_rss_mb"]) if hijos else None),
            "por_proceso": sorted(por_pid.values(), key=lambda q: q["primera_t"])}


def comprobar_virtual(muestras, techo_mb=80.0, rol=rol_de):
    """El control de la memoria VIRTUAL: el proceso que reserva 1 GB sin tocarlo se ve, y con RSS por debajo de `techo_mb`
    (si la sonda leyera la virtual, lo vería por encima del giga). Devuelve (ok, línea)."""
    picos = {}
    for m in muestras:
        for p in m["procesos"]:
            if rol(p.get("cmd", "")) == "control_virtual":
                picos[p["pid"]] = max(picos.get(p["pid"], 0.0), p["rss_mb"])
    if len(picos) != 1:
        return False, f"reservado 1 GB virtual sin tocar: se esperaba UN proceso y se vieron {len(picos)}  NO CUADRA"
    pid, visto = next(iter(picos.items()))
    ok = visto < techo_mb
    return ok, (f"reservado 1 GB virtual sin tocar: visto {visto} MB de residente (pid {pid}; tiene que ser < {techo_mb:.0f})  "
                f"{'OK' if ok else 'NO CUADRA'}")


def comprobar_reservas(muestras, esperadas_mb, tolerancia=0.10, rol=rol_de):
    """El control: cada reserva conocida (en MB) la ve el muestreador como UN proceso aparte (rol `control`) con un pico de
    RSS dentro de ±`tolerancia` de lo reservado. Devuelve (ok, líneas, medidas). Cada medida se empareja con UN proceso
    (el de pico más cercano, sin repetir)."""
    picos = {}
    for m in muestras:
        for p in m["procesos"]:
            if rol(p.get("cmd", "")) == "control":
                picos[p["pid"]] = max(picos.get(p["pid"], 0.0), p["rss_mb"])
    libres = dict(picos)
    ok, lin, medidas = True, [], []
    for esperada in sorted(esperadas_mb):
        if not libres:
            ok = False
            lin.append(f"reservado {esperada} MB: ningún proceso de control visto  NO CUADRA")
            continue
        pid = min(libres, key=lambda k: abs(libres[k] - esperada))
        visto = libres.pop(pid)
        cuadra = abs(visto - esperada) <= tolerancia * esperada
        ok &= cuadra
        medidas.append({"pid": pid, "reservado_mb": esperada, "visto_mb": visto})
        lin.append(f"reservado {esperada} MB: visto {visto} MB (pid {pid}, {100 * (visto - esperada) / esperada:+.1f} %)  "
                   f"{'OK' if cuadra else 'NO CUADRA'}")
    if len(picos) != len(esperadas_mb):
        ok = False
        lin.append(f"se esperaban {len(esperadas_mb)} procesos de control y se vieron {len(picos)}  NO CUADRA")
    return ok, lin, medidas


def motores_por_intento_de(leaderboard):
    """{índice del intento (0..): motor}. Los intentos de la búsqueda se ejecutan en orden y el leaderboard guarda las
    entradas en ese orden: el índice k de «intento k+1/N» es la k-ésima entrada de fase `seleccion`. INFERIDO, no
    declarado por el estado: el JSON lo marca así."""
    ent = [x for x in leaderboard or [] if x.get("fase") in (None, "seleccion")]
    return {i: x.get("engine") for i, x in enumerate(ent)}


def etiqueta_de_estado(est, total_previo=None):
    """La fase que dice el estado de un estudio por HTTP."""
    e = est.get("estado")
    if e == "queued":
        return "cola"
    total = est.get("n_intentos_total") or total_previo
    k = est.get("n_intentos_completados") or 0
    if e == "running":
        return f"intento {k + 1}/{total}" if total and k < total else "reajuste_final"
    return "fin"


# ============================================================================ Docker
_contenedor = None


def borrar_contenedor():
    global _contenedor
    if _contenedor:
        subprocess.run(["docker", "rm", "-f", _contenedor], capture_output=True)
        _contenedor = None


def comprobar_que_se_puede():
    la = float(open("/proc/loadavg").read().split()[0])
    if la > 6:
        raise SystemExit(f"PARO: loadavg {la} > 6; esperar (CLAUDE.md)")
    ps = subprocess.run(["docker", "ps", "--format", "{{.Names}} {{.Image}}"], capture_output=True, text=True).stdout
    otros = [x for x in ps.splitlines() if re.search(r"playwright|^r0-120-|^sonda-|matrixai-studio:", x)
             and not x.startswith(NOMBRE_CONTENEDOR + " ")]
    if otros:
        raise SystemExit(f"PARO: hay otro contenedor de medición vivo: {otros}")
    return la


def arrancar_contenedor(memoria, imagen=None):
    """Un contenedor con NOMBRE PROPIO, techo de memoria sin swap y `--init`. Lo de un intento anterior con el mismo nombre
    se borra antes (un contenedor sobrevive al `timeout` de su guion). Devuelve (base_url, nombre)."""
    global _contenedor
    subprocess.run(["docker", "rm", "-f", NOMBRE_CONTENEDOR], capture_output=True)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); puerto = s.getsockname()[1]; s.close()
    _contenedor = NOMBRE_CONTENEDOR
    atexit.register(borrar_contenedor)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: (borrar_contenedor(), os._exit(143)))
    r = subprocess.run(["docker", "run", "-d", "--init", "--name", NOMBRE_CONTENEDOR, "-p", f"127.0.0.1:{puerto}:8765",
                        f"--memory={memoria}", f"--memory-swap={memoria}", "--cpus=2",
                        "-e", "MATRIXAI_LICENSE_ENABLED=false", imagen or IMAGEN],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("docker run falló: " + r.stderr)
    base = f"http://127.0.0.1:{puerto}"
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            r0.pedir(base, "/api/studio/status", plazo=5)
            return base, NOMBRE_CONTENEDOR
        except Exception:
            time.sleep(2)
    raise SystemExit("el servidor no contestó en 120 s")


def poner_en_el_contenedor(nombre, texto, destino):
    """Escribe `texto` en `destino` dentro del contenedor (docker cp de un fichero temporal en el directorio de la sonda)."""
    tmp = AGENTE / f"_{os.getpid()}_{os.path.basename(destino)}"
    tmp.write_text(texto)
    try:
        r = subprocess.run(["docker", "cp", str(tmp), f"{nombre}:{destino}"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("docker cp falló: " + r.stderr)
    finally:
        tmp.unlink(missing_ok=True)


AGENTE = Path.home() / ".agentes" / "128-c0"


class Muestreador:
    """El muestreador DENTRO del contenedor: se lanza con `docker exec -d`, escribe en ARCHIVO_MUESTRAS y se para con el
    fichero ARCHIVO_PARAR. `leer()` trae la serie al host."""

    def __init__(self, nombre):
        self.nombre = nombre

    def __enter__(self):
        poner_en_el_contenedor(self.nombre, MUESTREADOR, "/tmp/muestreador_128_c0.py")
        subprocess.run(["docker", "exec", self.nombre, "rm", "-f", ARCHIVO_PARAR, ARCHIVO_MUESTRAS], capture_output=True)
        r = subprocess.run(["docker", "exec", "-d", self.nombre, "python3", "/tmp/muestreador_128_c0.py",
                            ARCHIVO_MUESTRAS, ARCHIVO_PARAR], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("no arrancó el muestreador: " + r.stderr)
        t0 = time.time()
        while time.time() - t0 < 30:        # hasta que escribe su primera muestra
            if self.leer():
                return self
            time.sleep(0.5)
        raise SystemExit("el muestreador no escribió ninguna muestra en 30 s")

    def leer(self):
        r = subprocess.run(["docker", "exec", self.nombre, "cat", ARCHIVO_MUESTRAS], capture_output=True, text=True)
        out = []
        for linea in r.stdout.splitlines():
            try:
                out.append(json.loads(linea))
            except ValueError:
                pass                          # la última línea puede estar a medias
        return out

    def __exit__(self, *_):
        subprocess.run(["docker", "exec", self.nombre, "touch", ARCHIVO_PARAR], capture_output=True)
        time.sleep(1.5)


# ============================================================================ el control: una reserva CONOCIDA
def control(memoria="2g"):
    """El instrumento tiene dientes: dos procesos que reservan 300 y 600 MB (un `bytearray` tocado entero) y la sonda
    tiene que verlos como procesos APARTE con ese tamaño ±10 %, y ver el servidor."""
    comprobar_que_se_puede()
    base, nombre = arrancar_contenedor(memoria)
    reservas = (300, 600)
    codigo = ("import time;{marca}=1;b=bytearray({mb}*1024*1024)\n"
              "for i in range(0,len(b),4096): b[i]=1\n"
              "time.sleep({s})")
    try:
        with Muestreador(nombre) as mu:
            time.sleep(3)
            lanzados = []
            for mb in reservas:
                subprocess.run(["docker", "exec", "-d", nombre, "python3", "-c",
                                codigo.format(marca=MARCA_CONTROL, mb=mb, s=14)], check=True)
                lanzados.append(mb)
            subprocess.run(["docker", "exec", "-d", nombre, "python3", "-c",
                            f"import time,mmap;{MARCA_VIRTUAL}=1;m=mmap.mmap(-1,1024*1024*1024);time.sleep(14)"],
                           check=True)
            time.sleep(20)
            muestras = mu.leer()
        ok, lin, medidas = comprobar_reservas(muestras, reservas)
        ok_v, lin_v = comprobar_virtual(muestras)
        ok &= ok_v
        lin.append(lin_v)
        roles = {}
        for m in muestras:
            for p in m["procesos"]:
                roles.setdefault(rol_de(p["cmd"]), set()).add(p["pid"])
        ve_servidor = bool(roles.get("servidor"))
        if not ve_servidor:
            ok = False
            lin.append("no se ve el proceso del servidor  NO CUADRA")
        sep = {pid for pid in roles.get("control", set())} & ({pid for s in ("servidor", "sh", "init") for pid in roles.get(s, set())})
        if sep:
            ok = False
            lin.append(f"un control comparte pid con otro rol: {sep}  NO CUADRA")
        res = {"cuadra": ok, "lineas": lin, "reservas_mb": list(reservas), "medidas": medidas,
               "roles_vistos": {k: sorted(v) for k, v in roles.items()}, "muestras": len(muestras),
               "resumen": resumir(muestras, [(muestras[0]["t"], "control")])["por_rol"]}
        print("\n".join(lin)); print("CONTROL:", "CUADRA" if ok else "NO CUADRA", flush=True)
        return res
    finally:
        borrar_contenedor()


# ============================================================================ estudios por HTTP
def estudiar(base, nombre_c, mu, nombre, c, gen, tope_s):
    """Un estudio de verdad por HTTP con el muestreador dentro. Devuelve el registro con el resumen de memoria."""
    csv_bytes, cols, excl, nfilas = r0.csv_de(gen, c["data_id"])
    cuerpo = {"csv_text": csv_bytes.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    eventos = [(time.time(), "preparacion")]
    ini, la, t0 = ahora(), open("/proc/loadavg").read().split()[:3], time.time()
    resp = {}
    hilo = threading.Thread(target=lambda: resp.update(zip(("http", "r"), r0.pedir_o_rechazo(base, "/api/studio/estudio/iniciar", cuerpo))),
                            daemon=True)
    hilo.start()
    while hilo.is_alive():
        hilo.join(timeout=2)
        if time.time() - t0 > tope_s:
            break
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "objetivo": c["objetivo"],
           "csv": {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes), "excluidas_por_string": excl},
           "inicio": ini, "loadavg_al_empezar": la, "iniciar_post_s": round(time.time() - t0, 1)}
    r = resp.get("r") or {}
    job = (r.get("job_id") or (r.get("estudio") or {}).get("job_id")) if isinstance(r, dict) else None
    if resp.get("http") != 200 or not job:
        rec.update(estado="rechazado" if resp else "plazo_en_la_preparacion", http=resp.get("http"), rechazo=r)
        return rec, mu.leer(), eventos
    eventos.append((time.time(), "cola"))
    est, total, vistos = {}, None, []
    while True:
        try:
            est = r0.pedir(base, f"/api/studio/estudio/estado/{job}", plazo=60)
        except Exception as e:
            est = {"estado": "sondeo_fallido", "error": repr(e)}
        total = est.get("n_intentos_total") or total
        if est.get("estado") in ("completed", "failed", "cancelled", "sondeo_fallido"):
            eventos.append((time.time(), "fin"))
            break
        et = etiqueta_de_estado(est, total)
        if not eventos or eventos[-1][1] != et:
            eventos.append((time.time(), et))
        if time.time() - t0 > tope_s:
            try: r0.pedir(base, f"/api/studio/estudio/cancelar/{job}", {}, 30)
            except Exception: pass
            est = dict(est, estado="plazo"); eventos.append((time.time(), "fin")); break
        time.sleep(2)
    pared = time.time() - t0
    _, filas = r0.intentos_de(est)
    sel = est.get("seleccion") or {}
    rec.update(job_id=job, http=200, estado=est.get("estado"), pared_estudio_s=round(pared, 1), error=est.get("error"),
               campeon=sel.get("candidate_engine"), media_de_la_seleccion=sel.get("media_de_la_seleccion"),
               n_intentos_total=total, intentos=filas, fin=ahora(),
               motores_fuera_por_memoria=est.get("motores_fuera_por_memoria"))
    if est.get("estado") in ("sondeo_fallido",):
        rec["contenedor"] = r0.estado_del_contenedor(nombre_c)
    return rec, mu.leer(), eventos


def medir_estudio(nombre, c, gen, memoria, tope_s, dir_series):
    comprobar_que_se_puede()
    base, nombre_c = arrancar_contenedor(memoria)
    try:
        dentro = r0.dentro(nombre_c, "import matrixai, matrixai_studio.estudio_job as j; print('matrixai', matrixai.__version__, j.__file__)")
        with Muestreador(nombre_c) as mu:
            rec, muestras, eventos = estudiar(base, nombre_c, mu, nombre, c, gen, tope_s)
            time.sleep(2)
            muestras = mu.leer() or muestras
        # el contenedor, ANTES de borrarlo: murió por memoria?
        rec["contenedor"] = r0.estado_del_contenedor(nombre_c)
        rec["dentro"] = dentro
        return cerrar_registro(rec, muestras, eventos, dir_series, nombre, c, memoria)
    finally:
        borrar_contenedor()


def cerrar_registro(rec, muestras, eventos, dir_series, nombre, c, memoria):
    lb = None
    # los motores de cada intento, inferidos del orden de las entradas de la búsqueda
    if rec.get("intentos"):
        lb = [{"engine": x["motor"], "fase": x.get("fase")} for x in rec["intentos"]]
    mpi = motores_por_intento_de(lb)
    rec["motor_de_cada_intento_inferido"] = {str(k + 1): v for k, v in mpi.items()}
    if (rec.get("medida") or {}).get("motores"):       # el guion del intento sin tope dice su motor, no se infiere
        mpi = {i: m["motor"] for i, m in enumerate(rec["medida"]["motores"])}
    rec["resumen_de_memoria"] = resumir(muestras, eventos, mpi)
    rec["memoria_del_contenedor"] = memoria
    rec["eventos_de_fase"] = [[round(t, 1), e] for t, e in eventos]
    dir_series.mkdir(exist_ok=True)
    f = dir_series / f"{nombre}.series.json.gz"
    with gzip.open(f, "wt") as fh:
        json.dump({"muestras": muestras, "eventos": eventos}, fh, separators=(",", ":"))
    rec["serie_temporal"] = {"fichero": f.name, "sha256": sha(f), "muestras": len(muestras)}
    return rec


def medir_intento_sin_tope(nombre, c, gen, memoria, dir_series):
    comprobar_que_se_puede()
    csv_bytes, cols, excl, nfilas = r0.csv_de(gen, c["data_id"])
    cuerpo = {"csv_text": csv_bytes.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    ruta = AGENTE / f"_{os.getpid()}_cuerpo.json"
    ruta.write_text(json.dumps(cuerpo))
    base, nombre_c = arrancar_contenedor(memoria)
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "inicio": ahora(),
           "csv": {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes)},
           "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
           "sha256_guion_dentro": hashlib.sha256(INTENTO_SIN_TOPE.encode()).hexdigest()}
    try:
        r = subprocess.run(["docker", "cp", str(ruta), f"{nombre_c}:/tmp/cuerpo_128_c0.json"], capture_output=True, text=True)
        ruta.unlink(missing_ok=True)
        if r.returncode:
            raise SystemExit("docker cp falló: " + r.stderr)
        poner_en_el_contenedor(nombre_c, INTENTO_SIN_TOPE, "/tmp/intento_sin_tope_128_c0.py")
        with Muestreador(nombre_c) as mu:
            eventos = [(time.time(), "arranque_del_guion")]
            t0 = time.time()
            p = subprocess.run(["docker", "exec", nombre_c, "python3", "/tmp/intento_sin_tope_128_c0.py",
                                "/tmp/cuerpo_128_c0.json", "/tmp/intento_128_c0.json"], capture_output=True, text=True)
            rec["guion_rc"], rec["guion_pared_s"] = p.returncode, round(time.time() - t0, 1)
            rec["guion_stderr_cola"] = p.stderr[-1500:]
            time.sleep(2)
            muestras = mu.leer()
        out = subprocess.run(["docker", "exec", nombre_c, "cat", "/tmp/intento_128_c0.json"], capture_output=True, text=True)
        try:
            med = json.loads(out.stdout)
        except ValueError:
            med = {"error": "el guion no escribió su salida", "stdout": out.stdout[-500:]}
        rec["medida"] = med
        eventos += [(t, e) for t, e in med.get("eventos", [])]
        rec["contenedor"] = r0.estado_del_contenedor(nombre_c)
        rec["estado"] = "completed" if p.returncode == 0 and len(med.get("motores", [])) else "failed"
        rec["fin"] = ahora()
        return cerrar_registro(rec, muestras, eventos, dir_series, nombre, c, memoria)
    finally:
        ruta.unlink(missing_ok=True)
        borrar_contenedor()


# ============================================================================ salida
def guardar(salida, datos):
    tmp = salida.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1))
    os.replace(tmp, salida)


def linea_de_resumen(rec):
    rs = rec.get("resumen_de_memoria") or {}
    pt, pc = rs.get("pico_total") or {}, rs.get("pico_cgroup") or {}
    hm = rs.get("hijo_mayor") or {}
    return (f"   {rec.get('estado')} {rec.get('pared_estudio_s', rec.get('guion_pared_s'))} s | pico RSS total {pt.get('rss_total_mb')} MB "
            f"({pt.get('grupo')}) | cgroup {pc.get('cgroup_mb')} MB | servidor {rs.get('pico_servidor_mb')} MB | "
            f"hijo mayor {hm.get('pico_rss_mb')} MB ({hm.get('sobre_pico_del_servidor')}x el servidor)")


def imprimir_intento(rec):
    print(linea_de_resumen(rec), flush=True)
    for m in (rec.get("medida") or {}).get("motores", []):
        print(f"   {m['motor']}: prep padre {m['preparacion_en_el_padre_s']} s, pickle {m['pickle_s']} s ({m['pickle_mb']} MB), "
              f"directo {m['directo_total_s']} s, aislado {m['aislado_total_s']} s (ajuste {m['aislado']['ajuste_wall_s']} s), "
              f"tope {m['tope_con_margen_s']} s", flush=True)


def main():
    global IMAGEN, SALIDA
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="", help="conjuntos separados por comas (de los del banco del 120)")
    ap.add_argument("--memoria", default=MEMORIA)
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--imagen", default=IMAGEN)
    ap.add_argument("--humo", action="store_true", help="dresses-sales (y su intento sin tope): minutos")
    ap.add_argument("--control", action="store_true", help="una reserva CONOCIDA (300 y 600 MB) y la sonda tiene que verla")
    ap.add_argument("--tope-estudio", type=int, default=TOPE_ESTUDIO_S)
    a = ap.parse_args()
    IMAGEN, SALIDA = a.imagen, Path(a.salida).resolve()
    AGENTE.mkdir(parents=True, exist_ok=True)
    imagen_id = subprocess.run(["docker", "image", "inspect", IMAGEN, "--format", "{{.Id}}"], capture_output=True, text=True).stdout.strip()
    if not imagen_id:
        raise SystemExit(f"PARO: la imagen {IMAGEN} no está cargada")
    gen = r0.cargar_generador()
    banco = {b["nombre"]: b for b in json.loads(r0.BORRADOR.read_text())["conjuntos"]["banco"]}
    declarados = {d["nombre"]: d for d in json.loads(r0.SONDA.read_text())["datasets_declarados"]}
    for n, b in banco.items():
        for k in ("data_id", "tarea", "objetivo", "clase_positiva"):
            if str(declarados[n][k]) != str(b[k]):
                raise SystemExit(f"PARO: {n}.{k}: borrador {b[k]!r} != 119-C5a {declarados[n][k]!r}")
    procedencia = {"imagen": IMAGEN, "imagen_id": imagen_id, "sha256_guion": sha(__file__),
                   "sha256_referencia_120_r0": sha(r0.__file__), "sha256_generador_de_csv": sha(r0.GENERADOR),
                   "sha256_muestreador": hashlib.sha256(MUESTREADOR.encode()).hexdigest(),
                   "sha256_intento_sin_tope": hashlib.sha256(INTENTO_SIN_TOPE.encode()).hexdigest(),
                   "memoria_del_contenedor": a.memoria, "cpus": 2, "inicio": ahora(),
                   "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
                   "instrumento": "estudio por HTTP; RSS por proceso de /proc dentro del contenedor cada ~1 s; sin tracemalloc",
                   "limites": ["la preparación y el diagnóstico ocurren dentro del POST de iniciar: una sola fase «preparacion»",
                               "un proceso que viva menos de ~1 s puede no aparecer",
                               "el motor de cada intento es INFERIDO del orden del leaderboard final",
                               "la suma de RSS cuenta dos veces lo compartido; el pico del cgroup es lo que mira el OOM"]}
    datos = {"procedencia": procedencia, "control": None, "conjuntos": {}}
    if a.control:
        datos["control"] = control()
        procedencia["fin"] = ahora()
        guardar(SALIDA, datos)
        sys.exit(0 if datos["control"]["cuadra"] else 1)
    nombres = list(HUMO) if a.humo else [x for x in a.solo.split(",") if x] or list(CONJUNTOS_DEL_C0)
    for n in nombres:
        if n not in banco:
            raise SystemExit(f"PARO: {n} no está en el banco del 120")
    dir_series = SALIDA.with_name(SALIDA.stem + "_series")
    for n in nombres:
        c = banco[n]
        print(f"[{ahora()}] {n} (loadavg {open('/proc/loadavg').read().split()[0]}, memoria {a.memoria}) ...", flush=True)
        if n == "Allstate_Claims_Severity":
            rec = medir_intento_sin_tope(n, c, gen, a.memoria, dir_series)
            datos["conjuntos"][n + ":intento_sin_tope"] = rec
            guardar(SALIDA, datos)
            imprimir_intento(rec)
            continue
        rec = medir_estudio(n, c, gen, a.memoria, a.tope_estudio, dir_series)
        datos["conjuntos"][n] = rec
        guardar(SALIDA, datos)
        print(linea_de_resumen(rec), flush=True)
        if a.humo:                                # el humo también prueba el camino del intento sin tope, con el mismo conjunto
            rec2 = medir_intento_sin_tope(n, c, gen, a.memoria, dir_series)
            datos["conjuntos"][n + ":intento_sin_tope"] = rec2
            guardar(SALIDA, datos)
            imprimir_intento(rec2)
    procedencia["fin"] = ahora()
    guardar(SALIDA, datos)


if __name__ == "__main__":
    main()
