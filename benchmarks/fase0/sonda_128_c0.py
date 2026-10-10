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
    python3 sonda_128_c0.py --retencion --humo             # 128-C1 paso 0: ¿devuelve el asignador lo que se suelta? (gc + malloc_trim)
    python3 sonda_128_c0.py                                # los tres, en ese orden (APS, KDDCup09, Allstate)
    python3 sonda_128_c0.py --tabm-forzado --memoria 6g    # 128-C3b: los tres con la guarda de TabM callada EN MEMORIA
    python3 sonda_128_c0.py --tabm-por-fases --humo        # 128-R4: un intento de TabM por FASES (dos plazos), dresses-sales
    python3 sonda_128_c0.py --tabm-por-fases --solo APSFailure,KDDCup09_appetency,Allstate_Claims_Severity --memoria 6g --tope-estudio 5400
    # 128-R4 en GPU (el KIT 128-C5; ver su LEEME): los CSV del kit, con la GPU, y la sonda DENTRO del contenedor de la imagen GPU
    python3 sonda_128_c0.py --tabm-por-fases --gpu --datos-del-kit kit_128_c5/datos --temporales /ruta/tmp --solo APSFailure
    docker run --gpus all … python3 /work/kit/sonda_128_c0.py --tabm-por-fases --gpu --dentro-del-contenedor --datos-del-kit /work/kit/datos …

El intento de Allstate se mide sin cambiar el producto: dentro del contenedor, un guion (`INTENTO_SIN_TOPE`) llama a
`estudio_job.iniciar_estudio` con el mismo cuerpo que el HTTP, sustituye EN MEMORIA (no en disco) `ejecutar_estudio`
por una medida del primer pliegue —un intento de cada motor— y por cada uno mide: construir el intento en el padre
(preparación), el pickle de lo que viaja al hijo, `ejecutar_intento` directo (sin hijo) y `ejecutar_intento_aislado`
(el camino real) con el tope puesto en 100.000 s. Del reparto salen el ajuste (`recursos.wall_seconds`) y «lo demás».

128-R4 (`--tabm-por-fases`): ¿en qué se va el tiempo de UN intento de TabM en un conjunto grande y cuánto necesita para entrenar
de verdad? El guion `TABM_POR_FASES` ejecuta el intento del primer pliegue «directo» (en el proceso del guion) dos veces, con el plazo
del producto y con uno amplio (`--plazo-amplio`, 1.800 s), con temporizadores puestos EN MEMORIA que envuelven funciones del motor y
de `preparacion_tabm`/`tabm_plr`. Los hijos `spawn` re-importan del disco: solo el intento directo lleva temporizadores.

Solo LEE los repos (no escribe, ni deja __pycache__). Los hijos `spawn` re-ejecutan el módulo: todo va tras `__main__`.
"""
import argparse, atexit, csv, gzip, hashlib, importlib.util, io, json, math, os, re, signal, socket, subprocess, sys, threading, time
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
PLAZO_AMPLIO_S = 1800.0        # 128-R4: el plazo «de sobra» de la segunda corrida de --tabm-por-fases
#: 128-R4 en GPU: los cinco conjuntos que decide el plazo de TabM en el paquete GPU (los tres grandes, donde en CPU no llega a
#: parar por paciencia ni con 1.800 s, y los dos medianos, que en CPU sí paran: 126 y 202 s).
CONJUNTOS_DE_FASES_EN_GPU = ("APSFailure", "KDDCup09_appetency", "Allstate_Claims_Severity", "house_16H", "mfeat-factors")
VARIABLE_DEL_DISPOSITIVO = "MATRIXAI_TABM_DISPOSITIVO"
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
            # 128-C1 paso 4 (M1 de su auditoría): con las particiones en un fichero la descripción lleva
            # `train/validation/test = None` y `en_fichero`; aquí se CARGAN (es lo que carga el hijo del producto).
            # Antes el guion leía `d.train` directamente, el hilo del estudio moría con AttributeError y la sonda
            # salía con `motores: []` y código 0, sin decir nada.
            if getattr(d, "en_fichero", None) is not None:
                train, validation, test = d.en_fichero.cargar()
            else:
                train, validation, test = d.train, d.validation, d.test
            m = {"motor": d.motor_id, "tope_del_producto_s": d.presupuesto.wall_seconds, "hilos": d.presupuesto.hilos,
                 "filas_train": len(train.row_ids), "filas_validation": len(validation.row_ids) if validation else 0,
                 "filas_test": len(test.row_ids), "columnas_tras_preparar": len(train.features[0]),
                 "preparacion_en_el_padre_s": round(prep, 2)}
            motor = motores[d.motor_id]
            sin_tope = dataclasses.replace(d.presupuesto, wall_seconds=100000.0)
            ev(f"pickle {i}")
            t0 = time.perf_counter()
            blob = pickle.dumps((motor, train, validation, test, d.spec), protocol=pickle.HIGHEST_PROTOCOL)
            m["pickle_s"] = round(time.perf_counter() - t0, 2)
            m["pickle_mb"] = round(len(blob) / 2**20, 1)
            del blob
            comun = dict(candidate=d.candidate_id(), split_plan_digest=d.split_plan_digest, dataset=d.dataset,
                         pliegue=d.pliegue, repeticion=d.repeticion)
            ev(f"directo {i}")
            t0 = time.perf_counter()
            a = ejecutar_intento(motor, train, validation, test, d.spec, sin_tope, **comun)
            m["directo_total_s"] = round(time.perf_counter() - t0, 2)
            m["directo"] = recursos(a)
            ev(f"aislado {i}")
            t0 = time.perf_counter()
            b = ejecutar_intento_aislado(motor, train, validation, test, d.spec, sin_tope, **comun)
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
            del d, train, validation, test
        if hasattr(intentos, "soltar"):
            intentos.soltar()                                           # y el fichero de particiones, si lo hay
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
    if not R["motores"]:
        # Una sonda que no midió NADA no sale con 0: el hilo del estudio puede haber muerto por dentro (el job lo dice).
        R["error"] = ("el intento sin tope no midió ningún motor; estado del job: "
                      f"{job.get('estado')!r}, error: {job.get('error')!r}")
        json.dump(R, open(salida, "w"), indent=1)
        print(R["error"], file=sys.stderr)
        sys.exit(3)
    json.dump(R, open(salida, "w"), indent=1)


if __name__ == "__main__":
    main()
'''


# ============================================================================ la RETENCIÓN (corre DENTRO del contenedor) — 128-C1 paso 0
RETENCION = r'''
# 128-C1 paso 0 (diagnóstico, no cambia el producto): ¿devuelve el asignador lo que se suelta? Corre un estudio ENTERO en
# este proceso (el mismo `iniciar_estudio` que el HTTP, con sus hijos de verdad) y, al terminar, lee la memoria del
# proceso antes y después de `gc.collect()` y de `malloc_trim(0)`. Lo que `malloc_trim` devuelve es memoria que el
# asignador (glibc) retenía libre; lo que `gc` suelta es memoria de objetos Python inalcanzables en ciclos.
import ctypes, gc, json, sys, time


def estado_del_proceso():
    campos = {}
    for linea in open("/proc/self/status"):
        k, _, v = linea.partition(":")
        if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
            campos[k] = round(int(v.split()[0]) / 1024, 1)
    roll = {}
    try:
        for linea in open("/proc/self/smaps_rollup"):
            k, _, v = linea.partition(":")
            if k in ("Rss", "Pss", "Anonymous", "Private_Dirty", "Shared_Dirty", "Private_Clean"):
                roll[k] = round(int(v.split()[0]) / 1024, 1)
    except OSError:
        pass
    return {"status_mb": campos, "smaps_rollup_mb": roll}


class MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
                                               "uordblks", "fordblks", "keepcost")]


def mallinfo(libc):
    try:
        libc.mallinfo2.restype = MallInfo2
        m = libc.mallinfo2()
    except AttributeError:
        return None
    mb = lambda x: round(x / 2**20, 1)
    return {"arena_mb": mb(m.arena), "en_uso_uordblks_mb": mb(m.uordblks), "libre_fordblks_mb": mb(m.fordblks),
            "mmap_hblkhd_mb": mb(m.hblkhd), "trozos_libres": m.ordblks}


def main():
    cuerpo = json.load(open(sys.argv[1]))
    salida = sys.argv[2]
    import matrixai_studio.estudio_job as ej
    libc = ctypes.CDLL("libc.so.6")
    R = {"pasos": [], "eventos": []}

    def paso(nombre, **extra):
        e = {"paso": nombre, "t": time.time(), **estado_del_proceso(), "mallinfo2": mallinfo(libc)}
        e.update(extra)
        R["pasos"].append(e)
        json.dump(R, open(salida, "w"), indent=1)

    paso("0_antes_de_iniciar")
    r = ej.iniciar_estudio(cuerpo)
    job = ej._jobs.get((r.get("estudio") or {}).get("job_id"))
    paso("1_tras_iniciar_la_preparacion_y_el_diagnostico")
    if job is None:
        R["respuesta"] = r
        json.dump(R, open(salida, "w"), indent=1)
        sys.exit(2)
    job["hilo"].join()
    R["estado_final"] = job.get("estado") if isinstance(job, dict) else str(type(job))
    paso("2_estudio_terminado_tal_cual")
    paso("3_otra_lectura_sin_hacer_nada")           # control: la lectura en sí no cambia nada
    n = gc.collect()
    paso("4_tras_gc_collect", objetos_recogidos=n)
    t0 = time.perf_counter()
    devuelto = libc.malloc_trim(0)
    paso("5_tras_malloc_trim_0", malloc_trim_devolvio=int(devuelto), malloc_trim_s=round(time.perf_counter() - t0, 3))
    gc.collect()
    libc.malloc_trim(0)
    paso("6_tras_una_segunda_pasada")
    P = {p["paso"]: p for p in R["pasos"]}
    rss = lambda k: (P[k]["status_mb"] or {}).get("VmRSS")
    R["resumen"] = {"rss_terminado_mb": rss("2_estudio_terminado_tal_cual"), "rss_tras_gc_mb": rss("4_tras_gc_collect"),
                    "rss_tras_trim_mb": rss("5_tras_malloc_trim_0"), "pico_hwm_mb": (P["2_estudio_terminado_tal_cual"]["status_mb"] or {}).get("VmHWM"),
                    "devuelto_por_gc_mb": round(rss("2_estudio_terminado_tal_cual") - rss("4_tras_gc_collect"), 1),
                    "devuelto_por_malloc_trim_mb": round(rss("4_tras_gc_collect") - rss("5_tras_malloc_trim_0"), 1)}
    json.dump(R, open(salida, "w"), indent=1)


if __name__ == "__main__":
    main()
'''

TABM_FORZADO = r'''
# 128-C3b (medida, no cambia el producto): ¿cabe TabM en un conjunto GRANDE con la memoria de la medida? Corre un estudio
# ENTERO en este proceso (el mismo `iniciar_estudio` que el HTTP, con sus hijos de verdad) con la GUARDA DE MEMORIA de TabM
# (`_memoria_de_tabm_que_no_cabe`, su punto de sustitución declarado) cambiada EN MEMORIA por una que nunca habla: el disco
# del producto no se toca, y los hijos `spawn` (que re-importan del disco) no deciden quién compite. Apunta lo que la guarda
# DE VERDAD habría dicho y, cada ~2 s, la fase (para repartir la memoria por motor). Si al terminar no hay NINGÚN intento de
# TabM, la medida no midió lo que dice (la sustitución no llegó): sale con 3.
import json, os, sys, time


def main():
    cuerpo = json.load(open(sys.argv[1]))
    salida = sys.argv[2]
    import matrixai_studio.estudio_job as ej
    R = {"guarda_de_verdad": [], "estados": [], "motor_tabm": ej._MOTOR_TABM}
    de_verdad = ej._memoria_de_tabm_que_no_cabe

    def callada(filas, columnas_de_entrada):
        R["guarda_de_verdad"].append({"filas": filas, "columnas_de_entrada": columnas_de_entrada,
                                      "habria_dicho_estimada_y_disponible_gb": de_verdad(filas, columnas_de_entrada)})
        return None

    ej._memoria_de_tabm_que_no_cabe = callada
    R["t_inicio"] = time.time()
    r = ej.iniciar_estudio(cuerpo)
    R["t_tras_iniciar"] = time.time()
    job_id = (r.get("estudio") or {}).get("job_id")
    job = ej._jobs.get(job_id)
    if job is None:
        R["respuesta"] = r
        escribir(R, salida)
        sys.exit(2)
    visto = None
    while True:                                   # como el sondeo del HTTP: el estado cada ~2 s
        vivo = job["hilo"].is_alive()
        est = ej.estudio_status(job_id)
        ahora = (est.get("estado"), est.get("n_intentos_completados"), est.get("n_intentos_total"))
        if ahora != visto:
            R["estados"].append([time.time(), *ahora])
            visto = ahora
            escribir(R, salida)                   # si el OOM se lleva ESTE proceso, las fases ya vistas quedan
        if not vivo:
            break
        job["hilo"].join(timeout=2.0)
    R["t_fin"] = time.time()
    R["estado_final"] = est
    R["intentos_de_tabm"] = sum(1 for e in est.get("leaderboard") or [] if e.get("engine") == ej._MOTOR_TABM)
    escribir(R, salida)
    if not R["intentos_de_tabm"]:
        sys.exit(3)


def escribir(R, salida):
    tmp = salida + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1, default=str)
    os.replace(tmp, salida)


if __name__ == "__main__":
    main()
'''

TABM_POR_FASES = r'''
# 128-R4 (medida, no cambia el producto): ¿EN QUÉ se va el tiempo de UN intento de TabM en un conjunto grande y CUÁNTO necesita
# para entrenar de verdad? Corre `iniciar_estudio` en este proceso (el mismo que el HTTP, con la guarda de memoria de TabM
# callada EN MEMORIA como hace TABM_FORZADO), sustituye `ejecutar_estudio` por una medida del primer pliegue y ahí ejecuta el
# intento de TabM «directo» (`ejecutar_intento`, en ESTE proceso) DOS veces: con el plazo del producto y con uno amplio.
#
# Los TEMPORIZADORES se ponen EN MEMORIA, envolviendo funciones del motor y de sus módulos en ESTE proceso (se quitan al acabar
# cada corrida): el disco del producto no se toca, y los hijos `spawn` (que re-importan del disco) NO los llevan -- por eso solo
# el intento directo está medido por fases. Cada fase guarda segundos de pared y RSS/VmHWM al entrar y al salir. VmHWM se
# REINICIA al entrar en cada fase hoja (`/proc/self/clear_refs` <- 5) para que el pico sea el de la fase; antes de reiniciar,
# el pico se pliega en las fases abiertas que la envuelven (así una envolvente conserva su pico entero).
#
# Por ÉPOCA: el entrenamiento y el checkpoint no son funciones, son el bucle de `_ajustar_con_torch`. Se derivan de dos marcas:
# la validación (`perdida_del_ensamblado_por_lotes`, envuelta) y la primera llamada a `_ahora` tras una marca (el bucle la
# llama al empezar cada época y justo después del checkpoint). entrenar = (entrada a la validación) - (primer `_ahora` tras la
# marca anterior); checkpoint = (primer `_ahora` tras la validación) - (salida de la validación).
import contextlib, dataclasses, functools, gc, json, math, os, sys, time


def leer_memoria():
    """(VmRSS, VmHWM) de este proceso en MB, o (None, None) si no se puede leer."""
    rss = hwm = None
    try:
        with open("/proc/self/status") as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    rss = round(int(linea.split()[1]) / 1024, 1)
                elif linea.startswith("VmHWM:"):
                    hwm = round(int(linea.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return rss, hwm


def reiniciar_hwm():
    """Pone VmHWM al RSS de ahora. True si el núcleo lo dejó."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class Traza:
    """Las fases de UNA corrida, con su anidamiento. Una fase HOJA reinicia VmHWM al entrar; una ENVOLVENTE no (su pico es el
    máximo de lo que pasó dentro)."""

    def __init__(self, reloj=time.perf_counter, memoria=leer_memoria, reiniciar=reiniciar_hwm):
        self.reloj, self.memoria, self._reiniciar = reloj, memoria, reiniciar
        self.fases, self.pila, self.epocas = [], [], []
        self.optimizador_s = None
        self.hwm_se_reinicia = None

    def reiniciar_hwm(self):
        """Devuelve el VmHWM de antes de reiniciar y lo pliega en las fases abiertas."""
        _, hwm = self.memoria()
        for f in self.pila:
            f["_acc"] = max(f["_acc"], hwm or 0.0)
        ok = bool(self._reiniciar())
        self.hwm_se_reinicia = ok if self.hwm_se_reinicia is None else (self.hwm_se_reinicia and ok)
        return hwm

    @contextlib.contextmanager
    def fase(self, nombre, envolvente=False):
        rss0, hwm0 = self.memoria()
        previo = None if envolvente else self.reiniciar_hwm()
        reg = {"i": len(self.fases), "nombre": nombre, "envolvente": bool(envolvente),
               "ruta": "/".join([f["nombre"] for f in self.pila] + [nombre]),
               "padre": self.pila[-1]["i"] if self.pila else None,
               "rss_ini_mb": rss0, "hwm_ini_mb": hwm0, "hwm_previo_mb": previo, "_acc": 0.0,
               "t_pared": time.time()}
        self.fases.append(reg)
        self.pila.append(reg)
        t0 = self.reloj()
        try:
            yield reg
        finally:
            t1 = self.reloj()
            rss1, hwm1 = self.memoria()
            self.pila.pop()
            pico = max(reg.pop("_acc"), hwm1 or 0.0)
            reg.update(wall_s=round(t1 - t0, 4), rss_fin_mb=rss1, hwm_fin_mb=hwm1, pico_mb=round(pico, 1) if pico else None)
            for f in self.pila:
                f["_acc"] = max(f["_acc"], pico)


def instalar(traza, densa, prep, plr):
    """Envuelve EN MEMORIA lo que mide. Un nombre que no existe en el motor ES UN ERROR (AttributeError), no una fase que no
    se mide: una medida que calla porque la función cambió de nombre no mide lo que dice. Devuelve `desinstalar()`."""
    guardado = []
    st = {"pendiente": None, "t_fin_disp": None, "t_fin_val": None, "inicio_entreno": None, "mejor": math.inf}

    def poner(objeto, atributo, nuevo):
        guardado.append((objeto, atributo, atributo in vars(objeto), getattr(objeto, atributo)))
        setattr(objeto, atributo, nuevo)

    def envolver(objeto, atributo, nombre, envolvente=False):
        original = getattr(objeto, atributo)

        @functools.wraps(original)
        def w(*a, **k):
            with traza.fase(nombre, envolvente):
                return original(*a, **k)
        poner(objeto, atributo, w)

    def desinstalar():
        for objeto, atributo, propio, original in reversed(guardado):
            if propio:
                setattr(objeto, atributo, original)
            else:
                delattr(objeto, atributo)
        guardado.clear()
    try:
        motor_cls = densa.MotorDensaTabM
        for atributo, nombre in (("fit", "fit"), ("_ajustar", "ajustar"), ("predict", "predict"), ("predict_proba", "predict_proba"),
                                 ("decision_scores", "decision_scores"), ("_forward", "forward")):
            envolver(motor_cls, atributo, nombre, envolvente=True)
        envolver(motor_cls, "_reconstruir_red", "reconstruir_red")
        envolver(densa, "_a_tensor", "a_tensor")
        envolver(densa, "_tensor_a_json", "exportar_pesos")
        envolver(prep, "ajustar_preparacion", "ajustar_preparacion", envolvente=True)
        envolver(prep, "_tipificar", "tipificar")
        envolver(prep, "aplicar_preparacion", "aplicar_preparacion")
        envolver(prep, "ajustar_estandarizacion_objetivo", "ajustar_estandarizacion_objetivo")
        envolver(prep, "aplicar_estandarizacion_objetivo", "aplicar_estandarizacion_objetivo")

        original_disp = densa._a_dispositivo

        def a_dispositivo(*a, **k):
            with traza.fase("a_dispositivo"):
                r = original_disp(*a, **k)
            st["pendiente"], st["t_fin_disp"] = "optimizador", traza.reloj()      # lo siguiente: AdamW, generadores, primer _ahora
            return r
        poner(densa, "_a_dispositivo", a_dispositivo)

        original_ahora = densa._ahora

        def ahora():
            if st["pendiente"] is not None:
                t = traza.reloj()
                if st["pendiente"] == "optimizador":
                    traza.optimizador_s = round(t - st["t_fin_disp"], 4)
                elif traza.epocas:
                    traza.epocas[-1]["checkpoint_s"] = round(t - st["t_fin_val"], 4)
                st["inicio_entreno"], st["pendiente"] = t, None
            return original_ahora()
        poner(densa, "_ahora", ahora)

        original_val = plr.perdida_del_ensamblado_por_lotes

        def validar(*a, **k):
            entra = traza.reloj()
            entrenar = None if st["inicio_entreno"] is None else round(entra - st["inicio_entreno"], 4)
            with traza.fase("validar") as reg:
                perdida = original_val(*a, **k)
            mejora = isinstance(perdida, (int, float)) and math.isfinite(perdida) and perdida < st["mejor"]
            if mejora:
                st["mejor"] = perdida
            traza.epocas.append({"n": len(traza.epocas) + 1, "entrenar_s": entrenar, "validar_s": reg["wall_s"],
                                 "checkpoint_s": None, "perdida_val": perdida, "mejoro": bool(mejora),
                                 "rss_mb": reg["rss_fin_mb"], "hwm_entrenar_mb": reg["hwm_previo_mb"], "t_pared": time.time()})
            st["pendiente"], st["t_fin_val"], st["inicio_entreno"] = "checkpoint", traza.reloj(), None
            return perdida
        poner(plr, "perdida_del_ensamblado_por_lotes", validar)

        original_red = plr.RedTabMPLR

        class RedTabMPLR(original_red):
            def __init__(self, *a, **k):
                with traza.fase("construir_red"):
                    super().__init__(*a, **k)
        RedTabMPLR.__name__, RedTabMPLR.__qualname__ = original_red.__name__, original_red.__qualname__
        poner(plr, "RedTabMPLR", RedTabMPLR)
    except BaseException:
        desinstalar()                  # a medias NO se deja nada puesto en el motor
        raise
    return desinstalar


def escribir(R, salida):
    tmp = salida + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1, default=str)
    os.replace(tmp, salida)


def buscar(obj, clave, hondura=4):
    """El valor de `clave` en un dict anidado (la declaración de lo que PASÓ en el entrenamiento: épocas, mejor época, si paró
    por plazo o por paciencia). `None` si no está: no se inventa."""
    if isinstance(obj, dict) and hondura:
        if clave in obj:
            return obj[clave]
        for v in obj.values():
            r = buscar(v, clave, hondura - 1)
            if r is not None:
                return r
    return None


#: 128-R4 en GPU: lo que el MOTOR declara de su dispositivo (`hiperparametros` y `entrenamiento_efectivo` del predictor). Se LEE
#: de lo que el motor dijo; si no lo dice, la clave no está (un valor ausente no es «cpu»).
CLAVES_DEL_DISPOSITIVO = ("device", "dispositivo", "dispositivo_pedido", "gpu", "determinista", "tf32", "tf32_cudnn",
                          "optimizador_implementacion", "dropout_desde", "caida_a_cpu", "clase_de_la_caida")


def declaracion_de_dispositivo(config):
    """El dispositivo REAL que declaró el motor de TabM en esta corrida (no el que se pidió con la variable de entorno)."""
    return {k: v for k in CLAVES_DEL_DISPOSITIVO for v in [buscar(config, k)] if v is not None}


def pidio_gpu():
    """True si el entorno del producto pide algo distinto de la CPU. Con la CPU (lo de siempre) NO se toca `torch.cuda`: la
    medida de la CPU no cambia por mirar una GPU que no se pidió."""
    return (os.environ.get("MATRIXAI_TABM_DISPOSITIVO") or "cpu").strip().lower() != "cpu"


def vram_reiniciar():
    """Pone a cero el pico de VRAM de torch antes de una corrida (el intento corre en ESTE proceso, así que torch lo ve)."""
    if not pidio_gpu():
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def vram_leer():
    """La VRAM de la corrida: picos de torch (asignada y reservada) y, si hay `nvidia-smi`, lo que dice el controlador (incluye
    el contexto de CUDA). `None` si no se pidió la GPU; `{"disponible": False}` si se pidió y torch no la ve."""
    if not pidio_gpu():
        return None
    import subprocess
    R = {"disponible": False, "gpu": None, "total_mb": None, "max_asignada_mb": None, "max_reservada_mb": None, "nvidia_smi": None}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            R.update(disponible=True, gpu=p.name, total_mb=round(p.total_memory / 2**20, 1),
                     max_asignada_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                     max_reservada_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1))
    except Exception as e:
        R["error_torch"] = repr(e)[:300]
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if o.returncode == 0 and o.stdout.strip():
            R["nvidia_smi"] = o.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return R


def main():
    cuerpo = json.load(open(sys.argv[1]))
    salida = sys.argv[2]
    plazo_amplio = float(sys.argv[3])
    import matrixai_studio.estudio_job as ej
    from matrixai_engines.harness import ejecutar_intento
    import matrixai_engines.motores.densa_tabm as densa
    import matrixai_engines.redes.preparacion_tabm as prep
    import matrixai_engines.redes.tabm_plr as plr
    R = {"corridas": [], "guarda_de_verdad": [], "motor_tabm": ej._MOTOR_TABM, "plazo_amplio_s": plazo_amplio,
         "densa_tabm": getattr(densa, "__file__", None), "estudio_job": getattr(ej, "__file__", None),
         "dispositivo_pedido_por_entorno": os.environ.get("MATRIXAI_TABM_DISPOSITIVO")}
    de_verdad = ej._memoria_de_tabm_que_no_cabe

    def callada(filas, columnas_de_entrada):
        R["guarda_de_verdad"].append({"filas": filas, "columnas_de_entrada": columnas_de_entrada,
                                      "habria_dicho_estimada_y_disponible_gb": de_verdad(filas, columnas_de_entrada)})
        return None

    ej._memoria_de_tabm_que_no_cabe = callada

    def medir(estudio, motores, intentos, *, fase, directorio_checkpoint, cancelado=lambda: False,
              aislar_por_proceso=False):
        claves = list(intentos._claves)
        elegidas = [c for c in claves if c[:2] == claves[0][:2] and c[2] == ej._MOTOR_TABM]
        if not elegidas:
            R["error"] = "TabM no tiene intento en el primer pliegue (la guarda callada no llegó, o está fuera por su biblioteca)"
            escribir(R, salida)
            return ej.Leaderboard(estudio_id=estudio.estudio_id)
        previa = Traza()
        with previa.fase("construir_intento"):                  # la preparación del PRODUCTO en el padre (compartida con los árboles)
            d = intentos._construir(*elegidas[0])
        R["construccion"] = previa.fases
        motor = motores[d.motor_id]
        R["tope_del_producto_s"] = d.presupuesto.wall_seconds
        R["hilos"] = d.presupuesto.hilos
        comun = dict(candidate=d.candidate_id(), split_plan_digest=d.split_plan_digest, dataset=d.dataset,
                     pliegue=d.pliegue, repeticion=d.repeticion)
        for rotulo, plazo in (("plazo_del_producto", d.presupuesto.wall_seconds), ("plazo_amplio", plazo_amplio)):
            traza = Traza()
            c = {"rotulo": rotulo, "plazo_s": plazo, "rss_ini_mb": leer_memoria()[0], "error": None}
            vram_reiniciar()
            desinstalar = instalar(traza, densa, prep, plr)
            t0 = time.perf_counter()
            try:
                with traza.fase("cargar_particion"):             # lo que carga el hijo del producto
                    if getattr(d, "en_fichero", None) is not None:
                        train, validation, test = d.en_fichero.cargar()
                    else:
                        train, validation, test = d.train, d.validation, d.test
                c["filas"] = {"train": len(train.row_ids), "validation": len(validation.row_ids) if validation else 0,
                              "test": len(test.row_ids), "columnas_tras_preparar": len(train.features[0])}
                with traza.fase("ejecutar_intento", envolvente=True):
                    a = ejecutar_intento(motor, train, validation, test, d.spec,
                                         dataclasses.replace(d.presupuesto, wall_seconds=plazo), **comun)
                r = a.recursos or {}
                c.update(estado=a.estado, motivo=a.motivo_del_estado, recursos={"wall_seconds": r.get("wall_seconds"),
                         "cpu_seconds": r.get("cpu_seconds")}, entrenamiento_efectivo=buscar(a.config_efectiva, "entrenamiento_efectivo"),
                         dispositivo_declarado=declaracion_de_dispositivo(a.config_efectiva))
            except Exception as e:                               # una corrida rota no quita la otra
                c.update(estado="excepcion", error=repr(e)[:600])
            finally:
                desinstalar()
            c["pared_s"] = round(time.perf_counter() - t0, 4)
            c["vram"] = vram_leer()
            c.update(fases=traza.fases, epocas=traza.epocas, optimizador_s=traza.optimizador_s,
                     hwm_se_reinicia=traza.hwm_se_reinicia, rss_fin_mb=leer_memoria()[0])
            R["corridas"].append(c)
            escribir(R, salida)
            train = validation = test = a = None
            gc.collect()
        escribir(R, salida)
        return ej.Leaderboard(estudio_id=estudio.estudio_id)

    ej.ejecutar_estudio = medir
    ej._reajustar_en_un_proceso_aparte = lambda *a, **k: None
    R["t_inicio"] = time.time()
    t0 = time.perf_counter()
    r0 = leer_memoria()
    r = ej.iniciar_estudio(cuerpo)
    R["iniciar_estudio"] = {"wall_s": round(time.perf_counter() - t0, 2), "rss_ini_mb": r0[0], "rss_fin_mb": leer_memoria()[0],
                            "hwm_fin_mb": leer_memoria()[1]}
    job = ej._jobs.get((r.get("estudio") or {}).get("job_id"))
    if job is None:
        R["respuesta"] = r
        escribir(R, salida)
        sys.exit(2)
    job["hilo"].join()
    R["t_fin"] = time.time()
    escribir(R, salida)
    if len(R["corridas"]) != 2:
        sys.exit(3)


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
    if "retencion_128_c1" in c:
        return "guion_de_retencion"
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


def eventos_de_los_estados(med):
    """128-C3b. Del sondeo que hace DENTRO el guion de TabM forzado (`[t, estado, completados, total]`) a los eventos de fase
    de la sonda («preparacion», «intento k/N», «reajuste_final», «fin»), con la misma `etiqueta_de_estado` que el sondeo
    por HTTP. Sin `t_fin` (el OOM se llevó el guion) no hay «fin»: la última fase vista sigue abierta."""
    eventos = [(med["t_inicio"], "preparacion")] if med.get("t_inicio") else []
    total = None
    for t, estado, k, n in med.get("estados") or []:
        total = n or total
        et = etiqueta_de_estado({"estado": estado, "n_intentos_completados": k, "n_intentos_total": n}, total)
        if et != "fin" and (not eventos or eventos[-1][1] != et):
            eventos.append((t, et))
    if med.get("t_fin"):
        eventos.append((med["t_fin"], "fin"))
    return eventos


def rol_con_el_guion_de_servidor(cmd):
    """128-C3b. En la medida de TabM forzado el estudio corre DENTRO del guion: ese proceso hace de servidor (tiene el job y
    lanza los hijos). El contenedor arranca SIN servidor HTTP (`arrancar_contenedor(sin_servidor=True)`); si aun así
    apareciera uno (otra imagen, otro arranque), sale aparte como «servidor_ocioso» y no se confunde con el que mide."""
    if "tabm_forzado_128_c3b" in (cmd or ""):
        return "servidor"
    r = rol_de(cmd)
    return "servidor_ocioso" if r == "servidor" else r


def veredicto_tabm_forzado(rc, med):
    """128-C3b. `completed`: el estudio terminó Y TabM compitió. `sin_tabm`: la sustitución no llegó (el guion sale con 3, o
    hay estado final sin ningún intento de TabM): la medida no mide lo que dice, y NO es un «completed». `failed`: lo demás,
    también morir por memoria (lo dicen el contenedor y el `memory.events` del cgroup)."""
    med = med or {}
    est = med.get("estado_final")
    if rc == 3 or (est is not None and not med.get("intentos_de_tabm")):
        return "sin_tabm"
    if rc == 0 and (est or {}).get("estado") == "completed":
        return "completed"
    return "failed"


# ============================================================================ 128-R4: las fases de TabM (sin Docker)
#: Lo que `TABM_POR_FASES` tuvo que llamar para que la corrida «mida lo que dice». Antes de la primera época SIEMPRE se llega
#: a estas (aunque el plazo se agote preparando); las de después, solo si hubo candidato.
FASES_SIEMPRE = ("fit", "ajustar", "ajustar_preparacion", "aplicar_preparacion", "a_tensor", "construir_red", "a_dispositivo")
FASES_CON_CANDIDATO = ("validar", "exportar_pesos", "predict")
#: Cuánto puede NO cuadrar una suma de fases con su total: el mayor de 1 s y el 5 %. Declarado, no escondido.
MARGEN_ABS_S, MARGEN_REL = 1.0, 0.05


def rol_con_el_guion_por_fases(cmd):
    """128-R4. El guion `TABM_POR_FASES` hace de servidor (tiene el job): como `rol_con_el_guion_de_servidor` del C3b."""
    if "tabm_por_fases_128_r4" in (cmd or ""):
        return "servidor"
    r = rol_de(cmd)
    return "servidor_ocioso" if r == "servidor" else r


def _margen(total):
    return max(MARGEN_ABS_S, MARGEN_REL * abs(total or 0.0))


def _suma(fases, nombre, padre=None):
    return round(sum(f.get("wall_s") or 0.0 for f in fases if f["nombre"] == nombre
                     and (padre is None or f["padre"] == padre)), 4)


def _mediana(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2, 4)


def partes_del_ajuste(c):
    """De una corrida (`fases` + `epocas` del guion) a EN QUÉ se va `ajustar` (= el reloj del plazo de TabM) y si la suma cuadra.
    `otros_s` es lo que no está en ninguna parte (poner la red en su sitio, AdamW, `load_state_dict`, armar el predictor, los
    hilos…): se da, no se esconde; tiene que caber en el margen o la corrida NO cuadra. Si no hubo `ajustar` (la corrida
    murió antes), devuelve None."""
    fases, epocas = c.get("fases") or [], c.get("epocas") or []
    aj = next((f for f in fases if f["nombre"] == "ajustar"), None)
    if aj is None or aj.get("wall_s") is None:
        return None
    hijas = lambda n: _suma(fases, n, aj["i"])
    p = {"ajustar_s": aj["wall_s"],
         "preparar_s": round(hijas("ajustar_preparacion") + hijas("aplicar_preparacion")
                             + hijas("ajustar_estandarizacion_objetivo") + hijas("aplicar_estandarizacion_objetivo"), 4),
         "tipificar_s": _suma(fases, "tipificar"),                # dentro de preparar_s: no se suma dos veces
         "a_tensor_s": hijas("a_tensor"), "construir_red_s": hijas("construir_red"), "a_dispositivo_s": hijas("a_dispositivo"),
         "optimizador_s": c.get("optimizador_s") or 0.0,
         "entrenar_s": round(sum(e.get("entrenar_s") or 0.0 for e in epocas), 4),
         "validar_s": round(sum(e.get("validar_s") or 0.0 for e in epocas), 4),
         "checkpoint_s": round(sum(e.get("checkpoint_s") or 0.0 for e in epocas), 4),
         "exportar_pesos_s": hijas("exportar_pesos")}
    conocido = sum(p[k] for k in ("preparar_s", "a_tensor_s", "construir_red_s", "a_dispositivo_s", "optimizador_s",
                                  "entrenar_s", "validar_s", "checkpoint_s", "exportar_pesos_s"))
    p["otros_s"] = round(p["ajustar_s"] - conocido, 4)
    p["otros_caben"] = -_margen(p["ajustar_s"]) <= p["otros_s"] <= _margen(p["ajustar_s"])
    return p


def cuadre_de_corrida(c):
    """Tres comprobaciones, cada una con su margen (el mayor de 1 s y el 5 %). Devuelve {comprobaciones, cuadra}.
    1. `otros_s` del ajuste cabe (la suma de partes llega a `ajustar`).
    2. `ajustar` medido por el guion = `recursos.wall_seconds` que mide el propio motor (se salta si no hubo candidato).
    3. las fases de primer nivel (cargar + ejecutar_intento) suman la pared medida FUERA de ellas."""
    fases, out = c.get("fases") or [], []
    p = partes_del_ajuste(c)
    if p is not None:
        out.append({"nombre": "las partes del ajuste suman el ajuste", "diferencia_s": p["otros_s"],
                    "margen_s": round(_margen(p["ajustar_s"]), 3), "cuadra": p["otros_caben"]})
        rec = (c.get("recursos") or {}).get("wall_seconds")
        if rec is not None:
            d = round(p["ajustar_s"] - rec, 4)
            out.append({"nombre": "ajustar medido = recursos.wall_seconds del motor", "diferencia_s": d,
                        "margen_s": round(_margen(rec), 3), "cuadra": abs(d) <= _margen(rec)})
    if c.get("pared_s") is not None:
        primer_nivel = round(sum(f.get("wall_s") or 0.0 for f in fases if f["padre"] is None), 4)
        d = round(c["pared_s"] - primer_nivel, 4)
        out.append({"nombre": "las fases de primer nivel suman la pared de la corrida", "diferencia_s": d,
                    "margen_s": round(_margen(c["pared_s"]), 3), "cuadra": abs(d) <= _margen(c["pared_s"])})
    return {"comprobaciones": out, "cuadra": bool(out) and all(x["cuadra"] for x in out)}


def faltan_fases(c):
    """Las fases que tenían que haberse llamado y no están: la prueba de que los temporizadores ENVOLVIERON algo."""
    llamadas = {f["nombre"] for f in c.get("fases") or []}
    pedidas = FASES_SIEMPRE + (FASES_CON_CANDIDATO if c.get("estado") in ("completed", "completed_budget_limited") else ())
    return [n for n in pedidas if n not in llamadas]


def resumir_corrida(c):
    """Todo lo derivado de una corrida del guion: partes del ajuste, épocas, cuadre, picos por ruta y lo que falta."""
    fases, epocas = c.get("fases") or [], c.get("epocas") or []
    por_ruta = {}
    for f in fases:
        q = por_ruta.setdefault(f["ruta"], {"llamadas": 0, "wall_s": 0.0, "rss_ini_mb": f.get("rss_ini_mb"), "pico_mb": None})
        q["llamadas"] += 1
        q["wall_s"] = round(q["wall_s"] + (f.get("wall_s") or 0.0), 4)
        q["rss_fin_mb"] = f.get("rss_fin_mb")
        if f.get("pico_mb") is not None:
            q["pico_mb"] = f["pico_mb"] if q["pico_mb"] is None else max(q["pico_mb"], f["pico_mb"])
    picos = [(f["pico_mb"], f["ruta"]) for f in fases if f.get("pico_mb") is not None]
    mejor = next((e["n"] for e in reversed(epocas) if e.get("mejoro")), None)
    finitas = [e["perdida_val"] for e in epocas if isinstance(e.get("perdida_val"), (int, float)) and math.isfinite(e["perdida_val"])]
    hwms = [e["hwm_entrenar_mb"] for e in epocas if e.get("hwm_entrenar_mb") is not None]
    return {"rotulo": c.get("rotulo"), "plazo_s": c.get("plazo_s"), "estado": c.get("estado"), "pared_s": c.get("pared_s"),
            "partes_del_ajuste": partes_del_ajuste(c),
            "epocas": {"n": len(epocas), "mejor_epoca": mejor,
                       "entrenar_mediana_s": _mediana([e.get("entrenar_s") for e in epocas]),
                       "validar_mediana_s": _mediana([e.get("validar_s") for e in epocas]),
                       "perdida_val_primera": finitas[0] if finitas else None,
                       "perdida_val_mejor": min(finitas) if finitas else None,
                       "hwm_entrenar_max_mb": max(hwms) if hwms else None},
            "cuadre": cuadre_de_corrida(c), "faltan_fases": faltan_fases(c),
            "pico_de_la_corrida_mb": max(picos)[0] if picos else None, "fase_del_pico": max(picos)[1] if picos else None,
            "por_ruta": por_ruta}


def linea_de_fases(res):
    """Una línea por corrida: dónde se fue el ajuste y si la suma cuadra."""
    p, e = res.get("partes_del_ajuste"), res["epocas"]
    if p is None:
        return f"   {res['rotulo']} ({res['plazo_s']} s): {res['estado']} sin `ajustar` medido"
    return (f"   {res['rotulo']} ({res['plazo_s']} s): {res['estado']} | ajustar {p['ajustar_s']} s = preparar {p['preparar_s']} "
            f"(tipificar {p['tipificar_s']}) + tensores {p['a_tensor_s']} + red {p['construir_red_s']} + a_dispositivo "
            f"{p['a_dispositivo_s']} + optimizador {p['optimizador_s']} + entrenar {p['entrenar_s']} + validar {p['validar_s']} + "
            f"checkpoint {p['checkpoint_s']} + exportar {p['exportar_pesos_s']} + otros {p['otros_s']} | {e['n']} épocas "
            f"(mejor {e['mejor_epoca']}; entrenar/época {e['entrenar_mediana_s']} s, validar {e['validar_mediana_s']} s) | pico "
            f"{res['pico_de_la_corrida_mb']} MB ({res['fase_del_pico']}) | cuadre {'SÍ' if res['cuadre']['cuadra'] else 'NO'}"
            + (f" | FALTAN {res['faltan_fases']}" if res["faltan_fases"] else ""))


def linea_de_gpu(rec):
    """128-R4 en GPU: una línea con lo que DECLARÓ el motor (no lo que se pidió) y la VRAM de cada corrida."""
    d = rec.get("dispositivo_de_tabm") or {}
    usada = {True: "la GPU SE USO", False: "la GPU NO SE USO (alguna corrida declara otro dispositivo)",
             None: "sin dispositivo declarado: no se sabe"}[d.get("gpu_usada")]
    partes = []
    for rotulo, x in (d.get("por_corrida") or {}).items():
        v = x.get("vram") or {}
        partes.append(f"{rotulo}: {x.get('dispositivo')} | VRAM asignada {v.get('max_asignada_mb')} MB, reservada "
                      f"{v.get('max_reservada_mb')} MB de {v.get('total_mb')} MB ({v.get('gpu')})")
    return f"   {usada} | " + " | ".join(partes)


def veredicto_tabm_por_fases(rc, med):
    """`completed`: las dos corridas se hicieron, miden lo que dicen (no falta ninguna fase) y cuadran. `sin_tabm`: el guion no
    encontró intento de TabM (sale con 3 sin medir nada). `no_cuadra`: hay las dos corridas pero una no cuadra o le falta una
    fase. `failed`: lo demás (también morir por memoria)."""
    cs = (med or {}).get("corridas") or []
    if rc == 3 and not cs and "error" in (med or {}):
        return "sin_tabm"
    if rc == 0 and len(cs) == 2:
        if any(faltan_fases(c) for c in cs) or not all(cuadre_de_corrida(c)["cuadra"] for c in cs):
            return "no_cuadra"
        return "completed"
    return "failed"


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


def arrancar_contenedor(memoria, imagen=None, sin_servidor=False, gpu=False):
    """Un contenedor con NOMBRE PROPIO, techo de memoria sin swap y `--init`. Lo de un intento anterior con el mismo nombre
    se borra antes (un contenedor sobrevive al `timeout` de su guion). Devuelve (base_url, nombre).

    `sin_servidor` (128-C3b): la imagen arranca `sleep infinity` en vez del servidor HTTP (su `CMD`; no tiene
    `ENTRYPOINT`), para un guion que hace él mismo de servidor: así el cgroup lleva UN servidor, como en producción, y no
    uno ocioso al lado que en un caso al borde del techo fabricaría un OOM. Sin puerto ni HTTP: `base_url` es `None`.

    `gpu` (128-R4 en GPU): añade `--gpus all` a la orden de `docker run`. Sin ella, la orden es la de siempre."""
    global _contenedor
    subprocess.run(["docker", "rm", "-f", NOMBRE_CONTENEDOR], capture_output=True)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); puerto = s.getsockname()[1]; s.close()
    _contenedor = NOMBRE_CONTENEDOR
    atexit.register(borrar_contenedor)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: (borrar_contenedor(), os._exit(143)))
    r = subprocess.run(["docker", "run", "-d", "--init", "--name", NOMBRE_CONTENEDOR,
                        *([] if sin_servidor else ["-p", f"127.0.0.1:{puerto}:8765"]),
                        f"--memory={memoria}", f"--memory-swap={memoria}", "--cpus=2",
                        *(["--gpus", "all"] if gpu else []),
                        "-e", "MATRIXAI_LICENSE_ENABLED=false", imagen or IMAGEN,
                        *(["sleep", "infinity"] if sin_servidor else [])],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("docker run falló: " + r.stderr)
    if sin_servidor:
        t0 = time.time()
        while time.time() - t0 < 60:
            if subprocess.run(["docker", "exec", NOMBRE_CONTENEDOR, "true"], capture_output=True).returncode == 0:
                return None, NOMBRE_CONTENEDOR
            time.sleep(1)
        raise SystemExit("el contenedor sin servidor no aceptó un exec en 60 s")
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


def cerrar_registro(rec, muestras, eventos, dir_series, nombre, c, memoria, rol=rol_de):
    lb = None
    # los motores de cada intento, inferidos del orden de las entradas de la búsqueda
    if rec.get("intentos"):
        lb = [{"engine": x["motor"], "fase": x.get("fase")} for x in rec["intentos"]]
    mpi = motores_por_intento_de(lb)
    rec["motor_de_cada_intento_inferido"] = {str(k + 1): v for k, v in mpi.items()}
    if (rec.get("medida") or {}).get("motores"):       # el guion del intento sin tope dice su motor, no se infiere
        mpi = {i: m["motor"] for i, m in enumerate(rec["medida"]["motores"])}
    rec["resumen_de_memoria"] = resumir(muestras, eventos, mpi, rol=rol)
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


def medir_retencion(nombre, c, gen, memoria, dir_series):
    """128-C1 paso 0: un estudio ENTERO dentro del contenedor (el guion `RETENCION`) y, al terminar, ¿cuánto devuelven `gc` y
    `malloc_trim(0)`? Mismo muestreador de procesos que el resto. No cambia el producto."""
    comprobar_que_se_puede()
    csv_bytes, cols, excl, nfilas = r0.csv_de(gen, c["data_id"])
    cuerpo = {"csv_text": csv_bytes.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    ruta = AGENTE / f"_{os.getpid()}_cuerpo_ret.json"
    ruta.write_text(json.dumps(cuerpo))
    base, nombre_c = arrancar_contenedor(memoria)
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "inicio": ahora(),
           "csv": {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes)},
           "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
           "sha256_guion_dentro": hashlib.sha256(RETENCION.encode()).hexdigest()}
    try:
        r = subprocess.run(["docker", "cp", str(ruta), f"{nombre_c}:/tmp/cuerpo_128_c1.json"], capture_output=True, text=True)
        ruta.unlink(missing_ok=True)
        if r.returncode:
            raise SystemExit("docker cp falló: " + r.stderr)
        poner_en_el_contenedor(nombre_c, RETENCION, "/tmp/retencion_128_c1.py")
        with Muestreador(nombre_c) as mu:
            eventos = [(time.time(), "arranque_del_guion")]
            t0 = time.time()
            p = subprocess.run(["docker", "exec", nombre_c, "python3", "/tmp/retencion_128_c1.py",
                                "/tmp/cuerpo_128_c1.json", "/tmp/retencion_128_c1.json"], capture_output=True, text=True)
            rec["guion_rc"], rec["guion_pared_s"] = p.returncode, round(time.time() - t0, 1)
            rec["guion_stderr_cola"] = p.stderr[-1500:]
            time.sleep(2)
            muestras = mu.leer()
        out = subprocess.run(["docker", "exec", nombre_c, "cat", "/tmp/retencion_128_c1.json"], capture_output=True, text=True)
        try:
            med = json.loads(out.stdout)
        except ValueError:
            med = {"error": "el guion no escribió su salida", "stdout": out.stdout[-500:]}
        rec["retencion"] = med
        eventos.append((time.time(), "fin"))
        rec["contenedor"] = r0.estado_del_contenedor(nombre_c)
        rec["estado"] = "completed" if p.returncode == 0 and med.get("resumen") else "failed"
        rec["fin"] = ahora()
        return cerrar_registro(rec, muestras, eventos, dir_series, nombre + "-retencion", c, memoria)
    finally:
        ruta.unlink(missing_ok=True)
        borrar_contenedor()


#: 128-C3b: el paquete CPU tal como se publica (la densa anterior fuera), las MISMAS variables que `--pcpu` del 120: así el
#: estudio con TabM forzado se compara con el de la referencia del producto, que en los grandes corre sin TabM por la guarda.
ENTORNO_TABM_FORZADO = r0.entorno_extra_de("pcpu")


def _memory_events(nombre_c):
    """`memory.events` del cgroup del contenedor (v2): cuántas veces llegó al techo (`max`) y cuántos procesos mató el OOM
    (`oom_kill`). Un hijo muerto por memoria NO marca `OOMKilled` en el contenedor (eso es solo su proceso 1)."""
    r = subprocess.run(["docker", "exec", nombre_c, "cat", "/sys/fs/cgroup/memory.events"], capture_output=True, text=True)
    if r.returncode:
        return {"lectura": "falló", "stderr": r.stderr.strip()[:300]}
    out = {}
    for linea in r.stdout.splitlines():
        k, _, v = linea.partition(" ")
        if v.strip().isdigit():
            out[k] = int(v)
    return out


def medir_tabm_forzado(nombre, c, gen, memoria, dir_series, tope_s):
    """128-C3b: un estudio ENTERO dentro del contenedor (el guion `TABM_FORZADO`) con la guarda de memoria de TabM callada EN
    MEMORIA, con el entorno del producto CPU (`ENTORNO_TABM_FORZADO`), en un contenedor SIN el servidor HTTP (el guion hace de
    servidor: un solo servidor en el cgroup, como en producción). Mismo muestreador que el resto; al acabar, lo que dice el
    cgroup de sus OOM. No cambia el producto: ni su disco ni su imagen."""
    comprobar_que_se_puede()
    csv_bytes, cols, excl, nfilas = r0.csv_de(gen, c["data_id"])
    cuerpo = {"csv_text": csv_bytes.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    ruta = AGENTE / f"_{os.getpid()}_cuerpo_tabm.json"
    ruta.write_text(json.dumps(cuerpo))
    _, nombre_c = arrancar_contenedor(memoria, sin_servidor=True)
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "objetivo": c["objetivo"], "inicio": ahora(),
           "csv": {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes), "excluidas_por_string": excl},
           "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
           "sha256_guion_dentro": hashlib.sha256(TABM_FORZADO.encode()).hexdigest(),
           "entorno_del_guion": dict(ENTORNO_TABM_FORZADO), "tope_s": tope_s,
           "contenedor_sin_servidor": True}
    try:
        r = subprocess.run(["docker", "cp", str(ruta), f"{nombre_c}:/tmp/cuerpo_128_c3b.json"], capture_output=True, text=True)
        ruta.unlink(missing_ok=True)
        if r.returncode:
            raise SystemExit("docker cp falló: " + r.stderr)
        poner_en_el_contenedor(nombre_c, TABM_FORZADO, "/tmp/tabm_forzado_128_c3b.py")
        orden = ["docker", "exec", *[x for k, v in sorted(ENTORNO_TABM_FORZADO.items()) for x in ("-e", f"{k}={v}")],
                 nombre_c, "python3", "/tmp/tabm_forzado_128_c3b.py", "/tmp/cuerpo_128_c3b.json", "/tmp/tabm_forzado_128_c3b.json"]
        with Muestreador(nombre_c) as mu:
            t0 = time.time()
            try:
                p = subprocess.run(orden, capture_output=True, text=True, timeout=tope_s)
                rc, err = p.returncode, p.stderr or ""
            except subprocess.TimeoutExpired as e:
                rc, err = "plazo", e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            rec["guion_rc"], rec["guion_pared_s"] = rc, round(time.time() - t0, 1)
            rec["guion_stderr_cola"] = err[-1500:]
            time.sleep(2)
            muestras = mu.leer()
        out = subprocess.run(["docker", "exec", nombre_c, "cat", "/tmp/tabm_forzado_128_c3b.json"], capture_output=True, text=True)
        try:
            med = json.loads(out.stdout)
        except ValueError:
            med = {"error": "el guion no escribió su salida", "stdout": out.stdout[-500:]}
        rec["memory_events"] = _memory_events(nombre_c)
        rec["contenedor"] = r0.estado_del_contenedor(nombre_c)
        est = med.get("estado_final") or {}
        _, filas = r0.intentos_de(est)
        sel = est.get("seleccion") or {}
        rec.update(estado="plazo" if rc == "plazo" else veredicto_tabm_forzado(rc, med), estado_del_estudio=est.get("estado"),
                   error=est.get("error") or med.get("error"), campeon=sel.get("candidate_engine"),
                   media_de_la_seleccion=sel.get("media_de_la_seleccion"), n_intentos_total=est.get("n_intentos_total"),
                   # la cifra del TEST y lo demás con las MISMAS funciones del 120: el registro se compara con
                   # `comparar_c4_128.py` contra el del producto (`c4_128_pcpu.json`), donde TabM no compitió
                   test=r0.test_de(sel), perfiles_declarados=r0.perfiles_declarados(est),
                   politica_de_arboles=sel.get("politica_de_arboles"),
                   intentos=filas, intentos_de_tabm=med.get("intentos_de_tabm"),
                   guarda_de_verdad=med.get("guarda_de_verdad"), motores_fuera_por_memoria=est.get("motores_fuera_por_memoria"),
                   presupuesto_segun_el_tamano=est.get("presupuesto_segun_el_tamano"),
                   pared_estudio_s=(round(med["t_fin"] - med["t_inicio"], 1) if med.get("t_fin") and med.get("t_inicio")
                                    else None), fin=ahora())
        if "respuesta" in med:
            rec["respuesta"] = med["respuesta"]
        eventos = eventos_de_los_estados(med) or [(t0, "arranque_del_guion")]
        rec = cerrar_registro(rec, muestras, eventos, dir_series, nombre + "-tabm-forzado", c, memoria,
                              rol=rol_con_el_guion_de_servidor)
        # el pico del cgroup con el nombre que usa el instrumento del 120 (lo que mira el OOM)
        rec["memoria_pico_mb"] = ((rec.get("resumen_de_memoria") or {}).get("pico_cgroup") or {}).get("cgroup_mb")
        if est:                                           # el estado ENTERO, para releerlo sin repetir la medida
            f = dir_series / f"{nombre}-tabm-forzado.estado.json.gz"
            with gzip.open(f, "wt") as fh:
                json.dump(est, fh, default=str)
            rec["estado_final_guardado"] = {"fichero": f.name, "sha256": sha(f)}
        return rec
    finally:
        ruta.unlink(missing_ok=True)
        borrar_contenedor()


def entorno_de_fases(gpu=False):
    """128-R4: el entorno del guion `TABM_POR_FASES`. Sin `gpu` es EXACTAMENTE `ENTORNO_TABM_FORZADO` (el paquete CPU); con `gpu`
    añade `MATRIXAI_TABM_DISPOSITIVO=auto`, que es lo que lleva el brazo B del kit 128-C5 y lo que hace el paquete GPU."""
    e = dict(ENTORNO_TABM_FORZADO)
    if gpu:
        e[VARIABLE_DEL_DISPOSITIVO] = "auto"
    return e


def banco_del_kit(datos):
    """128-R4 en GPU: los conjuntos del `manifiesto.json` del kit 128-C5 (en el PC de Roberto no están los ARFF ni el banco del
    120). Cada uno lleva lo mismo que el banco: data_id, tarea, objetivo y clase_positiva."""
    m = json.loads((Path(datos) / "manifiesto.json").read_text())
    return {c["nombre"]: dict(c) for c in m["conjuntos"]}


def csv_del_kit(datos, c):
    """El CSV del manifiesto con la misma forma que `r0.csv_de`: (bytes, columnas, excluidas, filas). Comprueba el sha256 del .gz
    Y el del CSV descomprimido contra el manifiesto: si no cuadra, PARA (medir TabM con otros datos no mide lo que dice)."""
    ruta = Path(datos) / c["fichero"]
    if not ruta.is_file():
        raise SystemExit(f"PARO: {c['nombre']}: falta {ruta}")
    crudo = ruta.read_bytes()
    if hashlib.sha256(crudo).hexdigest() != c["sha256_gz"]:
        raise SystemExit(f"PARO: {c['nombre']}: el .gz no es el del manifiesto (sha256)")
    try:
        contenido = gzip.decompress(crudo)
    except OSError as e:
        raise SystemExit(f"PARO: {c['nombre']}: no se descomprime ({e})")
    if hashlib.sha256(contenido).hexdigest() != c["sha256"] or len(contenido) != c["bytes"]:
        raise SystemExit(f"PARO: {c['nombre']}: el CSV no es el del manifiesto (sha256)")
    columnas = next(csv.reader(io.StringIO(contenido.decode("utf-8"), newline="")))
    if len(columnas) != c["columnas"] or c["objetivo"] not in columnas:
        raise SystemExit(f"PARO: {c['nombre']}: columnas {len(columnas)} != manifiesto {c['columnas']} o falta el objetivo")
    return contenido, columnas, list(c.get("excluidas_por_string") or []), c["filas"]


def _cuerpo_del_estudio(c, csv_bytes, cols):
    cuerpo = {"csv_text": csv_bytes.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    return cuerpo


def dispositivo_de_las_corridas(med):
    """128-R4 en GPU: en qué dispositivo dice el MOTOR que entrenó TabM en cada corrida (`dispositivo_declarado`, que el guion lee
    del predictor; la clave `dispositivo` de `entrenamiento_efectivo`, o `device` de `hiperparametros`). `gpu_usada`: True si todas
    las corridas con candidato declaran `cuda`; False si alguna declara otra cosa (también si CAYÓ a la CPU); None si ninguna
    declara nada (no se infiere de la VRAM: un valor ausente no es una respuesta)."""
    por_corrida, vistos = {}, []
    for x in (med or {}).get("corridas") or []:
        d = x.get("dispositivo_declarado") or {}
        real = d.get("dispositivo") or d.get("device")
        por_corrida[x.get("rotulo")] = {"dispositivo": real, "pedido": d.get("dispositivo_pedido"),
                                        "caida_a_cpu": d.get("caida_a_cpu"), "declaracion": d,
                                        "vram": x.get("vram")}
        if real is not None:
            vistos.append(real)
    return {"entorno_pedido": (med or {}).get("dispositivo_pedido_por_entorno"), "por_corrida": por_corrida,
            "gpu_usada": (all(v == "cuda" for v in vistos) if vistos else None)}


def _componer_registro_de_fases(rec, med, rc, muestras, dir_series, nombre, c, memoria, t0):
    """Lo que sigue a leer la salida del guion, igual en el contenedor lanzado por la sonda y en el modo del kit."""
    rec["medida"] = med
    rec["resumen_de_corridas"] = [resumir_corrida(x) for x in med.get("corridas") or []]
    rec["estado"] = "plazo" if rc == "plazo" else veredicto_tabm_por_fases(rc, med)
    rec["dispositivo_de_tabm"] = dispositivo_de_las_corridas(med)
    rec["error"] = med.get("error")
    rec["fin"] = ahora()
    eventos = [(t0, "arranque_del_guion")]
    for x in med.get("corridas") or []:                    # el instante de cada corrida, para repartir la memoria del contenedor
        if x.get("fases"):
            eventos.append((x["fases"][0]["t_pared"], x["rotulo"]))
    rec = cerrar_registro(rec, muestras, eventos, dir_series, nombre, c, memoria, rol=rol_con_el_guion_por_fases)
    rec["memoria_pico_mb"] = ((rec.get("resumen_de_memoria") or {}).get("pico_cgroup") or {}).get("cgroup_mb")
    return rec


def medir_tabm_por_fases(nombre, c, gen, memoria, dir_series, tope_s, plazo_amplio, datos_kit=None, gpu=False):
    """128-R4: un intento de TabM del primer pliegue, directo, por FASES y con dos plazos (el del producto y `plazo_amplio`),
    dentro de un contenedor SIN servidor HTTP (el guion hace de servidor, como en el C3b) y con el entorno del paquete CPU.
    Mismo muestreador de procesos que el resto. No cambia el producto: ni su disco ni su imagen; los hijos `spawn` no llevan
    temporizadores (solo el intento directo).

    `datos_kit`: la carpeta `datos/` del kit 128-C5 (los CSV salen de ahí, con su sha256 comprobado) en vez de los ARFF de este
    servidor. `gpu`: el contenedor lleva `--gpus all` y el guion `MATRIXAI_TABM_DISPOSITIVO=auto`; el registro dice el
    dispositivo que DECLARÓ el motor (`dispositivo_de_tabm`) y la VRAM de cada corrida."""
    comprobar_que_se_puede()
    csv_bytes, cols, excl, nfilas = csv_del_kit(datos_kit, c) if datos_kit else r0.csv_de(gen, c["data_id"])
    cuerpo = _cuerpo_del_estudio(c, csv_bytes, cols)
    AGENTE.mkdir(parents=True, exist_ok=True)
    ruta = AGENTE / f"_{os.getpid()}_cuerpo_fases.json"
    ruta.write_text(json.dumps(cuerpo))
    entorno = entorno_de_fases(gpu)
    _, nombre_c = arrancar_contenedor(memoria, sin_servidor=True, gpu=gpu)
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "objetivo": c["objetivo"], "inicio": ahora(),
           "csv": {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes), "excluidas_por_string": excl,
                   "origen": "kit" if datos_kit else "arff"},
           "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
           "sha256_guion_dentro": hashlib.sha256(TABM_POR_FASES.encode()).hexdigest(),
           "entorno_del_guion": entorno, "tope_s": tope_s, "plazo_amplio_s": plazo_amplio, "gpu": bool(gpu),
           "contenedor_sin_servidor": True}
    try:
        r = subprocess.run(["docker", "cp", str(ruta), f"{nombre_c}:/tmp/cuerpo_128_r4.json"], capture_output=True, text=True)
        ruta.unlink(missing_ok=True)
        if r.returncode:
            raise SystemExit("docker cp falló: " + r.stderr)
        poner_en_el_contenedor(nombre_c, TABM_POR_FASES, "/tmp/tabm_por_fases_128_r4.py")
        orden = ["docker", "exec", *[x for k, v in sorted(entorno.items()) for x in ("-e", f"{k}={v}")],
                 nombre_c, "python3", "/tmp/tabm_por_fases_128_r4.py", "/tmp/cuerpo_128_r4.json", "/tmp/tabm_por_fases_128_r4.json",
                 str(plazo_amplio)]
        with Muestreador(nombre_c) as mu:
            t0 = time.time()
            try:
                p = subprocess.run(orden, capture_output=True, text=True, timeout=tope_s)
                rc, err = p.returncode, p.stderr or ""
            except subprocess.TimeoutExpired as e:
                rc, err = "plazo", e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            rec["guion_rc"], rec["guion_pared_s"] = rc, round(time.time() - t0, 1)
            rec["guion_stderr_cola"] = err[-1500:]
            time.sleep(2)
            muestras = mu.leer()
        out = subprocess.run(["docker", "exec", nombre_c, "cat", "/tmp/tabm_por_fases_128_r4.json"], capture_output=True, text=True)
        try:
            med = json.loads(out.stdout)
        except ValueError:
            med = {"error": "el guion no escribió su salida", "stdout": out.stdout[-500:]}
        rec["memory_events"] = _memory_events(nombre_c)
        rec["contenedor"] = r0.estado_del_contenedor(nombre_c)
        return _componer_registro_de_fases(rec, med, rc, muestras, dir_series, nombre, c, memoria, t0)
    finally:
        ruta.unlink(missing_ok=True)
        borrar_contenedor()


# --- 128-R4 en GPU, modo del KIT: la sonda corre DENTRO del contenedor de la imagen GPU (lo lanza `lanzar.*`)
class MuestreadorLocal:
    """El muestreador de `MUESTREADOR` como proceso HERMANO de este (no hay otro contenedor al que hacer `docker exec`)."""

    def __init__(self, carpeta):
        self.carpeta = Path(carpeta)
        self.muestras, self.parar, self.proc = self.carpeta / "muestras.jsonl", self.carpeta / "parar", None

    def __enter__(self):
        self.carpeta.mkdir(parents=True, exist_ok=True)
        guion = self.carpeta / "muestreador_128_c0.py"
        guion.write_text(MUESTREADOR)
        for f in (self.muestras, self.parar):
            f.unlink(missing_ok=True)
        self.proc = subprocess.Popen([sys.executable, str(guion), str(self.muestras), str(self.parar)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 30:
            if self.leer():
                return self
            time.sleep(0.5)
        raise SystemExit("el muestreador no escribió ninguna muestra en 30 s")

    def leer(self):
        out = []
        try:
            lineas = self.muestras.read_text().splitlines()
        except OSError:
            return out
        for linea in lineas:
            try:
                out.append(json.loads(linea))
            except ValueError:
                pass
        return out

    def __exit__(self, *_):
        self.parar.write_text("")
        time.sleep(1.5)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _memory_events_aqui(ruta="/sys/fs/cgroup/memory.events"):
    """`memory.events` del cgroup de ESTE contenedor (v2), como `_memory_events` pero sin `docker exec`."""
    try:
        texto = Path(ruta).read_text()
    except OSError as e:
        return {"lectura": "falló", "stderr": str(e)[:300]}
    return {k: int(v) for k, _, v in (l.partition(" ") for l in texto.splitlines()) if v.strip().isdigit()}


def medir_tabm_por_fases_aqui(nombre, c, memoria, dir_series, tope_s, plazo_amplio, datos_kit, gpu=True):
    """128-R4 en GPU, modo del KIT: lo mismo que `medir_tabm_por_fases`, pero ESTE proceso ya está dentro del contenedor (el de la
    imagen GPU, con `--gpus all` y el techo de memoria que le puso el lanzador): el guion corre como subproceso, sin `docker`.
    En el PC de Roberto no hay Python ni se puede hacer `docker exec` desde dentro. Los CSV salen del kit (sha256 comprobado) y los
    temporales van a `AGENTE`. Lo que no hay aquí: `docker inspect` (OOMKilled) — el OOM se ve en `memory_events`."""
    csv_bytes, cols, excl, nfilas = csv_del_kit(datos_kit, c)
    cuerpo = _cuerpo_del_estudio(c, csv_bytes, cols)
    AGENTE.mkdir(parents=True, exist_ok=True)
    entorno = entorno_de_fases(gpu)
    ruta_cuerpo, ruta_guion = AGENTE / f"{os.getpid()}_cuerpo_128_r4.json", AGENTE / f"{os.getpid()}_tabm_por_fases_128_r4.py"
    ruta_salida = AGENTE / f"{os.getpid()}_tabm_por_fases_128_r4.json"
    ruta_cuerpo.write_text(json.dumps(cuerpo))
    ruta_guion.write_text(TABM_POR_FASES)
    ruta_salida.unlink(missing_ok=True)
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "objetivo": c["objetivo"], "inicio": ahora(),
           "csv": {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes), "excluidas_por_string": excl, "origen": "kit"},
           "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
           "sha256_guion_dentro": hashlib.sha256(TABM_POR_FASES.encode()).hexdigest(),
           "entorno_del_guion": entorno, "tope_s": tope_s, "plazo_amplio_s": plazo_amplio, "gpu": bool(gpu),
           "contenedor_sin_servidor": True, "dentro_del_contenedor": True}
    try:
        with MuestreadorLocal(AGENTE / f"{os.getpid()}_muestreo") as mu:
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, str(ruta_guion), str(ruta_cuerpo), str(ruta_salida), str(plazo_amplio)],
                                   capture_output=True, text=True, timeout=tope_s, env={**os.environ, **entorno})
                rc, err = p.returncode, p.stderr or ""
            except subprocess.TimeoutExpired as e:
                rc, err = "plazo", e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            rec["guion_rc"], rec["guion_pared_s"] = rc, round(time.time() - t0, 1)
            rec["guion_stderr_cola"] = err[-1500:]
            time.sleep(2)
            muestras = mu.leer()
        try:
            med = json.loads(ruta_salida.read_text())
        except (OSError, ValueError):
            med = {"error": "el guion no escribió su salida"}
        rec["memory_events"] = _memory_events_aqui()
        rec["contenedor"] = {"dentro_del_contenedor": True}
        return _componer_registro_de_fases(rec, med, rc, muestras, dir_series, nombre, c, memoria, t0)
    finally:
        for f in (ruta_cuerpo, ruta_guion, ruta_salida):
            f.unlink(missing_ok=True)


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
    global IMAGEN, SALIDA, AGENTE
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="", help="conjuntos separados por comas (de los del banco del 120)")
    ap.add_argument("--memoria", default=MEMORIA)
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--imagen", default=IMAGEN)
    ap.add_argument("--humo", action="store_true", help="dresses-sales (y su intento sin tope): minutos")
    ap.add_argument("--control", action="store_true", help="una reserva CONOCIDA (300 y 600 MB) y la sonda tiene que verla")
    ap.add_argument("--retencion", action="store_true",
                    help="128-C1 paso 0: un estudio entero en el contenedor y, al acabar, cuánto devuelven gc y malloc_trim(0) "
                         "(solo con --humo en el desarrollo; la de APSFailure va a la cola)")
    ap.add_argument("--tabm-forzado", action="store_true",
                    help="128-C3b: un estudio entero en el contenedor con la guarda de memoria de TabM callada EN MEMORIA "
                         "(el entorno de --pcpu): ¿cabe TabM en los grandes? (la de los tres va a la cola)")
    ap.add_argument("--tabm-por-fases", action="store_true",
                    help="128-R4: un intento de TabM del primer pliegue por FASES (carga, preparación, red, épocas, validación, "
                         "checkpoint, exportación) con el plazo del producto y con uno amplio (la de los tres va a la cola)")
    ap.add_argument("--plazo-amplio", type=float, default=PLAZO_AMPLIO_S,
                    help="el plazo (s) de la segunda corrida de --tabm-por-fases")
    ap.add_argument("--tope-estudio", type=int, default=TOPE_ESTUDIO_S)
    ap.add_argument("--datos-del-kit", default="",
                    help="128-R4 en GPU: la carpeta `datos/` del kit 128-C5 (manifiesto.json + CSV .gz): los conjuntos y sus CSV salen "
                         "de ahí (sha256 comprobado) en vez del banco del 120 y los ARFF de este servidor. Solo con --tabm-por-fases")
    ap.add_argument("--temporales", default="",
                    help="la carpeta de los ficheros temporales (el cuerpo del estudio con los datos); por omisión ~/.agentes/128-c0")
    ap.add_argument("--gpu", action="store_true",
                    help="128-R4 en GPU: el contenedor lleva `--gpus all` y el guion MATRIXAI_TABM_DISPOSITIVO=auto; el registro dice el "
                         "dispositivo que DECLARÓ el motor y la VRAM. Sin ella, nada cambia (CPU)")
    ap.add_argument("--dentro-del-contenedor", action="store_true",
                    help="modo del KIT: esta sonda YA corre dentro del contenedor de la imagen GPU (sin `docker`): el guion corre como "
                         "subproceso. Exige --tabm-por-fases y --datos-del-kit")
    ap.add_argument("--imagen-id", default="", help="con --dentro-del-contenedor: el id de la imagen (lo sabe el lanzador, no el contenedor)")
    a = ap.parse_args()
    IMAGEN, SALIDA = a.imagen, Path(a.salida).resolve()
    if a.temporales:
        AGENTE = Path(a.temporales).resolve()
    if a.dentro_del_contenedor and not (a.tabm_por_fases and a.datos_del_kit):
        raise SystemExit("PARO: --dentro-del-contenedor exige --tabm-por-fases y --datos-del-kit")
    if a.datos_del_kit and not a.tabm_por_fases:
        raise SystemExit("PARO: --datos-del-kit solo está hecho para --tabm-por-fases")
    if a.gpu and not a.tabm_por_fases:
        raise SystemExit("PARO: --gpu solo está hecho para --tabm-por-fases")
    AGENTE.mkdir(parents=True, exist_ok=True)
    if a.dentro_del_contenedor:
        imagen_id = a.imagen_id or "desconocido (dentro del contenedor)"
    else:
        imagen_id = subprocess.run(["docker", "image", "inspect", IMAGEN, "--format", "{{.Id}}"], capture_output=True, text=True).stdout.strip()
        if not imagen_id:
            raise SystemExit(f"PARO: la imagen {IMAGEN} no está cargada")
    if a.datos_del_kit:
        gen, banco = None, banco_del_kit(a.datos_del_kit)
    else:
        gen = r0.cargar_generador()
        banco = {b["nombre"]: b for b in json.loads(r0.BORRADOR.read_text())["conjuntos"]["banco"]}
        declarados = {d["nombre"]: d for d in json.loads(r0.SONDA.read_text())["datasets_declarados"]}
        for n, b in banco.items():
            for k in ("data_id", "tarea", "objetivo", "clase_positiva"):
                if str(declarados[n][k]) != str(b[k]):
                    raise SystemExit(f"PARO: {n}.{k}: borrador {b[k]!r} != 119-C5a {declarados[n][k]!r}")
    procedencia = {"imagen": IMAGEN, "imagen_id": imagen_id, "sha256_guion": sha(__file__),
                   "sha256_referencia_120_r0": sha(r0.__file__), "sha256_generador_de_csv": None if a.datos_del_kit else sha(r0.GENERADOR),
                   "sha256_muestreador": hashlib.sha256(MUESTREADOR.encode()).hexdigest(),
                   "sha256_intento_sin_tope": hashlib.sha256(INTENTO_SIN_TOPE.encode()).hexdigest(),
                   "sha256_retencion": hashlib.sha256(RETENCION.encode()).hexdigest(),
                   "sha256_tabm_forzado": hashlib.sha256(TABM_FORZADO.encode()).hexdigest(),
                   "sha256_tabm_por_fases": hashlib.sha256(TABM_POR_FASES.encode()).hexdigest(),
                   "memoria_del_contenedor": a.memoria, "cpus": 2, "inicio": ahora(),
                   "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3],
                   "datos_del_kit": ({"carpeta": a.datos_del_kit, "sha256_manifiesto": sha(Path(a.datos_del_kit) / "manifiesto.json")}
                                     if a.datos_del_kit else None),
                   "gpu": bool(a.gpu), "dentro_del_contenedor": bool(a.dentro_del_contenedor),
                   "instrumento": "estudio por HTTP; RSS por proceso de /proc dentro del contenedor cada ~1 s; sin tracemalloc",
                   "limites": ["la preparación y el diagnóstico ocurren dentro del POST de iniciar: una sola fase «preparacion»",
                               "un proceso que viva menos de ~1 s puede no aparecer",
                               "el motor de cada intento es INFERIDO del orden del leaderboard final",
                               "la suma de RSS cuenta dos veces lo compartido; el pico del cgroup es lo que mira el OOM"]}
    if a.tabm_por_fases:
        procedencia["tabm_por_fases"] = {
            "plazo_amplio_s": a.plazo_amplio, "tope_s": a.tope_estudio, "margen_de_cuadre": f"el mayor de {MARGEN_ABS_S} s y el {MARGEN_REL:.0%}",
            "limites": ["los temporizadores se ponen EN MEMORIA en el proceso del guion; los hijos `spawn` re-importan del disco y NO los "
                        "llevan: solo el intento DIRECTO está medido por fases (el aislado del producto añade pickle/fichero y el hijo)",
                        "las dos corridas van una tras otra en el MISMO proceso: la segunda hereda el asignador y las cachés de la primera",
                        "el intento se construye con `_construir` (la preparación del producto en el padre, que los árboles comparten); "
                        "su coste NO está en el plazo de TabM (el reloj empieza en `_ajustar`)",
                        "entrenar y checkpoint por época se DERIVAN de dos marcas (validación y primer `_ahora` tras ella), no son funciones",
                        "VmHWM se reinicia con /proc/self/clear_refs al entrar en cada fase hoja; si el núcleo no lo deja, "
                        "`hwm_se_reinicia` lo dice y el pico es acumulado",
                        "solo se mide un intento (primer pliegue, repetición 0, semilla del producto): una época por validación "
                        "cruzada de otros pliegues puede variar con la carga de la máquina"]}
    datos = {"procedencia": procedencia, "control": None, "conjuntos": {}}
    if a.control:
        datos["control"] = control()
        procedencia["fin"] = ahora()
        guardar(SALIDA, datos)
        sys.exit(0 if datos["control"]["cuadra"] else 1)
    por_omision = CONJUNTOS_DE_FASES_EN_GPU if a.datos_del_kit else CONJUNTOS_DEL_C0
    nombres = list(HUMO) if a.humo else [x for x in a.solo.split(",") if x] or list(por_omision)
    for n in nombres:
        if n not in banco:
            raise SystemExit(f"PARO: {n} no está en {'el manifiesto del kit' if a.datos_del_kit else 'el banco del 120'}")
    dir_series = SALIDA.with_name(SALIDA.stem + "_series")
    if a.tabm_por_fases:
        for n in nombres:
            print(f"[{ahora()}] {n}: TabM por fases (memoria {a.memoria}, plazo amplio {a.plazo_amplio} s, tope {a.tope_estudio} s) ...",
                  flush=True)
            if a.dentro_del_contenedor:
                rec = medir_tabm_por_fases_aqui(n, banco[n], a.memoria, dir_series, a.tope_estudio, a.plazo_amplio, a.datos_del_kit,
                                                gpu=a.gpu)
            else:
                rec = medir_tabm_por_fases(n, banco[n], gen, a.memoria, dir_series, a.tope_estudio, a.plazo_amplio,
                                           datos_kit=a.datos_del_kit or None, gpu=a.gpu)
            datos["conjuntos"][n + ":tabm_por_fases"] = rec
            guardar(SALIDA, datos)
            print(f"   {rec.get('estado')} | guion {rec.get('guion_pared_s')} s | cgroup {rec.get('memoria_pico_mb')} MB | "
                  f"oom_kill {(rec.get('memory_events') or {}).get('oom_kill')}", flush=True)
            for res in rec.get("resumen_de_corridas") or []:
                print(linea_de_fases(res), flush=True)
            if a.gpu:
                print(linea_de_gpu(rec), flush=True)
            if rec.get("estado") == "sin_tabm":
                procedencia["fin"] = ahora()
                guardar(SALIDA, datos)
                raise SystemExit(f"PARO: {n} sin intento de TabM; la sustitución de la guarda no llegó")
        procedencia["fin"] = ahora()
        guardar(SALIDA, datos)
        return
    if a.tabm_forzado:
        for n in nombres:
            print(f"[{ahora()}] {n}: TabM forzado (memoria {a.memoria}, tope {a.tope_estudio} s) ...", flush=True)
            rec = medir_tabm_forzado(n, banco[n], gen, a.memoria, dir_series, a.tope_estudio)
            datos["conjuntos"][n + ":tabm_forzado"] = rec
            guardar(SALIDA, datos)
            print(linea_de_resumen(rec), f"| TabM: {rec.get('intentos_de_tabm')} intentos | campeón {rec.get('campeon')} | "
                  f"oom_kill {(rec.get('memory_events') or {}).get('oom_kill')}", flush=True)
            if rec.get("estado") == "sin_tabm":         # el instrumento no mide lo que dice: seguir sería gastar la noche
                procedencia["fin"] = ahora()
                guardar(SALIDA, datos)
                raise SystemExit(f"PARO: {n} sin intentos de TabM; la sustitución de la guarda no llegó")
        procedencia["fin"] = ahora()
        guardar(SALIDA, datos)
        return
    if a.retencion:
        for n in nombres:
            print(f"[{ahora()}] {n}: retención (memoria {a.memoria}) ...", flush=True)
            rec = medir_retencion(n, banco[n], gen, a.memoria, dir_series)
            datos["conjuntos"][n + ":retencion"] = rec
            guardar(SALIDA, datos)
            res = (rec.get("retencion") or {}).get("resumen") or {}
            print(f"   {rec.get('estado')}: {res}", flush=True)
        procedencia["fin"] = ahora()
        guardar(SALIDA, datos)
        return
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
