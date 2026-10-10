"""128-C0: la parte de la sonda que no necesita Docker -- roles, fases, resumen por fase y proceso, el control de la reserva
conocida, y el muestreador leyendo `/proc` de ESTE proceso. La parte con Docker se prueba con `--control` y `--humo`."""
import importlib.util
import os
import sys
from pathlib import Path

RUTA = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "sonda_128_c0.py"


def _cargar():
    spec = importlib.util.spec_from_file_location("sonda_128_c0_bajo_prueba", RUTA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _cargar()
SRV = "python -m matrixai_studio --host 0.0.0.0 --port 8765"
HIJO = "/usr/local/bin/python3 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=9)"


def proc(pid, rss, cmd, ppid=1, hwm=None):
    return {"pid": pid, "ppid": ppid, "rss_mb": rss, "hwm_mb": hwm if hwm is not None else rss, "cmd": cmd}


def serie():
    """t=0..9. Preparación (0-2): servidor sube a 1500. Intento 1 (3-5): hijo 100 -> 900. Intento 2 (6-7): otro hijo 400.
    Reajuste (8-9): hijo 2.000 y servidor 1.600."""
    sh = proc(10, 2, "sh -c python -m matrixai_studio --host x")
    ms = []
    srv = [800, 1200, 1500, 1500, 1500, 1500, 1500, 1500, 1600, 1600]
    h1 = {3: 100, 4: 600, 5: 900}
    h2 = {6: 200, 7: 400}
    h3 = {8: 1200, 9: 2000}
    for t in range(10):
        ps = [sh, proc(11, srv[t], SRV, ppid=10)]
        if t in h1: ps.append(proc(21, h1[t], HIJO, ppid=11))
        if t in h2: ps.append(proc(22, h2[t], HIJO, ppid=11))
        if t in h3: ps.append(proc(23, h3[t], HIJO, ppid=11))
        ps.append(proc(99, 30, "python3 /tmp/muestreador_128_c0.py x y"))
        ms.append({"t": float(t), "procesos": ps, "cgroup_mb": sum(p["rss_mb"] for p in ps) + 50.0})
    return ms


EVENTOS = [(0.0, "preparacion"), (3.0, "intento 1/2"), (6.0, "intento 2/2"), (8.0, "reajuste_final"), (10.0, "fin")]


def test_roles():
    assert S.rol_de(SRV) == "servidor"
    assert S.rol_de("sh -c python -m matrixai_studio --host x") == "sh"
    assert S.rol_de(HIJO) == "hijo_spawn"
    assert S.rol_de("python -c from multiprocessing.resource_tracker import main;main(4)") == "resource_tracker"
    assert S.rol_de("python3 /tmp/muestreador_128_c0.py a b") == "sonda"
    assert S.rol_de(f"python3 -c import time;{S.MARCA_CONTROL}=1") == "control"
    assert S.rol_de("/sbin/docker-init -- sh -c python -m matrixai_studio --host x") == "init"   # lleva el nombre del servidor
    assert S.rol_de("python3 /tmp/intento_sin_tope_128_c0.py a b") == "guion_del_intento"
    assert S.rol_de("runc init") == "exec_de_docker"
    assert S.rol_de("") == "otro"


def test_fase_en_toma_el_ultimo_evento_ya_ocurrido():
    assert S.fase_en(EVENTOS, -1.0) == "antes"
    assert S.fase_en(EVENTOS, 0.0) == "preparacion"
    assert S.fase_en(EVENTOS, 5.99) == "intento 1/2"
    assert S.fase_en(EVENTOS, 6.0) == "intento 2/2"
    assert S.fase_en(EVENTOS, 99) == "fin"


def test_grupo_con_motor_inferido():
    assert S.grupo_de("intento 2/5", {0: "baseline", 1: "lightgbm"}) == "intento:lightgbm"
    assert S.grupo_de("intento 3/5", {0: "baseline"}) == "intento"
    assert S.grupo_de("preparacion") == "preparacion"
    assert S.grupo_de("directo 1", {0: "baseline", 1: "lightgbm"}) == "directo:lightgbm"
    assert S.grupo_de("aislado 5") == "aislado"


def test_resumen_pico_por_fase_proceso_y_total():
    r = S.resumir(serie(), EVENTOS, {0: "lightgbm", 1: "sklearn.hgb"})
    f = r["por_fase"]
    assert set(f) == {"preparacion", "intento:lightgbm", "intento:sklearn.hgb", "reajuste_final"}
    assert f["preparacion"]["pico_servidor_mb"] == 1500 and f["preparacion"]["pico_hijo_mb"] == 0
    assert f["intento:lightgbm"]["pico_hijo_mb"] == 900 and f["intento:lightgbm"]["pico_total_mb"] == 2 + 1500 + 900
    assert f["intento:sklearn.hgb"]["pico_hijo_mb"] == 400
    assert f["reajuste_final"]["pico_hijo_mb"] == 2000 and f["reajuste_final"]["pico_servidor_mb"] == 1600
    assert f["intento:lightgbm"]["muestras"] == 3
    # el pico total (suma de RSS, sin la sonda) está en el reajuste, y el del cgroup con él
    assert r["pico_total"]["rss_total_mb"] == 2 + 1600 + 2000 and r["pico_total"]["grupo"] == "reajuste_final"
    assert r["pico_cgroup"]["cgroup_mb"] == 2 + 1600 + 2000 + 30 + 50 and r["pico_cgroup"]["t"] == 9.0
    assert r["pico_servidor_mb"] == 1600


def test_resumen_pesa_cada_hijo_frente_al_servidor():
    r = S.resumir(serie(), EVENTOS)
    hijos = {q["pid"]: q for q in r["por_proceso"] if q["rol"] == "hijo_spawn"}
    assert set(hijos) == {21, 22, 23}                       # tres hijos distintos, cada uno con su pico
    assert hijos[21]["pico_rss_mb"] == 900 and hijos[21]["sobre_servidor_en_ese_instante"] == 0.6   # 900 / 1500
    assert hijos[23]["pico_rss_mb"] == 2000 and hijos[23]["sobre_pico_del_servidor"] == 1.25       # 2000 / 1600
    assert hijos[21]["vive_s"] == 2.0 and hijos[21]["fase_del_pico"] == "intento 1/2"
    assert r["hijo_mayor"]["pid"] == 23
    assert r["por_rol"]["hijo_spawn"] == {"procesos_distintos": 3, "pico_rss_mb": 2000}
    assert "sonda" not in r["por_rol"]                       # la sonda no se cuenta a sí misma


def test_resumen_sin_hijos_ni_cgroup():
    ms = [{"t": 0.0, "procesos": [proc(1, 100, SRV)]}, {"t": 1.0, "procesos": [proc(1, 150, SRV)]}]
    r = S.resumir(ms, [(0.0, "preparacion")])
    assert r["hijo_mayor"] is None and r["pico_cgroup"] is None and r["pico_servidor_mb"] == 150


def test_etiqueta_de_estado():
    assert S.etiqueta_de_estado({"estado": "queued"}) == "cola"
    assert S.etiqueta_de_estado({"estado": "running", "n_intentos_total": 15, "n_intentos_completados": 0}) == "intento 1/15"
    assert S.etiqueta_de_estado({"estado": "running", "n_intentos_total": 15, "n_intentos_completados": 14}) == "intento 15/15"
    assert S.etiqueta_de_estado({"estado": "running", "n_intentos_total": 15, "n_intentos_completados": 15}) == "reajuste_final"
    assert S.etiqueta_de_estado({"estado": "completed"}) == "fin"


def test_motores_por_intento_solo_cuenta_la_busqueda():
    lb = [{"engine": "baseline", "fase": "seleccion"}, {"engine": "lightgbm", "fase": "seleccion"},
          {"engine": "lightgbm", "fase": "otra"}]
    assert S.motores_por_intento_de(lb) == {0: "baseline", 1: "lightgbm"}
    assert S.motores_por_intento_de(None) == {}


def _muestras_de_control(visto_a, visto_b, extra=None):
    ctl = lambda pid, mb: proc(pid, mb, f"python3 -c import time;{S.MARCA_CONTROL}=1;b=bytearray(1)")
    ps = [proc(1, 90, SRV), ctl(50, visto_a), ctl(51, visto_b)] + (extra or [])
    return [{"t": 0.0, "procesos": ps}]


def test_control_acepta_lo_reservado_dentro_del_10_por_ciento():
    ok, lin, med = S.comprobar_reservas(_muestras_de_control(309, 612), (300, 600))
    assert ok and len(med) == 2 and {m["pid"] for m in med} == {50, 51}


def test_control_TIENE_DIENTES_rechaza_lo_que_no_cuadra():
    # un 15 % de más, un 15 % de menos, un proceso que falta, o uno de sobra: todos en rojo
    assert not S.comprobar_reservas(_muestras_de_control(345, 600), (300, 600))[0]
    assert not S.comprobar_reservas(_muestras_de_control(300, 510), (300, 600))[0]
    assert not S.comprobar_reservas([{"t": 0.0, "procesos": [proc(1, 90, SRV)]}], (300, 600))[0]
    sobra = [proc(52, 300, f"python3 -c {S.MARCA_CONTROL}")]
    assert not S.comprobar_reservas(_muestras_de_control(300, 600, sobra), (300, 600))[0]
    # y que dos reservas NO se vean como UN proceso de 900: la suma no cuenta
    assert not S.comprobar_reservas(_muestras_de_control(900, 0.5), (300, 600))[0]


def test_muestreador_lee_el_rss_de_este_proceso():
    """El muestreador de verdad (el texto que va al contenedor) ejecutado aquí: ve este proceso con su RSS y su cmdline, y
    deja fuera al que se le dice. Una reserva de 200 MB tocada entera se tiene que notar en ±10 %."""
    ns = {"__name__": "muestreador_bajo_prueba"}
    exec(compile(S.MUESTREADOR, "muestreador", "exec"), ns)
    antes = ns["leer_proceso"](os.getpid())
    assert antes is not None and antes["pid"] == os.getpid() and antes["ppid"] == os.getppid()
    assert "pytest" in antes["cmd"] or "python" in antes["cmd"]
    b = bytearray(200 * 1024 * 1024)
    for i in range(0, len(b), 4096):
        b[i] = 1
    despues = ns["leer_proceso"](os.getpid())
    assert abs((despues["rss_mb"] - antes["rss_mb"]) - 200) <= 20
    assert despues["hwm_mb"] >= despues["rss_mb"] - 1
    m = ns["muestrear_una_vez"]()
    assert any(p["pid"] == os.getpid() for p in m["procesos"]) is False   # por omisión se excluye a sí mismo
    m2 = ns["muestrear_una_vez"](excluir=set())
    assert any(p["pid"] == os.getpid() for p in m2["procesos"])
    assert m2["t"] > 0 and m2["procesos"] == sorted(m2["procesos"], key=lambda p: p["pid"])
    del b


def test_el_muestreador_dice_su_propia_marca_para_que_no_se_cuente():
    assert "muestreador_128_c0" in "python3 /tmp/muestreador_128_c0.py /tmp/x /tmp/y"
    assert S.rol_de("python3 /tmp/muestreador_128_c0.py /tmp/x /tmp/y") == "sonda"


def test_el_guion_del_intento_sin_tope_compila_y_no_toca_el_disco_del_producto():
    compile(S.INTENTO_SIN_TOPE, "intento", "exec")
    t = S.INTENTO_SIN_TOPE
    assert "ej.ejecutar_estudio = medir" in t and "100000.0" in t       # sustituye EN MEMORIA y pone el tope enorme
    assert "estudio_job.py" not in t and "shutil" not in t           # no escribe ni copia nada del producto


def test_importar_el_guion_no_deja_bytecode_ni_arranca_nada():
    assert sys.dont_write_bytecode is True
    assert S._contenedor is None


def test_el_control_virtual_distingue_la_residente_de_la_virtual():
    """Sabotaje del 08-10: leer VmSize en vez de VmRSS dejaba el control VERDE. Un proceso con 1 GB reservado sin tocar
    tiene que verse casi a cero; visto como un giga (lo que daría la virtual), el control dice NO CUADRA."""
    s = _cargar()
    def serie(rss):
        return [{"t": 0.0, "procesos": [{"pid": 7, "cmd": f"python3 -c import time,mmap;{s.MARCA_VIRTUAL}=1", "rss_mb": rss}]}]
    assert s.comprobar_virtual(serie(9.5))[0] is True
    assert s.comprobar_virtual(serie(1034.0))[0] is False
    assert s.comprobar_virtual([{"t": 0.0, "procesos": []}])[0] is False


# ----------------------------------------------------------------------------- 128-C1 paso 0: la RETENCIÓN
def test_la_retencion_compila_mide_antes_y_despues_y_no_toca_el_producto():
    compile(S.RETENCION, "retencion", "exec")
    t = S.RETENCION
    for paso in ("gc.collect()", "malloc_trim(0)", "smaps_rollup", "VmRSS", "mallinfo2"):
        assert paso in t
    assert "2_estudio_terminado_tal_cual" in t and "4_tras_gc_collect" in t and "5_tras_malloc_trim_0" in t
    assert "estudio_job.py" not in t and "shutil" not in t and "ej.ejecutar_estudio =" not in t   # no sustituye nada
    assert S.rol_de("python3 /tmp/retencion_128_c1.py /tmp/a /tmp/b") == "guion_de_retencion"


def test_la_retencion_lee_la_memoria_de_este_proceso_y_el_asignador():
    """Se ejecutan sus funciones (sin el estudio): el RSS de `/proc` y `mallinfo2`, con `malloc_trim` de verdad."""
    import ctypes
    ns = {"__name__": "retencion_bajo_prueba"}
    exec(compile(S.RETENCION, "retencion", "exec"), ns)
    est = ns["estado_del_proceso"]()
    assert est["status_mb"]["VmRSS"] > 5 and est["smaps_rollup_mb"]["Rss"] > 5
    libc = ctypes.CDLL("libc.so.6")
    antes = ns["mallinfo"](libc)
    assert antes is None or antes["arena_mb"] >= 0
    bloque = [bytearray(1024) for _ in range(200_000)]       # ~200 MB en trozos pequeños (el asignador, no mmap)
    grande = ns["estado_del_proceso"]()["status_mb"]["VmRSS"]
    assert grande > est["status_mb"]["VmRSS"] + 100
    del bloque
    libc.malloc_trim(0)
    assert ns["estado_del_proceso"]()["status_mb"]["VmRSS"] < grande - 50     # lo que se suelta, se devuelve


# ----------------------------------------------------------------------------- 128-C3b: TabM FORZADO en los grandes
_ESTUDIO_JOB_DE_MENTIRA = '''
import threading
_MOTOR_TABM = "matrixai.dense.tabm_cpu"
_jobs = {}


def _memoria_de_tabm_que_no_cabe(filas, columnas_de_entrada):
    return (7.5, 6.0)                                  # la de verdad: con 6 GB, NO cabe


def _decidir_motores(guarda):
    return ["sklearn.hgb"] + ([] if guarda(60000, 170) else [_MOTOR_TABM])


def iniciar_estudio(cuerpo):
    # como `estudio_job`: la guarda se busca por su nombre GLOBAL al decidir (la sustitución del módulo llega)
    motores = _decidir_motores(GUARDA_FIJADA or (lambda f, c: _memoria_de_tabm_que_no_cabe(f, c)))
    job = {"estado": "running", "leaderboard": []}

    def correr():
        for m in motores:
            job["leaderboard"].append({"engine": m, "wall_seconds": 1.0})
        job["estado"] = "completed"
    job["hilo"] = threading.Thread(target=correr)
    job["hilo"].start()
    _jobs["j1"] = job
    return {"ok": True, "estudio": {"job_id": "j1"}}


def estudio_status(job_id):
    j = _jobs[job_id]
    return {"estado": j["estado"], "n_intentos_completados": len(j["leaderboard"]), "n_intentos_total": 2,
            "leaderboard": list(j["leaderboard"]), "seleccion": {"candidate_engine": "sklearn.hgb"}}
'''


def _correr_el_guion_tabm(tmp_path, monkeypatch, *, guarda_fijada_al_importar):
    """El guion `TABM_FORZADO` de verdad contra un `estudio_job` de mentira. Con `guarda_fijada_al_importar`, el estudio usa
    una copia de la guarda tomada ANTES de sustituirla: la sustitución no llega (el caso que el guion tiene que cazar)."""
    import json as _json
    import types
    paquete, ej = types.ModuleType("matrixai_studio"), types.ModuleType("matrixai_studio.estudio_job")
    exec(compile(_ESTUDIO_JOB_DE_MENTIRA, "ej_de_mentira", "exec"), ej.__dict__)
    ej.GUARDA_FIJADA = ej._memoria_de_tabm_que_no_cabe if guarda_fijada_al_importar else None
    paquete.estudio_job = ej
    monkeypatch.setitem(sys.modules, "matrixai_studio", paquete)
    monkeypatch.setitem(sys.modules, "matrixai_studio.estudio_job", ej)
    cuerpo, salida = tmp_path / "cuerpo.json", tmp_path / "salida.json"
    cuerpo.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["guion", str(cuerpo), str(salida)])
    ns = {"__name__": "tabm_forzado_bajo_prueba"}
    exec(compile(S.TABM_FORZADO, "tabm_forzado", "exec"), ns)
    codigo = 0
    try:
        ns["main"]()
    except SystemExit as e:
        codigo = e.code
    return codigo, _json.loads(salida.read_text())


def test_el_guion_de_tabm_forzado_compila_sustituye_en_memoria_y_no_toca_el_producto():
    compile(S.TABM_FORZADO, "tabm_forzado", "exec")
    t = S.TABM_FORZADO
    assert "ej._memoria_de_tabm_que_no_cabe = callada" in t                  # EN MEMORIA, por su punto de sustitución
    assert "estudio_job.py" not in t and "shutil" not in t                   # ni escribe ni copia nada del producto
    assert S.rol_con_el_guion_de_servidor("python3 /tmp/tabm_forzado_128_c3b.py /tmp/a /tmp/b") == "servidor"
    assert S.rol_con_el_guion_de_servidor(SRV) == "servidor_ocioso"         # el servidor HTTP del contenedor, aparte
    assert S.rol_con_el_guion_de_servidor(HIJO) == "hijo_spawn"


def test_el_guion_de_tabm_forzado_hace_competir_a_tabm_y_apunta_lo_que_habria_dicho_la_guarda(tmp_path, monkeypatch):
    codigo, med = _correr_el_guion_tabm(tmp_path, monkeypatch, guarda_fijada_al_importar=False)
    assert codigo == 0
    assert med["intentos_de_tabm"] == 1
    assert med["guarda_de_verdad"] == [{"filas": 60000, "columnas_de_entrada": 170,
                                        "habria_dicho_estimada_y_disponible_gb": [7.5, 6.0]}]
    assert med["estado_final"]["estado"] == "completed" and med["estados"][-1][1] == "completed"
    assert S.veredicto_tabm_forzado(codigo, med) == "completed"


def test_el_guion_de_tabm_forzado_TIENE_DIENTES_si_la_sustitucion_no_llega_sale_con_3(tmp_path, monkeypatch):
    """Una guarda copiada antes de sustituirla (otro módulo que la importara por su nombre, por ejemplo) deja a TabM fuera:
    el estudio termina, pero la medida no mide lo que dice. Eso NO puede salir «completed»."""
    codigo, med = _correr_el_guion_tabm(tmp_path, monkeypatch, guarda_fijada_al_importar=True)
    assert codigo == 3 and med["intentos_de_tabm"] == 0
    assert S.veredicto_tabm_forzado(codigo, med) == "sin_tabm"
    assert S.veredicto_tabm_forzado(0, med) == "sin_tabm"                    # aunque el código no lo dijera


def test_veredicto_de_tabm_forzado():
    terminado = {"estado_final": {"estado": "completed"}, "intentos_de_tabm": 4}
    assert S.veredicto_tabm_forzado(0, terminado) == "completed"
    assert S.veredicto_tabm_forzado(0, {"estado_final": {"estado": "failed"}, "intentos_de_tabm": 4}) == "failed"
    assert S.veredicto_tabm_forzado(-9, {"estados": [[1.0, "running", 2, 9]]}) == "failed"     # el OOM se llevó el guion
    assert S.veredicto_tabm_forzado(-9, None) == "failed"


def test_eventos_de_los_estados_del_guion():
    med = {"t_inicio": 10.0, "t_fin": 60.0,
           "estados": [[11.0, "queued", 0, None], [12.0, "running", 0, 3], [20.0, "running", 1, 3], [30.0, "running", 2, 3],
                       [40.0, "running", 3, 3], [60.0, "completed", 3, 3]]}
    assert S.eventos_de_los_estados(med) == [(10.0, "preparacion"), (11.0, "cola"), (12.0, "intento 1/3"),
                                             (20.0, "intento 2/3"), (30.0, "intento 3/3"), (40.0, "reajuste_final"),
                                             (60.0, "fin")]
    sin_fin = {"t_inicio": 10.0, "estados": [[12.0, "running", 0, 3]]}      # el OOM se llevó el guion: la fase sigue abierta
    assert S.eventos_de_los_estados(sin_fin) == [(10.0, "preparacion"), (12.0, "intento 1/3")]


def test_tabm_forzado_lleva_el_entorno_del_producto_cpu_y_nunca_apaga_tabm():
    assert S.ENTORNO_TABM_FORZADO == S.r0.entorno_extra_de("pcpu")
    assert S.ENTORNO_TABM_FORZADO.get("MATRIXAI_ESTUDIO_SIN_DENSA") == "1"
    assert "MATRIXAI_ESTUDIO_SIN_TABM" not in S.ENTORNO_TABM_FORZADO


def test_tabm_forzado_arranca_el_contenedor_SIN_el_servidor_http(monkeypatch):
    """Un servidor ocioso al lado del guion sumaría su memoria al cgroup y, al borde del techo, fabricaría un OOM que en
    producción no pasa: la orden de `docker run` lleva `sleep infinity` y ningún puerto, y no se espera ningún HTTP."""
    s = _cargar()
    ordenes = []

    class Hecho:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(s.subprocess, "run", lambda orden, **kw: (ordenes.append(orden), Hecho())[1])
    monkeypatch.setattr(s.signal, "signal", lambda *a: None)
    monkeypatch.setattr(s.atexit, "register", lambda *a: None)
    monkeypatch.setattr(s.r0, "pedir", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sin servidor no hay HTTP")))
    base, nombre = s.arrancar_contenedor("6g", imagen="img:x", sin_servidor=True)
    run = next(o for o in ordenes if o[:2] == ["docker", "run"])
    assert base is None and nombre == s.NOMBRE_CONTENEDOR
    assert run[-3:] == ["img:x", "sleep", "infinity"] and "-p" not in run
    assert "--memory=6g" in run and "--memory-swap=6g" in run and "--init" in run
    assert inspect_fuente(s.medir_tabm_forzado).count("sin_servidor=True") == 1


def inspect_fuente(f):
    import inspect
    return inspect.getsource(f)


def test_medir_tabm_forzado_compone_el_registro_entero_y_comparable_con_el_del_producto(tmp_path, monkeypatch):
    """El lado ANFITRIÓN de la medida (Docker simulado): un fallo aquí, tras media hora de estudio, perdería el resultado.
    El registro lleva la cifra del test (`r0.test_de`), el pico del cgroup como `memoria_pico_mb`, los OOM del cgroup y el
    estado entero guardado; y `comparar_c4_128` lo compara con un registro del producto sin quejarse."""
    import gzip as _gzip
    import json as _json
    s = _cargar()
    med = {"t_inicio": 100.0, "t_fin": 160.0, "intentos_de_tabm": 1, "motor_tabm": "matrixai.dense.tabm_cpu",
           "guarda_de_verdad": [{"filas": 60000, "columnas_de_entrada": 170,
                                 "habria_dicho_estimada_y_disponible_gb": [7.0, 6.0]}],
           "estados": [[101.0, "running", 0, 1], [150.0, "running", 1, 1], [160.0, "completed", 1, 1]],
           "estado_final": {"estado": "completed", "n_intentos_total": 1, "motores_fuera_por_memoria": {},
                            "leaderboard": [{"candidate": "matrixai.dense.tabm_cpu-csv-p0-r0", "engine": "matrixai.dense.tabm_cpu",
                                             "fase": "seleccion", "estado": "completed", "wall_seconds": 40.0,
                                             "metrica_de_seleccion": {"metric_id": "auroc", "value": 0.95}}],
                            "seleccion": {"candidate_engine": "matrixai.dense.tabm_cpu",
                                          "media_de_la_seleccion": {"metrica": "auroc", "motores": [
                                              {"motor": "matrixai.dense.tabm_cpu", "valor": 0.95, "compite": True}]},
                                          "evaluacion_final": {"evaluated_role": "test", "evidence": "independent_test",
                                                               "metrics": [{"metric_id": "auroc", "value": 0.93}]}}}}

    class Hecho:
        def __init__(self, stdout="", returncode=0):
            self.stdout, self.returncode, self.stderr = stdout, returncode, ""

    def run(orden, **kw):
        texto = " ".join(map(str, orden))
        if "cat /tmp/tabm_forzado_128_c3b.json" in texto:
            return Hecho(_json.dumps(med))
        if "memory.events" in texto:
            return Hecho("low 0\nhigh 0\nmax 2\noom 0\noom_kill 0\n")
        if orden[:2] == ["docker", "inspect"]:
            return Hecho("false|0|running|")
        return Hecho()

    class MuestreadorDeMentira:
        def __init__(self, nombre): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def leer(self):
            return [{"t": 120.0, "cgroup_mb": 2500.0, "procesos": [
                proc(1, 2000.0, "python3 /tmp/tabm_forzado_128_c3b.py /tmp/c /tmp/s"), proc(2, 400.0, HIJO, ppid=1)]}]

    monkeypatch.setattr(s.subprocess, "run", run)
    monkeypatch.setattr(s, "comprobar_que_se_puede", lambda: 1.0)
    monkeypatch.setattr(s, "arrancar_contenedor", lambda memoria, imagen=None, sin_servidor=False: (None, "sonda-128-c0"))
    monkeypatch.setattr(s, "poner_en_el_contenedor", lambda *a: None)
    monkeypatch.setattr(s, "borrar_contenedor", lambda: None)
    monkeypatch.setattr(s, "Muestreador", MuestreadorDeMentira)
    monkeypatch.setattr(s, "AGENTE", tmp_path)
    monkeypatch.setattr(s.r0, "csv_de", lambda gen, data_id: (b"a,y\n1,0\n", ["a", "y"], [], 1))
    monkeypatch.setattr(s.time, "sleep", lambda x: None)
    c = {"data_id": 1, "objetivo": "y", "tarea": "binary_classification", "clase_positiva": "1"}
    rec = s.medir_tabm_forzado("APSFailure", c, None, "6g", tmp_path / "series", 3600)

    assert rec["estado"] == "completed" and rec["campeon"] == "matrixai.dense.tabm_cpu"
    assert rec["test"] == {"rol": "test", "evidencia": "independent_test", "metricas": {"auroc": 0.93}}
    assert rec["memoria_pico_mb"] == 2500.0 and rec["memory_events"]["oom_kill"] == 0
    assert rec["guarda_de_verdad"][0]["habria_dicho_estimada_y_disponible_gb"] == [7.0, 6.0]
    assert rec["resumen_de_memoria"]["pico_servidor_mb"] == 2000.0               # el guion cuenta como el servidor
    guardado = tmp_path / "series" / rec["estado_final_guardado"]["fichero"]
    assert _json.load(_gzip.open(guardado, "rt"))["seleccion"]["candidate_engine"] == "matrixai.dense.tabm_cpu"
    assert not list(tmp_path.glob("_*_cuerpo_tabm.json"))                         # el cuerpo con los datos no se queda

    spec = importlib.util.spec_from_file_location("comparar_c4_128_desde_la_sonda", RUTA.with_name("comparar_c4_128.py"))
    cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)
    producto = dict(rec, campeon="lightgbm", test={"rol": "test", "evidencia": "independent_test", "metricas": {"auroc": 0.92}},
                    media_de_la_seleccion={"metrica": "auroc", "motores": [{"motor": "lightgbm", "valor": 0.94, "compite": True}]})
    fila = cmp.comparar_uno("APSFailure", producto, rec, exacto=False, suelo=0.0)
    assert fila["clase"] == "sube" and fila["mismo_campeon"] is False and round(fila["diferencia"], 6) == 1.0


# ============================================================================ 128-R4: TabM por FASES (sin Docker)
# El guion `TABM_POR_FASES` corre dentro del contenedor; aquí se ejecuta su texto contra un motor DE MENTIRA con un reloj
# simulado: cada función falsa «tarda» lo que dice avanzando el reloj, así que los segundos esperados son exactos.
import json
import threading
import types

import pytest


def _guion():
    ns = {"__name__": "tabm_por_fases_bajo_prueba"}
    exec(compile(S.TABM_POR_FASES, "tabm_por_fases", "exec"), ns)
    return ns


class Reloj:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def avanza(self, s):
        self.t += s


class Memoria:
    """RSS/VmHWM simulados: `hwm` es el máximo desde el último reinicio, como en /proc."""

    def __init__(self):
        self.rss, self.hwm, self.reinicios, self.deja_reiniciar = 300.0, 300.0, 0, True

    def __call__(self):
        return self.rss, self.hwm

    def sube(self, mb):
        self.rss += mb
        self.hwm = max(self.hwm, self.rss)

    def reiniciar(self):
        if not self.deja_reiniciar:
            return False
        self.reinicios += 1
        self.hwm = self.rss
        return True


PERDIDAS = [1.0, 0.8, 0.9]            # época 1 mejora, 2 mejora, 3 no


def motor_de_mentira(reloj, mem):
    """Lo que el guion envuelve, con la forma de las tres piezas reales: la densa (funciones del módulo y una clase con
    `fit`/`_ajustar`/`predict`/..., donde `predict*` se HEREDAN de la base), `preparacion_tabm` y `tabm_plr`. `_ajustar` llama
    por NOMBRE DE MÓDULO a lo que el real importa al llamar, así que una sustitución en el módulo se ve."""
    prep, plr, densa = types.ModuleType("prep"), types.ModuleType("plr"), types.ModuleType("densa")

    def _tipificar(*a):
        reloj.avanza(1.0)

    def ajustar_preparacion(*a):
        reloj.avanza(2.0)
        prep._tipificar()                        # por el nombre del módulo: la sustitución tiene que llegar aquí
        return "preparacion"

    def aplicar_preparacion(*a):
        reloj.avanza(0.5)
        return "x", "c"

    def ajustar_estandarizacion_objetivo(*a):
        reloj.avanza(0.25)

    def aplicar_estandarizacion_objetivo(*a):
        reloj.avanza(0.25)

    for f in (_tipificar, ajustar_preparacion, aplicar_preparacion, ajustar_estandarizacion_objetivo, aplicar_estandarizacion_objetivo):
        setattr(prep, f.__name__, f)

    epoca = {"n": 0}

    class RedTabMPLR:
        def __init__(self, *a, **k):
            reloj.avanza(0.7)

    def perdida_del_ensamblado_por_lotes(*a):
        reloj.avanza(0.4)
        v = PERDIDAS[epoca["n"]]
        epoca["n"] += 1
        return v

    plr.RedTabMPLR, plr.perdida_del_ensamblado_por_lotes = RedTabMPLR, perdida_del_ensamblado_por_lotes

    def _ahora():
        return reloj()

    def _a_tensor(m):
        reloj.avanza(0.5)
        mem.sube(200)                            # un pico de +200 MB...
        mem.rss -= 200                           # ...que se suelta
        return m

    def _a_dispositivo(t, d):
        reloj.avanza(0.25)
        return t

    def _tensor_a_json(t):
        reloj.avanza(0.1)
        return {}

    class Base:
        def predict(self, *a):
            reloj.avanza(0.6)

        def predict_proba(self, *a):
            reloj.avanza(0.6)

        def decision_scores(self, *a):
            reloj.avanza(0.6)

    class MotorDensaTabM(Base):
        def fit(self, *a):
            r = self._ajustar(*a)
            reloj.avanza(0.3)                    # el digest del predictor, fuera de `_ajustar`
            return r

        def _ajustar(self, *a):
            epoca["n"] = 0
            densa._ahora()
            prep.ajustar_preparacion()
            prep.aplicar_preparacion()
            prep.aplicar_preparacion()
            prep.ajustar_estandarizacion_objetivo()
            prep.aplicar_estandarizacion_objetivo()
            densa._a_tensor(1)
            plr.RedTabMPLR()
            densa._a_dispositivo(1, 1)
            densa._a_dispositivo(1, 1)
            reloj.avanza(0.05)                   # AdamW y generadores
            densa._ahora()                       # «se agotó preparando?»
            mejor = float("inf")
            for _ in PERDIDAS:
                densa._ahora()                   # comienzo de la época
                reloj.avanza(2.0)                # entrenar
                mem.sube(400)                    # un pico de +400 MB en el bucle (nada envuelto) que se suelta antes de validar
                mem.rss -= 400
                densa._ahora()                   # una llamada por lote
                perdida = plr.perdida_del_ensamblado_por_lotes()
                if perdida < mejor:
                    mejor = perdida
                    reloj.avanza(0.2)            # el checkpoint
                densa._ahora()                   # duración de la anterior
            reloj.avanza(0.1)                    # load_state_dict y armar el predictor: NO está en ninguna parte
            densa._tensor_a_json(1)
            densa._tensor_a_json(1)
            return "resultado"

        def _forward(self, *a):
            prep.aplicar_preparacion()
            reloj.avanza(0.3)

        def _reconstruir_red(self, *a):
            plr.RedTabMPLR()

    for f in (_ahora, _a_tensor, _a_dispositivo, _tensor_a_json):
        setattr(densa, f.__name__, f)
    densa.MotorDensaTabM = MotorDensaTabM
    return densa, prep, plr


@pytest.fixture
def banco():
    g = _guion()
    reloj, mem = Reloj(), Memoria()
    densa, prep, plr = motor_de_mentira(reloj, mem)
    traza = g["Traza"](reloj=reloj, memoria=mem, reiniciar=mem.reiniciar)
    return types.SimpleNamespace(g=g, reloj=reloj, mem=mem, densa=densa, prep=prep, plr=plr, traza=traza)


def correr(b, motor=None):
    desinstalar = b.g["instalar"](b.traza, b.densa, b.prep, b.plr)
    m = motor or b.densa.MotorDensaTabM()
    with b.traza.fase("ejecutar_intento", envolvente=True):
        m.fit()
        m.predict()
        m.predict_proba()
    return desinstalar, m


def por_nombre(traza, nombre):
    return [f for f in traza.fases if f["nombre"] == nombre]


def test_R4_el_guion_compila_y_no_toca_el_disco_del_producto():
    compile(S.TABM_POR_FASES, "tabm_por_fases", "exec")
    t = S.TABM_POR_FASES
    assert "ej.ejecutar_estudio = medir" in t and "ej._memoria_de_tabm_que_no_cabe = callada" in t
    assert "shutil" not in t and "write_text" not in t and 'open(salida, "w")' not in t
    assert "en_fichero.cargar()" in t                    # las particiones del paso 4 viajan en un fichero
    assert "dataclasses.replace(d.presupuesto, wall_seconds=plazo)" in t


def test_R4_el_guion_de_intento_sin_tope_carga_las_particiones_del_fichero():
    """El arreglo del paso 4 (copiado TAL CUAL de la rama de integración): `d.train` ahora es None cuando viajan en fichero."""
    t = S.INTENTO_SIN_TOPE
    assert "d.en_fichero.cargar()" in t and "train, validation, test = d.train, d.validation, d.test" in t
    assert "d.train.row_ids" not in t and "pickle.dumps((motor, train, validation, test, d.spec)" in t
    assert "sys.exit(3)" in t and "intentos.soltar()" in t


def test_R4_los_temporizadores_envuelven_lo_que_dicen_con_sus_segundos_exactos(banco):
    desinstalar, _ = correr(banco)
    tz = banco.traza
    w = lambda n: round(sum(f["wall_s"] for f in por_nombre(tz, n)), 4)
    # preparación 3,0 (2,0 + tipificar 1,0) + 2 aplicar 1,0 + objetivo 0,5 + tensor 0,5 + red 0,7 + 2 a_dispositivo 0,5 + AdamW 0,05
    # + 3 épocas (2,0 + 0,4) + 2 checkpoints 0,4 + sin dueño 0,1 + exportar 0,2
    assert w("ajustar") == pytest.approx(3.0 + 1.0 + 0.5 + 0.5 + 0.7 + 0.5 + 0.05 + 3 * 2.4 + 0.4 + 0.1 + 0.2)
    assert w("fit") == pytest.approx(w("ajustar") + 0.3)
    assert w("ajustar_preparacion") == 3.0 and w("tipificar") == 1.0       # la envolvente CONTIENE a su hija
    assert w("aplicar_preparacion") == 1.0 + 0.0           # las dos de `_ajustar` (la de `_forward` no se llamó todavía)
    assert w("a_tensor") == 0.5 and w("construir_red") == 0.7 and w("a_dispositivo") == 0.5
    assert w("validar") == pytest.approx(1.2) and w("exportar_pesos") == pytest.approx(0.2)
    assert w("predict") == 0.6 and w("predict_proba") == 0.6
    # la anidación: `tipificar` cuelga de `ajustar_preparacion`, que cuelga de `ajustar`, que cuelga de `fit`
    assert por_nombre(tz, "tipificar")[0]["ruta"] == "ejecutar_intento/fit/ajustar/ajustar_preparacion/tipificar"
    aj = por_nombre(tz, "ajustar")[0]
    assert por_nombre(tz, "ajustar_preparacion")[0]["padre"] == aj["i"]
    assert por_nombre(tz, "fit")[0]["envolvente"] and por_nombre(tz, "tipificar")[0]["envolvente"] is False
    desinstalar()


def test_R4_el_entrenamiento_el_checkpoint_y_el_optimizador_salen_de_las_marcas(banco):
    desinstalar, _ = correr(banco)
    ep = banco.traza.epocas
    assert [e["n"] for e in ep] == [1, 2, 3]
    assert [e["entrenar_s"] for e in ep] == [2.0, 2.0, 2.0]
    assert [e["validar_s"] for e in ep] == [0.4, 0.4, 0.4]
    assert [e["checkpoint_s"] for e in ep] == [0.2, 0.2, 0.0]          # la 3.ª no mejora: no hay copia del estado
    assert [e["mejoro"] for e in ep] == [True, True, False]
    assert [e["perdida_val"] for e in ep] == PERDIDAS
    assert banco.traza.optimizador_s == 0.05
    desinstalar()


def test_R4_las_partes_del_ajuste_suman_y_lo_que_falta_se_da_como_otros(banco):
    desinstalar, _ = correr(banco)
    c = {"fases": banco.traza.fases, "epocas": banco.traza.epocas, "optimizador_s": banco.traza.optimizador_s}
    p = S.partes_del_ajuste(c)
    assert p["preparar_s"] == pytest.approx(3.0 + 0.5 + 0.5 + 0.25 + 0.25) and p["tipificar_s"] == 1.0   # tipificar va DENTRO
    assert p["entrenar_s"] == pytest.approx(6.0) and p["validar_s"] == pytest.approx(1.2) and p["checkpoint_s"] == pytest.approx(0.4)
    assert p["exportar_pesos_s"] == pytest.approx(0.2)
    assert p["otros_s"] == pytest.approx(0.1, abs=1e-3)               # el avance «sin dueño» de `_ajustar`
    assert p["otros_caben"] is True
    desinstalar()


def test_R4_desinstalar_deja_el_motor_como_estaba_tambien_lo_heredado(banco):
    antes = {(id(o), a): o.__dict__.get(a) for o, a in
             ((banco.densa, "_ahora"), (banco.densa, "_a_tensor"), (banco.prep, "_tipificar"), (banco.plr, "RedTabMPLR"),
              (banco.plr, "perdida_del_ensamblado_por_lotes"), (banco.densa.MotorDensaTabM, "_ajustar"))}
    cls = banco.densa.MotorDensaTabM
    assert "predict" not in vars(cls)                                  # se hereda
    desinstalar, _ = correr(banco)
    assert "predict" in vars(cls) and "fit" in vars(cls) and cls.fit.__wrapped__ is not None
    desinstalar()
    assert "predict" not in vars(cls)                                  # lo heredado vuelve a heredarse, no queda un envoltorio
    despues = {(id(o), a): o.__dict__.get(a) for o, a in
               ((banco.densa, "_ahora"), (banco.densa, "_a_tensor"), (banco.prep, "_tipificar"), (banco.plr, "RedTabMPLR"),
                (banco.plr, "perdida_del_ensamblado_por_lotes"), (cls, "_ajustar"))}
    assert antes == despues
    n = len(banco.traza.fases)
    cls().fit()
    assert len(banco.traza.fases) == n                                 # y ya no mide


def test_R4_un_nombre_que_no_existe_es_un_error_y_no_deja_nada_puesto(banco):
    del banco.prep._tipificar
    cls = banco.densa.MotorDensaTabM
    antes = {n: cls.__dict__[n] for n in ("fit", "_ajustar", "_forward")}
    with pytest.raises(AttributeError):
        banco.g["instalar"](banco.traza, banco.densa, banco.prep, banco.plr)
    assert {n: cls.__dict__[n] for n in antes} == antes                       # lo ya puesto se quitó
    assert "predict" not in vars(cls)


def test_R4_una_fase_que_falla_se_cierra_y_no_deja_la_pila_abierta(banco):
    with pytest.raises(RuntimeError):
        with banco.traza.fase("a"):
            banco.reloj.avanza(1.5)
            raise RuntimeError("x")
    assert banco.traza.pila == [] and banco.traza.fases[0]["wall_s"] == 1.5


def test_R4_el_pico_de_una_envolvente_conserva_lo_que_paso_dentro_aunque_se_reinicie_el_hwm(banco):
    desinstalar, _ = correr(banco)
    aj = por_nombre(banco.traza, "ajustar")[0]
    tensor = por_nombre(banco.traza, "a_tensor")[0]
    assert tensor["pico_mb"] == 500.0                                  # 300 + 200 de pico
    assert aj["pico_mb"] >= 700.0                                      # el pico del BUCLE (nada envuelto) también: se pliega al reiniciar
    assert [e["hwm_entrenar_mb"] for e in banco.traza.epocas] == [700.0, 700.0, 700.0]
    assert banco.traza.hwm_se_reinicia is True and banco.mem.reinicios > 0
    desinstalar()


def test_R4_si_el_nucleo_no_deja_reiniciar_el_hwm_se_dice(banco):
    banco.mem.deja_reiniciar = False
    desinstalar, _ = correr(banco)
    assert banco.traza.hwm_se_reinicia is False
    desinstalar()


def test_R4_cada_corrida_cuadra_y_no_falta_ninguna_fase(banco):
    t0 = banco.reloj()
    desinstalar, _ = correr(banco)
    banco.reloj.avanza(0.0)
    c = {"fases": banco.traza.fases, "epocas": banco.traza.epocas, "optimizador_s": banco.traza.optimizador_s,
         "estado": "completed", "pared_s": banco.reloj() - t0}
    c["recursos"] = {"wall_seconds": por_nombre(banco.traza, "ajustar")[0]["wall_s"]}
    q = S.cuadre_de_corrida(c)
    assert q["cuadra"] and len(q["comprobaciones"]) == 3
    assert S.faltan_fases(c) == []
    desinstalar()


def _corrida_a_mano(**cambios):
    fases = [{"i": 0, "nombre": "cargar_particion", "padre": None, "ruta": "cargar_particion", "wall_s": 2.0, "envolvente": False,
              "rss_ini_mb": 100.0, "rss_fin_mb": 300.0, "pico_mb": 320.0},
             {"i": 1, "nombre": "ejecutar_intento", "padre": None, "ruta": "ejecutar_intento", "wall_s": 20.0, "envolvente": True,
              "rss_ini_mb": 300.0, "rss_fin_mb": 400.0, "pico_mb": 900.0},
             {"i": 2, "nombre": "fit", "padre": 1, "ruta": "ejecutar_intento/fit", "wall_s": 19.0, "envolvente": True,
              "rss_ini_mb": 300.0, "rss_fin_mb": 400.0, "pico_mb": 900.0},
             {"i": 3, "nombre": "ajustar", "padre": 2, "ruta": "ejecutar_intento/fit/ajustar", "wall_s": 18.0, "envolvente": True,
              "rss_ini_mb": 300.0, "rss_fin_mb": 400.0, "pico_mb": 900.0},
             {"i": 4, "nombre": "ajustar_preparacion", "padre": 3, "ruta": "x/ajustar_preparacion", "wall_s": 6.0, "envolvente": True,
              "rss_ini_mb": 300.0, "rss_fin_mb": 500.0, "pico_mb": 600.0},
             {"i": 5, "nombre": "a_tensor", "padre": 3, "ruta": "x/a_tensor", "wall_s": 1.0, "envolvente": False,
              "rss_ini_mb": 500.0, "rss_fin_mb": 600.0, "pico_mb": 700.0},
             {"i": 6, "nombre": "construir_red", "padre": 3, "ruta": "x/construir_red", "wall_s": 0.5, "envolvente": False,
              "rss_ini_mb": 600.0, "rss_fin_mb": 600.0, "pico_mb": 600.0},
             {"i": 7, "nombre": "a_dispositivo", "padre": 3, "ruta": "x/a_dispositivo", "wall_s": 0.5, "envolvente": False,
              "rss_ini_mb": 600.0, "rss_fin_mb": 600.0, "pico_mb": 600.0}]
    epocas = [{"n": 1, "entrenar_s": 4.0, "validar_s": 1.0, "checkpoint_s": 0.5, "perdida_val": 0.7, "mejoro": True, "hwm_entrenar_mb": 900.0},
              {"n": 2, "entrenar_s": 4.0, "validar_s": 1.0, "checkpoint_s": 0.0, "perdida_val": 0.8, "mejoro": False, "hwm_entrenar_mb": 880.0}]
    c = {"rotulo": "plazo_del_producto", "plazo_s": 45.0, "estado": "completed_budget_limited", "fases": fases, "epocas": epocas,
         "optimizador_s": 0.0, "pared_s": 22.0, "recursos": {"wall_seconds": 18.0}}
    c.update(cambios)
    return c


def test_R4_el_resumen_de_una_corrida_a_mano():
    r = S.resumir_corrida(_corrida_a_mano())
    p = r["partes_del_ajuste"]
    assert (p["ajustar_s"], p["preparar_s"], p["entrenar_s"], p["validar_s"], p["checkpoint_s"]) == (18.0, 6.0, 8.0, 2.0, 0.5)
    assert p["otros_s"] == pytest.approx(18.0 - (6.0 + 1.0 + 0.5 + 0.5 + 8.0 + 2.0 + 0.5)) and p["otros_s"] == -0.5   # se da tal cual
    assert r["epocas"]["n"] == 2 and r["epocas"]["mejor_epoca"] == 1 and r["epocas"]["entrenar_mediana_s"] == 4.0
    assert r["epocas"]["perdida_val_mejor"] == 0.7 and r["epocas"]["hwm_entrenar_max_mb"] == 900.0
    assert r["pico_de_la_corrida_mb"] == 900.0 and r["por_ruta"]["x/a_tensor"]["llamadas"] == 1
    assert r["cuadre"]["cuadra"] is True and r["faltan_fases"] == ["aplicar_preparacion", "validar", "exportar_pesos", "predict"]


def test_R4_el_cuadre_TIENE_DIENTES():
    # otros muy grande: algo no está medido
    c = _corrida_a_mano()
    c["fases"][3] = dict(c["fases"][3], wall_s=40.0)
    assert not S.cuadre_de_corrida(dict(c, recursos={"wall_seconds": 40.0}))["cuadra"]
    # ajustar del guion distinto del que mide el motor
    assert not S.cuadre_de_corrida(_corrida_a_mano(recursos={"wall_seconds": 30.0}))["cuadra"]
    # las fases de primer nivel no suman la pared de fuera
    assert not S.cuadre_de_corrida(_corrida_a_mano(pared_s=60.0))["cuadra"]
    # un tiempo repetido (partes que se pisan): otros muy negativo
    e = _corrida_a_mano()
    e["epocas"][0] = dict(e["epocas"][0], entrenar_s=30.0)
    assert not S.cuadre_de_corrida(e)["cuadra"]
    # sin recursos del motor (no hubo candidato) se salta esa comprobación, no la da por buena ni por mala
    q = S.cuadre_de_corrida(_corrida_a_mano(recursos=None))
    assert len(q["comprobaciones"]) == 2 and q["cuadra"]
    # sin nada que comprobar NO cuadra
    assert S.cuadre_de_corrida({"fases": [], "epocas": []})["cuadra"] is False


def test_R4_faltan_fases_pide_mas_si_hubo_candidato():
    c = _corrida_a_mano(estado="failed")
    assert S.faltan_fases(c) == ["aplicar_preparacion"] and "validar" not in S.faltan_fases(c)
    assert "fit" in S.faltan_fases(dict(c, fases=[f for f in c["fases"] if f["nombre"] != "fit"]))
    assert "validar" in S.faltan_fases(_corrida_a_mano(estado="completed"))


def test_R4_el_veredicto_y_la_linea():
    ok = _corrida_a_mano()
    ok["fases"] += [{"i": 8, "nombre": n, "padre": 3, "ruta": n, "wall_s": 0.0, "envolvente": False} for n in
                    ("fit", "aplicar_preparacion", "validar", "exportar_pesos", "predict")]
    ok["estado"] = "completed"
    assert S.veredicto_tabm_por_fases(0, {"corridas": [ok, ok]}) == "completed"
    assert S.veredicto_tabm_por_fases(0, {"corridas": [ok, _corrida_a_mano(pared_s=99.0)]}) == "no_cuadra"
    assert S.veredicto_tabm_por_fases(0, {"corridas": [ok]}) == "failed"
    assert S.veredicto_tabm_por_fases(137, {"corridas": [ok]}) == "failed"
    assert S.veredicto_tabm_por_fases(3, {"corridas": [], "error": "TabM no tiene intento"}) == "sin_tabm"
    assert S.veredicto_tabm_por_fases(3, {"corridas": []}) == "failed"
    linea = S.linea_de_fases(S.resumir_corrida(ok))
    assert "plazo_del_producto (45.0 s)" in linea and "preparar 6.0" in linea and "2 épocas" in linea and "cuadre SÍ" in linea
    assert "sin `ajustar` medido" in S.linea_de_fases(S.resumir_corrida({"rotulo": "r", "plazo_s": 1, "estado": "failed"}))


def test_R4_el_rol_del_guion_de_fases_es_el_del_servidor():
    assert S.rol_con_el_guion_por_fases("python3 /tmp/tabm_por_fases_128_r4.py a b 1800") == "servidor"
    assert S.rol_con_el_guion_por_fases(SRV) == "servidor_ocioso" and S.rol_con_el_guion_por_fases(HIJO) == "hijo_spawn"


# ---- el guion ENTERO (`main`) contra un `estudio_job` de mentira: el cableado, no solo las piezas
def _correr_main(monkeypatch, tmp_path, claves, hay_fichero=True, falla_la_primera=False):
    reloj, mem = Reloj(), Memoria()
    densa, prep, plr = motor_de_mentira(reloj, mem)
    g = _guion()
    visto = {"construidas": [], "presupuestos": [], "guarda": []}

    class Presupuesto:
        def __init__(self, wall_seconds, hilos=1, seed=0):
            self.wall_seconds, self.hilos, self.seed = wall_seconds, hilos, seed

    import dataclasses
    Presupuesto = dataclasses.make_dataclass("Presupuesto", [("wall_seconds", float), ("hilos", int, 1), ("seed", int, 0)])

    class Part:
        row_ids = ["a", "b"]
        features = [[1, 2]]

    class EnFichero:
        def cargar(self):
            return Part(), Part(), Part()

    class Desc:
        def __init__(self, clave):
            self.motor_id, self.dataset, self.pliegue, self.repeticion = clave[2], "d", clave[1], clave[0]
            self.presupuesto = Presupuesto(45.0)
            self.spec, self.split_plan_digest = "spec", "dig"
            self.en_fichero = EnFichero() if hay_fichero else None
            self.train = self.validation = self.test = None if hay_fichero else Part()

        def candidate_id(self):
            return "cand"

    class Intentos:
        _claves = claves

        def _construir(self, *c):
            visto["construidas"].append(c)
            return Desc(c)

    ej = types.ModuleType("matrixai_studio.estudio_job")
    ej._MOTOR_TABM = "tabm"
    ej._memoria_de_tabm_que_no_cabe = lambda f, c: ("no cabe", 99)
    ej._jobs, ej.Leaderboard = {}, lambda estudio_id=None: "lb"

    def iniciar_estudio(cuerpo):
        visto["guarda"].append(ej._memoria_de_tabm_que_no_cabe(1, 2))        # lo que haría el producto al decidir

        def hilo():
            ej.ejecutar_estudio(types.SimpleNamespace(estudio_id="e"), {"tabm": densa.MotorDensaTabM()}, Intentos(),
                                fase="seleccion", directorio_checkpoint=tmp_path)
        t = threading.Thread(target=hilo)
        ej._jobs["j"] = {"hilo": t}
        t.start()
        return {"estudio": {"job_id": "j"}}

    ej.iniciar_estudio = iniciar_estudio
    ej.ejecutar_estudio = None

    arch = types.ModuleType("matrixai_engines.harness")

    def ejecutar_intento(motor, train, validation, test, spec, presupuesto, **comun):
        visto["presupuestos"].append(presupuesto.wall_seconds)
        if falla_la_primera and len(visto["presupuestos"]) == 1:
            raise RuntimeError("se rompió")
        motor.fit()
        motor.predict()
        return types.SimpleNamespace(estado="completed", motivo_del_estado=None, recursos={"wall_seconds": 0.0, "cpu_seconds": 0.0},
                                     config_efectiva={"a": {"entrenamiento_efectivo": {"epocas_ejecutadas": 3, "parado_por_plazo": False}}})
    arch.ejecutar_intento = ejecutar_intento
    for nombre, mod in (("matrixai_studio", types.ModuleType("matrixai_studio")), ("matrixai_studio.estudio_job", ej),
                        ("matrixai_engines", types.ModuleType("matrixai_engines")), ("matrixai_engines.harness", arch),
                        ("matrixai_engines.motores", types.ModuleType("matrixai_engines.motores")),
                        ("matrixai_engines.motores.densa_tabm", densa), ("matrixai_engines.redes", types.ModuleType("matrixai_engines.redes")),
                        ("matrixai_engines.redes.preparacion_tabm", prep), ("matrixai_engines.redes.tabm_plr", plr)):
        monkeypatch.setitem(sys.modules, nombre, mod)
    cuerpo, salida = tmp_path / "cuerpo.json", tmp_path / "salida.json"
    cuerpo.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["g", str(cuerpo), str(salida), "1800"])
    codigo = None
    try:
        g["main"]()
    except SystemExit as e:
        codigo = e.code
    return codigo, json.loads(salida.read_text()), visto, (densa, prep, plr)


CLAVES = [(0, 0, "baseline"), (0, 0, "tabm"), (0, 1, "tabm"), (1, 0, "tabm")]


def test_R4_main_construye_solo_el_intento_de_tabm_del_primer_pliegue_y_lo_corre_con_dos_plazos(monkeypatch, tmp_path):
    codigo, R, visto, (densa, prep, plr) = _correr_main(monkeypatch, tmp_path, CLAVES)
    assert codigo is None
    assert visto["construidas"] == [(0, 0, "tabm")]                       # ni el baseline ni otros pliegues
    assert visto["presupuestos"] == [45.0, 1800.0]                       # el del producto y el amplio, EN ESE ORDEN
    assert [c["rotulo"] for c in R["corridas"]] == ["plazo_del_producto", "plazo_amplio"]
    assert [c["plazo_s"] for c in R["corridas"]] == [45.0, 1800.0]
    for c in R["corridas"]:
        nombres = {f["nombre"] for f in c["fases"]}
        assert {"cargar_particion", "ejecutar_intento", "fit", "ajustar", "validar", "exportar_pesos", "predict"} <= nombres
        assert len(c["epocas"]) == 3 and c["filas"]["train"] == 2
        assert c["entrenamiento_efectivo"] == {"epocas_ejecutadas": 3, "parado_por_plazo": False}   # lo que PASÓ, del motor
    assert [f["nombre"] for f in R["construccion"]] == ["construir_intento"]
    assert R["guarda_de_verdad"] and visto["guarda"] == [None]            # la guarda calla, y dice lo que habría dicho
    assert R["guarda_de_verdad"][0]["habria_dicho_estimada_y_disponible_gb"] == ["no cabe", 99]
    assert R["iniciar_estudio"]["wall_s"] >= 0 and R["tope_del_producto_s"] == 45.0
    # y cada corrida se desinstala: el motor queda como estaba entre una y otra y al final
    assert not hasattr(densa.MotorDensaTabM.fit, "__wrapped__") and "predict" not in vars(densa.MotorDensaTabM)


def test_R4_main_con_las_particiones_en_memoria_tambien_funciona(monkeypatch, tmp_path):
    codigo, R, visto, _ = _correr_main(monkeypatch, tmp_path, CLAVES, hay_fichero=False)
    assert codigo is None and len(R["corridas"]) == 2


def test_R4_main_una_corrida_rota_no_quita_la_otra(monkeypatch, tmp_path):
    codigo, R, visto, _ = _correr_main(monkeypatch, tmp_path, CLAVES, falla_la_primera=True)
    assert [c["estado"] for c in R["corridas"]] == ["excepcion", "completed"]
    assert "se rompió" in R["corridas"][0]["error"] and R["corridas"][0]["fases"]       # las fases de lo que llegó a pasar


def test_R4_main_sin_intento_de_tabm_sale_con_3_y_lo_dice(monkeypatch, tmp_path):
    codigo, R, visto, _ = _correr_main(monkeypatch, tmp_path, [(0, 0, "baseline")])
    assert codigo == 3 and "TabM no tiene intento" in R["error"] and R["corridas"] == []
    assert S.veredicto_tabm_por_fases(3, R) == "sin_tabm"
