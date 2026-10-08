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
