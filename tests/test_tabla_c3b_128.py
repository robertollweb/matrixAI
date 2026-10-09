"""128-C3b: la tabla para recalibrar la guarda de TabM. La fórmula «de hoy» está COPIADA en la tabla a propósito (enseña
lo que decide hoy el Studio): esta prueba la compara con la del Studio, para que si cambia allí la tabla no mienta."""
import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _por_ruta(nombre, ruta):
    """Por RUTA (sin tocar `sys.path`); el módulo va en `sys.modules` solo mientras se ejecuta: `dataclass` lo busca allí."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = m
    try:
        spec.loader.exec_module(m)
    finally:
        del sys.modules[nombre]
    return m


T = _por_ruta("tabla_c3b_128_bajo_prueba", RAIZ / "benchmarks" / "fase0" / "tabla_c3b_128.py")


def test_la_formula_de_hoy_es_la_del_studio():
    studio = _por_ruta("memoria_del_estudio_para_la_tabla",
                       RAIZ.parent / "matrixaistudio" / "studio-backend" / "matrixai_studio" / "memoria_del_estudio.py")
    for filas, columnas in ((60000, 170), (50000, 230), (188318, 130), (3279, 1558), (500, 12)):
        assert T.guarda_de_hoy_gb(filas, columnas) == studio.memoria_estimada_del_estudio_gb(filas, columnas)


def test_techo_en_mib_como_docker():
    assert T.techo_mb("6g") == 6144.0 and T.techo_mb("1150m") == 1150.0


def rec(pico, oom=0, estado="completed", filas=60000, columnas=170, memoria="6g"):
    return {"estado": estado, "memoria_pico_mb": pico, "memoria_del_contenedor": memoria, "memory_events": {"oom_kill": oom},
            "intentos_de_tabm": 5, "campeon": "lightgbm",
            "guarda_de_verdad": [{"filas": filas, "columnas_de_entrada": columnas,
                                  "habria_dicho_estimada_y_disponible_gb": [7.0, 6.0]}],
            "resumen_de_memoria": {"por_fase": {"intento:matrixai.dense.tabm_cpu": {"pico_hijo_mb": 900.0,
                                                                                    "pico_servidor_mb": 2100.0}}}}


def test_cabe_hasta_el_90_por_ciento_sin_oom():
    assert T.fila("a", rec(0.9 * 6144), None)["cabe"] is True
    assert T.fila("a", rec(0.9 * 6144 + 1), None)["cabe"] is False
    assert T.fila("a", rec(3000.0, oom=1), None)["cabe"] is False          # un hijo muerto por memoria: no cabe
    assert T.fila("a", rec(3000.0, estado="failed"), None)["cabe"] is False


def test_la_fila_lleva_lo_medido_y_lo_que_decidia_la_guarda_de_hoy():
    f = T.fila("APSFailure", rec(3500.0), {"memoria_pico_mb": 3346.0})
    assert (f["filas"], f["columnas"], f["celdas_M"]) == (60000, 170, 10.2)
    assert f["pico_hijo_tabm_mb"] == 900.0 and f["pico_servidor_en_tabm_mb"] == 2100.0 and f["pico_sin_tabm_mb"] == 3346.0
    assert f["guarda_de_hoy_gb"] == 7.01 and f["guarda_de_hoy_deja_entrar"] is False
    ancho = T.fila("Internet-Advertisements", rec(6019.0, filas=3279, columnas=1558), None)
    assert ancho["guarda_de_hoy_deja_entrar"] is True and ancho["cabe"] is False     # el caso que motivó el C3b


def test_la_tabla_solo_lee_los_registros_de_tabm_forzado_y_ordena_por_columnas():
    docs = [{"conjuntos": {"APSFailure:tabm_forzado": rec(3500.0),
                           "APSFailure:retencion": {"estado": "completed"},
                           "micro-mass:tabm_forzado": rec(3400.0, filas=571, columnas=1300)}}]
    t = T.tabla(docs, {"conjuntos": {"APSFailure": {"memoria_pico_mb": 3346.0}}})
    assert [f["nombre"] for f in t] == ["APSFailure", "micro-mass"]
    assert t[0]["pico_sin_tabm_mb"] == 3346.0 and t[1]["pico_sin_tabm_mb"] is None
