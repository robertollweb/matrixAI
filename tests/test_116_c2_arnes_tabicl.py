# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C2 — el ARNÉS de `pasada_116c2_tabicl.py`, NO `tabicl.v2` en sí (que
tiene sus propias pruebas en `matrixai-engines/tests/test_116_c1_tabicl.py`,
y se salta las que necesitan la biblioteca porque este host no la tiene —
a propósito, terreno de este programa).

Todo aquí corre en el HOST, SIN `tabicl` instalado: el régimen es una
función PURA de la partición (importa el TÉCHO de `matrixai_engines.
motores.tabicl`, que no necesita la biblioteca para resolverse — ver su
docstring de módulo), y el resto de la aritmética (veredicto pareado, 60%,
regla de la cartera) trabaja sobre diccionarios de registros inventados o
sobre el protocolo v2 real (JSON + ARFF pequeños, milisegundos, sin
entrenar nada) — mismo patrón que `tests/test_114_c6_arnes_ensamblado.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pytest  # noqa: E402

import pasada_116c2_tabicl as c2  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402
from matrixai_engines.motores.tabicl import (  # noqa: E402
    MAX_COLUMNAS_EN_CPU, MAX_FILAS_DE_CONTEXTO_EN_CPU)

RUTA_PROTOCOLO_V2 = _RAIZ / "benchmarks" / "fase0" / "protocolo_exploratorio_v2.json"
PROTOCOLO_V2 = protocolo_mod.ProtocoloExploratorio.cargar(RUTA_PROTOCOLO_V2)


def _registro(dataset: str, motor: str, valor: float, *, repeticion: int = 0,
              pliegue: int = 0, estado: str = "completed", metrica: str = "auroc") -> dict:
    return {"dataset": dataset, "motor": motor, "repeticion": repeticion,
            "pliegue": pliegue, "estado": estado, metrica: valor}


# ---------------------------------------------------------------------------
# 1. EL RÉGIMEN — POR REGLA: filas de contexto y columnas, pliegue a pliegue
# ---------------------------------------------------------------------------

def test_rechazos_de_regimen_vacio_cuando_todo_cabe():
    folds = [(0, 0, 500), (0, 1, 1000), (1, 0, 2000)]
    assert c2.rechazos_de_regimen(folds, n_columnas=32) == []


def test_rechazos_de_regimen_en_el_borde_exacto_SI_cabe():
    """`MAX_FILAS_DE_CONTEXTO_EN_CPU` filas y `MAX_COLUMNAS_EN_CPU` columnas
    EXACTAS no se rechazan — la comparación es `>`, no `>=` (registro de
    C1: «Inclusive en el borde: 2.000 filas y 32 columnas SÍ corren»)."""
    folds = [(0, 0, MAX_FILAS_DE_CONTEXTO_EN_CPU)]
    assert c2.rechazos_de_regimen(folds, n_columnas=MAX_COLUMNAS_EN_CPU) == []


def test_rechazos_de_regimen_una_fila_de_mas_rechaza_SOLO_ese_pliegue():
    folds = [(0, 0, MAX_FILAS_DE_CONTEXTO_EN_CPU), (0, 1, MAX_FILAS_DE_CONTEXTO_EN_CPU + 1)]
    rechazos = c2.rechazos_de_regimen(folds, n_columnas=8)
    assert len(rechazos) == 1
    assert rechazos[0]["repeticion"] == 0 and rechazos[0]["pliegue"] == 1
    assert "filas de contexto" in rechazos[0]["motivo"]
    assert str(MAX_FILAS_DE_CONTEXTO_EN_CPU) in rechazos[0]["motivo"]


def test_rechazos_de_regimen_columnas_de_mas_rechaza_TODOS_los_pliegues():
    """Las columnas no varían por pliegue: si se pasan, rechazan CADA
    pliegue del conjunto, no solo uno."""
    folds = [(0, 0, 100), (0, 1, 200), (1, 0, 300)]
    rechazos = c2.rechazos_de_regimen(folds, n_columnas=MAX_COLUMNAS_EN_CPU + 1)
    assert len(rechazos) == 3
    assert all("columnas" in r["motivo"] for r in rechazos)


def test_rechazos_de_regimen_puede_rechazar_por_las_dos_razones_a_la_vez():
    folds = [(0, 0, MAX_FILAS_DE_CONTEXTO_EN_CPU + 1)]
    rechazos = c2.rechazos_de_regimen(folds, n_columnas=MAX_COLUMNAS_EN_CPU + 1)
    assert len(rechazos) == 1
    assert "filas de contexto" in rechazos[0]["motivo"]
    assert "columnas" in rechazos[0]["motivo"]


def test_regimen_del_dataset_dentro_cuando_TODOS_los_pliegues_aceptan():
    """`dresses-sales` (12 columnas, ~321 filas de entrenamiento por
    pliegue como mucho): dentro del régimen — medido, no supuesto."""
    todos = c2.c5.datasets_de_la_pasada(PROTOCOLO_V2)
    ds = next(d for d in todos if d.nombre == "dresses-sales")
    regimen, _base = c2.regimen_del_dataset(ds, PROTOCOLO_V2)
    assert regimen["en_regimen"] is True
    assert regimen["rechazos_por_pliegue"] == []
    assert regimen["n_columnas"] == 12


def test_regimen_del_dataset_fuera_cuando_UN_pliegue_rechaza_el_conjunto_entero():
    """`pc4` tiene 37 columnas (> 32): TODOS sus pliegues se rechazan por
    columnas, y el conjunto entero queda fuera — «el motor acepta en TODOS
    los pliegues», literal del registro de C2: un solo pliegue rechazado
    saca al conjunto entero, no solo ese pliegue."""
    todos = c2.c5.datasets_de_la_pasada(PROTOCOLO_V2)
    ds = next(d for d in todos if d.nombre == "pc4")
    regimen, _base = c2.regimen_del_dataset(ds, PROTOCOLO_V2)
    assert regimen["en_regimen"] is False
    assert regimen["n_columnas"] == 37
    assert len(regimen["rechazos_por_pliegue"]) == regimen["n_pliegues"]
    assert regimen["n_pliegues"] > 0


def test_motivo_fuera_de_regimen_dice_el_motivo_y_lo_que_pidio():
    regimen = {
        "n_pliegues": 2,
        "rechazos_por_pliegue": [
            {"repeticion": 0, "pliegue": 0, "motivo": "columnas: pide 37, tope 32"},
            {"repeticion": 0, "pliegue": 1, "motivo": "columnas: pide 37, tope 32"},
        ],
    }
    texto = c2.motivo_fuera_de_regimen(regimen)
    assert "2 de 2 pliegues" in texto
    assert "columnas: pide 37, tope 32" in texto


# ---------------------------------------------------------------------------
# 2. EL EMPAREJAMIENTO POR CLAVE (repeticion, pliegue) — no por orden
# ---------------------------------------------------------------------------

def test_diferencias_emparejadas_empareja_por_clave_no_por_orden_de_insercion():
    """Insertados en ORDEN DISTINTO a propósito: si el emparejamiento fuera
    posicional (p. ej. `zip(a.values(), b.values())`) en vez de por CLAVE,
    el resultado sería otro. `a` y `b` traen el MISMO valor en cada clave,
    así que emparejado por clave la diferencia es 0.0 en las dos."""
    a = {(0, 1): 10.0, (0, 0): 1.0}
    b = {(0, 0): 1.0, (0, 1): 10.0}
    diferencias, comunes = c2.diferencias_emparejadas(a, b)
    assert comunes == [(0, 0), (0, 1)]
    assert diferencias == [0.0, 0.0]


def test_diferencias_emparejadas_solo_las_claves_comunes():
    a = {(0, 0): 5.0, (0, 1): 6.0, (0, 2): 7.0}
    b = {(0, 0): 5.0, (0, 1): 4.0}
    diferencias, comunes = c2.diferencias_emparejadas(a, b)
    assert comunes == [(0, 0), (0, 1)]
    assert diferencias == [0.0, 2.0]


# ---------------------------------------------------------------------------
# 3. EL VEREDICTO POR CONJUNTO: mejora / dentro del margen / cruza el cero / fallo
# ---------------------------------------------------------------------------

def test_veredicto_del_conjunto_mejora():
    """Diferencia constante y grande (+0,05 en 5 pliegues): el intervalo de
    percentiles de una constante es un punto, excluye el cero por arriba y
    no cabe en ±0,005."""
    tabicl = {(0, i): 0.85 for i in range(5)}
    lightgbm = {(0, i): 0.80 for i in range(5)}
    v = c2.veredicto_del_conjunto(
        dataset="d1", tabicl_media=0.85, lightgbm_media=0.80,
        tabicl_por_pliegue=tabicl, lightgbm_por_pliegue=lightgbm, fallo_de_tabicl=None)
    assert v["mejora"] is True
    assert v["motivo"] is None
    assert v["intervalo"]["bajo"] == pytest.approx(0.05)
    assert v["intervalo"]["alto"] == pytest.approx(0.05)


def test_veredicto_del_conjunto_dentro_del_margen():
    """Diferencia constante pero diminuta (+0,002): excluye el cero por
    arriba pero cabe entera dentro de ±0,005 — no se nota."""
    tabicl = {(0, i): 0.802 for i in range(5)}
    lightgbm = {(0, i): 0.80 for i in range(5)}
    v = c2.veredicto_del_conjunto(
        dataset="d2", tabicl_media=0.802, lightgbm_media=0.80,
        tabicl_por_pliegue=tabicl, lightgbm_por_pliegue=lightgbm, fallo_de_tabicl=None)
    assert v["mejora"] is False
    assert "margen de equivalencia" in v["motivo"]


def test_veredicto_del_conjunto_cruza_el_cero():
    """Diferencias que alternan de signo: el intervalo cubre valores
    negativos y positivos, no excluye el cero por arriba."""
    tabicl = {(0, 0): 0.85, (0, 1): 0.75, (0, 2): 0.86, (0, 3): 0.74, (0, 4): 0.80}
    lightgbm = {(0, i): 0.80 for i in range(5)}
    v = c2.veredicto_del_conjunto(
        dataset="d3", tabicl_media=0.80, lightgbm_media=0.80,
        tabicl_por_pliegue=tabicl, lightgbm_por_pliegue=lightgbm, fallo_de_tabicl=None)
    assert v["mejora"] is False
    assert v["intervalo"]["bajo"] <= 0.0
    assert "no excluye el cero" in v["motivo"]


def test_veredicto_del_conjunto_fallo_de_tabicl_es_no_mejora():
    """«Un fallo de TabICL cuenta como no mejora» — literal del registro de
    C2. A diferencia de fuera-de-régimen y sin-lightgbm-emparejable (que NO
    llegan a `veredicto_del_conjunto`, ver `clasificar_conjunto`), esto SÍ
    produce un veredicto y SÍ cuenta en el 60%."""
    v = c2.veredicto_del_conjunto(
        dataset="d4", tabicl_media=None, lightgbm_media=0.80,
        tabicl_por_pliegue={}, lightgbm_por_pliegue={(0, i): 0.80 for i in range(5)},
        fallo_de_tabicl="rep=0 pliegue=2 estado=failed motivo=timeout")
    assert v["mejora"] is False
    assert v["intervalo"] is None
    assert "fallo de tabicl.v2" in v["motivo"]


def test_veredicto_del_conjunto_menos_de_dos_pliegues_comunes_no_mejora():
    v = c2.veredicto_del_conjunto(
        dataset="d5", tabicl_media=0.85, lightgbm_media=0.80,
        tabicl_por_pliegue={(0, 0): 0.85}, lightgbm_por_pliegue={(0, 0): 0.80},
        fallo_de_tabicl=None)
    assert v["mejora"] is False
    assert v["intervalo"] is None


# ---------------------------------------------------------------------------
# 4. CLASIFICAR EL CONJUNTO: régimen -> lightgbm emparejable -> veredicto
# ---------------------------------------------------------------------------

def _regimen_dentro(n_pliegues: int = 5) -> dict:
    return {"dataset": "dX", "n_columnas": 10, "max_filas_de_contexto": 300,
            "n_pliegues": n_pliegues, "n_test": 50, "en_regimen": True,
            "rechazos_por_pliegue": []}


def _regimen_fuera() -> dict:
    return {"dataset": "dX", "n_columnas": 40, "max_filas_de_contexto": 300, "n_pliegues": 5,
            "n_test": 50, "en_regimen": False,
            "rechazos_por_pliegue": [{"repeticion": 0, "pliegue": i,
                                      "motivo": "columnas: pide 40, tope 32"}
                                     for i in range(5)]}


def test_clasificar_conjunto_fuera_de_regimen_no_produce_veredicto():
    clasificacion = c2.clasificar_conjunto("dX", _regimen_fuera(), registros_tabicl=[],
                                           resultados_v2=[], metric_id="auroc")
    assert clasificacion["categoria"] == "fuera_de_regimen"
    assert clasificacion["medible"] is False
    assert clasificacion["veredicto"] is None
    assert "columnas" in clasificacion["motivo"]


def test_clasificar_conjunto_medible_cuando_lightgbm_cubre_todos_los_pliegues_de_tabicl():
    registros_tabicl = [_registro("dX", c2.NOMBRE_TABICL, 0.80 + 0.01 * i, pliegue=i)
                        for i in range(5)]
    resultados_v2 = [_registro("dX", c2.NOMBRE_LIGHTGBM, 0.75, pliegue=i) for i in range(5)]
    clasificacion = c2.clasificar_conjunto("dX", _regimen_dentro(), registros_tabicl,
                                           resultados_v2, metric_id="auroc")
    assert clasificacion["categoria"] == "medible"
    assert clasificacion["medible"] is True
    assert clasificacion["veredicto"] is not None
    assert clasificacion["veredicto"]["mejora"] is True  # tabicl >= 0.80 > lightgbm 0.75 siempre


def test_clasificar_conjunto_sin_lightgbm_emparejable_cuando_falta_UN_pliegue_de_lightgbm():
    """`tabicl.v2` completa los 5 pliegues; lightgbm (leído de la v2) solo
    tiene 4 — «si falta un pliegue suyo, dilo, ese conjunto fuera», literal
    del registro de C2. El conjunto entero queda `sin_lightgbm_emparejable`,
    no una comparación con 4 pliegues en vez de 5."""
    registros_tabicl = [_registro("dX", c2.NOMBRE_TABICL, 0.80, pliegue=i) for i in range(5)]
    resultados_v2 = [_registro("dX", c2.NOMBRE_LIGHTGBM, 0.75, pliegue=i) for i in range(4)]
    clasificacion = c2.clasificar_conjunto("dX", _regimen_dentro(), registros_tabicl,
                                           resultados_v2, metric_id="auroc")
    assert clasificacion["categoria"] == "sin_lightgbm_emparejable"
    assert clasificacion["medible"] is False
    assert clasificacion["veredicto"] is None
    assert "4" in clasificacion["motivo"] and "5" in clasificacion["motivo"]


def test_clasificar_conjunto_lightgbm_de_otro_motor_no_cuenta():
    """`por_pliegue` filtra por `motor == "lightgbm"`: un registro de OTRO
    motor de la v2 (p. ej. `sklearn.hgb`) en los mismos pliegues no basta
    para emparejar — «lightgbm sale de la v2» significa ESE motor, no
    cualquiera del artefacto."""
    registros_tabicl = [_registro("dX", c2.NOMBRE_TABICL, 0.80, pliegue=i) for i in range(5)]
    resultados_v2 = [_registro("dX", "sklearn.hgb", 0.75, pliegue=i) for i in range(5)]
    clasificacion = c2.clasificar_conjunto("dX", _regimen_dentro(), registros_tabicl,
                                           resultados_v2, metric_id="auroc")
    assert clasificacion["categoria"] == "sin_lightgbm_emparejable"


def test_clasificar_conjunto_lightgbm_fallido_no_cuenta_como_pliegue_disponible():
    """Un registro de lightgbm con `estado="failed"` no cuenta como medida
    (`ESTADOS_QUE_CUENTAN_COMO_MEDIDA`): es como si faltara ese pliegue."""
    registros_tabicl = [_registro("dX", c2.NOMBRE_TABICL, 0.80, pliegue=i) for i in range(5)]
    resultados_v2 = [_registro("dX", c2.NOMBRE_LIGHTGBM, 0.75, pliegue=i) for i in range(4)]
    resultados_v2.append(_registro("dX", c2.NOMBRE_LIGHTGBM, None, pliegue=4, estado="failed"))
    clasificacion = c2.clasificar_conjunto("dX", _regimen_dentro(), registros_tabicl,
                                           resultados_v2, metric_id="auroc")
    assert clasificacion["categoria"] == "sin_lightgbm_emparejable"


def test_clasificar_conjunto_fallo_de_tabicl_es_medible_con_no_mejora():
    registros_tabicl = [_registro("dX", c2.NOMBRE_TABICL, 0.80, pliegue=i) for i in range(4)]
    registros_tabicl.append({"dataset": "dX", "motor": c2.NOMBRE_TABICL, "repeticion": 0,
                             "pliegue": 4, "estado": "failed", "motivo": "timeout"})
    resultados_v2 = [_registro("dX", c2.NOMBRE_LIGHTGBM, 0.75, pliegue=i) for i in range(5)]
    clasificacion = c2.clasificar_conjunto("dX", _regimen_dentro(), registros_tabicl,
                                           resultados_v2, metric_id="auroc")
    assert clasificacion["categoria"] == "medible"
    assert clasificacion["medible"] is True
    assert clasificacion["veredicto"]["mejora"] is False
    assert "fallo de tabicl.v2" in clasificacion["veredicto"]["motivo"]


# ---------------------------------------------------------------------------
# 5. EL CRITERIO DEL CORTE: mejora en >= 60% de los conjuntos MEDIBLES
# ---------------------------------------------------------------------------

def _clasificacion_medible(dataset: str, mejora: bool) -> dict:
    return {"dataset": dataset, "categoria": "medible", "medible": True, "motivo": None,
            "veredicto": {"mejora": mejora}}


def test_veredicto_del_corte_60_por_ciento():
    clasificaciones = [_clasificacion_medible(f"d{i}", i < 6) for i in range(10)]
    v = c2.veredicto_del_corte(clasificaciones)
    assert v == {
        "n_conjuntos": 10, "n_mejora": 6, "fraccion_mejora": 0.6,
        "minimo_exigido": 0.60, "cumple": True,
        "conjuntos_que_mejoran": [f"d{i}" for i in range(6)],
    }


def test_veredicto_del_corte_justo_por_debajo_no_cumple():
    clasificaciones = [_clasificacion_medible(f"d{i}", i < 5) for i in range(9)]  # 5/9
    v = c2.veredicto_del_corte(clasificaciones)
    assert v["cumple"] is False
    assert v["fraccion_mejora"] < 0.60


def test_veredicto_del_corte_sin_conjuntos_medibles_no_cumple():
    assert c2.veredicto_del_corte([])["cumple"] is False


def test_veredicto_del_corte_solo_mira_los_medibles():
    """Las clasificaciones `fuera_de_regimen`/`sin_lightgbm_emparejable` NO
    se le pasan a `veredicto_del_corte` (el llamante ya filtra por
    `medible` antes, ver `main`): si se colaran, moverían el denominador
    sin que ningún dato nuevo lo justificara."""
    medibles = [_clasificacion_medible("d1", True), _clasificacion_medible("d2", True)]
    v = c2.veredicto_del_corte(medibles)
    assert v["n_conjuntos"] == 2
    assert v["fraccion_mejora"] == 1.0


# ---------------------------------------------------------------------------
# 6. LA REGLA DE LA CARTERA, CON TABICL DENTRO DEL CAMPO
# ---------------------------------------------------------------------------

def test_regla_de_la_cartera_con_tabicl_evalua_a_tabicl_y_a_cada_aprobado(monkeypatch):
    """Escenario compuesto a mano, con la cartera de HOY sustituida por
    ["lightgbm", "sklearn.hgb"] (el mismo patrón de sustitución que
    `test_114_c6_arnes_ensamblado.py` usa sobre `CARTERA_APROBADA`, aquí
    sobre la función ya resuelta): en `dsA` tabicl.v2 es el mejor con
    holgura (0.90 contra 0.80/0.75), así que tabicl.v2 CUMPLE y sklearn.hgb
    (0.75, a 15 puntos del nuevo mejor) DEJA de cumplir al entrar tabicl.v2
    en el campo."""
    monkeypatch.setattr(c2, "nombres_de_los_miembros_aprobados",
                        lambda: ("lightgbm", "sklearn.hgb"))
    resultados_v2 = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", "sklearn.hgb", 0.79),
    ]
    resultados_tabicl = [_registro("dsA", c2.NOMBRE_TABICL, 0.90)]
    salida = c2.regla_de_la_cartera_con_tabicl(
        resultados_tabicl, resultados_v2, regla=PROTOCOLO_V2.regla_de_cierre,
        metrica_por_dataset={"dsA": "auroc"}, nombres_del_regimen=["dsA"])

    assert salida[c2.NOMBRE_TABICL]["cumple_la_regla"] is True
    assert salida[c2.NOMBRE_TABICL]["cumplidos"] == 1

    assert salida["lightgbm"]["antes"]["cumple_la_regla"] is True  # mejor de los miembros (antes)
    # despues: el mejor pasa a ser tabicl.v2 (0.90); lightgbm queda a (0.90-0.80)*100 = 10
    # puntos, por encima del liston de la regla (2.0) -> deja de cumplir.
    assert salida["lightgbm"]["despues"]["cumple_la_regla"] is False
    assert salida["lightgbm"]["diferencia_cumplidos"] == -1

    assert salida["sklearn.hgb"]["despues"]["cumple_la_regla"] is False


def test_regla_de_la_cartera_con_tabicl_restringe_al_regimen(monkeypatch):
    """Un dataset FUERA del régimen (no está en `nombres_del_regimen`) no
    entra ni en `v2_en_regimen` ni en `campo_con_tabicl`: no puede mover el
    veredicto de nadie."""
    monkeypatch.setattr(c2, "nombres_de_los_miembros_aprobados", lambda: ("lightgbm",))
    resultados_v2 = [
        _registro("dsA", "lightgbm", 0.80),
        _registro("dsFUERA", "lightgbm", 0.10),  # si esto colara, cambiaria "mejor"
    ]
    resultados_tabicl = [_registro("dsA", c2.NOMBRE_TABICL, 0.81)]
    salida = c2.regla_de_la_cartera_con_tabicl(
        resultados_tabicl, resultados_v2, regla=PROTOCOLO_V2.regla_de_cierre,
        metrica_por_dataset={"dsA": "auroc", "dsFUERA": "auroc"},
        nombres_del_regimen=["dsA"])
    assert salida["lightgbm"]["antes"]["datasets"] == 1
    assert salida["lightgbm"]["despues"]["datasets"] == 1


# ---------------------------------------------------------------------------
# 7. EL RÉGIMEN ESTIMADO (--estimar): aritmética sobre `particion_por_dataset`
#    de la v2, SIN leer ni particionar el ARFF — pedido por el supervisor el
#    2026-09-25 tras medir que la vía exacta (`regimen_del_dataset`) sobre
#    los 32 no sellados enteros pasaba de 35 min de CPU sin terminar.
# ---------------------------------------------------------------------------

def test_regimen_estimado_desde_metadatos_aritmetica_conocida():
    """1.000 filas con objetivo, 200 de prueba -> reserva de desarrollo 800;
    5 pliegues -> cada entrenamiento es 4/5 de la reserva = 640, EXACTO (sin
    redondeo que noten)."""
    meta = {"n_filas_con_objetivo": 1000, "n_test": 200, "n_predictores": 10,
            "folds_obtenidos": 5, "n_pliegues_obtenidos": 5}
    regimen = c2.regimen_estimado_desde_metadatos("dX", meta)
    assert regimen["max_filas_de_contexto"] == 640
    assert regimen["n_columnas"] == 10
    assert regimen["en_regimen"] is True
    assert regimen["estimado"] is True


def test_regimen_estimado_desde_metadatos_redondea_hacia_arriba():
    """801 filas de reserva entre 5 pliegues: 801*4/5 = 640,8 -> 641, no 640
    (`math.ceil`, nunca hacia abajo: un techo que subestima el contexto
    podria meter dentro del regimen un conjunto que en la particion real no
    cabe)."""
    meta = {"n_filas_con_objetivo": 1001, "n_test": 200, "n_predictores": 10,
            "folds_obtenidos": 5, "n_pliegues_obtenidos": 5}
    regimen = c2.regimen_estimado_desde_metadatos("dX", meta)
    assert regimen["max_filas_de_contexto"] == 641


def test_regimen_estimado_desde_metadatos_fuera_por_columnas():
    meta = {"n_filas_con_objetivo": 1000, "n_test": 200,
            "n_predictores": MAX_COLUMNAS_EN_CPU + 1, "folds_obtenidos": 5,
            "n_pliegues_obtenidos": 5}
    regimen = c2.regimen_estimado_desde_metadatos("dX", meta)
    assert regimen["en_regimen"] is False
    assert len(regimen["rechazos_por_pliegue"]) == 5


def test_regimen_estimado_desde_metadatos_fuera_por_filas():
    meta = {"n_filas_con_objetivo": 20000, "n_test": 4000, "n_predictores": 10,
            "folds_obtenidos": 5, "n_pliegues_obtenidos": 5}
    regimen = c2.regimen_estimado_desde_metadatos("dX", meta)
    assert regimen["en_regimen"] is False


def test_regimen_estimado_coincide_con_el_exacto_para_los_conjuntos_reales():
    """Contra el protocolo v2 real: el régimen ESTIMADO (metadatos de
    `pasada_v2_113_resultado.json`, sin leer ARFF) y el régimen EXACTO
    (`regimen_del_dataset`, que SÍ lee y particiona) tienen que coincidir en
    `en_regimen` para los conjuntos que de verdad importan — los ocho que
    hoy caen dentro, y uno claramente fuera por columnas (pc4). No se compara
    el número exacto de filas (la estimación es una aproximación declarada,
    ver su docstring), solo la CLASIFICACIÓN, que es lo único de lo que
    depende el corte."""
    metadatos_v2 = c2.cargar_particion_por_dataset_v2()
    todos = c2.c5.datasets_de_la_pasada(PROTOCOLO_V2)
    por_nombre = {d.nombre: d for d in todos}
    for nombre in ("dresses-sales", "kc2", "climate-model-simulation-crashes",
                   "balance-scale", "diabetes", "pc1", "Moneyball", "yeast", "pc4"):
        exacto, _base = c2.regimen_del_dataset(por_nombre[nombre], PROTOCOLO_V2)
        estimado = c2.regimen_estimado_desde_metadatos(nombre, metadatos_v2[nombre])
        assert estimado["en_regimen"] == exacto["en_regimen"], (
            f"{nombre}: estimado={estimado['en_regimen']} exacto={exacto['en_regimen']}")


def test_cargar_particion_por_dataset_v2_trae_los_40_datasets():
    metadatos_v2 = c2.cargar_particion_por_dataset_v2()
    assert len(metadatos_v2) == 40
    assert "dresses-sales" in metadatos_v2
    assert metadatos_v2["dresses-sales"]["n_predictores"] == 12
