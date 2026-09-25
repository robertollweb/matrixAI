# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C6 — el ARNÉS del ensamblado (`pasada_114c6_ensamblado.py`), NO el
ensamblado en sí (que tiene sus propias pruebas en `matrixai-engines`).

Nada aquí entrena nada caro: lo que se prueba es la ARITMÉTICA del registro
de C6 —el oráculo, el bootstrap de la diferencia emparejada, el criterio del
60%, y la regla de la cartera antes/después del ensamblado— con cifras
inventadas y, donde toca, con el protocolo v2 real (que es barato: JSON,
milisegundos, sin entrenar nada).
"""
from __future__ import annotations

import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pasada_114c6_ensamblado as c6  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402
from matrixai_engines import cartera as _cartera_mod  # noqa: E402
from matrixai_engines.cartera import (  # noqa: E402
    AlcanceDeLaMedicion, MotorSoportado, RangoDeVersiones)

RUTA_PROTOCOLO_V2 = _RAIZ / "benchmarks" / "fase0" / "protocolo_exploratorio_v2.json"
PROTOCOLO_V2 = protocolo_mod.ProtocoloExploratorio.cargar(RUTA_PROTOCOLO_V2)


def _registro(dataset: str, motor: str, valor: float, *, repeticion: int = 0,
              pliegue: int = 0, estado: str = "completed", metrica: str = "auroc") -> dict:
    return {"dataset": dataset, "motor": motor, "repeticion": repeticion,
            "pliegue": pliegue, "estado": estado, metrica: valor}


def _entrada_de_cartera(motor: str) -> MotorSoportado:
    """Una entrada de cartera cualquiera para `motor`, con un alcance mínimo
    y coherente — mismo patrón que `_entrada()` de
    `matrixai-engines/tests/test_c102_c1_cartera.py`: estos tests miran el
    MECANISMO (que los miembros de la pasada salgan de la cartera), no el
    alcance real de ningún motor concreto."""
    return MotorSoportado(
        motor=motor,
        versiones_del_motor=RangoDeVersiones(minima="1.0.0", maxima_excluida="2.0.0"),
        biblioteca=None, versiones_de_la_biblioteca=None,
        aprobado_en="2026-01-01", evidencia="prueba", evidencia_digest="0" * 64,
        alcance=AlcanceDeLaMedicion(
            protocolo="X.v1", protocolo_digest="0" * 64,
            motores_del_protocolo=(motor,), motores_que_corrieron=(motor,),
            motores_que_no_compitieron=(),
            datasets_del_protocolo=1, datasets_que_corrieron=1,
            configuraciones_del_protocolo=1, configuraciones_que_corrieron=1,
            tareas_sin_medir=(), cubos_sin_medir=(),
            aciertos_por_ser_el_mejor=0, datasets_cumplidos=1,
            datasets_evaluados=1, margen_en_datasets=0))


# ---------------------------------------------------------------------------
# 1. SOLO ENTRAN LOS NO SELLADOS
# ---------------------------------------------------------------------------

def test_datasets_no_sellados_excluye_los_ocho_sellados():
    """Los 8 sellados del protocolo v2 (kr-vs-kp, letter, splice, pol, pc3,
    banknote-authentication, Satellite, diamonds) no entran nunca por
    omisión: 40 en el protocolo, 32 sin sellar."""
    todos = c6.c5.datasets_de_la_pasada(PROTOCOLO_V2)
    no_sellados = [d for d in todos if not d.sellado]
    assert len(todos) == 40
    assert len(no_sellados) == 32
    nombres_sellados = {"kr-vs-kp", "letter", "splice", "pol", "pc3",
                        "banknote-authentication", "Satellite", "diamonds"}
    assert {d.nombre for d in no_sellados}.isdisjoint(nombres_sellados)
    assert nombres_sellados == {d.nombre for d in todos if d.sellado}


def test_dresses_sales_es_el_no_sellado_mas_pequeno():
    """El conjunto que usa la prueba real mínima del corte — verificado
    contra el protocolo, no supuesto."""
    todos = c6.c5.datasets_de_la_pasada(PROTOCOLO_V2)
    no_sellados = [d for d in todos if not d.sellado]
    mas_pequeno = min(no_sellados, key=lambda d: d.n_filas_declaradas)
    assert mas_pequeno.nombre == "dresses-sales"


# ---------------------------------------------------------------------------
# 2. LOS MIEMBROS SALEN DE LA CARTERA, NUNCA ESCRITOS A MANO
# ---------------------------------------------------------------------------

def test_los_miembros_de_la_pasada_son_los_de_la_cartera_aprobada():
    """Hoy la cartera aprueba lightgbm y sklearn.hgb (113-C5) — leído, no
    tecleado en el guion."""
    nombres = c6.nombres_de_los_motores_de_la_pasada()
    assert nombres == ("lightgbm", "sklearn.hgb", c6.NOMBRE_ENSAMBLADO)


def test_un_cambio_en_la_cartera_cambia_los_miembros_de_la_pasada(monkeypatch):
    """Si la cartera aprobara solo UN motor mañana, esta pasada correría con
    ese motor y el ensamblado, sin tocar `pasada_114c6_ensamblado.py` — el
    mismo patrón de sustitución que usa `test_c102_c1_cartera.py` sobre
    `CARTERA_APROBADA`."""
    monkeypatch.setattr(_cartera_mod, "CARTERA_APROBADA", (_entrada_de_cartera("lightgbm"),))
    nombres = c6.nombres_de_los_motores_de_la_pasada()
    assert nombres == ("lightgbm", c6.NOMBRE_ENSAMBLADO)


def test_dos_entradas_del_mismo_motor_no_lo_duplican(monkeypatch):
    """La cartera puede tener dos entradas del mismo motor (dos rangos de
    version aprobados por separado) — no por eso aparece dos veces."""
    monkeypatch.setattr(_cartera_mod, "CARTERA_APROBADA",
                        (_entrada_de_cartera("lightgbm"), _entrada_de_cartera("lightgbm")))
    nombres = c6.nombres_de_los_motores_de_la_pasada()
    assert nombres == ("lightgbm", c6.NOMBRE_ENSAMBLADO)


# ---------------------------------------------------------------------------
# 3. EL ORÁCULO ELIGE EL MEJOR MIEMBRO
# ---------------------------------------------------------------------------

def test_elegir_oraculo_es_el_de_mejor_media():
    nombre, media = c6.elegir_oraculo({"lightgbm": 0.80, "sklearn.hgb": 0.85})
    assert (nombre, media) == ("sklearn.hgb", 0.85)


def test_elegir_oraculo_sin_miembros_medidos_no_inventa_uno():
    assert c6.elegir_oraculo({}) == (None, None)


def test_elegir_oraculo_con_un_solo_miembro():
    assert c6.elegir_oraculo({"lightgbm": 0.71}) == ("lightgbm", 0.71)


# ---------------------------------------------------------------------------
# 4. EL VEREDICTO POR CONJUNTO: los cuatro casos, con cifras inventadas
# ---------------------------------------------------------------------------

def test_veredicto_del_conjunto_mejora():
    """Diferencia constante y grande (+0,05 en los 5 pliegues): el intervalo
    de percentiles de una constante es un punto, ese punto excluye el cero
    por arriba y no cabe en ±0,005."""
    ensamblado = {(0, i): 0.85 for i in range(5)}
    oraculo = {(0, i): 0.80 for i in range(5)}
    v = c6.veredicto_del_conjunto(
        dataset="d1", oraculo_motor="lightgbm", oraculo_media=0.80,
        ensamblado_media=0.85, ensamblado_por_pliegue=ensamblado,
        oraculo_por_pliegue=oraculo, fallo_del_ensamblado=None)
    assert v["mejora"] is True
    assert v["motivo"] is None
    assert v["intervalo"]["bajo"] == v["intervalo"]["alto"] == pytest_approx(0.05)


def test_veredicto_del_conjunto_dentro_del_margen():
    """Diferencia constante pero diminuta (+0,002): excluye el cero por
    arriba (es positiva) pero cabe entera dentro de ±0,005 — no se nota."""
    ensamblado = {(0, i): 0.802 for i in range(5)}
    oraculo = {(0, i): 0.80 for i in range(5)}
    v = c6.veredicto_del_conjunto(
        dataset="d2", oraculo_motor="lightgbm", oraculo_media=0.80,
        ensamblado_media=0.802, ensamblado_por_pliegue=ensamblado,
        oraculo_por_pliegue=oraculo, fallo_del_ensamblado=None)
    assert v["mejora"] is False
    assert "margen de equivalencia" in v["motivo"]


def test_veredicto_del_conjunto_el_intervalo_cruza_el_cero():
    """Diferencias que alternan de signo: el intervalo de percentiles cubre
    valores negativos y positivos, así que NO excluye el cero por arriba."""
    ensamblado = {(0, 0): 0.85, (0, 1): 0.75, (0, 2): 0.86, (0, 3): 0.74, (0, 4): 0.80}
    oraculo = {(0, i): 0.80 for i in range(5)}
    v = c6.veredicto_del_conjunto(
        dataset="d3", oraculo_motor="lightgbm", oraculo_media=0.80,
        ensamblado_media=0.80, ensamblado_por_pliegue=ensamblado,
        oraculo_por_pliegue=oraculo, fallo_del_ensamblado=None)
    assert v["mejora"] is False
    assert v["intervalo"]["bajo"] <= 0.0
    assert "no excluye el cero" in v["motivo"]


def test_veredicto_del_conjunto_fallo_del_ensamblado_es_no_mejora():
    v = c6.veredicto_del_conjunto(
        dataset="d4", oraculo_motor="lightgbm", oraculo_media=0.80,
        ensamblado_media=None, ensamblado_por_pliegue={},
        oraculo_por_pliegue={(0, i): 0.80 for i in range(5)},
        fallo_del_ensamblado="rep=0 pliegue=2 estado=failed motivo=timeout")
    assert v["mejora"] is False
    assert v["intervalo"] is None
    assert "fallo del ensamblado" in v["motivo"]


def test_veredicto_del_conjunto_menos_de_dos_pliegues_comunes_no_mejora():
    """Un solo pliegue común no da nada que remuestrear — no se inventa un
    intervalo de anchura cero."""
    v = c6.veredicto_del_conjunto(
        dataset="d5", oraculo_motor="lightgbm", oraculo_media=0.80,
        ensamblado_media=0.85, ensamblado_por_pliegue={(0, 0): 0.85},
        oraculo_por_pliegue={(0, 0): 0.80}, fallo_del_ensamblado=None)
    assert v["mejora"] is False
    assert v["intervalo"] is None


# ---------------------------------------------------------------------------
# 5. EL CRITERIO DEL CORTE: mejora en >= 60% de los conjuntos
# ---------------------------------------------------------------------------

def test_veredicto_del_corte_60_por_ciento():
    veredictos = [{"dataset": f"d{i}", "mejora": i < 6} for i in range(10)]  # 6/10
    v = c6.veredicto_del_corte(veredictos)
    assert v == {
        "n_conjuntos": 10, "n_mejora": 6, "fraccion_mejora": 0.6,
        "minimo_exigido": 0.60, "cumple": True,
        "conjuntos_que_mejoran": [f"d{i}" for i in range(6)],
    }


def test_veredicto_del_corte_justo_por_debajo_no_cumple():
    veredictos = [{"dataset": f"d{i}", "mejora": i < 5} for i in range(9)]  # 5/9 = 0,555...
    v = c6.veredicto_del_corte(veredictos)
    assert v["cumple"] is False
    assert v["fraccion_mejora"] < 0.60


def test_veredicto_del_corte_sin_conjuntos_no_cumple():
    assert c6.veredicto_del_corte([])["cumple"] is False


# ---------------------------------------------------------------------------
# 6. LA REGLA DE LA CARTERA, CON Y SIN EL ENSAMBLADO DENTRO
# ---------------------------------------------------------------------------

def test_regla_de_la_cartera_antes_y_despues_dice_que_le_pasa_a_cada_aprobado():
    """Escenario compuesto a mano: en dsA el ensamblado mejora poco (no
    cambia nada); en dsB el ensamblado mejora mucho y hace que sklearn.hgb
    —que en dsB ERA el mejor y cumplía por distancia 0— deje de cumplir al
    quedarse a más de 2 puntos del nuevo mejor. `lightgbm` no se mueve."""
    resultados = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", "sklearn.hgb", 0.75),
        _registro("dsA", c6.NOMBRE_ENSAMBLADO, 0.81),
        _registro("dsB", "lightgbm", 0.70), _registro("dsB", "sklearn.hgb", 0.85),
        _registro("dsB", c6.NOMBRE_ENSAMBLADO, 0.90),
    ]
    salida = c6.regla_de_la_cartera_antes_y_despues(
        resultados, regla=PROTOCOLO_V2.regla_de_cierre,
        metrica_por_dataset={"dsA": "auroc", "dsB": "auroc"},
        miembros=["lightgbm", "sklearn.hgb"], nombres_de_los_conjuntos=["dsA", "dsB"])

    assert salida["lightgbm"]["antes"]["cumplidos"] == 1
    assert salida["lightgbm"]["despues"]["cumplidos"] == 1
    assert salida["lightgbm"]["diferencia_cumplidos"] == 0

    assert salida["sklearn.hgb"]["antes"] == {
        "cumplidos": 1, "datasets": 2, "fraccion": 0.5, "cumple_la_regla": False}
    assert salida["sklearn.hgb"]["despues"] == {
        "cumplidos": 0, "datasets": 2, "fraccion": 0.0, "cumple_la_regla": False}
    assert salida["sklearn.hgb"]["diferencia_cumplidos"] == -1


def test_regla_de_la_cartera_el_ensamblado_no_compite_en_antes():
    """`antes` tiene que ser EXACTAMENTE la regla de siempre, sin que el
    ensamblado participe ni de lejos: si el ensamblado fuera el único con la
    métrica más alta pero `antes` lo incluyera, el mejor cambiaría."""
    resultados = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", "sklearn.hgb", 0.78),
        _registro("dsA", c6.NOMBRE_ENSAMBLADO, 0.99),  # muy por encima, y NO debe contar en "antes"
    ]
    salida = c6.regla_de_la_cartera_antes_y_despues(
        resultados, regla=PROTOCOLO_V2.regla_de_cierre,
        metrica_por_dataset={"dsA": "auroc"},
        miembros=["lightgbm", "sklearn.hgb"], nombres_de_los_conjuntos=["dsA"])
    # lightgbm es el mejor de los MIEMBROS en dsA (0.80 > 0.78): "antes" cumple.
    assert salida["lightgbm"]["antes"]["cumple_la_regla"] is True
    assert salida["lightgbm"]["antes"]["cumplidos"] == 1


# ---------------------------------------------------------------------------
# 7. LA DIRECCIÓN DE LA MÉTRICA: las tres de la v2 son "más es mejor"
# ---------------------------------------------------------------------------

def test_direccion_de_las_tres_metricas_de_cierre_de_la_v2():
    assert c6.direccion_de("auroc") is True
    assert c6.direccion_de("accuracy") is True
    assert c6.direccion_de("r2") is True


def test_direccion_de_una_metrica_desconocida_para_en_vez_de_adivinar():
    import pytest
    with pytest.raises(SystemExit):
        c6.direccion_de("rmse")


# ---------------------------------------------------------------------------
# Utilidad mínima: comparación aproximada sin depender de pytest.approx en
# el cuerpo de arriba (evita un import repetido en cada función).
# ---------------------------------------------------------------------------

def pytest_approx(valor: float, tol: float = 1e-9):
    import pytest
    return pytest.approx(valor, abs=tol)
