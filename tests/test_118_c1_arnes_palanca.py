# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""118-C1 — el ARNÉS de la palanca (`pasada_118_palanca.py`), NO la palanca en
sí (que tiene sus propias pruebas en `matrixai-engines`,
`tests/test_118_c1_cuantiles.py`).

Nada aquí entrena nada caro: lo que se prueba es la ARITMÉTICA del veredicto
de `protocolo_118_v3.json` (`veredicto.por_palanca`) —el «campo» que sustituye
solo a la densa, el emparejamiento y bootstrap de la diferencia (palanca −
densa v2), y «ayuda si sube los cumplidos Y sus mejora superan a sus
inferioridad»— con cifras inventadas, mismo patrón que
`test_114_c6_arnes_ensamblado.py`. Lo único que toca el protocolo v3/v2 REAL
es barato: JSON, milisegundos, sin entrenar nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pasada_118_palanca as p118  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

RUTA_PROTOCOLO_V2 = _RAIZ / "benchmarks" / "fase0" / "protocolo_exploratorio_v2.json"
PROTOCOLO_V2 = protocolo_mod.ProtocoloExploratorio.cargar(RUTA_PROTOCOLO_V2)
RUTA_PROTOCOLO_V3 = _RAIZ / "benchmarks" / "fase0" / "protocolo_118_v3.json"


def _registro(dataset: str, motor: str, valor: float, *, repeticion: int = 0,
              pliegue: int = 0, estado: str = "completed", metrica: str = "auroc") -> dict:
    return {"dataset": dataset, "motor": motor, "repeticion": repeticion,
            "pliegue": pliegue, "estado": estado, metrica: valor}


# ---------------------------------------------------------------------------
# 1. LOS CONJUNTOS: los que la PROPIA palanca declara, nunca escritos a mano
# ---------------------------------------------------------------------------

def test_datasets_de_la_palanca_todos_los_no_sellados():
    protocolo_v3 = {"palancas": [{"id": "X", "conjuntos": "todos los NO sellados"}]}
    no_sellados = [d for d in p118.c5.datasets_de_la_pasada(PROTOCOLO_V2) if not d.sellado]
    datasets, texto = p118.datasets_de_la_palanca(protocolo_v3, "X", no_sellados)
    assert texto == "todos los NO sellados"
    assert datasets == no_sellados
    assert len(datasets) == 32


def test_datasets_de_la_palanca_118_c1_real_son_los_32_no_sellados():
    """Contra el protocolo v3 REAL: 118-C1.cuantiles pide 'todos los NO
    sellados', tal cual está registrado -- si alguien lo reescribe, esto se
    entera."""
    protocolo_v3 = p118._protocolo_v3()
    no_sellados = p118.c6.datasets_no_sellados(PROTOCOLO_V2)
    datasets, texto = p118.datasets_de_la_palanca(protocolo_v3, "118-C1.cuantiles", no_sellados)
    assert texto == "todos los NO sellados"
    assert len(datasets) == 32
    assert datasets == no_sellados


def test_datasets_de_la_palanca_solo_regresion():
    """118-C2 (todavía sin código de motor) declara este texto exacto en el
    protocolo v3 -- se prueba aquí porque el arnés ya sabe traducirlo, aunque
    el motor no lo aplique todavía."""
    protocolo_v3 = p118._protocolo_v3()
    definicion = next(p for p in protocolo_v3["palancas"] if p["id"] == "118-C2.objetivo")
    no_sellados = p118.c6.datasets_no_sellados(PROTOCOLO_V2)
    datasets, texto = p118.datasets_de_la_palanca(protocolo_v3, "118-C2.objetivo", no_sellados)
    assert texto == definicion["conjuntos"]
    assert datasets  # al menos una regresión entre los no sellados
    assert all(d.tarea == "regression" for d in datasets)
    assert len(datasets) < len(no_sellados)


def test_datasets_de_la_palanca_palanca_inexistente_para():
    with pytest.raises(SystemExit):
        p118.datasets_de_la_palanca({"palancas": []}, "no-existe", [])


def test_datasets_de_la_palanca_texto_de_conjuntos_desconocido_para():
    protocolo_v3 = {"palancas": [{"id": "X", "conjuntos": "un texto que nadie ha traducido"}]}
    with pytest.raises(SystemExit):
        p118.datasets_de_la_palanca(protocolo_v3, "X", [])


def test_datasets_de_la_palanca_sin_conjuntos_registrado_pendiente_para():
    """118-C4/C5: `parametros: null` y `registro_pendiente` -- pedir sus
    conjuntos antes de que existan tiene que fallar, no devolver una lista
    vacía en silencio."""
    protocolo_v3 = p118._protocolo_v3()
    with pytest.raises(SystemExit):
        p118.datasets_de_la_palanca(protocolo_v3, "118-C4.embeddings", [])


# ---------------------------------------------------------------------------
# 2. EL «CAMPO»: los motores de la v2, y la densa sustituida SOLO donde se
#    midió la palanca
# ---------------------------------------------------------------------------

def test_campo_sustituye_la_densa_solo_en_los_conjuntos_medidos():
    resultados_v2 = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", p118.NOMBRE_DENSA, 0.70),
        _registro("dsB", "lightgbm", 0.75), _registro("dsB", p118.NOMBRE_DENSA, 0.72),
    ]
    resultados_de_la_palanca = [_registro("dsA", p118.NOMBRE_DENSA, 0.90)]
    campo = p118.campo_de_la_comparacion(resultados_v2, resultados_de_la_palanca, {"dsA"})

    densa_dsA = [r for r in campo if r["dataset"] == "dsA" and r["motor"] == p118.NOMBRE_DENSA]
    densa_dsB = [r for r in campo if r["dataset"] == "dsB" and r["motor"] == p118.NOMBRE_DENSA]
    assert densa_dsA == [_registro("dsA", p118.NOMBRE_DENSA, 0.90)]  # la de la PALANCA
    assert densa_dsB == [_registro("dsB", p118.NOMBRE_DENSA, 0.72)]  # la de la v2, sin tocar
    # lightgbm no se toca en ningún conjunto.
    assert [r for r in campo if r["motor"] == "lightgbm"] == \
        [_registro("dsA", "lightgbm", 0.80), _registro("dsB", "lightgbm", 0.75)]
    assert len(campo) == 4  # ni se pierde ni se duplica nada


def test_cumplidos_se_acota_a_los_conjuntos_pedidos_no_a_todo_resultados():
    """Sin el filtro por dataset, `aplicar_regla_de_cierre` contaría TODOS los
    datasets presentes en `resultados` -- aquí hay tres, pero solo se pide
    veredicto sobre uno. `_cumplidos` tiene que acotar el denominador."""
    regla = PROTOCOLO_V2.regla_de_cierre
    resultados = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", p118.NOMBRE_DENSA, 0.79),
        # dos datasets AJENOS a la pregunta, con la densa perdiendo de calle:
        # si se colaran en el denominador, "cumplidos" bajaría.
        _registro("dsX", "lightgbm", 0.90), _registro("dsX", p118.NOMBRE_DENSA, 0.10),
        _registro("dsY", "lightgbm", 0.90), _registro("dsY", p118.NOMBRE_DENSA, 0.10),
    ]
    salida = p118._cumplidos(resultados, regla, metrica_por_dataset={"dsA": "auroc"},
                             nombres_de_los_conjuntos=["dsA"])
    assert salida == {"cumplidos": 1, "datasets": 1, "fraccion": 1.0, "cumple_la_regla": True}


# ---------------------------------------------------------------------------
# 3. EL VEREDICTO POR CONJUNTO: diferencia emparejada (palanca − densa v2)
# ---------------------------------------------------------------------------

def test_veredicto_del_conjunto_mejora():
    palanca = {(0, i): 0.85 for i in range(5)}
    v2 = {(0, i): 0.80 for i in range(5)}
    v = p118.veredicto_del_conjunto(dataset="d1", metric_id="auroc",
                                    palanca_por_pliegue=palanca, v2_densa_por_pliegue=v2)
    assert v["mejora"] is True
    assert v["inferioridad"] is False
    assert v["motivo"] is None
    assert v["intervalo"]["bajo"] == v["intervalo"]["alto"] == pytest.approx(0.05, abs=1e-9)
    assert v["pliegues_de_la_palanca_sin_intento_de_la_densa_v2"] == []


def test_veredicto_del_conjunto_inferioridad():
    palanca = {(0, i): 0.70 for i in range(5)}
    v2 = {(0, i): 0.80 for i in range(5)}
    v = p118.veredicto_del_conjunto(dataset="d2", metric_id="auroc",
                                    palanca_por_pliegue=palanca, v2_densa_por_pliegue=v2)
    assert v["mejora"] is False
    assert v["inferioridad"] is True
    assert v["intervalo"]["alto"] < 0.0


def test_veredicto_del_conjunto_intervalo_cruza_el_cero_ni_mejora_ni_inferioridad():
    palanca = {(0, 0): 0.85, (0, 1): 0.75, (0, 2): 0.86, (0, 3): 0.74, (0, 4): 0.80}
    v2 = {(0, i): 0.80 for i in range(5)}
    v = p118.veredicto_del_conjunto(dataset="d3", metric_id="auroc",
                                    palanca_por_pliegue=palanca, v2_densa_por_pliegue=v2)
    assert v["mejora"] is False
    assert v["inferioridad"] is False
    assert "no excluye el cero" in v["motivo"]


def test_veredicto_del_conjunto_menos_de_dos_comunes_no_mejora_ni_inferioridad():
    v = p118.veredicto_del_conjunto(dataset="d4", metric_id="auroc",
                                    palanca_por_pliegue={(0, 0): 0.90},
                                    v2_densa_por_pliegue={(0, 0): 0.80})
    assert v["mejora"] is False
    assert v["inferioridad"] is False
    assert v["intervalo"] is None
    assert "menos de dos pliegues comunes" in v["motivo"]


def test_veredicto_del_conjunto_declara_los_pliegues_sin_intento_v2_y_no_los_inventa():
    """La palanca midió (0,0)..(0,4); la densa v2 del artefacto SOLO tiene
    (0,0)-(0,2). Los dos que faltan se declaran y quedan FUERA del
    emparejamiento -- no se rellenan con nada."""
    palanca = {(0, i): 0.90 for i in range(5)}
    v2 = {(0, i): 0.80 for i in range(3)}
    v = p118.veredicto_del_conjunto(dataset="d5", metric_id="auroc",
                                    palanca_por_pliegue=palanca, v2_densa_por_pliegue=v2)
    assert v["n_pliegues_de_la_palanca"] == 5
    assert v["n_pliegues_de_la_densa_v2"] == 3
    faltan = v["pliegues_de_la_palanca_sin_intento_de_la_densa_v2"]
    assert sorted((f["repeticion"], f["pliegue"]) for f in faltan) == [(0, 3), (0, 4)]
    assert v["n_pliegues_comunes"] == 3
    assert v["mejora"] is True  # los 3 comunes SÍ alcanzan para remuestrear


# ---------------------------------------------------------------------------
# 4. EL VEREDICTO DE LA PALANCA: ayuda / no ayuda / mejora pero bajan
#    cumplidos
# ---------------------------------------------------------------------------

def _cumplidos_de(n: int) -> dict:
    return {"cumplidos": n, "datasets": 5, "fraccion": n / 5, "cumple_la_regla": n / 5 >= 0.8}


def test_veredicto_de_la_palanca_ayuda():
    veredictos = [{"dataset": f"d{i}", "mejora": i < 3, "inferioridad": i == 3}
                  for i in range(5)]  # 3 mejora, 1 inferioridad
    v = p118.veredicto_de_la_palanca(
        veredictos, cumplidos_con_la_palanca=_cumplidos_de(5),
        cumplidos_de_la_densa_v2=_cumplidos_de(3))
    assert v["sube_los_cumplidos"] is True
    assert v["n_mejora"] == 3
    assert v["n_inferioridad"] == 1
    assert v["mejoras_superan_inferioridades"] is True
    assert v["la_palanca_ayuda"] is True


def test_veredicto_de_la_palanca_no_ayuda_porque_no_sube_los_cumplidos():
    veredictos = [{"dataset": f"d{i}", "mejora": True, "inferioridad": False} for i in range(5)]
    v = p118.veredicto_de_la_palanca(
        veredictos, cumplidos_con_la_palanca=_cumplidos_de(3),
        cumplidos_de_la_densa_v2=_cumplidos_de(3))  # empate: NO sube
    assert v["mejoras_superan_inferioridades"] is True  # el lado bueno solo, a propósito
    assert v["sube_los_cumplidos"] is False
    assert v["la_palanca_ayuda"] is False


def test_veredicto_de_la_palanca_mejora_pero_bajan_cumplidos_no_ayuda():
    """El caso que pide el encargo explícitamente: la diferencia emparejada
    favorece a la palanca en la mayoría de los conjuntos (4 mejora, 1
    inferioridad) pero el LISTÓN de la cartera (a 2 puntos del mejor) lo
    cumplen MENOS conjuntos que con la densa v2 -- "ayuda" exige las DOS
    cosas a la vez, y aquí solo se da una."""
    veredictos = [{"dataset": f"d{i}", "mejora": i < 4, "inferioridad": i == 4}
                  for i in range(5)]  # 4 mejora, 1 inferioridad
    v = p118.veredicto_de_la_palanca(
        veredictos, cumplidos_con_la_palanca=_cumplidos_de(2),
        cumplidos_de_la_densa_v2=_cumplidos_de(4))  # bajan de 4 a 2
    assert v["mejoras_superan_inferioridades"] is True
    assert v["sube_los_cumplidos"] is False
    assert v["la_palanca_ayuda"] is False
