# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C3 — el ARNÉS de la pasada del motor nuevo (`pasada_119_c3.py`), NO el
motor en sí (que tiene sus propias pruebas en `matrixai-engines`,
`tests/test_119_c2_motor_densa_tabm.py`).

Mismo patrón que `test_118_c1_arnes_palanca.py`: nada aquí entrena nada
caro. Lo que se prueba es (1) que los 8 sellados se NIEGAN -- a diferencia
de `pasada_118_palanca.py`, que solo avisaba --, (2) que el caché no
reutiliza un intento `failed`, (3) que el digest del motor reacciona a un
cambio de fichero, y (4) la aritmética del veredicto (`veredicto_de_c3`)
con cifras fabricadas -- mismo patrón que `test_veredicto_de_la_palanca_*`
de 118, reutilizando `pasada_118_palanca.veredicto_de_la_palanca` por
debajo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pasada_119_c3 as p119  # noqa: E402
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

RUTA_PROTOCOLO_V2 = _RAIZ / "benchmarks" / "fase0" / "protocolo_exploratorio_v2.json"
PROTOCOLO_V2 = protocolo_mod.ProtocoloExploratorio.cargar(RUTA_PROTOCOLO_V2)


def _registro(dataset: str, motor: str, valor: float, *, repeticion: int = 0,
              pliegue: int = 0, estado: str = "completed", metrica: str = "auroc") -> dict:
    return {"dataset": dataset, "motor": motor, "repeticion": repeticion,
            "pliegue": pliegue, "estado": estado, metrica: valor}


# ---------------------------------------------------------------------------
# 1. LOS SELLADOS SE NIEGAN (a diferencia de 118, que solo avisaba)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _datasets():
    c6.preparar_protocolo_v2()
    protocolo = PROTOCOLO_V2
    todos = c5.datasets_de_la_pasada(protocolo)
    no_sellados = c6.datasets_no_sellados(protocolo)
    return todos, no_sellados


def test_sin_solo_son_los_32_no_sellados(_datasets):
    todos, no_sellados = _datasets
    datasets, subconjunto = p119.datasets_de_c3(todos, no_sellados, solo=None)
    assert len(datasets) == 32
    assert all(not d.sellado for d in datasets)
    assert datasets == no_sellados
    assert subconjunto is None


def test_solo_un_no_sellado_se_acepta(_datasets):
    todos, no_sellados = _datasets
    datasets, subconjunto = p119.datasets_de_c3(todos, no_sellados, solo="diabetes")
    assert [d.nombre for d in datasets] == ["diabetes"]
    assert subconjunto == ["diabetes"]


def test_solo_un_sellado_se_niega(_datasets):
    """El caso que 118 dejaba pasar avisando -- aquí PARA. La regla de
    subida del protocolo v4 reserva los sellados para C4, y este corte no
    los mide «solo como prueba»."""
    todos, no_sellados = _datasets
    sellados = [d.nombre for d in todos if d.sellado]
    assert sellados  # hay al menos uno, si no la prueba no prueba nada
    with pytest.raises(SystemExit, match="SELLADO"):
        p119.datasets_de_c3(todos, no_sellados, solo=sellados[0])


def test_solo_mezcla_sellado_y_no_sellado_tambien_se_niega(_datasets):
    todos, no_sellados = _datasets
    sellados = [d.nombre for d in todos if d.sellado]
    with pytest.raises(SystemExit, match="SELLADO"):
        p119.datasets_de_c3(todos, no_sellados, solo=f"diabetes,{sellados[0]}")


def test_solo_conjunto_desconocido_para(_datasets):
    todos, no_sellados = _datasets
    with pytest.raises(SystemExit, match="no están en el protocolo"):
        p119.datasets_de_c3(todos, no_sellados, solo="esto-no-existe-en-ningun-protocolo")


def test_los_32_no_sellados_coinciden_con_el_protocolo_v4():
    """protocolo_119_v4.json declara sus propios 40 datasets, con el MISMO
    campo `sellado` que la v2 (comprobado byte a byte al escribir el guion:
    los 8 mismos nombres). Esta prueba lo fija: si algún día divergen, algo
    tiene que chillar en vez de que la pasada mida contra el protocolo
    equivocado en silencio."""
    import json
    v4 = json.loads((_RAIZ / "benchmarks" / "fase0" / "protocolo_119_v4.json")
                    .read_text(encoding="utf-8"))
    sellados_v4 = sorted(d["nombre"] for d in v4["datasets"] if d["sellado"])
    sellados_v2 = sorted(d.nombre for d in c5.datasets_de_la_pasada(PROTOCOLO_V2) if d.sellado)
    assert sellados_v4 == sellados_v2
    assert len(sellados_v4) == 8


# ---------------------------------------------------------------------------
# 2. EL CACHÉ NO REUTILIZA UN `failed`
# ---------------------------------------------------------------------------

def test_reusable_c3_acepta_completed_con_digests_iguales():
    previo = {"entorno_digest": "e1", "motor_digest": "m1", "presupuesto_wall_s": 120.0,
              "estado": "completed"}
    assert p119._reusable_c3(previo, "e1", "m1", 120.0) is True


def test_reusable_c3_acepta_completed_budget_limited():
    previo = {"entorno_digest": "e1", "motor_digest": "m1", "presupuesto_wall_s": 120.0,
              "estado": "completed_budget_limited"}
    assert p119._reusable_c3(previo, "e1", "m1", 120.0) is True


def test_reusable_c3_rechaza_failed_aunque_los_digests_coincidan():
    """LA TRAMPA Nº1 de «ANTES DE RELANZAR UNA PASADA» (CLAUDE.md):
    `c3._reusable` por sí sola NO mira `estado`, así que un intento `failed`
    con los digests correctos se reutilizaría para siempre. `_reusable_c3`
    tiene que cerrar exactamente este hueco."""
    previo = {"entorno_digest": "e1", "motor_digest": "m1", "presupuesto_wall_s": 120.0,
              "estado": "failed"}
    # Confirma primero que la trampa es real: la función de C3 sola SÍ lo
    # reutilizaría -- si esto no fuera cierto, la prueba de abajo no
    # probaría nada.
    import pasada_exploratoria_101_c3 as c3
    assert c3._reusable(previo, "e1", "m1", 120.0) is True
    assert p119._reusable_c3(previo, "e1", "m1", 120.0) is False


def test_reusable_c3_rechaza_cancelled():
    previo = {"entorno_digest": "e1", "motor_digest": "m1", "presupuesto_wall_s": 120.0,
              "estado": "cancelled"}
    assert p119._reusable_c3(previo, "e1", "m1", 120.0) is False


def test_reusable_c3_rechaza_digest_de_entorno_distinto():
    previo = {"entorno_digest": "e1", "motor_digest": "m1", "presupuesto_wall_s": 120.0,
              "estado": "completed"}
    assert p119._reusable_c3(previo, "e2", "m1", 120.0) is False


def test_reusable_c3_ninguno_previo():
    assert p119._reusable_c3(None, "e1", "m1", 120.0) is False


# ---------------------------------------------------------------------------
# 3. EL DIGEST CAMBIA SI CAMBIA EL MOTOR
# ---------------------------------------------------------------------------

def test_digest_de_reacciona_al_contenido_de_los_ficheros(tmp_path):
    """`_digest_de` es el mecanismo que `_digest_motor_nuevo`/`_digest_entorno`
    usan por debajo. Se prueba con ficheros FABRICADOS (nunca tocando
    `motores/densa_tabm.py`, que el encargo prohíbe editar): mismo
    contenido -> mismo digest; contenido distinto -> digest distinto."""
    f1 = tmp_path / "a.py"
    f1.write_text("contenido version 1\n", encoding="utf-8")
    d1 = p119._digest_de((f1,))

    f1.write_text("contenido version 1\n", encoding="utf-8")  # sin cambiar
    assert p119._digest_de((f1,)) == d1

    f1.write_text("contenido version 2 -- una linea distinta\n", encoding="utf-8")
    d2 = p119._digest_de((f1,))
    assert d2 != d1


def test_digest_motor_nuevo_incluye_los_tres_ficheros_declarados():
    """El encargo pide explícitamente que el digest del motor incluya
    `densa_tabm.py`, `redes/tabm_plr.py` y `redes/preparacion_tabm.py` --
    se comprueba que los TRES están en la lista que `_digest_motor_nuevo`
    recorre, por nombre, sin tocar su contenido."""
    nombres = {p.name for p in p119._FICHEROS_DEL_MOTOR_NUEVO}
    assert nombres == {"densa_tabm.py", "tabm_plr.py", "preparacion_tabm.py"}
    for ruta in p119._FICHEROS_DEL_MOTOR_NUEVO:
        assert ruta.exists(), f"{ruta} no existe -- el digest no puede cubrir un fichero ausente"


def test_digest_motor_nuevo_es_estable_y_no_vacio():
    d1 = p119._digest_motor_nuevo()
    d2 = p119._digest_motor_nuevo()
    assert d1 == d2
    assert len(d1) == 16  # trunca a 16 hex, mismo criterio que c3._digest_fichero


def test_digest_entorno_incluye_el_protocolo_y_su_enmienda(tmp_path, monkeypatch):
    """El encargo pide que el digest de caché incluya «el protocolo y su
    enmienda». Se comprueba apuntando las constantes de ruta a DOS ficheros
    temporales (nunca al protocolo real) y viendo que `_digest_entorno`
    reacciona a su contenido -- si `RUTA_DEL_PROTOCOLO_V4`/`RUTA_DE_LA_
    ENMIENDA_1` no estuvieran en `_ficheros_compartidos()`, este cambio no
    movería nada."""
    protocolo_fake = tmp_path / "protocolo_fake.json"
    enmienda_fake = tmp_path / "enmienda_fake.json"
    protocolo_fake.write_text("{}", encoding="utf-8")
    enmienda_fake.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(p119, "RUTA_DEL_PROTOCOLO_V4", protocolo_fake)
    monkeypatch.setattr(p119, "RUTA_DE_LA_ENMIENDA_1", enmienda_fake)
    antes = p119._digest_entorno()

    protocolo_fake.write_text('{"version": 2}', encoding="utf-8")
    despues = p119._digest_entorno()
    assert despues != antes


# ---------------------------------------------------------------------------
# 4. EL VEREDICTO DE C3, SOBRE UN ARTEFACTO PEQUEÑO FABRICADO
# ---------------------------------------------------------------------------

def _cumplidos_de(n: int, de: int = 5) -> dict:
    return {"cumplidos": n, "datasets": de, "fraccion": n / de, "cumple_la_regla": n / de >= 0.8}


def test_veredicto_de_c3_sube():
    """3 mejoras, 1 inferioridad, y los cumplidos suben de 3 a 5 ->
    cumple_la_regla_de_subida."""
    veredictos = [{"dataset": f"d{i}", "mejora": i < 3, "inferioridad": i == 3}
                  for i in range(5)]
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(5),
                             cumplidos_de_la_densa_v2=_cumplidos_de(3))
    assert v["sube_los_cumplidos"] is True
    assert v["mejoras_superan_inferioridades"] is True
    assert v["n_mejora"] == 3
    assert v["n_inferioridad"] == 1
    assert v["cumple_la_regla_de_subida"] is True
    assert v["cumplidos_con_el_motor_nuevo"] == _cumplidos_de(5)
    assert v["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"] == _cumplidos_de(3)
    assert "matrixai.dense.tabm_cpu" in v["campo"]


def test_veredicto_de_c3_no_sube_porque_no_sube_los_cumplidos():
    veredictos = [{"dataset": f"d{i}", "mejora": True, "inferioridad": False} for i in range(5)]
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(3),
                             cumplidos_de_la_densa_v2=_cumplidos_de(3))  # empate: NO sube
    assert v["mejoras_superan_inferioridades"] is True
    assert v["sube_los_cumplidos"] is False
    assert v["cumple_la_regla_de_subida"] is False


def test_veredicto_de_c3_no_sube_porque_no_hay_mas_mejoras_que_inferioridades():
    veredictos = [{"dataset": f"d{i}", "mejora": i < 2, "inferioridad": i >= 2}
                  for i in range(5)]  # 2 mejora, 3 inferioridad
    v = p119.veredicto_de_c3(veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(5),
                             cumplidos_de_la_densa_v2=_cumplidos_de(3))
    assert v["sube_los_cumplidos"] is True
    assert v["mejoras_superan_inferioridades"] is False
    assert v["cumple_la_regla_de_subida"] is False


def test_veredicto_de_c3_reusa_veredicto_de_la_palanca_de_118():
    """No es una segunda implementación de la misma aritmética: se llama
    literalmente a la función de 118 por debajo."""
    import pasada_118_palanca as p118
    veredictos = [{"dataset": "d0", "mejora": True, "inferioridad": False}]
    directo = p118.veredicto_de_la_palanca(
        veredictos, cumplidos_con_la_palanca=_cumplidos_de(1, de=1),
        cumplidos_de_la_densa_v2=_cumplidos_de(0, de=1))
    envuelto = p119.veredicto_de_c3(
        veredictos, cumplidos_con_el_motor_nuevo=_cumplidos_de(1, de=1),
        cumplidos_de_la_densa_v2=_cumplidos_de(0, de=1))
    assert envuelto["cumple_la_regla_de_subida"] == directo["la_palanca_ayuda"]
    assert envuelto["n_mejora"] == directo["n_mejora"]
    assert envuelto["n_inferioridad"] == directo["n_inferioridad"]


# ---------------------------------------------------------------------------
# 5. EL «CAMPO» Y `_cumplidos_de`: sobre el mecanismo real de 118, reusado,
#    parametrizado con el motor nuevo en vez de con `NOMBRE_DENSA`
# ---------------------------------------------------------------------------

def test_campo_de_la_comparacion_sustituye_la_densa_por_el_motor_nuevo():
    import pasada_118_palanca as p118
    resultados_v2 = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", p119.NOMBRE_DENSA_V2, 0.70),
        _registro("dsB", "lightgbm", 0.75), _registro("dsB", p119.NOMBRE_DENSA_V2, 0.72),
    ]
    resultados_del_motor_nuevo = [_registro("dsA", p119.NOMBRE_MOTOR_NUEVO, 0.90)]
    campo = p118.campo_de_la_comparacion(resultados_v2, resultados_del_motor_nuevo, {"dsA"})

    nuevo_dsA = [r for r in campo if r["dataset"] == "dsA" and r["motor"] == p119.NOMBRE_MOTOR_NUEVO]
    densa_dsA = [r for r in campo if r["dataset"] == "dsA" and r["motor"] == p119.NOMBRE_DENSA_V2]
    densa_dsB = [r for r in campo if r["dataset"] == "dsB" and r["motor"] == p119.NOMBRE_DENSA_V2]
    assert nuevo_dsA == resultados_del_motor_nuevo
    assert densa_dsA == []  # la densa v2 de dsA se sustituye, no convive con la nueva
    assert densa_dsB == [_registro("dsB", p119.NOMBRE_DENSA_V2, 0.72)]  # dsB no se tocó


def test_cumplidos_de_se_acota_a_los_conjuntos_pedidos():
    regla = PROTOCOLO_V2.regla_de_cierre
    resultados = [
        _registro("dsA", "lightgbm", 0.80), _registro("dsA", p119.NOMBRE_MOTOR_NUEVO, 0.79),
        _registro("dsX", "lightgbm", 0.90), _registro("dsX", p119.NOMBRE_MOTOR_NUEVO, 0.10),
    ]
    salida = p119._cumplidos_de(resultados, regla, motor=p119.NOMBRE_MOTOR_NUEVO,
                                metrica_por_dataset={"dsA": "auroc"},
                                nombres_de_los_conjuntos=["dsA"])
    assert salida == {"cumplidos": 1, "datasets": 1, "fraccion": 1.0, "cumple_la_regla": True}


def test_cumplidos_de_puede_pedir_el_motor_de_la_densa_v2():
    """`_cumplidos_de` tiene que servir para los DOS lados de la
    comparación -- el motor nuevo Y la densa v2 -- a diferencia de
    `pasada_118_palanca._cumplidos`, que fija `motor=NOMBRE_DENSA` (la
    densa VIEJA) sin parámetro."""
    regla = PROTOCOLO_V2.regla_de_cierre
    resultados = [_registro("dsA", "lightgbm", 0.80),
                  _registro("dsA", p119.NOMBRE_DENSA_V2, 0.79)]
    salida = p119._cumplidos_de(resultados, regla, motor=p119.NOMBRE_DENSA_V2,
                                metrica_por_dataset={"dsA": "auroc"},
                                nombres_de_los_conjuntos=["dsA"])
    assert salida["cumplidos"] == 1


# ---------------------------------------------------------------------------
# 6. LA CONSTANTE DEL MOTOR NO HA DERIVADO DEL NOMBRE REAL
# ---------------------------------------------------------------------------

def test_nombre_motor_nuevo_coincide_con_la_clase_real():
    from matrixai_engines.motores.densa_tabm import MotorDensaTabM
    assert p119.NOMBRE_MOTOR_NUEVO == MotorDensaTabM().nombre
