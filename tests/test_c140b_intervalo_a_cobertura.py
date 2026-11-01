# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b-intervalo (decisión de Roberto, 01-11, tarjeta `140b-intervalo`, opción A) — el intervalo de la nota del reparto estricto
declara su cobertura MEDIDA, según el número de sujetos de prueba.

Lo que se afirma:
  · La tabla del código (`COBERTURA_MEDIDA_DEL_INTERVALO`) es la que midió el 140b-C0 (se compara con su `salidas/e3.json`; sin ese
    fichero la prueba se SALTA diciéndolo) y lleva su procedencia.
  · La regla para un número de sujetos que no está en la rejilla: el valor del punto medido menor o igual, NUNCA interpolado hacia
    arriba; por debajo del primer punto, «no medido» (`None`, no un cero); por encima del último, el del último.
  · `aproximado` con menos de 50 sujetos de prueba; la frase en los dos idiomas sale del núcleo, con la cifra «de cada 100» por
    debajo de lo medido.
  · Con un reparto REAL (8 sujetos de prueba: el punto medido; ~60: por encima de 50).
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_c140b_c1_particion_estricta import _cohorte, _estricta   # noqa: E402
from test_c140b_r1_reparacion_de_la_auditoria import _cohorte_con_pocos_elegibles, _pide   # noqa: E402

from matrixai.estudio import incertidumbre   # noqa: E402
from matrixai.training import particion_por_diseno_grupos_y_tiempo as modulo   # noqa: E402
from matrixai.training.particion_por_diseno_grupos_y_tiempo import (   # noqa: E402
    COBERTURA_MEDIDA_DEL_INTERVALO, NIVEL_NOMINAL_DEL_INTERVALO, PROCEDENCIA_DE_LA_COBERTURA,
    SUJETOS_PARA_QUE_NO_SEA_APROXIMADO, IntervaloDeclarado, cobertura_medida_del_intervalo)

#: La medición del C0, VERSIONADA (K4 de la tercera auditoría: antes vivía en `~/.agentes` y estas pruebas se saltaban en un
#: clon): el protocolo sellado, los guiones de E3 que sella y su salida `e3.json`.
C0 = Path(__file__).resolve().parents[1] / "benchmarks" / "cobertura_140b" / "c0"


# ---------------------------------------------------------------------------------------------------------------------------
# La tabla es la que midió el C0
# ---------------------------------------------------------------------------------------------------------------------------

def test_la_tabla_del_codigo_es_la_que_midio_el_c0():
    """Cobertura condicional SIN deriva de `e3.json` (RMSE): la media de las celdas de cada n, a tres decimales."""
    fichero = C0 / "salidas" / "e3.json"
    assert fichero.exists(), f"falta {fichero}: la procedencia de la tabla va versionada"
    celdas = json.loads(fichero.read_text())["rmse"]
    por_n: dict[int, list[float]] = {}
    for c in celdas:
        if c["tau"] == 0:
            por_n.setdefault(c["n"], []).append(c["cov_cond"])
    medida = tuple((n, round(sum(v) / len(v), 3)) for n, v in sorted(por_n.items()))
    assert medida, "e3.json no trae celdas sin deriva"
    assert medida == COBERTURA_MEDIDA_DEL_INTERVALO
    assert {len(v) for v in por_n.values()} == {24}   # 24 celdas por n: la media del C0


def test_la_procedencia_de_la_tabla_es_el_protocolo_sellado_del_c0():
    sellado = C0 / "PROTOCOLO.sha256"
    assert sellado.exists(), f"falta {sellado}"
    assert PROCEDENCIA_DE_LA_COBERTURA["protocolo_sha256"] in sellado.read_text()
    assert PROCEDENCIA_DE_LA_COBERTURA["fecha"] == "2026-11-01"
    assert "sintéticos" in PROCEDENCIA_DE_LA_COBERTURA["datos"] and "fijo" in PROCEDENCIA_DE_LA_COBERTURA["modelo"]


def test_la_tabla_es_creciente_con_los_sujetos_y_esta_ordenada():
    sujetos = [n for n, _ in COBERTURA_MEDIDA_DEL_INTERVALO]
    cobertura = [c for _, c in COBERTURA_MEDIDA_DEL_INTERVALO]
    assert sujetos == sorted(set(sujetos)) and sujetos[0] == 5 and sujetos[-1] == 200
    assert cobertura == sorted(cobertura) and all(0 < c < NIVEL_NOMINAL_DEL_INTERVALO for c in cobertura)


def test_el_nivel_nominal_de_la_frase_es_el_del_intervalo_de_la_nota():
    """«No 95»: el nivel por omisión de `incertidumbre.intervalo`, el que mide el intervalo de la nota."""
    assert inspect.signature(incertidumbre.intervalo).parameters["nivel"].default == NIVEL_NOMINAL_DEL_INTERVALO == 0.95


# ---------------------------------------------------------------------------------------------------------------------------
# La regla de los puntos que no están en la rejilla
# ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sujetos,esperado", [
    (0, (None, None)), (1, (None, None)), (4, (None, None)),            # por debajo del primer punto: no medido
    (5, (0.775, 5)), (6, (0.775, 5)), (7, (0.775, 5)),                  # el punto menor o igual
    (8, (0.835, 8)), (19, (0.835, 8)),
    (20, (0.890, 20)), (49, (0.890, 20)),
    (50, (0.922, 50)), (199, (0.922, 50)),
    (200, (0.936, 200)), (201, (0.936, 200)), (5000, (0.936, 200)),     # por encima del último: el del último
])
def test_un_punto_no_medido_toma_el_menor_o_igual_y_nunca_interpola_hacia_arriba(sujetos, esperado):
    assert cobertura_medida_del_intervalo(sujetos) == esperado


def test_nunca_se_promete_mas_de_lo_medido():
    """Para CUALQUIER n, la cobertura declarada no pasa de la medida en un punto con tantos sujetos o menos."""
    tabla = dict(COBERTURA_MEDIDA_DEL_INTERVALO)
    for n in range(0, 260):
        cobertura, punto = cobertura_medida_del_intervalo(n)
        if cobertura is None:
            assert n < 5 and punto is None
        else:
            assert punto <= n and tabla[punto] == cobertura
            assert cobertura == max(c for p, c in tabla.items() if p <= n)


# ---------------------------------------------------------------------------------------------------------------------------
# Un reparto REAL: 8 sujetos de prueba (un punto medido) y ~60 (por encima del umbral de «aproximado»)
# ---------------------------------------------------------------------------------------------------------------------------

def test_con_8_sujetos_de_prueba_el_intervalo_dice_unas_83_de_cada_100_y_es_aproximado():
    p = _pide(_cohorte_con_pocos_elegibles())
    r = p.reparto
    assert len(r.unidades_de_prueba) == 8
    iv = r.intervalo
    assert (iv.cobertura_medida, iv.sujetos_del_punto_medido, iv.aproximado) == (0.835, 8, True)
    assert iv.texto["es"] == ("con 8 sujetos de prueba, un intervalo así acierta unas 83 de cada 100 veces, no 95: vale para esta "
                              "ventana de tiempo (medido con datos sintéticos y un modelo fijo)")
    assert iv.texto["en"] == ("with 8 test subjects, an interval like this is right about 83 times in 100, not 95: it holds for "
                              "this time window (measured on synthetic data with a fixed model)")
    assert "intervalo aproximado" in iv.texto_aproximado["es"] and "approximate interval" in iv.texto_aproximado["en"]
    # y viaja en el sobre del reparto (lo que el backend manda tal cual)
    j = r.a_json()["intervalo"]
    assert j["cobertura_medida"] == 0.835 and j["sujetos_del_punto_medido"] == 8 and j["aproximado"] is True
    assert j["texto"] == iv.texto and j["texto_aproximado"] == iv.texto_aproximado


def test_con_unos_60_sujetos_de_prueba_ya_no_es_aproximado_y_dice_el_punto_de_50():
    p = _estricta(_cohorte(200, 10, entrada=60), folds=3)
    r = p.reparto
    n = len(r.unidades_de_prueba)
    assert 50 <= n < 200, n
    iv = r.intervalo
    assert (iv.cobertura_medida, iv.sujetos_del_punto_medido, iv.aproximado) == (0.922, 50, False)
    assert iv.texto_aproximado is None and r.a_json()["intervalo"]["texto_aproximado"] is None
    assert f"con {n} sujetos de prueba, un intervalo así acierta unas 92 de cada 100 veces o algo más, no 95 (se midió con 50" in iv.texto["es"]
    assert f"with {n} test subjects, an interval like this is right about 92 times in 100 or a little more, not 95 (it was measured with 50" in iv.texto["en"]


def _intervalo_de(n: int) -> IntervaloDeclarado:
    """El intervalo tal cual lo construye el reparto, para `n` sujetos de prueba (las mismas funciones, sin montar un reparto)."""
    cobertura, medido = cobertura_medida_del_intervalo(n)
    return IntervaloDeclarado(
        metodo="cluster_bootstrap_by_unit", diseno_de_remuestreo="groups", unidad_de_remuestreo="unit",
        condicional_a_la_ventana=True, unidades_de_prueba=n, ventana_de_prueba=(0.0, 1.0), cobertura_medida=cobertura,
        sujetos_del_punto_medido=medido, aproximado=n < SUJETOS_PARA_QUE_NO_SEA_APROXIMADO,
        texto=modulo._frase_de_cobertura(n, cobertura, medido), texto_aproximado=modulo._frase_de_aproximado(n))


def test_el_umbral_de_aproximado_es_el_de_50():
    assert SUJETOS_PARA_QUE_NO_SEA_APROXIMADO == 50
    assert modulo._frase_de_aproximado(49) is not None and modulo._frase_de_aproximado(50) is None


def test_el_booleano_aproximado_del_reparto_sale_de_la_misma_regla_que_su_marca_en_la_frontera_de_50(monkeypatch):
    """Supervisor, 01-11: el `aproximado` del reparto se calculaba aparte de su frase, y cambiarlo a `<=` (50 sujetos justos,
    «aproximado») dejaba 202 pruebas verdes. Ahora sale de `_frase_de_aproximado`, que está probada en 49/50: si esa regla
    dice «aproximado», el reparto lo dice; si dice que no, el reparto tampoco. Una comparación escrita aparte lo rompe."""
    marca = {"es": "intervalo aproximado (prueba)", "en": "approximate interval (test)"}
    for devuelve, esperado in ((marca, True), (None, False)):
        monkeypatch.setattr(modulo, "_frase_de_aproximado", lambda _n, _d=devuelve: _d)
        iv = _estricta(_cohorte(200, 10, entrada=60), folds=3).reparto.intervalo
        assert iv.aproximado is esperado and (iv.texto_aproximado is not None) is esperado
    monkeypatch.undo()
    assert (modulo._frase_de_aproximado(49) is not None, modulo._frase_de_aproximado(50) is not None) == (True, False)
    assert "50" in modulo._frase_de_aproximado(49)["es"] and "50" in modulo._frase_de_aproximado(49)["en"]


@pytest.mark.parametrize("n,cifra", [(5, 77), (8, 83), (20, 89), (50, 92), (200, 93), (7, 77), (500, 93)])
def test_la_cifra_de_cada_100_es_el_entero_por_debajo_de_lo_medido(n, cifra):
    iv = _intervalo_de(n)
    assert f"unas {cifra} de cada 100 veces" in iv.texto["es"] and f"about {cifra} times in 100" in iv.texto["en"]
    # los dos idiomas dicen las mismas cifras y ninguno copia al otro
    assert re.findall(r"\b\d+\b", iv.texto["es"]) == re.findall(r"\b\d+\b", iv.texto["en"])
    assert iv.texto["es"] != iv.texto["en"]


def test_un_numero_por_debajo_de_lo_medido_no_inventa_una_cifra():
    iv = _intervalo_de(3)
    assert (iv.cobertura_medida, iv.sujetos_del_punto_medido, iv.aproximado) == (None, None, True)
    assert "no se ha medido" in iv.texto["es"] and "has not been measured" in iv.texto["en"]
    assert "de cada 100" not in iv.texto["es"] and "times in 100" not in iv.texto["en"]
    assert iv.a_json()["cobertura_medida"] is None


def test_las_dos_formas_de_un_punto_no_medido_dicen_de_donde_sale_la_cifra():
    entre, encima = _intervalo_de(12), _intervalo_de(500)
    assert "se midió con 8 sujetos de prueba" in entre.texto["es"] and "it was measured with 8 test subjects" in entre.texto["en"]
    assert "llega hasta 200 sujetos de prueba" in encima.texto["es"] and "goes up to 200 test subjects" in encima.texto["en"]
    for iv in (entre, encima):
        assert "esta ventana de tiempo" in iv.texto["es"] and "this time window" in iv.texto["en"]


def test_el_reparto_pasa_por_la_misma_regla_que_la_funcion_publica():
    """Un único sitio: lo que el reparto declara es lo que `cobertura_medida_del_intervalo` da para sus sujetos de prueba."""
    vistos = set()
    for n, visitas, entrada in ((12, 12, 30), (20, 10, 60), (40, 10, 60), (100, 10, 60)):
        p = _estricta(_cohorte(n, visitas, entrada=entrada), folds=3)
        if not p.es_viable:
            continue
        iv = p.reparto.intervalo
        vistos.add(iv.unidades_de_prueba)
        assert (iv.cobertura_medida, iv.sujetos_del_punto_medido) == cobertura_medida_del_intervalo(iv.unidades_de_prueba)
        assert iv.aproximado is (iv.unidades_de_prueba < 50)
    assert len(vistos) >= 2, vistos
