# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C1 — el banco de pruebas de comportamiento (núcleo).

Las curvas son CONSTRUIDAS A MANO y el retroceso y el veredicto se calculan en el comentario, con
números exactos en binario (0,25; 0,5; 0,75) para que `==` signifique algo: un retroceso IGUAL a la
tolerancia es el caso que distingue `>` de `>=`.

Los datos tienen la forma que trae la gente: columnas con tilde y espacios (`Presión arterial`,
`edad_años`), `Sí/No` en mayúsculas y minúsculas, `0/1`, `NA`, huecos (`None`) y un valor de la columna
perturbada que ya era un hueco. El «modelo» de las pruebas es un sustituto que LEE esas formas, igual
que lo haría Usar.

Cada prueba que protege una línea que explica por qué NO hace lo obvio lleva el nombre de esa línea:
`test_un_retroceso_igual_a_la_tolerancia_pasa` guarda el `>`; `test_la_rejilla_crece` guarda el orden;
`test_una_columna_plana_no_dice_sin_contraejemplos` guarda el I5; `test_un_hueco_no_es_un_cero` el I8.
"""
from __future__ import annotations

import inspect
import json
import re
import statistics

import pytest

from matrixai.estudio import comportamiento as modulo
from matrixai.estudio import comportamiento_textos as textos
from matrixai.estudio.comportamiento import (
    DIRECCIONES_DE_MONOTONIA,
    MEDIDO_SOBRE,
    VERSION_DEL_COMPORTAMIENTO,
    VEREDICTOS,
    bloques_por_prefijo,
    caida_entre_pliegues,
    elegir_filas,
    escala_de_la_salida,
    k_mayores,
    monotonia_desde_curvas,
    monotonia_en_rejilla,
    ordenar_por_caida,
    rejilla,
    retroceso_de_una_curva,
    sensibilidad_de_una_columna,
)
from matrixai.estudio.comportamiento_textos import (
    CLAVE_DE_LOS_LIMITES,
    CLAVES_DE_VEREDICTO,
    IDIOMAS,
    MOTIVOS,
    huecos_de,
    motivo,
    palabras_prohibidas,
)
from matrixai.estudio.errores import EsquemaInvalido

BINARIA = escala_de_la_salida("binary_classification")      # escala 1,0: ε relativo == ε absoluto


def curvas(*cs, direccion="sube", eps=0.1, escala=BINARIA, ids=None, k=5):
    ids = list(ids) if ids is not None else [f"f{i}" for i in range(len(cs))]
    puntos = len(cs[0]) if cs else 4
    return monotonia_desde_curvas("x", direccion, [float(j) for j in range(puntos)], list(cs), ids,
                                  epsilon_relativo=eps, escala=escala, medido_sobre="muestra_de_desarrollo",
                                  semilla=3, k=k)


# ---------------------------------------------------------------------------
# El retroceso, a mano
# ---------------------------------------------------------------------------

def test_creciente_no_retrocede():
    r = retroceso_de_una_curva([0.0, 0.25, 0.5, 1.0], "sube")
    assert (r.valor, r.a, r.b) == (0.0, 0, 0)


def test_un_escalon_de_retroceso_conocido():
    # máximo corrido 0,5 en j=1; en j=2 vale 0,25 -> retroceso 0,25 entre a=1 y b=2; luego sube a 1,0.
    r = retroceso_de_una_curva([0.0, 0.5, 0.25, 1.0], "sube")
    assert (r.valor, r.a, r.b) == (0.25, 1, 2)


def test_baja_es_el_espejo_de_sube():
    # «baja»: el mayor ASCENSO respecto del mínimo corrido. Mínimo 0,5 en j=1, sube a 0,75 en j=2.
    r = retroceso_de_una_curva([1.0, 0.5, 0.75, 0.0], "baja")
    assert (r.valor, r.a, r.b) == (0.25, 1, 2)
    # y la misma curva declarada «sube» retrocede mucho más (de 1,0 a 0,0): el signo importa.
    assert retroceso_de_una_curva([1.0, 0.5, 0.75, 0.0], "sube").valor == 1.0


def test_el_signo_del_retroceso_no_es_intercambiable():
    creciente = [0.0, 0.25, 0.5, 1.0]
    assert retroceso_de_una_curva(creciente, "sube").valor == 0.0
    assert retroceso_de_una_curva(creciente, "baja").valor == 1.0


def test_retroceso_del_mayor_de_varios_escalones_y_empate_por_el_primero():
    # dos caídas iguales (0,25): se queda con la primera.
    r = retroceso_de_una_curva([0.5, 0.25, 0.5, 0.25], "sube")
    assert (r.valor, r.a, r.b) == (0.25, 0, 1)


def test_direccion_fuera_del_vocabulario_se_rechaza():
    with pytest.raises(EsquemaInvalido):
        retroceso_de_una_curva([0.0, 1.0], "creciente")
    assert DIRECCIONES_DE_MONOTONIA == ("sube", "baja")


# ---------------------------------------------------------------------------
# La rejilla
# ---------------------------------------------------------------------------

def test_la_rejilla_crece_y_incluye_los_extremos_exactos():
    g = rejilla(0.1, 0.7, 7)
    assert len(g) == 7
    assert g[0] == 0.1 and g[-1] == 0.7              # exactos, no `0.1 + 0.6*1`
    assert all(b > a for a, b in zip(g, g[1:]))        # creciente: el orden es lo que da sentido a «sube»


@pytest.mark.parametrize("minimo,maximo,puntos", [(1.0, 1.0, 5), (2.0, 1.0, 5), (0.0, 1.0, 1), (0.0, float("nan"), 5)])
def test_una_rejilla_imposible_se_rechaza(minimo, maximo, puntos):
    with pytest.raises(EsquemaInvalido):
        rejilla(minimo, maximo, puntos)


def test_el_orden_de_la_rejilla_llega_al_puntuar():
    """Con una salida que SUBE con la columna, la rejilla en orden creciente da «sin contraejemplos»; si
    el orden fuera el inverso la misma función saldría «con contraejemplos» sin que cambie un número."""
    vistos = []

    def puntuar(filas):
        vistos.append([f["x"] for f in filas])
        return [f["x"] / 10.0 for f in filas]

    r = monotonia_en_rejilla("x", "sube", [{"x": 5.0}], puntuar, minimo=0.0, maximo=1.0, puntos=5, filas_m=1, semilla=1,
                             epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    assert vistos == [[0.0, 0.25, 0.5, 0.75, 1.0]]
    assert r.veredicto == "sin_contraejemplos"
    # la contraria sobre la misma función: se viola en la fila, con retroceso == efecto.
    c = monotonia_en_rejilla("x", "baja", [{"x": 5.0}], lambda fs: [f["x"] / 10.0 for f in fs], minimo=0.0, maximo=1.0,
                             puntos=5, filas_m=1, semilla=1, epsilon_relativo=0.01, escala=BINARIA,
                             medido_sobre="muestra_de_desarrollo")
    assert c.veredicto == "con_contraejemplos"


# ---------------------------------------------------------------------------
# El veredicto, a mano
# ---------------------------------------------------------------------------

def test_creciente_sin_contraejemplos_y_con_su_efecto_total():
    r = curvas([0.0, 0.25, 0.5, 1.0], [0.25, 0.5, 0.75, 1.0], eps=0.1)
    assert r.veredicto == "sin_contraejemplos"
    assert r.efecto_total == statistics.fmean([1.0, 0.75])          # extremo a extremo, en la dirección declarada
    j = r.a_json()
    assert j["efecto_total"] == 0.875 and j["filas_con_violacion"] == 0 and j["contraejemplos"] == []
    assert j["veredicto"]["clave"] == "sin_contraejemplos"
    assert "0,875" in j["veredicto"]["lectura"]["es"] and "0.875" in j["veredicto"]["lectura"]["en"]   # I5: el efecto al lado


def test_con_un_escalon_de_retroceso_es_con_contraejemplos_y_lo_cuenta():
    r = curvas([0.0, 0.5, 0.25, 1.0], [0.0, 0.25, 0.5, 1.0], eps=0.1)
    assert r.veredicto == "con_contraejemplos"
    assert r.filas_con_violacion == 1
    assert r.contraejemplos == ({"fila": "f0", "x_a": 1.0, "x_b": 2.0, "y_a": 0.5, "y_b": 0.25, "retroceso": 0.25},)
    assert r.a_json()["retroceso_maximo"] == 0.25


def test_un_retroceso_igual_a_la_tolerancia_pasa():
    """`>` y no `>=`: un retroceso IGUAL a la tolerancia es lo que la tolerancia deja pasar."""
    curva = [0.0, 0.5, 0.25, 1.0]                   # retroceso exacto 0,25
    en_el_borde = curvas(curva, eps=0.25)
    assert en_el_borde.retrocesos == (0.25,) and en_el_borde.epsilon_absoluto == 0.25
    assert en_el_borde.veredicto == "sin_contraejemplos" and en_el_borde.filas_con_violacion == 0
    justo_debajo = curvas(curva, eps=0.2499)
    assert justo_debajo.veredicto == "con_contraejemplos"


def test_sin_tolerancia_el_mas_minimo_retroceso_cuenta():
    curva = [0.0, 0.5, 0.5 - 1e-9, 1.0]
    assert curvas(curva, eps=0.0).veredicto == "con_contraejemplos"
    assert curvas(curva, eps=0.01).veredicto == "sin_contraejemplos"


def test_una_columna_plana_no_dice_sin_contraejemplos():
    """I5: con efecto 0 la dirección contraria 'se cumple' igual que la declarada. Ni siquiera con ε = 0."""
    plana = [0.5, 0.5, 0.5, 0.5]
    for direccion in DIRECCIONES_DE_MONOTONIA:
        for eps in (0.0, 0.01, 0.5):
            r = curvas(plana, plana, direccion=direccion, eps=eps)
            assert r.veredicto == "no_se_pudo_comprobar", (direccion, eps)
            assert r.a_json()["veredicto"]["clave_del_motivo"] == "efecto_no_distingue"
            assert r.filas_con_violacion == 0 and r.efecto_total == 0.0


def test_un_efecto_que_no_supera_la_tolerancia_no_distingue():
    casi_plana = [0.5, 0.5001, 0.4999, 0.5]
    r = curvas(casi_plana, eps=0.01)
    assert r.veredicto == "no_se_pudo_comprobar"
    assert "no distingue" in r.a_json()["veredicto"]["motivo"]["es"]
    assert "cannot tell" in r.a_json()["veredicto"]["motivo"]["en"]


def test_el_efecto_total_se_mide_en_la_direccion_declarada():
    """Positivo = la salida se mueve como se declaró. Una curva que baja de 1,0 a 0,25 tiene efecto +0,75 si
    se declara «baja» y −0,75 si se declara «sube» (y en ese caso, además, retrocede: contraejemplo)."""
    baja = [1.0, 0.75, 0.5, 0.25]
    a = curvas(baja, direccion="baja", eps=0.1)
    assert a.efecto_total == 0.75 and a.veredicto == "sin_contraejemplos"
    b = curvas(baja, direccion="sube", eps=0.1)
    assert b.efecto_total == -0.75 and b.veredicto == "con_contraejemplos"


def test_la_direccion_contraria_de_una_columna_con_efecto_se_viola():
    """Los dientes de la prueba: con efecto la dirección contraria tiene contraejemplos; sin él, no."""
    sube = [0.0, 0.25, 0.5, 1.0]
    assert curvas(sube, direccion="sube", eps=0.05).veredicto == "sin_contraejemplos"
    assert curvas(sube, direccion="baja", eps=0.05).veredicto == "con_contraejemplos"


def test_vacia_no_se_pudo_comprobar_y_no_es_un_cero():
    r = curvas()
    assert r.veredicto == "no_se_pudo_comprobar"
    j = r.a_json()
    assert j["veredicto"]["clave_del_motivo"] == "sin_filas"
    assert j["retroceso_maximo"] is None and j["filas_con_violacion"] is None and j["efecto_total"] is None


def test_un_hueco_no_es_un_cero():
    """I8: una fila sin predicción se EXCLUYE y se cuenta; no es una curva plana ni un retroceso 0."""
    r = curvas([0.0, 0.5, 0.25, 1.0], [None, 0.2, 0.3, 0.4], [0.25, 0.5, 0.75, 1.0], eps=0.1)
    assert r.retrocesos == (0.25, None, 0.0) and r.efectos[1] is None
    assert r.filas_comprobadas == 2
    j = r.a_json()
    assert j["filas"] == {"pedidas": 3, "comprobadas": 2, "sin_prediccion": 1, "ids": ["f0", "f1", "f2"]}
    assert j["avisos"][0]["clave"] == "filas_sin_prediccion" and "1 de 3" in j["avisos"][0]["texto"]["es"]
    # el efecto medio es el de las DOS filas con predicción, no el de tres con un cero.
    assert r.efecto_total == statistics.fmean([1.0, 0.75])
    assert r.veredicto == "con_contraejemplos"


def test_sin_ninguna_prediccion_no_se_pudo_comprobar_con_su_motivo():
    r = curvas([None, None, None, None], [0.1, None, 0.3, 0.4])
    assert r.veredicto == "no_se_pudo_comprobar"
    assert r.a_json()["veredicto"]["clave_del_motivo"] == "sin_prediccion"
    assert r.retrocesos == (None, None) and r.filas_con_violacion is None and r.a_json()["retroceso_maximo"] is None


def test_una_salida_que_no_es_un_numero_no_se_comprueba():
    r = curvas([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    assert r.veredicto == "no_se_pudo_comprobar"
    assert r.a_json()["veredicto"]["clave_del_motivo"] == "salida_no_numerica"
    r = curvas([0.1, True, 0.3, 0.4])                                   # un booleano no es una probabilidad
    assert r.a_json()["veredicto"]["clave_del_motivo"] == "salida_no_numerica"
    nan = curvas([0.1, float("nan"), 0.3, 0.4])                         # NaN = sin predicción, no un número
    assert nan.retrocesos == (None,)


def test_los_k_peores_son_los_de_mayor_retroceso_y_no_dependen_de_epsilon():
    c = [[0.0, 0.5, 0.25, 1.0],        # 0,25
         [0.0, 0.5, 0.375, 1.0],       # 0,125
         [0.0, 0.5, 0.0, 1.0],         # 0,5
         [0.0, 0.25, 0.5, 1.0]]        # 0
    r = curvas(*c, eps=0.0, k=2)
    assert [p["fila"] for p in r.peores] == ["f2", "f0"]
    assert [p["retroceso"] for p in r.peores] == [0.5, 0.25]
    # con otra ε se guardan los MISMOS peores; solo cambian los que se enseñan como contraejemplos
    r2 = r.con_epsilon(0.3)
    assert r2.peores == r.peores
    assert [p["fila"] for p in r2.contraejemplos] == ["f2"]
    assert [p["fila"] for p in r.contraejemplos] == ["f2", "f0"]


# ---------------------------------------------------------------------------
# I7 — la tolerancia es del cálculo y se cambia sin volver a predecir
# ---------------------------------------------------------------------------

FILAS_DE_LA_GENTE = [
    {"id": "1", "Presión arterial": "Sí", "edad_años": 40, "Fumador": "SÍ", "Sexo": "M", "peso": 70.5},
    {"id": "2", "Presión arterial": "no", "edad_años": None, "Fumador": "NA", "Sexo": "m", "peso": None},
    {"id": "3", "Presión arterial": "1", "edad_años": 61, "Fumador": "Sí", "Sexo": "F", "peso": 82.0},
    {"id": "4", "Presión arterial": "0", "edad_años": "NA", "Fumador": "No", "Sexo": "f", "peso": 59.0},
    {"id": "5", "Presión arterial": "Sí", "edad_años": 33, "Fumador": "no", "Sexo": "F", "peso": 64.2},
]


def _si(valor):
    return 1.0 if str(valor).strip().lower() in ("sí", "si", "1", "true") else 0.0


def modelo_de_la_gente(filas):
    """Un sustituto de modelo que LEE las formas reales (Sí/No/0/1/NA/huecos) y sube con la edad con un
    escalón de retroceso en 50..60: la curva en `edad_años` NO es monótona, la de `peso` sí."""
    salidas = []
    for f in filas:
        edad, peso = f["edad_años"], f["peso"]
        if not isinstance(edad, (int, float)):
            edad = 45.0                                      # lo que Usar pondría ante un hueco
        if not isinstance(peso, (int, float)):
            peso = 70.0
        p = edad / 100.0 - (0.2 if 50 <= edad <= 60 else 0.0) + 0.001 * peso + 0.05 * _si(f["Presión arterial"])
        salidas.append(min(1.0, max(0.0, p)))
    return salidas


def test_la_muestra_del_llamante_no_se_toca_y_el_resto_de_la_fila_queda_fijo():
    antes = json.dumps(FILAS_DE_LA_GENTE, ensure_ascii=False, sort_keys=True)
    recibidas = []

    def puntuar(filas):
        recibidas.extend(filas)
        return modelo_de_la_gente(filas)

    monotonia_en_rejilla("edad_años", "sube", FILAS_DE_LA_GENTE, puntuar, minimo=20.0, maximo=80.0, puntos=7, filas_m=10,
                         semilla=2, epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    assert json.dumps(FILAS_DE_LA_GENTE, ensure_ascii=False, sort_keys=True) == antes
    assert len(recibidas) == 5 * 7                                    # m > n: todas, G puntos cada una
    for i, base in enumerate(FILAS_DE_LA_GENTE):
        for j in range(7):
            f = recibidas[i * 7 + j]
            assert f["edad_años"] == 20.0 + 60.0 * j / 6              # un hueco ('NA', None) se SOBRESCRIBE
            assert {k: v for k, v in f.items() if k != "edad_años"} == {k: v for k, v in base.items() if k != "edad_años"}


def test_con_otro_epsilon_el_veredicto_cambia_sin_volver_a_predecir():
    llamadas = []

    def puntuar(filas):
        llamadas.append(len(filas))
        return modelo_de_la_gente(filas)

    kw = dict(minimo=20.0, maximo=80.0, puntos=13, filas_m=5, semilla=2, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    estricto = monotonia_en_rejilla("edad_años", "sube", FILAS_DE_LA_GENTE, puntuar, epsilon_relativo=0.01, **kw)
    assert llamadas == [65]
    assert estricto.veredicto == "con_contraejemplos"
    flojo = estricto.con_epsilon(0.5)
    assert llamadas == [65]                                            # I7: ni una predicción más
    assert flojo.veredicto == "sin_contraejemplos"
    assert flojo.a_json()["tolerancia"]["relativa"] == 0.5             # y dice qué ε usó
    # recalcular == calcular desde cero con esa ε
    desde_cero = monotonia_en_rejilla("edad_años", "sube", FILAS_DE_LA_GENTE, puntuar, epsilon_relativo=0.5, **kw)
    assert flojo.a_json() == desde_cero.a_json() and flojo.digest() == desde_cero.digest()
    # cambiar ε cambia SOLO lo que depende de ε
    a, b = estricto.a_json(), flojo.a_json()
    cambian = {k for k in a if a[k] != b[k]}
    assert cambian == {"tolerancia", "filas_con_violacion", "contraejemplos", "veredicto"}
    assert a["para_recalcular"] == b["para_recalcular"] and a["efecto_total"] == b["efecto_total"]


def test_misma_semilla_misma_salida_bit_a_bit_y_otra_semilla_otras_filas():
    filas = [{"x": float(i), "ruido": i % 3} for i in range(300)]

    def puntuar(fs):
        return [f["x"] / 300.0 + 0.01 * ((f["ruido"] * 7 + int(f["x"])) % 5) for f in fs]

    def correr(semilla):
        return monotonia_en_rejilla("x", "sube", filas, puntuar, minimo=0.0, maximo=300.0, puntos=15, filas_m=40, semilla=semilla,
                                    epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo")

    a, b, c = correr(11), correr(11), correr(12)
    assert json.dumps(a.a_json(), sort_keys=True) == json.dumps(b.a_json(), sort_keys=True)
    assert a.digest() == b.digest()
    assert a.ids != c.ids and a.digest() != c.digest()
    assert a.ids == elegir_filas(300, 40, 11) and list(a.ids) == sorted(a.ids) and len(set(a.ids)) == 40


def test_elegir_filas_con_pocas_filas_las_da_todas():
    assert elegir_filas(5, 100, 0) == (0, 1, 2, 3, 4)
    assert elegir_filas(0, 100, 0) == ()


# ---------------------------------------------------------------------------
# I8 — sin escala, sin rango, sin predicción: «no se pudo comprobar» con su motivo
# ---------------------------------------------------------------------------

def _no_debe_predecir(filas):                     # pragma: no cover - solo falla si se llama
    raise AssertionError("no se debía predecir")


def test_binaria_con_pendiente_no_positiva_no_hay_escala_ni_se_predice():
    for b in (0.0, -0.3, float("nan"), float("inf")):
        escala = escala_de_la_salida("binary_classification", pendiente_de_la_recalibracion=b)
        assert escala.valor is None and escala.clave_del_motivo == "recalibracion_no_conserva_el_orden"
        r = monotonia_en_rejilla("x", "sube", [{"x": 1.0}], _no_debe_predecir, minimo=0.0, maximo=1.0, puntos=5, filas_m=1,
                                 semilla=1, epsilon_relativo=0.01, escala=escala, medido_sobre="muestra_de_desarrollo")
        assert r.veredicto == "no_se_pudo_comprobar"
        j = r.a_json()
        assert j["veredicto"]["clave_del_motivo"] == "recalibracion_no_conserva_el_orden"
        assert j["filas_con_violacion"] is None and j["retroceso_maximo"] is None
    assert escala_de_la_salida("binary_classification", pendiente_de_la_recalibracion=0.8).valor == 1.0
    assert escala_de_la_salida("binary_classification").valor == 1.0       # sin recalibración: la del modelo


def test_la_escala_de_regresion_es_la_amplitud_del_entrenamiento():
    assert escala_de_la_salida("regression", amplitud_p90_p10=12.5).valor == 12.5
    for malo in (None, 0.0, -1.0, float("nan")):
        e = escala_de_la_salida("regression", amplitud_p90_p10=malo)
        assert e.valor is None and e.clave_del_motivo == "escala_de_regresion_no_valida"
    # y la tolerancia absoluta es relativa × escala
    r = monotonia_desde_curvas("x", "sube", [0.0, 1.0], [[0.0, 10.0]], ["a"], epsilon_relativo=0.01,
                               escala=escala_de_la_salida("regression", amplitud_p90_p10=50.0),
                               medido_sobre="muestra_de_desarrollo", semilla=1)
    assert r.epsilon_absoluto == 0.5


def test_multiclase_sin_clase_declarada_no_se_comprueba():
    e = escala_de_la_salida("multiclass_classification")
    assert e.valor is None and e.clave_del_motivo == "multiclase_sin_clase_declarada"
    with pytest.raises(EsquemaInvalido):
        escala_de_la_salida("ordinal")


@pytest.mark.parametrize("minimo,maximo", [(None, None), (1.0, 1.0), (2.0, 1.0), (float("nan"), 3.0), (0.0, float("inf"))])
def test_una_columna_sin_rango_no_tiene_rejilla(minimo, maximo):
    r = monotonia_en_rejilla("x", "sube", [{"x": 1.0}], _no_debe_predecir, minimo=minimo, maximo=maximo, puntos=5, filas_m=1,
                             semilla=1, epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    assert r.a_json()["veredicto"]["clave_del_motivo"] == "rejilla_sin_rango" and r.veredicto == "no_se_pudo_comprobar"


def test_sin_filas_no_se_predice():
    r = monotonia_en_rejilla("x", "sube", [], _no_debe_predecir, minimo=0.0, maximo=1.0, puntos=5, filas_m=10, semilla=1,
                             epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    assert r.a_json()["veredicto"]["clave_del_motivo"] == "sin_filas"


def test_el_puntuar_que_devuelve_de_menos_se_rechaza():
    with pytest.raises(EsquemaInvalido):
        monotonia_en_rejilla("x", "sube", [{"x": 1.0}], lambda fs: [0.1], minimo=0.0, maximo=1.0, puntos=5, filas_m=1, semilla=1,
                             epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo")


def test_el_epsilon_no_tiene_valor_por_omision_ni_acepta_negativos():
    """Lo fija C0, no este módulo: sin parámetro no se llama, y un ε negativo no existe."""
    assert inspect.signature(monotonia_en_rejilla).parameters["epsilon_relativo"].default is inspect.Parameter.empty
    assert inspect.signature(monotonia_desde_curvas).parameters["epsilon_relativo"].default is inspect.Parameter.empty
    with pytest.raises(EsquemaInvalido):
        curvas([0.0, 1.0], eps=-0.01)


# ---------------------------------------------------------------------------
# I1 — solo desarrollo
# ---------------------------------------------------------------------------

def test_la_prueba_reservada_no_es_un_valor_admitido():
    assert "prueba_reservada" not in MEDIDO_SOBRE and "test" not in MEDIDO_SOBRE
    for malo in ("prueba_reservada", "test", "muestra_de_prueba", "en_muestra"):
        with pytest.raises(EsquemaInvalido):
            monotonia_desde_curvas("x", "sube", [0.0, 1.0], [[0.0, 1.0]], ["a"], epsilon_relativo=0.01, escala=BINARIA,
                                   medido_sobre=malo, semilla=1)
        with pytest.raises(EsquemaInvalido):
            sensibilidad_de_una_columna("x", [{"x": 1.0}], lambda fs: [0.0], minimo=0.0, maximo=1.0, semilla=1, filas_n=1,
                                        medido_sobre=malo)


def test_la_caida_de_falta_solo_se_acepta_fuera_de_pliegue():
    """Medir «falta» en muestra sale 10–40 veces más pequeño (cáncer de mama: 0,0001 frente a 0,0042)."""
    for malo in ("muestra_de_desarrollo", "prueba_reservada"):
        with pytest.raises(EsquemaInvalido):
            caida_entre_pliegues("x", [0.1], nunca_falto_al_entrenar=False, medido_sobre=malo)
    assert caida_entre_pliegues("x", [0.1], nunca_falto_al_entrenar=False)["medido_sobre"] == "validacion_de_cada_pliegue"


def test_el_modulo_no_sabe_de_motores_ni_del_studio_ni_de_los_ficheros_veneno():
    """I1 y I13: núcleo puro. Ningún nombre de los ficheros que contienen filas del test, ni acceso a disco,
    ni importar motores o Studio. (El Studio lo comprobará con ficheros-veneno; aquí, el fuente.)"""
    for modulo_ in (modulo, textos):
        fuente = inspect.getsource(modulo_)
        for prohibido in ("filas_de_referencia", "simulador_prueba_reservada", "mapa_de_la_decision", "matrixai_engines",
                          "matrixai_studio", "import os", "pathlib", "open(", "read_text", "json.load"):
            assert prohibido not in fuente, prohibido


# ---------------------------------------------------------------------------
# I2 — A y B describen, no tienen veredicto; solo C lo lleva
# ---------------------------------------------------------------------------

def _claves(doc):
    if isinstance(doc, dict):
        for k, v in doc.items():
            yield k
            yield from _claves(v)
    elif isinstance(doc, (list, tuple)):
        for v in doc:
            yield from _claves(v)


CLAVES_DE_VEREDICTO_PROHIBIDAS = {"veredicto", "verdict", "aprobado", "suspenso", "pasa", "falla", "semaforo", "semáforo", "superado"}


def test_a_y_b_no_llevan_ninguna_clave_de_veredicto_y_c_si():
    filas = [{"x": float(i)} for i in range(50)]
    b = sensibilidad_de_una_columna("x", filas, lambda fs: [f["x"] / 50 for f in fs], minimo=0.0, maximo=50.0, semilla=1,
                                    filas_n=20, medido_sobre="muestra_de_desarrollo", decidir=lambda y: y >= 0.5).a_json()
    a = caida_entre_pliegues("x", [0.01, 0.02, None], nunca_falto_al_entrenar=True)
    c = curvas([0.0, 0.25, 0.5, 1.0]).a_json()
    assert not CLAVES_DE_VEREDICTO_PROHIBIDAS & set(_claves(a))
    assert not CLAVES_DE_VEREDICTO_PROHIBIDAS & set(_claves(b))
    assert "veredicto" in c


# ---------------------------------------------------------------------------
# B — sensibilidad
# ---------------------------------------------------------------------------

def test_sensibilidad_a_mano():
    # 10 filas, x = 0..9; salida = (x/10)²; rango visto 0..10 y fracción 10 % -> δ = 1,0 (exacto).
    # Con +δ la salida pasa de x²/100 a (x+1)²/100: |Δ|/δ = (2x+1)/100 -> 0,01 … 0,19 (mediana 0,10, máximo en x = 9).
    # Decisión «salida >= 0,5»: con +δ solo cruza x = 7 (0,49 -> 0,64); con −δ solo x = 8 (0,64 -> 0,49).
    filas = [{"x": float(i)} for i in range(10)]
    r = sensibilidad_de_una_columna("x", filas, lambda fs: [(f["x"] / 10.0) ** 2 for f in fs], minimo=0.0, maximo=10.0, semilla=1,
                                    filas_n=10, medido_sobre="muestra_de_desarrollo", fracciones=(0.1,),
                                    decidir=lambda y: y >= 0.5)
    mas, menos = r.pruebas
    assert (mas["signo"], menos["signo"]) == ("mas", "menos") and mas["delta"] == 1.0
    assert mas["filas_comprobadas"] == 10 and mas["cambian_de_decision"] == 1
    assert menos["cambian_de_decision"] == 1
    assert mas["fraccion_que_cambia_de_decision"] == pytest.approx(0.1)
    assert mas["pendiente"]["mediana"] == pytest.approx(0.10) and mas["pendiente"]["maximo"] == pytest.approx(0.19)
    assert len(mas["mayores_pendientes"]) == 5
    primera = mas["mayores_pendientes"][0]
    assert (primera["fila"], primera["valor_antes"], primera["valor_despues"]) == (9, 9.0, 10.0)
    assert primera["salida_antes"] == pytest.approx(0.81) and primera["salida_despues"] == pytest.approx(1.0)
    assert [m["fila"] for m in mas["mayores_pendientes"]] == [9, 8, 7, 6, 5]           # de mayor a menor pendiente


def test_sensibilidad_en_regresion_no_inventa_un_cero_de_cambios():
    filas = [{"x": float(i)} for i in range(10)]
    r = sensibilidad_de_una_columna("x", filas, lambda fs: [f["x"] for f in fs], minimo=0.0, maximo=10.0, semilla=1, filas_n=10,
                                    medido_sobre="muestra_de_desarrollo")
    for p in r.pruebas:
        assert p["cambian_de_decision"] is None and p["fraccion_que_cambia_de_decision"] is None
        assert p["sin_decision"]["es"]
        assert p["pendiente"]["maximo"] == pytest.approx(1.0)


def test_sensibilidad_deja_los_huecos_como_huecos():
    filas = FILAS_DE_LA_GENTE
    r = sensibilidad_de_una_columna("edad_años", filas, modelo_de_la_gente, minimo=20.0, maximo=80.0, semilla=1, filas_n=10,
                                    medido_sobre="muestra_de_desarrollo", fracciones=(0.05,), ids=[f["id"] for f in filas])
    for p in r.pruebas:
        assert p["filas_sin_valor_en_la_columna"] == 2 and p["filas_comprobadas"] == 3      # None y 'NA' no se mueven
        assert {m["fila"] for m in p["mayores_pendientes"]} <= {"1", "3", "5"}
    sin = sensibilidad_de_una_columna("edad_años", [{"edad_años": None}], _no_debe_predecir, minimo=20.0, maximo=80.0, semilla=1,
                                      filas_n=3, medido_sobre="muestra_de_desarrollo")
    assert sin.a_json()["clave_del_motivo"] == "sensibilidad_sin_filas" and sin.pruebas == () and not sin.a_json()["disponible"]
    sin_rango = sensibilidad_de_una_columna("x", [{"x": 1.0}], _no_debe_predecir, minimo=None, maximo=None, semilla=1, filas_n=3,
                                            medido_sobre="muestra_de_desarrollo")
    assert sin_rango.a_json()["clave_del_motivo"] == "sensibilidad_sin_rango"


def test_sensibilidad_multiclase_usa_el_mayor_cambio_de_una_clase_y_es_determinista():
    filas = [{"x": float(i)} for i in range(20)]

    def puntuar(fs):
        return [[0.5 - 0.01 * f["x"], 0.5 + 0.01 * f["x"]] for f in fs]

    kw = dict(minimo=0.0, maximo=20.0, semilla=5, filas_n=8, medido_sobre="muestra_de_desarrollo", decidir=lambda y: y.index(max(y)))
    a, b = sensibilidad_de_una_columna("x", filas, puntuar, **kw), sensibilidad_de_una_columna("x", filas, puntuar, **kw)
    assert json.dumps(a.a_json(), sort_keys=True) == json.dumps(b.a_json(), sort_keys=True) and a.digest() == b.digest()
    # δ = 1 % de 20 = 0,2; cada clase se mueve 0,01·0,2 = 0,002 -> |Δ|/δ = 0,01 en todas las filas (L∞ de las dos clases)
    assert a.pruebas[0]["pendiente"]["mediana"] == pytest.approx(0.01, rel=1e-6)
    assert a.pruebas[0]["pendiente"]["maximo"] == pytest.approx(0.01, rel=1e-6)


# ---------------------------------------------------------------------------
# A — la agregación entre pliegues
# ---------------------------------------------------------------------------

def test_la_caida_entre_pliegues_a_mano():
    r = caida_entre_pliegues("peor_puntos_concavos", [0.004, 0.005, 0.003, None], nunca_falto_al_entrenar=True)
    assert r["caida_media"] == pytest.approx(0.004) and r["desviacion"] == pytest.approx(0.001)
    assert (r["pliegues"], r["pliegues_con_caida"]) == (4, 3) and r["caidas"][3] is None
    assert r["aviso"]["es"] and "nunca faltó" in r["aviso"]["es"] and "never went missing" in r["aviso"]["en"]
    assert caida_entre_pliegues("x", [0.1], nunca_falto_al_entrenar=False)["aviso"] is None
    assert caida_entre_pliegues("x", [0.1], nunca_falto_al_entrenar=None)["aviso"] is None   # no saberlo no es decir que no


def test_sin_ningun_pliegue_con_valor_no_hay_cifra_ni_cero():
    r = caida_entre_pliegues("x", [None, None], nunca_falto_al_entrenar=False)
    assert r["caida_media"] is None and r["desviacion"] is None and r["motivo"]["es"]
    assert caida_entre_pliegues("x", [0.2], nunca_falto_al_entrenar=False)["desviacion"] == 0.0


def test_el_orden_de_las_caidas_deja_las_sin_cifra_al_final_y_desempata_por_nombre():
    rs = [caida_entre_pliegues(n, c, nunca_falto_al_entrenar=False)
          for n, c in (("b_año", [0.01]), ("a_año", [0.01]), ("sin", [None]), ("gran", [0.2]), ("Ñu", [-0.01]))]
    o = ordenar_por_caida(rs)
    assert [r["columna"] for r in o] == ["gran", "a_año", "b_año", "Ñu", "sin"]
    assert k_mayores(o, 2) == ("gran", "a_año") and k_mayores(o, 10) == ("gran", "a_año", "b_año", "Ñu")


def test_bloques_por_prefijo():
    cols = ["media_radio", "peor_radio", "peor_textura", "peor_área", "media_textura", "Peor_x", "solo", "_raro", "años_a", "años_b"]
    b = bloques_por_prefijo(cols)
    assert b == {"media": ("media_radio", "media_textura"), "peor": ("peor_radio", "peor_textura", "peor_área"),
                 "años": ("años_a", "años_b")}
    assert "Peor" not in b and "" not in b                   # comparación exacta; un prefijo vacío no es un bloque


# ---------------------------------------------------------------------------
# I3 — vocabulario cerrado; I4 — límites pegados; las tres palabras
# ---------------------------------------------------------------------------

def test_las_tres_palabras_del_veredicto():
    assert VEREDICTOS == ("sin_contraejemplos", "con_contraejemplos", "no_se_pudo_comprobar")
    assert tuple(CLAVES_DE_VEREDICTO) == VEREDICTOS                    # dos listas del mismo vocabulario: la misma
    esperado = {"sin_contraejemplos": ("sin contraejemplos en esta rejilla", "no counterexamples on this grid"),
                "con_contraejemplos": ("con contraejemplos", "with counterexamples"),
                "no_se_pudo_comprobar": ("no se pudo comprobar", "could not be checked")}
    for clave, (es, en) in esperado.items():
        assert motivo(CLAVES_DE_VEREDICTO[clave]) == {"es": es, "en": en}


def _todos_los_veredictos():
    """Un documento de cada clase de veredicto y de cada motivo de «no se pudo comprobar»."""
    docs = [curvas([0.0, 0.25, 0.5, 1.0]), curvas([0.0, 0.5, 0.25, 1.0]), curvas([0.5] * 4), curvas(), curvas([None] * 4),
            curvas([[0.1, 0.2]] * 4)]
    for malo in (escala_de_la_salida("binary_classification", pendiente_de_la_recalibracion=-1.0),
                 escala_de_la_salida("regression", amplitud_p90_p10=0.0), escala_de_la_salida("multiclass_classification")):
        docs.append(monotonia_en_rejilla("x", "baja", [{"x": 1.0}], _no_debe_predecir, minimo=0.0, maximo=1.0, puntos=5,
                                         filas_m=1, semilla=1, epsilon_relativo=0.01, escala=malo,
                                         medido_sobre="fila_tipica"))
    docs.append(monotonia_en_rejilla("x", "sube", [{"x": 1.0}], _no_debe_predecir, minimo=None, maximo=None, puntos=5, filas_m=1,
                                     semilla=1, epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo"))
    return docs


def test_la_frase_de_limites_va_pegada_a_cada_veredicto():
    """I4: en TODOS los veredictos (también «no se pudo comprobar»), dentro de `veredicto` y no en un pie."""
    for r in _todos_los_veredictos():
        j = r.a_json()
        esperada = motivo(CLAVE_DE_LOS_LIMITES, puntos=len(r.rejilla), filas=r.filas_comprobadas, semilla=3 if r.medido_sobre["semilla"] == 3 else 1)
        assert j["veredicto"]["limites"] == esperada, j["veredicto"]["clave"]
        assert "limites" not in {k for k in j if k != "veredicto"}
    j = curvas([0.0, 0.25, 0.5, 1.0], [0.0, 0.5, 0.25, 1.0]).a_json()
    assert j["veredicto"]["limites"] == {
        "es": "Esto se ha comprobado en 4 puntos y 2 filas con semilla 3. No es una garantía: con otros valores, otras "
              "combinaciones o otras columnas a la vez puede comportarse de otra manera. La monotonía declarada no se "
              "impuso al entrenar.",
        "en": "This was checked at 4 points and 2 rows with seed 3. It is not a guarantee: with other values, other "
              "combinations or other columns at once it may behave differently. The declared monotonicity was not "
              "imposed during training."}


def _textos_de(doc, ruta=""):
    """Todos los `{es, en}` de un documento, con su ruta."""
    if isinstance(doc, dict):
        if set(doc) == {"es", "en"} and all(isinstance(v, str) for v in doc.values()):
            yield ruta, doc
        else:
            for k, v in doc.items():
                yield from _textos_de(v, f"{ruta}.{k}")
    elif isinstance(doc, (list, tuple)):
        for i, v in enumerate(doc):
            yield from _textos_de(v, f"{ruta}[{i}]")


def test_barrido_de_vocabulario_sobre_el_catalogo_entero_y_sobre_los_documentos():
    for clave, redaccion in MOTIVOS.items():
        for idioma in IDIOMAS:
            assert palabras_prohibidas(redaccion[idioma], idioma, es_la_frase_de_limites=(clave == CLAVE_DE_LOS_LIMITES)) == [], \
                (clave, idioma)
    for r in _todos_los_veredictos():
        for ruta, par in _textos_de(r.a_json()):
            for idioma in IDIOMAS:
                assert palabras_prohibidas(par[idioma], idioma, es_la_frase_de_limites=ruta.endswith("limites")) == [], (ruta, idioma)
    ba = sensibilidad_de_una_columna("x", [{"x": 1.0}, {"x": 2.0}], lambda fs: [f["x"] for f in fs], minimo=0.0, maximo=5.0, semilla=1,
                                     filas_n=2, medido_sobre="muestra_de_desarrollo").a_json()
    ab = caida_entre_pliegues("x", [None], nunca_falto_al_entrenar=True)
    for doc in (ba, ab):
        for ruta, par in _textos_de(doc):
            for idioma in IDIOMAS:
                assert palabras_prohibidas(par[idioma], idioma) == [], (ruta, idioma)


def test_la_unica_excepcion_es_la_garantia_dentro_de_la_frase_de_limites():
    limites = MOTIVOS[CLAVE_DE_LOS_LIMITES]
    assert palabras_prohibidas(limites["es"], "es") == ["garantia"] and palabras_prohibidas(limites["en"], "en") == ["guarantee"]
    assert palabras_prohibidas(limites["es"], "es", es_la_frase_de_limites=True) == []
    # y solo esa palabra: cualquier otra, incluso dentro de la frase de límites, sigue siendo un hallazgo
    assert palabras_prohibidas(limites["es"] + " Aprobado.", "es", es_la_frase_de_limites=True) == ["aprobado"]
    assert palabras_prohibidas("No garantiza nada", "es", es_la_frase_de_limites=True) == ["garantiza"]


@pytest.mark.parametrize("idioma,palabras", [
    ("es", ["garantiza", "Garantía", "demuestra", "demostrado", "cumple", "cumplen", "seguro", "segura", "aprobado", "aprobada",
            "suspenso", "pasa", "falla", "robusto", "robusta", "monótono", "monótona"]),
    ("en", ["guarantees", "guaranteed", "proves", "proven", "complies", "compliant", "safe", "safely", "passed", "passes", "fails",
            "failed", "robust", "monotone", "monotonic", "approved"]),
])
def test_el_barrido_caza_cada_palabra_prohibida(idioma, palabras):
    for p in palabras:
        assert palabras_prohibidas(f"el resultado {p} en todo", idioma), p


@pytest.mark.parametrize("idioma,frases", [
    ("es", ["la monotonía declarada", "la pasada de ayer", "la seguridad", "sin contraejemplos en esta rejilla", "no se pudo comprobar"]),
    ("en", ["the declared monotonicity", "a passage", "with counterexamples", "could not be checked", "it is not a guarantee of"]),
])
def test_el_barrido_no_caza_de_mas(idioma, frases):
    for f in frases:
        assert palabras_prohibidas(f, idioma) in ([], ["guarantee"]), f


# El idioma: PALABRAS FUNCIONALES, no una lista de contenido escogida a mano. Las que se escriben igual en
# los dos idiomas (`a`, `no`, `me`…) quedan fuera.
_ES = {"el", "la", "los", "las", "del", "que", "con", "por", "para", "se", "su", "sus", "lo", "una", "un", "de", "en", "es", "al",
       "sin", "más", "este", "esta", "ese", "como", "ya", "ni", "o", "y", "tanto", "otro", "otra", "otros", "otras", "no", "cada"}
_EN = {"the", "of", "and", "is", "are", "was", "were", "with", "from", "this", "that", "it", "its", "which", "than", "by", "for",
       "in", "on", "at", "to", "not", "or", "as", "be", "has", "have", "does", "a", "an", "any", "other", "each"}
_SOLO_ES = {"el", "la", "los", "las", "del", "que", "con", "por", "para", "se", "su", "sus", "lo", "una", "de", "al", "sin", "más", "este",
            "esta", "ese", "como", "ya", "ni", "tanto", "otro", "otra", "otros", "otras", "cada", "y", "es", "en", "un", "o"}
_SOLO_EN = {"the", "of", "and", "is", "are", "was", "were", "with", "from", "this", "that", "its", "which", "than", "by", "for",
            "at", "to", "not", "be", "has", "have", "does", "an", "any", "other", "each", "it", "or", "as", "on"}


def _palabras(texto):
    return re.findall(r"[a-záéíóúüñ]+", re.sub(r"\{[^}]*\}", " ", texto).lower())      # sin los nombres de los huecos


def test_cada_redaccion_esta_en_su_idioma_por_palabras_funcionales():
    textos_ = [("catalogo." + c, r) for c, r in MOTIVOS.items()]
    for r in _todos_los_veredictos():
        textos_ += [(ruta, par) for ruta, par in _textos_de(r.a_json())]
    for ruta, par in textos_:
        es, en = _palabras(par["es"]), _palabras(par["en"])
        if len(es) >= 4:                                           # «sube»/«goes up», «con contraejemplos»: demasiado cortas
            assert (set(es) & _SOLO_ES) and not (set(es) & (_SOLO_EN - _SOLO_ES - {"o", "y", "a", "en", "un", "es", "no", "lo"})), (ruta, "es")
            assert (set(en) & _SOLO_EN) and not (set(en) & (_SOLO_ES - _SOLO_EN - {"o", "y", "a", "en", "un", "es", "no", "lo", "de", "se", "la"})), (ruta, "en")


def test_las_dos_redacciones_piden_los_mismos_huecos():
    for clave, redaccion in MOTIVOS.items():
        assert set(redaccion) == {"es", "en"}
        assert huecos_de(redaccion["es"]) == huecos_de(redaccion["en"]), clave


def test_una_clave_que_no_esta_en_el_catalogo_revienta():
    with pytest.raises(KeyError):
        motivo("garantiza_que_todo_va_bien")


def test_cada_clave_del_catalogo_se_usa():
    """Una frase que nadie dice no se traduce ni se barre: o se usa o sobra."""
    fuente = inspect.getsource(modulo)
    usadas = {c for c in MOTIVOS if f'"{c}"' in fuente} | set(CLAVES_DE_VEREDICTO.values()) | {"direccion_sube", "direccion_baja", CLAVE_DE_LOS_LIMITES}
    assert set(MOTIVOS) - usadas == set(), set(MOTIVOS) - usadas


# ---------------------------------------------------------------------------
# El documento
# ---------------------------------------------------------------------------

def test_el_documento_es_json_y_su_digest_depende_de_epsilon():
    for r in _todos_los_veredictos():
        texto = json.dumps(r.a_json(), allow_nan=False, ensure_ascii=False)       # sin NaN ni Infinity
        assert json.loads(texto)["version"] == VERSION_DEL_COMPORTAMIENTO
        assert re.fullmatch(r"[0-9a-f]{64}", r.digest())
    a = curvas([0.0, 0.5, 0.25, 1.0], eps=0.1)
    assert a.digest() == curvas([0.0, 0.5, 0.25, 1.0], eps=0.1).digest()
    assert a.digest() != curvas([0.0, 0.5, 0.25, 1.0], eps=0.3).digest()
    assert a.a_json()["tolerancia"]["texto"] == {"es": "10 %", "en": "10%"}


def test_curvas_de_longitud_distinta_a_la_rejilla_se_rechazan():
    with pytest.raises(EsquemaInvalido):
        monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0], [[0.0, 1.0]], ["a"], epsilon_relativo=0.1, escala=BINARIA,
                               medido_sobre="muestra_de_desarrollo", semilla=1)
    with pytest.raises(EsquemaInvalido):
        monotonia_desde_curvas("x", "sube", [0.0, 1.0, 1.0], [[0.0, 1.0, 2.0]], ["a"], epsilon_relativo=0.1, escala=BINARIA,
                               medido_sobre="muestra_de_desarrollo", semilla=1)         # rejilla no creciente


def test_la_fila_tipica_es_un_ice_de_una_fila():
    """El ICE sintético de «Probar una fila»: una fila, sin datos, `medido_sobre = fila_tipica`."""
    tipica = {"Presión arterial": "No", "edad_años": 45.0, "Fumador": "No", "Sexo": "F", "peso": 70.0}
    r = monotonia_en_rejilla("peso", "sube", [tipica], lambda fs: [0.4 + 0.001 * f["peso"] for f in fs], minimo=50.0, maximo=100.0,
                             puntos=11, filas_m=200, semilla=9, epsilon_relativo=0.01, escala=BINARIA, medido_sobre="fila_tipica")
    j = r.a_json()
    assert j["medido_sobre"] == {"clave": "fila_tipica", "semilla": 9, "filas_pedidas": 1}
    assert j["filas"]["comprobadas"] == 1 and r.veredicto == "sin_contraejemplos"   # efecto 0,05 > 0,01
