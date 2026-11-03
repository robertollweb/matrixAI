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

149-C1-BIS: el veredicto es por FRACCIÓN de filas (q) y la puerta es la AMPLITUD, no el efecto con signo.
Las pruebas del C1 que daban por hecho «una fila basta» llevan `q=UNA_FILA_BASTA` (0,01: una fila de hasta
100) para CONSERVAR su intención —retroceso, desempates, huecos, ε— sin depender de que 1 de 2 filas pase
de un q por casualidad; las del veredicto nuevo están al final del fichero, con sus curvas a mano.
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
    TOLERANCIA_POR_OMISION,
    VERSION_DEL_COMPORTAMIENTO,
    VEREDICTOS,
    EscalaDeLaSalida,
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
    tolerancia_por_omision,
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
REGRESION_100 = escala_de_la_salida("regression", amplitud_p90_p10=100.0)    # ε 5 % = 5 unidades
UNA_FILA_BASTA = 0.01          # q: con menos de 100 filas comprobadas, UNA fila violadora llega a q (el veredicto del C1)


def curvas(*cs, direccion="sube", eps=0.1, q=UNA_FILA_BASTA, escala=BINARIA, ids=None, k=5, motor=None):
    ids = list(ids) if ids is not None else [f"f{i}" for i in range(len(cs))]
    puntos = len(cs[0]) if cs else 4
    return monotonia_desde_curvas("x", direccion, [float(j) for j in range(puntos)], list(cs), ids,
                                  epsilon_relativo=eps, q_relativo=q, escala=escala,
                                  medido_sobre="muestra_de_desarrollo", semilla=3, k=k, motor_de_arboles=motor)


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


def test_los_empates_del_retroceso_se_resuelven_por_el_primer_punto():
    # El docstring lo promete «para que dos ejecuciones den el mismo contraejemplo». Meseta en el máximo
    # (j=1 y j=2) y meseta en la caída (j=3 y j=4): a es el PRIMER máximo y b la PRIMERA caída. Con `>=`
    # en cualquiera de las dos comparaciones el contraejemplo se desplaza (sabotaje del supervisor, 02-11).
    r = retroceso_de_una_curva([0.0, 0.5, 0.5, 0.25, 0.25], "sube")
    assert (r.valor, r.a, r.b) == (0.25, 1, 3)
    r = retroceso_de_una_curva([1.0, 0.5, 0.5, 0.75, 0.75], "baja")
    assert (r.valor, r.a, r.b) == (0.25, 1, 3)


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
    assert j["amplitud_tipica"] == 0.875
    assert "0,875" in j["veredicto"]["lectura"]["es"] and "0.875" in j["veredicto"]["lectura"]["en"]   # I5: la amplitud al lado


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
            assert r.a_json()["veredicto"]["clave_del_motivo"] == "amplitud_no_distingue"
            assert r.filas_con_violacion == 0 and r.efecto_total == 0.0


def test_una_amplitud_que_no_supera_la_tolerancia_no_distingue():
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

    kw = dict(minimo=20.0, maximo=80.0, puntos=13, filas_m=5, semilla=2, escala=BINARIA, medido_sobre="muestra_de_desarrollo",
              q_relativo=UNA_FILA_BASTA)
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
    # `avisos`: en binaria el aviso de «tolerancia alta» sale con ε >= la de partida (0,5 lo es, 0,01 no): depende de ε.
    assert cambian == {"tolerancia", "filas_con_violacion", "fraccion_con_violacion", "contraejemplos", "veredicto", "avisos"}
    assert [x["clave"] for x in a["avisos"]] == [] and [x["clave"] for x in b["avisos"]] == ["tolerancia_alta_en_binaria"]
    assert a["para_recalcular"] == b["para_recalcular"] and a["efecto_total"] == b["efecto_total"]
    assert a["amplitud_tipica"] == b["amplitud_tipica"]


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


def test_la_tolerancia_por_omision_es_la_medida_por_tarea_y_un_epsilon_negativo_no_existe():
    """ε y q por omisión son los de la MEDICIÓN (149-C0-ter), por tarea, en UNA tabla: ningún otro número escrito. Pasar
    otros los cambia; un ε negativo no existe."""
    for f in (monotonia_en_rejilla, monotonia_desde_curvas):
        parametros = inspect.signature(f).parameters
        assert parametros["epsilon_relativo"].default is None and parametros["q_relativo"].default is None
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
    # 149-C1-bis: por fracción (CON y SIN con filas violadoras), por amplitud, con la tolerancia de partida de cada tarea
    docs += [_con_violadoras(1, 20, 0.05), _con_violadoras(1, 20, 0.2), curvas(*([_CRECIENTE] * 5), eps=0.2, q=None),
             monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [[0.0, 50.0, 43.0, 100.0]] + [[0.0, 25.0, 50.0, 100.0]] * 19,
                                    list(range(20)), escala=REGRESION_100, medido_sobre="muestra_de_desarrollo", semilla=3),
             monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [[0.0, 0.5, 0.25, 1.0]] * 3 + [_CRECIENTE] * 17,
                                    list(range(20)), escala=BINARIA, medido_sobre="muestra_de_desarrollo", semilla=3)]
    # remate del C1-bis: las cuatro redacciones del aviso de binaria, el motivo con filas que se mueven, una fila sin curva y un NaN
    docs += [curvas(_CRECIENTE, eps=0.2, q=0.2, motor=True), curvas(_CRECIENTE, eps=0.2, q=0.2, motor=False),
             curvas(_CRECIENTE, eps=0.3, q=0.2, motor=True), curvas(_CRECIENTE, eps=0.3, q=0.1, motor=None),
             curvas(*([[1.0, 0.66, 0.33, 0.0]] * 4 + [[0.5] * 4] * 6), eps=0.25, q=0.2), _con_violadoras(3, 20, 0.2, eps=0.2),
             monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [None, _CRECIENTE, [0.0, float("nan"), 0.5, 1.0]], ["a", "b", "c"],
                                    escala=BINARIA, epsilon_relativo=0.1, q_relativo=0.1, medido_sobre="muestra_de_desarrollo", semilla=3)]
    for malo in (escala_de_la_salida("binary_classification", pendiente_de_la_recalibracion=-1.0),
                 escala_de_la_salida("regression", amplitud_p90_p10=0.0), escala_de_la_salida("multiclass_classification")):
        docs.append(monotonia_en_rejilla("x", "baja", [{"x": 1.0}], _no_debe_predecir, minimo=0.0, maximo=1.0, puntos=5,
                                         filas_m=1, semilla=1, epsilon_relativo=0.01, escala=malo,
                                         medido_sobre="fila_tipica"))
    docs.append(monotonia_en_rejilla("x", "sube", [{"x": 1.0}], _no_debe_predecir, minimo=None, maximo=None, puntos=5, filas_m=1,
                                     semilla=1, epsilon_relativo=0.01, escala=BINARIA, medido_sobre="muestra_de_desarrollo"))
    return docs


def test_cada_clave_de_veredicto_de_los_documentos_de_la_batería_aparece():
    """La batería de arriba tiene que tocar las tres palabras y los dos motivos del C1-bis; si no, el barrido no los ve."""
    claves = {r.a_json()["veredicto"]["clave"] for r in _todos_los_veredictos()}
    motivos = {r.a_json()["veredicto"]["clave_del_motivo"] for r in _todos_los_veredictos()}
    avisos = {a["clave"] for r in _todos_los_veredictos() for a in r.a_json()["avisos"]}
    assert claves == set(VEREDICTOS)
    assert {"amplitud_no_distingue", "salida_no_numerica", "sin_filas", "sin_prediccion"} <= motivos
    assert "tolerancia_alta_en_binaria" in avisos
    textos_es = [a["texto"]["es"] for r in _todos_los_veredictos() for a in r.a_json()["avisos"] if a["clave"] == "tolerancia_alta_en_binaria"]
    for trozo in ("a propósito: al medirlo", "se fijó midiendo", "Al medir, con modelos"):
        assert any(trozo in t for t in textos_es), trozo
    assert any(t.endswith("de la probabilidad.") and "partida" not in t and "medir" not in t for t in textos_es)


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


# ---------------------------------------------------------------------------
# 149-C1-bis — el veredicto V'(ε, q): AMPLITUD primero, después la FRACCIÓN de filas
# (decidido por Roberto el 03-11, `149-veredicto`; regla medida en `benchmarks/c149/REGLA_DE_ELECCION_TER.md`).
# Curvas a mano en binario exacto (escala 1,0; ε 0,25): la fila violadora retrocede 0,5 y la buena no retrocede.
# ---------------------------------------------------------------------------

SIN, CON, NSC = VEREDICTOS
_CRECIENTE = [0.0, 0.25, 0.5, 1.0]          # amplitud 1,0 y retroceso 0
_VIOLADORA = [0.0, 1.0, 0.5, 1.0]           # amplitud 1,0 y retroceso 0,5 (máximo corrido 1,0 en j=1, cae a 0,5 en j=2)


def _con_violadoras(violadoras, n, q, eps=0.25, direccion="sube"):
    return curvas(*([_VIOLADORA] * violadoras + [_CRECIENTE] * (n - violadoras)), eps=eps, q=q, direccion=direccion)


def test_la_fraccion_justo_en_q_es_con_contraejemplos():
    """`>=` en la fracción: 1 de 20 es el 5 % y q = 5 % pide «al menos» el 5 %. Un poco más de q, ya no llega."""
    assert _con_violadoras(1, 20, 0.05).veredicto == CON
    assert _con_violadoras(1, 20, 0.06).veredicto == SIN
    assert _con_violadoras(2, 20, 0.10).veredicto == CON and _con_violadoras(2, 20, 0.11).veredicto == SIN
    assert _con_violadoras(4, 20, 0.20).veredicto == CON and _con_violadoras(3, 20, 0.20).veredicto == SIN
    r = _con_violadoras(1, 20, 0.05)
    assert (r.filas_con_violacion, r.filas_comprobadas, r.fraccion_con_violacion) == (1, 20, 0.05)


def test_la_fraccion_justo_en_q_no_se_pierde_por_la_coma_flotante():
    """0,07 · 100 = 7,000000000000001 en coma flotante: una comparación `violadoras >= q * filas` diría que 7 de 100 NO llega
    al 7 %. La comparación es exacta (decimal contra racional)."""
    assert 0.07 * 100 > 7
    assert _con_violadoras(7, 100, 0.07).veredicto == CON
    assert _con_violadoras(6, 100, 0.07).veredicto == SIN
    assert _con_violadoras(100, 100, 1.0).veredicto == CON and _con_violadoras(99, 100, 1.0).veredicto == SIN


def test_una_fila_no_basta_si_q_pide_mas():
    """Antes del C1-bis una sola fila decidía. Con q = 20 %, 1 de 20 filas violadoras NO es «con contraejemplos» (y 1 de 1 sí)."""
    r = _con_violadoras(1, 20, 0.2)
    assert r.veredicto == SIN and r.filas_con_violacion == 1
    assert curvas(_VIOLADORA, eps=0.25, q=0.2).veredicto == CON
    # q ES un parámetro: el mismo cálculo, otro q, otro veredicto, y el documento dice cuál usó.
    assert r.a_json()["tolerancia"]["fraccion_minima"] == {"relativa": 0.2, "texto": {"es": "20 %", "en": "20%"}}


def test_la_amplitud_justo_en_epsilon_no_distingue():
    """`>` y no `>=` en la amplitud: una curva que se mueve EXACTAMENTE lo que tolera la prueba no distingue nada."""
    c = [0.0, 0.125, 0.25, 0.25]                                     # monótona, amplitud 0,25
    r = curvas(c, c, eps=0.25)
    assert r.amplitud_tipica == 0.25 and r.epsilon_absoluto == 0.25
    assert r.veredicto == NSC and r.a_json()["veredicto"]["clave_del_motivo"] == "amplitud_no_distingue"
    assert curvas(c, c, eps=0.2499).veredicto == SIN


def test_la_amplitud_se_mira_antes_que_la_fraccion():
    """Una columna plana con UNA fila ruidosa (1 de 20 = 5 % = q) no es «con contraejemplos»: la curva típica no se mueve, y la
    prueba no distingue. Si se contara primero la fracción saldría CON."""
    plana, ruido = [0.5] * 4, [0.5, 1.0, 0.5, 0.5]                   # el ruido retrocede 0,5 > ε
    r = curvas(*([plana] * 19 + [ruido]), eps=0.25, q=0.05)
    assert (r.filas_con_violacion, r.fraccion_con_violacion) == (1, 0.05)       # la fracción SÍ llegaba a q
    assert r.amplitud_tipica == 0.0
    assert r.veredicto == NSC and r.a_json()["veredicto"]["clave_del_motivo"] == "amplitud_no_distingue"


def test_la_amplitud_no_tiene_signo_ni_depende_de_los_extremos():
    """Una U que acaba donde empezó (efecto 0) y una columna monótona declarada AL REVÉS (efecto negativo) se mueven mucho: son
    los dos contraejemplos más claros, y con el efecto con signo salían «no se pudo comprobar» (los dos errores del C0-bis)."""
    u = [0.5, 0.0, 0.0, 0.5]
    r = curvas(u, u, eps=0.25, q=0.05)
    assert r.efecto_total == 0.0 and r.amplitud_tipica == 0.5
    assert r.veredicto == CON
    al_reves = curvas(_CRECIENTE, _CRECIENTE, direccion="baja", eps=0.25, q=0.05)
    assert al_reves.efecto_total == -1.0 and al_reves.amplitud_tipica == 1.0
    assert al_reves.veredicto == CON
    assert curvas(_CRECIENTE, _CRECIENTE, direccion="sube", eps=0.25, q=0.05).veredicto == SIN       # y la buena, sin contraejemplos


def test_la_amplitud_de_la_columna_es_la_mediana_de_las_filas_no_la_media_ni_el_maximo():
    plana, a = [0.5] * 4, [0.0, 0.1, 0.2, 0.3]
    # amplitudes 0, 0, 0, 0, 1: mediana 0 (el máximo, 1, la haría pasar; la media, 0,2, tampoco la distingue)
    r = curvas(plana, plana, plana, plana, _CRECIENTE, eps=0.1)
    assert r.amplitud_tipica == 0.0 and r.veredicto == NSC
    # amplitudes 0, 0, 0,3, 0,3, 0,3: mediana 0,3 > 0,2 pasa la puerta; la media (0,18) no la pasaría
    r = curvas(plana, plana, a, a, a, eps=0.2)
    assert r.amplitud_tipica == 0.3 and statistics.fmean(r.amplitudes) < 0.2
    assert r.veredicto == SIN
    # y un hueco no cuenta como una curva plana: ni en la mediana ni en las filas comprobadas
    r = curvas(a, [None, 0.2, 0.3, 0.4], a, a, eps=0.2)
    assert r.amplitudes[1] is None and r.amplitud_tipica == 0.3 and r.veredicto == SIN


def test_los_contraejemplos_solo_se_enseñan_si_el_veredicto_es_con_contraejemplos():
    """Una o dos filas por debajo de q no son «contraejemplos» bajo un «sin contraejemplos»; siguen en `para_recalcular`."""
    sin = _con_violadoras(1, 20, 0.2)
    assert sin.veredicto == SIN and sin.contraejemplos == () and sin.a_json()["contraejemplos"] == []
    assert len(sin.a_json()["para_recalcular"]["peores"]) == 1
    con = sin.con_tolerancia(q_relativo=0.05)
    assert con.veredicto == CON and [c["fila"] for c in con.contraejemplos] == ["f0"]


def test_con_tolerancia_cambia_q_sin_volver_a_predecir_y_el_documento_lo_dice():
    llamadas = []

    def puntuar(filas):
        llamadas.append(len(filas))
        return [(1.0 if (f["x"] == 1.0 and f["i"] < 2) else 0.0) + f["x"] / 4 for f in filas]     # 2 de 10 filas con un pico en x=1 (retroceso 0,75)

    filas = [{"x": 0.0, "i": i} for i in range(10)]
    kw = dict(minimo=0.0, maximo=3.0, puntos=4, filas_m=10, semilla=1, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    exigente = monotonia_en_rejilla("x", "sube", filas, puntuar, epsilon_relativo=0.1, q_relativo=0.3, **kw)
    assert llamadas == [40] and exigente.fraccion_con_violacion == 0.2 and exigente.veredicto == SIN
    flojo = exigente.con_tolerancia(q_relativo=0.2)
    assert llamadas == [40] and flojo.veredicto == CON                       # I7: ni una predicción más
    desde_cero = monotonia_en_rejilla("x", "sube", filas, puntuar, epsilon_relativo=0.1, q_relativo=0.2, **kw)
    assert flojo.a_json() == desde_cero.a_json() and flojo.digest() == desde_cero.digest()
    assert flojo.digest() != exigente.digest()                                # q entra en el digest
    assert flojo.a_json()["tolerancia"]["fraccion_minima"]["relativa"] == 0.2
    assert flojo.con_tolerancia(0.5, None).q_relativo == 0.2 and flojo.con_epsilon(0.5).q_relativo == 0.2   # None = conservar


@pytest.mark.parametrize("malo", [0.0, -0.1, 1.5, float("nan"), True, "0.1", None])
def test_q_fuera_de_0_1_se_rechaza(malo):
    """q = 0 haría que cualquier columna tuviera contraejemplos (0 filas ≥ 0 %). `None` no es un q: pide el de la tarea."""
    if malo is None:
        assert curvas(_CRECIENTE, q=None).q_relativo == 0.2                    # el de partida de binaria
        with pytest.raises(EsquemaInvalido):
            curvas(_CRECIENTE).con_tolerancia(q_relativo=0.0)
        return
    with pytest.raises(EsquemaInvalido):
        curvas(_CRECIENTE, q=malo)
    with pytest.raises(EsquemaInvalido):
        curvas(_CRECIENTE).con_tolerancia(q_relativo=malo)


def test_los_valores_por_omision_son_los_medidos_por_tarea():
    """149-C0-ter: regresión ε 5 %, q 5 %; binaria ε 20 %, q 20 %; multiclase ninguno. Una sola tabla."""
    assert dict(TOLERANCIA_POR_OMISION) == {"regression": (0.05, 0.05), "binary_classification": (0.20, 0.20)}
    assert tolerancia_por_omision("regression") == (0.05, 0.05)
    assert tolerancia_por_omision("binary_classification") == (0.20, 0.20)
    assert tolerancia_por_omision("multiclass_classification") is None
    with pytest.raises(EsquemaInvalido):
        tolerancia_por_omision("ordinal")
    with pytest.raises(TypeError):
        TOLERANCIA_POR_OMISION["regression"] = (0.0, 0.0)                    # no se reescribe de pasada


def test_sin_pasarlos_cada_tarea_usa_los_suyos_y_no_los_de_la_otra():
    # Binaria (ε 20 %, q 20 %): 3 de 20 filas retroceden 0,25 > 0,2 -> 15 % < 20 %: SIN. Con los de REGRESIÓN (5 %, 5 %) saldría CON.
    binaria = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [[0.0, 0.5, 0.25, 1.0]] * 3 + [_CRECIENTE] * 17,
                                     list(range(20)), escala=BINARIA, medido_sobre="muestra_de_desarrollo", semilla=3)
    assert (binaria.epsilon_relativo, binaria.q_relativo, binaria.filas_con_violacion) == (0.2, 0.2, 3)
    assert binaria.veredicto == SIN
    # Regresión (ε 5 %, q 5 %) con escala 100: 1 de 20 filas retrocede 7 > 5 -> 5 % >= 5 %: CON. Con los de BINARIA (20 %) saldría SIN.
    regresion = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [[0.0, 50.0, 43.0, 100.0]] + [[0.0, 25.0, 50.0, 100.0]] * 19,
                                       list(range(20)), escala=REGRESION_100, medido_sobre="muestra_de_desarrollo", semilla=3)
    assert (regresion.epsilon_relativo, regresion.q_relativo, regresion.epsilon_absoluto) == (0.05, 0.05, 5.0)
    assert regresion.veredicto == CON
    # el documento dice que son los de partida; con otros, no
    assert binaria.a_json()["tolerancia"]["por_omision"] is True and regresion.a_json()["tolerancia"]["por_omision"] is True
    assert curvas(_CRECIENTE, eps=0.2, q=0.2).a_json()["tolerancia"]["por_omision"] is True
    assert curvas(_CRECIENTE, eps=0.2, q=0.1).a_json()["tolerancia"]["por_omision"] is False
    assert regresion.con_epsilon(0.2).a_json()["tolerancia"]["por_omision"] is False
    # ε sin q (o q sin ε): lo que falta es lo de la tarea
    solo_eps = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [_CRECIENTE], ["a"], epsilon_relativo=0.1, escala=BINARIA,
                                      medido_sobre="muestra_de_desarrollo", semilla=3)
    assert (solo_eps.epsilon_relativo, solo_eps.q_relativo) == (0.1, 0.2)


def test_multiclase_no_tiene_tolerancia_de_partida_y_no_se_inventa_una():
    mc = escala_de_la_salida("multiclass_classification")
    r = monotonia_en_rejilla("x", "sube", [{"x": 1.0}], _no_debe_predecir, minimo=0.0, maximo=1.0, puntos=5, filas_m=1, semilla=1,
                             escala=mc, medido_sobre="muestra_de_desarrollo")
    j = r.a_json()
    assert r.veredicto == NSC and j["veredicto"]["clave_del_motivo"] == "multiclase_sin_clase_declarada"
    assert j["tolerancia"]["relativa"] is None and j["tolerancia"]["texto"] is None and j["tolerancia"]["por_omision"] is False
    json.dumps(j, allow_nan=False)
    # una escala VÁLIDA de una tarea sin tolerancia de partida, sin dársela: falla ruidoso, no un número inventado
    inventada = EscalaDeLaSalida("multiclass_classification", 1.0, "escala_de_probabilidad", None)
    with pytest.raises(EsquemaInvalido):
        monotonia_desde_curvas("x", "sube", [0.0, 1.0], [[0.0, 1.0]], ["a"], escala=inventada, medido_sobre="muestra_de_desarrollo", semilla=1)
    ok = monotonia_desde_curvas("x", "sube", [0.0, 1.0], [[0.0, 1.0]], ["a"], escala=inventada, epsilon_relativo=0.1, q_relativo=0.1,
                                medido_sobre="muestra_de_desarrollo", semilla=1)
    assert ok.veredicto == SIN


def _aviso_de_tolerancia(r):
    avisos = [a for a in r.a_json()["avisos"] if a["clave"] == "tolerancia_alta_en_binaria"]
    assert len(avisos) <= 1
    return avisos[0]["texto"] if avisos else None


def test_en_binaria_el_documento_dice_que_la_tolerancia_de_partida_es_alta_y_por_que():
    """Con la tolerancia de partida y un motor de árboles: es alta, es la de partida y lo MEDIDO (columnas que no influyen,
    árboles, ~2.000 filas, 10–15 % de la probabilidad). Remate del C1-bis: ya no dice que «las curvas avanzan a escalones»."""
    binaria = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [_CRECIENTE] * 5, list(range(5)), escala=BINARIA,
                                     medido_sobre="muestra_de_desarrollo", semilla=3, motor_de_arboles=True)
    texto = _aviso_de_tolerancia(binaria)
    es, en = texto["es"], texto["en"]
    for trozo in ("alta", "20 % de la probabilidad", "de partida", "árboles (LightGBM y HGB)", "unas 2.000 filas", "no influyen",
                  "entre un 10 % y un 15 %"):
        assert trozo in es, trozo
    for trozo in ("high", "20% of the probability", "starting", "tree models (LightGBM and HGB)", "about 2,000 rows",
                  "no influence", "between 10% and 15%"):
        assert trozo in en, trozo
    for idioma, t in (("es", es), ("en", en)):
        assert palabras_prohibidas(t, idioma) == [], idioma               # sin «garantiza», «seguro», «robusto»…
        assert "escalon" not in t and "avanzan" not in t and "steps" not in t.replace("steps back", "")
    # no sale en regresión, ni en binaria con una tolerancia menor que la de partida, ni sin filas comprobadas
    assert _aviso_de_tolerancia(curvas(_CRECIENTE, eps=0.1, q=0.2)) is None
    reg = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [[0.0, 25.0, 50.0, 100.0]], ["a"], escala=REGRESION_100,
                                 medido_sobre="muestra_de_desarrollo", semilla=3)
    assert reg.a_json()["avisos"] == []
    assert curvas([None] * 4, eps=0.2, q=0.2).a_json()["avisos"] == []
    # con una tolerancia MAYOR sigue siendo alta
    assert [a["clave"] for a in curvas(_CRECIENTE, eps=0.3, q=0.2).a_json()["avisos"]] == ["tolerancia_alta_en_binaria"]


def test_el_aviso_de_binaria_solo_dice_que_es_la_de_partida_si_lo_usado_es_la_de_partida():
    """Hallazgo I2(a): con ε 30 % o 50 % —o con el ε de partida y otro q— el aviso decía «es la de partida» y era falso
    (`por_omision` False en el mismo documento)."""
    for eps, q in ((0.3, 0.2), (0.5, 0.2), (0.2, 0.1), (0.2, 0.5)):
        for motor in (True, False, None):
            r = curvas(_CRECIENTE, eps=eps, q=q, motor=motor)
            assert r.a_json()["tolerancia"]["por_omision"] is False
            t = _aviso_de_tolerancia(r)
            assert "partida" not in t["es"] and "starting" not in t["en"], (eps, q, motor)
            assert f"{round(eps * 100)} % de la probabilidad" in t["es"] and f"{round(eps * 100)}% of the probability" in t["en"]
    for motor in (True, False, None):
        r = curvas(_CRECIENTE, eps=0.2, q=0.2, motor=motor)
        assert r.a_json()["tolerancia"]["por_omision"] is True
        t = _aviso_de_tolerancia(r)
        assert "Es la de partida" in t["es"] and "It is the starting one" in t["en"], motor


def test_el_aviso_de_binaria_cita_la_medicion_de_los_arboles_solo_si_el_motor_es_de_arboles():
    """Hallazgo I2(b): lo medido (LightGBM y HGB, ~2.000 filas, columnas sin efecto, 10–15 %) NO se aplica a lineales ni a redes,
    que no se midieron: con un motor que no es de árboles —o que no se sabe— el aviso no afirma nada de la probabilidad de las
    columnas sin efecto. Con la de partida dice que se fijó con árboles y que con otros no se midió."""
    for eps, q in ((0.2, 0.2), (0.3, 0.2)):
        for motor in (False, None):
            t = _aviso_de_tolerancia(curvas(_CRECIENTE, eps=eps, q=q, motor=motor))
            for trozo in ("10 %", "15 %", "2.000", "no influyen", "movían"):
                assert trozo not in t["es"], (eps, motor, trozo)
            for trozo in ("10%", "15%", "2,000", "no influence", "moved"):
                assert trozo not in t["en"], (eps, motor, trozo)
        de_arboles = _aviso_de_tolerancia(curvas(_CRECIENTE, eps=eps, q=q, motor=True))
        assert "entre un 10 % y un 15 %" in de_arboles["es"] and "between 10% and 15%" in de_arboles["en"]
    partida_otro = _aviso_de_tolerancia(curvas(_CRECIENTE, eps=0.2, q=0.2, motor=False))
    assert "con otros motores no se ha medido" in partida_otro["es"] and "has not been measured with other engines" in partida_otro["en"]
    # y NINGUNA redacción del catálogo vuelve a generalizar «las curvas de los árboles avanzan a escalones»
    for clave, redaccion in MOTIVOS.items():
        assert "escalones" not in redaccion["es"] and "avanzan" not in redaccion["es"], clave
        assert "tree models move in steps" not in redaccion["en"] and "curves of tree" not in redaccion["en"], clave


@pytest.mark.parametrize("malo", ["si", 1, 0.5])
def test_motor_de_arboles_es_un_booleano_o_nada(malo):
    with pytest.raises(EsquemaInvalido):
        curvas(_CRECIENTE, motor=malo)


def test_el_documento_lleva_la_amplitud_la_fraccion_y_las_amplitudes_por_fila():
    r = _con_violadoras(1, 20, 0.05)
    j = r.a_json()
    assert j["version"] == VERSION_DEL_COMPORTAMIENTO == "2.0.0"
    assert j["amplitud_tipica"] == 1.0 and j["fraccion_con_violacion"] == 0.05
    assert j["para_recalcular"]["amplitudes"] == [1.0] * 20 and len(j["para_recalcular"]["retrocesos"]) == 20
    assert j["efecto_total"] == r.efecto_total                                  # el efecto sigue como DATO
    lectura = j["veredicto"]["lectura"]
    assert "5 % de las filas (1 de 20)" in lectura["es"] and "al menos el 5 %" in lectura["es"]       # «el 5 % de las filas retroceden…»
    assert "5% of the rows (1 of 20)" in lectura["en"] and "at least 5%" in lectura["en"]
    assert "1 entre su valor más bajo y el más alto" in lectura["es"]
    sin = _con_violadoras(1, 20, 0.2).a_json()["veredicto"]["lectura"]
    assert "por debajo del 20 %" in sin["es"] and "below the 20%" in sin["en"]


# ---------------------------------------------------------------------------
# 149-C1-bis, REMATE — lo que la auditoría Opus encontró (I1, I2, filas que retroceden, motivo falso, menores)
# ---------------------------------------------------------------------------

from fractions import Fraction                                   # noqa: E402
from decimal import Decimal                                      # noqa: E402

from matrixai.estudio.comportamiento import MonotoniaDeUnaColumna                      # noqa: E402
from matrixai.estudio.comportamiento_textos import porcentajes_de_la_lectura           # noqa: E402

_TABLA_DE_Q = (0.01, 0.05, 0.07, 0.1, 0.2, 0.25, 1 / 3, 0.5, 1.0)


def _numero(texto):
    """`«4,95 %»` / `«4.95%»` -> Decimal('4.95')."""
    return Decimal(re.sub(r"[^0-9,.]", "", texto).replace(",", "."))


def _coherente(v, n, q):
    """La cifra escrita de la fracción NO contradice al veredicto: < q si es menor, ≥ q si no (leyendo SOLO los textos)."""
    fraccion, minimo = porcentajes_de_la_lectura(v, n, q)
    es_menor = Fraction(v, n) < Fraction(repr(float(q)))
    for idioma in IDIOMAS:
        f, m = _numero(fraccion[idioma]), _numero(minimo[idioma])
        assert (f < m) if es_menor else (f >= m), (v, n, q, fraccion, minimo)
    return fraccion, minimo


def test_la_lectura_no_redondea_la_fraccion_hasta_contradecir_al_veredicto():
    """Hallazgo I1: `round(f, 3)` escribía «El 5 % de las filas (5 de 101) … por debajo del 5 % que pide». Tres casos del informe."""
    for v, n, q, esperado in ((5, 101, 0.05, "4,95 %"), (99, 2000, 0.05, "4,95 %"), (399, 2000, 0.2, "19,95 %")):
        r = _con_violadoras(v, n, q, eps=0.25)
        assert r.veredicto == SIN, (v, n)
        lectura = r.a_json()["veredicto"]["lectura"]
        assert lectura["es"].startswith(f"El {esperado} de las filas ({v} de {n})"), lectura["es"]
        assert f"por debajo del {round(q * 100)} %" in lectura["es"]
        assert lectura["en"].startswith(f"{esperado.replace(',', '.').replace(' %', '%')} of the rows ({v} of {n})")
    # y se sigue redondeando AL MÁS CERCANO a tres decimales (no se trunca): 2/3 = 66,6666… -> 66,667; 1/7 = 14,2857… -> 14,286
    assert _con_violadoras(2, 3, 0.5).a_json()["veredicto"]["lectura"]["es"].startswith("El 66,667 % de las filas (2 de 3)")
    assert _con_violadoras(1, 7, 0.1).a_json()["veredicto"]["lectura"]["en"].startswith("14.286% of the rows (1 of 7)")
    # el borde contrario: justo en q sigue siendo «el 5 %», y un poco por encima, nunca menos
    assert _con_violadoras(1, 20, 0.05).a_json()["veredicto"]["lectura"]["es"].startswith("El 5 % de las filas (1 de 20)")


def test_una_fraccion_que_con_tres_decimales_se_leeria_como_el_minimo_se_escribe_menor():
    """1.000 de 20.001 es el 4,99975 %: a tres decimales sale «5» (= q) y la frase diría «5 % … por debajo del 5 %». Se baja la
    cifra (4,999) y, si aun así empatara con el mínimo, se añaden decimales."""
    r = _con_violadoras(1000, 20001, 0.05)
    assert r.veredicto == SIN
    lectura = r.a_json()["veredicto"]["lectura"]
    assert lectura["es"].startswith("El 4,999 % de las filas (1000 de 20001)") and "por debajo del 5 %" in lectura["es"]
    assert lectura["en"].startswith("4.999% of the rows (1000 of 20001)")
    # con un q que no cabe en tres decimales: 1/3 = 33,333…; 33,3333 % de las filas por debajo del 33,3333… -> decimales de más
    fraccion, minimo = _coherente(3333333, 10000000, 1 / 3)
    assert fraccion["es"] != minimo["es"]


def test_barrido_coherencia_texto_veredicto_n_1_a_400_y_la_tabla_de_q():
    """Para cada n de 1 a 400 y cada q de la tabla, las filas violadoras de la vecindad del borde (donde el redondeo muerde) y
    los extremos: la cifra escrita y el mínimo escrito dicen LO MISMO que el veredicto."""
    cuentas = 0
    for n in range(1, 401):
        for q in _TABLA_DE_Q:
            borde = math_ceil_fraccion(n, q)
            for v in {0, n, borde - 2, borde - 1, borde, borde + 1}:
                if 0 <= v <= n:
                    _coherente(v, n, q)
                    cuentas += 1
    assert cuentas > 15000


def math_ceil_fraccion(n, q):
    return -((-n * Fraction(repr(float(q)))).__floor__())


def test_barrido_de_la_lectura_del_documento_con_los_q_de_la_tabla_para_n_1_a_400():
    """Lo mismo pero por el DOCUMENTO: el veredicto sale de la regla y la cifra, del texto de su `lectura` en castellano."""
    for n in range(1, 401):
        for q in (0.05, 0.2):
            borde = math_ceil_fraccion(n, q)
            for v in sorted({borde - 1, borde}):
                if not 0 <= v <= n:
                    continue
                r = _con_violadoras(v, n, q, eps=0.25)
                lectura = r.a_json()["veredicto"]["lectura"]["es"]
                m = re.match(r"El ([\d,]+) % de las filas \((\d+) de (\d+)\).*?(?:por debajo del |al menos el )([\d,]+) %", lectura)
                assert m, lectura
                f, minimo = _numero(m.group(1)), _numero(m.group(4))
                assert (f >= minimo) if r.veredicto == CON else (f < minimo), (n, v, q, lectura)
                assert r.veredicto == (CON if Fraction(v, n) >= Fraction(repr(q)) else SIN)


def test_bajo_sin_contraejemplos_las_filas_que_retroceden_se_ensenan_aparte_con_su_rotulo():
    """Decisión del supervisor: 3 de 20 filas retroceden 0,5 con ε = q = 20 %. El veredicto es «sin contraejemplos» y los
    `contraejemplos` siguen vacíos, pero las tres filas NO se esconden: `filas_que_retroceden`, con su rótulo."""
    r = _con_violadoras(3, 20, 0.2, eps=0.2)
    j = r.a_json()
    assert r.veredicto == SIN and j["contraejemplos"] == []
    assert [f["fila"] for f in j["filas_que_retroceden"]["filas"]] == ["f0", "f1", "f2"]
    assert all(f["retroceso"] == 0.5 for f in j["filas_que_retroceden"]["filas"])
    assert j["filas_que_retroceden"]["rotulo"] == {
        "es": "filas que retroceden más que la tolerancia, por debajo del mínimo que pide el veredicto",
        "en": "rows that go back by more than the tolerance, below the minimum the verdict asks for"}
    for idioma in IDIOMAS:
        assert "contraejemplo" not in j["filas_que_retroceden"]["rotulo"][idioma] and "counterexample" not in j["filas_que_retroceden"]["rotulo"][idioma]
    # con contraejemplos están en `contraejemplos` y NO se repiten aparte; sin filas que retrocedan (o sin comprobar), vacías
    con = _con_violadoras(4, 20, 0.2, eps=0.2).a_json()
    assert con["veredicto"]["clave"] == CON and len(con["contraejemplos"]) == 4 and con["filas_que_retroceden"]["filas"] == []
    assert curvas(_CRECIENTE, _CRECIENTE).a_json()["filas_que_retroceden"]["filas"] == []
    assert curvas([0.5] * 4).a_json()["filas_que_retroceden"]["filas"] == [] and curvas().a_json()["filas_que_retroceden"]["filas"] == []
    # son un dato del documento, no de la tolerancia: con otra q vuelven a ser contraejemplos y salen de aquí
    assert r.con_tolerancia(q_relativo=0.1).a_json()["filas_que_retroceden"]["filas"] == []
    assert len(r.con_tolerancia(q_relativo=0.1).a_json()["contraejemplos"]) == 3


def test_el_motivo_de_amplitud_dice_cuantas_filas_si_se_mueven_cuando_la_mediana_es_plana():
    """Hallazgo del informe: 40 de 100 filas claramente al revés y 60 planas -> mediana 0 -> «no se pudo comprobar» (la regla
    decidida se queda), pero el motivo ya no dice que la prueba «no distingue una dirección de la contraria» (falso para ese
    40 %): dice que en al menos la mitad la columna no mueve la salida y cuántas filas SÍ la mueven."""
    al_reves, plana = [1.0, 0.66, 0.33, 0.0], [0.5] * 4
    r = curvas(*([al_reves] * 40 + [plana] * 60), eps=0.25, q=0.2)
    j = r.a_json()["veredicto"]
    assert r.veredicto == NSC and r.amplitud_tipica == 0.0 and r.filas_con_violacion == 40
    assert j["clave_del_motivo"] == "amplitud_no_distingue"
    es, en = j["motivo"]["es"], j["motivo"]["en"]
    assert "40 de 100" in es and "al menos la mitad de las filas" in es and "no distingue una dirección de la contraria" not in es
    assert "40 of 100" in en and "at least half of the rows" in en and "cannot tell one direction" not in en
    # y si NINGUNA fila se mueve, sí es la frase de siempre, con las filas comprobadas
    plano = curvas(*([plana] * 5), eps=0.25, q=0.2).a_json()["veredicto"]["motivo"]
    assert "ninguna de las filas comprobadas (5)" in plano["es"] and "no distingue una dirección de la contraria" in plano["es"]
    assert "none of the rows checked (5)" in plano["en"]


def test_una_fila_sin_curva_no_revienta_y_cuenta_como_sin_prediccion():
    """Menor: una curva `None` ENTERA (no una lista con huecos) era un `TypeError` en `len(None)`. Es una fila sin predicción."""
    r = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0, 3.0], [None, _CRECIENTE, _VIOLADORA, None], ["a", "b", "c", "d"],
                               escala=BINARIA, epsilon_relativo=0.25, q_relativo=0.5, medido_sobre="muestra_de_desarrollo", semilla=3)
    assert r.retrocesos == (None, 0.0, 0.5, None) and r.efectos[0] is None and r.amplitudes[3] is None
    assert r.filas_comprobadas == 2 and r.veredicto == CON
    j = r.a_json()
    assert j["filas"] == {"pedidas": 4, "comprobadas": 2, "sin_prediccion": 2, "ids": ["a", "b", "c", "d"]}
    assert j["avisos"][0]["clave"] == "filas_sin_prediccion"
    todas = monotonia_desde_curvas("x", "sube", [0.0, 1.0], [None, None], ["a", "b"], escala=BINARIA, epsilon_relativo=0.1,
                                   q_relativo=0.1, medido_sobre="muestra_de_desarrollo", semilla=3)
    assert todas.veredicto == NSC and todas.a_json()["veredicto"]["clave_del_motivo"] == "sin_prediccion"
    assert todas.a_json()["amplitud_tipica"] is None and todas.a_json()["fraccion_con_violacion"] is None


@pytest.mark.parametrize("malo", [float("nan"), float("inf"), float("-inf")])
def test_una_curva_con_nan_o_inf_excluye_la_fila_y_no_envenena_la_amplitud(malo):
    """Menor: el guion `amplitud_de_curvas` solo descartaba `None` y un NaN entraba en la mediana; el núcleo EXCLUYE la fila (como
    un hueco): ni retroceso ni amplitud, se cuenta aparte y el documento sigue siendo JSON estricto."""
    r = curvas([0.0, malo, 0.5, 1.0], _CRECIENTE, _CRECIENTE, _VIOLADORA, eps=0.25, q=0.5)
    assert r.retrocesos[0] is None and r.amplitudes[0] is None and r.efectos[0] is None
    assert r.filas_comprobadas == 3 and r.amplitud_tipica == 1.0
    j = r.a_json()
    json.dumps(j, allow_nan=False)
    assert j["filas"]["sin_prediccion"] == 1 and j["avisos"][0]["clave"] == "filas_sin_prediccion"
    assert curvas([0.0, malo, 0.5, 1.0], [malo] * 4).veredicto == NSC


def _ida_y_vuelta(r):
    return MonotoniaDeUnaColumna.desde_json(json.loads(json.dumps(r.a_json(), allow_nan=False)))


def test_el_documento_se_carga_desde_json_y_se_reevalua_sin_reescribir_la_regla():
    """C3a lee `comportamiento.json` y tiene que poder reevaluarlo con otra tolerancia SIN copiar la regla al Studio:
    a_json -> JSON -> `desde_json` -> `con_tolerancia` da el MISMO documento y el MISMO digest que construirlo desde cero."""
    for r in _todos_los_veredictos():
        vuelta = _ida_y_vuelta(r)
        assert vuelta.a_json() == r.a_json() and vuelta.digest() == r.digest()
        for eps, q in ((0.05, 0.05), (0.2, 0.2), (0.4, 0.6)):
            if r.escala.valor is not None:
                assert vuelta.con_tolerancia(eps, q).digest() == r.con_tolerancia(eps, q).digest()
    filas = [{"x": 0.0, "i": i} for i in range(10)]
    kw = dict(minimo=0.0, maximo=3.0, puntos=4, filas_m=10, semilla=1, escala=BINARIA, medido_sobre="muestra_de_desarrollo")
    puntuar = lambda fs: [(1.0 if (f["x"] == 1.0 and f["i"] < 2) else 0.0) + f["x"] / 4 for f in fs]      # noqa: E731
    base = monotonia_en_rejilla("x", "sube", filas, puntuar, epsilon_relativo=0.1, q_relativo=0.3, motor_de_arboles=True, **kw)
    cargado = _ida_y_vuelta(base)
    assert cargado.motor_de_arboles is True and cargado.ids == base.ids
    otro = monotonia_en_rejilla("x", "sube", filas, puntuar, epsilon_relativo=0.1, q_relativo=0.2, motor_de_arboles=True, **kw)
    assert cargado.con_tolerancia(q_relativo=0.2).digest() == otro.digest() != base.digest()
    assert cargado.con_epsilon(0.5).a_json() == base.con_epsilon(0.5).a_json()


def test_desde_json_rechaza_lo_que_no_entiende():
    j = curvas(_CRECIENTE, _VIOLADORA, eps=0.25, q=0.5).a_json()
    cargar = lambda doc: MonotoniaDeUnaColumna.desde_json(json.loads(json.dumps(doc)))        # noqa: E731
    cargar(j)
    for roto in (dict(j, version="1.0.0"), {k: v for k, v in j.items() if k != "para_recalcular"},
                 dict(j, para_recalcular=dict(j["para_recalcular"], retrocesos=[0.0])),
                 dict(j, direccion="diagonal"), dict(j, tolerancia=dict(j["tolerancia"], relativa=-1.0)),
                 dict(j, medido_sobre=dict(j["medido_sobre"], clave="prueba_reservada"))):
        with pytest.raises(EsquemaInvalido):
            cargar(roto)
    with pytest.raises(EsquemaInvalido):
        MonotoniaDeUnaColumna.desde_json([])
