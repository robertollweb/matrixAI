"""ceros2 (10-11): la preparación DECLARA qué columnas emite como categoría, y `ProblemSpec` lo lleva al motor.

El defecto (medido con un estudio real, PLACSP 600 filas y uno sintético): con la decisión A de Roberto (09-11) un código
«03», «09» se queda como TEXTO y `ajustar_preparacion` lo dice categórico, pero los motores tipaban otra vez por el
CONTENIDO (`tipos_de_columna.tipificar`): «03» parsea, la columna salía numérica en el pliegue y la categoría no vista de la
validación (`__desconocida__`) reventaba lightgbm, hgb y tabm. El arreglo: la decisión es de la preparación
(`PoliticaDePreparacion.categoricas_de_salida`), viaja en `ProblemSpec.predictores_categoricos` y el motor la obedece.
"""
from __future__ import annotations

import pytest

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import ProblemSpec
from matrixai.training.preparacion import ajustar_preparacion, tipar_columnas_numericas


def _filas():
    # `cpv`: un código con ceros (decisión A: se queda texto); `n`: un número que llega como texto (el arreglo
    # original de `tipar_columnas_numericas`: se convierte); `organo`: texto de verdad; `fecha`: sus variables.
    return [{"cpv": c, "n": n, "organo": o, "fecha": f, "y": y} for c, n, o, f, y in [
        ("03", "1.5", "Ayto. A", "2024-01-03", 1.0), ("45", "2", "Ayto. B", "2024-02-03", 2.0),
        ("09", "", "Ayto. A", "2024-03-03", 3.0), ("45", "7.25", "", "2024-04-03", 4.0)]]


def _politica():
    filas = _filas()
    codigos: dict = {}
    tipar_columnas_numericas(filas, ["cpv", "n", "organo"], conservar_ceros_a_la_izquierda=True,
                             codigos_con_ceros=codigos)
    assert set(codigos) == {"cpv"} and filas[0]["cpv"] == "03" and filas[0]["n"] == 1.5
    return ajustar_preparacion(filas, objetivo="y", columnas=["cpv", "n", "organo", "fecha"],
                               admite_categoricas=True, admite_faltantes=True, con_fechas=True)


def test_categoricas_de_salida_son_las_que_transformar_fila_emite_como_categoria():
    politica = _politica()
    assert politica.categoricas_de_salida() == ("cpv", "organo")
    # todas están en lo que sale, en su orden; ni la numérica, ni su indicador, ni las variables de la fecha
    salida = politica.columnas_de_salida()
    assert [c for c in salida if c in politica.categoricas_de_salida()] == list(politica.categoricas_de_salida())
    assert "n" not in politica.categoricas_de_salida() and "n__faltante" not in politica.categoricas_de_salida()
    assert not any(c.startswith("fecha") for c in politica.categoricas_de_salida())


def test_un_indicador_que_pisa_una_categorica_del_mismo_nombre_no_se_declara_categorica():
    # `transformar_fila` escribe en un dict: el indicador de `x` (numérica) PISA la columna real `x__faltante`
    # (categórica) si va después. Lo que sale es el número, y declararlo categórico sería mentir sobre lo que sale.
    filas = [{"x__faltante": "a", "x": 1.0, "y": 1.0}, {"x__faltante": "b", "x": 2.0, "y": 2.0}]
    pisa = ajustar_preparacion(filas, objetivo="y", columnas=["x__faltante", "x"],
                               admite_categoricas=True, admite_faltantes=True)
    assert "x__faltante" in pisa.columnas_de_salida() and pisa.categoricas_de_salida() == ()
    no_pisa = ajustar_preparacion(filas, objetivo="y", columnas=["x", "x__faltante"],
                                  admite_categoricas=True, admite_faltantes=True)
    assert no_pisa.categoricas_de_salida() == ("x__faltante",)


def _spec(**extra):
    return ProblemSpec(problem_id="p", target="y", task="regression", observation_unit="una fila",
                       predictors=("cpv", "n"), **extra)


def test_problem_spec_lleva_las_categoricas_y_rechaza_una_que_no_es_predictor():
    assert _spec().predictores_categoricos == ()
    assert _spec(predictores_categoricos=["cpv"]).predictores_categoricos == ("cpv",)
    with pytest.raises(EsquemaInvalido) as error:
        _spec(predictores_categoricos=("otra",))
    assert error.value.clave == "categorico_de_desconocido"


def test_las_categoricas_no_cambian_el_problema_confirmado_ni_su_digest():
    # Es lo que VE el motor, no parte del problema: fuera del JSON y del digest (como `migracion`).
    sin, con = _spec(), _spec(predictores_categoricos=("cpv",))
    assert sin.a_json() == con.a_json() and sin.digest() == con.digest()
    assert "predictores_categoricos" not in ProblemSpec.CLAVES
