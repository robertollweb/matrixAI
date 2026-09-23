# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C5 — las fechas, cableadas en la preparación (medido antes: `benchmarks/fechas_114c5/`,
«mejora» en 4 de 5 conjuntos en un reparto temporal, núcleo `376b0fa`).

Premisa que se arregla (medida el 2026-09-21): una columna `fecha` ISO entraba categórica y una
fecha nueva salía `__desconocida__`. Con `con_fechas=True`, entra por sus variables."""
from __future__ import annotations

from matrixai.training.lo_visto import comprobar_lo_visto
from matrixai.training.preparacion import (CATEGORIA_DESCONOCIDA, PoliticaDePreparacion,
                                           ajustar_preparacion, nombre_de_indicador,
                                           transformar_fila)


def _filas(n=60):
    return [{"fecha": f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}", "x": float(i),
             "y": "si" if i % 2 else "no"} for i in range(n)]


def _politica(con_fechas=True, filas=None):
    return ajustar_preparacion(filas or _filas(), objetivo="y", columnas=["fecha", "x"],
                               admite_categoricas=True, admite_faltantes=False,
                               con_fechas=con_fechas)


def test_una_fecha_NUEVA_entra_por_sus_variables_y_no_como_desconocida():
    politica = _politica()
    fila = transformar_fila({"fecha": "2031-06-15", "x": 3.0}, politica)
    assert "fecha" not in fila and CATEGORIA_DESCONOCIDA not in fila.values()
    assert fila["fecha__anio"] == 2031.0 and fila["fecha__mes"] == 6.0
    assert fila["fecha__dia_de_la_semana"] == 6.0  # domingo
    assert fila[nombre_de_indicador("fecha__anio")] == 0.0


def test_sin_con_fechas_la_preparacion_es_la_de_siempre():
    """La Fase 0 no lo pide: su preparación tiene que seguir IDÉNTICA."""
    politica = _politica(con_fechas=False)
    assert [c.tipo for c in politica.columnas] == ["categorica", "numerica"]
    assert politica.derivadas_de_fecha == ()
    assert transformar_fila({"fecha": "2031-06-15", "x": 3.0}, politica)["fecha"] == CATEGORIA_DESCONOCIDA


def test_el_formulario_sigue_viendo_LA_FECHA_y_la_salida_sus_variables():
    politica = _politica()
    assert [(c.columna, c.tipo) for c in politica.columnas] == [("fecha", "fecha"), ("x", "numerica")]
    assert politica.columna("fecha").formato_de_fecha == "%Y-%m-%d"
    salida = politica.columnas_de_salida()
    assert "fecha" not in salida
    assert "fecha__dias_desde_1970" in salida and nombre_de_indicador("fecha__dias_desde_1970") in salida
    # Y lo que se declara es EXACTAMENTE lo que sale, en el mismo orden.
    assert tuple(transformar_fila({"fecha": "2024-03-03", "x": 1.0}, politica)) == salida


def test_una_fecha_ilegible_sale_FALTANTE_con_su_indicador_nunca_un_cero():
    politica = _politica()
    fila = transformar_fila({"fecha": "no es una fecha", "x": 1.0}, politica)
    derivadas = [c.columna for c in politica.derivadas_de_fecha]
    assert all(fila[nombre_de_indicador(d)] == 1.0 for d in derivadas)
    # Sin nativo, la mediana DE TRAIN, que no es cero (el año medio es 2024).
    assert fila["fecha__anio"] == politica.derivadas_de(politica.columna("fecha"))[0].mediana == 2024.0


def test_la_politica_va_y_vuelve_por_json_con_sus_fechas():
    politica = _politica()
    vuelta = PoliticaDePreparacion.desde_json(politica.a_json())
    assert vuelta == politica
    fila = {"fecha": "2025-12-31", "x": 2.0}
    assert transformar_fila(fila, vuelta) == transformar_fila(fila, politica)


def test_una_politica_ANTERIOR_sin_fechas_se_sigue_leyendo():
    payload = _politica(con_fechas=False).a_json()
    payload.pop("derivadas_de_fecha")
    for c in payload["columnas"]:
        c.pop("formato_de_fecha")
    assert PoliticaDePreparacion.desde_json(payload).derivadas_de_fecha == ()


def test_lo_visto_dice_que_la_fecha_NO_se_comprobo_en_vez_de_afirmar_dentro():
    politica = _politica()
    lo_visto = comprobar_lo_visto({"fecha": "2031-06-15", "x": 3.0}, politica)
    assert "fecha" in lo_visto.sin_rango and lo_visto.dentro_de_lo_visto is None
    vacia = comprobar_lo_visto({"fecha": "", "x": 3.0}, politica)
    assert [m.tipo for m in vacia.marcas] == ["faltante_nunca_visto"]
