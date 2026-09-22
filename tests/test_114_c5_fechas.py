# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C5 — las fechas como variables (`matrixai.training.fechas`)."""
from __future__ import annotations

import datetime

import pytest

from matrixai.training.fechas import (
    REFERENCIA,
    VARIABLES,
    VARIABLES_CON_HORA,
    RecetaDeFecha,
    RecetaDeFechaIncompatible,
    detectar_fechas,
    expandir_fila,
    fechas_ambiguas,
    variables_de_fecha,
)
from matrixai.training.preparacion import (
    ajustar_preparacion,
    tipar_columnas_numericas,
    transformar_fila,
)


def _filas(n=120):
    base = datetime.date(2024, 1, 1)
    return [{"fecha": (base + datetime.timedelta(days=i)).isoformat(), "x": str(i % 7),
             "y": "si" if i % 2 else "no"} for i in range(n)]


def test_una_fecha_da_sus_cuatro_numeros():
    # 2024-03-15 es viernes (weekday 4) y está a 19.797 días del 1970-01-01:
    # 2024-01-01 son 1.704.067.200 s = 19.723 días, más 31 + 29 + 14.
    assert variables_de_fecha("2024-03-15", "%Y-%m-%d") == (2024.0, 3.0, 4.0, 19797.0)
    # El mismo día escrito a la española da los MISMOS números.
    assert variables_de_fecha("15/03/2024", "%d/%m/%Y") == (2024.0, 3.0, 4.0, 19797.0)
    # Con hora, los días llevan la fracción (las 18:00 son 0,75 de día) y sale
    # una quinta variable, la hora: 18:30 → 18,5.
    con_hora = variables_de_fecha("2024-03-15T18:30", "%Y-%m-%dT%H:%M")
    assert con_hora[3] == pytest.approx(19797.0 + 18.5 / 24)
    assert len(con_hora) == 5 and con_hora[4] == pytest.approx(18.5)
    # Sin hora en el formato no se inventa una columna constante.
    assert len(variables_de_fecha("2024-03-15", "%Y-%m-%d")) == 4


def test_la_receta_con_hora_tiene_cinco_columnas_y_viaja():
    receta = RecetaDeFecha(columna="momento", formato="%Y-%m-%d %H:%M:%S")
    assert receta.variables == VARIABLES_CON_HORA
    assert receta.columnas_derivadas()[-1] == "momento__hora"
    assert RecetaDeFecha.desde_json(receta.a_json()) == receta
    nueva, _ = expandir_fila({"momento": "2012-10-02 09:15:00"}, (receta,))
    assert nueva["momento__hora"] == pytest.approx(9.25)
    vacia, _ = expandir_fila({"momento": ""}, (receta,))
    assert [vacia[c] for c in receta.columnas_derivadas()] == [None] * 5
    with pytest.raises(RecetaDeFechaIncompatible):  # una receta con hora que declara cuatro
        RecetaDeFecha.desde_json({**receta.a_json(), "variables": list(VARIABLES)})


def test_se_detecta_con_la_misma_regla_que_el_analisis_del_conjunto():
    filas = _filas()
    tipar_columnas_numericas(filas, ("fecha", "x"))
    recetas = detectar_fechas(filas, ("fecha", "x"))
    # En POSITIVO: la fecha sí, con su formato; la numérica no.
    assert recetas == (RecetaDeFecha(columna="fecha", formato="%Y-%m-%d"),)


def test_formatos_mezclados_no_se_detectan():
    filas = [{"f": "2024-01-0%d" % d} for d in range(1, 6)] + [{"f": "06/01/2024"}]
    assert detectar_fechas(filas, ("f",)) == ()


def test_una_fecha_NUNCA_VISTA_sale_con_numeros_y_no_desconocida():
    """El motivo del corte, de punta a punta por la preparación del núcleo.

    Hoy (medido el 2026-09-21) la fecha entra categórica y una fecha nueva
    sale `__desconocida__`. Con la receta, la preparación recibe números y
    una fecha de 2025 —fuera de todo el entrenamiento— se convierte igual."""
    filas = _filas()
    tipar_columnas_numericas(filas, ("fecha", "x"))
    recetas = detectar_fechas(filas, ("fecha", "x"))
    expandidas = [expandir_fila(f, recetas)[0] for f in filas]
    columnas = ("x",) + recetas[0].columnas_derivadas()
    politica = ajustar_preparacion(expandidas, objetivo="y", columnas=columnas,
                                   admite_categoricas=True, admite_faltantes=True)
    tipos = {c.columna: c.tipo for c in politica.columnas}
    assert all(tipos[c] == "numerica" for c in recetas[0].columnas_derivadas()), tipos

    nueva, ilegibles = expandir_fila({"fecha": "2025-06-01", "x": 3.0}, recetas)
    assert ilegibles == ()
    # SUSTITUIDA, no añadida: si el texto siguiera en la fila, cualquiera que
    # la pase entera a la preparación lo volvería a meter como categórica.
    # (Mirarlo después de `transformar_fila` no vale: esa ya tira lo que su
    # política no conoce, y la prueba pasaba con la columna dentro.)
    assert "fecha" not in nueva
    preparada = transformar_fila(nueva, politica)
    assert preparada["fecha__anio"] == 2025.0 and preparada["fecha__mes"] == 6.0
    assert "__desconocida__" not in preparada.values()


def test_un_valor_ilegible_sale_faltante_y_se_devuelve_para_decirlo():
    receta = RecetaDeFecha(columna="fecha", formato="%Y-%m-%d")
    nueva, ilegibles = expandir_fila({"fecha": "el martes", "x": 1.0}, (receta,))
    assert ilegibles == ("fecha",)
    # Faltantes, nunca un cero ni una fecha inventada.
    assert [nueva[c] for c in receta.columnas_derivadas()] == [None] * len(VARIABLES)
    assert nueva["x"] == 1.0


def test_un_vacio_es_un_faltante_no_un_ilegible():
    receta = RecetaDeFecha(columna="fecha", formato="%Y-%m-%d")
    for vacio in ("", "  ", None):
        nueva, ilegibles = expandir_fila({"fecha": vacio}, (receta,))
        assert ilegibles == ()
        assert [nueva[c] for c in receta.columnas_derivadas()] == [None] * len(VARIABLES)


def test_no_toca_la_fila_que_recibe():
    fila = {"fecha": "2024-03-15"}
    expandir_fila(fila, (RecetaDeFecha(columna="fecha", formato="%Y-%m-%d"),))
    assert fila == {"fecha": "2024-03-15"}


def test_la_receta_viaja_y_se_niega_a_leer_otra():
    receta = RecetaDeFecha(columna="alta", formato="%d/%m/%Y")
    assert RecetaDeFecha.desde_json(receta.a_json()) == receta
    with pytest.raises(RecetaDeFechaIncompatible):
        RecetaDeFecha.desde_json({**receta.a_json(), "referencia": "2000-01-01"})
    with pytest.raises(RecetaDeFechaIncompatible):
        RecetaDeFecha.desde_json({**receta.a_json(), "variables": ["anio", "mes"]})
    with pytest.raises(RecetaDeFechaIncompatible):
        RecetaDeFecha(columna="alta", formato="%Y%m%d")
    assert receta.a_json()["referencia"] == REFERENCIA


def test_una_columna_que_se_lee_igual_como_dia_mes_que_como_mes_dia_no_se_convierte():
    """Auditoría del 2026-09-22: `MM/01/AAAA` se detectaba `%d/%m/%Y` y el año entero
    caía en los días 1–12 de enero. Si los dos formatos leen TODA la columna, no se elige
    por orden: se deja fuera y se dice."""
    mensual_eeuu = [{"f": f"{m:02d}/01/2024"} for m in range(1, 13)]
    assert detectar_fechas(mensual_eeuu, ("f",)) == ()
    assert fechas_ambiguas(mensual_eeuu, ("f",)) == (("f", ("%d/%m/%Y", "%m/%d/%Y")),)
    # Con un solo día > 12 ya no hay duda, en ninguno de los dos sentidos.
    espanola = mensual_eeuu + [{"f": "15/01/2024"}]
    assert detectar_fechas(espanola, ("f",)) == (RecetaDeFecha(columna="f", formato="%d/%m/%Y"),)
    eeuu = mensual_eeuu + [{"f": "01/15/2024"}]
    assert detectar_fechas(eeuu, ("f",)) == (RecetaDeFecha(columna="f", formato="%m/%d/%Y"),)
    assert fechas_ambiguas(eeuu, ("f",)) == ()
