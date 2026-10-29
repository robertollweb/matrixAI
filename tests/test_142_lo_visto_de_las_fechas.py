# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""142 — «lo visto» de una FECHA: la que no se lee (I1) y la que cae fuera del periodo (M1).

Auditoría a ciegas del 2026-10-29. Medido con estudios reales: `15/06/2023` en un estudio con
fechas ISO salía idéntica al bit que la fecha vacía (LightGBM y la densa), porque
`transformar_fila` la convierte en un HUECO, y `comprobar_lo_visto` no decía nada. Y una fecha
posterior al periodo visto se mandaba a `sin_rango` («no guardó el rango») aunque la política sí
guarda los extremos de `{columna}__dias_desde_1970`.

Las formas, medidas antes de escribir las pruebas (`variables_de_fecha` con `%Y-%m-%d`):
`2023-6-15` SÍ se lee (strptime acepta el mes sin cero), y ` 2023-06-15 ` también (se recorta);
`15/06/2023`, `junio de 2023` y `2023-02-30` no."""
from __future__ import annotations

import pytest

from matrixai.training.fechas import variables_de_fecha
from matrixai.training.lo_visto import MOTIVOS, Marca, comprobar_lo_visto
from matrixai.training.preparacion import (PoliticaDePreparacion, ajustar_preparacion,
                                           transformar_fila)


def _filas(con_huecos=False, formato_es=False):
    filas = []
    for i in range(60):
        mes, dia = 1 + i % 12, 1 + i % 28
        fecha = f"{dia:02d}/{mes:02d}/2023" if formato_es else f"2023-{mes:02d}-{dia:02d}"
        if con_huecos and i % 10 == 0:
            fecha = ""
        filas.append({"fecha": fecha, "x": float(i), "y": "si" if i % 2 else "no"})
    return filas


def _politica(**kw):
    return ajustar_preparacion(_filas(**kw), objetivo="y", columnas=["fecha", "x"],
                               admite_categoricas=True, admite_faltantes=False, con_fechas=True)


def _tipos(fecha, politica):
    return [m.tipo for m in comprobar_lo_visto({"fecha": fecha, "x": 3.0}, politica).marcas]


# --------------------------------------------------------------------------- I1


@pytest.mark.parametrize("ilegible", ["15/06/2023", "junio de 2023", "2023-02-30", "mañana"])
def test_una_fecha_que_no_se_lee_se_marca_como_ilegible(ilegible):
    politica = _politica()
    assert politica.columna("fecha").formato_de_fecha == "%Y-%m-%d"
    comprobacion = comprobar_lo_visto({"fecha": ilegible, "x": 3.0}, politica)
    assert [m.tipo for m in comprobacion.marcas] == ["fecha_ilegible"], comprobacion
    assert comprobacion.dentro_de_lo_visto is False
    marca = comprobacion.marcas[0]
    # El valor TAL COMO LLEGÓ: es lo que la persona tiene que reconocer.
    assert marca.columna == "fecha" and marca.valor == ilegible
    motivo = marca.motivo()
    assert f"«{ilegible}» no se pudo leer como fecha" in motivo["es"]
    assert "como si faltara" in motivo["es"]
    assert "could not be read as a date" in motivo["en"] and "as if it were missing" in motivo["en"]


def test_la_premisa_la_ilegible_es_para_el_modelo_el_mismo_hueco_que_la_vacia():
    """Lo que hace verdad al motivo «la trata como si faltara»: sin esto, el aviso podría
    quedarse diciendo eso el día que la preparación cambie."""
    politica = _politica()
    assert (transformar_fila({"fecha": "15/06/2023", "x": 3.0}, politica)
            == transformar_fila({"fecha": "", "x": 3.0}, politica))


def test_la_ilegible_se_marca_AUNQUE_al_entrenar_la_fecha_faltase():
    """Un hueco donde ya faltaba no se marca (el modelo lo aprendió), pero una fecha escrita y
    no leída sí: la persona dio un dato y el modelo no lo usó."""
    politica = _politica(con_huecos=True)
    assert politica.columna("fecha").proporcion_faltante > 0
    assert _tipos("", politica) == []
    assert _tipos("15/06/2023", politica) == ["fecha_ilegible"]


@pytest.mark.parametrize("legible", ["2023-06-15", "2023-6-15", " 2023-06-15 "])
def test_una_fecha_ISO_que_el_nucleo_SI_lee_no_se_marca(legible):
    politica = _politica()
    assert variables_de_fecha(legible, "%Y-%m-%d") is not None  # la premisa, medida
    comprobacion = comprobar_lo_visto({"fecha": legible, "x": 3.0}, politica)
    assert comprobacion.marcas == () and comprobacion.dentro_de_lo_visto is True, comprobacion


def test_en_un_estudio_dd_mm_aaaa_la_ilegible_es_la_ISO():
    """Es la lectura del ESTUDIO la que manda, no una idea propia de qué es una fecha."""
    politica = _politica(formato_es=True)
    assert politica.columna("fecha").formato_de_fecha == "%d/%m/%Y"
    assert _tipos("15/06/2023", politica) == []
    assert _tipos("2023-06-15", politica) == ["fecha_ilegible"]


def test_un_valor_que_no_es_texto_tampoco_se_lee():
    assert _tipos(20230615, _politica()) == ["fecha_ilegible"]


# --------------------------------------------------------------------------- M1


def test_una_fecha_posterior_al_periodo_se_marca_con_el_periodo_escrito_como_fechas():
    politica = _politica()
    dias = next(d for d in politica.derivadas_de_fecha if d.columna == "fecha__dias_desde_1970")
    comprobacion = comprobar_lo_visto({"fecha": "2031-06-15", "x": 3.0}, politica)
    assert [m.tipo for m in comprobacion.marcas] == ["fecha_fuera_del_periodo"], comprobacion
    assert comprobacion.dentro_de_lo_visto is False and comprobacion.sin_rango == ()
    marca = comprobacion.marcas[0]
    assert (marca.minimo, marca.maximo) == (dias.minimo, dias.maximo)
    motivo = marca.motivo()
    # Los extremos, como FECHAS (la primera y la última de `_filas`), nunca en días desde 1970.
    assert motivo["es"] == ("«fecha»: 2031-06-15 queda fuera del periodo que se vio al entrenar "
                            "(de 2023-01-01 a 2023-12-24)")
    assert motivo["en"] == ("“fecha”: 2031-06-15 is outside the period training saw "
                            "(from 2023-01-01 to 2023-12-24)")
    assert str(int(dias.minimo)) not in motivo["es"]


def test_una_fecha_anterior_tambien_y_los_extremos_mismos_estan_dentro():
    politica = _politica()
    assert _tipos("2022-12-31", politica) == ["fecha_fuera_del_periodo"]
    assert _tipos("2023-01-01", politica) == []
    assert _tipos("2023-12-24", politica) == []
    assert _tipos("2023-12-25", politica) == ["fecha_fuera_del_periodo"]


def test_sin_los_extremos_guardados_la_fecha_va_a_sin_rango_y_no_a_un_dentro():
    payload = _politica().a_json()
    for derivada in payload["derivadas_de_fecha"]:
        derivada.pop("minimo")
        derivada.pop("maximo")
    vieja = PoliticaDePreparacion.desde_json(payload)
    comprobacion = comprobar_lo_visto({"fecha": "2031-06-15", "x": 3.0}, vieja)
    assert comprobacion.marcas == () and comprobacion.sin_rango == ("fecha",)
    assert comprobacion.dentro_de_lo_visto is None
    # La ilegible no necesita los extremos: se marca igual.
    assert [m.tipo for m in comprobar_lo_visto({"fecha": "15/06/2023", "x": 3.0}, vieja).marcas] \
        == ["fecha_ilegible"]


def test_con_hora_el_periodo_se_escribe_con_su_hora():
    filas = [{"t": f"2023-03-{1 + i % 20:02d} {8 + i % 10:02d}:30", "x": float(i),
              "y": "si" if i % 2 else "no"} for i in range(60)]
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["t", "x"], admite_categoricas=True,
                                   admite_faltantes=False, con_fechas=True)
    assert politica.columna("t").formato_de_fecha == "%Y-%m-%d %H:%M"
    marca = comprobar_lo_visto({"t": "2023-03-21 10:00", "x": 1.0}, politica).marcas[0]
    assert marca.tipo == "fecha_fuera_del_periodo"
    assert "(de 2023-03-01 08:30 a 2023-03-20 17:30)" in marca.motivo()["es"], marca.motivo()


def test_una_marca_del_periodo_sin_extremos_no_se_puede_construir():
    with pytest.raises(ValueError):
        Marca(columna="fecha", tipo="fecha_fuera_del_periodo", valor="2031-06-15")


def test_los_dos_tipos_nuevos_estan_en_el_catalogo_con_los_dos_idiomas():
    """El lote cuenta las marcas por tipo con las claves de `MOTIVOS` (`estudio_job`)."""
    for tipo in ("fecha_ilegible", "fecha_fuera_del_periodo"):
        assert set(MOTIVOS[tipo]) == {"es", "en"}
