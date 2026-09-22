# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C1 — «valor nunca visto» y «fuera del rango visto» (`matrixai.training.lo_visto`)."""
from __future__ import annotations

import pytest

from matrixai.training.lo_visto import ComprobacionDeLoVisto, Marca, comprobar_lo_visto
from matrixai.training.preparacion import (
    PoliticaDePreparacion,
    ajustar_preparacion,
    transformar_fila,
)


def _politica():
    filas = [{"edad": float(20 + i), "ciudad": ("Santander", "Laredo", "Castro")[i % 3],
              "y": "si" if i % 2 else "no"} for i in range(60)]
    return ajustar_preparacion(filas, objetivo="y", columnas=("edad", "ciudad"),
                               admite_categoricas=True, admite_faltantes=True)


def test_una_fila_normal_lo_AFIRMA():
    comprobacion = comprobar_lo_visto({"edad": 40.0, "ciudad": "Laredo"}, _politica())
    assert comprobacion.marcas == ()
    assert comprobacion.dentro_de_lo_visto is True


def test_una_categoria_nueva_sale_con_su_columna_y_su_valor():
    comprobacion = comprobar_lo_visto({"edad": 40.0, "ciudad": "Soria"}, _politica())
    assert comprobacion.dentro_de_lo_visto is False
    assert comprobacion.marcas == (Marca(columna="ciudad", tipo="categoria_nueva", valor="Soria"),)
    motivo = comprobacion.marcas[0].motivo()
    assert "«ciudad»" in motivo["es"] and "Soria" in motivo["es"]
    assert "“ciudad”" in motivo["en"] and "Soria" in motivo["en"]


def test_un_numero_fuera_del_rango_sale_con_los_extremos_que_se_vieron():
    politica = _politica()  # edad de 20 a 79
    comprobacion = comprobar_lo_visto({"edad": 95.5, "ciudad": "Laredo"}, politica)
    assert comprobacion.marcas == (Marca(columna="edad", tipo="fuera_del_rango", valor=95.5,
                                         minimo=20.0, maximo=79.0),)
    motivo = comprobacion.marcas[0].motivo()
    assert "95,5" in motivo["es"] and "de 20 a 79" in motivo["es"]
    assert "95.5" in motivo["en"] and "from 20 to 79" in motivo["en"]
    # Por debajo también.
    assert comprobar_lo_visto({"edad": 3.0, "ciudad": "Laredo"}, politica).marcas[0].tipo \
        == "fuera_del_rango"


def test_los_extremos_mismos_estan_dentro():
    politica = _politica()
    for edad in (20.0, 79.0):
        assert comprobar_lo_visto({"edad": edad, "ciudad": "Laredo"},
                                  politica).dentro_de_lo_visto is True


def test_un_faltante_que_el_entrenamiento_nunca_tuvo_se_marca():
    """Auditoría del 2026-09-22: la política de `_politica()` se entrenó SIN faltantes,
    y una fila con la columna vacía salía «dentro de lo visto»."""
    politica = _politica()
    for fila, columna in (({"ciudad": "Laredo"}, "edad"), ({"edad": "", "ciudad": "Laredo"}, "edad"),
                          ({"edad": 40.0, "ciudad": ""}, "ciudad")):
        comprobacion = comprobar_lo_visto(fila, politica)
        assert comprobacion.dentro_de_lo_visto is False, fila
        (marca,) = comprobacion.marcas
        assert (marca.columna, marca.tipo, marca.valor) == (columna, "faltante_nunca_visto", None)
        assert f"«{columna}»" in marca.motivo()["es"] and f"“{columna}”" in marca.motivo()["en"]


def test_un_faltante_donde_el_entrenamiento_ya_los_tenia_no_se_marca():
    filas = [{"edad": (None if i % 5 == 0 else float(20 + i)),
              "ciudad": (None if i % 7 == 0 else ("A", "B")[i % 2]),
              "y": "si" if i % 2 else "no"} for i in range(60)]
    politica = ajustar_preparacion(filas, objetivo="y", columnas=("edad", "ciudad"),
                                   admite_categoricas=True, admite_faltantes=True)
    for fila in ({"ciudad": "A"}, {"edad": 30.0, "ciudad": ""}):
        assert comprobar_lo_visto(fila, politica).dentro_de_lo_visto is True, fila


def test_una_politica_sin_extremos_NO_dice_que_esta_dentro():
    """Las escritas antes del 2026-09-14 no traen `minimo`/`maximo`: ese estudio
    no lo midió, y «sin marcas» no puede convertirse en «dentro»."""
    vieja = _politica().a_json()
    for columna in vieja["columnas"]:
        columna.pop("minimo", None)
        columna.pop("maximo", None)
    politica = PoliticaDePreparacion.desde_json(vieja)
    comprobacion = comprobar_lo_visto({"edad": 40.0, "ciudad": "Laredo"}, politica)
    assert comprobacion.marcas == ()
    assert comprobacion.sin_rango == ("edad",)
    assert comprobacion.dentro_de_lo_visto is None
    # Y una categoría nueva se sigue viendo: esa no necesita extremos.
    assert comprobar_lo_visto({"edad": 40.0, "ciudad": "Soria"}, politica).dentro_de_lo_visto \
        is False


def test_la_decision_de_desconocida_es_la_de_transformar_fila():
    """Una sola decisión: si la preparación trata un valor como conocido, aquí
    no se marca. `transformar_fila` compara `str(valor)`, así que un `1` entero
    frente a la categoría `"1"` es CONOCIDO para el modelo."""
    filas = [{"nivel": ("1", "2")[i % 2], "y": "si" if i % 3 else "no"} for i in range(40)]
    politica = ajustar_preparacion(filas, objetivo="y", columnas=("nivel",),
                                   admite_categoricas=True, admite_faltantes=True)
    assert transformar_fila({"nivel": 1}, politica)["nivel"] == "1"
    assert comprobar_lo_visto({"nivel": 1}, politica).dentro_de_lo_visto is True


def test_viaja_en_json_con_su_motivo():
    comprobacion = comprobar_lo_visto({"edad": 95.5, "ciudad": "Soria"}, _politica())
    datos = comprobacion.a_json()
    assert datos["dentro_de_lo_visto"] is False
    assert [m["columna"] for m in datos["marcas"]] == ["edad", "ciudad"]
    assert set(datos["marcas"][0]["motivo"]) == {"es", "en"}


def test_una_marca_fuera_del_rango_sin_extremos_no_se_puede_construir():
    with pytest.raises(ValueError):
        Marca(columna="edad", tipo="fuera_del_rango", valor=1.0)
    with pytest.raises(ValueError):
        Marca(columna="edad", tipo="rara", valor=1.0)
    assert ComprobacionDeLoVisto(marcas=()).dentro_de_lo_visto is True
