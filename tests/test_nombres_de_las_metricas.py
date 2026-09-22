# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cada métrica del registro tiene nombre en los dos idiomas, y ponérselo no mueve la huella
del catálogo (es presentación). Deuda del 2026-09-22: Resultados pintaba «sensitivity» en
castellano porque nadie nombraba las métricas."""
from __future__ import annotations

import pytest

from matrixai.estudio import nombre_de_la_metrica
from matrixai.estudio.metricas import (METRICAS_DIFERIDAS, REGISTRO, MetricaAplazada,
                                       MetricaDesconocida, _NOMBRES, digest_del_catalogo)


def test_cada_metrica_del_registro_tiene_nombre_en_los_dos_idiomas():
    for metric_id in REGISTRO:
        nombre = nombre_de_la_metrica(metric_id)
        assert set(nombre) == {"es", "en"}, metric_id
        assert nombre["es"].strip() and nombre["en"].strip(), metric_id


def test_no_hay_nombres_de_metricas_que_no_existen():
    """Dos listas con las mismas claves: la de nombres no puede llevar una que el
    registro ya no tiene (se quedaría sonando razonable)."""
    assert set(_NOMBRES) == set(REGISTRO)


def test_los_de_la_matriz_de_confusion_se_llaman_como_en_castellano():
    assert nombre_de_la_metrica("sensitivity") == {"es": "Sensibilidad", "en": "Sensitivity"}
    assert nombre_de_la_metrica("specificity") == {"es": "Especificidad", "en": "Specificity"}
    assert nombre_de_la_metrica("accuracy")["es"] == "Exactitud"


def test_una_desconocida_o_aplazada_no_se_nombra():
    with pytest.raises(MetricaDesconocida):
        nombre_de_la_metrica("no_existe")
    if METRICAS_DIFERIDAS:
        with pytest.raises(MetricaAplazada):
            nombre_de_la_metrica(next(iter(METRICAS_DIFERIDAS)))


def test_el_nombre_NO_entra_en_la_huella_del_catalogo(monkeypatch):
    """El nombre vive en `MetricaRegistrada`, no en `MetricSpec`: cambiarlo no puede
    decir que un informe «se midió con otro catálogo». La huella se calcula sobre
    `_CATALOGO` (no sobre `REGISTRO`), así que es ahí donde se cambia."""
    import dataclasses
    from matrixai.estudio import metricas
    antes = digest_del_catalogo()
    otro = tuple(dataclasses.replace(r, nombre=("Otro", "Other")) for r in metricas._CATALOGO)
    assert all(r.nombre == ("Otro", "Other") for r in otro)
    monkeypatch.setattr(metricas, "_CATALOGO", otro)
    assert metricas.catalogo()[0] is otro[0].spec
    assert digest_del_catalogo() == antes
