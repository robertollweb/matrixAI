# SPDX-License-Identifier: AGPL-3.0-only
"""H13 (11-11) — la tarea de un proyecto desde dataset NO se adivina leyendo los DATOS del usuario.

`generate_project_from_dataset` ya decide la tarea por el tipo del objetivo y compone
`predecir|clasificar\\nFEATURES:\\n<columnas con sus categorias>\\nSALIDA: …`. El generador volvia a
adivinarla buscando palabras en el prompt entero: una categoria «Actividades Clasificadas» volvia
multiclase una regresion y el CSV preparado (`predicted_value`) no pasaba la validacion del modelo.
"""
from __future__ import annotations

import csv
import io

import pytest

from matrixai.training.dataset_project import generate_project_from_dataset
from matrixai.training.dense_generator import DenseNetworkGenerator


def _csv(categorias, objetivo):
    b = io.StringIO()
    w = csv.writer(b)
    w.writerow(["organo", "importe_estimado", "objetivo"])
    for i in range(24):
        w.writerow([categorias[i % len(categorias)], 10 + i * 1.5, objetivo(i)])
    return b.getvalue()


def _generar(texto, destino="objetivo"):
    return generate_project_from_dataset(
        texto, destino, locale="es", column_type_overrides={"organo": "categorical"})


@pytest.mark.parametrize("categorias", [
    ["Ayuntamiento", "Actividades Clasificadas", "Diputacion"],
    ["Servicio de deteccion de fugas", "Ayuntamiento", "Diputacion"],
    ["Comision de categorias profesionales", "Ayuntamiento", "Diputacion"],
    ["Classification Board", "Ayuntamiento", "Diputacion"],
    # verbo de prediccion («detec») y luego « si »: lo leia como pregunta de si/no -> binaria
    ["Deteccion de fugas si procede", "Ayuntamiento", "Diputacion"],
], ids=["clasificadas", "deteccion", "categorias", "classification", "deteccion_si"])
def test_regresion_con_categoria_de_palabra_de_tarea_sigue_siendo_regresion(categorias):
    res = _generar(_csv(categorias, lambda i: 100.0 + i * 3.25))
    assert "predicted_value" in str(res)
    assert "predicted_class" not in str(res)
    assert "predicted_prob" not in str(res)


def test_clasificacion_con_objetivo_categorico_sigue_siendo_clasificacion_aunque_las_columnas_digan_precio():
    b = io.StringIO()
    w = csv.writer(b)
    w.writerow(["precio_medio", "valor_total", "objetivo"])
    for i in range(24):
        w.writerow([10 + i, 5 + i * 2, ["alto", "medio", "bajo"][i % 3]])
    res = generate_project_from_dataset(b.getvalue(), "objetivo", locale="es")
    assert "predicted_class" in str(res)
    assert "predicted_value" not in str(res)


def test_detect_task_no_lee_el_bloque_de_columnas_pero_si_la_frase():
    dg = DenseNetworkGenerator()
    bloque = "\nFEATURES:\n  organo: Categorical[Actividades Clasificadas, otro]\nSALIDA: y\n"
    assert dg._detect_task("predecir" + bloque, None) == "regression"
    assert "clasificar" not in dg._porque_esa_tarea("predecir" + bloque, None, "regression")
    # el verbo de FUERA del bloque sigue mandando
    assert dg._detect_task("clasificar" + bloque, None) == "multiclass"


@pytest.mark.parametrize("frase,esperada", [
    ("clasificar el nivel de riesgo en alto, medio, bajo a partir de la edad", "multiclass"),
    ("clasificar si un cliente cancela a partir de su uso", "binary"),
    ("predecir el precio de una casa a partir de superficie", "regression"),
    ("clasificacion binaria", "binary"),
])
def test_los_prompts_humanos_no_cambian(frase, esperada):
    assert DenseNetworkGenerator()._detect_task(frase, None) == esperada
