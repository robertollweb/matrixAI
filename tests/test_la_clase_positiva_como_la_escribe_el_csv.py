"""LA CLASE POSITIVA SE ACEPTA COMO LA ESCRIBE EL CSV (2026-09-23).

Medido conduciendo el Studio: con un objetivo `0/1` o `Sí/No`, la confirmación nombra
las clases NORMALIZADAS (`class_0`/`class_1`, `no`/`si`) y solo aceptaba esas. Quien
escribía «1» o «Sí» —lo que ve en su columna— se quedaba con la pregunta de la clase
positiva abierta para siempre, y el estudio no arrancaba. Se traduce con el MISMO mapa
con el que se nombran las clases.
"""
from __future__ import annotations

import pytest

from matrixai.training.objetivo import confirmar_desde_csv


def _csv(positiva: str, negativa: str, n: int = 60) -> str:
    filas = ["x,y"] + [f"{i},{positiva if i % 2 else negativa}" for i in range(n)]
    return "\n".join(filas) + "\n"


@pytest.mark.parametrize("positiva,negativa,dicha,esperada", [
    ("1", "0", "1", "class_1"),
    ("Sí", "No", "Sí", "si"),
    ("Y", "N", "Y", "y"),
    ("1", "0", "class_1", "class_1"),   # la etiqueta normalizada sigue valiendo
    ("si", "no", "si", "si"),           # y lo que no cambia, no cambia
])
def test_la_clase_tal_como_viene_confirma_con_su_etiqueta(positiva, negativa, dicha, esperada):
    conf = confirmar_desde_csv(_csv(positiva, negativa), objetivo="y",
                               tarea="binary_classification", clase_positiva=dicha,
                               unidad_de_observacion="una fila")
    assert conf.confirmado, [p.clave for p in conf.preguntas]
    assert conf.problema.positive_label == esperada


def test_un_valor_que_no_esta_en_la_columna_sigue_preguntando():
    """La otra mitad: traducir no es aceptar cualquier cosa."""
    conf = confirmar_desde_csv(_csv("1", "0"), objetivo="y", tarea="binary_classification",
                               clase_positiva="2", unidad_de_observacion="una fila")
    assert not conf.confirmado
    assert "clase_positiva" in [p.clave for p in conf.preguntas]
