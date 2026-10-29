"""DEUDA N-3 (06-10), CERRADA EL 29-10: la pregunta de la clase positiva nombra las clases COMO LAS TRAE EL CSV.

Con un objetivo 0/1 la confirmación decía «necesita saber cuál de class_0, class_1 es la clase POSITIVA»: los nombres
internos tras normalizar, que la persona nunca escribió. Visto conduciendo el 142 con un CSV propio. La pregunta vecina
(«dos clases en varias») ya los traducía con `_vistas_en_el_csv`; esta usa la misma función. `opciones` NO cambia: son
las normalizadas, que es lo que se manda de vuelta y se traduce con el mismo mapa.
"""
from __future__ import annotations

import pytest

from matrixai.training.objetivo import confirmar_desde_csv


def _csv(valores: list[str], n: int = 60) -> str:
    return "\n".join(["x,y"] + [f"{i},{valores[i % len(valores)]}" for i in range(n)]) + "\n"


def _pregunta(valores):
    conf = confirmar_desde_csv(_csv(valores), objetivo="y", tarea="binary_classification",
                               unidad_de_observacion="una fila")
    return next(p for p in conf.preguntas if p.clave == "clase_positiva")


@pytest.mark.parametrize("valores, vistas, internas", [
    (["0", "1"], ("«0»", "«1»"), ("class_0", "class_1")),
    (["Sí", "No"], ("«No»", "«Sí»"), None),
    (["ALTO", "bajo"], ("«ALTO»", "«bajo»"), None),
])
def test_la_pregunta_nombra_las_clases_como_las_trae_el_csv(valores, vistas, internas):
    p = _pregunta(valores)
    for idioma in ("es", "en"):
        for v in vistas:
            assert v in p.motivo[idioma], (idioma, p.motivo[idioma])
        if internas:
            for i in internas:
                assert i not in p.motivo[idioma], (idioma, p.motivo[idioma])


def test_las_opciones_siguen_siendo_las_normalizadas_y_contestar_con_lo_que_se_ve_confirma():
    p = _pregunta(["0", "1"])
    assert p.opciones == ("class_0", "class_1")
    conf = confirmar_desde_csv(_csv(["0", "1"]), objetivo="y", tarea="binary_classification",
                               unidad_de_observacion="una fila", clase_positiva="1")
    assert "clase_positiva" not in [q.clave for q in conf.preguntas]
