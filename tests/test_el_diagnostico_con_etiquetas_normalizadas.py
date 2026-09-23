"""EL DIAGNÓSTICO DE RIESGO VE LA CLASE POSITIVA AUNQUE EL CSV LA ESCRIBA DISTINTO (2026-09-23).

Auditoría cruzada: `diagnosticar_csv` comparaba el valor CRUDO del objetivo («1», «Sí»,
«Yes») con `problema.positive_label`, que es la etiqueta NORMALIZADA de la confirmación
(«class_1», «si», «yes»). Con los objetivos más corrientes el indicador salía todo ceros y
el detector de fuga numérica y el de precisión insuficiente se apagaban EN SILENCIO; con
`si`/`no` funcionaba, y por eso nada lo veía. Se prueba contra ese control.
"""
from __future__ import annotations

import random

import pytest

from matrixai.training.diagnostico import diagnosticar_csv
from matrixai.training.objetivo import confirmar_desde_csv


def _csv(positiva, negativa, eventos, n=200):
    rng = random.Random(0)
    lineas = ["edad,fuga,y"]
    for _ in range(n):
        es = rng.random() < eventos
        lineas.append(f"{rng.randint(20, 80)},{(1.0 if es else 0.0) + rng.random() * 0.01:.4f},"
                      f"{positiva if es else negativa}")
    return "\n".join(lineas) + "\n"


def _diagnostico(positiva, negativa, dicha, eventos):
    texto = _csv(positiva, negativa, eventos)
    conf = confirmar_desde_csv(texto, objetivo="y", tarea="binary_classification",
                               clase_positiva=dicha, unidad_de_observacion="una persona")
    assert conf.confirmado, [p.clave for p in conf.preguntas]
    d = diagnosticar_csv(texto, conf.problema)
    return {s.clave for s in d.sospechas}, {lim.clave for lim in d.limites}


@pytest.mark.parametrize("positiva,negativa,dicha", [
    ("si", "no", "si"),          # el control que siempre funcionó
    ("1", "0", "1"), ("Sí", "No", "Sí"), ("Yes", "No", "Yes"), ("1", "0", "class_1"),
])
def test_la_fuga_numerica_se_ve_con_cualquier_forma_de_la_clase(positiva, negativa, dicha):
    sospechas, _ = _diagnostico(positiva, negativa, dicha, eventos=0.5)
    assert "asociacion_muy_alta" in sospechas, sospechas


@pytest.mark.parametrize("positiva,negativa,dicha", [
    ("si", "no", "si"), ("1", "0", "1"), ("Sí", "No", "Sí"), ("Yes", "No", "Yes"),
])
def test_la_precision_insuficiente_cuenta_los_eventos_de_verdad(positiva, negativa, dicha):
    _, limites = _diagnostico(positiva, negativa, dicha, eventos=0.03)
    assert "precision_insuficiente" in limites, limites
