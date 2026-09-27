# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El diagnóstico de la confirmación lee cada VALOR DISTINTO una vez (2026-09-27).

Dos cosas se hacían de más, y las dos con el resultado idéntico al quitarlas (comparado
contra el módulo de antes en 1.000 CSV al azar con objetivos binarios, multiclase y de
regresión, columnas que copian el objetivo y predictores gemelos: 0 diferencias):

- `_preparar_columna` preguntaba si era nula y si era un número CELDA A CELDA.
- El indicador de la clase positiva —de la columna OBJETIVO, el mismo para todos— se
  recomponía DENTRO del bucle de predictores, una vez por predictor numérico.

Medido, la confirmación entera de 52.000 × 100 en un cgroup de 6 GB: ~19 → ~16 s con
flotantes de dos decimales. Estas pruebas atan el coste por RECUENTO."""
from __future__ import annotations

import pytest

from matrixai.estudio import ProblemSpec
from matrixai.training import dataset_analysis, diagnostico
from matrixai.training.dataset_analysis import analyze_dataset_csv
from matrixai.training.dataset_project import _read_rows

FILAS = 400
PREDICTORES = ("a", "b", "c", "d", "e")


def _csv() -> str:
    lineas = ["a,b,c,d,e,y"]
    for i in range(FILAS):
        lineas.append(",".join([str(i % 7), f"{(i % 11) / 2}", str(i % 3), str(i % 5), str(i % 13),
                                ("si", "no")[i % 3 == 0]]))
    return "\n".join(lineas) + "\n"


def _diagnosticar():
    texto = _csv()
    problema = ProblemSpec(problem_id="p", task="binary_classification", observation_unit="fila",
                           target="y", predictors=PREDICTORES, classes=("si", "no"),
                           positive_label="si")
    return diagnostico.diagnosticar_csv(texto, problema, analisis=analyze_dataset_csv(texto),
                                        filas=_read_rows(texto))


def test_una_columna_se_prepara_leyendo_cada_valor_distinto_una_vez(monkeypatch):
    real = dataset_analysis._is_null
    leidos: list[object] = []

    def contado(valor, tokens_de_ausencia=None):
        leidos.append(valor)
        return real(valor, tokens_de_ausencia)

    monkeypatch.setattr(dataset_analysis, "_is_null", contado)
    preparada = diagnostico._preparar_columna(["3", "3", " 3", "", "x", "3", None, "x"] * 100)
    # Cinco valores distintos ("3", " 3", "", "x", None): None no se memoriza —no es una
    # cadena— y se pregunta cada vez (100); las cadenas, una vez cada una (4).
    assert len(leidos) == 4 + 100, len(leidos)
    assert preparada[:8] == [3.0, 3.0, 3.0, None, "x", 3.0, None, "x"]


def test_el_indicador_de_la_positiva_se_compone_UNA_vez_no_una_por_predictor(monkeypatch):
    real = diagnostico._comparador_de_la_positiva
    llamadas = [0]

    def comparador(*a, **k):
        es_la_positiva = real(*a, **k)

        def contado(v):
            llamadas[0] += 1
            return es_la_positiva(v)
        return contado

    monkeypatch.setattr(diagnostico, "_comparador_de_la_positiva", comparador)
    _diagnosticar()
    # UNA vez para el indicador y otra para contar los eventos: 2 × 400. Por predictor
    # numérico serían 5 × 400 + 400.
    assert llamadas[0] == 2 * FILAS, llamadas[0]


def test_lo_que_diagnostica_no_cambia_con_la_lectura_por_valor():
    """Un predictor que COPIA el objetivo sigue bloqueando, y dos gemelos siguen saliendo."""
    texto = "a,b,copia,y\n" + "".join(f"{i % 4},{i % 4},{('si', 'no')[i % 2]},{('si', 'no')[i % 2]}\n"
                                        for i in range(60))
    problema = ProblemSpec(problem_id="p", task="binary_classification", observation_unit="fila",
                           target="y", predictors=("a", "b", "copia"), classes=("si", "no"),
                           positive_label="si")
    d = diagnostico.diagnosticar_csv(texto, problema, analisis=analyze_dataset_csv(texto),
                                     filas=_read_rows(texto))
    assert [b.campo for b in d.bloqueos if b.clave == "objetivo_duplicado_confirmado"] == ["copia"]
    assert any(s.clave == "igualdad_de_valores_sin_explicar" for s in d.sospechas), d.sospechas


@pytest.mark.parametrize("valores,esperado", [
    (["1", " 1", "1.0", "", "NA", "a", " a "], [1.0, 1.0, 1.0, None, None, "a", "a"]),
    ([None, "None", "nan", "0"], [None, None, None, 0.0]),
])
def test_preparar_da_lo_mismo_por_celda(valores, esperado):
    assert diagnostico._preparar_columna(valores * 3) == esperado * 3
