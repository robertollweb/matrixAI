# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C0 — el texto libre queda FUERA del estudio, con su motivo (2026-09-21).

Decisión de Roberto (107-C3, 1c). Medido el 2026-09-18: una nota clínica entraba
en el estudio como una categórica con un valor por fila —200 categorías en 200
filas de entrenamiento, las 100 de prueba en `__desconocida__`— sin un solo aviso.
El camino del prompt ya la detectaba (`dataset_analysis._free_text_evidence`); la
confirmación del estudio no la miraba.
"""
from __future__ import annotations

import csv
import io
import random

import pytest

from matrixai.training.objetivo import confirmar_desde_csv

_FRASES = ["dolor torácico de inicio brusco", "refiere disnea de esfuerzo desde hace días",
           "sin antecedentes de interés", "paciente con fiebre y tos productiva",
           "acude por mareo y cefalea intensa", "control rutinario sin incidencias"]


def _csv(n=300, seed=0):
    rng = random.Random(seed)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["row_id", "edad", "servicio", "nota", "y"])
    for i in range(n):
        w.writerow([i, rng.randint(20, 90), rng.choice(["urgencias", "planta", "consulta"]),
                    f"{rng.choice(_FRASES)}; {rng.choice(_FRASES)} (visita {i})",
                    rng.choice(["si", "no"])])
    return buf.getvalue()


def _confirmar(**kw):
    return confirmar_desde_csv(_csv(), objetivo="y", clase_positiva="si",
                               unidad_de_observacion="una visita", **kw)


@pytest.mark.parametrize("entradas", [None, ["edad", "servicio", "nota"]],
                         ids=["propuestas_por_el_nucleo", "declaradas"])
def test_la_nota_queda_fuera_y_las_demas_entran(entradas):
    c = _confirmar(entradas=entradas)
    assert c.confirmado
    # En POSITIVO lo que sí entra: una categórica normal no es texto libre.
    assert c.problema.predictors == ("edad", "servicio")
    assert [e.campo for e in c.excluidas] == ["nota"]


def test_lo_excluido_se_dice_con_su_motivo_y_su_medida_en_los_dos_idiomas():
    (excluida,) = _confirmar().excluidas
    assert excluida.clave == "texto_libre_excluido"
    assert "'nota'" in excluida.motivo["es"] and "texto libre" in excluida.motivo["es"]
    assert "'nota'" in excluida.motivo["en"] and "free text" in excluida.motivo["en"]
    # La medida que lo sostiene viaja con él: un umbral sin su medida no se discute.
    assert excluida.evidencia["median_words"] >= 5
    assert 0 < excluida.evidencia["distinct_word_ratio"] <= 1
    assert "excluidas" in _confirmar().a_json()
    assert _confirmar().a_json()["excluidas"][0]["campo"] == "nota"


def test_la_nota_sin_declarar_tambien_se_dice():
    """Al arrancar el estudio, la pantalla reenvía la lista que le propuso el
    núcleo, ya sin la nota: la confirmación de ese momento tiene que seguir
    diciendo que la nota se quedó fuera, o Resultados no podría contarlo."""
    c = _confirmar(entradas=["edad", "servicio"])
    assert c.problema.predictors == ("edad", "servicio")
    assert [e.campo for e in c.excluidas] == ["nota"]


def test_sin_texto_libre_no_se_excluye_nada():
    csv_sin_texto = "\n".join(l.rsplit(",", 2)[0] + "," + l.rsplit(",", 1)[1]
                               for l in _csv().splitlines())
    c = confirmar_desde_csv(csv_sin_texto, objetivo="y", clase_positiva="si",
                            unidad_de_observacion="una visita")
    assert c.confirmado
    assert "nota" not in csv_sin_texto
    assert c.excluidas == ()
    assert c.problema.predictors == ("edad", "servicio")


def test_si_solo_quedaba_texto_libre_no_se_confirma_un_estudio_sin_entradas():
    c = _confirmar(entradas=["nota"])
    assert not c.confirmado
    assert [b.clave for b in c.bloqueos] == ["sin_entradas_utilizables"]
    assert "'nota'" in c.bloqueos[0].motivo["es"] and "'nota'" in c.bloqueos[0].motivo["en"]
