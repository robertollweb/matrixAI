# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""El router del prompt entiende el español que escribe la gente.

QUÉ PASABA (Roberto, 2026-08-17, conduciendo `?fases`): «has cambiado el
diagrama, ahora no describe las funciones, ni parámetros, así no me dice
nada». El diagrama no había cambiado — lo que llegaba a dibujar sí.

MEDIDO contra el core: «predice si un cliente impaga a partir de su edad,
sus ingresos y su deuda» devolvía `PROJECT PromptGeneratedRiskAgent`, con
`VECTOR Signal[4]` de campos inventados (severity, confidence, history,
context) y **cero bloques NETWORK, cero LAYER**. Un modelo sin capas no
tiene capas que dibujar.

LA CAUSA: la lista de verbos de tarea llevaba los INFINITIVOS SIN TILDE
—«predecir», «clasificar», «estimar»— y se comparaba por subcadena contra
el texto en crudo. En español nadie encarga así: se escribe «predice»,
«clasifica», «estima». Y con tilde («predicción») tampoco casaba, porque
el texto no se normalizaba.

LO QUE LO CONVIERTE EN GRAVE: **los ejemplos del propio producto
fallaban**. El marcador de la caja de fases propone «Predice qué clientes
van a impagar el crédito…» y el manual del texto usa «clasifica reseñas
por sentimiento»: los dos iban al agente genérico. Los tres ejemplos EN
INGLÉS de la portada sí funcionaban, así que el producto parecía roto
solo para quien lo usaba en español — que es casi todo el mundo que lo ha
probado.

Esta prueba fija los ejemplos REALES de la interfaz. Si alguien cambia un
ejemplo por otro que no construye una red, salta aquí y no en la cara de
quien lo pruebe.
"""
from __future__ import annotations

import pytest

from matrixai.playground import _is_neural_prompt

#: Los ejemplos que la interfaz ofrece de verdad. Copiados de:
#:   · `studio/src/creacion/textosDeCreacion.ts` (marcador de la caja, ES y EN)
#:   · `studio/src/fases/…` y la portada de la web (los tres ejemplos)
#:   · el manual de los modelos de texto
#: No se resumen ni se «limpian»: se pegan tal cual se ofrecen, porque lo
#: que se está comprobando es exactamente lo que alguien va a escribir.
EJEMPLOS_DE_LA_INTERFAZ = [
    "Predice qué clientes van a impagar el crédito a partir del histórico de 2026, "
    "priorizando no dejar escapar impagos.",
    "Predict which customers will default on their credit from the 2026 history, "
    "prioritising not letting defaults slip through.",
    "Detect hospital readmissions within 30 days based on patient history",
    "Classify loan default risk from applicant financial data",
    "Predict equipment failure from sensor readings",
    "clasifica reseñas por sentimiento",
    "Predice si va a llover mañana, priorizando no dejar escapar los días de lluvia.",
]

#: Lo que una persona escribe de verdad, en español, en imperativo y con
#: tildes. Cada una de estas fallaba antes del 2026-08-18.
EN_ESPAÑOL_DE_VERDAD = [
    "predice si un cliente impaga a partir de su edad, sus ingresos y su deuda",
    "clasifica los tickets de soporte por urgencia",
    "estima el consumo eléctrico de una vivienda",
    "detecta transacciones fraudulentas",
    "identifica qué pacientes van a reingresar",
    "predicción de impago a partir de la deuda y los ingresos",
    "pronostica la demanda de la semana que viene",
]

#: Lo que NO es una red y tiene que seguir sin serlo: si esto se rompe,
#: el arreglo habría convertido el router en un «sí» a todo.
NO_SON_REDES = [
    "automatiza el flujo de aprobación con reglas de negocio",
    "orquesta un pipeline de decisiones cuando ocurra un evento",
    "Estimado cliente, redacta una respuesta al correo de reclamación",
]


@pytest.mark.parametrize("prompt", EJEMPLOS_DE_LA_INTERFAZ)
def test_los_ejemplos_que_ofrece_el_producto_construyen_una_red(prompt: str) -> None:
    assert _is_neural_prompt(prompt) is True, (
        "un ejemplo que la propia interfaz sugiere no construye una red: "
        "quien lo pruebe verá un agente con campos inventados y un diagrama sin capas"
    )


@pytest.mark.parametrize("prompt", EN_ESPAÑOL_DE_VERDAD)
def test_el_imperativo_y_las_tildes_tambien_construyen_una_red(prompt: str) -> None:
    assert _is_neural_prompt(prompt) is True, (
        "en español se encarga en imperativo: si solo vale el infinitivo, "
        "el producto está roto para quien lo use en su idioma"
    )


@pytest.mark.parametrize("prompt", NO_SON_REDES)
def test_lo_que_no_es_una_red_sigue_sin_serlo(prompt: str) -> None:
    assert _is_neural_prompt(prompt) is False, (
        "el arreglo del router no puede convertirse en un «sí» a todo: "
        "un flujo de trabajo se queda en el supervisor de prompts"
    )
