# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Qué dice el PROMPT que hay que predecir — y `None` cuando no lo dice (103-C1).

POR QUÉ EXISTE, con el caso medido delante. El contrato 71 dejó su segunda mitad
esperando decisión con esta frase: «detectarlo de verdad exige saber **qué dice
el prompt que hay que predecir**, y en la ruta de prompt el objetivo se llama
`predicted_class`: el nombre real no está en ninguna parte». Este módulo es ese
nombre real.

LO QUE NO HACE, que es lo que lo hace útil: **no inventa un objetivo para poder
seguir**. Medido hoy contra el core (`analyze_playground_request`, ruta
determinista) con las tres frases del 71 y con «analizar los datos de clientes
de una empresa de telefonía»: las cuatro salen con `OUTPUT predicted_class` o
`predicted_value` y `feature_1..4`, o sea que el core se inventa el objetivo Y
las entradas y nadie se entera. Aquí, una frase que no dice qué predecir
devuelve `None`, y quien llame tiene que preguntar.

TRES LECTURAS, EN ESTE ORDEN:

1. **Una declaración explícita manda.** `OUTPUT <nombre>: <tipo>`, `SALIDA:
   <nombre>`, `objetivo: <nombre>`, `target: <nombre>`. Es lo que escribe quien
   ya sabe cómo se llama su columna — y es también la forma en que se CONTESTA
   la pregunta cuando el core la hace.
2. **La frase**, cortada por el mismo conector que usa el core para separar el
   objetivo de las columnas (contrato 70 C1, `segmento_de_objetivo`): «predecir
   el salario de un empleado a partir del salario del año pasado» → el objetivo
   es lo de antes del «a partir de».
3. **Nada.** Y entonces `None`, no un nombre plausible.

DOS DETALLES QUE VALEN UN FALLO CADA UNO:

* Los nombres que el GENERADOR inventa —`predicted_class`, `predicted_value`,
  `predicted_prob`, `output`— **no cuentan como declaración**. Leerlos sería
  blanquear la invención: el prompt que los trae no los escribió una persona,
  los escribió el core en una vuelta anterior.
* Una **pregunta de sí o no** («predecir si un cliente va a impagar») declara la
  DEFINICIÓN y no el NOMBRE. Se devuelve con `nombre=None` y la definición
  entera: proponer «cliente» —el primer sustantivo tras el «si»— sería una
  respuesta silenciosamente equivocada, y quien la lea no tendría por qué
  sospechar.

La lista de verbos es CORTA y explícita, por la misma política que la lista de
nombres de identificador del 71: cada verbo de más es un objetivo leído donde no
lo hay, y un objetivo equivocado arranca un estudio entero en la dirección
equivocada. «Analizar» no está, y es deliberado: analizar no es predecir — el
caso del 71 («analizar los datos de clientes de una empresa de telefonía») es
justo el que tiene que salir sin objetivo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ObjetivoDelPrompt", "objetivo_declarado"]


#: Los nombres que pone el GENERADOR cuando nadie le dijo cómo se llama el
#: objetivo (`_output_name` en `dense_generator.py`). Un prompt que los trae
#: viene de una vuelta anterior del propio core, no de una persona.
NOMBRES_INVENTADOS = frozenset({
    "predicted_class", "predicted_value", "predicted_prob", "output",
})

#: Palabras clave que introducen una declaración explícita del objetivo. La
#: `SALIDA:` es la que escribe el propio core al sintetizar el prompt tipado
#: desde un CSV (`dataset_project.py`), así que por esa ruta el nombre REAL de
#: la columna sí viaja en el prompt.
_DECLARACION_RE = re.compile(
    r"(?:^|\n)[ \t]*"
    r"(?:OUTPUT|SALIDA|OBJETIVO|TARGET|"
    r"(?:COLUMNA|VARIABLE)[ \t]+OBJETIVO|TARGET[ \t]+COLUMN)"
    r"[ \t]*:?[ \t]+"
    r"(?P<nombre>[^\n:,;\[\]]+)",
    re.IGNORECASE,
)

#: Los verbos que introducen lo que se predice. Raíces, no conjugaciones: ir
#: añadiendo formas («prediga», «predijeran») sería perseguir el idioma sin
#: alcanzarlo, que es la conclusión a la que ya llegó el contrato 70.
_RAICES_DE_PREDICCION = (
    "predec", "predic", "prever", "preve", "pronostic", "estim", "calcul",
    "clasific", "categoriz", "detect", "determin", "anticip", "adivin",
    "predict", "forecast", "classif", "estimate", "determine", "score",
)

#: Lo que se salta entre el verbo y el nombre: artículos y muletillas.
_RELLENO = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "su", "sus",
    "cual", "cuales", "que", "cuanto", "cuanta", "cuantos", "cuantas",
    "the", "a", "an", "its", "his", "her", "what", "which", "how", "much", "many",
})

#: Los enlaces que SÍ forman parte del nombre cuando lo que sigue es otro
#: sustantivo: «el nivel DE riesgo» es un nombre, «el precio DE una vivienda» es
#: un complemento. Lo que los distingue es si detrás viene relleno.
_ENLACES = ("de", "del", "of")

#: Dónde ACABA el nombre del objetivo. Todas ellas empiezan otra cosa: un
#: complemento («de una vivienda»), una enumeración de clases («en barata,
#: media…») o una subordinada.
_CORTES = frozenset({
    "de", "del", "en", "para", "por", "que", "con", "a", "al", "y", "o", "sobre",
    "entre", "segun", "durante", "cuando", "si", "como", "desde", "hasta",
    "of", "for", "in", "on", "to", "and", "or", "from", "by", "with", "into",
    "as", "when", "at", "over", "between",
})

#: Cuántas palabras puede tener un nombre leído de la frase. Tres son «nivel de
#: riesgo» sin el corte; más allá se está copiando la frase entera y llamándola
#: nombre.
_MAX_PALABRAS = 3


@dataclass(frozen=True)
class ObjetivoDelPrompt:
    """Lo que la frase dice que hay que predecir.

    `nombre` es `None` cuando la frase declara la definición pero no un nombre
    —una pregunta de sí o no—: **ausente no es cero**, y rellenarlo con el
    primer sustantivo que aparezca sería una respuesta equivocada que nadie
    tendría por qué sospechar.
    """

    nombre: str | None
    definicion: str
    fuente: str          # "declaracion" | "frase"
    pregunta_si_no: bool = False

    def a_json(self) -> dict[str, object]:
        return {"nombre": self.nombre, "definicion": self.definicion,
                "fuente": self.fuente, "pregunta_si_no": self.pregunta_si_no}


def _sanear(texto: str) -> str:
    """El nombre, con la MISMA regla con la que el generador nombra sus campos.

    Se importa `_identifier` en vez de reescribir la normalización: si el nombre
    leído aquí y el nombre escrito en el `.mxai` se sanean con reglas distintas,
    comparar uno con otro deja de significar nada — y comparar es justo para lo
    que existe este módulo.
    """
    from matrixai.training.dense_generator import _identifier  # noqa: PLC0415

    return _identifier(texto)


def _nombre_de_la_frase(segmento: str) -> str | None:
    """El nombre del objetivo dentro del segmento que dice qué se predice."""
    palabras = re.findall(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ_]+", segmento)
    # Se busca el verbo, y lo que interesa es lo que viene DESPUÉS.
    inicio = None
    for i, palabra in enumerate(palabras):
        baja = palabra.lower()
        if any(baja.startswith(raiz) for raiz in _RAICES_DE_PREDICCION):
            inicio = i + 1
            break
    if inicio is None:
        return None

    resto = palabras[inicio:]
    while resto and resto[0].lower() in _RELLENO:
        resto = resto[1:]

    nombre: list[str] = []
    for i, palabra in enumerate(resto):
        baja = palabra.lower()
        if len(nombre) >= _MAX_PALABRAS:
            break
        if baja in _CORTES:
            # UN ENLACE NO SIEMPRE CORTA. «Clasificar el nivel de riesgo» tiene
            # su nombre partido en dos, y cortar en el «de» devolvía «nivel»,
            # que no es lo que nadie quiso decir. Se continúa UNA vez, y solo si
            # lo que sigue es otro sustantivo: en «el precio de una vivienda»
            # detrás del «de» hay relleno («una»), que es la señal de que
            # empieza el complemento y el nombre ya terminó.
            siguiente = resto[i + 1].lower() if i + 1 < len(resto) else ""
            if (baja in _ENLACES and nombre and siguiente
                    and siguiente not in _RELLENO and siguiente not in _CORTES):
                nombre.append(palabra)
                nombre.append(resto[i + 1])
            break
        nombre.append(palabra)
    if not nombre:
        return None
    return _sanear(" ".join(nombre)) or None


def objetivo_declarado(prompt: str) -> ObjetivoDelPrompt | None:
    """Qué dice este prompt que hay que predecir, o `None` si no lo dice.

    `None` NO es un fallo: es la respuesta correcta a «analizar los datos de
    clientes de una empresa de telefonía», y quien llame tiene que preguntar en
    vez de continuar. El 103-C1 lo dice con todas las letras: *ruta prompt sin
    objetivo: pedir definición/columna; no crear un objetivo ficticio para
    continuar*.
    """
    texto = str(prompt or "")
    if not texto.strip():
        return None

    # 1. Una declaración explícita manda sobre la frase.
    declaracion_del_core = False
    for match in _DECLARACION_RE.finditer(texto):
        crudo = match.group("nombre").strip()
        nombre = _sanear(crudo)
        if not nombre or nombre in NOMBRES_INVENTADOS:
            # Un `OUTPUT predicted_class:` no es una declaración de nadie: lo
            # escribió el generador. Se sigue mirando por si hay otra.
            declaracion_del_core = declaracion_del_core or nombre in NOMBRES_INVENTADOS
            continue
        return ObjetivoDelPrompt(nombre=nombre, definicion=crudo, fuente="declaracion")
    if declaracion_del_core:
        # ESTE PROMPT YA VIENE DE UNA VUELTA DEL CORE y lo único que declara es
        # el nombre que el core se inventó. Seguir leyendo la frase encontraría
        # la raíz «predic» dentro de `predicted_class` y sacaría de ahí un
        # objetivo —medido: devolvía `probabilitymap`—, que es exactamente la
        # invención que este módulo existe para no repetir.
        return None

    # 2. La frase. El corte entre objetivo y columnas es el del contrato 70 C1,
    #    y se pide al core en vez de repetirlo: dos listas de conectores acaban
    #    divergiendo, y ya hay una medida y auditada dos veces.
    from matrixai.training.dense_generator import lectura_del_objetivo  # noqa: PLC0415

    segmento, es_si_no = lectura_del_objetivo(texto)
    if es_si_no:
        # La definición está («si un cliente va a impagar»), el nombre no.
        return ObjetivoDelPrompt(nombre=None, definicion=segmento.strip(),
                                 fuente="frase", pregunta_si_no=True)
    nombre = _nombre_de_la_frase(segmento)
    if nombre is None or nombre in NOMBRES_INVENTADOS:
        return None
    return ObjetivoDelPrompt(nombre=nombre, definicion=segmento.strip(), fuente="frase")
