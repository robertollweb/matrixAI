# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice la confirmación del objetivo, EN LOS DOS IDIOMAS — 103-C1.

Mismo patrón que `matrixai/estudio/textos.py`, y por la misma razón: **lo que
redacta el core se traduce en el core**. «Esta columna es a la vez el objetivo y
una entrada» lo lee una persona y decide con ello; dejar media frase en inglés
para que la pantalla la remate es lo que ya costó un barrido entero de idioma en
este producto.

Y la parte fina, copiada tal cual del 104-C0: **no se traduce la frase, se
compone el hecho**. Quien pregunta o bloquea pasa una CLAVE y sus campos, y de
ahí salen las dos redacciones. Guardar en el estado una cadena ya compuesta hace
que al cambiar de idioma no cambie.

POR QUÉ UN CATÁLOGO PROPIO Y NO EL DEL 104-C0. El de `estudio/textos.py` es el
de los RECHAZOS DE ESQUEMA: un documento imposible que alguien intentó
construir. Éste es el de las PREGUNTAS: lo que todavía no se sabe del problema y
hay que preguntar antes de entrenar. Son dos vocabularios con dueños distintos
—el 104-C0 escribe el suyo, éste es del 103— y mezclarlos obligaría a los dos
cortes a editar el mismo fichero. Los dos comparten la FORMA (clave → {es, en},
`motivo()` que compone) a propósito, para que quien enseñe uno pueda enseñar el
otro sin distinguirlos.

Los dos diccionarios de cada clave van uno debajo del otro **a propósito**: así
salta a la vista cuando una se queda sin pareja. Hay una prueba que compara las
claves Y los huecos `{...}` de cada plantilla, porque un hueco que existe en
castellano y no en inglés solo revienta cuando alguien pide inglés.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

#: Los idiomas que este catálogo sabe escribir. No es una lista abierta.
IDIOMAS = ("es", "en")

#: Clave -> {idioma: plantilla}. Los huecos van con `str.format`.
MOTIVOS: dict[str, dict[str, str]] = {
    # ---- preguntas: lo que hay que confirmar antes de entrenar -----------
    "objetivo_no_declarado": {
        "es": "esta descripción no dice QUÉ hay que predecir: {valor}. Escribe el "
              "objetivo («predecir el precio de la vivienda…») o nombra la columna "
              "que lo contiene. No se inventa uno para poder seguir: un modelo "
              "entrenado contra un objetivo que nadie pidió acierta sobre algo que "
              "a nadie le importa",
        "en": "this description does not say WHAT is to be predicted: {valor}. Write "
              "the target («predict the price of the house…») or name the column that "
              "holds it. One is not invented so things can carry on: a model trained "
              "against a target nobody asked for scores well on something nobody cares "
              "about",
    },
    "objetivo_sin_nombre": {
        "es": "la frase dice QUÉ se predice ({valor}) pero no le pone NOMBRE ni "
              "nombra la columna que lo contiene. Ponle nombre («impago», "
              "«llueve_manana»): sin él no hay nada que comparar con las entradas, "
              "ni nada que escribir en el manifiesto",
        "en": "the phrase says WHAT is predicted ({valor}) but gives it no NAME and "
              "names no column holding it. Name it («default», «rains_tomorrow»): "
              "without a name there is nothing to compare against the inputs, and "
              "nothing to write in the manifest",
    },
    "clases_no_declaradas": {
        "es": "no se han podido leer las clases de {campo} en la descripción, y no se "
              "usan las de ejemplo: un modelo con clases de relleno emite "
              "probabilidades que no significan nada. Nómbralas —«clasificar en alto, "
              "medio o bajo», o ProbabilityMap[alto, medio, bajo]",
        "en": "the classes of {campo} could not be read from the description, and the "
              "example ones are not used: a model with placeholder classes emits "
              "probabilities that mean nothing. Name them — «classify as high, medium "
              "or low», or ProbabilityMap[high, medium, low]",
    },
    "objetivo_no_elegido": {
        "es": "hay que elegir la columna objetivo entre las candidatas {opciones}; "
              "cada una viene con su motivo. La propuesta es una propuesta: quien "
              "conoce los datos decide",
        "en": "the target column has to be chosen among the candidates {opciones}; "
              "each one comes with its reason. The proposal is a proposal: whoever "
              "knows the data decides",
    },
    "tipo_de_tarea": {
        "es": "el objetivo {campo} es numérico y solo toma {valor} valores distintos "
              "({opciones}): pueden ser {valor} clases o una magnitud que se repite. "
              "Dilo tú — decidirlo en silencio cambia la salida del modelo, las "
              "métricas y lo que significa cada número",
        "en": "target {campo} is numeric and takes only {valor} distinct values "
              "({opciones}): they may be {valor} classes or a magnitude that repeats. "
              "You say which — deciding it silently changes the model's output, the "
              "metrics and what every number means",
    },
    "clase_positiva": {
        "es": "una clasificación binaria sobre {campo} necesita saber cuál de "
              "{opciones} es la clase POSITIVA: sin ella, sensibilidad, VPP y el "
              "umbral quedan sin definir, y elegirla por orden alfabético es "
              "inventarse la mitad del problema",
        "en": "a binary classification on {campo} needs to know which of {opciones} is "
              "the POSITIVE class: without it sensitivity, PPV and the threshold are "
              "undefined, and picking it alphabetically is inventing half the problem",
    },
    "unidad_de_observacion": {
        "es": "falta decir qué representa UNA FILA (un paciente, un episodio, un día "
              "de una ciudad…). No se rellena con «una fila»: de esto depende que la "
              "partición no ponga al mismo sujeto a los dos lados, y eso no se puede "
              "adivinar mirando el fichero",
        "en": "it is not stated what ONE ROW represents (a patient, an episode, a city "
              "day…). It is not filled in with «a row»: whether the split keeps the "
              "same subject off both sides depends on this, and it cannot be guessed "
              "from the file",
    },
    "momento_de_prediccion": {
        "es": "estos datos traen fecha ({opciones}) y no se ha dicho EN QUÉ MOMENTO se "
              "predice. Sin ese momento no se puede decir qué variable estaba "
              "disponible y cuál llegó después del desenlace",
        "en": "this data carries a date ({opciones}) and it has not been said WHEN the "
              "prediction is made. Without that moment there is no telling which "
              "variable was available and which arrived after the outcome",
    },
    "horizonte_del_desenlace": {
        "es": "se predice desde «{campo}» y falta el HORIZONTE: cuánto tiempo hacia "
              "delante mira el desenlace. «Ingresará» a 48 horas y a un año no son el "
              "mismo problema ni se miden igual",
        "en": "the prediction is made at «{campo}» and the HORIZON is missing: how far "
              "ahead the outcome is looked for. «Will be admitted» within 48 hours and "
              "within a year are not the same problem and are not measured alike",
    },
    # ---- bloqueos: errores estructurales del diseño ----------------------
    "sin_entradas_utilizables": {
        "es": "no queda ninguna columna con la que predecir: {excluidas} quedaron fuera "
              "(ver su motivo) y no hay otras entradas. Un estudio sin entradas no puede "
              "aprender nada; añade al CSV alguna columna que se conozca en el momento de "
              "predecir",
        "en": "no column is left to predict with: {excluidas} were left out (see their "
              "reason) and there are no other inputs. A study without inputs cannot learn "
              "anything; add to the CSV some column that is known at prediction time",
    },
    "texto_libre_excluido": {
        "es": "{campo} es texto libre escrito por una persona (mediana de {palabras} "
              "palabras por valor; {distintas} de cada 100 palabras son distintas) y "
              "queda fuera del estudio: como categoría, cada valor sería casi único, "
              "el modelo solo podría memorizar las filas de entrenamiento y en las "
              "nuevas saldría «desconocida». Usar texto libre en el estudio llegará "
              "con su propio camino (contrato 107)",
        "en": "{campo} is free text written by a person (median of {palabras} words per "
              "value; {distintas} out of every 100 words are distinct) and is left out "
              "of the study: as a category every value would be almost unique, the "
              "model could only memorise the training rows and new rows would come "
              "out as «unknown». Using free text in the study will come with its own "
              "path (contract 107)",
    },
    "objetivo_entre_las_entradas": {
        "es": "{campo} es a la vez el objetivo confirmado y una entrada: el modelo "
              "acertaría el 100 % copiando la respuesta que ya le damos, y ese 100 % "
              "no significa nada. Quita la entrada, o corrige el objetivo si la "
              "columna que había que predecir es otra",
        "en": "{campo} is at once the confirmed target and an input: the model would "
              "score 100 % by copying the answer we already hand it, and that 100 % "
              "means nothing. Remove the input, or correct the target if the column to "
              "predict was another one",
    },
    "objetivo_con_una_sola_clase": {
        "es": "el objetivo confirmado {campo} solo toma un valor ({valor}) en las "
              "{opciones} filas de entrenamiento: no es entrenable para esta tarea. Un "
              "modelo así llega a pérdida 0 respondiendo siempre lo mismo, y ese 0 se "
              "lee como acierto perfecto. Trae datos en los que el resultado cambie, o "
              "reparte las filas de otra forma",
        "en": "confirmed target {campo} takes a single value ({valor}) across the "
              "{opciones} training rows: it is not trainable for this task. Such a "
              "model reaches loss 0 by always answering the same thing, and that 0 "
              "reads as a perfect score. Bring data where the outcome changes, or split "
              "the rows differently",
    },
    "objetivo_inexistente": {
        "es": "la columna objetivo {campo} no está en los datos. Columnas: {opciones}",
        "en": "target column {campo} is not in the data. Columns: {opciones}",
    },
    "objetivo_no_predecible": {
        "es": "la columna {campo} es de tipo {valor} y no se puede predecir: un "
              "identificador no significa nada fuera de su fila y una fecha no es un "
              "desenlace. Corrige su tipo si el análisis se equivocó, o elige otra "
              "columna",
        "en": "column {campo} is of type {valor} and cannot be predicted: an identifier "
              "means nothing outside its row and a date is not an outcome. Correct its "
              "type if the analysis got it wrong, or choose another column",
    },
    "clases_declaradas_que_no_estan": {
        "es": "se declaran como clases de {campo} valores que no aparecen en los datos: "
              "{opciones}. Una clase sin un solo ejemplo no se puede aprender ni medir",
        "en": "values declared as classes of {campo} do not appear in the data: "
              "{opciones}. A class without a single example can neither be learned nor "
              "measured",
    },
    # ---- pistas: señales por NOMBRE, que no son veredictos ---------------
    #
    # El contrato 71 lo dejó escrito con un caso medido: un detector por parecido
    # de nombres marcaría `last_year_salary` y acusaría a una columna legítima —el
    # sueldo del año pasado predice bien el siguiente—. Así que esto se dice como
    # lo que es: una señal para mirar, nunca una conclusión.
    "entrada_con_el_nombre_del_objetivo": {
        "es": "la entrada {campo} se llama igual que el objetivo {valor}. Es una PISTA, "
              "no un veredicto: si son la misma variable, el modelo se copiaría la "
              "respuesta; si son dos cosas distintas con el mismo nombre, no pasa nada. "
              "Confírmalo tú",
        "en": "input {campo} is named the same as target {valor}. This is a HINT, not a "
              "verdict: if they are the same variable the model would copy the answer; "
              "if they are two different things sharing a name, nothing is wrong. You "
              "confirm it",
    },
    "entrada_que_contiene_el_objetivo": {
        "es": "el nombre de la entrada {campo} contiene el del objetivo {valor}. Es una "
              "PISTA, no un veredicto: «el salario del año pasado» predice el salario "
              "de este año y es legítimo, y «el precio» como entrada para predecir «el "
              "precio» no lo es. Lo que decide es si son la misma variable, y eso lo "
              "sabes tú",
        "en": "the name of input {campo} contains that of target {valor}. This is a "
              "HINT, not a verdict: «last year's salary» predicts this year's salary "
              "and is legitimate, while «the price» as an input to predict «the price» "
              "is not. What decides is whether they are the same variable, and only you "
              "know that",
    },
    # ---- el rechazo de arrancar sin problema confirmado ------------------
    "estudio_sin_problema_confirmado": {
        "es": "no se puede empezar el estudio: el problema no está confirmado y quedan "
              "{valor} cosa(s) por decidir ({opciones}). Un objetivo inventado para "
              "poder continuar produce un modelo que nadie pidió",
        "en": "the study cannot start: the problem is not confirmed and {valor} "
              "thing(s) remain to be decided ({opciones}). A target invented so things "
              "can carry on produces a model nobody asked for",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla. Se usa para comprobar que las dos
    redacciones piden LOS MISMOS campos: si el castellano usa `{opciones}` y el
    inglés no, el fallo solo aparece cuando alguien pide inglés."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas.

    Devuelve un `dict` y no una cadena a propósito: **no se guarda en el estado
    una cadena ya compuesta en un idioma**, porque al cambiar de idioma no
    cambiaría. Quien pinte elige; quien registra guarda los dos.
    """
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de motivos del 103-C1: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
