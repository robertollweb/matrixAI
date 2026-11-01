# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice la propuesta de partición, EN LOS DOS IDIOMAS — 103-C4.

Mismo patrón que `objetivo_textos.py`, `diagnostico_textos.py` y
`preparacion_textos.py`: cada corte de este contrato tiene su propio
catálogo porque cada uno redacta sobre un momento distinto del flujo —
aquí, el de decidir CÓMO se van a repartir las filas, no qué se mide ni
qué se prepara.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    "grupos_insuficientes": {
        "es": "{campo} tiene {valor} grupo(s) distintos: hacen falta al menos "
              "2 para separar desarrollo de una prueba que generalice a "
              "unidades nuevas. No se rompe un grupo entre roles para "
              "conseguir dos partes artificialmente",
        "en": "{campo} has {valor} distinct group(s): at least 2 are needed "
              "to separate development from a test that generalizes to new "
              "units. A group is never split across roles just to force two parts",
    },
    "pliegues_reducidos_por_grupos": {
        "es": "se pidieron {opciones} pliegues y solo hay {valor} grupos en "
              "desarrollo: se reduce el número de pliegues a {valor} en vez "
              "de romper un grupo entre dos pliegues para completar el resto",
        "en": "{opciones} folds were requested and development only has "
              "{valor} groups: the fold count is reduced to {valor} instead "
              "of splitting a group across two folds to fill the rest",
    },
    "pliegues_reducidos_por_eventos": {
        "es": "se pidieron {opciones} pliegues y la clase minoritaria de "
              "desarrollo solo tiene {valor} evento(s): se reduce el número "
              "de pliegues a {valor} para que cada pliegue de validación "
              "pueda tener al menos uno",
        "en": "{opciones} folds were requested and development's minority "
              "class only has {valor} event(s): the fold count is reduced to "
              "{valor} so each validation fold can have at least one",
    },
    "una_sola_clase_no_estratifica": {
        "es": "{campo} tiene una sola clase en desarrollo: no hay nada que "
              "estratificar, los pliegues se reparten por filas sin más",
        "en": "{campo} has a single class in development: there is nothing "
              "to stratify, folds are assigned by row alone",
    },
    "cronologia_no_se_pudo_separar": {
        "es": "con la fracción de prueba pedida y el hueco declarado, no "
              "queda ninguna fila del lado de desarrollo, o ninguna del lado "
              "de prueba: el diseño temporal pedido no es viable con estos datos",
        "en": "with the requested test fraction and the declared gap, no rows "
              "remain on the development side, or none on the test side: the "
              "requested temporal design is not viable with this data",
    },

    # -- 140-C1: guardas de grupos y de tiempo ------------------------------
    "falta_columna_de_tiempo": {
        "es": "el diseño {valor} necesita la columna de tiempo (time_column) y no se "
              "ha indicado ninguna: sin ella no hay un orden hacia delante que respetar",
        "en": "the {valor} design needs the time column (time_column) and none was "
              "given: without it there is no forward order to respect",
    },
    "tiempo_ilegible": {
        "es": "{campo}: {valor} valor(es) no se pueden leer como fecha ni como número "
              "(por ejemplo {ejemplos}). No se ordena por una columna que no se entiende: "
              "una fila sin fecha no se puede colocar antes ni después del corte",
        "en": "{campo}: {valor} value(s) cannot be read as a date or as a number (for "
              "example {ejemplos}). Rows are not ordered by a column that is not "
              "understood: a row without a date cannot be placed before or after the cut",
    },
    "fecha_ambigua": {
        "es": "{campo} se lee igual de bien como día/mes/año que como mes/día/año (por "
              "ejemplo {ejemplos}): no se elige por orden, porque la mitad de las fechas "
              "cambiaría de mes y el corte caería en otro sitio. Escríbelas sin "
              "ambigüedad (AAAA-MM-DD) o confirma el orden",
        "en": "{campo} reads equally well as day/month/year and as month/day/year (for "
              "example {ejemplos}): the order is not chosen by default, because half of "
              "the dates would change month and the cut would fall elsewhere. Write them "
              "unambiguously (YYYY-MM-DD) or confirm the order",
    },
    "hueco_ilegible": {
        "es": "el hueco (gap) {valor} no se entiende: se espera una magnitud no negativa "
              "y una unidad de {opciones} (por ejemplo 30 days)",
        "en": "the gap {valor} is not understood: expected a non-negative magnitude and "
              "a unit among {opciones} (for example 30 days)",
    },
    "una_sola_clase_sin_guardas": {
        "es": "{campo} tiene una sola clase en los datos: no hay clase que repartir "
              "entre pliegues ni prueba, y las guardas de clase no se aplican",
        "en": "{campo} has a single class in the data: there is no class to spread "
              "across folds or the test, and the class guards do not apply",
    },
    "repeticiones_fijadas_a_1": {
        "es": "se pidieron {opciones} repeticiones y el reparto temporal solo tiene "
              "una: no se puede barajar sin mezclar pasado y futuro, y repetirlo daría "
              "los mismos pliegues; se usa {valor}",
        "en": "{opciones} repeats were requested and the temporal split has only one: "
              "it cannot be shuffled without mixing past and future, and repeating it "
              "would give the same folds; {valor} is used",
    },
    "tiempo_numerico_sin_unidad": {
        "es": "{campo} se lee como un número, no como una fecha, y no dice en qué unidad está: "
              "el hueco en segundos se aplica restando ese número tal cual",
        "en": "{campo} is read as a number, not as a date, and does not say its unit: the gap "
              "in seconds is applied by subtracting that number as is",
    },
    "instantes_insuficientes": {
        "es": "desarrollo solo tiene {valor} instante(s) distintos en {campo}: la "
              "validación hacia delante necesita al menos {minimo} (un bloque que solo "
              "entrena y uno o más que se validan), y las filas del mismo instante no "
              "se separan nunca",
        "en": "development has only {valor} distinct instant(s) in {campo}: forward "
              "validation needs at least {minimo} (a block that only trains and one or "
              "more that are validated), and rows with the same instant are never separated",
    },
    "pliegues_reducidos_por_instantes": {
        "es": "se pidieron {opciones} pliegues y desarrollo solo tiene {n} instantes "
              "distintos: se reducen a {valor} (las filas del mismo instante no se separan)",
        "en": "{opciones} folds were requested and development only has {n} distinct "
              "instants: reduced to {valor} (rows with the same instant are not separated)",
    },
    "pliegues_reducidos_por_hueco": {
        "es": "se pidieron {opciones} pliegues y con el hueco declarado el pliegue más "
              "antiguo se quedaría sin ninguna fila para entrenar: se reducen a {valor}",
        "en": "{opciones} folds were requested and with the declared gap the oldest fold "
              "would be left with no row to train on: reduced to {valor}",
    },
    "pliegues_reducidos_por_clase_en_bloques": {
        "es": "se pidieron {opciones} pliegues y con ellos algún bloque de validación o "
              "de entrenamiento se quedaría sin ninguna fila de la clase {clase}: se "
              "reducen a {valor} para que todos la tengan",
        "en": "{opciones} folds were requested and with them some validation or training "
              "block would have no row of class {clase}: reduced to {valor} so every "
              "block has it",
    },
    "bloques_sin_clase": {
        "es": "la clase {clase} solo tiene {valor} fila(s) en desarrollo y, repartidas en "
              "el tiempo, ni con {opciones} pliegues (el mínimo) están en todos los "
              "bloques (filas de la clase por bloque, del más antiguo al más reciente: "
              "{detalle}). Un pliegue sin esa clase no la mide: el estudio no arranca",
        "en": "class {clase} has only {valor} row(s) in development and, spread over "
              "time, not even with {opciones} folds (the minimum) is it in every block "
              "(rows of the class per block, oldest to newest: {detalle}). A fold "
              "without that class does not measure it: the study does not start",
    },
    "prueba_sin_una_clase": {
        "es": "la prueba reservada ({filas} filas, la parte más reciente) no tiene "
              "ninguna fila de la clase {clase}: una nota sin esa clase no se puede "
              "medir. Cambia la fracción de prueba o el momento del corte",
        "en": "the reserved test ({filas} rows, the most recent part) has no row of "
              "class {clase}: a score without that class cannot be measured. Change the "
              "test fraction or the cut",
    },
    "pliegues_reducidos_por_clase_en_grupos": {
        "es": "se pidieron {opciones} pliegues y la clase {clase} solo aparece en {n} "
              "grupo(s) de {campo}: para que cada pliegue y la prueba reservada tengan "
              "alguna fila de la clase hacen falta {partes} grupos con ella; se reducen "
              "los pliegues a {valor}",
        "en": "{opciones} folds were requested and class {clase} only appears in {n} "
              "group(s) of {campo}: for every fold and the reserved test to have some "
              "row of the class, {partes} groups with it are needed; folds reduced to {valor}",
    },
    "clase_sin_grupos_suficientes": {
        "es": "la clase {clase} solo aparece en {valor} grupo(s) de {campo}; con {opciones} "
              "pliegues y la prueba reservada hacen falta {partes}, y aun bajando a 2 "
              "pliegues hacen falta {minimo}. No se puede repartir la clase entre grupos "
              "sin partir un grupo, y un examen sin esa clase no la mide: el estudio no arranca",
        "en": "class {clase} only appears in {valor} group(s) of {campo}; with {opciones} "
              "folds and the reserved test, {partes} are needed, and even dropping to 2 "
              "folds {minimo} are needed. The class cannot be spread across groups "
              "without splitting a group, and an exam without that class does not "
              "measure it: the study does not start",
    },
    "clases_sin_reparto_por_grupos": {
        "es": "la clase {clase} aparece en {valor} grupo(s) de {campo}, los que hacen "
              "falta, pero los grupos que traen las clases coinciden y no se ha encontrado un "
              "reparto en el que la prueba reservada y cada pliegue tengan todas las clases, "
              "ni con 2 pliegues: el estudio no arranca",
        "en": "class {clase} appears in {valor} group(s) of {campo}, as many as needed, "
              "but the groups carrying the classes overlap and no split was found in which "
              "the reserved test and every fold have all classes, not even with 2 folds: "
              "the study does not start",
    },
    "valida_con_una_sola_unidad": {
        "es": "la validación solo tiene un valor de {campo}: no se puede partir en dos mitades sin "
              "partir esa unidad, así que las mitades se cortan por posición y comparten "
              "unidad",
        "en": "the validation fold has a single value of {campo}: it cannot be halved without "
              "splitting that unit, so the halves are cut by position and share a unit",
    },
    "valida_con_un_solo_instante": {
        "es": "todas las filas de la validación tienen el mismo instante en {campo}: no "
              "hay un antes y un después, y las mitades se cortan por posición",
        "en": "all validation rows share the same instant in {campo}: there is no before "
              "and after, and the halves are cut by position",
    },
    "mitad_sin_una_clase": {
        "es": "la clase {clase} solo está en una de las dos mitades de la validación "
              "({detalle}): no hay manera de repartirla sin partir una unidad",
        "en": "class {clase} is only in one of the two validation halves ({detalle}): "
              "there is no way to spread it without splitting a unit",
    },
    "hueco_sin_filas_para_entrenar": {
        "es": "con el hueco declarado ({valor}) no queda ninguna fila para entrenar "
              "antes del primer bloque de validación, ni con 2 pliegues: reduce el hueco "
              "o aporta más historia",
        "en": "with the declared gap ({valor}) no row is left to train on before the "
              "first validation block, not even with 2 folds: reduce the gap or provide "
              "more history",
    },
    "pliegues_reducidos_por_reparto_de_clases_en_grupos": {
        "es": "se pidieron {opciones} pliegues y, aunque la clase {clase} aparece en {n} "
              "grupo(s) de {campo}, los grupos que traen las clases coinciden y con ese "
              "número de pliegues algún pliegue o la prueba se quedarían sin una clase: "
              "se reducen los pliegues a {valor}",
        "en": "{opciones} folds were requested and, although class {clase} appears in "
              "{n} group(s) of {campo}, the groups carrying the classes overlap and with "
              "that many folds some fold or the test would lack a class: folds reduced "
              "to {valor}",
    },

    # -- reparación de la auditoría del 140-C1 (27-10) --------------------------
    "unidad_vacia": {
        "es": "{campo} está vacía en {valor} de {n} filas: una fila sin sujeto no se puede poner en desarrollo "
              "ni en la prueba sin adivinar de quién es. Rellena esas filas o quítalas",
        "en": "{campo} is empty in {valor} of {n} rows: a row without a subject cannot be put in development or "
              "in the test without guessing whose it is. Fill those rows in or remove them",
    },
    "cohorte_externa_comparte_unidad": {
        "es": "la cohorte externa comparte {valor} valor(es) de {campo} con el resto de los datos (por ejemplo "
              "{ejemplos}): una cohorte externa mide a sujetos que el modelo no ha visto, y estos sí los vería",
        "en": "the external cohort shares {valor} value(s) of {campo} with the rest of the data (for example "
              "{ejemplos}): an external cohort measures subjects the model has not seen, and it would see these",
    },
    "sin_prueba_reservada": {
        "es": "con la fracción de prueba pedida ({opciones}) y {n} filas, la prueba reservada se queda sin "
              "ninguna fila: sin ella no hay una nota medida con datos que el estudio no haya usado. Pide una "
              "fracción mayor que 0",
        "en": "with the requested test fraction ({opciones}) and {n} rows, the reserved test gets no row at all: "
              "without it there is no score measured on data the study has not used. Ask for a fraction above 0",
    },
    "grupos_insuficientes_para_prueba_y_pliegues": {
        "es": "{campo} tiene {valor} grupos distintos: la prueba reservada se lleva al menos uno entero y los "
              "pliegues necesitan otros {opciones} (uno por pliegue), así que hacen falta {partes}; aun bajando "
              "a 2 pliegues hacen falta {minimo}. No se parte un grupo para completarlos: el estudio no arranca",
        "en": "{campo} has {valor} distinct groups: the reserved test takes at least one whole group and the folds "
              "need {opciones} more (one per fold), so {partes} are needed; even dropping to 2 folds {minimo} are "
              "needed. A group is never split to fill them: the study does not start",
    },
    "prueba_recortada_para_dejar_grupos": {
        "es": "la prueba reservada se queda con {valor} filas en vez de {n}: con {partes} grupos de {campo}, "
              "el grupo siguiente habría dejado a desarrollo con menos de {minimo} y los pliegues sin nada que "
              "validar",
        "en": "the reserved test keeps {valor} rows instead of {n}: with {partes} groups of {campo}, the next "
              "group would have left development with fewer than {minimo} and the folds with nothing to validate",
    },
    "hueco_en_tiempo_numerico": {
        "es": "{campo} se lee como un número, no como una fecha (por ejemplo {ejemplos}), y un número no dice "
              "en qué unidad está: un hueco de {valor} no se le puede aplicar sin saber cuánto es un día en esa "
              "columna. Escribe la columna como fecha (AAAA-MM-DD, o DD/MM/AAAA) o quita el hueco",
        "en": "{campo} is read as a number, not as a date (for example {ejemplos}), and a number does not say "
              "its unit: a gap of {valor} cannot be applied without knowing how much a day is in that column. "
              "Write the column as a date (YYYY-MM-DD, or DD/MM/YYYY) or remove the gap",
    },
    "entrena_sin_clase": {
        "es": "ni con {opciones} pliegues (el mínimo) cada entrenamiento tiene la clase {clase}: el pliegue {valor} "
              "(0 es el más reciente) entrenaría con {n} filas y ninguna es {clase}, porque la primera fila con "
              "{clase} está en {campo} = {detalle}. Un modelo que no ha visto una clase no aprende a reconocerla: "
              "el estudio no arranca. Aporta más historia con esa clase o empieza los datos más cerca de {detalle}",
        "en": "not even with {opciones} folds (the minimum) does every training set have class {clase}: fold {valor} "
              "(0 is the most recent) would train on {n} rows and none is {clase}, because the first row with "
              "{clase} is at {campo} = {detalle}. A model that has never seen a class does not learn to recognise "
              "it: the study does not start. Provide more history with that class or start the data closer to {detalle}",
    },
    "entrena_sin_clase_por_el_hueco": {
        "es": "ni con {opciones} pliegues (el mínimo) cada entrenamiento tiene la clase {clase}: el pliegue {valor} "
              "(0 es el más reciente) se valida desde {campo} = {inicio}, y el hueco declarado ({minimo}) le quita "
              "las {partes} fila(s) de {clase} que tenía antes (la última, en {campo} = {detalle}): entrenaría con {n} "
              "filas y ninguna es {clase}. Un modelo que no ha visto una clase no aprende a reconocerla: el estudio "
              "no arranca. La causa es el hueco, no el comienzo de los datos: reduce el hueco o aporta más historia "
              "con esa clase antes de {detalle}",
        "en": "not even with {opciones} folds (the minimum) does every training set have class {clase}: fold {valor} "
              "(0 is the most recent) is validated from {campo} = {inicio}, and the declared gap ({minimo}) removes "
              "the {partes} row(s) of {clase} it had before (the last one at {campo} = {detalle}): it would train on "
              "{n} rows and none is {clase}. A model that has never seen a class does not learn to recognise it: the "
              "study does not start. The cause is the gap, not where the data starts: reduce the gap or provide more "
              "history with that class before {detalle}",
    },
    "fraccion_de_prueba_desviada": {
        "es": "se pidió una prueba reservada del {opciones} % de las filas y sale del {valor} % ({n} de {partes}): "
              "los grupos de {campo} no se parten y la prueba tiene que llevar todas las clases, y este es el "
              "reparto más cercano que se ha encontrado (el grupo más grande de la prueba, {grupo}, trae {minimo} filas)",
        "en": "a reserved test of {opciones} % of the rows was requested and it comes out at {valor} % ({n} of "
              "{partes}): groups of {campo} are never split and the test must carry every class, and this is the "
              "closest split found (the largest group in the test, {grupo}, has {minimo} rows)",
    },

    # -- auditoría I5 del 140b: el mismo sujeto escrito de dos maneras (groups y groups_and_time) -------------
    "sujetos_que_se_escriben_igual": {
        "es": "en {campo} hay {valor} grupo(s) de valores distintos ({valores} valores, {filas} filas) que se escriben igual "
              "sin los ceros a la izquierda, los espacios y las mayúsculas: {ejemplos}. Si son el mismo sujeto, el reparto "
              "los trata como sujetos distintos y pueden caer uno en el desarrollo y otro en la prueba. No se unifican solos: "
              "si lo son, corrígelo en los datos",
        "en": "in {campo} there are {valor} group(s) of distinct values ({valores} values, {filas} rows) that read the same "
              "without leading zeros, spaces and capital letters: {ejemplos}. If they are the same subject, the split treats "
              "them as different subjects and one may land in development and the other in the test. They are not merged "
              "automatically: if they are the same, fix it in the data",
    },

    # -- 140b-C1: «por sujeto y por el tiempo» estricto ----------------------
    "estricto_sin_sujetos_con_filas_posteriores": {
        "es": "con el corte en {tiempo} = {corte} y un hueco de {hueco}, ningún valor de {unidad} tiene filas "
              "posteriores (el último {tiempo} es {ultimo}): este reparto elige los sujetos de prueba entre los que "
              "tienen filas después del corte, y aquí no hay ninguno. El corte lo elige el reparto (el cuantil del "
              "tiempo): lo que se puede cambiar es el hueco, más pequeño o ninguno",
        "en": "with the cut at {tiempo} = {corte} and a gap of {hueco}, no value of {unidad} has later rows (the last "
              "{tiempo} is {ultimo}): this split picks the test subjects among those with rows after the cut, and "
              "there are none. The split picks the cut (a quantile of time): what can be changed is the gap, smaller "
              "or none",
    },
    "estricto_sujeto_y_tiempo_la_misma_columna": {
        "es": "{campo} es a la vez la columna del sujeto y la del tiempo: así cada instante sería un «sujeto» distinto y el "
              "reparto separaría solo por el tiempo, sin apartar a ningún sujeto. Elige como sujeto la columna que identifica "
              "a la persona (o la unidad) y como tiempo la de la fecha",
        "en": "{campo} is both the subject column and the time column: each instant would then be a different “subject” and "
              "the split would only separate by time, without setting any subject apart. Choose as subject the column that "
              "identifies the person (or the unit) and as time the date column",
    },
    "estricto_prueba_con_pocas_unidades": {
        "es": "la prueba tendría {valor} valor(es) de {unidad} y hacen falta al menos {minimo}: con menos sujetos la "
              "nota depende de quiénes hayan caído en la prueba más que del modelo. Se probaron {n} cortes y ninguno "
              "llegó (el mejor deja {filas} filas de prueba; hay {elegibles} sujeto(s) con filas posteriores al corte)",
        "en": "the test would have {valor} value(s) of {unidad} and at least {minimo} are needed: with fewer subjects "
              "the score depends on who landed in the test more than on the model. {n} cuts were tried and none "
              "reached it (the best leaves {filas} test rows; {elegibles} subject(s) have rows after the cut)",
    },
    "estricto_prueba_con_pocas_filas": {
        "es": "la prueba tendría {valor} fila(s), de {partes} valor(es) de {unidad}, y hacen falta al menos {minimo}. "
              "Se probaron {n} cortes y ninguno llegó",
        "en": "the test would have {valor} row(s), from {partes} value(s) of {unidad}, and at least {minimo} are "
              "needed. {n} cuts were tried and none reached it",
    },
    "estricto_clase_en_pocos_sujetos_de_prueba": {
        "es": "la clase {clase} estaría en {valor} valor(es) de {unidad} de la prueba y hacen falta al menos "
              "{minimo}: con menos, lo que se mide de esa clase depende de una sola persona. De los sujetos con filas "
              "posteriores al corte (los únicos que pueden ser de prueba), {partes} la traen en esas filas; en todo el "
              "conjunto, {total}. Se probaron {n} cortes y ninguno llegó",
        "en": "class {clase} would be in {valor} value(s) of {unidad} in the test and at least {minimo} are needed: "
              "with fewer, what is measured for that class depends on a single person. Of the subjects with rows "
              "after the cut (the only ones that can be in the test), {partes} have it in those rows; in the whole "
              "dataset, {total}. {n} cuts were tried and none reached it",
    },
    "estricto_clase_sin_prueba_que_la_cubra": {
        "es": "la clase {clase} la traen {partes} de los sujetos con filas posteriores al corte (en todo el conjunto, "
              "{total}), pero no se ha encontrado una prueba que la lleve en al menos {minimo} valor(es) de {unidad} sin "
              "pasarse de la parte de sujetos que le toca. Se probaron {n} cortes y ninguno llegó",
        "en": "class {clase} is present in {partes} of the subjects with rows after the cut ({total} in the whole "
              "dataset), but no test was found that carries it in at least {minimo} value(s) of {unidad} without "
              "exceeding its share of subjects. {n} cuts were tried and none reached it",
    },
    "estricto_pliegues_sin_la_clase": {
        "es": "en el desarrollo la clase {clase} queda en {valor} valor(es) de {unidad}, y los pliegues por sujeto "
              "necesitan al menos {minimo} para que cada validación tenga alguna fila de ella; en todo el conjunto la "
              "traen {total} y la prueba se lleva {prueba}. Se probaron {n} cortes y ninguno llegó. Un modelo que no se "
              "valida en una clase no se elige por ella: el estudio no arranca",
        "en": "in development class {clase} is left in {valor} value(s) of {unidad}, and folds by subject need at "
              "least {minimo} so that every validation has some row of it; the whole dataset has it in {total} and "
              "the test takes {prueba}. {n} cuts were tried and none reached it. A model that is never validated on a "
              "class is not chosen for it: the study does not start",
    },
    "estricto_pliegues_sin_reparto_de_la_clase": {
        "es": "en el desarrollo la clase {clase} está en {valor} valor(es) de {unidad}, pero no se ha encontrado un "
              "reparto en pliegues por sujeto con la clase en cada validación; en todo el conjunto la traen {total} y "
              "la prueba se lleva {prueba}. Se probaron {n} cortes y ninguno llegó: el estudio no arranca",
        "en": "in development class {clase} is in {valor} value(s) of {unidad}, but no split into folds by subject "
              "with the class in every validation was found; the whole dataset has it in {total} and the test takes "
              "{prueba}. {n} cuts were tried and none reached it: the study does not start",
    },
    "estricto_desarrollo_sin_clase": {
        "es": "el desarrollo ({filas} filas) no tiene ninguna fila de la clase {clase}: los sujetos que la traen están "
              "en la prueba o sus filas son posteriores al corte. Un modelo que no ha visto una clase no aprende a "
              "reconocerla: el estudio no arranca",
        "en": "development ({filas} rows) has no row of class {clase}: the subjects that have it are in the test or "
              "their rows come after the cut. A model that has never seen a class does not learn to recognise it: "
              "the study does not start",
    },
    "estricto_desarrollo_vacio": {
        "es": "con este reparto el desarrollo tiene {filas} fila(s) de {valor} valor(es) de {unidad}, y hacen falta al "
              "menos {minimo} sujetos y {minimo_de_filas} filas, los mismos mínimos que la prueba (que tiene {prueba} "
              "sujetos y {filas_de_prueba} filas): con menos, el modelo se elige con lo que aportan dos o tres personas. El "
              "resto de los sujetos son de prueba o solo tienen filas posteriores al corte. Se probaron {n} cortes y ninguno "
              "llegó",
        "en": "with this split development has {filas} row(s) from {valor} value(s) of {unidad}, and at least {minimo} "
              "subjects and {minimo_de_filas} rows are needed, the same minimums as the test (which has {prueba} subjects "
              "and {filas_de_prueba} rows): with fewer, the model is chosen on what two or three people contribute. The "
              "other subjects are in the test or only have rows after the cut. {n} cuts were tried and none reached it",
    },
    "estricto_filas_descartadas": {
        "es": "este reparto deja fuera {valor} de {n} filas ({porcentaje} %): las de los sujetos de prueba anteriores "
              "al corte, las de los demás sujetos posteriores al corte y las que caen en el hueco. Ningún sujeto "
              "queda a los dos lados, y la prueba es posterior al desarrollo",
        "en": "this split leaves out {valor} of {n} rows ({porcentaje} %): the test subjects' rows before the cut, "
              "the other subjects' rows after the cut, and those inside the gap. No subject is on both sides, and "
              "the test comes after development",
    },
    "estricto_corte_relajado": {
        "es": "se pidió una prueba con el {pedida} % de los sujetos y el corte en el cuantil {cuantil_pedido}, y no "
              "cumplía los mínimos de la prueba: se repartió pidiendo el {usada} % con el cuantil {cuantil_usado}, y la "
              "prueba tiene {n} de los {sujetos} sujetos",
        "en": "a test with {pedida} % of the subjects and the cut at quantile {cuantil_pedido} was requested and it "
              "did not meet the test minimums: the split asked for {usada} % with quantile {cuantil_usado}, and the "
              "test has {n} of the {sujetos} subjects",
    },
    "estricto_pliegues_reducidos": {
        "es": "se pidieron {pedidos} pliegues y el último tramo del desarrollo solo permite {usados}: con más, algún "
              "pliegue se quedaría con menos de {minimo} filas de validación o sin una clase",
        "en": "{pedidos} folds were requested and the last stretch of development only allows {usados}: with more, "
              "some fold would have fewer than {minimo} validation rows or no row of some class",
    },
    "estricto_pliegues_por_sujeto_por_pocas_filas": {
        "es": "el desarrollo tiene {filas} filas, menos de {minimo}: los pliegues se hacen por sujeto sin mirar el "
              "tiempo. La prueba reservada sí es posterior al corte",
        "en": "development has {filas} rows, fewer than {minimo}: folds are made by subject without looking at time. "
              "The reserved test does come after the cut",
    },
    "estricto_pliegues_por_sujeto_por_ventana_vacia": {
        "es": "el último {fraccion} % de las filas del desarrollo no deja, ni con {pliegues} pliegues, validaciones "
              "con al menos {minimo} filas y todas las clases: los pliegues se hacen por sujeto sin mirar el tiempo. "
              "La prueba reservada sí es posterior al corte",
        "en": "the last {fraccion} % of development's rows does not leave, even with {pliegues} folds, validations "
              "with at least {minimo} rows and every class: folds are made by subject without looking at time. The "
              "reserved test does come after the cut",
    },
    "estricto_unidad_dominante_en_la_prueba": {
        "es": "el valor {dominante} de {unidad} aporta el {porcentaje} % de las filas de la prueba: lo que se mide "
              "depende sobre todo de ese sujeto",
        "en": "the value {dominante} of {unidad} contributes {porcentaje} % of the test rows: what is measured "
              "depends mostly on that subject",
    },

    # -- 140b-intervalo (Roberto, 01-11, opción A): la cobertura MEDIDA del intervalo de la nota ------------------------
    # Cada frase dice cuánto acierta un intervalo así con ESE número de sujetos de prueba (la tabla medida por el 140b-C0, en
    # `particion_por_diseno_grupos_y_tiempo.COBERTURA_MEDIDA_DEL_INTERVALO`), que vale para esta ventana de tiempo y de dónde
    # sale la medida (datos sintéticos, modelo fijo). `{cobertura}` es un entero «de cada 100», por debajo de lo medido.
    "estricto_cobertura_medida": {
        "es": "con {n} sujetos de prueba, un intervalo así acierta unas {cobertura} de cada 100 veces, no {nominal}: vale "
              "para esta ventana de tiempo (medido con datos sintéticos y un modelo fijo)",
        "en": "with {n} test subjects, an interval like this is right about {cobertura} times in 100, not {nominal}: it "
              "holds for this time window (measured on synthetic data with a fixed model)",
    },
    "estricto_cobertura_del_punto_menor": {
        "es": "con {n} sujetos de prueba, un intervalo así acierta unas {cobertura} de cada 100 veces o algo más, no "
              "{nominal} (se midió con {medido} sujetos de prueba, el punto medido más cercano por debajo): vale para esta "
              "ventana de tiempo (medido con datos sintéticos y un modelo fijo)",
        "en": "with {n} test subjects, an interval like this is right about {cobertura} times in 100 or a little more, not "
              "{nominal} (it was measured with {medido} test subjects, the nearest measured point below): it holds for this "
              "time window (measured on synthetic data with a fixed model)",
    },
    "estricto_cobertura_mas_alla_de_lo_medido": {
        "es": "con {n} sujetos de prueba, un intervalo así acierta unas {cobertura} de cada 100 veces, no {nominal} (lo "
              "medido llega hasta {medido} sujetos de prueba): vale para esta ventana de tiempo (medido con datos "
              "sintéticos y un modelo fijo)",
        "en": "with {n} test subjects, an interval like this is right about {cobertura} times in 100, not {nominal} (what "
              "was measured goes up to {medido} test subjects): it holds for this time window (measured on synthetic data "
              "with a fixed model)",
    },
    "estricto_cobertura_no_medida": {
        "es": "con {n} sujeto(s) de prueba no se ha medido cuánto acierta un intervalo así (se midió desde {minimo} "
              "sujetos): no hay cifra que dar",
        "en": "with {n} test subject(s) it has not been measured how often an interval like this is right (it was measured "
              "from {minimo} subjects): there is no figure to give",
    },
    "estricto_intervalo_aproximado": {
        "es": "intervalo aproximado: con menos de {umbral} sujetos de prueba el remuestreo por sujeto se queda corto de su "
              "nivel del {nominal} %",
        "en": "approximate interval: with fewer than {umbral} test subjects the resampling by subject falls short of its "
              "{nominal}% level",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla — para comprobar que las dos
    redacciones piden LOS MISMOS campos."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas."""
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de motivos del 103-C4: escribir aquí "
            "una frase suelta dejaría media aplicación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(**campos) for idioma in IDIOMAS}
