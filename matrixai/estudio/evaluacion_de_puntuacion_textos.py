# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que dice la evaluación de una puntuación ya registrada, EN LOS DOS IDIOMAS — 137-C1.

Mismo patrón que `textos.py`, `diagnostico_de_riesgo_textos.py` y compañía, y por la
misma razón: lo que redacta el core se traduce en el core. Catálogo propio y no el
de `textos.py`: éste es el de «¿Funciona aquí?», y meter sus claves en el catálogo
compartido cambiaría un fichero que lee todo el paquete.

**Los motivos que el core YA sabe decir no se copian aquí**: «PPV y NPV no se publican
sin prevalencia declarada», «una puntuación no es una probabilidad», «segmento con
pocas filas», «sin curva de decisión sobre una escala»… los escribe `textos.py` y la
evaluación los pide con `matrixai.estudio.textos.motivo`. Dos sitios contando lo mismo
acaban divergiendo. Aquí solo está lo que es NUEVO de este corte: lo que se rechaza al
leer el CSV y las declaraciones, los avisos y lo que el informe dice que NO es.

Sin adjetivos: ninguna de estas frases dice «bueno», «malo», «aceptable». Dicen qué se
midió, qué falta y qué no se puede afirmar.
"""

from __future__ import annotations

from matrixai.estudio.textos import PorIdioma

__all__ = ["IDIOMAS", "MOTIVOS", "PorIdioma", "cifra", "huecos_de", "motivo"]

IDIOMAS = ("es", "en")

MOTIVOS: dict[str, dict[str, str]] = {
    # ---- rechazos: el CSV --------------------------------------------------
    "csv_vacio": {
        "es": "el CSV está vacío: no hay cabecera ni filas que evaluar",
        "en": "the CSV is empty: there is no header and no rows to evaluate",
    },
    "csv_ni_utf8_ni_windows1252": {
        "es": "el fichero no es UTF-8 válido ni Windows-1252 legible (byte problemático "
              "en la posición {valor}): no se adivina otra codificación",
        "en": "the file is neither valid UTF-8 nor readable Windows-1252 (problem byte "
              "at position {valor}): no other encoding is guessed",
    },
    "csv_ilegible": {
        "es": "el CSV es ilegible: {valor}",
        "en": "the CSV is unreadable: {valor}",
    },
    "csv_sin_filas": {
        "es": "el CSV tiene cabecera pero ninguna fila con el número de columnas de la "
              "cabecera ({valor} columnas): no hay nada que evaluar",
        "en": "the CSV has a header but no row with the header's number of columns "
              "({valor} columns): there is nothing to evaluate",
    },
    "csv_cabecera_invalida": {
        "es": "la cabecera del CSV tiene un nombre de columna vacío o repetido ({valor}): "
              "con eso no se sabe a qué columna se refiere cada declaración",
        "en": "the CSV header has an empty or repeated column name ({valor}): that way "
              "nobody can tell which column each declaration refers to",
    },
    "columna_inexistente": {
        "es": "{campo}: la columna {valor} no está en el CSV. Columnas que hay: {opciones}",
        "en": "{campo}: column {valor} is not in the CSV. Columns present: {opciones}",
    },
    "columna_repetida_en_el_encargo": {
        "es": "la columna {valor} se ha declarado para dos cosas distintas ({campo}): una "
              "columna no puede ser a la vez dos de esos papeles",
        "en": "column {valor} was declared for two different roles ({campo}): one column "
              "cannot play both",
    },
    # ---- rechazos: el desenlace -------------------------------------------
    "desenlace_con_mas_de_dos_valores": {
        "es": "el desenlace {campo} tiene {valor} valores distintos tras unir mayúsculas y "
              "tildes ({opciones}...): este contrato evalúa un desenlace binario",
        "en": "outcome {campo} has {valor} distinct values after merging case and accents "
              "({opciones}...): this evaluation is for a binary outcome",
    },
    "desenlace_sin_el_evento": {
        "es": "el valor del evento {valor} no aparece en la columna {campo}. Valores que hay: "
              "{opciones}. El evento lo declara quien evalúa; nunca se adivina",
        "en": "the event value {valor} does not appear in column {campo}. Values present: "
              "{opciones}. The event is declared by whoever evaluates; it is never guessed",
    },
    "desenlace_sin_filas": {
        "es": "ninguna fila de la columna {campo} trae un desenlace: no hay nada contra lo "
              "que medir",
        "en": "no row in column {campo} carries an outcome: there is nothing to measure against",
    },
    # ---- rechazos: las puntuaciones ---------------------------------------
    "puntuacion_no_numerica": {
        "es": "{campo} tiene {valor} valores que no son números (por ejemplo {opciones}). No "
              "se admiten separadores de miles ni símbolos como %: revisa la columna",
        "en": "{campo} has {valor} values that are not numbers (for example {opciones}). "
              "Thousands separators and symbols such as % are not accepted: check the column",
    },
    "puntuacion_decimal_mezclado": {
        "es": "{campo} mezcla coma decimal ({valor} valores) y punto decimal ({opciones} "
              "valores): no se adivina cuál vale. Unifica la columna",
        "en": "{campo} mixes decimal comma ({valor} values) and decimal point ({opciones} "
              "values): which one applies is not guessed. Make the column uniform",
    },
    "probabilidad_fuera_de_rango": {
        "es": "{campo} se ha declarado como {minimo}–{maximo} y tiene {valor} valores fuera de "
              "ese rango (por ejemplo {opciones}). No se recortan ni se corrigen: revisa si "
              "la columna es de otro tipo",
        "en": "{campo} was declared as {minimo}–{maximo} and has {valor} values outside that "
              "range (for example {opciones}). They are neither clipped nor fixed: check "
              "whether the column is of another type",
    },
    "ninguna_fila_utilizable": {
        "es": "{campo}: ninguna fila tiene a la vez desenlace y puntuación, así que no hay "
              "nada que medir",
        "en": "{campo}: no row has both an outcome and a score, so there is nothing to measure",
    },
    "subgrupo_con_demasiados_valores": {
        "es": "la columna de subgrupo {campo} tiene {valor} valores distintos (el máximo es "
              "{opciones}): parece un identificador, y un informe con un grupo por persona "
              "publicaría una cifra por persona",
        "en": "subgroup column {campo} has {valor} distinct values (the maximum is "
              "{opciones}): it looks like an identifier, and a report with one group per "
              "person would publish one figure per person",
    },
    # ---- rechazos: las declaraciones --------------------------------------
    "declaracion_fuera_del_vocabulario": {
        "es": "{campo} tiene que ser uno de {opciones}; llegó {valor}",
        "en": "{campo} must be one of {opciones}, got {valor}",
    },
    "declaracion_fuera_de_rango": {
        "es": "{campo} tiene que estar entre {minimo} y {maximo}; llegó {valor}",
        "en": "{campo} must be between {minimo} and {maximo}, got {valor}",
    },
    "declaracion_no_numerica": {
        "es": "{campo} tiene que ser un número; llegó {valor}",
        "en": "{campo} must be a number, got {valor}",
    },
    "declaracion_no_booleana": {
        "es": "{campo} tiene que decirse explícitamente (sí o no); llegó {valor}. No tiene "
              "valor por omisión: de ello depende si se publican VPP y VPN",
        "en": "{campo} must be stated explicitly (yes or no), got {valor}. There is no "
              "default: whether PPV and NPV are published depends on it",
    },
    "numero_de_puntuaciones": {
        "es": "se evalúan una o dos puntuaciones; llegaron {valor}",
        "en": "one or two scores are evaluated; got {valor}",
    },
    "umbrales_de_decision_invalidos": {
        "es": "los umbrales de la curva de decisión son probabilidades entre 0 y 1 (también "
              "para una puntuación en %), sin repetir y sin llegar a 1; llegó {valor}",
        "en": "decision-curve thresholds are probabilities between 0 and 1 (also for a score "
              "in %), without repeats and never reaching 1; got {valor}",
    },
    "recibo_sin_fecha": {
        "es": "para firmar el recibo hace falta la fecha de emisión (creado_en): esta función "
              "no lee el reloj, y un recibo firmado sin fecha no pasa el esquema",
        "en": "signing the receipt needs an issue date (creado_en): this function does not "
              "read the clock, and a signed receipt without a date fails the schema",
    },
    # ---- avisos del informe ------------------------------------------------
    "direccion_posiblemente_mal_declarada": {
        "es": "{campo}: el AUROC sale {valor}, por debajo de 0,5. La dirección declarada "
              "puede estar al revés (¿más alto = más riesgo?). El número se publica tal y "
              "como sale: la dirección no se corrige en silencio",
        "en": "{campo}: the AUROC is {valor}, below 0.5. The declared direction may be the "
              "wrong way round (is higher = more risk?). The number is published as it "
              "comes out: the direction is not corrected silently",
    },
    "falta_concentrada_en_un_desenlace": {
        "es": "{campo}: la puntuación falta en el {valor} de las filas con evento y en el "
              "{opciones} de las filas sin evento (diferencia no atribuible al azar, "
              "{minimo}). Esas filas quedan fuera, y eso puede sesgar todas las cifras "
              "del informe en una dirección que desde aquí no se puede saber",
        "en": "{campo}: the score is missing in {valor} of the rows with the event and in "
              "{opciones} of the rows without it (a difference not attributable to chance, "
              "{minimo}). Those rows are left out, and that may bias every figure in "
              "the report in a direction that cannot be known from here",
    },
    "filas_malformadas_fuera": {
        "es": "{valor} filas tienen un número de columnas distinto al de la cabecera y "
              "quedan fuera",
        "en": "{valor} rows have a different number of columns than the header and are "
              "left out",
    },
    "sin_umbral_de_uso": {
        "es": "{campo}: no se declaró umbral de uso, así que no hay tabla al umbral (no se "
              "inventa uno)",
        "en": "{campo}: no usage threshold was declared, so there is no threshold table "
              "(none is invented)",
    },
    "probabilidad_con_direccion_inversa": {
        "es": "{campo} se declaró como probabilidad con «más bajo = más riesgo»: no es la "
              "probabilidad del evento (sería la de otra cosa), así que no se mide su "
              "calibración contra el evento ni se invierte por ti. Solo se mide cómo ordena",
        "en": "{campo} was declared as a probability with «lower = more risk»: it is not "
              "the probability of the event (it would be the probability of something "
              "else), so its calibration against the event is not measured and it is not "
              "inverted for you. Only how it ranks is measured",
    },
    "umbrales_de_decision_no_declarados": {
        "es": "no se declaró el rango de umbrales de la curva de decisión: no se inventa un "
              "barrido de 0 a 1 en nombre de quien decide",
        "en": "no range of decision-curve thresholds was declared: no sweep from 0 to 1 is "
              "invented on behalf of whoever decides",
    },
    "cifra_prometida_sin_intervalo": {
        "es": "{campo}: no hay intervalo con el que comparar la cifra prometida",
        "en": "{campo}: there is no interval to compare the promised figure with",
    },
    "fiabilidad_pocas_filas": {
        "es": "tramo con {valor} filas: con menos de {opciones} no se publica su cifra "
              "(evidencia insuficiente; suprimir tramos pequeños ayuda, pero no garantiza "
              "anonimización)",
        "en": "bin with {valor} rows: with fewer than {opciones} its figure is not "
              "published (insufficient evidence; suppressing small bins helps, but does not "
              "guarantee anonymisation)",
    },
    "multiplicidad_en_subgrupos": {
        "es": "se miran {valor} grupos y no se corrige por multiplicidad: con tantas "
              "comparaciones, alguna diferencia puede ser azar",
        "en": "{valor} groups are examined and no multiplicity correction is applied: with "
              "that many comparisons, some difference may be chance",
    },
    "comparacion_sin_filas_en_comun": {
        "es": "las dos puntuaciones no tienen ninguna fila en común con desenlace: no hay "
              "comparación emparejada posible",
        "en": "the two scores share no row with an outcome: no paired comparison is possible",
    },
    "vpp_vpn_de_la_muestra": {
        "es": "VPP y VPN son los de esta muestra, con su prevalencia ({valor}); valen para la "
              "población de uso porque quien ejecuta la evaluación declaró que la muestra la "
              "representa. Su intervalo recoge la incertidumbre de esa prevalencia",
        "en": "PPV and NPV are this sample's, at its prevalence ({valor}); they hold for the "
              "population of use because whoever runs the evaluation declared that the sample "
              "represents it. Their interval includes the uncertainty of that prevalence",
    },
    # ---- la frase-veredicto de cada puntuación (la compone el núcleo con los números) ----
    "veredicto_ordena_si": {
        "es": "Ordena mejor que el azar (AUROC {valor})",
        "en": "Ranks better than chance (AUROC {valor})",
    },
    "veredicto_ordena_azar": {
        "es": "No se distingue del azar al ordenar (AUROC {valor}; el intervalo incluye 0,5)",
        "en": "Cannot be told apart from chance at ranking (AUROC {valor}; the interval "
              "includes 0.5)",
    },
    "veredicto_ordena_al_reves": {
        "es": "Ordena peor que el azar (AUROC {valor}): ¿está bien declarada la dirección?",
        "en": "Ranks worse than chance (AUROC {valor}): is the direction declared correctly?",
    },
    "veredicto_ordena_sin_intervalo": {
        "es": "AUROC {valor} (sin intervalo: no se puede decir si supera al azar)",
        "en": "AUROC {valor} (no interval: it cannot be said whether it beats chance)",
    },
    "veredicto_ordena_no_definido": {
        "es": "No se pudo calcular el AUROC",
        "en": "The AUROC could not be calculated",
    },
    "veredicto_riesgo_sobrestima": {
        "es": "sobrestima el riesgo en {valor} puntos porcentuales",
        "en": "overestimates the risk by {valor} percentage points",
    },
    "veredicto_riesgo_infraestima": {
        "es": "infraestima el riesgo en {valor} puntos porcentuales",
        "en": "underestimates the risk by {valor} percentage points",
    },
    "veredicto_riesgo_sin_diferencia": {
        "es": "no hay diferencia clara entre el riesgo que dice y el observado "
              "(diferencia de {valor} puntos porcentuales; el intervalo incluye 0)",
        "en": "there is no clear difference between the risk it states and the observed one "
              "(difference of {valor} percentage points; the interval includes 0)",
    },
    "veredicto_riesgo_sin_intervalo": {
        "es": "diferencia de riesgo medio de {valor} puntos porcentuales (sin intervalo: no se "
              "puede decir si es distinta de 0)",
        "en": "mean-risk difference of {valor} percentage points (no interval: it cannot be "
              "said whether it differs from 0)",
    },
    "veredicto_riesgo_no_aplica": {
        "es": "no se mide si acierta en el riesgo (no es una probabilidad del evento)",
        "en": "whether the risk is right is not measured (it is not a probability of the event)",
    },
    "veredicto_riesgo_no_definido": {
        "es": "no se pudo calcular la diferencia entre el riesgo que dice y el observado",
        "en": "the difference between the risk it states and the observed one could not be "
              "calculated",
    },
    # ---- R2: qué se predice, en qué plazo, y si estos casos participaron en el desarrollo ----
    "texto_obligatorio_vacio": {
        "es": "{campo} hay que declararlo con un texto (por ejemplo «{opciones}»); llegó {valor}. "
              "No tiene valor por omisión: sin él no se sabe qué se predice ni cuándo",
        "en": "{campo} must be declared as text (for example “{opciones}”); got {valor}. "
              "There is no default: without it nobody knows what is predicted or when",
    },
    "validacion_externa_si": {
        "es": "Los casos de esta muestra participaron en el desarrollo de «{campo}»: esto NO es "
              "una validación externa de esa puntuación.",
        "en": "The cases in this sample took part in the development of “{campo}”: this is NOT "
              "an external validation of that score.",
    },
    "validacion_externa_no_se_sabe": {
        "es": "No se sabe si los casos de esta muestra participaron en el desarrollo de "
              "«{campo}»: mientras no se sepa, esto NO es una validación externa de esa "
              "puntuación.",
        "en": "It is not known whether the cases in this sample took part in the development of "
              "“{campo}”: until that is known, this is NOT an external validation of that score.",
    },
    "validacion_externa_no": {
        "es": "Quien ejecuta la evaluación declaró que los casos de esta muestra NO participaron "
              "en el desarrollo de «{campo}» (MatrixAI no lo comprueba). Aun así, esto no es una "
              "validación para uso clínico.",
        "en": "Whoever runs the evaluation declared that the cases in this sample did NOT take "
              "part in the development of “{campo}” (MatrixAI does not check it). Even so, this "
              "is not a validation for clinical use.",
    },
    "no_es_validacion_externa": {
        "es": "No es una validación externa de: {campo}. Los casos participaron en su "
              "desarrollo, o no se sabe si lo hicieron.",
        "en": "It is not an external validation of: {campo}. The cases took part in its "
              "development, or it is not known whether they did.",
    },
    # ---- dónde se calculó (defecto del 24-10: «todo en local» no es verdad en la demo) ----
    "calculado_en_local": {
        "es": "Se calculó en local, en el equipo de quien ejecutó la evaluación: el CSV no salió "
              "de él.",
        "en": "It was calculated locally, on the machine of whoever ran the evaluation: the CSV "
              "did not leave it.",
    },
    "calculado_en_el_servidor_de_la_demo": {
        "es": "Se calculó en el servidor de la demo con el CSV que enviaste; el CSV no se "
              "guardó.",
        "en": "It was calculated on the demo server with the CSV you sent; the CSV was not "
              "stored.",
    },
    # ---- R3 (revisión del 24-10): la representatividad afecta a más que VPP y VPN ----
    "rotulo_muestra_aportada": {
        "es": "describen la muestra aportada, no la población de uso",
        "en": "describe the supplied sample, not the population of use",
    },
    "porque_muestra_aportada": {
        "es": "Un muestreo con más eventos de los habituales (por ejemplo, todos los casos y "
              "solo parte de los controles) distorsiona la calibración, el beneficio neto y "
              "la curva de decisión. No se aplica ningún ajuste por muestreo.",
        "en": "Sampling with more events than usual (for example, every case and only some of "
              "the controls) distorts calibration, net benefit and the decision curve. No "
              "sampling adjustment is applied.",
    },
    "representatividad_no_comprobada_si": {
        "es": "Que la muestra representa a la población de uso lo declara quien ejecuta la "
              "evaluación, y MatrixAI no lo comprueba. Si no fuera así, la calibración, el "
              "beneficio neto, la curva de decisión, el VPP y el VPN describirían solo la "
              "muestra aportada.",
        "en": "That the sample represents the population of use is declared by whoever runs "
              "the evaluation, and MatrixAI does not check it. If it did not, calibration, "
              "net benefit, the decision curve, PPV and NPV would describe only the supplied "
              "sample.",
    },
    "representatividad_no_comprobada_no": {
        "es": "Quien ejecuta la evaluación declaró que la muestra NO representa a la población "
              "de uso (MatrixAI no lo comprueba): el VPP y el VPN no se publican, y la "
              "calibración, el beneficio neto y la curva de decisión describen la muestra "
              "aportada.",
        "en": "Whoever runs the evaluation declared that the sample does NOT represent the "
              "population of use (MatrixAI does not check it): PPV and NPV are not published, "
              "and calibration, net benefit and the decision curve describe the supplied "
              "sample.",
    },
    # ---- R4: la cifra prometida es una referencia publicada frente a una estimación local ----
    "referencia_publicada_advertencia": {
        "es": "La referencia publicada y la estimación local pueden no ser comparables: la "
              "población, el desenlace, el seguimiento y la incertidumbre de la cifra original "
              "pueden ser distintos. Que la referencia caiga fuera del intervalo local no "
              "demuestra que la cifra original esté mal. La comparación más directamente "
              "interpretable es la emparejada de dos puntuaciones sobre los mismos casos.",
        "en": "The published reference and the local estimate may not be comparable: the "
              "population, the outcome, the follow-up and the uncertainty of the original "
              "figure may differ. A reference falling outside the local interval does not "
              "show that the original figure is wrong. The most directly interpretable "
              "comparison is the paired one, between two scores over the same cases.",
    },
    # ---- lo que el informe dice que NO es (reglas 17, 18 y 19) --------------
    "no_es_validacion_clinica": {
        "es": "Esto no es una validación para uso clínico ni una evaluación regulatoria. Es "
              "una evaluación retrospectiva con los datos aportados.",
        "en": "This is not a validation for clinical use nor a regulatory assessment. It is "
              "a retrospective evaluation of the data provided.",
    },
    "paradoja_del_tratamiento": {
        "es": "Si la puntuación ya se usaba para decidir (un riesgo alto llevó a tratar y el "
              "tratamiento evitó el desenlace), la puntuación parecerá peor de lo que es.",
        "en": "If the score was already used to decide (a high risk led to treatment and the "
              "treatment prevented the outcome), the score will look worse than it is.",
    },
    "no_es_anonimizacion": {
        "es": "La supresión de grupos pequeños ayuda, pero no garantiza anonimización: un "
              "informe sin filas puede seguir permitiendo identificar a alguien combinado con "
              "otra información. Quien lo comparta decide si puede hacerlo.",
        "en": "Suppressing small groups helps, but does not guarantee anonymisation: a report "
              "without rows can still allow someone to be identified when combined with other "
              "information. Whoever shares it decides whether it can be shared.",
    },
    "depende_del_registro": {
        "es": "Depende del registro: un desenlace mal o tarde registrado mueve todas las cifras.",
        "en": "It depends on the record: an outcome registered wrongly or late moves every figure.",
    },
    # ---- la exploración del umbral (137-C4) -----------------------------------
    "exploracion_no_valida_un_umbral": {
        "es": "Esta exploración no valida un nuevo umbral: enseña qué pasaría con estos mismos "
              "casos en cada umbral, y elegir uno mirando los datos que lo miden hace que "
              "parezca funcionar mejor de lo que lo haría con casos nuevos.",
        "en": "This exploration does not validate a new threshold: it shows what would happen "
              "with these same cases at each threshold, and choosing one by looking at the "
              "data that measure it makes it look better than it would on new cases.",
    },
    "exploracion_regla_mayor_o_igual": {
        "es": "Un caso se señala cuando su puntuación alcanza el umbral (mayor o igual).",
        "en": "A case is flagged when its score reaches the threshold (greater than or equal).",
    },
    "exploracion_regla_menor_o_igual": {
        "es": "Un caso se señala cuando su puntuación alcanza el umbral por abajo (menor o "
              "igual).",
        "en": "A case is flagged when its score reaches the threshold from below (less than "
              "or equal).",
    },
    "exploracion_direccion_invertida": {
        "es": "«Más bajo = más riesgo»: el umbral se aplica al revés. Con un umbral más alto "
              "se señalan MÁS casos, no menos.",
        "en": "\"Lower = higher risk\": the threshold is applied the other way round. A higher "
              "threshold flags MORE cases, not fewer.",
    },
    "exploracion_umbrales_ofrecidos": {
        "es": "Solo se ofrecen estos umbrales (vértices de la curva, como mucho {valor} de "
              "{opciones}): el deslizador salta de uno a otro y no interpola entre ellos.",
        "en": "Only these thresholds are offered (vertices of the curve, at most {valor} of "
              "{opciones}): the slider jumps from one to the next and does not interpolate.",
    },
    "exploracion_todos_los_umbrales": {
        "es": "Se ofrecen los {valor} umbrales distintos de la muestra (vértices de la curva): "
              "el deslizador salta de uno a otro y no interpola entre ellos.",
        "en": "All {valor} distinct thresholds in the sample are offered (vertices of the "
              "curve): the slider jumps from one to the next and does not interpolate.",
    },
    "exploracion_vpp_vpn_con_la_prevalencia_de_la_muestra": {
        "es": "VPP y VPN se calculan con la prevalencia de la muestra, que se ha declarado "
              "como la de la población de uso (lo declara quien ejecuta; MatrixAI no lo "
              "comprueba).",
        "en": "PPV and NPV are computed with the sample prevalence, which has been declared "
              "to be that of the population of use (declared by whoever runs it; MatrixAI "
              "does not check it).",
    },
    "exploracion_vpp_vpn_no_publicables": {
        "es": "VPP y VPN no se enseñan: se declaró que la muestra NO representa a la "
              "población de uso, y dependen de la prevalencia.",
        "en": "PPV and NPV are not shown: the sample was declared NOT to represent the "
              "population of use, and they depend on prevalence.",
    },
    "exploracion_sin_los_dos_desenlaces": {
        "es": "No hay exploración del umbral: en las filas usadas falta alguno de los dos "
              "desenlaces (con evento y sin evento) y no hay curva que recorrer.",
        "en": "There is no threshold exploration: the rows used lack one of the two outcomes "
              "(with and without the event), so there is no curve to walk along.",
    },
    "exploracion_umbral_de_uso_no_alcanza": {
        "es": "Con el umbral de uso declarado ({valor}) no se señala a ningún caso de esta "
              "muestra: ninguna puntuación lo alcanza.",
        "en": "With the declared threshold of use ({valor}) no case in this sample is "
              "flagged: no score reaches it.",
    },
    "exploracion_recuentos_pequenos": {
        "es": "En {valor} de los {opciones} umbrales hay un recuento de 1 o 2 casos. Son "
              "agregados, no filas, y no se ocultan; pero con umbrales extremos un recuento "
              "así puede describir a una persona concreta, y no garantiza anonimización.",
        "en": "At {valor} of the {opciones} thresholds a count is 1 or 2 cases. They are "
              "aggregates, not rows, and are not hidden; but at extreme thresholds a count "
              "like that can describe one particular person, and does not guarantee "
              "anonymisation.",
    },
    # ---- el recibo ---------------------------------------------------------
    "recibo_proposito": {
        "es": "evaluar, sin entrenar nada, una puntuación de riesgo ya registrada frente a un "
              "desenlace observado",
        "en": "evaluate, without training anything, a risk score already on record against an "
              "observed outcome",
    },
    "recibo_atestigua": {
        "es": "la evaluación, el entorno y las entradas: este fichero, con estas "
              "declaraciones, dio estas cifras, aquí y ahora",
        "en": "the evaluation, the environment and the inputs: this file, with these "
              "declarations, gave these figures, here and now",
    },
    "recibo_no_atestigua": {
        "es": "que la puntuación sea buena, que el desenlace esté bien registrado, de dónde "
              "salió la puntuación ni que valga para otra población; la garantía no pasa de A1",
        "en": "that the score is good, that the outcome was recorded correctly, where the "
              "score came from, nor that it holds for another population; the assurance "
              "does not exceed A1",
    },
    "recibo_actor_no_consta": {
        "es": "no consta quién lo pidió",
        "en": "who requested it is not recorded",
    },
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla — para comprobar que las dos redacciones piden
    LOS MISMOS campos: un hueco que existe en castellano y no en inglés solo revienta
    cuando alguien pide inglés."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def motivo(clave: str, **campos: object) -> dict[str, str]:
    """El motivo, compuesto en los dos idiomas — nunca una cadena ya compuesta en uno
    solo, que al cambiar de idioma no cambiaría."""
    if clave not in MOTIVOS:
        raise KeyError(
            f"{clave!r} no está en el catálogo de textos del 137: escribir aquí una frase "
            "suelta dejaría media evaluación sin traducir")
    return {idioma: MOTIVOS[clave][idioma].format(
        **{k: (v[idioma] if isinstance(v, PorIdioma) else v) for k, v in campos.items()})
        for idioma in IDIOMAS}


def valor_p(p: float) -> PorIdioma:
    """Un valor p como se escribe: «p = 0,0056», y «p < 0,0001» cuando es más pequeño que lo que cuatro decimales
    pueden decir. Antes salía «p = 0,0000» (visto en Chromium el 25-10), que afirma un cero que el test no da."""
    if p < 0.0001:
        return PorIdioma({"es": "p < 0,0001", "en": "p < 0.0001"})
    texto = f"{p:.4f}"
    return PorIdioma({"es": "p = " + texto.replace(".", ","), "en": "p = " + texto})


def cifra(valor: float, decimales: int = 3, *, porcentaje: bool = False) -> PorIdioma:
    """Un número escrito como lo escribe cada idioma: `0,838` en castellano y `0.838` en
    inglés. Un hueco que solo sirve en uno de los dos es justo lo que `PorIdioma` evita."""
    texto = f"{valor * 100:.1f}" if porcentaje else f"{valor:.{decimales}f}"
    sufijo = " %" if porcentaje else ""
    return PorIdioma({"es": texto.replace(".", ",") + sufijo,
                      "en": texto + (" %" if porcentaje else "")})
