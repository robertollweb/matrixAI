# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los motivos que este paquete escribe, EN LOS DOS IDIOMAS — 104-C0.

Por qué está aquí y no en quien pinta la pantalla: **lo que redacta el core se
traduce en el core**. Un rechazo de esquema no es un detalle de programación
—«esta partición mete al mismo paciente en desarrollo y en prueba» lo lee una
persona y decide con ello—, así que el core no puede dejar media frase en
inglés para que la UI la remate.

Cómo funciona, y es la parte fina: **no se traduce la frase, se compone el
hecho**. Quien rechaza pasa una CLAVE y sus campos; de ahí salen las dos
redacciones. Traducir la cadena ya compuesta sería adivinar, y guardar en el
estado una cadena compuesta hace que al cambiar de idioma no cambie.

Los dos diccionarios se escriben uno debajo del otro **a propósito**: así salta
a la vista cuando una clave se queda sin pareja. Hay una prueba que compara las
claves Y los huecos `{...}` de cada plantilla, porque un hueco que existe en
castellano y no en inglés solo revienta cuando alguien pide inglés.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "MOTIVOS", "huecos_de", "motivo"]

#: Los idiomas que este paquete sabe escribir. No es una lista abierta: si
#: mañana hay un tercero, se añaden las plantillas, no se traduce al vuelo.
IDIOMAS = ("es", "en")

#: Clave -> {idioma: plantilla}. Los huecos van con `str.format`.
MOTIVOS: dict[str, dict[str, str]] = {
    # ---- primitivas de forma -------------------------------------------
    "no_es_texto": {
        "es": "{campo} tiene que ser un texto no vacío; llegó {valor}",
        "en": "{campo} must be a non-empty string, got {valor}",
    },
    "no_es_entero": {
        "es": "{campo} tiene que ser un entero; llegó {valor} (y `True` no es un "
              "entero por mucho que Python lo cuente como tal)",
        "en": "{campo} must be an integer, got {valor} (and `True` is not an "
              "integer however much Python counts it as one)",
    },
    "no_es_booleano": {
        "es": "{campo} tiene que ser `true` o `false`; llegó {valor}",
        "en": "{campo} must be `true` or `false`, got {valor}",
    },
    "no_es_numero_finito": {
        "es": "{campo} tiene que ser un número finito; llegó {valor}. NaN e "
              "Infinity no existen en JSON y no se pueden canonicalizar",
        "en": "{campo} must be a finite number, got {valor}. NaN and Infinity "
              "are not JSON and cannot be canonicalised",
    },
    "fuera_de_rango": {
        "es": "{campo} tiene que estar entre {minimo} y {maximo}; llegó {valor}",
        "en": "{campo} must be between {minimo} and {maximo}, got {valor}",
    },
    "menor_que_el_minimo": {
        "es": "{campo} tiene que ser >= {minimo}; llegó {valor}",
        "en": "{campo} must be >= {minimo}, got {valor}",
    },
    "no_es_del_vocabulario": {
        "es": "{campo} tiene que ser uno de {opciones}; llegó {valor}. El "
              "vocabulario es cerrado: aceptar uno nuevo aquí sería prometer un "
              "comportamiento que nadie ha implementado",
        "en": "{campo} must be one of {opciones}, got {valor}. The vocabulary is "
              "closed: accepting a new one here would promise behaviour nobody "
              "implemented",
    },
    "no_es_lista_de_textos": {
        "es": "{campo} tiene que ser una lista de textos; llegó {valor}",
        "en": "{campo} must be a list of strings, got {valor}",
    },
    "no_es_mapa": {
        "es": "{campo} tiene que ser un objeto; llegó {valor}",
        "en": "{campo} must be an object, got {valor}",
    },
    "hay_duplicados": {
        "es": "{campo} no puede repetir valores; se repite {valor}",
        "en": "{campo} must not repeat values, {valor} appears twice",
    },
    "falta_campo": {
        "es": "falta {campo} y no se puede rellenar: un valor ausente no es un cero",
        "en": "{campo} is missing and cannot be filled in: an absent value is not a zero",
    },
    "clave_desconocida": {
        "es": "{campo} trae la clave {valor}, que este esquema no conoce. "
              "Interpretarla a medias sería justo lo que la versión de esquema "
              "existe para evitar",
        "en": "{campo} carries the key {valor}, unknown to this schema. Half "
              "interpreting it is exactly what the schema version exists to prevent",
    },
    "esquema_equivocado": {
        "es": "este documento dice ser {valor} y se está leyendo como {campo}",
        "en": "this document declares itself {valor} and is being read as {campo}",
    },
    # ---- ProblemSpec -----------------------------------------------------
    "clases_en_regresion": {
        "es": "una regresión no tiene clases; llegaron {valor}. Si son clases, la "
              "tarea no es regresión",
        "en": "a regression has no classes, got {valor}. If they are classes, the "
              "task is not regression",
    },
    "clasificacion_sin_clases": {
        "es": "una clasificación necesita sus clases ORDENADAS: sin ellas no se "
              "sabe a qué columna de probabilidades corresponde cada una",
        "en": "a classification needs its ORDERED classes: without them there is "
              "no telling which probability column belongs to which class",
    },
    "binaria_con_otro_numero_de_clases": {
        "es": "una clasificación binaria tiene exactamente dos clases; llegaron {valor}",
        "en": "a binary classification has exactly two classes, got {valor}",
    },
    "multiclase_con_menos_de_tres": {
        "es": "una clasificación multiclase tiene tres clases o más; llegaron {valor}",
        "en": "a multiclass classification has three classes or more, got {valor}",
    },
    "clase_positiva_fuera_de_clases": {
        "es": "la clase positiva {valor} no está entre las clases {opciones}",
        "en": "positive class {valor} is not among the classes {opciones}",
    },
    "clase_positiva_en_regresion": {
        "es": "una regresión no tiene clase positiva; llegó {valor}",
        "en": "a regression has no positive class, got {valor}",
    },
    "binaria_sin_clase_positiva": {
        "es": "una binaria sin clase positiva deja sin definir sensibilidad, PPV y "
              "el umbral: no se elige por orden alfabético",
        "en": "a binary problem without a positive class leaves sensitivity, PPV "
              "and the threshold undefined: it is not picked alphabetically",
    },
    "disponibilidad_de_desconocido": {
        "es": "se declara disponibilidad de {valor}, que no está entre los "
              "predictores declarados",
        "en": "availability is declared for {valor}, which is not among the "
              "declared predictors",
    },
    "horizonte_sin_momento": {
        "es": "hay horizonte ({valor}) y no hay momento de predicción: un "
              "horizonte sin ancla no dice desde cuándo se cuenta",
        "en": "there is a horizon ({valor}) and no prediction time: a horizon "
              "without an anchor does not say when it starts counting",
    },
    # ---- SplitPlan -------------------------------------------------------
    "particion_sin_test": {
        "es": "el plan no reserva ninguna observación al rol `test`. El test se "
              "separa AL PRINCIPIO; un plan sin él solo vale si declara "
              "`evaluacion_anidada`",
        "en": "the plan reserves no observation for the `test` role. The test is "
              "held out AT THE START; a plan without one is only valid if it "
              "declares `evaluacion_anidada`",
    },
    "grupos_sin_unidad": {
        "es": "una partición {valor} necesita la unidad de cada observación: sin "
              "ella no se puede comprobar que un grupo no cruce de rol",
        "en": "a {valor} split needs each observation's unit: without it there is "
              "no checking that a group does not cross roles",
    },
    "temporal_sin_tiempo": {
        "es": "una partición {valor} necesita la columna de tiempo: ordenar por el "
              "orden del CSV no es ordenar por fecha",
        "en": "a {valor} split needs the time column: sorting by CSV order is not "
              "sorting by date",
    },
    "unidad_en_dos_roles": {
        "es": "la unidad {valor} aparece en los roles {opciones}: el mismo sujeto "
              "en desarrollo y en prueba invalida la prueba",
        "en": "unit {valor} appears in roles {opciones}: the same subject in "
              "development and in test invalidates the test",
    },
    "unidad_de_observacion_desconocida": {
        "es": "se declara unidad para la observación {valor}, que no está en el plan",
        "en": "a unit is declared for observation {valor}, which is not in the plan",
    },
    "observacion_sin_unidad": {
        "es": "la observación {valor} no tiene unidad declarada y el plan es por "
              "grupos: media declaración no aísla nada",
        "en": "observation {valor} has no declared unit and the plan is grouped: "
              "half a declaration isolates nothing",
    },
    "pliegues_insuficientes": {
        "es": "{campo} tiene que ser 2 o más; llegó {valor}. Un pliegue no es "
              "validación cruzada",
        "en": "{campo} must be 2 or more, got {valor}. One fold is not "
              "cross-validation",
    },
    "plan_vacio": {
        "es": "el plan no asigna ninguna observación a ningún rol",
        "en": "the plan assigns no observation to any role",
    },
    # ---- PredictionRecord ------------------------------------------------
    "prediccion_sin_salida": {
        "es": "este registro no trae etiqueta, ni puntuaciones, ni probabilidades: "
              "no es una predicción",
        "en": "this record carries no label, no scores and no probabilities: it is "
              "not a prediction",
    },
    "columnas_desalineadas": {
        "es": "{campo} trae {valor} valores y hay {opciones} clases: sin "
              "correspondencia no se sabe cuál es de quién",
        "en": "{campo} carries {valor} values for {opciones} classes: with no "
              "correspondence there is no telling which belongs to which",
    },
    "columnas_sin_clases": {
        "es": "{campo} necesita las clases ORDENADAS para saber qué columna es de "
              "cada clase",
        "en": "{campo} needs the ORDERED classes to know which column belongs to "
              "which class",
    },
    "probabilidades_no_suman_uno": {
        "es": "las probabilidades suman {valor} y tienen que sumar 1: media "
              "distribución no se recompone sin saber qué clase falta",
        "en": "probabilities add up to {valor} and must add up to 1: half a "
              "distribution cannot be rebuilt without knowing which class is missing",
    },
    "no_vale_para_recomputar": {
        "es": "no se puede recomputar {campo}: falta {valor}. El esquema lo dice, "
              "no lo rellena",
        "en": "{campo} cannot be recomputed: {valor} missing. The schema says so, "
              "it does not fill it in",
    },
    # ---- MetricSpec ------------------------------------------------------
    "metrica_sin_direccion_ni_ideal": {
        "es": "{campo} no declara dirección ni valor/rango ideal: sin una de las "
              "dos no se puede decir si un número es mejor o peor",
        "en": "{campo} declares neither a direction nor an ideal value/range: "
              "without one of the two there is no telling whether a number is "
              "better or worse",
    },
    "metrica_con_direccion_e_ideal": {
        "es": "{campo} declara dirección Y valor/rango ideal a la vez: la "
              "calibración no es monótona y ordenarla por «mayor mejor» miente",
        "en": "{campo} declares both a direction AND an ideal value/range: "
              "calibration is not monotone and ordering it by «higher is better» lies",
    },
    "rango_ideal_al_reves": {
        "es": "el rango ideal {valor} tiene el mínimo por encima del máximo",
        "en": "ideal range {valor} has its minimum above its maximum",
    },
    "metrica_de_clase_positiva_sin_clase": {
        "es": "{campo} exige clase positiva y la métrica no declara ninguna",
        "en": "{campo} requires a positive class and the metric declares none",
    },
    # ---- pipeline, ajuste, evaluación y selección ------------------------
    "entrenar_con_test": {
        "es": "este pipeline declara haberse ajustado con el rol {valor}: la "
              "prueba reservada no ajusta preparación, familia, hiperparámetros, "
              "calibración ni umbral",
        "en": "this pipeline declares it was fitted on role {valor}: the held-out "
              "test does not tune preparation, family, hyperparameters, "
              "calibration or threshold",
    },
    "umbral_fuera_de_la_escala_calibrada": {
        "es": "el pipeline lleva calibrador y el umbral se declara sobre {valor}: "
              "un umbral en otra escala no es el umbral de este pipeline",
        "en": "the pipeline carries a calibrator and the threshold is declared on "
              "{valor}: a threshold on another scale is not this pipeline's threshold",
    },
    "umbral_sin_clase_positiva": {
        "es": "un umbral binario necesita saber de qué clase es la probabilidad "
              "que corta",
        "en": "a binary threshold needs to know which class the probability it "
              "cuts belongs to",
    },
    "completado_sin_pipeline": {
        "es": "el estado {valor} promete un candidato válido y no viene ninguno. "
              "Un proceso abortado sin candidato no se disfraza de completado",
        "en": "state {valor} promises a valid candidate and none came. A run "
              "aborted with no candidate is not dressed up as completed",
    },
    "sin_motivo_escrito": {
        "es": "el estado {valor} exige motivo escrito: «falló» sin más no le sirve "
              "a nadie",
        "en": "state {valor} requires a written reason: a bare «it failed» helps "
              "nobody",
    },
    "fallido_con_pipeline": {
        "es": "el estado {valor} no puede traer un pipeline como si fuera válido",
        "en": "state {valor} cannot carry a pipeline as if it were valid",
    },
    "motivo_incompleto": {
        "es": "{campo} tiene que traer los dos idiomas {opciones}; llegó {valor}. "
              "Media traducción presentada como completa es peor que ninguna",
        "en": "{campo} must carry both languages {opciones}, got {valor}. Half a "
              "translation presented as complete is worse than none",
    },
    "evidencia_independiente_fuera_del_test": {
        "es": "no se puede etiquetar {valor} una evaluación medida sobre el rol "
              "{opciones}",
        "en": "an evaluation measured on role {opciones} cannot be labelled {valor}",
    },
    "metrica_sin_valor_ni_motivo": {
        "es": "{campo} no trae valor y tampoco dice por qué está indefinida: un "
              "valor ausente no es un cero, pero tiene que explicarse",
        "en": "{campo} carries no value and does not say why it is undefined: an "
              "absent value is not a zero, but it has to be explained",
    },
    "metrica_con_valor_y_motivo": {
        "es": "{campo} trae valor Y motivo de indefinición a la vez",
        "en": "{campo} carries both a value AND an undefined reason",
    },
    "seleccionar_con_el_test": {
        "es": "esta decisión se apoya en una evaluación del rol {valor}: el test "
              "no alimenta la selección",
        "en": "this decision rests on an evaluation of role {valor}: the test does "
              "not feed selection",
    },
    "ganador_sin_candidato": {
        "es": "el resultado {valor} necesita decir QUÉ candidato ganó",
        "en": "result {valor} needs to say WHICH candidate won",
    },
    "sin_ganador_con_candidato": {
        "es": "el resultado {valor} no puede señalar un candidato elegido",
        "en": "result {valor} cannot point at a chosen candidate",
    },
    "ganador_sin_evidencia": {
        "es": "no se elige un candidato sin una sola evaluación que lo respalde",
        "en": "a candidate is not chosen without a single evaluation backing it",
    },
    # ---- protocolo y accesos --------------------------------------------
    "fuga_de_test": {
        "es": "acceso DENEGADO al rol {valor} durante la fase {campo}: la prueba "
              "reservada no participa en aprendizaje ni en selección",
        "en": "access DENIED to role {valor} during phase {campo}: the held-out "
              "test takes no part in learning or selection",
    },
    "proposito_indebido_en_el_test": {
        "es": "acceso DENEGADO al rol {valor} con propósito {campo}: en la prueba "
              "reservada solo se evalúa",
        "en": "access DENIED to role {valor} for purpose {campo}: the held-out "
              "test is only evaluated on",
    },
    "test_sin_congelar": {
        "es": "acceso DENEGADO al rol {valor}: el pipeline no está congelado, así "
              "que lo que se evalúe todavía puede cambiar después de mirarlo",
        "en": "access DENIED to role {valor}: the pipeline is not frozen, so "
              "whatever is evaluated can still change after being looked at",
    },
    "artefacto_distinto_del_congelado": {
        "es": "se evalúa el artefacto {valor} y el congelado es {opciones}",
        "en": "artefact {valor} is being evaluated and the frozen one is {opciones}",
    },
    "fase_desconocida": {
        "es": "no hay ninguna fase abierta: un acceso a los datos sin fase no se "
              "puede juzgar",
        "en": "no phase is open: an access to the data without a phase cannot be judged",
    },
    "anidada_no_implementada": {
        "es": "este plan declara evaluación anidada, que el esquema contempla y "
              "este MVP NO implementa: se declara fuera en vez de ejecutarla a medias",
        "en": "this plan declares nested evaluation, which the schema contemplates "
              "and this MVP does NOT implement: it is declared out of scope instead "
              "of half running it",
    },
    "rol_no_esta_en_el_plan": {
        "es": "el plan no tiene ninguna observación con el rol {valor}",
        "en": "the plan has no observation with role {valor}",
    },
    # ---- migración -------------------------------------------------------
    "version_no_legible": {
        "es": "{campo} viene en la versión {valor} y este core sabe leer {opciones}. "
              "Una versión desconocida se rechaza, no se adivina",
        "en": "{campo} comes in version {valor} and this core can read {opciones}. "
              "An unknown version is refused, not guessed",
    },
    "sin_camino_de_migracion": {
        "es": "{campo} viene en la versión {valor} y no hay camino hasta {opciones}",
        "en": "{campo} comes in version {valor} and there is no path to {opciones}",
    },
    "migracion_ya_registrada": {
        "es": "ya hay una migración de {campo} {valor}",
        "en": "there is already a migration for {campo} {valor}",
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
            f"{clave!r} no está en el catálogo de motivos: escribir aquí una frase "
            "suelta dejaría media aplicación sin traducir")
    formateados = {}
    for idioma in IDIOMAS:
        formateados[idioma] = MOTIVOS[clave][idioma].format(**campos)
    return formateados
