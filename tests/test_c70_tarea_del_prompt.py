"""CONTRATO 70 — la tarea sale de la PREGUNTA, no del verbo.

Encontrado el 2026-08-09 generando 15 modelos inventados por la puerta del
prompt (recorrido que pidió Roberto). Cuatro eran preguntas de sí/no y las
cuatro salieron REGRESIÓN: «predecir si un pedido llegará tarde», «predecir si
un paciente necesitará ingreso», «predecir si lloverá mañana», «detectar una
avería rara».

Una regresión con salida lineal sobre un sí/no devuelve 0.37 o 1.8: ni una
clase ni una probabilidad. Y el sistema se contradecía dentro de la misma
respuesta — el aviso del pipeline decía «typical binary classification task» y
el .mxai traía `units=1 activation=linear` con `OUTPUT: Scalar`.

Dos defectos con la misma raíz, en `_detect_task`:

  D-1  «predecir» está en `_REGRESSION_KEYWORDS` y se mira ANTES que las
       binarias. Es el verbo genérico en español y sirve para las dos tareas;
       el «si» que viene detrás no lo leía nadie.

  D-2  Esas mismas palabras (`temperatura`, `consumo`, `precio`, `valor`) son
       los nombres que la gente le pone a las COLUMNAS DE ENTRADA. «detectar
       una avería a partir de vibración, TEMPERATURA y viento» salía regresión:
       una feature decidía la tarea.

Y no fue un descuido: esa precedencia existe para impedir que un LLM demasiado
entusiasta convierta «predict price» en un clasificador — el sesgo que costó el
contrato 59. La corrección de un sesgo creó el sesgo contrario, y por eso este
fichero comprueba LOS DOS lados.
"""

import pytest

from matrixai.training.dense_generator import DenseNetworkGenerator


@pytest.fixture
def gen():
    return DenseNetworkGenerator()


# ── C2 — «predecir SI …» es una pregunta de sí/no ─────────────────────────
# Los cuatro casos reales del barrido de 15, tal cual se escribieron.

SI_NO = [
    "predecir si un pedido llegara tarde a partir de la distancia en km, el peso en kg y la hora de salida",
    "predecir si un paciente necesitara ingreso a partir de la edad, la saturacion de oxigeno y la frecuencia cardiaca",
    "predecir si llovera mañana a partir de la humedad",
    "detectar una averia rara en una turbina eolica a partir de vibracion, temperatura y velocidad del viento",
    # Y las formas equivalentes, que describen lo mismo.
    "saber si un cliente va a darse de baja",
    "determinar si una transaccion es fraudulenta",
    "predict whether a shipment will arrive late",
    "clasificar si un pedido llegara tarde",
]


@pytest.mark.parametrize("prompt", SI_NO)
def test_una_pregunta_de_si_o_no_es_binaria(gen, prompt):
    assert gen._detect_task(prompt, None) == "binary", (
        "una regresion sobre un si/no devuelve un numero sin unidades: "
        "ni una clase ni una probabilidad"
    )


# ── Invariante 2 — el contrato 59 no se rompe ─────────────────────────────
# «Predecir el precio» tiene que SEGUIR siendo regresión: es exactamente el
# caso que el 59 existe para proteger, y la precedencia de las palabras de
# regresión se puso para eso.

REGRESIONES = [
    "predecir el precio de una vivienda a partir de los metros cuadrados",
    "predecir la temperatura de mañana",
    "estimar el consumo diario de un edificio de oficinas",
    "predecir el coste en euros de una reparacion",
    "estimar los minutos de espera en urgencias",
    "predict the price of a house from its size",
]


@pytest.mark.parametrize("prompt", REGRESIONES)
def test_predecir_una_magnitud_sigue_siendo_regresion(gen, prompt):
    assert gen._detect_task(prompt, None) == "regression", (
        "el contrato 59 existe porque esto se rompio una vez"
    )


# ── C1 — el objetivo, no las columnas ─────────────────────────────────────

def test_una_palabra_de_regresion_ENTRE_LAS_COLUMNAS_no_decide(gen):
    """El caso medido que destapó D-2.

    «detectar» es binaria y «temperatura» es de regresión; la temperatura
    está entre las FEATURES, así que no debe decidir nada.
    """
    conFeature = "detectar una averia a partir de vibracion, temperatura y viento"
    assert gen._detect_task(conFeature, None) == "binary"


def test_la_misma_palabra_SI_decide_cuando_es_lo_que_se_predice(gen):
    """Y el otro lado: si la temperatura es el objetivo, es regresión.

    Sin esta prueba, «no mirar las columnas» podría implementarse ignorando
    la palabra siempre, que arreglaría un caso rompiendo el contrario.
    """
    comoObjetivo = "predecir la temperatura a partir de la humedad y la presion"
    assert gen._detect_task(comoObjetivo, None) == "regression"


@pytest.mark.parametrize("conector", [
    "a partir de", "segun", "en funcion de", "from", "based on", "using",
])
def test_los_conectores_separan_objetivo_de_columnas(gen, conector):
    assert gen._detect_task(f"detectar un fraude {conector} el importe y el precio", None) == "binary"


def test_sin_conector_se_mira_la_frase_entera(gen):
    """Cortar por donde no hay corte sería inventarse una separación."""
    assert gen._detect_task("predecir el precio de una vivienda", None) == "regression"


# ── Invariante 1 — el bracket explícito del 43-C4 sigue ganando ───────────

def test_un_bracket_explicito_gana_sobre_todo(gen):
    conBracket = "predecir si llovera\nSALIDA: lluvia: ProbabilityMap[NO, SI, QUIZA]"
    assert gen._detect_task(conBracket, ["NO", "SI", "QUIZA"]) == "multiclass"


def test_multiclase_declarada_sigue_siendo_multiclase(gen):
    assert gen._detect_task("clasificar el riesgo en bajo, medio o alto", None) == "multiclass"


# ── C3 — la suposición dice su motivo ─────────────────────────────────────

def test_la_suposicion_dice_POR_QUE_no_solo_que(gen):
    """`inferred task=regression` era un resultado sin razón.

    Quien lo lee no sabía si el core entendió su frase o cayó en el valor
    por defecto, y son cosas muy distintas.
    """
    r = gen.generate("predecir si llovera mañana a partir de la humedad")
    tarea = [a for a in r.assumptions if "inferred task" in a]
    assert tarea, "el pipeline tiene que declarar qué tarea eligió"
    assert "binary" in tarea[0]
    assert "SI o NO" in tarea[0], "y por qué la eligió"


def test_cuando_NO_se_ha_entendido_se_dice_que_es_por_defecto(gen):
    """«analizar los datos de mis clientes» no dice qué predecir.

    Antes se construía una regresión y el aviso decía `inferred
    task=regression`, indistinguible de haberlo entendido.
    """
    r = gen.generate("analizar los datos de mis clientes")
    tarea = [a for a in r.assumptions if "inferred task" in a][0]
    assert "POR DEFECTO" in tarea
    assert "clasificar si" in tarea, "y cómo se arregla"


# ── C4 — el barrido de los 15, fijado ─────────────────────────────────────
# No son un barrido que se hizo una vez: son la red que impide que vuelva.

BARRIDO_DE_15 = [
    ("predecir si un pedido llegara tarde a partir de la distancia en km", "binary"),
    ("predecir el coste en euros de una reparacion de coche a partir de los kilometros", "regression"),
    ("clasificar el tipo de incidencia de un ascensor en electrica, mecanica, puertas o software", "multiclass"),
    ("predecir si un alumno aprobara segun las horas de estudio y la asignatura", "binary"),
    ("estimar los minutos de espera en urgencias a partir del numero de pacientes", "regression"),
    ("detectar fraude en un seguro de hogar a partir de importe y valor asegurado", "binary"),
    ("detectar una averia rara en una turbina eolica a partir de vibracion y temperatura", "binary"),
    ("predecir la variacion diaria de temperatura en grados a partir de la humedad", "regression"),
    ("clasificar el riesgo de un credito en bajo, medio o alto", "multiclass"),
    ("predecir si un paciente necesitara ingreso a partir de la edad", "binary"),
    ("predecir si llovera mañana a partir de la humedad", "binary"),
    ("estimar el porcentaje de ocupacion de un hotel a partir del mes y el precio medio", "regression"),
]


@pytest.mark.parametrize("prompt,esperado", BARRIDO_DE_15)
def test_el_barrido_de_los_15(gen, prompt, esperado):
    assert gen._detect_task(prompt, None) == esperado
