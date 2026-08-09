"""Una multiclase con MENOS DE DOS clases no es una multiclase.

Encontrado el 2026-08-09 conduciendo la interfaz por fases, y medido
despues por el API: **1 de cada 6** generaciones de «clasificar si un
cliente cancela su suscripcion a partir del plan, los meses de antiguedad
y las incidencias» devolvia un modelo que el propio verificador del core
RECHAZA:

    Dense Network Generator  aviso  multiclass task requires at least 2 labels — using defaults
    Verifier Agent           error  NETWORK CustomerCancellationModel: softmax output requires units >= 2, got units=1
    Type Check               error  (el mismo)

El LLM proponia `multiclass` con UNA etiqueta y nadie lo corregia: el
`or _default_labels(task)` solo entra con la lista VACIA, y una lista de
un elemento es verdadera. Salia un `softmax` de una salida — que devuelve
siempre 1, y no clasifica nada.

Y el aviso MENTIA: decia «using defaults» sin usar ningun valor por
defecto.
"""

import pytest

from matrixai.training.dense_generator import DenseNetworkGenerator


def _generar(prompt, labels):
    return DenseNetworkGenerator().generate(prompt, labels=labels)


class TestUnSoftmaxDeUnaUnidadNoExiste:
    def test_el_caso_medido_del_core(self):
        """El de verdad: la frase de Roberto, con la etiqueta unica del LLM."""
        r = _generar(
            "clasificar si un cliente cancela su suscripcion a partir del plan, "
            "los meses de antiguedad y las incidencias",
            ["cancela"],
        )
        # Lo innegociable: NUNCA un softmax de menos de dos unidades.
        assert not (r.output_activation == "softmax" and r.output_units < 2)
        # Y la respuesta correcta a un si/no es una binaria.
        assert r.output_activation == "sigmoid"
        assert r.output_units == 1
        assert r.output_type == "Probability"

    def test_el_mxai_que_sale_ya_no_es_el_que_el_verificador_rechaza(self):
        r = _generar("clasificar si un cliente cancela su suscripcion", ["cancela"])
        assert "activation=softmax" not in r.mxai_text
        assert "LAYER Dense units=1 activation=sigmoid" in r.mxai_text

    def test_la_propuesta_del_LLM_no_se_descarta_en_silencio(self):
        """Alguien propuso una clase y se ignoro. Se dice, con el motivo.

        Sin esto queda un modelo binario donde el LLM pidio multiclase, y
        nadie puede saber si la frase estaba mal escrita o si el modelo
        esta mal construido.
        """
        r = _generar("clasificar si un cliente cancela su suscripcion", ["cancela"])
        aviso = " ".join(r.warnings)
        assert "['cancela']" in aviso
        assert "menos de dos" in aviso
        assert "task=binary" in aviso
        # Como pedir varias clases, ya que se esta.
        assert "ProbabilityMap[" in aviso
        # Y ya no miente diciendo que usa valores por defecto sin usarlos.
        assert "using defaults" not in aviso

    def test_sin_pregunta_de_si_o_no_usa_clases_de_ejemplo_Y_LO_DECLARA(self):
        """Aqui no se puede adivinar el conjunto de clases.

        Inventarselas en silencio seria peor que no tenerlas: alguien
        entrenaria contra `class_a/b/c` creyendo que son las suyas.
        """
        r = _generar("clasificar el nivel de riesgo del paciente", ["alto"])
        assert r.output_activation == "softmax"
        assert r.output_units >= 2
        aviso = " ".join(r.warnings)
        assert "NO son tus clases" in aviso

    def test_con_CERO_clases_tambien(self):
        """La lista vacia ya caia a los valores por defecto, pero el
        camino tiene que dar lo mismo: nunca menos de dos unidades."""
        r = _generar("clasificar el nivel de riesgo del paciente", [])
        assert not (r.output_activation == "softmax" and r.output_units < 2)


class TestClasesInventadasEnSilencio:
    """Encontrado en la 3a pasada: `ProbabilityMap[class_a, class_b, class_c]`
    con CERO avisos.

    Nadie habia nombrado esas clases — son marcadores de posicion — y quien
    lo leyera se llevaba un modelo cuyas salidas no significan nada. El
    aviso viejo no lo cubria: solo saltaba con MENOS de dos etiquetas, y
    los valores por defecto son tres. El exportador si lo advierte, pero
    eso llega despues de entrenar.
    """

    def test_las_clases_de_ejemplo_se_declaran_como_tales(self):
        r = _generar("clasificar el nivel de riesgo del paciente a partir de la edad y la tension", None)
        assert r.labels == ["class_a", "class_b", "class_c"]
        aviso = " ".join(r.warnings)
        assert "NO son tus clases" in aviso
        assert "marcadores de posicion" in aviso
        # Y como arreglarlo, en el mismo aviso.
        assert "ProbabilityMap[" in aviso

    def test_si_las_clases_se_leen_del_prompt_NO_se_avisa_de_nada(self):
        r = _generar("clasificar una incidencia en las categorias critica, media o baja", None)
        assert r.labels == ["critica", "media", "baja"]
        assert not any("marcadores de posicion" in w for w in r.warnings)

    def test_el_aviso_dice_QUE_NO_SE_PUDIERON_LEER_no_que_no_esten(self):
        """Matiz que importa: el aviso habla de lo que ESTE extractor pudo
        leer, no de lo que hay escrito. Decir «el prompt no las nombra»
        seria echarle la culpa a quien escribio bien — y el extractor no
        lo lee todo, asi que la afirmacion fuerte seria falsa.

        Este prompt no da ninguna pista de cuantas clases hay ni de como
        se llaman, que es el caso legitimo.
        """
        r = _generar("clasificar el nivel de riesgo del paciente", None)
        aviso = " ".join(r.warnings)
        assert "No se han podido leer las clases" in aviso

    def test_una_binaria_no_lleva_este_aviso(self):
        """`_default_labels("binary")` existe, pero ahi las etiquetas no
        salen al modelo: la salida es un `sigmoid`, no un `ProbabilityMap`.
        Avisar de unas clases que nadie va a ver seria ruido."""
        r = _generar("predecir si un pedido llegara tarde", None)
        assert not any("marcadores de posicion" in w for w in r.warnings)

    def test_una_regresion_tampoco(self):
        r = _generar("predecir el precio de una vivienda", None)
        assert not any("marcadores de posicion" in w for w in r.warnings)


class TestClasesDichasEnProsa:
    """«clasificar la incidencia EN critica, media o baja» — la forma
    normal de escribirlo, que el extractor NO leia.

    Exigia la palabra literal «clases/categorias/niveles/etiquetas» o un
    bracket. Medido en la 3a pasada: ese prompt salia
    `ProbabilityMap[class_a, class_b, class_c]`, o sea, un modelo cuyas
    salidas no significan nada, construido a partir de una frase que SI
    decia lo que significaban.
    """

    @pytest.mark.parametrize("prompt,esperadas", [
        ("clasificar una incidencia en critica, media o baja", ["critica", "media", "baja"]),
        ("clasificar la incidencia como critica, media o baja", ["critica", "media", "baja"]),
        ("classify the ticket into urgent, normal or low", ["urgent", "normal", "low"]),
    ])
    def test_las_lee(self, prompt, esperadas):
        r = _generar(prompt, None)
        assert r.labels == esperadas
        assert r.output_units == len(esperadas)

    @pytest.mark.parametrize("prompt", [
        # La trampa que decidio el diseño: sin exigir separador explicito,
        # el reparto por espacios convertiria esto en tres «clases».
        "clasificar los pedidos en funcion del peso",
        "predecir el precio a partir de la superficie y el barrio",
        "clasificar si un cliente cancela",
        "clasificar el riesgo segun la edad, la tension y el pulso",
        "detectar fraude en un seguro de hogar a partir de importe, antiguedad y codigo postal",
    ])
    def test_NO_se_inventa_clases_donde_no_las_hay(self, prompt):
        g = DenseNetworkGenerator()
        assert g._extract_labels(prompt) == []


class TestLoQueLaRaizNoAlcanza:
    """La red de seguridad NO es codigo muerto: se midio.

    `_detect_task` contesta «binary» cuando ve «clasificar SI …» pegado.
    Pero con el verbo separado —«clasificar PARA SABER si un cliente
    cancela»— cae en la rama de multiclase, y con una sola etiqueta del
    LLM saldria un softmax de tres clases INVENTADAS para una pregunta de
    si o no.

    Ir añadiendo formas a la lista de `_detect_task` seria perseguir el
    idioma sin alcanzarlo, que es lo que ya dijo la 1a auditoria del
    contrato 70. La red de seguridad lo resuelve por detras, leyendo la
    frase con el detector del 70.
    """

    @pytest.mark.parametrize("prompt", [
        "clasificar para saber si un cliente cancela",
        "clasificar el correo para determinar si es spam",
        "categorizar los pedidos y decir si llegaran tarde",
    ])
    def test_verbo_separado_del_si_sigue_siendo_una_pregunta_de_si_o_no(self, prompt):
        r = _generar(prompt, ["una_sola"])
        assert r.output_activation == "sigmoid"
        assert r.output_units == 1
        # Y se dice por que, con la propuesta que se descarto.
        assert any("no hay multiclase" in w for w in r.warnings)


class TestLoQueNoSeToca:
    """Arreglar un sesgo puede crear el contrario: se prueban los dos lados."""

    def test_una_multiclase_de_verdad_sigue_igual(self):
        r = _generar(
            "clasificar una incidencia en critica, media o baja",
            ["critica", "media", "baja"],
        )
        assert r.output_activation == "softmax"
        assert r.output_units == 3
        assert not any("no hay multiclase" in w for w in r.warnings)

    def test_dos_clases_siguen_siendo_softmax_de_dos(self):
        """El contrato GEN C4 lo pide explicitamente: con exactamente dos
        etiquetas declaradas va softmax de 2, nunca el sigmoid de una."""
        r = _generar("clasificar el correo en ProbabilityMap[spam, legitimo]", None)
        assert r.output_activation == "softmax"
        assert r.output_units == 2

    def test_una_regresion_sigue_siendo_una_regresion(self):
        r = _generar("predecir el precio de una vivienda", None)
        assert r.output_activation == "linear"
        assert r.output_type == "Scalar"

    def test_una_binaria_normal_sigue_igual(self):
        r = _generar("predecir si un pedido llegara tarde", None)
        assert r.output_activation == "sigmoid"
        assert r.output_units == 1


@pytest.mark.parametrize("prompt", [
    "clasificar si un cliente cancela",
    "saber si un paciente necesita ingreso",
    "predict whether an order will be late",
    "quiero un modelo que me diga si un cliente va a impagar",
])
def test_ninguna_pregunta_de_si_o_no_acaba_en_softmax_de_una(prompt):
    """El invariante, sobre las formas de preguntar un si/no del 70."""
    r = _generar(prompt, ["una_sola"])
    assert not (r.output_activation == "softmax" and r.output_units < 2)
