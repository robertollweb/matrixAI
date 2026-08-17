# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Contrato 80-C1 — la receta de datos escrita por una persona.

QUÉ PASABA, medido antes de escribir una línea (ver `77.1_sugerencias.md`):
los datos que generaba el Studio **no tenían relación con lo que se pedía
predecir**. Con 400 filas, la correlación máxima entre cualquier entrada y
el objetivo era 0,049 en binario y 0,054 en regresión — indistinguible del
azar, cuyo umbral al 95 % es 0,098. La única excepción era multiclase CON
clave de LLM, porque las reglas de dominio exigían las cuatro condiciones
a la vez (`playground.py`: modo coherente + `use_llm` + LLM activo +
multiclase con etiquetas declaradas).

Consecuencia real: quien instalaba el Studio sin clave, generaba datos y
entrenaba, veía un 50 % de exactitud y un aviso de colapso. El producto era
honesto; quien miraba concluía que no funciona.

Lo que este corte añade —y lo que estas pruebas vigilan— es que la regla la
pueda **escribir una persona**, que valga también para el **binario**, y
que declare su **ruido**: sin ruido el dataset es perfectamente separable y
el modelo entrena al 100 %, que es un número que no se sostiene delante de
nadie técnico.
"""
from __future__ import annotations

import pytest

from matrixai.playground import _generate_synthetic_dataset
from matrixai.training.domain_rules import parse_domain_rules

MXAI_BINARIO = """PROJECT Impago

VECTOR Cliente[3]
  edad: Scalar
  ingresos: Scalar
  deuda: Scalar
END

NETWORK N
  INPUT Cliente
  LAYER Dense units=8 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT impaga: Probability
END

GRAPH
  Cliente -> N
END
"""

TRAIN_BINARIO = """MODEL model.mxai

DATASET D
  SOURCE csv("d.csv")
  INPUT Cliente FROM COLUMNS [edad, ingresos, deuda]
  TARGET impaga: Probability
END

LOSS L
  TYPE binary_cross_entropy
  PREDICTION impaga
  TARGET impaga
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE N.*
END

RUN
  EPOCHS 20
END
"""

RECETA = "1: deuda > 0.6 OR ingresos < 0.3\nDEFAULT: 0"


def _genera(receta: str | None, filas: int = 400, semilla: int = 7):
    return _generate_synthetic_dataset(
        MXAI_BINARIO, TRAIN_BINARIO, rows=filas, seed=semilla,
        mode="coherent", use_llm=False, recipe_text=receta,
    )


def _columnas(resultado) -> tuple[list[str], dict[str, list[float]]]:
    filas = [l.strip().split(",") for l in resultado["csv_text"].strip().split("\n")]
    cabecera, datos = filas[0], [[float(x) for x in f] for f in filas[1:]]
    return cabecera, {c: [f[i] for f in datos] for i, c in enumerate(cabecera)}


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return 0.0 if da == 0 or db == 0 else num / (da * db)


class TestElIdiomaDeLaReceta:
    """Lo que una persona puede escribir, y que se le devuelva igual."""

    def test_ruido_comentarios_y_etiquetas_numericas(self):
        r = parse_domain_rules(
            "# a más deuda, más riesgo\n"
            "1: deuda > 0.6 OR ingresos < 0.3\n"
            "DEFAULT: 0\n"
            "RUIDO: 0.1"
        )
        # Etiquetas NUMÉRICAS: un objetivo binario se llama «1» y «0», que no
        # son identificadores. Antes el parser las descartaba en silencio.
        assert [x.label for x in r.rules] == ["1"]
        assert r.default_label == "0"
        assert r.noise == pytest.approx(0.1)

    def test_el_ruido_se_escribe_en_el_texto_auditable(self):
        """Si el ruido no sale en `to_text()`, la receta publicada no
        reproduce el dataset — y la reproducibilidad es el contrato.

        Se ESCRIBE en inglés (`NOISE`) aunque se acepten las dos: el resto
        del idioma ya era inglés (`DEFAULT`, `AND`, `OR`), y una receta
        mitad en cada idioma es el defecto que este proyecto ya ha pagado
        varias veces."""
        assert "NOISE: 0.25" in parse_domain_rules(RECETA + "\nRUIDO: 0.25").to_text()
        assert "NOISE: 0.25" in parse_domain_rules(RECETA + "\nNOISE: 0.25").to_text()

    def test_un_ruido_imposible_se_ignora_en_vez_de_reventar(self):
        """Este parser lo alimenta una persona: es tolerante a propósito."""
        assert parse_domain_rules(RECETA + "\nRUIDO: 7").noise == 0.0
        assert parse_domain_rules(RECETA + "\nRUIDO: pepe").noise == 0.0


class TestLosDatosTienenSenal:
    """El corazón del contrato, medido como se midió el defecto."""

    def test_sin_receta_el_objetivo_sigue_siendo_ruido(self):
        """La línea base. Si esto empieza a fallar es que algo da señal
        por accidente, y habría que saberlo."""
        cab, cols = _columnas(_genera(None))
        objetivo = cols[cab[-1]]
        maxima = max(abs(_pearson(cols[c], objetivo)) for c in cab[:-1])
        assert maxima < 0.098, f"sin receta no debería haber señal, y hay {maxima}"

    def test_con_receta_y_SIN_LLM_hay_senal_de_verdad(self):
        cab, cols = _columnas(_genera(RECETA))
        objetivo = cols[cab[-1]]
        maxima = max(abs(_pearson(cols[c], objetivo)) for c in cab[:-1])
        # El criterio de cierre del contrato: > 0,3. Medido: 0,497.
        assert maxima > 0.3, f"la receta no está dando señal: {maxima}"

    def test_el_binario_ya_no_esta_excluido(self):
        """Era la exclusión que rompía el caso más común: `Probability`
        estaba fuera por construcción."""
        r = _genera(RECETA)
        assert r["ok"] is True
        assert r["label_origin"] == "synthetic_domain"
        _, cols = _columnas(r)
        # Y guarda 0/1, no la palabra de la regla.
        assert set(cols["impaga"]) <= {0.0, 1.0}

    def test_la_regla_viaja_de_vuelta_para_poder_auditarla(self):
        r = _genera(RECETA + "\nRUIDO: 0.1")
        assert "deuda > 0.6" in (r.get("domain_rules") or "")
        assert "NOISE: 0.1" in (r.get("domain_rules") or "")


class TestElRuidoMuerde:
    """Sin esto, el modelo sale al 100 % y no se sostiene."""

    def _contradicen(self, receta: str) -> float:
        cab, cols = _columnas(_genera(receta, filas=4000))
        objetivo = cols[cab[-1]]
        fallos = 0
        for i in range(len(objetivo)):
            esperado = 1.0 if (cols["deuda"][i] > 0.6 or cols["ingresos"][i] < 0.3) else 0.0
            if objetivo[i] != esperado:
                fallos += 1
        return fallos / len(objetivo)

    def test_sin_ruido_el_dataset_obedece_la_regla_entera(self):
        assert self._contradicen(RECETA) == 0.0

    def test_el_ruido_declarado_es_el_ruido_real(self):
        """Una receta que declara un 10 % y produce un 6,5 % miente — y
        pasó: la primera versión elegía la etiqueta nueva ENTRE TODAS,
        así que la mitad de las veces «cambiar» dejaba lo mismo. Con 4.000
        filas el margen de muestreo es ~±1 punto."""
        real = self._contradicen(RECETA + "\nRUIDO: 0.1")
        assert 0.08 <= real <= 0.12, f"ruido declarado 0.1, real {real}"


class TestReproducibilidad:
    """Misma receta + misma semilla = mismo CSV. Es lo que hace publicable
    una receta: quien la recibe obtiene EXACTAMENTE el mismo dataset."""

    def test_misma_semilla_misma_huella_y_mismo_csv(self):
        a, b = _genera(RECETA + "\nRUIDO: 0.1"), _genera(RECETA + "\nRUIDO: 0.1")
        assert a["fingerprint"] == b["fingerprint"]
        assert a["csv_text"] == b["csv_text"]

    def test_otra_semilla_otro_csv(self):
        a = _genera(RECETA + "\nRUIDO: 0.1", semilla=7)
        b = _genera(RECETA + "\nRUIDO: 0.1", semilla=8)
        assert a["csv_text"] != b["csv_text"]


class TestLoQueLaRecetaNOHaceTodavia:
    """Media verdad tranquilizadora: una receta de regresión ignorada en
    silencio devolvería ruido con aspecto de datos buenos."""

    def test_una_receta_de_regresion_no_se_ignora_en_silencio(self):
        mxai = MXAI_BINARIO.replace(
            "  LAYER Dense units=1 activation=sigmoid\n  OUTPUT impaga: Probability",
            "  LAYER Dense units=1 activation=linear\n  OUTPUT impaga: Scalar",
        )
        train = TRAIN_BINARIO.replace("TARGET impaga: Probability", "TARGET impaga: Scalar")
        train = train.replace("TYPE binary_cross_entropy", "TYPE mse")
        r = _generate_synthetic_dataset(mxai, train, rows=100, seed=7, mode="coherent",
                                        use_llm=False, recipe_text=RECETA)
        # El dataset se genera igual (no se rompe el flujo), pero el
        # origen de las etiquetas NO puede decir que hubo reglas.
        assert r["ok"] is True
        assert r.get("label_origin") != "synthetic_domain"


# ---------------------------------------------------------------------------
# Contrato 80-C2 — la receta de REGRESIÓN
# ---------------------------------------------------------------------------

MXAI_REGRESION = """PROJECT Consumo

VECTOR Casa[3]
  metros: Scalar
  habitantes: Scalar
  antiguedad: Scalar
END

NETWORK N
  INPUT Casa
  LAYER Dense units=8 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT consumo: Scalar
END

GRAPH
  Casa -> N
END
"""

TRAIN_REGRESION = """MODEL model.mxai

DATASET D
  SOURCE csv("d.csv")
  INPUT Casa FROM COLUMNS [metros, habitantes, antiguedad]
  TARGET consumo: Scalar
END

LOSS L
  TYPE mse
  PREDICTION consumo
  TARGET consumo
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.1
  UPDATE N.*
END

RUN
  EPOCHS 30
END
"""

RECETA_REG = "consumo = 0.5*metros + 20*habitantes - 0.3*antiguedad + 10"


def _genera_reg(receta: str | None, filas: int = 400):
    return _generate_synthetic_dataset(
        MXAI_REGRESION, TRAIN_REGRESION, rows=filas, seed=7,
        mode="coherent", use_llm=False, recipe_text=receta,
    )


def _r2_de_la_mejor_recta(cab: list[str], cols: dict[str, list[float]]) -> float:
    """R² de la mejor recta simple. Basta para decir si el objetivo
    depende de las entradas o es ruido: no se está midiendo un modelo,
    se está midiendo el DATASET."""
    obj = cols[cab[-1]]
    n = len(obj)
    my = sum(obj) / n
    ss_tot = sum((y - my) ** 2 for y in obj)
    mejor = 0.0
    for c in cab[:-1]:
        x = cols[c]
        mx = sum(x) / n
        sxx = sum((a - mx) ** 2 for a in x)
        if sxx == 0 or ss_tot == 0:
            continue
        b1 = sum((a - mx) * (y - my) for a, y in zip(x, obj)) / sxx
        b0 = my - b1 * mx
        ss_res = sum((y - (b0 + b1 * a)) ** 2 for a, y in zip(x, obj))
        mejor = max(mejor, 1 - ss_res / ss_tot)
    return mejor


class TestLaRecetaDeRegresion:
    """Un objetivo continuo se CALCULA, no se elige de una lista."""

    def test_el_parser_lee_terminos_constante_y_ruido(self):
        from matrixai.training.domain_rules import parse_regression_recipe
        r = parse_regression_recipe("# comentario\ny = 0.4*edad - 0.5*deuda + 12\nNOISE: 0.1")
        assert r is not None
        assert r.objetivo == "y"
        assert r.coeficientes == (("edad", 0.4), ("deuda", -0.5))
        assert r.constante == pytest.approx(12.0)
        assert r.noise == pytest.approx(0.1)

    def test_una_regla_de_clases_NO_es_una_receta_de_regresion(self):
        """Devuelve `None` en vez de inventarse una expresión: escribir
        reglas de clase sobre un objetivo continuo es un error de quien
        escribe, y hay que decírselo."""
        from matrixai.training.domain_rules import parse_regression_recipe
        assert parse_regression_recipe("1: a > 1\nDEFAULT: 0") is None

    def test_normalizar_conserva_el_resultado(self):
        """Los coeficientes se escriben en la escala del dominio y el
        generador muestrea en [0,1]: la conversión no puede cambiar el
        valor que sale."""
        from matrixai.training.domain_rules import parse_regression_recipe
        r = parse_regression_recipe("y = 2*edad + 5")
        assert r is not None
        n = r.normalizada({"edad": (0.0, 100.0)})
        # edad = 40 años → normalizada 0,4 → las dos dan 85.
        assert r.valor_para({"edad": 40.0}) == pytest.approx(n.valor_para({"edad": 0.4}))

    def test_sin_receta_la_regresion_sigue_siendo_ruido(self):
        cab, cols = _columnas(_genera_reg(None))
        assert _r2_de_la_mejor_recta(cab, cols) < 0.05

    def test_con_receta_el_objetivo_depende_de_las_entradas(self):
        """El criterio de cierre de 80-C2: R² > 0,9 sin LLM."""
        cab, cols = _columnas(_genera_reg(RECETA_REG + "\nNOISE: 0.05"))
        assert _r2_de_la_mejor_recta(cab, cols) > 0.9

    def test_el_ruido_de_la_regresion_muerde(self):
        """Y muerde sobre el recorrido REAL de lo calculado. Primero se
        hizo sobre el rango declarado del objetivo —(-1, 1) cuando nadie
        lo declara— y un `NOISE: 0.05` desviaba ±0,1 sobre valores de
        cientos: el ruido no existía y el R² salía perfecto."""
        limpio = _r2_de_la_mejor_recta(*reversed(list(reversed(_columnas(_genera_reg(RECETA_REG))))))
        sucio = _r2_de_la_mejor_recta(*reversed(list(reversed(_columnas(_genera_reg(RECETA_REG + "\nNOISE: 0.3"))))))
        assert limpio > 0.99
        assert sucio < 0.9, f"con 30 % de ruido el R² debería caer, y es {sucio}"

    def test_la_receta_de_regresion_viaja_de_vuelta(self):
        r = _genera_reg(RECETA_REG + "\nNOISE: 0.05")
        assert r["label_origin"] == "synthetic_domain"
        assert "0.5*metros" in (r.get("domain_rules") or "")
        # Y la nota NO se lo atribuye al LLM, que no ha intervenido.
        assert "LLM" not in (r.get("domain_notice") or "")


# ---------------------------------------------------------------------------
# Contrato 80-C4 — el reparto de clases
# ---------------------------------------------------------------------------

MXAI_TRES = """PROJECT Riesgo

VECTOR P[3]
  edad: Scalar
  tension: Scalar
  glucosa: Scalar
END

NETWORK N
  INPUT P
  LAYER Dense units=8 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT riesgo: Label[BAJO, MEDIO, ALTO]
END

GRAPH
  P -> N
END
"""

TRAIN_TRES = """MODEL model.mxai

DATASET D
  SOURCE csv("d.csv")
  INPUT P FROM COLUMNS [edad, tension, glucosa]
  TARGET riesgo: Label[BAJO, MEDIO, ALTO]
END

LOSS L
  TYPE cross_entropy
  PREDICTION riesgo
  TARGET riesgo
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.1
  UPDATE N.*
END

RUN
  EPOCHS 20
END
"""

REGLA_TRES = "ALTO: glucosa > 0.8\nMEDIO: glucosa > 0.5\nDEFAULT: BAJO"


def _reparto(receta: str, filas: int = 500) -> dict[str, float]:
    r = _generate_synthetic_dataset(MXAI_TRES, TRAIN_TRES, rows=filas, seed=7,
                                    mode="coherent", use_llm=False, recipe_text=receta)
    assert r["ok"] is True, r.get("error")
    lineas = [l.strip().split(",") for l in r["csv_text"].strip().split("\n")]
    cab, datos = lineas[0], lineas[1:]
    i = len(cab) - 1
    total = len(datos)
    cuenta: dict[str, int] = {}
    for f in datos:
        cuenta[f[i]] = cuenta.get(f[i], 0) + 1
    return {k: v / total for k, v in cuenta.items()}


class TestElRepartoDeClases:
    """Que la clase rara sea rara, como en la vida."""

    def test_el_idioma_admite_porcentajes_y_proporciones(self):
        a = parse_domain_rules(REGLA_TRES + "\nBALANCE: ALTO=10%")
        b = parse_domain_rules(REGLA_TRES + "\nBALANCE: ALTO=0.1")
        assert dict(a.balance) == dict(b.balance) == {"ALTO": 0.1}
        assert "BALANCE: ALTO=0.1" in a.to_text()

    def test_el_reparto_pedido_se_cumple(self):
        """Criterio de cierre de 80-C4: ±2 puntos."""
        natural = _reparto(REGLA_TRES)
        pedido = _reparto(REGLA_TRES + "\nBALANCE: ALTO=0.1")
        # El reparto natural NO es el pedido: si lo fuera, esta prueba
        # pasaría sin que el mecanismo existiera.
        assert abs(natural.get("ALTO", 0) - 0.1) > 0.05
        assert abs(pedido.get("ALTO", 0) - 0.1) <= 0.02, pedido

    def test_las_filas_SIGUEN_obedeciendo_la_regla(self):
        """La invariante que hace honesto el mecanismo.

        El reparto se cumple RESEMBRANDO filas, no reetiquetándolas. Si se
        reetiquetara, el dataset contradiría la regla que su propia receta
        publica — y esa receta se exporta para que alguien la reproduzca.
        """
        r = _generate_synthetic_dataset(
            MXAI_TRES, TRAIN_TRES, rows=400, seed=7, mode="coherent",
            use_llm=False, recipe_text=REGLA_TRES + "\nBALANCE: ALTO=0.1",
        )
        lineas = [l.strip().split(",") for l in r["csv_text"].strip().split("\n")]
        cab, datos = lineas[0], lineas[1:]
        idx = {c: i for i, c in enumerate(cab)}
        for f in datos:
            glucosa = float(f[idx["glucosa"]])
            esperada = "ALTO" if glucosa > 0.8 else "MEDIO" if glucosa > 0.5 else "BAJO"
            assert f[idx[cab[-1]]] == esperada, (
                f"fila con glucosa={glucosa} etiquetada {f[idx[cab[-1]]]}: "
                "el reparto ha reetiquetado en vez de resembrar"
            )


def test_una_receta_que_no_discrimina_no_se_le_echa_al_LLM() -> None:
    """El aviso de degeneración dice de QUIÉN era la regla.

    MEDIDO conduciendo el 2026-08-18: con `deuda` de 0 a 100.000 y la
    receta ``deuda > 0.6`` —la forma del ejemplo que el propio producto
    sugiere— todas las filas caen en la misma clase, el core se pasa a
    etiquetas aleatorias (bien) y avisaba de que «las reglas de dominio
    propuestas por el LLM no discriminaron».

    Quien escribió esa regla es una persona, y el número que hay que
    tocar es suyo: leer que el fallo fue de la máquina es exactamente lo
    que impide arreglarlo. El caso sin receta sigue hablando del LLM,
    porque ahí sí es verdad.
    """
    rangos = {"edad": (18, 90), "ingresos": (0, 120000), "deuda": (0, 100000)}
    r = _generate_synthetic_dataset(
        MXAI_BINARIO, TRAIN_BINARIO, rows=200, seed=7, mode="coherent", use_llm=False,
        field_ranges_override=rangos, recipe_text="1: deuda > 0.6\nDEFAULT: 0",
    )

    assert r["ok"] is True
    # La honestidad de fondo no cambia: sin señal, etiquetas aleatorias.
    assert r["label_origin"] == "synthetic_random"
    aviso = r.get("domain_degenerate_warning", "")
    assert "no discriminó" in aviso, aviso
    assert "LLM" not in aviso, "se le atribuye al LLM la receta que escribió una persona"
    # Y con la salida: el umbral va en las unidades de los datos.
    assert "unidades" in aviso, aviso
