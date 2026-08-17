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
        reproduce el dataset — y la reproducibilidad es el contrato."""
        r = parse_domain_rules(RECETA + "\nRUIDO: 0.25")
        assert "RUIDO: 0.25" in r.to_text()

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
        assert "RUIDO: 0.1" in (r.get("domain_rules") or "")


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
