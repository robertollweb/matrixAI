# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Avisar AL GENERAR de que esta instalación no podrá entrenar un modelo de texto.

POR QUÉ EXISTE, medido el 2026-08-14 conduciendo la imagen publicada
`matrixai-studio:v1.8`: el paquete CPU **no lleva torch**, y un `BLOCK
TRANSFORMER` sin torch no entrena (invariante 6 del contrato 51). Antes de
esto se podía construir el modelo, preparar los datos y pulsar Entrenar —
y solo entonces salía el motivo. Roberto: «en cpu lo que tenemos es que
avisar si se pide una red de transformers».

Lo que se prueba aquí es que `text_training_blocked` dice la verdad en los
CUATRO estados que existen, porque cada uno lleva a un arreglo distinto:
falta torch (instala el paquete de GPU) ≠ se lo han forzado a stdlib
(quita el ajuste) ≠ puede ≠ no es un modelo de texto.
"""
from __future__ import annotations

import matrixai.playground as pg


# Un transformer mínimo y un denso mínimo, para separar "es de texto" de
# "esta máquina puede". Se parsean de verdad: `_network_is_transformer`
# mira el AST, no el texto, así que un .mxai de mentira no valdría.
# LOS DOS `.mxai` SON REALES: salieron del core el 2026-08-14 pidiéndoselos
# por prompt, no están escritos a mano. La primera versión de este fichero
# los inventó y `_network_is_transformer` devolvió `False` para el de texto
# — el aserto de abajo lo cazó. Un fixture describe algo que EXISTE.
MXAI_TEXTO = """PROJECT SentimentClassifierProject

SEQUENCE Comment
  length = 64
  vocab_size = 259
END

NETWORK SentimentClassifier
  INPUT Comment
  EMBEDDING tok FROM Comment DIM 64
  BLOCK enc TRANSFORMER
    LAYERS 2
    HEADS 4
  END
  POOL mean
  LAYER Dense units=1 activation=sigmoid
  OUTPUT predicted_prob: Probability
END

GRAPH
  Comment -> SentimentClassifier
END
"""

MXAI_DENSO = """PROJECT ComentarioClasificadorProject

VECTOR ComentarioCliente[1]
  comentario_texto: Scalar
END

NETWORK ComentarioClasificador
  INPUT ComentarioCliente
  LAYER Dense units=32 activation=relu
  BLOCK res1
    LAYER Dense units=32 activation=relu
    LAYER LayerNorm
    LAYER Dropout rate=0.2
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=1 activation=sigmoid
  OUTPUT predicted_prob: Probability
END

GRAPH
  ComentarioCliente -> ComentarioClasificador
END
"""


def _es_transformer(mxai: str) -> bool:
    return pg._network_is_transformer(mxai)


def test_los_dos_mxai_de_prueba_son_lo_que_dicen_ser():
    """UN FIXTURE DESCRIBE ALGO QUE EXISTE.

    Si el `.mxai` de texto dejara de parsearse (una palabra clave que
    cambia, un bloque que se renombra), `_network_is_transformer` diría
    `False` y TODAS las pruebas de abajo pasarían en verde sin comprobar
    nada: el aviso no sale porque no hay transformer, no porque funcione.
    Este aserto es el que impide ese verde vacío.
    """
    assert _es_transformer(MXAI_TEXTO) is True
    assert _es_transformer(MXAI_DENSO) is False


def test_sin_torch_avisa_y_manda_al_paquete_de_gpu(monkeypatch):
    monkeypatch.setattr(pg, "_select_transformer_train_device", lambda: (False, "cpu"))
    monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)

    es = pg.text_training_blocked(MXAI_TEXTO, "es")
    assert es is not None
    # Lo que tiene que llevarse quien lo lee: no puede entrenarlo AQUÍ, y
    # dónde sí. Sin la segunda mitad el aviso deja en un callejón.
    assert "torch" in es
    assert "GPU" in es
    # Y NO puede decir que haga falta una GPU: con torch de CPU entrena
    # (medido, accuracy 1,0 en un servidor sin tarjeta). Prometer que
    # hace falta hardware que no hace falta es la media verdad de siempre.
    assert "MATRIXAI_TRAIN_BACKEND" not in es


def test_el_aviso_de_sin_torch_tambien_en_ingles(monkeypatch):
    """Media aplicación traducida se ve peor que ninguna."""
    monkeypatch.setattr(pg, "_select_transformer_train_device", lambda: (False, "cpu"))
    monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)

    en = pg.text_training_blocked(MXAI_TEXTO, "en")
    assert en is not None
    assert "torch" in en
    assert "GPU" in en
    # Palabras FUNCIONALES del castellano, que no se pueden evitar
    # escribiendo en español: si el texto inglés cayera al de por defecto,
    # alguna aparecería.
    for palabra in (" el ", " la ", " que ", " no ", " con "):
        assert palabra not in f" {en} "


def test_backend_forzado_a_stdlib_dice_ESO_y_no_que_falte_torch(monkeypatch):
    """Dos causas distintas con arreglos distintos no se pueden juntar.

    Quien ha forzado `MATRIXAI_TRAIN_BACKEND=stdlib` YA TIENE torch:
    mandarle a instalar el paquete de GPU le haría descargar 1 GB para
    nada, y el problema seguiría ahí.
    """
    monkeypatch.setattr(pg, "_select_transformer_train_device", lambda: (False, "cpu"))
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")

    for locale in ("es", "en"):
        motivo = pg.text_training_blocked(MXAI_TEXTO, locale)
        assert motivo is not None
        assert "MATRIXAI_TRAIN_BACKEND" in motivo
        assert "GPU" not in motivo


def test_con_torch_no_avisa_de_nada(monkeypatch):
    """Un aviso que sale cuando no toca enseña a ignorar los avisos."""
    monkeypatch.setattr(pg, "_select_transformer_train_device", lambda: (True, "cpu"))
    assert pg.text_training_blocked(MXAI_TEXTO, "es") is None


def test_un_modelo_QUE_NO_ES_DE_TEXTO_nunca_avisa(monkeypatch):
    """Sin esto, el paquete CPU apagaría Entrenar en TODOS los modelos.

    Es el fallo que este test existe para cazar: la comprobación de
    máquina («no hay torch») es cierta en el paquete CPU para cualquier
    modelo, y solo debe bloquear a los de texto. Si alguien invierte el
    orden de las dos preguntas, aquí se pone rojo.
    """
    monkeypatch.setattr(pg, "_select_transformer_train_device", lambda: (False, "cpu"))
    assert pg.text_training_blocked(MXAI_DENSO, "es") is None
    assert pg.text_training_blocked(MXAI_DENSO, "en") is None


def test_un_mxai_ilegible_no_revienta_ni_bloquea(monkeypatch):
    """No se sabe si es de texto → no se afirma que no se pueda entrenar.

    Un valor ausente no es un `no`: negar aquí apagaría Entrenar por no
    haber sabido leer el modelo, que es exactamente el dibujo que afirma
    por omisión.
    """
    monkeypatch.setattr(pg, "_select_transformer_train_device", lambda: (False, "cpu"))
    assert pg.text_training_blocked("esto no es un mxai", "es") is None
    assert pg.text_training_blocked("", "es") is None
