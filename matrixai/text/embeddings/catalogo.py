# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — los CANDIDATOS del catálogo, y solo los candidatos.

Aquí se escribe a mano lo que es una DECISIÓN: qué repositorio, qué revisión
exacta, qué ficheros hacen falta, de qué familia es, cómo se agrupa la salida
y qué idiomas dice cubrir su autor (con el enlace donde lo dice).

Aquí NO se escribe ningún número medible. Tamaño, digest, dimensión, tiempo
por 1.000 textos, licencia SPDX efectiva y cobertura real de idioma salen de
`medicion.py` y viven en `catalogo_medido.json`. Si un número de la tabla se
puede teclear a mano, la tabla no vale: es la misma regla que ya gobierna
`TERCEROS.md`, y por el mismo motivo — para que caduque RUIDOSA.

`idiomas_segun_su_autor` es la excepción que confirma la regla: es una CITA,
no una medida, y por eso lleva su URL al lado y nunca se presenta como
cobertura comprobada. Lo comprobado lo dice `idiomas_medidos`.

Aquí NO están el pooling ni la normalización, y se quitaron a propósito: los
declara el propio paquete descargado (`modules.json` de sentence-transformers,
`config.json` de model2vec) y los lee `proveedor.py`. Estaban escritos aquí a
mano y ya mentían — `potion-base-8M` figuraba como que no normaliza cuando su
`config.json` dice `"normalize": true`. Dos sitios declarando lo mismo acaban
divergiendo, y este divergió antes de llegar a existir del todo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Familia = Literal["estatica", "onnx_frase"]


@dataclass(frozen=True)
class Candidato:
    """Un proveedor candidato. Todo lo de aquí es elección, no medida."""

    id: str
    familia: Familia
    repo: str
    #: Commit completo (40 hex). Nunca una rama: 102 invariante 7 — la
    #: licencia se fija por checkpoint concreto, y `main` no es un checkpoint.
    revision: str
    #: Ficheros a traer, por su ruta dentro del repo. El primero es el modelo.
    ficheros: tuple[str, ...]
    #: Ruta del fichero que ejecuta onnxruntime.
    modelo: str
    #: Qué tokenizador necesita. `wordpiece` lo implementa este paquete en
    #: stdlib; `unigram_sentencepiece` NO (ver 107_C1 en el informe).
    tokenizador: Literal["wordpiece", "unigram_sentencepiece"]
    #: ¿El pipeline del proveedor añade `[CLS]`/`[SEP]` al texto?
    #:
    #: MEDIDO, y no es un detalle: el `tokenizer.json` de `potion-base-8M`
    #: trae un post-procesador que inserta los ids 101 y 102, que en SU
    #: vocabulario podado (29.528 entradas) no son `[CLS]`/`[SEP]` —que están
    #: en el 2 y el 3— sino `×` y `ß`. Un estático promedia los vectores de
    #: sus tokens: dejarlo puesto mete `×` y `ß` en cada texto del mundo.
    #: `model2vec`, la librería de ese proveedor, tokeniza sin especiales.
    tokens_especiales: bool
    #: CITA de la ficha del autor — no es una medición.
    idiomas_segun_su_autor: str
    fuente_de_esa_cita: str
    #: Por qué está en la lista, y qué se espera de él.
    motivo: str
    #: Longitud efectiva que aplica la LIBRERÍA del proveedor, cuando no está
    #: en ningún fichero del paquete. `None` = el paquete la declara y basta.
    #:
    #: Esto existe por un hallazgo medido el 2026-09-14 y va con nombre y
    #: apellidos porque contradice el invariante 6 del 107 («sin el digest del
    #: tokenizer y el de los pesos, `predict.py` no promete el mismo vector»):
    #: con los dos digests correctos, el vector TAMPOCO está prometido.
    #: `model2vec.StaticModel.encode()` trunca a 512 tokens por el valor por
    #: defecto de su parámetro `max_length` — un número que no aparece en
    #: `config.json`, ni en `tokenizer.json`, ni en `modules.json`. Medido
    #: sobre un texto de 8.008 tokens: con 512 el coseno contra la librería
    #: es 0,658; con `max_length=None`, 1,000. Quien reimplemente el proveedor
    #: leyendo solo los ficheros da otro vector para los textos largos, y no
    #: hay nada en el paquete que avise.
    longitud_efectiva_de_su_libreria: int | None = None
    fuente_de_la_longitud_efectiva: str = ""
    #: Si no se descargó en esta pasada, por qué. Cadena vacía = sí se bajó.
    no_descargado_porque: str = ""
    notas: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError(f"{self.id}: revision debe ser el commit completo de 40 hex")
        if self.modelo not in self.ficheros:
            raise ValueError(f"{self.id}: el modelo {self.modelo!r} no está en ficheros")


#: Los candidatos de 107-C1. Dos familias, como pide el corte.
CANDIDATOS: tuple[Candidato, ...] = (
    # ---------------------------------------------------------------- estáticos
    Candidato(
        id="potion-base-8M",
        familia="estatica",
        repo="minishlab/potion-base-8M",
        revision="bf8b056651a2c21b8d2565580b8569da283cab23",
        ficheros=("onnx/model.onnx", "tokenizer.json", "vocab.txt", "config.json", "tokenizer_config.json", "modules.json"),
        modelo="onnx/model.onnx",
        tokenizador="wordpiece",
        tokens_especiales=False,
        longitud_efectiva_de_su_libreria=512,
        fuente_de_la_longitud_efectiva="valor por defecto de max_length en "
        "model2vec.StaticModel.encode (model2vec 0.9.0), leído con inspect.signature",
        idiomas_segun_su_autor="English",
        fuente_de_esa_cita="https://huggingface.co/minishlab/potion-base-8M",
        motivo="Familia estática de referencia: una tabla de vectores por token, sin "
        "transformer, destilada de un modelo de frase. Es el más barato de la lista "
        "y el que se usa para comprobar que un idioma NO cubierto se declara.",
    ),
    Candidato(
        id="potion-base-2M",
        familia="estatica",
        repo="minishlab/potion-base-2M",
        revision="389b9f64be5aa4ae7a6bc6fe95ef20ce485ae5da",
        ficheros=("onnx/model.onnx", "tokenizer.json", "vocab.txt", "config.json", "tokenizer_config.json", "modules.json"),
        modelo="onnx/model.onnx",
        tokenizador="wordpiece",
        tokens_especiales=False,
        longitud_efectiva_de_su_libreria=512,
        fuente_de_la_longitud_efectiva="valor por defecto de max_length en "
        "model2vec.StaticModel.encode (model2vec 0.9.0), leído con inspect.signature",
        idiomas_segun_su_autor="English",
        fuente_de_esa_cita="https://huggingface.co/minishlab/potion-base-2M",
        motivo="El mismo proveedor en su talla más pequeña: sirve para medir qué cuesta "
        "en calidad bajar de 8M a 2M parámetros, si alguna vez importa el tamaño.",
        no_descargado_porque="",
    ),
    Candidato(
        id="potion-multilingual-128M",
        familia="estatica",
        repo="minishlab/potion-multilingual-128M",
        revision="73908c3438cf03b6a01bcb9611d62b23d0726f08",
        ficheros=("onnx/model.onnx", "tokenizer.json", "vocab.txt", "config.json", "tokenizer_config.json", "modules.json"),
        modelo="onnx/model.onnx",
        # MEDIDO, no supuesto: el repo trae un `vocab.txt` de 6,4 MB que hace
        # pensar en WordPiece, pero su `tokenizer.json` declara `Unigram` con
        # 500.353 piezas y un `precompiled_charsmap`. El `vocab.txt` es un
        # volcado, no el tokenizador. Comprobado el 2026-09-14 leyendo el
        # fichero, después de haberlo declarado `wordpiece` por el nombre.
        tokenizador="unigram_sentencepiece",
        tokens_especiales=False,
        longitud_efectiva_de_su_libreria=512,
        fuente_de_la_longitud_efectiva="valor por defecto de max_length en "
        "model2vec.StaticModel.encode (model2vec 0.9.0), leído con inspect.signature",
        idiomas_segun_su_autor="101 idiomas, español incluido",
        fuente_de_esa_cita="https://huggingface.co/minishlab/potion-multilingual-128M",
        motivo="El único estático de la lista que dice cubrir español. Es la opción "
        "natural si se quiere texto en es sin pagar un transformer por fila.",
        # Se bajó el 2026-09-17 por decisión de Roberto del 16-09 («Sí, descargarlo
        # y medirlo»). El motivo de antes —537 MB por una tabla de vectores, y un
        # tokenizador Unigram que exige la misma dependencia que el multilingüe
        # de la otra familia— era una decisión pendiente, no una imposibilidad.
        no_descargado_porque="",
    ),
    # ------------------------------------------------------------ transformers
    Candidato(
        id="all-MiniLM-L6-v2-onnx",
        familia="onnx_frase",
        repo="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ficheros=(
            "onnx/model.onnx", "tokenizer.json", "vocab.txt", "config.json",
            "tokenizer_config.json", "special_tokens_map.json", "modules.json",
            "1_Pooling/config.json", "sentence_bert_config.json",
        ),
        modelo="onnx/model.onnx",
        tokenizador="wordpiece",
        tokens_especiales=True,
        idiomas_segun_su_autor="English",
        fuente_de_esa_cita="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        motivo="El transformer de frase pequeño más usado. Pesos en float32 (no "
        "cuantizados) para que el tiempo medido sea el del modelo, no el de una "
        "cuantización concreta.",
    ),
    Candidato(
        id="paraphrase-multilingual-MiniLM-L12-v2-onnx-int8",
        familia="onnx_frase",
        repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        ficheros=(
            "onnx/model_qint8_avx512_vnni.onnx", "tokenizer.json", "config.json",
            "tokenizer_config.json", "special_tokens_map.json", "modules.json",
            "1_Pooling/config.json", "sentence_bert_config.json",
        ),
        modelo="onnx/model_qint8_avx512_vnni.onnx",
        tokenizador="unigram_sentencepiece",
        tokens_especiales=True,
        idiomas_segun_su_autor="50+ idiomas, español incluido",
        fuente_de_esa_cita="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        motivo="Transformer de frase multilingüe; la versión int8 es la única talla "
        "pequeña que existe de él (la float32 son 470 MB).",
    ),
    Candidato(
        id="multilingual-e5-small-onnx-int8",
        familia="onnx_frase",
        repo="intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        ficheros=(
            "onnx/model_qint8_avx512_vnni.onnx", "tokenizer.json", "config.json",
            "tokenizer_config.json", "special_tokens_map.json",
        ),
        modelo="onnx/model_qint8_avx512_vnni.onnx",
        tokenizador="unigram_sentencepiece",
        tokens_especiales=True,
        idiomas_segun_su_autor="100 idiomas (base XLM-RoBERTa), español incluido",
        fuente_de_esa_cita="https://huggingface.co/intfloat/multilingual-e5-small",
        motivo="Alternativa multilingüe con licencia MIT en vez de Apache-2.0, y con "
        "prefijos de tarea ('query: '/'passage: ') que cambian el vector — eso hay "
        "que declararlo, no descubrirlo luego.",
        no_descargado_porque="su tokenizador es Unigram de SentencePiece, igual que el "
        "del otro multilingüe, y ese ya está descargado y medido: bajar 135 MB más para "
        "repetir la misma limitación no aporta un dato nuevo. Queda fijado y medido en "
        "lo que no exige bajarlo (tamaño, licencia, tokenizador, revisión).",
        notas=("Exige prefijar cada texto con 'query: ' o 'passage: '; sin el prefijo el vector no es el que su ficha mide.",),
    ),
)


def por_id(id_: str) -> Candidato:
    for c in CANDIDATOS:
        if c.id == id_:
            return c
    raise KeyError(f"candidato {id_!r} desconocido; hay {[c.id for c in CANDIDATOS]}")


def de_familia(familia: Familia) -> tuple[Candidato, ...]:
    return tuple(c for c in CANDIDATOS if c.familia == familia)
