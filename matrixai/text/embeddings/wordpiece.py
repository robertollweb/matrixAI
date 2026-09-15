# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — WordPiece de BERT en stdlib, leído del `tokenizer.json` del modelo.

El tokenizer es parte del modelo (107, invariante 6): el mismo `model.onnx`
con otro vocabulario produce OTRO vector. Por eso esto no es «un tokenizador
parecido»: se lee el `tokenizer.json` que viene con los pesos y se REPRODUCE
lo que ese fichero declara — normalizador, pre-tokenizador y vocabulario.

Y por eso mismo `desde_tokenizer_json` es ESTRICTO hasta la grosería: si el
fichero declara un normalizador, un pre-tokenizador o un modelo que esta
implementación no reproduce exactamente, se niega a construir nada. Un
tokenizador «aproximado» no da un error: da un vector plausible y equivocado,
que es la peor de las dos cosas.

Lo reproducido, con su nombre en `tokenizers` (la implementación de
referencia en Rust, que es contra la que se ha medido — ver
`test_c107_c1_wordpiece.py`):

  · `BertNormalizer`  — limpieza de controles, espacios a ' ', separación de
    caracteres CJK, quitado de acentos y minúsculas. Detalle que cuesta caro
    si se pasa por alto: cuando `strip_accents` viene a `null`, NO significa
    «no quitar acentos» — significa «lo que diga `lowercase`». Un modelo
    *uncased* quita los acentos, así que «canción» y «cancion» son el mismo
    texto para él.
  · `BertPreTokenizer` — corta por espacios y AÍSLA cada signo de puntuación.
  · `WordPiece`       — voraz de más largo a más corto, con `##` delante de
    cada trozo que no abre palabra; una palabra con un trozo desconocido va
    entera a `[UNK]`.
  · `TemplateProcessing` — `[CLS] … [SEP]`.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Lo que esta implementación sabe reproducir. Cualquier otra cosa se rechaza.
NORMALIZADOR = "BertNormalizer"
PRE_TOKENIZADOR = "BertPreTokenizer"
MODELO = "WordPiece"


class TokenizadorNoSoportado(Exception):
    """El `tokenizer.json` declara algo que esta implementación no reproduce
    exactamente. Nunca se aproxima: se para."""


def _es_control(c: str) -> bool:
    if c in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(c).startswith("C")


def _es_espacio(c: str) -> bool:
    if c in ("\t", "\n", "\r"):
        return True
    return c.isspace()


def _es_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


_PUNTUACION_ASCII = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _es_puntuacion(c: str) -> bool:
    """La misma definición que usa `BertPreTokenizer`: puntuación ASCII (que
    incluye símbolos como `$`, `+` o `~`, que Unicode NO clasifica como
    puntuación) más cualquier categoría Unicode `P*`."""
    return c in _PUNTUACION_ASCII or unicodedata.category(c).startswith("P")


@dataclass(frozen=True)
class Tokenizado:
    ids: list[int]
    tokens: list[str]


class WordPiece:
    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str,
        prefijo_continuacion: str,
        max_chars_por_palabra: int,
        minusculas: bool,
        quitar_acentos: bool,
        limpiar_texto: bool,
        cjk: bool,
        cls_token: str,
        sep_token: str,
        pad_token: str,
        max_longitud: int,
    ) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.prefijo = prefijo_continuacion
        self.max_chars = max_chars_por_palabra
        self.minusculas = minusculas
        self.quitar_acentos = quitar_acentos
        self.limpiar_texto = limpiar_texto
        self.cjk = cjk
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.max_longitud = max_longitud
        for nombre, tok in (("unk", unk_token), ("cls", cls_token), ("sep", sep_token), ("pad", pad_token)):
            if tok not in vocab:
                raise TokenizadorNoSoportado(f"el vocabulario no trae el token {nombre} {tok!r}")
        self.pad_id = vocab[pad_token]

    # ---------------------------------------------------------------- normaliza
    def normalizar(self, texto: str) -> str:
        if self.limpiar_texto:
            salida = []
            for c in texto:
                if c == "\0" or c == "�" or _es_control(c):
                    continue
                salida.append(" " if _es_espacio(c) else c)
            texto = "".join(salida)
        if self.cjk:
            salida = []
            for c in texto:
                if _es_cjk(ord(c)):
                    salida.append(" ")
                    salida.append(c)
                    salida.append(" ")
                else:
                    salida.append(c)
            texto = "".join(salida)
        if self.quitar_acentos:
            texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
        if self.minusculas:
            texto = texto.lower()
        return texto

    # ------------------------------------------------------------- pre-tokeniza
    @staticmethod
    def pre_tokenizar(texto: str) -> list[str]:
        palabras: list[str] = []
        for bruto in texto.split():
            actual = ""
            for c in bruto:
                if _es_puntuacion(c):
                    if actual:
                        palabras.append(actual)
                        actual = ""
                    palabras.append(c)
                else:
                    actual += c
            if actual:
                palabras.append(actual)
        return palabras

    # ------------------------------------------------------------------ trocea
    def trocear(self, palabra: str) -> list[str]:
        if len(palabra) > self.max_chars:
            return [self.unk_token]
        trozos: list[str] = []
        inicio = 0
        n = len(palabra)
        while inicio < n:
            fin = n
            encontrado = None
            while inicio < fin:
                pieza = palabra[inicio:fin]
                if inicio > 0:
                    pieza = self.prefijo + pieza
                if pieza in self.vocab:
                    encontrado = pieza
                    break
                fin -= 1
            if encontrado is None:
                return [self.unk_token]
            trozos.append(encontrado)
            inicio = fin
        return trozos

    # ----------------------------------------------------------------- codifica
    def codificar(self, texto: str, *, con_especiales: bool = True) -> Tokenizado:
        """`con_especiales=False` devuelve solo los tokens de contenido.

        No es un capricho: los dos pipelines de este catálogo difieren justo
        ahí. Un transformer de frase de `sentence-transformers` ESPERA
        `[CLS] … [SEP]` y los promedia con el resto. Un embedding estático
        (`model2vec`) NO los añade: su vector es la media de los tokens de
        contenido y nada más. Meterle `[CLS]`/`[SEP]` a un estático le suma
        dos vectores que no pintan nada, y el resultado es un vector
        plausible y equivocado.
        """
        tokens = [self.cls_token] if con_especiales else []
        for palabra in self.pre_tokenizar(self.normalizar(texto)):
            tokens.extend(self.trocear(palabra))
        if con_especiales:
            tokens.append(self.sep_token)
        if con_especiales and len(tokens) > self.max_longitud:
            # `LongestFirst` sobre una sola secuencia recorta por el final y
            # deja el [SEP] en su sitio: un texto truncado sigue estando bien
            # formado, que es lo que el modelo espera.
            tokens = tokens[: self.max_longitud - 1] + [self.sep_token]
        elif not con_especiales and len(tokens) > self.max_longitud:
            tokens = tokens[: self.max_longitud]
        return Tokenizado(ids=[self.vocab[t] for t in tokens], tokens=tokens)


def _exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise TokenizadorNoSoportado(mensaje)


def desde_tokenizer_json(
    ruta: Path, *, max_longitud: int | None = None, forzar_longitud: bool = False
) -> WordPiece:
    """Construye el tokenizador leyendo el fichero del modelo, o se niega.

    `max_longitud` se usa cuando el `tokenizer.json` no declara truncado.
    `forzar_longitud=True` la impone aunque lo declare, y solo lo usa
    `proveedor.py` cuando el paquete declara varias longitudes distintas en
    varios ficheros y hay que quedarse con la que usa la implementación
    oficial del modelo — ver `proveedor.elegir_longitud`.
    """
    datos: dict[str, Any] = json.loads(Path(ruta).read_text(encoding="utf-8"))

    modelo = datos.get("model") or {}
    _exigir(modelo.get("type") == MODELO, f"{ruta}: model.type es {modelo.get('type')!r}, no {MODELO!r}")

    norm = datos.get("normalizer") or {}
    _exigir(norm.get("type") == NORMALIZADOR, f"{ruta}: normalizer es {norm.get('type')!r}, no {NORMALIZADOR!r}")

    pre = datos.get("pre_tokenizer") or {}
    _exigir(pre.get("type") == PRE_TOKENIZADOR, f"{ruta}: pre_tokenizer es {pre.get('type')!r}, no {PRE_TOKENIZADOR!r}")

    post = datos.get("post_processor") or {}
    _exigir(
        post.get("type") == "TemplateProcessing",
        f"{ruta}: post_processor es {post.get('type')!r}, no 'TemplateProcessing'",
    )
    plantilla = [list(p.keys())[0] for p in post.get("single", [])]
    _exigir(
        plantilla == ["SpecialToken", "Sequence", "SpecialToken"],
        f"{ruta}: la plantilla de una secuencia es {plantilla}, no '[CLS] A [SEP]'",
    )
    cls_token = post["single"][0]["SpecialToken"]["id"]
    sep_token = post["single"][2]["SpecialToken"]["id"]

    vocab = modelo.get("vocab")
    _exigir(isinstance(vocab, dict) and bool(vocab), f"{ruta}: model.vocab no es un diccionario con contenido")

    minusculas = bool(norm.get("lowercase", False))
    quitar = norm.get("strip_accents")
    # `null` NO es `false`: es «lo que diga lowercase» (así lo resuelve
    # `BertNormalizer`). Leerlo como `false` dejaría los acentos puestos en un
    # modelo uncased y el vector saldría distinto sin que nadie se enterase.
    quitar_acentos = bool(minusculas) if quitar is None else bool(quitar)

    truncado = datos.get("truncation") or {}
    if forzar_longitud:
        if max_longitud is None:
            raise TokenizadorNoSoportado(f"{ruta}: forzar_longitud sin max_longitud")
        longitud = int(max_longitud)
    elif truncado:
        _exigir(
            truncado.get("strategy") in ("LongestFirst", "longest_first"),
            f"{ruta}: estrategia de truncado {truncado.get('strategy')!r} no soportada",
        )
        longitud = int(truncado["max_length"])
    elif max_longitud is not None:
        longitud = int(max_longitud)
    else:
        raise TokenizadorNoSoportado(
            f"{ruta}: no declara truncado y no se le ha dado `max_longitud` — "
            "un texto largo reventaría el modelo sin decir por qué"
        )

    relleno = datos.get("padding") or {}
    pad_token = relleno.get("pad_token") or "[PAD]"

    return WordPiece(
        vocab=vocab,
        unk_token=modelo.get("unk_token") or "[UNK]",
        prefijo_continuacion=modelo.get("continuing_subword_prefix") or "##",
        max_chars_por_palabra=int(modelo.get("max_input_chars_per_word") or 100),
        minusculas=minusculas,
        quitar_acentos=quitar_acentos,
        limpiar_texto=bool(norm.get("clean_text", True)),
        cjk=bool(norm.get("handle_chinese_chars", True)),
        cls_token=cls_token,
        sep_token=sep_token,
        pad_token=pad_token,
        max_longitud=longitud,
    )
