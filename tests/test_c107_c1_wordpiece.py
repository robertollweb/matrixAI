# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — el tokenizador es parte del modelo (107, invariante 6).

Lo que se prueba aquí no es «que tokenice»: es que tokenice EXACTAMENTE como
el fichero del modelo declara, y que cuando no sabe hacerlo lo diga en vez de
aproximar. Un tokenizador aproximado no produce un error: produce un vector
plausible y equivocado, que no se ve por ningún lado.

La comprobación contra la implementación de referencia (`tokenizers`, en
Rust) no vive en esta suite porque exigiría una dependencia que el núcleo no
lleva. Se hizo aparte el 2026-09-14 sobre 4.038 textos por modelo —2.009 en
español, 2.009 en inglés (FLORES-200) y 20 casos duros— con **0
discrepancias** en `all-MiniLM-L6-v2` y en `potion-base-8M`. Lo que sí vive
aquí son los casos que fijan las decisiones que se tomaron por el camino.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.text.embeddings.descarga import raiz_cache
from matrixai.text.embeddings.wordpiece import (
    TokenizadorNoSoportado,
    WordPiece,
    desde_tokenizer_json,
)

VOCAB_BASE = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "la": 4,
    "cancion": 5,
    "canción": 6,
    "nino": 7,
    "niño": 8,
    ",": 9,
    "!": 10,
    "una": 11,
    "##ffa": 12,
    "##ble": 13,
    "hola": 14,
    "##s": 15,
    # `un` Y `una`: sin los dos, un troceo que fuera de más CORTO a más largo
    # daría el mismo resultado y el sabotaje saldría verde. Medido: con solo
    # `una` en el vocabulario, invertir la dirección del bucle no rompía nada.
    "un": 16,
    "##affable": 17,
}


def _tok(**kw) -> WordPiece:
    base = dict(
        vocab=dict(VOCAB_BASE),
        unk_token="[UNK]",
        prefijo_continuacion="##",
        max_chars_por_palabra=100,
        minusculas=True,
        quitar_acentos=True,
        limpiar_texto=True,
        cjk=True,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        max_longitud=64,
    )
    base.update(kw)
    return WordPiece(**base)


def _fichero_tokenizer(tmp: Path, **cambios) -> Path:
    datos = {
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
        },
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": dict(VOCAB_BASE),
        },
    }
    datos.update(cambios)
    ruta = tmp / "tokenizer.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


class TestStripAccentsNuloNoEsFalso(unittest.TestCase):
    """`strip_accents: null` NO significa «no quitar acentos»: significa «lo
    que diga lowercase». Leerlo como `false` deja los acentos puestos en un
    modelo uncased y el vector sale distinto sin que nadie se entere — y en
    español eso pasa en casi todas las frases."""

    def test_nulo_con_minusculas_quita_los_acentos(self):
        with TemporaryDirectory() as tmp:
            t = desde_tokenizer_json(_fichero_tokenizer(Path(tmp)), max_longitud=64)
        self.assertTrue(t.quitar_acentos)
        self.assertEqual(t.normalizar("La Canción del Niño"), "la cancion del nino")

    def test_nulo_sin_minusculas_no_los_quita(self):
        with TemporaryDirectory() as tmp:
            ruta = _fichero_tokenizer(
                Path(tmp),
                normalizer={
                    "type": "BertNormalizer",
                    "clean_text": True,
                    "handle_chinese_chars": True,
                    "strip_accents": None,
                    "lowercase": False,
                },
            )
            t = desde_tokenizer_json(ruta, max_longitud=64)
        self.assertFalse(t.quitar_acentos)
        self.assertEqual(t.normalizar("Canción"), "Canción")

    def test_declarado_explicitamente_manda_sobre_lowercase(self):
        with TemporaryDirectory() as tmp:
            ruta = _fichero_tokenizer(
                Path(tmp),
                normalizer={
                    "type": "BertNormalizer",
                    "clean_text": True,
                    "handle_chinese_chars": True,
                    "strip_accents": False,
                    "lowercase": True,
                },
            )
            t = desde_tokenizer_json(ruta, max_longitud=64)
        self.assertFalse(t.quitar_acentos)
        self.assertEqual(t.normalizar("Canción"), "canción")

    def test_una_palabra_con_y_sin_acento_acaban_en_el_MISMO_token(self):
        # La consecuencia de verdad: para un modelo uncased, «canción» y
        # «cancion» son la misma palabra. Si esto deja de ser así, el vector
        # de media España cambia.
        t = _tok()
        self.assertEqual(t.codificar("canción").ids, t.codificar("cancion").ids)


class TestElTroceoEsElDeBERT(unittest.TestCase):
    def test_el_ejemplo_canonico_del_paper(self):
        t = _tok()
        self.assertEqual(t.trocear("unaffable"), ["una", "##ffa", "##ble"])

    def test_gana_el_trozo_mas_LARGO_no_el_primero_que_valga(self):
        """Con `un` y `una` los dos en el vocabulario, la dirección del bucle
        decide. De más largo a más corto: `una`. Al revés: `un`."""
        t = _tok()
        self.assertEqual(t.trocear("una"), ["una"])
        self.assertEqual(t.trocear("unaffable")[0], "una")

    def test_una_palabra_con_un_trozo_desconocido_va_ENTERA_a_unk(self):
        t = _tok()
        self.assertEqual(t.trocear("unaffablezzz"), ["[UNK]"])

    def test_una_palabra_larguisima_va_a_unk_sin_intentarlo(self):
        t = _tok(max_chars_por_palabra=10)
        self.assertEqual(t.trocear("a" * 11), ["[UNK]"])

    def test_la_puntuacion_se_aisla(self):
        self.assertEqual(WordPiece.pre_tokenizar("hola, mundo!"), ["hola", ",", "mundo", "!"])

    def test_los_simbolos_ascii_cuentan_como_puntuacion_aunque_unicode_diga_que_no(self):
        # `$`, `+`, `~` son Sc/Sm/Sk para Unicode, pero BertPreTokenizer los
        # trata como puntuación. Adivinarlo por categoría Unicode daría otros
        # tokens.
        self.assertEqual(WordPiece.pre_tokenizar("a$b+c~d"), ["a", "$", "b", "+", "c", "~", "d"])


class TestTokensEspecialesSegunElPipeline(unittest.TestCase):
    """`potion-base-8M` NO lleva `[CLS]`/`[SEP]`, y no es un detalle menor:
    su `tokenizer.json` trae un post-procesador que inserta los ids 101 y 102,
    que en su vocabulario podado son `×` y `ß`. Un estático promedia los
    vectores de sus tokens, así que dejarlo puesto mete `×` y `ß` en todos los
    textos del mundo."""

    def test_con_especiales_los_pone(self):
        t = _tok()
        self.assertEqual(t.codificar("hola").tokens, ["[CLS]", "hola", "[SEP]"])

    def test_sin_especiales_no_los_pone(self):
        t = _tok()
        self.assertEqual(t.codificar("hola", con_especiales=False).tokens, ["hola"])

    def test_un_texto_sin_tokens_no_se_inventa_ninguno(self):
        t = _tok()
        self.assertEqual(t.codificar("", con_especiales=False).ids, [])

    def test_el_truncado_deja_el_sep_al_final(self):
        t = _tok(max_longitud=4)
        r = t.codificar("hola hola hola hola hola")
        self.assertEqual(len(r.ids), 4)
        self.assertEqual(r.tokens[0], "[CLS]")
        self.assertEqual(r.tokens[-1], "[SEP]")


class TestSeNiegaAAproximar(unittest.TestCase):
    def test_un_modelo_unigram_se_rechaza(self):
        with TemporaryDirectory() as tmp:
            ruta = _fichero_tokenizer(Path(tmp), model={"type": "Unigram", "unk_id": 3, "vocab": [["a", -1.0]]})
            with self.assertRaises(TokenizadorNoSoportado) as ctx:
                desde_tokenizer_json(ruta, max_longitud=64)
        self.assertIn("Unigram", str(ctx.exception))

    def test_otro_normalizador_se_rechaza(self):
        with TemporaryDirectory() as tmp:
            ruta = _fichero_tokenizer(Path(tmp), normalizer={"type": "Precompiled", "precompiled_charsmap": "AA=="})
            with self.assertRaises(TokenizadorNoSoportado) as ctx:
                desde_tokenizer_json(ruta, max_longitud=64)
        self.assertIn("Precompiled", str(ctx.exception))

    def test_otro_pre_tokenizador_se_rechaza(self):
        with TemporaryDirectory() as tmp:
            ruta = _fichero_tokenizer(Path(tmp), pre_tokenizer={"type": "Metaspace", "replacement": "▁"})
            with self.assertRaises(TokenizadorNoSoportado):
                desde_tokenizer_json(ruta, max_longitud=64)

    def test_otra_plantilla_de_post_procesado_se_rechaza(self):
        with TemporaryDirectory() as tmp:
            ruta = _fichero_tokenizer(
                Path(tmp),
                post_processor={"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}]},
            )
            with self.assertRaises(TokenizadorNoSoportado) as ctx:
                desde_tokenizer_json(ruta, max_longitud=64)
        self.assertIn("plantilla", str(ctx.exception))

    def test_sin_truncado_ni_longitud_dada_se_niega_en_vez_de_inventarse_una(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(TokenizadorNoSoportado) as ctx:
                desde_tokenizer_json(_fichero_tokenizer(Path(tmp)))
        self.assertIn("max_longitud", str(ctx.exception))

    def test_un_vocabulario_sin_sus_tokens_especiales_no_construye(self):
        vocab = {k: v for k, v in VOCAB_BASE.items() if k != "[SEP]"}
        with self.assertRaises(TokenizadorNoSoportado):
            _tok(vocab=vocab)


CACHE = raiz_cache()


class TestContraLosFicherosDeVerdad(unittest.TestCase):
    """Sobre los paquetes descargados, si están. No se saltan en silencio:
    el motivo del salto sale por pantalla, y el catálogo medido declara
    aparte qué proveedores están descargados y cuáles no."""

    def _cargar(self, cid: str):
        ruta = CACHE / cid / "tokenizer.json"
        if not ruta.is_file():
            self.skipTest(f"{cid} no está descargado en {CACHE} (los pesos no van en el repositorio)")
        return desde_tokenizer_json(ruta, max_longitud=256, forzar_longitud=True)

    def test_all_minilm_es_uncased_y_quita_acentos(self):
        t = self._cargar("all-MiniLM-L6-v2-onnx")
        self.assertTrue(t.minusculas)
        self.assertTrue(t.quitar_acentos)
        self.assertEqual(t.codificar("Canción").ids, t.codificar("cancion").ids)

    def test_el_troceo_canonico_sobre_el_vocabulario_real(self):
        t = self._cargar("all-MiniLM-L6-v2-onnx")
        self.assertEqual(t.trocear("unaffable"), ["una", "##ffa", "##ble"])

    def test_en_potion_los_ids_del_post_procesador_NO_son_sus_tokens_especiales(self):
        """El hallazgo, fijado: si algún día el proveedor lo arregla, esta
        prueba se pondrá roja y habrá que volver a mirar `tokens_especiales`."""
        ruta = CACHE / "potion-base-8M" / "tokenizer.json"
        if not ruta.is_file():
            self.skipTest("potion-base-8M no está descargado")
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        vocab = datos["model"]["vocab"]
        inverso = {v: k for k, v in vocab.items()}
        puestos = datos["post_processor"]["special_tokens"]
        self.assertEqual(puestos["[CLS]"]["ids"], [101])
        self.assertEqual(puestos["[SEP]"]["ids"], [102])
        self.assertEqual(vocab["[CLS]"], 2)
        self.assertEqual(vocab["[SEP]"], 3)
        self.assertEqual(inverso[101], "×")
        self.assertEqual(inverso[102], "ß")


if __name__ == "__main__":
    unittest.main()
