# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — el proveedor ejecuta lo que el paquete declara, y declara lo que
no puede.

Dos mitades, y las dos hacen falta:

  · el proveedor compone el vector como dice el paquete descargado, y si el
    paquete no lo dice de una forma reproducible, no se construye;
  · un idioma que no se cubre se DECLARA (107, invariante 7) en vez de
    devolver un vector en silencio, que es lo que pasa por defecto: un modelo
    inglés no falla con texto en español, contesta algo que no significa nada.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.text.embeddings import cobertura
from matrixai.text.embeddings.catalogo import por_id
from matrixai.text.embeddings.descarga import raiz_cache
from matrixai.text.embeddings.proveedor import (
    TOPE_BOLSA,
    TOPE_SECUENCIA,
    IdiomaNoCubierto,
    ProveedorError,
    elegir_longitud,
    leer_composicion,
)

CACHE = raiz_cache()


class TestCuandoElPaqueteSeContradiceASiMismo(unittest.TestCase):
    """`all-MiniLM-L6-v2` declara TRES longitudes distintas en tres ficheros
    suyos: 256, 128 y 512. Elegir en silencio produce un vector que no es el
    que su ficha promete, y eso no se ve. Medido: con la de `tokenizer.json`
    (128) el coseno contra `sentence-transformers` baja a 0,999 en los textos
    largos; con la de `sentence_bert_config.json` (256), 1,000."""

    TRES = [
        ("sentence_bert_config.json:max_seq_length", 256),
        ("tokenizer.json:truncation.max_length", 128),
        ("tokenizer_config.json:model_max_length", 512),
    ]

    def test_manda_la_que_usa_la_implementacion_oficial(self):
        valor, motivo = elegir_longitud(self.TRES, tope=TOPE_SECUENCIA)
        self.assertEqual(valor, 256)
        self.assertIn("sentence_bert_config.json", motivo)

    def test_y_el_motivo_DICE_que_habia_tres(self):
        # Media verdad tranquilizadora: quedarse con una y callar las otras
        # dos deja a quien lea el catálogo sin saber que el paquete es
        # ambiguo.
        _, motivo = elegir_longitud(self.TRES, tope=TOPE_SECUENCIA)
        self.assertIn("3 longitudes distintas", motivo)
        for donde, valor in self.TRES:
            self.assertIn(f"{donde}={valor}", motivo)

    def test_cuando_solo_hay_una_no_se_habla_de_varias(self):
        _, motivo = elegir_longitud([("tokenizer.json:truncation.max_length", 128)], tope=TOPE_SECUENCIA)
        self.assertNotIn("distintas", motivo)

    def test_sin_ninguna_declarada_se_usa_el_tope_y_se_dice(self):
        valor, motivo = elegir_longitud([], tope=TOPE_SECUENCIA)
        self.assertEqual(valor, TOPE_SECUENCIA)
        self.assertIn("no declara ninguna", motivo)


class TestNoTruncarEsNoTruncar(unittest.TestCase):
    """`model2vec` declara `seq_length: 1000000`, que es «no truncar». En un
    grafo de bolsa el coste es lineal y truncar no protege de nada: solo
    cambia el vector. En uno de secuencia la atención es cuadrática y el tope
    sí hace falta."""

    DECLARA_UN_MILLON = [("config.json:seq_length", 1_000_000)]

    def test_en_bolsa_no_se_acota(self):
        valor, motivo = elegir_longitud(self.DECLARA_UN_MILLON, tope=TOPE_BOLSA)
        self.assertEqual(valor, 1_000_000)
        self.assertNotIn("acota", motivo)

    def test_en_secuencia_se_acota_Y_SE_DICE(self):
        valor, motivo = elegir_longitud(self.DECLARA_UN_MILLON, tope=TOPE_SECUENCIA)
        self.assertEqual(valor, TOPE_SECUENCIA)
        self.assertIn("acota", motivo)
        self.assertIn("1000000", motivo)

    def test_seq_length_manda_sobre_model_max_length(self):
        # `tokenizer_config.json` trae el 512 heredado del BERT del que se
        # destiló; truncar ahí daría otro vector para los textos largos.
        valor, _ = elegir_longitud(
            [("config.json:seq_length", 1_000_000), ("tokenizer_config.json:model_max_length", 512)],
            tope=TOPE_BOLSA,
        )
        self.assertEqual(valor, 1_000_000)

    def test_los_dos_topes_son_distintos_a_proposito(self):
        self.assertGreater(TOPE_BOLSA, TOPE_SECUENCIA)


class TestLeerLaComposicionDelPaquete(unittest.TestCase):
    def _paquete(self, tmp: Path, modules: list, config: dict | None = None, pooling: dict | None = None) -> Path:
        (tmp / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
        if config is not None:
            (tmp / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if pooling is not None:
            (tmp / "1_Pooling").mkdir(exist_ok=True)
            (tmp / "1_Pooling" / "config.json").write_text(json.dumps(pooling), encoding="utf-8")
        return tmp

    def test_lee_la_normalizacion_del_paquete_no_de_una_tabla_a_mano(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            self._paquete(
                tmp,
                [
                    {"idx": 0, "type": "sentence_transformers.models.Transformer", "path": ""},
                    {"idx": 1, "type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
                    {"idx": 2, "type": "sentence_transformers.models.Normalize", "path": "2_Normalize"},
                ],
                pooling={"word_embedding_dimension": 384, "pooling_mode_mean_tokens": True},
            )
            c = leer_composicion(tmp)
        self.assertTrue(c.normaliza)
        self.assertEqual(c.pooling, "media")
        self.assertEqual(c.dimension_declarada, 384)

    def test_sin_Normalize_no_normaliza(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            self._paquete(
                tmp,
                [
                    {"idx": 0, "type": "sentence_transformers.models.Transformer", "path": ""},
                    {"idx": 1, "type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
                ],
                pooling={"word_embedding_dimension": 384, "pooling_mode_mean_tokens": True},
            )
            c = leer_composicion(tmp)
        self.assertFalse(c.normaliza)

    def test_otro_modo_de_pooling_se_rechaza_en_vez_de_hacer_la_media(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            self._paquete(
                tmp,
                [
                    {"idx": 0, "type": "sentence_transformers.models.Transformer", "path": ""},
                    {"idx": 1, "type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
                ],
                pooling={"word_embedding_dimension": 384, "pooling_mode_cls_token": True},
            )
            with self.assertRaises(ProveedorError) as ctx:
                leer_composicion(tmp)
        self.assertIn("pooling", str(ctx.exception))

    def test_si_los_dos_ficheros_se_contradicen_se_para(self):
        with TemporaryDirectory() as t:
            tmp = Path(t)
            self._paquete(
                tmp,
                [
                    {"idx": 0, "type": "sentence_transformers.models.StaticEmbedding", "path": "."},
                    {"idx": 1, "type": "sentence_transformers.models.Normalize", "path": "1_Normalize"},
                ],
                config={"hidden_dim": 256, "normalize": False, "seq_length": 1000000},
            )
            with self.assertRaises(ProveedorError) as ctx:
                leer_composicion(tmp)
        self.assertIn("no se elige por cuenta propia", str(ctx.exception))

    def test_sin_modules_json_no_se_adivina(self):
        with TemporaryDirectory() as t:
            with self.assertRaises(ProveedorError) as ctx:
                leer_composicion(Path(t))
        self.assertIn("no hay dónde leer", str(ctx.exception))


class TestLosUmbralesDeCobertura(unittest.TestCase):
    def test_estan_ordenados_y_por_encima_del_azar(self):
        self.assertGreater(cobertura.UMBRAL_CUBIERTO, cobertura.UMBRAL_LIMITADO)
        self.assertGreater(cobertura.UMBRAL_LIMITADO, 0.5)

    def test_el_veredicto_sale_del_numero(self):
        self.assertEqual(cobertura.estado_por_auc(0.97), cobertura.CUBIERTO)
        self.assertEqual(cobertura.estado_por_auc(cobertura.UMBRAL_CUBIERTO), cobertura.CUBIERTO)
        self.assertEqual(cobertura.estado_por_auc(0.81), cobertura.LIMITADO)
        self.assertEqual(cobertura.estado_por_auc(0.63), cobertura.NO_CUBIERTO)
        self.assertEqual(cobertura.estado_por_auc(0.50), cobertura.NO_CUBIERTO)


class TestUnIdiomaNoCubiertoSeDeclara(unittest.TestCase):
    """Las dos mitades: que lo cubierto salga cubierto, y que lo no cubierto
    salga no cubierto. Un aserto negativo solo lo pasaría un artefacto vacío."""

    def _con(self, auc_es: float, auc_en: float) -> Path:
        tmp = Path(self.enterContext(TemporaryDirectory())) / "a.json"
        tmp.write_text(
            json.dumps(
                {
                    "proveedores": {
                        "p": {
                            "medicion": {
                                "coherencia_por_documento": {"es": {"auc": auc_es}, "en": {"auc": auc_en}},
                                "tokenizacion": {"es": {"tokens_por_palabra": 1.9}, "en": {"tokens_por_palabra": 1.2}},
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return tmp

    def test_un_idioma_cubierto_se_declara_cubierto(self):
        v = cobertura.veredicto("p", "en", artefacto=self._con(0.63, 0.94))
        self.assertEqual(v.estado, cobertura.CUBIERTO)
        self.assertTrue(v.es_utilizable())
        self.assertIn("0.94", v.frase)

    def test_un_idioma_no_cubierto_lo_dice_Y_avisa_de_que_dara_vector_igual(self):
        v = cobertura.veredicto("p", "es", artefacto=self._con(0.63, 0.94))
        self.assertEqual(v.estado, cobertura.NO_CUBIERTO)
        self.assertFalse(v.es_utilizable())
        self.assertIn("NO cubre", v.frase)
        self.assertIn("Dará un vector", v.frase)

    def test_exigir_cubierto_para_en_vez_de_seguir(self):
        a = self._con(0.63, 0.94)
        with self.assertRaises(IdiomaNoCubierto):
            cobertura.exigir_cubierto("p", "es", artefacto=a)
        self.assertEqual(cobertura.exigir_cubierto("p", "en", artefacto=a).estado, cobertura.CUBIERTO)

    def test_lo_no_medido_dice_no_medido_y_no_se_lo_inventa(self):
        v = cobertura.veredicto("no-existe", "es", artefacto=self._con(0.9, 0.9))
        self.assertEqual(v.estado, cobertura.NO_MEDIDO)
        self.assertIsNone(v.auc)

    def test_una_fila_que_existe_pero_sin_medir_tampoco_se_aprueba(self):
        """La otra rama: el proveedor está en el catálogo (fijado, con su
        licencia) pero no se ha ejecutado. Sin número, no hay veredicto."""
        tmp = Path(self.enterContext(TemporaryDirectory())) / "b.json"
        tmp.write_text(
            json.dumps({"proveedores": {"p": {"licencia": {"spdx": "MIT"}, "candidato": {"no_descargado_porque": "pesa 537 MB"}}}}),
            encoding="utf-8",
        )
        v = cobertura.veredicto("p", "es", artefacto=tmp)
        self.assertEqual(v.estado, cobertura.NO_MEDIDO)
        self.assertFalse(v.es_utilizable())
        self.assertIn("pesa 537 MB", v.frase)
        with self.assertRaises(IdiomaNoCubierto):
            cobertura.exigir_cubierto("p", "es", artefacto=tmp)

    def test_no_medido_NO_es_lo_mismo_que_no_cubierto(self):
        # Un valor ausente no es un cero: «no lo hemos medido» y «lo hemos
        # medido y no vale» son dos respuestas distintas.
        self.assertNotEqual(cobertura.NO_MEDIDO, cobertura.NO_CUBIERTO)

    def test_la_ficha_del_autor_no_entra_en_el_veredicto(self):
        """`multilingual-e5-small` dice cubrir español; como no se ha medido, el
        veredicto es «no medido», no «cubierto».

        Hasta el 2026-09-17 el ejemplo era `potion-multilingual-128M`. Ese día se
        midió, y su propia ficha («101 idiomas») tampoco decidió: el veredicto en
        es salió «limitado» (AUC 0,872 frente a 0,908 en en). La intención de la
        prueba es la misma; cambia el candidato que sigue sin medir."""
        c = por_id("multilingual-e5-small-onnx-int8")
        self.assertIn("español", c.idiomas_segun_su_autor)
        self.assertEqual(cobertura.veredicto(c.id, "es").estado, cobertura.NO_MEDIDO)

    def test_y_la_de_potion_multilingual_TAMPOCO_ahora_que_esta_medido(self):
        """La otra mitad: medido, su veredicto sale de la medida y no de su
        ficha — que dice cubrir español y sale «limitado», no «cubierto»."""
        c = por_id("potion-multilingual-128M")
        self.assertIn("español", c.idiomas_segun_su_autor)
        self.assertNotEqual(cobertura.veredicto(c.id, "es").estado, cobertura.NO_MEDIDO)
        self.assertNotEqual(cobertura.veredicto(c.id, "es").estado, cobertura.CUBIERTO)


class TestSobreLosPesosDeVerdad(unittest.TestCase):
    def _proveedor(self, cid: str):
        from matrixai.text.embeddings.proveedor import ProveedorDeEmbeddings

        if not (CACHE / cid / "tokenizer.json").is_file():
            self.skipTest(f"{cid} no está descargado (los pesos no van en el repositorio)")
        return ProveedorDeEmbeddings.cargar(por_id(cid))

    def test_un_texto_que_no_deja_tokens_NO_devuelve_un_vector_de_ceros(self):
        """Un valor ausente no es un cero: un vector de ceros pasaría por un
        texto legítimo en todo lo que venga después."""
        p = self._proveedor("potion-base-8M")
        with self.assertRaises(ProveedorError) as ctx:
            p.codificar(["texto normal", "   "])
        self.assertIn("no dejan ningún token", str(ctx.exception))

    def test_describe_declara_que_es_opaco(self):
        d = self._proveedor("potion-base-8M").describe()
        self.assertTrue(d["opaco"])
        self.assertIn("no auditados", d["opacidad"])
        self.assertEqual(len(d["revision"]), 40)

    def test_la_dimension_sale_del_grafo_no_de_una_constante(self):
        import numpy as np

        p = self._proveedor("potion-base-8M")
        v = p.codificar(["hola mundo", "otro texto"])
        self.assertEqual(v.shape, (2, p.dimension))
        self.assertEqual(v.dtype, np.float32)

    def test_si_el_paquete_promete_norma_1_los_vectores_la_tienen(self):
        import numpy as np

        p = self._proveedor("potion-base-8M")
        self.assertTrue(p.composicion.normaliza)
        normas = np.linalg.norm(p.codificar(["hola mundo", "el gato duerme"]), axis=1)
        self.assertTrue(np.allclose(normas, 1.0, atol=1e-5), normas)

    def test_codificar_un_texto_suelto_se_rechaza_en_vez_de_tokenizar_letras(self):
        p = self._proveedor("potion-base-8M")
        with self.assertRaises(ProveedorError):
            p.codificar("esto es un texto, no una lista")

    def test_el_lote_no_cambia_el_vector(self):
        import numpy as np

        p = self._proveedor("potion-base-8M")
        textos = [f"frase numero {i} con longitud variable " + "palabra " * (i % 7) for i in range(20)]
        self.assertTrue(np.allclose(p.codificar(textos, lote=3), p.codificar(textos, lote=20), atol=1e-6))

    def test_un_tokenizador_unigram_se_niega_sin_dependencia(self):
        from matrixai.text.embeddings.proveedor import ProveedorDeEmbeddings

        cid = "paraphrase-multilingual-MiniLM-L12-v2-onnx-int8"
        if not (CACHE / cid / "tokenizer.json").is_file():
            self.skipTest(f"{cid} no está descargado")
        with self.assertRaises(ProveedorError) as ctx:
            ProveedorDeEmbeddings.cargar(por_id(cid))
        self.assertIn("unigram_sentencepiece", str(ctx.exception))
        self.assertIn("tokenizers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
