"""EL DESCONOCIDO Y EL RELLENO SE LEEN DEL `tokenizer.json` (107-C1, 2026-09-17).

El instrumento de `scripts/medir_catalogo_embeddings.py` llevaba `<unk>` y
`<pad>` escritos a mano. `potion-multilingual-128M` no tiene ninguno de los dos:
sus tokens son `[PAD]` y `[UNK]`, y su `tokenizer_config.json` dice lo contrario.
Con `<unk>` a mano la cobertura de idioma contaba desconocidos comparando con un
token que nunca sale —cobertura inflada, sin error— y la medición ni arrancaba
porque exigía un `<pad>` que un modelo estático no usa.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "matrixai" / "embeddings"


def _guion():
    spec = importlib.util.spec_from_file_location(
        "medir_catalogo_embeddings", RAIZ / "scripts" / "medir_catalogo_embeddings.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _escribir(datos: dict) -> Path:
    carpeta = Path(tempfile.mkdtemp())
    ruta = carpeta / "tokenizer.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


class TokensEspecialesDelTokenizadorTest(unittest.TestCase):
    def test_el_desconocido_es_el_del_vocabulario_no_el_de_la_configuracion(self):
        ruta = _escribir({"model": {"type": "Unigram", "unk_id": 1,
                                    "vocab": [["[PAD]", 0.0], ["[UNK]", 0.0], [",", -1.0]]},
                          "padding": None})
        self.assertEqual(_guion().tokens_especiales_de(ruta), ("[UNK]", None))

    def test_con_padding_declarado_el_relleno_es_el_suyo(self):
        ruta = _escribir({"model": {"type": "Unigram", "unk_id": 3,
                                    "vocab": [["<s>", 0.0], ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0]]},
                          "padding": {"pad_token": "<pad>", "pad_id": 1}})
        self.assertEqual(_guion().tokens_especiales_de(ruta), ("<unk>", "<pad>"))

    def test_sin_desconocido_en_el_vocabulario_no_se_inventa_uno(self):
        ruta = _escribir({"model": {"type": "Unigram", "unk_id": 9, "vocab": [["a", 0.0]]}})
        with self.assertRaises(SystemExit):
            _guion().tokens_especiales_de(ruta)

    def test_sobre_los_tokenizadores_descargados_de_verdad(self):
        casos = {"potion-multilingual-128M": ("[UNK]", None),
                 "paraphrase-multilingual-MiniLM-L12-v2-onnx-int8": ("<unk>", "<pad>")}
        guion = _guion()
        medidos = 0
        for modelo, esperado in casos.items():
            ruta = CACHE / modelo / "tokenizer.json"
            if not ruta.is_file():
                continue
            with self.subTest(modelo=modelo):
                self.assertEqual(guion.tokens_especiales_de(ruta), esperado)
                medidos += 1
        if medidos == 0:
            self.skipTest("ningún tokenizador descargado (los pesos no van en el repositorio)")


if __name__ == "__main__":
    unittest.main()
