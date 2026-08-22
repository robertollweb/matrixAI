"""CONTRATO 81-C1 — canonicalización JCS (RFC 8785).

Decisión de Roberto (2026-08-20): **DSSE sobre bytes JSON canonicalizados
con JCS**. Sin una forma canónica, «firmar» no significa nada: el mismo
contenido serializado de dos maneras da dos firmas, y una firma buena
parece mala.

**Por qué no se reutiliza `canonical_json` del contrato 82.** Existe, pero
NO es JCS: usa `ensure_ascii=True` (JCS exige UTF-8 real) y ordena por
code points de Python (JCS ordena por unidades UTF-16). Y **no se puede
cambiar**: sus digests están dentro de manifiestos ya emitidos, así que
tocarlo invalidaría paquetes que hoy verifican. Son dos usos distintos y
conviven.

> **P23-R-0016.** La firma se verifica sobre los **bytes originales**. El
> verificador NO parsea y vuelve a serializar antes de comprobarla: ahí es
> donde se cuelan las diferencias que invalidan una firma buena o validan
> una mala.
"""

import json
import unittest

from matrixai.pipelines.canonical import jcs_canonical, jcs_bytes


class ElOrdenEsElDeJCSTest(unittest.TestCase):
    def test_las_claves_van_ordenadas(self):
        self.assertEqual(jcs_canonical({"b": 1, "a": 2}), '{"a":2,"b":1}')

    def test_ordena_por_unidades_UTF16_no_por_code_points(self):
        """La diferencia con `sorted()` de Python, y es real: los pares
        subrogados (emoji, U+1xxxx) ordenan DESPUÉS de U+E000-U+FFFF en
        UTF-16 y ANTES por code point. Un verificador que ordene distinto
        rechaza firmas buenas."""
        datos = {"\U0001F600": 1, "": 2}
        salida = jcs_canonical(datos)
        # En UTF-16, el emoji empieza por 0xD83D, que es MENOR que 0xE000.
        self.assertTrue(salida.index("\U0001F600") < salida.index(""), salida)

    def test_sin_espacios(self):
        self.assertEqual(jcs_canonical({"a": [1, 2]}), '{"a":[1,2]}')


class LosCaracteresNoSeESCAPANTest(unittest.TestCase):
    def test_el_texto_no_ascii_viaja_tal_cual(self):
        """JCS exige UTF-8: `ensure_ascii=True` daría `\\u00f1` y otro
        digest. Es justo la diferencia con `canonical_json` del 82."""
        self.assertEqual(jcs_canonical({"k": "ñ"}), '{"k":"ñ"}')

    def test_los_escapes_obligatorios_SI_se_ponen(self):
        salida = jcs_canonical({"k": 'a"b\\c\nd'})
        self.assertIn('\\"', salida)
        self.assertIn("\\\\", salida)
        self.assertIn("\\n", salida)

    def test_los_controles_van_en_forma_corta_cuando_existe(self):
        # RFC 8785: \b \f \n \r \t tienen forma corta; el resto \u00XX.
        self.assertIn("\\t", jcs_canonical({"k": "\t"}))
        self.assertIn("\\u0001", jcs_canonical({"k": "\x01"}))


class LosNumerosSiguenAECMAScriptTest(unittest.TestCase):
    def test_un_entero_no_lleva_punto(self):
        self.assertEqual(jcs_canonical({"n": 1}), '{"n":1}')

    def test_un_float_entero_tampoco(self):
        """`json.dumps(1.0)` da `1.0`; ECMAScript da `1`. Dos digests
        distintos para el mismo número."""
        self.assertEqual(jcs_canonical({"n": 1.0}), '{"n":1}')

    def test_los_grandes_van_en_exponencial_como_ECMAScript(self):
        self.assertEqual(jcs_canonical({"n": 1e30}), '{"n":1e+30}')

    def test_lo_que_no_es_un_numero_JSON_se_RECHAZA(self):
        """`NaN` e `Infinity` no existen en JSON. `json.dumps` los escribe
        igual y produce algo que ningún otro verificador puede leer: aquí
        se corta, en vez de firmar basura."""
        for imposible in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(n=imposible):
                with self.assertRaises(ValueError):
                    jcs_canonical({"n": imposible})


class LosBytesSonLoQueSEFIRMATest(unittest.TestCase):
    def test_devuelve_UTF8(self):
        self.assertEqual(jcs_bytes({"k": "ñ"}), '{"k":"ñ"}'.encode("utf-8"))

    def test_el_mismo_contenido_da_LOS_MISMOS_bytes(self):
        uno = jcs_bytes({"b": 1, "a": {"y": 2, "x": 1}})
        otro = jcs_bytes({"a": {"x": 1, "y": 2}, "b": 1})
        self.assertEqual(uno, otro)

    def test_y_contenidos_distintos_dan_bytes_distintos(self):
        """Si esto fallara, la canonicalización estaría borrando
        diferencias y dos payloads distintos compartirían firma."""
        self.assertNotEqual(jcs_bytes({"a": 1}), jcs_bytes({"a": 2}))


if __name__ == "__main__":
    unittest.main()
