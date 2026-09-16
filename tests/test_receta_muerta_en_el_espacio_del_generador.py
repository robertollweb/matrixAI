"""EL AVISO DE RECETA MUERTA ACUSABA, EN EL STUDIO, A UNA RECETA CORRECTA.

Medido el 2026-09-16 con el paquete real de `/casos/clinico`: el modelo declara
`edad: Scalar[18, 100]`, la receta es `alto: edad > 75 OR reingreso_previo >
0.5`, y la respuesta traía `recipe_dead_warning` («hay condiciones de tu receta
que no pueden decidir nada»). Era falso: la edad salía 18,1–99,9 y **65 filas
eran `alto` SOLO por la edad**.

Es el falso positivo que `test_c85_c5a_receta_desde_el_cli.py` ya había
cerrado… en la CLI. El playground llamaba a `condiciones_imposibles` sin
dominios, así que todo se juzgaba contra [0, 1]. El hueco de siempre: el core
tenía el dato y un llamante no lo usaba.

LA TRAMPA DEL ARREGLO, que es por lo que este fichero prueba los CUATRO
espacios y no solo el del caso: el generador muestrea cada campo en un sitio
distinto según de dónde venga su rango —
  · con rango del LLM o de la persona (`field_ranges`): en [0, 1], y la receta
    se normaliza con ESE rango;
  · con rango declarado en el `.mxai` y sin lo anterior: en unidades reales;
  · sin ninguno: en [0, 1], con la receta tal cual.
Un arreglo que mire solo los declarados acusa al que pasa rangos; uno que mire
las reglas ya normalizadas contra los declarados, también. Cada caso de abajo
comprueba las DOS mitades: lo que dice el aviso y lo que hizo el generador.
"""
from __future__ import annotations

import csv
import io
import unittest

from matrixai.playground_api import generate_synthetic_dataset

_MXAI = (
    "PROJECT R\n\nVECTOR Input[3]\n"
    "  edad: Scalar[18, 100]\n"
    "  reingreso_previo: Scalar[0, 1]\n"
    "  ingresos: Scalar\n"
    "END\n\n"
    "NETWORK C\n  INPUT Input\n  LAYER Dense units=4 activation=relu\n"
    "  LAYER Dense units=2 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
    "GRAPH\n  Input -> C\nEND\n"
)
_MXTRAIN = (
    "MODEL r.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT Input FROM COLUMNS [edad, reingreso_previo, ingresos]\n"
    "  TARGET predicted_class: Label[alto, bajo]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION C\n  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE C.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)


def _generar(receta: str, rangos: dict | None = None) -> dict:
    return generate_synthetic_dataset(
        _MXAI, _MXTRAIN, 300, 7, "coherent", False,
        field_ranges_override=rangos, recipe_text=receta)


def _filas(r: dict) -> list[dict]:
    return list(csv.DictReader(io.StringIO(r["csv_text"])))


class TestUnRangoDeclaradoEnElModelo(unittest.TestCase):
    """El caso de `/casos/clinico`: `edad: Scalar[18, 100]`, sin rangos
    pasados. El generador muestrea en años y la receta se evalúa en años."""

    def test_la_receta_del_caso_clinico_NO_saca_el_aviso(self):
        r = _generar("alto: edad > 75 OR reingreso_previo > 0.5\nDEFAULT: bajo")
        self.assertEqual(r["label_origin"], "synthetic_domain")
        self.assertIsNone(r.get("recipe_dead_conditions"))
        self.assertIsNone(r.get("recipe_dead_warning"))

    def test_y_la_condicion_de_la_edad_DECIDE_de_verdad(self):
        """La otra mitad: sin esto, callar el aviso lo pasaría también un
        generador que hubiera dejado la edad muerta de verdad."""
        filas = _filas(_generar("alto: edad > 75 OR reingreso_previo > 0.5\nDEFAULT: bajo"))
        solo_por_la_edad = [f for f in filas
                            if float(f["edad"]) > 75 and float(f["reingreso_previo"]) <= 0.5]
        self.assertTrue(solo_por_la_edad)
        self.assertTrue(all(f["predicted_class"] == "alto" for f in solo_por_la_edad))

    def test_un_umbral_fuera_del_rango_declarado_SIGUE_avisando(self):
        """El aviso no se ha apagado, se ha informado — y habla en años."""
        r = _generar("alto: edad > 150 OR reingreso_previo > 0.5\nDEFAULT: bajo")
        muertas = r.get("recipe_dead_conditions") or []
        self.assertEqual(len(muertas), 1)
        self.assertIn("edad > 150: nunca se cumple", muertas[0])
        self.assertIn("va de 18 a 100", muertas[0])
        self.assertIn("parece correcto", r["recipe_dead_warning"])
        self.assertFalse(any(float(f["edad"]) > 150 for f in _filas(r)))


class TestUnCampoSinRangoNinguno(unittest.TestCase):
    """`ingresos: Scalar`, sin declarar ni pasar: se muestrea en [0, 1]."""

    def test_un_umbral_en_unidades_reales_esta_muerto_y_se_dice(self):
        r = _generar("alto: ingresos > 75 OR reingreso_previo > 0.5\nDEFAULT: bajo")
        muertas = r.get("recipe_dead_conditions") or []
        self.assertEqual(len(muertas), 1)
        self.assertIn("ingresos > 75: nunca se cumple", muertas[0])
        self.assertIn("va de 0 a 1", muertas[0])
        self.assertFalse(any(float(f["ingresos"]) > 1 for f in _filas(r)))

    def test_y_uno_dentro_de_0_1_esta_vivo(self):
        r = _generar("alto: ingresos > 0.6\nDEFAULT: bajo")
        self.assertIsNone(r.get("recipe_dead_conditions"))


class TestUnRangoPasadoPorLaPersona(unittest.TestCase):
    """LA TRAMPA. Con `field_ranges_override` la receta se normaliza con ese
    rango y el generador muestrea en [0, 1]; el CSV vuelve a las unidades
    reales al final. Aquí el rango pasado (20–90) PISA al declarado (18–100)."""

    _RANGOS = {"edad": (20.0, 90.0)}

    def test_un_umbral_dentro_del_rango_pasado_NO_avisa(self):
        """Juzgar la regla ya normalizada (0,786) contra el rango declarado —o
        contra el pasado— diría «se cumple siempre»."""
        r = _generar("alto: edad > 75 OR reingreso_previo > 0.5\nDEFAULT: bajo", self._RANGOS)
        self.assertEqual(r["label_origin"], "synthetic_domain")
        self.assertIsNone(r.get("recipe_dead_conditions"))
        filas = _filas(r)
        solo_por_la_edad = [f for f in filas
                            if float(f["edad"]) > 75 and float(f["reingreso_previo"]) <= 0.5]
        self.assertTrue(solo_por_la_edad)
        self.assertTrue(all(f["predicted_class"] == "alto" for f in solo_por_la_edad))

    def test_uno_fuera_del_rango_pasado_avisa_con_LO_QUE_ESCRIBIO_la_persona(self):
        """Dentro del declarado (95 < 100) pero fuera del pasado (95 > 90): manda
        el pasado, que es el que usa el generador. Y el aviso habla en las
        unidades de la receta, no en las normalizadas —«edad > 1,07: va de 0 a
        1» no lo reconoce quien escribió 95—."""
        r = _generar("alto: edad > 95 OR reingreso_previo > 0.5\nDEFAULT: bajo", self._RANGOS)
        muertas = r.get("recipe_dead_conditions") or []
        self.assertEqual(len(muertas), 1)
        self.assertIn("edad > 95: nunca se cumple", muertas[0])
        self.assertIn("va de 20 a 90", muertas[0])
        self.assertFalse(any(float(f["edad"]) > 95 for f in _filas(r)))


class TestLaCliLoDiceConducida(unittest.TestCase):
    """La CLI llevaba el arreglo desde el 2026-08-25 y NADA lo sostenía: sus
    pruebas miraban la función de rangos, no que `generate-dataset` se la
    pasara al aviso. Quitarle los rangos a la llamada dejaba la suite verde."""

    def _cli(self, receta: str) -> str:
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            Path(d, "m.mxai").write_text(_MXAI, encoding="utf-8")
            Path(d, "m.mxtrain").write_text(_MXTRAIN.replace("MODEL r.mxai", "MODEL m.mxai"),
                                            encoding="utf-8")
            Path(d, "receta.txt").write_text(receta, encoding="utf-8")
            r = subprocess.run(
                [sys.executable, "-m", "matrixai", "generate-dataset", "m.mxai",
                 "--training", "m.mxtrain", "--rows", "60", "--mode", "coherent",
                 "--recipe", "receta.txt", "-o", "datos"],
                cwd=d, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stderr

    def test_con_el_rango_declarado_la_edad_no_se_acusa(self):
        self.assertNotIn("nunca se cumple", self._cli("alto: edad > 75\nDEFAULT: bajo"))

    def test_y_fuera_de_el_si_en_anos(self):
        self.assertIn("Warning: edad > 150: nunca se cumple — edad va de 18 a 100",
                      self._cli("alto: edad > 150\nDEFAULT: bajo"))


class TestLosDosCaminosUsanElMismoMapa(unittest.TestCase):
    """La CLI y el Studio juzgan con la MISMA función. Copiarla al playground
    habría sido el segundo sitio declarando lo mismo."""

    def test_la_cli_no_tiene_su_propia_copia(self):
        import matrixai.cli as cli
        self.assertFalse(hasattr(cli, "_rangos_declarados"))

    def test_sin_rangos_pasados_el_mapa_son_los_declarados(self):
        from matrixai.parser import parse_text
        from matrixai.training.domain_rules import dominios_de_muestreo
        self.assertEqual(dominios_de_muestreo(parse_text(_MXAI)),
                         {"edad": (18.0, 100.0), "reingreso_previo": (0.0, 1.0)})

    def test_un_rango_pasado_pisa_al_declarado(self):
        from matrixai.parser import parse_text
        from matrixai.training.domain_rules import dominios_de_muestreo
        self.assertEqual(dominios_de_muestreo(parse_text(_MXAI), {"edad": (20, 90)})["edad"],
                         (20.0, 90.0))


if __name__ == "__main__":
    unittest.main()
