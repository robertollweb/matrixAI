"""LA RECETA, ALCANZABLE DESDE EL CLI (85-C5a, 2026-08-25).

Medido contra el paquete publicado el 2026-08-24: `matrixai generate-dataset`
**no tenía `--recipe`**, así que quien hacía `pip install matrixai-core` no
podía generar un dataset con receta — justo el caso con el que se iba a
escribir el artículo público. La lógica existía, pero solo dentro de
`playground.py`, o sea solo para el Studio.

**No se copió al CLI**: se extrajo a `resolver_receta` y la usan los dos. Dos
sitios decidiendo qué es una receta válida acabarían divergiendo, y en este
proyecto eso ya ha costado catorce veces.

Y una diferencia de comportamiento **deliberada**: en el Studio una receta
ilegible avisa y sigue (hay una pantalla donde leer el aviso); en el CLI
**falla**, porque quien escribe `--recipe` la ha pedido explícitamente y
escribirle un CSV de ruido en el disco sería darle por bueno un dataset que no
sirve.
"""
from __future__ import annotations

import unittest

from matrixai.cli import _etiquetas_del_training, _target_es_continuo
from matrixai.training.domain_rules import resolver_receta
from matrixai.training.parser import parse_training_text

_MXTRAIN_CLASES = (
    "MODEL m.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT E FROM COLUMNS [edad, ingresos]\n"
    "  TARGET predicted_class: Label[alto, bajo]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION N\n  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE N.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)
_MXTRAIN_CONTINUO = _MXTRAIN_CLASES.replace(
    "TARGET predicted_class: Label[alto, bajo]", "TARGET valor: Scalar[0, 100]").replace(
    "TYPE cross_entropy", "TYPE mse").replace("TARGET predicted_class\nEND", "TARGET valor\nEND")


class ElTipoDelObjetivoSeLEE_noSeAdivinaTest(unittest.TestCase):
    """`target.type` NO es una cadena: es un `TypeSpec` con su `name` y sus
    `parameters`. Tratarlo como texto —lo que hizo la primera versión— daba
    «continuo» para TODO y rechazaba una receta de clases perfectamente válida
    diciendo que el objetivo era continuo. Se vio conduciendo el comando."""

    def test_un_objetivo_de_clases_NO_es_continuo(self):
        self.assertFalse(_target_es_continuo(parse_training_text(_MXTRAIN_CLASES)))

    def test_un_objetivo_escalar_SI(self):
        self.assertTrue(_target_es_continuo(parse_training_text(_MXTRAIN_CONTINUO)))

    def test_las_clases_salen_de_los_parametros_del_tipo(self):
        self.assertEqual(_etiquetas_del_training(parse_training_text(_MXTRAIN_CLASES)),
                         ["alto", "bajo"])

    def test_un_objetivo_continuo_no_tiene_clases_que_devolver(self):
        self.assertEqual(_etiquetas_del_training(parse_training_text(_MXTRAIN_CONTINUO)), [])


class LaFuncionEXTRAIDAEsLaMismaParaLosDosTest(unittest.TestCase):
    def test_una_receta_de_clases_da_reglas(self):
        reglas, regresion, texto, errores, caidas = resolver_receta(
            "alto: edad > 0.6\nDEFAULT: bajo",
            is_regression=False, typeable=["edad", "ingresos"],
            labels=["alto", "bajo"], field_ranges={})
        self.assertEqual(errores, [])
        # Y ninguna línea se quedó por el camino (hallazgo 12).
        self.assertEqual(caidas, [])
        self.assertIsNotNone(reglas)
        self.assertIsNone(regresion)
        self.assertIn("alto", texto)

    def test_una_receta_ILEGIBLE_no_da_reglas_a_medias(self):
        reglas, regresion, texto, errores, _caidas = resolver_receta(
            "alto: (edad > 0.6)\nDEFAULT: bajo",
            is_regression=False, typeable=["edad"], labels=["alto", "bajo"],
            field_ranges={})
        self.assertIn("no rules parsed", errores)
        self.assertIsNone(reglas)
        self.assertIsNone(regresion)
        self.assertEqual(texto, "")

    def test_reglas_de_clases_sobre_un_objetivo_CONTINUO_se_rechazan(self):
        """No se ignora en silencio: el dataset saldría aleatorio con aspecto
        de bueno."""
        _, _, _, errores, _ = resolver_receta(
            "alto: edad > 0.6\nDEFAULT: bajo",
            is_regression=True, typeable=["edad"], labels=[], field_ranges={})
        self.assertTrue(any("expression" in e for e in errores), errores)

    def test_sin_receta_no_hay_nada_que_resolver(self):
        # La quinta posición son las líneas que se cayeron (hallazgo 12): sin
        # receta no hay ninguna, y decir otra cosa sería inventar un problema.
        self.assertEqual(resolver_receta("", is_regression=False, typeable=["a"],
                                         labels=["x", "y"], field_ranges={}),
                         (None, None, "", [], []))


if __name__ == "__main__":
    unittest.main()


class LosRangosDeclaradosMandanTest(unittest.TestCase):
    """EL AVISO DE CONDICIONES MUERTAS ACUSABA A UN MODELO CORRECTO.

    Medido el 2026-08-25 preparando la galería: con `edad: Scalar[18, 100]` en
    el `.mxai`, el generador muestreaba 23,7 y 53,0 —unidades reales— y el aviso
    nuevo decía «edad va de 0 a 1: nunca se cumple». Un falso positivo que manda
    a arreglar lo que ya estaba bien, que es exactamente el defecto que ese
    aviso venía a quitar.
    """

    def _programa(self, con_rangos: bool):
        from matrixai.parser.parser import parse_text
        rango = "[18, 100]" if con_rangos else ""
        return parse_text(
            f"PROJECT P\n\nVECTOR E[1]\n  edad: Scalar{rango}\nEND\n\n"
            "NETWORK N\n  INPUT E\n  LAYER Dense units=2 activation=softmax\n"
            "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
            "GRAPH\n  E -> N\nEND\n")

    def test_los_rangos_del_mxai_llegan_al_aviso(self):
        from matrixai.cli import _rangos_declarados
        self.assertEqual(_rangos_declarados(self._programa(True)), {"edad": (18.0, 100.0)})

    def test_un_campo_sin_rango_no_entra_en_el_mapa(self):
        """Quien lo lea usará su valor por defecto —que es la misma suposición—
        en vez de uno inventado aquí."""
        from matrixai.cli import _rangos_declarados
        self.assertEqual(_rangos_declarados(self._programa(False)), {})

    def test_con_rangos_declarados_la_condicion_esta_VIVA(self):
        from matrixai.cli import _rangos_declarados
        from matrixai.training.domain_rules import condiciones_imposibles, parse_domain_rules
        dr = parse_domain_rules("alto: edad > 75\nDEFAULT: bajo")
        self.assertEqual(
            condiciones_imposibles(dr.rules, _rangos_declarados(self._programa(True))), [])
        # Y sin declararlos, sigue muerta: el aviso no se ha apagado, se ha
        # informado.
        self.assertEqual(
            len(condiciones_imposibles(dr.rules, _rangos_declarados(self._programa(False)))), 1)
