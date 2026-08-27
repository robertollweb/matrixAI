"""UNA RECETA QUE NO SE PUEDE LEER SE DICE (medido el 2026-08-24).

Preparando la galería contra el paquete **publicado**, sondeando el caso
clínico:

    alto: (edad > 75)                           -> synthetic_random · NADA
    alto: edad > 75 OR dias > 14 AND diags > 6  -> synthetic_random · NADA
    alto: dias > 14 AND diags > 6               -> synthetic_random · «no discriminó» (bien)
    alto: edad > 75 OR reingreso_previo > 0.5   -> synthetic_domain  · funciona

O sea: los paréntesis y el `AND` dentro de un `OR` no se entienden, y en vez de
decirlo el producto generaba etiquetas ALEATORIAS y sacaba su aviso genérico
—«datos sintéticos aleatorios»— que manda a mirar el modo de generación y no la
receta.

Y el core ya lo sabía: `parse_domain_rules('alto: (edad > 75)')` devuelve **cero
reglas** y `validate` contesta `['no rules parsed', …]`. Lo que fallaba es que
`recipe_errors` se calculaba en `playground.py` **y no se usaba nunca después**.
El hueco de siempre, dentro de un solo fichero.

Es lo que el contrato 80 vino a cerrar, un piso más arriba: quien escriba una
receta con paréntesis se lleva su 50 % de exactitud **y un aviso que le hace
buscar donde no es**.
"""
from __future__ import annotations

import unittest

from matrixai.playground_api import generate_synthetic_dataset
from matrixai.training.domain_rules import parse_domain_rules

_MXAI = (
    "PROJECT R\n\nVECTOR Input[2]\n  edad: Scalar\n  ingresos: Scalar\nEND\n\n"
    "NETWORK C\n  INPUT Input\n  LAYER Dense units=4 activation=relu\n"
    "  LAYER Dense units=2 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
    "GRAPH\n  Input -> C\nEND\n"
)
_MXTRAIN = (
    "MODEL r.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT Input FROM COLUMNS [edad, ingresos]\n"
    "  TARGET predicted_class: Label[alto, bajo]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION C\n  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE C.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)


def _generar(receta: str) -> dict:
    return generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 7, "coherent", False,
                                      recipe_text=receta)


class LaRecetaIlegibleSeCuentaTest(unittest.TestCase):
    def test_con_parentesis_el_producto_DICE_que_no_pudo_leerla(self):
        r = _generar("alto: (edad > 75)\nDEFAULT: bajo")
        self.assertEqual(r["label_origin"], "synthetic_random")
        self.assertIn("no rules parsed", r["recipe_errors"])
        aviso = r["recipe_warning"]
        self.assertIn("no se ha podido leer", aviso)
        # Y dice QUÉ admite el lenguaje, que es lo que permite arreglarlo.
        self.assertIn("DEFAULT", aviso)
        self.assertIn("paréntesis", aviso)

    def test_el_aviso_generico_ya_NO_manda_a_mirar_el_modo_de_generacion(self):
        """Mandar a buscar la avería donde no está cuesta más que callarse."""
        r = _generar("alto: (edad > 75)\nDEFAULT: bajo")
        self.assertIn("tu receta no se pudo leer", r["signal_warning"])
        self.assertNotIn("reglas de dominio del LLM", r["signal_warning"])

    def test_sin_receta_el_aviso_generico_sigue_como_estaba(self):
        r = generate_synthetic_dataset(_MXAI, _MXTRAIN, 60, 7, "coherent", False)
        self.assertNotIn("recipe_warning", r)
        self.assertIn("reglas de dominio del LLM", r["signal_warning"])

    def test_una_receta_QUE_SE_LEE_no_saca_ningun_aviso_de_receta(self):
        r = _generar("alto: edad > 0.6\nDEFAULT: bajo")
        self.assertEqual(r["label_origin"], "synthetic_domain")
        self.assertIsNone(r.get("recipe_warning"))
        self.assertIsNone(r.get("recipe_errors"))

    def test_los_codigos_del_validador_NO_se_meten_en_la_frase(self):
        """Vienen en inglés del validador: concatenarlos dentro dejaría media
        frase en otro idioma. Viajan aparte, para quien lea por programa."""
        r = _generar("alto: (edad > 75)\nDEFAULT: bajo")
        self.assertNotIn("no rules parsed", r["recipe_warning"])


class ElCoreYaSabiaLoQuePasabaTest(unittest.TestCase):
    def test_los_parentesis_no_dejan_ni_una_regla(self):
        dr = parse_domain_rules("alto: (edad > 75)\nDEFAULT: bajo")
        self.assertEqual(len(dr.rules), 0)
        self.assertIn("no rules parsed", dr.validate({"edad": None}, ["alto", "bajo"]))

    def test_y_una_regla_simple_si(self):
        dr = parse_domain_rules("alto: edad > 75\nDEFAULT: bajo")
        self.assertEqual(len(dr.rules), 1)
        self.assertEqual(dr.validate({"edad": None}, ["alto", "bajo"]), [])


if __name__ == "__main__":
    unittest.main()


# ── LA RECETA QUE SE LEE A MEDIAS (hallazgo 12, 2026-08-25) ─────────────────

_MXAI3 = (
    "PROJECT T\n\nVECTOR E[2]\n  urgencia: Scalar\n  impacto: Scalar\nEND\n\n"
    "NETWORK N\n  INPUT E\n  LAYER Dense units=6 activation=relu\n"
    "  LAYER Dense units=3 activation=softmax\n"
    "  OUTPUT prioridad: ProbabilityMap[alta, media, baja]\nEND\n\n"
    "GRAPH\n  E -> N\nEND\n"
)
_MXTRAIN3 = (
    "MODEL t.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT E FROM COLUMNS [urgencia, impacto]\n"
    "  TARGET prioridad: Label[alta, media, baja]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION N\n  TARGET prioridad\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE N.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)
_RECETA_A_MEDIAS = "alta: (urgencia > 0.7)\nbaja: impacto < 0.25\nDEFAULT: media"


class LaRecetaQueSeLeeAMediasTest(unittest.TestCase):
    """El caso PEOR del hallazgo 5, y salió de una medición equivocada de un
    agente que él mismo corrigió: `validate` solo protesta cuando NO queda
    ninguna regla. Con una de dos, el dataset sale `synthetic_domain`, sin un
    solo aviso, y la línea ilegible desaparece — cambiando el reparto de clases
    sin que su autor se entere.
    """

    def _generar(self):
        return generate_synthetic_dataset(_MXAI3, _MXTRAIN3, 40, 2108, "coherent",
                                          False, recipe_text=_RECETA_A_MEDIAS)

    def test_la_linea_que_se_cayo_se_ENUMERA(self):
        r = self._generar()
        self.assertEqual(r["recipe_dropped_lines"], ["alta: (urgencia > 0.7)"])

    def test_y_el_aviso_dice_lo_que_eso_significa(self):
        aviso = self._generar()["recipe_partial_warning"]
        # No basta con «hubo un problema»: tiene que decir la CONSECUENCIA.
        self.assertIn("reparto de clases", aviso)
        self.assertIn("paréntesis", aviso)

    def test_el_dataset_sigue_siendo_de_dominio_porque_lo_es(self):
        """Media receta leída NO es un error: lo que se leyó se aplicó. Decir
        que el dataset es aleatorio sería mentir en la otra dirección."""
        self.assertEqual(self._generar()["label_origin"], "synthetic_domain")

    def test_una_receta_ENTERA_no_declara_ninguna_caida(self):
        r = generate_synthetic_dataset(_MXAI3, _MXTRAIN3, 40, 2108, "coherent", False,
                                       recipe_text="alta: urgencia > 0.7\nbaja: impacto < 0.25\nDEFAULT: media")
        self.assertIsNone(r.get("recipe_dropped_lines"))
        self.assertIsNone(r.get("recipe_partial_warning"))

    def test_las_directivas_NO_cuentan_como_lineas_caidas(self):
        """`DEFAULT`, `NOISE` y `BALANCE` no son clases: contarlas como caídas
        avisaría de un problema que no existe en toda receta bien escrita."""
        from matrixai.training.domain_rules import lineas_que_no_se_leyeron, parse_domain_rules
        texto = "alta: urgencia > 0.7\nDEFAULT: media\nNOISE: 0.1\nBALANCE: alta=0.3\n# un comentario"
        dr = parse_domain_rules(texto)
        self.assertEqual(lineas_que_no_se_leyeron(texto, dr.rules), [])


# ── LAS CONDICIONES QUE NO PUEDEN DECIDIR NADA (hallazgo 11, 2026-08-25) ─────

class CondicionesMuertasTest(unittest.TestCase):
    """Salió al estrenar `--recipe`: con un modelo que no declara rangos el
    generador muestrea en 0-1, y `alto: edad > 75` se lee perfectamente y **no
    se cumple jamás**. El CSV salió con el 100 % de las filas siguiendo la
    receta… por la OTRA condición. Media receta muerta y un dataset con aspecto
    de bueno — y no lo decía nadie: no es un error (la receta es válida) ni un
    «no discriminó» (discriminaba por lo demás).
    """

    def test_un_umbral_fuera_del_dominio_no_se_cumple_nunca(self):
        from matrixai.training.domain_rules import condiciones_imposibles, parse_domain_rules
        dr = parse_domain_rules("alto: edad > 75\nDEFAULT: bajo")
        muertas = condiciones_imposibles(dr.rules)
        self.assertEqual(len(muertas), 1)
        self.assertIn("nunca se cumple", muertas[0])
        self.assertIn("edad va de 0 a 1", muertas[0])

    def test_y_uno_que_lo_abarca_todo_se_cumple_SIEMPRE(self):
        """Tan muerta es una condición que nunca se cumple como una que se
        cumple siempre: ninguna de las dos decide nada."""
        from matrixai.training.domain_rules import condiciones_imposibles, parse_domain_rules
        dr = parse_domain_rules("alto: edad < 200\nDEFAULT: bajo")
        muertas = condiciones_imposibles(dr.rules, {"edad": (18, 100)})
        self.assertIn("se cumple siempre", muertas[0])

    def test_con_el_dominio_DECLARADO_la_misma_condicion_esta_viva(self):
        from matrixai.training.domain_rules import condiciones_imposibles, parse_domain_rules
        dr = parse_domain_rules("alto: edad > 75\nDEFAULT: bajo")
        self.assertEqual(condiciones_imposibles(dr.rules, {"edad": (18, 100)}), [])

    def test_un_campo_sin_dominio_QUE_NO_SE_PUEDE_LEER_no_se_juzga(self):
        """Acusar sin saber es peor que callar."""
        from matrixai.training.domain_rules import condiciones_imposibles, parse_domain_rules
        dr = parse_domain_rules("alto: edad > 75\nDEFAULT: bajo")
        self.assertEqual(condiciones_imposibles(dr.rules, {"edad": ("a", "b")}), [])

    def test_el_producto_lo_DICE_al_generar(self):
        r = _generar("alto: edad > 75\nDEFAULT: bajo")
        self.assertIn("recipe_dead_conditions", r)
        self.assertIn("nunca se cumple", r["recipe_dead_conditions"][0])
        # Y dice por qué el dataset parece correcto, que es lo que engaña.
        self.assertIn("parece correcto", r["recipe_dead_warning"])

    def test_una_receta_con_todo_vivo_no_saca_el_aviso(self):
        r = _generar("alto: edad > 0.6\nDEFAULT: bajo")
        self.assertIsNone(r.get("recipe_dead_conditions"))
        self.assertIsNone(r.get("recipe_dead_warning"))
