"""El motivo de una restricción NO cumplida dice la verdad en los dos idiomas (04-10).

Hallado auditando 122-C1.1: el texto era UNO para todos los operadores («{campo} no alcanza el
mínimo exigido»), así que un `max` incumplido decía «mínimo», y el valor viajaba en bruto, con
punto decimal también en castellano («medido: 0.857142857142857»). El motivo llega tal cual a la
razón del estudio que lee la persona.
"""

from __future__ import annotations

import unittest

from matrixai.estudio.esquemas import EvaluationResult, Restriccion
from matrixai.estudio.metricas import ValorDeMetrica
from matrixai.estudio.seleccion import seleccionar


def _ev(pipeline_digest, **metricas):
    return EvaluationResult(
        evaluation_id=f"eval-{pipeline_digest}", pipeline_digest=pipeline_digest,
        split_plan_digest="split-1", evaluated_role="development", evidence="development_estimate",
        metrics=tuple(ValorDeMetrica(metric_id=k, formula_version="v1", value=v)
                      for k, v in metricas.items()))


def _motivo_del_rechazado(restriccion, *, necesita_red=None, **metricas_del_rechazado):
    """El motivo con que se descarta a `malo` por `restriccion`; `bueno` la cumple."""
    metricas_del_bueno = {"sensitivity": 0.9, "latencia_ms": 10.0, "calibrado": 1.0}
    evaluaciones = {"bueno": _ev("p1", **metricas_del_bueno),
                    "malo": _ev("p2", **{**metricas_del_bueno, **metricas_del_rechazado})}
    r = seleccionar(evaluaciones, restricciones=[restriccion], metric_id_calidad="sensitivity",
                    decision_id="d1", split_plan_digest="split-1", necesita_red=necesita_red)
    (rechazado,) = [x for x in r.rejected if x["candidate"] == "malo"]
    return rechazado["reason"]


class ElOperadorDiceLoQueSeIncumplio(unittest.TestCase):
    def test_un_minimo_dice_minimo_con_el_umbral_y_la_coma_en_castellano(self):
        m = _motivo_del_rechazado(Restriccion(clave="sensitivity", operador="min", valor=0.875),
                                  sensitivity=6 / 7)
        self.assertEqual(m["es"], "sensitivity no alcanza el mínimo exigido, 0,875 (medido: 0,8571)")
        self.assertEqual(m["en"], "sensitivity does not reach the required minimum of 0.875 "
                                  "(measured: 0.8571)")

    def test_un_maximo_NO_dice_minimo(self):
        m = _motivo_del_rechazado(Restriccion(clave="latencia_ms", operador="max", valor=50),
                                  latencia_ms=73.21)
        self.assertEqual(m["es"], "latencia_ms pasa del máximo permitido, 50 (medido: 73,21)")
        self.assertEqual(m["en"], "latencia_ms exceeds the allowed maximum of 50 (measured: 73.21)")
        self.assertNotIn("mínimo", m["es"])
        self.assertNotIn("minimum", m["en"])

    def test_un_igual_dice_que_no_es_lo_exigido(self):
        m = _motivo_del_rechazado(Restriccion(clave="sensitivity", operador="equal", valor=1.0),
                                  sensitivity=0.5)
        self.assertEqual(m["es"], "sensitivity no es lo exigido, 1 (medido: 0,5)")
        self.assertEqual(m["en"], "sensitivity is not the required value, 1 (measured: 0.5)")

    def test_un_booleano_dice_si_y_no_en_su_idioma(self):
        """Una métrica es siempre un número finito (`ValorDeMetrica`): el booleano llega como 0/1."""
        m = _motivo_del_rechazado(Restriccion(clave="calibrado", operador="boolean", valor=True),
                                  calibrado=0.0)
        self.assertEqual(m["es"], "calibrado no es lo exigido, sí (medido: no)")
        self.assertEqual(m["en"], "calibrado is not the required value, yes (measured: no)")

    def test_solo_local_dice_lo_medido_en_su_idioma_y_no_un_repr_de_python(self):
        m = _motivo_del_rechazado(Restriccion(clave="solo_local", operador="boolean", valor=True),
                                  necesita_red={"bueno": False, "malo": True})
        self.assertEqual(m["es"], "solo_local no es lo exigido, sí (medido: no)")
        self.assertEqual(m["en"], "solo_local is not the required value, yes (measured: no)")
        self.assertNotIn("necesita_red=", m["es"] + m["en"])


class LaCifraNoEngañaAlRedondear(unittest.TestCase):
    def test_un_medido_que_redondea_al_umbral_lleva_los_decimales_que_los_separan(self):
        """Con cuatro decimales, 0,849996 frente a un mínimo de 0,85 se leería «no alcanza el
        mínimo, 0,85 (medido: 0,85)»: se escriben los decimales que hacen falta para verlos
        distintos."""
        m = _motivo_del_rechazado(Restriccion(clave="sensitivity", operador="min", valor=0.85),
                                  sensitivity=0.849996)
        self.assertEqual(m["es"], "sensitivity no alcanza el mínimo exigido, 0,85 (medido: 0,849996)")
        self.assertEqual(m["en"], "sensitivity does not reach the required minimum of 0.85 "
                                  "(measured: 0.849996)")



class CadaOperadorTieneSuFrase(unittest.TestCase):
    def test_todos_los_operadores_del_vocabulario_tienen_motivo_en_el_catalogo(self):
        """Un operador nuevo sin su frase reventaría con `KeyError` al descartar a un candidato,
        en mitad de un estudio."""
        from matrixai.estudio.seleccion import _MOTIVO_DE_LA_RESTRICCION_POR_OPERADOR
        from matrixai.estudio.textos import MOTIVOS
        from matrixai.estudio.vocabulario import OPERADORES_DE_RESTRICCION

        self.assertEqual(set(_MOTIVO_DE_LA_RESTRICCION_POR_OPERADOR), set(OPERADORES_DE_RESTRICCION))
        for clave in _MOTIVO_DE_LA_RESTRICCION_POR_OPERADOR.values():
            self.assertIn(clave, MOTIVOS)

if __name__ == "__main__":
    unittest.main()
