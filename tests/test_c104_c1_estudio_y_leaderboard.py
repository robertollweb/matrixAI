# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C1 — el estudio y su presupuesto total, y el leaderboard.

Criterio de terminado: «suma de reservas/ejecuciones coherente; aumentar
número de motores no aumenta sin aviso el presupuesto total. Un estudio sin
restricciones usa valores por omisión visibles; una métrica sin definición
no se acepta.»
"""
from __future__ import annotations

import unittest

from matrixai.estudio import EsquemaInvalido, ValorDeMetrica
from matrixai.estudio.competicion import (
    EntradaDeLeaderboard,
    EstudioSpec,
    Leaderboard,
    ReservaDeTiempo,
    tiempo_de_busqueda_por_motor,
)
from matrixai.estudio.metricas import MetricaDesconocida


def _reserva(**kw):
    base = dict(busqueda=600.0, ajuste_final=120.0, calibracion=60.0, evaluacion=60.0)
    base.update(kw)
    return ReservaDeTiempo(**base)


def _estudio(**kw):
    base = dict(estudio_id="e1", problem_id="p1",
               motores_permitidos=("baseline", "lightgbm", "sklearn.lineal"),
               presupuesto_total_segundos=900.0, reserva=_reserva(),
               objetivo_medible="auroc", motor_de_reserva="baseline")
    base.update(kw)
    return EstudioSpec(**base)


def _metrica(valor=0.8, **kw):
    base = dict(metric_id="auroc", formula_version="v1", value=valor,
               uncertainty={"ci_low": 0.7, "ci_high": 0.9})
    base.update(kw)
    return ValorDeMetrica(**base)


def _entrada(**kw):
    base = dict(candidate="c1", engine="lightgbm", engine_version="1.0.0",
               config_efectiva={"n_estimators": 100}, metrica_de_seleccion=_metrica(),
               estado="completed", fase="seleccion")
    base.update(kw)
    return EntradaDeLeaderboard(**base)


# ---------------------------------------------------------------------------
# ReservaDeTiempo
# ---------------------------------------------------------------------------

class ReservaDeTiempoTest(unittest.TestCase):
    def test_total_suma_las_cuatro_fases(self):
        r = _reserva()
        self.assertEqual(r.total(), 840.0)

    def test_negativa_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            _reserva(busqueda=-1.0)

    def test_roundtrip_json(self):
        r = _reserva()
        r2 = ReservaDeTiempo.desde_json(r.a_json())
        self.assertEqual(r.total(), r2.total())


# ---------------------------------------------------------------------------
# EstudioSpec — el criterio de terminado literal
# ---------------------------------------------------------------------------

class EstudioSpecTest(unittest.TestCase):
    def test_reserva_que_supera_el_presupuesto_se_rechaza(self):
        """«Suma de reservas/ejecuciones coherente»: una reserva que promete
        más segundos de los que el estudio tiene se rechaza AL CONSTRUIR,
        no se descubre a mitad de estudio."""
        with self.assertRaises(EsquemaInvalido) as contexto:
            _estudio(presupuesto_total_segundos=100.0)
        self.assertEqual(contexto.exception.clave, "reserva_supera_presupuesto")

    def test_reserva_igual_al_presupuesto_se_acepta(self):
        # 840.0 exactos, el límite no es un error
        _estudio(presupuesto_total_segundos=840.0)  # no debe lanzar

    def test_anadir_un_motor_no_cambia_el_presupuesto_total(self):
        """El invariante que más cuesta dejar pasar: más motores reparten
        la búsqueda, nunca multiplican el total."""
        con_tres = _estudio(motores_permitidos=("baseline", "lightgbm", "sklearn.lineal"))
        con_cuatro = _estudio(motores_permitidos=("baseline", "lightgbm", "sklearn.lineal",
                                                  "matrixai.dense.torch_cpu"))
        self.assertEqual(con_tres.presupuesto_total_segundos, con_cuatro.presupuesto_total_segundos)
        self.assertGreater(tiempo_de_busqueda_por_motor(con_tres),
                           tiempo_de_busqueda_por_motor(con_cuatro))

    def test_tiempo_de_busqueda_por_motor_es_la_division_explicita(self):
        e = _estudio()
        self.assertAlmostEqual(tiempo_de_busqueda_por_motor(e), 600.0 / 3)

    def test_metrica_sin_definicion_no_se_acepta(self):
        """«Una métrica sin definición no se acepta» — reutiliza el
        registro CERRADO de 105-C1, no un vocabulario propio de este
        módulo."""
        with self.assertRaises(MetricaDesconocida):
            _estudio(objetivo_medible="esto_no_es_una_metrica_real")

    def test_motor_de_reserva_tiene_que_estar_permitido(self):
        with self.assertRaises(EsquemaInvalido) as contexto:
            _estudio(motor_de_reserva="motor_inventado")
        self.assertEqual(contexto.exception.clave, "motor_no_permitido")

    def test_sin_motores_permitidos_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            _estudio(motores_permitidos=())

    def test_modo_desconocido_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            _estudio(modo="ultra_rapido")

    def test_valores_por_omision_son_visibles_en_el_json(self):
        """«Un estudio sin restricciones usa valores por omisión VISIBLES»:
        `modo` y `recursos_declarados` no puestos explícitamente aparecen
        en `a_json()` de todos modos, con su valor por omisión."""
        e = _estudio()  # sin declarar modo ni recursos_declarados
        j = e.a_json()
        self.assertIn("modo", j)
        self.assertEqual(j["modo"], "completo")
        self.assertIn("recursos_declarados", j)
        self.assertEqual(j["recursos_declarados"], {})

    def test_digest_cambia_si_cambia_el_presupuesto(self):
        e1 = _estudio(presupuesto_total_segundos=900.0)
        e2 = _estudio(presupuesto_total_segundos=901.0, reserva=_reserva())
        self.assertNotEqual(e1.digest(), e2.digest())

    def test_digest_estable_para_el_mismo_contenido(self):
        self.assertEqual(_estudio().digest(), _estudio().digest())

    def test_roundtrip_json(self):
        e = _estudio()
        e2 = EstudioSpec.desde_json(e.a_json())
        self.assertEqual(e.digest(), e2.digest())


# ---------------------------------------------------------------------------
# Leaderboard — selección y evaluación final nunca se mezclan
# ---------------------------------------------------------------------------

class LeaderboardTest(unittest.TestCase):
    def test_de_seleccion_y_de_evaluacion_final_no_se_mezclan(self):
        lb = Leaderboard(estudio_id="e1")
        lb = lb.con_entrada(_entrada(candidate="c1", fase="seleccion"))
        lb = lb.con_entrada(_entrada(candidate="c2", fase="seleccion"))
        lb = lb.con_entrada(_entrada(candidate="c1", fase="evaluacion_final"))
        self.assertEqual(len(lb.de_seleccion()), 2)
        self.assertEqual(len(lb.de_evaluacion_final()), 1)
        self.assertEqual(len(lb.entradas), 3)
        # ninguna entrada de seleccion aparece en evaluacion_final ni viceversa
        candidatos_seleccion = {e.candidate for e in lb.de_seleccion()}
        candidatos_evaluacion = {e.candidate for e in lb.de_evaluacion_final()}
        for e in lb.de_seleccion():
            self.assertEqual(e.fase, "seleccion")
        for e in lb.de_evaluacion_final():
            self.assertEqual(e.fase, "evaluacion_final")

    def test_con_entrada_no_muta_el_original(self):
        lb1 = Leaderboard(estudio_id="e1")
        lb2 = lb1.con_entrada(_entrada())
        self.assertEqual(len(lb1.entradas), 0)
        self.assertEqual(len(lb2.entradas), 1)

    def test_fase_desconocida_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            _entrada(fase="a_medio_camino")

    def test_metrica_de_seleccion_tiene_que_ser_valordemetrica(self):
        with self.assertRaises(EsquemaInvalido):
            _entrada(metrica_de_seleccion={"value": 0.8})

    def test_recurso_negativo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            _entrada(cpu_seconds=-1.0)

    def test_metrica_de_seleccion_conserva_incertidumbre(self):
        """Las métricas del leaderboard llevan incertidumbre/alcance —
        nunca un número suelto (criterio del corte, texto literal)."""
        entrada = _entrada(metrica_de_seleccion=_metrica(
            uncertainty={"ci_low": 0.75, "ci_high": 0.85, "method": "bootstrap"}))
        j = entrada.a_json()
        self.assertIsNotNone(j["metrica_de_seleccion"]["uncertainty"])
        self.assertEqual(j["metrica_de_seleccion"]["uncertainty"]["method"], "bootstrap")

    def test_roundtrip_json(self):
        lb = Leaderboard(estudio_id="e1").con_entrada(_entrada())
        lb2 = Leaderboard.desde_json(lb.a_json())
        self.assertEqual(lb.a_json(), lb2.a_json())


if __name__ == "__main__":
    unittest.main()
