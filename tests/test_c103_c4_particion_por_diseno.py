# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C4 — particiones por diseño.

Criterio de terminado: «Fixtures de pacientes repetidos, fechas
desordenadas y grupos con tiempo cumplen sus invariantes. Se verifica
intersección de conjuntos y cronología, no solo hashes diferentes. La
misma semilla/protocolo reproduce particiones. Una cohorte externa se
identifica y no participa en búsqueda.»
"""
from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta

from matrixai.estudio import Horizonte
from matrixai.training.particion_por_diseno import proponer_particion
from matrixai.training.particion_por_diseno_textos import IDIOMAS, MOTIVOS, huecos_de, motivo


class TestElCatalogoHablaLosDosIdiomas(unittest.TestCase):
    def test_toda_clave_tiene_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            for idioma in IDIOMAS:
                self.assertIn(idioma, textos, f"{clave} sin {idioma}")

    def test_los_huecos_son_los_mismos_en_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            self.assertEqual(huecos_de(textos["es"]), huecos_de(textos["en"]),
                             f"{clave}: los huecos no coinciden")

    def test_clave_desconocida_se_rechaza(self):
        with self.assertRaises(KeyError):
            motivo("no_existe")


def _pacientes_repetidos(seed=0, n_pacientes=30):
    rng = random.Random(seed)
    filas = []
    i = 0
    for paciente in range(n_pacientes):
        for _visita in range(rng.randint(1, 4)):
            filas.append({"id": f"obs{i}", "paciente": f"p{paciente}",
                         "y": "si" if paciente % 3 == 0 else "no"})
            i += 1
    return filas


class GruposTest(unittest.TestCase):
    def test_pacientes_repetidos_nunca_se_parten_entre_roles(self):
        filas = _pacientes_repetidos()
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="groups",
                               seed=42, unit_id_field="paciente", objetivo="y", folds=5, repeats=2)
        self.assertTrue(r.es_viable)
        roles_por_paciente: dict[str, set] = {}
        for fila in filas:
            roles_por_paciente.setdefault(fila["paciente"], set()).add(r.plan.assignments[fila["id"]])
        mezclados = {p: roles for p, roles in roles_por_paciente.items() if len(roles) > 1}
        self.assertEqual(mezclados, {})

    def test_pacientes_repetidos_nunca_se_parten_entre_pliegues(self):
        filas = _pacientes_repetidos()
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="groups",
                               seed=42, unit_id_field="paciente", objetivo="y", folds=5, repeats=2)
        for pliegue in r.pliegues.pliegues:
            entrena_p = {f["paciente"] for f in filas if f["id"] in pliegue.entrena}
            valida_p = {f["paciente"] for f in filas if f["id"] in pliegue.valida}
            self.assertEqual(entrena_p & valida_p, set(),
                            f"pliegue {pliegue.repeticion}/{pliegue.pliegue} mezcla pacientes")

    def test_grupos_insuficientes_es_bloqueo(self):
        filas = [{"id": f"o{i}", "g": "g1"} for i in range(10)]
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="groups",
                               seed=1, unit_id_field="g")
        self.assertFalse(r.es_viable)
        self.assertEqual([b.clave for b in r.bloqueos], ["grupos_insuficientes"])

    def test_pliegues_se_reducen_por_grupos_insuficientes(self):
        filas = [{"id": f"o{g}_{i}", "g": f"g{g}"} for g in range(4) for i in range(20)]
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="groups",
                               seed=1, unit_id_field="g", folds=5, test_fraction=0.25)
        self.assertTrue(r.es_viable)
        self.assertTrue(any(l.clave == "pliegues_reducidos_por_grupos" for l in r.limites))
        folds_reales = max(p.pliegue for p in r.pliegues.pliegues) + 1
        self.assertLess(folds_reales, 5)


class TemporalTest(unittest.TestCase):
    def _fixture(self, seed=1, n=200):
        rng = random.Random(seed)
        base = datetime(2020, 1, 1)
        filas = [{"id": f"obs{i}", "fecha": base + timedelta(days=rng.randint(0, 365))}
                for i in range(n)]
        rng.shuffle(filas)  # el CSV llega desordenado a propósito
        return filas

    def test_fechas_desordenadas_se_ordenan_y_respetan_cronologia(self):
        """El texto literal: «una fecha desordenada sigue siendo fecha» --
        el orden de llegada de las filas no puede cambiar el resultado."""
        filas = self._fixture()
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal",
                               seed=1, time_column="fecha", test_fraction=0.2)
        self.assertTrue(r.es_viable)
        por_id = {f["id"]: f for f in filas}
        fechas_dev = [por_id[o]["fecha"] for o, rol in r.plan.assignments.items() if rol == "development"]
        fechas_test = [por_id[o]["fecha"] for o, rol in r.plan.assignments.items() if rol == "test"]
        self.assertLess(max(fechas_dev), min(fechas_test))

    def test_la_fraccion_de_prueba_pedida_se_respeta_aproximadamente(self):
        """El corte cronológico tiene que caer CERCA del percentil pedido,
        no en un punto arbitrario que igual separa antes/después
        correctamente pero deja la prueba con un tamaño cualquiera."""
        filas = self._fixture(n=500)
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal",
                               seed=1, time_column="fecha", test_fraction=0.2)
        n_test = sum(1 for rol in r.plan.assignments.values() if rol == "test")
        self.assertAlmostEqual(n_test / 500, 0.2, delta=0.05)

    def test_temporal_respeta_el_hueco_declarado(self):
        filas = self._fixture()
        gap = Horizonte(magnitud=10, unidad="days")
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal",
                               seed=1, time_column="fecha", test_fraction=0.2, gap=gap)
        self.assertTrue(r.es_viable)
        por_id = {f["id"]: f for f in filas}
        fechas_dev = [por_id[o]["fecha"] for o, rol in r.plan.assignments.items() if rol == "development"]
        fechas_test = [por_id[o]["fecha"] for o, rol in r.plan.assignments.items() if rol == "test"]
        distancia = (min(fechas_test) - max(fechas_dev)).total_seconds()
        self.assertGreaterEqual(distancia, 10 * 86400 - 1)

    def test_cronologia_no_se_pudo_separar_con_hueco_excesivo(self):
        base = datetime(2020, 1, 1)
        filas = [{"id": f"o{i}", "fecha": base + timedelta(days=i)} for i in range(50)]
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal",
                               seed=1, time_column="fecha", test_fraction=0.2,
                               gap=Horizonte(magnitud=1000, unidad="days"))
        self.assertFalse(r.es_viable)
        self.assertEqual([b.clave for b in r.bloqueos], ["cronologia_no_se_pudo_separar"])

    def test_misma_semilla_reproduce_la_particion(self):
        filas = self._fixture()
        r1 = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal",
                                seed=7, time_column="fecha", test_fraction=0.2)
        r2 = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="temporal",
                                seed=7, time_column="fecha", test_fraction=0.2)
        self.assertEqual(r1.plan.digest(), r2.plan.digest())


class GruposConTiempoTest(unittest.TestCase):
    def test_groups_and_time_no_mezcla_unidades_entre_roles(self):
        rng = random.Random(2)
        base = datetime(2020, 1, 1)
        filas = []
        i = 0
        for centro in range(20):
            fecha_centro = base + timedelta(days=rng.randint(0, 300))
            for visita in range(rng.randint(1, 5)):
                filas.append({"id": f"obs{i}", "centro": f"c{centro}",
                             "fecha": fecha_centro + timedelta(days=visita)})
                i += 1
        r = proponer_particion(filas, plan_id="p", observation_id_field="id",
                               split_type="groups_and_time", seed=3, unit_id_field="centro",
                               time_column="fecha", test_fraction=0.25,
                               gap=Horizonte(magnitud=2, unidad="days"))
        self.assertTrue(r.es_viable)
        roles_por_centro: dict[str, set] = {}
        for fila in filas:
            rol = r.plan.assignments.get(fila["id"])
            if rol is None:
                continue  # embargada por el hueco, no participa en ningun rol
            roles_por_centro.setdefault(fila["centro"], set()).add(rol)
        mezclados = {c: roles for c, roles in roles_por_centro.items() if len(roles) > 1}
        self.assertEqual(mezclados, {})


class CohorteExternaTest(unittest.TestCase):
    def test_cohorte_externa_se_identifica_y_no_participa_en_busqueda(self):
        filas = _pacientes_repetidos()
        externos = [f["id"] for f in filas[:5]]
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="iid",
                               seed=4, ids_cohorte_externa=externos)
        for obs in externos:
            self.assertEqual(r.plan.assignments[obs], "external_test")
        en_pliegues = {obs for p in r.pliegues.pliegues for obs in (*p.entrena, *p.valida)}
        self.assertEqual(en_pliegues & set(externos), set())


class EventosInsuficientesTest(unittest.TestCase):
    def test_pliegues_se_reducen_por_eventos_insuficientes(self):
        filas = [{"id": f"o{i}", "y": "si" if i < 2 else "no"} for i in range(100)]
        r = proponer_particion(filas, plan_id="p", observation_id_field="id", split_type="iid",
                               seed=1, objetivo="y", folds=5, test_fraction=0.1)
        self.assertTrue(any(l.clave == "pliegues_reducidos_por_eventos" for l in r.limites))


if __name__ == "__main__":
    unittest.main()
