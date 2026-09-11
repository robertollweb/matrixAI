# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C1 — el protocolo de la Fase 0 exploratoria, registrado con hash.

Criterio de terminado literal: «un cálculo a partir del protocolo produce
número de ejecuciones y cota de coste consistente con motores, pliegues,
repeticiones y presupuesto. La decisión sobre tareas, recursos y márgenes
consta antes de medir. No hay variantes contradictorias del presupuesto en
el informe.»

Dos partes: el ESQUEMA (`protocolo.py`, validado con fixtures de rechazo) y
el REGISTRO REAL (`protocolo_exploratorio.json`, 40 datasets de OpenML —
CC18 + AMLB, descargados y hasheados en vivo el 2026-09-06, no inventados)
verificado contra los mínimos de cobertura que el anexo C del documento 100
dejó escritos.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmarks.fase0.protocolo import (
    aplicar_regla_de_cierre,
    CosteDeLaPasada,
    DatasetRegistrado,
    DisenoDeParticion,
    Motor,
    PresupuestoPorCubo,
    ProtocoloError,
    ProtocoloExploratorio,
    ReglaDeCierre,
    calcular_coste,
)

_RUTA_PROTOCOLO_REAL = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "protocolo_exploratorio.json"


def _dataset(**kw) -> DatasetRegistrado:
    base = dict(data_id=1, nombre="x", fuente="cc18", version=1, sha256_arff="a" * 64,
               columna_objetivo="y", tarea="binary_classification", cubo_de_tamano="pequeno",
               n_filas=1000, n_columnas=10, tiene_faltantes=False, alta_cardinalidad=False,
               desbalanceado=False, solo_numericas=True, licencia="Public")
    base.update(kw)
    return DatasetRegistrado(**base)


def _protocolo_minimo(**kw) -> ProtocoloExploratorio:
    base = dict(
        version_protocolo="test.v1", fecha_registro="2026-09-06",
        datasets=(_dataset(),),
        motores=(Motor(id="dummy", configuraciones=1), Motor(id="logreg", configuraciones=2)),
        particion=DisenoDeParticion(),
        presupuesto=PresupuestoPorCubo(minutos_por_cubo={"pequeno": 2.0, "mediano": 5.0, "grande": 10.0}),
        regla_de_cierre=ReglaDeCierre(
            puntos=2.0, fraccion_minima=0.8,
            metrica_por_tarea={"binary_classification": "AUROC", "multiclass_classification": "accuracy",
                               "regression": "R2"},
            definicion_de_mejor="mejor media excluyendo el baseline"),
    )
    base.update(kw)
    return ProtocoloExploratorio(**base)


# ---------------------------------------------------------------------------
# DatasetRegistrado — no se cuela un dato mal formado
# ---------------------------------------------------------------------------

class DatasetRegistradoTest(unittest.TestCase):
    def test_construye_bien_con_datos_validos(self):
        ds = _dataset()
        self.assertEqual(ds.data_id, 1)

    def test_tarea_desconocida_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _dataset(tarea="clasificacion_binaria")

    def test_cubo_desconocido_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _dataset(cubo_de_tamano="enorme")

    def test_sha256_mal_formado_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _dataset(sha256_arff="no-es-un-hash")
        with self.assertRaises(ProtocoloError):
            _dataset(sha256_arff="a" * 63)  # longitud incorrecta

    def test_n_filas_no_positivo_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _dataset(n_filas=0)

    def test_licencia_vacia_se_rechaza(self):
        """«No fabricar lo que no se midió»: un dataset sin licencia
        registrada no entra, en vez de asumir una por omisión."""
        with self.assertRaises(ProtocoloError):
            _dataset(licencia="")

    def test_roundtrip_json(self):
        ds = _dataset(sellado=True)
        de_vuelta = DatasetRegistrado.desde_json(ds.a_json())
        self.assertEqual(de_vuelta, ds)


# ---------------------------------------------------------------------------
# Motor, DisenoDeParticion, PresupuestoPorCubo, ReglaDeCierre
# ---------------------------------------------------------------------------

class ComponentesDelProtocoloTest(unittest.TestCase):
    def test_motor_sin_id_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            Motor(id="")

    def test_motor_sin_configuraciones_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            Motor(id="x", configuraciones=0)

    def test_particion_con_menos_de_dos_folds_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            DisenoDeParticion(folds=1)

    def test_particion_repeticiones_para_distingue_cubo_grande(self):
        p = DisenoDeParticion(repeticiones_pequeno_mediano=3, repeticiones_grande=1)
        self.assertEqual(p.repeticiones_para("pequeno"), 3)
        self.assertEqual(p.repeticiones_para("mediano"), 3)
        self.assertEqual(p.repeticiones_para("grande"), 1)

    def test_presupuesto_sin_un_cubo_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            PresupuestoPorCubo(minutos_por_cubo={"pequeno": 2.0, "mediano": 5.0})  # falta "grande"

    def test_presupuesto_negativo_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            PresupuestoPorCubo(minutos_por_cubo={"pequeno": -1.0, "mediano": 5.0, "grande": 10.0})

    def test_regla_de_cierre_sin_metrica_de_alguna_tarea_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            ReglaDeCierre(puntos=2.0, fraccion_minima=0.8,
                         metrica_por_tarea={"binary_classification": "AUROC"},  # faltan 2
                         definicion_de_mejor="x")

    def test_regla_de_cierre_fraccion_fuera_de_rango_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            ReglaDeCierre(puntos=2.0, fraccion_minima=1.5,
                         metrica_por_tarea={"binary_classification": "AUROC",
                                           "multiclass_classification": "accuracy", "regression": "R2"},
                         definicion_de_mejor="x")


# ---------------------------------------------------------------------------
# ProtocoloExploratorio — el documento entero, y su digest
# ---------------------------------------------------------------------------

class ProtocoloExploratorioTest(unittest.TestCase):
    def test_sin_datasets_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _protocolo_minimo(datasets=())

    def test_sin_motores_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _protocolo_minimo(motores=())

    def test_data_id_repetido_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _protocolo_minimo(datasets=(_dataset(data_id=1), _dataset(data_id=1)))

    def test_digest_es_estable_para_el_mismo_contenido(self):
        p1 = _protocolo_minimo()
        p2 = _protocolo_minimo()
        self.assertEqual(p1.digest(), p2.digest())

    def test_digest_cambia_si_cambia_un_solo_dataset(self):
        p1 = _protocolo_minimo()
        p2 = _protocolo_minimo(datasets=(_dataset(n_filas=1001),))
        self.assertNotEqual(p1.digest(), p2.digest())

    def test_digest_cambia_si_cambia_el_presupuesto(self):
        p1 = _protocolo_minimo()
        p2 = _protocolo_minimo(presupuesto=PresupuestoPorCubo(
            minutos_por_cubo={"pequeno": 3.0, "mediano": 5.0, "grande": 10.0}))
        self.assertNotEqual(p1.digest(), p2.digest())

    def test_digest_cambia_si_cambia_la_regla_de_cierre(self):
        p1 = _protocolo_minimo()
        p2 = _protocolo_minimo(regla_de_cierre=ReglaDeCierre(
            puntos=1.0, fraccion_minima=0.8,
            metrica_por_tarea={"binary_classification": "AUROC", "multiclass_classification": "accuracy",
                               "regression": "R2"},
            definicion_de_mejor="mejor media excluyendo el baseline"))
        self.assertNotEqual(p1.digest(), p2.digest())

    def test_roundtrip_json_conserva_el_digest(self):
        p = _protocolo_minimo()
        de_vuelta = ProtocoloExploratorio.desde_json(p.a_json())
        self.assertEqual(p.digest(), de_vuelta.digest())


# ---------------------------------------------------------------------------
# calcular_coste — el criterio de terminado literal del corte
# ---------------------------------------------------------------------------

class CalcularCosteTest(unittest.TestCase):
    def test_un_dataset_pequeno_dos_motores(self):
        """A mano: 1 dataset, dummy(1 config) + logreg(2 configs) = 3
        configs-motor; 5 folds x 3 repeticiones = 15 pliegues; 15*3 = 45."""
        p = _protocolo_minimo()
        coste = calcular_coste(p)
        self.assertIsInstance(coste, CosteDeLaPasada)
        self.assertEqual(coste.total_ejecuciones, 45)
        self.assertEqual(coste.ejecuciones_por_cubo["pequeno"], 45)
        self.assertEqual(coste.ejecuciones_por_cubo["mediano"], 0)

    def test_cubo_grande_usa_sus_propias_repeticiones(self):
        """Un dataset «grande» con repeticiones_grande=1 (no 3): 5 folds x 1
        rep x 3 configs-motor (dummy+logreg) = 15, no 45."""
        p = _protocolo_minimo(datasets=(_dataset(cubo_de_tamano="grande", n_filas=50_000),))
        coste = calcular_coste(p)
        self.assertEqual(coste.total_ejecuciones, 15)
        self.assertEqual(coste.ejecuciones_por_cubo["grande"], 15)

    def test_coste_en_horas_usa_el_presupuesto_del_cubo_correcto(self):
        """pequeño = 2 min/ajuste: 45 ejecuciones x 2 min = 90 min = 1.5 h."""
        p = _protocolo_minimo()
        coste = calcular_coste(p)
        self.assertAlmostEqual(coste.horas_reloj_peor_caso_secuencial, 1.5, places=6)

    def test_el_paralelismo_divide_las_horas_secuenciales(self):
        p = _protocolo_minimo(presupuesto=PresupuestoPorCubo(
            minutos_por_cubo={"pequeno": 2.0, "mediano": 5.0, "grande": 10.0}, procesos_en_paralelo=3))
        coste = calcular_coste(p)
        self.assertAlmostEqual(coste.horas_reloj_peor_caso_con_paralelismo,
                               coste.horas_reloj_peor_caso_secuencial / 3, places=6)

    def test_mas_datasets_suma_linealmente(self):
        uno = _protocolo_minimo()
        dos = _protocolo_minimo(datasets=(_dataset(data_id=1), _dataset(data_id=2)))
        self.assertEqual(calcular_coste(dos).total_ejecuciones, 2 * calcular_coste(uno).total_ejecuciones)


# ---------------------------------------------------------------------------
# El protocolo REAL registrado — 40 datasets de OpenML, medidos el 2026-09-06
# ---------------------------------------------------------------------------

class ProtocoloRealRegistradoTest(unittest.TestCase):
    """No prueba una función: prueba que el documento que de verdad se va a
    usar en el 101-C2/C3 cumple lo que el anexo C del documento 100 dejó
    escrito como cobertura obligatoria — con datos reales, no un fixture."""

    @classmethod
    def setUpClass(cls):
        cls.protocolo = ProtocoloExploratorio.cargar(_RUTA_PROTOCOLO_REAL)
        cls.payload = json.loads(_RUTA_PROTOCOLO_REAL.read_text(encoding="utf-8"))

    def test_son_40_datasets(self):
        self.assertEqual(len(self.protocolo.datasets), 40)

    def test_el_digest_guardado_coincide_con_el_recalculado(self):
        """El fichero lleva su propio digest declarado (`digest_sha256`):
        tiene que coincidir con recalcularlo desde el contenido — si alguien
        edita el JSON a mano sin recalcular, esta prueba lo caza."""
        self.assertEqual(self.payload["digest_sha256"], self.protocolo.digest())

    def test_cubos_de_tamano_15_15_10(self):
        from collections import Counter
        cubos = Counter(d.cubo_de_tamano for d in self.protocolo.datasets)
        self.assertEqual(cubos["pequeno"], 15)
        self.assertEqual(cubos["mediano"], 15)
        self.assertEqual(cubos["grande"], 10)

    def test_tareas_20_binaria_10_multiclase_10_regresion(self):
        from collections import Counter
        tareas = Counter(d.tarea for d in self.protocolo.datasets)
        self.assertEqual(tareas["binary_classification"], 20)
        self.assertEqual(tareas["multiclass_classification"], 10)
        self.assertEqual(tareas["regression"], 10)

    def test_cobertura_obligatoria_del_anexo_c(self):
        ds = self.protocolo.datasets
        self.assertGreaterEqual(sum(1 for d in ds if d.tiene_faltantes), 10)
        self.assertGreaterEqual(sum(1 for d in ds if d.alta_cardinalidad), 12)
        self.assertGreaterEqual(sum(1 for d in ds if d.desbalanceado), 8)
        self.assertGreaterEqual(sum(1 for d in ds if d.solo_numericas), 6)

    def test_ocho_datasets_sellados(self):
        self.assertEqual(sum(1 for d in self.protocolo.datasets if d.sellado), 8)

    def test_ningun_data_id_repetido(self):
        ids = [d.data_id for d in self.protocolo.datasets]
        self.assertEqual(len(ids), len(set(ids)))

    def test_todos_los_sha256_tienen_forma_valida(self):
        for d in self.protocolo.datasets:
            self.assertEqual(len(d.sha256_arff), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in d.sha256_arff))

    def test_ninguna_licencia_restrictiva(self):
        """Invariante 10 del 101: las licencias se registran, y no se usa un
        dataset con condiciones que no se puedan cumplir. Público/CC0/dominio
        público, no copyleft de software (GPL) mal aplicado a datos — el
        caso real que se sustituyó (`black_friday`, GPL-1) el 2026-09-06."""
        permitidas = {"public", "cc0", "public domain", "cc0 public domain"}
        for d in self.protocolo.datasets:
            self.assertIn(d.licencia.strip().lower(), permitidas,
                         f"{d.nombre} ({d.data_id}) tiene licencia {d.licencia!r}, revisar antes de usarla")

    def test_siete_motores_declarados(self):
        self.assertEqual(len(self.protocolo.motores), 7)
        ids = {m.id for m in self.protocolo.motores}
        self.assertIn("matrixai.dense.torch_cpu", ids)
        self.assertNotIn("matrixai.dense.stdlib", ids,
                        "la decisión confirmada fue torch-CPU en el ranking, no stdlib")

    def test_coste_guardado_coincide_con_recalcularlo(self):
        recalculado = calcular_coste(self.protocolo)
        self.assertEqual(self.payload["coste_calculado"], recalculado.a_json())

    def test_regla_de_cierre_es_la_confirmada_dos_puntos_80_por_ciento(self):
        self.assertEqual(self.protocolo.regla_de_cierre.puntos, 2.0)
        self.assertEqual(self.protocolo.regla_de_cierre.fraccion_minima, 0.80)


if __name__ == "__main__":
    unittest.main()


class AplicarReglaDeCierreTest(unittest.TestCase):
    """101-C4, auditoría propia 2026-09-11: la regla estaba REGISTRADA CON
    HASH desde 101-C1 y ningún código la aplicaba — la conclusión de cartera
    se declaró sin pasar por el listón que el propio protocolo se había puesto
    antes de medir."""

    def _regla(self):
        return ReglaDeCierre(
            puntos=2.0, fraccion_minima=0.8,
            metrica_por_tarea={"binary_classification": "AUROC",
                              "multiclass_classification": "accuracy_o_f1_macro",
                              "regression": "R2"},
            definicion_de_mejor="el mejor por media, excluido el baseline; un fallo cuenta "
                               "como dataset perdido para ese motor")

    def _r(self, dataset, motor, auroc=None, estado="completed"):
        return {"dataset": dataset, "motor": motor, "estado": estado, "auroc": auroc}

    def test_a_menos_de_dos_puntos_cumple(self):
        datos = [self._r("d1", "a", 0.90), self._r("d1", "b", 0.915)]
        r = aplicar_regla_de_cierre(datos, self._regla(), motor="a")
        self.assertTrue(r["detalle"][0]["cumple"])
        self.assertAlmostEqual(r["detalle"][0]["distancia_en_puntos"], 1.5, places=6)

    def test_a_mas_de_dos_puntos_no_cumple(self):
        datos = [self._r("d1", "a", 0.90), self._r("d1", "b", 0.93)]
        self.assertFalse(aplicar_regla_de_cierre(datos, self._regla(), motor="a")["detalle"][0]["cumple"])

    def test_UN_FALLO_pierde_el_dataset_aunque_la_media_sea_buena(self):
        """El texto literal de `definicion_de_mejor`: «un fallo (timeout/crash)
        cuenta como dataset perdido para ese motor». Sin esto, un motor que
        revienta en la mitad de los pliegues y va bien en la otra mitad saldría
        cumpliendo — que es justo lo que la regla quiere impedir."""
        datos = [self._r("d1", "a", 0.90), self._r("d1", "a", None, estado="failed"),
                 self._r("d1", "b", 0.905)]
        r = aplicar_regla_de_cierre(datos, self._regla(), motor="a")
        self.assertFalse(r["detalle"][0]["cumple"])
        self.assertTrue(r["detalle"][0]["perdido_por_fallo"])

    def test_el_baseline_no_cuenta_como_mejor(self):
        """«excluido el baseline dummy» — si contara, cualquier motor real
        estaría siempre por encima y la regla no mediría nada."""
        datos = [self._r("d1", "a", 0.90), self._r("d1", "baseline", 0.999)]
        self.assertTrue(aplicar_regla_de_cierre(datos, self._regla(), motor="a")["detalle"][0]["cumple"])

    def test_la_fraccion_decide_el_veredicto(self):
        datos = []
        for i in range(10):
            datos.append(self._r(f"d{i}", "a", 0.90))
            datos.append(self._r(f"d{i}", "b", 0.90 if i < 8 else 0.99))
        r = aplicar_regla_de_cierre(datos, self._regla(), motor="a")
        self.assertEqual(r["cumplidos"], 8)
        self.assertAlmostEqual(r["fraccion"], 0.8)
        self.assertTrue(r["cumple_la_regla"])  # 0,8 >= 0,8, el borde cuenta

    def test_sobre_los_datos_REALES_de_101_C3_lightgbm_NO_llega(self):
        """El resultado que importa, medido sobre el JSON commiteado: 9/12 =
        0,75 < 0,80. No invalida que lightgbm sea el mejor de los tres (los
        otros dan 0,50), pero el listón prerregistrado NO se alcanza, y uno de
        los tres fallos (Internet-Advertisements, a 1,15 puntos) se pierde por
        el defecto de doble inferencia de tipo que 101-C3 ya declara sin
        corregir."""
        import json
        from pathlib import Path
        ruta = Path(__file__).parent.parent / "benchmarks/fase0/pasada_exploratoria_101_c3_resultado.json"
        if not ruta.exists():
            self.skipTest("el JSON de la pasada no está en este árbol")
        crudo = json.loads(ruta.read_text())
        registros = crudo["resultados"] if isinstance(crudo, dict) and "resultados" in crudo else crudo
        protocolo = ProtocoloExploratorio.cargar(
            str(Path(__file__).parent.parent / "benchmarks/fase0/protocolo_exploratorio.json"))
        r = aplicar_regla_de_cierre(registros, protocolo.regla_de_cierre, motor="lightgbm")
        self.assertEqual((r["cumplidos"], r["datasets"]), (9, 12))
        self.assertFalse(r["cumple_la_regla"])
