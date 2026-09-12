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
    ANCLAS_DE_ESCALADO,
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
    cpus_disponibles,
    nucleos_fisicos,
    reserva_segura,
    speedup_medido,
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
        """`cpus` va FIJADO a las CPUs lógicas que el propio protocolo declara
        en `recursos_declarados`: el coste guardado describe la máquina
        registrada, no el ordenador donde se corra la suite — si no, esta
        prueba se pondría roja en cualquier máquina con otro número de CPUs."""
        cpus = self.payload["recursos_declarados"]["cpu_logicas"]
        recalculado = calcular_coste(self.protocolo, cpus=cpus)
        self.assertEqual(self.payload["coste_calculado"], recalculado.a_json())

    def test_regla_de_cierre_es_la_confirmada_dos_puntos_80_por_ciento(self):
        self.assertEqual(self.protocolo.regla_de_cierre.puntos, 2.0)
        self.assertEqual(self.protocolo.regla_de_cierre.fraccion_minima, 0.80)


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


# ---------------------------------------------------------------------------
# 101-C1/C2 — sobre-reserva de concurrencia
# ---------------------------------------------------------------------------
# La deuda literal: «4 hilos x 6 procesos = 24 sobre 8 CPUs lógicas reales, y
# `calcular_coste()` divide linealmente sin contar la contención».
#
# Cada afirmación va con SUS DOS MITADES a propósito: sin la mitad de «con
# holgura sí usa lo que hay», un techo que reservase siempre 1 proceso pasaría
# por reparación y haría la pasada eterna.


class CpusDisponiblesTest(unittest.TestCase):
    def test_nunca_devuelve_cero_ni_negativo(self):
        self.assertGreaterEqual(cpus_disponibles(), 1)

    def test_no_pasa_de_la_afinidad_del_proceso(self):
        """`os.cpu_count()` cuenta las del HOST: en esta máquina 8 de afinidad
        frente a las 128 que declara `nproc --all`. Reservar por `cpu_count`
        sería reservar 16 veces lo que hay."""
        import os
        self.assertLessEqual(cpus_disponibles(), len(os.sched_getaffinity(0)))

    def test_bajo_taskset_manda_la_afinidad_y_no_el_recuento_del_host(self):
        """CON DIENTES: sin restringir, `os.cpu_count()` y la afinidad dan lo
        mismo (8) y confundirlos no se nota. Bajo `taskset -c 0,1` se separan
        —afinidad 2, `cpu_count` 8— y usar `cpu_count` reservaría CUATRO veces
        lo que toca. Se mide de verdad, en un subproceso restringido: no hay
        forma honesta de comprobar esto sin restringir la afinidad."""
        import shutil, subprocess, sys as _sys
        if shutil.which("taskset") is None:
            self.skipTest("sin taskset: no se puede restringir la afinidad")
        guion = (
            "import sys, os; sys.path.insert(0, %r);"
            "from protocolo import cpus_disponibles;"
            "print(cpus_disponibles(), os.cpu_count())"
            % str(_RUTA_PROTOCOLO_REAL.parent)
        )
        salida = subprocess.run(["taskset", "-c", "0,1", _sys.executable, "-c", guion],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(salida.returncode, 0, salida.stderr)
        medido, del_host = (int(x) for x in salida.stdout.split())
        self.assertEqual(medido, 2, "no respetó la afinidad de taskset")
        self.assertLessEqual(medido, del_host)

    def test_la_cuota_de_cgroup_manda_cuando_existe(self):
        """`docker --cpus=2` se ve como «200000 100000». Esta máquina no tiene
        cuota, así que sin fichero de prueba esta rama no se probaría nunca."""
        import tempfile
        from benchmarks.fase0.protocolo import _cuota_de_cgroup
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "cpu.max"
            ruta.write_text("200000 100000", encoding="utf-8")
            self.assertEqual(_cuota_de_cgroup(ruta), 2)

    def test_max_significa_sin_tope_no_cero(self):
        """«Un valor ausente no es un cero»: `max` es SIN cuota, y devolver 0
        aquí haría que `cpus_disponibles` reservase 1 en toda máquina sin
        contenedor."""
        import tempfile
        from benchmarks.fase0.protocolo import _cuota_de_cgroup
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "cpu.max"
            ruta.write_text("max 100000", encoding="utf-8")
            self.assertIsNone(_cuota_de_cgroup(ruta))

    def test_nucleos_fisicos_no_supera_a_las_logicas(self):
        """Medido aquí: 4 físicos, 8 lógicos (SMT x2). `None` es «no lo sé»,
        que es una respuesta válida y no un cero."""
        fisicos = nucleos_fisicos()
        if fisicos is not None:
            self.assertGreaterEqual(fisicos, 1)
            self.assertLessEqual(fisicos, cpus_disponibles())


class SpeedupMedidoTest(unittest.TestCase):
    """MITAD 1: no promete más de lo que la máquina da.
    MITAD 2: con holgura sí sube — si no, «reservar 1 siempre» pasaría."""

    def test_reservar_de_mas_no_da_mas_que_llenar_la_maquina(self):
        """El caso de la deuda: 24 hilos sobre 8 CPUs. La división lineal daba
        6x (un proceso por cada 4 hilos); lo medido es el techo 4,69x, y 24
        hilos no rinden más que 8."""
        lleno = speedup_medido(8, cpus=8)
        self.assertEqual(speedup_medido(24, cpus=8), lleno)
        self.assertEqual(speedup_medido(96, cpus=8), lleno)
        self.assertLess(speedup_medido(24, cpus=8), 6.0)

    def test_el_techo_es_el_medido_469_en_ocho_cpus(self):
        self.assertAlmostEqual(speedup_medido(8, cpus=8), 4.69, places=2)

    def test_nunca_supera_el_numero_de_cpus(self):
        """Ningún reparto puede sacar de N CPUs más de N veces un hilo."""
        for cpus in (1, 2, 4, 8, 16):
            for hilos in (1, 2, 3, 4, 8, 16, 24, 64):
                self.assertLessEqual(speedup_medido(hilos, cpus=cpus), cpus + 1e-9,
                                     f"{hilos} hilos sobre {cpus} CPUs prometen más que las CPUs")

    def test_con_holgura_usa_lo_que_hay(self):
        """LA OTRA MITAD. Con 4 hilos de 8 lo medido es 3,60x — muy por encima
        de 1. Una versión que devolviese siempre 1,0 «no se pasaría» de las
        CPUs y aun así sería una reparación inútil: esta prueba la caza."""
        self.assertGreater(speedup_medido(4, cpus=8), 3.0)
        self.assertGreater(speedup_medido(2, cpus=8), 1.5)
        self.assertGreater(speedup_medido(8, cpus=8), speedup_medido(4, cpus=8))

    def test_es_monotono_no_decreciente(self):
        previo = 0.0
        for hilos in range(1, 33):
            actual = speedup_medido(hilos, cpus=8)
            self.assertGreaterEqual(actual, previo - 1e-9,
                                    f"añadir hilos bajó el speedup en {hilos}")
            previo = actual

    def test_un_solo_hilo_es_la_base(self):
        self.assertAlmostEqual(speedup_medido(1, cpus=8), 1.0, places=2)

    def test_las_anclas_son_las_medidas_y_estan_ordenadas(self):
        """Si alguien retoca la tabla medida, que se vea aquí: son MEDICIONES
        del 2026-09-12, no parámetros que se ajusten a gusto."""
        self.assertEqual(ANCLAS_DE_ESCALADO, ((1, 1.00), (4, 3.60), (8, 4.69)))
        hilos = [h for h, _ in ANCLAS_DE_ESCALADO]
        self.assertEqual(hilos, sorted(hilos))

    def test_rechaza_entradas_absurdas(self):
        with self.assertRaises(ProtocoloError):
            speedup_medido(0, cpus=8)
        with self.assertRaises(ProtocoloError):
            speedup_medido(4, cpus=0)


class ReservaSeguraTest(unittest.TestCase):
    """El límite duro: nunca reservar más hilos que CPUs hay."""

    def test_el_caso_de_la_deuda_cuatro_hilos_en_ocho_cpus_son_dos_procesos(self):
        """Lo registrado eran 6 procesos (24 hilos). Caben 2."""
        self.assertEqual(reserva_segura(4, cpus=8), 2)

    def test_nunca_reserva_mas_hilos_que_cpus(self):
        for cpus in (1, 2, 4, 8, 16, 64):
            for hilos in (1, 2, 3, 4, 5, 8, 16):
                procesos = reserva_segura(hilos, cpus=cpus)
                if hilos <= cpus:
                    self.assertLessEqual(procesos * hilos, cpus,
                                         f"{procesos}x{hilos} se pasa de {cpus} CPUs")

    def test_con_holgura_usa_lo_que_hay(self):
        """LA OTRA MITAD: 1 hilo por proceso sobre 8 CPUs son 8 procesos, no 1.
        Un `return 1` pasaría la mitad de arriba y estrangularía la pasada."""
        self.assertEqual(reserva_segura(1, cpus=8), 8)
        self.assertEqual(reserva_segura(2, cpus=8), 4)
        self.assertEqual(reserva_segura(1, cpus=64), 64)

    def test_nunca_devuelve_cero_aunque_un_proceso_solo_ya_se_pase(self):
        """No lanzar nada no es una opción: 1 proceso, y `sobre_reserva` es
        quien avisa de que ese proceso solo ya se pasa."""
        self.assertEqual(reserva_segura(16, cpus=8), 1)
        self.assertEqual(reserva_segura(1000, cpus=1), 1)

    def test_rechaza_entradas_absurdas(self):
        with self.assertRaises(ProtocoloError):
            reserva_segura(0, cpus=8)
        with self.assertRaises(ProtocoloError):
            reserva_segura(4, cpus=0)


class PresupuestoReservaTest(unittest.TestCase):
    def _presupuesto(self, **kw):
        base = dict(minutos_por_cubo={"pequeno": 2.0, "mediano": 5.0, "grande": 10.0})
        base.update(kw)
        return PresupuestoPorCubo(**base)

    def test_hilos_reservados_son_hilos_por_procesos(self):
        """El número de la deuda: 4 x 6 = 24."""
        self.assertEqual(self._presupuesto(hilos=4, procesos_en_paralelo=6).hilos_reservados, 24)

    def test_sobre_reserva_cuenta_los_hilos_de_mas(self):
        self.assertEqual(self._presupuesto(hilos=4, procesos_en_paralelo=6).sobre_reserva(cpus=8), 16)

    def test_sin_sobre_reserva_cuando_cabe(self):
        """LA OTRA MITAD: si cabe, es 0 — no un aviso permanente que se acabe
        ignorando por salir siempre."""
        self.assertEqual(self._presupuesto(hilos=4, procesos_en_paralelo=2).sobre_reserva(cpus=8), 0)
        self.assertEqual(self._presupuesto(hilos=1, procesos_en_paralelo=8).sobre_reserva(cpus=8), 0)

    def test_procesos_que_caben_dice_el_arreglo(self):
        self.assertEqual(self._presupuesto(hilos=4, procesos_en_paralelo=6).procesos_que_caben(cpus=8), 2)

    def test_los_campos_derivados_no_entran_en_el_json_ni_en_el_digest(self):
        """Serializarlos ataría el digest registrado a la máquina que lo
        serializa: el mismo protocolo daría hashes distintos en dos
        ordenadores, y el invariante 1 dejaría de significar nada."""
        b = self._presupuesto(hilos=4, procesos_en_paralelo=6)
        self.assertEqual(set(b.a_json()), {"minutos_por_cubo", "hilos", "procesos_en_paralelo"})


class CosteConContencionTest(unittest.TestCase):
    """`calcular_coste` ya no divide linealmente y calla: declara las dos
    cuentas y en qué se diferencian."""

    def _protocolo(self, hilos=4, procesos=6):
        return _protocolo_minimo(presupuesto=PresupuestoPorCubo(
            minutos_por_cubo={"pequeno": 2.0, "mediano": 5.0, "grande": 10.0},
            hilos=hilos, procesos_en_paralelo=procesos))

    def test_declara_la_sobre_reserva_en_vez_de_callarla(self):
        c = calcular_coste(self._protocolo(), cpus=8)
        self.assertEqual(c.hilos_reservados, 24)
        self.assertEqual(c.cpus_disponibles, 8)
        self.assertEqual(c.sobre_reserva, 16)

    def test_el_suelo_medido_es_mayor_que_la_division_lineal(self):
        """Si se reserva de más, el tiempo REAL no puede ser el que promete
        dividir por 6: el suelo medido tiene que salir MAYOR. Un modelo que
        siguiera dividiendo linealmente daría los dos números iguales."""
        c = calcular_coste(self._protocolo(), cpus=8)
        self.assertGreater(c.horas_reloj_suelo_medido, c.horas_reloj_peor_caso_con_paralelismo)

    def test_conserva_la_division_lineal_ya_publicada(self):
        """La cota de 74,93 h del protocolo registrado NO se sustituye en
        silencio: sigue ahí, con su nombre, junto al número nuevo."""
        c = calcular_coste(self._protocolo(), cpus=8)
        self.assertAlmostEqual(c.horas_reloj_peor_caso_con_paralelismo,
                               c.horas_reloj_peor_caso_secuencial / 6, places=2)

    def test_sin_sobre_reserva_los_dos_numeros_se_acercan(self):
        """LA OTRA MITAD: con una reserva que cabe (2 procesos x 4 hilos = 8),
        no hay sobre-reserva y el suelo medido no se dispara. Sin esta mitad,
        un modelo que siempre inflara las horas pasaría por reparación."""
        c = calcular_coste(self._protocolo(procesos=2), cpus=8)
        self.assertEqual(c.sobre_reserva, 0)
        self.assertLess(c.horas_reloj_suelo_medido, 2.0 * c.horas_reloj_peor_caso_con_paralelismo)

    def test_el_speedup_declarado_nunca_pasa_de_las_cpus(self):
        for procesos in (1, 2, 6, 12):
            c = calcular_coste(self._protocolo(procesos=procesos), cpus=8)
            self.assertLessEqual(c.speedup_efectivo_medido, 8.0)

    def test_las_ejecuciones_no_cambian_al_contar_la_contencion(self):
        """La contención afecta al RELOJ, no al número de ajustes: 6.500
        siguen siendo 6.500."""
        self.assertEqual(calcular_coste(self._protocolo(), cpus=8).total_ejecuciones,
                         calcular_coste(self._protocolo(procesos=2), cpus=8).total_ejecuciones)

    def test_el_json_lleva_los_campos_nuevos(self):
        d = calcular_coste(self._protocolo(), cpus=8).a_json()
        for clave in ("hilos_reservados", "cpus_disponibles", "sobre_reserva",
                      "speedup_efectivo_medido", "horas_reloj_suelo_medido"):
            self.assertIn(clave, d)


class ProtocoloRegistradoSobreReservaTest(unittest.TestCase):
    """El protocolo REAL registrado, con los números de la deuda. Es la guarda
    de regresión: si alguien vuelve a poner 6 procesos x 4 hilos creyendo que
    caben, esto lo dice con nombres y números."""

    @classmethod
    def setUpClass(cls):
        cls.protocolo = ProtocoloExploratorio.cargar(_RUTA_PROTOCOLO_REAL)
        cls.coste = calcular_coste(cls.protocolo, cpus=8)

    def test_el_protocolo_registrado_reserva_24_hilos_sobre_8_cpus(self):
        self.assertEqual(self.protocolo.presupuesto.hilos_reservados, 24)
        self.assertEqual(self.coste.sobre_reserva, 16)

    def test_la_cota_lineal_publicada_sigue_siendo_7493_horas(self):
        self.assertAlmostEqual(self.coste.horas_reloj_peor_caso_con_paralelismo, 74.93, places=2)

    def test_el_suelo_medido_es_9586_horas(self):
        """Número NUEVO, no sustituto: con el techo medido 4,69x (no 6x) las
        449,58 h secuenciales dan 95,86 h, no 74,93."""
        self.assertAlmostEqual(self.coste.horas_reloj_suelo_medido, 95.86, places=2)
        self.assertAlmostEqual(self.coste.speedup_efectivo_medido, 4.69, places=2)

    def test_lo_que_cabria_son_dos_procesos_no_seis(self):
        self.assertEqual(self.protocolo.presupuesto.procesos_que_caben(cpus=8), 2)


if __name__ == "__main__":
    unittest.main()
