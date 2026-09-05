# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""105-C2 — LA INCERTIDUMBRE SEGÚN EL DISEÑO Y EL ESTIMANDO.

Lo que se prueba aquí no es que un intervalo salga «razonable»: es que el
intervalo **respeta lo que la muestra declara sobre sí misma** y que, cuando no
puede respetarlo, lo DICE en vez de fingir un IID que no hay. Cuatro cosas
concretas, una por cada diseño del contrato:

* **IID de clasificación remuestrea filas, ESTRATIFICADO por la clase
  observada** — el recuento de cada clase se comprueba directamente sobre el
  generador de índices, no se infiere del resultado.
* **Grupos remuestrea UNIDADES enteras**: las filas de la misma unidad
  aparecen replicadas el MISMO número de veces en cada remuestra, nunca por
  separado.
* **Temporal remuestrea BLOQUES contiguos** (circular): los índices de un
  mismo bloque son consecutivos módulo `n`, no puntuaciones sueltas.
* **Un diseño sin método soportado (`groups_and_time`) no cae en IID**: el
  intervalo sale NO DISPONIBLE con su motivo, con el mismo patrón «valor o
  motivo, exactamente uno» que ya usa `ValorDeMetrica` del 105-C1.

Y la frontera error/indefinición del C1 se hereda tal cual: pedir un IID sobre
una muestra con unidades repetidas, o un diseño de grupos sin unidad
declarada, es un fallo de CABLEADO —excepción `EntradaNoMedible`—; que la
muestra no tenga datos para estimar variabilidad (una sola unidad, menos de
dos filas, demasiadas remuestras degeneradas por un evento raro) es
INDEFINICIÓN motivada, con su `undefined_reason` y sin inventar un intervalo
de anchura cero.

Las referencias numéricas (NumPy) se usan SOLO para el percentil, con
`skipUnless`: el core sigue siendo stdlib puro.
"""
from __future__ import annotations

import random
import re
import unittest
from collections import Counter
from importlib import util
from pathlib import Path

from matrixai.estudio import EsquemaInvalido, PredictionRecord
from matrixai.estudio.incertidumbre import (
    DISENOS_SOPORTADOS,
    ESTIMANDOS_DE_INTERVALO,
    METODOS_DE_REMUESTREO,
    PROPORCION_MINIMA_VALIDA,
    UNIDADES_DE_REMUESTREO,
    Intervalo,
    _bloques_circulares,
    _indices_groups,
    _indices_iid,
    _longitud_de_bloque_por_omision,
    _percentil,
    _recortar,
    intervalo,
    medir,
)
from matrixai.estudio.metricas import EntradaNoMedible, Indefinida, Muestra, calcular
from matrixai.estudio.textos import motivo

_HAY_NUMPY = util.find_spec("numpy") is not None
MODULO = Path(__file__).resolve().parents[1] / "matrixai" / "estudio" / "incertidumbre.py"

CLASES = ("no", "si")


def _binaria(y, probabilidades=None, **extra) -> Muestra:
    return Muestra.binaria(y, classes=CLASES, positive_label="si",
                           probabilidades=probabilidades, **extra)


def _dataset_ruidoso(n: int, *, seed: int, base_si: float = 0.62, base_no: float = 0.42,
                     ruido: float = 0.35, prevalencia: float = 0.35) -> Muestra:
    """Un dataset con SOLAPE real entre clases: el AUROC no sale en 1,0 exacto y
    el intervalo por IID tiene anchura de verdad, que es lo que hace falta para
    poder distinguir dos semillas."""
    rng = random.Random(seed)
    y = ["si" if rng.random() < prevalencia else "no" for _ in range(n)]
    p = []
    for etiqueta in y:
        base = base_si if etiqueta == "si" else base_no
        p.append(min(max(base + rng.uniform(-ruido, ruido), 0.01), 0.99))
    return _binaria(y, probabilidades=p)


# ---------------------------------------------------------------------------
# Los generadores de índices, uno por diseño — la unidad de remuestreo
# comprobada DIRECTAMENTE, no inferida del resultado final
# ---------------------------------------------------------------------------

class GeneradorIidEstratificadoTest(unittest.TestCase):
    """Decisión 1 del módulo: cada remuestra IID de clasificación conserva el
    recuento EXACTO de cada clase de la muestra original."""

    def test_el_recuento_por_clase_es_EXACTO_en_cada_remuestra(self):
        """Tres positivas entre veinte: si no se estratifica, una remuestra
        puede omitir por azar las tres. Estratificado, nunca."""
        y = ["si"] * 3 + ["no"] * 17
        m = _binaria(y, probabilidades=[0.9, 0.8, 0.7] + [0.2] * 17)
        rng = random.Random(0)
        for repeticion in range(50):
            with self.subTest(repeticion=repeticion):
                indices = _indices_iid(m, rng)
                self.assertEqual(len(indices), 20)
                conteo = Counter(m.y_true[i] for i in indices)
                self.assertEqual(conteo["si"], 3)
                self.assertEqual(conteo["no"], 17)

    def test_un_evento_UNICO_sigue_presente_en_todas_las_remuestras(self):
        """El caso límite: UNA sola fila positiva. `rng.choices` de tamaño 1
        sobre una lista de tamaño 1 siempre devuelve esa fila."""
        y = ["si"] + ["no"] * 299
        m = _binaria(y, probabilidades=[0.9] + [0.3] * 299)
        rng = random.Random(1)
        for _ in range(30):
            indices = _indices_iid(m, rng)
            self.assertEqual(sum(1 for i in indices if m.y_true[i] == "si"), 1)

    def test_regresion_no_estratifica_y_remuestrea_con_reemplazo(self):
        """No hay clases discretas que preservar (decisión 1): el remuestreo es
        simple, y CON reemplazo — se comprueba que aparecen repetidos."""
        m = Muestra.regresion(list(range(30)), [float(i) for i in range(30)])
        rng = random.Random(2)
        vistas_alguna_vez_repetida = False
        for _ in range(20):
            indices = _indices_iid(m, rng)
            self.assertEqual(len(indices), 30)
            if len(set(indices)) < 30:
                vistas_alguna_vez_repetida = True
        self.assertTrue(vistas_alguna_vez_repetida,
                        "con reemplazo, en 20 remuestras de 30 filas tiene que "
                        "aparecer al menos una repetida")


class GeneradorDeGruposTest(unittest.TestCase):
    """Decisión 3: se remuestrean UNIDADES completas. Las filas de la MISMA
    unidad tienen que replicarse el MISMO número de veces — nunca una fila de
    `u1` suelta sin las otras dos."""

    def setUp(self):
        y = ["si", "no", "no", "si", "no", "no"]
        p = [0.9, 0.2, 0.3, 0.8, 0.1, 0.4]
        unidades = ["u1", "u1", "u1", "u2", "u2", "u3"]
        self.muestra = _binaria(y, probabilidades=p, unidades=unidades)

    def test_las_filas_de_la_misma_unidad_se_replican_igual_de_veces(self):
        rng = random.Random(3)
        for repeticion in range(40):
            with self.subTest(repeticion=repeticion):
                indices = _indices_groups(self.muestra, rng)
                conteo = Counter(indices)
                # u1 = filas 0,1,2 — u2 = filas 3,4 — u3 = fila 5
                self.assertEqual(conteo.get(0, 0), conteo.get(1, 0))
                self.assertEqual(conteo.get(1, 0), conteo.get(2, 0))
                self.assertEqual(conteo.get(3, 0), conteo.get(4, 0))

    def test_el_numero_de_unidades_elegidas_es_el_original(self):
        """Se dibujan tantas unidades como había —con reemplazo—, no tantas
        filas: es el bootstrap de grupos clásico, no una submuestra de filas."""
        rng = random.Random(4)
        for _ in range(10):
            indices = _indices_groups(self.muestra, rng)
            unidades_presentes = {self.muestra.units[i] for i in indices}
            self.assertLessEqual(len(unidades_presentes), 3)


class GeneradorDeBloquesTemporalesTest(unittest.TestCase):
    """Decisión 4: bloques CONTIGUOS, no puntuaciones sueltas."""

    def test_los_indices_de_un_bloque_son_consecutivos_modulo_n(self):
        rng = random.Random(5)
        n, longitud = 12, 4
        indices = _bloques_circulares(n, longitud, rng)
        self.assertEqual(len(indices), n)
        for inicio in range(0, n - longitud + 1, longitud):
            tramo = indices[inicio:inicio + longitud]
            for k in range(1, len(tramo)):
                self.assertEqual((tramo[k - 1] + 1) % n, tramo[k],
                                 f"tramo {tramo} no es contiguo módulo {n}")

    def test_envuelve_al_llegar_al_final(self):
        """Un bloque que empieza cerca del final tiene que dar la vuelta, no
        cortarse corto: cortarlo cambiaría la longitud total del bloque."""
        rng = random.Random(0)
        # Con longitud 5 y n=6, CUALQUIER inicio salvo 0 y 1 exige envolver.
        vio_envoltura = False
        for _ in range(30):
            indices = _bloques_circulares(6, 5, rng)
            primero = indices[0]
            tramo = indices[:5]
            if primero >= 2:  # el tramo tiene que cruzar el final para caber
                vio_envoltura = True
                for k in range(1, 5):
                    self.assertEqual((tramo[k - 1] + 1) % 6, tramo[k])
        self.assertTrue(vio_envoltura)

    def test_longitud_por_omision_es_la_heuristica_de_cordura(self):
        self.assertEqual(_longitud_de_bloque_por_omision(1000), round(1000 ** (1 / 3)))
        self.assertEqual(_longitud_de_bloque_por_omision(2), 2)  # nunca por debajo de 2
        self.assertEqual(_longitud_de_bloque_por_omision(3), 2)  # ni por encima de n


# ---------------------------------------------------------------------------
# `_recortar`: el atajo de coste, y que NO desalinee ningún campo
# ---------------------------------------------------------------------------

class RecortarSinRevalidarTest(unittest.TestCase):
    """El atajo de coste del módulo (ver «EL COSTE» en el docstring): tiene que
    reproducir EXACTAMENTE lo que haría reconstruir la `Muestra` fila a fila,
    campo por campo — un campo que se olvide de recortar desalinearía esa
    columna con el resto sin que ningún error lo avise."""

    def test_todos_los_campos_por_fila_quedan_alineados(self):
        m = _binaria(["si", "no", "si"], probabilidades=[0.9, 0.2, 0.7],
                     pesos=[1.0, 2.0, 3.0], unidades=["a", "b", "c"])
        recorte = _recortar(m, [2, 0, 0])
        self.assertEqual(recorte.y_true, ("si", "si", "si"))
        self.assertEqual(recorte.weights, (3.0, 1.0, 1.0))
        self.assertEqual(recorte.units, ("c", "a", "a"))
        self.assertEqual(recorte.probabilities, (m.probabilities[2], m.probabilities[0],
                                                  m.probabilities[0]))
        # Los campos que no son por fila viajan TAL CUAL.
        self.assertEqual(recorte.task, m.task)
        self.assertEqual(recorte.classes, m.classes)
        self.assertEqual(recorte.positive_label, m.positive_label)
        self.assertEqual(recorte.score_rule, m.score_rule)

    def test_un_campo_ausente_sigue_ausente(self):
        m = _binaria(["si", "no"], probabilidades=[0.9, 0.2])
        recorte = _recortar(m, [0, 1, 0])
        self.assertIsNone(recorte.weights)
        self.assertIsNone(recorte.units)
        self.assertIsNone(recorte.scores)


# ---------------------------------------------------------------------------
# `intervalo()`: los cuatro diseños, de punta a punta
# ---------------------------------------------------------------------------

class IntervaloIidTest(unittest.TestCase):

    def setUp(self):
        self.muestra = _dataset_ruidoso(300, seed=5)
        self.punto = calcular("auroc", self.muestra).value

    def test_declara_la_unidad_el_metodo_la_semilla_y_las_remuestras(self):
        r = intervalo("auroc", self.muestra, diseno="iid",
                      estimando="fixed_model_on_population", semilla=42, remuestras=300)
        self.assertTrue(r.disponible)
        self.assertEqual(r.resampling_unit, "row")
        self.assertEqual(r.method, "iid_paired_bootstrap_stratified")
        self.assertEqual(r.seed, 42)
        self.assertEqual(r.n_resamples, 300)
        self.assertEqual(r.design, "iid")
        self.assertEqual(r.estimand, "fixed_model_on_population")
        self.assertLessEqual(r.ci_low, r.ci_high)
        self.assertLess(r.ci_low, self.punto)
        self.assertGreater(r.ci_high, self.punto)

    def test_la_MISMA_semilla_da_el_MISMO_intervalo(self):
        r1 = intervalo("auroc", self.muestra, diseno="iid",
                       estimando="fixed_model_on_population", semilla=7, remuestras=250)
        r2 = intervalo("auroc", self.muestra, diseno="iid",
                       estimando="fixed_model_on_population", semilla=7, remuestras=250)
        self.assertEqual((r1.ci_low, r1.ci_high), (r2.ci_low, r2.ci_high))

    def test_OTRA_semilla_da_OTRO_intervalo(self):
        """Si la semilla no se usara de verdad, dos semillas distintas darían
        el mismo intervalo: exactamente el sabotaje que este test caza."""
        r1 = intervalo("auroc", self.muestra, diseno="iid",
                       estimando="fixed_model_on_population", semilla=1, remuestras=250)
        r2 = intervalo("auroc", self.muestra, diseno="iid",
                       estimando="fixed_model_on_population", semilla=2, remuestras=250)
        self.assertNotEqual((r1.ci_low, r1.ci_high), (r2.ci_low, r2.ci_high))

    def test_regresion_declara_el_metodo_SIN_estratificar(self):
        m = Muestra.regresion([float(i) for i in range(200)],
                              [float(i) + random.Random(9).uniform(-5, 5)
                               for i in range(200)])
        r = intervalo("rmse", m, diseno="iid", estimando="fixed_model_on_population",
                      semilla=1, remuestras=200)
        self.assertTrue(r.disponible)
        self.assertEqual(r.method, "iid_paired_bootstrap")
        self.assertEqual(r.resampling_unit, "row")


class IntervaloGruposTest(unittest.TestCase):

    def test_declara_la_unidad_por_grupo_y_el_metodo_de_cluster(self):
        n_unidades = 40
        y, p, u = [], [], []
        rng = random.Random(11)
        for i in range(n_unidades):
            etiqueta = "si" if rng.random() < 0.4 else "no"
            for _fila in range(3):
                y.append(etiqueta)
                base = 0.65 if etiqueta == "si" else 0.35
                p.append(min(max(base + rng.uniform(-0.3, 0.3), 0.01), 0.99))
                u.append(f"u{i}")
        m = _binaria(y, probabilidades=p, unidades=u)
        r = intervalo("auroc", m, diseno="groups", estimando="fixed_model_on_population",
                      semilla=3, remuestras=300)
        self.assertTrue(r.disponible)
        self.assertEqual(r.resampling_unit, "unit")
        self.assertEqual(r.method, "cluster_bootstrap_by_unit")

    def test_una_sola_unidad_no_dice_nada_de_la_variacion_ENTRE_unidades(self):
        m = _binaria(["si", "no", "no"], probabilidades=[0.8, 0.3, 0.2],
                     unidades=["u1", "u1", "u1"])
        r = intervalo("auroc", m, diseno="groups", estimando="fixed_model_on_population",
                      semilla=1, remuestras=100)
        self.assertFalse(r.disponible)
        self.assertEqual(r.undefined_reason, motivo("unidades_insuficientes", valor=1))

    def test_grupos_sin_unidad_declarada_es_un_fallo_de_cableado(self):
        m = _binaria(["si", "no", "si", "no"], probabilidades=[0.8, 0.3, 0.7, 0.2])
        with self.assertRaises(EntradaNoMedible) as e:
            intervalo("auroc", m, diseno="groups", estimando="fixed_model_on_population",
                     semilla=1, remuestras=100)
        self.assertEqual(e.exception.clave, "remuestreo_de_grupos_sin_unidad")


class IntervaloTemporalTest(unittest.TestCase):

    def test_declara_bloque_y_longitud(self):
        rng = random.Random(21)
        n = 150
        y = [float(i) + rng.uniform(-2, 2) for i in range(n)]
        pred = [float(i) for i in range(n)]
        m = Muestra.regresion(y, pred)
        r = intervalo("mae", m, diseno="temporal", estimando="fixed_model_on_population",
                      semilla=1, remuestras=200)
        self.assertTrue(r.disponible)
        self.assertEqual(r.resampling_unit, "block")
        self.assertEqual(r.method, "circular_block_bootstrap")
        self.assertIsNotNone(r.block_length)
        self.assertGreaterEqual(r.block_length, 2)

    def test_la_longitud_de_bloque_se_puede_declarar_explicitamente(self):
        rng = random.Random(22)
        n = 100
        m = Muestra.regresion([float(i) + rng.uniform(-1, 1) for i in range(n)],
                              [float(i) for i in range(n)])
        r = intervalo("mae", m, diseno="temporal", estimando="fixed_model_on_population",
                      semilla=1, remuestras=100, longitud_de_bloque=7)
        self.assertEqual(r.block_length, 7)


class IntervaloDisenoNoSoportadoTest(unittest.TestCase):
    """Invariante: si un método no está soportado, NUNCA se cae en IID en
    silencio. `groups_and_time` está en el vocabulario compartido
    (`TIPOS_DE_PARTICION`) y NO en `DISENOS_SOPORTADOS`."""

    def test_groups_and_time_no_esta_soportado_y_lo_dice(self):
        self.assertNotIn("groups_and_time", DISENOS_SOPORTADOS)
        self.assertIn("groups_and_time", ("iid", "temporal", "groups", "groups_and_time"))
        m = _binaria(["si", "no", "si", "no"], probabilidades=[0.8, 0.3, 0.7, 0.2],
                     unidades=["u1", "u2", "u3", "u4"])
        r = intervalo("auroc", m, diseno="groups_and_time",
                      estimando="fixed_model_on_population", semilla=1, remuestras=100)
        self.assertFalse(r.disponible)
        self.assertIsNone(r.method)  # nunca corrió ningún remuestreo
        self.assertEqual(r.undefined_reason,
                         motivo("diseno_no_soportado", valor="groups_and_time"))


# ---------------------------------------------------------------------------
# Eventos raros: la razón de ser de la estratificación y del umbral de
# remuestras degeneradas
# ---------------------------------------------------------------------------

class EventosRarosTest(unittest.TestCase):
    """Un solo positivo. Por IID estratificado, siempre presente: el intervalo
    sale disponible, aunque ancho. Por grupos, si ese único positivo vive en
    UNA unidad entre varias, el cluster bootstrap puede omitir esa unidad por
    azar — y cuando eso pasa a menudo, el intervalo se declara indefinido en
    vez de publicar un percentil calculado sobre remuestras degeneradas."""

    def _muestra_grupos(self, n_unidades: int = 8) -> Muestra:
        y, p, u = [], [], []
        for i in range(n_unidades):
            if i == 0:
                y.append("si")
                p.append(0.9)
                u.append(f"u{i}")
            else:
                for _fila in range(5):
                    y.append("no")
                    p.append(0.2)
                    u.append(f"u{i}")
        return _binaria(y, probabilidades=p, unidades=u)

    def test_IID_estratificado_SIGUE_disponible_con_un_solo_evento(self):
        y = ["si"] + ["no"] * 300
        p = [0.85] + [0.3 + random.Random(0).uniform(-0.2, 0.2) for _ in range(300)]
        m = _binaria(y, probabilidades=[min(max(v, 0.01), 0.99) for v in p])
        r = intervalo("auroc", m, diseno="iid", estimando="fixed_model_on_population",
                      semilla=1, remuestras=300)
        self.assertTrue(r.disponible,
                        "estratificar por clase preserva la única fila positiva en "
                        "TODAS las remuestras (decisión 1 del módulo)")

    def test_grupos_con_el_evento_raro_concentrado_en_UNA_unidad_se_declara_INDEFINIDO(self):
        """Medido, no adivinado (ver `PROPORCION_MINIMA_VALIDA`): con 8
        unidades y el único evento en una de ellas, la fracción de remuestras
        que la omiten converge a 1/e ≈ 0,368, muy por debajo del 0,75 exigido
        — y eso vale para CUALQUIER semilla, no una elegida a mano."""
        m = self._muestra_grupos(n_unidades=8)
        for semilla in (0, 1, 2, 3, 4, 123):
            with self.subTest(semilla=semilla):
                r = intervalo("auroc", m, diseno="groups",
                             estimando="fixed_model_on_population", semilla=semilla,
                             remuestras=500)
                self.assertFalse(r.disponible)
                self.assertGreater(r.n_resamples_degenerate, 0)
                # La fracción medida (documentada arriba) ronda 0,33-0,37: muy
                # lejos de ser un caso límite del umbral 0,75.
                self.assertGreater(r.n_resamples_degenerate / r.n_resamples, 0.20)


# ---------------------------------------------------------------------------
# El punto ya indefinido, y las muestras sin datos para estimar variabilidad
# ---------------------------------------------------------------------------

class SinDatosParaEstimarVariabilidadTest(unittest.TestCase):

    def test_si_el_PUNTO_ya_es_indefinido_el_intervalo_reusa_el_MISMO_motivo(self):
        """Una sola clase presente: `calcular()` ya da indefinida. No hay nada
        que remuestrear, y el motivo NO se reinventa."""
        m = _binaria(["no", "no", "no"], probabilidades=[0.1, 0.2, 0.3])
        punto = calcular("auroc", m)
        self.assertIsNone(punto.value)
        r = intervalo("auroc", m, diseno="iid", estimando="fixed_model_on_population",
                      semilla=1, remuestras=100)
        self.assertFalse(r.disponible)
        self.assertEqual(r.undefined_reason, punto.undefined_reason)
        self.assertEqual(r.resampling_unit, "none")

    def test_menos_de_dos_filas_no_da_para_estimar_variabilidad_IID(self):
        m = Muestra.regresion([1.0], [1.2])
        r = intervalo("mae", m, diseno="iid", estimando="fixed_model_on_population",
                      semilla=1, remuestras=100)
        self.assertFalse(r.disponible)
        self.assertEqual(r.undefined_reason, motivo("observaciones_insuficientes"))

    def test_menos_de_cuatro_filas_no_da_para_un_bloque_temporal(self):
        m = Muestra.regresion([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
        r = intervalo("mae", m, diseno="temporal", estimando="fixed_model_on_population",
                      semilla=1, remuestras=100)
        self.assertFalse(r.disponible)
        self.assertEqual(r.undefined_reason, motivo("observaciones_insuficientes"))


# ---------------------------------------------------------------------------
# No convertir repeticiones en observaciones independientes, y no equiparar
# AUROC agrupado con promedio de pliegues
# ---------------------------------------------------------------------------

class RepeticionesYPliguesTest(unittest.TestCase):
    """Diez observaciones de base, cada una repetida dos veces (como saldría
    de dos repeticiones de validación cruzada), con `resampling_unit` puesto a
    la observación de base — así es como el 104-C0 permite declarar que dos
    filas son la MISMA observación medida dos veces."""

    def _registros(self) -> list[PredictionRecord]:
        registros = []
        rng = random.Random(6)
        for i in range(10):
            etiqueta = "si" if i < 4 else "no"
            for repeticion in range(2):
                base = 0.7 if etiqueta == "si" else 0.3
                p = min(max(base + rng.uniform(-0.2, 0.2), 0.01), 0.99)
                registros.append(PredictionRecord(
                    row_id=f"o{i}_r{repeticion}", candidate="c1", y_true=etiqueta,
                    probabilities=(1.0 - p, p), classes=CLASES,
                    resampling_unit=f"o{i}", repeat=repeticion + 1))
        return registros

    def test_IID_sobre_repeticiones_se_RECHAZA_por_cableado(self):
        """20 filas, 10 unidades: pedir IID aquí fingiría 20 observaciones
        independientes cuando en realidad son 10, cada una medida dos veces."""
        m = Muestra.desde_registros(self._registros(), task="binary_classification",
                                    positive_label="si")
        self.assertEqual(m.n, 20)
        self.assertEqual(m.n_unidades, 10)
        with self.assertRaises(EntradaNoMedible) as e:
            intervalo("auroc", m, diseno="iid", estimando="fixed_model_on_population",
                     semilla=1, remuestras=100)
        self.assertEqual(e.exception.clave, "unidad_repetida_en_diseno_iid")

    def test_grupos_con_estimando_oof_pooled_SI_funciona(self):
        m = Muestra.desde_registros(self._registros(), task="binary_classification",
                                    positive_label="si")
        r = intervalo("auroc", m, diseno="groups", estimando="oof_pooled_estimate",
                      semilla=1, remuestras=200)
        self.assertTrue(r.disponible)
        self.assertEqual(r.resampling_unit, "unit")
        self.assertEqual(r.estimand, "oof_pooled_estimate")

    def test_promedio_de_pliegues_NO_esta_implementado_y_se_rechaza_EXPLICITO(self):
        """No se equipara: `oof_mean_of_folds` pediría resamplear PLIEGUES y
        promediar sus métricas, un mecanismo distinto que este corte no
        implementa. Pedirlo se rechaza — no se calcula a medias bajo el mismo
        nombre que `oof_pooled_estimate`."""
        m = Muestra.desde_registros(self._registros(), task="binary_classification",
                                    positive_label="si")
        self.assertNotIn("oof_mean_of_folds", ESTIMANDOS_DE_INTERVALO)
        with self.assertRaises(EsquemaInvalido):
            intervalo("auroc", m, diseno="groups", estimando="oof_mean_of_folds",
                     semilla=1, remuestras=100)


# ---------------------------------------------------------------------------
# `medir()`: el número y su incertidumbre, en un solo `ValorDeMetrica`
# ---------------------------------------------------------------------------

class MedirTest(unittest.TestCase):

    def test_el_valor_lleva_su_incertidumbre_embebida(self):
        m = _dataset_ruidoso(200, seed=8)
        valor = medir("auroc", m, diseno="iid", estimando="fixed_model_on_population",
                      semilla=1, remuestras=200)
        self.assertIsNotNone(valor.value)
        self.assertIsNotNone(valor.uncertainty)
        self.assertEqual(valor.uncertainty["design"], "iid")
        self.assertEqual(valor.uncertainty["seed"], 1)
        self.assertTrue(valor.uncertainty["ci_low"] <= valor.value <= valor.uncertainty["ci_high"])

    def test_el_diccionario_hace_una_ida_y_vuelta_por_Intervalo_desde_json(self):
        m = _dataset_ruidoso(150, seed=9)
        valor = medir("auroc", m, diseno="iid", estimando="fixed_model_on_population",
                      semilla=3, remuestras=150)
        reconstruido = Intervalo.desde_json(valor.uncertainty)
        self.assertEqual(reconstruido.a_json(), valor.uncertainty)


# ---------------------------------------------------------------------------
# `Intervalo`: valor o motivo, exactamente uno — como `ValorDeMetrica`
# ---------------------------------------------------------------------------

class IntervaloEsquemaTest(unittest.TestCase):

    def _base(self, **cambios):
        campos = dict(metric_id="auroc", design="iid", estimand="fixed_model_on_population",
                     resampling_unit="row", seed=1, n_resamples=100, level=0.95)
        campos.update(cambios)
        return campos

    def test_ni_valor_ni_motivo_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            Intervalo(**self._base())
        self.assertEqual(e.exception.clave, "intervalo_sin_valor_ni_motivo")

    def test_valor_Y_motivo_a_la_vez_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            Intervalo(**self._base(ci_low=0.1, ci_high=0.2,
                                   undefined_reason=motivo("observaciones_insuficientes")))
        self.assertEqual(e.exception.clave, "intervalo_con_valor_y_motivo")

    def test_un_limite_sin_el_otro_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            Intervalo(**self._base(ci_low=0.1))
        self.assertEqual(e.exception.clave, "intervalo_a_medias")

    def test_el_limite_inferior_por_encima_del_superior_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido) as e:
            Intervalo(**self._base(ci_low=0.9, ci_high=0.1))
        self.assertEqual(e.exception.clave, "intervalo_al_reves")

    def test_un_diseno_fuera_del_vocabulario_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            Intervalo(**self._base(design="lo_que_sea", ci_low=0.1, ci_high=0.2))

    def test_a_json_hace_ida_y_vuelta(self):
        original = Intervalo(**self._base(method="iid_paired_bootstrap_stratified",
                                          n_resamples_used=95, n_resamples_degenerate=5,
                                          ci_low=0.1, ci_high=0.4, elapsed_seconds=0.02))
        self.assertEqual(Intervalo.desde_json(original.a_json()), original)


# ---------------------------------------------------------------------------
# El percentil: interpolación lineal, contra NumPy cuando está disponible
# ---------------------------------------------------------------------------

class PercentilTest(unittest.TestCase):

    def test_casos_a_mano(self):
        datos = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_percentil(datos, 0.0), 1.0)
        self.assertAlmostEqual(_percentil(datos, 1.0), 4.0)
        self.assertAlmostEqual(_percentil(datos, 0.5), 2.5)

    @unittest.skipUnless(_HAY_NUMPY, "sin numpy en el entorno")
    def test_coincide_con_numpy_percentile_metodo_linear(self):
        import numpy as np

        datos = [float(x) for x in (5, 1, 9, 3, 7, 2, 8, 4, 6)]
        ordenados = sorted(datos)
        for q in (0.025, 0.1, 0.5, 0.9, 0.975):
            with self.subTest(q=q):
                propio = _percentil(ordenados, q)
                referencia = float(np.percentile(np.array(datos), q * 100, method="linear"))
                self.assertAlmostEqual(propio, referencia, places=12)


# ---------------------------------------------------------------------------
# Pureza y vocabulario compartido
# ---------------------------------------------------------------------------

class PurezaYVocabularioTest(unittest.TestCase):

    def test_el_modulo_es_stdlib_puro(self):
        prohibidos = re.compile(
            r"^\s*(?:import|from)\s+(numpy|scipy|sklearn|pandas|torch|onnx)\b",
            re.MULTILINE)
        self.assertIsNone(prohibidos.search(MODULO.read_text(encoding="utf-8")))

    def test_los_disenos_soportados_son_un_subconjunto_del_vocabulario_compartido(self):
        """Reutiliza `TIPOS_DE_PARTICION` del 104-C0 — no declara su propia
        lista de nombres de diseño."""
        from matrixai.estudio.vocabulario import TIPOS_DE_PARTICION
        for d in DISENOS_SOPORTADOS:
            self.assertIn(d, TIPOS_DE_PARTICION)

    def test_todos_los_diccionarios_de_vocabulario_son_tuplas_no_vacias(self):
        for coleccion in (DISENOS_SOPORTADOS, ESTIMANDOS_DE_INTERVALO,
                          METODOS_DE_REMUESTREO, UNIDADES_DE_REMUESTREO):
            self.assertTrue(coleccion)
            self.assertEqual(len(coleccion), len(set(coleccion)))

    def test_el_umbral_de_remuestras_validas_esta_por_debajo_de_uno(self):
        self.assertTrue(0.0 < PROPORCION_MINIMA_VALIDA < 1.0)


if __name__ == "__main__":
    unittest.main()
