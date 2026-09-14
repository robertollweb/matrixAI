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
import os
import unittest
from pathlib import Path

from benchmarks.fase0.protocolo import (
    ANCLAS_DE_ESCALADO,
    UMBRAL_ALTA_CARDINALIDAD,
    aplicar_regla_de_cierre,
    cardinalidad_nominal_declarada,
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
               n_filas=1000, n_columnas=10, tiene_faltantes=False,
               max_cardinalidad_nominal=0, alta_cardinalidad=False,
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
        edita el JSON a mano sin recalcular, esta prueba lo caza.

        **Pero SOLO caza eso**, y ese límite lo destapó la auditoría del
        2026-09-13: relajó el listón de la regla de cierre a 0,75, **recalculó
        el digest** y este aserto siguió VERDE. Comprueba auto-consistencia,
        que es otra cosa que estar registrado. El ancla real es el test de
        abajo."""
        self.assertEqual(self.payload["digest_sha256"], self.protocolo.digest())

    #: El digest del PRIMER registro: el que el protocolo tenía el 2026-09-06 a
    #: las 16:32, en el commit `a3d551a`, **antes de que se midiera nada** (la
    #: primera pasada es del 09-07 a las 11:49). Ya no es el digest del
    #: fichero, y se conserva porque sigue siendo un HECHO: es el protocolo
    #: contra el que se midieron los 720 intentos, y es el que la evidencia
    #: (`pasada_exploratoria_..._con_alcance.json`), la calibración y la
    #: cartera de `matrixai_engines` citan. Un artefacto medido contra él no
    #: se vuelve falso porque el catálogo se corrija después.
    DIGEST_DEL_PRIMER_REGISTRO_2026_09_06 = (
        "493a6f1df9175cfe0d4f736216af11a740f9c4913426700e0ac10c7eb165f0b6")

    #: El digest VIGENTE, tras la ÚNICA re-firma. Escrito a mano, como el
    #: anterior: es el ancla, y un ancla calculada no ancla nada.
    DIGEST_REGISTRADO_ANTES_DE_MEDIR = (
        "ea50ca482a815627364b3439bfcf10e78295a13ae5f821219d0fb40329e65301")

    def test_el_digest_es_EL_MISMO_que_antes_de_medir(self):
        """El ancla, y no existía: **nada ataba el digest a su valor
        literal.**

        **RE-FIRMA DEL 2026-09-13, la única que ha habido.** Esta constante
        pasó de `493a6f1d…` a `eb54f421…`. Se re-firmó por DOS correcciones, y
        la primera frase que alguien va a querer leer dentro de seis meses es
        esta: **LA REGLA DE CIERRE NO SE MOVIÓ.** Ni el listón (2,0 puntos), ni
        la fracción mínima (0,80), ni la definición de «mejor», ni las métricas
        por tarea, ni la lista de 40 datasets, ni las particiones. Lo comprueba
        `test_la_REGLA_DE_CIERRE_no_se_movio_en_la_re_firma`, byte a byte
        contra el literal de entonces, y no este comentario.

        Qué se corrigió, y por qué cada cosa era falsa:

        1. **La cardinalidad declarada del catálogo, falsa en DIEZ de los
           cuarenta.** El campo se llamaba `alta_cardinalidad` y se rellenaba
           con «columnas categóricas / columnas > 0,3», que es otra cosa.
           `KDDCup09_appetency` decía `false` con **15.415 niveles** en
           `Var200` (71.506 en todo el fichero); `kr-vs-kp`,
           `PhishingWebsites`, `connect-4` e `Internet-Advertisements` decían
           `true` con un máximo de 2 o 3 niveles. Ahora el catálogo registra el
           NÚMERO medido sobre el ARFF sellado (`max_cardinalidad_nominal`) y
           el booleano se lee contra el umbral que el anexo C §2.2 ya tenía
           escrito antes de medir nada: 50 niveles.
        2. **La reserva de concurrencia: 24 hilos sobre 8 CPUs**, `hilos=4` x
           `procesos_en_paralelo=6`, triple de lo que la máquina tiene — y
           `cpu_fisicas` decía 8 cuando son 4 (8 son las lógicas). Este
           servidor se ha caído dos veces por exactamente esto. Ahora
           `procesos_en_paralelo=2`, que es `reserva_segura(4)` sobre
           `cpus_disponibles()`, MEDIDO con las funciones de este módulo:
           sobre-reserva 0, y sin perder nada de velocidad porque 24 hilos
           sobre 8 CPUs rendían el mismo techo de 4,69x que 8.

        **El veredicto no se movió**: re-aplicar la regla a la evidencia sigue
        dando lightgbm 10/12 = 0,833 CUMPLE. Lo comprueba
        `test_el_veredicto_NO_se_movio_con_la_re_firma`.

        Lo de siempre, que la re-firma no deroga: un protocolo «registrado con
        hash» solo vale si el hash es EL QUE SE ESCRIBIÓ. Con la
        auto-consistencia sola, cualquiera puede aflojar la regla, recalcular y
        quedarse con un fichero que cuadra consigo mismo y con la suite en
        verde — que es exactamente lo que el auditor hizo el 2026-09-13 para
        demostrar el hueco. Por eso el número va ESCRITO aquí: si alguien toca
        el protocolo, esto se pone rojo, y esa es la conversación que tiene que
        haber.

        Un protocolo «registrado con hash» solo vale si el hash es EL DE
        ENTONCES. Con la auto-consistencia sola, cualquiera puede aflojar la
        Cambiar lo que se prometió medir **después de ver los números** no es
        un detalle de implementación. Si el cambio es legítimo —añadir un
        motor, corregir un dato falso del catálogo, bajar una reserva que
        tumba la máquina— se re-firma A PROPÓSITO, se cambia esta constante, y
        el commit explica qué se re-firmó y por qué. Lo que NO se re-firma
        nunca por este camino es la regla de cierre: eso es otra conversación,
        y tiene su propia prueba debajo.

        El otro cambio que el protocolo ha tenido, y que NO es una re-firma,
        es el bloque `coste_calculado`: es derivado y va deliberadamente FUERA
        de `a_json()`, así que no mueve el digest, y está bien que no lo mueva.
        """
        self.assertEqual(
            self.protocolo.digest(), self.DIGEST_REGISTRADO_ANTES_DE_MEDIR,
            "el protocolo ha cambiado desde la última re-firma deliberada. Si "
            "el cambio es a propósito hay que re-firmarlo explícitamente y "
            "decir aquí qué, cuándo y por qué; si no lo es, los números "
            "medidos ya no responden a lo que se prometió medir")
        self.assertNotEqual(
            self.protocolo.digest(), self.DIGEST_DEL_PRIMER_REGISTRO_2026_09_06,
            "el digest vigente ha vuelto a ser el del primer registro: o la "
            "re-firma del 2026-09-13 se ha revertido, o las dos constantes de "
            "arriba se han igualado y esta prueba ya no distingue nada")

    #: La regla de cierre EXACTAMENTE como se registró el 2026-09-06, en su
    #: forma canónica JCS — la misma con la que se calcula el digest. Escrita
    #: a mano y entera: el sentido de tenerla aquí es poder contrastar el
    #: fichero con algo que NO salga del fichero.
    REGLA_DE_CIERRE_REGISTRADA_JCS = (
        b'{"definicion_de_mejor":"el motor con mejor media de los ajustes en ESE dataset,'
        b' excluido el baseline dummy; un fallo (timeout/crash) cuenta como dataset perdido'
        b' para ese motor","fraccion_minima":0.8,"metrica_por_tarea":{"binary_classification'
        b'":"AUROC","multiclass_classification":"accuracy_o_f1_macro","regression":"R2"},'
        b'"puntos":2}')

    def test_la_REGLA_DE_CIERRE_no_se_movio_en_la_re_firma(self):
        """**La prueba que hace legítima la re-firma del 2026-09-13**, y la
        primera que alguien va a querer mirar dentro de seis meses.

        Un digest que cambia no dice QUÉ cambió. Lo que hace que corregir el
        catálogo sea una corrección y no un amaño es que lo que decide quién
        gana siga siendo, letra por letra, lo que se fijó ANTES de medir:
        2,0 puntos, 0,80 de fracción, esas tres métricas por tarea y esa
        definición de «mejor», con su cláusula de que un fallo cuenta como
        dataset perdido.

        Se compara en JCS —los mismos bytes que entran en el sha256—, no
        `assertEqual` de diccionarios: así también caza un 2,0 que se vuelva
        2,00001, un orden distinto o un espacio de más en la definición. Y el
        literal está ESCRITO, no leído de ningún sitio: contrastar el fichero
        consigo mismo es lo que ya hacía `test_el_digest_guardado_coincide…`,
        y la auditoría demostró que eso lo pasa cualquiera que recalcule.
        """
        from matrixai.estudio.validacion import jcs_bytes
        self.assertEqual(
            jcs_bytes(self.protocolo.regla_de_cierre.a_json()),
            self.REGLA_DE_CIERRE_REGISTRADA_JCS,
            "LA REGLA DE CIERRE SE HA MOVIDO. Ninguna corrección del catálogo "
            "ni de la reserva de recursos toca esto: si esta prueba está roja, "
            "lo que ha cambiado es el listón, y eso no se re-firma sin una "
            "conversación aparte")

    def test_el_veredicto_NO_se_movio_con_la_re_firma(self):
        """La otra mitad de la legitimidad: la regla no se movió Y aplicarla
        sigue dando lo mismo. Se re-deriva sobre la EVIDENCIA de los 720
        intentos, que no se ha tocado — 10 de 12, 0,833, CUMPLE.

        Vale la pena que esté aquí y no solo en el contrato: si una corrección
        del catálogo hubiera movido el veredicto, querría decir que el
        catálogo entraba en la cuenta, y entonces «corregir un dato falso»
        habría sido cambiar quién gana."""
        import json
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
        crudo = json.loads(
            (raiz / "pasada_exploratoria_101_c3_remedida_20260913.json")
            .read_text(encoding="utf-8"))
        r = aplicar_regla_de_cierre(crudo["resultados"], self.protocolo.regla_de_cierre,
                                    motor="lightgbm")
        self.assertEqual((r["cumplidos"], r["datasets"]), (10, 12))
        self.assertTrue(r["cumple_la_regla"])
        # La mitad positiva sola la pasaría un motor que ganara siempre: los
        # otros dos tienen que seguir sin cumplir, y por los mismos números.
        self.assertEqual(
            aplicar_regla_de_cierre(crudo["resultados"], self.protocolo.regla_de_cierre,
                                    motor="sklearn.lineal")["cumplidos"], 7)
        self.assertEqual(
            aplicar_regla_de_cierre(crudo["resultados"], self.protocolo.regla_de_cierre,
                                    motor="matrixai.dense.torch_cpu")["cumplidos"], 4)

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
        """Las cuatro cuentas del anexo C §2.2, literal: «≥ 10 con faltantes
        (> 1 % de celdas), **≥ 12 con categóricas (≥ 5 con alguna de
        cardinalidad ≥ 50)**, ≥ 8 desbalanceados (minoritaria ≤ 10 %), ≥ 6 con
        solo numéricas».

        **La del paréntesis no se comprobaba, y la de fuera se comprobaba con
        el campo equivocado.** Hasta el 2026-09-13 esta prueba pedía «≥ 12 con
        `alta_cardinalidad`» — el 12 es el de «con categóricas», y
        `alta_cardinalidad` no medía cardinalidad sino proporción de columnas
        categóricas. Fundidas las dos mitades, el resultado pasaba en verde
        con `KDDCup09_appetency` (15.415 niveles) marcado como de baja
        cardinalidad. Ahora van por separado y se comprueban **las dos**:
        aquí se exige MÁS que antes, no menos.

        «Con categóricas» se cuenta como `not solo_numericas` y no con un
        campo nuevo: serían dos sitios declarando lo mismo."""
        ds = self.protocolo.datasets
        self.assertGreaterEqual(sum(1 for d in ds if d.tiene_faltantes), 10)
        self.assertGreaterEqual(sum(1 for d in ds if not d.solo_numericas), 12)
        self.assertGreaterEqual(sum(1 for d in ds if d.alta_cardinalidad), 5)
        self.assertGreaterEqual(sum(1 for d in ds if d.desbalanceado), 8)
        self.assertGreaterEqual(sum(1 for d in ds if d.solo_numericas), 6)

    def test_la_cobertura_se_cumple_con_los_numeros_EXACTOS_que_se_midieron(self):
        """Un `assertGreaterEqual` no distingue 12 de 40, así que tampoco
        notaría que el catálogo se ha vuelto a llenar de `true` falsos. Los
        números medidos el 2026-09-13 sobre los cuarenta ARFF sellados son
        estos, y si cambian hay que volver a medir y decirlo, no ajustarlos."""
        ds = self.protocolo.datasets
        self.assertEqual(sum(1 for d in ds if not d.solo_numericas), 15)
        self.assertEqual(sum(1 for d in ds if d.alta_cardinalidad), 6)
        self.assertEqual(
            sorted(d.nombre for d in ds if d.alta_cardinalidad),
            ["Allstate_Claims_Severity", "Amazon_employee_access", "KDDCup09_appetency",
             "house_sales", "okcupid-stem", "splice"])

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

    def test_EL_MARGEN_dice_cuantos_datasets_puede_perder_sin_bajarse(self):
        """Lo destapó un sabotaje VERDE el 2026-09-14.

        `datasets_que_puede_perder_sin_incumplir` es el número que dice si un
        veredicto que cumple está holgado o en el borde, y **ningún test
        tocaba el código que lo calcula**: inflarlo en uno
        (`margen + 1`) dejaba las 21 pruebas de
        `test_c101_c3_alcance_declarado.py` en verde, porque ésas auditan el
        JSON ya escrito. *Probar el artefacto no es probar el código que lo
        produce.*

        Las DOS mitades, que es lo que hace que un margen inflado se caiga:
        con el margen declarado todavía cumple, y **con uno más, no**.
        """
        regla = self._regla()
        # 10 de 10: puede perder dos y seguir en 0,8 exacto; tres, no.
        datos = []
        for i in range(10):
            datos += [self._r(f"d{i}", "a", 0.90), self._r(f"d{i}", "b", 0.905)]
        r = aplicar_regla_de_cierre(datos, regla, motor="a")
        self.assertTrue(r["cumple_la_regla"])
        self.assertEqual(r["cumplidos"], 10)
        margen = r["datasets_que_puede_perder_sin_incumplir"]
        self.assertEqual(margen, 2)
        self.assertGreaterEqual((r["cumplidos"] - margen) / r["datasets"], regla.fraccion_minima)
        self.assertLess((r["cumplidos"] - margen - 1) / r["datasets"], regla.fraccion_minima)

    def test_EL_MARGEN_es_CERO_cuando_se_cumple_justo_en_el_borde(self):
        """8 de 10 es 0,80 clavado: cumple y no puede perder ni uno. Un margen
        que nunca da cero no distinguiría «holgado» de «en el borde», que es
        justo para lo que existe."""
        datos = []
        for i in range(8):
            datos += [self._r(f"d{i}", "a", 0.90), self._r(f"d{i}", "b", 0.905)]
        for i in range(8, 10):
            datos += [self._r(f"d{i}", "a", 0.90), self._r(f"d{i}", "b", 0.95)]
        r = aplicar_regla_de_cierre(datos, self._regla(), motor="a")
        self.assertTrue(r["cumple_la_regla"])
        self.assertEqual((r["cumplidos"], r["datasets"]), (8, 10))
        self.assertEqual(r["datasets_que_puede_perder_sin_incumplir"], 0)

    def test_el_que_NO_cumple_no_tiene_margen_y_dice_None_no_cero(self):
        """`None` no es `0`: cero sería «cumple justo en el borde», y éste no
        cumple. Un valor ausente no es un cero."""
        datos = []
        for i in range(10):
            datos += [self._r(f"d{i}", "a", 0.90), self._r(f"d{i}", "b", 0.95)]
        r = aplicar_regla_de_cierre(datos, self._regla(), motor="a")
        self.assertFalse(r["cumple_la_regla"])
        self.assertIsNone(r["datasets_que_puede_perder_sin_incumplir"])

    def test_sobre_la_pasada_CONTAMINADA_del_07_09_daba_9_12_dato_HISTORICO(self):
        """EL DATO HISTÓRICO, y solo eso — **no el resultado vigente**.

        Este test se llamaba «sobre los datos REALES de 101-C3 lightgbm NO
        llega» y su docstring empezaba por «el resultado que importa». Lo era
        el 2026-09-11: con la pasada del 07-09, la regla pre-registrada daba
        9/12 = 0,75 < 0,80 y la cartera se dejó VACÍA a propósito, porque
        mover el listón después de ver el número es justo lo que una regla
        pre-registrada existe para impedir.

        El 2026-09-12 se descubrió que ese 0,750 estaba CONTAMINADO en las dos
        direcciones por tres defectos de cableado —un dataset perdido por un
        fallo cuya distancia real (1,14) caía DENTRO del margen, y dos
        victorias contra un rival que no llegaba a competir—, se repararon y
        se repitió la medición. El resultado vigente es el de la re-medición:
        lo comprueba `test_c102_c1_evidencia_de_la_cartera.py`, que re-deriva
        la regla sobre `..._remedida_20260912.json` y lo contrasta con lo que
        la entrada de cartera afirma.

        Se conserva porque el dato histórico SÍ vale: es la prueba de que el
        listón no se movió, de que la misma regla aplicada a una medición
        peor daba peor, y de que el 9/12 no era una invención. Lo que no vale
        es seguir presentándolo como «el resultado que importa»: una nota
        vieja miente igual que un dato falso."""
        import json
        from pathlib import Path
        # Sin `skipTest`: los dos JSON están commiteados. Un salto silencioso
        # ante un fichero que falta es un banco de pruebas sin dientes.
        raiz = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
        crudo = json.loads((raiz / "pasada_exploratoria_101_c3_resultado.json")
                           .read_text(encoding="utf-8"))
        registros = crudo["resultados"] if isinstance(crudo, dict) and "resultados" in crudo else crudo
        protocolo = ProtocoloExploratorio.cargar(str(raiz / "protocolo_exploratorio.json"))
        r = aplicar_regla_de_cierre(registros, protocolo.regla_de_cierre, motor="lightgbm")
        self.assertEqual((r["cumplidos"], r["datasets"]), (9, 12))
        self.assertFalse(r["cumple_la_regla"])

        # LA OTRA MITAD, que es lo que impide volver a confundir las dos
        # pasadas: sobre la re-medición limpia la MISMA regla, sin tocarla,
        # da otra cosa. Si estas dos afirmaciones se cruzaran, este test
        # volvería a decir del JSON vigente lo que solo vale del viejo.
        crudo_limpio = json.loads((raiz / "pasada_exploratoria_101_c3_remedida_20260913.json")
                                  .read_text(encoding="utf-8"))
        limpio = aplicar_regla_de_cierre(crudo_limpio["resultados"],
                                         protocolo.regla_de_cierre, motor="lightgbm")
        self.assertGreater(limpio["cumplidos"], r["cumplidos"])
        self.assertTrue(limpio["cumple_la_regla"],
                        "la re-medición limpia es la vigente: si dejara de cumplir, "
                        "la entrada de cartera de matrixai_engines estaría afirmando "
                        "algo que ya no se sostiene")


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

    def test_el_speedup_NUNCA_baja_de_uno(self):
        """Re-auditoría del 2026-09-12. El speedup se define SOBRE UN HILO, así
        que nada puede rendir menos que la unidad con la que se mide.

        Sin el suelo, `speedup_medido(1, cpus=1)` daba **0,5863**: el modelo
        afirmaba que un hilo rinde menos que un hilo. Se alcanza de verdad en
        un `docker --cpus=1`, que es justo lo que `cpus_disponibles()` existe
        para detectar. Iba en dirección conservadora —inflaba las horas— pero
        una cifra imposible en un informe la acaba copiando alguien.
        """
        from benchmarks.fase0.protocolo import speedup_medido
        for hilos, cpus in ((1, 1), (4, 1), (1, 2), (1, 8), (100, 1)):
            self.assertGreaterEqual(speedup_medido(hilos, cpus=cpus), 1.0,
                                    f"{hilos} hilos sobre {cpus} CPUs")

    def test_pero_con_HOLGURA_sigue_subiendo_de_verdad(self):
        """La otra mitad: sin ella, el suelo lo pasaría una versión que
        devuelve 1,0 siempre y estrangularía la pasada."""
        from benchmarks.fase0.protocolo import speedup_medido
        self.assertGreater(speedup_medido(4, cpus=8), 3.0)
        self.assertGreater(speedup_medido(8, cpus=8), 4.0)

    def test_cpus_disponibles_USA_de_verdad_la_cuota_de_cgroup(self):
        """Re-auditoría del 2026-09-12: **un sabotaje salió VERDE aquí.**
        Neutralicé la rama de cgroup dentro de `cpus_disponibles()` —borrando
        la llamada y el `append`— y los 81 tests siguieron pasando.

        El test de al lado se llama «la cuota de cgroup manda cuando existe»
        pero solo asierta sobre el PARSEADOR con un fichero de prueba: nunca
        llama a `cpus_disponibles()`. El nombre declaraba un cableado que la
        prueba no medía, que es la forma más cara de banco sin dientes —
        parece cubierto y no lo está.

        Y no es teórico: `docker --cpus=N` es exactamente cómo las reglas de
        esta casa mandan correr las suites, y esa rama es la única que impide
        que dentro de un contenedor se reserve por las CPUs del host.
        """
        from unittest.mock import patch  # noqa: PLC0415

        from benchmarks.fase0 import protocolo  # noqa: PLC0415
        with patch.object(protocolo, "_cuota_de_cgroup", return_value=2):
            self.assertEqual(protocolo.cpus_disponibles(), 2)

    def test_y_SIN_cuota_no_se_inventa_un_tope(self):
        """La otra mitad: sin ella, la de arriba la pasaría una versión que
        devuelve 2 siempre. Sin cuota manda la afinidad, que en esta máquina
        es mayor que 2."""
        from unittest.mock import patch  # noqa: PLC0415

        from benchmarks.fase0 import protocolo  # noqa: PLC0415
        with patch.object(protocolo, "_cuota_de_cgroup", return_value=None):
            sin_cuota = protocolo.cpus_disponibles()
        self.assertGreater(sin_cuota, 2)
        self.assertEqual(sin_cuota, len(os.sched_getaffinity(0)))

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
    """El protocolo REAL, con los números de la reserva. Era la guarda de la
    DEUDA (24 hilos sobre 8 CPUs); desde la re-firma del 2026-09-13 es la
    guarda de la REPARACIÓN: si alguien vuelve a poner 6 procesos x 4 hilos
    creyendo que caben, esto lo dice con nombres y números."""

    @classmethod
    def setUpClass(cls):
        cls.protocolo = ProtocoloExploratorio.cargar(_RUTA_PROTOCOLO_REAL)
        cls.payload = json.loads(_RUTA_PROTOCOLO_REAL.read_text(encoding="utf-8"))
        cls.coste = calcular_coste(cls.protocolo, cpus=8)

    def test_el_protocolo_reserva_8_hilos_sobre_8_cpus_y_no_24(self):
        """El caso de la deuda, ya cerrado. 4 x 6 = 24 sobre 8 era TRIPLE
        reserva, y este servidor se ha caído dos veces por eso mismo."""
        self.assertEqual(self.protocolo.presupuesto.hilos_reservados, 8)
        self.assertEqual(self.coste.sobre_reserva, 0)

    def test_los_procesos_son_los_que_RESERVA_SEGURA_dice_no_un_numero_a_ojo(self):
        """«Dilo medido»: el 2 del protocolo tiene que ser exactamente lo que
        `reserva_segura` devuelve para sus propios hilos sobre la máquina que
        el propio protocolo declara. Las CPUs se toman de
        `recursos_declarados`, no de la máquina donde corra la suite: si no,
        esto se pondría rojo en cualquier ordenador con otro número de CPUs y
        dejaría de decir nada sobre el documento."""
        cpus = self.payload["recursos_declarados"]["cpu_logicas"]
        presupuesto = self.protocolo.presupuesto
        self.assertEqual(presupuesto.procesos_en_paralelo,
                         reserva_segura(presupuesto.hilos, cpus=cpus))
        self.assertEqual(presupuesto.procesos_en_paralelo,
                         presupuesto.procesos_que_caben(cpus=cpus))

    def test_LA_OTRA_MITAD_la_reserva_no_se_ha_estrangulado(self):
        """Un techo que reservase 1 proceso también daría sobre-reserva 0 y
        pasaría por reparación, haciendo la pasada eterna. La reserva tiene
        que USAR las CPUs que hay: los 8 hilos son los 8 de la máquina."""
        cpus = self.payload["recursos_declarados"]["cpu_logicas"]
        self.assertEqual(self.protocolo.presupuesto.hilos_reservados, cpus)
        self.assertGreater(self.protocolo.presupuesto.procesos_en_paralelo, 1)

    def test_el_presupuesto_y_los_recursos_declarados_NO_divergen(self):
        """El mismo par de números vive en `presupuesto` y en
        `recursos_declarados`, y dos sitios declarando lo mismo acaban
        divergiendo — de hecho la re-firma tuvo que tocar los dos. Si alguien
        cambia uno solo, esto lo caza."""
        recursos = self.payload["recursos_declarados"]
        self.assertEqual(recursos["hilos_por_proceso"], self.protocolo.presupuesto.hilos)
        self.assertEqual(recursos["procesos_en_paralelo"],
                         self.protocolo.presupuesto.procesos_en_paralelo)

    def test_cpu_fisicas_declaradas_son_las_MEDIDAS_no_las_logicas(self):
        """Decía `cpu_fisicas: 8` y son 4: 8 son las LÓGICAS, con SMT x2. No
        es cosmética — es el número con el que uno decide cuántos hilos caben.

        Solo se mide si la suite corre en la máquina que el protocolo
        describe; en otra se salta diciéndolo, porque comparar los núcleos de
        OTRO ordenador con los declarados aquí no probaría nada."""
        recursos = self.payload["recursos_declarados"]
        medidos = nucleos_fisicos()
        if medidos is None or cpus_disponibles() != recursos["cpu_logicas"]:
            self.skipTest(f"esta no es la máquina declarada "
                          f"({cpus_disponibles()} CPUs, físicos={medidos}); "
                          f"el protocolo describe {recursos['cpu_logicas']} lógicas")
        self.assertEqual(recursos["cpu_fisicas"], medidos)
        self.assertLess(recursos["cpu_fisicas"], recursos["cpu_logicas"],
                        "si vuelven a ser iguales, o la máquina cambió o alguien "
                        "ha copiado las lógicas encima de las físicas otra vez")

    def test_la_cota_lineal_ya_no_promete_algo_INALCANZABLE(self):
        """El cambio que de verdad importa de la re-firma, y no se ve en la
        sobre-reserva. Con 6 procesos el lineal daba 74,93 h y el suelo MEDIDO
        eran 95,86: el número publicado exigía un 6x que esta máquina no puede
        dar, o sea que era imposible por construcción. Con 2 procesos el
        lineal (224,79 h) va POR ENCIMA del suelo medido, que es lo que tiene
        que pasar con una cota de peor caso."""
        self.assertAlmostEqual(self.coste.horas_reloj_peor_caso_con_paralelismo, 224.79, places=2)
        self.assertGreater(self.coste.horas_reloj_peor_caso_con_paralelismo,
                           self.coste.horas_reloj_suelo_medido,
                           "la cota lineal vuelve a prometer menos horas de las que el "
                           "escalado medido permite: es una promesa que no se puede cumplir")

    def test_el_suelo_medido_es_9586_horas_y_NO_lo_movio_la_re_firma(self):
        """Número NUEVO en su día, no sustituto: con el techo medido 4,69x (no
        6x) las 449,58 h secuenciales dan 95,86 h.

        **Y bajar de 24 hilos a 8 no lo empeoró ni una hora**, que es lo que
        hace la corrección de la reserva indolora: 24 hilos sobre 8 CPUs
        rendían exactamente el mismo 4,69x que 8. Se reservaba el triple para
        no sacar nada."""
        self.assertAlmostEqual(self.coste.horas_reloj_suelo_medido, 95.86, places=2)
        self.assertAlmostEqual(self.coste.speedup_efectivo_medido, 4.69, places=2)
        self.assertAlmostEqual(speedup_medido(24, cpus=8), speedup_medido(8, cpus=8), places=4)

    def test_las_ejecuciones_NO_las_movio_la_re_firma(self):
        """La reserva reparte el MISMO trabajo, no menos: 6.500 ajustes antes
        y 6.500 después. Si esto bajara, la re-firma habría recortado la
        pasada en vez de repartirla."""
        self.assertEqual(self.coste.total_ejecuciones, 6500)
        self.assertAlmostEqual(self.coste.horas_reloj_peor_caso_secuencial, 449.58, places=2)


# ---------------------------------------------------------------------------
# Cardinalidad — la corrección del catálogo de la re-firma del 2026-09-13
# ---------------------------------------------------------------------------
#: Los ARFF sellados viven FUERA del repo (invariante 4 del runbook de
#: publicación: nada de blobs grandes en git), así que las pruebas que los
#: miden se saltan donde no estén — DICIÉNDOLO, no en silencio.
_ARFF_SELLADOS = Path.home() / "fase0_openml_datos" / "arff"


class CardinalidadNominalDeclaradaTest(unittest.TestCase):
    """La función que MIDE, con ficheros escritos aquí: si se prueba solo
    contra los cuarenta sellados, un fallo de parseo y un catálogo equivocado
    se tapan mutuamente."""

    def _arff(self, texto: str) -> Path:
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False, encoding="utf-8")
        f.write(texto)
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return Path(f.name)

    def test_cuenta_los_niveles_de_cada_columna_nominal(self):
        ruta = self._arff("@relation r\n@attribute a {x,y,z}\n@attribute b REAL\n"
                          "@attribute c {p,q}\n@data\nx,1.0,p\n")
        self.assertEqual(cardinalidad_nominal_declarada(ruta), {"a": 3, "c": 2})

    def test_una_columna_NUMERICA_no_aparece_con_cero(self):
        """Un valor ausente no es un cero: si las numéricas entraran como 0,
        `max(...)` sobre un dataset sin nominales parecería una medición en
        vez de una ausencia, y `min(...)` mentiría del todo."""
        ruta = self._arff("@relation r\n@attribute a REAL\n@attribute b INTEGER\n@data\n1,2\n")
        self.assertEqual(cardinalidad_nominal_declarada(ruta), {})

    def test_la_columna_OBJETIVO_no_cuenta(self):
        """La cardinalidad que importa es la de los PREDICTORES: un objetivo
        multiclase de 26 letras no es un problema de one-hot de entrada."""
        ruta = self._arff("@relation r\n@attribute a {x,y}\n@attribute clase {p,q,r}\n@data\nx,p\n")
        self.assertEqual(cardinalidad_nominal_declarada(ruta, "clase"), {"a": 2})
        self.assertEqual(cardinalidad_nominal_declarada(ruta), {"a": 2, "clase": 3})

    def test_una_coma_DENTRO_de_un_nivel_entrecomillado_no_separa(self):
        """`house_prices_nominal` y `diamonds` traen niveles con espacios y
        comillas. Contar comas a pelo partiría un nivel en dos y la
        cardinalidad saldría inflada."""
        ruta = self._arff("@relation r\n@attribute a {Fair, Good, 'Very Good', \"A, B\"}\n"
                          "@data\nFair\n")
        self.assertEqual(cardinalidad_nominal_declarada(ruta), {"a": 4})

    def test_una_declaracion_partida_en_VARIAS_lineas_se_cierra_entera(self):
        """Ninguno de los 40 sellados la parte —comprobado—, pero un ARFF
        futuro sí puede, y cerrar en falso contaría de menos justo en el
        dataset más cardinal. Esta es la prueba de esa línea."""
        ruta = self._arff("@relation r\n@attribute a {uno,dos,\n  tres,cuatro,\n  cinco}\n"
                          "@attribute b REAL\n@data\nuno,1\n")
        self.assertEqual(cardinalidad_nominal_declarada(ruta), {"a": 5, })

    def test_no_lee_mas_alla_de_arroba_data(self):
        """Solo la cabecera: 254 MB de datos no hacen falta para contar lo que
        la cabecera ya dice, y una fila que empezara por `@attribute` no es
        una declaración."""
        ruta = self._arff("@relation r\n@attribute a {x,y}\n@data\n"
                          "@attribute colado {1,2,3,4,5,6,7,8,9}\n")
        self.assertEqual(cardinalidad_nominal_declarada(ruta), {"a": 2})


class DatasetRegistradoCardinalidadTest(unittest.TestCase):
    """El invariante del esquema: el número medido y el booleano no se pueden
    separar. Antes el booleano viajaba solo y nadie podía contrastarlo."""

    def test_el_booleano_que_CONTRADICE_al_numero_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            _dataset(max_cardinalidad_nominal=15415, alta_cardinalidad=False)
        with self.assertRaises(ProtocoloError):
            _dataset(max_cardinalidad_nominal=3, alta_cardinalidad=True)

    def test_LA_OTRA_MITAD_los_que_cuadran_se_aceptan_en_los_dos_lados(self):
        """Un aserto de rechazo solo lo pasa un esquema que rechace TODO."""
        self.assertTrue(_dataset(max_cardinalidad_nominal=15415,
                                 alta_cardinalidad=True).alta_cardinalidad)
        self.assertFalse(_dataset(max_cardinalidad_nominal=3,
                                  alta_cardinalidad=False).alta_cardinalidad)

    def test_el_umbral_es_EL_DEL_ANEXO_y_es_inclusivo(self):
        """«≥ 50», anexo C §2.2 — escrito antes de medir nada. En el borde:
        49 no, 50 sí."""
        self.assertEqual(UMBRAL_ALTA_CARDINALIDAD, 50)
        self.assertFalse(_dataset(max_cardinalidad_nominal=49,
                                  alta_cardinalidad=False).alta_cardinalidad)
        self.assertTrue(_dataset(max_cardinalidad_nominal=50,
                                 alta_cardinalidad=True).alta_cardinalidad)
        with self.assertRaises(ProtocoloError):
            _dataset(max_cardinalidad_nominal=50, alta_cardinalidad=False)

    def test_un_catalogo_SIN_el_campo_revienta_en_vez_de_valer_cero(self):
        """Un valor ausente no es un cero. Un catálogo anterior a la re-firma
        no trae `max_cardinalidad_nominal`, y rellenarlo con 0 por defecto lo
        haría pasar por «ninguna columna nominal» — justo la afirmación falsa
        que la re-firma vino a corregir."""
        payload = _dataset().a_json()
        del payload["max_cardinalidad_nominal"]
        with self.assertRaises(KeyError):
            DatasetRegistrado.desde_json(payload)

    def test_el_campo_VIAJA_en_el_json_y_por_tanto_en_el_digest(self):
        """Si se quedara fuera de `a_json()`, el número medido no estaría
        sellado y se podría cambiar sin mover el digest — que es exactamente
        como el dato falso sobrevivió tanto tiempo."""
        self.assertIn("max_cardinalidad_nominal", _dataset().a_json())
        uno = _protocolo_minimo(datasets=(_dataset(max_cardinalidad_nominal=0,
                                                   alta_cardinalidad=False),))
        otro = _protocolo_minimo(datasets=(_dataset(max_cardinalidad_nominal=7,
                                                    alta_cardinalidad=False),))
        self.assertNotEqual(uno.digest(), otro.digest())


@unittest.skipUnless(_ARFF_SELLADOS.is_dir(),
                     f"los ARFF sellados no están en {_ARFF_SELLADOS} (viven fuera del repo)")
class CatalogoContraLosARFFRealesTest(unittest.TestCase):
    """**La prueba que habría cazado el dato falso, y no existía.** El
    catálogo declaraba la cardinalidad y NADA la contrastaba con los ficheros:
    `KDDCup09_appetency` dijo `alta_cardinalidad: false` con 15.415 niveles en
    `Var200` durante una semana, con la suite en verde todo el rato.

    Solo lee cabeceras, así que los cuarenta se miden en menos de un segundo.
    """

    @classmethod
    def setUpClass(cls):
        cls.protocolo = ProtocoloExploratorio.cargar(_RUTA_PROTOCOLO_REAL)

    def test_el_numero_declarado_es_EL_QUE_DICE_EL_FICHERO_en_los_cuarenta(self):
        medidos = 0
        for d in self.protocolo.datasets:
            ruta = _ARFF_SELLADOS / f"{d.data_id}.arff"
            # Sin `continue` silencioso: un directorio a medias no puede pasar
            # por barrido completo. O están los 40, o esto se cae por su nombre.
            self.assertTrue(ruta.is_file(), f"falta el ARFF sellado de {d.nombre}: {ruta}")
            niveles = cardinalidad_nominal_declarada(ruta, d.columna_objetivo)
            self.assertEqual(
                d.max_cardinalidad_nominal, max(niveles.values(), default=0),
                f"{d.nombre} ({d.data_id}): el catálogo declara "
                f"{d.max_cardinalidad_nominal} niveles y el ARFF dice otra cosa")
            medidos += 1
        self.assertEqual(medidos, 40, "se han medido menos de 40: el barrido no barrió")

    def test_los_dos_casos_que_destaparon_el_fallo_por_su_nombre_y_su_numero(self):
        """No «alguno estaba mal»: estos dos, con estos números, medidos sobre
        el ARFF sellado. `KDDCup09_appetency` es el falso negativo más grande
        (decía `false` con 15.415 niveles en `Var200`, y `Var214` declara
        otros 15.415) y `kr-vs-kp` el falso positivo más claro (decía `true`
        siendo 36 columnas categóricas de 3 niveles como mucho)."""
        por_nombre = {d.nombre: d for d in self.protocolo.datasets}
        kdd = cardinalidad_nominal_declarada(_ARFF_SELLADOS / "1111.arff", "APPETENCY")
        self.assertEqual(kdd["Var200"], 15415)
        self.assertEqual(kdd["Var214"], 15415)
        self.assertEqual(por_nombre["KDDCup09_appetency"].max_cardinalidad_nominal, 15415)
        self.assertTrue(por_nombre["KDDCup09_appetency"].alta_cardinalidad)

        krvskp = cardinalidad_nominal_declarada(_ARFF_SELLADOS / "3.arff", "class")
        self.assertEqual(max(krvskp.values()), 3)
        self.assertEqual(len(krvskp), 36)
        self.assertFalse(por_nombre["kr-vs-kp"].alta_cardinalidad)

    def test_pendigits_NO_tiene_ni_una_columna_nominal_y_su_false_era_correcto(self):
        """Medido, no heredado. `pendigits` venía señalado como mal marcado
        junto a `KDDCup09_appetency`, y **no lo estaba en cardinalidad**: sus
        16 predictores son `numeric` y su única columna nominal es el objetivo
        (`class {0..9}`), así que `alta_cardinalidad: false` era y sigue siendo
        lo que el fichero dice. Queda escrito aquí para que nadie lo vuelva a
        «corregir» de memoria."""
        por_nombre = {d.nombre: d for d in self.protocolo.datasets}
        self.assertEqual(
            cardinalidad_nominal_declarada(_ARFF_SELLADOS / "32.arff", "class"), {})
        self.assertEqual(por_nombre["pendigits"].max_cardinalidad_nominal, 0)
        self.assertFalse(por_nombre["pendigits"].alta_cardinalidad)

    def test_los_sha256_de_los_ARFF_SIGUEN_siendo_los_registrados(self):
        """Lo que hace que todo lo de arriba valga: si el fichero medido no es
        el sellado, la medición es de otra cosa. Se comprueban los cuarenta,
        porque la re-firma tocó el catálogo y no puede haber tocado los datos.
        """
        import hashlib
        for d in self.protocolo.datasets:
            ruta = _ARFF_SELLADOS / f"{d.data_id}.arff"
            self.assertEqual(hashlib.sha256(ruta.read_bytes()).hexdigest(), d.sha256_arff,
                            f"{d.nombre}: el ARFF del disco ya no es el que el protocolo registró")


class ConstruirProtocoloDerivaLaCardinalidadTest(unittest.TestCase):
    """Probar el artefacto no es probar el código que lo produce: el JSON
    cuadra con su digest aunque `generar_protocolo.py` vuelva a escribir el
    booleano a mano. Esto prueba el GENERADOR."""

    def _seleccion(self, maximo: int) -> list[dict]:
        # `url` no estaba, y el registro que produce `_candidato` SIEMPRE la
        # trae —`descargar_y_hashear` descarga por ella—, así que el fixture
        # describía un registro que no existe. Se completó el 2026-09-14, al
        # empezar el generador a sacar de ahí el `file_id`: un fixture que
        # miente por omisión deja de avisar justo cuando el productor cambia.
        return [dict(data_id=1, nombre="x", version=1, objetivo="y",
                     url="https://openml.org/data/v1/download/9999/x.arff",
                     n_filas=1000, n_columnas=10, bucket="pequeno",
                     tiene_faltantes=False, max_cardinalidad_nominal=maximo,
                     desbalanceado=False, solo_numericas=False, licencia="Public",
                     sellado=False, binaria=True, multiclase=False)]

    def test_el_booleano_sale_del_numero_medido_en_los_dos_lados(self):
        from benchmarks.fase0.generar_protocolo import construir_protocolo
        hashes = {1: "b" * 64}
        alto = construir_protocolo(self._seleccion(15415), hashes).datasets[0]
        bajo = construir_protocolo(self._seleccion(3), hashes).datasets[0]
        self.assertEqual((alto.max_cardinalidad_nominal, alto.alta_cardinalidad), (15415, True))
        self.assertEqual((bajo.max_cardinalidad_nominal, bajo.alta_cardinalidad), (3, False))

    def test_el_generador_escribe_la_reserva_que_CABE(self):
        """Si el generador se volviera a correr, no puede reintroducir los 6
        procesos: escribiría otra vez 24 hilos sobre 8 CPUs."""
        from benchmarks.fase0.generar_protocolo import construir_protocolo
        presupuesto = construir_protocolo(self._seleccion(3), {1: "b" * 64}).presupuesto
        self.assertEqual(presupuesto.hilos_reservados, 8)
        self.assertEqual(presupuesto.sobre_reserva(cpus=8), 0)

    def test_los_dos_minimos_del_anexo_van_SEPARADOS(self):
        """Eran uno solo (`MINIMO_ALTA_CARDINALIDAD = 12`, el 12 de «con
        categóricas»), y por eso la exigencia del paréntesis —«≥ 5 con alguna
        de cardinalidad ≥ 50»— no se comprobaba nunca."""
        from benchmarks.fase0 import generar_protocolo as gen
        self.assertEqual(gen.MINIMO_CON_CATEGORICAS, 12)
        self.assertEqual(gen.MINIMO_ALTA_CARDINALIDAD, 5)


if __name__ == "__main__":
    unittest.main()
