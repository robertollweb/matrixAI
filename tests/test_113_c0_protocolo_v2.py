# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""113-C0 — el protocolo de Fase 0 v2: la re-firma (P1, P2) y la receta de la
densa (113-C1), registradas ANTES de medir.

Decidido por Roberto el 2026-09-21 (`TASKS.md`, «Re-firma del protocolo de
Fase 0»; contrato `113_LA_RED_DENSA_A_LA_ALTURA_CONTRACT.md`, corte C0):

- **P1 (a)**: en MULTICLASE, `desbalanceado` exige cuota de la clase
  minoritaria `<= 0,2/k` (k = número de clases) — la misma fórmula que en
  binaria (0,2/2 = 10 %). Marca solo `yeast` de los 10 multiclase.
- **P2 (a)**: sustituye 3 datasets, uno por celda (tarea, cubo de tamaño), sin
  tocar ningún `sellado`: entran `dresses-sales`, `kick` y
  `SAT11-HAND-runtime-regression`; salen `breast-w`, `adult` y `elevators`.
  Conserva 40 = 20/10/10 por tarea y 15/15/10 por tamaño.
- **P3**: una sola versión nueva (`113-C0.v2`), que junta la re-firma con la
  receta de la densa del 113-C1 (`Motor.receta`, aditivo, solo en
  `matrixai.dense.torch_cpu`).

Esta prueba tiene DOS mitades, y las dos importan:

1. Que lo REGISTRADO en `protocolo_exploratorio_v2.json` sea lo decidido
   (huella literal, sellados intactos, celdas correctas, cobertura medida
   sobre los ARFF reales).
2. Que el CÓDIGO que lo produjo (`generar_protocolo_v2.py`) siga dando lo
   mismo si se vuelve a correr — «probar el artefacto no es probar el código
   que lo produce»: un artefacto se puede editar a mano sin tocar el
   generador, y un generador se puede romper sin que el artefacto ya escrito
   lo note.
"""
from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
_FASE0 = _RAIZ / "benchmarks" / "fase0"
if str(_FASE0) not in sys.path:
    sys.path.insert(0, str(_FASE0))
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.fase0.protocolo import (  # noqa: E402
    DatasetRegistrado,
    Motor,
    ProtocoloError,
    ProtocoloExploratorio,
)
from benchmarks.fase0 import generar_protocolo_v2 as g2  # noqa: E402
from test_101_c3_faltante_arff_e_identificador import _celdas_interrogante  # noqa: E402

RUTA_V1 = _FASE0 / "protocolo_exploratorio.json"
RUTA_V2 = _FASE0 / "protocolo_exploratorio_v2.json"
DATOS = Path("/home/deployer/fase0_openml_datos/arff")

con_los_arff = pytest.mark.skipif(
    not DATOS.is_dir(), reason="los ARFF del protocolo no están descargados")

#: Los tres que salen (P2a) y los tres que entran, con su celda (tarea, cubo)
#: — la comprobación del corte: "cada entrante cae en la celda del saliente".
SALIENTES = {
    15: ("breast-w", "binary_classification", "pequeno"),
    1590: ("adult", "binary_classification", "grande"),
    216: ("elevators", "regression", "mediano"),
}
ENTRANTES = {
    23381: ("dresses-sales", "binary_classification", "pequeno"),
    41162: ("kick", "binary_classification", "grande"),
    41980: ("SAT11-HAND-runtime-regression", "regression", "mediano"),
}


# ---------------------------------------------------------------------------
# Motor.receta — el esquema del campo aditivo (113-C1), sin fixtures del disco
# ---------------------------------------------------------------------------

class MotorRecetaTest(unittest.TestCase):
    """El esquema, aislado del catálogo real: si esto se rompe, no hace falta
    ARFF ni red para verlo."""

    def _receta(self, **kw):
        base = dict(optimizador="adam", learning_rate=0.001,
                   early_stop={"patience": 10, "metric": "validation_loss"},
                   epochs=50, batch_size=8)
        base.update(kw)
        return base

    def test_motor_sin_receta_sigue_funcionando_igual_que_en_la_v1(self):
        m = Motor(id="lightgbm", configuraciones=2)
        self.assertIsNone(m.receta)
        self.assertEqual(m.a_json(), {"id": "lightgbm", "configuraciones": 2})

    def test_motor_con_la_receta_de_113_c1_se_acepta(self):
        m = Motor(id="matrixai.dense.torch_cpu", configuraciones=2, receta=self._receta())
        self.assertEqual(m.a_json()["receta"]["optimizador"], "adam")
        self.assertEqual(m.a_json()["receta"]["learning_rate"], 0.001)
        self.assertEqual(m.a_json()["receta"]["early_stop"], {"patience": 10, "metric": "validation_loss"})

    def test_receta_con_clave_de_mas_se_rechaza_FALLO_CERRADO(self):
        with self.assertRaises(ProtocoloError):
            Motor(id="x", receta=self._receta(clave_que_no_existe=1))

    def test_receta_a_medias_se_rechaza(self):
        r = self._receta()
        del r["batch_size"]
        with self.assertRaises(ProtocoloError):
            Motor(id="x", receta=r)

    def test_early_stop_con_clave_de_mas_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            Motor(id="x", receta=self._receta(early_stop={"patience": 10, "metric": "x", "extra": 1}))

    def test_learning_rate_no_positivo_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            Motor(id="x", receta=self._receta(learning_rate=0.0))

    def test_patience_no_positiva_se_rechaza(self):
        with self.assertRaises(ProtocoloError):
            Motor(id="x", receta=self._receta(early_stop={"patience": 0, "metric": "x"}))

    def test_roundtrip_json_de_un_motor_con_receta(self):
        m = Motor(id="matrixai.dense.torch_cpu", configuraciones=2, receta=self._receta())
        de_vuelta = Motor(**m.a_json())
        self.assertEqual(de_vuelta, m)


# ---------------------------------------------------------------------------
# La v1 SIGUE SIENDO LA DE ANTES — el ancla que ninguna re-firma toca
# ---------------------------------------------------------------------------

class LaV1NoSeHaMovidoTest(unittest.TestCase):
    """El corte re-firma con un fichero NUEVO. Si esto se pone rojo, lo que ha
    pasado es que alguien tocó `protocolo_exploratorio.json` en vez de
    escribir la v2 — exactamente lo que el criterio de cierre prohíbe."""

    #: El mismo literal que `test_c101_c1_protocolo_fase0.py ::
    #: DIGEST_REGISTRADO_ANTES_DE_MEDIR`. Repetido a propósito: dos pruebas
    #: que dependieran la una de la otra no cazarían un `import` roto.
    DIGEST_V1 = "ea50ca482a815627364b3439bfcf10e78295a13ae5f821219d0fb40329e65301"

    def test_la_v1_sigue_dando_el_mismo_digest(self):
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        self.assertEqual(v1.digest(), self.DIGEST_V1)

    def test_la_v1_sigue_teniendo_los_TRES_que_v2_sustituye(self):
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        ids = {d.data_id for d in v1.datasets}
        self.assertTrue(set(SALIENTES) <= ids)

    def test_la_v1_no_trae_ningun_campo_receta(self):
        payload = json.loads(RUTA_V1.read_text(encoding="utf-8"))
        for m in payload["motores"]:
            self.assertNotIn("receta", m)


# ---------------------------------------------------------------------------
# El protocolo v2 REGISTRADO — 40 datasets, re-firmados el 2026-09-21
# ---------------------------------------------------------------------------

class ProtocoloV2RegistradoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUTA_V2.exists():
            raise unittest.SkipTest("protocolo_exploratorio_v2.json no está escrito todavía")
        cls.protocolo = ProtocoloExploratorio.cargar(RUTA_V2)
        cls.payload = json.loads(RUTA_V2.read_text(encoding="utf-8"))
        cls.por_id = {d.data_id: d for d in cls.protocolo.datasets}

    def test_son_40_datasets(self):
        self.assertEqual(len(self.protocolo.datasets), 40)

    def test_version_protocolo_es_v2(self):
        self.assertEqual(self.protocolo.version_protocolo, "113-C0.v2")

    def test_ningun_data_id_repetido(self):
        ids = [d.data_id for d in self.protocolo.datasets]
        self.assertEqual(len(ids), len(set(ids)))

    def test_el_digest_guardado_coincide_con_recalcularlo(self):
        self.assertEqual(self.payload["digest_sha256"], self.protocolo.digest())

    # --- P3: una sola versión, no distinta en todo -------------------------

    def test_particion_presupuesto_y_regla_de_cierre_NO_cambiaron(self):
        """P3 dice «una sola versión nueva», no «una versión distinta en
        todo»: lo que la decisión del 21-09 no tocó, se lee igual que en la
        v1, campo a campo."""
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        self.assertEqual(self.protocolo.particion.a_json(), v1.particion.a_json())
        self.assertEqual(self.protocolo.presupuesto.a_json(), v1.presupuesto.a_json())
        self.assertEqual(self.protocolo.regla_de_cierre.a_json(), v1.regla_de_cierre.a_json())

    # --- P2(a): las tres sustituciones, una por celda -----------------------

    def test_los_TRES_que_salen_YA_NO_ESTAN(self):
        ids = {d.data_id for d in self.protocolo.datasets}
        for data_id, (nombre, _tarea, _cubo) in SALIENTES.items():
            self.assertNotIn(data_id, ids, f"{nombre} ({data_id}) debería haber salido")

    def test_los_TRES_que_entran_ESTAN(self):
        for data_id, (nombre, tarea, cubo) in ENTRANTES.items():
            self.assertIn(data_id, self.por_id, f"falta {nombre} ({data_id})")
            ds = self.por_id[data_id]
            self.assertEqual(ds.nombre, nombre)
            self.assertEqual(ds.tarea, tarea)
            self.assertEqual(ds.cubo_de_tamano, cubo)
            self.assertFalse(ds.sellado, f"{nombre} no debería quedar sellado por la re-firma")

    def test_cada_entrante_cae_en_la_MISMA_celda_que_su_saliente(self):
        """"conserva 40 = 20/10/10 y 15/15/10 por tamaño": no basta con que
        entren y salgan tres, tienen que ser la MISMA celda (tarea, cubo)."""
        salientes_por_celda = {(t, c): n for n, t, c in SALIENTES.values()}
        entrantes_por_celda = {(t, c): n for n, t, c in ENTRANTES.values()}
        self.assertEqual(set(salientes_por_celda), set(entrantes_por_celda),
                         "las celdas de lo que sale y lo que entra no coinciden 1 a 1")

    def test_20_10_10_por_tarea(self):
        tareas = Counter(d.tarea for d in self.protocolo.datasets)
        self.assertEqual(tareas["binary_classification"], 20)
        self.assertEqual(tareas["multiclass_classification"], 10)
        self.assertEqual(tareas["regression"], 10)

    def test_15_15_10_por_cubo(self):
        cubos = Counter(d.cubo_de_tamano for d in self.protocolo.datasets)
        self.assertEqual(cubos["pequeno"], 15)
        self.assertEqual(cubos["mediano"], 15)
        self.assertEqual(cubos["grande"], 10)

    # --- Los 8 sellados: no se re-sellan; solo sigue la regla su campo derivado

    def test_los_8_sellados_son_IDENTICOS_a_la_v1_salvo_el_campo_derivado_de_P1(self):
        """«Los sellados no se re-sellan» es su SELECCIÓN (corrección del
        supervisor, 2026-09-21): siguen siendo los mismos 8, con todos sus
        datos byte a byte. Lo único que puede cambiar es `desbalanceado`, que
        DERIVA de la regla P1 (una sola fórmula, decisión de Roberto)."""
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        sellados_v1 = {d.data_id: d for d in v1.datasets if d.sellado}
        sellados_v2 = {d.data_id: d for d in self.protocolo.datasets if d.sellado}
        self.assertEqual(set(sellados_v1), set(sellados_v2), "el conjunto de sellados cambió")
        self.assertEqual(len(sellados_v2), 8)
        for data_id, ds_v1 in sellados_v1.items():
            a_v1 = {k: v for k, v in ds_v1.a_json().items() if k != "desbalanceado"}
            a_v2 = {k: v for k, v in sellados_v2[data_id].a_json().items() if k != "desbalanceado"}
            self.assertEqual(a_v1, a_v2, f"el sellado {ds_v1.nombre} ({data_id}) cambió en la v2")

    def test_ninguno_de_los_3_salientes_estaba_sellado(self):
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        for d in v1.datasets:
            if d.data_id in SALIENTES:
                self.assertFalse(d.sellado, f"{d.nombre} SÍ estaba sellado: P2 no podía sustituirlo")

    # --- Los 37 no tocados (aparte de `desbalanceado`) siguen igual ---------

    def test_los_37_NO_SUSTITUIDOS_conservan_todo_menos_desbalanceado(self):
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        v1_por_id = {d.data_id: d for d in v1.datasets}
        for data_id, ds_v2 in self.por_id.items():
            if data_id in ENTRANTES:
                continue
            ds_v1 = v1_por_id[data_id]
            a_v1, a_v2 = ds_v1.a_json(), ds_v2.a_json()
            # Comparación campo a campo, SALVO desbalanceado (que P1 puede mover
            # solo si es multiclase): así un cambio accidental en cualquier OTRO
            # campo de un no-sustituido revienta aquí.
            sin_desbalance_v1 = {k: v for k, v in a_v1.items() if k != "desbalanceado"}
            sin_desbalance_v2 = {k: v for k, v in a_v2.items() if k != "desbalanceado"}
            self.assertEqual(sin_desbalance_v1, sin_desbalance_v2,
                            f"{ds_v1.nombre} ({data_id}) cambió un campo que no era desbalanceado")

    # --- P1(a): multiclase, cuota <= 0,2/k -----------------------------------

    def test_P1_marca_solo_yeast(self):
        """P1(a) aplicado a los 10 multiclase, sellados incluidos, marca SOLO
        `yeast` (cuota*k = 0,034): exactamente el resultado de la decisión de
        Roberto («una sola fórmula… marca solo yeast»). `letter`, sellado,
        pasa de `True` (regla vieja) a `False` (cuota*k = 0,954)."""
        mc = [d for d in self.protocolo.datasets if d.tarea == "multiclass_classification"]
        self.assertEqual(len(mc), 10)
        marcados = sorted(d.nombre for d in mc if d.desbalanceado)
        self.assertEqual(marcados, ["yeast"])

    def test_P1_no_cambia_la_binaria(self):
        """"0,2/2 = 10 %": la fórmula de la binaria no se toca, así que su
        cuenta de desbalanceados tiene que seguir igual que en la v1."""
        v1 = ProtocoloExploratorio.cargar(RUTA_V1)
        bin_v1 = sum(1 for d in v1.datasets if d.tarea == "binary_classification" and d.desbalanceado)
        bin_v2 = sum(1 for d in self.protocolo.datasets
                    if d.tarea == "binary_classification" and d.desbalanceado)
        self.assertEqual(bin_v1, 9)
        self.assertEqual(bin_v2, 9)

    def test_el_minimo_de_8_desbalanceados_del_anexo_sigue_cumplido(self):
        """9 binarios + 1 multiclase (`yeast`) = 10, por encima del mínimo 8
        del anexo C §2.2 — la re-firma no rompe la cobertura obligatoria."""
        self.assertEqual(sum(1 for d in self.protocolo.datasets if d.desbalanceado), 10)

    # --- Cobertura: > 1 % de celdas, con categóricas, solo numéricas --------

    def test_con_categoricas_sube_de_15_a_17(self):
        self.assertEqual(sum(1 for d in self.protocolo.datasets if not d.solo_numericas), 17)

    def test_solo_numericas_baja_de_25_a_23(self):
        self.assertEqual(sum(1 for d in self.protocolo.datasets if d.solo_numericas), 23)

    @con_los_arff
    def test_faltantes_mayor_1pct_de_celdas_es_10(self):
        """La lectura ESTRICTA del anexo C §2.2 («faltantes > 1 % de celdas»),
        medida sobre los ARFF reales con la misma cuenta que
        `scratchpad/refirma/cobertura_actual.py` de deployer-02 — no la
        cuenta barata `tiene_faltantes` (cualquier `?`) que ya se comprueba
        aparte."""
        por_encima = []
        for d in self.protocolo.datasets:
            ruta = DATOS / f"{d.data_id}.arff"
            celdas = _celdas_interrogante(ruta)
            fraccion = celdas / (d.n_filas * (d.n_columnas + 1))
            if fraccion > 0.01:
                por_encima.append(d.nombre)
        self.assertEqual(len(por_encima), 10, sorted(por_encima))

    # --- 113-C1: la receta de la densa, declarada ANTES de medir ------------

    def test_siete_motores_y_SOLO_la_densa_trae_receta(self):
        self.assertEqual(len(self.protocolo.motores), 7)
        con_receta = [m.id for m in self.protocolo.motores if m.receta is not None]
        self.assertEqual(con_receta, ["matrixai.dense.torch_cpu"])

    def test_la_receta_de_la_densa_es_la_del_113_c1(self):
        densa = next(m for m in self.protocolo.motores if m.id == "matrixai.dense.torch_cpu")
        self.assertEqual(densa.receta, {
            "optimizador": "adam", "learning_rate": 0.001,
            "early_stop": {"patience": 10, "metric": "validation_loss"},
            "epochs": 50, "batch_size": 8,
        })

    def test_la_densa_conserva_sus_2_configuraciones(self):
        """La receta no cambia CUÁNTAS configuraciones compite (eso es otra
        decisión, del anexo C §2.3): sigue siendo 2, como en la v1."""
        densa = next(m for m in self.protocolo.motores if m.id == "matrixai.dense.torch_cpu")
        self.assertEqual(densa.configuraciones, 2)

    # --- Los 3 ARFF nuevos: sha256 declarado == sha256 del fichero ----------

    @con_los_arff
    def test_los_3_ARFF_nuevos_tienen_el_sha256_declarado(self):
        import hashlib
        for data_id in ENTRANTES:
            ds = self.por_id[data_id]
            ruta = DATOS / f"{data_id}.arff"
            self.assertTrue(ruta.exists(), f"falta el ARFF de {ds.nombre} ({data_id})")
            self.assertEqual(hashlib.sha256(ruta.read_bytes()).hexdigest(), ds.sha256_arff)

    def test_los_3_nuevos_traen_file_id(self):
        for data_id in ENTRANTES:
            self.assertIsNotNone(self.por_id[data_id].file_id)

    # --- La huella literal ---------------------------------------------------

    #: El digest EXACTO de la v2 tal como quedó escrita el 2026-09-21. Escrito
    #: a mano, como su hermano de la v1: un ancla calculada no ancla nada — si
    #: alguien toca el protocolo, esto se pone rojo y esa es la conversación
    #: que tiene que haber (misma razón que `DIGEST_REGISTRADO_ANTES_DE_MEDIR`
    #: en `test_c101_c1_protocolo_fase0.py`).
    DIGEST_V2 = "1b4e902e61e06edcfda8e9f4ccef1232a233d46718089c5f0a2f814feb9dc878"

    def test_la_huella_de_la_v2_es_la_registrada(self):
        self.assertEqual(self.protocolo.digest(), self.DIGEST_V2)
        self.assertNotEqual(self.protocolo.digest(),
                            LaV1NoSeHaMovidoTest.DIGEST_V1,
                            "la v2 no puede tener la MISMA huella que la v1")


# ---------------------------------------------------------------------------
# El CÓDIGO que produjo la v2 — "probar el artefacto no es probar el código
# que lo produce": se vuelve a calcular con la misma función, sobre los ARFF
# reales, y tiene que coincidir con lo que quedó escrito.
# ---------------------------------------------------------------------------

@con_los_arff
class GeneradorV2RederivaLoMismoTest(unittest.TestCase):
    """Si alguien cambia el umbral de P1 (o cualquier otro cálculo) en
    `generar_protocolo_v2.py` SIN volver a escribir el JSON, esta prueba lo
    caza: es la única de este fichero que no confía en el artefacto, confía
    en re-ejecutar el cálculo."""

    @classmethod
    def setUpClass(cls):
        if not RUTA_V1.exists() or not RUTA_V2.exists():
            raise unittest.SkipTest("faltan la v1 o la v2")
        cls.protocolo_v2 = ProtocoloExploratorio.cargar(RUTA_V2)
        cls.payload_v1 = json.loads(RUTA_V1.read_text(encoding="utf-8"))

    def test_P1_rederivado_desde_el_codigo_coincide_con_lo_registrado(self):
        """Para los multiclase NO sellados, lo registrado tiene que coincidir
        con re-aplicar P1 desde cero. Para el ÚNICO sellado (`letter`), lo
        registrado tiene que seguir siendo el de la v1 — NO lo que P1 daría
        crudo, que es justo lo que la excepción de "sellados no se tocan"
        significa. Las dos mitades hacen falta: sin la segunda, un código que
        aplicara P1 también a los sellados (el bug real que esta prueba cazó
        durante el desarrollo del corte) pasaría en silencio."""
        recalculado = g2.re_firmar_desbalanceado_multiclase(self.payload_v1["datasets"], DATOS)
        # Los 10 multiclase de la v1 son justo los 10 de la v2 (ninguno de los
        # 3 sustituidos es multiclase): el mapa cubre exactamente esos 10.
        self.assertEqual(len(recalculado), 10)
        for d in self.protocolo_v2.datasets:
            if d.tarea != "multiclass_classification":
                continue
            # Una sola fórmula para los 10, sellados incluidos.
            self.assertEqual(d.desbalanceado, recalculado[d.data_id],
                            f"{d.nombre}: lo registrado y el P1 recalculado discrepan")

    def test_letter_sellado_sigue_la_misma_formula_que_los_demas(self):
        """El caso que obligó a decidir (2026-09-21): `letter` está sellado y
        P1 le da `False` (cuota*k = 0,954 > 0,2), donde la regla vieja le daba
        `True`. Lo registrado tiene que ser lo de P1: una sola fórmula. Si esto
        se pone rojo, alguien volvió a hacer una excepción con los sellados."""
        letter_v2 = next(d for d in self.protocolo_v2.datasets if d.nombre == "letter")
        letter_v1 = next(d for d in self.payload_v1["datasets"] if d["nombre"] == "letter")
        self.assertTrue(letter_v2.sellado)
        self.assertTrue(letter_v1["desbalanceado"])
        self.assertFalse(letter_v2.desbalanceado)

    def test_el_umbral_de_P1_sigue_siendo_02_no_05(self):
        """Ancla directa a la constante: un umbral de 0,5/k (la variante (b)
        que Roberto NO eligió) marcaría 5 de los 10 multiclase, no 1."""
        self.assertEqual(g2.UMBRAL_P1_DESBALANCE, 0.2)

    def test_las_celdas_de_las_3_sustituciones_estan_declaradas_en_el_codigo(self):
        self.assertEqual(set(g2.DATA_IDS_SALIENTES), set(SALIENTES))
        self.assertEqual(set(g2.DATA_IDS_ENTRANTES), set(ENTRANTES))

    def test_la_receta_declarada_en_el_codigo_es_la_que_quedo_escrita(self):
        densa = next(m for m in self.protocolo_v2.motores if m.id == "matrixai.dense.torch_cpu")
        self.assertEqual(dict(g2.RECETA_DENSA_113_C1), densa.receta)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
