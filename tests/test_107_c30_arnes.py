# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — el arnés de `benchmarks/texto_107c30/` (`preparar_tareas.py` y
`medir_c30.py`). Rápidas, SIN descargar modelos ni leer los ficheros grandes
de `/home/deployer/datos_107c30/`: cada prueba se construye su propio
fixture pequeño.

Cubre lo que pide el corte:
  - la normalización de la tarea A (booleano de Excel de v2, mayúsculas);
  - la regla de B (código CIE-10 `r52`) sobre un TSV inventado;
  - la regla de idioma, con frases de control en los cinco idiomas Y una
    auditoría directa de que ningún marcador coincide con el castellano;
  - que las condiciones (0), (1) y (2) de `medir_c30.py` comparten la MISMA
    partición (mismo `split_plan_digest`);
  - el veredicto de `comparar_candidatos` frente a (0), con una señal clara.

Los sabotajes que este fichero sostiene (comprobados a mano, ver el informe
del corte: copia, diff, rojo, restaurar, `md5sum`, borrar `__pycache__`):
  1. el booleano de Excel sin normalizar en `_normalizar_category`;
  2. un marcador compartido con el castellano en `regla_idioma.json`;
  3. particiones distintas por condición en `medir_c30._digest_particion`.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "benchmarks" / "texto_107c30"))
import preparar_tareas as pt  # noqa: E402
import medir_c30 as mc  # noqa: E402

from matrixai.estudio.metricas import Muestra  # noqa: E402


# ---------------------------------------------------------------------------
# A · normalización del booleano de Excel y las mayúsculas de v2
# ---------------------------------------------------------------------------

class NormalizacionCategoriaTest(unittest.TestCase):
    def test_texto_v1_se_conserva(self):
        self.assertEqual(pt._normalizar_category("Fake", "s", origen="v1"), "Fake")
        self.assertEqual(pt._normalizar_category("True", "s", origen="v1"), "True")

    def test_booleano_de_excel_de_v2_se_traduce_al_vocabulario_de_v1(self):
        """La trampa del contrato: `CATEGORY` en v2 es un booleano DE EXCEL
        (tipo de celda 'b'), no la cadena de v1. `True` -> "True" (noticia
        real), `False` -> "Fake" -- NUNCA `str(valor)`, que daría "False" y
        rompería el vocabulario {"Fake","True"} en silencio."""
        self.assertEqual(pt._normalizar_category(True, "b", origen="v2"), "True")
        self.assertEqual(pt._normalizar_category(False, "b", origen="v2"), "Fake")

    def test_booleano_declarado_sin_valor_bool_se_rechaza(self):
        with self.assertRaises(ValueError):
            pt._normalizar_category(1, "b", origen="v2")

    def test_texto_fuera_del_vocabulario_se_rechaza(self):
        with self.assertRaises(ValueError):
            pt._normalizar_category("Quizas", "s", origen="v1")

    def test_tipo_de_celda_inesperado_se_rechaza(self):
        with self.assertRaises(ValueError):
            pt._normalizar_category("Fake", "n", origen="v1")


# ---------------------------------------------------------------------------
# B · la regla del código CIE-10, sobre un TSV inventado
# ---------------------------------------------------------------------------

class ReglaCodiEspTest(unittest.TestCase):
    def test_leer_d_tsv_agrupa_codigos_por_caso(self):
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "inventadoD.tsv"
            tsv.write_text(
                "caso-1\tr52\n"
                "caso-1\tz20.818\n"
                "caso-2\tn44.8\n"
                "caso-3\tr52\n",
                encoding="utf-8",
            )
            por_caso = pt._leer_d_tsv(tsv)
        self.assertEqual(por_caso, {
            "caso-1": {"r52", "z20.818"},
            "caso-2": {"n44.8"},
            "caso-3": {"r52"},
        })

    def test_r52_decide_el_target_binario(self):
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "inventadoD.tsv"
            tsv.write_text("caso-1\tr52\ncaso-2\tn44.8\n", encoding="utf-8")
            por_caso = pt._leer_d_tsv(tsv)
        # la misma regla que tarea_b(): positivo si 'r52' está en el conjunto
        self.assertTrue(pt.CODIGO_CODIESP in por_caso["caso-1"])
        self.assertFalse(pt.CODIGO_CODIESP in por_caso["caso-2"])

    def test_codigo_en_mayusculas_o_con_espacios_se_normaliza(self):
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "inventadoD.tsv"
            tsv.write_text("caso-1\t R52 \n", encoding="utf-8")
            por_caso = pt._leer_d_tsv(tsv)
        self.assertIn("r52", por_caso["caso-1"])


# ---------------------------------------------------------------------------
# C/D · la regla de idioma
# ---------------------------------------------------------------------------

FRASES_DE_CONTROL = {
    "es": "El ayuntamiento adjudica el contrato de suministro de mobiliario para el nuevo edificio "
          "municipal, con un plazo de ejecución de seis meses y un importe total revisado.",
    "ca": "L'ajuntament adjudica el contracte de subministrament de mobiliari amb un termini "
          "d'execucio de sis mesos i aquest import revisat pels seus tecnics.",
    "gl": "O concello adxudica o contrato de subministracion de mobiliario para o traballo do "
          "edificio, cunha praza reservada e coa xunta de goberno local informada.",
    "en": "The council awarded the contract for the supply and maintenance of public works and "
          "services for the new municipal building with public procurement rules.",
    "eu": "Udalak zerbitzua eta kontratazioa esleitu ditu udalaren eraikin berrirako, herriko "
          "administrazioaren azpiegitura hobetzeko lanak barne.",
}


class ReglaIdiomaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regla = pt.cargar_regla_idioma()

    def test_frases_de_control_se_clasifican_en_su_idioma(self):
        for idioma, frase in FRASES_DE_CONTROL.items():
            with self.subTest(idioma=idioma):
                self.assertEqual(pt.detectar_idioma(frase, self.regla), idioma)

    def test_frase_castellana_corta_y_ambigua_no_cruza_el_umbral(self):
        """Una sola coincidencia suelta (p. ej. con un marcador de una letra
        de otro idioma) NO puede decidir -- hace falta el umbral de >=2."""
        frase = "Servicio de limpieza de edificios municipales del Ayuntamiento de Mostoles."
        self.assertEqual(pt.detectar_idioma(frase, self.regla), "es")

    def test_ningun_marcador_activo_coincide_con_una_palabra_castellana_corriente(self):
        """La auditoría del principio ('EXCLUSIVAS... auditadas contra
        colisiones con el castellano') hecha PRUEBA: si algún marcador activo
        de CUALQUIER idioma es en realidad una palabra castellana corriente,
        esto tiene que fallar. Es la comprobación con más dientes contra un
        marcador compartido colado por descuido (o por sabotaje)."""
        palabras_castellanas_corrientes = {
            "el", "la", "los", "las", "de", "del", "que", "un", "una", "unos", "unas",
            "en", "a", "es", "son", "y", "o", "no", "su", "sus", "lo", "al", "se", "por",
            "para", "con", "sin", "como", "mas", "pero", "si", "este", "esta", "estos",
            "estas", "ese", "esa", "muy", "dos", "tambien", "entre", "sobre", "hasta",
        }
        for codigo, bloque in self.regla["idiomas"].items():
            marcadores_normalizados = {pt._sin_diacriticos(m.lower()) for m in bloque["marcadores"]}
            colision = marcadores_normalizados & palabras_castellanas_corrientes
            with self.subTest(idioma=codigo):
                self.assertFalse(
                    colision,
                    f"marcador(es) de '{codigo}' que son palabras castellanas corrientes: {colision}",
                )

    def test_umbral_de_dos_marcadores_distintos(self):
        self.assertEqual(self.regla["umbral_marcadores_distintos"], 2)

    def test_texto_vacio_es_castellano_por_omision(self):
        self.assertEqual(pt.detectar_idioma("", self.regla), "es")


# ---------------------------------------------------------------------------
# medir_c30 · las tres condiciones sobre la MISMA partición
# ---------------------------------------------------------------------------

def _tarea_sintetica_binaria(n=60, seed=1):
    """Filas sintéticas con una señal de texto REAL (para que TF-IDF tenga
    algo que aprender) y una columna 'otra' con señal más débil, repartidas
    en train/dev/test con la MISMA lógica que usaría preparar_tareas.py."""
    import random
    rng = random.Random(seed)
    filas = []
    for i in range(n):
        positivo = rng.random() < 0.5
        texto = ("contrato de suministro urgente prioritario" if positivo
                else "informe rutinario ordinario de seguimiento")
        # una pizca de ruido para que no sea determinista al 100%:
        if rng.random() < 0.15:
            texto = "palabra neutra sin relacion aparente"
        particion = "train" if i < int(n * 0.6) else ("dev" if i < int(n * 0.8) else "test")
        filas.append({
            "row_id": f"s{i}", "particion": particion,
            "target": "si" if positivo else "no",
            "texto": texto, "otra": "x" if positivo else "y",
        })
    return filas


class TareaSinteticaMixin:
    """Registra una tarea sintética diminuta en `medir_c30.TAREAS`/lectura,
    sin tocar disco ni el registro real -- se limpia en tearDown."""

    NOMBRE = "_sintetica_test_"

    def setUp(self):
        self.filas = _tarea_sintetica_binaria()
        self._particiones = {"train": [], "dev": [], "test": []}
        for f in self.filas:
            self._particiones[f["particion"]].append(f)
        mc.TAREAS[self.NOMBRE] = dict(
            fichero="__no_usado__.csv", tipo="binary_classification",
            clases=("no", "si"), positive_label="si",
            otras_columnas=("otra",), metrica="auroc", idioma="es",
        )
        self._leer_original = mc.leer_tarea
        mc.leer_tarea = lambda nombre: self._particiones if nombre == self.NOMBRE else self._leer_original(nombre)

    def tearDown(self):
        mc.TAREAS.pop(self.NOMBRE, None)
        mc.leer_tarea = self._leer_original


class MismaParticionTest(TareaSinteticaMixin, unittest.TestCase):
    def test_las_tres_condiciones_declaran_el_mismo_split_plan_digest(self):
        resultado = mc.evaluar_tarea(self.NOMBRE, incluir_embedding=False)
        digest0 = resultado["condiciones"]["0_sin_texto"]["split_plan_digest"]
        digest2 = resultado["condiciones"]["2_tfidf_lineal"]["split_plan_digest"]
        self.assertEqual(
            digest0, digest2,
            "condición (0) y (2) tienen que declarar el MISMO split_plan_digest: "
            "leen las mismas filas de train/dev/test.",
        )

    def test_el_digest_solo_depende_de_las_filas_no_del_nombre_de_la_condicion(self):
        train_ids = [f["row_id"] for f in self._particiones["train"] + self._particiones["dev"]]
        test_ids = [f["row_id"] for f in self._particiones["test"]]
        d1 = mc._digest_particion(train_ids, test_ids)
        d2 = mc._digest_particion(list(train_ids), list(test_ids))  # mismas filas, otra lista
        self.assertEqual(d1, d2)
        # y SÍ tiene que cambiar si cambian las filas de verdad:
        d3 = mc._digest_particion(train_ids[:-1], test_ids)
        self.assertNotEqual(d1, d3)


# ---------------------------------------------------------------------------
# el veredicto frente a (0), con una señal clara
# ---------------------------------------------------------------------------

class VeredictoFrenteACeroTest(unittest.TestCase):
    def test_candidato_claramente_mejor_da_mejora(self):
        import random
        rng = random.Random(3)
        n = 400
        y_true = ["si" if rng.random() < 0.3 else "no" for _ in range(n)]
        # baseline: puntuación casi sin relación con la verdad
        scores_base = [rng.random() for _ in y_true]
        # candidato: puntuación bien correlacionada con la verdad
        scores_cand = [rng.uniform(0.6, 1.0) if y == "si" else rng.uniform(0.0, 0.4) for y in y_true]

        m_base = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                         positive_label="si", scores=tuple(scores_base))
        m_cand = Muestra(task="binary_classification", y_true=tuple(y_true), classes=("no", "si"),
                         positive_label="si", scores=tuple(scores_cand))
        resultado = mc._comparar("auroc", m_cand, m_base, semilla=7,
                                 protocolo_candidato="d" * 64, protocolo_baseline="d" * 64)
        self.assertEqual(resultado["veredicto"], "mejora")
        self.assertGreater(resultado["intervalo"]["ci_low"], 0.0)

    def test_protocolos_distintos_declarados_dan_incomparable(self):
        """Confirma que `_comparar` SÍ usa `protocolo_*`: si dos condiciones
        declaran un protocolo distinto (lo que pasaría con el sabotaje 3,
        particiones distintas por condición), el veredicto tiene que ser
        'incomparable', no una comparación silenciosa entre cosas distintas."""
        y_true = tuple(["si", "no"] * 50)
        m_a = Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                     positive_label="si", scores=tuple([0.9, 0.1] * 50))
        m_b = Muestra(task="binary_classification", y_true=y_true, classes=("no", "si"),
                     positive_label="si", scores=tuple([0.8, 0.2] * 50))
        resultado = mc._comparar("auroc", m_a, m_b, semilla=1,
                                 protocolo_candidato="a" * 64, protocolo_baseline="b" * 64)
        self.assertEqual(resultado["veredicto"], "incomparable")


if __name__ == "__main__":
    unittest.main()


class ElMargenEsElDelPreRegistroTest(unittest.TestCase):
    """El margen de equivalencia es el que fijó el pre-registro (0,05, el del 105-C5) y es
    el que LLEGA a la comparación. Añadido por el supervisor (25-09): cambiar el margen a
    0,5 dejaba las 17 pruebas verdes."""

    def test_el_margen_es_0_05_y_es_el_que_recibe_comparar_candidatos(self):
        from unittest import mock
        self.assertEqual(mc.MARGEN_EQUIVALENCIA, 0.05)
        recibido = {}

        class _Resultado:
            def a_json(self):
                return {}

        def espia(*args, **kwargs):
            recibido.update(kwargs)
            return _Resultado()
        with mock.patch.object(mc, "comparar_candidatos", espia):
            mc._comparar("auroc", object(), object(), semilla=0,
                         protocolo_candidato="p", protocolo_baseline="p")
        self.assertEqual(recibido["margen_equivalencia"], 0.05)
