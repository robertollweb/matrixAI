# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C1 — EL PROBLEMA SE CONFIRMA ANTES DE ENTRENAR.

Lo que se prueba aquí no es que unas funciones devuelvan diccionarios: es que
**no se pueda arrancar un estudio contra un objetivo que nadie pidió**. Cada
caso de este fichero existe por un defecto medido:

* «analizar los datos de clientes de una empresa de telefonía» construía un
  modelo con `OUTPUT predicted_class` y `feature_1..4` — objetivo Y entradas
  inventados, cero avisos (medido contra `analyze_playground_request`);
* un objetivo numérico con tres valores distintos se proponía como
  clasificación en `_rank_target_candidates` y se entrenaba como regresión en
  `generate_project_from_dataset`, sin que nadie preguntara nada;
* un CSV ordenado por la columna objetivo deja UNA sola clase en el 80 % que de
  verdad entrena, y `constant_target_error` no lo ve porque mira el fichero
  entero: el modelo llega a pérdida 0 acertando siempre lo mismo;
* el orden de las clases y la clase positiva no viajaban en el manifiesto, así
  que quien recibiera el paquete no podía leer su vector de probabilidades;
* y el contrato 71 dejó por escrito el límite: un detector por parecido de
  nombres marcaría `last_year_salary` y acusaría a una columna legítima.

Los tres casos del 71 están probados con los nombres de columna que aquel
contrato MIDIÓ contra el backend real, no con nombres inventados para que salga
bien.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from matrixai.estudio import Horizonte, ProblemSpec, ProcesoActual
from matrixai.export.reproduce import (
    ReproduceManifestError,
    build_problem_block,
    build_reproduce_manifest,
)
from matrixai.generation.prompt_objetivo import (
    NOMBRES_INVENTADOS,
    objetivo_declarado,
)
from matrixai.training.dataset_analysis import analyze_dataset_csv, constant_target_error
from matrixai.training.dataset_project import generate_project_from_dataset
from matrixai.training.objetivo import (
    Confirmacion,
    ObjetivoSinConfirmar,
    confirmar_desde_csv,
    confirmar_desde_prompt,
    corte_de_entrenamiento,
    exigir_problema_confirmado,
    no_entrenable_en_train,
    pistas_por_nombre,
)
from matrixai.training.objetivo_textos import IDIOMAS, MOTIVOS, huecos_de, motivo


def _claves(items) -> list[str]:
    return [x.clave for x in items]


def _csv(cabecera: str, filas: list[str]) -> str:
    return "\n".join([cabecera] + filas) + "\n"


# ---------------------------------------------------------------------------
# El catálogo bilingüe
# ---------------------------------------------------------------------------

class TestElCatalogoHablaLosDosIdiomas(unittest.TestCase):
    """Media traducción presentada como completa es peor que ninguna, y un
    hueco que solo existe en un idioma revienta el día que alguien pide el
    otro."""

    def test_cada_motivo_trae_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            self.assertEqual(set(textos), set(IDIOMAS), clave)
            for idioma in IDIOMAS:
                self.assertTrue(textos[idioma].strip(), f"{clave}/{idioma}")

    def test_los_huecos_son_los_mismos_en_los_dos_idiomas(self):
        """Un `{opciones}` en castellano que no está en inglés solo falla
        cuando alguien pide inglés, y entonces es un `KeyError` en producción."""
        for clave, textos in MOTIVOS.items():
            self.assertEqual(huecos_de(textos["es"]), huecos_de(textos["en"]),
                             f"{clave}: los huecos no coinciden")

    def test_una_clave_que_no_existe_se_rechaza_en_vez_de_inventar_la_frase(self):
        with self.assertRaises(KeyError):
            motivo("esta_clave_no_existe")

    def test_la_redaccion_inglesa_no_lleva_castellano(self):
        """Barrido con palabras FUNCIONALES, no con una lista escogida a mano:
        `el`, `la`, `del`, `que`, `con`… no se pueden evitar escribiendo en
        castellano, y son las que delatan una traducción a medias."""
        funcionales = ("el", "la", "los", "las", "del", "que", "con", "para",
                       "por", "una", "unos", "unas", "desde", "hasta", "sin",
                       "sobre", "entre", "cuando", "porque", "esto", "este",
                       "esta", "como", "más", "año", "sí")
        patron = re.compile(r"\b(" + "|".join(funcionales) + r")\b", re.IGNORECASE)
        for clave, textos in MOTIVOS.items():
            encontrado = patron.findall(textos["en"])
            self.assertEqual(encontrado, [], f"{clave}: castellano en el inglés")

    def test_toda_plantilla_se_puede_componer_con_sus_huecos(self):
        """Un hueco que nadie rellena sale literal en pantalla como `{campo}`."""
        for clave, textos in MOTIVOS.items():
            campos = {h: "X" for h in huecos_de(textos["es"])}
            compuesto = motivo(clave, **campos)
            for idioma in IDIOMAS:
                self.assertNotIn("{", compuesto[idioma], f"{clave}/{idioma}")


# ---------------------------------------------------------------------------
# Criterio 1 — un prompt sin objetivo no inicia estudio
# ---------------------------------------------------------------------------

class TestUnPromptSinObjetivoNoIniciaEstudio(unittest.TestCase):
    """El caso del contrato 71, primera fila: «analizar los datos de clientes de
    una empresa de telefonía». Hoy el core construye un modelo entero con
    `OUTPUT predicted_class` y `feature_1..4`, y nadie avisa de nada."""

    PROMPT = "analizar los datos de clientes de una empresa de telefonia"

    def test_no_hay_problema_y_se_pide_el_objetivo(self):
        conf = confirmar_desde_prompt(self.PROMPT)
        self.assertIsNone(conf.problema)
        self.assertFalse(conf.confirmado)
        self.assertEqual(_claves(conf.preguntas), ["objetivo_no_declarado"])

    def test_la_pregunta_cita_la_frase_en_los_dos_idiomas(self):
        conf = confirmar_desde_prompt(self.PROMPT)
        pregunta = conf.preguntas[0]
        for idioma in IDIOMAS:
            self.assertIn("telefonia", pregunta.motivo[idioma])
        self.assertIn("predecir", pregunta.motivo["es"])
        self.assertIn("predicted", pregunta.motivo["en"])

    def test_la_puerta_del_estudio_lo_rechaza_diciendo_que_falta(self):
        conf = confirmar_desde_prompt(self.PROMPT)
        with self.assertRaises(ObjetivoSinConfirmar) as caja:
            exigir_problema_confirmado(conf)
        self.assertEqual(caja.exception.clave, "estudio_sin_problema_confirmado")
        self.assertIn("objetivo_no_declarado", caja.exception.es)
        self.assertIn("objetivo_no_declarado", caja.exception.en)

    def test_el_nombre_que_se_inventa_el_generador_no_cuenta_como_objetivo(self):
        """Un prompt que ya trae `OUTPUT predicted_class` viene de una vuelta
        anterior del propio core. Leerlo sería blanquear la invención — y
        antes de este corte la lectura de la frase encontraba la raíz «predic»
        dentro de `predicted_class` y devolvía `probabilitymap`."""
        vuelta = "PROJECT X\nOUTPUT predicted_class: ProbabilityMap[a, b]\n"
        self.assertIsNone(objetivo_declarado(vuelta))
        conf = confirmar_desde_prompt(vuelta)
        self.assertEqual(_claves(conf.preguntas), ["objetivo_no_declarado"])

    def test_ningun_objetivo_leido_es_uno_de_los_inventados(self):
        for prompt in ("predecir el precio de una vivienda",
                       "clasificar el riesgo del paciente en alto, medio o bajo",
                       "OUTPUT predicted_value: Scalar"):
            leido = objetivo_declarado(prompt)
            if leido is not None and leido.nombre:
                self.assertNotIn(leido.nombre, NOMBRES_INVENTADOS, prompt)

    def test_una_pregunta_de_si_o_no_pide_el_nombre_y_no_se_lo_inventa(self):
        """«Predecir si un cliente va a impagar» dice QUÉ y no cómo se llama.
        El primer sustantivo tras el «si» es `cliente`, que es el SUJETO: darlo
        por objetivo sería una respuesta equivocada que nadie sospecharía."""
        conf = confirmar_desde_prompt(
            "predecir si un cliente va a impagar a partir de la edad y los ingresos")
        self.assertIsNone(conf.problema)
        self.assertEqual(_claves(conf.preguntas), ["objetivo_sin_nombre"])
        self.assertIn("impagar", conf.preguntas[0].motivo["es"])
        self.assertNotIn("cliente", str(conf.propuesta.get("lectura", {}).get("nombre")))

    def test_contestar_con_la_columna_arranca_el_estudio(self):
        """La pregunta se contesta, y contestarla es lo que confirma. `objetivo`
        y `unidad_de_observacion` son las dos respuestas que faltaban."""
        conf = confirmar_desde_prompt(
            self.PROMPT, objetivo="churn", tarea="binary_classification",
            clases=("no", "si"), clase_positiva="si",
            unidad_de_observacion="un cliente", entradas=("edad", "plan"))
        self.assertTrue(conf.confirmado, _claves(conf.preguntas) + _claves(conf.bloqueos))
        problema = exigir_problema_confirmado(conf)
        self.assertIsInstance(problema, ProblemSpec)
        self.assertEqual(problema.target, "churn")
        self.assertEqual(problema.classes, ("no", "si"))
        self.assertEqual(problema.positive_label, "si")

    def test_el_proceso_actual_108_c1_viaja_tambien_por_la_ruta_prompt(self):
        proceso = ProcesoActual(tipo="regla", regla="marcar impago si mora > 90 días")
        conf = confirmar_desde_prompt(
            self.PROMPT, objetivo="churn", tarea="binary_classification",
            clases=("no", "si"), clase_positiva="si",
            unidad_de_observacion="un cliente", entradas=("edad", "plan"),
            proceso_actual=proceso)
        self.assertTrue(conf.confirmado, _claves(conf.preguntas))
        self.assertEqual(conf.problema.current_process, proceso)

    def test_una_clasificacion_sin_clases_nombradas_no_arranca(self):
        """Medido y documentado en el propio core: «clasificar el nivel de
        riesgo del paciente» devolvía `ProbabilityMap[class_a, class_b,
        class_c]` — marcadores de posición cuyas salidas no significan nada."""
        conf = confirmar_desde_prompt(
            "clasificar el nivel de riesgo del paciente a partir de la edad",
            unidad_de_observacion="un paciente")
        self.assertIn("clases_no_declaradas", _claves(conf.preguntas))
        self.assertIsNone(conf.problema)
        self.assertNotIn("class_a", json.dumps(conf.a_json()))

    def test_con_las_clases_nombradas_y_lo_demas_contestado_si_arranca(self):
        conf = confirmar_desde_prompt(
            "clasificar el nivel de riesgo del paciente en alto, medio o bajo "
            "a partir de la edad",
            unidad_de_observacion="un paciente")
        self.assertTrue(conf.confirmado, _claves(conf.preguntas))
        self.assertEqual(conf.problema.classes, ("alto", "medio", "bajo"))


# ---------------------------------------------------------------------------
# Criterio 4 — los tres casos del contrato 71
# ---------------------------------------------------------------------------

class TestLosTresCasosDelContrato71(unittest.TestCase):
    """`71_COLUMNAS_DE_ENTRADA_CONTRACT.md:63-67`, con las entradas que aquel
    contrato midió CONTRA EL BACKEND REAL (por eso vienen en inglés: las
    devolvió el LLM) y con el veredicto que escribió para cada una."""

    def test_caso_1_el_precio_a_partir_del_precio_es_fuga_y_bloquea(self):
        conf = confirmar_desde_prompt(
            "clasificar el precio de una vivienda en categorias barata, media y "
            "cara a partir del precio, los metros y el barrio",
            objetivo="price", tarea="regression",
            entradas=("price", "square_meters", "neighborhood"),
            unidad_de_observacion="una vivienda")
        self.assertIn("objetivo_entre_las_entradas", _claves(conf.bloqueos))
        self.assertIsNone(conf.problema)
        self.assertIn("100 %", conf.bloqueos[0].motivo["es"])

    def test_caso_2_el_salario_del_ano_pasado_es_legitimo_y_NO_se_bloquea(self):
        """«El segundo caso lo etiqueté como fuga y no lo es. Ahí está el
        límite»: el sueldo anterior predice bien el siguiente. El parecido de
        nombres AVISA y el estudio sigue."""
        conf = confirmar_desde_prompt(
            "predecir el salario de un empleado a partir del salario del ano "
            "pasado, la antiguedad y el departamento",
            objetivo="salary", tarea="regression",
            entradas=("last_year_salary", "seniority", "department"),
            unidad_de_observacion="un empleado")
        self.assertEqual(conf.bloqueos, ())
        self.assertTrue(conf.confirmado)
        self.assertEqual(_claves(conf.pistas), ["entrada_que_contiene_el_objetivo"])
        self.assertEqual(conf.pistas[0].entrada, "last_year_salary")
        # Y la pista dice que es una pista, en los dos idiomas.
        self.assertIn("PISTA", conf.pistas[0].motivo["es"])
        self.assertIn("HINT", conf.pistas[0].motivo["en"])

    def test_caso_3_el_tercero_esta_limpio_y_no_dispara_nada(self):
        conf = confirmar_desde_prompt(
            "predecir el precio de una vivienda a partir de los metros, el "
            "barrio y el ano de construccion",
            objetivo="price", tarea="regression",
            entradas=("square_meters", "neighborhood", "build_year"),
            unidad_de_observacion="una vivienda")
        self.assertTrue(conf.confirmado)
        self.assertEqual(conf.pistas, ())
        self.assertEqual(conf.bloqueos, ())

    def test_los_mismos_tres_con_las_columnas_en_castellano(self):
        """Las mismas tres frases cuando las columnas salen en el idioma del
        prompt: el objetivo se lee de la frase y no hace falta declararlo."""
        casos = [
            ("clasificar el precio de una vivienda en categorias barata, media y "
             "cara a partir del precio, los metros y el barrio",
             ("precio", "metros", "barrio"), ["objetivo_entre_las_entradas"], 1),
            ("predecir el salario de un empleado a partir del salario del ano "
             "pasado, la antiguedad y el departamento",
             ("salario_del_ano_pasado", "antiguedad", "departamento"), [], 1),
            ("predecir el precio de una vivienda a partir de los metros, el "
             "barrio y el ano de construccion",
             ("metros", "barrio", "ano_de_construccion"), [], 0),
        ]
        for prompt, entradas, bloqueos, pistas in casos:
            with self.subTest(prompt=prompt[:40]):
                conf = confirmar_desde_prompt(
                    prompt, entradas=entradas, tarea="regression",
                    unidad_de_observacion="una fila del histórico")
                self.assertEqual(_claves(conf.bloqueos), bloqueos)
                self.assertEqual(len(conf.pistas), pistas)

    def test_un_parecido_no_convierte_una_confirmacion_en_bloqueo(self):
        """Las pistas NO cuentan para `confirmado`. Si contaran, el caso
        legítimo del 71 quedaría bloqueado por el nombre, que es exactamente lo
        que aquel contrato midió que estaba mal."""
        conf = Confirmacion(
            problema=ProblemSpec(problem_id="p", target="salary", task="regression",
                                 observation_unit="un empleado"),
            pistas=pistas_por_nombre("salary", ("last_year_salary",)))
        self.assertEqual(len(conf.pistas), 1)
        self.assertTrue(conf.confirmado)

    def test_lo_que_el_parecido_por_nombre_NO_ve_esta_dicho(self):
        """El 71 registró que el LLM devolvía `price` para «el precio»: por
        nombre no hay parecido ninguno entre los dos idiomas. Se prueba la
        limitación en vez de dejarla implícita — lo que caza esa fuga es la
        identidad confirmada, no el parecido."""
        self.assertEqual(pistas_por_nombre("precio", ("price", "metros")), ())

    def test_el_generador_avisa_de_la_fuga_y_NO_quita_la_columna(self):
        """Invariante 1 del 71: se avisa, no se corrige. Quitar una columna que
        alguien ha pedido, sin decirlo, es decidir por su cuenta sobre sus
        datos."""
        from matrixai.training.dense_generator import (
            DenseNetworkGenerator,
            resolve_prompt_fields,
        )
        campos, _, _, _, avisos, _ = resolve_prompt_fields(
            DenseNetworkGenerator(),
            "clasificar el precio de una vivienda a partir del precio, los "
            "metros y el barrio",
            ["precio", "metros", "barrio"], "es")
        self.assertIn("precio", campos)          # NO se quita
        self.assertTrue(any("PISTA" in a for a in avisos), avisos)

    def test_el_generador_avisa_del_parecido_sin_llamarlo_fuga(self):
        from matrixai.training.dense_generator import (
            DenseNetworkGenerator,
            resolve_prompt_fields,
        )
        _, _, _, _, avisos, _ = resolve_prompt_fields(
            DenseNetworkGenerator(),
            "predecir el salario de un empleado a partir del salario del ano pasado",
            ["salario_del_ano_pasado", "antiguedad"], "es")
        pista = [a for a in avisos if "PISTA" in a]
        self.assertEqual(len(pista), 1, avisos)
        self.assertIn("legítimo", pista[0])

    def test_el_aviso_del_generador_tambien_sale_en_ingles(self):
        """Media aplicación traducida se ve así: conducir también en inglés."""
        from matrixai.training.dense_generator import (
            DenseNetworkGenerator,
            resolve_prompt_fields,
        )
        _, _, _, _, avisos, _ = resolve_prompt_fields(
            DenseNetworkGenerator(),
            "predict the price of a house from the price and the square meters",
            ["price", "square_meters"], "en")
        self.assertTrue(any("HINT" in a for a in avisos), avisos)
        self.assertFalse(any("PISTA" in a for a in avisos), avisos)

    def test_un_prompt_limpio_no_gana_ningun_aviso_nuevo(self):
        """Un aviso que salta cuando no toca enseña a ignorarlos todos."""
        from matrixai.training.dense_generator import (
            DenseNetworkGenerator,
            resolve_prompt_fields,
        )
        _, _, _, _, avisos, _ = resolve_prompt_fields(
            DenseNetworkGenerator(),
            "predecir el precio de una vivienda a partir de los metros y el barrio",
            ["metros", "barrio"], "es")
        self.assertEqual([a for a in avisos if "PISTA" in a], [])


# ---------------------------------------------------------------------------
# Criterio 2 — la columna objetivo no entra en X
# ---------------------------------------------------------------------------

class TestLaColumnaObjetivoNoEntraEnX(unittest.TestCase):

    CSV = _csv("metros,barrio,precio",
               [f"{60 + i},b{i % 3},{100000 + i * 1000}" for i in range(20)])

    def test_la_ruta_csv_deja_el_objetivo_fuera_de_las_features(self):
        res = generate_project_from_dataset(self.CSV, "precio")
        self.assertNotIn("precio", res["provenance"]["preparation_spec"]["feature_columns"])
        # Y tampoco en el VECTOR del modelo, que es donde de verdad importa.
        vector = res["mxai"].split("VECTOR", 1)[1].split("END", 1)[0]
        self.assertNotIn("precio", vector)

    def test_la_confirmacion_declara_los_predictores_reales_sin_el_objetivo(self):
        res = generate_project_from_dataset(
            self.CSV, "precio", unidad_de_observacion="una vivienda")
        problema = res["confirmacion"]["problema"]
        self.assertTrue(res["confirmacion"]["confirmado"])
        self.assertNotIn("precio", problema["predictors"])
        self.assertEqual(problema["target"], "precio")

    def test_declarar_el_objetivo_TAMBIEN_como_entrada_bloquea(self):
        conf = confirmar_desde_csv(
            self.CSV, objetivo="precio", entradas=("metros", "barrio", "precio"),
            unidad_de_observacion="una vivienda")
        self.assertEqual(_claves(conf.bloqueos), ["objetivo_entre_las_entradas"])
        self.assertIsNone(conf.problema)

    def test_sin_declarar_entradas_se_proponen_todas_menos_el_objetivo(self):
        conf = confirmar_desde_csv(self.CSV, objetivo="precio",
                                   unidad_de_observacion="una vivienda")
        self.assertNotIn("precio", conf.problema.predictors)

    def test_sin_declarar_entradas_row_id_tampoco_es_un_predictor(self):
        """108-C2, hallazgo real medido en navegador: sin `entradas`
        declaradas, `row_id` (el identificador de observación por
        omisión en TODO el producto, `proponer_particion` 103-C4/108-
        C1/C2) se proponía como predictor -- invisible mientras nadie
        entrenaba de verdad con la lista (`diagnostico_de_riesgo_view.py`
        la calcula pero nunca prepara datos con ella)."""
        csv_con_row_id = _csv("row_id,metros,barrio,precio",
                              [f"{i},{60 + i},b{i % 3},{100000 + i * 1000}" for i in range(20)])
        conf = confirmar_desde_csv(csv_con_row_id, objetivo="precio",
                                   unidad_de_observacion="una vivienda")
        self.assertTrue(conf.confirmado)
        self.assertNotIn("row_id", conf.problema.predictors)
        self.assertIn("metros", conf.problema.predictors)
        self.assertIn("barrio", conf.problema.predictors)

    def test_declarar_row_id_explicitamente_en_entradas_SI_lo_deja_pasar(self):
        """La exclusión es solo del AUTO-derivado (sin `entradas`) -- si
        alguien lo pide a propósito, no es a esta función a la que le
        toca impedirlo."""
        csv_con_row_id = _csv("row_id,metros,barrio,precio",
                              [f"{i},{60 + i},b{i % 3},{100000 + i * 1000}" for i in range(20)])
        conf = confirmar_desde_csv(csv_con_row_id, objetivo="precio",
                                   entradas=("row_id", "metros", "barrio"),
                                   unidad_de_observacion="una vivienda")
        self.assertIn("row_id", conf.problema.predictors)


# ---------------------------------------------------------------------------
# La ruta CSV: proponer con motivo y confirmar antes de entrenar
# ---------------------------------------------------------------------------

class TestLaRutaCsvProponeYConfirma(unittest.TestCase):

    CSV = _csv("edad,tension,riesgo",
               [f"{20 + i},{100 + i},{i % 3 + 1}" for i in range(30)])

    def test_sin_objetivo_elegido_se_proponen_candidatas_con_su_motivo(self):
        conf = confirmar_desde_csv(self.CSV)
        self.assertIsNone(conf.problema)
        self.assertEqual(_claves(conf.preguntas), ["objetivo_no_elegido"])
        self.assertTrue(conf.preguntas[0].opciones)
        # Los motivos son los que ya calcula `analyze_dataset_csv`, con su
        # código estructurado: aquí no se redacta una segunda explicación.
        for candidato in conf.propuesta["candidatos"]:
            self.assertTrue(candidato["reason_codes"], candidato)

    def test_un_objetivo_que_no_existe_bloquea_diciendo_las_columnas(self):
        conf = confirmar_desde_csv(self.CSV, objetivo="no_existe")
        self.assertEqual(_claves(conf.bloqueos), ["objetivo_inexistente"])
        self.assertIn("edad", conf.bloqueos[0].motivo["es"])

    def test_un_identificador_o_una_fecha_no_son_objetivos(self):
        csv = _csv("id,fecha,valor",
                   [f"{1000 + i},2026-01-{i + 1:02d},{i}" for i in range(20)])
        for columna in ("id", "fecha"):
            with self.subTest(columna=columna):
                conf = confirmar_desde_csv(csv, objetivo=columna)
                self.assertEqual(_claves(conf.bloqueos), ["objetivo_no_predecible"])

    def test_etiquetas_numericas_con_pocos_valores_PREGUNTAN_el_tipo_de_tarea(self):
        """Y no se deciden en silencio: 1/2/3 pueden ser tres niveles de riesgo
        o una magnitud que se repite, y la diferencia cambia la salida del
        modelo, las métricas y lo que significa cada número."""
        conf = confirmar_desde_csv(self.CSV, objetivo="riesgo",
                                   unidad_de_observacion="un paciente")
        pregunta = [p for p in conf.preguntas if p.clave == "tipo_de_tarea"]
        self.assertEqual(len(pregunta), 1, _claves(conf.preguntas))
        self.assertEqual(pregunta[0].opciones,
                         ("multiclass_classification", "regression"))
        self.assertIsNone(conf.problema)

    def test_el_producto_decidia_solo_y_ademas_distinto_en_cada_sitio(self):
        """El defecto que justifica la pregunta, MEDIDO sobre el mismo CSV: el
        análisis propone esa columna como CLASIFICACIÓN y la generación la
        entrena como REGRESIÓN. Dos sitios decidiendo lo mismo, y distinto."""
        analisis = analyze_dataset_csv(self.CSV)
        propuesto = [c for c in analisis["target_candidates"] if c["column"] == "riesgo"]
        self.assertEqual(propuesto[0]["task"], "classification")
        generado = generate_project_from_dataset(self.CSV, "riesgo")
        self.assertEqual(generado["provenance"]["preparation_spec"]["task"], "regression")
        # Y ahora hay una pregunta donde antes había dos decisiones calladas.
        self.assertIn("tipo_de_tarea",
                      [p["clave"] for p in generado["confirmacion"]["preguntas"]])

    def test_contestar_el_tipo_de_tarea_confirma_el_problema(self):
        conf = confirmar_desde_csv(self.CSV, objetivo="riesgo",
                                   tarea="multiclass_classification",
                                   unidad_de_observacion="un paciente")
        self.assertTrue(conf.confirmado, _claves(conf.preguntas))
        self.assertEqual(conf.problema.task, "multiclass_classification")
        self.assertEqual(len(conf.problema.classes), 3)

    def test_el_proceso_actual_108_c1_viaja_hasta_el_problema_confirmado(self):
        """`current_process` (108-C1) es opcional y no tiene reglas propias de
        esta ruta -- solo tiene que llegar intacto hasta el `ProblemSpec`, el
        mismo objeto que se le pasó, no una copia reconstruida."""
        proceso = ProcesoActual(tipo="columna", columna="tension")
        conf = confirmar_desde_csv(self.CSV, objetivo="riesgo",
                                   tarea="multiclass_classification",
                                   unidad_de_observacion="un paciente",
                                   proceso_actual=proceso)
        self.assertTrue(conf.confirmado, _claves(conf.preguntas))
        self.assertEqual(conf.problema.current_process, proceso)
        # Y por omisión, nadie lo ha preguntado -- `None`, no `tipo="ninguno"`.
        sin_proceso = confirmar_desde_csv(self.CSV, objetivo="riesgo",
                                          tarea="multiclass_classification",
                                          unidad_de_observacion="un paciente")
        self.assertIsNone(sin_proceso.problema.current_process)

    def test_una_binaria_pregunta_cual_es_su_clase_positiva(self):
        csv = _csv("edad,impago", [f"{20 + i},{'si' if i % 2 else 'no'}"
                                   for i in range(20)])
        conf = confirmar_desde_csv(csv, objetivo="impago",
                                   unidad_de_observacion="un cliente")
        pregunta = [p for p in conf.preguntas if p.clave == "clase_positiva"]
        self.assertEqual(len(pregunta), 1, _claves(conf.preguntas))
        self.assertEqual(set(pregunta[0].opciones), {"si", "no"})
        self.assertIsNone(conf.problema)

        conf = confirmar_desde_csv(csv, objetivo="impago", clase_positiva="si",
                                   unidad_de_observacion="un cliente")
        self.assertTrue(conf.confirmado)
        self.assertEqual(conf.problema.positive_label, "si")

    def test_una_clase_positiva_que_no_esta_entre_las_clases_vuelve_a_preguntar(self):
        csv = _csv("edad,impago", [f"{20 + i},{'si' if i % 2 else 'no'}"
                                   for i in range(20)])
        conf = confirmar_desde_csv(csv, objetivo="impago", clase_positiva="quizas",
                                   unidad_de_observacion="un cliente")
        self.assertIn("clase_positiva", _claves(conf.preguntas))
        self.assertIsNone(conf.problema)

    def test_la_unidad_de_observacion_no_se_rellena_con_una_fila(self):
        conf = confirmar_desde_csv(self.CSV, objetivo="riesgo",
                                   tarea="multiclass_classification")
        self.assertIn("unidad_de_observacion", _claves(conf.preguntas))
        self.assertIsNone(conf.problema)

    def test_con_fecha_se_pregunta_el_momento_y_despues_el_horizonte(self):
        """«Cuando aplique»: sin fecha no se pregunta —pedir un momento a un
        dataset transversal sería pedir un dato que no existe— y con fecha sí."""
        csv = _csv("fecha,tension,ingreso",
                   [f"2026-01-{i + 1:02d},{100 + i},{'si' if i % 2 else 'no'}"
                    for i in range(20)])
        conf = confirmar_desde_csv(csv, objetivo="ingreso", clase_positiva="si",
                                   unidad_de_observacion="un episodio")
        self.assertIn("momento_de_prediccion", _claves(conf.preguntas))

        conf = confirmar_desde_csv(csv, objetivo="ingreso", clase_positiva="si",
                                   unidad_de_observacion="un episodio",
                                   momento_de_prediccion="al triaje")
        self.assertIn("horizonte_del_desenlace", _claves(conf.preguntas))

        conf = confirmar_desde_csv(csv, objetivo="ingreso", clase_positiva="si",
                                   unidad_de_observacion="un episodio",
                                   momento_de_prediccion="al triaje",
                                   horizonte=Horizonte(48.0, "hours"))
        self.assertTrue(conf.confirmado, _claves(conf.preguntas))
        self.assertEqual(conf.problema.horizon.magnitud, 48.0)

    def test_sin_fecha_no_se_pregunta_por_el_momento(self):
        conf = confirmar_desde_csv(self.CSV, objetivo="riesgo",
                                   tarea="multiclass_classification",
                                   unidad_de_observacion="un paciente")
        self.assertNotIn("momento_de_prediccion", _claves(conf.preguntas))

    def test_declarar_clases_que_no_estan_en_los_datos_bloquea(self):
        conf = confirmar_desde_csv(
            self.CSV, objetivo="riesgo", tarea="multiclass_classification",
            clases=("1", "2", "3", "4"), unidad_de_observacion="un paciente")
        self.assertEqual(_claves(conf.bloqueos), ["clases_declaradas_que_no_estan"])

    def test_generar_el_proyecto_publica_la_confirmacion_y_no_la_impone(self):
        """La generación sigue funcionando igual que antes de este corte: lo que
        aparece al lado es qué falta por confirmar."""
        res = generate_project_from_dataset(self.CSV, "riesgo")
        self.assertTrue(res["ok"])
        self.assertIn("mxai", res)
        self.assertFalse(res["confirmacion"]["confirmado"])
        self.assertTrue(res["confirmacion"]["preguntas"])


# ---------------------------------------------------------------------------
# Criterio 5 — una única clase en train no es entrenable
# ---------------------------------------------------------------------------

class TestUnObjetivoConUnaSolaClaseEnTrain(unittest.TestCase):
    """El CSV tiene DOS clases y el entrenamiento ve UNA: el entrenador denso
    no baraja y corta por posición al 80 %, así que un fichero ordenado por la
    columna objetivo deja toda una clase fuera. La pérdida llega a 0 acertando
    siempre lo mismo, y ese 0 se lee como acierto perfecto."""

    ORDENADO = _csv("x,y", [f"{i},no" for i in range(16)] + [f"{i},si" for i in range(4)])
    CONSTANTE = _csv("x,y", [f"{i},no" for i in range(20)])

    def test_el_csv_tiene_dos_clases_y_el_entrenamiento_ve_una(self):
        valores = ["no"] * 16 + ["si"] * 4
        self.assertEqual(len(set(valores)), 2)
        bloqueo = no_entrenable_en_train(valores, objetivo="y")
        self.assertIsNotNone(bloqueo)
        self.assertEqual(bloqueo.clave, "objetivo_con_una_sola_clase")
        self.assertIn("16 filas de entrenamiento", bloqueo.motivo["es"])
        self.assertIn("16 training rows", bloqueo.motivo["en"])

    def test_la_confirmacion_lo_declara_no_entrenable_con_su_motivo(self):
        conf = confirmar_desde_csv(self.ORDENADO, objetivo="y", clase_positiva="si",
                                   unidad_de_observacion="un caso")
        self.assertIn("objetivo_con_una_sola_clase", _claves(conf.bloqueos))
        self.assertIsNone(conf.problema)
        with self.assertRaises(ObjetivoSinConfirmar):
            exigir_problema_confirmado(conf)

    def test_la_comprobacion_del_csv_entero_NO_lo_veia(self):
        """`constant_target_error` mira el fichero entero y acierta cuando la
        columna es constante de arriba abajo; este caso se le escapaba, y por
        eso hace falta la comprobación sobre la partición."""
        self.assertIsNone(constant_target_error(self.ORDENADO, "y"))

    def test_las_dos_comprobaciones_dicen_lo_MISMO_cuando_la_columna_es_constante(self):
        """El antídoto de «dos sitios declarando lo mismo acaban divergiendo»:
        sobre un objetivo constante las dos tienen que hablar."""
        self.assertIsNotNone(constant_target_error(self.CONSTANTE, "y"))
        self.assertIsNotNone(no_entrenable_en_train(["no"] * 20, objetivo="y"))

    def test_un_objetivo_que_varia_dentro_de_train_no_se_bloquea(self):
        mezclado = ["no" if i % 2 else "si" for i in range(20)]
        self.assertIsNone(no_entrenable_en_train(mezclado, objetivo="y"))

    def test_no_se_confunde_la_falta_de_datos_con_una_sola_clase(self):
        """Cero valores no nulos en train no es «una sola clase»: es que no hay
        objetivo con el que entrenar, y ése es otro hecho. Decirlo con este
        motivo mandaría a alguien a buscar variedad donde faltan datos."""
        self.assertIsNone(no_entrenable_en_train([""] * 20, objetivo="y"))
        self.assertIsNone(no_entrenable_en_train(["NA"] * 20, objetivo="y"))

    def test_los_nulos_no_cuentan_como_una_clase_mas(self):
        valores = ["no"] * 8 + [""] * 8 + ["si"] * 4
        self.assertIsNotNone(no_entrenable_en_train(valores, objetivo="y"))

    def test_una_particion_declarada_manda_sobre_el_corte_por_defecto(self):
        valores = ["no"] * 16 + ["si"] * 4
        self.assertIsNone(no_entrenable_en_train(
            valores, objetivo="y", filas_de_train=[0, 1, 2, 17, 18, 19]))

    def test_el_corte_es_el_del_ENTRENADOR_REAL(self):
        """Contrastado contra `DenseSupervisedTrainer`, no contra la fórmula
        copiada: si el entrenador cambia su reparto, este caso se pone rojo y
        el aviso deja de mentir sobre qué filas entrenan."""
        for filas in (10, 20, 37, 100):
            with self.subTest(filas=filas):
                self.assertEqual(corte_de_entrenamiento(filas),
                                 filas - len(self._validacion_real(filas)))

    def test_el_entrenador_no_baraja_asi_que_train_son_las_primeras_filas(self):
        """Lo importante no es el 80 %: es que el corte es por POSICIÓN."""
        self.assertEqual(self._validacion_real(10), [8, 9])

    @staticmethod
    def _validacion_real(filas: int) -> list[int]:
        """Qué filas ve la VALIDACIÓN en un entrenamiento denso de verdad.

        `x1` es el índice de la fila, así que las filas que llegan al evaluador
        se identifican una a una. Misma técnica que
        `tests/test_biblioteca_c3_split_temporal.py`.
        """
        import matrixai.training.dense_trainer as dt
        from matrixai.training.parser import parse_training_file

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            (raiz / "model.mxai").write_text(
                "PROJECT T\n\nVECTOR V[2]\n  x1: Scalar\n  x2: Scalar\nEND\n\n"
                "NETWORK Net\n  INPUT V\n  LAYER Dense units=2 activation=relu\n"
                "  LAYER Dense units=1 activation=linear\n  OUTPUT y: Scalar\nEND\n\n"
                "GRAPH\n  V -> Net\nEND\n", encoding="utf-8")
            (raiz / "data.csv").write_text(
                "\n".join(["x1,x2,y"] + [f"{i},0,{i}" for i in range(filas)]) + "\n",
                encoding="utf-8")
            (raiz / "train.mxtrain").write_text(
                'MODEL model.mxai\n\nDATASET D\n  SOURCE csv("data.csv")\n'
                "  INPUT V FROM COLUMNS [x1, x2]\n  TARGET y: Scalar\nEND\n\n"
                "LOSS L\n  TYPE mse\n  PREDICTION y\n  TARGET y\nEND\n\n"
                "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.01\n  UPDATE Net.*\nEND\n\n"
                "RUN\n  EPOCHS 1\nEND\n", encoding="utf-8")
            visto: dict = {}
            real = dt.evaluate_dense_network

            def _espia(net, ps, examples, loss_fn, labels=None):
                visto["val"] = sorted(int(x[0]) for x, _ in examples)
                return real(net, ps, examples, loss_fn, labels=labels)

            with mock.patch.object(dt, "evaluate_dense_network", _espia):
                dt.DenseSupervisedTrainer().train(
                    parse_training_file(raiz / "train.mxtrain"),
                    output_dir=str(raiz / "out"), base_path=raiz)
            return visto["val"]


# ---------------------------------------------------------------------------
# Criterio 3 — la clase positiva y el orden de las clases viajan al manifiesto
# ---------------------------------------------------------------------------

class TestElProblemaViajaEnElManifiesto(unittest.TestCase):

    PROBLEMA = ProblemSpec(
        problem_id="ingreso-uci", target="ingreso_uci",
        task="binary_classification", observation_unit="un episodio de urgencias",
        classes=("no", "si"), positive_label="si", predictors=("edad", "lactato"),
        prediction_time="al triaje", horizon=Horizonte(48.0, "hours"),
        intended_use="apoyo a la decisión; no sustituye al criterio clínico")

    def _manifiesto(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.mxai").write_text("PROJECT X\n", encoding="utf-8")
            return build_reproduce_manifest(tmp, **kwargs)

    def test_la_clase_positiva_y_las_clases_viajan(self):
        manifiesto = self._manifiesto(problem=self.PROBLEMA)
        self.assertEqual(manifiesto["problem"]["classes"], ["no", "si"])
        self.assertEqual(manifiesto["problem"]["positive_label"], "si")
        self.assertEqual(manifiesto["problem"]["target"], "ingreso_uci")
        self.assertEqual(manifiesto["problem"]["task"], "binary_classification")

    def test_el_ORDEN_de_las_clases_se_conserva_y_no_se_ordena(self):
        """`classes` no es un conjunto: es la correspondencia entre columna de
        probabilidades y clase. Ordenarlo alfabéticamente cambiaría qué
        significa la columna 0."""
        bloque = build_problem_block(
            {"target": "y", "task": "multiclass_classification",
             "classes": ["zeta", "alfa", "media"]})
        self.assertEqual(bloque["classes"], ["zeta", "alfa", "media"])

    def test_sin_problema_confirmado_el_manifiesto_dice_null_y_no_un_hueco(self):
        """Un valor ausente no es un cero: un paquete que no declara su problema
        no es un paquete cuyo problema esté vacío."""
        manifiesto = self._manifiesto()
        self.assertIn("problem", manifiesto)
        self.assertIsNone(manifiesto["problem"])

    def test_el_digest_del_manifiesto_cubre_el_problema(self):
        """Si el bloque no entrara en la huella, alguien podría cambiar la clase
        positiva de un paquete firmado sin que nada lo notara."""
        uno = self._manifiesto(problem=self.PROBLEMA)
        otro = self._manifiesto(problem={
            "target": "ingreso_uci", "task": "binary_classification",
            "classes": ["si", "no"], "positive_label": "si"})
        self.assertNotEqual(uno["manifest_sha256"], otro["manifest_sha256"])

    def test_la_version_de_esquema_sube_y_verify_la_sigue_conociendo(self):
        from matrixai.export.verify import _ESQUEMAS_CONOCIDOS

        manifiesto = self._manifiesto(problem=self.PROBLEMA)
        self.assertEqual(manifiesto["schema_version"], "1.1")
        self.assertIn(manifiesto["schema_version"], _ESQUEMAS_CONOCIDOS)

    def test_una_binaria_sin_clase_positiva_no_se_publica(self):
        with self.assertRaises(ReproduceManifestError) as caja:
            build_problem_block({"target": "y", "task": "binary_classification",
                                 "classes": ["no", "si"]})
        self.assertIn("positive_label", str(caja.exception))

    def test_una_clase_positiva_que_no_esta_entre_las_clases_no_se_publica(self):
        with self.assertRaises(ReproduceManifestError):
            build_problem_block({"target": "y", "task": "binary_classification",
                                 "classes": ["no", "si"], "positive_label": "quizas"})

    def test_una_clasificacion_sin_clases_no_se_publica(self):
        with self.assertRaises(ReproduceManifestError):
            build_problem_block({"target": "y", "task": "multiclass_classification"})

    def test_una_regresion_no_lleva_clases_ni_clase_positiva(self):
        for cuerpo in ({"target": "y", "task": "regression", "classes": ["a", "b"]},
                       {"target": "y", "task": "regression", "positive_label": "a"}):
            with self.subTest(cuerpo=cuerpo):
                with self.assertRaises(ReproduceManifestError):
                    build_problem_block(cuerpo)

    def test_una_tarea_que_no_existe_no_se_publica(self):
        with self.assertRaises(ReproduceManifestError):
            build_problem_block({"target": "y", "task": "ranking"})

    def test_una_clave_desconocida_se_rechaza_en_vez_de_interpretarse_a_medias(self):
        with self.assertRaises(ReproduceManifestError):
            build_problem_block({"target": "y", "task": "regression", "extra": 1})

    def test_un_horizonte_sin_momento_no_se_publica(self):
        with self.assertRaises(ReproduceManifestError):
            build_problem_block({"target": "y", "task": "regression",
                                 "horizon": {"magnitud": 48, "unidad": "hours"}})

    def test_una_clase_repetida_no_se_publica(self):
        with self.assertRaises(ReproduceManifestError):
            build_problem_block({"target": "y", "task": "multiclass_classification",
                                 "classes": ["a", "b", "a"]})

    def test_el_problema_confirmado_de_la_ruta_csv_llega_al_manifiesto(self):
        """El recorrido entero: CSV → confirmación → `ProblemSpec` → manifiesto,
        sin que nadie reescriba por el camino el orden de las clases."""
        csv = _csv("edad,impago", [f"{20 + i},{'si' if i % 2 else 'no'}"
                                   for i in range(20)])
        conf = confirmar_desde_csv(csv, objetivo="impago", clase_positiva="si",
                                   unidad_de_observacion="un cliente")
        manifiesto = self._manifiesto(problem=conf.problema)
        self.assertEqual(manifiesto["problem"]["classes"],
                         list(conf.problema.classes))
        self.assertEqual(manifiesto["problem"]["positive_label"], "si")


# ---------------------------------------------------------------------------
# La puerta
# ---------------------------------------------------------------------------

class TestLaPuertaDelEstudio(unittest.TestCase):

    def test_enumera_lo_que_falta_en_vez_de_decir_no_valido(self):
        conf = confirmar_desde_csv(
            _csv("edad,impago", [f"{20 + i},{'si' if i % 2 else 'no'}"
                                 for i in range(20)]),
            objetivo="impago")
        with self.assertRaises(ObjetivoSinConfirmar) as caja:
            exigir_problema_confirmado(conf)
        for clave in ("clase_positiva", "unidad_de_observacion"):
            self.assertIn(clave, caja.exception.es)
            self.assertIn(clave, caja.exception.en)

    def test_con_todo_contestado_devuelve_el_problema(self):
        conf = confirmar_desde_csv(
            _csv("edad,impago", [f"{20 + i},{'si' if i % 2 else 'no'}"
                                 for i in range(20)]),
            objetivo="impago", clase_positiva="si",
            unidad_de_observacion="un cliente")
        self.assertIsInstance(exigir_problema_confirmado(conf), ProblemSpec)

    def test_el_documento_entero_es_serializable_a_json(self):
        """Lo que no se pueda publicar tal cual no le sirve al C5."""
        conf = confirmar_desde_csv(
            _csv("edad,impago", [f"{20 + i},{'si' if i % 2 else 'no'}"
                                 for i in range(20)]),
            objetivo="impago")
        texto = json.dumps(conf.a_json(), ensure_ascii=False)
        self.assertIn("clase_positiva", texto)
        self.assertNotIn("{campo}", texto)
        self.assertNotIn("{valor}", texto)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
