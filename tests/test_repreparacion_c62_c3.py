# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 62 C3 — reimportar el CSV re-EJECUTA la preparación, no la imita.

El parche publicado en v1.4 (`remapOriginalColumnNames` + `remapTargetColumnValues`)
reescribía texto en la SPA: renombraba la cabecera y mapeaba los valores del
target. Eso solo cubre un target escalar o binario; con categóricas one-hot o
con una serie temporal el CSV reimportado no se parece al que entrenó el modelo,
y reentrenar tras recargar era imposible.

Aquí se prueba la vía correcta: `prepare_dataset_from_provenance` reutiliza la
implementación real con la receta congelada en la procedencia
(`preparation_spec`), y la verificación por hash decide si el resultado es
fiable, se avisa, o se aborta (invariante 4).
"""
from __future__ import annotations

import json
import unittest

from matrixai.training.dataset_project import (
    PREPARATION_SPEC_VERSION,
    DatasetProjectError,
    generate_project_from_dataset,
    generate_temporal_project_from_dataset,
    prepare_dataset_from_provenance,
)

CSV_CAT = (
    "a,ciudad,target\n"
    "0.1,MADRID,0\n0.5,BILBAO,1\n0.9,VIGO,0\n0.2,MADRID,1\n"
    "0.7,BILBAO,0\n0.3,VIGO,1\n0.4,MADRID,0\n0.6,BILBAO,1\n"
)
CSV_TEMP = (
    "fecha,temp,target\n"
    "2024-01-01,10,1.0\n2024-01-02,11,2.0\n2024-01-03,12,3.0\n2024-01-04,13,4.0\n"
    "2024-01-05,14,5.0\n2024-01-06,15,6.0\n2024-01-07,16,7.0\n2024-01-08,17,8.0\n"
    "2024-01-09,18,9.0\n2024-01-10,19,10.0\n"
)


class TestLaRecetaSeCongela(unittest.TestCase):
    def test_la_procedencia_trae_receta_versionada(self):
        prov = generate_project_from_dataset(CSV_CAT, target_column="target")["provenance"]
        spec = prov["preparation_spec"]
        self.assertEqual(spec["version"], PREPARATION_SPEC_VERSION)
        for clave in ("feature_columns", "feature_name_map", "column_types",
                      "category_vocabularies", "target_column", "target_header",
                      "task", "target_label_map"):
            self.assertIn(clave, spec)

    def test_el_vocabulario_efectivo_queda_congelado(self):
        """Sin esto no hay reproducción: para una categórica sin override el
        vocabulario se calculaba al vuelo, así que un CSV con un valor de más
        habría producido otras columnas one-hot en silencio."""
        prov = generate_project_from_dataset(CSV_CAT, target_column="target")["provenance"]
        vocab = prov["preparation_spec"]["category_vocabularies"]["ciudad"]
        self.assertEqual(vocab, ["MADRID", "BILBAO", "VIGO"], "orden de aparición")


class TestReproduccionExacta(unittest.TestCase):
    def test_categoricas_one_hot_se_reproducen_byte_a_byte(self):
        """El caso que el parche de v1.4 NO cubría."""
        res = generate_project_from_dataset(CSV_CAT, target_column="target")
        self.assertIn("ciudad__madrid", res["csv_text"])  # hay one-hot de verdad
        re = prepare_dataset_from_provenance(CSV_CAT, res["provenance"])
        self.assertEqual(re.csv_text, res["csv_text"])
        self.assertTrue(re.compatibility.ok)
        self.assertTrue(re.compatibility.prepared_matches)
        self.assertEqual(re.compatibility.errors, [])

    def test_serie_temporal_se_reproduce_byte_a_byte(self):
        """El otro caso que el parche no cubría: hay que re-ejecutar el
        pipeline (sort/shift/lag) desde el bloque ESTRUCTURADO, nunca desde
        las cadenas descriptivas de `operations`."""
        res = generate_temporal_project_from_dataset(
            CSV_TEMP, "target", temporal_column="fecha", horizon=1)
        re = prepare_dataset_from_provenance(CSV_TEMP, res["provenance"])
        self.assertEqual(re.csv_text, res["csv_text"])
        self.assertTrue(re.compatibility.prepared_matches)

    def test_no_regenera_arquitectura_ni_entrenamiento(self):
        """La API prepara datos y nada más (invariante 2)."""
        res = generate_project_from_dataset(CSV_CAT, target_column="target")
        re = prepare_dataset_from_provenance(CSV_CAT, res["provenance"])
        self.assertFalse(hasattr(re, "mxai"))
        self.assertNotIn("mxai", re.provenance)


class TestPoliticaDeCompatibilidad(unittest.TestCase):
    def setUp(self):
        self.res = generate_project_from_dataset(CSV_CAT, target_column="target")
        self.prov = self.res["provenance"]

    def test_columna_requerida_ausente_es_error(self):
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance("a,target\n0.1,0\n0.5,1\n", self.prov)
        self.assertIn("ciudad", str(ctx.exception))

    def test_columna_adicional_se_ignora_con_aviso(self):
        csv = CSV_CAT.replace("a,ciudad,target", "a,ciudad,target,sobra")
        csv = "\n".join(
            line + (",z" if i > 0 and line else "")
            for i, line in enumerate(csv.split("\n"))
        )
        re = prepare_dataset_from_provenance(csv, self.prov)
        self.assertTrue(re.compatibility.ok)
        self.assertTrue(any("sobra" in w for w in re.compatibility.warnings))
        self.assertNotIn("sobra", re.csv_text)

    def test_categoria_nueva_es_error_nunca_amplia_el_vector(self):
        csv = CSV_CAT.replace("0.4,MADRID,0", "0.4,SEVILLA,0")
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(csv, self.prov)
        msg = str(ctx.exception)
        self.assertIn("SEVILLA", msg)
        self.assertIn("regenerar", msg)

    def test_el_orden_de_las_columnas_no_importa(self):
        filas = [l.split(",") for l in CSV_CAT.strip().split("\n")]
        reordenado = "\n".join(",".join([f[2], f[0], f[1]]) for f in filas) + "\n"
        re = prepare_dataset_from_provenance(reordenado, self.prov)
        self.assertTrue(re.compatibility.ok)
        # Mismas columnas preparadas pese al orden distinto del crudo.
        self.assertEqual(
            re.csv_text.split("\n")[0], self.res["csv_text"].split("\n")[0])

    def test_expected_columns_ata_el_resultado_al_modelo_cargado(self):
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(
                CSV_CAT, self.prov, expected_columns=["otra_cosa"])
        self.assertIn("no encaja con el modelo", str(ctx.exception))


class TestVerificacionYDegradacion(unittest.TestCase):
    """Invariante 4: el fallo es informado, ni mudo ni innecesariamente fatal."""

    def setUp(self):
        self.res = generate_project_from_dataset(CSV_CAT, target_column="target")
        self.prov = self.res["provenance"]

    def test_version_desconocida_aborta_por_defecto_pero_ofrece_salida(self):
        """Dos exigencias que parecían chocar y no chocan: el contrato pide que
        una actualización del core no deje inservibles los modelos guardados
        (invariante 4), y la auditoría pide no ejecutar una versión desconocida
        como si fuera compatible. Solución: abortar por defecto con un error
        accionable, y permitir seguir SOLO con confirmación explícita, dejando
        constancia en el informe."""
        otra = json.loads(json.dumps(self.prov))
        otra["preparation_spec"]["version"] = PREPARATION_SPEC_VERSION + 98

        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(CSV_CAT, otra)
        self.assertIn("no sabe reproducir", str(ctx.exception))
        self.assertEqual(
            ctx.exception.details["error_kind"], "incompatible_preparation_spec")

        re = prepare_dataset_from_provenance(
            CSV_CAT, otra, allow_incompatible_spec=True)
        self.assertTrue(re.compatibility.ok)
        self.assertTrue(any("desconocida" in w for w in re.compatibility.warnings))

    def test_procedencia_legacy_se_deduce_y_se_demuestra_con_el_hash(self):
        legacy = dict(self.prov)
        legacy.pop("preparation_spec")
        re = prepare_dataset_from_provenance(CSV_CAT, legacy)
        self.assertTrue(re.compatibility.legacy_adapter)
        self.assertTrue(re.compatibility.prepared_matches)
        self.assertEqual(re.csv_text, self.res["csv_text"])

    def test_legacy_sin_datos_minimos_pide_regenerar_sin_romper_el_modelo(self):
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(CSV_CAT, {"source": "user_upload"})
        msg = str(ctx.exception)
        self.assertIn("Regenera el proyecto", msg)
        self.assertIn("inferir y exportar", msg)

    def test_mismo_crudo_misma_receta_y_resultado_distinto_aborta(self):
        """No es versionado: es un fallo de reproducibilidad, y con un dataset
        que no es el esperado no se reentrena."""
        corrupta = json.loads(json.dumps(self.prov))
        corrupta["prepared_csv_sha256"] = "0" * 64
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(CSV_CAT, corrupta)
        self.assertIn("reproducibilidad", str(ctx.exception))

    def test_crudo_distinto_genera_procedencia_nueva_con_referencia(self):
        mas = CSV_CAT + "0.8,VIGO,0\n0.15,MADRID,1\n"
        re = prepare_dataset_from_provenance(mas, self.prov)
        self.assertTrue(re.compatibility.ok)
        self.assertFalse(re.compatibility.same_raw_csv)
        # No se exige el hash antiguo del preparado.
        self.assertIsNone(re.compatibility.prepared_matches)
        self.assertIn("parent_provenance_sha256", re.provenance)
        self.assertNotEqual(
            re.provenance["raw_csv_sha256"], self.prov["raw_csv_sha256"])

    def test_el_hash_crudo_es_de_los_bytes_TAL_CUAL_llegan(self):
        """Identidad canónica ANTES de normalizar BOM/delimitador/EOL: si se
        hashea a un lado y se verifica al otro, la garantía se rompe en
        silencio."""
        con_bom = "﻿" + CSV_CAT
        re = prepare_dataset_from_provenance(con_bom, self.prov)
        self.assertFalse(re.compatibility.same_raw_csv, "el BOM cambia el crudo")
        # Pero el contenido sigue siendo compatible y se prepara igual.
        self.assertTrue(re.compatibility.ok)
        self.assertEqual(re.csv_text, self.res["csv_text"])


class TestRecetaCorruptaOIncompatible(unittest.TestCase):
    """AUDITORÍA C3 [MEDIO-ALTO]: una receta incompleta reventaba con
    `KeyError: 'feature_columns'` y el endpoint lo convertía en un 500. Y la
    versión era NOMINAL: ante una desconocida se avisaba pero se ejecutaba el
    preparador actual igual, que es justo lo que el versionado debe impedir."""

    def test_receta_incompleta_da_error_funcional_no_KeyError(self):
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(CSV_CAT, {"preparation_spec": {"version": 1}})
        msg = str(ctx.exception)
        self.assertIn("incompleta", msg)
        self.assertIn("Regenera el proyecto", msg)

    def test_receta_con_feature_sin_nombre_interno(self):
        spec = {
            "version": 1, "feature_columns": ["a", "huerfana"],
            "feature_name_map": {"a": "a"}, "column_types": {"a": "number"},
            "target_column": "target", "target_header": "predicted_class",
            "task": "classification",
        }
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(CSV_CAT, {"preparation_spec": spec})
        self.assertIn("huerfana", str(ctx.exception))

    def test_version_sin_preparador_NO_se_ejecuta_con_el_actual(self):
        spec = {
            "version": 99, "feature_columns": ["a"], "feature_name_map": {"a": "a"},
            "column_types": {"a": "number"}, "target_column": "target",
            "target_header": "predicted_class", "task": "classification",
        }
        with self.assertRaises(DatasetProjectError) as ctx:
            prepare_dataset_from_provenance(CSV_CAT, {"preparation_spec": spec})
        msg = str(ctx.exception)
        self.assertIn("v99", msg)
        self.assertIn("no sabe reproducir", msg)

    def test_hay_preparador_registrado_para_la_version_actual(self):
        from matrixai.training.dataset_project import PREPARERS
        self.assertIn(PREPARATION_SPEC_VERSION, PREPARERS)

    def test_receta_no_dict_se_rechaza(self):
        with self.assertRaises(DatasetProjectError):
            prepare_dataset_from_provenance(CSV_CAT, {"preparation_spec": "texto"})


class TestProcedenciaDelCSVNuevo(unittest.TestCase):
    """REAUDITORÍA [ALTO] — la procedencia se copiaba y solo se cambiaban los
    hashes, así que `schema_inferred` seguía describiendo el CSV PADRE: para un
    CSV nuevo con rango [10,15] la procedencia seguía diciendo [0,5]."""

    VIEJO = "a,target\n0,0\n1,1\n2,0\n3,1\n4,0\n5,1\n"
    NUEVO = "a,target\n10,0\n11,1\n12,0\n13,1\n14,0\n15,1\n"

    def test_el_esquema_contractual_NO_cambia(self):
        """Invariante 3: re-preparar reproduce el esquema con el que nació el
        modelo. El congelado no se toca nunca."""
        p = generate_project_from_dataset(self.VIEJO, target_column="target")
        re = prepare_dataset_from_provenance(self.NUEVO, p["provenance"])
        self.assertEqual(
            re.provenance["schema_final"]["a"]["observed_range"],
            p["provenance"]["schema_final"]["a"]["observed_range"])

    def test_pero_SE_REGISTRA_lo_observado_ahora(self):
        p = generate_project_from_dataset(self.VIEJO, target_column="target")
        re = prepare_dataset_from_provenance(self.NUEVO, p["provenance"])
        observado = re.provenance["observed_schema"]["a"]["observed_range"]
        self.assertEqual([int(observado[0]), int(observado[1])], [10, 15])

    def test_marca_de_tiempo_de_la_repreparacion(self):
        p = generate_project_from_dataset(self.VIEJO, target_column="target")
        re = prepare_dataset_from_provenance(self.NUEVO, p["provenance"])
        self.assertIn("reprepared_at", re.provenance)
        self.assertIn("parent_provenance_sha256", re.provenance)


class TestBordesDeCSVReal(unittest.TestCase):
    def test_comas_y_saltos_entre_comillas(self):
        """Un CSV real trae texto entrecomillado con comas y saltos dentro. La
        columna va con 12 filas todas distintas para que quede clasificada como
        identificador (y por tanto excluida): una CATEGÓRICA con comas la
        rechaza a propósito `_check_categorical_values_safe`, porque el prompt
        tipado no puede representarla sin ambigüedad. Lo que se prueba aquí es
        que el PARSEO entrecomillado sobrevive al viaje de ida y vuelta."""
        filas = "".join(
            f'"nota {i}, con coma\ny salto",{i / 10:.1f},{i % 2}\n' for i in range(12)
        )
        csv = "nota,valor,target\n" + filas
        res = generate_project_from_dataset(csv, target_column="target")
        re = prepare_dataset_from_provenance(csv, res["provenance"])
        self.assertEqual(re.csv_text, res["csv_text"])
        self.assertTrue(re.compatibility.prepared_matches)

    def test_csv_vacio_de_filas(self):
        with self.assertRaises(DatasetProjectError):
            prepare_dataset_from_provenance(
                "a,ciudad,target\n",
                generate_project_from_dataset(CSV_CAT, target_column="target")["provenance"],
            )


if __name__ == "__main__":
    unittest.main()
