# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 62 C2 — una feature constante se declara, no se cuela.

Caso real que lo motivó (2026-07-25): el dataset de lluvia con UNA sola ciudad
deja `lat`/`lon`/`elevation` con un único valor en las 31.611 filas. El análisis
las aceptaba como features y `_propose_margin` les inventaba un rango de ±1
(`lat` 43.46 → [42.46, 44.46]); con ese rango, el panel de inferencia pintaba
tres sliders que no significan nada y que solo sirven para sacar al modelo de la
distribución en la que se entrenó.

Ojo al INVARIANTE 3 del contrato: esta exclusión es una regla de GENERACIÓN.
Re-preparar un modelo que ya existe debe reproducir el esquema con el que nació
—constantes incluidas—, y para eso está `keep_constant_columns`. El guard de esa
reproducción vive aquí ya, aunque la API de re-preparación llegue en C3.
"""
from __future__ import annotations

import unittest

from matrixai.training.dataset_analysis import analyze_dataset_csv
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
)

# `a` varía, `lat` es constante — el caso de la ciudad única, en pequeño.
CSV = (
    "a,lat,target\n"
    "0.1,43.46,0\n0.5,43.46,1\n0.9,43.46,0\n"
    "0.2,43.46,1\n0.7,43.46,0\n0.3,43.46,1\n"
)


class TestDeteccion(unittest.TestCase):
    def test_marca_constante_numerica_booleana_y_categorica(self):
        csv = (
            "num,cat,boo,varia,target\n"
            "43.46,SOLO,si,1,0\n43.46,SOLO,si,2,1\n43.46,SOLO,si,3,0\n"
        )
        cols = analyze_dataset_csv(csv)["columns"]
        for name in ("num", "cat", "boo"):
            self.assertTrue(cols[name].get("constant"), f"{name} debería ser constante")
            self.assertEqual(cols[name]["cardinality"], 1)
        self.assertIn("constant_value", cols["num"])
        self.assertFalse(cols["varia"].get("constant"))

    def test_una_columna_que_varia_no_se_marca(self):
        cols = analyze_dataset_csv(CSV)["columns"]
        self.assertFalse(cols["a"].get("constant"))
        self.assertTrue(cols["lat"].get("constant"))

    def test_los_nulos_no_cuentan_como_segundo_valor(self):
        """"43.46" y vacío NO son dos valores distintos: la columna sigue
        siendo constante y debe excluirse igual."""
        csv = "a,lat,target\n0.1,43.46,0\n0.5,,1\n0.9,43.46,0\n0.2,,1\n"
        cols = analyze_dataset_csv(csv)["columns"]
        self.assertTrue(cols["lat"].get("constant"))


class TestExclusionPorDefecto(unittest.TestCase):
    def test_la_constante_no_entra_en_el_vector(self):
        res = generate_project_from_dataset(CSV, target_column="target")
        self.assertNotIn("lat", res["mxai"])
        self.assertIn("a", res["field_ranges"])
        self.assertNotIn("lat", res["field_ranges"])

    def test_no_se_inventa_un_rango_de_mas_menos_uno(self):
        """El daño concreto que este corte elimina."""
        res = generate_project_from_dataset(CSV, target_column="target")
        self.assertNotIn("lat", res["field_ranges"])
        for lo, hi in res["field_ranges"].values():
            self.assertLess(lo, hi, "ningún rango puede ser degenerado")

    def test_la_procedencia_dice_por_que_falta(self):
        prov = generate_project_from_dataset(CSV, target_column="target")["provenance"]
        self.assertIn("lat", prov["excluded_columns"])
        motivo = prov["excluded_column_reasons"]["lat"]
        self.assertEqual(motivo["reason"], "constant_feature")
        self.assertTrue(motivo["automatic"])
        self.assertEqual(prov["kept_constant_columns"], [])

    def test_las_exclusiones_por_tipo_conservan_su_motivo(self):
        """Una fecha se sigue excluyendo por su tipo, no por constante — el
        motivo estructurado tiene que distinguirlos."""
        csv = (
            "fecha,a,target\n"
            "2024-01-01,0.1,0\n2024-01-02,0.5,1\n2024-01-03,0.9,0\n"
            "2024-01-04,0.2,1\n2024-01-05,0.7,0\n2024-01-06,0.3,1\n"
        )
        prov = generate_project_from_dataset(csv, target_column="target")["provenance"]
        self.assertIn("fecha", prov["excluded_column_reasons"])
        self.assertTrue(
            prov["excluded_column_reasons"]["fecha"]["reason"].startswith("unusable_type:")
        )


class TestConservarlaEsExplicito(unittest.TestCase):
    def test_conservar_sin_rango_declarado_falla_y_dice_como(self):
        with self.assertRaises(DatasetProjectError) as ctx:
            generate_project_from_dataset(
                CSV, target_column="target", keep_constant_columns=["lat"])
        msg = str(ctx.exception)
        self.assertIn("lat", msg)
        self.assertIn("único valor", msg)
        self.assertIn("rango", msg)

    def test_conservar_con_rango_de_dominio_real(self):
        res = generate_project_from_dataset(
            CSV, target_column="target",
            keep_constant_columns=["lat"],
            column_range_overrides={"lat": (-90.0, 90.0)},
        )
        self.assertIn("lat", res["field_ranges"])
        self.assertEqual(tuple(res["field_ranges"]["lat"]), (-90.0, 90.0))
        prov = res["provenance"]
        self.assertEqual(prov["kept_constant_columns"], ["lat"])
        self.assertNotIn("lat", prov["excluded_columns"])
        # AUDITORÍA [ALTO]: comprobar el EFECTO, no solo la procedencia.
        self.assertIn("lat", res["mxai"])
        self.assertIn("lat", res["csv_text"].split("\n")[0])

    def test_un_rango_degenerado_se_rechaza(self):
        with self.assertRaises(DatasetProjectError):
            generate_project_from_dataset(
                CSV, target_column="target",
                keep_constant_columns=["lat"],
                column_range_overrides={"lat": (43.46, 43.46)},
            )

    CSV_CAT = (
        "a,cat,target\n"
        "0.1,SOLO,0\n0.5,SOLO,1\n0.9,SOLO,0\n"
        "0.2,SOLO,1\n0.7,SOLO,0\n0.3,SOLO,1\n"
    )

    def test_conservar_categorica_constante_exige_vocabulario_completo(self):
        """AUDITORÍA [ALTO]: antes esto decía que la conservaba y luego la
        columna DESAPARECÍA del .mxai y del CSV preparado — one-hot con un solo
        valor no existe (`len(values) < 2: continue`). El test viejo solo miraba
        la procedencia, por eso no lo cazó. El equivalente al rango de dominio
        de una numérica es aquí declarar el vocabulario completo."""
        with self.assertRaises(DatasetProjectError) as ctx:
            generate_project_from_dataset(
                self.CSV_CAT, target_column="target", keep_constant_columns=["cat"])
        msg = str(ctx.exception)
        self.assertIn("cat", msg)
        self.assertIn("vocabulario", msg)

    def test_con_vocabulario_la_categorica_entra_DE_VERDAD(self):
        res = generate_project_from_dataset(
            self.CSV_CAT, target_column="target",
            keep_constant_columns=["cat"],
            column_category_overrides={"cat": ["SOLO", "OTRO"]},
        )
        # Lo que fallaba: comprobar el EFECTO, no solo la procedencia.
        self.assertEqual(res["provenance"]["kept_constant_columns"], ["cat"])
        self.assertIn("cat__solo", res["mxai"])
        self.assertIn("cat__solo", res["csv_text"].split("\n")[0])

    def test_una_booleana_constante_conservada_SI_aparece(self):
        csv = (
            "a,boo,target\n"
            "0.1,si,0\n0.5,si,1\n0.9,si,0\n"
            "0.2,si,1\n0.7,si,0\n0.3,si,1\n"
        )
        res = generate_project_from_dataset(
            csv, target_column="target", keep_constant_columns=["boo"])
        self.assertIn("boo", res["mxai"])
        self.assertIn("boo", res["csv_text"].split("\n")[0])


class TestSinFeaturesInformativas(unittest.TestCase):
    def test_aborta_nombrando_las_constantes_y_como_conservarlas(self):
        csv = ("c1,c2,target\n"
               "5,SOLO,0\n5,SOLO,1\n5,SOLO,0\n5,SOLO,1\n5,SOLO,0\n5,SOLO,1\n")
        with self.assertRaises(DatasetProjectError) as ctx:
            generate_project_from_dataset(csv, target_column="target")
        msg = str(ctx.exception)
        self.assertIn("c1", msg)
        self.assertIn("c2", msg)
        self.assertIn("único valor", msg)
        # El mensaje genérico mandaría a buscar identificadores o fechas que
        # aquí no existen.
        self.assertNotIn("identificadores, fechas", msg)


class TestInvariante3GenerarNoEsRePreparar(unittest.TestCase):
    """Guard del invariante 3: un modelo nacido CON la constante debe poder
    reproducir su esquema. Sin esta palanca, C2 dejaría sin columnas a todos
    los modelos guardados antes de este corte."""

    def test_se_puede_reproducir_un_esquema_que_incluia_la_constante(self):
        antiguo = generate_project_from_dataset(
            CSV, target_column="target",
            keep_constant_columns=["lat"],
            column_range_overrides={"lat": (-90.0, 90.0)},
        )
        # Re-generar declarando lo que la procedencia registró reproduce el
        # MISMO conjunto de features, constante incluida.
        prov = antiguo["provenance"]
        repetido = generate_project_from_dataset(
            CSV, target_column="target",
            keep_constant_columns=prov["kept_constant_columns"],
            column_range_overrides={"lat": (-90.0, 90.0)},
        )
        self.assertEqual(
            sorted(antiguo["field_ranges"]), sorted(repetido["field_ranges"]))
        self.assertIn("lat", repetido["field_ranges"])
        self.assertEqual(repetido["csv_text"], antiguo["csv_text"])


if __name__ == "__main__":
    unittest.main()
