# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C2 — modelo desde los datos.

`generate_project_from_dataset(csv_text, target_column)`: del esquema FINAL
(C1 + correcciones) + target elegido → prompt tipado canónico → generadores
EXISTENTES (`analyze_playground_request`, cero caminos paralelos) → mxai +
training_text + esquema S2 + CSV preparado + procedencia. Cubre los dos
ejemplos canónicos del contrato y el ciclo E2E hasta entrenar por el flujo
EXISTENTE (`_validate_training_csv`/`_submit_training_job`), sin ninguna
rama especial (invariante 4: "proyecto normal").
"""
from __future__ import annotations

import time
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None

from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
)


_SEMANA_SANTA_CSV = (
    "fecha,temperatura,tipo_dia,es_festivo,resultado\n"
    "2024-01-01,12.5,laboral,no,lluvia\n"
    "2024-01-02,15.0,laboral,no,sol\n"
    "2024-01-03,9.2,festivo,si,nublado\n"
    "2024-01-04,18.1,laboral,no,sol\n"
    "2024-01-05,11.0,laboral,no,lluvia\n"
    "2024-01-06,16.0,festivo,si,sol\n"
    "2024-01-07,10.0,laboral,no,nublado\n"
    "2024-01-08,17.5,laboral,no,sol\n"
    "2024-01-09,13.0,laboral,no,lluvia\n"
    "2024-01-10,14.5,festivo,si,nublado\n"
)


def _mar_csv(rows: int = 30) -> str:
    lines = ["fecha,altura_ola,periodo,temperatura_agua"]
    for i in range(rows):
        lines.append(
            f"2024-01-{(i % 28) + 1:02d},{2.0 + i * 0.1:.2f},{6 + i % 3},{15.0 + i * 0.05:.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Los dos ejemplos canónicos del contrato
# ---------------------------------------------------------------------------

class TestCanonicalExamples:
    def test_semana_santa_generates_classification_project(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        assert res["ok"]
        assert "ProbabilityMap[lluvia, nublado, sol]" in res["mxai"]
        assert "OUTPUT predicted_class" in res["mxai"]
        assert "TARGET predicted_class: Label[lluvia, nublado, sol]" in res["training_text"]
        # fecha (tipo date) nunca es feature; tipo_dia se expande a one-hot
        assert "fecha" not in res["mxai"]
        assert "tipo_dia__laboral" in res["mxai"] or "tipo_dia__festivo" in res["mxai"]

    def test_semana_santa_prepared_csv_matches_existing_validation_flow(self):
        """El flujo EXISTENTE (`_validate_training_csv`, sin ninguna rama
        especial) acepta el proyecto generado — invariante 4."""
        from matrixai.playground import _validate_training_csv
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        v = _validate_training_csv(
            res["mxai"], res["training_text"], res["csv_text"],
            field_ranges=res["field_ranges"],
        )
        assert v.get("ok"), v.get("error")

    def test_edited_category_vocabulary_controls_prompt_csv_and_provenance(self):
        res = generate_project_from_dataset(
            _SEMANA_SANTA_CSV, target_column="resultado",
            column_category_overrides={"tipo_dia": ["festivo", "laboral", "puente"]},
        )
        assert "Categorical[festivo, laboral, puente]" in res["provenance"]["synthesized_prompt"]
        assert "tipo_dia__puente" in res["csv_text"].splitlines()[0]
        assert res["provenance"]["column_category_overrides"]["tipo_dia"] == [
            "festivo", "laboral", "puente",
        ]

    def test_edited_category_vocabulary_cannot_omit_observed_values(self):
        with pytest.raises(DatasetProjectError, match="omite valores presentes"):
            generate_project_from_dataset(
                _SEMANA_SANTA_CSV, target_column="resultado",
                column_category_overrides={"tipo_dia": ["laboral", "puente"]},
            )

    def test_mar_time_series_regression_target_from_numeric_column(self):
        """Sin salida explícita (serie temporal pura): una columna numérica
        continua (no la fecha) sirve de target de regresión — el
        desplazamiento temporal real es C3, aún no construido; C2 cubre el
        caso tabular general de regresión."""
        res = generate_project_from_dataset(_mar_csv(), target_column="temperatura_agua")
        assert res["ok"]
        assert "OUTPUT predicted_value: Scalar" in res["mxai"]
        assert "fecha" not in res["mxai"]
        assert "altura_ola" in res["mxai"] and "periodo" in res["mxai"]


# ---------------------------------------------------------------------------
# Ciclo E2E real hasta entrenar (flujo existente, sin rama especial)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TORCH, reason="torch required for training")
class TestEndToEndTraining:
    def test_classification_project_trains_and_learns_with_separable_signal(self):
        import random
        from matrixai.playground import _submit_training_job, _get_job_status

        rng = random.Random(7)
        rows = ["fecha,temperatura,tipo_dia,es_festivo,resultado"]
        for i in range(120):
            temp = rng.uniform(0, 30)
            label = "sol" if temp > 20 else ("lluvia" if temp < 10 else "nublado")
            tipo = rng.choice(["laboral", "festivo"])
            festivo = "si" if tipo == "festivo" else "no"
            rows.append(f"2024-01-{(i % 28) + 1:02d},{temp:.2f},{tipo},{festivo},{label}")
        csv_text = "\n".join(rows)

        res = generate_project_from_dataset(csv_text, target_column="resultado")
        job = _submit_training_job(
            res["mxai"], res["training_text"], res["csv_text"],
            epochs_override=60, field_ranges=res["field_ranges"],
        )
        assert job.get("ok"), job
        st = {}
        for _ in range(400):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.2)
        assert st["status"] == "done", st
        # señal separable de verdad -> debe aprender bien, no quedarse en azar (33%)
        assert st["accuracy"] > 0.8, st.get("accuracy")

    def test_regression_project_trains_end_to_end(self):
        from matrixai.playground import _run_playground_training
        res = generate_project_from_dataset(_mar_csv(rows=40), target_column="temperatura_agua")
        from matrixai.playground import _normalize_csv_with_ranges
        normalized = _normalize_csv_with_ranges(res["csv_text"], res["field_ranges"])
        tr = _run_playground_training(res["mxai"], res["training_text"], normalized, epochs_override=5)
        assert tr.get("ok"), tr.get("error")
        assert tr["task_kind"] == "regression"


# ---------------------------------------------------------------------------
# Exclusión de columnas no-feature (S2-C4 / invariante del contrato)
# ---------------------------------------------------------------------------

class TestColumnExclusion:
    def test_identifier_column_never_becomes_a_feature(self):
        rows = ["id,temp,resultado"] + [
            f"P{1000 + i},{10 + i % 5},{'a' if i % 2 else 'b'}" for i in range(15)
        ]
        res = generate_project_from_dataset("\n".join(rows), target_column="resultado")
        assert "id" not in res["mxai"].lower().split("vector")[1].split("end")[0]
        assert "id" not in res["csv_text"].splitlines()[0].split(",")

    def test_date_column_never_becomes_a_feature(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        header = res["csv_text"].splitlines()[0].split(",")
        assert "fecha" not in header

    def test_target_column_excluded_from_its_own_features(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        header = res["csv_text"].splitlines()[0].split(",")
        assert "resultado" not in header  # renombrada a predicted_class
        assert "predicted_class" in header


# ---------------------------------------------------------------------------
# Errores accionables (invariante 7)
# ---------------------------------------------------------------------------

class TestActionableErrors:
    def test_unknown_target_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe en el CSV"):
            generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="no_existe")

    def test_identifier_as_target_raises(self):
        rows = ["id,temp,resultado"] + [
            f"P{1000 + i},{10 + i % 5},{'a' if i % 2 else 'b'}" for i in range(15)
        ]
        with pytest.raises(DatasetProjectError, match="no es un target válido"):
            generate_project_from_dataset("\n".join(rows), target_column="id")

    def test_date_as_target_raises(self):
        with pytest.raises(DatasetProjectError, match="no es un target válido"):
            generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="fecha")

    def test_colliding_labels_after_normalization_raise(self):
        rows = ["temp,y"] + [f"{10 + i % 5},{'Sí' if i % 2 else 'SI'}" for i in range(15)]
        with pytest.raises(DatasetProjectError, match="misma etiqueta tras"):
            generate_project_from_dataset("\n".join(rows), target_column="y")

    def test_constant_target_raises(self):
        # temp NO puede ser secuencial (15 valores únicos, >=10 filas caería
        # en identificador por unicidad, C1) — un feature con repeticiones
        # aísla de verdad el caso "target constante".
        rows = ["temp,y"] + [f"{i % 4},constante" for i in range(15)]
        with pytest.raises(DatasetProjectError, match="menos de 2 valores"):
            generate_project_from_dataset("\n".join(rows), target_column="y")

    def test_type_override_referencing_unknown_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe en el CSV"):
            generate_project_from_dataset(
                _SEMANA_SANTA_CSV, target_column="resultado",
                column_type_overrides={"no_existe": "categorical"},
            )


# ---------------------------------------------------------------------------
# Correcciones del usuario ganan sobre lo inferido (invariante 8)
# ---------------------------------------------------------------------------

class TestUserOverridesWin:
    def test_type_override_changes_generated_schema(self):
        """Una columna que C1 tipó 'integer' se fuerza a 'categorical' — el
        prompt sintetizado debe reflejar la corrección, no lo inferido."""
        rows = ["codigo,temp,resultado"] + [
            f"{i % 4},{10 + i % 5},{'a' if i % 2 else 'b'}" for i in range(20)
        ]
        csv_text = "\n".join(rows)
        default = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "codigo: Integer" in default["mxai"] or "codigo: Scalar" in default["mxai"]

        corrected = generate_project_from_dataset(
            csv_text, target_column="resultado",
            column_type_overrides={"codigo": "categorical"},
        )
        assert "codigo__" in corrected["mxai"]  # expandido a one-hot

    def test_range_override_changes_proposed_range(self):
        res = generate_project_from_dataset(
            _SEMANA_SANTA_CSV, target_column="resultado",
            column_range_overrides={"temperatura": (0.0, 50.0)},
        )
        assert res["field_ranges"]["temperatura"] == (0.0, 50.0)


# ---------------------------------------------------------------------------
# Reproducibilidad (invariante 5)
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_csv_and_target_produce_identical_project(self):
        res1 = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        res2 = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        assert res1["mxai"] == res2["mxai"]
        assert res1["training_text"] == res2["training_text"]
        assert res1["csv_text"] == res2["csv_text"]
        assert res1["provenance"]["raw_csv_sha256"] == res2["provenance"]["raw_csv_sha256"]
        assert res1["provenance"]["prepared_csv_sha256"] == res2["provenance"]["prepared_csv_sha256"]


# ---------------------------------------------------------------------------
# Procedencia (invariante 3)
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_has_required_fields(self):
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        prov = res["provenance"]
        assert prov["source"] == "user_upload"
        assert prov["target_column"] == "resultado"
        assert prov["task"] == "classification"
        assert len(prov["raw_csv_sha256"]) == 64
        assert len(prov["prepared_csv_sha256"]) == 64
        assert prov["raw_csv_sha256"] != prov["prepared_csv_sha256"]
        assert "schema_inferred" in prov and "schema_final" in prov
        assert "synthesized_prompt" in prov
        assert "operations" in prov and prov["operations"]
        assert "created_at" in prov

    def test_provenance_records_user_overrides(self):
        res = generate_project_from_dataset(
            _SEMANA_SANTA_CSV, target_column="resultado",
            column_range_overrides={"temperatura": (0.0, 50.0)},
        )
        assert res["provenance"]["column_range_overrides"] == {"temperatura": [0.0, 50.0]}

    def test_raw_hash_is_of_the_original_bytes_not_normalized(self):
        """El hash CRUDO es del CSV tal cual llegó (antes de BOM/delimitador
        de C1) — reproducible incluso si el usuario re-sube el MISMO
        fichero con distinto salto de línea no debería mentir sobre qué
        bytes originales hasheó."""
        import hashlib
        res = generate_project_from_dataset(_SEMANA_SANTA_CSV, target_column="resultado")
        expected = hashlib.sha256(_SEMANA_SANTA_CSV.encode("utf-8")).hexdigest()
        assert res["provenance"]["raw_csv_sha256"] == expected


from matrixai.training.dataset_project import (  # noqa: E402
    DatasetProjectError, _normalize_labels, _slug)


class TestEtiquetasConSigno:
    """101-C3, cableado — `-1` y `1` se declaraban «la misma etiqueta».

    `_slug` metía el signo menos en la clase de «cualquier símbolo», lo
    convertía en `_`, y `.strip("_")` lo remataba: `-1` y `1` daban los dos
    `1`, así que `_normalize_labels` levantaba `DatasetProjectError` diciendo
    que no se pueden distinguir y pidiendo «unificar el texto de esas filas».

    Costó 15 intentos y un dataset entero, medido: en la pasada exploratoria
    del 101-C3 (2026-09-07, diagnosticado el 09-12), `PhishingWebsites` tiene
    la columna objetivo `Result` con valores `-1` y `1`, que es la
    codificación más común que existe para un problema binario. La red densa
    perdió los 15 intentos, o sea que en ese dataset **no compitió**, y
    LightGBM figuraba como mejor sin rival. La regla de cierre del 101-C1
    leyó esa victoria como buena.

    `-1` y `1` no son un dato ambiguo que alguien deba unificar: son dos
    valores perfectamente distintos que la normalización estaba fundiendo. Un
    mensaje correcto sobre un diagnóstico equivocado sigue siendo un fallo.
    """

    def test_menos_uno_y_uno_YA_NO_son_la_misma_etiqueta(self):
        etiquetas, mapa = _normalize_labels(["-1", "1"], "Result")
        assert len(set(etiquetas)) == 2, etiquetas
        assert mapa["-1"] != mapa["1"]

    def test_el_signo_se_ve_en_la_etiqueta_no_solo_se_evita_el_choque(self):
        """Evitar la colisión con un sufijo `_2` cualquiera también pasaría el
        test de arriba, y dejaría una etiqueta que no dice nada. Quien la lee
        tiene que reconocer el valor del que salió."""
        assert _slug("-1") == "neg_1"
        assert _slug("1") == "1"

    def test_un_guion_EN_MEDIO_no_es_un_signo(self):
        """La otra mitad. Sin ella, la reparación la pasaría una versión que
        trata cualquier guion como negativo y convierte `alto-riesgo` en
        `neg_alto_riesgo`, que sería peor que el fallo original."""
        assert _slug("alto-riesgo") == "alto_riesgo"
        assert _slug("post-venta") == "post_venta"

    def test_los_decimales_negativos_tenian_el_MISMO_problema(self):
        assert _slug("-0.5") != _slug("0.5")
        etiquetas, _ = _normalize_labels(["-0.5", "0.5"], "delta")
        assert len(set(etiquetas)) == 2

    def test_una_colision_DE_VERDAD_se_sigue_detectando(self):
        """El detector tiene que seguir haciendo su trabajo: `Sí` y `SI` sí
        son el mismo texto tras normalizar, y ahí el error es correcto."""
        with pytest.raises(DatasetProjectError):
            _normalize_labels(["Sí", "SI"], "respuesta")

    def test_un_valor_sin_nada_alfanumerico_sigue_siendo_error(self):
        with pytest.raises(DatasetProjectError):
            _normalize_labels(["###", "otro"], "col")


class TestEtiquetasDeAdultQueChocaban:
    """101-C5, preparando la pasada amplia — `<=50K` y `>50K` daban las dos
    `class_50k`.

    Es el mismo fallo que el del signo menos, un escalón más arriba: `_slug`
    nombraba el menos pero seguía metiendo `<`, `>` y `=` en la clase de
    «cualquier símbolo», así que los borraba — y en `adult` (el dataset de
    renta de OpenML) esos signos son **lo único** que distingue las dos
    clases. `_normalize_labels` las declaraba indistinguibles y la densa
    perdía el dataset ENTERO, los 15 pliegues, antes de entrenar nada.

    Medido el 2026-09-14 sobre los 30 datasets de clasificación del protocolo
    de Fase 0 leyendo su ARFF real: `adult` es el ÚNICO con este choque.
    """

    def test_las_dos_clases_de_adult_YA_NO_son_la_misma_etiqueta(self):
        etiquetas, mapa = _normalize_labels(["<=50K", ">50K"], "class")
        assert len(set(etiquetas)) == 2, etiquetas
        assert mapa["<=50K"] != mapa[">50K"]

    def test_el_signo_se_VE_en_la_etiqueta_no_solo_se_evita_el_choque(self):
        """La otra mitad, igual que en `TestEtiquetasConSigno`: un sufijo `_2`
        también evitaría el choque y dejaría dos etiquetas que no dicen de qué
        valor salieron. Quien lea `class_le_50k` tiene que reconocer `<=50K`."""
        assert _slug("<=50K") == "le_50k"
        assert _slug(">50K") == "gt_50k"

    def test_ningun_par_que_se_diferencie_solo_en_un_RELACIONAL_choca(self):
        """La propiedad, no el caso. Cada par de esta lista se diferencia
        ÚNICAMENTE en el signo relacional, que es justo lo que se borraba."""
        pares = [("<=50K", ">50K"), ("<18", ">18"), (">=65", "<=65"),
                 ("<5", ">5"), ("a=b", "a!=b"), ("x<y", "x>y")]
        for uno, otro in pares:
            assert _slug(uno) != _slug(otro), (uno, otro, _slug(uno))
            etiquetas, mapa = _normalize_labels([uno, otro], "col")
            assert len(set(etiquetas)) == 2, (uno, otro, etiquetas)
            assert mapa[uno] != mapa[otro], (uno, otro)

    def test_el_orden_de_dos_caracteres_importa(self):
        """Con `<` mirado antes que `<=`, el `<=` se parte y sale `lt_eq_50k`.
        No sería un choque, pero sí una etiqueta distinta de la declarada — y
        esta prueba es lo único que sostiene el orden de la tabla."""
        assert _slug("<=50K") == "le_50k"
        assert _slug(">=50K") == "ge_50k"
        assert _slug("<>a") == "ne_a"

    def test_un_separador_NO_es_un_signo(self):
        """La mitad que evita el sesgo contrario: si esto se hubiera arreglado
        nombrando toda la puntuación, un espacio o un punto también tendrían
        nombre y `alto-riesgo` o `0.5` cambiarían de etiqueta — moviendo
        números ya medidos. Solo los relacionales se nombran."""
        assert _slug("alto-riesgo") == "alto_riesgo"
        assert _slug("0.5") == "0_5"
        assert _slug("dos palabras") == "dos_palabras"

    def test_las_etiquetas_YA_MEDIDAS_no_se_mueven(self):
        """101-C3 está commiteado y su evidencia cita un digest de código. Los
        valores que pasaron por aquí en aquella pasada (`PhishingWebsites` con
        `-1`/`1`, `Internet-Advertisements` con `0`/`1`) tienen que salir con
        LA MISMA etiqueta que entonces, o el arreglo de hoy invalida una
        medición de ayer sin decirlo."""
        assert _slug("-1") == "neg_1"
        assert _slug("1") == "1"
        assert _slug("0") == "0"
        assert _slug("-0.5") == "neg_0_5"
        assert _normalize_labels(["-1", "1"], "Result")[1] == {
            "-1": "class_neg_1", "1": "class_1"}
        assert _normalize_labels(["0", "1"], "x")[1] == {
            "0": "class_0", "1": "class_1"}

    def test_un_par_que_choca_por_un_signo_NO_relacional_sigue_avisando(self):
        """La línea del docstring de `_slug` que explica por qué NO se nombra
        el resto de la puntuación, con su prueba. `a+` y `a*` siguen dando la
        misma etiqueta; lo que NO puede pasar es que se fundan en silencio."""
        assert _slug("a+") == _slug("a*")
        with pytest.raises(DatasetProjectError) as exc:
            _normalize_labels(["a+", "a*"], "col")
        assert "la misma etiqueta tras normalizar" in str(exc.value)


from matrixai.training.dataset_project import (  # noqa: E402
    _normalize_feature_names, generate_project_from_dataset)
from matrixai.playground import _validate_training_csv  # noqa: E402


class TestColumnasQueEmpiezanPorDigito:
    """101-C5, preparando la pasada amplia — `1stFlrSF` tiraba el dataset.

    Es el MISMO fallo que el de las etiquetas, en el otro lado del CSV: un
    identificador no puede empezar por número, así que `_identifier('1stFlrSF')`
    sale vacío y `_normalize_feature_names` levantaba «no tiene un nombre de
    campo válido tras normalizar (solo símbolos/espacios)» — sobre una columna
    con ocho letras dentro. Un mensaje correcto sobre un diagnóstico
    equivocado.

    Medido el 2026-09-14 sobre los 40 datasets del protocolo de Fase 0: el
    único afectado es `house_prices_nominal` (`1stFlrSF`, `2ndFlrSF`,
    `3SsnPorch`), y perdía **los 15 pliegues** con los 7 motores, antes de
    entrenar nada. `_normalize_labels` ya rescataba con `class_` el valor que
    empieza por dígito; aquí se rescata con `campo_`.
    """

    def test_una_columna_que_empieza_por_digito_YA_NO_tira_el_dataset(self):
        mapa = _normalize_feature_names(["1stFlrSF", "LotArea"], "predicted_value")
        assert mapa["1stFlrSF"] == "campo_1stflrsf"

    def test_el_nombre_ORIGINAL_se_reconoce_en_el_nuevo(self):
        """Un `campo_1`, un `col_0` o un hash también evitarían el choque y
        dejarían un nombre que no dice de qué columna salió."""
        mapa = _normalize_feature_names(
            ["1stFlrSF", "2ndFlrSF", "3SsnPorch"], "predicted_value")
        assert mapa == {"1stFlrSF": "campo_1stflrsf",
                        "2ndFlrSF": "campo_2ndflrsf",
                        "3SsnPorch": "campo_3ssnporch"}

    def test_una_columna_NORMAL_no_cambia_de_nombre(self):
        """La mitad que evita el sesgo contrario: si el rescate se aplicara
        siempre, TODAS las columnas cambiarían de nombre y con ellas el CSV
        preparado de todos los datasets que hoy funcionan."""
        mapa = _normalize_feature_names(
            ["LotArea", "alto-riesgo", "Año de alta"], "predicted_value")
        assert mapa == {"LotArea": "lotarea", "alto-riesgo": "alto_riesgo",
                        "Año de alta": "ano_de_alta"}

    def test_una_columna_SIN_NADA_alfanumerico_sigue_siendo_error(self):
        """El rescate no puede tragárselo todo: `###` sí es lo que el mensaje
        decía, y ahí el error es correcto."""
        with pytest.raises(DatasetProjectError):
            _normalize_feature_names(["###", "otra"], "predicted_value")

    def test_el_rescate_no_puede_colarse_encima_de_otra_columna(self):
        with pytest.raises(DatasetProjectError) as exc:
            _normalize_feature_names(["1stFlrSF", "campo_1stFlrSF"], "predicted_value")
        assert "el mismo nombre de campo" in str(exc.value)

    def test_el_proyecto_se_genera_Y_valida_con_esa_columna(self):
        """Probar la función no es probar el producto: el nombre rescatado
        tiene que sobrevivir al prompt, al modelo generado y a la validación
        del CSV contra ese modelo."""
        import csv as _csv
        import io as _io
        alturas = ["150.5", "163.2", "171.9", "158.4", "180.1", "149.7",
                   "167.3", "175.8", "152.6", "169.0", "177.4", "161.1"]
        out = _io.StringIO()
        w = _csv.DictWriter(out, fieldnames=["1stFlrSF", "altura", "target"])
        w.writeheader()
        for i in range(12):
            w.writerow({"1stFlrSF": str(900 + i * 37), "altura": alturas[i],
                        "target": ["si", "no"][i % 2]})
        res = generate_project_from_dataset(out.getvalue(), target_column="target")
        assert res["ok"]
        assert "campo_1stflrsf" in res["csv_text"].splitlines()[0].split(",")
        assert "campo_1stflrsf" in res["mxai"]
        v = _validate_training_csv(res["mxai"], res["training_text"], res["csv_text"],
                                   field_ranges=res.get("field_ranges"))
        assert v.get("ok"), v.get("errors") or v.get("error")
