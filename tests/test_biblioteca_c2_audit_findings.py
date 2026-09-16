# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C2 — regresión de la auditoría externa
(2026-07-17): 4 hallazgos ALTA + 2 MEDIA, todos reproducidos y corregidos.
Cada test reproduce el caso EXACTO del hallazgo y verifica el fix contra el
flujo de validación real (`_validate_training_csv`), no solo contra la
forma del dict devuelto — un CSV "aparentemente correcto" que la
validación real rechaza es el bug que estos hallazgos describen.
"""
from __future__ import annotations

import pytest

from matrixai.playground import _validate_training_csv
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
    prepare_dataset_from_provenance,
)


def _assert_prepared_csv_validates(res: dict) -> None:
    v = _validate_training_csv(
        res["mxai"], res["training_text"], res["csv_text"],
        field_ranges=res.get("field_ranges"),
    )
    assert v.get("ok"), v.get("errors") or v.get("error")


# ---------------------------------------------------------------------------
# ALTA-1 — cabeceras no normalizadas igual en el modelo y en el CSV
# ---------------------------------------------------------------------------

class TestAltaHeaderNormalization:
    def test_column_with_space_matches_between_mxai_and_csv(self):
        csv_text = (
            "customer age,region,resultado\n"
            "25,north,si\n30,south,no\n40,north,si\n"
            "50,south,no\n22,north,si\n60,south,no\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "customer_age" in res["mxai"]
        assert "customer age" not in res["csv_text"].splitlines()[0]
        assert "customer_age" in res["csv_text"].splitlines()[0]
        _assert_prepared_csv_validates(res)

    def test_categorical_column_with_space_matches(self):
        csv_text = (
            "tipo de dia,x,resultado\n"
            "laboral,1,si\nfestivo,2,no\nlaboral,3,si\n"
            "festivo,4,no\nlaboral,5,si\nfestivo,6,no\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        _assert_prepared_csv_validates(res)

    def test_colliding_column_names_after_normalization_raise(self):
        csv_text = (
            "cliente-edad,cliente_edad,resultado\n"
            "1,2,si\n3,4,no\n5,6,si\n7,8,no\n9,10,si\n11,12,no\n"
        )
        with pytest.raises(DatasetProjectError, match="mismo.*nombre de campo"):
            generate_project_from_dataset(csv_text, target_column="resultado")


# ---------------------------------------------------------------------------
# ALTA-2 — categorías sin serialización segura en el prompt
#
# DECISIÓN REVISADA (2026-09-16, Roberto): el hallazgo original rechazaba
# CUALQUIER valor con ',', ']' o salto de línea — correcto en su razón (el
# prompt tipado no tiene escape), pero perdía datasets reales enteros
# (`okcupid-stem` de OpenML: 4 de 18 columnas categóricas, hasta 7.015 de
# 7.018 niveles en una sola, valores naturales como "doesn't have kids, but
# might want them"). Ahora se RENOMBRA en vez de rechazar — mismo valor en el
# `Categorical[...]` del prompt y en el CSV preparado, con el mapa congelado
# en la receta para que la predicción traduzca la fila igual. La guarda
# original sigue viva para lo que el renombrado no pueda resolver (una
# colisión irresoluble): ver `TestAlta2ColisionesTrasRenombrar`.
# ---------------------------------------------------------------------------

class TestAltaUnsafeCategoryValues:
    def test_comma_in_categorical_value_is_renamed_not_rejected(self):
        csv_text = (
            'cat,x,resultado\n'
            '"red,blue",1,si\ngreen,2,no\n"red,blue",3,si\n'
            'green,4,no\n"red,blue",5,si\ngreen,6,no\n'
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        _assert_prepared_csv_validates(res)
        assert "red,blue" not in res["training_text"]
        renames = res["provenance"]["preparation_spec"]["category_value_renames"]
        assert renames["cat"]["red,blue"]
        assert "," not in renames["cat"]["red,blue"]
        # El valor SEGURO ("green") no se toca.
        assert "green" in res["provenance"]["preparation_spec"]["category_vocabularies"]["cat"]

    def test_closing_bracket_in_categorical_value_is_renamed_not_rejected(self):
        csv_text = (
            "cat,x,resultado\n"
            "a]b,1,si\nverde,2,no\na]b,3,si\nverde,4,no\na]b,5,si\nverde,6,no\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        _assert_prepared_csv_validates(res)
        renames = res["provenance"]["preparation_spec"]["category_value_renames"]
        assert renames["cat"]["a]b"]
        assert "]" not in renames["cat"]["a]b"]
        assert "verde" in res["provenance"]["preparation_spec"]["category_vocabularies"]["cat"]

    def test_la_prediccion_relee_el_renombrado_con_el_valor_crudo(self):
        """La vuelta: una fila NUEVA que trae el valor crudo (con su coma) se
        predice bien — `prepare_dataset_from_provenance` es el camino real de
        predicción del motor denso (ver `tests/test_c101_c5_la_prediccion_
        relee_la_receta.py`)."""
        csv_text = (
            'cat,x,resultado\n'
            '"red,blue",1,si\ngreen,2,no\n"red,blue",3,si\n'
            'green,4,no\n"red,blue",5,si\ngreen,6,no\n'
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        nuevo_csv = 'cat,x,resultado\n"red,blue",9,si\ngreen,10,no\n'
        re_prep = prepare_dataset_from_provenance(nuevo_csv, res["provenance"])
        assert not re_prep.compatibility.errors
        filas = re_prep.csv_text.splitlines()
        header = filas[0].split(",")
        idx_red_blue = header.index("cat__red_blue")
        # La fila con el valor crudo "red,blue" tiene que activar la MISMA
        # columna one-hot que activó al entrenar.
        assert filas[1].split(",")[idx_red_blue] == "1"

    def test_dos_valores_que_colisionan_tras_renombrar_siguen_dando_error(self):
        """Un valor inseguro y uno ya seguro que renombran al mismo token no
        se pueden distinguir — sigue siendo un error accionable, nunca un
        colapso silencioso (mismo principio que la colisión de
        `_normalize_labels`, y el mensaje tampoco culpa al CSV)."""
        csv_text = (
            'cat,x,resultado\n'
            '"red,blue",1,si\nred_blue,2,no\n"red,blue",3,si\n'
            'red_blue,4,no\n"red,blue",5,si\nred_blue,6,no\n'
        )
        with pytest.raises(DatasetProjectError) as exc_info:
            generate_project_from_dataset(csv_text, target_column="resultado")
        msg = str(exc_info.value)
        assert "red,blue" in msg and "red_blue" in msg
        assert "limitación nuestra" in msg

    def test_valores_ya_seguros_no_cambian_ni_un_byte(self):
        """No-regresión: una columna categórica SIN valores inseguros tiene
        que producir EXACTAMENTE el mismo CSV/prompt que antes de este
        arreglo — renombrar algo que ya era seguro rompería toda receta
        existente que lo usa tal cual."""
        csv_text = (
            "tipo,x,resultado\n"
            "Alto Riesgo,1,si\nBajo riesgo,2,no\nAlto Riesgo,3,si\n"
            "Bajo riesgo,4,no\nAlto Riesgo,5,si\nBajo riesgo,6,no\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert res["provenance"]["preparation_spec"]["category_value_renames"] == {}
        assert set(res["provenance"]["preparation_spec"]["category_vocabularies"]["tipo"]) == {
            "Alto Riesgo", "Bajo riesgo"}
        # El prompt sintetizado es donde el valor CRUDO viaja tal cual — la
        # cabecera one-hot del CSV ya tenía su PROPIA normalización
        # (`_build_group_names`/`_sanitize_value`, ajena a este arreglo) y no
        # es lo que este test tiene que vigilar.
        assert "Categorical[Alto Riesgo, Bajo riesgo]" in res["provenance"]["synthesized_prompt"]

    def test_column_category_overrides_con_un_valor_con_coma_ya_funciona(self):
        csv_text = (
            "cat,x,resultado\n"
            "red,1,si\ngreen,2,no\nred,3,si\ngreen,4,no\nred,5,si\ngreen,6,no\n"
        )
        res = generate_project_from_dataset(
            csv_text, target_column="resultado",
            column_category_overrides={"cat": ["red", "green", "blue,violet"]},
        )
        _assert_prepared_csv_validates(res)
        spec = res["provenance"]["preparation_spec"]
        assert spec["category_value_renames"]["cat"]["blue,violet"]
        assert "," not in spec["category_value_renames"]["cat"]["blue,violet"]
        # `column_category_overrides` en la procedencia sigue siendo lo que
        # el usuario declaró, sin renombrar — es SU entrada, no lo usado.
        assert res["provenance"]["column_category_overrides"]["cat"] == [
            "red", "green", "blue,violet"]


# ---------------------------------------------------------------------------
# ALTA-3 — categóricas de alta cardinalidad (> _ONEHOT_MAX) rompían el CSV
# ---------------------------------------------------------------------------

class TestAltaHighCardinalityCategorical:
    def _csv(self, n_cats: int = 20, rows: int = 60) -> str:
        cats = [f"c{i}" for i in range(n_cats)]
        lines = ["cat,x,resultado"]
        for i in range(rows):
            lines.append(f"{cats[i % n_cats]},{i % 7},{'si' if i % 2 == 0 else 'no'}")
        return "\n".join(lines) + "\n"

    def test_routes_to_embedding_not_onehot(self):
        res = generate_project_from_dataset(self._csv(), target_column="resultado")
        assert "FROM COLUMNS [cat, x]" in res["training_text"]
        header = res["csv_text"].splitlines()[0].split(",")
        assert "cat" in header
        assert not any(h.startswith("cat__") for h in header)

    def test_embedding_csv_validates_against_generated_model(self):
        res = generate_project_from_dataset(self._csv(), target_column="resultado")
        _assert_prepared_csv_validates(res)

    def test_embedding_column_values_are_vocab_indices(self):
        res = generate_project_from_dataset(self._csv(), target_column="resultado")
        header = res["csv_text"].splitlines()[0].split(",")
        cat_idx = header.index("cat")
        for line in res["csv_text"].splitlines()[1:]:
            value = line.split(",")[cat_idx]
            assert value.isdigit()

    def test_low_cardinality_still_uses_onehot(self):
        """Control: por debajo del umbral, el camino one-hot de siempre sigue activo."""
        res = generate_project_from_dataset(self._csv(n_cats=3, rows=30), target_column="resultado")
        header = res["csv_text"].splitlines()[0].split(",")
        assert any(h.startswith("cat__") for h in header)
        _assert_prepared_csv_validates(res)


# ---------------------------------------------------------------------------
# ALTA-4 — target booleano canónico 0/1 rechazado
# ---------------------------------------------------------------------------

class TestAltaNumericTargetLabels:
    def test_binary_0_1_target_generates_class_labels(self):
        csv_text = "x,y,resultado\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n"
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "ProbabilityMap[class_0, class_1]" in res["mxai"]
        _assert_prepared_csv_validates(res)

    def test_binary_0_1_prepared_csv_uses_class_labels(self):
        csv_text = "x,y,resultado\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n"
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        rows = [line.split(",")[-1] for line in res["csv_text"].splitlines()[1:]]
        assert set(rows) == {"class_0", "class_1"}

    def test_digit_leading_label_gets_class_prefix(self):
        csv_text = "x,y,resultado\n1,2,24h\n3,4,48h\n5,6,24h\n7,8,48h\n9,10,24h\n11,12,48h\n"
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "class_24h" in res["mxai"]
        assert "class_48h" in res["mxai"]

    def test_purely_symbolic_label_still_raises(self):
        csv_text = "x,y,resultado\n1,2,###\n3,4,@@@\n5,6,###\n7,8,@@@\n9,10,###\n11,12,@@@\n"
        with pytest.raises(DatasetProjectError, match="vacío tras normalizar"):
            generate_project_from_dataset(csv_text, target_column="resultado")


# ---------------------------------------------------------------------------
# MEDIA — validación del CSV preparado antes de responder ok
# ---------------------------------------------------------------------------

class TestMediaPreparedCsvIsValidated:
    def test_generate_raises_actionable_error_if_prepared_csv_would_fail_validation(self, monkeypatch):
        """El nuevo paso de validación (contrato §C2) debe convertir CUALQUIER
        desalineamiento futuro modelo<->CSV en un DatasetProjectError, nunca
        en un `ok: True` silenciosamente roto."""
        def _broken_validate(*args, **kwargs):
            return {"ok": False, "errors": ["forzado por el test"]}

        import matrixai.playground as pg
        monkeypatch.setattr(pg, "_validate_training_csv", _broken_validate)

        csv_text = "x,y,resultado\n1,2,si\n3,4,no\n5,6,si\n7,8,no\n9,10,si\n11,12,no\n"
        with pytest.raises(DatasetProjectError, match="no pasa la validación"):
            generate_project_from_dataset(csv_text, target_column="resultado")


# ---------------------------------------------------------------------------
# MEDIA — procedencia: operations solo por FEATURES + seed + excluidas + filas caídas
# ---------------------------------------------------------------------------

class TestMediaProvenanceCompleteness:
    def test_categorical_target_alone_does_not_flag_onehot_operation(self):
        """El target categórico (caso normal de clasificación) NO debe
        aparecer como 'expand_categoricals_onehot' si ninguna FEATURE se
        expandió — antes se miraba el esquema completo, incluido el target."""
        csv_text = (
            "x,y,resultado\n"
            "1,2,lluvia\n3,4,sol\n5,6,nublado\n7,8,lluvia\n9,10,sol\n11,12,nublado\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "expand_categoricals_onehot" not in res["provenance"]["operations"]

    def test_categorical_feature_does_flag_onehot_operation(self):
        csv_text = (
            "tipo,x,resultado\n"
            "a,1,si\nb,2,no\na,3,si\nb,4,no\na,5,si\nb,6,no\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert "expand_categoricals_onehot" in res["provenance"]["operations"]

    def test_seed_recorded_matches_training_text(self):
        csv_text = "x,y,resultado\n1,2,si\n3,4,no\n5,6,si\n7,8,no\n9,10,si\n11,12,no\n"
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert res["provenance"]["seed"] is not None
        assert f"seed={res['provenance']['seed']}" in res["training_text"]

    def test_excluded_columns_records_identifier_and_date(self):
        csv_text = (
            "id,fecha,x,resultado\n" +
            "\n".join(f"P{1000+i},2024-01-{(i % 28) + 1:02d},{i % 5},{'si' if i % 2 else 'no'}" for i in range(15))
            + "\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert set(res["provenance"]["excluded_columns"]) == {"id", "fecha"}

    def test_rows_dropped_null_target_recorded(self):
        csv_text = (
            "x,y,resultado\n1,2,si\n3,4,\n5,6,no\n7,8,\n9,10,si\n11,12,no\n"
        )
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert res["provenance"]["rows_dropped_null_target"] == 2

    def test_feature_name_map_and_target_label_map_recorded(self):
        csv_text = "customer age,resultado\n1,si\n3,no\n5,si\n7,no\n9,si\n11,no\n"
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert res["provenance"]["feature_name_map"] == {"customer age": "customer_age"}
        assert res["provenance"]["target_label_map"] == {"si": "si", "no": "no"}

    def test_target_label_map_is_none_for_regression(self):
        csv_text = "x,resultado\n1,2.5\n2,3.5\n3,4.5\n4,5.5\n5,6.5\n6,7.5\n"
        res = generate_project_from_dataset(csv_text, target_column="resultado")
        assert res["provenance"]["target_label_map"] is None


# ---------------------------------------------------------------------------
# MEDIA (reauditoría 2026-07-17) — rangos editados: min>=max / no finito
# ---------------------------------------------------------------------------

class TestMediaRangeOverrideSemanticValidation:
    _CSV = "x,y,resultado\n1,2,si\n3,4,no\n5,6,si\n7,8,no\n9,10,si\n11,12,no\n"

    def test_inverted_range_raises(self):
        with pytest.raises(DatasetProjectError, match="mínimo mayor o igual"):
            generate_project_from_dataset(self._CSV, target_column="resultado",
                                           column_range_overrides={"x": (10.0, 0.0)})

    def test_equal_min_max_raises(self):
        with pytest.raises(DatasetProjectError, match="mínimo mayor o igual"):
            generate_project_from_dataset(self._CSV, target_column="resultado",
                                           column_range_overrides={"x": (5.0, 5.0)})

    def test_nan_range_raises(self):
        with pytest.raises(DatasetProjectError, match="no es un rango finito"):
            generate_project_from_dataset(self._CSV, target_column="resultado",
                                           column_range_overrides={"x": (float("nan"), 10.0)})

    def test_infinite_range_raises(self):
        with pytest.raises(DatasetProjectError, match="no es un rango finito"):
            generate_project_from_dataset(self._CSV, target_column="resultado",
                                           column_range_overrides={"x": (0.0, float("inf"))})

    def test_valid_range_still_works(self):
        res = generate_project_from_dataset(self._CSV, target_column="resultado",
                                             column_range_overrides={"x": (0.0, 100.0)})
        assert tuple(res["field_ranges"]["x"]) == (0.0, 100.0)


# ---------------------------------------------------------------------------
# MEDIA (reauditoría 2026-07-17) — feature con el nombre reservado del target
# ---------------------------------------------------------------------------

class TestMediaReservedTargetNameCollision:
    def test_feature_named_predicted_class_raises_short_actionable_error(self):
        csv_text = "predicted_class,x,resultado\n1,2,si\n3,4,no\n5,6,si\n7,8,no\n9,10,si\n11,12,no\n"
        with pytest.raises(DatasetProjectError) as exc_info:
            generate_project_from_dataset(csv_text, target_column="resultado")
        msg = str(exc_info.value)
        assert "nombre reservado" in msg
        assert len(msg) < 500  # nunca el dict interno completo de GEN

    def test_feature_named_predicted_value_raises_for_regression(self):
        csv_text = "predicted_value,resultado\n1,2.5\n2,3.5\n3,4.5\n4,5.5\n5,6.5\n6,7.5\n"
        with pytest.raises(DatasetProjectError, match="nombre reservado"):
            generate_project_from_dataset(csv_text, target_column="resultado")

    def test_feature_that_normalizes_to_reserved_name_also_raises(self):
        csv_text = "Predicted Class,x,resultado\n1,2,si\n3,4,no\n5,6,si\n7,8,no\n9,10,si\n11,12,no\n"
        with pytest.raises(DatasetProjectError, match="nombre reservado"):
            generate_project_from_dataset(csv_text, target_column="resultado")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


class TestLaGuardaResidualTieneSuPropiaPrueba:
    """AUDITORIA DEL CONTROLADOR, 2026-09-16 — el hueco que dejó la reparación.

    Tras el renombrado, `_check_categorical_values_safe` se queda puesta como
    **defensa en profundidad**: en el camino normal ya no dispara nunca, porque
    el vocabulario llega renombrado. Medido ese día: **desactivarla entera dejó
    las 33 pruebas del fichero en verde**.

    Eso es exactamente el patrón que en esta casa ha costado tres sabotajes
    verdes en un solo día — «una línea que explica por qué NO hace lo obvio
    necesita una prueba con su nombre». Sin esto, el siguiente que pase la
    «simplifica» por muerta, y el día que el renombrado tenga un hueco el aborto
    ruidoso tampoco estará: se generaría un modelo desalineado **en silencio**,
    que es justo lo que la guarda existía para impedir.

    Dos mitades: que la guarda SIGUE mordiendo, y que SIGUE en el camino.
    """

    def test_la_guarda_SIGUE_rechazando_un_valor_inseguro(self):
        from matrixai.training.dataset_project import _check_categorical_values_safe
        with pytest.raises(DatasetProjectError) as exc:
            _check_categorical_values_safe(["red,blue", "green"], "cat")
        assert "red,blue" in str(exc.value)

    def test_la_guarda_SIGUE_EN_EL_CAMINO_despues_de_renombrar(self):
        """La otra mitad: una guarda que muerde pero a la que ya nadie llama
        protege lo mismo que una borrada. Se cuenta cuántas veces la invoca la
        generación de un proyecto con un valor inseguro dentro."""
        from matrixai.training import dataset_project as dp

        llamadas: list[tuple] = []
        original = dp._check_categorical_values_safe

        def espia(values, col):
            llamadas.append((tuple(values), col))
            return original(values, col)

        csv_text = (
            'cat,x,resultado\n'
            '"red,blue",1,si\ngreen,2,no\n"red,blue",3,si\n'
            'green,4,no\n"red,blue",5,si\ngreen,6,no\n'
        )
        dp._check_categorical_values_safe = espia
        try:
            dp.generate_project_from_dataset(csv_text=csv_text, target_column="resultado")
        finally:
            dp._check_categorical_values_safe = original

        assert llamadas, (
            "nadie llama ya a `_check_categorical_values_safe`: la defensa en "
            "profundidad que el comentario promete no existe")
        for values, _col in llamadas:
            assert not [v for v in values if "," in v or "]" in v], (
                "la guarda recibe valores SIN renombrar: o el renombrado no ha "
                "pasado antes, o la guarda esta puesta en el sitio equivocado")
