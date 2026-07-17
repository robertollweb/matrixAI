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
# ---------------------------------------------------------------------------

class TestAltaUnsafeCategoryValues:
    def test_comma_in_categorical_value_raises_actionable_error(self):
        csv_text = (
            'cat,x,resultado\n'
            '"red,blue",1,si\ngreen,2,no\n"red,blue",3,si\n'
            'green,4,no\n"red,blue",5,si\ngreen,6,no\n'
        )
        with pytest.raises(DatasetProjectError, match="coma"):
            generate_project_from_dataset(csv_text, target_column="resultado")

    def test_closing_bracket_in_categorical_value_raises(self):
        csv_text = (
            "cat,x,resultado\n"
            "a]b,1,si\nverde,2,no\na]b,3,si\nverde,4,no\na]b,5,si\nverde,6,no\n"
        )
        with pytest.raises(DatasetProjectError, match="coma"):
            generate_project_from_dataset(csv_text, target_column="resultado")


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
