# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C3 — motor de pipeline declarativo
(vocabulario cerrado de 8 operaciones) + anti-fuga temporal (invariante 6).
Cubre cada operación aislada, el vocabulario cerrado (op desconocida/
parámetro inválido -> error accionable), orden+determinismo, el ejemplo
canónico del mar (sort_temporal + shift_target + lag_window +
missing_values) y el "dataset trampa" del contrato (columna futura colada
-> detectada por HISTORIAL de operaciones, no por nombre).
"""
from __future__ import annotations

import pytest

from matrixai.training.dataset_pipeline import (
    PipelineError,
    check_anti_leakage,
    run_pipeline,
    validate_pipeline_output,
)


def _mar_rows(n: int = 10) -> list[dict[str, str]]:
    return [
        {
            "fecha": f"2024-01-{d:02d}",
            "altura_ola": str(round(2.0 + d * 0.1, 2)),
            "temperatura": str(round(15.0 + d * 0.05, 2)),
        }
        for d in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Cada operación aislada
# ---------------------------------------------------------------------------

class TestSortTemporal:
    def test_sorts_ascending(self):
        rows = [{"fecha": "2024-01-03", "x": "3"}, {"fecha": "2024-01-01", "x": "1"},
                {"fecha": "2024-01-02", "x": "2"}]
        res = run_pipeline(rows, [{"op": "sort_temporal", "column": "fecha"}])
        assert [r["x"] for r in res.rows] == ["1", "2", "3"]

    def test_unrecognized_date_format_raises(self):
        rows = [{"fecha": "not-a-date", "x": "1"}, {"fecha": "also-not", "x": "2"}]
        with pytest.raises(PipelineError, match="formato de fecha"):
            run_pipeline(rows, [{"op": "sort_temporal", "column": "fecha"}])

    def test_unknown_column_raises(self):
        rows = [{"fecha": "2024-01-01", "x": "1"}]
        with pytest.raises(PipelineError, match="no existe"):
            run_pipeline(rows, [{"op": "sort_temporal", "column": "no_existe"}])

    def test_stable_for_equal_dates(self):
        rows = [{"fecha": "2024-01-01", "x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "sort_temporal", "column": "fecha"}])
        assert [r["x"] for r in res.rows] == ["0", "1", "2", "3", "4"]


class TestDropDuplicates:
    def test_removes_exact_duplicates_keeps_first(self):
        rows = [{"a": "1", "b": "2"}, {"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        res = run_pipeline(rows, [{"op": "drop_duplicates"}])
        assert res.rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]

    def test_no_duplicates_is_noop(self):
        rows = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
        res = run_pipeline(rows, [{"op": "drop_duplicates"}])
        assert res.rows == rows


class TestMissingValues:
    def test_drop_removes_rows_with_null_in_any_column(self):
        rows = [{"a": "1", "b": "2"}, {"a": "", "b": "2"}, {"a": "3", "b": "n/a"}, {"a": "4", "b": "5"}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "drop"}])
        assert res.rows == [{"a": "1", "b": "2"}, {"a": "4", "b": "5"}]

    def test_drop_scoped_to_declared_columns(self):
        rows = [{"a": "1", "b": ""}, {"a": "", "b": "2"}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "drop", "columns": ["a"]}])
        assert res.rows == [{"a": "1", "b": ""}]

    def test_interpolate_fills_interior_gap(self):
        rows = [{"y": "0"}, {"y": "10"}, {"y": ""}, {"y": ""}, {"y": "40"}, {"y": "50"}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])
        assert [r["y"] for r in res.rows] == ["0", "10", "20", "30", "40", "50"]

    def test_interpolate_leaves_leading_trailing_gap_null(self):
        rows = [{"y": ""}, {"y": "10"}, {"y": "20"}, {"y": ""}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])
        assert [r["y"] for r in res.rows] == ["", "10", "20", ""]

    def test_interpolate_non_numeric_raises(self):
        rows = [{"y": "abc"}, {"y": ""}, {"y": "10"}]
        with pytest.raises(PipelineError, match="numérico"):
            run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])

    def test_unknown_strategy_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="strategy"):
            run_pipeline(rows, [{"op": "missing_values", "strategy": "invent"}])


class TestRename:
    def test_renames_column_in_every_row(self):
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        res = run_pipeline(rows, [{"op": "rename", "mapping": {"a": "alpha"}}])
        assert res.rows == [{"alpha": "1", "b": "2"}, {"alpha": "3", "b": "4"}]
        assert res.steps[0].columns_after == ["alpha", "b"]

    def test_collision_with_existing_column_raises(self):
        rows = [{"a": "1", "b": "2"}]
        with pytest.raises(PipelineError, match="choca"):
            run_pipeline(rows, [{"op": "rename", "mapping": {"a": "b"}}])

    def test_unknown_source_column_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="no existe"):
            run_pipeline(rows, [{"op": "rename", "mapping": {"no_existe": "x"}}])


class TestCast:
    def test_number_reformats_value(self):
        rows = [{"x": "3.0"}, {"x": "3.5"}]
        res = run_pipeline(rows, [{"op": "cast", "column": "x", "to": "number"}])
        assert [r["x"] for r in res.rows] == ["3", "3.5"]

    def test_integer_rejects_non_integer_value(self):
        rows = [{"x": "3.5"}]
        with pytest.raises(PipelineError, match="entero exacto"):
            run_pipeline(rows, [{"op": "cast", "column": "x", "to": "integer"}])

    def test_integer_accepts_whole_float(self):
        rows = [{"x": "4.0"}]
        res = run_pipeline(rows, [{"op": "cast", "column": "x", "to": "integer"}])
        assert res.rows[0]["x"] == "4"

    def test_string_is_noop(self):
        rows = [{"x": "hola"}]
        res = run_pipeline(rows, [{"op": "cast", "column": "x", "to": "string"}])
        assert res.rows[0]["x"] == "hola"

    def test_non_numeric_value_raises(self):
        rows = [{"x": "abc"}]
        with pytest.raises(PipelineError, match="no es numérico"):
            run_pipeline(rows, [{"op": "cast", "column": "x", "to": "number"}])

    def test_null_values_are_skipped(self):
        rows = [{"x": ""}, {"x": "3"}]
        res = run_pipeline(rows, [{"op": "cast", "column": "x", "to": "number"}])
        assert res.rows[0]["x"] == ""

    def test_unknown_to_type_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="to debe ser"):
            run_pipeline(rows, [{"op": "cast", "column": "x", "to": "date"}])


class TestLagWindow:
    def test_adds_k_lag_columns(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": 2}])
        assert res.rows[0] == {"x": "0", "x_lag1": "", "x_lag2": ""}
        assert res.rows[2] == {"x": "2", "x_lag1": "1", "x_lag2": "0"}
        assert res.rows[4] == {"x": "4", "x_lag1": "3", "x_lag2": "2"}

    def test_never_drops_rows(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": 3}])
        assert len(res.rows) == 5

    def test_k_zero_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="k debe ser un entero >= 1"):
            run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": 0}])

    def test_negative_k_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="k debe ser un entero >= 1"):
            run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": -1}])

    def test_non_int_k_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="k debe ser un entero"):
            run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": "3"}])

    def test_never_flagged_by_anti_leakage(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": 2}])
        assert check_anti_leakage(res, feature_columns=["x", "x_lag1", "x_lag2"]) == []


class TestShiftTarget:
    def test_shifts_value_forward_by_horizon(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 1}])
        assert [r["x_target_h1"] for r in res.rows] == ["1", "2", "3", "4", ""]

    def test_custom_as_name(self):
        rows = [{"x": str(i)} for i in range(3)]
        res = run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 1, "as": "y_future"}])
        assert "y_future" in res.rows[0]

    def test_tail_rows_get_empty_string_never_dropped(self):
        rows = [{"x": str(i)} for i in range(4)]
        res = run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 2}])
        assert len(res.rows) == 4
        assert res.rows[-1]["x_target_h2"] == ""
        assert res.rows[-2]["x_target_h2"] == ""

    def test_horizon_zero_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="horizon debe ser un entero >= 1"):
            run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 0}])

    def test_negative_horizon_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="horizon debe ser un entero >= 1"):
            run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": -1}])

    def test_result_column_tracked_as_forward_shifted(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 1}])
        assert res.forward_shifted_columns == frozenset({"x_target_h1"})


class TestDropColumns:
    def test_removes_declared_columns(self):
        rows = [{"a": "1", "b": "2", "c": "3"}]
        res = run_pipeline(rows, [{"op": "drop_columns", "columns": ["b"]}])
        assert res.rows == [{"a": "1", "c": "3"}]
        assert res.steps[0].columns_after == ["a", "c"]

    def test_unknown_column_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="no existe"):
            run_pipeline(rows, [{"op": "drop_columns", "columns": ["no_existe"]}])

    def test_empty_list_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="lista no vacía"):
            run_pipeline(rows, [{"op": "drop_columns", "columns": []}])


# ---------------------------------------------------------------------------
# Vocabulario cerrado
# ---------------------------------------------------------------------------

class TestClosedVocabulary:
    def test_unknown_operation_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="operación desconocida"):
            run_pipeline(rows, [{"op": "frobnicate"}])

    def test_missing_op_key_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="'op'"):
            run_pipeline(rows, [{"column": "a"}])

    def test_empty_rows_raises(self):
        with pytest.raises(PipelineError, match="al menos una fila"):
            run_pipeline([], [{"op": "drop_duplicates"}])

    def test_empty_operations_is_noop(self):
        rows = [{"a": "1"}, {"a": "2"}]
        res = run_pipeline(rows, [])
        assert res.rows == rows
        assert res.steps == []


# ---------------------------------------------------------------------------
# Orden y determinismo
# ---------------------------------------------------------------------------

class TestOrderAndDeterminism:
    def test_same_input_same_ops_gives_identical_output(self):
        rows = _mar_rows(12)
        ops = [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "altura_ola", "horizon": 1},
            {"op": "lag_window", "column": "altura_ola", "k": 2},
            {"op": "missing_values", "strategy": "drop"},
        ]
        res1 = run_pipeline(rows, ops)
        res2 = run_pipeline(rows, ops)
        assert res1.rows == res2.rows

    def test_never_mutates_caller_input(self):
        rows = _mar_rows(5)
        original = [dict(r) for r in rows]
        run_pipeline(rows, [{"op": "rename", "mapping": {"fecha": "f"}},
                             {"op": "drop_columns", "columns": ["temperatura"]}])
        assert rows == original

    def test_operation_order_changes_result(self):
        """rename ANTES de lag_window produce columnas de lag con el nombre
        NUEVO; en el orden inverso, lag_window vería la columna vieja."""
        rows = [{"x": str(i)} for i in range(3)]
        res_rename_first = run_pipeline(rows, [
            {"op": "rename", "mapping": {"x": "y"}},
            {"op": "lag_window", "column": "y", "k": 1},
        ])
        assert "y_lag1" in res_rename_first.rows[0]

    def test_steps_record_order_params_rows_before_after(self):
        rows = [{"a": "1", "b": "2"}, {"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        res = run_pipeline(rows, [
            {"op": "drop_duplicates"},
            {"op": "rename", "mapping": {"a": "alpha"}},
        ])
        assert [s.operation for s in res.steps] == ["drop_duplicates", "rename"]
        assert res.steps[0].rows_before == 3
        assert res.steps[0].rows_after == 2
        assert res.steps[1].params == {"mapping": {"a": "alpha"}}
        assert res.steps[1].columns_before == ["a", "b"]
        assert res.steps[1].columns_after == ["alpha", "b"]


# ---------------------------------------------------------------------------
# Ejemplo canónico del mar (E2E del módulo, sin entrenar)
# ---------------------------------------------------------------------------

class TestCanonicalMarExample:
    def test_full_pipeline_produces_clean_lagged_dataset(self):
        rows = _mar_rows(10)
        ops = [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "altura_ola", "horizon": 1},
            {"op": "lag_window", "column": "altura_ola", "k": 2},
            {"op": "missing_values", "strategy": "drop"},
        ]
        res = run_pipeline(rows, ops)
        # 2 filas caídas al principio (lag_window sin historia) + 1 al final
        # (shift_target sin futuro) = 3 de 10.
        assert len(res.rows) == 7
        errors = validate_pipeline_output(res.rows, target_column="altura_ola_target_h1")
        assert errors == []
        leaks = check_anti_leakage(
            res, feature_columns=["altura_ola", "altura_ola_lag1", "altura_ola_lag2", "temperatura"],
        )
        assert leaks == []


# ---------------------------------------------------------------------------
# Anti-fuga (invariante 6) — "dataset trampa" del contrato
# ---------------------------------------------------------------------------

class TestAntiLeakageTrapDataset:
    def test_shifted_column_used_as_feature_is_detected(self):
        """El caso trampa EXACTO del contrato: una columna con información
        posterior a t colada entre las features — detectada por HISTORIAL
        de shift_target, nunca por su nombre (aquí incluso se le da un
        nombre inocuo, "clima_manana", para probar que el nombre es
        irrelevante)."""
        rows = _mar_rows(10)
        res = run_pipeline(rows, [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "altura_ola", "horizon": 1, "as": "clima_manana"},
        ])
        leaks = check_anti_leakage(
            res, feature_columns=["altura_ola", "temperatura", "clima_manana"],
        )
        assert len(leaks) == 1
        assert "clima_manana" in leaks[0]

    def test_innocuous_looking_name_does_not_escape_detection(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [
            {"op": "shift_target", "column": "x", "horizon": 1, "as": "totally_safe_column"},
        ])
        assert check_anti_leakage(res, feature_columns=["totally_safe_column"]) != []

    def test_alarming_looking_name_without_shift_is_not_flagged(self):
        """Un nombre que SUENA a fuga ("futuro_x") pero que NUNCA pasó por
        shift_target no es fuga — la detección es por historial, no por
        nombre, en ambas direcciones."""
        rows = [{"x": "1", "futuro_x": "2"}]
        res = run_pipeline(rows, [{"op": "cast", "column": "x", "to": "number"}])
        assert check_anti_leakage(res, feature_columns=["x", "futuro_x"]) == []

    def test_shift_target_source_column_itself_is_not_flagged(self):
        """Usar altura_ola(t) como feature para predecir altura_ola(t+1) es
        autorregresión normal, NO fuga — solo el resultado DESPLAZADO lo es."""
        rows = _mar_rows(6)
        res = run_pipeline(rows, [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "altura_ola", "horizon": 1},
        ])
        assert check_anti_leakage(res, feature_columns=["altura_ola", "temperatura"]) == []


# ---------------------------------------------------------------------------
# validate_pipeline_output
# ---------------------------------------------------------------------------

class TestValidatePipelineOutput:
    def test_min_rows_violation(self):
        rows = [{"y": "1"}]
        errors = validate_pipeline_output(rows, target_column="y", min_rows=2)
        assert errors and "fila" in errors[0]

    def test_missing_target_column(self):
        rows = [{"a": "1"}, {"a": "2"}]
        errors = validate_pipeline_output(rows, target_column="y")
        assert errors and "objetivo" in errors[0]

    def test_residual_nulls_in_target_reported(self):
        rows = [{"y": "1"}, {"y": ""}, {"y": "3"}]
        errors = validate_pipeline_output(rows, target_column="y")
        assert errors and "1 fila" in errors[0]

    def test_clean_dataset_has_no_errors(self):
        rows = [{"y": "1"}, {"y": "2"}, {"y": "3"}]
        assert validate_pipeline_output(rows, target_column="y") == []
