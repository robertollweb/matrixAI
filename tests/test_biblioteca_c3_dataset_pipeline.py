# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C3 — motor de pipeline declarativo
(vocabulario cerrado de 8 operaciones) + anti-fuga temporal (invariante
6/13). Cubre cada operación aislada, el vocabulario cerrado (op
desconocida/parámetro inválido/desconocido -> error accionable),
orden+determinismo, el ejemplo canónico del mar, el linaje temporal
compuesto (anti-fuga que sobrevive a rename/lag_window encadenados) y la
reauditoría 2026-07-17 (4 ALTA + 2 MEDIA, todos reproducidos y
corregidos): linaje perdido tras rename/lag_window, formato de
lag_window distinto del contrato, corrupción de rename en swap/cadena,
validate_pipeline_output ciego a nulos en features, vocabulario abierto
en parámetros.
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

    def test_interpolate_is_causal_forward_fill_interior_gap(self):
        """Auditoría 2026-07-17 [ALTA]: interpolate ya NO usa el valor
        SIGUIENTE (eso era una fuga temporal real) — un hueco interior se
        rellena con el ÚLTIMO valor conocido hacia atrás (forward-fill),
        nunca con una media que incorpore un valor futuro."""
        rows = [{"y": "0"}, {"y": "10"}, {"y": ""}, {"y": ""}, {"y": "40"}, {"y": "50"}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])
        assert [r["y"] for r in res.rows] == ["0", "10", "10", "10", "40", "50"]

    def test_interpolate_leading_gap_stays_null_trailing_gap_now_fillable(self):
        """Un hueco INICIAL (sin valor previo) sigue sin poder rellenarse
        causalmente. Un hueco FINAL sí es causal (solo depende del pasado)
        y ahora se rellena — antes se dejaba vacío por prudencia excesiva,
        pero forward-fill de un tramo final no incorpora ningún futuro."""
        rows = [{"y": ""}, {"y": "10"}, {"y": "20"}, {"y": ""}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])
        assert [r["y"] for r in res.rows] == ["", "10", "20", "20"]

    def test_interpolate_never_uses_a_future_value(self):
        """Repro directo del hallazgo: una fuga por interpolación bidireccional
        se habría manifestado como que la fila en t reflejase el valor de
        t+1. Con una única fila conocida DESPUÉS del hueco (sin nada antes),
        el hueco debe seguir vacío — jamás tomar prestado ese valor futuro."""
        rows = [{"y": ""}, {"y": ""}, {"y": "99"}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])
        assert [r["y"] for r in res.rows] == ["", "", "99"]

    # -- Reauditoría 2026-07-17 [ALTA]: interpolate solo es causal si las
    # filas YA están en orden temporal — exige sort_temporal ANTES --

    def test_interpolate_before_sort_temporal_raises(self):
        """Repro exacto del hallazgo: sin ordenar primero, 'hacia atrás en
        la lista' no es 'hacia atrás en el tiempo' — una fila anterior en
        la lista puede ser una fecha FUTURA."""
        rows = [
            {"fecha": "2024-01-03", "y": ""},
            {"fecha": "2024-01-01", "y": "10"},
            {"fecha": "2024-01-02", "y": ""},
        ]
        with pytest.raises(PipelineError, match="antes de sort_temporal"):
            run_pipeline(rows, [
                {"op": "missing_values", "strategy": "interpolate", "columns": ["y"]},
                {"op": "sort_temporal", "column": "fecha"},
            ])

    def test_interpolate_after_sort_temporal_is_allowed(self):
        rows = [
            {"fecha": "2024-01-03", "y": "30"},
            {"fecha": "2024-01-01", "y": "10"},
            {"fecha": "2024-01-02", "y": ""},
        ]
        res = run_pipeline(rows, [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "missing_values", "strategy": "interpolate", "columns": ["y"]},
        ])
        # tras ordenar: 01-01=10, 01-02=(hueco->ffill de 10), 01-03=30
        assert [r["y"] for r in res.rows] == ["10", "10", "30"]

    def test_interpolate_without_any_sort_temporal_is_allowed(self):
        """Sin `sort_temporal` en el pipeline, se asume que el caller ya
        entrega las filas en el orden que le interesa — este motor no
        puede inventar una columna temporal que no se le ha señalado."""
        rows = [{"y": "1"}, {"y": ""}, {"y": "3"}]
        res = run_pipeline(rows, [{"op": "missing_values", "strategy": "interpolate", "columns": ["y"]}])
        assert [r["y"] for r in res.rows] == ["1", "1", "3"]

    def test_interpolate_multiple_before_single_sort_all_flagged(self):
        rows = [{"fecha": "2024-01-01", "y": "1", "z": "2"}]
        with pytest.raises(PipelineError, match="antes de sort_temporal"):
            run_pipeline(rows, [
                {"op": "missing_values", "strategy": "interpolate", "columns": ["y"]},
                {"op": "missing_values", "strategy": "interpolate", "columns": ["z"]},
                {"op": "sort_temporal", "column": "fecha"},
            ])

    def test_interpolate_between_two_temporal_sorts_raises(self):
        """La primera ordenación no basta: una segunda puede redefinir el
        eje temporal y convertir el forward-fill intermedio en una fuga."""
        rows = [
            {"fecha_a": "2024-01-01", "fecha_b": "2024-01-03", "y": "10"},
            {"fecha_a": "2024-01-02", "fecha_b": "2024-01-02", "y": ""},
            {"fecha_a": "2024-01-03", "fecha_b": "2024-01-01", "y": "30"},
        ]
        with pytest.raises(PipelineError, match="antes de sort_temporal"):
            run_pipeline(rows, [
                {"op": "sort_temporal", "column": "fecha_a"},
                {"op": "missing_values", "strategy": "interpolate", "columns": ["y"]},
                {"op": "sort_temporal", "column": "fecha_b"},
            ])

    def test_all_temporal_sorts_before_interpolate_are_allowed(self):
        rows = [
            {"fecha_a": "2024-01-03", "fecha_b": "2024-01-01", "y": "30"},
            {"fecha_a": "2024-01-01", "fecha_b": "2024-01-02", "y": "10"},
            {"fecha_a": "2024-01-02", "fecha_b": "2024-01-03", "y": ""},
        ]
        res = run_pipeline(rows, [
            {"op": "sort_temporal", "column": "fecha_a"},
            {"op": "sort_temporal", "column": "fecha_b"},
            {"op": "missing_values", "strategy": "interpolate", "columns": ["y"]},
        ])
        assert [r["y"] for r in res.rows] == ["30", "10", "10"]

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

    def test_non_string_destination_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="strings no vacíos"):
            run_pipeline(rows, [{"op": "rename", "mapping": {"a": 123}}])

    def test_empty_string_destination_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="strings no vacíos"):
            run_pipeline(rows, [{"op": "rename", "mapping": {"a": ""}}])

    # -- Auditoría 2026-07-17 [ALTA]: corrupción de datos en swap/cadena --

    def test_swap_exchanges_values_correctly(self):
        """{"x":"y","y":"x"} debe INTERCAMBIAR los valores, no perder una
        columna. Reproducción exacta del hallazgo: la versión previa dejaba
        una sola columna con el valor equivocado."""
        rows = [{"x": "1", "y": "2"}]
        res = run_pipeline(rows, [{"op": "rename", "mapping": {"x": "y", "y": "x"}}])
        assert res.rows == [{"y": "1", "x": "2"}]
        assert set(res.rows[0].keys()) == {"x", "y"}

    def test_chain_rename_relabels_simultaneously(self):
        """{"a":"b","b":"c"} es un relabel SIMULTÁNEO (foto de los valores
        originales) — a pasa a llamarse b, la b ORIGINAL pasa a llamarse c;
        ninguna se pierde ni se pisa a medio camino."""
        rows = [{"a": "1", "b": "2"}]
        res = run_pipeline(rows, [{"op": "rename", "mapping": {"a": "b", "b": "c"}}])
        assert res.rows == [{"b": "1", "c": "2"}]

    def test_swap_preserves_row_count_and_all_original_values(self):
        rows = [{"x": str(i), "y": str(i * 10)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "rename", "mapping": {"x": "y", "y": "x"}}])
        assert len(res.rows) == 5
        for i, row in enumerate(res.rows):
            assert row["y"] == str(i)
            assert row["x"] == str(i * 10)


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


# ---------------------------------------------------------------------------
# lag_window — formato EXACTO del contrato: columns (lista) + window
# ---------------------------------------------------------------------------

class TestLagWindow:
    def test_adds_window_lag_columns_per_column(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": 2}])
        assert res.rows[0] == {"x": "0", "x_lag1": "", "x_lag2": ""}
        assert res.rows[2] == {"x": "2", "x_lag1": "1", "x_lag2": "0"}
        assert res.rows[4] == {"x": "4", "x_lag1": "3", "x_lag2": "2"}

    def test_multiple_columns_same_window(self):
        """Reproduce el ejemplo EXACTO del contrato: una lista de columnas,
        la misma ventana para todas."""
        rows = [{"a": str(i), "b": str(i * 10)} for i in range(4)]
        res = run_pipeline(rows, [{"op": "lag_window", "columns": ["a", "b"], "window": 1}])
        header = set(res.rows[0].keys())
        assert header == {"a", "b", "a_lag1", "b_lag1"}
        assert res.rows[2]["a_lag1"] == "1"
        assert res.rows[2]["b_lag1"] == "10"

    def test_never_drops_rows(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": 3}])
        assert len(res.rows) == 5

    def test_window_zero_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="window debe ser un entero >= 1"):
            run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": 0}])

    def test_negative_window_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="window debe ser un entero >= 1"):
            run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": -1}])

    def test_non_int_window_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="window debe ser un entero"):
            run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": "3"}])

    def test_empty_columns_list_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="lista no vacía"):
            run_pipeline(rows, [{"op": "lag_window", "columns": [], "window": 1}])

    def test_singular_column_key_rejected_not_silently_ignored(self):
        """El formato ANTIGUO (column/k singular) ya no existe — debe
        fallar con un error accionable, nunca ignorar 'column' en silencio
        y reventar con un mensaje que confunda al caller."""
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError):
            run_pipeline(rows, [{"op": "lag_window", "column": "x", "k": 1}])

    def test_window_over_max_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="supera el máximo"):
            run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": 100000}])

    def test_never_flagged_by_anti_leakage(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "lag_window", "columns": ["x"], "window": 2}])
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

    def test_horizon_over_max_raises(self):
        rows = [{"x": "1"}]
        with pytest.raises(PipelineError, match="supera el máximo"):
            run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 100000}])

    def test_result_column_has_positive_offset(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [{"op": "shift_target", "column": "x", "horizon": 1}])
        assert res.column_offsets["x_target_h1"] == 1
        assert res.column_offsets["x"] == 0


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

    def test_drops_lineage_entry(self):
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [
            {"op": "shift_target", "column": "x", "horizon": 1},
            {"op": "drop_columns", "columns": ["x_target_h1"]},
        ])
        assert "x_target_h1" not in res.column_offsets


# ---------------------------------------------------------------------------
# Vocabulario cerrado (op + parámetros)
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

    # -- Auditoría 2026-07-17 [MEDIA]: parámetros extra ya NO se ignoran --

    def test_unexpected_extra_param_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="desconocido"):
            run_pipeline(rows, [{"op": "drop_duplicates", "unexpected": "ignored"}])

    def test_unexpected_param_on_typed_op_raises(self):
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="desconocido"):
            run_pipeline(rows, [{"op": "cast", "column": "a", "to": "number", "extra": 1}])

    def test_typo_in_param_name_is_caught(self):
        """Un typo ("columnn" en vez de "column") debe fallar, no
        interpretarse como "sin columna declarada"."""
        rows = [{"a": "1"}]
        with pytest.raises(PipelineError, match="desconocido"):
            run_pipeline(rows, [{"op": "sort_temporal", "columnn": "a"}])


# ---------------------------------------------------------------------------
# Orden y determinismo
# ---------------------------------------------------------------------------

class TestOrderAndDeterminism:
    def test_same_input_same_ops_gives_identical_output(self):
        rows = _mar_rows(12)
        ops = [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "altura_ola", "horizon": 1},
            {"op": "lag_window", "columns": ["altura_ola"], "window": 2},
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
            {"op": "lag_window", "columns": ["y"], "window": 1},
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
            {"op": "lag_window", "columns": ["altura_ola"], "window": 2},
            {"op": "missing_values", "strategy": "drop"},
        ]
        res = run_pipeline(rows, ops)
        # 2 filas caídas al principio (lag_window sin historia) + 1 al final
        # (shift_target sin futuro) = 3 de 10.
        assert len(res.rows) == 7
        feature_cols = ["altura_ola", "altura_ola_lag1", "altura_ola_lag2", "temperatura"]
        errors = validate_pipeline_output(
            res.rows, target_column="altura_ola_target_h1", feature_columns=feature_cols,
        )
        assert errors == []
        leaks = check_anti_leakage(res, feature_columns=feature_cols)
        assert leaks == []


# ---------------------------------------------------------------------------
# Anti-fuga (invariante 6/13) — "dataset trampa" del contrato + linaje
# compuesto (reauditoría 2026-07-17 [ALTA]: la versión anterior solo
# recordaba el NOMBRE creado directamente por shift_target — se perdía tras
# rename/lag_window, o se quedaba pegado a un nombre reciclado)
# ---------------------------------------------------------------------------

class TestAntiLeakageTrapDataset:
    def test_shifted_column_used_as_feature_is_detected(self):
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
        rows = [{"x": "1", "futuro_x": "2"}]
        res = run_pipeline(rows, [{"op": "cast", "column": "x", "to": "number"}])
        assert check_anti_leakage(res, feature_columns=["x", "futuro_x"]) == []

    def test_shift_target_source_column_itself_is_not_flagged(self):
        rows = _mar_rows(6)
        res = run_pipeline(rows, [
            {"op": "sort_temporal", "column": "fecha"},
            {"op": "shift_target", "column": "altura_ola", "horizon": 1},
        ])
        assert check_anti_leakage(res, feature_columns=["altura_ola", "temperatura"]) == []

    # -- Reauditoría [ALTA]: los 3 repros exactos del hallazgo --

    def test_shift_then_rename_still_flagged(self):
        """Repro 1: shift_target(as='future') -> rename(future->innocent).
        El linaje se TRANSFIERE en el rename, no se pierde."""
        rows = [{"x": str(i)} for i in range(5)]
        res = run_pipeline(rows, [
            {"op": "shift_target", "column": "x", "horizon": 1, "as": "future"},
            {"op": "rename", "mapping": {"future": "innocent"}},
        ])
        leaks = check_anti_leakage(res, feature_columns=["innocent"])
        assert len(leaks) == 1
        assert "innocent" in leaks[0]

    def test_shift_then_lag_still_flagged_with_residual_offset(self):
        """Repro 2: shift_target(horizon=3) -> lag_window(k=1) sobre la
        columna desplazada. x_target_h3_lag1 = x_target_h3 en t-1 = x en
        (t-1)+3 = t+2 — SIGUE en el futuro (offset neto +2), no +3 ni 0."""
        rows = [{"x": str(i)} for i in range(10)]
        res = run_pipeline(rows, [
            {"op": "shift_target", "column": "x", "horizon": 3},
            {"op": "lag_window", "columns": ["x_target_h3"], "window": 1},
        ])
        assert res.column_offsets["x_target_h3_lag1"] == 2
        leaks = check_anti_leakage(res, feature_columns=["x_target_h3_lag1"])
        assert len(leaks) == 1

    def test_drop_then_rename_of_unrelated_column_is_not_a_false_positive(self):
        """Repro 3: shift_target(as='future') -> drop_columns(['future'])
        -> rename(x -> 'future') [x es la columna PRESENTE original, sin
        relación]. El nombre reciclado 'future' NO debe arrastrar el
        linaje de la fuga ya eliminada — falso positivo si lo hiciera."""
        rows = [{"x": str(i), "y": str(i * 2)} for i in range(5)]
        res = run_pipeline(rows, [
            {"op": "shift_target", "column": "x", "horizon": 1, "as": "future"},
            {"op": "drop_columns", "columns": ["future"]},
            {"op": "rename", "mapping": {"y": "future"}},
        ])
        assert res.column_offsets["future"] == 0
        assert check_anti_leakage(res, feature_columns=["future"]) == []


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

    def test_min_rows_not_positive_int_raises(self):
        rows = [{"y": "1"}, {"y": "2"}]
        with pytest.raises(PipelineError, match="min_rows"):
            validate_pipeline_output(rows, target_column="y", min_rows=0)

    def test_min_rows_non_int_raises(self):
        rows = [{"y": "1"}, {"y": "2"}]
        with pytest.raises(PipelineError, match="min_rows"):
            validate_pipeline_output(rows, target_column="y", min_rows="2")

    # -- Reauditoría [MEDIA]: nulos residuales en FEATURES, no solo target --

    def test_residual_nulls_in_feature_reported(self):
        """El target completo NO basta — un lag_window(window=2) deja las
        2 primeras filas de la feature vacías; antes esto pasaba `[]`."""
        rows = [
            {"y": "1", "x_lag1": ""}, {"y": "2", "x_lag1": ""},
            {"y": "3", "x_lag1": "1"}, {"y": "4", "x_lag1": "2"},
        ]
        errors = validate_pipeline_output(rows, target_column="y", feature_columns=["x_lag1"])
        assert errors and "x_lag1" in errors[0]

    def test_missing_feature_column_reported(self):
        rows = [{"y": "1"}, {"y": "2"}]
        errors = validate_pipeline_output(rows, target_column="y", feature_columns=["no_existe"])
        assert errors and "no_existe" in errors[0]

    def test_clean_features_report_no_errors(self):
        rows = [{"y": "1", "x": "9"}, {"y": "2", "x": "8"}]
        assert validate_pipeline_output(rows, target_column="y", feature_columns=["x"]) == []

    # -- Reauditoría 2026-07-17 [MEDIA]: validación de tipos (opcional,
    # explícita — este módulo no tiene esquema propio) --

    def test_expected_types_number_rejects_non_numeric(self):
        rows = [{"y": "1", "x": "abc"}, {"y": "2", "x": "9"}]
        errors = validate_pipeline_output(rows, target_column="y", expected_types={"x": "number"})
        assert errors and "x" in errors[0] and "no numérico" in errors[0]

    def test_expected_types_integer_rejects_non_integer(self):
        rows = [{"y": "1", "x": "3.5"}, {"y": "2", "x": "9"}]
        errors = validate_pipeline_output(rows, target_column="y", expected_types={"x": "integer"})
        assert errors and "no entero" in errors[0]

    def test_expected_types_string_never_rejects(self):
        rows = [{"y": "1", "x": "anything at all"}, {"y": "2", "x": "más texto"}]
        assert validate_pipeline_output(rows, target_column="y", expected_types={"x": "string"}) == []

    def test_expected_types_null_values_are_skipped(self):
        rows = [{"y": "1", "x": ""}, {"y": "2", "x": "9"}]
        assert validate_pipeline_output(rows, target_column="y", expected_types={"x": "number"}) == []

    def test_expected_types_missing_column_reported(self):
        rows = [{"y": "1"}, {"y": "2"}]
        errors = validate_pipeline_output(rows, target_column="y", expected_types={"no_existe": "number"})
        assert errors and "no_existe" in errors[0]

    def test_expected_types_clean_types_report_no_errors(self):
        rows = [{"y": "1", "x": "9"}, {"y": "2", "x": "10"}]
        assert validate_pipeline_output(rows, target_column="y", expected_types={"x": "integer"}) == []

    def test_expected_types_unknown_type_raises(self):
        rows = [{"y": "1", "x": "9"}, {"y": "2", "x": "8"}]
        with pytest.raises(PipelineError, match="desconocido"):
            validate_pipeline_output(rows, target_column="y", expected_types={"x": "date"})

    def test_cast_alone_is_not_enough_reproduces_the_finding(self):
        """Repro exacto del hallazgo: un pipeline que declara `cast` para
        UNA columna y omite otra deja esa segunda columna sin validar si
        nadie pasa `expected_types` — pero SI se pasa, el hueco se cierra."""
        rows = [{"y": "1", "x": "9"}, {"y": "2", "x": "abc"}]  # x sin castear, valor sucio
        res = run_pipeline(rows, [{"op": "cast", "column": "y", "to": "integer"}])
        assert validate_pipeline_output(res.rows, target_column="y") == []  # ciego a x
        errors = validate_pipeline_output(
            res.rows, target_column="y", expected_types={"x": "number"},
        )
        assert errors and "x" in errors[0]
