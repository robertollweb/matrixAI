# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C4 — flujo A, caso serie temporal.

`generate_temporal_project_from_dataset(csv_text, target_column,
temporal_column=, horizon=, lag_window_columns=, lag_window_size=)`:
envoltorio delgado sobre C3 (`run_pipeline`: sort_temporal + shift_target +
lag_window opcional + missing_values drop) + C2 (`generate_project_from_
dataset`, cero caminos paralelos — invariante 4). Cubre el ejemplo canónico
del mar (serie temporal pura, sin salida explícita) end-to-end hasta
validar el CSV preparado contra el modelo generado, los errores accionables
de cada parámetro, y la defensa en profundidad anti-fuga (invariante 6/13)
para un pipeline mal construido.
"""
from __future__ import annotations

import pytest

from matrixai.playground import _validate_training_csv
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_temporal_project_from_dataset,
)


def _mar_rows(n: int = 20) -> str:
    lines = ["fecha,altura_ola,temperatura"]
    for d in range(1, n + 1):
        lines.append(f"2024-01-{d:02d},{2.0 + d * 0.1:.2f},{15.0 + d * 0.05:.2f}")
    return "\n".join(lines) + "\n"


class TestCanonicalMarExample:
    def test_generates_regression_project_with_lag_and_shift(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        assert res["ok"]
        assert "altura_ola_lag1: Scalar" in res["mxai"]
        assert "altura_ola_lag2: Scalar" in res["mxai"]
        assert "altura_ola: Scalar" in res["mxai"]  # el valor presente sigue siendo feature legítima
        assert "OUTPUT predicted_value" in res["mxai"]
        assert "fecha" not in res["mxai"]  # la columna temporal cruda nunca es feature

    def test_prepared_csv_validates_against_generated_model(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        v = _validate_training_csv(
            res["mxai"], res["training_text"], res["csv_text"],
            field_ranges=res["field_ranges"],
        )
        assert v.get("ok"), v.get("error")

    def test_rows_dropped_for_lag_and_horizon_edges(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(20), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        # 20 filas - 2 (sin historia para lag) - 1 (sin futuro para el horizonte) = 17
        assert len(res["csv_text"].splitlines()) - 1 == 17

    def test_without_lag_window_still_works(self):
        """horizon solo, sin lag_window — caso degenerado válido (predecir
        con features exógenas del propio día, sin autorregresión)."""
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
        )
        assert res["ok"]
        assert "temperatura: Scalar" in res["mxai"]
        v = _validate_training_csv(
            res["mxai"], res["training_text"], res["csv_text"],
            field_ranges=res["field_ranges"],
        )
        assert v.get("ok"), v.get("error")

    def test_provenance_records_temporal_config_and_pipeline_operations(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        temporal = res["provenance"]["temporal"]
        assert temporal["temporal_column"] == "fecha"
        assert temporal["raw_target_column"] == "altura_ola"
        assert temporal["horizon"] == 1
        assert temporal["lag_window_columns"] == ["altura_ola"]
        assert temporal["lag_window_size"] == 2
        ops = [step["operation"] for step in temporal["pipeline_operations"]]
        assert ops == ["sort_temporal", "shift_target", "lag_window", "missing_values"]

    def test_larger_horizon_and_window_both_respected(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(30), target_column="altura_ola",
            temporal_column="fecha", horizon=3,
            lag_window_columns=["altura_ola"], lag_window_size=5,
        )
        # 30 - 5 (lag) - 3 (horizon) = 22
        assert len(res["csv_text"].splitlines()) - 1 == 22


class TestMultipleLagColumns:
    def test_lag_window_applies_to_every_declared_column(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola", "temperatura"], lag_window_size=1,
        )
        assert "altura_ola_lag1: Scalar" in res["mxai"]
        assert "temperatura_lag1: Scalar" in res["mxai"]


class TestActionableErrors:
    def test_unknown_temporal_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe"):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola",
                temporal_column="no_existe", horizon=1,
            )

    def test_unknown_target_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe"):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="no_existe",
                temporal_column="fecha", horizon=1,
            )

    def test_horizon_zero_raises(self):
        with pytest.raises(DatasetProjectError, match="horizon"):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola",
                temporal_column="fecha", horizon=0,
            )

    def test_lag_window_columns_without_size_raises(self):
        with pytest.raises(DatasetProjectError, match="lag_window_size"):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola",
                temporal_column="fecha", horizon=1,
                lag_window_columns=["altura_ola"],
            )

    def test_window_larger_than_dataset_leaves_no_rows(self):
        with pytest.raises(DatasetProjectError, match="ninguna fila"):
            generate_temporal_project_from_dataset(
                _mar_rows(5), target_column="altura_ola",
                temporal_column="fecha", horizon=1,
                lag_window_columns=["altura_ola"], lag_window_size=10,
            )

    def test_unknown_lag_window_column_raises(self):
        with pytest.raises(DatasetProjectError, match="no existe"):
            generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola",
                temporal_column="fecha", horizon=1,
                lag_window_columns=["no_existe"], lag_window_size=1,
            )


class TestAntiLeakageDefenseInDepth:
    def test_lagging_the_shifted_target_with_residual_future_offset_raises(self):
        """Misuso deliberado: lag_window sobre la propia columna target ya
        desplazada (horizon=3) con window=1 deja un offset residual +2 —
        sigue siendo fuga, y la defensa en profundidad de
        generate_temporal_project_from_dataset la atrapa aunque nada en el
        flujo normal la produciría por accidente."""
        with pytest.raises(DatasetProjectError, match="fuga de información temporal"):
            generate_temporal_project_from_dataset(
                _mar_rows(30), target_column="altura_ola",
                temporal_column="fecha", horizon=3,
                lag_window_columns=["altura_ola_target_h3"], lag_window_size=1,
            )

    def test_lagging_the_shifted_target_by_exactly_the_horizon_is_safe(self):
        """lag_window(window=1) sobre una columna desplazada horizon=1
        reconstruye el valor PRESENTE (offset neto 1-1=0) — no es fuga, es
        aritmética de linaje correcta. (Con window>1 el resto de lags
        intermedios SÍ seguirían filtrando — ver el test anterior.)"""
        res = generate_temporal_project_from_dataset(
            _mar_rows(30), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola_target_h1"], lag_window_size=1,
        )
        assert res["ok"]


class TestUserOverridesApplyToTransformedColumns:
    def test_type_override_targets_lag_column_name(self):
        """Los overrides se aplican DESPUÉS del pipeline — sus claves son
        los nombres TRANSFORMADOS (invariante 8: el usuario corrige lo que
        VE en el esquema final, que ya incluye los _lag*)."""
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=1,
            column_range_overrides={"altura_ola_lag1": (0.0, 100.0)},
        )
        assert tuple(res["field_ranges"]["altura_ola_lag1"]) == (0.0, 100.0)
