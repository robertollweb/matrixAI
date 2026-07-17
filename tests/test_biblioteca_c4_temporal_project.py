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

    def test_category_vocabulary_override_reaches_a_derived_lag(self):
        lines = ["fecha,altura_ola,regimen"]
        values = ["calma", "mar", "tormenta"]
        for day in range(1, 21):
            lines.append(
                f"2024-01-{day:02d},{2.0 + day * 0.1:.2f},{values[day % 3]}"
            )
        res = generate_temporal_project_from_dataset(
            "\n".join(lines) + "\n", target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["regimen"], lag_window_size=1,
            column_category_overrides={
                "regimen": ["tormenta", "mar", "calma", "desconocido"],
                "regimen_lag1": ["tormenta", "mar", "calma", "desconocido"],
            },
        )
        assert res["ok"]
        assert "regimen_lag1__desconocido" in res["csv_text"].splitlines()[0]
        assert res["provenance"]["column_category_overrides"]["regimen_lag1"][-1] == "desconocido"


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
        with pytest.raises(DatasetProjectError, match="fila"):
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


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 [ALTA] — el proyecto temporal entrenaba con split
# aleatorio: generate_temporal_project_from_dataset reutilizaba sin tocar
# el training_text de GEN, que SIEMPRE declara SPLIT ...seed=42 (nunca
# mode=temporal) — la protección de C3 quedaba anulada justo en el
# entrenamiento real.
# ---------------------------------------------------------------------------

class TestTrainingTextDeclaresTemporalSplit:
    def test_split_line_declares_mode_temporal(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
        )
        split_lines = [l for l in res["training_text"].splitlines() if l.strip().startswith("SPLIT")]
        assert len(split_lines) == 1
        assert "mode=temporal" in split_lines[0]

    def test_split_line_never_declares_seed(self):
        """mode=temporal no admite seed (parser.py lo rechazaría si se
        volviera a parsear) — GEN escribe seed=42 siempre, se elimina."""
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
        )
        split_lines = [l for l in res["training_text"].splitlines() if l.strip().startswith("SPLIT")]
        assert "seed=" not in split_lines[0]

    def test_split_ratio_preserved(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
        )
        split_line = next(l for l in res["training_text"].splitlines() if l.strip().startswith("SPLIT"))
        assert "train=0.8" in split_line
        assert "validation=0.2" in split_line

    def test_rewritten_training_text_still_parses(self):
        from matrixai.training.parser import parse_training_text
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
        )
        spec = parse_training_text(res["training_text"])
        assert spec.dataset.split.mode == "temporal"
        assert spec.dataset.split.seed is None

    def test_training_actually_uses_temporal_split_no_shuffle(self):
        """Cierra el círculo hasta el trainer REAL (no solo el texto): con
        el mismo mecanismo que la reauditoría de C3 (espía sobre
        evaluate_dense_network), confirma que la validación es EXACTAMENTE
        el último tramo, sin barajar, en el camino stdlib real."""
        import matrixai.training.dense_trainer as dt
        from matrixai.training.parser import parse_training_text
        from matrixai.training.dense_generator import DenseNetworkGenerator
        import tempfile
        from pathlib import Path

        # x1 = índice de fila (0..N-1) tras el pipeline — identifica
        # exactamente qué filas terminan en validación. DECIMAL (i + 0.5),
        # no entero: un entero casi-todo-distinto dispara la heurística de
        # identificador de C1 (ver docstring de dataset_analysis.py) y x1
        # se excluiría como feature — ya ha pasado antes en este mismo
        # contrato, es un escollo recurrente al escribir CSVs de prueba.
        # y NO es secuencial-única (i % 5) por el mismo motivo.
        rows = ["fecha,x1,y"]
        for i in range(20):
            rows.append(f"2024-01-{i + 1:02d},{i + 0.5},{i % 5}")
        csv_text = "\n".join(rows) + "\n"
        res = generate_temporal_project_from_dataset(
            csv_text, target_column="y", temporal_column="fecha", horizon=1,
        )

        captured: dict = {}
        real_evaluate = dt.evaluate_dense_network

        def _spy(net, ps, examples, loss_fn, labels=None):
            captured["val_x1"] = sorted(x[0] for x, _y in examples)
            return real_evaluate(net, ps, examples, loss_fn, labels=labels)

        import unittest.mock
        with unittest.mock.patch.object(dt, "evaluate_dense_network", _spy):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                training = parse_training_text(res["training_text"])
                mxai_path = tmp_path / training.model
                mxai_path.write_text(res["mxai"], encoding="utf-8")
                csv_path = tmp_path / training.dataset.source
                csv_path.write_text(res["csv_text"], encoding="utf-8")
                mxtrain_path = tmp_path / "train.mxtrain"
                mxtrain_path.write_text(res["training_text"], encoding="utf-8")
                dt.DenseSupervisedTrainer().train(
                    training, output_dir=str(tmp_path / "out"), base_path=tmp_path,
                    training_path=mxtrain_path,
                )
        # 20 filas - 1 (horizon=1 recorta la última, sin historia futura) =
        # 19; 0.8*19=15.2->15 de train, 4 de validación.
        val_x1 = captured["val_x1"]
        assert val_x1 == sorted(val_x1)  # en orden (el trainer denso stdlib
        # nunca baraja en NINGÚN modo — la garantía real de mode=temporal
        # aquí es la RATIO declarada, ver TestTrainingTextDeclaresTemporalSplit
        # de más arriba para la comprobación de que el training_text la fija).
        assert val_x1 == [15.5, 16.5, 17.5, 18.5]  # el ÚLTIMO tramo


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 [ALTA] — la procedencia perdía el hash/esquema del
# CSV original: generate_project_from_dataset trataba el CSV YA
# transformado por el pipeline como si fuera "el crudo".
# ---------------------------------------------------------------------------

class TestProvenanceReflectsOriginalCsv:
    def test_raw_csv_sha256_matches_the_csv_the_user_actually_uploaded(self):
        import hashlib
        csv_text = _mar_rows()
        res = generate_temporal_project_from_dataset(
            csv_text, target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        assert res["provenance"]["raw_csv_sha256"] == hashlib.sha256(csv_text.encode("utf-8")).hexdigest()

    def test_post_pipeline_hash_is_recorded_separately(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        assert res["provenance"]["post_pipeline_csv_sha256"] != res["provenance"]["raw_csv_sha256"]
        assert res["provenance"]["post_pipeline_csv_sha256"] != res["provenance"]["prepared_csv_sha256"]

    def test_schema_inferred_reflects_original_columns_not_lag_columns(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        schema_cols = set(res["provenance"]["schema_inferred"].keys())
        assert schema_cols == {"fecha", "altura_ola", "temperatura"}
        assert "altura_ola_lag1" not in schema_cols

    def test_operations_include_the_c3_pipeline_steps_first(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            lag_window_columns=["altura_ola"], lag_window_size=2,
        )
        ops = res["provenance"]["operations"]
        assert ops[:4] == ["sort_temporal", "shift_target", "lag_window", "missing_values"]


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 [MEDIA] — C4 no ejecutaba validate_pipeline_output
# (solo comprobaba "queda alguna fila", no min_rows real).
# ---------------------------------------------------------------------------

class TestMinRowsValidation:
    def test_exactly_one_surviving_row_is_rejected(self):
        rows = ["fecha,altura_ola"] + [f"2024-01-{d:02d},{d}.0" for d in range(1, 5)]
        csv_text = "\n".join(rows) + "\n"
        with pytest.raises(DatasetProjectError, match="al menos 2"):
            generate_temporal_project_from_dataset(
                csv_text, target_column="altura_ola",
                temporal_column="fecha", horizon=3,
            )


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 (ronda 2) [MEDIA] — provenance["seed"] se extraía
# del training_text ALEATORIO original (seed=42) ANTES de la reescritura a
# mode=temporal, contradiciendo el training_text realmente devuelto.
# ---------------------------------------------------------------------------

class TestProvenanceSeedMatchesTemporalSplit:
    def test_provenance_seed_is_none_when_split_is_temporal(self):
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
        )
        assert "seed=" not in res["training_text"]
        assert res["provenance"]["seed"] is None

    def test_non_temporal_case_keeps_the_real_gen_seed(self):
        """Retrocompat: el caso NO temporal (generate_project_from_dataset
        directo) nunca pasa por _force_temporal_split — sigue reportando el
        seed real de GEN, sin cambios de comportamiento."""
        from matrixai.training.dataset_project import generate_project_from_dataset
        res = generate_project_from_dataset(_mar_rows(), target_column="temperatura")
        assert res["provenance"]["seed"] == 42
        assert "seed=42" in res["training_text"]


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 (ronda 2) [ALTA] — una corrección de tipo/rango
# declarada sobre el target CRUDO (el único nombre que el editor de esquema
# conoce) no llegaba al target REALMENTE usado para entrenar
# (`{target}_target_hN`), que `generate_project_from_dataset` re-infiere por
# su cuenta y podía volver a clasificar mal (p.ej. como identificador).
# ---------------------------------------------------------------------------

class TestTargetOverridePropagatesToShiftedTarget:
    def test_type_override_on_raw_target_also_lands_on_shifted_target(self):
        # x1 secuencial casi-único dispara la heurística de identificador de
        # C1 (ver docstring de dataset_analysis.py) — tanto para la columna
        # cruda como, sin el fix, para el target desplazado que hereda los
        # mismos valores. Un identificador no tiene rango calculado por C1,
        # así que corregirlo a "number" exige TAMBIÉN un rango (igual que
        # en el editor real: tipo y rango se corrigen juntos).
        rows = ["fecha,x1,otro"]
        for i in range(20):
            rows.append(f"2024-01-{i + 1:02d},{i},{i % 3}.5")
        csv_text = "\n".join(rows) + "\n"
        res = generate_temporal_project_from_dataset(
            csv_text, target_column="x1", temporal_column="fecha", horizon=1,
            column_type_overrides={"x1": "number"},
            column_range_overrides={"x1": (0.0, 25.0)},
        )
        assert res["ok"]
        assert "OUTPUT predicted_value" in res["mxai"]
        assert "x1: Scalar" in res["mxai"]
        # La corrección de TIPO también se aplicó al x1 crudo, que sigue
        # existiendo como feature tras el shift (el rango NO se propaga al
        # target — ver test de column_range_overrides más abajo — pero sí
        # se aplica aquí, a la feature x1 tal cual se declaró).
        assert tuple(res["field_ranges"]["x1"]) == (0.0, 25.0)

    def test_without_the_fix_the_shifted_target_would_be_rejected_as_identifier(self):
        """Reproduce el fallo exacto que motivó el fix: sin propagar el
        override, el target desplazado se re-infiere con el mismo tipo que
        C1 le dio al crudo (identificador) y `generate_project_from_dataset`
        lo rechaza por no ser un target válido."""
        rows = ["fecha,x1,otro"]
        for i in range(20):
            rows.append(f"2024-01-{i + 1:02d},{i},{i % 3}.5")
        csv_text = "\n".join(rows) + "\n"
        with pytest.raises(DatasetProjectError, match="identificador"):
            generate_temporal_project_from_dataset(
                csv_text, target_column="x1", temporal_column="fecha", horizon=1,
                # Sin el override, C1 clasifica x1 como identificador tanto
                # crudo como desplazado — target inválido.
            )

    def test_range_override_on_the_target_is_never_propagated_it_is_a_no_op_everywhere(self):
        """`column_range_overrides` NO se propaga al target desplazado a
        propósito: ningún target (temporal o no, regresión o
        clasificación) declara rango en el prompt sintetizado (`SALIDA:
        nombre`, sin `en [lo, hi]`) — verificado que ni siquiera el caso NO
        temporal aplica un rango declarado sobre su propio target. La
        FEATURE homónima (altura_ola, que sigue existiendo tras el shift)
        sí lo recibe, sin cambios respecto a antes de esta ronda."""
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            column_range_overrides={"altura_ola": (0.0, 50.0)},
        )
        assert tuple(res["field_ranges"]["altura_ola"]) == (0.0, 50.0)
        assert "altura_ola_target_h1" not in res["field_ranges"]

    def test_override_on_a_different_column_is_not_propagated_to_target(self):
        """No debe haber propagación cruzada: un override sobre una columna
        que NO es el target (p.ej. una feature exógena) se queda tal cual."""
        res = generate_temporal_project_from_dataset(
            _mar_rows(), target_column="altura_ola",
            temporal_column="fecha", horizon=1,
            column_type_overrides={"temperatura": "number"},
        )
        assert res["ok"]
        assert "altura_ola_target_h1" not in (res["provenance"]["column_type_overrides"] or {})


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 (ronda 2) [MEDIA] — C3 documenta feature_columns/
# expected_types como opcionales A PROPÓSITO para que el caller que conoce
# el esquema final (C4) los declare; C4 no lo hacía.
# ---------------------------------------------------------------------------

class TestFinalValidationReceivesTheKnownSchema:
    def test_feature_columns_and_expected_types_reach_validate_pipeline_output(self):
        import unittest.mock
        import matrixai.training.dataset_pipeline as pipeline_mod

        captured: dict = {}
        real = pipeline_mod.validate_pipeline_output

        def _spy(rows, *, target_column, feature_columns=None, expected_types=None, min_rows=2):
            captured["feature_columns"] = feature_columns
            captured["expected_types"] = expected_types
            return real(
                rows, target_column=target_column,
                feature_columns=feature_columns, expected_types=expected_types,
                min_rows=min_rows,
            )

        with unittest.mock.patch.object(pipeline_mod, "validate_pipeline_output", _spy):
            res = generate_temporal_project_from_dataset(
                _mar_rows(), target_column="altura_ola",
                temporal_column="fecha", horizon=1,
                lag_window_columns=["altura_ola"], lag_window_size=2,
            )
        assert res["ok"]
        assert set(captured["feature_columns"]) == {
            "fecha", "altura_ola", "temperatura", "altura_ola_lag1", "altura_ola_lag2",
        }
        assert captured["expected_types"]["altura_ola"] == "number"
        assert captured["expected_types"]["altura_ola_target_h1"] == "number"
        assert captured["expected_types"]["temperatura"] == "number"
        assert captured["expected_types"]["altura_ola_lag1"] == "number"
        assert captured["expected_types"]["altura_ola_lag2"] == "number"
        assert "fecha" not in captured["expected_types"]  # tipo date, no verificable como numérico

    def test_a_column_forced_to_number_that_is_not_actually_numeric_is_rejected_early(self):
        """El valor real de expected_types: si el usuario fuerza a
        number/integer una columna que en el CSV real no lo es, se rechaza
        aquí con un mensaje específico, antes del rechazo genérico de GEN."""
        rows = ["fecha,altura_ola,etiqueta"]
        for i in range(20):
            rows.append(f"2024-01-{i + 1:02d},{2.0 + i * 0.1:.2f},texto_no_numerico")
        csv_text = "\n".join(rows) + "\n"
        with pytest.raises(DatasetProjectError, match="no numérico"):
            generate_temporal_project_from_dataset(
                csv_text, target_column="altura_ola",
                temporal_column="fecha", horizon=1,
                column_type_overrides={"etiqueta": "number"},
            )


# ---------------------------------------------------------------------------
# Reauditoría 2026-07-17 (ronda 2) [MEDIA] — el E2E existente
# (TestTrainingTextDeclaresTemporalSplit) termina en el entrenamiento; el
# contrato pide también evaluar y exportar/guardar. Sin navegador real
# disponible en este sandbox (ver memoria project_sandbox_no_browser), se
# cubre con las funciones backend REALES (no mocks) que el SPA/CLI usan por
# debajo: generar → entrenar → evaluar (mae/rmse/r2 reales, no espiados) →
# exportar un bundle descargable — el mismo camino que ejercitan
# test_gen_c6_roundtrip.py y test_transformer_c6_audit_hard_close.py para
# otros contratos, adaptado al flujo de datos temporal.
# ---------------------------------------------------------------------------

class TestFullCycleTrainEvaluateExport:
    def test_generate_train_evaluate_export_bundle(self):
        import tempfile
        from pathlib import Path

        from matrixai.export import create_edge_bundle
        from matrixai.parameters.store import load_parameter_set
        from matrixai.parser import parse_file
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        from matrixai.training.parser import parse_training_text

        # Mismo dataset/heurística que test_training_actually_uses_temporal_
        # split_no_shuffle (x1 decimal para no disparar la heurística de
        # identificador, y no-secuencial-única por el mismo motivo).
        rows = ["fecha,x1,y"]
        for i in range(20):
            rows.append(f"2024-01-{i + 1:02d},{i + 0.5},{i % 5}")
        csv_text = "\n".join(rows) + "\n"
        res = generate_temporal_project_from_dataset(
            csv_text, target_column="y", temporal_column="fecha", horizon=1,
        )
        assert res["ok"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            training = parse_training_text(res["training_text"])
            mxai_path = tmp_path / training.model
            mxai_path.write_text(res["mxai"], encoding="utf-8")
            csv_path = tmp_path / training.dataset.source
            csv_path.write_text(res["csv_text"], encoding="utf-8")
            mxtrain_path = tmp_path / "train.mxtrain"
            mxtrain_path.write_text(res["training_text"], encoding="utf-8")

            # Entrenar de verdad (stdlib DenseSupervisedTrainer, mismo
            # camino que el SPA usaría vía /api/train-start sin torch).
            out_dir = tmp_path / "out"
            run_result = DenseSupervisedTrainer().train(
                training, output_dir=str(out_dir), base_path=tmp_path,
                training_path=mxtrain_path,
            )

            # Evaluar: el propio trainer ya llama a evaluate_dense_network
            # de verdad sobre el split de validación temporal (último
            # tramo) — su resultado (r2 acotado a [0,1] para regresión) es
            # la prueba de evaluación real, no una mockeada.
            assert 0.0 <= run_result.accuracy <= 1.0
            assert run_result.final_validation_loss >= 0.0
            assert Path(run_result.artifacts["parameter_set"]).exists()

            # Exportar: bundle descargable real (mxai + params + onnx +
            # predict.py + manifest), con el esquema (rangos/tipos) que el
            # propio flujo de datos generó — mismo camino que expone
            # matrixaistudio para "descargar modelo".
            program = parse_file(mxai_path)
            ps = load_parameter_set(Path(run_result.artifacts["parameter_set"]))
            bundle_dir = tmp_path / "bundle"
            bundle_result = create_edge_bundle(
                program, ps, mxai_path=mxai_path,
                params_path=run_result.artifacts["parameter_set"],
                outdir=bundle_dir, validate=False,
                field_ranges={k: tuple(v) for k, v in (res.get("field_ranges") or {}).items()},
                field_types=res.get("field_types"),
                field_categories=res.get("field_categories"),
            )
            assert bundle_dir.exists()
            assert "model.mxai" in bundle_result.files
            assert "params.best.json" in bundle_result.files
            assert (bundle_dir / "model.mxai").exists()
            assert (bundle_dir / "params.best.json").exists()
