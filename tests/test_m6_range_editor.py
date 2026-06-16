"""M6-Fase 1 — Editor de rangos pre-generación.

El usuario puede fijar el rango de dominio de cada campo antes de generar el
dataset. Precedencia: usuario > LLM > default (0-1). El LLM solo se consulta
para los campos que el usuario dejó vacíos ("solo rellena huecos").
"""
from unittest.mock import patch

MXAI = (
    "PROJECT LoanProject\n\n"
    "VECTOR LoanApplicant[2]\n"
    "  credit_score: Scalar\n"
    "  age: Scalar\n"
    "END\n\n"
    "NETWORK LoanDefaultClassifier\n"
    "  INPUT LoanApplicant\n"
    "  LAYER Dense units=32 activation=relu\n"
    "  LAYER Dense units=1 activation=sigmoid\n"
    "  OUTPUT predicted_prob: Probability\n"
    "END\n\n"
    "GRAPH\n  LoanApplicant -> LoanDefaultClassifier\nEND\n"
)

MXTRAIN = (
    "MODEL LoanProject.mxai\n\n"
    "DATASET LoanDataset\n"
    "  SOURCE csv(\"loan.train.csv\")\n"
    "  INPUT LoanApplicant FROM COLUMNS [credit_score, age]\n"
    "  TARGET predicted_prob: Probability\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n"
    "  BATCH size=8\n"
    "END\n\n"
    "LOSS LoanLoss\n"
    "  TYPE binary_cross_entropy\n"
    "  PREDICTION LoanDefaultClassifier\n"
    "  TARGET predicted_prob\n"
    "END\n\n"
    "OPTIMIZER LoanOptimizer\n"
    "  TYPE sgd\n"
    "  LEARNING_RATE 0.01\n"
    "  UPDATE LoanDefaultClassifier.*\n"
    "END\n\n"
    "RUN\n  EPOCHS 3\nEND\n"
)


def _generate(**kwargs):
    from matrixai.playground import _generate_synthetic_dataset
    defaults = dict(rows=10, seed=42, mode="random", use_llm=False)
    defaults.update(kwargs)
    return _generate_synthetic_dataset(MXAI, MXTRAIN, **defaults)


class TestSuggestRanges:
    def test_returns_columns_and_llm_ranges(self):
        from matrixai.playground import _suggest_field_ranges
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch("matrixai.playground._llm_field_ranges",
                       return_value={"credit_score": (300.0, 850.0)}):
                r = _suggest_field_ranges(MXAI, MXTRAIN)
        assert r["ok"], r
        assert r["columns"] == ["credit_score", "age"]
        assert r["field_ranges"] == {"credit_score": [300.0, 850.0]}
        assert r["llm_ranges_used"] is True

    def test_without_llm_returns_columns_with_empty_ranges(self):
        from matrixai.playground import _suggest_field_ranges
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": False}):
            r = _suggest_field_ranges(MXAI, MXTRAIN)
        assert r["ok"], r
        assert r["columns"] == ["credit_score", "age"]
        assert r["field_ranges"] == {}
        assert r["llm_ranges_used"] is False

    def test_llm_extra_columns_filtered(self):
        # El LLM puede alucinar campos que no existen: nunca llegan al editor
        from matrixai.playground import _suggest_field_ranges
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch("matrixai.playground._llm_field_ranges",
                       return_value={"credit_score": (300.0, 850.0), "ghost": (0.0, 9.0)}):
                r = _suggest_field_ranges(MXAI, MXTRAIN)
        assert "ghost" not in r["field_ranges"]

    def test_invalid_mxai_fails(self):
        from matrixai.playground import _suggest_field_ranges
        r = _suggest_field_ranges("NOT A MODEL", MXTRAIN)
        assert not r["ok"]

    def test_empty_inputs_fail(self):
        from matrixai.playground import _suggest_field_ranges
        assert not _suggest_field_ranges("", MXTRAIN)["ok"]
        assert not _suggest_field_ranges(MXAI, "")["ok"]


class TestGenerateWithOverride:
    def test_full_override_skips_llm(self):
        # Override completo → cero llamadas al LLM aunque esté activo
        from matrixai import playground
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch.object(playground, "_llm_field_ranges",
                              side_effect=AssertionError("no debe llamarse")) as llm:
                r = _generate(use_llm=True, field_ranges_override={
                    "credit_score": (300.0, 850.0), "age": (18.0, 90.0),
                })
        assert r["ok"], r
        llm.assert_not_called()
        assert r["field_ranges"] == {"credit_score": [300.0, 850.0], "age": [18.0, 90.0]}
        # CSV en escala de dominio según el override
        first = r["csv_text"].strip().splitlines()[1].split(",")
        assert float(first[0]) >= 300.0

    def test_partial_override_llm_fills_gaps_only(self):
        from matrixai.playground import _generate_synthetic_dataset
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch("matrixai.playground._llm_field_ranges",
                       return_value={"age": (18.0, 90.0)}) as llm:
                r = _generate_synthetic_dataset(
                    MXAI, MXTRAIN, 10, 42, "random", use_llm=True,
                    field_ranges_override={"credit_score": (500.0, 600.0)},
                )
        assert r["ok"], r
        # el LLM solo recibió los huecos
        assert llm.call_args[0][0] == ["age"]
        assert r["field_ranges"] == {"credit_score": [500.0, 600.0], "age": [18.0, 90.0]}
        assert r["llm_ranges_used"] is True

    def test_user_wins_even_if_llm_returns_same_column(self):
        # Aunque el LLM devuelva un rango para un campo cubierto, gana el usuario
        from matrixai.playground import _generate_synthetic_dataset
        with patch("matrixai.playground._detect_llm_mode", return_value={"active": True}):
            with patch("matrixai.playground._llm_field_ranges",
                       return_value={"credit_score": (0.0, 9999.0), "age": (18.0, 90.0)}):
                r = _generate_synthetic_dataset(
                    MXAI, MXTRAIN, 10, 42, "random", use_llm=True,
                    field_ranges_override={"credit_score": (500.0, 600.0)},
                )
        assert r["field_ranges"]["credit_score"] == [500.0, 600.0]

    def test_override_without_llm(self):
        r = _generate(field_ranges_override={"age": (18.0, 90.0)})
        assert r["ok"], r
        assert r["field_ranges"] == {"age": [18.0, 90.0]}
        assert r["llm_ranges_used"] is False
        # columna sin rango sigue en 0-1
        idx = r["columns"].index("credit_score")
        first = r["csv_text"].strip().splitlines()[1].split(",")
        assert 0.0 <= float(first[idx]) <= 1.0

    def test_override_unknown_column_filtered(self):
        r = _generate(field_ranges_override={"ghost": (0.0, 9.0)})
        assert r["ok"], r
        assert r["field_ranges"] == {}

    def test_no_override_keeps_previous_behaviour(self):
        r = _generate()
        assert r["ok"], r
        assert r["field_ranges"] == {}
        first = r["csv_text"].strip().splitlines()[1].split(",")
        assert 0.0 <= float(first[0]) <= 1.0
