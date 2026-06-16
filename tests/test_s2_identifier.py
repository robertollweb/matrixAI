"""S2-C4 — Identificadores excluidos del entrenamiento.

Una columna identificadora (patient_id) no debe ser feature: cero señal y
riesgo de memorización. Se elimina del VECTOR (campos + aridad) y del FROM
COLUMNS, y el dataset sintético ya no la genera. Nunca se elimina la última
columna (un modelo necesita al menos una entrada).
"""
import time

MXAI = (
    "PROJECT LoanProject\n\n"
    "VECTOR LoanApplicant[3]\n"
    "  patient_id: Scalar\n"
    "  credit_score: Scalar\n"
    "  age: Scalar\n"
    "END\n\n"
    "NETWORK LoanDefaultClassifier\n"
    "  INPUT LoanApplicant\n"
    "  LAYER Dense units=8 activation=relu\n"
    "  LAYER Dense units=1 activation=sigmoid\n"
    "  OUTPUT predicted_prob: Probability\n"
    "END\n\n"
    "GRAPH\n  LoanApplicant -> LoanDefaultClassifier\nEND\n"
)

MXTRAIN = (
    "MODEL LoanProject.mxai\n\n"
    "DATASET LoanDataset\n"
    "  SOURCE csv(\"loan.train.csv\")\n"
    "  INPUT LoanApplicant FROM COLUMNS [patient_id, credit_score, age]\n"
    "  TARGET predicted_prob: Probability\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n"
    "  BATCH size=8\n"
    "END\n\n"
    "LOSS LoanLoss\n  TYPE binary_cross_entropy\n  PREDICTION LoanDefaultClassifier\n"
    "  TARGET predicted_prob\nEND\n\n"
    "OPTIMIZER LoanOptimizer\n  TYPE sgd\n  LEARNING_RATE 0.01\n"
    "  UPDATE LoanDefaultClassifier.*\nEND\n\n"
    "RUN\n  EPOCHS 3\nEND\n"
)


class TestExclude:
    def test_removes_field_and_fixes_arity(self):
        from matrixai.training.categorical import exclude_identifiers
        mxai, training, excluded = exclude_identifiers(MXAI, MXTRAIN, ["patient_id"])
        assert excluded == ["patient_id"]
        assert "VECTOR LoanApplicant[2]" in mxai
        assert "patient_id" not in mxai
        assert "  credit_score: Scalar" in mxai
        # FROM COLUMNS no longer lists the identifier
        line = [l for l in training.splitlines() if "FROM COLUMNS" in l][0]
        assert "patient_id" not in line
        assert "credit_score" in line and "age" in line

    def test_parses_after_exclusion(self):
        from matrixai.training.categorical import exclude_identifiers
        from matrixai.parser import parse_text
        from matrixai.training.parser import parse_training_text
        mxai, training, _ = exclude_identifiers(MXAI, MXTRAIN, ["patient_id"])
        program = parse_text(mxai)
        spec = parse_training_text(training)
        assert len(program.vectors[0].fields) == 2
        assert list(spec.dataset.input.columns) == ["credit_score", "age"]

    def test_noop_without_identifiers(self):
        from matrixai.training.categorical import exclude_identifiers
        mxai, training, excluded = exclude_identifiers(MXAI, MXTRAIN, None)
        assert excluded == []
        assert mxai == MXAI and training == MXTRAIN

    def test_unknown_column_ignored(self):
        from matrixai.training.categorical import exclude_identifiers
        mxai, _, excluded = exclude_identifiers(MXAI, MXTRAIN, ["ghost"])
        assert excluded == []
        assert mxai == MXAI

    def test_never_removes_last_feature(self):
        from matrixai.training.categorical import exclude_identifiers
        mxai, _, excluded = exclude_identifiers(
            MXAI, MXTRAIN, ["patient_id", "credit_score", "age"],
        )
        assert excluded == []  # would leave 0 inputs → refused
        assert mxai == MXAI


class TestCoerce:
    def test_coerce_dedupes_and_filters(self):
        from matrixai.playground import _coerce_field_identifiers
        assert _coerce_field_identifiers(["a", "a", " ", "b"]) == ["a", "b"]
        assert _coerce_field_identifiers("nope") == []
        assert _coerce_field_identifiers(None) == []


class TestDatasetEndpoint:
    def test_excludes_identifier_from_csv_and_returns_reduced_model(self):
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(MXAI, MXTRAIN, 10, 42, "random", use_llm=False,
                                        field_identifiers=["patient_id"])
        assert r["ok"], r.get("error")
        assert r["excluded_identifiers"] == ["patient_id"]
        assert "patient_id" not in r["columns"]
        assert "VECTOR LoanApplicant[2]" in r["mxai_text"]
        header = r["csv_text"].strip().splitlines()[0]
        assert "patient_id" not in header
        assert "credit_score" in header

    def test_no_identifiers_no_model_change(self):
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(MXAI, MXTRAIN, 10, 42, "random", use_llm=False)
        assert r["ok"], r.get("error")
        assert r["excluded_identifiers"] == []
        assert "mxai_text" not in r  # model unchanged → not returned

    def test_identifier_and_category_together(self):
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(
            MXAI, MXTRAIN, 10, 42, "random", use_llm=False,
            field_identifiers=["patient_id"],
            field_categories={"age": ["joven", "adulto", "mayor"]},
        )
        assert r["ok"], r.get("error")
        assert r["excluded_identifiers"] == ["patient_id"]
        assert r["one_hot_groups"]["age"] == ["age__joven", "age__adulto", "age__mayor"]
        cols = r["columns"]
        assert "patient_id" not in cols
        assert "age__joven" in cols and "credit_score" in cols

    def test_train_on_reduced_model(self):
        from matrixai.playground import _generate_synthetic_dataset, _submit_training_job, _get_job_status
        r = _generate_synthetic_dataset(MXAI, MXTRAIN, 40, 42, "coherent", use_llm=False,
                                        field_identifiers=["patient_id"])
        assert r["ok"], r.get("error")
        job = _submit_training_job(r["mxai_text"], r["training_text"], r["csv_text"], epochs_override=2)
        assert job.get("ok"), job
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert st.get("params_best") is not None
