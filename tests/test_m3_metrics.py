"""M3 — Exponer macro_f1 / confusion_matrix en el resultado de entrenamiento.

dense_evaluator ya calculaba estas métricas; el hueco era que el evaluador
genérico falla con modelos NETWORK y el resultado llegaba sin evaluation_report.
Ahora _collect_training_result cae a DenseSupervisedEvaluator y aplana
macro_f1/confusion_matrix/labels/per_label para el cliente.
"""
import time

MULTI_MXAI = (
    "PROJECT LoanProject\n\n"
    "VECTOR LoanApplicant[2]\n  credit_score: Scalar\n  age: Scalar\nEND\n\n"
    "NETWORK LoanRiskClassifier\n"
    "  INPUT LoanApplicant\n"
    "  LAYER Dense units=16 activation=relu\n"
    "  LAYER Dense units=3 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[low, mid, high]\n"
    "END\n\n"
    "GRAPH\n  LoanApplicant -> LoanRiskClassifier\nEND\n"
)

MULTI_MXTRAIN = (
    "MODEL LoanProject.mxai\n\n"
    "DATASET LoanDataset\n"
    "  SOURCE csv(\"loan.train.csv\")\n"
    "  INPUT LoanApplicant FROM COLUMNS [credit_score, age]\n"
    "  TARGET predicted_class: Label[low, mid, high]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n"
    "  BATCH size=8\n"
    "END\n\n"
    "LOSS LoanLoss\n  TYPE cross_entropy\n  PREDICTION LoanRiskClassifier\n"
    "  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER LoanOptimizer\n  TYPE sgd\n  LEARNING_RATE 0.05\n"
    "  UPDATE LoanRiskClassifier.*\nEND\n\n"
    "RUN\n  EPOCHS 4\nEND\n"
)


def _multi_csv(rows: int = 60) -> str:
    lines = ["credit_score,age,predicted_class"]
    for i in range(rows):
        s = (i % 10) / 10.0
        label = "low" if s < 0.33 else ("mid" if s < 0.66 else "high")
        lines.append(f"{s},{(i % 7) / 7.0:.3f},{label}")
    return "\n".join(lines) + "\n"


class TestSyncTrainingMetrics:
    def test_multiclass_exposes_macro_f1_and_confusion(self):
        from matrixai.playground import _run_playground_training
        r = _run_playground_training(MULTI_MXAI, MULTI_MXTRAIN, _multi_csv(), epochs_override=4)
        assert r["ok"], r.get("error")
        assert isinstance(r["macro_f1"], float)
        assert 0.0 <= r["macro_f1"] <= 1.0
        assert r["labels"] == ["low", "mid", "high"]
        cm = r["confusion_matrix"]
        assert set(cm.keys()) == {"low", "mid", "high"}
        assert set(cm["low"].keys()) == {"low", "mid", "high"}
        # la matriz cubre todas las filas del dataset
        total = sum(sum(row.values()) for row in cm.values())
        assert total == 60
        # per_label con precision/recall/f1 por clase
        assert set(r["per_label"]["mid"].keys()) == {"precision", "recall", "f1"}

    def test_metrics_also_inside_evaluation_report(self):
        from matrixai.playground import _run_playground_training
        r = _run_playground_training(MULTI_MXAI, MULTI_MXTRAIN, _multi_csv(), epochs_override=4)
        assert r["ok"], r.get("error")
        ev = r["evaluation_report"]
        assert ev is not None
        assert ev["macro_f1"] == r["macro_f1"]
        assert ev["confusion_matrix"] == r["confusion_matrix"]

    def test_binary_classification_has_metrics(self):
        from matrixai.playground import _run_playground_training
        mxai = (
            "PROJECT BinProject\n\nVECTOR Applicant[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
            "NETWORK BinClassifier\n  INPUT Applicant\n"
            "  LAYER Dense units=8 activation=relu\n"
            "  LAYER Dense units=1 activation=sigmoid\n"
            "  OUTPUT predicted_prob: Probability\nEND\n\n"
            "GRAPH\n  Applicant -> BinClassifier\nEND\n"
        )
        mxtrain = (
            "MODEL BinProject.mxai\n\nDATASET BinDataset\n"
            "  SOURCE csv(\"bin.train.csv\")\n"
            "  INPUT Applicant FROM COLUMNS [a, b]\n"
            "  TARGET predicted_prob: Probability\n"
            "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
            "LOSS BinLoss\n  TYPE binary_cross_entropy\n  PREDICTION BinClassifier\n"
            "  TARGET predicted_prob\nEND\n\n"
            "OPTIMIZER BinOpt\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE BinClassifier.*\nEND\n\n"
            "RUN\n  EPOCHS 4\nEND\n"
        )
        lines = ["a,b,predicted_prob"]
        for i in range(40):
            x = (i % 10) / 10.0
            lines.append(f"{x},{(i % 5) / 5.0:.2f},{1.0 if x < 0.5 else 0.0}")
        r = _run_playground_training(mxai, mxtrain, "\n".join(lines) + "\n", epochs_override=4)
        assert r["ok"], r.get("error")
        assert isinstance(r["macro_f1"], float)
        assert r["confusion_matrix"] is not None


class TestJobStatusMetrics:
    def test_async_job_status_flattens_metrics(self):
        from matrixai.playground import _submit_training_job, _get_job_status
        job = _submit_training_job(MULTI_MXAI, MULTI_MXTRAIN, _multi_csv(), epochs_override=3)
        assert job.get("ok"), job
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert isinstance(st.get("macro_f1"), float)
        assert st.get("labels") == ["low", "mid", "high"]
        assert st.get("confusion_matrix") is not None
