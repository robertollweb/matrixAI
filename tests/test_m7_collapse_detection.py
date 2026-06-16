"""M7 — Detección de colapso post-entrenamiento (predictor constante).

Un modelo con dying ReLU (caso real: cuello Dense(4,relu) pre-softmax) entrena
"con éxito" pero devuelve softmax(bias) para cualquier entrada. El sondeo lanza
4 entradas deterministas y marca model_collapsed si la salida no varía.
"""
import time
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


def _initial_params() -> dict:
    from matrixai.parameters.store import build_initial_parameter_set
    from matrixai.parser import parse_text
    return build_initial_parameter_set(parse_text(MXAI)).to_dict()


def _zeroed(values):
    if isinstance(values, list):
        return [_zeroed(v) for v in values]
    return 0.0


def _collapsed_params() -> dict:
    # Pesos y biases a cero → cada capa emite una constante → sigmoid(0) = 0.5
    # para cualquier entrada: el predictor constante perfecto.
    params = _initial_params()
    for name, tensor in params["parameters"].items():
        if isinstance(tensor, dict) and "values" in tensor:
            tensor["values"] = _zeroed(tensor["values"])
        else:
            params["parameters"][name] = _zeroed(tensor)
    return params


class TestProbe:
    def test_collapsed_params_detected(self):
        from matrixai.playground import _probe_model_collapse
        r = _probe_model_collapse(MXAI, _collapsed_params())
        assert r is not None
        assert r["collapsed"] is True
        assert r["constant_output"] == [0.5]

    def test_healthy_params_not_flagged(self):
        from matrixai.playground import _probe_model_collapse
        r = _probe_model_collapse(MXAI, _initial_params())
        assert r is not None
        assert r["collapsed"] is False
        assert "constant_output" not in r

    def test_invalid_params_returns_none(self):
        from matrixai.playground import _probe_model_collapse
        assert _probe_model_collapse(MXAI, {"basura": True}) is None

    def test_model_without_network_returns_none(self):
        from matrixai.playground import _probe_model_collapse
        graph_only = (
            "PROJECT Flow\n\nVECTOR Input[1]\n  x: Scalar\nEND\n\n"
            "GRAPH\n  Input -> Input\nEND\n"
        )
        assert _probe_model_collapse(graph_only, _initial_params()) is None

    def test_unparseable_mxai_returns_none(self):
        from matrixai.playground import _probe_model_collapse
        assert _probe_model_collapse("NOT A MODEL", _initial_params()) is None


class TestAttach:
    def test_collapsed_result_gets_flag_and_warning(self):
        from matrixai.playground import _attach_collapse_check
        result = {"ok": True, "params_best": _collapsed_params()}
        out = _attach_collapse_check(result, MXAI)
        assert out["model_collapsed"] is True
        assert "constante" in out["collapse_warning"]
        assert out["collapse_constant_output"] == [0.5]

    def test_healthy_result_flag_false_without_warning(self):
        from matrixai.playground import _attach_collapse_check
        out = _attach_collapse_check({"ok": True, "params_best": _initial_params()}, MXAI)
        assert out["model_collapsed"] is False
        assert "collapse_warning" not in out

    def test_failed_result_untouched(self):
        from matrixai.playground import _attach_collapse_check
        result = {"ok": False, "error": "boom"}
        assert _attach_collapse_check(result, MXAI) == {"ok": False, "error": "boom"}

    def test_result_without_params_untouched(self):
        from matrixai.playground import _attach_collapse_check
        result = {"ok": True}
        assert _attach_collapse_check(result, MXAI) == {"ok": True}

    def test_probe_failure_never_breaks_result(self):
        from matrixai.playground import _attach_collapse_check
        with patch("matrixai.playground._probe_model_collapse", return_value=None):
            out = _attach_collapse_check({"ok": True, "params_best": {"x": 1}}, MXAI)
        assert "model_collapsed" not in out


class TestTrainingJobE2E:
    def _csv(self, rows=40):
        # señal aprendible: prob = 1 si credit_score < 0.5
        lines = ["credit_score,age,predicted_prob"]
        for i in range(rows):
            score = (i % 10) / 10.0
            lines.append(f"{score},{(i % 7) / 7.0:.4f},{1.0 if score < 0.5 else 0.0}")
        return "\n".join(lines) + "\n"

    def test_job_status_exposes_model_collapsed(self):
        from matrixai.playground import _submit_training_job, _get_job_status
        job = _submit_training_job(MXAI, MXTRAIN, self._csv(), epochs_override=2)
        assert job.get("ok"), job
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert st.get("model_collapsed") is False, st.get("model_collapsed")
