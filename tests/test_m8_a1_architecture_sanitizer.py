"""M8-A1 — Sanitizador de arquitectura (suelo de ancho para capas ReLU).

Un cuello ReLU estrecho antes del softmax (Dense(n_clases, relu) →
Dense(n_clases, softmax)) es una trampa dying-ReLU que colapsa el modelo a un
predictor constante. El sanitizador ensancha esas capas a un mínimo seguro,
gobernando TODA fuente (default / prompt / LLM), con nota auditable.
"""
import time
from unittest.mock import patch

from matrixai.training.dense_generator import sanitize_hidden_layers, DenseNetworkGenerator, _MIN_RELU_WIDTH


class TestSanitizeUnit:
    def test_widens_narrow_relu(self):
        out, notes = sanitize_hidden_layers([(64, "relu"), (4, "relu")])
        assert out == [(64, "relu"), (16, "relu")]
        assert len(notes) == 1 and "4" in notes[0] and "16" in notes[0]

    def test_keeps_wide_relu(self):
        out, notes = sanitize_hidden_layers([(64, "relu"), (32, "relu")])
        assert out == [(64, "relu"), (32, "relu")]
        assert notes == []

    def test_non_relu_untouched(self):
        # a narrow non-ReLU layer is not a dying-ReLU trap
        out, notes = sanitize_hidden_layers([(1, "sigmoid"), (2, "tanh")])
        assert out == [(1, "sigmoid"), (2, "tanh")]
        assert notes == []

    def test_multiple_narrow(self):
        out, notes = sanitize_hidden_layers([(3, "relu"), (8, "relu"), (64, "relu")])
        assert out == [(_MIN_RELU_WIDTH, "relu"), (_MIN_RELU_WIDTH, "relu"), (64, "relu")]
        assert len(notes) == 2

    def test_empty(self):
        assert sanitize_hidden_layers([]) == ([], [])


class TestGeneratorSanitizes:
    def test_narrow_bottleneck_widened_in_mxai(self):
        gen = DenseNetworkGenerator().generate(
            "clasificar en 3 niveles bajo medio alto",
            input_fields=["f1", "f2", "f3", "f4", "f5", "f6"],
            labels=["BAJO", "MEDIO", "ALTO"],
            hidden_layers=[(64, "relu"), (4, "relu")],  # bottleneck before softmax
        )
        import re
        narrow = [
            l for l in gen.mxai_text.splitlines()
            if (m := re.search(r"units=(\d+) activation=relu", l)) and int(m.group(1)) < _MIN_RELU_WIDTH
        ]
        assert narrow == [], gen.mxai_text
        assert any("cuello ReLU" in w for w in gen.warnings)

    def test_clean_architecture_no_warning(self):
        gen = DenseNetworkGenerator().generate(
            "clasificar spam",
            input_fields=["a", "b"], labels=["spam", "ham"],
            hidden_layers=[(32, "relu"), (16, "relu")],
        )
        assert not any("cuello ReLU" in w for w in gen.warnings)


class TestEndToEndViaLLMSchema:
    def test_llm_proposed_bottleneck_is_sanitized_and_warned(self):
        from matrixai.playground import analyze_playground_request
        # simulate an LLM proposing a narrow ReLU bottleneck
        schema = {
            "input_fields": ["f1", "f2", "f3", "f4", "f5", "f6"],
            "labels": ["BAJO", "MEDIO", "ALTO"],
            "hidden_layers": [(64, "relu"), (4, "relu")],
        }
        with patch("matrixai.playground._dense_llm_schema", return_value=schema):
            r = analyze_playground_request({
                "mode": "prompt",
                "prompt": "clasificar riesgo en 3 niveles bajo medio alto",
                "use_llm": True,
            })
        import re
        mxai = r["mxai"]
        narrow = [
            l for l in mxai.splitlines()
            if (m := re.search(r"units=(\d+) activation=relu", l)) and int(m.group(1)) < _MIN_RELU_WIDTH
        ]
        assert narrow == [], mxai
        # the sanitizer note is surfaced in the generator pipeline stage
        warns = [w for s in r.get("pipeline_stages", []) for w in (s.get("warnings") or [])]
        assert any("cuello ReLU" in w for w in warns)

    def test_sanitized_model_trains(self):
        from matrixai.playground import analyze_playground_request, _run_playground_training, _probe_model_collapse
        schema = {
            "input_fields": ["f1", "f2", "f3", "f4"],
            "labels": ["BAJO", "MEDIO", "ALTO"],
            "hidden_layers": [(32, "relu"), (3, "relu")],  # bottleneck = n_classes
        }
        with patch("matrixai.playground._dense_llm_schema", return_value=schema):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar en 3 niveles bajo medio alto",
                "use_llm": True,
            })
        mxai, training = r["mxai"], r["training_text"]
        # learnable signal: mean of features decides the class
        rows = ["f1,f2,f3,f4,predicted_class"]
        labels = ["BAJO", "MEDIO", "ALTO"]
        for i in range(60):
            vals = [(i % 10) / 10.0, (i % 7) / 7.0, (i % 5) / 5.0, (i % 3) / 3.0]
            s = sum(vals) / 4
            lab = labels[0] if s < 0.4 else (labels[1] if s < 0.6 else labels[2])
            rows.append(",".join(f"{v:.3f}" for v in vals) + f",{lab}")
        csv_text = "\n".join(rows) + "\n"
        tr = _run_playground_training(mxai, training, csv_text, epochs_override=6)
        assert tr["ok"], tr.get("error")
        probe = _probe_model_collapse(mxai, tr["params_best"])
        assert probe is not None and probe["collapsed"] is False
