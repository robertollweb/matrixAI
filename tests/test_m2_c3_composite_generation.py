"""M2-C3 — Generación por prompt enruta a CompositeNetworkGenerator.

Un prompt con hints explícitos de arquitectura compuesta (residual/layernorm/
dropout/complejo) genera un modelo compuesto P19; los prompts neurales normales
siguen por el camino denso. Categóricas vía one-hot S2 (sin EMBEDDING nativo en
v1). Avisos explícitos para secuencias y operaciones no soportadas.
"""
from matrixai.playground import (
    analyze_playground_request,
    _network_is_composite,
    _prompt_wants_composite,
    _prompt_is_sequence,
    _prompt_unsupported_ops,
)


def _stage_warnings(result) -> list[str]:
    return [w for s in result.get("pipeline_stages", []) for w in (s.get("warnings") or [])]


class TestDetectors:
    def test_wants_composite_on_hints(self):
        assert _prompt_wants_composite("red con bloques residuales y LayerNorm")
        assert _prompt_wants_composite("a complex nonlinear model")
        assert not _prompt_wants_composite("clasificar spam de email")

    def test_sequence_detection(self):
        assert _prompt_is_sequence("predecir una serie temporal")
        assert not _prompt_is_sequence("clasificar pacientes tabulares")

    def test_unsupported_ops(self):
        assert _prompt_unsupported_ops("modelo con atención") == ["attention"]
        assert "LSTM" in _prompt_unsupported_ops("una red lstm recurrente")
        assert _prompt_unsupported_ops("red densa normal") == []


class TestRouting:
    def test_composite_prompt_routes_to_composite(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar riesgo en 3 niveles con red profunda de bloques "
                      "residuales con LayerNorm y Dropout, 8 features clínicas",
        })
        assert r["supervision_source"] == "composite_generator"
        mxai = r["mxai"]
        assert _network_is_composite(mxai)
        assert "BLOCK" in mxai and "RESIDUAL" in mxai and "LayerNorm" in mxai
        names = [s.get("name") for s in r["pipeline_stages"]]
        assert "composite_generator" in names

    def test_normal_neural_prompt_stays_dense(self):
        r = analyze_playground_request({
            "mode": "prompt", "prompt": "detectar fraude en transacciones bancarias",
        })
        assert r["supervision_source"] == "dense_generator"
        assert not _network_is_composite(r["mxai"])
        names = [s.get("name") for s in r["pipeline_stages"]]
        assert "dense_generator" in names

    def test_no_native_embedding_in_v1(self):
        # v1: categoricals are handled by the S2 one-hot editor, never auto-EMBEDDING
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "red residual compleja para clasificar por tipo y region, 8 features",
        })
        assert "EMBEDDING" not in r["mxai"]


class TestWarnings:
    def test_sequence_prompt_falls_back_to_dense_with_warning(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "predecir serie temporal de demanda con bloques residuales",
        })
        assert r["supervision_source"] == "dense_generator"  # no composite for sequences in v1
        assert not _network_is_composite(r["mxai"])
        assert any("ecuencias" in w or "emporal" in w for w in _stage_warnings(r))

    def test_unsupported_op_warns_on_neural_composite(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "clasificar pacientes en 3 niveles con red profunda de bloques "
                      "residuales y mecanismo de atención, 8 features",
        })
        assert r["supervision_source"] == "composite_generator"
        assert any("attention" in w for w in _stage_warnings(r))


class TestGeneratedCompositeTrains:
    def test_generated_composite_trains_e2e(self):
        # C2+C3 integration: a prompt-generated composite actually trains
        from matrixai.playground import _run_playground_training
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "Clasificar en 3 niveles BAJO MEDIO ALTO con red profunda de "
                      "bloques residuales con LayerNorm y Dropout, features f1 f2 f3 f4",
        })
        assert r["supervision_source"] == "composite_generator"
        mxai = r["mxai"]
        training = r["training_text"]
        # build a tiny CSV matching the generated columns
        import re
        cols_m = re.search(r"FROM COLUMNS \[([^\]]*)\]", training)
        cols = [c.strip() for c in cols_m.group(1).split(",")]
        target_m = re.search(r"TARGET\s+(\w+)\s*:\s*Label\[([^\]]*)\]", training)
        target, labels = target_m.group(1), [l.strip() for l in target_m.group(2).split(",")]
        header = ",".join(cols + [target])
        rows = [header]
        for i in range(40):
            vals = [f"{((i + j) % 10) / 10.0:.3f}" for j in range(len(cols))]
            rows.append(",".join(vals + [labels[i % len(labels)]]))
        csv_text = "\n".join(rows) + "\n"
        tr = _run_playground_training(mxai, training, csv_text, epochs_override=3)
        assert tr["ok"], tr.get("error")
        assert tr["network_kind"] == "composite_network"
        assert tr["params_best"] is not None
