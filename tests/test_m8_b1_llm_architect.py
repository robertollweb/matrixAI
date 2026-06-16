"""M8-B1 — El LLM propone la arquitectura (techo de calidad).

El esquema LLM puede elegir TIPO (dense vs residual) con un análisis del problema
y una justificación. Su elección enruta el generador (alimenta el sanitizador A1)
y se registra para auditoría. Sin LLM, cae al determinista (suelo). El Studio es
descargable y el LLM de capacidad variable → A1 sigue gobernando.
"""
from unittest.mock import patch

from matrixai.playground import analyze_playground_request, _network_is_composite, _parse_dense_schema

_BASE = {
    "input_fields": ["a", "b", "c", "d", "e", "f"],
    "labels": ["BAJO", "MEDIO", "ALTO"],
    "network_name": "Riesgo",
    "input_name": "Paciente",
}


class TestParser:
    def test_parses_architecture_and_rationale(self):
        text = (
            "FIELDS: a, b\nLABELS: X, Y\nNAME: M\nENTITY: E\n"
            "ARCHITECTURE: residual\nLAYERS: 64, 32\nRATIONALE: non-linear interactions\n"
        )
        r = _parse_dense_schema(text)
        assert r["architecture"] == "residual"
        assert r["rationale"] == "non-linear interactions"

    def test_architecture_normalized_to_dense(self):
        r = _parse_dense_schema("ARCHITECTURE: Dense MLP\n")
        assert r["architecture"] == "dense"

    def test_no_architecture_line(self):
        r = _parse_dense_schema("FIELDS: a, b\n")
        assert "architecture" not in r


class TestRouting:
    def test_llm_residual_routes_composite_without_prompt_hints(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "residual", "rationale": "no lineal"}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo de pacientes", "use_llm": True,
            })
        assert _network_is_composite(r["mxai"])
        assert r["architecture_decision"] == {
            "kind": "residual", "source": "llm", "rationale": "no lineal",
        }

    def test_llm_dense_stays_dense(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "dense", "rationale": "tabular"}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo de pacientes", "use_llm": True,
            })
        assert not _network_is_composite(r["mxai"])
        assert r["architecture_decision"]["kind"] == "dense"
        assert r["architecture_decision"]["source"] == "llm"

    def test_no_llm_is_deterministic_default(self):
        r = analyze_playground_request({
            "mode": "prompt", "prompt": "clasificar riesgo de pacientes", "use_llm": False,
        })
        assert not _network_is_composite(r["mxai"])
        assert r["architecture_decision"]["source"] == "default"

    def test_prompt_hints_win_without_llm(self):
        r = analyze_playground_request({
            "mode": "prompt",
            "prompt": "clasificar riesgo con bloques residuales y LayerNorm", "use_llm": False,
        })
        assert _network_is_composite(r["mxai"])
        assert r["architecture_decision"]["source"] == "prompt_hints"

    def test_rationale_surfaced_in_pipeline(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "residual", "rationale": "interacciones no lineales"}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo", "use_llm": True,
            })
        warns = [w for s in r.get("pipeline_stages", []) for w in (s.get("warnings") or [])]
        assert any("interacciones no lineales" in w for w in warns)

    def test_llm_residual_still_sanitized(self):
        # M8-A1 still governs the LLM path: a narrow ReLU bottleneck is widened.
        # (residual generator manages its own widths; dense path sanitizes layers)
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "dense",
                                 "hidden_layers": [(64, "relu"), (3, "relu")], "rationale": "x"}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo", "use_llm": True,
            })
        import re
        narrow = [
            l for l in r["mxai"].splitlines()
            if (m := re.search(r"units=(\d+) activation=relu", l)) and int(m.group(1)) < 16
        ]
        assert narrow == []
