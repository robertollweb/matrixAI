"""M2 v2 — C5: el esquema LLM declara categóricas → EMBEDDING nativo.

El LLM puede devolver `CATEGORICALS: field:vocab` para features de alta cardinalidad
sin orden natural. Eso enruta al CompositeNetworkGenerator, que emite EMBEDDING+CONCAT.
Sin categóricas, el comportamiento previo (dense / residual) no cambia.
"""
from __future__ import annotations

from unittest.mock import patch

from matrixai.parser import parse_text
from matrixai.playground import (
    analyze_playground_request,
    _network_is_composite,
    _parse_dense_schema,
)

_BASE = {
    "input_fields": ["edad", "categoria_producto", "precio"],
    "labels": ["BAJO", "MEDIO", "ALTO"],
    "network_name": "Riesgo",
    "input_name": "Cliente",
}


class TestParser:
    def test_parses_categoricals(self):
        r = _parse_dense_schema(
            "FIELDS: a, b\nCATEGORICALS: product_id:5000, diagnosis:1200\n"
        )
        assert r["categorical_fields"] == {"product_id": 5000, "diagnosis": 1200}

    def test_ignores_malformed_and_tiny_vocab(self):
        r = _parse_dense_schema(
            "CATEGORICALS: ok_field:50, bad_field, x:1, y:notnum\n"
        )
        assert r["categorical_fields"] == {"ok_field": 50}

    def test_no_categoricals_line(self):
        r = _parse_dense_schema("FIELDS: a, b\nARCHITECTURE: dense\n")
        assert "categorical_fields" not in r


class TestRouting:
    def test_llm_categoricals_emit_embedding_composite(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "dense",
                                 "categorical_fields": {"categoria_producto": 40}}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo de clientes", "use_llm": True,
            })
        assert _network_is_composite(r["mxai"])
        assert "EMBEDDING" in r["mxai"]
        assert "categoria_producto" in r["mxai"]
        # embedding-only composite is labelled 'composite', not 'residual'
        assert r["architecture_decision"]["kind"] == "composite"
        # the generated mxai parses and the categorical is an Integer index field
        prog = parse_text(r["mxai"])
        assert prog.networks[0].kind == "composite_network"
        assert "categoria_producto: Integer[0, 39]" in r["mxai"]

    def test_categorical_missing_from_fields_is_folded_in(self):
        # LLM lists a categorical not present in FIELDS → it must still reach the VECTOR
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={"input_fields": ["edad", "precio"],
                                 "labels": ["A", "B"], "network_name": "M", "input_name": "E",
                                 "categorical_fields": {"zona": 30}}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar clientes", "use_llm": True,
            })
        assert "EMBEDDING" in r["mxai"] and "zona" in r["mxai"]
        prog = parse_text(r["mxai"])
        assert "zona" in [f for v in prog.vectors for f in v.fields]

    def test_no_categoricals_keeps_dense(self):
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "dense"}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo de clientes", "use_llm": True,
            })
        assert not _network_is_composite(r["mxai"])
        assert "EMBEDDING" not in r["mxai"]

    def test_residual_plus_categoricals(self):
        # both residual architecture and a categorical → residual block AND embedding
        with patch("matrixai.playground._dense_llm_schema",
                   return_value={**_BASE, "architecture": "residual",
                                 "categorical_fields": {"categoria_producto": 40}}):
            r = analyze_playground_request({
                "mode": "prompt", "prompt": "clasificar riesgo de clientes", "use_llm": True,
            })
        assert "EMBEDDING" in r["mxai"] and "BLOCK" in r["mxai"]
        assert r["architecture_decision"]["kind"] == "residual"
