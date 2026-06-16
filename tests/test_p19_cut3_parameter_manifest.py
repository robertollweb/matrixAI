"""P19 C3 — Parameter manifest jerárquico extendido: embeddings, gamma/beta, paths con blocks."""
from __future__ import annotations

import pytest

from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types
from matrixai.parameters.network_params import (
    composite_network_parameter_manifest,
    composite_network_parameter_schema_hash,
    build_composite_network_parameter_set,
    validate_composite_network_parameter_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_and_check(src: str):
    program = parse_text(src)
    net = program.networks[0]
    vbn = {v.name: v for v in program.vectors}
    result = check_composite_network_types(net, vbn)
    assert result.ok, result.errors
    return net, result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MXAI_EMBEDDING_BASIC = """
PROJECT EmbTest

VECTOR Product[3]
  category_id: Integer[0, 100]
  price: Scalar
  weight: Scalar
END

NETWORK CategoryNet
  INPUT Product
  EMBEDDING cat_emb FROM category_id VOCAB 100 DIM 8
  CONCAT [cat_emb, price, weight] -> features
  LAYER Dense units=16 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT label: ProbabilityMap[a, b, c]
END

GRAPH
  Product -> CategoryNet
END
"""

_MXAI_BLOCK_RESIDUAL_PREVIOUS = """
PROJECT ResTest

VECTOR H[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet
  INPUT H
  LAYER Dense units=8 activation=relu
  BLOCK res1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  H -> ResNet
END
"""

_MXAI_BLOCK_RESIDUAL_NAMED = """
PROJECT ResNamedTest

VECTOR H[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet
  INPUT H
  EMBEDDING emb FROM x1 VOCAB 10 DIM 4
  CONCAT [emb, x2] -> features
  BLOCK res1
    LAYER Dense units=5 activation=relu
    LAYER Dropout rate=0.1
    RESIDUAL FROM features
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[pos, neg]
END

GRAPH
  H -> ResNet
END
"""

_MXAI_ALL_LAYERS = """
PROJECT AllLayersTest

VECTOR V[2]
  a: Scalar
  b: Scalar
END

NETWORK AllNet
  INPUT V
  LAYER Dense units=16 activation=relu
  LAYER LayerNorm
  LAYER Dropout rate=0.2
  LAYER Activation kind=relu
  LAYER Dense units=4 activation=softmax
  OUTPUT label: ProbabilityMap[a, b, c, d]
END

GRAPH
  V -> AllNet
END
"""


# ---------------------------------------------------------------------------
# TestCompositeManifestEmbedding
# ---------------------------------------------------------------------------

class TestCompositeManifestEmbedding:
    def test_manifest_includes_embedding_table(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        table_entries = [e for e in manifest if e["role"] == "embedding_table"]
        assert len(table_entries) == 1
        assert table_entries[0]["path"] == "CategoryNet.cat_emb.table"

    def test_manifest_embedding_table_shape_is_vocab_by_dim(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        table = next(e for e in manifest if e["role"] == "embedding_table")
        assert table["shape"] == [100, 8]

    def test_manifest_embedding_table_initializer_is_xavier_normal(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        table = next(e for e in manifest if e["role"] == "embedding_table")
        assert table["initializer"] == "xavier_normal"

    def test_manifest_no_entries_for_concat_dropout_pool(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_NAMED)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        # Dropout inside block has no parameters
        roles = {e["role"] for e in manifest}
        assert "dropout" not in roles
        assert "concat" not in roles
        # Paths should not contain "Dropout" as a layer type
        paths = [e["path"] for e in manifest]
        assert not any("Dropout" in p for p in paths)


# ---------------------------------------------------------------------------
# TestCompositeManifestLayerNorm
# ---------------------------------------------------------------------------

class TestCompositeManifestLayerNorm:
    def test_manifest_includes_layernorm_gamma_and_beta(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        gamma = [e for e in manifest if e["role"] == "gamma"]
        beta = [e for e in manifest if e["role"] == "beta"]
        assert len(gamma) == 1
        assert len(beta) == 1

    def test_manifest_layernorm_gamma_initializer_is_ones(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        gamma = next(e for e in manifest if e["role"] == "gamma")
        assert gamma["initializer"] == "ones"

    def test_manifest_layernorm_beta_initializer_is_zeros(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        beta = next(e for e in manifest if e["role"] == "beta")
        assert beta["initializer"] == "zeros"

    def test_manifest_layernorm_gamma_shape_equals_features(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        gamma = next(e for e in manifest if e["role"] == "gamma")
        # LN is inside block after Dense(8), so input_shape=[8], features=8
        assert gamma["shape"] == [8]

    def test_manifest_top_level_layernorm_has_correct_path(self):
        net, result = _parse_and_check(_MXAI_ALL_LAYERS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        gamma = next(e for e in manifest if e["role"] == "gamma")
        # Top-level LayerNorm at index 2 → AllNet.L2.gamma
        assert gamma["path"] == "AllNet.L2.gamma"


# ---------------------------------------------------------------------------
# TestCompositeManifestHierarchicalPaths
# ---------------------------------------------------------------------------

class TestCompositeManifestHierarchicalPaths:
    def test_block_dense_path_includes_block_name(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        weight_paths = [e["path"] for e in manifest if e["role"] == "weights"]
        # Dense inside block res1 → ResNet.res1.L1.W
        assert any("res1" in p for p in weight_paths)

    def test_block_layernorm_path_includes_block_name(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        gamma_paths = [e["path"] for e in manifest if e["role"] == "gamma"]
        assert any("res1" in p for p in gamma_paths)

    def test_top_level_dense_path_does_not_include_block_name(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        weight_paths = [e["path"] for e in manifest if e["role"] == "weights"]
        # Top-level Dense at idx=1 → ResNet.L1.W (no block prefix)
        # Top-level Dense at idx=2 → ResNet.L2.W
        top_level = [p for p in weight_paths if "res1" not in p]
        assert len(top_level) == 2  # Dense idx=1 and Dense idx=2

    def test_full_path_structure_for_block_layers(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        manifest = composite_network_parameter_manifest(net.name, net, result)
        paths = {e["path"] for e in manifest}
        # Dense(8,relu) at top idx=1
        assert "ResNet.L1.W" in paths
        assert "ResNet.L1.b" in paths
        # Dense(8,relu) inside block res1 at block-idx=1
        assert "ResNet.res1.L1.W" in paths
        assert "ResNet.res1.L1.b" in paths
        # LayerNorm inside block res1 at block-idx=2
        assert "ResNet.res1.L2.gamma" in paths
        assert "ResNet.res1.L2.beta" in paths
        # Dense(1,linear) at top idx=2
        assert "ResNet.L2.W" in paths
        assert "ResNet.L2.b" in paths


# ---------------------------------------------------------------------------
# TestCompositeSchemaHash
# ---------------------------------------------------------------------------

class TestCompositeSchemaHash:
    def test_schema_hash_stable_for_same_architecture(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        h1 = composite_network_parameter_schema_hash(net.name, net, result)
        h2 = composite_network_parameter_schema_hash(net.name, net, result)
        assert h1 == h2

    def test_schema_hash_changes_when_embedding_dim_changes(self):
        net8, result8 = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        h8 = composite_network_parameter_schema_hash(net8.name, net8, result8)

        src16 = _MXAI_EMBEDDING_BASIC.replace("VOCAB 100 DIM 8", "VOCAB 100 DIM 16")
        net16, result16 = _parse_and_check(src16)
        h16 = composite_network_parameter_schema_hash(net16.name, net16, result16)

        assert h8 != h16

    def test_schema_hash_changes_when_dense_units_change(self):
        net16, result16 = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        h16 = composite_network_parameter_schema_hash(net16.name, net16, result16)

        src32 = _MXAI_EMBEDDING_BASIC.replace(
            "LAYER Dense units=16 activation=relu", "LAYER Dense units=32 activation=relu"
        )
        net32, result32 = _parse_and_check(src32)
        h32 = composite_network_parameter_schema_hash(net32.name, net32, result32)

        assert h16 != h32

    def test_schema_hash_changes_when_residual_source_changes(self):
        # RESIDUAL FROM PREVIOUS vs RESIDUAL FROM features — different architecture
        net_prev, result_prev = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        h_prev = composite_network_parameter_schema_hash(net_prev.name, net_prev, result_prev)

        # ResNet in _MXAI_BLOCK_RESIDUAL_NAMED also uses residual but FROM features
        net_named, result_named = _parse_and_check(_MXAI_BLOCK_RESIDUAL_NAMED)
        h_named = composite_network_parameter_schema_hash(net_named.name, net_named, result_named)

        assert h_prev != h_named


# ---------------------------------------------------------------------------
# TestCompositeParameterSet
# ---------------------------------------------------------------------------

class TestCompositeParameterSet:
    def test_build_composite_parameter_set_contains_all_params(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        ps = build_composite_network_parameter_set(net, result, model_hash_str="testhash123")
        manifest = composite_network_parameter_manifest(net.name, net, result)
        for entry in manifest:
            assert entry["path"] in ps.parameters, f"missing {entry['path']}"

    def test_build_composite_parameter_set_embedding_table_shape(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        ps = build_composite_network_parameter_set(net, result, model_hash_str="testhash123")
        table_param = ps.parameters["CategoryNet.cat_emb.table"]
        assert table_param["shape"] == [100, 8]

    def test_build_composite_parameter_set_gamma_values_are_ones(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        ps = build_composite_network_parameter_set(net, result, model_hash_str="testhash123")
        gamma_param = ps.parameters["ResNet.res1.L2.gamma"]
        assert all(v == 1.0 for v in gamma_param["values"])

    def test_build_composite_parameter_set_beta_values_are_zeros(self):
        net, result = _parse_and_check(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        ps = build_composite_network_parameter_set(net, result, model_hash_str="testhash123")
        beta_param = ps.parameters["ResNet.res1.L2.beta"]
        assert all(v == 0.0 for v in beta_param["values"])

    def test_validate_composite_parameter_set_ok_for_matching_ps(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        ps = build_composite_network_parameter_set(net, result, model_hash_str="testhash123")
        compat = validate_composite_network_parameter_set(
            net, result, ps, model_hash_str="testhash123"
        )
        assert compat.ok, compat.errors

    def test_validate_detects_missing_parameter(self):
        net, result = _parse_and_check(_MXAI_EMBEDDING_BASIC)
        ps = build_composite_network_parameter_set(net, result, model_hash_str="testhash123")
        # Remove one parameter
        params = dict(ps.parameters)
        del params["CategoryNet.cat_emb.table"]
        ps2 = ps.__class__(
            parameter_set_id=ps.parameter_set_id,
            model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
            source=ps.source,
            parameters=params,
        )
        compat = validate_composite_network_parameter_set(
            net, result, ps2, model_hash_str="testhash123"
        )
        assert not compat.ok
        assert any("cat_emb.table" in e for e in compat.errors)
