"""P19 C4 — BackendContractAnalyzer extendido para composite_network."""
from __future__ import annotations

import pytest

from matrixai.parser.parser import parse_text
from matrixai.compiler.backend_contract import BackendContractAnalyzer


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

_MXAI_BLOCK_WITH_DROPOUT = """
PROJECT DropTest

VECTOR V[2]
  a: Scalar
  b: Scalar
END

NETWORK DropNet
  INPUT V
  EMBEDDING emb FROM a VOCAB 10 DIM 4
  CONCAT [emb, b] -> features
  BLOCK b1
    LAYER Dense units=5 activation=relu
    LAYER Dropout rate=0.1
    RESIDUAL FROM features
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[pos, neg]
END

GRAPH
  V -> DropNet
END
"""

_MXAI_TWO_RESIDUAL_BLOCKS = """
PROJECT TwoResTest

VECTOR H[2]
  x1: Scalar
  x2: Scalar
END

NETWORK ResNet2
  INPUT H
  LAYER Dense units=8 activation=relu
  BLOCK res1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  BLOCK res2
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  H -> ResNet2
END
"""

_MXAI_LARGE_EMBEDDING_DIM = """
PROJECT LargEmbTest

VECTOR V[2]
  cat: Integer[0, 50]
  val: Scalar
END

NETWORK LargeNet
  INPUT V
  EMBEDDING cat_emb FROM cat VOCAB 50 DIM 32
  CONCAT [cat_emb, val] -> features
  LAYER Dense units=4 activation=softmax
  OUTPUT label: ProbabilityMap[a, b, c, d]
END

GRAPH
  V -> LargeNet
END
"""

_MXAI_P18_DENSE_ONLY = """
PROJECT P18Test

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK DenseNet
  INPUT V
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> DenseNet
END
"""


def _analyze(src: str) -> "BackendContractReport":
    program = parse_text(src)
    return BackendContractAnalyzer().analyze(program)


# ---------------------------------------------------------------------------
# TestCompositeNetworkNodeReport
# ---------------------------------------------------------------------------

class TestCompositeNetworkNodeReport:
    def test_node_type_is_composite_network(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        net_node = next(n for n in report.nodes if n.node == "CategoryNet")
        assert net_node.node_type == "composite_network"

    def test_composite_network_is_supported(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        net_node = next(n for n in report.nodes if n.node == "CategoryNet")
        assert net_node.supported

    def test_composite_network_is_differentiable(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        net_node = next(n for n in report.nodes if n.node == "CategoryNet")
        assert net_node.differentiable

    def test_composite_network_kind_is_composite_network(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        net_node = next(n for n in report.nodes if n.node == "CategoryNet")
        assert net_node.kind == "composite_network"

    def test_composite_network_output_shape_from_final_dense(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        net_node = next(n for n in report.nodes if n.node == "CategoryNet")
        assert net_node.output_shape == (3,)

    def test_report_ok_for_valid_composite_model(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        assert report.ok, report.parameter_errors


# ---------------------------------------------------------------------------
# TestCompositeTrainableParameters
# ---------------------------------------------------------------------------

class TestCompositeTrainableParameters:
    def test_trainable_params_include_embedding_table(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        paths = [p.path for p in report.trainable_parameters]
        assert "CategoryNet.cat_emb.table" in paths

    def test_trainable_params_include_dense_weights(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        paths = [p.path for p in report.trainable_parameters]
        assert any(".L1.W" in p for p in paths)
        assert any(".L1.b" in p for p in paths)

    def test_trainable_params_include_layernorm_gamma_beta(self):
        report = _analyze(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        paths = [p.path for p in report.trainable_parameters]
        assert any(".gamma" in p for p in paths)
        assert any(".beta" in p for p in paths)

    def test_trainable_params_have_hierarchical_block_paths(self):
        report = _analyze(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        paths = [p.path for p in report.trainable_parameters]
        assert any("res1" in p for p in paths)

    def test_embedding_table_shape_in_parameter_manifest(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        manifest = report.parameter_manifest
        table = next(e for e in manifest if "cat_emb.table" in e.get("path", ""))
        assert table["shape"] == [100, 8]

    def test_embedding_table_initializer_in_parameter_manifest(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        manifest = report.parameter_manifest
        table = next(e for e in manifest if "cat_emb.table" in e.get("path", ""))
        assert table["initializer"] == "xavier_normal"


# ---------------------------------------------------------------------------
# TestCompositeLayerManifest
# ---------------------------------------------------------------------------

class TestCompositeLayerManifest:
    def test_layer_manifest_includes_embedding_entry(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        entry = next(
            (e for e in report.layer_manifest if e.get("layer_type") == "Embedding"), None
        )
        assert entry is not None
        assert entry["embedding_name"] == "cat_emb"

    def test_layer_manifest_includes_block_entry(self):
        report = _analyze(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        block_entry = next(
            (e for e in report.layer_manifest if e.get("layer_type") == "Block"), None
        )
        assert block_entry is not None
        assert block_entry["block_name"] == "res1"

    def test_block_entry_has_sub_layers(self):
        report = _analyze(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        block_entry = next(e for e in report.layer_manifest if e.get("layer_type") == "Block")
        assert "layers" in block_entry
        assert len(block_entry["layers"]) >= 2

    def test_block_entry_reports_residual_from(self):
        report = _analyze(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        block_entry = next(e for e in report.layer_manifest if e.get("layer_type") == "Block")
        assert block_entry["residual_from"] == "PREVIOUS"

    def test_dropout_layer_has_dropout_active_flag(self):
        report = _analyze(_MXAI_BLOCK_WITH_DROPOUT)
        block_entry = next(e for e in report.layer_manifest if e.get("layer_type") == "Block")
        dropout_layers = [l for l in block_entry["layers"] if l.get("layer_type") == "Dropout"]
        assert len(dropout_layers) == 1
        assert dropout_layers[0]["dropout_active"] is True
        assert dropout_layers[0]["rate"] == pytest.approx(0.1)

    def test_dense_entry_has_correct_layer_type(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        dense_entries = [e for e in report.layer_manifest if e.get("layer_type") == "Dense"]
        assert len(dense_entries) >= 1

    def test_to_dict_includes_layer_manifest(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        d = report.to_dict()
        assert "layer_manifest" in d
        assert len(d["layer_manifest"]) > 0


# ---------------------------------------------------------------------------
# TestCompositeInterpretabilityWarnings
# ---------------------------------------------------------------------------

class TestCompositeInterpretabilityWarnings:
    def test_warns_interpretability_reduced_for_composite(self):
        report = _analyze(_MXAI_EMBEDDING_BASIC)
        assert any("interpretability_level=reduced" in w for w in report.warnings)

    def test_warns_very_reduced_for_two_residual_blocks(self):
        report = _analyze(_MXAI_TWO_RESIDUAL_BLOCKS)
        assert any("very_reduced" in w for w in report.warnings)

    def test_warns_very_reduced_for_large_embedding_dim(self):
        report = _analyze(_MXAI_LARGE_EMBEDDING_DIM)
        assert any("very_reduced" in w for w in report.warnings)

    def test_not_very_reduced_for_single_small_residual_block(self):
        report = _analyze(_MXAI_BLOCK_RESIDUAL_PREVIOUS)
        assert not any("very_reduced" in w for w in report.warnings)
        assert any("interpretability_level=reduced" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# TestDenseNetworkUnchangedByC4
# ---------------------------------------------------------------------------

class TestDenseNetworkUnchangedByC4:
    def test_dense_network_still_reports_dense_network_kind(self):
        report = _analyze(_MXAI_P18_DENSE_ONLY)
        net_node = next(n for n in report.nodes if n.node == "DenseNet")
        assert net_node.node_type == "dense_network"

    def test_dense_network_p18_warning_unchanged(self):
        report = _analyze(_MXAI_P18_DENSE_ONLY)
        assert any("interpretability_level=reduced" in w for w in report.warnings)
        assert not any("composite" in w for w in report.warnings)
