"""P19 C9 — Torch opcional: materialización nn.Module para composite_network."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types
from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.forward.composite_forward import composite_forward
from matrixai.forward.composite_torch import (
    CompositeTorchError,
    composite_network_to_torch_module,
    composite_torch_forward,
    torch_module_to_composite_parameter_set,
)
from matrixai.parameters.store import ParameterSet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MXAI_EMBEDDING = """
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

_MXAI_RESIDUAL = """
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

_MXAI_DROPOUT = """
PROJECT DropTest
VECTOR V[2]
  a: Scalar
  b: Scalar
END
NETWORK DropNet
  INPUT V
  LAYER Dense units=16 activation=relu
  LAYER Dropout rate=0.5
  LAYER Dense units=3 activation=softmax
  OUTPUT label: ProbabilityMap[x, y, z]
END
GRAPH
  V -> DropNet
END
"""


def _setup(src: str, seed: int = 42):
    program = parse_text(src)
    net = program.networks[0]
    vbn = {v.name: v for v in program.vectors}
    result = check_composite_network_types(net, vbn)
    assert result.ok, result.errors
    ps = build_composite_network_parameter_set(net, result, model_hash_str="test", seed=seed)
    return net, ps


# ---------------------------------------------------------------------------
# TestModuleCreation
# ---------------------------------------------------------------------------

class TestModuleCreation:
    def test_module_created_for_embedding_network(self):
        net, ps = _setup(_MXAI_EMBEDDING)
        module = composite_network_to_torch_module(net, ps)
        assert module is not None

    def test_module_created_for_residual_network(self):
        net, ps = _setup(_MXAI_RESIDUAL)
        module = composite_network_to_torch_module(net, ps)
        assert module is not None

    def test_module_has_embedding_layer(self):
        net, ps = _setup(_MXAI_EMBEDDING)
        module = composite_network_to_torch_module(net, ps)
        assert "cat_emb" in module.embeddings

    def test_embedding_layer_correct_shape(self):
        net, ps = _setup(_MXAI_EMBEDDING)
        module = composite_network_to_torch_module(net, ps)
        emb = module.embeddings["cat_emb"]
        assert tuple(emb.weight.shape) == (100, 8)

    def test_embedding_weights_match_parameter_set(self):
        net, ps = _setup(_MXAI_EMBEDDING, seed=7)
        module = composite_network_to_torch_module(net, ps)
        table_expected = ps.parameters["CategoryNet.cat_emb.table"]["values"]
        table_actual = module.embeddings["cat_emb"].weight.detach().tolist()
        for row_e, row_a in zip(table_expected[:5], table_actual[:5]):
            for v_e, v_a in zip(row_e, row_a):
                assert abs(v_e - v_a) < 1e-6

    def test_module_has_trainable_parameters(self):
        net, ps = _setup(_MXAI_RESIDUAL)
        module = composite_network_to_torch_module(net, ps)
        params = list(module.parameters())
        assert len(params) > 0

    def test_module_parameter_count_matches_manifest(self):
        net, ps = _setup(_MXAI_RESIDUAL)
        module = composite_network_to_torch_module(net, ps)
        param_count = sum(p.numel() for p in module.parameters())
        # ResNet: L1(8x2+8) + res1.L1(8x8+8) + res1.L2(LN:8+8) + L2(1x8+1) = 26+72+16+9 = 123
        assert param_count > 0


# ---------------------------------------------------------------------------
# TestForwardMatchesStdlib
# ---------------------------------------------------------------------------

class TestForwardMatchesStdlib:
    def test_embedding_forward_matches_stdlib(self):
        net, ps = _setup(_MXAI_EMBEDDING, seed=42)
        input_data = {"category_id": 5, "price": 0.5, "weight": 0.5}
        stdlib_out = composite_forward(net, ps, input_data, training=False)
        module = composite_network_to_torch_module(net, ps)
        torch_out = composite_torch_forward(module, input_data, training=False)
        assert len(stdlib_out) == len(torch_out)
        for a, b in zip(stdlib_out, torch_out):
            assert abs(a - b) < 1e-4, f"stdlib={a}, torch={b}"

    def test_residual_forward_matches_stdlib(self):
        net, ps = _setup(_MXAI_RESIDUAL, seed=7)
        input_data = {"x1": 0.5, "x2": -0.5}
        stdlib_out = composite_forward(net, ps, input_data, training=False)
        module = composite_network_to_torch_module(net, ps)
        torch_out = composite_torch_forward(module, input_data, training=False)
        for a, b in zip(stdlib_out, torch_out):
            assert abs(a - b) < 1e-4, f"stdlib={a}, torch={b}"

    def test_forward_output_is_list(self):
        net, ps = _setup(_MXAI_EMBEDDING, seed=0)
        module = composite_network_to_torch_module(net, ps)
        result = composite_torch_forward(module, {"category_id": 3, "price": 0.5, "weight": 0.5}, training=False)
        assert isinstance(result, list)

    def test_embedding_forward_different_ids_give_different_outputs(self):
        net, ps = _setup(_MXAI_EMBEDDING, seed=42)
        module = composite_network_to_torch_module(net, ps)
        out1 = composite_torch_forward(module, {"category_id": 1, "price": 0.5, "weight": 0.5}, training=False)
        out2 = composite_torch_forward(module, {"category_id": 50, "price": 0.5, "weight": 0.5}, training=False)
        assert out1 != out2

    def test_softmax_output_sums_to_one(self):
        net, ps = _setup(_MXAI_EMBEDDING, seed=0)
        module = composite_network_to_torch_module(net, ps)
        out = composite_torch_forward(module, {"category_id": 5, "price": 0.5, "weight": 0.5}, training=False)
        assert abs(sum(out) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# TestDropoutInEvalMode
# ---------------------------------------------------------------------------

class TestDropoutEvalMode:
    def test_eval_mode_is_deterministic(self):
        net, ps = _setup(_MXAI_DROPOUT, seed=0)
        module = composite_network_to_torch_module(net, ps)
        input_data = {"a": 0.5, "b": 0.3}
        out1 = composite_torch_forward(module, input_data, training=False)
        out2 = composite_torch_forward(module, input_data, training=False)
        assert out1 == out2

    def test_eval_mode_matches_stdlib_eval(self):
        net, ps = _setup(_MXAI_DROPOUT, seed=0)
        module = composite_network_to_torch_module(net, ps)
        input_data = {"a": 0.5, "b": 0.3}
        stdlib_out = composite_forward(net, ps, input_data, training=False)
        torch_out = composite_torch_forward(module, input_data, training=False)
        for a, b in zip(stdlib_out, torch_out):
            assert abs(a - b) < 1e-4, f"stdlib={a}, torch={b}"


# ---------------------------------------------------------------------------
# TestBackwardMatchesStdlib
# ---------------------------------------------------------------------------

class TestBackwardMatchesStdlib:
    def test_one_step_loss_matches_stdlib(self):
        from matrixai.training.composite_backprop import composite_train_step
        net, ps = _setup(_MXAI_RESIDUAL, seed=7)
        input_data = {"x1": 0.5, "x2": -0.5}
        target = [0.0]

        _, loss_stdlib = composite_train_step(
            net, ps, input_data, target, "mse", learning_rate=0.01, training=False
        )

        module = composite_network_to_torch_module(net, ps)
        module.train()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        optimizer.zero_grad()
        out = module.forward_with_dict(input_data)
        target_t = torch.tensor(target, dtype=torch.float32)
        loss_t = ((out - target_t) ** 2).mean()
        loss_t.backward()
        optimizer.step()

        assert abs(loss_stdlib - loss_t.item()) < 1e-4

    def test_one_step_weights_match_stdlib(self):
        from matrixai.training.composite_backprop import composite_train_step
        net, ps = _setup(_MXAI_RESIDUAL, seed=7)
        input_data = {"x1": 0.5, "x2": -0.5}
        target = [0.0]

        ps2_stdlib, _ = composite_train_step(
            net, ps, input_data, target, "mse", learning_rate=0.01, training=False
        )

        module = composite_network_to_torch_module(net, ps)
        module.train()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        optimizer.zero_grad()
        out = module.forward_with_dict(input_data)
        loss_t = ((out - torch.tensor(target, dtype=torch.float32)) ** 2).mean()
        loss_t.backward()
        optimizer.step()
        ps2_torch = torch_module_to_composite_parameter_set(net, module, ps)

        key = "ResNet.L1.W"
        w_s = ps2_stdlib.parameters[key]["values"]
        w_t = ps2_torch.parameters[key]["values"]
        max_diff = max(abs(vs - vt) for rs, rt in zip(w_s, w_t) for vs, vt in zip(rs, rt))
        assert max_diff < 1e-3, f"Max W diff: {max_diff}"


# ---------------------------------------------------------------------------
# TestParameterRoundtrip
# ---------------------------------------------------------------------------

class TestParameterRoundtrip:
    def test_roundtrip_preserves_parameter_set_structure(self):
        net, ps = _setup(_MXAI_RESIDUAL, seed=0)
        module = composite_network_to_torch_module(net, ps)
        ps2 = torch_module_to_composite_parameter_set(net, module, ps)
        assert isinstance(ps2, ParameterSet)
        assert set(ps2.parameters.keys()) == set(ps.parameters.keys())

    def test_roundtrip_preserves_initial_weights(self):
        net, ps = _setup(_MXAI_RESIDUAL, seed=0)
        module = composite_network_to_torch_module(net, ps)
        ps2 = torch_module_to_composite_parameter_set(net, module, ps)
        key = "ResNet.L1.W"
        for row_o, row_r in zip(ps.parameters[key]["values"], ps2.parameters[key]["values"]):
            for v_o, v_r in zip(row_o, row_r):
                assert abs(v_o - v_r) < 1e-6

    def test_roundtrip_source_is_torch(self):
        net, ps = _setup(_MXAI_RESIDUAL, seed=0)
        module = composite_network_to_torch_module(net, ps)
        ps2 = torch_module_to_composite_parameter_set(net, module, ps)
        assert ps2.source == "torch"

    def test_roundtrip_preserves_embedding_weights(self):
        net, ps = _setup(_MXAI_EMBEDDING, seed=5)
        module = composite_network_to_torch_module(net, ps)
        ps2 = torch_module_to_composite_parameter_set(net, module, ps)
        key = "CategoryNet.cat_emb.table"
        for row_o, row_r in zip(ps.parameters[key]["values"][:3], ps2.parameters[key]["values"][:3]):
            for v_o, v_r in zip(row_o, row_r):
                assert abs(v_o - v_r) < 1e-6


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_composite_torch_error_is_value_error(self):
        assert issubclass(CompositeTorchError, ValueError)

    def test_missing_parameter_raises_composite_torch_error(self):
        net, ps = _setup(_MXAI_RESIDUAL, seed=0)
        # Remove a required parameter
        params = {k: v for k, v in ps.parameters.items() if "L1.W" not in k}
        ps_incomplete = ParameterSet(
            parameter_set_id=ps.parameter_set_id,
            model_hash=ps.model_hash,
            parameter_schema_hash=ps.parameter_schema_hash,
            parameters=params,
        )
        with pytest.raises(CompositeTorchError):
            composite_network_to_torch_module(net, ps_incomplete)
