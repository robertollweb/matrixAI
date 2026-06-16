"""P18 audit fix tests — validate_parameter_set, runtime NETWORK, metrics, IR refs, CLI routing."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_net_and_ps(seed: int = 42):
    from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
    from matrixai.parameters.network_params import build_network_parameter_set

    layers = [
        DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4]),
        DenseLayerSpec(index=2, units=1, activation="linear", input_shape=[4], output_shape=[1]),
    ]
    net = NetworkSpec(name="Net", input="V", layers=layers, output="y", output_type_str="Scalar")
    ps = build_network_parameter_set(net, layers, "hash_audit", seed=seed)
    return net, ps


_MXAI_NET = """
PROJECT AuditTest

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK Net
  INPUT V
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> Net
END
"""

_MXAI_BINARY = """
PROJECT BinaryTest

VECTOR F[2]
  a: Scalar
  b: Scalar
END

NETWORK BinNet
  INPUT F
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT label: Probability
END

GRAPH
  F -> BinNet
END
"""

_MXAI_MULTICLASS = """
PROJECT MultiTest

VECTOR F[2]
  a: Scalar
  b: Scalar
END

NETWORK MCNet
  INPUT F
  LAYER Dense units=4 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT label: ProbabilityMap[A, B, C]
END

GRAPH
  F -> MCNet
END
"""


# ---------------------------------------------------------------------------
# Fix 1: TrainableParameter.initializer_override
# ---------------------------------------------------------------------------

class TestInitializerOverride:
    def test_weight_initializer_is_he_normal_for_relu(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        report = BackendContractAnalyzer().analyze(program)
        # Layer 1 is relu → W1 must use he_normal
        w1 = next(p for p in report.trainable_parameters if p.name == "W1")
        assert w1.initializer == "he_normal", f"expected he_normal, got {w1.initializer!r}"

    def test_bias_initializer_is_zeros(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        report = BackendContractAnalyzer().analyze(program)
        bias_params = [p for p in report.trainable_parameters if p.role == "bias"]
        for bp in bias_params:
            assert bp.initializer == "zeros"

    def test_linear_layer_uses_xavier_normal(self):
        from matrixai.compiler.backend_contract import BackendContractAnalyzer
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        report = BackendContractAnalyzer().analyze(program)
        # Last layer is linear → xavier_normal
        w2 = next(p for p in report.trainable_parameters if p.name == "W2")
        assert w2.initializer == "xavier_normal"

    def test_initializer_override_field_accessible(self):
        from matrixai.compiler.backend_contract import TrainableParameter
        tp = TrainableParameter(function="F", name="W", role="weights", initializer_override="custom_init")
        assert tp.initializer == "custom_init"

    def test_default_weight_without_override_is_deterministic_uniform(self):
        from matrixai.compiler.backend_contract import TrainableParameter
        tp = TrainableParameter(function="F", name="W", role="weights")
        assert tp.initializer == "deterministic_uniform"


# ---------------------------------------------------------------------------
# Fix 1b: validate_parameter_set schema hash compatibility
# ---------------------------------------------------------------------------

class TestValidateParameterSetCompatibility:
    def _build_compatible_ps(self):
        """Build a ParameterSet whose model_hash matches what validate_parameter_set computes."""
        from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
        from matrixai.parameters.network_params import build_network_parameter_set
        from matrixai.parameters.store import program_hash
        from matrixai.parser.parser import parse_text

        program = parse_text(_MXAI_NET)
        mhash = program_hash(program)
        layers = [
            DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4]),
            DenseLayerSpec(index=2, units=1, activation="linear", input_shape=[4], output_shape=[1]),
        ]
        net = NetworkSpec(name="Net", input="V", layers=layers, output="y", output_type_str="Scalar")
        ps = build_network_parameter_set(net, layers, mhash, seed=42)
        return program, ps

    def test_validate_parameter_set_compatible_with_dense_network(self):
        from matrixai.parameters.store import validate_parameter_set
        program, ps = self._build_compatible_ps()
        result = validate_parameter_set(program, ps)
        assert result.ok, f"validate_parameter_set failed: {result.errors}"

    def test_validate_parameter_set_schema_hashes_match(self):
        from matrixai.parameters.store import validate_parameter_set
        program, ps = self._build_compatible_ps()
        result = validate_parameter_set(program, ps)
        assert not result.errors, f"errors: {result.errors}"


# ---------------------------------------------------------------------------
# Fix 2: Runtime NETWORK execution
# ---------------------------------------------------------------------------

class TestRuntimeNetworkExecution:
    def test_runtime_executes_network_node(self):
        from matrixai.runtime.runtime import MatrixAIRuntime
        from matrixai.parser.parser import parse_text
        net, ps = _build_net_and_ps()
        program = parse_text(_MXAI_NET)
        # Build raw params in flat format (as runtime_parameters() returns)
        raw = {}
        for key, val in ps.parameters.items():
            raw[key] = val["values"] if isinstance(val, dict) and "values" in val else val
        result = MatrixAIRuntime().run(program, {"V": {"x1": 0.5, "x2": 0.3}}, parameters=raw)
        assert "Net" in result["state"], "NETWORK node not in state"

    def test_runtime_network_output_is_list(self):
        from matrixai.runtime.runtime import MatrixAIRuntime
        from matrixai.parser.parser import parse_text
        net, ps = _build_net_and_ps()
        program = parse_text(_MXAI_NET)
        raw = {k: v["values"] if isinstance(v, dict) and "values" in v else v
               for k, v in ps.parameters.items()}
        result = MatrixAIRuntime().run(program, {"V": {"x1": 0.1, "x2": 0.9}}, parameters=raw)
        assert isinstance(result["state"]["Net"], list)

    def test_runtime_network_trace_has_dense_network_step(self):
        from matrixai.runtime.runtime import MatrixAIRuntime
        from matrixai.parser.parser import parse_text
        net, ps = _build_net_and_ps()
        program = parse_text(_MXAI_NET)
        raw = {k: v["values"] if isinstance(v, dict) and "values" in v else v
               for k, v in ps.parameters.items()}
        result = MatrixAIRuntime().run(program, {"V": {"x1": 0.5, "x2": 0.5}}, parameters=raw)
        types = [step["node_type"] for step in result["trace"]]
        assert "dense_network" in types

    def test_runtime_network_output_bound_to_output_field(self):
        from matrixai.runtime.runtime import MatrixAIRuntime
        from matrixai.parser.parser import parse_text
        net, ps = _build_net_and_ps()
        program = parse_text(_MXAI_NET)
        raw = {k: v["values"] if isinstance(v, dict) and "values" in v else v
               for k, v in ps.parameters.items()}
        result = MatrixAIRuntime().run(program, {"V": {"x1": 0.5, "x2": 0.5}}, parameters=raw)
        assert "y" in result["state"], "output field 'y' not in state"
        assert result["state"]["y"] == result["state"]["Net"]

    def test_runtime_skips_network_gracefully_without_params(self):
        from matrixai.runtime.runtime import MatrixAIRuntime
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        # No params → dense_forward will raise DenseForwardError
        with pytest.raises(Exception):
            MatrixAIRuntime().run(program, {"V": {"x1": 0.5, "x2": 0.3}}, parameters={})


# ---------------------------------------------------------------------------
# Fix 3: precision/recall/f1/macro_f1 in DenseEvaluationResult
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def _binary_examples(self):
        from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
        from matrixai.parameters.network_params import build_network_parameter_set
        layers = [
            DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4]),
            DenseLayerSpec(index=2, units=1, activation="sigmoid", input_shape=[4], output_shape=[1]),
        ]
        net = NetworkSpec(name="BinNet", input="F", layers=layers, output="label", output_type_str="Probability")
        ps = build_network_parameter_set(net, layers, "hash_bin", seed=1)
        examples = [([float(i % 2), float(i % 3)], [float(i % 2)]) for i in range(10)]
        return net, ps, examples

    def test_binary_evaluation_has_precision(self):
        from matrixai.training.dense_evaluator import evaluate_dense_network
        net, ps, examples = self._binary_examples()
        result = evaluate_dense_network(net, ps, examples, "binary_cross_entropy", labels=["neg", "pos"])
        assert hasattr(result, "precision")
        assert "pos" in result.precision or "neg" in result.precision

    def test_binary_evaluation_has_recall(self):
        from matrixai.training.dense_evaluator import evaluate_dense_network
        net, ps, examples = self._binary_examples()
        result = evaluate_dense_network(net, ps, examples, "binary_cross_entropy", labels=["neg", "pos"])
        assert hasattr(result, "recall")

    def test_binary_evaluation_has_f1(self):
        from matrixai.training.dense_evaluator import evaluate_dense_network
        net, ps, examples = self._binary_examples()
        result = evaluate_dense_network(net, ps, examples, "binary_cross_entropy", labels=["neg", "pos"])
        assert hasattr(result, "f1")

    def test_binary_evaluation_has_macro_f1(self):
        from matrixai.training.dense_evaluator import evaluate_dense_network
        net, ps, examples = self._binary_examples()
        result = evaluate_dense_network(net, ps, examples, "binary_cross_entropy", labels=["neg", "pos"])
        assert hasattr(result, "macro_f1")
        assert 0.0 <= result.macro_f1 <= 1.0

    def test_multiclass_evaluation_has_per_class_metrics(self):
        from matrixai.ir.schema import DenseLayerSpec, NetworkSpec
        from matrixai.parameters.network_params import build_network_parameter_set
        from matrixai.training.dense_evaluator import evaluate_dense_network
        layers = [
            DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4]),
            DenseLayerSpec(index=2, units=3, activation="softmax", input_shape=[4], output_shape=[3]),
        ]
        net = NetworkSpec(name="MCNet", input="F", layers=layers, output="label", output_type_str="ProbabilityMap[A,B,C]")
        ps = build_network_parameter_set(net, layers, "hash_mc", seed=2)
        examples = [([float(i % 3), float(i)], [1.0 if i % 3 == j else 0.0 for j in range(3)]) for i in range(9)]
        result = evaluate_dense_network(net, ps, examples, "cross_entropy", labels=["A", "B", "C"])
        assert "A" in result.precision
        assert "B" in result.f1
        assert result.macro_f1 >= 0.0

    def test_to_dict_includes_precision_recall_f1(self):
        from matrixai.training.dense_evaluator import evaluate_dense_network
        net, ps, examples = self._binary_examples()
        result = evaluate_dense_network(net, ps, examples, "binary_cross_entropy", labels=["neg", "pos"])
        d = result.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert "macro_f1" in d

    def test_regression_does_not_include_classification_metrics(self):
        from matrixai.training.dense_evaluator import evaluate_dense_network
        net, ps = _build_net_and_ps()
        examples = [([0.5, 0.3], [1.0]), ([0.1, 0.9], [0.5])]
        result = evaluate_dense_network(net, ps, examples, "mse")
        d = result.to_dict()
        assert "mae" in d
        assert "precision" not in d


# ---------------------------------------------------------------------------
# Fix 4: IR serialization includes parameter references per layer
# ---------------------------------------------------------------------------

class TestIRParameterRefs:
    def test_network_to_dict_has_parameters_per_layer(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        d = program.to_dict()
        net_dict = d["networks"][0]
        for layer in net_dict["layers"]:
            assert "parameters" in layer, f"layer {layer['index']} missing parameters"
            assert "weights" in layer["parameters"]
            assert "bias" in layer["parameters"]

    def test_parameter_refs_use_correct_keys(self):
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        d = program.to_dict()
        net_dict = d["networks"][0]
        for layer in net_dict["layers"]:
            idx = layer["index"]
            assert layer["parameters"]["weights"] == f"Net.W{idx}"
            assert layer["parameters"]["bias"] == f"Net.b{idx}"


# ---------------------------------------------------------------------------
# Fix 5: DenseSupervisedTrainer imports and interface
# ---------------------------------------------------------------------------

class TestDenseSupervisedTrainer:
    def test_dense_supervised_trainer_importable(self):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        assert DenseSupervisedTrainer is not None

    def test_dense_supervised_evaluator_importable(self):
        from matrixai.training.dense_trainer import DenseSupervisedEvaluator
        assert DenseSupervisedEvaluator is not None

    def test_dense_trainer_exported_from_training(self):
        from matrixai.training import DenseSupervisedTrainer, DenseSupervisedEvaluator
        assert DenseSupervisedTrainer is not None
        assert DenseSupervisedEvaluator is not None

    def test_dense_trainer_train_method_exists(self):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        assert hasattr(DenseSupervisedTrainer(), "train")

    def test_dense_evaluator_evaluate_method_exists(self):
        from matrixai.training.dense_trainer import DenseSupervisedEvaluator
        assert hasattr(DenseSupervisedEvaluator(), "evaluate")


# ---------------------------------------------------------------------------
# Fix 5b: initial_value respects initializer_override
# ---------------------------------------------------------------------------

class TestInitialValueRespectOverride:
    def test_initial_value_uses_he_normal_when_override_set(self):
        from matrixai.compiler.backend_contract import TrainableParameter
        tp = TrainableParameter(
            function="Net", name="W1", role="weights",
            shape=(4, 2), initializer_override="he_normal",
        )
        val = tp.initial_value
        assert isinstance(val, list), "expected list for matrix shape"
        assert len(val) == 4
        assert len(val[0]) == 2
        # All values should be floats (not the deterministic_uniform pattern)
        flat = [v for row in val for v in row]
        assert all(isinstance(v, float) for v in flat)

    def test_initial_value_uses_xavier_normal_when_override_set(self):
        from matrixai.compiler.backend_contract import TrainableParameter
        tp = TrainableParameter(
            function="Net", name="W1", role="weights",
            shape=(4, 2), initializer_override="xavier_normal",
        )
        val = tp.initial_value
        assert isinstance(val, list)

    def test_initial_value_deterministic_without_override(self):
        from matrixai.compiler.backend_contract import TrainableParameter
        tp = TrainableParameter(function="F", name="W", role="weights", shape=(2, 2))
        val = tp.initial_value
        assert isinstance(val, list)
        # deterministic_uniform produces same result each call
        assert tp.initial_value == val

    def test_he_vs_xavier_initial_values_differ(self):
        from matrixai.compiler.backend_contract import TrainableParameter
        he = TrainableParameter(
            function="Net", name="W", role="weights",
            shape=(4, 4), initializer_override="he_normal",
        )
        xav = TrainableParameter(
            function="Net", name="W", role="weights",
            shape=(4, 4), initializer_override="xavier_normal",
        )
        assert he.initial_value != xav.initial_value


# ---------------------------------------------------------------------------
# Fix 6: output_name in network_parameter_schema_hash
# ---------------------------------------------------------------------------

class TestSchemaHashOutputName:
    def test_schema_hash_without_output_name_stable(self):
        from matrixai.ir.schema import DenseLayerSpec
        from matrixai.parameters.network_params import network_parameter_schema_hash
        layers = [DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4])]
        h1 = network_parameter_schema_hash("Net", layers)
        h2 = network_parameter_schema_hash("Net", layers)
        assert h1 == h2

    def test_schema_hash_changes_with_output_name(self):
        from matrixai.ir.schema import DenseLayerSpec
        from matrixai.parameters.network_params import network_parameter_schema_hash
        layers = [DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4])]
        h_base = network_parameter_schema_hash("Net", layers)
        h_out_a = network_parameter_schema_hash("Net", layers, output_name="price")
        h_out_b = network_parameter_schema_hash("Net", layers, output_name="revenue")
        assert h_base != h_out_a
        assert h_out_a != h_out_b

    def test_schema_hash_same_output_name_stable(self):
        from matrixai.ir.schema import DenseLayerSpec
        from matrixai.parameters.network_params import network_parameter_schema_hash
        layers = [DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4])]
        h1 = network_parameter_schema_hash("Net", layers, output_name="price")
        h2 = network_parameter_schema_hash("Net", layers, output_name="price")
        assert h1 == h2

    def test_schema_hash_starts_with_params(self):
        from matrixai.ir.schema import DenseLayerSpec
        from matrixai.parameters.network_params import network_parameter_schema_hash
        layers = [DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4])]
        h = network_parameter_schema_hash("Net", layers, output_name="y")
        assert h.startswith("params_")


# ---------------------------------------------------------------------------
# Fix 7: Real integration test — DenseSupervisedTrainer with temp CSV
# ---------------------------------------------------------------------------

class TestDenseSupervisedTrainerIntegration:
    def _write_mxai(self, tmp_path):
        """Write a minimal .mxai file for regression."""
        mxai = tmp_path / "model.mxai"
        mxai.write_text("""
PROJECT Test

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK Net
  INPUT V
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> Net
END
""", encoding="utf-8")
        return mxai

    def _write_mxtrain(self, tmp_path, mxai_path, csv_path):
        """Write a minimal .mxtrain file."""
        mxtrain = tmp_path / "train.mxtrain"
        mxtrain.write_text(f"""MODEL {mxai_path.name}

DATASET TrainData
  SOURCE csv("{csv_path.name}")
  INPUT V FROM COLUMNS [x1, x2]
  TARGET y: Scalar
END

LOSS NetLoss
  TYPE mse
  PREDICTION y
  TARGET y
END

OPTIMIZER NetOpt
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END

RUN
  EPOCHS 5
END
""", encoding="utf-8")
        return mxtrain

    def _write_csv(self, tmp_path):
        """Write 20 simple regression examples: y = x1 + x2."""
        csv_path = tmp_path / "data.csv"
        rows = ["x1,x2,y"]
        for i in range(20):
            x1 = round(i * 0.1, 2)
            x2 = round((20 - i) * 0.05, 2)
            y = round(x1 + x2, 4)
            rows.append(f"{x1},{x2},{y}")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return csv_path

    def test_dense_trainer_runs_and_returns_result(self, tmp_path):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        from matrixai.training.parser import parse_training_file

        mxai = self._write_mxai(tmp_path)
        csv_path = self._write_csv(tmp_path)
        mxtrain = self._write_mxtrain(tmp_path, mxai, csv_path)
        out_dir = tmp_path / "output"

        training = parse_training_file(mxtrain)
        trainer = DenseSupervisedTrainer()
        result = trainer.train(training, output_dir=str(out_dir), base_path=tmp_path)

        assert result.run_id
        assert result.best_epoch >= 1
        assert isinstance(result.best_validation_loss, float)
        assert (out_dir / "parameter_set.json").exists()
        assert (out_dir / "training_trace.json").exists()

    def test_dense_trainer_parameter_set_has_correct_model_hash(self, tmp_path):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        from matrixai.training.parser import parse_training_file
        from matrixai.parameters.store import program_hash, validate_parameter_set, load_parameter_set
        from matrixai.parser import parse_file

        mxai = self._write_mxai(tmp_path)
        csv_path = self._write_csv(tmp_path)
        mxtrain = self._write_mxtrain(tmp_path, mxai, csv_path)
        out_dir = tmp_path / "output"

        training = parse_training_file(mxtrain)
        trainer = DenseSupervisedTrainer()
        result = trainer.train(training, output_dir=str(out_dir), base_path=tmp_path)

        # Load the saved ParameterSet and validate it against the program
        ps = load_parameter_set(out_dir / "parameter_set.json")
        program = parse_file(mxai)
        expected_hash = program_hash(program)
        assert ps.model_hash == expected_hash, f"model_hash mismatch: {ps.model_hash!r} != {expected_hash!r}"

    def test_dense_trainer_parameter_set_passes_validate(self, tmp_path):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        from matrixai.training.parser import parse_training_file
        from matrixai.parameters.store import validate_parameter_set, load_parameter_set
        from matrixai.parser import parse_file

        mxai = self._write_mxai(tmp_path)
        csv_path = self._write_csv(tmp_path)
        mxtrain = self._write_mxtrain(tmp_path, mxai, csv_path)
        out_dir = tmp_path / "output"

        training = parse_training_file(mxtrain)
        DenseSupervisedTrainer().train(training, output_dir=str(out_dir), base_path=tmp_path)

        ps = load_parameter_set(out_dir / "parameter_set.json")
        program = parse_file(mxai)
        compat = validate_parameter_set(program, ps)
        assert compat.ok, f"validate_parameter_set failed: {compat.errors}"


# ---------------------------------------------------------------------------
# Fix 8: CLI imports DenseSupervisedTrainer without error
# ---------------------------------------------------------------------------

class TestCLIImports:
    def test_cli_imports_dense_trainer(self):
        import matrixai.cli  # should not raise
        assert hasattr(matrixai.cli, "DenseSupervisedTrainer")

    def test_cli_imports_dense_evaluator(self):
        import matrixai.cli
        assert hasattr(matrixai.cli, "DenseSupervisedEvaluator")


# ---------------------------------------------------------------------------
# Fix 9 (Alto): VerifierAgent declares NETWORK nodes — validate passes
# ---------------------------------------------------------------------------

class TestVerifierNetworkDeclaration:
    def test_verifier_ok_for_network_model(self):
        from matrixai.agents.verifier import VerifierAgent
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        result = VerifierAgent().verify(program)
        assert result.ok, f"VerifierAgent raised errors: {result.errors}"

    def test_verifier_no_undeclared_node_error_for_network(self):
        from matrixai.agents.verifier import VerifierAgent
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        result = VerifierAgent().verify(program)
        undeclared = [e for e in result.errors if "undeclared" in e and "Net" in e]
        assert not undeclared, f"Unexpected undeclared-node errors: {undeclared}"

    def test_verifier_declared_nodes_includes_network_names(self):
        from matrixai.agents.verifier import VerifierAgent
        from matrixai.parser.parser import parse_text
        program = parse_text(_MXAI_NET)
        agent = VerifierAgent()
        declared = agent._declared_nodes(program)
        assert "Net" in declared


# ---------------------------------------------------------------------------
# Fix 10 (Medio): BCE uses target_value for Probability numeric targets
# ---------------------------------------------------------------------------

class TestBCENumericProbabilityTarget:
    def _make_example(self, target_value=None, label="pos"):
        from matrixai.training.data import SupervisedExample
        return SupervisedExample(
            vector=[0.5, 0.3],
            label=label,
            row_index=0,
            row_hash="abc",
            target_value=target_value,
        )

    def test_bce_numeric_target_0_9_maps_to_0_9(self):
        from matrixai.training.dense_trainer import _examples_to_xy
        ex = self._make_example(target_value=0.9)
        result = _examples_to_xy([ex], "binary_cross_entropy", [])
        assert result[0][1] == [0.9], f"expected [0.9], got {result[0][1]}"

    def test_bce_numeric_target_0_0_maps_to_0_0(self):
        from matrixai.training.dense_trainer import _examples_to_xy
        ex = self._make_example(target_value=0.0)
        result = _examples_to_xy([ex], "binary_cross_entropy", [])
        assert result[0][1] == [0.0], f"expected [0.0], got {result[0][1]}"

    def test_bce_label_fallback_when_no_target_value(self):
        from matrixai.training.dense_trainer import _examples_to_xy
        ex = self._make_example(target_value=None, label="pos")
        result = _examples_to_xy([ex], "binary_cross_entropy", ["neg", "pos"])
        assert result[0][1] == [1.0], f"expected [1.0], got {result[0][1]}"


# ---------------------------------------------------------------------------
# Fix 11 (Medio): output_name wired into build/validate_network_parameter_set
# ---------------------------------------------------------------------------

class TestOutputNameWiredInBuildPS:
    def _layers(self):
        from matrixai.ir.schema import DenseLayerSpec
        return [
            DenseLayerSpec(index=1, units=4, activation="relu", input_shape=[2], output_shape=[4]),
            DenseLayerSpec(index=2, units=1, activation="linear", input_shape=[4], output_shape=[1]),
        ]

    def _net(self, output="y"):
        from matrixai.ir.schema import NetworkSpec
        return NetworkSpec(
            name="Net", input="V", layers=self._layers(),
            output=output, output_type_str="Scalar",
        )

    def test_different_output_name_gives_different_schema_hash(self):
        from matrixai.parameters.network_params import build_network_parameter_set
        layers = self._layers()
        ps_y = build_network_parameter_set(self._net("y"), layers, "h", output_name="y")
        ps_z = build_network_parameter_set(self._net("z"), layers, "h", output_name="z")
        assert ps_y.parameter_schema_hash != ps_z.parameter_schema_hash

    def test_same_output_name_gives_same_schema_hash(self):
        from matrixai.parameters.network_params import build_network_parameter_set
        layers = self._layers()
        ps1 = build_network_parameter_set(self._net("y"), layers, "h", output_name="y")
        ps2 = build_network_parameter_set(self._net("y"), layers, "h", output_name="y")
        assert ps1.parameter_schema_hash == ps2.parameter_schema_hash

    def test_validate_network_parameter_set_consistent_with_output_name(self):
        from matrixai.parameters.network_params import (
            build_network_parameter_set,
            validate_network_parameter_set,
        )
        layers = self._layers()
        net = self._net("y")
        ps = build_network_parameter_set(net, layers, "mhash", output_name="y")
        result = validate_network_parameter_set(net, layers, ps, "mhash", output_name="y")
        assert result.ok, f"validate failed: {result.errors}"
