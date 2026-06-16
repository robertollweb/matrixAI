import unittest
from unittest.mock import Mock

# Usamos Mock u objetos genéricos para simular la IR en caso de que cambien los required args
from matrixai.ir.schema import LayerSpec, ParameterSpec, LayerBodyOp
from matrixai.compiler.backend_contract import _analyze_layer_param_usage

class TestLayerLivenessAnalysis(unittest.TestCase):

    def test_all_params_used(self):
        layer = LayerSpec(
            name="Dense",
            input_type=None,
            output_type=None,
            params=[Mock(name="W"), Mock(name="b")],
            body_ops=(
                LayerBodyOp(output="proj", kind="matmul", args=("Input", "W")),
                LayerBodyOp(output="result", kind="add", args=("proj", "b")),
            )
        )
        # Configuramos el nombre simulado para los Mocks
        layer.params[0].name = "W"
        layer.params[1].name = "b"
        
        usage = _analyze_layer_param_usage(layer)
        self.assertTrue(usage["W"])
        self.assertTrue(usage["b"])

    def test_completely_dead_param(self):
        layer = LayerSpec(
            name="Attention",
            input_type=None,
            output_type=None,
            params=[Mock(name="Wq"), Mock(name="dead_param")],
            body_ops=(
                LayerBodyOp(output="q", kind="matmul", args=("Input", "Wq")),
                LayerBodyOp(output="result", kind="softmax", args=("q",)),
            )
        )
        layer.params[0].name = "Wq"
        layer.params[1].name = "dead_param"
        
        usage = _analyze_layer_param_usage(layer)
        self.assertTrue(usage["Wq"])
        self.assertFalse(usage["dead_param"])

    def test_used_in_disconnected_branch(self):
        layer = LayerSpec(
            name="ComplexLayer",
            input_type=None,
            output_type=None,
            params=[Mock(name="W"), Mock(name="debug_scale")],
            body_ops=(
                LayerBodyOp(output="proj", kind="matmul", args=("Input", "W")),
                LayerBodyOp(output="debug_val", kind="mul", args=("proj", "debug_scale")),
                LayerBodyOp(output="result", kind="relu", args=("proj",)),
            )
        )
        layer.params[0].name = "W"
        layer.params[1].name = "debug_scale"
        
        usage = _analyze_layer_param_usage(layer)
        self.assertTrue(usage["W"])
        self.assertFalse(usage["debug_scale"])

if __name__ == '__main__':
    unittest.main()