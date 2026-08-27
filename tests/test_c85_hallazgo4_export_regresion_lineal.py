"""HALLAZGO 4 — EL EJEMPLO QUE MEJOR ENSEÑA YA SE PUEDE EMPAQUETAR.

`matrixai export-bundle` sobre el Kelvin del propio repositorio fallaba:

    No exportable functions found in 'CelsiusToKelvin'.
    Supported: ['layer_call', 'sigmoid_linear', 'softmax_linear'].
    Found kinds: ['linear_regression']

Y el camino determinista del `prompt` produce **exactamente ese tipo**
(`FUNCTION … = linear(W1 * Reading + b1)`), así que quien llegaba por el CLI
entrenaba un modelo que el exportador no sabía exportar. Sin paquete no hay
`reproduce.json`, y sin él no hay nada que verificar.

Y la lección de cómo se arregló: **había TRES listas** diciendo qué se puede
exportar —`_SUPPORTED_KINDS`, la selección de funciones dentro de `export_onnx`
y otra en `equivalence.py`— y aparecieron de una en una, cada vez con el mismo
error y un comando distinto. Ahora es UNA sola y las demás la importan.
"""
from __future__ import annotations

import unittest
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory

_HAS = util.find_spec("onnx") is not None and util.find_spec("onnxruntime") is not None

_MXAI = (
    "PROJECT CelsiusToKelvin\n\n"
    "VECTOR Reading[1]\n  celsius\nEND\n\n"
    "PARAM W1 Vector[1]\nEND\n\n"
    "PARAM b1 Scalar\nEND\n\n"
    "FUNCTION PredictedKelvinModel\n"
    "  predicted_kelvin: Scalar = linear(W1 * Reading + b1)\nEND\n\n"
    "GRAPH\n  Reading -> PredictedKelvinModel\nEND\n"
)
#: Los pesos EXACTOS de la conversión: K = C + 273,15.
#:
#: El `ParameterSet` se construye con `build_initial_parameter_set` y se le
#: ponen los valores: fabricarlo a mano con `{"parameters": …}` le falta la
#: mitad de las claves (`parameter_set_id`, los dos hashes) y revienta con un
#: `KeyError` — otro fixture describiendo algo que no existe.
_VALORES = {"W1": [1.0], "b1": 273.15}


class UnaSolaListaDiceQueSeExportaTest(unittest.TestCase):
    def test_la_regresion_lineal_esta_soportada(self):
        from matrixai.export.onnx_exporter import _SUPPORTED_KINDS
        self.assertIn("linear_regression", _SUPPORTED_KINDS)

    def test_y_equivalence_usa_LA_MISMA_lista(self):
        """Tres listas separadas fue justo lo que hizo falta descubrir tres
        veces el mismo error."""
        from matrixai.export import equivalence, onnx_exporter
        self.assertIs(equivalence._SUPPORTED_KINDS, onnx_exporter._SUPPORTED_KINDS)


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class ElKelvinSeExportaYPREDICELoMismoTest(unittest.TestCase):
    def _exportar(self, d: Path):
        import json

        from matrixai.export import create_edge_bundle
        from matrixai.parameters.store import build_initial_parameter_set
        from matrixai.parser.parser import parse_text

        (d / "kelvin.mxai").write_text(_MXAI, encoding="utf-8")
        programa = parse_text(_MXAI)
        # Los valores viven en `parameters[nombre]["values"]`, no en un
        # `ps.values` — medido, porque suponerlo dio `AttributeError`.
        from matrixai.parameters import ParameterSet
        crudo = build_initial_parameter_set(programa).to_dict()
        for nombre, valor in _VALORES.items():
            crudo["parameters"][nombre]["values"] = valor
        ps = ParameterSet.from_dict(crudo)
        (d / "params.json").write_text(json.dumps(crudo), encoding="utf-8")
        return create_edge_bundle(programa, ps, mxai_path=d / "kelvin.mxai",
                                  params_path=d / "params.json", outdir=d / "bundle",
                                  force=True)

    def test_el_paquete_SALE(self):
        with TemporaryDirectory() as tmp:
            resultado = self._exportar(Path(tmp))
            self.assertTrue((Path(tmp) / "bundle" / "model.onnx").is_file())
        # Y la equivalencia con el forward de referencia se comprueba sola.
        self.assertTrue(resultado.equivalence_passed, resultado.equivalence_skipped_reason)

    def test_y_el_ONNX_convierte_de_verdad(self):
        """La comprobación que dijo Roberto que valía para este caso: se
        comprueba con una resta. 0 °C = 273,15 K."""
        import numpy as np
        import onnxruntime

        with TemporaryDirectory() as tmp:
            self._exportar(Path(tmp))
            sesion = onnxruntime.InferenceSession(
                str(Path(tmp) / "bundle" / "model.onnx"),
                providers=["CPUExecutionProvider"])
            nombre = sesion.get_inputs()[0].name
            salida = sesion.run(None, {nombre: np.array([[0.0], [100.0]], dtype=np.float32)})[0]
        self.assertAlmostEqual(float(salida[0]), 273.15, places=3)
        self.assertAlmostEqual(float(salida[1]), 373.15, places=3)

    def test_una_regresion_NO_lleva_etiquetas_de_clase(self):
        """Inventarle nombres de clase a un número sería peor que no poner
        ninguno."""
        import json
        with TemporaryDirectory() as tmp:
            self._exportar(Path(tmp))
            manifiesto = json.loads(
                (Path(tmp) / "bundle" / "model_manifest.json").read_text(encoding="utf-8"))
        self.assertIn(manifiesto.get("labels"), ([], None))

    def test_la_salida_NO_esta_aplastada_entre_0_y_1(self):
        """Es lo que la diferencia de `sigmoid_linear`: una sigmoide aquí
        convertiría 373,15 K en 1,0."""
        import numpy as np
        import onnxruntime

        with TemporaryDirectory() as tmp:
            self._exportar(Path(tmp))
            sesion = onnxruntime.InferenceSession(
                str(Path(tmp) / "bundle" / "model.onnx"),
                providers=["CPUExecutionProvider"])
            nombre = sesion.get_inputs()[0].name
            salida = sesion.run(None, {nombre: np.array([[1000.0]], dtype=np.float32)})[0]
        self.assertGreater(float(salida[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
