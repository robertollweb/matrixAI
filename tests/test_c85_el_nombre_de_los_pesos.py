"""EL FICHERO DE PESOS SE LLAMA IGUAL POR LOS DOS CAMINOS (2026-08-25).

Medido el 2026-08-24 contra el paquete publicado: `matrixai train` sobre un
modelo **FUNCTION** deja `params.best.json` —y lo declara en su manifiesto como
`selected_parameter_set`—, y sobre una **RED densa** dejaba solo
`parameter_set.json`. Dos nombres para la misma cosa según por dónde entres.

No es que la documentación mintiera —el QUICKSTART usa la plantilla
`classification`, que es un FUNCTION, y su camino es correcto—: es que quien
entrena una RED (lo que produce el generador por prompt) y sigue esa
documentación se lleva un `No such file or directory`.

**Se escriben los dos y no se retira ninguno**: hay guardado y export que ya
buscan `parameter_set.json`, y renombrar rompería lo que funciona para arreglar
un nombre.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from matrixai.parser import parse_text
from matrixai.training.dense_trainer import DenseSupervisedTrainer
from matrixai.training.parser import parse_training_text

_MXAI = (
    "PROJECT P\n\nVECTOR E[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
    "NETWORK N\n  INPUT E\n  LAYER Dense units=4 activation=relu\n"
    "  LAYER Dense units=2 activation=softmax\n"
    "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
    "GRAPH\n  E -> N\nEND\n"
)
_MXTRAIN = (
    "MODEL modelo.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT E FROM COLUMNS [a, b]\n"
    "  TARGET predicted_class: Label[alto, bajo]\n"
    "  SPLIT train=0.75 validation=0.25 seed=42\n  BATCH size=4\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION N\n  TARGET predicted_class\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE N.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)
_CSV = "a,b,predicted_class\n" + "".join(
    f"{i/20:.2f},{1 - i/20:.2f},{'alto' if i % 2 else 'bajo'}\n" for i in range(20))


class LosDosNombresTest(unittest.TestCase):
    def _entrenar(self, taller: Path):
        (taller / "modelo.mxai").write_text(_MXAI, encoding="utf-8")
        (taller / "d.csv").write_text(_CSV, encoding="utf-8")
        ruta_train = taller / "modelo.mxtrain"
        ruta_train.write_text(_MXTRAIN, encoding="utf-8")
        spec = parse_training_text(_MXTRAIN)
        DenseSupervisedTrainer().train(spec, output_dir=str(taller / "out"),
                                       base_path=taller, training_path=ruta_train)
        return taller / "out"

    def test_una_RED_deja_los_dos_ficheros(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._entrenar(Path(d))
            self.assertTrue((out / "parameter_set.json").exists(), "el de siempre")
            self.assertTrue((out / "params.best.json").exists(),
                            "el que enseña la documentación")

    def test_y_son_EL_MISMO_contenido(self):
        """Dos nombres para el mismo fichero, no dos ficheros que se parecen:
        si divergieran, `run --params` daría un resultado distinto según cuál
        se cogiera, que es peor que el problema original."""
        with tempfile.TemporaryDirectory() as d:
            out = self._entrenar(Path(d))
            a = json.loads((out / "parameter_set.json").read_text(encoding="utf-8"))
            b = json.loads((out / "params.best.json").read_text(encoding="utf-8"))
            self.assertEqual(a, b)

    def test_el_modelo_se_puede_parsear_con_lo_que_deja(self):
        """Guarda contra un fichero escrito a medias: lo que se deja tiene que
        ser un ParameterSet legible, no un JSON cualquiera."""
        from matrixai.parameters.store import load_parameter_set
        with tempfile.TemporaryDirectory() as d:
            out = self._entrenar(Path(d))
            ps = load_parameter_set(str(out / "params.best.json"))
            self.assertTrue(getattr(ps, "values", None) or getattr(ps, "to_dict", None))
            parse_text(_MXAI)  # el modelo sigue siendo válido


if __name__ == "__main__":
    unittest.main()
