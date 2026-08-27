"""HALLAZGO 13 — NORMALIZAR POR LOS RANGOS QUE EL MODELO DECLARA.

Decisión de Roberto (2026-08-25): **paridad con el Studio**. El `.mxai` declara
el dominio de cada entrada y el `.mxtrain` el del objetivo, y el camino del CLI
no los miraba. Medido antes y después, conduciendo:

| Caso | Antes | Después |
|---|---|---|
| clínico (`edad: Scalar[18, 100]`) | exactitud **0,687** | **0,969** |
| Kelvin (`celsius: Scalar[-50, 150]`) | **no converge** (val loss 17.000-37.000) | val loss **0,000000** |
| Kelvin, el paquete exportado, 0 °C | — | **273,150001** |

Y LO QUE DE VERDAD ESTABA ROTO no era la exactitud: `predict.py` del paquete
**ya normalizaba** las entradas y **desnormalizaba** la salida con esos mismos
rangos. O sea que un modelo entrenado con valores crudos se exportaba a un
paquete que lo alimentaba con valores escalados. Medido con el paquete viejo:
para «25 años, sin reingreso previo» decía **alto: 0,67**; el nuevo dice
**bajo: 0,9998**.

Tres reglas que salieron de hacerlo:

1. **Las dos mitades o ninguna.** Normalizar solo las entradas hacía DIVERGIR al
   Kelvin a tasas a las que antes no divergía: media normalización es peor que
   ninguna.
2. **Lo que no declara rango no se toca.** Suponer `[0, 1]` escalaría una
   columna que quizá ya viene en su escala.
3. **En clasificación la etiqueta NO se escala**: es un índice de clase, y
   escalarlo sería destruirlo.
"""
from __future__ import annotations

import unittest

from matrixai.parser.parser import parse_text
from matrixai.training.normalizacion import (
    normalizar_filas,
    rango_declarado_del_objetivo,
    rangos_declarados_del_vector,
)
from matrixai.training.parser import parse_training_text


def _vector(con_rangos: bool):
    rango = "[18, 100]" if con_rangos else ""
    programa = parse_text(
        f"PROJECT P\n\nVECTOR E[2]\n  edad: Scalar{rango}\n  otro: Scalar\nEND\n\n"
        "NETWORK N\n  INPUT E\n  LAYER Dense units=2 activation=softmax\n"
        "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
        "GRAPH\n  E -> N\nEND\n")
    return programa.vectors[0]


def _training(tipo_objetivo: str, loss: str = "mse"):
    return parse_training_text(
        "MODEL p.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
        "  INPUT E FROM COLUMNS [edad, otro]\n"
        f"  TARGET y: {tipo_objetivo}\n"
        "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
        f"LOSS L\n  TYPE {loss}\n  PREDICTION N\n  TARGET y\nEND\n\n"
        "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE N.*\nEND\n\n"
        "RUN\n  EPOCHS 2\nEND\n")


class LosRangosDeclaradosSeLeenTest(unittest.TestCase):
    def test_del_vector_sale_lo_que_declara(self):
        self.assertEqual(rangos_declarados_del_vector(_vector(True)), {"edad": (18.0, 100.0)})

    def test_lo_que_NO_declara_rango_no_entra(self):
        """Suponer [0, 1] escalaría una columna que quizá ya viene en su
        escala, y una suposición silenciosa en el camino de los datos es lo que
        este hallazgo vino a quitar."""
        self.assertEqual(rangos_declarados_del_vector(_vector(False)), {})

    def test_un_rango_al_reves_o_degenerado_no_se_usa(self):
        """Con `maximo <= minimo` la división reventaría o daría infinito.
        `RangeSpec` es `frozen` —la cuarta dataclase congelada del día—, así que
        el caso se construye, no se retoca."""
        import dataclasses

        vector = _vector(True)
        tipo = vector.field_types["edad"]
        vector.field_types["edad"] = dataclasses.replace(
            tipo, range=dataclasses.replace(tipo.range, maximum=18.0))
        self.assertEqual(rangos_declarados_del_vector(vector), {})

    def test_del_objetivo_solo_si_es_continuo(self):
        self.assertEqual(rango_declarado_del_objetivo(_training("Scalar[200, 450]")),
                         (200.0, 450.0))

    def test_en_CLASIFICACION_la_etiqueta_no_se_escala(self):
        """Es un índice de clase: escalarlo sería destruirlo."""
        self.assertIsNone(rango_declarado_del_objetivo(
            _training("Label[alto, bajo]", loss="cross_entropy")))

    def test_sin_rango_declarado_el_objetivo_no_se_toca(self):
        self.assertIsNone(rango_declarado_del_objetivo(_training("Scalar")))


class NormalizarFilasTest(unittest.TestCase):
    def test_escala_solo_las_columnas_con_dominio(self):
        filas, aplicados = normalizar_filas(
            [[59.0, 7.0], [100.0, 9.0]], ["edad", "otro"], {"edad": (18.0, 100.0)})
        self.assertAlmostEqual(filas[0][0], (59 - 18) / 82)
        self.assertEqual(filas[0][1], 7.0)          # sin dominio: intacta
        self.assertEqual(aplicados, {"edad": (18.0, 100.0)})

    def test_devuelve_lo_APLICADO_para_que_el_export_lo_repita(self):
        """Normalizar al entrenar y no al predecir produce un paquete que
        parece bueno y predice mal: por eso esto se devuelve en vez de darse
        por sabido."""
        _, aplicados = normalizar_filas([[1.0]], ["edad"], {"edad": (0.0, 10.0)})
        self.assertEqual(aplicados, {"edad": (0.0, 10.0)})

    def test_sin_dominios_no_toca_nada_y_lo_dice(self):
        filas, aplicados = normalizar_filas([[59.0]], ["edad"], {})
        self.assertEqual(filas, [[59.0]])
        self.assertEqual(aplicados, {})

    def test_una_fila_mas_corta_que_los_campos_no_revienta(self):
        filas, _ = normalizar_filas([[59.0]], ["edad", "otro"],
                                    {"edad": (0.0, 100.0), "otro": (0.0, 10.0)})
        self.assertAlmostEqual(filas[0][0], 0.59)


class ElEntrenadorLasAPLICATest(unittest.TestCase):
    def test_el_camino_FUNCTION_normaliza_entradas_y_objetivo(self):
        from matrixai.training.trainer import _normalizar_ejemplos
        from matrixai.training.trainer import TrainingExample

        ejemplos = [TrainingExample(vector=[59.0, 7.0], label="325.0",
                                    row_index=0, row_hash="h", target_value=325.0)]
        salida = _normalizar_ejemplos(ejemplos, _vector(True), _training("Scalar[200, 450]"))
        self.assertAlmostEqual(salida[0].vector[0], (59 - 18) / 82)
        self.assertAlmostEqual(salida[0].label, (325 - 200) / 250)
        # `TrainingExample` es `frozen`: normalizar aquí es CONSTRUIR, no asignar.
        self.assertEqual(ejemplos[0].vector, [59.0, 7.0])

    def test_sin_nada_declarado_devuelve_los_MISMOS_ejemplos(self):
        from matrixai.training.trainer import _normalizar_ejemplos
        from matrixai.training.trainer import TrainingExample

        ejemplos = [TrainingExample(vector=[59.0, 7.0], label="1.0",
                                    row_index=0, row_hash="h", target_value=1.0)]
        self.assertIs(_normalizar_ejemplos(ejemplos, _vector(False), _training("Scalar")),
                      ejemplos)


if __name__ == "__main__":
    unittest.main()
