# SPDX-License-Identifier: AGPL-3.0-only
"""PESOS_GRANDES C1 — estimador de recursos ANTES de entrenar.

Ver 48_PESOS_GRANDES_CONTRATO.md. Verifica: param_count exacto contra shapes
conocidos (el prompt gigante de misPromts.md 12.1, 16x16384), monotonía de las
fórmulas con las constantes nombradas, y coherencia con lo observado en Colab
(pesos ~15 GiB, tiempo de guardado json en minutos-decenas, binario en segundos).
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from matrixai.parser import parse_text
from matrixai.resources import (
    ASSUMED_DISK_WRITE_BYTES_PER_SECOND,
    JSON_BYTES_PER_VALUE,
    JSON_DUMP_VALUES_PER_SECOND,
    PYTHON_LIST_BYTES_PER_VALUE,
    TOLIST_VALUES_PER_SECOND,
    VRAM_MARGIN_FACTOR,
    estimate_model_resources,
)
from matrixai.training.dense_generator import DenseNetworkGenerator


# El prompt 12.1 de misPromts.md — 16 capas x 16384 unidades, ProbabilityMap[2].
_GIANT_PROMPT = (
    "red densa pura de 16 capas y 16384 unidades, entrena 10 epocas, "
    "para predecir el riesgo de recaida de cancer en pacientes con antecedentes previos\n"
    "FEATURES:\n"
    "  edad: Scalar en [18, 95]\n"
    "  num_tumores_previos: Integer[1, 10]\n"
    "  meses_desde_remision: Integer[0, 240]\n"
    "  marcador_tumoral: Scalar en [0, 500]\n"
    "  quimioterapia_previa: Boolean\n"
    "  radioterapia_previa: Boolean\n"
    "SALIDA: recaida: ProbabilityMap[NO, SI]"
)
_GIANT_PARAM_COUNT = 4_026_925_058


def _program_for(prompt: str):
    gen = DenseNetworkGenerator().generate(prompt)
    return parse_text(gen.mxai_text)


def _giant_program():
    # perfil "ilimitado": el prompt pide 16 capas, por encima del tope de
    # profundidad (12) del perfil equilibrado por defecto — igual que en Colab
    # (colab_studio.py arranca en MATRIXAI_LIMITS_PROFILE=ilimitado).
    with patch.dict(os.environ, {"MATRIXAI_LIMITS_PROFILE": "ilimitado"}):
        return _program_for(_GIANT_PROMPT)


class ParamCountExactTest(unittest.TestCase):
    def test_giant_prompt_param_count_matches_colab_observation(self) -> None:
        program = _giant_program()
        est = estimate_model_resources(program, rows=4000, device="cuda")
        self.assertEqual(est.param_count, _GIANT_PARAM_COUNT)

    def test_small_network_param_count_matches_hand_formula(self) -> None:
        # 2 capas x 8 unidades, input_dim=3 (a,b,c), sigmoide binario (U=1).
        program = _program_for(
            "detectar fraude\nFEATURES:\n  a: Scalar\n  b: Scalar\n  c: Scalar\n"
        )
        est = estimate_model_resources(program, rows=100, device="cpu")
        # formula cerrada: d*W + W + (L-1)*(W*W+W) + W*U + U, leída del propio
        # mxai generado (no asumida) para no acoplar el test al tapering por defecto.
        net = program.networks[0]
        vector = program.vectors[0]
        d = len(vector.fields)
        dims = [d] + [layer.units for layer in net.layers]
        expected = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
        self.assertEqual(est.param_count, expected)


class MonotonicityTest(unittest.TestCase):
    """Las fórmulas crecen con params/batch, y usan las constantes nombradas
    (no números mágicos): cambiar la constante cambia el resultado 1:1."""

    def _small_program(self):
        return _program_for("clasificar\nFEATURES:\n  a: Scalar\n  b: Scalar\n")

    def test_vram_grows_with_batch(self) -> None:
        # CPU respeta el batch pedido (CUDA fuerza un batch grande por diseño,
        # effective_batch_size — no monótono ahí a propósito).
        program = self._small_program()
        small = estimate_model_resources(program, rows=100000, batch=32, device="cpu")
        large = estimate_model_resources(program, rows=100000, batch=2048, device="cpu")
        self.assertLess(small.vram_train_gib, large.vram_train_gib)
        self.assertEqual(small.effective_batch, 32)
        self.assertEqual(large.effective_batch, 2048)

    def test_json_costs_exceed_binary_costs(self) -> None:
        program = self._small_program()
        est = estimate_model_resources(program, rows=1000)
        self.assertGreater(est.json_ram_gib, est.binary_ram_gib)
        self.assertGreater(est.json_disk_gib, est.binary_disk_gib)
        self.assertGreater(est.json_time_seconds, est.binary_time_seconds)

    def test_named_constants_drive_the_formula(self) -> None:
        import matrixai.resources as res

        program = self._small_program()
        est = estimate_model_resources(program, rows=1000)
        n = est.param_count
        self.assertAlmostEqual(
            est.json_ram_gib, n * PYTHON_LIST_BYTES_PER_VALUE / (1024 ** 3), places=6
        )
        self.assertAlmostEqual(
            est.json_disk_gib, n * JSON_BYTES_PER_VALUE / (1024 ** 3), places=6
        )
        self.assertAlmostEqual(
            est.json_time_seconds,
            n / TOLIST_VALUES_PER_SECOND + n / JSON_DUMP_VALUES_PER_SECOND,
            places=6,
        )
        self.assertAlmostEqual(
            est.binary_time_seconds,
            (n * 4) / ASSUMED_DISK_WRITE_BYTES_PER_SECOND,
            places=6,
        )
        # tocar la constante del modulo cambia el resultado (no esta “quemada” en la formula)
        original = res.VRAM_MARGIN_FACTOR
        try:
            res.VRAM_MARGIN_FACTOR = original * 2
            est2 = estimate_model_resources(program, rows=1000)
            self.assertAlmostEqual(est2.vram_train_gib, est.vram_train_gib * 2, places=6)
        finally:
            res.VRAM_MARGIN_FACTOR = original

    def test_estimate_never_raises_and_is_orientative(self) -> None:
        # invariante 6: la estimacion nunca bloquea; siempre se puede pedir y
        # marca su naturaleza orientativa en el payload serializado.
        program = self._small_program()
        est = estimate_model_resources(program)  # sin rows/batch/device
        d = est.to_dict()
        self.assertTrue(d["orientative"])
        self.assertGreater(d["param_count"], 0)


class GiantPromptColabCoherenceTest(unittest.TestCase):
    """La estimacion del prompt 12.1 es coherente con lo observado en Colab
    (A100 80GB/165GB RAM): ~15 GiB de pesos, guardado json en minutos-decenas,
    binario en segundos-baja-decena."""

    def test_estimate_matches_colab_order_of_magnitude(self) -> None:
        program = _giant_program()
        est = estimate_model_resources(program, rows=4000, device="cuda")
        # pesos: 4.03B * 4 bytes ~= 15 GiB (medido: 15.0 GiB)
        self.assertAlmostEqual(est.weights_gib, 15.0, delta=0.1)
        # json: coherente con la observacion real (>1h de CPU en el diagnostico
        # a params algo mayores con menos paralelismo; aqui minutos-decenas)
        self.assertGreater(est.json_time_seconds, 600)
        # binario: cabe en minutos, no horas (la promesa del contrato)
        self.assertLess(est.binary_time_seconds, 600)
        self.assertAlmostEqual(est.binary_disk_gib, 15.0, delta=0.1)


class ScalarParameterAuditTest(unittest.TestCase):
    """Auditoría (MEDIA): un tensor entrenable ESCALAR tiene shape=[] en el
    manifest (no ausencia de shape) — p.ej. el bias de sigmoid_linear/
    linear_regression (fall-risk.typed.mxai: b1, shape=[]). `if not shape`
    confundía `[]` (escalar real, 1 parámetro) con `shape is None` (sin
    información) y descontaba cada escalar entrenable del param_count."""

    def test_scalar_bias_counts_as_one_parameter(self) -> None:
        from pathlib import Path
        from matrixai.parser import parse_file

        root = Path(__file__).resolve().parents[1]
        program = parse_file(root / "examples" / "fall-risk.typed.mxai")
        est = estimate_model_resources(program)
        # manifest verificado: W1 shape=[5] (5 params) + b1 shape=[] (1 param escalar)
        self.assertEqual(est.param_count, 6)


class RealBatchFromTrainingTextTest(unittest.TestCase):
    """Auditoría (ALTA): la VRAM se estimaba con el batch por defecto del
    dispositivo si no se pasaba `batch` explícito, aunque el `.mxtrain` real
    declarase un BATCH size distinto — invisible para la estimación."""

    def test_batch_declared_in_training_text_is_honored(self) -> None:
        gen = DenseNetworkGenerator().generate(
            "clasificar\nFEATURES:\n  a: Scalar\n  b: Scalar\nSALIDA: y: ProbabilityMap[NO, SI]"
        )
        # el generador emite "BATCH size=8" por defecto (no rows/batch explícitos)
        self.assertIn("BATCH size=8", gen.training_text)
        program = parse_text(gen.mxai_text)
        est_with_training = estimate_model_resources(
            program, training_text=gen.training_text, device="cpu"
        )
        est_without = estimate_model_resources(program, device="cpu")
        self.assertEqual(est_with_training.effective_batch, 8)
        # sin el .mxtrain, cae al default del dispositivo (2048 en CPU) — muy
        # distinto del BATCH real declarado, y por tanto una VRAM distinta.
        self.assertEqual(est_without.effective_batch, 2048)
        self.assertNotEqual(est_with_training.vram_train_gib, est_without.vram_train_gib)

    def test_explicit_batch_wins_over_training_text(self) -> None:
        gen = DenseNetworkGenerator().generate(
            "clasificar\nFEATURES:\n  a: Scalar\n  b: Scalar\nSALIDA: y: ProbabilityMap[NO, SI]"
        )
        program = parse_text(gen.mxai_text)
        est = estimate_model_resources(
            program, training_text=gen.training_text, batch=128, device="cpu"
        )
        self.assertEqual(est.effective_batch, 128)


if __name__ == "__main__":
    unittest.main()
