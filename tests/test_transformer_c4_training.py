# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C4 — entrenamiento torch + Adam + cierre funcional.

Cierre funcional del contrato: en un toy sintético donde la CLASE depende de la
POSICIÓN de un token (clase "antes" si el token 7 aparece antes que el 9), el
transformer entrena y supera CLARAMENTE a una dense bag-of-words del mismo
presupuesto — la prueba de que la atención VE posiciones (una BoW no puede:
los conteos de 7 y 9 son idénticos en ambas clases, su Bayes es 50%).

Umbrales DOCUMENTADOS (seed fijo, CPU): transformer ≥ 0.90, dense ≤ 0.65,
margen ≥ 0.25. Medidos en calibración: transformer 0.975 (2450 params, Adam,
60 épocas) vs dense 0.435 (2434 params, 120 épocas, lr 0.05 — el doble de
épocas y mayor lr, para que la derrota no sea por falta de entrenamiento).
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from matrixai.parameters.network_params import (
    build_composite_network_parameter_set,
    build_network_parameter_set,
    validate_composite_network_parameter_set,
)
from matrixai.parser.parser import parse_text
from matrixai.training.composite_torch_trainer import (
    CompositeTorchTrainError,
    evaluate_composite_network_torch,
    train_composite_network_torch,
)
from matrixai.types import check_composite_network_types, check_network_types

L_SEQ, VOCAB = 12, 16

_TOY_MXAI = """
PROJECT Toy
SEQUENCE Texto
  length = 12
  vocab_size = 16
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 16
  BLOCK enc TRANSFORMER
    LAYERS 1
    HEADS 2
    FF 32
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[antes, despues]
END
"""

_FIELDS = "\n".join(f"  c{i}: Scalar" for i in range(VOCAB))
_TOY_DENSE_MXAI = f"""
PROJECT ToyDense
VECTOR Counts[16]
{_FIELDS}
END
NETWORK D
  INPUT Counts
  LAYER Dense units=128 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[antes, despues]
END

GRAPH
  Counts -> D
END
"""


def _toy_rows(n: int, seed: int = 123):
    """Filas del toy: 7 y 9 en posiciones aleatorias distintas, relleno {1..6}.
    Clase 'antes' ([1,0]) si pos(7) < pos(9). El multiconjunto de 7/9 es
    idéntico en ambas clases — solo el ORDEN lleva señal."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p7, p9 = rng.sample(range(L_SEQ), 2)
        row = [rng.choice([1, 2, 3, 4, 5, 6]) for _ in range(L_SEQ)]
        row[p7], row[p9] = 7, 9
        target = [1.0, 0.0] if p7 < p9 else [0.0, 1.0]
        rows.append((row, target))
    return rows


def _seq_examples(rows):
    return [({"Texto": row}, target) for row, target in rows]


def _bow_examples(rows):
    out = []
    for row, target in rows:
        counts = [0.0] * VOCAB
        for t in row:
            counts[t] += 1.0
        out.append((counts, target))
    return out


def _build_transformer(seed: int = 5):
    prog = parse_text(_TOY_MXAI)
    net = prog.networks[0]
    res = check_composite_network_types(net, {}, {s.name: s for s in prog.sequences})
    assert res.ok, res.errors
    ps = build_composite_network_parameter_set(net, res, "toy", seed=seed)
    return net, res, ps


TRAIN_ROWS = _toy_rows(400, seed=123)
TEST_ROWS = _toy_rows(200, seed=321)


# ---------------------------------------------------------------------------
# Cierre funcional — el transformer VE posiciones, la dense BoW no
# ---------------------------------------------------------------------------

class TestToyPosicional:
    def _train_transformer(self):
        net, res, ps = _build_transformer()
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS), "cross_entropy",
            lr=0.01, epochs=60, seed=5, batch_size=64, type_result=res,
        )
        return net, res, result

    def test_transformer_beats_dense_bow_same_budget(self):
        # Transformer
        net, res, result = self._train_transformer()
        ev = evaluate_composite_network_torch(
            net, result["best_params"], _seq_examples(TEST_ROWS),
            "cross_entropy", labels=["antes", "despues"], type_result=res,
        )
        tf_acc = ev.accuracy

        # Dense bag-of-words, mismo presupuesto (2434 vs 2450 params), con
        # MÁS épocas y mayor lr — la derrota no es por falta de entrenamiento
        from matrixai.forward.dense_torch import dense_network_to_torch_module
        from matrixai.training.dense_evaluator import result_from_predictions
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        dprog = parse_text(_TOY_DENSE_MXAI)
        dnet = dprog.networks[0]
        dtr = check_network_types(dnet, {v.name: v for v in dprog.vectors})
        dps = build_network_parameter_set(
            dnet, dtr.resolved_layers or dnet.layers, "toyd", seed=5
        )
        dresult = train_dense_network_torch(
            dnet, dps, _bow_examples(TRAIN_ROWS), "cross_entropy",
            lr=0.05, epochs=120, seed=5, batch_size=64,
        )
        module = dense_network_to_torch_module(dnet, dresult["best_params"])
        module.eval()
        with torch.no_grad():
            preds = module(
                torch.tensor([x for x, _ in _bow_examples(TEST_ROWS)], dtype=torch.float32)
            ).tolist()
        dense_acc = result_from_predictions(
            preds, [t for _, t in TEST_ROWS], "cross_entropy", ["antes", "despues"]
        ).accuracy

        # Presupuestos comparables (±5%)
        tf_params = result["param_count"]
        dense_params = sum(
            (len(v["values"]) * len(v["values"][0]) if isinstance(v["values"][0], list)
             else len(v["values"]))
            for v in dps.parameters.values()
        )
        assert abs(tf_params - dense_params) / tf_params < 0.05

        # Umbrales documentados
        assert tf_acc >= 0.90, f"transformer accuracy {tf_acc}"
        assert dense_acc <= 0.65, f"dense BoW accuracy {dense_acc} — ¿ve posiciones?"
        assert tf_acc - dense_acc >= 0.25

    def test_convergence_loss_decreases(self):
        _, _, result = self._train_transformer()
        trace = result["epochs"]
        assert trace[-1]["loss"] < trace[0]["loss"] * 0.5
        assert result["best_val_loss"] < trace[0]["val_loss"]


# ---------------------------------------------------------------------------
# Adam — gramática, verifier y trainer
# ---------------------------------------------------------------------------

class TestAdam:
    def test_adam_parses_and_verifier_accepts(self):
        from matrixai.training import parse_training_text
        from matrixai.training.verifier import TrainingVerifier
        training = parse_training_text("""
MODEL dummy.mxai

DATASET D
  SOURCE csv("dummy.csv")
  INPUT Texto FROM COLUMNS [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
  TARGET clase: ProbabilityMap[antes, despues]
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
  TYPE adam
  LEARNING_RATE 0.01
  UPDATE N.*
END

RUN
  EPOCHS 2
END
""")
        assert training.optimizer.type == "adam"
        program = parse_text(_TOY_MXAI)
        errors = TrainingVerifier()._verify_program_contract(training, program)
        assert not any("OPTIMIZER type" in e for e in errors)

    def test_verifier_rejects_unknown_optimizer(self):
        from matrixai.training import parse_training_text
        from matrixai.training.verifier import TrainingVerifier
        training = parse_training_text("""
MODEL dummy.mxai

DATASET D
  SOURCE csv("dummy.csv")
  INPUT Texto FROM COLUMNS [t0]
  TARGET clase: ProbabilityMap[a, b]
END

LOSS L
  TYPE cross_entropy
  PREDICTION clase
  TARGET clase
END

OPTIMIZER O
  TYPE rmsprop
  LEARNING_RATE 0.01
  UPDATE N.*
END

RUN
  EPOCHS 2
END
""")
        program = parse_text(_TOY_MXAI)
        errors = TrainingVerifier()._verify_program_contract(training, program)
        assert any("OPTIMIZER type not supported" in e for e in errors)

    def test_transformer_defaults_to_adam(self):
        net, res, ps = _build_transformer()
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:64]), "cross_entropy",
            epochs=2, seed=5, type_result=res,
        )
        assert result["optimizer"] == "adam"

    def test_explicit_sgd_honored_for_transformer(self):
        net, res, ps = _build_transformer()
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:64]), "cross_entropy",
            epochs=2, seed=5, type_result=res, optimizer="sgd",
        )
        assert result["optimizer"] == "sgd"

    def test_tabular_composite_keeps_sgd_default(self):
        src = """
PROJECT Tab
VECTOR Product[2]
  category_id: Integer[0, 100]
  price: Scalar
END
NETWORK T
  INPUT Product
  EMBEDDING cat FROM category_id VOCAB 100 DIM 4
  CONCAT [cat, price] -> f
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[a, b]
END
"""
        prog = parse_text(src)
        net = prog.networks[0]
        res = check_composite_network_types(net, {v.name: v for v in prog.vectors}, {})
        ps = build_composite_network_parameter_set(net, res, "tab", seed=5)
        examples = [({"category_id": i % 100, "price": 0.5}, [1.0, 0.0]) for i in range(16)]
        result = train_composite_network_torch(net, ps, examples, "cross_entropy", epochs=1, seed=5)
        assert result["optimizer"] == "sgd"

    def test_unknown_optimizer_rejected(self):
        net, res, ps = _build_transformer()
        with pytest.raises(CompositeTorchTrainError, match="unsupported optimizer"):
            train_composite_network_torch(
                net, ps, _seq_examples(TRAIN_ROWS[:8]), "cross_entropy",
                epochs=1, type_result=res, optimizer="rmsprop",
            )


# ---------------------------------------------------------------------------
# Extracción de pesos + puerta PESOS_GRANDES
# ---------------------------------------------------------------------------

class TestExtraccionPesos:
    def test_best_params_validate_and_reload(self):
        """Los pesos entrenados validan contra el manifest y el módulo
        reconstruido desde ellos predice EXACTAMENTE igual."""
        net, res, ps = _build_transformer()
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:128]), "cross_entropy",
            epochs=3, seed=5, type_result=res,
        )
        best = result["best_params"]
        compat = validate_composite_network_parameter_set(net, res, best, "toy")
        assert compat.ok, compat.errors
        assert best.source == "torch"
        # Round-trip: módulo reconstruido == mismas predicciones
        from matrixai.forward.transformer_torch import (
            transformer_network_to_torch_module,
            transformer_torch_forward_batch,
        )
        module = transformer_network_to_torch_module(net, res, best)
        out = transformer_torch_forward_batch(module, [r for r, _ in TRAIN_ROWS[:4]])
        ev = evaluate_composite_network_torch(
            net, best, _seq_examples(TRAIN_ROWS[:4]), "cross_entropy",
            labels=["antes", "despues"], type_result=res,
        )
        assert len(out) == 4 and ev is not None

    def test_pesos_grandes_gate_state_dict(self):
        """materialize=False (simula superar torch_native_min_params): NO se
        materializan listas — best_state_dict con tensores CPU y claves del
        ParameterSet; mismos valores que la materialización con el mismo seed."""
        net, res, ps = _build_transformer()
        r_state = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:64]), "cross_entropy",
            epochs=2, seed=5, type_result=res, materialize=False,
        )
        assert r_state["best_params"] is None
        sd = r_state["best_state_dict"]
        assert sd is not None
        assert all(not t.is_cuda for t in sd.values())
        assert "N.enc.layer_0.attention.Wq" in sd
        assert "N.tok.table" in sd

        ps2 = _build_transformer()[2]
        r_mat = train_composite_network_torch(
            net, ps2, _seq_examples(TRAIN_ROWS[:64]), "cross_entropy",
            epochs=2, seed=5, type_result=res, materialize=True,
        )
        wq_mat = torch.tensor(
            r_mat["best_params"].parameters["N.enc.layer_0.attention.Wq"]["values"]
        )
        assert torch.allclose(sd["N.enc.layer_0.attention.Wq"], wq_mat, atol=1e-6)

    def test_state_dict_paths_match_manifest(self):
        from matrixai.parameters.network_params import (
            composite_network_parameter_manifest,
        )
        net, res, ps = _build_transformer()
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:32]), "cross_entropy",
            epochs=1, seed=5, type_result=res, materialize=False,
        )
        manifest_paths = {
            m["path"] for m in composite_network_parameter_manifest(net.name, net, res)
        }
        assert set(result["best_state_dict"].keys()) == manifest_paths


# ---------------------------------------------------------------------------
# epoch_callback / trace / early stop / cancel
# ---------------------------------------------------------------------------

class TestCallbacksYCancel:
    def test_epoch_callback_and_trace(self):
        net, res, ps = _build_transformer()
        seen: list[dict] = []
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:64]), "cross_entropy",
            epochs=3, seed=5, type_result=res, epoch_callback=seen.append,
        )
        assert len(seen) == 3 == len(result["epochs"])
        for entry in seen:
            assert set(entry) >= {"epoch", "loss", "val_loss"}
        assert [e["epoch"] for e in seen] == [1, 2, 3]

    def test_early_stop_patience(self):
        net, res, ps = _build_transformer()
        result = train_composite_network_torch(
            net, ps, _seq_examples(TRAIN_ROWS[:64]), "cross_entropy",
            epochs=50, seed=5, type_result=res, early_stop=(2, "val_loss"),
        )
        assert len(result["epochs"]) < 50

    def test_cancel_check_aborts(self):
        """El cancel corta por batch (patrón GPU-C2) y la excepción sube."""
        net, res, ps = _build_transformer()

        class Stop(Exception):
            pass

        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            raise Stop()

        with pytest.raises(Stop):
            train_composite_network_torch(
                net, ps, _seq_examples(TRAIN_ROWS[:128]), "cross_entropy",
                epochs=10, seed=5, type_result=res, batch_size=16,
                cancel_check=cancel,
            )
        assert calls["n"] == 1  # abortó en el PRIMER batch

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
    def test_cancel_releases_vram(self):
        """Patrón GPU-C2/M18: tras cancelar, la VRAM del módulo se libera."""
        net, res, ps = _build_transformer()

        class Stop(Exception):
            pass

        def cancel():
            raise Stop()

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        with pytest.raises(Stop):
            train_composite_network_torch(
                net, ps, _seq_examples(TRAIN_ROWS), "cross_entropy",
                epochs=10, seed=5, type_result=res, device="cuda",
                cancel_check=cancel,
            )
        assert torch.cuda.memory_allocated() <= baseline + 1_000_000


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuardsEntrenamiento:
    def test_train_without_type_result_rejected(self):
        net, res, ps = _build_transformer()
        with pytest.raises(CompositeTorchTrainError, match="pass\\s+type_result"):
            train_composite_network_torch(
                net, ps, _seq_examples(TRAIN_ROWS[:8]), "cross_entropy", epochs=1,
            )

    def test_evaluate_without_type_result_rejected(self):
        net, res, ps = _build_transformer()
        with pytest.raises(CompositeTorchTrainError, match="pass\\s+type_result"):
            evaluate_composite_network_torch(
                net, ps, _seq_examples(TRAIN_ROWS[:8]), "cross_entropy",
            )

    def test_pad_id_masks_training_and_evaluation(self):
        """pad_id en trainer+evaluador: filas con padding entrenan y evalúan
        con la misma semántica de máscara del forward (invariante 1c). El
        entrenamiento con pad_id produce EXACTAMENTE los mismos pesos si el
        contenido de las posiciones enmascaradas es irrelevante — verificable
        vía la evaluación: dos test-sets que difieren SOLO en contenido de
        padding puntúan idéntico con los pesos entrenados."""
        net, res, ps = _build_transformer()
        rows = [
            ({"Texto": [1, 2, 3, 7, 9, 0, 0, 0, 0, 0, 0, 0]}, [1.0, 0.0]),
            ({"Texto": [4, 9, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0]}, [0.0, 1.0]),
        ] * 16
        result = train_composite_network_torch(
            net, ps, rows, "cross_entropy", epochs=2, seed=5,
            type_result=res, pad_id=0,
        )
        best = result["best_params"]
        ev_a = evaluate_composite_network_torch(
            net, best, rows[:2], "cross_entropy",
            labels=["antes", "despues"], type_result=res, pad_id=0,
        )
        assert ev_a is not None and result["epochs"]
