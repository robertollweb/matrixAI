# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""SECUENCIAS_PRODUCTO C4 — entrenamiento de modelos Text (BLOCK TRANSFORMER)
vía el mismo dispatch `network_call` que dense/composite.

Antes de este corte, `_run_playground_training`/`_submit_training_job`
enrutaban CUALQUIER NETWORK con `kind="composite_network"` (transformer
incluido — mismo AST que P19, `transformer_blocks` es solo un campo extra)
a `_run_playground_composite_training`, que revienta explícitamente para
`transformer_blocks` (`composite_forward`: "use transformer_network_forward
... TRANSFORMER_BLOQUE C4"). `_network_is_transformer` se comprueba PRIMERO
y desvía a `_run_playground_transformer_training` (torch real vía
`train_composite_network_torch`, reusando `_resolve_transformer_dataset`
para el boundary de tokenización/pad_id auditado en C3).
"""
from __future__ import annotations

import time

import pytest

from matrixai.training.transformer_generator import TransformerNetworkGenerator
from matrixai.training.synthetic_text import generate_text_examples


def _gen(prompt: str = "resenas: Text[16]\nOUTPUT clase: ProbabilityMap[NEG, POS]"):
    return TransformerNetworkGenerator().generate(prompt)


def _classification_csv(field_name: str = "resenas", target_name: str = "predicted_class",
                         rows: int = 20, seq_length: int = 16) -> str:
    text_rows, _ = generate_text_examples(
        "coherent", "clasificar", rows, ["neg", "pos"], seed=1, seq_length=seq_length, use_llm=False,
    )
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([field_name, target_name])
    for text, label in text_rows:
        writer.writerow([text, label])
    return buf.getvalue()


_DENSE_MXAI = (
    "PROJECT P\n\nVECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
    "NETWORK N\n  INPUT In\n  LAYER Dense units=4 activation=relu\n"
    "  LAYER Dense units=1 activation=sigmoid\n  OUTPUT y: Probability\nEND\n\n"
    "GRAPH\n  In -> N\nEND\n"
)

_COMPOSITE_MXAI = """PROJECT ResProject

VECTOR Patient[4]
  x1: Scalar
  x2: Scalar
  x3: Scalar
  x4: Scalar
END

NETWORK ResNet
  INPUT Patient
  LAYER Dense units=16 activation=relu
  BLOCK res1
    LAYER Dense units=16 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=3 activation=softmax
  OUTPUT predicted_class: ProbabilityMap[BAJO, MEDIO, ALTO]
END

GRAPH
  Patient -> ResNet
END
"""


class TestDetection:
    def test_network_is_transformer_true_for_text_model(self):
        from matrixai.playground import _network_is_transformer

        gen = _gen()
        assert _network_is_transformer(gen.mxai_text) is True

    def test_network_is_transformer_false_for_dense(self):
        from matrixai.playground import _network_is_transformer

        assert _network_is_transformer(_DENSE_MXAI) is False

    def test_network_is_transformer_false_for_plain_composite(self):
        """Un composite (P19) SIN BLOCK TRANSFORMER (kind idéntico) no debe
        confundirse con un modelo Text — solo `transformer_blocks` decide."""
        from matrixai.playground import _network_is_transformer, _network_is_composite

        assert _network_is_transformer(_COMPOSITE_MXAI) is False
        assert _network_is_composite(_COMPOSITE_MXAI) is True

    def test_network_is_composite_also_true_for_transformer(self):
        """Documenta la ambigüedad que motiva el orden del dispatch: un
        modelo Text también es kind=composite_network."""
        from matrixai.playground import _network_is_composite

        gen = _gen()
        assert _network_is_composite(gen.mxai_text) is True


class TestSyncTraining:
    def test_transformer_trains_and_learns(self):
        from matrixai.playground import _run_playground_training

        gen = _gen()
        r = _run_playground_training(gen.mxai_text, gen.training_text, _classification_csv(), epochs_override=4)
        assert r["ok"], r.get("error")
        assert r["network_kind"] == "composite_network"
        assert r["params_best"] is not None
        assert r["backend"] in ("cpu", "cuda")
        assert r["epochs"], "no epoch trace recorded"

    def test_transformer_exposes_classification_metrics(self):
        from matrixai.playground import _run_playground_training

        gen = _gen()
        r = _run_playground_training(gen.mxai_text, gen.training_text, _classification_csv(), epochs_override=4)
        assert r["ok"], r.get("error")
        assert r["task_kind"] == "classification"
        assert r["labels"] == ["neg", "pos"]
        assert isinstance(r["macro_f1"], float)
        assert set(r["confusion_matrix"].keys()) == {"neg", "pos"}
        assert r["evaluation_report"]["macro_f1"] == r["macro_f1"]

    def test_transformer_regression(self):
        """El generador también produce regresión (`OUTPUT x: Scalar`) — la
        generación sintética de C3 es solo-clasificación, así que el CSV se
        construye a mano (igual que un usuario que sube datos reales)."""
        from matrixai.playground import _run_playground_training

        gen = _gen("resenas: Text[16]\nOUTPUT puntuacion: Scalar")
        rows = ["resenas,predicted_value"]
        samples = [
            ("me encanta este producto", 0.9), ("terrible experiencia", 0.1),
            ("bastante bueno en general", 0.7), ("no lo recomiendo nunca", 0.2),
            ("calidad excelente de verdad", 0.95), ("una decepcion total", 0.05),
            ("cumple lo que promete bien", 0.75), ("muy malo no comprar", 0.15),
        ]
        for text, value in samples * 2:
            rows.append(f"{text},{value}")
        csv_text = "\n".join(rows) + "\n"
        r = _run_playground_training(gen.mxai_text, gen.training_text, csv_text, epochs_override=4)
        assert r["ok"], r.get("error")
        assert r["task_kind"] == "regression"
        assert r["params_best"] is not None

    def test_transformer_backend_stdlib_override_refuses_with_actionable_error(self, monkeypatch):
        """A diferencia de dense/composite, BLOCK TRANSFORMER no tiene
        fallback stdlib (invariante 6) — un operador que fuerce
        MATRIXAI_TRAIN_BACKEND=stdlib debe ver un error accionable, no un
        crash de composite_forward ni un entrenamiento silenciosamente
        degradado."""
        from matrixai.playground import _run_playground_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
        gen = _gen()
        r = _run_playground_training(gen.mxai_text, gen.training_text, _classification_csv(), epochs_override=2)
        assert r["ok"] is False
        assert "torch" in r["error"]

    def test_transformer_auto_mode_uses_torch_even_without_cuda(self, monkeypatch):
        """`_select_train_backend` (dense/composite) cae a stdlib en 'auto'
        sin CUDA; `_select_transformer_train_device` NO debe heredar ese
        atajo — CPU+torch es el único camino de producto para texto."""
        monkeypatch.delenv("MATRIXAI_TRAIN_BACKEND", raising=False)
        from matrixai.playground import _select_transformer_train_device
        from matrixai.parameters.tensor_bridge import torch_available

        use_torch, device = _select_transformer_train_device()
        if torch_available():
            assert use_torch is True
            assert device in ("cpu", "cuda")
        else:
            assert use_torch is False


class TestAsyncJob:
    def test_async_transformer_job_done_with_metrics(self):
        from matrixai.playground import _submit_training_job, _get_job_status

        gen = _gen()
        job = _submit_training_job(gen.mxai_text, gen.training_text, _classification_csv(), epochs_override=3)
        assert job.get("ok"), job
        st = {}
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert st.get("params_best") is not None
        assert isinstance(st.get("macro_f1"), float)
        assert "best_state_dict" not in st

    def test_async_job_payload_never_serializes_state_dict_marker_shape(self):
        """`params_best` debe ser un dict serializable siempre (valores
        reales o marcador PESOS_GRANDES) — nunca None con ok=True."""
        from matrixai.playground import _submit_training_job, _get_job_status

        gen = _gen()
        job = _submit_training_job(gen.mxai_text, gen.training_text, _classification_csv(), epochs_override=2)
        st = {}
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert isinstance(st.get("params_best"), dict)


class TestRegressionOtherArchitecturesUnaffected:
    """Decisión del contrato: prompts/tipos ya soportados no cambian."""

    def test_dense_still_routes_to_dense_trainer(self):
        from matrixai.playground import _run_playground_training

        csv_text = "a,b,y\n" + "\n".join(
            f"{(i % 10) / 10.0},{(i % 7) / 7.0:.3f},{1 if i % 2 else 0}" for i in range(30)
        )
        mxtrain = """MODEL P.mxai

DATASET D
  SOURCE csv("d.train.csv")
  INPUT In FROM COLUMNS [a, b]
  TARGET y: Probability
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END

LOSS L
  TYPE binary_cross_entropy
  PREDICTION N
  TARGET y
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.05
  UPDATE N.*
END

RUN
  EPOCHS 3
END
"""
        r = _run_playground_training(_DENSE_MXAI, mxtrain, csv_text, epochs_override=3)
        assert r["ok"], r.get("error")
        assert "network_kind" not in r or r.get("network_kind") != "composite_network"

    def test_plain_composite_still_routes_to_composite_trainer(self):
        from matrixai.playground import _run_playground_training

        mxtrain = """MODEL ResProject.mxai

DATASET D
  SOURCE csv("d.train.csv")
  INPUT Patient FROM COLUMNS [x1, x2, x3, x4]
  TARGET predicted_class: Label[BAJO, MEDIO, ALTO]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END

LOSS L
  TYPE cross_entropy
  PREDICTION ResNet
  TARGET predicted_class
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.05
  UPDATE ResNet.*
END

RUN
  EPOCHS 3
END
"""
        rows = ["x1,x2,x3,x4,predicted_class"]
        for i in range(30):
            s = (i % 10) / 10.0
            lab = "BAJO" if s < 0.33 else ("MEDIO" if s < 0.66 else "ALTO")
            rows.append(f"{s},{(i % 7) / 7.0:.3f},{(i % 5) / 5.0:.3f},{(i % 3) / 3.0:.3f},{lab}")
        r = _run_playground_training(_COMPOSITE_MXAI, mxtrain, "\n".join(rows) + "\n", epochs_override=3)
        assert r["ok"], r.get("error")
        assert r["network_kind"] == "composite_network"
