# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""SECUENCIAS_PRODUCTO — autoauditoría: M7 (detección de colapso, predictor
constante) nunca cubrió un modelo Text. `_probe_model_collapse` (playground.py)
mira exclusivamente `program.vectors` y devuelve `None` para CUALQUIER modelo
con INPUT SEQUENCE — el aviso "Modelo colapsado" de la UI (con el botón
"Reintentar con otra inicialización") nunca se disparaba para un transformer,
ni cuando colapsó de verdad. `probe_collapse_transformer_torch` (mismo patrón
que `probe_collapse_torch` para denso — dying ReLU, aquí "todos los pesos a
cero") cierra el hueco, enhebrado en `_run_playground_transformer_training`
exactamente como el camino GPU/torch de denso lo hace ya.
"""
from __future__ import annotations

import time
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch required")

from matrixai.parser.parser import parse_text
from matrixai.parameters.network_params import build_composite_network_parameter_set
from matrixai.parameters.store import program_hash
from matrixai.types import check_composite_network_types

_MXAI = """
PROJECT CollapseCheck

SEQUENCE Resenas
  length = 8
  vocab_size = 259
END

NETWORK N
  INPUT Resenas
  EMBEDDING tok FROM Resenas DIM 8
  BLOCK enc TRANSFORMER
    LAYERS 1
    HEADS 2
    FF 16
    ACTIVATION gelu
    POS sinusoidal
  END
  POOL mean
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[NEG, POS]
END

GRAPH
  Resenas -> N
END
"""


def _build(seed: int = 7):
    prog = parse_text(_MXAI)
    net = prog.networks[0]
    res = check_composite_network_types(
        net, {}, {s.name: s for s in prog.sequences}
    )
    assert res.ok, res.errors
    ps = build_composite_network_parameter_set(net, res, program_hash(prog), seed=seed)
    return prog, net, res, ps


def _zeroed(values):
    if isinstance(values, list):
        return [_zeroed(v) for v in values]
    return 0.0


def _collapsed_parameter_set(ps):
    for tensor in ps.parameters.values():
        if isinstance(tensor, dict) and "values" in tensor and tensor["values"] is not None:
            tensor["values"] = _zeroed(tensor["values"])
    return ps


class TestProbeCollapseTransformerTorch:
    def test_all_zero_weights_detected_as_collapsed(self):
        from matrixai.training.composite_torch_trainer import probe_collapse_transformer_torch
        prog, net, res, ps = _build()
        seq = prog.sequences[0]
        collapsed_ps = _collapsed_parameter_set(ps)
        r = probe_collapse_transformer_torch(
            net, res, seq.length, seq.vocab_size, parameter_set=collapsed_ps,
        )
        assert r is not None
        assert r["collapsed"] is True
        assert "constant_output" in r

    def test_freshly_initialized_weights_not_flagged(self):
        from matrixai.training.composite_torch_trainer import probe_collapse_transformer_torch
        prog, net, res, ps = _build()
        seq = prog.sequences[0]
        r = probe_collapse_transformer_torch(
            net, res, seq.length, seq.vocab_size, parameter_set=ps,
        )
        assert r is not None
        assert r["collapsed"] is False
        assert "constant_output" not in r

    def test_missing_both_weights_returns_none(self):
        from matrixai.training.composite_torch_trainer import probe_collapse_transformer_torch
        prog, net, res, ps = _build()
        seq = prog.sequences[0]
        assert probe_collapse_transformer_torch(
            net, res, seq.length, seq.vocab_size,
        ) is None

    def test_state_dict_path_matches_parameter_set_path(self):
        """PESOS_GRANDES: mismo camino torch para state_dict crudo que para
        una ParameterSet materializada — ambos deben coincidir en un modelo
        colapsado."""
        from matrixai.training.composite_torch_trainer import probe_collapse_transformer_torch
        from matrixai.forward.transformer_torch import transformer_network_to_torch_module
        import torch

        prog, net, res, ps = _build()
        seq = prog.sequences[0]
        collapsed_ps = _collapsed_parameter_set(ps)
        module = transformer_network_to_torch_module(net, res, collapsed_ps)
        state_dict = {k: v.detach().clone() for k, v in module.path_tensors.items()}

        via_state = probe_collapse_transformer_torch(
            net, res, seq.length, seq.vocab_size, state_dict=state_dict,
        )
        via_params = probe_collapse_transformer_torch(
            net, res, seq.length, seq.vocab_size, parameter_set=collapsed_ps,
        )
        assert via_state is not None and via_params is not None
        assert via_state["collapsed"] == via_params["collapsed"] is True

    def test_zero_seq_length_returns_none(self):
        from matrixai.training.composite_torch_trainer import probe_collapse_transformer_torch
        prog, net, res, ps = _build()
        assert probe_collapse_transformer_torch(
            net, res, 0, 259, parameter_set=ps,
        ) is None


class TestTransformerTrainingSurfacesCollapse:
    """`_run_playground_transformer_training` (Studio's dispatch, C4) ahora
    adjunta model_collapsed/collapse_warning — antes de este fix, nunca
    aparecía para NINGÚN modelo Text (ni sano ni colapsado, `None` en ambos
    casos porque `_probe_model_collapse` nunca llegaba a probarlo)."""

    PROMPT = "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]"

    def test_healthy_training_reports_not_collapsed(self):
        from matrixai.playground import analyze_playground_request, _generate_synthetic_dataset, _run_playground_training

        res = analyze_playground_request({"mode": "prompt", "prompt": self.PROMPT, "use_llm": False})
        assert res["ok"], res.get("error")
        ds = _generate_synthetic_dataset(res["mxai"], res["training_text"], 24, 7, "coherent", use_llm=False)
        assert ds["ok"], ds.get("error")
        tr = _run_playground_training(res["mxai"], res["training_text"], ds["csv_text"], epochs_override=10)
        assert tr["ok"], tr.get("error")
        assert "model_collapsed" in tr
        assert tr["model_collapsed"] is False
        assert "collapse_warning" not in tr


class TestJobStatusExposesModelCollapsedForText:
    """Mismo test end-to-end que `test_m7_collapse_detection.py::
    TestTrainingJobE2E::test_job_status_exposes_model_collapsed`, pero para
    un modelo Text vía el dispatch asíncrono real (`_submit_training_job`)."""

    def test_job_status_exposes_model_collapsed_for_text_model(self):
        from matrixai.playground import (
            analyze_playground_request, _generate_synthetic_dataset,
            _submit_training_job, _get_job_status,
        )

        prompt = "resenas: Text\nOUTPUT clase: ProbabilityMap[NEG, POS]"
        res = analyze_playground_request({"mode": "prompt", "prompt": prompt, "use_llm": False})
        assert res["ok"], res.get("error")
        ds = _generate_synthetic_dataset(res["mxai"], res["training_text"], 24, 7, "coherent", use_llm=False)
        assert ds["ok"], ds.get("error")

        job = _submit_training_job(res["mxai"], res["training_text"], ds["csv_text"], epochs_override=10)
        assert job.get("ok"), job
        st = {}
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert st.get("model_collapsed") is False, st.get("model_collapsed")
