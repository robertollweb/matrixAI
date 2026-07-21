"""CONTRATO 60 — paridad torch/GPU vs stdlib/CPU en regresión densa.

El backend torch entrenaba una regresión densa de pocas features hasta el
colapso (dead-ReLU) por usar el init NATIVO de `nn.Linear`
(`kaiming_uniform_(a=√5)` + sesgo aleatorio), mientras el camino stdlib usa
`he_normal`/`xavier_normal` + sesgo cero y aprende perfectamente. El fix
reinicializa el módulo torch construido desde plantilla con el mismo esquema
que stdlib, usando `torch.nn.init` (en el dispositivo, sin materializar
valores en Python — el camino de modelos grandes también queda bien).

El mismo hueco (y el mismo fix) existe en `composite_torch.py` — sus capas
`Dense` construyen `nn.Linear` de la misma forma. C4 midió que, ADEMÁS, un
bloque `LayerNorm` colapsa esta misma regresión de una sola feature
independientemente del init (byte-idéntico con init nativo, con este fix, o
con valores materializados completos) — causa distinta, sin investigar,
fuera de alcance de este contrato (ver §Fuera de alcance en el contrato).

Reproducible sin GPU: torch-CPU basta (el default `auto` elige stdlib en una
máquina sin CUDA, por eso el contrato 59 nunca ejerció este camino).
"""
from __future__ import annotations

import os
import time
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _linear_csv(header_x: str, header_y: str, fn, n: int = 100) -> str:
    rows = [f"{header_x},{header_y}"]
    for c in range(n):
        rows.append(f"{c},{fn(c)}")
    return "\n".join(rows) + "\n"


def _train_r2(csv_text: str, target_col: str, backend: str) -> float:
    """Genera un proyecto desde el CSV y lo entrena por `backend`, devolviendo R²."""
    from matrixai.training.dataset_project import generate_project_from_dataset
    from matrixai import playground as pg

    gen = generate_project_from_dataset(csv_text, target_col)
    assert gen["ok"], gen
    target_range = tuple(gen["target_range"]) if gen.get("target_range") else None

    prev = os.environ.get("MATRIXAI_TRAIN_BACKEND")
    os.environ["MATRIXAI_TRAIN_BACKEND"] = backend
    try:
        pg._training_jobs.clear()
        sub = pg._submit_training_job(
            gen["mxai"], gen["training_text"], gen["csv_text"], None,
            field_ranges=gen.get("field_ranges") or {}, seed=42,
            target_range=target_range,
        )
        assert sub.get("ok"), sub
        job_id = sub["job_id"]
        for _ in range(600):
            if pg._training_jobs[job_id]["status"] != "running":
                break
            time.sleep(0.1)
        job = pg._training_jobs[job_id]
        assert job["status"] == "done", job
        return float((job.get("result") or {})["r2"])
    finally:
        if prev is None:
            os.environ.pop("MATRIXAI_TRAIN_BACKEND", None)
        else:
            os.environ["MATRIXAI_TRAIN_BACKEND"] = prev


# --- Invariante 1: Kelvin aprende por torch, igual que por stdlib -----------

def test_kelvin_learns_on_torch_backend():
    csv = _linear_csv("centigrados", "prediccionKelvin", lambda c: c + 273.15)
    r2_torch = _train_r2(csv, "prediccionKelvin", "torch")
    assert r2_torch >= 0.99, f"torch no aprendió Kelvin: R²={r2_torch}"


def test_kelvin_torch_matches_stdlib_quality():
    csv = _linear_csv("centigrados", "prediccionKelvin", lambda c: c + 273.15)
    r2_stdlib = _train_r2(csv, "prediccionKelvin", "stdlib")
    r2_torch = _train_r2(csv, "prediccionKelvin", "torch")
    # No exigimos igualdad byte a byte (init/optimizador difieren) pero sí que
    # torch no quede materialmente por debajo de stdlib.
    assert r2_torch >= r2_stdlib - 0.02, f"torch={r2_torch} << stdlib={r2_stdlib}"


# --- Invariante 2: otra escala lineal (Fahrenheit) también aprende ----------

def test_fahrenheit_learns_on_torch_backend():
    csv = _linear_csv("celsius", "fahrenheit", lambda c: c * 1.8 + 32.0)
    r2_torch = _train_r2(csv, "fahrenheit", "torch")
    assert r2_torch >= 0.99, f"torch no aprendió Fahrenheit: R²={r2_torch}"


# --- Invariante 3: el módulo torch desde plantilla usa el init de stdlib ----

def test_template_module_uses_stdlib_init_scheme_not_native_kaiming():
    """Un `nn.Linear` construido desde plantilla (with_values=False) NO debe
    quedar con el default de torch: pesos ReLU he_normal, resto xavier, sesgo
    CERO. El default `kaiming_uniform_(√5)` dejaría sesgos no nulos — es lo que
    verificamos que NO ocurre."""
    import torch  # noqa: F401
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.forward.dense_torch import dense_network_to_torch_module

    mxai = (
        "PROJECT P\n\nVECTOR In[1]\n  x: Scalar\nEND\n"
        "NETWORK Net\n  INPUT In\n  LAYER Dense units=8 activation=relu\n"
        "  LAYER Dense units=1 activation=linear\n  OUTPUT y: Scalar\nEND\n"
        "GRAPH\n  In -> Net\nEND\n"
    )
    program = parse_text(mxai)
    net = program.networks[0]
    vmap = {v.name: v for v in program.vectors}
    resolved = check_network_types(net, vmap).resolved_layers or net.layers
    template = build_network_parameter_set(
        net, resolved, program_hash(program), seed=42, with_values=False,
    )
    module = dense_network_to_torch_module(net, template)

    linears = list(module._linears)
    # Todos los sesgos exactamente a cero (el default de nn.Linear NO lo sería).
    for lin in linears:
        assert float(lin.bias.detach().abs().max()) == 0.0, "el sesgo debería ser cero"
    # Los pesos ReLU con desviación típica ~ he_normal (√(2/fan_in)); muy por
    # debajo del bound uniforme de kaiming_uniform_(√5) para fan_in=1 (~1.0).
    relu_layer = linears[0]  # 1→8 con ReLU
    std = float(relu_layer.weight.detach().std())
    assert 0.2 < std < 5.0, f"std de pesos ReLU fuera de rango he_normal: {std}"


# --- Invariante 5: el camino stdlib no cambia -------------------------------

def test_stdlib_kelvin_unchanged():
    csv = _linear_csv("centigrados", "prediccionKelvin", lambda c: c + 273.15)
    r2_stdlib = _train_r2(csv, "prediccionKelvin", "stdlib")
    assert r2_stdlib >= 0.99, f"stdlib regresión (contrato 59): R²={r2_stdlib}"


# --- Invariante 6: composite sin LayerNorm usa el mismo esquema de init -----

def test_composite_dense_template_module_uses_stdlib_init_scheme():
    """Mismo invariante 3 pero para `composite_torch.py` — sus capas Dense
    construyen `nn.Linear` con el mismo patrón que dense_torch.py y el mismo
    fix aplica (`layer.activation` ya disponible en `_build_layer_module`)."""
    import torch  # noqa: F401
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.forward.composite_torch import composite_network_to_torch_module

    mxai = (
        "PROJECT P\n\nVECTOR Input[1]\n  x: Scalar\nEND\n"
        "NETWORK Net\n  INPUT Input\n  LAYER Dense units=8 activation=relu\n"
        "  BLOCK r1\n    LAYER Dense units=8 activation=relu\n"
        "    RESIDUAL FROM PREVIOUS\n  END\n"
        "  LAYER Dense units=1 activation=linear\n  OUTPUT y: Scalar\nEND\n"
        "GRAPH\n  Input -> Net\nEND\n"
    )
    program = parse_text(mxai)
    net = program.networks[0]
    type_result = check_composite_network_types(net, {v.name: v for v in program.vectors})
    template = build_composite_network_parameter_set(
        net, type_result, model_hash_str=program_hash(program), seed=42,
        with_values=False,
    )
    module = composite_network_to_torch_module(net, template)

    import torch.nn as nn
    for sub in module.sublayers.values():
        if isinstance(sub, nn.Linear):
            assert float(sub.bias.detach().abs().max()) == 0.0, "el sesgo debería ser cero"


def test_composite_regression_without_layernorm_learns_on_torch():
    """El síntoma del contrato (dead-ReLU en regresión de pocas features) y el
    mismo fix, ahora por el camino composite (Dense + BLOCK con RESIDUAL, sin
    LayerNorm — con LayerNorm hay un colapso DISTINTO, ver C4/§Fuera de
    alcance, no es objeto de este test)."""
    import time as _time
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash
    from matrixai.training.composite_torch_trainer import train_composite_network_torch
    from matrixai.forward.composite_torch import composite_network_to_torch_module, composite_torch_forward
    import statistics

    mxai = (
        "PROJECT P\n\nVECTOR Input[1]\n  x: Scalar\nEND\n"
        "NETWORK Net\n  INPUT Input\n  LAYER Dense units=32 activation=relu\n"
        "  BLOCK r1\n    LAYER Dense units=32 activation=relu\n"
        "    RESIDUAL FROM PREVIOUS\n  END\n"
        "  LAYER Dense units=16 activation=relu\n"
        "  LAYER Dense units=1 activation=linear\n  OUTPUT y: Scalar\nEND\n"
        "GRAPH\n  Input -> Net\nEND\n"
    )
    program = parse_text(mxai)
    net = program.networks[0]
    type_result = check_composite_network_types(net, {v.name: v for v in program.vectors})
    ps = build_composite_network_parameter_set(
        net, type_result, model_hash_str=program_hash(program), seed=42,
        with_values=False,
    )

    examples = []
    for c in range(100):
        xn = (c - (-10)) / (109 - (-10))
        yn = (c + 273.15 - 263.25) / (382.05 - 263.25)
        examples.append(({"x": xn}, [yn]))
    n_train = int(len(examples) * 0.8)
    train_ex, val_ex = examples[:n_train], examples[n_train:]

    result = train_composite_network_torch(
        net, ps, train_ex, "mse", lr=0.01, epochs=50, device="cpu", seed=42,
        batch_size=8, optimizer="sgd", validation_examples=val_ex,
    )
    module = composite_network_to_torch_module(net, result["best_params"])
    preds = [composite_torch_forward(module, d, False)[0] for d, _ in val_ex]
    targets = [t[0] for _, t in val_ex]
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, targets))
    ss_tot = sum((t - statistics.mean(targets)) ** 2 for t in targets)
    r2 = 1 - ss_res / ss_tot
    assert r2 >= 0.80, f"composite (sin LayerNorm) no aprendió: R²={r2}"
