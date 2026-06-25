# SPDX-License-Identifier: AGPL-3.0-only
"""M10 (opción A GLOBAL, 2026-06-24) — la generación de dataset NO ejecuta la red.

Histórico: el etiquetado `coherent` corría la red (runtime Python por fila, luego forward
batched torch) → con redes anchas/profundas se colgaba. Decisión del autor: la generación
NUNCA instancia ni ejecuta la red para etiquetar. Las etiquetas con señal vienen de
`domain_rules` (reglas del LLM, deterministas, sin red); sin reglas, etiquetas aleatorias.
Estos tests fijan el nuevo contrato y son guarda de regresión contra reintroducir la red.
"""
from __future__ import annotations

import time

DENSE_MXAI = """PROJECT P
VECTOR In[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  In -> Net
END
"""

DENSE_TRAIN = """MODEL P.mxai
DATASET D
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [a, b, c]
  TARGET y: Label[A, B, C]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=16
END
LOSS L
  TYPE cross_entropy
  PREDICTION Net
  TARGET y
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.05
  UPDATE Net.*
END
"""

# Red ANCHA y PROFUNDA: bajo el camino viejo, etiquetar en modo coherent construía y
# ejecutaba estos ~46M de parámetros → se colgaba. Con opción A debe ser instantáneo.
WIDE_MXAI = """PROJECT W
VECTOR In[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT In
""" + "".join(f"  LAYER Dense units=2048 activation=relu\n" for _ in range(12)) + """  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  In -> Net
END
"""
WIDE_TRAIN = DENSE_TRAIN.replace("MODEL P.mxai", "MODEL W.mxai")


def _gen(mxai, train, rows, mode="coherent", seed=42):
    from matrixai.parser import parse_text
    from matrixai.training.parser import parse_training_text
    from matrixai.training.synthetic import SyntheticDataGenerator
    program = parse_text(mxai)
    spec = parse_training_text(train)
    return SyntheticDataGenerator(program, spec, seed=seed, rows=rows, mode=mode)


def test_coherent_without_rules_equals_random():
    # Sin domain_rules, coherent == random (mismo rng, sin red): misma secuencia de labels.
    coh = [e.label for e in _gen(DENSE_MXAI, DENSE_TRAIN, rows=300, mode="coherent", seed=7).generate().examples()]
    rnd = [e.label for e in _gen(DENSE_MXAI, DENSE_TRAIN, rows=300, mode="random", seed=7).generate().examples()]
    assert coh == rnd


def test_coherent_is_reproducible():
    ex1 = [e.label for e in _gen(DENSE_MXAI, DENSE_TRAIN, rows=400, seed=3).generate().examples()]
    ex2 = [e.label for e in _gen(DENSE_MXAI, DENSE_TRAIN, rows=400, seed=3).generate().examples()]
    assert ex1 == ex2
    assert set(ex1).issubset({"A", "B", "C"})
    assert len(set(ex1)) >= 2  # no colapsa a una sola clase


def test_coherent_does_not_run_the_network_no_attrs():
    # La maquinaria de etiquetado-por-red fue retirada: estos atributos ya no existen.
    gen = _gen(DENSE_MXAI, DENSE_TRAIN, rows=50)
    assert not hasattr(gen, "_build_torch_label_module")
    assert not hasattr(gen, "_assign_torch_labels")


def test_wide_deep_net_generates_instantly():
    # Guarda de regresión: una red 12×2048 que antes colgaba al etiquetar debe generar
    # al instante (no se construye ni ejecuta la red). Margen amplio para CI lento.
    t0 = time.time()
    gen = _gen(WIDE_MXAI, WIDE_TRAIN, rows=3000)
    adapter = gen.generate()
    dt = time.time() - t0
    ex = list(adapter.examples())
    assert len(ex) == 3000
    assert {e.label for e in ex}.issubset({"A", "B", "C"})
    assert dt < 10.0, f"la generación tardó {dt:.1f}s — ¿se reintrodujo el etiquetado por red?"
