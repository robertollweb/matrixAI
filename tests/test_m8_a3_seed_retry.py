"""M8-A3 — Reintento con otra inicialización ante colapso.

El colapso dying-ReLU es estocástico: una semilla de inicialización distinta
suele resolverlo. El backend acepta `seed` en el camino de entrenamiento
(dense y composite) y lo usa para construir los parámetros iniciales.
"""
import time


def _flat(values):
    out = []
    def f(x):
        if isinstance(x, list):
            for y in x:
                f(y)
        else:
            out.append(round(float(x), 6))
    f(values)
    return out


DENSE_MXAI = """PROJECT P
VECTOR In[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""
DENSE_TRAIN = """MODEL P.mxai
DATASET D
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [a, b, c]
  TARGET y: Label[A, B]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
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
RUN
  EPOCHS 1
END
"""


def _csv(n=20):
    rows = ["a,b,c,y"]
    for i in range(n):
        rows.append(f"{(i % 10) / 10},{(i % 5) / 5},{(i % 3) / 3},{'A' if i % 2 else 'B'}")
    return "\n".join(rows) + "\n"


class TestDenseSeed:
    def test_different_seeds_differ(self):
        from matrixai.playground import _run_playground_dense_training
        r1 = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=1, seed=42)
        r2 = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=1, seed=999)
        assert r1["ok"] and r2["ok"]
        w1 = _flat(r1["params_best"]["parameters"]["Net.W1"]["values"])
        w2 = _flat(r2["params_best"]["parameters"]["Net.W1"]["values"])
        assert w1 != w2

    def test_same_seed_reproducible(self):
        from matrixai.playground import _run_playground_dense_training
        r1 = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=1, seed=7)
        r2 = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=1, seed=7)
        w1 = _flat(r1["params_best"]["parameters"]["Net.W1"]["values"])
        w2 = _flat(r2["params_best"]["parameters"]["Net.W1"]["values"])
        assert w1 == w2

    def test_he_normal_init_already_in_core(self):
        # M8-A2 is effectively already satisfied: dense params use He normal init
        from matrixai.playground import _run_playground_dense_training
        r = _run_playground_dense_training(DENSE_MXAI, DENSE_TRAIN, _csv(), epochs_override=1, seed=1)
        assert r["params_best"]["parameters"]["Net.W1"].get("initializer") == "he_normal"


class TestCompositeSeed:
    def test_different_seeds_differ(self):
        from matrixai.playground import _run_playground_composite_training
        mxai = """PROJECT C
VECTOR In[3]
  a: Scalar
  b: Scalar
  c: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  BLOCK r1
    LAYER Dense units=8 activation=relu
    LAYER LayerNorm
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""
        tr = DENSE_TRAIN.replace("PROJECT P", "PROJECT C").replace("P.mxai", "C.mxai")
        from matrixai.playground import _run_playground_composite_training as run
        r1 = run(mxai, tr, _csv(), epochs_override=1, seed=42)
        r2 = run(mxai, tr, _csv(), epochs_override=1, seed=999)
        assert r1["ok"], r1.get("error")
        assert r2["ok"], r2.get("error")
        # some weight tensor must differ between seeds
        p1, p2 = r1["params_best"]["parameters"], r2["params_best"]["parameters"]
        key = next(k for k in p1 if k.endswith(".W"))
        assert _flat(p1[key]["values"]) != _flat(p2[key]["values"])


class TestSubmitJobSeed:
    def test_submit_accepts_seed(self):
        from matrixai.playground import _submit_training_job, _get_job_status
        job = _submit_training_job(DENSE_MXAI, DENSE_TRAIN, _csv(40), epochs_override=2, seed=123)
        assert job.get("ok"), job
        for _ in range(240):
            st = _get_job_status(job["job_id"])
            if st["status"] != "running":
                break
            time.sleep(0.25)
        assert st["status"] == "done", st
        assert st.get("params_best") is not None
