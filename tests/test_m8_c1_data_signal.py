"""M8-C1 — Datos con señal real / origen de etiquetas honesto.

Un modelo es real solo si los pares (entrada → salida correcta) tienen relación
real. El modo `random` de clasificación produce etiquetas independientes de las
features → colapso al prior: se avisa. El origen de la etiqueta (coherente vs
random) se expone para auditar.
"""
MXAI = (
    "PROJECT P\n\nVECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
    "NETWORK Net\n  INPUT In\n  LAYER Dense units=8 activation=relu\n"
    "  LAYER Dense units=3 activation=softmax\n  OUTPUT y: ProbabilityMap[A, B, C]\nEND\n\n"
    "GRAPH\n  In -> Net\nEND\n"
)
MXTRAIN = (
    "MODEL P.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT In FROM COLUMNS [a, b]\n  TARGET y: Label[A, B, C]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET y\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE Net.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)

# regression model (no labels) — signal warning should NOT trigger on random
REG_MXAI = (
    "PROJECT R\n\nVECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND\n\n"
    "NETWORK Net\n  INPUT In\n  LAYER Dense units=8 activation=relu\n"
    "  LAYER Dense units=1 activation=linear\n  OUTPUT y: Scalar\nEND\n\n"
    "GRAPH\n  In -> Net\nEND\n"
)
REG_TRAIN = (
    "MODEL R.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT In FROM COLUMNS [a, b]\n  TARGET y: Scalar\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
    "LOSS L\n  TYPE mse\n  PREDICTION Net\n  TARGET y\nEND\n\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE Net.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n"
)


class TestLabelOrigin:
    def test_coherent_without_domain_rules_degrades_to_random(self):
        # Opción A: la generación no ejecuta la red para etiquetar. 'coherent' sin
        # reglas de dominio del LLM → valores realistas + etiquetas ALEATORIAS
        # (honesto: sin relación entrada→salida), no etiquetas derivadas de un
        # forward de una red sin entrenar.
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(MXAI, MXTRAIN, 12, 42, "coherent", use_llm=False)
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_random"
        assert "no depende de la entrada" in r["signal_warning"]

    def test_random_classification_warns(self):
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(MXAI, MXTRAIN, 12, 42, "random", use_llm=False)
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_random"
        assert "no depende de la entrada" in r["signal_warning"]

    def test_random_regression_no_signal_warning(self):
        # regression (no class prior to collapse to) → no signal warning
        from matrixai.playground import _generate_synthetic_dataset
        r = _generate_synthetic_dataset(REG_MXAI, REG_TRAIN, 12, 42, "random", use_llm=False)
        assert r["ok"], r.get("error")
        assert r["label_origin"] == "synthetic_random"
        assert "signal_warning" not in r
