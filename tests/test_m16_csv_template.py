"""M16 — plantilla del CSV subido + validación de cabecera accionable.

`_csv_template` devuelve las columnas exactas esperadas (features + target con su nombre
semántico) y un CSV de ejemplo descargable. `_validate_training_csv` da un mensaje
accionable (es/en) cuando la cabecera no casa, en vez del error técnico del verificador.
"""
from __future__ import annotations

from matrixai.playground import _csv_template, _validate_training_csv

_MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=4 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT estado: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""
_TRAIN = (
    "MODEL P.mxai\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
    "  INPUT In FROM COLUMNS [a, b]\n  TARGET estado: Label[A, B]\n"
    "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=16\nEND\n"
    "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET estado\nEND\n"
    "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE Net.*\nEND\n"
)


# ── plantilla ───────────────────────────────────────────────────────────────────

def test_template_columns_and_target():
    t = _csv_template(_MXAI, _TRAIN)
    assert t["ok"]
    assert t["columns"] == ["a", "b", "estado"]
    assert t["input_columns"] == ["a", "b"]
    assert t["target_column"] == "estado"        # nombre semántico, NO "predicted_class"
    assert t["labels"] == ["A", "B"]
    assert t["is_classification"] is True


def test_template_csv_is_downloadable_and_parseable():
    t = _csv_template(_MXAI, _TRAIN)
    lines = t["template_csv"].strip().splitlines()
    assert lines[0] == "a,b,estado"          # cabecera exacta
    assert len(lines) >= 2                    # al menos una fila de ejemplo
    # cada fila de ejemplo tiene el nº de columnas correcto
    for row in lines[1:]:
        assert len(row.split(",")) == 3


def test_template_regression_target():
    mxai = _MXAI.replace("OUTPUT estado: ProbabilityMap[A, B]", "OUTPUT estado: Scalar") \
                .replace("units=2 activation=softmax", "units=1 activation=linear")
    train = _TRAIN.replace("TARGET estado: Label[A, B]", "TARGET estado: Scalar")
    t = _csv_template(mxai, train)
    assert t["ok"]
    assert t["is_classification"] is False
    assert t["labels"] == []


def test_template_requires_inputs():
    assert _csv_template("", _TRAIN)["ok"] is False


# ── validación de cabecera accionable ─────────────────────────────────────────────

def test_validate_wrong_columns_actionable():
    r = _validate_training_csv(_MXAI, _TRAIN, "x,y,clase\n0.1,0.2,A\n")
    assert r["ok"] is False
    assert r["error_kind"] == "csv_columns"
    assert r["missing_columns"] == ["a", "b", "estado"]
    assert r["found_columns"] == ["x", "y", "clase"]
    assert r["target_column"] == "estado"
    assert "estado" in r["error"] and "estado" in r["error_en"]


def test_validate_partial_missing_target():
    """Inputs correctos pero falta la columna objetivo → lo dice claramente."""
    r = _validate_training_csv(_MXAI, _TRAIN, "a,b\n0.1,0.2\n")
    assert r["ok"] is False
    assert r["error_kind"] == "csv_columns"
    assert r["missing_columns"] == ["estado"]


def test_validate_ok_includes_expected_columns():
    r = _validate_training_csv(_MXAI, _TRAIN, "a,b,estado\n0.1,0.2,A\n0.3,0.4,B\n")
    assert r["ok"] is True
    assert r["expected_columns"] == ["a", "b", "estado"]


def test_validate_tolerates_reordered_columns():
    """Las columnas se leen por nombre → el orden distinto es válido."""
    r = _validate_training_csv(_MXAI, _TRAIN, "estado,a,b\nA,0.1,0.2\nB,0.3,0.4\n")
    assert r["ok"] is True


def test_validate_extra_column_ok_with_warning():
    """Una columna extra no bloquea (se ignora) pero se avisa — coherente con el texto."""
    r = _validate_training_csv(_MXAI, _TRAIN, "a,b,estado,extra\n0.1,0.2,A,9\n0.3,0.4,B,9\n")
    assert r["ok"] is True
    assert r["extra_columns"] == ["extra"]
    assert any("extra" in w for w in r.get("warnings", []))


def test_template_note_does_not_claim_exact():
    """El texto de la plantilla NO debe prometer 'exactas' (contradeciría la tolerancia)."""
    t = _csv_template(_MXAI, _TRAIN)
    assert "exactamente estas columnas" not in t["note"]
    assert "extra se ignoran" in t["note"]
