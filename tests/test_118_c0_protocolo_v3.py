"""118-C0 — el protocolo v3 de la red densa, registrado ANTES de construir las palancas.

Ata que la v3 no mueve nada de la v2 (conjuntos, sellado, partición, presupuesto, regla,
métricas: copiados byte a byte), que su digest es el de su contenido, que el JSON es el que
compone su guion, y que cada palanca que se va a medir tiene sus parámetros fijados."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
sys.path.insert(0, str(FASE0))
import generar_protocolo_118_v3 as g  # noqa: E402

V2 = json.loads((FASE0 / "protocolo_exploratorio_v2.json").read_text(encoding="utf-8"))
V3 = json.loads((FASE0 / "protocolo_118_v3.json").read_text(encoding="utf-8"))


def test_la_v3_copia_de_la_v2_todo_lo_que_decide_el_liston_y_el_reparto():
    for campo in ("datasets", "particion", "presupuesto", "recursos_declarados",
                  "regla_de_cierre", "metricas_por_tarea"):
        assert V3[campo] == V2[campo], campo
    assert V3["base"]["digest_sha256"] == V2["digest_sha256"]
    assert V3["regla_de_cierre"]["puntos"] == 2.0 and V3["regla_de_cierre"]["fraccion_minima"] == 0.8


def test_el_digest_es_el_de_su_contenido():
    sin = {k: v for k, v in V3.items() if k != "digest_sha256"}
    canonico = json.dumps(sin, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert V3["digest_sha256"] == hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def test_el_json_es_el_que_compone_su_guion():
    assert g.main(["--comprobar"]) == 0


def test_las_palancas_que_se_miden_tienen_sus_parametros_fijados():
    por_id = {p["id"]: p for p in V3["palancas"]}
    for pid in ("118-C1.cuantiles", "118-C2.objetivo", "118-C3.receta"):
        assert por_id[pid]["parametros"], pid
    for pid in ("118-C4.embeddings", "118-C5.semillas"):
        assert por_id[pid]["parametros"] is None and "ENMIENDA" in por_id[pid]["registro_pendiente"]
    assert por_id["118-C3.receta"]["parametros"]["plazo"].startswith("el mismo del motor")


def test_la_receta_base_es_la_de_la_v2():
    base = next(m["receta"] for m in V2["motores"] if m["id"] == "matrixai.dense.torch_cpu")
    assert V3["receta_base"] == base


def test_la_enmienda_1_de_C3_esta_atada_a_la_v3_y_dice_lo_que_no_se_aplica():
    """Enmienda registrada ANTES de medir 118-C3: el lenguaje del núcleo no admite AdamW,
    weight_decay ni programa coseno (comprobado ejecutándolo). Se mide la parte aplicable
    y lo que falta queda como palanca propia."""
    e = json.loads((FASE0 / "protocolo_118_v3_enmienda_1.json").read_text(encoding="utf-8"))
    assert e["de"]["digest_sha256"] == V3["digest_sha256"]
    assert e["palanca"] == "118-C3.receta"
    assert set(e["no_se_aplica"]) == {"optimizador_adamw", "weight_decay", "programa_coseno"}
    sin = {k: v for k, v in e.items() if k != "digest_sha256"}
    canonico = json.dumps(sin, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert e["digest_sha256"] == hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def test_la_enmienda_2_registra_la_receta_completa_atada_a_la_v3_y_a_la_1():
    e1 = json.loads((FASE0 / "protocolo_118_v3_enmienda_1.json").read_text(encoding="utf-8"))
    e2 = json.loads((FASE0 / "protocolo_118_v3_enmienda_2.json").read_text(encoding="utf-8"))
    assert e2["de"]["digest_sha256"] == V3["digest_sha256"]
    assert e2["de"]["digest_enmienda_anterior"] == e1["digest_sha256"]
    p = e2["parametros"]
    assert (p["optimizador"], p["weight_decay"], p["learning_rate"]) == ("adamw", 0.0001, 0.001)
    assert p["epocas_maximas"] == 200 and p["parada_temprana"]["paciencia"] == 20
    sin = {k: v for k, v in e2.items() if k != "digest_sha256"}
    canonico = json.dumps(sin, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert e2["digest_sha256"] == hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def test_la_enmienda_3_sustituye_C3_por_C3b_atada_a_la_v3_y_a_la_2_sin_registrar_palanca_nueva():
    """Enmienda registrada el 26-09 ANTES de medir C3 y C3b (decisión de Roberto, «solo C3b»):
    C3 no se mide porque C3b es su receta tal como se registró. No trae `conjuntos` ni
    `parametros`: si los trajera, el arnés (`_palancas_de_las_enmiendas`) la leería como una
    palanca NUEVA medible."""
    e2 = json.loads((FASE0 / "protocolo_118_v3_enmienda_2.json").read_text(encoding="utf-8"))
    e3 = json.loads((FASE0 / "protocolo_118_v3_enmienda_3.json").read_text(encoding="utf-8"))
    assert e3["de"]["digest_sha256"] == V3["digest_sha256"]
    assert e3["de"]["digest_enmienda_anterior"] == e2["digest_sha256"]
    assert e3["palanca"] == "118-C3.receta"
    assert "118-C3b.receta_completa" in e3["que_cambia"]
    assert "conjuntos" not in e3 and "parametros" not in e3
    sin = {k: v for k, v in e3.items() if k != "digest_sha256"}
    canonico = json.dumps(sin, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert e3["digest_sha256"] == hashlib.sha256(canonico.encode("utf-8")).hexdigest()
