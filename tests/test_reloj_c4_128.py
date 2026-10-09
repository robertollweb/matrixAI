"""128-C4: separar lo que pone el RELOJ de lo que pone el CÓDIGO, intento a intento. Con el mismo trabajo (árboles o
épocas), el valor tiene que ser idéntico; si no, es una violación."""
import importlib.util
import json
from pathlib import Path

RUTA = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "reloj_c4_128.py"
spec = importlib.util.spec_from_file_location("reloj_c4_128_bajo_prueba", RUTA)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def arboles(cand, n, valor, paro="reloj"):
    return {"candidate": cand, "estado": "completed", "metrica_de_seleccion": {"value": valor},
            "config_efectiva": {"hiperparametros": {"politica_por_tamano": {"arboles_entrenados": n, "paro": paro}}}}


def tabm(cand, epocas, valor, parcial=False):
    return {"candidate": cand, "estado": "completed", "metrica_de_seleccion": {"value": valor},
            "config_efectiva": {"entrenamiento_efectivo": {"epocas_ejecutadas": epocas, "ultima_epoca_parcial": parcial,
                                                           "parado_por_plazo": True}}}


def por_cand(*es):
    return {e["candidate"]: e for e in es}


def test_mismo_trabajo_mismo_valor_y_lo_distinto_es_del_reloj():
    ref = por_cand(arboles("hgb-p0", 149, 0.98), tabm("tabm-p0", 8, 0.96), tabm("tabm-p1", 9, 0.95))
    nuevo = por_cand(arboles("hgb-p0", 149, 0.98), tabm("tabm-p0", 9, 0.97), tabm("tabm-p1", 9, 0.95))
    r = R.comparar_conjunto(ref, nuevo)
    assert r["violaciones"] == [] and r["misma_cantidad"] == 2 and r["misma_cantidad_mismo_valor"] == 2
    assert [d["candidato"] for d in r["del_reloj"]] == ["tabm-p0"]
    assert r["cantidad_media"]["epocas"] == {"ref": 8.5, "nuevo": 9.0, "intentos": 2}


def test_TIENE_DIENTES_mismo_trabajo_y_otro_valor_es_una_violacion():
    r = R.comparar_conjunto(por_cand(tabm("tabm-p0", 9, 0.95)), por_cand(tabm("tabm-p0", 9, 0.951)))
    assert len(r["violaciones"]) == 1
    r = R.comparar_conjunto(por_cand(tabm("tabm-p0", 9, 0.95, parcial=True)), por_cand(tabm("tabm-p0", 9, 0.951)))
    assert r["violaciones"] == [] and len(r["del_reloj"]) == 1      # una época a medias no es el mismo trabajo


def test_un_motor_sin_reloj_con_otro_valor_tambien_es_violacion():
    sin = {"candidate": "lgbm-p0", "estado": "completed", "metrica_de_seleccion": {"value": 0.9}, "config_efectiva": {}}
    r = R.comparar_conjunto(por_cand(sin), por_cand(dict(sin, metrica_de_seleccion={"value": 0.91})))
    assert len(r["sin_cantidad_distinto_valor"]) == 1


def test_por_directorios_y_codigo_de_salida(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.json").write_text(json.dumps({"leaderboard": [tabm("tabm-p0", 9, 0.95)]}))
    (tmp_path / "b" / "x.json").write_text(json.dumps({"leaderboard": [tabm("tabm-p0", 9, 0.95)]}))
    assert R._main([str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    (tmp_path / "b" / "x.json").write_text(json.dumps({"leaderboard": [tabm("tabm-p0", 9, 0.96)]}))
    assert R._main([str(tmp_path / "a"), str(tmp_path / "b")]) == 1
