"""165-C1: «Filas a revisar» sin etiquetas. Datos de forma real: tildes, mayúsculas, coma decimal, columna vacía, un id."""
import json
import random
import subprocess
import sys

import pytest

from matrixai.estudio.anomalias import NO_AFIRMA, leer_csv, puntuar_filas


def _datos(n=300, semilla=1):
    r = random.Random(semilla)
    f = []
    for i in range(n):
        f.append({"id": f"CLI-{i:05d}", "Ciudad": r.choice(["Santander", "Málaga", "ÁVILA", "Logroño"]),
                  "Importe": f"{r.gauss(100, 10):.2f}".replace(".", ","), "Unidades": str(r.randint(1, 5)),
                  "Vacia": "", "Const": "X"})
    f[10].update(Importe="9999,50", Unidades="400", Ciudad="Ñuñoa")
    return f


def test_la_fila_sembrada_es_la_primera_y_el_orden_es_descendente():
    res = puntuar_filas(_datos())
    lista = res["filas_a_revisar"]
    assert lista[0]["fila"] == 11 and lista[0]["rango"] == 1
    assert [x["rango"] for x in lista] == list(range(1, 301))
    pts = [x["puntuacion"] for x in lista]
    assert pts == sorted(pts, reverse=True)
    assert {c["columna"] for c in lista[0]["columnas_a_la_cola"]} == {"Ciudad", "Importe", "Unidades"}


def test_identificador_constante_y_vacia_se_excluyen_y_se_dicen():
    res = puntuar_filas(_datos())
    exc = {e["columna"]: e["motivo"] for e in res["columnas_excluidas"]}
    assert set(exc) == {"id", "Vacia", "Const"}
    assert "identificador" in exc["id"] and "vacia" in exc["Vacia"] and "constante" in exc["Const"]
    assert {c["columna"] for c in res["columnas_usadas"]} == {"Ciudad", "Importe", "Unidades"}


def test_coma_decimal_es_numerica_no_categorica():
    tipos = {c["columna"]: c["tipo"] for c in puntuar_filas(_datos())["columnas_usadas"]}
    assert tipos["Importe"] == "numerica" and tipos["Ciudad"] == "categorica"


def test_un_id_numerico_consecutivo_tambien_se_excluye():
    f = [{"n": str(i), "x": str(random.Random(i).gauss(0, 1))} for i in range(1, 250)]
    res = puntuar_filas(f)
    assert [e["columna"] for e in res["columnas_excluidas"]] == ["n"]


def test_sin_cifras_de_acierto_y_con_no_afirma():
    res = puntuar_filas(_datos())
    txt = json.dumps(res, ensure_ascii=False).lower()
    for prohibida in ("accuracy", "exactitud\":", "auroc\":", "sensibilidad\":", "precision\":", "n_anomalias", "proporcion_anomalas"):
        assert prohibida not in txt
    assert res["etiquetas"] == "ninguna" and res["detector"]["contamination"] is None
    assert res["no_afirma"] == NO_AFIRMA and len(NO_AFIRMA) >= 5
    assert any("exactitud" in x and "no se calcula" in x for x in res["no_afirma"])


def test_estabilidad_con_cinco_semillas_y_aviso_si_baja():
    res = puntuar_filas(_datos())
    assert len(res["estabilidad"]["semillas"]) == 5 and 0 <= res["estabilidad"]["solape_medio"] <= 1


def test_aviso_si_la_lista_cambia_con_la_semilla(monkeypatch):
    import numpy as np
    import matrixai.estudio.anomalias as m
    monkeypatch.setattr(m, "_puntuar", lambda X, semilla: np.random.RandomState(semilla).rand(len(X)))
    res = puntuar_filas(_datos(), k=15)
    assert res["estabilidad"]["solape_medio"] < 0.5
    assert any("cambia con la semilla" in a for a in res["avisos"])
    monkeypatch.undo()
    assert not any("cambia con la semilla" in a for a in puntuar_filas(_datos())["avisos"])


def test_determinista_con_la_misma_semilla():
    assert puntuar_filas(_datos(), semilla=7) == puntuar_filas(_datos(), semilla=7)


def test_ninguna_fila_se_pierde_en_silencio_y_las_sin_dato_se_declaran():
    f = _datos()
    f[5].update(Ciudad="", Importe="", Unidades="")
    res = puntuar_filas(f)
    assert res["datos"]["filas_ignoradas"] == [6] and res["datos"]["filas_puntuadas"] == 299
    assert len(res["filas_a_revisar"]) == 299 and any("sin ningun dato" in a for a in res["avisos"])


def test_faltante_no_es_cero():
    f = _datos()
    for i in range(0, 300, 3):
        f[i]["Importe"] = ""
    res = puntuar_filas(f)
    assert res["filas_a_revisar"][0]["fila"] == 11     # un hueco (imputado por la mediana) no se vuelve un valor extremo


def test_pocas_filas_avisa_y_entrada_invalida_falla_con_motivo():
    assert any("no es fiable" in a for a in puntuar_filas(_datos(50))["avisos"])
    with pytest.raises(ValueError, match="no existen"):
        puntuar_filas(_datos(), ["Nada"])
    with pytest.raises(ValueError, match="ninguna columna util"):
        puntuar_filas(_datos(), ["id", "Const"])


def test_no_muta_la_entrada():
    f = _datos(); copia = [dict(x) for x in f]
    puntuar_filas(f)
    assert f == copia


def test_cli_csv_con_punto_y_coma_y_tildes(tmp_path):
    ruta = tmp_path / "d.csv"
    cols = ["id", "Ciudad", "Importe", "Unidades"]
    ruta.write_text("\n".join([";".join(cols)] + [";".join(f[c] for c in cols) for f in _datos()]), encoding="utf-8")
    out = tmp_path / "o.json"
    p = subprocess.run([sys.executable, "-m", "matrixai.estudio.anomalias", str(ruta), "--salida", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["filas_a_revisar"][0]["fila"] == 11 and "no_afirma" in res
    assert leer_csv(str(ruta))[10]["Ciudad"] == "Ñuñoa"
