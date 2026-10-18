# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""CONTRATO 133-C2 (núcleo) — el LABORATORIO de la red: lo medido del entrenamiento.

Lo que estas pruebas sostienen, cada una con su nombre (y cada una con su sabotaje en
rojo, ver el informe del corte):
- `test_captura_no_cambia_el_entrenamiento`: pedir el laboratorio NO mueve ni un bit de lo
  que se entrena; y sin pedirlo, el resultado no trae ni la clave;
- `test_normas_de_gradiente_son_las_medidas`: la norma por capa y época es la de un cálculo
  a mano en una red mínima (media de las normas de los lotes, antes del weight_decay);
- `test_peso_pintado_es_el_medido`: los pesos del bloque SON los de `best_params` (los de
  «Probar una fila») y los de cada época de la muestra reproducen la pérdida que se midió;
- `test_la_muestra_incluye_primera_mejor_ultima`;
- `test_tope_de_tamano_se_declara`: lo que no cabe sale en `omitido` con lo pedido, lo
  aplicado y el motivo (es/en), no recortado en silencio;
- `test_sin_torch_declara_que_no`;
- `test_composite_no_trae_laboratorio`.
"""
from __future__ import annotations

import math
import random
from importlib import util

import pytest

_HAS_TORCH = util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
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

MXAI_UNA_CAPA = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=1 activation=sigmoid
  OUTPUT y: Probability
END
GRAPH
  In -> Net
END
"""


def _setup(texto=MXAI):
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(texto)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    ps = build_network_parameter_set(net, rl, program_hash(prog), seed=1)
    return net, ps


def _ejemplos(n=200, seed=0):
    rng = random.Random(seed)
    filas = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        filas.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))
    return filas


def _entrenar(capturar=None, n=200, epochs=6, **kw):
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    net, ps = _setup()
    kw.setdefault("lr", 0.05)
    kw.setdefault("batch_size", 8)
    return train_dense_network_torch(
        net, ps, _ejemplos(n), "cross_entropy", epochs=epochs, device="cpu", seed=3,
        materialize=True, capturar=capturar, **kw)


def _pesos(res):
    return {k: v["values"] for k, v in res["best_params"].parameters.items()}


def _csv_proyecto(n=120, compuesto=False):
    from matrixai.playground_api import generate_project_from_dataset
    rng = random.Random(0)
    if compuesto:
        lineas = ["x,c,y"] + [
            f"{rng.random():.4f},v{rng.randrange(300)},{'si' if rng.random() > 0.5 else 'no'}"
            for _ in range(600)]
    else:
        lineas = ["a,b,y"] + [
            f"{rng.random():.4f},{rng.random():.4f},{'si' if i % 2 else 'no'}" for i in range(n)]
    return generate_project_from_dataset("\n".join(lineas) + "\n", "y", locale="es",
                                         column_type_overrides={"y": "categorical"})


def _entrenar_playground(monkeypatch, backend, gen=None, **kw):
    from matrixai import limits
    from matrixai.playground import MODO_TORCH_EN_CPU
    from matrixai.playground_api import run_playground_training
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", MODO_TORCH_EN_CPU if backend == "torch" else "stdlib")
    gen = gen or _csv_proyecto()
    with limits.sin_topes():
        return run_playground_training(
            gen["mxai"], gen["training_text"], gen["csv_text"], epochs_override=kw.pop("epochs", 5),
            field_ranges=gen.get("field_ranges"), seed=1, **kw)


# ---------------------------------------------------------------------------
@needs_torch
@pytest.mark.parametrize("extra", [
    {},
    {"optimizer": "adamw", "weight_decay": 0.01, "schedule": "cosine"},
])
def test_captura_no_cambia_el_entrenamiento(extra):
    from matrixai.training.laboratorio import CapturaSpec
    sin = _entrenar(**extra)
    con = _entrenar(capturar=CapturaSpec(), **extra)
    # Sin pedirlo, ni la clave: el resultado es el de siempre.
    assert "laboratorio" not in sin
    assert "laboratorio" in con and con["laboratorio"]["disponible"] is True
    # Pedirlo no mueve NI UN BIT: pesos, trazas y números del resultado.
    assert _pesos(sin) == _pesos(con)
    assert sin["epochs"] == con["epochs"]
    assert sin["best_epoch"] == con["best_epoch"] and sin["best_val_loss"] == con["best_val_loss"]
    assert sin["train_loss"] == con["train_loss"]
    resto_sin = {k: v for k, v in sin.items() if k not in ("best_params",)}
    resto_con = {k: v for k, v in con.items() if k not in ("best_params", "laboratorio")}
    assert resto_sin == resto_con


@needs_torch
def test_captura_no_cambia_el_entrenamiento_por_la_superficie_publica(monkeypatch):
    from matrixai.training.laboratorio import CapturaSpec
    sin = _entrenar_playground(monkeypatch, "torch")
    con = _entrenar_playground(monkeypatch, "torch", capturar=CapturaSpec())
    assert sin["ok"] and con["ok"] and sin["backend"] == "torch"
    assert "laboratorio" not in sin and "laboratorio_resumen" not in sin
    assert sin["params_best"] == con["params_best"]
    assert sin["epochs"] == con["epochs"]
    assert con["laboratorio"]["entrenador"] == "torch"
    # Y lo que se pinta (columnas y clases) lo pone quien conoce el modelo.
    assert con["laboratorio"]["filas"]["columnas"] == ["a", "b"]


# ---------------------------------------------------------------------------
def _sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z))


def _gradiente_a_mano(w, b, filas):
    """BCE medio de una neurona sigmoide: dL/dw = mean((p-y)·x), dL/db = mean(p-y)."""
    n = len(filas)
    gw = [0.0] * len(w)
    gb = 0.0
    for x, ys in filas:
        y = ys[0]
        p = _sigmoide(sum(wi * xi for wi, xi in zip(w, x)) + b)
        for j, xj in enumerate(x):
            gw[j] += (p - y) * xj / n
        gb += (p - y) / n
    return math.sqrt(sum(g * g for g in gw)), abs(gb)


def _una_neurona(filas, batch, lr=0.0, weight_decay=0.0, epochs=1, seed=5):
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.training.laboratorio import CapturaSpec
    net, ps = _setup(MXAI_UNA_CAPA)
    ps.parameters["Net.W1"]["values"] = [[0.3, -0.2]]
    ps.parameters["Net.b1"]["values"] = [-0.1]
    return train_dense_network_torch(
        net, ps, filas, "binary_cross_entropy", lr=lr, epochs=epochs, device="cpu", seed=seed,
        batch_size=batch, materialize=True, weight_decay=weight_decay,
        validation_examples=filas[:2], capturar=CapturaSpec(max_filas=0))


FILAS = [([0.9, 0.1], [1.0]), ([0.2, 0.8], [0.0]), ([0.7, 0.6], [1.0]), ([0.1, 0.3], [0.0])]


@needs_torch
def test_normas_de_gradiente_son_las_medidas():
    # (a) UN lote con las 4 filas: la norma de W es la raíz de dw1²+dw2² calculada a mano.
    r = _una_neurona(FILAS, batch=4, weight_decay=0.5)  # el weight_decay NO entra en la medida
    g = r["laboratorio"]["grad_norm"]
    nw, nb = _gradiente_a_mano([0.3, -0.2], -0.1, FILAS)
    assert g["epocas"] == [1]
    assert g["por_capa"][0]["capa"] == 1
    assert g["por_capa"][0]["W"][0] == pytest.approx(nw, abs=1e-6)
    assert g["por_capa"][0]["b"][0] == pytest.approx(nb, abs=1e-6)
    assert nw > 1e-3 and nb > 1e-3          # la medida no es un cero que cuadre por casualidad

    # (b) DOS lotes de 2 filas, con lr=0 (los pesos no se mueven entre lotes): la cifra de la
    # época es la MEDIA de las normas de los dos lotes. El reparto lo da el mismo
    # `randperm` sembrado que usa el entrenador (primera llamada tras manual_seed).
    import torch
    torch.manual_seed(5)
    perm = torch.randperm(4).tolist()
    lotes = [[FILAS[i] for i in perm[0:2]], [FILAS[i] for i in perm[2:4]]]
    r2 = _una_neurona(FILAS, batch=2, seed=5)
    esp = [_gradiente_a_mano([0.3, -0.2], -0.1, lote) for lote in lotes]
    g2 = r2["laboratorio"]["grad_norm"]["por_capa"][0]
    assert g2["W"][0] == pytest.approx((esp[0][0] + esp[1][0]) / 2, abs=1e-6)
    assert g2["b"][0] == pytest.approx((esp[0][1] + esp[1][1]) / 2, abs=1e-6)
    # y es distinta de la norma del gradiente de las 4 filas juntas (la media de normas
    # no es la norma de la media): lo medido es lo que dice su definición.
    assert abs(g2["W"][0] - nw) > 1e-4 or abs(g2["b"][0] - nb) > 1e-4


@needs_torch
def test_normas_de_gradiente_por_capa_y_epoca_en_una_red_de_dos_capas():
    from matrixai.training.laboratorio import CapturaSpec
    r = _entrenar(capturar=CapturaSpec(), epochs=4)
    g = r["laboratorio"]["grad_norm"]
    assert g["epocas"] == [1, 2, 3, 4] and g["parciales"] == []
    assert [c["capa"] for c in g["por_capa"]] == [1, 2]
    for c in g["por_capa"]:
        assert len(c["W"]) == 4 and len(c["b"]) == 4
        assert all(v > 0 and math.isfinite(v) for v in c["W"] + c["b"])
    assert g["definicion"]["es"] and g["definicion"]["en"]


# ---------------------------------------------------------------------------
def _perdida_con_pesos(capas, filas):
    import torch
    from matrixai.forward.dense_torch import dense_network_to_torch_module
    from matrixai.training.dense_torch_trainer import _loss_on_probabilities
    net, ps = _setup()
    for i, c in enumerate(capas, start=1):
        ps.parameters[f"Net.W{i}"]["values"] = c["W"]
        ps.parameters[f"Net.b{i}"]["values"] = c["b"]
    modulo = dense_network_to_torch_module(net, ps)
    modulo.eval()
    x = torch.tensor([f for f, _ in filas], dtype=torch.float32)
    y = torch.tensor([t for _, t in filas], dtype=torch.float32)
    with torch.no_grad():
        return float(_loss_on_probabilities(modulo(x), y, "cross_entropy", torch))


@needs_torch
def test_peso_pintado_es_el_medido():
    from matrixai.training.laboratorio import CapturaSpec
    n, epochs = 200, 8
    r = _entrenar(capturar=CapturaSpec(max_epocas_muestra=5), n=n, epochs=epochs)
    lab = r["laboratorio"]
    valores = _pesos(r)
    # 1. Los pesos finales SON best_params (los de «Probar una fila»), valor a valor.
    assert lab["pesos_finales"]["epoca"] == r["best_epoch"]
    for i, c in enumerate(lab["pesos_finales"]["capas"], start=1):
        assert c["W"] == valores[f"Net.W{i}"]
        assert c["b"] == valores[f"Net.b{i}"]
    # 2. Los de cada época de la muestra reproducen la pérdida de validación que se midió
    # en ESA época: no son una ilustración, son los pesos de esa época.
    val = _ejemplos(n)[160:]
    por_epoca = {e["epoch"]: e["val_loss"] for e in r["epochs"]}
    assert lab["pesos_por_epoca"], "tenía que haber muestra"
    for m in lab["pesos_por_epoca"]:
        assert _perdida_con_pesos(m["capas"], val) == pytest.approx(por_epoca[m["epoca"]], abs=1e-6)
    # 3. Y no son todas la misma: la red cambió de verdad entre la primera y la última.
    primera, ultima = lab["pesos_por_epoca"][0], lab["pesos_por_epoca"][-1]
    assert primera["capas"][0]["W"] != ultima["capas"][0]["W"]
    # La mejor de la muestra es la final.
    mejor = [m for m in lab["pesos_por_epoca"] if "mejor" in m["motivo"]][0]
    assert mejor["capas"] == lab["pesos_finales"]["capas"]


@needs_torch
def test_las_filas_son_las_preparadas_con_su_etiqueta():
    from matrixai.training.laboratorio import CapturaSpec
    spec = CapturaSpec(max_filas=5, columnas=("a", "b"), etiquetas=("A", "B"))
    r = _entrenar(capturar=spec, n=200, epochs=2)
    filas = r["laboratorio"]["filas"]
    val = _ejemplos(200)[160:]
    assert filas["preparadas"] is True and filas["columnas"] == ["a", "b"]
    assert len(filas["x"]) == 5 and filas["de"] == len(val)
    for idx, x, y, et in zip(filas["indices"], filas["x"], filas["y"], filas["etiqueta"]):
        assert x == [pytest.approx(v) for v in val[idx][0]]
        assert y == val[idx][1]
        assert et == ("B" if y == [0.0, 1.0] else "A")


# ---------------------------------------------------------------------------
@needs_torch
def test_la_muestra_incluye_primera_mejor_ultima():
    from matrixai.training.laboratorio import CapturaSpec
    # Lr alto: la pérdida de validación sube y baja y la mejor época cae en MEDIO.
    r = _entrenar(capturar=CapturaSpec(max_epocas_muestra=6), epochs=40, lr=2.0, n=60)
    lab = r["laboratorio"]
    eps = [m["epoca"] for m in lab["pesos_por_epoca"]]
    ultima = r["epochs"][-1]["epoch"]
    assert len(eps) <= 6 and eps == sorted(eps)
    assert 1 in eps and r["best_epoch"] in eps and ultima in eps
    assert 1 < r["best_epoch"] < ultima, "el caso de prueba tenía que tener la mejor en medio"
    motivos = {m["epoca"]: m["motivo"] for m in lab["pesos_por_epoca"]}
    assert "primera" in motivos[1] and "mejor" in motivos[r["best_epoch"]]
    assert "ultima" in motivos[ultima]

    # Con parada temprana la «última» es la última que se ENTRENÓ, no la pedida.
    r2 = _entrenar(capturar=CapturaSpec(), epochs=50, lr=2.0, n=60, early_stop=(2, "validation_loss"))
    ran = r2["epochs"][-1]["epoch"]
    eps2 = [m["epoca"] for m in r2["laboratorio"]["pesos_por_epoca"]]
    assert ran < 50 and ran in eps2 and 1 in eps2 and r2["best_epoch"] in eps2
    assert len(eps2) <= 12

    # El tope por omisión: 12 épocas muestreadas en 50.
    r3 = _entrenar(capturar=CapturaSpec(), epochs=50)
    assert len(r3["laboratorio"]["pesos_por_epoca"]) <= 12
    assert len(r3["laboratorio"]["pesos_por_epoca"]) >= 10


def test_la_muestra_incluye_primera_mejor_ultima_sin_torch_de_por_medio():
    """La elección pura: con pocos huecos manda la prioridad mejor > última > primera."""
    from matrixai.training.laboratorio import elegir_muestra, epocas_programadas
    disp = {e: object() for e in epocas_programadas(50, 12)}
    disp[37] = object()
    disp[44] = object()
    elegidas = dict(elegir_muestra(disp, 1, 37, 44, 12))
    assert len(elegidas) == 12 and {1, 37, 44} <= set(elegidas)
    assert elegidas[37] == ["mejor"] and elegidas[44] == ["ultima"] and elegidas[1] == ["primera"]
    assert set(dict(elegir_muestra(disp, 1, 37, 44, 2))) == {37, 44}
    assert set(dict(elegir_muestra(disp, 1, 37, 44, 3))) == {1, 37, 44}


# ---------------------------------------------------------------------------
@needs_torch
def test_tope_de_tamano_se_declara():
    from matrixai.training.laboratorio import CapturaSpec, captura_de_demo

    def omitidos(res):
        return {o["que"]: o for o in res["laboratorio"]["omitido"]}

    # Sin recortar: nada omitido.
    limpio = _entrenar(capturar=CapturaSpec(), epochs=4)
    assert limpio["laboratorio"]["omitido"] == []
    n_params = limpio["laboratorio"]["parametros"]
    assert n_params == 2 * 8 + 8 + 8 * 2 + 2

    # (a) Por encima del tope de la MUESTRA: solo pesos finales, y lo dice.
    r = _entrenar(capturar=CapturaSpec(max_params_muestra=n_params - 1), epochs=4)
    o = omitidos(r)["pesos_por_epoca"]
    assert r["laboratorio"]["pesos_por_epoca"] is None
    assert r["laboratorio"]["pesos_finales"] is not None and r["laboratorio"]["grad_norm"] is not None
    assert o["aplicado"] == 0 and o["pedido"] and o["motivo"]["es"] and o["motivo"]["en"]
    assert str(n_params) in o["motivo"]["es"]

    # (b) Por encima del tope de los FINALES: solo gradientes.
    r = _entrenar(capturar=CapturaSpec(max_params_finales=n_params - 1), epochs=4)
    assert r["laboratorio"]["pesos_finales"] is None and r["laboratorio"]["pesos_por_epoca"] is None
    assert r["laboratorio"]["grad_norm"] is not None
    assert "pesos_finales" in omitidos(r)

    # (c) El tope de NÚMEROS recorta la muestra y dice a cuántas épocas: lo aplicado es lo real.
    r = _entrenar(capturar=CapturaSpec(max_numeros=n_params * 4), epochs=20)  # finales + 3 épocas
    o = omitidos(r)["pesos_por_epoca"]
    assert o["aplicado"] == 3 and len(r["laboratorio"]["pesos_por_epoca"]) == 3
    assert r["laboratorio"]["numeros"] >= n_params * 4
    for m in r["laboratorio"]["pesos_por_epoca"]:
        assert len(m["capas"]) == 2

    # (d) Los topes viajan en el propio bloque, y la demo tiene los suyos.
    assert r["laboratorio"]["tope"]["max_numeros"] == n_params * 4
    d = captura_de_demo()
    assert (d.max_epocas_muestra, d.max_numeros, d.max_params_muestra) == (8, 20_000, 20_000)
    rd = _entrenar(capturar=d, epochs=30)
    assert len(rd["laboratorio"]["pesos_por_epoca"]) <= 8
    assert rd["laboratorio"]["numeros"] <= 20_000

    # (e) Una petición mal formada se rechaza, no se interpreta.
    with pytest.raises(ValueError):
        CapturaSpec(max_numeros=-1)


def test_el_plan_de_topes_es_puro_y_cuenta_los_numeros():
    from matrixai.training.laboratorio import CapturaSpec, planear
    p = planear(CapturaSpec(), 60_000, 4, 50, 100, 10)         # > 50.000: sin muestra
    assert p["pesos_finales"] is True and p["n_muestra"] == 0
    assert [o["que"] for o in p["omitido"]] == ["pesos_por_epoca"]
    p = planear(CapturaSpec(), 600_000, 4, 50, 100, 10)        # > 500.000: sin pesos
    assert p["pesos_finales"] is False and p["n_muestra"] == 0
    p = planear(CapturaSpec(), 50_000, 4, 50, 100, 10)         # 200.000 números = finales + 3
    assert p["n_muestra"] == 3
    p = planear(CapturaSpec(max_filas=20), 100, 2, 10, 5, 4)   # validación de 5 filas
    assert p["n_filas"] == 5 and [o["que"] for o in p["omitido"]] == ["filas"]


# ---------------------------------------------------------------------------
def test_sin_torch_declara_que_no(monkeypatch):
    from matrixai.training.laboratorio import CapturaSpec
    r = _entrenar_playground(monkeypatch, "stdlib", capturar=CapturaSpec())
    assert r["ok"] and r["backend"] != "torch"
    lab = r["laboratorio"]
    assert lab["entrenador"] == "stdlib" and lab["disponible"] is True
    # Solo los pesos finales y las filas…
    valores = r["params_best"]["parameters"]
    for i, c in enumerate(lab["pesos_finales"]["capas"], start=1):
        assert c["W"] == valores[f"{lab['capas'][0]['clave_W'].split('.')[0]}.W{i}"]["values"]
        assert c["b"] == valores[f"{lab['capas'][0]['clave_W'].split('.')[0]}.b{i}"]["values"]
    assert lab["pesos_finales"]["epoca"] == r["best_epoch"]
    assert lab["filas"] and lab["filas"]["columnas"] == ["a", "b"] and len(lab["filas"]["x"]) <= 20
    # …y lo que falta, DECLARADO y con su porqué.
    assert lab["pesos_por_epoca"] is None and lab["grad_norm"] is None
    om = {o["que"]: o for o in lab["omitido"]}
    assert {"pesos_por_epoca", "grad_norm"} <= set(om)
    for o in om.values():
        assert "sin torch" in o["motivo"]["es"] and "without torch" in o["motivo"]["en"]
    assert r["laboratorio_resumen"]["entrenador"] == "stdlib"
    assert r["laboratorio_resumen"]["gradientes"] is False
    # Y sin pedirlo, nada.
    sin = _entrenar_playground(monkeypatch, "stdlib")
    assert "laboratorio" not in sin and "laboratorio_resumen" not in sin
    assert sin["params_best"] == r["params_best"]


@needs_torch
def test_composite_no_trae_laboratorio(monkeypatch):
    from matrixai.training.laboratorio import CapturaSpec
    gen = _csv_proyecto(compuesto=True)
    r = _entrenar_playground(monkeypatch, "torch", gen=gen, capturar=CapturaSpec())
    assert r["ok"] and r["network_kind"] == "composite_network"
    lab = r["laboratorio"]
    assert lab["disponible"] is False
    assert lab["motivo"]["es"] and lab["motivo"]["en"]
    assert lab["pesos_finales"] is None and lab["pesos_por_epoca"] is None
    assert lab["grad_norm"] is None and lab["filas"] is None
    assert r["laboratorio_resumen"]["disponible"] is False
    # Sin pedirlo, no dice nada.
    sin = _entrenar_playground(monkeypatch, "torch", gen=gen)
    assert "laboratorio" not in sin and "laboratorio_resumen" not in sin


# ---------------------------------------------------------------------------
@needs_torch
def test_el_estado_publico_solo_lleva_el_resumen(monkeypatch):
    """El bloque entero se queda en el job; lo que sale por HTTP es el resumen."""
    import json
    import time
    from matrixai import limits, playground
    from matrixai.playground import MODO_TORCH_EN_CPU
    from matrixai.training.laboratorio import CapturaSpec
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", MODO_TORCH_EN_CPU)
    gen = _csv_proyecto()
    with limits.sin_topes():
        job = playground._submit_training_job(
            gen["mxai"], gen["training_text"], gen["csv_text"], epochs_override=4,
            field_ranges=gen.get("field_ranges"), seed=1, capturar=CapturaSpec())
        assert job["ok"], job
        estado = None
        for _ in range(600):
            estado = playground._get_job_status(job["job_id"])
            if estado["status"] in ("done", "error", "cancelled", "timeout"):
                break
            time.sleep(0.1)
    assert estado["status"] == "done", estado
    assert "laboratorio" not in estado
    resumen = estado["laboratorio_resumen"]
    assert resumen["disponible"] is True and resumen["entrenador"] == "torch"
    assert len(json.dumps(resumen)) < 600
    # El bloque entero sigue en el job (lo persistirá el backend) y es serializable.
    entero = playground._training_jobs[job["job_id"]]["result"]["laboratorio"]
    assert entero["pesos_finales"] is not None and len(json.dumps(entero)) > 5 * len(json.dumps(resumen))
    # El saneado público es la regla única.
    assert "laboratorio" not in playground._public_training_result({"laboratorio": {}, "ok": True})
