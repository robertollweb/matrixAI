# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""CONTRATO 133-C2 (núcleo) — el LABORATORIO de la red: lo medido del entrenamiento.

Lo que estas pruebas sostienen, cada una con su nombre (y cada una con su sabotaje en
rojo, ver el informe del corte y el de su reparación):
- `test_captura_no_cambia_el_entrenamiento`: pedir el laboratorio NO mueve ni un bit de lo
  que se entrena (ni el generador aleatorio); y sin pedirlo, el resultado no trae la clave;
- `test_sin_captura_el_resultado_es_el_de_antes_del_laboratorio`: sin pedirlo, el
  entrenamiento da lo mismo que antes del 133-C2 (huella fijada con `9552615`);
- `test_normas_de_gradiente_*`: la norma por capa y época es la de un cálculo a mano (una
  neurona; y dos capas y dos épocas), y la de los `p.grad` que usó el optimizador;
- `test_peso_pintado_es_el_medido` y `test_la_ultima_epoca_sin_programar_son_sus_pesos`:
  los pesos del bloque SON los de `best_params` con la mejor época ANTES de la última, y
  los de cada época reproducen la pérdida que se midió en ella;
- `test_plazo_vencido_marca_la_epoca_parcial`;
- `test_la_muestra_incluye_primera_mejor_ultima`;
- `test_tope_de_tamano_se_declara`, `test_el_tope_declarado_es_el_aplicado`,
  `test_la_demo_cumple_sus_20000_numeros` y `test_los_gradientes_tienen_su_tope_y_lo_dicen`:
  `max_numeros` es el TOTAL del bloque; lo que no cabe sale en `omitido` con lo pedido, lo
  aplicado y el motivo (es/en), no recortado en silencio;
- `test_sin_muestra_no_se_clona_la_ultima_epoca`;
- `test_un_fallo_*`: la captura nunca tumba el entrenamiento ni lo cambia de entrenador;
- `test_sin_torch_declara_que_no` y `test_filas_de_la_validacion_entera_en_los_dos_entrenadores`;
- `test_un_fallo_de_cada_guarda_*` (N-1), `test_aplicado_*` (N-4), `test_pedido_y_aplicado_*` (N-5),
  `test_lo_no_pedido_*` (N-6) y las de fronteras (N-7): la re-auditoría del corte;
- `test_compuesta_de_tu_red_dice_su_motivo_verdadero`, `test_red_residual_*`,
  `test_transformer_*` y `test_modelo_por_funciones_dice_que_no`: quien no trae laboratorio
  lo dice, con su motivo verdadero y el entrenador que entrenó.
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
    import torch
    from matrixai.training.laboratorio import CapturaSpec
    sin = _entrenar(**extra)
    rng_sin = torch.get_rng_state().clone()
    con = _entrenar(capturar=CapturaSpec(), **extra)
    # Ni el generador aleatorio: la captura no consume números al azar.
    assert torch.equal(rng_sin, torch.get_rng_state())
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


def _una_neurona(filas, batch, lr=0.0, weight_decay=0.0, epochs=1, seed=5, spec=None):
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.training.laboratorio import CapturaSpec
    net, ps = _setup(MXAI_UNA_CAPA)
    ps.parameters["Net.W1"]["values"] = [[0.3, -0.2]]
    ps.parameters["Net.b1"]["values"] = [-0.1]
    return train_dense_network_torch(
        net, ps, filas, "binary_cross_entropy", lr=lr, epochs=epochs, device="cpu", seed=seed,
        batch_size=batch, materialize=True, weight_decay=weight_decay,
        validation_examples=filas[:2],
        capturar=spec if spec is not None else CapturaSpec(max_filas=0))


FILAS = [([0.9, 0.1], [1.0]), ([0.2, 0.8], [0.0]), ([0.7, 0.6], [1.0]), ([0.1, 0.3], [0.0])]


def _con_permutaciones(fn):
    """Corre `fn()` grabando las permutaciones que pide el entrenador (`torch.randperm`),
    para saber QUÉ filas fueron a cada lote. No se deducen de la semilla: entre
    `manual_seed` y el primer `randperm` hay otras llamadas al generador (medido el 18-10:
    la permutación de la época 1 NO es la primera tras sembrar)."""
    import torch
    original = torch.randperm
    grabadas = []

    def grabar(*a, **k):
        out = original(*a, **k)
        grabadas.append(out.tolist())
        return out

    torch.randperm = grabar
    try:
        return fn(), grabadas
    finally:
        torch.randperm = original


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
    # época es la MEDIA de las normas de los dos lotes. El reparto, el que grabó el
    # entrenador (`_con_permutaciones`).
    r2, perms = _con_permutaciones(lambda: _una_neurona(FILAS, batch=2, seed=5))
    perm = perms[0]
    lotes = [[FILAS[i] for i in perm[0:2]], [FILAS[i] for i in perm[2:4]]]
    esp = [_gradiente_a_mano([0.3, -0.2], -0.1, lote) for lote in lotes]
    g2 = r2["laboratorio"]["grad_norm"]["por_capa"][0]
    assert g2["W"][0] == pytest.approx((esp[0][0] + esp[1][0]) / 2, abs=1e-6)
    assert g2["b"][0] == pytest.approx((esp[0][1] + esp[1][1]) / 2, abs=1e-6)
    # y es distinta de la norma del gradiente de las 4 filas juntas (la media de normas
    # no es la norma de la media): lo medido es lo que dice su definición.
    assert abs(g2["W"][0] - nw) > 1e-4 or abs(g2["b"][0] - nb) > 1e-4


@needs_torch
def test_el_gradiente_se_mide_en_hasta_n_lotes_por_epoca_y_lo_dice():
    # Punto 13: medir la norma en CADA lote costaba casi lo mismo por lote en cualquier red y
    # subía el entrenamiento un 10-11 % (redes de 50.000 y 198.146 parámetros). Se mide en hasta
    # `max_lotes_grad_por_epoca` lotes por época, repartidos (uno de cada k, el primero siempre),
    # la cifra de la época es la media de ESOS lotes y el bloque dice k y los lotes que había.
    # Aquí: 4 lotes de 1 fila con tope 2 => k = 2, se miden el 0 y el 2. lr=0: los pesos no se
    # mueven entre lotes.
    from matrixai.training.laboratorio import CapturaSpec
    r, perms = _con_permutaciones(lambda: _una_neurona(
        FILAS, batch=1, seed=5, spec=CapturaSpec(max_filas=0, max_lotes_grad_por_epoca=2)))
    perm = perms[0]
    g = r["laboratorio"]["grad_norm"]
    assert g["cada_k_lotes"] == 2 and g["lotes_por_epoca"] == 4
    medidos = [_gradiente_a_mano([0.3, -0.2], -0.1, [FILAS[perm[i]]]) for i in (0, 2)]
    todos = [_gradiente_a_mano([0.3, -0.2], -0.1, [FILAS[perm[i]]]) for i in range(4)]
    w = g["por_capa"][0]["W"][0]
    b = g["por_capa"][0]["b"][0]
    assert w == pytest.approx((medidos[0][0] + medidos[1][0]) / 2, abs=1e-6)
    assert b == pytest.approx((medidos[0][1] + medidos[1][1]) / 2, abs=1e-6)
    # y NO es la media de los cuatro (si lo fuera, «uno de cada 2» mentiría)
    assert abs(w - sum(t[0] for t in todos) / 4) > 1e-4
    assert "cada_k_lotes" in g["definicion"]["es"] and "cada_k_lotes" in g["definicion"]["en"]
    # Por debajo del tope (8 por omisión), todos los lotes.
    r1 = _una_neurona(FILAS, batch=1, seed=5)
    g1 = r1["laboratorio"]["grad_norm"]
    assert g1["cada_k_lotes"] == 1 and g1["lotes_por_epoca"] == 4
    assert g1["por_capa"][0]["W"][0] == pytest.approx(sum(t[0] for t in todos) / 4, abs=1e-6)


def test_lotes_cada_acota_los_lotes_medidos_por_epoca():
    from matrixai.training.laboratorio import CapturaSpec, lotes_cada
    spec = CapturaSpec()
    for n in (1, 7, 8, 9, 54, 75, 183, 10_000):
        k = lotes_cada(n, spec)
        medidos = len(range(0, n, k))
        assert 1 <= medidos <= 8, (n, k, medidos)
        assert k == 1 or len(range(0, n, k - 1)) > 8   # el k más pequeño que cabe
    assert lotes_cada(8, spec) == 1 and lotes_cada(9, spec) == 2
    with pytest.raises(ValueError):
        CapturaSpec(max_lotes_grad_por_epoca=0)


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
# Lo que se pinta es lo medido (B-1 de la auditoría): casos en los que la mejor época
# NO es la última, la norma calculada A MANO en dos capas y dos épocas, y la última
# época sin programar comparada por sus pesos.
# ---------------------------------------------------------------------------
MXAI_DOS_CAPAS_SIGMOIDE = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=3 activation=relu
  LAYER Dense units=1 activation=sigmoid
  OUTPUT y: Probability
END
GRAPH
  In -> Net
END
"""

W1 = [[0.5, -0.3], [-0.4, 0.6], [0.2, 0.1]]
B1 = [0.05, -0.25, 0.0]
W2 = [[0.7, -0.5, 0.3]]
B2 = [0.1]


def _gradiente_dos_capas_a_mano(filas):
    """BCE medio de 2 → 3 (relu) → 1 (sigmoide), retropropagado a mano. Devuelve las
    normas L2 de [dW1, db1, dW2, db2]."""
    n = len(filas)
    dW1 = [[0.0, 0.0] for _ in range(3)]
    db1 = [0.0] * 3
    dW2 = [0.0] * 3
    db2 = 0.0
    for x, ys in filas:
        z1 = [sum(W1[j][k] * x[k] for k in range(2)) + B1[j] for j in range(3)]
        h = [max(0.0, z) for z in z1]
        p = _sigmoide(sum(W2[0][j] * h[j] for j in range(3)) + B2[0])
        dz2 = (p - ys[0]) / n
        db2 += dz2
        for j in range(3):
            dW2[j] += dz2 * h[j]
            dz1 = dz2 * W2[0][j] * (1.0 if z1[j] > 0 else 0.0)
            db1[j] += dz1
            for k in range(2):
                dW1[j][k] += dz1 * x[k]

    def norma(v):
        return math.sqrt(sum(a * a for a in v))
    return [norma([a for fila in dW1 for a in fila]), norma(db1), norma(dW2), abs(db2)]


@needs_torch
def test_normas_de_gradiente_por_capa_y_epoca_a_mano_en_dos_capas():
    """La norma de CADA capa y CADA época es la retropropagación hecha a mano: dos capas
    (las normas de una no son las de la otra) y dos épocas (la segunda no arrastra la
    primera). lr=0: los pesos no se mueven y el gradiente a mano es exacto; el reparto
    de los lotes es el que GRABÓ el entrenador (`torch.randperm`), no uno deducido."""
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.training.laboratorio import CapturaSpec
    filas = FILAS + [([0.6, 0.9], [1.0]), ([0.4, 0.2], [0.0])]
    net, ps = _setup(MXAI_DOS_CAPAS_SIGMOIDE)
    ps.parameters["Net.W1"]["values"] = W1
    ps.parameters["Net.b1"]["values"] = B1
    ps.parameters["Net.W2"]["values"] = W2
    ps.parameters["Net.b2"]["values"] = B2
    r, perms = _con_permutaciones(lambda: train_dense_network_torch(
        net, ps, filas, "binary_cross_entropy", lr=0.0, epochs=2, device="cpu", seed=7,
        batch_size=2, materialize=True, weight_decay=0.5, validation_examples=filas[:2],
        capturar=CapturaSpec(max_filas=0)))
    g = r["laboratorio"]["grad_norm"]
    assert g["epocas"] == [1, 2] and g["cada_k_lotes"] == 1 and g["lotes_por_epoca"] == 3
    assert [c["capa"] for c in g["por_capa"]] == [1, 2]
    esperado = []                                   # [época][tensor]
    for perm in perms[:2]:
        lotes = [[filas[i] for i in perm[j:j + 2]] for j in range(0, 6, 2)]
        normas = [_gradiente_dos_capas_a_mano(lote) for lote in lotes]
        esperado.append([sum(nl[t] for nl in normas) / len(normas) for t in range(4)])
    for e in range(2):
        for i, capa in enumerate(g["por_capa"]):
            assert capa["W"][e] == pytest.approx(esperado[e][2 * i], abs=1e-6), (e, i, "W")
            assert capa["b"][e] == pytest.approx(esperado[e][2 * i + 1], abs=1e-6), (e, i, "b")
    # El caso tiene dientes: las capas y las épocas dan cifras distintas.
    assert abs(esperado[0][0] - esperado[0][2]) > 1e-3 and abs(esperado[0][1] - esperado[0][3]) > 1e-3
    assert abs(esperado[0][0] - esperado[1][0]) > 1e-4


@needs_torch
def test_normas_de_gradiente_son_las_que_uso_el_optimizador():
    """Con lr > 0 (los pesos se mueven), dos capas, dos épocas y uno de cada k lotes: la
    norma del bloque es la media de las normas de `p.grad` que vio `optimizer.step` en
    los lotes medidos. Se GRABA lo que el entrenador usó (envolviendo `step`)."""
    import torch
    from matrixai.training.laboratorio import CapturaSpec
    grabado = []
    original = torch.optim.SGD.step

    def step(self, *a, **k):
        grabado.append([float(p.grad.norm()) for g in self.param_groups for p in g["params"]])
        return original(self, *a, **k)
    torch.optim.SGD.step = step
    try:
        r = _entrenar(capturar=CapturaSpec(max_lotes_grad_por_epoca=3, max_filas=0),
                      n=200, epochs=2, lr=0.3, batch_size=8, weight_decay=0.2)
    finally:
        torch.optim.SGD.step = original
    g = r["laboratorio"]["grad_norm"]
    lotes = g["lotes_por_epoca"]
    k = g["cada_k_lotes"]
    assert lotes == 20 and k == 7 and len(grabado) == 2 * lotes
    for e in range(2):
        medidos = [grabado[e * lotes + j] for j in range(0, lotes, k)]
        for i, capa in enumerate(g["por_capa"]):
            for t, clave in ((2 * i, "W"), (2 * i + 1, "b")):
                esp = sum(m[t] for m in medidos) / len(medidos)
                assert capa[clave][e] == pytest.approx(esp, rel=1e-5), (e, i, clave)


@needs_torch
def test_peso_pintado_es_el_medido():
    """Los pesos finales del bloque SON `best_params` (los de «Probar una fila»), en un caso
    con la mejor época ANTES de la última; y los de cada época de la muestra reproducen la
    pérdida de validación que se midió en ESA época."""
    from matrixai.training.laboratorio import CapturaSpec
    n, epochs = 80, 30
    r = _entrenar(capturar=CapturaSpec(), n=n, epochs=epochs, lr=1.2)
    lab = r["laboratorio"]
    ultima = r["epochs"][-1]["epoch"]
    assert r["best_epoch"] < ultima, "el caso tenía que tener la mejor ANTES de la última"
    valores = _pesos(r)
    # 1. Finales = best_params, valor a valor (y no los de la última época).
    assert lab["pesos_finales"]["epoca"] == r["best_epoch"]
    for i, c in enumerate(lab["pesos_finales"]["capas"], start=1):
        assert c["W"] == valores[f"Net.W{i}"]
        assert c["b"] == valores[f"Net.b{i}"]
    # 2. Cada época de la muestra reproduce su val_loss: son los pesos de esa época.
    val = _ejemplos(n)[int(n * 0.8):]
    por_epoca = {e["epoch"]: e["val_loss"] for e in r["epochs"]}
    eps = [m["epoca"] for m in lab["pesos_por_epoca"]]
    assert ultima in eps and r["best_epoch"] in eps
    for m in lab["pesos_por_epoca"]:
        assert _perdida_con_pesos(m["capas"], val) == pytest.approx(por_epoca[m["epoca"]], abs=1e-6)
    # 3. La última NO es la mejor: el caso distingue «finales = mejor» de «finales = última».
    ult = [m for m in lab["pesos_por_epoca"] if m["epoca"] == ultima][0]
    assert ult["capas"] != lab["pesos_finales"]["capas"]
    mejor = [m for m in lab["pesos_por_epoca"] if "mejor" in m["motivo"]][0]
    assert mejor["capas"] == lab["pesos_finales"]["capas"]


@needs_torch
def test_la_ultima_epoca_sin_programar_son_sus_pesos():
    """Con parada temprana la última época no estaba programada en la muestra: se guarda al
    acabar, y sus pesos son los SUYOS (reproducen su val_loss), no los de la mejor."""
    from matrixai.training.laboratorio import CapturaSpec, epocas_programadas
    n = 80
    r = _entrenar(capturar=CapturaSpec(), n=n, epochs=50, lr=1.2,
                  early_stop=(3, "validation_loss"))
    ultima = r["epochs"][-1]["epoch"]
    assert ultima < 50 and ultima not in epocas_programadas(50, 12), "el caso tenía que parar antes"
    assert r["best_epoch"] < ultima
    lab = r["laboratorio"]
    m = [x for x in lab["pesos_por_epoca"] if x["epoca"] == ultima]
    assert m and "ultima" in m[0]["motivo"]
    val = _ejemplos(n)[int(n * 0.8):]
    vl = {e["epoch"]: e["val_loss"] for e in r["epochs"]}
    assert _perdida_con_pesos(m[0]["capas"], val) == pytest.approx(vl[ultima], abs=1e-6)
    assert abs(vl[ultima] - vl[r["best_epoch"]]) > 1e-4


@needs_torch
def test_plazo_vencido_marca_la_epoca_parcial():
    """Un plazo ya vencido es determinista: la época 1 para tras su primer lote, a medias.
    El gradiente, la muestra y los pesos finales lo dicen (`parcial`)."""
    import time
    from matrixai.training.laboratorio import CapturaSpec
    r = _entrenar(capturar=CapturaSpec(max_epocas_muestra=50, max_numeros=10**7), n=400,
                  epochs=6, batch_size=8, plazo=time.monotonic() - 1.0)
    assert r["parado_por_plazo"] is True
    assert [(e["epoch"], e.get("parcial")) for e in r["epochs"]] == [(1, True)]
    lab = r["laboratorio"]
    assert lab["grad_norm"]["epocas"] == [1] and lab["grad_norm"]["parciales"] == [1]
    assert [(m["epoca"], m["parcial"]) for m in lab["pesos_por_epoca"]] == [(1, True)]
    assert lab["pesos_finales"]["epoca"] == 1 and lab["pesos_finales"]["parcial"] is True
    # Y sin plazo, nada parcial.
    r2 = _entrenar(capturar=CapturaSpec(), epochs=2)
    assert r2["laboratorio"]["grad_norm"]["parciales"] == []
    assert r2["laboratorio"]["pesos_finales"]["parcial"] is False
    assert all(m["parcial"] is False for m in r2["laboratorio"]["pesos_por_epoca"])


# ---------------------------------------------------------------------------
# Topes: el que se declara es el que se aplica (I-4, M-3, M-4, M-5)
# ---------------------------------------------------------------------------
def _cuenta_partes(lab):
    """Los números de cada parte, CONTADOS en el bloque (no leídos de lo que declara)."""
    def capas(cs):
        return sum(len(c["W"]) * len(c["W"][0]) + len(c["b"]) for c in cs)
    return {
        "pesos_finales": capas(lab["pesos_finales"]["capas"]) if lab["pesos_finales"] else 0,
        "pesos_por_epoca": sum(capas(m["capas"]) for m in lab["pesos_por_epoca"] or []),
        "grad_norm": (sum(len(c["W"]) + len(c["b"]) for c in lab["grad_norm"]["por_capa"])
                      if lab["grad_norm"] else 0),
        "filas": (sum(len(x) for x in lab["filas"]["x"]) + sum(len(y) for y in lab["filas"]["y"])
                  if lab["filas"] else 0),
    }


def _omitidos(lab):
    return {o["que"]: o for o in lab["omitido"]}


@needs_torch
def test_tope_de_tamano_se_declara():
    from matrixai.training.laboratorio import CapturaSpec, TOPES_POR_PARTE

    # Sin recortar: nada omitido, y lo declarado cuadra con lo contado.
    limpio = _entrenar(capturar=CapturaSpec(), epochs=4)
    lab = limpio["laboratorio"]
    assert lab["omitido"] == []
    n_params = lab["parametros"]
    assert n_params == 2 * 8 + 8 + 8 * 2 + 2
    assert lab["numeros_por_parte"] == _cuenta_partes(lab)
    assert lab["numeros"] == sum(lab["numeros_por_parte"].values())
    assert lab["tope_por_parte"] == {k: list(v) for k, v in TOPES_POR_PARTE.items()}
    assert lab["definicion_numeros"]["es"] and lab["definicion_numeros"]["en"]

    # (a) Por encima del tope de la MUESTRA: solo pesos finales, y lo dice.
    r = _entrenar(capturar=CapturaSpec(max_params_muestra=n_params - 1), epochs=4)
    o = _omitidos(r["laboratorio"])["pesos_por_epoca"]
    assert r["laboratorio"]["pesos_por_epoca"] is None
    assert r["laboratorio"]["pesos_finales"] is not None and r["laboratorio"]["grad_norm"] is not None
    assert o["aplicado"] == 0 and (o["pedido"], o["unidad"]) == (4, "epocas") and o["motivo"]["es"] and o["motivo"]["en"]
    assert str(n_params) in o["motivo"]["es"]

    # (b) Por encima del tope de los FINALES: solo gradientes, y los dos huecos dichos (M-5).
    r = _entrenar(capturar=CapturaSpec(max_params_finales=n_params - 1), epochs=4)
    assert r["laboratorio"]["pesos_finales"] is None and r["laboratorio"]["pesos_por_epoca"] is None
    assert r["laboratorio"]["grad_norm"] is not None
    assert {"pesos_finales", "pesos_por_epoca"} <= set(_omitidos(r["laboratorio"]))

    # (c) El tope de NÚMEROS es el TOTAL: con 20 épocas, gradientes 20×4 = 80, finales 42 y
    # filas 20×4 = 80; un tope que deje sitio justo para 3 épocas de muestra.
    tope = 80 + n_params + 80 + 3 * n_params
    r = _entrenar(capturar=CapturaSpec(max_numeros=tope), epochs=20)
    lab = r["laboratorio"]
    o = _omitidos(lab)["pesos_por_epoca"]
    assert o["aplicado"] == 3 and len(lab["pesos_por_epoca"]) == 3
    assert lab["numeros"] == tope and lab["numeros_por_parte"] == _cuenta_partes(lab)
    assert lab["tope"]["max_numeros"] == tope

    # (d) Una petición mal formada se rechaza, no se interpreta.
    with pytest.raises(ValueError):
        CapturaSpec(max_numeros=-1)
    with pytest.raises(ValueError):
        CapturaSpec(max_epocas_grad=-1)


@needs_torch
def test_el_tope_declarado_es_el_aplicado():
    """`max_numeros` es el TOTAL del bloque (pesos finales + muestra + gradientes + filas)
    y nunca se pasa: lo que no cabe se recorta y se DECLARA. Antes, los pesos finales y
    las filas quedaban fuera de la cuenta (I-4, M-4)."""
    from matrixai.training.laboratorio import CapturaSpec
    # Los finales (42) no caben: gradientes 20 épocas × 4 = 80 de un tope de 100.
    r = _entrenar(capturar=CapturaSpec(max_numeros=100), epochs=20)
    lab = r["laboratorio"]
    om = _omitidos(lab)
    assert lab["pesos_finales"] is None and lab["pesos_por_epoca"] is None
    assert "100" in om["pesos_finales"]["motivo"]["es"] and "100" in om["pesos_finales"]["motivo"]["en"]
    assert "pesos_por_epoca" in om
    # Con lo que sobra (20), 5 filas de 4 números, y se dice.
    assert len(lab["filas"]["x"]) == 5 and om["filas"]["aplicado"] == 5
    assert lab["numeros"] == 100 and lab["numeros_por_parte"] == _cuenta_partes(lab)
    assert lab["numeros"] <= lab["tope"]["max_numeros"]


@needs_torch
def test_la_demo_cumple_sus_20000_numeros():
    """La demo (`captura_de_demo`): ≤ 8 épocas, ≤ 20.000 números EN TOTAL. Una red de
    18.542 parámetros y 100 columnas: con la cuenta vieja (solo la muestra) el bloque
    llevaba 20.782 números; ahora caben los finales, se recortan las filas y no hay
    muestra, y las dos cosas se dicen."""
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.training.laboratorio import captura_de_demo
    d = captura_de_demo()
    assert (d.max_epocas_muestra, d.max_numeros, d.max_params_muestra, d.max_params_finales) == (
        8, 20_000, 20_000, 20_000)
    campos = "".join(f"  f{i}: Scalar\n" for i in range(100))
    texto = (f"PROJECT P\nVECTOR In[100]\n{campos}END\nNETWORK Net\n  INPUT In\n"
             "  LAYER Dense units=180 activation=relu\n  LAYER Dense units=2 activation=softmax\n"
             "  OUTPUT y: ProbabilityMap[A, B]\nEND\nGRAPH\n  In -> Net\nEND\n")
    net, ps = _setup(texto)
    rng = random.Random(4)
    ex = []
    for _ in range(150):
        x = [rng.random() for _ in range(100)]
        ex.append((x, [1.0, 0.0] if sum(x[:5]) > 2.5 else [0.0, 1.0]))
    r = train_dense_network_torch(net, ps, ex, "cross_entropy", lr=0.05, epochs=50,
                                  device="cpu", seed=3, batch_size=32, materialize=True,
                                  capturar=d)
    lab = r["laboratorio"]
    assert lab["parametros"] == 100 * 180 + 180 + 180 * 2 + 2
    assert lab["numeros"] <= 20_000
    assert lab["numeros_por_parte"] == _cuenta_partes(lab)
    assert lab["pesos_finales"] is not None and len(lab["grad_norm"]["epocas"]) == 50
    om = _omitidos(lab)
    assert om["filas"]["aplicado"] == len(lab["filas"]["x"]) < 20
    assert om["pesos_por_epoca"]["aplicado"] == 0 and lab["pesos_por_epoca"] is None
    # Las 50 épocas caben en el tope de 50: ningún recorte de gradientes que declarar (N-7).
    assert "grad_norm" not in om and lab["grad_norm"]["epocas_medidas"] == 50
    # Climate en la demo (lo del vídeo): muestra entera y todo dentro del tope.
    r = _entrenar(capturar=d, epochs=30)
    assert len(r["laboratorio"]["pesos_por_epoca"]) == 8
    assert r["laboratorio"]["numeros"] <= 20_000 and r["laboratorio"]["omitido"] == []


@needs_torch
def test_los_gradientes_tienen_su_tope_y_lo_dicen():
    """Más épocas que `max_epocas_grad`: una muestra repartida que incluye SIEMPRE la
    primera, la mejor y la última, con los valores de esas épocas, y declarada (M-3)."""
    from matrixai.training.laboratorio import CapturaSpec
    entero = _entrenar(capturar=CapturaSpec(max_filas=0), n=80, epochs=30, lr=1.2)
    r = _entrenar(capturar=CapturaSpec(max_filas=0, max_epocas_grad=5), n=80, epochs=30, lr=1.2)
    assert r["best_epoch"] not in (1, 30)
    g = r["laboratorio"]["grad_norm"]
    assert len(g["epocas"]) == 5 and {1, r["best_epoch"], 30} <= set(g["epocas"])
    assert g["epocas_medidas"] == 30 and entero["laboratorio"]["grad_norm"]["epocas_medidas"] == 30
    o = _omitidos(r["laboratorio"])["grad_norm"]
    assert (o["pedido"], o["unidad"]) == (30, "epocas") and o["aplicado"] == 5 and o["motivo"]["es"] and o["motivo"]["en"]
    ge = entero["laboratorio"]["grad_norm"]
    for capa, capa_e in zip(g["por_capa"], ge["por_capa"]):
        for i, e in enumerate(g["epocas"]):
            assert capa["W"][i] == capa_e["W"][e - 1] and capa["b"][i] == capa_e["b"][e - 1]
    assert "grad_norm" not in _omitidos(entero["laboratorio"])


def test_el_plan_de_topes_es_puro_y_cuenta_los_numeros():
    from matrixai.training.laboratorio import CapturaSpec, planear
    p = planear(CapturaSpec(), 60_000, 4, 50, 100, 10)         # > 50.000: sin muestra
    assert p["pesos_finales"] is True and p["n_muestra"] == 0
    assert [o["que"] for o in p["omitido"]] == ["pesos_por_epoca"]
    p = planear(CapturaSpec(), 250_000, 4, 50, 100, 10)        # > 200.000: sin pesos
    assert p["pesos_finales"] is False and p["n_muestra"] == 0
    assert [o["que"] for o in p["omitido"]] == ["pesos_finales", "pesos_por_epoca"]
    # 200.000 en total: gradientes 50×8 = 400, finales 50.000, filas 20×10 = 200 → 2 épocas.
    p = planear(CapturaSpec(), 50_000, 4, 50, 100, 10)
    assert p["n_muestra"] == 2 and p["n_epocas_grad"] == 200
    p = planear(CapturaSpec(max_filas=20), 100, 2, 10, 5, 4)   # validación de 5 filas
    assert p["n_filas"] == 5 and [o["que"] for o in p["omitido"]] == ["filas"]
    # Sin torch no se reservan números para gradientes que no se miden.
    p = planear(CapturaSpec(max_numeros=1000), 100, 2, 10, 50, 4, con_gradientes=False)
    assert p["gradientes"] is False and p["n_epocas_grad"] == 0 and p["n_muestra"] == 8


@needs_torch
def test_pesos_finales_no_pedidos_se_declaran():
    """M-5: `pesos_finales=False` no deja huecos callados: ni finales ni por época, y los
    dos se dicen."""
    from matrixai.training.laboratorio import CapturaSpec
    r = _entrenar(capturar=CapturaSpec(pesos_finales=False), epochs=4)
    lab = r["laboratorio"]
    assert lab["pesos_finales"] is None and lab["pesos_por_epoca"] is None
    om = _omitidos(lab)
    assert (om["pesos_finales"]["pedido"], om["pesos_finales"]["aplicado"]) == (0, 0)
    assert om["pesos_por_epoca"]["aplicado"] == 0 and om["pesos_por_epoca"]["motivo"]["en"]


@needs_torch
def test_sin_muestra_no_se_clona_la_ultima_epoca(monkeypatch):
    """I-3: sin muestra que guardar (`n_muestra == 0`), la captura no clona la última
    época: con una red grande, esa copia era la red entera otra vez (+72,6 MB medidos con
    18,9 M de parámetros). Se cuentan las copias de tensores que hace el entrenador."""
    import torch
    from matrixai.training.laboratorio import CapturaSpec
    original = torch.Tensor.clone
    cuenta = {"n": 0}

    def contar(self, *a, **k):
        cuenta["n"] += 1
        return original(self, *a, **k)
    monkeypatch.setattr(torch.Tensor, "clone", contar)

    def copias(spec):
        cuenta["n"] = 0
        _entrenar(capturar=spec, n=80, epochs=30, lr=1.2, early_stop=(3, "validation_loss"))
        return cuenta["n"]
    sin = copias(None)
    assert copias(CapturaSpec(max_epocas_muestra=0)) == sin
    assert copias(CapturaSpec(max_params_muestra=1)) == sin
    assert copias(CapturaSpec(pesos_finales=False)) == sin
    # Control: con muestra sí se clona (el contador ve las copias de la captura).
    assert copias(CapturaSpec()) > sin


# ---------------------------------------------------------------------------
# La captura nunca tumba el entrenamiento, ni lo cambia de entrenador (I-5, S9, M-8)
# ---------------------------------------------------------------------------
@needs_torch
def test_un_fallo_en_la_medida_por_lote_no_cambia_de_entrenador(monkeypatch):
    """I-5: si la medida del gradiente falla en un lote, se apaga ESA parte, se declara
    (texto fijo y el tipo de la excepción aparte) y el entrenamiento sigue en torch con
    los mismos pesos que sin captura. Antes caía al entrenador sin torch."""
    import torch
    from matrixai.training.laboratorio import CapturaSpec, FALLO_GRADIENTE
    sin = _entrenar_playground(monkeypatch, "torch")

    def falla(*a, **k):
        raise TypeError("detalle interno que no se pinta")
    monkeypatch.setattr(torch, "_foreach_norm", falla)
    con = _entrenar_playground(monkeypatch, "torch", capturar=CapturaSpec())
    assert con["ok"] and con["backend"] == "torch"
    assert con["params_best"] == sin["params_best"] and con["epochs"] == sin["epochs"]
    lab = con["laboratorio"]
    assert lab["disponible"] is True and lab["entrenador"] == "torch"
    assert lab["grad_norm"] is None and lab["pesos_finales"] is not None
    o = _omitidos(lab)["grad_norm"]
    assert o["error"] == "TypeError" and o["epoca"] == 1
    assert (o["motivo"]["es"], o["motivo"]["en"]) == FALLO_GRADIENTE
    assert "detalle" not in str(lab)


@needs_torch
def test_un_fallo_al_armar_el_bloque_no_tumba_el_entrenamiento(monkeypatch):
    """S9: un fallo al armar el bloque deja `disponible: False` con un texto FIJO (M-8: el
    de la excepción no se pinta) y el entrenamiento intacto; con torch y sin torch."""
    import matrixai.training.dense_torch_trainer as m
    import matrixai.training.laboratorio as lab_mod
    from matrixai.training.laboratorio import CapturaSpec, FALLO_BLOQUE

    def falla(*a, **k):
        raise RuntimeError("detalle interno que no se pinta")
    sin = _entrenar()
    monkeypatch.setattr(m, "_bloque_laboratorio", falla)
    con = _entrenar(capturar=CapturaSpec())
    assert _pesos(con) == _pesos(sin) and con["epochs"] == sin["epochs"]
    lab = con["laboratorio"]
    assert lab["disponible"] is False and lab["entrenador"] == "torch"
    assert lab["error"] == "RuntimeError"
    assert (lab["motivo"]["es"], lab["motivo"]["en"]) == FALLO_BLOQUE

    sin = _entrenar_playground(monkeypatch, "stdlib", epochs=2)
    monkeypatch.setattr(lab_mod, "bloque_sin_torch", falla)
    con = _entrenar_playground(monkeypatch, "stdlib", epochs=2, capturar=CapturaSpec())
    assert con["ok"] and con["params_best"] == sin["params_best"]
    assert con["laboratorio"]["disponible"] is False and con["laboratorio"]["error"] == "RuntimeError"
    assert con["laboratorio"]["entrenador"] == "stdlib"
    assert (con["laboratorio"]["motivo"]["es"], con["laboratorio"]["motivo"]["en"]) == FALLO_BLOQUE


# ---------------------------------------------------------------------------
# Sin captura, el resultado es el de antes: una huella FIJADA (S12)
# ---------------------------------------------------------------------------
#: Calculadas con el código de `9552615` (release 1.17.0, ANTES del 133-C2), en un
#: worktree desacoplado, torch 2.11.0+cpu, con `_huella_*` de aquí. Ver la prueba.
HUELLAS_9552615 = {
    "sgd": "77b39cf1a9eaf65e59ad635fcc46374d0e37d032f045ea9879ea4c94067e1898",
    "adamw_wd_coseno": "ac01f4b2f2785b068980ba4ef327ce1de617b6b9d7b78e0ae4929cc1e5861899",
    "parada_temprana": "f276c260fb8047eeacd7ade15a25088418b85e801e2ea5584a4abd964281f989",
    "superficie_torch": "7ee84bbd6337fbd38f9114e665a6fca1c61b9e581dffabb73f0568bfcf202cd1",
    "superficie_stdlib": "c1b1ac8e17e1e29395765d824eb3d62dbee1719b3e6f8305dc18e05970649c29",
}
TORCH_DE_LAS_HUELLAS = "2.11.0+cpu"


def _huella(pesos, epochs, rng=True, **extra):
    """sha256 de los pesos (floats exactos), la traza de épocas y, con `rng`, el estado del
    generador de torch al terminar (la captura no puede consumir números al azar)."""
    import hashlib
    import json
    import torch
    datos = {"pesos": pesos, "epochs": epochs, **extra}
    if rng:
        datos["rng_torch"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(datos, sort_keys=True).encode()).hexdigest()


@needs_torch
def test_sin_captura_el_resultado_es_el_de_antes_del_laboratorio(monkeypatch):
    """«Con `capturar=None` el entrenamiento es byte a byte el de antes» lo sostiene ESTA
    prueba: pesos, curvas, mejor época y estado del generador de torch al terminar de cinco
    entrenamientos pequeños y deterministas en CPU, contra una huella calculada con el
    código de antes del 133-C2. (El de sin torch, sin el estado de torch: no es suyo.)"""
    import torch
    directos = {
        "sgd": {},
        "adamw_wd_coseno": {"optimizer": "adamw", "weight_decay": 0.01, "schedule": "cosine"},
        "parada_temprana": {"n": 60, "epochs": 50, "lr": 2.0, "early_stop": (2, "validation_loss")},
    }
    medidas = {}
    for nombre, kw in directos.items():
        r = _entrenar(**kw)
        medidas[nombre] = _huella(_pesos(r), r["epochs"], best_epoch=r["best_epoch"],
                                  best_val_loss=r["best_val_loss"], train_loss=r["train_loss"])
    for backend in ("torch", "stdlib"):
        r = _entrenar_playground(monkeypatch, backend)
        assert r["ok"] and r["backend"] == backend
        medidas[f"superficie_{backend}"] = _huella(
            {k: v["values"] for k, v in r["params_best"]["parameters"].items()}, r["epochs"],
            rng=backend == "torch")
    distintas = {k for k in HUELLAS_9552615 if medidas[k] != HUELLAS_9552615[k]}
    assert not distintas, (
        f"Sin captura, estos entrenamientos ya no dan lo que daban antes del laboratorio: "
        f"{sorted(distintas)} (torch {torch.__version__}; las huellas son de "
        f"{TORCH_DE_LAS_HUELLAS}). Si el entrenador cambió A PROPÓSITO, o cambió torch: "
        f"recalcula las huellas con el código de ANTES de tu cambio en un worktree "
        f"desacoplado (con PYTHONPATH a él), comprueba que la diferencia es la que buscabas "
        f"y, solo entonces, fija las nuevas aquí diciendo de qué commit salen. Si no cambiaste "
        f"el entrenador, la captura (o algo a su lado) está tocando el entrenamiento.")


# ---------------------------------------------------------------------------
# Quien no trae laboratorio lo dice, con su motivo VERDADERO (I-1, M-2, M-6, S7)
# ---------------------------------------------------------------------------
_RESIDUAL_MXAI = """PROJECT ResProject

VECTOR Patient[4]
  x1: Scalar
  x2: Scalar
  x3: Scalar
  x4: Scalar
END

NETWORK ResNet
  INPUT Patient
  LAYER Dense units=8 activation=relu
  BLOCK res1
    LAYER Dense units=8 activation=relu
    RESIDUAL FROM PREVIOUS
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT predicted_class: ProbabilityMap[BAJO, ALTO]
END

GRAPH
  Patient -> ResNet
END
"""

_RESIDUAL_MXTRAIN = """MODEL ResProject.mxai

DATASET D
  SOURCE csv("d.train.csv")
  INPUT Patient FROM COLUMNS [x1, x2, x3, x4]
  TARGET predicted_class: Label[BAJO, ALTO]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END

LOSS L
  TYPE cross_entropy
  PREDICTION ResNet
  TARGET predicted_class
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.05
  UPDATE ResNet.*
END

RUN
  EPOCHS 2
END
"""


def _residual_csv():
    filas = ["x1,x2,x3,x4,predicted_class"]
    for i in range(30):
        s = (i % 10) / 10.0
        filas.append(f"{s},{(i % 7) / 7.0:.3f},{(i % 5) / 5.0:.3f},{(i % 3) / 3.0:.3f},"
                     f"{'BAJO' if s < 0.5 else 'ALTO'}")
    return "\n".join(filas) + "\n"


def _sin_tabm(motivo):
    return "TabM" not in motivo["es"] and "TabM" not in motivo["en"]


@needs_torch
@pytest.mark.parametrize("backend", ["torch", "stdlib"])
def test_compuesta_de_tu_red_dice_su_motivo_verdadero(monkeypatch, backend):
    """I-1: tu red compuesta (aquí, con EMBEDDING y CONCAT, como la construye el núcleo
    desde un CSV con una categórica de muchos valores) no tiene laboratorio, y el motivo
    dice lo que TIENE, sin hablar de TabM (que no pasa por aquí). M-2: el entrenador
    declarado es el que entrenó (`backend`), también si fue el de sin torch."""
    from matrixai.training.laboratorio import CapturaSpec
    gen = _csv_proyecto(compuesto=True)
    r = _entrenar_playground(monkeypatch, backend, gen=gen, capturar=CapturaSpec(), epochs=2)
    assert r["ok"] and r["network_kind"] == "composite_network" and r["backend"] == backend
    lab = r["laboratorio"]
    assert lab["disponible"] is False and lab["entrenador"] == backend
    assert "embedding" in lab["motivo"]["es"] and "embedding" in lab["motivo"]["en"]
    assert "CONCAT" in lab["motivo"]["es"] and "CONCAT" in lab["motivo"]["en"]
    assert "densas" in lab["motivo"]["es"] and "dense" in lab["motivo"]["en"]
    assert _sin_tabm(lab["motivo"])
    assert lab["pesos_finales"] is None and lab["pesos_por_epoca"] is None
    assert lab["grad_norm"] is None and lab["filas"] is None
    assert r["laboratorio_resumen"]["disponible"] is False
    assert r["laboratorio_resumen"]["entrenador"] == backend
    # Sin pedirlo, no dice nada.
    sin = _entrenar_playground(monkeypatch, backend, gen=gen, epochs=2)
    assert "laboratorio" not in sin and "laboratorio_resumen" not in sin


@needs_torch
def test_red_residual_dice_su_motivo_verdadero():
    from matrixai.playground import _run_playground_training
    from matrixai.training.laboratorio import CapturaSpec
    r = _run_playground_training(_RESIDUAL_MXAI, _RESIDUAL_MXTRAIN, _residual_csv(),
                                 epochs_override=2, capturar=CapturaSpec())
    assert r["ok"], r.get("error")
    lab = r["laboratorio"]
    assert lab["disponible"] is False and lab["entrenador"] == r["backend"]
    assert "residual" in lab["motivo"]["es"] and "residual" in lab["motivo"]["en"]
    assert "embedding" not in lab["motivo"]["es"] and _sin_tabm(lab["motivo"])


def _texto_transformer():
    import csv
    import io
    from matrixai.training.synthetic_text import generate_text_examples
    from matrixai.training.transformer_generator import TransformerNetworkGenerator
    gen = TransformerNetworkGenerator().generate(
        "resenas: Text[16]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
    filas, _ = generate_text_examples("coherent", "clasificar", 20, ["neg", "pos"], seed=1,
                                      seq_length=16, use_llm=False)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["resenas", "predicted_class"])
    for texto, etiqueta in filas:
        w.writerow([texto, etiqueta])
    return gen.mxai_text, gen.training_text, buf.getvalue()


@needs_torch
def test_transformer_dice_que_no_por_las_dos_superficies():
    """S7: un transformer no tiene laboratorio y lo DICE, por la superficie síncrona y por
    el job; y sin pedirlo, calla."""
    import time
    from matrixai import limits, playground
    from matrixai.training.laboratorio import CapturaSpec
    mxai, train, csv_text = _texto_transformer()
    assert playground._network_is_transformer(mxai)
    with limits.sin_topes():
        r = playground._run_playground_training(mxai, train, csv_text, epochs_override=2,
                                                capturar=CapturaSpec())
        sin = playground._run_playground_training(mxai, train, csv_text, epochs_override=2)
        job = playground._submit_training_job(mxai, train, csv_text, epochs_override=2,
                                              capturar=CapturaSpec())
        assert job["ok"], job
        estado = None
        for _ in range(1200):
            estado = playground._get_job_status(job["job_id"])
            if estado["status"] in ("done", "error", "cancelled", "timeout"):
                break
            time.sleep(0.1)
    assert r["ok"], r.get("error")
    for lab, backend in ((r["laboratorio"], r["backend"]),
                         (playground._training_jobs[job["job_id"]]["result"]["laboratorio"],
                          playground._training_jobs[job["job_id"]]["result"]["backend"])):
        assert lab["disponible"] is False and lab["entrenador"] == backend == "torch"
        assert "transformer" in lab["motivo"]["es"] and "transformer" in lab["motivo"]["en"]
    assert estado["status"] == "done", estado
    assert "laboratorio" not in estado and estado["laboratorio_resumen"]["disponible"] is False
    assert "laboratorio" not in sin and "laboratorio_resumen" not in sin


def test_modelo_por_funciones_dice_que_no():
    """M-6: un modelo por funciones (FUNCTION) o que llama a capas (layer_call) tampoco
    tiene laboratorio, y lo dice en vez de callar, por la superficie síncrona y por el
    job; el entrenador, el del resultado. (El de capas entrena por diferencias finitas:
    se recorta su CSV a 8 filas para que la prueba no cueste.)"""
    import time
    from pathlib import Path
    from matrixai import limits, playground
    from matrixai.playground import _generate_training_from_mxai, _run_playground_training
    from matrixai.training.laboratorio import CapturaSpec
    ej = Path(__file__).parent.parent / "examples"
    casos = {}
    mx = (ej / "fall-risk.typed.mxai").read_text(encoding="utf-8")
    casos["FUNCTION"] = (mx, _generate_training_from_mxai(mx)["training_text"],
                         (ej / "fall-risk.train.csv").read_text(encoding="utf-8"))
    csv_capas = (ej / "transformer-classifier-vector.train.csv").read_text(encoding="utf-8")
    casos["LAYER"] = (
        (ej / "transformer-classifier-vector.mxai").read_text(encoding="utf-8"),
        (ej / "transformer-classifier-vector.mxtrain").read_text(encoding="utf-8")
        .replace("examples/", ""),
        "\n".join(csv_capas.splitlines()[:9]) + "\n")
    for clave, (mx, tt, cs) in casos.items():
        with limits.sin_topes():
            r = _run_playground_training(mx, tt, cs, epochs_override=1, capturar=CapturaSpec())
            job = playground._submit_training_job(mx, tt, cs, epochs_override=1,
                                                  capturar=CapturaSpec())
            assert job["ok"], (clave, job)
            for _ in range(1200):
                estado = playground._get_job_status(job["job_id"])
                if estado["status"] in ("done", "error", "cancelled", "timeout"):
                    break
                time.sleep(0.05)
        assert r["ok"], (clave, r.get("error"))
        assert estado["status"] == "done", (clave, estado)
        rj = playground._training_jobs[job["job_id"]]["result"]
        for res in (r, rj):
            lab = res["laboratorio"]
            assert lab["disponible"] is False and lab["entrenador"] == res["backend"], clave
            assert clave in lab["motivo"]["es"] and clave in lab["motivo"]["en"]
        assert estado["laboratorio_resumen"]["disponible"] is False and "laboratorio" not in estado
    with limits.sin_topes():
        sin = _run_playground_training(*casos["FUNCTION"], epochs_override=1)
    assert "laboratorio" not in sin


# ---------------------------------------------------------------------------
# Lo demás que se declara (I-2, M-7, M-9, M-11)
# ---------------------------------------------------------------------------
@needs_torch
def test_filas_de_la_validacion_entera_en_los_dos_entrenadores(monkeypatch):
    """I-2: las filas salen de la validación ENTERA y preparada, igual con torch que sin
    él, y su `de` es verdadero. En regresión con el objetivo sin recortar, torch elige la
    época con las filas alcanzables: antes enseñaba solo esas (21 de 40) y callaba."""
    from matrixai.playground_api import generate_project_from_dataset
    from matrixai.training.laboratorio import CapturaSpec
    rng = random.Random(1)
    lineas = ["a,y"] + [f"{i / 20.0:.4f},{3 * (i / 20.0) + rng.random():.4f}" for i in range(200)]
    gen = generate_project_from_dataset("\n".join(lineas) + "\n", "y", locale="es")
    filas = {}
    for backend, ep in (("torch", 3), ("stdlib", 1)):
        r = _entrenar_playground(monkeypatch, backend, gen=gen, epochs=ep,
                                 target_range=gen.get("target_range"),
                                 recortar_objetivo=False, capturar=CapturaSpec())
        assert r["ok"] and r["backend"] == backend, r.get("error")
        filas[backend] = r["laboratorio"]["filas"]
    assert filas["torch"] == filas["stdlib"]
    assert filas["torch"]["de"] == 40 and len(filas["torch"]["x"]) == 20
    # Las filas que el modelo no alcanza (objetivo > 1) están: son las de la validación.
    assert max(y[0] for y in filas["torch"]["y"]) > 1.0


@needs_torch
def test_arranque_en_caliente_sin_mejora_lo_dice():
    """M-7: si ninguna época mejora los pesos de partida, los finales son esos (época 0) y
    el bloque lo dice: `de_partida` y su motivo, y la época 0 de la muestra es «partida»."""
    from matrixai.forward.dense_torch import (
        dense_module_to_state_dict, dense_network_to_torch_module)
    from matrixai.training.dense_torch_trainer import train_dense_network_torch
    from matrixai.training.laboratorio import CapturaSpec
    net, _ps = _setup()
    r0 = _entrenar(epochs=30, lr=0.3)
    sd = dense_module_to_state_dict(net, dense_network_to_torch_module(net, r0["best_params"]))
    net, ps = _setup()
    r = train_dense_network_torch(net, ps, _ejemplos(200), "cross_entropy", lr=5.0, epochs=3,
                                  device="cpu", seed=3, batch_size=8, materialize=True,
                                  initial_state_dict=sd, capturar=CapturaSpec())
    assert r["best_epoch"] == 0, "el caso tenía que no mejorar los pesos de partida"
    lab = r["laboratorio"]
    pf = lab["pesos_finales"]
    assert pf["epoca"] == 0 and pf["de_partida"] is True
    assert pf["motivo"]["es"] and pf["motivo"]["en"]
    motivos = {m["epoca"]: m["motivo"] for m in lab["pesos_por_epoca"]}
    assert "partida" in motivos[0] and "mejor" in motivos[0]
    # Y sin arranque en caliente, nada de «partida».
    r = _entrenar(capturar=CapturaSpec(), epochs=3)
    assert "de_partida" not in r["laboratorio"]["pesos_finales"]


def test_sin_torch_declara_que_no(monkeypatch):
    """Sin torch: solo los pesos finales —los de la MEJOR época, en un caso con la mejor
    ANTES de la última— y las filas; lo que falta, declarado, con lo pedido de verdad."""
    from matrixai.playground_api import generate_project_from_dataset
    from matrixai.training.laboratorio import CapturaSpec
    rng = random.Random(0)
    lineas = ["a,b,y"] + [f"{rng.random():.4f},{rng.random():.4f},{'si' if rng.random() > 0.5 else 'no'}"
                          for _ in range(40)]
    gen = generate_project_from_dataset("\n".join(lineas) + "\n", "y", locale="es",
                                        column_type_overrides={"y": "categorical"})
    tt = gen["training_text"].replace("LEARNING_RATE 0.01", "LEARNING_RATE 0.3")
    assert tt != gen["training_text"]
    gen = {**gen, "training_text": tt}
    r = _entrenar_playground(monkeypatch, "stdlib", gen=gen, epochs=15, capturar=CapturaSpec())
    assert r["ok"] and r["backend"] == "stdlib"
    assert r["best_epoch"] < len(r["epochs"]), "el caso tenía que tener la mejor ANTES de la última"
    lab = r["laboratorio"]
    assert lab["entrenador"] == "stdlib" and lab["disponible"] is True
    valores = r["params_best"]["parameters"]
    for c, f in zip(lab["pesos_finales"]["capas"], lab["capas"]):
        assert c["W"] == valores[f["clave_W"]]["values"]
        assert c["b"] == valores[f["clave_b"]]["values"]
    assert lab["pesos_finales"]["epoca"] == r["best_epoch"]
    assert lab["filas"] and lab["filas"]["columnas"] == ["a", "b"] and len(lab["filas"]["x"]) <= 20
    assert lab["numeros_por_parte"] == _cuenta_partes(lab)
    assert lab["pesos_por_epoca"] is None and lab["grad_norm"] is None
    om = _omitidos(lab)
    assert {"pesos_por_epoca", "grad_norm"} <= set(om)
    for o in (om["pesos_por_epoca"], om["grad_norm"]):
        assert "sin torch" in o["motivo"]["es"] and "without torch" in o["motivo"]["en"]
    assert (om["pesos_por_epoca"]["pedido"], om["pesos_por_epoca"]["unidad"]) == (12, "epocas")
    # La validación es pequeña (8 filas): también se dice, y `de` es verdadero.
    assert om["filas"]["aplicado"] == len(lab["filas"]["x"]) == lab["filas"]["de"] == 8
    assert r["laboratorio_resumen"]["entrenador"] == "stdlib"
    assert r["laboratorio_resumen"]["gradientes"] is False
    # M-9: lo pedido es lo que se pidió (aquí, 3 épocas: menos que el tope de 12).
    r3 = _entrenar_playground(monkeypatch, "stdlib", gen=gen, epochs=3, capturar=CapturaSpec())
    assert _omitidos(r3["laboratorio"])["pesos_por_epoca"]["pedido"] == 3
    # Y sin pedirlo, nada.
    sin = _entrenar_playground(monkeypatch, "stdlib", gen=gen, epochs=15)
    assert "laboratorio" not in sin and "laboratorio_resumen" not in sin
    assert sin["params_best"] == r["params_best"]


def test_la_peticion_congela_sus_listas_y_respeta_los_nombres():
    """M-11: una lista que llega de JSON se congela (la petición sigue siendo hashable) y
    `con_nombres` no pisa los nombres que el llamante ya fijó."""
    from matrixai.training.laboratorio import CapturaSpec
    s = CapturaSpec(columnas=["a", "b"])
    assert s.columnas == ("a", "b") and hash(s) == hash(CapturaSpec(columnas=("a", "b")))
    n = s.con_nombres(["x", "y"], ["P", "Q"])
    assert n.columnas == ("a", "b") and n.etiquetas == ("P", "Q")
    assert CapturaSpec().con_nombres([], None).columnas is None


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


# ===========================================================================
# Re-auditoría del 133-C2: los menores (N-1 … N-7)
# ===========================================================================
def _sin_silencios(lab):
    """Nada que se dé por guardado falta, y nada se guarda ni se omite en silencio: cada
    parte que está a `None` (o sin filas) tiene su entrada en `omitido`, lo contado en el
    bloque es lo declarado, y el resumen público dice lo mismo que el bloque."""
    om = _omitidos(lab)
    for parte in ("grad_norm", "pesos_por_epoca", "pesos_finales", "filas"):
        if lab[parte] is None:
            assert parte in om, f"{parte} falta y no se declara"
    assert lab["numeros_por_parte"] == _cuenta_partes(lab)
    assert lab["numeros"] == sum(lab["numeros_por_parte"].values())
    assert lab["numeros"] <= lab["tope"]["max_numeros"]
    from matrixai.training.laboratorio import resumen_del_bloque
    res = resumen_del_bloque(lab)
    assert res["gradientes"] is (lab["grad_norm"] is not None)
    assert res["pesos_finales"] is (lab["pesos_finales"] is not None)
    assert res["epocas_con_pesos"] == [m["epoca"] for m in lab["pesos_por_epoca"] or []]
    assert res["omitido"] == [o["que"] for o in lab["omitido"]]


def _linea_del_entrenador(patron):
    """El número de línea (de `dense_torch_trainer.py`) de la ÚNICA línea que lleva `patron`."""
    import matrixai.training.dense_torch_trainer as m
    with open(m.__file__, encoding="utf-8") as f:
        hits = [i + 1 for i, l in enumerate(f) if patron in l]
    assert len(hits) == 1, (patron, hits)
    return hits[0]


def _epoca_si_llama_la_linea(lineas):
    """(¿lo llama el entrenador de torch desde una de esas líneas?, época en curso)."""
    import sys
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_name == "train_dense_network_torch":
            return f.f_lineno in lineas, f.f_locals.get("epoch")
        f = f.f_back
    return False, None


def _fallo_en_clone(monkeypatch, patron, desde_epoca, golpes):
    import torch
    original = torch.Tensor.clone
    lineas = {_linea_del_entrenador(patron)}

    def clone(self, *a, **k):
        dentro, ep = _epoca_si_llama_la_linea(lineas)
        if dentro and (ep or 0) >= desde_epoca:
            golpes.append(ep)
            raise RuntimeError("CUDA out of memory (simulado)")
        return original(self, *a, **k)
    monkeypatch.setattr(torch.Tensor, "clone", clone)


def _fallo_en_division(monkeypatch, patron, epoca, golpes):
    import torch
    original = torch.Tensor.__truediv__
    lineas = {_linea_del_entrenador(patron)}

    def div(self, otro):
        dentro, ep = _epoca_si_llama_la_linea(lineas)
        if dentro and ep == epoca:
            golpes.append(ep)
            raise RuntimeError("fallo simulado al cerrar la época")
        return original(self, otro)
    monkeypatch.setattr(torch.Tensor, "__truediv__", div)


def _fallo_en_norma_por_lote(monkeypatch, desde_epoca, golpes):
    import torch
    original = torch._foreach_norm

    def norma(*a, **k):
        _dentro, ep = _epoca_si_llama_la_linea(set(range(1, 100_000)))
        if (ep or 0) >= desde_epoca:
            golpes.append(ep)
            raise TypeError("fallo simulado en la medida")
        return original(*a, **k)
    monkeypatch.setattr(torch, "_foreach_norm", norma)


#: nombre → (cómo se inyecta, qué parte se apaga, la época del fallo, cómo se entrena)
_CORTO = {}
_LARGO = {"n": 100, "epochs": 50, "lr": 1.5, "early_stop": (2, "validation_loss")}
_FALLOS_DE_LA_CAPTURA = {
    "medida_por_lote_desde_la_epoca_3": ("grad_norm", 3, _CORTO),
    "cierre_de_epoca_en_la_2": ("grad_norm", 2, _CORTO),
    "copia_de_pesos_de_una_epoca_programada_en_la_3": ("pesos_por_epoca", 3, _CORTO),
    "copia_de_la_ultima_epoca_sin_programar": ("pesos_por_epoca", None, _LARGO),
}


@needs_torch
@pytest.mark.parametrize("nombre", list(_FALLOS_DE_LA_CAPTURA))
def test_un_fallo_de_cada_guarda_apaga_su_parte_y_el_entrenamiento_sigue(monkeypatch, nombre):
    """N-1: las cuatro guardas `try` del entrenador de torch (la medida por lote, el cierre
    de época de las normas, la copia de pesos de una época programada y la de la última
    época). Con un fallo inyectado EN esa línea: el entrenamiento sigue en torch con los
    mismos pesos, curvas, mejor época y generador que sin captura; solo se apaga la parte
    afectada y se declara en `omitido` (texto fijo, tipo de la excepción y época); las
    demás partes siguen; y nada falta ni se recorta en silencio. Quitar una guarda tumba el
    entrenamiento; quitar su declaración deja el hueco callado; no vaciar lo ya medido deja
    gradientes que el `omitido` dice que no hay."""
    import torch
    from matrixai.training.laboratorio import CapturaSpec, FALLO_GRADIENTE, FALLO_PESOS_POR_EPOCA
    parte, epoca, kw = _FALLOS_DE_LA_CAPTURA[nombre]
    sin = _entrenar(**kw)
    rng_sin = torch.get_rng_state().clone()
    golpes: list = []
    if nombre.startswith("medida_por_lote"):
        _fallo_en_norma_por_lote(monkeypatch, 3, golpes)
    elif nombre.startswith("cierre_de_epoca"):
        _fallo_en_division(monkeypatch, "cap_hist.append((epoch, cap_acc / max(1, cap_n)))", 2, golpes)
    elif nombre.startswith("copia_de_pesos"):
        _fallo_en_clone(monkeypatch, "cap_estados[epoch] = _snapshot()", 3, golpes)
    else:
        _fallo_en_clone(monkeypatch, "cap_estados[cap_ultima] = _snapshot()", 1, golpes)
    con = _entrenar(capturar=CapturaSpec(), **kw)
    rng_con = torch.get_rng_state().clone()
    assert golpes, "el fallo inyectado no se produjo: la prueba no mide nada"
    # El entrenamiento, igual que sin captura, y en torch.
    assert con["backend"] == "torch"
    assert _pesos(con) == _pesos(sin) and con["epochs"] == sin["epochs"]
    assert con["best_epoch"] == sin["best_epoch"] and torch.equal(rng_con, rng_sin)
    if epoca is None:
        epoca = sin["epochs"][-1]["epoch"]
        assert epoca < kw["epochs"], "el caso tenía que parar antes de tiempo (última sin programar)"
    # La parte afectada, apagada y declarada; las demás, intactas.
    lab = con["laboratorio"]
    assert lab["disponible"] is True and lab["entrenador"] == "torch"
    assert lab[parte] is None
    otras = {"grad_norm", "pesos_por_epoca"} - {parte}
    for otra in otras:
        assert lab[otra], f"{otra} no tenía por qué apagarse"
    assert lab["pesos_finales"] is not None and lab["filas"] is not None
    om = _omitidos(lab)
    assert set(om) == {parte}
    o = om[parte]
    assert o["error"] in ("RuntimeError", "TypeError") and o["epoca"] == epoca
    assert o["aplicado"] == 0 and o["pedido"] > 0 and o["unidad"] == "epocas"
    textos = FALLO_GRADIENTE if parte == "grad_norm" else FALLO_PESOS_POR_EPOCA
    assert (o["motivo"]["es"], o["motivo"]["en"]) == textos
    assert "simulado" not in str(lab) and "CUDA" not in str(lab)
    _sin_silencios(lab)


@needs_torch
def test_un_fallo_de_la_copia_de_pesos_por_la_superficie_publica_sigue_en_torch(monkeypatch):
    """N-1 por la superficie pública: una copia de pesos que falla no hace caer el
    entrenamiento a stdlib ni da `ok: false` (con adam/adamw lo daba)."""
    from matrixai.playground import MODO_TORCH_EN_CPU
    sin = _entrenar_playground(monkeypatch, "torch", epochs=6)
    golpes: list = []
    _fallo_en_clone(monkeypatch, "cap_estados[epoch] = _snapshot()", 2, golpes)
    from matrixai.training.laboratorio import CapturaSpec
    con = _entrenar_playground(monkeypatch, "torch", epochs=6, capturar=CapturaSpec())
    assert golpes
    assert con["ok"] and con["backend"] == "torch" and MODO_TORCH_EN_CPU
    assert con["params_best"] == sin["params_best"] and con["epochs"] == sin["epochs"]
    lab = con["laboratorio"]
    assert lab["entrenador"] == "torch" and lab["pesos_por_epoca"] is None
    assert [o["que"] for o in lab["omitido"]] == ["pesos_por_epoca"]
    _sin_silencios(lab)


# ---------------------------------------------------------------------------
# N-2: el motivo de «no hay pesos» es verdadero para cada entrenador
# ---------------------------------------------------------------------------
@needs_torch
def test_el_motivo_de_no_guardar_pesos_es_verdadero_con_y_sin_gradientes(monkeypatch):
    """N-2: «solo se guardan las normas de gradiente» era falso sin torch (no hay ninguna).
    El texto de los pesos finales no habla de lo que sí hay: dice lo que NO hay, y la
    verdad de lo demás la dicen sus propias entradas."""
    from matrixai.training.laboratorio import CapturaSpec
    spec = CapturaSpec(max_params_finales=1)
    con_torch = _entrenar(capturar=spec, epochs=4)["laboratorio"]
    sin_torch = _entrenar_playground(monkeypatch, "stdlib", epochs=3, capturar=spec)["laboratorio"]
    assert con_torch["grad_norm"] is not None and sin_torch["grad_norm"] is None
    for lab in (con_torch, sin_torch):
        o = _omitidos(lab)["pesos_finales"]
        assert lab["pesos_finales"] is None
        assert "ninguno de los pesos" in o["motivo"]["es"] and "none of" in o["motivo"]["en"]
        assert "gradiente" not in o["motivo"]["es"] and "gradient" not in o["motivo"]["en"]
        assert "gradiente" not in _omitidos(lab)["pesos_por_epoca"]["motivo"]["es"].split(
            "Sin pesos finales")[0]
    # Sin torch, lo que falta de los gradientes lo dice SU entrada.
    assert "sin torch" in _omitidos(sin_torch)["grad_norm"]["motivo"]["es"]


# ---------------------------------------------------------------------------
# N-3: el texto de la muestra de gradientes dice lo que PASA, y las fronteras
# ---------------------------------------------------------------------------
def _hist(n):
    return [(e, e) for e in range(1, n + 1)]


def test_la_muestra_de_gradientes_dice_cuales_de_primera_mejor_ultima_entraron():
    from matrixai.training.laboratorio import muestrear_gradientes
    for n, esperado_es, esperado_en, no_es in (
            (3, "la primera, la mejor y la última", "the first, the best and the last", ()),
            (2, "la mejor y la última", "the best and the last", ("primera",)),
            (1, "la mejor", "the best", ("primera", "última"))):
        guardadas, om = muestrear_gradientes(_hist(30), 17, n)
        assert len(guardadas) == n and om["aplicado"] == n == len(guardadas)
        assert esperado_es in om["motivo"]["es"] and esperado_en in om["motivo"]["en"]
        for palabra in no_es:
            assert palabra not in om["motivo"]["es"]
        assert 17 in [e for e, _v in guardadas]
    # Arranque en caliente sin mejora (mejor = 0): la «mejor» no tiene gradiente, y no se promete.
    guardadas, om = muestrear_gradientes(_hist(30), 0, 5)
    assert {1, 30} <= {e for e, _v in guardadas}
    assert "la primera y la última" in om["motivo"]["es"] and "mejor" not in om["motivo"]["es"]
    assert "the first and the last" in om["motivo"]["en"] and "best" not in om["motivo"]["en"]


def test_la_frontera_de_la_muestra_de_gradientes_no_declara_de_mas():
    """N-7 (R2): con tantas épocas como cupo NO hay muestra ni `omitido`; con una más, sí."""
    from matrixai.training.laboratorio import muestrear_gradientes
    guardadas, om = muestrear_gradientes(_hist(50), 10, 50)
    assert om is None and [e for e, _v in guardadas] == list(range(1, 51))
    guardadas, om = muestrear_gradientes(_hist(51), 10, 50)
    assert om is not None and len(guardadas) == 50 == om["aplicado"] and om["pedido"] == 51


# ---------------------------------------------------------------------------
# N-4: `aplicado` es lo GUARDADO
# ---------------------------------------------------------------------------
@needs_torch
@pytest.mark.parametrize("n_muestra", [1, 2, 3, 12])
def test_aplicado_de_los_pesos_por_epoca_es_lo_guardado(n_muestra):
    """N-4: con `n_muestra` de 1 o 2 no se programa ninguna época y la muestra es {mejor,
    última}; si son la misma época se guarda UNA, y `aplicado` lo dice (antes decía 2)."""
    from matrixai.training.laboratorio import CapturaSpec
    # Pérdida que baja siempre (lr bajo): la mejor ES la última.
    r = _entrenar(capturar=CapturaSpec(max_epocas_muestra=n_muestra), epochs=6, lr=0.05)
    lab = r["laboratorio"]
    assert r["best_epoch"] == r["epochs"][-1]["epoch"], "el caso tenía que tener mejor == última"
    guardadas = len(lab["pesos_por_epoca"] or [])
    om = _omitidos(lab).get("pesos_por_epoca")
    if n_muestra == 1:
        assert guardadas == 1 and om is None        # pidió 1 y tiene 1: no falta nada
    elif n_muestra == 2:
        assert guardadas == 1
        assert om is not None and om["aplicado"] == guardadas and om["pedido"] == n_muestra
        assert om["unidad"] == "epocas" and om["motivo"]["es"] and om["motivo"]["en"]
    else:
        assert om is None and guardadas > 1
    _sin_silencios(lab)
    # Y cuando es el TOPE de números el que da 2 huecos, el `aplicado` del recorte también
    # es lo guardado, no los 2 que cabían.
    tope = 6 * 4 + 42 + 2 * 42
    r = _entrenar(capturar=CapturaSpec(max_filas=0, max_numeros=tope), epochs=6, lr=0.05)
    lab = r["laboratorio"]
    om = _omitidos(lab)["pesos_por_epoca"]
    assert om["aplicado"] == len(lab["pesos_por_epoca"]) == 1 and om["pedido"] == 6
    _sin_silencios(lab)


# ---------------------------------------------------------------------------
# N-5: `pedido` y `aplicado` son DATOS; el texto, solo en `motivo`
# ---------------------------------------------------------------------------
def _todos_los_omitidos(monkeypatch):
    """Entradas de `omitido` de muchas configuraciones, con torch y sin torch."""
    from matrixai.training.laboratorio import CapturaSpec, planear
    pinchos = []
    for spec in (CapturaSpec(), CapturaSpec(gradientes=False), CapturaSpec(pesos_finales=False),
                 CapturaSpec(max_filas=0), CapturaSpec(max_epocas_muestra=0),
                 CapturaSpec(max_params_muestra=1), CapturaSpec(max_params_finales=1),
                 CapturaSpec(max_numeros=100), CapturaSpec(max_numeros=3),
                 CapturaSpec(max_epocas_grad=2, max_numeros=150)):
        pinchos.append(planear(spec, 42, 2, 20, 5, 4)["omitido"])
        if _HAS_TORCH:
            pinchos.append(_entrenar(capturar=spec, epochs=20)["laboratorio"]["omitido"])
        pinchos.append(_entrenar_playground(monkeypatch, "stdlib", epochs=3,
                                            capturar=spec)["laboratorio"]["omitido"])
    return [o for lista in pinchos for o in lista]


def test_pedido_y_aplicado_son_numeros_con_su_unidad_y_el_texto_va_en_motivo(monkeypatch):
    from matrixai.training.laboratorio import UNIDADES_OMITIDO
    todos = _todos_los_omitidos(monkeypatch)
    assert len(todos) > 20
    for o in todos:
        assert type(o["pedido"]) is int and type(o["aplicado"]) is int, o
        assert o["pedido"] >= 0 and o["aplicado"] >= 0
        assert o["unidad"] in UNIDADES_OMITIDO
        assert set(o["motivo"]) == {"es", "en"} and o["motivo"]["es"] and o["motivo"]["en"]
        assert o["aplicado"] <= o["pedido"] or o["pedido"] == 0 or o["que"] == "filas", o
    assert {o["unidad"] for o in todos} == set(UNIDADES_OMITIDO)


# ---------------------------------------------------------------------------
# N-6: lo no pedido se declara siempre igual
# ---------------------------------------------------------------------------
@needs_torch
def test_lo_no_pedido_se_declara_como_no_pedido_en_todas_las_partes(monkeypatch):
    from matrixai.training.laboratorio import CapturaSpec
    for entrenar in (lambda s: _entrenar(capturar=s, epochs=4)["laboratorio"],
                     lambda s: _entrenar_playground(monkeypatch, "stdlib", epochs=3,
                                                    capturar=s)["laboratorio"]):
        lab = entrenar(CapturaSpec(gradientes=False, max_filas=0, pesos_finales=False,
                                   max_epocas_muestra=0))
        om = _omitidos(lab)
        assert set(om) == {"grad_norm", "filas", "pesos_finales", "pesos_por_epoca"}
        assert [lista for lista in om.values()
                if (lista["pedido"], lista["aplicado"]) != (0, 0)] == []
        assert lab["grad_norm"] is None and lab["filas"] is None
        assert "No se pidieron" in om["grad_norm"]["motivo"]["es"]
        assert "not requested" in om["grad_norm"]["motivo"]["en"]
        assert "No se pidieron" in om["filas"]["motivo"]["es"]
        assert "were requested" in om["filas"]["motivo"]["en"]
        _sin_silencios(lab)
    # Pidiéndolo todo, sin torch: los gradientes no se declaran DOS veces.
    sin = _entrenar_playground(monkeypatch, "stdlib", epochs=3, capturar=CapturaSpec())
    assert [o["que"] for o in sin["laboratorio"]["omitido"]].count("grad_norm") == 1


# ---------------------------------------------------------------------------
# N-7: las fronteras de los topes
# ---------------------------------------------------------------------------
def test_los_gradientes_no_pasan_del_total_aunque_el_tope_de_epocas_sea_mayor():
    """N-7 (R11): el total de números manda sobre `max_epocas_grad`."""
    from matrixai.training.laboratorio import CapturaSpec, planear
    p = planear(CapturaSpec(max_numeros=100, max_epocas_grad=200), 10, 2, 200, 20, 4)
    assert p["gradientes"] is True and p["n_epocas_grad"] == 25      # 25 épocas × 4 = 100
    assert [o["que"] for o in p["omitido"]] == ["pesos_finales", "filas", "pesos_por_epoca"]
    # Y ni una época cabe: se apaga y se dice.
    p = planear(CapturaSpec(max_numeros=3), 10, 2, 200, 20, 4)
    assert p["gradientes"] is False and p["n_epocas_grad"] == 0
    assert _omitidos(p)["grad_norm"]["aplicado"] == 0


@needs_torch
def test_el_bloque_con_gradientes_no_pasa_del_total():
    """N-7 (R11), de punta a punta: 60 épocas medidas, un total de 100 números."""
    from matrixai.training.laboratorio import CapturaSpec
    lab = _entrenar(capturar=CapturaSpec(max_numeros=100, max_filas=0), epochs=60)["laboratorio"]
    assert len(lab["grad_norm"]["epocas"]) == 25 and lab["grad_norm"]["epocas_medidas"] == 60
    assert lab["numeros"] <= 100 and lab["numeros_por_parte"]["grad_norm"] == 100
    _sin_silencios(lab)


def test_sin_torch_las_filas_se_cuentan_por_su_ancho_real(monkeypatch):
    """N-7 (R12): sin torch, el tope de números recorta las filas con su ancho real."""
    from matrixai.training.laboratorio import CapturaSpec
    base = _entrenar_playground(monkeypatch, "stdlib", epochs=3, capturar=CapturaSpec())["laboratorio"]
    n_params = base["parametros"]
    ancho = len(base["filas"]["x"][0]) + len(base["filas"]["y"][0])
    assert ancho > 2 and base["filas"]["de"] >= 8
    tope = n_params + 3 * ancho + 1
    lab = _entrenar_playground(monkeypatch, "stdlib", epochs=3,
                               capturar=CapturaSpec(max_numeros=tope))["laboratorio"]
    assert len(lab["filas"]["x"]) == 3 and _omitidos(lab)["filas"]["aplicado"] == 3
    assert lab["numeros"] <= tope and lab["numeros_por_parte"] == _cuenta_partes(lab)
    _sin_silencios(lab)


def test_un_entrenamiento_fallido_no_lleva_bloque_de_laboratorio():
    """N-7 (R13): sin `ok`, el resultado se devuelve tal cual; con `ok`, lleva el bloque."""
    from matrixai import playground
    from matrixai.training.laboratorio import CapturaSpec
    motivo = ("no hay", "there is none")
    fallido = {"ok": False, "error": "x", "backend": "torch"}
    assert playground._sin_laboratorio(dict(fallido), CapturaSpec(), motivo) == fallido
    bueno = playground._sin_laboratorio({"ok": True, "backend": "torch"}, CapturaSpec(), motivo)
    assert bueno["laboratorio"]["disponible"] is False and "laboratorio_resumen" in bueno
    assert "laboratorio" not in playground._sin_laboratorio({"ok": True}, None, motivo)
