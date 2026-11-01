"""140b-C0 -- biblioteca de la medicion (solo lectura de datos y repos; no escribe fuera de c0/).
Generador "nuevo-prototipo" del reparto estricto (B), pliegues F1/F2, minimos, datos reales y sinteticos."""
import os, sys, math, random, json, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
NUCLEO = "/home/deployer/worktrees/rev30/matrixAI"
MOTORES = "/home/deployer/worktrees/rev30/matrixai-engines/src"
for _p in (NUCLEO, MOTORES, NUCLEO + "/benchmarks/fechas_114c5"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, pandas as pd

PARK = "/home/deployer/matrixaistudio/studio-backend/matrixai_studio/ejemplos_medidos/parkinsons-telemonitoring.csv"
AVO = "/home/deployer/fechas_114c5_datos/43820.arff"
CACHE_LLUVIA = "/home/deployer/.agentes/140b/inventario/cache_lluvia.npz"
SALIDAS = "/home/deployer/.agentes/140b/c0/salidas"

# ----------------------------------------------------------------------------- datos
def _ds(nombre, g, t, y_reg=None, y_clas=None, X=None):
    g = np.asarray(g).astype(str)
    uniq, gi = np.unique(g, return_inverse=True)
    return dict(nombre=nombre, g=g, gi=gi, U=len(uniq), t=np.asarray(t, float),
                y_reg=None if y_reg is None else np.asarray(y_reg, float),
                y_clas=None if y_clas is None else np.asarray(y_clas, int), X=X)

def datos_reales():
    d = pd.read_csv(PARK)
    g = d["subject#"].astype(str).values; trel = d["test_time"].values; y = d["total_UPDRS"].values
    X = d.drop(columns=["subject#", "total_UPDRS"]).values
    out = {}
    out["parkinson"] = _ds("parkinson", g, trel, y, (y >= 30).astype(int), X)
    us = sorted(set(g))
    for nombre, ventana, sem in (("parkinson_cal365", 365, 0), ("parkinson_cal1000", 1000, 1)):
        rnd = np.random.default_rng(sem)
        ent = {u: rnd.uniform(0, ventana) for u in us}
        out[nombre] = _ds(nombre, g, np.array([ent[u] for u in g]) + trel, y, (y >= 30).astype(int), X)
    from medir_114c5 import leer_arff
    from pathlib import Path
    from datetime import datetime
    filas, _ = leer_arff(Path(AVO))
    a = pd.DataFrame(filas)
    ta = np.array([float((datetime.strptime(s, "%Y-%m-%d") - datetime(1970, 1, 1)).days) for s in a["Date"]])
    ya = a["AveragePrice"].astype(float).values
    Xa = np.column_stack([a[c].astype(float).values for c in ("Total_Volume", "4046", "4225", "4770", "Total_Bags",
                          "Small_Bags", "Large_Bags", "XLarge_Bags", "year")] + [(a["type"] == "organic").astype(float).values])
    out["avocado"] = _ds("avocado", a["region"].values, ta, ya, (ya >= np.median(ya)).astype(int), Xa)
    z = np.load(CACHE_LLUVIA, allow_pickle=True)
    out["lluvia"] = _ds("lluvia", z["g"], z["t"], None, z["y"].astype(int), None)
    return out

def sintetico(n_suj, visitas, calendario, seed, sd_u=1.5, seguimiento=100.0):
    """Cohorte sintetica. visitas: entero (filas por sujeto) o 'd20' (desigual, lognormal con mediana 20).
    calendario: 'panel' (todos empiezan a la vez), 'esc1' (entrada uniforme en 1x seguimiento), 'esc3' (3x).
    Dos objetivos binarios con efecto-sujeto: equilibrado (a=0) y raro (a=-4.5)."""
    rng = np.random.default_rng([seed, n_suj, 7 if visitas == "d20" else int(visitas), {"panel": 0, "esc1": 1, "esc3": 2}[calendario]])
    if visitas == "d20":
        m = np.clip(np.round(rng.lognormal(math.log(20), 1.2, n_suj)).astype(int), 1, 2000)
    else:
        m = np.full(n_suj, int(visitas))
    ent = np.zeros(n_suj) if calendario == "panel" else rng.uniform(0, seguimiento * (1 if calendario == "esc1" else 3), n_suj)
    gi = np.repeat(np.arange(n_suj), m)
    t = ent[gi] + rng.uniform(0, seguimiento, len(gi))
    u = rng.normal(0, sd_u, n_suj)[gi]
    yb = (rng.random(len(gi)) < 1 / (1 + np.exp(-u))).astype(int)
    yr = (rng.random(len(gi)) < 1 / (1 + np.exp(-(u - 4.5)))).astype(int)
    return gi, t, yb, yr

# ----------------------------------------------------------------------------- reparto estricto (B)
def reparto_estricto(gi, U, t, s, q, gap, seed, elegibles=True, min_post=1):
    """B: sujetos de prueba (cuota s de TODOS los sujetos, al azar) + corte c = cuantil q de las filas.
    desarrollo = sujetos de desarrollo con t < c-gap/2 ; prueba = filas de los sujetos de prueba con t >= c+gap/2.
    elegibles=False reproduce el prototipo del inventario (B0); True escoge los sujetos de prueba entre los que
    tienen >= min_post filas posteriores (B1). Devuelve dict o None."""
    c = float(np.quantile(t, q)); mg = gap / 2.0
    post = t >= c + mg
    n_post = np.bincount(gi[post], minlength=U)
    orden = list(range(U)); random.Random(seed).shuffle(orden)   # los indices siguen el orden de los ids (como el inventario)
    n_te = max(1, round(U * s))
    if elegibles:
        orden = [u for u in orden if n_post[u] >= min_post]
    te_u = np.array(orden[:n_te], dtype=int)
    es_te = np.zeros(U, bool); es_te[te_u] = True
    dev = np.where(~es_te[gi] & (t < c - mg))[0]
    te = np.where(es_te[gi] & post)[0]
    if len(dev) == 0 or len(te) == 0:
        return None
    return dict(dev=dev, te=te, c=c, te_u=te_u, desc=len(t) - len(dev) - len(te), n=len(t))

MIN_SUJ, MIN_FILAS, MIN_SUJ_CLASE = 5, 50, 2

def minimos(r, gi, y_clas=None, clases=(0, 1)):
    """M1 >=5 sujetos con filas en la prueba; M2 >=50 filas; M3 cada clase en >=2 sujetos de prueba (si hay clase)."""
    te = r["te"]
    n_suj = len(np.unique(gi[te]))
    out = dict(n_suj=n_suj, n_filas=len(te), m1=n_suj >= MIN_SUJ, m2=len(te) >= MIN_FILAS)
    if y_clas is not None:
        yt = y_clas[te]; gt = gi[te]
        por_clase = {k: len(np.unique(gt[yt == k])) for k in clases}
        out["suj_por_clase"] = por_clase
        out["m3"] = all(v >= MIN_SUJ_CLASE for v in por_clase.values())
        out["pos_prueba"] = int(yt.sum())
    else:
        out["m3"] = True
    out["ok"] = bool(out["m1"] and out["m2"] and out["m3"])
    return out

S_ESCALERA = (0.2, 0.3, 0.4)
Q_ESCALERA = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)

def reparto_auto(gi, U, t, y_clas, seed, gap=0.0):
    """Corte automatico: por cuota de sujetos s creciente y, dentro, el mayor cuantil q (mas desarrollo) que cumple los
    minimos. Devuelve (reparto, mins, s, q) o (None, motivo, ...) con los numeros del mejor intento."""
    mejor = None
    for s in S_ESCALERA:
        for q in Q_ESCALERA:
            r = reparto_estricto(gi, U, t, s, q, gap, seed, elegibles=True)
            if r is None:
                continue
            m = minimos(r, gi, y_clas)
            if m["ok"]:
                return r, m, s, q
            if mejor is None or (m["n_suj"], m["n_filas"]) > (mejor[0]["n_suj"], mejor[0]["n_filas"]):
                mejor = (m, s, q)
    return None, mejor, None, None

# ----------------------------------------------------------------------------- pliegues
def _asignar(sujetos, K, seed, puntuacion=None):
    """Reparte sujetos en K cubos. Sin puntuacion: al azar (orden barajado, us[k::K]). Con puntuacion (positivos del
    sujeto): voraz, de mayor a menor, al cubo con menos positivos (empate: menos sujetos)."""
    us = list(sujetos); random.Random(seed).shuffle(us)
    if puntuacion is None:
        return [set(us[k::K]) for k in range(K)]
    us.sort(key=lambda u: -puntuacion[u])
    cubos = [set() for _ in range(K)]; carga = [0.0] * K
    for u in us:
        k = min(range(K), key=lambda j: (carga[j], len(cubos[j])))
        cubos[k].add(u); carga[k] += puntuacion[u]
    return cubos

def pliegues(tipo, gi, t, dev, c, K, seed, y_clas=None, estratificar=False, w=0.3, gap=0.0):
    """F1: K pliegues por sujeto dentro del desarrollo (ignora el tiempo). F2: cada pliegue valida los sujetos de su
    cubo SOLO en el ultimo tramo (fraccion w) del desarrollo y entrena con los otros sujetos antes de ese tramo."""
    dev = np.asarray(dev); gdev = gi[dev]; tdev = t[dev]
    sujetos = np.unique(gdev)
    if tipo == "F2":      # tramo = ultima fraccion w del SPAN de tiempo del desarrollo
        inicio = c - w * (c - tdev.min())
    elif tipo == "F2q":   # tramo = el ultimo w de las FILAS del desarrollo (cuantil 1-w del tiempo)
        inicio = float(np.quantile(tdev, 1 - w))
    punt = None
    if estratificar and y_clas is not None:
        base = (tdev >= inicio) if tipo != "F1" else np.ones(len(dev), bool)
        pos = np.bincount(gdev[base & (y_clas[dev] == 1)], minlength=gi.max() + 1)
        punt = {int(u): int(pos[u]) for u in sujetos}
    cubos = _asignar([int(u) for u in sujetos], K, seed, punt)
    out = []
    for cubo in cubos:
        ec = np.zeros(gi.max() + 1, bool); ec[list(cubo)] = True
        en_cubo = ec[gdev]
        if tipo == "F1":
            val = dev[en_cubo]; ent = dev[~en_cubo]
        else:
            val = dev[en_cubo & (tdev >= inicio)]; ent = dev[~en_cubo & (tdev < inicio - gap)]
        out.append((ent, val))
    return out

MIN_VAL, MIN_ENT = 30, 100

def viabilidad(plieg, y_clas=None):
    """(ok1, ok5): valida >=30 filas, entrena >=100; clasificacion: ambas clases en valida y entrena (>=1) / en valida >=5."""
    ok1 = ok5 = True
    min_val = min(len(v) for _, v in plieg); min_ent = min(len(e) for e, _ in plieg)
    if min_val < MIN_VAL or min_ent < MIN_ENT:
        return False, False, min_val, min_ent, 0
    pos_min = 0
    if y_clas is not None:
        pos_min = 10**9
        for e, v in plieg:
            pv = int(y_clas[v].sum()); nv = len(v) - pv; pe = int(y_clas[e].sum()); ne = len(e) - pe
            pos_min = min(pos_min, pv, nv)
            if min(pv, nv, pe, ne) < 1: ok1 = False
            if min(pv, nv) < 5 or min(pe, ne) < 1: ok5 = False
    return ok1, (ok1 and ok5), min_val, min_ent, pos_min

# ----------------------------------------------------------------------------- bootstrap por unidad (numpy)
def ic_unidad(S, M, B, rng, nivel=0.95):
    """Bootstrap por unidad de una media de filas: S = suma por unidad, M = filas por unidad. Devuelve (punto, lo, hi)."""
    n = len(S)
    idx = rng.integers(0, n, size=(B, n))
    est = S[idx].sum(1) / M[idx].sum(1)
    a = (1 - nivel) / 2
    lo, hi = np.quantile(est, [a, 1 - a])
    return S.sum() / M.sum(), float(lo), float(hi)
