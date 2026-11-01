"""140b-C0 -- simulacion de la cobertura del bootstrap por unidad (D4). Verdad conocida.
Modelo: y_ij = u_i + d(t_ij) + e_ij ; u_i ~ N(0, rho), e_ij ~ N(0, 1-rho) ; d = proceso gaussiano suave COMPARTIDO por todos los
sujetos (funcion del tiempo de calendario, no del sujeto), varianza tau^2, longitud de correlacion ELL en una ventana W.
Predictor fijo (no entrenado): y_hat = 0 -> error cuadratico = y^2 ; metrica = media de filas del error cuadratico (RMSE = su raiz).
Verdad CONDICIONAL = lo que mediria la poblacion de sujetos nuevos en ESTA ventana con ESTE d realizado.
Verdad MARGINAL = promediando tambien sobre el d (varianza total 1 + tau^2)."""
import math
import numpy as np

W = 100.0     # ventana de prueba (unidades de tiempo)
ELL = 30.0    # longitud de correlacion de la deriva
F = 24        # cosenos aleatorios

def deriva(tau, rng):
    if not tau:
        return lambda t: np.zeros_like(t, dtype=float)
    om = rng.normal(0, 1 / ELL, F); ph = rng.uniform(0, 2 * math.pi, F)
    return lambda t: tau * math.sqrt(2.0 / F) * np.cos(np.multiply.outer(np.asarray(t, float), om) + ph).sum(-1)

def visitas(n, m, cal, rng):
    """Devuelve (unidad por fila, tiempo por fila, grid de la distribucion de tiempos para la verdad)."""
    if m == "d10":
        mi = np.clip(np.round(rng.lognormal(math.log(10), 1.0, n)).astype(int), 1, 100)
    else:
        mi = np.full(n, int(m))
    u = np.repeat(np.arange(n), mi)
    if cal == "panel":
        if m == "d10":
            grid = (np.arange(100) + 0.5) / 100 * W
            t = np.concatenate([rng.choice(grid, size=k, replace=False) for k in mi])
        else:
            grid = (np.arange(int(m)) + 0.5) / int(m) * W
            t = np.concatenate([grid for _ in range(n)])
    else:
        grid = (np.arange(1000) + 0.5) / 1000 * W
        t = rng.uniform(0, W, len(u))
    return u, t, grid

def replica_rmse(n, m, rho, tau, cal, rng):
    """Una muestra de prueba. Devuelve y, unidad, verdad_condicional (MSE), verdad_marginal (MSE)."""
    d = deriva(tau, rng)
    u, t, grid = visitas(n, m, cal, rng)
    y = rng.normal(0, math.sqrt(rho), n)[u] + d(t) + rng.normal(0, math.sqrt(1 - rho), len(u))
    tc = 1.0 + float(np.mean(d(grid) ** 2))
    return y, u, tc, 1.0 + tau ** 2

def ic_mse(y, u, B, rng, nivel=0.95):
    n = int(u.max()) + 1
    S = np.bincount(u, weights=y ** 2, minlength=n); M = np.bincount(u, minlength=n).astype(float)
    idx = rng.integers(0, n, size=(B, n))
    est = S[idx].sum(1) / M[idx].sum(1)
    a = (1 - nivel) / 2
    lo, hi = np.quantile(est, [a, 1 - a])
    return S.sum() / M.sum(), float(lo), float(hi)

# ------------------------------------------------------------------ AUROC
def _auc_filas(x, y):
    from scipy.stats import rankdata
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def replica_auc(n, m, sd_u, tau, rng, n_verdad=6000):
    """m visitas en rejilla comun (panel). eta = -0.5 + 1.2 x + u_i + d(t). Puntuacion = x (modelo fijo que no ve u ni d).
    Verdad condicional por Monte Carlo con n_verdad sujetos nuevos y el MISMO d."""
    d = deriva(tau, rng)
    grid = (np.arange(m) + 0.5) / m * W
    def gen(k):
        u = rng.normal(0, sd_u, k)[:, None]
        x = rng.normal(0, 1, (k, m))
        eta = -0.5 + 1.2 * x + u + d(grid)[None, :]
        y = (rng.random((k, m)) < 1 / (1 + np.exp(-eta))).astype(int)
        return x, y
    x, y = gen(n)
    xv, yv = gen(n_verdad)
    return x, y, _auc_filas(xv.ravel(), yv.ravel())

def ic_auc(x, y, B, rng, nivel=0.95):
    from scipy.stats import rankdata
    n, m = x.shape
    idx = rng.integers(0, n, size=(B, n))
    X = x[idx].reshape(B, -1); Y = y[idx].reshape(B, -1)
    n1 = Y.sum(1).astype(float); n0 = Y.shape[1] - n1
    ok = (n1 > 0) & (n0 > 0)
    r = rankdata(X, axis=1)
    auc = ((r * Y).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1)
    auc = auc[ok]
    a = (1 - nivel) / 2
    return _auc_filas(x.ravel(), y.ravel()), float(np.quantile(auc, a)), float(np.quantile(auc, 1 - a)), float(ok.mean())
