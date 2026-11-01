"""140b-C0 / E3 -- cobertura del IC95 del bootstrap por unidad (D4) por simulacion con verdad conocida.
Controles: (a) i.i.d. (m=1 fila por unidad, rho=0, sin deriva) cubre ~95 %; (b) la verdad analitica coincide con Monte Carlo.
Salida: salidas/e3.json y salidas/e3.txt"""
import sys, json, time
from lib140b import *
import simcob as S
T0 = time.time()
PARCIAL = len(sys.argv) > 1 and sys.argv[1] == "--humo"
RR = 40 if PARCIAL else 500          # replicas por celda
B = 200 if PARCIAL else 500          # remuestras
log = open(f"{SALIDAS}/e3{'_humo' if PARCIAL else ''}.txt", "w")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

def celda(n, m, rho, tau, cal, semilla, R=RR):
    rng = np.random.default_rng(semilla)
    cc = cm = 0; anchos = []; sesgo = []
    for _ in range(R):
        y, u, tc, tm = S.replica_rmse(n, m, rho, tau, cal, rng)
        pt, lo, hi = S.ic_mse(y, u, B, rng)
        cc += (lo <= tc <= hi); cm += (lo <= tm <= hi); anchos.append(hi - lo); sesgo.append(pt - tc)
    return dict(cov_cond=cc / R, cov_marg=cm / R, ancho=float(np.mean(anchos)), sesgo_cond=float(np.mean(sesgo)), R=R)

R_ = {"controles": {}, "rmse": [], "auroc": []}
p("== CONTROL (a): i.i.d. -- n unidades de 1 fila, rho=0, sin deriva; R=%d (error MC de la cobertura ~ %.1f puntos)" % (RR * 2, 100 * math.sqrt(.95 * .05 / (RR * 2))))
ok_a = True
for n in (100, 300):
    c = celda(n, 1, 0.0, 0.0, "panel", 1000 + n, R=RR * 2)
    dentro = 0.93 <= c["cov_cond"] <= 0.97
    ok_a &= dentro
    p(f"  n={n}: cobertura {c['cov_cond']:.3f} ancho {c['ancho']:.4f} -> {'DENTRO de [0.93, 0.97]' if dentro else 'FUERA'}")
    R_["controles"][f"iid_n{n}"] = dict(**c, ok=bool(dentro))

p("\n== CONTROL (b): la verdad analitica contra Monte Carlo (media de y^2 con 400.000 sujetos nuevos, d fijo)")
ok_b = True
for (rho, tau, cal) in ((0.3, 0.0, "panel"), (0.8, 1.0, "panel"), (0.3, 1.0, "esc")):
    rng = np.random.default_rng(77)
    d = S.deriva(tau, rng)
    k = 400000
    u, t, grid = S.visitas(1, 10, cal, rng)   # solo para la rejilla/forma
    ts = (rng.uniform(0, S.W, k) if cal != "panel" else rng.choice(grid, k))
    yy = rng.normal(0, math.sqrt(rho), k) + d(ts) + rng.normal(0, math.sqrt(1 - rho), k)
    mc = float(np.mean(yy ** 2)); an = 1.0 + float(np.mean(d(grid) ** 2))
    dentro = abs(mc - an) < 4 * float(np.std(yy ** 2)) / math.sqrt(k)
    ok_b &= dentro
    p(f"  rho={rho} tau={tau} {cal}: analitica {an:.4f} Monte Carlo {mc:.4f} -> {'IGUAL (4 EE)' if dentro else 'NO CUADRA'}")
R_["controles"]["verdad_ok"] = bool(ok_b); R_["controles"]["iid_ok"] = bool(ok_a)
p(f"\nCONTROLES E3: {'TODOS OK' if ok_a and ok_b else 'ALGUNO FALLA -- NO SACAR CONCLUSIONES'}")

NS = (8, 50) if PARCIAL else (5, 8, 20, 50, 200)
MS = (3, "d10") if PARCIAL else (3, 10, 50, "d10")
RHOS = (0.0, 0.8) if PARCIAL else (0.0, 0.3, 0.8)
DERIVAS = ((0.0, "panel"), (1.0, "panel")) if PARCIAL else ((0.0, "panel"), (0.0, "esc"), (0.5, "panel"), (0.5, "esc"), (1.0, "panel"), (1.0, "esc"))
p("\n== RMSE de un modelo fijo: cobertura del IC95 por unidad (R=%d, B=%d). cond = verdad de ESTA ventana; marg = promediando sobre la deriva" % (RR, B))
k = 0
for tau, cal in DERIVAS:
    for n in NS:
        for m in MS:
            for rho in RHOS:
                k += 1
                c = celda(n, m, rho, tau, cal, 5000 + k)
                R_["rmse"].append(dict(n=n, m=m, rho=rho, tau=tau, cal=cal, **c))
                p(f"  tau={tau} {cal:5s} n={n:3d} m={str(m):4s} rho={rho}: cobertura cond {c['cov_cond']:.3f} marg {c['cov_marg']:.3f} ancho(MSE) {c['ancho']:.3f}")
    p(f"  -- deriva tau={tau} {cal} hecha ({time.time()-T0:.0f} s)")

# ------------------------------------------------------------------ AUROC
RA = 40 if PARCIAL else 300
BA = 200 if PARCIAL else 300
p("\n== AUROC de un modelo fijo (puntuacion x, y ~ Bernoulli(sigmoide(-0.5+1.2x+u_i+d(t)))); m=10 visitas en rejilla comun; R=%d B=%d" % (RA, BA))
k = 0
for tau in (0.0, 1.0):
    for sd_u in (0.0, 1.5):
        for n in (8, 50):
            k += 1
            rng = np.random.default_rng(9000 + k)
            cond = []; res = []
            for _ in range(RA):
                x, y, tcond = S.replica_auc(n, 10, sd_u, tau, rng)
                pt, lo, hi, vf = S.ic_auc(x, y, BA, rng)
                cond.append(tcond); res.append((lo, hi, vf))
            tm = float(np.mean(cond))
            cc = float(np.mean([lo <= tc <= hi for (lo, hi, _), tc in zip(res, cond)]))
            cm = float(np.mean([lo <= tm <= hi for (lo, hi, _) in res]))
            R_["auroc"].append(dict(n=n, m=10, sd_u=sd_u, tau=tau, cov_cond=cc, cov_marg=cm, ancho=float(np.mean([hi - lo for lo, hi, _ in res])),
                                    verdad_marg=tm, sd_verdad_cond=float(np.std(cond)), R=RA))
            p(f"  tau={tau} sd_u={sd_u} n={n:3d}: cobertura cond {cc:.3f} marg {cm:.3f} ancho {R_['auroc'][-1]['ancho']:.3f} verdad marg {tm:.3f} (sd entre ventanas {np.std(cond):.4f})")
json.dump(R_, open(f"{SALIDAS}/e3{'_humo' if PARCIAL else ''}.json", "w"))
p(f"\nE3 terminado en {time.time()-T0:.0f} s")
