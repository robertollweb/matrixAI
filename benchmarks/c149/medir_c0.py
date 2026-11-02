#!/usr/bin/env python3
"""149-C0 — protocolo y tolerancia del banco de comportamiento, MEDIDO.

La regla de elección (ε, G, m, n y los criterios de la hipótesis de la red y de la paridad) está en
`REGLA_DE_ELECCION.md`, commiteada ANTES que este guion; aquí se implementa tal cual y el resultado graba el
sha256 de ese fichero. Lo que el guion calcula con el veredicto es `matrixai.estudio.comportamiento` (C1), no una
copia: `monotonia_desde_curvas` y `con_epsilon`.

    python3 benchmarks/c149/medir_c0.py --tam completo --salida benchmarks/c149/resultados
    python3 benchmarks/c149/medir_c0.py --tam minimo   --salida /ruta/de/prueba          # < 2 min

Partes: P1 tabla de falsa alarma / potencia con verdad conocida (+ sabotaje de etiqueta invertida), P2 n de la
sensibilidad (B), P3 hipótesis de la red (TabM), P4 paridad y tiempos con estudios reales rápidos (cáncer de mama y
`us_crime`). Escribe `c0_resultado.json` y `c0_tabla.md` en `--salida`.

ESTIMACIÓN (se afina con la medición del tamaño mínimo, ver la cabecera del trabajo encolado): ver `estimar` más abajo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
AQUI = Path(__file__).resolve().parent

EPSILONES = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
UMBRAL_FA, UMBRAL_POT, MIN_RETROCESO_VERDADERO = 0.10, 0.80, 0.10
FRACCIONES_B = (0.01, 0.05)

CONFIG = {
    "completo": dict(filas=2000, semillas=(0, 1, 2), motores=("lightgbm", "sklearn.hgb"), G=(15, 25, 50), m=(100, 200, 500),
                     ns=(100, 200, 500, 1000, 2000), epocas_red=30, reales=(("breast-cancer-wisconsin-diagnostic", "lightgbm,baseline"),
                                                                             ("us_crime", "lightgbm,baseline")),
                     filas_tiempo=200, columnas_tiempo=5, puntos_tiempo=25),
    "minimo": dict(filas=500, semillas=(0,), motores=("lightgbm",), G=(15,), m=(30, 80),
                   ns=(50, 100, 500), epocas_red=2, reales=(("breast-cancer-wisconsin-diagnostic", "lightgbm,baseline"),),
                   filas_tiempo=40, columnas_tiempo=2, puntos_tiempo=15),
}


def registrar(*partes):
    print(time.strftime("%H:%M:%S"), *partes, flush=True)


# ---------------------------------------------------------------------------
# La verdad conocida
# ---------------------------------------------------------------------------

COLUMNAS = tuple(f"x{k}" for k in range(1, 11))
# (tipo, direccion verdadera o None)
VERDAD_DE_LA_COLUMNA = {
    "x1": ("lineal", "sube"), "x2": ("saturacion", "sube"), "x3": ("lineal", "baja"), "x4": ("saturacion", "baja"),
    "x5": ("U", None), "x6": ("senoidal", None), "x7": ("plana", None), "x8": ("interaccion", None),
    "x9": ("interaccion", None), "x10": ("plana", None),
}
OFFSET_BINARIA = 0.8789      # E[F] bajo uniformes en [0, 10]: centra el logit


def efectos(v):
    """`F(x)` = suma de la función verdadera de cada columna (v = lista de 10 valores)."""
    x1, x2, x3, x4, x5, x6, _x7, x8, x9, _x10 = v
    return (0.2 * x1 + 2.0 * (1.0 - math.exp(-x2 / 2.0)) - 0.2 * x3 - 2.0 * (1.0 - math.exp(-x4 / 2.0))
            + 2.0 * ((x5 - 5.0) / 5.0) ** 2 + math.sin(2.0 * math.pi * 1.5 * x6 / 10.0)
            + 2.0 * ((x8 - 5.0) / 5.0) * ((x9 - 5.0) / 5.0))


def salida_verdadera(tarea, v):
    f = efectos(v)
    return f if tarea == "regression" else 1.0 / (1.0 + math.exp(-1.2 * (f - OFFSET_BINARIA)))


def generar(tarea, n, semilla):
    azar = random.Random(f"c149-c0:{tarea}:{semilla}")
    filas = []
    for i in range(n):
        v = [azar.uniform(0.0, 10.0) for _ in COLUMNAS]
        f = efectos(v)
        if tarea == "regression":
            objetivo = f + azar.gauss(0.0, 0.3)
        else:
            p = 1.0 / (1.0 + math.exp(-1.2 * (f - OFFSET_BINARIA)))
            objetivo = "Sí" if azar.random() < p else "No"
        filas.append({**{c: x for c, x in zip(COLUMNAS, v)}, "y": objetivo, "__row_id__": str(i)})
    return filas


# ---------------------------------------------------------------------------
# El modelo del producto
# ---------------------------------------------------------------------------

class Modelo:
    def __init__(self, motor_id, tarea, semilla, crudas, *, epocas=None, tabm=False):
        from matrixai.estudio import ProblemSpec
        from matrixai_engines.particiones import Presupuesto
        from matrixai_engines.registro_de_motores import motor_para
        from matrixai_studio import estudio_job as ej
        self.ej, self.tarea, self.motor_id, self.crudas = ej, tarea, motor_id, crudas
        self.motor = motor_para(motor_id)
        if epocas is not None:
            self.motor = self.motor.con_epocas_fijas(epocas)
        n = len(crudas)
        a, b = int(n * 0.70), int(n * 0.85)
        self.entrenaron = crudas[:a]
        t0 = time.perf_counter()
        self.politica, train, val, _med = ej._particiones_preparadas(
            crudas[:a], crudas[a:b], crudas[b:], objetivo="y", predictores=COLUMNAS, capacidades=self.motor.capabilities(),
            columna_texto=None)
        clases = None if tarea == "regression" else ("No", "Sí")
        spec = ProblemSpec(problem_id="c0", target="y", task=tarea, observation_unit="fila", classes=clases,
                           positive_label=None if tarea == "regression" else "Sí",
                           predictors=tuple(self.politica.columnas_de_salida()))
        spec = ej._spec_para_el_motor(spec, self.politica)
        fr, self.fitted = self.motor.fit(train, val, spec, Presupuesto(wall_seconds=600, hilos=2, seed=semilla),
                                         candidate=f"c0-{motor_id}-{tarea}-{semilla}", split_plan_digest="d" * 64)
        if self.fitted is None:
            raise RuntimeError(f"no ajustó: {fr.state} {str(fr.reason)[:200]}")
        self.t_ajuste = time.perf_counter() - t0
        self.prep = ej._transformadas(crudas, "y", self.politica)
        if tarea == "regression":
            r = ej.reparto_del_objetivo_en_el_entrenamiento(self.entrenaron, "y")
            self.escala_valor = r["p90"] - r["p10"]
        else:
            self.escala_valor = 1.0

    def _predecir(self, filas_prep):
        from matrixai_engines.particiones import Particion
        d = Particion.desde_filas(filas_prep, row_id_field="__row_id__", target_field="y")
        if self.tarea == "regression":
            return [float(v) for v in self.motor.predict(self.fitted, d)]
        return [float(p[-1]) for p in self.motor.predict_proba(self.fitted, d)]

    def puntuar_crudas(self, filas):
        """El camino del producto: perturbar la fila CRUDA y preparar."""
        crudas = [{**f, "__row_id__": f"{k}", "y": f.get("y", "")} for k, f in enumerate(filas)]
        return self._predecir(self.ej._transformadas(crudas, "y", self.politica))

    def ice_preparado(self, indices, columna, malla):
        """«Preparar una vez y perturbar la fila ya preparada»: valor en la columna e indicador `__faltante` a 0."""
        from matrixai.training.preparacion import nombre_de_indicador
        ind = nombre_de_indicador(columna)
        lote = []
        for i in indices:
            base = self.prep[i]
            for k, x in enumerate(malla):
                lote.append({**base, columna: x, ind: 0.0, "__row_id__": f"{i}#{k}"})
        s = self._predecir(lote)
        G = len(malla)
        return [s[j * G:(j + 1) * G] for j in range(len(indices))]

    def salidas_preparadas(self, indices, columna=None, valores=None):
        from matrixai.training.preparacion import nombre_de_indicador
        lote = []
        for j, i in enumerate(indices):
            fila = dict(self.prep[i])
            if columna is not None:
                fila[columna] = valores[j]
                fila[nombre_de_indicador(columna)] = 0.0
            fila["__row_id__"] = f"{i}"
            lote.append(fila)
        return self._predecir(lote)

    def escala(self):
        from matrixai.estudio.comportamiento import escala_de_la_salida
        return escala_de_la_salida("regression", amplitud_p90_p10=self.escala_valor) if self.tarea == "regression" \
            else escala_de_la_salida("binary_classification")


# ---------------------------------------------------------------------------
# P1 — la tabla de falsa alarma y potencia
# ---------------------------------------------------------------------------

def rangos(crudas):
    return {c: (min(f[c] for f in crudas), max(f[c] for f in crudas)) for c in COLUMNAS}


def casos_de_verdad(tarea, crudas, indices, malla_por_g, escala):
    """Para cada (columna, dirección): el retroceso VERDADERO máximo (sobre las filas base) / escala, con G = el mayor."""
    from matrixai.estudio.comportamiento import retroceso_de_una_curva
    G = max(malla_por_g)
    malla = {c: malla_por_g[G][c] for c in COLUMNAS}
    casos = {}
    for k, c in enumerate(COLUMNAS):
        for direccion in ("sube", "baja"):
            peor = 0.0
            for i in indices:
                v = [crudas[i][cc] for cc in COLUMNAS]
                curva = []
                for x in malla[c]:
                    w = list(v); w[k] = x
                    curva.append(salida_verdadera(tarea, w))
                peor = max(peor, retroceso_de_una_curva(curva, direccion).valor)
            rel = peor / escala
            tipo = "monotono" if rel < 1e-9 else ("no_monotono" if rel >= MIN_RETROCESO_VERDADERO else "intermedio")
            casos[(c, direccion)] = {"retroceso_verdadero_rel": rel, "tipo": tipo}
    return casos


def evaluar_modelo(modelo, semilla, cfg, malla_de, resultado_celdas, detalle, grupo, verificar_modulo):
    """Acumula, por celda (G, m, ε), marcados y total de casos monótonos / no monótonos de este modelo."""
    from matrixai.estudio.comportamiento import elegir_filas, monotonia_desde_curvas
    base_m = max(cfg["m"])
    indices = elegir_filas(len(modelo.crudas), base_m, semilla)
    escala = modelo.escala()
    mallas_g = {G: {c: malla_de(modelo.crudas, c, G) for c in COLUMNAS} for G in cfg["G"]}
    casos = casos_de_verdad(modelo.tarea, modelo.crudas, indices, mallas_g, escala.valor)
    detalle["casos_de_verdad"][grupo + f"|s{semilla}"] = {f"{c}|{d}": v for (c, d), v in casos.items()}
    sub = {m: sorted(random.Random(f"c0:sub:{semilla}:{m}").sample(range(len(indices)), min(m, len(indices)))) for m in cfg["m"]}
    t0 = time.perf_counter()
    for G in cfg["G"]:
        for c in COLUMNAS:
            malla = mallas_g[G][c]
            curvas = modelo.ice_preparado(indices, c, malla)
            if verificar_modulo and c == "x1" and G == cfg["G"][0]:
                _verificar_contra_el_modulo(modelo, semilla, c, malla, indices, curvas, escala, detalle, grupo)
            for direccion in ("sube", "baja"):
                tipo = casos[(c, direccion)]["tipo"]
                for m in cfg["m"]:
                    pos = sub[m]
                    r0 = monotonia_desde_curvas(c, direccion, malla, [curvas[p] for p in pos], [indices[p] for p in pos],
                                                epsilon_relativo=0.0, escala=escala, medido_sobre="muestra_de_desarrollo", semilla=semilla)
                    retro = [x for x in r0.retrocesos if x is not None]
                    for eps in EPSILONES:
                        v = r0.con_epsilon(eps).veredicto
                        celda = resultado_celdas.setdefault((grupo, G, m, eps), {"monotono": [0, 0, 0], "no_monotono": [0, 0, 0]})
                        if tipo in celda:
                            celda[tipo][0] += 1
                            celda[tipo][1] += v == "con_contraejemplos"
                            celda[tipo][2] += v == "no_se_pudo_comprobar"
                        detalle["verdictos"].setdefault(f"{grupo}|s{semilla}|{c}|{direccion}|G{G}|m{m}", {})[str(eps)] = v[:3]
                    if G == cfg["G"][min(1, len(cfg["G"]) - 1)] and m == cfg["m"][min(1, len(cfg["m"]) - 1)]:
                        detalle["retroceso"][f"{grupo}|s{semilla}|{c}|{direccion}"] = {
                            "mediano_rel": statistics.median(retro) / escala.valor if retro else None,
                            "maximo_rel": max(retro) / escala.valor if retro else None,
                            "efecto_rel": (r0.efecto_total / escala.valor) if r0.efecto_total is not None else None,
                            # EXPLORATORIO (no entra en la regla): qué fracción de las filas base retrocede más que cada ε
                            "fraccion_de_filas_violadoras": {str(e): (sum(1 for x in retro if x > e * escala.valor) / len(retro)) if retro else None
                                                             for e in EPSILONES}}
    detalle["segundos_de_ice"][f"{grupo}|s{semilla}"] = round(time.perf_counter() - t0, 1)


def _verificar_contra_el_modulo(modelo, semilla, c, malla, indices, curvas, escala, detalle, grupo):
    """`monotonia_en_rejilla` (camino de producción: fila cruda -> preparar) debe dar EL MISMO documento que las curvas
    calculadas preparando una vez. Es la paridad medida sobre el modelo sintético y la prueba de que el guion usa el módulo."""
    from matrixai.estudio.comportamiento import monotonia_desde_curvas, monotonia_en_rejilla
    a = monotonia_en_rejilla(c, "sube", modelo.crudas, modelo.puntuar_crudas, minimo=malla[0], maximo=malla[-1], puntos=len(malla),
                             filas_m=len(indices), semilla=semilla, epsilon_relativo=0.01, escala=escala,
                             medido_sobre="muestra_de_desarrollo")
    b = monotonia_desde_curvas(c, "sube", malla, curvas, list(indices), epsilon_relativo=0.01, escala=escala,
                               medido_sobre="muestra_de_desarrollo", semilla=semilla)
    iguales_curvas = [x for x in a.retrocesos] == [x for x in b.retrocesos]
    detalle["paridad_sintetica"][f"{grupo}|s{semilla}|{c}"] = {"mismo_digest": a.digest() == b.digest(), "mismos_retrocesos": iguales_curvas}


def tasas(celdas, grupos, G, m, eps, invertida=False):
    """FA y POT por grupo (juntando semillas). Con `invertida`, los papeles de las dos etiquetas se intercambian."""
    por_grupo = {}
    for g in grupos:
        c = celdas.get((g, G, m, eps))
        if c is None:
            return None
        mon, nomon = (c["no_monotono"], c["monotono"]) if invertida else (c["monotono"], c["no_monotono"])
        fa = (mon[1] / mon[0]) if mon[0] else None
        pot = (nomon[1] / nomon[0]) if nomon[0] else None
        por_grupo[g] = {"FA": fa, "POT": pot, "n_mon": mon[0], "n_nomon": nomon[0],
                        "sin_distinguir_mon": mon[2], "sin_distinguir_nomon": nomon[2]}
    return por_grupo


def cumple(por_grupo):
    return por_grupo is not None and all(
        t["FA"] is not None and t["POT"] is not None and t["FA"] <= UMBRAL_FA and t["POT"] >= UMBRAL_POT for t in por_grupo.values())


def aplicar_regla(celdas, grupos, cfg, invertida=False):
    """La regla de REGLA_DE_ELECCION.md: ε*(G,m), orden por G×m, estabilidad en G'>=G y m'>=m."""
    eps_estrella = {}
    for G in cfg["G"]:
        for m in cfg["m"]:
            eps_estrella[(G, m)] = next((e for e in EPSILONES if cumple(tasas(celdas, grupos, G, m, e, invertida))), None)
    orden = sorted(eps_estrella, key=lambda gm: (gm[0] * gm[1], gm[0]))
    for G, m in orden:
        e = eps_estrella[(G, m)]
        if e is None:
            continue
        mayores = [(g2, m2) for g2 in cfg["G"] for m2 in cfg["m"] if g2 >= G and m2 >= m]
        if all(cumple(tasas(celdas, grupos, g2, m2, e, invertida)) for g2, m2 in mayores):
            return {"hay_epsilon": True, "G": G, "m": m, "epsilon": e, "epsilon_estrella_por_celda": {f"{k[0]}x{k[1]}": v for k, v in eps_estrella.items()}}
    return {"hay_epsilon": False, "G": None, "m": None, "epsilon": None, "epsilon_estrella_por_celda": {f"{k[0]}x{k[1]}": v for k, v in eps_estrella.items()}}


# ---------------------------------------------------------------------------
# P2 — n de la sensibilidad
# ---------------------------------------------------------------------------

def parte_n_de_b(modelos_binarios, cfg):
    from matrixai.estudio.comportamiento import elegir_filas, sensibilidad_de_una_columna
    desviaciones = {n: [] for n in cfg["ns"]}
    n_max = max(cfg["ns"])
    equivalencia = None
    for (semilla, modelo) in modelos_binarios:
        r = rangos(modelo.crudas)
        N = len(modelo.crudas)
        todos = list(range(N))
        for c in COLUMNAS:
            lo, hi = r[c]
            base = modelo.salidas_preparadas(todos)
            for frac in FRACCIONES_B:
                delta = frac * (hi - lo)
                for signo in (1.0, -1.0):
                    valores = [modelo.crudas[i][c] + signo * delta for i in todos]
                    mov = modelo.salidas_preparadas(todos, c, valores)
                    cambia = [(a >= 0.5) != (b >= 0.5) for a, b in zip(base, mov)]
                    pleno = sum(cambia[i] for i in elegir_filas(N, n_max, semilla)) / len(elegir_filas(N, n_max, semilla))
                    for n in cfg["ns"]:
                        idx = elegir_filas(N, n, semilla)
                        desviaciones[n].append(abs(sum(cambia[i] for i in idx) / len(idx) - pleno))
                    if equivalencia is None and c == "x1" and frac == FRACCIONES_B[-1] and signo > 0:
                        sens = sensibilidad_de_una_columna(c, modelo.crudas, modelo.puntuar_crudas, minimo=lo, maximo=hi, semilla=semilla,
                                                           filas_n=n_max, medido_sobre="muestra_de_desarrollo", fracciones=(frac,),
                                                           decidir=lambda y: y >= 0.5)
                        modulo = sens.pruebas[0]["cambian_de_decision"]
                        idx = elegir_filas(N, n_max, semilla)
                        equivalencia = {"modulo": modulo, "guion": sum(cambia[i] for i in idx), "coinciden": modulo == sum(cambia[i] for i in idx)}
    def p95(x):
        x = sorted(x)
        return x[min(len(x) - 1, math.ceil(0.95 * len(x)) - 1)] if x else None
    tabla = {n: {"p95_de_la_desviacion": p95(v), "maxima": max(v) if v else None, "unidades": len(v)} for n, v in desviaciones.items()}
    elegido = next((n for n in cfg["ns"] if tabla[n]["p95_de_la_desviacion"] is not None and tabla[n]["p95_de_la_desviacion"] <= 0.02), None)
    return {"tabla": {str(k): v for k, v in tabla.items()}, "n_por_omision": elegido if elegido is not None else max(cfg["ns"]),
            "ninguno_cumple_y_se_queda_el_mayor": elegido is None, "equivalencia_con_el_modulo": equivalencia}


# ---------------------------------------------------------------------------
# P3 — la hipótesis de la red
# ---------------------------------------------------------------------------

def retroceso_en_monotonos_con_efecto(modelo, semilla, cfg):
    from matrixai.estudio.comportamiento import elegir_filas, monotonia_desde_curvas, rejilla
    G, m = (25 if 25 in cfg["G"] else cfg["G"][0]), (200 if 200 in cfg["m"] else cfg["m"][-1])
    idx = elegir_filas(len(modelo.crudas), m, semilla)
    r = rangos(modelo.crudas)
    medianas = {}
    for c in ("x1", "x2", "x3", "x4"):
        direccion = VERDAD_DE_LA_COLUMNA[c][1]
        malla = rejilla(r[c][0], r[c][1], G)
        res = monotonia_desde_curvas(c, direccion, malla, modelo.ice_preparado(idx, c, malla), list(idx), epsilon_relativo=0.0,
                                     escala=modelo.escala(), medido_sobre="muestra_de_desarrollo", semilla=semilla)
        retro = [x for x in res.retrocesos if x is not None]
        medianas[c] = statistics.median(retro) / modelo.escala_valor
    return {"G": G, "m": m, "mediana_por_columna": medianas, "mediana_de_las_cuatro": statistics.median(medianas.values())}


# ---------------------------------------------------------------------------
# P4 — paridad y tiempos con estudios reales rápidos
# ---------------------------------------------------------------------------

def parte_real(nombre, motores, cfg, directorio):
    import copy  # noqa: F401
    from matrixai.training.preparacion import nombre_de_indicador
    from matrixai_engines.particiones import Particion
    from matrixai_studio import columnas_que_pesan as cqp
    from matrixai_studio import estudio_job as ej
    from matrixai_studio.estudio_job import estudio_status, iniciar_estudio
    D = Path(ej.__file__).parent / "ejemplos_medidos"
    cat = {c["nombre"]: c for c in json.load(open(D / "catalogo.json"))["conjuntos"]}[nombre]
    payload = {"csv_text": (D / cat["csv"]).read_text(encoding="utf-8"), "objetivo": cat["objetivo"], "tarea": cat["tarea"],
               "unidad_de_observacion": cat.get("unidad_de_observacion") or "una fila", "folds": 3, "repeats": 1, "modo": "rapido"}
    if cat.get("clase_positiva") is not None:
        payload["clase_positiva"] = str(cat["clase_positiva"])
    ej._MOTORES_PERMITIDOS = tuple(motores.split(","))
    t0 = time.perf_counter()
    jid = iniciar_estudio(payload, owner="s")["estudio"]["job_id"]
    while estudio_status(jid, "s")["estado"] not in ("completed", "failed", "cancelled"):
        time.sleep(0.5)
    out = {"ejemplo": nombre, "motores": motores, "estudio_s": round(time.perf_counter() - t0, 1)}
    contexto, err = ej._pipeline_para_usar(jid, "s")
    if err is not None:
        out["error"] = str(err)
        return out
    motor, fitted, politica = contexto["motor"], contexto["fitted"], contexto["politica"]
    ctx, _ = cqp._contexto(jid, "s")
    datos, motivo_ = cqp._cargar_los_pliegues(ctx["directorio"])
    if not datos:
        out["error"] = f"sin pliegues: {motivo_}"
        return out
    objetivo, tarea = datos["objetivo"], datos["tarea"]
    por_id = datos["filas"]
    ids = sorted(por_id)
    crudas = [{**por_id[i], "__row_id__": i} for i in ids]
    out.update(tarea=tarea, motor=ctx["seleccion"]["candidate_engine"], filas_dev=len(crudas), predictores=len(datos["predictores"]))

    def predecir(filas_prep):
        d = Particion.desde_filas(filas_prep, row_id_field="__row_id__", target_field=objetivo)
        if tarea == "regression":
            return [float(v) for v in motor.predict(fitted, d)]
        return [float(p[-1]) for p in motor.predict_proba(fitted, d)]

    numericas = [p for p in politica.columnas if p.tipo == "numerica" and p.minimo is not None and p.maximo is not None and p.maximo > p.minimo]
    azar = random.Random(1)
    base_idx = sorted(azar.sample(range(len(crudas)), min(cfg["filas_tiempo"], len(crudas))))
    G = cfg["puntos_tiempo"]
    prep_base = ej._transformadas([crudas[i] for i in base_idx], objetivo, politica)

    def camino_crudo(p, malla, medir):
        t = time.perf_counter()
        lote = [{**crudas[i], p.columna: x, "__row_id__": f"{i}#{k}"} for i in base_idx for k, x in enumerate(malla)]
        t_lote = time.perf_counter() - t
        t = time.perf_counter(); prep = ej._transformadas(lote, objetivo, politica); t_prep = time.perf_counter() - t
        t = time.perf_counter(); d = Particion.desde_filas(prep, row_id_field="__row_id__", target_field=objetivo); t_part = time.perf_counter() - t
        t = time.perf_counter()
        s = [float(v) for v in motor.predict(fitted, d)] if tarea == "regression" else [float(q[-1]) for q in motor.predict_proba(fitted, d)]
        t_pred = time.perf_counter() - t
        medir.append((len(lote), t_lote, t_prep, t_part, t_pred))
        return s

    def camino_preparado(p, malla, medir):
        ind = nombre_de_indicador(p.columna)
        t = time.perf_counter()
        lote = [{**prep_base[j], p.columna: x, ind: 0.0, "__row_id__": f"{base_idx[j]}#{k}"} for j in range(len(base_idx)) for k, x in enumerate(malla)]
        t_lote = time.perf_counter() - t
        t = time.perf_counter(); d = Particion.desde_filas(lote, row_id_field="__row_id__", target_field=objetivo); t_part = time.perf_counter() - t
        t = time.perf_counter()
        s = [float(v) for v in motor.predict(fitted, d)] if tarea == "regression" else [float(q[-1]) for q in motor.predict_proba(fitted, d)]
        t_pred = time.perf_counter() - t
        medir.append((len(lote), t_lote, 0.0, t_part, t_pred))
        return s

    paridad, medidas_crudo, medidas_prep = [], [], []
    for p in numericas[:cfg["columnas_tiempo"]]:
        malla = [p.minimo + (p.maximo - p.minimo) * k / (G - 1) for k in range(G)]
        malla[-1] = p.maximo
        a, b = camino_crudo(p, malla, medidas_crudo), camino_preparado(p, malla, medidas_prep)
        paridad.append({"columna": p.columna, "salidas": len(a), "identicas": a == b,
                        "diferencia_maxima": max((abs(x - y) for x, y in zip(a, b)), default=0.0)})

    def ms(medidas, k):
        total = sum(m[0] for m in medidas)
        return 1000.0 * sum(m[k] for m in medidas) / total if total else None
    out["paridad_crudo_frente_a_preparado_una_vez"] = {"columnas": paridad, "todas_identicas": all(x["identicas"] for x in paridad)}
    out["ms_por_fila_perturbada"] = {
        "crudo": {"montar": ms(medidas_crudo, 1), "preparar": ms(medidas_crudo, 2), "particion": ms(medidas_crudo, 3), "predecir": ms(medidas_crudo, 4),
                  "total": sum(ms(medidas_crudo, k) for k in (1, 2, 3, 4))},
        "preparado_una_vez": {"montar": ms(medidas_prep, 1), "particion": ms(medidas_prep, 3), "predecir": ms(medidas_prep, 4),
                              "total": sum(ms(medidas_prep, k) for k in (1, 3, 4))},
        "filas_perturbadas_por_camino": sum(m[0] for m in medidas_crudo), "columnas": len(paridad), "puntos": G, "filas_base": len(base_idx)}
    a, b = out["ms_por_fila_perturbada"]["crudo"]["total"], out["ms_por_fila_perturbada"]["preparado_una_vez"]["total"]
    out["ahorro_de_preparar_una_vez"] = {"razon_crudo_entre_preparado": (a / b) if b else None}
    if tarea == "regression":
        obj = ej._objetivo_para_usar(contexto)
        out["escala_de_regresion"] = {
            "objetivo_del_entrenamiento_del_sobre": obj,
            "p90_menos_p10_del_sobre": (obj["p90"] - obj["p10"]) if obj else None,
            "p90_menos_p10_de_todo_el_desarrollo": (lambda v: v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))])(
                sorted(float(f[objetivo]) for f in crudas))}
    return out


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def tabla_markdown(celdas, grupos, cfg):
    lineas = ["# 149-C0 — tablas por celda (FA = falsa alarma, POT = potencia; 4 grupos = motor × tarea; semillas juntas)", ""]
    for G in cfg["G"]:
        for m in cfg["m"]:
            lineas += [f"## G = {G}, m = {m}", "", "| ε | " + " | ".join(f"{g} FA / POT" for g in grupos) + " | cumple |", "|---|" + "---|" * (len(grupos) + 1)]
            for e in EPSILONES:
                t = tasas(celdas, grupos, G, m, e)
                f = lambda x: "—" if x is None else f"{100 * x:.0f} %"
                lineas.append(f"| {100 * e:g} % | " + " | ".join(f"{f(t[g]['FA'])} / {f(t[g]['POT'])}" for g in grupos) + f" | {'sí' if cumple(t) else 'no'} |")
            lineas.append("")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tam", choices=sorted(CONFIG), default="completo")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--sin-reales", action="store_true", help="omitir P4 (estudios reales)")
    args = ap.parse_args()
    cfg = CONFIG[args.tam]
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    # Los estudios reales rápidos van FUERA del worktree: la cola copia al resultado todo lo que el trabajo deja sin
    # commitear, y los checkpoints de un estudio no son resultado.
    estudios_tmp = tempfile.mkdtemp(prefix="c149-c0-estudios-")
    os.environ["MATRIXAI_ESTUDIOS_DIR"] = estudios_tmp
    t_inicio = time.perf_counter()
    regla = AQUI / "REGLA_DE_ELECCION.md"
    resultado = {"tam": args.tam, "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"), "regla_sha256": hashlib.sha256(regla.read_bytes()).hexdigest(),
                 "config": {k: v for k, v in cfg.items() if k != "reales"}, "codigo": {}}
    import matrixai, matrixai_engines, matrixai_studio
    resultado["codigo"] = {"matrixai": matrixai.__file__, "matrixai_engines": matrixai_engines.__file__, "matrixai_studio": matrixai_studio.__file__}
    registrar("código:", resultado["codigo"])
    from matrixai.estudio.comportamiento import rejilla

    def malla_de(crudas, columna, G):
        lo = min(f[columna] for f in crudas)
        hi = max(f[columna] for f in crudas)
        return rejilla(lo, hi, G)

    # ---- P1 ----
    celdas, detalle = {}, {"casos_de_verdad": {}, "verdictos": {}, "retroceso": {}, "segundos_de_ice": {}, "paridad_sintetica": {}, "ajuste_s": {}}
    grupos, binarios, modelos_para_red = [], [], {}
    for tarea in ("regression", "binary_classification"):
        for motor_id in cfg["motores"]:
            grupo = f"{motor_id}|{'reg' if tarea == 'regression' else 'bin'}"
            grupos.append(grupo)
            for semilla in cfg["semillas"]:
                crudas = generar(tarea, cfg["filas"], semilla)
                modelo = Modelo(motor_id, tarea, semilla, crudas)
                detalle["ajuste_s"][f"{grupo}|s{semilla}"] = round(modelo.t_ajuste, 1)
                registrar("P1", grupo, "semilla", semilla, "ajustado en", round(modelo.t_ajuste, 1), "s")
                evaluar_modelo(modelo, semilla, cfg, malla_de, celdas, detalle, grupo, verificar_modulo=True)
                registrar("P1", grupo, "semilla", semilla, "evaluado en", detalle["segundos_de_ice"][f"{grupo}|s{semilla}"], "s")
                if tarea == "binary_classification":
                    binarios.append((semilla, modelo))
                if tarea == "regression" and motor_id == "lightgbm" and semilla == cfg["semillas"][0]:
                    modelos_para_red["lightgbm"] = (semilla, modelo)
    resultado["P1"] = {
        "grupos": grupos,
        "celdas": {f"{g}|G{G}|m{m}|e{e}": v for (g, G, m, e), v in sorted(celdas.items())},
        "regla_aplicada": aplicar_regla(celdas, grupos, cfg),
        "sabotaje_etiqueta_invertida": aplicar_regla(celdas, grupos, cfg, invertida=True),
        "tablas_por_celda": {f"G{G}|m{m}|e{e}": tasas(celdas, grupos, G, m, e) for G in cfg["G"] for m in cfg["m"] for e in EPSILONES},
        "detalle": detalle}
    real, falso = resultado["P1"]["regla_aplicada"], resultado["P1"]["sabotaje_etiqueta_invertida"]
    resultado["P1"]["el_instrumento_distingue"] = (real["epsilon"], real["G"], real["m"]) != (falso["epsilon"], falso["G"], falso["m"])
    registrar("P1 regla:", real, "| invertida:", {k: falso[k] for k in ("hay_epsilon", "epsilon", "G", "m")})
    (salida / "c0_tabla.md").write_text(tabla_markdown(celdas, grupos, cfg), encoding="utf-8")

    # ---- P2 ----
    resultado["P2_n_de_B"] = parte_n_de_b(binarios, cfg)
    registrar("P2 n por omisión:", resultado["P2_n_de_B"]["n_por_omision"])

    # ---- P3 ----
    semilla, modelo_lgbm = modelos_para_red["lightgbm"]
    t0 = time.perf_counter()
    red = Modelo("matrixai.dense.tabm_cpu", "regression", semilla, modelo_lgbm.crudas, epocas=cfg["epocas_red"])
    a = retroceso_en_monotonos_con_efecto(modelo_lgbm, semilla, cfg)
    b = retroceso_en_monotonos_con_efecto(red, semilla, cfg)
    razon = (b["mediana_de_las_cuatro"] / a["mediana_de_las_cuatro"]) if a["mediana_de_las_cuatro"] else None
    veredicto = ("no calculable (el árbol no retrocede)" if razon is None else
                 "aceptada (< 50 %)" if razon < 0.5 else "rechazada (>= 100 %)" if razon >= 1.0 else "no concluyente (entre 50 % y 100 %)")
    resultado["P3_red"] = {"arbol": a, "red": b, "razon_red_entre_arbol": razon, "hipotesis": veredicto, "epocas": cfg["epocas_red"],
                           "ajuste_red_s": round(red.t_ajuste, 1), "alcance": "UNA red (tabm_cpu) en UN conjunto sintético de regresión, UNA semilla"}
    registrar("P3 red/arbol:", razon, veredicto, "en", round(time.perf_counter() - t0, 1), "s")

    # ---- P4 ----
    resultado["P4_estudios_reales"] = []
    if not args.sin_reales:
        for nombre, motores in cfg["reales"]:
            registrar("P4", nombre)
            resultado["P4_estudios_reales"].append(parte_real(nombre, motores, cfg, salida))
            registrar("P4", nombre, "->", {k: resultado["P4_estudios_reales"][-1].get(k) for k in ("estudio_s", "ahorro_de_preparar_una_vez")})

    resultado["segundos_totales"] = round(time.perf_counter() - t_inicio, 1)
    (salida / "c0_resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    shutil.rmtree(estudios_tmp, ignore_errors=True)
    registrar("FIN", resultado["segundos_totales"], "s ->", salida)


if __name__ == "__main__":
    main()
