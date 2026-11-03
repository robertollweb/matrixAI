#!/usr/bin/env python3
"""149-C0-ter — veredicto V'(ε, q) por AMPLITUD y fracción de filas de la monotonía declarada, MEDIDO (la última ronda).

La regla está en `REGLA_DE_ELECCION_TER.md`, commiteada ANTES que este guion (8010b3f); aquí se implementa tal cual y el
resultado graba su sha256. Los datos sintéticos, los motores, la preparación, la escala, la verdad de cada caso y los
retrocesos por fila (C1) son los de `medir_c0.py` / `medir_c0bis.py`, importados: no se copia nada. Lo único nuevo es el
veredicto V'(ε, q) —la amplitud sin signo en vez del efecto con signo de extremo a extremo— y la potencia separada.

LAS CURVAS POR FILA: C1 (`monotonia_desde_curvas`) NO conserva las curvas, solo sus retrocesos y efectos. Se recomputan aquí
con el MISMO camino que el C0-bis (`modelo.ice_preparado(indices, columna, malla)`, mismas filas `elegir_filas`, misma
malla); la amplitud sale de ellas y los retrocesos, del propio C1 con `epsilon_relativo=0`, como en el C0-bis.

    cd /home/deployer/worktrees/c149/matrixAI
    PYTHONPATH=<núcleo>:<motores>/src:<studio-backend> python3 benchmarks/c149/medir_c0ter.py [--salida DIR]

Escribe `c0ter_resultado.json` y `c0ter_tabla.md` en `--salida` (por omisión `benchmarks/c149/resultados`).
ESTIMACIÓN: ~30 s (el C0-bis, con las mismas dimensiones, tardó 20 s).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# El guion del C0-bis (que a su vez carga el del C0), tal cual; misma clave que usa su prueba.
if "medir_c0bis_149" in sys.modules:
    bis = sys.modules["medir_c0bis_149"]
else:
    _ESPEC = importlib.util.spec_from_file_location("medir_c0bis_149", AQUI / "medir_c0bis.py")
    bis = importlib.util.module_from_spec(_ESPEC)
    sys.modules["medir_c0bis_149"] = bis
    _ESPEC.loader.exec_module(bis)
c0 = bis.c0

SEMILLAS = (6, 7, 8)
G, M = bis.G, bis.M
MOTORES, FILAS = bis.MOTORES, bis.FILAS
EPS_PCT, Q_PCT = bis.EPS_PCT, bis.Q_PCT
TAREAS, PLANAS = bis.TAREAS, bis.PLANAS
UMBRAL_FA, UMBRAL_POT, MIN_PLANAS = bis.UMBRAL_FA, bis.UMBRAL_POT, bis.MIN_PLANAS
VEREDICTOS = bis.VEREDICTOS
registrar = c0.registrar
COLUMNAS_CONTRARIAS = ("x1", "x2", "x3", "x4")          # monótonas: el dato es la dirección CONTRARIA a la verdadera
COLUMNAS_NO_MONOTONAS = ("x5", "x6", "x8", "x9")


# ---------------------------------------------------------------------------
# El veredicto V'(ε, q) — funciones puras
# ---------------------------------------------------------------------------

def amplitud_de_curvas(curvas):
    """MEDIANA, sobre las filas con curva completa, de (máximo − mínimo) de la curva de cada fila. Sin signo y sin depender de
    los extremos (una U que acaba donde empezó tiene amplitud grande; la de extremo a extremo sería 0). Sin filas -> None."""
    amplitudes = [max(c) - min(c) for c in curvas if c is not None and all(v is not None for v in c)]
    return statistics.median(amplitudes) if amplitudes else None


def veredicto_ter(retrocesos, amplitud, epsilon_absoluto, q_pct):
    """V'(ε, q) de la regla, sobre los retrocesos por fila (`None` = fila sin predicción, no cuenta).

    1. AMPLITUD <= ε  ->  `no_se_pudo_comprobar` (se mira PRIMERO).
    2. Fracción de filas con retroceso > ε  >=  q  ->  `con_contraejemplos`.
    3. Si no  ->  `sin_contraejemplos`.
    `q_pct` es un entero en %: comparación exacta (violadoras·100 >= q·comprobadas), sin coma flotante."""
    comprobadas = [r for r in retrocesos if r is not None]
    if epsilon_absoluto is None or not comprobadas:
        return "no_se_pudo_comprobar"
    if amplitud is None or not amplitud > epsilon_absoluto:
        return "no_se_pudo_comprobar"
    violadoras = sum(1 for r in comprobadas if r > epsilon_absoluto)
    if violadoras * 100 >= q_pct * len(comprobadas):
        return "con_contraejemplos"
    return "sin_contraejemplos"


# ---------------------------------------------------------------------------
# Acumulación por grupo y celda
# ---------------------------------------------------------------------------

def _celda_vacia():
    return {**bis._celda_vacia(), "contraria": bis._vacio(), "no_monotona_x5689": bis._vacio()}


def evaluar_modelo_ter(modelo, semilla, grupo, acum, detalle):
    """Un modelo (grupo, semilla): sus casos de verdad y, por (ε, q), el veredicto V' de cada (columna, dirección)."""
    from matrixai.estudio.comportamiento import elegir_filas, monotonia_desde_curvas
    indices = elegir_filas(len(modelo.crudas), M, semilla)
    escala = modelo.escala()
    mallas = {c: bis._malla(modelo.crudas, c) for c in c0.COLUMNAS}
    casos = c0.casos_de_verdad(modelo.tarea, modelo.crudas, indices, {G: mallas}, escala.valor)
    t0 = time.perf_counter()
    for c in c0.COLUMNAS:
        curvas = modelo.ice_preparado(indices, c, mallas[c])
        amplitud = amplitud_de_curvas(curvas)                    # no depende de la dirección
        for direccion in ("sube", "baja"):
            tipo = casos[(c, direccion)]["tipo"]
            if c in PLANAS:
                tipo = "plana"
            r0 = monotonia_desde_curvas(c, direccion, mallas[c], curvas, list(indices), epsilon_relativo=0.0, escala=escala,
                                        medido_sobre="muestra_de_desarrollo", semilla=semilla)
            registro = {"tipo": tipo, "retroceso_verdadero_rel": casos[(c, direccion)]["retroceso_verdadero_rel"],
                        "amplitud_rel": None if amplitud is None else amplitud / escala.valor,
                        "efecto_rel": None if r0.efecto_total is None else r0.efecto_total / escala.valor,
                        "filas_comprobadas": r0.filas_comprobadas, "fraccion_violadoras": {}, "veredictos": {}}
            for e in EPS_PCT:
                r = r0.con_epsilon(e / 100.0)
                comp = r.filas_comprobadas
                registro["fraccion_violadoras"][str(e)] = (r.filas_con_violacion / comp) if comp else None
                for q in Q_PCT:
                    v = veredicto_ter(r.retrocesos, amplitud, r.epsilon_absoluto, q)
                    registro["veredictos"][f"e{e}|q{q}"] = v
                    celda = acum.setdefault((grupo, e, q), _celda_vacia())
                    bis._sumar(celda["plana"][c] if tipo == "plana" else celda[tipo], v)
                    if tipo == "no_monotono":                     # el DATO: la potencia separada (no decide)
                        if c in COLUMNAS_CONTRARIAS:
                            bis._sumar(celda["contraria"], v)
                        elif c in COLUMNAS_NO_MONOTONAS:
                            bis._sumar(celda["no_monotona_x5689"], v)
            detalle[f"{grupo}|s{semilla}|{c}|{direccion}"] = registro
    return round(time.perf_counter() - t0, 1)


def potencia_separada(acum, grupos, e, q):
    """POT por subgrupo de los no monótonos, por grupo y con los grupos juntos."""
    out = {}
    for nombre in ("contraria", "no_monotona_x5689"):
        por_grupo = {}
        tot = bis._vacio()
        for g in grupos:
            cu = acum[(g, e, q)][nombre]
            por_grupo[g] = {**cu, "POT": (cu["con"] / cu["n"]) if cu["n"] else None}
            for k in tot:
                tot[k] += cu[k]
        out[nombre] = {"por_grupo": por_grupo, "juntos": {**tot, "POT": (tot["con"] / tot["n"]) if tot["n"] else None}}
    return out


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

def tabla_markdown(acum, por_tarea, resultado):
    pc = bis._pc
    L = ["# 149-C0-ter — tablas por tarea y celda (ε, q)", "",
         f"Semillas {list(SEMILLAS)} · G = {G} · m = {M} · {FILAS} filas · FA = falsa alarma sobre casos monótonos, POT = potencia sobre no "
         "monótonos (semillas juntas). `cumple` = los DOS grupos con FA <= 10 % y POT >= 80 %. Veredicto V'(ε, q): AMPLITUD (mediana de "
         "máx − mín de la curva de cada fila) primero, luego fracción de filas con retroceso > ε frente a q.", "",
         f"Regla: sha256 `{resultado['regla_sha256']}`", ""]
    for tarea in TAREAS:
        grupos = por_tarea[tarea]["grupos"]
        d = por_tarea[tarea]
        L += [f"## {tarea}", "", f"**Resultado:** {d['veredicto']}"
              + ("" if not d.get("planas") or d["planas"]["todas_salen_no_se_pudo_comprobar_en_al_menos_90_pct"] else " · las planas no se distinguen"), ""]
        L += ["| ε | q | " + " | ".join(f"{g} FA / POT (n_mon, n_nomon)" for g in grupos) + " | cumple | invertida cumple |",
              "|---|---|" + "---|" * (len(grupos) + 2)]
        for e in EPS_PCT:
            for q in Q_PCT:
                t = bis.tasas(acum, grupos, e, q)
                ti = bis.tasas(acum, grupos, e, q, invertida=True)
                L.append(f"| {e} % | {q} % | " + " | ".join(f"{pc(t[g]['FA'])} / {pc(t[g]['POT'])} ({t[g]['n_mon']}, {t[g]['n_nomon']})" for g in grupos)
                         + f" | {'sí' if bis.cumple(t) else 'no'} | {'sí' if bis.cumple(ti) else 'no'} |")
        L += ["", f"### {tarea} — potencia separada (DATO, no decide): con de n", "",
              "| ε | q | " + " | ".join(f"{g}: contraria x1–x4 | {g}: no monótonas x5 x6 x8 x9" for g in grupos) + " | juntos: contraria | juntos: no monótonas |",
              "|---|---|" + "---|" * (2 * len(grupos) + 2)]
        for e in EPS_PCT:
            for q in Q_PCT:
                ps = potencia_separada(acum, grupos, e, q)
                cel = lambda x: f"{x['con']} de {x['n']} ({pc(x['POT'])})"
                L.append(f"| {e} % | {q} % | " + " | ".join(f"{cel(ps['contraria']['por_grupo'][g])} | {cel(ps['no_monotona_x5689']['por_grupo'][g])}" for g in grupos)
                         + f" | {cel(ps['contraria']['juntos'])} | {cel(ps['no_monotona_x5689']['juntos'])} |")
        L += ["", f"### {tarea} — planas ({', '.join(PLANAS)}), grupos juntos: con / sin / no_se_pudo_comprobar de n", "",
              "| ε | q | " + " | ".join(PLANAS) + " |", "|---|---|" + "---|" * len(PLANAS)]
        for e in EPS_PCT:
            for q in Q_PCT:
                pl = bis.planas_en(acum, grupos, e, q)
                L.append(f"| {e} % | {q} % | " + " | ".join(f"{pl[c]['con']} / {pl[c]['sin']} / {pl[c]['nsc']} de {pl[c]['n']}" for c in PLANAS) + " |")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=str(AQUI / "resultados"))
    args = ap.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()
    regla = AQUI / "REGLA_DE_ELECCION_TER.md"
    import matrixai, matrixai_engines, matrixai_studio
    from matrixai.estudio import comportamiento
    try:
        commit = subprocess.run(["git", "-C", str(AQUI), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    resultado = {
        "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit_del_arbol": commit,
        "regla_sha256": hashlib.sha256(regla.read_bytes()).hexdigest(),
        "medir_c0_sha256": hashlib.sha256((AQUI / "medir_c0.py").read_bytes()).hexdigest(),
        "medir_c0bis_sha256": hashlib.sha256((AQUI / "medir_c0bis.py").read_bytes()).hexdigest(),
        "comportamiento_sha256": hashlib.sha256(Path(comportamiento.__file__).read_bytes()).hexdigest(),
        "config": {"semillas": SEMILLAS, "G": G, "m": M, "filas": FILAS, "motores": MOTORES, "epsilon_pct": EPS_PCT, "q_pct": Q_PCT,
                   "umbrales": {"FA": UMBRAL_FA, "POT": UMBRAL_POT, "planas_no_se_pudo_comprobar": MIN_PLANAS}, "planas": PLANAS,
                   "curvas": "recomputadas con modelo.ice_preparado (el mismo camino que el C0-bis); C1 no las conserva"},
        "codigo": {"matrixai": matrixai.__file__, "matrixai_engines": matrixai_engines.__file__, "matrixai_studio": matrixai_studio.__file__,
                   "comportamiento": comportamiento.__file__, "medir_c0": c0.__file__, "medir_c0bis": bis.__file__, "medir_c0ter": __file__},
    }
    registrar("código:", resultado["codigo"])
    acum, detalle, grupos_de, ajuste, evaluacion = {}, {}, {t: [] for t in TAREAS}, {}, {}
    for tarea in TAREAS:
        for motor_id in MOTORES:
            grupo = f"{motor_id}|{'reg' if tarea == 'regression' else 'bin'}"
            grupos_de[tarea].append(grupo)
            for semilla in SEMILLAS:
                crudas = c0.generar(tarea, FILAS, semilla)
                modelo = c0.Modelo(motor_id, tarea, semilla, crudas)
                ajuste[f"{grupo}|s{semilla}"] = round(modelo.t_ajuste, 1)
                evaluacion[f"{grupo}|s{semilla}"] = evaluar_modelo_ter(modelo, semilla, grupo, acum, detalle)
                registrar(grupo, "semilla", semilla, "ajustado", ajuste[f"{grupo}|s{semilla}"], "s · evaluado", evaluacion[f"{grupo}|s{semilla}"], "s")
    por_tarea = {t: bis.decidir_tarea(acum, grupos_de[t]) for t in TAREAS}
    resultado["por_tarea"] = por_tarea
    resultado["tablas"] = {t: {f"e{e}|q{q}": {"tasas": bis.tasas(acum, grupos_de[t], e, q), "cumple": bis.cumple(bis.tasas(acum, grupos_de[t], e, q)),
                                               "tasas_con_etiqueta_invertida": bis.tasas(acum, grupos_de[t], e, q, invertida=True),
                                               "cumple_invertida": bis.cumple(bis.tasas(acum, grupos_de[t], e, q, invertida=True)),
                                               "potencia_separada": potencia_separada(acum, grupos_de[t], e, q),
                                               "planas": bis.planas_en(acum, grupos_de[t], e, q)}
                               for e in EPS_PCT for q in Q_PCT} for t in TAREAS}
    resultado["celdas"] = {f"{g}|e{e}|q{q}": v for (g, e, q), v in sorted(acum.items())}
    resultado["detalle_por_caso"] = detalle
    resultado["segundos"] = {"ajuste_por_modelo": ajuste, "evaluacion_por_modelo": evaluacion}
    resultado["segundos_totales"] = round(time.perf_counter() - t_inicio, 1)
    (salida / "c0ter_resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (salida / "c0ter_tabla.md").write_text(tabla_markdown(acum, por_tarea, resultado), encoding="utf-8")
    for t in TAREAS:
        registrar(t, "->", por_tarea[t]["veredicto"], "| invertida:", por_tarea[t]["eleccion_con_etiqueta_invertida"])
    registrar("FIN", resultado["segundos_totales"], "s ->", salida)


if __name__ == "__main__":
    main()
