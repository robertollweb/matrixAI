#!/usr/bin/env python3
"""149-C0-bis — veredicto por FRACCIÓN de filas de la monotonía declarada, MEDIDO.

La regla de elección está en `REGLA_DE_ELECCION_BIS.md`, commiteada ANTES que este guion; aquí se implementa tal cual y
el resultado graba su sha256. Los datos sintéticos (`generar`), los motores, la preparación (`Modelo`), la escala, la
verdad de cada caso (`casos_de_verdad`) y los retrocesos por fila (`monotonia_desde_curvas`, C1) son los de
`medir_c0.py`, importado: no se copia nada. Lo único nuevo es el veredicto V(ε, q) y la regla por tarea.

    cd /home/deployer/worktrees/c149/matrixAI
    PYTHONPATH=<núcleo>:<motores>/src:<studio-backend> python3 benchmarks/c149/medir_c0bis.py [--salida DIR]

Escribe `c0bis_resultado.json` y `c0bis_tabla.md` en `--salida` (por omisión `benchmarks/c149/resultados`).
ESTIMACIÓN: ~3 min (el C0 entero, con rejilla mayor y 500 filas, tardó 161 s).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# El guion del C0, tal cual (misma clave que usa su prueba: si ya está cargado se reutiliza).
if "medir_c0_149" in sys.modules:
    c0 = sys.modules["medir_c0_149"]
else:
    _ESPEC = importlib.util.spec_from_file_location("medir_c0_149", AQUI / "medir_c0.py")
    c0 = importlib.util.module_from_spec(_ESPEC)
    sys.modules["medir_c0_149"] = c0
    _ESPEC.loader.exec_module(c0)

SEMILLAS = (3, 4, 5)
G, M = 25, 200
MOTORES = ("lightgbm", "sklearn.hgb")
FILAS = 2000
EPS_PCT = (2, 5, 10, 15, 20)          # % de la escala
Q_PCT = (5, 10, 20)                   # % de las filas
UMBRAL_FA, UMBRAL_POT, MIN_PLANAS = 0.10, 0.80, 0.90
TAREAS = ("regression", "binary_classification")
PLANAS = tuple(c for c, (tipo, _d) in c0.VERDAD_DE_LA_COLUMNA.items() if tipo == "plana")
registrar = c0.registrar


# ---------------------------------------------------------------------------
# El veredicto V(ε, q) — función pura sobre los retrocesos por fila
# ---------------------------------------------------------------------------

def veredicto_bis(retrocesos, efecto_total, epsilon_absoluto, q_pct):
    """V(ε, q) de la regla, sobre los retrocesos por fila (`None` = fila sin predicción, no cuenta).

    1. EL EFECTO PRIMERO: efecto total <= ε  ->  `no_se_pudo_comprobar`, haya o no filas que retrocedan.
    2. Fracción de filas con retroceso > ε  >=  q  ->  `con_contraejemplos`.
    3. Si no  ->  `sin_contraejemplos`.
    `q_pct` es un entero en %: la comparación es exacta (violadoras·100 >= q·comprobadas), sin coma flotante."""
    comprobadas = [r for r in retrocesos if r is not None]
    if epsilon_absoluto is None or not comprobadas:
        return "no_se_pudo_comprobar"
    if efecto_total is None or not efecto_total > epsilon_absoluto:
        return "no_se_pudo_comprobar"
    violadoras = sum(1 for r in comprobadas if r > epsilon_absoluto)
    if violadoras * 100 >= q_pct * len(comprobadas):
        return "con_contraejemplos"
    return "sin_contraejemplos"


def veredicto_de_c1(resultado, q_pct):
    """V(ε, q) con los retrocesos, el efecto y la tolerancia absoluta de un `MonotoniaDeUnaColumna` de C1
    (ya con su `epsilon_relativo`)."""
    return veredicto_bis(resultado.retrocesos, resultado.efecto_total, resultado.epsilon_absoluto, q_pct)


# ---------------------------------------------------------------------------
# Acumulación por grupo y celda
# ---------------------------------------------------------------------------

VEREDICTOS = ("con_contraejemplos", "sin_contraejemplos", "no_se_pudo_comprobar")


def _vacio():
    return {"n": 0, "con": 0, "sin": 0, "nsc": 0}


def _sumar(cuenta, v):
    cuenta["n"] += 1
    cuenta["con" if v == VEREDICTOS[0] else "sin" if v == VEREDICTOS[1] else "nsc"] += 1


def _celda_vacia():
    return {"monotono": _vacio(), "no_monotono": _vacio(), "intermedio": _vacio(), "plana": {c: _vacio() for c in PLANAS}}


def evaluar_modelo_bis(modelo, semilla, grupo, acum, detalle):
    """Un modelo (grupo, semilla): sus casos de verdad y, por (ε, q), el veredicto de cada (columna, dirección)."""
    from matrixai.estudio.comportamiento import elegir_filas, monotonia_desde_curvas
    indices = elegir_filas(len(modelo.crudas), M, semilla)
    escala = modelo.escala()
    mallas = {c: _malla(modelo.crudas, c) for c in c0.COLUMNAS}
    casos = c0.casos_de_verdad(modelo.tarea, modelo.crudas, indices, {G: mallas}, escala.valor)
    t0 = time.perf_counter()
    for c in c0.COLUMNAS:
        curvas = modelo.ice_preparado(indices, c, mallas[c])
        for direccion in ("sube", "baja"):
            tipo = casos[(c, direccion)]["tipo"]
            if c in PLANAS:
                tipo = "plana"                                  # la regla las aparta: ni falsa alarma ni detección
            r0 = monotonia_desde_curvas(c, direccion, mallas[c], curvas, list(indices), epsilon_relativo=0.0, escala=escala,
                                        medido_sobre="muestra_de_desarrollo", semilla=semilla)
            registro = {"tipo": tipo, "retroceso_verdadero_rel": casos[(c, direccion)]["retroceso_verdadero_rel"],
                        "efecto_rel": None if r0.efecto_total is None else r0.efecto_total / escala.valor,
                        "filas_comprobadas": r0.filas_comprobadas, "fraccion_violadoras": {}, "veredictos": {}}
            for e in EPS_PCT:
                r = r0.con_epsilon(e / 100.0)                   # el mismo recálculo que C1 (I7), sin volver a predecir
                comp = r.filas_comprobadas
                registro["fraccion_violadoras"][str(e)] = (r.filas_con_violacion / comp) if comp else None
                for q in Q_PCT:
                    v = veredicto_de_c1(r, q)
                    registro["veredictos"][f"e{e}|q{q}"] = v
                    celda = acum.setdefault((grupo, e, q), _celda_vacia())
                    _sumar(celda["plana"][c] if tipo == "plana" else celda[tipo], v)
            detalle[f"{grupo}|s{semilla}|{c}|{direccion}"] = registro
    return round(time.perf_counter() - t0, 1)


def _malla(crudas, columna):
    from matrixai.estudio.comportamiento import rejilla
    return rejilla(min(f[columna] for f in crudas), max(f[columna] for f in crudas), G)


# ---------------------------------------------------------------------------
# La regla, por tarea
# ---------------------------------------------------------------------------

def tasas(acum, grupos, e, q, invertida=False):
    """FA y POT por grupo (semillas juntas). Con `invertida` las dos etiquetas de verdad cambian de papel."""
    por_grupo = {}
    for g in grupos:
        celda = acum[(g, e, q)]
        mon, nomon = (celda["no_monotono"], celda["monotono"]) if invertida else (celda["monotono"], celda["no_monotono"])
        por_grupo[g] = {"FA": (mon["con"] / mon["n"]) if mon["n"] else None, "POT": (nomon["con"] / nomon["n"]) if nomon["n"] else None,
                        "n_mon": mon["n"], "n_nomon": nomon["n"], "mon_con": mon["con"], "nomon_con": nomon["con"],
                        "mon_sin_comprobar": mon["nsc"], "nomon_sin_comprobar": nomon["nsc"]}
    return por_grupo


def cumple(por_grupo):
    """Los DOS grupos a la vez: FA <= 10 % y POT >= 80 %."""
    return all(t["FA"] is not None and t["POT"] is not None and t["FA"] <= UMBRAL_FA and t["POT"] >= UMBRAL_POT
               for t in por_grupo.values())


def elegir(acum, grupos, invertida=False):
    """(ε, q) por omisión de la tarea: el menor ε que cumple con algún q; con ese ε, el menor q. Sin celda: None."""
    for e in EPS_PCT:
        qs = [q for q in Q_PCT if cumple(tasas(acum, grupos, e, q, invertida))]
        if qs:
            return {"epsilon_pct": e, "q_pct": min(qs), "q_que_cumplen_con_ese_epsilon": qs}
    return None


def planas_en(acum, grupos, e, q):
    """Por plana: veredictos de sus casos (los grupos de la tarea juntos) y fracción `no_se_pudo_comprobar`."""
    out = {}
    for c in PLANAS:
        tot = _vacio()
        for g in grupos:
            for k in tot:
                tot[k] += acum[(g, e, q)]["plana"][c][k]
        out[c] = {**tot, "fraccion_no_se_pudo_comprobar": (tot["nsc"] / tot["n"]) if tot["n"] else None}
    return out


def decidir_tarea(acum, grupos):
    """La regla entera para una tarea: elección, planas (punto 4) y sabotaje (punto 5)."""
    eleccion = elegir(acum, grupos)
    falsa = elegir(acum, grupos, invertida=True)
    out = {"grupos": list(grupos), "eleccion": eleccion, "eleccion_con_etiqueta_invertida": falsa}
    if eleccion is None:
        out["veredicto"] = "sin veredicto por omisión"
        out["planas"] = None
        out["sabotaje"] = {"distingue": None, "nota": "sin elección real: no hay nada que comparar (la invertida "
                           + ("tampoco elige" if falsa is None else f"elige ε={falsa['epsilon_pct']} %, q={falsa['q_pct']} %") + ")"}
        return out
    e, q = eleccion["epsilon_pct"], eleccion["q_pct"]
    pl = planas_en(acum, grupos, e, q)
    todas_ok = all(v["fraccion_no_se_pudo_comprobar"] is not None and v["fraccion_no_se_pudo_comprobar"] >= MIN_PLANAS for v in pl.values())
    out["planas"] = {"por_columna": pl, "todas_salen_no_se_pudo_comprobar_en_al_menos_90_pct": todas_ok,
                     "declaracion": None if todas_ok else "las planas no se distinguen"}
    mismo = falsa is not None and (falsa["epsilon_pct"], falsa["q_pct"]) == (e, q)
    out["sabotaje"] = {"distingue": not mismo, "invertida_elige_lo_mismo": mismo}
    out["veredicto"] = "elección ANULADA por el sabotaje" if mismo else f"ε = {e} %, q = {q} %"
    return out


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

def _pc(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def tabla_markdown(acum, por_tarea, resultado):
    L = ["# 149-C0-bis — tablas por tarea y celda (ε, q)", "",
         f"Semillas {list(SEMILLAS)} · G = {G} · m = {M} · {FILAS} filas · FA = falsa alarma sobre casos monótonos, POT = potencia sobre no "
         "monótonos (semillas juntas). `cumple` = los DOS grupos con FA <= 10 % y POT >= 80 %. Veredicto V(ε, q): efecto total primero, "
         "luego fracción de filas con retroceso > ε frente a q.", "",
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
                t = tasas(acum, grupos, e, q)
                ti = tasas(acum, grupos, e, q, invertida=True)
                L.append(f"| {e} % | {q} % | " + " | ".join(f"{_pc(t[g]['FA'])} / {_pc(t[g]['POT'])} ({t[g]['n_mon']}, {t[g]['n_nomon']})" for g in grupos)
                         + f" | {'sí' if cumple(t) else 'no'} | {'sí' if cumple(ti) else 'no'} |")
        L += ["", f"### {tarea} — planas ({', '.join(PLANAS)}), grupos juntos: con / sin / no_se_pudo_comprobar de n", "",
              "| ε | q | " + " | ".join(PLANAS) + " |", "|---|---|" + "---|" * len(PLANAS)]
        for e in EPS_PCT:
            for q in Q_PCT:
                pl = planas_en(acum, grupos, e, q)
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
    print(time.strftime("%H:%M:%S"), "AVISO: desde 149-C1-bis el núcleo ya no es el que midió este guion (`comportamiento_sha256` "
          "cambia); su veredicto V(ε, q) sale de `veredicto_bis` sobre los retrocesos y el efecto del núcleo, que no cambian, pero un "
          "`.veredicto` del núcleo ya es V'(ε, q) por amplitud: no comparar con los archivados sin mirar ese sha.", flush=True)
    salida.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()
    regla = AQUI / "REGLA_DE_ELECCION_BIS.md"
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
        "comportamiento_sha256": hashlib.sha256(Path(comportamiento.__file__).read_bytes()).hexdigest(),
        "config": {"semillas": SEMILLAS, "G": G, "m": M, "filas": FILAS, "motores": MOTORES, "epsilon_pct": EPS_PCT, "q_pct": Q_PCT,
                   "umbrales": {"FA": UMBRAL_FA, "POT": UMBRAL_POT, "planas_no_se_pudo_comprobar": MIN_PLANAS}, "planas": PLANAS},
        "codigo": {"matrixai": matrixai.__file__, "matrixai_engines": matrixai_engines.__file__, "matrixai_studio": matrixai_studio.__file__,
                   "comportamiento": comportamiento.__file__, "medir_c0": c0.__file__, "medir_c0bis": __file__},
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
                evaluacion[f"{grupo}|s{semilla}"] = evaluar_modelo_bis(modelo, semilla, grupo, acum, detalle)
                registrar(grupo, "semilla", semilla, "ajustado", ajuste[f"{grupo}|s{semilla}"], "s · evaluado", evaluacion[f"{grupo}|s{semilla}"], "s")
    por_tarea = {t: decidir_tarea(acum, grupos_de[t]) for t in TAREAS}
    resultado["por_tarea"] = por_tarea
    resultado["tablas"] = {t: {f"e{e}|q{q}": {"tasas": tasas(acum, grupos_de[t], e, q), "cumple": cumple(tasas(acum, grupos_de[t], e, q)),
                                               "tasas_con_etiqueta_invertida": tasas(acum, grupos_de[t], e, q, invertida=True),
                                               "cumple_invertida": cumple(tasas(acum, grupos_de[t], e, q, invertida=True)),
                                               "planas": planas_en(acum, grupos_de[t], e, q)}
                               for e in EPS_PCT for q in Q_PCT} for t in TAREAS}
    resultado["celdas"] = {f"{g}|e{e}|q{q}": v for (g, e, q), v in sorted(acum.items())}
    resultado["detalle_por_caso"] = detalle
    resultado["segundos"] = {"ajuste_por_modelo": ajuste, "evaluacion_por_modelo": evaluacion}
    resultado["segundos_totales"] = round(time.perf_counter() - t_inicio, 1)
    (salida / "c0bis_resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (salida / "c0bis_tabla.md").write_text(tabla_markdown(acum, por_tarea, resultado), encoding="utf-8")
    for t in TAREAS:
        registrar(t, "->", por_tarea[t]["veredicto"], "| invertida:", por_tarea[t]["eleccion_con_etiqueta_invertida"])
    registrar("FIN", resultado["segundos_totales"], "s ->", salida)


if __name__ == "__main__":
    main()
