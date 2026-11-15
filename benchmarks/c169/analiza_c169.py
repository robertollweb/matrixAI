#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169 — EL ANALIZADOR: ¿la rejilla de 8 configuraciones de HGB mejora al perfil de hoy en los 6 conjuntos NUEVOS? La regla es `REGLA.md` (§2–§3).

    python3 benchmarks/c169/analiza_c169.py <resultado.json> [--json <salida.json>]

Veredictos: CONFIRMADA / NO CUMPLE / NO CONCLUYENTE (N < 3 no saturados) / SIN VEREDICTO (la medida no es completa o los sellos no cuadran) /
SONDA (una sonda no decide nada). Sale con 0 CONFIRMADA o SONDA, 1 NO CUMPLE, 3 NO CONCLUYENTE, 2 SIN VEREDICTO.

NO reescribe nada del 160 ni del 163-C2: importa `analiza_c160` (R, IC95 por bootstrap de pliegues, clasificar), `analiza_c163_c2` (la lectura de un
registro, el coste, las comprobaciones de brazo) y `protocolo._intervalo_pareado` (el MISMO bootstrap —1.000 remuestras, semilla 0, 95 %, percentiles—
para el IC de D).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parents[1]
for _ruta in (_AQUI, _RAIZ / "benchmarks" / "c163", _RAIZ / "benchmarks" / "c160", _RAIZ / "benchmarks" / "fase0", _RAIZ,
              _RAIZ.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import analiza_c160 as a160  # noqa: E402
import analiza_c163_c2 as an163  # noqa: E402
import conjuntos_c169 as cj  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# LOS UMBRALES DE LA REGLA (REGLA.md §2–§3). Un número por línea; las pruebas los fijan contra REGLA.md.
# ----------------------------------------------------------------------------------------------------------------------
UMBRAL_DE_SATURACION = 0.98          # saturado: media de la métrica de cierre de hgb_hoy sobre sus 5 pliegues >= 0,98
N_MINIMO_NO_SATURADOS = 3            # (0) potencia mínima
A_REDUCCION_MEDIA_MIN = 0.03         # (a) media de R sobre los no saturados >= +3 %, sin redondear
C_EMPEORA_MAS_DE = 0.03              # (c) no saturado: R < -3 % y IC95 de R < 0
C_SATURADO_EMPEORA_D = -0.01         # (c) saturado: D < -0,01 y IC95 de D < 0 (fijado DESPUÉS de ver `sick`)
D_COSTE_MAXIMO = 8.0                 # (d) Σ tiempo_de_ajuste rejilla <= 8 x Σ hgb_hoy, razón de sumas sobre los 6
PLIEGUES = [0, 1, 2, 3, 4]
REPETICION = 0
N_CONJUNTOS = 6

REFERENCIA, REJILLA = "hgb_hoy", "hgb_rejilla"
BRAZOS = (REFERENCIA, REJILLA)
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA

RUTA_DE_LA_REGLA = _AQUI / "REGLA.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()

#: sha256 de `REGLA.md` SELLADA. La pone el supervisor al sellar. `None` = sin sellar: ninguna medida real vale.
REGLA_SHA256_SELLADA: str | None = "eb65a7e92013fc3992072fdd66cd908f0ccf8844c26f12c6828229e964e3390e"  # sellada por el supervisor el 15-11


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia (las del 163-C2, que ya cubren el 160, la Fase 0 y el núcleo) MÁS el 163-C2 y el catálogo."""
    d = dict(an163.dependencias_de_ahora())
    d["benchmarks/c163/analiza_c163_c2.py"] = _sha256(Path(an163.__file__).resolve())
    d["benchmarks/c169/conjuntos_c169.py"] = _sha256(Path(cj.__file__).resolve())
    d["benchmarks/c169/conjuntos_c169.json"] = _sha256(cj.RUTA_DEL_CATALOGO)
    return d


def sellos_de_ahora() -> dict:
    return {"regla_sha256": _sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None,
            "analizador_sha256": _sha256(RUTA_DEL_ANALIZADOR), "dependencias_sha256": dependencias_de_ahora()}


# ----------------------------------------------------------------------------------------------------------------------
# 1. UN CONJUNTO
# ----------------------------------------------------------------------------------------------------------------------
def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, registros: dict[int, dict[str, dict]], esperados: list[int]) -> dict:
    """UN conjunto: por pliegue la métrica de los dos brazos, R y su IC, D y su IC, saturado (por la REFERENCIA sola), coste.
    `registros[pliegue][brazo]`."""
    ev = an163.evaluar_conjunto(nombre, tarea, metric_id, "hgb", registros, esperados, saturados=())
    ev["saturado"] = False
    if not ev["completo"]:
        return ev
    pl = ev["pliegues"]
    hoy = [ev["por_pliegue"][f][REFERENCIA] for f in pl]
    rej = [ev["por_pliegue"][f][REJILLA] for f in pl]
    media_hoy = statistics.fmean(hoy)
    ev["saturado"] = bool(media_hoy >= UMBRAL_DE_SATURACION)       # SOLO la referencia: la rejilla no decide quién es saturado
    diferencias = [b - a for a, b in zip(hoy, rej)]
    ev["d"] = statistics.fmean(diferencias)
    ic_d = protocolo_mod._intervalo_pareado(diferencias)
    ev["ic_d"] = list(ic_d) if ic_d else None
    if ev["saturado"]:
        # Para un saturado la medida es D, no R: R/«mejora»/«empeora» de R no cuentan.
        ev["mejora_clara"] = False
        ev["empeora_de_verdad"] = bool(ic_d is not None and ev["d"] < C_SATURADO_EMPEORA_D and ic_d[1] < 0.0)
    return ev


# ----------------------------------------------------------------------------------------------------------------------
# 2. LA REGLA
# ----------------------------------------------------------------------------------------------------------------------
def aplicar_la_regla(evaluados: list[dict]) -> dict:
    """(0) y (a)–(d) sobre conjuntos YA evaluados y completos. `veredicto`: CONFIRMADA / NO CUMPLE / NO CONCLUYENTE."""
    no_sat = [e for e in evaluados if not e["saturado"]]
    n = len(no_sat)
    cero = n >= N_MINIMO_NO_SATURADOS
    r_ns = [e["r"] for e in no_sat]
    a_media = statistics.fmean(r_ns) if r_ns else None
    a_ok = a_media is not None and a_media >= A_REDUCCION_MEDIA_MIN
    b_min = math.ceil(n / 2) if n else 0
    mejoras = [e["nombre"] for e in no_sat if e["mejora_clara"]]
    b_ok = n > 0 and len(mejoras) >= b_min
    empeoran = [e["nombre"] for e in evaluados if e["empeora_de_verdad"]]
    c_ok = not empeoran
    suma = {b: sum(e["coste_s"][b] for e in evaluados) for b in BRAZOS}
    ratios = [e["coste_ratio"] for e in evaluados if e["coste_ratio"] is not None]
    d_ratio = suma[REJILLA] / suma[REFERENCIA] if suma[REFERENCIA] > 0 else None
    d_ok = d_ratio is not None and d_ratio <= D_COSTE_MAXIMO
    if not cero:
        veredicto = "NO CONCLUYENTE"
    elif a_ok and b_ok and c_ok and d_ok:
        veredicto = "CONFIRMADA"
    else:
        veredicto = "NO CUMPLE"
    return {"veredicto": veredicto, "n_no_saturados": n, "saturados": [e["nombre"] for e in evaluados if e["saturado"]],
            "no_saturados": [e["nombre"] for e in no_sat],
            "cero": {"n": n, "umbral": N_MINIMO_NO_SATURADOS, "cumple": cero},
            "a": {"media_r_no_saturados": a_media, "umbral": A_REDUCCION_MEDIA_MIN, "cumple": a_ok},
            "b": {"mejoras_claras": mejoras, "n": len(mejoras), "umbral": b_min, "cumple": b_ok},
            "c": {"empeoran": empeoran, "cumple": c_ok,
                  "d_de_los_saturados": {e["nombre"]: e["d"] for e in evaluados if e["saturado"]}},
            "d": {"razon_de_sumas": d_ratio, "mediana_por_conjunto": statistics.median(ratios) if ratios else None,
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok}}


# ----------------------------------------------------------------------------------------------------------------------
# 3. COMPLETITUD, SELLOS Y LA MEDIDA ENTERA
# ----------------------------------------------------------------------------------------------------------------------
def problemas_de_completitud(payload: dict, es_sonda: bool) -> list[str]:
    p: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    for n in sorted({n for n in nombres if nombres.count(n) > 1}):
        p.append(f"conjunto DUPLICADO en la salida: {n}")
    if not es_sonda:
        esperados = {d.nombre for d in cj.conjuntos()}
        if set(nombres) != esperados:
            p.append(f"los conjuntos no son los {N_CONJUNTOS} de la regla: faltan {sorted(esperados - set(nombres))}, "
                     f"sobran {sorted(set(nombres) - esperados)}")
        for c in conjuntos:
            if list(c.get("pliegues_esperados") or []) != PLIEGUES:
                p.append(f"{c.get('nombre')}: pliegues declarados {c.get('pliegues_esperados')} != {PLIEGUES}")
    ids_registrados = {d.nombre: d.data_id for d in cj.conjuntos()}
    for c in conjuntos:
        if c.get("nombre") in ids_registrados and c.get("data_id") != ids_registrados[c["nombre"]]:
            p.append(f"{c.get('nombre')}: data_id {c.get('data_id')} != el registrado {ids_registrados[c['nombre']]}")
        if c.get("nombre") in ids_registrados:
            esperado_ign = cj.ignorados_registrados()[ids_registrados[c["nombre"]]]
            if c.get("columnas_ignoradas_por_openml") != esperado_ign:
                p.append(f"{c.get('nombre')}: declara haber excluido {c.get('columnas_ignoradas_por_openml')} y los `ignore_attribute` de "
                         f"OpenML son {esperado_ign}")
        if c.get("sellado"):
            p.append(f"{c.get('nombre')} está marcado como SELLADO")
    esperado = {(c.get("nombre"), b, int(f)) for c in conjuntos for b in BRAZOS for f in (c.get("pliegues_esperados") or [])}
    sha_ok = {d.nombre: d.sha256_arff for d in cj.conjuntos()}
    vistos: dict[tuple, int] = {}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("motor"), int(r.get("pliegue", -1)))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("motor") not in BRAZOS:
            p.append(f"el intento {k} es de un brazo que no es de esta medida ({r.get('motor')!r})")
            continue
        if r.get("repeticion") != REPETICION:
            p.append(f"un registro de la repetición {r.get('repeticion')} ({k}): la regla es sobre la 0")
        if r.get("estado") not in ESTADOS_DE_MEDIDA:
            p.append(f"el intento {k} no cuenta como medida (estado {r.get('estado')!r}): falta ese pliegue")
            continue
        if an163.metrica_del_registro(r) is None:
            p.append(f"el intento {k} no trae su métrica de cierre ({r.get('metrica_de_cierre')!r})")
        if an163.coste_del_registro(r) is None:
            p.append(f"el intento {k} no trae `tiempo_de_ajuste` (el coste de (d))")
        if r.get("dataset") in sha_ok and r.get("datos_sha256") != sha_ok[r["dataset"]]:
            p.append(f"el intento {k}: sha256 de los datos {str(r.get('datos_sha256'))[:12]} != el registrado "
                     f"{sha_ok[r['dataset']][:12]}")
        p += an163.problemas_de_la_declaracion(r)
    faltan_i, sobran_i = sorted(esperado - set(vistos)), sorted(set(vistos) - esperado)
    dobles = sorted(k for k, n in vistos.items() if n > 1)
    if faltan_i:
        p.append(f"faltan {len(faltan_i)} intentos (p. ej. {faltan_i[:3]})")
    if sobran_i:
        p.append(f"{len(sobran_i)} registros fuera de lo declarado (p. ej. {sobran_i[:3]})")
    if dobles:
        p.append(f"{len(dobles)} intentos registrados más de una vez (p. ej. {dobles[:3]})")
    return p


def problemas_de_sellos(payload: dict) -> list[str]:
    p: list[str] = []
    sellos = sellos_de_ahora()
    if REGLA_SHA256_SELLADA is None:
        p.append("la regla NO está sellada (`REGLA_SHA256_SELLADA` es None): ninguna medida vale todavía")
    elif sellos["regla_sha256"] != REGLA_SHA256_SELLADA:
        p.append(f"REGLA.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
    al_medir = payload.get("dependencias_sha256_al_medir")
    if not isinstance(al_medir, dict):
        p.append("la salida no declara `dependencias_sha256_al_medir`")
    else:
        for etiqueta, propio in sellos["dependencias_sha256"].items():
            if al_medir.get(etiqueta) != propio:
                p.append(f"`{etiqueta}` al medir ({str(al_medir.get(etiqueta))[:12]}) != el que importa este analizador ({propio[:12]})")
    for clave_, propio in (("regla_sha256_al_medir", sellos["regla_sha256"]), ("analizador_sha256_al_medir", sellos["analizador_sha256"])):
        valor = payload.get(clave_)
        if valor is None:
            p.append(f"la salida no declara `{clave_}`")
        elif valor != propio:
            p.append(f"`{clave_}` de la salida ({valor[:12]}) != el fichero de ahora ({str(propio)[:12]}): cambió DESPUÉS de medir")
    if payload.get("regla_sha256_sellada") != REGLA_SHA256_SELLADA:
        p.append("la salida declara otro sello de regla que el de este analizador")
    return p


def analizar(payload: dict, *, comprobar_sellos: bool = True) -> dict:
    """El veredicto de una salida de `medir_c169.py`."""
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    problemas += problemas_de_completitud(payload, es_sonda)
    rs = payload.get("resultados", [])
    for brazo in BRAZOS:
        for campo in ("entorno_digest", "motor_digest"):
            if len({r.get(campo) for r in rs if r.get("motor") == brazo}) > 1:
                problemas.append(f"los intentos del brazo {brazo} se midieron con CÓDIGO DISTINTO ({campo})")
    if len({r.get("entorno_digest") for r in rs}) > 1:
        problemas.append("los brazos se midieron con CÓDIGO DISTINTO (entorno_digest)")
    if comprobar_sellos:
        problemas += problemas_de_sellos(payload)
    por_clave = {(r["dataset"], r["motor"], int(r["pliegue"])): r for r in rs
                 if r.get("estado") in ESTADOS_DE_MEDIDA and r.get("motor") in BRAZOS}
    evaluados = []
    for c in payload.get("conjuntos", []):
        esperados = [int(f) for f in c.get("pliegues_esperados") or []]
        registros: dict[int, dict[str, dict]] = {}
        for f in esperados:
            for b in BRAZOS:
                r = por_clave.get((c["nombre"], b, f))
                if r is not None:
                    registros.setdefault(f, {})[b] = r
        e = evaluar_conjunto(c["nombre"], c["tarea"], c["metrica_de_cierre"], registros, esperados)
        if not e["completo"]:
            problemas.append(f"{c['nombre']}: faltan pliegues {e['pliegues_que_faltan']}")
        evaluados.append(e)
    regla = aplicar_la_regla(evaluados) if evaluados and all(e["completo"] for e in evaluados) else None
    if es_sonda:
        veredicto = "SONDA"
    elif problemas or regla is None:
        veredicto = "SIN VEREDICTO"
    else:
        veredicto = regla["veredicto"]
    return {"veredicto": veredicto, "problemas": problemas, "es_sonda": es_sonda, "sellos_de_ahora": sellos_de_ahora(),
            "regla": regla, "conjuntos": evaluados,
            "nota_del_umbral": "El umbral de -0,01 de los saturados se fijó DESPUÉS de ver `sick` (163-C2), donde habría quedado dentro "
                               "(-0,42 puntos). Con 6 conjuntos pequeños la potencia es baja: un NO CUMPLE no demuestra que la rejilla sea "
                               "mala y una CONFIRMADA no la integra (decisión de Roberto)."}


def _f(x, nd: int = 3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def informe_en_texto(r: dict) -> str:
    L = [f"=== 169: rejilla de HGB en 6 conjuntos nuevos — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  (0) no saturados: {g['cero']['n']} (>= {g['cero']['umbral']}) -> {'ok' if g['cero']['cumple'] else 'NO'}  saturados: {g['saturados']}")
        L.append(f"  (a) media de R sobre los no saturados: {_f(g['a']['media_r_no_saturados'], 4)} (>= {g['a']['umbral']}) -> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras']} -> {'ok' if g['b']['cumple'] else 'NO'}")
        L.append(f"  (c) empeoran de verdad: {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) razón de sumas de ajuste: {_f(g['d']['razon_de_sumas'], 2)} (<= {g['d']['umbral']}); mediana {_f(g['d']['mediana_por_conjunto'], 2)}, "
                 f"máxima {_f(g['d']['maxima_por_conjunto'], 2)} -> {'ok' if g['d']['cumple'] else 'NO'}")
    L.append(f"  {'conjunto':34s} sat      R        IC95 de R            D        IC95 de D           mejora empeora")
    for e in r["conjuntos"]:
        if "d" not in e:
            L.append(f"  {e['nombre']:34s} (sin medida completa)")
            continue
        ic, icd = e["ic"], e.get("ic_d")
        L.append(f"  {e['nombre']:34s} {'S' if e['saturado'] else ' '}   {e['r']:+8.4f}  [{_f(ic[0]) if ic else '—'}, {_f(ic[1]) if ic else '—'}]  "
                 f"{e['d']:+8.4f}  [{_f(icd[0], 4) if icd else '—'}, {_f(icd[1], 4) if icd else '—'}]  "
                 f"{'sí' if e['mejora_clara'] else '  '}    {'SÍ' if e['empeora_de_verdad'] else '  '}")
    L.append(f"  NOTA: {r['nota_del_umbral']}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    a = p.parse_args(argv)
    payload = json.loads(Path(a.resultado).resolve().read_text(encoding="utf-8"))
    r = analizar(payload)
    print(informe_en_texto(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return {"CONFIRMADA": 0, "SONDA": 0, "NO CUMPLE": 1, "NO CONCLUYENTE": 3}.get(r["veredicto"], 2)


if __name__ == "__main__":
    sys.exit(main())
