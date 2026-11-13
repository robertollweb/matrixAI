#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""163-C2 — EL ANALIZADOR: la REJILLA de 8 configuraciones por validación contra la política por tamaño de hoy, en LightGBM y en HGB, con la
regla de `REGLA_C2.md` (D1–D4 del contrato 163, aprobadas por Roberto con A el 12-11).

    python3 benchmarks/c163/analiza_c163_c2.py <resultado.json> [--json <salida.json>] [--confirmacion]

Cada MOTOR se juzga POR SEPARADO (su referencia contra su rejilla) con la misma regla; no hay un veredicto «combinado» que mezcle un motor que
cumple con otro que no. Sale con 0 si AL MENOS UN motor CUMPLE, 1 si ninguno, 2 si no hay veredicto (medida incompleta, código mezclado, regla o
analizador distintos de los que midieron, un brazo que declara lo que no es), 0 en una SONDA.

NO reescribe nada del 160: importa `analiza_c160` (reducción relativa, IC95 por bootstrap de pliegues, clasificación mejora/empeora, los saturados y
los 23 del congelado).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parents[1]
for _ruta in (_AQUI, _RAIZ / "benchmarks" / "c160", _RAIZ / "benchmarks" / "fase0", _RAIZ,
              _RAIZ.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import analiza_c160 as a160  # noqa: E402  (la reducción relativa, el IC, los saturados y los 23: los MISMOS que el 160)
import pasada_amplia_101_c5 as c5  # noqa: E402,F401
import protocolo as protocolo_mod  # noqa: E402
from matrixai.estudio import incertidumbre as _incertidumbre  # noqa: E402
from matrixai.estudio import metricas as _metricas  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# LOS UMBRALES DE LA REGLA. Un número por línea; las pruebas los fijan contra REGLA_C2.md.
# ----------------------------------------------------------------------------------------------------------------------
N_CONJUNTOS = a160.N_CONJUNTOS                           # 23
A_REDUCCION_MEDIA_MIN = a160.A_REDUCCION_MEDIA_MIN       # (a) media de R sobre los NO saturados >= +3 %
B_MEJORAS_CLARAS_MIN = a160.B_MEJORAS_CLARAS_MIN         # (b) «mejora clara» en >= 5 de los no saturados
C_EMPEORA_MAS_DE = a160.C_EMPEORA_MAS_DE                 # (c) ninguno de los 23 con R < -3 % y el IC95 < 0
D_COSTE_MAXIMO = 8.0                                     # (d) coste de ajustar de la rejilla <= 8 x el de la referencia (por motor)
N_CONFIGURACIONES = 8
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
SATURADOS = a160.SATURADOS

#: Los motores y sus dos brazos. La REFERENCIA primero: es el denominador de R.
MOTORES = ("lightgbm", "hgb")
REFERENCIA = {"lightgbm": "lightgbm_hoy", "hgb": "hgb_hoy"}
REJILLA = {"lightgbm": "lightgbm_rejilla", "hgb": "hgb_rejilla"}
BRAZOS = tuple(b for m in MOTORES for b in (REFERENCIA[m], REJILLA[m]))

RUTA_DE_LA_REGLA = _AQUI / "REGLA_C2.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()

#: sha256 de `REGLA_C2.md` SELLADA. `medir_c163_c2.py` la repite y se niega a medir si el fichero ya no es éste. `None` = sin sellar: ninguna
#: medida real vale. La sella el supervisor; el sha256 del borrador está en ESTADO.md.
REGLA_SHA256_SELLADA: str | None = "ee2d31e66cf56df11e9fad4e6ece9ef63dc165ae110f0035e943b87958550242"


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia. Mismas etiquetas que `medir_c163_c2.dependencias_sha256`."""
    return {"benchmarks/fase0/protocolo.py": _sha256(Path(protocolo_mod.__file__).resolve()),
            "benchmarks/fase0/pasada_amplia_101_c5.py": _sha256(Path(c5.__file__).resolve()),
            "matrixai/estudio/metricas.py": _sha256(Path(_metricas.__file__).resolve()),
            "matrixai/estudio/incertidumbre.py": _sha256(Path(_incertidumbre.__file__).resolve()),
            "benchmarks/c160/analiza_c160.py": _sha256(Path(a160.__file__).resolve())}


# ----------------------------------------------------------------------------------------------------------------------
# 1. UN CONJUNTO, UN MOTOR
# ----------------------------------------------------------------------------------------------------------------------
def metrica_del_registro(r: dict) -> float | None:
    v = r.get(r.get("metrica_de_cierre"))
    return None if v is None else float(v)


def declaracion_de_la_rejilla(r: dict) -> dict | None:
    return (r.get("hiperparametros") or {}).get("rejilla_por_validacion")


def coste_del_registro(r: dict) -> float | None:
    """Coste de AJUSTAR de un intento: `tiempo_de_ajuste` (los segundos de pared que el motor declara de su `fit`, rejilla entera incluida).
    No el `wall_s` del subproceso (lleva arrancar Python y volcar el modelo, un fijo que achata la razón)."""
    v = r.get("tiempo_de_ajuste")
    return None if v is None else float(v)


def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, motor: str, registros: dict[int, dict[str, dict]], esperados: list[int],
                     saturados: tuple[str, ...] = SATURADOS) -> dict:
    """UN conjunto y UN motor: por pliegue la métrica de sus dos brazos; R, su IC y su clasificación; coste; qué configuración eligió la
    rejilla. `registros[pliegue][brazo]`."""
    ref, rej = REFERENCIA[motor], REJILLA[motor]

    def ok(f):
        return f in registros and all(b in registros[f] and metrica_del_registro(registros[f][b]) is not None
                                      and coste_del_registro(registros[f][b]) is not None for b in (ref, rej))
    faltan = [f for f in esperados if not ok(f)]
    pliegues = [f for f in esperados if f not in faltan]
    por_pliegue = {f: {b: metrica_del_registro(registros[f][b]) for b in (ref, rej)} for f in pliegues}
    out = {"nombre": nombre, "tarea": tarea, "metrica": metric_id, "motor": motor, "saturado": nombre in saturados,
           "pliegues": pliegues, "pliegues_que_faltan": faltan, "completo": not faltan and bool(pliegues), "por_pliegue": por_pliegue}
    if not pliegues:
        return out
    e_ref = [a160.error_de(por_pliegue[f][ref]) for f in pliegues]
    e_rej = [a160.error_de(por_pliegue[f][rej]) for f in pliegues]
    r, ic = a160.reduccion_relativa(e_ref, e_rej), a160.ic_de_r(e_ref, e_rej)
    coste = {b: sum(coste_del_registro(registros[f][b]) for f in pliegues) for b in (ref, rej)}
    elegidas = [(declaracion_de_la_rejilla(registros[f][rej]) or {}).get("elegida") for f in pliegues]
    out.update(
        medias={b: statistics.fmean(por_pliegue[f][b] for f in pliegues) for b in (ref, rej)},
        error_referencia=statistics.fmean(e_ref), error_rejilla=statistics.fmean(e_rej),
        r=r, ic=list(ic) if ic else None, **a160.clasificar(r, ic),
        coste_s=coste, coste_ratio=(coste[rej] / coste[ref]) if coste[ref] > 0 else None,
        elegidas_por_pliegue=elegidas, se_movio_de_hoy=sum(1 for e in elegidas if e not in (None, 0)))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 2. LA REGLA (por motor)
# ----------------------------------------------------------------------------------------------------------------------
def aplicar_la_regla(evaluados: list[dict], *, confirmacion: bool = False) -> dict:
    """(a)–(d) para UN motor. (a) y (b) sobre los NO saturados; (c) y (d) sobre TODOS los medidos. Un conjunto incompleto no se salta:
    `aplicable=False`. En la confirmación (a) y (c) y (d) igual y (b) pide la mitad (hacia arriba) de los no saturados."""
    incompletos = [e["nombre"] for e in evaluados if not e["completo"]]
    medidos = [e for e in evaluados if e["completo"]]
    no_sat = [e for e in medidos if not e["saturado"]]
    r_ns = [e["r"] for e in no_sat]
    a_media = statistics.fmean(r_ns) if r_ns else None
    mejoras = [e["nombre"] for e in no_sat if e["mejora_clara"]]
    mejoras_sat = [e["nombre"] for e in medidos if e["saturado"] and e["mejora_clara"]]
    empeoran = [e["nombre"] for e in medidos if e["empeora_de_verdad"]]
    motor = medidos[0]["motor"] if medidos else None
    ref, rej = (REFERENCIA[motor], REJILLA[motor]) if motor else (None, None)
    suma = {b: sum(e["coste_s"][b] for e in medidos) for b in (ref, rej)} if motor else {}
    ratios = [e["coste_ratio"] for e in medidos if e["coste_ratio"] is not None]
    d_ratio = suma[rej] / suma[ref] if (motor and suma[ref] > 0) else None
    b_min = -(-len(no_sat) // 2) if confirmacion else B_MEJORAS_CLARAS_MIN
    a_ok = a_media is not None and a_media >= A_REDUCCION_MEDIA_MIN
    b_ok = len(mejoras) >= b_min and bool(no_sat)
    c_ok = not empeoran
    d_ok = d_ratio is not None and d_ratio <= D_COSTE_MAXIMO
    return {"motor": motor, "aplicable": not incompletos and bool(medidos), "incompletos": incompletos, "confirmacion": confirmacion,
            "no_saturados": [e["nombre"] for e in no_sat],
            "a": {"media_r_no_saturados": a_media, "n": len(r_ns), "umbral": A_REDUCCION_MEDIA_MIN, "cumple": a_ok},
            "b": {"mejoras_claras_no_saturados": mejoras, "n": len(mejoras), "umbral": b_min, "cumple": b_ok,
                  "mejoras_claras_saturados_que_no_cuentan": mejoras_sat},
            "c": {"empeoran": empeoran, "umbral": C_EMPEORA_MAS_DE, "cumple": c_ok},
            "d": {"razon_de_costes": d_ratio, "mediana_por_conjunto": statistics.median(ratios) if ratios else None,
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok},
            "cumple": bool(not incompletos and medidos and a_ok and b_ok and c_ok and d_ok)}


# ----------------------------------------------------------------------------------------------------------------------
# 3. COMPLETITUD, SELLOS Y LA MEDIDA ENTERA
# ----------------------------------------------------------------------------------------------------------------------
def problemas_de_la_declaracion(r: dict) -> list[str]:
    """«Declarar lo que PASÓ»: cada brazo declara lo suyo. La referencia lleva la política por tamaño aplicada y NINGUNA rejilla; la rejilla
    declara 8 configuraciones medidas SOLO con la validación, y que la elegida es la de menor pérdida."""
    k = (r.get("dataset"), r.get("motor"), r.get("pliegue"))
    d = declaracion_de_la_rejilla(r)
    pol = ((r.get("hiperparametros") or {}).get("politica_por_tamano")) or {}
    p = []
    es_rejilla = r.get("motor") in REJILLA.values()
    if pol.get("aplicada") is not True:
        p.append(f"el intento {k} no declara la política por tamaño aplicada (no mide lo declarado)")
    if not es_rejilla:
        if d is not None:
            p.append(f"el intento {k} es de la REFERENCIA y declara `rejilla_por_validacion`: brazos cruzados")
        if "validacion_mejor" in pol:
            p.append(f"el intento {k} es de la REFERENCIA y declara `validacion_mejor` (solo la rejilla lo hace)")
        return p
    if d is None:
        return p + [f"el intento {k} es de la REJILLA y no declara `rejilla_por_validacion`: brazos cruzados"]
    if d.get("aplicada") is not True:
        return p + [f"el intento {k}: la rejilla NO se aplicó ({d.get('motivo')}): no mide lo declarado"]
    if d.get("mira_la_prueba") is not False or d.get("criterio") != "perdida_en_validation_del_pliegue":
        p.append(f"el intento {k}: la rejilla no declara elegir solo por la validación")
    cfgs = d.get("configuraciones") or []
    if len(cfgs) != N_CONFIGURACIONES:
        return p + [f"el intento {k}: la rejilla declara {len(cfgs)} configuraciones, no {N_CONFIGURACIONES}"]
    perdidas = [c.get("validacion_mejor") for c in cfgs]
    if any(v is None for v in perdidas):
        return p + [f"el intento {k}: una configuración de la rejilla no trae su pérdida de validación"]
    if d.get("elegida") != perdidas.index(min(perdidas)):
        p.append(f"el intento {k}: la elegida ({d.get('elegida')}) no es la de menor pérdida de validación")
    return p


def problemas_de_completitud(payload: dict, es_sonda: bool, confirmacion: bool = False) -> list[str]:
    problemas: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    duplicados = sorted({n for n in nombres if nombres.count(n) > 1})
    if duplicados:
        problemas.append(f"conjuntos DUPLICADOS en la salida: {duplicados}")
    if not es_sonda and not confirmacion:
        congelado = a160.conjuntos_y_pliegues_del_congelado()
        faltan, sobran = sorted(set(congelado) - set(nombres)), sorted(set(nombres) - set(congelado))
        if faltan or sobran:
            problemas.append(f"los conjuntos no son los {N_CONJUNTOS} del congelado: faltan {faltan}, sobran {sobran}")
        for c in conjuntos:
            esperados = congelado.get(c.get("nombre"), {}).get("pliegues")
            if esperados is not None and list(c.get("pliegues_esperados") or []) != esperados:
                problemas.append(f"{c.get('nombre')}: pliegues declarados {c.get('pliegues_esperados')} != los de la partición {esperados}")
    esperado = {(c.get("nombre"), b, int(f)) for c in conjuntos for b in BRAZOS for f in (c.get("pliegues_esperados") or [])}
    vistos: dict[tuple, int] = {}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("motor"), int(r.get("pliegue", -1)))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("repeticion") != 0:
            problemas.append(f"un registro de la repetición {r.get('repeticion')} ({k}): la regla es sobre la 0")
        if r.get("estado") not in ESTADOS_DE_MEDIDA:
            problemas.append(f"el intento {k} no cuenta como medida (estado {r.get('estado')!r}): falta ese pliegue")
            continue
        if metrica_del_registro(r) is None:
            problemas.append(f"el intento {k} no trae su métrica de cierre ({r.get('metrica_de_cierre')!r})")
        if coste_del_registro(r) is None:
            problemas.append(f"el intento {k} no trae `tiempo_de_ajuste` (el coste de (d))")
        problemas += problemas_de_la_declaracion(r)
    faltan_i, sobran_i = sorted(esperado - set(vistos)), sorted(set(vistos) - esperado)
    dobles = sorted(k for k, n in vistos.items() if n > 1)
    if faltan_i:
        problemas.append(f"faltan {len(faltan_i)} intentos (p. ej. {faltan_i[:3]})")
    if sobran_i:
        problemas.append(f"{len(sobran_i)} registros fuera de lo declarado (p. ej. {sobran_i[:3]})")
    if dobles:
        problemas.append(f"{len(dobles)} intentos registrados más de una vez (p. ej. {dobles[:3]})")
    return problemas


def analizar(payload: dict, *, comprobar_sellos: bool = True, confirmacion: bool = False) -> dict:
    """El veredicto de una salida de `medir_c163_c2.py`. Por motor: CUMPLE / NO CUMPLE; global: SONDA / SIN VEREDICTO si algo no cuadra."""
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    conjuntos = payload.get("conjuntos", [])
    if not es_sonda and not confirmacion and len(conjuntos) != N_CONJUNTOS:
        problemas.append(f"hay {len(conjuntos)} conjuntos, la regla es sobre {N_CONJUNTOS}")
    if any(c.get("sellado") for c in conjuntos):
        problemas.append("hay un conjunto SELLADO en la medida")
    problemas += problemas_de_completitud(payload, es_sonda, confirmacion)
    rs = payload.get("resultados", [])
    for brazo in BRAZOS:
        for campo in ("entorno_digest", "motor_digest"):
            if len({r.get(campo) for r in rs if r.get("motor") == brazo}) > 1:
                problemas.append(f"los intentos del brazo {brazo} se midieron con CÓDIGO DISTINTO ({campo})")
    if len({r.get("entorno_digest") for r in rs}) > 1:
        problemas.append("los brazos se midieron con CÓDIGO DISTINTO (entorno_digest)")
    sellos = {"regla_sha256": _sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None,
              "analizador_sha256": _sha256(RUTA_DEL_ANALIZADOR), "dependencias_sha256": dependencias_de_ahora()}
    if comprobar_sellos:
        if REGLA_SHA256_SELLADA is None:
            problemas.append("la regla NO está sellada (`REGLA_SHA256_SELLADA` es None): ninguna medida vale todavía")
        elif sellos["regla_sha256"] != REGLA_SHA256_SELLADA:
            problemas.append(f"REGLA_C2.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
        al_medir = payload.get("dependencias_sha256_al_medir")
        if not isinstance(al_medir, dict):
            problemas.append("la salida no declara `dependencias_sha256_al_medir`")
        else:
            for etiqueta, propio in sellos["dependencias_sha256"].items():
                if al_medir.get(etiqueta) != propio:
                    problemas.append(f"`{etiqueta}` al medir ({str(al_medir.get(etiqueta))[:12]}) != el que importa este analizador ({propio[:12]})")
        for clave_, propio in (("regla_sha256_al_medir", sellos["regla_sha256"]), ("analizador_sha256_al_medir", sellos["analizador_sha256"])):
            valor = payload.get(clave_)
            if valor is None:
                problemas.append(f"la salida no declara `{clave_}`")
            elif valor != propio:
                problemas.append(f"`{clave_}` de la salida ({valor[:12]}) != el fichero de ahora ({str(propio)[:12]}): cambió DESPUÉS de medir")
    saturados = tuple(payload.get("saturados_declarados") or ()) if confirmacion else SATURADOS
    por_clave = {(r["dataset"], r["motor"], int(r["pliegue"])): r for r in rs if r.get("estado") in ESTADOS_DE_MEDIDA}
    por_motor: dict[str, list[dict]] = {m: [] for m in MOTORES}
    for c in conjuntos:
        esperados = [int(f) for f in c.get("pliegues_esperados") or []]
        for m in MOTORES:
            registros: dict[int, dict[str, dict]] = {}
            for f in esperados:
                for b in (REFERENCIA[m], REJILLA[m]):
                    r = por_clave.get((c["nombre"], b, f))
                    if r is not None:
                        registros.setdefault(f, {})[b] = r
            por_motor[m].append(evaluar_conjunto(c["nombre"], c["tarea"], c["metrica_de_cierre"], m, registros, esperados, saturados))
    for m in MOTORES:
        for e in por_motor[m]:
            if not e["completo"]:
                problemas.append(f"{m} {e['nombre']}: faltan pliegues {e['pliegues_que_faltan']}")
    reglas = {m: aplicar_la_regla(por_motor[m], confirmacion=confirmacion) if por_motor[m] else None for m in MOTORES}
    if problemas:
        veredicto, por_motor_v = "SIN VEREDICTO", {m: "SIN VEREDICTO" for m in MOTORES}
    elif es_sonda:
        veredicto, por_motor_v = "SONDA", {m: "SONDA" for m in MOTORES}
    else:
        por_motor_v = {m: ("CUMPLE" if reglas[m]["cumple"] else "NO CUMPLE") for m in MOTORES}
        veredicto = "CUMPLE (%s)" % ", ".join(m for m in MOTORES if por_motor_v[m] == "CUMPLE") \
            if "CUMPLE" in por_motor_v.values() else "NO CUMPLE"
    return {"veredicto": veredicto, "veredicto_por_motor": por_motor_v, "problemas": problemas, "es_sonda": es_sonda,
            "sellos_de_ahora": sellos, "reglas": reglas, "conjuntos": por_motor}


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def informe_en_texto(r: dict) -> str:
    L = [f"=== 163-C2: rejilla de 8 configuraciones por validación contra la política de hoy — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    for m in MOTORES:
        g = (r.get("reglas") or {}).get(m)
        if not g:
            continue
        L.append(f"--- {m}: {r['veredicto_por_motor'][m]} ---")
        L.append(f"  (a) media de R sobre {g['a']['n']} no saturados: {_f(g['a']['media_r_no_saturados'], 4)} (>= {g['a']['umbral']}) "
                 f"-> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras entre no saturados: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras_no_saturados']} "
                 f"-> {'ok' if g['b']['cumple'] else 'NO'}")
        L.append(f"  (c) empeoran de verdad (> {g['c']['umbral']:.0%}, IC < 0): {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) coste de ajustar, razón de sumas: {_f(g['d']['razon_de_costes'], 2)} (<= {g['d']['umbral']}); mediana "
                 f"{_f(g['d']['mediana_por_conjunto'], 2)}, máxima {_f(g['d']['maxima_por_conjunto'], 2)} -> {'ok' if g['d']['cumple'] else 'NO'}")
        L.append(f"  {'conjunto':34s} sat     R        IC95                 mejora empeora  se movió de hoy (pliegues)")
        for e in r["conjuntos"][m]:
            if "r" not in e:
                L.append(f"  {e['nombre']:34s} (sin medida)")
                continue
            ic = e["ic"]
            L.append(f"  {e['nombre']:34s} {'S' if e['saturado'] else ' '}   {e['r']:+8.4f}  "
                     f"[{_f(ic[0], 3) if ic else '—'}, {_f(ic[1], 3) if ic else '—'}]   {'sí' if e['mejora_clara'] else '  '}    "
                     f"{'SÍ' if e['empeora_de_verdad'] else '  '}     {e['se_movio_de_hoy']}/{len(e['pliegues'])}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    p.add_argument("--confirmacion", action="store_true", help="aplica la regla escalada de REGLA_C2.md §5 (conjuntos de CONFIRMACION_C2.md)")
    a = p.parse_args(argv)
    payload = json.loads(Path(a.resultado).resolve().read_text(encoding="utf-8"))
    r = analizar(payload, confirmacion=a.confirmacion)
    print(informe_en_texto(r))
    print(f"  sellos de ahora: regla {str(r['sellos_de_ahora']['regla_sha256'])[:16]} analizador {r['sellos_de_ahora']['analizador_sha256'][:16]}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if r["veredicto"].startswith("CUMPLE") or r["veredicto"] == "SONDA":
        return 0
    return 1 if r["veredicto"] == "NO CUMPLE" else 2


if __name__ == "__main__":
    sys.exit(main())
