#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""163-C1 — EL ANALIZADOR: LightGBM con `min_data_per_group` POR TAMAÑO contra LightGBM de hoy (100 por omisión), con la regla de
`REGLA_C1.md` (contrato 163, decisión de Roberto del 10-11 —tarjeta `lightgbm-min-data-per-group` = A— y D1–D4 del 12-11).

    python3 benchmarks/c163/analiza_c163_c1.py <resultado.json> [--json <salida.json>] [--confirmacion]

Sale con 0 si CUMPLE, 1 si NO CUMPLE, 2 si no hay veredicto (medida incompleta, código mezclado, regla o analizador distintos de los que
midieron, dependencias distintas, el control de los conjuntos que la palanca no puede tocar NO da lo mismo), 0 en una SONDA.

NO reescribe nada del 160: importa `analiza_c160` (reducción relativa, IC95 por bootstrap de pliegues, clasificación mejora/empeora y los
23 conjuntos del congelado). La palanca solo cambia algo en los conjuntos con columnas categóricas Y menos de 2.000 filas de train
(`min(100, max(5, filas // 20))` vale 100 desde 2.000): esos son los AFECTADOS, declarados abajo ANTES de medir; en los demás la palanca debe
dar EXACTAMENTE lo mismo y eso se comprueba (control), no se supone.
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
# LOS UMBRALES DE LA REGLA. Un número por línea; las pruebas los fijan contra REGLA_C1.md.
# ----------------------------------------------------------------------------------------------------------------------
N_CONJUNTOS = a160.N_CONJUNTOS                           # 23
A_REDUCCION_MEDIA_MIN = a160.A_REDUCCION_MEDIA_MIN       # (a) media de R sobre los AFECTADOS >= +3 %
B_MEJORAS_CLARAS_MIN = 2                                 # (b) afectados con «mejora clara» >= 2 de 3 (la mitad hacia arriba)
C_EMPEORA_MAS_DE = a160.C_EMPEORA_MAS_DE                 # (c) ningún conjunto de los 23 con R < -3 % y el IC95 < 0
D_COSTE_MAXIMO = 1.5                                     # (d) coste de la palanca <= 1,5 x el de la referencia (suma de los 23)
E_CONTROL_TOLERANCIA = 1e-9                              # (e) fuera de los afectados, |métrica palanca - métrica referencia| <= 1e-9
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
#: Mínimo de afectados para poder aplicar la regla en una CONFIRMACIÓN (sin ellos no hay nada que confirmar).
CONFIRMACION_AFECTADOS_MIN = 2

#: LOS AFECTADOS, medidos el 12-11 (23:07) ANTES de ver ninguna cifra de calidad: de los 23, tienen columnas categóricas Y menos de 2.000
#: filas de train (tras la preparación de la Fase 0, pliegue 0): n_train y categóricas (cardinalidades) entre paréntesis.
AFECTADOS = ("dresses-sales", "Moneyball", "house_prices_nominal")      # 319 (11), 788 (2), 934 (43)
#: Con categóricas pero >= 2.000 filas de train: la fórmula da 100 = la omisión. Efecto nulo POR CONSTRUCCIÓN (se controla).
CON_CATEGORICAS_SIN_EFECTO = ("sick", "SAT11-HAND-runtime-regression")   # 2.414 (23), 2.841 (1)
#: Los 18 restantes no traen ninguna columna categórica tras la preparación: la palanca no pasa ni el parámetro.
SATURADOS = a160.SATURADOS

#: Los dos brazos. La REFERENCIA primero: es el denominador de R.
REFERENCIA = "lightgbm_hoy"
PALANCA = "lightgbm_min_data_per_group"
BRAZOS = (REFERENCIA, PALANCA)

RUTA_DE_LA_REGLA = _AQUI / "REGLA_C1.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()

#: sha256 de `REGLA_C1.md` SELLADA. `medir_c163_c1.py` la repite y se niega a medir si el fichero ya no es éste. `None` = sin sellar: ninguna
#: medida real vale. El sellado definitivo lo pone el supervisor; el sha256 del borrador está en ESTADO.md.
REGLA_SHA256_SELLADA: str | None = "1777bf1736951d980e8b05898f3e5c11954161ac79b19bea936105360d853663"


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia. Mismas etiquetas que `medir_c163_c1.dependencias_sha256`."""
    return {"benchmarks/fase0/protocolo.py": _sha256(Path(protocolo_mod.__file__).resolve()),
            "benchmarks/fase0/pasada_amplia_101_c5.py": _sha256(Path(c5.__file__).resolve()),
            "matrixai/estudio/metricas.py": _sha256(Path(_metricas.__file__).resolve()),
            "matrixai/estudio/incertidumbre.py": _sha256(Path(_incertidumbre.__file__).resolve()),
            "benchmarks/c160/analiza_c160.py": _sha256(Path(a160.__file__).resolve())}


# ----------------------------------------------------------------------------------------------------------------------
# 1. UN CONJUNTO
# ----------------------------------------------------------------------------------------------------------------------
def metrica_del_registro(r: dict) -> float | None:
    v = r.get(r.get("metrica_de_cierre"))
    return None if v is None else float(v)


def _declaracion_de_la_palanca(r: dict) -> dict | None:
    return ((r.get("hiperparametros") or {}).get("min_data_per_group_por_tamano"))


def aplicada_en(r: dict) -> bool:
    """La palanca ACTUÓ en ese intento: pedida, aplicada (hay categóricas) y con un valor DISTINTO de la omisión de LightGBM (100)."""
    d = _declaracion_de_la_palanca(r)
    return bool(d and d.get("pedida") is True and d.get("aplicada") is True and d.get("min_data_per_group") != 100)


def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, registros: dict[int, dict[str, dict]], esperados: list[int],
                     saturados: tuple[str, ...] = SATURADOS, afectados: tuple[str, ...] = AFECTADOS) -> dict:
    """UN conjunto: por pliegue la métrica de cada brazo; R, su IC y su clasificación; coste; control. `registros[pliegue][brazo]`."""
    def ok(f):
        return f in registros and all(b in registros[f] and metrica_del_registro(registros[f][b]) is not None for b in BRAZOS)
    faltan = [f for f in esperados if not ok(f)]
    pliegues = [f for f in esperados if f not in faltan]
    por_pliegue = {f: {b: metrica_del_registro(registros[f][b]) for b in BRAZOS} for f in pliegues}
    out = {"nombre": nombre, "tarea": tarea, "metrica": metric_id, "saturado": nombre in saturados, "afectado": nombre in afectados,
           "pliegues": pliegues, "pliegues_que_faltan": faltan, "completo": not faltan and bool(pliegues), "por_pliegue": por_pliegue}
    if not pliegues:
        return out
    e_ref = [a160.error_de(por_pliegue[f][REFERENCIA]) for f in pliegues]
    e_pal = [a160.error_de(por_pliegue[f][PALANCA]) for f in pliegues]
    r, ic = a160.reduccion_relativa(e_ref, e_pal), a160.ic_de_r(e_ref, e_pal)
    coste = {b: sum(float(registros[f][b].get("wall_s") or 0.0) for f in pliegues) for b in BRAZOS}
    diferencia_max = max(abs(por_pliegue[f][PALANCA] - por_pliegue[f][REFERENCIA]) for f in pliegues)
    declaradas = [_declaracion_de_la_palanca(registros[f][PALANCA]) or {} for f in pliegues]
    out.update(
        medias={b: statistics.fmean(por_pliegue[f][b] for f in pliegues) for b in BRAZOS},
        error_referencia=statistics.fmean(e_ref), error_palanca=statistics.fmean(e_pal),
        r=r, ic=list(ic) if ic else None, **a160.clasificar(r, ic),
        coste_s=coste, coste_ratio=(coste[PALANCA] / coste[REFERENCIA]) if coste[REFERENCIA] > 0 else None,
        diferencia_maxima_entre_brazos=diferencia_max,
        aplicada_en_pliegues=[f for f in pliegues if aplicada_en(registros[f][PALANCA])],
        min_data_per_group_por_pliegue=[d.get("min_data_per_group") for d in declaradas],
        filas_de_train_por_pliegue=[d.get("filas_de_train") for d in declaradas])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 2. LA REGLA
# ----------------------------------------------------------------------------------------------------------------------
def problemas_del_control(evaluados: list[dict], *, confirmacion: bool = False) -> list[str]:
    """(e) El CONTROL: donde la palanca no aplica (sin categóricas, o con >= 2.000 filas) tiene que dar EXACTAMENTE lo mismo. Si no, el
    instrumento no es determinista o la palanca actúa donde no debe: SIN VEREDICTO, no un empeora/mejora."""
    p = []
    for e in evaluados:
        if not e["completo"] or e["aplicada_en_pliegues"]:
            continue
        if e["diferencia_maxima_entre_brazos"] > E_CONTROL_TOLERANCIA:
            p.append(f"{e['nombre']}: la palanca NO se aplicó y aun así los brazos difieren en {e['diferencia_maxima_entre_brazos']:.3g} "
                     f"(> {E_CONTROL_TOLERANCIA:g}): control roto")
    return p


def afectados_de_la_medida(evaluados: list[dict], *, confirmacion: bool) -> list[dict]:
    """Quiénes entran en (a) y (b). En la medida de los 23: los DECLARADOS de arriba. En una confirmación: los que la propia medida
    declara con la palanca aplicada (ninguno se conoce de antemano), con un mínimo."""
    if confirmacion:
        return [e for e in evaluados if e["completo"] and e["aplicada_en_pliegues"]]
    return [e for e in evaluados if e["completo"] and e["afectado"]]


def aplicar_la_regla(evaluados: list[dict], *, confirmacion: bool = False) -> dict:
    """(a)–(d) sobre los AFECTADOS; (c) y (d) sobre TODOS los conjuntos medidos. Un conjunto incompleto no se salta: `aplicable=False`."""
    incompletos = [e["nombre"] for e in evaluados if not e["completo"]]
    medidos = [e for e in evaluados if e["completo"]]
    af = afectados_de_la_medida(evaluados, confirmacion=confirmacion)
    r_af = [e["r"] for e in af]
    a_media = statistics.fmean(r_af) if r_af else None
    mejoras = [e["nombre"] for e in af if e["mejora_clara"]]
    empeoran = [e["nombre"] for e in medidos if e["empeora_de_verdad"]]
    suma = {b: sum(e["coste_s"][b] for e in medidos) for b in BRAZOS}
    ratios = [e["coste_ratio"] for e in medidos if e["coste_ratio"] is not None]
    d_ratio = suma[PALANCA] / suma[REFERENCIA] if (medidos and suma[REFERENCIA] > 0) else None
    b_min = -(-len(af) // 2) if confirmacion else B_MEJORAS_CLARAS_MIN
    suficientes = len(af) >= (CONFIRMACION_AFECTADOS_MIN if confirmacion else len(AFECTADOS))
    a_ok = a_media is not None and a_media >= A_REDUCCION_MEDIA_MIN
    b_ok = len(mejoras) >= b_min and bool(af)
    c_ok = not empeoran
    d_ok = d_ratio is not None and d_ratio <= D_COSTE_MAXIMO
    return {"aplicable": not incompletos and suficientes, "incompletos": incompletos, "confirmacion": confirmacion,
            "afectados": [e["nombre"] for e in af], "afectados_suficientes": suficientes,
            "a": {"media_r_afectados": a_media, "n": len(r_af), "umbral": A_REDUCCION_MEDIA_MIN, "cumple": a_ok},
            "b": {"mejoras_claras_afectados": mejoras, "n": len(mejoras), "umbral": b_min, "cumple": b_ok},
            "c": {"empeoran": empeoran, "umbral": C_EMPEORA_MAS_DE, "cumple": c_ok},
            "d": {"razon_de_costes": d_ratio, "mediana_por_conjunto": statistics.median(ratios) if ratios else None,
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok},
            "cumple": bool(not incompletos and suficientes and a_ok and b_ok and c_ok and d_ok)}


# ----------------------------------------------------------------------------------------------------------------------
# 3. COMPLETITUD, SELLOS Y LA MEDIDA ENTERA
# ----------------------------------------------------------------------------------------------------------------------
def problemas_de_completitud(payload: dict, es_sonda: bool, confirmacion: bool = False) -> list[str]:
    """«Si falta algún conjunto/pliegue/brazo: SIN VEREDICTO», comprobado y no supuesto; y cada brazo declara SU palanca."""
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
                problemas.append(f"{c.get('nombre')}: pliegues declarados {c.get('pliegues_esperados')} != los de la partición "
                                 f"{esperados}")
    esperado = {(c.get("nombre"), b, int(f)) for c in conjuntos for b in BRAZOS for f in (c.get("pliegues_esperados") or [])}
    vistos: dict[tuple, int] = {}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("motor"), int(r.get("pliegue", -1)))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("repeticion") != 0:
            problemas.append(f"un registro de la repetición {r.get('repeticion')} ({k}): la regla es sobre la 0")
        if r.get("estado") not in ESTADOS_DE_MEDIDA:
            problemas.append(f"el intento {k} no cuenta como medida (estado {r.get('estado')!r}): falta ese pliegue")
        elif metrica_del_registro(r) is None:
            problemas.append(f"el intento {k} no trae su métrica de cierre ({r.get('metrica_de_cierre')!r})")
        # DECLARAR LO QUE PASÓ: cada brazo declara SU palanca. Un brazo cruzado se lee igual de bien que uno correcto.
        d = _declaracion_de_la_palanca(r)
        if r.get("motor") == REFERENCIA and d is not None:
            problemas.append(f"el intento {k} es de la REFERENCIA y declara `min_data_per_group_por_tamano`: brazos cruzados")
        if r.get("motor") == PALANCA:
            if d is None or d.get("pedida") is not True:
                problemas.append(f"el intento {k} es de la PALANCA y no declara haberla pedido: brazos cruzados")
            elif r.get("dataset") in AFECTADOS and not confirmacion and not aplicada_en(r):
                problemas.append(f"el intento {k} es de un AFECTADO y la palanca NO actuó (¿sin categóricas, o >= 2.000 filas?): "
                                 f"no mide lo declarado")
            elif r.get("dataset") not in AFECTADOS and not confirmacion and aplicada_en(r):
                problemas.append(f"el intento {k} NO es de un afectado y la palanca actuó con {d.get('min_data_per_group')}: "
                                 f"la lista de afectados no cuadra con lo medido")
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
    """El veredicto de una salida de `medir_c163_c1.py`. `veredicto` ∈ CUMPLE / NO CUMPLE / SONDA / SIN VEREDICTO."""
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
    for campo in ("entorno_digest", "motor_digest"):
        for brazo in BRAZOS:
            if len({r.get(campo) for r in rs if r.get("motor") == brazo}) > 1:
                problemas.append(f"los intentos del brazo {brazo} se midieron con CÓDIGO DISTINTO ({campo})")
        if len({r.get(campo) for r in rs}) > 1:
            problemas.append(f"los dos brazos se midieron con CÓDIGO DISTINTO ({campo})")
    sellos = {"regla_sha256": _sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None,
              "analizador_sha256": _sha256(RUTA_DEL_ANALIZADOR), "dependencias_sha256": dependencias_de_ahora()}
    if comprobar_sellos:
        if REGLA_SHA256_SELLADA is None:
            problemas.append("la regla NO está sellada (`REGLA_SHA256_SELLADA` es None): ninguna medida vale todavía")
        elif sellos["regla_sha256"] != REGLA_SHA256_SELLADA:
            problemas.append(f"REGLA_C1.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
        al_medir = payload.get("dependencias_sha256_al_medir")
        if not isinstance(al_medir, dict):
            problemas.append("la salida no declara `dependencias_sha256_al_medir`")
        else:
            for etiqueta, propio in sellos["dependencias_sha256"].items():
                if al_medir.get(etiqueta) != propio:
                    problemas.append(f"`{etiqueta}` al medir ({str(al_medir.get(etiqueta))[:12]}) != el que importa este analizador "
                                     f"({propio[:12]})")
        for clave_, propio in (("regla_sha256_al_medir", sellos["regla_sha256"]),
                               ("analizador_sha256_al_medir", sellos["analizador_sha256"])):
            valor = payload.get(clave_)
            if valor is None:
                problemas.append(f"la salida no declara `{clave_}`")
            elif valor != propio:
                problemas.append(f"`{clave_}` de la salida ({valor[:12]}) != el fichero de ahora ({str(propio)[:12]}): cambió "
                                 f"DESPUÉS de medir")
    por_clave = {(r["dataset"], r["motor"], int(r["pliegue"])): r for r in rs if r.get("estado") in ESTADOS_DE_MEDIDA}
    evaluados = []
    for c in conjuntos:
        esperados = [int(f) for f in c.get("pliegues_esperados") or []]
        registros: dict[int, dict[str, dict]] = {}
        for f in esperados:
            for b in BRAZOS:
                r = por_clave.get((c["nombre"], b, f))
                if r is not None:
                    registros.setdefault(f, {})[b] = r
        evaluados.append(evaluar_conjunto(c["nombre"], c["tarea"], c["metrica_de_cierre"], registros, esperados))
    for e in evaluados:
        if not e["completo"]:
            problemas.append(f"{e['nombre']}: faltan pliegues {e['pliegues_que_faltan']}")
    problemas += problemas_del_control([e for e in evaluados if e["completo"]], confirmacion=confirmacion)
    regla = aplicar_la_regla(evaluados, confirmacion=confirmacion) if evaluados else None
    if regla is not None and not regla["afectados_suficientes"] and not problemas:
        problemas.append(f"hay {len(regla['afectados'])} conjuntos donde la palanca actúa; la regla pide "
                         f"{CONFIRMACION_AFECTADOS_MIN if confirmacion else len(AFECTADOS)}")
    if problemas:
        veredicto = "SIN VEREDICTO"
    elif es_sonda:
        veredicto = "SONDA"
    else:
        veredicto = "CUMPLE" if regla["cumple"] else "NO CUMPLE"
    return {"veredicto": veredicto, "problemas": problemas, "es_sonda": es_sonda, "sellos_de_ahora": sellos, "regla": regla,
            "conjuntos": evaluados}


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def informe_en_texto(r: dict) -> str:
    L = [f"=== 163-C1: LightGBM con min_data_per_group POR TAMAÑO contra el de HOY (100) — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  afectados (la palanca actúa): {g['afectados']}")
        L.append(f"  (a) media de R sobre {g['a']['n']} afectados: {_f(g['a']['media_r_afectados'], 4)} (>= {g['a']['umbral']}) "
                 f"-> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras entre afectados: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras_afectados']} "
                 f"-> {'ok' if g['b']['cumple'] else 'NO'}")
        L.append(f"  (c) empeoran de verdad (> {g['c']['umbral']:.0%}, IC < 0): {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) coste, razón de sumas: {_f(g['d']['razon_de_costes'], 3)} (<= {g['d']['umbral']}); mediana "
                 f"{_f(g['d']['mediana_por_conjunto'], 3)}, máxima {_f(g['d']['maxima_por_conjunto'], 3)} -> {'ok' if g['d']['cumple'] else 'NO'}")
    L.append(f"  {'conjunto':34s} afec     R        IC95                 mejora empeora  dif.máx")
    for e in r["conjuntos"]:
        if "r" not in e:
            L.append(f"  {e['nombre']:34s} (sin medida)")
            continue
        ic = e["ic"]
        L.append(f"  {e['nombre']:34s} {'A' if e['afectado'] else ' '}   {e['r']:+8.4f}  "
                 f"[{_f(ic[0], 3) if ic else '—'}, {_f(ic[1], 3) if ic else '—'}]   {'sí' if e['mejora_clara'] else '  '}    "
                 f"{'SÍ' if e['empeora_de_verdad'] else '  '}     {e['diferencia_maxima_entre_brazos']:.3g}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    p.add_argument("--confirmacion", action="store_true", help="aplica la regla escalada de REGLA_C1.md §5 (conjuntos de CONFIRMACION.md)")
    a = p.parse_args(argv)
    payload = json.loads(Path(a.resultado).resolve().read_text(encoding="utf-8"))
    r = analizar(payload, confirmacion=a.confirmacion)
    print(informe_en_texto(r))
    print(f"  sellos de ahora: regla {str(r['sellos_de_ahora']['regla_sha256'])[:16]} analizador {r['sellos_de_ahora']['analizador_sha256'][:16]}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return {"CUMPLE": 0, "NO CUMPLE": 1, "SONDA": 0}.get(r["veredicto"], 2)


if __name__ == "__main__":
    sys.exit(main())
