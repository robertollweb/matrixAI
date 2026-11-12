#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C1 — EL ANALIZADOR: TabM parando y entregando por la MÉTRICA DE CIERRE contra TabM parando por la PÉRDIDA, con la regla de
`REGLA_C1.md` (decisión D3 de Roberto, 12-11, tarjeta `ola5-39-decisiones` = A).

    python3 benchmarks/c162/analiza_c162_c1.py <resultado.json> [--json <salida.json>]

Sale con 0 si CUMPLE, 1 si NO CUMPLE, 2 si no hay veredicto (medida incompleta, código mezclado, regla o analizador distintos de los
que midieron, dependencias distintas), 0 en una SONDA (que no decide nada y lo dice).

NO reescribe nada del 160: importa `analiza_c160` (reducción relativa, IC95 por bootstrap de pliegues, clasificación mejora/empeora, los
23 conjuntos del congelado y el criterio de saturación recalculado de los artefactos) para que dos sitios no declaren lo mismo.
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
# LOS UMBRALES DE LA REGLA (D3). Un número por línea; las pruebas los fijan contra REGLA_C1.md.
# ----------------------------------------------------------------------------------------------------------------------
N_CONJUNTOS = a160.N_CONJUNTOS                           # 23
A_REDUCCION_MEDIA_MIN = a160.A_REDUCCION_MEDIA_MIN       # (a) media de R sobre los NO saturados >= +3 %
B_MEJORAS_CLARAS_MIN = a160.B_MEJORAS_CLARAS_MIN         # (b) conjuntos NO saturados con «mejora clara» >= 5
C_EMPEORA_MAS_DE = a160.C_EMPEORA_MAS_DE                 # (c) ningún conjunto (saturados INCLUIDOS) con R < -3 % y el IC95 < 0
D_COSTE_MAXIMO = 1.5                                     # (d) coste de la palanca <= 1,5 x el del intento de referencia
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
SATURADOS = a160.SATURADOS                               # los 6 del 160, declarados antes de medir
#: Confirmación (REGLA_C1.md §5): con 6 conjuntos, (b) es la mitad (redondeada hacia arriba) de los no saturados.
CONFIRMACION_N_CONJUNTOS = 6

#: Los dos brazos. La REFERENCIA primero: es el denominador de R.
REFERENCIA = "tabm_perdida"
PALANCA = "tabm_metrica"
BRAZOS = (REFERENCIA, PALANCA)

RUTA_DE_LA_REGLA = _AQUI / "REGLA_C1.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()

#: sha256 de `REGLA_C1.md` SELLADA. `medir_c162_c1.py` la repite y se niega a medir si el fichero ya no es éste. `None` = sin sellar: ninguna
#: medida real vale. Cambiar la regla es una enmienda con su sello, no una edición.
REGLA_SHA256_SELLADA: str | None = "272cc85f7b9cd58e2c6aa9509538614c782f45383a1cd686c50e8d13b6224c61"


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia. Mismas etiquetas que `medir_c162_c1.dependencias_sha256`."""
    return {"benchmarks/fase0/protocolo.py": _sha256(Path(protocolo_mod.__file__).resolve()),
            "benchmarks/fase0/pasada_amplia_101_c5.py": _sha256(Path(c5.__file__).resolve()),
            "matrixai/estudio/metricas.py": _sha256(Path(_metricas.__file__).resolve()),
            "matrixai/estudio/incertidumbre.py": _sha256(Path(_incertidumbre.__file__).resolve()),
            "benchmarks/c160/analiza_c160.py": _sha256(Path(a160.__file__).resolve())}


# ----------------------------------------------------------------------------------------------------------------------
# 1. UN CONJUNTO
# ----------------------------------------------------------------------------------------------------------------------
def metrica_del_registro(r: dict) -> float | None:
    """La métrica de cierre que el arnés midió sobre la prueba de ese intento (`registro[metric_id]`)."""
    v = r.get(r.get("metrica_de_cierre"))
    return None if v is None else float(v)


def _parada_declarada(r: dict) -> dict | None:
    return ((r.get("entrenamiento_efectivo") or {}).get("parada_por_metrica"))


def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, registros: dict[int, dict[str, dict]], esperados: list[int],
                     saturados: tuple[str, ...] = SATURADOS) -> dict:
    """UN conjunto: por pliegue la métrica de cada brazo; R, su IC y su clasificación; costes y declaraciones. `registros[pliegue][brazo]`.
    Un pliegue esperado sin los dos brazos (o sin su métrica) -> `completo=False` (SIN VEREDICTO arriba)."""
    def ok(f):
        return f in registros and all(b in registros[f] and metrica_del_registro(registros[f][b]) is not None for b in BRAZOS)
    faltan = [f for f in esperados if not ok(f)]
    pliegues = [f for f in esperados if f not in faltan]
    por_pliegue = {f: {b: metrica_del_registro(registros[f][b]) for b in BRAZOS} for f in pliegues}
    out = {"nombre": nombre, "tarea": tarea, "metrica": metric_id, "saturado": nombre in saturados, "pliegues": pliegues,
           "pliegues_que_faltan": faltan, "completo": not faltan and bool(pliegues), "por_pliegue": por_pliegue}
    if not pliegues:
        return out
    e_ref = [a160.error_de(por_pliegue[f][REFERENCIA]) for f in pliegues]
    e_pal = [a160.error_de(por_pliegue[f][PALANCA]) for f in pliegues]
    r, ic = a160.reduccion_relativa(e_ref, e_pal), a160.ic_de_r(e_ref, e_pal)
    coste = {b: sum(float(registros[f][b].get("wall_s") or 0.0) for f in pliegues) for b in BRAZOS}
    efectivo = {b: [(registros[f][b].get("entrenamiento_efectivo") or {}) for f in pliegues] for b in BRAZOS}
    caidas = [(f, (_parada_declarada(registros[f][PALANCA]) or {}).get("cae_a_la_perdida_por")) for f in pliegues
              if (_parada_declarada(registros[f][PALANCA]) or {}).get("efectiva") != "metrica"]
    out.update(
        medias={b: statistics.fmean(por_pliegue[f][b] for f in pliegues) for b in BRAZOS},
        error_referencia=statistics.fmean(e_ref), error_palanca=statistics.fmean(e_pal),
        r=r, ic=list(ic) if ic else None, **a160.clasificar(r, ic),
        coste_s=coste, coste_ratio=(coste[PALANCA] / coste[REFERENCIA]) if coste[REFERENCIA] > 0 else None,
        cae_a_la_perdida=[{"pliegue": f, "motivo": m} for f, m in caidas],
        epocas_ejecutadas={b: [e.get("epocas_ejecutadas") for e in efectivo[b]] for b in BRAZOS},
        mejor_epoca={b: [e.get("mejor_epoca") for e in efectivo[b]] for b in BRAZOS},
        parados_por_plazo={b: sum(1 for e in efectivo[b] if e.get("parado_por_plazo")) for b in BRAZOS},
        parados_por_paciencia={b: sum(1 for e in efectivo[b] if e.get("parado_por_paciencia")) for b in BRAZOS},
        misma_epoca_en={"n": sum(1 for e in zip(efectivo[REFERENCIA], efectivo[PALANCA])
                                 if e[0].get("mejor_epoca") == e[1].get("mejor_epoca")), "de": len(pliegues)})
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 2. LA REGLA
# ----------------------------------------------------------------------------------------------------------------------
def aplicar_la_regla(evaluados: list[dict], *, confirmacion: bool = False) -> dict:
    """(a)–(d). Los SATURADOS no entran en la media (a) ni en el recuento (b); SÍ en (c) y (d). Un conjunto incompleto no se salta: la regla
    no es aplicable (`aplicable=False`). `confirmacion=True`: (b) es la mitad (hacia arriba) de los no saturados (REGLA_C1.md §5)."""
    incompletos = [e["nombre"] for e in evaluados if not e["completo"]]
    medidos = [e for e in evaluados if e["completo"]]
    no_sat = [e for e in medidos if not e["saturado"]]
    r_no_sat = [e["r"] for e in no_sat]
    a_media = statistics.fmean(r_no_sat) if r_no_sat else None
    mejoras = [e["nombre"] for e in no_sat if e["mejora_clara"]]
    mejoras_saturados = [e["nombre"] for e in medidos if e["saturado"] and e["mejora_clara"]]
    empeoran = [e["nombre"] for e in medidos if e["empeora_de_verdad"]]
    suma = {b: sum(e["coste_s"][b] for e in medidos) for b in BRAZOS}
    ratios = [e["coste_ratio"] for e in medidos if e["coste_ratio"] is not None]
    d_ratio = suma[PALANCA] / suma[REFERENCIA] if (medidos and suma[REFERENCIA] > 0) else None
    b_min = -(-len(no_sat) // 2) if confirmacion else B_MEJORAS_CLARAS_MIN
    a_ok = a_media is not None and a_media >= A_REDUCCION_MEDIA_MIN
    b_ok = len(mejoras) >= b_min and bool(no_sat)
    c_ok = not empeoran
    d_ok = d_ratio is not None and d_ratio <= D_COSTE_MAXIMO
    return {"aplicable": not incompletos, "incompletos": incompletos, "confirmacion": confirmacion,
            "a": {"media_r_no_saturados": a_media, "n": len(r_no_sat), "umbral": A_REDUCCION_MEDIA_MIN, "cumple": a_ok},
            "b": {"mejoras_claras_no_saturados": mejoras, "n": len(mejoras), "umbral": b_min, "cumple": b_ok,
                  "informativo_mejoras_claras_en_saturados": mejoras_saturados},
            "c": {"empeoran": empeoran, "umbral": C_EMPEORA_MAS_DE, "cumple": c_ok},
            "d": {"razon_de_costes": d_ratio, "mediana_por_conjunto": statistics.median(ratios) if ratios else None,
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok},
            "cumple": bool(not incompletos and a_ok and b_ok and c_ok and d_ok)}


# ----------------------------------------------------------------------------------------------------------------------
# 3. COMPLETITUD, SELLOS Y LA MEDIDA ENTERA
# ----------------------------------------------------------------------------------------------------------------------
def problemas_de_completitud(payload: dict, es_sonda: bool) -> list[str]:
    """«Si falta algún conjunto/pliegue/brazo: SIN VEREDICTO», comprobado y no supuesto."""
    problemas: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    duplicados = sorted({n for n in nombres if nombres.count(n) > 1})
    if duplicados:
        problemas.append(f"conjuntos DUPLICADOS en la salida: {duplicados}")
    if not es_sonda:
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
        # DECLARAR LO QUE PASÓ: cada brazo declara SU parada. Un brazo cruzado se lee igual de bien que uno correcto.
        declarada = _parada_declarada(r)
        if r.get("motor") == REFERENCIA and declarada is not None:
            problemas.append(f"el intento {k} es de la REFERENCIA y declara `parada_por_metrica`: brazos cruzados")
        if r.get("motor") == PALANCA and (declarada is None or declarada.get("pedida") is not True):
            problemas.append(f"el intento {k} es de la PALANCA y no declara haber pedido la parada por métrica: brazos cruzados")
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
    """El veredicto de una salida de `medir_c162_c1.py`. `veredicto` ∈ CUMPLE / NO CUMPLE / SONDA / SIN VEREDICTO."""
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    conjuntos = payload.get("conjuntos", [])
    if not es_sonda and not confirmacion and len(conjuntos) != N_CONJUNTOS:
        problemas.append(f"hay {len(conjuntos)} conjuntos, la regla es sobre {N_CONJUNTOS}")
    if any(c.get("sellado") for c in conjuntos):
        problemas.append("hay un conjunto SELLADO en la medida")
    problemas += problemas_de_completitud(payload, es_sonda)
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
    regla = aplicar_la_regla(evaluados, confirmacion=confirmacion) if evaluados else None
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
    L = [f"=== 162-C1: TabM parando por la MÉTRICA de cierre contra parando por la PÉRDIDA — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  (a) media de R sobre {g['a']['n']} no saturados: {_f(g['a']['media_r_no_saturados'], 4)} (>= {g['a']['umbral']}) "
                 f"-> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras entre no saturados: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras_no_saturados']} "
                 f"-> {'ok' if g['b']['cumple'] else 'NO'}  [informativo, en saturados: {g['b']['informativo_mejoras_claras_en_saturados']}]")
        L.append(f"  (c) empeoran de verdad (> {g['c']['umbral']:.0%}, IC < 0): {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) coste (tiempo de intento), razón de sumas: {_f(g['d']['razon_de_costes'], 3)} (<= {g['d']['umbral']}); mediana "
                 f"{_f(g['d']['mediana_por_conjunto'], 3)}, máxima {_f(g['d']['maxima_por_conjunto'], 3)} -> {'ok' if g['d']['cumple'] else 'NO'}")
    L.append(f"  {'conjunto':34s} sat      R        IC95                 mejora empeora  caídas  plazo(p/m)")
    for e in r["conjuntos"]:
        if "r" not in e:
            L.append(f"  {e['nombre']:34s} (sin medida)")
            continue
        ic = e["ic"]
        L.append(f"  {e['nombre']:34s} {'S' if e['saturado'] else ' '}   {e['r']:+8.4f}  "
                 f"[{_f(ic[0], 3) if ic else '—'}, {_f(ic[1], 3) if ic else '—'}]   {'sí' if e['mejora_clara'] else '  '}    "
                 f"{'SÍ' if e['empeora_de_verdad'] else '  '}     {len(e['cae_a_la_perdida']):d}      "
                 f"{e['parados_por_plazo'][REFERENCIA]}/{e['parados_por_plazo'][PALANCA]}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    p.add_argument("--confirmacion", action="store_true", help="aplica la regla escalada de REGLA_C1.md §5 (6 conjuntos de CONFIRMACION.md)")
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
