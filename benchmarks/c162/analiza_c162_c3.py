#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C3 — EL ANALIZADOR: TabM-mini con entrada LINEAL A TROZOS (PLE, `con_variante("mini_ple")`) contra TabM `ref` (PLR, k = 8), en los 23
conjuntos no sellados, con la regla de `REGLA_C3.md` (idea 39, Ola 5): la del C1 (R >= +3 % en 17 no saturados, mejora clara en >= 5, ninguno
empeora) con (d) coste <= 1,5 x.

    python3 benchmarks/c162/analiza_c162_c3.py <resultado.json> [--json <salida.json>] [--confirmacion]

Sale con 0 si CUMPLE, 1 si NO CUMPLE, 2 si no hay veredicto, 0 en una SONDA (que no decide nada y lo dice).

NO reescribe nada del C1 ni del 160: importa `analiza_c162_c1` (la métrica de un registro, las dependencias, el hash, los saturados, los
estados que cuentan) y `analiza_c160` (reducción relativa, IC95 de pliegues, clasificación, los 23 del congelado). Lo propio de este
corte es solo: los brazos (`tabm_ref` / `tabm_mini_ple`), los umbrales (d) y qué declara cada brazo.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

import analiza_c162_c1 as an1  # noqa: E402  (primero: pone en `sys.path` el 160, la Fase 0 y el núcleo)
import analiza_c160 as a160  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# LOS UMBRALES (REGLA_C3.md). Un número por línea; las pruebas los fijan contra el fichero de la regla.
# ----------------------------------------------------------------------------------------------------------------------
A_REDUCCION_MEDIA_MIN = a160.A_REDUCCION_MEDIA_MIN     # (a) media de R sobre los 17 NO saturados >= +3 %
B_MEJORAS_CLARAS_MIN = an1.B_MEJORAS_CLARAS_MIN         # (b) «mejora clara» en >= 5 de los 17 no saturados
C_EMPEORA_MAS_DE = a160.C_EMPEORA_MAS_DE                # (c) ninguno de los 23 (saturados INCLUIDOS) con R < -3 % y el IC95 entero por debajo de 0
D_COSTE_MAXIMO = 1.5                                    # (d) Σ wall_s de la palanca <= 1,5 x el de la referencia (razón de sumas, los 23)
SATURADOS_DEL_CUBO = an1.SATURADOS
K_DE_LOS_DOS = 8                                        # los dos brazos son de k = 8 (el C2 es otra palanca)
N_CONJUNTOS_DEL_CUBO = a160.N_CONJUNTOS                 # 23 (el nombre del C2 se conserva: «los de la medida»)
#: Confirmación (REGLA_C3.md §5): (b) es la mitad (hacia arriba) de los no saturados.

REFERENCIA = "tabm_ref"
PALANCA = "tabm_mini_ple"
VARIANTE = "mini_ple"
N_BINS, D_EMBEDDING_NUMERICO = 48, 16     # los declarados por la red (ver `tabm_mini_ple.py`)
BRAZOS = (REFERENCIA, PALANCA)
ESTADOS_DE_MEDIDA = an1.ESTADOS_DE_MEDIDA

RUTA_DE_LA_REGLA = _AQUI / "REGLA_C3.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()

#: sha256 de `REGLA_C3.md` SELLADA. `medir_c162_c2.py` la repite y se niega a medir si el fichero ya no es éste. `None` = sin sellar.
REGLA_SHA256_SELLADA: str | None = "216e4b170af665b2f69755261ab2094d39477cc751fdd2501aa33e55278aebe6"


def conjuntos_del_cubo() -> dict[str, dict]:
    """Los 23 conjuntos no sellados, LEÍDOS del congelado (no escritos a mano): nombre -> {cubo, pliegues}."""
    return dict(a160.conjuntos_y_pliegues_del_congelado())


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia: las del C1 + el propio analizador del C1."""
    return {**an1.dependencias_de_ahora(),
            "benchmarks/c162/analiza_c162_c1.py": an1._sha256(Path(an1.__file__).resolve())}


def _declaracion_de_la_variante(r: dict) -> dict | None:
    return (r.get("entrenamiento_efectivo") or {}).get("variante")


def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, registros: dict[int, dict[str, dict]], esperados: list[int],
                     saturados: tuple[str, ...] = SATURADOS_DEL_CUBO) -> dict:
    """UN conjunto: por pliegue la métrica de cada brazo; R, su IC, su clasificación, costes y declaraciones.
    `registros[pliegue][brazo]`. Un pliegue esperado sin los dos brazos (o sin su métrica) -> `completo=False`."""
    def ok(f):
        return f in registros and all(b in registros[f] and an1.metrica_del_registro(registros[f][b]) is not None for b in BRAZOS)
    faltan = [f for f in esperados if not ok(f)]
    pliegues = [f for f in esperados if f not in faltan]
    por_pliegue = {f: {b: an1.metrica_del_registro(registros[f][b]) for b in BRAZOS} for f in pliegues}
    out = {"nombre": nombre, "tarea": tarea, "metrica": metric_id, "saturado": nombre in saturados, "pliegues": pliegues,
           "pliegues_que_faltan": faltan, "completo": not faltan and bool(pliegues), "por_pliegue": por_pliegue}
    if not pliegues:
        return out
    e_ref = [a160.error_de(por_pliegue[f][REFERENCIA]) for f in pliegues]
    e_pal = [a160.error_de(por_pliegue[f][PALANCA]) for f in pliegues]
    r, ic = a160.reduccion_relativa(e_ref, e_pal), a160.ic_de_r(e_ref, e_pal)
    coste = {b: sum(float(registros[f][b].get("wall_s") or 0.0) for f in pliegues) for b in BRAZOS}
    efectivo = {b: [(registros[f][b].get("entrenamiento_efectivo") or {}) for f in pliegues] for b in BRAZOS}
    rss = {b: max((v for f in pliegues if (v := registros[f][b].get("rss_pico_mb")) is not None), default=None) for b in BRAZOS}
    out.update(
        medias={b: statistics.fmean(por_pliegue[f][b] for f in pliegues) for b in BRAZOS},
        error_referencia=statistics.fmean(e_ref), error_palanca=statistics.fmean(e_pal),
        r=r, ic=list(ic) if ic else None, **a160.clasificar(r, ic),
        coste_s=coste, coste_ratio=(coste[PALANCA] / coste[REFERENCIA]) if coste[REFERENCIA] > 0 else None,
        rss_pico_mb=rss,
        epocas_ejecutadas={b: [e.get("epocas_ejecutadas") for e in efectivo[b]] for b in BRAZOS},
        mejor_epoca={b: [e.get("mejor_epoca") for e in efectivo[b]] for b in BRAZOS},
        parados_por_plazo={b: sum(1 for e in efectivo[b] if e.get("parado_por_plazo")) for b in BRAZOS},
        parados_por_paciencia={b: sum(1 for e in efectivo[b] if e.get("parado_por_paciencia")) for b in BRAZOS})
    return out


def aplicar_la_regla(evaluados: list[dict], *, confirmacion: bool = False) -> dict:
    """(a)–(d). Los SATURADOS no entran en la media (a) ni en el recuento (b); SÍ en (c) y (d). Un conjunto incompleto no se salta: la regla
    no es aplicable. `confirmacion=True`: (b) es la mitad (hacia arriba) de los no saturados."""
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
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok,
                  "rss_pico_mb": {b: max((v for e in medidos if (v := (e.get("rss_pico_mb") or {}).get(b)) is not None), default=None)
                                for b in BRAZOS}},
            "cumple": bool(not incompletos and a_ok and b_ok and c_ok and d_ok)}


def problemas_de_completitud(payload: dict, es_sonda: bool, *, confirmacion: bool = False) -> list[str]:
    """«Si falta algún conjunto/pliegue/brazo: SIN VEREDICTO», comprobado y no supuesto. Cada brazo declara SU red."""
    problemas: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    duplicados = sorted({n for n in nombres if nombres.count(n) > 1})
    if duplicados:
        problemas.append(f"conjuntos DUPLICADOS en la salida: {duplicados}")
    if not es_sonda and not confirmacion:
        cubo = conjuntos_del_cubo()
        faltan, sobran = sorted(set(cubo) - set(nombres)), sorted(set(nombres) - set(cubo))
        if faltan or sobran:
            problemas.append(f"los conjuntos no son los {N_CONJUNTOS_DEL_CUBO} de la medida: faltan {faltan}, sobran {sobran}")
        for c in conjuntos:
            esperados = cubo.get(c.get("nombre"), {}).get("pliegues")
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
        elif an1.metrica_del_registro(r) is None:
            problemas.append(f"el intento {k} no trae su métrica de cierre ({r.get('metrica_de_cierre')!r})")
        # DECLARAR LO QUE PASÓ: cada brazo declara SU red. Un brazo cruzado se lee igual de bien que uno correcto.
        var, arq = _declaracion_de_la_variante(r), r.get("arquitectura") or {}
        efectivo = r.get("entrenamiento_efectivo") or {}
        parada = (r.get("hiperparametros") or {}).get("parada_temprana") or {}
        if efectivo.get("parada_por_metrica") is not None or parada.get("metrica") != "validation_loss_del_ensamblado":
            problemas.append(f"el intento {k} no para por la pérdida: el C1 es otra palanca y no se mezcla")
        if efectivo.get("capacidad") is not None or arq.get("k") != K_DE_LOS_DOS:
            problemas.append(f"el intento {k} declara capacidad o k != {K_DE_LOS_DOS}: el C2 es otra palanca y no se mezcla")
        if r.get("motor") == REFERENCIA and (var is not None or arq.get("variante") is not None or "n_frequencies" not in arq):
            problemas.append(f"el intento {k} es de la REFERENCIA y declara una variante o no es la red PLR: brazos cruzados")
        if r.get("motor") == PALANCA and (not var or var.get("pedida") is not True or var.get("nombre") != VARIANTE
                                          or arq.get("variante") != VARIANTE or var.get("bordes_de") != "train"
                                          or var.get("n_bins") != N_BINS or var.get("d_embedding_numerico") != D_EMBEDDING_NUMERICO
                                          or not arq.get("bordes_ple")):
            problemas.append(f"el intento {k} es de la PALANCA y no declara la red mini_ple con bordes de train: brazos cruzados")
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
    """El veredicto de una salida de `medir_c162_c2.py`. `veredicto` ∈ CUMPLE / NO CUMPLE / SONDA / SIN VEREDICTO."""
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    conjuntos = payload.get("conjuntos", [])
    if not es_sonda and not confirmacion and len(conjuntos) != N_CONJUNTOS_DEL_CUBO:
        problemas.append(f"hay {len(conjuntos)} conjuntos, la regla es sobre {N_CONJUNTOS_DEL_CUBO}")
    if any(c.get("sellado") for c in conjuntos):
        problemas.append("hay un conjunto SELLADO en la medida")
    problemas += problemas_de_completitud(payload, es_sonda, confirmacion=confirmacion)
    rs = payload.get("resultados", [])
    for campo in ("entorno_digest", "motor_digest"):
        for brazo in BRAZOS:
            if len({r.get(campo) for r in rs if r.get("motor") == brazo}) > 1:
                problemas.append(f"los intentos del brazo {brazo} se midieron con CÓDIGO DISTINTO ({campo})")
        if len({r.get(campo) for r in rs}) > 1:
            problemas.append(f"los dos brazos se midieron con CÓDIGO DISTINTO ({campo})")
    sellos = {"regla_sha256": an1._sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None,
              "analizador_sha256": an1._sha256(RUTA_DEL_ANALIZADOR), "dependencias_sha256": dependencias_de_ahora()}
    if comprobar_sellos:
        if REGLA_SHA256_SELLADA is None:
            problemas.append("la regla NO está sellada (`REGLA_SHA256_SELLADA` es None): ninguna medida vale todavía")
        elif sellos["regla_sha256"] != REGLA_SHA256_SELLADA:
            problemas.append(f"REGLA_C3.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
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
        saturados = tuple(payload.get("saturados_declarados") or SATURADOS_DEL_CUBO) if confirmacion else SATURADOS_DEL_CUBO
        evaluados.append(evaluar_conjunto(c["nombre"], c["tarea"], c["metrica_de_cierre"], registros, esperados, saturados))
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


def _f(x, nd: int = 3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def informe_en_texto(r: dict) -> str:
    L = [f"=== 162-C3: TabM-mini + PLE contra TabM + PLR (los 23) — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  (a) media de R sobre {g['a']['n']} no saturados: {_f(g['a']['media_r_no_saturados'], 4)} (>= {g['a']['umbral']}) "
                 f"-> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras entre no saturados: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras_no_saturados']} "
                 f"-> {'ok' if g['b']['cumple'] else 'NO'}  [informativo, en saturados: {g['b']['informativo_mejoras_claras_en_saturados']}]")
        L.append(f"  (c) empeoran de verdad (> {g['c']['umbral']:.0%}, IC < 0): {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) coste (wall_s), razón de sumas: {_f(g['d']['razon_de_costes'], 3)} (<= {g['d']['umbral']}); mediana por conjunto "
                 f"{_f(g['d']['mediana_por_conjunto'], 3)}, máxima {_f(g['d']['maxima_por_conjunto'], 3)}; RSS pico {g['d']['rss_pico_mb']} "
                 f"-> {'ok' if g['d']['cumple'] else 'NO'}")
    L.append(f"  {'conjunto':34s} sat      R        IC95                 mejora empeora  plazo(ref/mini)")
    for e in r["conjuntos"]:
        if "r" not in e:
            L.append(f"  {e['nombre']:34s} (sin medida)")
            continue
        ic = e["ic"]
        L.append(f"  {e['nombre']:34s} {'S' if e['saturado'] else ' '}   {e['r']:+8.4f}  "
                 f"[{_f(ic[0], 3) if ic else '—'}, {_f(ic[1], 3) if ic else '—'}]   {'sí' if e['mejora_clara'] else '  '}    "
                 f"{'SÍ' if e['empeora_de_verdad'] else '  '}     {e['parados_por_plazo'][REFERENCIA]}/{e['parados_por_plazo'][PALANCA]}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    p.add_argument("--confirmacion", action="store_true", help="aplica la regla escalada de REGLA_C3.md §5")
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
