#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""160-C1 — EL ANALIZADOR: la mezcla de LightGBM y TabM por fila, contra el MEJOR de sus dos miembros, con la regla de
`REGLA_DE_DECISION.md` (decisiones D1–D4 de Roberto, 12-11, tarjeta `ola5-37-decisiones` = A).

    python3 benchmarks/c160/analiza_c160.py <resultado.json> [--json <salida.json>]

Sale con 0 si CUMPLE, 1 si NO CUMPLE, 2 si no hay veredicto (medida incompleta, código mezclado, regla o analizador distintos
de los que midieron, dependencias distintas), 0 en una SONDA (que no decide nada y lo dice).

LA MEZCLA es la media simple por fila de las probabilidades (clasificación; columnas alineadas por etiqueta) o de las
predicciones (regresión), emparejando las filas por su IDENTIFICADOR y no por su posición. La MÉTRICA DE CIERRE sale de la misma
cadena que usa la Fase 0 y el 155: `matrixai.estudio.metricas.evaluar(Muestra)` -> `pasada_amplia_101_c5.metricas_del_informe`
(importadas, no reescritas), sobre la misma prueba fija del conjunto.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parents[1]
for _ruta in (_AQUI, _RAIZ / "benchmarks" / "fase0", _RAIZ, _RAIZ.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import numpy as np  # noqa: E402

import pasada_amplia_101_c5 as c5  # noqa: E402  (`metricas_del_informe`: la lectura del informe que usa la Fase 0)
import protocolo as protocolo_mod  # noqa: E402  (semilla, remuestras y estados de medida de la Fase 0)
from matrixai.estudio import incertidumbre as _incertidumbre  # noqa: E402  (`_percentil`, el que usa el bootstrap de la Fase 0)
from matrixai.estudio import metricas as _metricas  # noqa: E402  (`Muestra`, `evaluar`)
import volcado_c160  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# LOS UMBRALES DE LA REGLA (D3). Un número por línea, con su condición; las pruebas los fijan contra REGLA_DE_DECISION.md.
# ----------------------------------------------------------------------------------------------------------------------
N_CONJUNTOS = 23
A_REDUCCION_MEDIA_MIN = 0.03        # (a) media de R sobre los NO saturados >= +3 %
B_MEJORAS_CLARAS_MIN = 5            # (b) conjuntos NO saturados con «mejora clara» (IC95 de R excluye 0 por arriba) >= 5
C_EMPEORA_MAS_DE = 0.03             # (c) ningún conjunto (saturados INCLUIDOS) con R < -3 % y el IC95 entero por debajo de 0
D_COSTE_MAXIMO = 3.0                # (d) coste al predecir de la mezcla <= 3 x el del mejor miembro
SEMILLA_DEL_IC = protocolo_mod.SEMILLA_DEL_INTERVALO        # 0
REMUESTRAS_DEL_IC = protocolo_mod.REMUESTRAS_DEL_INTERVALO  # 1000
NIVEL_DEL_IC = 0.95
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA

#: Los miembros (D2), con el nombre con que su motor se declara en los registros. LightGBM va PRIMERO: en un empate exacto de
#: medias es el «mejor miembro» (el árbol que el Studio ya tiene).
LIGHTGBM = "lightgbm"
TABM = "matrixai.dense.tabm_cpu"
MIEMBROS = (LIGHTGBM, TABM)

#: SATURADOS, declarados ANTES de medir (REGLA_DE_DECISION.md §2): la mejor de las dos referencias de los miembros en la Fase 0
#: congelada (media de LightGBM v2 y de TabM 119-C3/C4/120-C5, sobre todos sus intentos) es >= 98 %. `saturados_segun_el_criterio`
#: lo recalcula de los artefactos y una prueba exige que sea esta lista.
UMBRAL_DE_SATURACION = 0.98
SATURADOS = ("mfeat-factors", "optdigits", "pendigits", "sick", "wilt", "PhishingWebsites")

RUTA_DE_LA_REGLA = _AQUI / "REGLA_DE_DECISION.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()
_FASE0 = _RAIZ / "benchmarks" / "fase0"
RUTA_DEL_CONGELADO = _FASE0 / "resultado_pasada_119_c3.json"
CUBOS = ("pequeno", "mediano")

#: sha256 de `REGLA_DE_DECISION.md` SELLADA (la pone el supervisor al sellar; `medir_c160.py` la repite y se niega a medir si el
#: fichero ya no es éste). `None` = todavía sin sellar: el analizador y el guion lo dicen y no dan por válida ninguna medida.
REGLA_SHA256_SELLADA: str | None = "90b247f622181385c998acba6789ee5ed7b1c10a4ad3c358ffdb6dafe9b68a55"


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia: el IC y los estados de la Fase 0, la lectura del informe, la métrica
    (`Muestra`/`evaluar`), el percentil del bootstrap y el volcado. Mismas etiquetas que `medir_c160.dependencias_sha256`."""
    return {"benchmarks/fase0/protocolo.py": _sha256(Path(protocolo_mod.__file__).resolve()),
            "benchmarks/fase0/pasada_amplia_101_c5.py": _sha256(Path(c5.__file__).resolve()),
            "matrixai/estudio/metricas.py": _sha256(Path(_metricas.__file__).resolve()),
            "matrixai/estudio/incertidumbre.py": _sha256(Path(_incertidumbre.__file__).resolve()),
            "benchmarks/c160/volcado_c160.py": _sha256(Path(volcado_c160.__file__).resolve())}


# ----------------------------------------------------------------------------------------------------------------------
# 1. LA MEZCLA, por fila
# ----------------------------------------------------------------------------------------------------------------------
def alinear_por_fila(a: dict, b: dict) -> dict:
    """`b` reordenado a las filas de `a` POR IDENTIFICADOR. Se niega si no son las mismas filas o si la verdad difiere (dos
    pruebas distintas no se mezclan)."""
    ids_a, ids_b = list(a["row_ids"]), list(b["row_ids"])
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise ValueError("hay identificadores de fila repetidos: no se puede emparejar por fila")
    if set(ids_a) != set(ids_b):
        raise ValueError(f"los dos miembros no predijeron las mismas filas ({len(set(ids_a) ^ set(ids_b))} distintas)")
    pos = {r: i for i, r in enumerate(ids_b)}
    orden = [pos[r] for r in ids_a]
    y_b = [b["y_true"][i] for i in orden]
    if list(a["y_true"]) != y_b:
        raise ValueError("la verdad de las filas no coincide entre los dos miembros")
    salida = {k: v for k, v in b.items()}
    salida["row_ids"] = [ids_b[i] for i in orden]
    salida["y_true"] = y_b
    salida["predicciones"] = [b["predicciones"][i] for i in orden]
    if b.get("probabilidades") is not None:
        salida["probabilidades"] = np.asarray(b["probabilidades"])[orden]
    return salida


def mezclar(a: dict, b: dict) -> dict:
    """La mezcla de dos volcados: media simple por fila. Clasificación: de la distribución completa (columnas alineadas por
    ETIQUETA de clase), y la clase predicha es la de mayor probabilidad media (empate -> la primera clase). Regresión: media de las
    predicciones."""
    if a["es_clasificacion"] != b["es_clasificacion"]:
        raise ValueError("un miembro es de clasificación y el otro de regresión")
    b = alinear_por_fila(a, b)
    salida = {"row_ids": list(a["row_ids"]), "y_true": list(a["y_true"]), "es_clasificacion": a["es_clasificacion"],
              "t_predict_s": None, "t_proba_s": None}
    if not a["es_clasificacion"]:
        salida.update(predicciones=[(float(x) + float(y)) / 2.0 for x, y in zip(a["predicciones"], b["predicciones"])],
                      probabilidades=None, clases=None, positiva=None)
        return salida
    if set(a["clases"]) != set(b["clases"]) or a["positiva"] != b["positiva"]:
        raise ValueError("los dos miembros no tienen las mismas clases / clase positiva")
    col_b = [list(b["clases"]).index(c) for c in a["clases"]]
    pb = np.asarray(b["probabilidades"], dtype=np.float64)[:, col_b]
    pm = (np.asarray(a["probabilidades"], dtype=np.float64) + pb) / 2.0
    clases = list(a["clases"])
    salida.update(probabilidades=pm, clases=a["clases"], positiva=a["positiva"],
                  predicciones=[clases[int(i)] for i in np.argmax(pm, axis=1)])
    return salida


def muestra_de(v: dict, tarea: str):
    """La `Muestra` (105-C1) del volcado, como la arma `harness._muestra_de_test`."""
    campos: dict[str, Any] = dict(task=tarea, y_true=tuple(v["y_true"]), predictions=tuple(v["predicciones"]))
    if v["es_clasificacion"]:
        campos["classes"] = tuple(v["clases"])
        campos["positive_label"] = v["positiva"]
        if v.get("probabilidades") is not None:
            campos["probabilities"] = tuple(tuple(float(x) for x in fila) for fila in v["probabilidades"])
    return _metricas.Muestra(**campos)


def metrica_de(v: dict, tarea: str, metric_id: str) -> float:
    """La métrica de cierre del volcado: la MISMA cadena que la Fase 0 (`evaluar` + `metricas_del_informe`)."""
    informe = _metricas.evaluar(muestra_de(v, tarea))
    valor = c5.metricas_del_informe(informe.a_json()).get(metric_id)
    if valor is None:
        raise ValueError(f"la métrica {metric_id!r} no sale definida para esta muestra")
    return float(valor)


# ----------------------------------------------------------------------------------------------------------------------
# 2. REDUCCIÓN RELATIVA DEL ERROR e IC
# ----------------------------------------------------------------------------------------------------------------------
def error_de(metrica: float) -> float:
    """error = 1 − métrica (en r², 1 − r²)."""
    return 1.0 - float(metrica)


def reduccion_relativa(errores_ref: list[float], errores_nuevo: list[float]) -> float:
    """R = 1 − Σ error_nuevo / Σ error_ref (= reducción relativa del error MEDIO de los pliegues). Razón de medias y no media de
    razones: con un error de referencia de ~0 en un pliegue la razón por pliegue explota. Si el error de referencia es 0: R = 0 si
    el nuevo también, −1 (100 % peor) si no."""
    ref, nuevo = sum(errores_ref), sum(errores_nuevo)
    if ref <= 0.0:
        return 0.0 if nuevo <= 0.0 else -1.0
    return 1.0 - nuevo / ref


def ic_de_r(errores_ref: list[float], errores_nuevo: list[float]) -> tuple[float, float] | None:
    """Bootstrap de percentiles de PLIEGUES (1.000 remuestras, semilla 0, 95 %: el de la Fase 0) sobre R. `None` con menos de dos
    pliegues."""
    n = len(errores_ref)
    if n < 2 or n != len(errores_nuevo):
        return None
    azar = random.Random(SEMILLA_DEL_IC)
    valores = []
    for _ in range(REMUESTRAS_DEL_IC):
        idx = azar.choices(range(n), k=n)
        valores.append(reduccion_relativa([errores_ref[i] for i in idx], [errores_nuevo[i] for i in idx]))
    valores.sort()
    cola = (1.0 - NIVEL_DEL_IC) / 2.0
    return (_incertidumbre._percentil(valores, cola), _incertidumbre._percentil(valores, 1.0 - cola))


def mejor_miembro(medias: dict[str, float]) -> str:
    """El MEJOR de los miembros (D1): la mayor media de la métrica de cierre; empate exacto -> el primero de `MIEMBROS`. `max`, a
    propósito, no `min`: la mezcla se compara con el que MEJOR rinde."""
    return max(MIEMBROS, key=lambda m: (medias[m], -MIEMBROS.index(m)))


def clasificar(r: float, ic: tuple[float, float] | None) -> dict:
    return {"mejora_clara": bool(ic is not None and ic[0] > 0.0),
            "empeora_de_verdad": bool(ic is not None and r < -C_EMPEORA_MAS_DE and ic[1] < 0.0)}


def coste_por_fila_ms(v: dict) -> float | None:
    """ms por fila al predecir: la llamada que el arnés mide (probabilidades en clasificación, predicción en regresión)."""
    t = v["t_proba_s"] if v["es_clasificacion"] else v["t_predict_s"]
    n = len(v["row_ids"])
    return None if (t is None or n == 0) else 1000.0 * float(t) / n


def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, volcados: dict[int, dict[str, dict]], esperados: list[int],
                     saturados: tuple[str, ...] = SATURADOS) -> dict:
    """UN conjunto: por pliegue, la métrica de cada miembro y de la mezcla; el mejor miembro; R, su IC y su clasificación; los
    costes. `volcados[pliegue][miembro]`. Un pliegue esperado sin los dos miembros -> `completo=False` (SIN VEREDICTO arriba)."""
    faltan = [f for f in esperados if f not in volcados or any(m not in volcados[f] for m in MIEMBROS)]
    pliegues = [f for f in esperados if f not in faltan]
    por_pliegue: dict[int, dict[str, float]] = {}
    costes: dict[str, list[float]] = {m: [] for m in MIEMBROS}
    for f in pliegues:
        a, b = volcados[f][LIGHTGBM], volcados[f][TABM]
        por_pliegue[f] = {LIGHTGBM: metrica_de(a, tarea, metric_id), TABM: metrica_de(b, tarea, metric_id),
                          "mezcla": metrica_de(mezclar(a, b), tarea, metric_id)}
        for m, v in ((LIGHTGBM, a), (TABM, b)):
            c = coste_por_fila_ms(v)
            if c is not None:
                costes[m].append(c)
    saturado = nombre in saturados
    out = {"nombre": nombre, "tarea": tarea, "metrica": metric_id, "saturado": saturado, "pliegues": pliegues,
           "pliegues_que_faltan": faltan, "completo": not faltan and bool(pliegues), "por_pliegue": por_pliegue}
    if not pliegues:
        return out
    medias = {m: statistics.fmean(por_pliegue[f][m] for f in pliegues) for m in (*MIEMBROS, "mezcla")}
    mejor = mejor_miembro(medias)
    e_ref = [error_de(por_pliegue[f][mejor]) for f in pliegues]
    e_mez = [error_de(por_pliegue[f]["mezcla"]) for f in pliegues]
    r, ic = reduccion_relativa(e_ref, e_mez), ic_de_r(e_ref, e_mez)
    c_medio = {m: (statistics.fmean(costes[m]) if costes[m] else None) for m in MIEMBROS}
    ratio = None
    if all(c_medio[m] is not None for m in MIEMBROS) and c_medio[mejor] > 0:
        ratio = (c_medio[LIGHTGBM] + c_medio[TABM]) / c_medio[mejor]
    out.update(medias=medias, mejor_miembro=mejor, error_mejor=statistics.fmean(e_ref), error_mezcla=statistics.fmean(e_mez),
               r=r, ic=list(ic) if ic else None, **clasificar(r, ic), coste_ms_por_fila=c_medio, coste_ratio=ratio)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 3. LA REGLA
# ----------------------------------------------------------------------------------------------------------------------
def aplicar_la_regla(evaluados: list[dict]) -> dict:
    """(a)–(d) sobre los conjuntos evaluados. Los SATURADOS no entran en la media (a) ni en el recuento (b); SÍ en (c) y (d).
    Un conjunto incompleto no se salta: hace que la regla no sea aplicable (`aplicable=False`; el veredicto es SIN VEREDICTO)."""
    incompletos = [e["nombre"] for e in evaluados if not e["completo"]]
    medidos = [e for e in evaluados if e["completo"]]
    no_sat = [e for e in medidos if not e["saturado"]]
    r_no_sat = [e["r"] for e in no_sat]
    a_media = statistics.fmean(r_no_sat) if r_no_sat else None
    mejoras = [e["nombre"] for e in no_sat if e["mejora_clara"]]
    mejoras_saturados = [e["nombre"] for e in medidos if e["saturado"] and e["mejora_clara"]]
    empeoran = [e["nombre"] for e in medidos if e["empeora_de_verdad"]]
    suma_mezcla = suma_mejor = 0.0
    ratios = []
    coste_ok = bool(medidos)
    for e in medidos:
        c = e["coste_ms_por_fila"]
        if e["coste_ratio"] is None or any(c[m] is None for m in MIEMBROS):
            coste_ok = False
            continue
        suma_mezcla += c[LIGHTGBM] + c[TABM]
        suma_mejor += c[e["mejor_miembro"]]
        ratios.append(e["coste_ratio"])
    d_ratio = suma_mezcla / suma_mejor if (coste_ok and suma_mejor > 0) else None
    a_ok = a_media is not None and a_media >= A_REDUCCION_MEDIA_MIN
    b_ok = len(mejoras) >= B_MEJORAS_CLARAS_MIN
    c_ok = not empeoran
    d_ok = d_ratio is not None and d_ratio <= D_COSTE_MAXIMO
    return {"aplicable": not incompletos, "incompletos": incompletos,
            "a": {"media_r_no_saturados": a_media, "n": len(r_no_sat), "umbral": A_REDUCCION_MEDIA_MIN, "cumple": a_ok},
            "b": {"mejoras_claras_no_saturados": mejoras, "n": len(mejoras), "umbral": B_MEJORAS_CLARAS_MIN, "cumple": b_ok,
                  "informativo_mejoras_claras_en_saturados": mejoras_saturados},
            "c": {"empeoran": empeoran, "umbral": C_EMPEORA_MAS_DE, "cumple": c_ok},
            "d": {"razon_de_costes": d_ratio, "mediana_por_conjunto": statistics.median(ratios) if ratios else None,
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok},
            "cumple": bool(not incompletos and a_ok and b_ok and c_ok and d_ok)}


# ----------------------------------------------------------------------------------------------------------------------
# 4. COMPLETITUD, SELLOS Y LA MEDIDA ENTERA
# ----------------------------------------------------------------------------------------------------------------------
def conjuntos_y_pliegues_del_congelado() -> dict[str, dict]:
    """Los 23 y sus pliegues de la repetición 0, LEÍDOS del congelado de 119-C3 (no escritos a mano)."""
    d = json.loads(RUTA_DEL_CONGELADO.read_text(encoding="utf-8"))
    pliegues: dict[str, set[int]] = {}
    for r in d["resultados"]:
        if r.get("repeticion") == 0:
            pliegues.setdefault(r["dataset"], set()).add(int(r["pliegue"]))
    return {x["nombre"]: {"cubo": x["cubo"], "pliegues": sorted(pliegues.get(x["nombre"], ()))}
            for x in d["datasets_declarados"] if x["cubo"] in CUBOS and not x["sellado"]}


def saturados_segun_el_criterio(umbral: float = UMBRAL_DE_SATURACION) -> list[str]:
    """El criterio de saturación RECALCULADO de los artefactos congelados: la mejor, entre LightGBM (`pasada_v2_113_resultado`) y
    TabM (119-C3, 119-C4, 120-C5), de la media de la métrica de cierre sobre todos sus intentos que cuentan como medida, es >= umbral.
    Sobre los 23. Es lo que la regla declara y la prueba compara con `SATURADOS`."""
    def cargar(nombre):
        return json.loads((_FASE0 / nombre).read_text(encoding="utf-8"))["resultados"]
    lgbm = [x for x in cargar("pasada_v2_113_resultado.json") if x["motor"] == LIGHTGBM]
    tabm = [x for x in (cargar("resultado_pasada_119_c3.json") + cargar("resultado_pasada_119_c4.json")
                        + cargar("resultado_120_c5_tabm_1_2_0.json")) if x["motor"] == TABM]

    def media(lista, d):
        v = [x[x["metrica_de_cierre"]] for x in lista if x["dataset"] == d and x.get("estado") in ESTADOS_DE_MEDIDA
             and x.get(x["metrica_de_cierre"]) is not None]
        return statistics.fmean(v) if v else None
    salida = []
    for d in conjuntos_y_pliegues_del_congelado():
        mejores = [m for m in (media(lgbm, d), media(tabm, d)) if m is not None]
        if mejores and max(mejores) >= umbral:
            salida.append(d)
    return salida


def problemas_de_completitud(payload: dict, es_sonda: bool) -> list[str]:
    """«Si falta algún conjunto/pliegue: SIN VEREDICTO», comprobado y no supuesto."""
    problemas: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    duplicados = sorted({n for n in nombres if nombres.count(n) > 1})
    if duplicados:
        problemas.append(f"conjuntos DUPLICADOS en la salida: {duplicados}")
    if not es_sonda:
        congelado = conjuntos_y_pliegues_del_congelado()
        faltan, sobran = sorted(set(congelado) - set(nombres)), sorted(set(nombres) - set(congelado))
        if faltan or sobran:
            problemas.append(f"los conjuntos no son los {N_CONJUNTOS} del congelado: faltan {faltan}, sobran {sobran}")
        for c in conjuntos:
            esperados = congelado.get(c.get("nombre"), {}).get("pliegues")
            if esperados is not None and list(c.get("pliegues_esperados") or []) != esperados:
                problemas.append(f"{c.get('nombre')}: pliegues declarados {c.get('pliegues_esperados')} != los de la partición "
                                 f"{esperados}")
    esperado = {(c.get("nombre"), m, int(f)) for c in conjuntos for m in MIEMBROS for f in (c.get("pliegues_esperados") or [])}
    vistos: dict[tuple, int] = {}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("motor"), int(r.get("pliegue", -1)))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("repeticion") != 0:
            problemas.append(f"un registro de la repetición {r.get('repeticion')} ({k}): la regla es sobre la 0")
        if r.get("estado") not in ESTADOS_DE_MEDIDA:
            problemas.append(f"el intento {k} no cuenta como medida (estado {r.get('estado')!r}): falta ese pliegue")
        elif not r.get("volcado"):
            problemas.append(f"el intento {k} no trae su volcado")
    faltan_i, sobran_i = sorted(esperado - set(vistos)), sorted(set(vistos) - esperado)
    dobles = sorted(k for k, n in vistos.items() if n > 1)
    if faltan_i:
        problemas.append(f"faltan {len(faltan_i)} intentos (p. ej. {faltan_i[:3]})")
    if sobran_i:
        problemas.append(f"{len(sobran_i)} registros fuera de lo declarado (p. ej. {sobran_i[:3]})")
    if dobles:
        problemas.append(f"{len(dobles)} intentos registrados más de una vez (p. ej. {dobles[:3]})")
    return problemas


def _cargador_desde_disco(base: Path) -> Callable[[dict], dict]:
    def cargar(registro: dict) -> dict:
        ruta = base / registro["volcado"]["fichero"]
        if _sha256(ruta) != registro["volcado"]["sha256"]:
            raise ValueError(f"{ruta.name}: su sha256 no es el que se registró al medir")
        return volcado_c160.leer_volcado(ruta)
    return cargar


def analizar(payload: dict, *, cargador: Callable[[dict], dict] | None = None, base: Path | None = None,
             comprobar_sellos: bool = True) -> dict:
    """El veredicto de una salida de `medir_c160.py`. `veredicto` ∈ CUMPLE / NO CUMPLE / SONDA / SIN VEREDICTO.
    `cargador(registro) -> volcado` (las pruebas inyectan el suyo); por omisión lee de `base` y comprueba el sha256."""
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    conjuntos = payload.get("conjuntos", [])
    if not es_sonda and len(conjuntos) != N_CONJUNTOS:
        problemas.append(f"hay {len(conjuntos)} conjuntos, la regla es sobre {N_CONJUNTOS}")
    if any(c.get("sellado") for c in conjuntos):
        problemas.append("hay un conjunto SELLADO en la medida")
    problemas += problemas_de_completitud(payload, es_sonda)
    rs = payload.get("resultados", [])
    for campo in ("entorno_digest",):
        if len({r.get(campo) for r in rs if r.get("motor") == LIGHTGBM}) > 1 or len({r.get(campo) for r in rs if r.get("motor") == TABM}) > 1:
            problemas.append(f"los intentos de un mismo motor se midieron con CÓDIGO DISTINTO ({campo})")
    if len({r.get("motor_digest") for r in rs if r.get("motor") == TABM}) > 1 or len({r.get("motor_digest") for r in rs if r.get("motor") == LIGHTGBM}) > 1:
        problemas.append("los intentos de un mismo motor se midieron con CÓDIGO DISTINTO (motor_digest)")
    sellos = {"regla_sha256": _sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None,
              "analizador_sha256": _sha256(RUTA_DEL_ANALIZADOR), "dependencias_sha256": dependencias_de_ahora()}
    if comprobar_sellos:
        if REGLA_SHA256_SELLADA is None:
            problemas.append("la regla NO está sellada (`REGLA_SHA256_SELLADA` es None): ninguna medida vale todavía")
        elif sellos["regla_sha256"] != REGLA_SHA256_SELLADA:
            problemas.append(f"REGLA_DE_DECISION.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
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
    if cargador is None:
        cargador = _cargador_desde_disco(base if base is not None else Path(payload.get("directorio_de_volcados", ".")))
    por_clave = {(r["dataset"], r["motor"], int(r["pliegue"])): r for r in rs if r.get("estado") in ESTADOS_DE_MEDIDA and r.get("volcado")}
    evaluados = []
    for c in conjuntos:
        volcados: dict[int, dict[str, dict]] = {}
        for f in c.get("pliegues_esperados") or []:
            for m in MIEMBROS:
                r = por_clave.get((c["nombre"], m, int(f)))
                if r is None:
                    continue
                try:
                    volcados.setdefault(int(f), {})[m] = cargador(r)
                except (OSError, ValueError, KeyError) as e:
                    problemas.append(f"{c['nombre']} {m} pliegue {f}: el volcado no se puede leer ({e})")
        evaluados.append(evaluar_conjunto(c["nombre"], c["tarea"], c["metrica_de_cierre"], volcados,
                                          [int(f) for f in c.get("pliegues_esperados") or []]))
    for e in evaluados:
        if not e["completo"]:
            problemas.append(f"{e['nombre']}: faltan pliegues {e['pliegues_que_faltan']}")
    regla = aplicar_la_regla(evaluados) if evaluados else None
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
    L = [f"=== 160-C1: mezcla LightGBM + TabM contra el mejor miembro — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  (a) media de R sobre {g['a']['n']} no saturados: {_f(g['a']['media_r_no_saturados'], 4)} (>= {g['a']['umbral']}) "
                 f"-> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras entre no saturados: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras_no_saturados']} "
                 f"-> {'ok' if g['b']['cumple'] else 'NO'}  [informativo, en saturados: {g['b']['informativo_mejoras_claras_en_saturados']}]")
        L.append(f"  (c) empeoran de verdad (> {g['c']['umbral']:.0%}, IC < 0): {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) coste de predecir, razón de sumas: {_f(g['d']['razon_de_costes'], 2)} (<= {g['d']['umbral']}); mediana "
                 f"{_f(g['d']['mediana_por_conjunto'], 2)}, máxima {_f(g['d']['maxima_por_conjunto'], 2)} -> {'ok' if g['d']['cumple'] else 'NO'}")
    L.append(f"  {'conjunto':34s} sat mejor_miembro        R        IC95                 mejora empeora")
    for e in r["conjuntos"]:
        if "r" not in e:
            L.append(f"  {e['nombre']:34s} (sin medida)")
            continue
        ic = e["ic"]
        L.append(f"  {e['nombre']:34s} {'S' if e['saturado'] else ' '}   {e['mejor_miembro'][:16]:16s} {e['r']:+8.4f}  "
                 f"[{_f(ic[0], 3) if ic else '—'}, {_f(ic[1], 3) if ic else '—'}]   {'sí' if e['mejora_clara'] else '  '}    "
                 f"{'SÍ' if e['empeora_de_verdad'] else '  '}")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    a = p.parse_args(argv)
    ruta = Path(a.resultado).resolve()
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    r = analizar(payload, base=Path(payload["directorio_de_volcados"]) if payload.get("directorio_de_volcados") else ruta.parent)
    print(informe_en_texto(r))
    print(f"  sellos de ahora: regla {str(r['sellos_de_ahora']['regla_sha256'])[:16]} analizador {r['sellos_de_ahora']['analizador_sha256'][:16]}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return {"CUMPLE": 0, "NO CUMPLE": 1, "SONDA": 0}.get(r["veredicto"], 2)


if __name__ == "__main__":
    sys.exit(main())
