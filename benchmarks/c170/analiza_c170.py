#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""170 — EL ANALIZADOR: ¿queda el EBM (`interpret.ebm`) a ≤ 2 puntos del mejor de la cartera en ≥ 80 % de los conjuntos medibles? La regla es `REGLA.md`.

    python3 benchmarks/c170/analiza_c170.py <resultado.json> [--json <salida.json>]

Veredictos: CUMPLE / NO CUMPLE / NO CONCLUYENTE (M < 20 conjuntos comparables) / SIN VEREDICTO (la medida no es completa o los sellos no cuadran) /
SONDA (una sonda no decide nada). Código de salida: 0 CUMPLE, 1 NO CUMPLE, 2 SIN VEREDICTO, 3 NO CONCLUYENTE, 4 SONDA.

LA REGLA, CITADA POR SU sha256: `REGLA_SHA256_SELLADA` es el de `REGLA.md` (lo pone el supervisor al sellar; `None` = sin sellar: ninguna medida vale) y el veredicto lo graba en
`regla_sha256`. Después se sella el CÓDIGO (`SELLO_DEL_CODIGO.json`, lo escribe `sellar_codigo_c170.py`): este analizador, el medidor, el motor y lo que el veredicto importa.
Sin ese sello, con un nulo o con un fichero que ya no cuadra, el medidor no mide (salvo `--sonda`) y este analizador no da veredicto.

LOS RIVALES NO SE REMIDEN: salen de `pasada_v2_113_resultado.json` (lightgbm, sklearn.hgb) y `resultado_pasada_119_c3.json` (TabM), leídos aquí (`cargar_referencia`). LO DECLARADO NO SE
CREE: la métrica de cierre, la semilla, el presupuesto, el `plan_digest`, `n_test`, el sha256 de los datos, el número de predictores y los hilos se COMPRUEBAN contra la v2 en cada registro.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parents[1]
for _ruta in (_AQUI, _RAIZ / "benchmarks" / "fase0", _RAIZ, _RAIZ.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import protocolo as protocolo_mod  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# LOS UMBRALES DE LA REGLA (REGLA.md §1–§3). Un número por línea; `tests/test_c170_regla.py` los fija contra el TEXTO de REGLA.md.
# ----------------------------------------------------------------------------------------------------------------------
DISTANCIA_MINIMA = -0.02              # «a ≤ 2 puntos»: d = media(EBM) − media(mejor rival) >= −0,02 en la escala cruda de la métrica
FRACCION_MINIMA = Fraction(4, 5)      # ≥ 80 % de los conjuntos medibles, comparada como enteros (5·C >= 4·M)
MEDIBLES_MINIMOS = 20                 # (a) potencia: con menos de 20 comparables, NO CONCLUYENTE
COLUMNAS_MAXIMAS = 100                # régimen: no sellados con ≤ 100 predictores
N_DEL_REGIMEN = 24                    # lo que da la regla sobre la v2 (una prueba lo fija contra la v2 de verdad)
HILOS_POR_INTENTO = 4
VERSION_DE_INTERPRET_CORE = "0.7.8"
VERSION_DEL_MOTOR = f"1.0.0+interpret-core{VERSION_DE_INTERPRET_CORE}"

MOTOR = "interpret.ebm"
RIVALES = ("lightgbm", "sklearn.hgb", "matrixai.dense.tabm_cpu")      # EL ORDEN desempata (REGLA §1)
METRICA_POR_TAREA = {"binary_classification": "auroc", "multiclass_classification": "accuracy", "regression": "r2"}
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA

RUTA_DE_LA_REGLA = _AQUI / "REGLA.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()
RUTA_DEL_MEDIDOR = _AQUI / "medir_c170.py"
#: El sello del CÓDIGO. No está en `codigo_de_ahora()`: un sello no puede contener su propio sha256.
RUTA_DEL_SELLO_DEL_CODIGO = _AQUI / "SELLO_DEL_CODIGO.json"
RUTA_DE_LA_V2 = _RAIZ / "benchmarks" / "fase0" / "pasada_v2_113_resultado.json"
RUTA_DE_TABM = _RAIZ / "benchmarks" / "fase0" / "resultado_pasada_119_c3.json"
RUTA_DEL_MOTOR = _RAIZ.parent / "matrixai-engines" / "src" / "matrixai_engines" / "motores" / "interpret_ebm.py"

#: sha256 de `REGLA.md` SELLADA. La pone el supervisor al sellar. `None` = sin sellar: ninguna medida real vale.
REGLA_SHA256_SELLADA: str | None = "e485d0844e61ce61ef363a22ca47297ab40cb9baad5b076188c7274085e68a6b"


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto y el medidor IMPORTAN o LEEN y no copian: el motor, el plumbing de motores, la pasada 116-C2 y las de Fase 0 que usa, el protocolo v2 y los dos
    artefactos de los rivales. Cambiar cualquiera después de sellar es medir (o juzgar) con otra cosa."""
    engines = RUTA_DEL_MOTOR.parents[1]
    fase0 = _RAIZ / "benchmarks" / "fase0"
    ficheros = {
        "matrixai-engines/motores/interpret_ebm.py": RUTA_DEL_MOTOR,
        "matrixai-engines/motor.py": engines / "motor.py",
        "matrixai-engines/subproceso.py": engines / "subproceso.py",
        "matrixai-engines/tipos_de_columna.py": engines / "tipos_de_columna.py",
        "matrixai-engines/registro_de_motores.py": engines / "registro_de_motores.py",
        "benchmarks/fase0/pasada_116c2_tabicl.py": fase0 / "pasada_116c2_tabicl.py",
        "benchmarks/fase0/pasada_amplia_101_c5.py": fase0 / "pasada_amplia_101_c5.py",
        "benchmarks/fase0/pasada_exploratoria_101_c3.py": fase0 / "pasada_exploratoria_101_c3.py",
        "benchmarks/fase0/pasada_v2_113.py": fase0 / "pasada_v2_113.py",
        "benchmarks/fase0/protocolo.py": fase0 / "protocolo.py",
        "benchmarks/fase0/protocolo_exploratorio_v2.json": fase0 / "protocolo_exploratorio_v2.json",
        "benchmarks/fase0/pasada_v2_113_resultado.json": RUTA_DE_LA_V2,
        "benchmarks/fase0/resultado_pasada_119_c3.json": RUTA_DE_TABM,
    }
    return {etiqueta: _sha256(ruta) for etiqueta, ruta in ficheros.items()}


def codigo_de_ahora() -> dict[str, str]:
    """Lo que cubre el SELLO DEL CÓDIGO: este analizador (que lleva dentro el sha de la regla), el medidor y las dependencias. NO el propio `SELLO_DEL_CODIGO.json`."""
    d = {"benchmarks/c170/analiza_c170.py": _sha256(RUTA_DEL_ANALIZADOR), "benchmarks/c170/medir_c170.py": _sha256(RUTA_DEL_MEDIDOR)}
    d.update(dependencias_de_ahora())
    return d


def leer_sello_del_codigo(ruta: Path | None = None) -> dict | None:
    ruta = Path(ruta) if ruta is not None else RUTA_DEL_SELLO_DEL_CODIGO
    try:
        sello = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return sello if isinstance(sello, dict) else None


def problemas_del_sello_del_codigo(ruta: Path | None = None) -> list[str]:
    """Vacía SOLO si el sello existe, no tiene nulos, cubre EXACTAMENTE `codigo_de_ahora()`, cada sha cuadra con el fichero de ahora y se selló sobre la regla sellada."""
    ruta = Path(ruta) if ruta is not None else RUTA_DEL_SELLO_DEL_CODIGO
    sello = leer_sello_del_codigo(ruta)
    if sello is None:
        return [f"falta el sello del código o no se puede leer ({ruta.name}): el supervisor lo pone con `sellar_codigo_c170.py`"]
    sha = sello.get("sha256")
    if not isinstance(sha, dict) or not sha or any(v is None for v in sha.values()):
        return ["el sello del código está SIN SELLAR (valores nulos): ninguna medida vale todavía"]
    p: list[str] = []
    ahora = codigo_de_ahora()
    if set(sha) != set(ahora):
        p.append(f"el sello del código no cubre exactamente lo que importa el veredicto: faltan {sorted(set(ahora) - set(sha))}, sobran {sorted(set(sha) - set(ahora))}")
    for etiqueta in sorted(set(sha) & set(ahora)):
        if sha[etiqueta] != ahora[etiqueta]:
            p.append(f"`{etiqueta}` ({ahora[etiqueta][:12]}) no es el sellado ({str(sha[etiqueta])[:12]}): cambió DESPUÉS de sellar el código")
    if sello.get("regla_sha256_sellada") != REGLA_SHA256_SELLADA:
        p.append(f"el sello del código se hizo sobre otra regla ({str(sello.get('regla_sha256_sellada'))[:12]}) que la sellada en este analizador ({str(REGLA_SHA256_SELLADA)[:12]})")
    return p


def sellos_de_ahora() -> dict:
    sello = leer_sello_del_codigo()
    return {"regla_sha256": _sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None, "analizador_sha256": _sha256(RUTA_DEL_ANALIZADOR),
            "dependencias_sha256": dependencias_de_ahora(), "codigo_sha256": codigo_de_ahora(), "sello_del_codigo": (sello or {}).get("sha256")}


# ----------------------------------------------------------------------------------------------------------------------
# 1. LA REFERENCIA: el régimen y los rivales, de los artefactos YA medidos (no se remide nada)
# ----------------------------------------------------------------------------------------------------------------------
class Referencia:
    """Lo que la v2 y TabM dicen de cada conjunto no sellado: metadatos, pliegues esperados y, por rival, el valor de la métrica de cierre por (repetición, pliegue)."""

    def __init__(self, v2: dict, tabm: dict):
        self.v2, self.tabm = v2, tabm
        declarados = {d["nombre"]: d for d in v2["datasets_declarados"]}
        particion = v2["particion_por_dataset"]
        datos = (v2.get("procedencia") or {}).get("datos_de_entrada") or {}
        self.meta: dict[str, dict] = {}
        for nombre, d in declarados.items():
            if d.get("sellado"):
                continue
            p = particion[nombre]
            claves = {(r["repeticion"], r["pliegue"]) for r in v2["resultados"] if r["dataset"] == nombre and r["motor"] == RIVALES[0]}
            self.meta[nombre] = {
                "tarea": d["tarea"], "cubo": d["cubo"], "data_id": d["data_id"], "metrica": METRICA_POR_TAREA[d["tarea"]],
                "n_predictores": p["n_predictores"], "n_test": p["n_test"], "plan_digest": p["plan_digest"],
                "datos_sha256": (datos.get(nombre) or {}).get("sha256"), "claves": claves,
                "por_clave": {(r["repeticion"], r["pliegue"]): r for r in v2["resultados"] if r["dataset"] == nombre and r["motor"] == RIVALES[0]}}
        self._rivales: dict[tuple[str, str], dict] = {}
        for r in v2["resultados"] + tabm["resultados"]:
            if r["motor"] in RIVALES and r["dataset"] in self.meta:
                self._rivales.setdefault((r["dataset"], r["motor"]), {})[(r["repeticion"], r["pliegue"])] = r

    def regimen(self) -> list[str]:
        """Los NO sellados con ≤ `COLUMNAS_MAXIMAS` predictores, ordenados por nombre."""
        return sorted(n for n, m in self.meta.items() if m["n_predictores"] <= COLUMNAS_MAXIMAS)

    def fuera_del_regimen(self) -> dict[str, int]:
        return {n: m["n_predictores"] for n, m in sorted(self.meta.items()) if m["n_predictores"] > COLUMNAS_MAXIMAS}

    def registros_de_rival(self, nombre: str, rival: str) -> dict[tuple, dict]:
        return self._rivales.get((nombre, rival), {})

    def valores_completos(self, nombre: str, rival: str) -> dict[tuple, float] | None:
        """`(rep, pliegue) -> métrica de cierre` SOLO si el rival completó TODOS los pliegues esperados con un número finito; si no, `None` (no compite en ese conjunto)."""
        m = self.meta[nombre]
        regs = self.registros_de_rival(nombre, rival)
        valores: dict[tuple, float] = {}
        for k in m["claves"]:
            r = regs.get(k)
            if r is None or r.get("estado") not in ESTADOS_DE_MEDIDA:
                return None
            v = r.get(m["metrica"])
            if not _es_finito(v):
                return None
            valores[k] = float(v)
        return valores

    def problemas(self) -> list[str]:
        """La referencia tiene que ser la misma partición: el `plan_digest` por conjunto de TabM es el de la v2."""
        p: list[str] = []
        comparada = self.tabm.get("particion_comparada_con_la_v2") or {}
        particion_tabm = self.tabm.get("particion_por_dataset") or {}
        for n in self.regimen():
            propio = self.meta[n]["plan_digest"]
            if (comparada.get(n) or {}).get("plan_digest") != propio and (particion_tabm.get(n) or {}).get("plan_digest") != propio:
                p.append(f"{n}: el `plan_digest` de TabM no es el de la v2: no es la misma partición")
        return p


def cargar_referencia(ruta_v2: Path | None = None, ruta_tabm: Path | None = None) -> Referencia:
    v2 = json.loads(Path(ruta_v2 or RUTA_DE_LA_V2).read_text(encoding="utf-8"))
    tabm = json.loads(Path(ruta_tabm or RUTA_DE_TABM).read_text(encoding="utf-8"))
    return Referencia(v2, tabm)


def _es_finito(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ----------------------------------------------------------------------------------------------------------------------
# 2. UN CONJUNTO
# ----------------------------------------------------------------------------------------------------------------------
def mejor_rival(ref: Referencia, nombre: str) -> dict:
    """El mejor de la cartera en `nombre`: la mayor media sobre los pliegues esperados entre los rivales que completaron TODOS (empate: el orden de `RIVALES`)."""
    medias: dict[str, float] = {}
    excluidos: dict[str, str] = {}
    for rival in RIVALES:
        v = ref.valores_completos(nombre, rival)
        if v is None:
            excluidos[rival] = "no completó todos los pliegues esperados del conjunto"
        else:
            medias[rival] = statistics.fmean(v.values())
    if not medias:
        return {"rival": None, "media": None, "medias": {}, "excluidos": excluidos}
    mejor = max(RIVALES, key=lambda r: (medias.get(r, float("-inf")), -RIVALES.index(r)))
    return {"rival": mejor, "media": medias[mejor], "medias": medias, "excluidos": excluidos}


def evaluar_conjunto(ref: Referencia, nombre: str, registros: dict[tuple, dict]) -> dict:
    """UN conjunto del régimen. `registros[(rep, pliegue)]` = el registro del EBM. Categorías: `comparable_y_cumple`, `no_cumple_por_distancia`, `no_cumple_por_fallo`, `no_comparable`."""
    m = ref.meta[nombre]
    metrica = m["metrica"]
    mr = mejor_rival(ref, nombre)
    e: dict = {"nombre": nombre, "tarea": m["tarea"], "cubo": m["cubo"], "metrica": metrica, "n_pliegues": len(m["claves"]), "mejor_rival": mr["rival"],
               "media_del_mejor_rival": mr["media"], "medias_de_los_rivales": mr["medias"], "rivales_excluidos": mr["excluidos"]}
    fallos = []
    valores: dict[tuple, float] = {}
    for k in sorted(m["claves"]):
        r = registros.get(k)
        if r is None or r.get("estado") not in ESTADOS_DE_MEDIDA:
            fallos.append({"repeticion": k[0], "pliegue": k[1], "estado": None if r is None else r.get("estado"), "motivo": None if r is None else r.get("motivo"),
                           "wall_s": None if r is None else r.get("wall_s")})
            continue
        v = r.get(r.get("metrica_de_cierre"))
        if not _es_finito(v):
            fallos.append({"repeticion": k[0], "pliegue": k[1], "estado": r.get("estado"), "motivo": "la métrica de cierre no es un número finito", "wall_s": r.get("wall_s")})
            continue
        valores[k] = float(v)
    e["fallos_del_ebm"] = fallos
    e["completo"] = not fallos
    e["coste"] = _coste(ref, nombre, registros, mr["rival"])
    if mr["rival"] is None:
        e["categoria"], e["cumple"], e["medible"] = "no_comparable", False, False
        return e
    e["medible"] = True
    if fallos:
        e["categoria"], e["cumple"] = "no_cumple_por_fallo", False
        e["distancia"] = None
        return e
    rival_valores = ref.valores_completos(nombre, mr["rival"])
    diferencias = [valores[k] - rival_valores[k] for k in sorted(valores)]
    d = statistics.fmean(valores.values()) - statistics.fmean(rival_valores.values())
    ic = protocolo_mod._intervalo_pareado(diferencias)
    e.update({"media_del_ebm": statistics.fmean(valores.values()), "distancia": d, "ic95_de_la_distancia": list(ic) if ic else None,
              "el_ic_cruza_la_linea": bool(ic is not None and ic[0] <= DISTANCIA_MINIMA <= ic[1]),
              "por_pliegue": {f"{k[0]}.{k[1]}": {"ebm": valores[k], "rival": rival_valores[k]} for k in sorted(valores)}})
    e["cumple"] = bool(d >= DISTANCIA_MINIMA)
    e["categoria"] = "comparable_y_cumple" if e["cumple"] else "no_cumple_por_distancia"
    return e


def _coste(ref: Referencia, nombre: str, registros: dict[tuple, dict], mejor: str | None) -> dict:
    """Segundos de ajuste por pliegue (informativo, NO decide). El EBM sobre sus pliegues medidos; cada rival sobre los suyos."""
    def media(valores):
        valores = [float(v) for v in valores if _es_finito(v)]
        return statistics.fmean(valores) if valores else None
    ebm = [r.get("tiempo_de_ajuste") for r in registros.values() if r.get("estado") in ESTADOS_DE_MEDIDA]
    wall = [r.get("wall_s") for r in registros.values() if r.get("estado") in ESTADOS_DE_MEDIDA]
    por_rival = {rv: media(r.get("tiempo_de_ajuste") for r in ref.registros_de_rival(nombre, rv).values()) for rv in RIVALES}
    suma_rivales = sum(v for v in por_rival.values() if v is not None) if any(v is not None for v in por_rival.values()) else None
    return {"ebm_s_por_pliegue": media(ebm), "ebm_s_total": sum(float(v) for v in ebm if _es_finito(v)), "ebm_wall_s_por_pliegue": media(wall),
            "rivales_s_por_pliegue": por_rival, "suma_de_los_rivales_s_por_pliegue": suma_rivales, "n_pliegues_medidos": len(ebm)}


# ----------------------------------------------------------------------------------------------------------------------
# 3. LA REGLA
# ----------------------------------------------------------------------------------------------------------------------
def aplicar_la_regla(evaluados: list[dict]) -> dict:
    """(a) y (b) sobre los conjuntos del régimen YA evaluados. `veredicto`: CUMPLE / NO CUMPLE / NO CONCLUYENTE."""
    medibles = [e for e in evaluados if e["medible"]]
    no_comparables = [e["nombre"] for e in evaluados if not e["medible"]]
    cumplen = [e["nombre"] for e in medibles if e["cumple"]]
    por_fallo = [e["nombre"] for e in medibles if e["categoria"] == "no_cumple_por_fallo"]
    por_distancia = [e["nombre"] for e in medibles if e["categoria"] == "no_cumple_por_distancia"]
    M, C = len(medibles), len(cumplen)
    potencia = M >= MEDIBLES_MINIMOS
    fraccion = Fraction(C, M) if M else None
    b_ok = bool(M) and fraccion >= FRACCION_MINIMA                      # comparación EXACTA de enteros (Fraction), nunca de flotantes
    if not potencia:
        veredicto = "NO CONCLUYENTE"
    elif b_ok:
        veredicto = "CUMPLE"
    else:
        veredicto = "NO CUMPLE"
    distancias = [e["distancia"] for e in medibles if e.get("distancia") is not None]
    return {"veredicto": veredicto, "n_regimen": len(evaluados), "n_medibles": M, "n_no_comparables": len(no_comparables), "no_comparables": no_comparables,
            "a": {"medibles": M, "umbral": MEDIBLES_MINIMOS, "cumple": potencia},
            "b": {"cumplen": C, "medibles": M, "fraccion": None if fraccion is None else float(fraccion), "minimo": float(FRACCION_MINIMA),
                  "cumple": b_ok, "conjuntos_que_cumplen": cumplen, "no_cumplen_por_distancia": por_distancia, "no_cumplen_por_fallo": por_fallo,
                  "minimo_de_cumplen_para_este_M": math.ceil(FRACCION_MINIMA * M) if M else None},
            "el_ic_cruza_la_linea": [e["nombre"] for e in medibles if e.get("el_ic_cruza_la_linea")],
            "distancia_mediana": statistics.median(distancias) if distancias else None, "distancia_minima": min(distancias) if distancias else None}


def resumen_del_coste(evaluados: list[dict]) -> dict:
    """El coste global del EBM (informativo): Σ s, mediana y máximo por conjunto, razón de sumas frente a los tres rivales, pliegues fallidos y cuántos por tope."""
    por_conjunto = {e["nombre"]: e["coste"]["ebm_s_por_pliegue"] for e in evaluados if e["coste"]["ebm_s_por_pliegue"] is not None}
    ebm_suma, riv_suma = 0.0, 0.0
    for e in evaluados:
        c = e["coste"]
        if c["suma_de_los_rivales_s_por_pliegue"] is not None and c["ebm_s_por_pliegue"] is not None:
            ebm_suma += c["ebm_s_por_pliegue"] * c["n_pliegues_medidos"]
            riv_suma += c["suma_de_los_rivales_s_por_pliegue"] * c["n_pliegues_medidos"]
    fallos = [f for e in evaluados for f in e["fallos_del_ebm"]]
    return {"ebm_s_total": sum(e["coste"]["ebm_s_total"] for e in evaluados), "ebm_s_por_pliegue_mediana_por_conjunto": statistics.median(por_conjunto.values()) if por_conjunto else None,
            "ebm_s_por_pliegue_maxima_por_conjunto": max(por_conjunto.values()) if por_conjunto else None,
            "razon_de_sumas_ebm_sobre_los_tres_rivales": (ebm_suma / riv_suma) if riv_suma > 0 else None,
            "n_pliegues_fallidos": len(fallos), "por_conjunto_s_por_pliegue": por_conjunto,
            "que_es": "informativo, NO decide (REGLA.md §5): segundos de `tiempo_de_ajuste` del `fit`, en la geometría de la Fase 0. «Lo que añade a un estudio» = un ajuste más del EBM."}


# ----------------------------------------------------------------------------------------------------------------------
# 4. COMPLETITUD, DECLARACIONES Y SELLOS
# ----------------------------------------------------------------------------------------------------------------------
def problemas_de_completitud(payload: dict, ref: Referencia, es_sonda: bool) -> list[str]:
    p: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    for n in sorted({n for n in nombres if nombres.count(n) > 1}):
        p.append(f"conjunto DUPLICADO en la salida: {n}")
    regimen = ref.regimen()
    if not es_sonda:
        if set(nombres) != set(regimen):
            p.append(f"los conjuntos no son los {len(regimen)} del régimen: faltan {sorted(set(regimen) - set(nombres))}, sobran {sorted(set(nombres) - set(regimen))}")
    for c in conjuntos:
        n = c.get("nombre")
        if n not in ref.meta:
            p.append(f"{n}: no es un conjunto no sellado de la v2")
            continue
        if n not in regimen:
            p.append(f"{n}: está FUERA del régimen ({ref.meta[n]['n_predictores']} predictores > {COLUMNAS_MAXIMAS})")
        if c.get("sellado"):
            p.append(f"{n} está marcado como SELLADO")
        esperados = sorted(tuple(k) for k in c.get("pliegues_esperados") or [])
        if not es_sonda and esperados != sorted(ref.meta[n]["claves"]):
            p.append(f"{n}: pliegues declarados {esperados} != los de la v2 {sorted(ref.meta[n]['claves'])}")
    vistos: dict[tuple, int] = {}
    esperado_total = {(c["nombre"], tuple(k)) for c in conjuntos if c.get("nombre") in ref.meta for k in c.get("pliegues_esperados") or []}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), (r.get("repeticion"), r.get("pliegue")))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("motor") != MOTOR:
            p.append(f"el intento {k} es de otro motor que {MOTOR} ({r.get('motor')!r})")
            continue
        if r.get("estado") not in ESTADOS_DE_MEDIDA:
            continue                                                         # un fallo registrado es VÁLIDO: cuenta como no cumple (REGLA §3), no como hueco
        if not _es_finito(r.get(r.get("metrica_de_cierre"))):
            p.append(f"el intento {k} no trae su métrica de cierre finita ({r.get('metrica_de_cierre')!r})")
        coste = r.get("tiempo_de_ajuste")
        if not _es_finito(coste) or coste < 0:
            p.append(f"el intento {k}: `tiempo_de_ajuste` no es un número finito y >= 0 ({coste!r})")
    faltan_i = sorted(esperado_total - set(vistos))
    sobran_i = sorted(set(vistos) - esperado_total, key=str)
    dobles = sorted((k for k, n in vistos.items() if n > 1), key=str)
    if faltan_i and not es_sonda:
        p.append(f"faltan {len(faltan_i)} intentos (p. ej. {faltan_i[:3]})")
    if sobran_i:
        p.append(f"{len(sobran_i)} registros fuera de lo declarado (p. ej. {sobran_i[:3]})")
    if dobles:
        p.append(f"{len(dobles)} intentos registrados más de una vez (p. ej. {dobles[:3]})")
    return p


def problemas_de_lo_declarado(payload: dict, ref: Referencia) -> list[str]:
    """Lo que DECLARAN cada conjunto y cada registro se compara con la v2; no se cree. Por registro que cuenta como medida (y por conjunto)."""
    p: list[str] = []
    cache = payload.get("digest_de_la_cache") or {}
    if not cache.get("entorno") or not cache.get("motor"):
        p.append("la salida no declara `digest_de_la_cache.entorno` y `.motor`: no hay con qué comparar el código de cada intento")
    for c in payload.get("conjuntos", []):
        m = ref.meta.get(c.get("nombre"))
        if m is None:
            continue
        if c.get("n_predictores") != m["n_predictores"]:
            p.append(f"{c['nombre']}: declara {c.get('n_predictores')} predictores y la v2 registró {m['n_predictores']}")
        if c.get("plan_digest") != m["plan_digest"]:
            p.append(f"{c['nombre']}: su `plan_digest` no es el de la v2: otra partición")
        if m["datos_sha256"] is not None and c.get("datos_sha256") != m["datos_sha256"]:
            p.append(f"{c['nombre']}: el sha256 de los datos medidos no es el que midió la v2")
        if c.get("metrica_de_cierre") != m["metrica"] or c.get("tarea") != m["tarea"] or c.get("cubo") != m["cubo"]:
            p.append(f"{c['nombre']}: declara métrica/tarea/cubo distintos de los de la v2")
        if c.get("n_test") != m["n_test"]:
            p.append(f"{c['nombre']}: n_test={c.get('n_test')!r} y la v2 trae {m['n_test']}")
    agrupados: dict[str, list] = {}

    def mal(motivo: str, k) -> None:
        agrupados.setdefault(motivo, []).append(k)

    for r in payload.get("resultados", []):
        m = ref.meta.get(r.get("dataset"))
        if r.get("motor") != MOTOR or m is None:
            continue
        k = (r.get("dataset"), r.get("repeticion"), r.get("pliegue"))
        v2 = m["por_clave"].get((r.get("repeticion"), r.get("pliegue")))
        if v2 is None:
            mal("pliegue que la v2 no tiene", k)
            continue
        if r.get("metrica_de_cierre") != m["metrica"]:
            mal(f"métrica de cierre distinta de la de su tarea ({m['metrica']})", k)
        if r.get("semilla") != v2.get("semilla") or isinstance(r.get("semilla"), bool):
            mal("semilla distinta de la que usó la v2 en ese pliegue", k)
        if r.get("presupuesto_wall_s") != v2.get("presupuesto_wall_s"):
            mal("presupuesto de pared distinto del de la v2 en ese pliegue", k)
        if isinstance(r.get("hilos"), bool) or r.get("hilos") != HILOS_POR_INTENTO:
            mal(f"no graba {HILOS_POR_INTENTO} hilos (`hilos`)", k)
        if r.get("split_plan_digest") != m["plan_digest"]:
            mal("`split_plan_digest` distinto del `plan_digest` de la v2", k)
        if m["datos_sha256"] is not None and r.get("datos_sha256") != m["datos_sha256"]:
            mal("sha256 de los datos distinto del de la v2", k)
        if r.get("n_test") != m["n_test"]:
            mal("`n_test` distinto del de la v2", k)
        if r.get("entorno_digest") != (payload.get("digest_de_la_cache") or {}).get("entorno"):
            mal("`entorno_digest` distinto del de la salida (otro código: ¿caché de otra medida?)", k)
        if r.get("motor_digest") != (payload.get("digest_de_la_cache") or {}).get("motor"):
            mal("`motor_digest` distinto del de la salida", k)
        if r.get("estado") in ESTADOS_DE_MEDIDA:
            if r.get("engine_version") != VERSION_DEL_MOTOR:
                mal(f"`engine_version` {r.get('engine_version')!r} no es {VERSION_DEL_MOTOR!r}", k)
            if ((r.get("hiperparametros") or {}).get("interpret_core")) != VERSION_DE_INTERPRET_CORE:
                mal(f"no declara haber medido con `interpret-core` {VERSION_DE_INTERPRET_CORE}", k)
    for motivo, ks in agrupados.items():
        p.append(f"{len(ks)} intento(s) con {motivo} (p. ej. {ks[:3]})")
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
    p += problemas_del_sello_del_codigo()
    if payload.get("sello_del_codigo_al_medir") != sellos["sello_del_codigo"]:
        p.append("la salida no declara el sello del código con que se midió, o se midió con otro (`sello_del_codigo_al_medir`) que el de ahora")
    return p


def analizar(payload: dict, *, referencia: Referencia | None = None, comprobar_sellos: bool = True) -> dict:
    """El veredicto de una salida de `medir_c170.py`, citando la regla por su sha256."""
    ref = referencia if referencia is not None else cargar_referencia()
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    problemas += ref.problemas()
    problemas += problemas_de_completitud(payload, ref, es_sonda)
    problemas += problemas_de_lo_declarado(payload, ref)
    rs = payload.get("resultados", [])
    for campo in ("entorno_digest", "motor_digest"):
        if len({r.get(campo) for r in rs if r.get("motor") == MOTOR}) > 1:
            problemas.append(f"los intentos del EBM se midieron con CÓDIGO DISTINTO ({campo})")
    if comprobar_sellos:
        problemas += problemas_de_sellos(payload)
    por_clave: dict[tuple, dict] = {}
    for r in rs:
        if r.get("motor") == MOTOR:
            por_clave[(r.get("dataset"), (r.get("repeticion"), r.get("pliegue")))] = r
    regimen = ref.regimen()
    nombres = [c["nombre"] for c in payload.get("conjuntos", []) if c.get("nombre") in regimen]
    evaluados = []
    for n in sorted(set(nombres)):
        registros = {k[1]: r for k, r in por_clave.items() if k[0] == n}
        evaluados.append(evaluar_conjunto(ref, n, registros))
    completa = not problemas and evaluados and (es_sonda or {e["nombre"] for e in evaluados} == set(regimen))
    regla = aplicar_la_regla(evaluados) if completa and not es_sonda else None
    if es_sonda:
        veredicto = "SONDA"
    elif problemas or regla is None:
        veredicto = "SIN VEREDICTO"
    else:
        veredicto = regla["veredicto"]
    return {"veredicto": veredicto, "problemas": problemas, "es_sonda": es_sonda, "regla_sha256": sellos_de_ahora()["regla_sha256"],
            "regla_sha256_sellada": REGLA_SHA256_SELLADA, "sellos_de_ahora": sellos_de_ahora(), "regla": regla, "conjuntos": evaluados,
            "coste": resumen_del_coste(evaluados) if evaluados else None, "fuera_del_regimen": ref.fuera_del_regimen(),
            "lo_que_se_dice": ["Se compara lo que cada motor hace por omisión en el protocolo v2: lightgbm 100 iteraciones fijas, HGB con la política por tamaño, TabM con la receta 119 y el EBM con "
                               "su parada temprana, su validación interna y 14 bolsas.",
                               "El mejor rival por conjunto es un máximo de tres medias sobre las mismas pruebas: favorece al rival.",
                               "Cada conjunto pesa lo mismo (500 filas = 70.000); es la geometría de la Fase 0, no la del estudio del Studio.",
                               "Los conjuntos con más de 100 columnas NO se midieron: el veredicto no dice nada de ellos."]}


# ----------------------------------------------------------------------------------------------------------------------
# 5. EL INFORME
# ----------------------------------------------------------------------------------------------------------------------
def _f(x, nd: int = 4) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def informe_en_texto(r: dict) -> str:
    L = [f"=== 170: EBM (interpret.ebm) como versión legible — {r['veredicto']} ===", f"  regla sha256 {str(r.get('regla_sha256'))[:16]} (sellada: {str(r.get('regla_sha256_sellada'))[:16]})"]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  (a) conjuntos medibles: {g['a']['medibles']} de {g['n_regimen']} (>= {g['a']['umbral']}) -> {'ok' if g['a']['cumple'] else 'NO'}   no comparables: {g['no_comparables']}")
        L.append(f"  (b) cumplen: {g['b']['cumplen']}/{g['b']['medibles']} = {_f(g['b']['fraccion'], 3)} (>= {g['b']['minimo']}; hacen falta {g['b']['minimo_de_cumplen_para_este_M']}) "
                 f"-> {'ok' if g['b']['cumple'] else 'NO'}")
        L.append(f"      no cumplen por DISTANCIA: {g['b']['no_cumplen_por_distancia']}")
        L.append(f"      no cumplen por FALLO del EBM: {g['b']['no_cumplen_por_fallo']}")
        L.append(f"      el IC95 de la distancia cruza la línea de {DISTANCIA_MINIMA}: {g['el_ic_cruza_la_linea']}")
    L.append(f"  {'conjunto':34s} {'mejor rival':24s} {'media rival':>11s} {'media EBM':>10s} {'distancia':>13s}  IC95 de la distancia        cumple  s/ajuste EBM")
    for e in r["conjuntos"]:
        ic = e.get("ic95_de_la_distancia")
        dist = e.get("distancia")
        L.append(f"  {e['nombre']:34s} {str(e['mejor_rival']):24s} {_f(e['media_del_mejor_rival']):>11s} {_f(e.get('media_del_ebm')):>10s} "
                 f"{('%+.10f' % dist) if dist is not None else '—':>13s}  [{_f(ic[0]) if ic else '—'}, {_f(ic[1]) if ic else '—'}]  "
                 f"{'sí' if e['cumple'] else 'NO':>6s}  {_f(e['coste']['ebm_s_por_pliegue'], 1):>8s}  {e['categoria']}")
        for f in e["fallos_del_ebm"][:2]:
            L.append(f"      FALLO rep={f['repeticion']} pliegue={f['pliegue']} estado={f['estado']} wall={_f(f['wall_s'], 1)}: {str(f['motivo'])[:140]}")
    c = r.get("coste")
    if c:
        L.append(f"  COSTE (informativo): Σ ajuste del EBM {_f(c['ebm_s_total'], 0)} s; mediana por conjunto {_f(c['ebm_s_por_pliegue_mediana_por_conjunto'], 1)} s/pliegue, máxima "
                 f"{_f(c['ebm_s_por_pliegue_maxima_por_conjunto'], 1)}; razón EBM / (lightgbm+hgb+tabm) {_f(c['razon_de_sumas_ebm_sobre_los_tres_rivales'], 2)}; "
                 f"pliegues fallidos {c['n_pliegues_fallidos']}")
    for frase in r.get("lo_que_se_dice") or []:
        L.append(f"  NOTA: {frase}")
    return "\n".join(L)


CODIGOS_DE_SALIDA = {"CUMPLE": 0, "NO CUMPLE": 1, "SIN VEREDICTO": 2, "NO CONCLUYENTE": 3, "SONDA": 4}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    a = p.parse_args(argv)
    # una salida malformada que revienta el análisis NO puede salir con el código de NO CUMPLE (1)
    try:
        payload = json.loads(Path(a.resultado).resolve().read_text(encoding="utf-8"))
        r = analizar(payload)
    except Exception as e:  # noqa: BLE001
        print(f"SIN VEREDICTO: el análisis no se pudo completar ({type(e).__name__}: {e})")
        return CODIGOS_DE_SALIDA["SIN VEREDICTO"]
    print(informe_en_texto(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return CODIGOS_DE_SALIDA.get(r["veredicto"], CODIGOS_DE_SALIDA["SIN VEREDICTO"])


if __name__ == "__main__":
    sys.exit(main())
