#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2 — EL ANALIZADOR del banco interno de «Filas a revisar»: aplica `REGLA_C2.md` (sellada) a la salida de `medir_c165_c2.py`.

Dice qué defecto de cada eje se elige (`escalado`, `categoricas`, `max_samples`), que `contamination` queda sin definir, y las cifras SOLO para
la frase de límites («sobre anomalías sembradas por MatrixAI, no sobre las tuyas»). Veredicto ∈ DECIDIDO / SIN VEREDICTO / SONDA.

    python3 benchmarks/c165/analiza_c165_c2.py <salida.json> [--json veredicto.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
import banco_c165 as banco  # noqa: E402

RUTA_DE_LA_REGLA = _AQUI / "REGLA_C2.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()
#: sha256 de `REGLA_C2.md` SELLADA (la regla se escribió antes de existir ninguna cifra). `None` = sin sellar: ninguna medida vale.
REGLA_SHA256_SELLADA: str | None = "bab0a2c71b634090a2dd3cb732eea0a5f422461b824f5dbdb291a464a0c17855"

UMBRAL_DELTA = 0.03            # (a) media de Δ
UMBRAL_DELTA_CATEGORICAS = 0.05
FRACCION_NO_PEOR = 0.60        # (b)
CAIDA_MAXIMA_POR_TIPO = -0.03  # (c)
AVISO_DE_INTERACCION = 0.03    # §3
METRICA = "recuperacion_doble"
EJES = {"escalado": banco.ESCALADOS, "categoricas": banco.CATEGORICAS, "max_samples": banco.MAX_SAMPLES}
N_CONJUNTOS = len(banco.CONJUNTOS)
NOMBRES = [c[0] for c in banco.CONJUNTOS]


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia: el detector del producto, el banco y los 6 CSV."""
    d = {"matrixai_engines/anomalias.py": _sha256(Path(banco._an.__file__).resolve()),
         "benchmarks/c165/banco_c165.py": _sha256(Path(banco.__file__).resolve())}
    for nombre, csv_, _ in banco.CONJUNTOS:
        d[f"datos/{csv_}"] = _sha256(banco.DIR_DATOS / csv_)
    return d


def _media(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def celdas(registros: list[dict]) -> dict[tuple, dict[str, float]]:
    """{(conjunto, tipo, tasa): {nombre_de_variante: M medio de las repeticiones}}."""
    acum: dict[tuple, dict[str, list[float]]] = {}
    for r in registros:
        acum.setdefault((r["dataset"], r["tipo"], r["tasa"]), {}).setdefault(r["variante"], []).append(r["metricas"][METRICA])
    return {k: {v: _media(xs) for v, xs in d.items()} for k, d in acum.items()}


def _variante(**cambio) -> str:
    return banco.nombre_de_variante({**banco.HOY, **cambio})


def evaluar_eje(eje: str, tabla: dict[tuple, dict[str, float]]) -> dict:
    hoy = _variante()
    umbral = UMBRAL_DELTA_CATEGORICAS if eje == "categoricas" else UMBRAL_DELTA
    candidatos = []
    for valor in EJES[eje]:
        if valor == banco.HOY[eje]:
            continue
        cand = _variante(**{eje: valor})
        # celdas afectadas: aquellas donde el orden de ALGUNA repetición cambia no se conoce aquí; por la regla, `categoricas` solo cuenta los
        # conjuntos con categóricas y los otros ejes todas las celdas.
        afectadas = {k: v for k, v in tabla.items() if eje != "categoricas" or k[0] in banco_con_categoricas(tabla)}
        deltas = {k: v[cand] - v[hoy] for k, v in afectadas.items()}
        por_tipo: dict[str, list[float]] = {}
        for k, d in deltas.items():
            por_tipo.setdefault(k[1], []).append(d)
        media = _media(list(deltas.values())) if deltas else None
        frac = (sum(1 for d in deltas.values() if d >= 0) / len(deltas)) if deltas else None
        medias_tipo = {t: _media(ds) for t, ds in por_tipo.items()}
        a = media is not None and media >= umbral
        b = frac is not None and frac >= FRACCION_NO_PEOR
        c = all(m >= CAIDA_MAXIMA_POR_TIPO for m in medias_tipo.values())
        candidatos.append({"valor": valor, "media_delta": media, "fraccion_no_peor": frac, "media_por_tipo": medias_tipo,
                           "n_celdas": len(deltas), "a": a, "b": b, "c": c, "cumple": bool(a and b and c)})
    buenos = [c for c in candidatos if c["cumple"]]
    elegido = max(buenos, key=lambda c: c["media_delta"])["valor"] if buenos else banco.HOY[eje]
    return {"eje": eje, "hoy": banco.HOY[eje], "umbral_a": umbral, "candidatos": candidatos, "elegido": elegido,
            "cambia": elegido != banco.HOY[eje]}





def banco_con_categoricas(tabla) -> set[str]:
    """Conjuntos con categóricas = los que tienen celdas `rara` (el tipo solo existe donde las hay)."""
    return {k[0] for k in tabla if k[1] == "rara"}


def esperado_por_declaracion(payload: dict) -> set[tuple]:
    esperado = set()
    for c in payload.get("conjuntos", []):
        for tipo, motivo in (c.get("aplicabilidad") or {}).items():
            if motivo is None:
                for tasa in banco.TASAS:
                    for rep in banco.REPETICIONES:
                        for v in banco.VARIANTES:
                            esperado.add((c["nombre"], tipo, tasa, rep, banco.nombre_de_variante(v)))
    return esperado


def problemas_de_completitud(payload: dict, es_sonda: bool) -> list[str]:
    P: list[str] = []
    cs = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in cs]
    if len(set(nombres)) != len(nombres):
        P.append("conjuntos duplicados en la salida")
    if payload.get("parcial"):
        P.append("la salida es PARCIAL")
    if not es_sonda:
        if sorted(nombres) != sorted(NOMBRES):
            P.append(f"los conjuntos no son los {N_CONJUNTOS} de la regla: faltan {sorted(set(NOMBRES) - set(nombres))}, "
                     f"sobran {sorted(set(nombres) - set(NOMBRES))}")
        for c in cs:
            ap = c.get("aplicabilidad") or {}
            if ap.get("extremo") is not None:
                P.append(f"{c.get('nombre')}: `extremo` no es aplicable ({ap.get('extremo')}): la regla lo exige en los {N_CONJUNTOS}")
            if c.get("nombre") == "dresses-sales" and ap.get("rara") is not None:
                P.append("dresses-sales: `rara` no es aplicable: la regla lo exige ahí")
            if c.get("control_hoy_igual_al_producto") is not True:
                P.append(f"{c.get('nombre')}: la variante «hoy» NO reproduce el orden de `puntuar_filas` del producto")
            if c.get("control_contamination_invariante") is not True:
                P.append(f"{c.get('nombre')}: `contamination` cambia las puntuaciones (o no se comprobó)")
    esperado = esperado_por_declaracion(payload) if not es_sonda else None
    vistos: dict[tuple, int] = {}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("tipo"), r.get("tasa"), r.get("repeticion"), r.get("variante"))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("metricas") is None or METRICA not in r["metricas"]:
            P.append(f"el registro {k} no trae `{METRICA}`")
        # la fila sembrada tiene que coincidir con lo grabado: n_sembradas = la lista
        s = r.get("sembradas") or []
        if len(s) != len(set(s)) or len(s) != (r.get("metricas") or {}).get("n_sembradas"):
            P.append(f"el registro {k}: la lista de sembradas no cuadra con n_sembradas")
        elif len(s) != banco.n_sembradas(next((c.get("filas") for c in cs if c.get("nombre") == k[0]), 0), k[2] or 0):
            P.append(f"el registro {k}: n_sembradas distinto del que fija la regla")
    if esperado is not None:
        faltan, sobran = sorted(esperado - set(vistos)), sorted(set(vistos) - esperado)
        if faltan:
            P.append(f"faltan {len(faltan)} registros (p. ej. {faltan[:2]})")
        if sobran:
            P.append(f"{len(sobran)} registros fuera de lo declarado (p. ej. {sobran[:2]})")
    dobles = sorted(k for k, n in vistos.items() if n > 1)
    if dobles:
        P.append(f"{len(dobles)} registros repetidos (p. ej. {dobles[:2]})")
    # cada celda con TODAS las variantes y repeticiones (también en la sonda: una celda a medias no se compara)
    por_celda: dict[tuple, set] = {}
    for (d, t, tasa, rep, v) in vistos:
        por_celda.setdefault((d, t, tasa), set()).add(v)
    for k, vs in por_celda.items():
        if vs != {banco.nombre_de_variante(v) for v in banco.VARIANTES}:
            P.append(f"la celda {k} no tiene las 12 variantes")
    return P


def analizar(payload: dict, *, comprobar_sellos: bool = True) -> dict:
    es_sonda = bool(payload.get("es_sonda"))
    problemas = problemas_de_completitud(payload, es_sonda)
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
                    problemas.append(f"`{etiqueta}` al medir ({str(al_medir.get(etiqueta))[:12]}) != el de ahora ({propio[:12]})")
        for k, propio in (("regla_sha256_al_medir", sellos["regla_sha256"]), ("analizador_sha256_al_medir", sellos["analizador_sha256"])):
            v = payload.get(k)
            if v is None:
                problemas.append(f"la salida no declara `{k}`")
            elif v != propio:
                problemas.append(f"`{k}` de la salida ({v[:12]}) != el fichero de ahora ({str(propio)[:12]}): cambió DESPUÉS de medir")
    tabla = celdas(payload.get("resultados", [])) if not problemas or es_sonda else {}
    ejes, limites, interaccion, elegida = [], [], None, dict(banco.HOY)
    if tabla and not problemas:
        ejes = [evaluar_eje(e, tabla) for e in EJES]
        elegida = {e["eje"]: e["elegido"] for e in ejes}
        nombre = banco.nombre_de_variante(elegida)
        hoy = banco.nombre_de_variante(banco.HOY)
        if nombre != hoy:
            dif = _media([v[nombre] - v[hoy] for v in tabla.values()])
            interaccion = {"media_delta_combinacion": dif, "peor_que_hoy": dif < -AVISO_DE_INTERACCION}
            if interaccion["peor_que_hoy"]:
                elegida = dict(banco.HOY)
                nombre = hoy
        por: dict[tuple, list[float]] = {}
        for (d, t, tasa), v in tabla.items():
            por.setdefault((t, tasa), []).append(v[nombre])
        limites = [{"tipo": t, "tasa": tasa, "recuperacion_doble_media": _media(xs), "celdas": len(xs)} for (t, tasa), xs in sorted(por.items())]
    if problemas:
        veredicto = "SIN VEREDICTO"
    elif es_sonda:
        veredicto = "SONDA"
    else:
        veredicto = "DECIDIDO"
    return {"veredicto": veredicto, "problemas": problemas, "es_sonda": es_sonda, "sellos_de_ahora": sellos,
            "defectos_elegidos": {**elegida, "contamination": None} if veredicto == "DECIDIDO" else None,
            "ejes": ejes, "interaccion": interaccion, "para_el_texto_de_limites": limites if veredicto == "DECIDIDO" else [],
            "aviso": "Cifras sobre anomalías sembradas por MatrixAI, no sobre las del usuario; no son rendimiento de producto."}


def informe_en_texto(r: dict) -> str:
    L = [f"=== 165-C2: banco interno de anomalías sembradas — {r['veredicto']} ==="]
    L += [f"  PROBLEMA: {p}" for p in r["problemas"]]
    for e in r["ejes"]:
        L.append(f"  eje {e['eje']}: hoy={e['hoy']} -> elegido={e['elegido']}")
        for c in e["candidatos"]:
            L.append(f"     {c['valor']:10s} Δ medio {c['media_delta']:+.4f} (>= {e['umbral_a']}) no-peor {c['fraccion_no_peor']:.0%} "
                     f"celdas {c['n_celdas']} a={c['a']} b={c['b']} c={c['c']} por tipo {({t: round(m, 3) for t, m in c['media_por_tipo'].items()})}")
    if r["defectos_elegidos"]:
        L.append(f"  defectos: {r['defectos_elegidos']}")
    for x in r["para_el_texto_de_limites"]:
        L.append(f"  límites: {x['tipo']:11s} tasa {x['tasa']:.0%}: recuperación@2n media {x['recuperacion_doble_media']:.3f} ({x['celdas']} celdas)")
    L.append("  " + r["aviso"])
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    r = analizar(json.loads(Path(a.resultado).resolve().read_text(encoding="utf-8")))
    print(informe_en_texto(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return {"DECIDIDO": 0, "SONDA": 0}.get(r["veredicto"], 2)


if __name__ == "__main__":
    sys.exit(main())
