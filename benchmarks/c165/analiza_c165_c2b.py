#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2b — EL ANALIZADOR de la codificación de categóricas de «Filas a revisar»: aplica `REGLA_C2B.md` (sellada) a la salida de
`medir_c165_c2b.py`. Dice qué `orden` y qué `vacia` se eligen y la cifra de la categoría rara SOLO para la frase de límites («sobre anomalías
sembradas por MatrixAI…»). Veredicto ∈ DECIDIDO / SIN VEREDICTO / SONDA.

    python3 benchmarks/c165/analiza_c165_c2b.py <salida.json> [--json veredicto.json]
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
import banco_c165_c2b as b2  # noqa: E402

RUTA_DE_LA_REGLA = _AQUI / "REGLA_C2B.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()
#: sha256 de `REGLA_C2B.md` SELLADA (escrita antes de existir ninguna cifra). `None` = sin sellar: ninguna medida vale.
REGLA_SHA256_SELLADA: str | None = "845586e707ee700c259a2c6db2f765bd9092b6cf127b74b1b2c7f8e44bcd2892"

UMBRAL_PERDIDA = -0.02          # (a)
CAIDA_MAXIMA_POR_TIPO = -0.05   # (b)
UMBRAL_S = 0.10                 # (c)
UMBRAL_GANA = 0.03              # (c), segunda vía
TOLERANCIA_S = 0.02             # (c), segunda vía
UMBRAL_D = 0.5                  # (c')
AVISO_DE_INTERACCION = 0.03     # combinación
NEUTRA = (0.3, 0.7)             # posición media de las siembras neutras
METRICA = "recuperacion_doble"
NOMBRES = list(b2.NOMBRES)


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia: el detector del producto, los dos bancos y los 6 CSV."""
    banco = b2.banco
    d = {"matrixai_engines/anomalias.py": _sha256(Path(banco._an.__file__).resolve()),
         "benchmarks/c165/banco_c165.py": _sha256(Path(banco.__file__).resolve()),
         "benchmarks/c165/banco_c165_c2b.py": _sha256(Path(b2.__file__).resolve())}
    for nombre, csv_, _ in banco.CONJUNTOS:
        if nombre in b2.NOMBRES:
            d[f"datos/{csv_}"] = _sha256(banco.DIR_DATOS / csv_)
    return d


def _media(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def celdas(registros: list[dict]) -> dict[tuple, dict[str, float]]:
    """{(conjunto, tipo, tasa): {variante: M medio de las repeticiones}}."""
    acum: dict[tuple, dict[str, list[float]]] = {}
    for r in registros:
        acum.setdefault((r["dataset"], r["tipo"], r["tasa"]), {}).setdefault(r["variante"], []).append(r["metricas"][METRICA])
    return {k: {v: _media(xs) for v, xs in d.items()} for k, d in acum.items()}


def por_variante(registros: list[dict], campo: str) -> dict[str, float]:
    acum: dict[str, list[float]] = {}
    for r in registros:
        if r.get(campo) is not None:
            acum.setdefault(r["variante"], []).append(float(r[campo]))
    return {v: _media(xs) for v, xs in acum.items()}


def D_por_variante(inflacion: list[dict]) -> dict[str, float]:
    acum: dict[str, list[float]] = {}
    for r in inflacion:
        if r.get("inflacion") is not None:
            acum.setdefault(r["variante"], []).append(abs(float(r["inflacion"]) - 1.0))
    return {v: _media(xs) for v, xs in acum.items()}


def _v(**cambio) -> str:
    return b2.nombre_de_variante({**b2.HOY, **cambio})


def _deltas(tabla: dict, cand: str, hoy: str) -> tuple[float | None, dict[str, float]]:
    deltas = {k: v[cand] - v[hoy] for k, v in tabla.items() if k[1] in b2.TIPOS_DE_DECISION}
    por_tipo: dict[str, list[float]] = {}
    for k, d in deltas.items():
        por_tipo.setdefault(k[1], []).append(d)
    return (_media(list(deltas.values())) if deltas else None), {t: _media(ds) for t, ds in por_tipo.items()}


def evaluar_orden(tabla: dict, S: dict[str, float]) -> dict:
    hoy = _v()
    cand = _v(orden="frecuencia")
    media, por_tipo = _deltas(tabla, cand, hoy)
    a = media is not None and media >= UMBRAL_PERDIDA
    b = all(m >= CAIDA_MAXIMA_POR_TIPO for m in por_tipo.values())
    c = (S[cand] - S[hoy] >= UMBRAL_S) or (media is not None and media >= UMBRAL_GANA and S[cand] >= S[hoy] - TOLERANCIA_S)
    cumple = bool(a and b and c)
    return {"eje": "orden", "hoy": b2.HOY["orden"], "candidatos": [{"valor": "frecuencia", "media_delta": media, "media_por_tipo": por_tipo,
                                                                     "S": S[cand], "S_hoy": S[hoy], "a": a, "b": b, "c": c, "cumple": cumple}],
            "elegido": "frecuencia" if cumple else b2.HOY["orden"]}


def evaluar_vacia(tabla: dict, D: dict[str, float]) -> dict:
    hoy = _v()
    candidatos = []
    for valor in b2.VACIAS[1:]:
        cand = _v(vacia=valor)
        media, por_tipo = _deltas(tabla, cand, hoy)
        a = media is not None and media >= UMBRAL_PERDIDA
        b = all(m >= CAIDA_MAXIMA_POR_TIPO for m in por_tipo.values())
        c = D[cand] <= D[hoy] - UMBRAL_D
        candidatos.append({"valor": valor, "media_delta": media, "media_por_tipo": por_tipo, "D": D[cand], "D_hoy": D[hoy],
                           "a": a, "b": b, "c": c, "cumple": bool(a and b and c)})
    buenos = [c for c in candidatos if c["cumple"]]
    elegido = min(buenos, key=lambda c: c["D"])["valor"] if buenos else b2.HOY["vacia"]
    return {"eje": "vacia", "hoy": b2.HOY["vacia"], "candidatos": candidatos, "elegido": elegido}


def problemas_de_completitud(payload: dict, es_sonda: bool) -> list[str]:
    P: list[str] = []
    cs = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in cs]
    if len(set(nombres)) != len(nombres):
        P.append("conjuntos duplicados en la salida")
    if payload.get("parcial"):
        P.append("la salida es PARCIAL")
    if not es_sonda and sorted(nombres) != sorted(NOMBRES):
        P.append(f"los conjuntos no son los 6 de la regla: faltan {sorted(set(NOMBRES) - set(nombres))}, sobran {sorted(set(nombres) - set(NOMBRES))}")
    for c in cs:
        if c.get("control_hoy_igual_al_producto") is not True:
            P.append(f"{c.get('nombre')}: la variante «hoy» NO reproduce el orden de `puntuar_filas` del producto")
        if not c.get("categoricas"):
            P.append(f"{c.get('nombre')}: sin categóricas (no hay `rara_*`)")
    tasas = tuple(payload.get("tasas") or ()) if es_sonda else b2.TASAS
    reps = tuple(payload.get("repeticiones") or ()) if es_sonda else b2.REPETICIONES
    if not es_sonda and (tuple(payload.get("tasas") or ()) != b2.TASAS or tuple(payload.get("repeticiones") or ()) != b2.REPETICIONES):
        P.append("las tasas o las repeticiones no son las de la regla")
    variantes = {b2.nombre_de_variante(v) for v in b2.VARIANTES}
    esperado = {(n, t, ta, r, v) for n in nombres for t in b2.TIPOS for ta in tasas for r in reps for v in variantes}
    vistos: dict[tuple, int] = {}
    filas_de = {c.get("nombre"): c.get("filas") for c in cs}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("tipo"), r.get("tasa"), r.get("repeticion"), r.get("variante"))
        vistos[k] = vistos.get(k, 0) + 1
        m = r.get("metricas") or {}
        if METRICA not in m:
            P.append(f"el registro {k} no trae `{METRICA}`")
        s = r.get("sembradas") or []
        if len(s) != len(set(s)) or len(s) != m.get("n_sembradas"):
            P.append(f"el registro {k}: la lista de sembradas no cuadra con n_sembradas")
        elif len(s) != b2.banco.n_sembradas(filas_de.get(k[0]) or 0, k[2] or 0):
            P.append(f"el registro {k}: n_sembradas distinto del que fija la regla")
    faltan, sobran = sorted(esperado - set(vistos)), sorted(set(vistos) - esperado)
    if faltan:
        P.append(f"faltan {len(faltan)} registros de siembra (p. ej. {faltan[:2]})")
    if sobran:
        P.append(f"{len(sobran)} registros fuera de lo declarado (p. ej. {sobran[:2]})")
    dobles = sorted(k for k, n in vistos.items() if n > 1)
    if dobles:
        P.append(f"{len(dobles)} registros repetidos (p. ej. {dobles[:2]})")
    for n in nombres:
        for v in variantes:
            if not any(r.get("dataset") == n and r.get("variante") == v for r in payload.get("renombres", [])):
                P.append(f"{n} / {v}: faltan los renombres")
            if sorted(r.get("semilla") for r in payload.get("semillas", []) if r.get("dataset") == n and r.get("variante") == v) != [1, 2, 3, 4]:
                P.append(f"{n} / {v}: faltan las semillas 1–4 de la referencia")
            if not any(r.get("dataset") == n and r.get("variante") == v for r in payload.get("inflacion", [])):
                P.append(f"{n} / {v}: falta la inflación por la vacía")
    for tipo in ("rara_medio", "rara_azar"):
        pos = [r["detalle_siembra"]["posicion_media"] for r in payload.get("resultados", [])
               if r.get("tipo") == tipo and isinstance(r.get("detalle_siembra"), dict) and "posicion_media" in r["detalle_siembra"]]
        if pos and not NEUTRA[0] <= _media(pos) <= NEUTRA[1]:
            P.append(f"la siembra `{tipo}` no es neutra: posición alfabética media {_media(pos):.2f} fuera de {list(NEUTRA)}")
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
            problemas.append(f"REGLA_C2B.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
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
    tabla = celdas(payload.get("resultados", []))
    S = por_variante(payload.get("renombres", []), "jaccard")
    E = por_variante(payload.get("semillas", []), "jaccard")
    D = D_por_variante(payload.get("inflacion", []))
    ejes, interaccion, elegida, limites, informe = [], None, dict(b2.HOY), [], {}
    if tabla and not (problemas and not es_sonda) and all(len(v) == len(b2.VARIANTES) for v in tabla.values()) \
            and len(S) == len(D) == len(b2.VARIANTES):
        ejes = [evaluar_orden(tabla, S), evaluar_vacia(tabla, D)]
        elegida = {e["eje"]: e["elegido"] for e in ejes}
        nombre, hoy = b2.nombre_de_variante(elegida), b2.nombre_de_variante(b2.HOY)
        if nombre != hoy:
            dif, _ = _deltas(tabla, nombre, hoy)
            interaccion = {"media_delta_combinacion": dif, "peor_que_hoy": dif is not None and dif < -AVISO_DE_INTERACCION}
            if interaccion["peor_que_hoy"]:
                elegida, nombre = dict(b2.HOY), hoy
        for tasa in sorted({k[2] for k in tabla}):
            neutras = [v[nombre] for (d, t, ta), v in tabla.items() if ta == tasa and t in ("rara_medio", "rara_azar")]
            final = [v[nombre] for (d, t, ta), v in tabla.items() if ta == tasa and t == "rara_final"]
            limites.append({"tasa": tasa, "rara_neutra_media": _media(neutras) if neutras else None,
                            "rara_final_media_solo_informe": _media(final) if final else None, "celdas": len(neutras)})
        informe = {"S": S, "E": E, "D": D}
    veredicto = "SIN VEREDICTO" if problemas else ("SONDA" if es_sonda else "DECIDIDO")
    return {"veredicto": veredicto, "problemas": problemas, "es_sonda": es_sonda, "sellos_de_ahora": sellos,
            "codificacion_elegida": elegida if veredicto == "DECIDIDO" else None, "ejes": ejes, "interaccion": interaccion,
            "para_el_texto_de_limites": limites if veredicto == "DECIDIDO" else [], "informe": informe,
            "limites_de_la_sonda_no_deciden": limites if veredicto == "SONDA" else [],
            "aviso": "Cifras sobre anomalías sembradas por MatrixAI (1 conjunto real y 5 derivados), no sobre las del usuario; "
                     "no son rendimiento de producto."}


def informe_en_texto(r: dict) -> str:
    L = [f"=== 165-C2b: codificación de categóricas — {r['veredicto']} ==="]
    L += [f"  PROBLEMA: {p}" for p in r["problemas"]]
    for e in r["ejes"]:
        L.append(f"  eje {e['eje']}: hoy={e['hoy']} -> elegido={e['elegido']}")
        for c in e["candidatos"]:
            extra = f"S {c['S']:.3f} (hoy {c['S_hoy']:.3f})" if "S" in c else f"D {c['D']:.3f} (hoy {c['D_hoy']:.3f})"
            L.append(f"     {c['valor']:12s} Δ medio {c['media_delta']:+.4f} {extra} a={c['a']} b={c['b']} c={c['c']} "
                     f"por tipo {({t: round(m, 3) for t, m in c['media_por_tipo'].items()})}")
    for nombre, d in (r.get("informe") or {}).items():
        L.append(f"  {nombre}: " + ", ".join(f"{v}={x:.3f}" for v, x in sorted(d.items())))
    if r["codificacion_elegida"]:
        L.append(f"  codificación: {r['codificacion_elegida']}")
    for x in r["para_el_texto_de_limites"] or r["limites_de_la_sonda_no_deciden"]:
        L.append(f"  límites: tasa {x['tasa']:.0%}: rara (neutra) {x['rara_neutra_media']:.3f}; rara_final (solo informe) "
                 f"{x['rara_final_media_solo_informe']:.3f} ({x['celdas']} celdas)")
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
