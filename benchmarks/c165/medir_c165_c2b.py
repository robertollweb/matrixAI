#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2b — LA MEDIDA de la codificación de categóricas de «Filas a revisar»: 6 variantes × siembras de categoría rara con nombres neutros (y la del
C2), extremo, sensibilidad a renombrar, referencia de la semilla e inflación por la vacía, en 1 conjunto real y 5 derivados. Aplica la regla
`analiza_c165_c2b.py` (sellada en `REGLA_C2B.md`); este guion solo mide y guarda. Se NIEGA a medir si la regla no está sellada o no es la sellada.
Salida atómica, escrita al acabar cada conjunto.

    python3 benchmarks/c165/medir_c165_c2b.py --salida <ruta absoluta>          # la medida (a la COLA)
    python3 benchmarks/c165/medir_c165_c2b.py --sonda --salida <ruta>            # 2 conjuntos, 1 repetición, 1 tasa: no decide nada
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
import analiza_c165_c2b as an  # noqa: E402
import banco_c165_c2b as b2  # noqa: E402

CONJUNTOS_DE_LA_SONDA = ("dresses-sales", "breast-cancer-wisconsin-diagnostic")
REPETICIONES_DE_LA_SONDA = (0,)
TASAS_DE_LA_SONDA = (0.05,)


def _escribir(ruta: Path, payload: dict) -> None:
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=float), encoding="utf-8")
    os.replace(tmp, ruta)


def comprobar_sello(es_sonda: bool) -> None:
    """Una medida real no empieza sin el sello; la sonda sí (no decide nada)."""
    if es_sonda:
        return
    sha = an._sha256(an.RUTA_DE_LA_REGLA) if an.RUTA_DE_LA_REGLA.exists() else None
    if an.REGLA_SHA256_SELLADA is None:
        raise SystemExit("la regla NO está sellada (`analiza_c165_c2b.REGLA_SHA256_SELLADA` es None): una medida real no empieza sin el sello")
    if sha != an.REGLA_SHA256_SELLADA:
        raise SystemExit(f"REGLA_C2B.md ({str(sha)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la regla es una "
                         "enmienda con su sello, no una edición. No se mide")


def medir_conjunto(nombre: str, repeticiones, tasas) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    filas, detalle = b2.cargar(nombre)
    conjunto = {"nombre": nombre, "filas": len(filas), "columnas": len(filas[0]), "detalle": detalle,
                "categoricas": b2.categoricas(filas), "control_hoy_igual_al_producto": b2.control_hoy_igual_al_producto(filas)}
    resultados, renom, semillas, inflacion = [], [], [], []
    k = b2.k_del_producto(len(filas))
    for v in b2.VARIANTES:
        nv = b2.nombre_de_variante(v)
        base = b2.ordenar(filas, v, 0)
        inflacion.append({"dataset": nombre, "variante": nv, "inflacion": b2.inflacion_por_la_vacia(filas, base)})
        for s in (1, 2, 3, 4):
            semillas.append({"dataset": nombre, "variante": nv, "semilla": s,
                             "jaccard": b2.jaccard(base[:k], b2.ordenar(filas, v, s)[:k])})
        for r in b2.renombres(filas):
            renom.append({"dataset": nombre, "variante": nv, **r,
                          "jaccard": b2.jaccard(base[:k], b2.ordenar(b2.renombrar(filas, r), v, 0)[:k])})
    for tipo in b2.TIPOS:
        for tasa in tasas:
            for rep in repeticiones:
                nuevas, sembradas, det = b2.sembrar(filas, tipo, tasa, rep, nombre)
                for v in b2.VARIANTES:
                    resultados.append({"dataset": nombre, "tipo": tipo, "tasa": tasa, "repeticion": rep,
                                       "variante": b2.nombre_de_variante(v), "sembradas": sembradas, "detalle_siembra": det,
                                       "metricas": b2.metricas(b2.ordenar(nuevas, v, 0), sembradas)})
    return conjunto, resultados, renom, semillas, inflacion


def medir(nombres, repeticiones, tasas, ruta: Path, es_sonda: bool, *, escribir=_escribir) -> dict:
    payload = {"corte": "165-C2b", "es_sonda": es_sonda, "parcial": True, "conjuntos": [], "resultados": [], "renombres": [],
               "semillas": [], "inflacion": [], "tasas": list(tasas), "repeticiones": list(repeticiones),
               "regla_sha256_al_medir": an._sha256(an.RUTA_DE_LA_REGLA), "analizador_sha256_al_medir": an._sha256(an.RUTA_DEL_ANALIZADOR),
               "dependencias_sha256_al_medir": an.dependencias_de_ahora(), "segundos_por_conjunto": {}}
    for nombre in nombres:
        t0 = time.time()
        c, res, ren, sem, inf = medir_conjunto(nombre, repeticiones, tasas)
        payload["conjuntos"].append(c)
        payload["resultados"] += res
        payload["renombres"] += ren
        payload["semillas"] += sem
        payload["inflacion"] += inf
        payload["segundos_por_conjunto"][nombre] = round(time.time() - t0, 1)
        if escribir is not None:
            escribir(ruta, payload)
        print(f"{nombre}: {payload['segundos_por_conjunto'][nombre]} s", flush=True)
    payload["parcial"] = False
    if escribir is not None:
        escribir(ruta, payload)
    return payload


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", required=True)
    p.add_argument("--sonda", action="store_true")
    a = p.parse_args(argv)
    comprobar_sello(bool(a.sonda))
    if a.sonda:
        medir(list(CONJUNTOS_DE_LA_SONDA), REPETICIONES_DE_LA_SONDA, TASAS_DE_LA_SONDA, Path(a.salida).expanduser().resolve(), True)
    else:
        medir(list(b2.NOMBRES), b2.REPETICIONES, b2.TASAS, Path(a.salida).expanduser().resolve(), False)


if __name__ == "__main__":
    main()
