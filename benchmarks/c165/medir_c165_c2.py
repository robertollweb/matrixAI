#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C2 — LA MEDIDA del banco interno de «Filas a revisar»: siembra anomalías de tres tipos en 6 conjuntos del Studio y mide, para las 12
variantes de defectos, cuánto del sembrado recupera el detector. Aplica la regla `analiza_c165_c2.py` (sellada en `REGLA_C2.md`); este guion solo
mide y guarda. Se NIEGA a medir si la regla no está sellada o ya no es la sellada. Salida atómica y reanudable (por celda).

    python3 benchmarks/c165/medir_c165_c2.py --salida <ruta absoluta>            # la medida (a la COLA)
    python3 benchmarks/c165/medir_c165_c2.py --sonda --salida <ruta>              # 2 conjuntos, 1 repetición: no mide nada del corte
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
import analiza_c165_c2 as an  # noqa: E402
import banco_c165 as banco  # noqa: E402

CONJUNTOS_DE_LA_SONDA = ("dresses-sales", "breast-cancer-wisconsin-diagnostic")
REPETICIONES_DE_LA_SONDA = (0,)


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
        raise SystemExit("la regla NO está sellada (`analiza_c165_c2.REGLA_SHA256_SELLADA` es None): una medida real no empieza sin el sello")
    if sha != an.REGLA_SHA256_SELLADA:
        raise SystemExit(f"REGLA_C2.md ({str(sha)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la regla es una "
                         f"enmienda con su sello, no una edición. No se mide")


def medir(nombres, repeticiones, ruta: Path, es_sonda: bool, *, escribir=_escribir) -> dict:
    payload = {"corte": "165-C2", "es_sonda": es_sonda, "parcial": True, "conjuntos": [], "resultados": [],
               "regla_sha256_al_medir": an._sha256(an.RUTA_DE_LA_REGLA), "analizador_sha256_al_medir": an._sha256(an.RUTA_DEL_ANALIZADOR),
               "dependencias_sha256_al_medir": an.dependencias_de_ahora(), "segundos_por_conjunto": {}}
    for nombre in nombres:
        t0 = time.time()
        filas = banco.cargar(nombre)
        ap = banco.aplicabilidad(filas)
        payload["conjuntos"].append({"nombre": nombre, "filas": len(filas), "columnas": len(filas[0]), "aplicabilidad": ap,
                                     "control_hoy_igual_al_producto": banco.control_hoy_igual_al_producto(filas),
                                     "control_contamination_invariante": banco.control_contamination_invariante(filas)})
        for tipo in banco.TIPOS:
            if ap[tipo] is not None:
                continue
            for tasa in banco.TASAS:
                for rep in repeticiones:
                    nuevas, sembradas, detalle = banco.sembrar(filas, tipo, tasa, rep, nombre)
                    for v in banco.VARIANTES:
                        orden = banco.ordenar(nuevas, v)
                        payload["resultados"].append({"dataset": nombre, "tipo": tipo, "tasa": tasa, "repeticion": rep,
                                                      "variante": banco.nombre_de_variante(v), "sembradas": sembradas,
                                                      "detalle_siembra": detalle, "metricas": banco.metricas(orden, sembradas)})
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
    nombres = list(CONJUNTOS_DE_LA_SONDA) if a.sonda else an.NOMBRES
    reps = REPETICIONES_DE_LA_SONDA if a.sonda else banco.REPETICIONES
    medir(nombres, reps, Path(a.salida).expanduser().resolve(), bool(a.sonda))


if __name__ == "__main__":
    main()
