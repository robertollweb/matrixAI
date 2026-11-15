#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169 — ESCRIBE el sello del CÓDIGO (`SELLO_DEL_CODIGO.json`): el sha256 del analizador, del medidor y de las dependencias del veredicto.

Lo corre el SUPERVISOR tras la última auditoría y antes de encolar (rep1, I1 de la aud1). El orden es regla -> analizador -> sello:
  1. `REGLA.md` ya sellada en `analiza_c169.REGLA_SHA256_SELLADA` (si no lo está, o si REGLA.md ya no es ésa, esto se NIEGA);
  2. el analizador, con ese sha DENTRO, entra en el sello por su propio sha256;
  3. el sello NO se incluye a sí mismo (`codigo_de_ahora()` no lo lista).

    python3 benchmarks/c169/sellar_codigo_c169.py              # enseña lo que escribiría y si el sello de ahora cuadra
    python3 benchmarks/c169/sellar_codigo_c169.py --escribir   # lo escribe (atómico)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import analiza_c169 as an  # noqa: E402

QUE_ES = ("sha256 del analizador, del medidor y de las dependencias del veredicto del 169 (analiza_c169.codigo_de_ahora). Lo escribe "
          "sellar_codigo_c169.py; nulo = sin sellar. El medidor no mide (salvo --sonda) y el analizador no da veredicto si falta, es nulo o no "
          "cuadra. No se incluye a sí mismo.")


def sello_sin_sellar() -> dict:
    """El fichero tal como se commitea ANTES de sellar: las mismas etiquetas, con valores nulos."""
    return {"que_es": QUE_ES, "regla_sha256_sellada": None, "sha256": {k: None for k in sorted(an.codigo_de_ahora())}}


def componer_el_sello() -> dict:
    """El sello de AHORA. Se niega si la regla no está sellada o si REGLA.md ya no es la sellada."""
    if an.REGLA_SHA256_SELLADA is None:
        raise SystemExit("la regla NO está sellada (`analiza_c169.REGLA_SHA256_SELLADA` es None): primero la regla, después el código")
    sha_regla = an.sellos_de_ahora()["regla_sha256"]
    if sha_regla != an.REGLA_SHA256_SELLADA:
        raise SystemExit(f"REGLA.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): no se sella el código sobre otra regla")
    codigo = an.codigo_de_ahora()
    if any(an.RUTA_DEL_SELLO_DEL_CODIGO.name in k for k in codigo):
        raise SystemExit("el sello no puede incluirse a sí mismo")
    return {"que_es": QUE_ES, "regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "sha256": dict(sorted(codigo.items()))}


def escribir(sello: dict, ruta: Path) -> None:
    ruta = Path(ruta)
    tmp = ruta.with_suffix(f".json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(sello, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, ruta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--escribir", action="store_true")
    ap.add_argument("--ruta", default=None, help="por omisión, benchmarks/c169/SELLO_DEL_CODIGO.json")
    a = ap.parse_args(argv)
    ruta = Path(a.ruta) if a.ruta else an.RUTA_DEL_SELLO_DEL_CODIGO
    sello = componer_el_sello()
    if a.escribir:
        escribir(sello, ruta)
        problemas = an.problemas_del_sello_del_codigo(ruta)
        if problemas:                                        # lo recién escrito tiene que pasar la MISMA comprobación que el medidor
            raise SystemExit(f"el sello escrito no pasa su comprobación: {problemas}")
        print(f"escrito {ruta}")
    else:
        print(json.dumps(sello, indent=1, ensure_ascii=False))
        problemas = an.problemas_del_sello_del_codigo(ruta)
        print("el sello de ahora: " + ("CUADRA" if not problemas else " | ".join(problemas)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
