#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169-C1 — GENERA `conjuntos_c169.json` (los 6 de CONFIRMACION_C2.md como `DatasetRegistrado`). Solo metadatos de OpenML y el ARFF; ningún modelo.

El sha256 sale de la tabla de `benchmarks/c163/CONFIRMACION_C2.md` (apartada ANTES de medir): se descarga, se compara y, si no cuadra, no se
escribe nada. Los campos medidos (cardinalidad, desbalance) salen del propio ARFF con las mismas funciones que la v1/v2.

    python3 benchmarks/c169/registrar_conjuntos_c169.py --escribir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parents[1]
for p in (_AQUI, _RAIZ):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import conjuntos_c169 as cj  # noqa: E402
from benchmarks.fase0 import generar_protocolo as g1  # noqa: E402
from benchmarks.fase0 import generar_protocolo_v2 as g2  # noqa: E402
from benchmarks.fase0.file_id_para_la_re_firma import file_id_desde_url  # noqa: E402
from benchmarks.fase0.protocolo import DatasetRegistrado, UMBRAL_ALTA_CARDINALIDAD  # noqa: E402

RUTA_DE_LA_TABLA = _AQUI.parent / "c163" / "CONFIRMACION_C2.md"


def sha256_de_la_tabla(ruta: Path = RUTA_DE_LA_TABLA) -> dict[int, str]:
    out = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if len(celdas) == 9 and celdas[1].isdigit() and re.fullmatch(r"[0-9a-f]{64}", celdas[8]):
            out[int(celdas[1])] = celdas[8]
    return out


def construir(directorio: Path | None = None) -> list[dict]:
    tabla = sha256_de_la_tabla()
    if set(tabla) != set(cj.DATA_IDS):
        raise SystemExit(f"la tabla trae {sorted(tabla)}, no {cj.DATA_IDS}")
    # La regla de selección del 163-C2 admite `CC BY 4.0`/`CC BY`/`CC0` además de `Public`; el generador de la v1 solo `Public` y alguna más.
    g1.LICENCIAS_PERMITIDAS = frozenset(set(g1.LICENCIAS_PERMITIDAS) | {"public", "cc by 4.0", "cc by", "cc0", "cc 0: public domain"})
    filas = []
    for i in cj.DATA_IDS:
        es_regresion = i in (44960, 44962)
        c = g1._candidato(i, es_clasificacion=not es_regresion)
        if c is None or not c["objetivo"]:
            raise SystemExit(f"{i} ya no pasa los filtros del generador (activo/licencia/tamaño)")
        if c["nombre"] != cj.NOMBRES[i]:
            raise SystemExit(f"{i}: se esperaba {cj.NOMBRES[i]!r} y OpenML da {c['nombre']!r}")
        tarea = "regression" if es_regresion else ("binary_classification" if c["binaria"] else "multiclass_classification")
        destino = (Path(directorio) if directorio else Path.home() / "fase0_openml_datos" / "arff") / f"{i}.arff"
        sha = tabla[i]
        # Descarga verificada contra el sha256 de la TABLA (la que se apartó antes de medir); md5 contra OpenML después.
        provisional = DatasetRegistrado(
            data_id=i, nombre=c["nombre"], fuente="openml-cc18" if i < 44000 else "openml-ctr23", version=int(c["version"]),
            sha256_arff=sha, columna_objetivo=c["objetivo"], tarea=tarea, cubo_de_tamano=c["bucket"], n_filas=c["n_filas"],
            n_columnas=c["n_columnas"], tiene_faltantes=c["tiene_faltantes"], max_cardinalidad_nominal=0, alta_cardinalidad=False,
            desbalanceado=False, solo_numericas=c["solo_numericas"], licencia=c["licencia"] or "Public", sellado=False,
            file_id=file_id_desde_url(c["url"]))
        ruta = cj.asegurar_arff(provisional, destino.parent)
        import hashlib
        info = g1._info(i)
        if hashlib.md5(ruta.read_bytes()).hexdigest() != info.get("md5_checksum"):
            raise SystemExit(f"{i}: el md5 del fichero no es el que declara OpenML")
        niveles = g1.cardinalidad_nominal_declarada(ruta, c["objetivo"])
        maxc = max(niveles.values(), default=0)
        if tarea == "regression":
            desbalanceado = False
        elif tarea == "binary_classification":
            desbalanceado = c["desbalanceado"]
        else:
            cuota, k = g2._cuota_minoritaria_y_k(ruta, c["objetivo"])
            desbalanceado = bool(cuota * k <= g2.UMBRAL_P1_DESBALANCE)
        d = DatasetRegistrado(
            data_id=i, nombre=c["nombre"], fuente=provisional.fuente, version=provisional.version, sha256_arff=sha,
            columna_objetivo=c["objetivo"], tarea=tarea, cubo_de_tamano=c["bucket"], n_filas=c["n_filas"], n_columnas=c["n_columnas"],
            tiene_faltantes=c["tiene_faltantes"], max_cardinalidad_nominal=maxc, alta_cardinalidad=maxc >= UMBRAL_ALTA_CARDINALIDAD,
            desbalanceado=desbalanceado, solo_numericas=c["solo_numericas"], licencia=provisional.licencia, sellado=False,
            file_id=provisional.file_id)
        filas.append(d.a_json())
    return filas


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--escribir", action="store_true")
    a = ap.parse_args(argv)
    filas = construir()
    texto = json.dumps({"que_es": "los 6 conjuntos de benchmarks/c163/CONFIRMACION_C2.md como DatasetRegistrado (169-C1)", "datasets": filas},
                       indent=1, ensure_ascii=False) + "\n"
    if a.escribir:
        cj.RUTA_DEL_CATALOGO.write_text(texto, encoding="utf-8")
        print(f"escrito {cj.RUTA_DEL_CATALOGO}")
    else:
        print(texto)


if __name__ == "__main__":
    main()
