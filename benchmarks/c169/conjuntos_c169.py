#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169-C1 — los 6 conjuntos de CONFIRMACION_C2.md REGISTRADOS en el catálogo de la Fase 0, sin mover el protocolo ya firmado.

POR QUÉ UN CATÁLOGO APARTE Y NO UNA FILA MÁS EN `protocolo_exploratorio_v2.json`. Ese fichero está FIRMADO: su digest está citado en evidencia
ya commiteada (119-C3, 155, 160, 162, 163) y `exigir_los_protocolos_encadenados` lo encadena. Añadirle seis filas movería el digest de los 23 y de
todo lo ya medido. Aquí las seis filas son `DatasetRegistrado` de la MISMA clase (mismas exigencias: sha256, file_id, cardinalidad medida,
licencia) en `conjuntos_c169.json`, y `protocolo_ampliado()` las junta con el v2 SOLO en memoria, para esta medida. El v2 en disco y su digest no se
tocan (`tests/test_c169_conjuntos.py` lo fija).

DESCARGA. `asegurar_arff` baja el fichero por su URL directa (`file_id`), calcula su sha256 y lo compara con el REGISTRADO: si no cuadra, NO se
escribe nada y se para (falla cerrado). Un fichero ya presente con otro sha256 también para: no se «arregla» ni se pisa.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_FASE0 = _AQUI.parents[1] / "benchmarks" / "fase0"
if str(_FASE0) not in sys.path:
    sys.path.insert(0, str(_FASE0))

import protocolo as protocolo_mod  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402

RUTA_DEL_CATALOGO = _AQUI / "conjuntos_c169.json"
DATA_IDS = (458, 1067, 1464, 1475, 44960, 44962)
NOMBRES = {458: "analcatdata_authorship", 1067: "kc1", 1464: "blood-transfusion-service-center",
           1475: "first-order-theorem-proving", 44960: "energy_efficiency", 44962: "forest_fires"}


def url_de_descarga(file_id: int, nombre: str) -> str:
    return f"https://openml.org/data/v1/download/{int(file_id)}/{nombre}.arff"


def entradas_json(ruta: Path = RUTA_DEL_CATALOGO) -> list[dict]:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))["datasets"]


def conjuntos(ruta: Path = RUTA_DEL_CATALOGO) -> list[protocolo_mod.DatasetRegistrado]:
    """Los 6, como `DatasetRegistrado` (la clase valida sha256, licencia, cardinalidad y file_id al construirlos)."""
    ds = [protocolo_mod.DatasetRegistrado.desde_json(e) for e in entradas_json(ruta)]
    if tuple(d.data_id for d in ds) != DATA_IDS:
        raise SystemExit(f"el catálogo de 169 no trae EXACTAMENTE {DATA_IDS} (trae {[d.data_id for d in ds]})")
    if any(d.sellado for d in ds):
        raise SystemExit("un conjunto del catálogo de 169 está marcado como sellado")
    return ds


def ignorados_registrados() -> dict[int, list[str]]:
    """`data_id -> columnas que OpenML marca `ignore_attribute`` (REGLA.md §1: no entran como predictores)."""
    d = json.loads(RUTA_DEL_CATALOGO.read_text(encoding="utf-8")).get("ignore_attribute", {}).get("por_data_id")
    if not isinstance(d, dict) or {int(k) for k in d} != set(DATA_IDS):
        raise SystemExit("el catálogo de 169 no declara `ignore_attribute` para cada uno de los 6 (aunque sea vacío)")
    return {int(k): list(v) for k, v in d.items()}


def quitar_ignorados(filas: list[dict], objetivo: str, ignorados: list[str]) -> list[str]:
    """Quita de `filas` (en sitio) las columnas `ignorados`; devuelve las que quitó. Si una no está en el fichero o es el objetivo, PARA:
    una exclusión declarada que no se puede aplicar no es una exclusión."""
    quitadas = []
    for col in ignorados:
        if col == objetivo:
            raise SystemExit(f"`{col}` es el objetivo y está en ignore_attribute: no se mide")
        if not filas or col not in filas[0]:
            raise SystemExit(f"`{col}` (ignore_attribute) no está en el fichero: la exclusión declarada no se puede aplicar")
        for f in filas:
            del f[col]
        quitadas.append(col)
    return quitadas


def cargar_arff_sin_ignorados(data_id: int):
    """`c3.cargar_arff` + la exclusión de los `ignore_attribute`; la declara en `c3.LECTURA_DECLARADA[str(data_id)]['columnas_ignoradas_por_openml']`.
    `medir_c169` lo pone en el lugar de `cargar_arff` de `pasada_amplia_101_c5` (que es quien lo llama en `particiones_base`)."""
    filas, objetivo = _cargar_original(data_id)
    quitadas = quitar_ignorados(filas, objetivo, ignorados_registrados()[data_id])
    c3.LECTURA_DECLARADA[str(data_id)]["columnas_ignoradas_por_openml"] = quitadas
    return filas, objetivo


_cargar_original = c3.cargar_arff


def sha256_registrado() -> dict[int, str]:
    return {d.data_id: d.sha256_arff for d in conjuntos()}


def sha256_de(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def asegurar_arff(d: protocolo_mod.DatasetRegistrado, directorio: Path | None = None, *, descargar=None) -> Path:
    """El ARFF de `d` en `directorio/<data_id>.arff`, verificado contra su sha256 registrado. Falla cerrado (SystemExit) si no cuadra."""
    directorio = Path(directorio) if directorio is not None else c3.ARFF_DIR
    destino = directorio / f"{d.data_id}.arff"
    if destino.exists():
        visto = sha256_de(destino)
        if visto != d.sha256_arff:
            raise SystemExit(f"{d.nombre} ({d.data_id}): {destino} tiene sha256 {visto[:16]} y el registrado es {d.sha256_arff[:16]}. "
                             f"No se mide ni se pisa")
        return destino
    if d.file_id is None:
        raise SystemExit(f"{d.nombre}: sin file_id no hay URL directa que verificar")
    url = url_de_descarga(d.file_id, d.nombre)
    if descargar is None:
        def descargar(u):
            req = urllib.request.Request(u, headers={"User-Agent": "matrixai-fase0/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
    crudo = descargar(url)
    visto = hashlib.sha256(crudo).hexdigest()
    if visto != d.sha256_arff:
        raise SystemExit(f"{d.nombre} ({d.data_id}): lo descargado de {url} tiene sha256 {visto[:16]} y el registrado es "
                         f"{d.sha256_arff[:16]}. No se escribe nada")
    directorio.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(f".arff.tmp{os.getpid()}")
    tmp.write_bytes(crudo)
    os.replace(tmp, destino)
    return destino


def asegurar_todos(directorio: Path | None = None, *, descargar=None) -> dict[int, str]:
    return {d.data_id: str(asegurar_arff(d, directorio, descargar=descargar)) for d in conjuntos()}


def protocolo_ampliado(base: protocolo_mod.ProtocoloExploratorio) -> protocolo_mod.ProtocoloExploratorio:
    """`base` (el v2) + los 6, SOLO en memoria. El resto (partición, presupuesto por cubo, regla de cierre, motores) es el de `base`."""
    ya = {d.data_id for d in base.datasets}
    choque = sorted(ya & set(DATA_IDS))
    if choque:
        raise SystemExit(f"los conjuntos {choque} YA están en el protocolo base: no son «nuevos»")
    return dataclasses.replace(base, datasets=tuple(base.datasets) + tuple(conjuntos()))


def registrar_en_el_catalogo_de_lectura() -> None:
    """`cargar_arff` lee el objetivo declarado y las columnas a excluir de `c3._CATALOGO_POR_DATA_ID`: se le añaden los 6 (y se exige que los
    del v2 sigan igual). Hay que llamarlo DESPUÉS de `preparar_protocolo_v2()`, que lo fija al v2."""
    cat = c3._catalogo_registrado()
    if not cat:
        raise SystemExit("el catálogo de lectura está vacío: se llama después de `preparar_protocolo_v2()`")
    for d in conjuntos():
        cat[d.data_id] = d.a_json()
