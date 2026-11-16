# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C5 — prepara el kit: elige los conjuntos por REGLA, copia el código necesario y empaqueta los datos.

Corre AQUÍ (en el servidor, no en el portátil), con el árbol del núcleo y de motores a mano:

    python3 empaquetar_kit.py            # escribe codigo/, datos/ y conjuntos.json junto a este fichero

QUÉ ELIGE. Los conjuntos NO sellados del protocolo v2 que caben en el techo del paquete GPU (contrato 116,
§ «MEDIDO EN GPU», 12-11): **≤ 16.000 filas de contexto y ≤ 512.000 celdas (filas × columnas) en TODOS los pliegues**,
calculado con la partición REAL (la misma función que 116-C2: `folds_del_dataset`), no con una estimación. Los
sellados no entran nunca. Cada conjunto que se queda fuera lleva su motivo en `conjuntos.json`.

QUÉ COPIA.
- `codigo/matrixAI/matrixai/` y `codigo/matrixai-engines/src/matrixai_engines/` ENTEROS (16 MB), y de
  `benchmarks/fase0/` solo los módulos que importa la pasada de 116-C2 + el protocolo v2: la MISMA disposición de
  directorios que el árbol, porque esos módulos buscan a sus hermanos por `parents[2]`.
- `datos/<data_id>.arff.gz`: el ARFF original de cada conjunto elegido, con su sha256 en `datos/manifiesto.json`
  (el del ARFF sin comprimir, que es el que registra `~/fase0_openml_datos/hashes.json`).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ_NUCLEO = Path("/home/deployer/matrixAI")
RAIZ_MOTORES = Path("/home/deployer/matrixai-engines")
FASE0 = RAIZ_NUCLEO / "benchmarks" / "fase0"
ARFF_DIR = Path.home() / "fase0_openml_datos" / "arff"
HASHES = Path.home() / "fase0_openml_datos" / "hashes.json"

#: El techo del paquete GPU (contrato 116, «MEDIDO EN GPU» y «DECIDIDO … 12-11»; `techo_de_tabicl` de motores/tabicl.py
#: para 8 GB). Escritos aquí para que `conjuntos.json` los declare; `medir_acierto.py` los vuelve a comprobar al arrancar.
MAX_FILAS = 16_000
MAX_CELDAS = 512_000

#: Módulos de benchmarks/fase0 que hacen falta (los que importa 116-C2 y sus dependencias) + el protocolo v2.
FICHEROS_FASE0 = ("pasada_116c2_tabicl.py", "pasada_amplia_101_c5.py", "pasada_exploratoria_101_c3.py",
                  "pasada_v2_113.py", "protocolo.py", "lector_arff.py", "protocolo_exploratorio_v2.json")


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def copiar_codigo() -> None:
    destino = AQUI / "codigo"
    if destino.exists():
        shutil.rmtree(destino)
    ignorar = shutil.ignore_patterns("__pycache__", "*.pyc", "tests")
    shutil.copytree(RAIZ_NUCLEO / "matrixai", destino / "matrixAI" / "matrixai", ignore=ignorar)
    shutil.copytree(RAIZ_MOTORES / "src" / "matrixai_engines",
                    destino / "matrixai-engines" / "src" / "matrixai_engines", ignore=ignorar)
    (destino / "matrixAI" / "benchmarks" / "fase0").mkdir(parents=True)
    for nombre in FICHEROS_FASE0:
        shutil.copy2(FASE0 / nombre, destino / "matrixAI" / "benchmarks" / "fase0" / nombre)
    # los ficheros de código que el digest de caché de C3 lee por su ruta del árbol: ya están dentro (matrixai/,
    # matrixai_engines/ y benchmarks/fase0/lector_arff.py).


def elegir() -> dict:
    sys.path.insert(0, str(FASE0))
    sys.path.insert(0, str(RAIZ_NUCLEO))
    sys.path.insert(0, str(RAIZ_MOTORES / "src"))
    import pasada_116c2_tabicl as c2  # noqa: PLC0415
    import pasada_exploratoria_101_c3 as c3  # noqa: PLC0415

    c2.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = c2.c5.datasets_de_la_pasada(protocolo)
    sellados = [d.nombre for d in todos if d.sellado]
    no_sellados = [d for d in todos if not d.sellado]
    meta = c2.cargar_particion_por_dataset_v2()
    incluidos, excluidos = [], []
    for ds in no_sellados:
        m = meta[ds.nombre]
        dev = m["n_filas_con_objetivo"] - m["n_test"]
        k = m["folds_obtenidos"] or 1
        ctx_estimado = -(-dev * (k - 1) // k) if k > 1 else dev
        celdas_estimadas = ctx_estimado * m["n_predictores"]
        # Filtro barato con holgura (no se lee ni particiona un ARFF de 188.000 filas para decir que no cabe):
        if ctx_estimado > MAX_FILAS * 1.05 or celdas_estimadas > MAX_CELDAS * 1.10:
            excluidos.append({
                "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "motivo": "no_cabe_en_el_techo_gpu",
                "filas_de_contexto_estimadas": ctx_estimado, "n_columnas": m["n_predictores"],
                "celdas_estimadas": celdas_estimadas, "exacto": False,
                "detalle": (f"≈{ctx_estimado} filas de contexto y {m['n_predictores']} columnas = "
                            f"≈{celdas_estimadas} celdas; techo {MAX_FILAS} filas y {MAX_CELDAS} celdas "
                            f"(estimado con la partición que midió la v2, no leyendo el ARFF)")})
            continue
        regimen, _base = c2.regimen_del_dataset(ds, protocolo)  # EXACTO, pliegue a pliegue
        folds, n_cols, n_test, _ = c2.folds_del_dataset(ds, protocolo)
        max_ctx = max(f[2] for f in folds)
        celdas = max_ctx * n_cols
        ficha = {"dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea,
                 "n_pliegues": len(folds), "n_columnas": n_cols, "n_test": n_test,
                 "max_filas_de_contexto": max_ctx, "max_celdas": celdas, "exacto": True,
                 "metrica_de_cierre": c2.c5.metrica_de_cierre_por_dataset(protocolo, [ds])[ds.nombre]}
        if max_ctx <= MAX_FILAS and celdas <= MAX_CELDAS:
            incluidos.append(ficha)
        else:
            ficha.update(motivo="no_cabe_en_el_techo_gpu",
                         detalle=f"{max_ctx} filas de contexto y {n_cols} columnas = {celdas} celdas; "
                                 f"techo {MAX_FILAS} filas y {MAX_CELDAS} celdas")
            excluidos.append(ficha)
    return {"protocolo_version": protocolo.version_protocolo, "protocolo_digest_sha256": protocolo.digest(),
            "techo_gpu": {"max_filas_de_contexto": MAX_FILAS, "max_celdas": MAX_CELDAS,
                          "origen": "contrato 116, MEDIDO EN GPU (12-11): RTX 2000 Ada 8 GB, 512.000 celdas / 16.000 filas"},
            "incluidos": incluidos, "excluidos_por_no_caber": excluidos,
            "sellados_no_entran": sorted(sellados)}


def empaquetar_datos(incluidos: list[dict]) -> None:
    destino = AQUI / "datos"
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir()
    registrados = {h["data_id"]: h for h in json.loads(HASHES.read_text(encoding="utf-8"))}
    manifiesto = {}
    for ficha in incluidos:
        origen = ARFF_DIR / f"{ficha['data_id']}.arff"
        sha = _sha256(origen)
        oficial = registrados.get(ficha["data_id"], {}).get("sha256")
        if oficial is not None and oficial != sha:
            raise SystemExit(f"{origen}: sha256 {sha} no coincide con hashes.json ({oficial})")
        with origen.open("rb") as f, gzip.GzipFile(destino / f"{ficha['data_id']}.arff.gz", "wb", mtime=0) as g:
            shutil.copyfileobj(f, g)
        manifiesto[str(ficha["data_id"])] = {"dataset": ficha["dataset"], "sha256_del_arff": sha,
                                             "bytes_del_arff": origen.stat().st_size,
                                             "sha256_de_hashes_json": oficial}
    (destino / "manifiesto.json").write_text(json.dumps(manifiesto, indent=1, sort_keys=True), encoding="utf-8")


def main() -> None:
    seleccion = elegir()
    # el orden de medida: de lo barato a lo caro (celdas), para que un corte deje lo más posible hecho
    seleccion["incluidos"].sort(key=lambda f: (f["max_celdas"], f["dataset"]))
    (AQUI / "conjuntos.json").write_text(json.dumps(seleccion, indent=1, ensure_ascii=False), encoding="utf-8")
    copiar_codigo()
    empaquetar_datos(seleccion["incluidos"])
    print(f"incluidos: {len(seleccion['incluidos'])}; fuera por no caber: "
          f"{len(seleccion['excluidos_por_no_caber'])}; sellados: {len(seleccion['sellados_no_entran'])}")


if __name__ == "__main__":
    main()
