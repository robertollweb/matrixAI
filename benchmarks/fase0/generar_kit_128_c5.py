#!/usr/bin/env python3
"""128-C5 — genera los DATOS del kit (`kit_128_c5/datos/`): los 13 conjuntos de referencia del instrumento del 120
(`referencia_120_r0.py`: el MISMO banco, el MISMO conversor ARFF→CSV), comprimidos de forma determinista (`.csv.gz`,
mtime 0) y con un `manifiesto.json` (sha256 del CSV y del .gz, filas, columnas, objetivo, tarea, clase positiva).

SOLO en el servidor (lee los ARFF de `~/fase0_openml_datos/arff`); el PC de Roberto no lo necesita ni lo lleva.
Los 4 ejemplos del control se comprueban BYTE A BYTE contra los del Studio, como hace el instrumento del 120.

    python3 generar_kit_128_c5.py [--destino kit_128_c5/datos]
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent


def _cargar_r0():
    spec = importlib.util.spec_from_file_location("referencia_120_r0", AQUI / "referencia_120_r0.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=str(AQUI / "kit_128_c5" / "datos"))
    a = ap.parse_args(argv)
    r0 = _cargar_r0()
    destino = Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    banco = sorted(json.loads(r0.BORRADOR.read_text())["conjuntos"]["banco"], key=lambda x: x["orden"])
    declarados = {d["nombre"]: d for d in json.loads(r0.SONDA.read_text())["datasets_declarados"]}
    for b in banco:
        d = declarados[b["nombre"]]
        for k in ("data_id", "tarea", "objetivo", "clase_positiva"):
            if str(d[k]) != str(b[k]):
                raise SystemExit(f"PARO: {b['nombre']}.{k}: borrador {b[k]!r} != 119-C5a {d[k]!r}")
        if d.get("sellado"):
            raise SystemExit(f"PARO: {b['nombre']} está sellado")
    gen = r0.cargar_generador()
    r0.comprobar_bytes(gen, [b for b in banco if b["nombre"] in r0.CONTROL])
    filas_manifiesto = []
    for b in banco:
        contenido, columnas, excluidas, n = r0.csv_de(gen, b["data_id"])
        ruta = destino / f"{b['nombre']}.csv.gz"
        # determinista: sin nombre ni fecha dentro del .gz, para que el sha256 del .gz sea estable
        with open(ruta, "wb") as crudo, gzip.GzipFile(filename="", mode="wb", fileobj=crudo, mtime=0,
                                                       compresslevel=9) as z:
            z.write(contenido)
        filas_manifiesto.append({
            "orden": b["orden"], "nombre": b["nombre"], "cubo": b["cubo"], "data_id": b["data_id"],
            "tarea": b["tarea"], "objetivo": b["objetivo"], "clase_positiva": b["clase_positiva"],
            "es_control": b["nombre"] in r0.CONTROL, "fichero": ruta.name,
            "filas": n, "columnas": len(columnas), "excluidas_por_string": excluidas,
            "bytes": len(contenido), "sha256": hashlib.sha256(contenido).hexdigest(),
            "bytes_gz": ruta.stat().st_size, "sha256_gz": hashlib.sha256(ruta.read_bytes()).hexdigest()})
        print(f"{b['nombre']:34} {n:>7} x {len(columnas):<5} {len(contenido):>10} B -> {ruta.stat().st_size:>9} B gz", flush=True)
    manifiesto = {"generado_con": "benchmarks/fase0/generar_kit_128_c5.py",
                  "mismo_conversor_que": "referencia_120_r0.py (csv_de)", "control": sorted(r0.CONTROL),
                  "control_byte_a_byte_con_los_del_studio": True, "conjuntos": filas_manifiesto}
    (destino / "manifiesto.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n")
    print("manifiesto.json escrito;", sum(f["bytes_gz"] for f in filas_manifiesto) / 1e6, "MB comprimidos")


if __name__ == "__main__":
    main()
