"""150-C2 — cómo se hizo `paquete_antes/` (y `informe_antes.json`). NO lo corre ninguna prueba.

El paquete lo generó el código de ANTES de C2 (núcleo en el commit 85f2191, el HEAD de C1): se extrajo con
`git archive 85f2191 matrixai | tar -x -C <dir>` y se corrió con `PYTHONPATH=<dir>`. Es lo que pide el criterio 4
de C2 («un paquete generado antes con un CSV `;` se verifica sin pasar a FAIL»): la prueba
`tests/test_150_c2_decimal_es.py::test_un_paquete_de_antes_se_verifica_igual` lo verifica con el código de AHORA y
compara etapa por etapa con `informe_antes.json`, que escribió este mismo guion con el código de ANTES.

Lo que salió al medir con el código de antes (`sondas/viejo_entrena.py`): el entrenamiento RECHAZABA un CSV `;` con
decimal coma («DATASET row 2 field peso_kg must be numeric, got '64,3'»), así que NO existe ningún paquete anterior
hecho con un decimal coma convertible. Los que sí existen son los de un CSV `;` cuyas columnas numéricas ya iban con
punto, o cuya columna con coma no era un número (aquí `zona`: «norte, 2»), y de esos es este.

    PYTHONPATH=<arbol_de_antes> python3 generar_paquete_antes.py <carpeta_de_salida>
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from unittest import mock

MXAI = ("PROJECT P\n\nVECTOR V[2]\n  peso_kg: Scalar\n  edad: Scalar\nEND\n\n"
        "NETWORK R\n  INPUT V\n  LAYER Dense units=8 activation=relu\n  LAYER Dense units=2 activation=softmax\n"
        "  OUTPUT riesgo: ProbabilityMap[bajo, alto]\nEND\n\nGRAPH\n  V -> R\nEND\n")
MXTRAIN = ("MODEL model.mxai\n\nDATASET D\n  SOURCE csv(\"d.train.csv\")\n  INPUT V FROM COLUMNS [peso_kg, edad]\n"
           "  TARGET riesgo: Label[bajo, alto]\n  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=8\nEND\n\n"
           "LOSS L\n  TYPE cross_entropy\n  PREDICTION R\n  TARGET riesgo\nEND\n\n"
           "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.05\n  UPDATE R.*\nEND\n\nRUN\n  EPOCHS 4\nEND\n")


def csv_de_excel() -> str:
    """Un `;` de Excel: `peso_kg` con PUNTO (se entrenaba), `zona` con una coma que no es un decimal."""
    rng = random.Random(3)
    filas = ["peso_kg;edad;zona;riesgo"]
    for i in range(40):
        p = round(rng.uniform(50, 110), 1)
        filas.append(f"{p};{rng.randint(20, 80)};\"norte, {i % 3}\";{'alto' if p > 85 else 'bajo'}")
    return "\n".join(filas) + "\n"


def main(salida: Path) -> None:
    import matrixai
    from matrixai import playground as pg
    from matrixai.export.reproduce import añadir_inventario_de_ficheros, write_reproduce_manifest
    from matrixai.export.verify import verify_package
    from matrixai.training.metric_identity import DIRECCION
    print("matrixai importado de:", matrixai.__file__)
    csv = csv_de_excel()
    with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "stdlib"}):
        r = pg._submit_training_job(MXAI, MXTRAIN, csv, None,
                                    field_ranges={"peso_kg": (40.0, 120.0), "edad": (18.0, 90.0)}, seed=7)
        assert r.get("ok"), r
        job = pg._training_jobs[r["job_id"]]
        tope = time.monotonic() + 120
        while job["status"] == "running" and time.monotonic() < tope:
            time.sleep(0.1)
    assert job["status"] == "done", (job["status"], job.get("error"))
    res, cap = job["result"], job["run_provenance"]
    d = salida / "paquete_antes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.mxai").write_text(MXAI, encoding="utf-8")
    (d / "training.mxtrain").write_text(MXTRAIN, encoding="utf-8")
    metricas = []
    for nombre in DIRECCION:
        valor = res.get(nombre)
        if not isinstance(valor, (int, float)) or isinstance(valor, bool):
            continue
        m = {"name": nombre, "value": valor, "dataset_sha256": cap["dataset_sha256_prepared"]}
        if (res.get("metric_splits") or {}).get(nombre):
            m["split"] = res["metric_splits"][nombre]
        metricas.append(m)
    write_reproduce_manifest(d, training_filename="training.mxtrain", dataset_sha256=cap["dataset_sha256_raw"],
                             dataset_rows=cap["dataset_rows"], generation={"seeds": cap["seeds"]}, metrics=metricas,
                             run_provenance=cap, weights_source="trained")
    añadir_inventario_de_ficheros(d)
    informe = verify_package(d, run_training=True)
    etapas = {nombre: etapa.get("status") for nombre, etapa in informe["stages"].items()}
    (salida / "informe_antes.json").write_text(json.dumps(
        {"etapas": etapas, "raw_sha256_del_csv_semicolon": cap["dataset_sha256_raw"],
         "prepared_sha256": cap["dataset_sha256_prepared"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(etapas, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
