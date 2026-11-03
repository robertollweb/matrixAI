# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Cómo se hizo `paquete_antes/` (150-C4): un paquete de reproducción generado con el código de ANTES del corte.

El corte añade `origen_del_fichero` al recibo y al manifiesto. La invariante 3 («un paquete SIN `origen_del_fichero` se lee
y se verifica igual que hoy») no se prueba con un paquete que hace el código nuevo —saldría igual sin querer—, sino con uno
que hizo el código ANTERIOR (HEAD `0f2c052`, 150-C3). Se regenera, solo si hiciera falta, así:

    git -C <núcleo> archive 0f2c052 matrixai | tar -x -C /tmp/antes
    PYTHONPATH=/tmp/antes python3 <núcleo>/tests/fixtures/origen_del_fichero/generar_antes.py <destino> 0f2c052

(`PYTHONPATH` y no `cd`: un guion pone SU carpeta en `sys.path`, no el directorio actual, y la instalación editable de esta
máquina apunta a otro árbol; medido el 03-11 — la primera vez salió con el código del árbol principal. El guion comprueba que
`matrixai` NO lleva el módulo `origen_del_fichero`, pero no de dónde sale: `generado_con` lo apunta.)
Deja `paquete_antes/`, `captura_antes.json` y `esperado_antes.json` (lo que ese código verifica: etapas, código de salida).
"""
import hashlib
import json
import pathlib
import sys

import matrixai
from matrixai.export.verify import verify_package

destino = pathlib.Path(sys.argv[1]).resolve()
assert not (pathlib.Path(matrixai.__file__).parent / "export" / "origen_del_fichero.py").exists(), \
    f"este guion tiene que correr con el código de ANTES, y {matrixai.__file__} ya lleva el módulo del corte"

from matrixai.export.reproduce import _epochs_from_training, añadir_inventario_de_ficheros, write_reproduce_manifest  # noqa: E402
from matrixai.playground import _generate_synthetic_dataset  # noqa: E402
from matrixai.training.dataset_project import generate_project_from_dataset  # noqa: E402

filas = "".join(f"{c},{c + 273.15}\n" for c in range(30))
proj = generate_project_from_dataset(
    "centigrados,prediccionKelvin\n" + filas, "prediccionKelvin",
    column_type_overrides={"centigrados": "number"}, column_range_overrides={"centigrados": (0.0, 29.0)})
csv = _generate_synthetic_dataset(proj["mxai"], proj["training_text"], 30, 7, "coherent",
                                  field_ranges_override=proj.get("field_ranges"))["csv_text"]
sha = hashlib.sha256(csv.encode("utf-8")).hexdigest()
epocas = _epochs_from_training(proj["training_text"])

d = destino / "paquete_antes"
d.mkdir(parents=True)
(d / "model.mxai").write_text(proj["mxai"], encoding="utf-8")
(d / "training.mxtrain").write_text(proj["training_text"], encoding="utf-8")
(d / "recipe.txt").write_text("regenerable\n", encoding="utf-8")
captura = {
    "schema_version": "1.2",
    "mxai_sha256": hashlib.sha256(proj["mxai"].encode()).hexdigest(),
    "mxtrain_sha256": hashlib.sha256(proj["training_text"].encode()).hexdigest(),
    "mxtrain_text": proj["training_text"],
    "recipe_sha256": hashlib.sha256(b"regenerable\n").hexdigest(), "recipe_text": "regenerable\n",
    "seeds": {"dataset": 7}, "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
    "dataset_rows": 30, "dataset_rows_used": 30,
    "epochs_effective": epocas, "epochs_ran": epocas, "warm_start": False, "mode": "coherent",
    "field_ranges": proj.get("field_ranges") or {},
    "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
}
write_reproduce_manifest(d, training_filename="training.mxtrain", recipe_filename="recipe.txt", dataset_sha256=sha,
                         dataset_rows=30, generation={"seeds": {"dataset": 7}}, run_provenance=captura,
                         weights_source="trained")
añadir_inventario_de_ficheros(d)
informe = verify_package(d)
(destino / "captura_antes.json").write_text(json.dumps(captura, indent=2, sort_keys=True), encoding="utf-8")
(destino / "esperado_antes.json").write_text(json.dumps({
    "generado_con": sys.argv[2] if len(sys.argv) > 2 else "?", "dataset_sha256": sha, "exit": informe["exit_code"], "ok": informe["ok"],
    "fully": informe["fully_checked"],
    "etapas": {n: [e["status"], e.get("reason")] for n, e in informe["stages"].items()}}, indent=2, sort_keys=True),
    encoding="utf-8")
print(json.dumps(json.loads((destino / "esperado_antes.json").read_text()), indent=2))
