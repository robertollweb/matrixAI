#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Calibración previa a 101-C3, pedida por Roberto (2026-09-07): medir el
coste REAL por intento (dataset, motor) con datos reales antes de comprometerse
a la pasada exploratoria completa (8-12 datasets, 4 motores, 5 pliegues x 3
repeticiones). NO es el corte 101-C3 — es la medición que decide con qué
`minutos_por_cubo`/`wall_seconds` lanzarlo, y si el aislamiento por subproceso
(`matrixai_engines.subproceso`, HECHO el 2026-09-07) se comporta con datos
reales tan bien como con las fixtures sintéticas de su propia suite.

Tres datasets pequeños, sin sellar, sin faltantes, binarios (kc2=1063,
climate-model-simulation-crashes=40994, diabetes=37 — ver
`~/fase0_openml_datos/seleccion_40_final.json`), un único split
train/validation/test por dataset (vía `particion_por_diseno`, 103-C4, no un
protocolo de 5x3 pliegues completo: esto mide coste por intento, no reproduce
la pasada entera), los 4 motores de `matrixai_engines`, CADA intento aislado
en su propio proceso del SO vía `ejecutar_intento_aislado` (subproceso.py).
`procesos_en_paralelo=1` (secuencial) y techo de 2h de pared para la pasada
entera, ambos por decisión explícita de Roberto sobre los techos de 101-C3.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
_RAIZ_DE_ENGINES = _RAIZ_DEL_CORE.parent / "matrixai-engines" / "src"
for ruta in (_RAIZ_DEL_CORE, _RAIZ_DE_ENGINES):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from matrixai.estudio import ProblemSpec  # noqa: E402
from matrixai.training.particion_por_diseno import proponer_particion  # noqa: E402

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.motores.baseline import MotorBaseline  # noqa: E402
from matrixai_engines.motores.lineal import MotorLineal  # noqa: E402
from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM  # noqa: E402
from matrixai_engines.motores.densa import MotorDensaPropia  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

ARFF_DIR = Path.home() / "fase0_openml_datos" / "arff"

# (data_id, nombre, positivo, negativo) -- los tres, "pequeno", sin sellar,
# sin faltantes, binarios; ver seleccion_40_final.json.
DATASETS = [
    (1063, "kc2", "yes", "no"),
    (40994, "climate-model-simulation-crashes", "1", "0"),
    (37, "diabetes", "tested_positive", "tested_negative"),
]

WALL_SECONDS = 300.0  # generoso: lo medido en fixtures sinteticas fue <5s por motor


# LA TERCERA COPIA DEL LECTOR, BORRADA (2026-09-13).
#
# Este fichero tenía la suya, igual que la tenía la pasada, y las tres
# divergieron — que es lo que pasa siempre que dos sitios declaran lo mismo.
# La de aquí arrastraba los tres defectos de una vez:
#
#   · `objetivo = nombres[-1]`, cuando en CINCO de los cuarenta del protocolo
#     el objetivo declarado NO es el último atributo: habría medido tiempos
#     entrenando contra otra columna;
#   · `float(valor)` para todo lo que no es el objetivo, que revienta con
#     cualquier columna nominal y por eso este fichero solo podía medir tres
#     datasets numéricos;
#   · ni el `"?"` del estándar ARFF como faltante, ni la exclusión de la
#     columna identificadora que el catálogo dice que sobra.
#
# Ahora delega en el `cargar_arff` de la pasada, que a su vez delega en el
# lector único. Se importa de allí y no se reimplementa aquí: una cuarta copia
# «porque esto solo mide tiempos» es exactamente cómo nacieron las tres.
from pasada_exploratoria_101_c3 import cargar_arff  # noqa: E402


def particiones_para(data_id: int, nombre_ds: str, positivo: str, negativo: str):
    filas, objetivo = cargar_arff(data_id)
    predictores = tuple(k for k in filas[0] if k not in ("row_id", objetivo))

    propuesta = proponer_particion(
        filas, plan_id=f"calibracion-101c3-{nombre_ds}", observation_id_field="row_id",
        split_type="iid", seed=0, test_fraction=0.2, folds=5, repeats=1, objetivo=objetivo)
    if not propuesta.es_viable:
        raise RuntimeError(f"{nombre_ds}: particion no viable, bloqueos={propuesta.bloqueos}")

    por_id = {f["row_id"]: f for f in filas}
    pliegue0 = propuesta.pliegues.pliegue_de(repeticion=0, pliegue=0)
    test_ids = propuesta.plan.observaciones_del_rol("test")

    train = Particion.desde_filas([por_id[i] for i in pliegue0.entrena],
                                  row_id_field="row_id", target_field=objetivo)
    validation = Particion.desde_filas([por_id[i] for i in pliegue0.valida],
                                       row_id_field="row_id", target_field=objetivo)
    test = Particion.desde_filas([por_id[i] for i in test_ids],
                                 row_id_field="row_id", target_field=objetivo)

    spec = ProblemSpec(problem_id=f"calibracion-101c3-{nombre_ds}", target=objetivo,
                       task="binary_classification", observation_unit="fila",
                       classes=(negativo, positivo), positive_label=positivo,
                       predictors=predictores)
    return train, validation, test, spec, propuesta.plan.digest(), len(filas)


def main() -> None:
    motores = [MotorBaseline(), MotorLineal(), MotorArbolLightGBM(), MotorDensaPropia()]
    resultados = []
    inicio_total = time.perf_counter()

    for data_id, nombre_ds, positivo, negativo in DATASETS:
        train, validation, test, spec, split_digest, n_filas = particiones_para(
            data_id, nombre_ds, positivo, negativo)
        print(f"\n=== {nombre_ds} (data_id={data_id}, n={n_filas}, "
             f"train={len(train.row_ids)}, val={len(validation.row_ids)}, "
             f"test={len(test.row_ids)}) ===", flush=True)

        for motor in motores:
            presupuesto = Presupuesto(wall_seconds=WALL_SECONDS, hilos=4, seed=0)
            t0 = time.perf_counter()
            intento = ejecutar_intento_aislado(
                motor, train, validation, test, spec, presupuesto,
                candidate=f"{motor.nombre}-default", split_plan_digest=split_digest,
                dataset=nombre_ds, pliegue=0, repeticion=0)
            transcurrido = time.perf_counter() - t0

            auroc = accuracy = None
            if intento.informe is not None:
                for m in intento.informe.get("metrics", []):
                    if m["metric_id"] == "auroc":
                        auroc = m["value"]
                    elif m["metric_id"] == "accuracy":
                        accuracy = m["value"]

            registro = {
                "dataset": nombre_ds, "motor": motor.nombre, "estado": intento.estado,
                "wall_s": round(transcurrido, 3), "auroc": auroc, "accuracy": accuracy,
                "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
            }
            resultados.append(registro)
            print(f"  {motor.nombre:26s} estado={intento.estado:10s} "
                 f"wall={transcurrido:6.2f}s auroc={auroc} accuracy={accuracy}", flush=True)

    total = time.perf_counter() - inicio_total
    print(f"\n=== total: {total:.1f}s ({total/60:.1f} min) ===")

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds_por_intento": WALL_SECONDS,
        "procesos_en_paralelo": 1,
        "total_wall_s": round(total, 1),
        "resultados": resultados,
    }
    ruta_salida = Path(__file__).resolve().parent / "calibracion_101_c3_resultado.json"
    ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {ruta_salida}")


if __name__ == "__main__":
    main()
