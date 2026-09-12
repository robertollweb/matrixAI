#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3 — la pasada exploratoria real: «ejecutar el protocolo corto y
guardar resultados crudos bajo su hash, con costes reales y comparación de
los candidatos» (texto literal del criterio de terminado).

12 datasets reales de OpenML, no sellados, clasificación binaria, del
protocolo ya registrado (`protocolo_exploratorio.json`, 101-C1) — cubren
numéricas puras, categóricas de alta cardinalidad, faltantes y
desbalanceo, que es literalmente lo que el contrato 101 pide para la
exploración («8-12 datasets externos con casos numéricos/categóricos,
faltantes y desbalanceo»). Ninguno de los 8 sellados se toca.

DISEÑO DE PARTICIÓN: el YA REGISTRADO en el protocolo (`DisenoDeParticion`,
101-C1) — 5 pliegues x 3 repeticiones para pequeño/mediano — no uno
inventado aquí. Cambiar folds/repeticiones sin registrar un protocolo
nuevo sería precisamente lo que `protocolo.py` prohíbe por diseño.

UNA SOLA CONFIGURACIÓN POR MOTOR, DECLARADO. El protocolo registrado habla
de `Motor.configuraciones=2` («el rival en su mejor versión») como
abstracción de PLANIFICACIÓN para `calcular_coste` — pero los 4 adaptadores
reales de `matrixai_engines` (102-C2) son de hiperparámetros FIJOS, sin
búsqueda interna todavía. Ejecutar "2 configuraciones" hoy sería fabricar
una segunda pasada que no existe. Se declara aquí, no se esconde: 1
configuración por motor es el alcance real de esta pasada.

MINORÍA = POSITIVA, CONVENCIÓN EXPLÍCITA. Para AUROC hace falta una clase
"positiva" declarada; sin leer la semántica de cada dataset uno por uno,
la convención aplicada aquí es que la clase MINORITARIA es la positiva
(el caso de interés en un problema desbalanceado) — documentado por
dataset abajo, no adivinado fila a fila.

PREPARACIÓN POR MOTOR, NO UNA SOLA VEZ POR DATASET. `ajustar_preparacion`
(103-C3) recibe `admite_categoricas`/`admite_faltantes` de la
`Capacidades` de CADA motor — un motor que no soporta nativamente
categóricas/faltantes recibe datos imputados/codificados; uno que sí,
recibe los valores crudos (`None` en faltantes, para que su tratamiento
nativo los vea). Se ajusta sobre train de cada pliegue, nunca sobre
validation/test.

CACHÉ POR INTENTO, PARA NO RELANZAR LOS 80 MINUTOS ENTEROS POR UN
CAMBIO PEQUEÑO (pedido explícito de Roberto, 2026-09-07: «el tiempo es
oro»). Cada intento se reusa del resultado anterior si su digest de
código no cambió: un digest de los ficheros COMPARTIDOS (harness,
subproceso, particiones, preparación, este mismo script — cambiar
cualquiera invalida TODO) y un digest del fichero del MOTOR concreto
(cambiar `densa.py` invalida solo sus intentos, no los de los otros
tres). Determinismo verificado antes de confiar en él (mismo motor +
misma semilla + mismos datos + mismo código -> mismo resultado, ya
medido en `test_determinismo_misma_semilla_mismo_digest` de 102-C2).
`--forzar` en la línea de comandos ignora el caché entero, para la
validación final antes de cerrar el corte.
"""
from __future__ import annotations

import argparse
import hashlib
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
from matrixai.estudio.validacion import digest_canonico  # noqa: E402
from matrixai.training.particion_por_diseno import proponer_particion  # noqa: E402
from matrixai.training.preparacion import (ajustar_preparacion,  # noqa: E402
                                           tipar_columnas_numericas, transformar_fila)

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.motores.baseline import MotorBaseline  # noqa: E402
from matrixai_engines.motores.lineal import MotorLineal  # noqa: E402
from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM  # noqa: E402
from matrixai_engines.motores.densa import MotorDensaPropia  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

ARFF_DIR = Path.home() / "fase0_openml_datos" / "arff"

# (data_id, nombre, cubo, positiva, negativa) -- los 12, no sellados,
# binarios, pequeño/mediano, del protocolo registrado. Minoría = positiva.
DATASETS = [
    (1063, "kc2", "pequeno", "yes", "no"),
    (40994, "climate-model-simulation-crashes", "pequeno", "0", "1"),
    (37, "diabetes", "pequeno", "tested_positive", "tested_negative"),
    (15, "breast-w", "pequeno", "malignant", "benign"),
    (1049, "pc4", "pequeno", "TRUE", "FALSE"),
    (1068, "pc1", "pequeno", "true", "false"),
    (38, "sick", "mediano", "sick", "negative"),
    (1053, "jm1", "mediano", "true", "false"),
    (1487, "ozone-level-8hr", "mediano", "2", "1"),
    (4534, "PhishingWebsites", "mediano", "-1", "1"),
    (40978, "Internet-Advertisements", "mediano", "ad", "noad"),
    (40983, "wilt", "mediano", "2", "1"),
]

# Diseño de partición YA REGISTRADO en protocolo_exploratorio.json (101-C1),
# no inventado aquí.
FOLDS = 5
REPETICIONES_PEQUENO_MEDIANO = 3
SEMILLAS = (0, 1, 2)

WALL_SECONDS = 120.0  # generoso frente a lo medido (baseline<1s, lgbm~2s, densa~6s)

_DIR_ENGINES = _RAIZ_DE_ENGINES / "matrixai_engines"
# Compartidos: un cambio en cualquiera de estos invalida TODO el caché,
# porque afecta a los 4 motores por igual (o a la propia lógica de esta
# pasada).
_FICHEROS_COMPARTIDOS = (
    _DIR_ENGINES / "harness.py", _DIR_ENGINES / "subproceso.py",
    _DIR_ENGINES / "particiones.py", _DIR_ENGINES / "motor.py",
    _DIR_ENGINES / "capacidades.py", _DIR_ENGINES / "errores.py",
    _DIR_ENGINES / "textos.py",
    _RAIZ_DEL_CORE / "matrixai" / "training" / "particion_por_diseno.py",
    _RAIZ_DEL_CORE / "matrixai" / "training" / "preparacion.py",
    Path(__file__),
)
# Por motor: un cambio SOLO invalida los intentos de ESE motor.
_FICHERO_POR_MOTOR = {
    "baseline": _DIR_ENGINES / "motores" / "baseline.py",
    "sklearn.lineal": _DIR_ENGINES / "motores" / "lineal.py",
    "lightgbm": _DIR_ENGINES / "motores" / "arbol_lightgbm.py",
    "matrixai.dense.torch_cpu": _DIR_ENGINES / "motores" / "densa.py",
}


def _digest_fichero(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def _digest_entorno() -> str:
    return hashlib.sha256("".join(_digest_fichero(f) for f in _FICHEROS_COMPARTIDOS)
                         .encode()).hexdigest()[:16]


def _cargar_cache(ruta: Path) -> dict[tuple, dict]:
    if not ruta.exists():
        return {}
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    return {(r["dataset"], r["motor"], r["repeticion"], r["pliegue"]): r
           for r in payload.get("resultados", [])}


def cargar_arff(data_id: int) -> tuple[list[dict], str]:
    from scipy.io import arff

    datos, meta = arff.loadarff(ARFF_DIR / f"{data_id}.arff")
    nombres = meta.names()
    tipos = dict(zip(nombres, meta.types()))
    objetivo = nombres[-1]
    filas = []
    for i, registro in enumerate(datos):
        fila = {"row_id": f"{data_id}-{i}"}
        for nombre in nombres:
            valor = registro[nombre]
            if tipos[nombre] == "numeric":
                valor_final = None if valor != valor else float(valor)  # NaN -> None
            else:
                texto = valor.decode() if isinstance(valor, (bytes, bytearray)) else str(valor)
                valor_final = None if texto == "" else texto
            fila[nombre] = valor_final
        filas.append(fila)
    return filas, objetivo


def preparar_para_motor(train_filas: list[dict], todas_filas: list[dict], objetivo: str,
                        predictores: tuple[str, ...], motor) -> list[dict]:
    """`ajustar_preparacion` SOLO sobre train de este pliegue; `transformar_
    fila` sobre TODAS (train/validation/test) con esa misma política."""
    capacidades = motor.capabilities()
    politica = ajustar_preparacion(train_filas, objetivo=objetivo, columnas=predictores,
                                   admite_categoricas=capacidades.admite_categoricas,
                                   admite_faltantes=capacidades.admite_faltantes)
    resultado = []
    for fila in todas_filas:
        transformada = transformar_fila(fila, politica)
        transformada["row_id"] = fila["row_id"]
        transformada[objetivo] = fila[objetivo]
        resultado.append(transformada)
    return resultado


def particiones_base(data_id: int, nombre_ds: str, cubo: str, positiva: str, negativa: str):
    filas, objetivo = cargar_arff(data_id)
    filas_con_objetivo = [f for f in filas if f[objetivo] is not None]
    predictores = tuple(k for k in filas[0] if k not in ("row_id", objetivo))

    # SIN ESTO, 6 intentos perdidos en `Internet-Advertisements` — medido en la
    # pasada del 2026-09-07 y diagnosticado el 09-12. Un ARFF entrega como
    # TEXTO sus columnas nominales, aunque sus valores sean `"0"`/`"1"`.
    # `ajustar_preparacion` no parsea texto (por diseño del núcleo: que una
    # columna «parezca» numérica no es serlo), así que las ve categóricas y
    # mete el centinela `__desconocida__` cuando una categoría no aparecía en
    # train. Los motores SÍ parsean, ven la columna numérica, y
    # `float("__desconocida__")` revienta dentro del subproceso aislado.
    #
    # El dataset entero se perdía para la regla de cierre, que cuenta un fallo
    # como dataset perdido — y el veredicto «lightgbm 9/12 = 0,750 < 0,800» se
    # apoyaba en parte en eso, con una distancia REAL de 1,15 puntos, dentro
    # del margen de 2. El Studio ya tenía esta reparación desde el 09; este
    # camino nunca la recibió, y por eso la función vive ahora en el núcleo en
    # vez de en una tercera copia.
    tipar_columnas_numericas(filas_con_objetivo, predictores)

    propuesta = proponer_particion(
        filas_con_objetivo, plan_id=f"101c3-{nombre_ds}", observation_id_field="row_id",
        split_type="iid", seed=0, test_fraction=0.2, folds=FOLDS,
        repeats=REPETICIONES_PEQUENO_MEDIANO, objetivo=objetivo)
    if not propuesta.es_viable:
        raise RuntimeError(f"{nombre_ds}: particion no viable, bloqueos={propuesta.bloqueos}")

    por_id = {f["row_id"]: f for f in filas_con_objetivo}
    spec = ProblemSpec(problem_id=f"101c3-{nombre_ds}", target=objetivo,
                       task="binary_classification", observation_unit="fila",
                       classes=(negativa, positiva), positive_label=positiva,
                       predictors=predictores)
    return por_id, propuesta, spec, objetivo, predictores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forzar", action="store_true",
                       help="ignora el caché entero, re-ejecuta los 720 intentos")
    args = parser.parse_args()

    ruta_salida = Path(__file__).resolve().parent / "pasada_exploratoria_101_c3_resultado.json"
    cache_previo = {} if args.forzar else _cargar_cache(ruta_salida)
    entorno_digest = _digest_entorno()
    digest_por_motor = {nombre: _digest_fichero(ruta) for nombre, ruta in _FICHERO_POR_MOTOR.items()}

    motores = [MotorBaseline(), MotorLineal(), MotorArbolLightGBM(), MotorDensaPropia()]
    resultados = []
    reusados = 0
    inicio_total = time.perf_counter()

    for data_id, nombre_ds, cubo, positiva, negativa in DATASETS:
        por_id, propuesta, spec, objetivo, predictores = particiones_base(
            data_id, nombre_ds, cubo, positiva, negativa)
        test_ids = propuesta.plan.observaciones_del_rol("test")
        n_total = len(por_id)
        print(f"\n=== {nombre_ds} (data_id={data_id}, n={n_total}, "
             f"test={len(test_ids)}) ===", flush=True)

        for repeticion in range(REPETICIONES_PEQUENO_MEDIANO):
            for pliegue_i in range(FOLDS):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None:
                    continue
                for motor in motores:
                    clave = (nombre_ds, motor.nombre, repeticion, pliegue_i)
                    previo = cache_previo.get(clave)
                    if (previo is not None and previo.get("entorno_digest") == entorno_digest
                           and previo.get("motor_digest") == digest_por_motor[motor.nombre]):
                        registro = dict(previo, reusado=True)
                        resultados.append(registro)
                        reusados += 1
                        continue

                    filas_train_crudas = [por_id[i] for i in pliegue.entrena]
                    filas_val_crudas = [por_id[i] for i in pliegue.valida]
                    filas_test_crudas = [por_id[i] for i in test_ids]

                    transformadas = preparar_para_motor(
                        filas_train_crudas, filas_train_crudas + filas_val_crudas + filas_test_crudas,
                        objetivo, predictores, motor)
                    n_train, n_val = len(filas_train_crudas), len(filas_val_crudas)
                    filas_train_t = transformadas[:n_train]
                    filas_val_t = transformadas[n_train:n_train + n_val]
                    filas_test_t = transformadas[n_train + n_val:]

                    train = Particion.desde_filas(filas_train_t, row_id_field="row_id",
                                                  target_field=objetivo)
                    validation = Particion.desde_filas(filas_val_t, row_id_field="row_id",
                                                        target_field=objetivo)
                    test = Particion.desde_filas(filas_test_t, row_id_field="row_id",
                                                 target_field=objetivo)

                    presupuesto = Presupuesto(wall_seconds=WALL_SECONDS, hilos=4, seed=SEMILLAS[repeticion])
                    t0 = time.perf_counter()
                    intento = ejecutar_intento_aislado(
                        motor, train, validation, test, spec, presupuesto,
                        candidate=f"{motor.nombre}-default", split_plan_digest=propuesta.plan.digest(),
                        dataset=nombre_ds, pliegue=pliegue_i, repeticion=repeticion)
                    transcurrido = time.perf_counter() - t0

                    auroc = accuracy = None
                    if intento.informe is not None:
                        for m in intento.informe.get("metrics", []):
                            if m["metric_id"] == "auroc":
                                auroc = m["value"]
                            elif m["metric_id"] == "accuracy":
                                accuracy = m["value"]

                    registro = {
                        "dataset": nombre_ds, "cubo": cubo, "motor": motor.nombre,
                        "repeticion": repeticion, "pliegue": pliegue_i, "estado": intento.estado,
                        "wall_s": round(transcurrido, 3), "auroc": auroc, "accuracy": accuracy,
                        "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                        "entorno_digest": entorno_digest, "motor_digest": digest_por_motor[motor.nombre],
                        "reusado": False,
                    }
                    resultados.append(registro)
                print(f"  rep={repeticion} pliegue={pliegue_i}: "
                     f"{sum(1 for r in resultados if r['dataset']==nombre_ds and r['repeticion']==repeticion and r['pliegue']==pliegue_i and r['estado']=='completed')}/4 completed",
                     flush=True)

    total = time.perf_counter() - inicio_total
    print(f"\n=== total: {total:.1f}s ({total/60:.1f} min), {len(resultados)} intentos "
         f"({reusados} reusados del caché, {len(resultados) - reusados} ejecutados) ===")

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds_por_intento": WALL_SECONDS,
        "procesos_en_paralelo": 1,
        "folds": FOLDS, "repeticiones": REPETICIONES_PEQUENO_MEDIANO,
        "total_wall_s": round(total, 1),
        "n_intentos": len(resultados),
        "n_reusados": reusados,
        "resultados": resultados,
    }
    salida["digest_resultados_crudos"] = digest_canonico(salida)
    ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")


if __name__ == "__main__":
    main()
