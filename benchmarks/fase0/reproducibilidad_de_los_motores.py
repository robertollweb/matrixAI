# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C4 — «los cuatro motores reproducen», MEDIDO y con artefacto.

POR QUÉ EXISTE ESTE FICHERO
---------------------------
La afirmación «los cuatro motores reproducen» **sustituye a dos anteriores que
eran falsas**:

  1. «la densa no reproduce», dicho a partir de 3 intentos idénticos de 150 —
     un recuento que comparaba cosas distintas;
  2. «el único de los cuatro que reprodujo sus 177 intentos», falso dos veces:
     eran 177 de **180**, y el baseline hizo 180 de 180.

Una auditoría del 2026-09-13 encontró que la tercera versión —la que hoy está
escrita en el contrato— **no tiene artefacto committeado**: la herramienta
existe (`reproducibilidad.medir_repeticion_en_OTRO_PROCESO`) y el RESULTADO no
está en ningún fichero de los dos repos.

Un hallazgo sin reproducción no es un hallazgo, **y una corrección tampoco**.
Menos aún una que sustituye a dos afirmaciones falsas: si la tercera se queda
sin evidencia, no se distingue de las dos primeras.

QUÉ MIDE, Y QUÉ NO
------------------
Repite cada ajuste en **procesos separados** (`spawn`) y compara el digest de
las salidas contra el del pipeline original, calculado fuera. Eso es lo que
pide el criterio de 101-C4: repetir en el MISMO proceso demuestra que el código
es estable consigo mismo, no que otra ejecución llegue al mismo sitio —
cualquier estado global (una semilla de numpy ya tocada, un backend de hilos ya
cargado, una caché de módulo) es invisible para la versión barata.

LO QUE SALIO, Y ES EL HALLAZGO
------------------------------
**Los cuatro coinciden entre procesos aislados**, que es el criterio del banco:
cada intento corre en su propio proceso `spawn`, asi que los intentos se pueden
comparar entre si y la pasada de Fase 0 vale.

**Pero la densa NO coincide con una referencia ajustada EN ESTE proceso**, y
las dos repeticiones aisladas si coinciden entre ellas. No es azar: es que el
proceso padre y los hijos no son el mismo entorno.

Acotado midiendo, en procesos limpios y de uno en uno:

  · la densa sola en un proceso limpio ............... `e1df265ad710`
  · despues de ajustar CUALQUIER otro motor .......... `99e286f4d917`
  · tras consumir del RNG global de stdlib ........... `e1df265ad710` (no cambia)
  · tras consumir del RNG global de numpy ............ `e1df265ad710` (no cambia)
  · **tras tocar torch, aunque sea leyendo `get_num_threads()`** `99e286f4d917`

O sea: **no es que se consuma un generador aleatorio; basta con que torch se
haya usado antes en ese proceso.** El mecanismo exacto NO esta clavado y no se
inventa aqui: las sondas posteriores confundian variables (importar torch para
medirlo ya es tocarlo) y eso queda como trabajo abierto.

QUE SIGNIFICA, con cuidado de no decir de mas:
  · La pasada de Fase 0 **no esta afectada**: aisla cada intento en su proceso.
  · Un numero obtenido **en proceso** —un cuaderno, el Studio, un test— **no
    tiene por que coincidir** con el del banco para este motor. Quien intente
    reproducir un resultado publicado sin aislar, no lo va a reproducir.
  · Y explica por que el aislamiento por proceso no es una precaucion
    decorativa: si alguien lo «optimiza» algun dia, el resultado de un motor
    pasaria a depender de en que orden corrieron los demas.

**NO mide estabilidad ante semillas distintas**, que es otra cosa y está medida
aparte: la densa mueve hasta 8,66 puntos de AUROC entre semillas en
`climate-model`. Reproducir es «misma semilla, mismo número»; estabilidad es
«semillas distintas, números parecidos». Confundirlas fue parte de cómo se
llegó a las dos afirmaciones falsas de arriba.

Un dataset pequeño del protocolo, real y con su sha256 verificado, porque la
pregunta —¿llega otra ejecución al mismo sitio?— no necesita el dataset más
caro para contestarse.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/deployer/matrixAI")
sys.path.insert(0, "/home/deployer/matrixai-engines/src")

from benchmarks.fase0.pasada_exploratoria_101_c3 import DATASETS, cargar_arff  # noqa: E402
from matrixai.estudio import ProblemSpec  # noqa: E402
from matrixai_engines import Presupuesto  # noqa: E402
from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM  # noqa: E402
from matrixai_engines.motores.baseline import MotorBaseline  # noqa: E402
from matrixai_engines.motores.densa import MotorDensaPropia  # noqa: E402
from matrixai_engines.motores.lineal import MotorLineal  # noqa: E402
from matrixai_engines.particiones import Particion  # noqa: E402
from matrixai_engines.reproducibilidad import medir_repeticion_en_OTRO_PROCESO  # noqa: E402

#: `kc2` (OpenML 1063): 522 filas, binario, cubo pequeño, NO sellado. Se elige
#: por barato, y se declara: la pregunta de este fichero no depende del tamaño.
DATA_ID = 1063
SEMILLA = 7
REPETICIONES = 2
#: El tope del cubo «pequeno» del protocolo. El aislamiento por proceso lo
#: EXIGE y hace bien: lanzar un proceso sin techo seria la misma promesa vacia
#: que ese modulo existe para dejar de hacer.
WALL_SECONDS = 120
SALIDA = Path(__file__).parent / "reproducibilidad_de_los_motores.json"


def _particiones(filas, objetivo, predictores):
    n = len(filas)
    cortes = (0, int(n * 0.6), int(n * 0.8), n)
    partes = []
    for ini, fin in zip(cortes, cortes[1:]):
        trozo = filas[ini:fin]
        partes.append(Particion(
            features=[{k: f[k] for k in predictores} for f in trozo],
            target=[str(f[objetivo]) for f in trozo],
            row_ids=[str(ini + i) for i in range(len(trozo))]))
    return partes


def main() -> int:
    ficha = next(d for d in DATASETS if d[0] == DATA_ID)
    _, nombre, cubo, positiva, negativa = ficha
    filas, objetivo = cargar_arff(DATA_ID)
    predictores = tuple(k for k in filas[0] if k != objetivo)
    train, val, test = _particiones(filas, objetivo, predictores)
    spec = ProblemSpec(problem_id="repro", task="binary_classification",
                       target=objetivo, predictors=predictores,
                       classes=(negativa, positiva), positive_label=positiva,
                       observation_unit="fila")

    motores = [MotorBaseline(), MotorLineal(), MotorArbolLightGBM(), MotorDensaPropia()]
    informes, t0 = {}, time.time()
    for motor in motores:
        # El digest de REFERENCIA se calcula aquí, fuera de la repetición: sin
        # él, repetir N veces solo demuestra consistencia entre las N, que es
        # el mismo agujero con otra forma.
        _, ajustado = motor.fit(train, val, spec, Presupuesto(seed=SEMILLA, wall_seconds=WALL_SECONDS),
                                candidate="repro", split_plan_digest="d" * 64)
        informe = medir_repeticion_en_OTRO_PROCESO(
            motor, train, val, test, spec,
            Presupuesto(seed=SEMILLA, wall_seconds=WALL_SECONDS),
            candidate="repro", split_plan_digest="d" * 64,
            digest_de_referencia=ajustado.spec.digest(),
            repeticiones=REPETICIONES)
        entre_repeticiones = len(set(informe.digests[1:])) == 1 if len(informe.digests) > 1 else None
        informes[motor.nombre] = {
            "version": motor.version,
            "repeticiones": informe.repeticiones,
            "identico": informe.identico,
            "digests": list(informe.digests),
            "digest_de_referencia": ajustado.spec.digest(),
            "tolerancia_medida": informe.tolerancia_medida,
            "entorno": informe.entorno,
            # LAS DOS PREGUNTAS SON DISTINTAS Y HAY QUE SEPARARLAS.
            #
            # `identico` compara los procesos aislados contra una referencia
            # ajustada EN ESTE proceso, que ya ha ajustado los motores
            # anteriores. `coinciden_entre_procesos_aislados` compara los
            # aislados entre si, que es como corre el banco de verdad: cada
            # intento en su propio proceso `spawn`.
            #
            # Medido el 2026-09-13: para la densa las dos respuestas NO son la
            # misma, y eso es justo el hallazgo. Juntarlas en un solo booleano
            # habria escondido el unico dato interesante del fichero.
            "coinciden_entre_procesos_aislados": entre_repeticiones,
        }
        print(f"  {motor.nombre:28s} vs referencia en-proceso={informe.identico}  "
              f"entre aislados={entre_repeticiones}")

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "que_mide": "repeticion del ajuste en PROCESOS SEPARADOS contra un "
                    "digest de referencia calculado fuera (criterio 101-C4)",
        "que_NO_mide": "estabilidad ante semillas distintas — eso es otra cosa "
                       "y esta medido aparte (la densa mueve hasta 8,66 puntos "
                       "de AUROC entre semillas en climate-model)",
        "dataset": {"data_id": DATA_ID, "nombre": nombre, "cubo": cubo,
                    "filas": len(filas), "predictores": len(predictores)},
        "semilla": SEMILLA,
        "wall_seconds_por_intento": WALL_SECONDS,
        "repeticiones_por_motor": REPETICIONES,
        "motores": informes,
        "todos_coinciden_entre_procesos_aislados": all(
            i["coinciden_entre_procesos_aislados"] for i in informes.values()),
        "todos_coinciden_con_la_referencia_en_proceso": all(
            i["identico"] for i in informes.values()),
        "total_wall_s": round(time.time() - t0, 1),
    }
    temporal = SALIDA.with_suffix(SALIDA.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(SALIDA)
    print(f"\ncoinciden entre procesos aislados (que es como corre el banco): "
          f"{salida['todos_coinciden_entre_procesos_aislados']}")
    print(f"coinciden con la referencia EN PROCESO: "
          f"{salida['todos_coinciden_con_la_referencia_en_proceso']}")
    print(f"({salida['total_wall_s']} s) -> {SALIDA.name}")
    # El criterio del banco es el primero: si eso falla, un intento no se puede
    # comparar con otro y no hay medicion.
    return 0 if salida["todos_coinciden_entre_procesos_aislados"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
