#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C2 — `tabicl.v2` frente a `lightgbm` (v2), en su régimen de CPU.

Implementa TAL CUAL el «REGISTRO DE C2, escrito el 2026-09-25 ANTES de
construir su pasada» (`documentacion/116_FUNDACIONAL_TABULAR_CONTRACT.md`,
corte C2, contrato 116). No se cambia nada de ahí: este guion es su
implementación, no una segunda decisión.

QUÉ SE REUTILIZA, Y DE DÓNDE — nada de lo de abajo se copia, se importa,
mismo patrón que `pasada_114c6_ensamblado.py`:

* **El catálogo, la partición, la preparación y el presupuesto de la v2**
  (`pasada_v2_113.fijar_el_catalogo_v2`, `pasada_amplia_101_c5.
  particiones_base`/`.datasets_de_la_pasada`/`.metrica_de_cierre_por_dataset`/
  `.metricas_del_informe`/`.aplanar_metricas_en_el_registro`,
  `pasada_exploratoria_101_c3.preparar_para_motor`/`.wall_seconds_del_cubo`):
  la MISMA partición, semillas y presupuesto por cubo que midió lightgbm en
  la v2 (113-C4), para que la comparación empareje de verdad — «MISMA
  partición, semillas, presupuesto por cubo y métrica de cierre que la v2»,
  literal del registro de C2.
* **`matrixai_engines.subproceso.ejecutar_intento_aislado`**: el mismo
  aislamiento por proceso (con su tope de pared) que usan los 7 motores de
  la v2 y el ensamblado de 114-C6.
* **`protocolo._intervalo_pareado` y `protocolo.aplicar_regla_de_cierre`**:
  la MISMA aritmética de bootstrap pareado (1.000 remuestras de pliegues,
  semilla 0) y la MISMA regla de cartera («a 2 puntos del mejor en ≥80%») —
  no una segunda implementación de ninguna de las dos.
* **La caché por intento, la procedencia y la escritura atómica**
  (`_reusable`, `_cargar_cache`, `procedencia_de_la_medicion`,
  `procedencia_declarada`, `_procedencias_citadas`, `sellar_la_salida` de
  `pasada_exploratoria_101_c3.py`): el mismo mecanismo ya auditado.
* **El techo de CPU en sí** (`MAX_FILAS_DE_CONTEXTO_EN_CPU`,
  `MAX_COLUMNAS_EN_CPU`), importado de `matrixai_engines.motores.tabicl` —
  nunca copiado a una segunda constante que pudiera divergir.

QUÉ ES NUEVO AQUÍ, Y POR QUÉ NO PODÍA REUTILIZARSE:

1. **El régimen, por REGLA y no a ojo.** El registro de C2 dice: «los
   conjuntos NO sellados en los que el motor ACEPTA el ajuste en TODOS sus
   pliegues». Eso se calcula aquí como una función PURA de la partición —
   filas de entrenamiento de cada pliegue y número de predictores,
   comparados con `MAX_FILAS_DE_CONTEXTO_EN_CPU`/`MAX_COLUMNAS_EN_CPU` con
   la MISMA comparación (`>`) que `MotorTabICL._ajustar()` hace de PRIMERO,
   antes de tocar pesos o importar `tabicl` (ver `motores/tabicl.py`, paso
   1 de `_ajustar`). No hace falta `tabicl` instalado para calcularla —por
   eso corre en el host, y por eso tiene sus pruebas ahí
   (`tests/test_116_c2_arnes_tabicl.py`, sin la biblioteca—. Un conjunto
   con AL MENOS UN pliegue rechazado queda FUERA del régimen ENTERO, con
   el motivo y lo que pidió cada pliegue rechazado (campo/valor/opciones,
   como el propio motor los declara).
2. **El emparejamiento con lightgbm de la v2, SIN remedirlo.** Los
   registros de `lightgbm` salen de `pasada_v2_113_resultado.json` tal
   cual —«no se vuelve a medir», literal del registro—. Si a un conjunto
   del régimen le falta en ese artefacto algún pliegue que `tabicl.v2` SÍ
   completó, el conjunto entero se declara `sin_lightgbm_emparejable` y
   queda fuera de la comparación (60% y veredicto por conjunto), aunque
   sigue contando para la regla de la cartera (que promedia por dataset,
   no empareja por pliegue).
3. **El bucle de UN motor**, no de siete ni de un ensamblado: solo
   `tabicl.v2`, resuelto por `matrixai_engines.registro_de_motores.
   motor_para("tabicl.v2")` — el mismo camino que usa cualquier otro
   motor, sin caso especial.
4. **`--estimar` con la aritmética de 116-C0**, no con el «peor caso: agota
   el presupuesto entero» de C5/C6. El coste de `tabicl.v2` NO es function
   del presupuesto del cubo —es un modelo en contexto que predice en una
   sola llamada por lote (`_predecir`/`_probabilidades` de
   `motores/tabicl.py` pasan TODO el pliegue de prueba de una vez, nunca
   fila a fila)—, y crece con las filas de contexto y las columnas, medido
   en `benchmarks/tabicl_116c0/resultado_116c0.json` (`ms_por_fila["todas"]`
   por punto de la rejilla). El peor caso de C5/C6 (presupuesto del cubo
   agotado siempre) no dice nada de esto: sobrestimaría un conjunto de 8
   columnas y 500 filas de contexto (que tarda segundos) y no distinguiría
   uno de 32 columnas y 2.000 filas (que tarda minutos). Se usa la tabla
   medida, redondeando cada pliegue al punto de la rejilla que sea `>=` en
   filas Y en columnas (nunca hacia abajo — un techo que subestima el
   coste es el mismo defecto que costó `WALL_SECONDS = 120.0` en C3, ver
   `pasada_exploratoria_101_c3.wall_seconds_del_cubo`), y el MÁXIMO entre
   las tres fuentes que C0 midió en ese punto (`optdigits`/`adult`/
   `sintetico`).
5. **`--estimar` decide el régimen con ARITMÉTICA, no leyendo los ARFF —
   medido el 2026-09-25.** El régimen del punto 1 (`regimen_del_dataset`)
   llama a `particiones_base`, que lee y particiona el ARFF de cada
   conjunto: para `--estimar`, que solo necesita filas y columnas, eso
   significaba leer y particionar los 32 no sellados ENTEROS —incluidos
   los del cubo grande, hasta 188.318 filas (`Allstate_Claims_Severity`)—
   antes de imprimir una sola cifra. Medido en vivo: más de 35 minutos de
   CPU sin terminar. `regimen_estimado_desde_metadatos` calcula el mismo
   régimen (misma regla, misma `MAX_FILAS_DE_CONTEXTO_EN_CPU`/
   `MAX_COLUMNAS_EN_CPU`) a partir de `particion_por_dataset` de
   `pasada_v2_113_resultado.json` —la partición que la v2 YA MIDIÓ, una
   vez— con una aproximación aritmética (`(k-1)/k` de la reserva de
   desarrollo, redondeada hacia ARRIBA) declarada como tal
   (`estimado: True`). Es una ESTIMACIÓN, no la medición del corte: la
   pasada real (sin `--estimar`) sigue calculando el régimen EXACTO,
   pliegue a pliegue, sobre los datos leídos de verdad.

DÓNDE CORRE DE VERDAD: dentro de la imagen `medicion-tabicl-116c0` (el host
no tiene `tabicl`, a propósito — ver `benchmarks/fase0/
correr_116c2_en_contenedor.sh`), con `--network none`. El cálculo del
régimen y `--estimar` SÍ corren en el host (no necesitan `tabicl`).

CÓMO SE LANZA:

    python3 benchmarks/fase0/pasada_116c2_tabicl.py --estimar
    python3 benchmarks/fase0/pasada_116c2_tabicl.py --solo dresses-sales \\
        --salida /tmp/prueba_c2.json
    benchmarks/fase0/correr_116c2_en_contenedor.sh    # LA PASADA DE VERDAD,
        # dentro del contenedor, desde la cola nocturna — nunca a mano un
        # ratito.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import pasada_v2_113 as v2  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.ensamblado import (  # noqa: E402
    nombres_de_los_miembros_aprobados)
from matrixai_engines.motores.tabicl import (  # noqa: E402
    MAX_COLUMNAS_EN_CPU, MAX_FILAS_DE_CONTEXTO_EN_CPU)
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.registro_de_motores import motor_para  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

RUTA_DEL_PROTOCOLO_V2 = _AQUI / "protocolo_exploratorio_v2.json"
RUTA_DE_LA_V2 = _AQUI / "pasada_v2_113_resultado.json"
RUTA_DE_C0 = _AQUI.parent / "tabicl_116c0" / "resultado_116c0.json"
RUTA_DE_SALIDA = _AQUI / "resultado_116c2_tabicl.json"

NOMBRE_TABICL = "tabicl.v2"
NOMBRE_LIGHTGBM = "lightgbm"

#: EL MARGEN DE EQUIVALENCIA PRÁCTICA — el mismo que 114-C6 (registro de C2:
#: «el mismo margen que 114-C6»), en la escala CRUDA de la métrica de
#: cierre (AUROC/accuracy/R², 0-1), no en puntos x100.
MARGEN_DE_EQUIVALENCIA = 0.005

#: LA SEMILLA Y LAS REMUESTRAS DEL BOOTSTRAP — las de `protocolo.
#: _intervalo_pareado` por omisión, escritas aquí para que quien lea este
#: fichero no tenga que ir a comprobarlas a otro.
SEMILLA_DEL_BOOTSTRAP = protocolo_mod.SEMILLA_DEL_INTERVALO  # 0
REMUESTRAS_DEL_BOOTSTRAP = protocolo_mod.REMUESTRAS_DEL_INTERVALO  # 1000

#: EL MÍNIMO DEL CRITERIO DEL CORTE — «mejora en ≥60% de los conjuntos de
#: su régimen», el mismo listón que 114-C6.
MINIMO_DE_MEJORA = 0.60

#: DIRECCIÓN DE CADA MÉTRICA DE CIERRE: True si MÁS es MEJOR. Las tres que
#: la v2 usa (auroc, accuracy, r2) ya lo son. Un `metric_id` fuera de aquí
#: PARA la pasada en vez de asumir una dirección (mismo criterio que C6).
METRICAS_MAS_ES_MEJOR = {"auroc": True, "accuracy": True, "r2": True}


def direccion_de(metric_id: str) -> bool:
    if metric_id not in METRICAS_MAS_ES_MEJOR:
        raise SystemExit(
            f"la metrica de cierre {metric_id!r} no tiene direccion declarada en "
            f"METRICAS_MAS_ES_MEJOR ({sorted(METRICAS_MAS_ES_MEJOR)}). Anadirla es una "
            f"decision que se toma ANTES de medir y por escrito, no un valor de repuesto")
    return METRICAS_MAS_ES_MEJOR[metric_id]


# ---------------------------------------------------------------------------
# 1. EL PROTOCOLO Y LOS CONJUNTOS — la MISMA v2, sin sellados
# ---------------------------------------------------------------------------

def preparar_protocolo_v2() -> None:
    """Apunta la lectura de C3/C5 al protocolo v2 y fija su catálogo — la
    MISMA operación que `pasada_v2_113.main()` y `pasada_114c6_ensamblado.
    preparar_protocolo_v2()`, reutilizada tal cual."""
    c3.RUTA_DEL_PROTOCOLO = RUTA_DEL_PROTOCOLO_V2
    v2.fijar_el_catalogo_v2()


def datasets_no_sellados(protocolo: protocolo_mod.ProtocoloExploratorio):
    """Los 32 conjuntos NO sellados del protocolo v2 — el universo de
    partida antes de aplicar el régimen. Los 8 sellados no se tocan salvo
    que `--solo` los nombre a propósito (avisa en voz alta, ver `main`)."""
    return [d for d in c5.datasets_de_la_pasada(protocolo) if not d.sellado]


# ---------------------------------------------------------------------------
# 2. EL RÉGIMEN — POR REGLA, la MISMA que aplica `MotorTabICL._ajustar()`
# ---------------------------------------------------------------------------

def folds_del_dataset(ds, protocolo: protocolo_mod.ProtocoloExploratorio):
    """`(repeticion, pliegue, filas_de_entrenamiento)` de cada pliegue REAL
    del dataset, el número de columnas (`len(spec.predictors)`, el mismo
    valor que `MotorTabICL._ajustar()` compara) y de filas de prueba — todo
    de `particiones_base`, sin volver a leer el ARFF. Devuelve también la
    tupla `base` entera para que el bucle de medición la reutilice si el
    conjunto entra en régimen."""
    base = c5.particiones_base(ds, protocolo)
    por_id, propuesta, spec, objetivo, predictores, declarada = base
    repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
    folds: list[tuple[int, int, int]] = []
    for repeticion in range(repeticiones):
        for pliegue_i in range(protocolo.particion.folds):
            pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
            if pliegue is None:
                continue
            folds.append((repeticion, pliegue_i, len(pliegue.entrena)))
    n_test = len(propuesta.plan.observaciones_del_rol("test"))
    return folds, len(spec.predictors), n_test, base


def rechazos_de_regimen(folds: list[tuple[int, int, int]], n_columnas: int) -> list[dict]:
    """Los pliegues que el TÉCHO de CPU de `tabicl.v2` rechazaría, con la
    MISMA comparación (`>`, inclusive en el borde) que
    `MotorTabICL._ajustar()` hace de PRIMERO — importada de `matrixai_
    engines.motores.tabicl`, nunca copiada aquí como una segunda constante.

    Función PURA: ni `tabicl` ni el `Motor` entran en ella, así que corre
    en el host sin la biblioteca instalada — el mismo terreno donde vive su
    prueba (`tests/test_116_c2_arnes_tabicl.py`)."""
    columnas_exceden = n_columnas > MAX_COLUMNAS_EN_CPU
    rechazos: list[dict] = []
    for repeticion, pliegue, n_filas in folds:
        motivos = []
        if n_filas > MAX_FILAS_DE_CONTEXTO_EN_CPU:
            motivos.append(f"filas de contexto: pide {n_filas}, tope "
                           f"{MAX_FILAS_DE_CONTEXTO_EN_CPU}")
        if columnas_exceden:
            motivos.append(f"columnas: pide {n_columnas}, tope {MAX_COLUMNAS_EN_CPU}")
        if motivos:
            rechazos.append({"repeticion": repeticion, "pliegue": pliegue,
                             "n_filas_de_contexto": n_filas, "n_columnas": n_columnas,
                             "motivo": "; ".join(motivos)})
    return rechazos


def regimen_del_dataset(ds, protocolo: protocolo_mod.ProtocoloExploratorio):
    """El régimen de UN conjunto: `en_regimen` es `True` solo si NINGÚN
    pliegue fue rechazado — «el motor acepta en TODOS los pliegues»,
    literal del registro de C2. Un solo pliegue rechazado saca al conjunto
    ENTERO, no solo ese pliegue. Devuelve también la tupla `base` de
    `particiones_base`, para reutilizarla si el conjunto entra."""
    folds, n_columnas, n_test, base = folds_del_dataset(ds, protocolo)
    rechazos = rechazos_de_regimen(folds, n_columnas)
    regimen = {
        "dataset": ds.nombre, "n_columnas": n_columnas,
        "max_filas_de_contexto": max((f[2] for f in folds), default=0),
        "n_pliegues": len(folds), "n_test": n_test,
        "en_regimen": not rechazos, "rechazos_por_pliegue": rechazos,
    }
    return regimen, base


def motivo_fuera_de_regimen(regimen: dict) -> str:
    rechazos = regimen["rechazos_por_pliegue"]
    ejemplos = "; ".join(f"rep={r['repeticion']} pliegue={r['pliegue']}: {r['motivo']}"
                         for r in rechazos[:3])
    n = len(rechazos)
    cola = " ..." if n > 3 else ""
    return (f"{n} de {regimen['n_pliegues']} pliegues rechazados por el techo de CPU "
            f"(MAX_FILAS_DE_CONTEXTO_EN_CPU={MAX_FILAS_DE_CONTEXTO_EN_CPU}, "
            f"MAX_COLUMNAS_EN_CPU={MAX_COLUMNAS_EN_CPU}): {ejemplos}{cola}")


def cargar_particion_por_dataset_v2(ruta: Path = RUTA_DE_LA_V2) -> dict[str, dict]:
    """`particion_por_dataset` del artefacto v2 — la partición que 113-C4 YA
    MIDIÓ (`n_filas_con_objetivo`, `n_test`, `n_predictores`,
    `folds_obtenidos`, `n_pliegues_obtenidos`, por dataset), leída del JSON
    sin tocar un solo ARFF. Es lo único que `regimen_estimado_desde_metadatos`
    necesita para estimar el régimen SIN releer ni reparticionar los datos —
    ver su docstring y el punto 4 del docstring del módulo."""
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    return dict(payload["particion_por_dataset"])


def regimen_estimado_desde_metadatos(ds_nombre: str, meta: dict) -> dict:
    """El régimen de UN conjunto, ESTIMADO con la aritmética de la partición
    que la v2 YA MIDIÓ (`particion_por_dataset` de `pasada_v2_113_resultado.
    json`) — sin leer ni particionar el ARFF de nuevo. SOLO para `--estimar`:
    la pasada real sigue calculando el régimen EXACTO, pliegue a pliegue,
    sobre los datos leídos de verdad (`regimen_del_dataset`, que usa
    `particiones_base`).

    **Por qué esto hace falta, medido el 2026-09-25**: `regimen_del_dataset`
    llama a `particiones_base`, que lee y particiona el ARFF de CADA
    conjunto — para `--estimar`, que solo necesita un número de filas y
    columnas, eso es leer y particionar los 32 no sellados ENTEROS,
    incluidos los del cubo grande (`APSFailure`, `Allstate_Claims_Severity`,
    `kick`...: hasta 188.318 filas). Se midió que ese cálculo por sí solo
    tarda más de 35 minutos de CPU y no había terminado — un coste que
    `--estimar` no puede permitirse, porque su propósito es dar una cifra
    de un vistazo antes de medir nada. La aritmética de abajo usa los
    mismos números que YA están en el artefacto v2 (que sí leyó y
    particionó los datos, una vez, en su momento) y no vuelve a tocarlos.

    **La aproximación, dicha por su nombre**: el `k`-fold divide la reserva
    de desarrollo (`n_filas_con_objetivo - n_test`) en `folds_obtenidos`
    partes iguales por diseño; el pliegue de entrenamiento de cada vuelta es
    todas menos una, `(k-1)/k` de esa reserva — se redondea hacia ARRIBA
    (`math.ceil`), nunca hacia abajo (mismo criterio que `_bucket_hacia_
    arriba`: subestimar el contexto podría meter dentro del régimen un
    conjunto que en la partición real no cabe). Esto puede diferir en un
    puñado de filas del pliegue EXACTO si la partición real no reparte
    perfectamente igual entre pliegues (estratificación, redondeos) — por
    eso es una ESTIMACIÓN y se declara `estimado=True`, no el régimen que
    cuenta para el corte."""
    n_columnas = meta["n_predictores"]
    dev_pool = meta["n_filas_con_objetivo"] - meta["n_test"]
    folds = meta["folds_obtenidos"] or 1
    max_filas_de_contexto = (math.ceil(dev_pool * (folds - 1) / folds)
                             if folds > 1 else dev_pool)
    n_pliegues = meta["n_pliegues_obtenidos"]
    folds_sinteticos = [(0, i, max_filas_de_contexto) for i in range(n_pliegues)]
    rechazos = rechazos_de_regimen(folds_sinteticos, n_columnas)
    return {
        "dataset": ds_nombre, "n_columnas": n_columnas,
        "max_filas_de_contexto": max_filas_de_contexto,
        "n_pliegues": n_pliegues, "n_test": meta["n_test"],
        "en_regimen": not rechazos, "rechazos_por_pliegue": rechazos,
        "estimado": True,
    }


# ---------------------------------------------------------------------------
# 3. LIGHTGBM Y EL CAMPO DE LA V2 — LEÍDOS, NUNCA REMEDIDOS
# ---------------------------------------------------------------------------

def cargar_resultados_v2(ruta: Path = RUTA_DE_LA_V2) -> list[dict]:
    """TODOS los registros crudos de la v2 (113-C4, sus 7 motores) — «los
    demás motores de la v2 con sus resultados», el campo de la regla de la
    cartera del registro de C2. Se lee una vez; el llamante filtra por
    motor o por dataset según lo que necesite en cada sitio."""
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    return list(payload["resultados"])


def por_pliegue(registros: list[dict], *, dataset: str, motor: str,
                metric_id: str) -> dict[tuple, float]:
    """`(repeticion, pliegue) -> valor` de `metric_id`, SOLO de los
    intentos de `motor` en `dataset` cuyo estado cuenta como medida (la
    MISMA lista blanca de `protocolo.ESTADOS_QUE_CUENTAN_COMO_MEDIDA`) y
    con valor no ausente."""
    return {(r["repeticion"], r["pliegue"]): r[metric_id]
            for r in registros
            if r["dataset"] == dataset and r["motor"] == motor
            and r.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
            and r.get(metric_id) is not None}


def media_de(por_pliegue_mapa: dict[tuple, float]) -> float | None:
    valores = list(por_pliegue_mapa.values())
    return (sum(valores) / len(valores)) if valores else None


def fallos_de(registros: list[dict], *, dataset: str, motor: str) -> list[dict]:
    return [r for r in registros if r["dataset"] == dataset and r["motor"] == motor
            and r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]


def diferencias_emparejadas(por_pliegue_a: dict[tuple, float],
                            por_pliegue_b: dict[tuple, float],
                            ) -> tuple[list[float], list[tuple]]:
    """`a - b` en los pliegues COMUNES, en orden estable — el bootstrap
    remuestrea sobre una lista y su orden decide qué índice cae en qué
    remuestra (mismo patrón que `pasada_114c6_ensamblado`)."""
    comunes = sorted(set(por_pliegue_a) & set(por_pliegue_b))
    return [por_pliegue_a[k] - por_pliegue_b[k] for k in comunes], comunes


# ---------------------------------------------------------------------------
# 4. EL VEREDICTO POR CONJUNTO: tabicl.v2 − lightgbm, emparejado
# ---------------------------------------------------------------------------

def veredicto_del_conjunto(*, dataset: str, tabicl_media: float | None,
                            lightgbm_media: float | None,
                            tabicl_por_pliegue: dict[tuple, float],
                            lightgbm_por_pliegue: dict[tuple, float],
                            fallo_de_tabicl: str | None) -> dict:
    """El veredicto de UN conjunto MEDIBLE (dentro de régimen y con
    lightgbm emparejable), tal como lo define el registro de C2: diferencia
    emparejada (tabicl.v2 − lightgbm) por repetición y pliegue, intervalo
    al 95% por bootstrap de percentiles de PLIEGUES (`protocolo.
    _intervalo_pareado`, 1.000 remuestras, semilla 0). «Mejora» = el
    intervalo excluye el cero por arriba Y no cabe entero dentro de ±0,005.

    `fallo_de_tabicl` es la ÚNICA de las tres exclusiones del registro que
    SÍ llega hasta aquí y SÍ cuenta en el 60% —«un fallo de TabICL cuenta
    como no mejora», literal—: fuera de régimen y sin lightgbm emparejable
    se resuelven ANTES, en `clasificar_conjunto`, y no producen veredicto."""
    base = {"dataset": dataset, "tabicl_media": tabicl_media, "lightgbm_media": lightgbm_media}
    if fallo_de_tabicl:
        return {**base, "mejora": False, "intervalo": None, "n_pliegues_comunes": 0,
                "motivo": f"fallo de tabicl.v2: {fallo_de_tabicl}"}
    diferencias, comunes = diferencias_emparejadas(tabicl_por_pliegue, lightgbm_por_pliegue)
    if len(diferencias) < 2:
        return {**base, "mejora": False, "intervalo": None, "n_pliegues_comunes": len(comunes),
                "motivo": f"menos de dos pliegues comunes ({len(comunes)}) entre tabicl.v2 y "
                          f"lightgbm: no hay nada que remuestrear"}
    limites = protocolo_mod._intervalo_pareado(
        diferencias, semilla=SEMILLA_DEL_BOOTSTRAP, remuestras=REMUESTRAS_DEL_BOOTSTRAP)
    if limites is None:
        return {**base, "mejora": False, "intervalo": None, "n_pliegues_comunes": len(comunes),
                "motivo": "el bootstrap no devolvio intervalo"}
    bajo, alto = limites
    excluye_cero_arriba = bajo > 0.0
    cabe_en_el_margen = (bajo >= -MARGEN_DE_EQUIVALENCIA) and (alto <= MARGEN_DE_EQUIVALENCIA)
    mejora = excluye_cero_arriba and not cabe_en_el_margen
    motivo = None
    if not mejora:
        motivo = ("el intervalo no excluye el cero por arriba"
                   if not excluye_cero_arriba else
                   f"el intervalo [{bajo:.5f}, {alto:.5f}] cabe entero dentro del margen "
                   f"de equivalencia ±{MARGEN_DE_EQUIVALENCIA}")
    return {
        **base, "mejora": mejora, "n_pliegues_comunes": len(comunes), "motivo": motivo,
        "intervalo": {
            "bajo": bajo, "alto": alto, "nivel": 0.95,
            "emparejado_por": "repeticion y pliegue",
            "remuestreo": (f"bootstrap de percentiles, {REMUESTRAS_DEL_BOOTSTRAP} remuestras "
                           f"de PLIEGUES, semilla {SEMILLA_DEL_BOOTSTRAP} "
                           f"(protocolo._intervalo_pareado, reutilizada)"),
            "margen_de_equivalencia": MARGEN_DE_EQUIVALENCIA,
            "excluye_cero_arriba": excluye_cero_arriba, "cabe_en_el_margen": cabe_en_el_margen,
        },
    }


def clasificar_conjunto(ds_nombre: str, regimen: dict, registros_tabicl: list[dict],
                        resultados_v2: list[dict], metric_id: str) -> dict:
    """Clasifica UN conjunto no sellado en una de tres categorías, EN ESTE
    ORDEN (el registro de C2 las declara así, y el orden importa: un
    conjunto fuera de régimen no llega a preguntarse por lightgbm):

    1. `fuera_de_regimen` — algún pliegue rechazado por el técho de CPU.
       `medible=False`: NO cuenta en el 60% ni en el veredicto por
       conjunto. Sí puede seguir contando para la regla de la cartera si
       se le pide con su nombre incluido (aquí no: la cartera solo mira
       los del régimen, ver sección 5).
    2. `sin_lightgbm_emparejable` — dentro del régimen, `tabicl.v2` completó
       TODOS sus pliegues, pero lightgbm (de la v2, sin remedir) no tiene
       alguno de ellos. `medible=False`: «si falta un pliegue suyo, dilo,
       ese conjunto fuera», literal del registro. SÍ sigue contando para
       la regla de la cartera (que promedia por dataset, no empareja).
    3. `medible` — con un veredicto pareado (que puede ser «no mejora» por
       fallo de tabicl.v2, por intervalo que cruza el cero, o por caber
       dentro del margen; o «mejora»).
    """
    if not regimen["en_regimen"]:
        return {"dataset": ds_nombre, "categoria": "fuera_de_regimen", "medible": False,
                "motivo": motivo_fuera_de_regimen(regimen), "veredicto": None}

    fallos = fallos_de(registros_tabicl, dataset=ds_nombre, motor=NOMBRE_TABICL)
    tabicl_pp = por_pliegue(registros_tabicl, dataset=ds_nombre, motor=NOMBRE_TABICL,
                            metric_id=metric_id)
    lightgbm_pp = por_pliegue(resultados_v2, dataset=ds_nombre, motor=NOMBRE_LIGHTGBM,
                              metric_id=metric_id)
    if fallos:
        motivo_fallo = "; ".join(
            f"rep={r['repeticion']} pliegue={r['pliegue']} estado={r['estado']} "
            f"motivo={r.get('motivo')}" for r in fallos)
        veredicto = veredicto_del_conjunto(
            dataset=ds_nombre, tabicl_media=media_de(tabicl_pp),
            lightgbm_media=media_de(lightgbm_pp), tabicl_por_pliegue=tabicl_pp,
            lightgbm_por_pliegue=lightgbm_pp, fallo_de_tabicl=motivo_fallo)
        return {"dataset": ds_nombre, "categoria": "medible", "medible": True,
                "motivo": None, "veredicto": veredicto}

    faltan = sorted(set(tabicl_pp) - set(lightgbm_pp))
    if faltan:
        return {"dataset": ds_nombre, "categoria": "sin_lightgbm_emparejable",
                "medible": False,
                "motivo": (f"lightgbm (v2) no tiene {len(faltan)} de los {len(tabicl_pp)} "
                          f"pliegues que tabicl.v2 completo: {faltan}. No se remide "
                          f"lightgbm (registro de C2): este conjunto queda fuera de la "
                          f"comparacion, aunque sigue contando para la regla de la "
                          f"cartera."),
                "veredicto": None}

    veredicto = veredicto_del_conjunto(
        dataset=ds_nombre, tabicl_media=media_de(tabicl_pp), lightgbm_media=media_de(lightgbm_pp),
        tabicl_por_pliegue=tabicl_pp, lightgbm_por_pliegue=lightgbm_pp, fallo_de_tabicl=None)
    return {"dataset": ds_nombre, "categoria": "medible", "medible": True,
            "motivo": None, "veredicto": veredicto}


def veredicto_del_corte(clasificaciones_medibles: list[dict], *,
                        minimo: float = MINIMO_DE_MEJORA) -> dict:
    """La proporción de «mejora» sobre los conjuntos MEDIBLES (régimen y
    con lightgbm emparejable), y si cumple el mínimo del corte (60%)."""
    total = len(clasificaciones_medibles)
    mejoras = sum(1 for c in clasificaciones_medibles if c["veredicto"]["mejora"])
    fraccion = (mejoras / total) if total else 0.0
    return {"n_conjuntos": total, "n_mejora": mejoras, "fraccion_mejora": fraccion,
            "minimo_exigido": minimo, "cumple": total > 0 and fraccion >= minimo,
            "conjuntos_que_mejoran": sorted(c["dataset"] for c in clasificaciones_medibles
                                            if c["veredicto"]["mejora"])}


# ---------------------------------------------------------------------------
# 5. LA REGLA DE LA CARTERA, CON TABICL DENTRO DEL CAMPO
# ---------------------------------------------------------------------------

def _resumen(v: dict) -> dict:
    return {k: v[k] for k in ("cumplidos", "datasets", "fraccion", "cumple_la_regla")}


def regla_de_la_cartera_con_tabicl(resultados_tabicl: list[dict], resultados_v2: list[dict], *,
                                   regla: protocolo_mod.ReglaDeCierre,
                                   metrica_por_dataset: dict[str, str],
                                   nombres_del_regimen: list[str]) -> dict:
    """«La regla de cierre de la cartera... rehecha en los conjuntos del
    régimen con TabICL dentro del campo (los demás motores de la v2 con
    sus resultados)» — literal del registro de C2.

    El CAMPO es los 7 motores de la v2 YA MEDIDOS (sin remedir NINGUNO,
    tampoco lightgbm) restringidos a los conjuntos del régimen, MÁS
    `tabicl.v2` medido en esta pasada. Se calcula dos veces por motor
    aprobado de la cartera de HOY: sin `tabicl.v2` en el campo (`antes`) y
    con él dentro (`despues`) — mismo patrón que `pasada_114c6_ensamblado.
    regla_de_la_cartera_antes_y_despues`, generalizado aquí a un candidato
    con nombre en vez del ensamblado fijo. Y una entrada extra para
    `tabicl.v2` mismo: ¿CUMPLE la regla de la cartera, con el campo entero
    dentro?

    `datasets_exigidos` es el régimen ENTERO (no solo los «medibles» del
    veredicto pareado de la sección 4): la regla de la cartera promedia
    POR DATASET, no empareja por pliegue, así que un conjunto sin lightgbm
    emparejable en TODOS sus pliegues puede seguir aportando su media aquí.
    """
    v2_en_regimen = [r for r in resultados_v2 if r["dataset"] in nombres_del_regimen]
    tabicl_en_regimen = [r for r in resultados_tabicl if r["dataset"] in nombres_del_regimen]
    campo_con_tabicl = v2_en_regimen + tabicl_en_regimen

    salida: dict[str, dict] = {
        NOMBRE_TABICL: _resumen(protocolo_mod.aplicar_regla_de_cierre(
            campo_con_tabicl, regla, motor=NOMBRE_TABICL,
            metrica_por_dataset=metrica_por_dataset, datasets_exigidos=nombres_del_regimen)),
    }
    for aprobado in nombres_de_los_miembros_aprobados():
        antes = protocolo_mod.aplicar_regla_de_cierre(
            v2_en_regimen, regla, motor=aprobado, metrica_por_dataset=metrica_por_dataset,
            datasets_exigidos=nombres_del_regimen)
        despues = protocolo_mod.aplicar_regla_de_cierre(
            campo_con_tabicl, regla, motor=aprobado, metrica_por_dataset=metrica_por_dataset,
            datasets_exigidos=nombres_del_regimen)
        salida[aprobado] = {"antes": _resumen(antes), "despues": _resumen(despues),
                            "diferencia_cumplidos": despues["cumplidos"] - antes["cumplidos"]}
    return salida


# ---------------------------------------------------------------------------
# 6. LA CUENTA DE LO QUE VA A COSTAR (--estimar), CON LA ARITMÉTICA DE C0
# ---------------------------------------------------------------------------

#: La rejilla que 116-C0 midió DENTRO del régimen de CPU (las de 4.000/
#: 8.000 filas están fuera del régimen por sí solas: no hace falta
#: estimarlas). 14 columnas es la de `adult`; con este redondeo hacia
#: arriba un conjunto de, por ejemplo, 20 columnas usa el punto de 32.
_BUCKETS_DE_COLUMNAS = (8, 14, 32)
_BUCKETS_DE_FILAS = (500, 1000, 2000)


def _cargar_tabla_de_c0(ruta: Path = RUTA_DE_C0) -> dict[tuple[int, int], dict]:
    """`(columnas, filas) -> {ms_por_fila_todas, segundos_ajuste,
    segundos_import}`, el MÁXIMO entre las tres fuentes que C0 midió en ese
    punto (`optdigits`/`adult`/`sintetico`) — un máximo, no un promedio: es
    una COTA para `--estimar`, igual que el resto de la aritmética de peor
    caso de Fase 0 (`plan_de_la_pasada` de C5/C6)."""
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    tabla: dict[tuple[int, int], dict] = {}
    for r in payload["registros"]:
        if r.get("estado") != "medido":
            continue
        clave = (r["columnas"], r["filas"])
        ms_todas = r["ms_por_fila"]["todas"]
        previo = tabla.get(clave)
        if previo is None:
            tabla[clave] = {"ms_por_fila_todas": ms_todas,
                            "segundos_ajuste": r["segundos_ajuste"],
                            "segundos_import": r["segundos_import"]}
        else:
            previo["ms_por_fila_todas"] = max(previo["ms_por_fila_todas"], ms_todas)
            previo["segundos_ajuste"] = max(previo["segundos_ajuste"], r["segundos_ajuste"])
            previo["segundos_import"] = max(previo["segundos_import"], r["segundos_import"])
    return tabla


def _bucket_hacia_arriba(valor: int, buckets: tuple[int, ...]) -> int | None:
    """El bucket más pequeño que sea `>= valor` — REDONDEA HACIA ARRIBA,
    nunca hacia abajo: un techo que subestima el coste es el defecto que ya
    costó `WALL_SECONDS = 120.0` en C3. `None` si ni el mayor bucket
    alcanza (no debería pasar dentro del régimen; si pasa, se declara, no
    se adivina)."""
    candidatos = [b for b in buckets if b >= valor]
    return min(candidatos) if candidatos else None


def estimar_coste_del_pliegue(n_filas_de_contexto: int, n_columnas: int, n_test: int,
                              tabla: dict[tuple[int, int], dict]) -> dict:
    """El coste ESTIMADO de un pliegue en régimen, con la tabla de C0: el
    punto de la rejilla que redondea hacia arriba en filas Y en columnas, y
    `ms_por_fila["todas"]` — el lote con el que `MotorTabICL._predecir()`
    predice de verdad (TODO el pliegue de prueba en una sola llamada)."""
    columnas_bucket = _bucket_hacia_arriba(n_columnas, _BUCKETS_DE_COLUMNAS)
    filas_bucket = _bucket_hacia_arriba(n_filas_de_contexto, _BUCKETS_DE_FILAS)
    if columnas_bucket is None or filas_bucket is None:
        return {"estimable": False,
                "motivo": (f"({n_filas_de_contexto} filas, {n_columnas} columnas) no cabe "
                          f"en la rejilla medida por C0 (hasta {_BUCKETS_DE_FILAS[-1]} filas "
                          f"x {_BUCKETS_DE_COLUMNAS[-1]} columnas)")}
    punto = tabla.get((columnas_bucket, filas_bucket))
    if punto is None:
        return {"estimable": False,
                "motivo": f"C0 no midio el punto (columnas={columnas_bucket}, "
                         f"filas={filas_bucket})"}
    segundos_predecir = punto["ms_por_fila_todas"] * n_test / 1000.0
    total = punto["segundos_import"] + punto["segundos_ajuste"] + segundos_predecir
    return {"estimable": True, "segundos": total, "bucket_columnas": columnas_bucket,
            "bucket_filas": filas_bucket, "segundos_predecir": round(segundos_predecir, 2),
            "segundos_ajuste": punto["segundos_ajuste"], "segundos_import": punto["segundos_import"]}


def plan_de_la_pasada(regimenes: list[dict], datasets_por_nombre: dict,
                      tabla_c0: dict[tuple[int, int], dict]) -> dict:
    """El coste ESTIMADO (no un tope de presupuesto agotado, ver el
    docstring del módulo) de correr `tabicl.v2` sobre los conjuntos del
    régimen, con la aritmética de C0. Los conjuntos fuera de régimen se
    declaran aparte con coste despreciable: el técho se comprueba ANTES de
    leer pesos o importar `tabicl` (`motores/tabicl.py`, `_ajustar`, paso
    1), así que un pliegue rechazado cuesta lo que tarda en arrancar el
    subproceso hijo, no una predicción de verdad."""
    en_regimen = [r for r in regimenes if r["en_regimen"]]
    fuera = [r for r in regimenes if not r["en_regimen"]]
    por_dataset: list[dict] = []
    total_s = 0.0
    sin_estimar: list[dict] = []
    for r in en_regimen:
        ds = datasets_por_nombre[r["dataset"]]
        estimacion = estimar_coste_del_pliegue(
            r["max_filas_de_contexto"], r["n_columnas"], r["n_test"], tabla_c0)
        if not estimacion["estimable"]:
            sin_estimar.append({"dataset": r["dataset"], "motivo": estimacion["motivo"]})
            continue
        segundos_del_conjunto = estimacion["segundos"] * r["n_pliegues"]
        total_s += segundos_del_conjunto
        por_dataset.append({
            "dataset": r["dataset"], "cubo": ds.cubo, "n_pliegues": r["n_pliegues"],
            "n_test": r["n_test"], "n_columnas": r["n_columnas"],
            "max_filas_de_contexto": r["max_filas_de_contexto"],
            "segundos_por_pliegue_estimados": round(estimacion["segundos"], 2),
            "segundos_del_conjunto_estimados": round(segundos_del_conjunto, 1)})
    return {
        "n_en_regimen": len(en_regimen), "n_fuera_de_regimen": len(fuera),
        "datasets_fuera_de_regimen": [r["dataset"] for r in fuera],
        "por_dataset": por_dataset, "sin_estimar": sin_estimar,
        "segundos_totales_estimados": round(total_s, 1),
        "horas_totales_estimadas": round(total_s / 3600.0, 2),
        "que_es_la_estimacion": (
            "aritmetica con la tabla medida en 116-C0 (ms/fila del lote 'todas', el que "
            "MotorTabICL._predecir()/_probabilidades() usan de verdad: TODO el pliegue de "
            "prueba en una sola llamada), redondeando cada pliegue al punto de la rejilla "
            "que sea >= sus filas de contexto y sus columnas, y con el MAXIMO entre las "
            "tres fuentes que C0 midio en ese punto (optdigits/adult/sintetico). No es un "
            "tope de presupuesto ni una medicion real: los pliegues fuera de regimen se "
            "rechazan antes de leer pesos o importar tabicl, coste despreciable y no "
            "incluido en el total."),
    }


def _imprimir_estimacion(plan: dict) -> None:
    print(f"en regimen: {plan['n_en_regimen']}  fuera de regimen: {plan['n_fuera_de_regimen']}")
    if plan["datasets_fuera_de_regimen"]:
        print(f"  fuera: {plan['datasets_fuera_de_regimen']}")
    for e in sorted(plan["por_dataset"], key=lambda e: e["dataset"]):
        print(f"  {e['dataset']:35} cubo={e['cubo']:8} pliegues={e['n_pliegues']:3} "
              f"n_test={e['n_test']:5} columnas={e['n_columnas']:3} "
              f"max_filas_ctx={e['max_filas_de_contexto']:5} -> "
              f"{e['segundos_por_pliegue_estimados']:6.1f}s/pliegue, "
              f"{e['segundos_del_conjunto_estimados']:7.1f}s el conjunto")
    if plan["sin_estimar"]:
        print("  SIN ESTIMAR (fuera de la rejilla medida por C0):")
        for s in plan["sin_estimar"]:
            print(f"    {s['dataset']}: {s['motivo']}")
    print(f"\n  TOTAL ESTIMADO: {plan['segundos_totales_estimados']:.1f}s "
          f"({plan['horas_totales_estimadas']:.2f} h)")
    print(f"\n  {plan['que_es_la_estimacion']}")
    print("\n  Esto es la aritmetica de C0 aplicada al regimen de HOY, sin medir nada. El "
          "coste real se mide en la cola nocturna (~/encolar.sh) o con "
          "correr_116c2_en_contenedor.sh, nunca a mano un ratito.")


# ---------------------------------------------------------------------------
# 7. LA PASADA
# ---------------------------------------------------------------------------

_DIR_ENGINES = c3._DIR_ENGINES
RUTA_DEL_MOTOR_TABICL = _DIR_ENGINES / "motores" / "tabicl.py"

#: FICHEROS COMPARTIDOS: los de C5 (que ya incluyen los de C3) más este
#: guion y `registro_de_motores.py` (resuelve `tabicl.v2`) — un cambio ahí
#: invalida TODA la caché de intentos. El propio `motores/tabicl.py` NO va
#: aquí: va en `RUTA_DEL_MOTOR_TABICL`, como el `motor_digest` de
#: `_reusable` (un solo motor en esta pasada, pero el mismo esquema de dos
#: digests que C3/C5/C6 usan para los suyos).
_FICHEROS_COMPARTIDOS = c5._FICHEROS_COMPARTIDOS + (
    Path(__file__).resolve(), _DIR_ENGINES / "registro_de_motores.py")


def _digest_entorno() -> str:
    return hashlib.sha256(
        "".join(c3._digest_fichero(f) for f in _FICHEROS_COMPARTIDOS).encode()
    ).hexdigest()[:16]


def _exigir_que_quepa() -> None:
    """El mismo guardia de reserva de CPU que C3/C5/C6: comprueba que
    `PROCESOS_A_LA_VEZ x HILOS_POR_INTENTO` (los MISMOS de C3) caben en esta
    máquina antes de medir nada."""
    caben = protocolo_mod.reserva_segura(c3.HILOS_POR_INTENTO)
    if c3.PROCESOS_A_LA_VEZ > caben:
        raise SystemExit(
            f"esta pasada pediria {c3.PROCESOS_A_LA_VEZ} procesos x "
            f"{c3.HILOS_POR_INTENTO} hilos y en esta maquina "
            f"({protocolo_mod.cpus_disponibles()} CPUs) caben {caben}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta todos los intentos")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON de salida (por omision, el de este directorio)")
    parser.add_argument("--estimar", action="store_true",
                        help="imprime la cuenta de coste con la aritmetica de 116-C0 y NO "
                             "mide nada")
    parser.add_argument("--solo", default=None,
                        help="nombres de conjunto separados por coma: corre SOLO esos. Para "
                             "probar el guion, nunca para medir el corte de verdad")
    args = parser.parse_args(argv)

    preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    datasets = datasets_no_sellados(protocolo)
    subconjunto = None
    if args.solo:
        pedidos = [n.strip() for n in args.solo.split(",") if n.strip()]
        por_nombre_ds = {d.nombre: d for d in todos}
        desconocidos = [n for n in pedidos if n not in por_nombre_ds]
        if desconocidos:
            raise SystemExit(f"--solo nombra conjuntos que no estan en el protocolo v2: "
                             f"{desconocidos}")
        datasets = [por_nombre_ds[n] for n in pedidos]
        subconjunto = pedidos
        sellados_pedidos = [n for n in pedidos if por_nombre_ds[n].sellado]
        if sellados_pedidos:
            print(f"AVISO: --solo incluye conjunto(s) SELLADO(S) {sellados_pedidos} -- esto "
                  f"es una prueba del guion, NO CUENTA como medicion de sellados", flush=True)

    _exigir_que_quepa()
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]}")

    if args.estimar:
        # LA VÍA BARATA: régimen ESTIMADO con la partición que la v2 YA MIDIÓ
        # (`particion_por_dataset`), NUNCA `regimen_del_dataset` — ese lee y
        # particiona el ARFF de cada conjunto, y para los 32 no sellados
        # enteros (incluidos los del cubo grande, hasta 188.318 filas) eso se
        # midió en más de 35 minutos de CPU sin terminar (ver el docstring de
        # `regimen_estimado_desde_metadatos`). `--estimar` es aritmética de
        # segundos, no una medición.
        print(f"estimando el regimen de {len(datasets)} conjunto(s) con la particion YA "
              f"MEDIDA en la v2 ({RUTA_DE_LA_V2.name}) -- sin leer ni particionar ningun "
              f"ARFF...", flush=True)
        metadatos_v2 = cargar_particion_por_dataset_v2()
        faltan_metadatos = [d.nombre for d in datasets if d.nombre not in metadatos_v2]
        if faltan_metadatos:
            raise SystemExit(
                f"--estimar necesita la particion de {RUTA_DE_LA_V2.name} para estos "
                f"conjuntos y no la tiene: {faltan_metadatos}. No se estima con un metadato "
                f"inventado -- correr la pasada real (sin --estimar) para ellos, o "
                f"comprobar el artefacto v2.")
        regimenes = [regimen_estimado_desde_metadatos(d.nombre, metadatos_v2[d.nombre])
                    for d in datasets]
        datasets_por_nombre = {d.nombre: d for d in datasets}
        for regimen in regimenes:
            estado = "DENTRO" if regimen["en_regimen"] else "FUERA "
            print(f"  {estado} {regimen['dataset']}: columnas={regimen['n_columnas']} "
                  f"max_filas_de_contexto(estimado)={regimen['max_filas_de_contexto']}",
                  flush=True)
        en_reg = sum(1 for r in regimenes if r["en_regimen"])
        print(f"\nregimen ESTIMADO: {en_reg} dentro, {len(regimenes) - en_reg} fuera "
              f"(la pasada real lo recalcula EXACTO, pliegue a pliegue, sobre los datos "
              f"leidos de verdad)", flush=True)
        tabla_c0 = _cargar_tabla_de_c0()
        plan = plan_de_la_pasada(regimenes, datasets_por_nombre, tabla_c0)
        _imprimir_estimacion(plan)
        return

    # A PARTIR DE AQUÍ, LA PASADA REAL: régimen EXACTO, leyendo y
    # particionando los ARFF de verdad (`particiones_base`, pliegue a
    # pliegue) — nunca la vía barata de arriba, que es solo para `--estimar`.
    print(f"calculando el regimen de {len(datasets)} conjunto(s) (sin tabicl, por regla)...",
          flush=True)
    regimenes: list[dict] = []
    bases_por_dataset: dict[str, tuple] = {}
    datasets_por_nombre: dict[str, object] = {}
    for ds in datasets:
        regimen, base = regimen_del_dataset(ds, protocolo)
        regimenes.append(regimen)
        bases_por_dataset[ds.nombre] = base
        datasets_por_nombre[ds.nombre] = ds
        estado = "DENTRO" if regimen["en_regimen"] else "FUERA "
        print(f"  {estado} {ds.nombre}: columnas={regimen['n_columnas']} "
              f"max_filas_de_contexto={regimen['max_filas_de_contexto']}", flush=True)

    en_regimen = [r for r in regimenes if r["en_regimen"]]
    fuera_de_regimen = [r for r in regimenes if not r["en_regimen"]]
    print(f"\nregimen: {len(en_regimen)} dentro, {len(fuera_de_regimen)} fuera", flush=True)

    motor = motor_para(NOMBRE_TABICL)
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    for r in fuera_de_regimen:
        print(f"FUERA DE REGIMEN: {r['dataset']}: {motivo_fuera_de_regimen(r)}", flush=True)

    ruta_salida = Path(args.salida) if args.salida else RUTA_DE_SALIDA
    cache_previo, payload_previo = ({}, {}) if args.forzar else c3._cargar_cache(ruta_salida)
    entorno_digest = _digest_entorno()
    motor_digest = c3._digest_fichero(RUTA_DEL_MOTOR_TABICL)

    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor": motor_digest},
        datos_de_entrada={d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = c3.procedencia_declarada(payload_previo)
        print(f"cache previo: {len(cache_previo)} registros, procedencia "
              f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)

    resultados_v2 = cargar_resultados_v2()
    resultados: list[dict] = []
    clasificaciones: list[dict] = []
    reusados = 0
    inicio = time.perf_counter()
    regla = protocolo.regla_de_cierre
    nombres_del_regimen = [r["dataset"] for r in en_regimen]

    def guardar(parcial: bool) -> dict:
        medibles = [c for c in clasificaciones if c["medible"]]
        veredicto = None if parcial else veredicto_del_corte(medibles)
        regla_cartera = None if parcial else regla_de_la_cartera_con_tabicl(
            resultados, resultados_v2, regla=regla, metrica_por_dataset=metrica_por_dataset,
            nombres_del_regimen=nombres_del_regimen)
        return _componer_y_guardar(
            resultados, clasificaciones, veredicto, regla_cartera, procedencia, payload_previo,
            ruta_salida, datasets=datasets, regimenes=regimenes, protocolo=protocolo,
            metrica_por_dataset=metrica_por_dataset, subconjunto=subconjunto,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados, parcial=parcial)

    for regimen in en_regimen:
        ds = datasets_por_nombre[regimen["dataset"]]
        por_id, propuesta, spec, objetivo, predictores, declarada = bases_por_dataset[ds.nombre]
        test_ids = propuesta.plan.observaciones_del_rol("test")
        wall_seconds = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        metric_id = metrica_por_dataset[ds.nombre]
        print(f"\n=== {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"columnas={regimen['n_columnas']}, max_filas_ctx="
              f"{regimen['max_filas_de_contexto']}, test={len(test_ids)}, "
              f"tope={wall_seconds:.0f}s, metrica={metric_id}) ===", flush=True)

        registros_del_dataset: list[dict] = []
        for repeticion in range(repeticiones):
            for pliegue_i in range(protocolo.particion.folds):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None:
                    continue
                clave = (ds.nombre, NOMBRE_TABICL, repeticion, pliegue_i)
                previo = cache_previo.get(clave)
                if c3._reusable(previo, entorno_digest, motor_digest, wall_seconds):
                    registro = dict(previo, reusado=True)
                    registro.setdefault("procedencia_id", None)
                    resultados.append(registro)
                    registros_del_dataset.append(registro)
                    reusados += 1
                    print(f"  [reusado] {ds.nombre} rep={repeticion} pliegue={pliegue_i}: "
                          f"estado={registro.get('estado')}", flush=True)
                    continue

                crudas_train = [por_id[i] for i in pliegue.entrena]
                crudas_val = [por_id[i] for i in pliegue.valida]
                crudas_test = [por_id[i] for i in test_ids]
                transformadas = c3.preparar_para_motor(
                    crudas_train, crudas_train + crudas_val + crudas_test, objetivo,
                    predictores, motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)
                hacer = lambda xs: Particion.desde_filas(  # noqa: E731
                    xs, row_id_field="row_id", target_field=objetivo)

                presupuesto = Presupuesto(
                    wall_seconds=wall_seconds, hilos=c3.HILOS_POR_INTENTO,
                    seed=protocolo.particion.semillas[repeticion])
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, hacer(transformadas[:n_tr]),
                    hacer(transformadas[n_tr:n_tr + n_va]),
                    hacer(transformadas[n_tr + n_va:]),
                    spec, presupuesto,
                    candidate=f"{motor.nombre}-{c3.CONFIGURACION_UNICA}",
                    split_plan_digest=propuesta.plan.digest(),
                    dataset=ds.nombre, pliegue=pliegue_i, repeticion=repeticion)
                transcurrido = time.perf_counter() - t0

                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                    "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": motor.nombre, "repeticion": repeticion,
                    "pliegue": pliegue_i, "estado": intento.estado,
                    "semilla": protocolo.particion.semillas[repeticion],
                    "presupuesto_wall_s": wall_seconds,
                    "configuracion": c3.CONFIGURACION_UNICA,
                    "metrica_de_cierre": metric_id,
                    "wall_s": round(transcurrido, 3),
                    "metricas": metricas,
                    "tiempo_de_ajuste": recursos.get("wall_seconds"),
                    "cpu_segundos": recursos.get("cpu_seconds"),
                    "motivo": (intento.motivo_del_estado["es"]
                               if intento.motivo_del_estado else None),
                    "entorno_digest": entorno_digest,
                    "motor_digest": motor_digest,
                    "procedencia_id": procedencia["procedencia_id"],
                    "reusado": False,
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                resultados.append(registro)
                registros_del_dataset.append(registro)
                valor_de_cierre = registro.get(metric_id)
                print(f"  {ds.nombre} rep={repeticion} pliegue={pliegue_i}: "
                      f"estado={intento.estado} {metric_id}={valor_de_cierre} "
                      f"wall={transcurrido:.1f}s", flush=True)
            guardar(parcial=True)

        clasificacion = clasificar_conjunto(
            ds.nombre, regimen, registros_del_dataset, resultados_v2, metric_id)
        clasificaciones.append(clasificacion)
        v = clasificacion["veredicto"]
        print(f"  CLASIFICACION {ds.nombre}: {clasificacion['categoria']} "
              f"mejora={v['mejora'] if v else None} motivo="
              f"{(v or clasificacion)['motivo']}", flush=True)
        guardar(parcial=True)

    for r in fuera_de_regimen:
        clasificaciones.append({"dataset": r["dataset"], "categoria": "fuera_de_regimen",
                                "medible": False, "motivo": motivo_fuera_de_regimen(r),
                                "veredicto": None})

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados} reusados, {len(resultados)-reusados} ejecutados) ===")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    v = salida["veredicto"]
    if v is not None:
        print(f"VEREDICTO DEL CORTE: {v['n_mejora']}/{v['n_conjuntos']} = "
              f"{v['fraccion_mejora']:.3f} ({'CUMPLE' if v['cumple'] else 'NO CUMPLE'} "
              f"el minimo {v['minimo_exigido']})")


def _componer_y_guardar(resultados, clasificaciones, veredicto, regla_cartera, procedencia,
                        payload_previo, ruta_salida, *, datasets, regimenes, protocolo,
                        metrica_por_dataset, subconjunto, total_wall_s, reusados,
                        parcial) -> dict:
    """Compone el JSON y lo escribe, ATÓMICAMENTE — mismo patrón que
    C3/C5/C6 (`_procedencias_citadas`, `sellar_la_salida`, escritura a
    temporal + `replace`), y con lo medido tras CADA conjunto: si se corta,
    queda lo hecho ('resultado tras CADA conjunto', del encargo)."""
    procedencias, sin_procedencia = c3._procedencias_citadas(
        resultados, procedencia, (payload_previo.get("procedencias") or {}))

    medibles = [c for c in clasificaciones if c["medible"]]
    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "116-C2",
        "registro_del_corte": (
            "documentacion/116_FUNDACIONAL_TABULAR_CONTRACT.md, bloque 'REGISTRO DE C2, "
            "escrito el 2026-09-25 ANTES de construir su pasada'"),
        "candidato": NOMBRE_TABICL, "contra": NOMBRE_LIGHTGBM,
        "contra_declarado": ("pasada_v2_113_resultado.json (113-C4), SIN remedir -- misma "
                             "particion, semillas y presupuesto que esta pasada"),
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial, "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "tope_de_cpu": {"max_filas_de_contexto": MAX_FILAS_DE_CONTEXTO_EN_CPU,
                        "max_columnas": MAX_COLUMNAS_EN_CPU},
        "regimen_por_dataset": regimenes,
        "criterio_del_regimen": (
            "el motor acepta en TODOS los pliegues del conjunto (ningun pliegue rechazado "
            "por el techo de CPU); un conjunto con un solo pliegue rechazado queda FUERA "
            "del regimen entero. Es una regla calculada de la particion (filas de "
            "entrenamiento del pliegue, numero de predictores), no una medicion corriendo "
            "el motor: los rechazos por techo no son fallos."),
        "criterio_de_la_comparacion": (
            "lightgbm de la pasada v2 (113-C4, pasada_v2_113_resultado.json), SIN remedir; "
            "si le falta algun pliegue que tabicl.v2 SI completo en un conjunto del "
            "regimen, ese conjunto queda fuera de la comparacion (60% y veredicto por "
            "conjunto), aunque sigue contando para la regla de la cartera."),
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "margen_de_equivalencia": MARGEN_DE_EQUIVALENCIA,
        "minimo_de_mejora_exigido": MINIMO_DE_MEJORA,
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados), "n_reusados": reusados,
        "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
        "resultados": resultados,
        "clasificacion_por_conjunto": clasificaciones,
        "n_en_regimen": sum(1 for r in regimenes if r["en_regimen"]),
        "n_fuera_de_regimen": sum(1 for r in regimenes if not r["en_regimen"]),
        "n_medibles": len(medibles),
        "n_sin_lightgbm_emparejable": sum(1 for c in clasificaciones
                                          if c["categoria"] == "sin_lightgbm_emparejable"),
        "veredicto": veredicto,
        "regla_de_la_cartera_con_tabicl": regla_cartera,
    }
    c3.sellar_la_salida(salida)
    temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta_salida)
    return salida


if __name__ == "__main__":
    main()
