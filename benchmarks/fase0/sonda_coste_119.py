# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C1 — la sonda de COSTE de TabM+PLR, en CPU.

Mide SOLO tiempo (segundos por época, a 5 hilos) y parámetros, NUNCA una
métrica de acierto — `c1_plan.objetivo` del protocolo v4
(`protocolo_119_v4.json`): "medir SOLO coste [...] para fijar k y d_block del
motor nuevo antes de construirlo en C2". La red que mide es
`matrixai_engines.redes.tabm_plr.RedTabMPLR` (construida y citada en
119-C1, sección "QUÉ HACER" 1 del encargo).

**Qué mide, sobre qué**: los 3 conjuntos de `c1_plan.conjuntos` (los más
caros para la densa de hoy, según el contrato 119) — SOLO su partición de
TRAIN de la primera repetición y el primer pliegue (`repeticion=0,
pliegue=0`), construida con `matrixai.training.particion_por_diseno.
proponer_particion`, el MISMO mecanismo que usan las pasadas de Fase 0
(`pasada_amplia_101_c5.py`). Sin validación ni test: un sondeo de coste no
evalúa nada, así que no hace falta repartir para eso.

**Preparación** (protocolo v4, `arquitectura`, sin mirar los 40 conjuntos):
numéricas -> mediana de train para los faltantes, luego `QuantileTransformer
(output_distribution="normal")` ajustado con ruido gaussiano (media 0,
desviación 1e-5, semilla 0) SOLO para el ajuste, aplicado sin ruido; sin la
`n_quantiles = max(min(n_train // 30, 1000), 10)`. Categóricas -> un índice
por valor visto en train, más dos reservados (`INDICE_AUSENTE=0`,
`INDICE_DESCONOCIDO=1` — `tabm_plr.py`). Objetivo de regresión -> estandarizado
`(y - media_train) / desviacion_train`.

**La rejilla** (`c1_plan.rejilla_de_coste`): `k` en `{8, 16, 32}`, `d_block`
en `{256, 512}`, `n_blocks` FIJO en 2 (el valor por omisión de `TabM.make()`
con `num_embeddings` presente — no es un parámetro de la rejilla, así que no
se mide en distintos valores). 6 configuraciones x 3 conjuntos = 18 medidas
(`c1_plan.rejilla_de_coste.total_medidas`).

**Por cada medida**: 1 época de calentamiento (NO cuenta) + 3 épocas
cronometradas, AdamW (`receta_entrenamiento`: lr=0,002, weight_decay=0,0003,
recorte de gradiente a norma 1,0), lote `min(256, filas de train)`,
`torch.set_num_threads(5)`.

**La regla escrita ANTES de medir** (`c1_plan.regla_escrita_antes_de_medir`,
copiada aquí igual — no se decide mirando los números): se prueba primero la
config por omisión de la fuente (k=32, d_block=512); si en los 3 conjuntos
caben >= 17 épocas (`N_epocas_minimas` = paciencia 16 + 1) dentro de 0,75 x
600 s = 450 s (el presupuesto del cubo "grande"), esa es la elegida. Si no,
se baja k (32 -> 16 -> 8, MISMO d_block=512) y solo cuando los tres valores de
k a d_block=512 fallan se baja también la anchura (d_block 512 -> 256).

  **Nota de interpretación, porque el protocolo no lo hace 100% explícito**:
  "se baja k primero [...] y SOLO DESPUÉS la anchura" se lee aquí como la
  escalera COMPLETA de 6 peldaños, en este orden de preferencia (el primero
  que quepa en los 3 conjuntos a la vez es el elegido):
  (32,512) -> (16,512) -> (8,512) -> (32,256) -> (16,256) -> (8,256).
  Los 6 puntos se miden de todos modos (`total_medidas=18`), así que la regla
  solo decide en qué ORDEN se leen los resultados ya medidos — no cambia qué
  se mide. Si esta lectura no es la que Roberto quiso, está declarada aquí y
  en el JSON de salida (`regla_de_eleccion.orden_de_preferencia`) para poder
  corregirla sin repetir la medición.

**RSS pico**: `resource.getrusage(RUSAGE_SELF).ru_maxrss` es el máximo
ACUMULADO del PROCESO entero desde que arrancó, no un máximo aislado por
configuración (mismo problema que documenta `pasada_amplia_101_c5.py` en su
diccionario `donde_vive_cada_medida_siempre["rss_pico_mb"]`) — aquí se agrava
porque las 18 medidas comparten el mismo proceso. El campo se llama
`rss_pico_mb_acumulado_proceso` a propósito, para no leerse como el pico de
esa configuración sola.

Uso:

    cd /home/deployer/matrixAI && python3 benchmarks/fase0/sonda_coste_119.py
    cd /home/deployer/matrixAI && python3 benchmarks/fase0/sonda_coste_119.py --humo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import QuantileTransformer  # noqa: E402

import pasada_exploratoria_101_c3 as c3  # noqa: E402
from pasada_amplia_101_c5 import clase_positiva_medida  # noqa: E402

from matrixai.training.particion_por_diseno import proponer_particion  # noqa: E402
from matrixai.training.preparacion import tipar_columnas_numericas  # noqa: E402

from matrixai_engines.redes.tabm_plr import (  # noqa: E402
    INDICE_AUSENTE,
    INDICES_RESERVADOS,
    RedTabMPLR,
    contar_parametros,
    perdida_media_de_las_k_cabezas,
)

RUTA_DEL_PROTOCOLO = _AQUI / "protocolo_119_v4.json"
RUTA_DE_SALIDA = _AQUI / "resultado_sonda_coste_119.json"
RUTA_DE_SALIDA_HUMO = _AQUI / "resultado_sonda_coste_119_humo.json"
RUTA_DEL_MODULO_RED = (_AQUI.parent.parent.parent / "matrixai-engines" / "src"
                       / "matrixai_engines" / "redes" / "tabm_plr.py")

#: Igual que `ESTRATIFICACION` de `pasada_amplia_101_c5.py` -- extraído, no
#: reinventado, pero copiado literal porque C1 no importa ese módulo entero
#: por su nombre público (solo `clase_positiva_medida`). Misma tabla, mismo
#: motivo: con un objetivo continuo, estratificar rompe los pliegues.
ESTRATIFICACION = {
    "binary_classification": True,
    "multiclass_classification": True,
    "regression": False,
}

#: `n_blocks` de la rejilla (`c1_plan.rejilla_de_coste.n_blocks`): "fijo en 2
#: (el valor por omision de TabM.make() con num_embeddings presente)" -- un
#: texto descriptivo en el protocolo, no un entero; se fija aquí igual a la
#: cita literal de `arquitectura.backbone_ensamblado.parametros_por_omision_
#: de_TabM_make.n_blocks`.
N_BLOCKS_FIJO = 2

#: La escalera de degradación completa (ver nota de interpretación arriba).
ORDEN_DE_PREFERENCIA = [(32, 512), (16, 512), (8, 512), (32, 256), (16, 256), (8, 256)]


def _sha256_de(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _cargar_protocolo() -> dict[str, Any]:
    return json.loads(RUTA_DEL_PROTOCOLO.read_text(encoding="utf-8"))


def _dataset_del_protocolo(protocolo: dict, nombre: str) -> dict:
    for d in protocolo["datasets"]:
        if d["nombre"] == nombre:
            return d
    raise SystemExit(f"{nombre!r} no está en protocolo_119_v4.json[datasets]")


# ---------------------------------------------------------------------------
# Preparación: carga, partición, y la receta del protocolo v4
# ---------------------------------------------------------------------------
def _separar_numericas_categoricas(filas_train: list[dict], predictores: tuple[str, ...]
                                   ) -> tuple[list[str], list[str]]:
    """Sobre las filas YA TIPADAS (`tipar_columnas_numericas`): una columna es
    numérica si TODOS sus valores no nulos en TRAIN son `int`/`float`."""
    numericas, categoricas = [], []
    for columna in predictores:
        valores = [f.get(columna) for f in filas_train if f.get(columna) is not None]
        if valores and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valores):
            numericas.append(columna)
        else:
            categoricas.append(columna)
    return numericas, categoricas


def _preparar_numericas(filas_train: list[dict], numericas: list[str]) -> np.ndarray | None:
    """Mediana de train para los faltantes, luego QuantileTransformer con
    ruido SOLO para ajustar -- protocolo v4, `arquitectura.preprocesado_
    numericas`, cita literal de `example.ipynb` (repo `tabm`)."""
    if not numericas:
        return None
    matriz = np.array([[float(f[c]) if f.get(c) is not None else np.nan for c in numericas]
                       for f in filas_train], dtype=np.float64)
    for j in range(matriz.shape[1]):
        columna = matriz[:, j]
        faltan = np.isnan(columna)
        if faltan.any():
            mediana = np.nanmedian(columna)
            columna[faltan] = mediana
            matriz[:, j] = columna
    n_quantiles = max(min(len(filas_train) // 30, 1000), 10)
    ruido = np.random.default_rng(0).normal(0.0, 1e-5, matriz.shape).astype(matriz.dtype)
    transformador = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal",
                                        subsample=10**9).fit(matriz + ruido)
    return transformador.transform(matriz).astype(np.float32)


def _preparar_categoricas(filas_train: list[dict], categoricas: list[str]
                          ) -> tuple[np.ndarray | None, list[int]]:
    """Índices con `INDICE_AUSENTE`/`INDICE_DESCONOCIDO` reservados --
    protocolo v4, `arquitectura.embeddings_categoricas.decision_del_119` y
    `.faltantes_decision_del_119`. Dentro de TRAIN nunca sale «desconocido»
    (por definición: el vocabulario se construye DESDE train); el índice
    queda reservado para cuando C2 prepare validación/test.

    Una columna sin NINGÚN valor no nulo en train (medido en `--humo`: con
    solo 300 filas, una columna rara de KDDCup09 puede caer entera en
    `None`) da un vocabulario vacío; se le deja hueco para UNA categoría real
    igualmente (`max(vocabulario, 1)`) para que `EmbeddingsCategoricas` no
    la rechace por cardinalidad — esa hilera del embedding simplemente no se
    usa en esta medida de coste."""
    if not categoricas:
        return None, []
    columnas_idx = []
    cardinalidades = []
    for columna in categoricas:
        valores_train = sorted({str(f[columna]) for f in filas_train if f.get(columna) is not None})
        vocabulario = {v: i + INDICES_RESERVADOS for i, v in enumerate(valores_train)}
        cardinalidades.append(max(len(vocabulario), 1) + INDICES_RESERVADOS)
        idx = [INDICE_AUSENTE if f.get(columna) is None else vocabulario[str(f[columna])]
              for f in filas_train]
        columnas_idx.append(idx)
    return np.array(columnas_idx, dtype=np.int64).T, cardinalidades


def _preparar_objetivo(filas_train: list[dict], objetivo: str, tarea: str
                       ) -> tuple[np.ndarray, int]:
    """Regresión: estandarizado (protocolo v4, `arquitectura.objetivo_
    regresion`). Clasificación: sin transformación de valor, solo mapeo a
    índices -- la sonda de coste no mide acierto, así que la convención de
    «positiva» solo importa para que la BCE tenga una clase de referencia
    estable (se reutiliza `clase_positiva_medida` de `pasada_amplia_101_c5.py`,
    no se reinventa)."""
    if tarea == "regression":
        y = np.array([float(f[objetivo]) for f in filas_train], dtype=np.float64)
        media, desviacion = float(y.mean()), float(y.std())
        desviacion = desviacion if desviacion > 0 else 1.0
        return ((y - media) / desviacion).astype(np.float32).reshape(-1, 1), 1
    if tarea == "binary_classification":
        clases, positiva = clase_positiva_medida(filas_train, objetivo)
        if positiva is None:
            raise SystemExit(f"train tiene una sola clase para {objetivo!r}: {clases}")
        y = np.array([1.0 if str(f[objetivo]) == positiva else 0.0 for f in filas_train],
                    dtype=np.float32).reshape(-1, 1)
        return y, 1
    # multiclass_classification
    nombres = sorted({str(f[objetivo]) for f in filas_train})
    mapa = {c: i for i, c in enumerate(nombres)}
    y = np.array([mapa[str(f[objetivo])] for f in filas_train], dtype=np.int64)
    return y, len(nombres)


def preparar_conjunto(nombre: str, protocolo: dict, *, humo: bool) -> dict[str, Any]:
    """Carga el ARFF, arma la partición del protocolo (o una muestra pequeña
    en `--humo`) y prepara train según la receta del protocolo v4."""
    entrada = _dataset_del_protocolo(protocolo, nombre)
    data_id = entrada["data_id"]
    objetivo_declarado = entrada["columna_objetivo"]

    c3.RUTA_DEL_PROTOCOLO = RUTA_DEL_PROTOCOLO
    filas, objetivo = c3.cargar_arff(data_id)
    assert objetivo == objetivo_declarado, (
        f"{nombre}: el lector tomó {objetivo!r} como objetivo, el catálogo declara "
        f"{objetivo_declarado!r} -- el catálogo del protocolo v4 no se está usando de verdad")
    filas_con_objetivo = [f for f in filas if f[objetivo] is not None]
    predictores = tuple(k for k in filas[0] if k not in ("row_id", objetivo))
    tipar_columnas_numericas(filas_con_objetivo, predictores)

    tarea = entrada["tarea"]
    if humo:
        muestra = list(filas_con_objetivo)
        random.Random(0).shuffle(muestra)
        filas_train = muestra[:300]
    else:
        folds = protocolo["particion"]["folds"]
        cubo = entrada["cubo_de_tamano"]
        repeticiones = {
            "pequeno": protocolo["particion"]["repeticiones_pequeno_mediano"],
            "mediano": protocolo["particion"]["repeticiones_pequeno_mediano"],
            "grande": protocolo["particion"]["repeticiones_grande"],
        }[cubo]
        propuesta = proponer_particion(
            filas_con_objetivo, plan_id=f"119c1-{nombre}", observation_id_field="row_id",
            split_type="iid", seed=0, test_fraction=0.2, folds=folds, repeats=repeticiones,
            objetivo=(objetivo if ESTRATIFICACION[tarea] else None))
        if not propuesta.es_viable:
            raise SystemExit(f"{nombre}: partición no viable, bloqueos="
                             f"{[b.clave for b in propuesta.bloqueos]}")
        pliegue = propuesta.pliegues.pliegue_de(0, 0)
        if pliegue is None:
            raise SystemExit(f"{nombre}: no hay pliegue (repeticion=0, pliegue=0)")
        por_id = {f["row_id"]: f for f in filas_con_objetivo}
        filas_train = [por_id[i] for i in pliegue.entrena]

    numericas, categoricas = _separar_numericas_categoricas(filas_train, predictores)
    x_num = _preparar_numericas(filas_train, numericas)
    x_cat, cardinalidades = _preparar_categoricas(filas_train, categoricas)
    y, d_out = _preparar_objetivo(filas_train, objetivo, tarea)

    return {
        "nombre": nombre, "data_id": data_id, "tarea": tarea, "cubo": entrada["cubo_de_tamano"],
        "n_filas_train": len(filas_train), "columnas_numericas": numericas,
        "columnas_categoricas": categoricas, "cardinalidades_categoricas": cardinalidades,
        "d_out": d_out, "x_num": x_num, "x_cat": x_cat, "y": y,
    }


# ---------------------------------------------------------------------------
# Medición de coste: una configuración, un conjunto
# ---------------------------------------------------------------------------
def _funcion_de_perdida(tarea: str):
    if tarea == "regression":
        return torch.nn.functional.mse_loss
    if tarea == "binary_classification":
        return torch.nn.BCEWithLogitsLoss()
    return torch.nn.CrossEntropyLoss()


def medir_configuracion(preparado: dict, *, k: int, d_block: int, n_blocks: int,
                        d_embedding: int, n_frequencies: int, frequency_init_scale: float,
                        batch_size: int, lr: float, weight_decay: float, clip_norm: float,
                        n_epocas_calentamiento: int, n_epocas_cronometradas: int,
                        semilla: int = 0) -> dict[str, Any]:
    """Construye la red, cuenta parámetros, entrena `n_epocas_calentamiento`
    (no cronometradas) + `n_epocas_cronometradas` sobre TODO train (sin
    validación: la sonda no mide acierto), y devuelve tiempos + parámetros +
    el RSS acumulado del proceso hasta este punto."""
    red = RedTabMPLR(
        n_num_features=len(preparado["columnas_numericas"]),
        cardinalidades_categoricas=preparado["cardinalidades_categoricas"],
        d_out=preparado["d_out"], k=k, d_block=d_block, n_blocks=n_blocks,
        d_embedding=d_embedding, n_frequencies=n_frequencies,
        frequency_init_scale=frequency_init_scale, semilla=semilla)
    n_parametros = contar_parametros(red)

    x_num = torch.from_numpy(preparado["x_num"]).float() if preparado["x_num"] is not None else None
    x_cat = torch.from_numpy(preparado["x_cat"]).long() if preparado["x_cat"] is not None else None
    y = torch.from_numpy(preparado["y"])
    y = y.long() if preparado["tarea"] == "multiclass_classification" else y.float()

    n = preparado["n_filas_train"]
    optimizador = torch.optim.AdamW(red.parameters(), lr=lr, weight_decay=weight_decay)
    funcion_perdida = _funcion_de_perdida(preparado["tarea"])
    generador = torch.Generator().manual_seed(semilla)

    def _una_epoca() -> None:
        red.train()
        permutacion = torch.randperm(n, generator=generador)
        for inicio in range(0, n, batch_size):
            idx = permutacion[inicio:inicio + batch_size]
            xn = x_num[idx] if x_num is not None else None
            xc = x_cat[idx] if x_cat is not None else None
            yb = y[idx]
            optimizador.zero_grad()
            logits = red(xn, xc)
            perdida = perdida_media_de_las_k_cabezas(logits, yb, funcion_perdida)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), clip_norm)
            optimizador.step()

    for _ in range(n_epocas_calentamiento):
        _una_epoca()

    tiempos = []
    for _ in range(n_epocas_cronometradas):
        t0 = time.perf_counter()
        _una_epoca()
        tiempos.append(time.perf_counter() - t0)

    rss_pico_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    media = sum(tiempos) / len(tiempos) if tiempos else float("nan")
    return {
        "k": k, "d_block": d_block, "n_blocks": n_blocks, "n_parametros": n_parametros,
        "segundos_por_epoca": tiempos, "segundos_por_epoca_media": media,
        "rss_pico_mb_acumulado_proceso": rss_pico_mb,
    }


# ---------------------------------------------------------------------------
# La regla escrita antes de medir
# ---------------------------------------------------------------------------
def elegir_configuracion(resultados: list[dict], conjuntos: list[str], *,
                         plazo_s: float, n_epocas_minimas: int) -> dict[str, Any]:
    """Aplica `c1_plan.regla_escrita_antes_de_medir` sobre la tabla YA
    medida: recorre `ORDEN_DE_PREFERENCIA` y devuelve la primera (k,d_block)
    que cabe (>= `n_epocas_minimas` en `plazo_s`) en LOS TRES conjuntos a la
    vez. Nunca mira ninguna métrica: la sonda no calcula ninguna."""
    por_config: dict[tuple[int, int], dict[str, dict]] = {}
    for r in resultados:
        clave = (r["k"], r["d_block"])
        por_config.setdefault(clave, {})[r["nombre"]] = r

    tabla_decision = []
    elegida = None
    for k, d_block in ORDEN_DE_PREFERENCIA:
        medidas = por_config.get((k, d_block), {})
        detalle = {}
        cabe_en_todos = True
        for nombre in conjuntos:
            r = medidas.get(nombre)
            if r is None:
                cabe_en_todos = False
                detalle[nombre] = {"medido": False}
                continue
            epocas_que_caben = int(plazo_s // r["segundos_por_epoca_media"])
            cumple = epocas_que_caben >= n_epocas_minimas
            detalle[nombre] = {"segundos_por_epoca_media": r["segundos_por_epoca_media"],
                               "epocas_que_caben_en_el_plazo": epocas_que_caben,
                               "cumple_N_epocas_minimas": cumple}
            cabe_en_todos = cabe_en_todos and cumple
        tabla_decision.append({"k": k, "d_block": d_block, "cabe_en_los_3_conjuntos": cabe_en_todos,
                               "detalle": detalle})
        if cabe_en_todos and elegida is None:
            elegida = {"k": k, "d_block": d_block, "n_blocks": N_BLOCKS_FIJO}

    return {
        "orden_de_preferencia": [{"k": k, "d_block": d} for k, d in ORDEN_DE_PREFERENCIA],
        "plazo_s": plazo_s, "N_epocas_minimas": n_epocas_minimas,
        "tabla_decision": tabla_decision,
        "configuracion_elegida": elegida,
        "nota": ("ninguna configuración de la rejilla cabe en el plazo en los 3 conjuntos a la vez "
                "-- fuera de lo que esta regla decide; consta aquí para que Roberto lo decida"
                if elegida is None else None),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humo", action="store_true",
                       help="300 filas, 1 época, un solo (k,d_block) por conjunto -- probar el camino")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.set_num_threads(5)
    inicio_total = time.perf_counter()

    protocolo = _cargar_protocolo()
    c1_plan = protocolo["c1_plan"]
    conjuntos = c1_plan["conjuntos"]
    receta = protocolo["receta_entrenamiento"]
    params_plr = protocolo["arquitectura"]["embeddings_numericas"]["parametros_por_omision_del_constructor"]
    d_embedding = params_plr["d_embedding"]
    n_frequencies = params_plr["n_frequencies"]
    frequency_init_scale = params_plr["frequency_init_scale"]

    presupuesto_grande_s = protocolo["presupuesto"]["minutos_por_cubo"]["grande"] * 60
    plazo_s = 0.75 * presupuesto_grande_s
    n_epocas_minimas = c1_plan["regla_escrita_antes_de_medir"]["N_epocas_minimas"]["valor"]

    if args.humo:
        configuraciones = [(32, 512)]
        n_epocas_calentamiento, n_epocas_cronometradas = 0, 1
    else:
        configuraciones = ORDEN_DE_PREFERENCIA
        n_epocas_calentamiento, n_epocas_cronometradas = 1, 3

    resultados: list[dict[str, Any]] = []
    preparaciones: dict[str, dict] = {}
    for nombre in conjuntos:
        print(f"preparando {nombre}...", flush=True)
        t0 = time.perf_counter()
        preparado = preparar_conjunto(nombre, protocolo, humo=args.humo)
        preparaciones[nombre] = preparado
        print(f"  {nombre}: train={preparado['n_filas_train']} filas, "
             f"{len(preparado['columnas_numericas'])} numéricas, "
             f"{len(preparado['columnas_categoricas'])} categóricas "
             f"({time.perf_counter() - t0:.1f} s)", flush=True)

        batch_size = min(receta["batch_size"]["valor_fuente"], preparado["n_filas_train"])
        for k, d_block in configuraciones:
            print(f"    k={k} d_block={d_block} n_blocks={N_BLOCKS_FIJO} batch={batch_size}...",
                 flush=True)
            medida = medir_configuracion(
                preparado, k=k, d_block=d_block, n_blocks=N_BLOCKS_FIJO,
                d_embedding=d_embedding, n_frequencies=n_frequencies,
                frequency_init_scale=frequency_init_scale, batch_size=batch_size,
                lr=receta["learning_rate"], weight_decay=receta["weight_decay"],
                clip_norm=receta["recorte_de_gradiente"]["norma_maxima"],
                n_epocas_calentamiento=n_epocas_calentamiento,
                n_epocas_cronometradas=n_epocas_cronometradas, semilla=0)
            medida["nombre"] = nombre
            medida["cubo"] = preparado["cubo"]
            resultados.append(medida)
            print(f"      {medida['segundos_por_epoca_media']:.3f} s/época media, "
                 f"{medida['n_parametros']:,} parámetros, "
                 f"rss_acum={medida['rss_pico_mb_acumulado_proceso']:.0f} MB", flush=True)

    salida: dict[str, Any] = {
        "corte": "119-C1", "humo": args.humo,
        "resultados": resultados,
    }
    if not args.humo:
        salida["regla_de_eleccion"] = elegir_configuracion(
            resultados, conjuntos, plazo_s=plazo_s, n_epocas_minimas=n_epocas_minimas)

    salida["procedencia"] = {
        "protocolo_sha256": _sha256_de(RUTA_DEL_PROTOCOLO),
        "protocolo_ruta": str(RUTA_DEL_PROTOCOLO),
        "sonda_sha256": _sha256_de(Path(__file__).resolve()),
        "modulo_red_sha256": (_sha256_de(RUTA_DEL_MODULO_RED)
                              if RUTA_DEL_MODULO_RED.exists() else None),
        "modulo_red_ruta": str(RUTA_DEL_MODULO_RED),
        "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "hilos_torch": torch.get_num_threads(),
        "duracion_total_s": time.perf_counter() - inicio_total,
    }

    ruta_salida = args.salida or (RUTA_DE_SALIDA_HUMO if args.humo else RUTA_DE_SALIDA)
    ruta_salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False, sort_keys=True),
                           encoding="utf-8")
    print(f"escrito: {ruta_salida} ({time.perf_counter() - inicio_total:.1f} s totales)", flush=True)


if __name__ == "__main__":
    main()
