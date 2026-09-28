# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — condición (4): «Jev como respondedor», la MISMA idea que la
condición (3) (`medir_c30_respondedor.py`, Qwen2.5-0.5B local) pero con
**Jev** (TypeSafe, `typesafe/jev-1.13-20260917`) por la Decisions API alfa de
OpenRouter (`benchmarks/texto_107c30/jev/respondedor_jev.py`, ver
`jev/LEEME.md` para el formato exacto y sus fuentes).

**Solo las tareas A (noticias falsas) y C (contratos PLACSP, regresión del
log10 del importe) — públicas y no clínicas.** La tarea B (casos clínicos)
está PROHIBIDA: este guion se niega con su motivo (`rj.
verificar_tarea_permitida`, comprobada AQUÍ y en el propio adaptador --
defensa en dos capas, no una sola que dependa de que la otra se acuerde).

**Reusa, no copia** (mismo patrón que (3) con `medir_c30.py`):
  - `medir_c30` (`mc`): TAREAS, particiones, LightGBM, condición (0) y (2),
    comparación con margen, digest de partición.
  - `medir_c30_respondedor` (`mr`): la plantilla LITERAL de pregunta
    (`PLANTILLA_PREGUNTA`), la selección de términos por chi² (tarea A,
    binaria -- `_terminos_top_chi2`), el tokenizador y el truncado igual
    que el embedding de C1 (`_cargar_tokenizador_de_c1`,
    `truncar_como_embedding_c1`), y su escritor atómico.
  - `jev.respondedor_jev` (`rj`): la clave, el adaptador HTTP, la caché por
    (row_id, término), el análisis sí/no/faltante.

**Tarea C es regresión: chi² no aplica sobre un objetivo continuo.** Dos
opciones, sin decidir por defecto -- la decide el registro sellado, no este
guion (`--seleccion-terminos-c`, OBLIGATORIA para la tarea C, sin valor por
omisión):
  (a) `chi2_quintiles`: discretiza el objetivo de TRAIN en quintiles
      (calculados SOLO con TRAIN) y aplica el MISMO chi² que (3)/tarea A
      sobre esas etiquetas discretas.
  (b) `f_regression`: `sklearn.feature_selection.f_regression` sobre el
      TF-IDF de TRAIN directamente contra el objetivo continuo, sin
      discretizar.

**El batching nativo de Jev abarata esto frente a (3)**: una fila con
`N_PREGUNTAS` términos manda **UNA** petición (todas sus preguntas
pendientes juntas), no `N_PREGUNTAS` peticiones -- ver `jev/LEEME.md`,
"nunca encadenar llamadas de Jev". `--estimar` calcula sobre esa base, SIN
red (usa los supuestos de tokens declarados en `jev/LEEME.md`, no mide
nada).

Escribe `resultado_c30_jev_<tarea>.json` de forma ATÓMICA (`mr.
_escribir_json_atomico`) TRAS la tarea, y la caché de respuestas
(`respuestas_c30_jev_<tarea>.json`, un fichero POR TAREA) tras cada fila.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "jev"))
sys.path.insert(0, "/home/deployer/matrixAI")
sys.path.insert(0, "/home/deployer/matrixai-engines/src")

import medir_c30 as mc  # noqa: E402 -- (0)/(2), TAREAS, particiones, comparación, margen
import medir_c30_respondedor as mr  # noqa: E402 -- plantilla, selección chi2, truncado C1
import respondedor_jev as rj  # noqa: E402 -- clave, adaptador HTTP, caché, análisis

#: Los 5 términos de mayor puntuación por tarea -- mismo número que (3),
#: decisión de Roberto («A y B, 5 preguntas») extendida aquí a A y C.
N_PREGUNTAS = 5

#: Truncado aproximado (caracteres) usado SOLO por `--estimar` para no tener
#: que cargar el tokenizador real de C1 (que sí se usa al medir de verdad,
#: vía `mr.truncar_como_embedding_c1`) -- una cota superior generosa, no una
#: medida.
TRUNCADO_CARACTERES_APROX = 900

SALIDA_PLANTILLA = "resultado_c30_jev_{tarea}.json"
CACHE_PLANTILLA = "respuestas_c30_jev_{tarea}.json"
ESTIMACION_SALIDA = RAIZ / "estimacion_c30_jev.json"

SELECCIONES_TERMINOS_C: tuple[str, ...] = ("chi2_quintiles", "f_regression")


# ---------------------------------------------------------------------------
# selección de términos para la tarea C (regresión): dos opciones, sin decidir
# ---------------------------------------------------------------------------

def _top_k_de_puntuaciones(puntuaciones, vocabulario_inverso: dict[int, str], *, k: int) -> list[dict[str, Any]]:
    """El mismo criterio de desempate que `mr._terminos_top_chi2`:
    puntuación descendente, alfabético como desempate -- determinista, nada
    "elegido a mano"."""
    import numpy as np

    puntuaciones = np.nan_to_num(np.asarray(puntuaciones, dtype=float), nan=-1.0)
    orden = sorted(range(len(puntuaciones)), key=lambda i: (-puntuaciones[i], vocabulario_inverso[i]))
    return [
        {"termino": vocabulario_inverso[i], "puntuacion": float(puntuaciones[i])}
        for i in orden[:k]
    ]


def _terminos_top_chi2_quintiles(textos_train: list[str], target_train: list[Any], *,
                                 k: int = N_PREGUNTAS) -> list[dict[str, Any]]:
    """Opción (a) para la tarea C: discretiza el objetivo continuo de TRAIN
    en quintiles (bordes calculados SOLO con TRAIN, `np.quantile`) y aplica
    el MISMO chi² que (3)/tarea A sobre esas 5 etiquetas discretas -- chi²
    exige un target categórico, así que esto es lo mínimo para reusarlo sin
    inventar una prueba estadística nueva.

    Sin fuga, por construcción: como `mr._terminos_top_chi2`, no tiene
    parámetro para dev/test."""
    import numpy as np
    from sklearn.feature_selection import chi2

    y = np.asarray([float(v) for v in target_train], dtype=float)
    bordes = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    y_quintil = np.digitize(y, bordes)  # 5 etiquetas: 0..4

    vectorizador = mc.construir_vectorizador_tfidf()
    X = vectorizador.fit_transform(textos_train)
    puntuaciones, _ = chi2(X, y_quintil)
    vocabulario_inverso = {indice: termino for termino, indice in vectorizador.vocabulary_.items()}
    return _top_k_de_puntuaciones(puntuaciones, vocabulario_inverso, k=k)


def _terminos_top_f_regression(textos_train: list[str], target_train: list[Any], *,
                               k: int = N_PREGUNTAS) -> list[dict[str, Any]]:
    """Opción (b) para la tarea C: `f_regression` del TF-IDF de TRAIN
    directamente contra el objetivo continuo (sin discretizar). Mismo
    vectorizador que (2)/(3) (`mc.construir_vectorizador_tfidf`), mismo
    desempate que `mr._terminos_top_chi2`."""
    import numpy as np
    from sklearn.feature_selection import f_regression

    y = np.asarray([float(v) for v in target_train], dtype=float)
    vectorizador = mc.construir_vectorizador_tfidf()
    X = vectorizador.fit_transform(textos_train)
    puntuaciones, _ = f_regression(X, y)
    vocabulario_inverso = {indice: termino for termino, indice in vectorizador.vocabulary_.items()}
    return _top_k_de_puntuaciones(puntuaciones, vocabulario_inverso, k=k)


def _seleccionar_terminos(nombre: str, cfg: dict[str, Any], train: list[dict[str, Any]], *,
                          seleccion_terminos_c: str | None) -> list[dict[str, Any]]:
    """Despacha la selección de términos SOLO con TRAIN: chi² binario
    (`mr._terminos_top_chi2`, reusado) para tareas de clasificación, y la
    opción elegida por `--seleccion-terminos-c` para la tarea C. Se niega
    (sin elegir por defecto) si la tarea es C y no se dio la opción."""
    textos_train = [f["texto"] for f in train]
    target_train = [f["target"] for f in train]
    if cfg["tipo"] == "regression":
        if seleccion_terminos_c == "chi2_quintiles":
            return _terminos_top_chi2_quintiles(textos_train, target_train)
        if seleccion_terminos_c == "f_regression":
            return _terminos_top_f_regression(textos_train, target_train)
        raise SystemExit(
            f"tarea {nombre}: falta --seleccion-terminos-c ({'/'.join(SELECCIONES_TERMINOS_C)}) -- "
            "el registro sellado no decide entre chi2_quintiles y f_regression para una tarea de "
            "regresión, así que este guion no elige por defecto."
        )
    return mr._terminos_top_chi2(textos_train, target_train)


# ---------------------------------------------------------------------------
# una tarea entera (A o C)
# ---------------------------------------------------------------------------

def evaluar_tarea_jev(
    nombre: str, *, tokenizador_c1, cache: "rj.CacheRespuestasJev", clave: str,
    seleccion_terminos_c: str | None = None, modelo: str = rj.MODELO_JEV,
    base_url: str = rj.BASE_URL_POR_OMISION, timeout_s: float = rj.TIMEOUT_S_POR_OMISION,
    tope_reintentos: int = rj.TOPE_REINTENTOS_POR_OMISION,
    espera_base_s: float = rj.ESPERA_BASE_S_POR_OMISION, dormir=time.sleep,
    semilla: int = mc.SEMILLA,
) -> dict[str, Any]:
    """La condición (4) de la tarea `nombre` ("A" o "C"), con el veredicto
    frente a (0) y la diferencia descrita frente a (2) -- misma estructura
    que `mr.evaluar_tarea_respondedor`, con Jev en vez del modelo local:
    UNA petición por fila (todos los términos pendientes juntos), en vez de
    una generación por término."""
    rj.verificar_tarea_permitida(nombre)
    cfg = mc.TAREAS[nombre]
    particiones = mc.leer_tarea(nombre)
    train, val, test = particiones["train"], particiones["dev"], particiones["test"]
    metric_id = cfg["metrica"]
    todas = train + val + test

    digest = mc._digest_particion([f["row_id"] for f in train + val], [f["row_id"] for f in test])

    # ---- (0) recalculada AQUÍ, con el mismo código y semilla que medir_c30 ----
    columnas0 = mc._columnas_condicion0(cfg)
    if columnas0 == ("_sin_columnas",):
        for filas in (train, val, test):
            mc._inyectar_columna_constante(filas)
    spec0 = mc._spec(nombre, cfg, columnas0, candidato="condicion_0_sin_texto")
    p_train0 = mc._particion_tabular(train, columnas0, tipo=cfg["tipo"], con_target=True)
    p_val0 = mc._particion_tabular(val, columnas0, tipo=cfg["tipo"], con_target=True)
    p_test0 = mc._particion_tabular(test, columnas0, tipo=cfg["tipo"], con_target=True)
    muestra0, _tiempos0 = mc._ajustar_y_predecir_lightgbm(
        p_train0, p_val0, p_test0, spec0, candidate="condicion_0_sin_texto",
        split_plan_digest=digest, semilla=semilla)

    # ---- (2) recalculada AQUÍ, para la diferencia DESCRITA (sin veredicto) ----
    muestra2, _tiempos2 = mc._condicion2_tfidf_lineal(train, val, test, cfg, semilla=semilla)
    metrica_2 = mc._metrica_puntual(metric_id, muestra2)

    # ---- selección de términos: SOLO train ----
    terminos_info = _seleccionar_terminos(nombre, cfg, train, seleccion_terminos_c=seleccion_terminos_c)
    terminos = [t["termino"] for t in terminos_info]

    # ---- preguntas por fila: UNA petición por fila (batching nativo de Jev) ----
    t_preguntas0 = time.perf_counter()
    n_si = n_no = n_faltante = 0
    for fila in todas:
        texto_truncado = fila.get("_texto_truncado_c1")
        if texto_truncado is None:
            texto_truncado = mr.truncar_como_embedding_c1(fila["texto"], tokenizador_c1)
            fila["_texto_truncado_c1"] = texto_truncado
        respuestas = rj.responder_terminos(
            tarea=nombre, row_id=fila["row_id"], texto=texto_truncado, terminos=terminos,
            texto_pregunta=lambda t: mr.PLANTILLA_PREGUNTA.format(termino=t),
            cache=cache, clave=clave, modelo=modelo, base_url=base_url, timeout_s=timeout_s,
            tope_reintentos=tope_reintentos, espera_base_s=espera_base_s, dormir=dormir,
        )
        for indice, termino in enumerate(terminos):
            valor = respuestas[termino]
            fila[f"pregunta_{indice}"] = float(valor) if valor is not None else float("nan")
            if valor == 1:
                n_si += 1
            elif valor == 0:
                n_no += 1
            else:
                n_faltante += 1

    tiempos_preguntas = {"segundos": time.perf_counter() - t_preguntas0,
                        "n_respuestas": n_si + n_no + n_faltante,
                        "n_si": n_si, "n_no": n_no, "n_faltante": n_faltante,
                        "proporcion_faltante": (n_faltante / (n_si + n_no + n_faltante))
                        if (n_si + n_no + n_faltante) else None,
                        "n_peticiones_maximo": len(todas)}  # UNA por fila, nunca N_PREGUNTAS por fila

    # ---- (4): las N columnas de pregunta + las "otras columnas", en el MISMO LightGBM ----
    columnas_pregunta = tuple(f"pregunta_{i}" for i in range(len(terminos)))
    columnas4 = tuple(cfg["otras_columnas"]) + columnas_pregunta
    spec4 = mc._spec(nombre, cfg, columnas4, candidato="condicion_4_jev")
    p_train4 = mc._particion_tabular(train, columnas4, tipo=cfg["tipo"], con_target=True)
    p_val4 = mc._particion_tabular(val, columnas4, tipo=cfg["tipo"], con_target=True)
    p_test4 = mc._particion_tabular(test, columnas4, tipo=cfg["tipo"], con_target=True)
    muestra4, tiempos4 = mc._ajustar_y_predecir_lightgbm(
        p_train4, p_val4, p_test4, spec4, candidate="condicion_4_jev",
        split_plan_digest=digest, semilla=semilla)
    tiempos4["preguntas_al_modelo"] = tiempos_preguntas
    metrica_4 = mc._metrica_puntual(metric_id, muestra4)

    comparacion_4_vs_0 = mc._comparar(metric_id, muestra4, muestra0, semilla=semilla,
                                      protocolo_candidato=digest, protocolo_baseline=digest)

    return {
        "tarea": nombre, "metrica": metric_id,
        "n_train": len(train), "n_dev": len(val), "n_test": len(test),
        "seleccion_terminos_c": seleccion_terminos_c if cfg["tipo"] == "regression" else None,
        "terminos": terminos_info,
        "condiciones": {
            "4_jev": {
                "columnas": list(columnas4), "metrica_puntual": metrica_4,
                "tiempos": tiempos4, "split_plan_digest": digest,
            },
        },
        "comparaciones": {
            "4_vs_0": comparacion_4_vs_0,
            "4_vs_2_diferencia_descrita": {
                "metrica_puntual_2_tfidf_lineal": metrica_2,
                "metrica_puntual_4_jev": metrica_4,
                "diferencia_4_menos_2": (None if metrica_4 is None or metrica_2 is None
                                        else metrica_4 - metrica_2),
                "nota": "DESCRITA, sin veredicto -- igual que 3_vs_2 en (3): no estaba en el "
                        "pre-registro.",
            },
        },
    }


# ---------------------------------------------------------------------------
# --estimar: SIN red, con los supuestos declarados de jev/LEEME.md
# ---------------------------------------------------------------------------

def estimar() -> dict[str, Any]:
    resultado: dict[str, Any] = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0, condición (4) -- Jev",
        "modelo": rj.MODELO_JEV, "base_url": rj.BASE_URL_POR_OMISION,
        "n_preguntas": N_PREGUNTAS, "peticiones_por_fila": 1,
        "supuestos": {
            "tokens_por_caracter": rj.SUPUESTO_TOKENS_POR_CARACTER,
            "tokens_fijos_por_pregunta": rj.SUPUESTO_TOKENS_FIJOS_POR_PREGUNTA,
            "truncado_caracteres_aprox": TRUNCADO_CARACTERES_APROX,
            "fuente": "benchmarks/texto_107c30/jev/LEEME.md, 'Supuestos de tokens' -- NO medido",
            "peticiones_por_segundo_asumidas": rj.PETICIONES_POR_SEGUNDO_ASUMIDAS,
            "fuente_peticiones_por_segundo": "PriorBench, 8 peticiones concurrentes (10,41/s) -- "
                                              "punto de operación medido, NO un techo publicado",
        },
        "precio_usd_por_millon_tokens_entrada": rj.PRECIO_USD_POR_MILLON_TOKENS_ENTRADA,
        "tareas": {},
    }
    total_peticiones = 0
    total_tokens_entrada = 0.0
    for nombre in rj.TAREAS_ADMITIDAS_JEV:
        try:
            particiones = mc.leer_tarea(nombre)
        except SystemExit as e:
            resultado["tareas"][nombre] = {"error": str(e)}
            continue
        filas = particiones["train"] + particiones["dev"] + particiones["test"]
        n_filas = len(filas)
        caracteres_totales = sum(min(len(f["texto"]), TRUNCADO_CARACTERES_APROX) for f in filas)
        tokens_estado = caracteres_totales * rj.SUPUESTO_TOKENS_POR_CARACTER
        tokens_preguntas = n_filas * N_PREGUNTAS * rj.SUPUESTO_TOKENS_FIJOS_POR_PREGUNTA
        tokens_entrada_tarea = tokens_estado + tokens_preguntas
        peticiones_tarea = n_filas  # UNA petición por fila -- el batching nativo de Jev
        coste_usd = tokens_entrada_tarea / 1_000_000 * rj.PRECIO_USD_POR_MILLON_TOKENS_ENTRADA
        segundos = peticiones_tarea / rj.PETICIONES_POR_SEGUNDO_ASUMIDAS
        total_peticiones += peticiones_tarea
        total_tokens_entrada += tokens_entrada_tarea
        resultado["tareas"][nombre] = {
            "n_filas": n_filas, "n_peticiones": peticiones_tarea,
            "tokens_entrada_estimados": round(tokens_entrada_tarea),
            "coste_usd_estimado": round(coste_usd, 4),
            "tiempo_estimado_s": round(segundos, 1),
            "tiempo_estimado_min": round(segundos / 60, 2),
        }
    resultado["tareas"]["B"] = {"omitida": True, "motivo": rj.TAREAS_PROHIBIDAS_JEV["B"]}
    resultado["total_peticiones"] = total_peticiones
    resultado["total_tokens_entrada_estimados"] = round(total_tokens_entrada)
    resultado["total_coste_usd_estimado"] = round(
        total_tokens_entrada / 1_000_000 * rj.PRECIO_USD_POR_MILLON_TOKENS_ENTRADA, 4)
    resultado["total_tiempo_estimado_s"] = round(total_peticiones / rj.PETICIONES_POR_SEGUNDO_ASUMIDAS, 1)
    resultado["total_tiempo_estimado_min"] = round(resultado["total_tiempo_estimado_s"] / 60, 2)
    return resultado


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimar", action="store_true", help="coste/tiempo aproximados, sin red")
    ap.add_argument("--tarea", choices=rj.TAREAS_ADMITIDAS_JEV, help="A o C (B está prohibida)")
    ap.add_argument("--seleccion-terminos-c", choices=SELECCIONES_TERMINOS_C, default=None,
                    help="obligatoria para --tarea C: el registro sellado no la decide")
    ap.add_argument("--fichero-clave", type=Path, default=rj.FICHERO_CLAVE_POR_OMISION)
    ap.add_argument("--base-url", default=rj.BASE_URL_POR_OMISION)
    ap.add_argument("--modelo", default=rj.MODELO_JEV)
    ap.add_argument("--timeout-s", type=float, default=rj.TIMEOUT_S_POR_OMISION)
    ap.add_argument("--tope-reintentos", type=int, default=rj.TOPE_REINTENTOS_POR_OMISION)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--salida", type=Path, default=None)
    ns = ap.parse_args()

    if ns.estimar:
        tabla = estimar()
        mr._escribir_json_atomico(ESTIMACION_SALIDA, tabla)
        print(json.dumps(tabla, indent=2, ensure_ascii=False))
        print(f"\nescrito en {ESTIMACION_SALIDA}", file=sys.stderr)
        return 0

    if not ns.tarea:
        ap.error("--tarea es obligatoria salvo con --estimar")
    if ns.tarea == "C" and ns.seleccion_terminos_c is None:
        ap.error(
            "--tarea C requiere --seleccion-terminos-c (chi2_quintiles|f_regression): el registro "
            "sellado no lo decide, así que este guion no elige por defecto."
        )

    clave = rj.cargar_clave_api(ns.fichero_clave)
    cache_ruta = ns.cache or (RAIZ / CACHE_PLANTILLA.format(tarea=ns.tarea))
    salida_ruta = ns.salida or (RAIZ / SALIDA_PLANTILLA.format(tarea=ns.tarea))
    cache = rj.CacheRespuestasJev(cache_ruta)
    tokenizador_c1 = mr._cargar_tokenizador_de_c1()

    print(f"=== midiendo condición (4) [Jev], tarea {ns.tarea} ===", file=sys.stderr)
    t0 = time.perf_counter()
    resultado = evaluar_tarea_jev(
        ns.tarea, tokenizador_c1=tokenizador_c1, cache=cache, clave=clave,
        seleccion_terminos_c=ns.seleccion_terminos_c, modelo=ns.modelo, base_url=ns.base_url,
        timeout_s=ns.timeout_s, tope_reintentos=ns.tope_reintentos,
    )
    resultado["duracion_total_s"] = time.perf_counter() - t0
    resultado["contrato"] = "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0, condición (4) -- Jev"
    resultado["semilla"] = mc.SEMILLA
    resultado["margen_equivalencia"] = mc.MARGEN_EQUIVALENCIA
    resultado["modelo"] = ns.modelo
    resultado["base_url"] = ns.base_url
    resultado["script_sha256"] = mc.sha256_de(Path(__file__))
    resultado["generado"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    mr._escribir_json_atomico(salida_ruta, resultado)
    print(f"--- tarea {ns.tarea}: {resultado['duracion_total_s']:.1f}s, escrito en {salida_ruta} ---",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
