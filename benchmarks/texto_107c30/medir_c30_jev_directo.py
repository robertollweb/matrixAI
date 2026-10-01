# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — condición (5): «Jev directo, sin entrenar» (registro sellado del
2026-09-29). Mide lo que Jev PROMETE -- decidir sobre un texto sin entrenar
con los datos de nadie -- frente a nuestros modelos ENTRENADOS, en las MISMAS
filas de prueba. La condición (4) (`medir_c30_jev.py`) usaba a Jev para
fabricar COLUMNAS que entraban en un LightGBM; ésta le pregunta DIRECTAMENTE
la etiqueta: no hay ningún modelo que entrenar aquí, el `noul` de Jev ES la
puntuación.

**Tareas: A (noticias falsas) y D (BOE, vigencia) -- públicas y binarias.**
B se niega (datos clínicos, nunca con Jev) y **C se niega** (regresión: Jev
tipo `noul` da una probabilidad 0-1, no un número continuo). Ninguna de las
dos negativas depende de `respondedor_jev.verificar_tarea_permitida`: esa
función solo conoce las tareas de la condición (4) (A y C) y RECHAZARÍA a D,
que aquí sí está admitida -- por eso este guion tiene su PROPIA comprobación
(`verificar_tarea_permitida_directo`), con su propia tabla de admitidas/
prohibidas, y nunca pasa por `rj.responder_terminos` (que llamaría a la
comprobación equivocada).

**Reusa, no copia** (mismo patrón que (4)):
  - `medir_c30` (`mc`): TAREAS, particiones, LightGBM (condición (0),
    recalculada aquí para el emparejamiento -- ver más abajo), comparación
    con margen, digest de partición, `Muestra`.
  - `medir_c30_respondedor` (`mr`): el tokenizador y el truncado igual que el
    embedding de C1 (`_cargar_tokenizador_de_c1`, `truncar_como_embedding_c1`)
    y su escritor atómico.
  - `medir_c30_jev` (`mj`): la cota de caracteres para `--estimar` sin cargar
    el tokenizador real.
  - `jev.respondedor_jev` (`rj`): la clave, `preguntar` (la petición HTTP de
    bajo nivel, con reintentos), la caché por (row_id, término) y
    `analizar_respuesta_noul_puntuacion` -- el `noul` CRUDO, SIN umbralizar
    (la condición (5) lo usa tal cual como probabilidad de la clase
    positiva; a diferencia de (3)/(4), que sí umbralizan a 0/1 para tener
    una columna sí/no).

**La pregunta es LITERAL y ÚNICA por fila** (no hay chi² ni selección de
términos: la condición (5) no fabrica columnas), escrita en el registro
sellado ANTES de mandar ninguna fila:
  - A: «¿Es falsa esta noticia? Responde solo sí o no.»
  - D: «¿Esta norma ha dejado de estar vigente? Responde solo sí o no.»

**Solo las filas de TEST** (572 en A, 750 en D) -- el registro lo dice
("las mismas filas de prueba") y el coste estimado (~1.300 peticiones) solo
cuadra así: train+dev+test de A y D pasan de 10.000 filas.

**El veredicto frente a (0) necesita las PREDICCIONES de (0) en las mismas
filas** (no basta con el número ya escrito en `resultado_c30.json`: el
bootstrap emparejado necesita el vector fila a fila) -- se recalcula (0) AQUÍ
con el MISMO código y semilla que `medir_c30.py`, exactamente como hace
`medir_c30_jev.py` con la condición (4). Lo descrito frente a (1), (2) y, en
A, (4) SÍ usa los números ya escritos (`resultado_c30.json`,
`resultado_c30_jev_A.json`): esas comparaciones son DESCRITAS, sin bootstrap
ni veredicto -- no estaban en el pre-registro.

**Faltantes, contados y declarados**: un `noul` que no sea válido (ausente,
tipo distinto, fuera de rango) se cuenta y se EXCLUYE de la comparación
emparejada -- de las DOS muestras (Jev y la (0) recalculada), por fila, para
que sigan alineadas. Se dice cuántas filas quedan.

Escribe `resultado_c30_jev_directo.json` (UN fichero para las dos tareas,
como `medir_c30.py`: se actualiza SOLO la tarea que se acaba de medir,
conservando lo que ya hubiera de la otra) de forma ATÓMICA, y la caché de
respuestas en `respuestas_c30_jev_directo.json` -- UNA caché COMPARTIDA entre
A y D (nunca la de la condición (4)): sus `row_id` no colisionan (`a-v1-...`
frente a `BOE-A-...`), así que no hace falta partirla por tarea, igual que
`rj.CacheRespuestasJev` ya no lleva la tarea en su clave.
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
# Las raíces del núcleo y de motores, RELATIVAS a este fichero y SOLO si faltan (02-10, re-auditoría
# de 120-C1). Con rutas fijas a los árboles PRINCIPALES en `sys.path[0]`, importar este módulo desde
# una prueba hacía que, desde un worktree, todo proceso hijo de la pasada importara el código
# principal en vez del que se probaba. Lo vigila tests/test_120_ningun_guion_mete_rutas_fijas.py.
_RAIZ_DEL_NUCLEO = Path(__file__).resolve().parents[2]
for _ruta in (_RAIZ_DEL_NUCLEO, _RAIZ_DEL_NUCLEO.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import medir_c30 as mc  # noqa: E402 -- TAREAS, particiones, LightGBM, comparación, Muestra
import medir_c30_respondedor as mr  # noqa: E402 -- tokenizador, truncado, escritor atómico
import medir_c30_jev as mj  # noqa: E402 -- TRUNCADO_CARACTERES_APROX, para --estimar
import respondedor_jev as rj  # noqa: E402 -- clave, preguntar, caché, análisis del noul

#: Las tareas de la condición (5): públicas, binarias, con un `noul` que SÍ
#: puede leerse como probabilidad de la clase positiva. B es clínica (nunca
#: Jev); C es regresión (Jev/`noul` no da un número continuo).
TAREAS_ADMITIDAS_DIRECTO: tuple[str, ...] = ("A", "D")
TAREAS_PROHIBIDAS_DIRECTO: dict[str, str] = {
    "B": "datos clínicos: nunca con Jev",
    "C": "regresión: Jev (tipo noul) da una probabilidad 0-1, no un número continuo",
}

#: Las preguntas LITERALES del registro sellado (29-09) -- ni la lista ni la
#: redacción se retocan después de medir. La clase positiva de cada tarea
#: (`Fake` en A, `agotada` en D) es la que pregunta cada frase.
PREGUNTAS_LITERALES: dict[str, str] = {
    "A": "¿Es falsa esta noticia? Responde solo sí o no.",
    "D": "¿Esta norma ha dejado de estar vigente? Responde solo sí o no.",
}

#: Nombre neutro para la clave de caché (row_id, término): una sola pregunta
#: por fila, no un término del texto -- a diferencia de (3)/(4).
_TERMINO_CACHE = "etiqueta_directa"

SALIDA_POR_OMISION = RAIZ / "resultado_c30_jev_directo.json"
CACHE_POR_OMISION = RAIZ / "respuestas_c30_jev_directo.json"
ESTIMACION_SALIDA = RAIZ / "estimacion_c30_jev_directo.json"

RESULTADO_C30 = RAIZ / "resultado_c30.json"
RESULTADO_C30_JEV_A = RAIZ / "resultado_c30_jev_A.json"


def verificar_tarea_permitida_directo(nombre: str) -> None:
    """Se niega, con SU motivo, a cualquier tarea que no sea A o D. Nunca
    llama a `rj.verificar_tarea_permitida` (esa tabla es de la condición (4)
    y no admite D) -- esta es la comprobación propia de la condición (5), y
    es la PRIMERA línea de `evaluar_tarea_directo`, antes de leer ningún
    dato."""
    if nombre in TAREAS_PROHIBIDAS_DIRECTO:
        raise SystemExit(
            f"tarea {nombre} PROHIBIDA para la condición (5) [Jev directo]: "
            f"{TAREAS_PROHIBIDAS_DIRECTO[nombre]}"
        )
    if nombre not in TAREAS_ADMITIDAS_DIRECTO:
        raise SystemExit(
            f"tarea {nombre} no admitida para la condición (5) (Jev directo, sin entrenar): "
            f"solo {TAREAS_ADMITIDAS_DIRECTO}"
        )


# ---------------------------------------------------------------------------
# UNA pregunta por fila (sin términos): construida sobre rj.preguntar + la
# misma caché por (row_id, término), pero SIN pasar por rj.verificar_tarea_
# permitida (ver docstring del módulo).
# ---------------------------------------------------------------------------

def responder_etiqueta(
    *, row_id: str, texto: str, pregunta: str, cache: "rj.CacheRespuestasJev", clave: str,
    modelo: str = rj.MODELO_JEV, base_url: str = rj.BASE_URL_POR_OMISION,
    timeout_s: float = rj.TIMEOUT_S_POR_OMISION, tope_reintentos: int = rj.TOPE_REINTENTOS_POR_OMISION,
    espera_base_s: float = rj.ESPERA_BASE_S_POR_OMISION, dormir=time.sleep,
) -> float | None:
    """La puntuación (0..1, SIN umbralizar) para `pregunta` sobre `texto`,
    reusando la caché si ya se preguntó. Una fila = una petición con UNA
    pregunta `noul`. Devuelve `None` si la respuesta no es un noul válido
    (faltante, contado por quien llama)."""
    entrada = cache.obtener(row_id, _TERMINO_CACHE)
    if entrada is None:
        datos = rj.preguntar(
            estado=texto, preguntas={"q0": {"type": "noul", "instructions": pregunta}},
            clave=clave, modelo=modelo, base_url=base_url, timeout_s=timeout_s,
            tope_reintentos=tope_reintentos, espera_base_s=espera_base_s, dormir=dormir,
        )
        noul_bruto = datos.get("answers", {}).get("q0")
        cache.fijar(row_id, _TERMINO_CACHE, {
            "noul_bruto": noul_bruto,
            "puntuacion": rj.analizar_respuesta_noul_puntuacion(noul_bruto),
        })
        cache.persistir()
        entrada = cache.obtener(row_id, _TERMINO_CACHE)
    return entrada["puntuacion"]


# ---------------------------------------------------------------------------
# lectura de resultados YA escritos, para lo DESCRITO (sin veredicto)
# ---------------------------------------------------------------------------

def _leer_metrica_previa(ruta: Path, *claves: str) -> float | None:
    """Un valor anidado de un JSON ya escrito, o `None` si el fichero o la
    ruta de claves no existen -- nunca revienta: lo DESCRITO se declara
    como "no disponible" en vez de tirar abajo la tarea que sí se puede
    medir."""
    try:
        nodo: Any = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for c in claves:
        if not isinstance(nodo, dict) or c not in nodo:
            return None
        nodo = nodo[c]
    return nodo if isinstance(nodo, (int, float)) else None


def _diferencia_descrita(nombre_otro: str, metrica_otro: float | None, metrica_5: float | None,
                         *, fuente: str) -> dict[str, Any]:
    return {
        f"metrica_puntual_{nombre_otro}": metrica_otro,
        "metrica_puntual_5_jev_directo": metrica_5,
        f"diferencia_5_menos_{nombre_otro.split('_')[0]}": (
            None if metrica_5 is None or metrica_otro is None else metrica_5 - metrica_otro
        ),
        "fuente": fuente,
        "nota": "DESCRITA, sin veredicto: no estaba en el pre-registro.",
    }


# ---------------------------------------------------------------------------
# una tarea entera (A o D)
# ---------------------------------------------------------------------------

def evaluar_tarea_directo(
    nombre: str, *, tokenizador_c1, cache: "rj.CacheRespuestasJev", clave: str,
    modelo: str = rj.MODELO_JEV, base_url: str = rj.BASE_URL_POR_OMISION,
    timeout_s: float = rj.TIMEOUT_S_POR_OMISION, tope_reintentos: int = rj.TOPE_REINTENTOS_POR_OMISION,
    espera_base_s: float = rj.ESPERA_BASE_S_POR_OMISION, dormir=time.sleep,
    semilla: int = mc.SEMILLA,
) -> dict[str, Any]:
    """La condición (5) de la tarea `nombre` ("A" o "D"): UNA pregunta por
    fila de TEST, el `noul` crudo como puntuación de AUROC, y el veredicto
    emparejado frente a una condición (0) recalculada AQUÍ mismo (mismo
    código, misma semilla, mismas particiones que `medir_c30.py`)."""
    verificar_tarea_permitida_directo(nombre)
    cfg = mc.TAREAS[nombre]
    if cfg["tipo"] != "binary_classification":
        # No debería poder pasar (A y D son las dos binarias), pero declarado
        # explícitamente: un noul es una probabilidad 0-1, nunca un valor de
        # regresión.
        raise SystemExit(f"tarea {nombre}: la condición (5) solo mide binaria (AUROC)")
    particiones = mc.leer_tarea(nombre)
    train, val, test = particiones["train"], particiones["dev"], particiones["test"]
    metric_id = cfg["metrica"]
    pregunta = PREGUNTAS_LITERALES[nombre]

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
    muestra0_completa, _tiempos0 = mc._ajustar_y_predecir_lightgbm(
        p_train0, p_val0, p_test0, spec0, candidate="condicion_0_sin_texto",
        split_plan_digest=digest, semilla=semilla)

    # ---- UNA pregunta por fila de TEST (no train/dev: el registro mide "las
    # mismas filas de prueba") ----
    t_preguntas0 = time.perf_counter()
    puntuaciones: list[float | None] = []
    for fila in test:
        texto_truncado = fila.get("_texto_truncado_c1")
        if texto_truncado is None:
            texto_truncado = mr.truncar_como_embedding_c1(fila["texto"], tokenizador_c1)
            fila["_texto_truncado_c1"] = texto_truncado
        puntuacion = responder_etiqueta(
            row_id=fila["row_id"], texto=texto_truncado, pregunta=pregunta, cache=cache, clave=clave,
            modelo=modelo, base_url=base_url, timeout_s=timeout_s, tope_reintentos=tope_reintentos,
            espera_base_s=espera_base_s, dormir=dormir,
        )
        puntuaciones.append(puntuacion)
    segundos_preguntas = time.perf_counter() - t_preguntas0

    n_total = len(test)
    n_faltante = sum(1 for p in puntuaciones if p is None)
    n_con_respuesta = n_total - n_faltante

    # ---- filas CON respuesta: filtra las DOS muestras por los MISMOS
    # índices, para que sigan emparejadas fila a fila ----
    indices_con_respuesta = [i for i, p in enumerate(puntuaciones) if p is not None]
    test_con_respuesta = [test[i] for i in indices_con_respuesta]
    puntuaciones_con_respuesta = [float(puntuaciones[i]) for i in indices_con_respuesta]

    muestra_jev = mc.Muestra(
        task=cfg["tipo"], y_true=tuple(f["target"] for f in test_con_respuesta),
        classes=cfg["clases"], positive_label=cfg["positive_label"],
        scores=tuple(puntuaciones_con_respuesta),
    )
    muestra0_filtrada = mc.Muestra(
        task=cfg["tipo"],
        y_true=tuple(muestra0_completa.y_true[i] for i in indices_con_respuesta),
        classes=cfg["clases"], positive_label=cfg["positive_label"],
        scores=tuple(muestra0_completa.scores[i] for i in indices_con_respuesta),
    )

    metrica_5 = mc._metrica_puntual(metric_id, muestra_jev)
    metrica_0_misma_muestra = mc._metrica_puntual(metric_id, muestra0_filtrada)
    metrica_0_completa = mc._metrica_puntual(metric_id, muestra0_completa)

    comparacion_5_vs_0 = mc._comparar(
        metric_id, muestra_jev, muestra0_filtrada, semilla=semilla,
        protocolo_candidato=digest, protocolo_baseline=digest)

    # ---- descrito, sin veredicto: (1), (2) y, en A, (4) -- de los ficheros
    # YA escritos, no recalculados ----
    metrica_1_previa = _leer_metrica_previa(
        RESULTADO_C30, "tareas", nombre, "condiciones", "1_embedding", "metrica_puntual")
    metrica_2_previa = _leer_metrica_previa(
        RESULTADO_C30, "tareas", nombre, "condiciones", "2_tfidf_lineal", "metrica_puntual")
    comparaciones: dict[str, Any] = {
        "5_vs_0": comparacion_5_vs_0,
        "5_vs_1_diferencia_descrita": _diferencia_descrita(
            "1_embedding", metrica_1_previa, metrica_5, fuente="resultado_c30.json"),
        "5_vs_2_diferencia_descrita": _diferencia_descrita(
            "2_tfidf_lineal", metrica_2_previa, metrica_5, fuente="resultado_c30.json"),
    }
    if nombre == "A":
        metrica_4_previa = _leer_metrica_previa(
            RESULTADO_C30_JEV_A, "condiciones", "4_jev", "metrica_puntual")
        comparaciones["5_vs_4_diferencia_descrita"] = _diferencia_descrita(
            "4_jev_columnas", metrica_4_previa, metrica_5, fuente="resultado_c30_jev_A.json")

    return {
        "tarea": nombre, "metrica": metric_id, "condicion": "5_jev_directo",
        "pregunta": pregunta,
        "n_train": len(train), "n_dev": len(val), "n_test": n_total,
        "respuestas": {
            "segundos": segundos_preguntas, "n_total": n_total, "n_faltante": n_faltante,
            "n_con_respuesta": n_con_respuesta,
            "proporcion_faltante": (n_faltante / n_total) if n_total else None,
        },
        "condiciones": {
            "5_jev_directo": {
                "metrica_puntual": metrica_5, "n_filas": n_con_respuesta,
                "split_plan_digest": digest,
            },
            "0_sin_texto_recalculada": {
                "metrica_puntual_muestra_completa": metrica_0_completa,
                "metrica_puntual_misma_muestra_que_5": metrica_0_misma_muestra,
                "n_filas_muestra_completa": n_total,
            },
        },
        "comparaciones": comparaciones,
    }


# ---------------------------------------------------------------------------
# --estimar: SIN red, con los supuestos declarados de jev/LEEME.md, y solo
# sobre las filas de TEST (una pregunta por fila, no train+dev+test)
# ---------------------------------------------------------------------------

def estimar() -> dict[str, Any]:
    resultado: dict[str, Any] = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0, condición (5) -- "
                    "Jev directo, sin entrenar",
        "modelo": rj.MODELO_JEV, "base_url": rj.BASE_URL_POR_OMISION,
        "peticiones_por_fila": 1, "preguntas_por_peticion": 1,
        "solo_filas_de": "test",
        "supuestos": {
            "tokens_por_caracter": rj.SUPUESTO_TOKENS_POR_CARACTER,
            "tokens_fijos_por_pregunta": rj.SUPUESTO_TOKENS_FIJOS_POR_PREGUNTA,
            "truncado_caracteres_aprox": mj.TRUNCADO_CARACTERES_APROX,
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
    for nombre in TAREAS_ADMITIDAS_DIRECTO:
        try:
            particiones = mc.leer_tarea(nombre)
        except SystemExit as e:
            resultado["tareas"][nombre] = {"error": str(e)}
            continue
        filas = particiones["test"]  # SOLO test -- "las mismas filas de prueba" del registro
        n_filas = len(filas)
        caracteres_totales = sum(min(len(f["texto"]), mj.TRUNCADO_CARACTERES_APROX) for f in filas)
        tokens_estado = caracteres_totales * rj.SUPUESTO_TOKENS_POR_CARACTER
        tokens_preguntas = n_filas * 1 * rj.SUPUESTO_TOKENS_FIJOS_POR_PREGUNTA
        tokens_entrada_tarea = tokens_estado + tokens_preguntas
        peticiones_tarea = n_filas  # UNA petición por fila
        coste_usd = tokens_entrada_tarea / 1_000_000 * rj.PRECIO_USD_POR_MILLON_TOKENS_ENTRADA
        segundos = peticiones_tarea / rj.PETICIONES_POR_SEGUNDO_ASUMIDAS
        total_peticiones += peticiones_tarea
        total_tokens_entrada += tokens_entrada_tarea
        resultado["tareas"][nombre] = {
            "n_filas_test": n_filas, "n_peticiones": peticiones_tarea,
            "tokens_entrada_estimados": round(tokens_entrada_tarea),
            "coste_usd_estimado": round(coste_usd, 4),
            "tiempo_estimado_s": round(segundos, 1),
            "tiempo_estimado_min": round(segundos / 60, 2),
        }
    resultado["tareas"]["B"] = {"omitida": True, "motivo": TAREAS_PROHIBIDAS_DIRECTO["B"]}
    resultado["tareas"]["C"] = {"omitida": True, "motivo": TAREAS_PROHIBIDAS_DIRECTO["C"]}
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
    ap.add_argument("--tarea", choices=TAREAS_ADMITIDAS_DIRECTO, help="A o D (B y C están prohibidas)")
    ap.add_argument("--fichero-clave", type=Path, default=rj.FICHERO_CLAVE_POR_OMISION)
    ap.add_argument("--base-url", default=rj.BASE_URL_POR_OMISION)
    ap.add_argument("--modelo", default=rj.MODELO_JEV)
    ap.add_argument("--timeout-s", type=float, default=rj.TIMEOUT_S_POR_OMISION)
    ap.add_argument("--tope-reintentos", type=int, default=rj.TOPE_REINTENTOS_POR_OMISION)
    ap.add_argument("--cache", type=Path, default=CACHE_POR_OMISION,
                    help="propia de la condición (5) -- NUNCA la de la (4)")
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_OMISION)
    ns = ap.parse_args()

    if ns.estimar:
        tabla = estimar()
        mr._escribir_json_atomico(ESTIMACION_SALIDA, tabla)
        print(json.dumps(tabla, indent=2, ensure_ascii=False))
        print(f"\nescrito en {ESTIMACION_SALIDA}", file=sys.stderr)
        return 0

    if not ns.tarea:
        ap.error("--tarea es obligatoria salvo con --estimar")

    clave = rj.cargar_clave_api(ns.fichero_clave)
    cache = rj.CacheRespuestasJev(ns.cache)
    tokenizador_c1 = mr._cargar_tokenizador_de_c1()

    print(f"=== midiendo condición (5) [Jev directo, sin entrenar], tarea {ns.tarea} ===",
          file=sys.stderr)
    t0 = time.perf_counter()
    resultado = evaluar_tarea_directo(
        ns.tarea, tokenizador_c1=tokenizador_c1, cache=cache, clave=clave,
        modelo=ns.modelo, base_url=ns.base_url, timeout_s=ns.timeout_s,
        tope_reintentos=ns.tope_reintentos,
    )
    resultado["duracion_total_s"] = time.perf_counter() - t0

    # UN fichero para las dos tareas: se actualiza solo `ns.tarea`, se
    # conserva lo que ya hubiera de la otra (igual que medir_c30.py).
    salida: dict[str, Any] = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0, condición (5) -- "
                    "Jev directo, sin entrenar",
        "semilla": mc.SEMILLA, "margen_equivalencia": mc.MARGEN_EQUIVALENCIA,
        "modelo": ns.modelo, "base_url": ns.base_url,
        "script_sha256": mc.sha256_de(Path(__file__)),
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tareas": {},
    }
    if ns.salida.is_file():
        try:
            previo = json.loads(ns.salida.read_text(encoding="utf-8"))
            salida["tareas"].update(previo.get("tareas", {}))
        except (json.JSONDecodeError, OSError):
            pass
    salida["tareas"][ns.tarea] = resultado

    mr._escribir_json_atomico(ns.salida, salida)
    print(json.dumps(resultado, indent=2, ensure_ascii=False), file=sys.stderr)
    print(f"--- tarea {ns.tarea}: {resultado['duracion_total_s']:.1f}s, escrito en {ns.salida} ---",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
