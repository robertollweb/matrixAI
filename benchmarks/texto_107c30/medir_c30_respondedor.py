# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — condición (3): «un respondedor local de preguntas atómicas».

Implementa EXACTAMENTE el «REGISTRO SELLADO de la condición (3) — 26-09»
de `documentacion/107_TEXTO_CON_PREENTRENADOS_CONTRACT.md` (decisión de
Roberto «A y B, 5 preguntas»), escrito ANTES de construir esto y de ver
ninguna respuesta del respondedor. No se retoca después de medir.

  - **Tareas**: A y B. C y D quedan «no medidas por coste» (ver `TAREAS_OMITIDAS`).
  - **Preguntas**: por tarea, los 5 términos de mayor chi² sobre el TF-IDF de
    la condición (2) (`medir_c30.construir_vectorizador_tfidf`), calculados
    SOLO con las filas de TRAIN de la partición de desarrollo (sin fuga) --
    NUNCA con dev ni test. Ni la lista ni la plantilla se escogen a mano:
    `_terminos_top_chi2` es determinista y automática.
  - **Respondedor**: Qwen2.5-0.5B-Instruct en ONNX-GenAI (la conversión de
    terceros de `sondear_respondedor.py`, reutilizada de ahí -- MODELO_REPO,
    MODELO_SUBCARPETA, MODELO_LOCAL_POR_OMISION, `_fijar_hilos_en_config`),
    decodificación VORAZ (`do_sample=False`), 4 hilos, el texto con el MISMO
    truncado que el embedding de C1 (ver `truncar_como_embedding_c1`).
    Respuesta: sí->1, no->0 (primera palabra, sin mayúsculas ni tildes); lo
    demás -> faltante, CONTADO y declarado (`analizar_respuesta`).
  - **Modelo**: las 5 columnas, junto a las demás columnas de la tarea, en el
    MISMO LightGBM (`matrixai_engines.motores.arbol_lightgbm.
    MotorArbolLightGBM`, vía las funciones YA EXISTENTES de `medir_c30.py`)
    y con las MISMAS particiones y semillas de (0), (1) y (2) -- se llama
    literalmente a `medir_c30._ajustar_y_predecir_lightgbm` con el mismo
    `split_plan_digest`, no una réplica del cableado.
  - **Veredicto**: frente a (0), emparejado y con el margen 0,05 del
    pre-registro (`medir_c30._comparar`, que ya usa `medir_c30.
    MARGEN_EQUIVALENCIA`). Frente a (2), la diferencia se DESCRIBE (¿añade
    algo al TF-IDF?), sin veredicto: no estaba en el pre-registro.

**Una decisión que el registro NO fija literalmente, y que aquí se declara**:
el registro da la PREGUNTA («¿El texto trata de «t»...?») pero no dice cómo
se le entrega el texto al modelo junto a ella. Se reutiliza el mismo
envoltorio ya medido en `sondear_respondedor.PREGUNTA_ATOMICA` («Lee el
siguiente texto...\\nPregunta: ...\\nTexto: ...»), sustituyendo su pregunta
fija por la de este registro -- no se inventa un formato nuevo sin medir.
Si esto se audita, es el punto a revisar contra el original.

**No toca `medir_c30.py` más que lo ya extraído en él** (`construir_
vectorizador_tfidf`, sin cambiar lo que calcula: sus pruebas siguen verdes,
ver `tests/test_107_c30_arnes.py`) **ni `sondear_respondedor.py`**: de los
dos solo se IMPORTAN constantes y funciones (reuso, no copia).

Escribe el resultado en JSON con su procedencia, de forma ATÓMICA
(`_escribir_json_atomico`, `os.replace` -- nunca `write_text` a secas: es
justo el patrón que costó el `JSONDecodeError` de `estudio_status` en
producción) y TRAS CADA TAREA. La caché de respuestas
(`cache_respondedor.json` por omisión) es por (tarea, row_id, término): un
relanzamiento no vuelve a preguntarle al modelo lo que ya contestó.

`onnxruntime_genai` se importa PEREZOSAMENTE, solo dentro de las dos
funciones que de verdad llaman al modelo (`_cargar_modelo_y_tokenizer`,
`_generar_respuesta_atomica`): el resto de este módulo -- selección de
términos, plantilla, análisis de respuesta, caché, el guion de contenedor --
se puede importar y probar en el HOST, que no lo tiene instalado (26-09,
"NO instales NADA en el host").
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
# Las raíces del núcleo y de motores, RELATIVAS a este fichero y SOLO si faltan (02-10, re-auditoría
# de 120-C1). Con rutas fijas a los árboles PRINCIPALES en `sys.path[0]`, importar este módulo desde
# una prueba hacía que, desde un worktree, todo proceso hijo de la pasada importara el código
# principal en vez del que se probaba. Lo vigila tests/test_120_ningun_guion_mete_rutas_fijas.py.
_RAIZ_DEL_NUCLEO = Path(__file__).resolve().parents[2]
for _ruta in (_RAIZ_DEL_NUCLEO, _RAIZ_DEL_NUCLEO.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import medir_c30 as mc  # noqa: E402 -- (0)/(1)/(2), TAREAS, particiones, comparación, margen
import sondear_respondedor as sr  # noqa: E402 -- modelo, ruta de pesos, fijar hilos

# ---------------------------------------------------------------------------
# lo que fija el registro sellado
# ---------------------------------------------------------------------------

#: Tareas A y B, decisión de Roberto («A y B, 5 preguntas»). C y D NO se
#: miden aquí: con 5 preguntas su coste (D ~8,4h, C ~50,8h, PROPUESTA 25-09
#: "entre dos") las deja fuera. Se declaran igualmente en el resultado, no
#: en silencio.
TAREAS_A_MEDIR: tuple[str, ...] = ("A", "B")
TAREAS_OMITIDAS: dict[str, str] = {
    "C": "no medida por coste (registro sellado 26-09): con 5 preguntas, "
         "PLACSP costaría del orden de 50,8 h (mitad de las 101,5 h con 10 preguntas de la "
         "PROPUESTA del 25-09) -- 'no medida por coste' está en el propio registro.",
    "D": "no medida por coste (registro sellado 26-09): con 5 preguntas, BOE costaría del "
         "orden de 8,4 h (mitad de las 16,7 h con 10 preguntas de la PROPUESTA del 25-09) -- "
         "'no medida por coste' está en el propio registro.",
}

#: Los 5 términos de mayor chi² por tarea, decisión de Roberto.
N_PREGUNTAS = 5

#: Plantilla LITERAL del registro sellado -- «Cada término `t` -> «¿El texto
#: trata de «t» o de algo equivalente? Responde solo sí o no.»». No se
#: retoca: hay una prueba con su nombre (`test_107_c30_respondedor.py`).
PLANTILLA_PREGUNTA = "¿El texto trata de «{termino}» o de algo equivalente? Responde solo sí o no."

#: El envoltorio que de verdad recibe el modelo (pregunta + texto). El
#: registro fija la PREGUNTA; este envoltorio es el ya medido en
#: `sondear_respondedor.PREGUNTA_ATOMICA` con la pregunta fija sustituida por
#: la de este registro -- declarado arriba en el docstring del módulo.
PLANTILLA_MENSAJE = (
    'Lee el siguiente texto y responde SOLO con la palabra "sí" o la palabra "no", '
    "sin explicación.\n"
    "Pregunta: {pregunta}\n"
    "Texto: {texto}"
)

MAX_TOKENS_NUEVOS = sr.MAX_TOKENS_NUEVOS  # 8: una respuesta atómica sí/no no necesita más
HILOS = sr.HILOS  # 4, decisión del registro

#: 5 x filas x 1,2s -- la aritmética LITERAL del registro sellado. Se declara
#: aquí como valor de reserva; `_tasa_s_por_respuesta` prefiere la medida
#: real de `resultado_sonda_respondedor.json` si existe (medir, no suponer),
#: y cae a esta constante si esa sonda no se ha corrido todavía.
TASA_S_POR_RESPUESTA_DEL_REGISTRO = 1.2

SALIDA_POR_OMISION = RAIZ / "resultado_c30_respondedor.json"
CACHE_POR_OMISION = RAIZ / "cache_respondedor.json"

#: El proveedor de embeddings de C1 cuyo truncado hay que igualar (decisión
#: 3a del 107, la misma constante que usa `medir_c30.PROVEEDOR_EMBEDDING_ID`
#: -- se LEE de ahí, no se copia el id a mano).
PROVEEDOR_EMBEDDING_C1 = mc.PROVEEDOR_EMBEDDING_ID


def _escribir_json_atomico(ruta: Path, datos: Any, *, indent: int | None = 2) -> None:
    """`os.replace`, nunca `write_text` a secas -- un lector a medio escribir
    no puede ver un JSON roto (el `JSONDecodeError` de `estudio_status` en
    producción fue justo esto, con `write_text` en el sitio de `os.replace`)."""
    tmp = ruta.with_name(ruta.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(datos, indent=indent, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ruta)


# ---------------------------------------------------------------------------
# selección de términos: chi² sobre el TF-IDF de (2), SOLO con TRAIN
# ---------------------------------------------------------------------------

def _terminos_top_chi2(textos_train: list[str], target_train: list[Any], *, k: int = N_PREGUNTAS) -> list[dict[str, Any]]:
    """Los `k` términos de mayor chi² sobre el TF-IDF de la condición (2)
    (`medir_c30.construir_vectorizador_tfidf`, el MISMO vectorizador, no una
    copia de sus parámetros), ajustado y evaluado SOLO sobre `textos_train`/
    `target_train`.

    **Sin fuga, por construcción**: esta función no ve dev ni test -- no
    tiene ni siquiera un parámetro para pasárselos. Quien la llame con las
    filas de TRAIN de la tarea (y solo esas) cumple el registro; el sabotaje
    de la prueba que la acompaña es justo pasarle también filas de test y
    comprobar que el término que solo aparece ahí NO sale elegido.

    Determinista y automática (nada de "elegido a mano"): empate en la
    puntuación se rompe por orden alfabético del término, no por el orden en
    que `sklearn` los enumeró.
    """
    import numpy as np
    from sklearn.feature_selection import chi2

    vectorizador = mc.construir_vectorizador_tfidf()
    X = vectorizador.fit_transform(textos_train)
    puntuaciones, _ = chi2(X, target_train)
    puntuaciones = np.nan_to_num(puntuaciones, nan=-1.0)
    vocabulario_inverso = {indice: termino for termino, indice in vectorizador.vocabulary_.items()}
    orden = sorted(range(len(puntuaciones)), key=lambda i: (-puntuaciones[i], vocabulario_inverso[i]))
    return [
        {"termino": vocabulario_inverso[i], "chi2": float(puntuaciones[i])}
        for i in orden[:k]
    ]


# ---------------------------------------------------------------------------
# el truncado, IGUAL que el embedding de C1
# ---------------------------------------------------------------------------

def _cargar_tokenizador_de_c1():
    """El tokenizador Unigram REAL del proveedor de embeddings de C1 (107-C1,
    D1), cargado SOLO con sus ficheros pequeños (`tokenizer.json`,
    `special_tokens_map.json`) -- no hace falta cargar el grafo ONNX entero
    (los pesos del embedding) solo para truncar texto igual que él trunca.

    Usa `LONGITUDES_FIJADAS` de `matrixai_engines.embeddings.
    proveedor_de_texto` para el número de tokens: es la MISMA fuente que
    usaría la condición (1) real, no un 128 copiado a mano (107, invariante
    6: "un límite heredado... nunca se hereda del valor por omisión").
    """
    from matrixai.text.embeddings.descarga import raiz_cache
    from matrixai_engines.embeddings.proveedor_de_texto import LONGITUDES_FIJADAS
    from matrixai_engines.embeddings.tokenizador_unigram import TokenizadorUnigram

    fijada = LONGITUDES_FIJADAS[PROVEEDOR_EMBEDDING_C1]
    directorio = raiz_cache() / PROVEEDOR_EMBEDDING_C1
    especiales_ruta = directorio / "special_tokens_map.json"
    if not especiales_ruta.is_file():
        raise SystemExit(
            f"falta el tokenizador de {PROVEEDOR_EMBEDDING_C1} en {directorio} (para el truncado "
            "de la condición (3)); en el contenedor se monta con MATRIXAI_EMBEDDINGS_HOME, ver "
            "correr_respondedor_en_contenedor.sh"
        )
    especiales = json.loads(especiales_ruta.read_text(encoding="utf-8"))
    return TokenizadorUnigram(
        directorio / "tokenizer.json", max_longitud=fijada.tokens,
        unk_token=str(especiales["unk_token"]), pad_token=str(especiales["pad_token"]),
    )


#: SentencePiece marca el inicio de palabra con este carácter (U+2581, "▁"),
#: no con un espacio literal -- así reconstruye `truncar_como_embedding_c1`
#: texto normal a partir de las piezas.
_MARCADOR_DE_PALABRA_SENTENCEPIECE = "▁"


def truncar_como_embedding_c1(texto: str, tokenizador) -> str:
    """Recorta `texto` a las mismas piezas que vería el embedding de C1
    (`tokenizador.max_longitud`), usando el MISMO tokenizador real -- no una
    aproximación por caracteres.

    **Por qué hace falta mirar el conteo SIN truncar primero.**
    `TokenizadorUnigram` fija el truncado en el `tokenizers.Tokenizer` interno
    al construirse (`enable_truncation(max_length=...)`, 107-C1 D1): por eso
    `tokenizador.codificar(texto)` NUNCA devuelve más de `max_longitud`
    piezas, tanto si el texto cabía justo como si sobraban miles de piezas --
    medido el 26-09: con un texto de 500 repeticiones de "palabra" (mucho más
    largo que el límite), `codificar()` ya venía recortado a exactamente 128,
    y la primera versión de esta función confundía "recortado a 128" con
    "cabía en 128", devolviendo el texto ENTERO sin tocar. Por eso se apaga
    el truncado del tokenizador interno un instante para contar de verdad
    (`_t.no_truncation()` / `_t.encode(...)`, no hay una superficie pública
    para esto en `TokenizadorUnigram`), y se repone justo después.

    Si el conteo SIN truncar ya cabe en `max_longitud`, se devuelve el texto
    tal cual: no hay nada que cortar y no hace falta reconstruir nada byte a
    byte distinto. Si no cabe, se toman las `max_longitud` PRIMERAS piezas de
    la tokenización YA truncada (que en ese caso sí refleja el corte real) y
    se reconstruye el texto uniendo esas piezas y sustituyendo el marcador de
    palabra "▁" por un espacio -- así el respondedor ve, en piezas de C1, la
    misma ventana de texto que se le pasaría a ese embedding.

    Se pide `con_especiales=False`: `<s>`/`</s>` no son texto y reconstruirlos
    no aportaría nada; esto deja hasta `max_longitud` piezas de contenido
    real (dos más que las que usaría la condición (1), que gasta 2 de sus 128
    en esos dos especiales) -- diferencia declarada, irrelevante para lo que
    aquí importa (que el respondedor no vea más texto del que vio el
    embedding, no una paridad byte a byte de presupuesto de tokens).

    Estable al repetirse (`test_107_c30_respondedor.py`): un texto que ya
    cabe se devuelve intacto, así que aplicar esto dos veces da lo mismo que
    aplicarlo una.
    """
    tope = tokenizador.max_longitud
    interno = tokenizador._t  # el `tokenizers.Tokenizer` real -- ver el "por qué" arriba
    interno.no_truncation()
    try:
        conteo_sin_truncar = len(interno.encode(texto, add_special_tokens=False).tokens)
    finally:
        interno.enable_truncation(max_length=tope)
    if conteo_sin_truncar <= tope:
        return texto
    piezas = tokenizador.codificar(texto, con_especiales=False).tokens[:tope]
    reconstruido = "".join(piezas).replace(_MARCADOR_DE_PALABRA_SENTENCEPIECE, " ")
    return reconstruido.strip()


# ---------------------------------------------------------------------------
# análisis de la respuesta del modelo
# ---------------------------------------------------------------------------

def _sin_diacriticos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


_RE_PRIMERA_PALABRA = re.compile(r"[^a-z]*([a-z]+)")


def analizar_respuesta(texto_bruto: str) -> int | None:
    """sí -> 1, no -> 0 (primera palabra, en minúsculas y sin tildes); lo
    demás -> `None` (faltante). EXACTAMENTE lo que pide el registro: "primer
    token, sin mayúsculas ni tildes"; lo demás, faltante contado y declarado,
    NO relleno.

    Ejemplos medidos en la prueba que acompaña a esta función: "Sí" -> 1,
    "sí." -> 1, "No" -> 0, "no," -> 0, "quizá" -> None.
    """
    normalizado = _sin_diacriticos(texto_bruto.strip().lower())
    coincidencia = _RE_PRIMERA_PALABRA.match(normalizado)
    if not coincidencia:
        return None
    palabra = coincidencia.group(1)
    if palabra == "si":
        return 1
    if palabra == "no":
        return 0
    return None


# ---------------------------------------------------------------------------
# caché por (tarea, row_id, término)
# ---------------------------------------------------------------------------

class CacheRespondedor:
    """Respuestas ya dadas, por (tarea, row_id, término), para que un
    relanzamiento no vuelva a preguntarle al modelo lo mismo. Se escribe de
    forma ATÓMICA (`_escribir_json_atomico`) tras cada FILA completa (sus 5
    preguntas), no tras cada pregunta suelta -- suficiente para no perder más
    que la fila en curso si algo corta a media medición, y sin reescribir el
    fichero entero 5 veces más de las necesarias.
    """

    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta
        self._datos: dict[str, dict[str, Any]] = {}
        if ruta.is_file():
            try:
                self._datos = json.loads(ruta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._datos = {}

    @staticmethod
    def _clave(tarea: str, row_id: str, termino: str) -> str:
        return f"{tarea}\x1f{row_id}\x1f{termino}"

    def obtener(self, tarea: str, row_id: str, termino: str) -> dict[str, Any] | None:
        return self._datos.get(self._clave(tarea, row_id, termino))

    def fijar(self, tarea: str, row_id: str, termino: str, entrada: dict[str, Any]) -> None:
        self._datos[self._clave(tarea, row_id, termino)] = entrada

    def persistir(self) -> None:
        _escribir_json_atomico(self.ruta, self._datos, indent=None)

    def __len__(self) -> int:
        return len(self._datos)


# ---------------------------------------------------------------------------
# el modelo: carga perezosa y generación voraz
# ---------------------------------------------------------------------------

def _cargar_modelo_y_tokenizer(modelo_local: Path, *, hilos: int):
    """Carga Qwen2.5-0.5B-Instruct-onnx-genai. Reusa `sondear_respondedor.
    _fijar_hilos_en_config` (el MISMO fichero de configuración de hilos, no
    una copia). Este guion NO descarga el modelo (a diferencia de la sonda):
    en el contenedor de medición `--network none`, y en el host no se
    instala nada -- si falta, se para con el motivo, no se intenta bajar."""
    import onnxruntime_genai as og  # noqa: PLC0415 -- perezoso a propósito

    if not (modelo_local / "genai_config.json").is_file():
        raise SystemExit(
            f"falta el modelo en {modelo_local}. Este guion NO descarga nada (network none en el "
            "contenedor de medición): correr `sondear_respondedor.py` una vez para bajarlo, o "
            "montarlo ya descargado (ver correr_respondedor_en_contenedor.sh)."
        )
    sr._fijar_hilos_en_config(modelo_local, hilos=hilos)
    modelo = og.Model(str(modelo_local))
    tokenizer = og.Tokenizer(modelo)
    return modelo, tokenizer


def _generar_respuesta_atomica(modelo, tokenizer, mensaje: str, *, max_tokens_nuevos: int) -> str:
    """Una respuesta, decodificación VORAZ (`do_sample=False`) -- el mismo
    patrón de generación ya medido en `sondear_respondedor.medir()` (modelo,
    tokenizer, `apply_chat_template`, bucle `generate_next_token`). No se
    llama a `sondear_respondedor.medir()` directamente porque esa función
    mide UNA pregunta fija sobre N textos con instrumentación de latencia;
    aquí hace falta la misma llamada de generación para preguntas DISTINTAS
    por término y fila -- lo que se reutiliza es el modelo, el tokenizer y el
    patrón, no una función de una sola pregunta."""
    import onnxruntime_genai as og  # noqa: PLC0415 -- perezoso a propósito

    mensajes = json.dumps([{"role": "user", "content": mensaje}])
    prompt = tokenizer.apply_chat_template(messages=mensajes)
    tokens_prompt = tokenizer.encode(prompt)

    params = og.GeneratorParams(modelo)
    params.set_search_options(max_length=int(len(tokens_prompt)) + max_tokens_nuevos,
                              do_sample=False, temperature=1.0)
    generador = og.Generator(modelo, params)
    generador.append_tokens(tokens_prompt)
    tokens_generados: list[int] = []
    while not generador.is_done() and len(tokens_generados) < max_tokens_nuevos:
        generador.generate_next_token()
        tokens_generados.append(int(generador.get_next_tokens()[0]))
    return tokenizer.decode(tokens_generados) if tokens_generados else ""


# ---------------------------------------------------------------------------
# una tarea entera (A o B)
# ---------------------------------------------------------------------------

def evaluar_tarea_respondedor(
    nombre: str, *, tokenizador_c1, generar_respuesta: Callable[[str], str],
    cache: CacheRespondedor, semilla: int = mc.SEMILLA,
) -> dict[str, Any]:
    """La condición (3) de la tarea `nombre` ("A" o "B"), con el veredicto
    frente a (0) y la diferencia descrita frente a (2).

    `generar_respuesta` es lo que de verdad llama al modelo -- inyectable a
    propósito: la prueba de punta a punta le pasa un respondedor FALSO, sin
    tocar `onnxruntime_genai` en absoluto.
    """
    cfg = mc.TAREAS[nombre]
    particiones = mc.leer_tarea(nombre)
    train, val, test = particiones["train"], particiones["dev"], particiones["test"]
    metric_id = cfg["metrica"]
    todas = train + val + test

    digest = mc._digest_particion([f["row_id"] for f in train + val], [f["row_id"] for f in test])

    # ---- (0) recalculada AQUÍ, con el mismo código y semilla que medir_c30
    # (no se reutiliza el JSON de resultado_c30.json: solo guarda el número
    # puntual, no la Muestra completa que `comparar_candidatos` necesita) ----
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
    terminos_info = _terminos_top_chi2([f["texto"] for f in train], [f["target"] for f in train])
    terminos = [t["termino"] for t in terminos_info]

    # ---- preguntas por fila, con caché y truncado como C1 ----
    t_preguntas0 = time.perf_counter()
    n_si = n_no = n_faltante = 0
    respuestas_por_termino: dict[str, dict[str, int]] = {
        t: {"si": 0, "no": 0, "faltante": 0} for t in terminos
    }
    for fila in todas:
        texto_truncado = fila.get("_texto_truncado_c1")
        if texto_truncado is None:
            texto_truncado = truncar_como_embedding_c1(fila["texto"], tokenizador_c1)
            fila["_texto_truncado_c1"] = texto_truncado
        for indice, termino in enumerate(terminos):
            entrada = cache.obtener(nombre, fila["row_id"], termino)
            if entrada is None:
                pregunta = PLANTILLA_PREGUNTA.format(termino=termino)
                mensaje = PLANTILLA_MENSAJE.format(pregunta=pregunta, texto=texto_truncado)
                respuesta_bruta = generar_respuesta(mensaje)
                valor = analizar_respuesta(respuesta_bruta)
                entrada = {"respuesta_bruta": respuesta_bruta, "valor": valor}
                cache.fijar(nombre, fila["row_id"], termino, entrada)
            valor = entrada["valor"]
            fila[f"pregunta_{indice}"] = float(valor) if valor is not None else float("nan")
            if valor == 1:
                n_si += 1
                respuestas_por_termino[termino]["si"] += 1
            elif valor == 0:
                n_no += 1
                respuestas_por_termino[termino]["no"] += 1
            else:
                n_faltante += 1
                respuestas_por_termino[termino]["faltante"] += 1
        cache.persistir()  # tras la FILA completa (sus 5 preguntas), no tras cada pregunta

    tiempos_preguntas = {"segundos": time.perf_counter() - t_preguntas0,
                        "n_respuestas": n_si + n_no + n_faltante,
                        "n_si": n_si, "n_no": n_no, "n_faltante": n_faltante,
                        "proporcion_faltante": (n_faltante / (n_si + n_no + n_faltante))
                        if (n_si + n_no + n_faltante) else None}

    # ---- (3): las 5 columnas + las "otras columnas", en el MISMO LightGBM ----
    columnas_pregunta = tuple(f"pregunta_{i}" for i in range(len(terminos)))
    columnas3 = tuple(cfg["otras_columnas"]) + columnas_pregunta
    spec3 = mc._spec(nombre, cfg, columnas3, candidato="condicion_3_respondedor")
    p_train3 = mc._particion_tabular(train, columnas3, tipo=cfg["tipo"], con_target=True)
    p_val3 = mc._particion_tabular(val, columnas3, tipo=cfg["tipo"], con_target=True)
    p_test3 = mc._particion_tabular(test, columnas3, tipo=cfg["tipo"], con_target=True)
    muestra3, tiempos3 = mc._ajustar_y_predecir_lightgbm(
        p_train3, p_val3, p_test3, spec3, candidate="condicion_3_respondedor",
        split_plan_digest=digest, semilla=semilla)
    tiempos3["preguntas_al_modelo"] = tiempos_preguntas
    metrica_3 = mc._metrica_puntual(metric_id, muestra3)

    comparacion_3_vs_0 = mc._comparar(metric_id, muestra3, muestra0, semilla=semilla,
                                      protocolo_candidato=digest, protocolo_baseline=digest)

    return {
        "tarea": nombre, "metrica": metric_id,
        "n_train": len(train), "n_dev": len(val), "n_test": len(test),
        "terminos": terminos_info,
        "condiciones": {
            "3_respondedor": {
                "columnas": list(columnas3), "metrica_puntual": metrica_3,
                "tiempos": tiempos3, "split_plan_digest": digest,
            },
        },
        "comparaciones": {
            "3_vs_0": comparacion_3_vs_0,
            "3_vs_2_diferencia_descrita": {
                "metrica_puntual_2_tfidf_lineal": metrica_2,
                "metrica_puntual_3_respondedor": metrica_3,
                "diferencia_3_menos_2": (None if metrica_3 is None or metrica_2 is None
                                        else metrica_3 - metrica_2),
                "nota": "DESCRITA, sin veredicto -- esta comparación no estaba en el "
                        "pre-registro (registro sellado 26-09).",
            },
        },
    }


# ---------------------------------------------------------------------------
# --estimar: la aritmética LITERAL del registro (5 x filas x 1,2s)
# ---------------------------------------------------------------------------

def _tasa_s_por_respuesta() -> tuple[float, str]:
    """Prefiere la medida REAL de `sondear_respondedor.py` si ya se corrió
    (medir, no suponer); si no, cae a la constante del registro sellado."""
    ruta = RAIZ / "resultado_sonda_respondedor.json"
    if ruta.is_file():
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            media = (datos.get("latencia_por_respuesta_s") or {}).get("media")
            if media:
                return float(media), f"medida real de {ruta.name} (latencia_por_respuesta_s.media)"
        except (json.JSONDecodeError, OSError):
            pass
    return (TASA_S_POR_RESPUESTA_DEL_REGISTRO,
            "constante del registro sellado 26-09 ('5 x filas x 1,2 s'); "
            f"{ruta.name} no existe todavía en este árbol")


def estimar() -> dict[str, Any]:
    tasa, origen_tasa = _tasa_s_por_respuesta()
    resultado: dict[str, Any] = {
        "aritmetica": f"{N_PREGUNTAS} preguntas x filas x {tasa:.4f} s/respuesta (registro sellado 26-09)",
        "tasa_s_por_respuesta": tasa, "origen_de_la_tasa": origen_tasa,
        "n_preguntas": N_PREGUNTAS, "tareas": {},
    }
    total_s = 0.0
    for nombre in TAREAS_A_MEDIR:
        try:
            particiones = mc.leer_tarea(nombre)
        except SystemExit as e:
            resultado["tareas"][nombre] = {"error": str(e)}
            continue
        n_total = sum(len(v) for v in particiones.values())
        segundos = N_PREGUNTAS * n_total * tasa
        total_s += segundos
        resultado["tareas"][nombre] = {
            "n_total": n_total, "estimado_s": round(segundos, 1), "estimado_h": round(segundos / 3600, 3),
        }
    for nombre, motivo in TAREAS_OMITIDAS.items():
        resultado["tareas"][nombre] = {"omitida": True, "motivo": motivo}
    resultado["total_s_a_y_b"] = round(total_s, 1)
    resultado["total_h_a_y_b"] = round(total_s / 3600, 3)
    return resultado


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimar", action="store_true", help="coste con la aritmética del registro, sin medir")
    ap.add_argument("--solo", choices=TAREAS_A_MEDIR, help="mide solo esta tarea (A o B)")
    ap.add_argument("--modelo-local", type=Path, default=sr.MODELO_LOCAL_POR_OMISION)
    ap.add_argument("--hilos", type=int, default=HILOS)
    ap.add_argument("--max-tokens-nuevos", type=int, default=MAX_TOKENS_NUEVOS)
    ap.add_argument("--cache", type=Path, default=CACHE_POR_OMISION)
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_OMISION)
    ns = ap.parse_args()

    if ns.estimar:
        tabla = estimar()
        destino = RAIZ / "estimacion_c30_respondedor.json"
        _escribir_json_atomico(destino, tabla)
        print(json.dumps(tabla, indent=2, ensure_ascii=False))
        print(f"\nescrito en {destino}", file=sys.stderr)
        return 0

    tareas_a_medir = [ns.solo] if ns.solo else list(TAREAS_A_MEDIR)
    cache = CacheRespondedor(ns.cache)
    tokenizador_c1 = _cargar_tokenizador_de_c1()
    modelo, tokenizer = _cargar_modelo_y_tokenizer(ns.modelo_local, hilos=ns.hilos)

    def generar_respuesta(mensaje: str) -> str:
        return _generar_respuesta_atomica(modelo, tokenizer, mensaje, max_tokens_nuevos=ns.max_tokens_nuevos)

    salida: dict[str, Any] = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0, REGISTRO SELLADO de la "
                    "condición (3) -- 26-09",
        "semilla": mc.SEMILLA, "margen_equivalencia": mc.MARGEN_EQUIVALENCIA,
        "n_preguntas": N_PREGUNTAS, "plantilla_pregunta": PLANTILLA_PREGUNTA,
        "plantilla_mensaje": PLANTILLA_MENSAJE,
        "modelo": {
            "candidato": "Qwen2.5-0.5B-Instruct", "licencia": "Apache-2.0",
            "conversion_onnx_genai_de": sr.MODELO_REPO, "subcarpeta": sr.MODELO_SUBCARPETA,
            "camino_de_ejecucion": "onnxruntime-genai (CPU, int4 RTN block-32)",
            "ruta_local": str(ns.modelo_local), "hilos": ns.hilos,
            "max_tokens_nuevos": ns.max_tokens_nuevos, "decodificacion": "voraz (do_sample=False)",
        },
        "truncado": {
            "proveedor_embedding_c1": PROVEEDOR_EMBEDDING_C1,
            "max_longitud_tokens": tokenizador_c1.max_longitud,
            "metodo": "prefijo mínimo de caracteres cuya tokenización (Unigram real de C1) alcanza "
                      "el mismo número de tokens que el texto completo truncado",
        },
        "script_sha256": mc.sha256_de(Path(__file__)),
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tareas": {k: {"omitida": True, "motivo": v} for k, v in TAREAS_OMITIDAS.items()},
    }
    if ns.salida.is_file():
        try:
            previo = json.loads(ns.salida.read_text(encoding="utf-8"))
            salida["tareas"].update(previo.get("tareas", {}))
        except (json.JSONDecodeError, OSError):
            pass

    for nombre in tareas_a_medir:
        print(f"=== midiendo condición (3), tarea {nombre} ===", file=sys.stderr)
        t0 = time.perf_counter()
        resultado = evaluar_tarea_respondedor(nombre, tokenizador_c1=tokenizador_c1,
                                              generar_respuesta=generar_respuesta, cache=cache)
        resultado["duracion_total_s"] = time.perf_counter() - t0
        salida["tareas"][nombre] = resultado
        _escribir_json_atomico(ns.salida, salida)
        print(f"--- tarea {nombre}: {resultado['duracion_total_s']:.1f}s, escrito en {ns.salida} ---",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
