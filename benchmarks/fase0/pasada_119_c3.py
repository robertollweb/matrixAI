#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C3 — el motor nuevo (`matrixai.dense.tabm_cpu`, TabM + PLR) sobre los
32 conjuntos NO sellados de Fase 0, comparado contra la densa v2
(`pasada_v2_113_resultado.json`) en los MISMOS conjuntos.

Implementa el bloque `veredicto` de `protocolo_119_v4.json` (contrato 119,
`documentacion/119_LA_RED_DENSA_NUEVA_CONTRACT.md`): corre
`matrixai_engines.motores.densa_tabm.MotorDensaTabM()` — SIN parámetros, a
diferencia de la palanca de 118 — con las MISMAS particiones, semillas,
repeticiones y presupuesto por cubo que registra el protocolo v2
(`protocolo_exploratorio_v2.json`, que es el que DE VERDAD se lee). El v2 y
el v4 son idénticos en `particion`, `presupuesto`, `regla_de_cierre` y
`datasets`: se COMPRUEBA al arrancar (`exigir_los_protocolos_encadenados`,
que para la pasada si divergen) y lo fija una prueba bloque a bloque. El
«campo» se compone sustituyendo, SOLO en los conjuntos medidos aquí, la densa
v2 (`matrixai.dense.torch_cpu`) por el motor nuevo.

**LOS 8 SELLADOS SE NIEGAN.** El protocolo v4 (`veredicto.regla_de_subida`)
reserva los sellados para C4: un sellado en `--solo` para la pasada con
`SystemExit` (en 118 solo se avisaba).

LO QUE SE REPARÓ TRAS LA AUDITORÍA 2 (29-09), por hallazgo:

* **I2, los fallos en el veredicto** (decisión del supervisor, registrada en
  la enmienda 2 del protocolo, que la pasada REAL exige con su cadena entera;
  ver `COMO_SE_CUENTAN_LOS_FALLOS` y `exigir_la_enmienda_2`):
  (a) condición 1: un conjunto con CUALQUIER intento del motor nuevo sin
  medida —fallido, AUSENTE del artefacto o completado sin la métrica de
  cierre— cuenta como perdido; (b) condición 2: fallos del motor nuevo con la
  densa v2 completa = INFERIORIDAD, y donde la densa v2 falló TODOS sus
  intentos (Allstate, KDDCup09) = «sin comparación», que solo cuenta en la
  condición 1; (c) la emparejada, solo sobre repeticiones y pliegues
  completos en los DOS; (d) `veredicto_por_conjunto` lleva los fallos.
* **I4, el digest de la caché**: cubre el CIERRE ESTÁTICO de imports de este
  guion dentro de los dos repos (190 ficheros, medido; un intento de kc2 en
  proceso carga 124 y los 124 están dentro), los tres ficheros de protocolo
  que se leen, y las versiones de python, numpy, scikit-learn y torch. Lo
  que NO cubre va escrito en el propio resultado
  (`lo_que_no_cubre_el_digest`), y el sha256 del ARFF va en la clave de cada
  registro.
* **I6, las guardias**: al arrancar, el motor tiene que usar k=8,
  d_block=256, n_blocks=2 (sus CONSTANTES, no un comentario); el v4 cuadra
  con su digest y la enmienda 1 cita ese digest; cada intento registra
  `arquitectura` e `hiperparametros` y la pasada para si la arquitectura
  declarada no es la de la enmienda; la partición y la lectura de cada
  conjunto se comparan con las de la v2 ANTES de su primer intento.
* **I7, que una noche no se pierda**: la salida se FUSIONA con la que ya
  hay; `--solo`, `--humo` y `--estimar` escriben por omisión en ficheros
  aparte y se niegan a escribir en un resultado de otro tipo; `--salida`
  admite una ruta absoluta fuera del árbol (la cola nocturna borra sus
  worktrees: el caché tiene que vivir fuera para que la noche siguiente
  continúe).
* **I1, la estimación**: mide UN intento por conjunto de una lista con más
  de un conjunto por cubo (anchos incluidos), con la preparación del padre
  aparte, y dice si cabe en la ventana nocturna o de día — con 1 proceso,
  que es lo que este guion corre.

LO QUE SE REPARÓ TRAS LA AUDITORÍA 3 (30-09), por hallazgo:

* **I2, la receta entera**: la guardia compara con el protocolo (v4 +
  enmiendas 1 y 2, `receta_que_fija_el_protocolo`) TODAS las constantes de
  la receta (`CONSTANTES_DE_LA_RECETA`), y MIDE un ajuste pequeño con
  espías (`medir_la_receta_del_motor`): la clase del optimizador (AdamW), su
  lr y su weight_decay, los argumentos de la red, el recorte, el lote, la
  semilla y el plazo. Cada intento se compara con la receta ENTERA que
  declara (`DECLARADO_POR_EL_INTENTO`), no solo k/d_block/n_blocks.
* **I4, los reintentos**: un intento sin medida que se reintenta no
  desaparece: queda en `intentos_reintentados` (qué, cuándo, con qué error)
  y el veredicto lista `conjuntos_con_intentos_reintentados`.
* **I1, la orden de encolado**: `para_encolar` da la orden de DÍA con la
  estimación entera y un tope ×1,25 (`MARGEN_DEL_TOPE_SOBRE_LA_ESTIMACION`),
  la hora límite para lanzarla, y la de noche desde el fin de la suite
  MEDIDO, con una orden por noche y COMMITS= si no cabe.
* **I3, la parada temprana**: el resultado declara
  (`parada_temprana_declarada`) que en clasificación NO es como la fuente
  (la fuente para con accuracy; el motor, con la log-loss del ensamblado).
* **M3, la memoria**: `--estimar` declara el pico del ÁRBOL de procesos
  muestreado, no maxrss del padre + el de los hijos (que hereda el padre).

QUÉ SE REUTILIZA, Y DE DÓNDE — nada se copia a mano:

* catálogo, lectura, partición y guardia de CPU: `pasada_114c6_ensamblado`
  y `pasada_amplia_101_c5` (la MISMA partición, semillas y estratificación
  que la v2);
* el intento en subproceso con su tope: `matrixai_engines.subproceso`;
* procedencia, sellado y escritura: `pasada_exploratoria_101_c3`;
* el campo, la emparejada con bootstrap y la aritmética de la regla de
  subida: `pasada_118_palanca` (importadas TAL CUAL; aquí solo se añade,
  por fuera, cómo cuentan los fallos);
* la regla de cierre: `protocolo.aplicar_regla_de_cierre`.

CÓMO SE LANZA:

    python3 benchmarks/fase0/pasada_119_c3.py --estimar --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c3.py --humo --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c3.py --solo diabetes --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c3.py --salida <ruta absoluta, fuera del árbol>
        # LA PASADA ENTERA sobre los 32 no sellados -- horas, a la cola
        # nocturna (decisión del 25-09), nunca a mano.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import hashlib
import json
import math
import random
import re
import resource
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_118_palanca as p118  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa_tabm import MotorDensaTabM  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.procedencia import versiones_de_bibliotecas  # noqa: E402
from matrixai_engines.subproceso import (MARGEN_POR_DEFECTO_SEGUNDOS,  # noqa: E402
                                         ejecutar_intento_aislado)

RUTA_DEL_PROTOCOLO_V4 = _AQUI / "protocolo_119_v4.json"
RUTA_DE_LA_ENMIENDA_1 = _AQUI / "protocolo_119_v4_enmienda_1.json"
#: La enmienda 2 (núcleo cb91e94): cómo cuentan los fallos (I2), la parada
#: temprana y la paciencia. La pasada REAL no arranca sin ella y con su cadena
#: entera (`exigir_la_enmienda_2`); `--solo`/`--humo`/`--estimar` sí, diciéndolo.
#: Su sha256 va en la PROCEDENCIA; en el digest de la caché NO (no cambia lo
#: que ejecuta un intento: solo cómo se cuenta, y eso se recalcula cada vez).
RUTA_DE_LA_ENMIENDA_2 = _AQUI / "protocolo_119_v4_enmienda_2.json"
#: El protocolo que DE VERDAD se lee para medir (`c3.protocolo_registrado()`
#: tras `c6.preparar_protocolo_v2()`, y el catálogo de `fijar_el_catalogo_v2`).
RUTA_DEL_PROTOCOLO_V2 = c6.RUTA_DEL_PROTOCOLO_V2
RUTA_V2_RESULTADO = p118.RUTA_V2_RESULTADO  # pasada_v2_113_resultado.json

#: EL RESULTADO REAL del corte. Ni `--solo` ni `--humo` escriben aquí nunca.
RUTA_DEL_RESULTADO = _AQUI / "pasada_119_c3_resultado.json"
RUTA_DE_LA_ESTIMACION = _AQUI / "estimacion_pasada_119_c3.json"
SALIDA_POR_OMISION = {
    "pasada": RUTA_DEL_RESULTADO,
    "solo": _AQUI / "pasada_119_c3_solo_resultado.json",
    "humo": _AQUI / "pasada_119_c3_humo_resultado.json",
    "estimar": RUTA_DE_LA_ESTIMACION,
}

#: El motor NUEVO de este corte — SIN parámetros. Se comprueba al importar
#: que el nombre no ha derivado del que este guion tiene escrito.
NOMBRE_MOTOR_NUEVO = "matrixai.dense.tabm_cpu"
assert MotorDensaTabM().nombre == NOMBRE_MOTOR_NUEVO, (
    f"MotorDensaTabM().nombre es {MotorDensaTabM().nombre!r}, y este guion tiene escrito "
    f"{NOMBRE_MOTOR_NUEVO!r} -- actualizar la constante, no ignorar la discrepancia")

#: La densa VIEJA (v2), la clave que `campo_de_la_comparacion` sustituye.
NOMBRE_DENSA_V2 = p118.NOMBRE_DENSA

#: Los bloques que el v2 (el que se lee) y el v4 (el que se cita) tienen que
#: traer IDÉNTICOS para que medir con el v2 sea medir con el v4.
BLOQUES_QUE_LA_V2_Y_EL_V4_COMPARTEN = ("particion", "presupuesto", "regla_de_cierre", "datasets")

#: LA CONFIGURACIÓN DE LA ENMIENDA 1 (`que_cambia`: «k=8 y d_block=256
#: (n_blocks=2)»), y la constante del motor que la fija. La guardia lee las
#: CONSTANTES del módulo del motor, y cada intento, la arquitectura que el
#: motor DECLARA haber usado.
CONFIGURACION_DE_LA_ENMIENDA_1 = {"k": 8, "d_block": 256, "n_blocks": 2}
CONSTANTES_DEL_MOTOR = {"k": "K_CABEZAS", "d_block": "D_BLOCK", "n_blocks": "N_BLOCKS"}

#: LA RECETA ENTERA (reparación 3, hallazgo I2 de la auditoría 3): cada valor
#: que el protocolo (v4 + enmiendas 1 y 2) fija y el motor usa, con la
#: CONSTANTE del motor que lo lleva. La guardia compara cada una con lo que el
#: protocolo dice, y además MIDE un ajuste pequeño con espías
#: (`medir_la_receta_del_motor`): el optimizador no tiene constante —cambiar
#: AdamW por Adam en el código no lo ve ninguna— y una constante puede estar
#: bien y la llamada pasar otra cosa.
CONSTANTES_DE_LA_RECETA = {
    "k": "K_CABEZAS", "d_block": "D_BLOCK", "n_blocks": "N_BLOCKS", "dropout": "DROPOUT",
    "d_embedding": "D_EMBEDDING", "n_frequencies": "N_FREQUENCIES",
    "frequency_init_scale": "FREQUENCY_INIT_SCALE",
    "tasa_de_aprendizaje": "TASA_DE_APRENDIZAJE", "weight_decay": "WEIGHT_DECAY",
    "recorte_de_gradiente": "NORMA_MAXIMA_DE_RECORTE", "lote": "LOTE_MAXIMO",
    "paciencia": "PACIENCIA",
    "fraccion_del_presupuesto_para_entrenar": "FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR",
    "semilla_del_ruido_de_cuantiles": "SEMILLA_DEL_RUIDO_DE_CUANTILES",
}
#: Lo que cada intento DECLARA (`arquitectura` e `hiperparametros` del
#: predictor) y la clave de la receta con la que se compara.
DECLARADO_POR_EL_INTENTO = {
    ("arquitectura", "k"): "k", ("arquitectura", "d_block"): "d_block",
    ("arquitectura", "n_blocks"): "n_blocks", ("arquitectura", "dropout"): "dropout",
    ("arquitectura", "d_embedding"): "d_embedding",
    ("arquitectura", "n_frequencies"): "n_frequencies",
    ("arquitectura", "frequency_init_scale"): "frequency_init_scale",
    ("hiperparametros", "optimizador"): "optimizador",
    ("hiperparametros", "tasa_de_aprendizaje"): "tasa_de_aprendizaje",
    ("hiperparametros", "weight_decay"): "weight_decay",
    ("hiperparametros", "recorte_de_gradiente"): "recorte_de_gradiente",
    ("hiperparametros", "parada_temprana", "paciencia"): "paciencia",
    ("hiperparametros", "parada_temprana", "para_tras_epocas_sin_mejora"):
        "para_tras_epocas_sin_mejora",
    ("hiperparametros", "parada_temprana", "metrica"): "metrica_de_parada",
    ("hiperparametros", "fraccion_del_presupuesto_para_entrenar"):
        "fraccion_del_presupuesto_para_entrenar",
}
#: Lo que solo fija la enmienda 2: sin ella (`--solo`/`--humo`/`--estimar`
#: corren sin ella, diciéndolo) no hay con qué compararlo y se dice.
SOLO_EN_LA_ENMIENDA_2 = ("para_tras_epocas_sin_mejora", "metrica_de_parada")

SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0


# ---------------------------------------------------------------------------
# 1. LOS CONJUNTOS: los 32 NO sellados -- los 8 sellados se NIEGAN (son de C4)
# ---------------------------------------------------------------------------

def datasets_de_c3(todos: list, no_sellados: list, *, solo: str | None) -> tuple[list, list | None]:
    """Los conjuntos que esta pasada mide: los 32 no sellados, o el
    subconjunto de `--solo` -- SIEMPRE dentro de los no sellados."""
    if not solo:
        return list(no_sellados), None
    pedidos = [n.strip() for n in solo.split(",") if n.strip()]
    por_nombre = {d.nombre: d for d in todos}
    desconocidos = [n for n in pedidos if n not in por_nombre]
    if desconocidos:
        raise SystemExit(f"--solo nombra conjuntos que no están en el protocolo v2: "
                         f"{desconocidos}")
    sellados_pedidos = [n for n in pedidos if por_nombre[n].sellado]
    if sellados_pedidos:
        raise SystemExit(
            f"--solo pide conjunto(s) SELLADO(S) {sellados_pedidos}: 119-C3 mide SOLO los "
            f"32 no sellados -- los 8 sellados son de C4 (regla_de_subida del protocolo v4: "
            f"«sube a confirmación en los sellados... si cumple las DOS condiciones sobre "
            f"los 32 no sellados»). Se niega en vez de correr «solo como prueba»")
    return [por_nombre[n] for n in pedidos], pedidos


# ---------------------------------------------------------------------------
# 2. LAS GUARDIAS (I6): los protocolos encadenados y la configuración del motor
# ---------------------------------------------------------------------------

def _leer_json(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def autodigest(payload: dict) -> str:
    """El digest que el v4 y su enmienda llevan DENTRO (`digest_sha256`):
    sha256 del JSON sin ese campo, claves ordenadas, sin espacios y sin
    escapar. Comprobado sobre los dos ficheros: reproduce los dos valores."""
    sin_el_campo = {k: v for k, v in payload.items() if k != "digest_sha256"}
    return hashlib.sha256(json.dumps(sin_el_campo, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonico(valor) -> str:
    return json.dumps(valor, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def exigir_los_protocolos_encadenados() -> dict:
    """PARA la pasada si la cadena v4 -> enmienda 1 no cuadra, o si el v2
    (el que se lee) y el v4 (el que se cita) divergen en lo que importa.

    1. el v4 cuadra con su propio `digest_sha256`;
    2. la enmienda 1 cuadra con el suyo;
    3. la enmienda 1 dice ser DE ese v4 (`de.digest_sha256` y `de.protocolo`);
    4. `particion`, `presupuesto`, `regla_de_cierre` y `datasets` son
       idénticos en el v2 y en el v4 (JSON canónico).
    """
    v4 = _leer_json(RUTA_DEL_PROTOCOLO_V4)
    enmienda = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    v2 = _leer_json(RUTA_DEL_PROTOCOLO_V2)
    problemas = []
    if autodigest(v4) != v4.get("digest_sha256"):
        problemas.append(f"el v4 ({RUTA_DEL_PROTOCOLO_V4.name}) NO cuadra con su propio "
                         f"digest_sha256: se ha tocado después de registrarlo")
    if autodigest(enmienda) != enmienda.get("digest_sha256"):
        problemas.append(f"la enmienda 1 ({RUTA_DE_LA_ENMIENDA_1.name}) NO cuadra con su propio "
                         f"digest_sha256")
    de = enmienda.get("de") or {}
    if de.get("digest_sha256") != v4.get("digest_sha256"):
        problemas.append(f"la enmienda 1 dice ser del protocolo con digest "
                         f"{str(de.get('digest_sha256'))[:16]} y el v4 es "
                         f"{str(v4.get('digest_sha256'))[:16]}")
    if de.get("protocolo") != v4.get("version_protocolo"):
        problemas.append(f"la enmienda 1 dice ser de {de.get('protocolo')!r} y el v4 es "
                         f"{v4.get('version_protocolo')!r}")
    bloques = {}
    for bloque in BLOQUES_QUE_LA_V2_Y_EL_V4_COMPARTEN:
        iguales = bloque in v2 and bloque in v4 and _canonico(v2[bloque]) == _canonico(v4[bloque])
        bloques[bloque] = iguales
        if not iguales:
            problemas.append(f"el bloque {bloque!r} NO es idéntico en {RUTA_DEL_PROTOCOLO_V2.name} "
                             f"(el que se lee) y en {RUTA_DEL_PROTOCOLO_V4.name} (el que se cita)")
    if problemas:
        raise SystemExit("la cadena de protocolos no cuadra; no se mide:\n  - "
                         + "\n  - ".join(problemas))
    return {
        "v4": {"version": v4["version_protocolo"], "digest_sha256_declarado": v4["digest_sha256"],
               "cuadra_con_su_digest": True},
        "enmienda_1": {"enmienda": enmienda.get("enmienda"),
                       "digest_sha256_declarado": enmienda["digest_sha256"],
                       "cuadra_con_su_digest": True, "de": de, "cita_el_v4": True},
        "bloques_identicos_v2_v4": bloques,
        "como_se_comprueba": ("autodigest = sha256 del JSON sin `digest_sha256`, claves "
                              "ordenadas, separadores compactos, sin escapar; los bloques, "
                              "por su JSON canónico"),
    }


def cadena_de_la_enmienda_2() -> dict:
    """La enmienda 2 (núcleo cb91e94: fija cómo cuentan los fallos, la parada
    temprana y la paciencia), LEÍDA DEL DISCO en cada ejecución, y si su
    cadena cuadra: su propio `digest_sha256`, `de.digest_sha256` = el del v4
    y `de.enmienda_anterior.digest_sha256` = el de la enmienda 1. No decide
    nada por sí sola: `exigir_la_enmienda_2` es quien para la pasada real."""
    base = {"ruta": RUTA_DE_LA_ENMIENDA_2.name}
    if not RUTA_DE_LA_ENMIENDA_2.exists():
        return {**base, "presente": False, "cuadra": False,
                "problemas": [f"{RUTA_DE_LA_ENMIENDA_2.name} no está registrada"]}
    enmienda_2 = _leer_json(RUTA_DE_LA_ENMIENDA_2)
    v4 = _leer_json(RUTA_DEL_PROTOCOLO_V4)
    enmienda_1 = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    de = enmienda_2.get("de") or {}
    anterior = de.get("enmienda_anterior") or {}
    problemas = []
    if autodigest(enmienda_2) != enmienda_2.get("digest_sha256"):
        problemas.append("NO cuadra con su propio digest_sha256: se ha tocado después de "
                         "registrarla")
    if de.get("digest_sha256") != v4.get("digest_sha256"):
        problemas.append(f"dice ser del protocolo {str(de.get('digest_sha256'))[:16]} y el v4 es "
                         f"{str(v4.get('digest_sha256'))[:16]}")
    if anterior.get("digest_sha256") != enmienda_1.get("digest_sha256"):
        problemas.append(f"dice seguir a la enmienda {str(anterior.get('digest_sha256'))[:16]} y "
                         f"la enmienda 1 es {str(enmienda_1.get('digest_sha256'))[:16]}")
    return {**base, "presente": True, "cuadra": not problemas, "problemas": problemas,
            "enmienda": enmienda_2.get("enmienda"),
            "sha256_del_fichero": hashlib.sha256(RUTA_DE_LA_ENMIENDA_2.read_bytes()).hexdigest(),
            "digest_sha256_declarado": enmienda_2.get("digest_sha256"), "de": de}


def exigir_la_enmienda_2(tipo: str) -> dict:
    """La pasada REAL no arranca sin la enmienda 2 registrada y con su cadena
    entera: sus reglas del veredicto tienen que estar escritas ANTES de medir.
    `--solo`, `--humo` y `--estimar` no miden el corte: corren sin ella,
    diciéndolo en voz alta y en el resultado."""
    cadena = cadena_de_la_enmienda_2()
    if cadena["cuadra"]:
        return cadena
    motivo = "; ".join(cadena["problemas"])
    if tipo == "pasada":
        raise SystemExit(f"la pasada real exige la enmienda 2 del protocolo v4 con su cadena "
                         f"entera, y {motivo}. No se mide")
    print(f"AVISO: sin la enmienda 2 en regla ({motivo}): esta ejecución ({tipo}) no mide el "
          f"corte y corre igual, con los criterios de fallos del guion", flush=True)
    return cadena


def configuracion_que_declara_la_enmienda(texto: str) -> dict:
    """k, d_block y n_blocks tal como los escribe `que_cambia` de la
    enmienda 1 («k=8 y d_block=256 (n_blocks=2)»), o `None` si no los dice."""
    patrones = {"k": r"(?<![A-Za-z0-9_])k\s*=\s*(\d+)",
                "d_block": r"(?<![A-Za-z0-9_])d_block\s*=\s*(\d+)",
                "n_blocks": r"(?<![A-Za-z0-9_])n_blocks\s*=\s*(\d+)"}
    salida = {}
    for clave, patron in patrones.items():
        m = re.search(patron, texto or "")
        salida[clave] = int(m.group(1)) if m else None
    return salida


def _numero_en(texto: str, patron: str):
    """El número que `patron` (un grupo) encuentra en un texto del protocolo,
    con la coma decimal que escribe el v4 («0,75»), o `None`."""
    m = re.search(patron, texto or "")
    if not m:
        return None
    crudo = m.group(1).replace(",", ".")
    return float(crudo) if "." in crudo else int(crudo)


def receta_que_fija_el_protocolo() -> dict:
    """Cada valor de la receta TAL COMO LO ESCRIBEN los tres ficheros del
    protocolo, leídos del disco en cada llamada, y de dónde sale cada uno.
    `valor` None = el fichero no lo dice: la guardia para (salvo lo que solo
    fija la enmienda 2 cuando no está, `SOLO_EN_LA_ENMIENDA_2`). k, d_block y
    n_blocks son los de la enmienda 1 (el v4 traía los de la fuente, 32/512);
    lo demás, el v4, que la enmienda 1 deja igual («no_cambia»)."""
    v4 = _leer_json(RUTA_DEL_PROTOCOLO_V4)
    e1 = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    e2 = _leer_json(RUTA_DE_LA_ENMIENDA_2) if RUTA_DE_LA_ENMIENDA_2.exists() else {}
    arq = v4.get("arquitectura") or {}
    plr = (arq.get("embeddings_numericas") or {}).get("parametros_por_omision_del_constructor") or {}
    tabm = (arq.get("backbone_ensamblado") or {}).get("parametros_por_omision_de_TabM_make") or {}
    ruido = (((arq.get("preprocesado_numericas") or {}).get("ruido_antes_de_ajustar") or {})
             .get("distribucion", ""))
    receta = v4.get("receta_entrenamiento") or {}
    se_fija = e2.get("se_fija") or {}
    de_la_e1 = configuracion_que_declara_la_enmienda(e1.get("que_cambia", ""))
    v4n, e1n, e2n = (RUTA_DEL_PROTOCOLO_V4.name, RUTA_DE_LA_ENMIENDA_1.name,
                     RUTA_DE_LA_ENMIENDA_2.name)
    del_constructor = f"{v4n}: arquitectura.embeddings_numericas.parametros_por_omision_del_constructor"
    return {
        **{k: {"valor": de_la_e1[k], "de": f"{e1n}: que_cambia"} for k in ("k", "d_block", "n_blocks")},
        "dropout": {"valor": tabm.get("dropout"),
                    "de": f"{v4n}: arquitectura.backbone_ensamblado.parametros_por_omision_de_"
                          f"TabM_make.dropout (la enmienda 1 solo cambia k y d_block)"},
        "d_embedding": {"valor": plr.get("d_embedding"), "de": f"{del_constructor}.d_embedding "
                        f"(también el de las categóricas: embeddings_categoricas.decision_del_119)"},
        "n_frequencies": {"valor": plr.get("n_frequencies"), "de": f"{del_constructor}.n_frequencies"},
        "frequency_init_scale": {"valor": plr.get("frequency_init_scale"),
                                 "de": f"{del_constructor}.frequency_init_scale"},
        "optimizador": {"valor": receta.get("optimizador"),
                        "de": f"{v4n}: receta_entrenamiento.optimizador"},
        "tasa_de_aprendizaje": {"valor": receta.get("learning_rate"),
                                "de": f"{v4n}: receta_entrenamiento.learning_rate"},
        "weight_decay": {"valor": receta.get("weight_decay"),
                         "de": f"{v4n}: receta_entrenamiento.weight_decay"},
        "recorte_de_gradiente": {"valor": (receta.get("recorte_de_gradiente") or {}).get("norma_maxima"),
                                 "de": f"{v4n}: receta_entrenamiento.recorte_de_gradiente.norma_maxima"},
        "lote": {"valor": (receta.get("batch_size") or {}).get("valor_fuente"),
                 "de": f"{v4n}: receta_entrenamiento.batch_size (aplicado: min(256, filas de train))"},
        "paciencia": {"valor": (receta.get("parada_temprana") or {}).get("paciencia_epocas"),
                      "de": f"{v4n}: receta_entrenamiento.parada_temprana.paciencia_epocas"},
        "fraccion_del_presupuesto_para_entrenar": {
            "valor": _numero_en((receta.get("epocas_maximas") or {}).get("aplicado_aqui", ""),
                                r"(\d+[.,]\d+) del presupuesto de pared"),
            "de": f"{v4n}: receta_entrenamiento.epocas_maximas.aplicado_aqui («0,75 del "
                  f"presupuesto de pared»)"},
        "semilla_del_ruido_de_cuantiles": {
            "valor": _numero_en(ruido, r"semilla (\d+)"),
            "de": f"{v4n}: arquitectura.preprocesado_numericas.ruido_antes_de_ajustar.distribucion"},
        "para_tras_epocas_sin_mejora": {
            "valor": _numero_en(se_fija.get("paciencia", ""), r"tras (\d+) épocas seguidas sin mejora"),
            "de": f"{e2n}: se_fija.paciencia"},
        "metrica_de_parada": {
            "valor": ("validation_loss_del_ensamblado"
                      if re.search(r"pérdida del ENSAMBLADO", se_fija.get("parada_temprana", ""))
                      else None),
            "de": f"{e2n}: se_fija.parada_temprana («la pérdida del ENSAMBLADO en validación»), "
                  f"con el nombre que le da el motor"},
    }


def _lo_que_el_protocolo_no_dice(receta: dict) -> list[str]:
    con_la_e2 = RUTA_DE_LA_ENMIENDA_2.exists()
    return sorted(k for k, v in receta.items()
                  if v["valor"] is None and (con_la_e2 or k not in SOLO_EN_LA_ENMIENDA_2))


def _mismo_valor(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
    return a == b


def _distintas(de_verdad: dict, receta: dict) -> dict:
    """{clave: [lo de verdad, lo del protocolo]} de lo que no coincide; lo
    que el protocolo no fija (la enmienda 2 ausente) no se compara."""
    return {clave: [valor, receta[clave]["valor"]] for clave, valor in de_verdad.items()
            if clave in receta and receta[clave]["valor"] is not None
            and not _mismo_valor(valor, receta[clave]["valor"])}


#: Filas de train de la sonda de la receta: MÁS que el lote de 256, para ver
#: que el lote de entrenamiento es min(256, filas) y no otro.
FILAS_DE_TRAIN_DE_LA_SONDA = 300
#: Su presupuesto de pared: el plazo (0,75) es el TOPE; con el objetivo al
#: azar la paciencia para antes (medido: ~1 s).
PRESUPUESTO_DE_LA_SONDA_S = 20.0
_AUSENTE = object()
_RECETA_MEDIDA: dict = {}


def _datos_de_la_sonda_de_la_receta():
    """Dos numéricas y una categórica, objetivo binario AL AZAR (la pérdida de
    validación deja de mejorar enseguida y la paciencia corta en décimas)."""
    from matrixai.estudio import ProblemSpec  # noqa: PLC0415

    rng = random.Random(0)
    filas = [{"row_id": str(i), "x1": rng.uniform(0.0, 1.0), "x2": rng.gauss(0.0, 1.0),
              "c": rng.choice("abc"), "y": rng.choice(("no", "si"))}
             for i in range(FILAS_DE_TRAIN_DE_LA_SONDA + 60)]
    hacer = lambda xs: Particion.desde_filas(xs, row_id_field="row_id", target_field="y")  # noqa: E731
    spec = ProblemSpec(problem_id="sonda-de-la-receta-119-c3", target="y",
                       task="binary_classification", observation_unit="fila",
                       classes=("no", "si"), positive_label="si", predictors=("x1", "x2", "c"))
    return (hacer(filas[:FILAS_DE_TRAIN_DE_LA_SONDA]), hacer(filas[FILAS_DE_TRAIN_DE_LA_SONDA:]),
            spec)


def medir_la_receta_del_motor(*, presupuesto_s: float = PRESUPUESTO_DE_LA_SONDA_S) -> dict:
    """UN AJUSTE DE VERDAD del motor, pequeño (300 filas, ~1 s), con ESPÍAS en
    lo que usa: el optimizador que se construye (su CLASE, y el `lr` y el
    `weight_decay` de su grupo de parámetros), los argumentos con que se
    construye la red, el `max_norm` del recorte, el tamaño de los lotes de
    entrenamiento, y la semilla y la `d_embedding` de la preparación; el
    plazo, por el que el motor declara (0,75 × presupuesto). Lo que no se
    puede espiar sin tocar el motor (la paciencia y la métrica de parada) va
    por lo que el predictor DECLARA, y lo prueban las del motor (S12, S13).

    Mide lo que SE USA, no lo que se declara: cambiar `AdamW` por `Adam` en
    la llamada no cambia ninguna constante ni el `"optimizador": "adamw"` que
    el predictor escribe a mano (auditoría 3, S10). Los espías se quitan
    siempre; el ajuste corre con 1 hilo y su propia semilla."""
    import sklearn.preprocessing  # noqa: F401, PLC0415 -- en frío tarda segundos: fuera del plazo
    import torch  # noqa: PLC0415

    from matrixai_engines.redes import preparacion_tabm, tabm_plr  # noqa: PLC0415

    visto: dict[str, list] = {"optimizadores": [], "red": [], "lotes": [], "recortes": [],
                              "preparacion": []}
    red_cls = tabm_plr.RedTabMPLR
    originales = {(objeto, nombre): vars(objeto).get(nombre, _AUSENTE) for objeto, nombre in (
        (torch.optim.Optimizer, "__init__"), (red_cls, "__init__"), (red_cls, "forward"),
        (torch.nn.utils, "clip_grad_norm_"), (preparacion_tabm, "ajustar_preparacion"))}
    init_del_optimizador = torch.optim.Optimizer.__init__
    init_de_la_red, forward_de_la_red = red_cls.__init__, red_cls.forward
    recorte, ajustar_preparacion = torch.nn.utils.clip_grad_norm_, preparacion_tabm.ajustar_preparacion

    def espia_del_optimizador(self, *a, **kw):
        init_del_optimizador(self, *a, **kw)
        visto["optimizadores"].append(self)

    def espia_de_la_red(self, *a, **kw):
        visto["red"].append(dict(kw))
        init_de_la_red(self, *a, **kw)

    def espia_del_forward(self, *a, **kw):
        if self.training:
            x = next((t for t in list(a) + list(kw.values()) if t is not None), None)
            visto["lotes"].append(int(x.shape[0]))
        return forward_de_la_red(self, *a, **kw)

    def espia_del_recorte(parametros, max_norm, *a, **kw):
        visto["recortes"].append(float(max_norm))
        return recorte(parametros, max_norm, *a, **kw)

    def espia_de_la_preparacion(*a, **kw):
        visto["preparacion"].append({k: kw.get(k, _AUSENTE) for k in ("d_embedding", "semilla")})
        return ajustar_preparacion(*a, **kw)

    train, validacion, spec = _datos_de_la_sonda_de_la_receta()
    espias = {(torch.optim.Optimizer, "__init__"): espia_del_optimizador,
              (red_cls, "__init__"): espia_de_la_red, (red_cls, "forward"): espia_del_forward,
              (torch.nn.utils, "clip_grad_norm_"): espia_del_recorte,
              (preparacion_tabm, "ajustar_preparacion"): espia_de_la_preparacion}
    try:
        for (objeto, nombre), espia in espias.items():
            setattr(objeto, nombre, espia)
        resultado, ajustado = MotorDensaTabM().fit(
            train, validacion, spec, Presupuesto(seed=0, wall_seconds=presupuesto_s, hilos=1),
            candidate="sonda-de-la-receta-119-c3", split_plan_digest="0" * 64)
    finally:
        for (objeto, nombre), original in originales.items():
            if original is _AUSENTE:
                delattr(objeto, nombre)
            else:
                setattr(objeto, nombre, original)
    if ajustado is None:
        raise SystemExit(f"la sonda de la receta (un ajuste de {FILAS_DE_TRAIN_DE_LA_SONDA} filas) "
                         f"no terminó: {resultado.state} {resultado.reason}. No se mide")
    faltan = [k for k in ("optimizadores", "red", "lotes", "recortes", "preparacion") if not visto[k]]
    if faltan or len(visto["optimizadores"]) != 1:
        raise SystemExit(f"la sonda de la receta no vio {faltan or 'UN optimizador'} "
                         f"(optimizadores: {len(visto['optimizadores'])}): el motor ya no entrena "
                         f"como este guion sabe espiar. No se mide")
    optimizador = visto["optimizadores"][0]
    grupo = optimizador.param_groups[0]
    red = visto["red"][0]
    preparacion = visto["preparacion"][0]
    predictor = ajustado.spec.predictor
    parada = (predictor.get("hiperparametros") or {}).get("parada_temprana") or {}
    plazo = (predictor.get("entrenamiento_efectivo") or {}).get("plazo_de_entrenamiento_segundos")
    usado = {
        "optimizador": type(optimizador).__name__.lower(),
        "tasa_de_aprendizaje": grupo.get("lr"), "weight_decay": grupo.get("weight_decay"),
        **{k: red.get(k, "no se pasó") for k in ("k", "d_block", "n_blocks", "dropout",
                                                   "d_embedding", "n_frequencies",
                                                   "frequency_init_scale")},
        "recorte_de_gradiente": (visto["recortes"][0] if len(set(visto["recortes"])) == 1
                                 else sorted(set(visto["recortes"]))),
        "lote": max(visto["lotes"]),
        "semilla_del_ruido_de_cuantiles": preparacion["semilla"],
        "fraccion_del_presupuesto_para_entrenar": (plazo / presupuesto_s if plazo is not None
                                                   else None),
    }
    return {
        "usado": usado,
        "d_embedding_de_la_preparacion": (None if preparacion["d_embedding"] is _AUSENTE
                                          else preparacion["d_embedding"]),
        "declarado_por_el_predictor": {"paciencia": parada.get("paciencia"),
                                       "para_tras_epocas_sin_mejora":
                                           parada.get("para_tras_epocas_sin_mejora"),
                                       "metrica_de_parada": parada.get("metrica")},
        "clase_del_optimizador": f"{type(optimizador).__module__}.{type(optimizador).__qualname__}",
        "filas_de_train": FILAS_DE_TRAIN_DE_LA_SONDA, "presupuesto_s": presupuesto_s,
        "lotes_de_entrenamiento_vistos": sorted(set(visto["lotes"])),
        "epocas": (predictor.get("entrenamiento_efectivo") or {}).get("epocas_ejecutadas"),
        "como_se_mide": (
            "un ajuste de verdad (MotorDensaTabM().fit, 300 filas al azar, 1 hilo) con espías en "
            "torch.optim.Optimizer.__init__, RedTabMPLR.__init__/forward, torch.nn.utils."
            "clip_grad_norm_ y preparacion_tabm.ajustar_preparacion; el plazo, por el que declara "
            "el predictor. La paciencia y la métrica de parada, por lo que DECLARA el predictor"),
    }


def _receta_medida() -> dict:
    """`medir_la_receta_del_motor`, UNA vez por proceso y por código del motor
    y valor de sus constantes (las pruebas llaman a `main()` muchas veces)."""
    modulo = sys.modules[MotorDensaTabM.__module__]
    clave = (_digest_motor_nuevo(), tuple(repr(getattr(modulo, n, None))
                                          for n in CONSTANTES_DE_LA_RECETA.values()))
    if clave not in _RECETA_MEDIDA:
        _RECETA_MEDIDA[clave] = medir_la_receta_del_motor()
    return _RECETA_MEDIDA[clave]


def distintas_de_la_receta_medida(medida: dict, receta: dict) -> dict:
    """Lo que el ajuste espiado USÓ (o su predictor declaró) y el protocolo no
    fija así. El lote, contra min(lote del protocolo, filas de train)."""
    distintas = _distintas({k: v for k, v in medida["usado"].items() if k != "lote"}, receta)
    distintas.update(_distintas(medida["declarado_por_el_predictor"], receta))
    if receta["d_embedding"]["valor"] is not None and not _mismo_valor(
            medida["d_embedding_de_la_preparacion"], receta["d_embedding"]["valor"]):
        distintas["d_embedding_de_la_preparacion"] = [medida["d_embedding_de_la_preparacion"],
                                                      receta["d_embedding"]["valor"]]
    lote_esperado = min(receta["lote"]["valor"], medida["filas_de_train"])
    if not _mismo_valor(medida["usado"]["lote"], lote_esperado):
        distintas["lote"] = [medida["usado"]["lote"], lote_esperado]
    return distintas


def exigir_la_configuracion_de_la_enmienda(*, medir: bool = True) -> dict:
    """PARA si el motor no usa la receta ENTERA que fija el protocolo (v4 +
    enmiendas 1 y 2; `receta_que_fija_el_protocolo`), en tres pasos:

    1. k, d_block y n_blocks de la enmienda 1 son los que este guion tiene
       escritos (`CONFIGURACION_DE_LA_ENMIENDA_1`);
    2. las CONSTANTES del módulo del motor (`CONSTANTES_DE_LA_RECETA`, las que
       `_ajustar` pasa a la red, al optimizador, al recorte…) son las del
       protocolo, una a una;
    3. un ajuste de verdad con espías (`medir_la_receta_del_motor`) USA eso:
       la CLASE del optimizador, sus `lr`/`weight_decay`, los argumentos de la
       red, el recorte, el lote, la semilla y el plazo.

    Antes (auditoría 3, I2) solo miraba k, d_block y n_blocks: con el lr, el
    weight_decay, el dropout o las frecuencias cambiadas, o Adam en vez de
    AdamW, la pasada medía otra receta sin que nada se pusiera rojo."""
    enmienda = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    declarada = configuracion_que_declara_la_enmienda(enmienda.get("que_cambia", ""))
    if declarada != CONFIGURACION_DE_LA_ENMIENDA_1:
        raise SystemExit(
            f"la enmienda 1 declara {declarada} y este guion tiene escrito "
            f"{CONFIGURACION_DE_LA_ENMIENDA_1}: no se mide hasta que digan lo mismo")
    receta = receta_que_fija_el_protocolo()
    no_dice = _lo_que_el_protocolo_no_dice(receta)
    if no_dice:
        raise SystemExit(f"el protocolo (v4 + enmiendas) no dice {no_dice}: sin eso no se puede "
                         f"comprobar la receta del motor. No se mide")
    modulo = sys.modules[MotorDensaTabM.__module__]
    leida = {clave: getattr(modulo, nombre, None) for clave, nombre in CONSTANTES_DEL_MOTOR.items()}
    constantes = {clave: getattr(modulo, nombre, None)
                  for clave, nombre in CONSTANTES_DE_LA_RECETA.items()}
    distintas = _distintas(constantes, receta)
    if leida != CONFIGURACION_DE_LA_ENMIENDA_1 or distintas:
        raise SystemExit(
            f"el motor ({MotorDensaTabM.__module__}) no usa la receta del protocolo: "
            f"{ {k: {'motor': v[0], 'protocolo': v[1], 'constante': CONSTANTES_DE_LA_RECETA[k]} for k, v in distintas.items()} } "
            f"(k/d_block/n_blocks: {leida}; la enmienda 1 fija {CONFIGURACION_DE_LA_ENMIENDA_1}). "
            f"Medir otra configuración no es medir el corte. No se mide")
    medida = _receta_medida() if medir else None
    if medida is not None:
        distintas_medidas = distintas_de_la_receta_medida(medida, receta)
        if distintas_medidas:
            raise SystemExit(
                f"un ajuste de verdad del motor (con espías) USA otra receta que la del "
                f"protocolo: { {k: {'usado': v[0], 'protocolo': v[1]} for k, v in distintas_medidas.items()} } "
                f"(optimizador: {medida['clase_del_optimizador']}). Las constantes pueden estar "
                f"bien y la llamada no: medir otra receta no es medir el corte. No se mide")
    return {"declarada_por_la_enmienda_1": declarada, "leida_del_motor": leida,
            "constantes": dict(CONSTANTES_DEL_MOTOR), "modulo": MotorDensaTabM.__module__,
            "receta_del_protocolo": receta,
            "constantes_de_la_receta_leidas_del_motor": constantes,
            "receta_medida_en_un_ajuste": medida}


def _lo_declarado(registro: dict, ruta: tuple):
    valor = registro
    for parte in ruta:
        if not isinstance(valor, dict) or parte not in valor:
            return _AUSENTE
        valor = valor[parte]
    return valor


def exigir_la_arquitectura_del_intento(registro: dict, *, receta: dict | None = None,
                                       n_train: int | None = None) -> None:
    """Un intento COMPLETADO tiene que declarar la arquitectura de la
    enmienda 1 Y la receta del protocolo (`DECLARADO_POR_EL_INTENTO`; el
    lote, contra min(256, `n_train`) si se da). Si no la declara, no se puede
    saber qué corrió: también para."""
    if registro.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
        return
    donde = f"{registro['dataset']} rep={registro['repeticion']} pliegue={registro['pliegue']}"
    arquitectura = registro.get("arquitectura")
    if not isinstance(arquitectura, dict):
        raise SystemExit(
            f"{donde}: el motor completó el intento SIN declarar su arquitectura -- no se puede "
            f"comprobar que corrió la de la enmienda 1. Se para la pasada")
    leida = {k: arquitectura.get(k) for k in CONFIGURACION_DE_LA_ENMIENDA_1}
    if leida != CONFIGURACION_DE_LA_ENMIENDA_1:
        raise SystemExit(
            f"{donde}: el motor declara haber corrido {leida} y la enmienda 1 fija "
            f"{CONFIGURACION_DE_LA_ENMIENDA_1}. Se para la pasada")
    receta = receta if receta is not None else receta_que_fija_el_protocolo()
    sin_declarar, distintas = [], {}
    comprobar = dict(DECLARADO_POR_EL_INTENTO)
    if n_train is not None:
        comprobar[("hiperparametros", "lote")] = "lote"
    for ruta, clave in comprobar.items():
        esperado = receta[clave]["valor"]
        if clave == "lote" and esperado is not None:
            esperado = min(esperado, n_train)
        if esperado is None:
            continue  # solo lo de la enmienda 2 cuando no está (la guardia ya lo exigió)
        valor = _lo_declarado(registro, ruta)
        if valor is _AUSENTE:
            sin_declarar.append(".".join(ruta))
        elif not _mismo_valor(valor, esperado):
            distintas[".".join(ruta)] = {"declarado": valor, "protocolo": esperado}
    if sin_declarar:
        raise SystemExit(
            f"{donde}: el motor completó el intento SIN declarar {sin_declarar} -- no se puede "
            f"comprobar que corrió la receta del protocolo. Se para la pasada")
    if distintas:
        raise SystemExit(f"{donde}: el motor declara haber corrido {distintas}: no es la receta "
                         f"del protocolo. Se para la pasada")


def _payload_v2() -> dict:
    if not RUTA_V2_RESULTADO.exists():
        raise SystemExit(
            f"no está {RUTA_V2_RESULTADO}: sin la pasada v2 no hay con qué comparar el "
            f"motor nuevo -- este guion no inventa una densa v2 que no se ha medido")
    return _leer_json(RUTA_V2_RESULTADO)


def _resultados_v2() -> list[dict]:
    return _payload_v2()["resultados"]


def _normalizado(valor):
    return json.loads(json.dumps(valor, ensure_ascii=False))


def exigir_la_particion_de_la_v2(ds, particion_declarada: dict, lectura: dict | None,
                                 payload_v2: dict) -> dict:
    """PARA si la partición (plan_digest, pliegues obtenidos, límites, filas,
    test, predictores) o la lectura (objetivo, columnas excluidas,
    normalizaciones) de ESTE conjunto no son las que la v2 registró. Se
    comprueba ANTES del primer intento del conjunto: comparar los 32 al
    arrancar exigiría cargar todos los ARFF una vez más (Allstate tarda 46 s
    solo en su partición)."""
    registrada = (payload_v2.get("particion_por_dataset") or {}).get(ds.nombre)
    if registrada is None:
        raise SystemExit(f"{ds.nombre}: la v2 no registra su partición: no hay con qué "
                         f"comparar. No se mide")
    propia = _normalizado(particion_declarada)
    if propia != registrada:
        distintas = sorted(k for k in set(propia) | set(registrada)
                           if propia.get(k) != registrada.get(k))
        raise SystemExit(
            f"{ds.nombre}: la partición NO es la de la v2 (difieren {distintas}: "
            f"{ {k: (propia.get(k), registrada.get(k)) for k in distintas} }). "
            f"Comparar con la v2 sobre otra partición no es la comparación del protocolo")
    lectura_v2 = (payload_v2.get("lectura_de_los_datos") or {}).get(str(ds.data_id))
    if lectura is not None and lectura_v2 is not None and _normalizado(lectura) != lectura_v2:
        raise SystemExit(f"{ds.nombre}: la LECTURA del ARFF no es la de la v2 "
                         f"(hoy {lectura}, v2 {lectura_v2}). No se mide")
    return {"plan_digest": propia.get("plan_digest"), "identica_a_la_v2": True,
            "lectura_identica_a_la_v2": lectura_v2 is not None and lectura is not None}


def _exigir_que_quepa() -> None:
    c6._exigir_que_quepa()


# ---------------------------------------------------------------------------
# 3. EL ENTORNO Y LA CACHÉ (I4)
# ---------------------------------------------------------------------------

#: El motor nuevo entero: la clase, la red y su preparación (digest POR
#: MOTOR, como `_FICHERO_POR_MOTOR` de C3). También están en el del entorno.
_FICHEROS_DEL_MOTOR_NUEVO = (
    c3._DIR_ENGINES / "motores" / "densa_tabm.py",
    c3._DIR_ENGINES / "redes" / "tabm_plr.py",
    c3._DIR_ENGINES / "redes" / "preparacion_tabm.py",
)

#: Las bibliotecas que entran en el digest POR VERSIÓN (la clave es la de
#: `versiones_de_bibliotecas()`, la misma que registra la procedencia).
BIBLIOTECAS_EN_EL_DIGEST = ("python", "numpy", "scikit-learn", "torch")


def _raices_de_import() -> tuple[Path, ...]:
    """El orden de `sys.path` que ven el padre y el hijo: `c3` antepone los
    `src` de engines y el core, y este guion su propio directorio."""
    return (c3._RAIZ_DE_ENGINES.resolve(), c3._RAIZ_DEL_CORE.resolve(), _AQUI)


def _ficheros_de_un_import(nombre: str, raices: tuple[Path, ...]) -> list[Path]:
    """Los ficheros que ejecuta `import <nombre>` si es de los repos
    (paquetes padre incluidos, que también se ejecutan), o [] si es de
    terceros o de la biblioteca estándar."""
    partes = nombre.split(".")
    for raiz in raices:
        if not ((raiz / partes[0]).is_dir() or (raiz / f"{partes[0]}.py").is_file()):
            continue
        ficheros, actual = [], raiz
        for parte in partes:
            directorio, fichero = actual / parte, actual / f"{parte}.py"
            if (directorio / "__init__.py").is_file():
                ficheros.append(directorio / "__init__.py")
                actual = directorio
            elif fichero.is_file():
                ficheros.append(fichero)
                break
            elif directorio.is_dir():  # paquete de espacio de nombres
                actual = directorio
            else:
                break
        return ficheros
    return []


def _paquete_de(fichero: Path, raices: tuple[Path, ...]) -> list[str]:
    for raiz in raices:
        try:
            partes = list(fichero.relative_to(raiz).with_suffix("").parts)
        except ValueError:
            continue
        return partes[:-1]
    return []


@functools.lru_cache(maxsize=None)
def _cierre_de_imports(entrada: Path, raices: tuple[Path, ...]) -> tuple[tuple[Path, ...],
                                                                          tuple[str, ...]]:
    """El CIERRE ESTÁTICO de imports de `entrada` dentro de `raices`: cada
    `import`/`from … import` escrito en el código —TAMBIÉN los de dentro de
    funciones, que un `sys.modules` del padre no ve porque se importan en el
    hijo—, seguido recursivamente. Es un SUPERCONJUNTO de lo que corre (un
    import dentro de una rama no tomada también entra): mejor invalidar de
    más que reusar con código distinto. Devuelve además los sitios con un
    import DINÁMICO (`import_module`/`__import__`), que no se pueden seguir."""
    pendientes, vistos, dinamicos = [entrada.resolve()], set(), []
    while pendientes:
        fichero = pendientes.pop()
        if fichero in vistos:
            continue
        vistos.add(fichero)
        arbol = ast.parse(fichero.read_text(encoding="utf-8"), filename=str(fichero))
        paquete = _paquete_de(fichero, raices)
        for nodo in ast.walk(arbol):
            nombres: list[str] = []
            if isinstance(nodo, ast.Import):
                nombres = [a.name for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom):
                if nodo.level:
                    base = paquete[: len(paquete) - (nodo.level - 1)]
                    modulo = ".".join(base + ([nodo.module] if nodo.module else []))
                else:
                    modulo = nodo.module or ""
                nombres = [modulo] + [f"{modulo}.{a.name}" for a in nodo.names if a.name != "*"]
            elif isinstance(nodo, ast.Call):
                funcion = nodo.func
                nombre = (funcion.id if isinstance(funcion, ast.Name)
                          else funcion.attr if isinstance(funcion, ast.Attribute) else None)
                if nombre in ("import_module", "__import__"):
                    dinamicos.append(f"{_etiqueta(fichero)}:{nodo.lineno}: "
                                     f"{ast.unparse(nodo)[:120]}")
            for n in nombres:
                for g in _ficheros_de_un_import(n, raices):
                    if g.resolve() not in vistos:
                        pendientes.append(g.resolve())
    return tuple(sorted(vistos)), tuple(sorted(dinamicos))


def _etiqueta(ruta: Path) -> str:
    """La ruta RELATIVA a su repo: el digest no puede depender de dónde está
    el árbol (el principal, un worktree de la cola o una copia)."""
    r = Path(ruta).resolve()
    for nombre, raiz in (("engines", c3._RAIZ_DE_ENGINES), ("core", c3._RAIZ_DEL_CORE)):
        try:
            return f"{nombre}:{r.relative_to(raiz.resolve()).as_posix()}"
        except ValueError:
            continue
    return f"fuera:{r.as_posix()}"


def ficheros_del_entorno() -> tuple[Path, ...]:
    """El cierre de imports de este guion MÁS los compartidos de C5 (algunos,
    como `dense_forward.py`, ya están en el cierre) y los del motor nuevo."""
    cierre, _ = _cierre_de_imports(Path(__file__).resolve(), _raices_de_import())
    extra = {Path(p).resolve() for p in c5._FICHEROS_COMPARTIDOS + _FICHEROS_DEL_MOTOR_NUEVO}
    return tuple(sorted(set(cierre) | extra))


def _datos_del_entorno() -> dict[str, Path]:
    """Los ficheros de DATOS que deciden qué se mide: el protocolo que se lee
    y los dos que se citan."""
    return {"protocolo_exploratorio_v2": RUTA_DEL_PROTOCOLO_V2,
            "protocolo_119_v4": RUTA_DEL_PROTOCOLO_V4,
            "protocolo_119_v4_enmienda_1": RUTA_DE_LA_ENMIENDA_1}


def _versiones_en_el_digest() -> dict[str, str]:
    versiones = versiones_de_bibliotecas()
    return {b: str(versiones.get(b, "ausente")) for b in BIBLIOTECAS_EN_EL_DIGEST}


def componentes_del_digest_del_entorno() -> dict[str, str]:
    componentes = {_etiqueta(f): c3._digest_fichero(f) for f in ficheros_del_entorno()}
    for nombre, ruta in _datos_del_entorno().items():
        componentes[f"dato:{nombre}"] = c3._digest_fichero(ruta)
    for nombre, version in _versiones_en_el_digest().items():
        componentes[f"version:{nombre}"] = version
    return componentes


def _digest_de(rutas: tuple[Path, ...]) -> str:
    return hashlib.sha256(
        "".join(c3._digest_fichero(f) for f in rutas).encode()).hexdigest()[:16]


def _digest_de_componentes(componentes: dict[str, str]) -> str:
    return hashlib.sha256(_canonico(componentes).encode()).hexdigest()[:16]


def _digest_entorno() -> str:
    return _digest_de_componentes(componentes_del_digest_del_entorno())


def _digest_motor_nuevo() -> str:
    return _digest_de(_FICHEROS_DEL_MOTOR_NUEVO)


def lo_que_no_cubre_el_digest() -> dict:
    """Lo que el digest de la caché NO cubre, dicho dentro del resultado —
    como hace CLAUDE.md con el de 101-C5—, y lo que se puede, MEDIDO."""
    _, dinamicos = _cierre_de_imports(Path(__file__).resolve(), _raices_de_import())
    cubiertos = {str(p) for p in ficheros_del_entorno()}
    prefijos = tuple(f"{r}/" for r in (c3._RAIZ_DE_ENGINES.resolve(), c3._RAIZ_DEL_CORE.resolve()))
    # `is_file()`: torch registra módulos con un `__file__` RELATIVO
    # («_classes.py») que `resolve()` pega al directorio actual; no existen.
    fuera = sorted(_etiqueta(Path(f)) for f in c3._ficheros_importados_por_este_proceso()
                   if f.endswith(".py") and f.startswith(prefijos) and f not in cubiertos
                   and Path(f).is_file())
    versiones = versiones_de_bibliotecas()
    return {
        "como_se_calcula": (
            "cierre ESTÁTICO de imports de pasada_119_c3.py dentro de los dos repos (ast de cada "
            "fichero, TODOS sus import, también los de dentro de funciones), más los compartidos "
            "de C5 y los tres del motor. Es un superconjunto de lo que corre: medido el 29-09, un "
            "intento de kc2 en proceso cargó 124 módulos de los repos y los 124 estaban en el "
            "cierre (190)"),
        "bibliotecas_de_terceros_que_NO_invalidan": sorted(
            k for k in versiones if k not in BIBLIOTECAS_EN_EL_DIGEST),
        "bibliotecas_de_terceros": (
            f"solo por versión: {', '.join(BIBLIOTECAS_EN_EL_DIGEST)} entran en el digest; las "
            f"demás constan en procedencia.versiones_de_bibliotecas pero un cambio en ellas NO "
            f"invalida la caché. Su código tampoco entra, ni el de la biblioteca estándar"),
        "imports_dinamicos_en_el_cierre": list(dinamicos),
        "imports_dinamicos": ("un import_module/__import__ con una cadena no se sigue; los "
                              "listados (medidos en el cierre) cargan bibliotecas de terceros"),
        "datos": ("los ARFF NO entran en el digest del entorno: su sha256 va en la clave de cada "
                  "registro (`datos_sha256`, que `_reusable_c3` compara). pasada_v2_113_resultado"
                  ".json no entra: no cambia lo que ejecuta un intento, solo la comparación, que "
                  "se recalcula en cada ejecución"),
        "enmienda_2": ("no entra: fija cómo se CUENTAN los fallos, no lo que se ejecuta; el "
                       "veredicto se recalcula en cada ejecución"),
        "modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest": fuera,
        "que_dice_la_lista_de_arriba": (
            "COTA INFERIOR: los módulos del padre; el hijo de multiprocessing.spawn importa los "
            "suyos, que el cierre estático cubre salvo import dinámico. Tiene que estar vacía"),
    }


def _reusable_c3(previo: dict | None, entorno_digest: str, motor_digest: str,
                 wall_seconds: float, *, datos_sha256: str | None) -> bool:
    """`c3._reusable` MÁS lo que esa función no comprueba: el estado tiene
    que contar como medida (un `failed` con los digests buenos se reusaría
    para siempre: trampa nº1 de «ANTES DE RELANZAR UNA PASADA») y el ARFF
    tiene que ser el mismo (su sha256 va en el registro)."""
    if not c3._reusable(previo, entorno_digest, motor_digest, wall_seconds):
        return False
    if previo.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
        return False
    return datos_sha256 is not None and previo.get("datos_sha256") == datos_sha256


# ---------------------------------------------------------------------------
# 4. LA SALIDA (I7): de qué tipo es, dónde va, y la fusión con la que ya hay
# ---------------------------------------------------------------------------

TIPOS_DE_EJECUCION = ("pasada", "solo", "humo", "estimar")


def tipo_de_ejecucion(*, humo: bool, solo: str | None, estimar: bool) -> str:
    if estimar:
        return "estimar"
    if humo:
        return "humo"
    if solo:
        return "solo"
    return "pasada"


def ruta_de_salida(tipo: str, salida: str | None) -> Path:
    """Dónde escribe cada tipo. `--solo`/`--humo`/`--estimar` van, por
    omisión, a ficheros APARTE, y NUNCA al resultado real aunque se pida con
    `--salida`. Una ruta absoluta fuera del árbol vale (la cola nocturna)."""
    ruta = (Path(salida).expanduser() if salida else SALIDA_POR_OMISION[tipo]).resolve()
    if tipo != "pasada" and ruta == RUTA_DEL_RESULTADO.resolve():
        raise SystemExit(f"--{tipo} no escribe en el resultado real ({RUTA_DEL_RESULTADO.name}): "
                         f"una prueba no puede pisar una noche de medición")
    return ruta


def tipo_del_fichero(payload: dict) -> str:
    """De qué tipo es un resultado ya escrito (también los de antes de que
    existiera el campo `tipo_de_ejecucion`)."""
    if payload.get("tipo_de_ejecucion") in TIPOS_DE_EJECUCION:
        return payload["tipo_de_ejecucion"]
    if str(payload.get("sub_corte", "")).startswith("estimacion"):
        return "estimar"
    if payload.get("es_humo"):
        return "humo"
    if payload.get("es_subconjunto_de_prueba"):
        return "solo"
    return "pasada"


def cargar_salida_previa(ruta: Path, tipo: str) -> tuple[dict, dict]:
    """El caché y el fichero entero, NEGÁNDOSE a mezclar tipos: un `--humo`
    sobre un resultado real (o una pasada sobre uno de humo) para aquí, antes
    de medir nada y sin tocar el fichero."""
    if not ruta.exists():
        return {}, {}
    payload = _leer_json(ruta)
    del_fichero = tipo_del_fichero(payload)
    if del_fichero != tipo:
        raise SystemExit(f"{ruta} es un resultado de tipo «{del_fichero}» y esta ejecución es "
                         f"«{tipo}»: no se mezclan. Otra --salida, o mover ese fichero a mano")
    cache = {(r["dataset"], r["motor"], r["repeticion"], r["pliegue"]): r
             for r in payload.get("resultados", [])}
    return cache, payload


def _clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["repeticion"], r["pliegue"])


def fusionar_resultados(previos: list[dict], de_esta_ejecucion: list[dict]) -> tuple[
        list[dict], list[dict], list[dict]]:
    """Lo que ya había y NO se ha vuelto a medir (o reusar) en esta
    ejecución, seguido de lo de esta ejecución. Devuelve también los
    conservados, para declararlos. Un punto de control a mitad de la noche
    ya no puede dejar el fichero solo con los conjuntos recorridos.

    Y, TERCERO, los SUSTITUIDOS SIN MEDIDA (auditoría 3, I4): un registro
    previo que no contaba como medida (`failed`, `cancelled`…) y que esta
    ejecución ha vuelto a medir. La regla de la casa es reintentarlo
    (`_reusable_c3` no reusa un fallo), pero sustituirlo sin más borraba que
    falló: un conjunto PERDIDO la noche 1 salía cumplido la noche 2 sin que
    el artefacto lo dijera. Se devuelven para `rastro_de_los_reintentos`."""
    nuevas = {_clave(r) for r in de_esta_ejecucion}
    conservados = [r for r in previos if _clave(r) not in nuevas]
    sustituidos_sin_medida = [
        r for r in previos if _clave(r) in nuevas
        and r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]
    return conservados + list(de_esta_ejecucion), conservados, sustituidos_sin_medida


def rastro_de_los_reintentos(sustituidos_sin_medida: list[dict], de_esta_ejecucion: list[dict],
                             *, rastro_previo: list[dict], procedencias_previas: dict) -> list[dict]:
    """`intentos_reintentados` del artefacto: el que ya traía el fichero (se
    CONSERVA entero, noche tras noche) más cada intento sin medida que esta
    ejecución ha reintentado — qué (conjunto, repetición, pliegue, el
    registro sustituido ENTERO, con su `traza`), cuándo (`medido` de su
    procedencia), con qué error (`estado` y `motivo`) y qué salió al
    reintentarlo."""
    nuevos = {_clave(r): r for r in de_esta_ejecucion}
    rastro = list(rastro_previo)
    for r in sustituidos_sin_medida:
        reintento = nuevos[_clave(r)]
        procedencia_id = r.get("procedencia_id")
        rastro.append({
            "dataset": r["dataset"], "motor": r["motor"], "repeticion": r["repeticion"],
            "pliegue": r["pliegue"], "estado": r.get("estado"), "motivo": r.get("motivo"),
            "procedencia_id": procedencia_id,
            "medido": (procedencias_previas.get(procedencia_id) or {}).get("medido"),
            "reintentado_por": {"procedencia_id": reintento.get("procedencia_id"),
                                "estado": reintento.get("estado"),
                                "reusado": reintento.get("reusado")},
            "registro_sustituido": r,
        })
    return rastro


QUE_SON_LOS_REINTENTADOS = (
    "intentos que en una ejecución ANTERIOR no contaron como medida (fallidos, cancelados…) y "
    "una posterior volvió a medir: `resultados` y el veredicto cuentan el reintento (la regla de "
    "la casa es reintentar lo que no está completed: un tope por carga ajena no se cementa), y "
    "aquí queda lo que pasó antes, con su error y su hora (`intentos_reintentados`). Sin esto, un "
    "conjunto perdido una noche salía cumplido la siguiente sin que el artefacto lo dijera "
    "(auditoría 3 del 119, I4)")


# ---------------------------------------------------------------------------
# 5. EL VEREDICTO (I2): cómo cuentan los fallos
# ---------------------------------------------------------------------------

COMO_SE_CUENTAN_LOS_FALLOS = {
    "fuente": ("decisión del supervisor tras la auditoría 2 de 119-C3 (29-09), registrada en la "
               "enmienda 2 del protocolo v4 (`se_fija.fallos_en_el_veredicto`; ver `enmienda_2`)"),
    "que_es_un_fallo": ("un intento ESPERADO por la partición (repetición y pliegue que "
                        "`propuesta.pliegues` trae) que no está en el artefacto, cuyo estado no "
                        "cuenta como medida (ESTADOS_QUE_CUENTAN_COMO_MEDIDA), o que se completó "
                        "sin la métrica de cierre"),
    "a_condicion_1": ("un conjunto con CUALQUIER fallo del motor nuevo cuenta como PERDIDO (no "
                      "cumple), como dice regla_de_cierre.definicion_de_mejor. Mismo criterio "
                      "para la densa v2"),
    "b_condicion_2": ("fallos del motor nuevo con la densa v2 completa = INFERIORIDAD; si la densa "
                      "v2 falló TODOS sus intentos (Allstate, KDDCup09) = «sin comparación», ni "
                      "mejora ni inferioridad, y solo cuenta en la condición 1"),
    "c_emparejada": ("la diferencia emparejada solo sobre repeticiones y pliegues COMPLETOS en los "
                     "dos motores"),
    "caso_no_previsto": ("fallos del motor nuevo Y fallos parciales de la densa v2: la decisión no "
                         "lo cubre; se aplica (c) y se marca `caso_no_previsto`. En la v2 real "
                         "ningún conjunto no sellado tiene fallos PARCIALES de la densa (probado)"),
    "d_registro": "`veredicto_por_conjunto` lleva los fallos de cada motor: cuántos y por qué",
}


def fallos_del_motor(registros: list[dict], motor: str, metric_id: str,
                     esperadas) -> list[dict]:
    """Los intentos ESPERADOS de `motor` que no traen medida, con su motivo."""
    por_clave: dict[tuple, list[dict]] = {}
    for r in registros:
        if r.get("motor") == motor:
            por_clave.setdefault((r["repeticion"], r["pliegue"]), []).append(r)
    fallos = []
    for repeticion, pliegue in sorted(tuple(k) for k in esperadas):
        rs = por_clave.get((repeticion, pliegue), [])
        base = {"repeticion": repeticion, "pliegue": pliegue}
        if not rs:
            fallos.append({**base, "tipo": "ausente", "estado": None,
                           "motivo": "el artefacto no trae este intento"})
            continue
        no_medidos = [r for r in rs
                      if r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]
        if no_medidos:
            r = no_medidos[0]
            fallos.append({**base, "tipo": "no_completado", "estado": r.get("estado"),
                           "motivo": r.get("motivo")})
        elif any(r.get(metric_id) is None for r in rs):
            fallos.append({**base, "tipo": "sin_metrica_de_cierre", "estado": rs[0].get("estado"),
                           "motivo": f"completado sin {metric_id}"})
    return fallos


def registros_de_los_fallos(dataset: str, motor: str, fallos: list[dict]) -> list[dict]:
    """Un registro NO medido por cada fallo, para que `aplicar_regla_de_cierre`
    —que solo ve los registros que hay— cuente como perdido también un
    intento ausente o sin métrica. Solo entran en la cuenta, nunca en
    `resultados`."""
    return [{"dataset": dataset, "motor": motor, "repeticion": f["repeticion"],
             "pliegue": f["pliegue"], "estado": f"sin_medida:{f['tipo']}"} for f in fallos]


def _sin_la_palabra_palanca(veredicto: dict) -> dict:
    renombres = {
        "n_pliegues_de_la_palanca": "n_pliegues_completos_del_motor_nuevo",
        "n_pliegues_de_la_densa_v2": "n_pliegues_completos_de_la_densa_v2",
        "pliegues_de_la_palanca_sin_intento_de_la_densa_v2":
            "pliegues_del_motor_nuevo_sin_intento_de_la_densa_v2",
    }
    salida = {renombres.get(k, k): v for k, v in veredicto.items()}
    if isinstance(salida.get("intervalo"), dict):
        salida["intervalo"] = dict(salida["intervalo"], diferencia="motor nuevo - densa v2")
    return salida


def veredicto_del_conjunto_c3(*, dataset: str, metric_id: str,
                              registros_del_motor_nuevo: list[dict],
                              registros_de_la_densa_v2: list[dict], esperadas) -> dict:
    """Condición 2 en UN conjunto, con los fallos (I2 b, c, d). La emparejada
    es la de 118 (`p118.veredicto_del_conjunto`, motor nuevo − densa v2,
    bootstrap de pliegues) sobre los pliegues completos en los DOS."""
    esperadas = sorted(tuple(k) for k in esperadas)
    en_el_plan = set(esperadas)
    fallos_nuevo = fallos_del_motor(registros_del_motor_nuevo, NOMBRE_MOTOR_NUEVO, metric_id,
                                    esperadas)
    fallos_v2 = fallos_del_motor(registros_de_la_densa_v2, NOMBRE_DENSA_V2, metric_id, esperadas)
    nuevo_por_pliegue = {k: v for k, v in c6.medida_por_pliegue(
        registros_del_motor_nuevo, NOMBRE_MOTOR_NUEVO, metric_id).items() if k in en_el_plan}
    v2_por_pliegue = {k: v for k, v in c6.medida_por_pliegue(
        registros_de_la_densa_v2, NOMBRE_DENSA_V2, metric_id).items() if k in en_el_plan}
    emparejada = _sin_la_palabra_palanca(p118.veredicto_del_conjunto(
        dataset=dataset, metric_id=metric_id, palanca_por_pliegue=nuevo_por_pliegue,
        v2_densa_por_pliegue=v2_por_pliegue))
    salida = {
        **emparejada,
        "n_intentos_esperados": len(esperadas),
        "fallos_del_motor_nuevo": {"n": len(fallos_nuevo), "detalle": fallos_nuevo},
        "fallos_de_la_densa_v2": {"n": len(fallos_v2), "detalle": fallos_v2},
        "sin_comparacion": False, "caso_no_previsto": False,
        "regla_aplicada": "c_emparejada",
        "la_emparejada_decide": True,
    }
    if esperadas and len(fallos_v2) == len(esperadas):
        salida.update(mejora=False, inferioridad=False, sin_comparacion=True,
                      regla_aplicada="b_sin_comparacion", la_emparejada_decide=False,
                      motivo=(f"la densa v2 no completó NINGUNO de sus {len(esperadas)} intentos: "
                              f"no hay comparación emparejada posible; solo cuenta en la "
                              f"condición 1"))
    elif fallos_nuevo and not fallos_v2:
        salida.update(mejora=False, inferioridad=True, regla_aplicada="b_inferioridad_por_fallos",
                      la_emparejada_decide=False,
                      motivo=(f"el motor nuevo no completó {len(fallos_nuevo)} de "
                              f"{len(esperadas)} intentos que la densa v2 sí completó: cuenta "
                              f"como INFERIORIDAD (la emparejada de abajo no decide)"))
    elif fallos_nuevo and fallos_v2:
        salida.update(caso_no_previsto=True, regla_aplicada="caso_no_previsto_c_emparejada",
                      motivo=(f"fallos en los DOS ({len(fallos_nuevo)} del nuevo, "
                              f"{len(fallos_v2)} de la densa v2): caso que la decisión no cubre; "
                              f"se aplica la emparejada sobre los completos en los dos. "
                              f"{emparejada.get('motivo') or ''}").strip())
    return salida


def _cumplidos_de(resultados: list[dict], regla: protocolo_mod.ReglaDeCierre, *, motor: str,
                  metrica_por_dataset: dict[str, str], nombres_de_los_conjuntos: list[str],
                  con_detalle: bool = False) -> dict:
    """`aplicar_regla_de_cierre`, filtrando a los conjuntos pedidos ANTES de
    llamarla (mismo motivo que `pasada_118_palanca._cumplidos`, que fija la
    densa VIEJA y aquí hace falta cualquiera de los dos motores)."""
    conjuntos = set(nombres_de_los_conjuntos)
    filtrados = [r for r in resultados if r["dataset"] in conjuntos]
    resultado = protocolo_mod.aplicar_regla_de_cierre(
        filtrados, regla, motor=motor, metrica_por_dataset=metrica_por_dataset,
        datasets_exigidos=nombres_de_los_conjuntos)
    salida = {k: resultado[k] for k in ("cumplidos", "datasets", "fraccion", "cumple_la_regla")}
    if con_detalle:
        salida["detalle"] = [{k: d.get(k) for k in ("dataset", "cumple", "perdido_por_fallo",
                                                     "mejor", "distancia_en_puntos")}
                             for d in resultado["detalle"]]
    return salida


def veredicto_de_c3(veredictos_por_conjunto: list[dict], *,
                    cumplidos_con_el_motor_nuevo: dict, cumplidos_de_la_densa_v2: dict) -> dict:
    """La regla de subida (`p118.veredicto_de_la_palanca`, importada: la
    MISMA, copiada literal en el v4) con los nombres de C3 y el recuento de
    los «sin comparación» aparte."""
    bruto = p118.veredicto_de_la_palanca(
        veredictos_por_conjunto, cumplidos_con_la_palanca=cumplidos_con_el_motor_nuevo,
        cumplidos_de_la_densa_v2=cumplidos_de_la_densa_v2)
    por = lambda clave: sorted(v["dataset"] for v in veredictos_por_conjunto  # noqa: E731
                               if v.get(clave))
    return {
        "campo": ("los motores de la v2 (pasada_v2_113_resultado.json) con sus resultados de "
                 "esa pasada, y en el sitio de la densa v2, el motor nuevo (matrixai.dense."
                 "tabm_cpu) -- SOLO en los conjuntos medidos aquí"),
        "cumplidos_con_el_motor_nuevo": bruto["cumplidos_con_la_palanca"],
        "cumplidos_de_la_densa_v2_en_los_mismos_conjuntos": bruto[
            "cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"],
        "sube_los_cumplidos": bruto["sube_los_cumplidos"],
        "n_conjuntos": bruto["n_conjuntos"],
        "n_mejora": bruto["n_mejora"], "n_inferioridad": bruto["n_inferioridad"],
        "n_sin_comparacion": sum(1 for v in veredictos_por_conjunto if v.get("sin_comparacion")),
        "conjuntos_con_mejora": por("mejora"),
        "conjuntos_con_inferioridad": por("inferioridad"),
        "conjuntos_sin_comparacion": por("sin_comparacion"),
        "conjuntos_con_fallos_del_motor_nuevo": sorted(
            v["dataset"] for v in veredictos_por_conjunto
            if (v.get("fallos_del_motor_nuevo") or {}).get("n")),
        "mejoras_superan_inferioridades": bruto["mejoras_superan_inferioridades"],
        "cumple_la_regla_de_subida": bruto["la_palanca_ayuda"],
        "regla_de_subida": (
            "protocolo_119_v4.json, veredicto.regla_de_subida: sube a confirmación (C4) si "
            "cumple las DOS condiciones sobre los 32 no sellados frente a la densa v2 en los "
            "MISMOS conjuntos -- (1) el número de conjuntos donde el motor nuevo está a <=2 "
            "puntos del mejor es MAYOR que el de la densa v2, y (2) sus mejoras superan a sus "
            "inferioridades (diferencia emparejada por repetición y pliegue, bootstrap de "
            "1000 remuestras de pliegues, semilla 0)"),
        "como_se_cuentan_los_fallos": COMO_SE_CUENTAN_LOS_FALLOS,
    }


def veredicto_final(*, resultados_v2: list[dict], resultados_del_motor_nuevo: list[dict],
                    veredictos_por_conjunto: list[dict], esperadas_por_conjunto: dict,
                    regla: protocolo_mod.ReglaDeCierre, metrica_por_dataset: dict[str, str],
                    nombres_de_los_conjuntos: list[str]) -> dict:
    """Las dos condiciones. La 1, sobre el CAMPO para el motor nuevo (la
    densa v2 sustituida por él) y sobre la v2 SIN TOCAR para la densa v2 —
    la referencia no puede medirse con el motor nuevo dentro—, con los
    intentos sin medida de cada uno contados como fallo (I2 a)."""
    campo = p118.campo_de_la_comparacion(resultados_v2, resultados_del_motor_nuevo,
                                         set(nombres_de_los_conjuntos))
    densa_v2 = [r for r in resultados_v2 if r["motor"] == NOMBRE_DENSA_V2]
    fallos_nuevo, fallos_v2 = [], []
    for nombre in nombres_de_los_conjuntos:
        esperadas, metrica = esperadas_por_conjunto[nombre], metrica_por_dataset[nombre]
        de_aqui = lambda rs: [r for r in rs if r["dataset"] == nombre]  # noqa: E731
        fallos_nuevo += registros_de_los_fallos(nombre, NOMBRE_MOTOR_NUEVO, fallos_del_motor(
            de_aqui(resultados_del_motor_nuevo), NOMBRE_MOTOR_NUEVO, metrica, esperadas))
        fallos_v2 += registros_de_los_fallos(nombre, NOMBRE_DENSA_V2, fallos_del_motor(
            de_aqui(densa_v2), NOMBRE_DENSA_V2, metrica, esperadas))
    cumplidos_nuevo = _cumplidos_de(
        campo + fallos_nuevo, regla, motor=NOMBRE_MOTOR_NUEVO,
        metrica_por_dataset=metrica_por_dataset,
        nombres_de_los_conjuntos=nombres_de_los_conjuntos, con_detalle=True)
    cumplidos_v2 = _cumplidos_de(
        resultados_v2 + fallos_v2, regla, motor=NOMBRE_DENSA_V2,
        metrica_por_dataset=metrica_por_dataset,
        nombres_de_los_conjuntos=nombres_de_los_conjuntos, con_detalle=True)
    return veredicto_de_c3(veredictos_por_conjunto, cumplidos_con_el_motor_nuevo=cumplidos_nuevo,
                           cumplidos_de_la_densa_v2=cumplidos_v2)


# ---------------------------------------------------------------------------
# 6. EL PLAN DE INTENTOS DE UN CONJUNTO, y lo que se declara de él
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IntentoPlaneado:
    repeticion: int
    pliegue: int
    semilla: int
    presupuesto_wall_s: float
    entrena: tuple
    valida: tuple


def plan_de_intentos(ds, propuesta, protocolo, *, humo: bool) -> list[IntentoPlaneado]:
    """Los intentos de UN conjunto: repeticiones y pliegues del protocolo
    (1 y 1 con `--humo`) que la partición de verdad trae (yeast da 4 pliegues
    por repetición, no 5), cada uno con la semilla de SU repetición y el
    presupuesto de SU cubo."""
    repeticiones = 1 if humo else protocolo.particion.repeticiones_para(ds.cubo)
    folds = 1 if humo else protocolo.particion.folds
    wall = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
    plan = []
    for repeticion in range(repeticiones):
        for pliegue_i in range(folds):
            pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
            if pliegue is None:
                continue
            plan.append(IntentoPlaneado(repeticion, pliegue_i,
                                        protocolo.particion.semillas[repeticion], wall,
                                        tuple(pliegue.entrena), tuple(pliegue.valida)))
    return plan


def allstate_medido(resultados: list[dict]) -> dict:
    """Lo que Allstate entrenó DE VERDAD, junto a lo que la enmienda 1 declara
    («≈10 épocas, menos que las 17 de la regla»)."""
    nombre = "Allstate_Claims_Severity"
    propios = [r for r in resultados if r["dataset"] == nombre]
    if not propios:
        return {"medido": False, "motivo": f"{nombre} no se ha medido en esta ejecución"}
    detalle = []
    for r in sorted(propios, key=lambda r: (r["repeticion"], r["pliegue"])):
        e = r.get("entrenamiento_efectivo") or {}
        detalle.append({"repeticion": r["repeticion"], "pliegue": r["pliegue"],
                        "estado": r["estado"], "epocas_ejecutadas": e.get("epocas_ejecutadas"),
                        "mejor_epoca": e.get("mejor_epoca"),
                        "parado_por_plazo": e.get("parado_por_plazo")})
    epocas = [d["epocas_ejecutadas"] for d in detalle if d["epocas_ejecutadas"] is not None]
    minimo = _leer_json(RUTA_DE_LA_ENMIENDA_1)["medido_que_lo_motiva"]["minimo_de_la_regla"]
    return {"medido": True, "detalle": detalle,
            "epocas_min": min(epocas) if epocas else None,
            "epocas_max": max(epocas) if epocas else None,
            "minimo_de_la_regla": minimo,
            "todos_por_debajo_del_minimo": bool(epocas) and max(epocas) < minimo}


#: Lo que la FUENTE hace de verdad al parar (auditoría 3 del 119, I3: leído
#: el 30-09 en `example.ipynb`, rama main, `evaluate()`).
CITA_DE_LA_PARADA_DE_LA_FUENTE = (
    "score = (-(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5) if task_type == "
    "'regression' else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1)))  "
    "(example.ipynb, evaluate(), rama main)")


def parada_temprana_declarada(resultados: list[dict]) -> dict:
    """LA PARADA TEMPRANA, DECLARADA EN EL RESULTADO como las épocas de
    Allstate (auditoría 3, I3). La enmienda 2 (sellada: no se reescribe)
    fija parar con la pérdida del ENSAMBLADO «como la fuente», y en
    CLASIFICACIÓN no es como la fuente: la fuente para con la ACCURACY del
    ensamblado; este motor, con la LOG-LOSS de las probabilidades
    promediadas, que es el `validation_loss` que registró el v4. En regresión
    sí coincide (MSE y −RMSE ordenan igual). Con lo que DECLARAN los intentos
    de esta ejecución, no con lo que se pidió."""
    v4 = _leer_json(RUTA_DEL_PROTOCOLO_V4)
    e2 = _leer_json(RUTA_DE_LA_ENMIENDA_2) if RUTA_DE_LA_ENMIENDA_2.exists() else {}
    completados = [r for r in resultados
                   if r.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]
    declaradas = sorted({str(_lo_declarado(r, ("hiperparametros", "parada_temprana", "metrica")))
                         for r in completados
                         if _lo_declarado(r, ("hiperparametros", "parada_temprana", "metrica"))
                         is not _AUSENTE})
    clasificacion = sorted({r["dataset"] for r in completados
                            if r.get("tarea") in ("binary_classification",
                                                  "multiclass_classification")})
    return {
        "registrado_en_el_v4": ((v4.get("receta_entrenamiento") or {}).get("parada_temprana")
                                or {}).get("metrica"),
        "fija_la_enmienda_2": (e2.get("se_fija") or {}).get("parada_temprana"),
        "metrica_que_declaran_los_intentos": declaradas,
        "por_tarea": {
            "regression": "como la fuente: el error cuadrático del ensamblado (la fuente, −RMSE: "
                          "ordenan igual)",
            "binary_classification": "NO como la fuente: log-loss del ensamblado; la fuente para "
                                     "con su accuracy",
            "multiclass_classification": "NO como la fuente: log-loss del ensamblado; la fuente "
                                         "para con su accuracy",
        },
        "en_clasificacion_no_es_como_la_fuente": (
            "la fuente para en clasificación con la ACCURACY del ensamblado (accuracy_score de "
            "la media de las k predicciones), que tiene mesetas y empates que cuenta como «sin "
            "mejora»; este motor para con la LOG-LOSS de las probabilidades promediadas, que no "
            "los tiene. Cambia qué época se guarda y cuándo actúa la paciencia, o sea los "
            "números. El código cumple lo REGISTRADO (el v4 decía validation_loss); lo que no es "
            "cierto es el «como la fuente» de la enmienda 2 en clasificación. No se cambia: la "
            "enmienda va sellada y cambiar la parada después de auditar sería otra receta"),
        "cita_de_la_fuente": CITA_DE_LA_PARADA_DE_LA_FUENTE,
        "conjuntos_de_clasificacion_en_esta_ejecucion": clasificacion,
        "de_donde_sale": "auditoría 3 del 119 (30-09), hallazgo I3",
    }


def intentos_por_conjunto(datasets, particiones: dict, esperadas_por_conjunto: dict,
                          resultados: list[dict], protocolo) -> dict:
    """Pedidos por el protocolo, admitidos por la partición, planeados en esta
    ejecución, medidos y completos — por conjunto, CALCULADO."""
    salida = {}
    for ds in datasets:
        if ds.nombre not in particiones:
            continue
        propios = [r for r in resultados if r["dataset"] == ds.nombre]
        particion = particiones[ds.nombre]
        salida[ds.nombre] = {
            "pedidos_por_el_protocolo": (protocolo.particion.folds
                                         * protocolo.particion.repeticiones_para(ds.cubo)),
            "admitidos_por_la_particion": particion.get("n_pliegues_obtenidos"),
            "planeados_en_esta_ejecucion": len(esperadas_por_conjunto.get(ds.nombre, [])),
            "medidos": len(propios),
            "completos": sum(1 for r in propios if r.get("estado")
                             in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA),
            "limites": particion.get("limites"),
        }
    return salida


def _imprimir_estimacion_peor_caso(plan: dict) -> None:
    print(f"conjuntos: {plan['n_datasets']}, 1 motor por pliegue (matrixai.dense.tabm_cpu), "
          f"1 proceso")
    for cubo, e in sorted(plan["por_cubo"].items()):
        print(f"  cubo {cubo:8}: {e['n_datasets']:>2} datasets x {e['folds']} pliegues x "
              f"{e['repeticiones']} rep x {plan['n_motores']} motor = {e['n_intentos']:>5} "
              f"intentos, {e['wall_seconds_por_intento']:.0f}s de tope -> cota "
              f"{e['horas_cota']:.2f} h")
    print(f"  TOTAL: {plan['n_intentos']} intentos (pedidos; la partición puede dar menos)")
    print(f"  COTA de peor caso (cada intento agota su presupuesto ENTERO): "
          f"{plan['horas_de_reloj_cota_peor_caso']:.2f} h "
          f"({plan['horas_de_reloj_cota_peor_caso']/24:.2f} dias)")


# ---------------------------------------------------------------------------
# 7. LA PASADA
# ---------------------------------------------------------------------------

#: `--humo`: los DOS más baratos de los 32 no sellados por número de filas.
CONJUNTOS_DEL_HUMO = ("kc2", "dresses-sales")


def _argumentos(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta todos los intentos (lo que ya "
                             "hay en el fichero y no se re-mide se CONSERVA)")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON (admite una absoluta fuera del arbol). Por omision: "
                             + ", ".join(f"{t}={p.name}" for t, p in SALIDA_POR_OMISION.items()))
    parser.add_argument("--estimar", action="store_true",
                        help="NO mide la pasada: UN intento por conjunto de la lista de la "
                             "estimacion y la cuenta de si cabe en la ventana nocturna o de dia")
    parser.add_argument("--estimar-conjuntos", default=None,
                        help="con --estimar, los conjuntos que se miden (coma), en vez de "
                             "CONJUNTOS_DE_LA_ESTIMACION")
    parser.add_argument("--solo", default=None,
                        help="nombres de conjunto separados por coma, SIEMPRE dentro de los "
                             "32 no sellados (un sellado se NIEGA). Para probar el guion")
    parser.add_argument("--humo", action="store_true",
                        help=f"2 conjuntos pequeños ({', '.join(CONJUNTOS_DEL_HUMO)}), 1 "
                             f"repeticion, 1 pliegue -- prueba el camino entero, no mide nada "
                             f"del corte. Incompatible con --solo")
    args = parser.parse_args(argv)
    if args.humo and args.solo:
        raise SystemExit("--humo y --solo son incompatibles: --humo YA fija sus propios "
                         "conjuntos (CONJUNTOS_DEL_HUMO)")
    if args.estimar and (args.humo or args.solo):
        raise SystemExit("--estimar no se combina con --humo ni con --solo "
                         "(usa --estimar-conjuntos)")
    return args


def main(argv=None) -> None:
    args = _argumentos(argv)
    tipo = tipo_de_ejecucion(humo=args.humo, solo=args.solo, estimar=args.estimar)
    ruta_salida = ruta_de_salida(tipo, args.salida)

    c6.preparar_protocolo_v2()
    protocolos = exigir_los_protocolos_encadenados()
    configuracion_del_motor = exigir_la_configuracion_de_la_enmienda()
    # ANTES de cargar nada: la pasada real no arranca sin la enmienda 2 en regla.
    enmienda_2 = exigir_la_enmienda_2(tipo)
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    no_sellados = c6.datasets_no_sellados(protocolo)

    if tipo == "estimar":
        cmd_estimar(protocolo, todos, no_sellados, ruta_salida=ruta_salida,
                    conjuntos=args.estimar_conjuntos, protocolos=protocolos,
                    configuracion_del_motor=configuracion_del_motor, enmienda_2=enmienda_2)
        return

    solo = ",".join(CONJUNTOS_DEL_HUMO) if tipo == "humo" else args.solo
    datasets, subconjunto = datasets_de_c3(todos, no_sellados, solo=solo)
    if not datasets:
        raise SystemExit("no queda ningún conjunto que medir")
    humo = tipo == "humo"

    _exigir_que_quepa()
    motor = MotorDensaTabM()
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    payload_v2 = _payload_v2()
    resultados_v2 = payload_v2["resultados"]
    sin_particion_v2 = [d.nombre for d in datasets
                        if d.nombre not in (payload_v2.get("particion_por_dataset") or {})]
    if sin_particion_v2:
        raise SystemExit(f"la v2 no registra la partición de {sin_particion_v2}: no hay con qué "
                         f"comparar. No se mide")

    plan = c6.plan_de_la_pasada(datasets, protocolo, n_motores=1)
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]} "
          f"(v4 {protocolos['v4']['digest_sha256_declarado'][:16]}, enmienda 1 "
          f"{protocolos['enmienda_1']['digest_sha256_declarado'][:16]})")
    print(f"motor {NOMBRE_MOTOR_NUEVO}: {configuracion_del_motor['leida_del_motor']} "
          f"(la de la enmienda 1). Salida ({tipo}): {ruta_salida}")
    _imprimir_estimacion_peor_caso(plan)

    cache_previo, payload_previo = cargar_salida_previa(ruta_salida, tipo)
    if args.forzar:
        cache_previo = {}
    previos = list(payload_previo.get("resultados") or [])

    # UNA vez, al arrancar: los registros llevan ESTE digest, y el resultado
    # tiene que describir el mismo aunque un fichero cambie a media pasada.
    componentes_del_digest = componentes_del_digest_del_entorno()
    entorno_digest = _digest_de_componentes(componentes_del_digest)
    digest_del_motor = _digest_motor_nuevo()
    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor_nuevo": digest_del_motor},
        datos_de_entrada={
            **{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
            "protocolo_exploratorio_v2": RUTA_DEL_PROTOCOLO_V2,
            "protocolo_119_v4": RUTA_DEL_PROTOCOLO_V4,
            "protocolo_119_v4_enmienda_1": RUTA_DE_LA_ENMIENDA_1,
            # En la PROCEDENCIA sí (su sha256 ata el veredicto a sus reglas);
            # en el digest de la caché no: solo cambia cómo se cuenta.
            "protocolo_119_v4_enmienda_2": RUTA_DE_LA_ENMIENDA_2,
            "pasada_v2_113_resultado": RUTA_V2_RESULTADO,
        })
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = c3.procedencia_declarada(payload_previo)
        print(f"fichero previo: {len(previos)} registros, procedencia "
              f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)

    resultados: list[dict] = []
    veredictos_por_conjunto: list[dict] = []
    particiones: dict[str, dict] = {}
    comparacion_de_particiones: dict[str, dict] = {}
    pliegues_digest: dict[str, str] = {}
    tiempos_del_padre: dict[str, dict] = {}
    esperadas_por_conjunto: dict[str, list] = {}
    reusados = [0]
    inicio = time.perf_counter()
    regla = protocolo.regla_de_cierre
    nombres_de_los_conjuntos = [d.nombre for d in datasets]
    ultimo_punto_de_control = [time.perf_counter()]

    def guardar(parcial: bool) -> dict:
        ultimo_punto_de_control[0] = time.perf_counter()
        veredicto = None
        if not parcial:
            veredicto = veredicto_final(
                resultados_v2=resultados_v2, resultados_del_motor_nuevo=resultados,
                veredictos_por_conjunto=veredictos_por_conjunto,
                esperadas_por_conjunto=esperadas_por_conjunto, regla=regla,
                metrica_por_dataset=metrica_por_dataset,
                nombres_de_los_conjuntos=nombres_de_los_conjuntos)
        return _componer_y_guardar(
            resultados, previos, veredictos_por_conjunto, veredicto, procedencia, payload_previo,
            ruta_salida, tipo=tipo, datasets=datasets, protocolo=protocolo,
            metrica_por_dataset=metrica_por_dataset, subconjunto=subconjunto, plan=plan,
            particiones=particiones, comparacion_de_particiones=comparacion_de_particiones,
            pliegues_digest=pliegues_digest, tiempos_del_padre=tiempos_del_padre,
            esperadas_por_conjunto=esperadas_por_conjunto, protocolos=protocolos,
            configuracion_del_motor=configuracion_del_motor, enmienda_2=enmienda_2,
            componentes_del_digest=componentes_del_digest,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados[0], parcial=parcial)

    for ds in datasets:
        t_carga = time.perf_counter()
        por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(
            ds, protocolo)
        carga_s = time.perf_counter() - t_carga
        comparacion_de_particiones[ds.nombre] = exigir_la_particion_de_la_v2(
            ds, particion, c3.LECTURA_DECLARADA.get(str(ds.data_id)), payload_v2)
        particiones[ds.nombre] = _normalizado(particion)
        pliegues_digest[ds.nombre] = propuesta.pliegues.digest()
        test_ids = propuesta.plan.observaciones_del_rol("test")
        metric_id = metrica_por_dataset[ds.nombre]
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        intentos = plan_de_intentos(ds, propuesta, protocolo, humo=humo)
        esperadas_por_conjunto[ds.nombre] = [(i.repeticion, i.pliegue) for i in intentos]
        tiempos = tiempos_del_padre[ds.nombre] = {"carga_s": round(carga_s, 3),
                                                  "preparacion_s": 0.0, "n_preparaciones": 0}
        print(f"\n=== {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"n={len(por_id)}, test={len(test_ids)}, intentos={len(intentos)}, "
              f"metrica={metric_id}, carga={carga_s:.1f}s) ===", flush=True)

        registros_del_dataset: list[dict] = []
        for repeticion in sorted({i.repeticion for i in intentos}):
            for ip in [i for i in intentos if i.repeticion == repeticion]:
                clave = (ds.nombre, motor.nombre, ip.repeticion, ip.pliegue)
                previo = cache_previo.get(clave)
                if not humo and _reusable_c3(previo, entorno_digest, digest_del_motor,
                                             ip.presupuesto_wall_s, datos_sha256=datos_sha256):
                    registro = dict(previo, reusado=True)
                    registro.setdefault("procedencia_id", None)
                    resultados.append(registro)
                    registros_del_dataset.append(registro)
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} rep={ip.repeticion} pliegue={ip.pliegue}: "
                          f"estado={registro.get('estado')}", flush=True)
                    continue

                t_prep = time.perf_counter()
                crudas_train = [por_id[i] for i in ip.entrena]
                crudas_val = [por_id[i] for i in ip.valida]
                crudas_test = [por_id[i] for i in test_ids]
                transformadas = c3.preparar_para_motor(
                    crudas_train, crudas_train + crudas_val + crudas_test,
                    objetivo, predictores, motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)

                def hacer(xs):
                    return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

                train, val, test = (hacer(transformadas[:n_tr]),
                                    hacer(transformadas[n_tr:n_tr + n_va]),
                                    hacer(transformadas[n_tr + n_va:]))
                preparacion_s = time.perf_counter() - t_prep
                tiempos["preparacion_s"] = round(tiempos["preparacion_s"] + preparacion_s, 3)
                tiempos["n_preparaciones"] += 1

                presupuesto = Presupuesto(wall_seconds=ip.presupuesto_wall_s,
                                          hilos=c3.HILOS_POR_INTENTO, seed=ip.semilla)
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto,
                    candidate=f"{motor.nombre}-{c3.CONFIGURACION_UNICA}",
                    split_plan_digest=propuesta.plan.digest(),
                    dataset=ds.nombre, pliegue=ip.pliegue, repeticion=ip.repeticion)
                transcurrido = time.perf_counter() - t0

                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                # Lo que el motor DECLARA que pasó (de `config_efectiva`),
                # NUNCA los pesos: decenas de MB por intento.
                config = intento.config_efectiva or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                    "tarea": ds.tarea, "sellado": ds.sellado, "motor": motor.nombre,
                    "repeticion": ip.repeticion, "pliegue": ip.pliegue, "estado": intento.estado,
                    "semilla": ip.semilla, "presupuesto_wall_s": ip.presupuesto_wall_s,
                    "configuracion": c3.CONFIGURACION_UNICA, "metrica_de_cierre": metric_id,
                    "wall_s": round(transcurrido, 3), "preparacion_s": round(preparacion_s, 3),
                    "metricas": metricas, "tiempo_de_ajuste": recursos.get("wall_seconds"),
                    "cpu_segundos": recursos.get("cpu_seconds"),
                    "rss_pico_mb": recursos.get("peak_ram_mb"),
                    "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
                    "arquitectura": config.get("arquitectura"),
                    "hiperparametros": config.get("hiperparametros"),
                    "motivo": (intento.motivo_del_estado["es"]
                               if intento.motivo_del_estado else None),
                    "traza": intento.traza, "engine_version": intento.engine_version,
                    "pipeline_digest": intento.pipeline_digest,
                    "split_plan_digest": propuesta.plan.digest(),
                    "entorno_digest": entorno_digest, "motor_digest": digest_del_motor,
                    "datos_sha256": datos_sha256,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False,
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                resultados.append(registro)
                registros_del_dataset.append(registro)
                try:
                    exigir_la_arquitectura_del_intento(
                        registro, receta=configuracion_del_motor["receta_del_protocolo"],
                        n_train=n_tr)
                except SystemExit:
                    guardar(parcial=True)
                    raise
                print(f"  {ds.nombre} rep={ip.repeticion} pliegue={ip.pliegue}: "
                      f"estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"wall={transcurrido:.1f}s prep={preparacion_s:.1f}s "
                      f"entrenamiento_efectivo={registro['entrenamiento_efectivo']}", flush=True)
                if (time.perf_counter() - ultimo_punto_de_control[0]
                        >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL):
                    guardar(parcial=True)
            if not humo:
                guardar(parcial=True)

        v2_densa_del_dataset = [r for r in resultados_v2
                                if r["dataset"] == ds.nombre and r["motor"] == NOMBRE_DENSA_V2]
        veredicto_ds = veredicto_del_conjunto_c3(
            dataset=ds.nombre, metric_id=metric_id,
            registros_del_motor_nuevo=registros_del_dataset,
            registros_de_la_densa_v2=v2_densa_del_dataset,
            esperadas=esperadas_por_conjunto[ds.nombre])
        veredictos_por_conjunto.append(veredicto_ds)
        print(f"  VEREDICTO {ds.nombre}: mejora={veredicto_ds['mejora']} "
              f"inferioridad={veredicto_ds['inferioridad']} "
              f"sin_comparacion={veredicto_ds['sin_comparacion']} "
              f"fallos_del_nuevo={veredicto_ds['fallos_del_motor_nuevo']['n']} "
              f"motivo={veredicto_ds['motivo']}", flush=True)
        guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados[0]} reusados, {len(resultados)-reusados[0]} ejecutados) ===")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    v = salida["veredicto"]
    print(f"VEREDICTO 119-C3{' (' + tipo.upper() + ' -- no cuenta)' if tipo != 'pasada' else ''}: "
          f"cumplidos {v['cumplidos_con_el_motor_nuevo']['cumplidos']} vs densa v2 "
          f"{v['cumplidos_de_la_densa_v2_en_los_mismos_conjuntos']['cumplidos']} "
          f"(sube={v['sube_los_cumplidos']}); mejora={v['n_mejora']} "
          f"inferioridad={v['n_inferioridad']} sin_comparacion={v['n_sin_comparacion']} -> "
          f"{'SUBE a confirmacion' if v['cumple_la_regla_de_subida'] else 'NO SUBE'}")


def _digest_fichero_json(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _escribir_atomicamente(ruta: Path, salida: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta)


def _componer_y_guardar(resultados, previos, veredictos_por_conjunto, veredicto, procedencia,
                        payload_previo, ruta_salida, *, tipo, datasets, protocolo,
                        metrica_por_dataset, subconjunto, plan, particiones,
                        comparacion_de_particiones, pliegues_digest, tiempos_del_padre,
                        esperadas_por_conjunto, protocolos, configuracion_del_motor, enmienda_2,
                        componentes_del_digest, total_wall_s, reusados, parcial) -> dict:
    """Compone el JSON y lo escribe atómicamente, FUSIONADO con lo que ya
    había (I7): lo previo que esta ejecución no ha vuelto a tocar se
    conserva, con su procedencia."""
    en_el_fichero, conservados, sustituidos_sin_medida = fusionar_resultados(previos, resultados)
    procedencias_previas = payload_previo.get("procedencias") or {}
    reintentados = rastro_de_los_reintentos(
        sustituidos_sin_medida, resultados,
        rastro_previo=list(payload_previo.get("intentos_reintentados") or []),
        procedencias_previas=procedencias_previas)
    procedencias, sin_procedencia = c3._procedencias_citadas(
        en_el_fichero, procedencia, procedencias_previas)
    for t in reintentados:  # la procedencia del intento que falló, también (su «cuándo»)
        pid = t.get("procedencia_id")
        if pid and pid not in procedencias and pid in procedencias_previas:
            procedencias[pid] = procedencias_previas[pid]
    nombres_de_esta_ejecucion = {d.nombre for d in datasets}
    reintentados_del_nuevo = [t for t in reintentados if t["motor"] == NOMBRE_MOTOR_NUEVO
                              and t["dataset"] in nombres_de_esta_ejecucion]
    if veredicto is not None:
        por_conjunto: dict[str, int] = {}
        for t in reintentados_del_nuevo:
            por_conjunto[t["dataset"]] = por_conjunto.get(t["dataset"], 0) + 1
        veredicto = {**veredicto,
                     "conjuntos_con_intentos_reintentados": sorted(por_conjunto),
                     "intentos_reintentados_por_conjunto": dict(sorted(por_conjunto.items())),
                     "que_son_los_intentos_reintentados": QUE_SON_LOS_REINTENTADOS}
    enmienda_1 = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    reconciliacion = (
        {"no_aplica": "--humo corre 1 repetición y 1 pliegue por conjunto: el plan del "
                      "protocolo no es el de esta ejecución (ver intentos_por_conjunto)"}
        if tipo == "humo" else
        c5.reconciliar_el_plan_con_lo_medido(plan, particiones, resultados, n_motores=1))
    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "119-C3",
        "tipo_de_ejecucion": tipo,
        "motor": NOMBRE_MOTOR_NUEVO,
        "configuracion_del_motor": configuracion_del_motor,
        "protocolo_119_v4_digest_sha256": _digest_fichero_json(RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": _digest_fichero_json(RUTA_DE_LA_ENMIENDA_1),
        "protocolos_encadenados": protocolos,
        "enmienda_2": enmienda_2,
        "allstate_declarado": enmienda_1.get("allstate_declarado"),
        "allstate_medido": allstate_medido(resultados),
        "parada_temprana_declarada": parada_temprana_declarada(resultados),
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial, "es_humo": tipo == "humo",
        "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "plan": plan,
        "intentos_por_conjunto": intentos_por_conjunto(
            datasets, particiones, esperadas_por_conjunto, resultados, protocolo),
        "por_que_n_intentos_no_es_el_del_plan": reconciliacion,
        "particion_por_dataset": dict(particiones),
        "particion_comparada_con_la_v2": dict(comparacion_de_particiones),
        "pliegues_digest_por_dataset": dict(pliegues_digest),
        "pliegues_digest_que_es": ("digest de la asignación de pliegues (entrena/valida por "
                                   "repetición): la v2 no lo registró, así que con la v2 solo se "
                                   "compara `particion_por_dataset` (plan_digest incluido)"),
        "tiempos_del_padre_por_dataset": dict(tiempos_del_padre),
        "tiempos_del_padre_que_son": ("carga_s = leer el ARFF y particionar (una vez por "
                                      "conjunto); preparacion_s = preparar_para_motor y construir "
                                      "las particiones de los intentos EJECUTADOS. Ninguno está "
                                      "dentro de wall_s, que es solo ejecutar_intento_aislado"),
        "criterio_de_los_conjuntos": (
            "los 32 NO sellados del protocolo v4 (idénticos en particion/presupuesto/"
            "regla_de_cierre/datasets a protocolo_exploratorio_v2.json, comprobado al arrancar), "
            "o el subconjunto de --solo o --humo, declarado arriba. Los 8 sellados se niegan"),
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados), "n_reusados": reusados,
        "n_intentos_en_el_fichero": len(en_el_fichero),
        "n_registros_conservados_de_ejecuciones_anteriores": len(conservados),
        "registros_conservados_por_dataset": {
            d: sum(1 for r in conservados if r["dataset"] == d)
            for d in sorted({r["dataset"] for r in conservados})},
        "n_intentos_reintentados": len(reintentados),
        "conjuntos_con_intentos_reintentados": sorted({t["dataset"] for t in reintentados}),
        "que_son_los_intentos_reintentados": QUE_SON_LOS_REINTENTADOS,
        "intentos_reintentados": reintentados,
        "que_cubren_el_veredicto_y_n_intentos": (
            "SOLO los conjuntos de esta ejecución (datasets_declarados). `resultados` es el "
            "fichero fusionado: incluye los registros conservados de ejecuciones anteriores, "
            "que sirven de caché y no entran en el veredicto"),
        "digest_de_la_cache": {
            "entorno": procedencia["digests_de_codigo"].get("entorno"),
            "componentes": dict(componentes_del_digest),
            "motor_nuevo": {"digest": procedencia["digests_de_codigo"].get("motor_nuevo"),
                            "ficheros": [_etiqueta(f) for f in _FICHEROS_DEL_MOTOR_NUEVO]},
            "clave_de_cada_registro": ("entorno_digest, motor_digest, presupuesto_wall_s, "
                                       "estado que cuente como medida y datos_sha256 del ARFF "
                                       "(_reusable_c3)"),
        },
        "lo_que_no_cubre_el_digest": lo_que_no_cubre_el_digest(),
        "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
        "resultados": en_el_fichero,
        "veredicto_por_conjunto": veredictos_por_conjunto,
        "veredicto": veredicto,
    }
    c3.sellar_la_salida(salida)
    _escribir_atomicamente(ruta_salida, salida)
    return salida


# ---------------------------------------------------------------------------
# 8. --estimar (I1): UN intento real por conjunto de una lista con más de un
#    conjunto por cubo, la preparación del padre aparte, y si cabe en la
#    ventana nocturna o de día. 1 proceso: el guion es secuencial.
# ---------------------------------------------------------------------------

#: Más de uno por cubo, y los ANCHOS, que son los caros para TabM+PLR (cada
#: numérica es un embedding PLR): pequeño fino / 216 / 1300 columnas;
#: mediano fino / más filas / 1558 columnas; grande fino / el más caro, que
#: es el que la enmienda 1 declara que no llega a las 17 épocas.
CONJUNTOS_DE_LA_ESTIMACION = (
    "balance-scale", "mfeat-factors", "micro-mass",
    "wilt", "pendigits", "Internet-Advertisements",
    "house_16H", "Allstate_Claims_Severity",
)

#: La ventana de la cola nocturna: cron a las 00:30, la suite nocturna
#: (~1 h 45) primero, y ningún trabajo empieza después de las 08:00. El
#: «01:45» es lo SUPUESTO, y solo se usa si no hay un fin de la suite MEDIDO
#: (`fin_de_la_suite_nocturna_medido`): el 30-09 acabó a las 01:57:12 y
#: crece cada día (auditoría 3, I1).
VENTANA_NOCTURNA = ("01:45", "08:00")
#: De día (`COLA_HASTA=23`): ningún trabajo cuya estimación pase de las 23:00.
HORA_LIMITE_DE_DIA = "23:00"
HORA_DE_INICIO_DE_DIA = "08:00"

# --- I1 (auditoría 3): la orden de encolado que sirve -----------------------
#: `~/cola-nocturna.sh` (LEÍDO el 30-09, no supuesto): se salta un trabajo si
#: su ESTIMADA_S pasa de lo que queda hasta las 08:00 (de noche) o hasta
#: COLA_HASTA (de día, 23:00); después espera hasta 1 h a que la carga baje
#: de 4 (120 × 30 s), y lo corre con `timeout <tope>`. Con el candado cogido:
#: la nocturna de las 00:30 que lo encuentre NO corre (ni sus suites).
RESUMEN_DE_LA_COLA = Path.home() / "cola-nocturna" / "resumen.txt"
NOCHES_QUE_SE_MIRAN = 7
#: Sobre el fin MÁS TARDÍO medido: la suite crece (+12 min del 29 al 30-09).
MARGEN_SOBRE_EL_FIN_DE_LA_SUITE_S = 15 * 60
ESPERA_MAXIMA_POR_CARGA_S = 3600
HORA_DE_LA_COLA_NOCTURNA = "00:30"
#: EL TOPE DE DÍA: la estimación × 1,25 (y al menos +1 h), redondeado a
#: 100 s. Por qué ese margen, y no otro:
#: * el tope no es la estimación: con tope = estimada, cualquier desviación
#:   al alza corta la pasada (la orden de antes ponía 22.500 con 7,13 h
#:   estimadas: el corte estaba garantizado);
#: * la estimación mide UN intento de 15 en 8 conjuntos y SUPONE los otros
#:   24 por celdas; lo medido dos veces coincide (Allstate 410,1 y 412,9 s),
#:   la preparación del padre de KDDCup09 va +5 min por encima (M4), y un
#:   intento en carga puede doblar su duración (micro-mass, I5): un 25 %
#:   (1,8 h sobre 7,13) cubre que TODOS los supuestos salgan un 40 % más
#:   caros (son 4,07 de las 7,13 h en la estimación del 30-09);
#: * la cota de peor caso (cada intento su presupuesto entero, 29,8 h) no
#:   sirve de tope: no cabe en ningún día;
#: * un tope corto no pierde nada (el caché reanuda: se encola la
#:   continuación con COMMITS=), pero cuesta otra ventana; uno largo solo
#:   cuesta si algo se cuelga. Y lo acota la nocturna: lanzada de día, la
#:   pasada tiene que acabar antes de las 00:30.
MARGEN_DEL_TOPE_SOBRE_LA_ESTIMACION = 1.25
MARGEN_MINIMO_DEL_TOPE_S = 3600
SALIDA_EN_LA_COLA = ("/home/deployer/cola-nocturna/resultados/119-c3/"
                     "pasada_119_c3_resultado.json")
ORDEN_DE_LA_COLA_DE_DIA = ("setsid nohup env COLA_SIN_SUITE=1 COLA_HASTA=23 ~/cola-nocturna.sh "
                           ">> ~/cola-nocturna/dia.log 2>&1 < /dev/null & disown")


def _hhmmss(segundos: int) -> str:
    segundos = int(segundos) % 86400
    return f"{segundos // 3600:02d}:{segundos % 3600 // 60:02d}:{segundos % 60:02d}"


def _segundos_del_dia(hora: str) -> int:
    partes = [int(p) for p in hora.split(":")] + [0, 0]
    return partes[0] * 3600 + partes[1] * 60 + partes[2]


def fin_de_la_suite_nocturna_medido(ruta: Path | None = None) -> dict:
    """La hora a la que ACABÓ la suite nocturna las últimas noches, leída de
    `~/cola-nocturna/resumen.txt` (sus líneas «suite nocturna: AAAA-MM-DD
    HH:MM:SS · …» escritas al acabarla; la cola de día las repite y no
    cuentan), de las últimas `NOCHES_QUE_SE_MIRAN`. La ventana nocturna
    empieza en la MÁS TARDÍA + un margen. Sin fichero o sin líneas, lo
    supuesto (01:45), diciéndolo."""
    ruta = Path(ruta if ruta is not None else RESUMEN_DE_LA_COLA)
    medidos: dict[str, str] = {}
    try:
        texto = ruta.read_text(encoding="utf-8", errors="replace")
    except OSError:
        texto = ""
    # Solo la línea que la cola escribe AL ACABAR su suite (su hora = la del fin
    # de la suite, ±5 min): las de la cola de día repiten la última suite, y la
    # del 25-09 (13:18) repetía una de antes de que la cola existiera (04:19).
    patron = (r"(?m)^(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)\S* suite nocturna: "
              r"(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)")
    for fecha_linea, hora_linea, fecha, hora in re.findall(patron, texto):
        if fecha_linea == fecha and abs(_segundos_del_dia(hora_linea)
                                        - _segundos_del_dia(hora)) <= 300:
            medidos[fecha] = hora
    ultimas = dict(sorted(medidos.items())[-NOCHES_QUE_SE_MIRAN:])
    if not ultimas:
        return {"medido": False, "desde": VENTANA_NOCTURNA[0], "fuente": str(ruta),
                "motivo": "sin fin de la suite medido: se usa lo supuesto (01:45), que el 30-09 "
                          "ya se quedaba corto (acabó a las 01:57:12)"}
    mas_tarde = max(ultimas.values(), key=_segundos_del_dia)
    return {"medido": True, "fuente": str(ruta), "noches": ultimas, "mas_tarde": mas_tarde,
            "margen_s": MARGEN_SOBRE_EL_FIN_DE_LA_SUITE_S,
            "desde": _hhmmss(_segundos_del_dia(mas_tarde) + MARGEN_SOBRE_EL_FIN_DE_LA_SUITE_S)}


def _commits_de_ahora() -> dict:
    """El HEAD de los dos repos al estimar (los que la cola fija si no se
    le dice otro) y si tienen cambios sin commitear (que la cola NO lleva)."""
    import subprocess  # noqa: PLC0415

    salida = {}
    for nombre, raiz in (("matrixAI", c3._RAIZ_DEL_CORE), ("matrixai-engines",
                                                           c3._RAIZ_DE_ENGINES.parent)):
        try:
            sha = subprocess.run(["git", "-C", str(raiz), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, timeout=30, check=True).stdout.strip()
            sucio = bool(subprocess.run(["git", "-C", str(raiz), "status", "--short",
                                         "--untracked-files=no"], capture_output=True, text=True,
                                        timeout=30).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            sha, sucio = None, None
        salida[nombre] = {"sha": sha, "sin_commitear": sucio}
    return salida


def para_encolar(estimada_s: float, *, memoria: str, commits: dict,
                 fin_de_la_suite: dict) -> dict:
    """Las órdenes de encolado, PURA (se prueba con números fabricados).

    DE DÍA (`COLA_SIN_SUITE=1 COLA_HASTA=23`), lo recomendado: ESTIMADA_S =
    la estimación ENTERA (sin recortar a ninguna ventana) y el tope con su
    margen; se lanza antes de la hora a la que ya no cabría (la cola no la
    empieza si la estimación pasa de las 23:00, y el tope —más la hora que
    puede esperar a la carga— tiene que acabar antes de las 00:30, o se come
    la nocturna). DE NOCHE, desde el fin de la suite MEDIDO: si no cabe, lo
    dice y da una orden por noche con los MISMOS commits (COMMITS=), para
    que la continuación reuse lo medido."""
    estimada = int(math.ceil(estimada_s))
    tope = int(math.ceil(max(estimada * MARGEN_DEL_TOPE_SOBRE_LA_ESTIMACION,
                             estimada + MARGEN_MINIMO_DEL_TOPE_S) / 100.0) * 100)
    shas = {n: (c or {}).get("sha") for n, c in commits.items()}
    fijar = ("COMMITS=" + ",".join(f"{n}={shas.get(n) or '<sha>'}"
                                   for n in ("matrixAI", "matrixai-engines")))

    def orden(nombre: str, est: int, tope_s: int) -> str:
        return (f"{fijar} ESTIMADA_S={est} ~/encolar.sh {nombre} {tope_s} {memoria} matrixAI "
                f"python3 benchmarks/fase0/pasada_119_c3.py --salida {SALIDA_EN_LA_COLA}")

    # De día: la última hora de lanzar la cola.
    por_la_estimacion = _segundos_del_dia(HORA_LIMITE_DE_DIA) - estimada
    por_la_nocturna = (86400 + _segundos_del_dia(HORA_DE_LA_COLA_NOCTURNA) - tope
                       - ESPERA_MAXIMA_POR_CARGA_S)
    ultima = min(por_la_estimacion, por_la_nocturna)
    cabe_de_dia = ultima >= _segundos_del_dia(HORA_DE_INICIO_DE_DIA)
    de_dia = {
        "estimada_s": estimada, "tope_s": tope,
        "margen_del_tope": (f"x{MARGEN_DEL_TOPE_SOBRE_LA_ESTIMACION} sobre la estimación, al "
                            f"menos +{MARGEN_MINIMO_DEL_TOPE_S} s (MARGEN_DEL_TOPE_SOBRE_LA_"
                            f"ESTIMACION: el porqué, en el guion)"),
        "cabe": cabe_de_dia,
        "encolar": orden("119-c3", estimada, tope),
        "lanzar_la_cola": ORDEN_DE_LA_COLA_DE_DIA,
        "lanzar_antes_de": _hhmmss(ultima) if cabe_de_dia else None,
        "por_que_esa_hora": (
            f"la cola no empieza un trabajo cuya ESTIMADA_S pase de las {HORA_LIMITE_DE_DIA} "
            f"(-> {_hhmmss(por_la_estimacion)}), y con el tope entero más la hora que puede "
            f"esperar a la carga tiene que acabar antes de las {HORA_DE_LA_COLA_NOCTURNA}, o la "
            f"nocturna encuentra el candado y no corre sus suites (-> {_hhmmss(por_la_nocturna)})"),
        "si_se_corta": ("rc=124 en ~/cola-nocturna/resumen.txt: lo medido queda en la salida "
                        "(punto de control por repetición); se encola la continuación con los "
                        "MISMOS commits y ESTIMADA_S = lo que falte: "
                        + orden("119-c3-b", "<lo que falte>", tope)),
    }
    desde = fin_de_la_suite["desde"]
    ventana = _segundos_del_dia(VENTANA_NOCTURNA[1]) - _segundos_del_dia(desde)
    noches = max(1, math.ceil(estimada / ventana))
    ordenes, restante = [], estimada
    for i in range(noches):
        ordenes.append(orden("119-c3" if i == 0 else f"119-c3-noche{i + 1}",
                             min(ventana, restante), ventana))
        restante -= ventana
    de_noche = {
        "desde": desde, "hasta": VENTANA_NOCTURNA[1], "segundos": ventana,
        "fin_de_la_suite": fin_de_la_suite,
        "cabe": estimada <= ventana, "noches": noches,
        "ordenes_una_por_noche": ordenes,
        "como": ("cada noche, la suite; después, el primer pendiente cuya ESTIMADA_S quepa hasta "
                 "las 08:00, cortado por su tope. ESTIMADA_S de cada noche = la ventana (o lo que "
                 "falte): solo decide si arranca. Las continuaciones llevan COMMITS= para correr "
                 "el MISMO código y que el caché reconozca lo ya medido"),
    }
    if estimada > ventana:
        de_noche["aviso"] = (f"NO CABE EN UNA NOCHE: {estimada} s estimados y la ventana, desde el "
                             f"fin de la suite medido ({desde}) hasta las {VENTANA_NOCTURNA[1]}, "
                             f"es de {ventana} s: {noches} noches, una orden por noche, las "
                             f"de continuación con COMMITS=")
    return {
        "recomendada": "de_dia" if cabe_de_dia else "de_noche",
        "de_dia": de_dia, "de_noche": de_noche,
        "commits_al_estimar": commits,
        "aviso_de_commits": ("los HEAD de ahora: si se encola otro commit, se vuelve a estimar "
                             "(o se ponen los suyos). Lo no commiteado NO entra en la cola"),
        "por_que_la_salida_fuera_del_arbol": (
            "la cola borra sus worktrees al terminar: con la salida dentro, la noche siguiente "
            "no encontraría el caché y lo mediría todo otra vez"),
        "por_que_COMMITS": ("las continuaciones con el MISMO commit: con otro, el digest del "
                            "entorno puede cambiar e invalidar lo ya medido"),
    }


# --- M3 (auditoría 3): la memoria, medida y no inflada -----------------------
_PAGINA_KB = resource.getpagesize() // 1024


def _rss_del_arbol_kb(raiz: int) -> int:
    """La SUMA del RSS de `raiz` y de todos sus descendientes, leída de /proc."""
    import os  # noqa: PLC0415

    hijos: dict[int, list[int]] = {}
    for entrada in os.scandir("/proc"):
        if not entrada.name.isdigit():
            continue
        try:
            with open(f"/proc/{entrada.name}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue
        campos = stat[stat.rfind(b")") + 2:].split()
        hijos.setdefault(int(campos[1]), []).append(int(entrada.name))
    total, pendientes, vistos = 0, [raiz], set()
    while pendientes:
        pid = pendientes.pop()
        if pid in vistos:
            continue
        vistos.add(pid)
        try:
            with open(f"/proc/{pid}/statm", "rb") as f:
                total += int(f.read().split()[1]) * _PAGINA_KB
        except (OSError, IndexError, ValueError):
            continue
        pendientes.extend(hijos.get(pid, []))
    return total


class _PicoDeMemoriaDelArbol:
    """Muestrea cada `cada_s` la suma del RSS del proceso y sus descendientes
    y se queda con el pico: lo que un `MemoryMax` ve de verdad (más o menos:
    las páginas compartidas cuentan dos veces y la caché de ficheros no
    cuenta). `ru_maxrss` de los hijos NO sirve: hereda el RSS del padre en el
    fork previo al exec (medido: padre 2.816,8 MB = hijos 2.816,8 MB)."""

    def __init__(self, cada_s: float = 0.5):
        import os  # noqa: PLC0415
        import threading  # noqa: PLC0415

        self._pid, self._cada_s, self.pico_kb = os.getpid(), cada_s, 0
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)

    def _muestrear(self) -> None:
        while True:
            try:
                self.pico_kb = max(self.pico_kb, _rss_del_arbol_kb(self._pid))
            except Exception:  # noqa: BLE001 -- medir no puede tumbar la estimación
                pass
            if self._parar.wait(self._cada_s):
                return

    def __enter__(self):
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        self._hilo.join(timeout=10)
        self.pico_kb = max(self.pico_kb, _rss_del_arbol_kb(self._pid))

    @property
    def pico_mb(self) -> float:
        return round(self.pico_kb / 1024, 1)


def _segundos_entre(desde: str, hasta: str) -> int:
    h1, m1 = map(int, desde.split(":"))
    h2, m2 = map(int, hasta.split(":"))
    return ((h2 * 60 + m2) - (h1 * 60 + m1)) * 60


def _interpolar(x: float, puntos: list[tuple[float, float]]) -> float | None:
    """Lineal entre los dos medidos que lo rodean; por debajo del menor, el
    menor; por ENCIMA del mayor, `None` (no se extrapola: se usa la cota)."""
    puntos = sorted(puntos)
    if not puntos:
        return None
    if x <= puntos[0][0]:
        return puntos[0][1]
    for (x0, y0), (x1, y1) in zip(puntos, puntos[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def _escalar_por_celdas(x: float, puntos: list[tuple[float, float]]) -> float:
    """Proporcional a las celdas desde el medido más cercano en escala
    logarítmica (cargar y preparar escalan con el tamaño, no con el cubo)."""
    if not puntos:
        return 0.0
    cercano = min(puntos, key=lambda p: abs(math.log(max(p[0], 1)) - math.log(max(x, 1))))
    return cercano[1] * x / max(cercano[0], 1)


def estimar_desde_medidas(medidas: dict[str, dict], conjuntos: list[dict], *, margen_s: float,
                          ventana_nocturna_s: int, ventana_de_dia_s: int,
                          ventana_de_dia_desde_ahora_s: int | None = None,
                          ventana_nocturna_desde: str = VENTANA_NOCTURNA[0]) -> dict:
    """La cuenta, PURA (se prueba con medidas fabricadas).

    `conjuntos`: [{nombre, cubo, celdas, n_intentos, presupuesto_wall_s}].
    `medidas`: {nombre: {cubo, celdas, wall_s, preparacion_s, carga_s, estado}}.
    Por conjunto: carga + n_intentos × (preparación + intento). El intento, el
    medido si se midió; si no, interpolado por celdas entre los medidos de
    SU cubo, o la cota (tope + margen) por encima del mayor. Nunca más que
    la cota: el subproceso mata el intento ahí."""
    por_cubo: dict[str, list] = {}
    for nombre, m in medidas.items():
        por_cubo.setdefault(m["cubo"], []).append((m["celdas"], m["wall_s"]))
    cargas = [(m["celdas"], m["carga_s"]) for m in medidas.values()]
    preparaciones = [(m["celdas"], m["preparacion_s"]) for m in medidas.values()]
    detalle, total_s, cota_s = [], 0.0, 0.0
    for c in conjuntos:
        techo = c["presupuesto_wall_s"] + margen_s
        if c["nombre"] in medidas:
            m = medidas[c["nombre"]]
            intento, carga, prep, fuente = m["wall_s"], m["carga_s"], m["preparacion_s"], "medido"
        else:
            intento = _interpolar(c["celdas"], por_cubo.get(c["cubo"], []))
            fuente = "interpolado por celdas en su cubo"
            if intento is None:
                intento, fuente = techo, "cota: más celdas que cualquier medido de su cubo"
            carga = _escalar_por_celdas(c["celdas"], cargas)
            prep = _escalar_por_celdas(c["celdas"], preparaciones)
        intento = min(intento, techo)
        segundos = carga + c["n_intentos"] * (prep + intento)
        cota = carga + c["n_intentos"] * (prep + techo)
        total_s += segundos
        cota_s += cota
        detalle.append({**c, "s_por_intento": round(intento, 1), "carga_s": round(carga, 1),
                        "preparacion_s_por_intento": round(prep, 2), "fuente": fuente,
                        "horas": round(segundos / 3600, 3), "horas_cota": round(cota / 3600, 3)})
    fallidos = sorted(n for n, m in medidas.items()
                      if m.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA)
    salida = {
        "procesos": 1,
        "por_que_1_proceso": "el guion corre los intentos de uno en uno; no hay otra cifra medida",
        "por_conjunto": detalle,
        "total_horas": round(total_s / 3600, 2),
        "total_horas_cota": round(cota_s / 3600, 2),
        "ventana_nocturna": {"desde": ventana_nocturna_desde, "hasta": VENTANA_NOCTURNA[1],
                             "segundos": ventana_nocturna_s,
                             "cabe": total_s <= ventana_nocturna_s,
                             "noches_necesarias": max(1, math.ceil(total_s / ventana_nocturna_s)),
                             "cabe_la_cota": cota_s <= ventana_nocturna_s},
        "de_dia": {"desde": HORA_DE_INICIO_DE_DIA, "hasta": HORA_LIMITE_DE_DIA,
                   "segundos": ventana_de_dia_s, "cabe": total_s <= ventana_de_dia_s,
                   "como": "COLA_SIN_SUITE=1 COLA_HASTA=23 ~/cola-nocturna.sh (CLAUDE.md)"},
        "intentos_que_fallaron_al_estimar": fallidos,
        "que_supone": (
            "un conjunto no medido tarda por intento lo que dicen, interpolados por celdas "
            "(filas x predictores), los medidos de SU cubo; cargar y preparar escalan con las "
            "celdas desde el medido más cercano. Las épocas dependen de la paciencia y de los "
            "datos, que las celdas no ven: es una estimación, no una medición"),
    }
    if ventana_de_dia_desde_ahora_s is not None:
        salida["de_dia"]["segundos_desde_ahora"] = ventana_de_dia_desde_ahora_s
        salida["de_dia"]["cabe_desde_ahora"] = total_s <= ventana_de_dia_desde_ahora_s
    if fallidos:
        salida["aviso_de_fallos"] = (
            f"{fallidos} FALLARON al estimar: en la pasada, cada intento así es un fallo del "
            f"motor nuevo y su conjunto cuenta como perdido (condición 1) e inferioridad "
            f"(condición 2)")
    return salida


def _medir_un_intento_real(ds, protocolo, motor, payload_v2) -> dict:
    """UN intento real (repetición 0, pliegue 0) con el presupuesto REAL de su
    cubo y 4 hilos, con la carga y la preparación del padre medidas aparte, y
    el pico de memoria del ÁRBOL de procesos (padre + hijo) muestreado."""
    with _PicoDeMemoriaDelArbol() as memoria:
        medida = _medir_un_intento_real_sin_memoria(ds, protocolo, motor, payload_v2)
    medida["rss_pico_mb_del_arbol_de_procesos"] = memoria.pico_mb
    return medida


def _medir_un_intento_real_sin_memoria(ds, protocolo, motor, payload_v2) -> dict:
    t_carga = time.perf_counter()
    por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
    carga_s = time.perf_counter() - t_carga
    exigir_la_particion_de_la_v2(ds, particion, c3.LECTURA_DECLARADA.get(str(ds.data_id)),
                                 payload_v2)
    intentos = plan_de_intentos(ds, propuesta, protocolo, humo=True)
    if not intentos:
        raise SystemExit(f"{ds.nombre}: no hay pliegue (0, 0) en su partición")
    ip = intentos[0]
    t_prep = time.perf_counter()
    test_ids = propuesta.plan.observaciones_del_rol("test")
    crudas_train = [por_id[i] for i in ip.entrena]
    crudas_val = [por_id[i] for i in ip.valida]
    crudas_test = [por_id[i] for i in test_ids]
    transformadas = c3.preparar_para_motor(crudas_train, crudas_train + crudas_val + crudas_test,
                                           objetivo, predictores, motor)
    n_tr, n_va = len(crudas_train), len(crudas_val)

    def hacer(xs):
        return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

    train, val, test = (hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]),
                        hacer(transformadas[n_tr + n_va:]))
    preparacion_s = time.perf_counter() - t_prep
    presupuesto = Presupuesto(wall_seconds=ip.presupuesto_wall_s, hilos=c3.HILOS_POR_INTENTO,
                              seed=ip.semilla)
    t0 = time.perf_counter()
    intento = ejecutar_intento_aislado(
        motor, train, val, test, spec, presupuesto,
        candidate=f"{motor.nombre}-estimacion-119-c3", split_plan_digest=propuesta.plan.digest(),
        dataset=ds.nombre, pliegue=ip.pliegue, repeticion=ip.repeticion)
    wall_s = time.perf_counter() - t0
    config = intento.config_efectiva or {}
    recursos = intento.recursos or {}
    return {
        "dataset": ds.nombre, "cubo": ds.cubo, "n_filas": particion["n_filas_con_objetivo"],
        "n_predictores": particion["n_predictores"],
        "celdas": particion["n_filas_con_objetivo"] * particion["n_predictores"],
        "presupuesto_wall_s": ip.presupuesto_wall_s, "hilos": c3.HILOS_POR_INTENTO,
        "estado": intento.estado, "motivo": (intento.motivo_del_estado or {}).get("es"),
        "wall_s": round(wall_s, 3), "carga_s": round(carga_s, 3),
        "preparacion_s": round(preparacion_s, 3),
        "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
        "arquitectura": config.get("arquitectura"),
        "rss_pico_mb_del_hijo": recursos.get("peak_ram_mb"),
        "maxrss_mb_del_padre": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        # NO es el pico del hijo: hereda el RSS del padre en el fork previo al
        # exec (M3). Se deja, con su nombre; la memoria sale del muestreo.
        "maxrss_mb_de_los_hijos_hereda_el_del_padre": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
    }


def _conjuntos_para_estimar(no_sellados, protocolo, payload_v2) -> list[dict]:
    particiones_v2 = payload_v2.get("particion_por_dataset") or {}
    salida = []
    for d in no_sellados:
        p = particiones_v2.get(d.nombre)
        if p is None:
            raise SystemExit(f"{d.nombre}: la v2 no registra su partición; no se puede estimar")
        salida.append({"nombre": d.nombre, "cubo": d.cubo,
                       "celdas": p["n_filas_con_objetivo"] * p["n_predictores"],
                       "n_intentos": p["n_pliegues_obtenidos"],
                       "presupuesto_wall_s": c3.wall_seconds_del_cubo(d.cubo, protocolo)})
    return salida


def _segundos_hasta(hora: str, ahora: time.struct_time) -> int:
    h, m = map(int, hora.split(":"))
    return max(0, (h * 3600 + m * 60) - (ahora.tm_hour * 3600 + ahora.tm_min * 60 + ahora.tm_sec))


def cmd_estimar(protocolo, todos, no_sellados, *, ruta_salida: Path, conjuntos: str | None,
                protocolos: dict, configuracion_del_motor: dict, enmienda_2: dict) -> None:
    _exigir_que_quepa()
    if ruta_salida.exists() and tipo_del_fichero(_leer_json(ruta_salida)) != "estimar":
        raise SystemExit(f"{ruta_salida} no es una estimación: --estimar no lo pisa")
    motor = MotorDensaTabM()
    payload_v2 = _payload_v2()
    plan = c6.plan_de_la_pasada(no_sellados, protocolo, n_motores=1)
    _imprimir_estimacion_peor_caso(plan)
    nombres = ([n.strip() for n in conjuntos.split(",") if n.strip()] if conjuntos
               else list(CONJUNTOS_DE_LA_ESTIMACION))
    medir, _ = datasets_de_c3(todos, no_sellados, solo=",".join(nombres))
    sin_medida = sorted({d.cubo for d in no_sellados} - {d.cubo for d in medir})
    if sin_medida:
        print(f"AVISO: ningún conjunto medido en el cubo {sin_medida}: sus conjuntos irán a la "
              f"cota", flush=True)
    peor = sum(c3.wall_seconds_del_cubo(d.cubo, protocolo) + MARGEN_POR_DEFECTO_SEGUNDOS
               for d in medir)
    print(f"\nESTA MEDICIÓN: {len(medir)} intentos reales ({', '.join(d.nombre for d in medir)}), "
          f"1 proceso, 4 hilos; peor caso {peor/60:.0f} min sin contar la carga (Allstate: ~1,5 "
          f"min más). Techo de memoria: el del lanzador (systemd-run), no este guion",
          flush=True)
    medidas = {}
    for ds in medir:
        print(f"  midiendo {ds.nombre} ({ds.cubo})...", flush=True)
        m = medidas[ds.nombre] = _medir_un_intento_real(ds, protocolo, motor, payload_v2)
        print(f"  {ds.nombre}: intento {m['wall_s']:.1f}s, carga {m['carga_s']:.1f}s, prep "
              f"{m['preparacion_s']:.1f}s, estado={m['estado']}, "
              f"entrenamiento_efectivo={m['entrenamiento_efectivo']}", flush=True)
    ahora = time.localtime()
    fin_de_la_suite = fin_de_la_suite_nocturna_medido()
    estimacion = estimar_desde_medidas(
        medidas, _conjuntos_para_estimar(no_sellados, protocolo, payload_v2),
        margen_s=MARGEN_POR_DEFECTO_SEGUNDOS,
        ventana_nocturna_s=(_segundos_del_dia(VENTANA_NOCTURNA[1])
                            - _segundos_del_dia(fin_de_la_suite["desde"])),
        ventana_nocturna_desde=fin_de_la_suite["desde"],
        ventana_de_dia_s=_segundos_entre(HORA_DE_INICIO_DE_DIA, HORA_LIMITE_DE_DIA),
        ventana_de_dia_desde_ahora_s=_segundos_hasta(HORA_LIMITE_DE_DIA, ahora))
    pico_mb = max([m["rss_pico_mb_del_arbol_de_procesos"] or 0 for m in medidas.values()] or [0])
    memory_max = f"{max(4, math.ceil(pico_mb * 1.3 / 1024))}G"
    ventana = estimacion["ventana_nocturna"]["segundos"]
    enmienda_1 = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    salida = {
        "corte": "119-C3", "sub_corte": "estimacion (--estimar: UN intento por conjunto medido)",
        "tipo_de_ejecucion": "estimar",
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motor": NOMBRE_MOTOR_NUEVO, "configuracion_del_motor": configuracion_del_motor,
        "protocolos_encadenados": protocolos, "enmienda_2": enmienda_2,
        "protocolo_119_v4_digest_sha256": _digest_fichero_json(RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": _digest_fichero_json(RUTA_DE_LA_ENMIENDA_1),
        "digest_del_entorno": _digest_entorno(), "digest_del_motor": _digest_motor_nuevo(),
        "plan_peor_caso": plan,
        "conjuntos_medidos": [d.nombre for d in medir],
        "cubos_sin_medida": sin_medida,
        "medidas": medidas,
        "estimacion": estimacion,
        "sonda_de_coste_119_c1": enmienda_1["medido_que_lo_motiva"],
        "allstate": {"declarado_por_la_enmienda_1": enmienda_1.get("allstate_declarado"),
                     "medido_aqui": (medidas.get("Allstate_Claims_Severity") or {}).get(
                         "entrenamiento_efectivo")},
        "memoria": {
            "pico_mb_del_arbol_de_procesos": pico_mb,
            "que_mide": ("la SUMA del RSS del padre y de todos sus descendientes, muestreada cada "
                         "0,5 s mientras se carga, se prepara y corre cada intento medido: lo que "
                         "ve un MemoryMax, con las páginas compartidas contadas dos veces y sin la "
                         "caché de ficheros"),
            "por_que_ya_no_se_suma_maxrss": (
                "ru_maxrss de los hijos hereda el RSS del padre en el fork previo al exec "
                "(medido: padre 2.816,8 MB = hijos 2.816,8 MB); sumarlo al del padre contaba el "
                "padre dos veces (5.633 MB declarados; el pico del cgroup medido por la "
                "auditoría 3 fue 3.951 MB)"),
            "memory_max_sugerido": memory_max},
        "para_encolar": para_encolar(estimacion["total_horas"] * 3600, memoria=memory_max,
                                     commits=_commits_de_ahora(),
                                     fin_de_la_suite=fin_de_la_suite),
    }
    _escribir_atomicamente(ruta_salida, salida)
    print(f"\nestimación escrita en {ruta_salida}")
    print(f"  1 proceso: {estimacion['total_horas']:.2f} h (cota {estimacion['total_horas_cota']:.2f}"
          f" h); ventana nocturna {ventana/3600:.2f} h -> "
          f"{'CABE' if estimacion['ventana_nocturna']['cabe'] else 'NO CABE'} "
          f"({estimacion['ventana_nocturna']['noches_necesarias']} noche/s); de día "
          f"{'cabe' if estimacion['de_dia']['cabe'] else 'no cabe'}")
    encolar = salida["para_encolar"]
    dia, noche = encolar["de_dia"], encolar["de_noche"]
    print(f"  memoria: pico del árbol de procesos {pico_mb:.0f} MB -> MemoryMax {memory_max}")
    if dia["cabe"]:
        print(f"  DE DÍA (recomendada), lanzando la cola antes de las {dia['lanzar_antes_de']}:\n"
              f"    {dia['encolar']}\n    {dia['lanzar_la_cola']}")
    else:
        print("  DE DÍA NO CABE (ni lanzándola a las 08:00)")
    if not noche["cabe"]:
        print(f"  DE NOCHE: {noche['aviso']}")
    for i, orden in enumerate(noche["ordenes_una_por_noche"], start=1):
        print(f"    noche {i}: {orden}")


if __name__ == "__main__":
    main()
