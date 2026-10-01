#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C5a — la red nueva (`matrixai.dense.tabm_cpu`) en CONDICIONES DE STUDIO.

Contrato 119, sección «C5.4 — C5a». La Fase 0 (C3/C4) midió la red a 4 hilos y
120/300/600 s por intento; el Studio le da a cada intento de la red 1 hilo y
22,4 s de pared (plazo 16,8 s). Esta sonda mide, ANTES de integrar nada, si en
ese campo la red (a) COMPLETA sus intentos, (b) APRENDE (llega a una época
completa), (c) COMPITE con lightgbm, sklearn.hgb y la densa de hoy, y (d) CABE
en memoria y en tamaño de checkpoint. No decide nada del producto: alimenta
D4-D7 de Roberto.

QUÉ SE MIDE (13 conjuntos NO sellados, pliegues 0 y 1 de la repetición 0):

* condición S (la del Studio): TabM, la densa de hoy (`matrixai.dense.torch_cpu`),
  lightgbm y sklearn.hgb, cada uno con el presupuesto que le da el Studio;
* condición S+ (la curva): TabM a 60 y a 120 s, 1 hilo;
* UN reajuste final por motor (pliegue 0) en proceso aparte, con el presupuesto
  del ajuste final del Studio (16 s / plazo 12 s la red).

LAS CONDICIONES DEL STUDIO NO SE ESCRIBEN A MANO. Los segundos por intento
salen de `matrixai_studio.estudio_job` (`_RESERVA_POR_DEFECTO`,
`_PESO_DE_BUSQUEDA`, `_MOTORES_PERMITIDOS`, `_reparto_del_presupuesto`), y los
pliegues × repeticiones, los hilos y la semilla se LEEN del código fuente del
Studio (`condiciones_del_studio`; si no los encuentra o discrepan, PARA). La
fracción del presupuesto que el motor dedica a entrenar sale del motor. El
resultado registra los valores usados y el sha256 + commit del Studio.

LOS 8 SELLADOS NO SE TOCAN: la guarda `exigir_que_ningun_sellado_entre` corre AL
ARRANCAR, antes de cargar nada, y otra vuelve a mirar antes de cada conjunto.

LA REGLA (completa / aprende / compite / cabe) ESTÁ ESCRITA EN ESTE GUION,
ANTES DE MEDIR: `REGLA_C5A` y `componer_veredicto`. Su digest
(`digest_de_la_regla`) va en el resultado, y la pasada real exige que se le
pase el digest REGISTRADO (`--regla-registrada`): si el guion cambia después de
registrarlo, para. El guion compone el veredicto de cada línea con su número;
no toca k/d_block ni nada de la arquitectura.

QUÉ SE REUTILIZA, Y DE DÓNDE (nada se copia a mano): `pasada_119_c3` (caché por
intento con digest de entorno, `_reusable_c3`, cierre de imports, procedencia
anclada, escritura atómica, la orden de encolado, el medidor de «M3»),
`pasada_114c6_ensamblado`, `pasada_amplia_101_c5` (carga y partición),
`pasada_exploratoria_101_c3` (preparación y sello), `protocolo`
(`aplicar_regla_de_cierre`) y los motores por `motor_para`. Lo único propio:
el muestreo del pico de memoria DEL HIJO (`_PicoDelHijo`: M3 mide el árbol
entero, padre incluido), la clave del caché con la condición, y la fusión.

CÓMO SE LANZA:

    python3 benchmarks/fase0/sonda_studio_119_c5a.py --regla            # el digest a registrar
    python3 benchmarks/fase0/sonda_studio_119_c5a.py --comprobar        # todo menos medir
    python3 benchmarks/fase0/sonda_studio_119_c5a.py --humo --salida <ruta>
    python3 benchmarks/fase0/sonda_studio_119_c5a.py --solo dresses-sales --salida <ruta>
    python3 benchmarks/fase0/sonda_studio_119_c5a.py --estimar --salida <ruta>   # 1 intento real por conjunto
    python3 benchmarks/fase0/sonda_studio_119_c5a.py --regla-registrada <digest> --salida <ruta absoluta>
        # LA PASADA ENTERA -- a la cola (`~/encolar.sh`), nunca a mano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c3p  # noqa: E402 -- TODO lo reutilizable de C3 vive detrás de este alias
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa import FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR as FRACCION_DE_LA_DENSA  # noqa: E402
from matrixai_engines.motores.densa_tabm import (  # noqa: E402
    FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR as FRACCION_DE_TABM)
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.registro_de_motores import motor_para  # noqa: E402
from matrixai_engines.subproceso import (MARGEN_POR_DEFECTO_SEGUNDOS,  # noqa: E402
                                         ejecutar_intento_aislado)
from matrixai_engines.textos import motivo as _motivo_de_los_motores  # noqa: E402

#: El ejecutor REAL, guardado al importar (como en C4): las pruebas sustituyen
#: `ejecutar_intento_aislado` de ESTE módulo por un motor falso.
_EJECUTAR_INTENTO_AISLADO_REAL = ejecutar_intento_aislado

# ---------------------------------------------------------------------------
# 0. LO QUE SE MIDE
# ---------------------------------------------------------------------------

NOMBRE_TABM = "matrixai.dense.tabm_cpu"
NOMBRE_DENSA_DE_HOY = "matrixai.dense.torch_cpu"
NOMBRES_LIGEROS = ("lightgbm", "sklearn.hgb")
#: Los cuatro de la condición S, en el orden en que se miden.
MOTORES_DE_LA_CONDICION_S = (NOMBRE_TABM, NOMBRE_DENSA_DE_HOY, *NOMBRES_LIGEROS)

#: Los 13 de C5.4, por grupo, en el orden en que se miden (los pequeños primero).
CONJUNTOS_DE_C5A = (
    ("dresses-sales", "ejemplo"), ("climate-model-simulation-crashes", "ejemplo"),
    ("us_crime", "ejemplo"), ("pc1", "ejemplo"),
    ("micro-mass", "pequeno_ancho"), ("mfeat-factors", "pequeno_ancho"),
    ("wilt", "mediano"), ("pendigits", "mediano"), ("Internet-Advertisements", "mediano"),
    ("house_16H", "grande"), ("KDDCup09_appetency", "grande"), ("APSFailure", "grande"),
    ("Allstate_Claims_Severity", "grande"),
)
NOMBRES_DE_LOS_EJEMPLOS = tuple(n for n, g in CONJUNTOS_DE_C5A if g == "ejemplo")
CONJUNTOS_DEL_HUMO = ("dresses-sales",)

#: Pliegues 0 y 1 de la repetición 0: coste, no veredicto (C5.4).
REPETICION = 0
PLIEGUES = (0, 1)
PLIEGUES_DEL_HUMO = (0,)

#: La curva S+: TabM a 1 hilo con estos segundos de pared (C5.4).
SEGUNDOS_DE_LA_CURVA = (60.0, 120.0)

#: Cuántos sellados declara el protocolo. La guarda mira que el protocolo los
#: DECLARE: sin esto, un `sellado` que dejara de leerse (todos False) dejaría
#: pasar a cualquiera -- una guardia que no vigila (ver CLAUDE.md).
NUMERO_DE_SELLADOS_DEL_PROTOCOLO = 8

#: Margen del subproceso: el MISMO que usa el Studio (`MARGEN_POR_DEFECTO_
#: SEGUNDOS`, que `estudio_job` importa del mismo sitio). El reajuste del Studio
#: no tiene techo externo; aquí lo lleva igual, como red de seguridad de la sonda.
MARGEN_S = MARGEN_POR_DEFECTO_SEGUNDOS

#: Las claves del caché. La condición ENTRA (TabM aparece en S, S60 y S120).
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0

RUTA_DEL_RESULTADO = _AQUI / "resultado_sonda_studio_119_c5a.json"
SALIDA_POR_OMISION = {
    "pasada": RUTA_DEL_RESULTADO,
    "solo": _AQUI / "sonda_studio_119_c5a_solo_resultado.json",
    "humo": _AQUI / "sonda_studio_119_c5a_humo_resultado.json",
    "estimar": _AQUI / "estimacion_sonda_studio_119_c5a.json",
}
NOMBRE_DEL_TRABAJO_EN_LA_COLA = "01-119-c5a"
GUION = "benchmarks/fase0/sonda_studio_119_c5a.py"
SALIDA_EN_LA_COLA = ("/home/deployer/cola-nocturna/resultados/01-119-c5a/"
                     "resultado_sonda_studio_119_c5a.json")

TIPOS_DE_EJECUCION = ("pasada", "solo", "humo", "estimar")


# ---------------------------------------------------------------------------
# 1. LAS CONDICIONES DEL STUDIO: leídas, no escritas
# ---------------------------------------------------------------------------

def _estudio_job():
    """`matrixai_studio.estudio_job`, importado AQUÍ y no arriba: el hijo de
    `multiprocessing.spawn` re-importa este guion, y no tiene que cargar el
    Studio entero (falsearía su tiempo y su memoria). Si no se importa, PARA:
    no hay valores de repuesto escritos a mano."""
    try:
        import matrixai_studio.estudio_job as estudio_job  # noqa: PLC0415
        return estudio_job
    except ImportError:
        candidato = c3._RAIZ_DEL_CORE.parent / "matrixaistudio" / "studio-backend"
        if candidato.is_dir() and str(candidato) not in sys.path:
            sys.path.insert(0, str(candidato))
            try:
                import matrixai_studio.estudio_job as estudio_job  # noqa: PLC0415
                return estudio_job
            except ImportError:
                pass
        raise SystemExit("no se puede importar matrixai_studio.estudio_job: las condiciones del "
                         "Studio (reserva, pesos, reparto) salen de ahí y NO se escriben a mano "
                         "en esta sonda. Poner studio-backend en el PYTHONPATH") from None


def _unico(valores: list[str], que: str) -> str:
    """El valor que el código del Studio da a `que`: tiene que aparecer y que
    TODOS los sitios coincidan. Si no, PARA (leer mal es medir otro campo)."""
    if not valores:
        raise SystemExit(f"no encuentro «{que}» en el código del Studio (estudio_job.py): la "
                         f"sonda lo LEE de ahí y no lo inventa. ¿Cambió la forma de escribirlo?")
    if len(set(valores)) != 1:
        raise SystemExit(f"el código del Studio da valores DISTINTOS a «{que}»: {sorted(set(valores))}")
    return valores[0]


def lo_que_el_studio_da_por_omision(texto_del_studio: str) -> dict:
    """Pliegues, repeticiones, hilos, semilla y reajuste en proceso aparte,
    LEÍDOS del código del Studio (no son constantes importables: son literales
    del endpoint y del descriptor de intento)."""
    folds = int(_unico(re.findall(r'int\(payload\.get\("folds",\s*(\d+)\)\)', texto_del_studio),
                       'payload.get("folds", N)'))
    repeats = int(_unico(re.findall(r'int\(payload\.get\("repeats",\s*(\d+)\)\)',
                                    texto_del_studio), 'payload.get("repeats", N)'))
    pares = re.findall(r"Presupuesto\(wall_seconds=[^\n]*,\s*hilos=(\d+),\s*seed=(\w+)\)",
                       texto_del_studio)
    hilos = int(_unico([h for h, _ in pares], "Presupuesto(..., hilos=N, ...)"))
    semillas = sorted({s for _, s in pares})
    if not set(semillas) <= {"0", "repeticion"}:
        raise SystemExit(f"el Studio siembra sus intentos con {semillas}: esta sonda usa "
                         f"seed=repeticion en la búsqueda y 0 en el reajuste. No se reproduce el campo")
    aparte = re.findall(r"^_REAJUSTE_EN_UN_PROCESO_APARTE\s*=\s*(True|False)\s*$",
                        texto_del_studio, flags=re.M)
    return {"folds": folds, "repeats": repeats, "hilos": hilos,
            "semillas_de_los_intentos_en_el_codigo": semillas,
            "reajuste_en_un_proceso_aparte": _unico(aparte, "_REAJUSTE_EN_UN_PROCESO_APARTE") == "True"}


def condiciones_del_studio(estudio_job=None) -> dict:
    """LAS CONDICIONES, con su fuente. Los segundos por intento salen de
    `_reparto_del_presupuesto` sobre la reserva y los pesos del Studio, con los
    motores que el Studio permite hoy. TabM NO está en el Studio: ocupa el
    sitio de la densa de hoy (D5 recomendada, «sustituir»: mismo peso, mismos
    segundos). Si el día de mañana el Studio ya trae TabM, PARA: lo que se mide
    aquí es el campo de ANTES de integrarlo."""
    ej = estudio_job if estudio_job is not None else _estudio_job()
    fuente = Path(ej.__file__)
    texto = fuente.read_text(encoding="utf-8")
    leido = lo_que_el_studio_da_por_omision(texto)
    if NOMBRE_TABM in ej._PESO_DE_BUSQUEDA or NOMBRE_TABM in ej._MOTORES_PERMITIDOS:
        raise SystemExit(f"el Studio ya trae {NOMBRE_TABM}: esta sonda mide el campo de ANTES "
                         f"de integrarlo (la red ocupa el sitio de la densa). Revisar el corte")
    if NOMBRE_DENSA_DE_HOY not in ej._MOTORES_PERMITIDOS:
        raise SystemExit(f"la densa de hoy ({NOMBRE_DENSA_DE_HOY}) ya no compite en el Studio: "
                         f"«TabM sustituye a la densa» no tiene sitio donde sustituir")
    reserva = ej._RESERVA_POR_DEFECTO
    n_intentos = leido["folds"] * leido["repeats"]
    busqueda = ej._reparto_del_presupuesto(reserva.busqueda, n_intentos,
                                           motores_permitidos=ej._MOTORES_PERMITIDOS)
    ajuste_final = ej._reparto_del_presupuesto(reserva.ajuste_final, 1,
                                               motores_permitidos=ej._MOTORES_PERMITIDOS)
    sustituye = NOMBRE_DENSA_DE_HOY
    segundos = {m: busqueda[m] for m in (NOMBRE_DENSA_DE_HOY, *NOMBRES_LIGEROS)}
    segundos[NOMBRE_TABM] = busqueda[sustituye]
    segundos_final = {m: ajuste_final[m] for m in (NOMBRE_DENSA_DE_HOY, *NOMBRES_LIGEROS)}
    segundos_final[NOMBRE_TABM] = ajuste_final[sustituye]
    fracciones = {NOMBRE_TABM: FRACCION_DE_TABM, NOMBRE_DENSA_DE_HOY: FRACCION_DE_LA_DENSA}
    return {
        "fuente": {"fichero": str(fuente), "sha256": hashlib.sha256(texto.encode()).hexdigest(),
                   "modulo": "matrixai_studio.estudio_job"},
        "reserva_de_tiempo": {"busqueda": reserva.busqueda, "ajuste_final": reserva.ajuste_final,
                              "calibracion": reserva.calibracion, "evaluacion": reserva.evaluacion},
        "pesos_de_busqueda": {m: ej._PESO_DE_BUSQUEDA[m] for m in ej._MOTORES_PERMITIDOS},
        "motores_permitidos_en_el_studio": list(ej._MOTORES_PERMITIDOS),
        "tabm_ocupa_el_sitio_de": sustituye,
        "leido_del_codigo_del_studio": leido,
        "n_intentos_por_motor": n_intentos,
        "hilos": leido["hilos"], "semilla_de_los_intentos": REPETICION,
        "margen_del_subproceso_s": MARGEN_S,
        "segundos_por_intento": segundos,
        "segundos_del_ajuste_final": segundos_final,
        "fraccion_del_presupuesto_para_entrenar": fracciones,
        "plazo_de_entrenamiento_s": {m: segundos[m] * f for m, f in fracciones.items()},
        "plazo_del_ajuste_final_s": {m: segundos_final[m] * f for m, f in fracciones.items()},
        "curva_s_mas_segundos": list(SEGUNDOS_DE_LA_CURVA),
        "lo_que_no_es_del_studio": (
            "los 60 y 120 s de la curva S+ (los fija C5.4, no el Studio); el margen de 30 s del "
            "reajuste (el Studio no le pone techo externo: aquí es red de seguridad); la "
            "preparación, que es la de la Fase 0 (c3.preparar_para_motor), no la del Studio "
            "(`ajustar_preparacion(con_fechas=True)`) -- C5a-bis lo confirma por el camino real"),
    }


# ---------------------------------------------------------------------------
# 2. LOS CONJUNTOS Y LA GUARDA DE LOS SELLADOS
# ---------------------------------------------------------------------------

def exigir_que_ningun_sellado_entre(nombres, todos) -> None:
    """LA GUARDA: ningún conjunto SELLADO entra en esta sonda, nunca. Corre AL
    ARRANCAR, antes de cargar nada. Para también si el protocolo no declara los
    8 sellados que tiene que declarar (una guarda que no vigila no vale)."""
    por_nombre = {d.nombre: d for d in todos}
    sellados = sorted(n for n, d in por_nombre.items() if d.sellado)
    if len(sellados) != NUMERO_DE_SELLADOS_DEL_PROTOCOLO:
        raise SystemExit(
            f"AL ARRANCAR: el protocolo declara {len(sellados)} conjuntos sellados y este guion "
            f"espera {NUMERO_DE_SELLADOS_DEL_PROTOCOLO}: sin saber cuáles son, la guarda de los "
            f"sellados no vigila nada. No se mide")
    desconocidos = [n for n in nombres if n not in por_nombre]
    if desconocidos:
        raise SystemExit(f"AL ARRANCAR: conjuntos que no están en el protocolo: {desconocidos}")
    tocan = [n for n in nombres if por_nombre[n].sellado]
    if tocan:
        raise SystemExit(
            f"AL ARRANCAR: se piden conjuntos SELLADOS {tocan}. 119-C5a mide SOLO los 13 no "
            f"sellados de C5.4: los 8 sellados se quemaron en C4 y no se tocan, ni «como prueba»")


def datasets_de_c5a(todos, *, solo: str | None = None) -> tuple[list, list | None]:
    """Los conjuntos de esta ejecución, tras la guarda. `--solo` solo admite
    conjuntos de los 13."""
    de_c5a = [n for n, _ in CONJUNTOS_DE_C5A]
    pedidos = [n.strip() for n in solo.split(",") if n.strip()] if solo else None
    exigir_que_ningun_sellado_entre(pedidos if pedidos is not None else de_c5a, todos)
    if pedidos is not None:
        fuera = [n for n in pedidos if n not in de_c5a]
        if fuera:
            raise SystemExit(f"--solo nombra conjuntos que no son de los 13 de C5.4: {fuera}")
        de_c5a = pedidos
    por_nombre = {d.nombre: d for d in todos}
    return [por_nombre[n] for n in de_c5a], pedidos


def grupo_de(nombre: str) -> str | None:
    return dict(CONJUNTOS_DE_C5A).get(nombre)


# ---------------------------------------------------------------------------
# 3. LA REGLA -- escrita ANTES de medir, con su digest
# ---------------------------------------------------------------------------

#: Los umbrales de C5.4, tal cual. Los de «compite» (distancia en puntos y
#: fracción) NO se repiten aquí: son los de la `regla_de_cierre` del protocolo
#: (2,0 puntos y 0,8), que llegan a `componer_veredicto` por parámetro.
REGLA_C5A = {
    "completa": {"fraccion_minima": 0.90,
                 "que": ("al menos el 90 % de los intentos de TabM en la condición S terminan con "
                         "estado que cuenta como medida (completed o completed_budget_limited) y "
                         "con la métrica de cierre; un intento agotado por el techo, fallido, "
                         "muerto sin resultado (¿OOM?) o ausente NO cuenta")},
    "aprende": {"fraccion_minima": 0.80,
                "que": ("al menos el 80 % de los conjuntos tienen la época mejor: TODOS sus "
                        "intentos de TabM en S medidos y NINGUNO con "
                        "plazo_antes_de_la_primera_epoca_completa = true")},
    "compite": {"max_conjuntos_perdidos_contra_la_densa_de_hoy": 2,
                "puntos_para_perder_contra_la_densa": 0.0,
                "que": ("(1) con la regla de la cartera del protocolo (a <= regla.puntos de la "
                        "métrica de cierre del mejor entre TabM, la densa de hoy, lightgbm y "
                        "sklearn.hgb en S; un intento sin medida pierde el conjunto), TabM "
                        "cumple en al menos regla.fraccion_minima de los conjuntos; Y (2) TabM "
                        "no PIERDE contra la densa de hoy en más de 2 conjuntos. «Pierde» es la "
                        "lectura literal de C5.4 («la vieja GANA»): la media de la densa supera "
                        "a la de TabM en más de 0 puntos (con el ancho de empate de la cartera, "
                        "2 puntos, la condición (2) quedaría implícita en la (1): quien está a "
                        "más de 2 de la densa ya no cumple la cartera), o la densa tiene todos "
                        "sus intentos medidos y TabM no. Si la densa tiene un intento sin "
                        "medida, el conjunto no es comparable y no cuenta como pérdida")},
    "cabe": {"pico_hijo_mb_ejemplos": 2.0 * 1024, "pico_hijo_mb_resto": 3.5 * 1024,
             "checkpoint_bytes_ejemplos": 20 * 1024 * 1024,
             "que": ("pico de memoria del HIJO (VmHWM) de TabM en S y en el reajuste <= 2,0 GB "
                     "(2.048 MB) en los 4 ejemplos medidos del Studio y <= 3,5 GB (3.584 MB) en "
                     "el resto, y checkpoint (JSON del predictor, con los pesos) <= 20 MB en los "
                     "ejemplos. Un pico sin medir NO cabe (un valor ausente no es un cero)")},
}
CONSECUENCIA = ("Lo que falle se declara con su número, no se retoca (ni k, ni d_block, ni nada "
                "de la arquitectura): el resultado alimenta D4-D7. Si «completa» o «compite» "
                "fallan, C5 se detiene en C5a y se declara (la red queda como mejora medida de "
                "la Fase 0, fuera del Studio)")


def _medido(r: dict | None, metrica: str | None) -> bool:
    return bool(r and metrica and r.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
                and r.get(metrica) is not None)


def _de(registros: list[dict], motor: str, condicion: str) -> dict:
    return {(r["dataset"], r["pliegue"]): r for r in registros
            if r["motor"] == motor and r["condicion"] == condicion
            and r.get("repeticion") == REPETICION}


def componer_veredicto(registros: list[dict], *, nombres: list[str],
                       metrica_por_dataset: dict[str, str], regla_de_cierre,
                       pliegues: tuple = PLIEGUES) -> dict:
    """LA REGLA APLICADA: cada línea con su número y su cumple/no cumple.
    PURA (se prueba con registros fabricados). Los intentos AUSENTES cuentan
    como no medidos, igual que los fallidos."""
    esperados = [(d, p) for d in nombres for p in pliegues]
    tabm = _de(registros, NOMBRE_TABM, "S")

    # --- completa ---------------------------------------------------------
    ok = [k for k in esperados if _medido(tabm.get(k), metrica_por_dataset.get(k[0]))]
    fallan = [{"dataset": d, "pliegue": p,
               "estado": (tabm.get((d, p)) or {}).get("estado", "ausente"),
               "clasificacion": (tabm.get((d, p)) or {}).get("clasificacion_del_fallo")}
              for d, p in esperados if (d, p) not in ok]
    fraccion = len(ok) / len(esperados) if esperados else 0.0
    completa = {"numero": f"{len(ok)}/{len(esperados)}", "fraccion": fraccion,
                "umbral": REGLA_C5A["completa"]["fraccion_minima"],
                "cumple": bool(esperados) and fraccion >= REGLA_C5A["completa"]["fraccion_minima"],
                "intentos_que_no_completan": fallan}

    # --- aprende ----------------------------------------------------------
    por_conjunto_aprende = {}
    for d in nombres:
        rs = [tabm.get((d, p)) for p in pliegues]
        todos_medidos = all(_medido(r, metrica_por_dataset.get(d)) for r in rs)
        sin_epoca = [p for p, r in zip(pliegues, rs)
                     if r and r.get("plazo_antes_de_la_primera_epoca_completa")]
        por_conjunto_aprende[d] = {
            "aprende": todos_medidos and not sin_epoca, "intentos_medidos": todos_medidos,
            "pliegues_sin_ninguna_epoca_completa": sin_epoca,
            "epocas_completas": [(r or {}).get("epocas_completas") for r in rs],
            "mejor_epoca": [(r or {}).get("mejor_epoca") for r in rs]}
    n_aprenden = sum(1 for v in por_conjunto_aprende.values() if v["aprende"])
    fr_aprende = n_aprenden / len(nombres) if nombres else 0.0
    aprende = {"numero": f"{n_aprenden}/{len(nombres)}", "fraccion": fr_aprende,
               "umbral": REGLA_C5A["aprende"]["fraccion_minima"],
               "cumple": bool(nombres) and fr_aprende >= REGLA_C5A["aprende"]["fraccion_minima"],
               "por_conjunto": por_conjunto_aprende}

    # --- compite ----------------------------------------------------------
    campo = []
    for motor in MOTORES_DE_LA_CONDICION_S:
        propios = _de(registros, motor, "S")
        for d, p in esperados:
            r = propios.get((d, p))
            campo.append(r if _medido(r, metrica_por_dataset.get(d)) else
                         {"dataset": d, "motor": motor, "repeticion": REPETICION, "pliegue": p,
                          "estado": "sin_medida"})
    cartera = protocolo_mod.aplicar_regla_de_cierre(
        campo, regla_de_cierre, motor=NOMBRE_TABM, metrica_por_dataset=metrica_por_dataset,
        datasets_exigidos=nombres)
    densa = _de(registros, NOMBRE_DENSA_DE_HOY, "S")
    contra_la_densa = {}
    for d in nombres:
        m = metrica_por_dataset.get(d)
        densa_ok = all(_medido(densa.get((d, p)), m) for p in pliegues)
        tabm_ok = all(_medido(tabm.get((d, p)), m) for p in pliegues)
        if not densa_ok:
            contra_la_densa[d] = {"pierde": False, "motivo": "no comparable: la densa de hoy tiene "
                                  "algún intento sin medida", "diferencia_en_puntos": None}
        elif not tabm_ok:
            contra_la_densa[d] = {"pierde": True, "motivo": "TabM tiene algún intento sin medida "
                                  "y la densa de hoy los tiene todos", "diferencia_en_puntos": None}
        else:
            media = lambda mapa: sum(mapa[(d, p)][m] for p in pliegues) / len(pliegues)  # noqa: E731
            dif = (media(densa) - media(tabm)) * 100.0
            contra_la_densa[d] = {"pierde": dif > REGLA_C5A["compite"][
                "puntos_para_perder_contra_la_densa"],
                                  "motivo": "media de la densa - media de TabM, en puntos",
                                  "diferencia_en_puntos": dif}
    n_pierde = sum(1 for v in contra_la_densa.values() if v["pierde"])
    tope = REGLA_C5A["compite"]["max_conjuntos_perdidos_contra_la_densa_de_hoy"]
    compite = {
        "cumplidos_de_la_cartera": f"{cartera['cumplidos']}/{cartera['datasets']}",
        "fraccion": cartera["fraccion"], "umbral": regla_de_cierre.fraccion_minima,
        "puntos": regla_de_cierre.puntos, "cumple_la_cartera": cartera["cumple_la_regla"],
        "conjuntos_perdidos_contra_la_densa_de_hoy": n_pierde, "tope_de_perdidos": tope,
        "cumple_contra_la_densa": n_pierde <= tope,
        "cumple": bool(cartera["cumple_la_regla"]) and n_pierde <= tope,
        "detalle_de_la_cartera": [{k: d.get(k) for k in ("dataset", "cumple", "perdido_por_fallo",
                                                         "mejor", "distancia_en_puntos")}
                                  for d in cartera["detalle"]],
        "contra_la_densa_de_hoy": contra_la_densa}

    # --- cabe -------------------------------------------------------------
    limites = REGLA_C5A["cabe"]
    sin_pico, pasan_de_pico, pasan_de_checkpoint = [], [], []
    medidos_de_cabe = [r for r in registros if r["motor"] == NOMBRE_TABM
                       and r["condicion"] in ("S", "reajuste") and r["dataset"] in nombres]
    maximos: dict[str, dict] = {}
    for r in medidos_de_cabe:
        d = r["dataset"]
        ejemplo = d in NOMBRES_DE_LOS_EJEMPLOS
        tope_pico = limites["pico_hijo_mb_ejemplos" if ejemplo else "pico_hijo_mb_resto"]
        pico = r.get("pico_hijo_mb")
        m = maximos.setdefault(d, {"pico_hijo_mb_max": None, "checkpoint_bytes_max": None,
                                   "tope_pico_mb": tope_pico, "es_ejemplo": ejemplo})
        if pico is None:
            sin_pico.append({"dataset": d, "condicion": r["condicion"], "pliegue": r["pliegue"]})
        else:
            m["pico_hijo_mb_max"] = max(pico, m["pico_hijo_mb_max"] or 0.0)
            if pico > tope_pico:
                pasan_de_pico.append({"dataset": d, "condicion": r["condicion"],
                                      "pliegue": r["pliegue"], "pico_hijo_mb": pico,
                                      "tope_mb": tope_pico})
        ck = r.get("checkpoint_json_bytes")
        if ck is not None:
            m["checkpoint_bytes_max"] = max(ck, m["checkpoint_bytes_max"] or 0)
            if ejemplo and ck > limites["checkpoint_bytes_ejemplos"]:
                pasan_de_checkpoint.append({"dataset": d, "condicion": r["condicion"],
                                            "pliegue": r["pliegue"], "checkpoint_json_bytes": ck})
    cabe = {
        "cumple": (bool(medidos_de_cabe) and not sin_pico and not pasan_de_pico
                   and not pasan_de_checkpoint),
        "intentos_mirados": len(medidos_de_cabe), "sin_pico_medido": sin_pico,
        "pasan_del_tope_de_memoria": pasan_de_pico,
        "pasan_del_tope_de_checkpoint": pasan_de_checkpoint,
        "maximos_por_conjunto": maximos,
        "topes": {k: v for k, v in limites.items() if k != "que"}}

    return {
        "completa": completa, "aprende": aprende, "compite": compite, "cabe": cabe,
        "cumple_las_cuatro": all(x["cumple"] for x in (completa, aprende, compite, cabe)),
        "consecuencia": CONSECUENCIA,
        "regla": {k: v.get("que") for k, v in REGLA_C5A.items()},
        "n_conjuntos": len(nombres), "pliegues": list(pliegues),
    }


def digest_de_la_regla() -> str:
    """El digest de la regla: sus umbrales, su texto Y el código que la
    compone. Se REGISTRA antes de lanzar (contrato 119, C5.4) y la pasada real
    exige que se le pase (`--regla-registrada`): si cambia después, para."""
    import inspect  # noqa: PLC0415

    cuerpo = {"regla": REGLA_C5A, "consecuencia": CONSECUENCIA,
              "pliegues": list(PLIEGUES), "motores_de_s": list(MOTORES_DE_LA_CONDICION_S),
              "conjuntos": [list(c) for c in CONJUNTOS_DE_C5A],
              "curva": list(SEGUNDOS_DE_LA_CURVA),
              "codigo": [inspect.getsource(f) for f in (componer_veredicto, _medido, _de)]}
    return hashlib.sha256(json.dumps(cuerpo, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def exigir_la_regla_registrada(registrada: str | None, tipo: str) -> dict:
    """La pasada REAL no arranca sin el digest registrado y que cuadre.
    `--solo/--humo/--estimar` sí, diciéndolo."""
    actual = digest_de_la_regla()
    if tipo == "pasada" and registrada != actual:
        raise SystemExit(
            f"la pasada real exige --regla-registrada <digest>: el de la regla ESCRITA en este "
            f"guion es {actual}. Se registra en el contrato 119 (C5.4) ANTES de lanzar; "
            f"{'no se ha pasado ninguno' if not registrada else 'el pasado (' + registrada + ') no coincide'}")
    return {"digest": actual, "registrada": registrada,
            "coincide": registrada == actual,
            "nota": ("sin --regla-registrada: esta ejecución no es la pasada real" if not registrada
                     else None)}


# ---------------------------------------------------------------------------
# 4. LA MEMORIA DEL HIJO
# ---------------------------------------------------------------------------

def _hijos_por_padre() -> dict[int, list[int]]:
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
    return hijos


def _descendientes(raiz: int) -> list[int]:
    hijos, pendientes, vistos = _hijos_por_padre(), list(_hijos_por_padre().get(raiz, [])), []
    while pendientes:
        pid = pendientes.pop()
        if pid in vistos:
            continue
        vistos.append(pid)
        pendientes.extend(hijos.get(pid, []))
    return vistos


def _vm_kb(pid: int) -> dict[str, int]:
    """VmRSS y VmHWM (el máximo que el kernel apunta) de un proceso, en kB."""
    salida: dict[str, int] = {}
    with open(f"/proc/{pid}/status", "rb") as f:
        for linea in f:
            if linea.startswith((b"VmRSS:", b"VmHWM:")):
                nombre, valor = linea.decode().split(":", 1)
                salida[nombre] = int(valor.split()[0])
    return salida


class _PicoDelHijo:
    """EL PICO DE MEMORIA DEL HIJO, no el del padre: muestrea los DESCENDIENTES
    de este proceso cada `cada_s` y se queda con el mayor `VmHWM` (el máximo
    que el kernel lleva anotado, así que lo que pase entre muestras queda
    recogido hasta la última lectura) y con la mayor SUMA de RSS. `ru_maxrss`
    de los hijos hereda el RSS del padre en el fork (medido en C3, M3) y NO
    sirve. `None` si no se vio ningún descendiente (un valor ausente no es un
    cero). El hijo de `multiprocessing.spawn` re-importa este guion, así que su
    pico incluye ese arranque: es el del intento tal como corre aquí."""

    def __init__(self, cada_s: float = 0.2):
        self._pid, self._cada_s = os.getpid(), cada_s
        self.hwm_kb: dict[int, int] = {}
        self.pico_suma_rss_kb = 0
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)

    def _una_vez(self) -> None:
        suma = 0
        for pid in _descendientes(self._pid):
            try:
                vm = _vm_kb(pid)
            except OSError:
                continue
            suma += vm.get("VmRSS", 0)
            if "VmHWM" in vm:
                self.hwm_kb[pid] = max(self.hwm_kb.get(pid, 0), vm["VmHWM"])
        self.pico_suma_rss_kb = max(self.pico_suma_rss_kb, suma)

    def _muestrear(self) -> None:
        while True:
            try:
                self._una_vez()
            except Exception:  # noqa: BLE001 -- medir no puede tumbar la sonda
                pass
            if self._parar.wait(self._cada_s):
                return

    def __enter__(self):
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        self._hilo.join(timeout=10)

    @property
    def pico_mb(self) -> float | None:
        """El VmHWM mayor entre los descendientes vistos (el hijo del intento)."""
        return round(max(self.hwm_kb.values()) / 1024, 1) if self.hwm_kb else None

    @property
    def pico_suma_rss_mb(self) -> float | None:
        return round(self.pico_suma_rss_kb / 1024, 1) if self.hwm_kb else None


# ---------------------------------------------------------------------------
# 5. EL REGISTRO DE UN INTENTO
# ---------------------------------------------------------------------------

_PREFIJOS_DE_LOS_FALLOS = (
    ("agotado_por_el_techo", "el intento superó"),
    ("murio_sin_resultado", "el proceso aislado terminó"),
    ("excepcion_no_declarada", "el subproceso aislado capturó una excepción no declarada"),
)


def clasificar_el_fallo(estado: str, motivo_es: str | None) -> str | None:
    """Qué le pasó a un intento que no cuenta como medida. Los prefijos son los
    de `matrixai_engines.textos` (una prueba los compara con los de verdad)."""
    if estado in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
        return None
    if estado == "cancelled":
        return "cancelado"
    for nombre, prefijo in _PREFIJOS_DE_LOS_FALLOS:
        if (motivo_es or "").startswith(prefijo):
            return nombre
    return "fallo_declarado_por_el_motor"


def n_parametros_de(config: dict | None) -> int | None:
    """Parámetros de la red, contados de las formas de sus pesos (TabM). `None`
    si el predictor no los trae así (la densa vieja guarda un programa, no
    tensores): no se inventa."""
    pesos = (config or {}).get("pesos")
    if not isinstance(pesos, dict) or not pesos:
        return None
    total = 0
    for t in pesos.values():
        forma = t.get("forma") if isinstance(t, dict) else None
        if not isinstance(forma, list):
            return None
        total += math.prod(forma)
    return total


def tamano_del_checkpoint(config: dict | None) -> int | None:
    """Bytes del JSON del predictor (lleva los pesos): el grueso del checkpoint;
    el checkpoint entero añade la envoltura del spec, unos pocos kB."""
    if not config:
        return None
    return len(json.dumps(config, ensure_ascii=False, default=str).encode("utf-8"))


def _entrenamiento(config: dict | None) -> dict:
    e = (config or {}).get("entrenamiento_efectivo") or {}
    return {"epocas_ejecutadas": e.get("epocas_ejecutadas"),
            "epocas_completas": e.get("epocas_completas"), "mejor_epoca": e.get("mejor_epoca"),
            "parado_por_plazo": e.get("parado_por_plazo"),
            "parado_por_paciencia": e.get("parado_por_paciencia"),
            "plazo_antes_de_la_primera_epoca_completa":
                e.get("plazo_antes_de_la_primera_epoca_completa"),
            "plazo_de_entrenamiento_s": e.get("plazo_de_entrenamiento_segundos")}


def digest_del_motor(nombre: str) -> str:
    """El digest del código PROPIO de cada motor (como `_FICHERO_POR_MOTOR`)."""
    if nombre == NOMBRE_TABM:
        return c3p._digest_de(c3p._FICHEROS_DEL_MOTOR_NUEVO)
    return c3p._digest_de((c3._FICHERO_POR_MOTOR[nombre],))


def reusable(previo: dict | None, entorno_digest: str, motor_nombre: str, wall_s: float, *,
             datos_sha256: str | None, hilos: int, margen_s: float) -> bool:
    """`_reusable_c3` (mismos digests, mismo presupuesto, MISMO ARFF y un estado
    que cuente como medida: un `failed` con los digests buenos no se reusa,
    trampa 1 de «ANTES DE RELANZAR UNA PASADA») MÁS los hilos y el margen: son
    parte de la condición y no entran en ningún digest."""
    if not c3p._reusable_c3(previo, entorno_digest, digest_del_motor(motor_nombre), wall_s,
                            datos_sha256=datos_sha256):
        return False
    return previo.get("hilos") == hilos and previo.get("margen_s") == margen_s


def registro_de_un_intento(*, ds, condicion: str, motor_nombre: str, pliegue: int, wall_s: float,
                           semilla: int, metric_id: str, intento, transcurrido_s: float,
                           preparacion_s: float, pico, hilos: int, entorno_digest: str,
                           datos_sha256: str | None, procedencia_id: str | None,
                           split_plan_digest: str) -> dict:
    """El registro de UN intento: lo que lista C5.4 y lo que el caché necesita."""
    config = getattr(intento, "config_efectiva", None) or {}
    recursos = getattr(intento, "recursos", None) or {}
    motivo_es = (intento.motivo_del_estado or {}).get("es") if intento.motivo_del_estado else None
    metricas = c5.metricas_del_informe(intento.informe)
    registro = {
        "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea,
        "grupo": grupo_de(ds.nombre), "sellado": ds.sellado,
        "condicion": condicion, "motor": motor_nombre, "repeticion": REPETICION,
        "pliegue": pliegue, "semilla": semilla, "estado": intento.estado,
        "clasificacion_del_fallo": clasificar_el_fallo(intento.estado, motivo_es),
        "motivo": motivo_es, "traza": getattr(intento, "traza", None),
        "presupuesto_wall_s": wall_s, "hilos": hilos, "margen_s": MARGEN_S,
        "techo_externo_s": wall_s + MARGEN_S, "metrica_de_cierre": metric_id,
        "wall_s": round(transcurrido_s, 3), "preparacion_s": round(preparacion_s, 3),
        "tiempo_de_ajuste_s": recursos.get("wall_seconds"), "cpu_segundos": recursos.get("cpu_seconds"),
        **_entrenamiento(config),
        "n_parametros": n_parametros_de(config), "checkpoint_json_bytes": tamano_del_checkpoint(config),
        "pico_hijo_mb": pico.pico_mb, "pico_suma_rss_del_hijo_mb": pico.pico_suma_rss_mb,
        "arquitectura": config.get("arquitectura"), "hiperparametros": config.get("hiperparametros"),
        "engine_version": getattr(intento, "engine_version", None),
        "pipeline_digest": getattr(intento, "pipeline_digest", None),
        "split_plan_digest": split_plan_digest, "metricas": metricas,
        "entorno_digest": entorno_digest, "motor_digest": digest_del_motor(motor_nombre),
        "datos_sha256": datos_sha256, "procedencia_id": procedencia_id, "reusado": False,
    }
    c5.aplanar_metricas_en_el_registro(registro, metricas)
    return registro


# ---------------------------------------------------------------------------
# 6. EL PLAN DE UN PLIEGUE
# ---------------------------------------------------------------------------

def plan_de_un_pliegue(estudio: dict, pliegue: int, *, con_curva: bool = True,
                       con_reajuste: bool = True) -> list[dict]:
    """Los intentos de UN pliegue, en orden: S (los cuatro motores), el reajuste
    de cada motor (solo el pliegue 0) y la curva S+ (TabM a 60 y 120 s)."""
    plan = [{"condicion": "S", "motor": m, "wall_s": estudio["segundos_por_intento"][m]}
            for m in MOTORES_DE_LA_CONDICION_S]
    if con_reajuste and pliegue == PLIEGUES[0]:
        plan += [{"condicion": "reajuste", "motor": m,
                  "wall_s": estudio["segundos_del_ajuste_final"][m]}
                 for m in MOTORES_DE_LA_CONDICION_S]
    if con_curva:
        plan += [{"condicion": f"S+{int(s)}", "motor": NOMBRE_TABM, "wall_s": s}
                 for s in estudio["curva_s_mas_segundos"]]
    return plan


def _clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["condicion"], r["repeticion"], r["pliegue"])


def fusionar(previos: list[dict], de_esta_ejecucion: list[dict]) -> tuple[list[dict], list[dict]]:
    """Lo previo que esta ejecución no ha vuelto a tocar, más lo de esta; y los
    previos SIN MEDIDA que esta ejecución ha vuelto a medir (los reintentos: no
    desaparecen, quedan en `intentos_reintentados`)."""
    nuevas = {_clave(r) for r in de_esta_ejecucion}
    conservados = [r for r in previos if _clave(r) not in nuevas]
    reintentados = [r for r in previos if _clave(r) in nuevas
                    and r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]
    return conservados + list(de_esta_ejecucion), reintentados


def cargar_salida_previa(ruta: Path, tipo: str) -> tuple[dict, dict]:
    """El caché (con la condición en la clave) y el fichero entero; no mezcla tipos."""
    if not ruta.exists():
        return {}, {}
    payload = c3p._leer_json(ruta)
    del_fichero = c3p.tipo_del_fichero(payload)
    if del_fichero != tipo:
        raise SystemExit(f"{ruta} es un resultado de tipo «{del_fichero}» y esta ejecución es "
                         f"«{tipo}»: no se mezclan. Otra --salida, o mover ese fichero a mano")
    return {_clave(r): r for r in payload.get("resultados", [])}, payload


# ---------------------------------------------------------------------------
# 7. EL ENTORNO, EL CACHÉ Y LA PROCEDENCIA
# ---------------------------------------------------------------------------

def _raiz_del_studio(estudio_job=None) -> Path:
    ej = estudio_job if estudio_job is not None else _estudio_job()
    return Path(ej.__file__).resolve().parents[2]


def _etiqueta(ruta: Path) -> str:
    """Como `c3p._etiqueta` (relativa a su repo: el digest no depende de dónde
    está el árbol) y además `studio:` para los ficheros del Studio."""
    r = Path(ruta).resolve()
    try:
        return f"studio:{r.relative_to(_raiz_del_studio()).as_posix()}"
    except ValueError:
        return c3p._etiqueta(r)


def ficheros_del_entorno() -> tuple[Path, ...]:
    """El cierre estático de imports de ESTE guion dentro de los TRES repos (el
    Studio incluido: `estudio_job` decide las condiciones) más los compartidos
    de C5 y los de los cuatro motores."""
    raices = (*c3p._raices_de_import(), Path(_estudio_job().__file__).resolve().parents[1])
    cierre, _ = c3p._cierre_de_imports(Path(__file__).resolve(), raices)
    extra = {Path(p).resolve() for p in (*c5._FICHEROS_COMPARTIDOS, *c3p._FICHEROS_DEL_MOTOR_NUEVO,
                                         *(c3._FICHERO_POR_MOTOR[m] for m in MOTORES_DE_LA_CONDICION_S
                                           if m != NOMBRE_TABM))}
    return tuple(sorted(set(cierre) | extra))


def componentes_del_digest_del_entorno() -> dict[str, str]:
    componentes = {_etiqueta(f): c3._digest_fichero(f) for f in ficheros_del_entorno()}
    for nombre, ruta in c3p._datos_del_entorno().items():
        componentes[f"dato:{nombre}"] = c3._digest_fichero(ruta)
    for nombre, version in c3p._versiones_en_el_digest().items():
        componentes[f"version:{nombre}"] = version
    return componentes


def lo_que_no_cubre_el_digest() -> dict:
    """Lo que el digest NO cubre, dicho en el resultado (lección de 101-C5)."""
    _, dinamicos = c3p._cierre_de_imports(
        Path(__file__).resolve(),
        (*c3p._raices_de_import(), Path(_estudio_job().__file__).resolve().parents[1]))
    cubiertos = {str(p) for p in ficheros_del_entorno()}
    raices = (c3._RAIZ_DE_ENGINES.resolve(), c3._RAIZ_DEL_CORE.resolve(), _raiz_del_studio())
    prefijos = tuple(f"{r}/" for r in raices)
    fuera = sorted(_etiqueta(Path(f)) for f in c3._ficheros_importados_por_este_proceso()
                   if f.endswith(".py") and f.startswith(prefijos) and f not in cubiertos
                   and Path(f).is_file())
    versiones = c3p.versiones_de_bibliotecas()
    return {
        "como_se_calcula": ("cierre ESTÁTICO de imports de este guion dentro de los tres repos "
                            "(ast; también los import dentro de funciones), más los compartidos "
                            "de C5 y los ficheros de los cuatro motores. Superconjunto de lo que corre"),
        "bibliotecas_de_terceros_que_NO_invalidan": sorted(
            k for k in versiones if k not in c3p.BIBLIOTECAS_EN_EL_DIGEST),
        "bibliotecas_de_terceros": (f"solo por versión: {', '.join(c3p.BIBLIOTECAS_EN_EL_DIGEST)}"),
        "imports_dinamicos_en_el_cierre": list(dinamicos),
        "datos": "los ARFF no entran: su sha256 va en la clave de cada registro (datos_sha256)",
        "las_condiciones_del_studio": ("entran por el DIGEST de estudio_job.py (cierre de imports) "
                                       "y por su sha256 + commit en condiciones_del_studio y en la "
                                       "procedencia; el presupuesto de cada intento está además en "
                                       "la clave del caché"),
        "la_regla": "su digest va en el resultado (`regla`); la compone este guion, que está en el digest",
        "modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest": fuera,
        "que_dice_la_lista_de_arriba": "COTA INFERIOR (el hijo re-importa lo suyo); tiene que estar vacía",
    }


def procedencia_de_la_medicion(*, digests: dict, datasets) -> dict:
    """`c3.procedencia_de_la_medicion` con el Studio como TERCER repositorio
    anclado (su commit y si su árbol estaba sucio). La función de C3 lee el
    diccionario global de repos: se amplía durante la llamada y se restaura."""
    raices = c3._RUTAS_DE_REPOSITORIO
    original = dict(raices)
    raices["matrixaistudio"] = _raiz_del_studio()
    try:
        return c3.procedencia_de_la_medicion(
            digests_de_codigo=digests,
            datos_de_entrada={**{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
                              **{f"dato:{n}": r for n, r in c3p._datos_del_entorno().items()}})
    finally:
        raices.clear()
        raices.update(original)


def exigir_el_anclaje(procedencia: dict, tipo: str) -> None:
    """La pasada REAL se NIEGA con un árbol sucio, un commit sin resolver o un
    ARFF ausente en cualquiera de los TRES repos, ANTES de medir. Las otras
    ejecuciones avisan (no cuentan)."""
    if procedencia.get("anclable"):
        return
    if tipo == "pasada":
        raise SystemExit("NO ANCLABLE: la pasada real se niega a medir sin una procedencia "
                         "anclada (árboles limpios y commits):\n  - "
                         + "\n  - ".join(procedencia.get("avisos") or ["sin motivo declarado"]))
    for aviso in procedencia.get("avisos") or []:
        print(f"AVISO DE PROCEDENCIA ({tipo}, no cuenta): {aviso}", flush=True)


# ---------------------------------------------------------------------------
# 8. LA MEDICIÓN DE UN CONJUNTO
# ---------------------------------------------------------------------------

def _preparador(por_id, objetivo, predictores, ip_entrena, ip_valida, test_ids):
    """Prepara (una vez por pliegue y por tipo de capacidades de motor) las
    tres particiones, y mide cuánto cuesta."""
    cache: dict[tuple, tuple] = {}

    def preparar(motor):
        cap = motor.capabilities()
        clave = (cap.admite_categoricas, cap.admite_faltantes)
        if clave not in cache:
            t0 = time.perf_counter()
            crudas_train = [por_id[i] for i in ip_entrena]
            crudas_val = [por_id[i] for i in ip_valida]
            crudas_test = [por_id[i] for i in test_ids]
            transformadas = c3.preparar_para_motor(
                crudas_train, crudas_train + crudas_val + crudas_test, objetivo, predictores, motor)
            n_tr, n_va = len(crudas_train), len(crudas_val)

            def hacer(xs):
                return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

            cache[clave] = (hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]),
                            hacer(transformadas[n_tr + n_va:]), time.perf_counter() - t0)
        return cache[clave]
    return preparar


def medir_conjunto(ds, protocolo, estudio: dict, *, metric_id: str, pliegues: tuple,
                   con_curva: bool, con_reajuste: bool, cache_previo: dict, forzar: bool,
                   entorno_digest: str, procedencia: dict, motores: dict,
                   al_terminar_un_pliegue=None) -> tuple[list[dict], dict]:
    """TODOS los intentos de UN conjunto. Devuelve los registros y los tiempos
    del padre. Vuelve a mirar la guarda de los sellados ANTES de cargar nada."""
    if ds.sellado:
        raise SystemExit(f"{ds.nombre} es un conjunto SELLADO: esta sonda no lo toca "
                         f"(segunda guarda, antes de cargarlo)")
    t_carga = time.perf_counter()
    por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
    tiempos = {"carga_s": round(time.perf_counter() - t_carga, 3), "preparacion_s": 0.0,
               "n_preparaciones": 0, "n_filas": particion["n_filas_con_objetivo"],
               "n_predictores": particion["n_predictores"]}
    test_ids = propuesta.plan.observaciones_del_rol("test")
    datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
    registros: list[dict] = []
    print(f"\n=== {ds.nombre} ({ds.tarea}, n={len(por_id)}, metrica={metric_id}, "
          f"carga={tiempos['carga_s']:.1f}s) ===", flush=True)
    for pliegue in pliegues:
        pl = propuesta.pliegues.pliegue_de(repeticion=REPETICION, pliegue=pliegue)
        if pl is None:
            print(f"  pliegue {pliegue}: no existe en la partición (sus intentos faltan)", flush=True)
            continue
        preparar = _preparador(por_id, objetivo, predictores, tuple(pl.entrena), tuple(pl.valida),
                               test_ids)
        for paso in plan_de_un_pliegue(estudio, pliegue, con_curva=con_curva,
                                       con_reajuste=con_reajuste):
            motor = motores[paso["motor"]]
            clave = (ds.nombre, motor.nombre, paso["condicion"], REPETICION, pliegue)
            previo = None if forzar else cache_previo.get(clave)
            if reusable(previo, entorno_digest, motor.nombre, paso["wall_s"],
                        datos_sha256=datos_sha256, hilos=estudio["hilos"], margen_s=MARGEN_S):
                registro = dict(previo, reusado=True)
                registro.setdefault("procedencia_id", None)
                registros.append(registro)
                print(f"  [reusado] {ds.nombre} {paso['condicion']} {motor.nombre} "
                      f"pliegue={pliegue}: estado={registro.get('estado')}", flush=True)
                continue
            train, val, test, preparacion_s = preparar(motor)
            tiempos["preparacion_s"] = round(tiempos["preparacion_s"] + preparacion_s, 3)
            tiempos["n_preparaciones"] += 1
            semilla = estudio["semilla_de_los_intentos"] if paso["condicion"] != "reajuste" else 0
            presupuesto = Presupuesto(wall_seconds=paso["wall_s"], hilos=estudio["hilos"],
                                      seed=semilla)
            with _PicoDelHijo() as pico:
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto,
                    candidate=f"{motor.nombre}-c5a-{paso['condicion']}",
                    split_plan_digest=propuesta.plan.digest(), dataset=ds.nombre,
                    pliegue=pliegue, repeticion=REPETICION, margen_segundos=MARGEN_S)
                transcurrido = time.perf_counter() - t0
            registro = registro_de_un_intento(
                ds=ds, condicion=paso["condicion"], motor_nombre=motor.nombre, pliegue=pliegue,
                wall_s=paso["wall_s"], semilla=semilla, metric_id=metric_id, intento=intento,
                transcurrido_s=transcurrido, preparacion_s=preparacion_s, pico=pico,
                hilos=estudio["hilos"], entorno_digest=entorno_digest, datos_sha256=datos_sha256,
                procedencia_id=procedencia["procedencia_id"],
                split_plan_digest=propuesta.plan.digest())
            registros.append(registro)
            print(f"  {ds.nombre} {paso['condicion']} {motor.nombre} pliegue={pliegue}: "
                  f"estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                  f"wall={transcurrido:.1f}s epocas={registro['epocas_ejecutadas']} "
                  f"pico_hijo={registro['pico_hijo_mb']}MB", flush=True)
        if al_terminar_un_pliegue:
            al_terminar_un_pliegue(registros, tiempos)
    return registros, tiempos


# ---------------------------------------------------------------------------
# 9. EL CUADRO POR CONJUNTO (S y S+)
# ---------------------------------------------------------------------------

def cuadro_por_conjunto(registros: list[dict], nombres: list[str],
                        metrica_por_dataset: dict[str, str]) -> dict:
    """Por conjunto y por «condición|motor»: intentos medidos/esperados, media
    de la métrica de cierre, pared media, épocas, parado por plazo, pico y
    checkpoint máximos. Es la tabla que D4-D7 miran."""
    cuadro: dict[str, dict] = {}
    for d in nombres:
        m = metrica_por_dataset.get(d)
        grupos: dict[str, list[dict]] = {}
        for r in registros:
            if r["dataset"] == d:
                grupos.setdefault(f"{r['condicion']}|{r['motor']}", []).append(r)
        fila = {}
        for k, rs in sorted(grupos.items()):
            medidos = [r for r in rs if _medido(r, m)]
            valores = [r[m] for r in medidos]
            picos = [r["pico_hijo_mb"] for r in rs if r.get("pico_hijo_mb") is not None]
            cks = [r["checkpoint_json_bytes"] for r in rs if r.get("checkpoint_json_bytes")]
            fila[k] = {
                "intentos": len(rs), "medidos": len(medidos),
                "estados": sorted({r["estado"] for r in rs}),
                "metrica_media": (sum(valores) / len(valores)) if valores else None,
                "wall_s_medio": sum(r["wall_s"] for r in rs) / len(rs),
                "epocas_ejecutadas": [r.get("epocas_ejecutadas") for r in rs],
                "epocas_completas": [r.get("epocas_completas") for r in rs],
                "mejor_epoca": [r.get("mejor_epoca") for r in rs],
                "parado_por_plazo": [r.get("parado_por_plazo") for r in rs],
                "sin_epoca_completa": [r.get("plazo_antes_de_la_primera_epoca_completa") for r in rs],
                "n_parametros": next((r["n_parametros"] for r in rs if r.get("n_parametros")), None),
                "pico_hijo_mb_max": max(picos) if picos else None,
                "checkpoint_json_bytes_max": max(cks) if cks else None}
        cuadro[d] = fila
    return cuadro


# ---------------------------------------------------------------------------
# 10. LA SALIDA
# ---------------------------------------------------------------------------

def _componer_y_guardar(resultados, previos, veredicto, procedencia, payload_previo, ruta_salida, *,
                        tipo, datasets, subconjunto, estudio, regla, metrica_por_dataset, pliegues,
                        tiempos_del_padre, componentes_del_digest, total_wall_s, reusados,
                        parcial, protocolo, con_curva) -> dict:
    en_el_fichero, reintentados = fusionar(previos, resultados)
    reintentados = list(payload_previo.get("intentos_reintentados") or []) + [
        {"dataset": r["dataset"], "motor": r["motor"], "condicion": r["condicion"],
         "pliegue": r["pliegue"], "estado": r.get("estado"), "motivo": r.get("motivo"),
         "clasificacion_del_fallo": r.get("clasificacion_del_fallo"),
         "procedencia_id": r.get("procedencia_id"), "registro_sustituido": r}
        for r in reintentados]
    procedencias, sin_procedencia = c3._procedencias_citadas(
        en_el_fichero, procedencia, payload_previo.get("procedencias") or {})
    nombres = [d.nombre for d in datasets]
    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "119-C5a", "tipo_de_ejecucion": tipo, "parcial": parcial,
        "es_humo": tipo == "humo", "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "que_es": ("la red nueva en condiciones de Studio (1 hilo, presupuesto del Studio): "
                   "completa / aprende / compite / cabe. NO es el X/40 ni cambia la cartera"),
        "motores_de_la_condicion_s": list(MOTORES_DE_LA_CONDICION_S),
        "condiciones_del_studio": estudio,
        "regla": regla,
        "pliegues": list(pliegues), "con_curva_s_mas": con_curva,
        "protocolo_version": protocolo.version_protocolo, "protocolo_digest_sha256": protocolo.digest(),
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "tiempos_del_padre_por_dataset": dict(tiempos_del_padre),
        "total_wall_s": round(total_wall_s, 1), "n_intentos": len(resultados),
        "n_reusados": reusados, "n_intentos_en_el_fichero": len(en_el_fichero),
        "n_intentos_reintentados": len(reintentados), "intentos_reintentados": reintentados,
        "que_son_los_intentos_reintentados": (
            "intentos que en una ejecución anterior no contaron como medida (fallidos, agotados, "
            "cancelados) y que esta ejecución ha vuelto a medir. NO se reusan como medida (un "
            "`failed` con los digests buenos se reusaría para siempre); quedan aquí, no se borran"),
        "digest_de_la_cache": {"entorno": procedencia["digests_de_codigo"].get("entorno"),
                               "componentes": dict(componentes_del_digest),
                               "clave_de_cada_registro": (
                                   "entorno_digest, motor_digest, presupuesto_wall_s, hilos, margen_s, "
                                   "estado que cuente como medida y datos_sha256; mas dataset, "
                                   "motor, condicion, repeticion y pliegue")},
        "lo_que_no_cubre_el_digest": lo_que_no_cubre_el_digest(),
        "como_se_mide_la_memoria": (
            "pico_hijo_mb = el mayor VmHWM entre los descendientes del padre durante el intento, "
            "muestreado cada 0,2 s (no ru_maxrss, que hereda el RSS del padre); incluye el arranque "
            "del hijo spawn (re-importa este guion). pico_suma_rss_del_hijo_mb = la mayor suma de "
            "RSS de los descendientes"),
        "como_se_mide_el_checkpoint": ("bytes del JSON del predictor (config_efectiva), que lleva "
                                       "los pesos en base64; n_parametros solo si el predictor trae "
                                       "las formas (TabM)"),
        "que_cubre_el_veredicto": ("SOLO los conjuntos de esta ejecución. `resultados` es el fichero "
                                   "fusionado (puede traer registros de otras ejecuciones, que sirven "
                                   "de caché y no entran)"),
        "resultados": en_el_fichero,
        "cuadro_por_conjunto": cuadro_por_conjunto(
            [r for r in en_el_fichero if r["dataset"] in nombres], nombres, metrica_por_dataset),
        "veredicto": veredicto,
    }
    c3.sellar_la_salida(salida)
    c3p._escribir_atomicamente(ruta_salida, salida)
    return salida


# ---------------------------------------------------------------------------
# 11. --estimar: UN intento real por conjunto y la cuenta de la pasada entera
# ---------------------------------------------------------------------------

def estimar_desde_medidas(medidas: dict[str, dict], estudio: dict, *, pliegues: tuple = PLIEGUES
                          ) -> dict:
    """La cuenta, PURA. De cada conjunto se mide UN intento de TabM en S y UNO
    de lightgbm (pliegue 0). Se supone: la densa de hoy tarda lo de TabM (mismo
    presupuesto) y sklearn.hgb lo de lightgbm; TabM a 60/120 s tarda lo de S si
    paró por paciencia (convergió: más tiempo no cambia nada), y si paró por
    plazo, lo de S más la parte de plazo que se le añade; el reajuste, lo de S
    acotado por su presupuesto. Cada intento, acotado por su techo (presupuesto
    + margen). Es una ESTIMACIÓN, no una medición."""
    detalle, total_s, cota_s = [], 0.0, 0.0
    fraccion = estudio["fraccion_del_presupuesto_para_entrenar"][NOMBRE_TABM]
    base = estudio["segundos_por_intento"][NOMBRE_TABM]
    for nombre, m in sorted(medidas.items()):
        techo = lambda s: s + MARGEN_S  # noqa: E731
        s_tabm = min(m["wall_s"], techo(base))
        s_lgbm = min(m["wall_s_ligero"], techo(estudio["segundos_por_intento"]["lightgbm"]))
        parado = bool(m.get("parado_por_plazo"))
        curva = 0.0
        for s in estudio["curva_s_mas_segundos"]:
            extra = fraccion * (s - base) if parado else 0.0
            curva += min(s_tabm + extra, techo(s))
        reajuste = 2 * min(s_tabm, techo(estudio["segundos_del_ajuste_final"][NOMBRE_TABM])) \
            + 2 * min(s_lgbm, techo(estudio["segundos_del_ajuste_final"]["lightgbm"]))
        por_pliegue = 2 * s_tabm + 2 * s_lgbm + curva + 4 * m["preparacion_s"]
        segundos = m["carga_s"] + len(pliegues) * por_pliegue + reajuste
        cota = (m["carga_s"] + len(pliegues) * (
            2 * techo(base) + 2 * techo(estudio["segundos_por_intento"]["lightgbm"])
            + sum(techo(s) for s in estudio["curva_s_mas_segundos"]) + 4 * m["preparacion_s"])
            + 2 * techo(estudio["segundos_del_ajuste_final"][NOMBRE_TABM])
            + 2 * techo(estudio["segundos_del_ajuste_final"]["lightgbm"]))
        total_s += segundos
        cota_s += cota
        detalle.append({"dataset": nombre, "horas": round(segundos / 3600, 3),
                        "horas_cota": round(cota / 3600, 3),
                        "s_tabm_en_s": round(s_tabm, 1), "s_ligero_en_s": round(s_lgbm, 1),
                        "parado_por_plazo_en_s": parado, "carga_s": m["carga_s"],
                        "preparacion_s": m["preparacion_s"]})
    return {"procesos": 1, "por_conjunto": detalle, "total_horas": round(total_s / 3600, 2),
            "total_horas_cota": round(cota_s / 3600, 2), "total_s": round(total_s, 1),
            "que_supone": estimar_desde_medidas.__doc__}


def _medir_un_intento_real(ds, protocolo, estudio, motores, metric_id) -> dict:
    """UN intento TabM en S y UNO de lightgbm en S (pliegue 0) con la carga y la
    preparación del padre aparte, y el pico del hijo."""
    t_carga = time.perf_counter()
    por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
    carga_s = time.perf_counter() - t_carga
    pl = propuesta.pliegues.pliegue_de(repeticion=REPETICION, pliegue=PLIEGUES[0])
    if pl is None:
        raise SystemExit(f"{ds.nombre}: no hay pliegue (0, 0) en su partición")
    preparar = _preparador(por_id, objetivo, predictores, tuple(pl.entrena), tuple(pl.valida),
                           propuesta.plan.observaciones_del_rol("test"))
    salida = {"dataset": ds.nombre, "carga_s": round(carga_s, 3),
              "n_filas": particion["n_filas_con_objetivo"], "n_predictores": particion["n_predictores"]}
    for etiqueta, nombre in (("", NOMBRE_TABM), ("_ligero", "lightgbm")):
        motor = motores[nombre]
        train, val, test, prep_s = preparar(motor)
        wall = estudio["segundos_por_intento"][nombre]
        with _PicoDelHijo() as pico:
            t0 = time.perf_counter()
            intento = ejecutar_intento_aislado(
                motor, train, val, test, spec,
                Presupuesto(wall_seconds=wall, hilos=estudio["hilos"],
                            seed=estudio["semilla_de_los_intentos"]),
                candidate=f"{nombre}-estimacion-c5a", split_plan_digest=propuesta.plan.digest(),
                dataset=ds.nombre, pliegue=PLIEGUES[0], repeticion=REPETICION,
                margen_segundos=MARGEN_S)
            wall_s = time.perf_counter() - t0
        e = _entrenamiento(getattr(intento, "config_efectiva", None))
        salida[f"wall_s{etiqueta}"] = round(wall_s, 3)
        salida[f"estado{etiqueta}"] = intento.estado
        if nombre == NOMBRE_TABM:
            salida.update(preparacion_s=round(prep_s, 3), pico_hijo_mb=pico.pico_mb,
                          parado_por_plazo=e["parado_por_plazo"], **{
                              k: e[k] for k in ("epocas_ejecutadas", "epocas_completas",
                                                "plazo_antes_de_la_primera_epoca_completa")})
        else:
            salida["pico_hijo_mb_ligero"] = pico.pico_mb
    return salida


def para_encolar(estimada_s: float, *, memoria: str, digest_de_la_regla_: str) -> dict:
    """Las órdenes de encolado de `c3p.para_encolar` (parametrizada) con el
    digest de la regla en la orden y el commit del Studio en COMMITS=."""
    commits = c3p._commits_de_ahora()
    estado_studio = c3._estado_del_repositorio(_raiz_del_studio())
    commits["matrixaistudio"] = {"sha": estado_studio.get("commit"),
                                 "sin_commitear": bool(estado_studio.get("ficheros_modificados"))}
    orden = c3p.para_encolar(
        estimada_s, memoria=memoria, commits=commits,
        fin_de_la_suite=c3p.fin_de_la_suite_nocturna_medido(),
        nombre_del_trabajo=NOMBRE_DEL_TRABAJO_EN_LA_COLA,
        guion=f"{GUION} --regla-registrada {digest_de_la_regla_}",
        salida_en_la_cola=SALIDA_EN_LA_COLA)
    sha = commits["matrixaistudio"]["sha"] or "<sha>"

    def con_el_studio(x):
        if isinstance(x, str):
            return re.sub(r"(COMMITS=matrixAI=\S+?,matrixai-engines=[0-9a-f<>sha]+)",
                          lambda m: f"{m.group(1)},matrixaistudio={sha}", x)
        if isinstance(x, dict):
            return {k: con_el_studio(v) for k, v in x.items()}
        if isinstance(x, list):
            return [con_el_studio(v) for v in x]
        return x
    return con_el_studio(orden)


def cmd_estimar(protocolo, datasets, estudio, motores, metrica_por_dataset, *, ruta_salida: Path,
                regla: dict) -> None:
    if ruta_salida.exists() and c3p.tipo_del_fichero(c3p._leer_json(ruta_salida)) != "estimar":
        raise SystemExit(f"{ruta_salida} no es una estimación: --estimar no lo pisa")
    medidas: dict[str, dict] = {}
    print(f"ESTA MEDICIÓN: {len(datasets)} conjuntos x (TabM + lightgbm) en S, pliegue 0, "
          f"1 proceso, 1 hilo", flush=True)
    with c3p._PicoDeMemoriaDelArbol() as arbol:
        for ds in datasets:
            print(f"  midiendo {ds.nombre}...", flush=True)
            m = medidas[ds.nombre] = _medir_un_intento_real(ds, protocolo, estudio, motores,
                                                            metrica_por_dataset[ds.nombre])
            print(f"  {ds.nombre}: TabM {m['wall_s']:.1f}s ({m['estado']}, épocas "
                  f"{m['epocas_ejecutadas']}, plazo={m['parado_por_plazo']}, pico "
                  f"{m['pico_hijo_mb']} MB), lightgbm {m['wall_s_ligero']:.1f}s, carga "
                  f"{m['carga_s']:.1f}s, prep {m['preparacion_s']:.1f}s", flush=True)
    estimacion = estimar_desde_medidas(medidas, estudio)
    pico_hijo = max([m["pico_hijo_mb"] or 0 for m in medidas.values()] or [0])
    memory_max = f"{max(4, math.ceil(max(arbol.pico_mb, 1.0) * 1.3 / 1024))}G"
    salida = {
        "corte": "119-C5a", "sub_corte": "estimacion (--estimar: UN intento por conjunto)",
        "tipo_de_ejecucion": "estimar", "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "condiciones_del_studio": estudio, "regla": regla,
        "conjuntos_medidos": [d.nombre for d in datasets], "medidas": medidas,
        "estimacion": estimacion,
        "memoria": {"pico_mb_del_arbol_de_procesos": arbol.pico_mb,
                    "pico_mb_del_hijo_maximo": pico_hijo, "memory_max_sugerido": memory_max,
                    "que_mide": "el pico del árbol (padre + hijo) de esta estimación, como en C3 (M3)"},
        "para_encolar": para_encolar(estimacion["total_s"], memoria=memory_max,
                                     digest_de_la_regla_=regla["digest"]),
    }
    c3p._escribir_atomicamente(ruta_salida, salida)
    print(f"\nestimación escrita en {ruta_salida}")
    print(f"  1 proceso: {estimacion['total_horas']:.2f} h (cota {estimacion['total_horas_cota']:.2f} h); "
          f"memoria: árbol {arbol.pico_mb:.0f} MB -> MemoryMax {memory_max}")
    dia = salida["para_encolar"]["de_dia"]
    print(f"  DE DÍA ({'cabe' if dia['cabe'] else 'NO cabe'}), lanzar la cola antes de las "
          f"{dia['lanzar_antes_de']}:\n    {dia['encolar']}\n    {dia['lanzar_la_cola']}")


# ---------------------------------------------------------------------------
# 12. main
# ---------------------------------------------------------------------------

def _argumentos(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true", help="ignora el caché entero")
    parser.add_argument("--salida", default=None, help="ruta del JSON (absoluta fuera del árbol en la cola)")
    parser.add_argument("--estimar", action="store_true",
                        help="UN intento real (TabM y lightgbm en S) por conjunto y la cuenta")
    parser.add_argument("--estimar-conjuntos", default=None,
                        help="con --estimar, los conjuntos que se miden (coma, DENTRO de los 13)")
    parser.add_argument("--solo", default=None, help="conjuntos separados por coma, DENTRO de los 13")
    parser.add_argument("--humo", action="store_true",
                        help=f"{', '.join(CONJUNTOS_DEL_HUMO)}, pliegue 0, condición S y reajuste")
    parser.add_argument("--regla", action="store_true", help="imprime el digest de la regla y sale")
    parser.add_argument("--regla-registrada", default=None,
                        help="el digest de la regla registrado en el contrato (la pasada real lo exige)")
    parser.add_argument("--comprobar", action="store_true",
                        help="todo lo previo a medir (guardas, condiciones, regla, anclaje) y sale")
    args = parser.parse_args(argv)
    if args.humo and args.solo:
        raise SystemExit("--humo y --solo son incompatibles")
    if args.estimar and (args.humo or args.solo):
        raise SystemExit("--estimar no se combina con --humo ni con --solo (usa --estimar-conjuntos)")
    if args.estimar_conjuntos and not args.estimar:
        raise SystemExit("--estimar-conjuntos solo va con --estimar")
    return args


def tipo_de_ejecucion(*, humo: bool, solo: str | None, estimar: bool) -> str:
    return c3p.tipo_de_ejecucion(humo=humo, solo=solo, estimar=estimar)


def main(argv=None) -> None:
    args = _argumentos(argv)
    if args.regla:
        print(digest_de_la_regla())
        return
    tipo = tipo_de_ejecucion(humo=args.humo, solo=args.solo, estimar=args.estimar)
    ruta_salida = c3p.ruta_de_salida(tipo, args.salida, salida_por_omision=SALIDA_POR_OMISION,
                                     ruta_resultado_real=RUTA_DEL_RESULTADO)

    # AL ARRANCAR, ANTES DE CARGAR NADA: protocolo, guarda de los sellados, regla.
    c6.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    solo = (",".join(CONJUNTOS_DEL_HUMO) if tipo == "humo"
            else args.estimar_conjuntos if tipo == "estimar" else args.solo)
    datasets, subconjunto = datasets_de_c5a(todos, solo=solo)
    regla = exigir_la_regla_registrada(args.regla_registrada, tipo)
    estudio = condiciones_del_studio()
    c6._exigir_que_quepa()
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)
    motores = {m: motor_para(m) for m in MOTORES_DE_LA_CONDICION_S}

    pliegues = PLIEGUES_DEL_HUMO if tipo == "humo" else PLIEGUES
    con_curva = tipo != "humo"
    print(f"119-C5a ({tipo}): {len(datasets)} conjuntos, pliegues {list(pliegues)}, "
          f"1 proceso, {estudio['hilos']} hilo. Studio: TabM/densa "
          f"{estudio['segundos_por_intento'][NOMBRE_TABM]:.2f} s (plazo "
          f"{estudio['plazo_de_entrenamiento_s'][NOMBRE_TABM]:.2f} s), ligeros "
          f"{estudio['segundos_por_intento']['lightgbm']:.2f} s, reajuste "
          f"{estudio['segundos_del_ajuste_final'][NOMBRE_TABM]:.1f} s. Regla {regla['digest'][:16]}. "
          f"Salida: {ruta_salida}", flush=True)

    if tipo == "estimar":
        cmd_estimar(protocolo, datasets, estudio, motores, metrica_por_dataset,
                    ruta_salida=ruta_salida, regla=regla)
        return

    cache_previo, payload_previo = cargar_salida_previa(ruta_salida, tipo)
    if args.forzar:
        cache_previo = {}
    previos = list(payload_previo.get("resultados") or [])
    componentes = componentes_del_digest_del_entorno()
    entorno_digest = c3p._digest_de_componentes(componentes)
    procedencia = procedencia_de_la_medicion(
        digests={"entorno": entorno_digest,
                 **{f"motor:{m}": digest_del_motor(m) for m in MOTORES_DE_LA_CONDICION_S}},
        datasets=datasets)
    exigir_el_anclaje(procedencia, tipo)
    if args.comprobar:
        print(f"COMPROBADO: guardas, condiciones, regla ({regla['digest']}), entorno "
              f"{entorno_digest}, anclable={procedencia['anclable']}. No se mide.")
        return

    resultados: list[dict] = []
    tiempos: dict[str, dict] = {}
    reusados = [0]
    inicio = time.perf_counter()
    ultimo = [time.perf_counter()]
    nombres = [d.nombre for d in datasets]

    def guardar(parcial: bool) -> dict:
        ultimo[0] = time.perf_counter()
        veredicto = None
        if not parcial:
            veredicto = componer_veredicto(
                resultados, nombres=nombres, metrica_por_dataset=metrica_por_dataset,
                regla_de_cierre=protocolo.regla_de_cierre, pliegues=pliegues)
        return _componer_y_guardar(
            resultados, previos, veredicto, procedencia, payload_previo, ruta_salida, tipo=tipo,
            datasets=datasets, subconjunto=subconjunto, estudio=estudio, regla=regla,
            metrica_por_dataset=metrica_por_dataset, pliegues=pliegues, tiempos_del_padre=tiempos,
            componentes_del_digest=componentes, total_wall_s=time.perf_counter() - inicio,
            reusados=reusados[0], parcial=parcial, protocolo=protocolo, con_curva=con_curva)

    for ds in datasets:
        def tras_un_pliegue(registros, t, _ds=ds):
            tiempos[_ds.nombre] = t
            resultados[:] = [r for r in resultados if r["dataset"] != _ds.nombre] + list(registros)
            guardar(parcial=True)

        registros, t = medir_conjunto(
            ds, protocolo, estudio, metric_id=metrica_por_dataset[ds.nombre], pliegues=pliegues,
            con_curva=con_curva, con_reajuste=True, cache_previo=cache_previo, forzar=args.forzar,
            entorno_digest=entorno_digest, procedencia=procedencia, motores=motores,
            al_terminar_un_pliegue=tras_un_pliegue)
        tiempos[ds.nombre] = t
        resultados[:] = [r for r in resultados if r["dataset"] != ds.nombre] + registros
        reusados[0] = sum(1 for r in resultados if r.get("reusado"))
        guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados[0]} reusados) ===")
    salida = guardar(parcial=False)
    v = salida["veredicto"]
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    print(f"VEREDICTO 119-C5a{' (' + tipo.upper() + ' -- no cuenta)' if tipo != 'pasada' else ''}: "
          f"completa {v['completa']['numero']} ({'SÍ' if v['completa']['cumple'] else 'NO'}); "
          f"aprende {v['aprende']['numero']} ({'SÍ' if v['aprende']['cumple'] else 'NO'}); "
          f"compite {v['compite']['cumplidos_de_la_cartera']} y pierde "
          f"{v['compite']['conjuntos_perdidos_contra_la_densa_de_hoy']} contra la densa "
          f"({'SÍ' if v['compite']['cumple'] else 'NO'}); cabe ({'SÍ' if v['cabe']['cumple'] else 'NO'})")


if __name__ == "__main__":
    main()
