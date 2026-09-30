#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C4 — la CONFIRMACIÓN del motor nuevo (`matrixai.dense.tabm_cpu`, TabM +
PLR) en los 8 conjuntos SELLADOS de Fase 0, UNA SOLA VEZ, y el X/40 final que
se publica.

Contrato 119 (`documentacion/119_LA_RED_DENSA_NUEVA_CONTRACT.md`) y protocolo
v4 (`protocolo_119_v4.json`, `veredicto`):

* **`regla_de_subida`**: «el motor nuevo sube a confirmación en los sellados
  (C4) si cumple las DOS condiciones sobre los 32 no sellados (C3)». C3 lo
  cumplió (31/32 frente a 11/32 de la densa v2, `resultado_pasada_119_c3.json`,
  sellado y verificado aquí antes de componer nada).
* **`confirmacion`**: «la receta elegida, UNA vez, en los SELLADOS; es la
  cifra que se publica». Por eso este guion, a diferencia de C3, se NIEGA a
  volver a medir si CUALQUIERA de sus rutas fijas (la de la cola, las dos
  del árbol y la `--salida` pedida: `RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION`) ya
  tiene un resultado de tipo `pasada` completo — reanudar un PARCIAL (la
  misma ejecución, cortada por el tope) sí se permite, pero REUSANDO todo lo
  ya medido, fallos incluidos (un intento sellado fallido ES su medida), y
  PARANDO si algo de lo ya medido no es reusable (otro código): nunca se
  re-mide un sellado.
* **D2** (decisión de Roberto, `119_LA_RED_DENSA_NUEVA_CONTRACT.md`): ≥32/40
  entra en la cartera; entre el techo ESTIMADO de los retoques (~27/40,
  análisis del 24-09, nunca medido) y 32/40, avance publicado sin cartera;
  por debajo de 27/40, se cierra como el 118.

QUÉ MIDE: los 8 SELLADOS (`d.sellado` del protocolo v4) — kr-vs-kp, letter,
splice, pol, pc3, banknote-authentication, Satellite, diamonds — con la MISMA
receta, particiones, presupuesto y guardias que C3 (enmienda 2 incluida). El
X/40 = los cumplidos de C3 (YA sellados y verificados aquí, no recalculados
desde cero) + los cumplidos de C4 en los 8 sellados, con la MISMA función
(`c3p.cumplidos_de_los_dos_motores`, la que usa `veredicto_final` de C3,
sobre `_cumplidos_de` + `campo_de_la_comparacion` de C3 y de 118 — nunca una
segunda implementación: antes de la auditoría del guion era una COPIA, M6).

QUÉ NO HACE, Y POR QUÉ:

* **NUNCA ejecuta el motor real sobre un sellado más de una vez.** La única
  vez que este guion toca un ARFF sellado con el `ejecutar_intento_aislado`
  REAL es en la pasada real (`tipo_de_ejecucion() == "pasada"`), y esa
  pasada se niega a repetirse una vez completa. **`--solo` NO corre con el
  motor real** (auditoría del guion, 30-09, I2: antes lo hacía, y una orden
  documentada aquí gastaba la confirmación de ese conjunto): si
  `ejecutar_intento_aislado` es el de `matrixai_engines.subproceso`, PARA
  antes de cargar nada. Existe solo para que las PRUEBAS recorran el guion
  con un motor FALSO (que sustituye `ejecutar_intento_aislado`), siempre
  dentro de los 8 sellados (nunca un no sellado, que es de C3), y escribe en
  un fichero APARTE que nunca se confunde con la confirmación real (misma
  disciplina de `ruta_de_salida`/`tipo_del_fichero` que C3).
* **`--estimar` no toca ningún sellado con el motor real.** Usa los tiempos
  ya MEDIDOS de C3 por conjunto (`resultado_pasada_119_c3.json`, verificado
  aquí) e interpola por celdas dentro de cada cubo con
  `estimar_desde_medidas` (C3, reutilizada tal cual) — igual que C3 hace con
  los conjuntos de SU estimación que no mide directamente. La única llamada
  real al motor durante `--estimar` es la sonda sintética de la guardia de
  la receta (`medir_la_receta_del_motor`, 300 filas al azar, nunca un
  dataset de Fase 0), idéntica a la que C3 corre para CUALQUIER tipo de
  ejecución.
* **No implementa `--humo` ni `--forzar`.** El humo del corte (probar el
  camino entero sin medir nada) lo hacen las PRUEBAS (`--solo` o la pasada
  entera sobre sellados baratos, siempre con un motor falso), o el propio
  `--humo` de C3 sobre un NO sellado (encargo: «con un MOTOR FALSO, o sobre
  un conjunto NO sellado con el guion de C3»). `--forzar` de C3 ignora el
  caché entero para re-medir: eso es exactamente lo que la regla de «UNA
  SOLA VEZ» de C4 prohíbe, así que no se ofrece ese escape.
* **No re-mide los 32 no sellados.** Los lee, sellados, del resultado de C3
  y los suma tal cual — tocarlos de nuevo sería otra ejecución de C3, que su
  propio guion ya prohíbe reanudar sobre un resultado completo.

QUÉ SE REUTILIZA, Y DE DÓNDE — nada se copia a mano:

* TODAS las guardias (protocolos encadenados, configuración/receta del
  motor, enmienda 2, partición de un conjunto contra la v2, arquitectura
  declarada por cada intento), el intento en subproceso, el digest de la
  caché y su reusabilidad, la fusión de la salida, el rastro de
  reintentos, el veredicto por conjunto (I2), el sellado y la procedencia,
  la memoria de encolado y `estimar_desde_medidas`: TODO de
  `pasada_119_c3` (`import pasada_119_c3 as c3p`), que a su vez expone
  `c3p.c3` (`pasada_exploratoria_101_c3`), `c3p.c5` (`pasada_amplia_101_c5`),
  `c3p.c6` (`pasada_114c6_ensamblado`) y `c3p.p118` (`pasada_118_palanca`).
* `ruta_de_salida` y `para_encolar` de C3 se PARAMETRIZARON, y el cálculo de
  los cumplidos de `veredicto_final` se EXTRAJO a `cumplidos_de_los_dos_
  motores` (cambios mínimos, documentados en sus docstrings): con los
  valores por omisión, su comportamiento para C3 no cambia (probado:
  `test_119_c3_pasada.py` sigue en verde).
* La única pieza que C3 no tiene, porque no le hace falta, es la
  COMPOSICIÓN del X/40 (verificar los sellos de C3, exigir el mismo
  commit del motor, y sumar los cumplidos de los dos cortes) y la regla de
  «UNA vez» (rutas fijas, fallos que no se reintentan): eso es lo que añade
  este fichero.

CÓMO SE LANZA (después de COMMITEAR: la cola solo lleva lo commiteado, y
`--estimar` avisa si el HEAD no trae este guion o el árbol está sucio):

    python3 benchmarks/fase0/pasada_119_c4.py --estimar --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c4.py --salida <ruta absoluta, fuera del árbol>
        # LA CONFIRMACIÓN: los 8 sellados, UNA sola vez. Comprueba el anclaje
        # entero (sellos de C3, commit y digest del motor, árbol limpio, diff
        # del núcleo y entorno de C3) AL ARRANCAR, antes del primer intento.

`--solo` NO figura aquí a propósito: con el motor real PARA (ver arriba).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import pasada_119_c3 as c3p  # noqa: E402 -- TODO lo que se reutiliza vive detrás de este alias

from matrixai.estudio.validacion import digest_canonico  # noqa: E402 -- el mismo canonicalizador
from matrixai_engines.subproceso import (MARGEN_POR_DEFECTO_SEGUNDOS,  # noqa: E402
                                         ejecutar_intento_aislado)

#: El ejecutor REAL, guardado al importar: `main()` lo compara por IDENTIDAD
#: con el `ejecutar_intento_aislado` del módulo, que las pruebas sustituyen
#: por un motor falso. `--solo` con el real PARA (auditoría del guion, I2).
_EJECUTAR_INTENTO_AISLADO_REAL = ejecutar_intento_aislado

#: EL RESULTADO REAL de C3, YA SELLADO — se LEE y se VERIFICA, nunca se toca.
#: NÓTESE: no es `c3p.RUTA_DEL_RESULTADO` (el nombre por omisión que el
#: guion de C3 usa cuando se lanza SIN `--salida`, `pasada_119_c3_resultado.
#: json`) -- la pasada real de C3 corrió en la cola nocturna con `--salida`
#: explícito y el artefacto sellado que hay en el árbol, citado por el
#: contrato 119 y por `TASKS.md`, es este otro nombre (medido: el propio
#: `c3p.RUTA_DEL_RESULTADO` no existe en el árbol).
RUTA_DEL_RESULTADO_C3 = _AQUI / "resultado_pasada_119_c3.json"

#: EL RESULTADO REAL de este corte. Ni `--solo` ni `--estimar` escriben aquí.
RUTA_DEL_RESULTADO_C4 = _AQUI / "pasada_119_c4_resultado.json"
SALIDA_POR_OMISION_C4 = {
    "pasada": RUTA_DEL_RESULTADO_C4,
    "solo": _AQUI / "pasada_119_c4_solo_resultado.json",
    "estimar": _AQUI / "estimacion_pasada_119_c4.json",
}

#: Donde la COLA escribe la confirmación (absoluta, fuera del worktree: la cola
#: borra sus worktrees al terminar). La misma que imprime `--estimar`.
SALIDA_DE_C4_EN_LA_COLA = Path(
    "/home/deployer/cola-nocturna/resultados/119-c4/pasada_119_c4_resultado.json")
NOMBRE_DEL_TRABAJO_EN_LA_COLA = "119-c4"
#: Este guion, por su ruta en el repo: la que la cola ejecuta en su worktree.
GUION_DE_C4 = "benchmarks/fase0/pasada_119_c4.py"

#: Con qué nombre se COPIA al árbol el resultado de la cola para commitearlo:
#: C3 se copió como `resultado_pasada_119_c3.json`, no con su nombre por
#: omisión (`pasada_119_c3_resultado.json`), y C4 seguirá esa costumbre.
RUTA_DEL_RESULTADO_C4_EN_EL_ARBOL = _AQUI / "resultado_pasada_119_c4.json"

#: «UNA SOLA EJECUCIÓN» mira SIEMPRE estas rutas, además de la `--salida`
#: pedida (auditoría del guion, 30-09, I3: antes miraba solo la `--salida`, y
#: con OTRA `--salida` la pasada re-medía los 8 sellados y componía otro
#: X/40 -- medido). Si CUALQUIERA trae una pasada completa, PARA.
RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION = (
    SALIDA_DE_C4_EN_LA_COLA,
    RUTA_DEL_RESULTADO_C4,
    RUTA_DEL_RESULTADO_C4_EN_EL_ARBOL,
)

#: D2 del contrato 119 (decisión de Roberto): el listón de la cartera SÍ está
#: medido (80% de 40, `regla_de_cierre` del protocolo); el techo de los
#: retoques es una ESTIMACIÓN del análisis del 24-09, nunca una medida (ver
#: CLAUDE.md, «un techo no es una medida») — se declara como tal siempre.
UMBRAL_CARTERA_D2 = 32
TECHO_DE_LOS_RETOQUES_ESTIMADO_D2 = 27

#: CÓMO CUENTAN LOS FALLOS EN EL X/40 (auditoría del guion, M7 e I4): la
#: regla de C3 (`COMO_SE_CUENTAN_LOS_FALLOS`, enmienda 2) EXTENDIDA a los
#: sellados, dicho aquí porque la enmienda no lo dice: su campo `pasada`
#: habla de «SOLO el motor nuevo en los 32 no sellados».
COMO_SE_CUENTAN_LOS_FALLOS_EN_C4 = {
    **c3p.COMO_SE_CUENTAN_LOS_FALLOS,
    "extension_a_los_sellados": (
        "la enmienda 2 del protocolo v4 (se_fija.fallos_en_el_veredicto) se escribió para la "
        "pasada de C3 («SOLO el motor nuevo en los 32 no sellados»); 119-C4 aplica EXACTAMENTE "
        "la misma regla a los 8 sellados, con la misma función (cumplidos_de_los_dos_motores de "
        "119-C3): un intento del motor nuevo fallido, ausente o sin métrica de cierre PIERDE "
        "su conjunto. Es coherente con regla_de_cierre (un fallo es un conjunto perdido), pero "
        "la enmienda no lo escribe: se declara aquí"),
    "sin_reintentos_en_los_sellados": (
        "en C4 un fallo NO se reintenta nunca (a diferencia de C3, cuya regla de la casa es "
        "reintentar lo que no está completed): un intento sellado ya ejecutado ES la medida, "
        "falle o no, y al reanudar un parcial se REUSA tal cual; si un registro previo no es "
        "reusable (otro código, otro presupuesto u otros datos) la pasada PARA en vez de volver "
        "a medirlo. «La receta elegida, UNA vez, en los SELLADOS» (veredicto.confirmacion)"),
}

#: El SUELO de `memory_max_sugerido` cuando `resultado_pasada_119_c3.json` no
#: trae `rss_pico_mb` (medido: ninguno de sus 387 registros lo trae). NO es
#: un mínimo inventado: es el techo con el que C3 corrió DE VERDAD (cola
#: nocturna, `~/encolar.sh 119-c3 32100 8G ...`, contrato 119, estado del
#: 30-09) y está por encima del pico de cgroup que la auditoría 3 midió en
#: Allstate_Claims_Severity, el conjunto más caro de los no sellados (3.951
#: MB, CLAUDE.md «El techo NO es decorativo, y va MEDIDO»).
MEMORIA_MINIMA_SUGERIDA_GB = 8
FUENTE_DE_LA_MEMORIA_MINIMA = (
    "C3 corrió de verdad con techo 8 GiB (cola nocturna, `~/encolar.sh 119-c3 32100 8G ...`, "
    "contrato 119 -- estado del 30-09) y la auditoría 3 (119-C3) midió un pico de cgroup de "
    "3.951 MB sobre Allstate_Claims_Severity, el conjunto más caro de los no sellados "
    "(CLAUDE.md, «El techo NO es decorativo, y va MEDIDO»). Sin `rss_pico_mb` en los registros "
    "de C3, el suelo de C4 es ese mismo techo YA PROBADO, no un mínimo sin medir nada del "
    "motor nuevo")


# ---------------------------------------------------------------------------
# 1. LOS CONJUNTOS: los 8 SELLADOS -- los 32 no sellados se NIEGAN (son de C3)
# ---------------------------------------------------------------------------

def datasets_de_c4(todos: list, sellados: list, *, solo: str | None) -> tuple[list, list | None]:
    """Los conjuntos que esta pasada mide: los 8 sellados, o el subconjunto
    de `--solo` -- SIEMPRE dentro de los sellados. Espejo de
    `c3p.datasets_de_c3`, con la condición invertida: aquí lo que se niega es
    un NO sellado, porque esos son de C3. No es la misma función (la
    condición es la contraria) y vive en este fichero a propósito."""
    if not solo:
        return list(sellados), None
    pedidos = [n.strip() for n in solo.split(",") if n.strip()]
    por_nombre = {d.nombre: d for d in todos}
    desconocidos = [n for n in pedidos if n not in por_nombre]
    if desconocidos:
        raise SystemExit(f"--solo nombra conjuntos que no están en el protocolo v2: "
                         f"{desconocidos}")
    no_sellados_pedidos = [n for n in pedidos if not por_nombre[n].sellado]
    if no_sellados_pedidos:
        raise SystemExit(
            f"--solo pide conjunto(s) NO SELLADO(S) {no_sellados_pedidos}: 119-C4 mide SOLO "
            f"los 8 sellados -- los 32 no sellados son de C3 (protocolo_119_v4.json, "
            f"veredicto.confirmacion: «la receta elegida, UNA vez, en los SELLADOS»). Se niega "
            f"en vez de correr «solo como prueba»")
    return [por_nombre[n] for n in pedidos], pedidos


# ---------------------------------------------------------------------------
# 2. «UNA SOLA EJECUCIÓN»: se niega a repetir un resultado completo
# ---------------------------------------------------------------------------

def _rutas_de_una_sola_ejecucion(ruta_salida: Path) -> list[Path]:
    """La `--salida` pedida MÁS las rutas fijas, sin repetir (por ruta real)."""
    rutas, vistas = [], set()
    for ruta in (ruta_salida, *RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION):
        real = Path(ruta).expanduser().resolve()
        if real not in vistas:
            vistas.add(real)
            rutas.append(real)
    return rutas


def exigir_una_sola_ejecucion(ruta_salida: Path) -> None:
    """PARA la pasada real si la `--salida` pedida O CUALQUIERA de las rutas
    fijas (`RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION`: la de la cola y las dos del
    árbol) ya trae un resultado de tipo `pasada` que no sea explícitamente
    PARCIAL (`parcial` distinto de `true` cuenta como completo: falla
    cerrado). Reanudar un PARCIAL (la misma ejecución, cortada por el tope)
    SÍ se permite, con la regla de reuso de C4 (`decidir_con_el_previo`).

    Un fichero que no se puede leer revienta aquí (`_leer_json`), y eso
    también para: mejor no medir que medir por segunda vez."""
    for ruta in _rutas_de_una_sola_ejecucion(ruta_salida):
        if not ruta.exists():
            continue
        payload = c3p._leer_json(ruta)
        if c3p.tipo_del_fichero(payload) == "pasada" and payload.get("parcial") is not True:
            raise SystemExit(
                f"{ruta} ya tiene un resultado COMPLETO de 119-C4 (parcial: "
                f"{payload.get('parcial')!r}): protocolo_119_v4.json, veredicto.confirmacion dice "
                f"«la receta elegida, UNA vez, en los SELLADOS; es la cifra que se publica» -- no "
                f"se vuelve a medir, tampoco con otra --salida (rutas que se miran siempre: "
                f"{[str(r) for r in _rutas_de_una_sola_ejecucion(ruta_salida)]}). Si esto es un "
                f"error, mover o borrar el fichero a mano; no hay --forzar para un sellado")


def _reusable_c4(previo: dict | None, entorno_digest: str, motor_digest: str,
                 wall_seconds: float | None, *, datos_sha256: str | None) -> bool:
    """`c3._reusable` (mismo código de entorno y de motor, mismo presupuesto)
    y el MISMO ARFF, como `c3p._reusable_c3` -- pero SIN exigir que el estado
    cuente como medida: en C4 un fallido ES la medida (I4 a)."""
    if not c3p.c3._reusable(previo, entorno_digest, motor_digest, wall_seconds):
        return False
    return datos_sha256 is not None and previo.get("datos_sha256") == datos_sha256


def decidir_con_el_previo(previo: dict | None, *, entorno_digest: str, motor_digest: str,
                          wall_seconds: float, datos_sha256: str | None) -> str:
    """Qué hace C4 con el registro previo de UN intento: `"medir"` si no hay
    previo, `"reusar"` si es reusable (completado O fallido), y PARA si hay
    un previo que no lo es (auditoría del guion, I4: antes se re-medía en
    silencio -- medido: una reanudación con otro código re-midió los 110
    intentos sellados). Nunca «reintentar»."""
    if previo is None:
        return "medir"
    if _reusable_c4(previo, entorno_digest, motor_digest, wall_seconds,
                    datos_sha256=datos_sha256):
        return "reusar"
    raise SystemExit(
        f"el intento sellado {previo.get('dataset')} rep={previo.get('repeticion')} "
        f"pliegue={previo.get('pliegue')} YA se ejecutó (estado={previo.get('estado')!r}, "
        f"entorno {previo.get('entorno_digest')}, motor {previo.get('motor_digest')}, "
        f"presupuesto {previo.get('presupuesto_wall_s')}) y no es reusable con lo de ahora "
        f"(entorno {entorno_digest}, motor {motor_digest}, presupuesto {wall_seconds}, datos "
        f"{'iguales' if previo.get('datos_sha256') == datos_sha256 else 'DISTINTOS'}): un "
        f"sellado se mide UNA vez, así que no se vuelve a medir. Continuar con el MISMO commit "
        f"(COMMITS=) o decidir a mano qué hacer con el parcial")


def exigir_que_lo_previo_sea_reusable(previos: list[dict], *, nombres: set[str],
                                      entorno_digest: str, motor_digest: str,
                                      datos_sha256_por_conjunto: dict[str, str | None]) -> dict:
    """La misma regla de `decidir_con_el_previo`, AL ARRANCAR y sobre TODOS
    los registros previos de los conjuntos que se van a medir (sin el
    presupuesto, que sale del plan de cada conjunto y se vuelve a mirar
    intento a intento): con otro código, PARA antes del primer intento."""
    no_reusables = [
        r for r in previos if r.get("dataset") in nombres and not (
            r.get("entorno_digest") == entorno_digest and r.get("motor_digest") == motor_digest
            and datos_sha256_por_conjunto.get(r["dataset"]) is not None
            and r.get("datos_sha256") == datos_sha256_por_conjunto.get(r["dataset"]))]
    if no_reusables:
        raise SystemExit(
            f"{len(no_reusables)} de {len(previos)} registros previos de la pasada son de OTRO "
            f"código o de otros datos (p. ej. {no_reusables[0].get('dataset')} rep="
            f"{no_reusables[0].get('repeticion')} pliegue={no_reusables[0].get('pliegue')}: "
            f"entorno {no_reusables[0].get('entorno_digest')} frente a {entorno_digest}, motor "
            f"{no_reusables[0].get('motor_digest')} frente a {motor_digest}): un sellado se mide "
            f"UNA vez y esos ya se midieron, así que no se re-miden. Continuar con el MISMO commit "
            f"(COMMITS=) o decidir a mano qué hacer con el parcial")
    return {"n_previos": len(previos), "todos_reusables": True}


def ruta_de_salida_c4(tipo: str, salida: str | None) -> Path:
    """`c3p.ruta_de_salida` con las rutas de C4 y, además, ningún tipo que
    no sea la pasada real escribe en NINGUNA de las rutas fijas del resultado
    real (la de la cola y las dos del árbol), no solo en la por omisión."""
    ruta = c3p.ruta_de_salida(tipo, salida, salida_por_omision=SALIDA_POR_OMISION_C4,
                              ruta_resultado_real=RUTA_DEL_RESULTADO_C4)
    if tipo != "pasada" and ruta in {Path(r).expanduser().resolve()
                                     for r in RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION}:
        raise SystemExit(f"--{tipo} no escribe en una ruta del resultado real de 119-C4 "
                         f"({ruta}): una prueba no puede pisar la confirmación")
    return ruta


# ---------------------------------------------------------------------------
# 3. EL X/40: verificar los sellos de C3, exigir el mismo motor, sumar
# ---------------------------------------------------------------------------

def verificar_los_sellos_de_c3(payload_c3: dict) -> dict:
    """Recalcula los DOS sellos que `c3.sellar_la_salida` escribió en el
    resultado de C3 y los compara con los declarados: `digest_resultados_
    crudos` (`digest_canonico` de TODO el objeto salvo ese mismo campo) y
    `digest_solo_de_resultados` (`digest_canonico` de `resultados` a secas).
    Nunca los vuelve a escribir: solo verifica."""
    sin_el_campo = {k: v for k, v in payload_c3.items() if k != "digest_resultados_crudos"}
    crudos_recalculado = digest_canonico(sin_el_campo)
    solo_recalculado = digest_canonico(payload_c3.get("resultados"))
    problemas = []
    if crudos_recalculado != payload_c3.get("digest_resultados_crudos"):
        problemas.append(
            f"digest_resultados_crudos NO cuadra: declarado "
            f"{str(payload_c3.get('digest_resultados_crudos'))[:16]}, recalculado "
            f"{crudos_recalculado[:16]} -- el resultado de C3 se ha tocado después de sellarlo")
    if solo_recalculado != payload_c3.get("digest_solo_de_resultados"):
        problemas.append(
            f"digest_solo_de_resultados NO cuadra: declarado "
            f"{str(payload_c3.get('digest_solo_de_resultados'))[:16]}, recalculado "
            f"{solo_recalculado[:16]} -- los `resultados` de C3 no son los que su sello dice")
    return {"cuadra": not problemas, "problemas": problemas,
            "digest_resultados_crudos_recalculado": crudos_recalculado,
            "digest_solo_de_resultados_recalculado": solo_recalculado}


#: LO ÚNICO que puede haber cambiado en matrixAI entre el commit que midió
#: C3 y el que mide C4 sin que sea "otro motor" (defecto de diseño señalado
#: por el supervisor, 30-09: `~/encolar.sh` fija el HEAD COMMITEADO de cada
#: repo, así que este guion y la parametrización de C3 TIENEN que estar
#: commiteados para que la cola los lleve -- exigir el MISMO commit de
#: matrixAI que C3 bloquearía SIEMPRE el único camino legítimo). tests/ y
#: documentacion/ tampoco cambian lo que un estudio ejecuta. CUALQUIER otro
#: fichero -- sobre todo `matrixai/`, otro de `benchmarks/fase0/` (los JSON
#: del protocolo incluidos) -- PARA: eso sí sería medir con otro núcleo.
FICHEROS_DE_NUCLEO_ADMITIDOS_ENTRE_C3_Y_C4 = (
    "benchmarks/fase0/pasada_119_c4.py",
    "benchmarks/fase0/pasada_119_c3.py",
)
PREFIJOS_DE_NUCLEO_ADMITIDOS_ENTRE_C3_Y_C4 = ("tests/", "documentacion/")

#: EL RESULTADO DE C3 -- el único dato que el commit de C4 TIENE que traer y
#: el de C3 no puede tener (se escribió después de medir). Auditoría del
#: guion, 30-09, B1: sin admitirlo, la regla de arriba paraba SIEMPRE la
#: ejecución legítima (medido de punta a punta en un clon: los 8 sellados
#: medidos y, al final, «NO admitidos: [resultado_pasada_119_c3.json]»).
#: Se admite ESTE fichero exacto -- nunca el prefijo `benchmarks/fase0/`, que
#: abriría los JSON del protocolo -- y ATADO por los DOS lados, porque admitir
#: la ruta a secas admitiría también un C3 re-sellado:
#:
#: * por GIT: su blob en el commit de C4 tiene que ser el de `e078fe6` (el
#:   commit que lo añadió, «119-C3 MEDIDO»). Ata los BYTES que viajan en el
#:   commit que la cola saca en su worktree;
#: * SIN GIT: el `digest_resultados_crudos` del fichero que se LEE tiene que
#:   ser este exacto (y sus dos sellos, re-verificados, cuadrar). Ata el
#:   CONTENIDO leído de disco aunque el fichero se hubiera re-serializado, y
#:   no depende de que git responda.
#:
#: Los dos, no uno: el blob solo dice qué hay en el commit; el digest dice
#: qué se lee de verdad (con el árbol limpio coinciden, y eso es justo lo que
#: se comprueba). Medidos el 30-09: `git rev-parse e078fe6:<ruta>` y el campo
#: del JSON sellado.
RUTA_RELATIVA_DEL_RESULTADO_C3 = "benchmarks/fase0/resultado_pasada_119_c3.json"
BLOB_DEL_RESULTADO_C3_SELLADO = "998e7bdd5b53409dfdf1413596834c85943f855d"
DIGEST_RESULTADOS_CRUDOS_DE_C3 = (
    "640137070dc30a282a1cfc4d7702fd5f06b4e24ca9e2d720950b0c0b8bd995ce")

#: El REFUERZO SIN GIT del anclaje (propuesto por la auditoría del guion): los
#: componentes del digest del entorno de C3 (`c3p.componentes_del_digest_del_
#: entorno()`: el cierre de imports de C3, los compartidos de C5, el motor,
#: los protocolos y las versiones de python/numpy/sklearn/torch) calculados
#: AHORA tienen que ser, componente a componente, los que C3 registró en su
#: resultado. Solo puede diferir la parametrización de C3 (el fichero que la
#: regla de git también admite). Medido el 30-09 sobre el árbol principal:
#: 198 y 198 componentes; difieren ese y `objetivo_textos.py` (un cambio AJENO
#: sin commitear, que un worktree limpio de la cola no lleva).
COMPONENTES_DEL_ENTORNO_QUE_PUEDEN_DIFERIR_DE_C3 = frozenset({
    "core:benchmarks/fase0/pasada_119_c3.py",
})

POR_QUE_SE_ADMITEN_ESTOS_FICHEROS = (
    "el guion de C4 (pasada_119_c4.py) y la parametrización de C3 (pasada_119_c3.py) tienen "
    "que estar COMMITEADOS para que la cola los lleve (~/encolar.sh fija el HEAD committeado "
    "de cada repo): exigir el MISMO commit de matrixAI que C3 bloquearía SIEMPRE el único "
    "camino legítimo. tests/ y documentacion/ tampoco cambian lo que un estudio ejecuta. "
    "benchmarks/fase0/resultado_pasada_119_c3.json (el resultado de C3, que C4 lee y que el "
    "commit de C3 no podía tener) se admite SOLO con el blob de e078fe6. Cualquier otro "
    "fichero -- sobre todo matrixai/, otro de benchmarks/fase0/ (JSON del protocolo "
    "incluidos) -- significa que el núcleo que mide C4 no es el que midió C3")


def _fichero_admitido_entre_c3_y_c4(ruta: str) -> bool:
    return (ruta in FICHEROS_DE_NUCLEO_ADMITIDOS_ENTRE_C3_Y_C4
            or ruta.startswith(PREFIJOS_DE_NUCLEO_ADMITIDOS_ENTRE_C3_Y_C4))


def _raiz_del_nucleo() -> Path:
    """El repo de matrixAI del que la procedencia lee el commit de C4
    (`c3.procedencia_de_la_medicion` -> `_RUTAS_DE_REPOSITORIO`): el diff se
    mide en ESE mismo repo. En producción es `c3._RAIZ_DEL_CORE` (la raíz de
    este `__file__`: en la cola, el worktree)."""
    return Path(c3p.c3._RUTAS_DE_REPOSITORIO["matrixAI"])


def _blob_en(raiz: Path, commit: str, ruta: str) -> str | None:
    salida = subprocess.run(["git", "-C", str(raiz), "rev-parse", "--verify", "--quiet",
                             f"{commit}:{ruta}"], capture_output=True, text=True, timeout=30)
    return salida.stdout.strip() if salida.returncode == 0 and salida.stdout.strip() else None


def diferencia_de_nucleo_entre_c3_y_c4(raiz: Path, commit_c3: str, commit_c4: str) -> dict:
    """Los ficheros que cambiaron en el repo de `raiz` entre `commit_c3` y
    `commit_c4`, MEDIDOS con git (nunca supuestos) y comparados con lo
    admitido. `raiz` es la del repo que EJECUTA este guion (`_raiz_del_
    nucleo()`), no una ruta fija: un worktree tiene otra raíz. El resultado
    de C3 (`RUTA_RELATIVA_DEL_RESULTADO_C3`) solo se admite si su blob en
    `commit_c4` es `BLOB_DEL_RESULTADO_C3_SELLADO` (B1). PARA si `commit_c3`
    no existe en ESTE repo -- no hay con qué medir."""
    if commit_c3 == commit_c4:
        return {"raiz": str(raiz), "commit_c3": commit_c3, "commit_c4": commit_c4,
                "ficheros_cambiados": [], "ficheros_no_admitidos": [], "admitida": True,
                "motivo": "mismo commit en C3 y en C4: no hay nada que diferir",
                "resultado_de_c3_en_el_commit_de_c4": None,
                "por_que_se_admiten_estos": POR_QUE_SE_ADMITEN_ESTOS_FICHEROS}
    existe = subprocess.run(["git", "-C", str(raiz), "cat-file", "-e", f"{commit_c3}^{{commit}}"],
                            capture_output=True, text=True, timeout=30)
    if existe.returncode != 0:
        raise SystemExit(
            f"el commit de C3 en matrixAI ({commit_c3}) no existe en {raiz}: no se puede medir "
            f"qué cambió hasta el commit de C4 ({commit_c4}). No se compone el X/40")
    diff = subprocess.run(["git", "-C", str(raiz), "diff", "--name-only", commit_c3, commit_c4],
                          capture_output=True, text=True, timeout=30)
    if diff.returncode != 0:
        raise SystemExit(f"git diff --name-only {commit_c3} {commit_c4} en {raiz} falló "
                         f"(código {diff.returncode}): {diff.stderr.strip()}")
    ficheros = sorted(f for f in diff.stdout.splitlines() if f.strip())
    resultado_de_c3 = None
    no_admitidos = []
    for f in ficheros:
        if f == RUTA_RELATIVA_DEL_RESULTADO_C3:
            blob = _blob_en(raiz, commit_c4, f)
            resultado_de_c3 = {"ruta": f, "blob_en_el_commit_de_c4": blob,
                               "blob_esperado": BLOB_DEL_RESULTADO_C3_SELLADO,
                               "cuadra": blob == BLOB_DEL_RESULTADO_C3_SELLADO}
            if not resultado_de_c3["cuadra"]:
                no_admitidos.append(f)
        elif not _fichero_admitido_entre_c3_y_c4(f):
            no_admitidos.append(f)
    return {"raiz": str(raiz), "commit_c3": commit_c3, "commit_c4": commit_c4,
            "ficheros_cambiados": ficheros, "ficheros_no_admitidos": no_admitidos,
            "admitida": not no_admitidos, "motivo": None,
            "resultado_de_c3_en_el_commit_de_c4": resultado_de_c3,
            "por_que_se_admiten_estos": POR_QUE_SE_ADMITEN_ESTOS_FICHEROS}


def diferencia_del_entorno_entre_c3_y_c4(componentes_registrados_por_c3: dict,
                                         componentes_de_c3_ahora: dict) -> dict:
    """El refuerzo SIN GIT: qué componentes del digest del entorno de C3 son
    distintos hoy de los que C3 registró (`digest_de_la_cache.componentes`),
    y si todos caben en `COMPONENTES_DEL_ENTORNO_QUE_PUEDEN_DIFERIR_DE_C3`.
    Un componente que falta en un lado también es una diferencia."""
    claves = sorted(set(componentes_registrados_por_c3) | set(componentes_de_c3_ahora))
    distintos = [k for k in claves
                 if componentes_registrados_por_c3.get(k) != componentes_de_c3_ahora.get(k)]
    no_admitidos = [k for k in distintos if k not in COMPONENTES_DEL_ENTORNO_QUE_PUEDEN_DIFERIR_DE_C3]
    return {"n_componentes_c3": len(componentes_registrados_por_c3),
            "n_componentes_ahora": len(componentes_de_c3_ahora),
            "distintos": distintos, "no_admitidos": no_admitidos, "admitida": not no_admitidos,
            "que_puede_diferir": sorted(COMPONENTES_DEL_ENTORNO_QUE_PUEDEN_DIFERIR_DE_C3)}


def exigir_c3_sellado_y_anclado(procedencia_c4: dict, digest_motor_c4: str, *,
                                componentes_de_c3_ahora: dict) -> dict:
    """Lee `resultado_pasada_119_c3.json` del árbol, VERIFICA sus dos sellos
    y que es EL C3 medido (`DIGEST_RESULTADOS_CRUDOS_DE_C3`, una pasada
    completa y no un subconjunto), y exige que el motor esté anclado a lo
    mismo en C3 y en C4:

    1. **matrixai-engines**: el MISMO commit Y el MISMO digest del motor
       (sin cambio -- ahí SÍ es "otro motor" cualquier diferencia).
    2. **matrixAI**: puede ser OTRO commit -- la cola fija el HEAD
       committeado, y este guion tiene que estar commiteado para que la
       cola lo lleve -- pero solo si lo único que cambió entre el commit de
       C3 y el de C4 es este guion, la parametrización de C3, `tests/`,
       `documentacion/` o el resultado de C3 con su blob sellado
       (`diferencia_de_nucleo_entre_c3_y_c4`, medida con git). Un árbol
       SUCIO en cualquiera de las dos mediciones también PARA: no hay commit
       al que fiar el diff.
    3. **el entorno, sin git**: `componentes_de_c3_ahora` (los componentes
       del digest del entorno de C3 calculados HOY, en disco) iguales a los
       que C3 registró, salvo la parametrización de C3.

    `main()` la llama AL ARRANCAR, antes del primer intento (auditoría del
    guion, I1: antes solo al final, con los 8 sellados ya medidos), y otra
    vez al componer el X/40."""
    if not RUTA_DEL_RESULTADO_C3.exists():
        raise SystemExit(f"no está {RUTA_DEL_RESULTADO_C3}: sin C3 medido y sellado no hay con "
                         f"qué componer el X/40. No se mide")
    payload_c3 = c3p._leer_json(RUTA_DEL_RESULTADO_C3)
    if payload_c3.get("parcial") is not False:
        raise SystemExit(f"{RUTA_DEL_RESULTADO_C3} no es un resultado COMPLETO de C3 "
                         f"(parcial={payload_c3.get('parcial')!r}): no se compone el X/40 sobre "
                         f"una pasada de C3 a medias")
    if c3p.tipo_del_fichero(payload_c3) != "pasada" or payload_c3.get("es_subconjunto_de_prueba"):
        raise SystemExit(f"{RUTA_DEL_RESULTADO_C3} no es la PASADA real de C3 (tipo "
                         f"{c3p.tipo_del_fichero(payload_c3)!r}, es_subconjunto_de_prueba="
                         f"{payload_c3.get('es_subconjunto_de_prueba')!r}): no se compone el X/40")
    sellos = verificar_los_sellos_de_c3(payload_c3)
    if not sellos["cuadra"]:
        raise SystemExit("los sellos de C3 NO cuadran; no se mide:\n  - "
                         + "\n  - ".join(sellos["problemas"]))
    if payload_c3.get("digest_resultados_crudos") != DIGEST_RESULTADOS_CRUDOS_DE_C3:
        raise SystemExit(
            f"{RUTA_DEL_RESULTADO_C3} está sellado, pero NO es el C3 medido: su "
            f"digest_resultados_crudos es {str(payload_c3.get('digest_resultados_crudos'))[:16]}… "
            f"y el del C3 que subió a confirmación (e078fe6) es "
            f"{DIGEST_RESULTADOS_CRUDOS_DE_C3[:16]}… -- un C3 re-sellado no se suma")
    repos_c3 = (payload_c3.get("procedencia") or {}).get("repositorios") or {}
    repos_c4 = procedencia_c4.get("repositorios") or {}
    problemas = []

    # 1. matrixai-engines: MISMO commit, sin excepción.
    commit_c3_eng = (repos_c3.get("matrixai-engines") or {}).get("commit")
    commit_c4_eng = (repos_c4.get("matrixai-engines") or {}).get("commit")
    if commit_c3_eng is None or commit_c4_eng is None or commit_c3_eng != commit_c4_eng:
        problemas.append(f"matrixai-engines: C3 midió en {commit_c3_eng}, C4 en {commit_c4_eng}")
    digest_motor_c3 = ((payload_c3.get("digest_de_la_cache") or {}).get("motor_nuevo") or {}).get(
        "digest")
    if digest_motor_c3 != digest_motor_c4:
        problemas.append(f"digest del motor: C3 midió {digest_motor_c3}, C4 mide {digest_motor_c4}")

    # 2. matrixAI: puede ser otro commit, si el diff cabe en lo admitido.
    commit_c3_nucleo = (repos_c3.get("matrixAI") or {}).get("commit")
    commit_c4_nucleo = (repos_c4.get("matrixAI") or {}).get("commit")
    sucio_c3 = (repos_c3.get("matrixAI") or {}).get("arbol_sucio")
    sucio_c4 = (repos_c4.get("matrixAI") or {}).get("arbol_sucio")
    diferencia_nucleo = None
    if commit_c3_nucleo is None or commit_c4_nucleo is None:
        problemas.append(f"matrixAI: C3 midió en {commit_c3_nucleo}, C4 en {commit_c4_nucleo}")
    elif sucio_c3 or sucio_c4:
        problemas.append(
            f"matrixAI: árbol SUCIO al medir (C3 arbol_sucio={sucio_c3}, C4 "
            f"arbol_sucio={sucio_c4}) -- no hay commit al que fiar el diff entre C3 y C4")
    else:
        diferencia_nucleo = diferencia_de_nucleo_entre_c3_y_c4(
            _raiz_del_nucleo(), commit_c3_nucleo, commit_c4_nucleo)
        if not diferencia_nucleo["admitida"]:
            problemas.append(
                f"matrixAI: entre {commit_c3_nucleo} y {commit_c4_nucleo} cambiaron ficheros NO "
                f"admitidos: {diferencia_nucleo['ficheros_no_admitidos']}"
                + (f" (el resultado de C3 en el commit de C4 tiene el blob "
                   f"{diferencia_nucleo['resultado_de_c3_en_el_commit_de_c4']['blob_en_el_commit_de_c4']}"
                   f", no el sellado {BLOB_DEL_RESULTADO_C3_SELLADO})"
                   if (diferencia_nucleo.get("resultado_de_c3_en_el_commit_de_c4") or {}).get(
                       "cuadra") is False else ""))

    # 3. El entorno de C3, SIN git: componente a componente.
    componentes_c3 = (payload_c3.get("digest_de_la_cache") or {}).get("componentes")
    if not isinstance(componentes_c3, dict) or not componentes_c3:
        diferencia_entorno = None
        problemas.append("el resultado de C3 no registra `digest_de_la_cache.componentes`: no hay "
                         "con qué comparar el entorno de hoy")
    else:
        diferencia_entorno = diferencia_del_entorno_entre_c3_y_c4(componentes_c3,
                                                                  componentes_de_c3_ahora)
        if not diferencia_entorno["admitida"]:
            problemas.append(
                f"entorno: {len(diferencia_entorno['no_admitidos'])} componentes del digest de C3 "
                f"son distintos hoy (solo puede diferir "
                f"{diferencia_entorno['que_puede_diferir']}): "
                f"{diferencia_entorno['no_admitidos'][:20]}")

    if problemas:
        raise SystemExit(
            f"el motor NO está anclado a lo mismo en C3 y en C4: no se suman dos motores "
            f"distintos (encargo de 119-C4). Diferencias:\n  - " + "\n  - ".join(problemas))
    return {"payload_c3": payload_c3, "sellos": sellos, "diferencia_de_nucleo": diferencia_nucleo,
            "diferencia_del_entorno": diferencia_entorno}


def cumplidos_de_los_sellados(*, resultados_v2_todos: list[dict], resultados_c4: list[dict],
                              esperadas_por_conjunto: dict, regla, metrica_por_dataset: dict,
                              nombres_sellados: list[str]) -> tuple[dict, dict]:
    """Los cumplidos del motor nuevo y de la densa v2 en los 8 SELLADOS, con
    LA MISMA FUNCIÓN que usa `veredicto_final` de C3 para sus 32
    (`c3p.cumplidos_de_los_dos_motores`: `p118.campo_de_la_comparacion` +
    `c3p._cumplidos_de`, más los fallos de la enmienda 2 vía
    `fallos_del_motor`/`registros_de_los_fallos`). Hasta la auditoría del
    guion (M6) esto era una COPIA de ese cuerpo; ahora es una llamada."""
    return c3p.cumplidos_de_los_dos_motores(
        resultados_v2=resultados_v2_todos, resultados_del_motor_nuevo=resultados_c4,
        esperadas_por_conjunto=esperadas_por_conjunto, regla=regla,
        metrica_por_dataset=metrica_por_dataset, nombres_de_los_conjuntos=list(nombres_sellados))


def decision_segun_d2(x: int, n: int) -> dict:
    """D2 del contrato 119 (decisión de Roberto). El listón de la cartera
    (32/40) SÍ es una medida (80% de `regla_de_cierre`); el techo de los
    retoques (~27/40) es la ESTIMACIÓN del análisis del 24-09 y se declara
    como tal SIEMPRE junto al número (CLAUDE.md: «un techo no es una
    medida»)."""
    if n != 40:
        return {"decision": "incompleta", "motivo": f"{n} de 40 conjuntos: D2 exige los 40"}
    if x >= UMBRAL_CARTERA_D2:
        decision = "entra_en_la_cartera"
    elif x > TECHO_DE_LOS_RETOQUES_ESTIMADO_D2:
        decision = "avance_publicado_sin_cartera"
    else:
        decision = "se_cierra_como_el_118"
    return {"decision": decision, "x": x, "n": n,
            "frontera_27": (
                "D2 no define x = 27 exacto («por encima de los ~27/40» y «por debajo de 27»): "
                "aquí 27 cuenta como se_cierra_como_el_118 (no está POR ENCIMA del techo). "
                "Inalcanzable en 119-C4: C3 ya dio 31, así que X >= 31. Si llegara a importar, "
                "lo decide Roberto, no este guion (auditoría del guion, M1)"),
            "umbral_de_la_cartera": {"valor": UMBRAL_CARTERA_D2, "medido": True,
                                     "fuente": "regla_de_cierre: 80% de 40 (protocolo_119_v4.json)"},
            "techo_de_los_retoques": {"valor": TECHO_DE_LOS_RETOQUES_ESTIMADO_D2, "medido": False,
                                      "nota": "ESTIMADO, no medido (análisis del 24-09; CLAUDE.md: "
                                              "«un techo no es una medida»)"},
            "fuente": "119_LA_RED_DENSA_NUEVA_CONTRACT.md, decisión D2 de Roberto"}


def x_de_40(*, cumplidos_c3: dict, cumplidos_c4: dict, densa_v2_c3: dict,
           densa_v2_c4: dict) -> dict:
    """El X/40 = los cumplidos de C3 (31/32, YA sellados y verificados) + los
    cumplidos de C4 (y/8, medidos aquí), PURA (se prueba con diccionarios
    fabricados donde el resultado se sabe a mano)."""
    x = cumplidos_c3["cumplidos"] + cumplidos_c4["cumplidos"]
    n = cumplidos_c3["datasets"] + cumplidos_c4["datasets"]
    densa_v2_x = densa_v2_c3["cumplidos"] + densa_v2_c4["cumplidos"]
    densa_v2_n = densa_v2_c3["datasets"] + densa_v2_c4["datasets"]
    return {
        "x_de_40": x, "n_conjuntos": n, "fraccion": (x / n) if n else None,
        "reparto": {"c3_no_sellados": dict(cumplidos_c3), "c4_sellados": dict(cumplidos_c4)},
        "densa_v2_en_los_mismos_conjuntos": {
            "cumplidos": densa_v2_x, "n_conjuntos": densa_v2_n,
            "fraccion": (densa_v2_x / densa_v2_n) if densa_v2_n else None,
            "reparto": {"c3_no_sellados": dict(densa_v2_c3), "c4_sellados": dict(densa_v2_c4)}},
        "decision_segun_d2": decision_segun_d2(x, n),
        "como_se_compone": (
            f"los cumplidos de C3 en sus {cumplidos_c3['datasets']} no sellados (leídos de su "
            f"resultado YA SELLADO, con sus dos digests re-verificados aquí antes de sumar nada) "
            f"más los cumplidos de C4 en sus {cumplidos_c4['datasets']} sellados, con la MISMA "
            f"función (`cumplidos_de_los_dos_motores` de 119-C3, la de su `veredicto_final`) y el "
            f"mismo campo (la v2 con la densa sustituida por el motor nuevo SOLO en los conjuntos "
            f"medidos)"),
        "como_se_cuentan_los_fallos": COMO_SE_CUENTAN_LOS_FALLOS_EN_C4,
    }


def componer_x_de_40(*, resultados_v2: list[dict], resultados_c4: list[dict],
                     esperadas_por_conjunto: dict, regla, metrica_por_dataset: dict,
                     nombres_sellados: list[str], procedencia_c4: dict,
                     digest_motor_c4: str, componentes_de_c3_ahora: dict) -> dict:
    """El X/40, con el anclaje comprobado OTRA VEZ (ya se comprobó al
    arrancar): `componentes_de_c3_ahora` se recalcula al acabar, así que un
    fichero del entorno editado A MITAD de la pasada también para aquí."""
    anclaje = exigir_c3_sellado_y_anclado(procedencia_c4, digest_motor_c4,
                                          componentes_de_c3_ahora=componentes_de_c3_ahora)
    payload_c3 = anclaje["payload_c3"]
    cumplidos_c4, densa_v2_c4 = cumplidos_de_los_sellados(
        resultados_v2_todos=resultados_v2, resultados_c4=resultados_c4,
        esperadas_por_conjunto=esperadas_por_conjunto, regla=regla,
        metrica_por_dataset=metrica_por_dataset, nombres_sellados=nombres_sellados)
    cumplidos_c3 = payload_c3["veredicto"]["cumplidos_con_el_motor_nuevo"]
    densa_v2_c3 = payload_c3["veredicto"]["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"]
    return {
        **x_de_40(cumplidos_c3=cumplidos_c3, cumplidos_c4=cumplidos_c4,
                 densa_v2_c3=densa_v2_c3, densa_v2_c4=densa_v2_c4),
        "sellos_de_c3_verificados": anclaje["sellos"],
        "commits_de_c3": (payload_c3.get("procedencia") or {}).get("repositorios"),
        "commits_de_c4": procedencia_c4.get("repositorios"),
        "diferencia_de_nucleo_entre_c3_y_c4": anclaje["diferencia_de_nucleo"],
        "diferencia_del_entorno_entre_c3_y_c4": anclaje["diferencia_del_entorno"],
    }


# ---------------------------------------------------------------------------
# 3b. EL DIGEST DE LA CACHÉ DE C4: el de C3 MÁS este guion (I6)
# ---------------------------------------------------------------------------

def ficheros_que_c4_anade_al_entorno() -> tuple[Path, ...]:
    """Lo que el cierre ESTÁTICO de imports de ESTE guion (el mismo
    `_cierre_de_imports` de C3) tiene y el entorno de C3 no: medido, no
    escrito a mano. Hoy, `pasada_119_c4.py` y nada más -- y lleva el bucle de
    medida, así que tiene que invalidar la caché (auditoría del guion, I6:
    antes el artefacto se contradecía, listándolo como «fuera del digest»
    junto a su propio «tiene que estar vacía»)."""
    cierre, _ = c3p._cierre_de_imports(Path(__file__).resolve(), c3p._raices_de_import())
    return tuple(sorted(set(cierre) - set(c3p.ficheros_del_entorno())))


def componentes_del_digest_del_entorno_c4() -> dict[str, str]:
    """`c3p.componentes_del_digest_del_entorno()` MÁS los ficheros que C4
    añade. El anclaje a C3 compara los de C3 a secas (sin estos)."""
    componentes = c3p.componentes_del_digest_del_entorno()
    for f in ficheros_que_c4_anade_al_entorno():
        componentes[c3p._etiqueta(f)] = c3p.c3._digest_fichero(f)
    return componentes


def lo_que_no_cubre_el_digest_c4() -> dict:
    """`c3p.lo_que_no_cubre_el_digest()` con la lista de módulos fuera del
    digest calculada contra el entorno de C4 (que ya cubre este guion)."""
    salida = dict(c3p.lo_que_no_cubre_el_digest())
    anadidos = {c3p._etiqueta(f) for f in ficheros_que_c4_anade_al_entorno()}
    clave = "modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest"
    salida[clave] = [m for m in salida.get(clave) or [] if m not in anadidos]
    salida["que_anade_c4"] = (
        f"el entorno de C4 es el de C3 más el cierre de imports de pasada_119_c4.py que C3 no "
        f"tiene: {sorted(anadidos)}")
    return salida


# ---------------------------------------------------------------------------
# 4. --estimar: SIN TOCAR LOS SELLADOS CON EL MOTOR REAL
# ---------------------------------------------------------------------------

def medidas_desde_c3(payload_c3: dict) -> dict[str, dict]:
    """Los tiempos MEDIDOS de C3 por conjunto (wall_s medio, preparacion_s
    medio, carga_s del padre, celdas, cubo), leídos de su resultado YA
    SELLADO. `estimar_desde_medidas` (C3, reutilizada TAL CUAL) interpola
    con esto por celdas dentro de cada cubo para los 8 sellados: nunca se
    miden con el motor real.

    La preparación sale de los REGISTROS (cada intento de C3 trae su
    `preparacion_s`, medido), no de `tiempos_del_padre_por_dataset`: esos son
    los de la CONTINUACIÓN, que solo preparó Allstate (31 de 32 con
    `n_preparaciones: 0`), y antes de la auditoría del guion (M2) entraban
    aquí como un 0,0 que parecía medido. Un dato AUSENTE es `None`, nunca un
    cero; `cmd_estimar_c4` deja fuera de la cuenta lo que venga en `None`,
    diciéndolo."""
    resultados = payload_c3.get("resultados") or []
    particiones = payload_c3.get("particion_por_dataset") or {}
    tiempos_padre = payload_c3.get("tiempos_del_padre_por_dataset") or {}
    por_dataset: dict[str, list[dict]] = {}
    for r in resultados:
        if r.get("motor") == c3p.NOMBRE_MOTOR_NUEVO and r.get("wall_s") is not None:
            por_dataset.setdefault(r["dataset"], []).append(r)
    medidas = {}
    for nombre, registros in por_dataset.items():
        particion = particiones.get(nombre) or {}
        tiempos = tiempos_padre.get(nombre) or {}
        preparaciones = [r["preparacion_s"] for r in registros if r.get("preparacion_s") is not None]
        wall_media = sum(r["wall_s"] for r in registros) / len(registros)
        filas, predictores = particion.get("n_filas_con_objetivo"), particion.get("n_predictores")
        medidas[nombre] = {
            "cubo": registros[0]["cubo"],
            "celdas": (filas * predictores) if filas is not None and predictores is not None
                      else None,
            "wall_s": wall_media, "carga_s": tiempos.get("carga_s"),
            "preparacion_s": (sum(preparaciones) / len(preparaciones)) if preparaciones else None,
            "estado": "completed",
            "n_intentos_medidos_en_c3": len(registros),
            "n_intentos_con_preparacion_medida": len(preparaciones),
        }
    return medidas


def avisos_de_encolado() -> list[str]:
    """Por qué la orden de encolado que imprime `--estimar` NO serviría tal
    cual (auditoría del guion, M3: corrido antes de commitear, fijaba
    `COMMITS=matrixAI=<un commit sin este guion>` y solo lo decía el JSON).
    La cola saca un worktree del commit fijado: lo que no esté commiteado no
    viaja. Vacía = la orden sirve."""
    avisos = []
    raiz = _raiz_del_nucleo()
    if _blob_en(raiz, "HEAD", GUION_DE_C4) is None:
        avisos.append(f"el HEAD de matrixAI ({raiz}) NO contiene {GUION_DE_C4}: la cola sacaría "
                      f"un worktree sin este guion y el trabajo fallaría al arrancar. Commitear "
                      f"y volver a estimar")
    for nombre, raiz_repo in c3p.c3._RUTAS_DE_REPOSITORIO.items():
        try:
            estado = subprocess.run(["git", "-C", str(raiz_repo), "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            avisos.append(f"{nombre}: git no responde ({type(exc).__name__}): no se sabe si el "
                          f"árbol está limpio")
            continue
        lineas = [l for l in estado.stdout.splitlines() if l.strip()]
        if estado.returncode != 0:
            avisos.append(f"{nombre}: git status falló ({estado.stderr.strip()[:200]})")
        elif lineas:
            avisos.append(f"{nombre}: árbol SUCIO ({len(lineas)} ficheros sin commitear o sin "
                          f"seguimiento, p. ej. {lineas[:6]}): la cola NO lleva nada de eso, así "
                          f"que lo estimado no es lo que correría")
    return avisos


def cmd_estimar_c4(protocolo, sellados: list, *, ruta_salida: Path, conjuntos: str | None,
                   protocolos: dict, configuracion_del_motor: dict, enmienda_2: dict) -> None:
    c3p._exigir_que_quepa()
    if ruta_salida.exists() and c3p.tipo_del_fichero(c3p._leer_json(ruta_salida)) != "estimar":
        raise SystemExit(f"{ruta_salida} no es una estimación: --estimar no lo pisa")
    if not RUTA_DEL_RESULTADO_C3.exists():
        raise SystemExit(f"no está {RUTA_DEL_RESULTADO_C3}: sin C3 medido no hay tiempos de los "
                         f"que estimar C4 (--estimar nunca mide un sellado con el motor real)")
    payload_c3 = c3p._leer_json(RUTA_DEL_RESULTADO_C3)
    sellos = verificar_los_sellos_de_c3(payload_c3)
    if not sellos["cuadra"]:
        raise SystemExit("los sellos de C3 NO cuadran; no se estima sobre un resultado que "
                         "pudo haberse tocado:\n  - " + "\n  - ".join(sellos["problemas"]))
    medidas_todas = medidas_desde_c3(payload_c3)
    medidas = {n: m for n, m in medidas_todas.items()
               if all(m[k] is not None for k in ("celdas", "wall_s", "carga_s", "preparacion_s"))}
    fuera_de_la_cuenta = sorted(set(medidas_todas) - set(medidas))
    payload_v2 = c3p._payload_v2()
    nombres = ([n.strip() for n in conjuntos.split(",") if n.strip()] if conjuntos
              else [d.nombre for d in sellados])
    elegidos = [d for d in sellados if d.nombre in nombres]
    desconocidos = sorted(set(nombres) - {d.nombre for d in elegidos})
    if desconocidos:
        raise SystemExit(f"--estimar-conjuntos nombra {desconocidos}, que no son de los 8 "
                         f"sellados")
    conjuntos_desc = c3p._conjuntos_para_estimar(elegidos, protocolo, payload_v2)
    ahora = time.localtime()
    fin_de_la_suite = c3p.fin_de_la_suite_nocturna_medido()
    estimacion = c3p.estimar_desde_medidas(
        medidas, conjuntos_desc, margen_s=MARGEN_POR_DEFECTO_SEGUNDOS,
        ventana_nocturna_s=(c3p._segundos_del_dia(c3p.VENTANA_NOCTURNA[1])
                            - c3p._segundos_del_dia(fin_de_la_suite["desde"])),
        ventana_nocturna_desde=fin_de_la_suite["desde"],
        ventana_de_dia_s=c3p._segundos_entre(c3p.HORA_DE_INICIO_DE_DIA, c3p.HORA_LIMITE_DE_DIA),
        ventana_de_dia_desde_ahora_s=c3p._segundos_hasta(c3p.HORA_LIMITE_DE_DIA, ahora))
    picos_de_c3 = [r.get("rss_pico_mb") for r in payload_c3.get("resultados", [])
                  if r.get("motor") == c3p.NOMBRE_MOTOR_NUEVO]
    picos_con_dato = [p for p in picos_de_c3 if p is not None]
    pico_mb = max(picos_con_dato) if picos_con_dato else None  # ausente: null, no 0 (M2)
    memory_max = (f"{max(MEMORIA_MINIMA_SUGERIDA_GB, math.ceil(pico_mb * 1.3 / 1024))}G"
                  if pico_mb is not None else f"{MEMORIA_MINIMA_SUGERIDA_GB}G")
    aviso_memoria = (
        None if picos_con_dato else
        f"MEDIDO (2026-09-30): ninguno de los {len(picos_de_c3)} registros del motor nuevo en "
        f"resultado_pasada_119_c3.json trae `rss_pico_mb` (el campo existe, pero está en "
        f"`null` en todos) -- `ejecutar_intento_aislado` no lo rellenó en esa pasada. "
        f"`memory_max_sugerido` es por tanto el SUELO ya probado ({MEMORIA_MINIMA_SUGERIDA_GB}G), "
        f"no una cota derivada de un pico medido aquí -- {FUENTE_DE_LA_MEMORIA_MINIMA}")
    salida = {
        "corte": "119-C4", "sub_corte": "estimacion (--estimar: SIN tocar los sellados con el "
                                        "motor real)",
        "tipo_de_ejecucion": "estimar",
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motor": c3p.NOMBRE_MOTOR_NUEVO, "configuracion_del_motor": configuracion_del_motor,
        "protocolos_encadenados": protocolos, "enmienda_2": enmienda_2,
        "sellos_de_c3_verificados": sellos,
        "conjuntos_estimados": [d.nombre for d in elegidos],
        "medidas_de_c3_usadas": medidas,
        "estimacion": estimacion,
        "de_donde_salen_las_medidas": (
            "los tiempos YA MEDIDOS de C3 (resultado_pasada_119_c3.json, sellado y verificado "
            "aquí) por conjunto -- wall_s medio, carga_s y preparacion_s del padre -- "
            "interpolados por celdas dentro de cada cubo con estimar_desde_medidas (119-C3, "
            "reutilizada TAL CUAL). NUNCA un intento real sobre un sellado: 'un sellado se mide "
            "una vez, y esa vez es la pasada real' (encargo de 119-C4). La preparación, de los "
            "REGISTROS de C3 (cada intento trae la suya), no del padre de la continuación"),
        "medidas_de_c3_fuera_de_la_cuenta": fuera_de_la_cuenta,
        "memoria": {
            "pico_mb_medido_en_c3": pico_mb,
            "que_es": ("el máximo rss_pico_mb del motor nuevo YA MEDIDO en los 32 no sellados de "
                      "C3, con margen ×1,3 sobre el suelo probado -- no una medición nueva sobre "
                      "los sellados"),
            "suelo_gb": MEMORIA_MINIMA_SUGERIDA_GB, "de_donde_sale_el_suelo": FUENTE_DE_LA_MEMORIA_MINIMA,
            "memory_max_sugerido": memory_max, "aviso": aviso_memoria},
        "para_encolar": c3p.para_encolar(
            estimacion["total_horas"] * 3600, memoria=memory_max,
            commits=c3p._commits_de_ahora(), fin_de_la_suite=fin_de_la_suite,
            nombre_del_trabajo=NOMBRE_DEL_TRABAJO_EN_LA_COLA, guion=GUION_DE_C4,
            salida_en_la_cola=str(SALIDA_DE_C4_EN_LA_COLA)),
        "avisos_de_encolado": avisos_de_encolado(),
    }
    c3p._escribir_atomicamente(ruta_salida, salida)
    print(f"estimación escrita en {ruta_salida}")
    print(f"  1 proceso: {estimacion['total_horas']:.2f} h (cota {estimacion['total_horas_cota']:.2f} h)")
    print(f"  memoria (de C3, no medida aquí): pico "
          f"{'SIN DATO' if pico_mb is None else f'{pico_mb:.0f} MB'} -> MemoryMax {memory_max}"
          + (f" [SIN rss_pico_mb EN C3: {MEMORIA_MINIMA_SUGERIDA_GB}G es el suelo YA PROBADO por "
             f"C3, ver 'aviso']" if aviso_memoria else ""))
    if fuera_de_la_cuenta:
        print(f"  medidas de C3 SIN dato completo, fuera de la cuenta: {fuera_de_la_cuenta}")
    encolar = salida["para_encolar"]
    dia = encolar["de_dia"]
    if dia["cabe"]:
        print(f"  DE DÍA (recomendada), lanzando antes de las {dia['lanzar_antes_de']}:\n"
              f"    {dia['encolar']}\n    {dia['lanzar_la_cola']}")
    else:
        print("  DE DÍA NO CABE")
    for aviso in salida["avisos_de_encolado"]:
        print(f"  AVISO -- NO ENCOLAR ESTA ORDEN TAL CUAL: {aviso}", flush=True)


# ---------------------------------------------------------------------------
# 5. LA CONFIRMACIÓN (main)
# ---------------------------------------------------------------------------

def _argumentos(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON (admite una absoluta fuera del arbol). Por omision: "
                             + ", ".join(f"{t}={p.name}" for t, p in SALIDA_POR_OMISION_C4.items()))
    parser.add_argument("--estimar", action="store_true",
                        help="NO mide ningun sellado: estima con los tiempos YA MEDIDOS de C3")
    parser.add_argument("--estimar-conjuntos", default=None,
                        help="con --estimar, los sellados que se listan (coma); por omision, "
                             "los 8")
    parser.add_argument("--solo", default=None,
                        help="SOLO PARA LAS PRUEBAS, con un motor falso: con el motor real PARA "
                             "(un sellado se mide una vez, y esa vez es la pasada real). "
                             "Nombres de conjunto separados por coma, SIEMPRE dentro de los 8 "
                             "sellados (un no sellado se niega); escribe en un fichero aparte")
    args = parser.parse_args(argv)
    if args.estimar and args.solo:
        raise SystemExit("--estimar no se combina con --solo (usa --estimar-conjuntos)")
    return args


def main(argv=None) -> None:
    args = _argumentos(argv)
    tipo = c3p.tipo_de_ejecucion(humo=False, solo=args.solo, estimar=args.estimar)
    if tipo == "solo" and ejecutar_intento_aislado is _EJECUTAR_INTENTO_AISLADO_REAL:
        # ANTES de cargar nada (auditoría del guion, I2): `--solo` seguía el MISMO camino que la
        # pasada, con el motor real, y el docstring lo proponía sobre kr-vs-kp.
        raise SystemExit(
            "--solo NO corre con el motor real: los únicos conjuntos que admite son SELLADOS, y "
            "un sellado se mide UNA vez, en la pasada real (protocolo_119_v4.json, "
            "veredicto.confirmacion). --solo existe para que las pruebas recorran el guion con un "
            "motor falso; para medir, la pasada entera (sin --solo), por la cola")
    ruta_salida = ruta_de_salida_c4(tipo, args.salida)

    c3p.c6.preparar_protocolo_v2()
    protocolos = c3p.exigir_los_protocolos_encadenados()
    configuracion_del_motor = c3p.exigir_la_configuracion_de_la_enmienda()
    # ANTES de cargar nada: la pasada real no arranca sin la enmienda 2 en regla (misma
    # guardia que C3; --solo/--estimar corren sin ella, diciéndolo).
    enmienda_2 = c3p.exigir_la_enmienda_2(tipo)
    protocolo = c3p.c3.protocolo_registrado()
    todos = c3p.c5.datasets_de_la_pasada(protocolo)
    sellados = [d for d in todos if d.sellado]

    if tipo == "estimar":
        cmd_estimar_c4(protocolo, sellados, ruta_salida=ruta_salida,
                       conjuntos=args.estimar_conjuntos, protocolos=protocolos,
                       configuracion_del_motor=configuracion_del_motor, enmienda_2=enmienda_2)
        return

    if tipo == "pasada":
        exigir_una_sola_ejecucion(ruta_salida)

    datasets, subconjunto = datasets_de_c4(todos, sellados, solo=args.solo)
    if not datasets:
        raise SystemExit("no queda ningún conjunto que medir")

    c3p._exigir_que_quepa()
    motor = c3p.MotorDensaTabM()
    metrica_por_dataset = c3p.c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c3p.c6.direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    payload_v2 = c3p._payload_v2()
    resultados_v2 = payload_v2["resultados"]
    sin_particion_v2 = [d.nombre for d in datasets
                        if d.nombre not in (payload_v2.get("particion_por_dataset") or {})]
    if sin_particion_v2:
        raise SystemExit(f"la v2 no registra la partición de {sin_particion_v2}: no hay con qué "
                         f"comparar. No se mide")

    plan = c3p.c6.plan_de_la_pasada(datasets, protocolo, n_motores=1)
    print(f"119-C4: protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]} "
          f"(v4 {protocolos['v4']['digest_sha256_declarado'][:16]})")
    print(f"motor {c3p.NOMBRE_MOTOR_NUEVO}: {configuracion_del_motor['leida_del_motor']} "
          f"(la de la enmienda 1). Salida ({tipo}): {ruta_salida}")
    c3p._imprimir_estimacion_peor_caso(plan)

    cache_previo, payload_previo = c3p.cargar_salida_previa(ruta_salida, tipo)
    previos = list(payload_previo.get("resultados") or [])

    # I6: el digest de la caché de C4 es el de C3 MÁS este guion (lleva el bucle de medida).
    componentes_del_digest = componentes_del_digest_del_entorno_c4()
    entorno_digest = c3p._digest_de_componentes(componentes_del_digest)
    digest_del_motor = c3p._digest_motor_nuevo()
    procedencia = c3p.c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor_nuevo": digest_del_motor},
        datos_de_entrada={
            **{d.nombre: c3p.c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
            "protocolo_exploratorio_v2": c3p.RUTA_DEL_PROTOCOLO_V2,
            "protocolo_119_v4": c3p.RUTA_DEL_PROTOCOLO_V4,
            "protocolo_119_v4_enmienda_1": c3p.RUTA_DE_LA_ENMIENDA_1,
            "protocolo_119_v4_enmienda_2": c3p.RUTA_DE_LA_ENMIENDA_2,
            "pasada_v2_113_resultado": c3p.RUTA_V2_RESULTADO,
            "resultado_pasada_119_c3": RUTA_DEL_RESULTADO_C3,
        })
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = c3p.c3.procedencia_declarada(payload_previo)
        print(f"fichero previo: {len(previos)} registros, procedencia "
              f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)

    # I1: el anclaje ENTERO, AL ARRANCAR y antes del primer intento (y otra vez al componer el
    # X/40). Antes solo al final: un desanclaje se descubría con los 8 sellados ya medidos.
    if tipo == "pasada":
        exigir_c3_sellado_y_anclado(procedencia, digest_del_motor,
                                    componentes_de_c3_ahora=c3p.componentes_del_digest_del_entorno())
        print("anclaje a C3 comprobado AL ARRANCAR: sellos, C3 exacto, motores y digest del motor, "
              "árbol limpio, diff del núcleo y entorno", flush=True)
    # I4: lo ya medido de un parcial se REUSA entero (fallos incluidos) o la pasada PARA -- nunca
    # se vuelve a medir un sellado. Aquí sobre todos los previos; abajo, intento a intento.
    exigir_que_lo_previo_sea_reusable(
        previos, nombres={d.nombre for d in datasets}, entorno_digest=entorno_digest,
        motor_digest=digest_del_motor,
        datos_sha256_por_conjunto={d.nombre: (procedencia["datos_de_entrada"].get(d.nombre) or {})
                                   .get("sha256") for d in datasets})

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

    def guardar(parcial: bool) -> dict:
        veredicto_x_de_40 = None
        if not parcial and tipo == "pasada":
            veredicto_x_de_40 = componer_x_de_40(
                resultados_v2=resultados_v2, resultados_c4=resultados,
                esperadas_por_conjunto=esperadas_por_conjunto, regla=regla,
                metrica_por_dataset=metrica_por_dataset, nombres_sellados=nombres_de_los_conjuntos,
                procedencia_c4=procedencia, digest_motor_c4=digest_del_motor,
                componentes_de_c3_ahora=c3p.componentes_del_digest_del_entorno())
        return componer_y_guardar_c4(
            resultados, previos, veredictos_por_conjunto, procedencia, payload_previo,
            ruta_salida, tipo=tipo, datasets=datasets, protocolo=protocolo,
            metrica_por_dataset=metrica_por_dataset, subconjunto=subconjunto, plan=plan,
            particiones=particiones, comparacion_de_particiones=comparacion_de_particiones,
            pliegues_digest=pliegues_digest, tiempos_del_padre=tiempos_del_padre,
            esperadas_por_conjunto=esperadas_por_conjunto, protocolos=protocolos,
            configuracion_del_motor=configuracion_del_motor, enmienda_2=enmienda_2,
            componentes_del_digest=componentes_del_digest,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados[0], parcial=parcial,
            veredicto_x_de_40=veredicto_x_de_40)

    for ds in datasets:
        t_carga = time.perf_counter()
        por_id, propuesta, spec, objetivo, predictores, particion = c3p.c5.particiones_base(
            ds, protocolo)
        carga_s = time.perf_counter() - t_carga
        comparacion_de_particiones[ds.nombre] = c3p.exigir_la_particion_de_la_v2(
            ds, particion, c3p.c3.LECTURA_DECLARADA.get(str(ds.data_id)), payload_v2)
        particiones[ds.nombre] = c3p._normalizado(particion)
        pliegues_digest[ds.nombre] = propuesta.pliegues.digest()
        test_ids = propuesta.plan.observaciones_del_rol("test")
        metric_id = metrica_por_dataset[ds.nombre]
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        intentos = c3p.plan_de_intentos(ds, propuesta, protocolo, humo=False)
        esperadas_por_conjunto[ds.nombre] = [(i.repeticion, i.pliegue) for i in intentos]
        tiempos = tiempos_del_padre[ds.nombre] = {"carga_s": round(carga_s, 3),
                                                  "preparacion_s": 0.0, "n_preparaciones": 0}
        print(f"\n=== {ds.nombre} (SELLADO, data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"n={len(por_id)}, test={len(test_ids)}, intentos={len(intentos)}, "
              f"metrica={metric_id}, carga={carga_s:.1f}s) ===", flush=True)

        registros_del_dataset: list[dict] = []
        for repeticion in sorted({i.repeticion for i in intentos}):
            for ip in [i for i in intentos if i.repeticion == repeticion]:
                clave = (ds.nombre, motor.nombre, ip.repeticion, ip.pliegue)
                previo = cache_previo.get(clave)
                if decidir_con_el_previo(previo, entorno_digest=entorno_digest,
                                         motor_digest=digest_del_motor,
                                         wall_seconds=ip.presupuesto_wall_s,
                                         datos_sha256=datos_sha256) == "reusar":
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
                transformadas = c3p.c3.preparar_para_motor(
                    crudas_train, crudas_train + crudas_val + crudas_test, objetivo, predictores,
                    motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)

                def hacer(xs):
                    return c3p.Particion.desde_filas(xs, row_id_field="row_id",
                                                     target_field=objetivo)

                train, val, test = (hacer(transformadas[:n_tr]),
                                    hacer(transformadas[n_tr:n_tr + n_va]),
                                    hacer(transformadas[n_tr + n_va:]))
                preparacion_s = time.perf_counter() - t_prep
                tiempos["preparacion_s"] = round(tiempos["preparacion_s"] + preparacion_s, 3)
                tiempos["n_preparaciones"] += 1

                presupuesto = c3p.Presupuesto(wall_seconds=ip.presupuesto_wall_s,
                                              hilos=c3p.c3.HILOS_POR_INTENTO, seed=ip.semilla)
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto,
                    candidate=f"{motor.nombre}-{c3p.c3.CONFIGURACION_UNICA}",
                    split_plan_digest=propuesta.plan.digest(), dataset=ds.nombre,
                    pliegue=ip.pliegue, repeticion=ip.repeticion)
                transcurrido = time.perf_counter() - t0

                metricas = c3p.c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                config = intento.config_efectiva or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                    "tarea": ds.tarea, "sellado": ds.sellado, "motor": motor.nombre,
                    "repeticion": ip.repeticion, "pliegue": ip.pliegue, "estado": intento.estado,
                    "semilla": ip.semilla, "presupuesto_wall_s": ip.presupuesto_wall_s,
                    "configuracion": c3p.c3.CONFIGURACION_UNICA, "metrica_de_cierre": metric_id,
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
                c3p.c5.aplanar_metricas_en_el_registro(registro, metricas)
                resultados.append(registro)
                registros_del_dataset.append(registro)
                # M5 (auditoría del guion): punto de control tras CADA intento, ANTES de
                # imprimir su métrica. Con el de C3 (por repetición o cada 60 s) un corte
                # perdía un intento sellado ya visto en el registro, que se re-medía al
                # reanudar. Aquí un intento ejecutado queda escrito antes de verse.
                guardar(parcial=True)
                c3p.exigir_la_arquitectura_del_intento(
                    registro, receta=configuracion_del_motor["receta_del_protocolo"],
                    n_train=n_tr)
                print(f"  {ds.nombre} rep={ip.repeticion} pliegue={ip.pliegue}: "
                      f"estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"wall={transcurrido:.1f}s prep={preparacion_s:.1f}s", flush=True)

        v2_densa_del_dataset = [r for r in resultados_v2
                                if r["dataset"] == ds.nombre and r["motor"] == c3p.NOMBRE_DENSA_V2]
        veredicto_ds = c3p.veredicto_del_conjunto_c3(
            dataset=ds.nombre, metric_id=metric_id, registros_del_motor_nuevo=registros_del_dataset,
            registros_de_la_densa_v2=v2_densa_del_dataset,
            esperadas=esperadas_por_conjunto[ds.nombre])
        veredictos_por_conjunto.append(veredicto_ds)
        print(f"  VEREDICTO {ds.nombre} (SELLADO): mejora={veredicto_ds['mejora']} "
              f"inferioridad={veredicto_ds['inferioridad']} "
              f"sin_comparacion={veredicto_ds['sin_comparacion']} "
              f"fallos_del_nuevo={veredicto_ds['fallos_del_motor_nuevo']['n']}", flush=True)
        guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados[0]} reusados, {len(resultados)-reusados[0]} ejecutados) ===")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    if tipo == "pasada":
        x40 = salida["veredicto_x_de_40"]
        c3_, c4_ = x40["reparto"]["c3_no_sellados"], x40["reparto"]["c4_sellados"]
        print(f"VEREDICTO 119-C4 (CONFIRMACIÓN): X/40 = {x40['x_de_40']}/{x40['n_conjuntos']} "
              f"({c3_['cumplidos']}/{c3_['datasets']} de C3 + "
              f"{c4_['cumplidos']}/{c4_['datasets']} de C4) -> "
              f"{x40['decision_segun_d2']['decision']}")


def componer_y_guardar_c4(resultados, previos, veredictos_por_conjunto, procedencia,
                          payload_previo, ruta_salida, *, tipo, datasets, protocolo,
                          metrica_por_dataset, subconjunto, plan, particiones,
                          comparacion_de_particiones, pliegues_digest, tiempos_del_padre,
                          esperadas_por_conjunto, protocolos, configuracion_del_motor, enmienda_2,
                          componentes_del_digest, total_wall_s, reusados, parcial,
                          veredicto_x_de_40) -> dict:
    """Compone el JSON de C4 y lo escribe atómicamente, FUSIONADO con lo que
    ya había -- espejo de `c3p._componer_y_guardar`, reutilizando sus mismas
    piezas (`fusionar_resultados`, `rastro_de_los_reintentos`,
    `_procedencias_citadas`, `intentos_por_conjunto`,
    `reconciliar_el_plan_con_lo_medido`, `sellar_la_salida`), pero SIN los
    campos narrativos propios de C3 (Allstate, «32 NO sellados»...) y CON el
    X/40, que C3 no tiene."""
    en_el_fichero, conservados, sustituidos_sin_medida = c3p.fusionar_resultados(previos, resultados)
    # En C4 un fallido previo se REUSA (es la medida): el registro reusado ES el previo, no lo
    # sustituye, así que no es un reintento (sin esto, `rastro_de_los_reintentos` lo anotaría
    # como reintentado). Un reintento de verdad no puede darse: `decidir_con_el_previo` para.
    reusados_por_clave = {c3p._clave(r) for r in resultados if r.get("reusado")}
    sustituidos_sin_medida = [r for r in sustituidos_sin_medida
                              if c3p._clave(r) not in reusados_por_clave]
    procedencias_previas = payload_previo.get("procedencias") or {}
    reintentados = c3p.rastro_de_los_reintentos(
        sustituidos_sin_medida, resultados,
        rastro_previo=list(payload_previo.get("intentos_reintentados") or []),
        procedencias_previas=procedencias_previas)
    procedencias, sin_procedencia = c3p.c3._procedencias_citadas(en_el_fichero, procedencia,
                                                                 procedencias_previas)
    for t in reintentados:
        pid = t.get("procedencia_id")
        if pid and pid not in procedencias and pid in procedencias_previas:
            procedencias[pid] = procedencias_previas[pid]
    reconciliacion = c3p.c5.reconciliar_el_plan_con_lo_medido(plan, particiones, resultados,
                                                              n_motores=1)
    if veredicto_x_de_40 is not None:
        # M7: el X/40 dice, además de la regla, si hubo reintentos (en C4 tiene que ser 0).
        veredicto_x_de_40 = dict(
            veredicto_x_de_40, n_intentos_reintentados=len(reintentados),
            conjuntos_con_intentos_reintentados=sorted({t["dataset"] for t in reintentados}))
    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "119-C4",
        "tipo_de_ejecucion": tipo,
        "motor": c3p.NOMBRE_MOTOR_NUEVO,
        "configuracion_del_motor": configuracion_del_motor,
        "protocolo_119_v4_digest_sha256": c3p._digest_fichero_json(c3p.RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": c3p._digest_fichero_json(
            c3p.RUTA_DE_LA_ENMIENDA_1),
        "protocolos_encadenados": protocolos,
        "enmienda_2": enmienda_2,
        "parada_temprana_declarada": c3p.parada_temprana_declarada(resultados),
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial,
        "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "plan": plan,
        "intentos_por_conjunto": c3p.intentos_por_conjunto(
            datasets, particiones, esperadas_por_conjunto, resultados, protocolo),
        "por_que_n_intentos_no_es_el_del_plan": reconciliacion,
        "particion_por_dataset": dict(particiones),
        "particion_comparada_con_la_v2": dict(comparacion_de_particiones),
        "pliegues_digest_por_dataset": dict(pliegues_digest),
        "tiempos_del_padre_por_dataset": dict(tiempos_del_padre),
        "criterio_de_los_conjuntos": (
            "los 8 SELLADOS del protocolo v4 (veredicto.regla_de_subida: reservados para C4), o "
            "el subconjunto de --solo, declarado arriba. Los 32 no sellados son de C3 y se "
            "niegan"),
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados), "n_reusados": reusados,
        "n_intentos_en_el_fichero": len(en_el_fichero),
        "n_registros_conservados_de_ejecuciones_anteriores": len(conservados),
        "n_intentos_reintentados": len(reintentados),
        "conjuntos_con_intentos_reintentados": sorted({t["dataset"] for t in reintentados}),
        "que_son_los_intentos_reintentados": c3p.QUE_SON_LOS_REINTENTADOS,
        "intentos_reintentados": reintentados,
        "digest_de_la_cache": {
            "entorno": procedencia["digests_de_codigo"].get("entorno"),
            "componentes": dict(componentes_del_digest),
            "motor_nuevo": {"digest": procedencia["digests_de_codigo"].get("motor_nuevo"),
                            "ficheros": [c3p._etiqueta(f) for f in c3p._FICHEROS_DEL_MOTOR_NUEVO]},
            "clave_de_cada_registro": ("entorno_digest (el de C3 MÁS este guion), motor_digest, "
                                       "presupuesto_wall_s y datos_sha256 del ARFF "
                                       "(_reusable_c4: la de C3 SIN exigir que el estado cuente "
                                       "como medida -- en C4 un fallido ES la medida; un previo "
                                       "no reusable PARA la pasada)"),
        },
        "lo_que_no_cubre_el_digest": lo_que_no_cubre_el_digest_c4(),
        "lectura_de_los_datos": dict(c3p.c3.LECTURA_DECLARADA),
        "resultados": en_el_fichero,
        "veredicto_por_conjunto": veredictos_por_conjunto,
        "veredicto_x_de_40": veredicto_x_de_40,
        "que_es_veredicto_x_de_40": (
            "SOLO se compone cuando tipo_de_ejecucion=='pasada' y parcial=False: el X/40 real "
            "(protocolo_119_v4.json, veredicto.confirmacion: «la receta elegida, UNA vez, en "
            "los SELLADOS; es la cifra que se publica»). `null` en cualquier otro caso (--solo, "
            "--estimar, o una pasada real todavía a medias)"),
    }
    c3p.c3.sellar_la_salida(salida)
    c3p._escribir_atomicamente(ruta_salida, salida)
    return salida


if __name__ == "__main__":
    main()
