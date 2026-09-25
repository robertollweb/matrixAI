#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C6 — el ensamblado de los aprobados (`matrixai.ensamblado.media`)
medido contra el ORÁCULO de cada conjunto, en los NO sellados del protocolo v2.

Implementa TAL CUAL el «REGISTRO DE C6, escrito el 2026-09-25 ANTES de
construir el motor y de medir nada» (`documentacion/
114_EL_ESTUDIO_COMPLETO_CONTRACT.md`, bloque C6). No se cambia nada de ahí:
este guion es su implementación, no una segunda decisión.

QUÉ SE REUTILIZA, Y DE DÓNDE — nada de lo de abajo se copia, se importa:

* **El catálogo del protocolo v2** (`fijar_el_catalogo_v2` de
  `pasada_v2_113.py`): la MISMA función que fija el objetivo declarado y la
  exclusión de identificadores para los 40 conjuntos, para que `cargar_arff`
  lea exactamente lo que el protocolo v2 registró y no lo que el v1
  registraba (auditoría del 113, `kick` tomando `WarrantyCost` como
  objetivo).
* **La lectura, la partición y la preparación de cada conjunto**
  (`particiones_base`, `preparar_para_motor`, `problem_spec_de` de
  `pasada_amplia_101_c5.py`): pliegues, repeticiones, semillas y
  estratificación por tarea salen de ahí, sin un segundo diseño de
  partición escrito aquí.
* **La métrica de cierre por tarea** (`metrica_de_cierre_por_dataset`,
  `METRICA_DE_CIERRE_POR_TAREA` de `pasada_amplia_101_c5.py`): AUROC en
  binaria, accuracy en multiclase, R² en regresión — la MISMA traducción de
  `regla_de_cierre.metrica_por_tarea` que usa la pasada v2, no una nueva.
* **La ejecución de un intento en subproceso con su tope**
  (`ejecutar_intento_aislado` de `matrixai_engines.subproceso`): el mismo
  camino que usan los 7 motores de la pasada v2, para lightgbm, sklearn.hgb
  y el ensamblado por igual.
* **El bootstrap del veredicto** (`protocolo._intervalo_pareado`): bootstrap
  de percentiles sobre diferencias YA EMPAREJADAS, 1.000 remuestras de
  PLIEGUES, semilla 0 — el MISMO motor de remuestreo que usa
  `aplicar_regla_de_cierre`/`_intervalo_de_la_distancia` para el veredicto de
  Fase 0. **Por qué esta pasada llama a `_intervalo_pareado` y no a
  `_intervalo_de_la_distancia`**: esa función envuelve el bootstrap para una
  pregunta distinta —si un listón FIJO (`regla.puntos`, en puntos x100) cae
  DENTRO del intervalo (`cruza_el_liston`)—, y aquí la pregunta del registro
  de C6 es si el intervalo ENTERO cabe dentro de un margen de equivalencia
  (±0,005, en la escala cruda de la métrica). Son preguntas distintas con la
  misma aritmética de remuestreo por debajo; se reutiliza esa aritmética
  compartida (`_intervalo_pareado`) y se compone aquí la pregunta que el
  registro de C6 pide, en vez de forzar la envoltura ajena a decir algo que
  no calcula.
* **La regla de cierre de la cartera** (`aplicar_regla_de_cierre` de
  `protocolo.py`): la MISMA función que decide «¿a 2 puntos del mejor en
  ≥80%?», llamada dos veces por aprobado — sin el ensamblado compitiendo y
  con él dentro — para poder decir qué le pasa a cada aprobado.
* **La caché por intento, la procedencia y la escritura atómica**
  (`_reusable`, `_cargar_cache`, `procedencia_de_la_medicion`,
  `_procedencias_citadas`, `sellar_la_salida` de `pasada_exploratoria_101_c3.
  py`): el mismo mecanismo que ya está auditado, no uno nuevo para este
  guion.

QUÉ SÍ ES NUEVO AQUÍ, Y POR QUÉ NO PODÍA REUTILIZARSE: el BUCLE que decide
QUÉ motores corren (los miembros de `cartera.CARTERA_APROBADA`, leídos —
nunca escritos a mano, ver `nombres_de_los_motores_de_la_pasada()` — más el
ensamblado) y SOBRE QUÉ conjuntos (los NO sellados del protocolo v2, ver
`datasets_no_sellados()`). La pasada v2 corre los 7 motores del protocolo
sobre los 40; este corte corre 2 miembros + 1 ensamblado sobre 32. Es la
misma familia de guion que `pasada_v2_113.py` — que tampoco reescribe el
bucle de `pasada_amplia_101_c5.py`, lo AJUSTA por fuera — pero aquí el
conjunto de motores también cambia, así que el bucle en sí (no lo que hace
en cada vuelta) es del corte y vive aquí, no en `pasada_amplia_101_c5.py`.

**Nada de esto toca el protocolo v2 ni los guiones de C3/C5.** El fichero de
motor por nombre del ensamblado
(`_DIR_ENGINES / "motores" / "ensamblado.py"`) y su entrada en el mapa de
configuraciones se resuelven aquí, en memoria (`_fichero_por_motor`,
`_configuraciones_de_la_pasada`), extendiendo lo de C3 sin escribir en él.

**El motor del ensamblado corre por el MISMO camino que los demás.**
`matrixai_engines.registro_de_motores.motor_para("matrixai.ensamblado.
media")` ya lo resuelve (108-C5, extendido en 114-C6) — es un `Motor` más
para `ejecutar_intento_aislado`, sin caso especial: recibe el mismo
`Presupuesto.wall_seconds` que cualquier otro motor del cubo, y es
`MotorEnsamblado._ajustar()` quien reparte ESE presupuesto entre sus
miembros por dentro (ver `matrixai_engines/motores/ensamblado.py`).

CÓMO SE LANZA:

    python3 benchmarks/fase0/pasada_114c6_ensamblado.py --estimar
    python3 benchmarks/fase0/pasada_114c6_ensamblado.py --solo dresses-sales \\
        --salida /tmp/prueba_c6.json
    python3 benchmarks/fase0/pasada_114c6_ensamblado.py    # LA PASADA ENTERA,
        # horas — a la cola nocturna (~/encolar.sh), nunca a mano un ratito.
"""
from __future__ import annotations

import argparse
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
    NOMBRE as NOMBRE_ENSAMBLADO, nombres_de_los_miembros_aprobados)
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.registro_de_motores import motor_para  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

RUTA_DEL_PROTOCOLO_V2 = _AQUI / "protocolo_exploratorio_v2.json"
RUTA_DE_SALIDA = _AQUI / "resultado_114c6_ensamblado.json"

#: EL MARGEN DE EQUIVALENCIA PRÁCTICA — «medio punto de la métrica: por
#: debajo, una diferencia no se nota» (registro de C6, literal). En la escala
#: CRUDA de la métrica de cierre (AUROC/accuracy/R², todas 0-1), no en puntos
#: x100 como el listón de la cartera.
MARGEN_DE_EQUIVALENCIA = 0.005

#: LA SEMILLA Y LAS REMUESTRAS DEL BOOTSTRAP, tal como las fija el registro
#: de C6 — que son las mismas que trae `protocolo._intervalo_pareado` por
#: omisión. Se dejan aquí, escritas, para que quien lea este fichero no
#: tenga que ir a comprobar los valores por omisión de una función ajena.
SEMILLA_DEL_BOOTSTRAP = protocolo_mod.SEMILLA_DEL_INTERVALO  # 0
REMUESTRAS_DEL_BOOTSTRAP = protocolo_mod.REMUESTRAS_DEL_INTERVALO  # 1000

#: EL MÍNIMO DEL CRITERIO DEL CORTE — «mejora en ≥60% de los no sellados».
MINIMO_DE_MEJORA = 0.60

#: DIRECCIÓN DE CADA MÉTRICA DE CIERRE QUE LA V2 PUEDE USAR: True si MÁS es
#: MEJOR. Las tres que `METRICA_DE_CIERRE_POR_TAREA` traduce hoy (auroc,
#: accuracy, r2) YA son «más es mejor» — es precisamente por eso que C5
#: eligió r2 y no rmse para regresión (ver su docstring, punto de la
#: métrica de cierre). Un `metric_id` que no esté aquí PARA la pasada en vez
#: de asumir una dirección: «respeta la dirección como lo haga la v2» no se
#: cumple adivinando.
METRICAS_MAS_ES_MEJOR = {"auroc": True, "accuracy": True, "r2": True}


def direccion_de(metric_id: str) -> bool:
    """True si MÁS es MEJOR para esta métrica de cierre. Para la pasada
    (`SystemExit`) si la métrica no está declarada: medir con una dirección
    supuesta sería el mismo defecto que el `auroc` de repuesto que C5 cerró."""
    if metric_id not in METRICAS_MAS_ES_MEJOR:
        raise SystemExit(
            f"la metrica de cierre {metric_id!r} no tiene direccion declarada en "
            f"METRICAS_MAS_ES_MEJOR ({sorted(METRICAS_MAS_ES_MEJOR)}). Anadirla es "
            f"una decision que se toma ANTES de medir y por escrito, no un valor "
            f"de repuesto")
    return METRICAS_MAS_ES_MEJOR[metric_id]


# ---------------------------------------------------------------------------
# 1. LOS CONJUNTOS: los NO sellados del protocolo v2, con el catálogo v2 fijo
# ---------------------------------------------------------------------------

def preparar_protocolo_v2() -> None:
    """Apunta la lectura de C3/C5 al protocolo v2 y fija su catálogo — la
    MISMA operación que hace `pasada_v2_113.main()`, reutilizada tal cual, no
    reescrita aquí. Sin esto, `cargar_arff` leería con el catálogo v1 (el
    objetivo declarado y la exclusión de identificadores de OTRO protocolo)."""
    c3.RUTA_DEL_PROTOCOLO = RUTA_DEL_PROTOCOLO_V2
    v2.fijar_el_catalogo_v2()


def datasets_no_sellados(protocolo: protocolo_mod.ProtocoloExploratorio):
    """Los conjuntos NO sellados del protocolo v2 registrado, en el orden
    estable de `pasada_amplia_101_c5.datasets_de_la_pasada` (barato primero) —
    sin lista escrita a mano, y sin tocar ninguno de los 8 sellados salvo que
    se pida explícitamente por `--solo` (ver `main`, que lo avisa en voz
    alta si pasa)."""
    return [d for d in c5.datasets_de_la_pasada(protocolo) if not d.sellado]


# ---------------------------------------------------------------------------
# 2. LOS MOTORES: los miembros de la cartera (leídos, nunca escritos a mano)
#    más el ensamblado — resueltos por el MISMO `motor_para` que usa el resto
#    del árbol.
# ---------------------------------------------------------------------------

def nombres_de_los_motores_de_la_pasada() -> tuple[str, ...]:
    """Los miembros APROBADOS de HOY (`cartera.CARTERA_APROBADA`, vía
    `nombres_de_los_miembros_aprobados()` de `ensamblado.py` — el mismo
    lector que usa el propio motor) más el ensamblado, en ese orden.

    Se pregunta a la cartera EN CADA LLAMADA, sin caché de módulo: si la
    cartera crece o encoge, esta pasada cambia de miembros sin tocar este
    fichero, y una prueba puede sustituir `cartera.CARTERA_APROBADA` (mismo
    patrón que usa `test_c102_c1_cartera.py`) y verlo reflejado aquí."""
    return tuple(nombres_de_los_miembros_aprobados()) + (NOMBRE_ENSAMBLADO,)


def motores_de_la_pasada() -> list:
    return [motor_para(n) for n in nombres_de_los_motores_de_la_pasada()]


def _fichero_por_motor() -> dict[str, Path]:
    """`nombre de motor -> fichero cuyo digest invalida SUS intentos en la
    caché`, para los miembros (leído de `c3._FICHERO_POR_MOTOR`, sin
    copiarlo) MÁS el ensamblado — resuelto aquí, en memoria, sin tocar
    `pasada_exploratoria_101_c3.py`."""
    mapa = {n: c3._FICHERO_POR_MOTOR[n] for n in nombres_de_los_miembros_aprobados()}
    mapa[NOMBRE_ENSAMBLADO] = c3._DIR_ENGINES / "motores" / "ensamblado.py"
    return mapa


#: FICHEROS COMPARTIDOS: los de C5 (que ya incluyen los de C3) más este
#: guion, y ADEMÁS `cartera.py` y `registro_de_motores.py` — un cambio ahí
#: cambia QUÉ MOTORES corren, no solo cómo corre uno, así que invalida TODA
#: la caché de esta pasada y no solo la del ensamblado (que ya está cubierto
#: aparte por `_fichero_por_motor`, cuyo digest de `ensamblado.py` vive en
#: `motor_digest` y no aquí).
_FICHEROS_COMPARTIDOS = c5._FICHEROS_COMPARTIDOS + (
    Path(__file__).resolve(),
    c3._DIR_ENGINES / "cartera.py",
    c3._DIR_ENGINES / "registro_de_motores.py",
)


def _digest_entorno() -> str:
    return __import__("hashlib").sha256(
        "".join(c3._digest_fichero(f) for f in _FICHEROS_COMPARTIDOS).encode()
    ).hexdigest()[:16]


def _configuraciones_de_la_pasada() -> dict[str, list[str]]:
    """Una configuración por motor (`c3.CONFIGURACION_UNICA`), para los
    miembros y el ensamblado — resuelto en memoria, sin tocar
    `configuraciones_de_la_pasada()` de C3 (que solo conoce los 7 motores del
    protocolo)."""
    return {n: [c3.CONFIGURACION_UNICA] for n in nombres_de_los_motores_de_la_pasada()}


def _exigir_que_quepa() -> None:
    """El mismo guardia de reserva de CPU que C3/C5, reutilizado: comprueba
    que `PROCESOS_A_LA_VEZ x HILOS_POR_INTENTO` (los MISMOS de C3, no unos
    nuevos) caben en esta máquina antes de medir nada."""
    caben = protocolo_mod.reserva_segura(c3.HILOS_POR_INTENTO)
    if c3.PROCESOS_A_LA_VEZ > caben:
        raise SystemExit(
            f"esta pasada pediria {c3.PROCESOS_A_LA_VEZ} procesos x "
            f"{c3.HILOS_POR_INTENTO} hilos y en esta maquina "
            f"({protocolo_mod.cpus_disponibles()} CPUs) caben {caben}")


# ---------------------------------------------------------------------------
# 3. EL VEREDICTO POR CONJUNTO: oráculo, diferencia emparejada, bootstrap
# ---------------------------------------------------------------------------

def elegir_oraculo(medias_por_miembro: dict[str, float]) -> tuple[str | None, float | None]:
    """El ORÁCULO de un conjunto: el MIEMBRO con MEJOR media (nunca el peor
    — `max`, a propósito, no `min`). `medias_por_miembro` ya viene orientada
    «más alto es mejor» (quien construye el mapa aplica `direccion_de`
    antes). Vacío -> `(None, None)`: sin ningún miembro medido no hay
    oráculo que declarar, y no se inventa uno."""
    if not medias_por_miembro:
        return None, None
    nombre = max(medias_por_miembro, key=lambda m: medias_por_miembro[m])
    return nombre, medias_por_miembro[nombre]


def diferencias_emparejadas(por_pliegue_a: dict[tuple, float],
                             por_pliegue_b: dict[tuple, float],
                             ) -> tuple[list[float], list[tuple]]:
    """`a - b` en los pliegues COMUNES a los dos mapas, en orden estable
    (`sorted`) — el bootstrap remuestrea sobre una lista, y su orden decide
    qué índice cae en qué remuestra."""
    comunes = sorted(set(por_pliegue_a) & set(por_pliegue_b))
    return [por_pliegue_a[k] - por_pliegue_b[k] for k in comunes], comunes


def veredicto_del_conjunto(*, dataset: str, oraculo_motor: str | None,
                            oraculo_media: float | None, ensamblado_media: float | None,
                            ensamblado_por_pliegue: dict[tuple, float],
                            oraculo_por_pliegue: dict[tuple, float],
                            fallo_del_ensamblado: str | None) -> dict:
    """El veredicto de UN conjunto no sellado, tal como lo define el registro
    de C6: diferencia emparejada (ensamblado − oráculo) por repetición y
    pliegue, intervalo al 95% por bootstrap de percentiles de PLIEGUES
    (`protocolo._intervalo_pareado`, 1.000 remuestras, semilla 0). «Mejora» =
    el intervalo excluye el cero por arriba Y no cabe entero dentro de
    ±0,005. Un fallo del ensamblado = no mejora, con su motivo — comprobado
    ANTES del bootstrap, no después: un fallo no tiene pliegues que
    emparejar de forma que signifique algo."""
    base = {"dataset": dataset, "oraculo": oraculo_motor, "oraculo_media": oraculo_media,
            "ensamblado_media": ensamblado_media}
    if fallo_del_ensamblado:
        return {**base, "mejora": False, "intervalo": None, "n_pliegues_comunes": 0,
                "motivo": f"fallo del ensamblado: {fallo_del_ensamblado}"}
    if oraculo_motor is None:
        return {**base, "mejora": False, "intervalo": None, "n_pliegues_comunes": 0,
                "motivo": "sin oraculo: ningun miembro aprobado se pudo medir en este conjunto"}
    diferencias, comunes = diferencias_emparejadas(ensamblado_por_pliegue, oraculo_por_pliegue)
    if len(diferencias) < 2:
        return {**base, "mejora": False, "intervalo": None, "n_pliegues_comunes": len(comunes),
                "motivo": f"menos de dos pliegues comunes ({len(comunes)}) entre el ensamblado "
                          f"y el oraculo: no hay nada que remuestrear"}
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


def veredicto_del_corte(veredictos_por_conjunto: list[dict], *,
                         minimo: float = MINIMO_DE_MEJORA) -> dict:
    """La proporción de «mejora» sobre los conjuntos dados, y si cumple el
    mínimo del corte (60%)."""
    total = len(veredictos_por_conjunto)
    mejoras = sum(1 for v in veredictos_por_conjunto if v["mejora"])
    fraccion = (mejoras / total) if total else 0.0
    return {"n_conjuntos": total, "n_mejora": mejoras, "fraccion_mejora": fraccion,
            "minimo_exigido": minimo, "cumple": total > 0 and fraccion >= minimo,
            "conjuntos_que_mejoran": sorted(v["dataset"] for v in veredictos_por_conjunto
                                            if v["mejora"])}


def medida_por_pliegue(registros: list[dict], motor: str, metric_id: str) -> dict[tuple, float]:
    """`(repeticion, pliegue) -> valor` de la métrica de cierre, SOLO de los
    intentos de `motor` cuyo estado cuenta como medida (la MISMA lista blanca
    que usa `aplicar_regla_de_cierre`, `ESTADOS_QUE_CUENTAN_COMO_MEDIDA`) y
    con valor no ausente."""
    return {(r["repeticion"], r["pliegue"]): r[metric_id]
            for r in registros
            if r["motor"] == motor
            and r.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
            and r.get(metric_id) is not None}


def media_de(por_pliegue: dict[tuple, float]) -> float | None:
    valores = list(por_pliegue.values())
    return (sum(valores) / len(valores)) if valores else None


def fallos_de(registros: list[dict], motor: str) -> list[dict]:
    return [r for r in registros if r["motor"] == motor
            and r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]


def computar_veredicto_del_conjunto(ds, registros_del_dataset: list[dict],
                                     metric_id: str, miembros: list[str]) -> dict:
    """Compone el veredicto de UN conjunto a partir de sus registros crudos:
    la media de cada miembro, el oráculo, la del ensamblado, y si el
    ensamblado falló en algún pliegue de este conjunto."""
    medias_miembros: dict[str, float] = {}
    por_pliegue_miembros: dict[str, dict[tuple, float]] = {}
    for m in miembros:
        pp = medida_por_pliegue(registros_del_dataset, m, metric_id)
        por_pliegue_miembros[m] = pp
        media = media_de(pp)
        if media is not None:
            medias_miembros[m] = media
    oraculo_motor, oraculo_media = elegir_oraculo(medias_miembros)
    oraculo_por_pliegue = por_pliegue_miembros.get(oraculo_motor, {}) if oraculo_motor else {}
    ensamblado_por_pliegue = medida_por_pliegue(registros_del_dataset, NOMBRE_ENSAMBLADO, metric_id)
    ensamblado_media = media_de(ensamblado_por_pliegue)
    fallos_ens = fallos_de(registros_del_dataset, NOMBRE_ENSAMBLADO)
    fallo_del_ensamblado = None
    if fallos_ens:
        fallo_del_ensamblado = "; ".join(
            f"rep={r['repeticion']} pliegue={r['pliegue']} estado={r['estado']} "
            f"motivo={r.get('motivo')}" for r in fallos_ens)
    return veredicto_del_conjunto(
        dataset=ds.nombre, oraculo_motor=oraculo_motor, oraculo_media=oraculo_media,
        ensamblado_media=ensamblado_media, ensamblado_por_pliegue=ensamblado_por_pliegue,
        oraculo_por_pliegue=oraculo_por_pliegue, fallo_del_ensamblado=fallo_del_ensamblado)


# ---------------------------------------------------------------------------
# 4. LA REGLA DE LA CARTERA, ANTES Y DESPUÉS DEL ENSAMBLADO
# ---------------------------------------------------------------------------

def regla_de_la_cartera_antes_y_despues(resultados: list[dict], *,
                                         regla: protocolo_mod.ReglaDeCierre,
                                         metrica_por_dataset: dict[str, str],
                                         miembros: list[str],
                                         nombres_de_los_conjuntos: list[str]) -> dict:
    """Rehace `aplicar_regla_de_cierre` (la MISMA función que decide «a 2
    puntos del mejor en ≥80%») dos veces por aprobado: SIN el ensamblado
    compitiendo (`antes`, la cartera de hoy) y CON él dentro como un
    competidor más (`despues`) — sobre los conjuntos medidos en ESTA pasada
    (los no sellados de la v2), no sobre los 40 de la pasada v2 completa: se
    declara así y no se presenta como el cierre oficial de la cartera, que
    sigue siendo el de `pasada_v2_113_resultado.json`."""
    resultados_miembros = [r for r in resultados if r["motor"] in miembros]
    resultados_con_ensamblado = resultados_miembros + [
        r for r in resultados if r["motor"] == NOMBRE_ENSAMBLADO]
    salida: dict[str, dict] = {}
    for aprobado in miembros:
        antes = protocolo_mod.aplicar_regla_de_cierre(
            resultados_miembros, regla, motor=aprobado,
            metrica_por_dataset=metrica_por_dataset,
            datasets_exigidos=nombres_de_los_conjuntos)
        despues = protocolo_mod.aplicar_regla_de_cierre(
            resultados_con_ensamblado, regla, motor=aprobado,
            metrica_por_dataset=metrica_por_dataset,
            datasets_exigidos=nombres_de_los_conjuntos)
        resumen = lambda v: {k: v[k] for k in  # noqa: E731
                             ("cumplidos", "datasets", "fraccion", "cumple_la_regla")}
        salida[aprobado] = {
            "antes": resumen(antes), "despues": resumen(despues),
            "diferencia_cumplidos": despues["cumplidos"] - antes["cumplidos"],
        }
    return salida


# ---------------------------------------------------------------------------
# 5. LA CUENTA DE LO QUE VA A COSTAR (--estimar)
# ---------------------------------------------------------------------------

def plan_de_la_pasada(datasets, protocolo: protocolo_mod.ProtocoloExploratorio,
                       n_motores: int) -> dict:
    """Peor caso literal: cada intento agota el presupuesto de pared de su
    cubo. Los números de folds/repeticiones/presupuesto salen SIEMPRE de los
    accesores del protocolo registrado (`c3.wall_seconds_del_cubo`,
    `protocolo.particion.repeticiones_para`), nunca de una constante propia
    — lo único de este guion es CUÁNTOS motores corren por pliegue."""
    por_cubo: dict[str, dict] = {}
    total = 0
    segundos_cota = 0.0
    for ds in datasets:
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        intentos = protocolo.particion.folds * repeticiones * n_motores
        wall = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        total += intentos
        segundos_cota += intentos * wall
        entrada = por_cubo.setdefault(ds.cubo, {
            "n_datasets": 0, "folds": protocolo.particion.folds,
            "repeticiones": repeticiones, "wall_seconds_por_intento": wall,
            "n_intentos": 0, "horas_cota": 0.0})
        entrada["n_datasets"] += 1
        entrada["n_intentos"] += intentos
        entrada["horas_cota"] += intentos * wall / 3600.0
    for entrada in por_cubo.values():
        entrada["horas_cota"] = round(entrada["horas_cota"], 2)
    return {
        "n_datasets": len(datasets), "n_motores": n_motores, "n_intentos": total,
        "por_cubo": por_cubo,
        "horas_de_reloj_cota_peor_caso": round(segundos_cota / 3600.0, 2),
        "que_es_la_cota": (
            "peor caso literal: cada uno de los intentos (miembros Y ensamblado, "
            "cada uno con el presupuesto ENTERO de su cubo, que el ensamblado "
            "reparte por dentro entre sus miembros) agota el presupuesto de pared "
            "de su cubo. No es una prevision de lo que va a durar de verdad."),
    }


def _imprimir_estimacion(plan: dict, n_miembros: int) -> None:
    print(f"conjuntos NO sellados: {plan['n_datasets']}, motores por pliegue: "
          f"{plan['n_motores']} ({n_miembros} miembro(s) + el ensamblado)")
    for cubo, e in sorted(plan["por_cubo"].items()):
        print(f"  cubo {cubo:8}: {e['n_datasets']:>2} datasets x {e['folds']} pliegues x "
              f"{e['repeticiones']} rep x {plan['n_motores']} motores = {e['n_intentos']:>5} "
              f"intentos, {e['wall_seconds_por_intento']:.0f}s de tope -> cota "
              f"{e['horas_cota']:.2f} h")
    print(f"  TOTAL: {plan['n_intentos']} intentos")
    print(f"  COTA de peor caso: {plan['horas_de_reloj_cota_peor_caso']:.2f} h "
          f"({plan['horas_de_reloj_cota_peor_caso']/24:.2f} dias)")
    print(f"\n  {plan['que_es_la_cota']}")
    print("\n  Esto es SOLO la aritmetica del peor caso, sin medir nada. El coste real "
          "se mide en la cola nocturna (~/encolar.sh), no a mano.")


# ---------------------------------------------------------------------------
# 6. LA PASADA
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta todos los intentos")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON de salida (por omision, el de este directorio)")
    parser.add_argument("--estimar", action="store_true",
                        help="imprime la cuenta de coste con su aritmetica y NO mide nada")
    parser.add_argument("--solo", default=None,
                        help="nombres de conjunto separados por coma: corre SOLO esos. "
                             "Para probar el guion, nunca para medir el corte de verdad")
    args = parser.parse_args(argv)

    preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    datasets = [d for d in todos if not d.sellado]
    subconjunto = None
    if args.solo:
        pedidos = [n.strip() for n in args.solo.split(",") if n.strip()]
        por_nombre = {d.nombre: d for d in todos}
        desconocidos = [n for n in pedidos if n not in por_nombre]
        if desconocidos:
            raise SystemExit(f"--solo nombra conjuntos que no estan en el protocolo v2: "
                             f"{desconocidos}")
        datasets = [por_nombre[n] for n in pedidos]
        subconjunto = pedidos
        sellados_pedidos = [n for n in pedidos if por_nombre[n].sellado]
        if sellados_pedidos:
            print(f"AVISO: --solo incluye conjunto(s) SELLADO(S) {sellados_pedidos} -- esto "
                  f"es una prueba del guion, NO CUENTA como la medicion de sellados del "
                  f"registro de C6 ('una sola vez y despues')", flush=True)

    _exigir_que_quepa()
    miembros = list(nombres_de_los_miembros_aprobados())
    if not miembros:
        raise SystemExit("la cartera aprobada esta vacia: no hay oraculo posible")
    motores = motores_de_la_pasada()
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    plan = plan_de_la_pasada(datasets, protocolo, n_motores=len(motores))
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]}")
    print(f"miembros de la cartera: {miembros}; ensamblado: {NOMBRE_ENSAMBLADO}")
    _imprimir_estimacion(plan, len(miembros)) if args.estimar else None
    if args.estimar:
        return

    ruta_salida = Path(args.salida) if args.salida else RUTA_DE_SALIDA
    cache_previo, payload_previo = ({}, {}) if args.forzar else c3._cargar_cache(ruta_salida)
    entorno_digest = _digest_entorno()
    digest_por_motor = {n: c3._digest_fichero(r) for n, r in _fichero_por_motor().items()}

    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "por_motor": digest_por_motor},
        datos_de_entrada={d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = c3.procedencia_declarada(payload_previo)
        print(f"cache previo: {len(cache_previo)} registros, procedencia "
              f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)

    resultados: list[dict] = []
    veredictos_por_conjunto: list[dict] = []
    reusados = 0
    inicio = time.perf_counter()
    regla = protocolo.regla_de_cierre
    nombres_de_los_conjuntos = [d.nombre for d in datasets]

    SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0
    ultimo_punto_de_control = [time.perf_counter()]

    def guardar(parcial: bool) -> dict:
        ultimo_punto_de_control[0] = time.perf_counter()
        veredicto = (None if parcial else veredicto_del_corte(veredictos_por_conjunto))
        regla_cartera = (None if parcial else regla_de_la_cartera_antes_y_despues(
            resultados, regla=regla, metrica_por_dataset=metrica_por_dataset,
            miembros=miembros, nombres_de_los_conjuntos=nombres_de_los_conjuntos))
        return _componer_y_guardar(
            resultados, veredictos_por_conjunto, veredicto, regla_cartera,
            procedencia, payload_previo, ruta_salida, datasets=datasets, todos=todos,
            plan=plan, protocolo=protocolo, metrica_por_dataset=metrica_por_dataset,
            miembros=miembros, subconjunto=subconjunto,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados, parcial=parcial)

    for ds in datasets:
        por_id, propuesta, spec, objetivo, predictores, declarada = c5.particiones_base(
            ds, protocolo)
        test_ids = propuesta.plan.observaciones_del_rol("test")
        wall_seconds = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        metric_id = metrica_por_dataset[ds.nombre]
        print(f"\n=== {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"n={len(por_id)}, test={len(test_ids)}, tope={wall_seconds:.0f}s, "
              f"metrica={metric_id}) ===", flush=True)

        registros_del_dataset: list[dict] = []
        for repeticion in range(repeticiones):
            for pliegue_i in range(protocolo.particion.folds):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None:
                    continue
                for motor in motores:
                    clave = (ds.nombre, motor.nombre, repeticion, pliegue_i)
                    previo = cache_previo.get(clave)
                    if c3._reusable(previo, entorno_digest, digest_por_motor[motor.nombre],
                                     wall_seconds):
                        registro = dict(previo, reusado=True)
                        registro.setdefault("procedencia_id", None)
                        resultados.append(registro)
                        registros_del_dataset.append(registro)
                        reusados += 1
                        print(f"  [reusado] {ds.nombre} {motor.nombre} rep={repeticion} "
                              f"pliegue={pliegue_i}: estado={registro.get('estado')}", flush=True)
                        continue

                    crudas_train = [por_id[i] for i in pliegue.entrena]
                    crudas_val = [por_id[i] for i in pliegue.valida]
                    crudas_test = [por_id[i] for i in test_ids]
                    transformadas = c3.preparar_para_motor(
                        crudas_train, crudas_train + crudas_val + crudas_test,
                        objetivo, predictores, motor)
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
                        "motor_digest": digest_por_motor[motor.nombre],
                        "procedencia_id": procedencia["procedencia_id"],
                        "reusado": False,
                    }
                    c5.aplanar_metricas_en_el_registro(registro, metricas)
                    resultados.append(registro)
                    registros_del_dataset.append(registro)
                    valor_de_cierre = registro.get(metric_id)
                    print(f"  {ds.nombre} {motor.nombre} rep={repeticion} pliegue={pliegue_i}: "
                          f"estado={intento.estado} {metric_id}={valor_de_cierre} "
                          f"wall={transcurrido:.1f}s", flush=True)
                if (time.perf_counter() - ultimo_punto_de_control[0]
                        >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL):
                    guardar(parcial=True)
            guardar(parcial=True)

        veredicto_ds = computar_veredicto_del_conjunto(
            ds, registros_del_dataset, metric_id, miembros)
        veredictos_por_conjunto.append(veredicto_ds)
        print(f"  VEREDICTO {ds.nombre}: mejora={veredicto_ds['mejora']} "
              f"oraculo={veredicto_ds['oraculo']} motivo={veredicto_ds['motivo']}", flush=True)
        guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados} reusados, {len(resultados)-reusados} ejecutados) ===")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    v = salida["veredicto"]
    print(f"VEREDICTO DEL CORTE: {v['n_mejora']}/{v['n_conjuntos']} = "
          f"{v['fraccion_mejora']:.3f} ({'CUMPLE' if v['cumple'] else 'NO CUMPLE'} "
          f"el minimo {v['minimo_exigido']})")


def _componer_y_guardar(resultados, veredictos_por_conjunto, veredicto, regla_cartera,
                         procedencia, payload_previo, ruta_salida, *, datasets, todos,
                         plan, protocolo, metrica_por_dataset, miembros, subconjunto,
                         total_wall_s, reusados, parcial) -> dict:
    """Compone el JSON y lo escribe, atomicamente — mismo patron que C3/C5
    (`_procedencias_citadas`, `sellar_la_salida`, escritura a temporal +
    `replace`), y con lo medido tras CADA conjunto: si se corta, queda lo
    hecho."""
    import json

    procedencias, sin_procedencia = c3._procedencias_citadas(
        resultados, procedencia, (payload_previo.get("procedencias") or {}))

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "114-C6",
        "registro_del_corte": (
            "documentacion/114_EL_ESTUDIO_COMPLETO_CONTRACT.md, bloque 'REGISTRO DE C6, "
            "escrito el 2026-09-25 ANTES de construir el motor y de medir nada'"),
        "candidato": NOMBRE_ENSAMBLADO,
        "miembros_de_la_cartera": list(miembros),
        "procedencia": procedencia,
        "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial,
        "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "plan": plan,
        "criterio_de_los_conjuntos": (
            "los conjuntos NO sellados del protocolo v2 (o el subconjunto pedido por "
            "--solo, declarado arriba). Los sellados se miden 'una sola vez y despues' "
            "(registro de C6): esta pasada no los toca salvo que --solo los nombre a "
            "proposito, y entonces lo avisa por consola."),
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "margen_de_equivalencia": MARGEN_DE_EQUIVALENCIA,
        "minimo_de_mejora_exigido": MINIMO_DE_MEJORA,
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados),
        "n_reusados": reusados,
        "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
        "resultados": resultados,
        "veredicto_por_conjunto": veredictos_por_conjunto,
        "veredicto": veredicto,
        "regla_de_la_cartera_antes_y_despues": regla_cartera,
    }
    c3.sellar_la_salida(salida)
    temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta_salida)
    return salida


if __name__ == "__main__":
    main()
