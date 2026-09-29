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
(`protocolo_exploratorio_v2.json`, idéntico en `particion`/`presupuesto`/
`regla_de_cierre` al v4 — comprobado byte a byte, ver `tests/
test_119_c3_pasada.py`), y compone el «campo» sustituyendo, SOLO en los
conjuntos medidos aquí, la densa v2 (`matrixai.dense.torch_cpu`) por el
motor nuevo.

**LOS 8 SELLADOS SE NIEGAN.** El protocolo v4 (`veredicto.regla_de_subida`)
reserva los sellados para C4 («sube a confirmación... si cumple las DOS
condiciones sobre los 32 no sellados»): a diferencia de
`pasada_118_palanca.py`, que dejaba pasar un `--solo` con sellados avisando
en voz alta («esto es una prueba, no cuenta»), aquí un sellado en `--solo`
para la pasada con `SystemExit` — no hay «solo una prueba» que valga para un
corte cuya regla de subida los reserva explícitamente.

QUÉ SE REUTILIZA, Y DE DÓNDE — nada de lo de abajo se copia a mano:

* **El catálogo, la lectura, la partición y el guardia de CPU**
  (`pasada_114c6_ensamblado.preparar_protocolo_v2`/`datasets_no_sellados`/
  `_exigir_que_quepa`, `pasada_amplia_101_c5.datasets_de_la_pasada`/
  `particiones_base`/`metrica_de_cierre_por_dataset`/`metricas_del_informe`/
  `aplanar_metricas_en_el_registro`): la MISMA partición, semillas y
  estratificación que la v2.
* **La ejecución de un intento en subproceso, con su tope**
  (`matrixai_engines.subproceso.ejecutar_intento_aislado`).
* **La caché por intento, la procedencia y la escritura atómica**
  (`pasada_exploratoria_101_c3._cargar_cache`/`procedencia_de_la_medicion`/
  `_procedencias_citadas`/`sellar_la_salida`/`_digest_fichero`): el mismo
  mecanismo ya auditado. `_reusable` de C3 se envuelve aquí (`_reusable_c3`)
  para añadir la condición que C3 no comprueba por su cuenta: el estado
  tiene que ser uno de `protocolo.ESTADOS_QUE_CUENTAN_COMO_MEDIDA` — sin
  eso, un intento `failed` con los digests correctos se reutilizaría para
  siempre (la trampa nº1 de «ANTES DE RELANZAR UNA PASADA», CLAUDE.md).
* **El «campo» de la comparación, el veredicto por conjunto (diferencia
  emparejada + bootstrap de pliegues) y la aritmética de «sube los
  cumplidos Y las mejoras superan a las inferioridades»**
  (`pasada_118_palanca.campo_de_la_comparacion`/`veredicto_del_conjunto`/
  `veredicto_de_la_palanca`, importados TAL CUAL): 118 y 119 comparten la
  MISMA regla de subida, copiada literal en el protocolo v4
  (`veredicto.regla_de_subida`, «copiada literal de 118-C0.v3»). Solo
  `_cumplidos_de` es propio aquí, porque la de 118 fija `motor=NOMBRE_DENSA`
  (la densa VIEJA) y aquí hace falta parametrizarlo con el motor que toque
  (el nuevo, o la densa v2 en los mismos conjuntos).
* **La regla de cierre** (`protocolo.aplicar_regla_de_cierre`), llamada por
  `_cumplidos_de`.

QUÉ ES NUEVO AQUÍ: el motor que corre (`MotorDensaTabM()`, sin parámetros:
no hay «palanca» que traducir), la negación de los sellados, los digests
del caché (que incluyen el motor nuevo Y el protocolo v4 + su enmienda, no
solo `densa.py` como en 118) y el ensamblado final del JSON con el nombre y
forma que pide el encargo de C3.

CÓMO SE LANZA:

    python3 benchmarks/fase0/pasada_119_c3.py --estimar
    python3 benchmarks/fase0/pasada_119_c3.py --humo --salida /tmp/humo.json
    python3 benchmarks/fase0/pasada_119_c3.py --solo diabetes --salida /tmp/prueba.json
    python3 benchmarks/fase0/pasada_119_c3.py
        # LA PASADA ENTERA sobre los 32 no sellados -- horas, a la cola
        # nocturna (decisión del 25-09), nunca a mano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

RUTA_DEL_PROTOCOLO_V4 = _AQUI / "protocolo_119_v4.json"
RUTA_DE_LA_ENMIENDA_1 = _AQUI / "protocolo_119_v4_enmienda_1.json"
RUTA_V2_RESULTADO = p118.RUTA_V2_RESULTADO  # pasada_v2_113_resultado.json

#: El motor NUEVO de este corte — SIN parámetros (a diferencia de
#: `MotorDensaPropia(palanca=...)`): `densa_tabm.py` no tiene variantes.
#: Se comprueba en tiempo de importación que el nombre no ha derivado del
#: que este guion tiene escrito, en vez de confiar en que los dos digan lo
#: mismo para siempre sin que nada lo vigile.
NOMBRE_MOTOR_NUEVO = "matrixai.dense.tabm_cpu"
assert MotorDensaTabM().nombre == NOMBRE_MOTOR_NUEVO, (
    f"MotorDensaTabM().nombre es {MotorDensaTabM().nombre!r}, y este guion tiene escrito "
    f"{NOMBRE_MOTOR_NUEVO!r} -- actualizar la constante, no ignorar la discrepancia")

#: El nombre de la densa VIEJA (v2), tal cual lo usa `pasada_118_palanca.py`
#: -- reutilizado, no reescrito: es la clave que `campo_de_la_comparacion`
#: (importada de allí) usa para saber a QUIÉN sustituye el motor nuevo.
NOMBRE_DENSA_V2 = p118.NOMBRE_DENSA


# ---------------------------------------------------------------------------
# 1. LOS CONJUNTOS: los 32 NO sellados -- los 8 sellados se NIEGAN (son de C4)
# ---------------------------------------------------------------------------

def datasets_de_c3(todos: list, no_sellados: list, *, solo: str | None) -> tuple[list, list | None]:
    """Los conjuntos que esta pasada mide: los 32 no sellados, o el
    subconjunto de `--solo` -- SIEMPRE dentro de los no sellados. Un sellado
    en `--solo` para la pasada: la regla de subida del protocolo v4 los
    reserva para C4 y este corte no los prueba «solo para probar el guion»
    como sí dejaba `pasada_118_palanca.py` -- ver el docstring del módulo."""
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
# 2. EL ENTORNO Y LA CACHÉ
# ---------------------------------------------------------------------------

#: El motor nuevo entero: la clase del motor, la red y su preparación. Un
#: cambio en CUALQUIERA de los tres invalida los intentos de este motor
#: (y aquí solo corre este motor, así que en la práctica es como
#: `_FICHERO_POR_MOTOR`, pero con tres ficheros en vez de uno porque
#: `densa_tabm.py` reparte su receta entre los tres -- encargo explícito:
#: "digests ... que INCLUYAN el motor nuevo, redes/tabm_plr.py,
#: redes/preparacion_tabm.py").
_FICHEROS_DEL_MOTOR_NUEVO = (
    c3._DIR_ENGINES / "motores" / "densa_tabm.py",
    c3._DIR_ENGINES / "redes" / "tabm_plr.py",
    c3._DIR_ENGINES / "redes" / "preparacion_tabm.py",
)


def _ficheros_compartidos() -> tuple[Path, ...]:
    """Los compartidos de C5 (que ya incluyen los de C3) más este guion y
    los DOS ficheros del protocolo (v4 y su enmienda 1) -- encargo: "el
    protocolo y su enmienda" entran en el digest. Es una FUNCIÓN, no una
    tupla congelada al importar, para que una prueba pueda apuntar las
    constantes de ruta a un fichero temporal y ver el digest reaccionar sin
    tocar el protocolo real."""
    return c5._FICHEROS_COMPARTIDOS + (
        Path(__file__).resolve(), RUTA_DEL_PROTOCOLO_V4, RUTA_DE_LA_ENMIENDA_1)


def _digest_de(rutas: tuple[Path, ...]) -> str:
    return hashlib.sha256(
        "".join(c3._digest_fichero(f) for f in rutas).encode()).hexdigest()[:16]


def _digest_entorno() -> str:
    return _digest_de(_ficheros_compartidos())


def _digest_motor_nuevo() -> str:
    return _digest_de(_FICHEROS_DEL_MOTOR_NUEVO)


def _reusable_c3(previo: dict | None, entorno_digest: str, motor_digest: str,
                 wall_seconds: float) -> bool:
    """`c3._reusable` MÁS la condición que esa función no comprueba: el
    estado tiene que contar como medida (`completed`/
    `completed_budget_limited`). Sin esto, un intento `failed` con los
    digests correctos —por ejemplo, uno que perdió por un tope de pared
    ajeno a este código— se reutilizaría para siempre y el relanzamiento,
    que es el mecanismo de recuperación, sería justo el que no lo
    reintenta (la trampa nº1 de «ANTES DE RELANZAR UNA PASADA», CLAUDE.md,
    escrita para `pasada_amplia_101_c5.py` pero el mismo hueco exacto)."""
    if not c3._reusable(previo, entorno_digest, motor_digest, wall_seconds):
        return False
    return previo.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA


def _exigir_que_quepa() -> None:
    c6._exigir_que_quepa()


# ---------------------------------------------------------------------------
# 3. EL VEREDICTO: reutiliza `pasada_118_palanca` para el campo, el veredicto
#    por conjunto y la aritmética de la regla de subida; propio solo lo que
#    118 fija a `NOMBRE_DENSA` (la densa VIEJA) y aquí hace falta parametrizar.
# ---------------------------------------------------------------------------

def _resultados_v2() -> list[dict]:
    return p118._resultados_v2()


def _cumplidos_de(resultados: list[dict], regla: protocolo_mod.ReglaDeCierre, *, motor: str,
                  metrica_por_dataset: dict[str, str], nombres_de_los_conjuntos: list[str]) -> dict:
    """`aplicar_regla_de_cierre`, reutilizada -- filtrando a los conjuntos
    pedidos ANTES de llamarla, mismo motivo que `pasada_118_palanca._cumplidos`
    (que no se puede reusar tal cual: fija `motor=NOMBRE_DENSA`, la densa
    VIEJA, y aquí hace falta poder pedirla también para el motor nuevo)."""
    conjuntos = set(nombres_de_los_conjuntos)
    filtrados = [r for r in resultados if r["dataset"] in conjuntos]
    resultado = protocolo_mod.aplicar_regla_de_cierre(
        filtrados, regla, motor=motor, metrica_por_dataset=metrica_por_dataset,
        datasets_exigidos=nombres_de_los_conjuntos)
    return {k: resultado[k] for k in ("cumplidos", "datasets", "fraccion", "cumple_la_regla")}


def veredicto_de_c3(veredictos_por_conjunto: list[dict], *,
                    cumplidos_con_el_motor_nuevo: dict, cumplidos_de_la_densa_v2: dict) -> dict:
    """El veredicto de C3, con los nombres del encargo -- calculado por
    `pasada_118_palanca.veredicto_de_la_palanca` (importada, no reescrita:
    "sube los cumplidos Y sus mejora superan a sus inferioridad" es la MISMA
    regla, copiada literal en `protocolo_119_v4.json`,
    `veredicto.regla_de_subida`) y con las claves renombradas para que el
    JSON de C3 no hable de «palanca», que aquí no existe."""
    bruto = p118.veredicto_de_la_palanca(
        veredictos_por_conjunto, cumplidos_con_la_palanca=cumplidos_con_el_motor_nuevo,
        cumplidos_de_la_densa_v2=cumplidos_de_la_densa_v2)
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
        "mejoras_superan_inferioridades": bruto["mejoras_superan_inferioridades"],
        "cumple_la_regla_de_subida": bruto["la_palanca_ayuda"],
        "regla_de_subida": (
            "protocolo_119_v4.json, veredicto.regla_de_subida: sube a confirmación (C4) si "
            "cumple las DOS condiciones sobre los 32 no sellados frente a la densa v2 en los "
            "MISMOS conjuntos -- (1) el número de conjuntos donde el motor nuevo está a <=2 "
            "puntos del mejor es MAYOR que el de la densa v2, y (2) sus mejoras superan a sus "
            "inferioridades (diferencia emparejada por repetición y pliegue, bootstrap de "
            "1000 remuestras de pliegues, semilla 0)"),
    }


# ---------------------------------------------------------------------------
# 4. LA CUENTA DE PEOR CASO (parte de --estimar; la calibrada está en
#    `estimar_119_c3.py`... no: vive aquí abajo, en `cmd_estimar`)
# ---------------------------------------------------------------------------

def _imprimir_estimacion_peor_caso(plan: dict) -> None:
    print(f"conjuntos: {plan['n_datasets']} (los 32 no sellados), 1 motor por pliegue "
          f"(matrixai.dense.tabm_cpu)")
    for cubo, e in sorted(plan["por_cubo"].items()):
        print(f"  cubo {cubo:8}: {e['n_datasets']:>2} datasets x {e['folds']} pliegues x "
              f"{e['repeticiones']} rep x {plan['n_motores']} motor = {e['n_intentos']:>5} "
              f"intentos, {e['wall_seconds_por_intento']:.0f}s de tope -> cota "
              f"{e['horas_cota']:.2f} h")
    print(f"  TOTAL: {plan['n_intentos']} intentos")
    print(f"  COTA de peor caso (cada intento agota su presupuesto ENTERO): "
          f"{plan['horas_de_reloj_cota_peor_caso']:.2f} h "
          f"({plan['horas_de_reloj_cota_peor_caso']/24:.2f} dias)")


# ---------------------------------------------------------------------------
# 5. LA PASADA (item 1 del encargo): mide los 32 (o el subconjunto de --solo
#    o de --humo), con caché reanudable, y guarda el resultado + el veredicto.
# ---------------------------------------------------------------------------

#: `--humo`: 2 conjuntos PEQUEÑOS, 1 repetición, 1 pliegue -- "para probar
#: el camino entero" (encargo, item 3), nunca para medir el corte de
#: verdad. Los DOS más baratos de los 32 no sellados por número de filas
#: (`dresses-sales`, 500; `kc2`, 522 -- medido sobre `protocolo_119_v4.json`,
#: no a ojo).
CONJUNTOS_DEL_HUMO = ("kc2", "dresses-sales")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta todos los intentos")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON de salida (por omision, "
                             "pasada_119_c3_resultado.json en este directorio, o "
                             "pasada_119_c3_humo_resultado.json con --humo)")
    parser.add_argument("--estimar", action="store_true",
                        help="NO mide la pasada: escribe estimacion_pasada_119_c3.json a "
                             "partir de la sonda de coste, la aritmetica del protocolo y una "
                             "medida real corta de 1-2 conjuntos pequenos/medianos")
    parser.add_argument("--solo", default=None,
                        help="nombres de conjunto separados por coma, SIEMPRE dentro de los "
                             "32 no sellados (un sellado se NIEGA). Para probar el guion, "
                             "nunca para medir el corte de verdad")
    parser.add_argument("--humo", action="store_true",
                        help=f"2 conjuntos pequeños ({', '.join(CONJUNTOS_DEL_HUMO)}), 1 "
                             f"repeticion, 1 pliegue -- prueba el camino entero, no mide nada "
                             f"del corte. Incompatible con --solo")
    args = parser.parse_args(argv)

    if args.humo and args.solo:
        raise SystemExit("--humo y --solo son incompatibles: --humo YA fija sus propios "
                         "conjuntos (CONJUNTOS_DEL_HUMO)")

    c6.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    no_sellados = c6.datasets_no_sellados(protocolo)

    if args.estimar:
        cmd_estimar(protocolo, todos, no_sellados)
        return

    if args.humo:
        datasets, subconjunto = datasets_de_c3(todos, no_sellados,
                                               solo=",".join(CONJUNTOS_DEL_HUMO))
        salida_por_omision = _AQUI / "pasada_119_c3_humo_resultado.json"
    else:
        datasets, subconjunto = datasets_de_c3(todos, no_sellados, solo=args.solo)
        salida_por_omision = _AQUI / "pasada_119_c3_resultado.json"

    if not datasets:
        raise SystemExit("no queda ningún conjunto que medir")

    _exigir_que_quepa()
    motor = MotorDensaTabM()
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    plan = c6.plan_de_la_pasada(datasets, protocolo, n_motores=1)
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]}")
    print(f"protocolo_119_v4 digest {_digest_fichero_json(RUTA_DEL_PROTOCOLO_V4)[:16]}, "
          f"enmienda_1 digest {_digest_fichero_json(RUTA_DE_LA_ENMIENDA_1)[:16]}")
    _imprimir_estimacion_peor_caso(plan)

    ruta_salida = Path(args.salida) if args.salida else salida_por_omision
    cache_previo, payload_previo = ({}, {}) if args.forzar else c3._cargar_cache(ruta_salida)

    entorno_digest = _digest_entorno()
    digest_del_motor = _digest_motor_nuevo()
    resultados_v2 = _resultados_v2()

    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor_nuevo": digest_del_motor},
        datos_de_entrada={
            **{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
            "protocolo_119_v4": RUTA_DEL_PROTOCOLO_V4,
            "protocolo_119_v4_enmienda_1": RUTA_DE_LA_ENMIENDA_1,
            "pasada_v2_113_resultado": RUTA_V2_RESULTADO,
        })
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
        veredicto = None
        if not parcial:
            campo = p118.campo_de_la_comparacion(resultados_v2, resultados,
                                                 set(nombres_de_los_conjuntos))
            cumplidos_nuevo = _cumplidos_de(
                campo, regla, motor=NOMBRE_MOTOR_NUEVO, metrica_por_dataset=metrica_por_dataset,
                nombres_de_los_conjuntos=nombres_de_los_conjuntos)
            cumplidos_v2 = _cumplidos_de(
                resultados_v2, regla, motor=NOMBRE_DENSA_V2, metrica_por_dataset=metrica_por_dataset,
                nombres_de_los_conjuntos=nombres_de_los_conjuntos)
            veredicto = veredicto_de_c3(veredictos_por_conjunto,
                                       cumplidos_con_el_motor_nuevo=cumplidos_nuevo,
                                       cumplidos_de_la_densa_v2=cumplidos_v2)
        return _componer_y_guardar(
            resultados, veredictos_por_conjunto, veredicto, procedencia, payload_previo,
            ruta_salida, datasets=datasets, protocolo=protocolo,
            metrica_por_dataset=metrica_por_dataset, subconjunto=subconjunto, plan=plan,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados, parcial=parcial,
            es_humo=args.humo)

    for ds in datasets:
        por_id, propuesta, spec, objetivo, predictores, declarada = c5.particiones_base(
            ds, protocolo)
        test_ids = propuesta.plan.observaciones_del_rol("test")
        wall_seconds = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        repeticiones = 1 if args.humo else protocolo.particion.repeticiones_para(ds.cubo)
        metric_id = metrica_por_dataset[ds.nombre]
        print(f"\n=== {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"n={len(por_id)}, test={len(test_ids)}, tope={wall_seconds:.0f}s, "
              f"metrica={metric_id}) ===", flush=True)

        registros_del_dataset: list[dict] = []
        for repeticion in range(repeticiones):
            folds = 1 if args.humo else protocolo.particion.folds
            for pliegue_i in range(folds):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None:
                    continue
                clave = (ds.nombre, motor.nombre, repeticion, pliegue_i)
                previo = cache_previo.get(clave)
                if not args.humo and _reusable_c3(previo, entorno_digest, digest_del_motor,
                                                  wall_seconds):
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
                    crudas_train, crudas_train + crudas_val + crudas_test,
                    objetivo, predictores, motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)

                def hacer(xs):
                    return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

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
                # DECLARAR LO QUE EL MOTOR DIJO QUE PASÓ, no lo que se pidió
                # (encargo: "epocas_ejecutadas, mejor_epoca, parado_por_plazo
                # que el motor declara") -- de `config_efectiva`
                # (`Intento.config_efectiva`, el predictor devuelto por
                # `_ajustar`), NUNCA los pesos: guardar `predictor["pesos"]`
                # entero en cada registro dispararía el JSON a decenas de MB
                # por intento y el encargo no lo pide.
                config = intento.config_efectiva or {}
                entrenamiento_efectivo = config.get("entrenamiento_efectivo")
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                    "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": motor.nombre,
                    "repeticion": repeticion, "pliegue": pliegue_i, "estado": intento.estado,
                    "semilla": protocolo.particion.semillas[repeticion],
                    "presupuesto_wall_s": wall_seconds, "configuracion": c3.CONFIGURACION_UNICA,
                    "metrica_de_cierre": metric_id, "wall_s": round(transcurrido, 3),
                    "metricas": metricas, "tiempo_de_ajuste": recursos.get("wall_seconds"),
                    "cpu_segundos": recursos.get("cpu_seconds"),
                    "rss_pico_mb": recursos.get("peak_ram_mb"),
                    "entrenamiento_efectivo": entrenamiento_efectivo,
                    "motivo": (intento.motivo_del_estado["es"]
                              if intento.motivo_del_estado else None),
                    "entorno_digest": entorno_digest, "motor_digest": digest_del_motor,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False,
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                resultados.append(registro)
                registros_del_dataset.append(registro)
                valor_de_cierre = registro.get(metric_id)
                print(f"  {ds.nombre} rep={repeticion} pliegue={pliegue_i}: "
                      f"estado={intento.estado} {metric_id}={valor_de_cierre} "
                      f"wall={transcurrido:.1f}s entrenamiento_efectivo={entrenamiento_efectivo}",
                      flush=True)
                if (time.perf_counter() - ultimo_punto_de_control[0]
                        >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL):
                    guardar(parcial=True)
            if not args.humo:
                guardar(parcial=True)

        # Se calcula SIEMPRE, también con --humo: con 1 pliegue no hay nada
        # que remuestrear (`veredicto_del_conjunto` lo declara solo, "menos
        # de dos pliegues comunes"), pero es justo lo que --humo tiene que
        # probar -- el camino ENTERO, veredicto incluido, no solo el ajuste.
        v2_densa_del_dataset = [r for r in resultados_v2 if r["dataset"] == ds.nombre]
        nuevo_por_pliegue = c6.medida_por_pliegue(registros_del_dataset, motor.nombre,
                                                  metric_id)
        v2_densa_por_pliegue = c6.medida_por_pliegue(v2_densa_del_dataset, NOMBRE_DENSA_V2,
                                                      metric_id)
        veredicto_ds = p118.veredicto_del_conjunto(
            dataset=ds.nombre, metric_id=metric_id, palanca_por_pliegue=nuevo_por_pliegue,
            v2_densa_por_pliegue=v2_densa_por_pliegue)
        veredictos_por_conjunto.append(veredicto_ds)
        print(f"  VEREDICTO {ds.nombre}: mejora={veredicto_ds['mejora']} "
              f"inferioridad={veredicto_ds['inferioridad']} motivo={veredicto_ds['motivo']}",
              flush=True)
        guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados} reusados, {len(resultados)-reusados} ejecutados) ===")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    v = salida["veredicto"]
    print(f"VEREDICTO 119-C3{' (HUMO -- no cuenta)' if args.humo else ''}: cumplidos "
          f"{v['cumplidos_con_el_motor_nuevo']['cumplidos']} vs densa v2 "
          f"{v['cumplidos_de_la_densa_v2_en_los_mismos_conjuntos']['cumplidos']} "
          f"(sube={v['sube_los_cumplidos']}); mejora={v['n_mejora']} "
          f"inferioridad={v['n_inferioridad']} -> "
          f"{'SUBE a confirmacion' if v['cumple_la_regla_de_subida'] else 'NO SUBE'}")


def _digest_fichero_json(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _componer_y_guardar(resultados, veredictos_por_conjunto, veredicto, procedencia,
                        payload_previo, ruta_salida, *, datasets, protocolo,
                        metrica_por_dataset, subconjunto, plan, total_wall_s, reusados, parcial,
                        es_humo: bool) -> dict:
    """Compone el JSON y lo escribe atómicamente -- mismo patrón que
    `pasada_118_palanca._componer_y_guardar` (temporal + `replace`), con lo
    medido tras CADA conjunto: si se corta, queda lo hecho."""
    procedencias, sin_procedencia = c3._procedencias_citadas(
        resultados, procedencia, (payload_previo.get("procedencias") or {}))

    enmienda = json.loads(RUTA_DE_LA_ENMIENDA_1.read_text(encoding="utf-8"))

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "119-C3",
        "motor": NOMBRE_MOTOR_NUEVO,
        "protocolo_119_v4_digest_sha256": _digest_fichero_json(RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": _digest_fichero_json(RUTA_DE_LA_ENMIENDA_1),
        "allstate_declarado": enmienda.get("allstate_declarado"),
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial, "es_humo": es_humo, "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "plan": plan,
        "criterio_de_los_conjuntos": (
            "los 32 NO sellados del protocolo v4 (identicos en particion/presupuesto/"
            "regla_de_cierre a protocolo_exploratorio_v2.json), o el subconjunto de --solo "
            "o --humo, declarado arriba. Los 8 sellados se niegan (son de C4)"),
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados), "n_reusados": reusados,
        "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
        "resultados": resultados,
        "veredicto_por_conjunto": veredictos_por_conjunto,
        "veredicto": veredicto,
    }
    c3.sellar_la_salida(salida)
    temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta_salida)
    return salida


# ---------------------------------------------------------------------------
# 6. --estimar (item 2 del encargo): la aritmética del protocolo + la sonda
#    de coste de los grandes + una medida real corta de 1-2 pequeños/
#    medianos, con 1 y con 2 procesos en paralelo. Escribe
#    estimacion_pasada_119_c3.json. NO entrena los 32 -- eso es la pasada
#    entera, y esta pasada NO se lanza a mano (va a la cola nocturna).
# ---------------------------------------------------------------------------

RUTA_DE_LA_SONDA = _AQUI / "resultado_sonda_coste_119.json"
RUTA_DE_LA_ESTIMACION = _AQUI / "estimacion_pasada_119_c3.json"

#: k=8, d_block=256 -- la config de la enmienda 1, la que de verdad corre
#: `densa_tabm.py` (K_CABEZAS/D_BLOCK). Se comprueba en tiempo de uso que
#: sigue siendo la del motor, para no estimar con una configuración que ya
#: no es la que se mide.
_K_D_ELEGIDOS = {"k": 8, "d_block": 256}

#: Un conjunto pequeño y uno mediano, de los MÁS BARATOS entre los 32 (menos
#: filas x columnas), para la medida real corta -- `balance-scale` (625 x 5,
#: puramente numérico, sin faltantes: el más simple del cubo pequeño) y
#: `wilt` (4839 x 6, puramente numérico, sin faltantes: el más barato del
#: cubo mediano por columnas, aunque no el de menos filas -- se prefiere
#: pocas columnas porque el coste de TabM+PLR escala con columnas, no solo
#: filas: cada numérica es un embedding PLR y cada categórica uno propio).
CONJUNTOS_DE_LA_MEDIDA_CORTA = {"pequeno": "balance-scale", "mediano": "wilt"}


def _medir_un_intento_real(ds, protocolo, motor) -> dict:
    """UN intento real (repeticion=0, pliegue=0), con el presupuesto REAL de
    su cubo -- ni humo ni --solo: es la «medida real corta» del encargo,
    para calibrar la estimación con algo medido y no solo con aritmética."""
    por_id, propuesta, spec, objetivo, predictores, declarada = c5.particiones_base(
        ds, protocolo)
    test_ids = propuesta.plan.observaciones_del_rol("test")
    wall_seconds = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
    pliegue = propuesta.pliegues.pliegue_de(repeticion=0, pliegue=0)
    if pliegue is None:
        raise SystemExit(f"{ds.nombre}: no hay pliegue (0, 0) en su particion -- no se puede "
                         f"medir la estimacion con este conjunto")
    crudas_train = [por_id[i] for i in pliegue.entrena]
    crudas_val = [por_id[i] for i in pliegue.valida]
    crudas_test = [por_id[i] for i in test_ids]
    transformadas = c3.preparar_para_motor(crudas_train, crudas_train + crudas_val + crudas_test,
                                           objetivo, predictores, motor)
    n_tr, n_va = len(crudas_train), len(crudas_val)

    def hacer(xs):
        return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

    presupuesto = Presupuesto(wall_seconds=wall_seconds, hilos=c3.HILOS_POR_INTENTO,
                              seed=protocolo.particion.semillas[0])
    t0 = time.perf_counter()
    intento = ejecutar_intento_aislado(
        motor, hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]),
        hacer(transformadas[n_tr + n_va:]), spec, presupuesto,
        candidate=f"{motor.nombre}-estimacion-119-c3", split_plan_digest=propuesta.plan.digest(),
        dataset=ds.nombre, pliegue=0, repeticion=0)
    transcurrido = time.perf_counter() - t0
    config = intento.config_efectiva or {}
    return {
        "dataset": ds.nombre, "cubo": ds.cubo, "n_filas": len(por_id),
        "n_columnas": len(predictores), "presupuesto_wall_s": wall_seconds,
        "estado": intento.estado, "wall_s_medido": round(transcurrido, 3),
        "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
    }


def _segundos_por_epoca_de_la_sonda() -> dict:
    """Los `segundos_por_epoca` que la enmienda 1 midió para k=8/d_block=256
    en los 3 grandes más caros -- copiados de su propio JSON, no
    reescritos, y con la fuente declarada."""
    enmienda = json.loads(RUTA_DE_LA_ENMIENDA_1.read_text(encoding="utf-8"))
    medido = enmienda["medido_que_lo_motiva"]
    return {
        "segundos_por_epoca": medido["segundos_por_epoca_k8_d256"],
        "epocas_que_caben_en_450s": medido["epocas_que_caben_en_450_s"],
        "N_epocas_minimas_de_la_paciencia": medido["minimo_de_la_regla"],
        "fuente": f"{RUTA_DE_LA_ENMIENDA_1.name} (medido_que_lo_motiva), a su vez de "
                  f"{medido['fuente']}",
    }


def cmd_estimar(protocolo, todos, no_sellados) -> None:
    _exigir_que_quepa()
    motor = MotorDensaTabM()
    plan = c6.plan_de_la_pasada(no_sellados, protocolo, n_motores=1)
    _imprimir_estimacion_peor_caso(plan)

    por_nombre = {d.nombre: d for d in todos}
    por_cubo_no_sellados: dict[str, list] = {}
    for d in no_sellados:
        por_cubo_no_sellados.setdefault(d.cubo, []).append(d)
    n_datasets_por_cubo = {cubo: len(ds) for cubo, ds in por_cubo_no_sellados.items()}
    folds = protocolo.particion.folds

    print("\nmidiendo 1 intento real por cubo pequeño/mediano (presupuesto real, "
          "repeticion=0 pliegue=0)...", flush=True)
    medidas_reales = {}
    for cubo, nombre_ds in CONJUNTOS_DE_LA_MEDIDA_CORTA.items():
        ds = por_nombre[nombre_ds]
        if ds.sellado:
            raise SystemExit(f"{nombre_ds} esta sellado: la medida real corta no puede usar "
                             f"un sellado")
        print(f"  midiendo {nombre_ds} ({cubo})...", flush=True)
        medidas_reales[cubo] = _medir_un_intento_real(ds, protocolo, motor)
        print(f"  {nombre_ds}: {medidas_reales[cubo]['wall_s_medido']:.1f}s, "
              f"estado={medidas_reales[cubo]['estado']}, "
              f"entrenamiento_efectivo={medidas_reales[cubo]['entrenamiento_efectivo']}",
              flush=True)

    sonda = _segundos_por_epoca_de_la_sonda()
    repeticiones_grande = protocolo.particion.repeticiones_para("grande")
    repeticiones_pm = protocolo.particion.repeticiones_para("pequeno")
    plazo_grande_s = 0.75 * c3.wall_seconds_del_cubo("grande", protocolo)

    estimacion_por_cubo = {}
    for cubo in ("pequeno", "mediano"):
        n_ds = n_datasets_por_cubo.get(cubo, 0)
        n_intentos = n_ds * folds * repeticiones_pm
        s_por_intento = medidas_reales[cubo]["wall_s_medido"]
        estimacion_por_cubo[cubo] = {
            "n_datasets": n_ds, "n_intentos": n_intentos,
            "s_por_intento": s_por_intento,
            "calibrado_con": medidas_reales[cubo]["dataset"],
            "supuesto": (f"cada intento de este cubo tarda lo mismo que {medidas_reales[cubo]['dataset']} "
                        f"midio -- una unica muestra por cubo, no una por dataset"),
            "horas": round(n_intentos * s_por_intento / 3600.0, 3),
        }
    n_ds_grande = n_datasets_por_cubo.get("grande", 0)
    n_intentos_grande = n_ds_grande * folds * repeticiones_grande
    estimacion_por_cubo["grande"] = {
        "n_datasets": n_ds_grande, "n_intentos": n_intentos_grande,
        "s_por_intento": plazo_grande_s,
        "supuesto": (
            "cada intento del cubo grande agota el PLAZO (0.75 x presupuesto = "
            f"{plazo_grande_s:.0f}s), no medido por dataset: la sonda de C1 (enmienda 1) "
            "midio que con k=8/d_block=256, en los 3 conjuntos mas caros del cubo grande "
            "(Allstate, KDDCup09, APSFailure), Allstate NO alcanza siquiera las 17 epocas "
            "minimas de la paciencia dentro del plazo (le caben ~10), y aunque KDDCup09 y "
            "APSFailure si las alcanzan (26 y 22 epocas caben), no hay medida de que la "
            "paciencia corte antes del plazo en ningun caso real -- se toma el plazo entero "
            "como estimacion, y es previsiblemente una SOBRESTIMACION para los grandes con "
            "menos columnas que los 3 medidos por la sonda (los otros 6 del cubo grande no "
            "estan sondados)"),
        "horas": round(n_intentos_grande * plazo_grande_s / 3600.0, 3),
    }

    total_horas_1_proceso = sum(e["horas"] for e in estimacion_por_cubo.values())
    total_horas_2_procesos = total_horas_1_proceso / 2.0

    salida = {
        "corte": "119-C3", "sub_corte": "estimacion (no mide el corte, --estimar)",
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motor": NOMBRE_MOTOR_NUEVO,
        "configuracion_del_motor": _K_D_ELEGIDOS,
        "protocolo_119_v4_digest_sha256": _digest_fichero_json(RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": _digest_fichero_json(RUTA_DE_LA_ENMIENDA_1),
        "plan_peor_caso": plan,
        "que_es_el_plan_peor_caso": (
            "cota literal: cada uno de los 32 x 5 pliegues x repeticiones agota el "
            "presupuesto ENTERO (no el plazo del 0.75) de su cubo. No es una prevision de lo "
            "que va a durar de verdad -- por eso esta estimacion calibra con medidas reales"),
        "sonda_de_coste_119_c1": sonda,
        "medida_real_corta": medidas_reales,
        "estimacion_por_cubo": estimacion_por_cubo,
        "total_horas_1_proceso": round(total_horas_1_proceso, 2),
        "total_dias_1_proceso": round(total_horas_1_proceso / 24.0, 2),
        "total_horas_2_procesos_en_paralelo": round(total_horas_2_procesos, 2),
        "total_dias_2_procesos_en_paralelo": round(total_horas_2_procesos / 24.0, 2),
        "que_supone_2_procesos": (
            "division simple entre 2 de la suma secuencial -- NO una medicion de dos "
            "procesos corriendo a la vez de verdad (este guion no los lanza). Es optimista: "
            "no descuenta contencion de CPU/memoria entre los dos procesos, que el "
            "presupuesto.procesos_en_paralelo=2 del protocolo v4 ya asume posible en esta "
            "maquina (4 CPU fisicas, 8 logicas) pero que aqui no se mide"),
        "aviso": (
            "ESTIMACION, no medicion: el cubo grande en particular se apoya en el plazo "
            "teorico y en solo 3 de los 9 conjuntos grandes sondados. La pasada real (a la "
            "cola nocturna, nunca a mano) es la unica medida que cuenta para el veredicto"),
    }
    RUTA_DE_LA_ESTIMACION.write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nestimacion escrita en {RUTA_DE_LA_ESTIMACION}")
    print(f"  1 proceso:  {total_horas_1_proceso:.2f} h ({total_horas_1_proceso/24:.2f} dias)")
    print(f"  2 procesos: {total_horas_2_procesos:.2f} h ({total_horas_2_procesos/24:.2f} dias)")


if __name__ == "__main__":
    main()
