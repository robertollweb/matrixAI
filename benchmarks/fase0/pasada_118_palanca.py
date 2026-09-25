#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""118 — el arnés de UNA PALANCA de la densa, medida sola contra la densa v2.

Implementa el bloque `veredicto.por_palanca` de `protocolo_118_v3.json`
(contrato 118, `documentacion/118_LA_RED_DENSA_COMPITE_CONTRACT.md`): corre
`matrixai_engines.motores.densa.MotorDensaPropia(palanca=<id>)` sobre los
conjuntos que ESA palanca declara (campo `conjuntos` de su entrada en el
protocolo v3 -- nunca decidido a mano aquí), con las MISMAS particiones,
semillas, presupuesto y métrica de cierre que la pasada v2
(`protocolo_exploratorio_v2.json`, vía `pasada_v2_113.py`), y compara contra
lo que la densa SIN palanca ya midió en esa misma pasada
(`pasada_v2_113_resultado.json`).

QUÉ SE REUTILIZA, Y DE DÓNDE -- nada de lo de abajo se copia, se importa,
tal como pide el encargo:

* **El catálogo, la lectura y la partición** (`pasada_114c6_ensamblado.
  preparar_protocolo_v2`, `pasada_amplia_101_c5.datasets_de_la_pasada`/
  `particiones_base`): la MISMA partición, semillas y estratificación que la
  v2 -- nunca un segundo diseño de partición escrito aquí.
* **La ejecución de un intento en subproceso, con su tope** (`matrixai_
  engines.subproceso.ejecutar_intento_aislado`): el mismo camino que corrió
  la densa en la v2.
* **La caché por intento, la procedencia y la escritura atómica**
  (`pasada_exploratoria_101_c3._reusable`/`_cargar_cache`/
  `procedencia_de_la_medicion`/`_procedencias_citadas`/`sellar_la_salida`/
  `_digest_fichero`/`_FICHERO_POR_MOTOR`): el mismo mecanismo ya auditado.
* **El emparejamiento y el bootstrap del veredicto** (`pasada_114c6_
  ensamblado.diferencias_emparejadas`, `protocolo._intervalo_pareado`): la
  MISMA aritmética de remuestreo de pliegues que usa C6 y `aplicar_regla_
  de_cierre`, 1.000 remuestras de PLIEGUES, semilla 0.
* **La regla de cierre de la cartera** (`protocolo.aplicar_regla_de_cierre`):
  la MISMA función que decide «¿a 2 puntos del mejor en ≥80 %?», llamada
  sobre el «campo» que compone este guion (ver `_campo_de_la_comparacion`).

QUÉ ES NUEVO AQUÍ: el motor que corre (`MotorDensaPropia(palanca=...)`, que
NO pasa por `registro_de_motores.motor_para` -- ese registro no conoce
variantes con parámetros, y no se toca desde este corte) y la composición del
«campo» de la comparación (sustituir, SOLO en los conjuntos medidos aquí, los
registros de la densa v2 por los de la densa con la palanca).

CÓMO SE LANZA:

    python3 benchmarks/fase0/pasada_118_palanca.py --palanca 118-C1.cuantiles --estimar
    python3 benchmarks/fase0/pasada_118_palanca.py --palanca 118-C1.cuantiles \\
        --solo diabetes --salida /tmp/prueba_118_c1.json
    python3 benchmarks/fase0/pasada_118_palanca.py --palanca 118-C1.cuantiles
        # LA PASADA ENTERA de la palanca sobre los no sellados -- horas, a la
        # cola nocturna (decisión del 25-09, contrato 118), nunca a mano.
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
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa import MotorDensaPropia, PALANCA_CUANTILES  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

RUTA_DEL_PROTOCOLO_V3 = _AQUI / "protocolo_118_v3.json"
RUTA_V2_RESULTADO = _AQUI / "pasada_v2_113_resultado.json"

#: El nombre del motor, TAL CUAL lo declara el protocolo v3 y `MotorDensaPropia
#: ().nombre` -- la palanca NO lo cambia (encargo del corte: "nombre sigue
#: siendo matrixai.dense.torch_cpu"). Es lo que ata los registros de este
#: guion con los de la v2 en el mismo campo.
NOMBRE_DENSA = "matrixai.dense.torch_cpu"

#: Las palancas cuyo texto `conjuntos` este arnés sabe traducir a una lista de
#: datasets. Solo 118-C1 tiene código en el motor hoy (118-C2..C5 son
#: `registro_pendiente` en el protocolo v3); añadir un caso aquí es una
#: decisión que se toma cuando esa palanca exista de verdad, no una tabla
#: genérica que adivine.
_TEXTOS_DE_CONJUNTOS_CONOCIDOS = {
    "todos los NO sellados": lambda no_sellados: list(no_sellados),
    "los NO sellados de REGRESIÓN (en los demás la palanca no cambia nada)":
        lambda no_sellados: [d for d in no_sellados if d.tarea == "regression"],
}


# ---------------------------------------------------------------------------
# 1. LA PALANCA: leída del protocolo v3, nunca escrita a mano aquí
# ---------------------------------------------------------------------------

def _protocolo_v3() -> dict:
    return json.loads(RUTA_DEL_PROTOCOLO_V3.read_text(encoding="utf-8"))


def _definicion_de_la_palanca(protocolo_v3: dict, palanca_id: str) -> dict:
    palancas = {p["id"]: p for p in protocolo_v3["palancas"]}
    if palanca_id not in palancas:
        raise SystemExit(f"«{palanca_id}» no está entre las palancas de "
                         f"{RUTA_DEL_PROTOCOLO_V3}: {sorted(palancas)}")
    return palancas[palanca_id]


def datasets_de_la_palanca(protocolo_v3: dict, palanca_id: str,
                           no_sellados: list) -> tuple[list, str]:
    """Los conjuntos que ESTA palanca pide, leídos de su propio campo
    `conjuntos` en el protocolo v3 -- nunca una lista escrita a mano aquí.
    Devuelve la lista y el texto tal cual lo declara el protocolo, para que
    el JSON de salida cite lo mismo, no una paráfrasis."""
    definicion = _definicion_de_la_palanca(protocolo_v3, palanca_id)
    conjuntos_texto = definicion.get("conjuntos")
    if conjuntos_texto is None:
        raise SystemExit(
            f"la palanca «{palanca_id}» todavía no declara 'conjuntos' en el protocolo "
            f"({definicion.get('registro_pendiente', 'sin motivo declarado')})")
    traductor = _TEXTOS_DE_CONJUNTOS_CONOCIDOS.get(conjuntos_texto)
    if traductor is None:
        raise SystemExit(
            f"la palanca «{palanca_id}» declara conjuntos={conjuntos_texto!r} y este arnés "
            f"todavía no sabe traducir ese texto a una lista de datasets -- añadir el caso "
            f"a `_TEXTOS_DE_CONJUNTOS_CONOCIDOS` a mano, nunca adivinar")
    return traductor(no_sellados), conjuntos_texto


# ---------------------------------------------------------------------------
# 2. EL ENTORNO Y LA CACHÉ -- mismo mecanismo que C3/C5/C6, extendido aquí
# ---------------------------------------------------------------------------

#: Un cambio en cualquiera de estos invalida TODA la caché de esta pasada:
#: los compartidos de C5 (que ya incluyen los de C3) más este propio guion.
#: `densa.py` NO va aquí -- va como fichero POR MOTOR (`c3._FICHERO_POR_
#: MOTOR`), igual que en C3/C5/C6, así que solo invalida los intentos de la
#: densa y no obliga a repetir nada que no le toque (aquí solo hay un motor,
#: pero la distinción se mantiene por coherencia con el resto del árbol).
_FICHEROS_COMPARTIDOS = c5._FICHEROS_COMPARTIDOS + (Path(__file__).resolve(),)


def _digest_entorno() -> str:
    return hashlib.sha256(
        "".join(c3._digest_fichero(f) for f in _FICHEROS_COMPARTIDOS).encode()
    ).hexdigest()[:16]


def _exigir_que_quepa() -> None:
    """El mismo guardia de reserva de CPU que C3/C5/C6 -- reutilizado, no
    reescrito."""
    c6._exigir_que_quepa()


# ---------------------------------------------------------------------------
# 3. EL «CAMPO» DE LA COMPARACIÓN -- los motores de la v2, y en el sitio de
#    la densa v2, la densa con la palanca (SOLO en los conjuntos medidos aquí)
# ---------------------------------------------------------------------------

def _resultados_v2() -> list[dict]:
    if not RUTA_V2_RESULTADO.exists():
        raise SystemExit(
            f"no está {RUTA_V2_RESULTADO}: sin la pasada v2 no hay con qué comparar la "
            f"palanca -- este arnés no inventa una densa v2 que no se ha medido")
    payload = json.loads(RUTA_V2_RESULTADO.read_text(encoding="utf-8"))
    return payload["resultados"]


def campo_de_la_comparacion(resultados_v2: list[dict], resultados_de_la_palanca: list[dict],
                            nombres_de_los_conjuntos_medidos: set[str]) -> list[dict]:
    """«El campo = los motores de la v2 con sus resultados de la pasada v2
    (mismas particiones), y en el sitio de la densa v2, la densa con la
    palanca» (protocolo v3, `veredicto.por_palanca.campo`).

    Se sustituye SOLO en los conjuntos que esta pasada ha medido de verdad:
    para un dataset que no está en `nombres_de_los_conjuntos_medidos` (no
    tocado por `--solo`, o porque la palanca no lo pide), la densa sigue
    siendo la de la v2 -- no desaparece del campo, que dejaría a ese dataset
    sin densa con la que comparar nada.
    """
    otros_motores = [r for r in resultados_v2 if r["motor"] != NOMBRE_DENSA]
    densa_v2_no_tocada = [r for r in resultados_v2
                          if r["motor"] == NOMBRE_DENSA
                          and r["dataset"] not in nombres_de_los_conjuntos_medidos]
    return otros_motores + densa_v2_no_tocada + list(resultados_de_la_palanca)


def _cumplidos(resultados: list[dict], regla: protocolo_mod.ReglaDeCierre, *,
              metrica_por_dataset: dict[str, str], nombres_de_los_conjuntos: list[str]) -> dict:
    """`aplicar_regla_de_cierre`, reutilizada, PERO filtrando `resultados` a
    los conjuntos pedidos antes de llamarla: sin el filtro, `aplicar_regla_
    de_cierre` cuenta TODOS los datasets que aparecen en `resultados` (los 40
    de la v2, vía `otros_motores`), no solo los que esta pasada mide -- y
    «cumplidos» dejaría de ser «en los MISMOS conjuntos», que es justo lo que
    pide el protocolo. Filtrar por dataset no cambia el «mejor» de ningún
    dataset (esa cuenta ya mira solo sus propias filas)."""
    conjuntos = set(nombres_de_los_conjuntos)
    filtrados = [r for r in resultados if r["dataset"] in conjuntos]
    resultado = protocolo_mod.aplicar_regla_de_cierre(
        filtrados, regla, motor=NOMBRE_DENSA, metrica_por_dataset=metrica_por_dataset,
        datasets_exigidos=nombres_de_los_conjuntos)
    return {k: resultado[k] for k in ("cumplidos", "datasets", "fraccion", "cumple_la_regla")}


# ---------------------------------------------------------------------------
# 4. EL VEREDICTO POR CONJUNTO: diferencia emparejada (palanca − densa v2)
# ---------------------------------------------------------------------------

def veredicto_del_conjunto(*, dataset: str, metric_id: str,
                           palanca_por_pliegue: dict[tuple, float],
                           v2_densa_por_pliegue: dict[tuple, float]) -> dict:
    """«Por conjunto, la diferencia emparejada (palanca − densa v2) por
    repetición y pliegue con bootstrap de pliegues (1.000, semilla 0) ->
    mejora/inferioridad» (protocolo v3). Reutiliza `c6.diferencias_
    emparejadas` (resta en los pliegues COMUNES) y `protocolo._intervalo_
    pareado` (el mismo bootstrap que usa C6 y `aplicar_regla_de_cierre`).

    «Si un intento de la densa v2 no existe en el artefacto para algún
    pliegue, dilo y no inventes» (encargo): `faltan_en_v2` lista, por
    repetición y pliegue, lo que esta pasada SÍ midió y la v2 no tiene --
    esos pliegues quedan fuera del emparejamiento (la intersección de
    `diferencias_emparejadas` ya los excluye sola), y aquí se declaran en vez
    de desaparecer en silencio.
    """
    faltan_en_v2 = sorted(set(palanca_por_pliegue) - set(v2_densa_por_pliegue))
    diferencias, comunes = c6.diferencias_emparejadas(palanca_por_pliegue, v2_densa_por_pliegue)
    base = {
        "dataset": dataset, "metrica": metric_id,
        "n_pliegues_de_la_palanca": len(palanca_por_pliegue),
        "n_pliegues_de_la_densa_v2": len(v2_densa_por_pliegue),
        "pliegues_de_la_palanca_sin_intento_de_la_densa_v2": [
            {"repeticion": r, "pliegue": p} for r, p in faltan_en_v2],
    }
    if len(diferencias) < 2:
        return {**base, "mejora": False, "inferioridad": False, "intervalo": None,
                "n_pliegues_comunes": len(comunes),
                "motivo": f"menos de dos pliegues comunes con la densa v2 ({len(comunes)}): "
                          f"no hay nada que remuestrear"}
    limites = protocolo_mod._intervalo_pareado(
        diferencias, semilla=c6.SEMILLA_DEL_BOOTSTRAP, remuestras=c6.REMUESTRAS_DEL_BOOTSTRAP)
    if limites is None:
        return {**base, "mejora": False, "inferioridad": False, "intervalo": None,
                "n_pliegues_comunes": len(comunes), "motivo": "el bootstrap no devolvio intervalo"}
    bajo, alto = limites
    mejora = bajo > 0.0
    inferioridad = alto < 0.0
    motivo = None
    if not mejora and not inferioridad:
        motivo = f"el intervalo [{bajo:.5f}, {alto:.5f}] no excluye el cero"
    return {
        **base, "mejora": mejora, "inferioridad": inferioridad, "motivo": motivo,
        "n_pliegues_comunes": len(comunes),
        "intervalo": {
            "bajo": bajo, "alto": alto, "nivel": 0.95, "diferencia": "palanca - densa v2",
            "emparejado_por": "repeticion y pliegue",
            "remuestreo": (f"bootstrap de percentiles, {c6.REMUESTRAS_DEL_BOOTSTRAP} remuestras "
                           f"de PLIEGUES, semilla {c6.SEMILLA_DEL_BOOTSTRAP} "
                           f"(protocolo._intervalo_pareado, reutilizada)"),
        },
    }


def veredicto_de_la_palanca(veredictos_por_conjunto: list[dict], *,
                            cumplidos_con_la_palanca: dict,
                            cumplidos_de_la_densa_v2: dict) -> dict:
    """«La_palanca_ayuda_si: sube los cumplidos Y sus mejora superan a sus
    inferioridad» (protocolo v3, literal)."""
    n_mejora = sum(1 for v in veredictos_por_conjunto if v["mejora"])
    n_inferioridad = sum(1 for v in veredictos_por_conjunto if v["inferioridad"])
    sube_los_cumplidos = (cumplidos_con_la_palanca["cumplidos"]
                          > cumplidos_de_la_densa_v2["cumplidos"])
    mejoras_superan_inferioridades = n_mejora > n_inferioridad
    return {
        "campo": ("los motores de la v2 (pasada_v2_113_resultado.json) con sus resultados de "
                 "esa pasada, y en el sitio de la densa v2, la densa con la palanca"),
        "cumplidos_con_la_palanca": cumplidos_con_la_palanca,
        "cumplidos_de_la_densa_v2_en_los_mismos_conjuntos": cumplidos_de_la_densa_v2,
        "sube_los_cumplidos": sube_los_cumplidos,
        "n_conjuntos": len(veredictos_por_conjunto),
        "n_mejora": n_mejora, "n_inferioridad": n_inferioridad,
        "mejoras_superan_inferioridades": mejoras_superan_inferioridades,
        "la_palanca_ayuda": sube_los_cumplidos and mejoras_superan_inferioridades,
    }


# ---------------------------------------------------------------------------
# 5. LA CUENTA DE LO QUE VA A COSTAR (--estimar)
# ---------------------------------------------------------------------------

def _imprimir_estimacion(plan: dict) -> None:
    print(f"conjuntos que pide la palanca: {plan['n_datasets']}, 1 motor por pliegue "
          f"(la densa con la palanca)")
    for cubo, e in sorted(plan["por_cubo"].items()):
        print(f"  cubo {cubo:8}: {e['n_datasets']:>2} datasets x {e['folds']} pliegues x "
              f"{e['repeticiones']} rep x {plan['n_motores']} motor = {e['n_intentos']:>5} "
              f"intentos, {e['wall_seconds_por_intento']:.0f}s de tope -> cota "
              f"{e['horas_cota']:.2f} h")
    print(f"  TOTAL: {plan['n_intentos']} intentos")
    print(f"  COTA de peor caso: {plan['horas_de_reloj_cota_peor_caso']:.2f} h "
          f"({plan['horas_de_reloj_cota_peor_caso']/24:.2f} dias)")
    print(f"\n  {plan['que_es_la_cota']}")
    print("\n  Esto es SOLO la aritmetica del peor caso. Cada pasada, a la cola nocturna "
          "con su duracion estimada escrita antes de encolar (decision del 25-09).")


# ---------------------------------------------------------------------------
# 6. LA PASADA
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--palanca", required=True,
                        help="id de la palanca del protocolo v3, p.ej. 118-C1.cuantiles")
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta todos los intentos")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON de salida (por omision, una por palanca en este "
                             "directorio)")
    parser.add_argument("--estimar", action="store_true",
                        help="imprime la cuenta de coste con su aritmetica y NO mide nada")
    parser.add_argument("--solo", default=None,
                        help="nombres de conjunto separados por coma: corre SOLO esos. "
                             "Para probar el guion, nunca para medir el corte de verdad")
    args = parser.parse_args(argv)

    # Falla RAPIDO si el motor no sabe aplicar esta palanca -- antes de tocar
    # ningun ARFF ni el protocolo.
    if args.palanca != PALANCA_CUANTILES:
        MotorDensaPropia(palanca=args.palanca)  # deja que ErrorDeMotor lo diga

    protocolo_v3 = _protocolo_v3()

    c6.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    no_sellados = c6.datasets_no_sellados(protocolo)
    datasets, conjuntos_texto = datasets_de_la_palanca(protocolo_v3, args.palanca, no_sellados)

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
                  f"es una prueba del guion, NO CUENTA como la confirmacion en sellados del "
                  f"contrato 118 ('una sola vez y despues')", flush=True)
        fuera_de_la_palanca = [n for n in pedidos if por_nombre[n] not in datasets_de_la_palanca(
            protocolo_v3, args.palanca, no_sellados)[0] and not por_nombre[n].sellado]
        if fuera_de_la_palanca:
            print(f"AVISO: --solo incluye {fuera_de_la_palanca}, que la palanca «{args.palanca}» "
                  f"(conjuntos={conjuntos_texto!r}) no pediria por su cuenta -- se corren "
                  f"igual porque se pidieron explicitamente", flush=True)

    if not datasets:
        raise SystemExit(f"la palanca «{args.palanca}» (conjuntos={conjuntos_texto!r}) no deja "
                         f"ningun dataset que medir")

    _exigir_que_quepa()
    motor = MotorDensaPropia(palanca=args.palanca)
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    plan = c6.plan_de_la_pasada(datasets, protocolo, n_motores=1)
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]}")
    print(f"palanca {args.palanca!r} (protocolo v3 digest "
          f"{protocolo_v3.get('digest_sha256', '?')[:16]}), conjuntos: {conjuntos_texto!r}")
    _imprimir_estimacion(plan) if args.estimar else None
    if args.estimar:
        return

    ruta_salida = Path(args.salida) if args.salida else (
        _AQUI / f"resultado_118_palanca_{args.palanca.replace('.', '_').replace('-', '_')}.json")
    cache_previo, payload_previo = ({}, {}) if args.forzar else c3._cargar_cache(ruta_salida)
    if payload_previo and payload_previo.get("palanca") != args.palanca:
        print(f"AVISO: {ruta_salida} tiene una cache de otra palanca "
              f"({payload_previo.get('palanca')!r}) -- se ignora entera, no se mezcla",
              flush=True)
        cache_previo, payload_previo = {}, {}

    entorno_digest = _digest_entorno()
    digest_del_motor = c3._digest_fichero(c3._FICHERO_POR_MOTOR[NOMBRE_DENSA])
    resultados_v2 = _resultados_v2()

    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "densa": digest_del_motor},
        datos_de_entrada={
            **{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
            "protocolo_118_v3": RUTA_DEL_PROTOCOLO_V3,
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
            campo = campo_de_la_comparacion(resultados_v2, resultados, set(nombres_de_los_conjuntos))
            cumplidos_palanca = _cumplidos(campo, regla, metrica_por_dataset=metrica_por_dataset,
                                           nombres_de_los_conjuntos=nombres_de_los_conjuntos)
            cumplidos_v2 = _cumplidos(resultados_v2, regla, metrica_por_dataset=metrica_por_dataset,
                                      nombres_de_los_conjuntos=nombres_de_los_conjuntos)
            veredicto = veredicto_de_la_palanca(
                veredictos_por_conjunto, cumplidos_con_la_palanca=cumplidos_palanca,
                cumplidos_de_la_densa_v2=cumplidos_v2)
        return _componer_y_guardar(
            resultados, veredictos_por_conjunto, veredicto, procedencia, payload_previo,
            ruta_salida, datasets=datasets, protocolo=protocolo, protocolo_v3=protocolo_v3,
            palanca=args.palanca, conjuntos_texto=conjuntos_texto,
            metrica_por_dataset=metrica_por_dataset, subconjunto=subconjunto, plan=plan,
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
                clave = (ds.nombre, motor.nombre, repeticion, pliegue_i)
                previo = cache_previo.get(clave)
                if c3._reusable(previo, entorno_digest, digest_del_motor, wall_seconds):
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
                    candidate=f"{motor.nombre}-{args.palanca}-{c3.CONFIGURACION_UNICA}",
                    split_plan_digest=propuesta.plan.digest(),
                    dataset=ds.nombre, pliegue=pliegue_i, repeticion=repeticion)
                transcurrido = time.perf_counter() - t0

                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                    "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": motor.nombre, "palanca": args.palanca,
                    "repeticion": repeticion, "pliegue": pliegue_i, "estado": intento.estado,
                    "semilla": protocolo.particion.semillas[repeticion],
                    "presupuesto_wall_s": wall_seconds, "configuracion": c3.CONFIGURACION_UNICA,
                    "metrica_de_cierre": metric_id, "wall_s": round(transcurrido, 3),
                    "metricas": metricas, "tiempo_de_ajuste": recursos.get("wall_seconds"),
                    "cpu_segundos": recursos.get("cpu_seconds"),
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
                      f"wall={transcurrido:.1f}s", flush=True)
                if (time.perf_counter() - ultimo_punto_de_control[0]
                        >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL):
                    guardar(parcial=True)
            guardar(parcial=True)

        palanca_por_pliegue = c6.medida_por_pliegue(registros_del_dataset, motor.nombre, metric_id)
        v2_densa_del_dataset = [r for r in resultados_v2 if r["dataset"] == ds.nombre]
        v2_densa_por_pliegue = c6.medida_por_pliegue(v2_densa_del_dataset, NOMBRE_DENSA, metric_id)
        veredicto_ds = veredicto_del_conjunto(
            dataset=ds.nombre, metric_id=metric_id, palanca_por_pliegue=palanca_por_pliegue,
            v2_densa_por_pliegue=v2_densa_por_pliegue)
        veredictos_por_conjunto.append(veredicto_ds)
        if veredicto_ds["pliegues_de_la_palanca_sin_intento_de_la_densa_v2"]:
            print(f"  AVISO: {ds.nombre} tiene "
                  f"{len(veredicto_ds['pliegues_de_la_palanca_sin_intento_de_la_densa_v2'])} "
                  f"pliegue(s) medido(s) aqui SIN intento de la densa v2 en el artefacto -- "
                  f"quedan fuera del emparejamiento, no se inventan", flush=True)
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
    print(f"VEREDICTO DE LA PALANCA {args.palanca}: cumplidos "
          f"{v['cumplidos_con_la_palanca']['cumplidos']} vs densa v2 "
          f"{v['cumplidos_de_la_densa_v2_en_los_mismos_conjuntos']['cumplidos']} "
          f"(sube={v['sube_los_cumplidos']}); mejora={v['n_mejora']} "
          f"inferioridad={v['n_inferioridad']} -> "
          f"{'AYUDA' if v['la_palanca_ayuda'] else 'NO AYUDA'}")


def _componer_y_guardar(resultados, veredictos_por_conjunto, veredicto, procedencia,
                        payload_previo, ruta_salida, *, datasets, protocolo, protocolo_v3,
                        palanca, conjuntos_texto, metrica_por_dataset, subconjunto, plan,
                        total_wall_s, reusados, parcial) -> dict:
    """Compone el JSON y lo escribe atomicamente -- mismo patron que C3/C5/C6
    (`_procedencias_citadas`, `sellar_la_salida`, temporal + `replace`), con
    lo medido tras CADA conjunto: si se corta, queda lo hecho."""
    procedencias, sin_procedencia = c3._procedencias_citadas(
        resultados, procedencia, (payload_previo.get("procedencias") or {}))

    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "118-C1" if palanca == PALANCA_CUANTILES else "118",
        "palanca": palanca,
        "conjuntos_que_pide_la_palanca": conjuntos_texto,
        "motor": NOMBRE_DENSA,
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial, "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "protocolo_118_v3_digest_sha256": protocolo_v3.get("digest_sha256"),
        "plan": plan,
        "criterio_de_los_conjuntos": (
            "los que declara la propia palanca en protocolo_118_v3.json (campo 'conjuntos'), "
            "o el subconjunto pedido por --solo, declarado arriba"),
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


if __name__ == "__main__":
    main()
