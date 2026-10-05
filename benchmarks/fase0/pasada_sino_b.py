#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La palanca de las sí/no de la densa (opción B, Roberto 2026-10-04) contra la densa de HOY.

Implementa `protocolo_sino_b.json` (registrado ANTES de medir, `generar_protocolo_sino_b.py`):
corre las DOS variantes —`MotorDensaPropia()` y `MotorDensaPropia(palanca=PALANCA_SINO_
CATEGORICA)`— sobre los MISMOS pliegues de la v2 (partición, semillas, repeticiones, tope por
cubo y métrica de cierre), en los conjuntos que declara el protocolo, y aplica su regla: B
entra si ningún conjunto da «inferioridad» y B completa todo lo que completa la de hoy.

QUÉ SE REUTILIZA, sin copiarlo (lo mismo que `pasada_118_palanca.py`): el catálogo, la lectura
y la partición de la v2 (`pasada_114c6_ensamblado.preparar_protocolo_v2`, `pasada_amplia_101_
c5.particiones_base`), la preparación por motor (`pasada_exploratoria_101_c3.preparar_para_
motor`), el intento aislado con su tope (`ejecutar_intento_aislado`), la procedencia y el sello
de la salida, y la aritmética del veredicto (`diferencias_emparejadas` + `_intervalo_pareado`).

LO QUE NO SE REUTILIZA, a propósito: la caché de `pasada_exploratoria_101_c3`. Va por
(conjunto, motor, repetición, pliegue) y aquí las dos variantes son el MISMO motor; y reutiliza
también los intentos FALLIDOS (CLAUDE.md, «ANTES DE RELANZAR UNA PASADA»). Esta va por variante
y solo reutiliza un intento `completed` medido con el mismo código y el mismo tope.

    python3 benchmarks/fase0/pasada_sino_b.py --estimar
    python3 benchmarks/fase0/pasada_sino_b.py --humo --salida <ruta fuera>   # humo: 2 intentos
    python3 benchmarks/fase0/pasada_sino_b.py --salida <ruta>    # LA PASADA: cola nocturna
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

from matrixai_engines.motores.densa import MotorDensaPropia, PALANCA_SINO_CATEGORICA  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

RUTA_DEL_PROTOCOLO = _AQUI / "protocolo_sino_b.json"
RUTA_V2_RESULTADO = _AQUI / "pasada_v2_113_resultado.json"
NOMBRE_DENSA = "matrixai.dense.torch_cpu"

#: Las dos variantes, en el orden en que se corren en cada pliegue (la de hoy primero).
VARIANTES = {"hoy": None, "B": PALANCA_SINO_CATEGORICA}

#: Un cambio en cualquiera de estos invalida la caché ENTERA de esta pasada; `densa.py` va
#: aparte, como fichero del motor (igual que en C3/C5/C6/118).
_FICHEROS_COMPARTIDOS = c5._FICHEROS_COMPARTIDOS + (Path(__file__).resolve(), RUTA_DEL_PROTOCOLO)


def protocolo_sino_b() -> dict:
    protocolo = json.loads(RUTA_DEL_PROTOCOLO.read_text(encoding="utf-8"))
    sin_digest = {k: v for k, v in protocolo.items() if k != "digest_sha256"}
    canonico = json.dumps(sin_digest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonico.encode("utf-8")).hexdigest() != protocolo.get("digest_sha256"):
        raise SystemExit(f"{RUTA_DEL_PROTOCOLO} no cuadra con su digest: alguien lo tocó después "
                         "de registrarlo; la regla no se cambia a mano")
    return protocolo


def _digest_entorno() -> str:
    return hashlib.sha256(
        "".join(c3._digest_fichero(f) for f in _FICHEROS_COMPARTIDOS).encode()
    ).hexdigest()[:16]


def _cache(ruta: Path, entorno_digest: str, motor_digest: str) -> dict[tuple, dict]:
    """Los intentos de una salida anterior que se pueden reutilizar: SOLO `completed`, con el
    mismo código (entorno y motor) y el mismo tope. Uno fallido se vuelve a correr siempre."""
    if not ruta.exists():
        return {}
    previos = json.loads(ruta.read_text(encoding="utf-8")).get("resultados") or []
    return {(r["dataset"], r["variante"], r["repeticion"], r["pliegue"]): r for r in previos
            if r.get("estado") == "completed"
            and r.get("entorno_digest") == entorno_digest
            and r.get("motor_digest") == motor_digest}


# ---------------------------------------------------------------------------
# EL VEREDICTO — la regla de `protocolo_sino_b.json`, aplicada tal cual
# ---------------------------------------------------------------------------

def veredicto_del_conjunto(*, dataset: str, metric_id: str, hoy: list[dict],
                           b: list[dict]) -> dict:
    """«Diferencia emparejada (B − hoy) por repetición y pliegue; intervalo al 95 % por
    bootstrap de PLIEGUES (1.000, semilla 0): «inferioridad» si entero por debajo de 0, «mejora»
    si entero por encima» — y la completitud: un intento que la de hoy completa y B no."""
    hoy_por_pliegue = c6.medida_por_pliegue(hoy, NOMBRE_DENSA, metric_id)
    b_por_pliegue = c6.medida_por_pliegue(b, NOMBRE_DENSA, metric_id)
    completos = lambda rs: {(r["repeticion"], r["pliegue"]) for r in rs  # noqa: E731
                            if r.get("estado") == "completed"}
    b_deja_de_completar = sorted(completos(hoy) - completos(b))
    diferencias, comunes = c6.diferencias_emparejadas(b_por_pliegue, hoy_por_pliegue)
    base = {
        "dataset": dataset, "metrica": metric_id,
        "n_pliegues_hoy": len(hoy_por_pliegue), "n_pliegues_B": len(b_por_pliegue),
        "n_pliegues_comunes": len(comunes),
        "media_hoy": (sum(hoy_por_pliegue.values()) / len(hoy_por_pliegue)
                      if hoy_por_pliegue else None),
        "media_B": sum(b_por_pliegue.values()) / len(b_por_pliegue) if b_por_pliegue else None,
        "diferencia_media": sum(diferencias) / len(diferencias) if diferencias else None,
        "B_deja_de_completar": [{"repeticion": r, "pliegue": p} for r, p in b_deja_de_completar],
    }
    if len(diferencias) < 2:
        return {**base, "mejora": False, "inferioridad": False, "intervalo": None,
                "motivo": f"menos de dos pliegues comunes ({len(comunes)}): nada que remuestrear"}
    limites = protocolo_mod._intervalo_pareado(
        diferencias, semilla=c6.SEMILLA_DEL_BOOTSTRAP, remuestras=c6.REMUESTRAS_DEL_BOOTSTRAP)
    if limites is None:
        return {**base, "mejora": False, "inferioridad": False, "intervalo": None,
                "motivo": "el bootstrap no devolvió intervalo"}
    bajo, alto = limites
    return {**base, "mejora": bajo > 0.0, "inferioridad": alto < 0.0,
            "motivo": None if (bajo > 0.0 or alto < 0.0) else
            f"el intervalo [{bajo:.5f}, {alto:.5f}] no excluye el cero",
            "intervalo": {"bajo": bajo, "alto": alto, "nivel": 0.95, "diferencia": "B - hoy",
                          "emparejado_por": "repeticion y pliegue",
                          "remuestreo": (f"bootstrap de percentiles, {c6.REMUESTRAS_DEL_BOOTSTRAP} "
                                         f"remuestras de PLIEGUES, semilla "
                                         f"{c6.SEMILLA_DEL_BOOTSTRAP}")}}


def veredicto_de_la_pasada(por_conjunto: list[dict], conjuntos_del_protocolo: list[str]) -> dict:
    """«Entra si NINGÚN conjunto da «inferioridad» Y B completa todo lo que completa la de
    hoy». Con un conjunto del protocolo SIN medir no hay veredicto: no se decide a medias."""
    medidos = {v["dataset"] for v in por_conjunto}
    faltan = [c for c in conjuntos_del_protocolo if c not in medidos]
    inferioridades = [v["dataset"] for v in por_conjunto if v["inferioridad"]]
    deja_de_completar = [v["dataset"] for v in por_conjunto if v["B_deja_de_completar"]]
    return {
        "conjuntos_sin_medir": faltan,
        "inferioridad_en": inferioridades,
        "mejora_en": [v["dataset"] for v in por_conjunto if v["mejora"]],
        "B_deja_de_completar_en": deja_de_completar,
        "entra": (None if faltan else (not inferioridades and not deja_de_completar)),
        "regla": "entra si NINGÚN conjunto da «inferioridad» Y B completa todo lo que completa hoy",
    }


# ---------------------------------------------------------------------------
# LA PASADA
# ---------------------------------------------------------------------------

def _estimacion(datasets, protocolo) -> dict:
    v2 = json.loads(RUTA_V2_RESULTADO.read_text(encoding="utf-8"))["resultados"]
    por_conjunto = {d.nombre: round(sum(r["wall_s"] for r in v2
                                        if r["dataset"] == d.nombre and r["motor"] == NOMBRE_DENSA), 1)
                    for d in datasets}
    plan = c6.plan_de_la_pasada(datasets, protocolo, n_motores=len(VARIANTES))
    return {"v2_densa_wall_s_por_conjunto": por_conjunto,
            "estimado_s": round(sum(por_conjunto.values()) * len(VARIANTES) * 1.5),
            "cota_peor_caso_s": round(plan["horas_de_reloj_cota_peor_caso"] * 3600),
            "plan": plan}


def _exigir_la_densa_de_la_medida(version: str) -> None:
    """Esta pasada mide la palanca de las sí/no CONTRA la densa de entonces (la 1.1.x). Desde la 1.2.0
    (adopción, motores) la «de hoy» YA ES la B: relanzarla compararía B contra B y daría «entra» por
    construcción, sin medir nada (auditoría de la adopción, M-2). Se niega, con su motivo."""
    if not version.startswith("1.1."):
        raise SystemExit(
            f"pasada_sino_b mide la palanca contra la densa 1.1.x y aquí está la {version}: desde la 1.2.0 la "
            "de hoy ya es la B (adoptada tras el veredicto «entra», resultado_sino_b.json). Relanzarla no "
            "mediría nada; para volver a medir, hace falta el código de motores de la medida (bd3a436).")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", default=str(_AQUI / "resultado_sino_b.json"))
    parser.add_argument("--estimar", action="store_true",
                        help="imprime la estimación y la cota y NO mide nada")
    parser.add_argument("--solo", default=None,
                        help="conjuntos separados por coma: SOLO para probar el guion")
    parser.add_argument("--humo", action="store_true",
                        help="SOLO el primer conjunto, repetición 0 y pliegue 0, las dos "
                             "variantes: para probar el guion entero en un minuto, nunca para medir")
    args = parser.parse_args(argv)
    if not args.estimar:
        _exigir_la_densa_de_la_medida(MotorDensaPropia().version)

    registrado = protocolo_sino_b()
    c6.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    if protocolo.digest() != registrado["base"]["digest_sha256"]:
        raise SystemExit(f"el protocolo v2 cargado ({protocolo.digest()[:16]}) no es la base "
                         f"registrada ({registrado['base']['digest_sha256'][:16]})")
    por_nombre = {d.nombre: d for d in c5.datasets_de_la_pasada(protocolo)}
    pedidos = ([n.strip() for n in args.solo.split(",") if n.strip()] if args.solo
               else list(registrado["conjuntos"]))
    desconocidos = [n for n in pedidos if n not in por_nombre]
    if desconocidos:
        raise SystemExit(f"conjuntos que no están en el protocolo v2: {desconocidos}")
    sellados = [n for n in pedidos if por_nombre[n].sellado]
    if sellados:
        raise SystemExit(f"{sellados} está(n) SELLADO(S): el protocolo los excluye "
                         f"({registrado['excluidos']})")
    datasets = [por_nombre[n] for n in pedidos][:1 if args.humo else None]
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        # La regla resta «B − hoy» tal cual: solo vale si MÁS es MEJOR (AUROC y R², las de hoy).
        if not c6.direccion_de(metric_id):
            raise SystemExit(f"la métrica de cierre {metric_id!r} es de «menos es mejor» y la "
                             "regla registrada resta B − hoy sin girar el signo")

    estimacion = _estimacion(datasets, protocolo)
    print(f"protocolo sino-b {registrado['digest_sha256'][:16]} sobre la v2 "
          f"{protocolo.digest()[:16]}; conjuntos {pedidos}; variantes {list(VARIANTES)}")
    print(f"  v2 (la densa, por conjunto): {estimacion['v2_densa_wall_s_por_conjunto']} s")
    print(f"  ESTIMADO: {estimacion['estimado_s']} s ({estimacion['estimado_s'] / 3600:.2f} h) "
          f"— COTA de peor caso: {estimacion['cota_peor_caso_s']} s "
          f"({estimacion['cota_peor_caso_s'] / 3600:.2f} h)")
    if args.estimar:
        return

    c6._exigir_que_quepa()
    ruta_salida = Path(args.salida)
    entorno_digest = _digest_entorno()
    motor_digest = c3._digest_fichero(c3._FICHERO_POR_MOTOR[NOMBRE_DENSA])
    cache = _cache(ruta_salida, entorno_digest, motor_digest)
    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "densa": motor_digest},
        datos_de_entrada={**{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
                          "protocolo_sino_b": RUTA_DEL_PROTOCOLO})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    motores = {v: MotorDensaPropia(palanca=p) for v, p in VARIANTES.items()}

    resultados: list[dict] = []
    por_conjunto: list[dict] = []
    reusados = 0
    inicio = time.perf_counter()

    def guardar(parcial: bool) -> dict:
        veredicto = (None if parcial else
                     veredicto_de_la_pasada(por_conjunto, list(registrado["conjuntos"])))
        procedencias, sin_procedencia = c3._procedencias_citadas(resultados, procedencia, {})
        salida = {
            "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "corte": "sino-b", "motor": NOMBRE_DENSA, "variantes": {k: v for k, v in VARIANTES.items()},
            "protocolo_sino_b_digest_sha256": registrado["digest_sha256"],
            "protocolo_v2_digest_sha256": protocolo.digest(),
            "procedencia": procedencia, "procedencias": procedencias,
            "n_intentos_sin_procedencia": sin_procedencia,
            "parcial": parcial, "es_subconjunto_de_prueba": bool(args.solo or args.humo),
            "conjuntos_pedidos": pedidos, "estimacion": estimacion,
            "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
            "total_wall_s": round(time.perf_counter() - inicio, 1),
            "n_intentos": len(resultados), "n_reusados": reusados,
            "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
            "resultados": resultados, "veredicto_por_conjunto": por_conjunto,
            "veredicto": veredicto,
        }
        c3.sellar_la_salida(salida)
        temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
        temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        temporal.replace(ruta_salida)
        return salida

    for ds in datasets:
        por_id, propuesta, spec, objetivo, predictores, _declarada = c5.particiones_base(ds, protocolo)
        test_ids = propuesta.plan.observaciones_del_rol("test")
        wall_seconds = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        metric_id = metrica_por_dataset[ds.nombre]
        print(f"\n=== {ds.nombre} ({ds.tarea}, cubo={ds.cubo}, n={len(por_id)}, "
              f"tope={wall_seconds:.0f}s, métrica={metric_id}) ===", flush=True)
        del_conjunto = {v: [] for v in VARIANTES}
        for repeticion in range(repeticiones):
            for pliegue_i in range(protocolo.particion.folds):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None or (args.humo and (repeticion, pliegue_i) != (0, 0)):
                    continue
                for variante, motor in motores.items():
                    previo = cache.get((ds.nombre, variante, repeticion, pliegue_i))
                    if previo is not None and previo.get("presupuesto_wall_s") == wall_seconds:
                        registro = dict(previo, reusado=True)
                        reusados += 1
                    else:
                        crudas_train = [por_id[i] for i in pliegue.entrena]
                        crudas_val = [por_id[i] for i in pliegue.valida]
                        crudas_test = [por_id[i] for i in test_ids]
                        transformadas = c3.preparar_para_motor(
                            crudas_train, crudas_train + crudas_val + crudas_test,
                            objetivo, predictores, motor)
                        n_tr, n_va = len(crudas_train), len(crudas_val)
                        hacer = lambda xs: Particion.desde_filas(  # noqa: E731
                            xs, row_id_field="row_id", target_field=objetivo)
                        t0 = time.perf_counter()
                        intento = ejecutar_intento_aislado(
                            motor, hacer(transformadas[:n_tr]),
                            hacer(transformadas[n_tr:n_tr + n_va]),
                            hacer(transformadas[n_tr + n_va:]), spec,
                            Presupuesto(wall_seconds=wall_seconds, hilos=c3.HILOS_POR_INTENTO,
                                        seed=protocolo.particion.semillas[repeticion]),
                            candidate=f"{NOMBRE_DENSA}-{variante}-{c3.CONFIGURACION_UNICA}",
                            split_plan_digest=propuesta.plan.digest(),
                            dataset=ds.nombre, pliegue=pliegue_i, repeticion=repeticion)
                        transcurrido = time.perf_counter() - t0
                        metricas = c5.metricas_del_informe(intento.informe)
                        recursos = intento.recursos or {}
                        registro = {
                            "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                            "tarea": ds.tarea, "motor": NOMBRE_DENSA, "variante": variante,
                            "palanca": VARIANTES[variante], "repeticion": repeticion,
                            "pliegue": pliegue_i, "estado": intento.estado,
                            "semilla": protocolo.particion.semillas[repeticion],
                            "presupuesto_wall_s": wall_seconds, "metrica_de_cierre": metric_id,
                            "wall_s": round(transcurrido, 3), "metricas": metricas,
                            "tiempo_de_ajuste": recursos.get("wall_seconds"),
                            "cpu_segundos": recursos.get("cpu_seconds"),
                            "motivo": (intento.motivo_del_estado["es"]
                                       if intento.motivo_del_estado else None),
                            "entorno_digest": entorno_digest, "motor_digest": motor_digest,
                            "procedencia_id": procedencia["procedencia_id"], "reusado": False,
                        }
                        c5.aplanar_metricas_en_el_registro(registro, metricas)
                    resultados.append(registro)
                    del_conjunto[variante].append(registro)
                    print(f"  {ds.nombre} {variante:3s} rep={repeticion} pliegue={pliegue_i}: "
                          f"{registro.get('estado')} {metric_id}={registro.get(metric_id)} "
                          f"wall={registro.get('wall_s')}s{' [reusado]' if registro.get('reusado') else ''}",
                          flush=True)
            guardar(parcial=True)
        veredicto_ds = veredicto_del_conjunto(dataset=ds.nombre, metric_id=metric_id,
                                              hoy=del_conjunto["hoy"], b=del_conjunto["B"])
        por_conjunto.append(veredicto_ds)
        print(f"  VEREDICTO {ds.nombre}: media hoy={veredicto_ds['media_hoy']} "
              f"B={veredicto_ds['media_B']} mejora={veredicto_ds['mejora']} "
              f"inferioridad={veredicto_ds['inferioridad']} "
              f"B deja de completar={len(veredicto_ds['B_deja_de_completar'])} "
              f"motivo={veredicto_ds['motivo']}", flush=True)
        guardar(parcial=True)

    salida = guardar(parcial=False)
    v = salida["veredicto"]
    print(f"\nGuardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}; "
          f"{len(resultados)} intentos ({reusados} reusados) en {salida['total_wall_s']} s")
    print(f"VEREDICTO sino-b: entra={v['entra']} (inferioridad en {v['inferioridad_en']}, "
          f"mejora en {v['mejora_en']}, B deja de completar en {v['B_deja_de_completar_en']}, "
          f"sin medir {v['conjuntos_sin_medir']})")


if __name__ == "__main__":
    main()
