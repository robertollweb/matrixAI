#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""170 — LA MEDIDA: `interpret.ebm` (EBM, `interpret-core` 0.7.8, valores por omisión) en los conjuntos del RÉGIMEN (no sellados del protocolo v2 con ≤ 100 predictores: 24), todos sus pliegues.

La regla es `REGLA.md` y el veredicto lo da `analiza_c170.py`. Esto SOLO mide: no compara ni decide. Importa de la pasada 116-C2 lo común (catálogo y protocolo v2, partición, preparación, presupuesto por cubo, métrica
de cierre, caché por intento y procedencia) — nada se copia — y del analizador la `Referencia` (el régimen y los rivales ya medidos: no se remiden).

MISMO PROTOCOLO que la v2: misma partición (se COMPRUEBA su `plan_digest`), misma preparación (la del motor LightGBM, no la del EBM: se usa `preparar_para_motor` con el motor del EBM, que declara las MISMAS capacidades
de categóricas y faltantes), mismas semillas y presupuesto de pared por cubo (120/300/600 s), 4 hilos, UN intento por pliegue en un subproceso aislado con su tope.

SELLO. `analiza_c170.REGLA_SHA256_SELLADA` es el sha256 de `REGLA.md`, y `SELLO_DEL_CODIGO.json` el del analizador, este medidor, el motor y las dependencias (lo escribe `sellar_codigo_c170.py`). Una medida real se NIEGA a
empezar si falta cualquiera de los dos, es nulo o ya no cuadra. `--sonda` (UN conjunto, UN pliegue) se salta los sellos y marca la salida como SONDA: no decide nada.

Dónde corre: DENTRO de un contenedor con `interpret-core==0.7.8` (`correr_c170_en_contenedor.sh`; el host no la tiene, a propósito). Reanudable: reutiliza los intentos COMPLETADOS con el mismo código y presupuesto;
un intento FALLIDO NO se reutiliza (una caída por carga ajena no se cementa: el relanzamiento lo reintenta).

    python3 benchmarks/c170/medir_c170.py --plan                                              # lo que haría, sin leer datos
    python3 benchmarks/c170/medir_c170.py --sonda --solo diabetes --salida <ruta>            # 1 conjunto, 1 pliegue
    python3 benchmarks/c170/medir_c170.py --salida <ruta absoluta fuera del árbol>            # la medida (a la COLA, en el contenedor)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_FASE0 = _AQUI.parents[1] / "benchmarks" / "fase0"
for _p in (_AQUI, _FASE0):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c170 as an  # noqa: E402  (la regla, los sellos, la referencia)
import pasada_116c2_tabicl as p116  # noqa: E402  (lo común: protocolo v2, caché, procedencia)
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.registro_de_motores import motor_para  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

CORTE = "170"
MOTOR = an.MOTOR
RUTA_DE_LA_REGLA = an.RUTA_DE_LA_REGLA
RUTA_DEL_ANALIZADOR = an.RUTA_DEL_ANALIZADOR
RUTA_DEL_MOTOR = an.RUTA_DEL_MOTOR
SONDA_POR_OMISION = "diabetes"                # 768 filas, 8 predictores: el más pequeño del régimen
PLIEGUES_DE_LA_SONDA = 1
MEMORIA_MAX_MB_POR_OMISION = 3800.0           # por DEBAJO del techo del contenedor (5 g): el corte suave, que guarda lo medido, llega antes que el duro
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0
ORDEN_DE_CUBO = {"pequeno": 0, "mediano": 1, "grande": 2}


def _sha256_fichero(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def ficheros_del_entorno() -> tuple[Path, ...]:
    """Lo que, si cambia, invalida TODA la caché: los de C5 (que incluyen los de C3), la pasada 116-C2 y el registro de motores que ya usa, el analizador y este medidor."""
    return tuple(dict.fromkeys(tuple(p116._FICHEROS_COMPARTIDOS) + (Path(an.__file__).resolve(), Path(__file__).resolve())))


def digest_del_entorno() -> str:
    return hashlib.sha256("".join(c3._digest_fichero(f) for f in ficheros_del_entorno()).encode()).hexdigest()[:16]


def digest_del_motor() -> str:
    return c3._digest_fichero(RUTA_DEL_MOTOR)


def comprobar_el_sello(es_sonda: bool, sha_regla: str | None) -> None:
    """Una medida real se NIEGA a empezar sin el sello de la regla, con una regla que ya no es la sellada, sin el sello del CÓDIGO o con un fichero sellado que ya no cuadra. Una sonda no."""
    if es_sonda:
        return
    if an.REGLA_SHA256_SELLADA is None:
        raise SystemExit("la regla NO está sellada (`analiza_c170.REGLA_SHA256_SELLADA` es None): una medida real no empieza sin el sello. El supervisor sella ANTES de encolar")
    if sha_regla != an.REGLA_SHA256_SELLADA:
        raise SystemExit(f"REGLA.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la regla es una enmienda con su sello, no una edición. No se mide")
    problemas = an.problemas_del_sello_del_codigo()
    if problemas:
        raise SystemExit("el sello del CÓDIGO no cuadra, no se mide: " + " | ".join(problemas))


def _commits_del_lanzador() -> dict | None:
    """Lo que el guion del contenedor declara (el contenedor no trae `git`, y la procedencia no puede preguntárselo): commit y ficheros modificados de cada repo. `None` si no se lanzó por el guion."""
    crudo = os.environ.get("C170_COMMITS")
    if not crudo:
        return None
    try:
        return json.loads(crudo)
    except ValueError:
        return {"ilegible": crudo}


def clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["repeticion"], r["pliegue"])


def _mismo_protocolo(previo: dict | None, metric_id: str, semilla: int) -> bool:
    """Lo que `c3._reusable` no mira y la regla fija: el intento terminó BIEN, misma métrica de cierre, misma semilla y los mismos hilos GRABADOS."""
    return (bool(previo) and previo.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA and previo.get("metrica_de_cierre") == metric_id
            and previo.get("semilla") == semilla and previo.get("hilos") == c3.HILOS_POR_INTENTO)


def conjuntos_a_medir(ref: an.Referencia, solo: list[str] | None) -> list[str]:
    """El régimen de la v2 (por tabla, no a ojo), pequeño primero; `--solo` restringe a un subconjunto DENTRO del régimen y nada fuera."""
    regimen = sorted(ref.regimen(), key=lambda n: (ORDEN_DE_CUBO[ref.meta[n]["cubo"]], ref.meta[n]["n_test"], n))
    if solo:
        for n in solo:
            if n not in regimen:
                raise SystemExit(f"{n!r} no está en el régimen de la regla ({len(regimen)} no sellados con <= {an.COLUMNAS_MAXIMAS} predictores): los sellados y los de más columnas se NIEGAN")
        return [n for n in regimen if n in solo]
    return regimen


def imprimir_plan(ref: an.Referencia, nombres: list[str]) -> dict:
    """Cuántos ajustes y con qué tope por conjunto, SIN leer ni un ARFF (de la partición que la v2 ya midió)."""
    topes = {"pequeno": 120.0, "mediano": 300.0, "grande": 600.0}
    total, peor = 0, 0.0
    print(f"{'conjunto':34s} {'cubo':8s} {'tarea':26s} {'cols':>5s} {'pliegues':>8s} {'tope s':>7s}")
    for n in nombres:
        m = ref.meta[n]
        total += len(m["claves"])
        peor += len(m["claves"]) * topes[m["cubo"]]
        print(f"{n:34s} {m['cubo']:8s} {m['tarea']:26s} {m['n_predictores']:5d} {len(m['claves']):8d} {topes[m['cubo']]:7.0f}")
    print(f"\n{len(nombres)} conjuntos, {total} ajustes; peor caso (todos al tope de su cubo): {peor:.0f} s = {peor / 3600:.1f} h")
    return {"conjuntos": len(nombres), "ajustes": total, "peor_caso_s": peor}


def _argumentos(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", default=None, help="JSON de salida (ruta ABSOLUTA fuera del árbol; en la cola, ~/cola-nocturna/resultados/<nombre>/…)")
    p.add_argument("--forzar", action="store_true", help="ignora la caché entera")
    p.add_argument("--sonda", action="store_true", help=f"UN conjunto ({SONDA_POR_OMISION} por omisión, o el de --solo), {PLIEGUES_DE_LA_SONDA} pliegue: no mide nada del corte")
    p.add_argument("--solo", default=None, help="nombres separados por coma, DENTRO del régimen (probar; con --sonda es el conjunto de la sonda)")
    p.add_argument("--plan", action="store_true", help="imprime los ajustes y topes por conjunto (de la v2, sin leer datos) y sale")
    p.add_argument("--memoria-max-mb", type=float, default=MEMORIA_MAX_MB_POR_OMISION)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _argumentos(argv)
    es_sonda = bool(args.sonda)
    solo = [s.strip() for s in args.solo.split(",") if s.strip()] if args.solo else None
    ref = an.cargar_referencia()
    problemas_ref = ref.problemas()
    if problemas_ref:
        raise SystemExit("la referencia (v2 + TabM) no es la misma partición: " + " | ".join(problemas_ref))
    nombres = conjuntos_a_medir(ref, solo if solo else ([SONDA_POR_OMISION] if es_sonda else None))
    if es_sonda and len(nombres) != 1:
        raise SystemExit(f"una sonda es UN conjunto; --solo trae {len(nombres)}")
    if args.plan:
        imprimir_plan(ref, nombres)
        return
    if not args.salida:
        raise SystemExit("falta --salida")
    ruta_salida = Path(args.salida).expanduser().resolve()
    sha_regla = _sha256_fichero(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None
    comprobar_el_sello(es_sonda, sha_regla)

    motor = motor_para(MOTOR)
    no_disponible = motor.motivo_si_no_puede_entrenar()
    if no_disponible is not None:
        raise SystemExit(f"el motor {MOTOR} no está disponible aquí: {no_disponible['es']}")

    p116.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    todos = {d.nombre: d for d in c5.datasets_de_la_pasada(protocolo)}
    datasets = [todos[n] for n in nombres]
    if any(d.sellado for d in datasets):
        raise SystemExit("un conjunto sellado: no se toca")
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        p116.direccion_de(metric_id)

    entorno_digest, motor_digest = digest_del_entorno(), digest_del_motor()
    procedencia = c3.procedencia_de_la_medicion(digests_de_codigo={"entorno": entorno_digest, "motor": motor_digest},
                                                datos_de_entrada={d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets})
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)

    previo: dict[tuple, dict] = {}
    if ruta_salida.exists() and not args.forzar:
        try:
            previo = {clave(r): r for r in json.loads(ruta_salida.read_text(encoding="utf-8")).get("resultados", [])}
        except (ValueError, KeyError) as e:
            raise SystemExit(f"{ruta_salida} no se puede leer ({e}): se mueve o se usa --forzar") from e
    registros: dict[tuple, dict] = dict(previo)
    conjuntos_decl: list[dict] = []
    reusados = [0]
    inicio = time.perf_counter()
    ultimo_control = [inicio]
    sha_analizador = _sha256_fichero(RUTA_DEL_ANALIZADOR)
    sha_dependencias = an.dependencias_de_ahora()
    sello_del_codigo = (an.leer_sello_del_codigo() or {}).get("sha256")

    def guardar(parcial: bool) -> None:
        ultimo_control[0] = time.perf_counter()
        salida = {
            "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "corte": CORTE, "motor": MOTOR, "tipo_de_ejecucion": "sonda" if es_sonda else "pasada",
            "es_sonda": es_sonda, "parcial": parcial,
            "que_es": "interpret.ebm (interpret-core 0.7.8, valores por omisión) en el régimen de la regla; la regla, en analiza_c170.py / REGLA.md",
            "regimen": sorted(ref.regimen()), "fuera_del_regimen": ref.fuera_del_regimen(),
            "conjuntos": conjuntos_decl, "protocolo_digest_sha256": protocolo.digest(), "protocolo_version": protocolo.version_protocolo,
            "digest_de_la_cache": {"entorno": entorno_digest, "motor": motor_digest},
            "regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "regla_sha256_al_medir": sha_regla, "analizador_sha256_al_medir": sha_analizador,
            "dependencias_sha256_al_medir": sha_dependencias, "guion_sha256": _sha256_fichero(Path(__file__)), "sello_del_codigo_al_medir": sello_del_codigo,
            "procedencia": procedencia, "commits_declarados_por_el_lanzador": _commits_del_lanzador(), "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
            "total_wall_s": round(time.perf_counter() - inicio, 1), "n_reusados": reusados[0],
            "rss_pico_del_padre_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "resultados": sorted(registros.values(), key=lambda r: (r["dataset"], r["repeticion"], r["pliegue"])),
        }
        c3.sellar_la_salida(salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
        temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        temporal.replace(ruta_salida)

    def vigilar_memoria(etiqueta: str, rss_hijo_mb) -> None:
        padre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        for quien, v in (("padre", padre), ("hijo", rss_hijo_mb)):
            if v is not None and v > args.memoria_max_mb:
                guardar(parcial=True)
                raise SystemExit(f"{etiqueta}: el pico de memoria del {quien} ({v:.0f} MB) pasa el techo blando ({args.memoria_max_mb:.0f} MB). Se para; lo medido queda guardado")

    print(f"{CORTE} ({'SONDA' if es_sonda else 'PASADA'}): {len(datasets)} conjuntos, motor {MOTOR}, entorno {entorno_digest}. Salida: {ruta_salida}", flush=True)

    for ds in datasets:
        m = ref.meta[ds.nombre]
        t_carga = time.perf_counter()
        por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
        carga_s = time.perf_counter() - t_carga
        test_ids = propuesta.plan.observaciones_del_rol("test")
        metric_id = metrica_por_dataset[ds.nombre]
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        # LO QUE LA REGLA FIJA, comprobado con los datos LEÍDOS (no se cree lo declarado): predictores, partición y datos de la v2
        if len(predictores) != m["n_predictores"] or len(predictores) > an.COLUMNAS_MAXIMAS:
            raise SystemExit(f"{ds.nombre}: {len(predictores)} predictores leídos y la v2 registró {m['n_predictores']} (régimen: <= {an.COLUMNAS_MAXIMAS})")
        if propuesta.plan.digest() != m["plan_digest"]:
            raise SystemExit(f"{ds.nombre}: la partición de hoy ({propuesta.plan.digest()[:12]}) no es la de la v2 ({m['plan_digest'][:12]})")
        if m["datos_sha256"] is not None and datos_sha256 != m["datos_sha256"]:
            raise SystemExit(f"{ds.nombre}: el sha256 de los datos medidos ({datos_sha256}) no es el que midió la v2 ({m['datos_sha256']})")
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        wall_s_cubo = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        plan = [(rep, f) for rep in range(repeticiones) for f in range(protocolo.particion.folds)
                if propuesta.pliegues.pliegue_de(repeticion=rep, pliegue=f) is not None]
        if set(plan) != m["claves"]:
            raise SystemExit(f"{ds.nombre}: los pliegues de hoy {sorted(plan)} no son los de la v2 {sorted(m['claves'])}")
        esperados = sorted(plan)
        if es_sonda:
            plan = plan[:PLIEGUES_DE_LA_SONDA]
        conjuntos_decl.append({
            "nombre": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado, "metrica_de_cierre": metric_id,
            "pliegues_esperados": [list(k) for k in (plan if es_sonda else esperados)], "n_filas": len(por_id), "n_test": len(test_ids), "datos_sha256": datos_sha256,
            "plan_digest": propuesta.plan.digest(), "n_predictores": len(predictores), "particion_declarada": particion,
            "lectura_declarada": c3.LECTURA_DECLARADA.get(str(ds.data_id))})
        print(f"\n=== {ds.nombre} ({ds.cubo}, {ds.tarea}, n={len(por_id)}, columnas={len(predictores)}, metrica={metric_id}, pliegues={len(plan)}, tope={wall_s_cubo:.0f}s, carga={carga_s:.1f}s) ===", flush=True)

        for rep, f in plan:
            pliegue = propuesta.pliegues.pliegue_de(repeticion=rep, pliegue=f)
            semilla = protocolo.particion.semillas[rep]
            k = (ds.nombre, MOTOR, rep, f)
            p = previo.get(k)
            if _mismo_protocolo(p, metric_id, semilla) and c3._reusable(p, entorno_digest, motor_digest, wall_s_cubo):
                registros[k] = dict(p, reusado=True)
                reusados[0] += 1
                print(f"  [reusado] {ds.nombre} rep={rep} pliegue={f}: estado={p.get('estado')}", flush=True)
                continue
            t_prep = time.perf_counter()
            crudas_train = [por_id[i] for i in pliegue.entrena]
            crudas_val = [por_id[i] for i in pliegue.valida]
            crudas_test = [por_id[i] for i in test_ids]
            transformadas = c3.preparar_para_motor(crudas_train, crudas_train + crudas_val + crudas_test, objetivo, predictores, motor)
            n_tr, n_va = len(crudas_train), len(crudas_val)

            def hacer(xs):
                return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

            train, val, test = hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]), hacer(transformadas[n_tr + n_va:])
            preparacion_s = time.perf_counter() - t_prep
            presupuesto = Presupuesto(wall_seconds=wall_s_cubo, hilos=c3.HILOS_POR_INTENTO, seed=semilla)
            t0 = time.perf_counter()
            intento = ejecutar_intento_aislado(motor, train, val, test, spec, presupuesto, candidate=f"{motor.nombre}-{c3.CONFIGURACION_UNICA}",
                                               split_plan_digest=propuesta.plan.digest(), dataset=ds.nombre, pliegue=f, repeticion=rep)
            transcurrido = time.perf_counter() - t0
            metricas = c5.metricas_del_informe(intento.informe)
            recursos = intento.recursos or {}
            config = intento.config_efectiva or {}
            registro = {
                "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado, "motor": MOTOR, "repeticion": rep, "pliegue": f,
                "estado": intento.estado, "semilla": semilla, "presupuesto_wall_s": wall_s_cubo, "configuracion": c3.CONFIGURACION_UNICA, "metrica_de_cierre": metric_id,
                "wall_s": round(transcurrido, 3), "preparacion_s": round(preparacion_s, 3), "metricas": metricas,
                "tiempo_de_ajuste": recursos.get("wall_seconds"), "cpu_segundos": recursos.get("cpu_seconds"), "rss_pico_mb": recursos.get("peak_ram_mb"),
                # SOLO los hiperparámetros: `config_efectiva` entera es el predictor con TODAS las tablas del modelo (megas por pliegue)
                "hiperparametros": config.get("hiperparametros"), "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                "engine_version": intento.engine_version, "split_plan_digest": propuesta.plan.digest(), "entorno_digest": entorno_digest, "motor_digest": motor_digest,
                "datos_sha256": datos_sha256, "procedencia_id": procedencia["procedencia_id"], "reusado": False, "n_test": len(test_ids), "hilos": presupuesto.hilos,
            }
            c5.aplanar_metricas_en_el_registro(registro, metricas)
            registros[clave(registro)] = registro
            print(f"  {ds.nombre} rep={rep} pliegue={f}: estado={intento.estado} {metric_id}={registro.get(metric_id)} ajuste={registro['tiempo_de_ajuste']} "
                  f"wall={transcurrido:.1f}s rss={registro['rss_pico_mb']} {registro['motivo'] or ''}", flush=True)
            vigilar_memoria(f"{ds.nombre} rep={rep} pliegue={f}", registro["rss_pico_mb"])
            if time.perf_counter() - ultimo_control[0] >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL:
                guardar(parcial=True)
        guardar(parcial=True)

    # nada fuera de lo declarado: un registro de una caché de otro plan no entra en la salida
    en_el_plan = {(c["nombre"], MOTOR, k[0], k[1]) for c in conjuntos_decl for k in c["pliegues_esperados"]}
    de_estos = {c["nombre"] for c in conjuntos_decl}
    for k in [k for k in registros if k[0] in de_estos and k not in en_el_plan]:
        del registros[k]
    for k in [k for k in registros if k[0] not in de_estos]:
        del registros[k]
    guardar(parcial=False)
    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total / 3600:.2f} h), {len(registros)} intentos ({reusados[0]} reusados) ===", flush=True)
    print(f"Guardado en {ruta_salida}. Siguiente: python3 benchmarks/c170/analiza_c170.py {ruta_salida}", flush=True)


if __name__ == "__main__":
    main()
