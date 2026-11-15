#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169 — LA MEDIDA: `hgb_hoy` y `hgb_rejilla` en los 6 conjuntos NUEVOS de `benchmarks/c163/CONFIRMACION_C2.md`, repetición 0, pliegues 0–4.

Mismo protocolo que `medir_c163_c2.py` y SUS piezas, importadas (los dos brazos, el digest del motor, la reutilización por caché, las comprobaciones
de brazo): referencia primero y rejilla inmediatamente después, cada uno en un subproceso aislado con el presupuesto de su cubo (120 s pequeño, 300 s
mediano), 4 hilos, misma preparación y semilla. LightGBM no se mide. Lo único propio: el catálogo (`conjuntos_c169`, los 6 SIN tocar el protocolo
v2 firmado), la descarga verificada por sha256 y el sello de ESTE contrato.

SELLO. `analiza_c169.REGLA_SHA256_SELLADA` es el sha256 de `REGLA.md`, y `SELLO_DEL_CODIGO.json` el del analizador, este medidor y las dependencias
del veredicto (rep1, I1; lo escribe `sellar_codigo_c169.py`). Una medida real se NIEGA a empezar si falta cualquiera de los dos, es nulo o ya no
cuadra. `--sonda` (1 conjunto pequeño, 1 pliegue) se salta los sellos y marca la salida como SONDA: no decide nada.
La partición de los 6 no tiene una «v2 registrada» con la que compararla (no estaban en la Fase 0): se DECLARA en la salida (plan_digest, límites).

    python3 benchmarks/c169/medir_c169.py --salida <ruta absoluta fuera del árbol>        # la medida (a la COLA)
    python3 benchmarks/c169/medir_c169.py --sonda --salida <ruta>                        # 1 conjunto, 1 pliegue

AL ENCOLAR (rep1, aud1 M-7). Estimada ~20 min de cómputo (~30 con la preparación), NO medida; tope 3.600 s y 5 GB del contrato §5. El peor caso
(todos los intentos al tope de su cubo: 4 x 5 x 2 x 120 s + 2 x 5 x 2 x 300 s) son 3 h: saldría PARCIAL (SIN VEREDICTO) y se reanuda por la caché.
El techo blando por omisión (`--memoria-max-mb`, 4.000 MB) queda por DEBAJO de los 5 GB de `MemoryMax`, para que el corte suave (que guarda lo
medido) llegue antes que el duro:

    ESTIMADA_S=1800 ~/encolar.sh c169 3600 5G matrixAI python3 benchmarks/c169/medir_c169.py \
        --salida ~/cola-nocturna/resultados/c169/resultado_c169.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_FASE0 = _AQUI.parents[1] / "benchmarks" / "fase0"
for _p in (_AQUI, _AQUI.parent / "c163", _AQUI.parent / "c160", _FASE0):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c160 as a160  # noqa: E402,F401
import analiza_c163_c2 as an163  # noqa: E402
import analiza_c169 as an  # noqa: E402  (la regla, los sellos y los brazos)
import conjuntos_c169 as cj  # noqa: E402
import medir_c163_c2 as m163  # noqa: E402  (los brazos, el digest del motor, la caché: SUS piezas)
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c119  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

CORTE = "169"
REPETICION = an.REPETICION
RUTA_DE_LA_REGLA = an.RUTA_DE_LA_REGLA
RUTA_DEL_ANALIZADOR = an.RUTA_DEL_ANALIZADOR
CONJUNTOS_DE_LA_SONDA = ("blood-transfusion-service-center",)   # 748 filas, 4 predictores: el más pequeño
PLIEGUES_DE_LA_SONDA = 1
PRESUPUESTO_DE_LA_SONDA_S = 30.0
MEMORIA_MAX_MB_POR_OMISION = 4000.0      # por DEBAJO del `MemoryMax=5G` de la cola (aud1 M-7): con 6.000 el corte suave no llegaba nunca
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0


def _sha256_fichero(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def motores() -> dict:
    """SOLO los dos brazos de HGB, los de `medir_c163_c2.motores()` (la referencia primero)."""
    todos = m163.motores()
    brazos = {b: todos[b] for b in an.BRAZOS}
    if tuple(brazos) != an.BRAZOS or brazos[an.REFERENCIA].rejilla_por_validacion is not False \
            or brazos[an.REJILLA].rejilla_por_validacion is not True:
        raise SystemExit("los brazos no son (referencia sin rejilla, brazo con ella)")
    return brazos


def dependencias_sha256() -> dict[str, str]:
    return an.dependencias_de_ahora()


def componentes_del_digest() -> dict[str, str]:
    """Los del 163-C2 (cierre de imports en los dos repos, los dos motores y la política, versiones, protocolos) MÁS lo propio de este corte:
    el analizador, el catálogo de los 6 y este guion (que el cierre no sigue: `benchmarks/c169` no es raíz de import)."""
    componentes = dict(m163.componentes_del_digest())
    for f in (Path(an.__file__), Path(cj.__file__), cj.RUTA_DEL_CATALOGO, Path(m163.__file__), Path(an163.__file__), Path(__file__)):
        componentes[f"c169:{Path(f).name}"] = c3._digest_fichero(Path(f).resolve())
    return componentes


def clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["repeticion"], r["pliegue"])


def _argumentos(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", required=True, help="JSON de salida (ruta ABSOLUTA fuera del árbol; en la cola, ~/cola-nocturna/resultados/<nombre>/…)")
    p.add_argument("--forzar", action="store_true", help="ignora la caché entera")
    p.add_argument("--sonda", action="store_true", help=f"{', '.join(CONJUNTOS_DE_LA_SONDA)}, {PLIEGUES_DE_LA_SONDA} pliegue, "
                                                        f"{PRESUPUESTO_DE_LA_SONDA_S:g} s por intento: no mide nada del corte")
    p.add_argument("--solo", default=None, help="nombres separados por coma, dentro de los 6 (probar; con --sonda sustituye a los de la sonda)")
    p.add_argument("--memoria-max-mb", type=float, default=MEMORIA_MAX_MB_POR_OMISION)
    return p.parse_args(argv)


def comprobar_el_sello(es_sonda: bool, sha_regla: str | None) -> None:
    """Una medida real se NIEGA a empezar sin el sello o con una regla que ya no es la sellada, y sin el sello del CÓDIGO o con un fichero sellado
    que ya no cuadra (analizador, este medidor, dependencias del veredicto). Una sonda no."""
    if es_sonda:
        return
    if an.REGLA_SHA256_SELLADA is None:
        raise SystemExit("la regla NO está sellada (`analiza_c169.REGLA_SHA256_SELLADA` es None): una medida real no empieza sin el sello. "
                         "El supervisor sella ANTES de encolar")
    if sha_regla != an.REGLA_SHA256_SELLADA:
        raise SystemExit(f"REGLA.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la regla es una "
                         f"enmienda con su sello, no una edición. No se mide")
    problemas = an.problemas_del_sello_del_codigo()
    if problemas:
        raise SystemExit("el sello del CÓDIGO no cuadra, no se mide: " + " | ".join(problemas))


def _mismo_protocolo(previo: dict | None, metric_id: str, semilla: int) -> bool:
    """Lo que `_reutilizable` del 163-C2 no mira y la regla fija: misma métrica de cierre, misma semilla y los mismos hilos GRABADOS."""
    return bool(previo) and previo.get("metrica_de_cierre") == metric_id and previo.get("semilla") == semilla \
        and previo.get("hilos") == c3.HILOS_POR_INTENTO


def main(argv=None) -> None:
    args = _argumentos(argv)
    es_sonda = bool(args.sonda)
    ruta_salida = Path(args.salida).expanduser().resolve()
    sha_regla = _sha256_fichero(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None
    comprobar_el_sello(es_sonda, sha_regla)

    # LOS DATOS PRIMERO: bajar (si faltan) y verificar los 6 contra su sha256 registrado. Si uno no cuadra, no se mide NADA.
    cj.asegurar_todos()

    c6.preparar_protocolo_v2()
    protocolos = c119.exigir_los_protocolos_encadenados()
    configuracion_del_motor = c119.exigir_la_configuracion_de_la_enmienda()
    enmienda_2 = c119.exigir_la_enmienda_2("humo" if es_sonda else "pasada")
    base = c3.protocolo_registrado()
    protocolo = cj.protocolo_ampliado(base)      # el v2 + los 6, EN MEMORIA: el fichero firmado no se toca
    cj.registrar_en_el_catalogo_de_lectura()
    c5.cargar_arff = cj.cargar_arff_sin_ignorados   # los `ignore_attribute` de OpenML no entran como predictores (REGLA.md §1)
    nuevos = {d.nombre: d for d in c5.datasets_de_la_pasada(protocolo) if d.data_id in cj.DATA_IDS}
    if set(nuevos) != {d.nombre for d in cj.conjuntos()}:
        raise SystemExit(f"los 6 no salen del catálogo ampliado: {sorted(nuevos)}")
    orden = [d.nombre for d in cj.conjuntos()]
    pedidos = list(CONJUNTOS_DE_LA_SONDA) if es_sonda else list(orden)
    if args.solo:
        pedidos = [s.strip() for s in args.solo.split(",") if s.strip()]
    for n in pedidos:
        if n not in nuevos:
            raise SystemExit(f"{n!r} no está entre los 6 de esta medida (los 23, los sellados y cualquier otro se NIEGAN)")
    datasets = [nuevos[n] for n in pedidos]
    if any(d.sellado for d in datasets):
        raise SystemExit("un conjunto sellado: no se toca")
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)

    los_motores = motores()
    componentes = componentes_del_digest()
    entorno_digest = c119._digest_de_componentes(componentes)
    digest_motor = {b: m163.digest_del_motor(b) for b in los_motores}
    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motores": digest_motor},
        datos_de_entrada={**{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
                          "protocolo_exploratorio_v2": c119.RUTA_DEL_PROTOCOLO_V2, "protocolo_119_v4": c119.RUTA_DEL_PROTOCOLO_V4,
                          "protocolo_119_v4_enmienda_1": c119.RUTA_DE_LA_ENMIENDA_1, "catalogo_c169": cj.RUTA_DEL_CATALOGO})
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
    sha_dependencias = dependencias_sha256()
    sello_del_codigo = (an.leer_sello_del_codigo() or {}).get("sha256")

    def guardar(parcial: bool) -> None:
        ultimo_control[0] = time.perf_counter()
        c119._escribir_atomicamente(ruta_salida, {
            "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "corte": CORTE,
            "tipo_de_ejecucion": "sonda" if es_sonda else "pasada", "es_sonda": es_sonda, "parcial": parcial, "repeticion": REPETICION,
            "que_es": ("HGB con la política por tamaño de hoy y con la rejilla de 8 configuraciones por validación en los 6 conjuntos nuevos de "
                       "CONFIRMACION_C2.md; la regla, en analiza_c169.py / REGLA.md"),
            "brazos": list(an.BRAZOS), "politica_de_la_referencia": m163.POLITICA_DE_LA_REFERENCIA.a_json(),
            "conjuntos": conjuntos_decl, "protocolo_digest_sha256_v2_firmado": base.digest(),
            "protocolo_digest_sha256_ampliado_en_memoria": protocolo.digest(),
            "protocolos_encadenados": protocolos, "enmienda_2_cuadra": enmienda_2.get("cuadra"),
            "configuracion_del_motor": configuracion_del_motor,
            "digest_de_la_cache": {"entorno": entorno_digest, "motores": digest_motor, "componentes": componentes},
            "regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "regla_sha256_al_medir": sha_regla,
            "analizador_sha256_al_medir": sha_analizador, "dependencias_sha256_al_medir": sha_dependencias,
            "guion_sha256": _sha256_fichero(Path(__file__)), "sello_del_codigo_al_medir": sello_del_codigo,
            "procedencia": procedencia,
            "sha256_de_los_datos_registrados": cj.sha256_registrado(),
            "total_wall_s": round(time.perf_counter() - inicio, 1), "n_reusados": reusados[0],
            "rss_pico_del_padre_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "resultados": sorted(registros.values(), key=lambda r: (r["dataset"], r["pliegue"], r["motor"])),
        })

    def vigilar_memoria(etiqueta: str, rss_hijo_mb) -> None:
        padre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        for quien, v in (("padre", padre), ("hijo", rss_hijo_mb)):
            if v is not None and v > args.memoria_max_mb:
                guardar(parcial=True)
                raise SystemExit(f"{etiqueta}: el pico de memoria del {quien} ({v:.0f} MB) pasa el techo blando ({args.memoria_max_mb:.0f} MB). "
                                 f"Se para; lo medido queda guardado")

    print(f"{CORTE} ({'SONDA' if es_sonda else 'PASADA'}): {len(datasets)} conjuntos, brazos {list(an.BRAZOS)}, entorno {entorno_digest}. "
          f"Salida: {ruta_salida}", flush=True)

    for ds in datasets:
        t_carga = time.perf_counter()
        por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
        carga_s = time.perf_counter() - t_carga
        test_ids = propuesta.plan.observaciones_del_rol("test")
        metric_id = metrica_por_dataset[ds.nombre]
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        if datos_sha256 != cj.sha256_registrado()[ds.data_id]:
            raise SystemExit(f"{ds.nombre}: el sha256 de los datos medidos ({datos_sha256}) no es el registrado")
        plan = [i for i in c119.plan_de_intentos(ds, propuesta, protocolo, humo=False) if i.repeticion == REPETICION]
        if es_sonda:
            plan = [i for i in plan if i.pliegue < PLIEGUES_DE_LA_SONDA]
        elif [i.pliegue for i in plan] != an.PLIEGUES:
            raise SystemExit(f"{ds.nombre}: la partición trae los pliegues {[i.pliegue for i in plan]}, no {an.PLIEGUES}")
        conjuntos_decl.append({
            "nombre": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado,
            "metrica_de_cierre": metric_id, "pliegues_esperados": [i.pliegue for i in plan], "n_filas": len(por_id),
            "n_test": len(test_ids), "datos_sha256": datos_sha256, "particion_declarada": particion,
            "lectura_declarada": c3.LECTURA_DECLARADA.get(str(ds.data_id)),
            "columnas_ignoradas_por_openml": (c3.LECTURA_DECLARADA.get(str(ds.data_id)) or {}).get("columnas_ignoradas_por_openml"),
            "n_predictores": len(predictores), "predictores": list(predictores)})
        print(f"\n=== {ds.nombre} ({ds.cubo}, {ds.tarea}, n={len(por_id)}, metrica={metric_id}, pliegues={len(plan)}, carga={carga_s:.1f}s) ===", flush=True)

        for ip in plan:
            wall_s = PRESUPUESTO_DE_LA_SONDA_S if es_sonda else ip.presupuesto_wall_s
            pendientes = []
            for brazo in los_motores:
                k = (ds.nombre, brazo, REPETICION, ip.pliegue)
                p = previo.get(k)
                if not _mismo_protocolo(p, metric_id, ip.semilla):
                    p = None
                if m163._reutilizable(p, entorno_digest, digest_motor[brazo], wall_s, datos_sha256):
                    registros[k] = dict(p, reusado=True)
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} {brazo} pliegue={ip.pliegue}: estado={p.get('estado')}", flush=True)
                else:
                    pendientes.append(brazo)
            for brazo in pendientes:
                motor = los_motores[brazo]
                t_prep = time.perf_counter()
                crudas_train = [por_id[i] for i in ip.entrena]
                crudas_val = [por_id[i] for i in ip.valida]
                crudas_test = [por_id[i] for i in test_ids]
                transformadas = c3.preparar_para_motor(crudas_train, crudas_train + crudas_val + crudas_test, objetivo, predictores, motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)

                def hacer(xs):
                    return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

                train, val, test = hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]), hacer(transformadas[n_tr + n_va:])
                preparacion_s = time.perf_counter() - t_prep
                presupuesto = Presupuesto(wall_seconds=wall_s, hilos=c3.HILOS_POR_INTENTO, seed=ip.semilla)
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto, candidate=f"{brazo}", split_plan_digest=propuesta.plan.digest(),
                    dataset=ds.nombre, pliegue=ip.pliegue, repeticion=REPETICION)
                transcurrido = time.perf_counter() - t0
                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                config = intento.config_efectiva or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": brazo, "nombre_del_motor": motor.nombre, "repeticion": REPETICION, "pliegue": ip.pliegue,
                    "estado": intento.estado, "semilla": ip.semilla, "presupuesto_wall_s": wall_s, "metrica_de_cierre": metric_id,
                    "wall_s": round(transcurrido, 3), "preparacion_s": round(preparacion_s, 3), "metricas": metricas,
                    "tiempo_de_ajuste": recursos.get("wall_seconds"), "cpu_segundos": recursos.get("cpu_seconds"),
                    "rss_pico_mb": recursos.get("peak_ram_mb"), "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
                    "arquitectura": config.get("arquitectura"), "hiperparametros": config.get("hiperparametros"),
                    "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                    "engine_version": intento.engine_version, "split_plan_digest": propuesta.plan.digest(),
                    "entorno_digest": entorno_digest, "motor_digest": digest_motor[brazo], "datos_sha256": datos_sha256,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False, "n_test": len(test_ids),
                    "hilos": presupuesto.hilos,
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                registros[clave(registro)] = registro
                if intento.estado in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
                    problemas = an163.problemas_de_la_declaracion(registro)
                    if problemas:
                        guardar(parcial=True)
                        raise SystemExit(f"{ds.nombre} {brazo} pliegue {ip.pliegue}: declara otra cosa que su brazo ({problemas})")
                rj = (registro["hiperparametros"] or {}).get("rejilla_por_validacion") or {}
                print(f"  {ds.nombre} {brazo:12s} pliegue={ip.pliegue}: estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"elegida={rj.get('elegida', '-')} ajuste={registro['tiempo_de_ajuste']} wall={transcurrido:.1f}s rss={registro['rss_pico_mb']}", flush=True)
                vigilar_memoria(f"{ds.nombre} {brazo} pliegue {ip.pliegue}", registro["rss_pico_mb"])
                if time.perf_counter() - ultimo_control[0] >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL:
                    guardar(parcial=True)
        guardar(parcial=True)

    en_el_plan = {(c["nombre"], b, REPETICION, f) for c in conjuntos_decl for b in los_motores for f in c["pliegues_esperados"]}
    de_estos = {c["nombre"] for c in conjuntos_decl}
    for k in [k for k in registros if k[0] in de_estos and k not in en_el_plan]:
        del registros[k]
    guardar(parcial=False)
    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total / 3600:.2f} h), {len(registros)} intentos ({reusados[0]} reusados) ===", flush=True)
    print(f"Guardado en {ruta_salida}. Siguiente: python3 benchmarks/c169/analiza_c169.py {ruta_salida}", flush=True)


if __name__ == "__main__":
    main()
