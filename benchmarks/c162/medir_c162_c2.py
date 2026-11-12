#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C2 — la PASADA que mide TabM `ref` (k = 8 miembros) frente a TabM con MÁS CAPACIDAD (k = 32) en los conjuntos del cubo PEQUEÑO.

QUÉ HACE. Para cada uno de los 13 conjuntos del cubo pequeño entre los 23 no sellados de la Fase 0 (leídos del congelado
`resultado_pasada_119_c3.json`), repetición 0, TODOS los pliegues (yeast 0–3, el resto 0–4), con las MISMAS particiones, semillas,
preparación y presupuesto de pared (120 s por intento, 4 hilos) que 119-C3/155/160/162-C1, ejecuta TabM (`matrixai.dense.tabm_cpu`) en sus
DOS brazos con el arnés de la Fase 0 (`ejecutar_intento_aislado`): `tabm_k8` (`MotorDensaTabM()`, la referencia) y `tabm_k32`
(`MotorDensaTabM().con_capacidad(k=32)`). Por pliegue, la referencia primero y la palanca inmediatamente después (misma carga).

LA PARADA ES LA DE SIEMPRE (por la PÉRDIDA de validación del ensamblado) en los dos brazos: la parada por métrica es la palanca del C1 y NO
se mezcla (el analizador rechaza un intento que declare `parada_por_metrica`). La regla la aplica `analiza_c162_c2.py`.

REUTILIZA el arnés del C1 (`medir_c162_c1`: caché por digest, dependencias, congelado, escritura atómica) y su analizador por import. Lo único
propio es el bucle por pliegue: el `main` del C1 es un bloque con sus dos brazos escritos dentro (`motores()`, la declaración de la parada) y
no se puede parametrizar sin editar un guion ya sellado.

EL PLAZO ES DE RELOJ DE PARED. Con k = 32 la época cuesta más (ver la sonda en el INFORME): a igual plazo, k = 32 hace MENOS épocas. Es parte de
lo que se mide (el coste (d) y `parados_por_plazo` lo declaran), no un defecto del arnés.

CACHÉ Y REANUDACIÓN, SELLO: como el C1 (`--forzar`, `--sonda`, `--solo`). `REGLA_SHA256_SELLADA` None = sin sellar: una medida real no empieza.

DURACIÓN ESTIMADA (no medida): ver el docstring de `ESTIMACION` abajo y el INFORME.

    python3 benchmarks/c162/medir_c162_c2.py --salida <ruta absoluta fuera del árbol>          # la medida (a la COLA)
    python3 benchmarks/c162/medir_c162_c2.py --sonda --salida <ruta>    # 1 conjunto (kc2), 1 pliegue, 8 s: prueba el camino
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

import medir_c162_c1 as m1  # noqa: E402  (el arnés del C1; pone en `sys.path` lo demás)
import analiza_c162_c2 as an  # noqa: E402
import analiza_c160 as a160  # noqa: E402
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c119  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa_tabm import MotorDensaTabM  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

CORTE = "162-C2"
REPETICION = 0
RUTA_DE_LA_REGLA = an.RUTA_DE_LA_REGLA
RUTA_DEL_ANALIZADOR = an.RUTA_DEL_ANALIZADOR
CONJUNTOS_DE_LA_SONDA = ("kc2",)
PLIEGUES_DE_LA_SONDA = 1
PRESUPUESTO_DE_LA_SONDA_S = 8.0
MEMORIA_MAX_MB_POR_OMISION = 3000.0
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0

#: DURACIÓN ESTIMADA (no medida): la referencia (TabM `ref`, 13 conjuntos pequeños, 155-c3) suma 1.594 s de `wall_s`; k = 32 cuesta
#: ×1,3–3,1 por ajuste (sonda con la máquina en carga 10–12: orientativa) y el plazo corta a 120 s por intento -> ≈ 1.594 + ~3.800 s +
#: preparación (65 pliegues) ≈ 6.000 s (1,7 h), SIN medir. Tope propuesto 4 h (14.400 s) y 4 GB (techo duro) / 3 GB (blando).
ESTIMACION = {"duracion_s_estimada": 6000, "tope_s_propuesto": 14400, "memoria_mb_propuesta": 3000}


def motores() -> dict:
    """Los dos brazos, en el orden de `analiza_c162_c2.BRAZOS` (la REFERENCIA primero). La parada es la de la pérdida en ambos."""
    return {an.REFERENCIA: MotorDensaTabM(), an.PALANCA: MotorDensaTabM().con_capacidad(k=an.K_DE_LA_PALANCA)}


def clave(r: dict) -> tuple:
    return m1.clave(r)


def componentes_del_digest() -> dict[str, str]:
    """El del C1 (cierre de imports de ESTE guion + compartidos + motor + analizadores) más el analizador de este corte y el del C1 +
    el guion del C1, que el cierre no sigue (`benchmarks/c162` no es raíz de import de la Fase 0)."""
    cierre, _ = c119._cierre_de_imports(Path(__file__).resolve(), c119._raices_de_import())
    extra = {Path(p).resolve() for p in c5._FICHEROS_COMPARTIDOS + c119._FICHEROS_DEL_MOTOR_NUEVO}
    extra |= {Path(x.__file__).resolve() for x in (an, an.an1, a160, m1)}
    componentes = {c119._etiqueta(f): c3._digest_fichero(f) for f in sorted(set(cierre) | extra)}
    for nombre, ruta in c119._datos_del_entorno().items():
        componentes[f"dato:{nombre}"] = c3._digest_fichero(ruta)
    for nombre, version in c119._versiones_en_el_digest().items():
        componentes[f"version:{nombre}"] = version
    return componentes


def declaracion_coherente(brazo: str, registro: dict) -> list[str]:
    """«Declarar lo que PASÓ»: cada brazo declara SU red. Devuelve los problemas (vacío = coherente)."""
    efectivo = registro.get("entrenamiento_efectivo") or {}
    arq = registro.get("arquitectura") or {}
    parada = (registro.get("hiperparametros") or {}).get("parada_temprana") or {}
    cap = efectivo.get("capacidad")
    problemas = []
    if efectivo.get("parada_por_metrica") is not None or parada.get("metrica") != "validation_loss_del_ensamblado":
        problemas.append("el brazo no para por la pérdida (el C1 es otra palanca y no se mezcla)")
    if brazo == an.REFERENCIA:
        if cap is not None or arq.get("k") != an.K_DE_LA_REFERENCIA:
            problemas.append(f"la referencia no es una red de k={an.K_DE_LA_REFERENCIA} sin capacidad (k={arq.get('k')}, capacidad={cap})")
    elif not cap or cap.get("pedida") is not True or cap.get("k") != an.K_DE_LA_PALANCA or arq.get("k") != an.K_DE_LA_PALANCA \
            or arq.get("d_block") != cap.get("d_block_de_serie"):
        problemas.append(f"la palanca no declara una red de k={an.K_DE_LA_PALANCA} con el d_block de serie (k={arq.get('k')}, capacidad={cap})")
    return problemas


def _argumentos(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", required=True, help="JSON de salida (ruta ABSOLUTA fuera del árbol en la cola)")
    p.add_argument("--forzar", action="store_true", help="ignora la caché entera (lo viejo se sustituye al medir)")
    p.add_argument("--sonda", action="store_true", help="kc2, 1 pliegue, 8 s por intento: prueba el camino, no mide nada del corte")
    p.add_argument("--solo", default=None, help="nombres separados por coma, dentro de los 13 (probar, o continuar)")
    p.add_argument("--memoria-max-mb", type=float, default=MEMORIA_MAX_MB_POR_OMISION,
                   help="techo BLANDO: si el pico de memoria de un intento o del padre lo pasa, se para (el duro es el de systemd-run)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _argumentos(argv)
    es_sonda = bool(args.sonda)
    ruta_salida = Path(args.salida).expanduser().resolve()

    sha_regla = m1._sha256_fichero(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None
    if not es_sonda:
        if an.REGLA_SHA256_SELLADA is None:
            raise SystemExit("la regla NO está sellada (`analiza_c162_c2.REGLA_SHA256_SELLADA` es None): una medida real no empieza "
                             "sin el sello. El supervisor sella ANTES de encolar")
        if sha_regla != an.REGLA_SHA256_SELLADA:
            raise SystemExit(f"REGLA_C2.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la regla es "
                             f"una enmienda con su sello, no una edición. No se mide")

    c6.preparar_protocolo_v2()
    protocolos = c119.exigir_los_protocolos_encadenados()
    configuracion_del_motor = c119.exigir_la_configuracion_de_la_enmienda()  # lee las CONSTANTES del módulo: siguen en k=8 (la palanca es por instancia)
    enmienda_2 = c119.exigir_la_enmienda_2("humo" if es_sonda else "pasada")
    protocolo = c3.protocolo_registrado()
    c5.datasets_de_la_pasada(protocolo)
    no_sellados = {d.nombre: d for d in c6.datasets_no_sellados(protocolo)}

    nombres = list(an.conjuntos_del_cubo())
    if len(nombres) != an.N_CONJUNTOS_DEL_CUBO:
        raise SystemExit(f"el congelado trae {len(nombres)} conjuntos del cubo pequeño, no {an.N_CONJUNTOS_DEL_CUBO}")
    pedidos = list(CONJUNTOS_DE_LA_SONDA) if es_sonda else list(nombres)
    if args.solo:
        pedidos = [s.strip() for s in args.solo.split(",") if s.strip()]
    for n in pedidos:
        if n not in nombres:
            raise SystemExit(f"{n!r} no está entre los {an.N_CONJUNTOS_DEL_CUBO} del cubo pequeño (los medianos, los grandes y los sellados "
                             f"se NIEGAN)")
        if n not in no_sellados:
            raise SystemExit(f"{n!r} es un conjunto SELLADO: no se toca")
    datasets = [no_sellados[n] for n in pedidos]
    if any(d.cubo != an.CUBO for d in datasets):
        raise SystemExit("hay un conjunto que no es del cubo pequeño")
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)

    los_motores = motores()
    if tuple(los_motores) != an.BRAZOS or los_motores[an.REFERENCIA].k_pedido is not None or \
            los_motores[an.PALANCA].k_pedido != an.K_DE_LA_PALANCA or \
            {los_motores[b].parada for b in los_motores} != {"perdida"} or los_motores[an.PALANCA].d_block_pedido is not None:
        raise SystemExit("los brazos no son los que el analizador espera (referencia de serie, palanca k=32, los dos parando por la pérdida)")
    componentes = componentes_del_digest()
    entorno_digest = c119._digest_de_componentes(componentes)
    digest_motor = m1.digest_del_motor()
    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor_tabm": digest_motor},
        datos_de_entrada={**{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
                          "protocolo_exploratorio_v2": c119.RUTA_DEL_PROTOCOLO_V2,
                          "protocolo_119_v4": c119.RUTA_DEL_PROTOCOLO_V4,
                          "protocolo_119_v4_enmienda_1": c119.RUTA_DE_LA_ENMIENDA_1})
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
    sha_analizador = m1._sha256_fichero(RUTA_DEL_ANALIZADOR)
    sha_dependencias = an.dependencias_de_ahora()

    def guardar(parcial: bool) -> None:
        ultimo_control[0] = time.perf_counter()
        c119._escribir_atomicamente(ruta_salida, {
            "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "corte": CORTE,
            "tipo_de_ejecucion": "sonda" if es_sonda else "pasada", "es_sonda": es_sonda, "parcial": parcial,
            "repeticion": REPETICION,
            "que_es": ("TabM ref k=8 (tabm_k8) frente a TabM con k=32 miembros (tabm_k32), los dos parando por la pérdida, métrica de la "
                       "prueba fija de cada pliegue; la regla, en analiza_c162_c2.py"),
            "brazos": list(an.BRAZOS), "nombre_del_motor": los_motores[an.REFERENCIA].nombre,
            "conjuntos": conjuntos_decl, "protocolo_digest_sha256": protocolo.digest(),
            "protocolos_encadenados": protocolos, "enmienda_2_cuadra": enmienda_2.get("cuadra"),
            "configuracion_del_motor": configuracion_del_motor, "estimacion": ESTIMACION,
            "digest_de_la_cache": {"entorno": entorno_digest, "motor": digest_motor, "componentes": componentes},
            "regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "regla_sha256_al_medir": sha_regla,
            "analizador_sha256_al_medir": sha_analizador, "dependencias_sha256_al_medir": sha_dependencias,
            "guion_sha256": m1._sha256_fichero(Path(__file__)), "procedencia": procedencia,
            "total_wall_s": round(time.perf_counter() - inicio, 1), "n_reusados": reusados[0],
            "rss_pico_del_padre_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "resultados": sorted(registros.values(), key=lambda r: (r["dataset"], r["pliegue"], r["motor"])),
        })

    def vigilar_memoria(etiqueta: str, rss_hijo_mb) -> None:
        padre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        for quien, v in (("padre", padre), ("hijo", rss_hijo_mb)):
            if v is not None and v > args.memoria_max_mb:
                guardar(parcial=True)
                raise SystemExit(f"{etiqueta}: el pico de memoria del {quien} ({v:.0f} MB) pasa el techo blando "
                                 f"({args.memoria_max_mb:.0f} MB). Se para; lo medido queda guardado")

    print(f"{CORTE} ({'SONDA' if es_sonda else 'PASADA'}): {len(datasets)} conjuntos, brazos {list(an.BRAZOS)}, entorno "
          f"{entorno_digest}. Salida: {ruta_salida}", flush=True)

    for ds in datasets:
        t_carga = time.perf_counter()
        por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
        carga_s = time.perf_counter() - t_carga
        c119.exigir_la_particion_de_la_v2(ds, particion, c3.LECTURA_DECLARADA.get(str(ds.data_id)), c119._payload_v2())
        test_ids = propuesta.plan.observaciones_del_rol("test")
        metric_id = metrica_por_dataset[ds.nombre]
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        plan = [i for i in c119.plan_de_intentos(ds, propuesta, protocolo, humo=False) if i.repeticion == REPETICION]
        if es_sonda:
            plan = [i for i in plan if i.pliegue < PLIEGUES_DE_LA_SONDA]
        if not plan:
            raise SystemExit(f"{ds.nombre}: la partición no trae pliegues en la repetición {REPETICION}")
        conjuntos_decl.append({
            "nombre": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado,
            "metrica_de_cierre": metric_id, "pliegues_esperados": [i.pliegue for i in plan],
            "n_filas": len(por_id), "n_test": len(test_ids)})
        print(f"\n=== {ds.nombre} ({ds.cubo}, {ds.tarea}, n={len(por_id)}, metrica={metric_id}, pliegues={len(plan)}, "
              f"carga={carga_s:.1f}s) ===", flush=True)

        for ip in plan:
            wall_s = PRESUPUESTO_DE_LA_SONDA_S if es_sonda else ip.presupuesto_wall_s
            pendientes = []
            for brazo in los_motores:
                k = (ds.nombre, brazo, REPETICION, ip.pliegue)
                p = previo.get(k)
                if m1._reutilizable(p, entorno_digest, digest_motor, wall_s, datos_sha256):
                    registros[k] = dict(p, reusado=True)
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} {brazo} pliegue={ip.pliegue}: estado={p.get('estado')}", flush=True)
                else:
                    pendientes.append(brazo)
            for brazo in pendientes:
                motor = los_motores[brazo]
                # Los dos brazos son el MISMO motor con otra red: misma preparación, partición y semilla (la del intento).
                t_prep = time.perf_counter()
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
                presupuesto = Presupuesto(wall_seconds=wall_s, hilos=c3.HILOS_POR_INTENTO, seed=ip.semilla)
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto, candidate=f"{brazo}-ref",
                    split_plan_digest=propuesta.plan.digest(), dataset=ds.nombre, pliegue=ip.pliegue, repeticion=REPETICION)
                transcurrido = time.perf_counter() - t0
                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                config = intento.config_efectiva or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": brazo, "nombre_del_motor": motor.nombre, "repeticion": REPETICION, "pliegue": ip.pliegue,
                    "estado": intento.estado, "semilla": ip.semilla, "presupuesto_wall_s": wall_s,
                    "metrica_de_cierre": metric_id, "wall_s": round(transcurrido, 3),
                    "preparacion_s": round(preparacion_s, 3), "metricas": metricas,
                    "tiempo_de_ajuste": recursos.get("wall_seconds"), "cpu_segundos": recursos.get("cpu_seconds"),
                    "rss_pico_mb": recursos.get("peak_ram_mb"),
                    "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
                    "arquitectura": config.get("arquitectura"), "hiperparametros": config.get("hiperparametros"),
                    "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                    "engine_version": intento.engine_version, "split_plan_digest": propuesta.plan.digest(),
                    "entorno_digest": entorno_digest, "motor_digest": digest_motor, "datos_sha256": datos_sha256,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False, "n_test": len(test_ids),
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                registros[clave(registro)] = registro
                if intento.estado in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
                    problemas = declaracion_coherente(brazo, registro)
                    if problemas:
                        guardar(parcial=True)
                        raise SystemExit(f"{ds.nombre} {brazo} pliegue {ip.pliegue}: declara otra cosa que su brazo ({problemas})")
                efectivo = registro["entrenamiento_efectivo"] or {}
                print(f"  {ds.nombre} {brazo:9s} pliegue={ip.pliegue}: estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"epocas={efectivo.get('epocas_ejecutadas')} mejor={efectivo.get('mejor_epoca')} "
                      f"plazo={efectivo.get('parado_por_plazo')} k={(registro['arquitectura'] or {}).get('k')} wall={transcurrido:.1f}s "
                      f"rss={registro['rss_pico_mb']}", flush=True)
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
    print(f"Guardado en {ruta_salida}. Siguiente: python3 benchmarks/c162/analiza_c162_c2.py {ruta_salida}", flush=True)


if __name__ == "__main__":
    main()
