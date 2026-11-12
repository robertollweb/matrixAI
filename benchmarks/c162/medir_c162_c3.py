#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C3 — la PASADA que mide TabM `ref` (PLR, k = 8) frente a TabM-mini con entrada LINEAL A TROZOS (PLE) en los 23 conjuntos no sellados.

QUÉ HACE. Para cada uno de los 23 conjuntos NO sellados pequeños y medianos de la Fase 0 (los mismos que el 155, el 160 y el C1, leídos del
congelado `resultado_pasada_119_c3.json`), repetición 0, TODOS los pliegues (yeast 0–3, el resto 0–4), con las MISMAS particiones, semillas,
preparación y presupuesto de pared por cubo (120 s pequeños, 300 s medianos, 4 hilos), ejecuta TabM (`matrixai.dense.tabm_cpu`) en sus DOS
brazos con el arnés de la Fase 0 (`ejecutar_intento_aislado`): `tabm_ref` (`MotorDensaTabM()`, la referencia) y `tabm_mini_ple`
(`MotorDensaTabM().con_variante("mini_ple")`: un solo adaptador por miembro, a la entrada, y embeddings lineales a trozos con bordes por
cuantiles de TRAIN). Por pliegue, la referencia primero y la palanca inmediatamente después (misma carga).

LA PARADA ES LA DE SIEMPRE (por la PÉRDIDA de validación del ensamblado) en los dos brazos, y k = 8 en ambos: las otras palancas (C1 parada por
métrica, C2 capacidad) NO se mezclan; el analizador rechaza un intento que las declare. La regla la aplica `analiza_c162_c3.py`.

REUTILIZA el arnés del C1 (`medir_c162_c1`: caché por digest, dependencias, congelado, escritura atómica) y el analizador del C1 por import;
el bucle por pliegue es propio por la misma razón que el C2 (el `main` del C1 trae sus dos brazos escritos dentro).

EL PLAZO ES DE RELOJ DE PARED. La red mini + PLE tiene menos parámetros por miembro (sin `r`/`s` por capa, sin frecuencias de 2 x 48 por
columna), así que a igual plazo hace más épocas, o las mismas antes (ver la sonda de coste en el INFORME). Es parte de lo que se mide ((d) y
`parados_por_plazo` lo declaran).

CACHÉ Y REANUDACIÓN, SELLO: como el C1 (`--forzar`, `--sonda`, `--solo`). `REGLA_SHA256_SELLADA` None = sin sellar: una medida real no empieza.

DURACIÓN ESTIMADA (no medida): ver `ESTIMACION` abajo y el INFORME.

    python3 benchmarks/c162/medir_c162_c3.py --salida <ruta absoluta fuera del árbol>          # la medida (a la COLA)
    python3 benchmarks/c162/medir_c162_c3.py --sonda --salida <ruta>    # 1 conjunto (kc2), 1 pliegue, 8 s: prueba el camino
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
sys.path.insert(0, str(_AQUI))

import medir_c162_c1 as m1  # noqa: E402  (el arnés del C1; pone en `sys.path` lo demás)
import analiza_c162_c3 as an  # noqa: E402
import analiza_c160 as a160  # noqa: E402
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c119  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa_tabm import MotorDensaTabM  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

CORTE = "162-C3"
REPETICION = 0
RUTA_DE_LA_REGLA = an.RUTA_DE_LA_REGLA
RUTA_DEL_ANALIZADOR = an.RUTA_DEL_ANALIZADOR
CONJUNTOS_DE_LA_SONDA = ("kc2",)
PLIEGUES_DE_LA_SONDA = 1
PRESUPUESTO_DE_LA_SONDA_S = 8.0
MEMORIA_MAX_MB_POR_OMISION = 6000.0
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0

#: DURACIÓN ESTIMADA (no medida): TabM `ref` suma 6.009 s de `wall_s` en los 23 conjuntos (`resultado_c155.json`) y la palanca se espera más
#: barata por época (ver la sonda en el INFORME) pero el plazo es de pared: el peor caso es el mismo gasto. -> ≈ 2 x 6.000 s + preparación
#: (≈ 2 x 115 intentos) ≈ 12.600 s ≈ 3,5 h, SIN medir. Tope propuesto 5 h (18.000 s), 6 GB (techo duro y blando), como el C1.
ESTIMACION = {"duracion_s_estimada": 12600, "tope_s_propuesto": 18000, "memoria_mb_propuesta": 6000}


def motores() -> dict:
    """Los dos brazos, en el orden de `analiza_c162_c3.BRAZOS` (la REFERENCIA primero). La parada es la de la pérdida y k = 8 en ambos."""
    return {an.REFERENCIA: MotorDensaTabM(), an.PALANCA: MotorDensaTabM().con_variante(an.VARIANTE)}


#: La red de la palanca NO está en los ficheros del motor que el digest del 119 cubre (`_FICHEROS_DEL_MOTOR_NUEVO`): se añade AQUÍ, a mano,
#: al digest del motor y al del entorno, para que tocarla invalide la caché (y una prueba lo vigila).
FICHERO_DE_LA_RED_DE_LA_PALANCA = c3._DIR_ENGINES / "redes" / "tabm_mini_ple.py"


def digest_del_motor() -> str:
    return hashlib.sha256((m1.digest_del_motor() + c3._digest_fichero(FICHERO_DE_LA_RED_DE_LA_PALANCA.resolve())).encode()).hexdigest()[:16]


def clave(r: dict) -> tuple:
    return m1.clave(r)


def componentes_del_digest() -> dict[str, str]:
    """El del C1 (cierre de imports de ESTE guion + compartidos + motor + analizadores) más el analizador de este corte y el del C1 +
    el guion del C1, que el cierre no sigue (`benchmarks/c162` no es raíz de import de la Fase 0)."""
    cierre, _ = c119._cierre_de_imports(Path(__file__).resolve(), c119._raices_de_import())
    extra = {Path(p).resolve() for p in c5._FICHEROS_COMPARTIDOS + c119._FICHEROS_DEL_MOTOR_NUEVO}
    extra |= {Path(x.__file__).resolve() for x in (an, an.an1, a160, m1)} | {FICHERO_DE_LA_RED_DE_LA_PALANCA.resolve()}
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
    var = efectivo.get("variante")
    problemas = []
    if efectivo.get("parada_por_metrica") is not None or parada.get("metrica") != "validation_loss_del_ensamblado":
        problemas.append("el brazo no para por la pérdida (el C1 es otra palanca y no se mezcla)")
    if efectivo.get("capacidad") is not None or arq.get("k") != an.K_DE_LOS_DOS:
        problemas.append(f"el brazo declara capacidad o k != {an.K_DE_LOS_DOS} (el C2 es otra palanca y no se mezcla)")
    if brazo == an.REFERENCIA:
        if var is not None or arq.get("variante") is not None or "n_frequencies" not in arq:
            problemas.append("la referencia declara una variante o no es la red PLR de siempre")
    elif not var or var.get("pedida") is not True or var.get("nombre") != an.VARIANTE or arq.get("variante") != an.VARIANTE \
            or var.get("bordes_de") != "train" or not arq.get("bordes_ple"):
        problemas.append(f"la palanca no declara la red {an.VARIANTE} con bordes de train (variante={var}, arquitectura.variante={arq.get('variante')})")
    return problemas


def _argumentos(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", required=True, help="JSON de salida (ruta ABSOLUTA fuera del árbol en la cola)")
    p.add_argument("--forzar", action="store_true", help="ignora la caché entera (lo viejo se sustituye al medir)")
    p.add_argument("--sonda", action="store_true", help="kc2, 1 pliegue, 8 s por intento: prueba el camino, no mide nada del corte")
    p.add_argument("--solo", default=None, help="nombres separados por coma, dentro de los 23 (probar, o continuar)")
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
            raise SystemExit("la regla NO está sellada (`analiza_c162_c3.REGLA_SHA256_SELLADA` es None): una medida real no empieza "
                             "sin el sello. El supervisor sella ANTES de encolar")
        if sha_regla != an.REGLA_SHA256_SELLADA:
            raise SystemExit(f"REGLA_C3.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la regla es "
                             f"una enmienda con su sello, no una edición. No se mide")

    c6.preparar_protocolo_v2()
    protocolos = c119.exigir_los_protocolos_encadenados()
    configuracion_del_motor = c119.exigir_la_configuracion_de_la_enmienda()  # lee las CONSTANTES del módulo (la palanca es por instancia)
    enmienda_2 = c119.exigir_la_enmienda_2("humo" if es_sonda else "pasada")
    protocolo = c3.protocolo_registrado()
    c5.datasets_de_la_pasada(protocolo)
    no_sellados = {d.nombre: d for d in c6.datasets_no_sellados(protocolo)}

    nombres = list(an.conjuntos_del_cubo())
    if len(nombres) != an.N_CONJUNTOS_DEL_CUBO:
        raise SystemExit(f"el congelado trae {len(nombres)} conjuntos, no {an.N_CONJUNTOS_DEL_CUBO}")
    pedidos = list(CONJUNTOS_DE_LA_SONDA) if es_sonda else list(nombres)
    if args.solo:
        pedidos = [s.strip() for s in args.solo.split(",") if s.strip()]
    for n in pedidos:
        if n not in nombres:
            raise SystemExit(f"{n!r} no está entre los {an.N_CONJUNTOS_DEL_CUBO} de la medida (los sellados y los grandes se NIEGAN)")
        if n not in no_sellados:
            raise SystemExit(f"{n!r} es un conjunto SELLADO: no se toca")
    datasets = [no_sellados[n] for n in pedidos]
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)

    los_motores = motores()
    if tuple(los_motores) != an.BRAZOS or los_motores[an.REFERENCIA].variante is not None or \
            los_motores[an.PALANCA].variante != an.VARIANTE or \
            {los_motores[b].parada for b in los_motores} != {"perdida"} or \
            any(los_motores[b].k_pedido is not None or los_motores[b].d_block_pedido is not None for b in los_motores):
        raise SystemExit("los brazos no son los que el analizador espera (referencia de serie, palanca mini_ple, los dos por la pérdida y de serie)")
    componentes = componentes_del_digest()
    entorno_digest = c119._digest_de_componentes(componentes)
    digest_motor = digest_del_motor()
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
            "que_es": ("TabM ref con PLR (tabm_ref) frente a TabM-mini con PLE (tabm_mini_ple), k=8 y parando por la pérdida los dos, métrica de la "
                       "prueba fija de cada pliegue; la regla, en analiza_c162_c3.py"),
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
                      f"plazo={efectivo.get('parado_por_plazo')} var={(registro['arquitectura'] or {}).get('variante')} wall={transcurrido:.1f}s "
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
    print(f"Guardado en {ruta_salida}. Siguiente: python3 benchmarks/c162/analiza_c162_c3.py {ruta_salida}", flush=True)


if __name__ == "__main__":
    main()
