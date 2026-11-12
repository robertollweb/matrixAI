#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C1 — la PASADA que mide TabM `ref` parando por la PÉRDIDA (la referencia) y parando por la MÉTRICA de cierre (la palanca).

QUÉ HACE. Para cada uno de los 23 conjuntos NO sellados pequeños y medianos de la Fase 0 (los mismos que el 155 y el 160, leídos del
congelado `resultado_pasada_119_c3.json`), repetición 0, TODOS los pliegues (yeast 0–3, el resto 0–4), con las MISMAS particiones,
semillas, preparación y presupuesto de pared por cubo (120 s pequeños, 300 s medianos, 4 hilos) que 119-C3/155/160, ejecuta TabM `ref`
(`matrixai.dense.tabm_cpu`) en sus DOS brazos con el arnés de la Fase 0 (`ejecutar_intento_aislado`, subproceso con su tope):
`tabm_perdida` (`MotorDensaTabM()`: para y entrega por la pérdida de validación, como hoy) y `tabm_metrica`
(`MotorDensaTabM().con_parada_por_metrica()`: por AUROC / accuracy / R² de validación). Por pliegue, la referencia primero y la palanca
inmediatamente después (misma carga de la máquina). La regla la aplica `analiza_c162_c1.py`: este guion solo mide y guarda.

NO hay volcado de predicciones (a diferencia del 160): no se mezcla nada, la métrica de la prueba es la que el arnés ya mide.

CACHÉ Y REANUDACIÓN (como el 155/160). Cada intento lleva `entorno_digest` (cierre de imports de ESTE guion en los dos repos + el
analizador y el del 160 + protocolos + versiones), `motor_digest` (los ficheros del motor TabM), el sha256 del ARFF y el presupuesto. Se
reutiliza solo si todo coincide Y su estado cuenta como medida: un intento fallido o cortado NUNCA se reutiliza. La salida se escribe
atómica y parcial (cada conjunto y cada minuto): relanzar con la misma `--salida` continúa. `--forzar` ignora la caché. Tocar este guion o
el analizador invalida todo.

SELLO. `analiza_c162_c1.REGLA_SHA256_SELLADA` es el sha256 de `REGLA_C1.md`. Una medida real se NIEGA a empezar si es None o si el fichero
ya no es ése; la salida lleva `regla_sha256_al_medir`, el del analizador y los de lo que importa.

DURACIÓN ESTIMADA (no medida): TabM `ref` suma 6.009 s de `wall_s` en los 23 conjuntos (`resultado_c155.json`) y la palanca no se
espera más lenta que eso (para antes o igual, más el coste de la métrica por época) -> ≈ 2 × 6.000 s + carga/preparación de datos ≈ 400 s
(una vez por brazo y pliegue: ≈ 2 × 115 intentos de preparación) ≈ 12.600 s ≈ 3,5 h secuencial. Tope propuesto 5 h (18.000 s) y 6 GB de
memoria; ver INFORME.md.

    python3 benchmarks/c162/medir_c162_c1.py --salida <ruta absoluta fuera del árbol>          # la medida (a la COLA)
    python3 benchmarks/c162/medir_c162_c1.py --sonda --salida <ruta>    # 1 conjunto (kc2), 1 pliegue, 8 s: prueba el camino
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
for _p in (_AQUI, _AQUI.parent / "c160", _FASE0):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c160 as a160  # noqa: E402  (los 23 del congelado; el digest cubre su fichero)
import analiza_c162_c1 as an  # noqa: E402  (la regla, los sellos y los nombres de los brazos)
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c119  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa_tabm import MotorDensaTabM  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

CORTE = "162-C1"
REPETICION = 0
N_CONJUNTOS_ESPERADOS = an.N_CONJUNTOS
RUTA_DEL_CONGELADO = a160.RUTA_DEL_CONGELADO
RUTA_DE_LA_REGLA = an.RUTA_DE_LA_REGLA
RUTA_DEL_ANALIZADOR = an.RUTA_DEL_ANALIZADOR

CONJUNTOS_DE_LA_SONDA = ("kc2",)
PLIEGUES_DE_LA_SONDA = 1
PRESUPUESTO_DE_LA_SONDA_S = 8.0
MEMORIA_MAX_MB_POR_OMISION = 6000.0
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0


def _sha256_fichero(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def conjuntos_del_congelado() -> list[str]:
    return list(a160.conjuntos_y_pliegues_del_congelado())


def motores() -> dict:
    """Los dos brazos, en el orden de `analiza_c162_c1.BRAZOS` (la REFERENCIA primero). Ambos son TabM `ref` (el motor sin cambios de
    receta); el segundo, con la palanca. La etiqueta del brazo NO es el nombre del motor (los dos son `matrixai.dense.tabm_cpu`)."""
    return {an.REFERENCIA: MotorDensaTabM(), an.PALANCA: MotorDensaTabM().con_parada_por_metrica()}


def digest_del_motor() -> str:
    """Los ficheros del motor TabM (motor, red, preparación): los mismos para los dos brazos, que solo se distinguen por la palanca."""
    return c119._digest_motor_nuevo()


def clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["repeticion"], r["pliegue"])


def dependencias_sha256() -> dict[str, str]:
    return an.dependencias_de_ahora()


def componentes_del_digest() -> dict[str, str]:
    """Cierre de imports de ESTE guion en los dos repos + los compartidos + los ficheros del motor + el analizador y el del 160 (que el
    cierre no sigue: `benchmarks/c160` y `benchmarks/c162` no son raíz de import de la Fase 0) + los datos y versiones."""
    cierre, _ = c119._cierre_de_imports(Path(__file__).resolve(), c119._raices_de_import())
    extra = {Path(p).resolve() for p in c5._FICHEROS_COMPARTIDOS + c119._FICHEROS_DEL_MOTOR_NUEVO}
    extra |= {Path(an.__file__).resolve(), Path(a160.__file__).resolve()}
    componentes = {c119._etiqueta(f): c3._digest_fichero(f) for f in sorted(set(cierre) | extra)}
    for nombre, ruta in c119._datos_del_entorno().items():
        componentes[f"dato:{nombre}"] = c3._digest_fichero(ruta)
    for nombre, version in c119._versiones_en_el_digest().items():
        componentes[f"version:{nombre}"] = version
    return componentes


def _argumentos(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--salida", required=True, help="JSON de salida (ruta ABSOLUTA fuera del árbol en la cola)")
    p.add_argument("--forzar", action="store_true", help="ignora la caché entera (lo viejo se sustituye al medir)")
    p.add_argument("--sonda", action="store_true",
                   help=f"{', '.join(CONJUNTOS_DE_LA_SONDA)}, {PLIEGUES_DE_LA_SONDA} pliegue, {PRESUPUESTO_DE_LA_SONDA_S:g} s por "
                        f"intento: prueba el camino de punta a punta, no mide nada del corte")
    p.add_argument("--solo", default=None, help="nombres separados por coma, dentro de los 23 (probar, o continuar; con --sonda "
                                                "sustituye a los de la sonda)")
    p.add_argument("--memoria-max-mb", type=float, default=MEMORIA_MAX_MB_POR_OMISION,
                   help="techo BLANDO: si el pico de memoria de un intento o del padre lo pasa, se para (el duro es el de systemd-run)")
    return p.parse_args(argv)


def declaracion_de_la_parada_coherente(brazo: str, registro: dict) -> list[str]:
    """«Declarar lo que PASÓ»: cada brazo declara SU parada. Devuelve los problemas (vacío = coherente). El de la referencia no lleva
    `parada_por_metrica` y para por `validation_loss_del_ensamblado`; el de la palanca la lleva con `pedida` y, si no cayó a la pérdida,
    para por `<métrica>_de_validacion_del_ensamblado`."""
    efectivo = registro.get("entrenamiento_efectivo") or {}
    parada = (registro.get("hiperparametros") or {}).get("parada_temprana") or {}
    declarada = efectivo.get("parada_por_metrica")
    problemas = []
    if brazo == an.REFERENCIA:
        if declarada is not None:
            problemas.append("la referencia declara `parada_por_metrica`")
        if parada.get("metrica") != "validation_loss_del_ensamblado":
            problemas.append(f"la referencia no para por la pérdida ({parada.get('metrica')!r})")
    else:
        if not declarada or declarada.get("pedida") is not True:
            problemas.append("la palanca no declara haber pedido la parada por métrica")
        elif declarada.get("efectiva") == "metrica":
            esperada = f"{registro.get('metrica_de_cierre')}_de_validacion_del_ensamblado"
            if parada.get("metrica") != esperada:
                problemas.append(f"la palanca dice parar por {parada.get('metrica')!r}, no por {esperada!r}")
        elif parada.get("metrica") != "validation_loss_del_ensamblado":
            problemas.append("la palanca cayó a la pérdida y no lo declara en la parada")
    return problemas


def _reutilizable(previo, entorno, motor_d, wall_s, datos_sha) -> bool:
    return bool(c119._reusable_c3(previo, entorno, motor_d, wall_s, datos_sha256=datos_sha))


def main(argv=None) -> None:
    args = _argumentos(argv)
    es_sonda = bool(args.sonda)
    ruta_salida = Path(args.salida).expanduser().resolve()

    sha_regla = _sha256_fichero(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None
    if not es_sonda:
        if an.REGLA_SHA256_SELLADA is None:
            raise SystemExit("la regla NO está sellada (`analiza_c162_c1.REGLA_SHA256_SELLADA` es None): una medida real no empieza "
                             "sin el sello. El supervisor sella ANTES de encolar")
        if sha_regla != an.REGLA_SHA256_SELLADA:
            raise SystemExit(f"REGLA_C1.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): cambiar la "
                             f"regla es una enmienda con su sello, no una edición. No se mide")

    c6.preparar_protocolo_v2()
    protocolos = c119.exigir_los_protocolos_encadenados()
    configuracion_del_motor = c119.exigir_la_configuracion_de_la_enmienda()
    enmienda_2 = c119.exigir_la_enmienda_2("humo" if es_sonda else "pasada")
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)  # noqa: F841  (como el 155/160: fija el catálogo antes de resolver nombres)
    no_sellados = {d.nombre: d for d in c6.datasets_no_sellados(protocolo)}

    nombres = conjuntos_del_congelado()
    if len(nombres) != N_CONJUNTOS_ESPERADOS:
        raise SystemExit(f"el congelado trae {len(nombres)} conjuntos pequeños/medianos no sellados, no {N_CONJUNTOS_ESPERADOS}")
    pedidos = list(CONJUNTOS_DE_LA_SONDA) if es_sonda else list(nombres)
    if args.solo:
        pedidos = [s.strip() for s in args.solo.split(",") if s.strip()]
    for n in pedidos:
        if n not in nombres:
            raise SystemExit(f"{n!r} no está entre los {N_CONJUNTOS_ESPERADOS} de la medida (los sellados y los grandes se NIEGAN)")
        if n not in no_sellados:
            raise SystemExit(f"{n!r} es un conjunto SELLADO: no se toca")
    datasets = [no_sellados[n] for n in pedidos]
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)

    los_motores = motores()
    if tuple(los_motores) != an.BRAZOS or los_motores[an.REFERENCIA].parada != "perdida" or los_motores[an.PALANCA].parada != "metrica":
        raise SystemExit("los brazos no son los que el analizador espera (referencia por pérdida, palanca por métrica)")
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
    sha_analizador = _sha256_fichero(RUTA_DEL_ANALIZADOR)
    sha_dependencias = dependencias_sha256()

    def guardar(parcial: bool) -> None:
        ultimo_control[0] = time.perf_counter()
        c119._escribir_atomicamente(ruta_salida, {
            "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "corte": CORTE,
            "tipo_de_ejecucion": "sonda" if es_sonda else "pasada", "es_sonda": es_sonda, "parcial": parcial,
            "repeticion": REPETICION,
            "que_es": ("TabM ref parando por la pérdida (tabm_perdida) y por la métrica de cierre (tabm_metrica), métrica de la prueba "
                       "fija de cada pliegue; la regla, en analiza_c162_c1.py"),
            "brazos": list(an.BRAZOS), "nombre_del_motor": los_motores[an.REFERENCIA].nombre,
            "conjuntos": conjuntos_decl, "protocolo_digest_sha256": protocolo.digest(),
            "protocolos_encadenados": protocolos, "enmienda_2_cuadra": enmienda_2.get("cuadra"),
            "configuracion_del_motor": configuracion_del_motor,
            "digest_de_la_cache": {"entorno": entorno_digest, "motor": digest_motor, "componentes": componentes},
            "regla_sha256_sellada": an.REGLA_SHA256_SELLADA, "regla_sha256_al_medir": sha_regla,
            "analizador_sha256_al_medir": sha_analizador, "dependencias_sha256_al_medir": sha_dependencias,
            "guion_sha256": _sha256_fichero(Path(__file__)), "procedencia": procedencia,
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
                if _reutilizable(p, entorno_digest, digest_motor, wall_s, datos_sha256):
                    registros[k] = dict(p, reusado=True)
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} {brazo} pliegue={ip.pliegue}: estado={p.get('estado')}", flush=True)
                else:
                    pendientes.append(brazo)
            for brazo in pendientes:
                motor = los_motores[brazo]
                # Los dos brazos son el MISMO motor: misma preparación, misma partición, misma semilla (la del intento).
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
                    problemas = declaracion_de_la_parada_coherente(brazo, registro)
                    if problemas:
                        guardar(parcial=True)
                        raise SystemExit(f"{ds.nombre} {brazo} pliegue {ip.pliegue}: declara otra cosa que su brazo ({problemas})")
                efectivo = registro["entrenamiento_efectivo"] or {}
                pm = efectivo.get("parada_por_metrica") or {}
                print(f"  {ds.nombre} {brazo:12s} pliegue={ip.pliegue}: estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"epocas={efectivo.get('epocas_ejecutadas')} mejor={efectivo.get('mejor_epoca')} "
                      f"plazo={efectivo.get('parado_por_plazo')} efectiva={pm.get('efectiva', '-')} wall={transcurrido:.1f}s "
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
    print(f"Guardado en {ruta_salida}. Siguiente: python3 benchmarks/c162/analiza_c162_c1.py {ruta_salida}", flush=True)


if __name__ == "__main__":
    main()
