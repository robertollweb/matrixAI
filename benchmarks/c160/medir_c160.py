#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""160-C1 — la PASADA que vuelca las predicciones por fila de LightGBM y de TabM (`ref`) para el ensamblado diverso (contrato 160).

QUÉ HACE. Para cada uno de los 23 conjuntos NO sellados pequeños y medianos de la Fase 0 (los mismos que el 155, leídos del
congelado `resultado_pasada_119_c3.json`), repetición 0, TODOS los pliegues (yeast 0–3, el resto 0–4), con las MISMAS particiones,
semillas, preparación por motor y presupuesto de pared por cubo (120 s pequeños, 300 s medianos, 4 hilos) que 119-C3/155, ejecuta
LightGBM (el de la Fase 0, política por omisión) y TabM `ref` (la receta de referencia del producto) con el arnés de la Fase 0
(`ejecutar_intento_aislado`, subproceso con su tope) y guarda, por (conjunto, pliegue, motor), un `.npz` con las filas de la
PRUEBA fija, la verdad, las predicciones y las probabilidades (`volcado_c160.py`) y lo que tardó en predecir. La mezcla y la regla
las aplica `analiza_c160.py`: este guion solo mide y guarda.

CACHÉ Y REANUDACIÓN (como el 155). Cada intento lleva `entorno_digest` (cierre de imports de ESTE guion en los dos repos + volcado +
protocolos + versiones), `motor_digest` (el fichero del motor), el sha256 del ARFF y el presupuesto. Se reutiliza solo si todo
coincide, su estado cuenta como medida Y su `.npz` existe con el sha256 registrado: un intento fallido o cortado NUNCA se reutiliza.
La salida se escribe atómica y parcial (cada conjunto y cada minuto): relanzar con la misma `--salida` continúa. `--forzar` ignora la
caché. Tocar este guion o `volcado_c160.py` invalida todo.

SELLO. `analiza_c160.REGLA_SHA256_SELLADA` es el sha256 de `REGLA_DE_DECISION.md`. Una medida real se NIEGA a empezar si es None o si
el fichero ya no es ése; la salida lleva `regla_sha256_al_medir`, el del analizador y los de lo que importa.

DURACIÓN ESTIMADA (no medida): ≈ 6.000 s de TabM (suma de `wall_s` de `ref` en `resultado_c155.json`: 6.009 s) + ≈ 1.000 s de
LightGBM (23 conjuntos × ~5 pliegues, su `wall_s` en la Fase 0) + carga/preparación de datos ≈ 400 s  ->  ≈ 2,1 h secuencial.
Tope propuesto 4 h (14.400 s) y 6 GB de memoria; ver INFORME.md.

    python3 benchmarks/c160/medir_c160.py --salida <ruta absoluta fuera del árbol>          # la medida (a la COLA)
    python3 benchmarks/c160/medir_c160.py --sonda --salida <ruta>    # 1 conjunto (kc2), 1 pliegue, 8 s: prueba el camino
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
_FASE0 = _AQUI.parent / "fase0"
for _p in (_AQUI, _FASE0):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c160 as an  # noqa: E402  (la regla, los sellos y la comprobación de coherencia del volcado)
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as c119  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402
import volcado_c160 as vc  # noqa: E402

from matrixai_engines.motores import densa_tabm  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: E402

CORTE = "160-C1"
REPETICION = 0
CUBOS = ("pequeno", "mediano")
N_CONJUNTOS_ESPERADOS = 23
RUTA_DEL_CONGELADO = _FASE0 / "resultado_pasada_119_c3.json"
RUTA_DE_LA_REGLA = _AQUI / "REGLA_DE_DECISION.md"
RUTA_DEL_ANALIZADOR = _AQUI / "analiza_c160.py"

CONJUNTOS_DE_LA_SONDA = ("kc2",)
PLIEGUES_DE_LA_SONDA = 1
PRESUPUESTO_DE_LA_SONDA_S = 8.0
MEMORIA_MAX_MB_POR_OMISION = 6000.0
SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0
#: El volcado recomputado y la métrica que el arnés midió del MISMO intento tienen que coincidir a esto (misma cadena, mismas filas).
TOLERANCIA_DE_COHERENCIA = 1e-9


def _sha256_fichero(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def conjuntos_del_congelado() -> list[str]:
    d = json.loads(RUTA_DEL_CONGELADO.read_text(encoding="utf-8"))
    return [x["nombre"] for x in d["datasets_declarados"] if x["cubo"] in CUBOS and not x["sellado"]]


def motores():
    """Los dos miembros (D2), en el orden de `analiza_c160.MIEMBROS`: LightGBM de la Fase 0 y TabM `ref` (sin cambios)."""
    return {an.LIGHTGBM: vc.MotorLightGBMQueVuelca(), an.TABM: vc.MotorTabMQueVuelca()}


def digest_del_motor(nombre: str) -> str:
    if nombre == an.TABM:
        return c119._digest_motor_nuevo()
    return c3._digest_fichero(c3._FICHERO_POR_MOTOR[nombre])


def clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["repeticion"], r["pliegue"])


def dependencias_sha256() -> dict[str, str]:
    """Mismas etiquetas que `analiza_c160.dependencias_de_ahora` (una prueba lo fija)."""
    return an.dependencias_de_ahora()


def componentes_del_digest() -> dict[str, str]:
    """Como el 155: cierre de imports de ESTE guion en los dos repos + los compartidos + los datos y versiones; MÁS el volcado
    (que el cierre no sigue: vive en `benchmarks/c160`, que no es raíz de import de la Fase 0)."""
    cierre, _ = c119._cierre_de_imports(Path(__file__).resolve(), c119._raices_de_import())
    extra = {Path(p).resolve() for p in c5._FICHEROS_COMPARTIDOS + c119._FICHEROS_DEL_MOTOR_NUEVO}
    extra |= {Path(vc.__file__).resolve(), c3._DIR_ENGINES / "motores" / "arbol_lightgbm.py"}
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


def _declarado_tabm_coincide(registro: dict) -> list[str]:
    """«Declarar lo que PASÓ»: lo que TabM dice haber entrenado es la receta `ref` (la primera de la rejilla del producto). La rejilla
    (`densa_tabm.CONFIGURACIONES_DEL_AJUSTE`) llegó con el 155-C1: si el motor de ESTE árbol no la tiene, `ref` es el motor sin cambios y
    no hay nada que comparar (se dice en el registro, no se inventa)."""
    rejilla = getattr(densa_tabm, "CONFIGURACIONES_DEL_AJUSTE", None)
    pedido_f = getattr(vc.MotorTabMQueVuelca, "hiperparametros_ajustables", None)
    if rejilla is None or pedido_f is None:
        return []
    ref = dict(rejilla)["ref"]
    motor = vc.MotorTabMQueVuelca().con_hiperparametros(**ref)
    h, a = registro.get("hiperparametros") or {}, registro.get("arquitectura") or {}
    pedido = motor.hiperparametros_ajustables()
    problemas = []
    if h.get("tasa_de_aprendizaje") != pedido["tasa_de_aprendizaje"] or h.get("weight_decay") != pedido["weight_decay"]:
        problemas.append("lr/weight_decay declarados != los de `ref`")
    if a.get("n_blocks") != pedido["n_blocks"]:
        problemas.append(f"n_blocks declarado {a.get('n_blocks')} != {pedido['n_blocks']}")
    return problemas


def _reutilizable(previo, entorno, motor_d, wall_s, datos_sha, directorio: Path) -> bool:
    if not c119._reusable_c3(previo, entorno, motor_d, wall_s, datos_sha256=datos_sha):
        return False
    v = previo.get("volcado") or {}
    ruta = directorio / (v.get("fichero") or "")
    return bool(v.get("fichero")) and ruta.is_file() and _sha256_fichero(ruta) == v.get("sha256")


def main(argv=None) -> None:
    args = _argumentos(argv)
    es_sonda = bool(args.sonda)
    ruta_salida = Path(args.salida).expanduser().resolve()
    directorio_de_volcados = ruta_salida.with_name(ruta_salida.stem + "_volcados")

    sha_regla = _sha256_fichero(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None
    if not es_sonda:
        if an.REGLA_SHA256_SELLADA is None:
            raise SystemExit("la regla NO está sellada (`analiza_c160.REGLA_SHA256_SELLADA` es None): una medida real no empieza "
                             "sin el sello. El supervisor sella ANTES de encolar")
        if sha_regla != an.REGLA_SHA256_SELLADA:
            raise SystemExit(f"REGLA_DE_DECISION.md ({str(sha_regla)[:12]}) no es la sellada ({an.REGLA_SHA256_SELLADA[:12]}): "
                             f"cambiar la regla es una enmienda con su sello, no una edición. No se mide")

    c6.preparar_protocolo_v2()
    protocolos = c119.exigir_los_protocolos_encadenados()
    configuracion_del_motor = c119.exigir_la_configuracion_de_la_enmienda()
    enmienda_2 = c119.exigir_la_enmienda_2("humo" if es_sonda else "pasada")
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)  # noqa: F841  (como el 155: fija el catálogo antes de resolver nombres)
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
    if los_motores[an.TABM].nombre != an.TABM or los_motores[an.LIGHTGBM].nombre != an.LIGHTGBM:
        raise SystemExit("los nombres de los motores no son los que el analizador espera")
    entorno_digest = c119._digest_de_componentes(componentes_del_digest())
    componentes = componentes_del_digest()
    digests_motor = {m: digest_del_motor(m) for m in los_motores}
    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor_tabm": digests_motor[an.TABM],
                           "motor_lightgbm": digests_motor[an.LIGHTGBM]},
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
            "que_es": ("las probabilidades/predicciones por fila de LightGBM y TabM ref sobre la prueba fija de cada pliegue "
                       "(un .npz por intento en `directorio_de_volcados`); la mezcla y la regla, en analiza_c160.py"),
            "miembros": list(an.MIEMBROS), "directorio_de_volcados": str(directorio_de_volcados),
            "conjuntos": conjuntos_decl, "protocolo_digest_sha256": protocolo.digest(),
            "protocolos_encadenados": protocolos, "enmienda_2_cuadra": enmienda_2.get("cuadra"),
            "configuracion_del_motor": configuracion_del_motor,
            "digest_de_la_cache": {"entorno": entorno_digest, "motores": digests_motor, "componentes": componentes},
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

    print(f"{CORTE} ({'SONDA' if es_sonda else 'PASADA'}): {len(datasets)} conjuntos, miembros {list(an.MIEMBROS)}, entorno "
          f"{entorno_digest}. Salida: {ruta_salida}  Volcados: {directorio_de_volcados}", flush=True)

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
            for m in los_motores:
                k = (ds.nombre, m, REPETICION, ip.pliegue)
                p = previo.get(k)
                if _reutilizable(p, entorno_digest, digests_motor[m], wall_s, datos_sha256, directorio_de_volcados):
                    registros[k] = dict(p, reusado=True)
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} {m} pliegue={ip.pliegue}: estado={p.get('estado')}", flush=True)
                else:
                    pendientes.append(m)
            for m in pendientes:
                motor = los_motores[m]
                # La preparación depende del motor (categóricas/faltantes): la de la Fase 0, por motor.
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
                ruta_npz = vc.ruta_del_volcado(directorio_de_volcados, ds.nombre, m, REPETICION, ip.pliegue)
                if ruta_npz.exists():
                    ruta_npz.unlink()  # un volcado viejo nunca se confunde con el de este intento
                vc.con_volcado(motor, ruta_npz, test.row_ids)
                presupuesto = Presupuesto(wall_seconds=wall_s, hilos=c3.HILOS_POR_INTENTO, seed=ip.semilla)
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto, candidate=f"{motor.nombre}-ref",
                    split_plan_digest=propuesta.plan.digest(), dataset=ds.nombre, pliegue=ip.pliegue, repeticion=REPETICION)
                transcurrido = time.perf_counter() - t0
                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                config = intento.config_efectiva or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": motor.nombre, "repeticion": REPETICION, "pliegue": ip.pliegue,
                    "estado": intento.estado, "semilla": ip.semilla, "presupuesto_wall_s": wall_s,
                    "metrica_de_cierre": metric_id, "wall_s": round(transcurrido, 3),
                    "preparacion_s": round(preparacion_s, 3), "metricas": metricas,
                    "tiempo_de_ajuste": recursos.get("wall_seconds"), "cpu_segundos": recursos.get("cpu_seconds"),
                    "rss_pico_mb": recursos.get("peak_ram_mb"),
                    "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
                    "arquitectura": config.get("arquitectura"), "hiperparametros": config.get("hiperparametros"),
                    "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                    "engine_version": intento.engine_version, "split_plan_digest": propuesta.plan.digest(),
                    "entorno_digest": entorno_digest, "motor_digest": digests_motor[m], "datos_sha256": datos_sha256,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False, "volcado": None,
                    "n_test": len(test_ids),
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                registros[clave(registro)] = registro
                if intento.estado in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
                    if not ruta_npz.is_file():
                        guardar(parcial=True)
                        raise SystemExit(f"{ds.nombre} {m} pliegue {ip.pliegue}: el intento terminó ({intento.estado}) y NO dejó "
                                         f"su volcado: el motor volcador no se activó. No se sigue")
                    volcado = vc.leer_volcado(ruta_npz)
                    # Coherencia: lo volcado, medido con la MISMA cadena que el arnés, da la métrica que el arnés midió.
                    recomputada = an.metrica_de(volcado, ds.tarea, metric_id)
                    medida = metricas.get(metric_id)
                    if medida is None or abs(recomputada - float(medida)) > TOLERANCIA_DE_COHERENCIA:
                        guardar(parcial=True)
                        raise SystemExit(f"{ds.nombre} {m} pliegue {ip.pliegue}: el volcado recomputado da {metric_id}="
                                         f"{recomputada!r} y el arnés midió {medida!r}: el volcado no es el de este intento")
                    if volcado["row_ids"] != [str(r) for r in test_ids]:
                        guardar(parcial=True)
                        raise SystemExit(f"{ds.nombre} {m} pliegue {ip.pliegue}: las filas volcadas no son las de la prueba")
                    registro["volcado"] = {"fichero": ruta_npz.name, "sha256": _sha256_fichero(ruta_npz),
                                           "coste_ms_por_fila": an.coste_por_fila_ms(volcado)}
                    if m == an.TABM:
                        problemas = _declarado_tabm_coincide(registro)
                        if problemas:
                            guardar(parcial=True)
                            raise SystemExit(f"{ds.nombre} TabM pliegue {ip.pliegue}: declara otra cosa que `ref` ({problemas})")
                efectivo = registro["entrenamiento_efectivo"] or {}
                print(f"  {ds.nombre} {m[:12]:12s} pliegue={ip.pliegue}: estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"coste={((registro['volcado'] or {}).get('coste_ms_por_fila'))} ms/fila epocas={efectivo.get('epocas_completas')} "
                      f"plazo={efectivo.get('parado_por_plazo')} wall={transcurrido:.1f}s rss={registro['rss_pico_mb']}", flush=True)
                vigilar_memoria(f"{ds.nombre} {m} pliegue {ip.pliegue}", registro["rss_pico_mb"])
                if time.perf_counter() - ultimo_control[0] >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL:
                    guardar(parcial=True)
        guardar(parcial=True)

    en_el_plan = {(c["nombre"], m, REPETICION, f) for c in conjuntos_decl for m in los_motores for f in c["pliegues_esperados"]}
    de_estos = {c["nombre"] for c in conjuntos_decl}
    for k in [k for k in registros if k[0] in de_estos and k not in en_el_plan]:
        del registros[k]
    guardar(parcial=False)
    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total / 3600:.2f} h), {len(registros)} intentos ({reusados[0]} reusados) ===", flush=True)
    print(f"Guardado en {ruta_salida}. Siguiente: python3 benchmarks/c160/analiza_c160.py {ruta_salida}", flush=True)


if __name__ == "__main__":
    main()
