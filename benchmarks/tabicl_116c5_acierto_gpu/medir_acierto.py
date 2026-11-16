# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C5 — el ACIERTO de `tabicl.v2` en GPU, en los conjuntos de la Fase 0 que caben en el techo del paquete GPU.

Corre DENTRO de la imagen (sin red) y recorre TODOS los conjuntos de `conjuntos.json`, uno tras otro, con la MISMA
partición, semillas, repeticiones, presupuesto por cubo y métrica de cierre del protocolo `113-C0.v2` que la pasada de
CPU (116-C2). No reescribe nada de eso: importa `pasada_116c2_tabicl` y, con ella, la partición (`particiones_base`),
la preparación (`preparar_para_motor`), el presupuesto (`wall_seconds_del_cubo`), el aislamiento por proceso
(`ejecutar_intento_aislado`) y la métrica (`metrica_de_cierre_por_dataset`) de la v2. Lo único propio es el BUCLE (el
régimen es el de la GPU, no el de CPU), el reanudado y el log.

    python3 medir_acierto.py --salida /salida/resultado_116c5_acierto_gpu.json          # la medición entera
    python3 medir_acierto.py --estimar                                                 # solo la cuenta, no mide
    python3 medir_acierto.py --prueba --salida /salida/prueba.json                     # 1 conjunto, 1 pliegue

- **Se puede cortar (Ctrl+C) y relanzar la misma orden**: guarda tras CADA pliegue y reaprovecha los pliegues ya
  `completed` (los fallidos se REINTENTAN: un tope agotado por carga ajena no queda cementado).
- **Sin GPU se niega a empezar** (salvo `--permitir-cpu`, que es la sonda del servidor y reduce el techo al de CPU).
- Escribe un log (`log_acierto_gpu.txt`, junto al JSON) con lo mismo que enseña por pantalla.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CODIGO = AQUI / "codigo"
SALIDA_POR_OMISION = Path("/salida/resultado_116c5_acierto_gpu.json")
PESOS_POR_OMISION = "/pesos"

#: Estimación (NO medida) del coste de un pliegue en la GPU del contrato: ver `estimar_pliegue`.
ARRANQUE_DE_UN_INTENTO_S = 25.0   # proceso hijo nuevo: importar torch+tabicl y pasar los pesos (215 MB) a la GPU. SUPUESTO.
LLAMADAS_POR_PLIEGUE = 2          # el contrato mide «ajuste + validación»; aquí además se predice el test: se dobla.
TOPE_DE_CONJUNTO_MIN_S = 600.0
TOPE_DE_CONJUNTO_FACTOR = 3.0

#: Tiempo de UN intento (ajuste + validación) en la RTX 2000 Ada del contrato 116 («MEDIDO EN GPU», 12-11): el
#: MÁXIMO de las tres tareas por tramo de celdas (filas × columnas), el mismo que usa `techo_de_tabicl` en motores.
TABLA_DE_TIEMPOS_S = ((64_000, 4.35), (128_000, 6.69), (256_000, 13.77), (512_000, 13.77))


def estimar_pliegue(celdas: int) -> float:
    """Segundos ESTIMADOS (no medidos) de un pliegue con ese número de celdas de contexto."""
    for tope, segundos in TABLA_DE_TIEMPOS_S:
        if celdas <= tope:
            return ARRANQUE_DE_UN_INTENTO_S + LLAMADAS_POR_PLIEGUE * segundos
    raise ValueError(f"{celdas} celdas: por encima de la tabla medida (512.000)")


def estimar_conjunto(ficha: dict) -> dict:
    por_pliegue = estimar_pliegue(ficha["max_celdas"])
    total = por_pliegue * ficha["n_pliegues"]
    return {"dataset": ficha["dataset"], "n_pliegues": ficha["n_pliegues"], "celdas": ficha["max_celdas"],
            "s_por_pliegue_estimados": round(por_pliegue, 1), "s_del_conjunto_estimados": round(total, 1),
            "tope_del_conjunto_s": round(max(TOPE_DE_CONJUNTO_MIN_S, TOPE_DE_CONJUNTO_FACTOR * total), 1)}


# ---------------------------------------------------------------------------
# entorno: rutas, datos y log
# ---------------------------------------------------------------------------

def preparar_rutas(home: Path) -> None:
    """Las rutas de `codigo/` al `sys.path` y `HOME` apuntando donde `pasada_exploratoria_101_c3` busca los ARFF
    (`~/fase0_openml_datos/arff/<data_id>.arff`). Tiene que ir ANTES de importar esos módulos."""
    for ruta in (CODIGO / "matrixAI" / "benchmarks" / "fase0", CODIGO / "matrixai-engines" / "src", CODIGO / "matrixAI"):
        sys.path.insert(0, str(ruta))
    os.environ["HOME"] = str(home)


def desempaquetar_datos(home: Path) -> dict:
    """gz -> ARFF en `home`, verificando el sha256 del manifiesto (si no casa, NO se mide)."""
    manifiesto = json.loads((AQUI / "datos" / "manifiesto.json").read_text(encoding="utf-8"))
    destino = home / "fase0_openml_datos" / "arff"
    destino.mkdir(parents=True, exist_ok=True)
    for data_id, m in manifiesto.items():
        arff = destino / f"{data_id}.arff"
        if arff.exists() and _sha256(arff) == m["sha256_del_arff"]:
            continue
        with gzip.open(AQUI / "datos" / f"{data_id}.arff.gz", "rb") as g, arff.open("wb") as f:
            shutil.copyfileobj(g, f)
        if (calculado := _sha256(arff)) != m["sha256_del_arff"]:
            raise SystemExit(f"{m['dataset']}: sha256 {calculado} en vez de {m['sha256_del_arff']}: el kit está dañado")
    return manifiesto


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


class Tee:
    """stdout a pantalla Y al log (append, con flush), para que el log lo escriba el propio guion."""

    def __init__(self, ruta: Path, original):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self._f = ruta.open("a", encoding="utf-8")
        self._o = original

    def write(self, s: str) -> int:
        self._o.write(s)
        self._f.write(s)
        self._f.flush()
        return len(s)

    def flush(self) -> None:
        self._o.flush()
        self._f.flush()


def datos_de_la_maquina() -> dict:
    datos = {"python": platform.python_version(), "plataforma": platform.platform(), "cpus": os.cpu_count()}
    try:
        for linea in Path("/proc/meminfo").read_text().splitlines():
            if linea.startswith("MemTotal"):
                datos["ram_mb"] = round(int(linea.split()[1]) / 1024)
    except OSError:
        pass
    try:
        import torch  # noqa: PLC0415
        datos.update({"torch": torch.__version__, "cuda_de_torch": torch.version.cuda,
                      "hay_gpu": bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            datos.update({"gpu": p.name, "vram_total_mb": round(p.total_memory / 2 ** 20)})
    except Exception as error:  # noqa: BLE001
        datos["torch_error"] = f"{type(error).__name__}: {error}"
    try:
        from importlib.metadata import version  # noqa: PLC0415
        datos["tabicl"] = version("tabicl")
    except Exception:  # noqa: BLE001
        datos["tabicl"] = None
    return datos


def _buscar(obj, clave: str):
    """Primer valor de `clave` en un dict anidado (para anotar el `device` con el que el motor DECLARA haber corrido)."""
    if isinstance(obj, dict):
        if clave in obj:
            return obj[clave]
        for v in obj.values():
            if (r := _buscar(v, clave)) is not None:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if (r := _buscar(v, clave)) is not None:
                return r
    return None


# ---------------------------------------------------------------------------
# la medición
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", default=str(SALIDA_POR_OMISION))
    ap.add_argument("--solo", default=None, help="nombres separados por coma (solo para probar el guion)")
    ap.add_argument("--max-pliegues", type=int, default=None, help="solo los N primeros pliegues de cada conjunto (prueba)")
    ap.add_argument("--prueba", action="store_true", help="1 conjunto pequeño (dresses-sales), 1 pliegue")
    ap.add_argument("--estimar", action="store_true", help="imprime la estimación y NO mide nada")
    ap.add_argument("--forzar", action="store_true", help="ignora lo ya medido")
    ap.add_argument("--permitir-cpu", action="store_true", help="sonda del servidor: sin GPU (techo de CPU, no es la medición)")
    ap.add_argument("--pesos", default=PESOS_POR_OMISION)
    a = ap.parse_args(argv)
    if a.prueba:
        a.solo, a.max_pliegues = a.solo or "dresses-sales", a.max_pliegues or 1

    seleccion = json.loads((AQUI / "conjuntos.json").read_text(encoding="utf-8"))
    fichas = list(seleccion["incluidos"])
    if a.solo:
        pedidos = [n.strip() for n in a.solo.split(",") if n.strip()]
        desconocidos = [n for n in pedidos if n not in {f["dataset"] for f in fichas}]
        if desconocidos:
            raise SystemExit(f"--solo nombra conjuntos que no están en el kit: {desconocidos}")
        fichas = [f for f in fichas if f["dataset"] in pedidos]
    if a.estimar:
        _imprimir_estimacion(fichas)
        return

    home = Path(tempfile.gettempdir()) / "kit_116c5_home"
    preparar_rutas(home)
    desempaquetar_datos(home)
    os.environ["MATRIXAI_TABICL_PESOS_DIR"] = a.pesos
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    ruta_salida = Path(a.salida)
    sys.stdout = Tee(ruta_salida.with_name("log_acierto_gpu.txt"), sys.stdout)

    maquina = datos_de_la_maquina()
    print(f"=== kit 116-C5 · {time.strftime('%Y-%m-%d %H:%M:%S')} · {json.dumps(maquina, ensure_ascii=False)}", flush=True)
    if not maquina.get("hay_gpu") and not a.permitir_cpu:
        raise SystemExit("torch no ve ninguna GPU y esto en CPU serían días: ¿falta `--gpus all` en la orden de docker? "
                         "(para la sonda de un servidor sin GPU: --permitir-cpu)")
    es_sonda_cpu = not maquina.get("hay_gpu")
    if es_sonda_cpu:
        print("AVISO: SIN GPU. Esto es la sonda del servidor (techo de CPU de TabICL), NO la medición.", flush=True)

    import pasada_116c2_tabicl as c2  # noqa: PLC0415
    import pasada_exploratoria_101_c3 as c3  # noqa: PLC0415
    from matrixai_engines.particiones import Particion, Presupuesto  # noqa: PLC0415
    from matrixai_engines.registro_de_motores import motor_para  # noqa: PLC0415
    from matrixai_engines.subproceso import ejecutar_intento_aislado  # noqa: PLC0415
    from matrixai.estudio.validacion import digest_canonico  # noqa: PLC0415
    c5 = c2.c5

    c2.preparar_protocolo_v2()
    protocolo = c3.protocolo_registrado()
    if (protocolo.version_protocolo, protocolo.digest()[:8]) != (
            seleccion["protocolo_version"], seleccion["protocolo_digest_sha256"][:8]):
        raise SystemExit(f"el protocolo del kit ({protocolo.version_protocolo}, {protocolo.digest()[:8]}) no es el de "
                         f"conjuntos.json ({seleccion['protocolo_version']}, {seleccion['protocolo_digest_sha256'][:8]})")
    por_nombre = {d.nombre: d for d in c5.datasets_de_la_pasada(protocolo)}
    datasets = [por_nombre[f["dataset"]] for f in fichas]
    if any(d.sellado for d in datasets):
        raise SystemExit("un conjunto SELLADO en la lista: los sellados no se miden aquí")
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c2.direccion_de(metric_id)
    if len(os.sched_getaffinity(0)) < c3.HILOS_POR_INTENTO:
        print(f"AVISO: el contenedor ve {len(os.sched_getaffinity(0))} CPUs y el protocolo pide "
              f"{c3.HILOS_POR_INTENTO} hilos por intento", flush=True)

    # ---- régimen EXACTO con el techo de la GPU (el kit lo calculó al empaquetar; aquí se vuelve a comprobar) ----
    techo = seleccion["techo_gpu"]
    bases, regimenes = {}, {}
    for ds, ficha in zip(datasets, fichas):
        folds, n_cols, n_test, base = c2.folds_del_dataset(ds, protocolo)
        max_ctx = max(f[2] for f in folds)
        regimenes[ds.nombre] = {"n_pliegues": len(folds), "n_columnas": n_cols, "max_filas_de_contexto": max_ctx,
                                "max_celdas": max_ctx * n_cols, "n_test": n_test}
        bases[ds.nombre] = base
        if not es_sonda_cpu and (max_ctx > techo["max_filas_de_contexto"] or max_ctx * n_cols > techo["max_celdas"]):
            raise SystemExit(f"{ds.nombre}: {max_ctx} filas × {n_cols} columnas pasa del techo del kit {techo}")
        if (max_ctx, n_cols, len(folds)) != (ficha["max_filas_de_contexto"], ficha["n_columnas"], ficha["n_pliegues"]):
            raise SystemExit(f"{ds.nombre}: la partición de aquí ({max_ctx}, {n_cols}, {len(folds)}) no es la del "
                             f"kit al empaquetar ({ficha['max_filas_de_contexto']}, {ficha['n_columnas']}, {ficha['n_pliegues']})")
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]}; {len(datasets)} conjunto(s)", flush=True)

    motor = motor_para(c2.NOMBRE_TABICL)
    regla_sha = _sha256(AQUI / "REGLA.md") if (AQUI / "REGLA.md").exists() else None
    entorno_digest = hashlib.sha256((c2._digest_entorno() + _sha256(Path(__file__).resolve())).encode()).hexdigest()[:16]
    motor_digest = c3._digest_fichero(c2.RUTA_DEL_MOTOR_TABICL)
    try:
        procedencia = c3.procedencia_de_la_medicion(
            digests_de_codigo={"entorno": entorno_digest, "motor": motor_digest},
            datos_de_entrada={d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets})
    except Exception as error:  # noqa: BLE001 — dentro de la imagen no hay git: se declara, no se inventa
        procedencia = {"procedencia_id": "sin_procedencia", "anclable": False,
                       "avisos": [f"procedencia no disponible aquí: {type(error).__name__}: {error}"]}
    cache_previo, payload_previo = ({}, {}) if a.forzar else c3._cargar_cache(ruta_salida)
    if payload_previo and payload_previo.get("es_subconjunto_de_prueba") and not (a.solo or a.prueba):
        print("AVISO: el fichero de salida previo era de una PRUEBA; se ignora y se mide entero", flush=True)
        cache_previo = {}

    resultados: list[dict] = []
    pliegues_digest: dict[str, str] = {}
    plan_digest: dict[str, str] = {}
    clasificacion: dict[str, dict] = {}
    inicio = time.perf_counter()
    reusados = 0

    def guardar(parcial: bool) -> None:
        salida = {
            "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "corte": "116-C5",
            "que_es": ("acierto de tabicl.v2 en GPU, mismas particiones/semillas/presupuesto/métrica que 116-C2 (CPU); "
                       "se compara en el servidor con analiza_kit_acierto.py según REGLA.md"),
            "candidato": c2.NOMBRE_TABICL, "regla_sha256": regla_sha,
            "protocolo_version": protocolo.version_protocolo, "protocolo_digest_sha256": protocolo.digest(),
            "es_sonda_cpu": es_sonda_cpu, "es_subconjunto_de_prueba": bool(a.solo or a.prueba or a.max_pliegues),
            "maquina": maquina, "techo_gpu": techo, "parcial": parcial,
            "conjuntos_pedidos": [f["dataset"] for f in fichas],
            "regimen_por_dataset": regimenes, "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
            "pliegues_digest_por_dataset": dict(pliegues_digest), "plan_digest_por_dataset": dict(plan_digest),
            "estado_por_conjunto": clasificacion,
            "procedencia": procedencia, "entorno_digest": entorno_digest, "motor_digest": motor_digest,
            "total_wall_s": round(time.perf_counter() - inicio, 1), "n_intentos": len(resultados), "n_reusados": reusados,
            "resultados": resultados,
        }
        temporal = ruta_salida.with_suffix(ruta_salida.suffix + ".parcial")
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        temporal.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
        temporal.replace(ruta_salida)

    plan = {f["dataset"]: estimar_conjunto(f) for f in fichas}
    for n, (ds, ficha) in enumerate(zip(datasets, fichas), 1):
        por_id, propuesta, spec, objetivo, predictores, _declarada = bases[ds.nombre]
        pliegues_digest[ds.nombre] = propuesta.pliegues.digest()
        plan_digest[ds.nombre] = propuesta.plan.digest()
        test_ids = propuesta.plan.observaciones_del_rol("test")
        wall_seconds = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo)
        metric_id = metrica_por_dataset[ds.nombre]
        tope_conjunto = plan[ds.nombre]["tope_del_conjunto_s"]
        print(f"\n=== [{n}/{len(datasets)}] {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"columnas={regimenes[ds.nombre]['n_columnas']}, filas_ctx={regimenes[ds.nombre]['max_filas_de_contexto']}, "
              f"test={len(test_ids)}, tope por intento={wall_seconds:.0f}s, métrica={metric_id}) · estimado "
              f"{plan[ds.nombre]['s_del_conjunto_estimados'] / 60:.1f} min, tope del conjunto {tope_conjunto / 60:.0f} min ===",
              flush=True)
        t_conjunto = time.perf_counter()
        hechos = fallidos = omitidos = 0
        hechos_en_total = 0
        for repeticion in range(repeticiones):
            for pliegue_i in range(protocolo.particion.folds):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
                if pliegue is None:
                    continue
                if a.max_pliegues is not None and hechos_en_total >= a.max_pliegues:
                    continue
                hechos_en_total += 1
                clave = (ds.nombre, c2.NOMBRE_TABICL, repeticion, pliegue_i)
                previo = cache_previo.get(clave)
                if previo and previo.get("estado") == "completed" and c3._reusable(
                        previo, entorno_digest, motor_digest, wall_seconds):
                    resultados.append(dict(previo, reusado=True))
                    reusados += 1
                    hechos += 1
                    print(f"  [reusado] rep={repeticion} pliegue={pliegue_i} {metric_id}={previo.get(metric_id)}", flush=True)
                    continue
                if time.perf_counter() - t_conjunto > tope_conjunto:
                    omitidos += 1
                    resultados.append({
                        "dataset": ds.nombre, "motor": c2.NOMBRE_TABICL, "repeticion": repeticion, "pliegue": pliegue_i,
                        "estado": "no_lanzado_tope_del_conjunto", "metrica_de_cierre": metric_id, "reusado": False,
                        "motivo": f"el conjunto pasó de su tope ({tope_conjunto:.0f} s) en esta ejecución"})
                    continue
                crudas_train = [por_id[i] for i in pliegue.entrena]
                crudas_val = [por_id[i] for i in pliegue.valida]
                crudas_test = [por_id[i] for i in test_ids]
                transformadas = c3.preparar_para_motor(
                    crudas_train, crudas_train + crudas_val + crudas_test, objetivo, predictores, motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)
                hacer = lambda xs: Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)  # noqa: E731
                presupuesto = Presupuesto(wall_seconds=wall_seconds, hilos=c3.HILOS_POR_INTENTO,
                                          seed=protocolo.particion.semillas[repeticion])
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]),
                    hacer(transformadas[n_tr + n_va:]), spec, presupuesto,
                    candidate=f"{motor.nombre}-{c3.CONFIGURACION_UNICA}", split_plan_digest=propuesta.plan.digest(),
                    dataset=ds.nombre, pliegue=pliegue_i, repeticion=repeticion)
                transcurrido = time.perf_counter() - t0
                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea, "sellado": ds.sellado,
                    "motor": motor.nombre, "repeticion": repeticion, "pliegue": pliegue_i, "estado": intento.estado,
                    "semilla": protocolo.particion.semillas[repeticion], "presupuesto_wall_s": wall_seconds,
                    "configuracion": c3.CONFIGURACION_UNICA, "metrica_de_cierre": metric_id,
                    "wall_s": round(transcurrido, 3), "metricas": metricas,
                    "tiempo_de_ajuste": recursos.get("wall_seconds"), "cpu_segundos": recursos.get("cpu_seconds"),
                    "n_filas_de_contexto": n_tr, "n_filas_de_test": len(crudas_test),
                    "indices_digest": digest_canonico({"entrena": list(pliegue.entrena), "valida": list(pliegue.valida),
                                                       "test": list(test_ids)}),
                    "dispositivo_declarado_por_el_motor": _buscar(intento.config_efectiva, "device"),
                    "motivo": intento.motivo_del_estado["es"] if intento.motivo_del_estado else None,
                    "traza": intento.traza,
                    "entorno_digest": entorno_digest, "motor_digest": motor_digest,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False,
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                resultados.append(registro)
                if intento.estado == "completed":
                    hechos += 1
                else:
                    fallidos += 1
                print(f"  rep={repeticion} pliegue={pliegue_i}: estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"wall={transcurrido:.1f}s dispositivo={registro['dispositivo_declarado_por_el_motor']}"
                      f"{'  MOTIVO: ' + str(registro['motivo']) if registro['motivo'] else ''}", flush=True)
                guardar(parcial=True)
        clasificacion[ds.nombre] = {"completados": hechos, "fallidos": fallidos, "no_lanzados_por_tope": omitidos,
                                    "wall_s": round(time.perf_counter() - t_conjunto, 1),
                                    "estimado_s": plan[ds.nombre]["s_del_conjunto_estimados"]}
        print(f"  RESUMEN {ds.nombre}: {hechos} completados, {fallidos} fallidos, {omitidos} no lanzados por el tope, "
              f"{time.perf_counter() - t_conjunto:.0f} s (estimado {plan[ds.nombre]['s_del_conjunto_estimados']:.0f} s)",
              flush=True)
        guardar(parcial=True)

    guardar(parcial=False)
    total = time.perf_counter() - inicio
    print(f"\n=== TERMINADO: {len(resultados)} intentos ({reusados} reusados) en {total / 60:.1f} min. "
          f"Guardado en {ruta_salida}. Mándame ese fichero (y log_acierto_gpu.txt). ===", flush=True)


def _imprimir_estimacion(fichas: list[dict]) -> None:
    filas = [estimar_conjunto(f) for f in fichas]
    print(f"{'conjunto':34} {'pliegues':>8} {'celdas':>8} {'s/pliegue':>10} {'conjunto (min)':>15} {'tope (min)':>11}")
    for e in filas:
        print(f"{e['dataset']:34} {e['n_pliegues']:8} {e['celdas']:8} {e['s_por_pliegue_estimados']:10.1f} "
              f"{e['s_del_conjunto_estimados'] / 60:15.1f} {e['tope_del_conjunto_s'] / 60:11.0f}")
    total = sum(e["s_del_conjunto_estimados"] for e in filas)
    print(f"\nTOTAL ESTIMADO (no medido): {total / 3600:.2f} h; peor caso con todos los topes de conjunto: "
          f"{sum(e['tope_del_conjunto_s'] for e in filas) / 3600:.2f} h")
    print(f"Supuestos: {ARRANQUE_DE_UN_INTENTO_S:.0f} s de arranque por pliegue (supuesto) + {LLAMADAS_POR_PLIEGUE} × el tiempo "
          f"de un intento de la tabla del contrato 116 por tramo de celdas {TABLA_DE_TIEMPOS_S}.")


if __name__ == "__main__":
    main()
