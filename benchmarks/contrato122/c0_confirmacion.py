#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""122-C0 — la medida de CONFIRMACIÓN: TabM frente a la red del núcleo, en las
condiciones del paquete CPU del Studio (torch de CPU, 2 hilos, 6 GB).

NO es otra Fase 0. Es una medida pequeña (8 conjuntos NO sellados de la Fase 0,
las MISMAS particiones que ella) con una regla escrita ANTES (el protocolo
`protocolo_122_c0.json`, estado REGISTRADO y su digest) para contestar tres
cosas: (1) ¿la ventaja de TabM (38/40 en Fase 0) se sostiene a 2 hilos?
(2) ¿cuánto espera la persona por modelo? (3) ¿cuánta memoria pide cada red?

LAS TRES REDES (`REDES`), sobre la MISMA partición entrenamiento/validación/test
(la de la Fase 0, repetición 0; el test no entra NUNCA a `fit()`, y
`ejecutar_intento` del arnés lo comprueba — `_comprobar_sin_contaminacion`):

* `nucleo_modo_experto` — la red que entrena HOY el modo experto del Studio.
  Es el puente `MotorDensaPropia` (`generate_project_from_dataset` +
  `run_playground_training`, la superficie pública del núcleo) con TRES
  diferencias respecto al motor de la Fase 0, las tres medidas y no supuestas
  (`MotorDensaModoExperto`): (a) la RECETA del generador tal cual — SGD a
  0,01, lote 8, 50 épocas, SIN parada temprana (`dense_generator`: es lo que
  el modo experto manda a `/api/train-start`), y no la del motor (Adam +
  `EARLY_STOP`, 113-C1); (b) SIN plazo (`budget.wall_seconds=None`: el modo
  experto no lo pone); (c) `recortar_objetivo=False`, que es lo que pasa
  `_studio_train_start` (A8) y que el motor no pasa. Lo que NO se puede
  reproducir es el `SPLIT` del generador (0,8/0,2 barajado, sin test): la
  partición la decide el arnés, igual que en la Fase 0, para que las tres
  redes vean las mismas filas y el test quede fuera.
* `nucleo_estudio` — el motor `matrixai.dense.torch_cpu` TAL CUAL (Adam +
  parada temprana): «la densa de hoy» de la Fase 0 (16/40), con la única
  diferencia de que aquí va SIN plazo. Referencia, no la red del modo experto.
* `tabm` — `matrixai.dense.tabm_cpu` (TabM 1.2.0) con el presupuesto de pared
  de su cubo en la Fase 0 (`presupuesto.tabm_wall_s_por_cubo`); el plazo es el
  0,75 de ese presupuesto y el resultado DECLARA si lo cortó.

CADA AJUSTE CORRE EN UN PROCESO APARTE (`multiprocessing.spawn`) y ese proceso
mide su PICO DE MEMORIA: `resource.getrusage(RUSAGE_SELF).ru_maxrss` al
terminar (la marca máxima de SU RSS, incluidas las importaciones; antes del
ajuste se guarda también la de entonces, para ver el delta). Los motores no lo
miden (`recursos.peak_ram_mb` sale `None`), así que no hay otra fuente.

QUÉ NÚCLEO SE MIDE. El del ÁRBOL de este guion (`RAIZ`), no el instalado: los
módulos de `fase0/` ya anteponen su raíz a `sys.path`, y aquí se hace
EXPLÍCITO y se exige (`exigir_el_nucleo_del_arbol`): si `matrixai` viene de
otro sitio, el guion para. Los motores salen de lo que haya en el entorno (en
el paquete CPU: la imagen) y su ruta y versión van en la procedencia. Esta
máquina miente sobre lo que hay instalado (CLAUDE.md, «editables»): no se
confía en `pip`, se declara `matrixai.__file__`.

RESULTADO: un JSON con procedencia (commits de los TRES repos y si están
limpios, versiones, máquina, carga al empezar, condiciones), el veredicto de la
regla (`aplicar_regla_de_cierre` de la Fase 0, la MISMA aritmética) y un SELLO
(`digest_canonico` del documento entero sin el sello). La caché es por intento
(`reusable`): digest de código (cierre estático de imports, como 119-C3), de
los datos, de la configuración y de la partición; un intento FALLIDO no se
reaprovecha nunca.

    python3 benchmarks/contrato122/c0_confirmacion.py --humo --salida <ruta>
    python3 benchmarks/contrato122/c0_confirmacion.py --solo dresses-sales --salida <ruta>
    python3 benchmarks/contrato122/c0_confirmacion.py --salida <ruta absoluta>   # la pasada: a la COLA
    python3 benchmarks/contrato122/c0_confirmacion.py --sellar-protocolo        # tras fijar la regla
    python3 benchmarks/contrato122/c0_confirmacion.py --escribir-git-del-host F # para correr SIN git
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import platform
import queue
import resource
import shutil
import statistics
import sys
import time
import unittest.mock
from pathlib import Path

# --- Hilos: se fijan ANTES de importar torch/numpy (las condiciones del paquete CPU) ---------
HILOS = 2
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = str(HILOS)

# --- El núcleo del ÁRBOL va primero, a propósito y a la vista ---------------------------------
RAIZ = Path(__file__).resolve().parents[2]
_AQUI = Path(__file__).resolve().parent
_FASE0 = RAIZ / "benchmarks" / "fase0"
for _ruta in (_FASE0, RAIZ):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_119_c3 as p119  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

import matrixai  # noqa: E402
from matrixai.estudio.validacion import digest_canonico  # noqa: E402

import matrixai_engines  # noqa: E402
from matrixai_engines.harness import ErrorDeHarness, ejecutar_intento  # noqa: E402
from matrixai_engines.motores import densa as densa_mod  # noqa: E402
from matrixai_engines.motores.densa import MotorDensaPropia  # noqa: E402
from matrixai_engines.motores.densa_tabm import MotorDensaTabM  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.procedencia import versiones_de_bibliotecas  # noqa: E402

RUTA_DEL_PROTOCOLO = _AQUI / "protocolo_122_c0.json"
ESTADO_REGISTRADO = "REGISTRADO"

NUCLEO_MODO_EXPERTO = "nucleo_modo_experto"
NUCLEO_ESTUDIO = "nucleo_estudio"
TABM = "tabm"
#: Orden de ejecución dentro de un pliegue: la barata primero.
REDES = (NUCLEO_ESTUDIO, NUCLEO_MODO_EXPERTO, TABM)

#: El nombre del motor de cada red (lo que declara `Motor.nombre`).
MOTOR_DE_LA_RED = {NUCLEO_MODO_EXPERTO: "matrixai.dense.torch_cpu",
                   NUCLEO_ESTUDIO: "matrixai.dense.torch_cpu",
                   TABM: "matrixai.dense.tabm_cpu"}

MARGEN_DEL_TOPE_SOBRE_EL_PRESUPUESTO_S = 30.0  # el de `subproceso.MARGEN_POR_DEFECTO_SEGUNDOS`
SEGUNDOS_ENTRE_SONDEOS = 0.5


# ===============================================================================================
# 1. LAS REDES
# ===============================================================================================

def receta_del_generador(training_text: str) -> str:
    """El `training_text` TAL CUAL lo escribió el generador del núcleo: sin la
    receta Adam + `EARLY_STOP` que `densa.aplicar_receta` le pone en el estudio.
    Es la sustitución que hace `MotorDensaModoExperto`; vive aparte para que la
    prueba pueda llamarla y ver que NO toca nada."""
    return training_text


def _entrenar_como_el_modo_experto(original):
    """Un envoltorio de `run_playground_training` que añade lo que
    `_studio_train_start` pasa y el motor no: `recortar_objetivo=False` (A8).
    Lleva `**kwargs` a propósito: el motor acepta un envoltorio con
    `VAR_KEYWORD` como transmisor del `plazo`."""
    def entrenar(*args, **kwargs):
        return original(*args, recortar_objetivo=False, **kwargs)
    return entrenar


class MotorDensaModoExperto(MotorDensaPropia):
    """La red del núcleo por el camino del MODO EXPERTO: el mismo puente
    (`_ajustar` del padre, entero: partición, CSV, `generate_project_from_dataset`,
    `run_playground_training`, predicción) con la receta del generador, sin
    plazo y sin recorte del objetivo. No reimplementa nada: sustituye, durante
    el ajuste y solo en este proceso, las DOS cosas que el modo experto hace
    distinto, y lo que se declara (`hiperparametros`) sale del texto que de
    verdad se entrenó, así que si la sustitución dejara de aplicarse el
    predictor diría Adam y no SGD."""

    def _ajustar(self, train, validation, spec, budget, *, semilla):
        import matrixai.playground_api as api  # noqa: PLC0415
        if budget.wall_seconds is not None:
            # El modo experto no pone plazo: pedírselo aquí mediría otra cosa.
            raise ValueError(f"la red del modo experto va SIN plazo (wall_seconds="
                             f"{budget.wall_seconds!r}): pedírselo mediría otra cosa")
        with unittest.mock.patch.object(densa_mod, "aplicar_receta", receta_del_generador), \
                unittest.mock.patch.object(
                    api, "run_playground_training",
                    _entrenar_como_el_modo_experto(api.run_playground_training)):
            return super()._ajustar(train, validation, spec, budget, semilla=semilla)


def motor_de_la_red(red: str):
    if red == NUCLEO_MODO_EXPERTO:
        return MotorDensaModoExperto()
    if red == NUCLEO_ESTUDIO:
        return MotorDensaPropia()
    if red == TABM:
        return MotorDensaTabM()
    raise SystemExit(f"red desconocida: {red!r} (conocidas: {REDES})")


def presupuesto_de_la_red(red: str, *, cubo: str, proto: dict, semilla: int,
                          tabm_wall_s: float | None = None) -> tuple[float | None, float]:
    """`(wall_seconds del Presupuesto, tope duro del padre)` de una red.

    Las dos del núcleo van SIN plazo (`None`); el tope duro del padre es solo la
    red de seguridad contra un cuelgue y se declara en el registro. TabM lleva
    el presupuesto de pared de su cubo en la Fase 0 (o el del humo)."""
    p = proto["presupuesto"]
    if red == TABM:
        wall = float(tabm_wall_s if tabm_wall_s is not None
                     else p["tabm_wall_s_por_cubo"][cubo])
        return wall, wall + MARGEN_DEL_TOPE_SOBRE_EL_PRESUPUESTO_S
    return None, float(p["tope_duro_nucleo_s_por_cubo"][cubo])


# ===============================================================================================
# 2. EL INTENTO EN UN PROCESO APARTE (con su pico de memoria)
# ===============================================================================================

def _rss_pico_mb() -> float:
    """El RSS máximo de ESTE proceso hasta ahora (KiB en Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _config_pequena(config: dict | None) -> dict:
    """Lo que el motor DECLARA que pasó, sin los pesos ni el `.mxai` (decenas de MB)."""
    config = config or {}
    pequena = {k: config.get(k) for k in ("entrenamiento_efectivo", "hiperparametros",
                                          "arquitectura", "epochs", "seed_efectiva")
               if config.get(k) is not None}
    return json.loads(json.dumps(pequena, default=str))


def _hijo(cola, red, train, validation, test, spec, wall_seconds, hilos, seed, candidate,
          split_plan_digest, dataset, pliegue, repeticion) -> None:
    """El cuerpo del proceso que ajusta UNA red y mide su pico de memoria."""
    try:
        import torch  # noqa: PLC0415
        hilos_de_torch_al_entrar = torch.get_num_threads()
        rss_antes = _rss_pico_mb()
        motor = motor_de_la_red(red)
        presupuesto = Presupuesto(wall_seconds=wall_seconds, hilos=hilos, seed=seed)
        t0 = time.perf_counter()
        intento = ejecutar_intento(motor, train, validation, test, spec, presupuesto,
                                   candidate=candidate, split_plan_digest=split_plan_digest,
                                   dataset=dataset, pliegue=pliegue, repeticion=repeticion)
        transcurrido = time.perf_counter() - t0
        cola.put(("ok", {
            "estado": intento.estado, "motivo_del_estado": intento.motivo_del_estado,
            "informe": intento.informe, "recursos": intento.recursos,
            "pipeline_digest": intento.pipeline_digest, "engine_version": intento.engine_version,
            "config": _config_pequena(intento.config_efectiva),
            "wall_del_intento_s": transcurrido,
            "rss_antes_del_ajuste_mb": rss_antes, "rss_pico_mb": _rss_pico_mb(),
            "hilos_de_torch_al_entrar": hilos_de_torch_al_entrar,
            "hilos_pedidos": hilos, "pid": os.getpid()}))
    except ErrorDeHarness as fuga:
        cola.put(("fuga", fuga.clave, dict(fuga.campos)))
    except BaseException as excepcion:  # noqa: BLE001 — aislar CUALQUIER fallo
        import traceback  # noqa: PLC0415
        cola.put(("excepcion", f"{type(excepcion).__name__}: {excepcion}",
                  "".join(traceback.format_exception(excepcion))[-4000:]))


def ejecutar_aislado(red: str, train, validation, test, spec, *, wall_seconds: float | None,
                     tope_duro_s: float, hilos: int, seed: int, candidate: str,
                     split_plan_digest: str, dataset: str, pliegue: int,
                     repeticion: int) -> dict:
    """Corre `_hijo` en un proceso `spawn` y devuelve su mensaje como dict:
    `{"clase": "ok"|"fuga"|"excepcion"|"tope"|"muerto", ...}`. Si el proceso
    no termina en `tope_duro_s` se mata desde fuera (`clase: "tope"`). Lee la
    cola ANTES de esperar al proceso (el interbloqueo de `subproceso.py`)."""
    contexto = multiprocessing.get_context("spawn")
    cola = contexto.Queue()
    proceso = contexto.Process(target=_hijo, args=(
        cola, red, train, validation, test, spec, wall_seconds, hilos, seed, candidate,
        split_plan_digest, dataset, pliegue, repeticion))
    inicio = time.monotonic()
    proceso.start()
    mensaje = None
    while time.monotonic() - inicio < tope_duro_s:
        try:
            mensaje = cola.get(timeout=SEGUNDOS_ENTRE_SONDEOS)
            break
        except queue.Empty:
            if not proceso.is_alive():
                try:  # lo que dejó justo antes de morir
                    mensaje = cola.get(timeout=2.0)
                except queue.Empty:
                    pass
                break
    transcurrido = time.monotonic() - inicio
    if mensaje is None:
        clase = "tope" if proceso.is_alive() else "muerto"
        if proceso.is_alive():
            proceso.terminate()
            proceso.join(5.0)
            if proceso.is_alive():
                proceso.kill()
        proceso.join(5.0)
        return {"clase": clase, "wall_s": transcurrido, "exitcode": proceso.exitcode}
    proceso.join(30.0)
    if proceso.is_alive():
        proceso.kill()
        proceso.join(5.0)
    if mensaje[0] == "ok":
        return {"clase": "ok", "wall_s": transcurrido, **mensaje[1]}
    if mensaje[0] == "fuga":
        return {"clase": "fuga", "wall_s": transcurrido, "clave": mensaje[1], "campos": mensaje[2]}
    return {"clase": "excepcion", "wall_s": transcurrido, "error": mensaje[1], "traza": mensaje[2]}


# ===============================================================================================
# 3. LOS DATOS Y LA PARTICIÓN (la de la Fase 0)
# ===============================================================================================

def digest_de_ids(ids) -> str:
    """Qué FILAS son (el conjunto, no el orden): lo que dice que dos redes vieron
    la misma partición."""
    return digest_canonico(sorted(str(i) for i in ids))[:16]


def ids_de_la_particion(entrena, valida, test_ids) -> dict:
    return {"train": digest_de_ids(entrena), "validation": digest_de_ids(valida),
            "test": digest_de_ids(test_ids),
            "n_train": len(entrena), "n_validation": len(valida), "n_test": len(test_ids)}


def particiones_para(red: str, conjunto, entrena, valida, test_ids):
    """`(train, validation, test)` como `Particion`, preparados con la política
    de ESTA red (la misma función que la Fase 0: `ajustar_preparacion` SOLO con
    train, `transformar_fila` sobre las tres). Mismas filas para todas las redes;
    cambia solo cómo se preparan según lo que admita el motor."""
    por_id, objetivo, predictores = conjunto["por_id"], conjunto["objetivo"], conjunto["predictores"]
    crudas_train = [por_id[i] for i in entrena]
    crudas_val = [por_id[i] for i in valida]
    crudas_test = [por_id[i] for i in test_ids]
    motor = motor_de_la_red(red)
    transformadas = c3.preparar_para_motor(
        crudas_train, crudas_train + crudas_val + crudas_test, objetivo, predictores, motor)
    n_tr, n_va = len(crudas_train), len(crudas_val)

    def hacer(xs):
        return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

    return (hacer(transformadas[:n_tr]), hacer(transformadas[n_tr:n_tr + n_va]),
            hacer(transformadas[n_tr + n_va:]))


def cargar_conjunto(ds, protocolo_v2) -> dict:
    por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(
        ds, protocolo_v2)
    return {"por_id": por_id, "propuesta": propuesta, "spec": spec, "objetivo": objetivo,
            "predictores": predictores, "particion_declarada": particion,
            "test_ids": tuple(propuesta.plan.observaciones_del_rol("test"))}


def pliegue_del_conjunto(conjunto, repeticion: int, pliegue: int):
    p = conjunto["propuesta"].pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue)
    if p is None:
        raise SystemExit(f"la partición no trae el pliegue {pliegue} de la repetición "
                         f"{repeticion}: no se puede medir")
    return tuple(p.entrena), tuple(p.valida)


# ===============================================================================================
# 4. EL PROTOCOLO (borrador → registrado) Y SU DIGEST
# ===============================================================================================

def autodigest(protocolo: dict) -> str:
    """El sha256 del protocolo canónico SIN su propio `digest_sha256` (el estilo de
    la Fase 0: `generar_protocolo_118_v3.py`)."""
    sin = {k: v for k, v in protocolo.items() if k != "digest_sha256"}
    return hashlib.sha256(json.dumps(sin, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def cargar_protocolo(ruta: Path = RUTA_DEL_PROTOCOLO) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def sellar_protocolo(ruta: Path = RUTA_DEL_PROTOCOLO) -> str:
    """Recalcula `digest_sha256` (lo que hace el controlador al registrarlo)."""
    proto = cargar_protocolo(ruta)
    proto["digest_sha256"] = autodigest(proto)
    Path(ruta).write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return proto["digest_sha256"]


def exigir_el_protocolo(proto: dict, protocolo_v2, *, tipo: str) -> list:
    """Para ANTES de medir si el protocolo no es el que dice ser. Una pasada REAL
    exige `estado == REGISTRADO` y su digest; el humo y `--solo` admiten el
    borrador (y se dice). Cada conjunto tiene que estar en el v2, con su
    `data_id` y su tarea, y NO ser sellado. Devuelve los `DatasetDeLaPasada`."""
    if tipo == "pasada":
        if proto.get("estado") != ESTADO_REGISTRADO:
            raise SystemExit(f"el protocolo está en estado {proto.get('estado')!r}: una pasada REAL "
                             f"exige {ESTADO_REGISTRADO!r} (la regla se fija ANTES de medir)")
        if proto.get("digest_sha256") != autodigest(proto):
            raise SystemExit("el digest del protocolo NO es el de su contenido: se editó después "
                             "de registrarlo. `--sellar-protocolo` lo recalcula, a la vista")
    por_nombre = {d.nombre: d for d in c5.datasets_de_la_pasada(protocolo_v2)}
    elegidos = []
    for entrada in proto["datasets"]:
        ds = por_nombre.get(entrada["nombre"])
        if ds is None:
            raise SystemExit(f"{entrada['nombre']!r} no está en el protocolo v2 de la Fase 0")
        if ds.sellado:
            raise SystemExit(f"{ds.nombre!r} es un conjunto SELLADO: C0 no los toca")
        if ds.data_id != entrada["data_id"] or ds.tarea != entrada["tarea"]:
            raise SystemExit(f"{ds.nombre!r}: data_id/tarea del protocolo de C0 no coinciden con "
                             f"el v2 ({entrada['data_id']}/{entrada['tarea']} contra "
                             f"{ds.data_id}/{ds.tarea})")
        elegidos.append(ds)
    return elegidos


def exigir_el_nucleo_del_arbol() -> str:
    """El núcleo que se mide es el de este árbol; cualquier otro, para."""
    fichero = Path(matrixai.__file__).resolve()
    if RAIZ not in fichero.parents:
        raise SystemExit(f"`matrixai` se importa de {fichero}, no del árbol de este guion "
                         f"({RAIZ}): se estaría midiendo OTRO núcleo del que la procedencia "
                         f"declara. Antepón {RAIZ} a sys.path")
    return str(fichero)


# ===============================================================================================
# 5. EL DIGEST DEL ENTORNO Y LA CACHÉ POR INTENTO
# ===============================================================================================

def _raices_de_import() -> tuple[Path, ...]:
    """Las raíces REALES de lo que se importa (donde está cada paquete, no donde
    «debería»), núcleo primero: la del hijo es la misma que la del padre."""
    nucleo = Path(matrixai.__file__).resolve().parent.parent
    motores = Path(matrixai_engines.__file__).resolve().parent.parent
    return (nucleo, motores, _FASE0, _AQUI)


def _etiqueta(ruta: Path) -> str:
    r = Path(ruta).resolve()
    nucleo, motores, fase0, aqui = _raices_de_import()
    for nombre, raiz in (("c0", aqui), ("fase0", fase0), ("engines", motores), ("core", nucleo)):
        try:
            return f"{nombre}:{r.relative_to(raiz.resolve()).as_posix()}"
        except ValueError:
            continue
    return f"fuera:{r.as_posix()}"


PAQUETES_PROPIOS = ("matrixai", "matrixai_engines")
#: El de 119-C3, guardado ANTES de sustituirlo (si no, la sustituta se llama a sí misma).
_FICHEROS_DE_UN_IMPORT_DE_119 = p119._ficheros_de_un_import


def _ficheros_de_lo_propio(nombre: str, raices: tuple[Path, ...]) -> list[Path]:
    """`p119._ficheros_de_un_import` pero SOLO para lo nuestro: los dos paquetes y
    los módulos sueltos de `fase0/` y de este directorio. Hace falta porque en el
    paquete CPU los motores viven en `site-packages`, y como raíz de imports
    `numpy` o `torch` también «existen» allí: el cierre se iba a miles de ficheros
    de terceros (83 s medidos en el contenedor, 3 s en el host). De los terceros
    entra su VERSIÓN en el digest (`componentes_del_digest_del_entorno`), no su código."""
    cabeza = nombre.split(".")[0]
    if cabeza in PAQUETES_PROPIOS:
        return _FICHEROS_DE_UN_IMPORT_DE_119(nombre, raices)
    for raiz in (_AQUI, _FASE0):
        if (raiz / f"{cabeza}.py").is_file():
            return _FICHEROS_DE_UN_IMPORT_DE_119(nombre, (raiz,))
    return []


def ficheros_del_entorno() -> tuple[Path, ...]:
    """El cierre ESTÁTICO de imports de este guion (todos los `import`, también los
    de dentro de funciones), como el de 119-C3: un superconjunto de lo que corre.
    Reutiliza SU recorrido (`_cierre_de_imports`, sin su caché: `__wrapped__`) con la
    resolución de nombres restringida a lo propio."""
    with unittest.mock.patch.object(p119, "_ficheros_de_un_import", _ficheros_de_lo_propio):
        cierre, _ = p119._cierre_de_imports.__wrapped__(Path(__file__).resolve(),
                                                        _raices_de_import())
    return tuple(sorted(cierre))


def componentes_del_digest_del_entorno(ruta_del_protocolo: Path = RUTA_DEL_PROTOCOLO) -> dict:
    componentes = {_etiqueta(f): c3._digest_fichero(f) for f in ficheros_del_entorno()}
    componentes["dato:protocolo_122_c0"] = c3._digest_fichero(Path(ruta_del_protocolo))
    componentes["dato:protocolo_exploratorio_v2"] = c3._digest_fichero(p119.RUTA_DEL_PROTOCOLO_V2)
    versiones = versiones_de_bibliotecas()
    for b in p119.BIBLIOTECAS_EN_EL_DIGEST:
        componentes[f"version:{b}"] = str(versiones.get(b, "ausente"))
    componentes["version:hilos"] = str(HILOS)
    return componentes


def digest_del_entorno(componentes: dict) -> str:
    return hashlib.sha256(json.dumps(componentes, sort_keys=True).encode()).hexdigest()[:16]


def digest_de_la_configuracion(*, red: str, wall_s, tope_duro_s, hilos, semilla, repeticion,
                               pliegue, particion: dict) -> str:
    return digest_canonico({"red": red, "wall_s": wall_s, "tope_duro_s": tope_duro_s,
                            "hilos": hilos, "semilla": semilla, "repeticion": repeticion,
                            "pliegue": pliegue, "particion": particion})[:16]


def reusable(previo: dict | None, *, entorno_digest: str, config_digest: str,
             datos_sha256: str | None) -> bool:
    """¿Se puede reaprovechar un registro? Solo si es una MEDIDA (un `failed` con
    los digests buenos NO — trampa nº 1 de relanzar pasadas), con el MISMO código,
    la MISMA configuración y partición y los MISMOS datos."""
    if not previo:
        return False
    if previo.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
        return False
    if datos_sha256 is None or previo.get("datos_sha256") != datos_sha256:
        return False
    return (previo.get("entorno_digest") == entorno_digest
            and previo.get("config_digest") == config_digest)


# ===============================================================================================
# 6. LA PROCEDENCIA
# ===============================================================================================

def _raices_de_los_repositorios() -> dict[str, Path]:
    return {"matrixAI": RAIZ, "matrixai-engines": RAIZ.parent / "matrixai-engines",
            "matrixaistudio": RAIZ.parent / "matrixaistudio"}


def estado_de_los_repositorios(*, git_del_host: dict | None = None,
                               raices: dict[str, Path] | None = None) -> dict:
    """Commit y suciedad de los TRES repos. Sin `git` (la imagen del paquete no lo
    trae) se usa el fichero que escribió el host (`--escribir-git-del-host`),
    diciendo de dónde salió; sin nada, `commit: None` CON su motivo, nunca un
    relleno."""
    raices = raices or _raices_de_los_repositorios()
    hay_git = shutil.which("git") is not None
    estado = {}
    for nombre, raiz in raices.items():
        if hay_git and Path(raiz).exists():
            estado[nombre] = dict(c3._estado_del_repositorio(Path(raiz)), fuente="git")
        elif git_del_host and nombre in git_del_host.get("repositorios", {}):
            estado[nombre] = dict(git_del_host["repositorios"][nombre], fuente="host")
        else:
            estado[nombre] = {"commit": None, "arbol_sucio": None, "fuente": None,
                              "motivo": "sin git en este entorno y sin fichero del host"}
    return estado


def escribir_git_del_host(ruta: Path) -> None:
    estado = {n: c3._estado_del_repositorio(r) for n, r in _raices_de_los_repositorios().items()}
    Path(ruta).write_text(json.dumps({"repositorios": estado,
                                      "medido": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                     indent=2) + "\n", encoding="utf-8")


def _leer(ruta: str) -> str | None:
    try:
        return Path(ruta).read_text().strip()
    except OSError:
        return None


def la_maquina() -> dict:
    return {"plataforma": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpus_con_afinidad": len(os.sched_getaffinity(0)),
            "cgroup_cpu_max": _leer("/sys/fs/cgroup/cpu.max"),
            "cgroup_memory_max": _leer("/sys/fs/cgroup/memory.max"),
            "memtotal_kb": next((int(l.split()[1]) for l in (_leer("/proc/meminfo") or "").splitlines()
                                 if l.startswith("MemTotal")), None)}


def componer_procedencia(*, datos_de_entrada: dict[str, Path], digests_de_codigo: dict,
                         git_del_host: dict | None = None) -> dict:
    import torch  # noqa: PLC0415
    repositorios = estado_de_los_repositorios(git_del_host=git_del_host)
    avisos = []
    for nombre, e in repositorios.items():
        if e.get("commit") is None:
            avisos.append(f"{nombre}: SIN COMMIT ({e.get('motivo')})")
        elif e.get("arbol_sucio"):
            avisos.append(f"{nombre}: ÁRBOL SUCIO ({len(e.get('ficheros_modificados') or [])} "
                          f"modificados, {len(e.get('ficheros_sin_seguimiento') or [])} sin "
                          f"seguimiento): el commit no identifica el código medido")
    datos = {}
    for nombre, ruta in sorted(datos_de_entrada.items()):
        ruta = Path(ruta)
        if ruta.exists():
            datos[nombre] = {"ruta": str(ruta), "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
                             "bytes": ruta.stat().st_size}
        else:
            datos[nombre] = {"ruta": str(ruta), "sha256": None, "motivo": "no existe"}
            avisos.append(f"datos «{nombre}»: {ruta} no existe, sin sha256")
    bloque = {
        "medido": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repositorios": repositorios,
        "versiones_de_bibliotecas": versiones_de_bibliotecas(),
        "codigo_que_se_importa": {
            "matrixai": str(Path(matrixai.__file__).resolve()),
            "matrixai_version": getattr(matrixai, "__version__", None),
            "matrixai_engines": str(Path(matrixai_engines.__file__).resolve()),
            "torch": torch.__version__, "torch_hilos_por_omision": torch.get_num_threads()},
        "maquina": la_maquina(),
        "carga_al_empezar": os.getloadavg(),
        "condiciones": {"hilos_pedidos": HILOS,
                        "variables_de_hilos": {v: os.environ.get(v) for v in (
                            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "digests_de_codigo": dict(digests_de_codigo),
        "datos_de_entrada": datos,
        "anclable": not avisos, "avisos": avisos,
    }
    bloque["procedencia_id"] = digest_canonico(bloque)
    return bloque


# ===============================================================================================
# 7. EL REGISTRO DE UN INTENTO, EL VEREDICTO Y EL SELLO
# ===============================================================================================

def registro_de_un_intento(*, ds, red: str, repeticion: int, pliegue: int, semilla: int,
                           wall_s_pedido, tope_duro_s, particion: dict, metric_id: str,
                           datos_sha256, entorno_digest: str, config_digest: str,
                           preparacion_s: float, salida: dict) -> dict:
    """El registro de un intento a partir de lo que devolvió `ejecutar_aislado`."""
    registro = {
        "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo, "tarea": ds.tarea,
        "motor": red, "motor_declarado": MOTOR_DE_LA_RED[red],
        "repeticion": repeticion, "pliegue": pliegue, "semilla": semilla,
        "presupuesto_wall_s": wall_s_pedido, "tope_duro_s": tope_duro_s, "hilos": HILOS,
        "metrica_de_cierre": metric_id, "particion": particion,
        "preparacion_s": round(preparacion_s, 3),
        "datos_sha256": datos_sha256, "entorno_digest": entorno_digest,
        "config_digest": config_digest, "reusado": False,
    }
    if salida["clase"] != "ok":
        motivo = {"tope": f"el proceso superó su tope duro de {tope_duro_s} s y se mató",
                  "muerto": f"el proceso murió sin dejar mensaje (exitcode {salida.get('exitcode')})",
                  "fuga": f"fuga train/test: {salida.get('clave')} {salida.get('campos')}",
                  "excepcion": salida.get("error")}[salida["clase"]]
        registro.update({"estado": "failed", "motivo": motivo, "traza": salida.get("traza"),
                         "wall_s": round(salida["wall_s"], 3)})
        return registro
    recursos = salida.get("recursos") or {}
    config = salida.get("config") or {}
    efectivo = config.get("entrenamiento_efectivo") or {}
    metricas = c5.metricas_del_informe(salida.get("informe"))
    motivo = salida.get("motivo_del_estado")
    registro.update({
        "estado": salida["estado"],
        "motivo": (motivo.get("es") if isinstance(motivo, dict) else motivo),
        "wall_s": round(salida["wall_s"], 3),
        "wall_del_intento_s": round(salida["wall_del_intento_s"], 3),
        "tiempo_de_ajuste": recursos.get("wall_seconds"),
        "cpu_segundos": recursos.get("cpu_seconds"),
        "rss_pico_mb": round(salida["rss_pico_mb"], 1),
        "rss_antes_del_ajuste_mb": round(salida["rss_antes_del_ajuste_mb"], 1),
        "hilos_de_torch_al_entrar": salida["hilos_de_torch_al_entrar"],
        "epocas_ejecutadas": efectivo.get("epocas_ejecutadas"),
        "mejor_epoca": efectivo.get("mejor_epoca"),
        "parado_por_plazo": bool(efectivo.get("parado_por_plazo")),
        "entrenamiento_efectivo": efectivo,
        "hiperparametros": config.get("hiperparametros"),
        "arquitectura": config.get("arquitectura"),
        "metricas": metricas,
        "engine_version": salida.get("engine_version"),
        "pipeline_digest": salida.get("pipeline_digest"),
    })
    c5.aplanar_metricas_en_el_registro(registro, metricas)
    return registro


def _regla_de_c0(proto: dict, protocolo_v2):
    r = proto["regla_de_confirmacion"]
    return dataclasses.replace(protocolo_v2.regla_de_cierre, puntos=float(r["puntos"]),
                               fraccion_minima=float(r["fraccion_minima"]),
                               definicion_de_mejor=r["definicion_de_mejor"])


def veredicto_de_la_regla(resultados: list[dict], proto: dict, protocolo_v2, datasets,
                          pliegues: dict) -> dict:
    """La regla escrita en el protocolo, con la aritmética de la Fase 0
    (`aplicar_regla_de_cierre`): TabM completa TODOS los conjuntos y queda a
    ≤ `puntos` de la mejor de las redes en ≥ `fraccion_minima` de ellos. Un fallo
    de TabM en cualquier pliegue cuenta como conjunto perdido. Además de lo que
    decide, el resultado enumera lo que se REGISTRA (plazo cortado, tiempo,
    memoria) y declara que no decide."""
    regla = _regla_de_c0(proto, protocolo_v2)
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo_v2, datasets)
    nombres = [d.nombre for d in datasets]
    planos = [r for r in resultados]
    cierre = protocolo_mod.aplicar_regla_de_cierre(
        planos, regla, motor=TABM, metrica_por_dataset=metrica_por_dataset,
        datasets_exigidos=nombres)
    esperados = {d.nombre: set(pliegues[d.nombre]) for d in datasets}
    incompletos = []
    for nombre in nombres:
        hechos = {r["pliegue"] for r in resultados
                  if r["dataset"] == nombre and r["motor"] == TABM
                  and r.get("estado") in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA}
        if hechos != esperados[nombre]:
            incompletos.append({"dataset": nombre, "faltan": sorted(esperados[nombre] - hechos)})
    completa = not incompletos
    con_plazo = sorted({(r["dataset"], r["pliegue"]) for r in resultados
                        if r["motor"] == TABM and r.get("parado_por_plazo")})
    cumple = bool(completa and cierre["cumple_la_regla"])
    return {
        "regla": regla.a_json(),
        "tabm_completa_todos": completa, "conjuntos_incompletos_de_tabm": incompletos,
        "cumplidos": cierre["cumplidos"], "de": cierre["datasets"],
        "cumple_la_regla_de_distancia": cierre["cumple_la_regla"],
        "veredicto": ("SE_OFRECE_RECOMENDADA_PARA_TABLAS" if cumple
                      else "SE_OFRECE_SIN_RECOMENDAR"),
        "tabm_con_el_plazo_cortado": [{"dataset": d, "pliegue": p} for d, p in con_plazo],
        "detalle": cierre["detalle"],
        "metrica_por_dataset": cierre.get("metrica_por_dataset"),
        "nota": ("el veredicto lo da la regla escrita en el protocolo; tiempo y memoria (`coste`) "
                 "se REGISTRAN para el aviso de antes de entrenar y NO deciden"),
    }


def _mediana(valores):
    valores = [v for v in valores if v is not None]
    return statistics.median(valores) if valores else None


def resumen_de_coste(resultados: list[dict]) -> dict:
    """Tiempo de pared y pico de memoria por red y por conjunto (media de los
    pliegues completados; máximo de memoria). Se registra, no decide."""
    por_red: dict = {}
    for r in resultados:
        if r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
            continue
        por_red.setdefault(r["motor"], {}).setdefault(r["dataset"], []).append(r)
    resumen = {}
    for red, por_ds in por_red.items():
        filas = {}
        for ds, rs in por_ds.items():
            filas[ds] = {
                "pliegues": len(rs), "cubo": rs[0]["cubo"],
                "n_train": rs[0]["particion"]["n_train"],
                "wall_s_medio": round(statistics.fmean(r["wall_s"] for r in rs), 2),
                "wall_s_max": round(max(r["wall_s"] for r in rs), 2),
                "rss_pico_mb_max": max(r["rss_pico_mb"] for r in rs),
                "epocas_medias": (round(statistics.fmean(r["epocas_ejecutadas"] for r in rs
                                                         if r.get("epocas_ejecutadas") is not None), 1)
                                  if any(r.get("epocas_ejecutadas") is not None for r in rs) else None),
                "con_plazo_cortado": sum(1 for r in rs if r.get("parado_por_plazo")),
            }
        resumen[red] = {"por_conjunto": filas,
                        "wall_s_mediana_de_conjuntos": _mediana(f["wall_s_medio"] for f in filas.values()),
                        "wall_s_max": max(f["wall_s_max"] for f in filas.values()),
                        "rss_pico_mb_max": max(f["rss_pico_mb_max"] for f in filas.values())}
    return resumen


def sello_de(documento: dict) -> str:
    """El sello: `digest_canonico` del documento ENTERO sin la clave `sello`."""
    return digest_canonico({k: v for k, v in documento.items() if k != "sello"})


def verificar_sello(documento: dict) -> bool:
    return documento.get("sello") == sello_de(documento)


def componer_resultado(*, tipo: str, proto: dict, resultados: list[dict], procedencia: dict,
                       veredicto: dict | None, coste: dict, parcial: bool,
                       reusados: int) -> dict:
    documento = {
        "corte": "122-C0", "tipo": tipo, "parcial": parcial,
        "protocolo": {"version": proto.get("version_protocolo"), "estado": proto.get("estado"),
                      "digest_sha256": proto.get("digest_sha256"),
                      "digest_calculado": autodigest(proto)},
        "procedencia": procedencia, "procedencia_id": procedencia["procedencia_id"],
        "resultados": resultados, "n_intentos": len(resultados), "n_reusados": reusados,
        "veredicto": veredicto, "coste": coste,
    }
    documento["sello"] = sello_de(documento)
    return documento


def escribir_atomicamente(ruta: Path, documento: dict) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(json.dumps(documento, indent=1, ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, ruta)


def cargar_salida_previa(ruta: Path, tipo: str) -> list[dict]:
    """Los registros de una salida anterior del MISMO tipo y con sello válido; de
    otro tipo se NIEGA (no mezclar un humo con una pasada) y con sello roto se
    ignora, diciéndolo."""
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    try:
        previo = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"AVISO: {ruta} no se puede leer: sin caché", flush=True)
        return []
    if previo.get("tipo") != tipo:
        raise SystemExit(f"{ruta} es de tipo {previo.get('tipo')!r} y esta ejecución es {tipo!r}: "
                         f"no se mezclan (usa otra --salida)")
    if not verificar_sello(previo):
        print(f"AVISO: el sello de {ruta} no cuadra con su contenido: sin caché", flush=True)
        return []
    return list(previo.get("resultados") or [])


# ===============================================================================================
# 8. LA EJECUCIÓN
# ===============================================================================================

def _argumentos(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--protocolo", default=str(RUTA_DEL_PROTOCOLO))
    p.add_argument("--salida", default=None)
    p.add_argument("--solo", default=None, help="nombres de conjuntos, separados por comas")
    p.add_argument("--humo", action="store_true", help="el bloque `humo` del protocolo")
    p.add_argument("--redes", default=",".join(REDES))
    p.add_argument("--forzar", action="store_true", help="ignora la caché entera")
    p.add_argument("--arff", default=None, help="directorio de los ARFF (por omisión el de la Fase 0)")
    p.add_argument("--git-del-host", default=None)
    p.add_argument("--escribir-git-del-host", default=None)
    p.add_argument("--sellar-protocolo", action="store_true")
    return p.parse_args(argv)


def tipo_de_ejecucion(args) -> str:
    return "humo" if args.humo else ("solo" if args.solo else "pasada")


def ruta_de_salida(tipo: str, salida: str | None) -> Path:
    if salida:
        return Path(salida)
    return _AQUI / {"pasada": "resultado_122_c0.json", "solo": "resultado_122_c0_solo.json",
                    "humo": "resultado_122_c0_humo.json"}[tipo]


def _plan(proto: dict, datasets, tipo: str, solo: str | None):
    """Qué conjuntos y qué pliegues de cada uno (y los topes del humo)."""
    humo = proto.get("humo") or {}
    if tipo == "humo":
        nombres = list(humo["conjuntos"])
        pliegues = {n: list(humo["pliegues"]) for n in nombres}
    else:
        nombres = ([n.strip() for n in solo.split(",")] if solo else [d.nombre for d in datasets])
        pliegues = {e["nombre"]: list(e["pliegues"]) for e in proto["datasets"]}
    desconocidos = [n for n in nombres if n not in {d.nombre for d in datasets}]
    if desconocidos:
        raise SystemExit(f"no están en el protocolo de C0: {desconocidos}")
    return [d for d in datasets if d.nombre in nombres], pliegues


def main(argv=None) -> None:
    args = _argumentos(argv)
    if args.escribir_git_del_host:
        escribir_git_del_host(Path(args.escribir_git_del_host))
        return
    if args.sellar_protocolo:
        print(sellar_protocolo(Path(args.protocolo)))
        return
    tipo = tipo_de_ejecucion(args)
    ruta_salida = ruta_de_salida(tipo, args.salida)
    ruta_protocolo = Path(args.protocolo)
    if args.arff:
        c3.ARFF_DIR = Path(args.arff)
    exigir_el_nucleo_del_arbol()

    c6.preparar_protocolo_v2()
    protocolo_v2 = c3.protocolo_registrado()
    proto = cargar_protocolo(ruta_protocolo)
    todos = exigir_el_protocolo(proto, protocolo_v2, tipo=tipo)
    if tipo != "pasada" and proto.get("estado") != ESTADO_REGISTRADO:
        print(f"AVISO: protocolo en estado {proto.get('estado')!r}: esta ejecución ({tipo}) NO "
              f"cuenta como medida de C0", flush=True)
    datasets, pliegues = _plan(proto, todos, tipo, args.solo)
    redes = [r for r in args.redes.split(",") if r]
    for r in redes:
        if r not in REDES:
            raise SystemExit(f"red desconocida {r!r}; conocidas: {REDES}")
    humo = proto.get("humo") or {}
    tabm_wall_s = float(humo["tabm_wall_s"]) if tipo == "humo" else None
    tope_nucleo_humo = float(humo["tope_duro_s"]) if tipo == "humo" else None

    git_del_host = (json.loads(Path(args.git_del_host).read_text()) if args.git_del_host else None)
    componentes = componentes_del_digest_del_entorno(ruta_protocolo)
    entorno_digest = digest_del_entorno(componentes)
    procedencia = componer_procedencia(
        datos_de_entrada={**{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
                          "protocolo_122_c0": ruta_protocolo,
                          "protocolo_exploratorio_v2": p119.RUTA_DEL_PROTOCOLO_V2},
        digests_de_codigo={"entorno": entorno_digest}, git_del_host=git_del_host)
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    print(f"carga al empezar: {procedencia['carga_al_empezar']}; núcleo "
          f"{procedencia['codigo_que_se_importa']['matrixai']}; motores "
          f"{procedencia['codigo_que_se_importa']['matrixai_engines']}; torch "
          f"{procedencia['codigo_que_se_importa']['torch']}; hilos {HILOS}", flush=True)

    previos = [] if args.forzar else cargar_salida_previa(ruta_salida, tipo)
    cache = {(r["dataset"], r["motor"], r["repeticion"], r["pliegue"]): r for r in previos}
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo_v2, datasets)
    semillas = protocolo_v2.particion.semillas
    repeticion = int(proto["particion"]["repeticion"])
    resultados: list[dict] = []
    reusados = [0]

    def guardar(parcial: bool) -> dict:
        veredicto = None
        if not parcial:
            veredicto = veredicto_de_la_regla(resultados, proto, protocolo_v2, datasets, pliegues)
        documento = componer_resultado(
            tipo=tipo, proto=proto,
            resultados=resultados, procedencia=procedencia, veredicto=veredicto,
            coste=resumen_de_coste(resultados), parcial=parcial, reusados=reusados[0])
        escribir_atomicamente(ruta_salida, documento)
        return documento

    inicio = time.perf_counter()
    for ds in datasets:
        t0 = time.perf_counter()
        conjunto = cargar_conjunto(ds, protocolo_v2)
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        print(f"\n=== {ds.nombre} ({ds.tarea}, cubo={ds.cubo}, n={len(conjunto['por_id'])}, "
              f"test={len(conjunto['test_ids'])}, carga={time.perf_counter() - t0:.1f}s) ===",
              flush=True)
        semilla = semillas[repeticion]
        for pliegue in pliegues[ds.nombre]:
            entrena, valida = pliegue_del_conjunto(conjunto, repeticion, pliegue)
            particion = ids_de_la_particion(entrena, valida, conjunto["test_ids"])
            plan_digest = conjunto["propuesta"].plan.digest()
            for red in redes:
                wall, tope = presupuesto_de_la_red(red, cubo=ds.cubo, proto=proto, semilla=semilla,
                                                   tabm_wall_s=tabm_wall_s)
                if tope_nucleo_humo is not None and red != TABM:
                    tope = tope_nucleo_humo
                config_digest = digest_de_la_configuracion(
                    red=red, wall_s=wall, tope_duro_s=tope, hilos=HILOS, semilla=semilla,
                    repeticion=repeticion, pliegue=pliegue, particion=particion)
                previo = cache.get((ds.nombre, red, repeticion, pliegue))
                if reusable(previo, entorno_digest=entorno_digest, config_digest=config_digest,
                            datos_sha256=datos_sha256):
                    resultados.append(dict(previo, reusado=True))
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} {red} pliegue={pliegue}", flush=True)
                    continue
                t_prep = time.perf_counter()
                train, val, test = particiones_para(red, conjunto, entrena, valida,
                                                    conjunto["test_ids"])
                preparacion_s = time.perf_counter() - t_prep
                salida = ejecutar_aislado(
                    red, train, val, test, conjunto["spec"], wall_seconds=wall, tope_duro_s=tope,
                    hilos=HILOS, seed=semilla, candidate=f"{MOTOR_DE_LA_RED[red]}-{red}",
                    split_plan_digest=plan_digest, dataset=ds.nombre, pliegue=pliegue,
                    repeticion=repeticion)
                registro = registro_de_un_intento(
                    ds=ds, red=red, repeticion=repeticion, pliegue=pliegue, semilla=semilla,
                    wall_s_pedido=wall, tope_duro_s=tope, particion=particion,
                    metric_id=metrica_por_dataset[ds.nombre], datos_sha256=datos_sha256,
                    entorno_digest=entorno_digest, config_digest=config_digest,
                    preparacion_s=preparacion_s, salida=salida)
                resultados.append(registro)
                print(f"  {ds.nombre} {red} pliegue={pliegue}: estado={registro['estado']} "
                      f"{registro['metrica_de_cierre']}={registro.get(registro['metrica_de_cierre'])} "
                      f"wall={registro['wall_s']}s rss_pico={registro.get('rss_pico_mb')}MB "
                      f"epocas={registro.get('epocas_ejecutadas')} "
                      f"plazo={registro.get('parado_por_plazo')} "
                      f"{('motivo=' + str(registro.get('motivo'))) if registro['estado'] == 'failed' else ''}",
                      flush=True)
                guardar(parcial=True)
    total = time.perf_counter() - inicio
    documento = guardar(parcial=False)
    v = documento["veredicto"]
    print(f"\n=== total {total:.1f}s, {len(resultados)} intentos ({reusados[0]} reusados) ===")
    print(f"VEREDICTO 122-C0{'' if tipo == 'pasada' else ' (' + tipo.upper() + ' -- no cuenta)'}: "
          f"TabM completa={v['tabm_completa_todos']}, a <= {v['regla']['puntos']} puntos en "
          f"{v['cumplidos']}/{v['de']} -> {v['veredicto']}; plazo cortado en "
          f"{len(v['tabm_con_el_plazo_cortado'])} pliegues. Sello {documento['sello'][:16]}; "
          f"guardado en {ruta_salida}")


if __name__ == "__main__":
    main()
