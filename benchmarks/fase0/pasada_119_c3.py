#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C3 — el motor nuevo (`matrixai.dense.tabm_cpu`, TabM + PLR) sobre los
32 conjuntos NO sellados de Fase 0, comparado contra la densa v2
(`pasada_v2_113_resultado.json`) en los MISMOS conjuntos.

Implementa el bloque `veredicto` de `protocolo_119_v4.json` (contrato 119,
`documentacion/119_LA_RED_DENSA_NUEVA_CONTRACT.md`): corre
`matrixai_engines.motores.densa_tabm.MotorDensaTabM()` — SIN parámetros, a
diferencia de la palanca de 118 — con las MISMAS particiones, semillas,
repeticiones y presupuesto por cubo que registra el protocolo v2
(`protocolo_exploratorio_v2.json`, que es el que DE VERDAD se lee). El v2 y
el v4 son idénticos en `particion`, `presupuesto`, `regla_de_cierre` y
`datasets`: se COMPRUEBA al arrancar (`exigir_los_protocolos_encadenados`,
que para la pasada si divergen) y lo fija una prueba bloque a bloque. El
«campo» se compone sustituyendo, SOLO en los conjuntos medidos aquí, la densa
v2 (`matrixai.dense.torch_cpu`) por el motor nuevo.

**LOS 8 SELLADOS SE NIEGAN.** El protocolo v4 (`veredicto.regla_de_subida`)
reserva los sellados para C4: un sellado en `--solo` para la pasada con
`SystemExit` (en 118 solo se avisaba).

LO QUE SE REPARÓ TRAS LA AUDITORÍA 2 (29-09), por hallazgo:

* **I2, los fallos en el veredicto** (decisión del supervisor, que irá
  declarada en la enmienda 2 del protocolo; ver `COMO_SE_CUENTAN_LOS_FALLOS`):
  (a) condición 1: un conjunto con CUALQUIER intento del motor nuevo sin
  medida —fallido, AUSENTE del artefacto o completado sin la métrica de
  cierre— cuenta como perdido; (b) condición 2: fallos del motor nuevo con la
  densa v2 completa = INFERIORIDAD, y donde la densa v2 falló TODOS sus
  intentos (Allstate, KDDCup09) = «sin comparación», que solo cuenta en la
  condición 1; (c) la emparejada, solo sobre repeticiones y pliegues
  completos en los DOS; (d) `veredicto_por_conjunto` lleva los fallos.
* **I4, el digest de la caché**: cubre el CIERRE ESTÁTICO de imports de este
  guion dentro de los dos repos (190 ficheros, medido; un intento de kc2 en
  proceso carga 124 y los 124 están dentro), los tres ficheros de protocolo
  que se leen, y las versiones de python, numpy, scikit-learn y torch. Lo
  que NO cubre va escrito en el propio resultado
  (`lo_que_no_cubre_el_digest`), y el sha256 del ARFF va en la clave de cada
  registro.
* **I6, las guardias**: al arrancar, el motor tiene que usar k=8,
  d_block=256, n_blocks=2 (sus CONSTANTES, no un comentario); el v4 cuadra
  con su digest y la enmienda 1 cita ese digest; cada intento registra
  `arquitectura` e `hiperparametros` y la pasada para si la arquitectura
  declarada no es la de la enmienda; la partición y la lectura de cada
  conjunto se comparan con las de la v2 ANTES de su primer intento.
* **I7, que una noche no se pierda**: la salida se FUSIONA con la que ya
  hay; `--solo`, `--humo` y `--estimar` escriben por omisión en ficheros
  aparte y se niegan a escribir en un resultado de otro tipo; `--salida`
  admite una ruta absoluta fuera del árbol (la cola nocturna borra sus
  worktrees: el caché tiene que vivir fuera para que la noche siguiente
  continúe).
* **I1, la estimación**: mide UN intento por conjunto de una lista con más
  de un conjunto por cubo (anchos incluidos), con la preparación del padre
  aparte, y dice si cabe en la ventana nocturna o de día — con 1 proceso,
  que es lo que este guion corre.

QUÉ SE REUTILIZA, Y DE DÓNDE — nada se copia a mano:

* catálogo, lectura, partición y guardia de CPU: `pasada_114c6_ensamblado`
  y `pasada_amplia_101_c5` (la MISMA partición, semillas y estratificación
  que la v2);
* el intento en subproceso con su tope: `matrixai_engines.subproceso`;
* procedencia, sellado y escritura: `pasada_exploratoria_101_c3`;
* el campo, la emparejada con bootstrap y la aritmética de la regla de
  subida: `pasada_118_palanca` (importadas TAL CUAL; aquí solo se añade,
  por fuera, cómo cuentan los fallos);
* la regla de cierre: `protocolo.aplicar_regla_de_cierre`.

CÓMO SE LANZA:

    python3 benchmarks/fase0/pasada_119_c3.py --estimar --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c3.py --humo --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c3.py --solo diabetes --salida <ruta>
    python3 benchmarks/fase0/pasada_119_c3.py --salida <ruta absoluta, fuera del árbol>
        # LA PASADA ENTERA sobre los 32 no sellados -- horas, a la cola
        # nocturna (decisión del 25-09), nunca a mano.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import hashlib
import json
import math
import re
import resource
import sys
import time
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_118_palanca as p118  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

from matrixai_engines.motores.densa_tabm import MotorDensaTabM  # noqa: E402
from matrixai_engines.particiones import Particion, Presupuesto  # noqa: E402
from matrixai_engines.procedencia import versiones_de_bibliotecas  # noqa: E402
from matrixai_engines.subproceso import (MARGEN_POR_DEFECTO_SEGUNDOS,  # noqa: E402
                                         ejecutar_intento_aislado)

RUTA_DEL_PROTOCOLO_V4 = _AQUI / "protocolo_119_v4.json"
RUTA_DE_LA_ENMIENDA_1 = _AQUI / "protocolo_119_v4_enmienda_1.json"
#: La enmienda que declarará cómo cuentan los fallos (I2). Si no existe aún,
#: el resultado lo DICE; no entra en el digest de la caché (no cambia lo que
#: ejecuta un intento: solo cómo se cuenta, y eso se recalcula cada vez).
RUTA_DE_LA_ENMIENDA_2 = _AQUI / "protocolo_119_v4_enmienda_2.json"
#: El protocolo que DE VERDAD se lee para medir (`c3.protocolo_registrado()`
#: tras `c6.preparar_protocolo_v2()`, y el catálogo de `fijar_el_catalogo_v2`).
RUTA_DEL_PROTOCOLO_V2 = c6.RUTA_DEL_PROTOCOLO_V2
RUTA_V2_RESULTADO = p118.RUTA_V2_RESULTADO  # pasada_v2_113_resultado.json

#: EL RESULTADO REAL del corte. Ni `--solo` ni `--humo` escriben aquí nunca.
RUTA_DEL_RESULTADO = _AQUI / "pasada_119_c3_resultado.json"
RUTA_DE_LA_ESTIMACION = _AQUI / "estimacion_pasada_119_c3.json"
SALIDA_POR_OMISION = {
    "pasada": RUTA_DEL_RESULTADO,
    "solo": _AQUI / "pasada_119_c3_solo_resultado.json",
    "humo": _AQUI / "pasada_119_c3_humo_resultado.json",
    "estimar": RUTA_DE_LA_ESTIMACION,
}

#: El motor NUEVO de este corte — SIN parámetros. Se comprueba al importar
#: que el nombre no ha derivado del que este guion tiene escrito.
NOMBRE_MOTOR_NUEVO = "matrixai.dense.tabm_cpu"
assert MotorDensaTabM().nombre == NOMBRE_MOTOR_NUEVO, (
    f"MotorDensaTabM().nombre es {MotorDensaTabM().nombre!r}, y este guion tiene escrito "
    f"{NOMBRE_MOTOR_NUEVO!r} -- actualizar la constante, no ignorar la discrepancia")

#: La densa VIEJA (v2), la clave que `campo_de_la_comparacion` sustituye.
NOMBRE_DENSA_V2 = p118.NOMBRE_DENSA

#: Los bloques que el v2 (el que se lee) y el v4 (el que se cita) tienen que
#: traer IDÉNTICOS para que medir con el v2 sea medir con el v4.
BLOQUES_QUE_LA_V2_Y_EL_V4_COMPARTEN = ("particion", "presupuesto", "regla_de_cierre", "datasets")

#: LA CONFIGURACIÓN DE LA ENMIENDA 1 (`que_cambia`: «k=8 y d_block=256
#: (n_blocks=2)»), y la constante del motor que la fija. La guardia lee las
#: CONSTANTES del módulo del motor, y cada intento, la arquitectura que el
#: motor DECLARA haber usado.
CONFIGURACION_DE_LA_ENMIENDA_1 = {"k": 8, "d_block": 256, "n_blocks": 2}
CONSTANTES_DEL_MOTOR = {"k": "K_CABEZAS", "d_block": "D_BLOCK", "n_blocks": "N_BLOCKS"}

SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL = 60.0


# ---------------------------------------------------------------------------
# 1. LOS CONJUNTOS: los 32 NO sellados -- los 8 sellados se NIEGAN (son de C4)
# ---------------------------------------------------------------------------

def datasets_de_c3(todos: list, no_sellados: list, *, solo: str | None) -> tuple[list, list | None]:
    """Los conjuntos que esta pasada mide: los 32 no sellados, o el
    subconjunto de `--solo` -- SIEMPRE dentro de los no sellados."""
    if not solo:
        return list(no_sellados), None
    pedidos = [n.strip() for n in solo.split(",") if n.strip()]
    por_nombre = {d.nombre: d for d in todos}
    desconocidos = [n for n in pedidos if n not in por_nombre]
    if desconocidos:
        raise SystemExit(f"--solo nombra conjuntos que no están en el protocolo v2: "
                         f"{desconocidos}")
    sellados_pedidos = [n for n in pedidos if por_nombre[n].sellado]
    if sellados_pedidos:
        raise SystemExit(
            f"--solo pide conjunto(s) SELLADO(S) {sellados_pedidos}: 119-C3 mide SOLO los "
            f"32 no sellados -- los 8 sellados son de C4 (regla_de_subida del protocolo v4: "
            f"«sube a confirmación en los sellados... si cumple las DOS condiciones sobre "
            f"los 32 no sellados»). Se niega en vez de correr «solo como prueba»")
    return [por_nombre[n] for n in pedidos], pedidos


# ---------------------------------------------------------------------------
# 2. LAS GUARDIAS (I6): los protocolos encadenados y la configuración del motor
# ---------------------------------------------------------------------------

def _leer_json(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def autodigest(payload: dict) -> str:
    """El digest que el v4 y su enmienda llevan DENTRO (`digest_sha256`):
    sha256 del JSON sin ese campo, claves ordenadas, sin espacios y sin
    escapar. Comprobado sobre los dos ficheros: reproduce los dos valores."""
    sin_el_campo = {k: v for k, v in payload.items() if k != "digest_sha256"}
    return hashlib.sha256(json.dumps(sin_el_campo, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonico(valor) -> str:
    return json.dumps(valor, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def exigir_los_protocolos_encadenados() -> dict:
    """PARA la pasada si la cadena v4 -> enmienda 1 no cuadra, o si el v2
    (el que se lee) y el v4 (el que se cita) divergen en lo que importa.

    1. el v4 cuadra con su propio `digest_sha256`;
    2. la enmienda 1 cuadra con el suyo;
    3. la enmienda 1 dice ser DE ese v4 (`de.digest_sha256` y `de.protocolo`);
    4. `particion`, `presupuesto`, `regla_de_cierre` y `datasets` son
       idénticos en el v2 y en el v4 (JSON canónico).
    """
    v4 = _leer_json(RUTA_DEL_PROTOCOLO_V4)
    enmienda = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    v2 = _leer_json(RUTA_DEL_PROTOCOLO_V2)
    problemas = []
    if autodigest(v4) != v4.get("digest_sha256"):
        problemas.append(f"el v4 ({RUTA_DEL_PROTOCOLO_V4.name}) NO cuadra con su propio "
                         f"digest_sha256: se ha tocado después de registrarlo")
    if autodigest(enmienda) != enmienda.get("digest_sha256"):
        problemas.append(f"la enmienda 1 ({RUTA_DE_LA_ENMIENDA_1.name}) NO cuadra con su propio "
                         f"digest_sha256")
    de = enmienda.get("de") or {}
    if de.get("digest_sha256") != v4.get("digest_sha256"):
        problemas.append(f"la enmienda 1 dice ser del protocolo con digest "
                         f"{str(de.get('digest_sha256'))[:16]} y el v4 es "
                         f"{str(v4.get('digest_sha256'))[:16]}")
    if de.get("protocolo") != v4.get("version_protocolo"):
        problemas.append(f"la enmienda 1 dice ser de {de.get('protocolo')!r} y el v4 es "
                         f"{v4.get('version_protocolo')!r}")
    bloques = {}
    for bloque in BLOQUES_QUE_LA_V2_Y_EL_V4_COMPARTEN:
        iguales = bloque in v2 and bloque in v4 and _canonico(v2[bloque]) == _canonico(v4[bloque])
        bloques[bloque] = iguales
        if not iguales:
            problemas.append(f"el bloque {bloque!r} NO es idéntico en {RUTA_DEL_PROTOCOLO_V2.name} "
                             f"(el que se lee) y en {RUTA_DEL_PROTOCOLO_V4.name} (el que se cita)")
    if problemas:
        raise SystemExit("la cadena de protocolos no cuadra; no se mide:\n  - "
                         + "\n  - ".join(problemas))
    return {
        "v4": {"version": v4["version_protocolo"], "digest_sha256_declarado": v4["digest_sha256"],
               "cuadra_con_su_digest": True},
        "enmienda_1": {"enmienda": enmienda.get("enmienda"),
                       "digest_sha256_declarado": enmienda["digest_sha256"],
                       "cuadra_con_su_digest": True, "de": de, "cita_el_v4": True},
        "bloques_identicos_v2_v4": bloques,
        "como_se_comprueba": ("autodigest = sha256 del JSON sin `digest_sha256`, claves "
                              "ordenadas, separadores compactos, sin escapar; los bloques, "
                              "por su JSON canónico"),
    }


def enmienda_2_declarada() -> dict:
    """La enmienda 2 (cómo cuentan los fallos), si ya está registrada. Si no
    lo está, se DICE; si está y no cuadra con su propio digest, PARA."""
    if not RUTA_DE_LA_ENMIENDA_2.exists():
        return {"presente": False, "ruta": RUTA_DE_LA_ENMIENDA_2.name,
                "aviso": ("la enmienda 2 del protocolo v4 NO está registrada todavía: los "
                          "criterios de fallos aplicados (COMO_SE_CUENTAN_LOS_FALLOS) son los del "
                          "encargo del supervisor tras la auditoría 2, sin registro escrito")}
    payload = _leer_json(RUTA_DE_LA_ENMIENDA_2)
    declarado = payload.get("digest_sha256")
    if declarado is not None and autodigest(payload) != declarado:
        raise SystemExit(f"{RUTA_DE_LA_ENMIENDA_2.name} NO cuadra con su propio digest_sha256")
    return {"presente": True, "ruta": RUTA_DE_LA_ENMIENDA_2.name,
            "sha256_del_fichero": hashlib.sha256(RUTA_DE_LA_ENMIENDA_2.read_bytes()).hexdigest(),
            "digest_sha256_declarado": declarado, "de": payload.get("de")}


def configuracion_que_declara_la_enmienda(texto: str) -> dict:
    """k, d_block y n_blocks tal como los escribe `que_cambia` de la
    enmienda 1 («k=8 y d_block=256 (n_blocks=2)»), o `None` si no los dice."""
    patrones = {"k": r"(?<![A-Za-z0-9_])k\s*=\s*(\d+)",
                "d_block": r"(?<![A-Za-z0-9_])d_block\s*=\s*(\d+)",
                "n_blocks": r"(?<![A-Za-z0-9_])n_blocks\s*=\s*(\d+)"}
    salida = {}
    for clave, patron in patrones.items():
        m = re.search(patron, texto or "")
        salida[clave] = int(m.group(1)) if m else None
    return salida


def exigir_la_configuracion_de_la_enmienda() -> dict:
    """PARA si el motor no está configurado como dice la enmienda 1: lee las
    CONSTANTES del módulo donde vive `MotorDensaTabM` (las que `_ajustar`
    pasa a la red), no un comentario ni un docstring."""
    enmienda = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    declarada = configuracion_que_declara_la_enmienda(enmienda.get("que_cambia", ""))
    if declarada != CONFIGURACION_DE_LA_ENMIENDA_1:
        raise SystemExit(
            f"la enmienda 1 declara {declarada} y este guion tiene escrito "
            f"{CONFIGURACION_DE_LA_ENMIENDA_1}: no se mide hasta que digan lo mismo")
    modulo = sys.modules[MotorDensaTabM.__module__]
    leida = {clave: getattr(modulo, nombre, None) for clave, nombre in CONSTANTES_DEL_MOTOR.items()}
    if leida != CONFIGURACION_DE_LA_ENMIENDA_1:
        raise SystemExit(
            f"el motor ({MotorDensaTabM.__module__}: {CONSTANTES_DEL_MOTOR}) usa {leida} y la "
            f"enmienda 1 fija {CONFIGURACION_DE_LA_ENMIENDA_1}: medir otra configuración no es "
            f"medir el corte. No se mide")
    return {"declarada_por_la_enmienda_1": declarada, "leida_del_motor": leida,
            "constantes": dict(CONSTANTES_DEL_MOTOR), "modulo": MotorDensaTabM.__module__}


def exigir_la_arquitectura_del_intento(registro: dict) -> None:
    """Un intento COMPLETADO tiene que declarar la arquitectura de la
    enmienda 1. Si no la declara, no se puede saber qué corrió: también para."""
    if registro.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
        return
    arquitectura = registro.get("arquitectura")
    if not isinstance(arquitectura, dict):
        raise SystemExit(
            f"{registro['dataset']} rep={registro['repeticion']} pliegue={registro['pliegue']}: "
            f"el motor completó el intento SIN declarar su arquitectura -- no se puede "
            f"comprobar que corrió la de la enmienda 1. Se para la pasada")
    leida = {k: arquitectura.get(k) for k in CONFIGURACION_DE_LA_ENMIENDA_1}
    if leida != CONFIGURACION_DE_LA_ENMIENDA_1:
        raise SystemExit(
            f"{registro['dataset']} rep={registro['repeticion']} pliegue={registro['pliegue']}: "
            f"el motor declara haber corrido {leida} y la enmienda 1 fija "
            f"{CONFIGURACION_DE_LA_ENMIENDA_1}. Se para la pasada")


def _payload_v2() -> dict:
    if not RUTA_V2_RESULTADO.exists():
        raise SystemExit(
            f"no está {RUTA_V2_RESULTADO}: sin la pasada v2 no hay con qué comparar el "
            f"motor nuevo -- este guion no inventa una densa v2 que no se ha medido")
    return _leer_json(RUTA_V2_RESULTADO)


def _resultados_v2() -> list[dict]:
    return _payload_v2()["resultados"]


def _normalizado(valor):
    return json.loads(json.dumps(valor, ensure_ascii=False))


def exigir_la_particion_de_la_v2(ds, particion_declarada: dict, lectura: dict | None,
                                 payload_v2: dict) -> dict:
    """PARA si la partición (plan_digest, pliegues obtenidos, límites, filas,
    test, predictores) o la lectura (objetivo, columnas excluidas,
    normalizaciones) de ESTE conjunto no son las que la v2 registró. Se
    comprueba ANTES del primer intento del conjunto: comparar los 32 al
    arrancar exigiría cargar todos los ARFF una vez más (Allstate tarda 46 s
    solo en su partición)."""
    registrada = (payload_v2.get("particion_por_dataset") or {}).get(ds.nombre)
    if registrada is None:
        raise SystemExit(f"{ds.nombre}: la v2 no registra su partición: no hay con qué "
                         f"comparar. No se mide")
    propia = _normalizado(particion_declarada)
    if propia != registrada:
        distintas = sorted(k for k in set(propia) | set(registrada)
                           if propia.get(k) != registrada.get(k))
        raise SystemExit(
            f"{ds.nombre}: la partición NO es la de la v2 (difieren {distintas}: "
            f"{ {k: (propia.get(k), registrada.get(k)) for k in distintas} }). "
            f"Comparar con la v2 sobre otra partición no es la comparación del protocolo")
    lectura_v2 = (payload_v2.get("lectura_de_los_datos") or {}).get(str(ds.data_id))
    if lectura is not None and lectura_v2 is not None and _normalizado(lectura) != lectura_v2:
        raise SystemExit(f"{ds.nombre}: la LECTURA del ARFF no es la de la v2 "
                         f"(hoy {lectura}, v2 {lectura_v2}). No se mide")
    return {"plan_digest": propia.get("plan_digest"), "identica_a_la_v2": True,
            "lectura_identica_a_la_v2": lectura_v2 is not None and lectura is not None}


def _exigir_que_quepa() -> None:
    c6._exigir_que_quepa()


# ---------------------------------------------------------------------------
# 3. EL ENTORNO Y LA CACHÉ (I4)
# ---------------------------------------------------------------------------

#: El motor nuevo entero: la clase, la red y su preparación (digest POR
#: MOTOR, como `_FICHERO_POR_MOTOR` de C3). También están en el del entorno.
_FICHEROS_DEL_MOTOR_NUEVO = (
    c3._DIR_ENGINES / "motores" / "densa_tabm.py",
    c3._DIR_ENGINES / "redes" / "tabm_plr.py",
    c3._DIR_ENGINES / "redes" / "preparacion_tabm.py",
)

#: Las bibliotecas que entran en el digest POR VERSIÓN (la clave es la de
#: `versiones_de_bibliotecas()`, la misma que registra la procedencia).
BIBLIOTECAS_EN_EL_DIGEST = ("python", "numpy", "scikit-learn", "torch")


def _raices_de_import() -> tuple[Path, ...]:
    """El orden de `sys.path` que ven el padre y el hijo: `c3` antepone los
    `src` de engines y el core, y este guion su propio directorio."""
    return (c3._RAIZ_DE_ENGINES.resolve(), c3._RAIZ_DEL_CORE.resolve(), _AQUI)


def _ficheros_de_un_import(nombre: str, raices: tuple[Path, ...]) -> list[Path]:
    """Los ficheros que ejecuta `import <nombre>` si es de los repos
    (paquetes padre incluidos, que también se ejecutan), o [] si es de
    terceros o de la biblioteca estándar."""
    partes = nombre.split(".")
    for raiz in raices:
        if not ((raiz / partes[0]).is_dir() or (raiz / f"{partes[0]}.py").is_file()):
            continue
        ficheros, actual = [], raiz
        for parte in partes:
            directorio, fichero = actual / parte, actual / f"{parte}.py"
            if (directorio / "__init__.py").is_file():
                ficheros.append(directorio / "__init__.py")
                actual = directorio
            elif fichero.is_file():
                ficheros.append(fichero)
                break
            elif directorio.is_dir():  # paquete de espacio de nombres
                actual = directorio
            else:
                break
        return ficheros
    return []


def _paquete_de(fichero: Path, raices: tuple[Path, ...]) -> list[str]:
    for raiz in raices:
        try:
            partes = list(fichero.relative_to(raiz).with_suffix("").parts)
        except ValueError:
            continue
        return partes[:-1]
    return []


@functools.lru_cache(maxsize=None)
def _cierre_de_imports(entrada: Path, raices: tuple[Path, ...]) -> tuple[tuple[Path, ...],
                                                                          tuple[str, ...]]:
    """El CIERRE ESTÁTICO de imports de `entrada` dentro de `raices`: cada
    `import`/`from … import` escrito en el código —TAMBIÉN los de dentro de
    funciones, que un `sys.modules` del padre no ve porque se importan en el
    hijo—, seguido recursivamente. Es un SUPERCONJUNTO de lo que corre (un
    import dentro de una rama no tomada también entra): mejor invalidar de
    más que reusar con código distinto. Devuelve además los sitios con un
    import DINÁMICO (`import_module`/`__import__`), que no se pueden seguir."""
    pendientes, vistos, dinamicos = [entrada.resolve()], set(), []
    while pendientes:
        fichero = pendientes.pop()
        if fichero in vistos:
            continue
        vistos.add(fichero)
        arbol = ast.parse(fichero.read_text(encoding="utf-8"), filename=str(fichero))
        paquete = _paquete_de(fichero, raices)
        for nodo in ast.walk(arbol):
            nombres: list[str] = []
            if isinstance(nodo, ast.Import):
                nombres = [a.name for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom):
                if nodo.level:
                    base = paquete[: len(paquete) - (nodo.level - 1)]
                    modulo = ".".join(base + ([nodo.module] if nodo.module else []))
                else:
                    modulo = nodo.module or ""
                nombres = [modulo] + [f"{modulo}.{a.name}" for a in nodo.names if a.name != "*"]
            elif isinstance(nodo, ast.Call):
                funcion = nodo.func
                nombre = (funcion.id if isinstance(funcion, ast.Name)
                          else funcion.attr if isinstance(funcion, ast.Attribute) else None)
                if nombre in ("import_module", "__import__"):
                    dinamicos.append(f"{_etiqueta(fichero)}:{nodo.lineno}: "
                                     f"{ast.unparse(nodo)[:120]}")
            for n in nombres:
                for g in _ficheros_de_un_import(n, raices):
                    if g.resolve() not in vistos:
                        pendientes.append(g.resolve())
    return tuple(sorted(vistos)), tuple(sorted(dinamicos))


def _etiqueta(ruta: Path) -> str:
    """La ruta RELATIVA a su repo: el digest no puede depender de dónde está
    el árbol (el principal, un worktree de la cola o una copia)."""
    r = Path(ruta).resolve()
    for nombre, raiz in (("engines", c3._RAIZ_DE_ENGINES), ("core", c3._RAIZ_DEL_CORE)):
        try:
            return f"{nombre}:{r.relative_to(raiz.resolve()).as_posix()}"
        except ValueError:
            continue
    return f"fuera:{r.as_posix()}"


def ficheros_del_entorno() -> tuple[Path, ...]:
    """El cierre de imports de este guion MÁS los compartidos de C5 (algunos,
    como `dense_forward.py`, ya están en el cierre) y los del motor nuevo."""
    cierre, _ = _cierre_de_imports(Path(__file__).resolve(), _raices_de_import())
    extra = {Path(p).resolve() for p in c5._FICHEROS_COMPARTIDOS + _FICHEROS_DEL_MOTOR_NUEVO}
    return tuple(sorted(set(cierre) | extra))


def _datos_del_entorno() -> dict[str, Path]:
    """Los ficheros de DATOS que deciden qué se mide: el protocolo que se lee
    y los dos que se citan."""
    return {"protocolo_exploratorio_v2": RUTA_DEL_PROTOCOLO_V2,
            "protocolo_119_v4": RUTA_DEL_PROTOCOLO_V4,
            "protocolo_119_v4_enmienda_1": RUTA_DE_LA_ENMIENDA_1}


def _versiones_en_el_digest() -> dict[str, str]:
    versiones = versiones_de_bibliotecas()
    return {b: str(versiones.get(b, "ausente")) for b in BIBLIOTECAS_EN_EL_DIGEST}


def componentes_del_digest_del_entorno() -> dict[str, str]:
    componentes = {_etiqueta(f): c3._digest_fichero(f) for f in ficheros_del_entorno()}
    for nombre, ruta in _datos_del_entorno().items():
        componentes[f"dato:{nombre}"] = c3._digest_fichero(ruta)
    for nombre, version in _versiones_en_el_digest().items():
        componentes[f"version:{nombre}"] = version
    return componentes


def _digest_de(rutas: tuple[Path, ...]) -> str:
    return hashlib.sha256(
        "".join(c3._digest_fichero(f) for f in rutas).encode()).hexdigest()[:16]


def _digest_de_componentes(componentes: dict[str, str]) -> str:
    return hashlib.sha256(_canonico(componentes).encode()).hexdigest()[:16]


def _digest_entorno() -> str:
    return _digest_de_componentes(componentes_del_digest_del_entorno())


def _digest_motor_nuevo() -> str:
    return _digest_de(_FICHEROS_DEL_MOTOR_NUEVO)


def lo_que_no_cubre_el_digest() -> dict:
    """Lo que el digest de la caché NO cubre, dicho dentro del resultado —
    como hace CLAUDE.md con el de 101-C5—, y lo que se puede, MEDIDO."""
    _, dinamicos = _cierre_de_imports(Path(__file__).resolve(), _raices_de_import())
    cubiertos = {str(p) for p in ficheros_del_entorno()}
    prefijos = tuple(f"{r}/" for r in (c3._RAIZ_DE_ENGINES.resolve(), c3._RAIZ_DEL_CORE.resolve()))
    # `is_file()`: torch registra módulos con un `__file__` RELATIVO
    # («_classes.py») que `resolve()` pega al directorio actual; no existen.
    fuera = sorted(_etiqueta(Path(f)) for f in c3._ficheros_importados_por_este_proceso()
                   if f.endswith(".py") and f.startswith(prefijos) and f not in cubiertos
                   and Path(f).is_file())
    versiones = versiones_de_bibliotecas()
    return {
        "como_se_calcula": (
            "cierre ESTÁTICO de imports de pasada_119_c3.py dentro de los dos repos (ast de cada "
            "fichero, TODOS sus import, también los de dentro de funciones), más los compartidos "
            "de C5 y los tres del motor. Es un superconjunto de lo que corre: medido el 29-09, un "
            "intento de kc2 en proceso cargó 124 módulos de los repos y los 124 estaban en el "
            "cierre (190)"),
        "bibliotecas_de_terceros_que_NO_invalidan": sorted(
            k for k in versiones if k not in BIBLIOTECAS_EN_EL_DIGEST),
        "bibliotecas_de_terceros": (
            f"solo por versión: {', '.join(BIBLIOTECAS_EN_EL_DIGEST)} entran en el digest; las "
            f"demás constan en procedencia.versiones_de_bibliotecas pero un cambio en ellas NO "
            f"invalida la caché. Su código tampoco entra, ni el de la biblioteca estándar"),
        "imports_dinamicos_en_el_cierre": list(dinamicos),
        "imports_dinamicos": ("un import_module/__import__ con una cadena no se sigue; los "
                              "listados (medidos en el cierre) cargan bibliotecas de terceros"),
        "datos": ("los ARFF NO entran en el digest del entorno: su sha256 va en la clave de cada "
                  "registro (`datos_sha256`, que `_reusable_c3` compara). pasada_v2_113_resultado"
                  ".json no entra: no cambia lo que ejecuta un intento, solo la comparación, que "
                  "se recalcula en cada ejecución"),
        "enmienda_2": ("no entra: fija cómo se CUENTAN los fallos, no lo que se ejecuta; el "
                       "veredicto se recalcula en cada ejecución"),
        "modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest": fuera,
        "que_dice_la_lista_de_arriba": (
            "COTA INFERIOR: los módulos del padre; el hijo de multiprocessing.spawn importa los "
            "suyos, que el cierre estático cubre salvo import dinámico. Tiene que estar vacía"),
    }


def _reusable_c3(previo: dict | None, entorno_digest: str, motor_digest: str,
                 wall_seconds: float, *, datos_sha256: str | None) -> bool:
    """`c3._reusable` MÁS lo que esa función no comprueba: el estado tiene
    que contar como medida (un `failed` con los digests buenos se reusaría
    para siempre: trampa nº1 de «ANTES DE RELANZAR UNA PASADA») y el ARFF
    tiene que ser el mismo (su sha256 va en el registro)."""
    if not c3._reusable(previo, entorno_digest, motor_digest, wall_seconds):
        return False
    if previo.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
        return False
    return datos_sha256 is not None and previo.get("datos_sha256") == datos_sha256


# ---------------------------------------------------------------------------
# 4. LA SALIDA (I7): de qué tipo es, dónde va, y la fusión con la que ya hay
# ---------------------------------------------------------------------------

TIPOS_DE_EJECUCION = ("pasada", "solo", "humo", "estimar")


def tipo_de_ejecucion(*, humo: bool, solo: str | None, estimar: bool) -> str:
    if estimar:
        return "estimar"
    if humo:
        return "humo"
    if solo:
        return "solo"
    return "pasada"


def ruta_de_salida(tipo: str, salida: str | None) -> Path:
    """Dónde escribe cada tipo. `--solo`/`--humo`/`--estimar` van, por
    omisión, a ficheros APARTE, y NUNCA al resultado real aunque se pida con
    `--salida`. Una ruta absoluta fuera del árbol vale (la cola nocturna)."""
    ruta = (Path(salida).expanduser() if salida else SALIDA_POR_OMISION[tipo]).resolve()
    if tipo != "pasada" and ruta == RUTA_DEL_RESULTADO.resolve():
        raise SystemExit(f"--{tipo} no escribe en el resultado real ({RUTA_DEL_RESULTADO.name}): "
                         f"una prueba no puede pisar una noche de medición")
    return ruta


def tipo_del_fichero(payload: dict) -> str:
    """De qué tipo es un resultado ya escrito (también los de antes de que
    existiera el campo `tipo_de_ejecucion`)."""
    if payload.get("tipo_de_ejecucion") in TIPOS_DE_EJECUCION:
        return payload["tipo_de_ejecucion"]
    if str(payload.get("sub_corte", "")).startswith("estimacion"):
        return "estimar"
    if payload.get("es_humo"):
        return "humo"
    if payload.get("es_subconjunto_de_prueba"):
        return "solo"
    return "pasada"


def cargar_salida_previa(ruta: Path, tipo: str) -> tuple[dict, dict]:
    """El caché y el fichero entero, NEGÁNDOSE a mezclar tipos: un `--humo`
    sobre un resultado real (o una pasada sobre uno de humo) para aquí, antes
    de medir nada y sin tocar el fichero."""
    if not ruta.exists():
        return {}, {}
    payload = _leer_json(ruta)
    del_fichero = tipo_del_fichero(payload)
    if del_fichero != tipo:
        raise SystemExit(f"{ruta} es un resultado de tipo «{del_fichero}» y esta ejecución es "
                         f"«{tipo}»: no se mezclan. Otra --salida, o mover ese fichero a mano")
    cache = {(r["dataset"], r["motor"], r["repeticion"], r["pliegue"]): r
             for r in payload.get("resultados", [])}
    return cache, payload


def _clave(r: dict) -> tuple:
    return (r["dataset"], r["motor"], r["repeticion"], r["pliegue"])


def fusionar_resultados(previos: list[dict], de_esta_ejecucion: list[dict]) -> tuple[list[dict],
                                                                                     list[dict]]:
    """Lo que ya había y NO se ha vuelto a medir (o reusar) en esta
    ejecución, seguido de lo de esta ejecución. Devuelve también los
    conservados, para declararlos. Un punto de control a mitad de la noche
    ya no puede dejar el fichero solo con los conjuntos recorridos."""
    nuevas = {_clave(r) for r in de_esta_ejecucion}
    conservados = [r for r in previos if _clave(r) not in nuevas]
    return conservados + list(de_esta_ejecucion), conservados


# ---------------------------------------------------------------------------
# 5. EL VEREDICTO (I2): cómo cuentan los fallos
# ---------------------------------------------------------------------------

COMO_SE_CUENTAN_LOS_FALLOS = {
    "fuente": ("decisión del supervisor tras la auditoría 2 de 119-C3 (29-09), que irá declarada "
               "en la enmienda 2 del protocolo v4 (ver `enmienda_2`)"),
    "que_es_un_fallo": ("un intento ESPERADO por la partición (repetición y pliegue que "
                        "`propuesta.pliegues` trae) que no está en el artefacto, cuyo estado no "
                        "cuenta como medida (ESTADOS_QUE_CUENTAN_COMO_MEDIDA), o que se completó "
                        "sin la métrica de cierre"),
    "a_condicion_1": ("un conjunto con CUALQUIER fallo del motor nuevo cuenta como PERDIDO (no "
                      "cumple), como dice regla_de_cierre.definicion_de_mejor. Mismo criterio "
                      "para la densa v2"),
    "b_condicion_2": ("fallos del motor nuevo con la densa v2 completa = INFERIORIDAD; si la densa "
                      "v2 falló TODOS sus intentos (Allstate, KDDCup09) = «sin comparación», ni "
                      "mejora ni inferioridad, y solo cuenta en la condición 1"),
    "c_emparejada": ("la diferencia emparejada solo sobre repeticiones y pliegues COMPLETOS en los "
                     "dos motores"),
    "caso_no_previsto": ("fallos del motor nuevo Y fallos parciales de la densa v2: la decisión no "
                         "lo cubre; se aplica (c) y se marca `caso_no_previsto`. En la v2 real "
                         "ningún conjunto no sellado tiene fallos PARCIALES de la densa (probado)"),
    "d_registro": "`veredicto_por_conjunto` lleva los fallos de cada motor: cuántos y por qué",
}


def fallos_del_motor(registros: list[dict], motor: str, metric_id: str,
                     esperadas) -> list[dict]:
    """Los intentos ESPERADOS de `motor` que no traen medida, con su motivo."""
    por_clave: dict[tuple, list[dict]] = {}
    for r in registros:
        if r.get("motor") == motor:
            por_clave.setdefault((r["repeticion"], r["pliegue"]), []).append(r)
    fallos = []
    for repeticion, pliegue in sorted(tuple(k) for k in esperadas):
        rs = por_clave.get((repeticion, pliegue), [])
        base = {"repeticion": repeticion, "pliegue": pliegue}
        if not rs:
            fallos.append({**base, "tipo": "ausente", "estado": None,
                           "motivo": "el artefacto no trae este intento"})
            continue
        no_medidos = [r for r in rs
                      if r.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA]
        if no_medidos:
            r = no_medidos[0]
            fallos.append({**base, "tipo": "no_completado", "estado": r.get("estado"),
                           "motivo": r.get("motivo")})
        elif any(r.get(metric_id) is None for r in rs):
            fallos.append({**base, "tipo": "sin_metrica_de_cierre", "estado": rs[0].get("estado"),
                           "motivo": f"completado sin {metric_id}"})
    return fallos


def registros_de_los_fallos(dataset: str, motor: str, fallos: list[dict]) -> list[dict]:
    """Un registro NO medido por cada fallo, para que `aplicar_regla_de_cierre`
    —que solo ve los registros que hay— cuente como perdido también un
    intento ausente o sin métrica. Solo entran en la cuenta, nunca en
    `resultados`."""
    return [{"dataset": dataset, "motor": motor, "repeticion": f["repeticion"],
             "pliegue": f["pliegue"], "estado": f"sin_medida:{f['tipo']}"} for f in fallos]


def _sin_la_palabra_palanca(veredicto: dict) -> dict:
    renombres = {
        "n_pliegues_de_la_palanca": "n_pliegues_completos_del_motor_nuevo",
        "n_pliegues_de_la_densa_v2": "n_pliegues_completos_de_la_densa_v2",
        "pliegues_de_la_palanca_sin_intento_de_la_densa_v2":
            "pliegues_del_motor_nuevo_sin_intento_de_la_densa_v2",
    }
    salida = {renombres.get(k, k): v for k, v in veredicto.items()}
    if isinstance(salida.get("intervalo"), dict):
        salida["intervalo"] = dict(salida["intervalo"], diferencia="motor nuevo - densa v2")
    return salida


def veredicto_del_conjunto_c3(*, dataset: str, metric_id: str,
                              registros_del_motor_nuevo: list[dict],
                              registros_de_la_densa_v2: list[dict], esperadas) -> dict:
    """Condición 2 en UN conjunto, con los fallos (I2 b, c, d). La emparejada
    es la de 118 (`p118.veredicto_del_conjunto`, motor nuevo − densa v2,
    bootstrap de pliegues) sobre los pliegues completos en los DOS."""
    esperadas = sorted(tuple(k) for k in esperadas)
    en_el_plan = set(esperadas)
    fallos_nuevo = fallos_del_motor(registros_del_motor_nuevo, NOMBRE_MOTOR_NUEVO, metric_id,
                                    esperadas)
    fallos_v2 = fallos_del_motor(registros_de_la_densa_v2, NOMBRE_DENSA_V2, metric_id, esperadas)
    nuevo_por_pliegue = {k: v for k, v in c6.medida_por_pliegue(
        registros_del_motor_nuevo, NOMBRE_MOTOR_NUEVO, metric_id).items() if k in en_el_plan}
    v2_por_pliegue = {k: v for k, v in c6.medida_por_pliegue(
        registros_de_la_densa_v2, NOMBRE_DENSA_V2, metric_id).items() if k in en_el_plan}
    emparejada = _sin_la_palabra_palanca(p118.veredicto_del_conjunto(
        dataset=dataset, metric_id=metric_id, palanca_por_pliegue=nuevo_por_pliegue,
        v2_densa_por_pliegue=v2_por_pliegue))
    salida = {
        **emparejada,
        "n_intentos_esperados": len(esperadas),
        "fallos_del_motor_nuevo": {"n": len(fallos_nuevo), "detalle": fallos_nuevo},
        "fallos_de_la_densa_v2": {"n": len(fallos_v2), "detalle": fallos_v2},
        "sin_comparacion": False, "caso_no_previsto": False,
        "regla_aplicada": "c_emparejada",
        "la_emparejada_decide": True,
    }
    if esperadas and len(fallos_v2) == len(esperadas):
        salida.update(mejora=False, inferioridad=False, sin_comparacion=True,
                      regla_aplicada="b_sin_comparacion", la_emparejada_decide=False,
                      motivo=(f"la densa v2 no completó NINGUNO de sus {len(esperadas)} intentos: "
                              f"no hay comparación emparejada posible; solo cuenta en la "
                              f"condición 1"))
    elif fallos_nuevo and not fallos_v2:
        salida.update(mejora=False, inferioridad=True, regla_aplicada="b_inferioridad_por_fallos",
                      la_emparejada_decide=False,
                      motivo=(f"el motor nuevo no completó {len(fallos_nuevo)} de "
                              f"{len(esperadas)} intentos que la densa v2 sí completó: cuenta "
                              f"como INFERIORIDAD (la emparejada de abajo no decide)"))
    elif fallos_nuevo and fallos_v2:
        salida.update(caso_no_previsto=True, regla_aplicada="caso_no_previsto_c_emparejada",
                      motivo=(f"fallos en los DOS ({len(fallos_nuevo)} del nuevo, "
                              f"{len(fallos_v2)} de la densa v2): caso que la decisión no cubre; "
                              f"se aplica la emparejada sobre los completos en los dos. "
                              f"{emparejada.get('motivo') or ''}").strip())
    return salida


def _cumplidos_de(resultados: list[dict], regla: protocolo_mod.ReglaDeCierre, *, motor: str,
                  metrica_por_dataset: dict[str, str], nombres_de_los_conjuntos: list[str],
                  con_detalle: bool = False) -> dict:
    """`aplicar_regla_de_cierre`, filtrando a los conjuntos pedidos ANTES de
    llamarla (mismo motivo que `pasada_118_palanca._cumplidos`, que fija la
    densa VIEJA y aquí hace falta cualquiera de los dos motores)."""
    conjuntos = set(nombres_de_los_conjuntos)
    filtrados = [r for r in resultados if r["dataset"] in conjuntos]
    resultado = protocolo_mod.aplicar_regla_de_cierre(
        filtrados, regla, motor=motor, metrica_por_dataset=metrica_por_dataset,
        datasets_exigidos=nombres_de_los_conjuntos)
    salida = {k: resultado[k] for k in ("cumplidos", "datasets", "fraccion", "cumple_la_regla")}
    if con_detalle:
        salida["detalle"] = [{k: d.get(k) for k in ("dataset", "cumple", "perdido_por_fallo",
                                                     "mejor", "distancia_en_puntos")}
                             for d in resultado["detalle"]]
    return salida


def veredicto_de_c3(veredictos_por_conjunto: list[dict], *,
                    cumplidos_con_el_motor_nuevo: dict, cumplidos_de_la_densa_v2: dict) -> dict:
    """La regla de subida (`p118.veredicto_de_la_palanca`, importada: la
    MISMA, copiada literal en el v4) con los nombres de C3 y el recuento de
    los «sin comparación» aparte."""
    bruto = p118.veredicto_de_la_palanca(
        veredictos_por_conjunto, cumplidos_con_la_palanca=cumplidos_con_el_motor_nuevo,
        cumplidos_de_la_densa_v2=cumplidos_de_la_densa_v2)
    por = lambda clave: sorted(v["dataset"] for v in veredictos_por_conjunto  # noqa: E731
                               if v.get(clave))
    return {
        "campo": ("los motores de la v2 (pasada_v2_113_resultado.json) con sus resultados de "
                 "esa pasada, y en el sitio de la densa v2, el motor nuevo (matrixai.dense."
                 "tabm_cpu) -- SOLO en los conjuntos medidos aquí"),
        "cumplidos_con_el_motor_nuevo": bruto["cumplidos_con_la_palanca"],
        "cumplidos_de_la_densa_v2_en_los_mismos_conjuntos": bruto[
            "cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"],
        "sube_los_cumplidos": bruto["sube_los_cumplidos"],
        "n_conjuntos": bruto["n_conjuntos"],
        "n_mejora": bruto["n_mejora"], "n_inferioridad": bruto["n_inferioridad"],
        "n_sin_comparacion": sum(1 for v in veredictos_por_conjunto if v.get("sin_comparacion")),
        "conjuntos_con_mejora": por("mejora"),
        "conjuntos_con_inferioridad": por("inferioridad"),
        "conjuntos_sin_comparacion": por("sin_comparacion"),
        "conjuntos_con_fallos_del_motor_nuevo": sorted(
            v["dataset"] for v in veredictos_por_conjunto
            if (v.get("fallos_del_motor_nuevo") or {}).get("n")),
        "mejoras_superan_inferioridades": bruto["mejoras_superan_inferioridades"],
        "cumple_la_regla_de_subida": bruto["la_palanca_ayuda"],
        "regla_de_subida": (
            "protocolo_119_v4.json, veredicto.regla_de_subida: sube a confirmación (C4) si "
            "cumple las DOS condiciones sobre los 32 no sellados frente a la densa v2 en los "
            "MISMOS conjuntos -- (1) el número de conjuntos donde el motor nuevo está a <=2 "
            "puntos del mejor es MAYOR que el de la densa v2, y (2) sus mejoras superan a sus "
            "inferioridades (diferencia emparejada por repetición y pliegue, bootstrap de "
            "1000 remuestras de pliegues, semilla 0)"),
        "como_se_cuentan_los_fallos": COMO_SE_CUENTAN_LOS_FALLOS,
    }


def veredicto_final(*, resultados_v2: list[dict], resultados_del_motor_nuevo: list[dict],
                    veredictos_por_conjunto: list[dict], esperadas_por_conjunto: dict,
                    regla: protocolo_mod.ReglaDeCierre, metrica_por_dataset: dict[str, str],
                    nombres_de_los_conjuntos: list[str]) -> dict:
    """Las dos condiciones. La 1, sobre el CAMPO para el motor nuevo (la
    densa v2 sustituida por él) y sobre la v2 SIN TOCAR para la densa v2 —
    la referencia no puede medirse con el motor nuevo dentro—, con los
    intentos sin medida de cada uno contados como fallo (I2 a)."""
    campo = p118.campo_de_la_comparacion(resultados_v2, resultados_del_motor_nuevo,
                                         set(nombres_de_los_conjuntos))
    densa_v2 = [r for r in resultados_v2 if r["motor"] == NOMBRE_DENSA_V2]
    fallos_nuevo, fallos_v2 = [], []
    for nombre in nombres_de_los_conjuntos:
        esperadas, metrica = esperadas_por_conjunto[nombre], metrica_por_dataset[nombre]
        de_aqui = lambda rs: [r for r in rs if r["dataset"] == nombre]  # noqa: E731
        fallos_nuevo += registros_de_los_fallos(nombre, NOMBRE_MOTOR_NUEVO, fallos_del_motor(
            de_aqui(resultados_del_motor_nuevo), NOMBRE_MOTOR_NUEVO, metrica, esperadas))
        fallos_v2 += registros_de_los_fallos(nombre, NOMBRE_DENSA_V2, fallos_del_motor(
            de_aqui(densa_v2), NOMBRE_DENSA_V2, metrica, esperadas))
    cumplidos_nuevo = _cumplidos_de(
        campo + fallos_nuevo, regla, motor=NOMBRE_MOTOR_NUEVO,
        metrica_por_dataset=metrica_por_dataset,
        nombres_de_los_conjuntos=nombres_de_los_conjuntos, con_detalle=True)
    cumplidos_v2 = _cumplidos_de(
        resultados_v2 + fallos_v2, regla, motor=NOMBRE_DENSA_V2,
        metrica_por_dataset=metrica_por_dataset,
        nombres_de_los_conjuntos=nombres_de_los_conjuntos, con_detalle=True)
    return veredicto_de_c3(veredictos_por_conjunto, cumplidos_con_el_motor_nuevo=cumplidos_nuevo,
                           cumplidos_de_la_densa_v2=cumplidos_v2)


# ---------------------------------------------------------------------------
# 6. EL PLAN DE INTENTOS DE UN CONJUNTO, y lo que se declara de él
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class IntentoPlaneado:
    repeticion: int
    pliegue: int
    semilla: int
    presupuesto_wall_s: float
    entrena: tuple
    valida: tuple


def plan_de_intentos(ds, propuesta, protocolo, *, humo: bool) -> list[IntentoPlaneado]:
    """Los intentos de UN conjunto: repeticiones y pliegues del protocolo
    (1 y 1 con `--humo`) que la partición de verdad trae (yeast da 4 pliegues
    por repetición, no 5), cada uno con la semilla de SU repetición y el
    presupuesto de SU cubo."""
    repeticiones = 1 if humo else protocolo.particion.repeticiones_para(ds.cubo)
    folds = 1 if humo else protocolo.particion.folds
    wall = c3.wall_seconds_del_cubo(ds.cubo, protocolo)
    plan = []
    for repeticion in range(repeticiones):
        for pliegue_i in range(folds):
            pliegue = propuesta.pliegues.pliegue_de(repeticion=repeticion, pliegue=pliegue_i)
            if pliegue is None:
                continue
            plan.append(IntentoPlaneado(repeticion, pliegue_i,
                                        protocolo.particion.semillas[repeticion], wall,
                                        tuple(pliegue.entrena), tuple(pliegue.valida)))
    return plan


def allstate_medido(resultados: list[dict]) -> dict:
    """Lo que Allstate entrenó DE VERDAD, junto a lo que la enmienda 1 declara
    («≈10 épocas, menos que las 17 de la regla»)."""
    nombre = "Allstate_Claims_Severity"
    propios = [r for r in resultados if r["dataset"] == nombre]
    if not propios:
        return {"medido": False, "motivo": f"{nombre} no se ha medido en esta ejecución"}
    detalle = []
    for r in sorted(propios, key=lambda r: (r["repeticion"], r["pliegue"])):
        e = r.get("entrenamiento_efectivo") or {}
        detalle.append({"repeticion": r["repeticion"], "pliegue": r["pliegue"],
                        "estado": r["estado"], "epocas_ejecutadas": e.get("epocas_ejecutadas"),
                        "mejor_epoca": e.get("mejor_epoca"),
                        "parado_por_plazo": e.get("parado_por_plazo")})
    epocas = [d["epocas_ejecutadas"] for d in detalle if d["epocas_ejecutadas"] is not None]
    minimo = _leer_json(RUTA_DE_LA_ENMIENDA_1)["medido_que_lo_motiva"]["minimo_de_la_regla"]
    return {"medido": True, "detalle": detalle,
            "epocas_min": min(epocas) if epocas else None,
            "epocas_max": max(epocas) if epocas else None,
            "minimo_de_la_regla": minimo,
            "todos_por_debajo_del_minimo": bool(epocas) and max(epocas) < minimo}


def intentos_por_conjunto(datasets, particiones: dict, esperadas_por_conjunto: dict,
                          resultados: list[dict], protocolo) -> dict:
    """Pedidos por el protocolo, admitidos por la partición, planeados en esta
    ejecución, medidos y completos — por conjunto, CALCULADO."""
    salida = {}
    for ds in datasets:
        if ds.nombre not in particiones:
            continue
        propios = [r for r in resultados if r["dataset"] == ds.nombre]
        particion = particiones[ds.nombre]
        salida[ds.nombre] = {
            "pedidos_por_el_protocolo": (protocolo.particion.folds
                                         * protocolo.particion.repeticiones_para(ds.cubo)),
            "admitidos_por_la_particion": particion.get("n_pliegues_obtenidos"),
            "planeados_en_esta_ejecucion": len(esperadas_por_conjunto.get(ds.nombre, [])),
            "medidos": len(propios),
            "completos": sum(1 for r in propios if r.get("estado")
                             in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA),
            "limites": particion.get("limites"),
        }
    return salida


def _imprimir_estimacion_peor_caso(plan: dict) -> None:
    print(f"conjuntos: {plan['n_datasets']}, 1 motor por pliegue (matrixai.dense.tabm_cpu), "
          f"1 proceso")
    for cubo, e in sorted(plan["por_cubo"].items()):
        print(f"  cubo {cubo:8}: {e['n_datasets']:>2} datasets x {e['folds']} pliegues x "
              f"{e['repeticiones']} rep x {plan['n_motores']} motor = {e['n_intentos']:>5} "
              f"intentos, {e['wall_seconds_por_intento']:.0f}s de tope -> cota "
              f"{e['horas_cota']:.2f} h")
    print(f"  TOTAL: {plan['n_intentos']} intentos (pedidos; la partición puede dar menos)")
    print(f"  COTA de peor caso (cada intento agota su presupuesto ENTERO): "
          f"{plan['horas_de_reloj_cota_peor_caso']:.2f} h "
          f"({plan['horas_de_reloj_cota_peor_caso']/24:.2f} dias)")


# ---------------------------------------------------------------------------
# 7. LA PASADA
# ---------------------------------------------------------------------------

#: `--humo`: los DOS más baratos de los 32 no sellados por número de filas.
CONJUNTOS_DEL_HUMO = ("kc2", "dresses-sales")


def _argumentos(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forzar", action="store_true",
                        help="ignora el cache entero y re-ejecuta todos los intentos (lo que ya "
                             "hay en el fichero y no se re-mide se CONSERVA)")
    parser.add_argument("--salida", default=None,
                        help="ruta del JSON (admite una absoluta fuera del arbol). Por omision: "
                             + ", ".join(f"{t}={p.name}" for t, p in SALIDA_POR_OMISION.items()))
    parser.add_argument("--estimar", action="store_true",
                        help="NO mide la pasada: UN intento por conjunto de la lista de la "
                             "estimacion y la cuenta de si cabe en la ventana nocturna o de dia")
    parser.add_argument("--estimar-conjuntos", default=None,
                        help="con --estimar, los conjuntos que se miden (coma), en vez de "
                             "CONJUNTOS_DE_LA_ESTIMACION")
    parser.add_argument("--solo", default=None,
                        help="nombres de conjunto separados por coma, SIEMPRE dentro de los "
                             "32 no sellados (un sellado se NIEGA). Para probar el guion")
    parser.add_argument("--humo", action="store_true",
                        help=f"2 conjuntos pequeños ({', '.join(CONJUNTOS_DEL_HUMO)}), 1 "
                             f"repeticion, 1 pliegue -- prueba el camino entero, no mide nada "
                             f"del corte. Incompatible con --solo")
    args = parser.parse_args(argv)
    if args.humo and args.solo:
        raise SystemExit("--humo y --solo son incompatibles: --humo YA fija sus propios "
                         "conjuntos (CONJUNTOS_DEL_HUMO)")
    if args.estimar and (args.humo or args.solo):
        raise SystemExit("--estimar no se combina con --humo ni con --solo "
                         "(usa --estimar-conjuntos)")
    return args


def main(argv=None) -> None:
    args = _argumentos(argv)
    tipo = tipo_de_ejecucion(humo=args.humo, solo=args.solo, estimar=args.estimar)
    ruta_salida = ruta_de_salida(tipo, args.salida)

    c6.preparar_protocolo_v2()
    protocolos = exigir_los_protocolos_encadenados()
    configuracion_del_motor = exigir_la_configuracion_de_la_enmienda()
    protocolo = c3.protocolo_registrado()
    todos = c5.datasets_de_la_pasada(protocolo)
    no_sellados = c6.datasets_no_sellados(protocolo)

    if tipo == "estimar":
        cmd_estimar(protocolo, todos, no_sellados, ruta_salida=ruta_salida,
                    conjuntos=args.estimar_conjuntos, protocolos=protocolos,
                    configuracion_del_motor=configuracion_del_motor)
        return

    solo = ",".join(CONJUNTOS_DEL_HUMO) if tipo == "humo" else args.solo
    datasets, subconjunto = datasets_de_c3(todos, no_sellados, solo=solo)
    if not datasets:
        raise SystemExit("no queda ningún conjunto que medir")
    humo = tipo == "humo"

    _exigir_que_quepa()
    motor = MotorDensaTabM()
    metrica_por_dataset = c5.metrica_de_cierre_por_dataset(protocolo, datasets)
    for metric_id in set(metrica_por_dataset.values()):
        c6.direccion_de(metric_id)  # PARA si alguna metrica no tiene direccion declarada

    payload_v2 = _payload_v2()
    resultados_v2 = payload_v2["resultados"]
    sin_particion_v2 = [d.nombre for d in datasets
                        if d.nombre not in (payload_v2.get("particion_por_dataset") or {})]
    if sin_particion_v2:
        raise SystemExit(f"la v2 no registra la partición de {sin_particion_v2}: no hay con qué "
                         f"comparar. No se mide")

    plan = c6.plan_de_la_pasada(datasets, protocolo, n_motores=1)
    print(f"protocolo {protocolo.version_protocolo}, digest {protocolo.digest()[:16]} "
          f"(v4 {protocolos['v4']['digest_sha256_declarado'][:16]}, enmienda 1 "
          f"{protocolos['enmienda_1']['digest_sha256_declarado'][:16]})")
    print(f"motor {NOMBRE_MOTOR_NUEVO}: {configuracion_del_motor['leida_del_motor']} "
          f"(la de la enmienda 1). Salida ({tipo}): {ruta_salida}")
    _imprimir_estimacion_peor_caso(plan)

    cache_previo, payload_previo = cargar_salida_previa(ruta_salida, tipo)
    if args.forzar:
        cache_previo = {}
    previos = list(payload_previo.get("resultados") or [])

    # UNA vez, al arrancar: los registros llevan ESTE digest, y el resultado
    # tiene que describir el mismo aunque un fichero cambie a media pasada.
    componentes_del_digest = componentes_del_digest_del_entorno()
    entorno_digest = _digest_de_componentes(componentes_del_digest)
    digest_del_motor = _digest_motor_nuevo()
    procedencia = c3.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": entorno_digest, "motor_nuevo": digest_del_motor},
        datos_de_entrada={
            **{d.nombre: c3.ARFF_DIR / f"{d.data_id}.arff" for d in datasets},
            "protocolo_exploratorio_v2": RUTA_DEL_PROTOCOLO_V2,
            "protocolo_119_v4": RUTA_DEL_PROTOCOLO_V4,
            "protocolo_119_v4_enmienda_1": RUTA_DE_LA_ENMIENDA_1,
            "pasada_v2_113_resultado": RUTA_V2_RESULTADO,
        })
    for aviso in procedencia["avisos"]:
        print(f"AVISO DE PROCEDENCIA: {aviso}", flush=True)
    if payload_previo:
        declarada = c3.procedencia_declarada(payload_previo)
        print(f"fichero previo: {len(previos)} registros, procedencia "
              f"{declarada['estado']} -- {declarada['explicacion']}", flush=True)
    enmienda_2 = enmienda_2_declarada()
    if not enmienda_2["presente"]:
        print(f"AVISO: {enmienda_2['aviso']}", flush=True)

    resultados: list[dict] = []
    veredictos_por_conjunto: list[dict] = []
    particiones: dict[str, dict] = {}
    comparacion_de_particiones: dict[str, dict] = {}
    pliegues_digest: dict[str, str] = {}
    tiempos_del_padre: dict[str, dict] = {}
    esperadas_por_conjunto: dict[str, list] = {}
    reusados = [0]
    inicio = time.perf_counter()
    regla = protocolo.regla_de_cierre
    nombres_de_los_conjuntos = [d.nombre for d in datasets]
    ultimo_punto_de_control = [time.perf_counter()]

    def guardar(parcial: bool) -> dict:
        ultimo_punto_de_control[0] = time.perf_counter()
        veredicto = None
        if not parcial:
            veredicto = veredicto_final(
                resultados_v2=resultados_v2, resultados_del_motor_nuevo=resultados,
                veredictos_por_conjunto=veredictos_por_conjunto,
                esperadas_por_conjunto=esperadas_por_conjunto, regla=regla,
                metrica_por_dataset=metrica_por_dataset,
                nombres_de_los_conjuntos=nombres_de_los_conjuntos)
        return _componer_y_guardar(
            resultados, previos, veredictos_por_conjunto, veredicto, procedencia, payload_previo,
            ruta_salida, tipo=tipo, datasets=datasets, protocolo=protocolo,
            metrica_por_dataset=metrica_por_dataset, subconjunto=subconjunto, plan=plan,
            particiones=particiones, comparacion_de_particiones=comparacion_de_particiones,
            pliegues_digest=pliegues_digest, tiempos_del_padre=tiempos_del_padre,
            esperadas_por_conjunto=esperadas_por_conjunto, protocolos=protocolos,
            configuracion_del_motor=configuracion_del_motor, enmienda_2=enmienda_2,
            componentes_del_digest=componentes_del_digest,
            total_wall_s=time.perf_counter() - inicio, reusados=reusados[0], parcial=parcial)

    for ds in datasets:
        t_carga = time.perf_counter()
        por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(
            ds, protocolo)
        carga_s = time.perf_counter() - t_carga
        comparacion_de_particiones[ds.nombre] = exigir_la_particion_de_la_v2(
            ds, particion, c3.LECTURA_DECLARADA.get(str(ds.data_id)), payload_v2)
        particiones[ds.nombre] = _normalizado(particion)
        pliegues_digest[ds.nombre] = propuesta.pliegues.digest()
        test_ids = propuesta.plan.observaciones_del_rol("test")
        metric_id = metrica_por_dataset[ds.nombre]
        datos_sha256 = (procedencia["datos_de_entrada"].get(ds.nombre) or {}).get("sha256")
        intentos = plan_de_intentos(ds, propuesta, protocolo, humo=humo)
        esperadas_por_conjunto[ds.nombre] = [(i.repeticion, i.pliegue) for i in intentos]
        tiempos = tiempos_del_padre[ds.nombre] = {"carga_s": round(carga_s, 3),
                                                  "preparacion_s": 0.0, "n_preparaciones": 0}
        print(f"\n=== {ds.nombre} (data_id={ds.data_id}, {ds.tarea}, cubo={ds.cubo}, "
              f"n={len(por_id)}, test={len(test_ids)}, intentos={len(intentos)}, "
              f"metrica={metric_id}, carga={carga_s:.1f}s) ===", flush=True)

        registros_del_dataset: list[dict] = []
        for repeticion in sorted({i.repeticion for i in intentos}):
            for ip in [i for i in intentos if i.repeticion == repeticion]:
                clave = (ds.nombre, motor.nombre, ip.repeticion, ip.pliegue)
                previo = cache_previo.get(clave)
                if not humo and _reusable_c3(previo, entorno_digest, digest_del_motor,
                                             ip.presupuesto_wall_s, datos_sha256=datos_sha256):
                    registro = dict(previo, reusado=True)
                    registro.setdefault("procedencia_id", None)
                    resultados.append(registro)
                    registros_del_dataset.append(registro)
                    reusados[0] += 1
                    print(f"  [reusado] {ds.nombre} rep={ip.repeticion} pliegue={ip.pliegue}: "
                          f"estado={registro.get('estado')}", flush=True)
                    continue

                t_prep = time.perf_counter()
                crudas_train = [por_id[i] for i in ip.entrena]
                crudas_val = [por_id[i] for i in ip.valida]
                crudas_test = [por_id[i] for i in test_ids]
                transformadas = c3.preparar_para_motor(
                    crudas_train, crudas_train + crudas_val + crudas_test,
                    objetivo, predictores, motor)
                n_tr, n_va = len(crudas_train), len(crudas_val)

                def hacer(xs):
                    return Particion.desde_filas(xs, row_id_field="row_id", target_field=objetivo)

                train, val, test = (hacer(transformadas[:n_tr]),
                                    hacer(transformadas[n_tr:n_tr + n_va]),
                                    hacer(transformadas[n_tr + n_va:]))
                preparacion_s = time.perf_counter() - t_prep
                tiempos["preparacion_s"] = round(tiempos["preparacion_s"] + preparacion_s, 3)
                tiempos["n_preparaciones"] += 1

                presupuesto = Presupuesto(wall_seconds=ip.presupuesto_wall_s,
                                          hilos=c3.HILOS_POR_INTENTO, seed=ip.semilla)
                t0 = time.perf_counter()
                intento = ejecutar_intento_aislado(
                    motor, train, val, test, spec, presupuesto,
                    candidate=f"{motor.nombre}-{c3.CONFIGURACION_UNICA}",
                    split_plan_digest=propuesta.plan.digest(),
                    dataset=ds.nombre, pliegue=ip.pliegue, repeticion=ip.repeticion)
                transcurrido = time.perf_counter() - t0

                metricas = c5.metricas_del_informe(intento.informe)
                recursos = intento.recursos or {}
                # Lo que el motor DECLARA que pasó (de `config_efectiva`),
                # NUNCA los pesos: decenas de MB por intento.
                config = intento.config_efectiva or {}
                registro = {
                    "dataset": ds.nombre, "data_id": ds.data_id, "cubo": ds.cubo,
                    "tarea": ds.tarea, "sellado": ds.sellado, "motor": motor.nombre,
                    "repeticion": ip.repeticion, "pliegue": ip.pliegue, "estado": intento.estado,
                    "semilla": ip.semilla, "presupuesto_wall_s": ip.presupuesto_wall_s,
                    "configuracion": c3.CONFIGURACION_UNICA, "metrica_de_cierre": metric_id,
                    "wall_s": round(transcurrido, 3), "preparacion_s": round(preparacion_s, 3),
                    "metricas": metricas, "tiempo_de_ajuste": recursos.get("wall_seconds"),
                    "cpu_segundos": recursos.get("cpu_seconds"),
                    "rss_pico_mb": recursos.get("peak_ram_mb"),
                    "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
                    "arquitectura": config.get("arquitectura"),
                    "hiperparametros": config.get("hiperparametros"),
                    "motivo": (intento.motivo_del_estado["es"]
                               if intento.motivo_del_estado else None),
                    "traza": intento.traza, "engine_version": intento.engine_version,
                    "pipeline_digest": intento.pipeline_digest,
                    "split_plan_digest": propuesta.plan.digest(),
                    "entorno_digest": entorno_digest, "motor_digest": digest_del_motor,
                    "datos_sha256": datos_sha256,
                    "procedencia_id": procedencia["procedencia_id"], "reusado": False,
                }
                c5.aplanar_metricas_en_el_registro(registro, metricas)
                resultados.append(registro)
                registros_del_dataset.append(registro)
                try:
                    exigir_la_arquitectura_del_intento(registro)
                except SystemExit:
                    guardar(parcial=True)
                    raise
                print(f"  {ds.nombre} rep={ip.repeticion} pliegue={ip.pliegue}: "
                      f"estado={intento.estado} {metric_id}={registro.get(metric_id)} "
                      f"wall={transcurrido:.1f}s prep={preparacion_s:.1f}s "
                      f"entrenamiento_efectivo={registro['entrenamiento_efectivo']}", flush=True)
                if (time.perf_counter() - ultimo_punto_de_control[0]
                        >= SEGUNDOS_ENTRE_PUNTOS_DE_CONTROL):
                    guardar(parcial=True)
            if not humo:
                guardar(parcial=True)

        v2_densa_del_dataset = [r for r in resultados_v2
                                if r["dataset"] == ds.nombre and r["motor"] == NOMBRE_DENSA_V2]
        veredicto_ds = veredicto_del_conjunto_c3(
            dataset=ds.nombre, metric_id=metric_id,
            registros_del_motor_nuevo=registros_del_dataset,
            registros_de_la_densa_v2=v2_densa_del_dataset,
            esperadas=esperadas_por_conjunto[ds.nombre])
        veredictos_por_conjunto.append(veredicto_ds)
        print(f"  VEREDICTO {ds.nombre}: mejora={veredicto_ds['mejora']} "
              f"inferioridad={veredicto_ds['inferioridad']} "
              f"sin_comparacion={veredicto_ds['sin_comparacion']} "
              f"fallos_del_nuevo={veredicto_ds['fallos_del_motor_nuevo']['n']} "
              f"motivo={veredicto_ds['motivo']}", flush=True)
        guardar(parcial=True)

    total = time.perf_counter() - inicio
    print(f"\n=== total: {total:.1f}s ({total/3600:.2f} h), {len(resultados)} intentos "
          f"({reusados[0]} reusados, {len(resultados)-reusados[0]} ejecutados) ===")
    salida = guardar(parcial=False)
    print(f"Guardado en {ruta_salida}, digest={salida['digest_resultados_crudos'][:16]}")
    v = salida["veredicto"]
    print(f"VEREDICTO 119-C3{' (' + tipo.upper() + ' -- no cuenta)' if tipo != 'pasada' else ''}: "
          f"cumplidos {v['cumplidos_con_el_motor_nuevo']['cumplidos']} vs densa v2 "
          f"{v['cumplidos_de_la_densa_v2_en_los_mismos_conjuntos']['cumplidos']} "
          f"(sube={v['sube_los_cumplidos']}); mejora={v['n_mejora']} "
          f"inferioridad={v['n_inferioridad']} sin_comparacion={v['n_sin_comparacion']} -> "
          f"{'SUBE a confirmacion' if v['cumple_la_regla_de_subida'] else 'NO SUBE'}")


def _digest_fichero_json(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _escribir_atomicamente(ruta: Path, salida: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".parcial")
    temporal.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(ruta)


def _componer_y_guardar(resultados, previos, veredictos_por_conjunto, veredicto, procedencia,
                        payload_previo, ruta_salida, *, tipo, datasets, protocolo,
                        metrica_por_dataset, subconjunto, plan, particiones,
                        comparacion_de_particiones, pliegues_digest, tiempos_del_padre,
                        esperadas_por_conjunto, protocolos, configuracion_del_motor, enmienda_2,
                        componentes_del_digest, total_wall_s, reusados, parcial) -> dict:
    """Compone el JSON y lo escribe atómicamente, FUSIONADO con lo que ya
    había (I7): lo previo que esta ejecución no ha vuelto a tocar se
    conserva, con su procedencia."""
    en_el_fichero, conservados = fusionar_resultados(previos, resultados)
    procedencias, sin_procedencia = c3._procedencias_citadas(
        en_el_fichero, procedencia, (payload_previo.get("procedencias") or {}))
    enmienda_1 = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    reconciliacion = (
        {"no_aplica": "--humo corre 1 repetición y 1 pliegue por conjunto: el plan del "
                      "protocolo no es el de esta ejecución (ver intentos_por_conjunto)"}
        if tipo == "humo" else
        c5.reconciliar_el_plan_con_lo_medido(plan, particiones, resultados, n_motores=1))
    salida = {
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corte": "119-C3",
        "tipo_de_ejecucion": tipo,
        "motor": NOMBRE_MOTOR_NUEVO,
        "configuracion_del_motor": configuracion_del_motor,
        "protocolo_119_v4_digest_sha256": _digest_fichero_json(RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": _digest_fichero_json(RUTA_DE_LA_ENMIENDA_1),
        "protocolos_encadenados": protocolos,
        "enmienda_2": enmienda_2,
        "allstate_declarado": enmienda_1.get("allstate_declarado"),
        "allstate_medido": allstate_medido(resultados),
        "procedencia": procedencia, "procedencias": procedencias,
        "n_intentos_sin_procedencia": sin_procedencia,
        "parcial": parcial, "es_humo": tipo == "humo",
        "es_subconjunto_de_prueba": subconjunto is not None,
        "subconjunto_pedido": subconjunto,
        "protocolo_version": protocolo.version_protocolo,
        "protocolo_digest_sha256": protocolo.digest(),
        "plan": plan,
        "intentos_por_conjunto": intentos_por_conjunto(
            datasets, particiones, esperadas_por_conjunto, resultados, protocolo),
        "por_que_n_intentos_no_es_el_del_plan": reconciliacion,
        "particion_por_dataset": dict(particiones),
        "particion_comparada_con_la_v2": dict(comparacion_de_particiones),
        "pliegues_digest_por_dataset": dict(pliegues_digest),
        "pliegues_digest_que_es": ("digest de la asignación de pliegues (entrena/valida por "
                                   "repetición): la v2 no lo registró, así que con la v2 solo se "
                                   "compara `particion_por_dataset` (plan_digest incluido)"),
        "tiempos_del_padre_por_dataset": dict(tiempos_del_padre),
        "tiempos_del_padre_que_son": ("carga_s = leer el ARFF y particionar (una vez por "
                                      "conjunto); preparacion_s = preparar_para_motor y construir "
                                      "las particiones de los intentos EJECUTADOS. Ninguno está "
                                      "dentro de wall_s, que es solo ejecutar_intento_aislado"),
        "criterio_de_los_conjuntos": (
            "los 32 NO sellados del protocolo v4 (idénticos en particion/presupuesto/"
            "regla_de_cierre/datasets a protocolo_exploratorio_v2.json, comprobado al arrancar), "
            "o el subconjunto de --solo o --humo, declarado arriba. Los 8 sellados se niegan"),
        "datasets_declarados": [d.a_json() for d in datasets],
        "metrica_de_cierre_por_dataset": dict(metrica_por_dataset),
        "total_wall_s": round(total_wall_s, 1),
        "n_intentos": len(resultados), "n_reusados": reusados,
        "n_intentos_en_el_fichero": len(en_el_fichero),
        "n_registros_conservados_de_ejecuciones_anteriores": len(conservados),
        "registros_conservados_por_dataset": {
            d: sum(1 for r in conservados if r["dataset"] == d)
            for d in sorted({r["dataset"] for r in conservados})},
        "que_cubren_el_veredicto_y_n_intentos": (
            "SOLO los conjuntos de esta ejecución (datasets_declarados). `resultados` es el "
            "fichero fusionado: incluye los registros conservados de ejecuciones anteriores, "
            "que sirven de caché y no entran en el veredicto"),
        "digest_de_la_cache": {
            "entorno": procedencia["digests_de_codigo"].get("entorno"),
            "componentes": dict(componentes_del_digest),
            "motor_nuevo": {"digest": procedencia["digests_de_codigo"].get("motor_nuevo"),
                            "ficheros": [_etiqueta(f) for f in _FICHEROS_DEL_MOTOR_NUEVO]},
            "clave_de_cada_registro": ("entorno_digest, motor_digest, presupuesto_wall_s, "
                                       "estado que cuente como medida y datos_sha256 del ARFF "
                                       "(_reusable_c3)"),
        },
        "lo_que_no_cubre_el_digest": lo_que_no_cubre_el_digest(),
        "lectura_de_los_datos": dict(c3.LECTURA_DECLARADA),
        "resultados": en_el_fichero,
        "veredicto_por_conjunto": veredictos_por_conjunto,
        "veredicto": veredicto,
    }
    c3.sellar_la_salida(salida)
    _escribir_atomicamente(ruta_salida, salida)
    return salida


# ---------------------------------------------------------------------------
# 8. --estimar (I1): UN intento real por conjunto de una lista con más de un
#    conjunto por cubo, la preparación del padre aparte, y si cabe en la
#    ventana nocturna o de día. 1 proceso: el guion es secuencial.
# ---------------------------------------------------------------------------

#: Más de uno por cubo, y los ANCHOS, que son los caros para TabM+PLR (cada
#: numérica es un embedding PLR): pequeño fino / 216 / 1300 columnas;
#: mediano fino / más filas / 1558 columnas; grande fino / el más caro, que
#: es el que la enmienda 1 declara que no llega a las 17 épocas.
CONJUNTOS_DE_LA_ESTIMACION = (
    "balance-scale", "mfeat-factors", "micro-mass",
    "wilt", "pendigits", "Internet-Advertisements",
    "house_16H", "Allstate_Claims_Severity",
)

#: La ventana de la cola nocturna: cron a las 00:30, la suite nocturna
#: (~1 h 45) primero, y ningún trabajo empieza después de las 08:00.
VENTANA_NOCTURNA = ("01:45", "08:00")
#: De día (`COLA_HASTA=23`): ningún trabajo cuya estimación pase de las 23:00.
HORA_LIMITE_DE_DIA = "23:00"
HORA_DE_INICIO_DE_DIA = "08:00"


def _segundos_entre(desde: str, hasta: str) -> int:
    h1, m1 = map(int, desde.split(":"))
    h2, m2 = map(int, hasta.split(":"))
    return ((h2 * 60 + m2) - (h1 * 60 + m1)) * 60


def _interpolar(x: float, puntos: list[tuple[float, float]]) -> float | None:
    """Lineal entre los dos medidos que lo rodean; por debajo del menor, el
    menor; por ENCIMA del mayor, `None` (no se extrapola: se usa la cota)."""
    puntos = sorted(puntos)
    if not puntos:
        return None
    if x <= puntos[0][0]:
        return puntos[0][1]
    for (x0, y0), (x1, y1) in zip(puntos, puntos[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def _escalar_por_celdas(x: float, puntos: list[tuple[float, float]]) -> float:
    """Proporcional a las celdas desde el medido más cercano en escala
    logarítmica (cargar y preparar escalan con el tamaño, no con el cubo)."""
    if not puntos:
        return 0.0
    cercano = min(puntos, key=lambda p: abs(math.log(max(p[0], 1)) - math.log(max(x, 1))))
    return cercano[1] * x / max(cercano[0], 1)


def estimar_desde_medidas(medidas: dict[str, dict], conjuntos: list[dict], *, margen_s: float,
                          ventana_nocturna_s: int, ventana_de_dia_s: int,
                          ventana_de_dia_desde_ahora_s: int | None = None) -> dict:
    """La cuenta, PURA (se prueba con medidas fabricadas).

    `conjuntos`: [{nombre, cubo, celdas, n_intentos, presupuesto_wall_s}].
    `medidas`: {nombre: {cubo, celdas, wall_s, preparacion_s, carga_s, estado}}.
    Por conjunto: carga + n_intentos × (preparación + intento). El intento, el
    medido si se midió; si no, interpolado por celdas entre los medidos de
    SU cubo, o la cota (tope + margen) por encima del mayor. Nunca más que
    la cota: el subproceso mata el intento ahí."""
    por_cubo: dict[str, list] = {}
    for nombre, m in medidas.items():
        por_cubo.setdefault(m["cubo"], []).append((m["celdas"], m["wall_s"]))
    cargas = [(m["celdas"], m["carga_s"]) for m in medidas.values()]
    preparaciones = [(m["celdas"], m["preparacion_s"]) for m in medidas.values()]
    detalle, total_s, cota_s = [], 0.0, 0.0
    for c in conjuntos:
        techo = c["presupuesto_wall_s"] + margen_s
        if c["nombre"] in medidas:
            m = medidas[c["nombre"]]
            intento, carga, prep, fuente = m["wall_s"], m["carga_s"], m["preparacion_s"], "medido"
        else:
            intento = _interpolar(c["celdas"], por_cubo.get(c["cubo"], []))
            fuente = "interpolado por celdas en su cubo"
            if intento is None:
                intento, fuente = techo, "cota: más celdas que cualquier medido de su cubo"
            carga = _escalar_por_celdas(c["celdas"], cargas)
            prep = _escalar_por_celdas(c["celdas"], preparaciones)
        intento = min(intento, techo)
        segundos = carga + c["n_intentos"] * (prep + intento)
        cota = carga + c["n_intentos"] * (prep + techo)
        total_s += segundos
        cota_s += cota
        detalle.append({**c, "s_por_intento": round(intento, 1), "carga_s": round(carga, 1),
                        "preparacion_s_por_intento": round(prep, 2), "fuente": fuente,
                        "horas": round(segundos / 3600, 3), "horas_cota": round(cota / 3600, 3)})
    fallidos = sorted(n for n, m in medidas.items()
                      if m.get("estado") not in protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA)
    salida = {
        "procesos": 1,
        "por_que_1_proceso": "el guion corre los intentos de uno en uno; no hay otra cifra medida",
        "por_conjunto": detalle,
        "total_horas": round(total_s / 3600, 2),
        "total_horas_cota": round(cota_s / 3600, 2),
        "ventana_nocturna": {"desde": VENTANA_NOCTURNA[0], "hasta": VENTANA_NOCTURNA[1],
                             "segundos": ventana_nocturna_s,
                             "cabe": total_s <= ventana_nocturna_s,
                             "noches_necesarias": max(1, math.ceil(total_s / ventana_nocturna_s)),
                             "cabe_la_cota": cota_s <= ventana_nocturna_s},
        "de_dia": {"desde": HORA_DE_INICIO_DE_DIA, "hasta": HORA_LIMITE_DE_DIA,
                   "segundos": ventana_de_dia_s, "cabe": total_s <= ventana_de_dia_s,
                   "como": "COLA_SIN_SUITE=1 COLA_HASTA=23 ~/cola-nocturna.sh (CLAUDE.md)"},
        "intentos_que_fallaron_al_estimar": fallidos,
        "que_supone": (
            "un conjunto no medido tarda por intento lo que dicen, interpolados por celdas "
            "(filas x predictores), los medidos de SU cubo; cargar y preparar escalan con las "
            "celdas desde el medido más cercano. Las épocas dependen de la paciencia y de los "
            "datos, que las celdas no ven: es una estimación, no una medición"),
    }
    if ventana_de_dia_desde_ahora_s is not None:
        salida["de_dia"]["segundos_desde_ahora"] = ventana_de_dia_desde_ahora_s
        salida["de_dia"]["cabe_desde_ahora"] = total_s <= ventana_de_dia_desde_ahora_s
    if fallidos:
        salida["aviso_de_fallos"] = (
            f"{fallidos} FALLARON al estimar: en la pasada, cada intento así es un fallo del "
            f"motor nuevo y su conjunto cuenta como perdido (condición 1) e inferioridad "
            f"(condición 2)")
    return salida


def _medir_un_intento_real(ds, protocolo, motor, payload_v2) -> dict:
    """UN intento real (repetición 0, pliegue 0) con el presupuesto REAL de su
    cubo y 4 hilos, con la carga y la preparación del padre medidas aparte."""
    t_carga = time.perf_counter()
    por_id, propuesta, spec, objetivo, predictores, particion = c5.particiones_base(ds, protocolo)
    carga_s = time.perf_counter() - t_carga
    exigir_la_particion_de_la_v2(ds, particion, c3.LECTURA_DECLARADA.get(str(ds.data_id)),
                                 payload_v2)
    intentos = plan_de_intentos(ds, propuesta, protocolo, humo=True)
    if not intentos:
        raise SystemExit(f"{ds.nombre}: no hay pliegue (0, 0) en su partición")
    ip = intentos[0]
    t_prep = time.perf_counter()
    test_ids = propuesta.plan.observaciones_del_rol("test")
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
    presupuesto = Presupuesto(wall_seconds=ip.presupuesto_wall_s, hilos=c3.HILOS_POR_INTENTO,
                              seed=ip.semilla)
    t0 = time.perf_counter()
    intento = ejecutar_intento_aislado(
        motor, train, val, test, spec, presupuesto,
        candidate=f"{motor.nombre}-estimacion-119-c3", split_plan_digest=propuesta.plan.digest(),
        dataset=ds.nombre, pliegue=ip.pliegue, repeticion=ip.repeticion)
    wall_s = time.perf_counter() - t0
    config = intento.config_efectiva or {}
    recursos = intento.recursos or {}
    return {
        "dataset": ds.nombre, "cubo": ds.cubo, "n_filas": particion["n_filas_con_objetivo"],
        "n_predictores": particion["n_predictores"],
        "celdas": particion["n_filas_con_objetivo"] * particion["n_predictores"],
        "presupuesto_wall_s": ip.presupuesto_wall_s, "hilos": c3.HILOS_POR_INTENTO,
        "estado": intento.estado, "motivo": (intento.motivo_del_estado or {}).get("es"),
        "wall_s": round(wall_s, 3), "carga_s": round(carga_s, 3),
        "preparacion_s": round(preparacion_s, 3),
        "entrenamiento_efectivo": config.get("entrenamiento_efectivo"),
        "arquitectura": config.get("arquitectura"),
        "rss_pico_mb_del_hijo": recursos.get("peak_ram_mb"),
        "maxrss_mb_del_padre": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "maxrss_mb_de_los_hijos": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
    }


def _conjuntos_para_estimar(no_sellados, protocolo, payload_v2) -> list[dict]:
    particiones_v2 = payload_v2.get("particion_por_dataset") or {}
    salida = []
    for d in no_sellados:
        p = particiones_v2.get(d.nombre)
        if p is None:
            raise SystemExit(f"{d.nombre}: la v2 no registra su partición; no se puede estimar")
        salida.append({"nombre": d.nombre, "cubo": d.cubo,
                       "celdas": p["n_filas_con_objetivo"] * p["n_predictores"],
                       "n_intentos": p["n_pliegues_obtenidos"],
                       "presupuesto_wall_s": c3.wall_seconds_del_cubo(d.cubo, protocolo)})
    return salida


def _segundos_hasta(hora: str, ahora: time.struct_time) -> int:
    h, m = map(int, hora.split(":"))
    return max(0, (h * 3600 + m * 60) - (ahora.tm_hour * 3600 + ahora.tm_min * 60 + ahora.tm_sec))


def cmd_estimar(protocolo, todos, no_sellados, *, ruta_salida: Path, conjuntos: str | None,
                protocolos: dict, configuracion_del_motor: dict) -> None:
    _exigir_que_quepa()
    if ruta_salida.exists() and tipo_del_fichero(_leer_json(ruta_salida)) != "estimar":
        raise SystemExit(f"{ruta_salida} no es una estimación: --estimar no lo pisa")
    motor = MotorDensaTabM()
    payload_v2 = _payload_v2()
    plan = c6.plan_de_la_pasada(no_sellados, protocolo, n_motores=1)
    _imprimir_estimacion_peor_caso(plan)
    nombres = ([n.strip() for n in conjuntos.split(",") if n.strip()] if conjuntos
               else list(CONJUNTOS_DE_LA_ESTIMACION))
    medir, _ = datasets_de_c3(todos, no_sellados, solo=",".join(nombres))
    sin_medida = sorted({d.cubo for d in no_sellados} - {d.cubo for d in medir})
    if sin_medida:
        print(f"AVISO: ningún conjunto medido en el cubo {sin_medida}: sus conjuntos irán a la "
              f"cota", flush=True)
    peor = sum(c3.wall_seconds_del_cubo(d.cubo, protocolo) + MARGEN_POR_DEFECTO_SEGUNDOS
               for d in medir)
    print(f"\nESTA MEDICIÓN: {len(medir)} intentos reales ({', '.join(d.nombre for d in medir)}), "
          f"1 proceso, 4 hilos; peor caso {peor/60:.0f} min sin contar la carga (Allstate: ~1,5 "
          f"min más). Techo de memoria: el del lanzador (systemd-run), no este guion",
          flush=True)
    medidas = {}
    for ds in medir:
        print(f"  midiendo {ds.nombre} ({ds.cubo})...", flush=True)
        m = medidas[ds.nombre] = _medir_un_intento_real(ds, protocolo, motor, payload_v2)
        print(f"  {ds.nombre}: intento {m['wall_s']:.1f}s, carga {m['carga_s']:.1f}s, prep "
              f"{m['preparacion_s']:.1f}s, estado={m['estado']}, "
              f"entrenamiento_efectivo={m['entrenamiento_efectivo']}", flush=True)
    ahora = time.localtime()
    estimacion = estimar_desde_medidas(
        medidas, _conjuntos_para_estimar(no_sellados, protocolo, payload_v2),
        margen_s=MARGEN_POR_DEFECTO_SEGUNDOS,
        ventana_nocturna_s=_segundos_entre(*VENTANA_NOCTURNA),
        ventana_de_dia_s=_segundos_entre(HORA_DE_INICIO_DE_DIA, HORA_LIMITE_DE_DIA),
        ventana_de_dia_desde_ahora_s=_segundos_hasta(HORA_LIMITE_DE_DIA, ahora))
    pico_mb = max([(m["maxrss_mb_del_padre"] or 0) + max(m["rss_pico_mb_del_hijo"] or 0,
                                                          m["maxrss_mb_de_los_hijos"] or 0)
                   for m in medidas.values()] or [0])
    ventana = estimacion["ventana_nocturna"]["segundos"]
    enmienda_1 = _leer_json(RUTA_DE_LA_ENMIENDA_1)
    salida = {
        "corte": "119-C3", "sub_corte": "estimacion (--estimar: UN intento por conjunto medido)",
        "tipo_de_ejecucion": "estimar",
        "creado": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motor": NOMBRE_MOTOR_NUEVO, "configuracion_del_motor": configuracion_del_motor,
        "protocolos_encadenados": protocolos,
        "protocolo_119_v4_digest_sha256": _digest_fichero_json(RUTA_DEL_PROTOCOLO_V4),
        "protocolo_119_v4_enmienda_1_digest_sha256": _digest_fichero_json(RUTA_DE_LA_ENMIENDA_1),
        "digest_del_entorno": _digest_entorno(), "digest_del_motor": _digest_motor_nuevo(),
        "plan_peor_caso": plan,
        "conjuntos_medidos": [d.nombre for d in medir],
        "cubos_sin_medida": sin_medida,
        "medidas": medidas,
        "estimacion": estimacion,
        "sonda_de_coste_119_c1": enmienda_1["medido_que_lo_motiva"],
        "allstate": {"declarado_por_la_enmienda_1": enmienda_1.get("allstate_declarado"),
                     "medido_aqui": (medidas.get("Allstate_Claims_Severity") or {}).get(
                         "entrenamiento_efectivo")},
        "memoria": {"pico_medido_mb_padre_mas_hijo": pico_mb,
                    "memory_max_sugerido": f"{max(4, math.ceil(pico_mb * 1.3 / 1024))}G"},
        "para_encolar": {
            "tope_s": ventana, "estimada_s": int(min(estimacion["total_horas"] * 3600, ventana)),
            "noches": estimacion["ventana_nocturna"]["noches_necesarias"],
            "orden": ("ESTIMADA_S=<estimada_s> [COMMITS=matrixAI=<sha>,matrixai-engines=<sha>] "
                      "~/encolar.sh 119-c3 <tope_s> <memory_max> matrixAI python3 "
                      "benchmarks/fase0/pasada_119_c3.py --salida "
                      "/home/deployer/cola-nocturna/resultados/119-c3/pasada_119_c3_resultado.json"),
            "por_que_la_salida_fuera_del_arbol": (
                "la cola borra sus worktrees al terminar: con la salida dentro, la noche siguiente "
                "no encontraría el caché y lo mediría todo otra vez"),
            "por_que_COMMITS": ("las noches de continuación con el MISMO commit: con otro, el "
                                "digest del entorno puede cambiar e invalidar lo ya medido"),
        },
    }
    _escribir_atomicamente(ruta_salida, salida)
    print(f"\nestimación escrita en {ruta_salida}")
    print(f"  1 proceso: {estimacion['total_horas']:.2f} h (cota {estimacion['total_horas_cota']:.2f}"
          f" h); ventana nocturna {ventana/3600:.2f} h -> "
          f"{'CABE' if estimacion['ventana_nocturna']['cabe'] else 'NO CABE'} "
          f"({estimacion['ventana_nocturna']['noches_necesarias']} noche/s); de día "
          f"{'cabe' if estimacion['de_dia']['cabe'] else 'no cabe'}")


if __name__ == "__main__":
    main()
