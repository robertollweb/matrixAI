# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""M12 — Límites operativos configurables del playground.

Los topes (filas, épocas, tamaño CSV, profundidad, nº de clases) protegen el playground
HOSTED compartido (anti-DoS). El Studio DESCARGABLE corre en la máquina del usuario, así
que debe poder subirlos o quitarlos (su máquina, su responsabilidad) — misma filosofía que
`MATRIXAI_TRAIN_TIMEOUT=0` (ver project_training_limits).

Resolución por límite, en este orden:
  1. `MATRIXAI_HOSTED=1` → topes DUROS de `_HOSTED`; se ignora cualquier
     override (anti-DoS). "Sin límite" NO existe en hosted.
  2. `sin_topes()` activo → sin tope (BANCO DE PRUEBAS, ver abajo). Va por
     delante del env y del perfil a propósito: una medición de laboratorio no
     puede depender de cómo tenga configurada su máquina quien la lanza.
  3. Override por-límite por env (`MATRIXAI_MAX_ROWS`, `MATRIXAI_MAX_EPOCHS`, ...): un
     entero positivo (tope) o `0`/`none`/`unlimited` (sin tope).
  4. Perfil `MATRIXAI_LIMITS_PROFILE` = equilibrado (default) | avanzado | ilimitado.
  5. Default = equilibrado.

Un límite `None` significa "sin tope" (el código lo trata como ilimitado). `_MIN_RELU_WIDTH`
NO se gestiona aquí: es una corrección de sanidad (la ReLU muere por debajo de 16), no un
tope de capacidad.

EL TAMAÑO DE LOS DATOS DEJÓ DE SER UNA DECISIÓN DE PRODUCTO (2026-09-12,
decisión de Roberto: «el límite de 50 MB es absurdo, estamos en el año 2026,
no debe haber ningún tipo de límite, ya el Studio lo contempla en su
configuración»). `max_csv_bytes` (50 MB) y `max_rows` (50.000) eran números
heredados que en una máquina de hoy no protegen de nada, y el Studio
descargable YA declaraba `max_csv_bytes: None` en sus capacidades
(`endpoints.py`, perfil completo) mientras el core seguía rechazando a los
50 MB: media verdad. Ahora los perfiles gradúan lo que CUESTA CALCULAR
(épocas, profundidad, parámetros, longitud de secuencia), no lo que ocupan
los datos.

Lo que NO desapareció con ellos, porque sí protegía de algo real:
  · `_HOSTED` — el playground compartido (matrixaistudio.org corre con
    `MATRIXAI_HOSTED=1`, ver `scripts/demo-backend.sh`) conserva EXACTAMENTE
    los topes de antes. Ahí la máquina no es de quien sube el fichero.
  · La guarda de MEMORIA de más abajo (`memoria_insuficiente`), que dice
    ANTES de empezar que un CSV no cabe en la RAM de esta máquina. Un tope
    que desaparece sin que nadie lo note es tan malo como uno absurdo: sin
    esto, un fichero enorme ya no daría un mensaje, daría un OOM del kernel.
"""
from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar

# Perfil "equilibrado" = el DEFAULT en la máquina de quien lo usa. Cada valor:
# int (tope) o None (sin tope).
#
# `max_rows` y `max_csv_bytes` valen None desde el 2026-09-12 (ver el docstring):
# el tamaño de los datos ya no es un tope de producto. Los que quedan acotan lo
# que cuesta CALCULAR, que es otra cosa: un número tecleado de más (500 capas,
# 100.000 épocas) convierte un clic en horas de CPU, y eso sí merece un techo
# con su aviso.
_EQUILIBRADO: dict[str, int | None] = {
    "max_rows": None,
    "max_epochs": 1_000,
    "max_csv_bytes": None,
    "max_depth": 12,
    "max_labels": 12,
    # SECUENCIAS_PRODUCTO C2 (auditoría [ALTA]): Text[L] sin tope permite pedir
    # una SEQUENCE arbitrariamente grande — la atención del bloque transformer
    # escala O(L²) y el .mxtrain/CSV generados escalan O(L) por fila (varias
    # veces el texto: mxai, training_text, dataset_template_text).
    "max_sequence_length": 512,
    # CONTRATO 64 C1 — presupuesto de PARÁMETROS de la red. Sin esto no se podía
    # prometer "topes del perfil" para el tamaño del modelo: el resto de topes
    # acota los datos y el entrenamiento, nunca la capacidad. 2.000.000 son unos
    # 8 MB de pesos en float32 y entrenan en CPU en minutos, que es el listón del
    # perfil por defecto.
    "max_params": 2_000_000,
}
# Perfil "avanzado": máquina potente; topes altos pero aún con red de seguridad anti-typo.
# `max_rows`/`max_csv_bytes` a None como en "equilibrado": si el perfil por
# defecto ya no topa los datos, dejarlos aquí en 1 GB / 1.000.000 de filas
# haría que ELEGIR "avanzado" BAJARA el techo — un perfil "más potente" que
# admite menos datos que el normal.
_AVANZADO: dict[str, int | None] = {
    "max_rows": None,
    "max_epochs": 100_000,
    "max_csv_bytes": None,
    "max_depth": 128,
    "max_labels": 128,
    "max_sequence_length": 8_192,
    # ~200 MB de pesos: razonable en una GPU de portátil y en Colab, inviable en
    # CPU — que es exactamente la diferencia que separa este perfil del anterior.
    "max_params": 50_000_000,
}
# Perfil "ilimitado": sin topes. SOLO descargable (hosted nunca lo ofrece).
_ILIMITADO: dict[str, int | None] = {k: None for k in _EQUILIBRADO}

# TOPES DEL SERVICIO COMPARTIDO (`MATRIXAI_HOSTED=1`). Eran "los del perfil
# equilibrado" y se congelan aquí el 2026-09-12, cuando el perfil local dejó de
# topar el tamaño de los datos: en el playground hosted la máquina NO es de
# quien sube el fichero, así que ahí 50 MB / 50.000 filas siguen siendo lo que
# eran. Valores idénticos a los que `_EQUILIBRADO` tenía hasta esa fecha — esto
# no cambia nada de lo que ve la demo.
_HOSTED: dict[str, int | None] = {
    "max_rows": 50_000,
    "max_epochs": 1_000,
    "max_csv_bytes": 50_000_000,
    "max_depth": 12,
    "max_labels": 12,
    "max_sequence_length": 512,
    "max_params": 2_000_000,
}

_PROFILES: dict[str, dict[str, int | None]] = {
    "equilibrado": _EQUILIBRADO,
    "avanzado": _AVANZADO,
    "ilimitado": _ILIMITADO,
}

_ENV_BY_KEY = {
    "max_rows": "MATRIXAI_MAX_ROWS",
    "max_epochs": "MATRIXAI_MAX_EPOCHS",
    "max_csv_bytes": "MATRIXAI_MAX_CSV_BYTES",
    "max_depth": "MATRIXAI_MAX_DEPTH",
    "max_labels": "MATRIXAI_MAX_LABELS",
    "max_sequence_length": "MATRIXAI_MAX_SEQUENCE_LENGTH",
    "max_params": "MATRIXAI_MAX_PARAMS",
}

_UNLIMITED_TOKENS = {"0", "none", "unlimited", "ilimitado", "sin", "off"}

_INVALID = object()  # centinela: override que no parsea (se ignora, cae al perfil)


def is_hosted() -> bool:
    """True si corre como playground hosted compartido (topes duros, anti-DoS)."""
    return os.environ.get("MATRIXAI_HOSTED", "0") == "1"


# ── BANCO DE PRUEBAS: el camino donde un tope de producto no protege a nadie ──
#
# El 2026-09-12, midiendo tiempos en Colab, la red densa perdió el dataset
# `KDDCup09_appetency` (50.000 x 231) con «El CSV ocupa 87.6 MB y el máximo es
# 50.0 MB»: un tope pensado para quien SUBE un fichero a un servicio, aplicado a
# una medición de laboratorio. El motor `matrixai.dense.torch_cpu` pasa por el
# core vía CSV (`matrixai_engines/motores/densa.py`), así que heredaba los topes
# del producto — y en la tabla de resultados eso sale como «dataset perdido» sin
# que nadie sepa que el motivo no tiene nada que ver con la calidad del modelo.
#
# `ContextVar` y no una variable de entorno: el banco corre motores en el mismo
# proceso (y en hilos, `threadpool_limits`), y una env var la ven TODOS a la vez
# —justo el fallo que la auditoría externa del 2026-09-09 encontró con
# `MATRIXAI_TRAIN_BACKEND`—. Con un ContextVar, lo que vale dentro del `with` no
# se escapa fuera ni pisa a nadie.
_SIN_TOPES: ContextVar[bool] = ContextVar("matrixai_sin_topes", default=False)


@contextlib.contextmanager
def sin_topes():
    """Dentro del bloque, TODOS los topes de producto valen «sin tope».

    Para el BANCO DE PRUEBAS (`benchmarks/fase0`, los motores de
    `matrixai_engines`), nunca para una petición de usuario. Dos cosas que NO
    apaga, porque no son topes de producto:

      · `MATRIXAI_HOSTED=1` — ahí los topes son duros y esto no los toca (si el
        banco corriera dentro del servicio compartido, seguiría acotado).
      · La guarda de memoria (`memoria_insuficiente`) — que un CSV no quepa en
        la RAM de la máquina no es una política, es física, y en el banco
        importa igual: mejor el mensaje que el OOM.

    LO QUE NO ALCANZA, DICHO: un `threading.Thread` arranca con un contexto
    VACÍO, así que dentro de un hilo lanzado desde el bloque los topes vuelven
    a valer. Hoy no afecta —medido el 2026-09-12: los siete sitios que
    consultan `limits` (architecture_policy, dataset_analysis, dense_generator,
    dataset_project, transformer_generator, playground, cli) se ejecutan en el
    hilo que llama, y el hilo del entrenamiento (`_run_playground_training`) no
    consulta ninguno—, pero quien meta una comprobación de tope dentro de un
    hilo trabajador tiene que pasarle el contexto (`contextvars.copy_context`).
    """
    token = _SIN_TOPES.set(True)
    try:
        yield
    finally:
        _SIN_TOPES.reset(token)


def en_banco_de_pruebas() -> bool:
    """True si hay un `sin_topes()` activo en este contexto."""
    return _SIN_TOPES.get()


def _profile_name() -> str:
    name = os.environ.get("MATRIXAI_LIMITS_PROFILE", "equilibrado").strip().lower()
    return name if name in _PROFILES else "equilibrado"


def _parse_override(raw: str):
    """int (tope), None (sin tope) o _INVALID si no parsea."""
    token = raw.strip().lower()
    if token in _UNLIMITED_TOKENS:
        return None
    try:
        value = int(token)
    except ValueError:
        return _INVALID
    return None if value <= 0 else value


def get_limit(key: str) -> int | None:
    """Tope efectivo (int) o None (sin tope) para `key`, aplicando hosted/env/perfil."""
    if key not in _EQUILIBRADO:
        raise KeyError(f"unknown limit {key!r}")
    # 1. Hosted: topes duros del servicio compartido; ignora overrides (anti-DoS).
    if is_hosted():
        return _HOSTED[key]
    # 2. Banco de pruebas: sin topes de producto (ver `sin_topes`).
    if _SIN_TOPES.get():
        return None
    # 3. Override por-límite por env.
    raw = os.environ.get(_ENV_BY_KEY[key])
    if raw is not None:
        parsed = _parse_override(raw)
        if parsed is not _INVALID:
            return parsed  # int o None
    # 4/5. Perfil (equilibrado por defecto).
    return _PROFILES[_profile_name()][key]


def cap(value: int, key: str) -> int:
    """Aplica el tope `key` a `value`: `value` si el tope es None, si no `min(value, tope)`."""
    limit = get_limit(key)
    return value if limit is None else min(value, limit)


def exceeds(value: int, key: str) -> bool:
    """True si `value` supera el tope `key` (False si no hay tope)."""
    limit = get_limit(key)
    return limit is not None and value > limit


# ── Error estructurado de tope superado (CONTRATO 62 C1) ─────────────────────
#
# Antes, cada sitio que comprobaba un tope componía su propia frase ("CSV tiene
# N filas, máximo M", "Dataset sintetico supera el límite de N KB"), y quien la
# recibía solo podía reconocerla parseando texto. Peor: `dataset_project` la
# envolvía en "esto indica un hueco en la preparación del CSV, no un problema de
# tus datos", que para un tope es FALSO — la causa es el perfil y la solución es
# cambiarlo. Ahora el core emite datos; el texto humano es solo respaldo y la
# localización ES/EN vive en la SPA, que es quien sabe de idiomas.

LIMIT_EXCEEDED = "limit_exceeded"

# Unidades por límite, para que el consumidor formatee sin adivinar.
_LIMIT_UNITS = {
    "max_rows": "rows",
    "max_epochs": "epochs",
    "max_csv_bytes": "bytes",
    "max_depth": "layers",
    "max_labels": "labels",
    "max_sequence_length": "chars",
    "max_params": "params",
}


def origen_del_tope(key: str) -> str:
    """De dónde sale el tope efectivo: "hosted" | "override" | "perfil".

    Importa desde el 2026-09-12, y es un aviso que este cambio volvió FALSO si
    no se toca: quitado el techo de datos del perfil, la ÚNICA forma de toparse
    con `max_csv_bytes`/`max_rows` es haberlo puesto a mano — y a quien lo puso
    a mano decirle «cambia el perfil a avanzado» no le arregla nada, porque el
    override por-límite gana al perfil (ver `get_limit`).
    """
    if is_hosted():
        return "hosted"
    raw = os.environ.get(_ENV_BY_KEY[key])
    if raw is not None and _parse_override(raw) is not _INVALID:
        return "override"
    return "perfil"


def limit_error(key: str, actual: int) -> dict:
    """Payload estructurado de un tope superado, listo para `{"ok": False, **payload}`.

    `configurable` es exactamente la condición con la que `get_limit` ignora los
    overrides: en hosted los topes son duros (anti-DoS) y ofrecer "cámbialo en
    Ajustes" sería mentir. `error` es texto de respaldo en español para quien no
    entienda la forma estructurada (CLI, logs, clientes viejos).
    """
    maximum = get_limit(key)
    hosted = is_hosted()
    origen = origen_del_tope(key)
    return {
        "error_kind": LIMIT_EXCEEDED,
        "limit_key": key,
        "unit": _LIMIT_UNITS.get(key, "units"),
        "actual": actual,
        "maximum": maximum,
        "profile": "equilibrado" if hosted else _profile_name(),
        "configurable": not hosted,
        # 2026-09-12: QUIÉN puso este tope. Sin esto, el consumidor solo puede
        # ofrecer «cambia de perfil», que para un tope puesto a mano es falso.
        "origen": origen,
        "error": _limit_error_text(key, actual, maximum, hosted, origen),
    }


# Etiqueta en español SOLO para el texto de respaldo; `unit` (en inglés) es el
# token estable que consume la SPA para localizar. No se mezclan.
_LIMIT_UNITS_ES = {
    "max_rows": "filas",
    "max_epochs": "épocas",
    "max_depth": "capas",
    "max_labels": "etiquetas",
    "max_sequence_length": "caracteres",
    "max_params": "parámetros",
}


def human_bytes(n: int) -> str:
    """Tamaño legible con la unidad ADECUADA a la magnitud.

    Formatear siempre en MB daba "El CSV ocupa 0.0 MB y el máximo es 0 MB" para
    un tope pequeño — un mensaje inútil justo cuando más falta hace entenderlo.
    El escalón de GB se añadió el 2026-09-12 por el mismo motivo, en el otro
    extremo: la guarda de memoria habla de la RAM de la máquina, y "7959.1 MB
    disponibles" es un número que hay que traducir a mano para entenderlo.
    """
    if n < 1_000:
        return f"{n} B"
    if n < 1_000_000:
        return f"{n / 1_000:.1f} KB"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n / 1_000_000_000:.1f} GB"


def _limit_error_text(key: str, actual: int, maximum: int | None, hosted: bool,
                      origen: str = "perfil") -> str:
    if key == "max_csv_bytes":
        actual_h = human_bytes(actual)
        max_h = "sin límite" if maximum is None else human_bytes(maximum)
        cuerpo = f"El CSV ocupa {actual_h} y el máximo es {max_h}"
    elif key == "max_params":
        # El sujeto no es el dataset sino la RED: decir "el dataset tiene
        # 3.400.000 parámetros" no significaría nada.
        max_h = "sin límite" if maximum is None else f"{maximum:,}".replace(",", ".")
        cuerpo = (f"La red tiene {actual:,}".replace(",", ".")
                  + f" parámetros y el máximo es {max_h}")
    else:
        unit = _LIMIT_UNITS_ES.get(key, "unidades")
        max_h = "sin límite" if maximum is None else f"{maximum}"
        cuerpo = f"El dataset tiene {actual} {unit} y el máximo es {max_h}"
    if hosted:
        return f"{cuerpo}. Es un límite del servicio compartido y no se puede subir."
    if origen == "override":
        # Un tope puesto a mano gana al perfil: mandar a cambiar el perfil sería
        # mandar a hacer algo que NO lo sube (medido en `get_limit`, orden 3 > 4).
        return (f"{cuerpo}. Es un tope que está puesto a mano "
                f"({_ENV_BY_KEY[key]}): súbelo o quítalo en Ajustes → Límites.")
    return (f"{cuerpo}. Puedes subirlo en Ajustes → Límites "
            f"(perfil «avanzado» o «ilimitado»).")


def is_limit_error(payload: object) -> bool:
    """True si `payload` es (o contiene) un error estructurado de tope."""
    return isinstance(payload, dict) and payload.get("error_kind") == LIMIT_EXCEEDED


# ── Guarda de MEMORIA: lo que sustituye al número arbitrario ──────────────────
#
# Quitar el tope de 50 MB sin poner nada en su sitio dejaría que un fichero
# enorme se llevara por delante el proceso SIN UN MENSAJE: el kernel mata a
# quien más memoria ocupa y no hay excepción que capturar. Esto no es un tope de
# producto —no se elige, no se sube en Ajustes—: es la máquina diciendo lo que
# no cabe, ANTES de empezar a leer.
#
# EL FACTOR ESTÁ MEDIDO, no supuesto (2026-09-12, esta máquina, Python 3.12,
# `analyze_dataset_csv` sobre tres formas de CSV distintas):
#
#     50.000 x 231 celdas cortas   23,1 MB de CSV -> 0,51 GB de RSS  (x22,3)
#    200.000 x  11 celdas largas   30,4 MB de CSV -> 0,40 GB de RSS  (x13,1)
#     20.000 x 101 celdas medias   14,0 MB de CSV -> 0,24 GB de RSS  (x17,3)
#
# Se usa el MÍNIMO medido redondeado hacia abajo (x12) a propósito: así solo se
# rechaza lo que no cabría NI EN EL MEJOR CASO. Pasarse por arriba convertiría
# esta guarda en otro techo falso, que es justo lo que se viene a quitar.
MEMORIA_INSUFICIENTE = "memoria_insuficiente"
FACTOR_RAM_POR_BYTE_DE_CSV = 12
#: Bytes por celda de un CSV sintético (medido el 2026-09-12: 2.000 filas x 3
#: columnas = 33.571 bytes, 5,6 por celda). Para estimar lo que se va a generar
#: ANTES de generarlo.
BYTES_POR_CELDA_SINTETICA = 6


def _memoria_disponible_del_sistema() -> int | None:
    """Bytes de RAM disponible, o None si esta máquina no sabe decirlo.

    `MemAvailable` (no `MemFree`): es lo que el kernel estima que se puede pedir
    sin empezar a intercambiar, que es la pregunta de verdad. Si no hay
    /proc/meminfo (macOS, Windows) se devuelve None y la guarda NO opina — un
    guardián que no puede medir no debe bloquear (mismo criterio fail-open que
    `onnx_size_limit_error`).
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for linea in fh:
                if linea.startswith("MemAvailable:"):
                    return int(linea.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def guarda_de_memoria_activa() -> bool:
    """False si `MATRIXAI_GUARDIA_MEMORIA` la apaga (`0`/`off`/`no`).

    Se puede apagar porque la estimación es eso, una estimación: quien sepa que
    su caso cabe (o quiera que el kernel decida) tiene que poder seguir. Por
    omisión está ENCENDIDA: el mensaje antes del OOM es el motivo de existir.
    """
    return os.environ.get("MATRIXAI_GUARDIA_MEMORIA", "1").strip().lower() not in (
        "0", "off", "no", "false")


def memoria_insuficiente(csv_bytes: int) -> dict | None:
    """Payload estructurado si leer un CSV de `csv_bytes` NO cabe en esta
    máquina; `None` si cabe, si no se puede medir o si la guarda está apagada.

    `configurable: False` y sin «cámbialo en Ajustes»: subir el perfil no añade
    memoria, y ofrecer una acción que no arregla nada es peor que no ofrecer
    ninguna.
    """
    if not guarda_de_memoria_activa():
        return None
    disponible = _memoria_disponible_del_sistema()
    if disponible is None:
        return None
    necesario = csv_bytes * FACTOR_RAM_POR_BYTE_DE_CSV
    if necesario <= disponible:
        return None
    return {
        "error_kind": MEMORIA_INSUFICIENTE,
        "actual": csv_bytes,
        "necesario_estimado": necesario,
        "disponible": disponible,
        "factor": FACTOR_RAM_POR_BYTE_DE_CSV,
        "configurable": False,
        "error": (
            f"El CSV ocupa {human_bytes(csv_bytes)} y leerlo necesita unos "
            f"{human_bytes(necesario)} de memoria; esta máquina tiene "
            f"{human_bytes(disponible)} disponibles. No es un tope que se pueda "
            f"subir: hacen falta menos datos o más memoria."
        ),
        "error_en": (
            f"The CSV is {human_bytes(csv_bytes)} and reading it needs about "
            f"{human_bytes(necesario)} of memory; this machine has "
            f"{human_bytes(disponible)} available. This is not a limit you can "
            f"raise: it needs less data or more memory."
        ),
    }


def es_error_de_memoria(payload: object) -> bool:
    """True si `payload` es (o contiene) el error estructurado de memoria."""
    return isinstance(payload, dict) and payload.get("error_kind") == MEMORIA_INSUFICIENTE


def limits_snapshot() -> dict:
    """Estado de los límites para el endpoint /config y la UI (M12 Corte UI).

    En hosted el nombre que se publica sigue siendo "equilibrado" (es el que la
    SPA sabe rotular, y el que este perfil ha tenido siempre); lo que manda son
    los VALORES de `limits` —los de `_HOSTED`— y `hosted: True`, que es lo que
    dice que no se pueden cambiar.

    `guarda_de_memoria` viaja desde el 2026-09-12: sin `max_csv_bytes`, es lo
    único que puede rechazar un CSV por tamaño, y una pantalla de Límites que no
    lo nombrara estaría enseñando «sin límite» a secas — media verdad.
    """
    hosted = is_hosted()
    return {
        "hosted": hosted,
        "profile": "equilibrado" if hosted else _profile_name(),
        "limits": {k: get_limit(k) for k in _EQUILIBRADO},
        "profiles_available": ["equilibrado", "avanzado"] + ([] if hosted else ["ilimitado"]),
        "guarda_de_memoria": {
            "activa": guarda_de_memoria_activa(),
            "factor": FACTOR_RAM_POR_BYTE_DE_CSV,
            "disponible": _memoria_disponible_del_sistema(),
        },
    }
