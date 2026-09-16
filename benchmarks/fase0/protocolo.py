# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C1 — el protocolo de la Fase 0 EXPLORATORIA, registrado con hash.

Invariante 1 del contrato 101: «lista de datasets, versiones, métricas,
espacios de búsqueda, presupuesto y reglas de decisión quedan registrados con
hash ANTES de cada pasada». Este módulo es la FORMA de ese registro —
`ProtocoloExploratorio`— y el cálculo que el criterio de terminado del corte
exige explícitamente: «un cálculo a partir del protocolo produce número de
ejecuciones y cota de coste consistente con motores, pliegues, repeticiones y
presupuesto» (`calcular_coste`).

**Vive fuera del paquete `matrixai`, a propósito.** `benchmarks/` no se
publica en PyPI (invariante 1 del 102, aplicado también aquí: el core sigue
`dependencies = []`) — este módulo sí puede importar lo que haga falta para
medir (aquí, nada más que la stdlib y `matrixai.estudio.validacion.
digest_canonico`, reutilizado en vez de inventar un segundo canonicalizador:
«dos sitios declarando lo mismo acaban divergiendo»).

**Qué NO decide este módulo.** La lista real de 40 datasets con sus sha256 la
construyó `generar_protocolo.py` consultando OpenML en vivo (medido, no
inventado) y quedó fijada en `protocolo_exploratorio.json` — este fichero
solo sabe LEER y VALIDAR esa forma, y calcular sobre ella. Cambiar un dataset,
un presupuesto o una regla de cierre después de fijado el protocolo no es
tocar este módulo: es escribir un protocolo NUEVO, con su propio digest.
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
if str(_RAIZ_DEL_CORE) not in sys.path:
    sys.path.insert(0, str(_RAIZ_DEL_CORE))

from matrixai.estudio.validacion import digest_canonico  # noqa: E402

__all__ = [
    "ANCLAS_DE_ESCALADO",
    "CUBOS_DE_TAMANO",
    "TAREAS",
    "UMBRAL_ALTA_CARDINALIDAD",
    "cardinalidad_nominal_declarada",
    "CosteDeLaPasada",
    "DatasetRegistrado",
    "DisenoDeParticion",
    "Motor",
    "PresupuestoPorCubo",
    "ProtocoloError",
    "ProtocoloExploratorio",
    "ReglaDeCierre",
    "EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR",
    "alcance_de_una_pasada",
    "aplicar_regla_de_cierre",
    "metrica_del_dataset",
    "calcular_coste",
    "dispersion_de_un_motor",
    "estabilidad_del_ganador",
    "veredicto_con_su_alcance",
    "cpus_disponibles",
    "nucleos_fisicos",
    "reserva_segura",
    "speedup_medido",
]

TAREAS = ("binary_classification", "multiclass_classification", "regression")
CUBOS_DE_TAMANO = ("pequeno", "mediano", "grande")


class ProtocoloError(ValueError):
    """Un protocolo mal formado: no se corrige a medias, se rechaza entero —
    igual que un `EsquemaInvalido` del 104-C0, del que esto es primo, no
    heredero (este paquete no depende de `matrixai.estudio` para lo demás)."""


def _exigir(cond: bool, mensaje: str) -> None:
    if not cond:
        raise ProtocoloError(mensaje)


# ---------------------------------------------------------------------------
# Reserva de CPU — 101-C1/C2, sobre-reserva de concurrencia
# ---------------------------------------------------------------------------
# El protocolo registrado PEDÍA `hilos=4` x `procesos_en_paralelo=6` = 24
# hilos. Esta máquina tiene 8 CPUs LÓGICAS (4 núcleos físicos con SMT x2),
# medido el 2026-09-12 y vuelto a medir el 2026-09-13: `os.sched_getaffinity`
# 8, `os.cpu_count()` 8, `nucleos_fisicos()` 4 (pares `physical id`/`core id`
# únicos de `/proc/cpuinfo`), sin cuota de cgroup (`cpu.max` = «max 100000»).
# 24 sobre 8 era TRIPLE reserva — y `recursos_declarados` decía «cpu_fisicas:
# 8», que no es lo medido: son 8 lógicas y 4 físicas.
#
# **RE-FIRMADO el 2026-09-13**: el protocolo pide ahora `hilos=4` x
# `procesos_en_paralelo=2` = 8 hilos, que es exactamente `reserva_segura(4)`
# sobre `cpus_disponibles()` — la propia función de este módulo, no un número a
# ojo — y `cpu_fisicas` dice 4. La sobre-reserva pasa de 16 hilos a CERO. Lo
# que NO se movió en esa re-firma: la regla de cierre, la definición de
# «mejor», el listón, la lista de datasets y las particiones.
#
# No es teoría en este servidor: se cayó dos veces (2026-08-10 y 2026-08-16)
# por reservar más de lo que hay, y no tiene swap suficiente para salvarlo —
# el kernel no muere, se ATASCA, y se pierden el SSH y la web a la vez.


def cpus_disponibles() -> int:
    """Las CPUs que este proceso puede usar DE VERDAD, medidas — nunca un 8
    escrito a mano.

    Tres fuentes, y se queda con la más restrictiva, porque cada una puede
    mentir por separado:

    - `os.sched_getaffinity(0)`: respeta `taskset`. Aquí sin restringir da 8,
      igual que `os.cpu_count()` — los dos coinciden y por eso el `cpu_count`
      solo NO parece peligroso. Bajo `taskset -c 0,1` se separan y se ve el
      fallo: afinidad 2, `os.cpu_count()` 8 (medido el 2026-09-12), o sea
      CUATRO veces más de lo que toca. `nproc --all` dice 128 en esta
      máquina: ese es el host entero, y no es lo que nos han dado.
    - `cpu.max` de cgroup v2: el techo de un contenedor (`docker --cpus=N`).
      «max» significa sin cuota, y una AUSENCIA no es un cero.
    - `os.cpu_count()` como último recurso, si las otras dos no están.
    """
    candidatos: list[int] = []
    try:
        candidatos.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # no todas las plataformas la tienen
        pass
    cuota = _cuota_de_cgroup()
    if cuota is not None:
        candidatos.append(cuota)
    if not candidatos:
        candidatos.append(os.cpu_count() or 1)
    return max(1, min(candidatos))


_CPU_MAX_DE_CGROUP = Path("/sys/fs/cgroup/cpu.max")


def _cuota_de_cgroup(ruta: Path | None = None) -> int | None:
    """CPUs enteras que permite la cuota de cgroup v2, o `None` si no hay
    cuota. `cpu.max` es «<cuota> <periodo>» en microsegundos, y la cuota
    literal «max» quiere decir SIN TOPE — no cero.

    `ruta` existe para poder PROBAR el parseo con ficheros de verdad en vez
    de suponer qué escribe el kernel: en esta máquina no hay cuota, así que
    sin este parámetro la rama que importa (`docker --cpus=N`) no se probaría
    nunca aquí."""
    try:
        crudo = (ruta or _CPU_MAX_DE_CGROUP).read_text(encoding="utf-8").split()
    except (OSError, ValueError):
        return None
    if len(crudo) != 2 or crudo[0] == "max":
        return None
    try:
        cuota, periodo = int(crudo[0]), int(crudo[1])
    except ValueError:
        return None
    if cuota <= 0 or periodo <= 0:
        return None
    return max(1, cuota // periodo)


def nucleos_fisicos() -> int | None:
    """Núcleos FÍSICOS, o `None` si no se pueden medir aquí.

    Importa porque el SMT no duplica el rendimiento: medido abajo, pasar de 4
    hilos (4 físicos) a 8 (los 8 lógicos) sube el rendimiento solo 1,30x, no
    2x. Un `None` es «no lo sé», y no se rellena con un número inventado.
    """
    try:
        texto = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    nucleos: set[tuple[str, str]] = set()
    fisico = core = None
    for linea in texto.splitlines():
        if ":" not in linea:
            fisico = core = None
            continue
        clave, _, valor = linea.partition(":")
        clave, valor = clave.strip(), valor.strip()
        if clave == "physical id":
            fisico = valor
        elif clave == "core id":
            core = valor
        if fisico is not None and core is not None:
            nucleos.add((fisico, core))
            fisico = core = None
    return len(nucleos) or None


# Escalado MEDIDO en esta máquina el 2026-09-12, con un ajuste real de
# `sklearn.hgb` (uno de los 7 motores del protocolo: CPU-bound y paralelizado
# por OpenMP, como lightgbm/xgboost/catboost) sobre 60.000 x 60 sintético,
# repetido y quedándose con el mejor tiempo, con la carga del sistema medida
# ANTES de cada punto. Cada par es (hilos reservados, speedup medido):
#
#   1 proceso  x 1 hilo  =  1 hilo  -> 1,00x  (base, 4,92 s por ajuste, carga 0,91)
#   4 procesos x 1 hilo  =  4 hilos -> 3,60x  (5,47 s por ajuste, carga 1,24)
#   8 procesos x 1 hilo  =  8 hilos -> 4,69x  (8,39 s por ajuste, carga 2,30)
#
# Los 8 hilos son los 8 LÓGICOS: el salto de 4 a 8 da 1,30x, no 2x, porque
# solo hay 4 núcleos físicos. 4,69x es el TECHO de rendimiento de la máquina:
# ninguna repartición de más hilos puede sacar más trabajo de 8 CPUs.
#
# Un cuarto punto medido, que es el que condena la forma que eligió el
# protocolo: 1 proceso x 4 HILOS (4 hilos dentro del mismo proceso) da 2,53x,
# no 3,60x — empaquetar hilos dentro de un proceso escala MUCHO peor que
# repartirlos en procesos. El protocolo pide justo eso (4 hilos por proceso),
# así que su speedup real está POR DEBAJO de esta curva. Ver `speedup_medido`.
ANCLAS_DE_ESCALADO: tuple[tuple[int, float], ...] = ((1, 1.00), (4, 3.60), (8, 4.69))

# Las anclas se midieron sobre 8 CPUs lógicas. Se aplican a otra máquina por
# FRACCIÓN de sus CPUs, que es una extrapolación sin medir: queda declarado
# como deuda en `speedup_medido`, no disfrazado de medición.
_CPUS_DE_LA_MEDICION = 8


def speedup_medido(hilos_reservados: int, cpus: int | None = None) -> float:
    """Cuánto rinde DE VERDAD reservar `hilos_reservados`, según lo medido —
    en vez de suponer que N hilos van N veces más rápido.

    Devuelve un speedup sobre un solo hilo. Dos propiedades que el modelo
    lineal no tenía:

    - **Nunca pasa del techo de la máquina.** Por encima de `cpus` hilos el
      speedup se queda en el medido con la máquina llena (4,69x con 8 CPUs):
      reservar 24 hilos no da 24x ni 6x, da como mucho lo que las CPUs pueden.
    - **Con holgura sí sube.** Con 4 hilos de 8 devuelve 3,60x, no 1x: esto
      limita la reserva, no la estrangula.
    - **Nunca baja de 1,00x**, porque el speedup se mide SOBRE un hilo y nada
      rinde menos que su propia unidad de medida.

    **Lo que este número NO es**: una cota superior del tiempo. Es un SUELO
    del tiempo, porque es un TECHO del rendimiento — y además las anclas se
    midieron con 1 hilo por proceso, que escala mejor que los 4 hilos por
    proceso que pide el protocolo (3,60x frente a 2,53x con 4 hilos). La
    pasada real irá más lenta que esto, no más rápida.

    **DEUDA declarada, dos trozos, para no vender como medido lo que no lo
    está:**

    1. *El reparto exacto del protocolo (6 procesos x 4 hilos) no está
       medido.* Se intentó el 2026-09-12 y se ABORTÓ a mitad: la propia
       medición puso la máquina en carga 19,8 con otros agentes trabajando,
       que es exactamente el fallo que este módulo repara. Lo que se usa es el
       TECHO de la máquina (4,69x), que ese reparto no puede superar — pero
       cuánto queda por debajo no se sabe, y por eso el resultado se llama
       `horas_reloj_suelo_medido` y no «cota».
    2. *Las anclas valen para ESTA máquina* (8 lógicas / 4 físicas). Aplicarlas
       a otra por fracción de CPUs es una extrapolación sin medir: una máquina
       sin SMT, o con 32 núcleos, tiene otra curva. Quien mueva la pasada de
       servidor vuelve a medir estos tres puntos y cambia `ANCLAS_DE_ESCALADO`.

    Lo que NO es deuda y vale en cualquier máquina: `reserva_segura` nunca
    reserva más hilos que CPUs haya. El límite duro no depende de la curva.
    """
    _exigir(hilos_reservados >= 1, "hilos_reservados tiene que ser >= 1")
    disponibles = cpus if cpus is not None else cpus_disponibles()
    _exigir(disponibles >= 1, "cpus tiene que ser >= 1")
    # Saturada la máquina, más hilos no dan más trabajo: el techo es el ancla
    # más alta, escalada al tamaño de esta máquina. Lo aplica el `min` final y
    # SOLO ese sitio — un `return techo` temprano aquí daba exactamente el
    # mismo resultado (queda comprobado: al neutralizarlo la suite seguía
    # verde), y dos sitios imponiendo el mismo techo acaban divergiendo.
    # `max(1.0, …)`: el speedup se define SOBRE UN HILO, así que un hilo es
    # 1,00x por definición y nada puede rendir menos que la unidad con la que
    # se mide. Re-auditoría del 2026-09-12: sin ese suelo, con 1 CPU el techo
    # salía `1 * (4,69/8) = 0,586` y el modelo afirmaba que un hilo rinde
    # menos que un hilo. Se alcanza de verdad en un `docker --cpus=1`, que es
    # justo lo que `cpus_disponibles()` existe para detectar.
    techo = max(1.0, disponibles * (ANCLAS_DE_ESCALADO[-1][1] / _CPUS_DE_LA_MEDICION))
    # Por debajo del lleno: interpolación lineal entre anclas por FRACCIÓN de
    # CPUs ocupadas, para que la curva no dependa de que la máquina tenga 8.
    fraccion = hilos_reservados / disponibles
    puntos = [(h / _CPUS_DE_LA_MEDICION, sp / h) for h, sp in ANCLAS_DE_ESCALADO]
    eficiencia = puntos[0][1]
    for (f0, e0), (f1, e1) in zip(puntos, puntos[1:]):
        if fraccion <= f0:
            eficiencia = e0
            break
        if f0 < fraccion <= f1:
            eficiencia = e0 + (e1 - e0) * (fraccion - f0) / (f1 - f0)
            break
    else:
        eficiencia = puntos[-1][1]
    # El suelo va en el RESULTADO, no solo en el techo. Con 1 CPU la
    # interpolación por fracción da la eficiencia de «máquina llena» (0,586)
    # y multiplicarla por 1 hilo daba 0,586: el modelo afirmaba que un hilo
    # rinde menos que un hilo. Con una sola CPU no hay contención porque no
    # hay con quién competir — la curva por fracción no lo sabe, y el suelo
    # sí. Re-auditoría del 2026-09-12.
    return round(max(1.0, min(hilos_reservados * eficiencia, techo)), 4)


def reserva_segura(hilos_por_proceso: int, cpus: int | None = None) -> int:
    """Cuántos procesos caben SIN pasarse de las CPUs que hay: el límite duro.

    `4 hilos x 6 procesos = 24` sobre 8 CPUs es lo que había; con 4 hilos por
    proceso esto devuelve **2**. Y con holgura devuelve lo que hay: 1 hilo por
    proceso sobre 8 CPUs devuelve 8, no 1 — un techo que reservase siempre 1
    haría la pasada eterna y pasaría igual de «seguro».

    Nunca devuelve 0: si un solo proceso ya pide más hilos que CPUs hay, la
    respuesta es 1 proceso (no lanzar nada no es una opción), y quien mire
    `sobre_reserva` verá que ese proceso solo ya se pasa.
    """
    _exigir(hilos_por_proceso >= 1, "hilos_por_proceso tiene que ser >= 1")
    disponibles = cpus if cpus is not None else cpus_disponibles()
    _exigir(disponibles >= 1, "cpus tiene que ser >= 1")
    return max(1, disponibles // hilos_por_proceso)


# ---------------------------------------------------------------------------
# Cardinalidad — el dato del catálogo que estaba FALSO hasta la re-firma
# ---------------------------------------------------------------------------
#: A partir de cuántos niveles una columna nominal cuenta como «de alta
#: cardinalidad». **No es un número elegido aquí**: es el que el anexo C del
#: documento 100 escribió en §2.2 antes de medir nada — «≥ 12 con categóricas
#: (**≥ 5 con alguna de cardinalidad ≥ 50**)».
#:
#: HASTA EL 2026-09-13 EL CATÁLOGO NO MEDÍA ESTO. `generar_protocolo.py`
#: rellenaba `alta_cardinalidad` con `columnas_simbólicas / columnas > 0,3`,
#: que no es cardinalidad sino PROPORCIÓN DE COLUMNAS CATEGÓRICAS — la otra
#: mitad de la misma frase del anexo, con el nombre de la primera. Las dos
#: direcciones salían mal, medido sobre los ARFF sellados:
#:
#:   · `KDDCup09_appetency` declaraba `alta_cardinalidad: false` con **15.415
#:     niveles** en `Var200` (y otros tantos en `Var214`; 71.506 en total,
#:     contando el objetivo). 38 columnas nominales de 230 son 0,165, así que
#:     la regla vieja decía «no» sobre el dataset MÁS cardinal de los cuarenta.
#:   · `kr-vs-kp`, `PhishingWebsites` y `connect-4` declaraban `true` con un
#:     máximo de **3 niveles**: son enteramente categóricos y no tienen ni una
#:     columna cardinal.
#:
#: Un aserto negativo lo pasa un fichero vacío, así que el catálogo registra el
#: NÚMERO medido (`max_cardinalidad_nominal`) y no solo el booleano, y
#: `DatasetRegistrado` exige que los dos digan lo mismo.
UMBRAL_ALTA_CARDINALIDAD = 50


def cardinalidad_nominal_declarada(ruta_arff: str | Path,
                                   columna_objetivo: str = "") -> dict[str, int]:
    """Los niveles que la CABECERA del ARFF declara para cada columna nominal,
    excluida `columna_objetivo`. Solo lee hasta `@data`: los 254 MB de los
    cuarenta ficheros no hacen falta para contar lo que la cabecera ya dice.

    Devuelve `{nombre de columna: nº de niveles}`, y las columnas numéricas no
    aparecen — **una columna sin niveles nominales no es una columna con cero
    niveles**, y meterla como 0 haría que `max(...)` sobre un dataset sin
    nominales pareciera una medición en vez de una ausencia.

    Se mide lo que el fichero DECLARA, sin reinterpretarlo: no se descuentan
    columnas identificadoras. `splice` llega a 3.178 niveles por
    `Instance_name` (un id por fila) y `house_sales` a 70 por `zipcode`; las
    dos son columnas nominales que un one-hot se comería tal cual, que es
    justo el caso que el anexo quería cubrir. Quién es un identificador y quién
    no es otra pregunta, y se responde con datos, no dentro de un contador.
    """
    niveles: dict[str, int] = {}
    pendiente: str | None = None
    acumulado = ""
    with open(ruta_arff, "r", encoding="utf-8", errors="replace") as fichero:
        for linea in fichero:
            if pendiente is not None:
                # Una declaración `{...}` puede seguir en la línea de abajo.
                # Ninguno de los 40 sellados la parte (comprobado), pero un
                # ARFF futuro sí puede: cerrar en falso contaría de menos.
                acumulado += " " + linea.strip()
                if "}" in linea:
                    niveles[pendiente] = _contar_niveles(acumulado)
                    pendiente, acumulado = None, ""
                continue
            despuntada = linea.strip()
            if despuntada.lower().startswith("@data"):
                break
            if not despuntada.lower().startswith("@attribute"):
                continue
            casado = _ATRIBUTO_ARFF.match(linea)
            if casado is None:
                continue
            nombre = _sin_comillas_arff(casado.group("nombre"))
            tipo = casado.group("tipo").strip()
            if not tipo.startswith("{"):
                continue
            if "}" in tipo:
                niveles[nombre] = _contar_niveles(tipo)
            else:
                pendiente, acumulado = nombre, tipo
    niveles.pop(columna_objetivo, None)
    return niveles


_ATRIBUTO_ARFF = re.compile(
    r"^\s*@attribute\s+(?P<nombre>'[^']*'|\"[^\"]*\"|\S+)\s+(?P<tipo>.+?)\s*$",
    re.IGNORECASE)


def _sin_comillas_arff(texto: str) -> str:
    limpio = texto.strip()
    if len(limpio) >= 2 and limpio[0] == limpio[-1] and limpio[0] in "'\"":
        return limpio[1:-1]
    return limpio


def _contar_niveles(declaracion: str) -> int:
    """Cuenta los niveles de un `{a, b, 'c, d'}` respetando las comillas: una
    coma DENTRO de un nivel entrecomillado no separa nada, y contarla partiría
    un nivel en dos (`house_prices_nominal` y `diamonds` los traen)."""
    dentro = declaracion.strip()
    dentro = dentro[dentro.index("{") + 1:dentro.rindex("}")]
    niveles, actual, comilla = [], [], None
    for caracter in dentro:
        if comilla is not None:
            if caracter == comilla:
                comilla = None
            else:
                actual.append(caracter)
        elif caracter in "'\"":
            comilla = caracter
        elif caracter == ",":
            niveles.append("".join(actual).strip())
            actual = []
        else:
            actual.append(caracter)
    niveles.append("".join(actual).strip())
    return len([n for n in niveles if n != ""])


def _file_id_entero(valor: Any) -> int:
    """El `file_id` tal como llega, convertido a entero SIN redondear nada.

    `int(valor)` a secas parecía bastar —la API de OpenML devuelve `file_id`
    como cadena de dígitos, y convertirla es correcto— pero `int(3.5)` da `3`
    en silencio: un identificador equivocado, dentro del sello, sin una línea
    en ningún sitio. Lo cazó su propia prueba. Una cadena de dígitos sí se
    convierte; un flotante, un booleano o cualquier otra cosa, no.
    """
    if isinstance(valor, bool):
        raise ProtocoloError(f"file_id no puede ser un booleano: {valor!r}")
    if isinstance(valor, int):
        return valor
    if isinstance(valor, str) and valor.strip().isdigit():
        return int(valor.strip())
    raise ProtocoloError(
        f"file_id no es un entero ni una cadena de digitos: {valor!r}")


@dataclass(frozen=True)
class DatasetRegistrado:
    """Un dataset PRE-REGISTRADO: el sha256 es del ARFF tal como se descargó,
    no de una URL ni de un `data_id` — dos descargas del mismo id en momentos
    distintos tienen que dar el mismo contenido, o el protocolo lo rechaza al
    cargar (`ProtocoloExploratorio.__post_init__` lo comprueba: 64 hex).

    `sellado=True` marca uno de los 8 datasets que el protocolo NO toca
    mientras se desarrolla el harness (anti-favoritismo, medido en
    `100_anexos/C_fase0_protocolo_y_motores.md` §2.3): elegidos por posición
    determinista sobre la lista ordenada por `data_id`, ANTES de correr nada,
    no a mano después de ver qué conviene sellar.

    `max_cardinalidad_nominal` es **lo medido** sobre el ARFF sellado: los
    niveles que declara la columna nominal más cardinal, sin contar el
    objetivo, y 0 si el dataset no tiene ni una columna nominal predictora.
    `alta_cardinalidad` es su LECTURA contra el umbral del anexo
    (`UMBRAL_ALTA_CARDINALIDAD`), y `__post_init__` exige que no se separen:
    hasta el 2026-09-13 el booleano viajaba solo, nadie podía contrastarlo con
    nada, y decía lo contrario de lo que los ficheros dicen en diez de los
    cuarenta. Un número medido se puede refutar; un booleano heredado no.
    """

    data_id: int
    nombre: str
    fuente: str
    version: int | None
    sha256_arff: str
    columna_objetivo: str
    tarea: str
    cubo_de_tamano: str
    n_filas: int
    n_columnas: int
    tiene_faltantes: bool
    max_cardinalidad_nominal: int
    alta_cardinalidad: bool
    desbalanceado: bool
    solo_numericas: bool
    licencia: str
    sellado: bool = False
    #: EL IDENTIFICADOR DEL FICHERO EN OPENML, opcional HASTA LA RE-FIRMA.
    #:
    #: `data_id` identifica el DATASET; `file_id` identifica el FICHERO que se
    #: descargó. No son lo mismo: de los 40 del protocolo, **31 tienen un
    #: `file_id` distinto de su `data_id`** (medido el 2026-09-14), así que
    #: quien suponga que coinciden acierta en 9.
    #:
    #: Con él, el ARFF exacto se pide por URL directa
    #: (`file_id_para_la_re_firma.url_de_descarga`). Sin él se llega igual
    #: —`GET /api/v1/json/data/{data_id}` lo devuelve, y el 2026-09-14 los 40
    #: resolvían al mismo fichero, mismo `md5_checksum` que los ARFF
    #: descargados—, pero por una indirección VIVA: la reproducción depende de
    #: que ese mapa siga siendo el mismo dentro de cinco años. El
    #: `sha256_arff` detecta que dejó de serlo; detectarlo no es repararlo.
    #:
    #: **`None` por omisión NO es «este dataset no tiene fichero»**: es «el
    #: protocolo registrado es anterior a la re-firma que añade el campo».
    #: Para que esa tolerancia no se convierta en un campo a medias,
    #: `ProtocoloExploratorio.__post_init__` exige que lo traigan TODOS los
    #: datasets o NINGUNO, y `a_json` solo lo emite cuando lo hay — así el
    #: digest de hoy no se mueve, y el día que se re-firme SÍ se mueve, que es
    #: lo que mete el campo dentro del sello.
    file_id: int | None = None

    def __post_init__(self) -> None:
        _exigir(self.tarea in TAREAS, f"tarea desconocida: {self.tarea!r}")
        _exigir(self.cubo_de_tamano in CUBOS_DE_TAMANO,
               f"cubo de tamaño desconocido: {self.cubo_de_tamano!r}")
        _exigir(len(self.sha256_arff) == 64 and all(c in "0123456789abcdef"
                                                     for c in self.sha256_arff),
               f"sha256_arff no es un sha256 hexadecimal: {self.sha256_arff!r}")
        _exigir(self.n_filas > 0, "n_filas tiene que ser positivo")
        _exigir(bool(self.columna_objetivo), "columna_objetivo no puede estar vacía")
        _exigir(bool(self.licencia), "licencia no puede estar vacía — no fabricar lo que no se midió")
        _exigir(self.max_cardinalidad_nominal >= 0,
               "max_cardinalidad_nominal no puede ser negativo")
        _exigir(self.file_id is None
                or (isinstance(self.file_id, int) and not isinstance(self.file_id, bool)
                    and self.file_id > 0),
               f"file_id tiene que ser un entero positivo o no estar: {self.file_id!r}")
        # El booleano NO se calcula aquí en vez de exigirse: si se calculase,
        # un catálogo con el booleano mal escrito se «arreglaría» solo al
        # cargarlo y nadie se enteraría de que el fichero registrado miente.
        # Se exige, para que el fichero tenga que decir la verdad ÉL.
        _exigir(self.alta_cardinalidad
                == (self.max_cardinalidad_nominal >= UMBRAL_ALTA_CARDINALIDAD),
               f"{self.nombre}: alta_cardinalidad={self.alta_cardinalidad} no cuadra con "
               f"max_cardinalidad_nominal={self.max_cardinalidad_nominal} frente al umbral "
               f"{UMBRAL_ALTA_CARDINALIDAD} del anexo C §2.2")

    def a_json(self) -> dict[str, Any]:
        payload = {
            "data_id": self.data_id, "nombre": self.nombre, "fuente": self.fuente,
            "version": self.version, "sha256_arff": self.sha256_arff,
            "columna_objetivo": self.columna_objetivo, "tarea": self.tarea,
            "cubo_de_tamano": self.cubo_de_tamano, "n_filas": self.n_filas,
            "n_columnas": self.n_columnas, "tiene_faltantes": self.tiene_faltantes,
            "max_cardinalidad_nominal": self.max_cardinalidad_nominal,
            "alta_cardinalidad": self.alta_cardinalidad, "desbalanceado": self.desbalanceado,
            "solo_numericas": self.solo_numericas, "licencia": self.licencia,
            "sellado": self.sellado,
        }
        # LA CLAVE SE EMITE SOLO CUANDO HAY VALOR, y no es cosmética: `a_json`
        # es lo que se hashea (`ProtocoloExploratorio.digest`). Emitir
        # `"file_id": null` movería el digest del protocolo YA REGISTRADO sin
        # añadir ni un dato — y ese digest está citado dentro de evidencia
        # commiteada. Omitirlo deja el digest de hoy donde está y hace que el
        # día de la re-firma el digest cambie SOLO porque el campo entró.
        #
        # Lo que esto NO permite es un protocolo medio poblado: eso lo corta
        # `ProtocoloExploratorio.__post_init__`, que exige todos o ninguno.
        if self.file_id is not None:
            payload["file_id"] = self.file_id
        return payload

    @classmethod
    def desde_json(cls, payload: dict[str, Any]) -> "DatasetRegistrado":
        # `payload[k]` y no `payload.get(k)` a propósito: un catálogo anterior
        # a la re-firma no trae `max_cardinalidad_nominal`, y rellenarlo con un
        # 0 por defecto lo haría pasar por «ninguna columna nominal» —
        # **un valor ausente no es un cero**. Que reviente por su nombre.
        campos = {k: payload[k] for k in (
            "data_id", "nombre", "fuente", "version", "sha256_arff", "columna_objetivo",
            "tarea", "cubo_de_tamano", "n_filas", "n_columnas", "tiene_faltantes",
            "max_cardinalidad_nominal", "alta_cardinalidad", "desbalanceado",
            "solo_numericas", "licencia", "sellado")}
        # `file_id` SÍ con `.get`, al revés que los de arriba, y la diferencia
        # tiene motivo: ausente no significa «cero columnas nominales» ni nada
        # medible, significa «protocolo anterior a la re-firma del campo». La
        # tolerancia no abre el agujero de la mitad porque el protocolo entero
        # exige después que lo traigan todos o ninguno.
        if payload.get("file_id") is not None:
            campos["file_id"] = _file_id_entero(payload["file_id"])
        return cls(**campos)


@dataclass(frozen=True)
class Motor:
    """Un motor que compite en el ranking, y cuántas configuraciones publica.

    `configuraciones=2` es «el rival en su mejor versión» (§2.3 del anexo,
    medido: con solo `defaults` los árboles SOBREAJUSTAN en datasets
    pequeños — HGB/LightGBM/XGBoost dieron AUROC 0,775–0,789 en lluvia
    mientras LogReg con defaults ya daba 0,829 — así que compararlos solo por
    `defaults` habría sido injusto en el sentido que perjudica al árbol, no a
    MatrixAI). `dummy` no tiene hiperparámetro que buscar: una sola
    configuración es honesta, no un descuido.
    """

    id: str
    configuraciones: int = 2

    def __post_init__(self) -> None:
        _exigir(bool(self.id), "el id del motor no puede estar vacío")
        _exigir(self.configuraciones >= 1, "configuraciones tiene que ser >= 1")

    def a_json(self) -> dict[str, Any]:
        return {"id": self.id, "configuraciones": self.configuraciones}


@dataclass(frozen=True)
class DisenoDeParticion:
    """5 folds estratificados × 3 repeticiones en pequeño/mediano; en
    grande, 5 × 1 — medido que 15 ajustes por (motor, dataset) en un dataset
    de 100.000 filas ya cuesta horas por sí solo (§3.4 del anexo); repetirlo
    3 veces multiplicaría el coste sin cambiar la conclusión del cubo grande,
    que es de escala, no de varianza de muestreo."""

    folds: int = 5
    repeticiones_pequeno_mediano: int = 3
    repeticiones_grande: int = 1
    semillas: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        _exigir(self.folds >= 2, "folds tiene que ser >= 2")
        _exigir(self.repeticiones_pequeno_mediano >= 1, "repeticiones_pequeno_mediano >= 1")
        _exigir(self.repeticiones_grande >= 1, "repeticiones_grande >= 1")
        _exigir(len(self.semillas) >= self.repeticiones_pequeno_mediano,
               "hacen falta tantas semillas registradas como repeticiones máximas")

    def repeticiones_para(self, cubo: str) -> int:
        _exigir(cubo in CUBOS_DE_TAMANO, f"cubo de tamaño desconocido: {cubo!r}")
        return self.repeticiones_grande if cubo == "grande" else self.repeticiones_pequeno_mediano

    def a_json(self) -> dict[str, Any]:
        return {"folds": self.folds, "repeticiones_pequeno_mediano": self.repeticiones_pequeno_mediano,
                "repeticiones_grande": self.repeticiones_grande, "semillas": list(self.semillas)}


@dataclass(frozen=True)
class PresupuestoPorCubo:
    """Lo que un ajuste puede gastar: minutos de reloj, ESCALADOS por cubo de
    tamaño (invariante 5 del 101: «por motor y dataset/pliegue» para estudiar
    motores) — nunca un único número fijo para 500 filas y 100.000 filas a
    la vez, que es justo la mezcla que la primera versión del 101 tenía y
    esta revisión corrige."""

    minutos_por_cubo: dict[str, float]
    hilos: int = 4
    procesos_en_paralelo: int = 6

    def __post_init__(self) -> None:
        for cubo in CUBOS_DE_TAMANO:
            _exigir(cubo in self.minutos_por_cubo, f"falta el presupuesto del cubo {cubo!r}")
            _exigir(self.minutos_por_cubo[cubo] > 0, f"el presupuesto de {cubo!r} tiene que ser positivo")
        _exigir(self.hilos >= 1, "hilos tiene que ser >= 1")
        _exigir(self.procesos_en_paralelo >= 1, "procesos_en_paralelo tiene que ser >= 1")

    @property
    def hilos_reservados(self) -> int:
        """Los hilos que la pasada reserva A LA VEZ: hilos x procesos. El
        protocolo re-firmado el 2026-09-13 da 4 x 2 = **8**, que es lo que la
        máquina tiene. Antes daba 4 x 6 = 24, la sobre-reserva que el
        101-C1/C2 dejó apuntada como deuda y que la re-firma cerró."""
        return self.hilos * self.procesos_en_paralelo

    def sobre_reserva(self, cpus: int | None = None) -> int:
        """Hilos de MÁS sobre las CPUs que hay, o 0 si cabe. Con el protocolo
        re-firmado y esta máquina: 8 - 8 = **0**. Con el registrado
        originalmente eran 24 - 8 = 16 de más."""
        disponibles = cpus if cpus is not None else cpus_disponibles()
        return max(0, self.hilos_reservados - disponibles)

    def procesos_que_caben(self, cpus: int | None = None) -> int:
        """Los procesos que cabrían con estos `hilos` sin pasarse. No cambia
        el protocolo cargado —eso sería otro protocolo con otro digest—: dice
        lo que habría que poner. Es la función con la que se midió el 2 de la
        re-firma, no un número escrito a mano encima de ella."""
        return reserva_segura(self.hilos, cpus)

    # `a_json` NO lleva los campos de arriba a propósito: son DERIVADOS de
    # `hilos`/`procesos_en_paralelo` y de la máquina, no decisiones del
    # protocolo. Meterlos aquí cambiaría el digest registrado (invariante 1) y
    # lo haría depender del ordenador donde se serializa.
    def a_json(self) -> dict[str, Any]:
        return {"minutos_por_cubo": dict(self.minutos_por_cubo), "hilos": self.hilos,
                "procesos_en_paralelo": self.procesos_en_paralelo}


@dataclass(frozen=True)
class ReglaDeCierre:
    """«A menos de X puntos del mejor en ≥ Y % de los datasets» — el
    criterio 117 del anexo, con X e Y fijados ANTES de medir (invariante 1).
    `metrica_por_tarea` dice de qué escala son los «puntos» según la tarea:
    AUROC en binaria, accuracy (o F1-macro) en multiclase, R² en regresión —
    nunca sumar distancias absolutas de métricas de escalas distintas."""

    puntos: float
    fraccion_minima: float
    metrica_por_tarea: dict[str, str]
    definicion_de_mejor: str

    def __post_init__(self) -> None:
        _exigir(self.puntos > 0, "puntos tiene que ser positivo")
        _exigir(0.0 < self.fraccion_minima <= 1.0, "fraccion_minima tiene que estar en (0, 1]")
        for tarea in TAREAS:
            _exigir(tarea in self.metrica_por_tarea, f"falta la métrica de {tarea!r}")
        _exigir(bool(self.definicion_de_mejor), "definicion_de_mejor no puede estar vacía")

    def a_json(self) -> dict[str, Any]:
        return {"puntos": self.puntos, "fraccion_minima": self.fraccion_minima,
                "metrica_por_tarea": dict(self.metrica_por_tarea),
                "definicion_de_mejor": self.definicion_de_mejor}


#: La semilla del remuestreo de `_intervalo_pareado`. Escrita, no elegida en
#: cada llamada: un intervalo que cambia de límites cada vez que se recalcula
#: no se puede citar, y «el que salió» sería el que a uno le conviniera.
SEMILLA_DEL_INTERVALO = 0

#: Remuestras, las mismas 1.000 que `metricas_por_tarea.intervalos` del
#: protocolo registrado ya declaraba.
REMUESTRAS_DEL_INTERVALO = 1000


def _intervalo_pareado(diferencias: Sequence[float], *,
                       semilla: int = SEMILLA_DEL_INTERVALO,
                       remuestras: int = REMUESTRAS_DEL_INTERVALO,
                       nivel: float = 0.95) -> tuple[float, float] | None:
    """Bootstrap de percentiles sobre diferencias YA EMPAREJADAS.

    **Emparejado, no dos intervalos sueltos.** Los dos motores se miden sobre
    los MISMOS pliegues de los MISMOS datos: sus aciertos y sus fallos están
    correlacionados, y remuestrear cada uno por su cuenta para ver si los
    intervalos se solapan ignora esa correlación y es sistemáticamente
    conservador. Es la misma razón —y la misma aritmética— que
    `matrixai.estudio.comparaciones` (105-C5): se resta ANTES de guardar.

    **Qué se reutiliza y qué no, dicho por su nombre.** El percentil es
    `_percentil` de 105-C2, importado, no reescrito: interpolar entre dos
    posiciones tiene una convención y dos implementaciones divergen. Lo que
    NO se puede reutilizar es `intervalo()` entera: pide una `Muestra` de
    filas (verdad y predicción por observación) y aquí no hay filas — los
    registros crudos de esta pasada guardan UNA métrica ya agregada por
    pliegue, no las predicciones fuera-de-fold. El remuestreo es por tanto de
    PLIEGUES, no de observaciones, y eso no es lo mismo que el
    `metricas_por_tarea.intervalos` del protocolo: es más grueso, y por eso
    el campo que lo publica dice de qué está hecho en vez de llamarse
    «intervalo» a secas.

    Devuelve `None` con menos de dos diferencias: con una sola no hay nada
    que remuestrear, y un intervalo de anchura cero ahí sería una afirmación
    de precisión que el dato no sostiene.
    """
    if len(diferencias) < 2:
        return None
    from matrixai.estudio.incertidumbre import _percentil

    azar = random.Random(semilla)
    n = len(diferencias)
    medias: list[float] = []
    for _ in range(remuestras):
        # La MISMA línea que la rama no-clasificación de `_indices_iid`
        # (105-C2): índices con reemplazo sobre las n unidades. Aquí la unidad
        # es el pliegue, no la fila, y por eso no se puede llamar a aquella
        # función —pide una `Muestra`— sin fabricar un objeto que mentiría
        # sobre lo que se está remuestreando.
        indices = azar.choices(range(n), k=n)
        medias.append(sum(diferencias[i] for i in indices) / n)
    medias.sort()
    cola = (1.0 - nivel) / 2.0
    return (_percentil(medias, cola), _percentil(medias, 1.0 - cola))


def _intervalo_de_la_distancia(medidas_por_pliegue: Mapping[str, Mapping[tuple, float]], *,
                               motor: str, mejor: str | None,
                               puntos: float) -> dict[str, Any] | None:
    """El intervalo de la DISTANCIA de `motor` al `mejor`, en los mismos
    puntos porcentuales que `distancia_en_puntos` — y si el listón cae dentro.

    Devuelve `None`, y no un intervalo cualquiera, cuando no hay nada que
    emparejar: sin un mejor declarado, cuando el mejor ES `motor` (la
    distancia es 0,0000 por construcción y un intervalo alrededor de una
    identidad no dice nada), o con menos de dos pliegues comunes. Un intervalo
    inventado donde no hay datos es peor que ninguno: se lee igual que uno
    medido.

    `cruza_el_liston` es el campo entero de este añadido: con él, «no cumple»
    y «no cumple y el listón está dentro del intervalo» dejan de escribirse
    igual.
    """
    if mejor is None or mejor == motor:
        return None
    mios = medidas_por_pliegue.get(motor) or {}
    suyos = medidas_por_pliegue.get(mejor) or {}
    comunes = sorted(set(mios) & set(suyos))
    if len(comunes) < 2:
        return None
    # La MISMA escala que `distancia_en_puntos`: (mejor - mio) x 100.
    diferencias = [(suyos[k] - mios[k]) * 100.0 for k in comunes]
    limites = _intervalo_pareado(diferencias)
    if limites is None:
        return None
    bajo, alto = limites
    return {
        "bajo": bajo, "alto": alto, "nivel": 0.95,
        "unidad": "puntos porcentuales de la metrica, igual que distancia_en_puntos",
        "emparejado_por": "repeticion y pliegue",
        "n_pliegues_emparejados": len(comunes),
        "remuestreo": (f"bootstrap de percentiles, {REMUESTRAS_DEL_INTERVALO} remuestras "
                       f"de PLIEGUES (no de filas: los registros crudos guardan la "
                       f"metrica ya agregada por pliegue), semilla "
                       f"{SEMILLA_DEL_INTERVALO}"),
        "contra": mejor,
        "cruza_el_liston": bajo <= puntos <= alto,
    }


#: LOS ESTADOS QUE CUENTAN COMO MEDIDA — hallazgo M4, 2026-09-16.
#:
#: La regla registrada dice, con estas palabras, «un fallo (**timeout/crash**)
#: cuenta como dataset perdido». `completed_budget_limited` no es ninguna de las
#: dos: es un motor que **entrego predictor e informe** tras limitarse para
#: caber en el presupuesto, y `matrixai_engines.motor` lo documenta como LA
#: forma de cooperar con el tope — justo lo que se espera de un motor pesado en
#: el cubo grande. Tratarlo como fallo contradice el texto que se sello.
#:
#: **SE ARREGLA HOY PRECISAMENTE PORQUE HOY NO PUEDE CAMBIAR NADA**: medido
#: sobre la pasada de los 40, los 3.479 registros son `completed` (3.464) o
#: `failed` (15), y **ningun motor emite este estado todavia**. Hacerlo ahora es
#: alinear el codigo con la regla registrada; hacerlo el dia que un motor lo
#: emita seria cambiar la regla despues de ver los numeros. Hay una prueba que
#: fija que sobre el artefacto real el veredicto es identico.
#:
#: Es una LISTA BLANCA y no un `startswith("completed")`: un estado nuevo tiene
#: que entrar aqui a mano, con alguien mirando si de verdad trae medida. Un
#: prefijo aceptaria en silencio cualquier `completed_lo_que_sea` futuro.
ESTADOS_QUE_CUENTAN_COMO_MEDIDA: frozenset[str] = frozenset({
    "completed",
    "completed_budget_limited",
})


def metrica_del_dataset(dataset: str, metrica: str,
                        metrica_por_dataset: Mapping[str, str] | None) -> str | None:
    """Con qué métrica se mide ESTE dataset.

    Sin mapa, la de siempre para todos —el comportamiento con el que se
    midieron los doce de 101-C3, que no se mueve—. Con mapa, la suya; y si el
    mapa no lo nombra devuelve `None`, que el llamante tiene que DECLARAR en
    vez de tragarse. Un dataset sin métrica declarada no es un dataset que
    cumpla ni uno que falle: es uno del que no se sabe, y los tres se escriben
    distinto.
    """
    if metrica_por_dataset is None:
        return metrica
    return metrica_por_dataset.get(dataset)


def aplicar_regla_de_cierre(resultados: Sequence[dict], regla: ReglaDeCierre, *,
                            motor: str, metrica: str = "auroc",
                            baseline: str = "baseline",
                            metrica_por_dataset: Mapping[str, str] | None = None,
                            datasets_exigidos: Sequence[str] | None = None,
                            ) -> dict[str, Any]:
    """¿`motor` queda a menos de `regla.puntos` del mejor en al menos
    `regla.fraccion_minima` de los datasets?

    **UNA MÉTRICA PARA TODOS LOS DATASETS SOLO VALE SI TODOS SON DE LA MISMA
    TAREA, y esto lo daba por hecho (2026-09-14, al construir 101-C5).** La
    `regla_de_cierre` registrada trae `metrica_por_tarea` desde 101-C1 —AUROC
    en binaria, accuracy/F1-macro en multiclase, R² en regresión— y esta
    función recibía UN nombre de métrica para los cuarenta datasets. Medido
    sobre registros de las tres tareas con `metrica="auroc"`:

      · con todos los intentos completados, los datasets de multiclase y de
        regresión **desaparecen del denominador sin dejar rastro** (entran
        tres, sale «1/1 = 1,000 CUMPLE»);
      · basta UN intento fallido en uno de ellos para que **reaparezca, y
        contado como perdido** («1/2 = 0,500»).

    O sea que el denominador de «cumple en el 80 % de los datasets» dependía
    de si un motor ajeno se había caído, y en la dirección tranquilizadora
    cuando no se caía nadie. `metrica_por_dataset` lo cierra: cada dataset se
    mide con la métrica de SU tarea.

    `datasets_exigidos` es la otra mitad, y sin ella la primera no basta: un
    dataset en el que NINGÚN motor produjo la métrica sigue cayéndose del
    recuento porque no aparece en los registros con valor. Pasando la lista
    declarada, aparece en el detalle con `sin_medida` y su motivo — un aserto
    negativo lo pasa un `{}`, y un denominador que encoge solo se lee igual de
    bien que uno correcto.

    **Los dos parámetros son opcionales y su ausencia NO cambia nada.** Con
    los dos a `None` esta función devuelve exactamente el mismo diccionario
    que devolvía antes, con las mismas claves y sin ninguna nueva: los
    artefactos de 101-C3 ya commiteados cuadran con su digest y tienen que
    seguir cuadrando (`test_sin_los_parametros_nuevos_el_veredicto_es_identico`).

    AUDITORÍA PROPIA 2026-09-11: esta regla estaba REGISTRADA CON HASH desde
    101-C1 —fijada antes de medir, que es justo su razón de ser (invariante
    1)— y NINGÚN código la aplicaba. Ni `pasada_exploratoria_101_c3.py` ni
    `informe_101_c4.py` la invocan: la conclusión «LightGBM gana» se declaró
    sin pasar por el listón que el propio protocolo se había puesto, y que el
    contrato justifica porque «1 punto no es resoluble» con el IC medido.

    `definicion_de_mejor` de la regla registrada dice, con todas las letras:
    «el motor con mejor media de los ajustes en ESE dataset, excluido el
    baseline dummy; **un fallo (timeout/crash) cuenta como dataset perdido
    para ese motor**». Eso se implementa tal cual: un dataset donde `motor`
    tenga algún intento fallido NO cuenta como cumplido, por buena que sea la
    media de los que sí completaron. Interpretarlo de otro modo sería elegir
    la lectura que da el resultado que conviene, después de ver los números.

    `resultados` son los registros crudos de la pasada (un dict por intento,
    con `dataset`, `motor`, `estado` y la métrica).
    """
    por_dataset: dict[str, dict[str, list[float]]] = {}
    fallos: dict[str, set] = {}
    # Las mismas medidas, indexadas por el pliegue que las produjo. Es lo que
    # permite EMPAREJAR dos motores antes de restarlos; la media de arriba no
    # se toca y sigue saliendo de la lista de siempre.
    por_pliegue: dict[str, dict[str, dict[tuple, float]]] = {}
    sin_pliegue = 0
    #: Los datasets que el mapa por tarea no nombra. Se cuentan y se declaran:
    #: tragárselos es volver al denominador que encoge solo.
    sin_metrica_declarada: set[str] = set()
    #: Con qué se midió cada dataset de verdad, para que el artefacto no
    #: obligue a recomponer el mapa desde fuera para entenderse.
    metrica_usada_por_dataset: dict[str, str] = {}
    for r in resultados:
        ds, mt = r["dataset"], r["motor"]
        if r.get("estado") not in ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
            fallos.setdefault(ds, set()).add(mt)
            continue
        metrica_de_este = metrica_del_dataset(ds, metrica, metrica_por_dataset)
        if metrica_de_este is None:
            sin_metrica_declarada.add(ds)
            continue
        metrica_usada_por_dataset[ds] = metrica_de_este
        valor = r.get(metrica_de_este)
        if valor is None:
            continue
        por_dataset.setdefault(ds, {}).setdefault(mt, []).append(float(valor))
        if r.get("repeticion") is None or r.get("pliegue") is None:
            sin_pliegue += 1
            continue
        por_pliegue.setdefault(ds, {}).setdefault(mt, {})[
            (r["repeticion"], r["pliegue"])] = float(valor)

    detalle: list[dict[str, Any]] = []
    cumplidos = 0
    #: Modo ampliado: solo cuando quien llama ha pedido una de las dos cosas
    #: nuevas. Sin eso, ni una clave de más — los artefactos de C3 ya escritos
    #: cuadran con su digest y tienen que seguir cuadrando.
    ampliado = metrica_por_dataset is not None or datasets_exigidos is not None
    exigidos = set(datasets_exigidos or ())
    for ds in sorted(set(por_dataset) | set(fallos) | exigidos):
        medias = {m: sum(v) / len(v) for m, v in por_dataset.get(ds, {}).items()
                  if m != baseline}
        perdido = motor in fallos.get(ds, set())
        mejor_motor = max(medias, key=lambda m: medias[m]) if medias else None
        mio = medias.get(motor)
        # Los «puntos» son de la escala de la métrica (AUROC 0-1), y la regla
        # los declara en PUNTOS porcentuales: 2,0 puntos = 0,02 de AUROC.
        distancia = None if (mio is None or mejor_motor is None) else (
            (medias[mejor_motor] - mio) * 100.0)
        cumple = (not perdido) and distancia is not None and distancia <= regla.puntos
        cumplidos += 1 if cumple else 0
        # LA VENTAJA SOBRE EL SEGUNDO, y por qué no basta con `distancia`.
        # Auditoría del 2026-09-12: `distancia_en_puntos` vale 0,0000 SIEMPRE
        # que `motor` sea el mejor, así que «ganó de calle» y «ganó por la
        # décima de una décima» se escriben IGUAL. Sobre la re-medición limpia
        # eso son OCHO de los diez aciertos de lightgbm: leídos sin esta
        # columna, los ocho dicen lo mismo que un dominio real.
        #
        # Se declara solo cuando `motor` ES el mejor —si no lo es, `distancia`
        # ya dice todo lo que hay que saber— y va en las mismas unidades:
        # puntos porcentuales de la métrica.
        ordenados = sorted(medias, key=lambda m: medias[m], reverse=True)
        segundo = ordenados[1] if len(ordenados) > 1 else None
        ventaja = None
        if mejor_motor == motor and segundo is not None:
            ventaja = (medias[motor] - medias[segundo]) * 100.0
        # LA INCERTIDUMBRE DEL NÚMERO QUE DECIDE, y por qué el veredicto sin
        # ella se lee más holgado de lo que es.
        #
        # Auditoría del 2026-09-13: la ventaja de catboost sobre lightgbm en
        # `pc1` son 2,926 puntos con IC95 emparejado [1,66 · 4,20] — un
        # intervalo que CRUZA el listón de 2,0. El veredicto según la regla
        # registrada es correcto e inequívoco (2,926 > 2,0, y la regla es la
        # que es), pero publicarlo sin el intervalo hace que «no cumple» y «no
        # cumple, y con estos 15 pliegues no se distingue de cumplir» se lean
        # exactamente igual.
        #
        # Esto NO cambia la regla ni su veredicto: `cumple` se calcula arriba,
        # con la distancia puntual y el listón registrados, y ni lo mira. Lo
        # que añade es el dato que faltaba para leerlo.
        intervalo = _intervalo_de_la_distancia(
            por_pliegue.get(ds, {}), motor=motor, mejor=mejor_motor, puntos=regla.puntos)
        entrada = {"dataset": ds, "cumple": cumple, "perdido_por_fallo": perdido,
                   "mejor": mejor_motor, "distancia_en_puntos": distancia,
                   "segundo": segundo if mejor_motor == motor else None,
                   "ventaja_sobre_el_segundo_en_puntos": ventaja,
                   "intervalo_de_la_distancia": intervalo,
                   "motores_que_compitieron": sorted(medias)}
        if ampliado:
            # POR QUÉ este dataset no tiene número, cuando no lo tiene. Las
            # tres ausencias se escriben distinto porque son distintas: no se
            # declaró con qué medirlo, se declaró y ningún motor lo produjo, o
            # este motor concreto no lo produjo aunque otros sí.
            if ds in sin_metrica_declarada and ds not in metrica_usada_por_dataset:
                sin_medida = ("no se declaro con que metrica se mide este dataset: "
                              "el mapa por tarea no lo nombra")
            elif not medias and not perdido:
                sin_medida = ("ningun motor produjo la metrica de cierre de este "
                              "dataset, y no consta ningun intento fallido: el "
                              "dataset entro en el denominador y no se midio")
            elif mio is None and not perdido:
                sin_medida = (f"{motor} no produjo la metrica de cierre de este "
                              f"dataset en ningun pliegue completado")
            else:
                sin_medida = None
            entrada["metrica"] = metrica_usada_por_dataset.get(ds)
            entrada["sin_medida"] = sin_medida
        detalle.append(entrada)

    total = len(detalle)
    fraccion = (cumplidos / total) if total else 0.0
    # CUÁNTOS de los aciertos son «es el mejor», que en esta escala se escribe
    # 0,0000 — y cuántos datasets puede perder antes de bajarse del listón.
    # Los dos números van AQUÍ y no en un informe aparte: «10/12 CUMPLE» y
    # «ocho de esos diez son un empate a cero, y con uno menos no cumple» son
    # la misma frase, y separarlas es lo que permitió leer la primera sola.
    por_distancia_cero = sum(
        1 for d in detalle
        if d["cumple"] and d["distancia_en_puntos"] is not None
        and abs(d["distancia_en_puntos"]) < 1e-12)
    # El margen solo significa algo si HOY cumple: «puede perder 0» dicho de un
    # motor que ya está por debajo se lee como si estuviera al borde por arriba.
    # Cuando no cumple, el número que informa es el contrario: cuántos le faltan.
    cumple_hoy = total > 0 and (cumplidos / total) >= regla.fraccion_minima
    margen: int | None = None
    le_faltan: int | None = None
    if cumple_hoy:
        margen = 0
        while ((cumplidos - margen - 1) / total) >= regla.fraccion_minima:
            margen += 1
    elif total:
        le_faltan = 0
        while ((cumplidos + le_faltan) / total) < regla.fraccion_minima:
            le_faltan += 1
    # CUÁNTOS de los incumplidos son incumplidos «de calle» y cuántos tienen el
    # listón dentro de su intervalo. Sin este recuento hay que abrir el detalle
    # dataset por dataset para saberlo, y el número de portada («9/12 NO
    # cumple») no cambia de aspecto por mucho que los tres fallos estén al
    # borde.
    no_cumplen_con_el_liston_dentro = sum(
        1 for d in detalle
        if not d["cumple"] and (d["intervalo_de_la_distancia"] or {}).get("cruza_el_liston"))
    veredicto = {"motor": motor, "metrica": metrica, "puntos_exigidos": regla.puntos,
                 "intervalos_de_los_que_NO_cumplen_que_cruzan_el_liston":
                     no_cumplen_con_el_liston_dentro,
                 "n_medidas_sin_pliegue_declarado": sin_pliegue,
                 "fraccion_minima": regla.fraccion_minima, "datasets": total,
                 "cumplidos": cumplidos, "fraccion": fraccion,
                 "cumple_la_regla": fraccion >= regla.fraccion_minima,
                 "aciertos_por_ser_el_mejor": por_distancia_cero,
                 "datasets_que_puede_perder_sin_incumplir": margen,
                 "datasets_que_le_faltan_para_cumplir": le_faltan,
                 "definicion_de_mejor": regla.definicion_de_mejor, "detalle": detalle}
    if ampliado:
        # `metrica` de arriba sigue diciendo el valor por omisión, que con un
        # mapa por tarea NO es lo que se midió. Se deja donde estaba —quitarla
        # movería un artefacto— y al lado va lo que de verdad se usó, dataset a
        # dataset. Media verdad tranquilizadora también es media limpieza.
        veredicto["metrica_por_dataset"] = dict(metrica_usada_por_dataset)
        veredicto["datasets_sin_metrica_declarada"] = sorted(sin_metrica_declarada)
        veredicto["datasets_exigidos_que_no_se_midieron"] = sorted(
            d["dataset"] for d in detalle if d.get("sin_medida"))
        veredicto["la_metrica_de_arriba_no_gobierna"] = (
            "hay un mapa por tarea: `metrica` es el valor por omision del "
            "parametro y NO es con lo que se midio. Lo que gobierna cada "
            "dataset esta en `metrica_por_dataset`.")
    return veredicto


@dataclass(frozen=True)
class ProtocoloExploratorio:
    """El protocolo entero, registrado con hash — invariante 1 del 101."""

    version_protocolo: str
    fecha_registro: str
    datasets: tuple[DatasetRegistrado, ...]
    motores: tuple[Motor, ...]
    particion: DisenoDeParticion
    presupuesto: PresupuestoPorCubo
    regla_de_cierre: ReglaDeCierre
    recursos_declarados: dict[str, Any] = field(default_factory=dict)
    metricas_por_tarea: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _exigir(bool(self.version_protocolo), "version_protocolo no puede estar vacía")
        _exigir(len(self.datasets) > 0, "el protocolo tiene que traer al menos un dataset")
        _exigir(len(self.motores) > 0, "el protocolo tiene que traer al menos un motor")
        ids = [d.data_id for d in self.datasets]
        _exigir(len(ids) == len(set(ids)), "hay un data_id repetido en la lista de datasets")
        # `file_id`: TODOS O NINGUNO. Media limpieza es peor que ninguna — un
        # protocolo con 38 de 40 se lee como «se puede bajar por URL directa»
        # y se puede bajar 38. Ausente en los 40 es el estado legítimo de
        # antes de la re-firma; ausente en algunos es un campo roto.
        con_file_id = [d.data_id for d in self.datasets if d.file_id is not None]
        _exigir(len(con_file_id) in (0, len(self.datasets)),
               f"file_id a medias: lo traen {len(con_file_id)} de {len(self.datasets)} "
               f"datasets. O lo traen todos o ninguno — un catálogo con el campo en "
               f"parte de las filas promete algo que no cumple")

    def a_json(self) -> dict[str, Any]:
        return {
            "esquema": "matrixai.benchmarks.fase0.protocolo_exploratorio",
            "version_protocolo": self.version_protocolo,
            "fecha_registro": self.fecha_registro,
            "datasets": [d.a_json() for d in self.datasets],
            "motores": [m.a_json() for m in self.motores],
            "particion": self.particion.a_json(),
            "presupuesto": self.presupuesto.a_json(),
            "regla_de_cierre": self.regla_de_cierre.a_json(),
            "recursos_declarados": dict(self.recursos_declarados),
            "metricas_por_tarea": dict(self.metricas_por_tarea),
        }

    def digest(self) -> str:
        """El hash que invariante 1 exige registrar ANTES de la primera
        pasada. Cambiar UN campo de un dataset, un minuto de presupuesto o
        el umbral de la regla de cierre cambia este digest — es la
        comprobación de que nadie ajustó el protocolo después de mirar un
        resultado parcial."""
        return digest_canonico(self.a_json())

    @classmethod
    def desde_json(cls, payload: dict[str, Any]) -> "ProtocoloExploratorio":
        return cls(
            version_protocolo=payload["version_protocolo"],
            fecha_registro=payload["fecha_registro"],
            datasets=tuple(DatasetRegistrado.desde_json(d) for d in payload["datasets"]),
            motores=tuple(Motor(**m) for m in payload["motores"]),
            particion=DisenoDeParticion(
                folds=payload["particion"]["folds"],
                repeticiones_pequeno_mediano=payload["particion"]["repeticiones_pequeno_mediano"],
                repeticiones_grande=payload["particion"]["repeticiones_grande"],
                semillas=tuple(payload["particion"]["semillas"])),
            presupuesto=PresupuestoPorCubo(
                minutos_por_cubo=dict(payload["presupuesto"]["minutos_por_cubo"]),
                hilos=payload["presupuesto"]["hilos"],
                procesos_en_paralelo=payload["presupuesto"]["procesos_en_paralelo"]),
            regla_de_cierre=ReglaDeCierre(**payload["regla_de_cierre"]),
            recursos_declarados=dict(payload.get("recursos_declarados") or {}),
            metricas_por_tarea=dict(payload.get("metricas_por_tarea") or {}),
        )

    @classmethod
    def cargar(cls, ruta: str | Path) -> "ProtocoloExploratorio":
        import json
        payload = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls.desde_json(payload)


@dataclass(frozen=True)
class CosteDeLaPasada:
    """El resultado de `calcular_coste`: número de ejecuciones y cota de
    coste, NUNCA una estimación de tiempo real — es el PEOR CASO si todos
    los ajustes agotasen su presupuesto, que es justo lo que el presupuesto
    garantiza que no se supera (criterio de terminado del 101-C1: «cota de
    coste», no «tiempo esperado»)."""

    total_ejecuciones: int
    ejecuciones_por_cubo: dict[str, int]
    horas_reloj_peor_caso_secuencial: float
    horas_reloj_peor_caso_con_paralelismo: float
    dias_peor_caso_con_paralelismo: float
    procesos_en_paralelo: int
    # --- 101-C1/C2, sobre-reserva: lo que la división lineal no decía ---
    hilos_reservados: int = 0
    cpus_disponibles: int = 0
    sobre_reserva: int = 0
    speedup_efectivo_medido: float = 0.0
    horas_reloj_suelo_medido: float = 0.0

    def a_json(self) -> dict[str, Any]:
        return {
            "total_ejecuciones": self.total_ejecuciones,
            "ejecuciones_por_cubo": dict(self.ejecuciones_por_cubo),
            "horas_reloj_peor_caso_secuencial": self.horas_reloj_peor_caso_secuencial,
            "horas_reloj_peor_caso_con_paralelismo": self.horas_reloj_peor_caso_con_paralelismo,
            "dias_peor_caso_con_paralelismo": self.dias_peor_caso_con_paralelismo,
            "procesos_en_paralelo": self.procesos_en_paralelo,
            "hilos_reservados": self.hilos_reservados,
            "cpus_disponibles": self.cpus_disponibles,
            "sobre_reserva": self.sobre_reserva,
            "speedup_efectivo_medido": self.speedup_efectivo_medido,
            "horas_reloj_suelo_medido": self.horas_reloj_suelo_medido,
        }


def calcular_coste(protocolo: ProtocoloExploratorio,
                   cpus: int | None = None) -> CosteDeLaPasada:
    """Número de ejecuciones y cota de coste, consistente con motores,
    pliegues, repeticiones y presupuesto — el criterio de terminado literal
    del 101-C1.

    Una «ejecución» es UN ajuste: un (dataset, pliegue, repetición, motor,
    configuración). El peor caso supone que CADA ejecución agota su
    presupuesto — no es lo que se espera que pase (la mayoría termina antes),
    es la COTA que ningún run individual puede superar sin contarse como
    fallo (`completed_budget_limited`/`failed` por timeout, C2).

    **Los dos números de reloj dicen cosas distintas, y por eso están los
    dos** (101-C1/C2, sobre-reserva de concurrencia):

    - `horas_reloj_peor_caso_con_paralelismo` divide LINEALMENTE entre
      `procesos_en_paralelo`, como si N procesos fueran N veces más rápidos.
      Con el protocolo re-firmado (2 procesos) da **224,79 h**.
    - `horas_reloj_suelo_medido` usa el escalado MEDIDO (`speedup_medido`) y
      da **95,86 h**, el mismo número que antes de la re-firma: 8 hilos sobre
      8 CPUs rinden el techo medido 4,69x, y 24 hilos sobre esas mismas 8
      CPUs rendían exactamente lo mismo — reservar tres veces más no sacaba
      ni un minuto, solo arriesgaba la máquina.

    **LA RE-FIRMA DEL 2026-09-13 INVIRTIÓ LA RELACIÓN ENTRE LOS DOS, y eso es
    lo que hay que mirar.** Con 6 procesos, el lineal prometía 74,93 h y el
    suelo medido eran 95,86: el número publicado era **inalcanzable**, exigía
    un 6x que esta máquina no puede dar. Con 2 procesos el lineal (224,79 h)
    va POR ENCIMA del suelo medido (95,86 h), que es como tiene que ser: una
    cota de peor caso que la máquina puede batir, no una promesa que no puede
    cumplir. El 74,93 h queda donde debe quedar, en el historial.

    `horas_reloj_suelo_medido` sigue siendo un SUELO, no un techo: las anclas
    se midieron con 1 hilo por proceso, que escala mejor (3,60x) que los 4
    hilos por proceso que el protocolo pide (2,53x medido), así que la pasada
    real irá más lenta, no más rápida.

    `sobre_reserva` > 0 es la señal de que el protocolo pide más hilos que
    CPUs hay. Este cálculo lo DECLARA, no lo corrige: bajar
    `procesos_en_paralelo` a los `procesos_que_caben` cambia el presupuesto y
    por tanto el digest, y eso es re-firmar el protocolo A PROPÓSITO
    (invariante 1) — lo que se hizo el 2026-09-13—, nunca una decisión que
    esta función tome sola al calcular.
    """
    total = 0
    por_cubo: dict[str, int] = {c: 0 for c in CUBOS_DE_TAMANO}
    horas_secuencial = 0.0
    for ds in protocolo.datasets:
        repeticiones = protocolo.particion.repeticiones_para(ds.cubo_de_tamano)
        pliegues_x_repeticiones = protocolo.particion.folds * repeticiones
        minutos = protocolo.presupuesto.minutos_por_cubo[ds.cubo_de_tamano]
        for motor in protocolo.motores:
            n = pliegues_x_repeticiones * motor.configuraciones
            total += n
            por_cubo[ds.cubo_de_tamano] += n
            horas_secuencial += n * minutos / 60.0

    # La división LINEAL, tal como estaba: supone que N procesos van N veces
    # más rápido. Se CONSERVA —la fórmula, no el número— para que el par
    # «lineal vs medido» siga siendo comparable entre el protocolo de antes de
    # la re-firma y el de después: quitarla dejaría el 74,93 h inalcanzable
    # sin nada que lo contradiga.
    horas_paralelo = horas_secuencial / protocolo.presupuesto.procesos_en_paralelo

    presupuesto = protocolo.presupuesto
    disponibles = cpus if cpus is not None else cpus_disponibles()
    reservados = presupuesto.hilos_reservados
    speedup = speedup_medido(reservados, disponibles)
    horas_suelo = horas_secuencial / speedup

    return CosteDeLaPasada(
        total_ejecuciones=total,
        ejecuciones_por_cubo=por_cubo,
        horas_reloj_peor_caso_secuencial=round(horas_secuencial, 2),
        horas_reloj_peor_caso_con_paralelismo=round(horas_paralelo, 2),
        dias_peor_caso_con_paralelismo=round(horas_paralelo / 24.0, 2),
        procesos_en_paralelo=presupuesto.procesos_en_paralelo,
        hilos_reservados=reservados,
        cpus_disponibles=disponibles,
        sobre_reserva=presupuesto.sobre_reserva(disponibles),
        speedup_efectivo_medido=speedup,
        horas_reloj_suelo_medido=round(horas_suelo, 2),
    )


# ---------------------------------------------------------------------------
# EL ALCANCE DE UNA PASADA — «10/12 CUMPLE» no se puede leer sin él
# ---------------------------------------------------------------------------
# Auditoría interna del 2026-09-12, clasificada GRAVE: el protocolo registrado
# pide 40 datasets y 7 motores; la pasada exploratoria mide 12 y 4. El script
# declaraba el subconjunto de datasets en un comentario, y el recorte de
# motores NO LO DECLARABA EN NINGUNA PARTE — y los tres que faltan
# (`sklearn.hgb`, `xgboost`, `catboost`) son los rivales DIRECTOS de un GBM.
#
# El problema no es el recorte: una pasada exploratoria recorta, y es legítimo.
# El problema es que el ARTEFACTO no llevaba escrito su propio alcance, así que
# «lightgbm 10/12 = 0,833 CUMPLE» se leía sin enterarse de que compitió contra
# tres motores y no contra seis.
#
# Por eso esto vive en el mismo módulo que la regla y viaja en el MISMO
# fichero que el veredicto: dos ficheros que se pueden separar se separan.


#: Los ids del protocolo NO son los ids implementados, y esa es exactamente la
#: trampa que haría inútil una comparación literal: `baseline` es el `dummy`
#: del protocolo (mayoritaria/distribución de train: no mira las entradas) y
#: `sklearn.lineal` es su `sklearn.logreg` (logística). Comparadas a pelo, las
#: dos listas dirían que faltan CINCO motores, y ese número sería falso en la
#: dirección alarmista — que miente igual que el tranquilizador.
#:
#: Se declara aquí, en un solo sitio, en vez de resolverlo a ojo en cada
#: llamante. Un motor implementado que no aparezca en este mapa se compara por
#: su propio nombre, que es lo correcto para `lightgbm` y
#: `matrixai.dense.torch_cpu`: se llaman igual en los dos sitios.
EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR = {
    "baseline": "dummy",
    "sklearn.lineal": "sklearn.logreg",
}


def _id_en_el_protocolo(motor_implementado: str) -> str:
    return EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR.get(motor_implementado, motor_implementado)


def _alcance_de_las_configuraciones(
        protocolo: "ProtocoloExploratorio", resultados: Sequence[dict], *,
        motores_declarados: Sequence[str],
        configuraciones_declaradas: Mapping[str, Sequence[str]] | None,
        criterio_de_las_configuraciones: str) -> dict[str, Any]:
    """El TERCER recorte, que el bloque de alcance se callaba.

    AUDITORÍA DEL 2026-09-13: el protocolo registrado declara
    `configuraciones: 2` por motor (1 para el dummy, 13 en total), y su propio
    modelo de coste **solo cuadra con 13** — `calcular_coste` da 6.500
    ejecuciones, que son 13 x (30 datasets x 15 ajustes + 10 x 5), y con una
    configuración por motor darían 3.500. La pasada de siete motores del
    2026-09-13 corrió UNA por motor, 7 de 13; el script lo declaraba en su
    docstring —o sea, en el código— y el ARTEFACTO no lo decía en ninguna
    parte.

    Y es el recorte que más pesa de los tres, porque la `definicion_de_mejor`
    registrada dice «el motor con mejor media **de los ajustes**»: aplicarla
    sobre una sola familia de ajuste por motor no es lo que el protocolo
    describe. Exactamente la misma clase de defecto que el recorte de motores
    del 2026-09-12 —«lightgbm 10/12 CUMPLE» leído sin saber que tres de los
    siete rivales no corrieron—, y se repara igual: el número viaja con su
    alcance.

    `observadas` sale del campo `configuracion` de los registros. Un fichero
    anterior a que ese campo existiera NO se convierte en «ninguna
    configuración corrió»: se declara `observable=False` y las listas quedan
    vacías **con su motivo al lado**, porque un aserto negativo lo pasa un
    diccionario vacío.
    """
    del_protocolo = {m.id: m.configuraciones for m in protocolo.motores}
    n_del_protocolo = sum(del_protocolo.values())

    declaradas = {m: sorted(configuraciones_declaradas.get(m, ()))
                  for m in motores_declarados} if configuraciones_declaradas else {}
    n_que_corrieron = sum(len(v) for v in declaradas.values())

    con_campo = [r for r in resultados if r.get("configuracion") is not None]
    observable = bool(con_campo)
    observadas: dict[str, list[str]] = {}
    for r in con_campo:
        observadas.setdefault(r["motor"], [])
        if r["configuracion"] not in observadas[r["motor"]]:
            observadas[r["motor"]].append(r["configuracion"])
    observadas = {m: sorted(v) for m, v in sorted(observadas.items())}

    # Cuántas le faltan a CADA motor, por su id del protocolo. Un total solo
    # («7 de 13») no dice si el recorte cayó repartido o entero sobre uno.
    que_faltan: dict[str, int] = {}
    for motor in motores_declarados:
        pedidas = del_protocolo.get(_id_en_el_protocolo(motor))
        if pedidas is None:
            continue
        que_faltan[_id_en_el_protocolo(motor)] = max(0, pedidas - len(declaradas.get(motor, ())))

    return {
        "criterio_de_las_configuraciones": criterio_de_las_configuraciones,
        "del_protocolo": del_protocolo,
        "n_del_protocolo": n_del_protocolo,
        "declaradas_por_la_pasada": declaradas,
        "n_que_corrieron": n_que_corrieron,
        "que_faltan_por_motor": que_faltan,
        "observadas_en_los_resultados": observadas,
        "observable_en_los_registros": observable,
        "por_que_no_es_observable": ("" if observable else
                                     "ninguno de los registros trae el campo "
                                     "`configuracion`: son de antes de que la pasada lo "
                                     "escribiera, asi que de aqui no se sigue que "
                                     "corrieran cero configuraciones -- se sigue que no "
                                     "se puede saber cuales corrieron"),
    }


def alcance_de_una_pasada(protocolo: "ProtocoloExploratorio",
                          resultados: Sequence[dict], *,
                          motores_declarados: Sequence[str],
                          datasets_declarados: Sequence[str],
                          criterio_del_subconjunto: str = "",
                          configuraciones_declaradas: Mapping[str, Sequence[str]] | None = None,
                          criterio_de_las_configuraciones: str = "") -> dict[str, Any]:
    """Qué pidió el protocolo, qué dijo la pasada que iba a correr, y qué
    aparece DE VERDAD en los registros. Las tres cosas, porque cada par
    detecta un fallo distinto:

    * declarado vs protocolo → el recorte, que es lo que no se declaraba;
    * declarado vs observado → la lista que se tocó sin tocar la declaración,
      o el motor que falló entero y no dejó un solo registro.

    `observados` sale de las CLAVES REALES de `resultados`, nunca de una lista
    escrita a mano: una lista copiada es la que acaba divergiendo.

    **TRES EJES, no dos (2026-09-13).** El protocolo recorta por datasets, por
    motores **y por configuraciones**, y hasta hoy este bloque declaraba los
    dos primeros y se callaba el tercero — con el agravante de que es el que
    toca la `definicion_de_mejor` de la regla. Ver
    `_alcance_de_las_configuraciones`.
    """
    motores_protocolo = [m.id for m in protocolo.motores]
    datasets_protocolo = [d.nombre for d in protocolo.datasets]

    motores_observados = sorted({r["motor"] for r in resultados})
    datasets_observados = sorted({r["dataset"] for r in resultados})

    declarados_en_ids_del_protocolo = [_id_en_el_protocolo(m) for m in motores_declarados]
    motores_que_faltan = [m for m in motores_protocolo
                          if m not in declarados_en_ids_del_protocolo]
    datasets_que_faltan = [d for d in datasets_protocolo if d not in datasets_declarados]

    # Lo que el recorte deja ENTERAMENTE sin medir. Un motor menos se ve
    # contando; una TAREA sin un solo dataset no se ve en ningún recuento, y es
    # más grave: de esta pasada no se sigue absolutamente nada sobre regresión
    # ni multiclase, por muy alto que sea el 0,833.
    corridos = {d.nombre for d in protocolo.datasets if d.nombre in datasets_declarados}
    tareas_cubiertas = {d.tarea for d in protocolo.datasets if d.nombre in corridos}
    cubos_cubiertos = {d.cubo_de_tamano for d in protocolo.datasets if d.nombre in corridos}

    return {
        "protocolo": {
            "version": protocolo.version_protocolo,
            "fecha_registro": protocolo.fecha_registro,
            "digest_sha256": protocolo.digest(),
            "n_datasets": len(datasets_protocolo),
            "n_motores": len(motores_protocolo),
        },
        "motores": {
            "del_protocolo": motores_protocolo,
            "declarados_por_la_pasada": list(motores_declarados),
            "observados_en_los_resultados": motores_observados,
            "equivalencias_de_nombre": dict(EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR),
            "que_faltan": motores_que_faltan,
            "n_del_protocolo": len(motores_protocolo),
            "n_que_corrieron": len(motores_declarados),
        },
        "datasets": {
            "criterio_del_subconjunto": criterio_del_subconjunto,
            "del_protocolo": datasets_protocolo,
            "declarados_por_la_pasada": list(datasets_declarados),
            "observados_en_los_resultados": datasets_observados,
            "que_faltan": datasets_que_faltan,
            "n_del_protocolo": len(datasets_protocolo),
            "n_que_corrieron": len(datasets_declarados),
        },
        "configuraciones": _alcance_de_las_configuraciones(
            protocolo, resultados, motores_declarados=motores_declarados,
            configuraciones_declaradas=configuraciones_declaradas,
            criterio_de_las_configuraciones=criterio_de_las_configuraciones),
        "sin_medir": {
            "tareas": [t for t in TAREAS if t not in tareas_cubiertas],
            "cubos_de_tamano": [c for c in CUBOS_DE_TAMANO if c not in cubos_cubiertos],
        },
    }


def _configuraciones_que_faltan_en_texto(bloque: dict[str, Any]) -> str:
    """«a quien le falta cuánta», por su nombre. Un total («7 de 13») no dice
    si el recorte cayó repartido o entero sobre un motor."""
    faltan = [(m, n) for m, n in sorted((bloque.get("que_faltan_por_motor") or {}).items()) if n]
    if not faltan:
        return "ninguna falta"
    return "; ".join(f"{m}: falta(n) {n}" for m, n in faltan)


def _configuraciones_por_motor_en_texto(bloque: dict[str, Any]) -> str:
    """«cada motor aporta ...» — dicho con los numeros medidos, no con una
    frase escrita que deja de ser verdad en cuanto cambian.

    El total (7 de 13) no dice si el recorte cayo repartido o entero sobre
    uno: si todos los motores aportan lo mismo se dice ese numero, y si no,
    se dice el rango. Un «1» dicho de una pasada donde un motor aporto dos
    seria falso en la direccion tranquilizadora.
    """
    cuentas = {m: len(v) for m, v in (bloque.get("declaradas_por_la_pasada") or {}).items()}
    if not cuentas:
        return "un numero de familias de ajuste que la pasada no declaro"
    bajo, alto = min(cuentas.values()), max(cuentas.values())
    if bajo == alto:
        return f"{bajo} familia(s) de ajuste"
    return f"entre {bajo} y {alto} familias de ajuste segun el motor"


def _medidas_por_pliegue(resultados: Sequence[dict], metrica: str,
                         metrica_por_dataset: Mapping[str, str] | None = None,
                         ) -> dict[str, dict[str, dict[tuple, float]]]:
    """`dataset -> motor -> (repeticion, pliegue) -> valor`, de los intentos
    COMPLETADOS que traen la métrica y su pliegue.

    Es el mismo reparto que hace `aplicar_regla_de_cierre`; se extrae aquí
    porque ya lo quieren dos funciones y **dos sitios declarando lo mismo
    acaban divergiendo**. La regla de cierre sigue con su copia porque además
    necesita contar los FALLOS, y tocarla movería un veredicto publicado.

    `metrica_por_dataset` hace aquí lo mismo que en la regla: con mapa, cada
    dataset se lee con la métrica de su tarea. Sin mapa, todo con `metrica`,
    que es como se midieron los doce de 101-C3."""
    por_pliegue: dict[str, dict[str, dict[tuple, float]]] = {}
    for r in resultados:
        if r.get("estado") not in ESTADOS_QUE_CUENTAN_COMO_MEDIDA:
            continue
        metrica_de_este = metrica_del_dataset(r["dataset"], metrica, metrica_por_dataset)
        if metrica_de_este is None:
            continue
        valor = r.get(metrica_de_este)
        if valor is None or r.get("repeticion") is None or r.get("pliegue") is None:
            continue
        por_pliegue.setdefault(r["dataset"], {}).setdefault(r["motor"], {})[
            (r["repeticion"], r["pliegue"])] = float(valor)
    return por_pliegue


def _mediana(valores: Sequence[float]) -> float | None:
    if not valores:
        return None
    x = sorted(valores)
    n = len(x)
    return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2.0


def _desviacion_tipica(valores: Sequence[float]) -> float | None:
    """Desviación típica MUESTRAL (n-1). Con menos de dos medidas no hay
    dispersión que declarar, y devolver 0.0 diría que no la hay: **un valor
    ausente no es un cero**."""
    if len(valores) < 2:
        return None
    media = sum(valores) / len(valores)
    return (sum((v - media) ** 2 for v in valores) / (len(valores) - 1)) ** 0.5


def dispersion_de_un_motor(resultados: Sequence[dict], *, motor: str,
                           metrica: str = "auroc",
                           liston_en_puntos: float | None = None,
                           metrica_por_dataset: Mapping[str, str] | None = None,
                           ) -> dict[str, Any]:
    """LA DISPERSIÓN DE UN MOTOR, dataset a dataset, en las mismas unidades
    que `distancia_en_puntos` (puntos porcentuales de la métrica).

    **Por qué existe (2026-09-14).** El veredicto se lee con MEDIAS, y dos
    motores con la misma media y dispersiones muy distintas no son el mismo
    resultado. El propio protocolo registrado lo exige desde el primer día:
    `metricas_por_tarea.siempre` incluye `dispersion_entre_semillas`, y no se
    calculaba en ninguna parte — ni en el script de la pasada ni en el
    artefacto. Un campo prerregistrado que nadie calcula es una promesa, no
    una medida.

    Se declaran DOS dispersiones porque responden a preguntas distintas:

    * `sd` y `rango` sobre los 15 pliegues (3 repeticiones x 5 pliegues):
      cuánto se mueve el motor entre particiones **y** semillas juntas.
    * `sd_entre_semillas` y `rango_entre_semillas` sobre las 3 medias por
      repetición: **la que el protocolo pide por su nombre**, y la que aisla
      la inicialización del reparto de filas.

    `liston_en_puntos` es el margen de la regla de cierre (2,0). Sirve para
    contar cuántos datasets tienen un rango MAYOR que el margen entero con el
    que se decide quién cumple: ahí la media sola no describe al motor.
    """
    por_pliegue = _medidas_por_pliegue(resultados, metrica, metrica_por_dataset)
    por_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted(por_pliegue):
        medidas = por_pliegue[dataset].get(motor)
        if not medidas:
            continue
        valores = [v * 100.0 for v in medidas.values()]
        por_semilla: dict[Any, list[float]] = {}
        for (repeticion, _pliegue), valor in medidas.items():
            por_semilla.setdefault(repeticion, []).append(valor * 100.0)
        medias_por_semilla = [sum(v) / len(v) for _k, v in sorted(por_semilla.items())]
        por_dataset[dataset] = {
            "n_medidas": len(valores),
            "media": sum(valores) / len(valores),
            "sd": _desviacion_tipica(valores),
            "rango": (max(valores) - min(valores)) if valores else None,
            "n_semillas": len(medias_por_semilla),
            "sd_entre_semillas": _desviacion_tipica(medias_por_semilla),
            "rango_entre_semillas": ((max(medias_por_semilla) - min(medias_por_semilla))
                                     if medias_por_semilla else None),
        }

    def _columna(clave: str) -> list[float]:
        return [d[clave] for d in por_dataset.values() if d.get(clave) is not None]

    sds, rangos = _columna("sd"), _columna("rango")
    sds_semilla = _columna("sd_entre_semillas")
    peor = max(por_dataset, key=lambda d: por_dataset[d]["rango"] or -1.0) if por_dataset else None
    sobre_el_liston = None if liston_en_puntos is None else [
        d for d, v in por_dataset.items()
        if v["rango"] is not None and v["rango"] > liston_en_puntos]
    salida: dict[str, Any] = {
        "motor": motor,
        "metrica": metrica,
        "unidad": "puntos porcentuales de la metrica, igual que distancia_en_puntos",
        "n_datasets": len(por_dataset),
        "sd_mediana": _mediana(sds),
        "sd_maxima": max(sds) if sds else None,
        "rango_mediano": _mediana(rangos),
        "rango_maximo": max(rangos) if rangos else None,
        "dataset_mas_disperso": peor,
        "sd_entre_semillas_mediana": _mediana(sds_semilla),
        "sd_entre_semillas_maxima": max(sds_semilla) if sds_semilla else None,
        "liston_en_puntos": liston_en_puntos,
        "datasets_con_rango_mayor_que_el_liston": (
            None if sobre_el_liston is None else len(sobre_el_liston)),
        "cuales_con_rango_mayor_que_el_liston": (
            None if sobre_el_liston is None else sorted(sobre_el_liston)),
        "por_dataset": por_dataset,
    }
    # LA MISMA MEDIA VERDAD QUE LA REGLA YA CERRO, y aqui seguia abierta —
    # hallazgo M2, reparado el 2026-09-16. Con un mapa por tarea, `metrica`
    # sigue diciendo `auroc` mientras el bloque agrega rangos y desviaciones de
    # datasets medidos en `accuracy` y en `r2`. Los NUMEROS eran correctos
    # —cada dataset se midio con la suya, comprobado— pero la ETIQUETA no, y un
    # lector que viera «rango mediano 1,118 puntos, metrica: auroc» creeria que
    # los 40 son AUROC.
    #
    # Se usa la convencion que `aplicar_regla_de_cierre` ya tiene en vez de
    # inventar otra: `metrica` se queda donde estaba —quitarla moveria un
    # artefacto— y al lado va lo que de verdad se uso, dataset a dataset. Dos
    # bloques del mismo fichero declarando lo mismo de dos formas distintas es
    # como empieza la divergencia.
    if metrica_por_dataset:
        salida["metrica_por_dataset"] = {
            d: metrica_del_dataset(d, metrica, metrica_por_dataset)
            for d in por_dataset}
        salida["la_metrica_de_arriba_no_gobierna"] = (
            "hay un mapa por tarea: `metrica` es el valor por omision del "
            "parametro y NO es con lo que se midio. Lo que gobierna cada "
            "dataset esta en `metrica_por_dataset`. Y ojo al agregar: "
            "`rango_mediano`, `rango_maximo` y las desviaciones mezclan "
            "escalas de tareas distintas bajo una sola cifra.")
    # Redactada con los números, no guardada escrita: una frase compuesta y
    # almacenada deja de ser verdad en cuanto cambian los números que la
    # sostenían, y sigue sonando razonable.
    if por_dataset:
        # `_pt` y no un `:.3f` directo: con UNA sola medida la sd es `None` a
        # propósito (ausente no es cero) y formatear `None` reventaba la
        # función entera. Lo cazó su propia prueba: el bloque que declara la
        # dispersión no puede caerse justo cuando no hay dispersión que
        # declarar — y escribir «0,000» ahí sería peor todavía.
        def _pt(valor: float | None) -> str:
            return "no medible" if valor is None else f"{valor:.3f}"

        salida["como_hay_que_leer_este_numero"] = (
            f"{motor}: la media de cada dataset sale de "
            f"{por_dataset[next(iter(por_dataset))]['n_medidas']} "
            f"medidas cuya desviacion tipica mediana es {_pt(salida['sd_mediana'])} puntos "
            f"(maxima {_pt(salida['sd_maxima'])}, en {peor}); rango mediano "
            f"{_pt(salida['rango_mediano'])} puntos y maximo {_pt(salida['rango_maximo'])}. "
            + (f"En {len(sobre_el_liston)} de {len(por_dataset)} datasets el rango supera "
               f"los {liston_en_puntos} puntos del liston entero de la regla: ahi la media "
               f"sola NO describe al motor. " if sobre_el_liston is not None else "")
            + (f"Dispersion ENTRE SEMILLAS (la que el protocolo exige por su nombre): "
               f"mediana {_pt(salida['sd_entre_semillas_mediana'])}, maxima "
               f"{_pt(salida['sd_entre_semillas_maxima'])} puntos."
               if salida["sd_entre_semillas_mediana"] is not None else
               "Sin repeticiones suficientes para la dispersion entre semillas."))
    else:
        salida["como_hay_que_leer_este_numero"] = (
            f"{motor}: no hay ni una medida con pliegue declarado, asi que no hay "
            "dispersion que declarar. NO es dispersion cero.")
    return salida


def estabilidad_del_ganador(resultados: Sequence[dict], *,
                            metrica: str = "auroc",
                            baseline: str = "baseline",
                            metrica_por_dataset: Mapping[str, str] | None = None,
                            ) -> dict[str, Any]:
    """¿El ganador POR MEDIA de cada dataset gana de verdad, o gana el
    promedio?

    La regla de cierre define «mejor» como la mejor MEDIA de los ajustes, y
    eso se queda. Lo que esto añade es el dato que falta para leer esa media:
    emparejando por `(repeticion, pliegue)` —las mismas filas para los dos
    motores—, en cuántos de los 15 pliegues el primero le gana al segundo, y
    quién habría ganado con cada semilla por separado.

    `se_discute` es verdad cuando el primero gana MENOS de la mitad de los
    pliegues emparejados o cuando el ganador cambia según la semilla. No
    cambia ningún veredicto: el veredicto es el de la regla registrada. Dice
    cuándo la diferencia que lo decide es más pequeña que el ruido que la
    rodea.
    """
    por_pliegue = _medidas_por_pliegue(resultados, metrica, metrica_por_dataset)
    detalle: list[dict[str, Any]] = []
    for dataset in sorted(por_pliegue):
        medidas = {m: v for m, v in por_pliegue[dataset].items() if m != baseline}
        medias = {m: sum(v.values()) / len(v) for m, v in medidas.items() if v}
        if len(medias) < 2:
            continue
        orden = sorted(medias, key=lambda m: medias[m], reverse=True)
        primero, segundo = orden[0], orden[1]
        comunes = sorted(set(medidas[primero]) & set(medidas[segundo]))
        diferencias = [(medidas[primero][k] - medidas[segundo][k]) * 100.0 for k in comunes]
        ganados = sum(1 for d in diferencias if d > 0)
        semillas = sorted({rep for rep, _ in comunes})
        ganador_por_semilla = {}
        for rep in semillas:
            medias_rep = {}
            for m, v in medidas.items():
                suyos = [x for (r, _f), x in v.items() if r == rep]
                if suyos:
                    medias_rep[m] = sum(suyos) / len(suyos)
            if medias_rep:
                ganador_por_semilla[str(rep)] = max(medias_rep, key=lambda m: medias_rep[m])
        distintos = len(set(ganador_por_semilla.values())) > 1
        detalle.append({
            "dataset": dataset,
            "mejor_por_media": primero,
            "segundo": segundo,
            "ventaja_en_puntos": (medias[primero] - medias[segundo]) * 100.0,
            "pliegues_emparejados": len(diferencias),
            "pliegues_que_le_gana_al_segundo": ganados,
            "sd_de_la_diferencia": _desviacion_tipica(diferencias),
            "ganador_por_semilla": ganador_por_semilla,
            "el_ganador_cambia_con_la_semilla": distintos,
            "se_discute": distintos or (len(diferencias) > 0
                                        and ganados * 2 < len(diferencias)),
        })
    discutidos = [d["dataset"] for d in detalle if d["se_discute"]]
    pierde_la_mayoria = [d["dataset"] for d in detalle
                         if d["pliegues_emparejados"]
                         and d["pliegues_que_le_gana_al_segundo"] * 2 < d["pliegues_emparejados"]]
    return {
        "metrica": (metrica if metrica_por_dataset is None else
                    "por tarea, ver `metrica_por_dataset`"),
        **({} if metrica_por_dataset is None
           else {"metrica_por_dataset": {d: metrica_del_dataset(d, metrica, metrica_por_dataset)
                                         for d in sorted(por_pliegue)}}),
        "emparejado_por": "repeticion y pliegue",
        "n_datasets": len(detalle),
        "datasets_en_los_que_la_ordenacion_se_discute": sorted(discutidos),
        "datasets_en_los_que_el_mejor_por_media_PIERDE_la_mayoria_de_pliegues":
            sorted(pierde_la_mayoria),
        "detalle": detalle,
        "como_hay_que_leer_este_numero": (
            f"La regla decide con la MEDIA, y eso no cambia. En {len(discutidos)} de "
            f"{len(detalle)} datasets la ordenacion primero/segundo se discute al mirar "
            f"los pliegues emparejados o al separar las semillas; en "
            f"{len(pierde_la_mayoria)} el mejor por media PIERDE la mayoria de los "
            f"pliegues emparejados contra el segundo. Ahi «gano» significa «gano el "
            f"promedio», no «gano»."),
    }


def veredicto_con_su_alcance(protocolo: "ProtocoloExploratorio",
                             resultados: Sequence[dict], *,
                             motor: str,
                             motores_declarados: Sequence[str],
                             datasets_declarados: Sequence[str],
                             criterio_del_subconjunto: str = "",
                             configuraciones_declaradas: Mapping[str, Sequence[str]] | None = None,
                             criterio_de_las_configuraciones: str = "",
                             metrica: str = "auroc",
                             metrica_por_dataset: Mapping[str, str] | None = None,
                             datasets_exigidos: Sequence[str] | None = None,
                             ) -> dict[str, Any]:
    """El número y su alcance, EN EL MISMO OBJETO.

    Lo no negociable de la reparación del 2026-09-12: quien lea
    `cumple_la_regla` tiene delante, sin abrir otro fichero, contra cuántos
    motores se midió y cuáles faltan por nombre.

    `metrica_por_dataset` y `datasets_exigidos` se pasan TAL CUAL a la regla,
    a la dispersión y a la estabilidad — las tres leen el mismo mapa. Pasarlo
    a una y no a las otras daría un veredicto medido con la métrica de cada
    tarea y una dispersión medida con la de otra, y las dos viajan en el mismo
    objeto como si dijeran lo mismo.
    """
    regla = aplicar_regla_de_cierre(resultados, protocolo.regla_de_cierre,
                                    motor=motor, metrica=metrica,
                                    metrica_por_dataset=metrica_por_dataset,
                                    datasets_exigidos=datasets_exigidos)
    alcance = alcance_de_una_pasada(
        protocolo, resultados, motores_declarados=motores_declarados,
        datasets_declarados=datasets_declarados,
        criterio_del_subconjunto=criterio_del_subconjunto,
        configuraciones_declaradas=configuraciones_declaradas,
        criterio_de_las_configuraciones=criterio_de_las_configuraciones)
    faltan = alcance["motores"]["que_faltan"]
    compitieron = [m for m in alcance["motores"]["observados_en_los_resultados"]
                   if m != "baseline"]
    regla["alcance"] = alcance
    # LA DISPERSIÓN, EN EL MISMO OBJETO QUE EL VEREDICTO, por el mismo motivo
    # que el alcance: quien lea `cumple_la_regla` tiene delante, sin abrir otro
    # fichero, cuánto se mueve ese motor. Un veredicto leído solo con medias
    # da el mismo número para un motor estable y para uno que varía más que el
    # listón entero.
    regla["dispersion"] = dispersion_de_un_motor(
        resultados, motor=motor, metrica=metrica,
        liston_en_puntos=protocolo.regla_de_cierre.puntos,
        metrica_por_dataset=metrica_por_dataset)
    # La advertencia se REDACTA con los números medidos, no se guarda escrita:
    # una frase compuesta y guardada deja de ser verdad en cuanto cambian los
    # números que la sostenían, y sigue sonando razonable.
    # CON QUE SE MIDIO CADA DATASET, en la primera frase y no en un campo
    # aparte. Un "32/40 CUMPLE" de tres tareas medidas cada una con su metrica
    # y uno de tres tareas medidas todas con AUROC se escriben igual, y el
    # segundo tiene 20 datasets fuera del denominador sin decirlo.
    con_que_se_midio = ""
    if metrica_por_dataset is not None:
        usadas = sorted(set((regla.get("metrica_por_dataset") or {}).values()))
        faltan_medida = regla.get("datasets_exigidos_que_no_se_midieron") or []
        con_que_se_midio = (
            f"Cada dataset se midio con la metrica de SU tarea ({', '.join(usadas) or 'ninguna'}), "
            f"no con una sola para todos. "
            + (f"{len(faltan_medida)} dataset(s) exigido(s) SIN medida: "
               f"{', '.join(faltan_medida)}. " if faltan_medida
               else "Todos los datasets exigidos tienen medida. "))
    regla["como_hay_que_leer_este_numero"] = (
        con_que_se_midio +
        f"{regla['cumplidos']}/{regla['datasets']} sobre un ALCANCE RECORTADO: "
        f"{alcance['motores']['n_que_corrieron']} de "
        f"{alcance['motores']['n_del_protocolo']} motores del protocolo y "
        f"{alcance['datasets']['n_que_corrieron']} de "
        f"{alcance['datasets']['n_del_protocolo']} datasets. "
        # LA TERCERA CIFRA DEL RECORTE, y la que toca la propia regla: su
        # `definicion_de_mejor` habla de «la mejor media de LOS AJUSTES», y una
        # sola configuracion por motor no es eso. Va en la MISMA frase que las
        # otras dos, no en un campo aparte que se pueda leer por separado.
        f"Configuraciones: {alcance['configuraciones']['n_que_corrieron']} de "
        f"{alcance['configuraciones']['n_del_protocolo']} del protocolo "
        f"({_configuraciones_que_faltan_en_texto(alcance['configuraciones'])}); "
        f"la regla mide «la mejor media de los AJUSTES» y aqui cada motor aporta "
        f"{_configuraciones_por_motor_en_texto(alcance['configuraciones'])}. "
        f"NO corrieron: {', '.join(faltan) if faltan else 'ninguno'}. "
        f"Compitieron por el primer puesto {len(compitieron)} motores "
        f"(el baseline no compite, la regla lo excluye). "
        f"{regla['aciertos_por_ser_el_mejor']} de los {regla['cumplidos']} "
        f"aciertos lo son por SER el mejor de esos, o sea distancia 0,0000: "
        f"mirar `ventaja_sobre_el_segundo_en_puntos` antes de leerlos como dominio. "
        + (f"Puede perder {regla['datasets_que_puede_perder_sin_incumplir']} "
           f"dataset(s) mas sin bajarse del liston. "
           if regla["cumple_la_regla"]
           else f"NO cumple: le faltan {regla['datasets_que_le_faltan_para_cumplir']} "
                f"dataset(s) para llegar al liston. ") +
        f"Sin medir: {', '.join(alcance['sin_medir']['tareas']) or 'ninguna tarea'}; "
        f"cubos {', '.join(alcance['sin_medir']['cubos_de_tamano']) or 'todos cubiertos'}. "
        + regla["dispersion"]["como_hay_que_leer_este_numero"])
    return regla
