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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
if str(_RAIZ_DEL_CORE) not in sys.path:
    sys.path.insert(0, str(_RAIZ_DEL_CORE))

from matrixai.estudio.validacion import digest_canonico  # noqa: E402

__all__ = [
    "ANCLAS_DE_ESCALADO",
    "CUBOS_DE_TAMANO",
    "TAREAS",
    "CosteDeLaPasada",
    "DatasetRegistrado",
    "DisenoDeParticion",
    "Motor",
    "PresupuestoPorCubo",
    "ProtocoloError",
    "ProtocoloExploratorio",
    "ReglaDeCierre",
    "calcular_coste",
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
# El protocolo registrado pide `hilos=4` x `procesos_en_paralelo=6` = 24 hilos.
# Esta máquina tiene 8 CPUs LÓGICAS (4 núcleos físicos con SMT x2), medido el
# 2026-09-12: `nproc` 8, `os.sched_getaffinity` 8, `lscpu` «Core(s) per socket:
# 4, Thread(s) per core: 2», sin cuota de cgroup (`cpu.max` = «max 100000»).
# 24 sobre 8 es TRIPLE reserva — y `recursos_declarados` del protocolo dice
# «cpu_fisicas: 8», que no es lo medido: son 8 lógicas y 4 físicas.
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
    alta_cardinalidad: bool
    desbalanceado: bool
    solo_numericas: bool
    licencia: str
    sellado: bool = False

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

    def a_json(self) -> dict[str, Any]:
        return {
            "data_id": self.data_id, "nombre": self.nombre, "fuente": self.fuente,
            "version": self.version, "sha256_arff": self.sha256_arff,
            "columna_objetivo": self.columna_objetivo, "tarea": self.tarea,
            "cubo_de_tamano": self.cubo_de_tamano, "n_filas": self.n_filas,
            "n_columnas": self.n_columnas, "tiene_faltantes": self.tiene_faltantes,
            "alta_cardinalidad": self.alta_cardinalidad, "desbalanceado": self.desbalanceado,
            "solo_numericas": self.solo_numericas, "licencia": self.licencia,
            "sellado": self.sellado,
        }

    @classmethod
    def desde_json(cls, payload: dict[str, Any]) -> "DatasetRegistrado":
        return cls(**{k: payload[k] for k in (
            "data_id", "nombre", "fuente", "version", "sha256_arff", "columna_objetivo",
            "tarea", "cubo_de_tamano", "n_filas", "n_columnas", "tiene_faltantes",
            "alta_cardinalidad", "desbalanceado", "solo_numericas", "licencia", "sellado")})


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
        protocolo registrado da 4 x 6 = **24**, sobre 8 CPUs — que es la
        sobre-reserva que el 101-C1/C2 dejó apuntada como deuda."""
        return self.hilos * self.procesos_en_paralelo

    def sobre_reserva(self, cpus: int | None = None) -> int:
        """Hilos de MÁS sobre las CPUs que hay, o 0 si cabe. Con el protocolo
        registrado y esta máquina: 24 - 8 = 16 de más."""
        disponibles = cpus if cpus is not None else cpus_disponibles()
        return max(0, self.hilos_reservados - disponibles)

    def procesos_que_caben(self, cpus: int | None = None) -> int:
        """Los procesos que cabrían con estos `hilos` sin pasarse. No cambia
        el protocolo registrado —eso sería otro protocolo con otro digest—:
        dice lo que habría que poner."""
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


def aplicar_regla_de_cierre(resultados: Sequence[dict], regla: ReglaDeCierre, *,
                            motor: str, metrica: str = "auroc",
                            baseline: str = "baseline") -> dict[str, Any]:
    """¿`motor` queda a menos de `regla.puntos` del mejor en al menos
    `regla.fraccion_minima` de los datasets?

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
    for r in resultados:
        ds, mt = r["dataset"], r["motor"]
        if r.get("estado") != "completed":
            fallos.setdefault(ds, set()).add(mt)
            continue
        valor = r.get(metrica)
        if valor is None:
            continue
        por_dataset.setdefault(ds, {}).setdefault(mt, []).append(float(valor))

    detalle: list[dict[str, Any]] = []
    cumplidos = 0
    for ds in sorted(set(por_dataset) | set(fallos)):
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
        detalle.append({"dataset": ds, "cumple": cumple, "perdido_por_fallo": perdido,
                        "mejor": mejor_motor, "distancia_en_puntos": distancia})

    total = len(detalle)
    fraccion = (cumplidos / total) if total else 0.0
    return {"motor": motor, "metrica": metrica, "puntos_exigidos": regla.puntos,
            "fraccion_minima": regla.fraccion_minima, "datasets": total,
            "cumplidos": cumplidos, "fraccion": fraccion,
            "cumple_la_regla": fraccion >= regla.fraccion_minima,
            "definicion_de_mejor": regla.definicion_de_mejor, "detalle": detalle}


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
      `procesos_en_paralelo`, como si 6 procesos fueran 6 veces más rápidos.
      Es el número ya publicado del protocolo registrado (**74,93 h**) y se
      conserva tal cual para no cambiar en silencio algo que ya se dijo.
    - `horas_reloj_suelo_medido` usa el escalado MEDIDO (`speedup_medido`).
      Con el protocolo registrado en esta máquina da **95,86 h**: 6 procesos
      x 4 hilos = 24 hilos sobre 8 CPUs no rinden 6x, rinden como mucho el
      techo medido 4,69x.

    El segundo NO invalida el primero como cuenta de ejecuciones — son las
    mismas 6.500 —, pero sí dice que 74,93 h era **inalcanzable**: exigía un
    6x que esta máquina no puede dar. Y `horas_reloj_suelo_medido` es un
    SUELO, no un techo: las anclas se midieron con 1 hilo por proceso, que
    escala mejor (3,60x) que los 4 hilos por proceso que el protocolo pide
    (2,53x medido), así que la pasada real irá más lenta, no más rápida.

    `sobre_reserva` > 0 es la señal de que el protocolo pide más hilos que
    CPUs hay. Este cálculo lo DECLARA, no lo corrige: bajar
    `procesos_en_paralelo` a los `procesos_que_caben` (2, aquí) cambia el
    presupuesto y por tanto el digest, y eso es escribir un protocolo NUEVO
    (invariante 1), no una decisión de esta función.
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

    # La división LINEAL, tal como estaba: supone que 6 procesos van 6 veces
    # más rápido. Se CONSERVA —es la cota de 74,93 h ya publicada del
    # protocolo registrado, y sustituirla en silencio sería cambiar un número
    # publicado sin decirlo— pero ya no viaja sola.
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
