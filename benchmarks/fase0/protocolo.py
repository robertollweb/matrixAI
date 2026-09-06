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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
if str(_RAIZ_DEL_CORE) not in sys.path:
    sys.path.insert(0, str(_RAIZ_DEL_CORE))

from matrixai.estudio.validacion import digest_canonico  # noqa: E402

__all__ = [
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

    def a_json(self) -> dict[str, Any]:
        return {
            "total_ejecuciones": self.total_ejecuciones,
            "ejecuciones_por_cubo": dict(self.ejecuciones_por_cubo),
            "horas_reloj_peor_caso_secuencial": self.horas_reloj_peor_caso_secuencial,
            "horas_reloj_peor_caso_con_paralelismo": self.horas_reloj_peor_caso_con_paralelismo,
            "dias_peor_caso_con_paralelismo": self.dias_peor_caso_con_paralelismo,
            "procesos_en_paralelo": self.procesos_en_paralelo,
        }


def calcular_coste(protocolo: ProtocoloExploratorio) -> CosteDeLaPasada:
    """Número de ejecuciones y cota de coste, consistente con motores,
    pliegues, repeticiones y presupuesto — el criterio de terminado literal
    del 101-C1.

    Una «ejecución» es UN ajuste: un (dataset, pliegue, repetición, motor,
    configuración). El peor caso supone que CADA ejecución agota su
    presupuesto — no es lo que se espera que pase (la mayoría termina antes),
    es la COTA que ningún run individual puede superar sin contarse como
    fallo (`completed_budget_limited`/`failed` por timeout, C2).
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

    horas_paralelo = horas_secuencial / protocolo.presupuesto.procesos_en_paralelo
    return CosteDeLaPasada(
        total_ejecuciones=total,
        ejecuciones_por_cubo=por_cubo,
        horas_reloj_peor_caso_secuencial=round(horas_secuencial, 2),
        horas_reloj_peor_caso_con_paralelismo=round(horas_paralelo, 2),
        dias_peor_caso_con_paralelismo=round(horas_paralelo / 24.0, 2),
        procesos_en_paralelo=protocolo.presupuesto.procesos_en_paralelo,
    )
