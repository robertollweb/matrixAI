# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""104-C1 — el estudio y su presupuesto total, y el leaderboard.

QUÉ ENTREGA. `EstudioSpec` es la declaración de UN estudio: qué problema,
qué motores puede usar, cuánto tiempo total tiene y cómo se reparte ese
tiempo entre buscar candidatos, ajustar el final, calibrar y evaluar.
`Leaderboard` es el registro de lo que pasó: cada intento —reutilizando
`Intento`/`ejecutar_intento` de `matrixai_engines.harness`, 101-C2, nunca una
segunda estructura para lo mismo— con su métrica de selección, su coste
medido y si fue una fila de SELECCIÓN o de EVALUACIÓN FINAL. Esas dos cosas
NUNCA se mezclan (invariante del corte): una fila de selección puede haber
visto varios pliegues de desarrollo; una de evaluación final es la ÚNICA vez
que algo toca el `test` reservado, y `matrixai.estudio.accesos.
RegistroDeAccesos` (104-C0, reutilizado) es quien lo hace cumplir un piso
más abajo — este módulo no reimplementa ese guardián, solo declara la
DISTINCIÓN en la forma del dato.

EL INVARIANTE QUE MÁS CUESTA DEJAR PASAR POR ALTO, hecho código: «aumentar
el número de motores no aumenta sin aviso el presupuesto total». Un
`EstudioSpec` con `presupuesto_total_segundos` fijo y `motores_permitidos`
variable NO multiplica el presupuesto por motor en ningún sitio de este
módulo — `tiempo_de_busqueda_por_motor()` hace la división explícita
(reserva de búsqueda ENTRE el número de motores), para que quien lea el
código vea la resta, no una suposición.

«Un estudio sin restricciones usa valores por omisión VISIBLES»: los
valores por omisión son campos con default explícito en el propio
`EstudioSpec`, y `a_json()` los escribe SIEMPRE — nunca se omiten del
documento por ser «los de siempre».

«Una métrica sin definición no se acepta»: `objetivo_medible` se valida
contra el registro CERRADO de 105-C1 (`especificacion()`, reutilizado; si el
id no existe, `especificacion()` ya levanta `MetricaDesconocida` — no se
duplica esa comprobación aquí con otro mensaje).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from matrixai.estudio.errores import EsquemaInvalido
from matrixai.estudio.esquemas import ValorDeMetrica
from matrixai.estudio.metricas import especificacion
from matrixai.estudio.validacion import (
    digest_canonico,
    exigir_entero,
    exigir_mapa,
    exigir_real,
    exigir_real_o_nulo,
    exigir_texto,
    exigir_tupla_de_textos,
)
from matrixai.estudio.vocabulario import exigir_opcion

__all__ = [
    "FASES_DE_LEADERBOARD",
    "MODOS_DE_ESTUDIO",
    "EntradaDeLeaderboard",
    "EstudioSpec",
    "Leaderboard",
    "ReservaDeTiempo",
    "tiempo_de_busqueda_por_motor",
]

#: Cerrado, como todo vocabulario de esta casa: un tercer modo a mitad de
#: programa prometería un protocolo que nadie ha escrito todavía.
MODOS_DE_ESTUDIO = ("rapido", "completo")

#: La distinción que el corte pide que nunca se mezcle: una fila que vio
#: pliegues de desarrollo para ELEGIR, frente a la única fila que corrió
#: sobre el `test` reservado para el número que se publica.
FASES_DE_LEADERBOARD = ("seleccion", "evaluacion_final")


@dataclass(frozen=True)
class ReservaDeTiempo:
    """Cómo se reparte el presupuesto total entre las cuatro fases que el
    corte nombra explícitamente. Sumar más motores a `EstudioSpec` reparte
    `busqueda` entre más candidatos — no toca `ajuste_final`, `calibracion`
    ni `evaluacion`, que son del ganador, no de la búsqueda."""

    busqueda: float
    ajuste_final: float
    calibracion: float
    evaluacion: float

    def __post_init__(self) -> None:
        for nombre in ("busqueda", "ajuste_final", "calibracion", "evaluacion"):
            exigir_real(getattr(self, nombre), f"reserva.{nombre}", minimo=0.0)

    def total(self) -> float:
        return self.busqueda + self.ajuste_final + self.calibracion + self.evaluacion

    def a_json(self) -> dict[str, float]:
        return {"busqueda": self.busqueda, "ajuste_final": self.ajuste_final,
                "calibracion": self.calibracion, "evaluacion": self.evaluacion,
                "total": self.total()}

    @classmethod
    def desde_json(cls, payload: Any) -> "ReservaDeTiempo":
        mapa = exigir_mapa(payload, "reserva")
        return cls(busqueda=mapa["busqueda"], ajuste_final=mapa["ajuste_final"],
                   calibracion=mapa["calibracion"], evaluacion=mapa["evaluacion"])


def tiempo_de_busqueda_por_motor(estudio: "EstudioSpec") -> float:
    """La reserva de búsqueda ENTRE el número de motores permitidos — la
    resta explícita que demuestra que el presupuesto TOTAL no cambia al
    añadir un motor: lo que cambia es cuánto le toca a cada uno."""
    return estudio.reserva.busqueda / len(estudio.motores_permitidos)


@dataclass(frozen=True)
class EstudioSpec:
    """La declaración de UN estudio: problema, motores, presupuesto total
    y cómo se reparte. `presupuesto_total_segundos` es la única cifra que
    manda; todo lo demás (incluida `reserva`) tiene que caber dentro."""

    ESQUEMA: ClassVar[str] = "matrixai.estudio.estudio_spec"

    estudio_id: str
    problem_id: str
    motores_permitidos: tuple[str, ...]
    presupuesto_total_segundos: float
    reserva: ReservaDeTiempo
    objetivo_medible: str
    motor_de_reserva: str
    modo: str = "completo"
    recursos_declarados: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        exigir_texto(self.estudio_id, "estudio_id")
        exigir_texto(self.problem_id, "problem_id")
        object.__setattr__(self, "motores_permitidos",
                           exigir_tupla_de_textos(self.motores_permitidos,
                                                  "motores_permitidos", minimo=1))
        exigir_real(self.presupuesto_total_segundos, "presupuesto_total_segundos", minimo=0.0)
        if not isinstance(self.reserva, ReservaDeTiempo):
            raise EsquemaInvalido("no_es_mapa", campo="reserva", valor=repr(self.reserva))
        # LA COMPROBACIÓN QUE EL CRITERIO DE TERMINADO PIDE LITERALMENTE:
        # «suma de reservas/ejecuciones coherente». Una reserva que promete
        # más de lo que el presupuesto total tiene es una promesa rota antes
        # de medir nada — se rechaza aquí, no se descubre a mitad de estudio.
        if self.reserva.total() > self.presupuesto_total_segundos:
            raise EsquemaInvalido("reserva_supera_presupuesto",
                                  valor=round(self.reserva.total(), 3),
                                  opciones=round(self.presupuesto_total_segundos, 3))
        exigir_opcion(self.modo, "modo", MODOS_DE_ESTUDIO)
        exigir_texto(self.motor_de_reserva, "motor_de_reserva")
        if self.motor_de_reserva not in self.motores_permitidos:
            raise EsquemaInvalido("motor_no_permitido", campo="motor_de_reserva",
                                  valor=repr(self.motor_de_reserva),
                                  opciones=list(self.motores_permitidos))
        # «Una métrica sin definición no se acepta»: se reutiliza el
        # registro CERRADO de 105-C1 tal cual — si `objetivo_medible` no
        # existe ahí, `especificacion()` ya levanta `MetricaDesconocida`.
        especificacion(self.objetivo_medible)
        object.__setattr__(self, "recursos_declarados", dict(self.recursos_declarados))

    def a_json(self) -> dict[str, Any]:
        return {
            "esquema": self.ESQUEMA, "estudio_id": self.estudio_id,
            "problem_id": self.problem_id,
            "motores_permitidos": list(self.motores_permitidos),
            "presupuesto_total_segundos": self.presupuesto_total_segundos,
            "reserva": self.reserva.a_json(),
            "objetivo_medible": self.objetivo_medible,
            "motor_de_reserva": self.motor_de_reserva,
            # `modo` y `recursos_declarados` viajan SIEMPRE, tengan o no su
            # valor por omisión — «valores por omisión VISIBLES», no
            # omitidos del documento por ser "los de siempre".
            "modo": self.modo, "recursos_declarados": dict(self.recursos_declarados),
        }

    def digest(self) -> str:
        return digest_canonico(self.a_json())

    @classmethod
    def desde_json(cls, payload: Any) -> "EstudioSpec":
        mapa = exigir_mapa(payload, cls.ESQUEMA)
        return cls(
            estudio_id=mapa["estudio_id"], problem_id=mapa["problem_id"],
            motores_permitidos=tuple(mapa["motores_permitidos"]),
            presupuesto_total_segundos=mapa["presupuesto_total_segundos"],
            reserva=ReservaDeTiempo.desde_json(mapa["reserva"]),
            objetivo_medible=mapa["objetivo_medible"],
            motor_de_reserva=mapa["motor_de_reserva"],
            modo=mapa.get("modo", "completo"),
            recursos_declarados=dict(mapa.get("recursos_declarados") or {}))


@dataclass(frozen=True)
class EntradaDeLeaderboard:
    """Un candidato registrado: métrica de selección CON incertidumbre y
    alcance (reutiliza `ValorDeMetrica.uncertainty` de 104-C0/105-C2, nunca
    un número suelto), coste medido, y de qué FASE es — nunca ambas."""

    candidate: str
    engine: str
    engine_version: str
    config_efectiva: dict[str, Any]
    metrica_de_seleccion: ValorDeMetrica
    estado: str
    fase: str
    motivo: dict[str, str] | None = None
    cpu_seconds: float | None = None
    wall_seconds: float | None = None
    ram_mb: float | None = None
    latencia_end_to_end_ms: float | None = None

    def __post_init__(self) -> None:
        exigir_texto(self.candidate, "candidate")
        exigir_texto(self.engine, "engine")
        exigir_texto(self.engine_version, "engine_version")
        object.__setattr__(self, "config_efectiva", dict(exigir_mapa(
            self.config_efectiva, "config_efectiva")))
        if not isinstance(self.metrica_de_seleccion, ValorDeMetrica):
            raise EsquemaInvalido("no_es_mapa", campo="metrica_de_seleccion",
                                  valor=repr(self.metrica_de_seleccion))
        exigir_texto(self.estado, "estado")
        exigir_opcion(self.fase, "fase", FASES_DE_LEADERBOARD)
        for nombre in ("cpu_seconds", "wall_seconds", "ram_mb", "latencia_end_to_end_ms"):
            exigir_real_o_nulo(getattr(self, nombre), nombre, minimo=0.0)

    def a_json(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate, "engine": self.engine,
            "engine_version": self.engine_version,
            "config_efectiva": dict(self.config_efectiva),
            "metrica_de_seleccion": self.metrica_de_seleccion.a_json(),
            "estado": self.estado, "fase": self.fase, "motivo": self.motivo,
            "cpu_seconds": self.cpu_seconds, "wall_seconds": self.wall_seconds,
            "ram_mb": self.ram_mb, "latencia_end_to_end_ms": self.latencia_end_to_end_ms,
        }

    @classmethod
    def desde_json(cls, payload: Any) -> "EntradaDeLeaderboard":
        mapa = exigir_mapa(payload, "entrada_de_leaderboard")
        return cls(
            candidate=mapa["candidate"], engine=mapa["engine"],
            engine_version=mapa["engine_version"],
            config_efectiva=dict(mapa["config_efectiva"]),
            metrica_de_seleccion=ValorDeMetrica.desde_json(mapa["metrica_de_seleccion"]),
            estado=mapa["estado"], fase=mapa["fase"], motivo=mapa.get("motivo"),
            cpu_seconds=mapa.get("cpu_seconds"), wall_seconds=mapa.get("wall_seconds"),
            ram_mb=mapa.get("ram_mb"),
            latencia_end_to_end_ms=mapa.get("latencia_end_to_end_ms"))


@dataclass(frozen=True)
class Leaderboard:
    """El registro entero de un estudio. `de_seleccion()`/
    `de_evaluacion_final()` son la forma de consultar UNA de las dos fases
    sin arriesgarse a mezclar sus filas por accidente en un cálculo."""

    estudio_id: str
    entradas: tuple[EntradaDeLeaderboard, ...] = ()

    def __post_init__(self) -> None:
        exigir_texto(self.estudio_id, "estudio_id")
        for entrada in self.entradas:
            if not isinstance(entrada, EntradaDeLeaderboard):
                raise EsquemaInvalido("no_es_mapa", campo="entradas", valor=repr(entrada))

    def de_seleccion(self) -> tuple[EntradaDeLeaderboard, ...]:
        return tuple(e for e in self.entradas if e.fase == "seleccion")

    def de_evaluacion_final(self) -> tuple[EntradaDeLeaderboard, ...]:
        return tuple(e for e in self.entradas if e.fase == "evaluacion_final")

    def con_entrada(self, entrada: EntradaDeLeaderboard) -> "Leaderboard":
        """Añade una entrada SIN mutar — el leaderboard es un valor, como
        todo lo demás en `matrixai.estudio`; una lista mutable escondida
        dentro sería el mismo defecto que un `Particion` con las filas en
        vez de los índices (101-C0): dos copias que dejan de coincidir."""
        return Leaderboard(estudio_id=self.estudio_id, entradas=self.entradas + (entrada,))

    def a_json(self) -> dict[str, Any]:
        return {"estudio_id": self.estudio_id,
                "entradas": [e.a_json() for e in self.entradas]}

    @classmethod
    def desde_json(cls, payload: Any) -> "Leaderboard":
        mapa = exigir_mapa(payload, "leaderboard")
        return cls(estudio_id=mapa["estudio_id"],
                   entradas=tuple(EntradaDeLeaderboard.desde_json(e)
                                 for e in mapa.get("entradas") or ()))
