# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El registro de accesos y el protocolo mínimo — 104-C0.

**Esta es la pieza central del corte.** Los esquemas de al lado describen
documentos; esto es lo único que puede impedir que la prueba reservada alimente
el aprendizaje, y sin ello el resto es papel: nada obliga a que un `SplitPlan`
con rol `test` se respete si cualquiera puede leer esas filas cuando quiera.

EL PROTOCOLO, tal y como lo escribe el contrato:

1. separar el test AL PRINCIPIO (lo hace el `SplitPlan`, que sin rol `test` no
   se construye salvo evaluación anidada declarada);
2. todo aprendizaje y toda selección ocurren en desarrollo;
3. las predicciones para calibrador y umbral salen de particiones apropiadas
   dentro de desarrollo;
4. congelar el pipeline;
5. evaluar UNA vez en el test.

CÓMO SE HACE CUMPLIR. Todo acceso a los datos pasa por aquí y declara TRES
cosas: en qué fase se está, qué rol se lee y PARA QUÉ. Con eso:

* leer `test` o `external_test` en una fase de desarrollo, selección,
  calibración o ajuste final se **deniega** con `FugaDeTest`;
* leerlo en su fase pero con un propósito que aprende (`fit`, `tune`, `select`,
  `calibrate`, `tune_threshold`) también se deniega: mirar el test para elegir
  el umbral es usarlo para decidir aunque se haga al final;
* leerlo sin haber congelado el pipeline se deniega: lo que se evalúe todavía
  puede cambiar después de mirarlo, y entonces el número no es de nadie.

Y LO QUE NO HACE, que importa igual: **el intento denegado se ANOTA**. Un
registro que solo guardara los accesos concedidos contaría una historia limpia
de un estudio sucio. Se declara lo que pasó, no lo que se pidió.

Después de `aprender_del_test`, la independencia no se recupera: el registro
queda marcado y `etiqueta_de_evidencia` deja de conceder `independent_test`.
Ese es el estado `test_used_for_development` del contrato, y el artefacto nuevo
no hereda la etiqueta del anterior.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, NoReturn

from matrixai.estudio.errores import EsquemaInvalido, FugaDeTest, ProtocoloRoto
from matrixai.estudio.esquemas import SplitPlan
from matrixai.estudio.migracion import ESTUDIO_SCHEMA_VERSION
from matrixai.estudio.validacion import (
    digest_canonico,
    exigir_motivo_bilingue,
    exigir_texto,
)
from matrixai.estudio.vocabulario import (
    FASES,
    PROPOSITOS,
    PROPOSITOS_QUE_APRENDEN,
    ROLES,
    ROLES_RESERVADOS,
    exigir_opcion,
    fase_del_rol_reservado,
)

__all__ = ["Acceso", "GRANULARIDADES", "LectorDeParticiones", "RegistroDeAccesos",
           "TIPOS_DE_ANOTACION"]

#: Qué clase de línea es. `read` es un acceso a datos; las otras tres son usos
#: que también cuentan para juzgar la evidencia.
TIPOS_DE_ANOTACION = ("phase", "read", "freeze", "learn_from_test")

#: Qué se contó en un acceso. Seis filas de tres pacientes son `6 observations` o
#: `3 units`: el mismo número sin decir de qué es se lee mal.
GRANULARIDADES = ("observations", "units")

ESQUEMA_DEL_REGISTRO = "matrixai.estudio.access_log"


@dataclass(frozen=True)
class Acceso:
    """Una línea del registro. Concedida o denegada: las dos se guardan."""

    orden: int
    tipo: str
    fase: str | None = None
    rol: str | None = None
    proposito: str | None = None
    candidato: str | None = None
    artefacto: str | None = None
    observaciones: int | None = None
    granularidad: str = "observations"
    concedido: bool = True
    motivo: dict[str, str] | None = None

    def a_json(self) -> dict[str, Any]:
        return {"orden": self.orden, "tipo": self.tipo, "fase": self.fase,
                "rol": self.rol, "proposito": self.proposito,
                "candidato": self.candidato, "artefacto": self.artefacto,
                "observaciones": self.observaciones,
                "granularidad": self.granularidad, "concedido": self.concedido,
                "motivo": dict(self.motivo) if self.motivo else None}


class RegistroDeAccesos:
    """Quién ha leído qué, en qué fase y para qué.

    Se construye sobre un `SplitPlan` porque las reglas dependen de él: un plan
    que declara evaluación anidada no se puede conducir con este protocolo, y
    decirlo al construir es mejor que descubrirlo a mitad del estudio.
    """

    def __init__(self, plan: SplitPlan, *, estudio: str = "estudio") -> None:
        if plan.nested_evaluation:
            raise ProtocoloRoto("anidada_no_implementada")
        self.plan = plan
        self.estudio = exigir_texto(estudio, "estudio")
        self._accesos: list[Acceso] = []
        self._fase: str | None = None
        self._congelado: str | None = None
        self._orden_de_contaminacion: int | None = None
        self._evaluaciones: dict[str, list[tuple[int, str]]] = {}

    # -- fases ------------------------------------------------------------
    def abrir_fase(self, fase: str) -> None:
        exigir_opcion(fase, "fase", FASES)
        self._fase = fase
        self._anotar(Acceso(orden=len(self._accesos), tipo="phase", fase=fase))

    @contextmanager
    def fase(self, nombre: str) -> Iterator["RegistroDeAccesos"]:
        """Azúcar para conducir el estudio por fases sin olvidarse de cerrarlas.

        Al salir se restaura la fase anterior: si no, un bloque anidado dejaría
        abierta la de dentro y el siguiente acceso se juzgaría con la fase
        equivocada — que es exactamente el fallo que esto persigue.
        """
        anterior = self._fase
        self.abrir_fase(nombre)
        try:
            yield self
        finally:
            self._fase = anterior

    @property
    def fase_actual(self) -> str | None:
        return self._fase

    # -- congelar y contaminar --------------------------------------------
    def congelar(self, artefacto: str) -> None:
        """Fija QUÉ artefacto se va a evaluar. Sin esto no se puede leer el test."""
        exigir_texto(artefacto, "artefacto")
        self._congelado = artefacto
        self._anotar(Acceso(orden=len(self._accesos), tipo="freeze", fase=self._fase,
                            artefacto=artefacto))

    @property
    def artefacto_congelado(self) -> str | None:
        return self._congelado

    def aprender_del_test(self, motivo: dict[str, str]) -> None:
        """Se declara que a partir de aquí se aprende CON el test.

        No es un permiso que devuelva la independencia: es lo contrario. Queda
        anotado, y desde este punto ningún artefacto vuelve a recibir la etiqueta
        `independent_test`. El contrato lo llama `test_used_for_development`.
        """
        exigir_motivo_bilingue(motivo, "motivo")
        if self._orden_de_contaminacion is None:
            self._orden_de_contaminacion = len(self._accesos)
        self._anotar(Acceso(orden=len(self._accesos), tipo="learn_from_test",
                            fase=self._fase, motivo=dict(motivo)))

    @property
    def test_usado_para_desarrollar(self) -> bool:
        return self._orden_de_contaminacion is not None

    # -- el acceso ---------------------------------------------------------
    def anotar(self, rol: str, *, proposito: str, candidato: str | None = None,
               artefacto: str | None = None, observaciones: int | None = None,
               granularidad: str = "observations") -> None:
        """Registra un acceso a los datos de un rol, o lo DENIEGA con motivo.

        `granularidad` dice QUÉ se contó: seis filas de tres pacientes son
        `6 observations` o `3 units` según lo que se pidiera, y un registro que
        no lo distinguiera enseñaría un 3 donde hubo 6.
        """
        exigir_opcion(rol, "rol", ROLES)
        exigir_opcion(proposito, "proposito", PROPOSITOS)
        if self._fase is None:
            raise ProtocoloRoto("fase_desconocida")
        if rol not in self.plan.roles_presentes():
            raise EsquemaInvalido("rol_no_esta_en_el_plan", valor=rol)

        exigir_opcion(granularidad, "granularidad", GRANULARIDADES)
        linea = dict(fase=self._fase, rol=rol, proposito=proposito,
                     candidato=candidato, artefacto=artefacto,
                     observaciones=observaciones, granularidad=granularidad)

        if rol in ROLES_RESERVADOS:
            fase_permitida = fase_del_rol_reservado(rol)
            if self._fase != fase_permitida:
                self._denegar(linea, "fuga_de_test", FugaDeTest,
                              valor=rol, campo=self._fase)
            if proposito in PROPOSITOS_QUE_APRENDEN:
                self._denegar(linea, "proposito_indebido_en_el_test", FugaDeTest,
                              valor=rol, campo=proposito)
            if self._congelado is None:
                self._denegar(linea, "test_sin_congelar", ProtocoloRoto, valor=rol)
            if artefacto is None:
                self._denegar(linea, "artefacto_distinto_del_congelado", ProtocoloRoto,
                              valor="None", opciones=self._congelado)
            if artefacto != self._congelado:
                self._denegar(linea, "artefacto_distinto_del_congelado", ProtocoloRoto,
                              valor=artefacto, opciones=self._congelado)
            self._evaluaciones.setdefault(artefacto, []).append((len(self._accesos), rol))

        self._anotar(Acceso(orden=len(self._accesos), tipo="read", concedido=True, **linea))

    def _denegar(self, linea: dict[str, Any], clave: str, excepcion: type,
                 **campos: Any) -> NoReturn:
        """Anota el intento ANTES de levantar. Un registro que solo guarda lo
        concedido cuenta una historia limpia de un estudio sucio."""
        error = excepcion(clave, **campos)
        self._anotar(Acceso(orden=len(self._accesos), tipo="read", concedido=False,
                            motivo=error.bilingue, **linea))
        raise error

    def _anotar(self, acceso: Acceso) -> None:
        self._accesos.append(acceso)

    # -- lectura del registro ---------------------------------------------
    @property
    def accesos(self) -> tuple[Acceso, ...]:
        return tuple(self._accesos)

    def denegados(self) -> tuple[Acceso, ...]:
        return tuple(a for a in self._accesos if not a.concedido)

    def lecturas_del_test(self) -> tuple[Acceso, ...]:
        """Todo lo que ha tocado un rol reservado, concedido o no. Es lo que el
        106 publica como «uso del test»."""
        return tuple(a for a in self._accesos
                     if a.tipo == "read" and a.rol in ROLES_RESERVADOS)

    def etiqueta_de_evidencia(self, artefacto: str) -> str:
        """Qué vale lo medido para ese artefacto. Lo decide el REGISTRO, no quien
        escribe el informe: es el único que sabe si alguien había mirado antes.

        * `independent_test` solo para la PRIMERA evaluación en el test
          reservado, con el pipeline congelado y sin contaminación previa;
        * `repeated_test_use` para las siguientes: el número es correcto y la
          independencia ya no;
        * `test_used_for_development` en cuanto se ha aprendido del test;
        * `external_validation` para la cohorte externa;
        * `not_evaluated` cuando no se ha medido — que no es un cero ni un
          aprobado.
        """
        exigir_texto(artefacto, "artefacto")
        mias = self._evaluaciones.get(artefacto, [])
        contaminacion = self._orden_de_contaminacion
        limpias = [(orden, rol) for orden, rol in mias
                   if contaminacion is None or orden < contaminacion]
        if not limpias:
            if contaminacion is not None:
                return "test_used_for_development"
            return "not_evaluated"

        en_test = [orden for orden, rol in limpias if rol == "test"]
        if not en_test:
            return "external_validation"
        primera_del_estudio = min(
            (orden for evals in self._evaluaciones.values() for orden, rol in evals
             if rol == "test"), default=None)
        return "independent_test" if min(en_test) == primera_del_estudio else "repeated_test_use"

    # -- serialización -----------------------------------------------------
    def a_json(self) -> dict[str, Any]:
        return {
            "schema": ESQUEMA_DEL_REGISTRO,
            "schema_version": ESTUDIO_SCHEMA_VERSION,
            "estudio": self.estudio,
            "split_plan_digest": self.plan.digest(),
            "frozen_artifact": self._congelado,
            "test_used_for_development": self.test_usado_para_desarrollar,
            "accesses": [a.a_json() for a in self._accesos],
        }

    def digest(self) -> str:
        return digest_canonico(self.a_json())


class LectorDeParticiones:
    """El ÚNICO camino a los datos de una partición.

    Existe para que «leer el test» no sea una decisión de buena fe de cada
    llamante: quien quiera las filas de un rol tiene que decir en qué fase está y
    para qué las quiere, y si no le corresponde no las recibe. Un motor que se
    salte esto y lea el CSV por su cuenta no queda registrado — y por eso el
    103-C3 y el 102-C1 reciben los datos por aquí y no por una ruta.
    """

    def __init__(self, plan: SplitPlan, registro: RegistroDeAccesos) -> None:
        if registro.plan is not plan:
            raise EsquemaInvalido("hay_duplicados", campo="split_plan",
                                  valor=repr(plan.plan_id))
        self.plan = plan
        self.registro = registro

    def filas(self, rol: str, *, proposito: str, candidato: str | None = None,
              artefacto: str | None = None) -> tuple[str, ...]:
        """Los ids de observación de ese rol, si el protocolo lo permite."""
        observaciones = self.plan.observaciones_del_rol(rol)
        self.registro.anotar(rol, proposito=proposito, candidato=candidato,
                             artefacto=artefacto, observaciones=len(observaciones))
        return observaciones

    def unidades(self, rol: str, *, proposito: str, candidato: str | None = None,
                 artefacto: str | None = None) -> tuple[str, ...]:
        """Las unidades (pacientes, centros) de ese rol, con la misma vigilancia.

        Lleva `artefacto` como las filas porque el remuestreo por unidad del
        105-C2 se hace sobre el test: sin poder declarar qué se está evaluando,
        la única forma de calcular un intervalo sería saltarse el guardia.
        """
        unidades = self.plan.unidades_del_rol(rol)
        self.registro.anotar(rol, proposito=proposito, candidato=candidato,
                             artefacto=artefacto, observaciones=len(unidades),
                             granularidad="units")
        return unidades
