# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los esquemas comunes del estudio — contrato 104-C0.

Este paquete es lo PRIMERO del programa 101-106 y lo que todos comparten: las
formas de documento (`ProblemSpec`, `SplitPlan`, `PredictionRecord`,
`MetricSpec`, `FittedPipelineSpec`, `FitResult`, `EvaluationResult`,
`SelectionDecision`), los vocabularios cerrados —estados incluidos—, la política
de versiones y el registro de accesos que hace cumplir el protocolo.

    from matrixai.estudio import RegistroDeAccesos, SplitPlan

    plan = SplitPlan(plan_id="p1", split_type="iid", observation_id_field="id",
                     assignments={"o1": "development", "o2": "test"})
    registro = RegistroDeAccesos(plan)
    with registro.fase("development"):
        registro.anotar("development", proposito="fit")   # concedido
        registro.anotar("test", proposito="fit")           # FugaDeTest

Aquí NO hay motores, ni fórmulas, ni búsqueda: eso es de 102, 105 y 104-C1..C5.
Y `matrixai-core` sigue siendo stdlib puro — `dataclasses`, `typing`, `hashlib`,
`json` y el canonicalizador JCS del 81. Nada más.
"""

from __future__ import annotations

from matrixai.estudio.accesos import (
    Acceso,
    LectorDeParticiones,
    RegistroDeAccesos,
)
from matrixai.estudio.errores import (
    ErrorDeEstudio,
    EsquemaInvalido,
    FugaDeTest,
    MigracionImposible,
    ProtocoloRoto,
)
from matrixai.estudio.esquemas import (
    Aptitud,
    EvaluationResult,
    FitResult,
    FittedPipelineSpec,
    Horizonte,
    MetricSpec,
    PoliticaDeDecision,
    PredictionRecord,
    ProblemSpec,
    Recursos,
    Restriccion,
    SelectionDecision,
    SplitPlan,
    ValorDeMetrica,
    version_tras_aprender_del_test,
)
from matrixai.estudio.metricas import (
    CLIP_LOG_LOSS,
    INFORME_VERSION,
    METRICAS_DIFERIDAS,
    REGISTRO,
    UMBRAL_POR_DEFECTO,
    EntradaNoMedible,
    Indefinida,
    InformeMetrico,
    MatrizDeConfusion,
    MetricaAplazada,
    MetricaDesconocida,
    MetricaRegistrada,
    Muestra,
    aplazamiento,
    aptitud,
    calcular,
    catalogo,
    digest_del_catalogo,
    direccion_de,
    distancia_al_ideal,
    es_mejor,
    especificacion,
    evaluar,
    matriz_de_confusion,
    metricas_aplicables,
    ordenar_por,
)
from matrixai.estudio.migracion import (
    ESTUDIO_SCHEMA_VERSION,
    MIGRACIONES,
    RegistroDeMigraciones,
)
from matrixai.estudio.textos import IDIOMAS, MOTIVOS, motivo
from matrixai.estudio.validacion import digest_canonico
from matrixai.estudio.vocabulario import (
    ESTADOS,
    ETIQUETAS_DE_EVIDENCIA,
    FASES,
    PROPOSITOS,
    RESULTADOS_DE_SELECCION,
    ROLES,
    ROLES_DE_DESARROLLO,
    ROLES_RESERVADOS,
    TAREAS,
    TIPOS_DE_PARTICION,
)

__all__ = [
    "Acceso", "Aptitud", "CLIP_LOG_LOSS", "ESTADOS", "ESTUDIO_SCHEMA_VERSION",
    "ETIQUETAS_DE_EVIDENCIA", "ErrorDeEstudio", "EsquemaInvalido",
    "EvaluationResult", "FASES", "FitResult", "FittedPipelineSpec", "FugaDeTest",
    "Horizonte", "IDIOMAS", "LectorDeParticiones", "MIGRACIONES", "MOTIVOS",
    "INFORME_VERSION", "METRICAS_DIFERIDAS", "REGISTRO", "UMBRAL_POR_DEFECTO",
    "EntradaNoMedible", "Indefinida", "InformeMetrico", "MatrizDeConfusion",
    "MetricaAplazada", "MetricaDesconocida", "MetricaRegistrada", "Muestra",
    "aplazamiento", "aptitud", "calcular", "catalogo", "digest_del_catalogo",
    "direccion_de", "distancia_al_ideal", "es_mejor", "especificacion", "evaluar",
    "matriz_de_confusion", "metricas_aplicables", "ordenar_por",
    "MetricSpec", "MigracionImposible", "PROPOSITOS", "PoliticaDeDecision",
    "PredictionRecord", "ProblemSpec", "ProtocoloRoto", "RESULTADOS_DE_SELECCION",
    "ROLES", "ROLES_DE_DESARROLLO", "ROLES_RESERVADOS", "Recursos",
    "RegistroDeAccesos", "RegistroDeMigraciones", "Restriccion",
    "SelectionDecision", "SplitPlan", "TAREAS", "TIPOS_DE_PARTICION",
    "ValorDeMetrica", "digest_canonico", "motivo",
    "version_tras_aprender_del_test",
]
