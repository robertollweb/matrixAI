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

Aquí NO hay motores, ni fórmulas propias de un solo estudio: eso es de 102 y
105. `competicion.py` (104-C1) sí vive aquí — `EstudioSpec`/`Leaderboard` son
FORMA, no ejecución; la orquestación real es de `matrixai-engines` (104-C2).
Y `matrixai-core` sigue siendo stdlib puro — `dataclasses`, `typing`, `hashlib`,
`json` y el canonicalizador JCS del 81. Nada más.
"""

from __future__ import annotations

from matrixai.estudio.accesos import (
    Acceso,
    LectorDeParticiones,
    RegistroDeAccesos,
)
from matrixai.estudio.calibracion import (
    METODOS_DE_BINS,
    METODOS_DE_RECALIBRACION,
    BinDeFiabilidad,
    CurvaDeFiabilidad,
    RecalibracionLogistica,
    ajustar_recalibracion_logistica,
    aplicar_recalibracion,
    curva_de_fiabilidad,
)
from matrixai.estudio.comparaciones import (
    VEREDICTOS_DE_COMPARACION,
    ComparacionEmparejada,
    comparar_candidatos,
)
from matrixai.estudio.competicion import (
    FASES_DE_LEADERBOARD,
    MODOS_DE_ESTUDIO,
    EntradaDeLeaderboard,
    EstudioSpec,
    Leaderboard,
    ReservaDeTiempo,
    tiempo_de_busqueda_por_motor,
)
from matrixai.estudio.evaluacion_final import (
    RESULTADOS_DE_VALIDACION_FINAL,
    ValidacionFinal,
    evaluar_en_test,
    promover_candidato,
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
from matrixai.estudio.informe_final import InformeDeEvaluacion
from matrixai.estudio.incertidumbre import (
    DISENOS_SOPORTADOS,
    ESTIMANDOS_DE_INTERVALO,
    METODOS_DE_REMUESTREO,
    PROPORCION_MINIMA_VALIDA,
    UNIDADES_DE_REMUESTREO,
    Intervalo,
    intervalo,
    medir,
)
from matrixai.estudio.seleccion import (
    POLITICA_UTILIDAD_MAXIMA,
    seleccionar,
)
from matrixai.estudio.segmentos import (
    MINIMO_EVENTOS_POR_SEGMENTO,
    MINIMO_FILAS_POR_SEGMENTO,
    AnalisisDeSegmento,
    ImportanciaDeVariable,
    analizar_segmento,
    importancia_por_permutacion,
)
from matrixai.estudio.migracion import (
    ESTUDIO_SCHEMA_VERSION,
    MIGRACIONES,
    RegistroDeMigraciones,
)
from matrixai.estudio.textos import IDIOMAS, MOTIVOS, motivo
from matrixai.estudio.umbral import elegir_umbral
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
    "Acceso", "Aptitud", "CLIP_LOG_LOSS", "DISENOS_SOPORTADOS", "ESTADOS",
    "ESTIMANDOS_DE_INTERVALO", "ESTUDIO_SCHEMA_VERSION",
    "ETIQUETAS_DE_EVIDENCIA", "ErrorDeEstudio", "EsquemaInvalido",
    "BinDeFiabilidad", "CurvaDeFiabilidad", "METODOS_DE_BINS",
    "METODOS_DE_RECALIBRACION", "RecalibracionLogistica",
    "ajustar_recalibracion_logistica", "aplicar_recalibracion", "curva_de_fiabilidad",
    "VEREDICTOS_DE_COMPARACION", "ComparacionEmparejada", "comparar_candidatos",
    "EntradaDeLeaderboard", "EstudioSpec", "FASES_DE_LEADERBOARD",
    "Leaderboard", "MODOS_DE_ESTUDIO", "ReservaDeTiempo",
    "tiempo_de_busqueda_por_motor",
    "RESULTADOS_DE_VALIDACION_FINAL", "ValidacionFinal", "evaluar_en_test",
    "promover_candidato",
    "InformeDeEvaluacion",
    "EvaluationResult", "FASES", "FitResult", "FittedPipelineSpec", "FugaDeTest",
    "Horizonte", "IDIOMAS", "Intervalo", "LectorDeParticiones", "MIGRACIONES",
    "METODOS_DE_REMUESTREO", "MOTIVOS",
    "INFORME_VERSION", "METRICAS_DIFERIDAS", "PROPORCION_MINIMA_VALIDA",
    "REGISTRO", "UMBRAL_POR_DEFECTO", "UNIDADES_DE_REMUESTREO",
    "POLITICA_UTILIDAD_MAXIMA", "seleccionar",
    "MINIMO_EVENTOS_POR_SEGMENTO", "MINIMO_FILAS_POR_SEGMENTO",
    "AnalisisDeSegmento", "ImportanciaDeVariable", "analizar_segmento",
    "importancia_por_permutacion",
    "EntradaNoMedible", "Indefinida", "InformeMetrico", "MatrizDeConfusion",
    "MetricaAplazada", "MetricaDesconocida", "MetricaRegistrada", "Muestra",
    "aplazamiento", "aptitud", "calcular", "catalogo", "digest_del_catalogo",
    "direccion_de", "distancia_al_ideal", "es_mejor", "especificacion", "evaluar",
    "intervalo", "matriz_de_confusion", "medir", "metricas_aplicables",
    "ordenar_por",
    "MetricSpec", "MigracionImposible", "PROPOSITOS", "PoliticaDeDecision",
    "PredictionRecord", "ProblemSpec", "ProtocoloRoto", "RESULTADOS_DE_SELECCION",
    "ROLES", "ROLES_DE_DESARROLLO", "ROLES_RESERVADOS", "Recursos",
    "RegistroDeAccesos", "RegistroDeMigraciones", "Restriccion",
    "SelectionDecision", "SplitPlan", "TAREAS", "TIPOS_DE_PARTICION",
    "ValorDeMetrica", "digest_canonico", "elegir_umbral", "motivo",
    "version_tras_aprender_del_test",
]
