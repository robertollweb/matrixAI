# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Pronóstico temporal general (contrato 164, C1): ventanas por serie, referencias ingenuas y
métrica relativa sobre el MISMO reparto hacia delante que ya existe (`proponer_particion`, 140).

Qué hace este módulo y qué NO:

- `construir_ventanas` arma, para un horizonte `h` (en PASOS de la serie), el objetivo `{y}_target_h{h}`
  y los retardos `{col}_lag1..w`, **por serie** (`group_by` de `lag_window` / `shift_target`) y con el
  linaje anti-fuga del pipeline (`check_anti_leakage`): toda entrada tiene desplazamiento <= 0 o se
  lanza `PronosticoError`. Declara lo que pasó: filas de entrada, filas que quedan, filas quitadas por
  el borde de cada serie y por huecos, series, regularidad del calendario.
- Dos referencias, medidas con el mismo examen que el modelo: **persistencia** (el último valor
  conocido, `y(t)`) y **estacional ingenua** (el valor del mismo punto de la estación anterior,
  `y(t + h - m·ceil(h/m))`; con `m` dado por la persona, nunca adivinado).
- `evaluar_hacia_delante` parte las ventanas con `proponer_particion(split_type="temporal")` con un
  hueco >= h pasos, ajusta el modelo por pliegue y en la prueba reservada, y publica la nota del
  modelo SIEMPRE junto a la de cada referencia, con la mejora relativa y la MASE.
- `verificar_sin_fuga` es el banco: el objetivo de cada fila de entrenamiento no puede caer después
  del origen más temprano de lo que se valida.

La escala de la MASE es el error absoluto medio de la estacional ingenua sobre las filas de
ENTRENAMIENTO de ese pliegue (misma definición de pronóstico que la que se compara; si vale 0 la MASE
es `None`, no un número inventado).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matrixai.training.dataset_pipeline import (
    PipelineError, _is_null, _posiciones_por_serie, check_anti_leakage, run_pipeline,
)

_SEGUNDOS_POR_DEFECTO_DE_SERIE_UNICA = "__serie_unica__"


class PronosticoError(ValueError):
    """Un pronóstico que no se puede plantear tal como se pidió: error accionable, nunca un resultado a medias."""


@dataclass(frozen=True)
class Ventanas:
    filas: list[dict[str, str]]
    campo_id: str
    campo_tiempo: str
    campo_serie: str | None
    columna_objetivo: str
    columna_persistencia: str
    columna_estacional: str
    columnas_de_entrada: tuple[str, ...]
    horizonte: int
    estacion: int | None
    retraso_estacional: int | None
    linaje: dict[str, int]
    tiempo_por_id: dict[str, float]
    tiempo_del_objetivo_por_id: dict[str, float]
    paso_mediano: float
    declaracion: dict[str, Any] = field(default_factory=dict)


def _numero(valor: Any) -> float:
    return float(str(valor).strip())


def construir_ventanas(
    filas: Sequence[Mapping[str, Any]], *, tiempo: str, objetivo: str, horizonte: int, retardos: int,
    serie: str | None = None, columnas_con_retardo: Sequence[str] = (), estacion: int | None = None,
    campo_id: str = "_fila", tokens_de_ausencia: set[str] | None = None,
) -> Ventanas:
    """`horizonte` y `retardos` son pasos de la serie (no unidades de tiempo). El objetivo siempre lleva retardos."""
    from matrixai.training.particion_por_diseno_tiempo import Rechazo, convertir_tiempos

    if not filas:
        raise PronosticoError("construir_ventanas necesita al menos una fila.")
    for nombre, v in (("horizonte", horizonte), ("retardos", retardos)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise PronosticoError(f"{nombre} debe ser un entero >= 1 (recibido {v!r}).")
    if estacion is not None and (not isinstance(estacion, int) or isinstance(estacion, bool) or estacion < 1):
        raise PronosticoError(f"estacion debe ser un entero >= 1 (recibido {estacion!r}).")
    cols = list(filas[0].keys())
    for c, rol in ((tiempo, "tiempo"), (objetivo, "objetivo"), *(((serie, "serie"),) if serie else ())):
        if c not in cols:
            raise PronosticoError(f"La columna de {rol} {c!r} no existe. Columnas: {cols}.")
    if campo_id in cols and campo_id != "_fila":
        ids_entrada = [str(f[campo_id]) for f in filas]
    else:
        ids_entrada = [str(i) for i in range(len(filas))]
    if len(set(ids_entrada)) != len(ids_entrada):
        raise PronosticoError(f"La columna de id {campo_id!r} tiene valores repetidos.")
    extras = [c for c in columnas_con_retardo if c != objetivo]
    for c in extras:
        if c not in cols:
            raise PronosticoError(f"La columna con retardo {c!r} no existe. Columnas: {cols}.")
    if serie is not None:
        for i, f in enumerate(filas):
            if _is_null(f.get(serie), tokens_de_ausencia):
                raise PronosticoError(f"La fila {i} no tiene valor en la columna de serie {serie!r}: no se adivina.")

    try:
        conv = convertir_tiempos({str(i): f.get(tiempo) for i, f in enumerate(filas)}, tiempo)
    except Rechazo as rechazo:
        raise PronosticoError(rechazo.bloqueo().motivo) from None
    segundos = conv.valores
    orden = sorted(range(len(filas)), key=lambda i: (segundos[str(i)], i))
    base: list[dict[str, str]] = []
    t_base: list[float] = []
    for i in orden:
        fila = {k: ("" if v is None else str(v)) for k, v in filas[i].items()}
        fila[campo_id] = ids_entrada[i]
        base.append(fila)
        t_base.append(segundos[str(i)])
    if campo_id not in cols:
        cols = cols + [campo_id]

    # Series y calendario: lo medido, no lo supuesto. Dos filas en el mismo instante y serie no tienen orden.
    clave_serie = serie if serie is not None else _SEGUNDOS_POR_DEFECTO_DE_SERIE_UNICA
    if serie is None:
        for fila in base:
            fila[clave_serie] = "unica"
    t_por_id = {f[campo_id]: t for f, t in zip(base, t_base)}
    posiciones = _posiciones_por_serie(base, clave_serie, list(base[0].keys()), "construir_ventanas")
    duplicados = 0
    pasos: list[float] = []
    irregulares = 0
    for pos in posiciones:
        ts = [t_base[i] for i in pos]
        d = [b - a for a, b in zip(ts, ts[1:])]
        duplicados += sum(1 for x in d if x == 0)
        pasos.extend(x for x in d if x > 0)
        if d and len({x for x in d}) > 1:
            irregulares += 1
    if duplicados:
        raise PronosticoError(
            f"{duplicados} par(es) de filas de la misma serie comparten instante en {tiempo!r}: no tienen orden "
            "entre sí y el retardo dependería de cuál va primero. Quita o agrega los duplicados.")
    paso_mediano = float(statistics.median(pasos)) if pasos else 0.0
    if paso_mediano <= 0:
        raise PronosticoError("Ninguna serie tiene dos filas: no hay paso de tiempo que medir ni retardo que construir.")

    lag_cols = [objetivo, *extras]
    operaciones: list[dict[str, Any]] = [
        {"op": "lag_window", "columns": lag_cols, "window": retardos, "group_by": clave_serie},
        {"op": "shift_target", "column": objetivo, "horizon": horizonte, "group_by": clave_serie},
    ]
    try:
        resultado = run_pipeline(base, operaciones, tokens_de_ausencia=tokens_de_ausencia)
    except PipelineError as e:
        raise PronosticoError(str(e)) from None
    filas_p = resultado.rows
    col_objetivo = f"{objetivo}_target_h{horizonte}"
    cols_retardo = [f"{c}_lag{i}" for c in lag_cols for i in range(1, retardos + 1)]

    # Referencias, por serie y hacia atrás: persistencia y(t); estacional y(t + h - m·ceil(h/m)).
    retraso_est = None if estacion is None else estacion * math.ceil(horizonte / estacion) - horizonte
    col_pers, col_est = f"{objetivo}_ref_persistencia", f"{objetivo}_ref_estacional"
    linaje = dict(resultado.column_offsets)
    linaje[col_pers] = linaje.get(objetivo, 0)
    for fila in filas_p:
        fila[col_pers] = fila[objetivo]
        fila[col_est] = ""
    if retraso_est is not None:
        origen = [f[objetivo] for f in filas_p]
        for pos in posiciones:
            for k, idx in enumerate(pos):
                filas_p[idx][col_est] = origen[pos[k - retraso_est]] if k - retraso_est >= 0 else ""
        linaje[col_est] = linaje.get(objetivo, 0) - retraso_est
    # tiempo del objetivo por fila (para el banco de fugas)
    t_obj: dict[str, float] = {}
    for pos in posiciones:
        for k, idx in enumerate(pos):
            if k + horizonte < len(pos):
                t_obj[filas_p[idx][campo_id]] = t_base[pos[k + horizonte]]

    entradas = [*cols_retardo, *lag_cols]
    fugas = check_anti_leakage(resultado, entradas)
    fugas += check_anti_leakage(type(resultado)(rows=[], column_offsets=linaje), [col_pers] + ([col_est] if retraso_est is not None else []))
    if fugas:
        raise PronosticoError("Fuga de información temporal en las entradas: " + " | ".join(fugas))

    generadas = [col_objetivo, *cols_retardo] + ([col_est] if retraso_est is not None else [])
    quitadas_borde = 0
    quitadas_hueco = 0
    quedan: list[dict[str, str]] = []
    for fila in filas_p:
        if any(_is_null(fila.get(c), tokens_de_ausencia) for c in lag_cols):
            quitadas_hueco += 1       # un valor ausente en el dato original (no un borde)
        elif any(_is_null(fila.get(c), tokens_de_ausencia) for c in generadas):
            quitadas_borde += 1       # solo faltan columnas desplazadas: el arranque o el final de su serie
        else:
            quedan.append(fila)
    if not quedan:
        raise PronosticoError(
            f"Con horizonte {horizonte}, {retardos} retardos y las series dadas no queda ninguna fila "
            f"completa de {len(filas)}.")
    persist_ok = sum(1 for f in quedan if not _is_null(f[col_pers], tokens_de_ausencia))
    if persist_ok != len(quedan):
        raise PronosticoError("La persistencia (el valor actual del objetivo) está ausente en filas que quedan.")
    series_cortas = sum(1 for pos in posiciones if len(pos) <= retardos + horizonte)
    declaracion = {
        "horizonte_pasos": horizonte, "retardos": retardos, "serie": serie, "objetivo": objetivo,
        "columnas_con_retardo": lag_cols, "estacion": estacion, "retraso_estacional": retraso_est,
        "estacional_igual_a_persistencia": retraso_est == 0,
        "filas_de_entrada": len(filas), "filas_resultantes": len(quedan),
        "filas_quitadas_por_el_borde_de_la_serie": quitadas_borde,
        "filas_quitadas_por_valores_ausentes": quitadas_hueco,
        "series": len(posiciones), "series_sin_filas_completas": series_cortas,
        "series_con_calendario_irregular": irregulares, "paso_mediano_segundos_o_unidades": paso_mediano,
        "linaje": {c: linaje[c] for c in (*entradas, col_objetivo, col_pers) if c in linaje},
    }
    return Ventanas(
        filas=quedan, campo_id=campo_id, campo_tiempo=tiempo, campo_serie=serie, columna_objetivo=col_objetivo,
        columna_persistencia=col_pers, columna_estacional=col_est, columnas_de_entrada=tuple(entradas),
        horizonte=horizonte, estacion=estacion, retraso_estacional=retraso_est, linaje=linaje,
        tiempo_por_id={f[campo_id]: t_por_id[f[campo_id]] for f in quedan},
        tiempo_del_objetivo_por_id={f[campo_id]: t_obj[f[campo_id]] for f in quedan},
        paso_mediano=paso_mediano, declaracion=declaracion)


# --------------------------------------------------------------------------------------------------
# Examen hacia delante contra las referencias
# --------------------------------------------------------------------------------------------------

def _metricas(real: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    n = len(real)
    err = [abs(a - b) for a, b in zip(real, pred)]
    return {"mae": sum(err) / n, "rmse": math.sqrt(sum(e * e for e in err) / n)}


def _comparar(real, modelo, pers, est, escala_est: float | None) -> dict[str, Any]:
    """La nota del modelo SIEMPRE con la de cada referencia (invariante 5). `mejora_*` = 1 - error del
    modelo / error de la referencia (positiva: gana); `None` si la referencia no se equivoca nunca (no se divide
    por cero ni se inventa una cifra). `mase` = MAE del modelo / `escala_est` (ver cabecera)."""
    m = _metricas(real, modelo)
    out: dict[str, Any] = {"n": len(real), "modelo": m, "persistencia": _metricas(real, pers)}
    if est is not None:
        out["estacional"] = _metricas(real, est)
    def mejora(ref):
        return None if ref["mae"] == 0 else 1.0 - m["mae"] / ref["mae"]
    out["mejora_mae_sobre_persistencia"] = mejora(out["persistencia"])
    out["mejora_rmse_sobre_persistencia"] = (None if out["persistencia"]["rmse"] == 0
                                             else 1.0 - m["rmse"] / out["persistencia"]["rmse"])
    if est is not None:
        out["mejora_mae_sobre_estacional"] = mejora(out["estacional"])
        out["mase"] = None if not escala_est else m["mae"] / escala_est
    return out


def verificar_sin_fuga(v: Ventanas, pliegues: Sequence[tuple[Sequence[str], Sequence[str]]]) -> list[str]:
    """Banco de fugas. Para cada `(entrena, valida)`: el objetivo de ninguna fila de entrenamiento puede caer DESPUÉS del
    origen más temprano de lo que se valida (sería entrenar con un valor que, en ese origen, aún no ha ocurrido), ni
    puede solaparse una fila en los dos lados. Devuelve las violaciones, con números."""
    problemas = []
    for k, (entrena, valida) in enumerate(pliegues):
        if not entrena or not valida:
            continue
        comunes = set(entrena) & set(valida)
        if comunes:
            problemas.append(f"pliegue {k}: {len(comunes)} fila(s) en entrena y valida")
        origen_min = min(v.tiempo_por_id[i] for i in valida)
        peor = max(v.tiempo_del_objetivo_por_id[i] for i in entrena)
        if peor > origen_min:
            n = sum(1 for i in entrena if v.tiempo_del_objetivo_por_id[i] > origen_min)
            problemas.append(f"pliegue {k}: {n} fila(s) de entrenamiento con objetivo en {peor:g}, "
                             f"después del primer origen validado ({origen_min:g})")
    return problemas


def evaluar_hacia_delante(
    v: Ventanas, *, columnas_de_entrada: Sequence[str] | None = None, fabricar_modelo: Callable[[], Any],
    plan_id: str = "pronostico", seed: int = 0, folds: int = 3, test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Pliegues hacia delante (ventana que se expande, con hueco >= horizonte × paso) y prueba reservada: el MISMO
    `proponer_particion(temporal)` que usa el estudio. `fabricar_modelo()` devuelve algo con `fit(X, y)`/`predict(X)`;
    las entradas deben ser numéricas (la codificación es de quien llama)."""
    import numpy as np
    from matrixai.estudio.esquemas import Horizonte
    from matrixai.training.particion_por_diseno import proponer_particion

    columnas = list(columnas_de_entrada if columnas_de_entrada is not None else v.columnas_de_entrada)
    for c in columnas:
        linea = v.linaje.get(c, 0)
        if linea > 0:
            raise PronosticoError(f"La entrada {c!r} tiene linaje +{linea}: es del futuro.")
    hueco = Horizonte(v.horizonte * v.paso_mediano, "seconds")
    prop = proponer_particion(
        v.filas, plan_id=plan_id, observation_id_field=v.campo_id, split_type="temporal", seed=seed,
        test_fraction=test_fraction, folds=folds, time_column=v.campo_tiempo, gap=hueco)
    if not prop.es_viable:
        return {"viable": False, "bloqueos": [{"clave": b.clave, "motivo": b.motivo} for b in prop.bloqueos],
                "declaracion": v.declaracion}
    por_id = {f[v.campo_id]: f for f in v.filas}
    est_ok = v.retraso_estacional is not None

    def matriz(ids):
        return np.array([[_numero(por_id[i][c]) for c in columnas] for i in ids], float)

    def y(ids, col):
        return [_numero(por_id[i][col]) for i in ids]

    def examen(entrena, valida):
        modelo = fabricar_modelo()
        modelo.fit(matriz(entrena), np.array(y(entrena, v.columna_objetivo), float))
        pred = [float(x) for x in modelo.predict(matriz(valida))]
        escala = None
        if est_ok:
            escala = _metricas(y(entrena, v.columna_objetivo), y(entrena, v.columna_estacional))["mae"]
        return _comparar(y(valida, v.columna_objetivo), pred, y(valida, v.columna_persistencia),
                         y(valida, v.columna_estacional) if est_ok else None, escala)

    pliegues = [(p.entrena, p.valida) for p in prop.pliegues.pliegues]
    desarrollo = [i for i, r in prop.plan.assignments.items() if r == "development"]
    prueba = [i for i, r in prop.plan.assignments.items() if r == "test"]
    # La prueba reservada: ajuste con TODO el desarrollo, que (por el hueco) termina antes de que ella empiece.
    resultado = {
        "viable": True, "declaracion": v.declaracion,
        "hueco_segundos_o_unidades": hueco.magnitud,
        "pliegues": [examen(e, va) for e, va in pliegues],
        "prueba": examen(desarrollo, prueba),
        "violaciones_de_fuga": verificar_sin_fuga(v, [*pliegues, (desarrollo, prueba)]),
        "limites": [l.clave for l in prop.limites],
    }
    return resultado


# --------------------------------------------------------------------------------------------------
# Predecir con datos nuevos (contrato 164, C4): el contexto que el modelo necesita
# --------------------------------------------------------------------------------------------------

def contexto_necesario(*, retardos: int, columnas_con_retardo: Sequence[str], columnas_sin_retardo: Sequence[str] = (),
                       tiempo: str, serie: str | None, objetivo: str, horizonte: int) -> dict[str, Any]:
    """Lo que hay que TENER para pronosticar con datos nuevos, dicho con números (D4 = A, invariante 7). Para predecir
    el instante `t` el modelo lee los `retardos` valores anteriores de cada columna con retardo y los valores de `t`
    mismo (las entradas «actuales»): por serie hacen falta, pues, `retardos + 1` filas, las últimas. Con menos no se
    rellena: esa serie no se predice."""
    columnas = [tiempo, *([serie] if serie else []), objetivo]
    for c in (*columnas_con_retardo, *columnas_sin_retardo):
        if c not in columnas:
            columnas.append(c)
    return {"filas_por_serie": retardos + 1, "filas_de_historial": retardos, "fila_de_origen": 1,
            "columnas": columnas, "columna_de_tiempo": tiempo, "columna_de_serie": serie, "objetivo": objetivo,
            "columnas_con_retardo": list(dict.fromkeys([objetivo, *columnas_con_retardo])),
            "columnas_sin_retardo": list(columnas_sin_retardo), "horizonte_pasos": horizonte,
            "una_fila_suelta": False}


@dataclass(frozen=True)
class SerieSinContexto:
    """Una serie que NO se predice, y por qué: `clave` más los números (el redactor es quien llama, en es/en)."""
    serie: str | None
    clave: str                      # historial_corto | valores_ausentes | instantes_repetidos | sin_serie
    tiene: int = 0                  # filas de la serie en el fichero
    necesita: int = 0
    columnas: tuple[str, ...] = ()  # las que faltan (valores_ausentes)
    filas: int = 0                  # sin_serie: cuántas filas


@dataclass(frozen=True)
class SeriePredecible:
    serie: str | None
    entradas: dict[str, str]        # justo las columnas con las que se entrenó, listas para el modelo
    origen: str                     # el valor de la columna de tiempo en la última fila, tal cual llegó
    origen_segundos: float
    filas_usadas: int
    calendario_irregular: bool


@dataclass(frozen=True)
class ContextoDePrediccion:
    predecibles: list[SeriePredecible]
    rechazadas: list[SerieSinContexto]
    series: int
    es_fecha: bool
    formato_de_fecha: str | None


def construir_contexto_de_prediccion(
    filas: Sequence[Mapping[str, Any]], *, tiempo: str, objetivo: str, retardos: int, serie: str | None = None,
    columnas_con_retardo: Sequence[str] = (), columnas_sin_retardo: Sequence[str] = (),
    formato_de_fecha: str | None = None, tokens_de_ausencia: set[str] | None = None,
) -> ContextoDePrediccion:
    """Para CADA serie del fichero, la fila de entradas del ÚLTIMO instante: los retardos se arman con el MISMO
    `lag_window` por serie que el estudio (`construir_ventanas`), sin `shift_target` (no hay objetivo futuro que
    conocer). Una serie sin las `retardos + 1` filas completas que hacen falta, con un valor ausente en lo que el
    modelo lee, o con dos filas en el mismo instante, NO se predice y sale en `rechazadas` con su porqué: ni se rellena
    con ceros ni se completa con filas de otra serie. Las demás se predicen."""
    from datetime import datetime

    from matrixai.training.particion_por_diseno_tiempo import Rechazo, _ORIGEN, convertir_tiempos

    if not filas:
        raise PronosticoError("construir_contexto_de_prediccion necesita al menos una fila.")
    if not isinstance(retardos, int) or isinstance(retardos, bool) or retardos < 1:
        raise PronosticoError(f"retardos debe ser un entero >= 1 (recibido {retardos!r}).")
    cols = list(filas[0].keys())
    extras = [c for c in columnas_con_retardo if c != objetivo]
    for c, rol in ((tiempo, "tiempo"), (objetivo, "objetivo"), *(((serie, "serie"),) if serie else ()),
                   *((x, "retardo") for x in extras), *((x, "entrada") for x in columnas_sin_retardo)):
        if c not in cols:
            raise PronosticoError(f"La columna de {rol} {c!r} no existe. Columnas: {cols}.")
    lag_cols = [objetivo, *extras]

    sin_serie = 0
    utiles: list[int] = []
    for i, f in enumerate(filas):
        if serie is not None and _is_null(f.get(serie), tokens_de_ausencia):
            sin_serie += 1
        else:
            utiles.append(i)
    rechazadas: list[SerieSinContexto] = []
    if sin_serie:
        rechazadas.append(SerieSinContexto(None, "sin_serie", filas=sin_serie))
    if not utiles:
        return ContextoDePrediccion([], rechazadas, 0, False, None)

    crudas = {str(i): filas[i].get(tiempo) for i in utiles}
    try:
        if formato_de_fecha:      # el formato con que se ENTRENÓ: un lote de pocas filas puede ser ambiguo por sí solo
            segundos = {k: (datetime.strptime(str(v).strip(), formato_de_fecha) - _ORIGEN).total_seconds()
                        for k, v in crudas.items()}
            es_fecha, formato = True, formato_de_fecha
        else:
            conv = convertir_tiempos(crudas, tiempo)
            segundos, es_fecha, formato = conv.valores, conv.es_fecha, getattr(conv, "formato", None)
    except Rechazo as rechazo:
        raise PronosticoError(rechazo.bloqueo().motivo) from None
    except (ValueError, TypeError):
        raise PronosticoError(f"La columna de tiempo {tiempo!r} trae valores que no se leen con el formato "
                              f"{formato_de_fecha!r} con el que se entrenó el modelo.") from None

    orden = sorted(utiles, key=lambda i: (segundos[str(i)], i))
    clave_serie = serie if serie is not None else _SEGUNDOS_POR_DEFECTO_DE_SERIE_UNICA
    base: list[dict[str, str]] = []
    t_base: list[float] = []
    for i in orden:
        fila = {k: ("" if v is None else str(v)) for k, v in filas[i].items()}
        fila["_fila"] = str(i)
        if serie is None:
            fila[clave_serie] = "unica"
        base.append(fila)
        t_base.append(segundos[str(i)])
    try:
        resultado = run_pipeline(base, [{"op": "lag_window", "columns": lag_cols, "window": retardos,
                                         "group_by": clave_serie}], tokens_de_ausencia=tokens_de_ausencia)
    except PipelineError as e:
        raise PronosticoError(str(e)) from None
    cols_retardo = [f"{c}_lag{i}" for c in lag_cols for i in range(1, retardos + 1)]
    fugas = check_anti_leakage(resultado, cols_retardo)
    if fugas:
        raise PronosticoError("Fuga de información temporal en las entradas: " + " | ".join(fugas))
    filas_p = resultado.rows
    posiciones = _posiciones_por_serie(base, clave_serie, list(base[0].keys()), "construir_contexto_de_prediccion")

    necesita = retardos + 1
    entradas = [*cols_retardo, *lag_cols, *[c for c in columnas_sin_retardo if c not in lag_cols]]
    predecibles: list[SeriePredecible] = []
    for pos in posiciones:
        nombre = None if serie is None else base[pos[0]][serie]
        ts = [t_base[i] for i in pos]
        if any(b == a for a, b in zip(ts, ts[1:])):
            rechazadas.append(SerieSinContexto(nombre, "instantes_repetidos", tiene=len(pos), necesita=necesita))
            continue
        if len(pos) < necesita:
            rechazadas.append(SerieSinContexto(nombre, "historial_corto", tiene=len(pos), necesita=necesita))
            continue
        ultima = filas_p[pos[-1]]
        faltan = tuple(c for c in entradas if _is_null(ultima.get(c), tokens_de_ausencia))
        if faltan:
            rechazadas.append(SerieSinContexto(nombre, "valores_ausentes", tiene=len(pos), necesita=necesita,
                                               columnas=faltan))
            continue
        ventana = ts[-necesita:]
        irregular = len({b - a for a, b in zip(ventana, ventana[1:])}) > 1
        origen = filas[int(ultima["_fila"])].get(tiempo)
        predecibles.append(SeriePredecible(
            serie=nombre, entradas={c: str(ultima[c]) for c in entradas}, origen=str(origen).strip(),
            origen_segundos=ts[-1], filas_usadas=necesita, calendario_irregular=irregular))
    return ContextoDePrediccion(predecibles, rechazadas, len(posiciones), es_fecha, formato)
