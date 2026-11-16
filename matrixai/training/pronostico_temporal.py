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
    """Un pronóstico que no se puede plantear tal como se pidió: error accionable, nunca un resultado a medias.

    El mensaje (`str(exc)`) es castellano. 164-rep1 (M-2): los rechazos que la persona provoca con datos normales llevan además
    su `clave` y el motivo en inglés (`en`), redactados AQUÍ (lo que redacta el núcleo se traduce en el núcleo), y su `medida`
    con los números; los demás llevan `clave = None` y quien llama no inventa traducción."""

    def __init__(self, mensaje: str, *, clave: str | None = None, en: str | None = None,
                 medida: Mapping[str, Any] | None = None) -> None:
        super().__init__(mensaje)
        self.clave = clave
        self.en = en
        self.medida = dict(medida or {})

    @property
    def motivo(self) -> dict[str, str] | None:
        """`{es, en}` cuando el núcleo lo redactó en los dos idiomas; `None` si solo hay castellano."""
        return {"es": str(self), "en": self.en} if self.clave and self.en else None


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
    #: 164-rep2 (aud2 A-1): el hueco que el reparto tiene que dejar (segundos o unidades): `horizonte × paso_mediano` o MÁS, si el
    #: calendario lo pide (ver `_elegir_hueco`). `0.0` = el de siempre (`horizonte × paso_mediano`), por compatibilidad.
    hueco: float = 0.0

    def __post_init__(self) -> None:
        if not self.hueco:
            object.__setattr__(self, "hueco", self.horizonte * self.paso_mediano)


def _numero(valor: Any) -> float:
    return float(str(valor).strip())


def construir_ventanas(
    filas: Sequence[Mapping[str, Any]], *, tiempo: str, objetivo: str, horizonte: int, retardos: int,
    serie: str | None = None, columnas_con_retardo: Sequence[str] = (), estacion: int | None = None,
    campo_id: str = "_fila", tokens_de_ausencia: set[str] | None = None, hueco: float | None = None,
) -> Ventanas:
    """`horizonte` y `retardos` son pasos de la serie (no unidades de tiempo). El objetivo siempre lleva retardos.

    `hueco` (segundos o unidades): el que PIDIÓ la persona para el reparto; `None` = lo elige `_elegir_hueco` (164-rep2)."""
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
    primer_duplicado: tuple[str | None, str] | None = None
    pasos: list[float] = []
    irregulares = 0
    paso_de_la_serie: list[float] = []        # 164-rep3 (aud3 M-3): el paso mediano de CADA serie
    for pos in posiciones:
        ts = [t_base[i] for i in pos]
        d = [b - a for a, b in zip(ts, ts[1:])]
        positivos = [x for x in d if x > 0]
        paso_de_la_serie.append(float(statistics.median(positivos)) if positivos else 0.0)
        duplicados += sum(1 for x in d if x == 0)
        if primer_duplicado is None:
            for j, x in enumerate(d):
                if x == 0:     # 164-rep1 (M-2): la serie y el instante, tal como vienen en el fichero (no segundos desde 1970)
                    primer_duplicado = (None if serie is None else base[pos[j]][serie], base[pos[j + 1]][tiempo])
                    break
        pasos.extend(x for x in d if x > 0)
        if d and len({x for x in d}) > 1:
            irregulares += 1
    if duplicados:
        nombre, instante = primer_duplicado or (None, "")
        donde_es = f"la serie «{nombre}»" if nombre is not None else "la serie"
        donde_en = f"series “{nombre}”" if nombre is not None else "the series"
        raise PronosticoError(
            f"{duplicados} par(es) de filas de la misma serie comparten instante en «{tiempo}» (el primero, en {donde_es}, "
            f"«{instante}»): no tienen orden entre sí y el retardo dependería de cuál va primero. Quita o agrega los duplicados.",
            clave="instantes_repetidos",
            en=(f"{duplicados} pair(s) of rows of the same series share an instant in “{tiempo}” (the first one, in {donde_en}, "
                f"“{instante}”): they have no order between them and the lag would depend on which comes first. Remove or "
                "aggregate the duplicates."),
            medida={"pares": duplicados, "serie": nombre, "instante": instante, "columna_de_tiempo": tiempo})
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
    # 164-rep1 — POR QUÉ se quita cada fila, dicho por su CAUSA REAL (I-4 y M-4 de la auditoría 1): la causa sale de la POSICIÓN
    # de la fila en su serie, no de qué columna quedó vacía. «Borde» = le faltan pasos de su propia serie para los retardos (al
    # principio) o para el objetivo (al final); «estación» = tiene retardos y objetivo pero no la referencia estacional (la serie
    # no lleva aún `retraso_estacional` pasos): la quita PEDIR la estación, no los retardos; «ausentes» = un valor ausente en el
    # dato, en esa fila o en una de las que mira (retardos, objetivo, referencia).
    serie_de: dict[int, tuple[int, int, int]] = {}
    for n_serie, pos in enumerate(posiciones):
        for k, idx in enumerate(pos):
            serie_de[idx] = (n_serie, k, len(pos))
    arranque_est = retraso_est if retraso_est is not None else 0
    quitadas_borde = 0
    quitadas_hueco = 0
    quitadas_estacion = 0
    candidatas: list[int] = []
    for idx, fila in enumerate(filas_p):
        _, k, largo = serie_de[idx]
        if any(_is_null(fila.get(c), tokens_de_ausencia) for c in lag_cols):
            quitadas_hueco += 1       # un valor ausente en el dato original de ESTA fila
        elif k < retardos or k + horizonte >= largo:
            quitadas_borde += 1       # el arranque (sin retardos completos) o el final (sin objetivo) de su serie
        elif k < arranque_est:
            quitadas_estacion += 1    # tiene retardos y objetivo: solo le falta la referencia estacional
        elif any(_is_null(fila.get(c), tokens_de_ausencia) for c in generadas):
            quitadas_hueco += 1       # un valor ausente en una fila que esta mira (un retardo, el objetivo o la referencia)
        else:
            candidatas.append(idx)

    # 164-rep1 (I-1) y 164-rep2 (aud2 A-1) — LOS CORTES DEL CALENDARIO. El objetivo «h pasos» va por posición: si entre el origen
    # y el objetivo de una fila su serie da pasos más largos que el mediano (un corte, un fin de semana, un festivo), el objetivo
    # cae más lejos que h × paso. El reparto deja un hueco `G` entre lo que entrena y lo que valida (`inicio − t > G` en los
    # pliegues, `particion_por_diseno_tiempo`; `t < corte − G/2` y `t ≥ corte + G/2` en la prueba): una fila entrena con un
    # valor aún no ocurrido si ALGÚN instante del fichero (de su serie o de otra) cae estrictamente entre su origen + G y su
    # objetivo. Con `necesita` = el último instante del fichero anterior a su objetivo, menos su origen, eso es `necesita > G`.
    #
    # rep1 fijaba G = h × paso mediano y quitaba toda fila con `necesita > G`. En un calendario LABORABLE común con h > 1 eso
    # quitaba del 19 % (h = 2) al 76 % (h = 5) de las filas (aud2 A-1, medido): 5 pasos laborables son 7 días naturales, no 5.
    # Ahora G se ELIGE (`_elegir_hueco`): el que menos filas cuesta, contando cada fila quitada una vez y cada día (o unidad)
    # de hueco por encima de h × paso como las filas que caben en él en cada examen. Con un calendario laborable común sale un
    # hueco de 3 días (h = 2) o 6 (h = 5) y no se quita ninguna fila; con un corte de UNA serie (o un cierre común de semanas) sale
    # más barato quitar sus pocas filas. Si la persona pidió el hueco, es el suyo (nunca menos que h × paso).
    import bisect

    hueco_minimo = horizonte * paso_mediano
    instantes = sorted(set(t_base))
    necesita_por_idx: dict[int, float] = {}
    for idx in candidatas:
        n_serie, k, _ = serie_de[idx]
        t0, t1 = t_base[idx], t_base[posiciones[n_serie][k + horizonte]]
        j = bisect.bisect_left(instantes, t1) - 1          # el último instante del fichero ANTES del objetivo
        necesita_por_idx[idx] = max(0.0, instantes[j] - t0) if j >= 0 else 0.0
    otro_hueco_conservaria: dict[str, Any] | None = None
    if hueco is not None:
        hueco_usado, eleccion = max(float(hueco), hueco_minimo), "pedido"
        # 164-rep3 (aud3 M-2): el hueco PEDIDO se respeta, pero si uno MÁS ANCHO quitaría menos filas (la misma cuenta de
        # `_elegir_hueco`, partiendo del pedido), se dice con sus números: un hueco de h × paso en un calendario laborable
        # quitaba 884 de 1.128 filas sin pista de que con 8 días se conservaban.
        mas_ancho, _ = _elegir_hueco(
            hueco_usado, [necesita_por_idx[i] for i in candidatas], [t_base[i] for i in candidatas])
        if mas_ancho > hueco_usado:
            con_pedido = sum(1 for i in candidatas if necesita_por_idx[i] > hueco_usado)
            con_otro = sum(1 for i in candidatas if necesita_por_idx[i] > mas_ancho)
            if con_otro < con_pedido:
                otro_hueco_conservaria = {"segundos_o_unidades": mas_ancho, "filas_que_quitaria": con_otro,
                                          "filas_que_quita_el_pedido": con_pedido}
    else:
        hueco_usado, eleccion = _elegir_hueco(
            hueco_minimo, [necesita_por_idx[i] for i in candidatas], [t_base[i] for i in candidatas])
    quedan: list[dict[str, str]] = []
    cortes_por_serie: dict[int, dict[str, Any]] = {}
    quitadas_corte = 0
    quitaria_el_minimo = sum(1 for i in candidatas if necesita_por_idx[i] > hueco_minimo)
    for idx in candidatas:
        fila = filas_p[idx]
        n_serie, k, _ = serie_de[idx]
        pos = posiciones[n_serie]
        if necesita_por_idx[idx] > hueco_usado:
            quitadas_corte += 1
            c = cortes_por_serie.setdefault(n_serie, {
                "serie": None if serie is None else base[pos[0]][serie], "filas": 0,
                "primer_origen": base[idx][tiempo], "primer_objetivo": base[pos[k + horizonte]][tiempo]})
            c["filas"] += 1
            continue
        quedan.append(fila)
    # 164-rep3 (aud3 M-3) — UNA SERIE DE OTRO PASO que se pierde ENTERA: su paso mediano es más largo que el hueco del examen
    # (una serie diaria entre series horarias), así que cada objetivo suyo cae con momentos de predecir de otras series en medio
    # y se quitan TODAS sus filas. No es «un corte o un salto»: es otra frecuencia. Cuenta entre las series sin filas completas
    # (el modelo no la ve) y se marca en su corte, para que quien lo redacte lo diga como lo que es.
    quedan_por_serie: dict[int, int] = {}
    for idx in candidatas:
        if necesita_por_idx[idx] <= hueco_usado:
            n_serie = serie_de[idx][0]
            quedan_por_serie[n_serie] = quedan_por_serie.get(n_serie, 0) + 1
    series_de_otro_paso: list[dict[str, Any]] = []
    for n_serie, c in sorted(cortes_por_serie.items()):
        if not quedan_por_serie.get(n_serie) and paso_de_la_serie[n_serie] > hueco_usado:
            c["por_su_paso"] = True
            series_de_otro_paso.append({"serie": c["serie"], "paso_mediano": paso_de_la_serie[n_serie], "filas": c["filas"]})
    if not quedan:
        causas = [(quitadas_borde, "por el borde de cada serie", "at the edge of each series"),
                  (quitadas_estacion, "por la estación", "because of the season"),
                  (quitadas_hueco, "por valores ausentes", "for missing values"),
                  (quitadas_corte, "porque su objetivo cae tras un corte del calendario",
                   "because their target falls after a break in the calendar")]
        dichas = [c for c in causas if c[0]]      # un cero no es una causa: no se enumera

        def _juntar(trozos: list[str], y: str) -> str:
            return trozos[0] if len(trozos) == 1 else ", ".join(trozos[:-1]) + f" {y} " + trozos[-1]
        es = (f"Con horizonte {horizonte}, {retardos} retardos"
              + (f" y estación {estacion}" if estacion is not None else "")
              + f" no queda ninguna fila completa de {len(filas)}: "
              + _juntar([f"{n} {t}" for n, t, _ in dichas], "y") + ".")
        en = (f"With horizon {horizonte}, {retardos} lags"
              + (f" and season {estacion}" if estacion is not None else "")
              + f" not a single complete row is left of {len(filas)}: "
              + _juntar([f"{n} {t}" for n, _, t in dichas], "and") + ".")
        raise PronosticoError(es, clave="sin_filas_completas", en=en, medida={
            "filas_de_entrada": len(filas), "borde": quitadas_borde, "estacion": quitadas_estacion,
            "ausentes": quitadas_hueco, "cortes_del_calendario": quitadas_corte})
    persist_ok = sum(1 for f in quedan if not _is_null(f[col_pers], tokens_de_ausencia))
    if persist_ok != len(quedan):
        raise PronosticoError("La persistencia (el valor actual del objetivo) está ausente en filas que quedan.")
    # Una serie no aporta ninguna fila si su largo no pasa de lo que piden los retardos (o la estación, si mira más atrás) más
    # el horizonte. Con la estación, la cuenta es la de la estación: la que de verdad la deja sin filas (I-4).
    series_cortas = sum(1 for pos in posiciones if len(pos) <= max(retardos, arranque_est) + horizonte) + len(series_de_otro_paso)
    series_cortas_por_la_estacion = sum(1 for pos in posiciones
                                        if retardos + horizonte < len(pos) <= max(retardos, arranque_est) + horizonte)
    declaracion = {
        "horizonte_pasos": horizonte, "retardos": retardos, "serie": serie, "objetivo": objetivo,
        "columnas_con_retardo": lag_cols, "estacion": estacion, "retraso_estacional": retraso_est,
        "estacional_igual_a_persistencia": retraso_est == 0,
        "filas_de_entrada": len(filas), "filas_resultantes": len(quedan),
        "filas_quitadas_por_el_borde_de_la_serie": quitadas_borde,
        "filas_quitadas_por_valores_ausentes": quitadas_hueco,
        "filas_quitadas_por_la_estacion": quitadas_estacion,
        "filas_quitadas_por_cortes_del_calendario": quitadas_corte,
        "cortes_del_calendario": [cortes_por_serie[n] for n in sorted(cortes_por_serie)],
        "series": len(posiciones), "series_sin_filas_completas": series_cortas,
        "series_sin_filas_completas_por_la_estacion": series_cortas_por_la_estacion,
        # 164-rep3 (aud3 M-3): las que no aportan filas porque su paso es más largo que el hueco del examen (ya contadas arriba).
        "series_sin_filas_por_su_paso": series_de_otro_paso,
        "series_con_calendario_irregular": irregulares, "paso_mediano_segundos_o_unidades": paso_mediano,
        # 164-rep2 (aud2 A-1): el hueco del reparto y por qué es ese. `eleccion`: `minimo` (h × paso), `calendario` (más ancho
        # porque el calendario lo pide: cuesta menos filas que quitarlas) o `pedido`. `filas_que_quitaria_el_minimo`: las que
        # se habrían quitado con h × paso (con `calendario`, las que el hueco más ancho conserva).
        "hueco": {"segundos_o_unidades": hueco_usado, "minimo": hueco_minimo, "eleccion": eleccion,
                  "filas_que_quitaria_el_minimo": quitaria_el_minimo,
                  # 164-rep3 (aud3 M-2): con `pedido`, un hueco más ancho que quitaría menos filas (`None` = no lo hay).
                  "otro_conservaria": otro_hueco_conservaria},
        "linaje": {c: linaje[c] for c in (*entradas, col_objetivo, col_pers) if c in linaje},
    }
    return Ventanas(
        filas=quedan, campo_id=campo_id, campo_tiempo=tiempo, campo_serie=serie, columna_objetivo=col_objetivo,
        columna_persistencia=col_pers, columna_estacional=col_est, columnas_de_entrada=tuple(entradas),
        horizonte=horizonte, estacion=estacion, retraso_estacional=retraso_est, linaje=linaje,
        tiempo_por_id={f[campo_id]: t_por_id[f[campo_id]] for f in quedan},
        tiempo_del_objetivo_por_id={f[campo_id]: t_obj[f[campo_id]] for f in quedan},
        paso_mediano=paso_mediano, declaracion=declaracion, hueco=hueco_usado)


def _elegir_hueco(minimo: float, necesita: Sequence[float], origenes: Sequence[float]) -> tuple[float, str]:
    """164-rep2 (aud2 A-1) — el hueco del reparto que MENOS filas cuesta. Con el hueco `G`, se quitan las filas con
    `necesita > G` (ver `construir_ventanas`); ensanchar el hueco por encima del mínimo (`h × paso`) cuesta, en cada examen
    (cada pliegue y la prueba), las filas cuyo origen cae en lo ensanchado: `densidad × (G − mínimo)`, con `densidad` = filas por
    segundo (o unidad) entre el primer y el último origen. Coste(G) = filas quitadas + densidad × (G − mínimo); los candidatos
    son el mínimo y cada `necesita` que lo supera; a igual coste, el más estrecho. Devuelve `(G, "minimo" | "calendario")`."""
    import bisect

    por_encima = sorted(x for x in necesita if x > minimo)
    if not por_encima:
        return minimo, "minimo"
    lapso = max(origenes) - min(origenes)
    densidad = len(origenes) / lapso if lapso > 0 else math.inf
    mejor, coste_mejor = minimo, float(len(por_encima))
    for g in dict.fromkeys(por_encima):
        coste = (len(por_encima) - bisect.bisect_right(por_encima, g)) + densidad * (g - minimo)
        if coste < coste_mejor:
            mejor, coste_mejor = g, coste
    return mejor, ("minimo" if mejor == minimo else "calendario")


# --------------------------------------------------------------------------------------------------
# 164-rep2 (aud2 A-2) — LAS ENTRADAS QUE DESCRIBEN EL FUTURO
# --------------------------------------------------------------------------------------------------

#: Una entrada «describe el futuro» si su valor en una fila explica el objetivo `k` pasos DESPUÉS (k = 1 o k = h) al menos
#: `_FUTURO_DIFERENCIA` mejor Y al menos `_FUTURO_RAZON` veces mejor que el objetivo de SU PROPIA fila. Medido el 16-11 (rep2,
#: `medir_futuro.py`, η² sobre rangos dentro de cada serie): `llueve_manana` de lluvia_MUNDIAL, 0,48 frente a 0,15 (+0,33, ×3,2);
#: con un 10 % de sus valores cambiados, +0,21 (×3,3); con un 20 %, +0,12 (×3,2). Las columnas legítimas de lluvia (16), Avocado
#: (10) y Uber (10, h = 1 y h = 6): la mayor diferencia, +0,010 (`spd` de Uber con h = 6).
_FUTURO_DIFERENCIA = 0.10
_FUTURO_RAZON = 2.0
#: Con menos pares no se mide (el η² de 10 grupos tiene un sesgo de ~9/n): se DECLARA que no se comprobó, no se da por buena.
_FUTURO_MIN_PARES = 50


def _rangos(valores: Any) -> Any:
    """Rangos medios (empates con el rango medio), con numpy."""
    import numpy as np

    v = np.asarray(valores, float)
    orden = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    vs = v[orden]
    i = 0
    n = len(v)
    while i < n:
        j = i
        while j + 1 < n and vs[j + 1] == vs[i]:
            j += 1
        r[orden[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def _grupos(valores: Sequence[str]) -> Any:
    """Los grupos de una columna para la razón de correlación: sus valores si tiene 20 o menos distintos (o no es numérica);
    si no, sus deciles."""
    import numpy as np

    try:
        x = np.asarray([float(a) for a in valores], float)
    except ValueError:
        etiquetas = {a: i for i, a in enumerate(sorted(set(valores)))}
        return np.asarray([etiquetas[a] for a in valores], int)
    if len(set(x.tolist())) <= 20:
        return x
    return np.minimum(9, (10 * _rangos(x) / len(x)).astype(int))


def _eta2(grupos: Any, y: Any) -> float | None:
    import numpy as np

    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return None
    entre = 0.0
    for g in np.unique(grupos):
        sel = y[grupos == g]
        entre += len(sel) * (float(sel.mean()) - float(y.mean())) ** 2
    return entre / total


def entradas_que_describen_el_futuro(
    filas: Sequence[Mapping[str, Any]], *, tiempo: str, objetivo: str, horizonte: int, columnas: Sequence[str],
    serie: str | None = None, tokens_de_ausencia: set[str] | None = None,
) -> list[dict[str, Any]]:
    """164-rep2 (aud2 A-2): una entrada que entra «con su valor en el momento de predecir» tiene linaje 0, pero puede DESCRIBIR el
    futuro (`llueve_manana` = «mañana llueve»): el linaje mira de qué fila sale el valor, no qué dice. Esto lo MIDE.

    El criterio: un valor anotado en una fila describe esa fila. Si explica el objetivo `k` pasos después (k = 1 y k = h) mucho
    mejor que el objetivo de su propia fila, lo que describe es el futuro. «Explica» = la razón de correlación η² del RANGO del
    objetivo (normalizado dentro de cada serie, para que el nivel de cada serie no cuente) agrupado por los valores de la entrada
    (sus deciles si es numérica con más de 20 valores). Se compara con su propia fila y no con la persistencia: en una serie con
    ciclo (Uber con h = 6) la persistencia explica casi nada y cualquier entrada la superaría.

    Devuelve, por columna: `columna`, `pares`, `ahora` (η² con el objetivo de su fila), `futuro` (el mayor de k = 1 y k = h),
    `k`, y `describe_el_futuro` (`True`/`False`; `None` = no se pudo medir: menos de `_FUTURO_MIN_PARES` pares o el objetivo
    no varía). No lanza: si el tiempo no se lee, devuelve `[]` y quien llama deja que `construir_ventanas` lo rechace."""
    import numpy as np
    from matrixai.training.particion_por_diseno_tiempo import convertir_tiempos

    columnas = [c for c in dict.fromkeys(columnas) if c not in (tiempo, objetivo, serie)]
    if not filas or not columnas:
        return []
    try:
        segundos = convertir_tiempos({str(i): f.get(tiempo) for i, f in enumerate(filas)}, tiempo).valores
    except Exception:  # noqa: BLE001 -- lo ilegible lo rechaza `construir_ventanas` con su motivo
        return []
    por_serie: dict[Any, list[int]] = {}
    for i, f in enumerate(filas):
        clave = None if serie is None else f.get(serie)
        if serie is not None and _is_null(clave, tokens_de_ausencia):
            continue
        por_serie.setdefault(clave, []).append(i)
    pasos = sorted({1, horizonte})
    # Por serie, en orden de tiempo: el rango normalizado del objetivo, y los pares (fila de origen, fila k pasos después).
    y_norm: dict[int, float] = {}
    origenes: list[tuple[int, dict[int, int]]] = []
    for idx in por_serie.values():
        idx = sorted(idx, key=lambda i: (segundos[str(i)], i))
        validas = []
        for i in idx:
            try:
                validas.append((i, float(str(filas[i].get(objetivo)).strip())))
            except (TypeError, ValueError):
                validas.append((i, None))
        con_y = [(i, y) for i, y in validas if y is not None and math.isfinite(y)]
        if len(con_y) >= 2:
            r = _rangos([y for _, y in con_y])
            for (i, _), ri in zip(con_y, r):
                y_norm[i] = float(ri) / (len(con_y) - 1)
        for p in range(len(idx) - horizonte):
            if idx[p] in y_norm and all(idx[p + k] in y_norm for k in pasos):
                origenes.append((idx[p], {k: idx[p + k] for k in pasos}))
    salida: list[dict[str, Any]] = []
    for c in columnas:
        pares = [(o, d) for o, d in origenes if not _is_null(filas[o].get(c), tokens_de_ausencia)]
        medida: dict[str, Any] = {"columna": c, "pares": len(pares), "ahora": None, "futuro": None, "k": None,
                                  "describe_el_futuro": None}
        if len(pares) >= _FUTURO_MIN_PARES:
            g = _grupos([str(filas[o].get(c)).strip() for o, _ in pares])
            ahora = _eta2(g, _rangos([y_norm[o] for o, _ in pares]))
            futuros = {k: _eta2(g, _rangos([y_norm[d[k]] for _, d in pares])) for k in pasos}
            if ahora is not None and all(v is not None for v in futuros.values()):
                k, fut = max(futuros.items(), key=lambda kv: (kv[1], -kv[0]))
                medida.update(ahora=ahora, futuro=fut, k=k, describe_el_futuro=bool(
                    fut - ahora >= _FUTURO_DIFERENCIA and fut >= _FUTURO_RAZON * ahora))
        salida.append(medida)
    return salida


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
    for f in fugas_detalladas(v, pliegues):
        if f["tipo"] == "solape":
            problemas.append(f"pliegue {f['pliegue']}: {f['filas']} fila(s) en entrena y valida")
        else:
            problemas.append(f"pliegue {f['pliegue']}: {f['filas']} fila(s) de entrenamiento con objetivo en "
                             f"{f['peor_objetivo']:g}, después del primer origen validado ({f['primer_origen_validado']:g})")
    return problemas


def fugas_detalladas(v: Ventanas, pliegues: Sequence[tuple[Sequence[str], Sequence[str]]]) -> list[dict[str, Any]]:
    """Lo mismo que `verificar_sin_fuga`, con los números sueltos (164-rep1): quien lo enseña escribe la fecha en el formato de
    la persona en vez de segundos desde 1970. `tipo` = `solape` (filas a los dos lados) u `objetivo_del_futuro`."""
    detalle: list[dict[str, Any]] = []
    for k, (entrena, valida) in enumerate(pliegues):
        if not entrena or not valida:
            continue
        comunes = set(entrena) & set(valida)
        if comunes:
            detalle.append({"pliegue": k, "tipo": "solape", "filas": len(comunes)})
        origen_min = min(v.tiempo_por_id[i] for i in valida)
        peor = max(v.tiempo_del_objetivo_por_id[i] for i in entrena)
        if peor > origen_min:
            n = sum(1 for i in entrena if v.tiempo_del_objetivo_por_id[i] > origen_min)
            detalle.append({"pliegue": k, "tipo": "objetivo_del_futuro", "filas": n, "peor_objetivo": peor,
                            "primer_origen_validado": origen_min})
    return detalle


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
    hueco = Horizonte(v.hueco, "seconds")      # 164-rep2: el de las ventanas (h × paso, o más si el calendario lo pide)
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
    #: 164-rep3 (aud3 I-3): el paso de la ventana si es UNO solo (todos los pasos iguales); `None` si son desiguales o no hay.
    paso_de_la_ventana: float | None = None


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
        pasos_de_la_ventana = {b - a for a, b in zip(ventana, ventana[1:])}
        irregular = len(pasos_de_la_ventana) > 1
        origen = filas[int(ultima["_fila"])].get(tiempo)
        predecibles.append(SeriePredecible(
            serie=nombre, entradas={c: str(ultima[c]) for c in entradas}, origen=str(origen).strip(),
            origen_segundos=ts[-1], filas_usadas=necesita, calendario_irregular=irregular,
            paso_de_la_ventana=next(iter(pasos_de_la_ventana)) if len(pasos_de_la_ventana) == 1 else None))
    return ContextoDePrediccion(predecibles, rechazadas, len(posiciones), es_fecha, formato)
