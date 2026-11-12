# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""160-C1 — los motores que VUELCAN lo que predicen sobre la prueba del pliegue.

El arnés de la Fase 0 (`harness.ejecutar_intento`) construye la `Muestra` de test (`_muestra_de_test`), mide y la TIRA: el `Intento`
solo lleva métricas agregadas. Para mezclar dos motores por fila hacen falta sus probabilidades por fila, y no están en ningún
artefacto. Esto no toca el paquete: son dos subclases de los motores de la Fase 0 que, cuando el arnés les pide predecir sobre
la partición de PRUEBA (se reconoce por sus `row_ids`), guardan en un `.npz` lo que acaban de devolver, con el índice de fila, la
verdad y lo que tardaron en predecir. Devuelven EXACTAMENTE lo que devolvería el motor sin envolver (no cambian ni un valor).

Viven en un módulo propio y no en `medir_c160.py` porque el hijo `spawn` tiene que poder importarlas por nombre.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM
from matrixai_engines.motores.densa_tabm import MotorDensaTabM


def ruta_del_volcado(directorio: Path, dataset: str, motor: str, repeticion: int, pliegue: int) -> Path:
    """Un único sitio que nombra el fichero (quien escribe y quien lee)."""
    seguro = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{dataset}__{motor}")
    return Path(directorio) / f"{seguro}__r{repeticion}_f{pliegue}.npz"


def guardar_volcado(ruta: Path, *, row_ids, y_true, predicciones, probabilidades, clases, positiva,
                    t_predict_s: float | None, t_proba_s: float | None) -> None:
    """Escribe el `.npz` de forma atómica (temporal + `os.replace`): un lector nunca ve un fichero a medias."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    es_clasificacion = clases is not None
    datos = {
        "row_ids": np.asarray([str(r) for r in row_ids]),
        "y_true": np.asarray([str(y) for y in y_true]) if es_clasificacion else np.asarray(y_true, dtype=np.float64),
        "predicciones": (np.asarray([str(p) for p in predicciones]) if es_clasificacion
                         else np.asarray(predicciones, dtype=np.float64)),
        "probabilidades": (np.asarray(probabilidades, dtype=np.float64) if probabilidades is not None
                           else np.zeros((0, 0), dtype=np.float64)),
        "clases": np.asarray([str(c) for c in clases]) if es_clasificacion else np.asarray([], dtype=str),
        "positiva": np.asarray(str(positiva) if positiva is not None else ""),
        "es_clasificacion": np.asarray(bool(es_clasificacion)),
        "t_predict_s": np.asarray(np.nan if t_predict_s is None else t_predict_s),
        "t_proba_s": np.asarray(np.nan if t_proba_s is None else t_proba_s),
    }
    temporal = ruta.with_name(ruta.name + f".tmp{os.getpid()}")
    with open(temporal, "wb") as f:
        np.savez(f, **datos)
    os.replace(temporal, ruta)


def leer_volcado(ruta: Path) -> dict:
    """Lo contrario de `guardar_volcado`, a tipos de Python/numpy sencillos (`allow_pickle` apagado: no se ejecuta nada)."""
    with np.load(Path(ruta), allow_pickle=False) as z:
        es_clasificacion = bool(z["es_clasificacion"])
        t_predict, t_proba = float(z["t_predict_s"]), float(z["t_proba_s"])
        return {
            "row_ids": [str(r) for r in z["row_ids"]],
            "y_true": [str(y) for y in z["y_true"]] if es_clasificacion else [float(y) for y in z["y_true"]],
            "predicciones": ([str(p) for p in z["predicciones"]] if es_clasificacion
                             else [float(p) for p in z["predicciones"]]),
            "probabilidades": z["probabilidades"].copy() if es_clasificacion else None,
            "clases": tuple(str(c) for c in z["clases"]) if es_clasificacion else None,
            "positiva": (str(z["positiva"]) or None) if es_clasificacion else None,
            "es_clasificacion": es_clasificacion,
            "t_predict_s": None if np.isnan(t_predict) else t_predict,
            "t_proba_s": None if np.isnan(t_proba) else t_proba,
        }


class _Volcador:
    """Mezcla para cualquier `Motor`: atributos de INSTANCIA (viajan pickleados al hijo `spawn`)."""

    volcado_ruta: str | None = None
    volcado_row_ids_de_test: tuple = ()

    def _es_la_prueba(self, datos) -> bool:
        return (self.volcado_ruta is not None
                and tuple(str(r) for r in datos.row_ids) == tuple(self.volcado_row_ids_de_test))

    def predict(self, fitted, datos):
        t0 = time.perf_counter()
        salida = super().predict(fitted, datos)
        dt = time.perf_counter() - t0
        if self._es_la_prueba(datos):
            self._v_predicciones, self._v_t_predict = salida, dt
            if fitted.classes is None:  # regresión: aquí no habrá `predict_proba`
                guardar_volcado(Path(self.volcado_ruta), row_ids=datos.row_ids, y_true=datos.target,
                                predicciones=salida, probabilidades=None, clases=None, positiva=None,
                                t_predict_s=dt, t_proba_s=None)
        return salida

    def predict_proba(self, fitted, datos):
        t0 = time.perf_counter()
        salida = super().predict_proba(fitted, datos)
        dt = time.perf_counter() - t0
        if self._es_la_prueba(datos):
            if not hasattr(self, "_v_predicciones"):
                raise RuntimeError("160-C1: el arnés pidió las probabilidades antes que las predicciones; "
                                   "el volcado no sabe qué guardar")
            guardar_volcado(Path(self.volcado_ruta), row_ids=datos.row_ids, y_true=datos.target,
                            predicciones=self._v_predicciones, probabilidades=salida, clases=fitted.classes,
                            positiva=fitted.positive_label, t_predict_s=self._v_t_predict, t_proba_s=dt)
        return salida


class MotorLightGBMQueVuelca(_Volcador, MotorArbolLightGBM):
    """El `lightgbm` de la Fase 0 (`motor_para("lightgbm")`, política por omisión), que vuelca su prueba."""


class MotorTabMQueVuelca(_Volcador, MotorDensaTabM):
    """TabM `ref` (la configuración de referencia del producto), que vuelca su prueba."""


def con_volcado(motor, ruta: Path, row_ids_de_test) -> None:
    """Apunta un motor volcador a su fichero y a las filas de la prueba que reconocerá."""
    motor.volcado_ruta = str(ruta)
    motor.volcado_row_ids_de_test = tuple(str(r) for r in row_ids_de_test)
    if hasattr(motor, "_v_predicciones"):
        del motor._v_predicciones
