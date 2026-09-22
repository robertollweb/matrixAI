# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C0 — el arnés de memoria y tiempos de TabICL (`benchmarks/tabicl_116c0/`).
Lo que se prueba sin Docker: la rejilla registrada, la regla de parada, la lectura
de la memoria del proceso y los datos sintéticos."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "tabicl_116c0"))
import medir_punto  # noqa: E402
import orquestar  # noqa: E402


def test_la_rejilla_es_la_registrada():
    assert orquestar.FILAS == (500, 1000, 2000, 4000, 8000)
    assert orquestar.REJILLA == {"optdigits": (8, 32, 64), "adult": (8, 14), "sintetico": (8, 32, 64)}
    assert len(orquestar.puntos()) == 5 * (3 + 2 + 3)


def test_tras_un_no_cabe_no_se_lanzan_las_filas_mayores_de_esa_fuente_y_columnas():
    lanzados = []

    def falso(fuente, columnas, filas):
        lanzados.append((fuente, columnas, filas))
        return {"estado": "no_cabe"} if (fuente, columnas, filas) == ("optdigits", 64, 2000) \
            else {"estado": "medido"}

    guardados = []
    registros = orquestar.recorrer(falso, guardar=lambda r: guardados.append(len(r)))
    assert ("optdigits", 64, 4000) not in lanzados and ("optdigits", 64, 8000) not in lanzados
    no_lanzados = [(r["fuente"], r["columnas"], r["filas"]) for r in registros
                   if r["estado"] == "no_lanzado_por_el_anterior"]
    assert no_lanzados == [("optdigits", 64, 4000)]
    # 8.000 + 512 filas no caben en las 5.620 de optdigits: no se lanza y se dice.
    sin_filas = {(r["fuente"], r["columnas"], r["filas"]) for r in registros
                 if r["estado"] == "no_hay_tantas_filas"}
    assert sin_filas == {("optdigits", c, 8000) for c in (8, 32, 64)}
    assert not any(p[0] == "optdigits" and p[2] == 8000 for p in lanzados)
    # Y lo medido se escribe tras CADA punto, no solo al final.
    assert guardados == list(range(1, len(orquestar.puntos()) + 1))
    # Las demás columnas y fuentes siguen: la regla no se lleva por delante lo que no toca.
    assert ("sintetico", 64, 8000) in lanzados and ("optdigits", 32, 4000) in lanzados
    assert len(registros) == len(orquestar.puntos())


def test_la_memoria_del_proceso_se_lee_de_su_status():
    assert medir_punto.vmhwm_mb() > 1.0


def test_los_datos_sinteticos_tienen_la_forma_pedida_y_no_cambian():
    X, y, Xt, yt = medir_punto.datos("sintetico", 300, 16, 50)
    assert X.shape == (300, 16) and Xt.shape == (50, 16) and set(y) <= {0, 1}
    X2, *_ = medir_punto.datos("sintetico", 300, 16, 50)
    assert (X == X2).all()
