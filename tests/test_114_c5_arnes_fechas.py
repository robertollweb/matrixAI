# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C5 — el arnés de la medición de fechas (`benchmarks/fechas_114c5/`)."""
from __future__ import annotations

import datetime
import json
import random
import sys
from pathlib import Path

import pytest

ARNES = Path(__file__).resolve().parents[1] / "benchmarks" / "fechas_114c5"
sys.path.insert(0, str(ARNES))

import generar_protocolo  # noqa: E402
import medir_114c5 as m  # noqa: E402


def test_el_protocolo_sellado_cuadra_con_su_huella_y_con_su_generador():
    sellado = json.loads((ARNES / "protocolo_114c5.json").read_text(encoding="utf-8"))
    assert sellado["huella"] == generar_protocolo.huella(sellado)
    # Y el fichero es EXACTAMENTE lo que genera el generador: nadie lo retocó a mano.
    assert {k: v for k, v in sellado.items() if k != "huella"} == \
        json.loads(json.dumps(generar_protocolo.PROTOCOLO, ensure_ascii=False))


def test_un_protocolo_tocado_se_niega(tmp_path, monkeypatch):
    sellado = json.loads((ARNES / "protocolo_114c5.json").read_text(encoding="utf-8"))
    sellado["criterio"] = "«mejora» en al menos 1 de los 5"
    ruta = tmp_path / "p.json"
    ruta.write_text(json.dumps(sellado), encoding="utf-8")
    monkeypatch.setattr(m, "RUTA_PROTOCOLO", ruta)
    with pytest.raises(SystemExit):
        m.cargar_protocolo()


def _arff(tmp_path, n=400):
    """Un ARFF con la fecha como STRING entre comillas: justo lo que el lector de
    Fase 0 excluye. El objetivo sube con los días y con el fin de semana."""
    azar = random.Random(0)
    base = datetime.datetime(2020, 1, 1, 6)
    lineas = ["@RELATION prueba", "@ATTRIBUTE ciudad STRING", "@ATTRIBUTE momento STRING",
              "@ATTRIBUTE x REAL", "@ATTRIBUTE y REAL", "@DATA"]
    for i in range(n):
        t = base + datetime.timedelta(hours=7 * i)
        y = i * 0.05 + (5.0 if t.weekday() >= 5 else 0.0) + azar.gauss(0, 0.3)
        lineas.append(f"'{azar.choice(['A', 'B'])}','{t:%Y-%m-%d %H:%M:%S}',"
                      f"{azar.random():.3f},{y:.3f}")
    ruta = tmp_path / "prueba.arff"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def test_el_lector_conserva_la_fecha_como_texto(tmp_path):
    filas, tipos = m.leer_arff(_arff(tmp_path))
    assert tipos == {"ciudad": "texto", "momento": "texto", "x": "numerico", "y": "numerico"}
    assert len(filas) == 400
    assert filas[0]["momento"] == "2020-01-01 06:00:00" and filas[0]["ciudad"] in ("A", "B")


def test_el_reparto_temporal_prueba_en_el_futuro(tmp_path):
    filas, _ = m.leer_arff(_arff(tmp_path))
    conjunto = {"fecha": "momento", "formato": "%Y-%m-%d %H:%M:%S"}
    train, test = m.repartir(filas, conjunto, "temporal", 0)
    assert max(f["momento"] for f in train) < min(f["momento"] for f in test)
    assert len(test) == 80


def test_el_reparto_temporal_no_parte_una_misma_fecha():
    """Auditoría del 2026-09-22: con el corte en el 80 % exacto, una misma fecha quedaba
    a los dos lados (86 filas en Avocado)."""
    filas = [{"row_id": str(i), "fecha": f"2020-01-{1 + i // 10:02d}"} for i in range(95)]
    conjunto = {"fecha": "fecha", "formato": "%Y-%m-%d"}
    train, test = m.repartir(filas, conjunto, "temporal", 0)
    assert not {f["fecha"] for f in train} & {f["fecha"] for f in test}
    assert len(train) == 80 and len(train) + len(test) == 95  # el 80 % cae en un cambio de día
    filas = [{"row_id": str(i), "fecha": f"2020-01-{1 + i // 7:02d}"} for i in range(100)]
    train, test = m.repartir(filas, conjunto, "temporal", 0)
    assert not {f["fecha"] for f in train} & {f["fecha"] for f in test}
    assert len(train) == 84  # 80 cae dentro del día 12 (filas 77–83): el corte se lleva a 84


def test_las_exclusiones_y_el_objetivo_derivado():
    filas = [{"row_id": str(i), "fecha": "2020-01-0%d" % (i % 9 + 1), "home_score": float(i % 3),
              "away_score": 1.0, "equipo": "X"} for i in range(9)]
    conjunto = {"nombre": "f", "objetivo": "gana", "fecha": "fecha",
                "objetivo_derivado": {"regla": "home_score > away_score", "positiva": "si",
                                      "negativa": "no"},
                "excluidas": {"home_score": "desenlace", "away_score": "desenlace"}}
    filas, objetivo, predictores = m.preparar_conjunto(conjunto, filas)
    assert predictores == ("fecha", "equipo")          # sin el desenlace, sin row_id
    assert [f["gana"] for f in filas[:3]] == ["no", "no", "si"]  # 0>1 no, 1>1 no (empate), 2>1 sí


def test_de_punta_a_punta_con_y_sin_la_fecha(tmp_path):
    """Con un objetivo que depende del tiempo y del fin de semana, convertir la
    fecha tiene que ayudar en el reparto temporal: como texto, cada fecha de
    test es una categoría que no se vio al entrenar."""
    protocolo = json.loads((ARNES / "protocolo_114c5.json").read_text(encoding="utf-8"))
    conjunto = {"data_id": 0, "nombre": "prueba", "tarea": "regression", "objetivo": "y",
                "fecha": "momento", "formato": "%Y-%m-%d %H:%M:%S", "excluidas": {}}
    filas, _ = m.leer_arff(_arff(tmp_path))
    filas, objetivo, predictores = m.preparar_conjunto(conjunto, filas)
    train, test = m.repartir(filas, conjunto, "temporal", 0)
    sin, _ = m.medir_variante("sin", train, test, conjunto, objetivo, predictores, 0, "lightgbm")
    con, info = m.medir_variante("con", train, test, conjunto, objetivo, predictores, 0, "lightgbm")
    assert [r["formato"] for r in info["recetas"]] == ["%Y-%m-%d %H:%M:%S"]
    resultado = m.comparar(conjunto, "temporal", protocolo, sin, con)
    assert resultado["metrica"] == "rmse" and resultado["con"] < resultado["sin"], resultado
    assert resultado["comparacion"]["veredicto"] == "mejora", resultado
