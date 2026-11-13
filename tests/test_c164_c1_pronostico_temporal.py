"""164-C1: `group_by` en `lag_window`/`shift_target`, `construir_ventanas`, referencias y examen hacia delante."""
import copy
import random

import pytest

from matrixai.training.dataset_pipeline import PipelineError, check_anti_leakage, run_pipeline
from matrixai.training.pronostico_temporal import (
    PronosticoError, construir_ventanas, evaluar_hacia_delante, verificar_sin_fuga,
)


# --- La implementación de ANTES (copiada tal cual, sin `group_by`): la clásica depende de ella. ---------------------
def _lag_antes(rows, columns, window):
    for idx, row in enumerate(rows):
        for column in columns:
            for i in range(1, window + 1):
                src = idx - i
                row[f"{column}_lag{i}"] = rows[src][column] if src >= 0 else ""


def _shift_antes(rows, column, horizon):
    n = len(rows)
    for idx, row in enumerate(rows):
        src = idx + horizon
        row[f"{column}_target_h{horizon}"] = rows[src][column] if src < n else ""


def _filas_azar(n, semilla):
    r = random.Random(semilla)
    return [{"t": f"2024-01-{(i % 28) + 1:02d}", "y": str(r.randint(0, 99)), "x": f"{r.random():.4f}",
             "s": r.choice("ABC")} for i in range(n)]


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_sin_group_by_es_bit_a_bit_la_de_antes(semilla):
    filas = _filas_azar(60, semilla)
    esperado = copy.deepcopy(filas)
    _lag_antes(esperado, ["y", "x"], 3)
    _shift_antes(esperado, "y", 2)
    res = run_pipeline(filas, [{"op": "lag_window", "columns": ["y", "x"], "window": 3},
                               {"op": "shift_target", "column": "y", "horizon": 2}])
    assert res.rows == esperado
    assert list(res.rows[0].keys()) == list(esperado[0].keys())
    assert res.column_offsets["y_lag3"] == -3 and res.column_offsets["y_target_h2"] == 2


def test_group_by_con_una_sola_serie_da_lo_mismo_que_sin_el():
    filas = [{**f, "s": "A"} for f in _filas_azar(40, 5)]
    sin = run_pipeline(filas, [{"op": "lag_window", "columns": ["y"], "window": 4},
                               {"op": "shift_target", "column": "y", "horizon": 3}])
    con = run_pipeline(filas, [{"op": "lag_window", "columns": ["y"], "window": 4, "group_by": "s"},
                               {"op": "shift_target", "column": "y", "horizon": 3, "group_by": "s"}])
    assert con.rows == sin.rows and con.column_offsets == sin.column_offsets


def _panel():
    # A y B INTERCALADAS en el fichero; dentro de cada serie el valor es 10, 11, 12… / 20, 21, 22…
    filas = []
    for k in range(4):
        filas.append({"t": f"2024-01-0{k + 1}", "s": "A", "y": str(10 + k)})
        filas.append({"t": f"2024-01-0{k + 1}", "s": "B", "y": str(20 + k)})
    return filas


def test_group_by_cuenta_dentro_de_la_serie_y_no_mezcla():
    res = run_pipeline(_panel(), [{"op": "lag_window", "columns": ["y"], "window": 2, "group_by": "s"},
                                  {"op": "shift_target", "column": "y", "horizon": 1, "group_by": "s"}])
    a = [r for r in res.rows if r["s"] == "A"]
    b = [r for r in res.rows if r["s"] == "B"]
    assert [r["y_lag1"] for r in a] == ["", "10", "11", "12"]
    assert [r["y_lag2"] for r in a] == ["", "", "10", "11"]
    assert [r["y_lag1"] for r in b] == ["", "20", "21", "22"]
    assert [r["y_target_h1"] for r in a] == ["11", "12", "13", ""]
    assert [r["y_target_h1"] for r in b] == ["21", "22", "23", ""]
    # y el defecto que esto cura: sin `group_by`, B hereda valores de A
    sin = run_pipeline(_panel(), [{"op": "lag_window", "columns": ["y"], "window": 1}])
    assert sin.rows[1]["s"] == "B" and sin.rows[1]["y_lag1"] == "10"     # B hereda el primer valor de A


def test_group_by_no_deja_nada_a_las_series_de_una_fila():
    filas = _panel() + [{"t": "2024-01-09", "s": "C", "y": "99"}]
    res = run_pipeline(filas, [{"op": "lag_window", "columns": ["y"], "window": 1, "group_by": "s"},
                               {"op": "shift_target", "column": "y", "horizon": 1, "group_by": "s"}])
    c = [r for r in res.rows if r["s"] == "C"][0]
    assert c["y_lag1"] == "" and c["y_target_h1"] == ""


def test_group_by_rechaza_columna_inexistente_y_serie_sin_nombre():
    with pytest.raises(PipelineError, match="group_by"):
        run_pipeline(_panel(), [{"op": "lag_window", "columns": ["y"], "window": 1, "group_by": "nada"}])
    filas = _panel()
    filas[3]["s"] = ""
    with pytest.raises(PipelineError, match="no se adivina"):
        run_pipeline(filas, [{"op": "shift_target", "column": "y", "horizon": 1, "group_by": "s"}])


# --- construir_ventanas -----------------------------------------------------------------------------------------
def _serie_larga(n=30, series=("A", "B"), formato="%Y-%m-%d"):
    from datetime import date, timedelta
    filas = []
    for k in range(n):
        for j, s in enumerate(series):
            filas.append({"fecha": (date(2024, 1, 1) + timedelta(days=k)).strftime(formato), "ciudad": s,
                          "y": str(100 * (j + 1) + k), "x": str(k * 2)})
    return filas


def test_ventanas_por_serie_valores_linaje_y_declaracion():
    filas = _serie_larga(30)
    v = construir_ventanas(filas, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=2, retardos=3,
                           columnas_con_retardo=["x"], estacion=7)
    d = v.declaracion
    assert d["series"] == 2 and d["filas_de_entrada"] == 60
    # por serie: 5 de borde por delante (la estacional mira 5 atrás, más que los 3 retardos) y 2 por detrás
    assert d["filas_resultantes"] == 46 and d["filas_quitadas_por_el_borde_de_la_serie"] == 14
    for f in v.filas:
        base = int(f["y"])
        assert int(f["y_lag1"]) == base - 1 and int(f["y_lag3"]) == base - 3     # misma serie, hacia atrás
        assert int(f["y_target_h2"]) == base + 2                                  # hacia delante
        assert f[v.columna_persistencia] == f["y"]
        # estacional con s=7, h=2: m*ceil(h/m)-h = 5 pasos atrás
        assert int(f[v.columna_estacional]) == base - 5
        assert int(f["x_lag2"]) == int(f["x"]) - 4
    assert v.retraso_estacional == 5
    assert all(v.linaje[c] <= 0 for c in v.columnas_de_entrada)
    assert v.linaje[v.columna_objetivo] == 2 and v.linaje[v.columna_estacional] == -5
    assert not check_anti_leakage(type("R", (), {"column_offsets": v.linaje})(), list(v.columnas_de_entrada))


def test_ventanas_no_dependen_del_orden_del_fichero_ni_del_formato_de_fecha():
    filas = _serie_larga(20)
    a = construir_ventanas(filas, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=2)
    barajadas = filas[:]
    random.Random(3).shuffle(barajadas)
    b = construir_ventanas(barajadas, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=2)
    clave = lambda v: sorted((f["ciudad"], f["fecha"], f["y_lag1"], f["y_lag2"], f["y_target_h1"]) for f in v.filas)
    assert clave(a) == clave(b)
    dmy = construir_ventanas(_serie_larga(20, formato="%d/%m/%Y"), tiempo="fecha", serie="ciudad", objetivo="y",
                             horizonte=1, retardos=2)
    assert sorted(f["y_target_h1"] for f in dmy.filas) == sorted(f["y_target_h1"] for f in a.filas)


def test_una_serie_de_una_sola_fila_no_rompe_y_se_declara():
    filas = _serie_larga(12) + [{"fecha": "2024-01-05", "ciudad": "SOLA", "y": "7", "x": "0"}]
    v = construir_ventanas(filas, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=2)
    assert v.declaracion["series"] == 3 and v.declaracion["series_sin_filas_completas"] == 1
    assert all(f["ciudad"] != "SOLA" for f in v.filas)


def test_sin_serie_es_una_serie_unica_y_series_distintas_no_se_mezclan():
    una = [f for f in _serie_larga(15, series=("A",))]
    v = construir_ventanas(una, tiempo="fecha", objetivo="y", horizonte=1, retardos=1)
    assert all(int(f["y_lag1"]) == int(f["y"]) - 1 for f in v.filas)
    assert v.declaracion["series"] == 1


def test_duplicados_calendario_irregular_y_errores_con_motivo():
    filas = _serie_larga(10)
    with pytest.raises(PronosticoError, match="comparten instante"):
        construir_ventanas(filas + [dict(filas[0])], tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=1)
    irregular = [f for f in _serie_larga(10) if f["fecha"] != "2024-01-05"]
    v = construir_ventanas(irregular, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=1)
    assert v.declaracion["series_con_calendario_irregular"] == 2
    with pytest.raises(PronosticoError, match="horizonte"):
        construir_ventanas(filas, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=0, retardos=1)
    with pytest.raises(PronosticoError, match="no existe"):
        construir_ventanas(filas, tiempo="fecha", serie="nada", objetivo="y", horizonte=1, retardos=1)
    with pytest.raises(PronosticoError):
        construir_ventanas(filas, tiempo="ciudad", serie="ciudad", objetivo="y", horizonte=1, retardos=1)


def test_valor_ausente_se_cuenta_como_ausente_no_como_borde():
    filas = _serie_larga(12)
    filas[10]["y"] = ""
    v = construir_ventanas(filas, tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=1)
    assert v.declaracion["filas_quitadas_por_valores_ausentes"] >= 1


# --- examen hacia delante ----------------------------------------------------------------------------------------
class _Persistencia:
    def __init__(self, col): self.col = col
    def fit(self, X, y): return self
    def predict(self, X): return X[:, self.col]


class _Exacto:
    """Sabe la respuesta (y(t+1) = y(t) + 1): el techo de la mejora."""
    def __init__(self, col): self.col = col
    def fit(self, X, y): return self
    def predict(self, X): return X[:, self.col] + 1


def _tendencia(n=120):
    return _serie_larga(n)


def test_el_modelo_que_copia_la_persistencia_mejora_cero_y_el_exacto_gana():
    v = construir_ventanas(_tendencia(), tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=2,
                           estacion=7)
    cols = list(v.columnas_de_entrada)
    i = cols.index("y")
    r0 = evaluar_hacia_delante(v, fabricar_modelo=lambda: _Persistencia(i), folds=3)
    assert r0["viable"] and r0["violaciones_de_fuga"] == []
    assert abs(r0["prueba"]["mejora_mae_sobre_persistencia"]) < 1e-12
    r1 = evaluar_hacia_delante(v, fabricar_modelo=lambda: _Exacto(i), folds=3)
    assert r1["prueba"]["modelo"]["mae"] < 1e-9
    assert r1["prueba"]["mejora_mae_sobre_persistencia"] == pytest.approx(1.0)
    assert r1["prueba"]["mase"] == pytest.approx(0.0, abs=1e-9)
    assert r1["prueba"]["persistencia"]["mae"] == pytest.approx(1.0)   # y sube 1 por paso
    assert r1["prueba"]["estacional"]["mae"] == pytest.approx(7.0)      # y(t+1) contra y(t-6): 6 de retraso + 1 de horizonte
    assert len(r1["pliegues"]) == 3


def test_el_hueco_cubre_el_horizonte_y_el_banco_de_fugas_no_ve_nada_en_el_reparto_real():
    for h in (1, 3, 7):
        v = construir_ventanas(_tendencia(150), tiempo="fecha", serie="ciudad", objetivo="y", horizonte=h, retardos=2)
        r = evaluar_hacia_delante(v, fabricar_modelo=lambda: _Persistencia(list(v.columnas_de_entrada).index("y")))
        assert r["viable"], r
        assert r["violaciones_de_fuga"] == [], (h, r["violaciones_de_fuga"])
        assert r["hueco_segundos_o_unidades"] == h * v.paso_mediano


def test_el_banco_de_fugas_tiene_dientes():
    v = construir_ventanas(_tendencia(60), tiempo="fecha", serie="ciudad", objetivo="y", horizonte=3, retardos=1)
    ordenadas = sorted(v.tiempo_por_id, key=lambda i: v.tiempo_por_id[i])
    mitad = len(ordenadas) // 2
    # sin hueco: entrena hasta justo antes de lo que se valida => el objetivo de las últimas ya está en el futuro
    malo = [(ordenadas[:mitad], ordenadas[mitad:])]
    assert verificar_sin_fuga(v, malo)
    assert verificar_sin_fuga(v, [(ordenadas[:mitad], ordenadas[:mitad + 2])])    # y una fila en los dos lados


def test_una_entrada_del_futuro_se_rechaza():
    v = construir_ventanas(_tendencia(80), tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=1)
    sucia = type(v)(**{**v.__dict__, "linaje": {**v.linaje, "y_lag1": 1}})
    with pytest.raises(PronosticoError, match="futuro"):
        evaluar_hacia_delante(sucia, fabricar_modelo=lambda: _Persistencia(0))


def test_la_persistencia_es_el_valor_actual_no_el_futuro():
    """Sabotaje conocido: una «persistencia» tomada de la columna desplazada daría mejora enorme y falsa."""
    v = construir_ventanas(_tendencia(80), tiempo="fecha", serie="ciudad", objetivo="y", horizonte=2, retardos=1)
    assert all(f[v.columna_persistencia] == f["y"] and f[v.columna_persistencia] != f[v.columna_objetivo]
               for f in v.filas)


def test_construir_ventanas_se_niega_si_el_linaje_dice_futuro(monkeypatch):
    """La guarda de `check_anti_leakage` en `construir_ventanas` no se alcanza por el camino normal: se le entrega un
    pipeline que declara un retardo con desplazamiento positivo y debe negarse (sabotaje 9: sin la guarda, pasaba)."""
    import dataclasses
    import matrixai.training.pronostico_temporal as pt
    real = pt.run_pipeline

    def sucio(*a, **k):
        r = real(*a, **k)
        return dataclasses.replace(r, column_offsets={**r.column_offsets, "y_lag1": 2})

    monkeypatch.setattr(pt, "run_pipeline", sucio)
    with pytest.raises(PronosticoError, match="Fuga"):
        construir_ventanas(_serie_larga(20), tiempo="fecha", serie="ciudad", objetivo="y", horizonte=1, retardos=2)
