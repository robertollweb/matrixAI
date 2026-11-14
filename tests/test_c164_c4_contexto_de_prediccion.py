"""164-C4 (núcleo): el contexto que hace falta para predecir con datos nuevos. Por serie, sin rellenar, sin mezclar series."""
from datetime import datetime, timedelta

import pytest

from matrixai.training.pronostico_temporal import (
    PronosticoError, construir_contexto_de_prediccion, contexto_necesario,
)


def _serie(nombre, n, inicio=datetime(2025, 1, 1), base=10.0, paso=1):
    return [{"ciudad": nombre, "fecha": (inicio + timedelta(days=paso * i)).strftime("%Y-%m-%d"),
             "temp": f"{base + i:.1f}", "hum": f"{50 + i * 0.5:.1f}"} for i in range(n)]


def _armar(filas, **kw):
    kw = {"tiempo": "fecha", "objetivo": "temp", "retardos": 3, "serie": "ciudad", "columnas_con_retardo": ["hum"], **kw}
    return construir_contexto_de_prediccion(filas, **kw)


def test_contexto_necesario_cuenta_las_filas_del_historial_mas_la_de_partida():
    c = contexto_necesario(retardos=3, columnas_con_retardo=["hum"], columnas_sin_retardo=["festivo"], tiempo="fecha",
                           serie="ciudad", objetivo="temp", horizonte=2)
    assert c["filas_por_serie"] == 4 and c["filas_de_historial"] == 3 and c["una_fila_suelta"] is False
    assert c["columnas"] == ["fecha", "ciudad", "temp", "hum", "festivo"] and c["horizonte_pasos"] == 2


def test_cada_serie_se_predice_con_sus_propios_valores_y_no_con_los_de_la_vecina():
    filas = _serie("A", 10, base=100.0) + _serie("B", 6, base=500.0, inicio=datetime(2025, 1, 3))
    filas.reverse()                                     # desordenado: manda el tiempo, no la posición
    c = _armar(filas)
    por = {p.serie: p for p in c.predecibles}
    assert set(por) == {"A", "B"} and not c.rechazadas and c.series == 2
    a, b = por["A"].entradas, por["B"].entradas
    # el último instante de A es el 10 (valor 109) y sus retardos son 108, 107, 106: nunca un 5xx de B
    assert (a["temp"], a["temp_lag1"], a["temp_lag2"], a["temp_lag3"]) == ("109.0", "108.0", "107.0", "106.0")
    assert (b["temp"], b["temp_lag1"], b["temp_lag2"], b["temp_lag3"]) == ("505.0", "504.0", "503.0", "502.0")
    assert por["A"].origen == "2025-01-10" and por["B"].origen == "2025-01-08"
    assert a["hum_lag1"] == "54.0" and set(a) == {"temp_lag1", "temp_lag2", "temp_lag3", "hum_lag1", "hum_lag2", "hum_lag3", "temp", "hum"}


def test_una_serie_con_poco_historial_se_rechaza_con_sus_numeros_y_las_demas_se_predicen():
    # `C` empieza antes que `A` y acaba después: con lag_window POSICIONAL sus dos filas heredarían retardos de otras series
    filas = _serie("A", 12) + _serie("C", 3, inicio=datetime(2025, 1, 5)) + _serie("D", 1)
    c = _armar(filas)
    assert [p.serie for p in c.predecibles] == ["A"]
    r = {x.serie: x for x in c.rechazadas}
    assert r["C"].clave == "historial_corto" and (r["C"].tiene, r["C"].necesita) == (3, 4)
    assert r["D"].clave == "historial_corto" and (r["D"].tiene, r["D"].necesita) == (1, 4)


def test_justo_las_filas_que_hacen_falta_se_predice_y_una_menos_no():
    assert [p.serie for p in _armar(_serie("A", 4)).predecibles] == ["A"]
    c = _armar(_serie("A", 3))
    assert not c.predecibles and c.rechazadas[0].clave == "historial_corto"


def test_un_valor_ausente_en_lo_que_lee_el_modelo_rechaza_la_serie_y_no_se_rellena():
    filas = _serie("A", 8)
    filas[-2]["temp"] = ""                              # un hueco dentro de la ventana
    filas += _serie("B", 8)
    c = _armar(filas)
    assert [p.serie for p in c.predecibles] == ["B"]
    r = c.rechazadas[0]
    assert r.serie == "A" and r.clave == "valores_ausentes" and "temp_lag1" in r.columnas


def test_una_columna_sin_retardo_ausente_en_la_ultima_fila_rechaza_la_serie():
    filas = [{**f, "festivo": "0"} for f in _serie("A", 6)]
    filas[-1]["festivo"] = ""
    c = _armar(filas, columnas_sin_retardo=["festivo"])
    assert not c.predecibles and c.rechazadas[0].columnas == ("festivo",)


def test_filas_sin_serie_se_cuentan_y_no_se_adivinan():
    filas = _serie("A", 6) + [{**_serie("X", 1)[0], "ciudad": ""}]
    c = _armar(filas)
    assert [p.serie for p in c.predecibles] == ["A"]
    assert [(r.clave, r.filas) for r in c.rechazadas] == [("sin_serie", 1)]


def test_dos_filas_en_el_mismo_instante_de_una_serie_la_rechazan():
    filas = _serie("A", 6) + [dict(_serie("A", 6)[3])]
    c = _armar(filas)
    assert not c.predecibles and c.rechazadas[0].clave == "instantes_repetidos"


def test_sin_columna_de_serie_hay_una_sola_serie_y_las_columnas_que_faltan_se_dicen():
    c = _armar([{k: v for k, v in f.items() if k != "ciudad"} for f in _serie("A", 6)], serie=None)
    assert len(c.predecibles) == 1 and c.predecibles[0].serie is None
    with pytest.raises(PronosticoError, match="no existe"):
        _armar(_serie("A", 6), columnas_con_retardo=["lluvia"])


def test_un_lote_corto_con_dias_menores_de_13_necesita_el_formato_con_que_se_entreno():
    filas = [{**f, "fecha": (datetime(2025, 3, 6) + timedelta(days=i)).strftime("%d/%m/%Y")} for i, f in enumerate(_serie("A", 4))]
    with pytest.raises(PronosticoError):                # solo, 06/03…09/03 es ambiguo: no se adivina
        _armar(filas)
    c = _armar(filas, formato_de_fecha="%d/%m/%Y")
    assert c.predecibles[0].origen == "09/03/2025" and c.formato_de_fecha == "%d/%m/%Y"


def test_el_calendario_irregular_en_la_ventana_se_marca():
    filas = _serie("A", 8)
    del filas[5]
    assert _armar(filas).predecibles[0].calendario_irregular is True
    assert _armar(_serie("A", 8)).predecibles[0].calendario_irregular is False
