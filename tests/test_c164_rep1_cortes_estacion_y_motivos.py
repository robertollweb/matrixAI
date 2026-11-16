"""164 · reparación de la auditoría 1 (rep1), en el núcleo: los cortes del calendario de un panel (I-1), la causa real de las filas
que quita la estación (I-4) y de las que quita un valor ausente (M-4), y los motivos bilingües de los rechazos que la persona
provoca con datos normales (M-2)."""
from datetime import date, timedelta

import pytest

from matrixai.training.pronostico_temporal import (
    PronosticoError, construir_ventanas, evaluar_hacia_delante,
)


class _Media:
    """Un modelo mínimo: predice la media del objetivo con que se ajustó (el examen solo necesita `fit`/`predict`)."""

    def fit(self, X, y):
        self.m = float(sum(y) / len(y))
        return self

    def predict(self, X):
        return [self.m] * len(X)


def _panel(dias: int, fuera: dict[str, set[int]], series=("A", "B", "C"), inicio=date(2025, 1, 1)):
    filas = []
    for n, s in enumerate(series):
        for i in range(dias):
            if i in fuera.get(s, set()):
                continue
            filas.append({"serie": s, "fecha": (inicio + timedelta(days=i)).isoformat(), "y": str(10 * n + (i * 7) % 13)})
    return filas


def test_un_panel_con_un_corte_quita_solo_la_fila_que_lo_cruza_y_la_declara_por_serie_con_sus_fechas():
    # B pierde los días 40..44: su fila del día 39 tiene el objetivo (h = 1) en el 45, y A y C ponen orígenes en medio.
    filas = _panel(120, {"B": set(range(40, 45))})
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=1, retardos=3)
    d = v.declaracion
    assert d["filas_quitadas_por_cortes_del_calendario"] == 1
    assert d["cortes_del_calendario"] == [{"serie": "B", "filas": 1, "primer_origen": "2025-02-09",
                                           "primer_objetivo": "2025-02-15"}]
    assert d["filas_de_entrada"] == (d["filas_resultantes"] + d["filas_quitadas_por_el_borde_de_la_serie"]
                                     + d["filas_quitadas_por_valores_ausentes"] + d["filas_quitadas_por_la_estacion"]
                                     + d["filas_quitadas_por_cortes_del_calendario"])
    # y con lo que queda, NINGÚN reparto hacia delante deja una fuga: el examen entero, con 3 y con 5 pliegues
    for folds in (3, 5):
        r = evaluar_hacia_delante(v, fabricar_modelo=_Media, folds=folds)
        assert r["viable"] and r["violaciones_de_fuga"] == []


def test_un_panel_con_muchos_cortes_no_deja_ninguna_fuga_en_ningun_reparto():
    # La forma de la sonda de la auditoría (10 series × 200 días, 2 cortes de 3–10 días por serie): antes, 10 de 10 con fuga.
    import random
    rng = random.Random(7)
    fuera = {}
    for s in "ABCDEFGHIJ":
        f = set()
        for _ in range(2):
            a = rng.randrange(200)
            f |= set(range(a, a + rng.randint(3, 10)))
        fuera[s] = f
    filas = _panel(200, fuera, series=tuple("ABCDEFGHIJ"))
    for h in (1, 3):
        v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=h, retardos=3)
        assert v.declaracion["filas_quitadas_por_cortes_del_calendario"] > 0
        for folds in (3, 5):
            r = evaluar_hacia_delante(v, fabricar_modelo=_Media, folds=folds)
            assert r["viable"] and r["violaciones_de_fuga"] == [], (h, folds, r["violaciones_de_fuga"])


def test_un_calendario_laborable_comun_a_todas_las_series_no_pierde_ninguna_fila():
    # Lunes a viernes en las tres series: el objetivo del viernes es el lunes (3 días, más que h × paso = 1 día), pero ningún
    # origen cae en el fin de semana, así que ningún reparto puede validar entre el viernes y el lunes: no se quita nada.
    finde = {i for i in range(140) if (date(2025, 1, 6) + timedelta(days=i)).weekday() >= 5}
    filas = _panel(140, {s: finde for s in "ABC"}, inicio=date(2025, 1, 6))
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=1, retardos=3)
    assert v.declaracion["filas_quitadas_por_cortes_del_calendario"] == 0
    assert v.declaracion["cortes_del_calendario"] == []
    r = evaluar_hacia_delante(v, fabricar_modelo=_Media, folds=3)
    assert r["viable"] and r["violaciones_de_fuga"] == []


def test_una_serie_sola_con_un_corte_no_pierde_filas_su_siguiente_instante_es_su_objetivo():
    filas = _panel(100, {"A": set(range(30, 38))}, series=("A",))
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=1, retardos=3)
    assert v.declaracion["filas_quitadas_por_cortes_del_calendario"] == 0


def test_pedir_la_estacion_declara_las_filas_que_quita_por_la_estacion_y_no_por_el_borde():
    # 30 filas, 3 retardos, h = 1, estación 7 (mira 6 atrás): 3 + 1 de borde, 3 por la estación.
    filas = [{"t": str(i), "y": str(float(i * i % 7))} for i in range(30)]
    v = construir_ventanas(filas, tiempo="t", objetivo="y", horizonte=1, retardos=3, estacion=7)
    d = v.declaracion
    assert (d["filas_quitadas_por_el_borde_de_la_serie"], d["filas_quitadas_por_la_estacion"], d["filas_resultantes"]) == (4, 3, 23)
    sin = construir_ventanas(filas, tiempo="t", objetivo="y", horizonte=1, retardos=3).declaracion
    assert sin["filas_quitadas_por_la_estacion"] == 0 and sin["filas_quitadas_por_el_borde_de_la_serie"] == 4


def test_una_serie_que_solo_la_estacion_deja_sin_filas_cuenta_como_serie_sin_filas_completas():
    # Auditoría 1, I-4: una serie de 30 filas con s = 52 no aporta NINGUNA fila y antes se declaraba `series_sin_filas_completas: 0`.
    filas = ([{"s": "corta", "t": str(i), "y": str(i % 5)} for i in range(30)]
             + [{"s": "larga", "t": str(i), "y": str(i % 5)} for i in range(120)])
    v = construir_ventanas(filas, tiempo="t", serie="s", objetivo="y", horizonte=1, retardos=4, estacion=52)
    d = v.declaracion
    assert d["series_sin_filas_completas"] == 1 and d["series_sin_filas_completas_por_la_estacion"] == 1
    assert not any(f["s"] == "corta" for f in v.filas)
    assert d["filas_quitadas_por_la_estacion"] == (51 - 4) + (30 - 4 - 1)


def test_un_valor_ausente_se_cuenta_como_ausente_tambien_en_las_filas_que_lo_miran():
    # Auditoría 1, M-4 (la sonda `sonda_ausentes.py`): un ausente en la fila 10 quita la suya, la 9 (su objetivo), las 11–13 (sus
    # retardos) y la 16 (su referencia estacional): 6 por valores ausentes, y el borde es solo el borde (3 + 1).
    filas = [{"t": str(i), "y": ("" if i == 10 else str(float(i * i % 7)))} for i in range(30)]
    d = construir_ventanas(filas, tiempo="t", objetivo="y", horizonte=1, retardos=3, estacion=7).declaracion
    assert d["filas_quitadas_por_valores_ausentes"] == 6
    assert d["filas_quitadas_por_el_borde_de_la_serie"] == 4 and d["filas_quitadas_por_la_estacion"] == 3
    assert d["filas_resultantes"] == 30 - 6 - 4 - 3


def test_una_hora_repetida_se_rechaza_con_la_serie_y_el_instante_legibles_en_los_dos_idiomas():
    # Auditoría 1, M-2: el cambio de hora de octubre repite las 02:00 en datos horarios.
    horas = ["2025-10-26 00:00", "2025-10-26 01:00", "2025-10-26 02:00", "2025-10-26 02:00", "2025-10-26 03:00",
             "2025-10-26 04:00", "2025-10-26 05:00", "2025-10-26 06:00"]
    filas = [{"zona": "Norte", "hora": h, "kwh": str(i)} for i, h in enumerate(horas)]
    with pytest.raises(PronosticoError) as exc:
        construir_ventanas(filas, tiempo="hora", serie="zona", objetivo="kwh", horizonte=1, retardos=2)
    e = exc.value
    assert e.clave == "instantes_repetidos" and e.medida["serie"] == "Norte"
    assert e.motivo is not None
    for texto in (e.motivo["es"], e.motivo["en"]):
        assert "Norte" in texto and "2025-10-26 02:00" in texto
    assert "comparten instante" in e.motivo["es"] and "share an instant" in e.motivo["en"]


def test_sin_ninguna_fila_completa_el_rechazo_es_bilingue_y_dice_cuantas_quito_cada_causa():
    filas = [{"t": str(i), "y": str(i)} for i in range(10)]
    with pytest.raises(PronosticoError) as exc:
        construir_ventanas(filas, tiempo="t", objetivo="y", horizonte=1, retardos=3, estacion=12)
    e = exc.value
    assert e.clave == "sin_filas_completas" and e.motivo is not None
    assert e.medida == {"filas_de_entrada": 10, "borde": 4, "estacion": 6, "ausentes": 0, "cortes_del_calendario": 0}
    assert "estación" in e.motivo["es"] and "season" in e.motivo["en"]


def test_un_error_sin_traduccion_no_finge_un_motivo_bilingue():
    with pytest.raises(PronosticoError) as exc:
        construir_ventanas([{"t": "1", "y": "2"}], tiempo="t", objetivo="no_existe", horizonte=1, retardos=1)
    assert exc.value.clave is None and exc.value.motivo is None
