# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C0-bis — el veredicto V(ε, q) y la regla por tarea del guion, con retrocesos escritos a mano.

`benchmarks/c149/medir_c0bis.py` implementa `REGLA_DE_ELECCION_BIS.md`. Aquí se prueba solo esa lógica —sin entrenar
nada— contra números que se cuentan en el comentario de cada prueba. Lo que NO prueba: que el instrumento mida bien
(eso lo dice el sabotaje de la etiqueta invertida dentro de la propia medición).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
_ESPEC = importlib.util.spec_from_file_location("medir_c0bis_149", RAIZ / "benchmarks" / "c149" / "medir_c0bis.py")
bis = importlib.util.module_from_spec(_ESPEC)
sys.modules["medir_c0bis_149"] = bis
_ESPEC.loader.exec_module(bis)

CON, SIN, NSC = "con_contraejemplos", "sin_contraejemplos", "no_se_pudo_comprobar"
EPS = 0.25          # potencia de dos: las comparaciones con ε son exactas, sin coma flotante


def filas(n_violadoras, n=20, retroceso=0.5):
    """`n` retrocesos por fila: `n_violadoras` de ellos por encima de ε = 0,25 y el resto en 0."""
    return [retroceso] * n_violadoras + [0.0] * (n - n_violadoras)


def test_el_efecto_se_mira_primero_aunque_todas_las_filas_retrocedan():
    # 20 de 20 filas retroceden 0,5 > ε (fracción 100 % >= cualquier q), pero el efecto total (0,125) es <= ε = 0,25:
    # la prueba no distingue, y NO es una violación.
    for q in bis.Q_PCT:
        assert bis.veredicto_bis(filas(20), 0.125, EPS, q) == NSC


def test_un_efecto_igual_a_epsilon_tampoco_distingue():
    # «efecto total <= ε»: con el efecto IGUAL a ε sigue sin poder comprobar (y con una pizca más, sí).
    assert bis.veredicto_bis(filas(20), 0.25, EPS, 5) == NSC
    assert bis.veredicto_bis(filas(20), 0.2500001, EPS, 5) == CON


def test_sin_efecto_calculable_no_se_pudo_comprobar():
    assert bis.veredicto_bis(filas(20), None, EPS, 5) == NSC


def test_fraccion_justo_en_q_es_con_y_justo_por_debajo_es_sin():
    # 20 filas: 1 violadora = 5 %, 2 = 10 %, 4 = 20 %. El efecto (1,0) supera ε.
    assert bis.veredicto_bis(filas(1), 1.0, EPS, 5) == CON      # 5 % >= 5 %   (justo en q)
    assert bis.veredicto_bis(filas(1), 1.0, EPS, 10) == SIN     # 5 % <  10 %  (justo por debajo)
    assert bis.veredicto_bis(filas(2), 1.0, EPS, 10) == CON     # 10 % >= 10 %
    assert bis.veredicto_bis(filas(2), 1.0, EPS, 20) == SIN     # 10 % <  20 %
    assert bis.veredicto_bis(filas(4), 1.0, EPS, 20) == CON     # 20 % >= 20 %
    assert bis.veredicto_bis(filas(3), 1.0, EPS, 20) == SIN     # 15 % <  20 %
    assert bis.veredicto_bis(filas(0), 1.0, EPS, 5) == SIN


def test_un_retroceso_igual_a_epsilon_no_viola():
    # `>` y no `>=` (como C1): 20 filas que retroceden EXACTAMENTE ε no son violadoras.
    assert bis.veredicto_bis([EPS] * 20, 1.0, EPS, 5) == SIN


def test_las_filas_sin_prediccion_no_cuentan_ni_en_el_denominador():
    # 20 filas, 10 sin predicción (None): 1 violadora de 10 comprobadas = 10 % (con 20 como denominador serían 5 %).
    r = [0.5] + [0.0] * 9 + [None] * 10
    assert bis.veredicto_bis(r, 1.0, EPS, 10) == CON
    assert bis.veredicto_bis(r, 1.0, EPS, 20) == SIN
    assert bis.veredicto_bis([None] * 5, 1.0, EPS, 5) == NSC


def test_con_los_retrocesos_y_el_efecto_de_C1_y_no_con_los_del_veredicto_de_C1():
    # Curvas a mano sobre una malla de 3 puntos, escala de probabilidad (1,0), ε = 5 %. Cada fila es [0, E + r, E]:
    # efecto E y retroceso r. 20 filas con E = 0,04 (<= ε) y r = 0,10 (> ε).
    from matrixai.estudio.comportamiento import escala_de_la_salida, monotonia_desde_curvas
    escala = escala_de_la_salida("binary_classification")
    curvas = [[0.0, 0.14, 0.04] for _ in range(20)]
    r0 = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0], curvas, list(range(20)), epsilon_relativo=0.0, escala=escala,
                                medido_sobre="muestra_de_desarrollo", semilla=3)
    r = r0.con_epsilon(0.05)
    # El veredicto de C1 (una sola fila basta) dice «con contraejemplos» ... y V(ε, q) mira el efecto primero.
    assert r.veredicto == CON
    assert abs(r.efecto_total - 0.04) < 1e-12 and r.filas_con_violacion == 20
    assert bis.veredicto_de_c1(r, 5) == NSC
    # Con un efecto claro (E = 0,5) y 2 filas de 20 que retroceden 0,10: C1 dice «con», V(5 %, q) depende de q.
    curvas = [[0.0, 0.6, 0.5] if i < 2 else [0.0, 0.25, 0.5] for i in range(20)]
    # (C1-bis: el veredicto del núcleo ya no es «una fila basta»; q = 0,01 con 20 filas es exactamente esa regla del C1, que
    # es lo que `veredicto_de_c1` del guion reproduce. La intención de la prueba se conserva: C1 «con», V(ε, q) según q.)
    r = monotonia_desde_curvas("x", "sube", [0.0, 1.0, 2.0], curvas, list(range(20)), epsilon_relativo=0.05, q_relativo=0.01,
                               escala=escala, medido_sobre="muestra_de_desarrollo", semilla=3)
    assert r.veredicto == CON and r.filas_con_violacion == 2
    assert bis.veredicto_de_c1(r, 10) == CON and bis.veredicto_de_c1(r, 20) == SIN


# ---------------------------------------------------------------------------
# La regla por tarea
# ---------------------------------------------------------------------------

GRUPOS = ["a|x", "b|x"]


def acumulado(f, planas_nsc=12, grupos=GRUPOS):
    """`f(e, q) -> (monótonos marcados de 10, no monótonos marcados de 10)`; planas: `planas_nsc` de 12 por columna."""
    acum = {}
    for g in grupos:
        for e in bis.EPS_PCT:
            for q in bis.Q_PCT:
                celda = bis._celda_vacia()
                mon, nomon = f(e, q)
                celda["monotono"].update(n=10, con=mon, sin=10 - mon)
                celda["no_monotono"].update(n=10, con=nomon, sin=10 - nomon)
                for c in bis.PLANAS:
                    celda["plana"][c].update(n=6, nsc=planas_nsc // 2, sin=6 - planas_nsc // 2)
                acum[(g, e, q)] = celda
    return acum


def test_elige_el_menor_epsilon_y_con_el_el_menor_q():
    # Cumple (FA 10 %, POT 80 %) con ε >= 5 % y q <= 10 %; con ε = 2 % nunca.
    acum = acumulado(lambda e, q: (1, 8) if (e >= 5 and q <= 10) else (9, 10))
    r = bis.elegir(acum, GRUPOS)
    assert (r["epsilon_pct"], r["q_pct"]) == (5, 5) and r["q_que_cumplen_con_ese_epsilon"] == [5, 10]


def test_hace_falta_que_cumplan_los_dos_grupos_a_la_vez():
    acum = acumulado(lambda e, q: (1, 8))
    acum[("b|x", 2, 5)]["monotono"]["con"] = 5              # el segundo grupo falla solo en (2 %, 5 %)
    r = bis.elegir(acum, GRUPOS)
    assert (r["epsilon_pct"], r["q_pct"]) == (2, 10)         # mismo ε, siguiente q


def test_sin_celda_que_cumpla_no_hay_veredicto_por_omision():
    acum = acumulado(lambda e, q: (9, 10))
    d = bis.decidir_tarea(acum, GRUPOS)
    assert d["eleccion"] is None and d["veredicto"] == "sin veredicto por omisión" and d["planas"] is None


def test_la_etiqueta_invertida_no_elige_lo_mismo_en_un_instrumento_coherente():
    acum = acumulado(lambda e, q: (1, 8))
    d = bis.decidir_tarea(acum, GRUPOS)
    assert d["eleccion"]["epsilon_pct"] == 2 and d["eleccion_con_etiqueta_invertida"] is None
    assert d["sabotaje"]["distingue"] is True and d["veredicto"] == "ε = 2 %, q = 5 %"


def test_si_la_invertida_eligiera_lo_mismo_se_anula_la_eleccion(monkeypatch):
    acum = acumulado(lambda e, q: (1, 8))
    monkeypatch.setattr(bis, "elegir", lambda acum, grupos, invertida=False: {"epsilon_pct": 5, "q_pct": 10, "q_que_cumplen_con_ese_epsilon": [10]})
    d = bis.decidir_tarea(acum, GRUPOS)
    assert d["sabotaje"]["distingue"] is False and d["veredicto"] == "elección ANULADA por el sabotaje"


def test_las_planas_bajo_el_90_por_ciento_se_declaran_sin_anular_la_eleccion():
    # 2 grupos × 6 casos = 12 por plana: 10 de 12 = 83 % < 90 % -> «las planas no se distinguen», la elección sigue.
    d = bis.decidir_tarea(acumulado(lambda e, q: (1, 8), planas_nsc=10 * 1), GRUPOS)
    assert d["eleccion"] is not None and d["planas"]["declaracion"] == "las planas no se distinguen"
    assert not d["planas"]["todas_salen_no_se_pudo_comprobar_en_al_menos_90_pct"]
    # 12 de 12 -> bien. (El acumulador reparte `planas_nsc // 2` por grupo.)
    d = bis.decidir_tarea(acumulado(lambda e, q: (1, 8), planas_nsc=12), GRUPOS)
    assert d["planas"]["declaracion"] is None and d["planas"]["por_columna"]["x7"]["fraccion_no_se_pudo_comprobar"] == 1.0


def test_las_planas_no_cuentan_como_monotonas_ni_como_no_monotonas():
    # Un acumulado donde las planas marcan «con» en todo no cambia FA ni POT: van aparte.
    acum = acumulado(lambda e, q: (1, 8))
    for celda in acum.values():
        for c in bis.PLANAS:
            celda["plana"][c].update(con=6, sin=0, nsc=0)
    t = bis.tasas(acum, GRUPOS, 5, 5)
    assert t["a|x"]["FA"] == 0.1 and t["a|x"]["POT"] == 0.8 and t["a|x"]["n_mon"] == 10


def _lo_que_dice_al_arrancar(guion, capsys, monkeypatch, tmp_path):
    """Arranca el `main()` de un guion de la medición con una salida IMPOSIBLE (dentro de un fichero): escribe su aviso y revienta
    al crear la carpeta, antes de medir nada ni tocar ningún resultado."""
    (tmp_path / "fichero").write_text("x")
    monkeypatch.setattr(sys, "argv", [guion.__file__, "--salida", str(tmp_path / "fichero" / "x")])
    try:
        guion.main()
    except (NotADirectoryError, FileExistsError, OSError):
        pass
    return capsys.readouterr().out


def test_los_guiones_del_c0_y_c0bis_dicen_al_arrancar_que_con_el_nucleo_nuevo_ya_no_reproducen_lo_archivado(capsys, monkeypatch, tmp_path):
    """Remate del C1-bis: desde que el núcleo da V'(ε, q) por amplitud, estos guiones (que llaman a su `.veredicto` / dependen de
    su `comportamiento_sha256`) ya no reproducen sus resultados archivados, y NADA lo avisaba. Una línea al arrancar."""
    c0 = bis.c0
    dicho = _lo_que_dice_al_arrancar(c0, capsys, monkeypatch, tmp_path)
    assert "AVISO" in dicho and "149-C1-bis" in dicho and "NO reproduce" in dicho and "comportamiento_sha256" in dicho
    dicho = _lo_que_dice_al_arrancar(bis, capsys, monkeypatch, tmp_path)
    assert "AVISO" in dicho and "149-C1-bis" in dicho and "comportamiento_sha256" in dicho
