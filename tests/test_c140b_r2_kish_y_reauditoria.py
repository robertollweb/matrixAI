# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b — la corrección de Kish (decisión de Roberto del 01-11, tarjeta `140b-wilson-por-sujetos`, opción A) y la segunda
reparación (re-auditoría `~/.agentes/140b/reauditoria/INFORME.md`, N2 y N4–N7), en el núcleo.

- Kish: el intervalo de Wilson de la cobertura conforme, de las bandas y de la sensibilidad de la matriz se corrige por sujeto
  con el efecto de diseño (`intervalo_por_filas.correccion_por_sujetos`), lleva la cobertura MEDIDA de ese intervalo y la
  marca de aproximado; con un sujeto por fila, o sin sujetos, todo queda como antes.
- N2: el rechazo nombra lo que no llegó (el desarrollo), no la prueba. N4: cada banda cuenta SUS sujetos. N5: un sujeto por
  fila, ni una clave más. N6: «7.0»/«7» y «P-007»/«P7» se avisan, y un id de miles de cifras no lanza. N7: la coma decimal en
  los límites del estricto en castellano.

Cada prueba con nombre de hallazgo salió ROJA con el código anterior (revirtiendo solo su arreglo).
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.estudio.intervalo_por_filas import (   # noqa: E402
    COBERTURA_MEDIDA_POR_SUJETOS, SUJETOS_PARA_QUE_NO_SEA_APROXIMADO, cobertura_medida_por_sujetos,
    correccion_de_recuentos, correccion_por_sujetos, intervalo_recortado)
from matrixai.estudio.metricas import Muestra   # noqa: E402

#: La medición, VERSIONADA (K4 de la tercera auditoría: antes vivía en `~/.agentes` y la prueba se saltaba en un clon).
MEDICION = Path(__file__).resolve().parents[1] / "benchmarks" / "cobertura_140b" / "kish" / "medir_kish.json"


# ---------------------------------------------------------------------------------------------------------------------------
# Kish: el instrumento
# ---------------------------------------------------------------------------------------------------------------------------

def _cobertura_simulada(sujetos: int, filas: int, sd: float, tasa: float, repeticiones: int, seed: int) -> tuple[float, float]:
    """(cobertura del corregido, cobertura del de filas) con sujetos de tasa propia (logit normal), en Python puro."""
    rng = random.Random(seed)
    base = math.log(tasa / (1 - tasa))
    verdad = sum(1 / (1 + math.exp(-(base + sd * rng.gauss(0, 1)))) for _ in range(40000)) / 40000
    dentro_k = dentro_f = 0
    for _ in range(repeticiones):
        exitos = []
        for _ in range(sujetos):
            p = 1 / (1 + math.exp(-(base + sd * rng.gauss(0, 1))))
            exitos.append(sum(rng.random() < p for _ in range(filas)))
        c = correccion_de_recuentos(exitos, [filas] * sujetos)
        bajo, alto = c.intervalo_95
        dentro_k += bajo <= verdad <= alto
        bajo, alto = intervalo_recortado(sum(exitos), filas * sujetos)
        dentro_f += bajo <= verdad <= alto
    return dentro_k / repeticiones, dentro_f / repeticiones


def test_kish_el_control_sin_efecto_del_sujeto_cubre_alrededor_del_95():
    """Sin efecto del sujeto (todas las tasas iguales) las filas SÍ son independientes: el corregido cubre como el nominal
    (el instrumento está bien), y no se ensancha de más."""
    kish, filas = _cobertura_simulada(13, 40, 0.0, 0.9, 500, seed=1)
    assert 0.92 <= kish <= 0.985, kish
    assert 0.92 <= filas <= 0.985, filas


@pytest.mark.parametrize("sujetos,filas,sd", [(8, 40, 0.5), (8, 40, 1.0), (13, 40, 0.5), (13, 40, 1.0), (17, 20, 1.0),
                                              (50, 20, 1.0)])
def test_kish_con_efecto_del_sujeto_la_cobertura_mejora_en_las_celdas_de_la_sonda(sujetos, filas, sd):
    """Las celdas de `reparacion/sonda_wilson_kish.py`: el de filas cubre 0,60–0,90; el corregido, mucho más (medido con R =
    2.000: 0,87–0,94)."""
    kish, por_filas = _cobertura_simulada(sujetos, filas, sd, 0.9, 300, seed=sujetos * 100 + int(sd * 10))
    assert kish >= por_filas + (0.03 if sd == 0.5 else 0.10), (kish, por_filas)
    assert kish >= 0.84, kish


def test_kish_la_tabla_del_codigo_es_la_medida():
    """`COBERTURA_MEDIDA_POR_SUJETOS` es, número a número, la PEOR forma de tamaños (en media de sus celdas con efecto) de
    `medir_kish.json`, que está en el repositorio: sin `skip`. Y ese resumen se rehace AQUÍ de las celdas, no se lee."""
    assert MEDICION.exists(), f"falta {MEDICION}: la procedencia de la tabla va versionada"
    medicion = json.loads(MEDICION.read_text())
    tabla = medicion["tabla"]
    assert [(int(k), v["kish_peor_forma_media"]) for k, v in tabla.items()] == list(COBERTURA_MEDIDA_POR_SUJETOS)
    rehecha = []
    for k, _ in COBERTURA_MEDIDA_POR_SUJETOS:
        por_forma: dict[str, list[float]] = {}
        for c in medicion["celdas"]:
            if c["sujetos"] == k and c["sd_logit"] > 0:
                por_forma.setdefault(c["forma"], []).append(c["kish"])
        assert set(por_forma) == {"5", "20", "40", "1-39", "lognormal_cv1.5"} and {len(v) for v in por_forma.values()} == {6}
        rehecha.append((k, min(round(sum(v) / len(v), 3) for v in por_forma.values())))
    assert rehecha == list(COBERTURA_MEDIDA_POR_SUJETOS)
    assert medicion["nucleo"] == "matrixai/__init__.py" and medicion["R"] == 2000


def test_kish_no_depende_del_orden_de_los_sujetos():
    """El Informe Vivo tabula `n_efectivo` y la página rehace el intervalo con él: tiene que ser el mismo al último bit,
    vengan los sujetos en el orden que vengan (`math.fsum`)."""
    rng = random.Random(3)
    filas = [rng.randint(1, 30) for _ in range(40)]
    exitos = [rng.randint(0, m) for m in filas]
    referencia = correccion_de_recuentos(exitos, filas)
    for _ in range(20):
        orden = list(range(40))
        rng.shuffle(orden)
        otra = correccion_de_recuentos([exitos[i] for i in orden], [filas[i] for i in orden])
        assert (otra.icc, otra.efecto_de_diseno, otra.n_efectivo, otra.intervalo_95) == \
               (referencia.icc, referencia.efecto_de_diseno, referencia.n_efectivo, referencia.intervalo_95)


# ---------------------------------------------------------------------------------------------------------------------------
# Kish: los casos raros
# ---------------------------------------------------------------------------------------------------------------------------

def test_kish_un_icc_negativo_se_trunca_a_cero_y_el_intervalo_es_el_de_filas():
    """Cada sujeto con la MITAD exacta de aciertos: entre sujetos no hay varianza, el ANOVA da un ICC negativo y se trunca."""
    c = correccion_de_recuentos([5] * 10, [10] * 10)
    assert c.icc == 0.0 and c.efecto_de_diseno == 1.0 and c.n_efectivo == 100.0
    assert c.intervalo_95 == c.intervalo_por_filas == intervalo_recortado(50, 100)


@pytest.mark.parametrize("exitos", [[0] * 8, [10] * 8])
def test_kish_todas_cero_o_todas_uno_toma_lo_prudente_y_no_da_cobertura(exitos):
    """K2 (tercera auditoría): sin varianza se tomaba ICC = 0 —el Wilson por filas— y ese intervalo no cubría NUNCA. Ahora,
    como si ICC = 1: `n_eff = N²/Σm²` (aquí, los 8 sujetos), sin cifra de cobertura y con la marca."""
    c = correccion_de_recuentos(exitos, [10] * 8)
    assert c.icc is None and c.sin_varianza is True and c.n_efectivo == 8.0
    assert c.intervalo_95 == intervalo_recortado(sum(exitos) / 80 * 8, 8)
    assert c.intervalo_95 != intervalo_recortado(sum(exitos), 80)
    assert c.cobertura_medida is None and c.sujetos_del_punto_medido is None and c.aproximado is True
    assert "de cada 100" not in c.texto["es"] and "in 100" not in c.texto["en"]
    assert "todas dan lo mismo" in c.texto["es"] and "no hay cifra que dar" in c.texto["es"]
    assert "all give the same result" in c.texto["en"] and "there is no figure to give" in c.texto["en"]
    assert "todas las filas dan lo mismo" in c.texto_aproximado["es"]


def test_kish_un_solo_sujeto_vale_como_una_fila_y_no_tiene_cobertura_medida():
    c = correccion_por_sujetos([1, 1, 0, 1, 1], ["P1"] * 5)
    assert c.unidades == 1 and c.icc is None and c.n_efectivo == 1.0
    assert c.intervalo_95 == intervalo_recortado(4 / 5, 1.0)
    assert c.cobertura_medida is None and c.aproximado is True
    assert "son todas de un solo sujeto y valen como una" in c.texto["es"] and "no hay cifra que dar" in c.texto["es"]
    assert "all come from a single subject and count as one" in c.texto["en"]


def test_kish_un_sujeto_por_fila_o_sin_sujetos_no_corrige_nada():
    assert correccion_por_sujetos([1, 0, 1], ["a", "b", "c"]) is None
    assert correccion_por_sujetos([1, 0, 1], None) is None
    assert correccion_de_recuentos([1, 0, 1], [1, 1, 1]) is None
    assert correccion_de_recuentos([], []) is None


def test_kish_la_cobertura_de_un_numero_no_medido_es_la_del_punto_menor_y_nunca_la_de_arriba():
    assert cobertura_medida_por_sujetos(1) == (None, None)
    assert cobertura_medida_por_sujetos(7) == (0.907, 6)
    assert cobertura_medida_por_sujetos(12) == (0.923, 10)
    assert cobertura_medida_por_sujetos(5000) == (0.938, 100)


# ---------------------------------------------------------------------------------------------------------------------------
# Kish: las frases, en el NÚCLEO y en los dos idiomas
# ---------------------------------------------------------------------------------------------------------------------------

def test_kish_las_frases_dicen_la_correccion_y_la_cobertura_medida_en_los_dos_idiomas():
    exitos = [38, 30, 40, 22, 35, 39, 28, 40, 33, 37, 25, 36, 39]
    c = correccion_de_recuentos(exitos, [40] * 13)
    es, en = c.texto["es"], c.texto["en"]
    assert c.cobertura_medida == 0.924 and c.sujetos_del_punto_medido == 13 and not c.aproximado
    icc_es, icc_en = f"{c.icc:.2f}".replace(".", ","), f"{c.icc:.2f}"
    assert es.startswith("intervalo del 95 % corregido por sujeto: las 520 filas son de 13 sujetos")
    assert f"correlación intraclase {icc_es}" in es and f"efecto de diseño de Kish: {c.efecto_de_diseno:.2f}".replace(".", ",") in es
    assert "con 13 sujetos, un intervalo así acierta unas 92 de cada 100 veces, no 95 (medido por simulación)" in es
    assert en.startswith("95% interval corrected by subject: the 520 rows come from 13 subjects")
    assert f"intraclass correlation {icc_en}" in en
    assert "with 13 subjects, an interval like this is right about 92 times in 100, not 95 (measured by simulation)" in en
    assert c.texto_aproximado is None


def test_kish_la_marca_de_aproximado_va_por_debajo_de_10_sujetos_y_no_en_10():
    assert SUJETOS_PARA_QUE_NO_SEA_APROXIMADO == 10
    nueve = correccion_de_recuentos([3, 4, 5, 6, 7, 8, 9, 2, 1], [10] * 9)
    diez = correccion_de_recuentos([3, 4, 5, 6, 7, 8, 9, 2, 1, 5], [10] * 10)
    assert nueve.aproximado is True and nueve.texto_aproximado is not None
    assert "intervalo aproximado (el de Wilson corregido por sujeto): con menos de 10 sujetos" in nueve.texto_aproximado["es"]
    assert "approximate interval (the Wilson one corrected by subject): with fewer than 10 subjects" in \
        nueve.texto_aproximado["en"]
    assert diez.aproximado is False and diez.texto_aproximado is None
    assert nueve.a_json()["aproximado"] is True and diez.a_json()["texto_aproximado"] is None


# ---------------------------------------------------------------------------------------------------------------------------
# Kish en la cobertura conforme, las bandas y la matriz
# ---------------------------------------------------------------------------------------------------------------------------

def _regresion(unidades, n=200, seed=4, efecto=0.0):
    rng = random.Random(seed)
    sesgo = {}
    y, pred = [], []
    for i in range(n):
        u = unidades[i] if unidades is not None else None
        b = sesgo.setdefault(u, rng.gauss(0, efecto)) if efecto else 0.0
        v = rng.gauss(0, 1)
        y.append(v)
        pred.append(v + b + rng.gauss(0, 1))
    return Muestra.regresion(y, pred, unidades=unidades)


def test_kish_la_cobertura_conforme_vuelve_a_decir_compatible_con_el_intervalo_corregido():
    """10 sujetos de 20 filas, cada uno con su sesgo: el Wilson por filas es demasiado estrecho; el corregido, más ancho, y el
    veredicto es el de ESE intervalo."""
    from matrixai.estudio.conforme import calibrar, medir_cobertura
    calibracion = calibrar(_regresion(None, seed=1), alfa=0.1)
    unidades = [f"S{i // 20}" for i in range(200)]
    con = medir_cobertura(_regresion(unidades, efecto=1.5), calibracion)
    ps = con.por_sujetos
    assert ps is not None and ps.unidades == 10 and ps.efecto_de_diseno > 1
    assert con.intervalo_95 == ps.intervalo_95
    assert (ps.intervalo_95[1] - ps.intervalo_95[0]) > (ps.intervalo_por_filas[1] - ps.intervalo_por_filas[0])
    bajo, alto = ps.intervalo_95
    assert con.compatible_con_la_nominal is (bajo <= 0.9 <= alto)
    j = con.a_json()
    assert j["compatible_con_la_nominal"] is con.compatible_con_la_nominal and j["por_sujetos"]["n_efectivo"] == ps.n_efectivo


def test_N5_con_un_sujeto_por_fila_la_cobertura_y_las_bandas_no_ganan_ni_una_clave():
    """El comentario decía «con un sujeto por fila, ni una clave más» y el JSON ganaba `unidades` e `intervalo_por_filas: null`
    (re-auditoría N5): ahora es VERDAD, byte a byte con la muestra sin sujetos."""
    from matrixai.estudio.bandas import PoliticaDeBandas, medir_bandas
    from matrixai.estudio.conforme import calibrar, medir_cobertura
    calibracion = calibrar(_regresion(None, seed=1), alfa=0.1)
    uno = medir_cobertura(_regresion([f"S{i}" for i in range(200)]), calibracion)
    ninguno = medir_cobertura(_regresion(None), calibracion)
    assert json.dumps(uno.a_json()) == json.dumps(ninguno.a_json())
    rng = random.Random(5)
    p = [rng.random() for _ in range(120)]
    y = ["si" if rng.random() < q else "no" for q in p]
    politica = PoliticaDeBandas(umbral_bajo=0.3, umbral_alto=0.7, positive_label="si", negative_label="no",
                                coste_falso_positivo=1.0, coste_falso_negativo=1.0, coste_de_revision=0.1)
    con = medir_bandas(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p,
                                       unidades=[f"S{i}" for i in range(120)]), politica)
    sin = medir_bandas(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p), politica)
    assert [json.dumps(m.a_json()) for m in con] == [json.dumps(m.a_json()) for m in sin]


def test_N4_cada_banda_cuenta_los_sujetos_de_SUS_filas():
    """El sujeto de cada fila según el rango de su probabilidad: cada banda tiene sujetos DISTINTOS (los de la prueba entera son
    12). Contar los de toda la prueba (el sabotaje R6 de la re-auditoría) da 12 en las tres y esta prueba lo caza."""
    from matrixai.estudio.bandas import PoliticaDeBandas, medir_bandas
    rng = random.Random(5)
    p = sorted(rng.random() for _ in range(240))
    y = ["si" if rng.random() < q else "no" for q in p]
    unidades = [f"S{i // 20}" for i in range(240)]     # 12 sujetos, 20 filas seguidas cada uno
    politica = PoliticaDeBandas(umbral_bajo=0.3, umbral_alto=0.7, positive_label="si", negative_label="no",
                                coste_falso_positivo=1.0, coste_falso_negativo=1.0, coste_de_revision=0.1)
    medidas = medir_bandas(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p,
                                           unidades=unidades), politica)
    vistos = []
    for m in medidas:
        suyos = {unidades[i] for i in range(240) if politica.banda_de(p[i]) == m.banda}
        assert m.por_sujetos is not None and m.por_sujetos.unidades == len(suyos) < 12, (m.banda, len(suyos))
        assert m.a_json()["unidades"] == len(suyos)
        assert f"las {m.n} filas son de {len(suyos)} sujetos" in m.a_json()["por_sujetos"]["texto"]["es"]
        vistos.append(len(suyos))
    assert sum(vistos) >= 12


def test_kish_cada_banda_se_pinta_o_no_con_el_margen_del_intervalo_corregido():
    """Una banda de 300 filas de 3 sujetos con tasas muy distintas: por filas se daría por medida (±5 puntos); corregida, el
    margen pasa de ±10 y no se pinta, con un motivo que lo dice."""
    from matrixai.estudio.bandas import PoliticaDeBandas, medir_bandas
    p, y, u = [], [], []
    for s, tasa in enumerate((0.95, 0.6, 0.98)):
        rng = random.Random(s)
        for _ in range(100):
            p.append(0.9)
            y.append("si" if rng.random() < tasa else "no")
            u.append(f"S{s}")
    politica = PoliticaDeBandas(umbral_bajo=0.3, umbral_alto=0.7, positive_label="si", negative_label="no",
                                coste_falso_positivo=1.0, coste_falso_negativo=1.0, coste_de_revision=0.1)
    [positiva] = [m for m in medir_bandas(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p,
                                                          unidades=u), politica) if m.banda == "positiva"]
    [por_filas] = [m for m in medir_bandas(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p),
                                           politica) if m.banda == "positiva"]
    assert por_filas.medida is True and positiva.medida is False
    assert "de 3 sujetos en esta banda su acierto, corregido por sujeto, no se conoce" in positiva.motivo["es"]
    assert "from 3 subjects in this band its accuracy, corrected by subject, is not known" in positiva.motivo["en"]


def test_kish_la_sensibilidad_de_la_matriz_va_corregida_por_sujeto_y_sin_sujetos_no_hay_clave():
    from matrixai.estudio.curvas_de_la_evaluacion import curvas_de_la_evaluacion
    rng = random.Random(8)
    p, y, u = [], [], []
    for s in range(12):
        sesgo = rng.gauss(0, 1.5)
        for _ in range(15):
            positivo = rng.random() < 0.5
            y.append("si" if positivo else "no")
            p.append(min(0.99, max(0.01, 1 / (1 + math.exp(-((1.2 if positivo else -1.2) + sesgo + rng.gauss(0, 1)))))))
            u.append(f"S{s}")
    con = curvas_de_la_evaluacion(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p,
                                                  unidades=u), umbral=0.5)
    sin = curvas_de_la_evaluacion(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p), umbral=0.5)
    assert "sensibilidad_por_sujetos" not in sin["matriz"]
    s = con["matriz"]["sensibilidad_por_sujetos"]
    k = con["matriz"]["casillas"]
    assert (s["filas"], s["unidades"]) == (k["tp"] + k["fn"], 12)
    assert abs(s["intervalo_por_filas"][0] - intervalo_recortado(k["tp"], k["tp"] + k["fn"])[0]) < 1e-15
    assert {kk: v for kk, v in con["matriz"].items() if kk != "sensibilidad_por_sujetos"} == sin["matriz"]
    uno = curvas_de_la_evaluacion(Muestra.binaria(y, classes=["no", "si"], positive_label="si", probabilidades=p,
                                                  unidades=[f"F{i}" for i in range(len(y))]), umbral=0.5)
    assert uno["matriz"] == sin["matriz"]


# ---------------------------------------------------------------------------------------------------------------------------
# N2 · el rechazo culpa a lo que no llegó
# ---------------------------------------------------------------------------------------------------------------------------

def _cohorte_m1(n, vmin, vmax, seed):
    """La de la re-auditoría (`sondas/sonda_motivo_m1.py`): `n` pacientes con `vmin`–`vmax` visitas cada 3 días, desde el
    mismo día, la clase al 40 %."""
    rng = random.Random(seed)
    filas = []
    for k in range(n):
        ini = datetime(2023, 1, 1)
        for v in range(rng.randint(vmin, vmax)):
            filas.append({"pac": f"P{k}", "t": (ini + timedelta(days=3 * v)).strftime("%Y-%m-%d"),
                          "y": "Sí" if rng.random() < .4 else "No"})
    for i, f in enumerate(filas):
        f["rid"] = f"R{i}"
    return filas


@pytest.mark.parametrize("n,vmin,vmax,seed", [(30, 3, 7, 0), (20, 8, 15, 2), (10, 15, 30, 2)])
def test_N2_si_la_prueba_cumplia_y_no_llego_el_desarrollo_el_rechazo_dice_el_desarrollo(n, vmin, vmax, seed):
    from matrixai.training.particion_por_diseno_grupos_y_tiempo import MINIMO_FILAS_DE_PRUEBA, proponer_particion_estricta
    p = proponer_particion_estricta(_cohorte_m1(n, vmin, vmax, seed), plan_id="x", observation_id_field="rid", seed=0,
                                    unit_id_field="pac", time_column="t", objetivo="y")
    assert not p.es_viable
    [b] = p.bloqueos
    assert b.clave == "estricto_desarrollo_vacio", b.clave
    assert b.medida["filas_de_prueba"] >= MINIMO_FILAS_DE_PRUEBA      # la prueba de ese corte SÍ cumplía
    assert "la prueba tendría" not in b.motivo["es"] and "el desarrollo tiene" in b.motivo["es"]


# ---------------------------------------------------------------------------------------------------------------------------
# N6 · «7.0»/«7», «P-007»/«P7» y un id de miles de cifras
# ---------------------------------------------------------------------------------------------------------------------------

def test_N6_la_escritura_normalizada_ve_el_punto_cero_y_los_guiones_sin_confundir_decimales():
    from matrixai.training.particion_por_diseno_comun import escritura_normalizada as e
    assert e("7.0") == e("7") == e("7.00") == e("007") == "7"
    assert e("P-007") == e("P7") == e("p_7") == e(" P - 07 ") == "p7"
    assert e("1.05") != e("1.5") and e("7.5") != e("7") and e("10.0") != e("1")
    assert e("0") == e("000") == "0"


def test_N6_un_id_de_miles_de_cifras_no_hace_lanzar_al_aviso():
    from matrixai.training.particion_por_diseno_comun import limite_de_sujetos_que_se_escriben_igual
    largo = "1" * 5000
    limite = limite_de_sujetos_que_se_escriben_igual({"a": largo, "b": "0" + largo, "c": "9"}, "pac")
    assert limite is not None and limite.medida["grupos"] == 1 and limite.medida["valores"] == 2


@pytest.mark.parametrize("a,b", [("7.0", "7"), ("P-007", "P7")])
def test_N6_las_dos_escrituras_del_mismo_paciente_se_avisan_en_groups_y_en_el_estricto(a, b):
    from matrixai.training.particion_por_diseno import proponer_particion
    from matrixai.training.particion_por_diseno_grupos_y_tiempo import proponer_particion_estricta
    rng = random.Random(1)
    filas = []
    for k in range(40):
        ini = datetime(2023, 1, 1) + timedelta(days=rng.randrange(0, 200))
        for v in range(rng.randint(5, 20)):
            pid = f"P{k}" if k != 7 else (a if v % 2 else b)
            filas.append({"pac": pid, "t": (ini + timedelta(days=7 * v)).strftime("%d/%m/%Y"),
                          "y": "Sí" if rng.random() < .4 else "No"})
    for i, f in enumerate(filas):
        f["rid"] = f"R{i}"
    g = proponer_particion(filas, plan_id="x", observation_id_field="rid", split_type="groups", seed=0, unit_id_field="pac",
                           objetivo="y")
    s = proponer_particion_estricta(filas, plan_id="x", observation_id_field="rid", seed=0, unit_id_field="pac",
                                    time_column="t", objetivo="y")
    for p in (g, s):
        assert p.es_viable
        [aviso] = [l for l in p.limites if l.clave == "sujetos_que_se_escriben_igual"]
        assert aviso.medida["ejemplos"][0]["valores"] == sorted([a, b])
        assert "un «.0» final, los guiones" in aviso.motivo["es"] and "a trailing “.0”, dashes" in aviso.motivo["en"]


# ---------------------------------------------------------------------------------------------------------------------------
# N7 · la coma decimal en los límites del estricto
# ---------------------------------------------------------------------------------------------------------------------------

def test_N7_los_limites_del_estricto_escriben_los_decimales_como_cada_idioma():
    from test_c140b_c1_particion_estricta import _cohorte
    from matrixai.training.particion_por_diseno_grupos_y_tiempo import proponer_particion_estricta
    filas = _cohorte(40, 7, entrada=30, tiempo="iso")
    for f in filas:
        f["pac"], f["rid"] = f["sujeto"], f["id"]
    p = proponer_particion_estricta(filas, plan_id="x", observation_id_field="rid", seed=0, unit_id_field="pac",
                                    time_column="t", objetivo="y")
    [relajado] = [l for l in p.limites if l.clave == "estricto_corte_relajado"]
    [fuera] = [l for l in p.limites if l.clave == "estricto_filas_descartadas"]
    cuantil = f"{relajado.medida['cuantil_usado']:g}"
    porcentaje = f"{fuera.medida['porcentaje_descartado']:g}"
    assert "." in cuantil and "." in porcentaje       # los datos de la prueba tienen decimales que escribir
    assert f"cuantil {cuantil.replace('.', ',')}" in relajado.motivo["es"] and f"cuantil {cuantil}" not in relajado.motivo["es"]
    assert f"quantile {cuantil}" in relajado.motivo["en"]
    assert f"({porcentaje.replace('.', ',')} %)" in fuera.motivo["es"] and porcentaje in fuera.motivo["en"]


def test_el_efecto_de_diseno_es_el_de_kish_calculado_a_mano_con_el_icc_de_anova():
    """Supervisor, 01-11: `deff = 1 + m0·ICC` (sin el «−1») dejaba 49 pruebas verdes. Tercera auditoría (K1): con tamaños
    DESIGUALES el efecto de diseño de la varianza de la proporción por filas usa el tamaño PONDERADO `Σm²/N`, no el `m0` del
    ANOVA (que solo estima el ICC). Aquí el ICC de ANOVA de un factor se calcula APARTE, con la fórmula de libro, sobre
    sujetos de tamaños MUY distintos, y se exige `deff = 1 + (Σm²/N − 1)·ICC` y su n efectivo; y que NO sea el de `m0`."""
    from matrixai.estudio.intervalo_por_filas import correccion_de_recuentos

    exitos, filas = [36, 1, 7, 0, 5, 2, 19], [40, 3, 9, 6, 12, 2, 25]
    N, k = sum(filas), len(filas)
    p = sum(exitos) / N
    ssb = sum(m * (e / m - p) ** 2 for e, m in zip(exitos, filas))
    ssw = sum(e - e * e / m for e, m in zip(exitos, filas))          # Σ (yᵢⱼ − ȳᵢ)² con y ∈ {0, 1}
    msb, msw = ssb / (k - 1), ssw / (N - k)
    m0 = (N - sum(m * m for m in filas) / N) / (k - 1)
    icc = max(0.0, (msb - msw) / (msb + (m0 - 1) * msw))
    m_ponderado = sum(m * m for m in filas) / N
    deff = 1 + (m_ponderado - 1) * icc
    assert icc > 0.3, icc                                            # un caso con efecto de verdad
    assert m_ponderado > m0 + 10, (m_ponderado, m0)                  # tamaños desiguales: los dos tamaños se separan
    c = correccion_de_recuentos(exitos, filas)
    assert c is not None
    assert c.icc == pytest.approx(icc, rel=1e-9) and c.m0 == pytest.approx(m0, rel=1e-9)
    assert c.m_ponderado == pytest.approx(m_ponderado, rel=1e-12)
    assert c.efecto_de_diseno == pytest.approx(deff, rel=1e-9)
    assert c.n_efectivo == round(N / deff, 3)
    assert c.n_efectivo != round(N / (1 + (m0 - 1) * icc), 3)        # el de m0 (K1) se distingue
    assert c.n_efectivo != round(N / (1 + m_ponderado * icc), 3)     # y el que olvida el «−1»
