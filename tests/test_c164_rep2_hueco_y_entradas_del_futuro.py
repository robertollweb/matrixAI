"""164 · reparación de la auditoría 2 (rep2), en el núcleo.

- A-1: con un calendario LABORABLE común y h > 1, el criterio de los cortes quitaba del 19 al 76 % de las filas con una causa falsa:
  el hueco del reparto era h × paso mediano en tiempo natural, y h pasos laborables no son h días. Ahora el hueco se ELIGE
  (`_elegir_hueco`: el que menos filas cuesta) y solo se quitan las filas que siguen pudiendo entrenar con un valor aún no
  ocurrido. Comprobado contra TODO instante posible como primer origen validado (no contra un reparto concreto).
- A-2: una columna que describe el FUTURO (`llueve_manana` de lluvia_MUNDIAL) entraba como «valor en el momento de predecir»:
  `entradas_que_describen_el_futuro` la mide; las legítimas de lluvia, Avocado y Uber no salen.
- M-1: el expediente dice con qué métrica se cuenta el «gana en X de Y pliegues».
- M-3: el banco del criterio de los cortes, con dientes en su FRONTERA (un corte de exactamente 2 días).
"""
import bisect
import csv
import math
import random
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from matrixai.training.pronostico_temporal import (
    construir_ventanas, entradas_que_describen_el_futuro, evaluar_hacia_delante,
)

LLUVIA = Path("/home/deployer/lluvia_MUNDIAL.csv")
DATOS_114 = Path("/home/deployer/fechas_114c5_datos")
DIA = 86400.0


class _Media:
    def fit(self, X, y):
        self.m = float(sum(y) / len(y))
        return self

    def predict(self, X):
        return [self.m] * len(X)


def fuga_exhaustiva(v) -> int:
    """Para CADA instante `s` de las filas que quedan, tomado como primer origen validado de un reparto hacia delante con el hueco
    de las ventanas (`s − t0 > hueco` entrena): ¿alguna fila que entrenaría tiene su objetivo DESPUÉS de `s`? Devuelve cuántos
    instantes `s` tienen fuga. Independiente del reparto concreto (el de la aud2, `sonda_cortes.py`, con `v.hueco`)."""
    orden = sorted((v.tiempo_por_id[i], v.tiempo_del_objetivo_por_id[i]) for i in v.tiempo_por_id)
    ts = [a for a, _ in orden]
    maximo, prefijo = -math.inf, []
    for _, b in orden:
        maximo = max(maximo, b)
        prefijo.append(maximo)
    peor = 0
    for s in sorted(set(ts)):
        k = bisect.bisect_left(ts, s - v.hueco)
        if k and prefijo[k - 1] > s:
            peor += 1
    return peor


def _laborable(tiendas: int = 4, dias: int = 200, seed: int = 3):
    rng = random.Random(seed)
    base = datetime(2025, 1, 6)       # un lunes
    return [{"serie": f"T{s}", "fecha": (base + timedelta(days=d)).strftime("%Y-%m-%d"), "y": f"{rng.gauss(10, 1):.2f}"}
            for s in range(tiendas) for d in range(dias) if (base + timedelta(days=d)).weekday() < 5]


def _con_cortes_de_2_dias(seed: int = 5):
    """5 series diarias, 3 con tres cortes de EXACTAMENTE 2 días (un fin de semana sin datos): la frontera del criterio."""
    rng = random.Random(seed)
    base, filas = datetime(2025, 1, 1), []
    for s in range(5):
        fuera = {d for a in rng.sample(range(10, 190), 3) for d in (a, a + 1)} if s < 3 else set()
        filas += [{"serie": f"S{s}", "fecha": (base + timedelta(days=d)).strftime("%Y-%m-%d"), "y": f"{rng.gauss(10, 1):.2f}"}
                  for d in range(200) if d not in fuera]
    return filas


def _un_corte_de_una_serie():
    rng = random.Random(3)
    base = datetime(2025, 1, 6)
    return [{"serie": f"S{s}", "fecha": (base + timedelta(days=d)).strftime("%Y-%m-%d"), "y": f"{rng.gauss(10, 1):.2f}"}
            for s in range(5) for d in range(300) if not (s == 0 and 100 <= d < 107)]


# ----------------------------------------------------------------------------------------------------------------------
# A-1 — el calendario laborable común con h > 1
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("h, hueco_en_dias, quitaria_el_minimo", [(2, 3, 112), (3, 4, 224), (5, 6, 436)])
def test_un_calendario_laborable_comun_con_h_mayor_que_1_no_quita_filas_y_ensancha_el_hueco(h, hueco_en_dias, quitaria_el_minimo):
    """aud2 A-1: antes, h = 2 → 444 de 576 (19 % fuera), h = 3 → 328 (39 %), h = 5 → 108 (76 %, solo lunes), con la causa «su
    serie tiene un corte». Ahora no se quita ninguna por cortes: el hueco del reparto es el de h pasos LABORABLES."""
    filas = _laborable()
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=h, retardos=3)
    d = v.declaracion
    assert d["filas_quitadas_por_cortes_del_calendario"] == 0 and d["cortes_del_calendario"] == []
    assert d["filas_resultantes"] == d["filas_de_entrada"] - d["filas_quitadas_por_el_borde_de_la_serie"]
    assert v.hueco == hueco_en_dias * DIA
    assert d["hueco"] == {"segundos_o_unidades": hueco_en_dias * DIA, "minimo": h * DIA, "eleccion": "calendario",
                          "filas_que_quitaria_el_minimo": quitaria_el_minimo,
                          "otro_conservaria": None}      # 164-rep3 (aud3 M-2): solo con un hueco PEDIDO
    # quedan orígenes en LOS CINCO días de la semana (antes, con h = 5, solo lunes)
    dias = Counter(datetime.strptime(f["fecha"], "%Y-%m-%d").weekday() for f in v.filas)
    assert set(dias) == {0, 1, 2, 3, 4} and min(dias.values()) >= 100
    assert fuga_exhaustiva(v) == 0


def test_con_h_1_el_laborable_sigue_con_el_hueco_minimo():
    v = construir_ventanas(_laborable(), tiempo="fecha", serie="serie", objetivo="y", horizonte=1, retardos=3)
    assert v.hueco == DIA and v.declaracion["hueco"]["eleccion"] == "minimo" and fuga_exhaustiva(v) == 0


def test_el_examen_hacia_delante_usa_el_hueco_de_las_ventanas_y_no_tiene_fuga():
    v = construir_ventanas(_laborable(), tiempo="fecha", serie="serie", objetivo="y", horizonte=5, retardos=3)
    r = evaluar_hacia_delante(v, columnas_de_entrada=["y_lag1", "y"], fabricar_modelo=_Media, folds=4)
    assert r["viable"] and r["hueco_segundos_o_unidades"] == 6 * DIA and r["violaciones_de_fuga"] == []


def test_un_corte_de_una_sola_serie_sigue_costando_sus_pocas_filas_y_no_ensancha_el_hueco():
    """El lado contrario (arreglar un sesgo puede crear el otro): un sensor caído 7 días en 1 de 5 series. Ensanchar el hueco a 9
    días costaría ~5 filas por día y examen; quitar las 3 filas que cruzan el corte cuesta 3. Se quitan, con el hueco mínimo."""
    v = construir_ventanas(_un_corte_de_una_serie(), tiempo="fecha", serie="serie", objetivo="y", horizonte=3, retardos=3)
    d = v.declaracion
    assert d["filas_quitadas_por_cortes_del_calendario"] == 3 and [c["serie"] for c in d["cortes_del_calendario"]] == ["S0"]
    assert v.hueco == 3 * DIA and d["hueco"]["eleccion"] == "minimo"
    assert fuga_exhaustiva(v) == 0


def test_el_hueco_pedido_es_el_suyo_y_con_el_minimo_vuelven_a_quitarse_las_filas_que_tendrian_fuga():
    v = construir_ventanas(_laborable(), tiempo="fecha", serie="serie", objetivo="y", horizonte=5, retardos=3, hueco=5 * DIA)
    d = v.declaracion
    assert d["hueco"]["eleccion"] == "pedido" and v.hueco == 5 * DIA
    assert d["filas_quitadas_por_cortes_del_calendario"] == 436
    assert fuga_exhaustiva(v) == 0


# ----------------------------------------------------------------------------------------------------------------------
# M-3 — el criterio contra TODO instante posible, en su frontera
# ----------------------------------------------------------------------------------------------------------------------

def _festivos_locales():
    rng = random.Random(3)
    base, filas = datetime(2025, 1, 6), []
    for s in range(4):
        fest = set(rng.sample(range(200), 6))
        filas += [{"serie": f"T{s}", "fecha": (base + timedelta(days=d)).strftime("%Y-%m-%d"), "y": f"{rng.gauss(10, 1):.2f}"}
                  for d in range(200) if (base + timedelta(days=d)).weekday() < 5 and d not in fest]
    return filas


@pytest.mark.parametrize("nombre, filas, h, hueco_pedido", [
    ("cortes de 2 días, h = 1, hueco mínimo pedido", _con_cortes_de_2_dias(), 1, DIA),
    ("cortes de 2 días, h = 2, hueco mínimo pedido", _con_cortes_de_2_dias(), 2, 2 * DIA),
    ("cortes de 2 días, h = 1, hueco elegido", _con_cortes_de_2_dias(), 1, None),
    ("laborable, h = 3, hueco mínimo pedido", _laborable(), 3, 3 * DIA),
    ("laborable, h = 5, hueco elegido", _laborable(), 5, None),
    ("festivos locales, h = 1", _festivos_locales(), 1, None),
    ("un corte de 7 días en una serie, h = 3", _un_corte_de_una_serie(), 3, None),
])
def test_ningun_instante_posible_como_primer_origen_validado_deja_una_fuga(nombre, filas, h, hueco_pedido):
    """aud2 M-3: el sabotaje B2 (`instantes[j] < t1 - paso_mediano`: saltarse el último instante antes del objetivo) salía VERDE
    en el núcleo y en el backend. Con el hueco MÍNIMO pedido, los cortes de 2 días caen justo en la frontera: hay filas que se
    tienen que quitar y el comprobador exhaustivo dice si alguna se queda."""
    v = construir_ventanas(filas, tiempo="fecha", serie="serie", objetivo="y", horizonte=h, retardos=3, hueco=hueco_pedido)
    if hueco_pedido is not None and "cortes de 2" in nombre:
        assert v.declaracion["filas_quitadas_por_cortes_del_calendario"] > 0     # la frontera se ejercita de verdad
    assert fuga_exhaustiva(v) == 0, nombre


# ----------------------------------------------------------------------------------------------------------------------
# A-2 — las entradas que describen el futuro
# ----------------------------------------------------------------------------------------------------------------------

def _panel_sintetico(dias: int = 300, ciudades: int = 5, seed: int = 7):
    """Lluvia sintética con la FORMA de lluvia_MUNDIAL: muchos ceros, humedad ligada a la lluvia DE HOY y una columna
    `llueve_manana` que es `rain(t+1) > 0` (lo que trae el fichero real)."""
    rng = random.Random(seed)
    filas = []
    for c in range(ciudades):
        lluvia = []
        humedo = 0.0
        for _ in range(dias + 1):
            humedo = 0.6 * humedo + rng.gauss(0, 1)
            lluvia.append(max(0.0, humedo + rng.gauss(0, 0.7) - 0.4) * 3)
        for d in range(dias):
            filas.append({"ciudad": f"C{c}", "dia": (date(2020, 1, 1) + timedelta(days=d)).isoformat(),
                          "rain": f"{lluvia[d]:.1f}", "humidity": f"{60 + 8 * lluvia[d] / 3 + rng.gauss(0, 4):.1f}",
                          "elevation": str(100 * c), "llueve_manana": "1" if lluvia[d + 1] > 0 else "0"})
    return filas


def _salen(medidas):
    return sorted(m["columna"] for m in medidas if m["describe_el_futuro"])


def test_una_columna_sintetica_que_es_el_objetivo_de_manana_sale_y_la_humedad_de_hoy_no():
    filas = _panel_sintetico()
    for h in (1, 3):
        r = entradas_que_describen_el_futuro(filas, tiempo="dia", objetivo="rain", horizonte=h, serie="ciudad",
                                             columnas=["humidity", "elevation", "llueve_manana"])
        assert _salen(r) == ["llueve_manana"], r
        m = next(x for x in r if x["columna"] == "llueve_manana")
        assert m["k"] == 1 and m["futuro"] - m["ahora"] >= 0.1 and m["pares"] >= 1000


def _lluvia_real(ciudades: int = 5, dias: int = 300):
    """Las primeras `ciudades` del fichero real (agrupado por ciudad), `dias` días de cada una, con un día reconstruido."""
    por: dict = {}
    with LLUVIA.open(newline="") as fh:
        for row in csv.DictReader(fh):
            k = (row["lat"], row["lon"])
            if k not in por:
                if len(por) == ciudades:
                    break
                por[k] = []
            if len(por[k]) < dias:
                por[k].append(row)
    filas = []
    for n, rows in enumerate(por.values()):
        for d, row in enumerate(rows):
            filas.append({**row, "ciudad": f"C{n}", "dia": (date(2020, 1, 1) + timedelta(days=d)).isoformat()})
    return filas


@pytest.mark.skipif(not LLUVIA.exists(), reason="lluvia_MUNDIAL.csv no está en esta máquina")
def test_en_lluvia_mundial_llueve_manana_sale_y_ninguna_de_las_otras_15_columnas():
    """El caso REAL de la aud2 A-2: `llueve_manana(t) == (rain(t+1) > 0)` en 1.495 de 1.495 pares. Con h = 1 y con h = 3 (la
    columna habla del día siguiente, que también es futuro con h = 3). Control: las 15 legítimas, incluida `precip_hours`, la
    que más explica la lluvia de HOY."""
    filas = _lluvia_real()
    columnas = [c for c in filas[0] if c not in ("rain", "dia", "ciudad")]
    assert len(columnas) == 16
    for h in (1, 3):
        r = entradas_que_describen_el_futuro(filas, tiempo="dia", objetivo="rain", horizonte=h, serie="ciudad", columnas=columnas)
        assert _salen(r) == ["llueve_manana"], [(m["columna"], m["ahora"], m["futuro"]) for m in r]
        assert all(m["describe_el_futuro"] is False for m in r if m["columna"] != "llueve_manana")


def _arff(nombre: str):
    raiz = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(raiz / "benchmarks" / "fechas_114c5"))
    from medir_114c5 import leer_arff  # noqa: PLC0415
    n, _ = leer_arff(DATOS_114 / nombre)
    return [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in n]


@pytest.mark.skipif(not (DATOS_114 / "43820.arff").exists(), reason="Avocado (OpenML 43820) no está en esta máquina")
def test_ninguna_columna_de_avocado_sale():
    filas = [{**r, "serie": f'{r["region"]}|{r["type"]}'} for r in _arff("43820.arff")]
    columnas = [c for c in filas[0] if c not in ("row_id", "Date", "AveragePrice", "serie")]
    r = entradas_que_describen_el_futuro(filas, tiempo="Date", objetivo="AveragePrice", horizonte=1, serie="serie", columnas=columnas)
    assert _salen(r) == [] and all(m["describe_el_futuro"] is False for m in r), r


@pytest.mark.skipif(not (DATOS_114 / "43746.arff").exists(), reason="Uber (OpenML 43746) no está en esta máquina")
@pytest.mark.parametrize("h", [1, 6])
def test_ninguna_columna_de_uber_sale_con_h_1_ni_con_h_6(h):
    """Con h = 6 la persistencia de Uber explica casi nada (ciclo diario): por eso se compara con la propia fila, no con ella."""
    filas = [r for r in _arff("43746.arff") if r.get("borough")]
    columnas = [c for c in filas[0] if c not in ("row_id", "pickup_dt", "borough", "pickups")]
    r = entradas_que_describen_el_futuro(filas, tiempo="pickup_dt", objetivo="pickups", horizonte=h, serie="borough",
                                         columnas=columnas)
    assert _salen(r) == [] and all(m["describe_el_futuro"] is False for m in r), r


def test_con_pocas_filas_no_se_da_por_comprobada():
    filas = _panel_sintetico(dias=12, ciudades=2)
    r = entradas_que_describen_el_futuro(filas, tiempo="dia", objetivo="rain", horizonte=1, serie="ciudad",
                                         columnas=["llueve_manana"])
    assert r[0]["describe_el_futuro"] is None and r[0]["pares"] < 50


# ----------------------------------------------------------------------------------------------------------------------
# M-1 — la métrica del «gana en X de Y pliegues» en el expediente
# ----------------------------------------------------------------------------------------------------------------------

def test_el_expediente_dice_con_que_metrica_se_cuentan_los_pliegues(tmp_path):
    from matrixai.export.tripod import ficha_tripod
    from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio, manifiesto_de_estudio
    from tests.test_c164_c4b_pronostico_en_el_expediente import PRONOSTICO

    con = {**PRONOSTICO, "veredicto_por_pliegue": {**PRONOSTICO["veredicto_por_pliegue"], "metrica": "rmse"}}
    assert manifiesto_de_estudio(pronostico=con)["study"]["forecast"]["fold_verdict"]["metric"] == "rmse"
    escribir_manifiesto_de_estudio(tmp_path, filas=1000, pronostico=con)
    es = next(l for l in ficha_tripod(tmp_path, locale="es").splitlines() if "study.forecast" in l)
    en = next(l for l in ficha_tripod(tmp_path, locale="en").splitlines() if "study.forecast" in l)
    assert "gana en 2 de 3 pliegues de la búsqueda (medido con el RMSE, la métrica con que se eligió el campeón)" in es
    assert "wins on 2 of 3 folds of the search (measured with the RMSE, the metric the champion was chosen by)" in en
    # un sobre anterior (sin la métrica) no la inventa, pero tampoco deja que se lea como el MAE de la prueba
    otro = tmp_path / "sin"
    otro.mkdir()
    escribir_manifiesto_de_estudio(otro, filas=1000, pronostico=PRONOSTICO)
    es = next(l for l in ficha_tripod(otro, locale="es").splitlines() if "study.forecast" in l)
    assert "no con el MAE de la prueba" in es and "RMSE" not in es
