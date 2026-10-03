"""120 · C3′ — el veredicto de TabM como un motor más contra R1-GPU (enmienda 7, registrada antes de medir).

Cada prueba guarda UNA regla de `benchmarks/fase0/veredicto_120_c3p.py`. Se importa por RUTA, como hace el guion:
importar un módulo de `benchmarks/` no puede tocar `sys.path`.
"""
import importlib.util
from pathlib import Path

import pytest

_RUTA = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "veredicto_120_c3p.py"
_spec = importlib.util.spec_from_file_location("veredicto_120_c3p", _RUTA)
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)

TABM, DENSA = v.MOTOR_TABM, v.MOTOR_DENSA


def rec(nombre="ds", *, tarea="binary_classification", campeon="lightgbm", test=0.80, estado="completed",
        motores=("lightgbm", "sklearn.hgb", "baseline", DENSA), media=None, pared=100.0, metrica=None,
        rol="test", evidencia="independent_test"):
    m = metrica or v.METRICA_DE_CIERRE[tarea]
    return {"nombre": nombre, "tarea": tarea, "estado": estado, "campeon": campeon if estado == "completed" else None,
            "test": (None if test is None else {"rol": rol, "evidencia": evidencia, "metricas": {m: test}}),
            "intentos": [{"motor": x} for x in motores], "pared_estudio_s": pared,
            "media_de_la_seleccion": media}


# ------------------------------------------------------------------ la cifra
def test_la_cifra_es_la_del_TEST_con_la_metrica_de_cierre_de_cada_tarea():
    assert v.cifra_de_test(rec(test=0.8123)) == pytest.approx(81.23)
    assert v.cifra_de_test(rec(tarea="multiclass_classification", test=0.9)) == pytest.approx(90.0)
    assert v.cifra_de_test(rec(tarea="regression", test=0.42)) == pytest.approx(42.0)


def test_la_media_de_seleccion_NO_decide_aunque_suba():
    """La razón de la enmienda 7: con un candidato más, la media de selección del campeón nunca baja. Un C3′ cuya
    media sube y cuyo test baja tiene que salir BAJA."""
    r1 = rec(test=0.80, media={"metrica": "auroc", "motores": [{"motor": "lightgbm", "valor": 0.80, "compite": True}]})
    c3 = rec(campeon=TABM, test=0.77, motores=("lightgbm", DENSA, TABM),
             media={"metrica": "auroc", "motores": [{"motor": TABM, "valor": 0.90, "compite": True}]})
    comp = v.comparar(r1, c3, 0.0)
    assert comp["clase"] == "baja_2"
    assert comp["diferencia"] == pytest.approx(-3.0)


def test_completado_sin_test_o_sin_su_metrica_es_un_instrumento_roto():
    with pytest.raises(v.Incomparable):
        v.cifra_de_test(rec(test=None))
    with pytest.raises(v.Incomparable):
        v.cifra_de_test(rec(metrica="accuracy"))            # binaria sin AUROC en el test
    with pytest.raises(v.Incomparable):
        v.cifra_de_test(rec(rol="validation"))
    with pytest.raises(v.Incomparable):
        v.cifra_de_test(rec(evidencia="development_estimate"))
    assert v.cifra_de_test(rec(estado="failed")) is None   # no completar no es un instrumento roto: es un resultado


# ------------------------------------------------------------------ el suelo de ruido
def test_suelo_cero_si_las_dos_repeticiones_son_identicas():
    a = {"x": rec("x", test=0.8), "y": rec("y", tarea="regression", test=0.3)}
    b = {"x": rec("x", test=0.8), "y": rec("y", tarea="regression", test=0.3)}
    assert v.suelo_de_ruido(a, b) == 0.0


def test_suelo_es_la_mayor_diferencia_al_repetir():
    a = {"x": rec("x", test=0.800), "y": rec("y", test=0.700)}
    b = {"x": rec("x", test=0.803), "y": rec("y", test=0.695)}
    assert v.suelo_de_ruido(a, b) == pytest.approx(0.5)


@pytest.mark.parametrize("segunda", [
    {"x": rec("x", estado="failed")},                       # completa en una y no en la otra
    {"x": rec("x", campeon="sklearn.hgb")},                 # cambia el campeón
    {"z": rec("z")},                                        # otros conjuntos
])
def test_un_control_que_no_se_repite_es_incomparable(segunda):
    with pytest.raises(v.Incomparable):
        v.suelo_de_ruido({"x": rec("x")}, segunda)


# ------------------------------------------------------------------ presencia
def test_presencia_en_R1_GPU_la_densa_si_y_TabM_no():
    assert v.presencia(rec(), c3p=False) is None
    assert "torch" in v.presencia(rec(motores=("lightgbm", "baseline")), c3p=False)
    assert "TabM" in v.presencia(rec(motores=("lightgbm", DENSA, TABM)), c3p=False)


def test_presencia_en_C3p_las_dos_redes():
    assert v.presencia(rec(motores=("lightgbm", DENSA, TABM)), c3p=True) is None
    assert "TabM" in v.presencia(rec(), c3p=True)
    assert "torch" in v.presencia(rec(motores=("lightgbm", TABM)), c3p=True)
    assert v.presencia(rec(estado="failed", motores=()), c3p=True) is None   # no completó: ya cuenta como pérdida


# ------------------------------------------------------------------ comparar y el suelo
@pytest.mark.parametrize("test_c3, suelo, clase", [
    (0.812, 0.0, "sube"), (0.805, 0.0, "igual"), (0.785, 0.0, "baja"), (0.775, 0.0, "baja_2"),
    (0.812, 1.5, "igual"),                                  # 1,2 puntos bajo un suelo de 1,5 no cuentan
    (0.775, 1.5, "baja_2"),                                 # con suelo < 2, la bajada que corta sigue en 2
    (0.775, 3.0, "igual"),                                  # −2,5 bajo un suelo de 3 no cuenta, ni corta
])
def test_comparar_con_umbral_y_suelo(test_c3, suelo, clase):
    comp = v.comparar(rec(test=0.80), rec(campeon=TABM, test=test_c3, motores=(DENSA, TABM)), suelo)
    assert comp["clase"] == clase


def test_completar_cuenta_en_los_dos_sentidos():
    assert v.comparar(rec(), rec(estado="plazo"), 0.0)["clase"] == "deja_de_completar"
    assert v.comparar(rec(estado="failed"), rec(campeon=TABM), 0.0)["clase"] == "empieza_a_completar"
    assert v.comparar(None, rec(estado="failed"), 0.0)["clase"] == "ninguno_completa"


# ------------------------------------------------------------------ paridad
def test_paridad_sin_TabM_de_campeon_exige_la_cifra_exacta_con_suelo_cero():
    assert v.paridad(v.comparar(rec(), rec(motores=(DENSA, TABM)), 0.0), 0.0) is None
    c = v.comparar(rec(test=0.8), rec(test=0.80001, motores=(DENSA, TABM)), 0.0)
    assert "R1-GPU" in v.paridad(c, 0.0)


def test_paridad_admite_el_suelo_y_no_mira_a_TabM():
    c = v.comparar(rec(test=0.800), rec(test=0.802, motores=(DENSA, TABM)), 0.0)
    assert v.paridad(c, 0.5) is None
    c = v.comparar(rec(test=0.80), rec(campeon=TABM, test=0.70, motores=(DENSA, TABM)), 0.0)
    assert v.paridad(c, 0.0) is None                        # ganó TabM: eso es lo que se mide, no paridad


def test_paridad_el_campeon_no_puede_cambiar_entre_los_de_siempre():
    c = v.comparar(rec(campeon="lightgbm"), rec(campeon=DENSA, motores=(DENSA, TABM)), 0.0)
    assert "campeón cambia" in v.paridad(c, 0.0)


# ------------------------------------------------------------------ regla 9 y regla 4
def test_corta_en_la_bajada_de_2_y_al_dejar_de_completar():
    assert v.corta([v.comparar(rec(test=0.8), rec(campeon=TABM, test=0.77), 0.0)], faltan=10)
    assert v.corta([v.comparar(rec(), rec(estado="plazo"), 0.0)], faltan=10)
    assert v.corta([v.comparar(rec(test=0.8), rec(campeon=TABM, test=0.79), 0.0)], faltan=10) is None


def test_corta_por_recuento_cuando_ya_no_puede_ganar():
    baja = v.comparar(rec(test=0.8), rec(campeon=TABM, test=0.788), 0.0)
    assert v.corta([baja, baja, baja], faltan=2) is not None
    assert v.corta([baja, baja, baja], faltan=3) is None


def test_veredicto_regla_4():
    sube = v.comparar(rec(test=0.8), rec(campeon=TABM, test=0.82), 0.0)
    igual = v.comparar(rec(test=0.8), rec(test=0.8), 0.0)
    baja = v.comparar(rec(test=0.8), rec(campeon=TABM, test=0.788), 0.0)
    assert v.veredicto([sube, igual])["mejora"] is True
    assert v.veredicto([sube, baja])["mejora"] is False
    assert v.veredicto([sube, sube, v.comparar(rec(), rec(estado="plazo"), 0.0)])["mejora"] is False
    assert v.veredicto([igual, igual])["mejora"] is False
    assert v.veredicto([sube, igual])["tabm_campeon_en"] == [sube["nombre"]]


def test_presencia_admite_a_TabM_fuera_por_memoria_solo_si_se_declara():
    """C3‴ (enmienda 9): la guarda de memoria deja fuera a TabM en un estudio grande y lo declara."""
    sin_tabm = rec(motores=("lightgbm", "baseline"))
    assert "TabM" in v.presencia(sin_tabm, c3p=True, con_densa=False)
    declarado = dict(sin_tabm, motores_fuera_por_memoria={TABM: {"es": "no cabe", "estimada_gb": 7.8, "disponible_gb": 6.0}})
    assert v.presencia(declarado, c3p=True, con_densa=False) is None
    otro = dict(sin_tabm, motores_fuera_por_memoria={"lightgbm": {"es": "x"}})
    assert "TabM" in v.presencia(otro, c3p=True, con_densa=False)
