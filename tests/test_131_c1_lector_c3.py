"""131 · C1 — el LECTOR de la regla del C3 (`lector_131_c3.py`), con su CONTROL sobre datos conocidos.

Los datos conocidos son el C5 del 129 (`c5_129_diez.json`, `c5_129_grandes.json`, en el repo): comparado consigo mismo tiene que dar
«bit a bit» y ENTRA; con un caso FABRICADO que baja 3 puntos en las 5 semillas, tiene que dar «empeora». Sin estudios.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
spec = importlib.util.spec_from_file_location("lector_131_c3_en_pruebas", _FASE0 / "lector_131_c3.py")
L = importlib.util.module_from_spec(spec)
_antes = sys.dont_write_bytecode
try:
    spec.loader.exec_module(L)
finally:
    sys.dont_write_bytecode = _antes

C5 = {**json.loads((_FASE0 / "c5_129_diez.json").read_text())["conjuntos"],
      **json.loads((_FASE0 / "c5_129_grandes.json").read_text())["conjuntos"]}
GRANDES = ("KDDCup09_appetency", "APSFailure", "Allstate_Claims_Severity")


def _mundo():
    """El C5 como si lo hubieran medido los dos brazos: es el control (nada cambia)."""
    e = {b: {s: {n: copy.deepcopy(C5[n]) for n in L.GRUPO_E} for s in L.SEMILLAS} for b in ("pub", "c2")}
    s = {n: copy.deepcopy(C5[n]) for n in L.GRUPO_S}
    prov = {f"e_{b}_s{sem}": {"semilla": sem, "modo": "pcpu"} for b in ("pub", "c2") for sem in L.SEMILLAS}
    prov.update({"c3_131_s_c2_diez": {"semilla": None, "modo": "pcpu"},
                 "c3_131_s_c2_grandes": {"semilla": None, "modo": "pcpu"}})
    return e, s, prov


def _bajar(rec, puntos):
    """Baja la cifra de test del registro `puntos` puntos (×100)."""
    from_ = L.V3.METRICA_DE_CIERRE[rec["tarea"]]
    rec["test"]["metricas"][from_] -= puntos / 100


def _aplicar(e, s, prov):
    return L.aplicar(e, s, C5, prov)


# ------------------------------------------------------------------ el control del lector: el C5 contra sí mismo
def test_el_c5_comparado_consigo_mismo_da_bit_a_bit_y_entra():
    r = _aplicar(*_mundo())
    assert r["veredicto"] == "ENTRA", (r["incumple"], r["falta"])
    assert all(f["bit_a_bit"] for f in r["grupo_s"].values()) and len(r["grupo_s"]) == 10
    for f in r["grupo_e"].values():
        assert all(d == 0 for d in f["d"].values()) and f["media_d"] == 0 and f["empeora"] is False
    assert r["control_de_arboles"]["cuadra"] is True
    assert set(r["publicado_semilla0_contra_c5"].values()) == {True}


def test_el_mundo_del_control_cubre_los_13_y_los_tres_de_E():
    assert set(L.GRUPO_E) | set(L.GRUPO_S) == set(C5) and len(set(C5)) == 13 and not set(L.GRUPO_E) & set(L.GRUPO_S)
    assert set(GRANDES) <= set(L.GRUPO_S)


# ------------------------------------------------------------------ el caso fabricado: baja 3 puntos en las 5 semillas
@pytest.mark.parametrize("conjunto", L.GRUPO_E)      # binaria (AUROC) y regresión (R²)
def test_un_caso_que_baja_3_puntos_en_las_5_semillas_EMPEORA(conjunto):
    e, s, prov = _mundo()
    for sem in L.SEMILLAS:
        _bajar(e["c2"][sem][conjunto], 3.0)
    r = _aplicar(e, s, prov)
    f = r["grupo_e"][conjunto]
    assert f["empeora"] is True and f["media_d"] == pytest.approx(-3.0) and f["error_tipico_d"] == pytest.approx(0, abs=1e-9)
    assert r["veredicto"] == "NO ENTRA" and any("EMPEORA" in m and conjunto in m for m in r["incumple"])
    assert [n for n, g in r["grupo_e"].items() if g["empeora"]] == [conjunto]          # los otros dos, no


def _con_d(conjunto, ds):
    e, s, prov = _mundo()
    for sem, d in zip(L.SEMILLAS, ds):
        _bajar(e["c2"][sem][conjunto], -d)                   # d es C2 − publicado: bajar(−d) la mueve a d
    return _aplicar(e, s, prov)["grupo_e"][conjunto]


def test_baja_3_pero_con_mucho_ruido_no_empeora_porque_no_pasa_de_dos_errores_tipicos():
    f = _con_d("pc1", [-10, 4, -8, 0, -1])        # media −3, error típico ≈ 2,7: 2·2,7 > 3
    assert f["media_d"] == pytest.approx(-3) and f["error_tipico_d"] * 2 > 3 and f["empeora"] is False


def test_baja_menos_de_2_puntos_aunque_sea_constante_no_empeora():
    assert _con_d("pc1", [-1.9] * 5)["empeora"] is False
    assert _con_d("pc1", [-2.0] * 5)["empeora"] is False        # «más de 2»: estricto
    assert _con_d("pc1", [-2.1] * 5)["empeora"] is True


def test_sube_nunca_empeora():
    assert _con_d("us_crime", [3, 3, 3, 3, 3])["empeora"] is False


def test_la_desviacion_es_la_muestral_y_si_ddof_cambiaria_el_veredicto_se_dice():
    # media −3; error típico 1,59 con ddof=1 (2·1,59 = 3,18 > 3: NO empeora) y 1,42 con ddof=0 (2,84 < 3: empeora)
    ds = [-3 - 3.55, -3 + 3.55, -3 - 3.55, -3 + 3.55, -3]
    media, et1, emp1 = L.regla_e(ds, ddof=1)
    _, et0, emp0 = L.regla_e(ds, ddof=0)
    assert media == pytest.approx(-3) and emp1 is False and emp0 is True
    e, s, prov = _mundo()
    for sem, d in zip(L.SEMILLAS, ds):
        _bajar(e["c2"][sem]["pc1"], -d)
    r = _aplicar(e, s, prov)
    assert r["grupo_e"]["pc1"]["empeora"] is False and r["grupo_e"]["pc1"]["empeora_ddof0"] is True
    assert any("SENSIBLE a ddof" in a and "pc1" in a for a in r["avisos"])


# ------------------------------------------------------------------ el grupo S: bit a bit, o no entra
def test_un_cambio_de_un_ulp_en_el_test_de_un_conjunto_del_S_no_es_bit_a_bit():
    e, s, prov = _mundo()
    m = L.V3.METRICA_DE_CIERRE[s["wilt"]["tarea"]]
    s["wilt"]["test"]["metricas"][m] += 1e-12
    r = _aplicar(e, s, prov)
    assert r["veredicto"] == "NO ENTRA" and r["grupo_s"]["wilt"]["difiere"] == ["test"]
    assert all(f["bit_a_bit"] for n, f in r["grupo_s"].items() if n != "wilt")


def test_un_cambio_en_la_media_de_la_seleccion_o_en_el_campeon_del_S_tampoco_lo_es():
    e, s, prov = _mundo()
    s["pendigits"]["media_de_la_seleccion"]["motores"][0]["valor"] += 1e-9
    s["house_16H"]["campeon"] = "lightgbm"
    r = _aplicar(e, s, prov)
    assert r["grupo_s"]["pendigits"]["difiere"] == ["media_de_la_seleccion"] and r["grupo_s"]["house_16H"]["difiere"] == ["campeon"]
    assert r["veredicto"] == "NO ENTRA"


def test_un_conjunto_que_deja_de_terminar_es_no_entra_en_E_y_en_S():
    e, s, prov = _mundo()
    s["micro-mass"]["estado"] = "failed"
    assert _aplicar(e, s, prov)["veredicto"] == "NO ENTRA"
    e, s, prov = _mundo()
    e["c2"][3]["us_crime"]["estado"] = "plazo"
    r = _aplicar(e, s, prov)
    assert r["veredicto"] == "NO ENTRA" and any("DEJA DE TERMINAR" in m and "us_crime" in m for m in r["incumple"])


# ------------------------------------------------------------------ el control de árboles
def test_el_control_de_arboles_que_no_cuadra_es_no_entra():
    e, s, prov = _mundo()
    for m in e["c2"][0]["dresses-sales"]["media_de_la_seleccion"]["motores"]:
        if m["motor"] == "lightgbm":
            m["valor"] += 0.0005
    r = _aplicar(e, s, prov)
    assert r["control_de_arboles"]["cuadra"] is False and r["veredicto"] == "NO ENTRA"


def test_el_control_de_arboles_usa_climate_del_grupo_S():
    e, s, prov = _mundo()
    for m in s["climate-model-simulation-crashes"]["media_de_la_seleccion"]["motores"]:
        if m["motor"] == "sklearn.hgb":
            m["valor"] += 0.0005
    assert _aplicar(e, s, prov)["control_de_arboles"]["cuadra"] is False


# ------------------------------------------------------------------ lo que falta y lo que se declara
def test_faltando_una_semilla_es_incompleto_nunca_entra():
    e, s, prov = _mundo()
    del e["c2"][4]
    r = _aplicar(e, s, prov)
    assert r["veredicto"] == "INCOMPLETO" and r["grupo_e"]["pc1"]["empeora"] is None


def test_un_incumplimiento_ya_es_no_entra_aunque_falten_cosas():
    e, s, prov = _mundo()
    del e["c2"][4]
    s["wilt"]["campeon"] = "otro"
    assert _aplicar(e, s, prov)["veredicto"] == "NO ENTRA"


def test_una_procedencia_con_otra_semilla_no_vale_y_deja_incompleto():
    e, s, prov = _mundo()
    prov["e_c2_s2"]["semilla"] = 1
    r = _aplicar(e, s, prov)
    assert r["veredicto"] == "INCOMPLETO" and any("e_c2_s2" in m for m in r["falta"])
    e, s, prov = _mundo()
    prov["c3_131_s_c2_diez"]["semilla"] = 3
    assert _aplicar(e, s, prov)["veredicto"] == "INCOMPLETO"


def test_se_declara_cuantas_veces_cambia_el_campeon_entre_semillas_en_cada_brazo():
    e, s, prov = _mundo()
    base = e["c2"][0]["pc1"]["campeon"]
    e["c2"][1]["pc1"]["campeon"] = e["c2"][2]["pc1"]["campeon"] = "lightgbm" if base != "lightgbm" else "sklearn.hgb"
    r = _aplicar(e, s, prov)["grupo_e"]["pc1"]
    assert r["cambios_de_campeon"] == {"pub": 0, "c2": 2}
    assert L.cambios_de_campeon({0: "a", 1: "a", 2: "b", 3: "c", 4: "a"}) == 2 and L.cambios_de_campeon({1: "a"}) is None


def test_el_publicado_que_no_es_el_c5_se_avisa():
    e, s, prov = _mundo()
    e["pub"][0]["dresses-sales"]["campeon"] = "otro"
    r = _aplicar(e, s, prov)
    assert any("NO es bit a bit el C5" in a for a in r["avisos"])


# ------------------------------------------------------------------ de punta a punta, con ficheros (la ruta de la cola)
def _escribir(directorio, e, s, prov):
    for b in ("pub", "c2"):
        for sem in L.SEMILLAS:
            (directorio / f"c3_131_e_{b}_s{sem}.json").write_text(json.dumps(
                {"procedencia": prov[f"e_{b}_s{sem}"], "conjuntos": e[b][sem]}))
    diez = {n: r for n, r in s.items() if n not in GRANDES}
    (directorio / "c3_131_s_c2_diez.json").write_text(json.dumps({"procedencia": prov["c3_131_s_c2_diez"], "conjuntos": diez}))
    (directorio / "c3_131_s_c2_grandes.json").write_text(json.dumps(
        {"procedencia": prov["c3_131_s_c2_grandes"], "conjuntos": {n: s[n] for n in GRANDES}}))


def _cli(directorio):
    return subprocess.run([sys.executable, "-B", str(_FASE0 / "lector_131_c3.py"), str(directorio)],
                          capture_output=True, text=True)


def test_el_lector_por_la_linea_de_ordenes_el_c5_contra_si_mismo_sale_0_y_dice_bit_a_bit(tmp_path):
    _escribir(tmp_path, *_mundo())
    r = _cli(tmp_path)
    assert r.returncode == 0 and "C3: ENTRA" in r.stdout and r.stdout.count("BIT A BIT") == 10 and "CUADRA" in r.stdout


def test_el_lector_por_la_linea_de_ordenes_el_caso_fabricado_sale_1_y_dice_empeora(tmp_path):
    e, s, prov = _mundo()
    for sem in L.SEMILLAS:
        _bajar(e["c2"][sem]["pc1"], 3.0)
    _escribir(tmp_path, e, s, prov)
    r = _cli(tmp_path)
    assert r.returncode == 1 and "EMPEORA" in r.stdout and "C3: NO ENTRA" in r.stdout


def test_el_lector_por_la_linea_de_ordenes_sin_datos_sale_2(tmp_path):
    r = _cli(tmp_path)
    assert r.returncode == 2 and "C3: INCOMPLETO" in r.stdout


# ------------------------------------------------------------------ bit a bit = lo MEDIDO (lista blanca), no el diccionario entero
_HOY = _FASE0.parents[0] / "fase0" / "c5_129_diez.json"
_REAL = Path(__file__).parent / "fixtures_131_c1" / "control_131_c1_s0.json"


def _real():
    return json.loads(_REAL.read_text())["conjuntos"]


@pytest.mark.parametrize("n", ["dresses-sales", "climate-model-simulation-crashes", "pc1"])
def test_la_medida_real_de_hoy_es_bit_a_bit_el_c5_aunque_declare_fuera_de_la_decision(n):
    rec = _real()[n]
    assert "fuera_de_la_decision" in rec["media_de_la_seleccion"]["motores"][0]        # la declaración nueva SÍ está
    assert rec["media_de_la_seleccion"] != C5[n]["media_de_la_seleccion"]              # y el diccionario entero difiere
    assert L.bit_a_bit(rec, C5[n]) == (True, [])


def test_una_clave_desempate_inyectada_no_rompe_el_bit_a_bit():
    for n, rec in _real().items():
        rec = copy.deepcopy(rec)
        rec["media_de_la_seleccion"]["desempate"] = {"aplicado": True}
        rec["media_de_la_seleccion"]["motores"][0]["desempate"] = {"x": 1}
        rec["test"]["desempate"] = "algo"
        rec["desempate"] = {"campeon": "x"}
        assert L.bit_a_bit(rec, C5[n]) == (True, []), n


@pytest.mark.parametrize("donde", ["media", "test", "campeon", "de"])
def test_un_cambio_en_lo_medido_no_es_bit_a_bit(donde):
    rec = copy.deepcopy(_real()["pc1"])
    if donde == "media":
        rec["media_de_la_seleccion"]["motores"][0]["valor"] += 1e-16 + 2 ** -52         # el último decimal
        esperado = ["media_de_la_seleccion"]
    elif donde == "test":
        rec["test"]["metricas"]["auroc"] = rec["test"]["metricas"]["auroc"] + 2 ** -53 * 4
        esperado = ["test"]
    elif donde == "campeon":
        rec["campeon"] = "otro"; esperado = ["campeon"]
    else:
        rec["media_de_la_seleccion"]["motores"][1]["de"] = 4; esperado = ["media_de_la_seleccion"]
    ok, dif = L.bit_a_bit(rec, C5["pc1"])
    assert not ok and dif == esperado
