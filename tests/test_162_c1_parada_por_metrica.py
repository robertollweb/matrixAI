# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C1 — TabM parando por la MÉTRICA de cierre: el analizador, la regla sellada y el guion de medida (`benchmarks/c162/`). Nada entrena:
todo con salidas FABRICADAS con los 23 nombres reales del congelado.

Fija: cada condición (a)–(d) con un caso que la cumple y otro que no; los SATURADOS (fuera de la media y del recuento, dentro de «ninguno
empeora» y del coste) y que son los del criterio recalculado; SIN VEREDICTO si falta un pliegue, un brazo, un conjunto, si la salida es
parcial, si un intento no cuenta como medida, si los brazos están cruzados o el código mezclado; el sello (analizador y guion); que
la regla escrita y los umbrales del código dicen lo mismo; la confirmación (escalada) y que sus conjuntos no son los del 160.
"""
from __future__ import annotations

import copy
import hashlib
import re
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
for _r in ("c162", "c160", "fase0"):
    sys.path.insert(0, str(_RAIZ / "benchmarks" / _r))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import analiza_c160 as a160  # noqa: E402
import analiza_c162_c1 as an  # noqa: E402
import medir_c162_c1 as med  # noqa: E402

R, P = an.REFERENCIA, an.PALANCA
CONGELADO = a160.conjuntos_y_pliegues_del_congelado()
NOMBRES = list(CONGELADO)
REGLA = (_RAIZ / "benchmarks" / "c162" / "REGLA_C1.md").read_text(encoding="utf-8")
SATURADOS = list(an.SATURADOS)
NO_SATURADOS = [n for n in NOMBRES if n not in SATURADOS]


# ---------------------------------------------------------------------------
# 0. SALIDAS FABRICADAS
# ---------------------------------------------------------------------------
def _registro(ds, brazo, pliegue, metrica, wall=10.0, estado="completed", cae=None, metric_id="auroc"):
    pedida = {"pedida": True, "efectiva": "perdida" if cae else "metrica", "metrica": metric_id,
              "cae_a_la_perdida_por": cae} if brazo == P else None
    efectivo = {"epocas_ejecutadas": 40, "mejor_epoca": 20 if brazo == R else 23, "parado_por_plazo": False,
                "parado_por_paciencia": True}
    if pedida:
        efectivo["parada_por_metrica"] = pedida
    parada = {"metrica": "validation_loss_del_ensamblado" if (brazo == R or cae) else f"{metric_id}_de_validacion_del_ensamblado"}
    return {"dataset": ds, "motor": brazo, "repeticion": 0, "pliegue": pliegue, "estado": estado, "metrica_de_cierre": metric_id,
            metric_id: metrica, "wall_s": wall, "entrenamiento_efectivo": efectivo, "hiperparametros": {"parada_temprana": parada},
            "entorno_digest": "e" * 16, "motor_digest": "m" * 16}


def _payload(ref=0.90, pal=0.92, wall_pal=10.0, *, a_medida=None, es_sonda=False):
    """Una salida completa de 23 conjuntos. `a_medida(ds, brazo, pliegue) -> métrica` para sobreescribir casos."""
    conjuntos, resultados = [], []
    for n, info in CONGELADO.items():
        conjuntos.append({"nombre": n, "tarea": "binary_classification", "sellado": False, "metrica_de_cierre": "auroc",
                          "pliegues_esperados": list(info["pliegues"])})
        for f in info["pliegues"]:
            for brazo, base in ((R, ref), (P, pal)):
                m = a_medida(n, brazo, f) if a_medida else None
                m = base + 0.001 * f if m is None else m
                resultados.append(_registro(n, brazo, f, m, wall=10.0 if brazo == R else wall_pal))
    return {"es_sonda": es_sonda, "parcial": False, "conjuntos": conjuntos, "resultados": resultados}


def _veredicto(payload, **kw):
    return an.analizar(payload, comprobar_sellos=False, **kw)


# ---------------------------------------------------------------------------
# 1. LA REGLA, condición a condición
# ---------------------------------------------------------------------------
def test_una_mejora_clara_en_todos_cumple():
    r = _veredicto(_payload(0.90, 0.95))  # el error baja de 0,10 a 0,05: R = 50 %
    assert r["veredicto"] == "CUMPLE", r["problemas"]
    assert r["regla"]["a"]["n"] == 17 and r["regla"]["b"]["n"] == 17


def test_sin_diferencia_no_cumple_a_ni_b():
    r = _veredicto(_payload(0.90, 0.90))
    assert r["veredicto"] == "NO CUMPLE"
    assert not r["regla"]["a"]["cumple"] and not r["regla"]["b"]["cumple"]


def test_a_la_media_exige_3_por_ciento_exacto_sin_redondear():
    # error de referencia 0,10 en todos; el de la palanca 0,0970 -> R = 3 % exacto; 0,0971 -> 2,9 %
    justo = _veredicto(_payload(0.90, 0.9030))
    assert justo["regla"]["a"]["media_r_no_saturados"] == pytest.approx(0.03, abs=1e-3)
    ev = [{"nombre": f"c{i}", "completo": True, "saturado": False, "r": 0.03 - (1e-9 if i == 0 else 0.0), "mejora_clara": True,
           "empeora_de_verdad": False, "coste_s": {R: 1.0, P: 1.0}, "coste_ratio": 1.0} for i in range(17)]
    assert not an.aplicar_la_regla(ev)["a"]["cumple"]
    ev[0]["r"] = 0.03
    assert an.aplicar_la_regla(ev)["a"]["cumple"]


def test_b_exige_cinco_mejoras_claras_entre_los_no_saturados():
    def ev(n_claras):
        return [{"nombre": f"c{i}", "completo": True, "saturado": False, "r": 0.05, "mejora_clara": i < n_claras,
                 "empeora_de_verdad": False, "coste_s": {R: 1.0, P: 1.0}, "coste_ratio": 1.0} for i in range(17)]
    assert an.aplicar_la_regla(ev(5))["b"]["cumple"] and an.aplicar_la_regla(ev(5))["cumple"]
    assert not an.aplicar_la_regla(ev(4))["b"]["cumple"] and not an.aplicar_la_regla(ev(4))["cumple"]


def test_c_un_conjunto_que_empeora_de_verdad_mata_el_veredicto_aunque_sea_saturado():
    def a_medida(ds, brazo, f):
        return (0.99 if brazo == R else 0.90) if ds == "sick" else None  # sick (saturado) empeora mucho y en todos los pliegues
    r = _veredicto(_payload(0.90, 0.95, a_medida=a_medida))
    assert "sick" in r["regla"]["c"]["empeoran"] and r["veredicto"] == "NO CUMPLE"


def test_c_empeorar_poco_o_con_ic_que_cruza_cero_no_cuenta():
    # R = -2 % (menos del 3 %): no «empeora de verdad»
    assert not a160.clasificar(-0.02, (-0.05, -0.001))["empeora_de_verdad"]
    # R = -5 % pero el IC95 llega a > 0: tampoco
    assert not a160.clasificar(-0.05, (-0.2, 0.01))["empeora_de_verdad"]
    assert a160.clasificar(-0.05, (-0.2, -0.01))["empeora_de_verdad"]


def test_d_el_coste_pasa_de_1_5_veces_el_de_la_referencia_no_cumple():
    ok = _veredicto(_payload(0.90, 0.95, wall_pal=15.0))   # 1,5 exacto
    assert ok["regla"]["d"]["razon_de_costes"] == pytest.approx(1.5) and ok["regla"]["d"]["cumple"] and ok["veredicto"] == "CUMPLE"
    mal = _veredicto(_payload(0.90, 0.95, wall_pal=15.1))
    assert not mal["regla"]["d"]["cumple"] and mal["veredicto"] == "NO CUMPLE"


def test_d_la_razon_es_de_sumas_y_cuenta_los_saturados():
    def wall(ds, brazo, f):  # solo para el cálculo directo
        return 1.0
    ev = [{"nombre": "x", "completo": True, "saturado": True, "r": 0.0, "mejora_clara": False, "empeora_de_verdad": False,
           "coste_s": {R: 100.0, P: 300.0}, "coste_ratio": 3.0},
          {"nombre": "y", "completo": True, "saturado": False, "r": 0.0, "mejora_clara": False, "empeora_de_verdad": False,
           "coste_s": {R: 100.0, P: 100.0}, "coste_ratio": 1.0}]
    assert an.aplicar_la_regla(ev)["d"]["razon_de_costes"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 2. LOS SATURADOS
# ---------------------------------------------------------------------------
def test_los_saturados_son_los_6_del_160_y_los_del_criterio_recalculado_de_los_artefactos():
    esperados = ["mfeat-factors", "optdigits", "pendigits", "sick", "wilt", "PhishingWebsites"]
    assert sorted(an.SATURADOS) == sorted(esperados) == sorted(a160.SATURADOS)
    assert sorted(a160.saturados_segun_el_criterio()) == sorted(esperados)
    for n in esperados:
        assert n in REGLA, f"la regla escrita no declara el saturado {n}"
    assert len(NO_SATURADOS) == 17


def test_los_saturados_no_entran_en_la_media_ni_en_el_recuento():
    def a_medida(ds, brazo, f):  # los saturados ganan muchísimo, los demás nada
        if ds in SATURADOS:
            return 0.90 if brazo == R else 0.999
        return 0.90
    r = _veredicto(_payload(a_medida=a_medida))
    g = r["regla"]
    assert g["a"]["n"] == 17 and g["a"]["media_r_no_saturados"] == pytest.approx(0.0, abs=1e-9)
    assert not g["a"]["cumple"] and g["b"]["n"] == 0
    assert sorted(g["b"]["informativo_mejoras_claras_en_saturados"]) == sorted(SATURADOS)  # se listan, no cuentan
    assert r["veredicto"] == "NO CUMPLE"


# ---------------------------------------------------------------------------
# 3. COMPLETITUD -> SIN VEREDICTO
# ---------------------------------------------------------------------------
def _sin(payload, pred):
    payload["resultados"] = [r for r in payload["resultados"] if not pred(r)]
    return payload


def test_completa_da_veredicto():
    assert _veredicto(_payload())["veredicto"] in ("CUMPLE", "NO CUMPLE")


@pytest.mark.parametrize("quitar", [
    lambda r: r["dataset"] == "kc2" and r["pliegue"] == 2 and r["motor"] == P,   # un pliegue de la palanca
    lambda r: r["dataset"] == "kc2" and r["pliegue"] == 2 and r["motor"] == R,   # un pliegue de la referencia
    lambda r: r["dataset"] == "kc2" and r["motor"] == P,                          # un brazo entero de un conjunto
])
def test_sin_veredicto_si_falta_un_pliegue_o_un_brazo(quitar):
    r = _veredicto(_sin(_payload(), quitar))
    assert r["veredicto"] == "SIN VEREDICTO" and not r["regla"]["aplicable"]


def test_sin_veredicto_si_falta_un_conjunto():
    p = _payload()
    p["conjuntos"] = [c for c in p["conjuntos"] if c["nombre"] != "kc2"]
    p["resultados"] = [r for r in p["resultados"] if r["dataset"] != "kc2"]
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_sin_veredicto_si_la_salida_es_parcial():
    p = _payload()
    p["parcial"] = True
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_un_intento_fallido_es_un_pliegue_que_falta_no_un_cero():
    p = _payload()
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 1 and r["motor"] == P:
            r["estado"] = "failed"
    r = _veredicto(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("no cuenta como medida" in x for x in r["problemas"])


def test_un_intento_sin_metrica_no_se_cuenta_como_cero():
    p = _payload()
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 1 and r["motor"] == P:
            r["auroc"] = None
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_registros_repetidos_o_de_mas_son_sin_veredicto():
    p = _payload()
    p["resultados"].append(copy.deepcopy(p["resultados"][0]))
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()
    p["resultados"].append(_registro("kc2", R, 9, 0.9))
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_brazos_cruzados_son_sin_veredicto():
    p = _payload()
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == R:
            r["entrenamiento_efectivo"]["parada_por_metrica"] = {"pedida": True, "efectiva": "metrica"}
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == P:
            del r["entrenamiento_efectivo"]["parada_por_metrica"]
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_codigo_distinto_entre_intentos_es_sin_veredicto():
    p = _payload()
    p["resultados"][5]["entorno_digest"] = "otro"
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()
    p["resultados"][5]["motor_digest"] = "otro"
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_un_conjunto_sellado_en_la_medida_es_sin_veredicto():
    p = _payload()
    p["conjuntos"][0]["sellado"] = True
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_la_palanca_que_cae_a_la_perdida_cuenta_pero_se_lista():
    p = _payload()
    for i, r in enumerate(p["resultados"]):
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == P:
            p["resultados"][i] = _registro("kc2", P, 0, 0.93, cae="validacion_con_una_sola_clase")
    r = _veredicto(p)
    assert r["veredicto"] in ("CUMPLE", "NO CUMPLE")
    kc2 = next(e for e in r["conjuntos"] if e["nombre"] == "kc2")
    assert kc2["cae_a_la_perdida"] == [{"pliegue": 0, "motivo": "validacion_con_una_sola_clase"}]


def test_una_sonda_no_decide_nada():
    p = _payload(es_sonda=True)
    p["conjuntos"] = p["conjuntos"][:1]
    p["resultados"] = [r for r in p["resultados"] if r["dataset"] == p["conjuntos"][0]["nombre"]]
    assert _veredicto(p)["veredicto"] == "SONDA"


# ---------------------------------------------------------------------------
# 4. EL SELLO
# ---------------------------------------------------------------------------
def _con_sellos(p):
    p = copy.deepcopy(p)
    s = an.dependencias_de_ahora()
    p.update(dependencias_sha256_al_medir=s, regla_sha256_al_medir=hashlib.sha256(an.RUTA_DE_LA_REGLA.read_bytes()).hexdigest(),
             analizador_sha256_al_medir=hashlib.sha256(an.RUTA_DEL_ANALIZADOR.read_bytes()).hexdigest())
    return p


def test_la_regla_esta_sellada_y_el_sello_es_el_del_fichero():
    assert an.REGLA_SHA256_SELLADA is not None, "la regla no está sellada"
    assert an.REGLA_SHA256_SELLADA == hashlib.sha256((_RAIZ / "benchmarks/c162/REGLA_C1.md").read_bytes()).hexdigest()


def test_con_los_sellos_buenos_hay_veredicto_y_con_uno_distinto_no():
    p = _con_sellos(_payload())
    assert an.analizar(p)["veredicto"] in ("CUMPLE", "NO CUMPLE")
    for campo in ("regla_sha256_al_medir", "analizador_sha256_al_medir"):
        q = copy.deepcopy(p)
        q[campo] = "0" * 64
        assert an.analizar(q)["veredicto"] == "SIN VEREDICTO", campo
    q = copy.deepcopy(p)
    q["dependencias_sha256_al_medir"]["matrixai/estudio/metricas.py"] = "0" * 64
    assert an.analizar(q)["veredicto"] == "SIN VEREDICTO"
    q = copy.deepcopy(p)
    del q["dependencias_sha256_al_medir"]
    assert an.analizar(q)["veredicto"] == "SIN VEREDICTO"


def test_sin_sellar_ninguna_medida_vale(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    assert an.analizar(_con_sellos(_payload()))["veredicto"] == "SIN VEREDICTO"


def test_una_regla_cambiada_despues_de_sellar_no_vale(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    assert an.analizar(_con_sellos(_payload()))["veredicto"] == "SIN VEREDICTO"


def test_el_guion_de_medida_se_niega_con_la_regla_sin_sellar_o_cambiada(monkeypatch, tmp_path):
    for sello in (None, "0" * 64):
        monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", sello)
        with pytest.raises(SystemExit) as e:
            med.main(["--salida", str(tmp_path / "x.json")])
        assert "sell" in str(e.value).lower()


def test_las_dependencias_del_guion_y_del_analizador_llevan_las_mismas_etiquetas():
    assert med.dependencias_sha256() == an.dependencias_de_ahora()


# ---------------------------------------------------------------------------
# 5. LA REGLA ESCRITA Y LOS UMBRALES DEL CÓDIGO DICEN LO MISMO
# ---------------------------------------------------------------------------
def test_los_umbrales_del_codigo_son_los_de_la_regla_escrita():
    assert "≥ **+3 %**" in REGLA and an.A_REDUCCION_MEDIA_MIN == 0.03
    assert "≥ **5**" in REGLA and an.B_MEJORAS_CLARAS_MIN == 5
    assert "R < −3 %" in REGLA and an.C_EMPEORA_MAS_DE == 0.03
    assert "≤ **1,5 ×**" in REGLA and an.D_COSTE_MAXIMO == 1.5
    assert an.N_CONJUNTOS == 23 and "**17 no saturados**" in REGLA
    assert "1.000 remuestras, semilla 0" in REGLA and a160.REMUESTRAS_DEL_IC == 1000 and a160.SEMILLA_DEL_IC == 0


def test_la_regla_usa_las_funciones_del_160_y_no_copias():
    assert an.a160.reduccion_relativa is a160.reduccion_relativa and an.a160.ic_de_r is a160.ic_de_r
    assert not hasattr(an, "reduccion_relativa") and not hasattr(an, "ic_de_r")


def test_la_referencia_es_el_denominador_de_r():
    # Si la palanca no cambia nada R = 0; si el error de la palanca es la mitad, R = +50 %; si es el doble, R = -100 %.
    assert a160.reduccion_relativa([0.1, 0.1], [0.1, 0.1]) == 0.0
    assert a160.reduccion_relativa([0.1, 0.1], [0.05, 0.05]) == pytest.approx(0.5)
    assert a160.reduccion_relativa([0.1, 0.1], [0.2, 0.2]) == pytest.approx(-1.0)
    r = _veredicto(_payload(0.90, 0.80))
    assert all(e["r"] < 0 for e in r["conjuntos"]) and r["veredicto"] == "NO CUMPLE"


# ---------------------------------------------------------------------------
# 6. EL GUION DE MEDIDA
# ---------------------------------------------------------------------------
def test_los_dos_brazos_son_el_mismo_motor_con_y_sin_la_palanca():
    m = med.motores()
    assert tuple(m) == (R, P)
    assert m[R].parada == "perdida" and m[P].parada == "metrica"
    assert m[R].nombre == m[P].nombre == "matrixai.dense.tabm_cpu"
    assert m[R].epocas_fijas is None and m[P].epocas_fijas is None


def test_la_declaracion_de_cada_brazo_se_comprueba():
    ok_r, ok_p = _registro("kc2", R, 0, 0.9), _registro("kc2", P, 0, 0.9)
    assert med.declaracion_de_la_parada_coherente(R, ok_r) == [] and med.declaracion_de_la_parada_coherente(P, ok_p) == []
    assert med.declaracion_de_la_parada_coherente(R, ok_p)       # la referencia con el registro de la palanca
    assert med.declaracion_de_la_parada_coherente(P, ok_r)       # la palanca con el registro de la referencia
    cae = _registro("kc2", P, 0, 0.9, cae="validacion_constante")
    assert med.declaracion_de_la_parada_coherente(P, cae) == []  # cayó a la pérdida y lo dice: coherente


def test_el_digest_de_la_cache_cubre_el_analizador_el_del_160_y_los_ficheros_del_motor():
    claves = " ".join(med.componentes_del_digest())
    for trozo in ("analiza_c162_c1", "analiza_c160", "densa_tabm", "tabm_plr", "preparacion_tabm"):
        assert trozo in claves, trozo


def test_los_23_son_los_del_congelado():
    assert med.conjuntos_del_congelado() == NOMBRES and len(NOMBRES) == 23


# ---------------------------------------------------------------------------
# 7. LA CONFIRMACIÓN
# ---------------------------------------------------------------------------
CONF = (_RAIZ / "benchmarks" / "c162" / "CONFIRMACION.md").read_text(encoding="utf-8")
CONF_160 = (_RAIZ / "benchmarks" / "c160" / "CONFIRMACION.md").read_text(encoding="utf-8")


def _filas_de_la_tabla(texto):
    return [l for l in texto.splitlines() if re.match(r"\| (binaria|multiclase|regresión|regresion) \|", l)]


def test_la_confirmacion_trae_seis_conjuntos_que_no_son_los_del_160():
    f162, f160 = _filas_de_la_tabla(CONF), _filas_de_la_tabla(CONF_160)
    assert len(f162) == len(f160) == 6
    ids162 = {l.split("|")[2].strip() for l in f162}
    ids160 = {l.split("|")[2].strip() for l in f160}
    assert not ids162 & ids160, ids162 & ids160
    shas162 = {l.split("|")[-2].strip() for l in f162}
    assert len(shas162) == 6 and all(re.fullmatch(r"[0-9a-f]{64}", s) for s in shas162)
    tareas = sorted(l.split("|")[1].strip() for l in f162)
    assert tareas == ["binaria", "binaria", "multiclase", "multiclase", "regresión", "regresión"]


def test_la_confirmacion_no_incluye_ningun_conjunto_de_la_fase_0():
    import pasada_amplia_101_c5 as c5
    import pasada_exploratoria_101_c3 as c3
    fase0 = {int(d.data_id) for d in c5.datasets_de_la_pasada(c3.protocolo_registrado())} | {int(x) for x in re.findall(r"\b\d+\b", "")}
    declarados = {int(x["data_id"]) for x in __import__("json").loads(a160.RUTA_DEL_CONGELADO.read_text())["datasets_declarados"]}
    ids = {int(l.split("|")[2].strip()) for l in _filas_de_la_tabla(CONF)}
    assert not ids & (fase0 | declarados)


def test_la_regla_escalada_de_confirmacion_pide_la_mitad_hacia_arriba():
    def ev(n_no_sat, n_claras):
        return [{"nombre": f"c{i}", "completo": True, "saturado": False, "r": 0.05, "mejora_clara": i < n_claras,
                 "empeora_de_verdad": False, "coste_s": {R: 1.0, P: 1.0}, "coste_ratio": 1.0} for i in range(n_no_sat)]
    assert an.aplicar_la_regla(ev(5, 3), confirmacion=True)["b"]["umbral"] == 3
    assert an.aplicar_la_regla(ev(5, 3), confirmacion=True)["cumple"]
    assert not an.aplicar_la_regla(ev(5, 2), confirmacion=True)["cumple"]
    assert an.aplicar_la_regla(ev(6, 3), confirmacion=True)["b"]["umbral"] == 3
    assert not an.aplicar_la_regla(ev(5, 3), confirmacion=False)["b"]["cumple"]   # sin escalar siguen siendo 5
