# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""163-C2 — la rejilla de 8 configuraciones por validación: el analizador, la regla y el guion de medida (`benchmarks/c163/`). Nada entrena:
todo con salidas FABRICADAS con los 23 nombres reales del congelado.

Fija: que la regla escrita y los umbrales del código dicen lo mismo; cada condición (a)–(d) con un caso que la cumple y otro que no, POR MOTOR
(un motor puede cumplir y el otro no); SIN VEREDICTO si falta un pliegue, un brazo, un conjunto, `tiempo_de_ajuste`, si la salida es parcial, si
los brazos están cruzados, si la rejilla no se aplicó / no trae 8 configuraciones / eligió otra que la de menor pérdida de validación, o si la regla
no está sellada / no es la sellada; la confirmación (escalada); y que el guion monta los cuatro brazos y se niega a medir sin sello.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
for _r in ("c163", "c160", "fase0"):
    sys.path.insert(0, str(_RAIZ / "benchmarks" / _r))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import analiza_c160 as a160  # noqa: E402
import analiza_c163_c2 as an  # noqa: E402
import medir_c163_c2 as med  # noqa: E402

CONGELADO = a160.conjuntos_y_pliegues_del_congelado()
NOMBRES = list(CONGELADO)
REGLA = (_RAIZ / "benchmarks" / "c163" / "REGLA_C2.md").read_text(encoding="utf-8")
LG, HG = "lightgbm", "hgb"


# ---------------------------------------------------------------------------
# 0. SALIDAS FABRICADAS
# ---------------------------------------------------------------------------
def _declaracion_rejilla(elegida=0, perdidas=None, **cambios):
    perdidas = perdidas if perdidas is not None else [0.5 - 0.01 * i for i in range(8)]
    d = {"pedida": True, "aplicada": True, "criterio": "perdida_en_validation_del_pliegue", "mira_la_prueba": False,
         "elegida": elegida if perdidas is not None else 0, "elegida_es_la_de_hoy": elegida == 0,
         "configuraciones": [{"hojas": 7, "tasa": 0.05, "hojas_minimas": 10, "validacion_mejor": v, "arboles_efectivos": 50, "paro": "temprana"}
                             for v in perdidas]}
    d.update(cambios)
    return d


def _registro(ds, brazo, pliegue, metrica, ajuste=1.0, estado="completed", metric_id="auroc", rejilla=None):
    es_rejilla = brazo.endswith("_rejilla")
    pol = {"aplicada": True, "tamano": "pequeno"}
    hip = {"politica_por_tamano": pol}
    if es_rejilla:
        pol["validacion_mejor"] = 0.4
        hip["rejilla_por_validacion"] = rejilla if rejilla is not None else _declaracion_rejilla(elegida=7)
    return {"dataset": ds, "motor": brazo, "repeticion": 0, "pliegue": pliegue, "estado": estado, "metrica_de_cierre": metric_id,
            metric_id: metrica, "wall_s": 10.0, "tiempo_de_ajuste": ajuste, "hiperparametros": hip, "entorno_digest": "e" * 16,
            "motor_digest": "m" * 16}


def _payload(ref=0.90, rej=0.92, ajuste_rej=5.0, *, a_medida=None, es_sonda=False):
    """Una salida completa: 23 conjuntos x 4 brazos. La rejilla de cada motor da `rej` y la referencia `ref` (+0,001 por pliegue)."""
    conjuntos, resultados = [], []
    for n, info in CONGELADO.items():
        conjuntos.append({"nombre": n, "tarea": "binary_classification", "sellado": False, "metrica_de_cierre": "auroc",
                          "pliegues_esperados": list(info["pliegues"])})
        for f in info["pliegues"]:
            for brazo in an.BRAZOS:
                base = rej if brazo.endswith("_rejilla") else ref
                m = a_medida(n, brazo, f) if a_medida else None
                m = base + 0.001 * f if m is None else m
                resultados.append(_registro(n, brazo, f, m, ajuste=ajuste_rej if brazo.endswith("_rejilla") else 1.0))
    return {"es_sonda": es_sonda, "parcial": False, "conjuntos": conjuntos, "resultados": resultados}


def _veredicto(payload, **kw):
    return an.analizar(payload, comprobar_sellos=False, **kw)


def _ev(rs, claras, *, saturados=0, empeora=False, coste=5.0, motor=LG):
    """Evaluados sintéticos de un motor: `rs` y `claras` por conjunto; los `saturados` primeros son saturados."""
    ref, rej = an.REFERENCIA[motor], an.REJILLA[motor]
    out = []
    for i, (r, c) in enumerate(zip(rs, claras)):
        out.append({"nombre": f"c{i}", "motor": motor, "completo": True, "saturado": i < saturados, "r": r, "mejora_clara": c,
                    "empeora_de_verdad": empeora and i == len(rs) - 1, "coste_s": {ref: 1.0, rej: coste}, "coste_ratio": coste})
    return out


# ---------------------------------------------------------------------------
# 1. DECLARACIONES ANTES DE MEDIR
# ---------------------------------------------------------------------------
def test_los_umbrales_del_codigo_son_los_de_la_regla_escrita():
    assert an.A_REDUCCION_MEDIA_MIN == 0.03 and "≥ **+3 %**" in REGLA
    assert an.B_MEJORAS_CLARAS_MIN == 5 and "≥ **5** de los 17" in REGLA
    assert an.C_EMPEORA_MAS_DE == 0.03 and "R < −3 %" in REGLA
    assert an.D_COSTE_MAXIMO == 8.0 and "≤ **8 ×**" in REGLA
    assert an.N_CONFIGURACIONES == 8 and an.N_CONJUNTOS == 23


def test_la_rejilla_escrita_en_la_regla_es_la_del_motor():
    from matrixai_engines.motores.politica_de_arboles import REJILLA_DE_8
    assert len(REJILLA_DE_8) == an.N_CONFIGURACIONES
    for fila in ("| 0 | ×1 | ×1 | ×1 |", "| 1, 2 | ×½, ×2 | ×1 | ×1 |", "| 3, 4 | ×1 | ×½, ×2 | ×1 |", "| 5, 6 | ×1 | ×1 | ×½, ×2 |",
                 "| 7 | ×2 | ×1 | ×2 |"):
        assert fila in REGLA
    assert REJILLA_DE_8 == ((1, 1, 1), (.5, 1, 1), (2, 1, 1), (1, .5, 1), (1, 2, 1), (1, 1, .5), (1, 1, 2), (2, 1, 2))


def test_los_brazos_y_motores_estan_en_el_orden_que_el_analizador_espera():
    assert an.MOTORES == ("lightgbm", "hgb")
    assert an.BRAZOS == ("lightgbm_hoy", "lightgbm_rejilla", "hgb_hoy", "hgb_rejilla")


def test_la_regla_dice_que_cada_motor_se_juzga_por_separado_y_la_confirmacion_es_de_conjuntos_nuevos():
    assert "Cada motor se juzga POR SEPARADO" in REGLA
    assert "CONFIRMACION_C2.md" in REGLA and "no son los de `benchmarks/c160/`" in REGLA


def test_los_saturados_son_los_del_160():
    assert an.SATURADOS == a160.SATURADOS and len(an.SATURADOS) == 6 and set(an.SATURADOS) <= set(NOMBRES)


def test_la_referencia_lleva_la_politica_en_los_tres_tamanos_y_cuatro_brazos_se_montan_bien():
    m = med.motores()
    assert tuple(m) == an.BRAZOS
    for motor in an.MOTORES:
        assert m[an.REFERENCIA[motor]].rejilla_por_validacion is False and m[an.REJILLA[motor]].rejilla_por_validacion is True
        assert m[an.REFERENCIA[motor]].politica.l2 == 1.0 and m[an.REFERENCIA[motor]].politica.tamanos == ("pequeno", "mediano", "grande")


# ---------------------------------------------------------------------------
# 2. LA REGLA, condición a condición
# ---------------------------------------------------------------------------
def _regla(rs, claras, **kw):
    return an.aplicar_la_regla(_ev(rs, claras, **{k: v for k, v in kw.items() if k != "confirmacion"}), confirmacion=kw.get("confirmacion", False))


def test_cumple_con_17_no_saturados_mejorando_y_ninguno_empeora():
    g = _regla([0.05] * 23, [True] * 23, saturados=6)
    assert g["cumple"] and g["a"]["n"] == 17 and g["b"]["n"] == 17


def test_a_la_media_de_los_no_saturados_no_llega_al_3_por_ciento():
    g = _regla([0.029] * 23, [True] * 23, saturados=6)
    assert not g["a"]["cumple"] and not g["cumple"]


def test_los_saturados_no_entran_en_la_media_ni_en_el_recuento():
    # los 6 saturados mejoran mucho y claro; los 17 no: ni (a) ni (b) lo ven
    g = _regla([0.5] * 6 + [0.0] * 17, [True] * 6 + [False] * 17, saturados=6)
    assert g["a"]["media_r_no_saturados"] == 0.0 and g["b"]["n"] == 0
    assert g["b"]["mejoras_claras_saturados_que_no_cuentan"] and not g["cumple"]


def test_b_hacen_falta_5_mejoras_claras_y_4_no_bastan():
    # los 6 primeros son los saturados: solo cuentan los 17 siguientes
    cinco = [False] * 6 + [True] * 5 + [False] * 12
    cuatro = [False] * 6 + [True] * 4 + [False] * 13
    assert _regla([0.05] * 23, cinco, saturados=6)["b"]["cumple"] and not _regla([0.05] * 23, cuatro, saturados=6)["b"]["cumple"]


def test_c_un_conjunto_que_empeora_de_verdad_lo_tumba_aunque_la_media_suba_y_aunque_sea_saturado():
    g = _regla([0.05] * 22 + [-0.2], [True] * 23, saturados=6, empeora=True)
    assert g["c"]["empeoran"] == ["c22"] and not g["cumple"]
    ev = _ev([0.05] * 23, [True] * 23, saturados=6)
    ev[0]["empeora_de_verdad"] = True   # un saturado
    assert not an.aplicar_la_regla(ev)["cumple"]


def test_d_el_coste_8_pasa_y_8_01_no():
    assert _regla([0.05] * 23, [True] * 23, saturados=6, coste=8.0)["d"]["cumple"]
    assert not _regla([0.05] * 23, [True] * 23, saturados=6, coste=8.01)["d"]["cumple"]


def test_un_conjunto_incompleto_no_se_salta_la_regla_no_es_aplicable():
    ev = _ev([0.05] * 23, [True] * 23, saturados=6)
    ev[3]["completo"] = False
    g = an.aplicar_la_regla(ev)
    assert g["aplicable"] is False and not g["cumple"] and g["incompletos"] == ["c3"]


def test_la_confirmacion_pide_la_mitad_hacia_arriba_de_los_no_saturados():
    ev = _ev([0.05] * 5, [True, True, True, False, False], saturados=0)   # 3 de 5 = la mitad hacia arriba
    assert an.aplicar_la_regla(ev, confirmacion=True)["b"]["umbral"] == 3 and an.aplicar_la_regla(ev, confirmacion=True)["cumple"]
    assert not an.aplicar_la_regla(_ev([0.05] * 5, [True, True, False, False, False]), confirmacion=True)["cumple"]


# ---------------------------------------------------------------------------
# 3. EL ANALIZADOR ENTERO, POR MOTOR
# ---------------------------------------------------------------------------
def test_los_dos_motores_que_mejoran_cumplen_cada_uno_por_separado():
    r = _veredicto(_payload(0.80, 0.90))
    assert r["problemas"] == [] and r["veredicto_por_motor"] == {LG: "CUMPLE", HG: "CUMPLE"} and r["veredicto"] == "CUMPLE (lightgbm, hgb)"


def test_un_motor_cumple_y_el_otro_no_se_juzgan_por_separado():
    def a(n, brazo, f):   # HGB: la rejilla es peor que la referencia en todo
        return 0.80 + 0.001 * f if brazo == "hgb_rejilla" else (0.90 + 0.001 * f if brazo == "hgb_hoy" else None)
    r = _veredicto(_payload(0.80, 0.90, a_medida=a))
    assert r["veredicto_por_motor"] == {LG: "CUMPLE", HG: "NO CUMPLE"} and r["veredicto"] == "CUMPLE (lightgbm)"


def test_ninguno_cumple_es_no_cumple():
    r = _veredicto(_payload(0.90, 0.90))
    assert r["veredicto"] == "NO CUMPLE" and set(r["veredicto_por_motor"].values()) == {"NO CUMPLE"}


def test_el_coste_de_un_motor_no_arrastra_al_otro():
    p = _payload(0.80, 0.90)
    for x in p["resultados"]:
        if x["motor"] == "hgb_rejilla":
            x["tiempo_de_ajuste"] = 9.0   # 9x la referencia
    r = _veredicto(p)
    assert r["veredicto_por_motor"] == {LG: "CUMPLE", HG: "NO CUMPLE"}


def test_una_sonda_no_decide_nada():
    r = _veredicto(_payload(es_sonda=True))
    assert r["problemas"] == [] and r["veredicto"] == "SONDA"


# ---------------------------------------------------------------------------
# 4. SIN VEREDICTO
# ---------------------------------------------------------------------------
def _sin_veredicto(payload, texto, **kw):
    r = _veredicto(payload, **kw)
    assert r["veredicto"] == "SIN VEREDICTO", r["veredicto"]
    assert any(texto in p for p in r["problemas"]), r["problemas"][:5]


def test_falta_un_pliegue_de_un_brazo():
    p = _payload()
    p["resultados"].remove(next(x for x in p["resultados"] if x["motor"] == "hgb_rejilla" and x["pliegue"] == 1))
    _sin_veredicto(p, "faltan 1 intentos")


def test_un_conjunto_de_menos():
    p = _payload()
    p["conjuntos"].pop()
    _sin_veredicto(p, "conjuntos")


def test_salida_parcial():
    p = _payload()
    p["parcial"] = True
    _sin_veredicto(p, "PARCIAL")


def test_un_intento_que_no_cuenta_como_medida():
    p = _payload()
    p["resultados"][0]["estado"] = "failed"
    _sin_veredicto(p, "no cuenta como medida")


def test_sin_tiempo_de_ajuste_no_hay_coste_y_no_hay_veredicto():
    p = _payload()
    p["resultados"][0]["tiempo_de_ajuste"] = None
    _sin_veredicto(p, "tiempo_de_ajuste")


def test_registros_repetidos():
    p = _payload()
    p["resultados"].append(dict(p["resultados"][0]))
    _sin_veredicto(p, "más de una vez")


def test_codigo_distinto_en_un_mismo_brazo():
    p = _payload()
    p["resultados"][0]["motor_digest"] = "otro"
    _sin_veredicto(p, "CÓDIGO DISTINTO")


def test_la_referencia_que_declara_una_rejilla_es_un_brazo_cruzado():
    p = _payload()
    r0 = next(x for x in p["resultados"] if x["motor"] == "lightgbm_hoy")
    r0["hiperparametros"]["rejilla_por_validacion"] = _declaracion_rejilla()
    _sin_veredicto(p, "brazos cruzados")


def test_la_referencia_que_declara_validacion_mejor_es_un_brazo_cruzado():
    p = _payload()
    next(x for x in p["resultados"] if x["motor"] == "hgb_hoy")["hiperparametros"]["politica_por_tamano"]["validacion_mejor"] = 0.3
    _sin_veredicto(p, "REFERENCIA y declara `validacion_mejor`")


def test_la_rejilla_sin_declarar_la_rejilla_es_un_brazo_cruzado():
    p = _payload()
    del next(x for x in p["resultados"] if x["motor"] == "hgb_rejilla")["hiperparametros"]["rejilla_por_validacion"]
    _sin_veredicto(p, "no declara `rejilla_por_validacion`")


def test_la_rejilla_que_no_se_aplico_no_mide_lo_declarado():
    p = _payload()
    next(x for x in p["resultados"] if x["motor"] == "lightgbm_rejilla")["hiperparametros"]["rejilla_por_validacion"] = {
        "pedida": True, "aplicada": False, "motivo": "sin_validation_util"}
    _sin_veredicto(p, "NO se aplicó")


def test_la_rejilla_con_menos_de_8_configuraciones():
    p = _payload()
    d = _declaracion_rejilla(elegida=0, perdidas=[0.5] * 7)
    next(x for x in p["resultados"] if x["motor"] == "lightgbm_rejilla")["hiperparametros"]["rejilla_por_validacion"] = d
    _sin_veredicto(p, "7 configuraciones, no 8")


def test_la_elegida_tiene_que_ser_la_de_menor_perdida_de_validacion():
    p = _payload()
    d = _declaracion_rejilla(elegida=3)   # perdidas decrecientes: la mejor es la 7
    next(x for x in p["resultados"] if x["motor"] == "lightgbm_rejilla")["hiperparametros"]["rejilla_por_validacion"] = d
    _sin_veredicto(p, "no es la de menor pérdida de validación")


def test_la_rejilla_que_no_declara_no_mirar_la_prueba():
    p = _payload()
    d = _declaracion_rejilla(elegida=7, mira_la_prueba=True)
    next(x for x in p["resultados"] if x["motor"] == "hgb_rejilla")["hiperparametros"]["rejilla_por_validacion"] = d
    _sin_veredicto(p, "solo por la validación")


def test_un_brazo_sin_la_politica_aplicada():
    p = _payload()
    next(x for x in p["resultados"] if x["motor"] == "hgb_hoy")["hiperparametros"]["politica_por_tamano"]["aplicada"] = False
    _sin_veredicto(p, "política por tamaño aplicada")


def test_un_conjunto_sellado():
    p = _payload()
    p["conjuntos"][0]["sellado"] = True
    _sin_veredicto(p, "SELLADO")


# ---------------------------------------------------------------------------
# 5. LOS SELLOS
# ---------------------------------------------------------------------------
def test_la_regla_esta_sellada_con_el_sha256_del_fichero():
    assert an.REGLA_SHA256_SELLADA == an._sha256(an.RUTA_DE_LA_REGLA), "la regla cambió después de sellarla"


def test_sin_sello_ninguna_medida_vale(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    r = an.analizar(_payload())
    assert any("NO está sellada" in p for p in r["problemas"]) and r["veredicto"] == "SIN VEREDICTO"


def test_la_regla_distinta_de_la_sellada_niega_el_veredicto(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    assert any("no es la sellada" in p for p in an.analizar(_payload())["problemas"])


def test_con_la_regla_sellada_y_las_dependencias_al_medir_el_veredicto_sale(monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", an._sha256(an.RUTA_DE_LA_REGLA))
    p = _payload(0.80, 0.90)
    p["dependencias_sha256_al_medir"] = an.dependencias_de_ahora()
    p["regla_sha256_al_medir"] = an._sha256(an.RUTA_DE_LA_REGLA)
    p["analizador_sha256_al_medir"] = an._sha256(an.RUTA_DEL_ANALIZADOR)
    assert an.analizar(p)["veredicto"] == "CUMPLE (lightgbm, hgb)"
    p["dependencias_sha256_al_medir"]["matrixai/estudio/metricas.py"] = "0" * 64
    assert any("metricas.py" in x for x in an.analizar(p)["problemas"])


# ---------------------------------------------------------------------------
# 6. EL GUION DE MEDIDA
# ---------------------------------------------------------------------------
def test_el_guion_se_niega_a_medir_sin_sello(tmp_path, monkeypatch):
    # Con el sello de verdad, `main` MEDIRÍA (pasó el 13-11: la prueba arrancó una medida real hasta su tope).
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    with pytest.raises(SystemExit, match="NO está sellada"):
        med.main(["--salida", str(tmp_path / "x.json")])


def test_el_guion_se_niega_si_la_regla_no_es_la_sellada(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "0" * 64)
    with pytest.raises(SystemExit, match="no es la sellada"):
        med.main(["--salida", str(tmp_path / "x.json")])


def test_el_digest_del_motor_cubre_el_fichero_del_motor_y_la_politica_donde_vive_la_rejilla():
    fuente = (Path(med.__file__)).read_text(encoding="utf-8")
    assert "politica_de_arboles.py" in fuente and "arbol_sklearn_hgb.py" in fuente
    assert med.digest_del_motor("lightgbm_hoy") == med.digest_del_motor("lightgbm_rejilla") != med.digest_del_motor("hgb_hoy")


def test_el_guion_comprueba_la_declaracion_de_cada_intento_con_la_del_analizador():
    fuente = (Path(med.__file__)).read_text(encoding="utf-8")
    assert "an.problemas_de_la_declaracion(registro)" in fuente


def test_el_guion_usa_el_tiempo_de_ajuste_que_el_analizador_lee():
    fuente = (Path(med.__file__)).read_text(encoding="utf-8")
    assert re.search(r'"tiempo_de_ajuste":\s*recursos\.get\("wall_seconds"\)', fuente)


def test_los_23_del_guion_son_los_del_congelado():
    assert med.conjuntos_del_congelado() == NOMBRES and len(NOMBRES) == 23
