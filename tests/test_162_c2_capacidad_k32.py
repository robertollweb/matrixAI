# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""162-C2 — TabM con k = 32 miembros (cubo PEQUEÑO): el analizador, la regla sellada y el guion de medida (`benchmarks/c162/`). Nada entrena:
todo con salidas FABRICADAS con los 13 nombres reales del cubo pequeño del congelado.

Fija: cada condición (a)–(d) con un caso que la cumple y otro que no; el CUBO (13 pequeños, ningún mediano) y los SATURADOS (los del 160 que
caen en él: mfeat-factors; fuera de la media y del recuento, dentro de «ninguno empeora» y del coste); SIN VEREDICTO si falta un pliegue,
un brazo o un conjunto, si la salida es parcial, si un intento no cuenta, si los brazos están cruzados, si se mezcla la parada por métrica
o si hay un conjunto mediano; el sello; que la regla escrita y los umbrales del código dicen lo mismo; la confirmación (4 pequeños).
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
import analiza_c162_c1 as an1  # noqa: E402
import analiza_c162_c2 as an  # noqa: E402
import medir_c162_c2 as med  # noqa: E402

R, P = an.REFERENCIA, an.PALANCA
CUBO = an.conjuntos_del_cubo()
NOMBRES = list(CUBO)
REGLA = (_RAIZ / "benchmarks" / "c162" / "REGLA_C2.md").read_text(encoding="utf-8")
SATURADOS = list(an.SATURADOS_DEL_CUBO)
NO_SATURADOS = [n for n in NOMBRES if n not in SATURADOS]


def _registro(ds, brazo, pliegue, metrica, wall=10.0, estado="completed", metric_id="auroc"):
    arq = {"k": 8 if brazo == R else 32, "d_block": 256}
    efectivo = {"epocas_ejecutadas": 40, "mejor_epoca": 20, "parado_por_plazo": False, "parado_por_paciencia": True}
    if brazo == P:
        efectivo["capacidad"] = {"pedida": True, "k": 32, "d_block": 256, "k_de_serie": 8, "d_block_de_serie": 256}
    return {"dataset": ds, "motor": brazo, "repeticion": 0, "pliegue": pliegue, "estado": estado, "metrica_de_cierre": metric_id,
            metric_id: metrica, "wall_s": wall, "rss_pico_mb": 400.0 if brazo == R else 600.0, "arquitectura": arq,
            "entrenamiento_efectivo": efectivo,
            "hiperparametros": {"parada_temprana": {"metrica": "validation_loss_del_ensamblado"}},
            "entorno_digest": "e" * 16, "motor_digest": "m" * 16}


def _payload(ref=0.90, pal=0.92, wall_pal=10.0, *, a_medida=None, es_sonda=False):
    conjuntos, resultados = [], []
    for n, info in CUBO.items():
        conjuntos.append({"nombre": n, "tarea": "binary_classification", "sellado": False, "cubo": "pequeno",
                          "metrica_de_cierre": "auroc", "pliegues_esperados": list(info["pliegues"])})
        for f in info["pliegues"]:
            for brazo, base in ((R, ref), (P, pal)):
                m = a_medida(n, brazo, f) if a_medida else None
                m = base + 0.001 * f if m is None else m
                resultados.append(_registro(n, brazo, f, m, wall=10.0 if brazo == R else wall_pal))
    return {"es_sonda": es_sonda, "parcial": False, "conjuntos": conjuntos, "resultados": resultados}


def _veredicto(payload, **kw):
    return an.analizar(payload, comprobar_sellos=False, **kw)


def _ev(n, claras=0, r=0.05, saturados=0, wall=(1.0, 1.0), empeora=()):
    return [{"nombre": f"c{i}", "completo": True, "saturado": i < saturados, "r": r, "mejora_clara": i >= saturados and i - saturados < claras,
             "empeora_de_verdad": i in empeora, "coste_s": {R: wall[0], P: wall[1]}, "coste_ratio": wall[1] / wall[0]} for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------------------------
# 1. EL CUBO Y LOS SATURADOS
# ---------------------------------------------------------------------------------------------------------------------------------
def test_el_cubo_son_los_13_pequenos_del_congelado_y_ningun_mediano():
    assert len(NOMBRES) == 13 == an.N_CONJUNTOS_DEL_CUBO
    todos = a160.conjuntos_y_pliegues_del_congelado()
    assert len(todos) == 23
    assert set(NOMBRES) == {n for n, v in todos.items() if v["cubo"] == "pequeno"}
    for medianos in ("ozone-level-8hr", "Internet-Advertisements", "sick", "jm1", "wine_quality", "wilt", "optdigits", "pendigits"):
        assert medianos not in NOMBRES
    assert med.__doc__ and an.CUBO == "pequeno"


def test_los_saturados_del_cubo_son_los_del_160_que_caen_en_el_y_los_del_criterio_recalculado():
    assert SATURADOS == ["mfeat-factors"]
    assert set(SATURADOS) == set(an1.SATURADOS) & set(NOMBRES) == set(a160.saturados_segun_el_criterio()) & set(NOMBRES)
    assert "mfeat-factors" in REGLA and len(NO_SATURADOS) == 12


def test_los_saturados_no_entran_en_la_media_ni_en_el_recuento_pero_si_en_c_y_d():
    def a_medida(ds, brazo, f):
        if ds in SATURADOS:
            return 0.90 if brazo == R else 0.999
        return 0.90
    r = _veredicto(_payload(a_medida=a_medida))
    g = r["regla"]
    assert g["a"]["n"] == 12 and g["a"]["media_r_no_saturados"] == pytest.approx(0.0, abs=1e-9) and not g["a"]["cumple"]
    assert g["b"]["n"] == 0 and g["b"]["informativo_mejoras_claras_en_saturados"] == SATURADOS
    def empeora(ds, brazo, f):
        return (0.99 if brazo == R else 0.90) if ds == "mfeat-factors" else None
    r = _veredicto(_payload(0.90, 0.95, a_medida=empeora))
    assert r["regla"]["c"]["empeoran"] == ["mfeat-factors"] and r["veredicto"] == "NO CUMPLE"
    assert an.aplicar_la_regla(_ev(13, claras=12, saturados=1, wall=(1.0, 1.0)) )["d"]["razon_de_costes"] == 1.0
    ev = _ev(13, claras=12, saturados=1)
    ev[0]["coste_s"] = {R: 100.0, P: 900.0}   # el saturado pesa en el coste
    assert an.aplicar_la_regla(ev)["d"]["razon_de_costes"] == pytest.approx((900 + 12) / (100 + 12))


# ---------------------------------------------------------------------------------------------------------------------------------
# 2. LA REGLA, condición a condición
# ---------------------------------------------------------------------------------------------------------------------------------
def test_una_mejora_clara_en_todos_cumple():
    r = _veredicto(_payload(0.90, 0.95))
    assert r["veredicto"] == "CUMPLE", r["problemas"]
    assert r["regla"]["a"]["n"] == 12 and r["regla"]["b"]["n"] == 12


def test_sin_diferencia_no_cumple_a_ni_b():
    r = _veredicto(_payload(0.90, 0.90))
    assert r["veredicto"] == "NO CUMPLE" and not r["regla"]["a"]["cumple"] and not r["regla"]["b"]["cumple"]


def test_a_exige_3_por_ciento_exacto_sin_redondear():
    ev = _ev(13, claras=12, saturados=1, r=0.03 - 1e-9)
    assert not an.aplicar_la_regla(ev)["a"]["cumple"]
    assert an.aplicar_la_regla(_ev(13, claras=12, saturados=1, r=0.03))["a"]["cumple"]


def test_b_exige_cuatro_mejoras_claras_entre_los_12_no_saturados():
    assert an.aplicar_la_regla(_ev(13, claras=4, saturados=1))["b"]["cumple"]
    assert an.aplicar_la_regla(_ev(13, claras=4, saturados=1))["cumple"]
    assert not an.aplicar_la_regla(_ev(13, claras=3, saturados=1))["b"]["cumple"]
    assert not an.aplicar_la_regla(_ev(13, claras=3, saturados=1))["cumple"]


def test_c_un_conjunto_que_empeora_de_verdad_mata_el_veredicto():
    assert not an.aplicar_la_regla(_ev(13, claras=12, saturados=1, empeora=(5,)))["cumple"]
    assert an.aplicar_la_regla(_ev(13, claras=12, saturados=1))["cumple"]
    assert not a160.clasificar(-0.02, (-0.05, -0.001))["empeora_de_verdad"]
    assert not a160.clasificar(-0.05, (-0.2, 0.01))["empeora_de_verdad"]
    assert a160.clasificar(-0.05, (-0.2, -0.01))["empeora_de_verdad"]


def test_d_el_coste_pasa_de_la_razon_maxima_no_cumple():
    tope = an.D_COSTE_MAXIMO
    ok = _veredicto(_payload(0.90, 0.95, wall_pal=10.0 * tope))
    assert ok["regla"]["d"]["razon_de_costes"] == pytest.approx(tope) and ok["regla"]["d"]["cumple"] and ok["veredicto"] == "CUMPLE"
    mal = _veredicto(_payload(0.90, 0.95, wall_pal=10.0 * tope + 0.1))
    assert not mal["regla"]["d"]["cumple"] and mal["veredicto"] == "NO CUMPLE"


def test_d_declara_la_memoria_pico_de_cada_brazo_sin_decidir_con_ella():
    d = _veredicto(_payload(0.90, 0.95))["regla"]["d"]
    assert d["rss_pico_mb"] == {R: 400.0, P: 600.0}


# ---------------------------------------------------------------------------------------------------------------------------------
# 3. COMPLETITUD -> SIN VEREDICTO
# ---------------------------------------------------------------------------------------------------------------------------------
def _sin(payload, pred):
    payload["resultados"] = [r for r in payload["resultados"] if not pred(r)]
    return payload


def test_completa_da_veredicto():
    assert _veredicto(_payload())["veredicto"] in ("CUMPLE", "NO CUMPLE")


@pytest.mark.parametrize("quitar", [
    lambda r: r["dataset"] == "kc2" and r["pliegue"] == 2 and r["motor"] == P,
    lambda r: r["dataset"] == "kc2" and r["pliegue"] == 2 and r["motor"] == R,
    lambda r: r["dataset"] == "kc2" and r["motor"] == P,
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


def test_un_intento_fallido_o_sin_metrica_es_un_pliegue_que_falta_no_un_cero():
    p = _payload()
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 1 and r["motor"] == P:
            r["estado"] = "failed"
    r = _veredicto(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("no cuenta como medida" in x for x in r["problemas"])
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
    p = _payload()   # la referencia con la red de k=32
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == R:
            r["arquitectura"]["k"] = 32
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()   # la palanca sin declarar la capacidad
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == P:
            del r["entrenamiento_efectivo"]["capacidad"]
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()   # la palanca con una red de 8
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == P:
            r["arquitectura"]["k"] = 8
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_mezclar_la_parada_por_metrica_del_c1_es_sin_veredicto():
    p = _payload()
    for r in p["resultados"]:
        if r["dataset"] == "kc2" and r["pliegue"] == 0 and r["motor"] == P:
            r["entrenamiento_efectivo"]["parada_por_metrica"] = {"pedida": True, "efectiva": "metrica"}
    r = _veredicto(p)
    assert r["veredicto"] == "SIN VEREDICTO" and any("no se mezcla" in x for x in r["problemas"])


def test_codigo_distinto_entre_intentos_es_sin_veredicto():
    for campo in ("entorno_digest", "motor_digest"):
        p = _payload()
        p["resultados"][5][campo] = "otro"
        assert _veredicto(p)["veredicto"] == "SIN VEREDICTO", campo


def test_un_conjunto_sellado_o_mediano_en_la_medida_es_sin_veredicto():
    p = _payload()
    p["conjuntos"][0]["sellado"] = True
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()
    p["conjuntos"][0]["cubo"] = "mediano"
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"
    p = _payload()   # un mediano del 23 colado en vez de uno del cubo
    p["conjuntos"][0]["nombre"] = "jm1"
    for r in p["resultados"]:
        if r["dataset"] == p["conjuntos"][0]["nombre"] or r["dataset"] == NOMBRES[0]:
            if r["dataset"] == NOMBRES[0]:
                r["dataset"] = "jm1"
    assert _veredicto(p)["veredicto"] == "SIN VEREDICTO"


def test_una_sonda_no_decide_nada():
    p = _payload(es_sonda=True)
    p["conjuntos"] = p["conjuntos"][:1]
    p["resultados"] = [r for r in p["resultados"] if r["dataset"] == p["conjuntos"][0]["nombre"]]
    assert _veredicto(p)["veredicto"] == "SONDA"


# ---------------------------------------------------------------------------------------------------------------------------------
# 4. EL SELLO
# ---------------------------------------------------------------------------------------------------------------------------------
def _con_sellos(p):
    p = copy.deepcopy(p)
    p.update(dependencias_sha256_al_medir=an.dependencias_de_ahora(),
             regla_sha256_al_medir=hashlib.sha256(an.RUTA_DE_LA_REGLA.read_bytes()).hexdigest(),
             analizador_sha256_al_medir=hashlib.sha256(an.RUTA_DEL_ANALIZADOR.read_bytes()).hexdigest())
    return p


def test_la_regla_esta_sellada_y_el_sello_es_el_del_fichero():
    assert an.REGLA_SHA256_SELLADA is not None, "la regla no está sellada"
    assert an.REGLA_SHA256_SELLADA == hashlib.sha256((_RAIZ / "benchmarks/c162/REGLA_C2.md").read_bytes()).hexdigest()


def test_con_los_sellos_buenos_hay_veredicto_y_con_uno_distinto_no():
    p = _con_sellos(_payload())
    assert an.analizar(p)["veredicto"] in ("CUMPLE", "NO CUMPLE")
    for campo in ("regla_sha256_al_medir", "analizador_sha256_al_medir"):
        q = copy.deepcopy(p)
        q[campo] = "0" * 64
        assert an.analizar(q)["veredicto"] == "SIN VEREDICTO", campo
    for etiqueta in ("matrixai/estudio/metricas.py", "benchmarks/c162/analiza_c162_c1.py"):
        q = copy.deepcopy(p)
        q["dependencias_sha256_al_medir"][etiqueta] = "0" * 64
        assert an.analizar(q)["veredicto"] == "SIN VEREDICTO", etiqueta
    q = copy.deepcopy(p)
    del q["dependencias_sha256_al_medir"]
    assert an.analizar(q)["veredicto"] == "SIN VEREDICTO"


def test_sin_sellar_o_con_la_regla_cambiada_ninguna_medida_vale(monkeypatch):
    for sello in (None, "0" * 64):
        monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", sello)
        assert an.analizar(_con_sellos(_payload()))["veredicto"] == "SIN VEREDICTO"


def test_el_guion_de_medida_se_niega_con_la_regla_sin_sellar_o_cambiada(monkeypatch, tmp_path):
    for sello in (None, "0" * 64):
        monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", sello)
        with pytest.raises(SystemExit) as e:
            med.main(["--salida", str(tmp_path / "x.json")])
        assert "sell" in str(e.value).lower()


# ---------------------------------------------------------------------------------------------------------------------------------
# 5. LA REGLA ESCRITA Y LOS UMBRALES DEL CÓDIGO DICEN LO MISMO; LO COMÚN SE IMPORTA
# ---------------------------------------------------------------------------------------------------------------------------------
def test_los_umbrales_del_codigo_son_los_de_la_regla_escrita():
    assert "≥ **+3 %**" in REGLA and an.A_REDUCCION_MEDIA_MIN == 0.03
    assert "≥ **4**" in REGLA and an.B_MEJORAS_CLARAS_MIN == 4
    assert "R < −3 %" in REGLA and an.C_EMPEORA_MAS_DE == 0.03
    assert "≤ **3,5 ×**" in REGLA and an.D_COSTE_MAXIMO == 3.5
    assert "**12 no saturados**" in REGLA and "**13**" in REGLA
    assert "1.000 remuestras, semilla 0" in REGLA and a160.REMUESTRAS_DEL_IC == 1000 and a160.SEMILLA_DEL_IC == 0


def test_la_proporcion_de_b_es_la_del_c1_redondeada_hacia_arriba():
    assert an.B_MEJORAS_CLARAS_MIN == -(-an1.B_MEJORAS_CLARAS_MIN * len(NO_SATURADOS) // 17)


def test_lo_comun_se_importa_del_c1_y_del_160_no_se_copia():
    assert an.a160.reduccion_relativa is a160.reduccion_relativa and an.a160.ic_de_r is a160.ic_de_r
    assert not hasattr(an, "reduccion_relativa") and not hasattr(an, "ic_de_r")
    assert an.an1 is an1 and an.ESTADOS_DE_MEDIDA is an1.ESTADOS_DE_MEDIDA
    assert med.m1.digest_del_motor is not None and med.clave is not None


def test_la_referencia_es_el_denominador_de_r():
    assert a160.reduccion_relativa([0.1, 0.1], [0.05, 0.05]) == pytest.approx(0.5)
    r = _veredicto(_payload(0.90, 0.80))
    assert all(e["r"] < 0 for e in r["conjuntos"]) and r["veredicto"] == "NO CUMPLE"


# ---------------------------------------------------------------------------------------------------------------------------------
# 6. EL GUION DE MEDIDA
# ---------------------------------------------------------------------------------------------------------------------------------
def test_los_dos_brazos_son_el_mismo_motor_con_y_sin_k_32_y_los_dos_paran_por_la_perdida():
    m = med.motores()
    assert tuple(m) == (R, P) == an.BRAZOS
    assert m[R].k_pedido is None and m[P].k_pedido == 32 and m[P].d_block_pedido is None
    assert m[R].parada == m[P].parada == "perdida"        # el C1 es otra palanca y no se mezcla
    assert m[R].nombre == m[P].nombre == "matrixai.dense.tabm_cpu"


def test_la_declaracion_de_cada_brazo_se_comprueba():
    ok_r, ok_p = _registro("kc2", R, 0, 0.9), _registro("kc2", P, 0, 0.9)
    assert med.declaracion_coherente(R, ok_r) == [] and med.declaracion_coherente(P, ok_p) == []
    assert med.declaracion_coherente(R, ok_p) and med.declaracion_coherente(P, ok_r)
    mezcla = _registro("kc2", P, 0, 0.9)
    mezcla["entrenamiento_efectivo"]["parada_por_metrica"] = {"pedida": True}
    assert med.declaracion_coherente(P, mezcla)


def test_el_digest_de_la_cache_cubre_los_dos_analizadores_el_guion_del_c1_el_del_160_y_los_ficheros_del_motor():
    claves = " ".join(med.componentes_del_digest())
    for trozo in ("analiza_c162_c2", "analiza_c162_c1", "medir_c162_c1", "analiza_c160", "densa_tabm", "tabm_plr", "preparacion_tabm"):
        assert trozo in claves, trozo


def test_el_cubo_que_mide_el_guion_es_el_del_analizador():
    assert list(an.conjuntos_del_cubo()) == NOMBRES and len(NOMBRES) == 13


# ---------------------------------------------------------------------------------------------------------------------------------
# 7. LA CONFIRMACIÓN: los pequeños de la del C1
# ---------------------------------------------------------------------------------------------------------------------------------
CONF1 = (_RAIZ / "benchmarks" / "c162" / "CONFIRMACION.md").read_text(encoding="utf-8")
CONF2 = (_RAIZ / "benchmarks" / "c162" / "CONFIRMACION_C2.md").read_text(encoding="utf-8")


def _filas(texto):
    return [l for l in texto.splitlines() if re.match(r"\| (binaria|multiclase|regresión|regresion) \|", l)]


def test_la_confirmacion_son_los_pequenos_de_la_del_c1_y_son_al_menos_3():
    f1, f2 = _filas(CONF1), _filas(CONF2)
    ids2 = [l.split("|")[2].strip() for l in f2]
    assert len(ids2) >= 3 and set(ids2) <= {l.split("|")[2].strip() for l in f1}
    for l in f2:                                             # pequeño = 500–2.000 filas (el cubo del protocolo v2)
        filas = int(l.split("|")[4].split("×")[0].replace(".", "").strip())
        assert 500 <= filas <= 2000, l
    for l in f1:                                             # y los que se descartan no lo son
        if l.split("|")[2].strip() not in ids2:
            assert int(l.split("|")[5].split("×")[0].replace(".", "").strip()) > 2000, l


def test_la_confirmacion_no_incluye_ningun_conjunto_de_la_fase_0_ni_del_160():
    import json
    import pasada_amplia_101_c5 as c5
    import pasada_exploratoria_101_c3 as c3
    fase0 = {int(d.data_id) for d in c5.datasets_de_la_pasada(c3.protocolo_registrado())}
    declarados = {int(x["data_id"]) for x in json.loads(a160.RUTA_DEL_CONGELADO.read_text())["datasets_declarados"]}
    ids = {int(l.split("|")[2].strip()) for l in _filas(CONF2)}
    assert not ids & (fase0 | declarados | {15, 29, 23, 54, 44956, 44957})


def test_la_regla_escalada_de_confirmacion_pide_la_mitad_hacia_arriba():
    assert an.aplicar_la_regla(_ev(4, claras=2), confirmacion=True)["b"]["umbral"] == 2
    assert an.aplicar_la_regla(_ev(4, claras=2), confirmacion=True)["cumple"]
    assert not an.aplicar_la_regla(_ev(4, claras=1), confirmacion=True)["cumple"]
    assert an.aplicar_la_regla(_ev(3, claras=2), confirmacion=True)["b"]["umbral"] == 2
    assert not an.aplicar_la_regla(_ev(4, claras=2), confirmacion=False)["b"]["cumple"]
