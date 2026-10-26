# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""137 — los hallazgos de la auditoría externa del 26-10 (traída por Roberto), en el NÚCLEO.

Cada prueba nombra el defecto que reproducía antes del arreglo (se escribieron ANTES de
reparar, y se vieron en rojo) y cada una tiene su sabotaje en rojo, anotado en el informe:

1. (P1) La tabla al umbral y el deslizador cuentan IGUAL en las seis combinaciones
   tipo × dirección. Antes, con `probabilidad_pct` y «más bajo = más riesgo», la tabla
   comparaba las puntuaciones negadas en % (−20, −80) con el umbral negado en 0–1 (−0,5).
2. (P1) El intervalo de la calibración recoge la incertidumbre de la prevalencia: remuestreo
   NO estratificado por desenlace. Antes, 30 casos con 15 eventos y probabilidad 0,51 daban
   IC [0,01; 0,01] y «sobrestima el riesgo».
5. El punto «umbral de uso» del deslizador es el DECLARADO (0,5), no el vértice que lo cierra
   (0,8); y si no alcanza a nadie, se dice así.
6. El contador de filas que usa la demo es el MISMO lector del núcleo (separador y comillas).
8. Las claves numéricas no funden valores distintos: 1000000, 1000001 y 1000002 son tres.
9. El HTML descargado lleva la firma COMPLETA, el sobre firmado y el informe que cubre, y dice
   cómo se verifica.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import re
from html import unescape

import pytest

from matrixai.estudio import evaluacion_de_puntuacion as ep
from matrixai.estudio import incertidumbre
from matrixai.estudio.evaluacion_de_puntuacion import (
    DeclaracionDePuntuacion,
    evaluar_puntuacion,
)
from matrixai.estudio.metricas import Muestra
from matrixai.export.informe_de_puntuacion import informe_html
from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.receipt import verificar_recibo

ALTO, BAJO = "mas_alto_mas_riesgo", "mas_bajo_mas_riesgo"
R2 = dict(horizonte_del_desenlace="durante el ingreso", momento_de_la_puntuacion="al ingreso",
          donde_se_calcula="local")


def _decl(columna, tipo, direccion, **kw):
    kw.setdefault("casos_en_su_desarrollo", "no")
    return DeclaracionDePuntuacion(columna, tipo, direccion, **kw)


def _evaluar(texto, decls, *, representa=True, remuestras=40, **kw):
    for k, v in R2.items():
        kw.setdefault(k, v)
    return evaluar_puntuacion(texto, columna_desenlace="y", valor_del_evento="1",
                              puntuaciones=decls, muestra_representativa=representa,
                              remuestras=remuestras, **kw)


def _texto(filas, columnas=("y", "s")):
    return "\n".join([",".join(columnas)] + [",".join(str(c) for c in f) for f in filas]) + "\n"


# ---------------------------------------------------------------------------
# 1 · la tabla al umbral y el deslizador cuentan igual, en TODAS las combinaciones
# ---------------------------------------------------------------------------

#: (tipo, escala de la columna, umbral en las unidades de la columna)
_ESCALAS = {"probabilidad": (1.0, 0.5), "probabilidad_pct": (100.0, 50.0),
            "puntuacion": (20.0, 10.0)}


def _datos_de_combinacion(tipo: str, direccion: str) -> tuple[str, list[float], list[int], float]:
    """30 casos: con «más alto» los eventos están arriba; con «más bajo», abajo. Valores que NO
    caen en el umbral, para que el recuento no dependa de un empate."""
    escala, umbral = _ESCALAS[tipo]
    r = random.Random(f"{tipo}-{direccion}")
    valores, ys = [], []
    for i in range(30):
        y = 1 if i < 15 else 0
        alto = (0.55 + 0.4 * r.random()) if y else (0.05 + 0.4 * r.random())
        if i in (3, 20):  # un evento por debajo y un no evento por encima: la tabla no es trivial
            alto = 1.0 - alto
        frac = alto if direccion == ALTO else 1.0 - alto
        valores.append(round(frac * escala, 3))
        ys.append(y)
    return _texto(list(zip(ys, valores))), valores, ys, umbral


def _esperada(valores, ys, umbral, direccion):
    senal = [(v >= umbral) if direccion == ALTO else (v <= umbral) for v in valores]
    return {"tp": sum(1 for s, y in zip(senal, ys) if s and y),
            "fp": sum(1 for s, y in zip(senal, ys) if s and not y),
            "fn": sum(1 for s, y in zip(senal, ys) if not s and y),
            "tn": sum(1 for s, y in zip(senal, ys) if not s and not y)}


@pytest.mark.parametrize("tipo", ["probabilidad", "probabilidad_pct", "puntuacion"])
@pytest.mark.parametrize("direccion", [ALTO, BAJO])
def test_la_tabla_al_umbral_y_el_deslizador_cuentan_igual_en_cada_tipo_y_direccion(tipo, direccion):
    texto, valores, ys, umbral = _datos_de_combinacion(tipo, direccion)
    res = _evaluar(texto, [_decl("s", tipo, direccion, umbral=umbral)], remuestras=20)
    al = res.informe["puntuaciones"][0]["al_umbral"]
    esperada = _esperada(valores, ys, umbral, direccion)
    assert al["matriz"] == esperada, (tipo, direccion, al["matriz"], esperada)
    uso = [p for p in al["exploracion"]["puntos"] if p["es_el_umbral_de_uso"]]
    assert len(uso) == 1
    assert {c: uso[0][c] for c in ("tp", "fp", "fn", "tn")} == al["matriz"]
    # Y el recuento no es trivial: hay aciertos y errores en las dos filas.
    assert esperada["tp"] and esperada["fn"] and esperada["fp"] and esperada["tn"]


def test_el_caso_de_la_auditoria_pct_mas_bajo_umbral_50_da_15_vp_y_0_fn():
    """Los números de la auditoría: con el defecto, la tabla publicaba 0 VP y 15 FN."""
    filas = [(1, 20.0)] * 15 + [(0, 80.0)] * 15
    res = _evaluar(_texto(filas), [_decl("s", "probabilidad_pct", BAJO, umbral=50)], remuestras=20)
    assert res.informe["puntuaciones"][0]["al_umbral"]["matriz"] == {"tp": 15, "fp": 0, "fn": 0,
                                                                     "tn": 15}


# ---------------------------------------------------------------------------
# 2 · el intervalo de la calibración recoge la incertidumbre de la prevalencia
# ---------------------------------------------------------------------------

def _constante(p: float = 0.51, n: int = 30, eventos: int = 15) -> str:
    return _texto([(1 if i < eventos else 0, p) for i in range(n)])


def test_la_calibracion_en_grande_con_prevalencia_incierta_no_da_un_intervalo_de_un_punto():
    """El caso reproductor: 30 casos, 15 eventos, probabilidad constante 0,51. La diferencia es
    0,01; con el bootstrap estratificado el IC era [0,01; 0,01] y el veredicto «sobrestima»."""
    res = _evaluar(_constante(), [_decl("s", "probabilidad", ALTO)], remuestras=200)
    p = res.informe["puntuaciones"][0]
    g = p["calibracion"]["en_grande"]
    assert g["valor"] == pytest.approx(0.01)
    assert g["ic"] is not None
    assert g["ic"]["bajo"] < 0 < g["ic"]["alto"], g["ic"]
    assert g["ic"]["metodo"] == "iid_paired_bootstrap"
    assert p["veredicto"]["riesgo"] == "sin_diferencia"
    assert "sobrestima" not in p["veredicto"]["texto"]["es"]


def test_el_intercepto_y_la_pendiente_tambien_remuestrean_sin_estratificar():
    r = random.Random(7)
    filas = []
    for _ in range(300):
        p = 0.05 + 0.9 * r.random()
        filas.append((1 if r.random() < p else 0, round(p, 4)))
    res = _evaluar(_texto(filas), [_decl("s", "probabilidad", ALTO)], remuestras=60)
    cal = res.informe["puntuaciones"][0]["calibracion"]
    assert cal["intercepto"]["ic"]["metodo"] == "iid_paired_bootstrap"
    assert cal["pendiente"]["ic"]["metodo"] == "iid_paired_bootstrap"
    # La AP depende de la prevalencia: también.
    ap = res.informe["puntuaciones"][0]["discriminacion"]["average_precision"]
    assert ap["ic"]["metodo"] == "iid_paired_bootstrap"
    # El AUROC, la sensibilidad y la especificidad son condicionales al desenlace: siguen
    # estratificados.
    auroc = res.informe["puntuaciones"][0]["discriminacion"]["auroc"]
    assert auroc["ic"]["metodo"] == "iid_paired_bootstrap_stratified"


def test_vpp_y_vpn_son_los_de_la_muestra_y_su_intervalo_recoge_la_prevalencia():
    """Re-auditoría del 26-10, N1: la prevalencia del VPP/VPN es la OBSERVADA (la persona solo
    declara que la muestra representa); transportarlos con el número de eventos fijo daba un
    intervalo ~25 % más estrecho. Ahora: el VPP/VPN empírico, remuestreado sin estratificar, y la
    nota dice de qué prevalencia son sin afirmar que su incertidumbre no entra."""
    r = random.Random(11)
    filas = []
    for i in range(120):
        y = 1 if i < 12 else 0
        filas.append((y, round(min(0.99, max(0.01, (0.35 if y else 0.0) + 0.6 * r.random())), 3)))
    res = _evaluar(_texto(filas), [_decl("s", "probabilidad", ALTO, umbral=0.5)], remuestras=200)
    u = res.informe["puntuaciones"][0]["al_umbral"]
    m = u["matriz"]
    assert u["vpp"]["valor"] == m["tp"] / (m["tp"] + m["fp"])
    assert u["vpn"]["valor"] == m["tn"] / (m["tn"] + m["fn"])
    assert u["vpp"]["ic"]["metodo"] == "iid_paired_bootstrap"
    assert u["vpn"]["ic"]["metodo"] == "iid_paired_bootstrap"
    # La sensibilidad y la especificidad, condicionales al desenlace, siguen estratificadas.
    assert u["sensibilidad"]["ic"]["metodo"] == "iid_paired_bootstrap_stratified"
    assert u["prevalencia_declarada"] == pytest.approx(12 / 120)
    assert "recoge la incertidumbre de esa prevalencia" in u["nota"]["es"]
    assert "no recoge" not in u["nota"]["es"]
    assert "includes the uncertainty" in u["nota"]["en"]
    # Sin «la muestra representa» no hay VPP/VPN ni nota.
    sin = _evaluar(_texto(filas), [_decl("s", "probabilidad", ALTO, umbral=0.5)],
                   representa=False, remuestras=40)
    u2 = sin.informe["puntuaciones"][0]["al_umbral"]
    assert u2["vpp"]["valor"] is None and u2["nota"] is None


def test_con_columna_de_paciente_la_calibracion_remuestrea_pacientes():
    filas = [(f"P{i // 2}", 1 if i % 4 < 2 else 0, 0.51) for i in range(60)]
    texto = "\n".join(["id,y,s"] + [",".join(map(str, f)) for f in filas]) + "\n"
    res = _evaluar(texto, [_decl("s", "probabilidad", ALTO)], remuestras=60,
                   columna_de_paciente="id")
    g = res.informe["puntuaciones"][0]["calibracion"]["en_grande"]
    assert g["ic"]["metodo"] == "cluster_bootstrap_by_unit"


def test_los_otros_llamantes_de_incertidumbre_siguen_estratificando_por_omision():
    """`incertidumbre` es compartido con el estudio: la opción nueva no cambia a nadie más."""
    m = Muestra.binaria(["e" if i < 15 else "n" for i in range(30)], probabilidades=[0.51] * 30,
                        classes=("n", "e"), positive_label="e")
    por_omision = incertidumbre.medir("calibration_in_the_large", m, diseno="iid",
                                      estimando="fixed_model_on_population", semilla=1,
                                      remuestras=50)
    assert por_omision.uncertainty["method"] == "iid_paired_bootstrap_stratified"
    assert por_omision.uncertainty["ci_low"] == pytest.approx(por_omision.uncertainty["ci_high"])
    sin = incertidumbre.medir("calibration_in_the_large", m, diseno="iid",
                              estimando="fixed_model_on_population", semilla=1, remuestras=50,
                              estratificar=False)
    assert sin.uncertainty["method"] == "iid_paired_bootstrap"
    assert sin.uncertainty["ci_low"] < sin.uncertainty["ci_high"]


# ---------------------------------------------------------------------------
# 5 · el punto «umbral de uso» del deslizador es el declarado
# ---------------------------------------------------------------------------

def _dos_valores() -> str:
    return _texto([(1, 0.8)] * 12 + [(0, 0.2)] * 12 + [(1, 0.2)] * 3 + [(0, 0.8)] * 3)


def test_el_umbral_de_uso_del_deslizador_es_el_declarado_aunque_no_sea_un_vertice():
    res = _evaluar(_dos_valores(), [_decl("s", "probabilidad", ALTO, umbral=0.5)], remuestras=20)
    al = res.informe["puntuaciones"][0]["al_umbral"]
    ex = al["exploracion"]
    uso = [p for p in ex["puntos"] if p["es_el_umbral_de_uso"]]
    assert [p["umbral"] for p in uso] == [0.5]
    assert ex["puntos"][ex["indice_del_umbral_de_uso"]]["umbral"] == 0.5
    assert {c: uso[0][c] for c in ("tp", "fp", "fn", "tn")} == al["matriz"]
    # Ningún vértice real queda marcado como el de uso.
    assert all(not p["es_el_umbral_de_uso"] for p in ex["puntos"] if p["umbral"] in (0.2, 0.8))
    # Y se ofrece en el orden ascendente de siempre.
    assert [p["umbral"] for p in ex["puntos"]] == sorted(p["umbral"] for p in ex["puntos"])


def test_un_umbral_de_uso_que_no_alcanza_a_nadie_se_dice_asi_y_no_desaparece():
    res = _evaluar(_dos_valores(), [_decl("s", "probabilidad", ALTO, umbral=0.9)], remuestras=20)
    ex = res.informe["puntuaciones"][0]["al_umbral"]["exploracion"]
    uso = [p for p in ex["puntos"] if p["es_el_umbral_de_uso"]]
    assert len(uso) == 1 and uso[0]["umbral"] == 0.9
    assert uso[0]["tp"] == 0 and uso[0]["fp"] == 0
    assert ex["indice_del_umbral_de_uso"] is not None
    nota = ex["nota_del_umbral_de_uso"]
    assert nota is not None and "0,9" in nota["es"] and "0.9" in nota["en"]
    assert "ningún" in nota["es"] and "no case" in nota["en"]


def test_con_mas_bajo_el_umbral_declarado_tambien_es_el_punto_de_uso():
    res = _evaluar(_dos_valores(), [_decl("s", "probabilidad", BAJO, umbral=0.5)], remuestras=20)
    al = res.informe["puntuaciones"][0]["al_umbral"]
    uso = [p for p in al["exploracion"]["puntos"] if p["es_el_umbral_de_uso"]]
    assert [p["umbral"] for p in uso] == [0.5]
    assert {c: uso[0][c] for c in ("tp", "fp", "fn", "tn")} == al["matriz"]


def test_un_umbral_que_coincide_con_un_vertice_no_se_duplica():
    res = _evaluar(_dos_valores(), [_decl("s", "probabilidad", ALTO, umbral=0.8)], remuestras=20)
    ex = res.informe["puntuaciones"][0]["al_umbral"]["exploracion"]
    assert [p["umbral"] for p in ex["puntos"]] == [0.2, 0.8]
    assert ex["nota_del_umbral_de_uso"] is None


# ---------------------------------------------------------------------------
# 6 · el contador de filas es el lector del núcleo
# ---------------------------------------------------------------------------

def test_el_contador_de_filas_usa_el_separador_y_las_comillas_del_nucleo():
    texto = 'y;s;nota\n1;0,3;"una nota\ncon salto"\n0;0,1;"otra\ny otra\nmás"\n'
    assert ep.contar_filas_de_datos(texto.encode("utf-8")) == 2
    assert ep.contar_filas_de_datos(texto.replace(";", "\t").encode("utf-8")) == 2
    # Y es lo mismo que el núcleo lee al evaluar.
    res = evaluar_puntuacion(texto, columna_desenlace="y", valor_del_evento="1",
                             puntuaciones=[_decl("s", "puntuacion", ALTO)],
                             muestra_representativa=False, remuestras=20, **R2)
    assert res.informe["entrada"]["filas_leidas"] == 2


def test_el_contador_cuenta_las_lineas_en_blanco_como_el_nucleo_y_el_windows_1252():
    texto = "y,s\n1,2\n\n0,3\n\n"
    assert ep.contar_filas_de_datos(texto.encode()) == 2
    assert ep.contar_filas_de_datos("y;sexo\n1;Sí\n0;No\n".encode("cp1252")) == 2


# ---------------------------------------------------------------------------
# 8 · las claves numéricas no funden valores distintos
# ---------------------------------------------------------------------------

def test_las_claves_numericas_no_funden_valores_distintos():
    claves = {ep._clave_de_texto(t) for t in ("1000000", "1000001", "1000002")}
    assert len(claves) == 3
    largos = {ep._clave_de_texto(t) for t in ("12345678901234567", "12345678901234568")}
    assert len(largos) == 2
    # Más de 28 cifras: `Decimal.normalize()` redondea al contexto (28) y los fundiría
    # (re-auditoría del 26-10, prueba que le faltaba al comentario «sin normalize()»).
    enormes = {ep._clave_de_texto(t) for t in ("1234567890123456789012345678901",
                                                "1234567890123456789012345678902")}
    assert len(enormes) == 2
    # La clave EXACTA, la misma que escribe la pantalla (`claveDeValor` de leerCsv.ts, con su prueba).
    assert [ep._clave_de_texto(t) for t in ("1000000", "0,25", "-0", "-12.50")] == [
        "1e6", "25e-2", "0", "-125e-1"]


@pytest.mark.parametrize("variantes", [("1", "1.0", "1,0", "01", "+1", "1.00", "1e0"),
                                       ("0", "0.0", "0,0", "-0", "0e5"),
                                       ("1000", "1e3", "1.0E3", "1000.0")])
def test_el_mismo_numero_escrito_de_otra_forma_sigue_siendo_uno(variantes):
    assert len({ep._clave_de_texto(t) for t in variantes}) == 1


def test_un_desenlace_1000000_frente_a_1000001_son_dos_clases():
    r = random.Random(3)
    filas = []
    for i in range(80):
        y = i % 2
        filas.append(("1000001" if y else "1000000", round(0.6 * y + 0.4 * r.random(), 3)))
    res = evaluar_puntuacion(_texto(filas), columna_desenlace="y", valor_del_evento="1000001",
                             puntuaciones=[_decl("s", "probabilidad", ALTO)],
                             muestra_representativa=True, remuestras=20, **R2)
    assert res.informe["desenlace"]["eventos"] == 40
    assert res.informe["desenlace"]["no_eventos"] == 40


def test_un_subgrupo_con_codigos_grandes_da_un_grupo_por_codigo():
    r = random.Random(4)
    filas = []
    for i in range(240):
        y = 1 if r.random() < 0.4 else 0
        filas.append((y, round(0.5 * y + 0.5 * r.random(), 3), 1000000 + i % 3))
    texto = "\n".join(["y,s,centro"] + [",".join(map(str, f)) for f in filas]) + "\n"
    res = _evaluar(texto, [_decl("s", "probabilidad", ALTO)], remuestras=20,
                   columnas_de_subgrupo=["centro"])
    grupos = res.informe["puntuaciones"][0]["subgrupos"][0]["grupos"]
    assert sorted(g["valor"] for g in grupos) == ["1000000", "1000001", "1000002"]


# ---------------------------------------------------------------------------
# 9 · el HTML lleva la firma completa, lo firmado y cómo se verifica
# ---------------------------------------------------------------------------

_CLAVE = b"clave-de-la-instalacion"


def _firmado(clave=_CLAVE):
    r = random.Random(9)
    filas = []
    for _ in range(200):
        p = 0.05 + 0.9 * r.random()
        filas.append((1 if r.random() < p else 0, round(p, 4)))
    return _evaluar(_texto(filas), [_decl("s", "probabilidad", ALTO, umbral=0.5)],
                    remuestras=20, clave=clave, creado_en="2026-10-26T12:00:00Z")


def _bloque(html: str) -> dict:
    m = re.search(r'<pre class="datos" id="matrixai-recibo-firmado">(.*?)</pre>', html, re.S)
    assert m, "el HTML no lleva el recuadro de los datos firmados"
    return json.loads(unescape(m.group(1)))


@pytest.mark.parametrize("idioma", ["es", "en"])
def test_el_html_lleva_la_firma_completa_y_el_sobre_firmado_verificable_con_la_clave(idioma):
    res = _firmado()
    html = informe_html(res, idioma=idioma)
    firma = res.sobre["signatures"][0]["sig"]
    # COMPLETA en la tabla del recibo que se lee, no los 16 primeros caracteres (y no solo en el recuadro de datos).
    assert f'<code class="firma">{firma}</code>' in html
    datos = _bloque(html)
    assert datos["envelope"] == res.sobre
    assert verificar_recibo(datos["envelope"], clave=_CLAVE)["ok"] is True
    assert verificar_recibo(datos["envelope"], clave=b"otra")["ok"] is False
    # Lo firmado lleva las cifras principales y la huella del informe completo del núcleo.
    recibo = json.loads(base64.b64decode(datos["envelope"]["payload"]))
    auroc = res.informe["puntuaciones"][0]["discriminacion"]["auroc"]
    firmada = next(m for m in recibo["metrics"] if m["name"] == "auroc")
    assert (firmada["value"], firmada["ci_low"]) == (auroc["valor"], auroc["ic"]["bajo"])
    assert (recibo["output"]["output_digests"]["report"]
            == hashlib.sha256(jcs_bytes(res.informe)).hexdigest())
    # Y no lleva el informe entero: sus umbrales de exploración son valores de la columna.
    assert set(datos) == {"format", "envelope"}
    # Y dice cómo se verifica, en su idioma.
    assert "matrixai receipt verify" in html
    assert ("Cómo se comprueba" if idioma == "es" else "How to check it") in html


def test_un_bloque_manipulado_no_verifica():
    res = _firmado()
    datos = _bloque(informe_html(res, idioma="es"))
    recibo = json.loads(base64.b64decode(datos["envelope"]["payload"]))
    recibo["metrics"][0]["value"] = 0.99
    datos["envelope"]["payload"] = base64.b64encode(jcs_bytes(recibo)).decode()
    assert verificar_recibo(datos["envelope"], clave=_CLAVE)["ok"] is False


def test_el_bloque_no_se_puede_cerrar_desde_un_texto_declarado():
    """Un origen declarado con `</pre>` o `</script>` no puede partir el recuadro de datos."""
    r = random.Random(9)
    filas = [(1 if r.random() < 0.5 else 0, round(r.random(), 3)) for _ in range(120)]
    res = _evaluar(_texto(filas), [_decl("s", "probabilidad", ALTO,
                                         origen="</pre></script><b>x</b>")],
                   remuestras=20, clave=_CLAVE, creado_en="2026-10-26T12:00:00Z")
    html = informe_html(res, idioma="es")
    datos = _bloque(html)
    recibo = json.loads(base64.b64decode(datos["envelope"]["payload"]))
    assert recibo["models"][0]["origin_declared"] == "</pre></script><b>x</b>"
    assert "<b>x</b>" not in html


def test_sin_clave_el_html_no_se_presenta_como_firmado_y_dice_por_que():
    res = _firmado(clave=None)
    html = informe_html(res, idioma="es")
    assert "A0" in html and "sin firmar" in html
    assert "· firmado" not in html
    datos = _bloque(html)
    assert datos["envelope"] is None and datos["receipt"] == res.recibo
