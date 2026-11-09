# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C6 — el banco de comportamiento en el EXPEDIENTE (ficha TRIPOD+AI y huecos PROBAST+AI) y lo que viaja en el paquete.

Se construyen los documentos con las MISMAS funciones del núcleo que usa el Studio (`monotonia_en_rejilla`, `sensibilidad_de_una_columna`),
sobre filas con IDENTIFICADORES y VALORES CENTINELA, y se comprueba lo que sale:

* el documento del paquete no contiene ni un centinela ni una clave de fila real (lista blanca, D5);
* sin banco la línea no se imprime y el expediente es el de antes: añadir el banco solo AÑADE líneas (las que llevan su ruta);
* con banco, la línea sale en los dos idiomas, con los límites pegados y sin vocabulario prohibido, y el hueco de lo no calculado
  sale por el mecanismo `_NO_DISPONIBLE` solo cuando el paquete trae el banco.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from matrixai.estudio import comportamiento as cm
from matrixai.estudio.banco_en_el_paquete import (
    CLAVES_DE_FILA,
    NOMBRE_DEL_FICHERO,
    BancoConFilas,
    banco_para_el_paquete,
    exigir_sin_filas,
    partes_sin_calcular,
    texto_del_banco,
)
from matrixai.estudio.comportamiento_textos import palabras_prohibidas
from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.tripod import ficha_tripod

#: Valores y ids que SOLO existen en filas de desarrollo. Si uno aparece en el paquete, viajó una fila.
ID_CENTINELA = "FILA-CENTINELA-{:03d}"
VALORES_CENTINELA = (51.987654, 33.123457, 71.555551)
IDS = [ID_CENTINELA.format(i) for i in range(40)]


def _filas():
    filas = []
    for i in range(40):
        edad = VALORES_CENTINELA[i % 3] + (i // 3) * 0.000013
        filas.append({"edad": edad, "otra": 1.0 + i * 0.37})
    return filas


def _puntuador(columna):
    def puntuar(filas):
        # `edad` sube… salvo un bache entre 50 y 60 (contraejemplo), `otra` no cambia nada.
        salida = []
        for f in filas:
            x = f["edad"]
            salida.append(x / 100.0 - (0.5 if columna == "edad" and 52.0 <= x <= 56.0 else 0.0))
        return salida
    return puntuar


def _resultado(momento="antes", direccion="sube"):
    filas, ids = _filas(), IDS
    escala = cm.escala_de_la_salida("binary_classification")
    doc = cm.monotonia_en_rejilla("edad", direccion, filas, _puntuador("edad"), minimo=20.0, maximo=80.0, puntos=25, filas_m=30,
                                  semilla=149, escala=escala, medido_sobre="muestra_de_desarrollo", ids=ids, motor_de_arboles=True)
    sens = cm.sensibilidad_de_una_columna("edad", filas, _puntuador("edad"), minimo=20.0, maximo=80.0, semilla=149, filas_n=30,
                                          medido_sobre="muestra_de_desarrollo", ids=ids, decidir=lambda y: y > 0.5)
    return {
        "version": 2, "version_del_banco": cm.VERSION_DEL_COMPORTAMIENTO, "job_id": "j", "motor": "lightgbm",
        "tarea": "binary_classification", "modelo": {"digest": "d" * 64, "motor": "lightgbm"},
        "medido_sobre": "muestra_de_desarrollo",
        "muestra": {"origen": "desarrollo", "semilla": 149, "filas": 40, "filas_de_desarrollo": 400},
        "calculado_en": "2026-11-08T10:00:00+00:00",
        "sensibilidad": [sens.a_json()],
        "monotonia": [{"declaracion": {"id": "decl-1", "columna": "edad", "direccion": direccion, "momento": momento,
                                       "declarada_en": "2026-11-08T09:59:00+00:00"},
                       "documento": doc.a_json(),
                       "fila_tipica": {"medido_sobre": "fila_tipica", "de_donde": "valor_de_referencia_de_train", "columna": "edad",
                                       "puntos": 3, "curva": {"x": [20.0, 50.0, 80.0], "y": [0.2, 0.5, 0.8]}}}],
        "no_aplica": [{"columna": "Ciudad", "prueba": "sensibilidad", "tipo": "categorica",
                       "motivo": {"es": "columna no numérica", "en": "non-numeric column"}}],
    }


def _falta():
    return {"version": 1, "version_del_banco": cm.VERSION_DEL_COMPORTAMIENTO, "prueba": "falta", "modelo": {"digest": "d" * 64},
            "metrica": "auroc", "medido_sobre": "validacion_de_cada_pliegue", "pliegues_medidos": 3, "filas_medidas": 300,
            "calculado_en": "2026-11-08T10:05:00+00:00",
            "columnas": [{"columna": "edad", "caida_media": 0.041, "desviacion": 0.01, "pliegues_con_caida": 3,
                          "nunca_falto_al_entrenar": True, "motivo": None, "caidas": [0.04, 0.05, 0.03]},
                         {"columna": "otra", "caida_media": 0.0005, "desviacion": 0.0001, "pliegues_con_caida": 3,
                          "nunca_falto_al_entrenar": False, "motivo": None, "caidas": [0.0, 0.0, 0.0]}]}


def _todo(valor):
    return json.dumps(valor, ensure_ascii=False)


def _claves(valor):
    if isinstance(valor, dict):
        for k, v in valor.items():
            yield k
            yield from _claves(v)
    elif isinstance(valor, list):
        for v in valor:
            yield from _claves(v)


def _paquete(tmp_path: Path, banco=None, *, nombre="p") -> Path:
    ruta = tmp_path / nombre
    ruta.mkdir()
    escribir_manifiesto_de_estudio(ruta, filas=300, motor_del_modelo_entregado="sklearn.hgb", filas_de_desarrollo=240,
                                   filas_de_prueba_reservada=60, filas_que_entrenaron_el_modelo=200,
                                   filas_de_cola_de_parada_temprana=40)
    if banco is not None:
        (ruta / NOMBRE_DEL_FICHERO).write_text(json.dumps(banco, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


# ── El documento que viaja ───────────────────────────────────────────────────────────────────────────────────────────────

def test_la_entrada_de_la_prueba_SI_lleva_filas_reales():
    """Control del instrumento: sin esto, «el paquete no tiene centinelas» podría ser una entrada que nunca los tuvo."""
    crudo = _todo(_resultado())
    assert ID_CENTINELA.format(3) in crudo and "51.987654" in crudo
    assert {"ids", "fila", "mayores_pendientes", "valor_antes", "retrocesos", "para_recalcular"} <= set(_claves(_resultado()))
    assert _resultado()["monotonia"][0]["documento"]["contraejemplos"], "el bache tiene que dar contraejemplos o la prueba no mide nada"


def test_el_banco_del_paquete_no_lleva_ninguna_fila_de_desarrollo():
    banco = banco_para_el_paquete(resultado=_resultado(), falta=_falta())
    texto = _todo(banco)
    assert "FILA-CENTINELA" not in texto
    for v in VALORES_CENTINELA:
        assert str(v) not in texto and f"{v:.4f}" not in texto
    assert not (set(_claves(banco)) & CLAVES_DE_FILA), set(_claves(banco)) & CLAVES_DE_FILA
    assert banco["sin_filas_de_desarrollo"] is True
    # Y lo que debe llevar, lo lleva: el veredicto con sus límites pegados y los tramos sin fila.
    m = banco["monotonia"][0]
    assert m["veredicto"]["clave"] == "con_contraejemplos" and m["veredicto"]["limites"]["es"] and m["veredicto"]["limites"]["en"]
    assert m["tramos_con_retroceso"] and all(set(t) == {"x_a", "x_b", "retroceso"} for t in m["tramos_con_retroceso"])
    assert m["fila_tipica"]["x"] == [20.0, 50.0, 80.0]
    assert "id" not in m["declaracion"] and m["declaracion"]["momento"] == "antes"
    assert banco["sensibilidad"][0]["pruebas"] and all("mayores_pendientes" not in p for p in banco["sensibilidad"][0]["pruebas"])


def test_un_campo_nuevo_del_banco_no_viaja_solo():
    """Lista blanca: una clave que mañana añada el banco a su documento (con una fila dentro) no llega al paquete."""
    r = _resultado()
    r["monotonia"][0]["documento"]["secreto_nuevo"] = "FILA-CENTINELA-999"
    r["monotonia"][0]["documento"]["filas"]["valores"] = [51.987654]
    r["sensibilidad"][0]["otra_clave"] = {"fila": "FILA-CENTINELA-998"}
    assert "CENTINELA" not in _todo(banco_para_el_paquete(resultado=r, falta=None))


def test_sin_resultado_no_hay_banco():
    assert banco_para_el_paquete(resultado=None, falta=None) is None


def test_es_determinista():
    a = banco_para_el_paquete(resultado=_resultado(), falta=_falta())
    assert a == banco_para_el_paquete(resultado=copy.deepcopy(_resultado()), falta=_falta())


@pytest.mark.parametrize("clave", sorted(CLAVES_DE_FILA))
def test_la_segunda_cerradura_se_niega_con_cada_clave_de_fila(clave):
    with pytest.raises(BancoConFilas):
        exigir_sin_filas({"monotonia": [{"x": {clave: 1}}]})


def test_solo_falta_o_solo_comportamiento_dicen_lo_que_no_calcularon():
    solo_b = banco_para_el_paquete(resultado=_resultado(), falta=None)
    assert [p["parte"] for p in solo_b["no_calculado"]] == ["sensibilidad", "falta"]  # el no_aplica de Ciudad + falta
    solo_a = banco_para_el_paquete(resultado=None, falta=_falta())
    partes = {p["parte"] for p in solo_a["no_calculado"]}
    assert "sensibilidad" in partes and "falta" not in partes
    assert solo_a["medido_sobre"] is None


# ── El expediente ────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("idioma", ["es", "en"])
def test_sin_banco_la_linea_no_se_imprime(tmp_path, idioma):
    p = _paquete(tmp_path)
    for texto in (ficha_tripod(p, locale=idioma), huecos_probast(p, locale=idioma), json.dumps(mapa_probast(p, locale=idioma))):
        assert "banco_de_comportamiento" not in texto
        assert "Comportamiento del modelo" not in texto and "Model behaviour" not in texto
    assert "comportamiento" not in {e["clave"] for d in mapa_probast(p, locale=idioma)["por_dominio"].values() for e in d}


@pytest.mark.parametrize("idioma", ["es", "en"])
def test_con_banco_el_expediente_solo_AÑADE_lineas(tmp_path, idioma):
    """El expediente con banco es el de sin banco MÁS las líneas del banco: ni una línea de antes cambia ni desaparece."""
    sin = _paquete(tmp_path, nombre="sin")
    con = _paquete(tmp_path, banco_para_el_paquete(resultado=_resultado(), falta=_falta()), nombre="con")
    for componer in (lambda p: ficha_tripod(p, locale=idioma), lambda p: huecos_probast(p, locale=idioma)):
        antes, despues = componer(sin).splitlines(), componer(con).splitlines()
        anadidas = [l for l in despues if "banco_de_comportamiento.json#" in l]
        assert anadidas, "la línea del banco tiene que salir"
        # (El resumen de PROBAST cuenta cuántas líneas hay MEDIDAS/DECLARADAS/que faltan: con una línea más, esa cuenta sube. Es lo único.)
        resumen = re.compile(r"^- (MEDIDO por el core|DECLARADO por una persona|falta|MEASURED by the core|DECLARED by a person|missing): \d+$")
        assert [l for l in despues if l not in anadidas and not resumen.match(l)] == [l for l in antes if not resumen.match(l)]


@pytest.mark.parametrize("idioma", ["es", "en"])
def test_la_linea_dice_lo_medido_con_los_limites_pegados_y_sin_vocabulario_prohibido(tmp_path, idioma):
    banco = banco_para_el_paquete(resultado=_resultado(), falta=_falta())
    p = _paquete(tmp_path, banco)
    ficha, huecos = ficha_tripod(p, locale=idioma), huecos_probast(p, locale=idioma)
    rotulo = "Comportamiento del modelo" if idioma == "es" else "Model behaviour"
    linea = next(l for l in ficha.splitlines() if l.startswith(f"- **{rotulo}**") and "banco_de_comportamiento.json#" in l)
    assert rotulo in huecos
    resumen = texto_del_banco(banco, idioma)
    assert resumen in linea
    assert banco["monotonia"][0]["veredicto"]["limites"][idioma] in linea, "los límites van PEGADOS a la línea (I4)"
    assert "con contraejemplos" in linea if idioma == "es" else "with counterexamples" in linea
    assert palabras_prohibidas(resumen, idioma, es_la_frase_de_limites=True) == []
    assert "FILA-CENTINELA" not in ficha + huecos


def test_las_dos_redacciones_dicen_lo_mismo_con_las_mismas_cifras():
    banco = banco_para_el_paquete(resultado=_resultado(), falta=_falta())
    es, en = texto_del_banco(banco, "es"), texto_del_banco(banco, "en")
    for cifra in ("40", "149", "0,041"):
        assert cifra in es
    for cifra in ("40", "149", "0.041"):
        assert cifra in en


def test_el_hueco_sale_por_NO_DISPONIBLE_solo_con_banco_y_solo_lo_no_calculado(tmp_path):
    completo = banco_para_el_paquete(resultado=_resultado(), falta=_falta())
    parcial = banco_para_el_paquete(resultado=_resultado(), falta=None)
    rotulo = "Comportamiento del modelo: lo que el banco no calculó"
    assert rotulo not in ficha_tripod(_paquete(tmp_path, nombre="sin"), locale="es")
    ficha_parcial = ficha_tripod(_paquete(tmp_path, parcial, nombre="parcial"), locale="es")
    assert f"**{rotulo}**: si falta una columna: «si falta una columna» no se ha calculado para este modelo" in ficha_parcial
    # Un no_aplica del banco (una columna no numérica) también se dice, con su motivo.
    assert "sensibilidad · `Ciudad`: columna no numérica" in ficha_parcial
    assert partes_sin_calcular(parcial, "en")[-1].startswith("if a column is missing")
    # Con el banco entero sin nada por decir salvo lo que el propio banco declaró no aplicable, no sale la de «falta».
    ficha_completa = ficha_tripod(_paquete(tmp_path, completo, nombre="completo"), locale="es")
    assert "«si falta una columna» no se ha calculado" not in ficha_completa


def test_un_banco_ilegible_se_dice_y_no_rompe(tmp_path):
    p = _paquete(tmp_path)
    (p / NOMBRE_DEL_FICHERO).write_text("{no es json", encoding="utf-8")
    ficha = ficha_tripod(p, locale="es")
    assert "banco_de_comportamiento.json#" in ficha and "no disponible" in ficha
    huecos_probast(p, locale="es")  # tampoco rompe PROBAST


def test_el_inventario_es_una_sola_lista_para_los_dos_informes(tmp_path):
    p = _paquete(tmp_path, banco_para_el_paquete(resultado=_resultado(), falta=_falta()))
    evidencias = [e for d in mapa_probast(p)["por_dominio"].values() for e in d if e["clave"] == "comportamiento"]
    assert len(evidencias) == 1 and evidencias[0]["estado"] == "medido" and evidencias[0]["ruta"] == "banco_de_comportamiento.json#"
