# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Ola 2 — el expediente y la ficha TRIPOD dicen el MOTOR del modelo entregado EN PALABRAS.

El manifiesto de un estudio acepta `nombre_del_motor_entregado` (`{es, en}`, lo trae quien conoce los nombres: el núcleo no
lleva su lista) y lo graba en `study.delivered_engine_name`. La línea «Motor del modelo entregado» de la ficha TRIPOD y de los
huecos PROBAST dice «<nombre en el idioma> (<id>)»; sin nombre, el id TAL CUAL, como antes (nunca un nombre inventado aquí).
Se afirma el TEXTO de la línea, no solo que exista la clave.
"""
from __future__ import annotations

import json

import pytest

from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio, manifiesto_de_estudio
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.reproduce import ReproduceManifestError, verify_manifest_digest
from matrixai.export.tripod import ficha_tripod

ID = "matrixai.dense.tabm_cpu"
R = "`reproduce.json#study.delivered_engine`"
NOMBRE = {"es": "Red neuronal TabM de MatrixAI", "en": "MatrixAI TabM neural network"}
BASE = dict(filas=360, filas_de_desarrollo=288, filas_de_prueba_reservada=72, filas_que_entrenaron_el_modelo=260,
            motor_del_modelo_entregado=ID)


def _paquete(tmp_path, **kw):
    escribir_manifiesto_de_estudio(tmp_path, **{**BASE, **kw})
    return tmp_path


# --- el manifiesto -------------------------------------------------------------------------------------------------------

def test_el_manifiesto_graba_el_nombre_y_el_digest_lo_cubre():
    con = manifiesto_de_estudio(**BASE, nombre_del_motor_entregado=NOMBRE)
    assert con["study"]["delivered_engine"] == ID
    assert con["study"]["delivered_engine_name"] == NOMBRE
    assert verify_manifest_digest(con)
    sin = manifiesto_de_estudio(**BASE)
    assert con["manifest_sha256"] != sin["manifest_sha256"]


def test_sin_el_campo_el_manifiesto_es_el_de_siempre():
    sin = manifiesto_de_estudio(**BASE)
    assert "delivered_engine_name" not in sin["study"]
    assert "delivered_engine_name" not in json.dumps(sin)
    nada = manifiesto_de_estudio(**BASE, nombre_del_motor_entregado=None)
    assert nada == sin
    # y el resto del manifiesto no se mueve por llevar nombre
    con = manifiesto_de_estudio(**BASE, nombre_del_motor_entregado=NOMBRE)
    sin_nombre = {k: v for k, v in con["study"].items() if k != "delivered_engine_name"}
    assert sin_nombre == sin["study"]


@pytest.mark.parametrize("malo", [
    {"es": "solo castellano"},
    {"en": "english only"},
    {"es": "a", "en": "b", "fr": "c"},
    {"es": "", "en": "b"},
    {"es": "a", "en": "   "},
    {"es": "a", "en": 3},
    "Red neuronal TabM",
    ["a", "b"],
])
def test_un_nombre_mal_formado_se_rechaza(malo):
    with pytest.raises(ReproduceManifestError):
        manifiesto_de_estudio(**BASE, nombre_del_motor_entregado=malo)


def test_un_nombre_sin_su_id_se_rechaza():
    base = {k: v for k, v in BASE.items() if k != "motor_del_modelo_entregado"}
    with pytest.raises(ReproduceManifestError, match="without motor_del_modelo_entregado"):
        manifiesto_de_estudio(**base, nombre_del_motor_entregado=NOMBRE)


# --- la ficha TRIPOD ------------------------------------------------------------------------------------------------------

def _motor(ficha: str, rotulo: str) -> str:
    return next(l for l in ficha.splitlines() if l.startswith(f"- **{rotulo}**"))


def test_la_ficha_dice_el_nombre_y_el_id_en_cada_idioma(tmp_path):
    paquete = _paquete(tmp_path, nombre_del_motor_entregado=NOMBRE)
    es = _motor(ficha_tripod(paquete, locale="es"), "Motor del modelo entregado")
    en = _motor(ficha_tripod(paquete, locale="en"), "Engine of the delivered model")
    assert es == f"- **Motor del modelo entregado**: Red neuronal TabM de MatrixAI ({ID}) · {R}"
    assert en == f"- **Engine of the delivered model**: MatrixAI TabM neural network ({ID}) · {R}"


def test_sin_nombre_la_ficha_dice_el_id_como_antes(tmp_path):
    paquete = _paquete(tmp_path)
    es = _motor(ficha_tripod(paquete, locale="es"), "Motor del modelo entregado")
    en = _motor(ficha_tripod(paquete, locale="en"), "Engine of the delivered model")
    assert es == f"- **Motor del modelo entregado**: {ID} · {R}"
    assert en == f"- **Engine of the delivered model**: {ID} · {R}"


def test_un_estudio_sin_motor_sigue_diciendo_no_disponible_aunque_haya_nombre_suelto(tmp_path):
    # el manifiesto rechaza un nombre sin id; un paquete a mano con el nombre y sin id no inventa el motor
    paquete = _paquete(tmp_path, motor_del_modelo_entregado=None)
    ruta = paquete / "reproduce.json"
    m = json.loads(ruta.read_text(encoding="utf-8"))
    m["study"]["delivered_engine_name"] = NOMBRE
    ruta.write_text(json.dumps(m), encoding="utf-8")
    es = _motor(ficha_tripod(paquete, locale="es"), "Motor del modelo entregado")
    assert "_no disponible_" in es and NOMBRE["es"] not in es


@pytest.mark.parametrize("roto", [{"es": "solo"}, {"es": "a", "en": ""}, "texto", 7, {"es": "a", "en": "b", "x": "c"}])
def test_un_nombre_roto_a_mano_se_ignora_y_sale_el_id(tmp_path, roto):
    paquete = _paquete(tmp_path)
    ruta = paquete / "reproduce.json"
    m = json.loads(ruta.read_text(encoding="utf-8"))
    m["study"]["delivered_engine_name"] = roto
    ruta.write_text(json.dumps(m), encoding="utf-8")
    es = _motor(ficha_tripod(paquete, locale="es"), "Motor del modelo entregado")
    assert es == f"- **Motor del modelo entregado**: {ID} · {R}"


# --- PROBAST: el mismo inventario ----------------------------------------------------------------------------------------

def test_probast_dice_lo_mismo_que_la_ficha(tmp_path):
    paquete = _paquete(tmp_path, nombre_del_motor_entregado=NOMBRE)
    md_es = huecos_probast(paquete, locale="es")
    md_en = huecos_probast(paquete, locale="en")
    assert f"- **Motor del modelo entregado**: Red neuronal TabM de MatrixAI ({ID}) · MEDIDO por el core" in md_es
    assert f"- **Engine of the delivered model**: MatrixAI TabM neural network ({ID}) · MEASURED by the core" in md_en
    mapa = mapa_probast(paquete, locale="es")
    todas = {c["clave"]: c for dom in mapa["por_dominio"].values() for c in dom}
    assert todas["motor_del_modelo_entregado"]["estado"] == "medido"
    assert todas["motor_del_modelo_entregado"]["valor"] == {"id": ID, "nombre": NOMBRE}


def test_probast_sin_nombre_es_identico_a_antes(tmp_path):
    paquete = _paquete(tmp_path)
    mapa = mapa_probast(paquete, locale="es")
    todas = {c["clave"]: c for dom in mapa["por_dominio"].values() for c in dom}
    assert todas["motor_del_modelo_entregado"]["valor"] == ID
    assert f"- **Motor del modelo entregado**: {ID} · MEDIDO por el core" in huecos_probast(paquete, locale="es")


def test_lo_demas_de_la_ficha_no_se_mueve_por_llevar_nombre(tmp_path):
    for sub in ("con", "sin"):
        (tmp_path / sub).mkdir()
    con = ficha_tripod(_paquete(tmp_path / "con", nombre_del_motor_entregado=NOMBRE), locale="es")
    sin = ficha_tripod(_paquete(tmp_path / "sin"), locale="es")
    quitar = lambda t: [l for l in t.splitlines() if "Motor del modelo entregado" not in l]  # noqa: E731
    assert con != sin
    assert quitar(con) == quitar(sin)
