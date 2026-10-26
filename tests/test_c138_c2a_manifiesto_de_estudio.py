# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""138-C2a — el `reproduce.json` de un ESTUDIO.

Lo que se prueba no es la función suelta, sino que **las tres lectoras del núcleo
(`ficha_tripod`, `mapa_probast`/`huecos_probast`, `ml_bom`) producen su informe** sobre un
paquete con este manifiesto —con las cifras del estudio dentro— y dicen «no disponible» donde el
estudio no sabe. Antes de este corte contestaban «no reproduce.json» con cualquier estudio real.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from matrixai.export.bom import BomNoDisponible, ml_bom
from matrixai.export.manifiesto_de_estudio import (
    MOTIVO_NO_REPRODUCIBLE,
    escribir_manifiesto_de_estudio,
    manifiesto_de_estudio,
)
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.reproduce import ReproduceManifestError, verify_manifest_digest
from matrixai.export.tripod import FichaNoDisponible, ficha_tripod

HUELLA = "46a6b5fb901ade9afb61dcd93451e05ddbf2d09f2b7d7643a32111922d5e1b9e"

LO_QUE_SABE = dict(
    filas=200, huella_del_dataset=HUELLA,
    tipos_de_campo={"x1": "numerica", "x2": "numerica", "categoria": "categorica"},
    columnas_excluidas=["row_id"], semilla_del_reparto=7,
    metricas=[
        {"name": "auroc", "value": 0.7506265664160401, "split": "test", "n_observations": 40},
        {"name": "accuracy", "value": 0.675, "split": "test", "n_observations": 40},
        {"name": "auroc", "value": 0.85982653837611, "split": "validation", "n_observations": 160},
    ],
)


def _paquete(tmp_path: Path, **kwargs) -> Path:
    escribir_manifiesto_de_estudio(tmp_path, **kwargs)
    return tmp_path


def test_un_estudio_nunca_se_declara_reproducible_y_dice_por_que_en_los_dos_idiomas():
    m = manifiesto_de_estudio(**LO_QUE_SABE)
    assert m["reproducible"] is False
    assert m["reproducible_reason_i18n"] == MOTIVO_NO_REPRODUCIBLE
    assert set(MOTIVO_NO_REPRODUCIBLE) == {"es", "en"}
    assert all(len(v) > 40 for v in MOTIVO_NO_REPRODUCIBLE.values())
    # El camino clásico (CLI, BOM, bundle) lee `reproducible_reason` como texto en inglés.
    assert m["reproducible_reason"] == MOTIVO_NO_REPRODUCIBLE["en"]
    assert "certification" in m["claim"]


def test_el_manifiesto_trae_su_digest_y_este_cuadra():
    m = manifiesto_de_estudio(**LO_QUE_SABE)
    assert verify_manifest_digest(m)
    otro = manifiesto_de_estudio(**{**LO_QUE_SABE, "filas": 201})
    assert otro["manifest_sha256"] != m["manifest_sha256"]
    m["metrics"][0]["value"] = 0.99
    assert not verify_manifest_digest(m)


def test_lo_que_el_estudio_no_sabe_se_omite_y_no_se_inventa():
    """Sin ningún dato: ni un cero, ni un modo, ni un sha, ni una semilla, ni un `.mxai`."""
    m = manifiesto_de_estudio()
    assert m["artifacts"] == {"model": None, "training": None, "recipe": None, "dataset": None}
    assert m["generation"] == {}
    assert m["metrics"] == []
    assert m["problem"] is None
    # Las claves de `generation` que sólo el camino clásico sabe NO aparecen ni a `null`.
    sabidos = manifiesto_de_estudio(**LO_QUE_SABE)["generation"]
    assert set(sabidos) == {"field_types", "excluded_identifiers", "seeds"}
    assert sabidos["seeds"] == {"split": 7}


def test_el_conjunto_de_claves_es_el_del_esquema_y_no_lleva_campos_de_mas():
    m = manifiesto_de_estudio(**LO_QUE_SABE)
    assert set(m) == {
        "schema_version", "claim", "reproducible", "reproducible_reason",
        "reproducible_reason_i18n", "artifacts", "problem", "generation", "environment",
        "metrics", "study", "manifest_canonicalization", "manifest_sha256"}
    assert m["artifacts"]["dataset"] == {"sha256": HUELLA, "rows": 200}
    # El bloque `study` va SIEMPRE (su presencia dice «esto es un estudio»), y vacío si no se le dice nada.
    assert m["study"] == {}


def test_una_huella_solo_sin_filas_y_unas_filas_sin_huella_se_publican_tal_cual():
    solo_filas = manifiesto_de_estudio(filas=12)["artifacts"]["dataset"]
    assert solo_filas == {"rows": 12}
    solo_huella = manifiesto_de_estudio(huella_del_dataset=HUELLA)["artifacts"]["dataset"]
    assert solo_huella == {"sha256": HUELLA}


def test_las_tres_funciones_del_nucleo_producen_su_informe_con_las_cifras_del_estudio(tmp_path):
    paquete = _paquete(tmp_path, **LO_QUE_SABE)

    ficha = ficha_tripod(paquete, locale="es")
    assert "0.7506265664160401" in ficha and "0.85982653837611" in ficha
    assert HUELLA in ficha and "200" in ficha
    assert "x1, x2, categoria" in ficha and "row_id" in ficha and '"split": 7' in ficha

    mapa = mapa_probast(paquete, locale="es")
    assert mapa["schema"] == "matrixai.export.probast"
    assert mapa["juicio_de_riesgo"] is None
    medidas = {c["clave"] for dom in mapa["por_dominio"].values() for c in dom
               if c.get("estado") == "medido"}
    assert {"filas", "huella_del_dataset", "predictores", "columnas_excluidas", "metricas"} <= medidas
    assert "Huecos PROBAST" in huecos_probast(paquete, locale="es")

    bom = ml_bom(paquete)
    assert bom["bomFormat"] == "CycloneDX"
    metricas = bom["metadata"]["component"]["modelCard"]["quantitativeAnalysis"]["performanceMetrics"]
    assert {(x["type"], x["slice"], x["value"]) for x in metricas} == {
        ("auroc", "test", "0.7506265664160401"), ("accuracy", "test", "0.675"),
        ("auroc", "validation", "0.85982653837611")}
    datos = next(c for c in bom["components"] if c["type"] == "data")
    props = {p["name"]: p["value"] for p in datos["data"][0]["contents"]["properties"]}
    assert props["matrixai:dataset.sha256"] == HUELLA and props["matrixai:dataset.rows"] == "200"
    # Y NO afirma lo que el estudio no sabe: ni "synthetic", ni un modelo con hash.
    assert "matrixai:dataset.origin" not in props
    assert "hashes" not in bom["metadata"]["component"]
    assert any("does not declare itself reproducible" in x
               for x in bom["metadata"]["component"]["modelCard"]["considerations"]["technicalLimitations"])


def test_donde_el_estudio_no_sabe_las_tres_dicen_no_disponible_y_no_una_cifra(tmp_path):
    paquete = _paquete(tmp_path)        # un estudio que no registró nada de lo opcional
    ficha = ficha_tripod(paquete, locale="es")
    # En un estudio la cifra de `rows` son las filas del CONJUNTO, no las que entrenaron (auditoría de 138-C1..C3).
    assert "**Filas del conjunto de datos (desarrollo y prueba reservada)**: _no disponible_" in ficha
    assert "Filas que entrenaron**" not in ficha
    assert ("**Filas que entrenaron el modelo entregado**: _no disponible_ "
            "(el estudio no guardó cómo repartió sus filas)") in ficha
    assert "**Modo del estudio**: _no disponible_" in ficha
    assert "**Huella del dataset (sha256)**: _no disponible_" in ficha
    assert "**Predictores (entradas)**: _no disponible_" in ficha
    assert "**Partición y semillas**: _no disponible_" in ficha
    assert "_no disponible_ (el paquete no publica ninguna métrica)" in ficha
    assert "None" not in ficha
    bom = ml_bom(paquete)
    nombres = {p["name"] for p in bom["metadata"]["component"]["properties"]}
    assert "matrixai:generation.seeds" not in nombres and "matrixai:epochs.ran" not in nombres
    assert not any(c["type"] == "data" for c in bom["components"])
    mapa = mapa_probast(paquete, locale="es")
    faltan = {c["clave"] for dom in mapa["por_dominio"].values() for c in dom
              if c.get("estado") == "falta"}
    assert {"filas", "huella_del_dataset", "predictores", "metricas"} <= faltan


def test_la_ficha_enseña_el_motivo_de_no_reproducible_en_el_idioma_pedido(tmp_path):
    paquete = _paquete(tmp_path, **LO_QUE_SABE)
    es = ficha_tripod(paquete, locale="es")
    en = ficha_tripod(paquete, locale="en")
    assert MOTIVO_NO_REPRODUCIBLE["es"] in es and MOTIVO_NO_REPRODUCIBLE["en"] not in es
    assert MOTIVO_NO_REPRODUCIBLE["en"] in en and MOTIVO_NO_REPRODUCIBLE["es"] not in en


def test_la_ficha_distingue_la_cifra_de_la_seleccion_de_la_de_la_prueba_reservada(tmp_path):
    ficha = ficha_tripod(_paquete(tmp_path, **LO_QUE_SABE), locale="es")
    assert "**auroc**: 0.7506265664160401 [test]" in ficha
    assert "**auroc**: 0.85982653837611 [validation]" in ficha
    assert "**accuracy**: 0.675 [test]" in ficha


def test_un_manifiesto_clasico_sin_motivo_bilingue_sigue_enseñando_su_reproducible_reason(tmp_path):
    (tmp_path / "reproduce.json").write_text(json.dumps({
        "reproducible": False, "reproducible_reason": "falta la receta", "metrics": [
            {"name": "accuracy", "value": 0.9, "split": "validation"}]}), encoding="utf-8")
    ficha = ficha_tripod(tmp_path, locale="es")
    assert "NO reproducible: falta la receta" in ficha
    # Una sola métrica por nombre: sin etiqueta de partición, como siempre.
    assert "**accuracy**: 0.9 · `reproduce.json#metrics`" in ficha


def test_la_ficha_no_dice_que_predice_sin_instalar_nada_si_el_paquete_no_lleva_el_modelo_onnx(tmp_path):
    """El paquete de un estudio no lleva `model.onnx` ni `inference_spec.json`; la frase fija decía que sí."""
    paquete = _paquete(tmp_path, **LO_QUE_SABE)
    ficha = ficha_tripod(paquete, locale="es")
    assert "El paquete incluye `model.onnx`" not in ficha
    assert "NO incluye `model.onnx`, `predict.py`, `inference_spec.json`" in ficha
    for nombre in ("model.onnx", "predict.py", "inference_spec.json"):
        (paquete / nombre).write_text("x", encoding="utf-8")
    con_todo = ficha_tripod(paquete, locale="es")
    assert "El paquete incluye `model.onnx`, `predict.py` e `inference_spec.json`" in con_todo
    assert "NO incluye" not in con_todo


@pytest.mark.parametrize("malo, quien", [
    (dict(huella_del_dataset="data_49219efba673b8d0"), "huella"),
    (dict(huella_del_dataset="A" * 63), "huella"),
    (dict(filas=0), "filas"),
    (dict(filas=True), "filas"),
    (dict(filas=-3), "filas"),
    (dict(semilla_del_reparto="7"), "semilla"),
    (dict(semilla_del_reparto=1.5), "semilla"),
    (dict(tipos_de_campo={"x": ""}), "tipo"),
    (dict(columnas_excluidas="row_id"), "excluidas"),
    (dict(metricas=[{"name": "auroc", "value": 0.5}]), "sin partición"),
    (dict(metricas=[{"name": "auroc", "value": 0.5, "split": "full"}]), "partición fuera del estudio"),
    (dict(metricas=[{"name": "auroc", "value": "muy alta", "split": "test"}]), "valor no numérico"),
    (dict(metricas=[{"name": "auroc", "value": None, "split": "test"}]), "valor ausente"),
    (dict(metricas=[{"name": "a", "value": 1.0, "split": "test"},
                    {"name": "a", "value": 0.5, "split": "test"}]), "dos valores para la misma métrica"),
])
def test_un_dato_imposible_se_rechaza_en_vez_de_publicarse(malo, quien):
    with pytest.raises((ReproduceManifestError, ValueError)):
        manifiesto_de_estudio(**malo)


# ── Reparación de la auditoría de 138-C1..C3: filas, reparto y modo ────────────────────────────────────────────────

REPARTO = dict(filas_de_desarrollo=160, filas_de_prueba_reservada=40, filas_que_entrenaron_el_modelo=68,
               modo="rapido", motores_fuera_por_el_modo=["matrixai.dense.torch_cpu", "matrixai.dense.tabm_cpu"])


def test_el_bloque_study_lleva_el_reparto_y_el_modo_tal_cual_se_le_dan():
    m = manifiesto_de_estudio(**LO_QUE_SABE, **REPARTO)
    assert m["study"] == {
        "mode": "rapido",
        "engines_left_out_by_mode": ["matrixai.dense.tabm_cpu", "matrixai.dense.torch_cpu"],
        "rows": {"development": 160, "holdout_test": 40, "trained_the_delivered_model": 68}}
    # `rows` del conjunto NO cambia: sigue siendo la cifra de la huella.
    assert m["artifacts"]["dataset"] == {"sha256": HUELLA, "rows": 200}
    assert verify_manifest_digest(m)
    # Un completo no deja a nadie fuera por el modo: la clave no aparece (ni a `[]`).
    completo = manifiesto_de_estudio(modo="completo", motores_fuera_por_el_modo=[])
    assert completo["study"] == {"mode": "completo"}


def test_la_ficha_de_un_estudio_dice_cuantas_filas_entrenaron_de_verdad_y_cuantas_se_reservaron(tmp_path):
    """EL DEFECTO: «Filas que entrenaron: 200» con 40 de ellas en la prueba reservada que el modelo nunca vio."""
    paquete = _paquete(tmp_path, **LO_QUE_SABE, **REPARTO)
    es = ficha_tripod(paquete, locale="es")
    assert "Filas que entrenaron**: 200" not in es
    assert ("**Filas del conjunto de datos (desarrollo y prueba reservada)**: 200 "
            "· `reproduce.json#artifacts.dataset.rows`") in es
    assert "**Filas de desarrollo (validación cruzada y selección)**: 160 · `reproduce.json#study.rows.development`" in es
    assert ("**Filas de la prueba reservada (apartadas antes; ni eligen ni entrenan)**: 40 "
            "· `reproduce.json#study.rows.holdout_test`") in es
    assert ("**Filas que entrenaron el modelo entregado**: 68 "
            "· `reproduce.json#study.rows.trained_the_delivered_model`") in es
    en = ficha_tripod(paquete, locale="en")
    assert "**Dataset rows (development and holdout test)**: 200" in en
    assert "**Rows that trained the delivered model**: 68" in en
    assert "Rows that trained**: 200" not in en


def test_probast_dice_lo_mismo_que_la_ficha_de_las_filas_de_un_estudio(tmp_path):
    paquete = _paquete(tmp_path, **LO_QUE_SABE, **REPARTO)
    huecos = huecos_probast(paquete, locale="es")
    assert "Filas que entrenaron**: 200" not in huecos
    assert "**Filas del conjunto de datos (desarrollo y prueba reservada)**: 200 · MEDIDO por el core" in huecos
    assert "**Filas que entrenaron el modelo entregado**: 68 · MEDIDO por el core" in huecos
    assert "**Filas de la prueba reservada (apartadas antes; ni eligen ni entrenan)**: 40" in huecos
    mapa = mapa_probast(paquete, locale="en")
    por_clave = {c["clave"]: c for dom in mapa["por_dominio"].values() for c in dom}
    assert por_clave["filas"]["rotulo"] == "Dataset rows (development and holdout test)"
    assert por_clave["filas_que_entrenaron_el_modelo"]["valor"] == 68
    assert por_clave["filas_de_prueba_reservada"]["valor"] == 40


def test_la_ficha_dice_el_modo_del_estudio_y_quien_no_compitio_por_el(tmp_path):
    paquete = _paquete(tmp_path, **LO_QUE_SABE, **REPARTO)
    es = ficha_tripod(paquete, locale="es")
    assert "**Modo del estudio**: rápido · `reproduce.json#study.mode`" in es
    assert ("**Motores que no compitieron por el modo (no se midieron; no es que puntuaran mal)**: "
            "matrixai.dense.tabm_cpu, matrixai.dense.torch_cpu") in es
    en = ficha_tripod(paquete, locale="en")
    assert "**Study mode**: quick · `reproduce.json#study.mode`" in en
    assert "**Engines that did not compete because of the mode" in en
    assert "rápido" not in en
    huecos = huecos_probast(paquete, locale="es")
    # El modo lo ELIGE quien lanza el estudio y el estudio lo registra; los motores fuera son lo que ese modo deja
    # fuera («no se midieron»): ninguno de los dos es MEDIDO (re-auditoría de 138, 26-10).
    assert "**Modo del estudio**: rápido · DECLARADO por una persona" in huecos
    assert ("**Motores que no compitieron por el modo (no se midieron; no es que puntuaran mal)**: "
            "matrixai.dense.tabm_cpu, matrixai.dense.torch_cpu · DECLARADO por una persona") in huecos
    assert "rápido · MEDIDO" not in huecos and "torch_cpu · MEDIDO" not in huecos
    huecos_en = huecos_probast(paquete, locale="en")
    assert "**Study mode**: quick · DECLARED by a person" in huecos_en
    assert "torch_cpu · DECLARED by a person" in huecos_en
    por_clave = {c["clave"]: c for dom in mapa_probast(paquete, locale="es")["por_dominio"].values() for c in dom}
    assert por_clave["modo_del_estudio"]["estado"] == "declarado"
    assert por_clave["motores_fuera_por_el_modo"]["estado"] == "declarado"
    # Y lo que sí mide el core sigue MEDIDO: no se cambió de más.
    assert por_clave["filas_que_entrenaron_el_modelo"]["estado"] == "medido"
    # El COMPLETO no deja a nadie fuera: no se escribe una línea de «falta» que sería un hueco inventado.
    otro = tmp_path / "completo"
    otro.mkdir()
    completo = ficha_tripod(_paquete(otro, **LO_QUE_SABE, modo="completo"), locale="es")
    assert "**Modo del estudio**: completo" in completo
    assert "no compitieron por el modo" not in completo
    assert "no compitieron por el modo" not in huecos_probast(otro, locale="es")


def test_la_fuente_de_un_estudio_no_habla_del_modo_del_generador_junto_al_modo_del_estudio(tmp_path):
    """«Fuente: no disponible (no declara con qué modo se generó)» encima de «Modo del estudio: rápido» se leía como una
    contradicción: en un estudio lo que falta es de dónde salió el CSV."""
    paquete = _paquete(tmp_path, **LO_QUE_SABE, **REPARTO)
    for texto in (ficha_tripod(paquete, locale="es"), huecos_probast(paquete, locale="es")):
        assert "con qué modo se generó" not in texto
        assert "el estudio trabaja con el CSV de quien lo hizo, y el paquete no declara de dónde salió" in texto
    assert "the study works on the CSV of whoever ran it" in ficha_tripod(paquete, locale="en")


def test_un_manifiesto_clasico_sigue_diciendo_filas_que_entrenaron_y_sin_reparto_ni_modo(tmp_path):
    """El camino de `matrixai export` no cambia: sin bloque `study`, la ficha de siempre."""
    (tmp_path / "reproduce.json").write_text(json.dumps({
        "reproducible": True, "artifacts": {"dataset": {"rows": 300}}, "metrics": []}), encoding="utf-8")
    ficha = ficha_tripod(tmp_path, locale="es")
    assert "**Filas que entrenaron**: 300 · `reproduce.json#artifacts.dataset.rows`" in ficha
    assert "Filas del conjunto de datos" not in ficha and "Modo del estudio" not in ficha
    assert "el paquete no declara con qué modo se generó" in ficha
    huecos = huecos_probast(tmp_path, locale="es")
    assert "**Filas que entrenaron**: 300 · MEDIDO por el core" in huecos
    assert "prueba reservada" not in huecos and "Modo del estudio" not in huecos


def test_la_ficha_no_afirma_una_orden_que_no_se_ejecuto(tmp_path):
    """La compone el núcleo por los dos caminos (CLI y Studio): nombrar `matrixai report --tripod` era falso en el Studio."""
    for locale in ("es", "en"):
        ficha = ficha_tripod(_paquete(tmp_path, **LO_QUE_SABE), locale=locale)
        assert "matrixai report" not in ficha
        assert ("La compone el núcleo de MatrixAI" if locale == "es" else "Composed by the MatrixAI core") in ficha


@pytest.mark.parametrize("malo", [
    dict(filas_de_desarrollo=0), dict(filas_de_prueba_reservada=True), dict(filas_que_entrenaron_el_modelo=-1),
    dict(filas_de_desarrollo=10, filas_que_entrenaron_el_modelo=11),
    dict(filas=100, filas_de_desarrollo=101), dict(filas=100, filas_de_prueba_reservada=101),
    dict(modo="turbo"), dict(modo=""), dict(motores_fuera_por_el_modo=["x"]),
    dict(modo="rapido", motores_fuera_por_el_modo="matrixai.dense.torch_cpu"),
])
def test_un_reparto_o_un_modo_imposible_se_rechaza(malo):
    with pytest.raises((ReproduceManifestError, ValueError)):
        manifiesto_de_estudio(**malo)


def test_un_paquete_sin_manifiesto_sigue_diciendo_que_no_hay(tmp_path):
    """El comportamiento de antes: sin `reproduce.json` las funciones se niegan."""
    with pytest.raises(FichaNoDisponible):
        ficha_tripod(tmp_path)
    with pytest.raises(BomNoDisponible):
        ml_bom(tmp_path)
