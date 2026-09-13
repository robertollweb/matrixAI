# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — UNA categórica por encima de `_ONEHOT_MAX` cambia el trato de TODAS.

EL BLOQUEANTE, reproducido con 14 filas sintéticas y cero valores faltantes:
un CSV con una categórica de 13 valores (> `_ONEHOT_MAX`) junto a otra de 2
mataba `generate_project_from_dataset` con

    El CSV preparado no pasa la validación del modelo que acaba de generarse.
    Faltan: cat_baja

porque el enrutado manda el prompt ENTERO al generador composite en cuanto UNA
categórica pasa el umbral, y ese generador materializa como EMBEDDING **todas**
las declaradas — también las de 2 valores —, mientras `_prepare_training_csv`
volvía a decidirlo columna a columna por cardinalidad y expandía la pequeña a
one-hot. Dos sitios declarando lo mismo.

Se verifica por el camino REAL (el producto: `generate_project_from_dataset` y
`_validate_training_csv`), y por LOS DOS LADOS — un one-hot legítimo tiene que
seguir expandiéndose; el arreglo no puede ser "no expandir nunca".
"""
from __future__ import annotations

import csv
import io

import pytest

from matrixai.playground import _validate_training_csv
from matrixai.training.categorical import embedding_source_columns
from matrixai.training.dataset_project import (
    _sha256_text,
    generate_project_from_dataset,
    prepare_dataset_from_provenance,
)
from matrixai.training.dense_generator import _ONEHOT_MAX


def _csv(n_alta: int, filas: int = 14) -> str:
    """CSV mínimo: `cat_baja` con 2 valores, `cat_alta` con `n_alta`."""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["cat_baja", "cat_alta", "target"])
    w.writeheader()
    for i in range(filas):
        w.writerow({
            "cat_baja": ["a", "b"][i % 2],
            "cat_alta": f"v{i % n_alta}",
            "target": ["si", "no"][i % 2],
        })
    return out.getvalue()


def _csv_solo_alta(n_alta: int, filas: int = 14) -> str:
    """El caso que SÍ funcionaba antes del arreglo: una única categórica, por
    encima del umbral — el modelo la hace EMBEDDING y el CSV escribía ya el
    índice."""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["cat_alta", "target"])
    w.writeheader()
    for i in range(filas):
        w.writerow({"cat_alta": f"v{i % n_alta}", "target": ["si", "no"][i % 2]})
    return out.getvalue()


def _cabecera(res: dict) -> list[str]:
    return res["csv_text"].splitlines()[0].split(",")


def _valida_contra_su_modelo(res: dict) -> None:
    v = _validate_training_csv(
        res["mxai"], res["training_text"], res["csv_text"],
        field_ranges=res.get("field_ranges"),
    )
    assert v.get("ok"), v.get("errors") or v.get("error")


class TestCategoricaMixta:
    def test_baja_y_alta_juntas_generan_un_proyecto_que_valida(self):
        """EL BLOQUEANTE. Antes: DatasetProjectError 'Faltan: cat_baja'."""
        res = generate_project_from_dataset(_csv(_ONEHOT_MAX + 1), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)

    def test_la_pequena_viaja_como_indice_cuando_el_modelo_la_hizo_embedding(self):
        res = generate_project_from_dataset(_csv(_ONEHOT_MAX + 1), target_column="target")
        # El modelo REAL decide: ambas son fuente de EMBEDDING.
        assert embedding_source_columns(res["mxai"]) == {"cat_baja", "cat_alta"}
        cabecera = _cabecera(res)
        # Mitad positiva: la columna cruda está...
        assert "cat_baja" in cabecera
        # ...y mitad negativa: ninguna one-hot suya.
        assert not [c for c in cabecera if c.startswith("cat_baja__")]

    def test_el_indice_escrito_es_la_posicion_en_el_vocabulario(self):
        """Un CSV con la cabecera correcta y las celdas vacías también pasaría
        los asertos de arriba: aquí se mira el CONTENIDO."""
        res = generate_project_from_dataset(_csv(_ONEHOT_MAX + 1), target_column="target")
        filas = list(csv.DictReader(io.StringIO(res["csv_text"])))
        assert len(filas) == 14
        vocab = res["provenance"]["preparation_spec"]["category_vocabularies"]["cat_baja"]
        assert vocab == ["a", "b"]
        # La fila 0 del crudo es "a" (índice 0) y la 1 es "b" (índice 1).
        assert filas[0]["cat_baja"] == "0"
        assert filas[1]["cat_baja"] == "1"


class TestOneHotLegitimoSigueExpandiendo:
    """EL SESGO CONTRARIO. Si el arreglo fuese 'no expandir nunca', el one-hot
    de verdad —ninguna categórica por encima del umbral— dejaría de funcionar."""

    def test_sin_ninguna_alta_las_categoricas_se_expanden_a_one_hot(self):
        res = generate_project_from_dataset(_csv(3), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)
        assert embedding_source_columns(res["mxai"]) == set()
        cabecera = _cabecera(res)
        assert "cat_baja__a" in cabecera and "cat_baja__b" in cabecera
        assert "cat_baja" not in cabecera
        assert "expand_categoricals_onehot" in res["provenance"]["operations"]

    def test_la_columna_one_hot_marcada_es_la_del_valor_de_la_fila(self):
        res = generate_project_from_dataset(_csv(3), target_column="target")
        filas = list(csv.DictReader(io.StringIO(res["csv_text"])))
        assert filas[0]["cat_baja__a"] == "1" and filas[0]["cat_baja__b"] == "0"
        assert filas[1]["cat_baja__a"] == "0" and filas[1]["cat_baja__b"] == "1"


class TestRecetaCongelaLaDecision:
    def test_la_receta_registra_las_columnas_de_embedding(self):
        res = generate_project_from_dataset(_csv(_ONEHOT_MAX + 1), target_column="target")
        spec = res["provenance"]["preparation_spec"]
        assert sorted(spec["embedding_columns"]) == ["cat_alta", "cat_baja"]

    def test_una_receta_sin_alta_registra_la_lista_vacia_no_la_ausencia(self):
        """VACÍA ('ninguna fue a embedding') y AUSENTE ('no se sabe') no son lo
        mismo: `_prepare_v1` elige criterio según eso."""
        res = generate_project_from_dataset(_csv(3), target_column="target")
        spec = res["provenance"]["preparation_spec"]
        assert "embedding_columns" in spec
        assert spec["embedding_columns"] == []

    def test_repreparar_reproduce_el_mismo_csv_byte_a_byte(self):
        raw = _csv(_ONEHOT_MAX + 1)
        res = generate_project_from_dataset(raw, target_column="target")
        re_prep = prepare_dataset_from_provenance(raw, res["provenance"])
        assert re_prep.csv_text == res["csv_text"]
        assert _sha256_text(re_prep.csv_text) == \
            res["provenance"]["prepared_csv_sha256"]

    def test_una_receta_ANTERIOR_al_arreglo_sigue_reproduciendo_su_CSV(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. `_prepare_v1` pasa
        `None` —no `set()`— cuando la receta no trae `embedding_columns`.

        Un modelo guardado ANTES de este arreglo y con una sola categórica por
        encima del umbral SÍ se generó bien (el caso mixto era el que moría), y
        su CSV lleva esa columna como ÍNDICE. Su receta no tiene la clave
        nueva: forzar ahí `set()` la volvería one-hot y
        `prepare_dataset_from_provenance` abortaría por reproducibilidad — el
        modelo guardado dejaría de poder reimportar su dataset."""
        raw = _csv_solo_alta(_ONEHOT_MAX + 1)
        res = generate_project_from_dataset(raw, target_column="target")
        assert res["csv_text"].splitlines()[0].split(",")[0] == "cat_alta"  # índice
        antigua = dict(res["provenance"])
        spec = dict(antigua["preparation_spec"])
        spec.pop("embedding_columns")  # tal y como la escribía el código anterior
        antigua["preparation_spec"] = spec
        re_prep = prepare_dataset_from_provenance(raw, antigua)
        assert re_prep.csv_text == res["csv_text"]
        assert re_prep.compatibility.prepared_matches is True


class TestEmbeddingSourceColumns:
    def test_lee_las_fuentes_declaradas_por_el_modelo(self):
        res = generate_project_from_dataset(_csv(_ONEHOT_MAX + 1), target_column="target")
        assert "EMBEDDING" in res["mxai"].upper()
        assert embedding_source_columns(res["mxai"]) == {"cat_baja", "cat_alta"}

    def test_un_modelo_sin_embeddings_no_declara_ninguna(self):
        res = generate_project_from_dataset(_csv(3), target_column="target")
        assert "EMBEDDING" not in res["mxai"].upper()
        assert embedding_source_columns(res["mxai"]) == set()
