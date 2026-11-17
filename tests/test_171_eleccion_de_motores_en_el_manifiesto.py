# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""171 — la ELECCIÓN de motores de un estudio, en su `reproduce.json` (`study.engine_selection`).

El Studio deja elegir qué motores compiten (contrato 171) y lo declara en el sobre del estudio como `eleccion_de_motores`. El manifiesto
del núcleo tenía la firma cerrada y el paquete de cualquier estudio con elección reventaba con `TypeError` (auditoría 1 del 171, B-1).
Aquí: el campo se acepta, se escribe tal cual se da (en el orden del estudio), se comprueba entero y, sin él, el manifiesto es el de
siempre byte a byte.
"""
from __future__ import annotations

import json

import pytest

from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio, manifiesto_de_estudio
from matrixai.export.reproduce import ReproduceManifestError, canonical_json, verify_manifest_digest

ENTORNO = {"python": "3.12", "matrixai_core": "x"}       # fijo: lo que se compara es el manifiesto, no el proceso
ELECCION = {"pedida": ["lightgbm"], "compiten": ["lightgbm", "baseline"],
            "fuera_por_la_eleccion": ["sklearn.hgb", "matrixai.dense.torch_cpu"]}


#: Un estudio con casi todo lo que el manifiesto sabe declarar, y el sha256 de su forma canónica CALCULADO CON EL NÚCLEO DE ANTES del 171
#: (`git show 126c795:matrixai/export/manifiesto_de_estudio.py`, reparación 2 del 171, M-i de su auditoría 2). Fijado aquí: compara con el
#: de antes, no el módulo consigo mismo. Si el manifiesto cambia A PROPÓSITO (un campo nuevo para todos), se recalcula y se dice por qué.
KW_DE_REFERENCIA = dict(
    filas=200, huella_del_dataset="46a6b5fb901ade9afb61dcd93451e05ddbf2d09f2b7d7643a32111922d5e1b9e",
    tipos_de_campo={"x1": "numerica", "categoria": "categorica"}, columnas_excluidas=["row_id"], semilla_del_reparto=7,
    metricas=[{"name": "auroc", "value": 0.75, "split": "test", "n_observations": 40}],
    filas_de_desarrollo=160, filas_de_prueba_reservada=40, filas_que_entrenaron_el_modelo=136,
    modo="rapido", motores_fuera_por_el_modo=["matrixai.dense.torch_cpu"], motor_del_modelo_entregado="lightgbm",
    nombre_del_motor_entregado={"es": "LightGBM", "en": "LightGBM"}, entorno={"python": "3.12", "matrixai_core": "x"})
SHA_DEL_MANIFIESTO_DE_ANTES = "9a151e379afedacf5e22111f84570b46323566346956da9a3db5e1c58063930d"


def test_sin_eleccion_el_manifiesto_es_byte_a_byte_el_de_antes_del_171():
    import hashlib
    for ahora in (manifiesto_de_estudio(**KW_DE_REFERENCIA), manifiesto_de_estudio(**KW_DE_REFERENCIA, eleccion_de_motores=None)):
        assert hashlib.sha256(canonical_json(ahora).encode()).hexdigest() == SHA_DEL_MANIFIESTO_DE_ANTES
        assert "engine_selection" not in ahora["study"]


def test_la_eleccion_se_escribe_en_el_bloque_study_tal_cual_y_en_su_orden(tmp_path):
    escrito = escribir_manifiesto_de_estudio(tmp_path, modo="completo", entorno=ENTORNO, eleccion_de_motores=ELECCION)
    leido = json.loads((tmp_path / "reproduce.json").read_text(encoding="utf-8"))
    assert leido == escrito and verify_manifest_digest(leido)
    assert leido["study"]["engine_selection"] == {
        "requested": ["lightgbm"], "competed": ["lightgbm", "baseline"],
        "left_out_by_choice": ["sklearn.hgb", "matrixai.dense.torch_cpu"]}
    assert leido["study"]["mode"] == "completo"


def test_los_pedidos_que_no_pudieron_competir_viajan_con_su_motivo_en_los_dos_idiomas():
    motivo = {"es": "TabICL no compite en la demostración pública.", "en": "TabICL does not compete in the public demo."}
    m = manifiesto_de_estudio(entorno=ENTORNO, eleccion_de_motores={
        **ELECCION, "pedida": ["lightgbm", "tabicl.v2"], "no_pudieron_competir": [{"motor": "tabicl.v2", "motivo": motivo}]})
    assert m["study"]["engine_selection"]["could_not_compete"] == [{"engine": "tabicl.v2", "reason": motivo}]
    # Una lista vacía no añade la clave: no había nadie que declarar.
    vacia = manifiesto_de_estudio(entorno=ENTORNO, eleccion_de_motores={**ELECCION, "no_pudieron_competir": []})
    assert "could_not_compete" not in vacia["study"]["engine_selection"]


@pytest.mark.parametrize("mala", [
    "lightgbm", ["lightgbm"], {"pedida": ["lightgbm"]},
    {**ELECCION, "compiten": "lightgbm"}, {**ELECCION, "compiten": []}, {**ELECCION, "pedida": [1]},
    {**ELECCION, "inventada": True},
    {**ELECCION, "no_pudieron_competir": [{"motor": "tabicl.v2", "motivo": {"es": "x"}}]},
    {**ELECCION, "no_pudieron_competir": [{"motor": "", "motivo": {"es": "x", "en": "y"}}]},
])
def test_una_eleccion_a_medias_no_se_publica(mala):
    with pytest.raises((ReproduceManifestError, ValueError)):
        manifiesto_de_estudio(entorno=ENTORNO, eleccion_de_motores=mala)
