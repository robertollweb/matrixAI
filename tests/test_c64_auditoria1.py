# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 64 — regresiones de la 1ª ronda de auditoría (2026-07-26).

Cada test reproduce el escenario EXACTO del informe. Los dos funcionales se han
verificado revirtiendo su corrección.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from matrixai import limits
from matrixai.training import architecture_policy as ap
from matrixai.training.dense_generator import (
    DenseNetworkGenerator, DenseNetworkGeneratorError)


@contextmanager
def tope(valor: str):
    previo = os.environ.get("MATRIXAI_MAX_PARAMS")
    os.environ["MATRIXAI_MAX_PARAMS"] = valor
    try:
        yield
    finally:
        if previo is None:
            os.environ.pop("MATRIXAI_MAX_PARAMS", None)
        else:
            os.environ["MATRIXAI_MAX_PARAMS"] = previo


def _campos(n: int) -> tuple[str, list[str]]:
    c = [f"f{i}" for i in range(n)]
    return "clasificar\nFEATURES:\n" + "\n".join(c) + "\n", c


class TestTopeDuroInevadible:
    """[ALTO] — el estrechado paraba en el suelo de ancho y nadie volvía a
    comprobar si la red había quedado dentro del tope."""

    def test_la_politica_marca_el_tope_como_insatisfacible(self):
        with tope("100"):
            d = ap.propose(input_dim=12, output_units=2, rows=0)
        assert d.limit_error is not None
        assert d.limit_error["limit_key"] == "max_params"
        assert "max_params_unsatisfiable" in d.limits_applied

    def test_el_generador_RECHAZA_en_vez_de_entregar_una_red_ilegal(self):
        p, campos = _campos(12)
        with tope("100"):
            with pytest.raises(DenseNetworkGeneratorError) as exc:
                DenseNetworkGenerator().generate(
                    p, input_fields=campos, labels=["no", "si"])
        assert exc.value.details["limit_key"] == "max_params"
        assert exc.value.details["error_kind"] == limits.LIMIT_EXCEEDED

    def test_tambien_cuando_la_arquitectura_viene_de_FUERA(self):
        """El prompt, el LLM y el Modo experto pasan por el mismo sitio."""
        p, campos = _campos(12)
        with tope("100"):
            for kwargs in ({"hidden_layers": [(512, "relu")]},
                           {"hidden_layers": [(16, "relu"), (16, "relu")]}):
                with pytest.raises(DenseNetworkGeneratorError):
                    DenseNetworkGenerator().generate(
                        p, input_fields=campos, labels=["no", "si"], **kwargs)

    def test_el_techo_de_DATOS_no_rechaza_nada_solo_avisa(self):
        """La distinción que faltaba: el techo blando acepta la mínima y avisa;
        solo el tope duro rechaza."""
        d = ap.propose(input_dim=12, output_units=2, rows=300)
        assert d.limit_error is None
        assert "min_width_floor" in d.limits_applied
        assert d.warnings, "hay que decir que el dataset es pequeño"

    def test_lo_que_SI_cabe_sigue_generandose(self):
        p, campos = _campos(12)
        with tope("5000"):
            r = DenseNetworkGenerator().generate(
                p, input_fields=campos, labels=["no", "si"])
        assert ap.param_count(len(campos), r.hidden_layers, 1) <= 5000


class TestCandidatasCompletas:
    """[MEDIA-BAJA] — para fuentes externas faltaba la petición original."""

    def test_una_propuesta_externa_aceptada_se_registra(self):
        p, campos = _campos(12)
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"],
            hidden_layers=[(48, "relu")], hidden_layers_source="llm")
        cand = r.architecture_decision["candidates"]
        assert cand, "una candidata aceptada también es una decisión"
        assert cand[0]["hidden_layers"] == [[48, "relu"]]
        assert cand[0]["accepted"] is True
        assert "llm" in cand[0]["reason"]

    def test_una_estrechada_registra_la_PETICION_original(self):
        p, campos = _campos(12)
        with tope("20000"):
            r = DenseNetworkGenerator().generate(
                p.replace("clasificar", "clasificar units=512"),
                input_fields=campos, labels=["no", "si"])
        cand = r.architecture_decision["candidates"]
        assert cand[0]["hidden_layers"][0][0] == 512, "lo que se pidió, primero"
        assert cand[0]["accepted"] is False
        assert cand[-1]["accepted"] is True, "y lo que se aceptó, al final"


class TestFilasEntrenables:
    """[MEDIA] — el presupuesto usaba filas que luego se descartan."""

    def test_las_filas_sin_target_no_cuentan(self):
        from matrixai.training.dataset_project import generate_project_from_dataset
        filas = ["a,b,c,d,e,f,g,h,i,j,k,l,y"]
        for i in range(100):
            v = ",".join(f"{(i * 7 + j) % 97 / 97:.3f}" for j in range(12))
            # 90 de las 100 filas NO tienen target: no entrenan.
            filas.append(v + ("," + ("si" if i % 2 else "no") if i < 10 else ","))
        proj = generate_project_from_dataset("\n".join(filas) + "\n", "y")

        assert proj["provenance"]["rows_dropped_null_target"] == 90
        usadas = proj["architecture_decision"]["policy"]["inputs"]["rows"]
        assert usadas == 10, "el presupuesto se calcula con lo que de verdad entrena"

    def test_y_eso_puede_cambiar_la_red_elegida(self):
        muchas = ap.propose(input_dim=30, output_units=2, rows=50_000)
        pocas = ap.propose(input_dim=30, output_units=2, rows=1_000)
        assert muchas.hidden_layers != pocas.hidden_layers


class TestEstimacionDeRecursos:
    """[MEDIA] — el contrato la exigía y no se adjuntaba."""

    def test_la_decision_lleva_la_estimacion(self):
        p, campos = _campos(12)
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"], rows=94833)
        est = r.architecture_decision["resource_estimate"]
        assert est["param_count"] == r.architecture_decision["params"]
        assert est["vram_train_gib"] >= 0
        # Explícitamente INFORMATIVA: el estimador está definido como orientativo
        # y sin umbral; tratarlo como límite duro sería inventarse un tope.
        assert est["orientative"] is True

    def test_no_es_una_puerta(self):
        """Una estimación alta no impide generar: el límite duro es max_params."""
        p, campos = _campos(64)
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"])
        assert r.mxai_text
        assert r.architecture_decision["resource_estimate"]["param_count"] > 0


class TestM12CabeEnAvanzado:
    """El informe verificó que 12x2048 son 46.176.259 parámetros y caben en
    `avanzado` — no hace falta `ilimitado`."""

    def test_doce_por_2048_cabe_en_avanzado(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "avanzado")
        r = DenseNetworkGenerator().generate(
            "clasificar riesgo con 12 capas ocultas de 2048 unidades",
            labels=["A", "B", "C"])
        assert all(u == 2048 for u, _ in r.hidden_layers)
        assert len(r.hidden_layers) == 12
        assert r.architecture_decision["params"] <= 50_000_000

    def test_y_NO_cabe_en_equilibrado(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_LIMITS_PROFILE", "equilibrado")
        r = DenseNetworkGenerator().generate(
            "clasificar riesgo con 12 capas ocultas de 2048 unidades",
            labels=["A", "B", "C"])
        assert max(u for u, _ in r.hidden_layers) < 2048
        assert "max_params" in r.architecture_decision["limits_applied"]


if __name__ == "__main__":
    pytest.main([__file__])
