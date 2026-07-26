# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 64 — regresiones de la 2ª ronda de auditoría (2026-07-26).

El tope duro se comprobaba contra `DenseNetworkGenerator.generate`, que es una
puerta interna. Estos tests entran por las puertas PÚBLICAS —las que usa el
Studio— porque ahí es donde el fallback lo estaba evadiendo.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from matrixai import limits


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


PROMPT = ("clasificar\nFEATURES:\n" + "\n".join(f"f{i}" for i in range(12))
          + "\nSALIDA: y: ProbabilityMap[no, si]\n")


def _csv(n: int = 50) -> str:
    filas = ["a,b,c,d,e,f,g,h,i,j,k,l,y"]
    for i in range(n):
        filas.append(",".join(f"{(i * 7 + j) % 97 / 97:.3f}" for j in range(12))
                     + "," + ("si" if i % 2 else "no"))
    return "\n".join(filas) + "\n"


class TestElFlujoPublicoNoEvadeElTope:
    """[ALTO] — `analyze_playground_request` capturaba CUALQUIER error del
    generador y caía a `PromptSupervisor`, así que un tope superado salía como
    `ok: true` con otro modelo distinto y sin rastro del límite."""

    def test_analyze_playground_request_devuelve_el_error_estructurado(self):
        from matrixai.playground import analyze_playground_request
        with tope("100"):
            res = analyze_playground_request(
                {"mode": "prompt", "prompt": PROMPT, "use_llm": False})
        assert res["ok"] is False
        assert res["accepted"] is False
        assert res["error_kind"] == limits.LIMIT_EXCEEDED
        assert res["limit_key"] == "max_params"
        assert res["maximum"] == 100
        assert res["configurable"] is True

    def test_NO_cae_al_supervisor_con_otro_modelo(self):
        """El síntoma exacto del informe: un modelo generado por el fallback."""
        from matrixai.playground import analyze_playground_request
        with tope("100"):
            res = analyze_playground_request(
                {"mode": "prompt", "prompt": PROMPT, "use_llm": False})
        assert not res.get("mxai"), "no puede entregarse un modelo alternativo"
        assert res.get("supervision_source") is None

    def test_el_fallback_SIGUE_existiendo_para_lo_suyo(self):
        """Un prompt que el generador denso no sabe interpretar debe seguir
        cayendo al supervisor: la corrección acota el fallback, no lo elimina."""
        from matrixai.playground import analyze_playground_request
        res = analyze_playground_request(
            {"mode": "prompt", "prompt": "hola", "use_llm": False})
        assert res.get("mxai"), "un prompt raro sigue produciendo algo"

    def test_el_flujo_desde_datos_dice_QUE_tope_es(self):
        """Antes acababa como un falso error de preparación del CSV, porque el
        modelo del fallback no casaba con las columnas del dataset."""
        from matrixai.training.dataset_project import (
            DatasetProjectError, generate_project_from_dataset)
        with tope("100"):
            with pytest.raises(DatasetProjectError) as exc:
                generate_project_from_dataset(_csv(), "y")
        detalles = getattr(exc.value, "details", {}) or {}
        assert detalles.get("limit_key") == "max_params"
        assert "preparación" not in str(exc.value)
        assert "no generó un modelo válido" not in str(exc.value)

    def test_y_el_endpoint_del_Studio_tambien(self):
        from matrixai.playground import analyze_playground_request
        with tope("2000000"):
            ok = analyze_playground_request(
                {"mode": "prompt", "prompt": PROMPT, "use_llm": False})
        assert ok["ok"] is True, "con un tope razonable se genera igual que siempre"


class TestEstimacionConSupuestos:
    """[MEDIA] — la VRAM se estimaba sin filas ni dispositivo, y el número
    parecía una verdad sobre la GPU cuando describía una CPU con lote 8."""

    def test_los_supuestos_van_declarados(self):
        from matrixai.training.dense_generator import DenseNetworkGenerator
        campos = [f"f{i}" for i in range(12)]
        r = DenseNetworkGenerator().generate(
            PROMPT, input_fields=campos, labels=["no", "si"], rows=94833)
        est = r.architecture_decision["resource_estimate"]
        assert est["assumptions"]["device"] == "cpu"
        assert est["assumptions"]["rows"] == 94833
        assert est["assumptions"]["batch"] == est["effective_batch"]

    def test_los_intrinsecos_no_dependen_del_contexto(self):
        """`param_count` y `weights_gib` valen para cualquier máquina; son los
        que tiene sentido persistir tal cual."""
        from matrixai.training.dense_generator import DenseNetworkGenerator
        campos = [f"f{i}" for i in range(12)]
        gen = DenseNetworkGenerator()
        a = gen.generate(PROMPT, input_fields=campos, labels=["no", "si"], rows=0)
        b = gen.generate(PROMPT, input_fields=campos, labels=["no", "si"], rows=94833)
        ea = a.architecture_decision["resource_estimate"]
        eb = b.architecture_decision["resource_estimate"]
        assert ea["param_count"] == eb["param_count"]
        assert ea["weights_gib"] == eb["weights_gib"]


if __name__ == "__main__":
    pytest.main([__file__])
