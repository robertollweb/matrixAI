# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 64 — política de arquitectura en la generación desde datos.

C1 presupuesto de parámetros · C2 política determinista · C3 precedencia ·
C4 decisión auditable. (C5, búsqueda de candidatas, queda fuera por decisión de
Roberto del 2026-07-26.)

Los tests afirman EFECTOS observables: qué red sale, con qué parámetros, qué se
registra y qué se le dice a la persona. Es la lección de las seis rondas de
auditoría del contrato 63.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager

import pytest

from matrixai import limits
from matrixai.training import architecture_policy as ap
from matrixai.training.dense_generator import DenseNetworkGenerator


@contextmanager
def perfil(nombre: str):
    previo = os.environ.get("MATRIXAI_LIMITS_PROFILE")
    os.environ["MATRIXAI_LIMITS_PROFILE"] = nombre
    try:
        yield
    finally:
        if previo is None:
            os.environ.pop("MATRIXAI_LIMITS_PROFILE", None)
        else:
            os.environ["MATRIXAI_LIMITS_PROFILE"] = previo


def _prompt(n_features: int, *, clases: list[str] | None = None) -> tuple[str, list[str]]:
    campos = [f"f{i}" for i in range(n_features)]
    salida = (f"\nSALIDA: y: ProbabilityMap[{', '.join(clases)}]" if clases else "")
    return ("clasificar\nFEATURES:\n" + "\n".join(campos) + salida + "\n"), campos


def _anchos(resultado) -> list[int]:
    return [u for u, _ in resultado.hidden_layers]


# ── C1 — presupuesto de parámetros de primera clase ──────────────────────────

class TestC1Presupuesto:
    def test_cada_perfil_tiene_su_tope(self):
        with perfil("equilibrado"):
            assert limits.get_limit("max_params") == 2_000_000
        with perfil("avanzado"):
            assert limits.get_limit("max_params") == 50_000_000
        with perfil("ilimitado"):
            assert limits.get_limit("max_params") is None

    def test_override_por_variable_de_entorno(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "12345")
        assert limits.get_limit("max_params") == 12_345
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "0")
        assert limits.get_limit("max_params") is None

    def test_hosted_ignora_el_override(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_HOSTED", "1")
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "999999999")
        assert limits.get_limit("max_params") == 2_000_000

    def test_aparece_en_el_snapshot_de_la_UI(self):
        assert "max_params" in limits.limits_snapshot()["limits"]

    def test_el_error_habla_de_la_RED_no_del_dataset(self):
        payload = limits.limit_error("max_params", 3_400_000)
        assert payload["unit"] == "params"
        assert "La red tiene" in payload["error"]
        assert "dataset" not in payload["error"]

    def test_una_arquitectura_que_excede_el_presupuesto_se_estrecha(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "5000")
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p.replace("clasificar", "clasificar units=512"),
            input_fields=campos, labels=["no", "si"])
        assert max(_anchos(r)) < 512
        assert ap.param_count(len(campos), r.hidden_layers, 2) <= 5000

    def test_el_conteo_de_parametros_es_el_real(self):
        # 3 entradas → 8 → 4 → 1 salida.
        capas = [(8, "relu"), (4, "relu")]
        esperado = (3 * 8 + 8) + (8 * 4 + 4) + (4 * 1 + 1)
        assert ap.param_count(3, capas, 1) == esperado


# ── C2 — política determinista de dimensionado ───────────────────────────────

class TestC2Politica:
    def test_es_determinista(self):
        a = ap.propose(input_dim=12, output_units=2, rows=94833)
        b = ap.propose(input_dim=12, output_units=2, rows=94833)
        assert a.hidden_layers == b.hidden_layers
        assert a.params == b.params

    def test_el_ancho_sale_de_la_dimension_de_entrada(self):
        anchos = {d: ap.propose(input_dim=d, output_units=2, rows=0).hidden_layers[0][0]
                  for d in (1, 4, 16, 64)}
        # Monótono: más entradas nunca dan una primera capa más estrecha.
        assert anchos[1] <= anchos[4] <= anchos[16] <= anchos[64]

    def test_MAS_FILAS_no_agrandan_la_red(self):
        """Invariante 1, que es la razón de ser de este contrato."""
        pocas = ap.propose(input_dim=12, output_units=2, rows=50_000)
        muchas = ap.propose(input_dim=12, output_units=2, rows=5_000_000)
        assert pocas.hidden_layers == muchas.hidden_layers
        # Lo que cambia es el PRESUPUESTO, no lo que se consume de él.
        assert muchas.budget["rows_ceiling"] > pocas.budget["rows_ceiling"]

    def test_pero_MENOS_filas_si_la_encogen(self):
        grande = ap.propose(input_dim=64, output_units=2, rows=1_000_000)
        pequeno = ap.propose(input_dim=64, output_units=2, rows=2_000)
        assert pequeno.params < grande.params
        assert "budget" in pequeno.limits_applied

    def test_la_dimension_EFECTIVA_cuenta_el_one_hot(self):
        """Una categórica de 8 valores ocupa 8 columnas, no una."""
        assert ap.effective_input_dim(5, one_hot_widths={"ciudad": 8}) == 12
        assert ap.effective_input_dim(5, lag_columns=6) == 11
        assert ap.effective_input_dim(5, one_hot_widths={"c": 3}, lag_columns=2) == 9

    def test_una_categorica_declarada_ensancha_la_red(self):
        """El efecto de verdad, no solo la aritmética: el mismo prompt con una
        categórica de 8 valores produce una red más ancha que sin ella."""
        gen = DenseNetworkGenerator()
        sin_cat = gen.generate(
            "clasificar\nFEATURES:\na\nb\nc\nSALIDA: y: ProbabilityMap[no, si]",
            labels=["no", "si"])
        con_cat = gen.generate(
            "clasificar\nFEATURES:\na\nb\nc: Categorical[l, m, x, n, o, p, q, r]\n"
            "SALIDA: y: ProbabilityMap[no, si]",
            labels=["no", "si"])
        assert max(_anchos(con_cat)) >= max(_anchos(sin_cat))

    def test_mas_clases_dan_una_capa_mas(self):
        pocas = ap.propose(input_dim=12, output_units=3, rows=0)
        muchas = ap.propose(input_dim=12, output_units=20, rows=0)
        assert len(muchas.hidden_layers) == len(pocas.hidden_layers) + 1

    def test_regresion_y_clasificacion_binaria_no_se_quedan_sin_capacidad(self):
        """El caso centígrados→Kelvin: d=1. Con `4·d` a secas salía 16-16 y el
        R² caía de 0,99 a 0,94 — la red dejaba de poder representar la relación.
        """
        d = ap.propose(input_dim=1, output_units=1, task="regression", rows=100)
        assert d.hidden_layers[0][0] >= 32

    def test_ya_no_sale_siempre_128_64_32(self):
        """El hallazgo que originó el contrato: 3.651 filas y 94.833 daban la
        MISMA red, y cualquier entrada de más de diez columnas también."""
        anchos = {ap.propose(input_dim=d, output_units=2, rows=0).hidden_layers[0][0]
                  for d in (11, 12, 30, 100)}
        assert len(anchos) > 1, "distintas entradas deben dar distintas redes"

    def test_el_presupuesto_PUEDE_hacerlas_converger(self):
        """Honestidad sobre el alcance: con un techo apretado, entradas distintas
        acaban en la misma red. No es el bug de antes —entonces convergían por un
        tapering ciego, ahora por un presupuesto explícito— y queda registrado en
        `limits_applied`, que es justo la diferencia que persigue el contrato.
        """
        d30 = ap.propose(input_dim=30, output_units=2, rows=94_833)
        d12 = ap.propose(input_dim=12, output_units=2, rows=94_833)
        assert d30.hidden_layers == d12.hidden_layers
        assert "budget" in d30.limits_applied
        assert "budget" not in d12.limits_applied


# ── C3 — precedencia explícita ───────────────────────────────────────────────

class TestC3Precedencia:
    def test_el_prompt_gana_a_la_politica(self):
        p, campos = _prompt(12, clases=["no", "si"])
        gen = DenseNetworkGenerator()
        politica = gen.generate(p, input_fields=campos, labels=["no", "si"], rows=94833)
        explicito = gen.generate(p.replace("clasificar", "clasificar units=128"),
                                 input_fields=campos, labels=["no", "si"], rows=94833)
        assert _anchos(politica) != _anchos(explicito)
        assert all(u == 128 for u in _anchos(explicito))

    def test_el_llamante_gana_al_prompt(self):
        """`hidden_layers` explícito (LLM aceptado o Modo experto)."""
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p.replace("clasificar", "clasificar units=128"),
            input_fields=campos, labels=["no", "si"], rows=94833,
            hidden_layers=[(48, "relu"), (24, "relu")],
            hidden_layers_source="llm")
        assert _anchos(r) == [48, 24]
        assert r.architecture_decision["source"] == "llm"

    def test_NINGUNA_fuente_se_salta_el_limite_duro(self, monkeypatch):
        """Invariante 2. Ni el prompt, ni el LLM, ni el Modo experto."""
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "3000")
        p, campos = _prompt(12, clases=["no", "si"])
        gen = DenseNetworkGenerator()
        for kwargs in (
            {},                                                    # política
            {"hidden_layers": [(512, "relu"), (512, "relu")]},     # llamante
        ):
            r = gen.generate(p.replace("clasificar", "clasificar units=1024"),
                             input_fields=campos, labels=["no", "si"], **kwargs)
            assert ap.param_count(len(campos), r.hidden_layers, 2) <= 3000

    def test_el_suelo_de_ancho_tambien_es_innegociable(self):
        """`_MIN_RELU_WIDTH`: una capa ReLU más estrecha no aprende."""
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"],
            hidden_layers=[(2, "relu"), (2, "relu")])
        assert min(_anchos(r)) >= 16


# ── C4 — decisión auditable ──────────────────────────────────────────────────

class TestC4Auditable:
    def test_la_politica_registra_entradas_presupuesto_y_candidatas(self):
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"], rows=94833)
        d = r.architecture_decision
        assert d["source"] == "policy"
        assert d["params"] > 0
        assert d["inputs"]["rows"] == 94833
        assert d["budget"]["max_params"] == limits.get_limit("max_params")
        assert d["budget"]["rows_ceiling"] == 9483
        assert d["candidates"], "hay que poder ver qué se consideró"
        assert d["rationale"]

    def test_se_registra_TAMBIEN_cuando_gana_otra_fuente(self):
        """"La decisión se registra o no existe" (invariante 3)."""
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"],
            hidden_layers=[(48, "relu")], hidden_layers_source="user_override")
        d = r.architecture_decision
        assert d["source"] == "user_override"
        assert d["params"] == ap.param_count(len(campos), [(48, "relu")], 2)

    def test_el_limite_aplicado_queda_anotado(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "3000")
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p.replace("clasificar", "clasificar units=512"),
            input_fields=campos, labels=["no", "si"])
        assert "max_params" in r.architecture_decision["limits_applied"]

    def test_estrechar_NO_es_silencioso(self, monkeypatch):
        monkeypatch.setenv("MATRIXAI_MAX_PARAMS", "3000")
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p.replace("clasificar", "clasificar units=512"),
            input_fields=campos, labels=["no", "si"])
        aviso = "\n".join(r.warnings)
        assert "512" in aviso and "Ajustes" in aviso

    def test_un_dataset_pequeno_para_su_entrada_se_avisa(self):
        p, campos = _prompt(12, clases=["no", "si"])
        r = DenseNetworkGenerator().generate(
            p, input_fields=campos, labels=["no", "si"], rows=300)
        aviso = "\n".join(r.warnings)
        assert "pequeño" in aviso and "memorizar" in aviso

    def test_la_decision_llega_al_resultado_del_playground(self):
        from matrixai.playground import analyze_playground_request
        p, _ = _prompt(12, clases=["no", "si"])
        res = analyze_playground_request({
            "mode": "prompt", "prompt": p, "use_llm": False,
            "dataset_rows": 94833,
        })
        assert res["ok"]
        d = res["architecture_decision"]
        assert d["policy"]["budget"]["rows_ceiling"] == 9483
        assert d["layers_source"] == "policy"
        assert d["params"] > 0

    def test_el_flujo_desde_datos_completo_usa_las_filas_reales(self):
        from matrixai.training.dataset_project import generate_project_from_dataset
        import random
        random.seed(7)

        def csv(n: int) -> str:
            filas = ["a,b,c,d,e,f,g,h,i,j,k,l,llueve"]
            for _ in range(n):
                vals = [f"{random.random():.3f}" for _ in range(12)]
                filas.append(",".join(vals) + "," + ("si" if random.random() > 0.4 else "no"))
            return "\n".join(filas) + "\n"

        # 40.000 filas: por debajo de `max_rows` del perfil por defecto
        # (50.000), que es otro tope y no el que se prueba aquí.
        chico = generate_project_from_dataset(csv(2_000), "llueve")
        grande = generate_project_from_dataset(csv(40_000), "llueve")
        anchos = lambda p: [int(u) for u in re.findall(r"LAYER Dense units=(\d+)", p["mxai"])]  # noqa: E731
        # Con pocas filas el presupuesto aprieta; con muchas, la red es la que
        # pide la dimensión de entrada.
        assert max(anchos(chico)) <= max(anchos(grande))


if __name__ == "__main__":
    pytest.main([__file__])
