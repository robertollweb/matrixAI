"""Dos decisiones de Roberto del 2026-08-09, tomadas juntas porque van al
mismo sitio: lo que el core DICE y la MAQUINA sobre la que lo dice.

**1. Los avisos del pipeline llegaban siempre en español.**
Con `locale=en` el core ya traducia `understanding.safety_limits` y el
veredicto, pero no los avisos: se leia «Arquitectura (dense, propuesta
por el LLM): The problem is a binary classification task…» — marco
español y contenido ingles en la misma frase. Y el hueco no estaba en el
core: el endpoint del Studio TENIA el locale y no se lo pasaba, igual que
`useHome` en la interfaz.

Lo que se traduce es el MARCO. La frase del LLM se deja como viene: son
sus palabras, y reescribirlas seria cambiar lo que dijo.

**2. La estimacion de recursos suponia `cpu` SIEMPRE.**
El comentario lo justificaba diciendo que «el core no detecta hardware».
Sigue sin detectarlo en el generador — se lo DICEN desde donde ya se
sabe—. En una maquina con GPU el numero describia otra maquina: el mismo
modelo estima 0,000008 GiB con lote 8 (cpu) y 0,003742 con lote 16384
(cuda), x468.
"""

from __future__ import annotations

from matrixai.playground import _marcos, analyze_playground_request
from matrixai.training.dense_generator import DenseNetworkGenerator

PROMPT = "clasificar una incidencia en critica, media o baja a partir del tiempo y los afectados"


class TestLosAvisosHablanElIdiomaQueSePide:
    def test_los_marcos_estan_en_los_dos_idiomas(self):
        es, en = _marcos("es"), _marcos("en")
        assert set(es) == set(en)
        assert es["contrato_mxtrain"] != en["contrato_mxtrain"]
        assert es["origen_llm"] != en["origen_llm"]

    def test_un_locale_desconocido_cae_a_español_y_no_lanza(self):
        # Un `locale` raro NO puede tumbar una generacion: se contesta en
        # el idioma de siempre.
        assert _marcos("kl") is _marcos("es")
        assert _marcos(None) is _marcos("es")
        assert _marcos("") is _marcos("es")

    def test_el_marco_se_traduce_y_la_frase_del_LLM_NO(self):
        porque = "The problem is a typical binary classification task."
        en = _marcos("en")["arquitectura"]("dense", porque)
        assert en.startswith("Architecture (dense, proposed by the LLM):")
        # Sus palabras, intactas.
        assert porque in en

    def test_por_el_pipeline_de_verdad_el_aviso_sale_en_ingles(self):
        """Por el producto, no por la funcion: se pide `locale=en` al
        analizador y se mira lo que sale en los avisos del pipeline."""
        r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT, "locale": "en"})
        avisos = [w for e in r.get("pipeline_stages", []) for w in e.get("warnings", [])]
        assert avisos, "sin avisos no se puede medir nada"
        assert not any("Contrato .mxtrain" in w for w in avisos), avisos
        assert any("mxtrain contract" in w for w in avisos), avisos

    def test_sin_pedir_idioma_sigue_saliendo_en_español(self):
        # Retrocompatibilidad: quien no pida idioma no cambia de idioma.
        r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT})
        avisos = [w for e in r.get("pipeline_stages", []) for w in e.get("warnings", [])]
        assert any("Contrato .mxtrain" in w for w in avisos), avisos


class TestLaEstimacionDiceDeQueMAQUINA:
    def test_el_supuesto_declarado_es_la_maquina_que_se_le_dice(self):
        for device in ("cpu", "cuda"):
            r = DenseNetworkGenerator().generate(PROMPT, rows=200_000, device=device)
            est = (r.architecture_decision or {})["resource_estimate"]
            assert est["assumptions"]["device"] == device

    def test_y_el_NUMERO_cambia_con_la_maquina_no_solo_la_etiqueta(self):
        """Media verdad: cambiar la etiqueta y dejar el numero de una CPU
        seria peor que decir «cpu» siempre."""
        gen = DenseNetworkGenerator()
        cpu = (gen.generate(PROMPT, rows=200_000, device="cpu").architecture_decision or {})["resource_estimate"]
        gpu = (gen.generate(PROMPT, rows=200_000, device="cuda").architecture_decision or {})["resource_estimate"]
        assert gpu["effective_batch"] > cpu["effective_batch"] * 100
        assert gpu["vram_train_gib"] > cpu["vram_train_gib"]
        # Y el supuesto sigue AL LADO del numero: es lo que lo hace
        # juzgable (contrato 64).
        assert gpu["assumptions"]["batch"] == gpu["effective_batch"]
        assert gpu["orientative"] is True

    def test_lo_INTRINSECO_no_depende_de_la_maquina(self):
        """Los parametros y el peso del modelo son los mismos en las dos:
        si cambiaran, seria que se esta construyendo otra red."""
        gen = DenseNetworkGenerator()
        cpu = (gen.generate(PROMPT, rows=200_000, device="cpu").architecture_decision or {})["resource_estimate"]
        gpu = (gen.generate(PROMPT, rows=200_000, device="cuda").architecture_decision or {})["resource_estimate"]
        assert cpu["param_count"] == gpu["param_count"]
        assert cpu["weights_gib"] == gpu["weights_gib"]

    def test_sin_decir_nada_sigue_suponiendo_cpu(self):
        r = DenseNetworkGenerator().generate(PROMPT, rows=200_000)
        assert (r.architecture_decision or {})["resource_estimate"]["assumptions"]["device"] == "cpu"

    def test_por_el_PIPELINE_llega_la_maquina_detectada__simulando_GPU(self, monkeypatch):
        """El cableado, no la funcion: que `analyze_playground_request`
        le diga al generador la maquina que YA detecta para entrenar.

        Se simula la GPU sustituyendo el detector, que es el metodo que
        pidio Roberto para validar el camino de GPU sin tenerla: la
        decision es logica pura y solo depende de una cadena.
        """
        import matrixai.playground as pg

        for device in ("cpu", "cuda"):
            monkeypatch.setattr(pg, "_select_train_backend", lambda d=device: (d == "cuda", d))
            r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT})
            est = ((r.get("architecture_decision") or {}).get("policy") or {}).get("resource_estimate")
            assert est is not None, "sin estimacion no se puede medir nada"
            assert est["assumptions"]["device"] == device, est["assumptions"]
