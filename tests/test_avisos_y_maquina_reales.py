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


class TestLoQueElGeneradorEXPLICA:
    """La ultima cadena en español que quedaba en la pantalla inglesa.

    Es un REGISTRO de decision —«arquitectura de origen 'llm': 64-32,
    2.369 parámetros»— y por eso se traduce en el core y no al pintarlo:
    reescribirlo en la interfaz seria tener dos versiones de lo que el
    core decidio.

    Con el aviso de dataset pequeño va el mismo argumento, y ese ademas
    es de los que sirven para algo: dice que el modelo puede memorizar en
    vez de aprender.
    """

    def test_el_porque_de_la_arquitectura_y_el_aviso_de_datos_pequeños(self):
        gen = DenseNetworkGenerator()
        es = (gen.generate(PROMPT, rows=40, locale="es").architecture_decision or {})
        en = (gen.generate(PROMPT, rows=40, locale="en").architecture_decision or {})
        assert "dimensión efectiva" in es["rationale"]
        assert "effective input dimension" in en["rationale"]
        assert any("es pequeño para" in w for w in (es.get("warnings") or []))
        assert any("is small for" in w for w in (en.get("warnings") or []))

    def test_sin_pedir_idioma_sigue_en_español(self):
        r = DenseNetworkGenerator().generate(PROMPT, rows=40)
        assert "dimensión efectiva" in (r.architecture_decision or {})["rationale"]

    def test_por_el_PIPELINE_llega_el_idioma_al_generador(self):
        """El cableado, que es donde estaba el hueco las dos veces."""
        r = analyze_playground_request({"mode": "prompt", "prompt": PROMPT, "locale": "en"})
        porque = ((r.get("architecture_decision") or {}).get("policy") or {}).get("rationale") or ""
        assert "effective input dimension" in porque, porque


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


class TestElEsfuerzoDelRun:
    """Recomendacion #1 de Roberto: que el producto DIGA que las mismas
    «50 epocas» no son lo mismo en CPU y en GPU.

    La epoca no es una unidad comparable entre maquinas; la actualizacion
    de pesos si. Medido con un millon de filas:

    | camino                       | lote   | pasos por epoca |
    |------------------------------|--------|-----------------|
    | stdlib (por defecto sin GPU) |      1 |       1.000.000 |
    | torch en CPU                 |      8 |         125.000 |
    | torch en CUDA                | 16.384 |              62 |

    Dieciseis mil veces entre los extremos.
    """

    def test_la_cuenta_de_pasos_redondea_HACIA_ARRIBA(self):
        """Con 51 filas y lote 8 son SIETE pasos, no seis: la ultima
        hornada cuenta aunque no llene el lote."""
        from matrixai.training.spec import esfuerzo_de_entrenamiento as e
        assert e(51, 8, 3)["steps_per_epoch"] == 7
        assert e(51, 8, 3)["weight_updates"] == 21

    def test_los_tres_caminos_del_barrido(self):
        from matrixai.training.spec import esfuerzo_de_entrenamiento as e
        filas = 1_000_000
        assert e(filas, 1, 50)["steps_per_epoch"] == 1_000_000
        assert e(filas, 8, 50)["steps_per_epoch"] == 125_000
        assert e(filas, 16384, 50)["steps_per_epoch"] == 62

    def test_los_bordes_no_lanzan_y_no_dividen_por_cero(self):
        from matrixai.training.spec import esfuerzo_de_entrenamiento as e
        assert e(0, 8, 3)["steps_per_epoch"] == 0
        assert e(51, 0, 3)["effective_batch_size"] == 1  # lote 0 no existe
        assert e(51, 8, 0)["weight_updates"] == 0

    def test_el_camino_stdlib_declara_su_lote_REAL_que_es_1(self, monkeypatch, tmp_path):
        """Y no el del spec.

        Medido con una sonda antes de escribir esto: 153 actualizaciones
        para 51 filas y 3 epocas, con el `.mxtrain` pidiendo `BATCH
        size=8`. El bucle de stdlib actualiza ejemplo a ejemplo e IGNORA
        el lote declarado — asi que decir el del spec seria mentir sobre
        lo que paso.
        """
        import sys
        import time

        sys.path.insert(0, "tests")
        from matrixai.playground import _get_job_status, _submit_training_job
        from matrixai.training.dense_generator import DenseNetworkGenerator
        from test_camino_gpu_modelos_del_prompt import _csv_para

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "stdlib")
        r = DenseNetworkGenerator().generate(
            "clasificar si un pedido llega tarde a partir de la distancia y el peso")
        assert "BATCH size=8" in r.training_text
        env = _submit_training_job(r.mxai_text, r.training_text, _csv_para(r, 64), epochs_override=3)
        for _ in range(600):
            st = _get_job_status(env["job_id"])
            if st.get("status") in ("done", "error"):
                break
            time.sleep(0.1)
        assert st.get("status") == "done"
        esfuerzo = st.get("effort")
        assert esfuerzo is not None, "el core no declara el esfuerzo del run"
        assert esfuerzo["effective_batch_size"] == 1, esfuerzo
        assert esfuerzo["weight_updates"] == esfuerzo["train_rows"] * 3, esfuerzo

    def test_el_camino_torch_declara_el_SUYO_y_es_distinto(self, monkeypatch):
        import sys
        import time

        sys.path.insert(0, "tests")
        from matrixai.playground import _get_job_status, _submit_training_job
        from matrixai.training.dense_generator import DenseNetworkGenerator
        from test_camino_gpu_modelos_del_prompt import _csv_para

        r = DenseNetworkGenerator().generate(
            "clasificar si un pedido llega tarde a partir de la distancia y el peso")
        csv = _csv_para(r, 64)
        esfuerzos = {}
        for backend in ("stdlib", "torch"):
            monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", backend)
            env = _submit_training_job(r.mxai_text, r.training_text, csv, epochs_override=3)
            for _ in range(600):
                st = _get_job_status(env["job_id"])
                if st.get("status") in ("done", "error"):
                    break
                time.sleep(0.1)
            assert st.get("status") == "done", st.get("error")
            esfuerzos[backend] = st.get("effort")
        assert esfuerzos["stdlib"] is not None and esfuerzos["torch"] is not None
        # Lo que hay que poder ver: MISMAS epocas, esfuerzo distinto.
        assert esfuerzos["stdlib"]["weight_updates"] > esfuerzos["torch"]["weight_updates"]


class TestElMotorYLaMaquinaSonDosHechos:
    """Recomendacion #2 de Roberto, aprobada.

    El campo `backend` hablaba DOS idiomas: el camino stdlib decia el
    MOTOR («stdlib») y el de torch decia el DISPOSITIVO («cpu», «cuda»).
    Juntarlos perdia informacion en los dos sentidos — con «stdlib» no se
    sabia la maquina, y con «cpu» no se sabia si habia sido torch.
    """

    def test_los_dos_campos_salen_separados(self):
        from matrixai.playground import motor_y_maquina
        assert motor_y_maquina("torch", "cuda") == {"backend": "torch", "device": "cuda"}
        assert motor_y_maquina("stdlib", "cpu") == {"backend": "stdlib", "device": "cpu"}

    def test_un_run_GUARDADO_con_la_forma_vieja_no_se_pierde(self):
        """Lo importante del cambio: no romper lo que ya existe.

        Un run de antes solo trae `backend`. Si nombra una MAQUINA, el
        motor era torch — y deducirlo es mejor que perder el dato.
        """
        from matrixai.playground import motor_y_maquina_de as leer
        assert leer({"backend": "cuda"}) == {"backend": "torch", "device": "cuda"}
        assert leer({"backend": "cpu"}) == {"backend": "torch", "device": "cpu"}
        assert leer({"backend": "stdlib"}) == {"backend": "stdlib", "device": "cpu"}

    def test_la_forma_nueva_manda_sobre_la_deduccion(self):
        from matrixai.playground import motor_y_maquina_de as leer
        assert leer({"backend": "torch", "device": "cuda"})["device"] == "cuda"

    def test_sin_nada_no_se_inventa_una_gpu(self):
        from matrixai.playground import motor_y_maquina_de as leer
        assert leer({}) == {"backend": "stdlib", "device": "cpu"}

    def test_por_el_PIPELINE_los_dos_caminos_declaran_las_dos_cosas(self, monkeypatch):
        import sys
        import time

        sys.path.insert(0, "tests")
        from matrixai.playground import _get_job_status, _submit_training_job
        from matrixai.training.dense_generator import DenseNetworkGenerator
        from test_camino_gpu_modelos_del_prompt import _csv_para

        r = DenseNetworkGenerator().generate(
            "clasificar si un pedido llega tarde a partir de la distancia y el peso")
        csv = _csv_para(r, 64)
        for backend, motor in (("stdlib", "stdlib"), ("torch", "torch")):
            monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", backend)
            env = _submit_training_job(r.mxai_text, r.training_text, csv, epochs_override=2)
            for _ in range(600):
                st = _get_job_status(env["job_id"])
                if st.get("status") in ("done", "error"):
                    break
                time.sleep(0.1)
            assert st.get("status") == "done", st.get("error")
            assert st.get("backend") == motor, st.get("backend")
            # La maquina, SIEMPRE declarada — antes con stdlib no se sabia.
            assert st.get("device") in ("cpu", "cuda"), st.get("device")
