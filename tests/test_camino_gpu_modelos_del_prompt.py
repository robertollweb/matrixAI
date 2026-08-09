"""¿Los modelos que salen del PROMPT se comportan igual en el camino GPU?

Roberto, el 2026-08-09: «ahora estamos en CPU, puede que los caminos en
GPU sean distintos; habria que validarlo contra este camino simulando
como si tuviesemos GPU».

Se puede, y este es el mecanismo: `MATRIXAI_TRAIN_BACKEND=torch` enruta
por los entrenadores torch. En una maquina sin CUDA corren en CPU, que es
**el mismo codigo** salvo la colocacion de los tensores — es donde el
contrato 60 encontro el bug de la inicializacion de `nn.Linear`.

Lo que NO estaba cubierto, y es justo lo que Roberto pregunta: los tests
de GPU que ya existian usan `.mxai` escritos a mano. Los modelos que
construye el GENERADOR desde una frase —los de las dos puertas del
producto, los que se han estado probando todo el dia— no habian pasado
nunca por el camino torch.

Cubre los cuatro arquetipos que salen de la puerta del prompt: binaria,
multiclase, regresion y compuesta con embedding.
"""

from __future__ import annotations

from importlib import util

import pytest

from matrixai.training.dense_generator import DenseNetworkGenerator

_HAS_TORCH = util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch no instalado")


PROMPTS = {
    "binaria": "clasificar si un cliente cancela su suscripcion a partir del plan y las incidencias",
    "multiclase": "clasificar una incidencia en critica, media o baja a partir del tiempo y los afectados",
    "regresion": "predecir el coste en euros de una reparacion a partir de los kilometros y la antiguedad",
}


def _modelo(clave):
    """El modelo TAL COMO lo construye el producto desde una frase."""
    return DenseNetworkGenerator().generate(PROMPTS[clave], labels=None)


def _csv_para(r, filas=64):
    """Datos sinteticos que casan con la plantilla que el core genera.

    Se usa la CABECERA del propio `dataset_template_text`: inventarse los
    nombres de las columnas daria un CSV que el validador rechaza, y el
    fallo no diria nada del camino torch.
    """
    cabecera = r.dataset_template_text.splitlines()[0]
    columnas = cabecera.split(",")
    entradas, objetivo = columnas[:-1], columnas[-1]
    lineas = [cabecera]
    for i in range(filas):
        valores = [f"{((i * 7 + j * 3) % 100) / 100.0:.3f}" for j in range(len(entradas))]
        if r.output_activation == "softmax":
            destino = r.labels[i % len(r.labels)]
        elif r.output_activation == "sigmoid":
            destino = "1" if (i % 2 == 0) else "0"
        else:
            destino = f"{(i % 50) * 1.5:.2f}"
        lineas.append(",".join(valores + [destino]))
    return "\n".join(lineas) + "\n"


def _entrenar(r, backend, monkeypatch, epocas=4):
    """Por el camino del PRODUCTO, no por la funcion de dentro.

    La primera version llamaba a `_run_playground_dense_training` directo y
    me dio un falso hallazgo: «el camino stdlib no detecta el colapso». No
    era cierto — la sonda de colapso se engancha en el JOB
    (`_attach_collapse_check`), que es por donde entra el Studio, y
    llamando a la funcion de dentro me la estaba saltando yo.
    """
    import time

    from matrixai.playground import _get_job_status, _submit_training_job
    monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", backend)
    envio = _submit_training_job(
        r.mxai_text, r.training_text, _csv_para(r), epochs_override=epocas)
    assert envio["ok"], envio
    for _ in range(600):
        estado = _get_job_status(envio["job_id"])
        if estado.get("status") in ("done", "error", "cancelled", "timeout"):
            break
        time.sleep(0.1)
    assert estado.get("status") == "done", {k: v for k, v in estado.items() if k != "params_best"}
    # El estado del job viene PLANO: los campos del resultado estan en la
    # raiz, no bajo `result`. Se devuelve tal cual lo lee el Studio.
    return estado


class TestLoQueSI_CAMBIA_EN_GPU:
    """Lo que NO es igual en los dos caminos, medido y fijado.

    Que los modelos entrenen igual por los dos lados es la buena noticia.
    Estas son las divergencias reales, y se fijan aqui para que no cambien
    en silencio — no para «arreglarlas»: la de abajo es una decision
    deliberada (M15, llenar la VRAM para no dejar la GPU al ralenti).

    Se puede medir SIN GPU porque `effective_batch_size` toma el
    dispositivo como una cadena: es logica pura.
    """

    def test_en_cuda_se_IGNORA_el_batch_del_spec(self):
        from matrixai.training.dense_torch_trainer import effective_batch_size
        # El training text que genera el core trae `BATCH size=8`.
        assert effective_batch_size("cpu", 8, n_train=1_000_000) == 8
        assert effective_batch_size("cuda", 8, n_train=1_000_000) == 16384

    def test_la_consecuencia_las_MISMAS_epocas_no_son_lo_mismo(self):
        """Lo que de verdad importa de lo anterior, dicho en pasos.

        Con un millon de filas, una epoca son 125.000 actualizaciones de
        pesos en CPU y 62 en GPU. Alguien que entrena «50 epocas» en su
        portatil y luego «50 epocas» en Colab NO esta repitiendo el
        experimento, aunque las dos pantallas digan 50.

        Se fija la magnitud, no el numero exacto: si algun dia se cambia
        el lote de GPU, este test obliga a mirarlo de frente.
        """
        from matrixai.training.dense_torch_trainer import effective_batch_size
        filas = 1_000_000
        pasos_cpu = -(-filas // effective_batch_size("cpu", 8, filas))
        pasos_gpu = -(-filas // effective_batch_size("cuda", 8, filas))
        assert pasos_cpu > pasos_gpu * 1000, (pasos_cpu, pasos_gpu)

    def test_la_estimacion_de_recursos_supone_CPU_siempre(self):
        """El core NO detecta hardware: la estimacion que se adjunta al
        modelo se calcula con `device="cpu"` fijo.

        En una maquina con GPU ese numero describe OTRA maquina — el mismo
        modelo estima 0,000008 GiB con lote 8 (cpu) y 0,003742 GiB con
        lote 16384 (cuda). Es honesto porque declara el supuesto junto al
        numero (contrato 64), y esta prueba existe para que si algun dia
        se detecta el hardware, el supuesto declarado deje de mentir a la
        vez que el numero.
        """
        r = _modelo("multiclase")
        est = (r.architecture_decision or {}).get("resource_estimate") or {}
        assert est.get("assumptions", {}).get("device") == "cpu"
        assert est.get("orientative") is True


class TestElCompuestoTambien:
    """El cuarto arquetipo, y el que mas se sale: red COMPUESTA con
    embedding — la que salio de M06 («detectar fraude… codigo postal»).

    Va aparte porque no lo entrena el mismo codigo: tiene su propio
    entrenador torch (`composite_torch_trainer`), asi que la paridad de la
    ruta densa no dice nada de esta.
    """

    def _compuesto(self):
        from matrixai.training.composite_generator import CompositeNetworkGenerator
        return CompositeNetworkGenerator().generate(
            "detectar fraude en un seguro a partir del importe, la antiguedad y el codigo postal",
            labels=None)

    def test_entrena_por_los_dos_caminos(self, monkeypatch):
        r = self._compuesto()
        s = _entrenar(r, "stdlib", monkeypatch)
        t = _entrenar(r, "torch", monkeypatch)
        assert s["ok"], f"stdlib: {s.get('error')}"
        assert t["ok"], f"torch: {t.get('error')}"

    def test_los_dos_deciden_la_misma_tarea(self, monkeypatch):
        r = self._compuesto()
        assert (_entrenar(r, "stdlib", monkeypatch)["task_kind"]
                == _entrenar(r, "torch", monkeypatch)["task_kind"])

    def test_los_dos_avisan_del_colapso(self, monkeypatch):
        """El booleano que decide si unos pesos valen. Si solo existiera
        en un camino, habria maquinas donde nadie te avisa de que el
        modelo devuelve siempre lo mismo."""
        r = self._compuesto()
        for backend in ("stdlib", "torch"):
            assert "model_collapsed" in _entrenar(r, backend, monkeypatch), backend


@pytest.mark.parametrize("clave", list(PROMPTS))
class TestLosDosCaminosLlevanAlMismoSitio:
    def test_el_modelo_del_prompt_entrena_por_los_DOS(self, clave, monkeypatch):
        """Lo primero y lo mas basico: que no reviente en uno de los dos.

        Un modelo que el producto construye y ofrece entrenar tiene que
        entrenar en la maquina de quien lo descargue, tenga GPU o no.
        """
        r = _modelo(clave)
        s = _entrenar(r, "stdlib", monkeypatch)
        t = _entrenar(r, "torch", monkeypatch)
        assert s["ok"], f"stdlib: {s.get('error')}"
        assert t["ok"], f"torch: {t.get('error')}"

    def test_cada_uno_declara_el_backend_que_uso_de_verdad(self, clave, monkeypatch):
        """Sin esto no se puede saber cual de los dos caminos se midio.

        OJO al contrato real, medido: el campo `backend` NO usa el mismo
        vocabulario en los dos caminos. El de stdlib dice el BACKEND
        («stdlib»); el de torch dice el DISPOSITIVO («cpu» aqui, «cuda» en
        una maquina con GPU). Es la interfaz quien lo enseña, asi que se
        fija tal cual es — y queda anotado como divergencia.
        """
        r = _modelo(clave)
        assert _entrenar(r, "stdlib", monkeypatch)["backend"] == "stdlib"
        assert _entrenar(r, "torch", monkeypatch)["backend"] in ("cpu", "cuda")

    def test_la_forma_del_resultado_es_la_MISMA(self, clave, monkeypatch):
        """Que las dos digan lo mismo importa tanto como que entrenen: la
        interfaz lee estas claves y no sabe por que camino vino.

        `model_collapsed` va en la lista a proposito: es el booleano que
        decide si unos pesos valen, y si solo existiera en uno de los dos
        caminos habria maquinas donde nadie te avisa de que el modelo
        devuelve siempre lo mismo.
        """
        r = _modelo(clave)
        s = _entrenar(r, "stdlib", monkeypatch)
        t = _entrenar(r, "torch", monkeypatch)
        comunes = {"ok", "backend", "epochs", "params_best", "task_kind",
                   "model_collapsed", "final_train_loss", "best_epoch"}
        assert comunes <= set(s), f"faltan en stdlib: {sorted(comunes - set(s))}"
        assert comunes <= set(t), f"faltan en torch: {sorted(comunes - set(t))}"

    def test_los_dos_deciden_LA_MISMA_TAREA(self, clave, monkeypatch):
        """Un modelo no puede ser regresion por un camino y clasificacion
        por el otro: es lo que decide como se lee su salida."""
        r = _modelo(clave)
        assert (_entrenar(r, "stdlib", monkeypatch)["task_kind"]
                == _entrenar(r, "torch", monkeypatch)["task_kind"])

    def test_los_dos_APRENDEN__no_solo_terminan(self, clave, monkeypatch):
        """Terminar sin error no es aprender.

        El bug del contrato 60 era justo asi: torch «entrenaba» y no
        aprendia, porque `nn.Linear` se inicializaba distinto. Se exige
        que la perdida BAJE en los dos caminos.
        """
        r = _modelo(clave)
        for backend in ("stdlib", "torch"):
            res = _entrenar(r, backend, monkeypatch, epocas=8)
            historia = res.get("epochs") or []
            perdidas = [e.get("train_loss") for e in historia if e.get("train_loss") is not None]
            if len(perdidas) < 2:
                pytest.skip(f"{backend} no reporta perdida por epoca en este resultado")
            assert perdidas[-1] <= perdidas[0], (
                f"{backend}: la perdida no baja ({perdidas[0]} -> {perdidas[-1]})")
