"""CONTRATO 81-C3 §13.4/§13.5 — concurrencia, interrupción y benchmark.

Lo que faltaba de los entregables del corte. Y una aclaración que importa
más que las pruebas: **el motor ejecuta los nodos de uno en uno**, en
orden topológico. Dos ramas independientes NO corren a la vez, y es una
decisión, no un descuido:

> El orden de la traza tiene que ser **estable** —el propio contrato lo
> exige, o dos trazas del mismo grafo no se podrían comparar—, y con
> varios nodos escribiendo en la traza a la vez el orden lo decidiría
> quién termine antes. Se pierde la comparación, que es lo único que hace
> auditable un recibo, a cambio de ir más rápido.

Lo que sí se exige y aquí se comprueba es que el motor sea **seguro de
usar en concurrencia**: varios pipelines a la vez, cada uno con su traza,
sin mezclarse. Un runtime con estado compartido daría recibos que
describen la ejecución de otro.
"""

import threading
import unittest

from matrixai.pipelines.engine import (
    Cancelacion,
    digest_de,
    ejecutar_pipeline,
    emitir_recibo,
)

REGISTRY = {"m": "sha256:aaa"}


def _pipeline(pid, pasos=3):
    return {
        "pipeline_id": pid, "version": "1.0", "timeout_s": 60,
        "nodes": [{"id": f"n{i}", "model": "m", "entry_hash": "sha256:aaa",
                   "kind": "t", **({"depends_on": [f"n{i-1}"]} if i else {})}
                  for i in range(pasos)],
    }


def _ejecutores(marca=""):
    def t(*, entradas, nodo, contexto):
        previo = "".join(str(v) for v in entradas.values())
        return f"{previo}{marca}{nodo['id']}"
    return {"t": t}


class VariosPipelinesALaVezNoSeMEZCLANTest(unittest.TestCase):
    def test_ocho_a_la_vez_dan_cada_uno_SU_traza(self):
        salidas: dict[str, dict] = {}
        errores: list[Exception] = []

        def correr(i):
            try:
                salidas[f"p{i}"] = ejecutar_pipeline(
                    _pipeline(f"p{i}"), registry=REGISTRY,
                    ejecutores=_ejecutores(marca=f"[{i}]"))
            except Exception as exc:  # noqa: BLE001
                errores.append(exc)

        hilos = [threading.Thread(target=correr, args=(i,)) for i in range(8)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(timeout=60)

        self.assertEqual(errores, [])
        self.assertEqual(len(salidas), 8)
        for i in range(8):
            r = salidas[f"p{i}"]
            self.assertEqual(r["report"]["status"], "ok")
            # Cada traza atada a SU raíz: si hubiera estado compartido,
            # aquí aparecerían nodos de otro pipeline.
            # La raíz ya NO es el `pipeline_id` (auditoría externa
            # 2026-08-20): es de la EJECUCIÓN, y lleva el id del pipeline
            # como prefijo. Lo que esta prueba defiende sigue igual —cada
            # traza atada a UNA raíz, y la suya— y ahora además distingue
            # dos ejecuciones del mismo pipeline.
            raices = {n["root_id"] for n in r["trace"]["nodes"]}
            self.assertEqual(len(raices), 1)
            self.assertTrue(next(iter(raices)).startswith(f"p{i}#"), raices)
            self.assertEqual(len(r["trace"]["nodes"]), 3)

    def test_lo_que_produjo_cada_uno_es_SUYO(self):
        """La comprobación que de verdad detectaría una mezcla: el valor
        final lleva la marca del pipeline que lo hizo."""
        salidas: dict[int, str] = {}

        def correr(i):
            r = ejecutar_pipeline(_pipeline(f"p{i}"), registry=REGISTRY,
                                  ejecutores=_ejecutores(marca=f"[{i}]"))
            salidas[i] = r["values"]["n2"]["value"]

        hilos = [threading.Thread(target=correr, args=(i,)) for i in range(8)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(timeout=60)

        # ANCLA POSITIVA (auditoría 2ª pasada): sin ella, si los ocho
        # hilos murieran el diccionario quedaría vacío, el bucle no se
        # ejecutaría y esta prueba pasaría **sin comprobar nada**. Un
        # aserto negativo lo pasa un vacío.
        self.assertEqual(len(salidas), 8, "algún hilo no llegó a producir")
        for i, valor in salidas.items():
            self.assertIn(f"[{i}]", valor, f"p{i} no lleva ni su propia marca")
            self.assertNotIn("[", valor.replace(f"[{i}]", ""), f"p{i} vio otra marca")

    def test_los_recibos_de_dos_a_la_vez_no_se_pisan(self):
        resultados = {}

        def correr(i):
            r = ejecutar_pipeline(_pipeline(f"p{i}"), registry=REGISTRY,
                                  ejecutores=_ejecutores())
            resultados[i] = emitir_recibo(r, pipeline=_pipeline(f"p{i}"),
                                          receipt_id=f"r{i}")

        hilos = [threading.Thread(target=correr, args=(i,)) for i in (0, 1)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(timeout=60)

        self.assertTrue(resultados[0]["evidence"]["trace_root"].startswith("p0#"))
        self.assertTrue(resultados[1]["evidence"]["trace_root"].startswith("p1#"))
        # Y son distintas entre sí, que es lo que el recibo necesita para
        # no describir la ejecución de otro.
        self.assertNotEqual(resultados[0]["evidence"]["trace_root"],
                            resultados[1]["evidence"]["trace_root"])
        self.assertNotEqual(resultados[0]["receipt_id"], resultados[1]["receipt_id"])

    def test_una_cancelacion_solo_para_SU_pipeline(self):
        """El testigo es del que lo lanzó, no global: si lo fuera, cancelar
        un pipeline pararía los de todos los demás."""
        testigo = Cancelacion()
        testigo.cancelar("solo este")

        cancelado = ejecutar_pipeline(_pipeline("p0"), registry=REGISTRY,
                                      ejecutores=_ejecutores(), cancelacion=testigo)
        libre = ejecutar_pipeline(_pipeline("p1"), registry=REGISTRY,
                                  ejecutores=_ejecutores())

        self.assertEqual(cancelado["report"]["status"], "cancelled")
        self.assertEqual(libre["report"]["status"], "ok")


class ElMotorEsSECUENCIAL_Y_SeDICETest(unittest.TestCase):
    def test_los_nodos_NO_se_solapan(self):
        """Fijado por prueba para que el día que se decida paralelizar, se
        decida a la vista: hoy el orden de la traza es estable porque nadie
        escribe en ella a la vez."""
        dentro = []
        solapes = []

        def t(*, entradas, nodo, contexto):
            dentro.append(nodo["id"])
            if len(dentro) > 1:
                solapes.append(list(dentro))
            dentro.pop()
            return nodo["id"]

        ejecutar_pipeline(_pipeline("p", pasos=4), registry=REGISTRY,
                          ejecutores={"t": t})
        self.assertEqual(solapes, [])

    def test_el_orden_de_la_traza_es_el_MISMO_en_dos_ejecuciones(self):
        uno = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY, ejecutores=_ejecutores())
        otro = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY, ejecutores=_ejecutores())
        self.assertEqual([n["node_id"] for n in uno["trace"]["nodes"]],
                         [n["node_id"] for n in otro["trace"]["nodes"]])


class UnPipelineDETERMINISTA_DaElMismoDIGESTTest(unittest.TestCase):
    """§13.5: «los pipelines deterministas del corpus de prueba producen el
    mismo digest de salida bajo el entorno de referencia»."""

    def test_diez_ejecuciones_dan_el_MISMO_digest_de_salida(self):
        digests = {
            ejecutar_pipeline(_pipeline("p"), registry=REGISTRY,
                              ejecutores=_ejecutores())["report"]["outputs"]["n2"]
            for _ in range(10)
        }
        self.assertEqual(len(digests), 1, digests)

    def test_y_ese_digest_es_el_del_valor_que_de_verdad_salio(self):
        """Un digest estable de algo equivocado también sería estable."""
        r = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY, ejecutores=_ejecutores())
        self.assertEqual(r["report"]["outputs"]["n2"],
                         digest_de(r["values"]["n2"]["value"]))

    def test_cambiar_una_ENTRADA_cambia_el_digest(self):
        """Si no cambiara, el digest no estaría describiendo la salida."""
        uno = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY,
                                ejecutores=_ejecutores(marca="a"))
        otro = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY,
                                 ejecutores=_ejecutores(marca="b"))
        self.assertNotEqual(uno["report"]["outputs"]["n2"],
                            otro["report"]["outputs"]["n2"])


class ElBENCHMARK_BaseTest(unittest.TestCase):
    """§13.4: «un benchmark reproducible contra el que comparar cortes
    posteriores». Lo que se comprueba aquí NO es el tiempo —un umbral se
    pone en rojo el día que la máquina está cargada, y una prueba que
    falla por el vecino acaba desactivada—, sino que el resultado sea
    **reproducible**, que es el criterio que el contrato escribió.

    Medido el 2026-08-20 en esta máquina: **0,0167 ms por nodo** de coste
    del motor, con 20 nodos en cadena y 5 pasadas.
    """

    def test_el_benchmark_corre_y_da_numeros(self):
        from matrixai.pipelines.benchmark import correr_benchmark

        r = correr_benchmark(nodos=5, pasadas=3)
        self.assertEqual((r["nodes"], r["runs"]), (5, 3))
        self.assertGreater(r["seconds_median"], 0.0)
        self.assertGreater(r["ms_per_node_median"], 0.0)

    def test_y_es_REPRODUCIBLE(self):
        """Un motor rápido que devuelve otra cosa cada vez no sirve para
        sostener un recibo."""
        from matrixai.pipelines.benchmark import correr_benchmark

        r = correr_benchmark(nodos=5, pasadas=5)
        self.assertTrue(r["reproducible"])

    def test_dos_llamadas_dan_el_MISMO_digest(self):
        """Que la reproducibilidad no dependa de estar en la misma llamada."""
        from matrixai.pipelines.benchmark import correr_benchmark

        self.assertEqual(correr_benchmark(nodos=5, pasadas=1)["output_digest"],
                         correr_benchmark(nodos=5, pasadas=1)["output_digest"])

    def test_mide_el_MOTOR_no_los_modelos(self):
        """El nodo de referencia no hace nada a propósito: si ejecutara
        algo, el número diría más del modelo que del runtime."""
        import inspect

        from matrixai.pipelines import benchmark

        cuerpo = inspect.getsource(benchmark._nodo_que_no_hace_nada)
        self.assertIn('return nodo["id"]', cuerpo)

    def test_si_el_pipeline_de_referencia_NO_termina_bien_se_CORTA(self):
        """Un benchmark sobre una ejecución que falló mide el camino de
        error, no el del trabajo — y daría un número tranquilizador."""
        from unittest.mock import patch

        from matrixai.pipelines import benchmark

        def roto(*args, **kw):
            return {"report": {"status": "failed", "reason": "a propósito",
                               "outputs": {}}}

        with patch.object(benchmark, "ejecutar_pipeline", side_effect=roto):
            with self.assertRaises(RuntimeError) as caja:
                benchmark.correr_benchmark(nodos=3, pasadas=1)
        self.assertIn("no terminó bien", str(caja.exception))


class InyectarUnFALLO_EnCadaNodoTest(unittest.TestCase):
    """§13.4, «pruebas de inyección de fallos»: que el nodo que rompe sea
    el que aparece señalado, rompa donde rompa."""

    def _rompe_en(self, objetivo):
        def t(*, entradas, nodo, contexto):
            if nodo["id"] == objetivo:
                raise RuntimeError(f"reventó {objetivo}")
            return nodo["id"]
        return {"t": t}

    def test_el_nodo_señalado_es_SIEMPRE_el_que_rompio(self):
        for i in range(3):
            with self.subTest(nodo=f"n{i}"):
                r = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY,
                                      ejecutores=self._rompe_en(f"n{i}"))
                self.assertEqual(r["report"]["status"], "failed")
                self.assertEqual(r["report"]["node"], f"n{i}")
                # Y lo que venía detrás se declara sin arrancar.
                self.assertEqual(r["report"]["not_started"],
                                 [f"n{j}" for j in range(i + 1, 3)])

    def test_un_fallo_en_el_ULTIMO_no_deja_nada_sin_arrancar(self):
        r = ejecutar_pipeline(_pipeline("p"), registry=REGISTRY,
                              ejecutores=self._rompe_en("n2"))
        self.assertEqual(r["report"]["not_started"], [])
        self.assertEqual(r["report"]["executed"], ["n0", "n1", "n2"])


if __name__ == "__main__":
    unittest.main()
