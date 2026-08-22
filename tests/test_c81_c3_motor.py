"""CONTRATO 81-C3 — el motor que une las piezas y ejecuta de punta a punta.

Hasta este fichero el 81 tenía siete módulos correctos y **nada que los
uniera**. Aquí se fija lo que el motor promete, y cada afirmación se
comprueba por su lado malo:

* no se empieza si la verificación previa falla, y se dicen TODOS los
  problemas;
* un `kind` sin ejecutor **no corre** —ni se salta ni cae en otro—;
* `abstain` **no autoriza**: si se declaró política y no dijo `allow`,
  nadie ha dicho que sí;
* `dry-run` **no llama al ejecutor** y **no emite recibo**;
* lo que no llegó a arrancar **se dice**;
* la traza **no guarda el dato**, solo su huella (§13.3).
"""

import unittest

from matrixai.pipelines.engine import (
    Cancelacion,
    EjecutorDesconocido,
    PipelineInvalido,
    ReciboImposibleDeEmitir,
    digest_de,
    ejecutar_pipeline,
    emitir_recibo,
)
from matrixai.pipelines.receipt import firmar_recibo, nivel_del_recibo
from matrixai.pipelines.runtime import entradas_para, verificar_antes_de_ejecutar

REGISTRY = {"limpiador": "sha256:aaa", "clasificador": "sha256:bbb"}


def _pipeline(**cambios):
    base = {
        "pipeline_id": "p-1",
        "version": "1.0",
        "timeout_s": 30,
        "nodes": [
            {"id": "limpia", "model": "limpiador", "entry_hash": "sha256:aaa",
             "kind": "texto"},
            {"id": "clasifica", "model": "clasificador", "entry_hash": "sha256:bbb",
             "kind": "texto", "depends_on": ["limpia"]},
        ],
    }
    base.update(cambios)
    return base


def _ejecutores(registro=None):
    def texto(*, entradas, nodo, contexto):
        if registro is not None:
            registro.append(nodo["id"])
        if not entradas:
            return "hola"
        return "".join(str(v) for v in entradas.values()).upper()
    return {"texto": texto}


class DePuntaAPuntaTest(unittest.TestCase):
    def test_ejecuta_los_dos_nodos_en_orden_y_transporta_la_salida(self):
        pasados = []
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(pasados))
        self.assertEqual(pasados, ["limpia", "clasifica"])
        self.assertEqual(salida["report"]["status"], "ok")
        self.assertEqual(salida["values"]["clasifica"]["value"], "HOLA")

    def test_la_traza_queda_completa_y_con_los_dos_nodos_atados_a_su_raiz(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        traza = salida["trace"]
        self.assertTrue(traza["complete"])
        self.assertEqual([n["node_id"] for n in traza["nodes"]], ["limpia", "clasifica"])
        # Una raíz, la suya, y con el id del pipeline como prefijo: desde
        # la auditoría externa la raíz identifica la EJECUCIÓN, no el
        # grafo — dos ejecuciones del mismo pipeline compartían raíz.
        raices = {n["root_id"] for n in traza["nodes"]}
        self.assertEqual(len(raices), 1)
        self.assertTrue(next(iter(raices)).startswith("p-1#"), raices)

    def test_el_digest_que_declara_el_motor_es_el_que_el_transporte_comprueba(self):
        """Si el motor digestara de otra forma que `entradas_para`, TODA
        salida daría discrepancia — y parecería una manipulación."""
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        producido = salida["values"]["limpia"]
        self.assertEqual(producido["digest"], digest_de(producido["value"]))
        # Y el transporte lo acepta sin quejarse, que es la comprobación
        # que de verdad ata las dos reglas.
        self.assertEqual(
            entradas_para("otro", ["limpia"], salida["values"], comprobar_digests=True),
            {"limpia": "hola"})

    def test_la_huella_de_entrada_del_segundo_nodo_NO_es_nula(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        nodos = {n["node_id"]: n for n in salida["trace"]["nodes"]}
        self.assertIsNone(nodos["limpia"]["input_digest"],
                          "un nodo sin dependencias no tiene entrada que describir")
        self.assertTrue(nodos["clasifica"]["input_digest"].startswith("sha256:"))


class NoSeEmpiezaSiNoVerificaTest(unittest.TestCase):
    def test_un_ciclo_no_ejecuta_NADA(self):
        pasados = []
        malo = _pipeline(nodes=[
            {"id": "a", "model": "limpiador", "entry_hash": "sha256:aaa",
             "kind": "texto", "depends_on": ["b"]},
            {"id": "b", "model": "clasificador", "entry_hash": "sha256:bbb",
             "kind": "texto", "depends_on": ["a"]},
        ])
        with self.assertRaises(PipelineInvalido):
            ejecutar_pipeline(malo, registry=REGISTRY, ejecutores=_ejecutores(pasados))
        self.assertEqual(pasados, [], "empezar un pipeline roto deja trabajo a medias")

    def test_se_dicen_TODOS_los_problemas_no_el_primero(self):
        malo = _pipeline(timeout_s=0, nodes=[
            {"id": "a", "model": "limpiador", "entry_hash": "sha256:OTRO", "kind": "texto"},
        ])
        with self.assertRaises(PipelineInvalido) as caja:
            ejecutar_pipeline(malo, registry=REGISTRY, ejecutores=_ejecutores())
        self.assertEqual(len(caja.exception.problemas), 2, caja.exception.problemas)


class UnKindSinEjecutorNoCorreTest(unittest.TestCase):
    def test_falla_cerrado_en_vez_de_saltarse_el_nodo(self):
        pipeline = _pipeline(nodes=[
            {"id": "raro", "model": "limpiador", "entry_hash": "sha256:aaa",
             "kind": "inventado"},
        ])
        salida = ejecutar_pipeline(pipeline, registry=REGISTRY, ejecutores=_ejecutores())
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertEqual(salida["report"]["node"], "raro")

    def test_dice_QUE_ejecutores_hay_para_no_tener_que_adivinarlo(self):
        pipeline = _pipeline(nodes=[
            {"id": "raro", "model": "limpiador", "entry_hash": "sha256:aaa",
             "kind": "inventado"},
        ])
        salida = ejecutar_pipeline(pipeline, registry=REGISTRY, ejecutores=_ejecutores())
        self.assertIn("inventado", salida["report"]["reason"])
        self.assertIn("texto", salida["report"]["reason"])

    def test_NO_lo_ejecuta_con_el_unico_que_hay(self):
        pasados = []
        pipeline = _pipeline(nodes=[
            {"id": "raro", "model": "limpiador", "entry_hash": "sha256:aaa",
             "kind": "inventado"},
        ])
        ejecutar_pipeline(pipeline, registry=REGISTRY, ejecutores=_ejecutores(pasados))
        self.assertEqual(pasados, [])


class LaPoliticaDecideAntesDeEjecutarTest(unittest.TestCase):
    POLITICA_QUE_NIEGA = {
        "schema_version": "1.0", "name": "corte", "version": "1.0",
        "default": "deny",
        "rules": [{"id": "R1", "when": {"eq": [{"var": "listo"}, True]}, "then": "allow"}],
    }
    POLITICA_QUE_SE_ABSTIENE = {
        "schema_version": "1.0", "name": "muda", "version": "1.0",
        "default": "abstain", "rules": [],
    }

    def test_un_deny_para_el_pipeline_y_el_nodo_NO_ejecuta(self):
        pasados = []
        salida = ejecutar_pipeline(
            _pipeline(policy=self.POLITICA_QUE_NIEGA), registry=REGISTRY,
            ejecutores=_ejecutores(pasados), contexto={"listo": False})
        self.assertEqual(salida["report"]["status"], "denied")
        self.assertEqual(pasados, [])

    def test_el_motivo_NOMBRA_la_regla_que_decidio(self):
        salida = ejecutar_pipeline(
            _pipeline(policy=self.POLITICA_QUE_NIEGA), registry=REGISTRY,
            ejecutores=_ejecutores(), contexto={"listo": False})
        self.assertIn("corte@1.0", salida["report"]["reason"])

    def test_ABSTAIN_TAMPOCO_autoriza(self):
        """«Esta política no opina» no es un permiso: si nadie ha dicho
        que sí, el nodo no corre."""
        pasados = []
        salida = ejecutar_pipeline(
            _pipeline(policy=self.POLITICA_QUE_SE_ABSTIENE), registry=REGISTRY,
            ejecutores=_ejecutores(pasados), contexto={})
        self.assertEqual(salida["report"]["status"], "denied")
        self.assertEqual(pasados, [])

    def test_con_allow_ejecuta_y_la_decision_queda_en_la_traza(self):
        salida = ejecutar_pipeline(
            _pipeline(policy=self.POLITICA_QUE_NIEGA), registry=REGISTRY,
            ejecutores=_ejecutores(), contexto={"listo": True})
        self.assertEqual(salida["report"]["status"], "ok")
        politicas = salida["trace"]["nodes"][0]["policies"]
        self.assertEqual(politicas[0]["decision"], "allow")
        self.assertEqual(politicas[0]["rule_id"], "R1")

    def test_sin_politica_declarada_el_nodo_corre(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        self.assertEqual(salida["report"]["status"], "ok")


class DryRunNoEjecutaNadaTest(unittest.TestCase):
    def test_no_llama_al_ejecutor_ni_una_vez(self):
        pasados = []
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(pasados), dry_run=True)
        self.assertEqual(pasados, [], "simular con efectos reales no es simular")
        self.assertEqual(salida["report"]["status"], "ok")

    def test_recorre_el_grafo_ENTERO_aunque_no_ejecute(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(), dry_run=True)
        self.assertEqual([n["node_id"] for n in salida["trace"]["nodes"]],
                         ["limpia", "clasifica"])
        self.assertEqual(salida["report"]["not_started"], [])

    def test_no_declara_salidas_que_no_ha_producido(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(), dry_run=True)
        self.assertEqual(salida["report"]["outputs"], {})

    def test_declara_lo_que_NO_ha_comprobado(self):
        """Un «dry-run OK» a secas se lee como que el cableado de datos
        también está bien, y una simulación no transporta nada."""
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(), dry_run=True)
        self.assertIn("output_transport", salida["report"]["not_checked"])
        self.assertIn("node_execution", salida["report"]["not_checked"])
        self.assertIn("policies", salida["report"]["checked"])

    def test_una_ejecucion_de_verdad_NO_lleva_esa_lista(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        self.assertNotIn("not_checked", salida["report"])

    def test_una_simulacion_NO_emite_recibo(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(), dry_run=True)
        with self.assertRaises(ReciboImposibleDeEmitir) as caja:
            emitir_recibo(salida, pipeline=_pipeline(), receipt_id="r1")
        self.assertIn("dry-run", str(caja.exception))


class LoQueNoArrancoSeDiceTest(unittest.TestCase):
    def test_un_fallo_a_la_mitad_enumera_lo_que_nunca_empezo(self):
        def rompe(*, entradas, nodo, contexto):
            raise RuntimeError("el modelo no cargó")
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores={"texto": rompe})
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertEqual(salida["report"]["node"], "limpia")
        self.assertEqual(salida["report"]["not_started"], ["clasifica"])

    def test_el_error_del_nodo_viaja_declarado_no_tragado(self):
        def rompe(*, entradas, nodo, contexto):
            raise RuntimeError("el modelo no cargó")
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores={"texto": rompe})
        self.assertIn("el modelo no cargó", salida["report"]["reason"])
        self.assertIn("RuntimeError", salida["report"]["reason"])

    def test_una_ejecucion_entera_no_deja_nada_sin_arrancar(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        self.assertEqual(salida["report"]["not_started"], [])


class HaberCorridoNoEsHaberProducidoTest(unittest.TestCase):
    def test_un_nodo_que_devuelve_None_corta_AHI_y_no_dos_nodos_despues(self):
        def vacio(*, entradas, nodo, contexto):
            return None
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores={"texto": vacio})
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertEqual(salida["report"]["node"], "limpia",
                         "el que no produjo es quien tiene que aparecer")
        self.assertIn("sin producir", salida["report"]["reason"])


class CancelarDetieneLoPendienteTest(unittest.TestCase):
    def test_lo_que_no_ha_arrancado_NO_arranca(self):
        testigo = Cancelacion()
        pasados = []

        def cancela_despues_del_primero(*, entradas, nodo, contexto):
            pasados.append(nodo["id"])
            testigo.cancelar("lo paró quien lo lanzó")
            return "x"

        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores={"texto": cancela_despues_del_primero},
                                   cancelacion=testigo)
        self.assertEqual(pasados, ["limpia"])
        self.assertEqual(salida["report"]["status"], "cancelled")
        self.assertEqual(salida["report"]["not_started"], ["clasifica"])

    def test_el_motivo_de_la_cancelacion_viaja(self):
        testigo = Cancelacion()
        testigo.cancelar("se apagó la máquina")
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(), cancelacion=testigo)
        self.assertEqual(salida["report"]["reason"], "se apagó la máquina")

    def test_un_testigo_sin_activar_no_para_nada(self):
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores(), cancelacion=Cancelacion())
        self.assertEqual(salida["report"]["status"], "ok")


class ElLimiteDeTiempoTest(unittest.TestCase):
    def _reloj(self, marcas):
        it = iter(marcas)
        ultimo = [0.0]

        def reloj():
            try:
                ultimo[0] = next(it)
            except StopIteration:
                pass
            return ultimo[0]
        return reloj

    def test_pasado_el_limite_el_siguiente_nodo_NO_arranca(self):
        pasados = []
        salida = ejecutar_pipeline(
            _pipeline(timeout_s=10), registry=REGISTRY,
            ejecutores=_ejecutores(pasados),
            reloj=self._reloj([0.0, 0.0, 0.5, 99.0]))
        self.assertEqual(salida["report"]["status"], "timed_out")
        self.assertEqual(pasados, ["limpia"])
        self.assertEqual(salida["report"]["not_started"], ["clasifica"])

    def test_el_motivo_lleva_el_tiempo_Y_el_limite_no_un_se_pasó(self):
        salida = ejecutar_pipeline(
            _pipeline(timeout_s=10), registry=REGISTRY, ejecutores=_ejecutores(),
            reloj=self._reloj([0.0, 0.0, 0.5, 99.0]))
        self.assertIn("99.000s", salida["report"]["reason"])
        self.assertIn("10.000s", salida["report"]["reason"])

    def test_el_ALCANCE_del_limite_se_declara_en_vez_de_dejarlo_suponer(self):
        """Comprobar entre nodos no es poder cortar uno por dentro, y
        prometer lo segundo sería peor que no ofrecer nada."""
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores=_ejecutores())
        self.assertIn("between-nodes", salida["report"]["timeout_scope"])

    def test_al_ejecutor_se_le_pasa_el_tiempo_que_QUEDA(self):
        visto = {}

        def mira(*, entradas, nodo, contexto):
            visto[nodo["id"]] = contexto["remaining_s"]
            return "x"
        ejecutar_pipeline(_pipeline(timeout_s=10), registry=REGISTRY,
                          ejecutores={"texto": mira})
        self.assertLessEqual(visto["limpia"], 10.0)
        self.assertGreater(visto["limpia"], 0.0)


class ReintentarSoloLoQueSePuedeTest(unittest.TestCase):
    def _pipeline_de_uno(self, **extra):
        nodo = {"id": "a", "model": "limpiador", "entry_hash": "sha256:aaa",
                "kind": "texto"}
        nodo.update(extra)
        return _pipeline(nodes=[nodo])

    def test_un_nodo_idempotente_se_reintenta_y_sale_adelante(self):
        intentos = []

        def falla_una_vez(*, entradas, nodo, contexto):
            intentos.append(1)
            if len(intentos) == 1:
                raise RuntimeError("un tropiezo de red")
            return "ya"

        salida = ejecutar_pipeline(
            self._pipeline_de_uno(retries=1, idempotent=True),
            registry=REGISTRY, ejecutores={"texto": falla_una_vez})
        self.assertEqual(salida["report"]["status"], "ok")
        self.assertEqual(len(intentos), 2)
        self.assertEqual(salida["trace"]["nodes"][0]["attempts"], 2)

    def test_por_defecto_NO_se_reintenta_aunque_se_pidan_reintentos(self):
        intentos = []

        def siempre_falla(*, entradas, nodo, contexto):
            intentos.append(1)
            raise RuntimeError("otra vez")

        salida = ejecutar_pipeline(
            self._pipeline_de_uno(retries=3), registry=REGISTRY,
            ejecutores={"texto": siempre_falla})
        self.assertEqual(len(intentos), 1,
                         "dar por seguro lo no declarado duplicaría cobros y correos")
        self.assertEqual(salida["report"]["status"], "failed")

    def test_el_reintento_negado_DICE_por_que_no_se_hizo(self):
        def siempre_falla(*, entradas, nodo, contexto):
            raise RuntimeError("otra vez")
        salida = ejecutar_pipeline(
            self._pipeline_de_uno(retries=3, effects=True), registry=REGISTRY,
            ejecutores={"texto": siempre_falla})
        self.assertIn("reintento no realizado", salida["report"]["reason"])
        self.assertIn("idempotente", salida["report"]["reason"])


class LaTrazaNoGuardaElDatoTest(unittest.TestCase):
    def test_el_valor_producido_NO_aparece_en_la_traza_serializada(self):
        """§13.3: identificadores, digests y estados sí; el dato en claro
        no. Un secreto en la traza se copia a cada sitio donde se guarde."""
        secreto = "NUMERO-DE-HISTORIA-CLINICA-449213"

        def suelta_un_secreto(*, entradas, nodo, contexto):
            return secreto

        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores={"texto": suelta_un_secreto})
        import json
        self.assertNotIn(secreto, json.dumps(salida["trace"], ensure_ascii=False))
        self.assertNotIn(secreto, json.dumps(salida["report"], ensure_ascii=False))
        # Y el valor sí sigue disponible para quien ejecuta, que lo necesita.
        self.assertEqual(salida["values"]["limpia"]["value"], secreto)


class ElReciboDiceLoQueLaTrazaVIOTest(unittest.TestCase):
    def _ok(self):
        return ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                 ejecutores=_ejecutores())

    def test_un_recibo_de_una_ejecucion_buena_lleva_sus_dos_pasos(self):
        recibo = emitir_recibo(self._ok(), pipeline=_pipeline(), receipt_id="r-1")
        self.assertEqual([p["id"] for p in recibo["steps"]], ["limpia", "clasifica"])
        self.assertEqual(recibo["output"]["outcome"], "ok")

    def test_se_puede_firmar_y_alcanza_A1(self):
        recibo = emitir_recibo(self._ok(), pipeline=_pipeline(), receipt_id="r-1")
        sobre = firmar_recibo(recibo, clave=b"secreta", key_id="k1")
        self.assertEqual(nivel_del_recibo(sobre), "A1")

    def test_un_pipeline_CORTADO_no_pasa_por_uno_entero(self):
        def rompe(*, entradas, nodo, contexto):
            raise RuntimeError("no cargó")
        salida = ejecutar_pipeline(_pipeline(), registry=REGISTRY,
                                   ejecutores={"texto": rompe})
        recibo = emitir_recibo(salida, pipeline=_pipeline(), receipt_id="r-2")
        self.assertEqual(recibo["output"]["outcome"], "failed")
        self.assertEqual(recibo["output"]["not_started"], ["clasifica"],
                         "sin esto, el recibo de un pipeline cortado se lee igual "
                         "que el de uno entero")

    def test_una_traza_con_nodos_ABIERTOS_no_produce_recibo(self):
        from matrixai.pipelines.trace import TrazaDeEjecucion
        traza = TrazaDeEjecucion("p-1")
        traza.abrir_nodo("a", modelo="limpiador")
        con_hueco = {"trace": traza.a_dict(), "report": {"status": "ok"}}
        with self.assertRaises(ReciboImposibleDeEmitir) as caja:
            emitir_recibo(con_hueco, pipeline=_pipeline(), receipt_id="r-3")
        self.assertIn("sin cerrar", str(caja.exception))

    def test_lleva_el_CONTENIDO_MINIMO_del_contrato(self):
        """AUDITORÍA 2ª pasada: faltaban CUATRO de las doce secciones del
        §14.2 —`created_at`, `models`, `input` y `attestation`— y la que
        más duele es `models`: §16.6 pide que el recibo identifique todas
        las versiones de modelos, y no identificaba ninguna."""
        recibo = emitir_recibo(self._ok(), pipeline=_pipeline(), receipt_id="r-1")
        for seccion in ("schema_version", "receipt_id", "event_type", "created_at",
                        "subject", "pipeline", "models", "input", "checks",
                        "output", "evidence", "steps"):
            self.assertIn(seccion, recibo, seccion)

    def test_los_MODELOS_van_con_su_digest(self):
        recibo = emitir_recibo(self._ok(), pipeline=_pipeline(), receipt_id="r-1")
        self.assertEqual({m["model_id"] for m in recibo["models"]},
                         {"limpiador", "clasificador"})
        self.assertTrue(all(m["digest"].startswith("sha256:") for m in recibo["models"]))

    def test_la_ATESTACION_vive_en_el_SOBRE_no_en_el_payload(self):
        """Repetirla dentro serían dos sitios declarando lo mismo, y el de
        dentro va firmado: podría contradecir al de fuera sin que se note."""
        recibo = emitir_recibo(self._ok(), pipeline=_pipeline(), receipt_id="r-1")
        self.assertNotIn("attestation", recibo)
        sobre = firmar_recibo(recibo, clave=b"secreta", key_id="k1")
        self.assertTrue(sobre["signatures"])
        self.assertEqual(sobre["signatures"][0]["keyid"], "k1")

    def test_el_recibo_NO_inventa_la_huella_del_dato(self):
        """El motor no ve el dato del caso: solo lo que produjo cada nodo.
        Un `payload_digest` aquí sería firmar la huella de algo que no se
        ha visto."""
        recibo = emitir_recibo(self._ok(), pipeline=_pipeline(), receipt_id="r-1")
        self.assertIsNone(recibo["input"]["payload_digest"])
        self.assertFalse(recibo["input"]["raw_data_included"])
        self.assertEqual(recibo["input"]["entry_nodes"], ["limpia"])

    def test_la_fecha_es_la_de_la_EJECUCION_no_la_de_emitir(self):
        """La de emitir puede ser mucho después de lo que describe."""
        salida = self._ok()
        recibo = emitir_recibo(salida, pipeline=_pipeline(), receipt_id="r-1")
        self.assertEqual(recibo["created_at"], salida["trace"]["nodes"][0]["started_at"])

    def test_las_politicas_evaluadas_viajan_en_el_recibo(self):
        salida = ejecutar_pipeline(
            _pipeline(policy={"schema_version": "1.0", "name": "p", "version": "1",
                              "default": "allow", "rules": []}),
            registry=REGISTRY, ejecutores=_ejecutores())
        recibo = emitir_recibo(salida, pipeline=_pipeline(), receipt_id="r-4")
        self.assertEqual(len(recibo["checks"]["policy_results"]), 2)


if __name__ == "__main__":
    unittest.main()


class UnIdRepetidoNoTAPA_LoDemasTest(unittest.TestCase):
    """§13.5: la verificación previa dice **TODOS** los problemas.

    Sonda adversaria del C3 (2026-08-20), el corte que el refutador dejó
    sin refutar. El `continue` que salta el nodo con el id repetido —y
    que está ahí por un motivo bueno: en el grafo pisaría al primero—
    **se saltaba también lo suyo**. Los MISMOS dos nodos daban tres
    problemas con ids distintos y solo dos con el id repetido.

    Es exactamente el fallo que la promesa existe para evitar: arreglabas
    el id, volvías a lanzar, y aparecían dos problemas nuevos que ya
    estaban ahí desde el principio.
    """

    _REG = {"limpiador": "sha256:aaa"}

    def _problemas(self, ids):
        pipeline = {
            "pipeline_id": "p", "version": "1.0", "timeout_s": 30,
            "nodes": [
                {"id": ids[0], "model": "NO_EXISTE", "entry_hash": "sha256:z", "kind": "k"},
                {"id": ids[1], "model": "TAMPOCO", "entry_hash": "sha256:y",
                 "kind": "k", "depends_on": ["inexistente"]},
            ],
        }
        return verificar_antes_de_ejecutar(pipeline, self._REG)["problems"]

    def test_con_ids_distintos_salen_los_TRES(self):
        """El otro lado primero: sin esto, la prueba de abajo la pasaría
        una verificación que no encuentra nada nunca."""
        problemas = " | ".join(self._problemas(("a", "b")))
        self.assertIn("NO_EXISTE", problemas)
        self.assertIn("TAMPOCO", problemas)
        self.assertIn("inexistente", problemas)

    def test_y_con_el_id_REPETIDO_tambien_salen_TODOS(self):
        problemas = " | ".join(self._problemas(("x", "x")))
        # El id repetido, que es lo que ya se decía…
        self.assertIn("dos pasos con el id 'x'", problemas)
        # …y lo del nodo TAPADO, que es lo que se perdía.
        self.assertIn("TAMPOCO", problemas)
        self.assertIn("inexistente", problemas)

    def test_el_id_repetido_NO_entra_en_el_grafo(self):
        """El `continue` estaba por un motivo bueno y ese motivo sigue en
        pie: el segundo pisaría al primero y uno desaparecería."""
        pipeline = {
            "pipeline_id": "p", "version": "1.0", "timeout_s": 30,
            "nodes": [{"id": "x", "model": "limpiador", "entry_hash": "sha256:aaa", "kind": "k"},
                      {"id": "x", "model": "limpiador", "entry_hash": "sha256:aaa", "kind": "k"}],
        }
        r = verificar_antes_de_ejecutar(pipeline, self._REG)
        self.assertFalse(r["ok"])
        self.assertEqual(r["order"].count("x"), 1)
