"""CONTRATO 81-C3 — el ejecutor que corre modelos REALES del registry.

El motor sabe orquestar y no sabe ejecutar. Este es el ejecutor que lo
ata a P21, y con él un pipeline del 81 corre de verdad: dos modelos del
registry de ejemplo, la salida del primero entrando en el segundo, y un
recibo al final que describe lo que pasó.

Lo que este fichero fija, y cada regla se comprueba por su lado malo:

* el digest se comprueba **otra vez al ejecutar**, no solo en la
  verificación previa;
* una entrada **manipulada no ejecuta**;
* un modelo **sin versión no se resuelve**;
* con dos fuentes de entrada **no se elige una**;
* los tipos se comprueban si están declarados, y **si no, se dice**.
"""

import json
import shutil
import unittest
from pathlib import Path

from matrixai.pipelines.engine import ejecutar_pipeline, emitir_recibo
from matrixai.pipelines.executors import (
    EntradaAmbigua,
    ModeloAlterado,
    ModeloCambiado,
    TipoDeEntradaIncompatible,
    ejecutor_de_modelos,
    mapa_del_registry,
)
from matrixai.registry.model_registry import ModelRegistry

EJEMPLO = Path(__file__).resolve().parent.parent / "examples" / "text-routing"

#: Un ticket de facturación, en el vocabulario del ejemplo.
_PALABRAS_DE_FACTURA = ("bow_invoice", "bow_charge", "bow_refund", "bow_payment")


def _vector_del_ejemplo():
    import re
    texto = (EJEMPLO / "registry/entries/feature_extractor/v1/model.mxai").read_text()
    campos = re.findall(r"^  (bow_\w+): Score", texto, re.M)
    return {"TicketBOW": {c: (1.0 if c in _PALABRAS_DE_FACTURA else 0.0) for c in campos}}


class _ConRegistryPropio(unittest.TestCase):
    """Una copia del registry del ejemplo: las pruebas que lo manipulan no
    pueden tocar el que se publica."""

    def setUp(self):
        import tempfile
        self.raiz = Path(tempfile.mkdtemp())
        shutil.copytree(EJEMPLO / "registry", self.raiz / "registry")
        self.registro = ModelRegistry(self.raiz / "registry")
        self.fe = self.registro.get("feature_extractor", "v1")

    def tearDown(self):
        shutil.rmtree(self.raiz, ignore_errors=True)

    def _pipeline_de_uno(self, **extra):
        nodo = {"id": "encode", "model": "feature_extractor@v1",
                "entry_hash": self.fe.entry_hash, "kind": "modelo",
                "input": _vector_del_ejemplo()}
        nodo.update(extra)
        return {"pipeline_id": "routing", "version": "1.0", "timeout_s": 60,
                "nodes": [nodo]}

    def _ejecutar(self, pipeline, **kw):
        return ejecutar_pipeline(
            pipeline, registry=mapa_del_registry(self.registro),
            ejecutores={"modelo": ejecutor_de_modelos(self.registro)}, **kw)


class UnModeloDeVerdadCorreTest(_ConRegistryPropio):
    def test_el_pipeline_ejecuta_el_modelo_del_registry_y_produce_salida(self):
        salida = self._ejecutar(self._pipeline_de_uno())
        self.assertEqual(salida["report"]["status"], "ok", salida["report"]["reason"])
        self.assertIsNotNone(salida["values"]["encode"]["value"])

    def test_la_salida_lleva_la_señal_que_declara_el_modelo(self):
        salida = self._ejecutar(self._pipeline_de_uno())
        producido = salida["values"]["encode"]["value"]
        self.assertIn("routing_signal", json.dumps(producido, default=str))

    def test_el_recibo_describe_la_ejecucion_con_su_digest_de_salida(self):
        salida = self._ejecutar(self._pipeline_de_uno())
        recibo = emitir_recibo(salida, pipeline=self._pipeline_de_uno(),
                               receipt_id="r-real")
        paso = recibo["steps"][0]
        self.assertEqual(paso["status"], "ok")
        self.assertTrue(paso["output_digest"].startswith("sha256:"))
        self.assertEqual(paso["model"], "feature_extractor@v1")

    def test_el_mapa_del_registry_se_MIDE_del_registry_no_se_escribe_a_mano(self):
        mapa = mapa_del_registry(self.registro)
        self.assertEqual(mapa["feature_extractor@v1"], self.fe.entry_hash)
        self.assertIn("route_classifier@v1", mapa)


class DosModelosREALESEncadenadosTest(_ConRegistryPropio):
    """La prueba que de verdad dice que el 81 ejecuta: el ejemplo del
    repositorio, entero, por el motor del contrato."""

    def _pipeline_de_dos(self, **cambios_del_segundo):
        rc = self.registro.get("route_classifier", "v1")
        segundo = {"id": "clasifica", "model": "route_classifier@v1",
                   "entry_hash": rc.entry_hash, "kind": "modelo",
                   "depends_on": ["encode"],
                   "input_map": {"Signal": {"routing_signal": "routing_signal"}}}
        segundo.update(cambios_del_segundo)
        return {"pipeline_id": "routing", "version": "1.0", "timeout_s": 60,
                "nodes": [{"id": "encode", "model": "feature_extractor@v1",
                           "entry_hash": self.fe.entry_hash, "kind": "modelo",
                           "input": _vector_del_ejemplo()},
                          segundo]}

    def test_la_señal_del_primero_alimenta_al_segundo_y_sale_una_clasificacion(self):
        salida = self._ejecutar(self._pipeline_de_dos())
        self.assertEqual(salida["report"]["status"], "ok", salida["report"]["reason"])
        probs = salida["values"]["clasifica"]["value"]["state"]["probs"]
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_el_recibo_lleva_los_DOS_pasos_con_sus_digests(self):
        salida = self._ejecutar(self._pipeline_de_dos())
        recibo = emitir_recibo(salida, pipeline=self._pipeline_de_dos(),
                               receipt_id="r-dos")
        self.assertEqual([p["id"] for p in recibo["steps"]], ["encode", "clasifica"])
        self.assertTrue(all(p["output_digest"] for p in recibo["steps"]))
        # El segundo declara de QUÉ entrada salió: sin eso, el recibo no
        # ataría los dos pasos más que por el orden.
        self.assertTrue(recibo["steps"][1]["input_digest"])

    def test_SIN_receta_de_enlace_NO_se_mapea_por_su_cuenta(self):
        """La regla de la casa: no se mapean los datos de nadie
        automáticamente, salvo con la receta del propio modelo."""
        pipeline = self._pipeline_de_dos()
        del pipeline["nodes"][1]["input_map"]
        salida = self._ejecutar(pipeline)
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertEqual(salida["report"]["node"], "clasifica")
        self.assertIn("input_map", salida["report"]["reason"])

    def test_una_receta_que_apunta_a_lo_que_NO_existe_para_y_dice_que_hay(self):
        salida = self._ejecutar(self._pipeline_de_dos(
            input_map={"Signal": {"routing_signal": "señal_inventada"}}))
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertIn("señal_inventada", salida["report"]["reason"])
        self.assertIn("routing_signal", salida["report"]["reason"])

    def test_un_origen_con_VARIOS_valores_no_se_recorta_al_primero(self):
        """`TicketBOW` tiene 30: quedarse con el primero sería elegir por
        el modelo, y el resultado saldría plausible y equivocado."""
        salida = self._ejecutar(self._pipeline_de_dos(
            input_map={"Signal": {"routing_signal": "TicketBOW"}}))
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertIn("30 valores", salida["report"]["reason"])


class ElDigestSeComprubaOtraVezAlEjecutarTest(_ConRegistryPropio):
    def test_si_la_entrada_CAMBIA_entre_la_verificacion_y_el_nodo_no_ejecuta(self):
        """La verificación previa mira el registry en el segundo cero. Un
        recibo que se apoyara en eso afirmaría sobre algo que ya no miró."""
        pipeline = self._pipeline_de_uno()
        mapa = mapa_del_registry(self.registro)

        manifiesto = (self.raiz / "registry/entries/feature_extractor/v1/manifest.json")
        datos = json.loads(manifiesto.read_text())
        datos["entry_hash"] = "sha256:" + "0" * 64
        manifiesto.write_text(json.dumps(datos, indent=2) + "\n")

        salida = ejecutar_pipeline(
            pipeline, registry=mapa,               # el mapa de ANTES: pasa la previa
            ejecutores={"modelo": ejecutor_de_modelos(self.registro)})
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertIn("ModeloCambiado", salida["report"]["reason"])

    def test_una_entrada_MANIPULADA_no_ejecuta(self):
        params = self.raiz / "registry/entries/feature_extractor/v1/params.json"
        params.write_text(json.dumps(json.loads(params.read_text())))  # otros bytes
        salida = self._ejecutar(self._pipeline_de_uno())
        self.assertEqual(salida["report"]["status"], "failed")
        # El motivo tiene que ser el de ESTE módulo, con el de P21 dentro:
        # `verify` LEVANTA en vez de devolver `False`, y un `if not
        # verify(...)` dejaría este camino escrito y nunca recorrido.
        self.assertIn("ModeloAlterado", salida["report"]["reason"])
        self.assertIn("integridad", salida["report"]["reason"])
        self.assertIn("content hash mismatch", salida["report"]["reason"])

    def test_saltarse_la_comprobacion_es_una_DECISION_explicita(self):
        """Existe la salida, pero hay que pedirla por su nombre: lo que se
        apaga sin querer no se ve al leer el pipeline."""
        params = self.raiz / "registry/entries/feature_extractor/v1/params.json"
        params.write_text(json.dumps(json.loads(params.read_text())))
        salida = ejecutar_pipeline(
            self._pipeline_de_uno(), registry=mapa_del_registry(self.registro),
            ejecutores={"modelo": ejecutor_de_modelos(self.registro,
                                                     comprobar_integridad=False)})
        self.assertEqual(salida["report"]["status"], "ok")


class UnNombreSinVersionNoSeResuelveTest(_ConRegistryPropio):
    def test_falla_diciendo_por_que_un_nombre_a_secas_no_vale(self):
        pipeline = self._pipeline_de_uno(model="feature_extractor")
        pipeline["nodes"][0]["entry_hash"] = self.fe.entry_hash
        mapa = dict(mapa_del_registry(self.registro))
        mapa["feature_extractor"] = self.fe.entry_hash  # que pase la previa
        salida = ejecutar_pipeline(
            pipeline, registry=mapa,
            ejecutores={"modelo": ejecutor_de_modelos(self.registro)})
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertIn("versión", salida["report"]["reason"])


class ConDosFuentesNoSeEligeUnaTest(_ConRegistryPropio):
    def test_declarar_entrada_Y_recibirla_de_otro_nodo_es_ambiguo(self):
        rc = self.registro.get("route_classifier", "v1")
        pipeline = {
            "pipeline_id": "routing", "version": "1.0", "timeout_s": 60,
            "nodes": [
                {"id": "encode", "model": "feature_extractor@v1",
                 "entry_hash": self.fe.entry_hash, "kind": "modelo",
                 "input": _vector_del_ejemplo()},
                {"id": "clasifica", "model": "route_classifier@v1",
                 "entry_hash": rc.entry_hash, "kind": "modelo",
                 "depends_on": ["encode"], "input": {"otra": "cosa"}},
            ],
        }
        salida = self._ejecutar(pipeline)
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertEqual(salida["report"]["node"], "clasifica")
        self.assertIn("EntradaAmbigua", salida["report"]["reason"])

    def test_un_nodo_sin_entrada_y_sin_dependencias_no_se_ejecuta_con_nada(self):
        pipeline = self._pipeline_de_uno()
        del pipeline["nodes"][0]["input"]
        salida = self._ejecutar(pipeline)
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertIn("no hay con qué ejecutarlo", salida["report"]["reason"])


class LosTiposSeComprubanSiEstanDeclaradosTest(_ConRegistryPropio):
    def _manifiesto(self, **cambios):
        """El manifiesto del ejemplo, con los tipos que este caso quiera.

        **Antes esto solo sabía PONER tipos, y el caso «sin tipos» se
        apoyaba en que el ejemplo publicado no los traía.** Y no los traía
        porque su registry se generó con matrixai **0.21.0** y llevaba
        `input_type: {}`; en cuanto se vuelve a correr el ejemplo, el
        registry se regenera CON tipos y esa prueba caía. Estaba midiendo
        lo rancio del ejemplo, no el camino del código — *un fixture
        describe un modelo que existe*, y aquel describía uno de hace tres
        versiones. Ahora cada caso pone lo suyo y no depende de nada.
        """
        manifiesto = self.raiz / "registry/entries/feature_extractor/v1/manifest.json"
        datos = json.loads(manifiesto.read_text())
        datos.update(cambios)
        manifiesto.write_text(json.dumps(datos, indent=2) + "\n")

    def _con_tipos(self):
        self._manifiesto(input_type={"name": "TicketBOW", "kind": "VECTOR", "size": 30})

    def _sin_tipos(self):
        """Una entrada SIN tipos declarados: existe —las de antes del 83 son
        así— y el motor tiene que decirlo en vez de darla por comprobada."""
        self._manifiesto(input_type={}, output_type={})

    def test_una_entrada_con_OTRO_vector_no_ejecuta(self):
        self._con_tipos()
        salida = self._ejecutar(self._pipeline_de_uno(input={"OtroVector": {}}))
        self.assertEqual(salida["report"]["status"], "failed")
        self.assertIn("TicketBOW", salida["report"]["reason"])

    def test_una_entrada_con_MENOS_campos_dice_las_DOS_tallas(self):
        self._con_tipos()
        salida = self._ejecutar(
            self._pipeline_de_uno(input={"TicketBOW": {"bow_invoice": 1.0}}))
        self.assertIn("30", salida["report"]["reason"])
        self.assertIn("1", salida["report"]["reason"])

    def test_SIN_tipos_declarados_se_DICE_en_vez_de_dar_por_comprobado(self):
        """Afirmar por omisión es el defecto que este repositorio ya
        arregló en `check_composite_program_types`."""
        self._sin_tipos()
        salida = self._ejecutar(self._pipeline_de_uno())
        politicas = salida["trace"]["nodes"][0]["policies"]
        self.assertEqual(politicas[0]["rule_id"], "input_type")
        self.assertIn("sin tipos declarados", politicas[0]["explain"])

    def test_CON_tipos_declarados_la_traza_dice_que_se_contrasto(self):
        self._con_tipos()
        salida = self._ejecutar(self._pipeline_de_uno())
        politicas = salida["trace"]["nodes"][0]["policies"]
        self.assertIn("contrastada", politicas[0]["explain"])
        self.assertEqual(salida["report"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
