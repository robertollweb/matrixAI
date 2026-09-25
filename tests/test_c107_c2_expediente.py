# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C2 — el componente de terceros DENTRO del paquete: manifiesto, BOM y verify.

El criterio de terminado del corte, literal, y cada mitad con sus pruebas:

* **«Un paquete con texto sin `embedding_provider` completo no valida.»** Se
  comprueba por el producto (`verify_package` sobre un paquete en disco), no
  llamando al validador: probar la función no es probar el producto.
* **«El BOM enumera el proveedor con licencia.»** Y el BOM entero se valida
  contra el esquema OFICIAL de CycloneDX 1.6, como ya hace 86-C3: sin eso,
  «emitimos CycloneDX» sería una afirmación sobre nuestro propio JSON.
* **«La tarjeta (106-C4) lo nombra como componente de terceros.»** Esa mitad
  NO se cierra aquí y no se finge: `tarjeta_de_uso.py` y `recibo_de_estudio.py`
  viven en `matrixai-engines`, otro repositorio. Consumen este mismo
  diccionario; el cableado es trabajo de allí y está declarado en el contrato.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from matrixai.export import terceros
from matrixai.export.bom import lo_que_falta, ml_bom
from matrixai.export.inference_spec import InferenceSpecError, build_inference_spec
from matrixai.export.reproduce import (
    REPRODUCE_SCHEMA_VERSION,
    ReproduceManifestError,
    añadir_inventario_de_ficheros,
    manifest_digest,
    write_reproduce_manifest,
)
from matrixai.export.verify import verify_package
from matrixai.text.embeddings.declaracion import componente_de_terceros

_ESQUEMA_CYCLONEDX = Path(__file__).parent / "data" / "cyclonedx-bom-1.6.schema.json"

LONGITUD = {
    "tokens": 512,
    "origen": "libreria_del_proveedor",
    "fuente": "valor por omisión de `max_length` en `model2vec.StaticModel.encode` "
              "(model2vec 0.9.0), leído con `inspect.signature` el 2026-09-14",
    "por_que": "no está en ningún fichero del paquete",
    "fijada_en": "matrixai_engines.embeddings.proveedor_de_texto.LONGITUDES_FIJADAS",
}


def _componente() -> dict:
    return componente_de_terceros("potion-base-8M", longitud=LONGITUD,
                                  ejecutado_por="matrixai-engines")


def _spec_con_texto(componente: dict | None) -> dict:
    """La `inference_spec` de un paquete cuyo modelo recibe columnas de texto.

    Se escribe a mano porque 107-C3 —el corte que convierte una columna de
    texto en `<col>_e0..e{d-1}` dentro del pliegue— todavía no existe: lo que
    se prueba aquí es la PUERTA, y la puerta tiene que cerrar sobre el paquete
    tal como llegará cuando C3 exista.
    """
    spec = {
        "spec_version": 1,
        "input_order": ["edad", "opinion_e0", "opinion_e1"],
        "fields": {
            "edad": {"encoding": "scalar", "range": [0, 120]},
            "opinion": {"encoding": terceros.ENCODING_DE_TEXTO,
                        "columns": ["opinion_e0", "opinion_e1"]},
        },
        "output": {"kind": "regression"},
    }
    if componente is not None:
        spec[terceros.CLAVE_EN_LA_SPEC] = terceros.bloque_para_la_spec(componente)
    return spec


def _paquete(*, componente: dict | None, spec: dict | None) -> Path:
    """Un paquete honesto en disco, con su manifiesto y su inventario de verdad."""
    d = Path(tempfile.mkdtemp())
    (d / "model.mxai").write_text("NETWORK N\n  DENSE 4\n", encoding="utf-8")
    if spec is not None:
        (d / "inference_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    write_reproduce_manifest(d, weights_source="trained", embedding_provider=componente)
    añadir_inventario_de_ficheros(d)
    return d


def _con_texto() -> Path:
    componente = _componente()
    return _paquete(componente=componente, spec=_spec_con_texto(componente))


def _reescribir(bundle: Path, retoque, *, resellar: bool = True) -> Path:
    """Cambia el `reproduce.json` de un paquete y vuelve a sellarlo.

    **Resellar es lo normal y no es hacer trampa**: quien fabrica un paquete
    calcula ese `manifest_sha256` sin esfuerzo —lo dice el propio `verify.py`—
    y `reproduce.json` está fuera del inventario de ficheros, así que un
    manifiesto con un componente a medias puede ser perfectamente coherente
    consigo mismo. Es exactamente el caso en el que la puerta de 107-C2 es la
    única defensa: sin resellar, lo que salta antes es el digest y la puerta no
    llega a probarse.

    Con `resellar=False` se prueba lo contrario: que el sello cubre el
    componente.
    """
    m = json.loads((bundle / "reproduce.json").read_text(encoding="utf-8"))
    retoque(m)
    if resellar:
        m.pop("manifest_sha256", None)
        m["manifest_sha256"] = manifest_digest(m)
    (bundle / "reproduce.json").write_text(json.dumps(m, ensure_ascii=False),
                                           encoding="utf-8")
    return bundle


class TestElManifiestoLlevaElComponenteDeTerceros(unittest.TestCase):
    def test_un_paquete_sin_pesos_ajenos_lo_declara_como_null(self):
        """Un paquete que calla no dice «no uso pesos ajenos», dice nada."""
        m = json.loads((_paquete(componente=None, spec=None) / "reproduce.json")
                       .read_text(encoding="utf-8"))
        self.assertIn("embedding_provider", m)
        self.assertIsNone(m["embedding_provider"])

    def test_el_componente_viaja_entero_en_reproduce_json(self):
        m = json.loads((_con_texto() / "reproduce.json").read_text(encoding="utf-8"))
        self.assertEqual(terceros.validar(m["embedding_provider"]), [])
        self.assertEqual(m["embedding_provider"]["id"], "potion-base-8M")

    def test_un_componente_a_medias_no_se_escribe_y_dice_que_le_falta(self):
        roto = copy.deepcopy(_componente())
        roto.pop("digest_del_tokenizer")
        with self.assertRaises(ReproduceManifestError) as e:
            _paquete(componente=roto, spec=None)
        self.assertIn("digest_del_tokenizer", str(e.exception))

    def test_el_manifest_sha256_cubre_el_componente(self):
        """Cambiar un digest del proveedor dentro del manifiesto ya publicado
        rompe el sello: el componente no es un adjunto, es parte del papel."""
        d = _con_texto()

        def retocar(m):
            m["embedding_provider"]["digest_de_los_pesos"]["digest"] = "0" * 64

        etapa = verify_package(
            _reescribir(d, retocar, resellar=False))["stages"]["manifest"]
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("manifest_sha256", json.dumps(etapa))


class TestElBomEnumeraElProveedorConSuLicencia(unittest.TestCase):
    """«El BOM enumera el proveedor con licencia» — criterio de terminado."""

    def setUp(self):
        self.bom = ml_bom(_con_texto())
        self.proveedor = next(
            c for c in self.bom["components"]
            if c["bom-ref"].startswith("embedding-provider:"))

    def test_el_bom_entero_valida_contra_el_esquema_oficial(self):
        try:
            from jsonschema import Draft7Validator
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema no está instalado")
        from jsonschema import RefResolver
        esquema = json.loads(_ESQUEMA_CYCLONEDX.read_text(encoding="utf-8"))
        # SIN RED (2026-09-25): el esquema oficial referencia `spdx.schema.json` (y
        # `jsf-0.82.schema.json`) por ruta relativa a su `$id`, y `jsonschema` los iba a
        # buscar a cyclonedx.org. La suite nocturna del 25-09 dio un rojo por un
        # `MaxRetryError` de madrugada que no tenía nada que ver con el producto. Van
        # guardados al lado (CycloneDX/specification, etiqueta 1.6, Apache-2.0) y se
        # resuelven desde aquí.
        almacen = {}
        for nombre in ("spdx.schema.json", "jsf-0.82.schema.json"):
            referenciado = json.loads((_ESQUEMA_CYCLONEDX.parent / nombre).read_text(encoding="utf-8"))
            almacen[referenciado["$id"]] = referenciado
        resolutor = RefResolver.from_schema(esquema, store=almacen)
        errores = [f"{list(e.path)}: {e.message}"
                   for e in Draft7Validator(esquema, resolver=resolutor).iter_errors(self.bom)]
        self.assertEqual(errores, [])

    def test_es_un_machine_learning_model_con_su_licencia_spdx(self):
        self.assertEqual(self.proveedor["type"], "machine-learning-model")
        self.assertEqual(self.proveedor["licenses"], [{"license": {"id": "MIT"}}])
        self.assertEqual(self.proveedor["name"], "potion-base-8M")

    def test_la_version_del_componente_es_EL_CHECKPOINT(self):
        """Licencia por checkpoint concreto (invariante 5): un `1.0` inventado
        aquí diría que la licencia vale para cualquier revisión."""
        self.assertEqual(self.proveedor["version"],
                         _componente()["revision"])

    def test_las_referencias_externas_llevan_origen_y_licencia(self):
        por_tipo = {r["type"]: r["url"] for r in self.proveedor["externalReferences"]}
        self.assertIn("distribution", por_tipo)
        self.assertIn("license", por_tipo)
        # El origen se enlaza EN SU REVISIÓN, no en `main`.
        self.assertIn(_componente()["revision"], por_tipo["distribution"])

    def test_el_modelo_DEPENDE_del_proveedor_y_se_ve(self):
        """Una lista de componentes sueltos no dice quién usa a quién."""
        refs = {d["ref"]: d["dependsOn"] for d in self.bom["dependencies"]}
        modelo = self.bom["metadata"]["component"]["bom-ref"]
        self.assertIn(self.proveedor["bom-ref"], refs[modelo])

    def test_el_digest_del_tokenizer_no_se_disfraza_de_SHA1(self):
        """El origen publica el `oid` de git para los ficheros pequeños, que es
        el sha1 de `blob <n>\\0` + contenido — **no** el sha1 del fichero. La
        lista `hash-alg` de CycloneDX solo admite `SHA-1`, así que meterlo ahí
        afirmaría que un `sha1sum` del `tokenizer.json` da ese número, y no lo
        da. Viaja como propiedad, con su algoritmo escrito al lado."""
        componente = _componente()
        self.assertEqual(componente["digest_del_tokenizer"]["algoritmo"], "git-blob-sha1")
        propiedades = {p["name"]: p["value"] for p in self.proveedor["properties"]}
        self.assertEqual(propiedades["matrixai:embedding.tokenizer.digest"],
                         componente["digest_del_tokenizer"]["digest"])
        self.assertEqual(propiedades["matrixai:embedding.tokenizer.digest_algorithm"],
                         "git-blob-sha1")
        for h in self.proveedor.get("hashes", []):
            with self.subTest(alg=h["alg"]):
                self.assertNotEqual(h["content"],
                                    componente["digest_del_tokenizer"]["digest"])

    def test_los_pesos_si_van_en_hashes_porque_su_digest_ES_un_sha256(self):
        self.assertEqual(self.proveedor["hashes"],
                         [{"alg": "SHA-256",
                           "content": _componente()["digest_de_los_pesos"]["digest"]}])

    def test_las_tres_cosas_del_invariante_6_estan_en_el_bom(self):
        propiedades = {p["name"]: p["value"] for p in self.proveedor["properties"]}
        self.assertIn("matrixai:embedding.tokenizer.digest", propiedades)
        self.assertEqual(propiedades["matrixai:embedding.max_tokens"], "512")
        self.assertEqual(propiedades["matrixai:embedding.max_tokens.origin"],
                         "libreria_del_proveedor")
        self.assertIn("model2vec", propiedades["matrixai:embedding.max_tokens.source"])

    def test_la_opacidad_y_el_truncado_van_donde_los_busca_quien_lee_un_bom(self):
        limitaciones = self.proveedor["modelCard"]["considerations"]["technicalLimitations"]
        junto = " ".join(limitaciones)
        self.assertIn("no auditados", junto)
        self.assertIn("truncated", junto)
        # Y el idioma que NO cubre, con su veredicto.
        self.assertIn("language es: no_cubierto", junto)

    def test_el_proveedor_queda_marcado_en_el_modelo(self):
        propiedades = {p["name"]: p["value"]
                       for p in self.bom["metadata"]["component"]["properties"]}
        self.assertEqual(propiedades["matrixai:embedding_provider"], "potion-base-8M")


class TestUnComponenteAMediasNoSeEnumeraAMedias(unittest.TestCase):
    """Un componente de terceros sin licencia se lee como si el proveedor
    estuviera declarado, y eso es peor que no enumerarlo: media verdad
    tranquilizadora. O está entero, o se dice qué le falta."""

    def setUp(self):
        roto = copy.deepcopy(_componente())
        roto["licencia"].pop("spdx")
        self.bundle = _reescribir(_con_texto(),
                                  lambda m: m.__setitem__("embedding_provider", roto))
        self.bom = ml_bom(self.bundle)

    def test_no_aparece_ningun_componente_de_proveedor(self):
        refs = [c["bom-ref"] for c in self.bom["components"]]
        self.assertEqual([r for r in refs if r.startswith("embedding-provider:")], [])

    def test_lo_que_falta_dice_QUE_campo_falta(self):
        falta = " ".join(lo_que_falta(self.bom))
        self.assertIn("third-party embedding provider", falta)
        self.assertIn("licencia.spdx", falta)

    def test_y_queda_como_aviso_en_las_limitaciones_del_modelo(self):
        limitaciones = (self.bom["metadata"]["component"]["modelCard"]["considerations"]
                        ["technicalLimitations"])
        self.assertIn("incomplete", " ".join(limitaciones))


class TestUnPaqueteConTextoSinProveedorNoValida(unittest.TestCase):
    """El criterio de terminado de 107-C2, comprobado por el producto."""

    def _manifest(self, bundle: Path, locale: str = "es") -> dict:
        return verify_package(bundle, locale=locale)["stages"]["manifest"]

    def test_el_paquete_honesto_con_texto_pasa(self):
        etapa = self._manifest(_con_texto())
        self.assertEqual(etapa["status"], "PASS", etapa.get("reason"))

    def test_sin_proveedor_en_el_manifiesto_FALLA(self):
        bundle = _paquete(componente=None, spec=_spec_con_texto(None))
        etapa = self._manifest(bundle)
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("no declara ninguno", etapa["reason"])

    def test_con_el_proveedor_a_medias_FALLA_y_nombra_los_campos(self):
        """«Está incompleto» obliga a adivinar cuál; esto dice cuál."""
        def quitar(m):
            m["embedding_provider"].pop("licencia")

        etapa = self._manifest(_reescribir(_con_texto(), quitar))
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("licencia", etapa["reason"])
        self.assertIn("licencia", etapa["embedding_provider"]["missing"])

    def test_la_spec_sin_su_bloque_de_proveedor_FALLA(self):
        """Sin el bloque en la spec, `predict.py` no puede reproducir el mismo
        vector fuera — que es el criterio de 107-C4."""
        bundle = _paquete(componente=_componente(), spec=_spec_con_texto(None))
        etapa = self._manifest(bundle)
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("inference_spec.json", etapa["reason"])

    def test_si_el_TOKENIZER_de_la_spec_no_es_el_del_manifiesto_FALLA(self):
        """Mismos pesos con otro tokenizer es otro vector: el tokenizer es
        parte del modelo (invariante 6)."""
        componente = _componente()
        spec = _spec_con_texto(componente)
        spec[terceros.CLAVE_EN_LA_SPEC]["digest_del_tokenizer"]["digest"] = "f" * 40
        etapa = self._manifest(_paquete(componente=componente, spec=spec))
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("digest_del_tokenizer", etapa["reason"])

    def test_si_el_TRUNCADO_de_la_spec_no_es_el_del_manifiesto_FALLA(self):
        """La tercera cosa. Con 512 el coseno contra la librería del proveedor
        es 0,658 y sin truncar 1,000: no es un matiz de configuración."""
        componente = _componente()
        spec = _spec_con_texto(componente)
        spec[terceros.CLAVE_EN_LA_SPEC]["longitud_maxima"]["tokens"] = 128
        etapa = self._manifest(_paquete(componente=componente, spec=spec))
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("longitud_maxima_tokens", etapa["reason"])

    def test_si_la_spec_declara_OTRO_proveedor_FALLA(self):
        componente = _componente()
        spec = _spec_con_texto(componente)
        spec[terceros.CLAVE_EN_LA_SPEC]["id"] = "potion-base-2M"
        etapa = self._manifest(_paquete(componente=componente, spec=spec))
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("id", etapa["reason"])

    def test_el_motivo_se_redacta_en_los_DOS_idiomas(self):
        """Lo que redacta el core se traduce en el core, no al pintarlo."""
        bundle = _paquete(componente=None, spec=_spec_con_texto(None))
        en_es = self._manifest(bundle, "es")["reason"]
        en_en = self._manifest(bundle, "en")["reason"]
        self.assertNotEqual(en_es, en_en)
        self.assertIn("no declara ninguno", en_es)
        self.assertIn("declares none", en_en)
        # Y lo que es un DATO no cambia de idioma.
        self.assertIn("inference_spec.json", en_es)
        self.assertIn("inference_spec.json", en_en)

    def test_borrar_la_spec_para_esquivar_la_puerta_no_cuela(self):
        """`_leer_la_spec` trata una spec ilegible como «no se puede saber» y
        no como fallo, a propósito. Lo que impide que eso sea un agujero es el
        inventario: cubre TODOS los ficheros del paquete con su sha256, así que
        borrarla o corromperla sale `FAIL` de todos modos."""
        for como in ("borrada", "corrompida"):
            with self.subTest(como=como):
                bundle = _con_texto()
                if como == "borrada":
                    (bundle / "inference_spec.json").unlink()
                else:
                    (bundle / "inference_spec.json").write_text("{roto", encoding="utf-8")
                self.assertEqual(self._manifest(bundle)["status"], "FAIL")

    def test_la_puerta_cierra_con_schema_version_1_1(self):
        """107-C2 añade `embedding_provider` y NO sube la versión de esquema, a
        sabiendas: `verify` ya acepta 1.0, 1.1 y 1.2, así que subirla no habría
        cerrado nada que esto no cierre. La prueba es que la puerta muerde con
        la versión que el manifiesto declara hoy."""
        bundle = _paquete(componente=None, spec=_spec_con_texto(None))
        m = json.loads((bundle / "reproduce.json").read_text(encoding="utf-8"))
        self.assertEqual(m["schema_version"], REPRODUCE_SCHEMA_VERSION)
        self.assertEqual(self._manifest(bundle)["status"], "FAIL")

    def test_las_etapas_siguen_siendo_CUATRO(self):
        """Los nombres de etapa son vocabulario público: quien encadena
        `verify && desplegar` depende de ellos. Esto no añade una quinta."""
        self.assertEqual(set(verify_package(_con_texto())["stages"]),
                         {"manifest", "R1", "training", "R3"})

    def test_un_paquete_SIN_texto_no_se_ve_afectado(self):
        """Arreglar un sesgo puede crear el contrario: hay que probar los dos
        lados. Un paquete tabular de siempre sigue pasando igual."""
        etapa = self._manifest(_paquete(componente=None, spec=None))
        self.assertEqual(etapa["status"], "PASS", etapa.get("reason"))

    def test_un_proveedor_declarado_a_medias_falla_aunque_no_haya_texto(self):
        """Declarar unos pesos ajenos y no decir cuáles no se arregla porque el
        paquete no lleve columnas de texto."""
        bundle = _paquete(componente=None, spec=None)

        def meter(m):
            roto = copy.deepcopy(_componente())
            roto.pop("opacidad")
            m["embedding_provider"] = roto

        etapa = self._manifest(_reescribir(bundle, meter))
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("opacidad", etapa["reason"])


class TestLaSpecLlevaLasTresCosas(unittest.TestCase):
    """«Las tres viajan en el `inference_spec`» (107, invariante 6)."""

    def _programa(self):
        from matrixai.ir.schema import MatrixAIProgram, SequenceSpec, VectorSpec
        vector = VectorSpec(name="V", size=2, fields=["a", "b"], field_types={})
        return MatrixAIProgram(project="T", vectors=[vector], networks=[], sequences=[]), \
            SequenceSpec

    def _ps(self):
        from matrixai.parameters.store import ParameterSet
        return ParameterSet(parameter_set_id="PS", model_hash="mxai_x",
                            parameter_schema_hash="p_x", parameters={})

    def _export(self):
        from matrixai.export.onnx_exporter import OnnxExportResult
        return OnnxExportResult(
            output_path="model.onnx", opset_version=17, model_hash="mxai_x",
            parameter_set_id="PS", parameter_schema_hash="p_x", input_name="features",
            input_shape=[-1, 2], output_name="out", output_shape=[-1],
            exported_functions=["net"], labels=[])

    def test_la_spec_lleva_el_bloque_con_las_tres_cosas(self):
        programa, _ = self._programa()
        spec = build_inference_spec(programa, self._ps(), self._export(),
                                    embedding_provider=_componente())
        bloque = spec[terceros.CLAVE_EN_LA_SPEC]
        self.assertEqual(terceros.las_tres_cosas(bloque),
                         terceros.las_tres_cosas(_componente()))

    def test_el_bloque_de_la_spec_no_arrastra_el_expediente(self):
        """Es un subconjunto, no un resumen: lo que se AUDITA —licencia,
        cobertura de idioma, opacidad— vive en el manifiesto y en el BOM. Que
        la spec lo repitiera sería el segundo sitio declarando lo mismo."""
        bloque = terceros.bloque_para_la_spec(_componente())
        for clave in ("licencia", "idiomas_medidos", "opacidad", "lo_que_no_se_sabe"):
            with self.subTest(clave=clave):
                self.assertNotIn(clave, bloque)
        self.assertIn("sin_el_proveedor_no_se_predice", bloque)

    def test_una_declaracion_a_medias_no_produce_spec(self):
        roto = copy.deepcopy(_componente())
        roto["longitud_maxima"].pop("tokens")
        programa, _ = self._programa()
        with self.assertRaises(InferenceSpecError) as e:
            build_inference_spec(programa, self._ps(), self._export(),
                                 embedding_provider=roto)
        self.assertIn("longitud_maxima.tokens", str(e.exception))

    def test_un_proveedor_con_entrada_SEQUENCE_se_rechaza(self):
        """Una entrada SEQUENCE es el transformer PROPIO por bytes, con su
        tokenizador embebido: ahí no hay pesos de terceros que declarar. Una
        spec que dijera las dos cosas a la vez no se ve.

        **El aserto mira la frase de ESTE guardián y no la palabra «SEQUENCE»,
        y es por un sabotaje que salió verde.** Con un programa de secuencia
        cualquiera, `_build_sequence_inference_spec` ya revienta por su cuenta
        —«SEQUENCE input without a BLOCK TRANSFORMER network»—, así que un
        aserto por esa palabra lo pasaba también con el guardián quitado: un
        banco sin dientes. Por eso el programa de aquí SÍ lleva su bloque
        transformer, y lo que se exige es el motivo propio.
        """
        from matrixai.ir.schema import (
            MatrixAIProgram, NetworkSpec, SequenceSpec, VectorSpec)
        red = NetworkSpec(name="N", input="S", layers=[], output="y",
                          output_type_str="Real", kind="composite_network",
                          transformer_blocks=[object()])
        programa = MatrixAIProgram(
            project="T", vectors=[VectorSpec(name="V", size=1, fields=["a"],
                                             field_types={})],
            networks=[red], sequences=[SequenceSpec(name="S", length=8,
                                                    vocab_size=259)])
        with self.assertRaises(InferenceSpecError) as e:
            build_inference_spec(programa, self._ps(), self._export(),
                                 embedding_provider=_componente())
        self.assertIn("no un proveedor de terceros", str(e.exception))
        self.assertIn("107-C3", str(e.exception))

    def test_sin_proveedor_la_spec_no_cambia_nada(self):
        programa, _ = self._programa()
        spec = build_inference_spec(programa, self._ps(), self._export())
        self.assertNotIn(terceros.CLAVE_EN_LA_SPEC, spec)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
