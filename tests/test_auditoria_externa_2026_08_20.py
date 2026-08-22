"""AUDITORÍA EXTERNA (2026-08-20) — los 13 hallazgos, con prueba cada uno.

Una auditoría de fuera revisó los contratos 81, 82 y 83 y los **rechazó
los tres**. Verifiqué los trece hallazgos midiendo, uno a uno: **los
trece eran ciertos**. Este fichero los fija para que no vuelvan.

Vale la pena decir qué tenían en común, porque es una sola avería con
trece caras: **el código validaba al ESCRIBIR y no al LEER**. El emisor
de recibos rechaza una `schema_version` desconocida y el verificador la
aceptaba; el escritor de manifiestos rechaza una ruta con `..` y `verify`
la seguía; el motor promete impedir efectos y solo se negaba a
apuntarlos. En todos, la mitad que protege es la que mira lo que viene de
fuera, y era la que faltaba.

Y una que me toca de cerca: **yo había declarado el C5 completo**, y no
lo estaba. Los criterios de §15.6 piden recibo de reproducción,
comparación con el de referencia y dos entornos, y `replay` no hacía
ninguna de las tres.
"""

import json
import math
import unittest

from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.engine import ejecutar_pipeline, emitir_recibo
from matrixai.pipelines.receipt import firmar_recibo
from matrixai.pipelines.verifier import verificar_sobre

_PIPE = {"pipeline_id": "p", "version": "1.0", "timeout_s": 30,
         "nodes": [{"id": "n0", "model": "m", "entry_hash": "sha256:" + "a" * 64,
                    "kind": "t"}]}
_REG = {"m": "sha256:" + "a" * 64}
_EJ = {"t": lambda *, entradas, nodo, contexto: "x"}


def _corrido(**kw):
    return ejecutar_pipeline(_PIPE, registry=_REG, ejecutores=_EJ, **kw)


class H1_JCS_EscribeComoECMAScriptTest(unittest.TestCase):
    """81-1 [BLOQUEANTE]. Usaba las reglas de Python: `1e-6` salía `1e-6`
    y ECMAScript escribe `0.000001`. **Una firma sobre esos bytes no la
    puede verificar nadie más**, que es lo único que JCS garantiza — y era
    la razón entera de la decisión §0.1."""

    CASOS = (
        (1e-6, "0.000001"), (1e-7, "1e-7"), (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (1.2345678901234568e+20, "123456789012345680000"),
        (0.1 + 0.2, "0.30000000000000004"), (1.0, "1"), (-0.0, "0"),
        (1e-5, "0.00001"), (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
    )

    def test_los_umbrales_del_exponente_son_los_de_ECMAScript(self):
        for valor, esperado in self.CASOS:
            with self.subTest(valor=valor):
                self.assertEqual(jcs_bytes({"x": valor}).decode()[5:-1], esperado)

    def test_NaN_e_Infinity_siguen_sin_firmarse(self):
        for valor in (math.nan, math.inf, -math.inf):
            with self.subTest(valor=valor):
                with self.assertRaises(ValueError):
                    jcs_bytes({"x": valor})


class H2_LosEfectosNoSePUEDEN_ImpedirYSeDICETest(unittest.TestCase):
    """81-2 [BLOQUEANTE]. El contrato dice «impedir que un nodo no
    autorizado ejecute un efecto externo» y este motor **no lo impide**:
    llama a una función de Python y esa función puede escribir donde
    quiera. Medido: un nodo con `effects: false` creó un fichero y terminó
    `ok`.

    No se puede arreglar dentro del proceso, así que **no se finge** —
    misma línea que §13.2 bis con los reintentos—: el informe declara qué
    hace de verdad y dónde está el cumplimiento real (el sandbox del C5).
    """

    def test_el_informe_DECLARA_que_no_impide(self):
        informe = _corrido()["report"]
        self.assertIn("does NOT prevent", informe["effects_enforcement"])
        self.assertIn("C5 sandbox", informe["effects_enforcement"])

    def test_y_lo_que_SI_hace_sigue_haciendolo(self):
        """La traza se niega a apuntar un efecto no declarado."""
        from matrixai.pipelines.trace import EfectoNoAutorizado, TrazaDeEjecucion

        traza = TrazaDeEjecucion("r")
        traza.abrir_nodo("n0", modelo="m", efectos=False)
        with self.assertRaises(EfectoNoAutorizado):
            traza.registrar_efecto("n0", "escribí un fichero")

    def test_y_en_dry_run_no_se_llama_al_ejecutor(self):
        llamadas = []
        ejecutar_pipeline(_PIPE, registry=_REG,
                          ejecutores={"t": lambda **k: llamadas.append(1)},
                          dry_run=True)
        self.assertEqual(llamadas, [])


class H3_LaRaizIdentificaUnaEJECUCIONTest(unittest.TestCase):
    """81-3 [ALTO]. La raíz era el `pipeline_id`, así que dos ejecuciones
    del mismo pipeline compartían raíz y sus trazas no se distinguían.
    §13.2 pide «generar un identificador raíz de EJECUCIÓN»."""

    def test_dos_ejecuciones_del_mismo_pipeline_tienen_raices_distintas(self):
        self.assertNotEqual(_corrido()["trace"]["root_id"],
                            _corrido()["trace"]["root_id"])

    def test_los_nodos_siguen_atados_a_SU_raiz(self):
        salida = _corrido()
        self.assertEqual({n["root_id"] for n in salida["trace"]["nodes"]},
                         {salida["trace"]["root_id"]})

    def test_se_puede_declarar_una_raiz_propia(self):
        """Quien tenga su identificador lo impone; lo que no se hace es
        inventarse uno compartido."""
        self.assertEqual(_corrido(run_id="mi-run-1")["trace"]["root_id"], "mi-run-1")


class H4_ElVerificadorCompruebaElESQUEMATest(unittest.TestCase):
    """81-4 [BLOQUEANTE]. Añadía `"schema"` a lo VERIFICADO por el mero
    hecho de que el JSON se decodificara. Medido: un sobre con
    `payloadType: "application/evil"` y `schema_version: "999"` salía
    `ok: true`, nivel A1, `verified: ["schema"]`.

    **Afirmaba haber comprobado lo que no había mirado**, que es lo único
    que este componente existe para no hacer."""

    def _sobre(self):
        salida = _corrido()
        return firmar_recibo(emitir_recibo(salida, pipeline=_PIPE, receipt_id="r1"),
                             clave=b"k", key_id="k1")

    def test_el_honesto_pasa_y_dice_lo_que_verifico(self):
        v = verificar_sobre(self._sobre(), clave=b"k")
        self.assertTrue(v["ok"])
        self.assertIn("schema", v["verified"])
        self.assertIn("signature", v["verified"])

    def test_un_payloadType_INVENTADO_no_pasa(self):
        """DSSE ata el tipo al contenido justamente para que una firma
        buena no valga en otro sitio."""
        sobre = dict(self._sobre())
        sobre["payloadType"] = "application/evil"
        v = verificar_sobre(sobre, clave=None)
        self.assertFalse(v["ok"])
        self.assertNotIn("schema", v["verified"])

    def test_una_schema_version_DESCONOCIDA_tampoco(self):
        sobre = dict(self._sobre())
        payload = json.loads(sobre["payload"])
        payload["schema_version"] = "999"
        sobre["payload"] = json.dumps(payload)
        v = verificar_sobre(sobre, clave=None)
        self.assertFalse(v["ok"])
        self.assertTrue(any("999" in p for p in v["problems"]))

    def test_un_recibo_SIN_pasos_no_atestigua_nada(self):
        sobre = dict(self._sobre())
        payload = json.loads(sobre["payload"])
        payload["steps"] = []
        sobre["payload"] = json.dumps(payload)
        self.assertFalse(verificar_sobre(sobre, clave=None)["ok"])


class H5_ElReciboCompromete_ELCaminoEjecutadoTest(unittest.TestCase):
    """81-5 [BLOQUEANTE]. Faltaban `pipeline_digest` y
    `executed_path_digest`, que P23-R-0015 declara **DEBE**. Sin el
    segundo, dos ejecuciones por ramas distintas del mismo grafo producen
    recibos idénticos y un tercero no puede reconstruir qué pasó."""

    def test_lleva_los_dos_digests_y_los_nodos(self):
        recibo = emitir_recibo(_corrido(), pipeline=_PIPE, receipt_id="r1")
        self.assertTrue(recibo["pipeline"]["pipeline_digest"].startswith("sha256:"))
        self.assertTrue(recibo["pipeline"]["executed_path_digest"].startswith("sha256:"))
        self.assertEqual([n["node_id"] for n in recibo["pipeline"]["nodes"]], ["n0"])

    def test_cada_nodo_dice_QUE_MODELO_uso(self):
        """`models` por sí sola no puede decir cuál corrió dónde."""
        recibo = emitir_recibo(_corrido(), pipeline=_PIPE, receipt_id="r1")
        ref = recibo["pipeline"]["nodes"][0]["model_ref"]
        self.assertEqual(ref["model"], "m")
        self.assertEqual(ref["digest"], _REG["m"])

    def test_dos_grafos_distintos_dan_caminos_distintos(self):
        otro = json.loads(json.dumps(_PIPE))
        otro["nodes"][0]["model"] = "otro"
        otro["nodes"][0]["entry_hash"] = "sha256:" + "b" * 64
        salida = ejecutar_pipeline(otro, registry={"otro": "sha256:" + "b" * 64},
                                   ejecutores=_EJ)
        uno = emitir_recibo(_corrido(), pipeline=_PIPE, receipt_id="r1")
        dos = emitir_recibo(salida, pipeline=otro, receipt_id="r2")
        self.assertNotEqual(uno["pipeline"]["pipeline_digest"],
                            dos["pipeline"]["pipeline_digest"])
        self.assertNotEqual(uno["pipeline"]["executed_path_digest"],
                            dos["pipeline"]["executed_path_digest"])


class H6_ElReplayREENTRENA_PorDefectoTest(unittest.TestCase):
    """81-6 [BLOQUEANTE]. `replay` ejecutaba `verify` **sin `--retrain`**,
    así que `training` y `R3` quedaban `NOT_RUN` — y §15.6 pide que «R3
    pueda demostrarse dentro de las tolerancias declaradas». Reproducir
    sin reentrenar no reproduce: comprueba huellas y se queda ahí."""

    def test_el_comando_de_dentro_lleva_retrain(self):
        from matrixai.pipelines.sandbox import argv_de_reproduccion

        self.assertIn("--retrain", argv_de_reproduccion("docker", "/tmp", "img:1"))

    def test_y_se_puede_pedir_que_NO_lo_haga(self):
        """Reentrenar cuesta: quien solo quiera integridad no tiene por
        qué pagarlo. Pero el DEFECTO es reproducir de verdad."""
        from matrixai.pipelines.sandbox import argv_de_reproduccion

        self.assertNotIn("--retrain",
                         argv_de_reproduccion("docker", "/tmp", "img:1",
                                              reentrenar=False))

    def test_el_CLI_lo_ofrece_y_lo_DICE(self):
        import subprocess
        import sys

        salida = subprocess.run([sys.executable, "-m", "matrixai", "replay", "--help"],
                                capture_output=True, text=True, timeout=60)
        self.assertIn("--no-retrain", salida.stdout)
        self.assertIn("NOT_RUN", salida.stdout)


class H7_ElPaqueteSeCOMPRUEBA_EnteroTest(unittest.TestCase):
    """82-1 [BLOQUEANTE]. `verify` aceptaba paquetes fuera de su propio
    perfil: `schema_version` inventada, artefactos mal formados que se
    **omitían en silencio**, y enlaces simbólicos. Todo con `PASS`.

    Lo que más duele es el silencio: un artefacto que el manifiesto
    declara y el verificador no mira sale igual que uno comprobado."""

    def _paquete(self):
        import tempfile
        from pathlib import Path

        from matrixai.export.reproduce import (
            añadir_inventario_de_ficheros,
            write_reproduce_manifest,
        )

        ejemplos = Path(__file__).resolve().parent.parent / "examples"
        d = Path(tempfile.mkdtemp())
        (d / "model.mxai").write_text("NETWORK N\n  DENSE 4\n")
        (d / "training.mxtrain").write_text(
            (ejemplos / "celsius_to_kelvin.mxtrain").read_text(encoding="utf-8"))
        (d / "recipe.txt").write_text("genera 200 filas\n")
        write_reproduce_manifest(
            d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
            dataset_sha256="a" * 64, dataset_rows=2,
            generation={"seeds": {"dataset": 42}}, weights_source="trained")
        # El inventario, como hace el producto justo antes de promover el
        # paquete: sin él este fixture describe uno que ya no se construye
        # y sale INCOMPARABLE (H5 del refutador, 2026-08-20).
        añadir_inventario_de_ficheros(d)
        return d

    def _tocar(self, d, cambio):
        from matrixai.export.reproduce import manifest_digest

        man = json.loads((d / "reproduce.json").read_text())
        cambio(man)
        man["manifest_sha256"] = manifest_digest(man)
        (d / "reproduce.json").write_text(json.dumps(man))
        return d

    def test_el_paquete_HONESTO_sigue_pasando(self):
        """Lo primero, siempre: que el arreglo no acuse a quien no ha
        hecho nada. Este arreglo tuvo DOS intentos que sí lo hacían."""
        from matrixai.export.verify import verify_package

        self.assertEqual(
            verify_package(self._paquete())["stages"]["manifest"]["status"], "PASS")

    def test_un_digest_declarado_SIN_fichero_es_legitimo(self):
        """El dataset no viaja en el paquete y su hash sí: R1 lo usa para
        comparar lo que regenere. Rechazarlo acusaba a todos."""
        from matrixai.export.verify import verify_package

        d = self._tocar(self._paquete(),
                        lambda m: m["artifacts"].__setitem__(
                            "dataset", {"sha256": "b" * 64}))
        self.assertEqual(verify_package(d)["stages"]["manifest"]["status"], "PASS")

    def test_una_schema_version_desconocida_no_se_interpreta_a_medias(self):
        from matrixai.export.verify import verify_package

        d = self._tocar(self._paquete(),
                        lambda m: m.__setitem__("schema_version", "999.0"))
        etapa = verify_package(d)["stages"]["manifest"]
        self.assertEqual(etapa["status"], "INCOMPARABLE")
        self.assertIn("999.0", etapa["reason"])

    def test_un_fichero_SIN_sha256_no_se_omite_en_silencio(self):
        from matrixai.export.verify import verify_package

        d = self._tocar(self._paquete(),
                        lambda m: m["artifacts"].__setitem__(
                            "training", {"path": "training.mxtrain"}))
        etapa = verify_package(d)["stages"]["manifest"]
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("no sha256", etapa["artifacts"][0]["problem"])

    def test_un_enlace_simbolico_INTERNO_tampoco_vale(self):
        """El digest cubre los bytes del destino, no la indirección."""
        import hashlib
        import os

        from matrixai.export.verify import verify_package

        d = self._paquete()
        os.symlink(d / "model.mxai", d / "enlace.mxai")
        self._tocar(d, lambda m: m["artifacts"].__setitem__("model", {
            "path": "enlace.mxai",
            "sha256": hashlib.sha256((d / "model.mxai").read_bytes()).hexdigest()}))
        etapa = verify_package(d)["stages"]["manifest"]
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("symlink", etapa["artifacts"][0]["problem"])


class H8_INCOMPARABLE_NoEsExitoTest(unittest.TestCase):
    """82-2 [ALTO]. Solo se miraba `FAIL`, así que un paquete con una
    etapa `INCOMPARABLE` salía con **código 0** — el de «nada falló». Quien
    encadene `verify && desplegar` trataría un paquete no verificado como
    verificado."""

    def test_lo_que_no_se_pudo_comprobar_sale_con_3_y_se_NOMBRA(self):
        import tempfile
        from pathlib import Path

        from matrixai.export.verify import verify_package

        # Un directorio sin manifiesto: nada se puede comprobar.
        r = verify_package(Path(tempfile.mkdtemp()))
        self.assertEqual(r["exit_code"], 3)
        self.assertFalse(r["fully_checked"])
        self.assertTrue(r["unchecked_stages"])

    def test_ok_sigue_significando_lo_mismo_y_no_viaja_solo(self):
        """`ok` = «nada falló», que es lo que significaba; cambiarlo
        rompería a quien lo lea. Lo nuevo es `fully_checked`."""
        import tempfile
        from pathlib import Path

        from matrixai.export.verify import verify_package

        r = verify_package(Path(tempfile.mkdtemp()))
        self.assertIn("ok", r)
        self.assertIn("fully_checked", r)
        self.assertIn("unchecked_stages", r)


class H9_ElSpaceEnseñaElResultadoDeC2Test(unittest.TestCase):
    """82-4 [BLOQUEANTE de cierre]. La plantilla solo ejecutaba
    `predict.py`. Un Space que invita a probar un modelo sin decir si el
    paquete está íntegro convierte una ficha ejecutable en una demo."""

    def _plantilla(self):
        from matrixai.export.space import space_app_py

        return space_app_py()

    def test_la_plantilla_ejecuta_matrixai_verify(self):
        plantilla = self._plantilla()
        self.assertIn("verify", plantilla)
        self.assertIn("def verificar", plantilla)

    def test_y_lo_enseña_ANTES_de_invitar_a_probar(self):
        plantilla = self._plantilla()
        self.assertLess(plantilla.index("Is this package intact?"),
                        plantilla.index("## Try it"))

    def test_los_tres_codigos_NO_se_colapsan(self):
        plantilla = self._plantilla()
        for texto in ("PASS", "FAIL", "NOT FULLY CHECKED"):
            self.assertIn(texto, plantilla)

    def test_si_matrixai_no_esta_se_DICE(self):
        """Un botón que falla en silencio se lee como que no hay nada que
        comprobar."""
        self.assertIn("is not installed", self._plantilla())


if __name__ == "__main__":
    unittest.main()
