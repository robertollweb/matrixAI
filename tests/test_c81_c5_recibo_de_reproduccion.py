"""CONTRATO 81-C5 §15.6 — lo que le faltaba de verdad al corte.

Yo lo había declarado completo y **no lo estaba**; lo destapó la
auditoría externa del 2026-08-20. De los criterios del §15.6 faltaban
cuatro, y ninguno era cosmético:

* **recibo de reproducción** — no se emitía ninguno, así que no quedaba
  nada que archivar de lo que se reprodujo;
* **comparación con el recibo de referencia** — sin recibo no hay con qué
  comparar, y el `--compare-reference` del §15.5 no existía;
* **dos entornos de referencia** — «reproducible» dicho desde una sola
  máquina es la misma media verdad que una tolerancia de un solo entorno;
* **inventario de dependencias y licencias**.

Y al conducirlo salieron **dos defectos míos**, que van fijados aquí:

1. `rc=3` se etiquetaba `mismatch`. `3` es «no se pudo comprobar del
   todo», no «alguien lo tocó» — acusaba a un paquete honesto. Lo vi
   porque dos entornos con las MISMAS etapas daban veredictos distintos.
2. La huella del entorno miraba **la etiqueta de la imagen y no lo que
   hay dentro**. Dos imágenes con distinto nombre y el mismo matrixai no
   son dos entornos, y dos con el mismo nombre y distinto código tampoco
   son el mismo: mi primera comparación «entre entornos» estaba midiendo
   una diferencia de versión.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matrixai.pipelines import sandbox
from matrixai.pipelines.sandbox import (
    _dependencias_declaradas,
    comparar_con_referencia,
    entorno_de_reproduccion,
    inventario_de_dependencias,
    recibo_de_reproduccion,
)


def _informe(codigo, *, checked=True, etapas=None, entorno=None):
    salida = json.dumps({"stages": {k: {"status": v}
                                    for k, v in (etapas or {}).items()}})
    return {
        "ok": codigo == 0, "checked": checked, "retrained": True,
        "environment": entorno or {"image": "img:a", "backend": "docker",
                                   "matrixai_version": "1.5.0",
                                   "environment_sha256": "a" * 64},
        "dependencies": {"count": 3, "packages": [], "without_declared_license": []},
        "result": {"returncode": codigo if checked else None,
                   "stdout": salida, "command": ["docker", "run"]},
    }


class ElRECIBO_SeEmiteSIEMPRETest(unittest.TestCase):
    """Un recibo que solo existe cuando todo va bien no sirve para
    archivar lo que pasó, que es para lo que existe."""

    def setUp(self):
        self.paquete = Path(tempfile.mkdtemp())
        (self.paquete / "model.mxai").write_text("NETWORK N\n")
        (self.paquete / "recipe.txt").write_text("genera 200 filas\n")

    def _recibo(self, codigo, **kw):
        return recibo_de_reproduccion(_informe(codigo, **kw), str(self.paquete), "r1")

    def test_lleva_la_huella_del_PAQUETE(self):
        """Sin ella el recibo no dice DE QUÉ habla."""
        recibo = self._recibo(0)
        self.assertTrue(recibo["package"]["content_digest"].startswith("sha256:"))
        self.assertEqual(recibo["package"]["files"], 2)
        self.assertTrue(recibo["package"]["covers_whole_package"])

    def test_dos_paquetes_distintos_dan_huellas_distintas(self):
        uno = self._recibo(0)["package"]["content_digest"]
        (self.paquete / "recipe.txt").write_text("otra receta\n")
        self.assertNotEqual(uno, self._recibo(0)["package"]["content_digest"])

    def test_un_fichero_ILEGIBLE_no_se_traga(self):
        """El digest dejaría de cubrir el paquete entero y el recibo
        estaría afirmando sobre menos de lo que dice."""
        with patch.object(Path, "read_bytes", side_effect=OSError(13, "Permission denied")):
            recibo = self._recibo(0)
        self.assertFalse(recibo["package"]["covers_whole_package"])
        self.assertEqual(len(recibo["package"]["unreadable"]), 2)

    def test_los_TRES_desenlaces_no_se_colapsan(self):
        self.assertEqual(self._recibo(0)["output"]["outcome"], "reproduced")
        self.assertEqual(self._recibo(2)["output"]["outcome"], "mismatch")
        self.assertEqual(self._recibo(3)["output"]["outcome"], "not_fully_checked")
        self.assertEqual(self._recibo(None, checked=False)["output"]["outcome"],
                         "not_checked")

    def test_rc3_NO_es_una_acusacion(self):
        """El defecto que salió conduciéndolo: `3` se etiquetaba
        `mismatch`, y eso acusa a un paquete que nadie ha tocado."""
        self.assertNotEqual(self._recibo(3)["output"]["outcome"], "mismatch")

    def test_lleva_las_cuatro_etapas_del_verify_de_dentro(self):
        recibo = self._recibo(0, etapas={"manifest": "PASS", "R1": "PASS",
                                         "training": "PASS", "R3": "PASS"})
        self.assertEqual(recibo["output"]["stages"]["R3"], "PASS")

    def test_y_dice_si_se_REENTRENÓ(self):
        self.assertTrue(self._recibo(0)["steps"][0]["retrained"])


class ElINVENTARIO_DeDependenciasTest(unittest.TestCase):
    """§15.6: «las dependencias y licencias estén inventariadas».
    «Reproducible» sin decir con qué es media promesa."""

    def test_lista_paquetes_con_version(self):
        inv = inventario_de_dependencias()
        self.assertGreater(inv["count"], 0)
        self.assertTrue(all(p["name"] and p["version"] for p in inv["packages"]))

    def test_una_licencia_que_no_consta_sale_NULL_y_se_cuenta(self):
        """Inventar una licencia es peor que no tenerla: alguien la usaría
        para decidir."""
        inv = inventario_de_dependencias()
        sin = inv["without_declared_license"]
        self.assertIsInstance(sin, list)
        for paquete in inv["packages"]:
            if paquete["license"] is None:
                self.assertIn(paquete["name"], sin)

    def test_van_ordenados_para_poder_comparar_dos_inventarios(self):
        nombres = [p["name"].lower() for p in inventario_de_dependencias()["packages"]]
        self.assertEqual(nombres, sorted(nombres))


class ElINVENTARIO_SeMideDENTROTest(unittest.TestCase):
    """H3 del refutador (2026-08-20): el recibo publicaba el inventario
    del ANFITRIÓN presentándolo como el entorno de reproducción.

    Medido por el refutador: 109 paquetes del host (`Twisted`, `boto3`,
    `bcc`…) frente a los 27 de la imagen, y dos imágenes DISTINTAS daban
    inventarios idénticos — porque ninguna de las dos se estaba mirando.
    Consecuencia directa sobre §15.7, que exige auditar «sustitución de
    una dependencia»: una sustituida dentro de la imagen no aparecía en
    ningún recibo.
    """

    def test_si_no_se_pudo_mirar_dentro_NO_se_cuela_el_del_anfitrion(self):
        """Un inventario que describe otra máquina es peor que no tener
        inventario: se usaría para decidir."""
        declarado = _dependencias_declaradas(None)
        self.assertFalse(declarado["measured"])
        self.assertIsNone(declarado["packages"])
        # Y un ausente NO es un cero: `count: 0` se lee como «no hay
        # dependencias», que es una afirmación, y falsa.
        self.assertIsNone(declarado["count"])
        self.assertIn("anfitrión", declarado["reason"])

    def test_y_cuando_si_se_pudo_dice_DONDE_lo_midio(self):
        dentro = {"packages": [{"name": "x", "version": "1", "license": None}],
                  "count": 1, "without_declared_license": ["x"],
                  "python_version": "3.12.14"}
        declarado = _dependencias_declaradas(dentro)
        self.assertTrue(declarado["measured"])
        self.assertEqual(declarado["measured_in"], "sandbox")
        self.assertEqual(declarado["count"], 1)

    def test_el_inventario_de_dentro_ENTRA_en_la_huella_del_entorno(self):
        """§15.7 pide auditar la sustitución de una dependencia: si el
        inventario no entrara en la huella, sustituir una no cambiaría
        nada de lo que el recibo dice."""
        uno = {"packages": [{"name": "x", "version": "1", "license": None}],
               "python_version": "3.12.14"}
        otro = {"packages": [{"name": "x", "version": "2", "license": None}],
                "python_version": "3.12.14"}
        with patch("matrixai.pipelines.sandbox._identidad_de_imagen",
                   return_value="sha256:" + "1" * 64), \
             patch("matrixai.pipelines.sandbox._version_dentro_de",
                   return_value="1.5.0"):
            backend = {"backend": "docker", "version": "29", "probe": "docker"}
            a = entorno_de_reproduccion("img:a", backend, uno)
            b = entorno_de_reproduccion("img:a", backend, otro)
        self.assertNotEqual(a["environment_sha256"], b["environment_sha256"])

    def test_la_ETIQUETA_no_entra_en_la_huella(self):
        """H4: `docker tag` fabricaba un segundo entorno, y §15.6 pide
        «al menos dos entornos de referencia aprobados»."""
        dentro = {"packages": [], "python_version": "3.12.14"}
        with patch("matrixai.pipelines.sandbox._identidad_de_imagen",
                   return_value="sha256:" + "1" * 64), \
             patch("matrixai.pipelines.sandbox._version_dentro_de",
                   return_value="1.5.0"):
            backend = {"backend": "docker", "version": "29", "probe": "docker"}
            a = entorno_de_reproduccion("img:a", backend, dentro)
            clon = entorno_de_reproduccion("img:a-clon", backend, dentro)
        self.assertEqual(a["environment_sha256"], clon["environment_sha256"])
        # Pero la etiqueta se CONSERVA: hace falta para saber qué se
        # invocó, y tirarla sería perder un dato por arreglar otro.
        self.assertEqual(clon["image"], "img:a-clon")

    def test_si_no_se_pudo_mirar_dentro_la_huella_lo_DECLARA(self):
        """Una huella construida sobre cuatro `null` es idéntica para dos
        entornos cualesquiera: presentarla sin avisar los daría por el
        mismo."""
        with patch("matrixai.pipelines.sandbox._identidad_de_imagen", return_value=None), \
             patch("matrixai.pipelines.sandbox._version_dentro_de", return_value=None):
            entorno = entorno_de_reproduccion(
                "img:a", {"backend": "docker", "version": "29", "probe": "docker"}, None)
        self.assertFalse(entorno["measured_inside"])


class ElENTORNO_MiraLoQueHayDENTROTest(unittest.TestCase):
    """El segundo defecto que salió conduciéndolo: la huella miraba la
    etiqueta de la imagen y no su contenido."""

    def test_la_huella_incluye_la_version_de_dentro(self):
        with patch.object(sandbox, "_version_dentro_de", return_value="1.5.0"):
            entorno = entorno_de_reproduccion("img:a", {"backend": "docker",
                                                        "version": "29", "probe": "docker"})
        self.assertEqual(entorno["matrixai_version"], "1.5.0")

    def test_la_MISMA_imagen_con_otro_matrixai_NO_es_el_mismo_entorno(self):
        def entorno(version):
            with patch.object(sandbox, "_version_dentro_de", return_value=version):
                return entorno_de_reproduccion("img:a", {"backend": "docker",
                                                         "version": "29", "probe": "docker"})
        self.assertNotEqual(entorno("1.5.0")["environment_sha256"],
                            entorno("1.4.0")["environment_sha256"])

    def test_no_saber_la_version_NO_se_rellena(self):
        """Un valor por defecto haría parecer iguales dos entornos que no
        lo son."""
        with patch.object(sandbox, "_version_dentro_de", return_value=None):
            entorno = entorno_de_reproduccion("img:a", {"backend": "docker",
                                                        "version": "29", "probe": "docker"})
        self.assertIsNone(entorno["matrixai_version"])


class CompararConLaREFERENCIATest(unittest.TestCase):
    def setUp(self):
        self.paquete = Path(tempfile.mkdtemp())
        (self.paquete / "model.mxai").write_text("NETWORK N\n")

    def _recibo(self, codigo, entorno, etapas=None):
        return recibo_de_reproduccion(
            _informe(codigo, etapas=etapas, entorno=entorno), str(self.paquete), "r")

    #: DOS ENTORNOS DE VERDAD DISTINTOS, no dos etiquetas.
    #:
    #: Esto eran `img:a` e `img:b` con TODO lo demás igual, o sea el
    #: escenario que H4 del refutador convirtió en hallazgo: `docker tag`
    #: fabricaba un segundo entorno. Ahora lo que diferencia es lo que hay
    #: dentro —el id de la imagen, su Python, su inventario—, y la
    #: etiqueta viaja sin entrar en la huella.
    _A = {"image": "img:a", "image_id": "sha256:" + "1" * 64,
          "backend": "docker", "backend_version": "29",
          "matrixai_version": "1.5.0", "python_version": "3.12.14",
          "dependencies_sha256": "d" * 64, "measured_inside": True,
          "environment_sha256": "a" * 64}
    _B = {"image": "img:b", "image_id": "sha256:" + "2" * 64,
          "backend": "docker", "backend_version": "29",
          "matrixai_version": "1.5.0", "python_version": "3.10.12",
          "dependencies_sha256": "e" * 64, "measured_inside": True,
          "environment_sha256": "b" * 64}
    #: La MISMA imagen con otra etiqueta: mismo entorno, y la comparación
    #: tiene que decirlo (H4).
    _A_RENOMBRADA = {**_A, "image": "img:a-clon"}

    def test_dos_paquetes_distintos_NO_son_comparables(self):
        uno = self._recibo(0, self._A)
        (self.paquete / "model.mxai").write_text("NETWORK OTRA\n")
        otro = self._recibo(0, self._B)
        r = comparar_con_referencia(otro, uno)
        self.assertFalse(r["comparable"])
        self.assertIn("paquetes distintos", r["reason"])

    def test_en_el_MISMO_entorno_lo_dice_y_avisa(self):
        """Coincidir en el mismo entorno prueba repetibilidad, no
        reproducibilidad — y ésa es la media verdad que §15.6 evita."""
        r = comparar_con_referencia(self._recibo(0, self._A), self._recibo(0, self._A))
        self.assertTrue(r["same_environment"])
        self.assertIn("repetibilidad, no reproducibilidad", r["note"])

    def test_en_entornos_DISTINTOS_tambien_lo_dice(self):
        r = comparar_con_referencia(self._recibo(0, self._B), self._recibo(0, self._A))
        self.assertFalse(r["same_environment"])
        self.assertIn("sí sostiene reproducibilidad", r["note"])
        # Y EN QUÉ se diferencian, nombrado. `image` NO está: cambiar la
        # etiqueta no cambia el entorno.
        self.assertEqual(r["environment_differences"],
                         ["dependencies_sha256", "image_id", "python_version"])

    def test_la_MISMA_imagen_con_otra_ETIQUETA_es_el_MISMO_entorno(self):
        """H4 del refutador: `docker tag` daba `same_environment: False`,
        así que §15.6 —«al menos dos entornos de referencia aprobados»— se
        satisfacía con un renombrado."""
        r = comparar_con_referencia(self._recibo(0, self._A_RENOMBRADA),
                                    self._recibo(0, self._A))
        self.assertTrue(r["same_environment"])
        self.assertEqual(r["environment_differences"], [])
        self.assertIn("repetibilidad, no reproducibilidad", r["note"])

    def test_dos_NO_COMPROBADAS_que_coinciden_no_sostienen_nada(self):
        """La nota dependía solo de `same_environment`, así que con los
        dos desenlaces en `not_fully_checked` seguía diciendo «coincidir
        aquí sí sostiene reproducibilidad» (H4, segunda mitad)."""
        r = comparar_con_referencia(self._recibo(3, self._B), self._recibo(3, self._A))
        self.assertFalse(r["same_environment"])
        self.assertIn("no llegó a comprobarse", r["note"])
        self.assertNotIn("sí sostiene reproducibilidad", r["note"])

    def test_si_no_se_pudo_mirar_DENTRO_la_comparacion_no_sostiene_nada(self):
        """Una huella construida sobre `null` es idéntica para dos
        entornos cualesquiera: presentarla sin avisar diría que son el
        mismo."""
        ciego = {**self._A, "measured_inside": False}
        r = comparar_con_referencia(self._recibo(0, ciego), self._recibo(0, ciego))
        self.assertIn("no se pudo mirar DENTRO", r["note"])

    def test_las_etapas_que_difieren_se_NOMBRAN(self):
        """«Algo cambió» sin decir cuál obliga a abrir los dos recibos."""
        uno = self._recibo(0, self._A, {"manifest": "PASS", "R1": "PASS"})
        otro = self._recibo(2, self._B, {"manifest": "PASS", "R1": "FAIL"})
        r = comparar_con_referencia(otro, uno)
        self.assertEqual(list(r["stages_differing"]), ["R1"])
        self.assertEqual(r["stages_differing"]["R1"],
                         {"reference": "PASS", "new": "FAIL"})
        self.assertFalse(r["same_outcome"])


if __name__ == "__main__":
    unittest.main()


class ElENLACEQueNoSeCubriaTest(unittest.TestCase):
    """H7 del refutador (2026-08-20): `recibo_de_reproduccion` saltaba los
    enlaces simbólicos con un `continue` seco **antes** del bloque que
    cuenta lo ilegible, así que no entraban en `unreadable` y
    `covers_whole_package` seguía diciendo `true` sobre un paquete que el
    digest no cubría — con el comentario de tres líneas más abajo
    afirmando exactamente lo contrario.
    """

    def setUp(self):
        self.paquete = Path(tempfile.mkdtemp())
        (self.paquete / "model.mxai").write_text("NETWORK N\n")

    def _recibo(self):
        return recibo_de_reproduccion(_informe(0), str(self.paquete), "r")

    def test_un_paquete_honesto_SI_esta_cubierto(self):
        """El otro lado primero: si esto no fuera cierto, la prueba de
        abajo la pasaría un verificador que acusa a todo el mundo."""
        pk = self._recibo()["package"]
        self.assertEqual(pk["files"], 1)
        self.assertEqual(pk["unreadable"], [])
        self.assertTrue(pk["covers_whole_package"])

    def test_un_enlace_COLADO_se_cuenta_y_se_NOMBRA(self):
        (self.paquete / "extra.txt").symlink_to("/etc/passwd")
        pk = self._recibo()["package"]

        # No se sigue —apunta fuera del paquete y `/etc/passwd` no es
        # contenido suyo—, pero SÍ se cuenta: es una entrada que el digest
        # no cubre, que es lo que `unreadable` significa.
        self.assertEqual(pk["files"], 1)
        self.assertFalse(pk["covers_whole_package"])
        self.assertEqual(len(pk["unreadable"]), 1)
        self.assertIn("extra.txt", pk["unreadable"][0])
        self.assertIn("enlace simbólico", pk["unreadable"][0])

    def test_el_recibo_de_replay_lleva_su_FECHA(self):
        """§14.2 la pide en todo recibo, y éste no la llevaba."""
        self.assertTrue(self._recibo()["created_at"])
