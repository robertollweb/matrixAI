"""AUDITORÍA del 82, 1ª pasada — un paquete NO elige qué fichero se abre.

Encontrado el 2026-08-20 sondeando los bordes de `matrixai verify`: un
paquete podía declarar un artefacto en `../../algo` —o en una ruta
absoluta, o detrás de un enlace simbólico— y **`verify` leía el fichero
de fuera y lo daba por bueno con `manifest PASS`**.

Dicho de otra forma: un paquete ajeno elegía qué fichero de tu máquina se
abría, y el informe llamaba «modelo» a lo que hubiera dentro. En un
contrato cuyo objeto entero es la integridad verificable, ése es el
agujero que más caro sale.

**Y no bastaba con que el manifiesto estuviera firmado consigo mismo**:
ésa fue la primera lectura y era falsa. `manifest_sha256` protege contra
que alguien MANIPULE un paquete honesto, pero quien lo FABRICA calcula ese
digest sin esfuerzo. La comprobación que faltaba no es de firma: es que el
artefacto **esté dentro**.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from matrixai.export.reproduce import (
    añadir_inventario_de_ficheros,
    manifest_digest,
    write_reproduce_manifest,
)
from matrixai.export.verify import _ruta_fuera_del_paquete, verify_package

EJEMPLOS = Path(__file__).resolve().parent.parent / "examples"


class UnPaqueteNoEligeQueFicheroSeABRETest(unittest.TestCase):
    def setUp(self):
        self.fuera = Path(tempfile.mkdtemp())
        (self.fuera / "secreto.txt").write_text("FUERA DEL PAQUETE\n", encoding="utf-8")
        self.sha_fuera = hashlib.sha256(
            (self.fuera / "secreto.txt").read_bytes()).hexdigest()

        self.bundle = Path(tempfile.mkdtemp())
        (self.bundle / "model.mxai").write_text("NETWORK N\n  DENSE 4\n")
        (self.bundle / "training.mxtrain").write_text(
            (EJEMPLOS / "celsius_to_kelvin.mxtrain").read_text(encoding="utf-8"))
        (self.bundle / "recipe.txt").write_text("genera 200 filas\n")
        write_reproduce_manifest(
            self.bundle, training_filename="training.mxtrain",
            recipe_filename="recipe.txt", dataset_sha256="a" * 64, dataset_rows=2,
            generation={"seeds": {"dataset": 42}}, weights_source="trained")
        # El inventario, como hace el producto justo antes de promover el
        # paquete: sin él este fixture describe uno que ya no se
        # construye, y desde H5 (2026-08-20) sale INCOMPARABLE.
        añadir_inventario_de_ficheros(self.bundle)

    def _con_ruta(self, ruta: str) -> dict:
        """Un paquete MALICIOSO DESDE EL ORIGEN: la ruta apunta fuera **y**
        el `manifest_sha256` está bien calculado, como lo calcularía quien
        fabrica el paquete."""
        man = json.loads((self.bundle / "reproduce.json").read_text())
        man["artifacts"]["model"] = {"path": ruta, "sha256": self.sha_fuera}
        man["manifest_sha256"] = manifest_digest(man)
        (self.bundle / "reproduce.json").write_text(json.dumps(man))
        # `locale="en"` FIJADO (85-C2b): lo que estos tres asertos miden es
        # que el `problem` NOMBRE la fuga —«escapes», «absolute», «outside»—,
        # y eso solo se puede escribir en un idioma concreto. Que la misma
        # frase exista en español lo mide el barrido del 85-C2b; aquí se
        # sigue midiendo la fuga, que es lo que este fichero existe para
        # medir.
        return verify_package(self.bundle, locale="en")["stages"]["manifest"]

    def test_el_paquete_HONESTO_sigue_pasando(self):
        """Lo primero: que el arreglo no acuse a quien no ha hecho nada."""
        self.assertEqual(
            verify_package(self.bundle)["stages"]["manifest"]["status"], "PASS")

    def test_una_ruta_con_dos_puntos_NO_pasa(self):
        etapa = self._con_ruta("../fuera/secreto.txt")
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("escapes the package", etapa["artifacts"][0]["problem"])

    def test_una_ruta_ABSOLUTA_no_pasa(self):
        etapa = self._con_ruta(str(self.fuera / "secreto.txt"))
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("absolute", etapa["artifacts"][0]["problem"])

    def test_un_ENLACE_SIMBOLICO_que_apunta_fuera_tampoco(self):
        """La ruta escrita parece de dentro y el fichero real no lo es:
        la misma fuga con otra ropa."""
        os.symlink(self.fuera / "secreto.txt", self.bundle / "enlace.txt")
        etapa = self._con_ruta("enlace.txt")
        self.assertEqual(etapa["status"], "FAIL")
        self.assertIn("outside the package", etapa["artifacts"][0]["problem"])

    def test_y_se_NOMBRA_el_artefacto_que_lo_intenta(self):
        """«Algo no cuadra» obliga a abrir el ZIP y comparar a mano."""
        etapa = self._con_ruta("../fuera/secreto.txt")
        self.assertEqual(etapa["artifacts"][0]["artifact"], "model")
        self.assertEqual(etapa["artifacts"][0]["path"], "../fuera/secreto.txt")


class LaComprobacionDeRutaEnSIMismaTest(unittest.TestCase):
    """La función suelta, por sus dos lados — que una ruta buena pase
    importa tanto como que una mala no."""

    def setUp(self):
        self.raiz = Path(tempfile.mkdtemp())
        (self.raiz / "sub").mkdir()
        (self.raiz / "sub" / "a.txt").write_text("x")

    def test_una_ruta_de_dentro_pasa(self):
        self.assertIsNone(_ruta_fuera_del_paquete(self.raiz, "sub/a.txt"))

    def test_y_una_de_dentro_en_la_raiz_tambien(self):
        (self.raiz / "b.txt").write_text("y")
        self.assertIsNone(_ruta_fuera_del_paquete(self.raiz, "b.txt"))

    def test_dos_puntos_en_MEDIO_de_la_ruta_tampoco(self):
        """`sub/../../fuera` sale igual, y la comprobación mira las
        partes, no solo el principio."""
        self.assertIsNotNone(_ruta_fuera_del_paquete(self.raiz, "sub/../../fuera/x"))

    def test_una_ruta_vacia_no_es_una_ruta(self):
        self.assertIsNotNone(_ruta_fuera_del_paquete(self.raiz, ""))

    def test_una_ruta_con_espacios_alrededor_se_rechaza(self):
        """Un `" model.mxai"` no es el mismo fichero que `model.mxai`, y
        aceptarlo dejaría dos nombres para la misma cosa."""
        self.assertIsNotNone(_ruta_fuera_del_paquete(self.raiz, " sub/a.txt"))


if __name__ == "__main__":
    unittest.main()
