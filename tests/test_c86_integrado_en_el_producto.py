"""86-C2 Y 86-C4, OFRECIDOS POR EL PRODUCTO (auditoría externa, hallazgo 3).

El veredicto era exacto: `sobre_in_toto()` y `firmar_con_sigstore()` tenían sus
primitivas probadas y **ningún llamante de producción**. Con eso:

* no se «emitían los dos» sobres — solo el nativo;
* Sigstore no estaba disponible como opción del producto;
* y por tanto C2 y C4 no podían declararse hechos aunque sus unidades pasaran.

Aquí se fija que el CLI los OFRECE, y las dos reglas que hacen que ofrecerlos no
sea peor que no hacerlo:

1. **`--in-toto` sin `--key` se rechaza**: un Statement sin firmar no dice de
   quién es, y emitirlo sería un sobre estándar vacío de garantía.
2. **`--sigstore` no cae a HMAC en silencio**: si falta la biblioteca o la
   identidad, se dice **cuál** y no se escribe ningún recibo. Caer a otra firma
   daría por hecho que a quien lo pide le da igual quién responde por él.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

_HAS = util.find_spec("onnxruntime") is not None and util.find_spec("onnx") is not None
if _HAS:
    from test_c87_c2_attest import _sigmoide


def _datos_del_sigmoide(d: Path) -> Path:
    """El CSV que le corresponde a ESTE modelo: dos entradas y la clase.

    La primera versión reutilizaba el `_datos` del otro fichero —tres columnas—
    y el ejecutor lo rechazaba con razón («no se recorta ni se rellena»). Un
    fixture que no encaja con su modelo mide otra cosa.
    """
    ruta = d / "eval.csv"
    ruta.write_text("a,b,y\n0.1,0.1,0\n0.2,0.0,0\n0.9,0.9,1\n1.0,0.8,1\n",
                    encoding="utf-8")
    return ruta

_RAIZ = Path(__file__).resolve().parent.parent


def _attest(d: Path, *extra: str):
    modelo = _sigmoide(d)
    datos = _datos_del_sigmoide(d)
    return subprocess.run(
        [sys.executable, "-m", "matrixai", "attest", str(modelo), "--data", str(datos),
         "-o", str(d / "r.json"), *extra],
        capture_output=True, text=True, cwd=str(_RAIZ))


def _attest_sin_fichero(d: Path, *extra: str):
    """Lo mismo pero SIN `-o`: lo que ve quien encadena con una tubería."""
    modelo = _sigmoide(d)
    datos = _datos_del_sigmoide(d)
    return subprocess.run(
        [sys.executable, "-m", "matrixai", "attest", str(modelo), "--data", str(datos),
         *extra],
        capture_output=True, text=True, cwd=str(_RAIZ))


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class ElProductoEmiteLosDosSobresTest(unittest.TestCase):
    def test_con_key_e_in_toto_se_escriben_LOS_DOS(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            r = _attest(d, "--key", "aabb", "--in-toto")
            self.assertEqual(r.returncode, 0, r.stderr)
            nativo = json.loads((d / "r.json").read_text(encoding="utf-8"))
            statement = json.loads((d / "r.in_toto.json").read_text(encoding="utf-8"))
        self.assertEqual(nativo["payloadType"],
                         "application/vnd.matrixai.receipt+json;version=1.0")
        self.assertEqual(statement["payloadType"], "application/vnd.in-toto+json")

    def test_el_statement_lleva_el_SUJETO_del_modelo(self):
        import base64
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            _attest(d, "--key", "aabb", "--in-toto")
            statement = json.loads((d / "r.in_toto.json").read_text(encoding="utf-8"))
            payload = json.loads(base64.b64decode(statement["payload"]))
        self.assertEqual(payload["_type"], "https://in-toto.io/Statement/v1")
        self.assertTrue(payload["subject"])
        self.assertIn("sigmoide.onnx", payload["subject"][0]["name"])

    def test_sin_key_el_in_toto_se_RECHAZA(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            r = _attest(d, "--in-toto")
        self.assertEqual(r.returncode, 2)
        self.assertIn("necesita --key", r.stderr)


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class SigstoreNoCaeAHmacEnSilencioTest(unittest.TestCase):
    def test_dice_QUE_falta_y_no_escribe_recibo(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            r = _attest(d, "--sigstore")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Sigstore", r.stderr)
        # Y no se ha escrito nada: una firma que no se hizo no deja recibo.
        self.assertFalse((Path(tmp) / "r.json").exists())

    def test_el_CLI_lo_OFRECE(self):
        """Un corte «hecho» cuyo camino no existe en el producto no está hecho."""
        ayuda = subprocess.run(
            [sys.executable, "-m", "matrixai", "attest", "--help"],
            capture_output=True, text=True, cwd=str(_RAIZ)).stdout
        self.assertIn("--in-toto", ayuda)
        self.assertIn("--sigstore", ayuda)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class LosSobresEXTRA_TAMBIEN_SIN_FicheroTest(unittest.TestCase):
    """2ª auditoría externa (2026-08-25), hallazgo 4 residual.

    Los sobres extra se escribían **solo dentro del `if args.output`**, así que
    `--in-toto` sin `-o` se aceptaba, no avisaba de nada y el Statement se
    perdía por el camino. Pedir un sobre, no recibirlo y no enterarse es peor
    que no ofrecerlo.
    """

    def test_in_toto_sin_o_SALE_por_la_salida(self):
        with TemporaryDirectory() as tmp:
            r = _attest_sin_fichero(Path(tmp), "--key", "aa" * 16, "--in-toto")
        self.assertEqual(r.returncode, 0, r.stderr)
        emitido = json.loads(r.stdout)
        self.assertEqual(sorted(emitido), ["in_toto", "receipt"])
        self.assertEqual(emitido["in_toto"]["payloadType"], "application/vnd.in-toto+json")
        self.assertTrue(emitido["in_toto"]["signatures"])

    def test_y_dice_como_separarlos_en_ficheros(self):
        with TemporaryDirectory() as tmp:
            r = _attest_sin_fichero(Path(tmp), "--key", "aa" * 16, "--in-toto")
        self.assertIn("use -o", r.stderr)

    def test_sin_sobres_extra_la_salida_NO_cambia_de_forma(self):
        # Quien ya encadenaba `attest | jq .metrics` sigue viendo el recibo
        # pelado: la envoltura solo aparece cuando hay algo más que envolver.
        with TemporaryDirectory() as tmp:
            r = _attest_sin_fichero(Path(tmp), "--key", "aa" * 16)
        emitido = json.loads(r.stdout)
        self.assertEqual(sorted(emitido), ["payload", "payloadType", "signatures"])


class UnReciboFirmadoConSigstoreNoSeDeclaraSIN_FIRMARTest(unittest.TestCase):
    """2ª auditoría externa, hallazgo 4 residual (la otra mitad).

    El nivel se calculaba sobre el recibo PELADO, no sobre lo que se emite: con
    `--sigstore` salía **A0** y el CLI recomendaba «pass --key to sign it»
    justo encima de una firma Sigstore recién hecha. Sigstore sigue sin subir el
    nivel —eso es del 86-C4 y tiene su propia prueba—; lo que se arregla es
    dejar de decir que no está firmado cuando lo está.
    """

    def _emitido(self, evidencia: dict) -> dict:
        return {"payload": {"schema_version": "1.0", "evidence": evidencia},
                "sigstore": {"bundle": "{...}", "mode": "sigstore-keyless"}}

    def test_con_bundle_de_sigstore_NO_es_A0(self):
        from matrixai.pipelines.receipt import nivel_del_recibo
        self.assertEqual(nivel_del_recibo(self._emitido({"attests": "algo"})), "A1")

    def test_y_sigue_SIN_subir_de_nivel_por_firmar_mejor(self):
        # Con evidencia reproducible es A2, igual que con HMAC: el nivel lo da
        # lo que se comprobó, no la fuerza de la firma.
        from matrixai.pipelines.receipt import nivel_del_recibo
        con_evidencia = {"reproducible": True, "attests": "algo",
                         "recipe_sha256": "a" * 64, "dataset_sha256": "b" * 64}
        self.assertIn(nivel_del_recibo(self._emitido(con_evidencia)), ("A1", "A2"))

    def test_un_bundle_VACIO_no_cuenta_como_firma(self):
        from matrixai.pipelines.receipt import nivel_del_recibo
        self.assertEqual(nivel_del_recibo(
            {"payload": {"evidence": {}}, "sigstore": {"mode": "sigstore-keyless"}}), "A0")
