"""`matrixai report --probast`, LA OTRA MITAD DEL CRITERIO DE 109-C3 (2026-09-15).

109-C3 construyó los huecos PROBAST+AI (`matrixai/export/probast.py`) y su
criterio de terminado pide que se alcancen **por el CLI**, igual que
`--tripod`. El corte se entregó con esa mitad **declarada como no hecha**: el
agente que lo escribió tenía `cli.py` fuera de su territorio porque una
medición de 14 horas estaba corriendo, así que verificó el cambio sobre una
COPIA y lo dejó dicho en vez de colarlo.

**Por qué esto tiene prueba propia y no se dio por hecho.** El mismo día, un
corte de esta casa (111-C1) se commiteó afirmando que impedía que su tick
verde mintiera, y no lo impedía: sus pruebas medían la función que decide, no
la que devuelve el código de salida. Un comando de línea de órdenes es
exactamente eso — lo que vale es **con qué código sale**, porque es lo que
mira un guion ajeno, y lo que nadie prueba es lo que se rompe en silencio.

Los cuatro códigos van FIJADOS aquí:

    sin bandera            -> 2   (no se adivina qué informe se quería)
    --tripod --probast     -> 2   (dos informes en una salida no se distinguen
                                   después: se pide uno, o se llama dos veces)
    paquete que no existe  -> 1
    --probast correcto     -> 0

**Y `--tripod` se comprueba en el mismo fichero, a propósito**: el cambio le
tocó su rama, y una regresión ahí sería justo la clase de daño que no se ve
—quien ya tenía `--tripod` en un guion no se entera hasta que falla en su
máquina—. El texto de ayuda de `--tripod` decía «el día que haya un segundo
formato»; hoy lo hay, así que esa frase también dejó de ser cierta y se
corrigió con el cambio.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_PAQUETE_REAL = Path("/tmp/kelvin_bundle")


def _correr(*argumentos: str) -> subprocess.CompletedProcess:
    """El CLI de VERDAD, en su propio proceso.

    No se llama a `_cmd_report` directamente: lo que este corte promete es un
    COMANDO, y un comando incluye su análisis de argumentos y su código de
    salida. Probar la función se habría saltado las dos cosas que aquí importan.
    """
    return subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "report", *argumentos],
        capture_output=True, text=True, timeout=180,
        cwd=str(Path(__file__).resolve().parent.parent))


@unittest.skipUnless(_PAQUETE_REAL.is_dir(),
                     f"hace falta un paquete real en {_PAQUETE_REAL}")
class ElComandoSaleConElCodigoQueDiceTest(unittest.TestCase):
    def test_sin_bandera_NO_adivina_que_informe_se_queria(self):
        r = _correr(str(_PAQUETE_REAL))
        self.assertEqual(r.returncode, 2, r.stderr)
        # Y nombra las dos opciones: un error que no dice qué poner obliga a
        # leer el código fuente, que es justo a quien no lo tiene.
        self.assertIn("--tripod", r.stderr)
        self.assertIn("--probast", r.stderr)

    def test_las_DOS_banderas_a_la_vez_se_rechazan(self):
        r = _correr(str(_PAQUETE_REAL), "--tripod", "--probast")
        self.assertEqual(r.returncode, 2, r.stderr)

    def test_un_paquete_que_no_existe_sale_en_ROJO(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _correr(str(Path(tmp) / "no_esta"), "--probast")
        self.assertEqual(r.returncode, 1, r.stderr)

    def test_probast_sobre_un_paquete_real_sale_en_VERDE_y_escribe_la_ficha(self):
        r = _correr(str(_PAQUETE_REAL), "--probast")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PROBAST", r.stdout)

    def test_tripod_SIGUE_funcionando_por_el_mismo_camino(self):
        """La mitad que ya existía. El cambio tocó su rama."""
        r = _correr(str(_PAQUETE_REAL), "--tripod")
        self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipUnless(_PAQUETE_REAL.is_dir(),
                     f"hace falta un paquete real en {_PAQUETE_REAL}")
class LaFichaNoEmiteUnVeredictoDeRiesgoTest(unittest.TestCase):
    """El invariante 6 del 109, comprobado **por el comando** y no por la función.

    El módulo tiene su propia puerta (`sin_veredicto_de_riesgo`) y está
    probada aparte. Esto comprueba la otra mitad: que lo que de verdad SALE
    por la salida estándar tampoco lo trae. Entre la función y el terminal hay
    un formateador, una traducción y un `print`, y cualquiera de los tres podría
    añadir una frase.
    """

    def test_lo_que_SALE_por_pantalla_no_juzga_el_riesgo(self):
        r = _correr(str(_PAQUETE_REAL), "--probast")
        self.assertEqual(r.returncode, 0, r.stderr)
        plano = " ".join(r.stdout.lower().replace("*", "").split())
        for veredicto in ("low risk of bias", "high risk of bias",
                          "unclear risk of bias", "bajo riesgo de sesgo",
                          "alto riesgo de sesgo"):
            self.assertNotIn(veredicto, plano,
                             f"la salida del comando emite «{veredicto}», y ese "
                             "juicio lo hace quien revisa, no el núcleo")

    def test_y_SÍ_dice_que_no_puntua(self):
        """La otra mitad del aserto negativo de arriba: un `assertNotIn` lo
        pasa una salida vacía, así que hay que comprobar que además dice algo."""
        r = _correr(str(_PAQUETE_REAL), "--probast")
        plano = r.stdout.lower()
        self.assertTrue("does not score" in plano or "no puntúa" in plano,
                        "la ficha tiene que DECIR que no puntúa, no solo callarlo")


if __name__ == "__main__":
    unittest.main()
