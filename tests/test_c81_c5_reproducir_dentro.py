"""CONTRATO 81-C5 — reproducir DENTRO del contenedor, no solo declararlo.

Era la última pieza del corte que no dependía de una decisión: el
aislamiento se detectaba y se declaraba, y `_ejecutar_en` levantaba un
`NotImplementedError`. Ya ejecuta.

Medido el 2026-08-20 con Docker de verdad: un paquete honesto sale
`rc=0` con `manifest PASS`, y el mismo paquete con **una línea de más en
la receta** sale `rc=2` con `manifest FAIL` — dentro del contenedor, sin
red y con 2 GB de tope.

Lo que estas pruebas fijan sin necesitar Docker (el argv es puro y se
puede leer), más el caso que sí lo necesita:

* **el argv lleva SIEMPRE los límites que `LIMITES` declara** — decir
  «aislado» y no aplicarlo sería la peor de las dos mentiras posibles;
* **la imagen no se descarga sola**: dentro no hay red a propósito, y
  bajar bytes que nadie ha mirado sería peor que parar;
* **`bubblewrap` no reproduce**: aísla, pero correría el MatrixAI del
  anfitrión, que no reproduce nada;
* **un `rc=2` de `argparse` NO es un paquete manipulado.** `verify`
  devuelve 2 cuando alguien tocó el paquete y `argparse` devuelve 2
  cuando el subcomando no existe: el mismo número. Por eso se le pregunta
  antes a la imagen si sabe verificar.
"""

import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from matrixai.pipelines.sandbox import (
    IMAGEN_POR_DEFECTO,
    LIMITES,
    SinAislamiento,
    argv_de_comprobacion_previa,
    argv_de_reproduccion,
    imagen_del_sandbox,
    replay_and_verify,
)


class ElComandoAPLICALoQueDECLARATest(unittest.TestCase):
    def setUp(self):
        self.argv = argv_de_reproduccion("docker", "/tmp", "img:1")

    def test_la_red_va_DESACTIVADA(self):
        """Un replay que puede llamar a casa no es un replay: es ejecutar
        código ajeno con salida a internet."""
        self.assertIn("--network=none", self.argv)
        self.assertEqual(LIMITES["network"], "disabled")

    def test_el_tope_de_memoria_va_CON_su_swap(self):
        """Sin `--memory-swap` el contenedor se derrama al swap y el tope
        deja de topar. Medido en esta casa, y costó el servidor dos veces."""
        self.assertIn(f"--memory={LIMITES['memory']}", self.argv)
        self.assertIn(f"--memory-swap={LIMITES['memory']}", self.argv)

    def test_va_con_el_USUARIO_de_quien_llama(self):
        """Sin `--user` el contenedor escribe como ROOT, y lo siguiente es
        un `EACCES` que no dice nada. También costó dos veces."""
        self.assertIn(f"{os.getuid()}:{os.getgid()}", self.argv)

    def test_no_se_pueden_subir_privilegios_dentro(self):
        """Reproducir es correr un paquete que viene de cualquier sitio: sin
        esto, un `setuid` dentro puede subir privilegios aunque se entre con
        un uid que no los tiene."""
        self.assertIn("no-new-privileges", self.argv)

    def test_el_paquete_se_monta_de_SOLO_LECTURA(self):
        """Si el paquete pudiera cambiar durante su propia verificación,
        el veredicto no diría nada de los bytes que llegaron."""
        self.assertTrue(any(a.endswith(":/pkg:ro") for a in self.argv), self.argv)
        self.assertIn("--read-only", self.argv)

    def test_lo_que_corre_dentro_es_el_verify_del_82(self):
        """Dos comprobaciones distintas serían dos sitios diciendo lo
        mismo, y acabarían discrepando."""
        # `--retrain` va al final desde la auditoría externa: reproducir
        # sin reentrenar deja `training` y `R3` en `NOT_RUN`, y eso no es
        # reproducir (§15.6).
        self.assertEqual(self.argv[-5:],
                         ["matrixai", "verify", "/pkg", "--json", "--retrain"])

    def test_los_limites_del_argv_son_los_MISMOS_que_se_declaran(self):
        """La comprobación que ata las dos mitades: el informe declara
        `LIMITES` y quien audite tiene que poder ver que se aplicaron."""
        self.assertIn(f"--cpus={LIMITES['cpu']}", self.argv)


class LaImagenNoSeDescargaSolaTest(unittest.TestCase):
    def test_sin_imagen_NO_se_ejecuta_y_se_dice_cual_falta(self):
        from matrixai.pipelines import sandbox

        with patch.object(sandbox, "describir_aislamiento", return_value={
                "available": True, "chosen": {"backend": "docker", "version": "29"},
                "checked": []}), \
             patch.object(sandbox, "_hay_imagen", return_value=False):
            with self.assertRaises(SinAislamiento) as caja:
                replay_and_verify("/tmp")
        self.assertIn("no se descarga", str(caja.exception))
        self.assertIn(imagen_del_sandbox(), str(caja.exception))

    def test_la_imagen_por_defecto_es_la_del_PRODUCTO(self):
        """Una imagen genérica exigiría instalar algo, y dentro no hay red."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MATRIXAI_SANDBOX_IMAGE", None)
            self.assertEqual(imagen_del_sandbox(), IMAGEN_POR_DEFECTO)

    def test_se_puede_declarar_otra(self):
        with patch.dict(os.environ, {"MATRIXAI_SANDBOX_IMAGE": "otra:1"}):
            self.assertEqual(imagen_del_sandbox(), "otra:1")


class LosMENSAJES_DicenQueHacerTest(unittest.TestCase):
    """3ª pasada de auditoría (2026-08-20), conduciendo el producto: las
    dos negativas del replay decían «constrúyela» y **no decían cómo**.

    Es la misma regla que este módulo ya aplica a los backends —«no hay
    sandbox» a secas no dice qué instalar— y se estaba incumpliendo dos
    veces en el mismo fichero. Un mensaje que manda a otro sitio a buscar
    la respuesta hace trabajar de más a quien ya está atascado.

    Y el comando que dan **existe y funciona**: `examples/sandbox/`
    lleva el Dockerfile, y con la imagen que construye la reproducción
    sale `PASS` (medido el 2026-08-20).
    """

    _DOCKERFILE = (Path(__file__).resolve().parent.parent
                   / "examples" / "sandbox" / "Dockerfile")

    def test_sin_imagen_el_mensaje_trae_el_COMANDO(self):
        from matrixai.pipelines import sandbox

        with patch.object(sandbox, "describir_aislamiento", return_value={
                "available": True, "chosen": {"backend": "docker", "version": "29"},
                "checked": []}), \
             patch.object(sandbox, "_hay_imagen", return_value=False):
            with self.assertRaises(SinAislamiento) as caja:
                replay_and_verify("/tmp")
        self.assertIn("docker build", str(caja.exception))
        self.assertIn("examples/sandbox/Dockerfile", str(caja.exception))

    def test_y_si_la_imagen_no_sabe_verificar_tambien(self):
        from matrixai.pipelines import sandbox

        def falso(argv, **kw):
            return subprocess.CompletedProcess(argv, 2, "", "usage: matrixai …")

        with patch.object(sandbox, "_hay_imagen", return_value=True), \
             patch.object(sandbox.subprocess, "run", side_effect=falso):
            r = sandbox._ejecutar_en({"backend": "docker", "probe": "docker"}, "/tmp")
        self.assertIn("docker build", r["reason"])

    def test_el_Dockerfile_que_nombra_el_mensaje_EXISTE(self):
        """Un mensaje que cita un fichero que no está es peor que no
        citarlo: manda a buscar algo que no se va a encontrar."""
        self.assertTrue(self._DOCKERFILE.is_file(), self._DOCKERFILE)
        contenido = self._DOCKERFILE.read_text(encoding="utf-8")
        # Y hace lo que el mensaje promete: meter ESTE árbol dentro.
        self.assertIn("COPY matrixai", contenido)
        self.assertIn("PYTHONPATH", contenido)


class UnBackendQueNoMontaImagenesNoReproduceTest(unittest.TestCase):
    def test_bubblewrap_se_NIEGA_en_vez_de_correr_el_matrixai_del_anfitrion(self):
        from matrixai.pipelines import sandbox

        with self.assertRaises(SinAislamiento) as caja:
            sandbox._ejecutar_en({"backend": "bubblewrap", "probe": "bwrap"}, "/tmp")
        self.assertIn("no reproduce nada", str(caja.exception))


class UnVERIFICADOR_VIEJO_NoAcusaAlPaqueteTest(unittest.TestCase):
    """El hallazgo que salió al conducirlo: `matrixai-studio:v2.0` lleva un
    matrixai anterior al 82-C2, y su `rc=2` de argparse se leía como
    «alguien tocó el paquete». El mismo número para dos cosas opuestas."""

    def _con_previa(self, rc):
        from matrixai.pipelines import sandbox

        def falso(argv, **kw):
            return subprocess.CompletedProcess(argv, rc, "", "usage: matrixai …")
        return patch.object(sandbox.subprocess, "run", side_effect=falso)

    def test_si_la_imagen_no_sabe_verificar_NO_se_reproduce(self):
        from matrixai.pipelines import sandbox

        with patch.object(sandbox, "_hay_imagen", return_value=True), self._con_previa(2):
            r = sandbox._ejecutar_en({"backend": "docker", "probe": "docker"}, "/tmp")
        self.assertIsNone(r["returncode"], "no se puede dar un veredicto del paquete")
        self.assertIn("no tiene el comando `verify`", r["reason"])

    def test_y_el_informe_lo_dice_ARRIBA_no_enterrado(self):
        """Un `ok: false` a secas se lee como «el paquete está mal»."""
        from matrixai.pipelines import sandbox

        with patch.object(sandbox, "describir_aislamiento", return_value={
                "available": True, "chosen": {"backend": "docker", "version": "29"},
                "checked": []}), \
             patch.object(sandbox, "_hay_imagen", return_value=True), self._con_previa(2):
            informe = replay_and_verify("/tmp")
        self.assertFalse(informe["ok"])
        self.assertFalse(informe["checked"], "«no se pudo comprobar» tiene que verse")

    def test_la_comprobacion_previa_pregunta_por_verify(self):
        argv = argv_de_comprobacion_previa("docker", "img:1")
        self.assertEqual(argv[-3:], ["matrixai", "verify", "--help"])
        # Y va con los mismos límites: preguntar tampoco es barra libre.
        self.assertIn("--network=none", argv)

    def test_una_imagen_que_SI_sabe_deja_reproducir(self):
        from matrixai.pipelines import sandbox

        llamadas = []

        def falso(argv, **kw):
            llamadas.append(argv)
            if "--help" in argv:
                return subprocess.CompletedProcess(argv, 0, "usage", "")
            return subprocess.CompletedProcess(argv, 0, '{"stages":{}}', "")

        with patch.object(sandbox, "_hay_imagen", return_value=True), \
             patch.object(sandbox.subprocess, "run", side_effect=falso):
            r = sandbox._ejecutar_en({"backend": "docker", "probe": "docker"}, "/tmp")
        self.assertTrue(r["ok"])
        self.assertEqual(r["returncode"], 0)
        self.assertEqual(len(llamadas), 2, "previa y reproducción")


class AgotarElTiempoNoEsFALLARTest(unittest.TestCase):
    def test_un_timeout_se_declara_como_tal_y_no_como_paquete_malo(self):
        """Colapsarlo en «falla» acusaría a un paquete honesto que tarda."""
        from matrixai.pipelines import sandbox

        def falso(argv, **kw):
            if "--help" in argv:
                return subprocess.CompletedProcess(argv, 0, "usage", "")
            raise subprocess.TimeoutExpired(argv, 900)

        with patch.object(sandbox, "_hay_imagen", return_value=True), \
             patch.object(sandbox.subprocess, "run", side_effect=falso):
            r = sandbox._ejecutar_en({"backend": "docker", "probe": "docker"}, "/tmp")
        self.assertTrue(r["timed_out"])
        self.assertIsNone(r["returncode"])
        self.assertIn("no se pudo comprobar", r["reason"])


class ElCOMANDO_replayTest(unittest.TestCase):
    """`matrixai replay` — el C5, conducible desde el producto.

    Existía la función y **no había forma de invocarla**: un sandbox al
    que solo se llega importando un módulo no protege a nadie.

    Los tres códigos NO se colapsan, y esto es lo que fija la diferencia:
    `2` es «alguien lo tocó» y `3` es «no se pudo comprobar». Darles el
    mismo número obligaría a leer el informe para distinguir una
    manipulación de una falta de acceso.

    Medido a mano con Docker (2026-08-20, `matrixai-sandbox:prueba`):

        honesto         → `replay PASS`,        rc=0
        receta tocada   → `replay FAIL`,        rc=2
        sin imagen      → `replay NOT CHECKED`, rc=3
    """

    def setUp(self):
        import tempfile

        self.paquete = Path(tempfile.mkdtemp())
        (self.paquete / "model.mxai").write_text("NETWORK N\n")

    def _correr(self, argv_extra, salida):
        from matrixai import cli

        class _Args:
            package = str(self.paquete)
            json = False
            receipt_out = None
            compare_reference = None
            no_retrain = False

        for k, v in argv_extra.items():
            setattr(_Args, k, v)
        with patch("matrixai.pipelines.sandbox.replay_and_verify", return_value=salida):
            return cli._cmd_replay(_Args())

    def _informe(self, *, checked=True, ok=True, reason=None, codigo=None):
        """El informe CON su recibo, que es lo que devuelve el replay.

        Antes se escribía a mano y sin `receipt`: al pasar el veredicto al
        recibo (§15.6) este fixture describía una forma que la función ya
        no produce. Ahora se construye con el MISMO constructor, así que
        no puede volver a divergir.
        """
        from matrixai.pipelines.sandbox import recibo_de_reproduccion

        if codigo is None:
            codigo = 0 if ok else 2
        informe = {
            "ok": ok, "checked": checked, "package": str(self.paquete),
            "retrained": True,
            "isolation": {"backend": "docker", "version": "29", "image": "img:1",
                          "limits": {"network": "disabled", "cpu": "2",
                                     "memory": "2g", "timeout_s": 900,
                                     "filesystem": "read-only"}},
            "environment": {"image": "img:1", "backend": "docker",
                            "matrixai_version": "1.5.0",
                            "environment_sha256": "a" * 64},
            "dependencies": {"count": 0, "packages": [],
                             "without_declared_license": []},
            "result": {"stdout": "", "reason": reason,
                       "returncode": codigo if checked else None},
        }
        informe["receipt"] = recibo_de_reproduccion(
            informe, str(self.paquete), "r-prueba")
        return informe

    def test_un_paquete_que_cuadra_sale_con_CERO(self):
        self.assertEqual(self._correr({}, self._informe()), 0)

    def test_uno_MANIPULADO_sale_con_DOS(self):
        self.assertEqual(self._correr({}, self._informe(ok=False)), 2)

    def test_y_lo_que_NO_SE_PUDO_COMPROBAR_sale_con_TRES(self):
        """Un fallo por falta de acceso no es una manipulación."""
        self.assertEqual(
            self._correr({}, self._informe(checked=False, ok=False,
                                           reason="la imagen no sabe verificar")),
            3)

    def test_y_rc3_NO_es_lo_mismo_que_rc2(self):
        """`3` es «no se pudo comprobar del todo» y `2` es «alguien lo
        tocó». Colapsarlos acusa a un paquete honesto — el defecto que
        salió comparando dos entornos."""
        self.assertEqual(self._correr({}, self._informe(ok=False, codigo=3)), 3)

    def test_sin_aislamiento_tampoco_es_un_FAIL(self):
        from matrixai import cli
        from matrixai.pipelines.sandbox import SinAislamiento

        class _Args:
            package = str(self.paquete)
            json = False
            receipt_out = None
            compare_reference = None
            no_retrain = False

        with patch("matrixai.pipelines.sandbox.replay_and_verify",
                   side_effect=SinAislamiento("no hay backend")):
            self.assertEqual(cli._cmd_replay(_Args()), 3)

    def test_el_comando_esta_REGISTRADO_no_solo_escrito(self):
        """La comprobación que faltaba en el C5: que se pueda invocar."""
        import subprocess
        import sys

        # En la LISTA de subcomandos: es lo que significa «registrado», y
        # es donde argparse imprime el `help=` del parser. El `--help` del
        # propio subcomando no lo repite — el primer aserto lo buscaba ahí
        # y el que estaba mal era EL ASERTO, no el producto.
        listado = subprocess.run(
            [sys.executable, "-m", "matrixai", "--help"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(listado.returncode, 0)
        self.assertIn("replay", listado.stdout)

        propio = subprocess.run(
            [sys.executable, "-m", "matrixai", "replay", "--help"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(propio.returncode, 0)
        self.assertIn("package", propio.stdout)


@unittest.skipUnless(
    shutil.which("docker") and os.environ.get("MATRIXAI_SANDBOX_IMAGE"),
    "necesita Docker y una imagen con el matrixai de este árbol "
    "(MATRIXAI_SANDBOX_IMAGE)")
class DentroDelContenedorDeVerdadTest(unittest.TestCase):
    """Se salta sin Docker a propósito, pero **está medido a mano**
    (2026-08-20, `matrixai-sandbox:prueba` construida sobre
    `matrixai-studio:v2.0` con el árbol de hoy):

        honesto   → ok=True,  rc=0, manifest PASS
        saboteado → ok=False, rc=2, manifest FAIL
    """

    def _paquete(self):
        import hashlib
        import tempfile
        from pathlib import Path

        from matrixai.export.reproduce import write_reproduce_manifest
        receta, csv = "genera 200 filas\n", "a,b\n1,2\n3,4\n"
        d = Path(tempfile.mkdtemp())
        (d / "model.mxai").write_text("NETWORK N\n  DENSE 4\n")
        (d / "training.mxtrain").write_text(
            Path("examples/celsius_to_kelvin.mxtrain").read_text(encoding="utf-8"))
        (d / "recipe.txt").write_text(receta)
        sha = hashlib.sha256(csv.encode()).hexdigest()
        train = (d / "training.mxtrain").read_text(encoding="utf-8")
        write_reproduce_manifest(
            d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
            dataset_sha256=sha, dataset_rows=2,
            generation={"seeds": {"dataset": 42}}, weights_source="trained",
            run_provenance={
                "schema_version": "1.2",
                "mxai_sha256": hashlib.sha256((d / "model.mxai").read_bytes()).hexdigest(),
                "mxtrain_sha256": hashlib.sha256(train.encode()).hexdigest(),
                "mxtrain_text": train,
                "recipe_sha256": hashlib.sha256(receta.encode()).hexdigest(),
                "recipe_text": receta, "seeds": {"dataset": 42},
                "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
                "dataset_rows": 2, "dataset_rows_used": 2,
                "epochs_effective": 10, "epochs_ran": 10, "warm_start": False,
                "recipe_verification": {"verified": True,
                                        "code": "regenera_el_dataset"}})
        return d

    def test_un_paquete_honesto_se_reproduce_dentro(self):
        informe = replay_and_verify(str(self._paquete()))
        self.assertTrue(informe["checked"])
        self.assertTrue(informe["ok"], informe["result"])
        self.assertEqual(informe["isolation"]["limits"]["network"], "disabled")

    def test_una_linea_de_mas_en_la_receta_se_CAZA_dentro(self):
        """Un banco de pruebas sin dientes no vale: se sabotea a propósito."""
        import json

        paquete = self._paquete()
        (paquete / "recipe.txt").write_text("genera 200 filas\nuna linea de mas\n")
        informe = replay_and_verify(str(paquete))
        self.assertTrue(informe["checked"])
        self.assertFalse(informe["ok"])
        self.assertEqual(informe["result"]["returncode"], 2)
        etapas = json.loads(informe["result"]["stdout"])["stages"]
        self.assertEqual(etapas["manifest"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
