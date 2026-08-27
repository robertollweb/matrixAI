"""EL PAQUETE QUE SE DESCARGA REPRODUCE LO QUE ANUNCIA — 2ª auditoría externa.

El hallazgo 2 residual del 2026-08-25 decía que la galería no reproducía lo que
publica. Medido el 2026-08-26 sobre el ZIP de kelvin recién descomprimido —que es
lo que hace quien se lo baja— salían DOS defectos del verificador, no del
paquete:

1. **El dataset regenerado se escribía sin su carpeta.** El `.mxtrain` cita
   `datos/…-train.csv` y esto escribía `…-train.csv` en la raíz del taller, así
   que el entrenador respondía «DATASET source not found» y `training` salía
   `INCOMPARABLE`. No se había visto porque se probaba desde el directorio donde
   se construyó el paquete: allí `datos/` existía **en el disco** y el
   entrenamiento tiraba de ése.

2. **Se escribía el CSV ENTERO donde va la parte de train.** `generate-dataset`
   parte las filas 80/20 y el contrato cita el de train; verify reentrenaba con
   300 filas un run hecho con 240. Publicado `mae 6.505213034913027e-17`,
   obtenido `6.499430623326438e-17`: R3 `FAIL` sobre un paquete honesto, con una
   tolerancia que está MEDIDA en 0,0 y por tanto no absorbe nada.

Las dos cosas juntas: la página anunciaba `training PASS / R3 PASS` y quien
descargara el ZIP veía otra cosa.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from matrixai.export.verify import _reparto_del_dataset, _ruta_del_dataset
from matrixai.training.particion_sintetica import corte_train_eval, nombres_del_dataset


class ElDatasetSeEscribeDONDE_EL_CONTRATO_LO_CITATest(unittest.TestCase):

    def test_la_carpeta_del_contrato_se_RESPETA(self):
        self.assertEqual(_ruta_del_dataset("datos/kelvin-synthetic-train.csv"),
                         Path("datos/kelvin-synthetic-train.csv"))

    def test_sin_carpeta_sigue_siendo_el_nombre(self):
        self.assertEqual(_ruta_del_dataset("d.csv"), Path("d.csv"))

    def test_una_ruta_ABSOLUTA_no_se_obedece(self):
        # La escribe el paquete: obedecerla elegiría un fichero de la máquina de
        # quien verifica, y verificar no escribe fuera de su taller.
        self.assertEqual(_ruta_del_dataset("/etc/passwd"), Path("passwd"))

    def test_ni_una_que_SUBE_de_directorio(self):
        self.assertEqual(_ruta_del_dataset("../../fuera.csv"), Path("fuera.csv"))

    def test_sin_fuente_hay_un_nombre_por_defecto(self):
        self.assertEqual(_ruta_del_dataset(None), Path("dataset.csv"))


class SeReentrenaConLA_MISMA_PARTETest(unittest.TestCase):

    _CSV = "a,y\n" + "".join(f"{i},{i * 2}\n" for i in range(10))

    def test_un_train_de_generate_dataset_se_PARTE_80_20(self):
        reparto = _reparto_del_dataset("datos/p-synthetic-train.csv", self._CSV)
        self.assertEqual(sorted(reparto), ["datos/p-synthetic-eval.csv",
                                           "datos/p-synthetic-train.csv"])
        train = reparto["datos/p-synthetic-train.csv"].splitlines()
        evalu = reparto["datos/p-synthetic-eval.csv"].splitlines()
        self.assertEqual(len(train) - 1, 8)
        self.assertEqual(len(evalu) - 1, 2)
        # Y las dos partes llevan la CABECERA: sin ella el entrenador leería la
        # primera fila de datos como nombres de columna.
        self.assertEqual(train[0], "a,y")
        self.assertEqual(evalu[0], "a,y")

    def test_y_las_filas_no_se_pierden_ni_se_repiten(self):
        reparto = _reparto_del_dataset("datos/p-synthetic-train.csv", self._CSV)
        juntas = ([l for l in reparto["datos/p-synthetic-train.csv"].splitlines()[1:]]
                  + [l for l in reparto["datos/p-synthetic-eval.csv"].splitlines()[1:]])
        self.assertEqual(juntas, self._CSV.splitlines()[1:])

    def test_un_CSV_propio_se_escribe_ENTERO(self):
        # Sin la convención de `generate-dataset` no hay nada que suponer, y
        # partirlo escribiría un fichero que nadie pidió.
        reparto = _reparto_del_dataset("mis_datos.csv", self._CSV)
        self.assertEqual(reparto, {"mis_datos.csv": self._CSV})

    def test_con_una_sola_fila_no_se_parte(self):
        # Partir dejaría una de las dos partes vacía, y el entrenador la leería
        # como un dataset sin filas.
        uno = "a,y\n1,2\n"
        self.assertEqual(_reparto_del_dataset("d-synthetic-train.csv", uno),
                         {"d-synthetic-train.csv": uno})


class ElCorteLO_DECIDE_UN_SOLO_SITIOTest(unittest.TestCase):
    """`generate-dataset` y `verify` parten con la MISMA regla.

    Escrita dos veces, dejaría de coincidir sin que nadie se entere — y lo que
    se nota es un `FAIL` de R3 sobre un paquete correcto.
    """

    def test_el_CLI_llama_a_la_regla_comun(self):
        fuente = (Path(__file__).resolve().parents[1] / "matrixai" / "cli.py").read_text(
            encoding="utf-8")
        self.assertIn("particion_sintetica", fuente)
        self.assertIn("corte_train_eval(len(all_rows))", fuente)

    def test_ninguna_parte_se_queda_VACIA(self):
        for total in range(2, 40):
            with self.subTest(total=total):
                train = corte_train_eval(total)
                self.assertGreaterEqual(train, 1)
                self.assertGreaterEqual(total - train, 1)

    def test_el_nombre_del_eval_sale_del_de_train(self):
        self.assertEqual(nombres_del_dataset("d/x-synthetic-train.csv"),
                         ("d/x-synthetic-train.csv", "d/x-synthetic-eval.csv"))
        self.assertIsNone(nombres_del_dataset("x.csv"))
        self.assertIsNone(nombres_del_dataset(""))


class ConducidoDE_VERDAD_DesdeOtroDirectorioTest(unittest.TestCase):
    """Probar la función no es probar el producto.

    Los tests de arriba miran las dos piezas; éste construye un paquete con el
    CLI, lo LLEVA A OTRO SITIO —como hace quien descomprime el ZIP— y verifica
    desde allí. Ése es el escenario donde los dos defectos vivían: desde el
    directorio de construcción, `datos/` seguía en el disco y todo parecía bien.
    """

    _MXAI = """PROJECT CelsiusToKelvin

VECTOR Reading[1]
  celsius: Scalar[-50, 150]
END

PARAM W1 Vector[1]
END

PARAM b1 Scalar
END

FUNCTION PredictedKelvinModel
  predicted_kelvin: Scalar = linear(W1 * Reading + b1)
END

GRAPH
  Reading -> PredictedKelvinModel
END
"""

    _MXTRAIN = """MODEL kelvin.mxai

DATASET KelvinData
  SOURCE csv("datos/celsiustokelvin-synthetic-train.csv")
  INPUT Reading FROM COLUMNS [celsius]
  TARGET predicted_kelvin: Scalar[200, 450]
  SPLIT train=0.8 validation=0.2 seed=42
  BATCH size=8
END

LOSS KelvinLoss
  TYPE mse
  PREDICTION predicted_kelvin
  TARGET predicted_kelvin
END

OPTIMIZER KelvinOptimizer
  TYPE sgd
  LEARNING_RATE 0.5
  UPDATE W1, b1
END

RUN
  EPOCHS 200
END
"""

    _RECETA = "kelvin = celsius + 273.15\n"

    def _construir(self, taller: Path) -> Path:
        import os
        import subprocess
        import sys

        (taller / "kelvin.mxai").write_text(self._MXAI, encoding="utf-8")
        (taller / "kelvin.mxtrain").write_text(self._MXTRAIN, encoding="utf-8")
        (taller / "receta.txt").write_text(self._RECETA, encoding="utf-8")
        raiz = Path(__file__).resolve().parents[1]
        for orden in (
            ["generate-dataset", "kelvin.mxai", "--training", "kelvin.mxtrain",
             "--rows", "300", "--seed", "20260825", "--mode", "coherent",
             "--recipe", "receta.txt", "-o", "datos"],
            ["train", "kelvin.mxai", "--training", "kelvin.mxtrain", "--output",
             "runs/v1", "--recipe", "receta.txt", "--dataset-manifest",
             "datos/celsiustokelvin-synthetic-manifest.json"],
            ["export-bundle", "kelvin.mxai", "--params", "runs/v1/params.best.json",
             "--outdir", "paquete", "--training", "kelvin.mxtrain",
             "--data-recipe", "receta.txt", "--from-run", "runs/v1"],
        ):
            entorno = dict(os.environ, PYTHONPATH=str(raiz))
            r = subprocess.run([sys.executable, "-m", "matrixai", *orden],
                               capture_output=True, text=True, cwd=str(taller),
                               env=entorno)
            self.assertEqual(r.returncode, 0, f"{orden[0]}: {r.stderr}")
        return taller / "paquete"

    def test_desde_un_directorio_LIMPIO_las_cuatro_etapas_pasan(self):
        import shutil
        from tempfile import TemporaryDirectory

        from matrixai.export.verify import verify_package

        with TemporaryDirectory() as tmp:
            taller = Path(tmp) / "obra"
            taller.mkdir()
            paquete = self._construir(taller)
            # AQUÍ está la prueba: el paquete se lleva LEJOS de `datos/`, que es
            # exactamente lo que pasa al descomprimir el ZIP en otra máquina.
            lejos = Path(tmp) / "descargado" / "paquete"
            lejos.parent.mkdir()
            shutil.copytree(paquete, lejos)
            informe = verify_package(lejos, run_training=True, locale="en")

        etapas = {n: e["status"] for n, e in informe["stages"].items()}
        self.assertEqual(etapas, {"manifest": "PASS", "R1": "PASS",
                                  "training": "PASS", "R3": "PASS"},
                         informe["stages"])


class UnZIP_SE_PUEDE_VerificarTest(unittest.TestCase):
    """Lo que la gente se descarga es un ZIP.

    `verify` solo sabía abrir directorios, y con un `.zip` respondía «the package
    carries no reproduce.json» — que es FALSO: lo lleva dentro, y el verificador
    nunca lo abrió. Medido el 2026-08-26 sobre los tres paquetes de la galería:
    los tres decían eso y los tres lo traían. Un motivo equivocado manda a
    arreglar lo que no está roto.
    """

    def _zip_de(self, carpeta: Path, destino: Path, prefijo: str = "paquete") -> Path:
        import zipfile

        with zipfile.ZipFile(destino, "w") as z:
            for fichero in sorted(carpeta.rglob("*")):
                if fichero.is_file():
                    z.write(fichero, f"{prefijo}/{fichero.relative_to(carpeta).as_posix()}")
        return destino

    def test_las_cuatro_etapas_salen_igual_que_desde_la_carpeta(self):
        from tempfile import TemporaryDirectory

        from matrixai.export.verify import verify_package

        constructor = ConducidoDE_VERDAD_DesdeOtroDirectorioTest()
        with TemporaryDirectory() as tmp:
            taller = Path(tmp) / "obra"
            taller.mkdir()
            paquete = constructor._construir(taller)
            zipeado = self._zip_de(paquete, Path(tmp) / "paquete.zip")
            desde_carpeta = verify_package(paquete, run_training=True, locale="en")
            desde_zip = verify_package(zipeado, run_training=True, locale="en")

        self.assertEqual(
            {n: e["status"] for n, e in desde_zip["stages"].items()},
            {n: e["status"] for n, e in desde_carpeta["stages"].items()})
        self.assertEqual({n: e["status"] for n, e in desde_zip["stages"].items()},
                         {"manifest": "PASS", "R1": "PASS", "training": "PASS",
                          "R3": "PASS"})

    def test_un_ZIP_que_ESCAPA_no_se_descomprime_a_medias(self):
        import zipfile
        from tempfile import TemporaryDirectory

        from matrixai.export.verify import verify_package

        with TemporaryDirectory() as tmp:
            malo = Path(tmp) / "malo.zip"
            with zipfile.ZipFile(malo, "w") as z:
                z.writestr("paquete/reproduce.json", "{}")
                z.writestr("../fuera.txt", "no debería salir de aquí")
            informe = verify_package(malo, locale="en")

        # Ni una entrada: se rechaza el archivo ENTERO. Saltarse solo la mala
        # dejaría escrito lo demás de un archivo que ya demostró qué pretende.
        self.assertFalse((Path(tmp).parent / "fuera.txt").exists())
        self.assertEqual({e["status"] for e in informe["stages"].values()},
                         {"INCOMPARABLE"})
        self.assertIn("escapes it", informe["stages"]["manifest"]["reason"])


class ElCoreNoDECLARA_LaLicenciaDeUnosDatosQueNoHaVistoTest(unittest.TestCase):
    """2ª auditoría externa, hallazgo 2 residual — la contradicción de la galería.

    Cuando falta la receta, el manifiesto redactaba «it was trained on real data
    that is **not redistributable**». Eso es un hecho sobre la licencia de unos
    datos que este código no ha visto: un modelo sin receta puede haberse
    entrenado con datos perfectamente publicables. En la galería quedó a la
    vista — el caso de la lluvia ofrece su CSV para descargar mientras su propio
    paquete decía que no se puede redistribuir.
    """

    def test_el_motivo_dice_lo_que_FALTA_no_por_que(self):
        from matrixai.export.reproduce import _MISSING_REASONS as MOTIVOS

        texto = MOTIVOS["recipe"]
        self.assertIn("no data recipe", texto)
        self.assertIn("cannot be regenerated", texto)

    def test_y_ya_NO_afirma_nada_sobre_redistribuir(self):
        from matrixai.export import reproduce

        fuente = Path(reproduce.__file__).read_text(encoding="utf-8")
        # Solo en el comentario que explica por qué se quitó, no en lo que se
        # publica: un barrido a secas daría un falso rojo, y por eso se mira el
        # diccionario de motivos y no el fichero entero.
        from matrixai.export.reproduce import _MISSING_REASONS as MOTIVOS

        for clave, texto in MOTIVOS.items():
            with self.subTest(clave=clave):
                self.assertNotIn("redistributable", texto)
        self.assertIn("hallazgo 2 residual", fuente)
