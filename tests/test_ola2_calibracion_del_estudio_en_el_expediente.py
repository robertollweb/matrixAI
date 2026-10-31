"""Ola 2 (30-10-2026) — el expediente de un ESTUDIO dice la calibración que el estudio midió.

Antes: la línea «Calibración (ECE)» leía `clinical_profile.json`, que un estudio no trae, y el expediente decía «no se ha medido
la calibración» / «este paquete no trae la curva de calibración» con las curvas del estudio (141) llevando su ECE. Ahora el
manifiesto del estudio lleva `study.evaluation_calibration` (leído de las curvas con `calibracion_de_las_curvas`, la misma
función que usa el Informe Vivo) y el expediente distingue TRES cosas:

* MEDIDO por el estudio (la cifra, con sus filas y tramos);
* NO MEDIDO / NO APLICA (el motivo del núcleo);
* NO FIGURA en esta exportación (un estudio sin curvas: no se sabe si se midió).
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from matrixai.estudio.curvas_de_la_evaluacion import calibracion_de_las_curvas, curvas_de_la_evaluacion
from matrixai.estudio.metricas import Muestra
from matrixai.export.manifiesto_de_estudio import (
    ReproduceManifestError,
    escribir_manifiesto_de_estudio,
    manifiesto_de_estudio,
)
from matrixai.export.probast import huecos_probast, mapa_probast
from matrixai.export.tripod import ficha_tripod

RUTA = "reproduce.json#study.evaluation_calibration"


def _curvas_binarias(n=300, semilla=1):
    azar = random.Random(semilla)
    p = [azar.random() for _ in range(n)]
    y = ["1" if azar.random() < q else "0" for q in p]
    m = Muestra.binaria(y, classes=("0", "1"), positive_label="1", probabilidades=p)
    return {"disponible": True, **curvas_de_la_evaluacion(m, n_bins=10, metodo_de_bins="frecuencia_igual")}


def _curvas_con_pocas_filas():
    return _curvas_binarias(n=40)


def _curvas_multiclase(n=300):
    azar = random.Random(3)
    clases = ("a", "b", "c")
    probs, y = [], []
    for _ in range(n):
        crudo = [azar.random() for _ in clases]
        t = sum(crudo)
        probs.append([c / t for c in crudo])
        y.append(azar.choice(clases))
    m = Muestra.multiclase(y, classes=clases, probabilidades=probs)
    return {"disponible": True, **curvas_de_la_evaluacion(m, n_bins=10, metodo_de_bins="frecuencia_igual")}


def _curvas_de_regresion(n=100):
    azar = random.Random(5)
    real = [float(i) for i in range(n)]
    m = Muestra.regresion(real, [r + azar.random() for r in real])
    return {"disponible": True, **curvas_de_la_evaluacion(m)}


class _ConPaquete(unittest.TestCase):
    def paquete(self, curvas, **extra):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        ruta = Path(t.name)
        escribir_manifiesto_de_estudio(ruta, filas=300, motor_del_modelo_entregado="sklearn.hgb",
                                       curvas_de_la_evaluacion=curvas, **extra)
        return ruta

    @staticmethod
    def linea(ficha, rotulo):
        return next(l for l in ficha.splitlines() if l.startswith(f"- **{rotulo}**"))

    @staticmethod
    def item_probast(ruta, idioma):
        mapa = mapa_probast(ruta, locale=idioma)
        return next(i for dominio in mapa["por_dominio"].values() for i in dominio if i["clave"] == "calibracion")


class ElExtractorDiceTresCosasTest(unittest.TestCase):
    def test_medido(self):
        c = calibracion_de_las_curvas(_curvas_binarias())
        self.assertEqual(c["estado"], "medido")
        self.assertGreaterEqual(c["ece"], 0.0)
        self.assertEqual((c["filas"], c["tramos"], c["tipo"]), (300, 10, "clase_positiva"))
        self.assertIn("positivos", c)

    def test_multiclase_no_inventa_una_clase_positiva(self):
        c = calibracion_de_las_curvas(_curvas_multiclase())
        self.assertEqual(c["estado"], "medido")
        self.assertEqual(c["tipo"], "clase_predicha")
        self.assertNotIn("positivos", c)

    def test_pocas_filas_es_no_medido_con_el_motivo_del_nucleo(self):
        c = calibracion_de_las_curvas(_curvas_con_pocas_filas())
        self.assertEqual(c["estado"], "no_medido")
        self.assertEqual(c["clave"], "ece_con_pocos_tramos")
        self.assertTrue(c["motivo"]["es"] and c["motivo"]["en"])

    def test_regresion_no_aplica_y_no_es_no_medido(self):
        self.assertEqual(calibracion_de_las_curvas(_curvas_de_regresion())["estado"], "no_aplica")

    def test_sin_curvas_no_es_no_medido(self):
        for curvas in (None, {}, {"disponible": False, "clave": "calculo_fallido"}, "x", {"disponible": True}):
            with self.subTest(curvas=curvas):
                self.assertEqual(calibracion_de_las_curvas(curvas), {"estado": "sin_curvas"})


class ElManifiestoLaLlevaTest(_ConPaquete):
    def test_medido_escribe_el_bloque_con_su_cifra(self):
        bloque = manifiesto_de_estudio(curvas_de_la_evaluacion=_curvas_binarias())["study"]["evaluation_calibration"]
        self.assertEqual(bloque["status"], "measured")
        self.assertEqual((bloque["test_rows"], bloque["bins_with_figure"], bloque["kind"]), (300, 10, "positive_class"))
        self.assertEqual(bloque["ece"], calibracion_de_las_curvas(_curvas_binarias())["ece"])

    def test_sin_curvas_no_escribe_NADA_ni_un_null(self):
        for curvas in (None, {"disponible": False, "clave": "x"}):
            self.assertNotIn("evaluation_calibration", manifiesto_de_estudio(curvas_de_la_evaluacion=curvas)["study"])

    def test_el_digest_cubre_el_bloque(self):
        a = manifiesto_de_estudio(curvas_de_la_evaluacion=_curvas_binarias(semilla=1), entorno={"x": 1})
        b = manifiesto_de_estudio(curvas_de_la_evaluacion=_curvas_binarias(semilla=2), entorno={"x": 1})
        self.assertNotEqual(a["manifest_sha256"], b["manifest_sha256"])

    def test_unas_curvas_que_no_son_un_objeto_se_rechazan(self):
        with self.assertRaises(ReproduceManifestError):
            manifiesto_de_estudio(curvas_de_la_evaluacion=[1, 2])


class LaFichaDiceLoQueElEstudioMidioTest(_ConPaquete):
    def test_medido_en_los_dos_idiomas_y_ya_no_dice_que_no_se_midio(self):
        ece = f"{calibracion_de_las_curvas(_curvas_binarias())['ece']:.4f}".rstrip("0")
        for idioma, rotulo in (("es", "Calibración (ECE)"), ("en", "Calibration (ECE)")):
            with self.subTest(idioma=idioma):
                ficha = ficha_tripod(self.paquete(_curvas_binarias()), locale=idioma)
                linea = self.linea(ficha, rotulo)
                self.assertIn(ece, linea)
                self.assertIn(RUTA, linea)
                self.assertIn("300", linea)
                self.assertNotIn("_no disponible_", linea)
                self.assertNotIn("_not available_", linea)
                for frase in ("no trae la curva de calibración", "this package ships no calibration curve",
                              "clinical_profile.json#calibracion.ece"):
                    self.assertNotIn(frase, ficha)

    def test_no_medido_dice_el_motivo_del_nucleo(self):
        ficha = ficha_tripod(self.paquete(_curvas_con_pocas_filas()), locale="es")
        linea = self.linea(ficha, "Calibración (ECE)")
        self.assertIn("intentó medirla", linea)
        self.assertIn("al menos 3 tramos", linea)
        self.assertNotIn("no figura en esta exportación", linea)

    def test_regresion_dice_que_no_aplica(self):
        linea = self.linea(ficha_tripod(self.paquete(_curvas_de_regresion()), locale="en"), "Calibration (ECE)")
        self.assertIn("does not apply", linea)
        self.assertNotIn("not in this export", linea)

    def test_sin_curvas_dice_no_figura_y_no_que_no_se_midio(self):
        es = self.linea(ficha_tripod(self.paquete(None), locale="es"), "Calibración (ECE)")
        self.assertIn("no figura en esta exportación", es)
        self.assertIn("no se sabe si se midió", es)
        en = self.linea(ficha_tripod(self.paquete(None), locale="en"), "Calibration (ECE)")
        self.assertIn("not in this export", en)

    def test_no_se_repite_en_la_seccion_de_no_disponible(self):
        ficha = ficha_tripod(self.paquete(None), locale="es")
        self.assertEqual(sum(1 for l in ficha.splitlines() if l.startswith(("- **Calibración (ECE)**", "- **Calibración**"))), 1)

    def test_multiclase_no_habla_de_positivos(self):
        linea = self.linea(ficha_tripod(self.paquete(_curvas_multiclase()), locale="es"), "Calibración (ECE)")
        self.assertIn("confianza de la clase predicha", linea)
        self.assertNotIn("positivos", linea)
        self.assertNotIn("None", linea)


class ProbastLeeLoMismoTest(_ConPaquete):
    def test_medido(self):
        item = self.item_probast(self.paquete(_curvas_binarias()), "es")
        self.assertEqual(item["estado"], "medido")
        self.assertEqual(item["ruta"], RUTA)
        self.assertEqual(item["valor"]["status"], "measured")

    def test_no_medido_y_no_figura_son_falta_con_motivos_DISTINTOS(self):
        a = self.item_probast(self.paquete(_curvas_con_pocas_filas()), "es")
        b = self.item_probast(self.paquete(None), "es")
        self.assertEqual((a["estado"], b["estado"]), ("falta", "falta"))
        self.assertNotEqual(a["motivo"], b["motivo"])
        self.assertIn("no figura en esta exportación", b["motivo"])
        self.assertNotIn("no figura", a["motivo"])

    def test_los_huecos_markdown_tambien(self):
        self.assertNotIn("Calibración (ECE)**: _no disponible_",
                         huecos_probast(self.paquete(_curvas_binarias()), locale="es"))


class SiSeDeclaroUnPerfilClinicoSigueLeyendoseTest(_ConPaquete):
    """Un estudio sin curvas pero con un perfil clínico que SÍ trae el ECE no pierde esa cifra."""

    def test_el_perfil_cubre_un_estudio_sin_curvas(self):
        ruta = self.paquete(None)
        (ruta / "clinical_profile.json").write_text(json.dumps({"calibracion": {"ece": 0.05}}), encoding="utf-8")
        linea = self.linea(ficha_tripod(ruta, locale="es"), "Calibración (ECE)")
        self.assertIn("0.05", linea)
        self.assertIn("clinical_profile.json#calibracion.ece", linea)

    def test_las_curvas_del_estudio_mandan_sobre_el_perfil(self):
        ruta = self.paquete(_curvas_binarias())
        (ruta / "clinical_profile.json").write_text(json.dumps({"calibracion": {"ece": 0.99}}), encoding="utf-8")
        linea = self.linea(ficha_tripod(ruta, locale="es"), "Calibración (ECE)")
        self.assertNotIn("0.99", linea)
        self.assertIn(RUTA, linea)


if __name__ == "__main__":
    unittest.main()
