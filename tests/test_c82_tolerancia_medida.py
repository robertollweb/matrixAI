"""CONTRATO 82 §5 bis — la TOLERANCIA, medida y con su alcance.

Decisión de Roberto (2026-08-20): medirla en ESTE entorno y publicarla
**con su alcance escrito**, en vez de dejar `R3` en `INCOMPARABLE` para
siempre. Era lo único que faltaba: `split`, `dataset_sha256` y
`direction` ya estaban.

**La medición**: el mismo entrenamiento, misma semilla, misma máquina —
`SupervisedTrainer` 5 pasadas y `DenseSupervisedTrainer` 3. **Rango
0.000e+00 en las dos**: los valores salieron idénticos hasta el último
dígito, no parecidos. Por eso `tolerance_abs = 0.0` **no es** el cero
inventado que el contrato rechaza: la diferencia entre los dos ceros es
exactamente el alcance que va escrito al lado.

Y de ahí sale la regla que hace honesto todo esto: **una tolerancia solo
vale donde se midió**. Un paquete medido en una máquina y verificado en
otra saldría `FAIL` por diferencias que su tolerancia nunca prometió
cubrir, y el informe diría «se sale de su tolerancia» — que se lee como
que alguien tocó algo. Es la misma línea que «un fallo por falta de
acceso no es una manipulación».
"""

import unittest
from unittest.mock import patch

from matrixai.export.reproduce import _normalize_metrics
from matrixai.export.verify import (
    _fuera_del_alcance_de_la_tolerancia,
    _verificar_r3,
)
from matrixai.training.metric_identity import (
    ALCANCE_DE_LA_TOLERANCIA,
    TOLERANCIA_MEDIDA,
    tolerancia_medida,
)

_SHA = "b" * 64


def _metrica(**extra):
    base = {"name": "accuracy", "value": 0.9, "split": "validation",
            "dataset_sha256": _SHA}
    base.update(extra)
    return _normalize_metrics([base])[0]


class LaToleranciaVIAJACON_SU_ALCANCETest(unittest.TestCase):
    def test_las_dos_cosas_juntas_o_ninguna(self):
        """Una tolerancia sin alcance haría que un `PASS` de R3 se leyera
        como «reproduce igual en cualquier sitio»."""
        m = _metrica()
        self.assertEqual(m["tolerance_abs"], TOLERANCIA_MEDIDA)
        self.assertEqual(m["tolerance_scope"], ALCANCE_DE_LA_TOLERANCIA)

    def test_con_ella_la_metrica_YA_es_comparable(self):
        self.assertTrue(_metrica()["comparable"])

    def test_una_metrica_que_el_catalogo_NO_conoce_no_recibe_tolerancia(self):
        """Declararla para algo cuya repetibilidad nadie ha medido sería el
        cero inventado otra vez, solo que con más letra pequeña."""
        self.assertIsNone(tolerancia_medida("kappa_de_cohen"))
        m = _metrica(name="kappa_de_cohen")
        self.assertIsNone(m["tolerance_abs"])
        self.assertFalse(m["comparable"])

    def test_lo_que_declara_el_llamante_MANDA(self):
        m = _metrica(tolerance_abs=0.05, tolerance_scope="matriz v1")
        self.assertEqual(m["tolerance_abs"], 0.05)
        self.assertEqual(m["tolerance_scope"], "matriz v1")

    def test_si_el_llamante_declara_la_RELATIVA_tampoco_se_le_repone(self):
        """Media tolerancia suya y media del catálogo serían dos umbrales
        de dos sitios distintos aplicados a la misma métrica."""
        m = _metrica(tolerance_rel=0.01)
        self.assertIsNone(m["tolerance_abs"])
        self.assertEqual(m["tolerance_rel"], 0.01)


class UnaToleranciaSoloValeDondeSeMIDIOTest(unittest.TestCase):
    _COMPARABLE = [{"name": "accuracy", "tolerance_scope": "same_environment_same_seed"}]

    def _manifiesto(self, digest):
        return {"environment": {"environment_sha256": digest}}

    def test_en_EL_MISMO_entorno_la_tolerancia_aplica(self):
        from matrixai.export.reproduce import build_environment
        actual = build_environment()["environment_sha256"]
        self.assertIsNone(
            _fuera_del_alcance_de_la_tolerancia(self._manifiesto(actual), self._COMPARABLE))

    def test_en_OTRO_entorno_NO_aplica_y_se_dice_por_que(self):
        motivo = _fuera_del_alcance_de_la_tolerancia(
            self._manifiesto("a" * 64), self._COMPARABLE)
        self.assertIsNotNone(motivo)
        self.assertIn("would not prove the package wrong", motivo)

    def test_el_motivo_enseña_LOS_DOS_digests(self):
        """«Otro entorno» a secas no deja ver en qué se diferencia."""
        motivo = _fuera_del_alcance_de_la_tolerancia(
            self._manifiesto("a" * 64), self._COMPARABLE)
        self.assertIn("a" * 16, motivo)

    def test_sin_digest_de_entorno_tampoco_se_da_por_bueno(self):
        motivo = _fuera_del_alcance_de_la_tolerancia({}, self._COMPARABLE)
        self.assertIn("no environment digest", motivo)

    def test_un_alcance_DESCONOCIDO_no_se_da_por_bueno(self):
        """Fallo cerrado: lo que este verificador no sabe interpretar no
        se presume aplicable."""
        motivo = _fuera_del_alcance_de_la_tolerancia(
            self._manifiesto("a" * 64),
            [{"name": "accuracy", "tolerance_scope": "lo_que_sea"}])
        self.assertIn("does not know", motivo)

    def test_sin_alcance_declarado_no_se_estorba(self):
        """Un paquete anterior a esto no declara alcance; no se le inventa
        una restricción que nadie escribió."""
        self.assertIsNone(
            _fuera_del_alcance_de_la_tolerancia(self._manifiesto("a" * 64),
                                                [{"name": "accuracy"}]))


class R3DA_UN_VEREDICTO_DE_VERDADTest(unittest.TestCase):
    """Lo que cambia todo esto: R3 deja de contestar `INCOMPARABLE` a un
    paquete completo y pasa a decir PASS o FAIL. Con el manifiesto que
    escribe el escritor de verdad, no uno a mano."""

    def _manifiesto(self, valor=0.9):
        from matrixai.export.reproduce import build_environment
        return {"environment": build_environment(),
                "metrics": _normalize_metrics([{
                    "name": "accuracy", "value": valor, "split": "validation",
                    "dataset_sha256": _SHA}])}

    def _entrenamiento(self, valor):
        return {"status": "PASS", "metrics": {"accuracy": valor}}

    def test_el_MISMO_valor_en_el_MISMO_entorno_es_PASS(self):
        r = _verificar_r3(self._manifiesto(0.9), self._entrenamiento(0.9))
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["checked"], ["accuracy"])

    def test_OTRO_valor_en_el_mismo_entorno_es_FAIL_con_los_dos_numeros(self):
        """La tolerancia medida es 0.0: aquí reproducir es reproducir
        EXACTO, y cualquier diferencia es una discrepancia de verdad."""
        r = _verificar_r3(self._manifiesto(0.9), self._entrenamiento(0.7))
        self.assertEqual(r["status"], "FAIL")
        fuera = r["metrics"][0]
        self.assertEqual((fuera["published"], fuera["obtained"]), (0.9, 0.7))
        self.assertEqual(fuera["tolerance"], 0.0)

    def test_en_OTRO_entorno_una_diferencia_NO_es_FAIL(self):
        """Lo que separa un verificador honesto de uno que acusa: la
        tolerancia no se midió para esa máquina."""
        manifiesto = self._manifiesto(0.9)
        manifiesto["environment"]["environment_sha256"] = "c" * 64
        r = _verificar_r3(manifiesto, self._entrenamiento(0.7))
        self.assertEqual(r["status"], "INCOMPARABLE")
        self.assertIn("would not prove the package wrong", r["reason"])

    def test_y_en_otro_entorno_tampoco_se_apunta_un_PASS_gratis(self):
        """Igual de malo por el otro lado: dar por bueno lo que no se ha
        podido comprobar."""
        manifiesto = self._manifiesto(0.9)
        manifiesto["environment"]["environment_sha256"] = "c" * 64
        r = _verificar_r3(manifiesto, self._entrenamiento(0.9))
        self.assertEqual(r["status"], "INCOMPARABLE")


class ElValorMEDIDOTest(unittest.TestCase):
    def test_el_cero_esta_MEDIDO_no_supuesto(self):
        """Si alguien sube este número sin volver a medir, esta prueba se
        pone en rojo — que es justo lo que tiene que pasar."""
        self.assertEqual(TOLERANCIA_MEDIDA, 0.0)
        self.assertEqual(ALCANCE_DE_LA_TOLERANCIA, "same_environment_same_seed")

    def test_el_entrenamiento_de_este_entorno_ES_repetible(self):
        """La medición, en pequeño y dentro de la suite: dos pasadas del
        mismo entrenamiento con la misma semilla dan lo MISMO. Si dejara de
        cumplirse, la tolerancia declarada dejaría de tener respaldo."""
        import shutil
        import tempfile
        from pathlib import Path

        from matrixai.training.parser import parse_training_file
        from matrixai.training.trainer import SupervisedTrainer

        def pasada():
            taller = Path(tempfile.mkdtemp())
            try:
                shutil.copy2("examples/celsius_to_kelvin.mxtrain", taller / "t.mxtrain")
                (taller / "examples").mkdir()
                for f in ("celsius_to_kelvin.mxai", "celsius_to_kelvin.train.csv"):
                    shutil.copy2(f"examples/{f}", taller / "examples" / f)
                spec = parse_training_file(taller / "t.mxtrain")
                r = SupervisedTrainer().train(
                    spec, output_dir=str(taller / "out"), base_path=taller,
                    training_path=taller / "t.mxtrain")
                return (getattr(r, "final_train_loss", None),
                        getattr(r, "best_validation_loss", None))
            finally:
                shutil.rmtree(taller, ignore_errors=True)

        self.assertEqual(pasada(), pasada())


if __name__ == "__main__":
    unittest.main()
