"""CONTRATO 83-C4 — la propuesta del RefinementAgent, supervisada.

*«…la propuesta del `RefinementAgent` se enseña **como diff** con sus dos
salidas: aceptar —que dispara un ciclo nuevo y versionado— o rechazar,
**con motivo, y el motivo se guarda**.»*

Lo que estas pruebas defienden:

* **El diff se calcula UNA vez y en el core** (invariante §5.1: no se
  recalcula lo que el core ya dice; dos sitios calculando lo mismo acaban
  dando dos respuestas).
* **Un rechazo sin motivo no es un rechazo válido.** El contrato pide que
  el motivo se guarde: aceptar un rechazo mudo dejaría constancia de que
  alguien dijo que no y de nada más, que es casi no dejar constancia.
* **La decisión la toma una persona y queda registrada** con quién y
  cuándo (§5.2: promover es un acto humano).
"""

import unittest

from matrixai.continual.supervision import (
    DecisionInvalida,
    diff_de_la_propuesta,
    registrar_decision,
)


class ElDiffSeVeTest(unittest.TestCase):
    def test_marca_lo_que_se_quita_y_lo_que_se_pone(self):
        lineas = diff_de_la_propuesta("predice A\ncon B\n", "predice A\ncon C\n")
        marcas = {l["mark"] for l in lineas}
        self.assertIn("-", marcas)
        self.assertIn("+", marcas)
        # Y lo que NO cambia también viaja: un diff que solo enseña los
        # cambios obliga a abrir el original al lado para entenderlos.
        self.assertIn(" ", marcas)

    def test_sin_cambios_lo_dice_en_vez_de_devolver_una_lista_vacia(self):
        """Una lista vacía se lee como «no hay propuesta». Que la
        propuesta sea IGUAL al original es otra cosa, y hay que verla."""
        lineas = diff_de_la_propuesta("igual\n", "igual\n")
        self.assertTrue(lineas)
        self.assertEqual({l["mark"] for l in lineas}, {" "})

    def test_no_se_pierde_ninguna_linea(self):
        original, propuesto = "a\nb\nc\n", "a\nX\nc\n"
        lineas = diff_de_la_propuesta(original, propuesto)
        texto = "".join(l["text"] for l in lineas)
        for pieza in ("a", "b", "c", "X"):
            self.assertIn(pieza, texto)


class RechazarSinMotivoNoValeTest(unittest.TestCase):
    def test_un_rechazo_sin_motivo_se_RECHAZA(self):
        for vacio in (None, "", "   ", "\n"):
            with self.subTest(motivo=vacio):
                with self.assertRaises(DecisionInvalida):
                    registrar_decision("ref-1", aceptada=False, motivo=vacio,
                                       quien="roberto", cuando="2026-08-20T10:00:00Z")

    def test_aceptar_no_exige_motivo(self):
        """Aceptar es seguir el camino que el sistema propuso; rechazar es
        apartarse de él, y eso es lo que hay que poder explicar después."""
        registro = registrar_decision("ref-1", aceptada=True, motivo=None,
                                      quien="roberto", cuando="2026-08-20T10:00:00Z")
        self.assertTrue(registro["accepted"])

    def test_el_registro_dice_QUIEN_y_CUANDO(self):
        registro = registrar_decision("ref-1", aceptada=False, motivo="no me convence",
                                      quien="roberto", cuando="2026-08-20T10:00:00Z")
        self.assertEqual(registro["who"], "roberto")
        self.assertEqual(registro["when"], "2026-08-20T10:00:00Z")
        self.assertEqual(registro["reason"], "no me convence")
        self.assertEqual(registro["refinement_id"], "ref-1")

    def test_sin_quien_no_hay_decision(self):
        """§5.2: promover es un acto humano y queda registrado con quién.
        Un registro anónimo no cumple eso, así que no se acepta."""
        with self.assertRaises(DecisionInvalida):
            registrar_decision("ref-1", aceptada=True, motivo=None,
                               quien="  ", cuando="2026-08-20T10:00:00Z")


if __name__ == "__main__":
    unittest.main()


class LaDerivaEnElTiempoTest(unittest.TestCase):
    """*«La deriva que P22 calcula, visible en el tiempo»*.

    Se PIDE y se pinta: no se recalcula (§5.1). Y lo que no se midió no
    se dibuja como si se hubiera medido — **un dibujo afirma por
    omisión**, y una serie con un hueco pintado a cero diría que ese día
    no hubo deriva cuando lo que hubo fue que no se miró.
    """

    def _informe(self, cuando, *, detectada=False, valor=0.1, saltada=False,
                 suficientes=True):
        return {
            "checked_at": cuando,
            "drift_detected": detectada,
            "enough_samples": suficientes,
            "total_production_samples": 100 if suficientes else 3,
            # `results` es un DICCIONARIO feature → resultado: es lo que
            # produce el core (`DriftReport.results`). Este fixture describía
            # una LISTA —una forma que el core no produce— y por eso la
            # función pasaba sus pruebas mientras la serie salía vacía con
            # datos de verdad. Un fixture describe un modelo que existe.
            "results": {"edad": {
                "feature": "edad", "method": "psi",
                "observed_value": None if saltada else valor,
                "threshold": 0.2, "drift_detected": detectada,
                "samples_used": 0 if saltada else 100,
                "enough_samples": suficientes,
                "skipped": saltada,
                "skip_reason": "not enough samples" if saltada else None,
            }},
        }

    def test_los_puntos_van_en_ORDEN_por_su_fecha(self):
        from matrixai.continual.supervision import serie_de_deriva
        serie = serie_de_deriva([
            self._informe("2026-08-03T00:00:00Z"),
            self._informe("2026-08-01T00:00:00Z"),
            self._informe("2026-08-02T00:00:00Z"),
        ])
        fechas = [p["checked_at"] for p in serie["points"]]
        self.assertEqual(fechas, sorted(fechas))

    def test_una_medicion_SALTADA_no_se_dibuja_como_un_cero(self):
        from matrixai.continual.supervision import serie_de_deriva
        serie = serie_de_deriva([self._informe("2026-08-01T00:00:00Z", saltada=True)])
        punto = serie["points"][0]["features"]["edad"]
        self.assertIsNone(punto["value"])
        self.assertTrue(punto["skipped"])
        # Y con su motivo: un hueco sin explicación se lee como un fallo.
        self.assertTrue(punto["skip_reason"])

    def test_sin_muestras_suficientes_se_DICE(self):
        from matrixai.continual.supervision import serie_de_deriva
        serie = serie_de_deriva([
            self._informe("2026-08-01T00:00:00Z", suficientes=False)])
        self.assertFalse(serie["points"][0]["enough_samples"])

    def test_lista_las_features_que_aparecen_para_poder_pintarlas(self):
        from matrixai.continual.supervision import serie_de_deriva
        serie = serie_de_deriva([self._informe("2026-08-01T00:00:00Z")])
        self.assertEqual(serie["features"], ["edad"])

    def test_sin_informes_lo_dice_en_vez_de_una_serie_vacia(self):
        """Una serie vacía se pinta como una gráfica en blanco, que se lee
        como «no hay deriva». Lo que hay es que no se ha medido nunca."""
        from matrixai.continual.supervision import serie_de_deriva
        serie = serie_de_deriva([])
        self.assertEqual(serie["points"], [])
        self.assertTrue(serie["never_measured"])

    def test_no_recalcula_el_veredicto_del_core(self):
        """§5.1: la pantalla pide y pinta. Si aquí se recalculara
        `drift_detected` a partir del valor y el umbral, habría dos sitios
        decidiendo lo mismo — y acabarían discrepando."""
        from matrixai.continual.supervision import serie_de_deriva
        # Valor por DEBAJO del umbral pero el core dice que SÍ hay deriva:
        # manda el core.
        serie = serie_de_deriva([
            self._informe("2026-08-01T00:00:00Z", detectada=True, valor=0.01)])
        self.assertTrue(serie["points"][0]["drift_detected"])
