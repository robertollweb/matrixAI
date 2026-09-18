"""109-C3 — la comparación con el baseline llega al perfil clínico y al expediente.

Hasta hoy esa línea del expediente salía SIEMPRE «falta»: el core sabía
compararlo (`comparar_candidatos`, 105-C5), pero el perfil no tenía dónde
guardarlo. Ahora `PerfilClinico` lleva la comparación emparejada y el nombre del
baseline, o el motivo cerrado de por qué no se pudo plantear. Son tres casos
distintos —comparación, «no se pudo» y «no consta»— y cada uno dice lo suyo.

El número de la ficha se comprueba contra la aritmética hecha a mano, no contra
otra ejecución del mismo código: sobre la cohorte de diez filas de 109-C2, el
AUROC del modelo es 22/24 (de los 24 pares positivo-negativo, 22 los ordena
bien: 0,90, 0,75 y 0,60 superan a los seis negativos, y 0,35 a cuatro), y el del
baseline constante es 0,5. Diferencia: 22/24 − 1/2 = 0,41666…
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.estudio import EsquemaInvalido  # noqa: E402
from matrixai.estudio.comparaciones import ComparacionEmparejada, comparar_candidatos  # noqa: E402
from matrixai.estudio.incertidumbre import Intervalo  # noqa: E402
from matrixai.estudio.metricas import Muestra  # noqa: E402
from matrixai.estudio.perfil_clinico import (  # noqa: E402
    _T,
    MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE,
    PerfilClinico,
    curva_de_decision,
    ficha_del_perfil,
    tabla_de_umbrales,
)
from matrixai.export.expediente_clinico import (  # noqa: E402
    _MOTIVOS,
    FALTA,
    MEDIDO,
    POR_CLAVE,
    ExpedienteClinico,
    texto_del_valor,
)
from test_c109_c3_expediente import paquete_clinico  # noqa: E402

_Y = ("si", "si", "si", "si", "no", "no", "no", "no", "no", "no")
_P = (0.90, 0.75, 0.60, 0.35, 0.55, 0.45, 0.30, 0.20, 0.10, 0.05)
_DIFERENCIA_A_MANO = 22 / 24 - 1 / 2
_RUTA = "clinical_profile.json#comparacion_con_el_baseline"


def _modelo() -> Muestra:
    return Muestra.binaria(_Y, classes=("no", "si"), positive_label="si", probabilidades=_P)


def _baseline(y=_Y) -> Muestra:
    """El baseline del Studio: predice lo mismo para todas las filas."""
    return Muestra.binaria(y, classes=("no", "si"), positive_label="si",
                           probabilidades=(0.4,) * len(y))


def _comparacion(*, diseno: str = "iid", baseline: Muestra | None = None):
    return comparar_candidatos("auroc", _modelo(), baseline or _baseline(), diseno=diseno,
                               estimando="fixed_model_on_population", semilla=3,
                               remuestras=200)


VEREDICTOS_CON_INTERVALO = ("mejora", "inferioridad", "equivalencia_practica", "inconcluso")
_MOTIVO_DE_PRUEBA = {"es": "motivo de prueba en castellano", "en": "test reason in English"}


def _a_mano(veredicto, *, intervalo_disponible=True, motivo=None):
    """Una comparación construida a mano, para ejercitar cada veredicto: el
    cálculo real (`comparar_candidatos`) ya tiene sus pruebas, y aquí lo que se
    prueba es cómo lo ESCRIBE el perfil."""
    comun = dict(metric_id="auroc", design="iid", estimand="fixed_model_on_population",
                 seed=0, n_resamples=10, level=0.95)
    if veredicto == "incomparable":
        return ComparacionEmparejada(metric_id="auroc", diseno="iid",
                                     estimando="fixed_model_on_population",
                                     veredicto="incomparable", margen_equivalencia=None,
                                     diferencia_puntual=None, intervalo=None,
                                     undefined_reason=motivo or _MOTIVO_DE_PRUEBA)
    if intervalo_disponible:
        intervalo = Intervalo(**comun, resampling_unit="row", method="iid_paired_bootstrap_stratified",
                              ci_low=0.1234, ci_high=0.5678)
    else:
        intervalo = Intervalo(**comun, resampling_unit="none",
                              undefined_reason=motivo or _MOTIVO_DE_PRUEBA)
    return ComparacionEmparejada(
        metric_id="auroc", diseno="iid", estimando="fixed_model_on_population",
        veredicto=veredicto, margen_equivalencia=0.01 if veredicto == "equivalencia_practica" else None,
        diferencia_puntual=0.3456, intervalo=intervalo)


def _perfil(**cambios) -> PerfilClinico:
    muestra = _modelo()
    base = dict(
        perfil_id="perfil-c109-c3-baseline", evidencia="independent_test", diseno="iid",
        datos_sinteticos=True,
        tabla=tabla_de_umbrales(muestra, umbrales=(0.5,), diseno="iid",
                                estimando="fixed_model_on_population", semilla=1,
                                remuestras=50),
        curva=curva_de_decision(muestra, umbrales_de_probabilidad=(0.2, 0.5)))
    base.update(cambios)
    return PerfilClinico(**base)


class ElPerfilLaGuardaTest(unittest.TestCase):

    def test_viaja_al_json_con_la_diferencia_hecha_a_mano_y_su_baseline(self):
        comparacion = _comparacion()
        cuerpo = _perfil(comparacion_con_el_baseline=comparacion,
                         baseline_comparado="baseline-seleccion").a_json()
        guardada = cuerpo["comparacion_con_el_baseline"]
        self.assertEqual(guardada["baseline"], "baseline-seleccion")
        self.assertEqual(guardada["metric_id"], "auroc")
        self.assertAlmostEqual(guardada["diferencia_puntual"], _DIFERENCIA_A_MANO, places=12)
        self.assertEqual(guardada["veredicto"], comparacion.veredicto)
        self.assertIsNotNone(guardada["intervalo"]["ci_low"])
        self.assertIsNone(cuerpo["sin_comparacion_con_el_baseline"])

    def test_sin_nada_los_dos_campos_salen_nulos(self):
        cuerpo = _perfil().a_json()
        self.assertIsNone(cuerpo["comparacion_con_el_baseline"])
        self.assertIsNone(cuerpo["sin_comparacion_con_el_baseline"])

    def test_el_motivo_de_ausencia_viaja_al_json(self):
        cuerpo = _perfil(sin_comparacion_con_el_baseline="el_ganador_es_el_baseline"
                         ).a_json()
        self.assertIsNone(cuerpo["comparacion_con_el_baseline"])
        self.assertEqual(cuerpo["sin_comparacion_con_el_baseline"],
                         "el_ganador_es_el_baseline")


class NoSeContradiceTest(unittest.TestCase):

    def test_comparacion_y_motivo_de_ausencia_a_la_vez_no(self):
        with self.assertRaises(EsquemaInvalido):
            _perfil(comparacion_con_el_baseline=_comparacion(),
                    baseline_comparado="baseline-seleccion",
                    sin_comparacion_con_el_baseline="el_baseline_no_puntuo")

    def test_una_comparacion_sin_decir_contra_quien_no(self):
        with self.assertRaises(EsquemaInvalido):
            _perfil(comparacion_con_el_baseline=_comparacion())

    def test_un_baseline_nombrado_sin_comparacion_no(self):
        with self.assertRaises(EsquemaInvalido):
            _perfil(baseline_comparado="baseline-seleccion")

    def test_un_motivo_fuera_del_vocabulario_no(self):
        with self.assertRaises(EsquemaInvalido):
            _perfil(sin_comparacion_con_el_baseline="no_me_apetecia")

    def test_la_comparacion_tiene_que_ser_una_ComparacionEmparejada(self):
        """Un mapa con la misma forma no vale: el núcleo la valida porque es tipada."""
        with self.assertRaises(EsquemaInvalido):
            _perfil(comparacion_con_el_baseline=_comparacion().a_json(),
                    baseline_comparado="baseline-seleccion")

    def test_la_comparacion_remuestrea_con_el_MISMO_diseno_que_el_perfil(self):
        """Con otro diseño, el intervalo de la comparación y los de la tabla
        tratarían las mismas filas de dos maneras distintas."""
        otra = _comparacion(diseno="temporal")
        self.assertEqual(otra.diseno, "temporal")
        with self.assertRaises(EsquemaInvalido):
            _perfil(comparacion_con_el_baseline=otra, baseline_comparado="baseline-seleccion")
        # Y el mismo perfil con el diseño bueno sí se construye: el rechazo es
        # por el diseño, no por otra cosa.
        _perfil(comparacion_con_el_baseline=_comparacion(),
                baseline_comparado="baseline-seleccion")


class LaFichaLoDiceTest(unittest.TestCase):

    def test_escribe_la_diferencia_el_intervalo_y_la_frase_del_veredicto(self):
        comparacion = _comparacion()
        perfil = _perfil(comparacion_con_el_baseline=comparacion,
                         baseline_comparado="baseline-seleccion")
        for idioma in ("es", "en"):
            ficha = ficha_del_perfil(perfil, locale=idioma)
            with self.subTest(idioma=idioma):
                self.assertIn(f"## {_T[idioma]['comparacion']}", ficha)
                self.assertIn(f"{_DIFERENCIA_A_MANO:.4f}", ficha)
                self.assertIn(f"{comparacion.intervalo.ci_low:.4f}", ficha)
                self.assertIn(f"{comparacion.intervalo.ci_high:.4f}", ficha)
                self.assertIn("baseline-seleccion", ficha)
                self.assertIn(_T[idioma][f"comparacion_{comparacion.veredicto}"], ficha)
                self.assertNotIn(_T[idioma]["comparacion_no_consta"], ficha)

    def test_cada_veredicto_escribe_SU_frase_y_ninguna_otra(self):
        """Solo se ejercitaba `mejora`: cruzar las frases de dos veredictos dejaba
        todo verde (re-auditoría del 2026-09-18)."""
        for veredicto in VEREDICTOS_CON_INTERVALO:
            perfil = _perfil(comparacion_con_el_baseline=_a_mano(veredicto),
                             baseline_comparado="baseline-seleccion")
            for idioma in ("es", "en"):
                ficha = ficha_del_perfil(perfil, locale=idioma)
                with self.subTest(veredicto=veredicto, idioma=idioma):
                    self.assertIn(_T[idioma][f"comparacion_{veredicto}"], ficha)
                    for otro in set(VEREDICTOS_CON_INTERVALO) - {veredicto}:
                        self.assertNotIn(_T[idioma][f"comparacion_{otro}"], ficha)
                    self.assertIn("[0.1234, 0.5678]", ficha)

    def test_un_intervalo_que_no_se_pudo_calcular_dice_POR_QUE_en_su_idioma(self):
        perfil = _perfil(comparacion_con_el_baseline=_a_mano("inconcluso", intervalo_disponible=False),
                         baseline_comparado="baseline-seleccion")
        for idioma, otro in (("es", "en"), ("en", "es")):
            ficha = ficha_del_perfil(perfil, locale=idioma)
            with self.subTest(idioma=idioma):
                self.assertIn(f"**{_T[idioma]['sin_ic']}**: {_MOTIVO_DE_PRUEBA[idioma]}", ficha)
                self.assertNotIn(_MOTIVO_DE_PRUEBA[otro], ficha)

    def test_incomparable_dice_que_se_intento_y_por_que_en_su_idioma_sin_decir_mismas_filas(self):
        perfil = _perfil(comparacion_con_el_baseline=_a_mano("incomparable"),
                         baseline_comparado="baseline-seleccion")
        for idioma, otro in (("es", "en"), ("en", "es")):
            ficha = ficha_del_perfil(perfil, locale=idioma)
            with self.subTest(idioma=idioma):
                self.assertIn(_T[idioma]["comparacion_incomparable_que"].format(
                    baseline="baseline-seleccion"), ficha)
                self.assertIn(_MOTIVO_DE_PRUEBA[idioma], ficha)
                self.assertNotIn(_MOTIVO_DE_PRUEBA[otro], ficha)
                lead = _T[idioma]["comparacion_que"].split("{")[0].strip()
                self.assertNotIn(lead, ficha, "dice «emparejada sobre las mismas filas» de unas "
                                              "filas que no lo eran")

    def test_cada_motivo_de_ausencia_tiene_SU_frase(self):
        for motivo in MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE:
            perfil = _perfil(sin_comparacion_con_el_baseline=motivo)
            for idioma in ("es", "en"):
                ficha = ficha_del_perfil(perfil, locale=idioma)
                with self.subTest(motivo=motivo, idioma=idioma):
                    self.assertIn(_T[idioma][f"sin_comparacion_{motivo}"], ficha)
                    self.assertNotIn(_T[idioma]["comparacion_no_consta"], ficha)

    def test_sin_nada_dice_que_no_consta_y_no_que_no_se_pudo(self):
        ficha = ficha_del_perfil(_perfil(), locale="es")
        self.assertIn(_T["es"]["comparacion_no_consta"], ficha)
        for motivo in MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE:
            self.assertNotIn(_T["es"][f"sin_comparacion_{motivo}"], ficha)

    def test_filas_que_no_casan_dan_incomparable_y_la_ficha_dice_por_que(self):
        otras = ("no",) + _Y[1:]
        comparacion = _comparacion(baseline=_baseline(otras))
        self.assertEqual(comparacion.veredicto, "incomparable")
        ficha = ficha_del_perfil(_perfil(comparacion_con_el_baseline=comparacion,
                                         baseline_comparado="baseline-seleccion"),
                                 locale="es")
        self.assertIn(comparacion.undefined_reason["es"], ficha)
        self.assertNotIn(_T["es"]["comparacion_diferencia"], ficha)


class ElExpedienteLaLeeTest(unittest.TestCase):
    """La línea que salía SIEMPRE «falta». Por el camino real: un paquete con el
    `clinical_profile.json` que escribe `PerfilClinico.a_json()`."""

    def _campo(self, perfil):
        with TemporaryDirectory() as d:
            expediente = ExpedienteClinico.desde_paquete(paquete_clinico(Path(d), perfil=perfil))
            return expediente.campo(_RUTA)

    def test_con_comparacion_sale_MEDIDO_y_resumida_con_sus_numeros(self):
        comparacion = _comparacion()
        campo = self._campo(_perfil(comparacion_con_el_baseline=comparacion,
                                    baseline_comparado="baseline-seleccion"))
        self.assertEqual(campo.estado, MEDIDO)
        texto = texto_del_valor(POR_CLAVE["comparacion_con_el_baseline"], campo, "es")
        self.assertTrue(texto.startswith("auroc 0.4167 ["), texto)
        self.assertIn(f"· {comparacion.veredicto} · baseline=baseline-seleccion", texto)

    def _resuelto(self, perfil, idioma):
        """Por el camino que usan los DOS informes (`resolver`), no por `campo()`."""
        with TemporaryDirectory() as d:
            expediente = ExpedienteClinico.desde_paquete(paquete_clinico(Path(d), perfil=perfil))
            pares, _ = expediente.resolver(idioma)
        return next(c for e, c in pares if e.clave == "comparacion_con_el_baseline")

    def test_una_comparacion_incomparable_NO_sale_medida_y_dice_por_que(self):
        perfil = _perfil(comparacion_con_el_baseline=_a_mano("incomparable"),
                         baseline_comparado="baseline-seleccion")
        for idioma in ("es", "en"):
            campo = self._resuelto(perfil, idioma)
            with self.subTest(idioma=idioma):
                self.assertEqual(campo.estado, FALTA)
                self.assertIn(_MOTIVOS[idioma]["comparacion_incomparable"], campo.motivo)
                self.assertIn(_MOTIVO_DE_PRUEBA[idioma], campo.motivo)

    def test_con_motivo_de_ausencia_el_expediente_dice_ESE_motivo_y_no_el_generico(self):
        for motivo in MOTIVOS_SIN_COMPARACION_CON_EL_BASELINE:
            perfil = _perfil(sin_comparacion_con_el_baseline=motivo)
            for idioma in ("es", "en"):
                campo = self._resuelto(perfil, idioma)
                with self.subTest(motivo=motivo, idioma=idioma):
                    self.assertEqual(campo.estado, FALTA)
                    self.assertEqual(campo.motivo, _T[idioma][f"sin_comparacion_{motivo}"])

    def test_una_comparacion_de_verdad_sigue_saliendo_medida_por_el_resolver(self):
        """La otra mitad: que lo anterior no convierta en FALTA lo que sí se midió."""
        perfil = _perfil(comparacion_con_el_baseline=_comparacion(),
                         baseline_comparado="baseline-seleccion")
        self.assertEqual(self._resuelto(perfil, "es").estado, MEDIDO)

    def test_sin_comparacion_sale_FALTA_y_el_motivo_ya_no_dice_que_no_se_escribe(self):
        for perfil in (_perfil(), _perfil(sin_comparacion_con_el_baseline="el_baseline_no_puntuo")):
            motivo = POR_CLAVE["comparacion_con_el_baseline"].motivo_si_falta
            campo = self._campo(perfil)
            with self.subTest(sin=perfil.sin_comparacion_con_el_baseline):
                self.assertEqual(campo.estado, FALTA)
                self.assertIn("sin_comparacion_con_el_baseline", motivo["es"])
                self.assertIn("sin_comparacion_con_el_baseline", motivo["en"])
                self.assertNotIn("no se escribe", motivo["es"])


if __name__ == "__main__":
    unittest.main()
