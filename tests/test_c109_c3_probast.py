"""109-C3 — LOS HUECOS PROBAST+AI: qué sostiene un paquete, y qué NO se escribe.

PROBAST+AI es la herramienta con la que un revisor juzga el riesgo de sesgo y la
aplicabilidad de un modelo de predicción. Lo que se fija aquí:

1. **No se puntúa, no se cierra ningún dominio y no se califica el riesgo de
   sesgo** (invariante 6). `juicio_de_riesgo` va en el JSON y va a `None`
   siempre, para que quien lo busque lo encuentre vacío y lea por qué en vez de
   deducir que se olvidó.
2. **No se inventan los enunciados de las preguntas señal.** No se han podido
   verificar contra la fuente en este entorno; escribirlos de memoria sería una
   cita fabricada, que es el defecto que el propio 109 ya retiró de su fila de
   C2. Sin instrumento se va por DOMINIO y se dice por qué.
3. **Un paquete sin uso previsto marca «falta»** — criterio de terminado.
4. **Cada línea traza a un campo**, y la ruta va aunque el campo falte.
5. **El JSON y el Markdown salen de la misma lectura**: el hueco de este
   repositorio está en el cableado catorce veces de cada catorce.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.export.expediente_clinico import (  # noqa: E402
    DECLARADO,
    FALTA,
    MEDIDO,
    VeredictoDeRiesgo,
    sin_veredicto_de_riesgo,
)
from matrixai.export.probast import (  # noqa: E402
    CUENTAS_SEGUN_EL_CONTRATO,
    ExpedienteNoDisponible,
    huecos_probast,
    mapa_probast,
)
from test_c109_c3_expediente import (  # noqa: E402
    DECLARACION,
    paquete_clinico,
    perfil_real,
)


def _instrumento(n: int = 16, modo: str = "development") -> dict:
    """Un instrumento como el que dejaría el equipo, que sí lo tiene.

    Los enunciados son de mentira A PROPÓSITO y la prueba no los mira como si
    fueran los oficiales: lo que se comprueba es que, cuando llegan, este
    informe responde a lo que TRAIGA el fichero y no a una lista propia.
    """
    dominios = ("participantes", "predictores", "desenlace", "analisis")
    return {"version": "instrumento de prueba", "modo": modo,
            "preguntas": [{"id": f"{i + 1}.1", "dominio": dominios[i % 4],
                           "texto": f"enunciado del equipo {i + 1}"}
                          for i in range(n)]}


class NoJuzgaTest(unittest.TestCase):
    def test_el_juicio_de_riesgo_va_explicito_y_vacio(self):
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(Path(d), declaracion=DECLARACION))
        self.assertIn("juicio_de_riesgo", mapa)
        self.assertIsNone(mapa["juicio_de_riesgo"])

    def test_la_ficha_no_califica_el_riesgo_de_sesgo_en_ningun_idioma(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION,
                                  instrumento=_instrumento())
            for locale in ("es", "en"):
                with self.subTest(locale=locale):
                    ficha = huecos_probast(paq, locale=locale)
                    # La propia ficha ya pasa por la puerta; esto comprueba que
                    # la puerta sigue puesta y que no la esquiva nadie.
                    self.assertEqual(
                        sin_veredicto_de_riesgo(ficha, origen="prueba"), ficha)

    def test_ningun_estado_del_mapa_es_un_veredicto(self):
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(Path(d), declaracion=DECLARACION))
        estados = {e["estado"] for lista in mapa["por_dominio"].values()
                   for e in lista}
        self.assertTrue(estados)
        self.assertLessEqual(estados, {MEDIDO, DECLARADO, FALTA})

    def test_la_puerta_tiene_dientes(self):
        """Si mañana alguien mete el veredicto en una traducción, esto revienta.
        Se comprueba aquí para que un verde no pueda significar «no lo probé»."""
        with self.assertRaises(VeredictoDeRiesgo):
            sin_veredicto_de_riesgo("# Huecos\n\nDominio a bajo riesgo de sesgo.",
                                    origen="prueba")


class NoSeInventanLosEnunciadosTest(unittest.TestCase):
    def test_sin_instrumento_NO_hay_preguntas(self):
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(Path(d)))
        self.assertFalse(mapa["instrumento"]["presente"])
        self.assertEqual(mapa["preguntas"], [])

    def test_sin_instrumento_la_ficha_DICE_por_que_va_por_dominio(self):
        with TemporaryDirectory() as d:
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="es")
        self.assertIn("no viajan en este informe", ficha)
        self.assertIn("cita fabricada", ficha)
        self.assertIn("probast_instrument.json", ficha)

    def test_la_cuenta_del_contrato_se_cita_SIN_darla_por_comprobada(self):
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(Path(d)))
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="es")
        self.assertEqual(CUENTAS_SEGUN_EL_CONTRATO,
                         {"development": 16, "evaluation": 18})
        self.assertFalse(
            mapa["instrumento"]["cuentas_comprobadas_contra_el_instrumento_oficial"])
        self.assertIn("no está comprobado aquí", ficha)

    def test_con_instrumento_se_responde_a_LO_QUE_TRAE_el_fichero(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION,
                                  instrumento=_instrumento(16, "development"))
            mapa = mapa_probast(paq)
            ficha = huecos_probast(paq, locale="es")
        self.assertTrue(mapa["instrumento"]["presente"])
        self.assertEqual(len(mapa["preguntas"]), 16)
        self.assertIsNone(mapa["instrumento"]["desajuste_de_cuenta"])
        self.assertIn("1.1", ficha)
        for pregunta in mapa["preguntas"]:
            with self.subTest(pregunta=pregunta["pregunta_id"]):
                self.assertTrue(pregunta["apoyos"])
                self.assertIn(pregunta["estado"], (MEDIDO, DECLARADO, FALTA))

    def test_el_estado_de_una_pregunta_dice_QUE_CLASE_de_evidencia_hay(self):
        """`MEDIDO` solo si el core midió algo. Una pregunta cuyo dominio solo
        tiene lo que escribió una persona sale `DECLARADO`, y si no hay nada,
        `falta`. Marcar de oficio «medido» lo que declaró alguien es la
        confusión que este corte existe para impedir."""
        instrumento = {"modo": "development", "preguntas": [
            {"id": "ap.1", "dominio": "aplicabilidad", "texto": "uso previsto"}]}
        with TemporaryDirectory() as d:
            con = mapa_probast(paquete_clinico(Path(d) / "con",
                                               declaracion=DECLARACION,
                                               instrumento=instrumento))
            sin = mapa_probast(paquete_clinico(Path(d) / "sin",
                                               instrumento=instrumento))
        self.assertEqual(con["preguntas"][0]["estado"], DECLARADO)
        self.assertEqual(sin["preguntas"][0]["estado"], FALTA)

    def test_una_cuenta_que_no_cuadra_con_el_contrato_SE_DICE(self):
        """Manda el instrumento, y la diferencia se escribe en vez de callarla."""
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), instrumento=_instrumento(14, "development"))
            mapa = mapa_probast(paq)
            ficha = huecos_probast(paq, locale="es")
        self.assertEqual(mapa["instrumento"]["desajuste_de_cuenta"], "14 != 16")
        self.assertIn("14", ficha)
        self.assertIn("16", ficha)

    def test_la_lista_de_evaluacion_tiene_su_propia_cuenta(self):
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(
                Path(d), instrumento=_instrumento(18, "evaluation")))
        self.assertIsNone(mapa["instrumento"]["desajuste_de_cuenta"])

    def test_un_instrumento_roto_no_se_lee_como_ausente(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d))
            (paq / "probast_instrument.json").write_text("{roto", encoding="utf-8")
            mapa = mapa_probast(paq)
            ficha = huecos_probast(paq, locale="es")
        self.assertTrue(mapa["instrumento"]["motivo"].startswith("roto"))
        self.assertIn("no se puede leer", ficha)


class LoQueFaltaSeEnumeraTest(unittest.TestCase):
    def test_un_paquete_SIN_USO_PREVISTO_marca_falta(self):
        """Criterio de terminado del 109-C3, literal."""
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(Path(d)))
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="es")
        uso = [e for e in mapa["por_dominio"]["aplicabilidad"]
               if e["clave"] == "uso_previsto"]
        self.assertEqual(len(uso), 1)
        self.assertEqual(uso[0]["estado"], FALTA)
        self.assertIsNone(uso[0]["valor"])
        self.assertIn("`team_declaration.json#uso_previsto`", ficha)

    def test_la_comparacion_con_el_proceso_actual_ausente_se_dice(self):
        """Invariante 4: sin proceso actual declarado, se dice."""
        with TemporaryDirectory() as d:
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="es")
        self.assertIn("sin proceso actual declarado", ficha)
        self.assertIn("`team_declaration.json#proceso_actual`", ficha)

    def test_lo_que_el_paquete_no_trae_lleva_el_campo_que_lo_arreglaria(self):
        with TemporaryDirectory() as d:
            mapa = mapa_probast(paquete_clinico(Path(d), con_perfil=False))
        faltan = [e for lista in mapa["por_dominio"].values() for e in lista
                  if e["estado"] == FALTA]
        self.assertGreater(len(faltan), 5)
        for entrada in faltan:
            with self.subTest(clave=entrada["clave"]):
                self.assertIn("#", entrada["ruta"])
                self.assertTrue(entrada["motivo"])

    def test_el_resumen_cuenta_lo_que_hay(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION)
            mapa = mapa_probast(paq)
            ficha = huecos_probast(paq, locale="es")
        medidos = sum(1 for lista in mapa["por_dominio"].values() for e in lista
                      if e["estado"] == MEDIDO)
        self.assertIn(f"MEDIDO por el core: {medidos}", ficha)


class CadaLineaTrazaTest(unittest.TestCase):
    def test_toda_linea_del_inventario_lleva_su_ruta(self):
        with TemporaryDirectory() as d:
            ficha = huecos_probast(paquete_clinico(Path(d), declaracion=DECLARACION),
                                   locale="es")
        lineas = [l for l in ficha.splitlines()
                  if l.startswith("- **") and "PROBAST" not in l]
        self.assertGreaterEqual(len(lineas), 20)
        for linea in lineas:
            with self.subTest(linea=linea[:60]):
                self.assertRegex(linea, r"`[a-z_]+\.json#")

    def test_el_json_y_el_markdown_salen_de_la_misma_lectura(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION)
            mapa = mapa_probast(paq, locale="es")
            ficha = huecos_probast(paq, locale="es")
        for lista in mapa["por_dominio"].values():
            for entrada in lista:
                with self.subTest(clave=entrada["clave"]):
                    self.assertIn(f"`{entrada['ruta']}`", ficha)
                    self.assertIn(entrada["rotulo"], ficha)


class BilingueTest(unittest.TestCase):
    def test_en_espanol_no_se_cuela_el_ingles(self):
        with TemporaryDirectory() as d:
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="es")
        for ingles in ("the package", "MEASURED by", "DECLARED by", "missing",
                       "the team declares"):
            with self.subTest(ingles=ingles):
                self.assertNotIn(ingles, ficha)

    def test_y_en_ingles_no_se_cuela_el_castellano(self):
        """Palabras FUNCIONALES: no se pueden evitar escribiendo en castellano."""
        with TemporaryDirectory() as d:
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="en")
        for castellano in (" el ", " la ", " que ", " con ", " del ", " desde ",
                           "MEDIDO", "falta —"):
            with self.subTest(castellano=castellano):
                self.assertNotIn(castellano, ficha)


    def test_un_fichero_roto_tampoco_cuela_castellano_en_la_ficha_inglesa(self):
        """El motivo de «este fichero no se puede leer» lo redacta el lector, no
        el inventario, y por eso se olvidaba de traducirse: se cazó midiendo, no
        leyendo, y por eso hay una prueba con su nombre."""
        # SIN declaración del equipo: lo que ESCRIBE un equipo viaja tal cual y
        # en su idioma —no se traduce lo que declaró una persona—, así que un
        # barrido de castellano sobre un paquete con declaración española daría
        # rojo por el dato y no por el producto.
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d))
            (paq / "clinical_profile.json").write_text("{roto", encoding="utf-8")
            ficha = huecos_probast(paq, locale="en")
        self.assertIn("cannot be read", ficha)
        for castellano in (" el ", " la ", " que ", "no se puede leer",
                           "el paquete no trae"):
            with self.subTest(castellano=castellano):
                self.assertNotIn(castellano, ficha)


class SinPaqueteNoHayFichaTest(unittest.TestCase):
    def test_sin_manifiesto_se_levanta_en_vez_de_devolver_un_formulario(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ExpedienteNoDisponible):
                huecos_probast(Path(d))


class DiceQueNoHayPilotoTest(unittest.TestCase):
    def test_la_ficha_declara_que_se_construyo_sin_disparador(self):
        """C3 está marcado «solo si el piloto lo necesita» y no hay piloto. El
        109 se abrió a sabiendas; eso va escrito, como en el 107."""
        with TemporaryDirectory() as d:
            ficha = huecos_probast(paquete_clinico(Path(d)), locale="es")
        self.assertIn("no hay piloto", ficha)
        self.assertIn("2026-09-14", ficha)


if __name__ == "__main__":
    unittest.main()
