"""109-C3 — LA FICHA TRIPOD+AI PRERRELLENADA: lo nuevo, y el aviso que caducó.

`tripod.py` venía de 85-C6 y su lista de «lo que esta ficha NO puede rellenar»
era CERRADA: calibración, subgrupos, datos ausentes y uso previsto salían
siempre como imposibles. Entonces era verdad. Desde 105-C3/C4 y 109-C2 el core
los mide, así que dejar el aviso puesto lo convertiría en una frase FALSA dentro
de una ficha cuyo único valor es no decir ninguna — **un apaño que caduca sin
avisar es peor que no ponerlo**, y arreglar algo puede volver falso un aviso.

Lo que se prueba:

1. **Lo que el paquete SÍ trae deja de enumerarse como hueco**, y lo que no trae
   lo sigue siendo, con el campo que lo arreglaría.
2. **La sección clínica solo sale si el paquete la trae.** El invariante 7 del
   109 dice que las funciones clínicas se construyen cuando un piloto las pide;
   veinticuatro huecos clínicos en un paquete que no es clínico son ruido.
3. **Un paquete sin uso previsto marca «falta»** — criterio de terminado.
4. **Cada frase traza a un campo.**
5. **Ninguna frase califica el riesgo de sesgo** (invariante 6).
6. Y la ficha de 85-C6 **sigue entera** para un paquete sin perfil: esto añade,
   no sustituye.
"""
from __future__ import annotations

import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.export.expediente_clinico import (  # noqa: E402
    VeredictoDeRiesgo,
    sin_veredicto_de_riesgo,
)
from matrixai.export.tripod import ficha_tripod  # noqa: E402
from test_c109_c3_expediente import (  # noqa: E402
    DECLARACION,
    paquete_clinico,
    perfil_real,
)


class LaSeccionClinicaSoloSiElPaqueteLaTraeTest(unittest.TestCase):
    def test_un_paquete_sin_perfil_ni_declaracion_no_la_lleva(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), con_perfil=False),
                                 locale="es")
        self.assertNotIn("Perfil clínico de la tarea", ficha)

    def test_con_perfil_sale_entera(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d)), locale="es")
        self.assertIn("Perfil clínico de la tarea (109-C2)", ficha)
        for seccion in ("Participantes", "Predictores", "Desenlace", "Análisis",
                        "Aplicabilidad"):
            with self.subTest(seccion=seccion):
                self.assertIn(f"### {seccion}", ficha)

    def test_con_declaracion_del_equipo_pero_sin_perfil_tambien(self):
        """Un equipo que declara su uso previsto merece verlo escrito aunque el
        paquete todavía no traiga el perfil."""
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), con_perfil=False,
                                                 declaracion=DECLARACION),
                                 locale="es")
        self.assertIn("Perfil clínico de la tarea", ficha)
        self.assertIn("cribado de riesgo de reingreso", ficha)


class ElAvisoQueCaducoTest(unittest.TestCase):
    def test_lo_que_el_paquete_TRAE_deja_de_enumerarse_como_hueco(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), declaracion=DECLARACION),
                                 locale="es")
            cola = ficha[ficha.index("## Lo que esta ficha NO puede rellenar"):]
        for ya_no_falta in ("Datos ausentes", "Calibración", "Equidad por subgrupos",
                            "Uso previsto y población destinataria"):
            with self.subTest(tema=ya_no_falta):
                self.assertNotIn(ya_no_falta, cola)
    def test_lo_que_ningun_run_puede_rellenar_sigue_enumerado(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), declaracion=DECLARACION),
                                 locale="es")
            cola = ficha[ficha.index("## Lo que esta ficha NO puede rellenar"):]
        for siempre in ("Financiación", "Conflictos de interés",
                        "Aprobación ética"):
            with self.subTest(tema=siempre):
                self.assertIn(siempre, cola)

    def test_lo_que_el_paquete_NO_trae_sigue_siendo_un_hueco(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), perfil=perfil_real(
                con_calibracion=False, con_segmentos=False, con_faltantes=False))
            ficha = ficha_tripod(paq, locale="es")
            cola = ficha[ficha.index("## Lo que esta ficha NO puede rellenar"):]
        for falta in ("Datos ausentes", "Calibración", "Equidad por subgrupos"):
            with self.subTest(tema=falta):
                self.assertIn(falta, cola)

    def test_el_motivo_NO_dice_que_el_core_no_sepa_medirlo(self):
        """El core lo mide desde 105-C3/C4; lo que pasa es que ESTE paquete no
        lo trae. Decir «no se calcula» sería heredar una nota vieja, y una nota
        vieja miente igual que un dato falso."""
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), con_perfil=False),
                                 locale="es")
            cola = ficha[ficha.index("## Lo que esta ficha NO puede rellenar"):]
        self.assertIn("105-C3", cola)
        self.assertIn("105-C4", cola)
        self.assertNotIn("no se calcula", cola)

    def test_la_ficha_de_85_C6_sigue_entera_sin_perfil(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), con_perfil=False),
                                 locale="es")
        for tema in ("Datos ausentes", "Calibración", "Equidad por subgrupos",
                     "Financiación", "Conflictos de interés",
                     "Uso previsto y población destinataria"):
            with self.subTest(tema=tema):
                self.assertIn(tema, ficha)


class NoFabricarTest(unittest.TestCase):
    def test_un_paquete_sin_uso_previsto_marca_FALTA(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d)), locale="es")
        self.assertIn("**Uso previsto**: _falta_", ficha)
        self.assertIn("`team_declaration.json#uso_previsto`", ficha)

    def test_lo_declarado_se_marca_como_declarado_y_no_como_medido(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), declaracion=DECLARACION),
                                 locale="es")
        linea = [l for l in ficha.splitlines() if l.startswith("- **Uso previsto**")]
        self.assertEqual(len(linea), 1)
        self.assertIn("DECLARADO por una persona", linea[0])
        self.assertNotIn("MEDIDO", linea[0])

    def test_la_declaracion_del_equipo_no_puede_cambiar_los_umbrales(self):
        """Los umbrales y quién los declaró viven en el perfil y en ningún otro
        sitio: dos sitios declarando lo mismo acaban divergiendo."""
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(
                paquete_clinico(Path(d),
                                declaracion=dict(DECLARACION,
                                                 umbrales_clinicos=[0.91, 0.97])),
                locale="es")
        self.assertIn("**Umbrales medidos**: 0.2, 0.35, 0.5", ficha)
        self.assertNotIn("0.91", ficha)

    def test_el_alcance_no_afirma_lo_que_no_puede_afirmar(self):
        """La ficha del perfil dice «no hay separación temporal» en su redacción
        de `internal_only`, y eso es falso cuando el diseño SÍ es temporal
        (hallazgo abierto, 2026-09-15). Aquí se compone desde los datos."""
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(
                Path(d), perfil=perfil_real(evidencia="development_estimate",
                                            diseno="temporal")), locale="es")
        self.assertIn("internal_only · evidencia=development_estimate · "
                      "diseno=temporal", ficha)
        self.assertNotIn("separación temporal", ficha)


class CadaFraseTrazaAUnCampoTest(unittest.TestCase):
    def test_toda_linea_de_la_seccion_clinica_lleva_su_ruta(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), declaracion=DECLARACION),
                                 locale="es")
        cuerpo = ficha[ficha.index("## Perfil clínico"):
                       ficha.index("## Disponibilidad")]
        lineas = [l for l in cuerpo.splitlines() if l.startswith("- **")]
        self.assertGreaterEqual(len(lineas), 20)
        for linea in lineas:
            with self.subTest(linea=linea[:60]):
                self.assertRegex(linea, r"`[a-z_]+\.json#")


    def test_TODA_la_ficha_traza_a_un_campo_no_solo_la_seccion_nueva(self):
        """Criterio de terminado: **cada frase de la ficha traza a un campo**.
        Las secciones de 85-C6 imprimían el valor sin decir de dónde salía, así
        que quien la leía no podía comprobar una sola línea."""
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), declaracion=DECLARACION),
                                 locale="es")
        # Lo único sin campo es lo que NO sale de ningún run: es del autor.
        del_autor = ("Financiación", "Conflictos de interés",
                     "Aprobación ética y registro del estudio")
        sin_ruta = []
        for linea in ficha.splitlines():
            if not linea.startswith("- **"):
                continue
            if "#" in linea and "`" in linea:
                continue
            sin_ruta.append(linea.split("**")[1])
        self.assertEqual(sorted(sin_ruta), sorted(del_autor))

    def test_los_predictores_dicen_DE_DONDE_salen(self):
        """El comentario del módulo decía «por eso se dice de dónde salen» y
        nada lo sostenía: la lista se imprimía igual viniera de la captura del
        run o del `.mxai` que viaja al lado. Ahora la ruta los distingue."""
        mxai = ("PROJECT P\n\nVECTOR Input[2]\n  edad: Scalar[18, 100]\n"
                "  ingresos_previos: Scalar\nEND\n\n"
                "NETWORK N\n  INPUT Input\n  LAYER Dense units=2 activation=softmax\n"
                "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
                "GRAPH\n  Input -> N\nEND\n")
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d))
            self.assertIn("`reproduce.json#generation.field_types`",
                          ficha_tripod(paq, locale="es"))
            # Sin la captura del run, el dato sale del modelo empaquetado — y se
            # dice, porque citar el fichero de al lado no es inventarlo.
            cuerpo = json.loads((paq / "reproduce.json").read_text(encoding="utf-8"))
            cuerpo["generation"].pop("field_types")
            (paq / "reproduce.json").write_text(json.dumps(cuerpo), encoding="utf-8")
            (paq / "model.mxai").write_text(mxai, encoding="utf-8")
            ficha = ficha_tripod(paq, locale="es")
        self.assertIn("`model.mxai#vectors`", ficha)
        self.assertIn("edad (18–100)", ficha)

    def test_un_hueco_lleva_el_campo_que_lo_rellenaria(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d), con_perfil=False),
                                 locale="es")
            cola = ficha[ficha.index("## Lo que esta ficha NO puede rellenar"):]
        self.assertIn("`clinical_profile.json#calibracion.ece`", cola)
        self.assertIn("`clinical_profile.json#segmentos`", cola)
        self.assertIn("`team_declaration.json#uso_previsto`", cola)


class NuncaUnJuicioDeRiesgoTest(unittest.TestCase):
    def test_la_ficha_no_califica_el_riesgo_de_sesgo(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION)
            for locale in ("es", "en"):
                with self.subTest(locale=locale):
                    ficha = ficha_tripod(paq, locale=locale)
                    self.assertEqual(
                        sin_veredicto_de_riesgo(ficha, origen="prueba"), ficha)

    def test_la_puerta_esta_puesta_en_la_ficha_tambien(self):
        with self.assertRaises(VeredictoDeRiesgo):
            sin_veredicto_de_riesgo("# Ficha\n\nEste modelo es de bajo riesgo.",
                                    origen="prueba")


class BilingueTest(unittest.TestCase):
    def test_en_espanol_no_se_cuela_el_ingles(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d)), locale="es")
        for ingles in ("the package", "MEASURED by", "DECLARED by",
                       "this package", "a person writes"):
            with self.subTest(ingles=ingles):
                self.assertNotIn(ingles, ficha)

    def test_y_en_ingles_no_se_cuela_el_castellano(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(paquete_clinico(Path(d)), locale="en")
        for castellano in (" el ", " la ", " que ", " con ", " del ", "disponible",
                           "MEDIDO", "DECLARADO"):
            with self.subTest(castellano=castellano):
                self.assertNotIn(castellano, ficha)


class PorElCliTest(unittest.TestCase):
    """`matrixai report --tripod` sobre el paquete de C2 — criterio de terminado.

    La mitad `--probast` NO está cableada: `matrixai/cli.py` queda fuera del
    territorio de este corte y el cambio va DECLARADO, no hecho. Esta prueba
    cubre la mitad que sí existe, de punta a punta.
    """

    def test_el_comando_escribe_la_seccion_nueva(self):
        from matrixai.cli import main

        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION)
            salida = StringIO()
            with patch("sys.stdout", salida), patch(
                    "sys.argv", ["matrixai", "report", str(paq), "--tripod",
                                 "--locale", "es"]):
                codigo = main()
        texto = salida.getvalue()
        self.assertEqual(codigo, 0)
        self.assertIn("Perfil clínico de la tarea (109-C2)", texto)
        self.assertIn("`team_declaration.json#uso_previsto`", texto)


if __name__ == "__main__":
    unittest.main()
