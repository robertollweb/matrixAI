"""109-C3 — EL EXPEDIENTE CLÍNICO: de dónde sale cada frase, y qué no se escribe.

Los dos invariantes que manda este corte, y son los que se prueban aquí:

**6 · el core rellena lo medido, marca lo declarado y enumera lo que falta;
NUNCA «bajo riesgo de sesgo».** Un juicio de riesgo de sesgo lo emite un revisor
humano leyendo el estudio. Aquí eso no es una convención de estilo: `ESTADOS` es
una tupla cerrada de tres, y `sin_veredicto_de_riesgo()` revienta si el
vocabulario de veredicto del instrumento llega al texto renderizado.

**5 · NO FABRICAR.** Población, uso previsto, criterios de inclusión, definición
y momento del desenlace y los umbrales clínicos los declara el equipo. Si no
están, sale «falta» **con el campo que lo arreglaría**, nunca relleno de oficio.
Un `Campo` en estado `FALTA` no puede llevar valor, y uno con estado no puede no
llevarlo: las dos direcciones, porque las dos han sido un fallo.

Y una tercera cosa que esta prueba vigila: **lo medido se COMPONE, no se copia**.
El alcance de la validación se vuelve a derivar de `evidencia` + `diseno` con la
misma función del core, y si el `alcance` guardado en el paquete no cuadra, se
dice en vez de creerle al campo.
"""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.estudio.calibracion import curva_de_fiabilidad  # noqa: E402
from matrixai.estudio.metricas import Muestra  # noqa: E402
from matrixai.estudio.perfil_clinico import (  # noqa: E402
    PerfilClinico,
    curva_de_decision,
    tabla_de_umbrales,
)
from matrixai.estudio.segmentos import analizar_segmento  # noqa: E402
from matrixai.export.expediente_clinico import (  # noqa: E402
    DECLARADO,
    ESTADOS,
    FALTA,
    INVENTARIO,
    MEDIDO,
    POR_CLAVE,
    VEREDICTOS_DE_RIESGO,
    Campo,
    Evidencia,
    ExpedienteClinico,
    ExpedienteNoDisponible,
    VeredictoDeRiesgo,
    linea_de_campo,
    sin_veredicto_de_riesgo,
)

_MANIFIESTO = {
    "schema_version": "1.0",
    "reproducible": True,
    "generation": {
        "mode": "coherent", "seeds": {"dataset": 1, "split": 2, "init": 3},
        "epochs_declared": 60, "epochs_effective": 60, "epochs_ran": 60,
        "backend": "stdlib", "device": "cpu",
        "field_types": {"edad": "scalar", "ingresos_previos": "scalar"},
        "excluded_identifiers": ["id_paciente"],
    },
    "artifacts": {"model": {"sha256": "a" * 64}, "training": {"sha256": "b" * 64},
                  "dataset": {"sha256": "c" * 64, "rows": 400}},
    "environment": {"matrixai_version": "1.7.1", "python": {"version": "3.12.3"}},
    "metrics": [{"name": "sensitivity", "value": 0.81, "dataset_sha256": "c" * 64}],
}

#: Lo que declararía un equipo. **No lleva umbrales a propósito**: los umbrales y
#: quién los declaró viven en el perfil (109-C2) y en ningún otro sitio. Dos
#: sitios declarando lo mismo acaban divergiendo, y hay una prueba con su nombre.
DECLARACION = {
    "uso_previsto": "cribado de riesgo de reingreso a 30 días al alta",
    "poblacion": "adultos ingresados en medicina interna del Hospital X",
    "criterios_de_inclusion": ["edad >= 18", "alta a domicilio"],
    "definicion_del_desenlace": "reingreso no programado por cualquier causa",
    "momento_del_desenlace": "30 días desde el alta",
    "proceso_actual": "escala LACE aplicada a mano por enfermería",
    "declarado_por": "Comité de reingresos, Hospital X",
}


def _muestra(n: int = 400, semilla: int = 20260915) -> Muestra:
    rnd = random.Random(semilla)
    y, probs = [], []
    for _ in range(n):
        real = rnd.random() < 0.28
        p = min(0.999, max(0.001, rnd.gauss(0.68 if real else 0.32, 0.17)))
        y.append("reingreso" if real else "no_reingreso")
        probs.append((1.0 - p, p))
    return Muestra(task="binary_classification", y_true=tuple(y),
                   classes=("no_reingreso", "reingreso"),
                   positive_label="reingreso", probabilities=tuple(probs))


def perfil_real(*, evidencia: str = "independent_test", diseno: str = "iid",
                con_calibracion: bool = True, con_segmentos: bool = True,
                con_faltantes: bool = True, prevalencia: float | None = 0.08,
                declarados_por: str | None = "Comité de reingresos, Hospital X",
                ) -> PerfilClinico:
    """Un perfil clínico DE VERDAD, del mismo código que corre en producción.

    Escribir el JSON a mano probaría el artefacto y no el código que lo produce:
    un `clinical_profile.json` inventado cuadra consigo mismo aunque
    `perfil_clinico.py` cambie de forma debajo.
    """
    muestra = _muestra()
    umbrales = (0.2, 0.35, 0.5)
    tabla = tabla_de_umbrales(muestra, umbrales=umbrales, diseno=diseno,
                             estimando="fixed_model_on_population", semilla=7,
                             prevalencia=prevalencia,
                             declarados_por=declarados_por, remuestras=40)
    segmentos = ()
    if con_segmentos:
        segmentos = (analizar_segmento("sensitivity", muestra, tuple(range(200)),
                                       segmento_id="mayores_de_75",
                                       predefinido=True),)
    return PerfilClinico(
        perfil_id="perfil-c109-c3", evidencia=evidencia, diseno=diseno,
        datos_sinteticos=True, tabla=tabla,
        curva=curva_de_decision(muestra, umbrales_de_probabilidad=umbrales),
        calibracion=curva_de_fiabilidad(muestra, n_bins=5) if con_calibracion else None,
        segmentos=segmentos,
        segmentos_predefinidos_por="Comité de reingresos" if con_segmentos else None,
        politica_de_faltantes=({"estrategia": "descartar_fila",
                                "declarada_por": "Comité"} if con_faltantes else None))


def paquete_clinico(destino: Path, *, perfil: PerfilClinico | None = None,
                    declaracion: dict | None = None, manifiesto: dict | None = None,
                    instrumento: dict | None = None,
                    con_perfil: bool = True) -> Path:
    """Un paquete exportado como el que produciría un run clínico."""
    destino.mkdir(parents=True, exist_ok=True)
    m = json.loads(json.dumps(_MANIFIESTO))
    if manifiesto:
        m.update(manifiesto)
    (destino / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
    (destino / "data_recipe.txt").write_text("reingreso: edad > 75\n", encoding="utf-8")
    if con_perfil:
        p = perfil if perfil is not None else perfil_real()
        (destino / "clinical_profile.json").write_text(
            json.dumps(p.a_json()), encoding="utf-8")
    if declaracion is not None:
        (destino / "team_declaration.json").write_text(
            json.dumps(declaracion, ensure_ascii=False), encoding="utf-8")
    if instrumento is not None:
        (destino / "probast_instrument.json").write_text(
            json.dumps(instrumento, ensure_ascii=False), encoding="utf-8")
    return destino


class ElEstadoLoDiceElFicheroTest(unittest.TestCase):
    """Un argumento `estado=` invitaría a marcar MEDIDO lo que escribió una
    persona, que es justo la confusión que este corte existe para impedir."""

    def test_lo_que_escribe_el_core_sale_MEDIDO(self):
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(Path(d)))
            campo = e.campo("reproduce.json#artifacts.dataset.rows")
        self.assertEqual(campo.estado, MEDIDO)
        self.assertEqual(campo.valor, 400)

    def test_lo_que_escribe_una_persona_sale_DECLARADO(self):
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(
                paquete_clinico(Path(d), declaracion=DECLARACION))
            campo = e.campo("team_declaration.json#uso_previsto")
        self.assertEqual(campo.estado, DECLARADO)

    def test_lo_que_el_PERFIL_registra_pero_declaro_una_persona_no_sale_medido(self):
        """La excepción es cerrada y está escrita: los umbrales, la prevalencia
        de la población destino, quién predefinió los subgrupos, la política de
        ausentes y si los datos son sintéticos los pone una persona, aunque el
        perfil sea quien los guarde."""
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(Path(d)))
            for ruta in ("clinical_profile.json#tabla_de_umbrales.declarados_por",
                         "clinical_profile.json#tabla_de_umbrales.prevalencia_declarada",
                         "clinical_profile.json#segmentos_predefinidos_por",
                         "clinical_profile.json#politica_de_faltantes",
                         "clinical_profile.json#datos_sinteticos"):
                with self.subTest(ruta=ruta):
                    self.assertEqual(e.campo(ruta).estado, DECLARADO)
            # Y lo que el core sí midió sigue siendo MEDIDO.
            self.assertEqual(
                e.campo("clinical_profile.json#tabla_de_umbrales.filas").estado,
                MEDIDO)
            self.assertEqual(e.campo("clinical_profile.json#calibracion.ece").estado,
                             MEDIDO)


class NoFabricarTest(unittest.TestCase):
    def test_un_hueco_NO_PUEDE_llevar_valor(self):
        with self.assertRaises(ValueError):
            Campo("team_declaration.json#uso_previsto", FALTA,
                  valor="no declarado", motivo="x")

    def test_un_dato_no_puede_no_llevar_valor(self):
        """Un `MEDIDO` sin valor es una casilla que parece rellena porque tiene
        rótulo."""
        with self.assertRaises(ValueError):
            Campo("reproduce.json#artifacts.dataset.rows", MEDIDO)

    def test_un_hueco_dice_POR_QUE_falta(self):
        with self.assertRaises(ValueError):
            Campo("team_declaration.json#uso_previsto", FALTA)

    def test_lo_que_el_equipo_no_declaro_sale_FALTA_y_no_relleno(self):
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(Path(d)))
            for clave in ("uso_previsto", "poblacion", "criterios_de_inclusion",
                          "definicion_del_desenlace", "momento_del_desenlace",
                          "proceso_actual"):
                with self.subTest(clave=clave):
                    campo = e.campo(POR_CLAVE[clave].ruta)
                    self.assertEqual(campo.estado, FALTA)
                    self.assertIsNone(campo.valor)
                    self.assertTrue(campo.motivo)

    def test_un_campo_declarado_VACIO_tampoco_cuenta_como_declarado(self):
        """Una cadena vacía o una lista vacía es un formulario a medio rellenar,
        no una declaración."""
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(
                Path(d), declaracion={"uso_previsto": "", "poblacion": []}))
            self.assertEqual(e.campo("team_declaration.json#uso_previsto").estado,
                             FALTA)
            self.assertEqual(e.campo("team_declaration.json#poblacion").estado, FALTA)

    def test_los_umbrales_los_manda_el_PERFIL_y_nadie_mas(self):
        """Si la declaración del equipo pudiera traer umbrales, habría dos
        sitios declarando lo mismo. Aquí solo hay uno, y ponerlos en el otro no
        cambia nada."""
        with TemporaryDirectory() as d:
            mentira = dict(DECLARACION, umbrales_clinicos=[0.9, 0.95])
            e = ExpedienteClinico.desde_paquete(
                paquete_clinico(Path(d), declaracion=mentira))
            pares, _ = e.resolver("es")
            texto = "\n".join(linea_de_campo(ev, c, "es") for ev, c in pares)
        self.assertIn("0.2, 0.35, 0.5", texto)
        self.assertNotIn("0.9", texto)
        self.assertNotIn("umbrales_clinicos", texto)


class CadaFraseTrazaAUnCampoTest(unittest.TestCase):
    def test_toda_linea_lleva_su_ruta_aunque_el_campo_falte(self):
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(Path(d)))
            pares, _ = e.resolver("es")
        self.assertEqual(len(pares), len(INVENTARIO))
        for evidencia, campo in pares:
            with self.subTest(clave=evidencia.clave):
                linea = linea_de_campo(evidencia, campo, "es")
                self.assertIn(f"`{campo.ruta}`", linea)
                self.assertIn("#", linea)

    def test_una_ruta_sin_campo_se_rechaza(self):
        with self.assertRaises(ValueError):
            Campo("reproduce.json", MEDIDO, valor=1)

    def test_una_ruta_a_un_fichero_que_no_es_del_paquete_se_rechaza(self):
        with self.assertRaises(ValueError):
            Campo("inventado.json#cosa", MEDIDO, valor=1)


class NuncaUnJuicioDeRiesgoTest(unittest.TestCase):
    """El corazón del invariante 6."""

    def test_solo_hay_TRES_estados(self):
        self.assertEqual(ESTADOS, ("medido", "declarado", "falta"))

    def test_un_cuarto_estado_se_rechaza(self):
        for inventado in ("bajo_riesgo", "low_risk", "alto_riesgo", "aceptable"):
            with self.subTest(estado=inventado), self.assertRaises(ValueError):
                Campo("reproduce.json#metrics", inventado, valor=1)

    def test_el_vocabulario_de_veredicto_revienta_en_los_dos_idiomas(self):
        for frase in ("El modelo está a bajo riesgo de sesgo.",
                      "This study is at low risk of bias.",
                      "Riesgo alto en el dominio de análisis.",
                      "unclear risk for the outcome domain",
                      "baja preocupación sobre la aplicabilidad"):
            with self.subTest(frase=frase), self.assertRaises(VeredictoDeRiesgo):
                sin_veredicto_de_riesgo(frase, origen="prueba")

    def test_el_MARKDOWN_no_sirve_para_colarlo(self):
        """`**bajo** riesgo` y `BAJO RIESGO` son la misma frase."""
        for frase in ("**bajo** riesgo de sesgo", "BAJO  RIESGO", "bajo-riesgo",
                      "Bájo riesgo"):
            with self.subTest(frase=frase), self.assertRaises(VeredictoDeRiesgo):
                sin_veredicto_de_riesgo(frase, origen="prueba")

    def test_NOMBRAR_el_riesgo_de_sesgo_si_se_puede(self):
        """Hay que poder decir qué es PROBAST sin calificar nada."""
        texto = ("PROBAST+AI es la herramienta con la que un revisor juzga el "
                 "riesgo de sesgo; este informe no lo juzga.")
        self.assertEqual(sin_veredicto_de_riesgo(texto, origen="prueba"), texto)

    def test_la_lista_de_veredictos_no_esta_vacia(self):
        """Un banco sin dientes: si alguien vaciase la lista, todo lo de arriba
        seguiría en verde sin proteger nada."""
        self.assertGreaterEqual(len(VEREDICTOS_DE_RIESGO), 10)


class ElAlcanceSeComponeNoSeCopiaTest(unittest.TestCase):
    def test_se_re_deriva_de_evidencia_y_diseno(self):
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(
                Path(d), perfil=perfil_real(evidencia="external_validation",
                                            diseno="temporal")))
            campo, aviso = e.alcance_de_validacion()
        self.assertEqual(campo.valor["alcance"], "external")
        self.assertEqual(campo.valor["evidencia"], "external_validation")
        self.assertEqual(campo.valor["diseno"], "temporal")
        self.assertEqual(aviso, "")

    def test_un_alcance_guardado_que_no_cuadra_SE_DICE(self):
        """El campo `alcance` del paquete se puede editar a mano después de
        exportar. No se le cree: se vuelve a derivar y se compara."""
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d))
            ruta = paq / "clinical_profile.json"
            cuerpo = json.loads(ruta.read_text(encoding="utf-8"))
            cuerpo["alcance"] = "external"          # mentira puesta a mano
            ruta.write_text(json.dumps(cuerpo), encoding="utf-8")
            campo, aviso = ExpedienteClinico.desde_paquete(paq).alcance_de_validacion()
        self.assertEqual(campo.valor["alcance"], "internal_only")
        self.assertIn("external", aviso)
        self.assertIn("internal_only", aviso)

    def test_NO_se_afirma_que_no_haya_separacion_temporal(self):
        """La ficha del perfil (109-C2) dice en su redacción de `internal_only`
        que «no hay separación temporal», y eso es FALSO cuando el diseño sí es
        temporal y la etiqueta de evidencia no llega — hallazgo abierto de la
        auditoría del 2026-09-15. Aquí no se copia esa frase: se publican el
        token y los dos campos de los que sale."""
        with TemporaryDirectory() as d:
            e = ExpedienteClinico.desde_paquete(paquete_clinico(
                Path(d), perfil=perfil_real(evidencia="development_estimate",
                                            diseno="temporal")))
            campo, _ = e.alcance_de_validacion()
            linea = linea_de_campo(POR_CLAVE["alcance_de_validacion"], campo, "es")
        self.assertEqual(campo.valor["alcance"], "internal_only")
        self.assertIn("diseno=temporal", linea)
        self.assertNotIn("separación temporal", linea)


class UnFicheroRotoNoEsUnFicheroAusenteTest(unittest.TestCase):
    def test_una_declaracion_rota_no_se_lee_como_no_declarada(self):
        with TemporaryDirectory() as d:
            paq = paquete_clinico(Path(d), declaracion=DECLARACION)
            (paq / "team_declaration.json").write_text("{roto", encoding="utf-8")
            e = ExpedienteClinico.desde_paquete(paq)
            campo = e.campo("team_declaration.json#uso_previsto", locale="es")
            en_ingles = e.campo("team_declaration.json#uso_previsto", locale="en")
        self.assertEqual(campo.estado, FALTA)
        self.assertIn("no se puede leer", campo.motivo)
        # Y el motivo habla el idioma de quien lee: media aplicación traducida
        # se ve enseguida, y este motivo lo redacta el lector, no el inventario.
        self.assertIn("cannot be read", en_ingles.motivo)
        self.assertNotIn("no se puede leer", en_ingles.motivo)

    def test_sin_manifiesto_no_hay_expediente(self):
        with TemporaryDirectory() as d, self.assertRaises(ExpedienteNoDisponible):
            ExpedienteClinico.desde_paquete(Path(d))


class BilingueTest(unittest.TestCase):
    def test_toda_evidencia_trae_las_DOS_redacciones(self):
        for evidencia in INVENTARIO:
            with self.subTest(clave=evidencia.clave):
                self.assertEqual(set(evidencia.rotulo), {"es", "en"})
                self.assertEqual(set(evidencia.motivo_si_falta), {"es", "en"})

    def test_media_traduccion_se_rechaza_al_construir(self):
        with self.assertRaises(ValueError):
            Evidencia(clave="x", dominio="analisis", ruta="reproduce.json#x",
                      rotulo={"es": "X"}, motivo_si_falta={"es": "y", "en": "y"})


if __name__ == "__main__":
    unittest.main()
