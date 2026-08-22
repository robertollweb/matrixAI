"""CONTRATO 81-C6 — las catorce pruebas obligatorias del §16.5.

Y antes que ninguna, lo que el caso NO es (§16.2): **demostrativo, no
clínicamente validado, no apto para decisiones asistenciales, sobre datos
sintéticos**. La primera prueba de este fichero comprueba que esa frase
sale **donde se lee el resultado** — porque un caso de reingreso
hospitalario sin ella se lee como una herramienta clínica.

Lo que estas pruebas defienden, y cada una es una forma concreta de
mentir que el caso tiene prohibida:

* un valor ausente **no es un cero**, y una categoría desconocida **no se
  parece a ninguna**: se abstiene, no se inventa;
* truncar un texto **se dice**, y una nota vacía **no es una nota normal**;
* un byte que no es UTF-8 **no se reemplaza**;
* un dato posterior a la decisión **no entra**;
* una fusión que no encaja **no se recorta**;
* por debajo del umbral **no se predice**;
* el dato clínico **no aparece en la traza**;
* un submodelo alterado **invalida la verificación**.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from matrixai.reference.readmission import (
    ADVERTENCIA,
    CAMPOS_SENSIBLES,
    POLITICAS,
    DimensionesIncompatibles,
    TextoNoUTF8,
    ValorNoNumerico,
    analisis_de_confianza,
    campos_con_fuga_temporal,
    cobertura_por_clase,
    codificar_tabular,
    codificar_texto,
    fusionar,
    preparar_texto,
)
from matrixai.reference.readmission_datos import columnas, generar_dataset, vector_de
from matrixai.reference.readmission_pipeline import (
    TALLA_TABULAR,
    VOCABULARIO,
    clasificador_del_registry,
    evaluar_caso,
    identidad_de_nodo,
)
from matrixai.registry.model_registry import ModelRegistry

EJEMPLO = Path(__file__).resolve().parent.parent / "examples" / "readmission"
MOMENTO = "2026-08-20T10:00:00Z"

CASO = {
    "edad": 88, "ingresos_previos_12m": 6, "dias_de_estancia": 30,
    "servicio": "cardiologia", "alta_voluntaria": 0,
}
NOTA = "disnea de reposo y edema en miembros inferiores"

#: Un secreto reconocible, para poder buscarlo en la traza.
_NOTA_CON_DATO = "NHC-4491203 paciente con disnea de reposo y edema"


def _clasificador_de_mentira(probabilidad_reingreso=0.9):
    """Un clasificador declarado, para las pruebas que no van del modelo."""
    def clasificar(vector):
        return {"no_reingreso": 1 - probabilidad_reingreso,
                "reingreso": probabilidad_reingreso}
    return clasificar


def _evaluar(registro=None, nota=NOTA, *, probabilidad=0.9, momento=MOMENTO, **kw):
    return evaluar_caso(
        dict(CASO if registro is None else registro), nota,
        momento_de_decision=momento,
        clasificador=_clasificador_de_mentira(probabilidad),
        clasificador_digest="sha256:" + "c" * 64, **kw)


class ElCasoDiceLoQueNO_EsTest(unittest.TestCase):
    """§16.2 y §16.6: «el caso no se presenta como validación clínica»."""

    def test_la_advertencia_sale_DONDE_se_lee_el_resultado(self):
        r = _evaluar()
        self.assertIn("NO validado clínicamente", r["disclaimer"])
        self.assertIn("no sustituye al profesional", r["disclaimer"])

    def test_tambien_cuando_el_pipeline_NO_termina(self):
        """Si solo apareciera en el camino bueno, el único sitio donde
        alguien lee un resultado sin ella sería justo el raro."""
        r = _evaluar(nota=b"\xff\xfe sin utf-8")
        self.assertEqual(r["outcome"], "not_run")
        self.assertIn("NO validado clínicamente", r["disclaimer"])

    def test_y_en_ingles_dice_lo_mismo(self):
        r = _evaluar(locale="en")
        self.assertIn("NOT clinically validated", r["disclaimer"])
        self.assertNotIn("clínicamente", r["disclaimer"])

    def test_la_advertencia_viaja_DENTRO_del_recibo(self):
        """Un recibo que circula sin ella se lee sin ella."""
        r = _evaluar()
        self.assertIn("NO validado", r["receipt"]["subject"]["disclaimer"])


class NotaClinicaVACIATest(unittest.TestCase):
    def test_una_nota_vacia_NO_es_una_nota_normal(self):
        """Un vector de ceros se lee como «sin hallazgos»; la ausencia de
        señal es otra cosa y se declara."""
        codificado = codificar_texto(preparar_texto(""), VOCABULARIO)
        self.assertFalse(codificado["signal_present"])
        self.assertEqual(set(codificado["features"].values()), {0.0})

    def test_y_la_ausencia_de_señal_llega_HASTA_la_salida(self):
        r = _evaluar(nota="   ")
        self.assertIs(r["text_signal_present"], False)
        self.assertTrue(r["text"]["empty"])

    def test_con_nota_SI_la_señal_se_declara_presente(self):
        self.assertIs(_evaluar()["text_signal_present"], True)


class NotaExcesivamenteLARGATest(unittest.TestCase):
    def test_truncar_se_DICE(self):
        """El final de una nota es donde suele estar lo agudo."""
        preparado = preparar_texto("a" * (int(POLITICAS["text_max_bytes"]) + 50))
        self.assertTrue(preparado["truncated"])
        self.assertEqual(preparado["truncated_at"], POLITICAS["text_max_bytes"])

    def test_y_llega_a_la_salida_del_caso(self):
        r = _evaluar(nota="disnea " + "x" * 2000)
        self.assertTrue(r["text"]["truncated"])
        self.assertEqual(r["text"]["truncated_at"], POLITICAS["text_max_bytes"])

    def test_una_nota_normal_NO_se_declara_truncada(self):
        self.assertFalse(_evaluar()["text"]["truncated"])


class VariablesTabularesINCOMPLETASTest(unittest.TestCase):
    def test_una_obligatoria_que_falta_NO_se_imputa(self):
        r = codificar_tabular({k: v for k, v in CASO.items() if k != "dias_de_estancia"})
        self.assertIn("dias_de_estancia", r["missing_required"])
        self.assertFalse(r["complete"])
        self.assertIsNone(r["values"]["dias_de_estancia"])

    def test_y_el_caso_SE_ABSTIENE_con_su_regla(self):
        """Abstenerse no es caerse: hay algo que decir, y es «no me
        pronuncio», con la regla al lado."""
        r = _evaluar({k: v for k, v in CASO.items() if k != "edad"})
        self.assertEqual(r["outcome"], "abstained")
        self.assertIsNone(r["class"])
        self.assertEqual(r["policy"]["rule_id"], "R1-datos-incompletos")

    def test_una_OPCIONAL_que_falta_NO_abstiene(self):
        """Si abstuviera, «opcional» no significaría nada."""
        r = _evaluar({k: v for k, v in CASO.items() if k != "alta_voluntaria"})
        self.assertEqual(r["outcome"], "prediction")
        self.assertEqual(r["tabular"]["missing_optional"], ["alta_voluntaria"])

    def test_la_ausencia_de_una_opcional_va_DENTRO_del_vector(self):
        """Un cero solo se leería como una medida; al lado va la bandera
        que dice que no lo es."""
        sin = codificar_tabular({k: v for k, v in CASO.items() if k != "alta_voluntaria"})
        con = codificar_tabular(CASO)
        self.assertEqual(sin["values"]["alta_voluntaria__ausente"], 1.0)
        self.assertEqual(con["values"]["alta_voluntaria__ausente"], 0.0)
        # Y la talla NO cambia: un vector que encoge no encaja con el modelo.
        self.assertEqual(len(sin["values"]), len(con["values"]))

    def test_un_numero_que_no_es_numero_se_RECHAZA(self):
        """No es incertidumbre clínica: es un dato mal recogido, y pasarlo
        por ausente lo escondería tras una abstención razonable."""
        with self.assertRaises(ValorNoNumerico):
            codificar_tabular({**CASO, "edad": "ochenta y ocho"})


class CategoriaDESCONOCIDATest(unittest.TestCase):
    def test_no_se_mapea_a_la_mas_parecida(self):
        r = codificar_tabular({**CASO, "servicio": "urgencias"})
        self.assertFalse(r["known_categories"])
        self.assertIn("servicio=urgencias", r["unknown_categories"])
        self.assertTrue(all(v is None for k, v in r["values"].items()
                            if k.startswith("servicio_")))

    def test_y_el_caso_se_abstiene_con_SU_regla(self):
        r = _evaluar({**CASO, "servicio": "urgencias"})
        self.assertEqual(r["outcome"], "abstained")
        self.assertEqual(r["policy"]["rule_id"], "R2-categoria-desconocida")


class DatosPOSTERIORES_ALaDecisionTest(unittest.TestCase):
    def test_se_detecta_la_fuga_temporal(self):
        campos = campos_con_fuga_temporal(
            {**CASO, "analitica_at": "2026-08-21T09:00:00Z"}, MOMENTO)
        self.assertEqual(campos, ["analitica_at"])

    def test_un_dato_ANTERIOR_no_es_fuga(self):
        self.assertEqual(
            campos_con_fuga_temporal({**CASO, "analitica_at": "2026-08-19T09:00:00Z"}, MOMENTO),
            [])

    def test_SIN_momento_declarado_todo_es_sospechoso(self):
        """Devolver «no hay fuga» porque nadie dijo contra qué comparar
        sería fallo abierto."""
        self.assertTrue(campos_con_fuga_temporal(CASO, ""))

    def test_y_el_caso_se_abstiene_con_su_regla(self):
        r = _evaluar({**CASO, "analitica_at": "2026-08-21T09:00:00Z"})
        self.assertEqual(r["outcome"], "abstained")
        self.assertEqual(r["policy"]["rule_id"], "R3-fuga-temporal")
        self.assertEqual(r["temporal_leakage"], ["analitica_at"])


class TextoConCODIFICACION_IncorrectaTest(unittest.TestCase):
    def test_un_byte_que_no_es_utf8_NO_se_reemplaza(self):
        """Sustituirlo por `?` cambiaría el texto sobre el que se decide."""
        with self.assertRaises(TextoNoUTF8) as caja:
            preparar_texto(b"\xff\xfe nota")
        self.assertIn("no se reemplazan", str(caja.exception))

    def test_y_el_caso_NO_produce_decision(self):
        r = _evaluar(nota=b"\xff\xfe nota")
        self.assertEqual(r["outcome"], "not_run")
        self.assertNotIn("class", r)

    def test_utf8_valido_con_acentos_pasa(self):
        self.assertFalse(preparar_texto("disnea, ¿edemas? sí").
                         get("empty"))


class FusionConDIMENSIONES_IncompatiblesTest(unittest.TestCase):
    def test_no_se_recorta_para_que_cuadre(self):
        tabular = codificar_tabular(CASO)
        texto = codificar_texto(preparar_texto(NOTA), VOCABULARIO)
        with self.assertRaises(DimensionesIncompatibles) as caja:
            fusionar(tabular, texto, tallas_esperadas=(TALLA_TABULAR, len(VOCABULARIO) + 3))
        # Las DOS tallas en el mensaje: «no encaja» no deja ver qué lado.
        self.assertIn(str(len(VOCABULARIO)), str(caja.exception))

    def test_las_tallas_declaradas_encajan(self):
        f = fusionar(codificar_tabular(CASO),
                     codificar_texto(preparar_texto(NOTA), VOCABULARIO),
                     tallas_esperadas=(TALLA_TABULAR, len(VOCABULARIO)))
        self.assertEqual(len(f["vector"]), TALLA_TABULAR + len(VOCABULARIO))


class BajaCONFIANZA_YAbstencionTest(unittest.TestCase):
    def test_por_debajo_del_umbral_NO_se_predice(self):
        r = _evaluar(probabilidad=0.55)
        self.assertEqual(r["outcome"], "abstained")
        self.assertEqual(r["policy"]["rule_id"], "R4-baja-confianza")

    def test_y_al_abstenerse_NO_viaja_una_clase(self):
        """Que estuviera ahí, aunque fuera «informativa», es lo que
        alguien acabaría leyendo."""
        self.assertIsNone(_evaluar(probabilidad=0.55)["class"])

    def test_por_encima_del_umbral_si_predice(self):
        r = _evaluar(probabilidad=0.9)
        self.assertEqual(r["outcome"], "prediction")
        self.assertEqual(r["class"], "reingreso")

    def test_el_umbral_esta_DECLARADO_no_escondido_en_el_codigo(self):
        self.assertEqual(_evaluar()["threshold"], POLITICAS["confidence_threshold"])


class LaTrazaNoLLEVA_ElDatoClinicoTest(unittest.TestCase):
    def test_ni_la_nota_ni_el_identificador_aparecen(self):
        r = _evaluar(nota=_NOTA_CON_DATO)
        # ANCLA POSITIVA (auditoría 2ª pasada): si el pipeline no hubiera
        # corrido, el secreto tampoco estaría en la traza y la prueba
        # pasaría sin haber comprobado nada.
        self.assertEqual(r["outcome"], "prediction")
        self.assertEqual(len(r["trace"]["nodes"]), 4)
        serializado = json.dumps(r["trace"], ensure_ascii=False, default=str)
        self.assertNotIn("NHC-4491203", serializado)
        self.assertNotIn("disnea de reposo", serializado)

    def test_ni_en_el_recibo(self):
        r = _evaluar(nota=_NOTA_CON_DATO)
        self.assertEqual(len(r["receipt"]["steps"]), 4)
        serializado = json.dumps(r["receipt"], ensure_ascii=False, default=str)
        self.assertNotIn("NHC-4491203", serializado)

    def test_pero_SI_su_huella_para_poder_compararlo(self):
        r = _evaluar(nota=_NOTA_CON_DATO)
        self.assertTrue(r["receipt"]["input"]["text_digest"].startswith("sha256:"))
        self.assertFalse(r["receipt"]["input"]["raw_data_included"])

    def test_el_perfil_de_privacidad_declara_QUE_no_entra(self):
        r = _evaluar()
        self.assertEqual(r["privacy"]["network"], "disabled")
        for campo in CAMPOS_SENSIBLES:
            self.assertIn(campo, r["privacy"]["trace_never_contains"])


class ElRECIBO_IdentificaTodasLasVersionesTest(unittest.TestCase):
    """§16.6: «el recibo identifica todas las versiones de modelos»."""

    def test_los_cuatro_nodos_con_su_digest(self):
        recibo = _evaluar()["receipt"]
        modelos = {m["model_id"] for m in recibo["models"]}
        self.assertEqual(modelos, {"tabular", "texto", "fusion", "clasificador"})
        self.assertTrue(all(m["digest"].startswith("sha256:") for m in recibo["models"]))

    def test_cambiar_una_POLITICA_cambia_el_digest_del_nodo(self):
        """Si no cambiara, el recibo declararía el mismo nodo para dos
        comportamientos distintos."""
        uno = identidad_de_nodo("v1", {"text_max_bytes": 512})
        otro = identidad_de_nodo("v1", {"text_max_bytes": 4096})
        self.assertNotEqual(uno, otro)

    def test_el_recibo_lleva_el_momento_de_corte(self):
        recibo = _evaluar()["receipt"]
        self.assertEqual(recibo["input"]["data_cutoff"], MOMENTO)

    def test_y_la_regla_que_decidio_la_abstencion(self):
        recibo = _evaluar(probabilidad=0.55)["receipt"]
        reglas = [p.get("rule_id") for p in recibo["checks"]["policy_results"]]
        self.assertIn("R4-baja-confianza", reglas)


class EjecucionSIN_RED_YReproduccionTest(unittest.TestCase):
    def test_el_perfil_declara_ejecucion_local_sin_red(self):
        self.assertEqual(_evaluar()["privacy"]["profile"], "local-only")

    def test_el_dataset_se_REGENERA_igual_con_la_misma_semilla(self):
        """Sin esto, `R1` del paquete no puede comparar nada."""
        self.assertEqual(generar_dataset(50, 7), generar_dataset(50, 7))

    def test_con_otra_semilla_NO_es_el_mismo(self):
        self.assertNotEqual(generar_dataset(50, 7), generar_dataset(50, 8))

    def test_la_regla_de_la_etiqueta_esta_PUBLICADA(self):
        """Un dataset sintético cuya regla no se publica es tan opaco como
        uno real, y encima falsamente tranquilizador."""
        from matrixai.reference.readmission_datos import REGLA_DE_LA_ETIQUETA
        self.assertIn("riesgo", REGLA_DE_LA_ETIQUETA)
        self.assertIn("etiqueta =", REGLA_DE_LA_ETIQUETA)

    def test_las_columnas_del_csv_son_LAS_de_la_fusion(self):
        """Si el entrenamiento viera unas columnas y el servicio armara
        otras, el modelo predeciría bien en la suite y mal en el
        producto."""
        self.assertEqual(len(columnas()), TALLA_TABULAR + len(VOCABULARIO))
        self.assertEqual(len(vector_de(CASO, NOTA)), len(columnas()))


class UnSUBMODELO_AlteradoInvalidaLaVerificacionTest(unittest.TestCase):
    """§16.6: «la modificación de un submodelo invalida la verificación»."""

    def setUp(self):
        if not (EJEMPLO / "registry").exists():
            self.skipTest("el registry del ejemplo no está construido")
        self.raiz = Path(tempfile.mkdtemp())
        shutil.copytree(EJEMPLO / "registry", self.raiz / "registry")
        self.registro = ModelRegistry(self.raiz / "registry")

    def tearDown(self):
        shutil.rmtree(self.raiz, ignore_errors=True)

    def test_el_modelo_INTACTO_se_puede_usar(self):
        clasificar, digest, version = clasificador_del_registry(
            self.registro, "readmission_head", "v1")
        self.assertTrue(digest.startswith("sha256:"))
        probabilidades = clasificar(vector_de(CASO, NOTA))
        self.assertEqual(set(probabilidades), {"no_reingreso", "reingreso"})
        self.assertAlmostEqual(sum(probabilidades.values()), 1.0, places=5)

    def test_TOCAR_los_pesos_impide_usarlo(self):
        params = self.raiz / "registry/entries/readmission_head/v1/params.json"
        params.write_text(json.dumps(json.loads(params.read_text())))  # otros bytes
        with self.assertRaises(ValueError) as caja:
            clasificador_del_registry(self.registro, "readmission_head", "v1")
        self.assertIn("integridad", str(caja.exception))

    def test_un_vector_de_otra_talla_NO_se_ejecuta(self):
        clasificar, _, _ = clasificador_del_registry(
            self.registro, "readmission_head", "v1")
        with self.assertRaises(ValueError) as caja:
            clasificar([0.0] * 5)
        self.assertIn("espera 20", str(caja.exception))


class ACuienDejaSinRespuestaLaABSTENCIONTest(unittest.TestCase):
    """El hallazgo más importante del corte, medido el 2026-08-20 sobre el
    propio dataset del caso: se abstiene en el **39 %** en global, pero en
    el **89,5 %** de los reingresos reales y en el **16,3 %** de los
    demás — y cuando habla acierta el **94,7 %**.

    Ese 94,7 % es un número estupendo conseguido **callándose justo donde
    importa**. La tasa global lo esconde entero, y por eso se mide por
    clase."""

    _PREDICCIONES = (
        # nueve «no» respondidos con seguridad…
        [("no", 0.95, "no")] * 9
        # …y nueve «sí» que se quedan por debajo del umbral.
        + [("si", 0.55, "si")] * 9
    )

    def test_la_tasa_GLOBAL_esconde_a_quien_se_deja_fuera(self):
        r = cobertura_por_clase(list(self._PREDICCIONES))
        self.assertAlmostEqual(r["abstention_rate"], 0.5)
        # …y por clase se ve que no es prudencia repartida.
        self.assertAlmostEqual(r["by_class"]["si"]["abstention_rate"], 1.0)
        self.assertAlmostEqual(r["by_class"]["no"]["abstention_rate"], 0.0)

    def test_se_NOMBRA_la_clase_peor_cubierta(self):
        r = cobertura_por_clase(list(self._PREDICCIONES))
        self.assertEqual(r["worst_covered_class"], "si")
        self.assertAlmostEqual(r["spread"], 1.0)

    def test_la_exactitud_AL_RESPONDER_va_con_su_aviso(self):
        """Sube justamente por callarse los casos difíciles."""
        r = cobertura_por_clase(list(self._PREDICCIONES))
        self.assertEqual(r["accuracy_when_answering"], 1.0)
        self.assertIn("callarse los casos difíciles", r["caveat"])

    def test_la_cobertura_se_enseña_ademas_de_la_abstencion(self):
        """«Cubre el 10 %» se entiende peor que «se calla el 90 %», y son
        el mismo dato: se dan los dos."""
        r = cobertura_por_clase(list(self._PREDICCIONES))
        for datos in r["by_class"].values():
            self.assertAlmostEqual(datos["coverage"], 1 - datos["abstention_rate"])

    def test_sin_predicciones_no_dice_cobertura_perfecta(self):
        r = cobertura_por_clase([])
        self.assertFalse(r["measured"])
        self.assertIsNone(r["abstention_rate"])


class LaDEMOSTRACION_DiceLoQueHaceTest(unittest.TestCase):
    """El docstring del programa listaba **seis** pasos cuando ya imprimía
    **siete**, y en otro orden: al insertar el de la cobertura por clase
    se quedó atrás.

    Es el mismo defecto que la 3ª pasada de auditoría encontró en el caso
    de routing —una «salida esperada» que no es la que sale— cometido aquí
    por quien lo estaba arreglando allí. Y **a mano se vuelve a
    descolgar**, así que se comprueba: los títulos que el programa imprime
    y los que su docstring promete tienen que ser los mismos.
    """

    _PROGRAMA = EJEMPLO / "run_case.py"

    def _titulos_impresos(self):
        import re
        fuente = self._PROGRAMA.read_text(encoding="utf-8")
        return re.findall(r'_separador\("Paso (\d+) — ([^"]+)"\)', fuente)

    def _titulos_prometidos(self):
        import ast
        import re
        doc = ast.get_docstring(ast.parse(self._PROGRAMA.read_text(encoding="utf-8"))) or ""
        return re.findall(r"^ (\d+)\. (.+?)[;.]$", doc, re.M)

    def test_el_docstring_lista_LOS_MISMOS_pasos_que_se_imprimen(self):
        impresos, prometidos = self._titulos_impresos(), self._titulos_prometidos()
        self.assertTrue(impresos, "no se encontró ningún paso impreso")
        self.assertEqual([n for n, _ in impresos], [n for n, _ in prometidos])
        self.assertEqual([t.strip() for _, t in impresos],
                         [t.strip() for _, t in prometidos])

    def test_y_van_numerados_sin_saltos(self):
        """Un salto de numeración es la señal de que se insertó uno y no
        se renumeró el resto — que fue justo lo que pasó."""
        numeros = [int(n) for n, _ in self._titulos_impresos()]
        self.assertEqual(numeros, list(range(1, len(numeros) + 1)))


class ExportacionONNX_YPaqueteTest(unittest.TestCase):
    """§16.4: «exportación ONNX cuando sea compatible». Lo es: medido el
    2026-08-20, opset 17, entrada `[-1, 20]` y salida `[-1, 2]`."""

    def setUp(self):
        if not (EJEMPLO / "run" / "parameter_set.json").exists():
            self.skipTest("el caso no está entrenado")

    def test_el_modelo_del_caso_exporta_a_ONNX(self):
        import subprocess
        import sys

        destino = Path(tempfile.mkdtemp()) / "readmission.onnx"
        try:
            salida = subprocess.run(
                [sys.executable, "-m", "matrixai", "export-onnx",
                 str(EJEMPLO / "readmission.mxai"),
                 "--params", str(EJEMPLO / "run" / "parameter_set.json"),
                 "--output", str(destino)],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(salida.returncode, 0, salida.stderr)
            self.assertTrue(destino.exists())
            # La talla de entrada del ONNX es la del vector de la fusión:
            # si no lo fuera, el modelo exportado no serviría para este caso.
            self.assertIn("20", salida.stdout)
        finally:
            shutil.rmtree(destino.parent, ignore_errors=True)


class ElANALISIS_DeConfianzaTest(unittest.TestCase):
    """§16.4 pide «calibración o análisis de confianza». Y el aviso que va
    con el número importa tanto como el número."""

    def test_mide_la_brecha_entre_lo_que_dice_y_lo_que_acierta(self):
        a = analisis_de_confianza([("a", 0.9, "a")] * 9 + [("a", 0.9, "b")])
        self.assertTrue(a["measured"])
        self.assertAlmostEqual(a["bins"][0]["accuracy"], 0.9)
        self.assertLess(a["ece"], 0.01)

    def test_un_modelo_SOBRECONFIADO_se_ve(self):
        a = analisis_de_confianza([("a", 0.95, "b")] * 10)
        self.assertGreater(a["ece"], 0.9)
        self.assertLess(a["bins"][0]["gap"], 0)

    def test_sin_predicciones_NO_dice_calibracion_perfecta(self):
        """Un ECE de 0.0 sobre una tabla vacía se leería como perfecta."""
        a = analisis_de_confianza([])
        self.assertFalse(a["measured"])
        self.assertIsNone(a["ece"])

    def test_el_aviso_de_ALCANCE_va_con_el_numero(self):
        """La calibración no es una defensa contra lo que queda fuera de
        distribución, y confundirlas es la media verdad tranquilizadora."""
        a = analisis_de_confianza([("a", 0.9, "a")])
        self.assertEqual(a["scope"], "in-distribution only")
        self.assertIn("fuera de distribución", a["caveat"])

    def test_una_confianza_imposible_se_rechaza(self):
        with self.assertRaises(ValueError):
            analisis_de_confianza([("a", 1.4, "a")])


if __name__ == "__main__":
    unittest.main()
