# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0, condición (3) — el arnés de
`benchmarks/texto_107c30/medir_c30_respondedor.py` y de `correr_respondedor_en_
contenedor.sh`. Corre en el HOST, SIN `onnxruntime_genai` (no está instalado a
propósito -- "no instales nada en el host", 26-09): el módulo bajo prueba solo
lo importa DENTRO de las dos funciones que llaman al modelo de verdad
(`_cargar_modelo_y_tokenizer`, `_generar_respuesta_atomica`), que estas
pruebas no ejercitan -- se sustituyen por un respondedor FALSO inyectado.

Cubre lo que pide el corte:
  - los términos de la pregunta salen SOLO de TRAIN (sin fuga a dev/test);
  - la plantilla de pregunta es la LITERAL del registro sellado;
  - el análisis de la respuesta (sí/no/faltante, con los casos del registro);
  - la caché por (tarea, row_id, término) no repite en un relanzamiento;
  - el truncado iguala al del embedding de C1 (con el tokenizador REAL,
    tokenizers/Unigram -- ya está en el host, 107-C1 D1);
  - una tarea sintética pequeña de punta a punta con un respondedor FALSO;
  - el guion de contenedor lleva `--init` y `& wait` (la trampa del 26-09 en
    `correr_116c2_en_contenedor.sh`, que este guion copia a propósito).

Los sabotajes que este fichero sostiene (comprobados a mano: copia, diff,
rojo, restaurar, `md5sum -c`, borrar `__pycache__` -- ver el informe del
corte):
  1. dejar que `evaluar_tarea_respondedor` pase también filas de TEST (o de
     dev) a la selección de términos;
  2. una plantilla de pregunta retocada;
  3. un análisis de respuesta que acepta "quizá" como "no";
  4. una caché que no evita repetir una pregunta ya contestada;
  5. quitar `--init`/`& wait` del guion de contenedor.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "benchmarks" / "texto_107c30"))
import medir_c30 as mc  # noqa: E402
import medir_c30_respondedor as mr  # noqa: E402

GUION_CONTENEDOR = _RAIZ / "benchmarks" / "texto_107c30" / "correr_respondedor_en_contenedor.sh"


# ---------------------------------------------------------------------------
# utilidades para construir una tarea sintética y registrarla en mc.TAREAS
# ---------------------------------------------------------------------------

def _tarea_sintetica(n: int = 60, seed: int = 1) -> list[dict]:
    """Filas con una señal de texto REAL (para que TF-IDF/chi² tengan algo
    que aprender) y una columna 'otra' con señal más débil -- mismo patrón
    que `test_107_c30_arnes._tarea_sintetica_binaria`, repetido aquí para no
    acoplar los dos ficheros de prueba entre sí."""
    import random
    rng = random.Random(seed)
    filas = []
    for i in range(n):
        positivo = rng.random() < 0.5
        texto = ("contrato de suministro urgente prioritario" if positivo
                else "informe rutinario ordinario de seguimiento")
        if rng.random() < 0.15:
            texto = "palabra neutra sin relacion aparente"
        particion = "train" if i < int(n * 0.6) else ("dev" if i < int(n * 0.8) else "test")
        filas.append({
            "row_id": f"s{i}", "particion": particion,
            "target": "si" if positivo else "no",
            "texto": texto, "otra": "x" if positivo else "y",
        })
    return filas


def _tarea_con_marcador_solo_en_test(n_train: int = 40, n_test: int = 20) -> list[dict]:
    """Una tarea donde el término 'exclusivotest' aparece SOLO en las filas
    POSITIVAS de test -- nunca en negativas de test, nunca en train ni en
    dev -- así que está PERFECTAMENTE correlacionado con el target dentro de
    train+test combinados (chi² altísimo, sin competencia real). Si la
    selección de términos mirase test (fuga), 'exclusivotest' ganaría; si
    mira solo train, ni siquiera puede: la palabra no existe en ese
    vocabulario. (La primera versión de esta tarea usaba 'exclusivotest' en
    los DOS lados de test -- positivo y negativo -- y por eso no
    correlacionaba con nada: la sabotage la cazó vía el espía de cableado
    pero NO vía este comportamiento, hasta que se corrigió el fixture.)"""
    filas = []
    for i in range(n_train):
        positivo = i % 2 == 0
        texto = ("trainpositivo termino comun de entrenamiento aqui" if positivo
                else "trainnegativo termino comun de entrenamiento aqui")
        filas.append({"row_id": f"tr{i}", "particion": "train",
                      "target": "si" if positivo else "no", "texto": texto, "otra": "x"})
    # un puñado de dev, sin marcador, para que la partición no quede vacía
    for i in range(6):
        positivo = i % 2 == 0
        filas.append({"row_id": f"dv{i}", "particion": "dev",
                      "target": "si" if positivo else "no",
                      "texto": "texto de desarrollo neutro", "otra": "x"})
    for i in range(n_test):
        positivo = i % 2 == 0
        texto = "exclusivotest exclusivotest aparece aqui" if positivo else "otro contenido de test cualquiera"
        filas.append({"row_id": f"te{i}", "particion": "test",
                      "target": "si" if positivo else "no", "texto": texto, "otra": "x"})
    return filas


class TareaSinteticaMixin:
    """Registra una tarea sintética en `mc.TAREAS`/`mc.leer_tarea`, sin tocar
    disco ni el registro real -- se limpia en tearDown. Mismo mecanismo que
    `test_107_c30_arnes.TareaSinteticaMixin`, repetido aquí para no acoplar
    los dos ficheros."""

    NOMBRE = "_sintetica_respondedor_test_"
    FILAS_FACTORY = staticmethod(_tarea_sintetica)

    def setUp(self):
        self.filas = self.FILAS_FACTORY()
        self._particiones = {"train": [], "dev": [], "test": []}
        for f in self.filas:
            self._particiones[f["particion"]].append(f)
        mc.TAREAS[self.NOMBRE] = dict(
            fichero="__no_usado__.csv", tipo="binary_classification",
            clases=("no", "si"), positive_label="si",
            otras_columnas=("otra",), metrica="auroc", idioma="es",
        )
        self._leer_original = mc.leer_tarea
        mc.leer_tarea = lambda nombre: self._particiones if nombre == self.NOMBRE else self._leer_original(nombre)

    def tearDown(self):
        mc.TAREAS.pop(self.NOMBRE, None)
        mc.leer_tarea = self._leer_original


def _cache_temporal() -> mr.CacheRespondedor:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    Path(tmp.name).unlink()  # que no exista: CacheRespondedor arranca vacía
    return mr.CacheRespondedor(Path(tmp.name))


def _respondedor_falso_por_contenido(pistas_si: tuple[str, ...] = ("positivo", "trainpositivo")):
    """Un respondedor FALSO: contesta 'sí' si el mensaje contiene alguna de
    las pistas, 'no' si no -- determinista, sin tocar onnxruntime_genai."""
    llamadas: list[str] = []

    def generar(mensaje: str) -> str:
        llamadas.append(mensaje)
        return "sí" if any(p in mensaje for p in pistas_si) else "no"

    return generar, llamadas


# ---------------------------------------------------------------------------
# la plantilla de pregunta, LITERAL del registro sellado
# ---------------------------------------------------------------------------

class PlantillaPreguntaTest(unittest.TestCase):
    def test_plantilla_es_la_literal_del_registro_sellado(self):
        self.assertEqual(
            mr.PLANTILLA_PREGUNTA.format(termino="mobiliario"),
            "¿El texto trata de «mobiliario» o de algo equivalente? Responde solo sí o no.",
        )

    def test_n_preguntas_es_5(self):
        self.assertEqual(mr.N_PREGUNTAS, 5)

    def test_tareas_a_medir_son_a_y_b(self):
        self.assertEqual(mr.TAREAS_A_MEDIR, ("A", "B"))

    def test_c_y_d_declaradas_omitidas_por_coste(self):
        self.assertIn("C", mr.TAREAS_OMITIDAS)
        self.assertIn("D", mr.TAREAS_OMITIDAS)
        for motivo in mr.TAREAS_OMITIDAS.values():
            self.assertIn("coste", motivo)


# ---------------------------------------------------------------------------
# análisis de la respuesta
# ---------------------------------------------------------------------------

class AnalizarRespuestaTest(unittest.TestCase):
    def test_casos_del_registro(self):
        casos = {"Sí": 1, "sí.": 1, "No": 0, "no,": 0, "quizá": None}
        for bruta, esperado in casos.items():
            with self.subTest(bruta=bruta):
                self.assertEqual(mr.analizar_respuesta(bruta), esperado)

    def test_mayusculas_y_puntuacion_variadas(self):
        self.assertEqual(mr.analizar_respuesta("SÍ, claro que sí"), 1)
        self.assertEqual(mr.analizar_respuesta("  no.\n"), 0)
        self.assertEqual(mr.analizar_respuesta("Sin duda"), None)

    def test_respuesta_vacia_es_faltante(self):
        self.assertIsNone(mr.analizar_respuesta(""))

    def test_solo_puntuacion_es_faltante(self):
        self.assertIsNone(mr.analizar_respuesta("..."))


# ---------------------------------------------------------------------------
# la selección de términos, SOLO con train (sin fuga)
# ---------------------------------------------------------------------------

class SeleccionDeTerminosSinFugaTest(TareaSinteticaMixin, unittest.TestCase):
    FILAS_FACTORY = staticmethod(_tarea_con_marcador_solo_en_test)

    def test_terminos_top_chi2_directa_no_ve_el_marcador_de_test(self):
        """Llamada directa: si accidentalmente se le pasaran filas de test,
        'exclusivotest' ganaría con un chi² sin competencia. Pasándole SOLO
        train (que es lo único que `evaluar_tarea_respondedor` debe hacer),
        la palabra ni siquiera puede aparecer: no está en su vocabulario."""
        train = self._particiones["train"]
        terminos = mr._terminos_top_chi2([f["texto"] for f in train], [f["target"] for f in train])
        nombres = {t["termino"] for t in terminos}
        self.assertNotIn("exclusivotest", nombres)

    def test_evaluar_tarea_respondedor_llama_a_terminos_top_chi2_solo_con_train(self):
        """Prueba de cableado: espía `_terminos_top_chi2` y comprueba que
        recibe EXACTAMENTE los textos/targets de TRAIN -- ni una fila de dev
        ni de test. Esta es la prueba con nombre para el sabotaje 1."""
        train = self._particiones["train"]
        textos_train_esperados = [f["texto"] for f in train]
        target_train_esperado = [f["target"] for f in train]
        recibido = {}
        original = mr._terminos_top_chi2  # ANTES de parchear: si se llamara a mr._terminos_top_chi2
        # dentro de la propia espía, recursión infinita -- la espía tiene que llamar a ESTA
        # referencia, capturada antes del patch, no al nombre del módulo (que para entonces ya es
        # la propia espía).

        def espia(textos, target, **kwargs):
            recibido["textos"] = list(textos)
            recibido["target"] = list(target)
            return original(textos, target, **kwargs)

        generar, _ = _respondedor_falso_por_contenido()
        cache = _cache_temporal()
        tokenizador = mr._cargar_tokenizador_de_c1()
        with mock.patch.object(mr, "_terminos_top_chi2", espia):
            mr.evaluar_tarea_respondedor(self.NOMBRE, tokenizador_c1=tokenizador,
                                         generar_respuesta=generar, cache=cache)
        self.assertEqual(recibido["textos"], textos_train_esperados)
        self.assertEqual(recibido["target"], target_train_esperado)

    def test_evaluar_tarea_respondedor_no_elige_el_marcador_de_test_de_punta_a_punta(self):
        """La misma garantía, de punta a punta (sin espiar nada): el
        resultado final no puede traer 'exclusivotest' entre sus términos."""
        generar, _ = _respondedor_falso_por_contenido()
        cache = _cache_temporal()
        tokenizador = mr._cargar_tokenizador_de_c1()
        resultado = mr.evaluar_tarea_respondedor(self.NOMBRE, tokenizador_c1=tokenizador,
                                                 generar_respuesta=generar, cache=cache)
        nombres = {t["termino"] for t in resultado["terminos"]}
        self.assertNotIn("exclusivotest", nombres)


class SeleccionDeTerminosUsaElVectorizadorDeC2Test(unittest.TestCase):
    def test_usa_construir_vectorizador_tfidf_de_medir_c30(self):
        """El registro dice 'el TF-IDF de la condición (2)': si algún día
        `medir_c30_respondedor.py` se independiza con su propio
        `TfidfVectorizer`, esta prueba lo cazaría -- comprueba que la
        FUNCIÓN que usa es literalmente `mc.construir_vectorizador_tfidf`."""
        llamado = {"veces": 0}
        original = mc.construir_vectorizador_tfidf

        def espia():
            llamado["veces"] += 1
            return original()

        with mock.patch.object(mc, "construir_vectorizador_tfidf", espia):
            mr._terminos_top_chi2(
                ["contrato de suministro"] * 3 + ["informe rutinario"] * 3,
                ["si", "si", "si", "no", "no", "no"],
            )
        self.assertGreaterEqual(llamado["veces"], 1)


# ---------------------------------------------------------------------------
# el truncado, igual que el embedding de C1
# ---------------------------------------------------------------------------

class TruncadoComoEmbeddingC1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizador = mr._cargar_tokenizador_de_c1()

    def test_texto_corto_no_se_toca(self):
        texto = "un texto corto que no necesita truncarse"
        self.assertEqual(mr.truncar_como_embedding_c1(texto, self.tokenizador), texto)

    def test_texto_largo_queda_en_como_mucho_max_longitud_piezas(self):
        texto_largo = ("palabra " * 500).strip()
        truncado = mr.truncar_como_embedding_c1(texto_largo, self.tokenizador)
        self.assertLess(len(truncado), len(texto_largo))
        n_piezas = len(self.tokenizador.codificar(truncado, con_especiales=False).tokens)
        self.assertLessEqual(n_piezas, self.tokenizador.max_longitud)
        # y no se ha cortado de más: tiene que estar CERCA del tope, no muy por debajo
        self.assertGreater(n_piezas, self.tokenizador.max_longitud - 5)

    def test_truncar_dos_veces_es_estable(self):
        texto_largo = ("palabra " * 500).strip()
        una_vez = mr.truncar_como_embedding_c1(texto_largo, self.tokenizador)
        dos_veces = mr.truncar_como_embedding_c1(una_vez, self.tokenizador)
        self.assertEqual(una_vez, dos_veces)

    def test_usa_la_longitud_fijada_de_matrixai_engines(self):
        from matrixai_engines.embeddings.proveedor_de_texto import LONGITUDES_FIJADAS
        esperado = LONGITUDES_FIJADAS[mr.PROVEEDOR_EMBEDDING_C1].tokens
        self.assertEqual(self.tokenizador.max_longitud, esperado)


# ---------------------------------------------------------------------------
# la caché por (tarea, row_id, término)
# ---------------------------------------------------------------------------

class CacheRespondedorTest(unittest.TestCase):
    def test_obtener_sin_entrada_es_none(self):
        cache = _cache_temporal()
        self.assertIsNone(cache.obtener("A", "fila-1", "termino"))

    def test_fijar_y_obtener_en_memoria(self):
        cache = _cache_temporal()
        cache.fijar("A", "fila-1", "termino", {"respuesta_bruta": "sí", "valor": 1})
        self.assertEqual(cache.obtener("A", "fila-1", "termino"), {"respuesta_bruta": "sí", "valor": 1})

    def test_persistir_y_recargar_desde_disco(self):
        cache = _cache_temporal()
        cache.fijar("B", "fila-9", "dolor", {"respuesta_bruta": "no", "valor": 0})
        cache.persistir()
        recargada = mr.CacheRespondedor(cache.ruta)
        self.assertEqual(recargada.obtener("B", "fila-9", "dolor"), {"respuesta_bruta": "no", "valor": 0})

    def test_claves_distintas_no_se_confunden(self):
        """(tarea, row_id, término) tiene que ser la clave ENTERA: la misma
        fila con otro término, u otra tarea con el mismo row_id, no puede
        devolver la respuesta de otra pregunta."""
        cache = _cache_temporal()
        cache.fijar("A", "f1", "contrato", {"respuesta_bruta": "sí", "valor": 1})
        cache.fijar("A", "f1", "servicio", {"respuesta_bruta": "no", "valor": 0})
        cache.fijar("B", "f1", "contrato", {"respuesta_bruta": "no", "valor": 0})
        self.assertEqual(cache.obtener("A", "f1", "contrato")["valor"], 1)
        self.assertEqual(cache.obtener("A", "f1", "servicio")["valor"], 0)
        self.assertEqual(cache.obtener("B", "f1", "contrato")["valor"], 0)

    def test_fichero_corrupto_no_revienta_la_carga(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{esto no es json")
        tmp.close()
        cache = mr.CacheRespondedor(Path(tmp.name))  # no debe lanzar
        self.assertIsNone(cache.obtener("A", "f1", "x"))


class CacheEvitaRepetirPreguntasTest(TareaSinteticaMixin, unittest.TestCase):
    def test_un_relanzamiento_no_vuelve_a_preguntar_lo_ya_contestado(self):
        """Prueba con nombre para el sabotaje 4: corre la tarea sintética dos
        veces con la MISMA caché (recargada de disco entre medias, como en un
        relanzamiento real) y comprueba que la SEGUNDA vuelta no le pide NADA
        nuevo al respondedor."""
        generar, llamadas = _respondedor_falso_por_contenido()
        cache_ruta = _cache_temporal().ruta
        tokenizador = mr._cargar_tokenizador_de_c1()

        cache1 = mr.CacheRespondedor(cache_ruta)
        mr.evaluar_tarea_respondedor(self.NOMBRE, tokenizador_c1=tokenizador,
                                     generar_respuesta=generar, cache=cache1)
        n_llamadas_primera_vuelta = len(llamadas)
        self.assertGreater(n_llamadas_primera_vuelta, 0, "la primera vuelta tiene que preguntar algo")

        # "relanzamiento": una caché NUEVA, recargada del mismo fichero en disco
        cache2 = mr.CacheRespondedor(cache_ruta)
        mr.evaluar_tarea_respondedor(self.NOMBRE, tokenizador_c1=tokenizador,
                                     generar_respuesta=generar, cache=cache2)
        self.assertEqual(len(llamadas), n_llamadas_primera_vuelta,
                         "la segunda vuelta no debería haber llamado al respondedor ni una vez más")


# ---------------------------------------------------------------------------
# punta a punta con un respondedor FALSO
# ---------------------------------------------------------------------------

class PuntaAPuntaRespondedorFalsoTest(TareaSinteticaMixin, unittest.TestCase):
    def test_tarea_pequena_completa_con_respondedor_falso(self):
        generar, llamadas = _respondedor_falso_por_contenido(pistas_si=("contrato",))
        cache = _cache_temporal()
        tokenizador = mr._cargar_tokenizador_de_c1()

        resultado = mr.evaluar_tarea_respondedor(self.NOMBRE, tokenizador_c1=tokenizador,
                                                 generar_respuesta=generar, cache=cache)

        # estructura básica
        self.assertEqual(resultado["tarea"], self.NOMBRE)
        self.assertEqual(len(resultado["terminos"]), 5)
        cond3 = resultado["condiciones"]["3_respondedor"]
        self.assertIsInstance(cond3["metrica_puntual"], float)
        self.assertIn("otra", cond3["columnas"])
        self.assertEqual(len([c for c in cond3["columnas"] if c.startswith("pregunta_")]), 5)

        # veredicto frente a (0), con el margen del registro
        comparacion = resultado["comparaciones"]["3_vs_0"]
        self.assertIn(comparacion["veredicto"], ("mejora", "inferioridad", "equivalencia_practica", "inconcluso"))
        self.assertEqual(comparacion["margen_equivalencia"], mc.MARGEN_EQUIVALENCIA)

        # diferencia frente a (2), DESCRITA, sin veredicto
        descrita = resultado["comparaciones"]["3_vs_2_diferencia_descrita"]
        self.assertNotIn("veredicto", descrita)
        self.assertIsInstance(descrita["metrica_puntual_2_tfidf_lineal"], float)
        self.assertIsInstance(descrita["metrica_puntual_3_respondedor"], float)

        # el respondedor tuvo que contestar 5 preguntas por cada fila de la tarea
        n_filas = len(self.filas)
        n_si = cond3["tiempos"]["preguntas_al_modelo"]["n_si"]
        n_no = cond3["tiempos"]["preguntas_al_modelo"]["n_no"]
        n_faltante = cond3["tiempos"]["preguntas_al_modelo"]["n_faltante"]
        self.assertEqual(n_si + n_no + n_faltante, n_filas * 5)
        self.assertEqual(n_faltante, 0)  # el respondedor falso siempre contesta si/no
        self.assertGreater(len(llamadas), 0)


# ---------------------------------------------------------------------------
# el guion de contenedor: --init y & wait (la trampa del 26-09)
# ---------------------------------------------------------------------------

class GuionDeContenedorTest(unittest.TestCase):
    """**Trampa cazada en el propio sabotaje 5 (26-09)**: la primera versión
    de estas pruebas comprobaba `assertIn("--init", contenido)` sobre el
    FICHERO ENTERO -- y el propio comentario que deja el sabotaje ("SABOTAJE
    5: sin --init, sin & wait $!...") vuelve a meter esas dos cadenas como
    PROSA, así que el banco seguía en verde con la orden real ya sin `--init`
    ni `wait $!`. Reescritas para mirar solo líneas SIN COMENTAR y, para
    `--init`/`exec`, solo el bloque de la orden `docker run` de la medición
    de verdad (no el de `--comprobar`, que nunca llevó `--init` a propósito).
    """

    @classmethod
    def setUpClass(cls):
        cls.contenido = GUION_CONTENEDOR.read_text(encoding="utf-8")
        cls.lineas = cls.contenido.splitlines()
        cls.lineas_sin_comentar = [l for l in cls.lineas if not l.strip().startswith("#")]
        cls.contenido_sin_comentar = "\n".join(cls.lineas_sin_comentar)
        cls.indice_medicion = next(
            i for i, l in enumerate(cls.lineas) if "python3 medir_c30_respondedor.py" in l
        )

    def _bloque_medicion_sin_comentar(self, *, antes: int = 15, despues: int = 3) -> list[str]:
        ini = max(0, self.indice_medicion - antes)
        fin = min(len(self.lineas), self.indice_medicion + despues + 1)
        return [l for l in self.lineas[ini:fin] if not l.strip().startswith("#")]

    def test_la_orden_docker_run_de_la_medicion_lleva_init(self):
        bloque = self._bloque_medicion_sin_comentar(despues=0)
        lineas_docker_run = [l for l in bloque if "docker run" in l]
        self.assertTrue(lineas_docker_run, "no se encontró la orden docker run de la medición")
        self.assertTrue(any("--init" in l for l in lineas_docker_run),
                        "la orden docker run de la MEDICIÓN tiene que llevar --init")

    def test_hay_wait_dollar_bang_sin_comentar_tras_el_docker_run_de_medicion(self):
        """`wait $!` es lo que corta la espera cuando el guion recibe la
        señal -- sin esto el trap queda aplazado y el contenedor sobrevive a
        su propio tope (medido el 26-09 en `correr_116c2_en_contenedor.sh`,
        que este guion copia a propósito). Se busca SIN comentarios: un
        comentario que solo MENCIONE "wait $!" no cuenta (ver la nota de la
        clase)."""
        bloque = self._bloque_medicion_sin_comentar()
        self.assertTrue(any("wait $!" in l for l in bloque),
                        "tiene que haber una línea SIN COMENTAR con 'wait $!' junto al docker run de la medición")

    def test_no_usa_exec_para_el_docker_run_de_medicion(self):
        """Con `exec docker run ...` el proceso de bash se REEMPLAZA por
        docker y el trap ya puesto nunca se ejecuta -- tiene que lanzarse en
        segundo plano (`&`) y esperarse con `wait $!`, no con `exec`."""
        bloque = self._bloque_medicion_sin_comentar(despues=0)
        self.assertFalse(any("exec docker run" in l for l in bloque),
                         "la orden docker run de la MEDICIÓN no puede llevar 'exec'")

    def test_red_ninguna(self):
        self.assertIn("--network none", self.contenido_sin_comentar)

    def test_tiene_nombre_de_contenedor_y_trap(self):
        self.assertIn("--name", self.contenido_sin_comentar)
        self.assertIn("trap", self.contenido_sin_comentar)

    def test_techo_de_memoria_y_cpu(self):
        self.assertIn("--memory=4g", self.contenido_sin_comentar)
        self.assertIn("--cpus=4", self.contenido_sin_comentar)

    def test_comprueba_la_carga_antes_de_arrancar(self):
        self.assertIn("/proc/loadavg", self.contenido_sin_comentar)
        self.assertIn("carga", self.contenido_sin_comentar.lower())

    def test_se_niega_si_hay_otro_contenedor_de_medicion_vivo(self):
        self.assertIn("docker ps", self.contenido_sin_comentar)

    def test_es_ejecutable(self):
        import stat
        modo = GUION_CONTENEDOR.stat().st_mode
        self.assertTrue(modo & stat.S_IXUSR, "el guion tiene que ser ejecutable")


if __name__ == "__main__":
    unittest.main()
