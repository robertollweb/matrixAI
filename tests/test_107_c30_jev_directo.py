# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0, condición (5) — «Jev directo, sin entrenar»
(`benchmarks/texto_107c30/medir_c30_jev_directo.py`). Corre en el HOST, SIN
NINGUNA petición de red de verdad: el mismo servidor HTTP falso local de
`test_107_c30_jev.py` (imita la Decisions API de OpenRouter/Jev).

Cubre lo que pide el corte:
  - B se niega (datos clínicos) y **C se niega** (regresión: un `noul` no da
    un número continuo) -- con su PROPIO motivo cada una, ANTES de leer
    datos, y sin pasar por `respondedor_jev.verificar_tarea_permitida` (esa
    tabla es de la condición (4) y no admite D);
  - la pregunta enviada es la LITERAL del registro sellado, por tarea, y las
    de A y D no se intercambian;
  - el `noul` se usa como PUNTUACIÓN sin umbral (0,63 sigue siendo 0,63, no
    se convierte en 1; 0,41 sigue siendo 0,41, no se convierte en 0);
  - un faltante (respuesta no-`noul`) se cuenta y se EXCLUYE de la
    comparación emparejada -- se dice cuántas filas quedan;
  - la caché evita volver a preguntar en un relanzamiento;
  - `--estimar` (`estimar()`) no hace ninguna petición, usa SOLO las filas de
    TEST (no train+dev+test), y declara B y C omitidas con su motivo;
  - la clave nunca aparece en el resultado ni en la caché.

Los DOS sabotajes que este fichero sostiene (copia, diff, rojo, restaurar,
`md5sum -c`, borrar `__pycache__`):
  1. usar el `noul` con umbral 0,5 (en vez de la puntuación cruda) --
     `PuntuacionSinUmbralTest`;
  2. mandar la pregunta de A en D, o al revés -- `PreguntaPorTareaTest`.
"""
from __future__ import annotations

import http.server
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "benchmarks" / "texto_107c30"))
sys.path.insert(0, str(_RAIZ / "benchmarks" / "texto_107c30" / "jev"))
import medir_c30 as mc  # noqa: E402
import medir_c30_jev_directo as mjd  # noqa: E402
import respondedor_jev as rj  # noqa: E402

CLAVE_SENUELO = "sk-senuelo-c30-jev-directo-NUNCA-DEBE-APARECER"


# ---------------------------------------------------------------------------
# servidor HTTP falso: el MISMO patrón que test_107_c30_jev.py
# ---------------------------------------------------------------------------

class _ManejadorJevFalso(http.server.BaseHTTPRequestHandler):
    """Registra cada petición y responde según `respuestas_programadas` (cola
    FIFO de `(status, cuerpo)`); si está vacía, contesta 'sí' (noul=0.9) si
    alguna pista de `pistas_si` aparece en `instructions`, 'no' (noul=0.1) si
    no."""

    def do_POST(self) -> None:  # noqa: N802
        longitud = int(self.headers.get("Content-Length", 0))
        cuerpo_bruto = self.rfile.read(longitud)
        peticion = json.loads(cuerpo_bruto.decode("utf-8"))
        self.server.peticiones.append({
            "cuerpo": peticion,
            "authorization": self.headers.get("Authorization", ""),
        })

        if self.server.respuestas_programadas:
            status, cuerpo = self.server.respuestas_programadas.pop(0)
            self._responder(status, cuerpo)
            return

        estado = str(peticion.get("state", ""))
        respuestas: dict[str, Any] = {}
        for id_pregunta, definicion in peticion.get("questions", {}).items():
            texto = f"{estado} {definicion.get('instructions', '')}"
            noul = 0.9 if any(p in texto for p in self.server.pistas_si) else 0.1
            respuestas[id_pregunta] = {"type": "noul", "noul": noul}
        cuerpo_resp = {"model": peticion.get("model", "?"), "answers": respuestas,
                       "usage": {"input_tokens": 10, "output_tokens": 0}}
        self._responder(200, cuerpo_resp)

    def _responder(self, status: int, cuerpo: Any) -> None:
        if isinstance(cuerpo, (dict, list)):
            datos = json.dumps(cuerpo).encode("utf-8")
        elif isinstance(cuerpo, str):
            datos = cuerpo.encode("utf-8")
        else:
            datos = b"{}"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(datos)

    def log_message(self, *args: Any) -> None:
        pass


class _ServidorJevFalsoMixin:
    def setUp(self) -> None:
        super().setUp()
        self.servidor = http.server.HTTPServer(("127.0.0.1", 0), _ManejadorJevFalso)
        self.servidor.peticiones = []
        self.servidor.respuestas_programadas = []
        self.servidor.pistas_si = ("positivo",)
        self._hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self._hilo.start()
        self.base_url = f"http://127.0.0.1:{self.servidor.server_port}/api/alpha/decisions"

    def tearDown(self) -> None:
        self.servidor.shutdown()
        self.servidor.server_close()
        self._hilo.join(timeout=5)
        super().tearDown()


def _cache_temporal() -> "rj.CacheRespuestasJev":
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    Path(tmp.name).unlink()
    return rj.CacheRespuestasJev(Path(tmp.name))


def _dormir_falso(_segundos: float) -> None:
    pass


def _cuerpo_noul(valor: float) -> dict[str, Any]:
    return {"answers": {"q0": {"type": "noul", "noul": valor}},
            "usage": {"input_tokens": 10, "output_tokens": 0}}


def _cuerpo_no_noul() -> dict[str, Any]:
    """Una respuesta bien formada pero que NO es un `noul` válido (otro
    `type`) -- el caso de faltante que hay que contar y excluir."""
    return {"answers": {"q0": {"type": "choice", "choice": "x"}},
            "usage": {"input_tokens": 10, "output_tokens": 0}}


# ---------------------------------------------------------------------------
# tareas admitidas / prohibidas -- PROPIAS de la condición (5), no las de (4)
# ---------------------------------------------------------------------------

class TareasAdmitidasDirectoTest(unittest.TestCase):
    def test_solo_a_y_d(self):
        self.assertEqual(mjd.TAREAS_ADMITIDAS_DIRECTO, ("A", "D"))

    def test_b_prohibida_con_motivo_clinico(self):
        self.assertIn("B", mjd.TAREAS_PROHIBIDAS_DIRECTO)
        self.assertIn("clínic", mjd.TAREAS_PROHIBIDAS_DIRECTO["B"])

    def test_c_prohibida_con_motivo_regresion(self):
        self.assertIn("C", mjd.TAREAS_PROHIBIDAS_DIRECTO)
        self.assertIn("regresión", mjd.TAREAS_PROHIBIDAS_DIRECTO["C"])

    def test_verificar_acepta_a_y_d(self):
        mjd.verificar_tarea_permitida_directo("A")  # no debe lanzar
        mjd.verificar_tarea_permitida_directo("D")  # no debe lanzar

    def test_verificar_b_se_niega_con_motivo(self):
        with self.assertRaises(SystemExit) as cm:
            mjd.verificar_tarea_permitida_directo("B")
        self.assertIn("clínic", str(cm.exception))
        self.assertIn("PROHIBIDA", str(cm.exception))

    def test_verificar_c_se_niega_con_motivo(self):
        with self.assertRaises(SystemExit) as cm:
            mjd.verificar_tarea_permitida_directo("C")
        self.assertIn("regresión", str(cm.exception))
        self.assertIn("PROHIBIDA", str(cm.exception))

    def test_verificar_tarea_desconocida_se_niega(self):
        with self.assertRaises(SystemExit):
            mjd.verificar_tarea_permitida_directo("Z")

    def test_no_usa_la_tabla_de_la_condicion_4(self):
        """D está PROHIBIDA para (4) (`rj.TAREAS_ADMITIDAS_JEV = ("A", "C")`)
        pero admitida para (5): si `verificar_tarea_permitida_directo`
        delegara por error en `rj.verificar_tarea_permitida`, esta prueba lo
        cazaría."""
        with self.assertRaises(SystemExit):
            rj.verificar_tarea_permitida("D")
        mjd.verificar_tarea_permitida_directo("D")  # no debe lanzar

    def test_evaluar_tarea_directo_niega_b_antes_de_leer_datos(self):
        with self.assertRaises(SystemExit) as cm:
            mjd.evaluar_tarea_directo("B", tokenizador_c1=None, cache=None, clave=None)
        self.assertIn("clínic", str(cm.exception))

    def test_evaluar_tarea_directo_niega_c_antes_de_leer_datos(self):
        with self.assertRaises(SystemExit) as cm:
            mjd.evaluar_tarea_directo("C", tokenizador_c1=None, cache=None, clave=None)
        self.assertIn("regresión", str(cm.exception))


# ---------------------------------------------------------------------------
# la pregunta LITERAL, por tarea, y que no se retoca
# ---------------------------------------------------------------------------

class PreguntasLiteralesTest(unittest.TestCase):
    def test_pregunta_a(self):
        self.assertEqual(mjd.PREGUNTAS_LITERALES["A"],
                         "¿Es falsa esta noticia? Responde solo sí o no.")

    def test_pregunta_d(self):
        self.assertEqual(mjd.PREGUNTAS_LITERALES["D"],
                         "¿Esta norma ha dejado de estar vigente? Responde solo sí o no.")

    def test_a_y_d_son_preguntas_distintas(self):
        self.assertNotEqual(mjd.PREGUNTAS_LITERALES["A"], mjd.PREGUNTAS_LITERALES["D"])


# ---------------------------------------------------------------------------
# responder_etiqueta: UNA pregunta por fila, SIN pasar por rj.verificar_tarea
# ---------------------------------------------------------------------------

class ResponderEtiquetaTest(_ServidorJevFalsoMixin, unittest.TestCase):
    def test_una_peticion_con_una_sola_pregunta(self):
        cache = _cache_temporal()
        mjd.responder_etiqueta(row_id="f1", texto="un texto", pregunta="¿es X?", cache=cache,
                               clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(len(self.servidor.peticiones), 1)
        preguntas = self.servidor.peticiones[0]["cuerpo"]["questions"]
        self.assertEqual(list(preguntas), ["q0"])
        self.assertEqual(preguntas["q0"]["instructions"], "¿es X?")

    def test_segunda_llamada_con_cache_recargada_no_repite_la_peticion(self):
        cache_ruta = _cache_temporal().ruta
        cache1 = rj.CacheRespuestasJev(cache_ruta)
        mjd.responder_etiqueta(row_id="f1", texto="x", pregunta="¿es X?", cache=cache1,
                               clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        n_tras_primera = len(self.servidor.peticiones)
        self.assertGreater(n_tras_primera, 0)

        cache2 = rj.CacheRespuestasJev(cache_ruta)  # "relanzamiento": recargada de disco
        mjd.responder_etiqueta(row_id="f1", texto="x", pregunta="¿es X?", cache=cache2,
                               clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(len(self.servidor.peticiones), n_tras_primera)

    def test_faltante_si_no_es_noul(self):
        self.servidor.respuestas_programadas = [(200, _cuerpo_no_noul())]
        cache = _cache_temporal()
        puntuacion = mjd.responder_etiqueta(row_id="f1", texto="x", pregunta="¿es X?", cache=cache,
                                            clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        self.assertIsNone(puntuacion)
        # y queda contado en la caché, no descartado en silencio
        self.assertIsNone(cache.obtener("f1", mjd._TERMINO_CACHE)["puntuacion"])

    def test_clave_no_aparece_en_la_cache(self):
        cache = _cache_temporal()
        mjd.responder_etiqueta(row_id="f1", texto="un texto con datos", pregunta="¿es X?", cache=cache,
                               clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        contenido = cache.ruta.read_text(encoding="utf-8")
        self.assertNotIn(CLAVE_SENUELO, contenido)


# ---------------------------------------------------------------------------
# SABOTAJE 1: el noul se usa como PUNTUACIÓN, sin umbralizar en 0,5
# ---------------------------------------------------------------------------

class PuntuacionSinUmbralTest(_ServidorJevFalsoMixin, unittest.TestCase):
    def test_noul_alto_no_se_convierte_en_uno(self):
        """0,63 tiene que seguir siendo 0,63 -- un umbral a 0,5 lo
        convertiría en 1 (`rj.analizar_respuesta_noul`, el umbralizado que SÍ
        usan (3)/(4), pero no (5))."""
        self.servidor.respuestas_programadas = [(200, _cuerpo_noul(0.63))]
        cache = _cache_temporal()
        puntuacion = mjd.responder_etiqueta(row_id="f1", texto="x", pregunta="¿es X?", cache=cache,
                                            clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(puntuacion, 0.63)
        self.assertNotEqual(puntuacion, 1)

    def test_noul_bajo_no_se_convierte_en_cero(self):
        """0,41 tiene que seguir siendo 0,41 -- un umbral a 0,5 lo
        convertiría en 0. Sin este caso, un sabotaje que umbralizara pero
        dejara pasar por casualidad los `noul` >= 0,5 tal cual no se
        cazaría."""
        self.servidor.respuestas_programadas = [(200, _cuerpo_noul(0.41))]
        cache = _cache_temporal()
        puntuacion = mjd.responder_etiqueta(row_id="f1", texto="x", pregunta="¿es X?", cache=cache,
                                            clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(puntuacion, 0.41)
        self.assertNotEqual(puntuacion, 0)

    def test_analizar_respuesta_noul_puntuacion_no_umbraliza(self):
        """La función de bajo nivel que reutiliza `responder_etiqueta`."""
        self.assertEqual(rj.analizar_respuesta_noul_puntuacion({"type": "noul", "noul": 0.63}), 0.63)
        self.assertEqual(rj.analizar_respuesta_noul_puntuacion({"type": "noul", "noul": 0.0}), 0.0)
        self.assertEqual(rj.analizar_respuesta_noul_puntuacion({"type": "noul", "noul": 1.0}), 1.0)

    def test_analizar_respuesta_noul_umbralizado_sigue_correcto(self):
        """La función umbralizada (para (3)/(4)) no se ha roto al refactorizarla
        para reusar la puntuación cruda."""
        self.assertEqual(rj.analizar_respuesta_noul({"type": "noul", "noul": 0.63}), 1)
        self.assertEqual(rj.analizar_respuesta_noul({"type": "noul", "noul": 0.41}), 0)


# ---------------------------------------------------------------------------
# SABOTAJE 2: la pregunta de cada tarea no se intercambia
# ---------------------------------------------------------------------------

def _fila(row_id: str, particion: str, target: str, texto: str, otra: str = "x") -> dict[str, Any]:
    return {"row_id": row_id, "particion": particion, "target": target, "texto": texto, "otra": otra}


def _tarea_sintetica_directo(nombre: str, pista_si: str, *, n_train: int = 18, n_dev: int = 6,
                             n_test: int = 6, seed: int = 1) -> list[dict[str, Any]]:
    import random
    rng = random.Random(seed)
    filas: list[dict[str, Any]] = []
    for grupo, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
        for i in range(n):
            positivo = i % 2 == 0
            texto = (f"texto {pista_si} relevante" if positivo
                    else "texto neutro sin relacion aparente")
            filas.append(_fila(f"{nombre}-{grupo}-{i}", grupo, "si" if positivo else "no", texto,
                               otra="x" if positivo else "y"))
    rng.shuffle(filas)
    return filas


class TareaSinteticaDirectoMixin:
    """Da de alta DOS tareas sintéticas ("A"-como y "D"-como) en
    `mc.TAREAS`/`mc.leer_tarea`, y las admite temporalmente en
    `mjd.TAREAS_ADMITIDAS_DIRECTO`/`mjd.PREGUNTAS_LITERALES` -- mismo patrón
    que `TareaSinteticaMixin` de `test_107_c30_jev.py`."""

    NOMBRE_A = "_sintetica_directo_a_"
    NOMBRE_D = "_sintetica_directo_d_"
    PREGUNTA_A = "¿PREGUNTA-SOLO-DE-A?"
    PREGUNTA_D = "¿PREGUNTA-SOLO-DE-D?"

    def setUp(self) -> None:
        super().setUp()
        self._filas = {
            self.NOMBRE_A: _tarea_sintetica_directo(self.NOMBRE_A, "alfa"),
            self.NOMBRE_D: _tarea_sintetica_directo(self.NOMBRE_D, "delta"),
        }
        self._particiones = {}
        for nombre, filas in self._filas.items():
            por_particion = {"train": [], "dev": [], "test": []}
            for f in filas:
                por_particion[f["particion"]].append(f)
            self._particiones[nombre] = por_particion
            mc.TAREAS[nombre] = dict(
                fichero="__no_usado__.csv", tipo="binary_classification",
                clases=("no", "si"), positive_label="si",
                otras_columnas=("otra",), metrica="auroc", idioma="es",
            )
        self._leer_original = mc.leer_tarea
        mc.leer_tarea = lambda nombre: (self._particiones[nombre] if nombre in self._particiones
                                        else self._leer_original(nombre))

        self._admitidas_original = mjd.TAREAS_ADMITIDAS_DIRECTO
        self._preguntas_original = dict(mjd.PREGUNTAS_LITERALES)
        mjd.TAREAS_ADMITIDAS_DIRECTO = self._admitidas_original + (self.NOMBRE_A, self.NOMBRE_D)
        mjd.PREGUNTAS_LITERALES = {**self._preguntas_original,
                                   self.NOMBRE_A: self.PREGUNTA_A, self.NOMBRE_D: self.PREGUNTA_D}

    def tearDown(self) -> None:
        for nombre in self._filas:
            mc.TAREAS.pop(nombre, None)
        mc.leer_tarea = self._leer_original
        mjd.TAREAS_ADMITIDAS_DIRECTO = self._admitidas_original
        mjd.PREGUNTAS_LITERALES = self._preguntas_original
        super().tearDown()


class PreguntaPorTareaTest(_ServidorJevFalsoMixin, TareaSinteticaDirectoMixin, unittest.TestCase):
    def _instructions_enviadas(self) -> list[str]:
        return [p["cuerpo"]["questions"]["q0"]["instructions"] for p in self.servidor.peticiones]

    def test_tarea_a_manda_solo_su_propia_pregunta(self):
        self.servidor.pistas_si = ("alfa",)
        cache = _cache_temporal()
        with _parche_truncado():
            mjd.evaluar_tarea_directo(self.NOMBRE_A, tokenizador_c1=object(), cache=cache,
                                      clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        instrucciones = self._instructions_enviadas()
        self.assertTrue(instrucciones)
        self.assertTrue(all(i == self.PREGUNTA_A for i in instrucciones))
        self.assertNotIn(self.PREGUNTA_D, instrucciones)

    def test_tarea_d_manda_solo_su_propia_pregunta(self):
        self.servidor.pistas_si = ("delta",)
        cache = _cache_temporal()
        with _parche_truncado():
            mjd.evaluar_tarea_directo(self.NOMBRE_D, tokenizador_c1=object(), cache=cache,
                                      clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso)
        instrucciones = self._instructions_enviadas()
        self.assertTrue(instrucciones)
        self.assertTrue(all(i == self.PREGUNTA_D for i in instrucciones))
        self.assertNotIn(self.PREGUNTA_A, instrucciones)


def _parche_truncado():
    """Un `truncar_como_embedding_c1` real exige el tokenizador Unigram de
    C1 (ficheros descargados en la máquina). Estas pruebas no miden el
    truncado -- lo sustituyen por el texto tal cual, sin tocar disco;
    `tokenizador_c1` puede ser cualquier objeto (`object()`), porque el
    parche ignora el segundo argumento."""
    return mock.patch("medir_c30_respondedor.truncar_como_embedding_c1",
                      side_effect=lambda texto, _tok: texto)


# ---------------------------------------------------------------------------
# de punta a punta: faltantes contados y EXCLUIDOS de la comparación
# ---------------------------------------------------------------------------

class PuntaAPuntaDirectoTest(_ServidorJevFalsoMixin, TareaSinteticaDirectoMixin, unittest.TestCase):
    def test_una_fila_sin_respuesta_se_cuenta_y_se_excluye(self):
        n_test = len(self._particiones[self.NOMBRE_A]["test"])
        self.assertGreaterEqual(n_test, 4)
        # la primera fila de test responde mal formado; el resto, con pistas
        self.servidor.respuestas_programadas = [(200, _cuerpo_no_noul())]
        self.servidor.pistas_si = ("alfa",)
        cache = _cache_temporal()

        with _parche_truncado():
            resultado = mjd.evaluar_tarea_directo(
                self.NOMBRE_A, tokenizador_c1=object(), cache=cache, clave=CLAVE_SENUELO,
                base_url=self.base_url, dormir=_dormir_falso)

        self.assertEqual(resultado["respuestas"]["n_total"], n_test)
        self.assertEqual(resultado["respuestas"]["n_faltante"], 1)
        self.assertEqual(resultado["respuestas"]["n_con_respuesta"], n_test - 1)
        self.assertEqual(resultado["condiciones"]["5_jev_directo"]["n_filas"], n_test - 1)
        # una petición por fila de TEST, ninguna por train/dev
        self.assertEqual(len(self.servidor.peticiones), n_test)

        veredicto = resultado["comparaciones"]["5_vs_0"]
        self.assertIn(veredicto["veredicto"],
                      ("mejora", "inferioridad", "equivalencia_practica", "inconcluso"))
        self.assertEqual(veredicto["margen_equivalencia"], mc.MARGEN_EQUIVALENCIA)

        # descrito, sin veredicto
        for clave_desc in ("5_vs_1_diferencia_descrita", "5_vs_2_diferencia_descrita"):
            self.assertNotIn("veredicto", resultado["comparaciones"][clave_desc])
        self.assertNotIn("5_vs_4_diferencia_descrita", resultado["comparaciones"])  # solo en A real

    def test_sin_faltantes_todas_las_filas_de_test_entran(self):
        self.servidor.pistas_si = ("delta",)
        cache = _cache_temporal()
        n_test = len(self._particiones[self.NOMBRE_D]["test"])

        with _parche_truncado():
            resultado = mjd.evaluar_tarea_directo(
                self.NOMBRE_D, tokenizador_c1=object(), cache=cache, clave=CLAVE_SENUELO,
                base_url=self.base_url, dormir=_dormir_falso)

        self.assertEqual(resultado["respuestas"]["n_faltante"], 0)
        self.assertEqual(resultado["respuestas"]["n_con_respuesta"], n_test)
        self.assertEqual(resultado["condiciones"]["5_jev_directo"]["n_filas"], n_test)

    def test_clave_no_aparece_en_el_resultado(self):
        self.servidor.pistas_si = ("alfa",)
        cache = _cache_temporal()
        with _parche_truncado():
            resultado = mjd.evaluar_tarea_directo(
                self.NOMBRE_A, tokenizador_c1=object(), cache=cache, clave=CLAVE_SENUELO,
                base_url=self.base_url, dormir=_dormir_falso)
        self.assertNotIn(CLAVE_SENUELO, json.dumps(resultado, ensure_ascii=False))


# ---------------------------------------------------------------------------
# --estimar: sin red, solo TEST, B y C declaradas omitidas
# ---------------------------------------------------------------------------

class EstimarSinRedTest(unittest.TestCase):
    def test_estimar_no_llama_a_peticion_decisions(self):
        with mock.patch.object(rj, "_peticion_decisions",
                               side_effect=AssertionError("¡--estimar no debe hacer peticiones!")):
            mjd.estimar()  # no debe lanzar

    def test_estimar_declara_b_y_c_omitidas(self):
        resultado = mjd.estimar()
        self.assertTrue(resultado["tareas"]["B"]["omitida"])
        self.assertIn("clínic", resultado["tareas"]["B"]["motivo"])
        self.assertTrue(resultado["tareas"]["C"]["omitida"])
        self.assertIn("regresión", resultado["tareas"]["C"]["motivo"])

    def test_estimar_usa_solo_filas_de_test(self):
        """A diferencia de la condición (4) (que pregunta por TODA la
        tarea), la (5) solo pregunta por TEST: `n_peticiones` tiene que
        contar solo esas filas, no train+dev+test."""
        filas_a = [{"row_id": f"a{i}", "particion": "train", "target": "si",
                   "texto": "x" * 50, "otra": "x"} for i in range(20)]
        filas_test_a = [{"row_id": f"at{i}", "particion": "test", "target": "si" if i % 2 else "no",
                         "texto": "y" * 50, "otra": "x"} for i in range(5)]
        particiones_a = {"train": filas_a, "dev": [], "test": filas_test_a}
        filas_d_test = [{"row_id": f"dt{i}", "particion": "test", "target": "si" if i % 2 else "no",
                         "texto": "z" * 50, "otra": "x"} for i in range(7)]
        particiones_d = {"train": [], "dev": [], "test": filas_d_test}

        original = mc.leer_tarea
        try:
            mc.leer_tarea = lambda nombre: {"A": particiones_a, "D": particiones_d}.get(
                nombre, original(nombre))
            resultado = mjd.estimar()
        finally:
            mc.leer_tarea = original

        self.assertEqual(resultado["tareas"]["A"]["n_filas_test"], 5)
        self.assertEqual(resultado["tareas"]["A"]["n_peticiones"], 5)  # no 25 (20 train + 5 test)
        self.assertEqual(resultado["tareas"]["D"]["n_filas_test"], 7)
        self.assertEqual(resultado["tareas"]["D"]["n_peticiones"], 7)
        self.assertEqual(resultado["total_peticiones"], 12)
        self.assertGreater(resultado["total_tokens_entrada_estimados"], 0)


# ---------------------------------------------------------------------------
# lectura de lo YA medido, para lo descrito (sin veredicto)
# ---------------------------------------------------------------------------

class LeerMetricaPreviaTest(unittest.TestCase):
    def test_lee_un_valor_anidado(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "r.json"
            ruta.write_text(json.dumps({"tareas": {"A": {"condiciones": {"1_embedding": {
                "metrica_puntual": 0.645}}}}}), encoding="utf-8")
            self.assertEqual(
                mjd._leer_metrica_previa(ruta, "tareas", "A", "condiciones", "1_embedding",
                                         "metrica_puntual"),
                0.645)

    def test_fichero_ausente_da_none(self):
        self.assertIsNone(mjd._leer_metrica_previa(Path("/no/existe/nunca.json"), "a", "b"))

    def test_ruta_de_claves_ausente_da_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "r.json"
            ruta.write_text(json.dumps({"tareas": {}}), encoding="utf-8")
            self.assertIsNone(mjd._leer_metrica_previa(ruta, "tareas", "A", "condiciones"))


class DiferenciaDescritaTest(unittest.TestCase):
    def test_calcula_la_diferencia_y_no_lleva_veredicto(self):
        d = mjd._diferencia_descrita("2_tfidf_lineal", 0.795, 0.715, fuente="resultado_c30.json")
        self.assertNotIn("veredicto", d)
        self.assertAlmostEqual(d["diferencia_5_menos_2"], 0.715 - 0.795)
        self.assertEqual(d["fuente"], "resultado_c30.json")

    def test_none_si_falta_cualquiera_de_las_dos_metricas(self):
        d = mjd._diferencia_descrita("1_embedding", None, 0.715, fuente="x")
        self.assertIsNone(d["diferencia_5_menos_1"])


if __name__ == "__main__":
    unittest.main()
