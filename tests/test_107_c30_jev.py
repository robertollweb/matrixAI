# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0, condición (4) — el adaptador de Jev
(`benchmarks/texto_107c30/jev/respondedor_jev.py`) y el medidor
(`benchmarks/texto_107c30/medir_c30_jev.py`). Corre en el HOST, **SIN
NINGUNA petición de red de verdad**: un servidor HTTP falso local
(`http.server` en un hilo) imita la Decisions API de OpenRouter/Jev tal
como la documenta `jev/LEEME.md`.

Cubre lo que pide el corte:
  - el mapeo `noul` -> sí(1)/no(0)/faltante(None), y que el faltante se
    cuenta;
  - la caché por (row_id, término) -- un relanzamiento no repite una
    pregunta ya contestada;
  - sin clave (fichero ausente o variable vacía) -- se niega, con motivo;
  - la clave NUNCA aparece en ningún fichero escrito ni en el texto de una
    excepción, ni siquiera cuando el propio cuerpo de una respuesta de error
    la trae de vuelta (una clave señuelo única, buscada byte a byte);
  - la tarea B se rechaza ANTES de tocar la red (en el adaptador Y en el
    medidor -- dos capas);
  - 429 se reintenta con espera exponencial y se rinde tras el tope; un
    error fatal (401...) no reintenta nunca;
  - el batching nativo: una fila con N_PREGUNTAS términos manda UNA sola
    petición, no N_PREGUNTAS;
  - `--estimar` no hace NINGUNA petición (ni una fila real, ni B);
  - la tarea C (regresión) exige `--seleccion-terminos-c` sin valor por
    omisión, y sus dos opciones (`chi2_quintiles`, `f_regression`) no ven
    dev/test;
  - de punta a punta, una tarea sintética pequeña con el servidor falso.

Los sabotajes que este fichero sostiene (copia, diff, rojo, restaurar,
`md5sum -c`, borrar `__pycache__` -- ver el informe del corte):
  1. escribir la clave en el resultado o en la caché (o dejar de redactarla
     de un mensaje de error);
  2. no contar el faltante (aceptar cualquier cosa como sí/no);
  3. admitir la tarea B;
  4. no usar la caché (repetir una pregunta ya contestada).
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
import medir_c30_respondedor as mr  # noqa: E402
import medir_c30_jev as mj  # noqa: E402
import respondedor_jev as rj  # noqa: E402

CLAVE_SENUELO = "sk-senuelo-9f31c2b7e5a04d18-NUNCA-DEBE-APARECER"


# ---------------------------------------------------------------------------
# servidor HTTP falso: imita la Decisions API (ver jev/LEEME.md)
# ---------------------------------------------------------------------------

class _ManejadorJevFalso(http.server.BaseHTTPRequestHandler):
    """Registra cada petición (cuerpo decodificado + cabecera Authorization)
    en `self.server.peticiones`, y responde según `self.server.
    respuestas_programadas` (una cola de `(status, cuerpo_dict_o_texto)`); si
    está vacía, responde de verdad: 'sí' (noul=0.9) para cada pregunta cuyo
    `instructions` o el `state` contenga alguna pista de
    `self.server.pistas_si`, 'no' (noul=0.1) si no."""

    def do_POST(self) -> None:  # noqa: N802 -- nombre fijado por BaseHTTPRequestHandler
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

    def log_message(self, *args: Any) -> None:  # silencio: no ensuciar la salida de la prueba
        pass


class _ServidorJevFalsoMixin:
    """Levanta un `_ManejadorJevFalso` en un puerto libre de 127.0.0.1, en un
    hilo daemon, y lo apaga en `tearDown`. `self.base_url` apunta a él;
    `self.servidor.peticiones` es la lista de peticiones recibidas;
    `self.servidor.respuestas_programadas` permite forzar códigos de error."""

    def setUp(self) -> None:
        super().setUp()
        self.servidor = http.server.HTTPServer(("127.0.0.1", 0), _ManejadorJevFalso)
        self.servidor.peticiones = []
        self.servidor.respuestas_programadas = []
        self.servidor.pistas_si = ("positivo", "trainpositivo")
        self._hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self._hilo.start()
        self.base_url = f"http://127.0.0.1:{self.servidor.server_port}/api/alpha/decisions"

    def tearDown(self) -> None:
        self.servidor.shutdown()
        self.servidor.server_close()
        self._hilo.join(timeout=5)
        super().tearDown()


def _cache_temporal() -> rj.CacheRespuestasJev:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    Path(tmp.name).unlink()  # que no exista: la caché arranca vacía
    return rj.CacheRespuestasJev(Path(tmp.name))


def _dormir_falso(_segundos: float) -> None:
    pass  # no dormir de verdad en las pruebas de reintento


# ---------------------------------------------------------------------------
# tareas admitidas / prohibidas
# ---------------------------------------------------------------------------

class TareasAdmitidasTest(unittest.TestCase):
    def test_solo_a_y_c(self):
        self.assertEqual(rj.TAREAS_ADMITIDAS_JEV, ("A", "C"))

    def test_b_prohibida_con_motivo_clinico(self):
        self.assertIn("B", rj.TAREAS_PROHIBIDAS_JEV)
        self.assertIn("clínic", rj.TAREAS_PROHIBIDAS_JEV["B"])

    def test_verificar_tarea_permitida_acepta_a_y_c(self):
        rj.verificar_tarea_permitida("A")  # no debe lanzar
        rj.verificar_tarea_permitida("C")  # no debe lanzar

    def test_verificar_tarea_permitida_b_se_niega_con_motivo(self):
        with self.assertRaises(SystemExit) as cm:
            rj.verificar_tarea_permitida("B")
        self.assertIn("clínic", str(cm.exception))
        self.assertIn("PROHIBIDA", str(cm.exception))

    def test_verificar_tarea_permitida_tarea_desconocida_se_niega(self):
        with self.assertRaises(SystemExit):
            rj.verificar_tarea_permitida("D")

    def test_medidor_evaluar_tarea_jev_niega_b_antes_de_leer_datos(self):
        """`rj.verificar_tarea_permitida` es la PRIMERA línea de
        `evaluar_tarea_jev`: se niega sin siquiera intentar `mc.leer_tarea`,
        que fallaría con otro motivo (fichero ausente) si se llegara a
        invocar -- comprobado pasando `tokenizador_c1=None, cache=None,
        clave=None`, que reventarían más adelante si el guion no parase
        antes."""
        with self.assertRaises(SystemExit) as cm:
            mj.evaluar_tarea_jev("B", tokenizador_c1=None, cache=None, clave=None)
        self.assertIn("clínic", str(cm.exception))


# ---------------------------------------------------------------------------
# la clave: fichero ausente, variable ausente/vacía, nunca en registros
# ---------------------------------------------------------------------------

class ClaveApiTest(unittest.TestCase):
    def test_fichero_ausente_se_niega(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "no-existe.env"
            with self.assertRaises(SystemExit) as cm:
                rj.cargar_clave_api(ruta)
            self.assertIn(str(ruta), str(cm.exception))
            self.assertIn(rj.VAR_CLAVE, str(cm.exception))

    def test_variable_ausente_se_niega(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "jev.env"
            ruta.write_text("OTRA_VARIABLE=algo\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                rj.cargar_clave_api(ruta)
            self.assertIn(str(ruta), str(cm.exception))
            self.assertIn(rj.VAR_CLAVE, str(cm.exception))

    def test_variable_vacia_se_niega(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "jev.env"
            ruta.write_text(f"{rj.VAR_CLAVE}=\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                rj.cargar_clave_api(ruta)

    def test_variable_presente_se_lee(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "jev.env"
            ruta.write_text(f"{rj.VAR_CLAVE}={CLAVE_SENUELO}\n", encoding="utf-8")
            self.assertEqual(rj.cargar_clave_api(ruta), CLAVE_SENUELO)

    def test_variable_con_comillas_y_comentarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "jev.env"
            ruta.write_text(f"# comentario\n{rj.VAR_CLAVE}=\"{CLAVE_SENUELO}\"\n", encoding="utf-8")
            self.assertEqual(rj.cargar_clave_api(ruta), CLAVE_SENUELO)

    def test_redactar_sustituye_la_clave(self):
        texto = f"error: cabecera Authorization: Bearer {CLAVE_SENUELO} no válida"
        redactado = rj.redactar(texto, CLAVE_SENUELO)
        self.assertNotIn(CLAVE_SENUELO, redactado)
        self.assertIn("REDACTADA", redactado)


# ---------------------------------------------------------------------------
# análisis de la respuesta noul: sí/no/faltante, CONTADO
# ---------------------------------------------------------------------------

class AnalizarRespuestaNoulTest(unittest.TestCase):
    def test_si_umbral(self):
        self.assertEqual(rj.analizar_respuesta_noul({"type": "noul", "noul": 0.5}), 1)
        self.assertEqual(rj.analizar_respuesta_noul({"type": "noul", "noul": 0.99}), 1)

    def test_no_umbral(self):
        self.assertEqual(rj.analizar_respuesta_noul({"type": "noul", "noul": 0.49}), 0)
        self.assertEqual(rj.analizar_respuesta_noul({"type": "noul", "noul": 0.0}), 0)

    def test_ausente_es_faltante(self):
        self.assertIsNone(rj.analizar_respuesta_noul(None))

    def test_type_distinto_es_faltante(self):
        self.assertIsNone(rj.analizar_respuesta_noul({"type": "choice", "choice": "x"}))

    def test_valor_no_numerico_es_faltante(self):
        self.assertIsNone(rj.analizar_respuesta_noul({"type": "noul", "noul": "sí"}))
        self.assertIsNone(rj.analizar_respuesta_noul({"type": "noul"}))

    def test_valor_booleano_es_faltante(self):
        """`bool` es subclase de `int` en Python: sin la comprobación
        explícita, `True`/`False` colarían como 1/0 aunque la API nunca los
        devuelva -- se cuentan como faltante, no como una respuesta válida
        por casualidad de tipos."""
        self.assertIsNone(rj.analizar_respuesta_noul({"type": "noul", "noul": True}))

    def test_valor_fuera_de_rango_es_faltante(self):
        self.assertIsNone(rj.analizar_respuesta_noul({"type": "noul", "noul": 1.5}))
        self.assertIsNone(rj.analizar_respuesta_noul({"type": "noul", "noul": -0.1}))


# ---------------------------------------------------------------------------
# la caché por (row_id, término)
# ---------------------------------------------------------------------------

class CacheRespuestasJevTest(unittest.TestCase):
    def test_obtener_sin_entrada_es_none(self):
        cache = _cache_temporal()
        self.assertIsNone(cache.obtener("f1", "termino"))

    def test_fijar_y_persistir_y_recargar(self):
        cache = _cache_temporal()
        cache.fijar("f1", "termino", {"noul_bruto": {"type": "noul", "noul": 0.9}, "valor": 1})
        cache.persistir()
        recargada = rj.CacheRespuestasJev(cache.ruta)
        self.assertEqual(recargada.obtener("f1", "termino")["valor"], 1)

    def test_claves_distintas_no_se_confunden(self):
        cache = _cache_temporal()
        cache.fijar("f1", "contrato", {"valor": 1})
        cache.fijar("f1", "servicio", {"valor": 0})
        cache.fijar("f2", "contrato", {"valor": 0})
        self.assertEqual(cache.obtener("f1", "contrato")["valor"], 1)
        self.assertEqual(cache.obtener("f1", "servicio")["valor"], 0)
        self.assertEqual(cache.obtener("f2", "contrato")["valor"], 0)

    def test_fichero_corrupto_no_revienta_la_carga(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{esto no es json")
        tmp.close()
        cache = rj.CacheRespuestasJev(Path(tmp.name))  # no debe lanzar
        self.assertIsNone(cache.obtener("f1", "x"))


# ---------------------------------------------------------------------------
# peticiones HTTP: batching, caché evita repetir, B se niega antes de la red
# ---------------------------------------------------------------------------

class ResponderTerminosTest(_ServidorJevFalsoMixin, unittest.TestCase):
    def test_una_peticion_por_fila_con_todos_los_terminos(self):
        """El batching nativo de Jev (LEEME.md): 5 términos, UNA petición
        con 5 preguntas dentro -- no 5 peticiones."""
        cache = _cache_temporal()
        terminos = ["a", "b", "c", "d", "e"]
        rj.responder_terminos(
            tarea="A", row_id="f1", texto="un texto cualquiera", terminos=terminos,
            texto_pregunta=lambda t: f"¿trata de {t}?", cache=cache, clave=CLAVE_SENUELO,
            base_url=self.base_url, dormir=_dormir_falso,
        )
        self.assertEqual(len(self.servidor.peticiones), 1)
        preguntas = self.servidor.peticiones[0]["cuerpo"]["questions"]
        # Identificadores neutros (`q0`…), y cada uno lleva la pregunta de SU término.
        self.assertEqual(sorted(preguntas), [f"q{i}" for i in range(5)])
        self.assertEqual(sorted(p["instructions"] for p in preguntas.values()),
                         sorted(f"¿trata de {t}?" for t in terminos))

    def test_terminos_con_tildes_y_espacios_viajan_con_identificadores_neutros(self):
        """Un término es una palabra del texto, con lo que traiga; la API no documenta qué admite
        como identificador de pregunta. Los identificadores son `q0`, `q1`… y cada respuesta
        vuelve a SU término."""
        self.servidor.pistas_si = ("trata de contratación",)
        cache = _cache_temporal()
        respuestas = rj.responder_terminos(
            tarea="A", row_id="f1", texto="neutro", terminos=["contratación", "año fiscal"],
            texto_pregunta=lambda t: f"¿trata de {t}?", cache=cache, clave=CLAVE_SENUELO,
            base_url=self.base_url, dormir=_dormir_falso,
        )
        ids = list(self.servidor.peticiones[0]["cuerpo"]["questions"])
        self.assertTrue(all(i.isascii() and i[0] == "q" and i[1:].isdigit() for i in ids), ids)
        self.assertEqual(respuestas, {"contratación": 1, "año fiscal": 0})

    def test_cabecera_authorization_lleva_la_clave(self):
        cache = _cache_temporal()
        rj.responder_terminos(
            tarea="A", row_id="f1", texto="x", terminos=["a"], texto_pregunta=lambda t: f"¿{t}?",
            cache=cache, clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso,
        )
        self.assertEqual(self.servidor.peticiones[0]["authorization"], f"Bearer {CLAVE_SENUELO}")

    def test_mapeo_noul_a_valor_1_0(self):
        """La pista tiene que estar en la PREGUNTA de un término concreto,
        no en el `state` (que se manda igual para todas las preguntas de la
        fila -- si la pista estuviera en el `state`, las dos preguntas
        saldrían 'sí' y la prueba no distinguiría nada)."""
        self.servidor.pistas_si = ("trata de positivo",)
        cache = _cache_temporal()
        respuestas = rj.responder_terminos(
            tarea="A", row_id="f1", texto="un texto neutro sin pistas", terminos=["positivo", "negativo"],
            texto_pregunta=lambda t: f"¿trata de {t}?", cache=cache, clave=CLAVE_SENUELO,
            base_url=self.base_url, dormir=_dormir_falso,
        )
        self.assertEqual(respuestas["positivo"], 1)
        self.assertEqual(respuestas["negativo"], 0)

    def test_segunda_llamada_con_la_misma_cache_no_repite_la_peticion(self):
        """Prueba con nombre para el sabotaje 4: recarga la caché de disco
        (como en un relanzamiento real) y comprueba que la SEGUNDA vuelta no
        manda ninguna petición nueva."""
        cache_ruta = _cache_temporal().ruta
        terminos = ["a", "b", "c"]
        cache1 = rj.CacheRespuestasJev(cache_ruta)
        rj.responder_terminos(tarea="A", row_id="f1", texto="x", terminos=terminos,
                              texto_pregunta=lambda t: f"¿{t}?", cache=cache1, clave=CLAVE_SENUELO,
                              base_url=self.base_url, dormir=_dormir_falso)
        n_tras_primera = len(self.servidor.peticiones)
        self.assertGreater(n_tras_primera, 0)

        cache2 = rj.CacheRespuestasJev(cache_ruta)  # "relanzamiento": recargada de disco
        rj.responder_terminos(tarea="A", row_id="f1", texto="x", terminos=terminos,
                              texto_pregunta=lambda t: f"¿{t}?", cache=cache2, clave=CLAVE_SENUELO,
                              base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(len(self.servidor.peticiones), n_tras_primera,
                         "la segunda vuelta no debería haber mandado ninguna petición más")

    def test_termino_parcialmente_en_cache_solo_pregunta_lo_pendiente(self):
        cache = _cache_temporal()
        cache.fijar("f1", "a", {"noul_bruto": {"type": "noul", "noul": 0.9}, "valor": 1})
        cache.persistir()
        rj.responder_terminos(tarea="A", row_id="f1", texto="x", terminos=["a", "b"],
                              texto_pregunta=lambda t: f"¿{t}?", cache=cache, clave=CLAVE_SENUELO,
                              base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(len(self.servidor.peticiones), 1)
        preguntas = self.servidor.peticiones[0]["cuerpo"]["questions"]
        self.assertEqual([p["instructions"] for p in preguntas.values()], ["¿b?"])

    def test_tarea_b_se_niega_sin_tocar_la_red(self):
        cache = _cache_temporal()
        with self.assertRaises(SystemExit):
            rj.responder_terminos(tarea="B", row_id="f1", texto="datos clínicos", terminos=["a"],
                                  texto_pregunta=lambda t: f"¿{t}?", cache=cache, clave=CLAVE_SENUELO,
                                  base_url=self.base_url, dormir=_dormir_falso)
        self.assertEqual(len(self.servidor.peticiones), 0, "no debe haber mandado NINGUNA petición")


# ---------------------------------------------------------------------------
# la clave nunca en ficheros escritos ni en el texto de una excepción
# ---------------------------------------------------------------------------

class ClaveNuncaExpuestaTest(_ServidorJevFalsoMixin, unittest.TestCase):
    def test_clave_no_esta_en_el_fichero_de_cache(self):
        cache = _cache_temporal()
        rj.responder_terminos(tarea="A", row_id="f1", texto="x", terminos=["a", "b"],
                              texto_pregunta=lambda t: f"¿{t}?", cache=cache, clave=CLAVE_SENUELO,
                              base_url=self.base_url, dormir=_dormir_falso)
        contenido = cache.ruta.read_text(encoding="utf-8")
        self.assertNotIn(CLAVE_SENUELO, contenido)

    def test_clave_no_aparece_si_el_servidor_la_devuelve_en_el_cuerpo_de_error(self):
        """El peor caso: un servidor (mal escrito, o un intermediario de
        depuración) que ECOA la cabecera Authorization en el cuerpo de un
        error 400. `_peticion_decisions` tiene que redactarla igualmente
        antes de construir el mensaje de la excepción -- no basta con
        confiar en que el servidor nunca la devuelva."""
        self.servidor.respuestas_programadas.append((
            400,
            json.dumps({"error": {"code": 400, "message": f"bad Authorization: Bearer {CLAVE_SENUELO}"}}),
        ))
        cache = _cache_temporal()
        with self.assertRaises(rj.ErrorJevFatal) as cm:
            rj.responder_terminos(tarea="A", row_id="f1", texto="x", terminos=["a"],
                                  texto_pregunta=lambda t: "¿a?", cache=cache, clave=CLAVE_SENUELO,
                                  base_url=self.base_url, dormir=_dormir_falso)
        self.assertNotIn(CLAVE_SENUELO, str(cm.exception))
        self.assertIn("REDACTADA", str(cm.exception))

    def test_clave_no_aparece_en_el_resultado_de_una_tarea_sintetica(self):
        """De punta a punta: el JSON que produciría `evaluar_tarea_jev` para
        una tarea (aquí, solo la parte de preguntas por fila que sí corre
        sin datos reales) no lleva la clave en ningún campo -- ni siquiera
        en `noul_bruto`, que guarda la respuesta CRUDA del servidor, pero
        nunca la petición ni sus cabeceras."""
        cache = _cache_temporal()
        respuestas = rj.responder_terminos(
            tarea="A", row_id="f1", texto="un texto positivo de verdad", terminos=["positivo"],
            texto_pregunta=lambda t: f"¿trata de {t}, clave {CLAVE_SENUELO} no debería colarse?",
            cache=cache, clave=CLAVE_SENUELO, base_url=self.base_url, dormir=_dormir_falso,
        )
        del respuestas
        contenido_cache = json.dumps(cache._datos)
        self.assertNotIn(CLAVE_SENUELO, contenido_cache)


# ---------------------------------------------------------------------------
# reintentos: 429/5xx reintentan con espera; el resto no reintenta
# ---------------------------------------------------------------------------

class ReintentosTest(_ServidorJevFalsoMixin, unittest.TestCase):
    def test_429_se_reintenta_y_termina_en_exito(self):
        self.servidor.respuestas_programadas = [
            (429, json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}})),
            (429, json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}})),
        ]
        esperas: list[float] = []
        datos = rj.preguntar(
            estado="x", preguntas={"a": {"type": "noul", "instructions": "¿a?"}},
            clave=CLAVE_SENUELO, base_url=self.base_url, tope_reintentos=5,
            espera_base_s=0.001, dormir=esperas.append,
        )
        self.assertIn("a", datos["answers"])
        self.assertEqual(len(self.servidor.peticiones), 3)  # 2 fallos + 1 éxito
        self.assertEqual(len(esperas), 2)
        self.assertEqual(esperas, sorted(esperas))  # espera exponencial: no decreciente

    def test_429_agota_el_tope_y_se_rinde(self):
        self.servidor.respuestas_programadas = [
            (429, json.dumps({"error": {"code": 429, "message": "Rate limit exceeded"}}))
            for _ in range(10)
        ]
        esperas: list[float] = []
        with self.assertRaises(rj.ErrorJevReintentable):
            rj.preguntar(
                estado="x", preguntas={"a": {"type": "noul", "instructions": "¿a?"}},
                clave=CLAVE_SENUELO, base_url=self.base_url, tope_reintentos=3,
                espera_base_s=0.001, dormir=esperas.append,
            )
        self.assertEqual(len(self.servidor.peticiones), 4)  # intento inicial + 3 reintentos
        self.assertEqual(len(esperas), 3)

    def test_5xx_tambien_se_reintenta(self):
        self.servidor.respuestas_programadas = [(503, json.dumps({"error": {"code": 503, "message": "unavailable"}}))]
        datos = rj.preguntar(
            estado="x", preguntas={"a": {"type": "noul", "instructions": "¿a?"}},
            clave=CLAVE_SENUELO, base_url=self.base_url, tope_reintentos=3,
            espera_base_s=0.001, dormir=_dormir_falso,
        )
        self.assertIn("a", datos["answers"])
        self.assertEqual(len(self.servidor.peticiones), 2)

    def test_error_fatal_no_reintenta(self):
        self.servidor.respuestas_programadas = [
            (401, json.dumps({"error": {"code": 401, "message": "invalid key"}}))
        ]
        esperas: list[float] = []
        with self.assertRaises(rj.ErrorJevFatal):
            rj.preguntar(
                estado="x", preguntas={"a": {"type": "noul", "instructions": "¿a?"}},
                clave=CLAVE_SENUELO, base_url=self.base_url, tope_reintentos=5,
                espera_base_s=0.001, dormir=esperas.append,
            )
        self.assertEqual(len(self.servidor.peticiones), 1, "un error fatal no debe reintentarse")
        self.assertEqual(esperas, [])


# ---------------------------------------------------------------------------
# --estimar: SIN red, aritmética del batching
# ---------------------------------------------------------------------------

class ErroresDeRedAlLeerTest(unittest.TestCase):
    """29-09: la tarea C se paró entera por UN `TimeoutError` al leer la respuesta. `urlopen`
    solo envuelve en `URLError` los fallos al conectar; los de LEER llegan crudos (`TimeoutError`,
    conexión cortada). Son de red: se reintentan, y agotado el tope se rinden sin la clave."""

    class _Respuesta:
        def __init__(self, datos):
            self._datos = datos

        def read(self):
            return json.dumps(self._datos).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen_que_falla(self, fallos):
        llamadas = []

        def urlopen(peticion, timeout=None):
            llamadas.append(1)
            if len(llamadas) <= fallos:
                raise TimeoutError("The read operation timed out")
            return self._Respuesta({"answers": {"q0": {"type": "noul", "noul": 0.9}}})
        return urlopen, llamadas

    def test_un_tiempo_agotado_al_leer_se_reintenta_y_sigue(self):
        urlopen, llamadas = self._urlopen_que_falla(2)
        with mock.patch.object(rj.urllib.request, "urlopen", urlopen):
            datos = rj.preguntar(estado="x", preguntas={"q0": {"type": "noul", "instructions": "?"}},
                                 clave=CLAVE_SENUELO, base_url="http://127.0.0.1:9", tope_reintentos=3,
                                 dormir=_dormir_falso)
        self.assertEqual(datos["answers"]["q0"]["noul"], 0.9)
        self.assertEqual(len(llamadas), 3)

    def test_si_nunca_contesta_se_rinde_tras_el_tope_sin_la_clave(self):
        urlopen, llamadas = self._urlopen_que_falla(99)
        with mock.patch.object(rj.urllib.request, "urlopen", urlopen):
            with self.assertRaises(rj.ErrorJevReintentable) as ctx:
                rj.preguntar(estado="x", preguntas={"q0": {"type": "noul", "instructions": "?"}},
                             clave=CLAVE_SENUELO, base_url="http://127.0.0.1:9", tope_reintentos=2,
                             dormir=_dormir_falso)
        self.assertEqual(len(llamadas), 3)
        self.assertNotIn(CLAVE_SENUELO, str(ctx.exception))
        self.assertIn("TimeoutError", str(ctx.exception))


class EstimarSinRedTest(unittest.TestCase):
    def test_estimar_no_llama_a_peticion_decisions(self):
        """Chokepoint único: TODA petición de red de este módulo pasa por
        `rj._peticion_decisions`. Se sustituye por una función que revienta
        si se la llama -- una prueba de `--estimar` con un servidor falso al
        lado no demostraría nada (`estimar()` ni siquiera recibe una
        `base_url`), así que la comprobación real es esta: nadie llama a la
        única puerta de salida a la red."""
        with mock.patch.object(rj, "_peticion_decisions",
                               side_effect=AssertionError("¡--estimar no debe hacer peticiones!")):
            mj.estimar()  # no debe lanzar

    def test_estimar_declara_la_tarea_b_omitida(self):
        resultado = mj.estimar()
        self.assertTrue(resultado["tareas"]["B"]["omitida"])
        self.assertIn("clínic", resultado["tareas"]["B"]["motivo"])

    def test_estimar_una_peticion_por_fila_no_n_preguntas_por_fila(self):
        """El batching: con datos sintéticos para A, `n_peticiones` tiene
        que ser `n_filas`, NO `n_filas * N_PREGUNTAS` -- justo la ventaja
        frente al respondedor local de (3), que sí necesita una llamada por
        pregunta."""
        filas = [
            {"row_id": f"f{i}", "particion": "train", "target": "si" if i % 2 else "no",
             "texto": "contrato de suministro " * 5, "otra": "x"}
            for i in range(10)
        ]
        particiones = {"train": filas[:6], "dev": filas[6:8], "test": filas[8:]}
        original = mc.leer_tarea
        try:
            mc.leer_tarea = lambda nombre: particiones if nombre in ("A", "C") else original(nombre)
            resultado = mj.estimar()
        finally:
            mc.leer_tarea = original
        self.assertEqual(resultado["tareas"]["A"]["n_filas"], 10)
        self.assertEqual(resultado["tareas"]["A"]["n_peticiones"], 10)  # no 50
        self.assertGreater(resultado["tareas"]["A"]["coste_usd_estimado"], 0.0)
        self.assertGreater(resultado["total_tiempo_estimado_s"], 0.0)


# ---------------------------------------------------------------------------
# tarea C: selección de términos sin decidir por defecto, sin fuga
# ---------------------------------------------------------------------------

def _tarea_regresion_con_marcador_solo_en_test(n_train: int = 40, n_test: int = 20):
    filas = []
    for i in range(n_train):
        alto = i % 2 == 0
        texto = ("importe elevado contrato mayor cuantia" if alto
                else "importe reducido contrato menor cuantia")
        filas.append({"row_id": f"tr{i}", "particion": "train",
                      "target": "6.0" if alto else "3.0", "texto": texto, "otra": "x"})
    for i in range(6):
        alto = i % 2 == 0
        filas.append({"row_id": f"dv{i}", "particion": "dev",
                      "target": "6.0" if alto else "3.0",
                      "texto": "texto de desarrollo neutro", "otra": "x"})
    for i in range(n_test):
        alto = i % 2 == 0
        texto = "exclusivotest exclusivotest aparece aqui" if alto else "otro contenido cualquiera"
        filas.append({"row_id": f"te{i}", "particion": "test",
                      "target": "6.0" if alto else "3.0", "texto": texto, "otra": "x"})
    return filas


class SeleccionTerminosTareaCTest(unittest.TestCase):
    def setUp(self):
        self.filas = _tarea_regresion_con_marcador_solo_en_test()
        self.train = [f for f in self.filas if f["particion"] == "train"]

    def test_seleccionar_terminos_sin_opcion_se_niega(self):
        cfg = {"tipo": "regression"}
        with self.assertRaises(SystemExit) as cm:
            mj._seleccionar_terminos("C", cfg, self.train, seleccion_terminos_c=None)
        self.assertIn("--seleccion-terminos-c", str(cm.exception))

    def test_chi2_quintiles_no_ve_el_marcador_de_test(self):
        textos_train = [f["texto"] for f in self.train]
        target_train = [f["target"] for f in self.train]
        terminos = mj._terminos_top_chi2_quintiles(textos_train, target_train)
        self.assertEqual(len(terminos), 5)
        nombres = {t["termino"] for t in terminos}
        self.assertNotIn("exclusivotest", nombres)

    def test_f_regression_no_ve_el_marcador_de_test(self):
        textos_train = [f["texto"] for f in self.train]
        target_train = [f["target"] for f in self.train]
        terminos = mj._terminos_top_f_regression(textos_train, target_train)
        self.assertEqual(len(terminos), 5)
        nombres = {t["termino"] for t in terminos}
        self.assertNotIn("exclusivotest", nombres)

    def test_seleccionar_terminos_despacha_segun_la_opcion(self):
        cfg = {"tipo": "regression"}
        textos_train = [f["texto"] for f in self.train]
        target_train = [f["target"] for f in self.train]
        esperado_a = mj._terminos_top_chi2_quintiles(textos_train, target_train)
        esperado_b = mj._terminos_top_f_regression(textos_train, target_train)
        obtenido_a = mj._seleccionar_terminos("C", cfg, self.train, seleccion_terminos_c="chi2_quintiles")
        obtenido_b = mj._seleccionar_terminos("C", cfg, self.train, seleccion_terminos_c="f_regression")
        self.assertEqual([t["termino"] for t in obtenido_a], [t["termino"] for t in esperado_a])
        self.assertEqual([t["termino"] for t in obtenido_b], [t["termino"] for t in esperado_b])

    def test_tarea_a_usa_chi2_binario_no_las_opciones_de_c(self):
        """Tarea A (clasificación) no pasa por las opciones de C: reusa
        `mr._terminos_top_chi2` tal cual, sin pedir `--seleccion-terminos-c`."""
        cfg = {"tipo": "binary_classification"}
        train = [{"texto": "contrato urgente", "target": "si"},
                 {"texto": "informe rutinario", "target": "no"}] * 5
        terminos = mj._seleccionar_terminos("A", cfg, train, seleccion_terminos_c=None)
        self.assertEqual(len(terminos), 5)


# ---------------------------------------------------------------------------
# punta a punta: una tarea sintética pequeña con el servidor falso
# ---------------------------------------------------------------------------

def _tarea_sintetica_jev(n: int = 60, seed: int = 1):
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
        filas.append({"row_id": f"s{i}", "particion": particion,
                      "target": "si" if positivo else "no", "texto": texto,
                      "otra": "x" if positivo else "y"})
    return filas


class TareaSinteticaMixin:
    NOMBRE = "_sintetica_jev_test_"

    def setUp(self):
        self.filas = _tarea_sintetica_jev()
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
        self._admitidas_original = rj.TAREAS_ADMITIDAS_JEV
        rj.TAREAS_ADMITIDAS_JEV = rj.TAREAS_ADMITIDAS_JEV + (self.NOMBRE,)

    def tearDown(self):
        mc.TAREAS.pop(self.NOMBRE, None)
        mc.leer_tarea = self._leer_original
        rj.TAREAS_ADMITIDAS_JEV = self._admitidas_original


class PuntaAPuntaJevFalsoTest(_ServidorJevFalsoMixin, TareaSinteticaMixin, unittest.TestCase):
    def test_tarea_pequena_completa_con_servidor_falso(self):
        self.servidor.pistas_si = ("contrato",)
        cache = _cache_temporal()
        tokenizador = mr._cargar_tokenizador_de_c1()

        resultado = mj.evaluar_tarea_jev(
            self.NOMBRE, tokenizador_c1=tokenizador, cache=cache, clave=CLAVE_SENUELO,
            base_url=self.base_url, dormir=_dormir_falso,
        )

        self.assertEqual(resultado["tarea"], self.NOMBRE)
        self.assertEqual(len(resultado["terminos"]), 5)
        cond4 = resultado["condiciones"]["4_jev"]
        self.assertIsInstance(cond4["metrica_puntual"], float)
        self.assertIn("otra", cond4["columnas"])
        self.assertEqual(len([c for c in cond4["columnas"] if c.startswith("pregunta_")]), 5)

        comparacion = resultado["comparaciones"]["4_vs_0"]
        self.assertIn(comparacion["veredicto"], ("mejora", "inferioridad", "equivalencia_practica", "inconcluso"))
        self.assertEqual(comparacion["margen_equivalencia"], mc.MARGEN_EQUIVALENCIA)

        descrita = resultado["comparaciones"]["4_vs_2_diferencia_descrita"]
        self.assertNotIn("veredicto", descrita)

        # el batching: UNA petición por fila, no 5
        n_filas = len(self.filas)
        self.assertEqual(len(self.servidor.peticiones), n_filas)

        n_si = cond4["tiempos"]["preguntas_al_modelo"]["n_si"]
        n_no = cond4["tiempos"]["preguntas_al_modelo"]["n_no"]
        n_faltante = cond4["tiempos"]["preguntas_al_modelo"]["n_faltante"]
        self.assertEqual(n_si + n_no + n_faltante, n_filas * 5)
        self.assertEqual(n_faltante, 0)  # el servidor falso siempre contesta un noul válido


if __name__ == "__main__":
    unittest.main()
