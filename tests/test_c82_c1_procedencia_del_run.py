# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 82 — el RUN guarda de qué está hecho (`run_provenance`).

Hasta hoy el paquete exportado se armaba con lo que mandaba la PANTALLA en el
momento de exportar. Medido con sondas: cambiando BATCH, EPOCHS, receta, filas,
digest y semilla, el manifiesto se declaraba reproducible igual y esos valores
viajaban dentro — describiendo otro dataset y otro contrato que los pesos que
llevaba.

Aquí se fija la captura autoritativa, compuesta en la frontera del
entrenamiento. Lo que estos tests defienden:

* **El digest del CSV CRUDO existe y NO es el de siempre.** `trained_csv_sha256`
  se calcula DESPUÉS de normalizar (`_normalize_external_csv` +
  `_normalize_csv_with_ranges`), así que es el hash del fichero PREPARADO, no
  del que alguien regeneraría con la receta. Medido el 2026-08-19 con el CSV
  Kelvin: crudo `30e16e64cad4b8ab…`, preparado `bae119faf9293b7b…`, y la
  primera fila pasa de `0,273.15` a `0.0,0.083333`. Los dos se guardan: el
  crudo es contra el que se compara un dataset regenerado; el preparado es el
  que ata estos pesos.
* **Lo que no llega se queda en `null`.** Una semilla o una receta inventadas
  convierten un paquete irreproducible en uno que PARECE reproducible y falla
  al comprobarlo.
* **El motor y la máquina salen del resultado REAL**, no de una predicción
  hecha al enviar el job.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import unittest
from email.message import Message as HTTPMessage
from unittest.mock import patch

from matrixai.playground import (
    _contar_filas_csv,
    _get_job_status,
    _handler_class,
    _submit_training_job,
    _training_jobs,
)
from matrixai.training.dataset_project import generate_project_from_dataset


def _kelvin_csv(n: int = 100) -> str:
    """El caso real de Roberto: centigrados -> Kelvin (y = x + 273.15).

    Sirve aquí porque su target vive en escala de dominio (273-372), así que
    `_normalize_csv_with_ranges` lo reescribe de verdad — que es justo la
    condición bajo la que crudo y preparado tienen que separarse.
    """
    filas = "".join(f"{c},{c + 273.15}\n" for c in range(n))
    return "centigrados,prediccionKelvin\n" + filas


def _proyecto_kelvin() -> dict:
    return generate_project_from_dataset(
        _kelvin_csv(), "prediccionKelvin",
        column_type_overrides={"centigrados": "number"},
        column_range_overrides={"centigrados": (0.0, 99.0)})


def _entrenar(proj: dict, **kwargs) -> dict:
    """Envía el job y espera a que termine; devuelve el JOB (no el status).

    Se espera de verdad —y no se deja corriendo— porque el core impone un solo
    entrenamiento simultáneo: un job vivo haría fallar al test siguiente por un
    motivo que no es el suyo.
    """
    sub = _submit_training_job(
        proj["mxai"], proj["training_text"], proj["csv_text"],
        epochs_override=2,
        field_ranges=proj.get("field_ranges"),
        target_range=proj.get("target_range"),
        **kwargs)
    assert sub.get("ok"), sub.get("error")
    job_id = sub["job_id"]
    for _ in range(300):
        estado = _get_job_status(job_id)
        if estado["status"] in ("done", "error", "cancelled", "timeout"):
            break
        time.sleep(0.2)
    return _training_jobs[job_id]


class TestElCrudoNoEsElPreparado(unittest.TestCase):
    """La pieza que no existía: el digest del CSV tal como entra."""

    def test_el_crudo_y_el_preparado_son_distintos_de_verdad(self) -> None:
        """No se afirma la diferencia: se MIDE, sobre el mismo CSV.

        Si algún día la normalización dejara el fichero intacto, este test
        avisaría de que guardar los dos digests ya no distingue nada — en vez
        de seguir prometiendo una distinción que no existe.
        """
        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        prov = job["run_provenance"]

        # La normalización cambió el fichero: esa es la premisa del contrato.
        self.assertNotEqual(
            prov["dataset_sha256_raw"], prov["dataset_sha256_prepared"],
            "crudo y preparado salieron iguales: la normalización no tocó el "
            "CSV, así que este caso ya no prueba lo que dice probar")

        # Y cada uno es de quien dice ser.
        self.assertEqual(
            prov["dataset_sha256_raw"],
            hashlib.sha256(proj["csv_text"].encode("utf-8")).hexdigest(),
            "el digest CRUDO tiene que ser el del CSV que ENTRA, sin tocar")
        self.assertEqual(
            prov["dataset_sha256_prepared"], job["trained_csv_sha256"],
            "el PREPARADO es el que ya existía: el del fichero con el que se "
            "entrenó de verdad")

    def test_el_preparado_solo_no_bastaba(self) -> None:
        """Por qué hacía falta el crudo: quien regenerase el dataset obtendría
        el CRUDO, y compararlo contra el preparado daría 'no coincide' sobre
        unos datos que en realidad SON los mismos."""
        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        regenerado = hashlib.sha256(proj["csv_text"].encode("utf-8")).hexdigest()
        self.assertNotEqual(regenerado, job["trained_csv_sha256"])
        self.assertEqual(regenerado, job["run_provenance"]["dataset_sha256_raw"])

    def test_las_filas_son_las_del_csv_crudo(self) -> None:
        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        self.assertEqual(job["run_provenance"]["dataset_rows"], 100)


class TestAusenteEsNull(unittest.TestCase):
    """Un valor que no llega NO se rellena con uno razonable."""

    def test_un_run_sin_receta_deja_los_campos_de_receta_en_null(self) -> None:
        """El caso del hospital: datos reales, ninguna receta que compartir.

        Fabricar una haría que el paquete prometiera un reentrenamiento que no
        puede cumplir.
        """
        job = _entrenar(_proyecto_kelvin())
        prov = job["run_provenance"]
        self.assertIsNone(prov["recipe_text"])
        self.assertIsNone(prov["recipe_sha256"])
        # Y las claves ESTÁN: `null` es una respuesta, no una ausencia.
        self.assertIn("recipe_text", prov)
        self.assertIn("recipe_sha256", prov)

    def test_una_receta_en_blanco_no_cuenta_como_receta(self) -> None:
        """`""` no es una receta; tratarla como tal daría un `recipe_sha256`
        del vacío, que pasaría por un digest legítimo."""
        job = _entrenar(_proyecto_kelvin(), recipe_text="   \n  ")
        self.assertIsNone(job["run_provenance"]["recipe_text"])
        self.assertIsNone(job["run_provenance"]["recipe_sha256"])

    def test_sin_semilla_de_datos_ni_versiones_se_quedan_en_null(self) -> None:
        """El entrenador no las conoce; si nadie se las dice, no las inventa."""
        prov = _entrenar(_proyecto_kelvin())["run_provenance"]
        self.assertIsNone(prov["seeds"]["dataset"])
        self.assertIsNone(prov["generator_version"])
        self.assertIsNone(prov["csv_serialization_version"])

    def test_lo_que_si_llega_se_guarda_tal_cual(self) -> None:
        """El otro lado del mismo criterio: si se declara, viaja intacto."""
        prov = _entrenar(
            _proyecto_kelvin(),
            recipe_text="RECETA demo v1\n", dataset_seed=7,
            generator_version="matrixai.synthetic.v1",
            csv_serialization_version="matrixai.csv.v1")["run_provenance"]
        self.assertEqual(prov["recipe_text"], "RECETA demo v1\n")
        self.assertEqual(prov["seeds"]["dataset"], 7)
        self.assertEqual(prov["generator_version"], "matrixai.synthetic.v1")
        self.assertEqual(prov["csv_serialization_version"], "matrixai.csv.v1")


class TestLaCapturaSeVerificaSola(unittest.TestCase):
    def test_el_digest_del_mxtrain_cubre_el_texto_que_viaja(self) -> None:
        """`mxtrain_text` va entero y su digest es el de ESE texto: quien lo
        escriba a un fichero obtiene el mismo sha256 sin tener que saber cómo
        lo serializamos aquí."""
        proj = _proyecto_kelvin()
        prov = _entrenar(proj)["run_provenance"]
        self.assertEqual(prov["mxtrain_text"], proj["training_text"])
        self.assertEqual(
            prov["mxtrain_sha256"],
            hashlib.sha256(prov["mxtrain_text"].encode("utf-8")).hexdigest())

    def test_el_digest_de_la_receta_cubre_la_receta_que_viaja(self) -> None:
        prov = _entrenar(_proyecto_kelvin(), recipe_text="R\n")["run_provenance"]
        self.assertEqual(
            prov["recipe_sha256"],
            hashlib.sha256(prov["recipe_text"].encode("utf-8")).hexdigest())

    def test_el_mxai_usa_LA_MISMA_regla_que_el_digest_que_ya_existia(self) -> None:
        """Un solo cálculo, una sola regla. Dos sitios hasheando «el .mxai»
        —uno con `.strip()` y otro sin él— harían que la procedencia se
        contradijera a sí misma."""
        job = _entrenar(_proyecto_kelvin())
        self.assertEqual(
            job["run_provenance"]["mxai_sha256"], job["trained_mxai_sha256"])

    def test_la_semilla_del_split_sale_del_contrato_parseado(self) -> None:
        """Y coincide con lo que el `.mxtrain` declara (el generador emite
        `SPLIT train=0.8 validation=0.2 seed=42`)."""
        proj = _proyecto_kelvin()
        prov = _entrenar(proj)["run_provenance"]
        self.assertIn(f"seed={prov['seeds']['split']}", proj["training_text"])

    def test_declara_su_version_de_esquema(self) -> None:
        prov = _entrenar(_proyecto_kelvin())["run_provenance"]
        self.assertEqual(prov["schema_version"], "1.0")


class TestElMotorYLaMaquinaSonLosDeVerdad(unittest.TestCase):
    def test_se_anotan_desde_el_resultado_del_run(self) -> None:
        """No se predicen al enviar el job: los tres caminos de entrenamiento
        eligen backend por su cuenta y composite puede caer a stdlib si torch
        falla, así que solo el resultado sabe qué entrenó de verdad."""
        job = _entrenar(_proyecto_kelvin())
        self.assertEqual(job["status"], "done", job.get("error"))
        prov, resultado = job["run_provenance"], job["result"]
        self.assertIsNotNone(prov["backend"])
        self.assertEqual(prov["backend"], resultado.get("backend"))
        self.assertEqual(prov["device"], resultado.get("device"))


class TestContarFilas(unittest.TestCase):
    """El recuento es de REGISTROS, no de líneas."""

    def test_un_campo_entrecomillado_con_saltos_de_linea_es_UNA_fila(self) -> None:
        """Los modelos de texto entrenan con reseñas, y una reseña lleva saltos
        de línea dentro. Con `splitlines()` esa fila contaría como tres y el
        paquete declararía un dataset que nadie puede regenerar."""
        csv_text = 'texto,y\n"linea uno\nlinea dos\nlinea tres",pos\nsuelta,neg\n'
        self.assertEqual(len(csv_text.splitlines()), 5)  # lo que se veía antes
        self.assertEqual(_contar_filas_csv(csv_text), 2)

    def test_solo_cabecera_son_cero_filas(self) -> None:
        """Cero es aquí una MEDIDA (no hay filas de datos), no un hueco."""
        self.assertEqual(_contar_filas_csv("a,b\n"), 0)

    def test_un_csv_vacio_no_da_filas_negativas(self) -> None:
        self.assertEqual(_contar_filas_csv(""), 0)


class TestElCableadoDelEndpoint(unittest.TestCase):
    """El hueco suele estar en el CABLEADO, no en el API: el core puede aceptar
    la procedencia y el endpoint no pasársela. Aquí se comprueba `/api/train-start`
    de verdad (el `do_POST` real), capturando los kwargs sin entrenar."""

    def _handler(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        handler = object.__new__(_handler_class(None))
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.path = "/api/train-start"
        handler.command = "POST"
        handler.request_version = "HTTP/1.1"
        handler.headers = HTTPMessage()
        handler.headers["Content-Length"] = str(len(body))
        handler.close_connection = True
        return handler

    def _capturar(self, payload: dict) -> dict:
        capturado: dict = {}

        def _falso(*args, **kwargs):
            capturado.update(kwargs)
            return {"ok": True, "job_id": "x"}

        handler = self._handler({"mxai_text": "x", "training_text": "y",
                                 "csv_text": "z", **payload})
        with patch("matrixai.playground._submit_training_job", _falso), \
                patch.object(handler, "_send_json", lambda *a, **k: None):
            handler.do_POST()
        return capturado

    def test_la_procedencia_declarada_llega_al_core(self) -> None:
        capturado = self._capturar({
            "recipe_text": "RECETA v1\n", "dataset_seed": 7,
            "generator_version": "matrixai.synthetic.v1",
            "csv_serialization_version": "matrixai.csv.v1"})
        self.assertEqual(capturado.get("recipe_text"), "RECETA v1\n")
        self.assertEqual(capturado.get("dataset_seed"), 7)
        self.assertEqual(capturado.get("generator_version"), "matrixai.synthetic.v1")
        self.assertEqual(capturado.get("csv_serialization_version"), "matrixai.csv.v1")

    def test_una_peticion_sin_procedencia_la_pasa_como_null(self) -> None:
        """Retrocompat y honestidad a la vez: un cliente viejo no falla, y lo
        que no manda NO se rellena. `int(x or 42)` habría convertido «no me lo
        dijeron» en «la semilla fue 42»."""
        capturado = self._capturar({})
        self.assertIsNone(capturado.get("recipe_text"))
        self.assertIsNone(capturado.get("dataset_seed"))
        self.assertIsNone(capturado.get("generator_version"))
        self.assertIsNone(capturado.get("csv_serialization_version"))

    def test_una_semilla_de_datos_cero_es_cero_y_no_se_pierde(self) -> None:
        """`0` es una semilla legítima y además es FALSY: con `or` se habría
        caído a `None` y el paquete habría perdido la semilla que sí le
        dijeron."""
        self.assertEqual(self._capturar({"dataset_seed": 0}).get("dataset_seed"), 0)


if __name__ == "__main__":
    unittest.main()
