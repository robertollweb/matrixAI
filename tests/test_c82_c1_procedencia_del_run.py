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

import array
import hashlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from email.message import Message as HTTPMessage
from unittest.mock import patch

from matrixai.playground import (
    _compose_normalize_ranges,
    _contar_filas_usadas,
    _huella_de_pesos,
    _normalize_csv_with_ranges,
    _normalize_external_csv,
    _get_job_status,
    _handler_class,
    _submit_training_job,
    _training_jobs,
)
from matrixai.training.data import CSVDataAdapter
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


def _entrenar(proj: dict, csv_text: str | None = None, **kwargs) -> dict:
    """Envía el job y espera a que termine; devuelve el JOB (no el status).

    Se espera de verdad —y no se deja corriendo— porque el core impone un solo
    entrenamiento simultáneo: un job vivo haría fallar al test siguiente por un
    motivo que no es el suyo.

    Por defecto pide 2 épocas (los tests no miden aprendizaje, miden lo que la
    captura declara), pero un caso puede pasar `epochs_override=None` para
    dejar que mande el contrato — que es donde se ve el tope del operador.

    `csv_text` sustituye al del proyecto SIN tocar el resto: los casos de
    filas necesitan el mismo entrenamiento con un CSV distinto.
    """
    kwargs.setdefault("epochs_override", 2)
    sub = _submit_training_job(
        proj["mxai"], proj["training_text"],
        proj["csv_text"] if csv_text is None else csv_text,
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

    def test_las_filas_son_las_que_ENTRENAN(self) -> None:
        """Las 100 filas del dataset, que aquí son también las que entrenan.

        Antes esta prueba se llamaba «las del csv crudo» y medía lo mismo con
        un fichero limpio, donde las dos cuentas coinciden: por eso no veía el
        defecto. Lo que separa una de otra está en
        `TestLasFilasSonLasQueEntrenan`, con líneas en blanco de por medio.
        """
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
        """Contra la CONSTANTE del core, no contra un literal.

        Lo que se mide es que la captura declare su versión —sin ella, quien
        la lea no sabe cómo interpretarla—, no cuál es hoy. Escrito así tras
        el salto a 1.2 (A2), que rompió el literal sin que nada del producto
        estuviera mal. Y se comprueba además que el lector la ACEPTA: es la
        dependencia dura entre quien escribe la captura y quien la empaqueta.
        """
        from matrixai.export.reproduce import _RUN_PROVENANCE_SCHEMA_VERSIONS
        from matrixai.playground import RUN_PROVENANCE_SCHEMA_VERSION
        prov = _entrenar(_proyecto_kelvin())["run_provenance"]
        self.assertEqual(prov["schema_version"], RUN_PROVENANCE_SCHEMA_VERSION)
        self.assertIn(prov["schema_version"], _RUN_PROVENANCE_SCHEMA_VERSIONS)


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
    """El recuento es de REGISTROS QUE ENTRENAN, no de líneas del fichero."""

    def test_un_campo_entrecomillado_con_saltos_de_linea_es_UNA_fila(self) -> None:
        """Los modelos de texto entrenan con reseñas, y una reseña lleva saltos
        de línea dentro. Con `splitlines()` esa fila contaría como tres y el
        paquete declararía un dataset que nadie puede regenerar."""
        csv_text = 'texto,y\n"linea uno\nlinea dos\nlinea tres",pos\nsuelta,neg\n'
        self.assertEqual(len(csv_text.splitlines()), 5)  # lo que se veía antes
        self.assertEqual(_contar_filas_usadas(csv_text), 2)

    def test_solo_cabecera_son_cero_filas(self) -> None:
        """Cero es aquí una MEDIDA (no hay filas de datos), no un hueco."""
        self.assertEqual(_contar_filas_usadas("a,b\n"), 0)

    def test_un_csv_vacio_no_da_filas_negativas(self) -> None:
        self.assertEqual(_contar_filas_usadas(""), 0)

    def test_una_linea_en_blanco_no_es_una_fila_y_lo_dice_el_lector_del_trainer(self) -> None:
        """El contador tiene que contar lo que CUENTA EL ENTRENAMIENTO.

        No se afirma: se compara contra `CSVDataAdapter`, que es el lector con
        el que `training/data.py` carga los ejemplos de verdad. Los cinco casos
        son los que separan una cuenta de la otra —blancos al final,
        intercalados, CRLF, sin salto final y una línea de espacios, que SÍ es
        una fila (y que el entrenamiento rechazará por otro motivo)—.
        """
        casos = {
            "limpio": "a,b\n1,2\n3,4\n",
            "blancos al final": "a,b\n1,2\n3,4\n\n\n\n",
            "blancos en medio": "a,b\n1,2\n\n3,4\n",
            "sin salto final": "a,b\n1,2\n3,4",
            "crlf": "a,b\r\n1,2\r\n\r\n3,4\r\n",
        }
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "datos.csv"
            for nombre, texto in casos.items():
                ruta.write_text(texto, encoding="utf-8")
                cargadas = len(CSVDataAdapter(str(ruta), "Input", ["a"], "b").examples())
                self.assertEqual(
                    _contar_filas_usadas(texto), cargadas,
                    f"[{nombre}] el contador dice otra cosa que el lector del entrenamiento")
                self.assertEqual(_contar_filas_usadas(texto), 2, nombre)

            # UNA LÍNEA DE ESPACIOS SÍ ES UNA FILA, y se cuenta. Medido: el
            # lector del entrenamiento tampoco la salta —revienta al convertir
            # `'   '` a número—, así que descontarla aquí publicaría un
            # recuento de un entrenamiento que no puede ni empezar.
            espacios = "a,b\n1,2\n   \n3,4\n"
            ruta.write_text(espacios, encoding="utf-8")
            with self.assertRaises(ValueError):
                CSVDataAdapter(str(ruta), "Input", ["a"], "b").examples()
            self.assertEqual(_contar_filas_usadas(espacios), 3)


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


# ═══════════════════════════════════════════════════════════════════════════
# 82 · LA CAPTURA DESCRIBE EL ENTRENAMIENTO QUE OCURRIÓ
#
# Lo que la refutación encontró: la captura describía lo que se PIDIÓ. Corrían
# 3 épocas y el paquete decía 50; el CSV se normalizaba con unos rangos que no
# viajaban; se contaban líneas que el entrenamiento descarta; y reanudar desde
# unos pesos guardados daba otros pesos con la MISMA captura byte a byte.
# ═══════════════════════════════════════════════════════════════════════════


class TestLasEpocasQueCorrieron(unittest.TestCase):
    """Las épocas EFECTIVAS, no las que dice el papel."""

    def test_el_tope_del_operador_recorta_y_la_captura_lo_declara(self) -> None:
        """El caso que NO necesita que el cliente pida nada.

        MEDIDO el 2026-08-19 antes de tocar nada: con `MATRIXAI_MAX_EPOCHS=3` y
        un `.mxtrain` que declara `EPOCHS 50`, el run corre 3 épocas y la
        captura guardaba el contrato crudo — el paquete viajaba diciendo
        «EPOCHS 50» y `reproducible: true` sobre unos pesos de 3 épocas. El
        tope lo traen los perfiles de `matrixai.limits`, así que pasa solo.
        """
        proj = _proyecto_kelvin()
        with patch.dict(os.environ, {"MATRIXAI_MAX_EPOCHS": "3"}):
            job = _entrenar(proj, epochs_override=None)
        prov = job["run_provenance"]
        self.assertIn("EPOCHS 50", prov["mxtrain_text"], "lo que dice el papel")
        self.assertEqual(prov["epochs_effective"], 3,
                         "las del spec que se entrena, ya con el tope")
        # Y lo que de verdad corrió, medido contra la traza del propio job.
        self.assertEqual(prov["epochs_ran"], 3)
        self.assertEqual(prov["epochs_ran"], len(job["epochs"]))
        self.assertEqual(job["epochs"][-1]["epoch"], 3)

    def test_el_mxtrain_que_viaja_NO_se_reescribe(self) -> None:
        """Decisión declarada, no descuido: el contrato viaja tal como llegó.

        Reescribirle el `EPOCHS` declararía un contrato que nadie envió y
        rompería el par texto/digest, que existe para verificarse solo contra
        el fichero que el usuario tiene. Y ningún número único serviría: el
        tope recorta, pero el early stop y el Cancelar paran antes incluso de
        esa cifra. Por eso la captura declara los cuatro hechos por separado y
        el `.mxtrain` sigue diciendo lo que decía.
        """
        proj = _proyecto_kelvin()
        with patch.dict(os.environ, {"MATRIXAI_MAX_EPOCHS": "3"}):
            job = _entrenar(proj, epochs_override=None)
        prov = job["run_provenance"]
        self.assertIn("EPOCHS 50", prov["mxtrain_text"])
        self.assertEqual(
            prov["mxtrain_sha256"],
            hashlib.sha256(proj["training_text"].encode("utf-8")).hexdigest(),
            "el digest tiene que seguir siendo el del contrato que se envió")
        # Lo que impide que eso sea una mentira: la captura dice cuántas
        # corrieron de verdad.
        self.assertEqual(prov["epochs_ran"], 3)

    def test_lo_que_pide_el_cliente_tampoco_es_lo_que_dice_el_contrato(self) -> None:
        """El otro camino que separa el papel del run: el override del cliente.

        Sin tope de por medio (`MATRIXAI_MAX_EPOCHS=0`), el `.mxtrain` sigue
        diciendo 50 y el run corre 2 porque quien lanzó el job pidió 2.
        """
        proj = _proyecto_kelvin()
        with patch.dict(os.environ, {"MATRIXAI_MAX_EPOCHS": "0"}):  # 0 = sin tope
            job = _entrenar(proj)  # el ayudante de este fichero pide 2
        prov = job["run_provenance"]
        self.assertIn("EPOCHS 50", prov["mxtrain_text"])
        self.assertEqual(prov["epochs_effective"], 2)
        self.assertEqual(prov["epochs_ran"], 2)

    def test_el_early_stop_para_ANTES_del_numero_efectivo(self) -> None:
        """Por qué `epochs_ran` no se puede deducir de `epochs_effective`.

        MEDIDO el 2026-08-19 con 60 filas de RUIDO (el objetivo no depende de
        la entrada, así que la pérdida de validación deja de mejorar enseguida)
        y `EARLY_STOP patience=1`: el spec se configuró a 50 épocas y corrieron
        20, sin que ningún tope tocara nada. Quien leyera solo el número
        efectivo declararía 50 épocas para unos pesos de 20.

        Se fija el motor a stdlib para que la medida no dependa de si la
        máquina que corre esto tiene torch: lo que se prueba es el registro de
        lo que pasó, no qué backend lo hizo.
        """
        import random
        random.seed(1)
        filas = [(i, random.uniform(273, 373)) for i in range(60)]
        ruido = ("centigrados,prediccionKelvin\n"
                 + "".join(f"{a},{b:.2f}\n" for a, b in filas))
        proj = generate_project_from_dataset(
            ruido, "prediccionKelvin",
            column_type_overrides={"centigrados": "number"},
            column_range_overrides={"centigrados": (0.0, 59.0)})
        proj["training_text"] = proj["training_text"].replace(
            "RUN\n  EPOCHS 50\nEND",
            "RUN\n  EPOCHS 50\n  EARLY_STOP patience=1 metric=validation_loss\nEND")

        with patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "stdlib",
                                     "MATRIXAI_MAX_EPOCHS": "0"}):
            job = _entrenar(proj, epochs_override=None)

        prov = job["run_provenance"]
        self.assertEqual(job["status"], "done")
        self.assertEqual(prov["epochs_effective"], 50)
        self.assertLess(prov["epochs_ran"], prov["epochs_effective"],
                        "el early stop paró antes y la captura tiene que decirlo")
        self.assertEqual(prov["epochs_ran"], len(job["epochs"]))


class TestLasFilasSonLasQueEntrenan(unittest.TestCase):
    """`dataset_rows` cuenta filas de entrenamiento, no líneas de fichero."""

    def test_las_lineas_en_blanco_no_inflan_el_recuento(self) -> None:
        """MEDIDO: el mismo CSV con 5 líneas en blanco al final entrena
        exactamente igual —mismo `sha256_prepared`— y antes declaraba 105 filas
        para unos pesos entrenados con 100. Quien regenerase 105 no obtendría
        el dataset del run: R1 se rompía por un fichero que era el mismo.
        """
        proj = _proyecto_kelvin()
        sucio = proj["csv_text"].rstrip("\n") + "\n" + "\n" * 5
        self.assertEqual(len(sucio.splitlines()) - 1, 105, "premisa del caso")

        job_sucio = _entrenar(proj, csv_text=sucio)
        job_limpio = _entrenar(proj)
        sucia, limpia = job_sucio["run_provenance"], job_limpio["run_provenance"]

        # `dataset_rows` son las filas del dataset que hay que REGENERAR y
        # `dataset_rows_used` las que el entrenamiento consumió: dos preguntas,
        # y con este fichero las dos respuestas son 100, no 105.
        self.assertEqual(sucia["dataset_rows"], 100)
        self.assertEqual(sucia["dataset_rows_used"], 100)
        self.assertEqual(sucia["dataset_rows"], limpia["dataset_rows"])
        self.assertEqual(sucia["dataset_rows_used"], limpia["dataset_rows_used"])
        # La premisa que hace grave el defecto: son el MISMO entrenamiento.
        self.assertEqual(sucia["dataset_sha256_prepared"],
                         limpia["dataset_sha256_prepared"])
        # Y el crudo sí es otro fichero: por eso el recuento no puede salir de él.
        self.assertNotEqual(sucia["dataset_sha256_raw"], limpia["dataset_sha256_raw"])


class TestLosRangosDecidenElPreparado(unittest.TestCase):
    """`field_ranges`/`target_range` entran en la captura: sin ellos, el CSV
    con el que entrenó la red no se puede rehacer."""

    def test_la_captura_lleva_los_rangos_con_los_que_se_normalizo(self) -> None:
        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        prov = job["run_provenance"]
        self.assertEqual(prov["field_ranges"], {"centigrados": [0.0, 99.0]})
        self.assertEqual(prov["target_range"], list(proj["target_range"]))
        # Listas de floats de Python, no tuplas ni numpy: la captura viaja a un
        # JSON y el manifiesto valida la forma `[min, max]`.
        for par in list(prov["field_ranges"].values()) + [prov["target_range"]]:
            self.assertIsInstance(par, list)
            self.assertTrue(all(type(x) is float for x in par), par)

    def test_el_preparado_se_rehace_desde_el_crudo_con_lo_publicado(self) -> None:
        """La prueba de que los dos campos BASTAN: se rehace el fichero con el
        que entrenó la red partiendo solo de lo que el paquete publica (el CSV
        crudo, el `.mxai` y estos dos campos) y el digest casa."""
        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        prov = job["run_provenance"]

        rangos = {c: (v[0], v[1]) for c, v in (prov["field_ranges"] or {}).items()}
        objetivo = tuple(prov["target_range"]) if prov["target_range"] else None
        rehecho, error = _normalize_external_csv(proj["csv_text"])
        self.assertIsNone(error)
        compuestos = _compose_normalize_ranges(proj["mxai"], rangos, objetivo)
        rehecho = _normalize_csv_with_ranges(rehecho, compuestos)
        self.assertEqual(hashlib.sha256(rehecho.encode("utf-8")).hexdigest(),
                         prov["dataset_sha256_prepared"])

    def test_sin_los_rangos_el_preparado_NO_casa(self) -> None:
        """Por qué hacían falta: con el mismo CSV crudo y sin ellos sale otro
        fichero, otra normalización y otros pesos. El paquete publicaba
        `dataset_sha256_prepared` y nadie podía llegar a él."""
        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        prov = job["run_provenance"]
        sin_rangos, _ = _normalize_external_csv(proj["csv_text"])
        self.assertNotEqual(hashlib.sha256(sin_rangos.encode("utf-8")).hexdigest(),
                            prov["dataset_sha256_prepared"])


def _pesos_de(job: dict, proj: dict) -> dict:
    """Un `state_dict` de torch a partir de los pesos que dejó un run.

    Es el mismo camino que usa el producto para reanudar un entrenamiento
    (`endpoints._studio_train_start` carga el `.mxw`/los params del modelo
    guardado y los pasa como `initial_state_dict`), reconstruido aquí desde el
    resultado del job para no depender del backend del Studio.
    """
    from matrixai.parameters.store import ParameterSet
    from matrixai.forward.dense_torch import (
        dense_module_to_state_dict, dense_network_to_torch_module,
    )
    from matrixai.parser import parse_text as _parse
    red = _parse(proj["mxai"]).networks[0]
    ps = ParameterSet.from_dict(job["result"]["params_best"])
    return dense_module_to_state_dict(red, dense_network_to_torch_module(red, ps))


def _hay_torch() -> bool:
    try:
        from matrixai.parameters.tensor_bridge import torch_available
        return bool(torch_available())
    except Exception:  # noqa: BLE001
        return False


class TestDeQuePesosPartio(unittest.TestCase):
    """El warm start es un cuarto insumo del run, y no estaba en la captura.

    MEDIDO con torch el 2026-08-19 (el camino GPU/Colab): desde cero loss
    1.102227, reanudado 1.094253 — y la captura salía IDÉNTICA byte a byte.
    """

    def test_sin_pesos_de_partida_la_captura_lo_dice(self) -> None:
        """«Partió de cero» es una afirmación (`false`), no un hueco (`null`)."""
        job = _entrenar(_proyecto_kelvin())
        self.assertIs(job["run_provenance"]["warm_start"], False)

    def test_unos_pesos_que_el_stdlib_IGNORA_no_se_declaran_usados(self) -> None:
        """La mitad silenciosa del defecto: en stdlib el warm start se ignora.

        Declararlo aplicado por el hecho de haberlo recibido sería la mentira
        del otro lado. Aquí se comprueba con la MEDIDA que lo prueba: el run
        que recibe los pesos da exactamente la misma loss que el que no los
        recibe, porque no los usó.
        """
        proj = _proyecto_kelvin()
        with patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "stdlib"}):
            desde_cero = _entrenar(proj)
            estado = _pesos_de(desde_cero, proj) if _hay_torch() else {
                # Sin torch no hay tensores que ofrecer; el caso que importa
                # aquí es «llegaron unos pesos», y para eso vale un buffer.
                "capa.w": array.array("f", [0.1, 0.2, 0.3]),
            }
            reanudado = _entrenar(proj, initial_state_dict=estado)

        self.assertIs(reanudado["run_provenance"]["warm_start"], False,
                      "llegaron pesos, no se usaron: el run arrancó de cero")
        self.assertEqual(
            reanudado["result"]["final_train_loss"],
            desde_cero["result"]["final_train_loss"],
            "misma loss exacta: la prueba de que el warm start se ignoró")

    @unittest.skipUnless(_hay_torch(), "sin torch no hay camino de warm start")
    def test_torch_los_usa_y_la_captura_lo_declara_con_su_huella(self) -> None:
        proj = _proyecto_kelvin()
        with patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "torch"}):
            desde_cero = _entrenar(proj)
            estado = _pesos_de(desde_cero, proj)
            reanudado = _entrenar(proj, initial_state_dict=estado)

        warm = reanudado["run_provenance"]["warm_start"]
        self.assertIsInstance(warm, dict,
                              "un objeto = arrancó de unos pesos que ya existían")
        self.assertEqual(warm["sha256"], _huella_de_pesos(estado)["sha256"],
                         "y los identifica: quien tenga esos pesos lo recomputa")
        self.assertEqual(warm["tensors"], len(estado))
        self.assertGreater(warm["params"], 0)
        # Y el run de partida, que no partió de nada, sigue diciéndolo.
        self.assertIs(desde_cero["run_provenance"]["warm_start"], False)

    @unittest.skipUnless(_hay_torch(), "sin torch no hay camino de warm start")
    def test_dos_runs_distintos_ya_no_dan_la_MISMA_captura(self) -> None:
        """El defecto, dicho como lo medió la refutación: dos entrenamientos
        que producen pesos distintos daban capturas idénticas byte a byte, así
        que el paquete no podía distinguir cuál de los dos llevaba dentro."""
        proj = _proyecto_kelvin()
        with patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "torch"}):
            desde_cero = _entrenar(proj)
            reanudado = _entrenar(proj, initial_state_dict=_pesos_de(desde_cero, proj))

        def _digest(captura: dict) -> str:
            return hashlib.sha256(
                json.dumps(captura, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

        self.assertNotEqual(_digest(desde_cero["run_provenance"]),
                            _digest(reanudado["run_provenance"]))
        # Todo lo demás del run ES lo mismo: la diferencia está justo donde
        # debe estar, no en un campo cualquiera que la disimule.
        for clave in ("mxai_sha256", "mxtrain_sha256", "dataset_sha256_raw",
                      "dataset_sha256_prepared", "dataset_rows", "seeds"):
            self.assertEqual(desde_cero["run_provenance"][clave],
                             reanudado["run_provenance"][clave], clave)


class TestLaHuellaDeUnosPesos(unittest.TestCase):
    """`_huella_de_pesos`: identificar unos pesos sin copiarlos."""

    def _tensores(self, valores: list[float]) -> dict:
        import numpy as np
        return {"capa.w": np.array(valores, dtype=np.float32)}

    def test_los_mismos_pesos_dan_el_mismo_digest_en_cualquier_orden(self) -> None:
        import numpy as np
        w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([0.5], dtype=np.float32)
        self.assertEqual(_huella_de_pesos({"w": w, "b": b})["sha256"],
                         _huella_de_pesos({"b": b, "w": w})["sha256"])

    def test_cambiar_UN_numero_cambia_el_digest(self) -> None:
        """Si no, la captura no distinguiría dos reentrenamientos seguidos."""
        self.assertNotEqual(_huella_de_pesos(self._tensores([1.0, 2.0, 3.0]))["sha256"],
                            _huella_de_pesos(self._tensores([1.0, 2.0, 3.5]))["sha256"])

    def test_cuenta_los_parametros_y_los_tensores(self) -> None:
        import numpy as np
        huella = _huella_de_pesos({"w": np.zeros((3, 4), dtype=np.float32),
                                   "b": np.zeros(4, dtype=np.float32)})
        self.assertEqual(huella["tensors"], 2)
        self.assertEqual(huella["params"], 16)

    def test_unos_pesos_que_no_se_pueden_leer_dan_null_y_no_medio_digest(self) -> None:
        """Un digest de la mitad de los tensores no casaría con nada y
        parecería una comprobación: `null` dice la verdad («no supe
        identificarlos») y el recuento de tensores sí se conserva."""
        huella = _huella_de_pesos({"w": [1, 2, 3]})
        self.assertIsNone(huella["sha256"])
        self.assertEqual(huella["tensors"], 1)
        self.assertIsNone(huella["params"])

    def test_un_buffer_sin_numero_de_elementos_no_cuenta_cero_parametros(self) -> None:
        """Un valor ausente no es un cero: `array.array` no declara `.size`."""
        huella = _huella_de_pesos({"w": array.array("f", [1.0, 2.0, 3.0])})
        self.assertIsNotNone(huella["sha256"], "los bytes sí se pudieron digerir")
        self.assertIsNone(huella["params"])


class TestLaCapturaLaLeeQuienEmpaqueta(unittest.TestCase):
    """El hueco suele estar en el CABLEADO, no en el API.

    El core puede componer una captura impecable y quien la lee rechazarla
    entera —o quedarse justo con los campos que no son—. Aquí se comprueba con
    la captura DE UN RUN DE VERDAD, no con un diccionario escrito a mano: es el
    único sitio donde se ve si el productor y el lector hablan el mismo idioma.

    Importa especialmente por la VERSIÓN del bloque: `_normalize_run_provenance`
    no degrada, LEVANTA. Si el core sube a "1.1" y el lector solo admite "1.0",
    no es que el paquete pierda un campo: es que no se puede exportar nada.
    """

    def test_un_manifiesto_construido_con_la_captura_del_run(self) -> None:
        import shutil
        import tempfile as _tmp
        from matrixai.export import build_reproduce_manifest

        proj = _proyecto_kelvin()
        job = _entrenar(proj)
        prov = job["run_provenance"]

        bundle = Path(_tmp.mkdtemp())
        self.addCleanup(shutil.rmtree, str(bundle), True)
        (bundle / "model.mxai").write_text(proj["mxai"], encoding="utf-8")
        (bundle / "model.mxtrain").write_text(proj["training_text"], encoding="utf-8")

        manifiesto = build_reproduce_manifest(
            bundle, training_filename="model.mxtrain", run_provenance=prov)

        # Lo que este corte añadió tiene que LLEGAR al paquete; si se queda en
        # la captura, el paquete sigue sin poder demostrar nada.
        generacion = manifiesto["generation"]
        self.assertEqual(generacion["epochs_effective"], prov["epochs_effective"])
        self.assertEqual(generacion["target_range"], prov["target_range"])
        self.assertEqual(generacion["field_ranges"], prov["field_ranges"])
        self.assertEqual(generacion["warm_start"], prov["warm_start"])
        self.assertEqual(manifiesto["artifacts"]["dataset"]["rows"],
                         prov["dataset_rows"])
