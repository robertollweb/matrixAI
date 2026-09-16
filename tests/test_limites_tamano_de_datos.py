# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El tamaño de los datos dejó de ser una decisión de producto (2026-09-12).

Decisión de Roberto, literal: «el límite de 50 MB es absurdo, estamos en el año
2026, no debe haber ningún tipo de límite, ya el Studio lo contempla en su
configuración». Salió de medir tiempos en Colab: la red densa perdió
`KDDCup09_appetency` (50.000 x 231) con «El CSV ocupa 87.6 MB y el máximo es
50.0 MB» — un tope de producto colándose en un banco de pruebas.

Este fichero prueba LAS DOS MITADES, porque una sola la pasaría una versión que
borra la comprobación entera:

  1. Lo que SE QUITÓ — `max_csv_bytes`/`max_rows` ya no topan por omisión, y el
     banco de pruebas (`limits.sin_topes()`) no hereda ningún tope.
  2. Lo que SIGUE PROTEGIENDO — el servicio compartido (`MATRIXAI_HOSTED=1`)
     conserva los topes de siempre, un tope puesto a mano se sigue aplicando, y
     la guarda de memoria rechaza ANTES de leer lo que no cabe en la máquina.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from matrixai import limits as _limits
from matrixai.training.dataset_analysis import DatasetAnalysisError, analyze_dataset_csv

_LIMPIO = {"MATRIXAI_HOSTED": "0", "MATRIXAI_LIMITS_PROFILE": "equilibrado",
           "MATRIXAI_MAX_CSV_BYTES": "", "MATRIXAI_MAX_ROWS": "",
           "MATRIXAI_GUARDIA_MEMORIA": "1"}


def _env(**extra):
    """Entorno sin overrides heredados (cadena vacía = no parsea = cae al perfil)."""
    return patch.dict(os.environ, {**_LIMPIO, **extra}, clear=False)


def _csv(filas: int, columnas: int = 3) -> str:
    cabecera = ",".join(f"c{i}" for i in range(columnas)) + ",objetivo\n"
    fila = ",".join("1.0" for _ in range(columnas))
    return cabecera + "".join(f"{fila},{'a' if i % 2 else 'b'}\n" for i in range(filas))


class TamanoDeDatosSinTopeTest(unittest.TestCase):
    """Mitad 1: lo que se quitó."""

    def test_el_csv_ya_no_tiene_tope_por_omision(self):
        with _env():
            self.assertIsNone(_limits.get_limit("max_csv_bytes"))
            # El tamaño que mató a KDDCup09_appetency (87,6 MB en Colab; 90,7 MB
            # medidos aquí con la preparación real) ya no supera nada.
            self.assertFalse(_limits.exceeds(90_738_266, "max_csv_bytes"))

    def test_las_filas_ya_no_se_recortan_en_el_analisis(self):
        """Analizar 60.000 filas usaba solo las 50.000 primeras y lo decía en un
        aviso — media verdad sobre un dataset que estaba entero."""
        with _env():
            resultado = analyze_dataset_csv(_csv(60_000, columnas=2))
        self.assertEqual(resultado["rows_total"], 60_000)
        self.assertEqual(resultado["rows_analyzed"], 60_000)

    def test_el_banco_de_pruebas_no_hereda_ningun_tope(self):
        with _env(MATRIXAI_MAX_CSV_BYTES="10", MATRIXAI_MAX_ROWS="5",
                  MATRIXAI_MAX_EPOCHS="3", MATRIXAI_MAX_PARAMS="100"):
            # Fuera del bloque, los topes puestos a mano MANDAN (si no, este
            # test no probaría nada: ver `test_un_tope_puesto_a_mano_se_aplica`).
            self.assertEqual(_limits.get_limit("max_csv_bytes"), 10)
            with _limits.sin_topes():
                self.assertTrue(_limits.en_banco_de_pruebas())
                for clave in ("max_csv_bytes", "max_rows", "max_epochs", "max_params",
                              "max_depth", "max_labels", "max_sequence_length"):
                    self.assertIsNone(_limits.get_limit(clave), clave)
            # Y al salir, todo vuelve a estar donde estaba.
            self.assertFalse(_limits.en_banco_de_pruebas())
            self.assertEqual(_limits.get_limit("max_csv_bytes"), 10)

    def test_el_bloque_se_deshace_aunque_falle_lo_de_dentro(self):
        with _env(MATRIXAI_MAX_ROWS="7"):
            with self.assertRaises(ValueError):
                with _limits.sin_topes():
                    raise ValueError("lo que sea")
            self.assertEqual(_limits.get_limit("max_rows"), 7)

    def test_el_analisis_pasa_dentro_del_banco_con_un_tope_ridiculo(self):
        """El caso exacto del motor `matrixai.dense.torch_cpu`: el core rechaza
        por tope, y el mismo CSV entra dentro de `sin_topes()`."""
        texto = _csv(50, columnas=4)
        with _env(MATRIXAI_MAX_CSV_BYTES="100"):
            with self.assertRaises(DatasetAnalysisError):
                analyze_dataset_csv(texto)
            with _limits.sin_topes():
                self.assertTrue(analyze_dataset_csv(texto)["ok"])


class LoQueSigueProtegidoTest(unittest.TestCase):
    """Mitad 2: sin esto, la reparación la pasaría un `git rm` de la comprobación."""

    def test_el_servicio_compartido_conserva_sus_topes(self):
        """matrixaistudio.org corre con `MATRIXAI_HOSTED=1` (scripts/demo-backend.sh):
        ahí la máquina NO es de quien sube el fichero."""
        with _env(MATRIXAI_HOSTED="1"):
            self.assertEqual(_limits.get_limit("max_csv_bytes"), 50_000_000)
            self.assertEqual(_limits.get_limit("max_rows"), 50_000)
            self.assertTrue(_limits.exceeds(90_738_266, "max_csv_bytes"))

    def test_el_banco_de_pruebas_no_afloja_el_servicio_compartido(self):
        with _env(MATRIXAI_HOSTED="1"):
            with _limits.sin_topes():
                self.assertEqual(_limits.get_limit("max_csv_bytes"), 50_000_000)
                self.assertEqual(_limits.get_limit("max_rows"), 50_000)

    def test_un_tope_puesto_a_mano_se_aplica(self):
        with _env(MATRIXAI_MAX_CSV_BYTES="1000"):
            self.assertEqual(_limits.get_limit("max_csv_bytes"), 1_000)
            self.assertTrue(_limits.exceeds(1_001, "max_csv_bytes"))
            with self.assertRaises(DatasetAnalysisError) as ctx:
                analyze_dataset_csv(_csv(200, columnas=5))
            self.assertEqual(ctx.exception.details["error_kind"], "limit_exceeded")

    def test_el_aviso_dice_quien_puso_el_tope(self):
        """«Arreglar algo puede volver FALSO un aviso»: quitado el techo del
        perfil, la única forma de toparse con `max_csv_bytes` es haberlo puesto
        a mano — y un override GANA al perfil, así que mandar a cambiar de
        perfil sería mandar a hacer algo que no lo sube."""
        with _env(MATRIXAI_MAX_CSV_BYTES="1000"):
            puesto_a_mano = _limits.limit_error("max_csv_bytes", 2_000)
        self.assertEqual(puesto_a_mano["origen"], "override")
        self.assertIn("MATRIXAI_MAX_CSV_BYTES", puesto_a_mano["error"])
        self.assertNotIn("avanzado", puesto_a_mano["error"])
        # Un tope que SÍ viene del perfil conserva el consejo de siempre.
        with _env():
            del_perfil = _limits.limit_error("max_epochs", 5_000)
        self.assertEqual(del_perfil["origen"], "perfil")
        self.assertIn("avanzado", del_perfil["error"])
        with _env(MATRIXAI_HOSTED="1"):
            del_servicio = _limits.limit_error("max_rows", 60_000)
        self.assertEqual(del_servicio["origen"], "hosted")
        self.assertNotIn("Ajustes", del_servicio["error"])

    def test_lo_que_cuesta_calcular_sigue_topado(self):
        """Quitar el techo de los DATOS no quita el de las épocas/capas/parámetros:
        un cero de más ahí convierte un clic en horas de CPU."""
        with _env():
            self.assertEqual(_limits.get_limit("max_epochs"), 1_000)
            self.assertEqual(_limits.get_limit("max_depth"), 12)
            self.assertEqual(_limits.get_limit("max_params"), 2_000_000)


class GuardaDeMemoriaTest(unittest.TestCase):
    """Un tope que desaparece sin que nadie lo note es tan malo como uno absurdo."""

    def test_rechaza_lo_que_no_cabe_y_lo_dice_con_numeros(self):
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=1_000_000_000):
            payload = _limits.memoria_insuficiente(500_000_000)  # x12 = 6 GB > 1 GB
        self.assertIsNotNone(payload)
        self.assertEqual(payload["error_kind"], "memoria_insuficiente")
        self.assertEqual(payload["disponible"], 1_000_000_000)
        self.assertEqual(payload["necesario_estimado"],
                         500_000_000 * _limits.FACTOR_RAM_POR_BYTE_DE_CSV)
        self.assertIn("500.0 MB", payload["error"])
        self.assertIn("1.0 GB", payload["error"])

    def test_no_ofrece_una_accion_que_no_arregla_nada(self):
        """Subir el perfil no añade memoria: decir «cámbialo en Ajustes» aquí
        sería la media verdad tranquilizadora de siempre."""
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=1_000):
            payload = _limits.memoria_insuficiente(1_000_000)
        self.assertFalse(payload["configurable"])
        self.assertNotIn("Ajustes", payload["error"])
        self.assertNotIn("Settings", payload["error_en"])

    def test_deja_pasar_lo_que_si_cabe(self):
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=8_000_000_000):
            self.assertIsNone(_limits.memoria_insuficiente(90_738_266))  # el CSV de KDD

    def test_si_no_se_puede_medir_la_memoria_no_opina(self):
        """Fail-open, como `onnx_size_limit_error`: un guardián que no puede
        medir no debe bloquear (macOS/Windows no tienen /proc/meminfo)."""
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=None):
            self.assertIsNone(_limits.memoria_insuficiente(10 ** 12))

    def test_se_puede_apagar(self):
        with _env(MATRIXAI_GUARDIA_MEMORIA="0"), patch.object(
                _limits, "_memoria_disponible_del_sistema", return_value=1_000):
            self.assertFalse(_limits.guarda_de_memoria_activa())
            self.assertIsNone(_limits.memoria_insuficiente(10 ** 9))

    def test_el_analisis_avisa_antes_de_leer_lo_que_no_cabe(self):
        """El mensaje tiene que llegar ANTES: si se lee primero, no hay mensaje,
        hay un proceso muerto por el kernel."""
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=1_000):
            with self.assertRaises(DatasetAnalysisError) as ctx:
                analyze_dataset_csv(_csv(100, columnas=3))
        self.assertEqual(ctx.exception.details["error_kind"], "memoria_insuficiente")
        self.assertTrue(_limits.es_error_de_memoria(ctx.exception.details))

    def test_el_banco_de_pruebas_tampoco_la_apaga(self):
        """`sin_topes()` quita los topes de PRODUCTO; que un CSV no quepa en la
        RAM no es una política, es física."""
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=1_000):
            with _limits.sin_topes():
                self.assertIsNotNone(_limits.memoria_insuficiente(10 ** 9))

    def test_la_generacion_sintetica_estima_antes_de_generar(self):
        """`max_rows` ya no recorta: lo que impide que un cero de más se lleve el
        proceso es esta estimación, hecha ANTES de generar ninguna fila."""
        from matrixai.playground import _generate_synthetic_dataset
        mxai = ("PROJECT P\nVECTOR In[2]\n  a: Scalar\n  b: Scalar\nEND\n"
                "NETWORK Net\n  INPUT In\n  LAYER Dense units=4 activation=relu\n"
                "  LAYER Dense units=2 activation=softmax\n"
                "  OUTPUT y: ProbabilityMap[A, B]\nEND\nGRAPH\n  In -> Net\nEND\n")
        mxtrain = ("MODEL P.mxai\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
                   "  INPUT In FROM COLUMNS [a, b]\n  TARGET y: Label[A, B]\n"
                   "  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=16\nEND\n"
                   "LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET y\nEND\n"
                   "OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE Net.*\nEND\n")
        # 20.000 filas y NO 10.000.000, aunque el caso real sea el cero de más:
        # si esta comprobación se rompe, el test tiene que ponerse rojo por un
        # aserto, no generar diez millones de filas en el servidor donde corre
        # la suite. La memoria disponible se finge en 100 KB, así que 20.000
        # filas ya no caben y el caso queda igual de probado.
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=100_000):
            resultado = _generate_synthetic_dataset(mxai, mxtrain, rows=20_000,
                                                    seed=1, mode="random")
        self.assertFalse(resultado["ok"])
        self.assertEqual(resultado["error_kind"], "memoria_insuficiente")
        # Y con memoria de sobra, las mismas 200 filas se generan sin ruido.
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=8_000_000_000):
            ok = _generate_synthetic_dataset(mxai, mxtrain, rows=200, seed=1, mode="random")
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["rows"], 200)


    def test_el_texto_sintetico_tambien_estima_antes_de_generar(self):
        """La rama de TEXTO (SEQUENCE) va por otro camino y también se le quitó
        el tope de filas: una fila suya no son celdas cortas, es un texto de
        hasta `seq.length` caracteres."""
        from matrixai.playground import _generate_synthetic_dataset
        from matrixai.training.transformer_generator import TransformerNetworkGenerator
        gen = TransformerNetworkGenerator().generate(
            "resenas: Text[16]\nOUTPUT clase: ProbabilityMap[NEG, POS]")
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=10_000):
            resultado = _generate_synthetic_dataset(gen.mxai_text, gen.training_text,
                                                    rows=5_000, seed=1, mode="random")
        self.assertFalse(resultado["ok"], resultado)
        self.assertEqual(resultado["error_kind"], "memoria_insuficiente")
        with _env(), patch.object(_limits, "_memoria_disponible_del_sistema",
                                  return_value=8_000_000_000):
            ok = _generate_synthetic_dataset(gen.mxai_text, gen.training_text,
                                             rows=20, seed=1, mode="random")
        self.assertTrue(ok["ok"], ok.get("error"))


class ElTopeDeENTRENAMIENTO_TambienCaeDentroDelBancoTest(unittest.TestCase):
    """EL TOPE DE 300 s QUE `sin_topes()` NO APAGABA — reparado el 2026-09-16.

    `sin_topes()` promete, con estas palabras, que dentro del bloque «TODOS los
    topes de producto valen sin tope». El de entrenamiento se le escapaba:
    `playground._P9_TRAIN_TIMEOUT` sale de una variable de entorno **cacheada en
    el import**, asi que el `setdefault` que el banco hacia para quitarlo llegaba
    SIEMPRE tarde — el modulo ya estaba importado. Dos mecanismos que nadie
    habia unido: una env var cacheada y un `ContextVar`.

    **Lo que costaba**: un ajuste denso EN PROCESO que pasara de 300 s devolvia
    «Entrenamiento supero el limite de 300s», y el banco lo registraba como
    `ajuste_del_core_fallido` — un dataset perdido por un tope de producto, que
    es justo la clase de perdida que `sin_topes()` existe para impedir.

    **Y la pasada de 101-C5 NO esta contaminada, medido antes de tocar nada**:
    cero resultados mencionan el tope y 10 ajustes densos completaron por encima
    de 300 s (maximo 425,1 s), porque el banco los corre en un SUBPROCESO
    aislado donde esta ruta no se usa. El defecto era real y estaba a un cambio
    de camino de morder.
    """

    def test_fuera_del_banco_el_tope_SIGUE_puesto(self):
        """La mitad que impide que esto se lea como «quitar el tope».

        Es un tope de PRODUCTO y sigue valiendo para quien usa el producto. Sin
        este aserto, la reparacion podria haber sido «devolver None siempre» y
        nadie lo notaria hasta que un entrenamiento colgado se comiera el hilo.
        """
        from matrixai.playground import _P9_TRAIN_TIMEOUT, _train_join_timeout

        self.assertFalse(_limits.en_banco_de_pruebas())
        self.assertEqual(_train_join_timeout(), _P9_TRAIN_TIMEOUT)
        self.assertGreater(_train_join_timeout(), 0)

    def test_dentro_del_banco_NO_hay_tope_de_entrenamiento(self):
        """El defecto que esta prueba existe para impedir."""
        from matrixai.playground import _train_join_timeout

        with _limits.sin_topes():
            self.assertIsNone(
                _train_join_timeout(),
                "un ajuste del banco que pase de 300 s vuelve a morir por un tope "
                "de producto, y el banco lo cuenta como dataset perdido")

    def test_el_bloque_no_se_escapa(self):
        """Mismo criterio que el resto del fichero: lo que vale dentro del
        `with` no puede seguir valiendo fuera, ni al reves."""
        from matrixai.playground import _train_join_timeout

        antes = _train_join_timeout()
        with _limits.sin_topes():
            pass
        self.assertEqual(_train_join_timeout(), antes)

    def test_lo_decide_el_CONTEXTO_y_no_la_variable_de_entorno(self):
        """La raiz del defecto, fijada por su nombre.

        `_P9_TRAIN_TIMEOUT` se congela en el import: cambiar la env var DESPUES
        no mueve nada, y por eso el `setdefault` del banco no servia. La
        reparacion no puede depender de la env var o vuelve el mismo problema.
        """
        import os
        from matrixai import playground

        previo = os.environ.get("MATRIXAI_TRAIN_TIMEOUT")
        os.environ["MATRIXAI_TRAIN_TIMEOUT"] = "0"
        try:
            self.assertEqual(playground._train_join_timeout(),
                             playground._P9_TRAIN_TIMEOUT,
                             "la env var de despues del import NO debe influir: si "
                             "influyera, el arreglo estaria colgando del mecanismo "
                             "que ya habia fallado")
            with _limits.sin_topes():
                self.assertIsNone(playground._train_join_timeout())
        finally:
            if previo is None:
                os.environ.pop("MATRIXAI_TRAIN_TIMEOUT", None)
            else:
                os.environ["MATRIXAI_TRAIN_TIMEOUT"] = previo


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
