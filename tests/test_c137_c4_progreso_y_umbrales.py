# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""137-C4 — el progreso REAL de «¿Funciona aquí?» y los recuentos por umbral del deslizador.

Dos cosas que la pantalla del paso 3 necesita del núcleo:

* `evaluar_puntuacion(..., al_progresar=f)`: `f(hechas, total, fase)` con lo que el núcleo
  CUENTA —remuestras hechas de las que va a hacer—, sin porcentajes ni tiempos inventados, y
  una cancelación limpia (`EvaluacionCancelada`).
* `al_umbral.exploracion`: por puntuación, los cuatro recuentos de la matriz de confusión en
  los vértices de la curva, en las unidades de la columna, para el deslizador del umbral.

Cada prueba lleva el nombre de la regla que sostiene. La referencia independiente de los
recuentos es `sklearn.metrics.confusion_matrix`; el núcleo es stdlib puro y sklearn solo vive
aquí. El total del progreso se contrasta con un contador EXTERNO de remuestras (el observador
de `incertidumbre`, instalado desde fuera), no con el plan que el propio núcleo calcula.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import unittest

import matrixai.estudio.evaluacion_de_puntuacion as ev
from matrixai.estudio.evaluacion_de_puntuacion import (
    FASES_DE_PROGRESO,
    PUNTOS_DE_LA_EXPLORACION,
    VERSION_DEL_INFORME,
    DeclaracionDePuntuacion as _Decl,
    EvaluacionCancelada,
    EvaluacionRechazada,
    evaluar_puntuacion,
)
from matrixai.estudio.evaluacion_de_puntuacion_textos import MOTIVOS, huecos_de
from matrixai.estudio.incertidumbre import (
    intervalo,
    observador_de_remuestras,
    observando_remuestras,
)
from matrixai.estudio.metricas import Muestra
from matrixai.pipelines.canonical import jcs_bytes

try:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    _HAY_SKLEARN = True
except Exception:  # pragma: no cover
    _HAY_SKLEARN = False

REM = 40
ALTO, BAJO = "mas_alto_mas_riesgo", "mas_bajo_mas_riesgo"
R2 = dict(horizonte_del_desenlace="durante el ingreso", momento_de_la_puntuacion="al ingreso",
          donde_se_calcula="local")
RARO = "ZX9-RARO-4471"  # un identificador que NO puede aparecer en el informe


def Decl(columna, tipo, direccion, **kw):
    kw.setdefault("casos_en_su_desarrollo", "no")
    return _Decl(columna, tipo, direccion, **kw)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _filas(n: int = 700, semilla: int = 5, desplazamiento: float = 0.6) -> list[dict]:
    """Riesgo `p` (probabilidad con 4 decimales: hay empates), escala `pts` (0–10, muchos
    empates), `pct` (la misma probabilidad en %), `sexo` y `sala` (la primera, de 12 filas:
    sin soporte). Lo que se escribe en el CSV es lo que se mide: ya redondeado."""
    r = random.Random(semilla)
    filas = []
    for i in range(n):
        z = r.gauss(-0.5, 1.2)
        y = 1 if r.random() < _sig(z) else 0
        p = round(min(0.999, max(0.001, _sig(z + desplazamiento))), 4)
        filas.append({"id": RARO if i == 3 else f"P{i // 2:05d}", "p": p, "pct": round(p * 100, 1),
                      "pts": max(0, min(10, round(1.5 * z + 5 + r.gauss(0, 1)))), "y": y,
                      "sexo": "Mujer" if r.random() < 0.5 else "Hombre",
                      "sala": "Chica" if i < 12 else "Grande"})
    return filas


def _csv(filas, columnas=("id", "p", "pct", "pts", "y", "sexo", "sala"), sin=None):
    """`sin` = {columna: {filas sin valor}}: la fila queda con el campo vacío."""
    sin = sin or {}
    lineas = [",".join(columnas)]
    for i, f in enumerate(filas):
        lineas.append(",".join("" if i in sin.get(c, ()) else str(f[c]) for c in columnas))
    return "\n".join(lineas) + "\n"


def _evaluar(texto, *, puntuaciones, representa=True, **kw):
    kw.setdefault("remuestras", REM)
    for k, v in R2.items():
        kw.setdefault(k, v)
    return evaluar_puntuacion(texto, columna_desenlace="y", valor_del_evento="1",
                              puntuaciones=puntuaciones, muestra_representativa=representa,
                              **kw)


def _filas_c1(n: int = 1500, semilla: int = 1, desplazamiento: float = 0.0) -> list[dict]:
    """COPIA EXACTA de `_filas` de `test_c137_c1_evaluar_puntuacion.py`: los digests de
    «el informe de antes» se fijaron con esos datos y esa semilla."""
    r = random.Random(semilla)
    filas = []
    for i in range(n):
        z = r.gauss(-0.5, 1.2)
        y = 1 if r.random() < _sig(z) else 0
        p = min(0.999, max(0.001, _sig(z + desplazamiento)))
        pts = max(0, min(10, round(1.5 * z + 5 + r.gauss(0, 1))))
        filas.append({"id": f"P{i:05d}", "p": p, "pts": pts, "y": y,
                      "sexo": "Mujer" if r.random() < 0.5 else "Hombre"})
    return filas


def _csv_c1(filas, columnas=("id", "p", "pts", "y", "sexo")):
    """COPIA de `_csv` de ese mismo fichero (los flotantes con `repr(round(v, 6))`)."""
    lineas = [",".join(columnas)]
    for f in filas:
        celdas = []
        for c in columnas:
            v = f[c]
            if isinstance(v, float):
                v = repr(round(v, 6))
            celdas.append(str(v))
        lineas.append(",".join(celdas))
    return "\n".join(lineas) + "\n"


def _configs(filas):
    """Las configuraciones del progreso: cada una pide un conjunto distinto de intervalos."""
    t = _csv(filas)
    return {
        "una_prob_umbral_y_subgrupos": dict(
            texto=t, puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)],
            columnas_de_subgrupo=["sexo", "sala"]),
        "una_prob_sin_umbral": dict(texto=t, puntuaciones=[Decl("p", "probabilidad", ALTO)]),
        "una_escala_baja_no_representa": dict(
            texto=t, puntuaciones=[Decl("pts", "puntuacion", BAJO, umbral=4)], representa=False),
        "dos_puntuaciones_con_todo": dict(
            texto=t, puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.4),
                                   Decl("pts", "puntuacion", ALTO, umbral=5)],
            columnas_de_subgrupo=["sexo"], umbrales_de_decision=[0.2, 0.4]),
        "pct_con_pacientes": dict(
            texto=t, puntuaciones=[Decl("pct", "probabilidad_pct", ALTO, umbral=30.0)],
            columna_de_paciente="id"),
        "prob_con_direccion_inversa": dict(
            texto=t, puntuaciones=[Decl("p", "probabilidad", BAJO, umbral=0.3)]),
    }


class _Espia:
    """Se cuela en `ev._Avance` para leer, tras la evaluación, lo que el contador anotó."""

    def __init__(self):
        self.instancias = []

    def __enter__(self):
        espia, original = self, ev._Avance

        class Anotado(original):
            def __init__(s, *a, **k):
                super().__init__(*a, **k)
                espia.instancias.append(s)

        self._original = original
        ev._Avance = Anotado
        return self

    def __exit__(self, *e):
        ev._Avance = self._original


def _con_progreso(config, cancelar_si=None):
    """(resultado, llamadas [(hechas, total, fase)], avance) de una evaluación con progreso."""
    llamadas = []

    def al_progresar(h, tot, fase):
        llamadas.append((h, tot, fase))
        if cancelar_si is not None and cancelar_si(len(llamadas), h, tot, fase):
            raise EvaluacionCancelada("parar")

    kw = dict(config)
    texto = kw.pop("texto")
    with _Espia() as espia:
        try:
            r = _evaluar(texto, al_progresar=al_progresar, **kw)
        except EvaluacionCancelada:
            r = None
    return r, llamadas, espia.instancias[0]


def _remuestras_reales(config):
    """Las remuestras que el código HACE de verdad, contadas desde fuera con el observador de
    `incertidumbre` (y SIN `al_progresar`: el contador del núcleo no está de por medio)."""
    n = [0]
    kw = dict(config)
    texto = kw.pop("texto")
    with observando_remuestras(lambda: n.__setitem__(0, n[0] + 1)):
        _evaluar(texto, **kw)
    return n[0]


class DatosBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filas = _filas()
        cls.configs = _configs(cls.filas)


# ---------------------------------------------------------------------------
# El progreso
# ---------------------------------------------------------------------------

class ElProgresoEsElQueSeCuenta(DatosBase):
    def test_el_progreso_es_monotono_y_acaba_EXACTAMENTE_en_el_total_con_sus_fases(self):
        for nombre, config in self.configs.items():
            with self.subTest(nombre):
                r, llamadas, avance = _con_progreso(config)
                self.assertIsNotNone(r)
                self.assertGreater(len(llamadas), 4)
                self.assertEqual(llamadas[0], (0, 0, "lectura"),
                                 "antes de leer el CSV el total no se sabe: vale 0")
                total = llamadas[-1][1]
                self.assertGreater(total, 0)
                self.assertEqual(llamadas[-1], (total, total, "informe"),
                                 "la última llamada es el total EXACTO, en la fase informe")
                hechas = [h for h, _, _ in llamadas]
                self.assertEqual(hechas, sorted(hechas), "nunca retrocede")
                # Desde que hay plan, el total no cambia y nunca se pasa.
                self.assertTrue(all(t == total for _, t, _ in llamadas[1:]))
                self.assertTrue(all(0 <= h <= total for h, _, _ in llamadas))
                # Las fases son del vocabulario y salen en el orden canónico: `informe` va
                # sola al final y una fase no se anuncia dos veces seguidas.
                fases = [f for _, _, f in llamadas]
                self.assertTrue(set(fases) <= set(FASES_DE_PROGRESO))
                self.assertEqual(fases[0], "lectura")
                self.assertEqual(fases[-1], "informe")
                self.assertEqual(fases.count("informe"), 1)
                self.assertGreater(len(set(fases)), 3)

    def test_las_fases_que_salen_son_las_del_trabajo_que_se_pidio(self):
        _, ll, _ = _con_progreso(self.configs["dos_puntuaciones_con_todo"])
        fases = {f for _, _, f in ll}
        self.assertEqual(fases, set(FASES_DE_PROGRESO))
        _, ll, _ = _con_progreso(self.configs["una_prob_sin_umbral"])
        fases = {f for _, _, f in ll}
        self.assertEqual(fases, {"lectura", "intervalos", "calibracion", "informe"},
                         "sin umbral, sin subgrupos y con una sola puntuación no hay esas fases")
        _, ll, _ = _con_progreso(self.configs["una_escala_baja_no_representa"])
        self.assertNotIn("calibracion", {f for _, _, f in ll},
                         "una escala no se calibra: no se anuncia una fase que no existe")

    def test_el_total_son_las_remuestras_que_se_hacen_de_verdad_contadas_desde_fuera(self):
        for nombre, config in self.configs.items():
            with self.subTest(nombre):
                _, llamadas, avance = _con_progreso(config)
                real = _remuestras_reales(config)
                self.assertEqual(llamadas[-1][1], real,
                                 "el plan del núcleo no cuadra con lo que el código remuestrea")
                self.assertEqual(avance.acreditadas, 0,
                                 "ningún paso se dio por hecho sin hacerse")
                self.assertFalse(avance.desbordado, "se hicieron más remuestras que las previstas")

    def test_el_total_cambia_con_lo_que_se_pide_y_solo_con_eso(self):
        """Números a mano: remuestras × intervalos. Un hueco en el plan (calibración, umbral,
        VPP/VPN, subgrupos con soporte, comparación) cambiaría uno de estos totales."""
        c = self.configs
        esperados = {
            # AUROC + AP + (en grande, intercepto/pendiente) + (sens, esp, VPP, VPN)
            # + los dos subgrupos con soporte de `sexo` ("sala": Grande sí, Chica no)
            "una_prob_umbral_y_subgrupos": (2 + 2 + 4 + 2 + 1) * REM,
            "una_prob_sin_umbral": (2 + 2) * REM,
            # una escala: AUROC + AP + (sens, esp) — sin representatividad no hay VPP/VPN
            "una_escala_baja_no_representa": (2 + 2) * REM,
            # prob: 2+2+4 · escala: 2+4 · subgrupos sexo: 2 por puntuación · comparación 1
            "dos_puntuaciones_con_todo": (2 + 2 + 4 + 2 + 4 + 2 + 2 + 1) * REM,
            "pct_con_pacientes": (2 + 2 + 4) * REM,
            # una probabilidad al revés no es la del evento: sin calibración, solo sens/esp/VPP/VPN
            "prob_con_direccion_inversa": (2 + 4) * REM,
        }
        for nombre, total in esperados.items():
            with self.subTest(nombre):
                _, ll, _ = _con_progreso(c[nombre])
                self.assertEqual(ll[-1][1], total)

    def test_un_intervalo_que_no_se_puede_calcular_se_da_por_hecho_y_el_total_sigue_cuadrando(self):
        """Todos con evento: el AUROC no está definido y no hay nada que remuestrear. El
        contador no se queda a medias: da esos pasos por hechos y acaba en el total."""
        filas = [dict(f, y=1) for f in self.filas[:200]]
        texto = _csv(filas)
        ll = []
        with _Espia() as espia:
            _evaluar(texto, puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)],
                     al_progresar=lambda h, t, f: ll.append((h, t, f)))
        avance = espia.instancias[0]
        self.assertGreater(avance.acreditadas, 0, "esta prueba no ejercitó el camino")
        self.assertEqual(ll[-1], (ll[-1][1], ll[-1][1], "informe"))
        self.assertEqual([h for h, _, _ in ll], sorted(h for h, _, _ in ll))

    def test_el_contador_no_inunda_a_quien_escucha(self):
        """Con 400 remuestras y varios intervalos hay miles de remuestras: la función de
        progreso se llama a lo sumo una vez por 0,5 % de la barra, más los cambios de fase."""
        config = dict(self.configs["dos_puntuaciones_con_todo"])
        texto = config.pop("texto")
        ll = []
        _evaluar(texto, remuestras=400, al_progresar=lambda h, t, f: ll.append((h, t, f)),
                 **{k: v for k, v in config.items()})
        total = ll[-1][1]
        self.assertGreater(total, 4000)
        self.assertLess(len(ll), 320, f"{len(ll)} llamadas para {total} remuestras")
        self.assertGreater(len(ll), 50, "una barra de pocos saltos no es un progreso")

    def test_al_progresar_tiene_que_ser_una_funcion(self):
        with self.assertRaises(EvaluacionRechazada):
            _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO)],
                     al_progresar="no soy una función")


class ElContadorPorDentro(unittest.TestCase):
    """`_Avance` solo, sin evaluar nada: sus garantías no dependen de que el plan acierte."""

    def _avance(self, total):
        ll = []
        a = ev._Avance(lambda h, t, f: ll.append((h, t, f)))
        a.planificar(total)
        return a, ll

    def test_un_tramo_que_hace_mas_de_lo_previsto_no_pasa_del_total_y_lo_anota(self):
        a, ll = self._avance(10)
        with a.tramo("intervalos", 4):
            for _ in range(9):
                a.paso()
        self.assertEqual(a.hechas, 4, "el contador se pasó del tramo")
        self.assertTrue(a.desbordado)
        self.assertTrue(all(h <= 4 for h, _, _ in ll))

    def test_un_tramo_que_hace_menos_se_acredita_y_acaba_en_su_tope(self):
        a, ll = self._avance(10)
        with a.tramo("intervalos", 6):
            for _ in range(2):
                a.paso()
        self.assertEqual((a.hechas, a.acreditadas, a.desbordado), (6, 4, False))
        self.assertEqual(ll[-1][0], 6, "la barra tiene que saltar a lo acreditado")

    def test_un_tramo_sin_pasos_no_anuncia_su_fase(self):
        a, ll = self._avance(10)
        with a.tramo("calibracion", 0):
            pass
        self.assertEqual(ll, [])

    def test_si_el_tramo_se_cancela_no_se_acredita_ni_se_avisa_otra_vez(self):
        a, ll = self._avance(10)
        with self.assertRaises(EvaluacionCancelada):
            with a.tramo("intervalos", 6):
                a.paso()
                raise EvaluacionCancelada("basta")
        n = len(ll)
        self.assertEqual(a.hechas, 1)
        self.assertEqual(len(ll), n, "al cancelar no se vuelve a llamar")

    def test_sin_funcion_de_progreso_no_hace_nada(self):
        a = ev._Avance(None)
        a.planificar(10)
        self.assertFalse(a.activo)
        with a.tramo("intervalos", 5):
            a.paso()  # nadie lo instala como observador, pero tampoco rompe
        self.assertEqual(a.acreditadas, 0)


class SinProgresoNadaCambia(DatosBase):
    def test_con_o_sin_al_progresar_el_informe_y_el_recibo_son_identicos(self):
        for nombre, config in self.configs.items():
            with self.subTest(nombre):
                kw = dict(config)
                texto = kw.pop("texto")
                a = _evaluar(texto, **kw)
                b = _evaluar(texto, al_progresar=lambda *_: None, **kw)
                self.assertEqual(jcs_bytes(a.informe), jcs_bytes(b.informe))
                self.assertEqual(jcs_bytes(a.recibo), jcs_bytes(b.recibo))

    def test_el_informe_de_antes_del_contador_sale_igual_bit_a_bit_salvo_lo_nuevo(self):
        """Los digests son de la ejecución del código de ANTES de este corte (HEAD 5d54f8e)
        con las mismas semillas, sin `version_del_informe`, `versiones` ni `exploracion`
        (lo único que cambia). Si cambias a propósito un número o un texto del informe,
        regenera estos digests y dilo."""
        filas = _filas_c1(900, semilla=7, desplazamiento=0.8)
        texto = _csv_c1(filas)
        pct = _csv_c1([dict(f, p=round(f["p"] * 100, 1)) for f in filas])
        D = Decl
        casos = {
            "a": (texto, dict(puntuaciones=[D("p", "probabilidad", ALTO, umbral=0.3,
                                              cifra_prometida=0.8)],
                              columnas_de_subgrupo=["sexo"], umbrales_de_decision=[0.1, 0.2, 0.3])),
            "b": (texto, dict(puntuaciones=[D("p", "probabilidad", ALTO, umbral=0.4),
                                            D("pts", "puntuacion", ALTO, umbral=5)],
                              representa=False, columnas_de_subgrupo=["sexo"])),
            "c": (texto, dict(puntuaciones=[D("pts", "puntuacion", BAJO, umbral=4)])),
            "d": (pct, dict(puntuaciones=[D("p", "probabilidad_pct", ALTO, umbral=23.5)],
                            columna_de_paciente="id")),
            "e": (texto, dict(puntuaciones=[D("p", "probabilidad", ALTO),
                                            D("pts", "puntuacion", BAJO)],
                              columnas_de_subgrupo=["sexo"])),
        }
        esperados = {
            "a": "85936a5a4973c724ebc1b4865d6875ea552a6b5bed3e0abef3820b1758fb9f3a",
            "b": "e09f09188bfd7011928e744e479fa75cbbe716686deede6ecc5656b406f156e7",
            "c": "d9ceda6f607ec9501182c5949aca61ca098d84c16bc5d1a1c11816aaba18f629",
            "d": "c394bcacc3a0472b9b28c34e05de48fb8d147e684f4ab70fa369cd8a3e588aca",
            "e": "8a2d15185a0a685a11aa869227ef1e6f9e621c021dea4b9d49ad8541194f0629",
        }
        for clave, (t, kw) in casos.items():
            with self.subTest(clave):
                inf = json.loads(json.dumps(_evaluar(t, **kw).informe))
                inf.pop("version_del_informe")
                inf.pop("versiones")
                for p in inf["puntuaciones"]:
                    p["al_umbral"].pop("exploracion")
                self.assertEqual(hashlib.sha256(jcs_bytes(inf)).hexdigest(), esperados[clave])

    def test_la_version_del_informe_sube_a_2_2_0_con_la_exploracion(self):
        self.assertEqual(VERSION_DEL_INFORME, "2.2.0")
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO)])
        self.assertEqual(r.informe["version_del_informe"], "2.2.0")
        self.assertEqual(r.recibo["environment"]["report_version"], "2.2.0")


class LaCancelacionEsLimpia(DatosBase):
    def test_cancelar_detiene_la_evaluacion_sin_resultado_y_ya_no_se_llama_mas(self):
        config = self.configs["dos_puntuaciones_con_todo"]
        _, completo, _ = _con_progreso(config)
        for punto in (2, len(completo) // 3, len(completo) // 2, len(completo) - 2):
            with self.subTest(llamada=punto):
                r, ll, _ = _con_progreso(config, cancelar_si=lambda n, *_: n == punto)
                self.assertIsNone(r, "una evaluación cancelada no devuelve nada a medias")
                self.assertEqual(len(ll), punto, "tras cancelar no se vuelve a llamar")

    def test_se_puede_cancelar_en_cada_fase_incluida_la_lectura_y_el_informe(self):
        config = self.configs["dos_puntuaciones_con_todo"]
        for fase in FASES_DE_PROGRESO:
            with self.subTest(fase):
                visto = []

                def parar(n, h, t, f):
                    if f == fase and not visto:
                        visto.append(n)
                        return True
                    return False

                r, ll, _ = _con_progreso(config, cancelar_si=parar)
                self.assertTrue(visto, f"la fase {fase} no se anunció")
                self.assertIsNone(r)
                self.assertEqual(ll[-1][2], fase)

    def test_cancelar_no_deja_estado_y_la_siguiente_evaluacion_es_la_de_siempre(self):
        config = self.configs["una_prob_umbral_y_subgrupos"]
        kw = dict(config)
        texto = kw.pop("texto")
        antes = _evaluar(texto, **kw)
        _con_progreso(config, cancelar_si=lambda n, h, t, f: h > 20)
        self.assertIsNone(observador_de_remuestras(),
                          "el observador de remuestras se quedó puesto tras cancelar")
        despues = _evaluar(texto, **kw)
        self.assertEqual(jcs_bytes(antes.informe), jcs_bytes(despues.informe))
        n = [0]
        with observando_remuestras(lambda: n.__setitem__(0, n[0] + 1)):
            _evaluar(texto, **kw)
        self.assertGreater(n[0], 0)

    def test_cualquier_excepcion_de_al_progresar_sale_tal_cual_y_tampoco_deja_estado(self):
        class Mia(RuntimeError):
            pass

        def mala(h, t, f):
            if h > 10:
                raise Mia("fallo mío")

        kw = dict(self.configs["una_prob_sin_umbral"])
        texto = kw.pop("texto")
        with self.assertRaises(Mia):
            _evaluar(texto, al_progresar=mala, **kw)
        self.assertIsNone(observador_de_remuestras())

    def test_la_cancelacion_no_es_un_rechazo_de_los_datos(self):
        """Quien captura los rechazos del paquete (`ErrorDeEstudio`) no debe tragarse una
        cancelación como si los datos estuvieran mal."""
        self.assertFalse(issubclass(EvaluacionCancelada, ev.ErrorDeEstudio))
        self.assertFalse(issubclass(EvaluacionCancelada, EvaluacionRechazada))


class ElObservadorDeRemuestras(unittest.TestCase):
    """El enganche del núcleo compartido (`incertidumbre`): sin observador, nada cambia."""

    @classmethod
    def setUpClass(cls):
        r = random.Random(2)
        ys, ps = [], []
        for _ in range(300):
            z = r.gauss(0, 1)
            ys.append("evento" if r.random() < _sig(z) else "no_evento")
            ps.append(_sig(z + 0.3))
        cls.muestra = Muestra.binaria(ys, probabilidades=ps, classes=("no_evento", "evento"),
                                      positive_label="evento")

    def _iv(self, **kw):
        return intervalo("auroc", self.muestra, diseno="iid",
                         estimando="fixed_model_on_population", semilla=11, remuestras=60, **kw)

    def test_se_llama_una_vez_por_remuestra_y_el_intervalo_es_el_mismo(self):
        sin = self._iv()
        n = [0]
        with observando_remuestras(lambda: n.__setitem__(0, n[0] + 1)):
            con = self._iv()
        self.assertEqual(n[0], 60)
        self.assertEqual((sin.ci_low, sin.ci_high), (con.ci_low, con.ci_high))
        self.assertIsNone(observador_de_remuestras(), "al salir se restaura")

    def test_se_instala_dentro_y_se_restaura_el_anterior(self):
        a, b = [0], [0]
        with observando_remuestras(lambda: a.__setitem__(0, a[0] + 1)):
            with observando_remuestras(lambda: b.__setitem__(0, b[0] + 1)):
                self._iv()
            self.assertEqual((a[0], b[0]), (0, 60))
            with observando_remuestras(None):  # None no instala nada: manda el de fuera
                self._iv()
            self.assertEqual((a[0], b[0]), (60, 60))

    def test_un_observador_que_lanza_corta_el_intervalo_y_sale_la_excepcion(self):
        vistas = [0]

        def corta():
            vistas[0] += 1
            if vistas[0] == 7:
                raise EvaluacionCancelada("basta")

        with self.assertRaises(EvaluacionCancelada):
            with observando_remuestras(corta):
                self._iv()
        self.assertEqual(vistas[0], 7, "se siguió remuestreando tras cancelar")
        self.assertIsNone(observador_de_remuestras())

    def test_la_comparacion_emparejada_tambien_cuenta_sus_remuestras(self):
        from matrixai.estudio.comparaciones import comparar_candidatos  # noqa: PLC0415

        otra = Muestra.binaria(
            list(self.muestra.y_true),
            probabilidades=[min(0.99, max(0.01, p + 0.05)) for p in
                            self.muestra.probabilidad_del_positivo],  # type: ignore[union-attr]
            classes=("no_evento", "evento"), positive_label="evento")
        n = [0]
        with observando_remuestras(lambda: n.__setitem__(0, n[0] + 1)):
            comparar_candidatos("auroc", self.muestra, otra, diseno="iid",
                                estimando="fixed_model_on_population", semilla=3, remuestras=45)
        self.assertEqual(n[0], 45)


# ---------------------------------------------------------------------------
# La exploración del umbral
# ---------------------------------------------------------------------------

def _explor(informe, k=0):
    return informe["puntuaciones"][k]["al_umbral"]["exploracion"]


def _usadas(filas, columna, sin=()):
    """(valores, ys) de las filas que SÍ entran: con puntuación y con desenlace."""
    keep = [i for i in range(len(filas)) if i not in sin]
    return [float(filas[i][columna]) for i in keep], [filas[i]["y"] for i in keep]


@unittest.skipUnless(_HAY_SKLEARN, "scikit-learn es la referencia independiente")
class LosRecuentosPorUmbralSonLosDeSklearn(DatosBase):
    def _contra_sklearn(self, informe, valores, ys, *, mas_alto, k=0):
        e = _explor(informe, k)
        self.assertTrue(e["disponible"])
        v, y = np.array(valores), np.array(ys)
        for p in e["puntos"]:
            pred = (v >= p["umbral"]) if mas_alto else (v <= p["umbral"])
            tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel()
            self.assertEqual((p["tp"], p["fp"], p["fn"], p["tn"]),
                             (int(tp), int(fp), int(fn), int(tn)), f"umbral {p['umbral']}")
        return e

    def test_probabilidad_mas_alto_en_varios_umbrales_con_filas_sin_puntuacion(self):
        sin = set(range(20, 700, 37))
        r = _evaluar(_csv(self.filas, sin={"p": sin}), puntuaciones=[
            Decl("p", "probabilidad", ALTO, umbral=0.3)])
        valores, ys = _usadas(self.filas, "p", sin)
        e = self._contra_sklearn(r.informe, valores, ys, mas_alto=True)
        self.assertGreater(e["n_puntos"], 20)
        self.assertEqual(e["regla"], "mayor_o_igual")

    def test_escala_ordinal_con_empates_un_umbral_por_valor_distinto(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("pts", "puntuacion", ALTO)])
        valores, ys = _usadas(self.filas, "pts")
        e = self._contra_sklearn(r.informe, valores, ys, mas_alto=True)
        self.assertEqual([p["umbral"] for p in e["puntos"]], sorted(set(valores)),
                         "los empates se cierran juntos: un vértice por valor distinto")
        self.assertEqual(e["vertices_de_la_curva"], len(set(valores)))

    def test_porcentaje_en_las_unidades_de_la_columna(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[
            Decl("pct", "probabilidad_pct", ALTO, umbral=30.0)])
        valores, ys = _usadas(self.filas, "pct")
        e = self._contra_sklearn(r.informe, valores, ys, mas_alto=True)
        self.assertEqual(e["unidad"], "probabilidad_pct")
        # Los umbrales son valores de la columna TAL COMO están (23,5, no 0,23500000000000003).
        existentes = set(valores)
        self.assertTrue(all(p["umbral"] in existentes for p in e["puntos"]))
        self.assertTrue(any(p["umbral"] > 1 for p in e["puntos"]), "en %, no en 0–1")

    def test_direccion_inversa_el_umbral_se_aplica_al_reves_y_se_dice(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("pts", "puntuacion", BAJO, umbral=4)])
        valores, ys = _usadas(self.filas, "pts")
        e = self._contra_sklearn(r.informe, valores, ys, mas_alto=False)
        self.assertEqual(e["regla"], "menor_o_igual")
        self.assertTrue(e["direccion_invertida"])
        self.assertIn("al revés", e["nota_direccion"]["es"])
        self.assertIn("other way round", e["nota_direccion"]["en"])
        # Subir el umbral señala MÁS casos (con «más alto» sería al contrario).
        señalados = [p["tp"] + p["fp"] for p in e["puntos"]]
        self.assertEqual(señalados, sorted(señalados))
        self.assertGreater(señalados[-1], señalados[0])
        # Y no es el cálculo directo con otro nombre: aplicado «>=» saldría otra tabla.
        v, y = np.array(valores), np.array(ys)
        distintos = 0
        for p in e["puntos"]:
            tn, fp, fn, tp = confusion_matrix(y, (v >= p["umbral"]).astype(int),
                                              labels=[0, 1]).ravel()
            distintos += (p["tp"], p["fp"]) != (int(tp), int(fp))
        self.assertGreater(distintos, 0)

    def test_con_direccion_directa_no_hay_nota_de_inversion(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO)])
        e = _explor(r.informe)
        self.assertFalse(e["direccion_invertida"])
        self.assertIsNone(e["nota_direccion"])

    def test_la_probabilidad_con_direccion_inversa_se_explora_como_escala(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", BAJO)])
        valores, ys = _usadas(self.filas, "p")
        self._contra_sklearn(r.informe, valores, ys, mas_alto=False)

    def test_dos_puntuaciones_cada_una_con_su_exploracion_y_su_direccion(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO),
                                                     Decl("pts", "puntuacion", BAJO)])
        v1, y1 = _usadas(self.filas, "p")
        v2, y2 = _usadas(self.filas, "pts")
        self._contra_sklearn(r.informe, v1, y1, mas_alto=True, k=0)
        self._contra_sklearn(r.informe, v2, y2, mas_alto=False, k=1)


class VppYVpnSoloSiLaMuestraRepresenta(DatosBase):
    def test_con_representativa_se_calculan_y_son_los_del_recuento(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("pts", "puntuacion", ALTO)],
                     representa=True)
        e = _explor(r.informe)
        self.assertTrue(e["vpp_vpn"]["publicables"])
        con_vpp = 0
        for p in e["puntos"]:
            if p["tp"] + p["fp"]:
                self.assertAlmostEqual(p["vpp"], p["tp"] / (p["tp"] + p["fp"]))
                con_vpp += 1
            else:
                self.assertIsNone(p["vpp"], "sin señalados, el VPP es un hueco, no un cero")
            if p["tn"] + p["fn"]:
                self.assertAlmostEqual(p["vpn"], p["tn"] / (p["tn"] + p["fn"]))
            else:
                self.assertIsNone(p["vpn"])
        self.assertGreater(con_vpp, 3)
        self.assertIn("prevalencia de la muestra", e["vpp_vpn"]["nota"]["es"])

    def test_sin_representatividad_ni_VPP_ni_VPN_pero_si_sensibilidad_y_especificidad(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("pts", "puntuacion", ALTO)],
                     representa=False)
        e = _explor(r.informe)
        self.assertFalse(e["vpp_vpn"]["publicables"])
        self.assertIn("NO representa", e["vpp_vpn"]["nota"]["es"])
        for p in e["puntos"]:
            self.assertIsNone(p["vpp"])
            self.assertIsNone(p["vpn"])
            self.assertIsNotNone(p["sensibilidad"])
            self.assertIsNotNone(p["especificidad"])
            self.assertAlmostEqual(p["sensibilidad"], p["tp"] / (p["tp"] + p["fn"]))
            self.assertAlmostEqual(p["especificidad"], p["tn"] / (p["tn"] + p["fp"]))

    def test_en_el_umbral_de_uso_coincide_con_la_tabla_al_umbral_del_informe(self):
        """Dos sitios dando el mismo número: la exploración y `_al_umbral` (que usa el
        registro métrico y la prevalencia declarada) tienen que decir lo mismo."""
        for tipo, col, umbral in (("probabilidad", "p", 0.3), ("puntuacion", "pts", 5.0),
                                  ("probabilidad_pct", "pct", 30.0)):
            with self.subTest(col):
                r = _evaluar(_csv(self.filas), puntuaciones=[Decl(col, tipo, ALTO, umbral=umbral)])
                au = r.informe["puntuaciones"][0]["al_umbral"]
                e = au["exploracion"]
                self.assertIsNotNone(e["indice_del_umbral_de_uso"])
                p = e["puntos"][e["indice_del_umbral_de_uso"]]
                self.assertTrue(p["es_el_umbral_de_uso"])
                self.assertEqual((p["tp"], p["fp"], p["fn"], p["tn"]),
                                 (au["matriz"]["tp"], au["matriz"]["fp"], au["matriz"]["fn"],
                                  au["matriz"]["tn"]))
                for mio, suyo in (("sensibilidad", "sensibilidad"), ("especificidad",
                                  "especificidad"), ("vpp", "vpp"), ("vpn", "vpn")):
                    self.assertAlmostEqual(p[mio], au[suyo]["valor"], places=9)


class TopeDePuntosYOrden(DatosBase):
    def test_con_mas_vertices_que_el_tope_se_adelgaza_y_nunca_se_pasa(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)])
        e = _explor(r.informe)
        self.assertGreater(e["vertices_de_la_curva"], PUNTOS_DE_LA_EXPLORACION * 3)
        self.assertLessEqual(e["n_puntos"], PUNTOS_DE_LA_EXPLORACION)
        self.assertGreater(e["n_puntos"], PUNTOS_DE_LA_EXPLORACION - 6)
        self.assertEqual(len(e["puntos"]), e["n_puntos"])
        self.assertEqual(e["tope_de_puntos"], PUNTOS_DE_LA_EXPLORACION)
        self.assertIn("como mucho", e["nota_de_los_umbrales"]["es"])

    def test_los_puntos_son_vertices_reales_ordenados_y_con_los_dos_extremos(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO)])
        e = _explor(r.informe)
        umbrales = [p["umbral"] for p in e["puntos"]]
        self.assertEqual(umbrales, sorted(umbrales))
        self.assertEqual(len(set(umbrales)), len(umbrales))
        valores = {float(f["p"]) for f in self.filas}
        self.assertTrue(set(umbrales) <= valores, "un umbral inventado, no un valor de la columna")
        self.assertEqual(umbrales[0], min(valores))
        self.assertEqual(umbrales[-1], max(valores))
        self.assertEqual(e["orden"], "umbral_ascendente")
        # Extremos: el valor menor señala a todos; el mayor, solo a los que lo alcanzan.
        n = len(self.filas)
        primero, ultimo = e["puntos"][0], e["puntos"][-1]
        self.assertEqual(primero["tp"] + primero["fp"], n)
        self.assertEqual(primero["fn"] + primero["tn"], 0)
        self.assertLess(ultimo["tp"] + ultimo["fp"], 10)

    def test_con_pocos_vertices_estan_todos_y_la_nota_lo_dice(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("pts", "puntuacion", ALTO)])
        e = _explor(r.informe)
        self.assertEqual(e["n_puntos"], e["vertices_de_la_curva"])
        self.assertIn(f"Se ofrecen los {e['vertices_de_la_curva']} umbrales distintos",
                      e["nota_de_los_umbrales"]["es"])
        self.assertNotIn("como mucho", e["nota_de_los_umbrales"]["es"])

    def test_el_umbral_de_uso_sobrevive_al_adelgazado(self):
        """Hay ~600 vértices y 50 puntos: aun así el vértice del umbral de uso está (se
        fuerza), para que la pantalla parta de él."""
        for umbral in (0.17, 0.33, 0.5, 0.71):
            with self.subTest(umbral):
                r = _evaluar(_csv(self.filas), puntuaciones=[
                    Decl("p", "probabilidad", ALTO, umbral=umbral)])
                au = r.informe["puntuaciones"][0]["al_umbral"]
                e = au["exploracion"]
                self.assertLessEqual(e["n_puntos"], PUNTOS_DE_LA_EXPLORACION)
                p = e["puntos"][e["indice_del_umbral_de_uso"]]
                self.assertEqual((p["tp"], p["fp"], p["fn"], p["tn"]),
                                 tuple(au["matriz"][k] for k in ("tp", "fp", "fn", "tn")))
                self.assertGreaterEqual(p["umbral"], umbral)

    def test_un_umbral_de_uso_que_no_alcanza_a_nadie_no_tiene_punto(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("pts", "puntuacion", ALTO, umbral=99)])
        e = _explor(r.informe)
        self.assertIsNone(e["indice_del_umbral_de_uso"])
        self.assertFalse(any(p["es_el_umbral_de_uso"] for p in e["puntos"]))

    def test_sin_umbral_de_uso_tambien_hay_exploracion(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO)])
        au = r.informe["puntuaciones"][0]["al_umbral"]
        self.assertFalse(au["declarado"])
        self.assertTrue(au["exploracion"]["disponible"])
        self.assertIsNone(au["exploracion"]["indice_del_umbral_de_uso"])

    def test_sin_los_dos_desenlaces_no_hay_curva_y_se_dice_en_vez_de_inventarla(self):
        r = _evaluar(_csv([dict(f, y=1) for f in self.filas[:100]]),
                     puntuaciones=[Decl("p", "probabilidad", ALTO)])
        e = _explor(r.informe)
        self.assertFalse(e["disponible"])
        self.assertIn("falta alguno de los dos desenlaces", e["motivo"]["es"])
        self.assertNotIn("puntos", e)


class SoloAgregadosYSinOcultarRecuentosPequenos(DatosBase):
    def test_no_hay_filas_ni_identificadores_en_la_exploracion(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3),
                                                     Decl("pts", "puntuacion", ALTO)],
                     columna_de_paciente="id")
        texto = json.dumps(r.informe)
        self.assertNotIn(RARO, texto)
        permitidas = {"umbral", "tp", "fp", "fn", "tn", "sensibilidad", "especificidad", "vpp",
                      "vpn", "es_el_umbral_de_uso"}
        for k in (0, 1):
            for p in _explor(r.informe, k)["puntos"]:
                self.assertEqual(set(p), permitidas)

    def test_un_recuento_de_1_o_2_casos_no_se_oculta_pero_se_cuenta_y_se_avisa(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO)])
        e = _explor(r.informe)
        con_pocos = [p for p in e["puntos"]
                     if any(0 < p[c] <= 2 for c in ("tp", "fp", "fn", "tn"))]
        self.assertGreater(len(con_pocos), 0, "el caso de prueba no tiene celdas pequeñas")
        self.assertEqual(e["recuentos_pequenos"]["umbrales"], len(con_pocos))
        # No se ocultó nada: el recuento pequeño está ahí, con su número.
        self.assertTrue(any(p["tp"] in (1, 2) or p["fp"] in (1, 2) for p in e["puntos"]))
        self.assertIn("no garantiza anonimización", e["recuentos_pequenos"]["nota"]["es"])
        self.assertIn("does not guarantee anonymisation", e["recuentos_pequenos"]["nota"]["en"])

    def test_dice_que_no_valida_un_umbral_nuevo_en_los_dos_idiomas(self):
        e = _explor(_evaluar(_csv(self.filas), puntuaciones=[
            Decl("p", "probabilidad", ALTO)]).informe)
        self.assertIn("no valida un nuevo umbral", e["aviso"]["es"])
        self.assertIn("does not validate a new threshold", e["aviso"]["en"])

    def test_los_textos_nuevos_estan_en_los_dos_idiomas_con_los_mismos_huecos(self):
        nuevos = [k for k in MOTIVOS if k.startswith("exploracion_")]
        self.assertGreaterEqual(len(nuevos), 9)
        for k in nuevos:
            with self.subTest(k):
                self.assertEqual(huecos_de(MOTIVOS[k]["es"]), huecos_de(MOTIVOS[k]["en"]))
                self.assertTrue(MOTIVOS[k]["es"].strip() and MOTIVOS[k]["en"].strip())

    def test_el_informe_con_exploracion_se_puede_canonicalizar_y_firmar(self):
        r = _evaluar(_csv(self.filas), puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)],
                     clave=b"k" * 32, creado_en="2026-10-25T10:00:00Z")
        self.assertEqual(r.nivel_de_garantia, "A1")
        self.assertTrue(jcs_bytes(r.informe))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
