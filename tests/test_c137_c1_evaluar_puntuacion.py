# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""137-C1 — «¿Funciona aquí?»: evaluar una puntuación YA REGISTRADA, sin entrenar.

Cada prueba lleva el NOMBRE de la regla que sostiene (§2 y §3 del contrato) y tiene su
sabotaje en rojo anotado en el informe del corte. Los datos son sintéticos pero con las
FORMAS que trae la gente: `Sí`/`SI`/`sí`, `0/1`, `fallecido/vivo`, coma decimal, `NA`,
Windows-1252 y el índice de fila sin nombre que escribe `write.csv` de R.

Dos cosas que conviene saber antes de leer:

* **Se prueba el producto, no solo la función**: la última clase corre
  `python -m matrixai evaluate-score` de punta a punta sobre ficheros reales.
* **La referencia independiente es scikit-learn** (AUROC, AP, Brier y la curva de
  fiabilidad); el núcleo es stdlib puro y sklearn solo vive aquí, con `skipUnless`.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matrixai.estudio.evaluacion_de_puntuacion import (
    MAXIMO_DE_VALORES_POR_SUBGRUPO,
    DeclaracionDePuntuacion as _Decl,
    EvaluacionRechazada,
    evaluar_puntuacion,
)
from matrixai.estudio.evaluacion_de_puntuacion_textos import MOTIVOS, huecos_de
from matrixai.pipelines.receipt import nivel_del_recibo, verificar_recibo

try:
    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    _HAY_SKLEARN = True
except Exception:  # pragma: no cover
    _HAY_SKLEARN = False

_RAIZ = Path(__file__).resolve().parent.parent
REM = 40  # remuestras de las pruebas: el intervalo no es lo que se prueba casi nunca
ALTO, BAJO = "mas_alto_mas_riesgo", "mas_bajo_mas_riesgo"
R2 = dict(horizonte_del_desenlace="durante el ingreso", momento_de_la_puntuacion="al ingreso",
          donde_se_calcula="local")


def Decl(columna, tipo, direccion, **kw):
    """Las pruebas que no hablan de la R2 declaran «no participaron»; las que sí, lo pasan."""
    kw.setdefault("casos_en_su_desarrollo", "no")
    return _Decl(columna, tipo, direccion, **kw)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _filas(n: int = 1500, semilla: int = 1, desplazamiento: float = 0.0) -> list[dict]:
    """Un riesgo `p` (probabilidad) y una escala `pts` (0–10) sobre un riesgo latente `z`.
    Con `desplazamiento == 0` el riesgo está calibrado POR CONSTRUCCIÓN (y ~ Bernoulli(p));
    con otro, `p` está corrido en logit y el desenlace no."""
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


def _csv(filas: list[dict], columnas: list[str], *, sep: str = ",", coma: bool = False,
         si: str = "1", no: str = "0", ausentes: dict[str, set[int]] | None = None,
         ausente: str = "") -> str:
    ausentes = ausentes or {}
    lineas = [sep.join(columnas)]
    for i, f in enumerate(filas):
        celdas = []
        for c in columnas:
            if i in ausentes.get(c, ()):
                celdas.append(ausente)
                continue
            v = f[c]
            if c == "y":
                v = si if v else no
            elif isinstance(v, float):
                v = repr(round(v, 6))
                if coma:
                    v = v.replace(".", ",")
            celdas.append(str(v))
        lineas.append(sep.join(celdas))
    return "\n".join(lineas) + "\n"


def _evaluar(texto, *, puntuaciones=None, evento="1", representa=True, **kw):
    kw.setdefault("remuestras", REM)
    for k, v in R2.items():
        kw.setdefault(k, v)
    return evaluar_puntuacion(
        texto, columna_desenlace="y", valor_del_evento=evento,
        puntuaciones=puntuaciones or [Decl("p", "probabilidad", ALTO)],
        muestra_representativa=representa, **kw)


def _auc(informe, k=0):
    return informe["puntuaciones"][k]["discriminacion"]["auroc"]["valor"]


class DatosBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filas = _filas()


# ---------------------------------------------------------------------------
# §3 — las pruebas con nombre
# ---------------------------------------------------------------------------

class CalibracionTest(DatosBase):
    def test_el_desplazamiento_plantado_en_logit_se_encuentra_el_ic_del_intercepto_excluye_el_cero(self):
        corrido = _filas(desplazamiento=0.8)
        r = _evaluar(_csv(corrido, ["p", "y"]))
        cal = r.informe["puntuaciones"][0]["calibracion"]
        a = cal["intercepto"]
        self.assertTrue(cal["aplica"])
        self.assertLess(a["ic"]["alto"], 0.0, "el IC del intercepto tiene que excluir el 0")
        self.assertAlmostEqual(a["valor"], -0.8, delta=0.25)
        # la calibración en grande dice lo mismo: se promete más riesgo del que hubo
        self.assertGreater(cal["en_grande"]["ic"]["bajo"], 0.0)

    def test_control_conocido_bueno_un_riesgo_calibrado_no_excluye_el_cero_ni_el_uno(self):
        r = _evaluar(_csv(self.filas, ["p", "y"]))
        cal = r.informe["puntuaciones"][0]["calibracion"]
        self.assertLessEqual(cal["intercepto"]["ic"]["bajo"], 0.0)
        self.assertGreaterEqual(cal["intercepto"]["ic"]["alto"], 0.0)
        self.assertLessEqual(cal["pendiente"]["ic"]["bajo"], 1.0)
        self.assertGreaterEqual(cal["pendiente"]["ic"]["alto"], 1.0)

    def test_una_puntuacion_ordinal_da_calibracion_no_aplica_con_su_motivo_y_ningun_numero(self):
        r = _evaluar(_csv(self.filas, ["pts", "y"]),
                     puntuaciones=[Decl("pts", "puntuacion", ALTO, umbral=5)],
                     umbrales_de_decision=[0.1, 0.2])
        res = r.informe["puntuaciones"][0]
        cal = res["calibracion"]
        self.assertFalse(cal["aplica"])
        self.assertTrue(cal["motivo"]["es"] and cal["motivo"]["en"])
        for clave in ("en_grande", "intercepto", "pendiente", "brier", "fiabilidad"):
            self.assertNotIn(clave, cal, f"{clave} no puede existir con una escala de puntos")
        # y la curva de decisión tampoco: un pt de probabilidad no vive en una escala
        self.assertFalse(res["curva_de_decision"]["aplica"])
        self.assertNotIn("puntos", res["curva_de_decision"])
        # lo que SÍ mide una escala, lo mide
        self.assertIsNotNone(res["discriminacion"]["auroc"]["valor"])

    def test_una_probabilidad_con_direccion_inversa_no_se_calibra_contra_el_evento(self):
        r = _evaluar(_csv(self.filas, ["p", "y"]),
                     puntuaciones=[Decl("p", "probabilidad", BAJO)])
        res = r.informe["puntuaciones"][0]
        self.assertFalse(res["calibracion"]["aplica"])
        self.assertIn("más bajo", res["calibracion"]["motivo"]["es"])
        self.assertIn("lower", res["calibracion"]["motivo"]["en"])
        self.assertNotIn("intercepto", res["calibracion"])
        self.assertFalse(res["curva_de_decision"]["aplica"])


class RangoTest(DatosBase):
    def test_una_probabilidad_fuera_de_0_1_se_rechaza_diciendo_cuantos_y_cuales_y_no_se_recorta(self):
        filas = [dict(f) for f in self.filas]
        filas[3]["p"], filas[7]["p"] = 1.2, -0.1
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(_csv(filas, ["p", "y"]))
        self.assertEqual(cm.exception.clave, "probabilidad_fuera_de_rango")
        self.assertIn("2", cm.exception.es)
        self.assertIn("1.2", cm.exception.es)
        self.assertIn("-0.1", cm.exception.es)

    def test_una_probabilidad_en_pct_fuera_de_0_100_se_rechaza(self):
        filas = [dict(f, p=f["p"] * 100) for f in self.filas]
        filas[0]["p"] = 100.5
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(_csv(filas, ["p", "y"]), puntuaciones=[Decl("p", "probabilidad_pct", ALTO)])
        self.assertEqual(cm.exception.clave, "probabilidad_fuera_de_rango")

    def test_un_porcentaje_NO_se_acepta_como_probabilidad_aunque_parezca_valido_el_tipo_lo_declara_quien_evalua(self):
        # 0–100 declarado como probabilidad: los valores >1 se rechazan, no se deducen como %
        filas = [dict(f, p=f["p"] * 100) for f in self.filas]
        with self.assertRaises(EvaluacionRechazada):
            _evaluar(_csv(filas, ["p", "y"]))

    def test_probabilidad_pct_equivale_a_la_probabilidad_dividida_por_100(self):
        filas = self.filas
        a = _evaluar(_csv(filas, ["p", "y"]),
                     puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.25)]).informe
        pct = [dict(f, p=round(f["p"], 6) * 100) for f in filas]
        b = _evaluar(_csv(pct, ["p", "y"]),
                     puntuaciones=[Decl("p", "probabilidad_pct", ALTO, umbral=25)]).informe
        ra, rb = a["puntuaciones"][0], b["puntuaciones"][0]
        self.assertAlmostEqual(_auc(a), _auc(b), places=9)
        self.assertAlmostEqual(ra["calibracion"]["brier"]["valor"],
                               rb["calibracion"]["brier"]["valor"], places=9)
        self.assertAlmostEqual(ra["calibracion"]["intercepto"]["valor"],
                               rb["calibracion"]["intercepto"]["valor"], places=6)
        self.assertEqual(ra["al_umbral"]["matriz"], rb["al_umbral"]["matriz"])
        self.assertEqual(rb["al_umbral"]["umbral"], 25)            # lo declarado, en su unidad
        self.assertAlmostEqual(rb["al_umbral"]["umbral_en_la_escala_medida"], 0.25)


class DireccionTest(DatosBase):
    def test_una_direccion_declarada_al_reves_no_se_corrige_en_silencio_y_se_avisa(self):
        bien = _evaluar(_csv(self.filas, ["pts", "y"]),
                        puntuaciones=[Decl("pts", "puntuacion", ALTO)]).informe
        mal = _evaluar(_csv(self.filas, ["pts", "y"]),
                       puntuaciones=[Decl("pts", "puntuacion", BAJO)]).informe
        self.assertGreater(_auc(bien), 0.5)
        self.assertEqual([a["id"] for a in bien["puntuaciones"][0]["avisos"]], [])
        # se publica TAL COMO SALE (1 - AUROC), no el corregido
        self.assertLess(_auc(mal), 0.5)
        self.assertAlmostEqual(_auc(mal), 1.0 - _auc(bien), places=9)
        avisos = mal["puntuaciones"][0]["avisos"]
        self.assertEqual([a["id"] for a in avisos], ["direccion_posiblemente_mal_declarada"])
        self.assertIn("0,5", avisos[0]["es"])
        self.assertIn("0.5", avisos[0]["en"])

    def test_mas_bajo_mas_riesgo_con_umbral_avisa_cuando_la_puntuacion_es_menor_o_igual(self):
        filas = [dict(f, pts=10 - f["pts"]) for f in self.filas]          # escala invertida
        r = _evaluar(_csv(filas, ["pts", "y"]),
                     puntuaciones=[Decl("pts", "puntuacion", BAJO, umbral=4)]).informe
        u = r["puntuaciones"][0]["al_umbral"]
        self.assertEqual(u["regla_de_aviso"], "menor_o_igual")
        tp = sum(1 for f in filas if f["pts"] <= 4 and f["y"])
        fp = sum(1 for f in filas if f["pts"] <= 4 and not f["y"])
        self.assertEqual((u["matriz"]["tp"], u["matriz"]["fp"]), (tp, fp))
        self.assertGreater(_auc(r), 0.5)


class SubgruposTest(unittest.TestCase):
    @staticmethod
    def _con_grupos(grupos: list[tuple[str, int, int]]):
        r = random.Random(5)
        filas = []
        for g, n, eventos in grupos:
            for i in range(n):
                ev = i < eventos
                p = min(0.99, max(0.01, (0.6 if ev else 0.3) + 0.2 * r.gauss(0, 1)))
                filas.append({"p": p, "y": 1 if ev else 0, "g": g})
        r.shuffle(filas)
        return _csv(filas, ["p", "y", "g"])

    def _grupo(self, grupos, nombre):
        inf = _evaluar(self._con_grupos(grupos), columnas_de_subgrupo=["g"]).informe
        return next(x for x in inf["puntuaciones"][0]["subgrupos"][0]["grupos"]
                    if x["valor"] == nombre)

    def test_un_subgrupo_de_29_filas_no_publica_numero(self):
        g = self._grupo([("A", 29, 15), ("B", 400, 150)], "A")
        self.assertFalse(g["soporte_suficiente"])
        for clave in ("auroc", "calibracion_en_grande", "eventos"):
            self.assertNotIn(clave, g)
        self.assertTrue(g["motivo"]["es"] and g["motivo"]["en"])

    def test_un_subgrupo_de_30_filas_con_9_eventos_no_publica_numero(self):
        g = self._grupo([("A", 30, 9), ("B", 400, 150)], "A")
        self.assertFalse(g["soporte_suficiente"])
        self.assertNotIn("auroc", g)
        self.assertNotIn("eventos", g)

    def test_en_el_limite_30_filas_y_10_eventos_si_hay_numero(self):
        g = self._grupo([("A", 30, 10), ("B", 400, 150)], "A")
        self.assertTrue(g["soporte_suficiente"])
        self.assertIsNotNone(g["auroc"]["valor"])
        self.assertEqual(g["eventos"], 10)

    def test_el_numero_del_grupo_grande_no_revela_el_del_pequeno(self):
        # con dos grupos, «el resto» del grande SERÍA el pequeño: no se publica
        inf = _evaluar(self._con_grupos([("A", 29, 15), ("B", 400, 150)]),
                       columnas_de_subgrupo=["g"]).informe
        texto = json.dumps(inf, ensure_ascii=False)
        self.assertNotIn("resto", texto)

    def test_una_columna_de_subgrupo_parecida_a_un_identificador_se_rechaza(self):
        filas = [{"p": 0.3 + 0.001 * (i % 50), "y": i % 2, "g": f"X{i}"}
                 for i in range(MAXIMO_DE_VALORES_POR_SUBGRUPO * 4)]
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(_csv(filas, ["p", "y", "g"]), columnas_de_subgrupo=["g"])
        self.assertEqual(cm.exception.clave, "subgrupo_con_demasiados_valores")
        # y el rechazo no enseña los valores (serían identificadores)
        self.assertNotIn("X1", cm.exception.es)

    def test_los_subgrupos_unen_mayusculas_y_tildes_y_lo_dicen(self):
        filas = _filas(600)
        for i, f in enumerate(filas):
            f["sexo"] = ["Mujer", "mujer", "MUJER", "Hombre"][i % 4]
        inf = _evaluar(_csv(filas, ["p", "y", "sexo"]), columnas_de_subgrupo=["sexo"]).informe
        sub = inf["puntuaciones"][0]["subgrupos"][0]
        self.assertEqual(len(sub["grupos"]), 2)
        self.assertEqual(len(sub["variantes_unidas"][0]["variantes"]), 3)


class UmbralTest(DatosBase):
    def test_vpp_y_vpn_no_salen_si_la_muestra_no_representa_a_la_poblacion(self):
        csv = _csv(self.filas, ["p", "y"])
        decl = [Decl("p", "probabilidad", ALTO, umbral=0.3)]
        no = _evaluar(csv, puntuaciones=decl, representa=False).informe["puntuaciones"][0]["al_umbral"]
        for clave in ("vpp", "vpn"):
            self.assertIsNone(no[clave]["valor"])
            self.assertIsNone(no[clave]["ic"])
            self.assertTrue(no[clave]["no_definido"]["es"] and no[clave]["no_definido"]["en"])
        self.assertIsNone(no["prevalencia_declarada"])
        # sensibilidad y especificidad NO dependen de la prevalencia: salen igual
        self.assertIsNotNone(no["sensibilidad"]["valor"])
        self.assertIsNotNone(no["especificidad"]["valor"])

    def test_con_la_muestra_representativa_vpp_y_vpn_salen_con_la_prevalencia_de_la_muestra(self):
        csv = _csv(self.filas, ["p", "y"])
        r = _evaluar(csv, puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)],
                     representa=True).informe["puntuaciones"][0]
        u = r["al_umbral"]
        m = u["matriz"]
        self.assertAlmostEqual(u["vpp"]["valor"], m["tp"] / (m["tp"] + m["fp"]), places=9)
        self.assertAlmostEqual(u["vpn"]["valor"], m["tn"] / (m["tn"] + m["fn"]), places=9)
        self.assertAlmostEqual(u["prevalencia_declarada"], r["prevalencia"], places=12)
        self.assertIsNotNone(u["nota"])           # dice que su IC no recoge la prevalencia

    def test_sin_umbral_de_uso_no_hay_tabla_al_umbral(self):
        r = _evaluar(_csv(self.filas, ["p", "y"])).informe["puntuaciones"][0]
        self.assertFalse(r["al_umbral"]["declarado"])
        self.assertNotIn("sensibilidad", r["al_umbral"])
        self.assertTrue(r["al_umbral"]["motivo"]["es"])

    def test_la_matriz_al_umbral_coincide_con_el_recuento_a_mano(self):
        r = _evaluar(_csv(self.filas, ["p", "y"]),
                     puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.4)]).informe
        m = r["puntuaciones"][0]["al_umbral"]["matriz"]
        ps = [(round(f["p"], 6), f["y"]) for f in self.filas]
        self.assertEqual(m["tp"], sum(1 for p, y in ps if p >= 0.4 and y))
        self.assertEqual(m["fp"], sum(1 for p, y in ps if p >= 0.4 and not y))
        self.assertEqual(m["fn"], sum(1 for p, y in ps if p < 0.4 and y))
        self.assertEqual(m["tn"], sum(1 for p, y in ps if p < 0.4 and not y))


class CifraPrometidaTest(DatosBase):
    def test_la_cifra_prometida_se_compara_con_el_intervalo_dentro_o_fuera_sin_adjetivos(self):
        csv = _csv(self.filas, ["p", "y"])
        medido = _auc(_evaluar(csv).informe)
        dentro = _evaluar(csv, puntuaciones=[Decl("p", "probabilidad", ALTO, cifra_prometida=medido)])
        fuera = _evaluar(csv, puntuaciones=[Decl("p", "probabilidad", ALTO, cifra_prometida=0.99)])
        pd_ = dentro.informe["puntuaciones"][0]["cifra_prometida"]
        pf = fuera.informe["puntuaciones"][0]["cifra_prometida"]
        self.assertEqual((pd_["resultado"], pf["resultado"]), ("dentro", "fuera"))
        self.assertEqual(len(pf["intervalo"]), 2)
        texto = json.dumps(fuera.informe, ensure_ascii=False).lower()
        for adjetivo in ("bueno", "malo", "aceptable", "excelente", "good", "poor", "bad"):
            self.assertNotIn(f" {adjetivo} ", texto)


class FaltaDePuntuacionTest(DatosBase):
    def _con_falta(self, tasa_ev: float, tasa_no: float):
        r = random.Random(11)
        quitar = {i for i, f in enumerate(self.filas)
                  if r.random() < (tasa_ev if f["y"] else tasa_no)}
        csv = _csv(self.filas, ["p", "y"], ausentes={"p": quitar}, ausente="NA")
        return _evaluar(csv).informe["puntuaciones"][0], len(quitar)

    def test_la_falta_de_puntuacion_concentrada_en_un_desenlace_se_avisa_sin_afirmar_la_direccion_del_sesgo(self):
        res, quitadas = self._con_falta(0.15, 0.01)
        ids = [a["id"] for a in res["avisos"]]
        self.assertIn("falta_concentrada_en_un_desenlace", ids)
        self.assertTrue(res["filas"]["falta_concentrada"])
        self.assertEqual(res["filas"]["sin_puntuacion"], quitadas)
        aviso = next(a for a in res["avisos"] if a["id"] == "falta_concentrada_en_un_desenlace")
        self.assertIn("no se puede saber", aviso["es"])
        self.assertIn("cannot be known", aviso["en"])

    def test_la_falta_concentrada_en_los_NO_eventos_tambien_se_avisa(self):
        res, _ = self._con_falta(0.01, 0.15)
        self.assertIn("falta_concentrada_en_un_desenlace", [a["id"] for a in res["avisos"]])

    def test_control_una_falta_repartida_por_igual_no_avisa_pero_se_cuenta(self):
        res, quitadas = self._con_falta(0.05, 0.05)
        self.assertEqual([a["id"] for a in res["avisos"]], [])
        self.assertGreater(res["filas"]["sin_puntuacion"], 0)
        self.assertEqual(res["filas"]["sin_puntuacion"], quitadas)


class DosPuntuacionesTest(DatosBase):
    def test_dos_puntuaciones_se_comparan_solo_sobre_las_filas_que_tienen_las_dos(self):
        faltan = set(range(0, 300))
        csv = _csv(self.filas, ["p", "pts", "y"], ausentes={"pts": faltan}, ausente="NA")
        inf = _evaluar(csv, puntuaciones=[Decl("p", "probabilidad", ALTO),
                                          Decl("pts", "puntuacion", ALTO)]).informe
        a, b = inf["puntuaciones"]
        c = inf["comparacion"]
        self.assertEqual(a["filas"]["usadas"], 1500)
        self.assertEqual(b["filas"]["usadas"], 1200)
        self.assertEqual(c["filas_comunes"], 1200)
        # y los AUROC de la comparación son los de las filas COMUNES, no los de cada una
        comunes = self.filas[300:]
        if _HAY_SKLEARN:
            ya = [f["y"] for f in comunes]
            self.assertAlmostEqual(c["auroc_a"]["valor"],
                                   roc_auc_score(ya, [round(f["p"], 6) for f in comunes]), places=9)
            self.assertAlmostEqual(c["auroc_b"]["valor"],
                                   roc_auc_score(ya, [f["pts"] for f in comunes]), places=9)
        self.assertNotAlmostEqual(c["auroc_a"]["valor"], _auc(inf, 0), places=6)
        self.assertAlmostEqual(c["diferencia"], c["auroc_a"]["valor"] - c["auroc_b"]["valor"],
                               places=9)
        self.assertIsNotNone(c["ic"])

    def test_sin_ninguna_fila_en_comun_la_comparacion_no_se_inventa(self):
        a = set(range(0, 750))
        b = set(range(750, 1500))
        csv = _csv(self.filas, ["p", "pts", "y"], ausentes={"p": a, "pts": b}, ausente="NA")
        inf = _evaluar(csv, puntuaciones=[Decl("p", "probabilidad", ALTO),
                                          Decl("pts", "puntuacion", ALTO)]).informe
        self.assertFalse(inf["comparacion"]["posible"])
        self.assertTrue(inf["comparacion"]["motivo"]["es"])

    def test_la_misma_columna_dos_veces_se_rechaza(self):
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(_csv(self.filas, ["p", "y"]),
                     puntuaciones=[Decl("p", "probabilidad", ALTO), Decl("p", "probabilidad", BAJO)])
        self.assertEqual(cm.exception.clave, "columna_repetida_en_el_encargo")


# ---------------------------------------------------------------------------
# La entrada: las formas que trae la gente
# ---------------------------------------------------------------------------

class FormasDelDesenlaceTest(DatosBase):
    def test_si_no_con_mayusculas_y_tildes_se_unen_y_se_dice_cuales(self):
        r = random.Random(3)
        filas = [dict(f) for f in self.filas]
        texto = _csv(filas, ["p", "y"], si="Sí", no="No")
        lineas = texto.splitlines()
        for i in range(1, len(lineas)):
            if lineas[i].endswith(",Sí") and r.random() < 0.5:
                lineas[i] = lineas[i][:-2] + r.choice(["SI", "sí", "si"])
            elif lineas[i].endswith(",No") and r.random() < 0.5:
                lineas[i] = lineas[i][:-2] + r.choice(["NO", "no"])
        inf = _evaluar("\n".join(lineas) + "\n", evento="SÍ").informe
        d = inf["desenlace"]
        self.assertEqual(d["eventos"], sum(f["y"] for f in filas))
        self.assertEqual(d["no_eventos"], sum(1 - f["y"] for f in filas))
        unidos = {u["valor"]: {v["texto"] for v in u["variantes"]} for u in d["variantes_unidas"]}
        self.assertEqual(len(unidos), 2)
        self.assertTrue(any({"SI", "sí", "si"} & vs for vs in unidos.values()))
        self.assertEqual(sum(v["filas"] for u in d["variantes_unidas"] for v in u["variantes"]),
                         len(filas))

    def test_0_1_y_1_0_son_el_mismo_valor_y_fallecido_vivo_tambien_se_lee(self):
        a = _evaluar(_csv(self.filas, ["p", "y"], si="1", no="0"), evento="1").informe
        b = _evaluar(_csv(self.filas, ["p", "y"], si="fallecido", no="vivo"),
                     evento="Fallecido").informe
        self.assertEqual(_auc(a), _auc(b))
        # «1.0», «1,0» y «1» son el mismo valor: se unen, y se dice
        lineas = _csv(self.filas, ["p", "y"]).splitlines()
        for i in range(1, len(lineas)):
            if lineas[i].endswith(",1") and i % 3 == 0:
                lineas[i] += ".0"
        c = _evaluar("\n".join(lineas) + "\n", evento="1").informe
        self.assertEqual(_auc(a), _auc(c))
        unidas = c["desenlace"]["variantes_unidas"]
        self.assertEqual(len(unidas), 1)
        self.assertEqual({v["texto"] for v in unidas[0]["variantes"]}, {"1", "1.0"})

    def test_el_evento_nunca_se_adivina_uno_que_no_esta_se_rechaza_diciendo_los_que_hay(self):
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(_csv(self.filas, ["p", "y"], si="Sí", no="No"), evento="muerto")
        self.assertEqual(cm.exception.clave, "desenlace_sin_el_evento")
        self.assertIn("Sí", cm.exception.es)

    def test_un_desenlace_con_tres_valores_se_rechaza(self):
        texto = _csv(self.filas, ["p", "y"]).replace(",0\n", ",2\n", 3)
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(texto)
        self.assertEqual(cm.exception.clave, "desenlace_con_mas_de_dos_valores")

    def test_las_filas_sin_desenlace_se_cuentan_y_no_cuentan(self):
        texto = _csv(self.filas, ["p", "y"], ausentes={"y": {0, 1, 2}}, ausente="NA")
        inf = _evaluar(texto).informe
        self.assertEqual(inf["entrada"]["filas_sin_desenlace"], 3)
        self.assertEqual(inf["puntuaciones"][0]["filas"]["con_desenlace"], 1497)


class FormasDeLaPuntuacionTest(DatosBase):
    def test_coma_decimal_y_vacios_variados_se_leen_y_se_declaran(self):
        quitar = {0: "", 1: "NA", 2: "-", 3: "NULL", 4: "nan", 5: "na"}
        lineas = _csv(self.filas, ["p", "pts", "y"], sep=";", coma=True).splitlines()
        for k, marca in quitar.items():
            celdas = lineas[k + 1].split(";")
            celdas[0] = marca
            lineas[k + 1] = ";".join(celdas)
        inf = _evaluar("\n".join(lineas) + "\n").informe
        res = inf["puntuaciones"][0]
        self.assertEqual(res["lectura"]["decimal"], "coma")
        self.assertEqual(inf["entrada"]["separador"], ";")
        self.assertEqual(res["filas"]["sin_puntuacion"], 6)
        self.assertEqual(sorted(res["lectura"]["marcas_de_ausencia"]),
                         sorted(["(vacío)", "NA", "-", "NULL", "nan", "na"]))
        # la coma decimal da el MISMO número que el punto
        punto = _evaluar(_csv(self.filas[6:], ["p", "y"])).informe
        sin6 = _evaluar("\n".join([lineas[0]] + lineas[7:]) + "\n").informe
        self.assertAlmostEqual(_auc(punto), _auc(sin6), places=9)

    def test_mezclar_coma_y_punto_en_una_columna_se_rechaza_no_se_adivina(self):
        lineas = _csv(self.filas, ["p", "y"], sep=";").splitlines()
        lineas[1] = lineas[1].replace(".", ",", 1)       # UNA celda con coma, el resto con punto
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar("\n".join(lineas) + "\n")
        self.assertEqual(cm.exception.clave, "puntuacion_decimal_mezclado")

    def test_un_texto_en_la_columna_de_puntuacion_se_rechaza_diciendo_cuales(self):
        lineas = _csv(self.filas, ["p", "y"]).splitlines()
        lineas[2] = "alta,1"
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar("\n".join(lineas) + "\n")
        self.assertEqual(cm.exception.clave, "puntuacion_no_numerica")
        self.assertIn("alta", cm.exception.es)

    def test_windows_1252_se_lee_y_se_dice_las_tildes_sobreviven(self):
        texto = _csv(self.filas, ["p", "y"], sep=";", coma=True, si="Sí", no="No")
        inf = _evaluar(texto.encode("cp1252"), evento="Sí").informe
        self.assertEqual(inf["entrada"]["codificacion"], "windows-1252")
        self.assertEqual(inf["desenlace"]["evento"], "Sí")
        utf = _evaluar(texto.encode("utf-8"), evento="Sí").informe
        self.assertEqual(utf["entrada"]["codificacion"], "utf-8")
        self.assertEqual(_auc(inf), _auc(utf))

    def test_el_bom_de_excel_se_quita(self):
        crudo = b"\xef\xbb\xbf" + _csv(self.filas, ["p", "y"]).encode("utf-8")
        inf = _evaluar(crudo).informe
        self.assertTrue(inf["entrada"]["bom"])
        self.assertEqual(inf["entrada"]["columnas"][0], "p")

    def test_el_indice_de_fila_sin_nombre_de_R_se_reconoce_se_dice_y_los_espacios_se_recortan(self):
        # write.csv: cabecera de N nombres, CADA fila con N+1 campos (el primero, el índice) y
        # valores con espacios delante. Medido con support2.csv de hbiostat.
        filas = self.filas[:600]
        cab = "p,y"
        cuerpo = [f"{i + 1}, {round(f['p'], 6)},  {f['y']}" for i, f in enumerate(filas)]
        inf = _evaluar("\n".join([cab] + cuerpo) + "\n").informe
        self.assertTrue(inf["entrada"]["indice_sin_nombre"])
        self.assertTrue(inf["entrada"]["indice_sin_nombre_texto"]["es"])
        self.assertEqual(inf["entrada"]["columnas"], ["p", "y"])
        self.assertEqual(inf["entrada"]["filas_malformadas_fuera"], 0)
        limpio = _evaluar(_csv(filas, ["p", "y"])).informe
        self.assertEqual(_auc(inf), _auc(limpio))
        self.assertFalse(limpio["entrada"]["indice_sin_nombre"])

    def test_los_espacios_alrededor_de_cada_valor_y_de_cada_nombre_de_columna_se_recortan(self):
        filas = self.filas[:600]
        lineas = [" id , p ,  y "]
        for i, f in enumerate(filas):
            lineas.append(f"  P{i % 200} ,  {round(f['p'], 6)} ,  {f['y']}  " if i % 2
                          else f"P{i % 200},{round(f['p'], 6)},{f['y']}")
        inf = _evaluar("\n".join(lineas) + "\n", columna_de_paciente="id").informe
        self.assertEqual(inf["entrada"]["columnas"], ["id", "p", "y"])
        # «  P5 » y «P5» son el mismo paciente: 200 unidades, no 400
        self.assertEqual(inf["puntuaciones"][0]["filas"]["unidades"], 200)

    def test_si_solo_algunas_filas_traen_un_campo_de_mas_no_es_un_indice_se_excluyen_y_se_cuentan(self):
        lineas = _csv(self.filas[:400], ["p", "y"]).splitlines()
        for k in (5, 6, 7):
            lineas[k] = "9," + lineas[k]
        inf = _evaluar("\n".join(lineas) + "\n").informe
        self.assertFalse(inf["entrada"]["indice_sin_nombre"])
        self.assertEqual(inf["entrada"]["filas_malformadas_fuera"], 3)
        self.assertEqual(inf["puntuaciones"][0]["filas"]["usadas"], 397)
        self.assertIn("filas_malformadas_fuera", [a["id"] for a in inf["avisos"]])

    def test_una_sola_clase_en_el_desenlace_deja_las_metricas_sin_valor_y_con_motivo_nunca_cero(self):
        texto = _csv(self.filas[:200], ["p", "y"], si="Sí", no="Sí")     # todos son evento
        res = _evaluar(texto, evento="Sí").informe["puntuaciones"][0]
        auroc = res["discriminacion"]["auroc"]
        self.assertIsNone(auroc["valor"])
        self.assertTrue(auroc["no_definido"]["es"] and auroc["no_definido"]["en"])
        self.assertIsNone(auroc["ic"])
        self.assertIsNone(res["calibracion"]["intercepto"]["valor"])
        self.assertTrue(res["calibracion"]["intercepto"]["no_definido"])


# ---------------------------------------------------------------------------
# Diseño, parámetros, textos
# ---------------------------------------------------------------------------

class DisenoYParametrosTest(DatosBase):
    def test_con_identificador_de_paciente_el_remuestreo_es_por_paciente(self):
        filas = [dict(f, id=f"P{i % 400}") for i, f in enumerate(self.filas)]
        inf = _evaluar(_csv(filas, ["id", "p", "y"]), columna_de_paciente="id").informe
        self.assertEqual(inf["remuestreo"]["diseno"], "groups")
        ic = inf["puntuaciones"][0]["discriminacion"]["auroc"]["ic"]
        self.assertEqual(ic["unidad_de_remuestreo"], "unit")
        self.assertEqual(inf["puntuaciones"][0]["filas"]["unidades"], 400)
        iid = _evaluar(_csv(filas, ["id", "p", "y"])).informe
        self.assertEqual(iid["remuestreo"]["diseno"], "iid")

    def test_las_remuestras_y_la_semilla_quedan_escritas_en_el_informe(self):
        inf = _evaluar(_csv(self.filas, ["p", "y"]), remuestras=55, semilla=9).informe
        self.assertEqual(inf["remuestreo"]["remuestras"], 55)
        self.assertEqual(inf["remuestreo"]["semilla"], 9)
        self.assertEqual(inf["puntuaciones"][0]["discriminacion"]["auroc"]["ic"]["remuestras"], 55)

    def test_menos_de_20_remuestras_se_rechaza_y_misma_semilla_mismo_informe(self):
        with self.assertRaises(EvaluacionRechazada):
            _evaluar(_csv(self.filas, ["p", "y"]), remuestras=5)
        a = _evaluar(_csv(self.filas, ["p", "y"])).informe
        b = _evaluar(_csv(self.filas, ["p", "y"])).informe
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_la_representatividad_no_tiene_valor_por_omision(self):
        with self.assertRaises(TypeError):
            evaluar_puntuacion(_csv(self.filas, ["p", "y"]), columna_desenlace="y",
                               valor_del_evento="1", **R2,
                               puntuaciones=[Decl("p", "probabilidad", ALTO)])
        with self.assertRaises(EvaluacionRechazada):
            _evaluar(_csv(self.filas, ["p", "y"]), representa="si")

    def test_la_curva_de_decision_sale_con_los_umbrales_declarados_y_no_se_inventa_un_barrido(self):
        csv = _csv(self.filas, ["p", "y"])
        sin = _evaluar(csv).informe["puntuaciones"][0]["curva_de_decision"]
        self.assertFalse(sin["declarada"])
        con = _evaluar(csv, umbrales_de_decision=[0.2, 0.4]).informe["puntuaciones"][0]["curva_de_decision"]
        self.assertEqual([p["umbral"] for p in con["puntos"]], [0.2, 0.4])
        n = len(self.filas)
        ps = [(round(f["p"], 6), f["y"]) for f in self.filas]
        tp = sum(1 for p, y in ps if p >= 0.2 and y)
        fp = sum(1 for p, y in ps if p >= 0.2 and not y)
        esperado = tp / n - fp / n * (0.2 / 0.8)
        self.assertAlmostEqual(con["puntos"][0]["beneficio_neto"], esperado, places=12)
        self.assertEqual(con["puntos"][0]["tratar_a_nadie"], 0.0)
        for malo in ([1.0], [0.2, 0.2], [-0.1], [20]):
            with self.assertRaises(EvaluacionRechazada):
                _evaluar(csv, umbrales_de_decision=malo)

    def test_la_fiabilidad_no_publica_tramos_de_menos_de_30_filas(self):
        inf = _evaluar(_csv(self.filas[:200], ["p", "y"])).informe
        bins = inf["puntuaciones"][0]["calibracion"]["fiabilidad"]["bins"]
        pequenos = [b for b in bins if 0 < b["n"] < 30]
        self.assertTrue(pequenos, "el caso de la prueba tiene que tener algún tramo pequeño")
        for b in pequenos:
            self.assertIsNone(b["tasa_observada"])
            self.assertIsNone(b["eventos"])
            self.assertTrue(b["motivo"]["es"])
        self.assertTrue(any(b["tasa_observada"] is not None for b in bins))


class TextosBilingueTest(unittest.TestCase):
    def test_cada_texto_tiene_los_mismos_huecos_en_castellano_y_en_ingles(self):
        for clave, plantillas in MOTIVOS.items():
            self.assertEqual(set(plantillas), {"es", "en"}, clave)
            self.assertEqual(huecos_de(plantillas["es"]), huecos_de(plantillas["en"]), clave)

    def test_todo_lo_que_dice_el_informe_esta_en_los_dos_idiomas(self):
        filas = _filas(400)
        inf = _evaluar(_csv(filas, ["p", "pts", "y", "sexo"], ausentes={"p": set(range(0, 80))},
                            ausente="NA"),
                       puntuaciones=[Decl("p", "probabilidad", BAJO, umbral=0.3),
                                     Decl("pts", "puntuacion", ALTO, umbral=5)],
                       representa=False, columnas_de_subgrupo=["sexo"]).informe

        def recorre(x, ruta=""):
            if isinstance(x, dict):
                if set(x) >= {"es", "en"} and isinstance(x["es"], str):
                    self.assertTrue(x["es"].strip() and x["en"].strip(), ruta)
                    self.assertNotEqual(x["es"], x["en"], ruta)
                for k, v in x.items():
                    recorre(v, f"{ruta}/{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    recorre(v, f"{ruta}[{i}]")
        recorre(inf)

    def test_el_informe_dice_siempre_lo_que_no_es_y_la_paradoja_solo_con_umbral_de_uso(self):
        csv = _csv(self.filas_(), ["p", "y"])
        sin = _evaluar(csv).informe
        con = _evaluar(csv, puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)]).informe
        self.assertEqual([t["id"] for t in sin["lo_que_no_es"]],
                         ["no_es_validacion_clinica", "no_es_anonimizacion",
                          "depende_del_registro"])
        self.assertEqual([t["id"] for t in con["lo_que_no_es"]],
                         ["no_es_validacion_clinica", "paradoja_del_tratamiento",
                          "no_es_anonimizacion", "depende_del_registro"])
        self.assertIn("validación para uso clínico", con["lo_que_no_es"][0]["es"])

    @staticmethod
    def filas_():
        return _filas(300)

    def test_ninguna_frase_dice_bueno_ni_malo(self):
        for clave, plantillas in MOTIVOS.items():
            for idioma, t in plantillas.items():
                for adjetivo in (" bueno", " buena", " malo", " mala", " good ", " bad ",
                                 "excelente", "aceptable"):
                    # «que la puntuación sea buena» (lo que el recibo NO atestigua) es una negación
                    if clave == "recibo_no_atestigua":
                        continue
                    self.assertNotIn(adjetivo, t.lower(), f"{clave}/{idioma}")


# ---------------------------------------------------------------------------
# El informe no lleva filas
# ---------------------------------------------------------------------------

class SinFilasTest(DatosBase):
    def test_el_informe_y_el_recibo_no_llevan_ninguna_fila_solo_agregados(self):
        filas = [dict(f, id=f"PACIENTE-{i:05d}") for i, f in enumerate(self.filas)]
        r = _evaluar(_csv(filas, ["id", "p", "y"]), columna_de_paciente="id",
                     clave=b"k" * 16, creado_en="2026-10-23T12:00:00Z")
        volcado = json.dumps(r.informe) + json.dumps(r.recibo) + json.dumps(r.sobre)
        self.assertNotIn("PACIENTE-", volcado)
        self.assertNotIn(repr(round(filas[0]["p"], 6)), volcado)


# ---------------------------------------------------------------------------
# Referencia independiente: scikit-learn
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAY_SKLEARN, "scikit-learn solo vive en las pruebas")
class ContraScikitLearnTest(DatosBase):
    TOL = 1e-9

    def setUp(self):
        self.p = [round(f["p"], 6) for f in self.filas]
        self.y = [f["y"] for f in self.filas]
        self.res = _evaluar(_csv(self.filas, ["p", "y"])).informe["puntuaciones"][0]

    def test_auroc_coincide_con_sklearn(self):
        self.assertAlmostEqual(self.res["discriminacion"]["auroc"]["valor"],
                               roc_auc_score(self.y, self.p), delta=self.TOL)

    def test_average_precision_coincide_con_sklearn(self):
        self.assertAlmostEqual(self.res["discriminacion"]["average_precision"]["valor"],
                               average_precision_score(self.y, self.p), delta=self.TOL)

    def test_brier_coincide_con_sklearn(self):
        self.assertAlmostEqual(self.res["calibracion"]["brier"]["valor"],
                               brier_score_loss(self.y, self.p), delta=self.TOL)

    def test_la_curva_de_fiabilidad_coincide_con_sklearn_en_los_tramos_con_30_filas_o_mas(self):
        tasa, media = calibration_curve(np.array(self.y), np.array(self.p), n_bins=10,
                                        strategy="uniform")
        de_sklearn = {round(m, 9): t for m, t in zip(media, tasa)}
        bins = [b for b in self.res["calibracion"]["fiabilidad"]["bins"] if b["n"] >= 30]
        self.assertGreaterEqual(len(bins), 4)
        for b in bins:
            self.assertIn(round(b["probabilidad_media"], 9), de_sklearn)
            self.assertAlmostEqual(b["tasa_observada"],
                                   de_sklearn[round(b["probabilidad_media"], 9)], delta=self.TOL)

    def test_el_auroc_de_una_escala_con_empates_tambien_coincide(self):
        r = _evaluar(_csv(self.filas, ["pts", "y"]),
                     puntuaciones=[Decl("pts", "puntuacion", ALTO)]).informe
        self.assertAlmostEqual(_auc(r), roc_auc_score(self.y, [f["pts"] for f in self.filas]),
                               delta=self.TOL)


# ---------------------------------------------------------------------------
# La curva ROC (137-C2): vértices REALES, adelgazados, solo agregados
# ---------------------------------------------------------------------------

def _trapecio(curva):
    return sum((b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2 for a, b in zip(curva, curva[1:]))


class CurvaRocTest(DatosBase):
    def _curva(self, columna="p", tipo="probabilidad", direccion=ALTO):
        r = _evaluar(_csv(self.filas, [columna, "y"]),
                     puntuaciones=[Decl(columna, tipo, direccion)]).informe["puntuaciones"][0]
        return r["discriminacion"]["curva_roc"], _auc({"puntuaciones": [r]})

    def test_la_curva_roc_tiene_los_extremos_y_no_pasa_de_40_puntos(self):
        curva, _ = self._curva()
        self.assertEqual(curva[0], {"fpr": 0.0, "tpr": 0.0})
        self.assertEqual(curva[-1], {"fpr": 1.0, "tpr": 1.0})
        self.assertLessEqual(len(curva), 40)
        self.assertGreaterEqual(len(curva), 30)              # 1.500 valores distintos: se adelgaza
        for a, b in zip(curva, curva[1:]):                    # y es una curva: no baja
            self.assertLessEqual(a["fpr"], b["fpr"])
            self.assertLessEqual(a["tpr"], b["tpr"])
        self.assertEqual({tuple(sorted(p)) for p in curva}, {("fpr", "tpr")})   # dos números

    def test_el_auroc_trapezoidal_de_los_puntos_esta_a_menos_de_0_01_del_del_nucleo(self):
        for columna, tipo in (("p", "probabilidad"), ("pts", "puntuacion")):
            curva, auc = self._curva(columna, tipo)
            self.assertAlmostEqual(_trapecio(curva), auc, delta=0.01, msg=columna)

    def test_con_pocos_valores_distintos_la_curva_es_exacta_sin_adelgazar(self):
        curva, auc = self._curva("pts", "puntuacion")        # una escala de 0 a 10 con empates
        self.assertLessEqual(len(curva), 12)
        self.assertAlmostEqual(_trapecio(curva), auc, delta=1e-9)

    def test_una_direccion_inversa_dibuja_la_curva_de_la_puntuacion_orientada(self):
        curva, auc = self._curva("pts", "puntuacion", BAJO)
        self.assertAlmostEqual(_trapecio(curva), auc, delta=1e-9)
        self.assertLess(auc, 0.5)                             # la escala va a favor: al revés, baja

    @unittest.skipUnless(_HAY_SKLEARN, "scikit-learn solo vive en las pruebas")
    def test_cada_punto_es_un_vertice_de_la_curva_de_sklearn_nada_se_inventa(self):
        from sklearn.metrics import roc_curve
        p = [round(f["p"], 6) for f in self.filas]
        y = [f["y"] for f in self.filas]
        fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
        de_sklearn = {(round(a, 9), round(b, 9)) for a, b in zip(fpr, tpr)}
        curva, _ = self._curva()
        for punto in curva:
            self.assertIn((round(punto["fpr"], 9), round(punto["tpr"], 9)), de_sklearn)

    def test_la_curva_no_lleva_filas(self):
        filas = [dict(f, id=f"PACIENTE-{i:05d}") for i, f in enumerate(self.filas)]
        r = _evaluar(_csv(filas, ["id", "p", "y"]), columna_de_paciente="id")
        volcado = json.dumps(r.informe["puntuaciones"][0]["discriminacion"]["curva_roc"])
        self.assertNotIn("PACIENTE-", volcado)
        self.assertNotIn(repr(round(filas[0]["p"], 6)), volcado)


# ---------------------------------------------------------------------------
# El recibo: HMAC con el mecanismo de `attest`, nunca por encima de A1
# ---------------------------------------------------------------------------

class ReciboTest(DatosBase):
    CLAVE = bytes.fromhex("aa" * 16)

    def _firmado(self, **kw):
        return _evaluar(_csv(self.filas, ["p", "y"]), clave=self.CLAVE,
                        creado_en="2026-10-23T12:00:00Z", **kw)

    def test_firmado_es_a1_y_nunca_sube_sin_firma_es_a0(self):
        sin = _evaluar(_csv(self.filas, ["p", "y"]))
        con = self._firmado()
        self.assertEqual(sin.nivel_de_garantia, "A0")
        self.assertIsNone(sin.sobre)
        self.assertEqual(con.nivel_de_garantia, "A1")
        self.assertEqual(nivel_del_recibo(con.sobre), "A1")
        self.assertEqual(con.recibo["assurance_ceiling"], "A1")
        for clave in ("replay_reference", "reproduce", "package", "package_sha256"):
            self.assertNotIn(clave, con.recibo["evidence"])

    def test_la_firma_verifica_con_la_clave_y_se_rompe_si_se_toca_el_recibo_o_se_cambia_la_clave(self):
        con = self._firmado()
        self.assertTrue(verificar_recibo(con.sobre, clave=self.CLAVE)["ok"])
        self.assertFalse(verificar_recibo(con.sobre, clave=b"otra-clave-xxxxx")["ok"])
        import base64
        contenido = json.loads(base64.b64decode(con.sobre["payload"]))
        contenido["parameters"]["resamples"] = 9999
        falso = dict(con.sobre, payload=base64.b64encode(
            json.dumps(contenido, sort_keys=True).encode()).decode())
        self.assertFalse(verificar_recibo(falso, clave=self.CLAVE)["ok"])

    def test_el_recibo_lleva_huella_declaraciones_semilla_y_versiones(self):
        crudo = _csv(self.filas, ["p", "y"]).encode("utf-8")
        con = evaluar_puntuacion(
            crudo, columna_desenlace="y", valor_del_evento="1",
            puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3, origen="Proveedor X v2")],
            muestra_representativa=False, remuestras=REM, semilla=77, **R2,
            clave=self.CLAVE, creado_en="2026-10-23T12:00:00Z", nombre_del_fichero="datos.csv")
        r = con.recibo
        self.assertEqual(r["dataset"]["digest"], "sha256:" + hashlib.sha256(crudo).hexdigest())
        self.assertEqual(r["dataset"]["name"], "datos.csv")
        self.assertEqual(r["parameters"]["seed"], 77)
        self.assertEqual(r["parameters"]["resamples"], REM)
        d = r["declarations"]
        self.assertEqual(d["scores"][0]["origen"], "Proveedor X v2")
        self.assertEqual(d["scores"][0]["umbral"], 0.3)
        self.assertEqual(d["event_value"], "1")
        self.assertFalse(d["sample_represents_population"])
        self.assertEqual(r["environment"]["metric_catalog_digest"],
                         con.informe["versiones"]["catalogo_de_metricas"])
        self.assertTrue(r["environment"]["matrixai_core"])
        self.assertEqual(r["models"][0]["provenance"][:9], "external:")
        # y compromete el informe: el digest del informe está en el recibo
        from matrixai.pipelines.canonical import jcs_bytes
        self.assertEqual(r["output"]["output_digests"]["report"],
                         hashlib.sha256(jcs_bytes(con.informe)).hexdigest())

    def test_firmar_sin_fecha_se_rechaza_la_funcion_no_lee_el_reloj(self):
        with self.assertRaises(EvaluacionRechazada) as cm:
            _evaluar(_csv(self.filas, ["p", "y"]), clave=self.CLAVE)
        self.assertEqual(cm.exception.clave, "recibo_sin_fecha")

    def test_la_huella_cambia_si_cambia_una_sola_celda(self):
        a = _evaluar(_csv(self.filas, ["p", "y"])).informe["entrada"]["huella_csv"]
        texto = _csv(self.filas, ["p", "y"])
        b = _evaluar(texto.replace(",1\n", ",0\n", 1)).informe["entrada"]["huella_csv"]
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# Ronda de rigor del 24-10: R2 (qué se predice y en qué plazo), dónde se calcula, R3, R4, R5
# ---------------------------------------------------------------------------

class R2DeclaracionesTest(DatosBase):
    def _csv(self):
        return _csv(self.filas, ["p", "pts", "y"])

    def _llamar(self, **cambios):
        kw = dict(columna_desenlace="y", valor_del_evento="1",
                  puntuaciones=[Decl("p", "probabilidad", ALTO)], muestra_representativa=True,
                  remuestras=REM, **R2)
        kw.update(cambios)
        for borrar in [k for k, v in kw.items() if v is _FALTA]:
            del kw[borrar]
        return evaluar_puntuacion(self._csv(), **kw)

    def test_R2_sin_horizonte_momento_o_dondese_calcula_no_hay_evaluacion(self):
        for campo in ("horizonte_del_desenlace", "momento_de_la_puntuacion", "donde_se_calcula"):
            with self.assertRaises(TypeError, msg=campo):
                self._llamar(**{campo: _FALTA})

    def test_R2_un_horizonte_o_un_momento_vacio_o_que_no_es_texto_se_rechaza(self):
        for campo in ("horizonte_del_desenlace", "momento_de_la_puntuacion"):
            for malo in ("", "   ", None, 30, "x" * 201):
                with self.assertRaises(EvaluacionRechazada, msg=f"{campo}={malo!r}") as cm:
                    self._llamar(**{campo: malo})
                self.assertEqual(cm.exception.clave, "texto_obligatorio_vacio")
                self.assertIn(campo, cm.exception.es)
                self.assertIn(campo, cm.exception.en)

    def test_R2_casos_en_su_desarrollo_es_obligatorio_y_de_vocabulario_cerrado(self):
        with self.assertRaises(TypeError):
            _Decl("p", "probabilidad", ALTO)
        for malo in ("", "tal vez", "yes", None, True):
            with self.assertRaises(EvaluacionRechazada, msg=repr(malo)) as cm:
                _Decl("p", "probabilidad", ALTO, casos_en_su_desarrollo=malo)
            self.assertEqual(cm.exception.clave, "declaracion_fuera_del_vocabulario")
        for bueno in ("si", "no", "no_se_sabe"):
            self.assertEqual(Decl("p", "probabilidad", ALTO,
                                  casos_en_su_desarrollo=bueno).casos_en_su_desarrollo, bueno)

    def test_R2_las_tres_declaraciones_van_al_informe_y_al_recibo_y_el_recibo_las_compromete(self):
        clave = b"k" * 32
        a = self._llamar(horizonte_del_desenlace="a 30 días", momento_de_la_puntuacion="día 3",
                         puntuaciones=[Decl("p", "probabilidad", ALTO, casos_en_su_desarrollo="si")],
                         clave=clave, creado_en="2026-10-24T10:00:00Z")
        self.assertEqual(a.informe["alcance"], {"horizonte_del_desenlace": "a 30 días",
                                                "momento_de_la_puntuacion": "día 3"})
        self.assertEqual(a.informe["puntuaciones"][0]["declaracion"]["casos_en_su_desarrollo"], "si")
        d = a.recibo["declarations"]
        self.assertEqual((d["outcome_horizon"], d["score_timing"]), ("a 30 días", "día 3"))
        self.assertEqual(d["scores"][0]["casos_en_su_desarrollo"], "si")
        self.assertTrue(verificar_recibo(a.sobre, clave=clave)["ok"])
        # cambiar CUALQUIERA de las tres cambia el identificador del recibo
        base = self._llamar(horizonte_del_desenlace="a 30 días", momento_de_la_puntuacion="día 3",
                            puntuaciones=[Decl("p", "probabilidad", ALTO, casos_en_su_desarrollo="si")])
        for cambio in (dict(horizonte_del_desenlace="a 6 meses"),
                       dict(momento_de_la_puntuacion="al ingreso"),
                       dict(puntuaciones=[Decl("p", "probabilidad", ALTO,
                                               casos_en_su_desarrollo="no_se_sabe")])):
            kw = dict(horizonte_del_desenlace="a 30 días", momento_de_la_puntuacion="día 3",
                      puntuaciones=[Decl("p", "probabilidad", ALTO, casos_en_su_desarrollo="si")])
            kw.update(cambio)
            otro = self._llamar(**kw)
            self.assertNotEqual(otro.recibo["receipt_id"], base.recibo["receipt_id"], cambio)

    def test_R2_con_si_o_no_se_sabe_el_informe_dice_que_no_es_una_validacion_externa(self):
        for valor, frase in (("si", "took part in the development"),
                             ("no_se_sabe", "not known whether")):
            r = self._llamar(puntuaciones=[Decl("p", "probabilidad", ALTO,
                                                casos_en_su_desarrollo=valor)])
            v = r.informe["puntuaciones"][0]["validacion_externa"]
            self.assertFalse(v["es_validacion_externa"])
            self.assertFalse(v["comprobada_por_matrixai"])
            self.assertIn("NO es una validación externa", v["nota"]["es"])
            self.assertIn("NOT an external validation", v["nota"]["en"])
            self.assertIn(frase, v["nota"]["en"])
            ids = [x["id"] for x in r.informe["lo_que_no_es"]]
            self.assertIn("no_es_validacion_externa", ids)
            self.assertIn("No es una validación externa de: p", r.informe["lo_que_no_es"][1]["es"])

    def test_R2_con_no_no_se_afirma_validacion_externa_solo_que_esta_declarado_sin_comprobar(self):
        r = self._llamar(puntuaciones=[Decl("p", "probabilidad", ALTO, casos_en_su_desarrollo="no")])
        v = r.informe["puntuaciones"][0]["validacion_externa"]
        self.assertTrue(v["es_validacion_externa"])
        self.assertIn("MatrixAI no lo comprueba", v["nota"]["es"])
        self.assertIn("no es una validación para uso clínico", v["nota"]["es"])
        self.assertNotIn("no_es_validacion_externa", [x["id"] for x in r.informe["lo_que_no_es"]])

    def test_R2_con_dos_puntuaciones_solo_se_nombra_la_que_no_es_externa(self):
        r = self._llamar(puntuaciones=[
            Decl("p", "probabilidad", ALTO, casos_en_su_desarrollo="no"),
            Decl("pts", "puntuacion", ALTO, casos_en_su_desarrollo="si")])
        texto = next(x for x in r.informe["lo_que_no_es"] if x["id"] == "no_es_validacion_externa")
        self.assertIn("pts", texto["es"])
        self.assertNotIn(" p.", texto["es"])
        self.assertNotIn("p, pts", texto["es"])
        self.assertTrue(r.informe["puntuaciones"][0]["validacion_externa"]["es_validacion_externa"])
        self.assertFalse(r.informe["puntuaciones"][1]["validacion_externa"]["es_validacion_externa"])


class DondeSeCalculaTest(DatosBase):
    def _llamar(self, donde):
        return _evaluar(_csv(self.filas, ["p", "y"]), donde_se_calcula=donde,
                        clave=b"k" * 32, creado_en="2026-10-24T10:00:00Z")

    def test_donde_se_calcula_es_obligatorio_y_de_vocabulario_cerrado(self):
        with self.assertRaises(TypeError):
            evaluar_puntuacion(_csv(self.filas, ["p", "y"]), columna_desenlace="y",
                               valor_del_evento="1", puntuaciones=[Decl("p", "probabilidad", ALTO)],
                               muestra_representativa=True, horizonte_del_desenlace="h",
                               momento_de_la_puntuacion="m")
        for malo in ("", "nube", None, "Local", True):
            with self.assertRaises(EvaluacionRechazada, msg=repr(malo)) as cm:
                self._llamar(malo)
            self.assertEqual(cm.exception.clave, "declaracion_fuera_del_vocabulario")
            self.assertIn("donde_se_calcula", cm.exception.es)

    def test_en_local_el_informe_dice_que_el_csv_no_salio_del_equipo(self):
        d = self._llamar("local").informe["donde_se_calcula"]
        self.assertEqual(d["valor"], "local")
        self.assertIn("Se calculó en local", d["texto"]["es"])
        self.assertIn("el CSV no salió de él", d["texto"]["es"])
        self.assertIn("calculated locally", d["texto"]["en"])
        self.assertNotIn("servidor", d["texto"]["es"])

    def test_en_el_servidor_de_la_demo_el_informe_dice_la_verdad_y_no_dice_local(self):
        r = self._llamar("servidor_de_la_demo")
        d = r.informe["donde_se_calcula"]
        self.assertEqual(d["valor"], "servidor_de_la_demo")
        self.assertEqual(d["texto"]["es"], "Se calculó en el servidor de la demo con el CSV que "
                                           "enviaste; el CSV no se guardó.")
        self.assertEqual(d["texto"]["en"], "It was calculated on the demo server with the CSV "
                                           "you sent; the CSV was not stored.")
        self.assertNotIn("en local", d["texto"]["es"])
        self.assertNotIn("locally", d["texto"]["en"])

    def test_donde_se_calcula_va_al_recibo_y_lo_cambia_todo(self):
        a, b = self._llamar("local"), self._llamar("servidor_de_la_demo")
        self.assertEqual(a.recibo["environment"]["computed_where"], "local")
        self.assertEqual(b.recibo["environment"]["computed_where"], "servidor_de_la_demo")
        self.assertNotEqual(a.recibo["output"]["output_digests"]["report"],
                            b.recibo["output"]["output_digests"]["report"])
        self.assertTrue(verificar_recibo(b.sobre, clave=b"k" * 32)["ok"])


class RondaDeRigorNucleoTest(DatosBase):
    def test_R3_el_rotulo_de_alcance_depende_de_la_representatividad_en_cada_seccion(self):
        csv = _csv(self.filas, ["p", "y"])
        for representa in (True, False):
            p = _evaluar(csv, representa=representa, umbrales_de_decision=[0.2, 0.4],
                         puntuaciones=[Decl("p", "probabilidad", ALTO, umbral=0.3)]
                         ).informe["puntuaciones"][0]
            for seccion in (p["calibracion"], p["al_umbral"]["beneficio_neto"], p["curva_de_decision"]):
                self.assertEqual(seccion["rotulo_de_alcance"] is not None, not representa)
                if not representa:
                    self.assertEqual(seccion["rotulo_de_alcance"]["rotulo"]["en"],
                                     "describe the supplied sample, not the population of use")

    def test_R4_la_cifra_prometida_lleva_las_dos_cifras_y_la_advertencia_nunca_un_veredicto(self):
        p = _evaluar(_csv(self.filas, ["p", "y"]),
                     puntuaciones=[Decl("p", "probabilidad", ALTO, cifra_prometida=0.9)]
                     ).informe["puntuaciones"][0]
        c = p["cifra_prometida"]
        self.assertEqual(c["referencia_publicada"], 0.9)
        self.assertEqual(c["estimacion_local"], p["discriminacion"]["auroc"]["valor"])
        self.assertIn("no demuestra que la cifra original esté mal", c["advertencia"]["es"])
        self.assertIn("does not show that the original figure is wrong", c["advertencia"]["en"])
        self.assertIn(c["resultado"], ("dentro", "fuera"))

    def test_R5_el_auroc_de_un_grupo_con_evidencia_lleva_intervalo_y_el_pobre_ni_numero(self):
        filas = _filas(1200, semilla=4)
        for i, f in enumerate(filas):
            f["sala"] = "chica" if i < 12 else ("A" if i % 2 else "B")
        csv = _csv(filas, ["p", "y", "sala"])
        grupos = {g["valor"]: g for g in _evaluar(csv, columnas_de_subgrupo=["sala"]).informe[
            "puntuaciones"][0]["subgrupos"][0]["grupos"]}
        a = grupos["A"]["auroc"]
        self.assertLess(a["ic"]["bajo"], a["valor"])
        self.assertLess(a["valor"], a["ic"]["alto"])
        self.assertEqual(a["ic"]["remuestras"], REM)
        self.assertNotIn("auroc", grupos["chica"])
        self.assertNotIn("eventos", grupos["chica"])

    def test_la_version_del_informe_sube_con_el_esquema(self):
        r = _evaluar(_csv(self.filas, ["p", "y"]))
        self.assertEqual(r.informe["version_del_informe"], "2.2.0")
        self.assertEqual(r.recibo["environment"]["report_version"], "2.2.0")


class VeredictoDelNucleoTest(DatosBase):
    """La frase-veredicto de cada puntuación la compone el NÚCLEO con los números y sin umbrales
    inventados: «ordena» si el IC del AUROC excluye 0,5; «sobrestima/infraestima» si el IC de la
    calibración en grande excluye 0; si no, dice que no hay diferencia clara."""

    def _v(self, filas, columnas=("p", "y"), tipo="probabilidad", rem=200):
        r = _evaluar(_csv(filas, list(columnas)), remuestras=rem,
                     puntuaciones=[Decl(columnas[0], tipo, ALTO)])
        return r.informe["puntuaciones"][0]

    def test_sobrestima_si_el_ic_de_la_calibracion_en_grande_excluye_0_por_arriba(self):
        p = self._v(_filas(1500, semilla=1, desplazamiento=0.8))
        g = p["calibracion"]["en_grande"]
        self.assertGreater(g["ic"]["bajo"], 0)
        v = p["veredicto"]
        self.assertEqual((v["ordena"], v["riesgo"]), ("si", "sobrestima"))
        auc = p["discriminacion"]["auroc"]["valor"]
        pp = f'{g["valor"] * 100:.1f}'.replace(".", ",")
        self.assertEqual(v["texto"]["es"], f"Ordena mejor que el azar (AUROC {auc:.2f}); "
                                           f"sobrestima el riesgo en {pp} puntos porcentuales.".replace(
                                               f"{auc:.2f}", f"{auc:.2f}".replace(".", ",")))
        self.assertIn("overestimates the risk by", v["texto"]["en"])
        self.assertIn(f'{g["valor"] * 100:.1f} percentage points', v["texto"]["en"])

    def test_infraestima_si_el_ic_excluye_0_por_debajo(self):
        p = self._v(_filas(1500, semilla=1, desplazamiento=-0.8))
        self.assertLess(p["calibracion"]["en_grande"]["ic"]["alto"], 0)
        self.assertEqual(p["veredicto"]["riesgo"], "infraestima")
        self.assertIn("infraestima el riesgo en", p["veredicto"]["texto"]["es"])
        self.assertNotIn("-", p["veredicto"]["texto"]["es"].split("riesgo en")[1][:6])   # magnitud, sin signo

    def test_sin_diferencia_clara_si_el_ic_incluye_0_y_no_se_dice_ni_sobre_ni_infra(self):
        p = self._v(_filas(1500, semilla=1, desplazamiento=0.0))
        ic = p["calibracion"]["en_grande"]["ic"]
        self.assertLessEqual(ic["bajo"], 0)
        self.assertGreaterEqual(ic["alto"], 0)
        self.assertEqual(p["veredicto"]["riesgo"], "sin_diferencia")
        self.assertIn("no hay diferencia clara", p["veredicto"]["texto"]["es"])
        self.assertIn("there is no clear difference", p["veredicto"]["texto"]["en"])
        for prohibida in ("sobrestima", "infraestima", "overestimates", "underestimates"):
            self.assertNotIn(prohibida, p["veredicto"]["texto"]["es"] + p["veredicto"]["texto"]["en"])

    def test_una_escala_de_puntos_no_dice_nada_de_si_acierta_en_el_riesgo(self):
        p = self._v(_filas(1500), columnas=("pts", "y"), tipo="puntuacion")
        self.assertEqual(p["veredicto"]["riesgo"], "no_aplica")
        self.assertIn("no se mide si acierta en el riesgo", p["veredicto"]["texto"]["es"])
        self.assertIn("is not measured", p["veredicto"]["texto"]["en"])

    def test_ordena_se_dice_solo_si_el_ic_excluye_0_5_azar_y_al_reves_lo_dicen_aparte(self):
        filas = _filas(1500)
        r = random.Random(9)
        azar = [dict(f, p=r.random()) for f in filas]
        pa = self._v(azar)
        self.assertLessEqual(pa["discriminacion"]["auroc"]["ic"]["bajo"], 0.5)
        self.assertEqual(pa["veredicto"]["ordena"], "azar")
        self.assertIn("No se distingue del azar al ordenar", pa["veredicto"]["texto"]["es"])
        self.assertNotIn("Ordena mejor que el azar", pa["veredicto"]["texto"]["es"])
        reves = self._v([dict(f, p=1 - f["p"]) for f in filas])
        self.assertEqual(reves["veredicto"]["ordena"], "al_reves")
        self.assertIn("¿está bien declarada la dirección?", reves["veredicto"]["texto"]["es"])

    def test_ninguna_frase_lleva_adjetivos_ni_umbrales_inventados(self):
        for d in (0.8, -0.8, 0.0):
            v = self._v(_filas(1500, semilla=1, desplazamiento=d))["veredicto"]["texto"]
            for idioma, frase in v.items():
                for adjetivo in (" bien", " mal ", "buen", "mala", "good", "bad", "poor", "excelente"):
                    self.assertNotIn(adjetivo, frase.lower() + " ")
                self.assertTrue(frase.endswith("."))

    def test_el_veredicto_viaja_con_el_informe_y_comparte_la_version(self):
        r = _evaluar(_csv(self.filas, ["p", "pts", "y"]), puntuaciones=[
            Decl("p", "probabilidad", ALTO), Decl("pts", "puntuacion", ALTO)])
        for p in r.informe["puntuaciones"]:
            self.assertEqual(set(p["veredicto"]), {"ordena", "riesgo", "texto"})
            self.assertEqual(set(p["veredicto"]["texto"]), {"es", "en"})
        self.assertEqual(r.informe["version_del_informe"], "2.2.0")


_FALTA = object()      # «este argumento no se pasa»


# ---------------------------------------------------------------------------
# La orden de la terminal, de punta a punta
# ---------------------------------------------------------------------------

class CliTest(DatosBase):
    def _correr(self, d: Path, datos: bytes, *extra: str):
        (d / "datos.csv").write_bytes(datos)
        return subprocess.run(
            [sys.executable, "-m", "matrixai", "evaluate-score", str(d / "datos.csv"),
             "--outcome", "y", "--event", "Sí", "--represents-population", "yes",
             "--outcome-horizon", "durante el ingreso", "--score-timing", "al ingreso",
             "--score", "p", "--score-type", "probability", "--direction", "higher",
             "--score-development-cases", "no",
             "--threshold", "0,3", "--resamples", "30", *extra],
            capture_output=True, text=True, cwd=str(_RAIZ))

    def test_de_punta_a_punta_con_un_csv_de_excel_en_windows_1252_firmado(self):
        texto = _csv(self.filas, ["id", "p", "pts", "y"], sep=";", coma=True, si="Sí", no="No",
                     ausentes={"pts": {1, 2, 3}}, ausente="NA")
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            r = self._correr(d, texto.encode("cp1252"), "--key", "aa" * 16, "-o",
                             str(d / "informe.json"), "--score2", "pts", "--score2-type",
                             "score", "--score2-direction", "higher", "--score2-threshold",
                             "5", "--score2-development-cases", "unknown", "--patient-id", "id")
            self.assertEqual(r.returncode, 0, r.stderr)
            informe = json.loads((d / "informe.json").read_text(encoding="utf-8"))
            recibo = json.loads((d / "informe.receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(informe["entrada"]["codificacion"], "windows-1252")
            self.assertEqual(informe["entrada"]["separador"], ";")
            self.assertEqual(informe["desenlace"]["evento"], "Sí")
            self.assertEqual(informe["puntuaciones"][0]["lectura"]["decimal"], "coma")
            self.assertEqual(informe["puntuaciones"][0]["al_umbral"]["umbral"], 0.3)  # «0,3»
            self.assertEqual(informe["puntuaciones"][1]["filas"]["sin_puntuacion"], 3)
            self.assertTrue(informe["comparacion"]["posible"])
            self.assertEqual(informe["remuestreo"]["diseno"], "groups")
            # R2 y «dónde se calculó»: lo que se escribió en la línea de órdenes llega al informe
            self.assertEqual(informe["alcance"], {"horizonte_del_desenlace": "durante el ingreso",
                                                  "momento_de_la_puntuacion": "al ingreso"})
            self.assertEqual([p["declaracion"]["casos_en_su_desarrollo"]
                              for p in informe["puntuaciones"]], ["no", "no_se_sabe"])
            self.assertEqual(informe["donde_se_calcula"]["valor"], "local")   # la CLI corre aquí
            self.assertIn("NO es una validación externa", r.stderr)
            self.assertIn("Se calculó en local", r.stderr)
            self.assertEqual(nivel_del_recibo(recibo), "A1")
            self.assertTrue(verificar_recibo(recibo, clave=bytes.fromhex("aa" * 16))["ok"])
            self.assertIn("AUROC", r.stderr)
            self.assertIn("assurance: A1", r.stderr)
            self.assertIn("no es una validación", r.stderr)

    def test_sin_clave_el_recibo_sale_sin_firmar_y_lo_dice(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            r = self._correr(d, _csv(self.filas, ["p", "y"], si="Sí", no="No").encode("utf-8"),
                             "--locale", "en")
            self.assertEqual(r.returncode, 0, r.stderr)
            salida = json.loads(r.stdout)
            self.assertEqual(nivel_del_recibo(salida["receipt"]), "A0")
            self.assertIn("UNSIGNED", r.stderr)
            self.assertIn("not a validation", r.stderr)

    def test_un_rechazo_sale_con_codigo_1_y_su_motivo_en_el_idioma_pedido(self):
        filas = [dict(f) for f in self.filas]
        filas[2]["p"] = 1.7
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            datos = _csv(filas, ["p", "y"], si="Sí", no="No").encode("utf-8")
            es = self._correr(d, datos)
            en = self._correr(d, datos, "--locale", "en")
            self.assertEqual((es.returncode, en.returncode), (1, 1))
            self.assertIn("fuera de ese rango", es.stderr)
            self.assertIn("outside that range", en.stderr)
            self.assertEqual(es.stdout, "")

    def test_el_desenlace_sin_el_evento_declarado_se_rechaza_no_se_adivina(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "d.csv").write_text(_csv(self.filas, ["p", "y"], si="yes", no="no"),
                                     encoding="utf-8")
            r = subprocess.run(
                [sys.executable, "-m", "matrixai", "evaluate-score", str(d / "d.csv"),
                 "--outcome", "y", "--event", "Sí", "--represents-population", "no",
                 "--outcome-horizon", "durante el ingreso", "--score-timing", "al ingreso",
                 "--score", "p", "--score-type", "probability", "--direction", "higher",
                 "--score-development-cases", "no"],
                capture_output=True, text=True, cwd=str(_RAIZ))
            self.assertEqual(r.returncode, 1)
            self.assertIn("nunca se adivina", r.stderr)

    def test_las_opciones_obligatorias_no_tienen_valor_por_omision(self):
        r = subprocess.run([sys.executable, "-m", "matrixai", "evaluate-score", "x.csv",
                            "--outcome", "y", "--event", "1", "--score", "p"],
                           capture_output=True, text=True, cwd=str(_RAIZ))
        self.assertEqual(r.returncode, 2)
        for opcion in ("--represents-population", "--score-type", "--direction",
                       "--outcome-horizon", "--score-timing", "--score-development-cases"):
            self.assertIn(opcion, r.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class ValorPDeLaFaltaConcentradaTest(unittest.TestCase):
    """Visto en Chromium el 25-10: el aviso de falta concentrada decía «p = 0,0000». Un p más pequeño que lo que
    cuatro decimales dicen se escribe «p < 0,0001»; nunca un cero que el test exacto de Fisher no da."""

    def test_un_p_minusculo_se_escribe_menor_que_y_nunca_cero(self):
        from matrixai.estudio.evaluacion_de_puntuacion_textos import valor_p
        self.assertEqual(dict(valor_p(1e-9)), {"es": "p < 0,0001", "en": "p < 0.0001"})
        self.assertEqual(dict(valor_p(0.0056)), {"es": "p = 0,0056", "en": "p = 0.0056"})

    def test_el_aviso_real_no_dice_p_igual_a_cero(self):
        filas = ["evento,p"]
        for i in range(2000):
            ev = i % 3 == 0
            falta = (ev and i % 6 == 0) or (not ev and i % 97 == 0)
            filas.append(f"{'1' if ev else '0'},{'' if falta else round((i % 100) / 100, 2)}")
        r = evaluar_puntuacion("\n".join(filas) + "\n", columna_desenlace="evento", valor_del_evento="1",
                               muestra_representativa=True, horizonte_del_desenlace="a 30 días",
                               momento_de_la_puntuacion="al ingreso", donde_se_calcula="local",
                               puntuaciones=[_Decl(columna="p", tipo="probabilidad",
                                                                     direccion="mas_alto_mas_riesgo",
                                                                     casos_en_su_desarrollo="no")],
                               remuestras=20, semilla=137)
        avisos = [a for a in r.informe["puntuaciones"][0]["avisos"] if a["id"] == "falta_concentrada_en_un_desenlace"]
        self.assertEqual(len(avisos), 1, "el CSV planta una falta muy concentrada en los eventos")
        self.assertIn("p < 0,0001", avisos[0]["es"])
        self.assertIn("p < 0.0001", avisos[0]["en"])
        self.assertNotIn("0,0000", avisos[0]["es"])
