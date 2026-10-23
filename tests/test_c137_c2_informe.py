# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""137-C2 — el informe HTML de «¿Funciona aquí?» (es/en), con TRIPOD+AI de validación externa.

Los resultados son REALES: salen de `evaluar_puntuacion` sobre CSV sintéticos con las formas de
la gente (`Sí/No`, coma decimal no hace falta aquí: eso es del C1), con pocas remuestras para
que sea rápido. El informe se prueba por la salida, no por sus piezas: se parsea el HTML que
sale y se mira qué dice una persona que lo abre.

Dos cosas que conviene saber:

* **El inglés se prueba con DATOS en inglés.** Los nombres de columnas, orígenes y grupos los
  escribe quien evalúa y pueden ser de cualquier idioma; lo que se comprueba es que la
  PLANTILLA no deja una palabra del otro idioma. Por eso estos datos son neutros.
* **Un aserto negativo lo pasa un render en blanco**: cada «no aparece» va junto a un «sí
  aparece» sobre el mismo informe.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
import unittest
from html.parser import HTMLParser
from types import SimpleNamespace

from matrixai.estudio.evaluacion_de_puntuacion import (
    DeclaracionDePuntuacion as Decl,
    evaluar_puntuacion,
)
from matrixai.export import informe_de_puntuacion as modulo
from matrixai.export.expediente_clinico import POR_CLAVE, FALTA, MEDIDO, DECLARADO, ESTADOS
from matrixai.export.informe_de_puntuacion import (
    CLAVES_DEL_INVENTARIO_USADAS,
    informe_html,
    items_tripod,
    resolver_ruta,
)
from matrixai.export.informe_de_puntuacion_textos import IDIOMAS, TEXTOS, huecos_de, t

REM = 30
ALTO = "mas_alto_mas_riesgo"
RARO = "ZX9-RARO-4471"           # un identificador que NO puede aparecer en el informe


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _filas(n: int = 1000, semilla: int = 3, desplazamiento: float = 0.8) -> list[dict]:
    r = random.Random(semilla)
    filas = []
    for i in range(n):
        z = r.gauss(-0.5, 1.2)
        y = 1 if r.random() < _sig(z) else 0
        p = min(0.999, max(0.001, _sig(z + desplazamiento)))
        pts = max(0, min(10, round(1.5 * z + 5 + r.gauss(0, 1))))
        sala = "Small ward" if i < 12 else ("Ward A" if i % 3 else "Ward B")
        filas.append({"id": RARO if i == 7 else f"P{i:05d}", "score_a": p, "score_b": pts,
                      "outcome": "Yes" if y else "No",
                      "sex": "Female" if r.random() < 0.5 else "Male", "ward": sala})
    return filas


def _csv(filas: list[dict], columnas: list[str]) -> str:
    lineas = [",".join(columnas)]
    for f in filas:
        lineas.append(",".join(repr(round(f[c], 6)) if isinstance(f[c], float) else str(f[c])
                               for c in columnas))
    return "\n".join(lineas) + "\n"


_FILAS = _filas()
_COLS = ["id", "score_a", "score_b", "outcome", "sex", "ward"]
_CLAVE = b"k" * 32


def _evaluar(*, puntuaciones=None, representa=True, columnas=_COLS, filas=None, firmar=True,
             **kw):
    kw.setdefault("remuestras", REM)
    kw.setdefault("semilla", 137)
    texto = _csv(filas if filas is not None else _FILAS, columnas)
    return evaluar_puntuacion(
        texto, columna_desenlace="outcome", valor_del_evento="Yes",
        puntuaciones=puntuaciones,
        muestra_representativa=representa, nombre_del_fichero="sample.csv",
        actor="Internal Medicine", clave=_CLAVE if firmar else None,
        creado_en="2026-10-23T12:00:00Z", **kw)


def _dos(**kw):
    kw.setdefault("columnas_de_subgrupo", ("sex", "ward"))
    kw.setdefault("umbrales_de_decision", [0.1, 0.2, 0.3, 0.4])
    return _evaluar(puntuaciones=[
        Decl("score_a", "probabilidad", ALTO, umbral=0.3, origen="Vendor X v2",
             cifra_prometida=0.82),
        Decl("score_b", "puntuacion", ALTO, umbral=6, origen="Published scale (Author 2019)")],
        **kw)


# --- lo que ve una persona: texto y atributos del HTML --------------------------------------

class _Lector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.texto: list[str] = []
        self.atributos: list[tuple[str, str, str]] = []
        self._ignorar = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self._ignorar += 1
        for k, v in attrs:
            self.atributos.append((tag, k, v or ""))

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._ignorar -= 1

    def handle_data(self, data):
        if not self._ignorar:
            self.texto.append(data)


def _leer(doc: str) -> _Lector:
    lector = _Lector()
    lector.feed(doc)
    return lector


def _texto(doc: str) -> str:
    return " ".join(" ".join(_leer(doc).texto).split())


def _palabras_visibles(doc: str) -> list[str]:
    """Texto + valores de atributos que una persona o un lector de pantalla ve. Se descartan las
    clases y los ids/`data-*` (son nuestros, no se leen), no `aria-label`/`title`/`alt`."""
    lector = _leer(doc)
    piezas = list(lector.texto)
    piezas += [v for (_, k, v) in lector.atributos if k in ("aria-label", "title", "alt", "placeholder")]
    # los <title> dentro de SVG son hijos con texto, ya están en `texto`
    return re.findall(r"[^\W\d_]+", " ".join(piezas).lower())


def _fila(doc: str, contiene: str) -> str:
    for tr in re.findall(r"<tr.*?</tr>", doc, flags=re.S):
        if contiene in tr:
            return tr
    raise AssertionError(f"no hay ninguna fila con {contiene!r}")


def _svg(doc: str, tipo: str) -> str | None:
    m = re.search(rf'<svg data-chart="{tipo}".*?</svg>', doc, flags=re.S)
    return m.group(0) if m else None


def _con_informe(r, modificar):
    """Un resultado con el informe cambiado a mano: para probar qué hace el PINTADO ante un
    informe que el núcleo no produciría hoy (un valor ausente, un campo que aún no existe)."""
    informe = copy.deepcopy(r.informe)
    modificar(informe)
    return SimpleNamespace(informe=informe, recibo=r.recibo, sobre=r.sobre,
                           nivel_de_garantia=r.nivel_de_garantia)


ES_FUNCIONALES = {"el", "la", "del", "con", "que", "desde", "para", "una", "los", "las", "por",
                  "se", "sus", "al", "es", "un", "como", "pero", "sin", "sobre"}
EN_FUNCIONALES = {"the", "of", "and", "with", "for", "from", "are", "this", "that", "which",
                  "was", "were", "have", "has", "does", "than", "into", "over"}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _dos()
        cls.es = informe_html(cls.r, idioma="es")
        cls.en = informe_html(cls.r, idioma="en")


# ---------------------------------------------------------------------------------------------
# Las secciones, el idioma y la autonomía del documento
# ---------------------------------------------------------------------------------------------

class SeccionesTest(Base):
    def test_cada_seccion_aparece_en_castellano(self):
        txt = _texto(self.es)
        for frase in (
                "¿Funcionan aquí «score_a» y «score_b»?", "¿Ordena bien?", "¿Acierta en el riesgo?",
                "Al umbral de hoy", "¿Con quién falla?", "Avisos", "Gráficas",
                "Cómo ordena: AUROC con su intervalo", "Cuánto acierta en el riesgo",
                "Utilidad: curva de decisión", "Por subgrupos", "Una frente a otra",
                "Lo que este informe no es", "Cómo se hizo", "Recibo",
                "Ficha TRIPOD+AI de validación externa"):
            self.assertIn(frase, txt)

    def test_cada_seccion_aparece_en_ingles(self):
        txt = _texto(self.en)
        for frase in (
                "Do “score_a” and “score_b” work here?", "Does it rank well?",
                "Is the risk right?", "At today's threshold", "Who does it fail for?",
                "Warnings", "Charts", "How it ranks: AUROC with its interval",
                "How right the risk is", "Usefulness: decision curve", "By subgroup",
                "One against the other", "What this report is not", "How it was done", "Receipt",
                "TRIPOD+AI external-validation record"):
            self.assertIn(frase, txt)

    def test_el_documento_declara_su_idioma(self):
        self.assertIn('<html lang="es">', self.es)
        self.assertIn('<html lang="en">', self.en)

    def test_autocontenido_sin_javascript_ni_recursos_externos_ni_una_url(self):
        for doc in (self.es, self.en):
            self.assertNotIn("http", doc.lower())
            self.assertNotIn("<script", doc.lower())
            self.assertNotIn("<link", doc.lower())
            self.assertNotIn("@import", doc.lower())
            self.assertNotIn("url(", doc.lower())
            self.assertNotIn(" src=", doc.lower())
            self.assertNotIn(" href=", doc.lower())
            self.assertNotIn("onclick", doc.lower())
            self.assertTrue(doc.lstrip().startswith("<!doctype html>"))
            self.assertIn("<style>", doc)       # el CSS va en línea
            self.assertIn("@page{size:A4", doc)  # y se imprime en A4

    def test_en_ingles_no_queda_una_palabra_funcional_castellana_ni_en_atributos(self):
        palabras = _palabras_visibles(self.en)
        self.assertGreater(len(palabras), 400)       # que no sea un render en blanco
        self.assertIn("sensitivity", palabras)
        colados = sorted(set(palabras) & ES_FUNCIONALES)
        self.assertEqual(colados, [], f"palabras castellanas en el informe inglés: {colados}")
        # el inventario de TRIPOD y los motivos del núcleo también son de la plantilla:
        for palabra in ("el", "la", "del", "con", "que", "desde", "para", "una"):
            self.assertNotIn(palabra, palabras)

    def test_en_castellano_no_queda_una_palabra_funcional_inglesa_ni_en_atributos(self):
        palabras = _palabras_visibles(self.es)
        self.assertGreater(len(palabras), 400)
        self.assertIn("sensibilidad", palabras)
        colados = sorted(set(palabras) & EN_FUNCIONALES)
        self.assertEqual(colados, [], f"palabras inglesas en el informe castellano: {colados}")

    def test_los_atributos_de_las_graficas_estan_en_su_idioma(self):
        for doc, esperado in ((self.es, "Sensibilidad frente a falsas alarmas"),
                              (self.en, "Sensitivity against false alarms")):
            etiquetas = [v for (_, k, v) in _leer(doc).atributos if k == "aria-label"]
            self.assertTrue(any(esperado in e for e in etiquetas), etiquetas)

    def test_idioma_desconocido_revienta(self):
        with self.assertRaises(ValueError):
            informe_html(self.r, idioma="fr")
        with self.assertRaises(ValueError):
            informe_html(self.r, idioma="")

    def test_es_pura_misma_entrada_mismo_documento_y_no_toca_el_informe(self):
        antes = json.dumps(self.r.informe, sort_keys=True)
        self.assertEqual(informe_html(self.r, idioma="es"), self.es)
        self.assertEqual(antes, json.dumps(self.r.informe, sort_keys=True))


class CifrasTest(Base):
    def test_cifras_con_el_formato_del_idioma(self):
        auc = self.r.informe["puntuaciones"][0]["discriminacion"]["auroc"]["valor"]
        self.assertIn(f"{auc:.3f}".replace(".", ","), _texto(self.es))
        self.assertIn(f"{auc:.3f}", _texto(self.en))
        self.assertNotIn(f"{auc:.3f}", _texto(self.es))          # en castellano, coma
        self.assertNotIn(f"{auc:.3f}".replace(".", ","), _texto(self.en))
        sens = self.r.informe["puntuaciones"][0]["al_umbral"]["sensibilidad"]["valor"]
        self.assertIn(f"{sens * 100:.1f}".replace(".", ",") + " %", _texto(self.es))
        self.assertIn(f"{sens * 100:.1f} %", _texto(self.en))
        self.assertIn("1.000 casos con desenlace", _texto(self.es))
        self.assertIn("1,000 cases with an outcome", _texto(self.en))

    def test_la_cifra_prometida_dice_dentro_o_fuera_sin_adjetivos(self):
        pr = self.r.informe["puntuaciones"][0]["cifra_prometida"]
        self.assertIn(pr["resultado"], ("dentro", "fuera"))
        palabra = {"dentro": ("dentro del intervalo", "inside the interval"),
                   "fuera": ("fuera del intervalo", "outside the interval")}[pr["resultado"]]
        self.assertIn(f"Prometía 0,820: {palabra[0]}", _texto(self.es))
        self.assertIn(f"Promised 0.820: {palabra[1]}", _texto(self.en))
        otra = "dentro" if pr["resultado"] == "fuera" else "fuera"
        self.assertNotIn(f"0,820: {otra} del intervalo", _texto(self.es))

    def test_el_informe_no_califica_con_adjetivos_en_sus_propios_rotulos(self):
        prohibidas = ("bueno", "buena", "malo", "mala", "aceptable", "excelente", "pobre",
                      "good", "bad", "poor", "excellent", "acceptable", "fine")
        for clave, textos in TEXTOS.items():
            for idioma, frase in textos.items():
                palabras = re.findall(r"[^\W\d_]+", frase.lower())
                for p in prohibidas:
                    self.assertNotIn(p, palabras, f"{clave}/{idioma} dice «{p}»")


# ---------------------------------------------------------------------------------------------
# Calibración: solo si el núcleo dijo que aplica
# ---------------------------------------------------------------------------------------------

class CalibracionTest(Base):
    def test_una_escala_de_puntos_dice_no_aplica_y_no_pinta_la_curva_de_fiabilidad(self):
        r = _evaluar(puntuaciones=[Decl("score_b", "puntuacion", ALTO, umbral=6)])
        for idioma, no_aplica, motivo in (("es", "no aplica", "necesita probabilidades"),
                                          ("en", "not applicable", "needs probabilities")):
            doc = informe_html(r, idioma=idioma)
            self.assertIsNone(_svg(doc, "reliability"))
            self.assertIsNone(_svg(doc, "decision"))
            self.assertIn(no_aplica, _texto(doc))
            self.assertIn(motivo, _texto(doc))            # el motivo es el del núcleo
            self.assertIsNotNone(_svg(doc, "auroc"))      # y lo que sí aplica, se pinta
            self.assertNotIn("ECE", _texto(doc).split("TRIPOD")[0])  # ni cifra de calibración
        # y en la ficha: «No aplica», no «Medido»
        fila = _fila(informe_html(r, idioma="es"), "Calibración (ECE)")
        self.assertIn("No aplica", fila)

    def test_con_una_probabilidad_la_curva_de_fiabilidad_si_se_pinta_y_solo_con_ella(self):
        fiab = _svg(self.es, "reliability")
        self.assertIsNotNone(fiab)
        self.assertIn("score_a:", fiab)
        self.assertNotIn("score_b", fiab)       # la escala de puntos no entra en la curva
        self.assertIn("<polyline", fiab)
        self.assertIsNotNone(_svg(self.es, "decision"))

    def test_aunque_el_informe_traiga_tramos_si_aplica_es_falso_no_se_pinta(self):
        """Defensa en profundidad: la regla es «solo si el núcleo dijo que aplica», no «solo si
        hay datos». Un informe de una escala de puntos con tramos colados NO dibuja la curva."""
        r = _evaluar(puntuaciones=[Decl("score_b", "puntuacion", ALTO)])
        fiab_prob = self.r.informe["puntuaciones"][0]["calibracion"]["fiabilidad"]

        def colar(inf):
            inf["puntuaciones"][0]["calibracion"]["fiabilidad"] = copy.deepcopy(fiab_prob)
        colado = _con_informe(r, colar)
        self.assertIn("fiabilidad", colado.informe["puntuaciones"][0]["calibracion"])
        for idioma in IDIOMAS:
            self.assertIsNone(_svg(informe_html(colado, idioma=idioma), "reliability"))

    def test_los_tramos_con_pocas_filas_no_publican_cifra_y_se_cuentan(self):
        fia = self.r.informe["puntuaciones"][0]["calibracion"]["fiabilidad"]
        sin_cifra = [b for b in fia["bins"] if b["tasa_observada"] is None]
        self.assertTrue(sin_cifra, "el caso de prueba debe tener algún tramo sin cifra")
        con_cifra = [b for b in fia["bins"] if b["tasa_observada"] is not None]
        self.assertEqual(_svg(self.es, "reliability").count("<circle"), len(con_cifra))
        self.assertIn(f"Tramos sin cifra (pocas filas, evidencia insuficiente): {len(sin_cifra)}",
                      _texto(self.es))


# ---------------------------------------------------------------------------------------------
# Subgrupos, VPP/VPN, paradoja
# ---------------------------------------------------------------------------------------------

class SubgruposTest(Base):
    def test_un_subgrupo_pequeno_dice_evidencia_insuficiente_sin_numero(self):
        for doc, frase in ((self.es, "Evidencia insuficiente"), (self.en, "Insufficient evidence")):
            filas = [tr for tr in re.findall(r"<tr.*?</tr>", doc, flags=re.S)
                     if "Small ward" in tr]
            self.assertEqual(len(filas), 2)             # una por puntuación
            for tr in filas:
                celdas = re.findall(r"<td.*?</td>", tr, flags=re.S)
                self.assertEqual(len(celdas), 2)        # nombre + «insuficiente», NADA más
                self.assertIn(frase, tr)
                # los únicos números son los MÍNIMOS de la regla (30 filas, 10 casos)
                numeros = re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"<[^>]+>", " ", celdas[1]))
                self.assertEqual(sorted(numeros), ["10", "30"])
        # y ni su tamaño, ni su AUROC, ni su calibración, ni en el tooltip
        self.assertNotIn("12", _fila(self.es, "Small ward"))
        grupo = next(g for s in self.r.informe["puntuaciones"][0]["subgrupos"]
                     for g in s["grupos"] if g["valor"] == "Small ward")
        self.assertFalse(grupo["soporte_suficiente"])
        self.assertNotIn("auroc", grupo)

    def test_un_subgrupo_con_soporte_publica_casos_eventos_y_auroc(self):
        tr = _fila(self.en, "Ward A")
        celdas = [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td.*?</td>", tr, flags=re.S)]
        grupo = next(g for s in self.r.informe["puntuaciones"][0]["subgrupos"]
                     for g in s["grupos"] if g["valor"] == "Ward A")
        self.assertEqual(celdas[1], f"{grupo['n']:,}")
        self.assertEqual(celdas[2], f"{grupo['eventos']:,}")
        self.assertEqual(celdas[3], f"{grupo['auroc']['valor']:.3f}")

    def test_la_tarjeta_de_con_quien_falla_cuenta_los_insuficientes_sin_decir_cuales_son_sus_n(self):
        self.assertIn("Grupos con evidencia insuficiente: 1", _texto(self.es))
        self.assertIn("Groups with insufficient evidence: 1", _texto(self.en))


class UmbralTest(Base):
    def test_vpp_y_vpn_ausentes_si_la_muestra_no_representa(self):
        rep_vpp = self.r.informe["puntuaciones"][0]["al_umbral"]["vpp"]["valor"]
        rep_vpn = self.r.informe["puntuaciones"][0]["al_umbral"]["vpn"]["valor"]
        self.assertIsNotNone(rep_vpp)
        # con representatividad, se publican
        self.assertIn(f"{rep_vpp * 100:.1f} %", _texto(self.en))
        r = _dos(representa=False)
        for idioma, vpp in (("es", "VPP"), ("en", "PPV")):
            doc = informe_html(r, idioma=idioma)
            fila = _fila(doc, f"<td>{vpp}</td>")
            self.assertIn("nadie ha declarado" if idioma == "es" else "nobody declared", fila)
            self.assertNotIn('class="n">0', fila)          # nunca un cero
            celdas = re.findall(r'<td class="n">(.*?)</td>', fila)
            self.assertEqual(celdas, ["—", "—"])
            for valor in (rep_vpp, rep_vpn):
                txt = f"{valor * 100:.1f}" if idioma == "en" else f"{valor * 100:.1f}".replace(".", ",")
                self.assertNotIn(txt + " %", _texto(doc))
            self.assertNotIn("39,3 %), declarada" if idioma == "es" else "39.3 %), declared",
                             _texto(doc))
        # la prevalencia de la muestra sigue declarándose como MEDIDA, no como la de uso
        self.assertIn("Medido", _fila(informe_html(r, idioma="es"), "Prevalencia observada"))
        self.assertIn("No consta", _fila(informe_html(r, idioma="es"), "Prevalencia declarada"))

    def test_un_valor_ausente_no_se_pinta_como_cero(self):
        def vaciar(inf):
            u = inf["puntuaciones"][0]["al_umbral"]
            u["sensibilidad"] = {"valor": None, "ic": None, "ic_no_disponible": None,
                                 "no_definido": {"es": "sin eventos: no hay sensibilidad",
                                                 "en": "no events: there is no sensitivity"}}
            inf["puntuaciones"][0]["discriminacion"]["auroc"]["ic"] = None
        r = _con_informe(self.r, vaciar)
        for idioma, motivo in (("es", "sin eventos: no hay sensibilidad"),
                               ("en", "no events: there is no sensitivity")):
            doc = informe_html(r, idioma=idioma)
            fila = _fila(doc, "<td>Sensibilidad</td>" if idioma == "es" else "<td>Sensitivity</td>")
            self.assertIn(motivo, fila)
            self.assertNotRegex(re.sub(r"<[^>]+>", " ", fila).split(motivo)[0], r"\b0[.,]?0*\s*%")
            self.assertIn("sin intervalo" if idioma == "es" else "no interval", _texto(doc))

    def test_la_paradoja_del_tratamiento_solo_con_umbral_de_uso(self):
        con = {"es": "un riesgo alto llevó a tratar", "en": "a high risk led to treatment"}
        for idioma in IDIOMAS:
            self.assertIn(con[idioma], _texto(informe_html(self.r, idioma=idioma)))
        r = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO)])   # sin umbral
        for idioma in IDIOMAS:
            doc = informe_html(r, idioma=idioma)
            self.assertNotIn(con[idioma], _texto(doc))
            self.assertIn("no se declaró umbral de uso" if idioma == "es"
                          else "no usage threshold was declared", _texto(doc))
            # lo demás de «lo que no es» sigue
            self.assertIn("Depende del registro" if idioma == "es" else "It depends on the record",
                          _texto(doc))

    def test_sin_umbral_no_hay_tabla_al_umbral(self):
        r = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO)])
        doc = informe_html(r, idioma="en")
        self.assertNotIn("At the threshold of use", doc)
        self.assertIn("no threshold", _texto(doc))


class HtmlPorLaOrdenTest(unittest.TestCase):
    """De punta a punta: `matrixai evaluate-score --html FILE` escribe el informe."""

    def _correr(self, d, locale):
        import subprocess
        import sys
        from pathlib import Path
        (d / "datos.csv").write_text(_csv(_FILAS, _COLS), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "matrixai", "evaluate-score", str(d / "datos.csv"),
             "--outcome", "outcome", "--event", "Yes", "--represents-population", "yes",
             "--score", "score_a", "--score-type", "probability", "--direction", "higher",
             "--threshold", "0.3", "--score2", "score_b", "--score2-type", "score",
             "--score2-direction", "higher", "--score2-threshold", "6", "--subgroup", "sex",
             "--resamples", "30", "--locale", locale, "--html", str(d / f"informe_{locale}.html"),
             "-o", str(d / "informe.json")],
            capture_output=True, text=True, cwd=str(Path(modulo.__file__).resolve().parents[2]))
        return r

    def test_el_fichero_existe_es_del_idioma_pedido_y_no_tiene_http(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for locale, lang, frase in (("es", "es", "¿Funcionan aquí"), ("en", "en", "work here?")):
                r = self._correr(d, locale)
                self.assertEqual(r.returncode, 0, r.stderr)
                ruta = d / f"informe_{locale}.html"
                self.assertTrue(ruta.is_file())
                doc = ruta.read_text(encoding="utf-8")
                self.assertIn(f'<html lang="{lang}">', doc)
                self.assertIn(frase, _texto(doc))
                self.assertNotIn("http", doc.lower())
                self.assertIn("<polyline", _svg(doc, "roc"))        # la curva ROC del núcleo
                self.assertIn(f"HTML report written to {ruta}", r.stdout)
                self.assertTrue((d / "informe.json").is_file())     # y el JSON sigue saliendo
            es = (d / "informe_es.html").read_text(encoding="utf-8")
            self.assertEqual(set(_palabras_visibles(es)) & EN_FUNCIONALES, set())

    def test_sin_html_no_se_escribe_ningun_html(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            r = self._correr(d, "es")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(p.suffix for p in d.iterdir()), [".csv", ".html", ".json", ".json"])


class ComparacionTest(Base):
    def test_con_dos_puntuaciones_hay_comparacion_con_su_intervalo_y_las_filas_comunes(self):
        c = self.r.informe["comparacion"]
        txt = _texto(self.en)
        self.assertIn(f"{c['diferencia']:+.3f}", txt)
        self.assertIn(f"Over the {c['filas_comunes']:,} rows that have both", txt)
        self.assertIn("AUROC of “score_a” higher than that of “score_b”", txt)

    def test_con_una_sola_puntuacion_no_hay_seccion_de_comparacion(self):
        r = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO)])
        for idioma, titulo in (("es", "Una frente a otra"), ("en", "One against the other")):
            self.assertNotIn(titulo, _texto(informe_html(r, idioma=idioma)))
        # en la ficha, el tema «comparación» consta como que NO consta, con su motivo
        fila = _fila(informe_html(r, idioma="es"), "<td>Comparación entre las dos puntuaciones</td>")
        self.assertIn("No consta", fila)
        self.assertIn("se evaluó una sola puntuación", fila)


# ---------------------------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------------------------

class GraficasTest(Base):
    def test_las_cuatro_graficas_son_svg_en_linea_con_su_etiqueta_accesible(self):
        for tipo in ("auroc", "roc", "reliability", "decision"):
            svg = _svg(self.en, tipo)
            self.assertIsNotNone(svg, tipo)
            self.assertIn('role="img"', svg)
            self.assertIn("aria-label=", svg)
            self.assertNotIn("xmlns", svg)               # ni siquiera el espacio de nombres (URL)

    @staticmethod
    def _plano(svg: str):
        """(x0, y0, ancho, alto) del marco de la gráfica, para devolver un píxel a su valor."""
        m = re.search(r'<rect class="ax" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)
        return tuple(float(v) for v in m.groups())

    def test_el_punto_del_umbral_cae_donde_dicen_la_sensibilidad_y_la_especificidad(self):
        svg = _svg(self.en, "roc")
        x0, y0, w, h = self._plano(svg)
        circulos = re.findall(r'<circle class="f(\d)" cx="([\d.]+)" cy="([\d.]+)"', svg)
        self.assertEqual([c[0] for c in circulos], ["1", "2"])
        for (k, cx, cy), p in zip(circulos, self.r.informe["puntuaciones"]):
            u = p["al_umbral"]
            self.assertAlmostEqual((float(cx) - x0) / w, 1 - u["especificidad"]["valor"], delta=0.01)
            self.assertAlmostEqual(1 - (float(cy) - y0) / h, u["sensibilidad"]["valor"], delta=0.01)

    def test_los_puntos_de_fiabilidad_caen_donde_dicen_los_tramos(self):
        svg = _svg(self.en, "reliability")
        x0, y0, w, h = self._plano(svg)
        puntos = [(float(a), float(b)) for a, b in re.findall(r'<circle class="f1" cx="([\d.]+)" cy="([\d.]+)"', svg)]
        bins = [b for b in self.r.informe["puntuaciones"][0]["calibracion"]["fiabilidad"]["bins"]
                if b["tasa_observada"] is not None]
        self.assertEqual(len(puntos), len(bins))
        for (cx, cy), b in zip(puntos, bins):
            self.assertAlmostEqual((cx - x0) / w, b["probabilidad_media"], delta=0.01)
            self.assertAlmostEqual(1 - (cy - y0) / h, b["tasa_observada"], delta=0.01)

    def test_la_linea_de_la_fiabilidad_une_exactamente_sus_puntos(self):
        svg = _svg(self.en, "reliability")
        linea = re.search(r'<polyline class="l1" points="([^"]+)"', svg).group(1).split()
        puntos = [f"{a},{b}" for a, b in re.findall(r'<circle class="f1" cx="([\d.]+)" cy="([\d.]+)"', svg)]
        self.assertEqual(linea, puntos)

    def test_la_curva_roc_es_la_que_trae_el_nucleo_y_sin_ella_no_se_inventa(self):
        svg = _svg(self.en, "roc")
        lineas = re.findall(r'<polyline class="l(\d)" points="([^"]+)"', svg)
        self.assertEqual([k for k, _ in lineas], ["1", "2"])
        x0, y0, w, h = self._plano(svg)
        for (k, pts), p in zip(lineas, self.r.informe["puntuaciones"]):
            curva = p["discriminacion"]["curva_roc"]
            dibujada = [tuple(map(float, q.split(","))) for q in pts.split()]
            self.assertEqual(len(dibujada), len(curva))
            for (cx, cy), q in zip(dibujada, curva):
                self.assertAlmostEqual((cx - x0) / w, q["fpr"], delta=0.01)
                self.assertAlmostEqual(1 - (cy - y0) / h, q["tpr"], delta=0.01)

        def sin_curva(inf):
            for p in inf["puntuaciones"]:
                p["discriminacion"]["curva_roc"] = None
        self.assertNotIn("<polyline", _svg(informe_html(_con_informe(self.r, sin_curva), idioma="en"), "roc"))

    def test_las_leyendas_y_ejes_estan_en_su_idioma(self):
        self.assertIn("Falsas alarmas (1 − especificidad)", self.es)
        self.assertIn("False alarms (1 − specificity)", self.en)
        self.assertNotIn("False alarms", self.es)
        self.assertNotIn("Falsas alarmas", self.en)


# ---------------------------------------------------------------------------------------------
# El documento no lleva filas
# ---------------------------------------------------------------------------------------------

class SinFilasTest(unittest.TestCase):
    def test_ni_un_identificador_ni_un_valor_de_fila_aparece_en_el_html(self):
        r = _dos(columna_de_paciente="id")        # la columna con el identificador raro, USADA
        for idioma in IDIOMAS:
            doc = informe_html(r, idioma=idioma)
            self.assertNotIn(RARO, doc)
            self.assertNotRegex(doc, r"P\d{5}")
            self.assertIn("score_a", doc)           # (control: el informe no está en blanco)
            self.assertIn("id", _texto(doc))
            # los valores de las filas con seis decimales (la probabilidad cruda) tampoco
            for f in _FILAS[:60]:
                cruda = repr(round(f["score_a"], 6))
                self.assertNotIn(cruda, doc)
                self.assertNotIn(cruda.replace(".", ","), doc)

    def test_el_html_tiene_solo_agregados_cuenta_las_filas_pero_no_las_enseña(self):
        r = _dos()
        doc = informe_html(r, idioma="en")
        self.assertIn("1,000 cases with an outcome", _texto(doc))
        self.assertIn("carries no data rows", _texto(doc))


class EscapadoTest(unittest.TestCase):
    def test_todo_texto_de_usuario_se_escapa(self):
        sucio = "<b>x</b>"
        filas = copy.deepcopy(_FILAS)
        cols = ["id", "score_a", "outcome", "sex"]
        r = evaluar_puntuacion(
            _csv(filas, cols).replace("score_a", "sa"), columna_desenlace="outcome",
            valor_del_evento="Yes",
            puntuaciones=[Decl("sa", "probabilidad", ALTO, umbral=0.3, origen=sucio + "<i>o</i>")],
            muestra_representativa=True, columnas_de_subgrupo=("sex",), remuestras=REM,
            nombre_del_fichero="a<img src=x onerror=1>.csv", actor="<u>quien</u>",
            clave=_CLAVE, creado_en="2026-10-23T12:00:00Z")
        for idioma in IDIOMAS:
            doc = informe_html(r, idioma=idioma)
            self.assertNotIn("<img", doc)
            self.assertNotIn("<u>quien", doc)
            self.assertNotIn("<i>o</i>", doc)
            self.assertIn("&lt;img src=x onerror=1&gt;.csv", doc)
            self.assertIn("&lt;i&gt;o&lt;/i&gt;", doc)
            self.assertIn("&lt;u&gt;quien&lt;/u&gt;", doc)


# ---------------------------------------------------------------------------------------------
# Recibo y garantía
# ---------------------------------------------------------------------------------------------

class ReciboTest(unittest.TestCase):
    def test_a1_dice_firmado_y_a0_dice_sin_firmar_con_lo_que_significan(self):
        firmado = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO)])
        sin = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO)], firmar=False)
        self.assertEqual((firmado.nivel_de_garantia, sin.nivel_de_garantia), ("A1", "A0"))
        a1, a0 = informe_html(firmado, idioma="es"), informe_html(sin, idioma="es")
        self.assertIn("Garantía A1 · firmado", _texto(a1))
        self.assertIn("A1: el recibo está firmado con la clave de esta instalación", _texto(a1))
        self.assertIn("clave evaluate-score, firma", _texto(a1))
        self.assertIn("Garantía A0 · sin firmar", _texto(a0))
        self.assertIn("A0: el recibo no lleva firma", _texto(a0))
        self.assertNotIn("firma ", _fila(a0, "Firma (HMAC)").replace("Firma (HMAC)", ""))
        en = informe_html(firmado, idioma="en")
        self.assertIn("Assurance A1 · signed", _texto(en))
        self.assertIn("Whoever holds that key", _texto(en))

    def test_el_recibo_enseña_las_huellas_y_la_version_del_nucleo(self):
        r = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO)])
        doc = informe_html(r, idioma="en")
        self.assertIn(r.informe["entrada"]["huella_csv"], doc)
        self.assertIn(r.recibo["receipt_id"], doc)
        self.assertIn(r.recibo["output"]["output_digests"]["report"], doc)
        self.assertIn(r.informe["versiones"]["nucleo"], doc)
        self.assertIn("seed 137", _texto(doc))
        self.assertIn(f"{REM} resamples", _texto(doc))
        self.assertIn("Internal Medicine", doc)


# ---------------------------------------------------------------------------------------------
# TRIPOD+AI de validación externa
# ---------------------------------------------------------------------------------------------

class TripodTest(Base):
    def test_los_temas_sin_dato_dicen_no_consta_y_con_su_motivo(self):
        for idioma, no_consta in (("es", "No consta"), ("en", "Not reported")):
            doc = informe_html(self.r, idioma=idioma)
            for clave in ("poblacion", "criterios_de_inclusion", "predictores",
                          "definicion_del_desenlace", "momento_del_desenlace", "proceso_actual",
                          "uso_previsto"):
                fila = _fila(doc, f"<td>{POR_CLAVE[clave].rotulo[idioma]}</td>")
                self.assertIn(no_consta, fila, clave)
                celdas = [re.sub(r"<[^>]+>", "", c).strip()
                          for c in re.findall(r"<td.*?</td>", fila, flags=re.S)]
                self.assertEqual(len(celdas), 3)
                self.assertTrue(all(celdas), f"{clave}: celda vacía en {celdas}")

    def test_lo_que_solo_declara_el_autor_viene_de_tripod_py_y_dice_no_consta(self):
        from matrixai.export.tripod import _NO_DISPONIBLE
        autor = [(c, tx) for c, tx in _NO_DISPONIBLE if c is None]
        self.assertGreaterEqual(len(autor), 3)
        for idioma, no_consta in (("es", "No consta"), ("en", "Not reported")):
            doc = informe_html(self.r, idioma=idioma)
            for _, tx in autor:
                self.assertIn(no_consta, _fila(doc, f"<td>{tx[idioma][0]}</td>"))

    def test_los_rotulos_son_los_del_inventario_de_expediente_clinico(self):
        for idioma in IDIOMAS:
            items = {i.clave: i for i in items_tripod(self.r, idioma)}
            for clave in CLAVES_DEL_INVENTARIO_USADAS:
                self.assertIn(clave, items, clave)
                self.assertEqual(items[clave].rotulo, POR_CLAVE[clave].rotulo[idioma])
                self.assertEqual(items[clave].dominio, POR_CLAVE[clave].dominio)

    def test_solo_hay_tres_estados_y_el_de_cada_tema_es_el_que_le_toca(self):
        items = {i.clave: i for i in items_tripod(self.r, "es")}
        self.assertTrue({i.estado for i in items.values()} <= set(ESTADOS))
        self.assertEqual(items["huella_del_dataset"].estado, MEDIDO)
        self.assertEqual(items["metricas"].estado, MEDIDO)
        self.assertEqual(items["vx_representatividad"].estado, DECLARADO)
        self.assertEqual(items["fuente_de_los_datos"].estado, DECLARADO)
        for clave in ("poblacion", "predictores", "uso_previsto", "proceso_actual"):
            self.assertEqual(items[clave].estado, FALTA, clave)

    def test_cada_linea_traza_a_un_campo_que_existe_en_el_informe(self):
        for idioma in IDIOMAS:
            for i in items_tripod(self.r, idioma):
                self.assertTrue(i.ruta.startswith("informe#"), i.ruta)
                resolver_ruta(self.r.informe, i.ruta)
        with self.assertRaises(KeyError):
            resolver_ruta(self.r.informe, "informe#puntuaciones[0].no_existe")

    def test_la_ficha_no_califica_el_riesgo_de_sesgo(self):
        from matrixai.export.expediente_clinico import VeredictoDeRiesgo, sin_veredicto_de_riesgo
        for idioma in IDIOMAS:
            seccion = re.search(r'<section class="blk tripod">.*?</section>',
                                informe_html(self.r, idioma=idioma), flags=re.S).group(0)
            sin_veredicto_de_riesgo(re.sub(r"<[^>]+>", " ", seccion), origen="prueba")
        # y la puerta está puesta: un dato de usuario con vocabulario de veredicto REVIENTA
        sucio = _con_informe(self.r, lambda inf: inf["desenlace"].__setitem__("columna", "bajo riesgo"))
        with self.assertRaises(VeredictoDeRiesgo):
            informe_html(sucio, idioma="es")

    def test_la_ficha_dice_que_no_numera_y_que_no_es_una_declaracion_de_cumplimiento(self):
        self.assertIn("sin numerar", _texto(self.es))
        self.assertIn("No es una declaración de cumplimiento", _texto(self.es))
        self.assertIn("unnumbered", _texto(self.en))
        self.assertIn("This is not a statement of compliance", _texto(self.en))

    def test_con_una_escala_de_puntos_la_calibracion_y_la_curva_dicen_no_aplica_en_la_ficha(self):
        r = _evaluar(puntuaciones=[Decl("score_b", "puntuacion", ALTO)])
        items = {i.clave: i for i in items_tripod(r, "es")}
        self.assertTrue(items["calibracion"].no_aplica)
        self.assertEqual(items["calibracion"].estado, FALTA)
        self.assertTrue(items["curva_de_decision"].no_aplica)
        self.assertEqual(items["umbrales"].estado, FALTA)         # sin umbral declarado
        self.assertEqual(items["subgrupos"].estado, FALTA)        # sin columnas de subgrupo
        self.assertIn("No aplica", _fila(informe_html(r, idioma="es"), "Calibración (ECE)"))


# ---------------------------------------------------------------------------------------------
# El catálogo propio
# ---------------------------------------------------------------------------------------------

class CatalogoTest(unittest.TestCase):
    def test_las_dos_redacciones_piden_los_mismos_huecos(self):
        for clave, textos in TEXTOS.items():
            self.assertEqual(set(textos), set(IDIOMAS), clave)
            self.assertEqual(huecos_de(textos["es"]), huecos_de(textos["en"]), clave)

    def test_una_clave_inexistente_revienta_y_un_idioma_desconocido_tambien(self):
        with self.assertRaises(KeyError):
            t("no_existe", "es")
        with self.assertRaises(ValueError):
            t("na", "de")

    def test_todos_los_rotulos_que_usa_el_modulo_existen_en_el_catalogo(self):
        usados = set(re.findall(r'\bt\(\s*"([a-z_0-9]+)"', open(modulo.__file__, encoding="utf-8").read()))
        usados |= set(re.findall(r'\bt\(\s*"([a-z_0-9]+)" if', open(modulo.__file__, encoding="utf-8").read()))
        faltan = sorted(u for u in usados if u not in TEXTOS)
        self.assertEqual(faltan, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
