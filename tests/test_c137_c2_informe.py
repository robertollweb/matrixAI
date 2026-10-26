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
    DeclaracionDePuntuacion as _Decl,
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
R2 = dict(horizonte_del_desenlace="hospital stay", momento_de_la_puntuacion="admission day",
          donde_se_calcula="local")


def Decl(columna, tipo, direccion, **kw):
    """Las pruebas que no hablan de la R2 declaran «no participaron»; las que sí, lo pasan."""
    kw.setdefault("casos_en_su_desarrollo", "no")
    return _Decl(columna, tipo, direccion, **kw)
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
    for k, v in R2.items():
        kw.setdefault(k, v)
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
        # El recuadro de los datos firmados (auditoría del 26-10, hallazgo 9) es JSON con base64:
        # datos para una máquina, no texto de la plantilla.
        if tag in ("style", "script") or (tag == "pre" and ("class", "datos") in attrs):
            self._ignorar += 1
        for k, v in attrs:
            self.atributos.append((tag, k, v or ""))

    def handle_endtag(self, tag):
        if tag in ("style", "script") or (tag == "pre" and self._ignorar):
            self._ignorar -= 1

    def handle_data(self, data):
        if not self._ignorar:
            self.texto.append(data)


def _sin_datos_firmados(doc: str) -> str:
    """El documento sin el recuadro de los datos firmados: su base64 puede traer, por azar, las
    letras «http» seguidas, y eso no es una URL."""
    return re.sub(r'<pre class="datos"[^>]*>.*?</pre>', "", doc, flags=re.S)


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
            self.assertNotIn("http", _sin_datos_firmados(doc).lower())
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

    def test_la_referencia_publicada_dice_donde_cae_sin_adjetivos_y_sin_el_viejo_veredicto(self):
        # R4. Antes: «Prometía 0,820: fuera del intervalo» (un veredicto). Ahora: las DOS cifras
        # y dónde cae la referencia respecto al intervalo local; la intención de la prueba
        # (dentro/fuera dicho sin adjetivos) se conserva.
        a = self.r.informe["puntuaciones"][0]
        pr = a["cifra_prometida"]
        self.assertIn(pr["resultado"], ("dentro", "fuera"))
        palabra = {"dentro": ("dentro", "inside"), "fuera": ("fuera", "outside")}[pr["resultado"]]
        local = f'{a["discriminacion"]["auroc"]["valor"]:.3f}'
        self.assertIn(f"Referencia publicada 0,820 · estimación local {local.replace('.', ',')}: "
                      f"la referencia cae {palabra[0]} del intervalo local", _texto(self.es))
        self.assertIn(f"Published reference 0.820 · local estimate {local}: the reference falls "
                      f"{palabra[1]} the local interval", _texto(self.en))
        otra = "dentro" if pr["resultado"] == "fuera" else "fuera"
        self.assertNotIn(f"la referencia cae {otra} del intervalo", _texto(self.es))
        for doc in (self.es, self.en):
            self.assertNotIn("Prometía", doc)
            self.assertNotIn("Promised 0.", doc)

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
        ic = grupo["auroc"]["ic"]                     # R5: el AUROC del grupo lleva su intervalo
        self.assertEqual(celdas[3], f"{grupo['auroc']['valor']:.3f} ({ic['bajo']:.3f} – {ic['alto']:.3f})")

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
             "--outcome-horizon", "hospital stay", "--score-timing", "admission day",
             "--score", "score_a", "--score-type", "probability", "--direction", "higher",
             "--score-development-cases", "no", "--score2-development-cases", "yes",
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
                self.assertNotIn("http", _sin_datos_firmados(doc).lower())
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
            muestra_representativa=True, columnas_de_subgrupo=("sex",), remuestras=REM, **R2,
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
        # La firma COMPLETA (auditoría del 26-10), no sus 16 primeros caracteres.
        self.assertIn("clave «evaluate-score», firma: " + firmado.sobre["signatures"][0]["sig"],
                      _texto(a1))
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
                          "definicion_del_desenlace", "proceso_actual",
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


# ---------------------------------------------------------------------------------------------
# Ronda de rigor del 24-10 (R3, R4, R5, R6)
# ---------------------------------------------------------------------------------------------

def _trozo_de_grafica(doc: str, tipo: str) -> str:
    for caja in doc.split('<div class="chart">'):
        if f'data-chart="{tipo}"' in caja:
            return caja
    raise AssertionError(f"no hay gráfica {tipo}")


ROTULO_ES = "describen la muestra aportada, no la población de uso"
ROTULO_EN = "describe the supplied sample, not the population of use"


class R3RepresentatividadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.no = _dos(representa=False)
        cls.si = _dos(representa=True)
        cls.no_es, cls.no_en = (informe_html(cls.no, idioma=i) for i in ("es", "en"))
        cls.si_es, cls.si_en = (informe_html(cls.si, idioma=i) for i in ("es", "en"))

    def test_R3_sin_representatividad_la_calibracion_lleva_el_rotulo_y_su_porque(self):
        tarjeta = next(c for c in self.no_es.split('<div class="tile') if "intercepto" in c.lower()
                       and "Acierta en el riesgo" in c)
        self.assertIn(ROTULO_ES, tarjeta.lower())
        self.assertIn("más eventos de los habituales", tarjeta)
        tarjeta_en = next(c for c in self.no_en.split('<div class="tile') if "Is the risk right" in c
                          and "Intercept" in c)
        self.assertIn(ROTULO_EN, tarjeta_en.lower())
        self.assertIn("more events than usual", tarjeta_en)

    def test_R3_sin_representatividad_el_beneficio_neto_y_la_curva_de_decision_lo_llevan(self):
        for doc, rot in ((self.no_es, ROTULO_ES), (self.no_en, ROTULO_EN)):
            self.assertIn(rot, _fila(doc, "Beneficio neto" if rot == ROTULO_ES else "Net benefit").lower())
            self.assertIn(rot, _trozo_de_grafica(doc, "decision").lower())
            self.assertIn(rot, _trozo_de_grafica(doc, "reliability").lower())

    def test_R3_el_rotulo_tambien_esta_en_la_pagina_de_conclusiones(self):
        for doc, rot in ((self.no_es, ROTULO_ES), (self.no_en, ROTULO_EN)):
            conclusiones = doc[:doc.index("<details")]
            self.assertIn(rot, _texto(conclusiones).lower())

    def test_R3_con_representatividad_no_hay_rotulo_y_se_dice_que_lo_declara_quien_ejecuta(self):
        for doc in (self.si_es, self.si_en):
            self.assertNotIn(ROTULO_ES, doc.lower())
            self.assertNotIn(ROTULO_EN, doc.lower())
        c_es, c_en = (_texto(d[:d.index("<details")]) for d in (self.si_es, self.si_en))
        self.assertIn("lo declara quien ejecuta la evaluación, y MatrixAI no lo comprueba", c_es)
        self.assertIn("is declared by whoever runs the evaluation, and MatrixAI does not check it", c_en)

    def test_R3_el_informe_del_nucleo_lleva_la_misma_declaracion_en_el_dato(self):
        rep_no, rep_si = self.no.informe["representatividad"], self.si.informe["representatividad"]
        self.assertIs(rep_no["declarada"], False)
        self.assertIs(rep_no["comprobada_por_matrixai"], False)
        self.assertEqual(rep_no["rotulo"]["es"], ROTULO_ES)
        self.assertIsNone(rep_si["rotulo"])
        for p in self.si.informe["puntuaciones"]:
            self.assertIsNone((p["calibracion"] if p["calibracion"]["aplica"] else {}).get("rotulo_de_alcance"))

    def test_R3_una_escala_de_puntos_no_lleva_rotulo_de_calibracion_que_no_tiene(self):
        # score_b no tiene calibración ni curva de decisión: el rótulo no se cuela donde no
        # hay nada que rotular (sería afirmar que existe una calibración).
        b = self.no.informe["puntuaciones"][1]
        self.assertFalse(b["calibracion"]["aplica"])
        self.assertNotIn("rotulo_de_alcance", b["calibracion"])
        self.assertNotIn("rotulo_de_alcance", b["curva_de_decision"])


class R4ReferenciaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _dos()
        cls.es = informe_html(cls.r, idioma="es")
        cls.en = informe_html(cls.r, idioma="en")

    def test_R4_se_dan_las_dos_cifras_y_la_advertencia_siempre(self):
        adv_es = "no demuestra que la cifra original esté mal"
        adv_en = "does not show that the original figure is wrong"
        self.assertIn(adv_es, _texto(self.es))
        self.assertIn(adv_en, _texto(self.en))
        # la advertencia COMPLETA está en el anexo; la página de conclusiones lleva la corta, siempre
        for doc, completa, corta in (
                (self.es, "población, el desenlace, el seguimiento y la incertidumbre",
                 "Pueden no ser comparables (población, desenlace, seguimiento e incertidumbre); "
                 "caer fuera del intervalo no demuestra que la cifra original esté mal"),
                (self.en, "population, the outcome, the follow-up and the uncertainty",
                 "They may not be comparable (population, outcome, follow-up and uncertainty); "
                 "falling outside the interval does not show that the original figure is wrong")):
            self.assertIn(completa, _texto(doc[doc.index("<details"):]))
            conclusiones = _texto(doc[:doc.index("<details")])
            self.assertIn(corta, conclusiones)
            self.assertIn("0,820" if doc is self.es else "0.820", conclusiones)

    def test_R4_la_advertencia_sale_tanto_si_cae_dentro_como_si_cae_fuera(self):
        a = self.r.informe["puntuaciones"][0]
        bajo, alto = a["cifra_prometida"]["intervalo"]
        dentro = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO, umbral=0.3,
                                             cifra_prometida=round((bajo + alto) / 2, 3))])
        fuera = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO, umbral=0.3,
                                            cifra_prometida=0.5)])
        self.assertEqual(dentro.informe["puntuaciones"][0]["cifra_prometida"]["resultado"], "dentro")
        self.assertEqual(fuera.informe["puntuaciones"][0]["cifra_prometida"]["resultado"], "fuera")
        for r, pos in ((dentro, "dentro"), (fuera, "fuera")):
            doc = informe_html(r, idioma="es")
            self.assertIn(f"la referencia cae {pos} del intervalo local", _texto(doc))
            self.assertIn("no demuestra que la cifra original esté mal", _texto(doc))
            self.assertIn("estimacion_local", r.informe["puntuaciones"][0]["cifra_prometida"])

    def test_R4_sin_cifra_prometida_no_hay_referencia_ni_advertencia(self):
        r = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO, umbral=0.3)])
        doc = informe_html(r, idioma="es")
        self.assertNotIn("Referencia publicada", doc)
        self.assertNotIn("no demuestra que la cifra original", doc)
        self.assertIsNone(r.informe["puntuaciones"][0]["cifra_prometida"])

    def test_R4_la_comparacion_emparejada_se_senala_como_la_mas_directamente_interpretable(self):
        self.assertIn("la comparación más directamente interpretable", _texto(self.es))
        self.assertIn("most directly interpretable comparison", _texto(self.en))
        conclusiones = _texto(self.es[:self.es.index("<details")])
        self.assertIn("Es la comparación más directamente interpretable: las dos puntuaciones se "
                      "miden sobre los mismos casos", conclusiones)
        self.assertIn("Frente a la otra: «score_a» frente a «score_b»", conclusiones)


class R5PrivacidadTest(unittest.TestCase):
    PROMESAS = re.compile(r"protege|protect|garantiza la privacidad|guarantees? privacy|"
                          r"casi identifica|nearly identif|acerca a una identificaci|"
                          r"identificar[ií]a|would identify|privacidad", re.I)

    @classmethod
    def setUpClass(cls):
        cls.r = _dos()
        cls.es = informe_html(cls.r, idioma="es")
        cls.en = informe_html(cls.r, idioma="en")

    def test_R5_ningun_texto_promete_privacidad_ni_dice_que_el_minimo_protege(self):
        from matrixai.estudio.evaluacion_de_puntuacion_textos import MOTIVOS
        for nombre, catalogo in (("informe", TEXTOS), ("nucleo", MOTIVOS)):
            for clave, textos in catalogo.items():
                for idioma, frase in textos.items():
                    self.assertIsNone(self.PROMESAS.search(frase), f"{nombre}/{clave}/{idioma}")
        for doc in (self.es, self.en):
            self.assertIsNone(self.PROMESAS.search(_texto(doc)))
        from matrixai.estudio import evaluacion_de_puntuacion as nucleo
        self.assertNotIn("también protege la privacidad", nucleo.__doc__)

    def test_R5_dice_que_suprimir_grupos_pequenos_ayuda_pero_no_garantiza_anonimizacion(self):
        self.assertIn("Suprimir grupos pequeños ayuda, pero no garantiza anonimización",
                      _texto(self.es))
        self.assertIn("Suppressing small groups helps, but does not guarantee anonymisation",
                      _texto(self.en))
        # el límite completo vive en el anexo de límites (la página de conclusiones es corta)
        for doc, frase in ((self.es, "no garantiza anonimización"),
                           (self.en, "does not guarantee anonymisation")):
            anexo = doc[doc.index("Límites completos" if doc is self.es else "Full limits"):]
            self.assertIn(frase, _texto(anexo))
        self.assertIn("no_es_anonimizacion", [x["id"] for x in self.r.informe["lo_que_no_es"]])

    def test_R5_el_auroc_de_cada_grupo_con_evidencia_lleva_su_intervalo_en_el_html(self):
        sub = self.r.informe["puntuaciones"][0]["subgrupos"]
        medido = next(g for s in sub for g in s["grupos"] if g["soporte_suficiente"])
        ic = medido["auroc"]["ic"]
        self.assertLess(ic["bajo"], medido["auroc"]["valor"])
        self.assertLess(medido["auroc"]["valor"], ic["alto"])
        celda = f'{medido["auroc"]["valor"]:.3f}'.replace(".", ",")
        fila = _fila(self.es, f'<b>{medido["valor"]}</b>')
        self.assertIn(f'{celda} ({ic["bajo"]:.3f}'.replace(".", ","), fila)

    def test_R5_un_grupo_sin_evidencia_sigue_sin_numero_ni_intervalo(self):
        sub = self.r.informe["puntuaciones"][0]["subgrupos"]
        pobre = next(g for s in sub for g in s["grupos"] if not g["soporte_suficiente"])
        self.assertNotIn("auroc", pobre)
        self.assertNotIn("eventos", pobre)
        self.assertIn("Evidencia insuficiente", _fila(self.es, f'<b>{pobre["valor"]}</b>'))


class R6ConclusionesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _dos()
        cls.es = informe_html(cls.r, idioma="es")
        cls.en = informe_html(cls.r, idioma="en")

    def test_R6_las_conclusiones_van_antes_que_cualquier_anexo(self):
        for doc, titulo, anexo, ficha in ((self.es, "Conclusiones", "Anexo A", "Ficha TRIPOD+AI"),
                                          (self.en, "Conclusions", "Annex A", "TRIPOD+AI")):
            self.assertIn('<section class="conclusiones">', doc)
            self.assertLess(doc.index('<section class="conclusiones">'), doc.index("<details"))
            self.assertLess(doc.index(titulo), doc.index(anexo))
            # los detalles técnicos SOLO aparecen dentro de los anexos
            primer = doc.index("<details")
            for tecnico in ("Matriz de confusión" if doc is self.es else "Confusion matrix",
                            "Huella del CSV" if doc is self.es else "CSV digest", ficha):
                self.assertGreater(doc.index(tecnico), primer, tecnico)

    def test_R6_los_anexos_son_details_nativos_sin_javascript(self):
        for doc in (self.es, self.en):
            self.assertEqual(doc.count('<details class="anexo">'), 7)   # 2 puntuaciones + 5
            # + el recuadro de los datos firmados, dentro del anexo «cómo se hizo» (26-10)
            self.assertEqual(doc.count('<details class="datos-firmados">'), 1)
            self.assertEqual(doc.count("</details>"), 8)
            self.assertEqual(doc.count("<summary>"), 8)
            self.assertNotIn("<script", doc.lower())
            self.assertNotIn("http", _sin_datos_firmados(doc).lower())
            self.assertNotIn("onclick", doc.lower())

    def test_R6_al_imprimir_los_anexos_se_piden_abiertos_y_las_conclusiones_hacen_hoja(self):
        css = self.es[self.es.index("<style>"):self.es.index("</style>")]
        self.assertRegex(css, r"@media print\{details\.anexo::details-content\{[^}]*content-visibility:visible")
        self.assertIn("section.conclusiones{font-size:11px;break-after:page}", css)

    def test_R6_las_conclusiones_llevan_veredicto_cifras_y_limites_y_las_cinco_preguntas_van_al_anexo(self):
        c = _texto(self.es[:self.es.index("<details")])
        for frase in ("Qué se evaluó", "Cifras clave:", "Frente a la otra:", "Límites de este informe",
                      "Ordena mejor que el azar"):
            self.assertIn(frase, c)
        ce = _texto(self.en[:self.en.index("<details")])
        for frase in ("What was evaluated", "Key figures:", "Against the other score:",
                      "Limits of this report", "Ranks better than chance"):
            self.assertIn(frase, ce)
        # lo que baja a los anexos, y que SIGUE estando en el documento
        for doc, arriba, abajo in (
                (self.es, ("Dicho en llano", "¿Acierta en el riesgo?", "de cada 100 eventos"),
                 ("En lenguaje llano", "Dicho en llano", "¿Acierta en el riesgo?", "de cada 100 eventos",
                  "Lo que este informe no es")),
                (self.en, ("In plain terms", "Is the risk right?", "out of every 100 events"),
                 ("In plain language", "In plain terms", "Is the risk right?", "out of every 100 events",
                  "What this report is not"))):
            cabeza, resto = _texto(doc[:doc.index("<details")]), _texto(doc[doc.index("<details"):])
            for frase in arriba:
                self.assertNotIn(frase, cabeza)
            for frase in abajo:
                self.assertIn(frase, resto)

    def test_R6_la_frase_de_la_firma_esta_en_las_conclusiones(self):
        self.assertIn("La firma acredita integridad comprobable con la clave; el valor científico "
                      "depende del diseño y de los datos", _texto(self.es[:self.es.index("<details")]))
        self.assertIn("The signature certifies integrity that can be checked with the key; the "
                      "scientific value depends on the design and on the data",
                      _texto(self.en[:self.en.index("<details")]))

    def test_R6_sin_firma_las_conclusiones_lo_dicen(self):
        r = _dos(firmar=False)
        doc = informe_html(r, idioma="es")
        self.assertIn("Este informe no está firmado (garantía A0)", _texto(doc[:doc.index("<details")]))
        self.assertNotIn("Este informe no está firmado", _texto(self.es))

    def test_R6_las_cifras_de_las_conclusiones_son_las_del_informe(self):
        a = self.r.informe["puntuaciones"][0]
        c = _texto(self.es[:self.es.index("<details")])
        self.assertIn(f'AUROC {a["discriminacion"]["auroc"]["valor"]:.3f}'.replace(".", ","), c)
        sens = a["al_umbral"]["sensibilidad"]["valor"]
        self.assertIn(f"sensibilidad {sens * 100:.1f}".replace(".", ","), c)



# ---------------------------------------------------------------------------------------------
# R2 y «dónde se calculó» en el HTML
# ---------------------------------------------------------------------------------------------

def _conclusiones(doc: str) -> str:
    return _texto(doc[:doc.index("<details")])


class R2HtmlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dos = _dos(horizonte_del_desenlace="30 days", momento_de_la_puntuacion="day 3")
        cls.es = informe_html(cls.dos, idioma="es")
        cls.en = informe_html(cls.dos, idioma="en")
        # score_a: casos en su desarrollo = «no» (por la envoltura); score_b: «no se sabe»
        cls.mixto = _evaluar(puntuaciones=[
            Decl("score_a", "probabilidad", ALTO, umbral=0.3, casos_en_su_desarrollo="si"),
            Decl("score_b", "puntuacion", ALTO, umbral=6, casos_en_su_desarrollo="no_se_sabe")])
        cls.todo_no = _evaluar(puntuaciones=[
            Decl("score_a", "probabilidad", ALTO, umbral=0.3, casos_en_su_desarrollo="no")])

    def test_R2_el_horizonte_y_el_momento_estan_en_las_conclusiones_en_los_dos_idiomas(self):
        self.assertIn("Horizonte del desenlace: 30 days. Momento de la puntuación: day 3",
                      _conclusiones(self.es))
        self.assertIn("Outcome horizon: 30 days. Score timing: day 3", _conclusiones(self.en))
        self.assertIn("30 days", _fila(self.es, "Horizonte del desenlace"))
        self.assertIn("day 3", _fila(self.en, "Score timing"))

    def test_R2_con_si_o_no_se_sabe_el_html_dice_no_es_una_validacion_externa(self):
        es, en = (informe_html(self.mixto, idioma=i) for i in ("es", "en"))
        c_es, c_en = _conclusiones(es), _conclusiones(en)
        self.assertIn("No es una validación externa de: score_a, score_b", c_es)
        self.assertIn("It is not an external validation of: score_a, score_b", c_en)
        # las notas por puntuación, completas, están en el anexo de límites
        a_es, a_en = (_texto(d[d.index("<details"):]) for d in (es, en))
        self.assertIn("esto NO es una validación externa de esa puntuación", a_es)
        self.assertIn("mientras no se sepa, esto NO es una validación externa", a_es)
        self.assertIn("this is NOT an external validation of that score", a_en)

    def test_R2_la_ficha_tripod_lo_refleja(self):
        for idioma, doc, rotulo, frase in (
                ("es", informe_html(self.mixto, idioma="es"),
                 "¿Participaron estos casos en el desarrollo de la puntuación?",
                 "no es una validación externa"),
                ("en", informe_html(self.mixto, idioma="en"),
                 "Did these cases take part in the development of the score?",
                 "this is not an external validation")):
            fila = _fila(doc, f"<td>{rotulo}</td>")
            self.assertIn(frase, fila)
            self.assertIn("«score_a»" if idioma == "es" else "“score_a”", fila)
            alcance = _fila(doc, "Alcance de la validación" if idioma == "es" else "Validation scope")
            self.assertIn("score_a, score_b", alcance)
        items = {i.clave: i for i in items_tripod(self.mixto, "es")}
        self.assertEqual(items["vx_desarrollo"].estado, DECLARADO)
        self.assertEqual(items["momento_del_desenlace"].estado, DECLARADO)
        self.assertIn("hospital stay", items["momento_del_desenlace"].dato)
        self.assertEqual(resolver_ruta(self.mixto.informe, items["vx_desarrollo"].ruta)["casos_en_su_desarrollo"], "si")

    def test_R2_con_no_el_html_no_dice_que_no_es_validacion_externa_pero_si_que_no_se_comprueba(self):
        es = informe_html(self.todo_no, idioma="es")
        self.assertNotIn("no es una validación externa", _texto(es).lower())
        self.assertNotIn("NO es una validación externa", _texto(es))
        self.assertIn("Se declaró que los casos no participaron en el desarrollo de la puntuación "
                      "(MatrixAI no lo comprueba)", _conclusiones(es))
        fila = _fila(es, "<td>¿Participaron estos casos en el desarrollo de la puntuación?</td>")
        self.assertIn("declarado, MatrixAI no lo comprueba", fila)

    def test_R2_la_declaracion_de_cada_puntuacion_enseña_sus_casos_en_su_desarrollo(self):
        self.assertIn("casos en su desarrollo: sí", _texto(informe_html(self.mixto, idioma="es")))
        self.assertIn("cases in its development: unknown", _texto(informe_html(self.mixto, idioma="en")))


class DondeSeCalculaHtmlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = _dos(donde_se_calcula="local")
        cls.demo = _dos(donde_se_calcula="servidor_de_la_demo")

    def test_la_demo_dice_que_el_servidor_calculo_con_el_csv_enviado_y_que_no_se_guardo(self):
        es, en = (informe_html(self.demo, idioma=i) for i in ("es", "en"))
        self.assertIn("Se calculó en el servidor de la demo con el CSV que enviaste; el CSV no se guardó.",
                      _conclusiones(es))
        self.assertIn("It was calculated on the demo server with the CSV you sent; the CSV was not stored.",
                      _conclusiones(en))
        for doc in (es, en):                       # la cabecera y el anexo «Cómo se hizo» también
            self.assertGreaterEqual(_texto(doc).count("servidor de la demo" if doc is es else "demo server"), 2)
        self.assertNotIn("se calculó en local", _texto(es).lower())
        self.assertNotIn("calculated locally", _texto(en).lower())

    def test_en_local_dice_que_no_salio_del_equipo_y_no_menciona_la_demo(self):
        es, en = (informe_html(self.local, idioma=i) for i in ("es", "en"))
        self.assertIn("Se calculó en local, en el equipo de quien ejecutó la evaluación: el CSV no salió de él",
                      _conclusiones(es))
        self.assertIn("It was calculated locally", _conclusiones(en))
        self.assertNotIn("servidor de la demo", _texto(es))
        self.assertNotIn("demo server", _texto(en))

    def test_la_frase_vieja_todo_se_midio_en_local_no_aparece_en_ningun_caso(self):
        for r in (self.local, self.demo):
            for idioma in IDIOMAS:
                doc = informe_html(r, idioma=idioma)
                for vieja in ("Todo se midió en local", "Se ha generado en local",
                              "Everything was measured locally", "It was generated locally"):
                    self.assertNotIn(vieja, doc)
        for idioma in IDIOMAS:
            for clave, textos in TEXTOS.items():
                self.assertNotIn("generado en local", textos[idioma])


# ---------------------------------------------------------------------------------------------
# Redacción del informe (25-10): una hoja, la frase-veredicto, la puntuación y «VPP y VPN»
# ---------------------------------------------------------------------------------------------

_BLOQUES = {"p", "li", "h1", "h2", "h3", "h4"}
_TAMANO = {"h1": 26, "h2": 19, "h3": 16, "h4": 11.5}


def _altura_estimada_px(doc: str, *, ancho: int = 1000, fuente: float = 14.0) -> float:
    """ESTIMACIÓN (sin navegador) de la altura de la cabecera + la página de conclusiones en un
    ancho de `ancho` px: cada bloque ocupa ceil(caracteres × 0,5 em / ancho útil) líneas de 1,5 em.
    Calibrada con la versión del 24-10, que Chromium midió en ~1.900 px (header + conclusiones, a
    1.000 px): este estimador da 1.821 para ella (-4 %). Una hoja A4 a ese ancho son 1.414 px: lo que
    pasa de ahí ya no cabe en una hoja."""
    bloques: list[tuple[str, str]] = []
    pila: list[str] = []
    actual: list[str] = []

    class _P(HTMLParser):
        ignorar = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("style", "script", "title"):
                self.ignorar += 1
            if tag in _BLOQUES:
                pila.append(tag)
                actual.clear()
            if tag == "div" and dict(attrs).get("class", "").startswith("note"):
                pila.append("note")
                actual.clear()

        def handle_endtag(self, tag):
            if tag in ("style", "script", "title"):
                self.ignorar -= 1
            if pila and (tag == pila[-1] or (tag == "div" and pila[-1] == "note")):
                bloque = pila.pop()
                texto = " ".join("".join(actual).split())
                if texto:
                    bloques.append((bloque, texto))
                actual.clear()

        def handle_data(self, data):
            if not self.ignorar and pila:
                actual.append(data)

    _P().feed(doc)
    util = ancho - 32 - 24
    alto = 0.0
    for bloque, texto in bloques:
        tam = _TAMANO.get(bloque, fuente)
        alto += max(1, math.ceil(len(texto) * tam * 0.5 / util)) * tam * 1.5 + (6 if bloque == "li" else 10)
    return alto


A4_A_1000_PX = 1414


class UnaHojaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _dos()
        cls.docs = {i: informe_html(cls.r, idioma=i) for i in IDIOMAS}

    def test_las_conclusiones_caben_en_UNA_hoja_A4_en_es_y_en_con_dos_puntuaciones_umbral_y_subgrupos(self):
        # el caso de las muestras: dos puntuaciones, umbral en las dos, subgrupos, cifra prometida
        for idioma, doc in self.docs.items():
            cabeza = doc[doc.index("<header"):doc.index("<details")]
            alto = _altura_estimada_px(cabeza)
            self.assertGreater(alto, 400, f"{idioma}: ¿una página en blanco? {alto:.0f} px")
            self.assertLess(alto, 0.85 * A4_A_1000_PX, f"{idioma}: {alto:.0f} px de {A4_A_1000_PX}")

    def test_la_estimacion_distingue_una_pagina_corta_de_la_larga_de_antes(self):
        # control del INSTRUMENTO: el mismo informe con el texto largo de antes (definiciones en llano
        # y advertencias, que hoy están en el anexo) NO cabe: la prueba no pasa por pasar.
        doc = self.docs["es"]
        cabeza = doc[doc.index("<header"):doc.index("<details")]
        en_llano = doc[doc.index("En lenguaje llano"):]
        largo = cabeza + en_llano[:en_llano.index("</section>")] * 2
        self.assertGreater(_altura_estimada_px(largo), 0.85 * A4_A_1000_PX)

    def test_la_pagina_tiene_un_veredicto_por_puntuacion_y_los_limites_son_tres_o_cuatro_vinetas(self):
        for idioma, doc in self.docs.items():
            cabeza = doc[:doc.index("<details")]
            self.assertEqual(cabeza.count('class="veredicto"'), 2)
            limites = cabeza[cabeza.index('<section class="not">'):]
            vinetas = re.findall(r"<li>(.*?)</li>", limites, flags=re.S)
            self.assertIn(len(vinetas), (3, 4), idioma)
            self.assertTrue(all(len(re.sub(r"<[^>]+>", "", v)) <= 260 for v in vinetas), idioma)

    def test_lo_largo_baja_a_los_anexos_y_sigue_en_el_documento(self):
        cabeza_es = _texto(self.docs["es"][:self.docs["es"].index("<details")])
        resto_es = _texto(self.docs["es"][self.docs["es"].index("<details"):])
        for largo in ("Matriz de confusión", "Intercepto", "un número exacto y falso"):
            self.assertNotIn(largo, cabeza_es)
        for largo in ("Matriz de confusión", "Intercepto", "Dicho en llano", "de cada 100 eventos"):
            self.assertIn(largo, resto_es)


class VeredictoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _dos()

    def test_la_frase_del_html_es_la_que_compuso_el_nucleo_y_va_la_primera_de_cada_puntuacion(self):
        for idioma in IDIOMAS:
            doc = informe_html(self.r, idioma=idioma)
            for k, p in enumerate(self.r.informe["puntuaciones"]):
                frase = p["veredicto"]["texto"][idioma]
                bloque = doc.split(f'<div class="cs c{k + 1}">')[1]
                self.assertTrue(bloque.split("</h3>")[1].startswith(
                    f'<p class="veredicto"><b>{frase}</b></p>'), idioma)

    def test_el_html_no_recompone_la_frase_lo_que_cambia_el_nucleo_cambia_el_documento(self):
        cambiado = _con_informe(self.r, lambda i: i["puntuaciones"][0]["veredicto"]["texto"].update(
            es="FRASE DEL NUCLEO", en="CORE SENTENCE"))
        self.assertIn("<b>FRASE DEL NUCLEO</b>", informe_html(cambiado, idioma="es"))
        self.assertIn("<b>CORE SENTENCE</b>", informe_html(cambiado, idioma="en"))


class PuntuacionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _dos()
        cls.una = _evaluar(puntuaciones=[Decl("score_a", "probabilidad", ALTO, umbral=0.3)])

    def test_los_rotulos_llevan_dos_puntos_antes_de_su_texto_en_es_y_en(self):
        for idioma, esperado in (
                ("es", ("Al umbral de hoy: Con el umbral de uso", "Frente a la otra: «score_a» frente a «score_b»")),
                ("en", ("At today's threshold: At the threshold of use", "Against the other score: “score_a” against “score_b”"))):
            texto = _texto(informe_html(self.r, idioma=idioma))
            for frase in esperado:
                self.assertIn(frase, texto)

    def test_ningun_rotulo_en_negrita_queda_pegado_a_su_texto_sin_puntuacion(self):
        for r in (self.r, self.una):
            for idioma in IDIOMAS:
                doc = informe_html(r, idioma=idioma)
                pegados = [m for m in re.findall(r"<b>([^<]+)</b> ", doc)
                           if not m.rstrip().endswith(("?", ":", ".", ")"))]
                self.assertEqual(pegados, [], f"{idioma}: rótulos sin puntuación {pegados}")

    def test_la_diferencia_de_auroc_se_escribe_con_su_valor_y_luego_su_intervalo(self):
        c = self.r.informe["comparacion"]
        es = _texto(informe_html(self.r, idioma="es"))
        self.assertIn("diferencia de AUROC " + f'{c["diferencia"]:+.3f}'.replace(".", ","), es)
        self.assertNotIn("a +0,069: +0,057", es)
        self.assertNotRegex(es, r"IC 95 %: [+\-−]?\d,\d+ a [+\-−]?\d,\d+: [+\-−]?\d")


class VppVpnTextoTest(unittest.TestCase):
    def test_el_motivo_compartido_dice_VPP_y_VPN_en_castellano_y_no_lleva_doble_guion(self):
        from matrixai.estudio.textos import motivo
        m = motivo("ppv_npv_sin_prevalencia_declarada")
        self.assertTrue(m["es"].startswith("VPP y VPN no se publican"))
        self.assertNotIn("PPV", m["es"])
        self.assertTrue(m["en"].startswith("PPV and NPV are not published"))
        for idioma in ("es", "en"):
            self.assertNotIn("--", m[idioma])

    def test_el_informe_sin_representatividad_dice_VPP_y_VPN_sin_doble_guion_en_los_dos_idiomas(self):
        r = _dos(representa=False)
        for idioma, esperado, prohibido in (("es", "VPP y VPN no se publican", "PPV y NPV"),
                                            ("en", "PPV and NPV are not published", "VPP")):
            doc = informe_html(r, idioma=idioma)
            self.assertIn(esperado, _texto(doc))
            self.assertNotIn(prohibido, _texto(doc))
            # el CSS lleva `--fg`: se mira el texto visible; `--key` es la opción de la orden que
            # verifica el recibo (26-10), no un guion doble de redacción
            self.assertNotIn("--", _texto(doc).replace("--key", ""))
            self.assertIn(esperado, _texto(doc[doc.index("<details"):]))   # el motivo vive en el anexo
