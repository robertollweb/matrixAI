# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El informe HTML de «¿Funciona aquí?» — 137-C2.

Convierte el `ResultadoDeEvaluacion` de `evaluar_puntuacion` en UN documento HTML
autocontenido (CSS en línea, gráficas en SVG escritas aquí, sin JavaScript, sin fuentes ni
recursos externos y sin una sola URL), en castellano o en inglés, que un hospital puede abrir
sin conexión, imprimir en A4 o adjuntar. Función pura: ni disco ni reloj; la fecha que
enseña es la del recibo.

**ESTE MÓDULO PINTA; NO MIDE.** Todo número sale del informe del núcleo, tal cual. Lo que el
núcleo redactó en los dos idiomas (avisos, motivos de «no aplica», «lo que este informe no
es», el recibo) se toma de su idioma; aquí solo están los rótulos del documento
(`informe_de_puntuacion_textos.py`).

LO QUE SE DECIDIÓ AL PINTAR (para que no se «simplifique» sin saberlo):

* **Un valor ausente no es un cero.** Una cifra que el núcleo no dio (`valor: None`) se pinta
  como su motivo, nunca como «0». Un intervalo ausente dice «sin intervalo».
* **Sin adjetivos** (regla 16). Las etiquetas junto a una cifra comparan con una referencia
  («el intervalo no incluye 0,5», «fuera del intervalo»); ninguna dice «bueno», «malo» ni
  «aceptable», y el informe no inventa umbrales de «grupo peor».
* **Calibración y curva de fiabilidad SOLO si el núcleo dijo `aplica`.** Una escala de puntos
  dice «no aplica» con el motivo del núcleo y no dibuja la curva.
* **Un subgrupo sin evidencia no publica ni su tamaño** (la regla 13 lo protege por
  privacidad: con tres pacientes, el número casi los identifica). El núcleo ya omite eventos
  y AUROC; aquí tampoco se pinta `n`.
* **Los tramos de fiabilidad sin cifra no se dibujan**; se cuentan.
* **La curva ROC** es `discriminacion.curva_roc` del núcleo (≈40 vértices reales, `{"fpr",
  "tpr"}`, sin filas). Este módulo no la reconstruye: si el campo falta o es `None`, no se
  dibuja y quedan el AUROC con su intervalo y el punto del umbral de uso.
* **Todo texto de usuario se escapa** (columnas, orígenes, valores de subgrupo, actor).
* **La ficha TRIPOD+AI reutiliza el inventario de `expediente_clinico`** —rótulos, motivos,
  dominios y los tres estados cerrados— y los rótulos de lo que solo puede declarar el autor
  de `tripod.py`. Los temas van agrupados y SIN número (la numeración oficial nunca se extrajo
  de forma fiable, ver `85_C6_MAPA_TRIPOD.md`), y pasa por `sin_veredicto_de_riesgo`: es la
  última puerta del invariante 6 del contrato 109. Esa puerta NO se aplica a todo el
  documento: la frase obligatoria de la paradoja del tratamiento («un riesgo alto llevó a
  tratar…») contiene, legítimamente, vocabulario que la puerta bloquea.

Y sigue siendo stdlib puro.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from matrixai.estudio.segmentos import MINIMO_EVENTOS_POR_SEGMENTO, MINIMO_FILAS_POR_SEGMENTO
from matrixai.export.expediente_clinico import (
    DECLARADO,
    DOMINIOS,
    ESTADOS,
    FALTA,
    MEDIDO,
    POR_CLAVE,
    rotulo_de_dominio,
    sin_veredicto_de_riesgo,
)
from matrixai.export.informe_de_puntuacion_textos import IDIOMAS, t
from matrixai.export.tripod import NO_DISPONIBLE  # rótulos de lo que solo declara el autor

__all__ = ["CLAVES_DEL_INVENTARIO_USADAS", "informe_html", "items_tripod", "resolver_ruta"]

#: Los avisos del núcleo que se pintan en ámbar: los que dicen que las cifras pueden estar
#: sesgadas o leídas al revés. Los demás se pintan neutros.
_AVISOS_QUE_IMPORTAN = frozenset({"falta_concentrada_en_un_desenlace",
                                  "direccion_posiblemente_mal_declarada"})

#: Las claves del inventario de `expediente_clinico` de las que esta ficha toma rótulo (y, si
#: procede, motivo). Que sea una tupla a la vista lo hace comprobable: si alguien renombra una
#: clave allí, el import de este módulo revienta en vez de dejar un tema sin rótulo.
CLAVES_DEL_INVENTARIO_USADAS: tuple[str, ...] = (
    "fuente_de_los_datos", "poblacion", "criterios_de_inclusion", "huella_del_dataset",
    "predictores", "definicion_del_desenlace", "momento_del_desenlace",
    "prevalencia_observada", "prevalencia_declarada", "metricas", "calibracion", "umbrales",
    "curva_de_decision", "subgrupos", "politica_de_faltantes", "proceso_actual",
    "alcance_de_validacion", "uso_previsto",
)
for _k in CLAVES_DEL_INVENTARIO_USADAS:
    if _k not in POR_CLAVE:  # pragma: no cover - guarda de import
        raise ImportError(f"el inventario de expediente_clinico ya no tiene {_k!r}")

#: R2: el rótulo de cada valor de `casos_en_su_desarrollo` (un mapa a la vista, no una clave compuesta).
_DEV = {"si": "dev_si", "no": "dev_no", "no_se_sabe": "dev_no_se_sabe"}
_ETIQUETA_DE_ESTADO = {MEDIDO: "tp_medido", DECLARADO: "tp_declarado", FALTA: "tp_falta"}
assert set(_ETIQUETA_DE_ESTADO) == set(ESTADOS), "los tres estados son cerrados"


# ===========================================================================================
# Formato
# ===========================================================================================

class _F:
    """Cifras escritas como las escribe cada idioma: `0,838` / `0.838`, `1.500` / `1,500`."""

    def __init__(self, idioma: str) -> None:
        self.idioma = idioma

    def n(self, x: float, d: int = 3) -> str:
        s = f"{x:.{d}f}"
        if float(s) == 0.0:
            s = s.lstrip("-")
        return s.replace(".", ",") if self.idioma == "es" else s

    def signed(self, x: float, d: int = 3) -> str:
        s = self.n(x, d)
        if float(f"{x:.{d}f}") > 0:
            return "+" + s
        return s

    def pct(self, x: float, d: int = 1) -> str:
        return self.n(x * 100.0, d) + " %"

    def miles(self, n: int) -> str:
        s = f"{int(n):,}"
        return s.replace(",", ".") if self.idioma == "es" else s

    def u(self, x: float) -> str:
        """Un umbral o un valor de escala tal cual lo declaró quien evalúa (sin ceros de más)."""
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s.replace(".", ",") if self.idioma == "es" else s


def _e(texto: Any) -> str:
    return html.escape("" if texto is None else str(texto), quote=True)


def _bi(textos: Any, idioma: str) -> str:
    """El texto del núcleo en el idioma pedido. Un texto bilingüe `{"es","en"}` se toma por su
    idioma; si no lo es, se devuelve vacío (no se cuela el idioma contrario)."""
    if isinstance(textos, dict):
        return str(textos.get(idioma, ""))
    return ""


def resolver_ruta(raiz: dict[str, Any], ruta: str) -> Any:
    """`informe#puntuaciones[0].al_umbral.umbral` → el dato. Reviente (KeyError) si no existe:
    es lo que usan las pruebas para comprobar que cada línea de la ficha traza a un campo."""
    _, _, camino = ruta.partition("#")
    actual: Any = raiz
    for trozo in camino.split("."):
        m = re.fullmatch(r"([^\[\]]+)((?:\[\d+\])*)", trozo)
        if m is None:
            raise KeyError(ruta)
        actual = actual[m.group(1)]
        for indice in re.findall(r"\[(\d+)\]", m.group(2)):
            actual = actual[int(indice)]
    return actual


# ===========================================================================================
# Pequeñas piezas de HTML
# ===========================================================================================

def _pill(texto: str, tono: str = "neutral") -> str:
    return f'<span class="pill {tono}">{_e(texto)}</span>'


def _tarjeta(titulo: str, pill: str, valor: str, ic: str, nota: str, *, na: bool = False,
             texto: bool = False) -> str:
    clase = "tile na" if na else "tile"
    clase_v = "v txt" if texto else "v"
    cola_ic = '<span class="ci">' + ic + "</span>" if ic else ""
    cola_q = '<span class="q">' + nota + "</span>" if nota else ""
    return (f'<div class="{clase}"><h4>{_e(titulo)}</h4>{pill}'
            f'<span class="{clase_v}">{valor}</span>{cola_ic}{cola_q}</div>')


def _ic(m: dict[str, Any] | None, fmt, idioma: str, nivel_fmt: str = "95 %") -> str:
    """El intervalo de una métrica del núcleo escrito, o por qué no hay. Nunca un 0 inventado."""
    if not m or m.get("valor") is None:
        return ""
    ic = m.get("ic")
    if ic:
        return t("ic_linea", idioma, nivel=nivel_fmt, a=fmt(ic["bajo"]), b=fmt(ic["alto"]))
    razon = _bi(m.get("ic_no_disponible"), idioma)
    return razon or t("ic_sin", idioma)


def _celda_valor(m: dict[str, Any] | None, fmt, idioma: str) -> str:
    """El valor de una métrica; si el núcleo no lo dio, su motivo (nunca «0»)."""
    if not m:
        return "—"
    if m.get("valor") is None:
        return _e(_bi(m.get("no_definido"), idioma) or "—")
    return _e(fmt(m["valor"]))


# ===========================================================================================
# Gráficas (SVG escrito aquí, sin librerías)
# ===========================================================================================

class _Ejes:
    W, H, L, T, R, B = 320, 300, 46, 12, 22, 42

    def __init__(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.w = self.W - self.L - self.R
        self.h = self.H - self.T - self.B

    def X(self, v: float) -> float:
        return self.L + (v - self.x0) / (self.x1 - self.x0) * self.w

    def Y(self, v: float) -> float:
        return self.T + (1 - (v - self.y0) / (self.y1 - self.y0)) * self.h


def _ticks_bonitos(lo: float, hi: float, n: int = 5) -> list[float]:
    """Marcas de eje en pasos de 1, 2 o 5 por una potencia de diez: 0,05 · 0,10 · 0,15, no
    0,07 · 0,17 · 0,28."""
    span = hi - lo
    if span <= 0:
        return [lo]
    bruto = span / n
    pot = 10 ** math.floor(math.log10(bruto))
    paso = next(m * pot for m in (1, 2, 5, 10) if m * pot >= bruto)
    k = math.ceil(lo / paso - 1e-9)
    marcas = []
    while k * paso <= hi + 1e-9:
        marcas.append(round(k * paso, 10))
        k += 1
    return marcas


def _marco(ej: _Ejes, aria: str, etiqueta_x: str, etiqueta_y: str, chart: str,
           fx, fy, cuerpo: str, ticks: Sequence[float] = (0, .25, .5, .75, 1),
           ticks_y: Sequence[float] | None = None) -> str:
    g = []
    for tk in ticks:
        if ej.x0 <= tk <= ej.x1:
            g.append(f'<line class="gridl" x1="{ej.X(tk):.1f}" y1="{ej.Y(ej.y0):.1f}" '
                     f'x2="{ej.X(tk):.1f}" y2="{ej.Y(ej.y1):.1f}"/>'
                     f'<text class="tl" x="{ej.X(tk):.1f}" y="{ej.Y(ej.y0) + 14:.1f}" '
                     f'text-anchor="middle">{_e(fx(tk))}</text>')
    for tk in (ticks_y if ticks_y is not None else ticks):
        if ej.y0 <= tk <= ej.y1:
            g.append(f'<line class="gridl" x1="{ej.X(ej.x0):.1f}" y1="{ej.Y(tk):.1f}" '
                     f'x2="{ej.X(ej.x1):.1f}" y2="{ej.Y(tk):.1f}"/>'
                     f'<text class="tl" x="{ej.L - 6}" y="{ej.Y(tk) + 4:.1f}" '
                     f'text-anchor="end">{_e(fy(tk))}</text>')
    g.append(cuerpo)
    g.append(f'<rect class="ax" x="{ej.L}" y="{ej.T}" width="{ej.w}" height="{ej.h}"/>')
    g.append(f'<text class="tl" x="{ej.L + ej.w / 2:.1f}" y="{ej.H - 4}" '
             f'text-anchor="middle">{_e(etiqueta_x)}</text>')
    g.append(f'<text class="tl" transform="translate(11 {ej.T + ej.h / 2:.1f}) rotate(-90)" '
             f'text-anchor="middle">{_e(etiqueta_y)}</text>')
    return (f'<svg data-chart="{chart}" viewBox="0 0 {ej.W} {ej.H}" role="img" '
            f'aria-label="{_e(aria)}">{"".join(g)}</svg>')


def _svg_auroc(ps: Sequence[dict[str, Any]], idioma: str, f: _F) -> str:
    """El AUROC de cada puntuación con su intervalo, frente a 0,5, y la cifra prometida."""
    W, L, R, fila, T = 320, 14, 14, 44, 10
    bajos = [p["discriminacion"]["auroc"]["ic"]["bajo"] for p in ps
             if (p["discriminacion"]["auroc"].get("ic"))]
    valores = [p["discriminacion"]["auroc"]["valor"] for p in ps
               if p["discriminacion"]["auroc"].get("valor") is not None]
    prom = [p["cifra_prometida"]["prometida"] for p in ps if p.get("cifra_prometida")]
    minimo = min([0.5, *bajos, *valores, *prom])
    x0 = max(0.0, math.floor((minimo - 0.02) * 10) / 10)
    x1 = 1.0
    w = W - L - R

    def X(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * w

    alto = T + fila * len(ps) + 30
    g = []
    marcas = _ticks_bonitos(x0, x1, 6)
    for tk in marcas:
        g.append(f'<line class="gridl" x1="{X(tk):.1f}" y1="{T}" x2="{X(tk):.1f}" '
                 f'y2="{alto - 24}"/><text class="tl" x="{X(tk):.1f}" y="{alto - 10}" '
                 f'text-anchor="middle">{_e(f.n(tk, 2))}</text>')
    g.append(f'<line class="diag" x1="{X(0.5):.1f}" y1="{T}" x2="{X(0.5):.1f}" y2="{alto - 24}"/>')
    for k, p in enumerate(ps):
        y = T + fila * k + 28
        m = p["discriminacion"]["auroc"]
        g.append(f'<text class="tl lbl" x="{L}" y="{y - 14}">{_e(p["columna"])}</text>')
        if m.get("valor") is None:
            continue
        ic = m.get("ic")
        if ic:
            g.append(f'<line class="s{k + 1}" x1="{X(ic["bajo"]):.1f}" y1="{y}" '
                     f'x2="{X(ic["alto"]):.1f}" y2="{y}"/>'
                     f'<line class="s{k + 1}" x1="{X(ic["bajo"]):.1f}" y1="{y - 4}" '
                     f'x2="{X(ic["bajo"]):.1f}" y2="{y + 4}"/>'
                     f'<line class="s{k + 1}" x1="{X(ic["alto"]):.1f}" y1="{y - 4}" '
                     f'x2="{X(ic["alto"]):.1f}" y2="{y + 4}"/>')
        g.append(f'<circle class="f{k + 1}" cx="{X(m["valor"]):.1f}" cy="{y}" r="4.5">'
                 f'<title>AUROC {_e(f.n(m["valor"]))}</title></circle>')
        pr = p.get("cifra_prometida")
        if pr:
            cx = X(pr["prometida"])
            g.append(f'<path class="prom" d="M{cx:.1f} {y - 15}l5 5l-5 5l-5 -5z"><title>'
                     f'{_e(t("g_auroc_prometida", idioma))} {_e(f.n(pr["prometida"]))}'
                     f'</title></path>')
    return (f'<svg data-chart="auroc" viewBox="0 0 {W} {alto}" role="img" '
            f'aria-label="{_e(t("g_auroc_aria", idioma))}">{"".join(g)}</svg>')


def _svg_roc(ps: Sequence[dict[str, Any]], idioma: str, f: _F) -> str | None:
    """El espacio ROC: la diagonal del azar, la curva si el núcleo la trae y el punto del
    umbral de uso de cada puntuación con el intervalo de cada eje. `None` si no hay nada que
    poner (una diagonal sola no dice nada)."""
    cuerpo = []
    algo = False
    ej = _Ejes(0, 1, 0, 1)
    cuerpo.append(f'<line class="diag" x1="{ej.X(0):.1f}" y1="{ej.Y(0):.1f}" '
                  f'x2="{ej.X(1):.1f}" y2="{ej.Y(1):.1f}"/>')
    for k, p in enumerate(ps):
        curva = (p["discriminacion"] or {}).get("curva_roc")
        if curva:
            pts = " ".join(f'{ej.X(q["fpr"]):.1f},{ej.Y(q["tpr"]):.1f}' for q in curva)
            cuerpo.append(f'<polyline class="l{k + 1}" points="{pts}"/>')
            algo = True
        u = p.get("al_umbral") or {}
        s, e = u.get("sensibilidad"), u.get("especificidad")
        if u.get("declarado") and s and e and s.get("valor") is not None \
                and e.get("valor") is not None:
            x, y = 1 - e["valor"], s["valor"]
            if e.get("ic"):
                cuerpo.append(f'<line class="s{k + 1}" x1="{ej.X(1 - e["ic"]["alto"]):.1f}" '
                              f'y1="{ej.Y(y):.1f}" x2="{ej.X(1 - e["ic"]["bajo"]):.1f}" '
                              f'y2="{ej.Y(y):.1f}"/>')
            if s.get("ic"):
                cuerpo.append(f'<line class="s{k + 1}" x1="{ej.X(x):.1f}" '
                              f'y1="{ej.Y(s["ic"]["bajo"]):.1f}" x2="{ej.X(x):.1f}" '
                              f'y2="{ej.Y(s["ic"]["alto"]):.1f}"/>')
            cuerpo.append(f'<circle class="f{k + 1}" cx="{ej.X(x):.1f}" cy="{ej.Y(y):.1f}" '
                          f'r="4.5"><title>{_e(p["columna"])}: '
                          f'{_e(t("m_sens", idioma))} {_e(f.pct(y))}, '
                          f'{_e(t("m_esp", idioma))} {_e(f.pct(e["valor"]))}</title></circle>')
            algo = True
    if not algo:
        return None
    return _marco(ej, t("g_roc_aria", idioma), t("g_roc_x", idioma), t("g_roc_y", idioma),
                  "roc", lambda v: f.n(v, 2), lambda v: f.n(v, 2), "".join(cuerpo))


def _svg_fiabilidad(ps: Sequence[dict[str, Any]], idioma: str, f: _F) -> tuple[str, int] | None:
    """La curva de fiabilidad de las puntuaciones para las que el núcleo dijo que la
    calibración APLICA. Una escala de puntos no entra jamás. Devuelve (svg, tramos sin cifra)."""
    ej = _Ejes(0, 1, 0, 1)
    cuerpo = [f'<line class="diag" x1="{ej.X(0):.1f}" y1="{ej.Y(0):.1f}" '
              f'x2="{ej.X(1):.1f}" y2="{ej.Y(1):.1f}"/>']
    sin_cifra = 0
    hay = False
    for k, p in enumerate(ps):
        cal = p.get("calibracion") or {}
        fia = cal.get("fiabilidad")
        if not cal.get("aplica") or not fia:
            continue
        con_cifra = [b for b in fia["bins"] if b.get("tasa_observada") is not None]
        sin_cifra += len(fia["bins"]) - len(con_cifra)
        if not con_cifra:
            continue
        hay = True
        maxn = max(b["n"] for b in con_cifra)
        pts = " ".join(f'{ej.X(b["probabilidad_media"]):.1f},{ej.Y(b["tasa_observada"]):.1f}'
                       for b in con_cifra)
        cuerpo.append(f'<polyline class="l{k + 1}" points="{pts}"/>')
        for b in con_cifra:
            r = 2.5 + 5 * math.sqrt(b["n"] / maxn)
            tip = t("g_fiab_tip", idioma, n=f.miles(b["n"]),
                    p=f.pct(b["probabilidad_media"]), o=f.pct(b["tasa_observada"]))
            cuerpo.append(f'<circle class="f{k + 1}" cx="{ej.X(b["probabilidad_media"]):.1f}" '
                          f'cy="{ej.Y(b["tasa_observada"]):.1f}" r="{r:.1f}">'
                          f'<title>{_e(p["columna"])}: {_e(tip)}</title></circle>')
    if not hay:
        return None
    pct0 = lambda v: f.pct(v, 0)  # noqa: E731
    return (_marco(ej, t("g_fiab_aria", idioma), t("g_fiab_x", idioma), t("g_fiab_y", idioma),
                   "reliability", pct0, pct0, "".join(cuerpo)), sin_cifra)


def _svg_decision(ps: Sequence[dict[str, Any]], idioma: str, f: _F) -> str | None:
    curvas = [(k, p) for k, p in enumerate(ps)
              if (p.get("curva_de_decision") or {}).get("aplica")
              and (p["curva_de_decision"] or {}).get("declarada")
              and p["curva_de_decision"].get("puntos")]
    if not curvas:
        return None
    xs = [q["umbral"] for _, p in curvas for q in p["curva_de_decision"]["puntos"]]
    ys = [v for _, p in curvas for q in p["curva_de_decision"]["puntos"]
          for v in (q["beneficio_neto"], q["tratar_a_todos"], 0.0)]
    x0, x1 = min(xs), max(xs)
    if x1 - x0 < 1e-9:
        x0, x1 = max(0.0, x0 - 0.05), min(1.0, x1 + 0.05)
    pad = (max(ys) - min(ys)) * 0.08 or 0.05
    ej = _Ejes(x0, x1, min(ys) - pad, max(ys) + pad)
    cuerpo = [f'<line class="diag" x1="{ej.X(x0):.1f}" y1="{ej.Y(0):.1f}" '
              f'x2="{ej.X(x1):.1f}" y2="{ej.Y(0):.1f}"/>']
    for k, p in curvas:
        pts = p["curva_de_decision"]["puntos"]
        todos = " ".join(f'{ej.X(q["umbral"]):.1f},{ej.Y(q["tratar_a_todos"]):.1f}' for q in pts)
        mio = " ".join(f'{ej.X(q["umbral"]):.1f},{ej.Y(q["beneficio_neto"]):.1f}' for q in pts)
        cuerpo.append(f'<polyline class="all" points="{todos}"/>')
        cuerpo.append(f'<polyline class="l{k + 1}" points="{mio}"/>')
        for q in pts:
            cuerpo.append(f'<circle class="f{k + 1}" cx="{ej.X(q["umbral"]):.1f}" '
                          f'cy="{ej.Y(q["beneficio_neto"]):.1f}" r="3"><title>'
                          f'{_e(p["columna"])}: {_e(f.pct(q["umbral"], 0))} → '
                          f'{_e(f.n(q["beneficio_neto"]))}</title></circle>')
    ticks_x = sorted(set(xs)) if len(set(xs)) <= 6 else _ticks_bonitos(x0, x1, 5)
    ticks_y = _ticks_bonitos(ej.y0, ej.y1, 5)
    return _marco(ej, t("g_dca_aria", idioma), t("g_dca_x", idioma), t("g_dca_y", idioma),
                  "decision", lambda v: f.pct(v, 0), lambda v: f.n(v, 2 if (ticks_y[-1] - ticks_y[0]) / max(1, len(ticks_y) - 1) >= 0.01 else 3), "".join(cuerpo),
                  ticks=ticks_x, ticks_y=ticks_y)


# ===========================================================================================
# Secciones
# ===========================================================================================

def _nombre_tipo(tipo: str, idioma: str) -> str:
    return t(f"decl_tipo_{tipo}", idioma)


def _umbral_texto(p: dict[str, Any], f: _F, idioma: str) -> str | None:
    u = (p.get("declaracion") or {}).get("umbral")
    if u is None:
        return None
    s = f.u(u)
    return s + " %" if p["declaracion"]["tipo"] == "probabilidad_pct" else s


def _referencia_linea(pr: dict[str, Any], auroc: dict[str, Any], f: _F, idioma: str) -> str:
    """R4: la cifra prometida es una REFERENCIA PUBLICADA frente a una ESTIMACIÓN LOCAL. Se
    dan las dos y dónde cae la primera respecto al intervalo de la segunda; no se dice que la
    cifra «se cumpla» o «falle». Sin intervalo, el motivo del núcleo."""
    if pr.get("resultado") in ("dentro", "fuera"):
        return t("ref_linea", idioma, r=f.n(pr["prometida"]), e=f.n(auroc["valor"]),
                 pos=t("ref_dentro" if pr["resultado"] == "dentro" else "ref_fuera", idioma))
    return _bi(pr.get("motivo"), idioma)


def _rotulo_de_alcance(seccion: dict[str, Any] | None, idioma: str) -> str:
    """R3: «describen la muestra aportada, no la población de uso» + por qué, si la sección
    lo lleva (solo con «la muestra NO representa»). Vacío si no."""
    r = (seccion or {}).get("rotulo_de_alcance")
    if not r:
        return ""
    return (_bi(r.get("rotulo"), idioma)[:1].upper() + _bi(r.get("rotulo"), idioma)[1:] + ". "
            + _bi(r.get("porque"), idioma))


def _tarjetas(p: dict[str, Any], f: _F, idioma: str) -> str:
    out = []
    # --- ¿ordena? ---
    a = p["discriminacion"]["auroc"]
    ap = p["discriminacion"]["average_precision"]
    if a.get("valor") is None:
        out.append(_tarjeta(t("q_ordena", idioma), "", _e(_bi(a.get("no_definido"), idioma)),
                            "", "", na=True))
    else:
        ic = a.get("ic")
        if ic is None:
            pill = ""
        elif ic["alto"] < 0.5:
            pill = _pill(t("pill_ic_bajo_azar", idioma), "warn")
        elif ic["bajo"] > 0.5:
            pill = _pill(t("pill_ic_sobre_azar", idioma))
        else:
            pill = _pill(t("pill_ic_con_azar", idioma))
        nota = []
        if ap.get("valor") is not None:
            nota.append(_e(t("ap_linea", idioma, v=f.n(ap["valor"]),
                             z=f.n(p["discriminacion"]["average_precision_del_azar"]))))
        pr = p.get("cifra_prometida")
        if pr:
            nota.append(_e(_referencia_linea(pr, a, f, idioma)))
            nota.append(_e(_bi(pr.get("advertencia"), idioma)))
        ic_txt = _e(t("auroc_rotulo", idioma) + " · " + (_ic(a, f.n, idioma)))
        out.append(_tarjeta(t("q_ordena", idioma), pill, _e(f.n(a["valor"])), ic_txt,
                            "<br>".join(nota)))
    # --- ¿acierta en el riesgo? ---
    cal = p["calibracion"]
    if not cal.get("aplica"):
        out.append(_tarjeta(t("q_riesgo", idioma), "", _e(t("na", idioma)), "",
                            _e(_bi(cal.get("motivo"), idioma)), na=True))
    else:
        g = cal["en_grande"]
        ia, ib = cal["intercepto"], cal["pendiente"]
        if g.get("valor") is None:
            out.append(_tarjeta(t("q_riesgo", idioma), "", _e(_bi(g.get("no_definido"), idioma)),
                                "", "", na=True))
        else:
            pill = ""
            if ia.get("ic"):
                pill = _pill(t("pill_interc_excluye" if ia["ic"]["bajo"] > 0 or ia["ic"]["alto"] < 0
                               else "pill_interc_incluye", idioma))
            ic_txt = _e(t("citl_linea", idioma,
                          ic=_ic(g, lambda x: f.signed(x), idioma)))

            def par(m: dict[str, Any]) -> str:
                return _ic(m, lambda x: f.n(x, 2), idioma) or "—"

            def num(m: dict[str, Any], d: int = 2) -> str:
                return f.n(m["valor"], d) if m.get("valor") is not None else "—"

            nota = t("calib_detalle", idioma, a=num(ia), ia=par(ia), b=num(ib), ib=par(ib),
                     br=num(cal["brier"], 3),
                     ece=f.n(cal["fiabilidad"]["ece"], 3)
                     if cal.get("fiabilidad", {}).get("ece") is not None else "—")
            rotulo = _rotulo_de_alcance(cal, idioma)
            out.append(_tarjeta(
                t("q_riesgo", idioma), pill, _e(f.signed(g["valor"])), ic_txt,
                _e(nota) + "<br>" + _e(t("citl_pp", idioma, v=f.signed(g["valor"] * 100, 1)))
                + (f'<br><b>{_e(rotulo)}</b>' if rotulo else "")))
    # --- al umbral de hoy ---
    u = p["al_umbral"]
    if not u.get("declarado"):
        out.append(_tarjeta(t("q_umbral", idioma), _pill(t("pill_sin_umbral", idioma)),
                            _e(t("na", idioma)), "", _e(_bi(u.get("motivo"), idioma)), na=True))
    else:
        s, e = u["sensibilidad"], u["especificidad"]
        titulo = t("umbral_titulo", idioma, u=_umbral_texto(p, f, idioma))
        out.append(_tarjeta(
            titulo, _pill(t("pill_avisos", idioma, k=f.miles(u["avisos"]))),
            _celda_valor(s, f.pct, idioma),
            _e(t("umbral_sens_linea", idioma, ic=_ic(s, f.pct, idioma))) if s.get("valor") is not None else "",
            _e(t("umbral_esp_linea", idioma, v=f.pct(e["valor"]) if e.get("valor") is not None else "—",
                 ic=_ic(e, f.pct, idioma) or "—"))))
    # --- ¿con quién falla? ---
    out.append(_tarjeta_grupos(p, f, idioma))
    return '<div class="tiles">' + "".join(out) + "</div>"


def _grupos_de(p: dict[str, Any]) -> tuple[list[tuple[str, float]], int, int]:
    medidos: list[tuple[str, float]] = []
    insuf = total = 0
    for s in p.get("subgrupos") or []:
        for g in s["grupos"]:
            total += 1
            if not g.get("soporte_suficiente"):
                insuf += 1
            elif g.get("auroc", {}).get("valor") is not None:
                medidos.append((f'{s["columna"]} = {g["valor"] if not g.get("sin_dato") else "—"}',
                                g["auroc"]["valor"]))
    return medidos, insuf, total


def _tarjeta_grupos(p: dict[str, Any], f: _F, idioma: str) -> str:
    medidos, insuf, total = _grupos_de(p)
    if not total:
        return _tarjeta(t("q_grupos", idioma), _pill(t("pill_sin_grupos", idioma)),
                        _e(t("na", idioma)), "", _e(t("grupos_ninguno", idioma)), na=True)
    notas = []
    if len(medidos) >= 2:
        lo, hi = min(medidos, key=lambda x: x[1]), max(medidos, key=lambda x: x[1])
        notas.append(_e(t("grupos_extremos", idioma, a=lo[0], va=f.n(lo[1]), b=hi[0],
                          vb=f.n(hi[1]))))
    notas.append(_e(t("grupos_insuf", idioma, k=insuf) if insuf else t("grupos_todos", idioma)))
    return _tarjeta(t("q_grupos", idioma), "", _e(t("grupos_medidos", idioma, k=len(medidos))),
                    "", "<br>".join(notas), texto=True)


def _tabla_umbral(p: dict[str, Any], f: _F, idioma: str) -> str:
    u = p["al_umbral"]
    if not u.get("declarado"):
        return ""
    filas = []

    def fila(clave: str, m: dict[str, Any] | None, sentido: str) -> str:
        if m is None:
            return ""
        if m.get("valor") is None:
            return (f'<tr><td>{_e(t(clave, idioma))}</td><td class="n">—</td><td class="n">—</td>'
                    f'<td class="muted">{_e(_bi(m.get("no_definido"), idioma))}</td></tr>')
        return (f'<tr><td>{_e(t(clave, idioma))}</td><td class="n">{_e(f.pct(m["valor"]))}</td>'
                f'<td class="n">{_e(_ic_corto(m, f.pct, idioma))}</td>'
                f'<td>{_e(sentido)}</td></tr>')

    filas.append(fila("m_sens", u["sensibilidad"], t("d_sens", idioma)))
    filas.append(fila("m_esp", u["especificidad"], t("d_esp", idioma)))
    filas.append(fila("m_vpp", u["vpp"], t("d_vpp", idioma)))
    filas.append(fila("m_vpn", u["vpn"], t("d_vpn", idioma)))
    bn = u.get("beneficio_neto")
    if bn:
        rotulo_bn = _rotulo_de_alcance(bn, idioma)
        filas.append(f'<tr><td>{_e(t("m_bn", idioma))}</td><td class="n">{_e(f.n(bn["modelo"]))}'
                     f'</td><td class="n">—</td><td>'
                     f'{_e(t("d_bn", idioma, todos=f.n(bn["tratar_a_todos"]), nadie=f.n(bn["tratar_a_nadie"])))}'
                     + (f' <b>{_e(rotulo_bn)}</b>' if rotulo_bn else "")
                     + '</td></tr>')
    m = u["matriz"]
    regla = t("regla_mayor" if u["regla_de_aviso"] == "mayor_o_igual" else "regla_menor", idioma)
    pie = [t("matriz", idioma, tp=f.miles(m["tp"]), fp=f.miles(m["fp"]), fn=f.miles(m["fn"]),
             tn=f.miles(m["tn"]), regla=regla)]
    if u.get("nota"):
        pie.append(_bi(u["nota"], idioma))
    cab = "".join(f"<th>{_e(t(c, idioma))}</th>" for c in ("th_medida", "th_valor", "th_ic", "th_sentido"))
    return (f'<div class="blk"><h4>{_e(t("s_umbral", idioma, u=_umbral_texto(p, f, idioma)))} · '
            f'{_e(p["columna"])}</h4><div class="tw"><table><thead><tr>{cab}</tr></thead>'
            f'<tbody>{"".join(filas)}</tbody></table></div>'
            f'<p class="small muted">{" ".join(_e(x) for x in pie)}</p></div>')


def _ic_corto(m: dict[str, Any], fmt, idioma: str) -> str:
    ic = m.get("ic")
    if ic:
        return f'{fmt(ic["bajo"])} – {fmt(ic["alto"])}'
    return _bi(m.get("ic_no_disponible"), idioma) or t("ic_sin", idioma)


def _tabla_subgrupos(p: dict[str, Any], f: _F, idioma: str) -> str:
    sub = p.get("subgrupos") or []
    if not sub:
        return ""
    con_cal = bool(p["calibracion"].get("aplica"))
    cols = ["th_grupo", "th_casos", "th_eventos", "th_auroc"] + (["th_citl"] if con_cal else [])
    cab = "".join(f"<th>{_e(t(c, idioma))}</th>" for c in cols)
    filas = []
    for s in sub:
        for g in s["grupos"]:
            etiqueta = t("sin_dato_grupo", idioma) if g.get("sin_dato") else g["valor"]
            nombre = f'{_e(s["columna"])} = <b>{_e(etiqueta)}</b>'
            if not g.get("soporte_suficiente"):
                filas.append(
                    f'<tr><td>{nombre}</td><td colspan="{len(cols) - 1}" class="muted">'
                    f'{_e(t("evidencia_insuficiente", idioma, f=MINIMO_FILAS_POR_SEGMENTO, e=MINIMO_EVENTOS_POR_SEGMENTO))}'
                    f'</td></tr>')
                continue
            celdas = [f'<td class="n">{_e(f.miles(g["n"]))}</td>',
                      f'<td class="n">{_e(f.miles(g["eventos"]))}</td>',
                      f'<td class="n">{_celda_valor(g.get("auroc"), f.n, idioma)}'
                      + (f' ({_e(_ic_corto(g["auroc"], f.n, idioma))})'
                         if (g.get("auroc") or {}).get("valor") is not None else "")
                      + '</td>']
            if con_cal:
                celdas.append(f'<td class="n">'
                              f'{_celda_valor(g.get("calibracion_en_grande"), f.signed, idioma)}</td>')
            filas.append(f"<tr><td>{nombre}</td>{''.join(celdas)}</tr>")
    return (f'<div class="blk"><h4>{_e(t("s_subgrupos", idioma))} · {_e(p["columna"])}</h4>'
            f'<div class="tw"><table><thead><tr>{cab}</tr></thead><tbody>{"".join(filas)}'
            f'</tbody></table></div><p class="small muted">{_e(t("subgrupos_pie", idioma))}</p></div>')


def _comparacion(inf: dict[str, Any], f: _F, idioma: str) -> str:
    c = inf.get("comparacion")
    if not c:
        return ""
    if not c.get("posible"):
        return (f'<section class="blk"><h3>{_e(t("s_comparacion", idioma))}</h3>'
                f'<p>{_e(_bi(c.get("motivo"), idioma))}</p></section>')
    a, b = c["columna_a"], c["columna_b"]
    est = {"a_mayor_que_b": ("frente_a_mayor", "neutral"),
           "a_menor_que_b": ("frente_b_mayor", "neutral"),
           "el_intervalo_incluye_el_cero": ("frente_sin_dif", "neutral")}.get(c.get("resultado"))
    if c.get("diferencia") is None:
        cuerpo = _e(_bi(c.get("motivo"), idioma) or "—")
        valor, ic_txt, pill = cuerpo, "", ""
    else:
        pill = _pill(t(est[0], idioma, a=a, b=b), est[1]) if est else ""
        valor = _e(f.signed(c["diferencia"]))
        ic_txt = _e(t("frente_dif_linea", idioma, ic=_ic(
            {"valor": c["diferencia"], "ic": c.get("ic"), "ic_no_disponible": c.get("ic_no_disponible")},
            lambda x: f.signed(x), idioma)))
    nota = [t("frente_filas", idioma, n=f.miles(c["filas_comunes"]), e=f.miles(c["eventos_comunes"])),
            t("frente_directa", idioma)]
    if (c.get("auroc_a") or {}).get("valor") is not None and (c.get("auroc_b") or {}).get("valor") is not None:
        nota.append(t("frente_valores", idioma, a=a, b=b, va=f.n(c["auroc_a"]["valor"]),
                      vb=f.n(c["auroc_b"]["valor"])))
    tarjeta = _tarjeta(t("frente_titulo", idioma, a=a, b=b), pill, valor, ic_txt,
                       "<br>".join(_e(x) for x in nota))
    return (f'<section class="blk"><h3>{_e(t("s_comparacion", idioma))}</h3>'
            f'<div class="tiles one">{tarjeta}</div></section>')


def _avisos(inf: dict[str, Any], idioma: str) -> str:
    vistos: set[str] = set()
    notas = []

    def poner(a: dict[str, Any]) -> None:
        txt = _bi(a, idioma)
        if txt and txt not in vistos:
            vistos.add(txt)
            tono = " warn" if a.get("id") in _AVISOS_QUE_IMPORTAN else ""
            notas.append(f'<div class="note{tono}">{_e(txt)}</div>')

    for a in inf.get("avisos") or []:
        poner(a)
    for p in inf["puntuaciones"]:
        for a in p.get("avisos") or []:
            poner(a)
    ind = (inf.get("entrada") or {}).get("indice_sin_nombre_texto")
    if ind:
        poner(ind)
    if not notas:
        return ""
    return f'<section class="blk"><h3>{_e(t("s_avisos", idioma))}</h3>{"".join(notas)}</section>'


def _rotulos_unicos(rotulos: Sequence[str]) -> str:
    vistos: list[str] = []
    for r in rotulos:
        if r and r not in vistos:
            vistos.append(r)
    return " ".join(vistos)


def _graficas(inf: dict[str, Any], f: _F, idioma: str) -> str:
    ps = inf["puntuaciones"]
    leyenda = "".join(
        f'<span><i class="sw s{k + 1}"></i>{_e(p["columna"])}</span>' for k, p in enumerate(ps))
    cajas = []
    cajas.append(
        f'<div class="chart"><h4>{_e(t("g_auroc", idioma))}</h4>{_svg_auroc(ps, idioma, f)}'
        f'<div class="leg">{leyenda}<span>{_e(t("g_auroc_azar", idioma))}</span>'
        f'<span>{_e(t("g_auroc_prometida", idioma))}</span></div></div>')
    roc = _svg_roc(ps, idioma, f)
    if roc:
        cajas.append(f'<div class="chart"><h4>{_e(t("g_roc", idioma))}</h4>{roc}'
                     f'<div class="leg">{leyenda}</div>'
                     f'<p class="small muted">{_e(t("g_roc_pie", idioma))}</p></div>')
    fia = _svg_fiabilidad(ps, idioma, f)
    if fia:
        svg, sin_cifra = fia
        pie = (t("g_fiab_pie", idioma, k=sin_cifra) if sin_cifra else t("g_fiab_pie_sin", idioma))
        rot = _rotulos_unicos([_rotulo_de_alcance(p.get("calibracion"), idioma) for p in ps])
        cajas.append(f'<div class="chart"><h4>{_e(t("g_fiab", idioma))}</h4>{svg}'
                     f'<div class="leg">{leyenda}</div><p class="small muted">{_e(pie)}</p>'
                     + (f'<p class="small"><b>{_e(rot)}</b></p>' if rot else "") + '</div>')
    dca = _svg_decision(ps, idioma, f)
    if dca:
        rot = _rotulos_unicos([_rotulo_de_alcance(p.get("curva_de_decision"), idioma) for p in ps])
        cajas.append(f'<div class="chart"><h4>{_e(t("g_dca", idioma))}</h4>{dca}'
                     f'<div class="leg">{leyenda}<span><i class="sw all"></i>'
                     f'{_e(t("g_dca_todos", idioma))}</span><span>{_e(t("g_dca_nadie", idioma))}'
                     f'</span></div><p class="small muted">{_e(t("g_dca_pie", idioma))}</p>'
                     + (f'<p class="small"><b>{_e(rot)}</b></p>' if rot else "") + '</div>')
    # lo que NO se dibuja, dicho: calibración/decisión que no aplican (con el motivo del núcleo)
    no_aplica = []
    if not fia:
        for p in ps:
            c = p.get("calibracion") or {}
            if not c.get("aplica"):
                no_aplica.append(t("g_no_aplica", idioma, col=p["columna"],
                                   motivo=_bi(c.get("motivo"), idioma)))
    pie = "".join(f'<p class="small muted">{_e(x)}</p>' for x in no_aplica[:2])
    return (f'<section class="blk"><h3>{_e(t("s_graficas", idioma))}</h3>'
            f'<div class="charts">{"".join(cajas)}</div>{pie}</section>')


def _declaracion(p: dict[str, Any], f: _F, idioma: str) -> str:
    d = p["declaracion"]
    partes = [_nombre_tipo(d["tipo"], idioma),
              t("decl_dir_mas_alto" if d["direccion"] == "mas_alto_mas_riesgo"
                else "decl_dir_mas_bajo", idioma)]
    ut = _umbral_texto(p, f, idioma)
    partes.append(t("decl_umbral", idioma, u=ut) if ut is not None else t("decl_sin_umbral", idioma))
    partes.append(t("decl_origen", idioma, o=d["origen"]) if d.get("origen")
                  else t("decl_sin_origen", idioma))
    if d.get("casos_en_su_desarrollo"):
        partes.append(t("decl_desarrollo", idioma,
                        v=t(_DEV[d["casos_en_su_desarrollo"]], idioma)))
    if d.get("cifra_prometida_auroc") is not None:
        partes.append(t("decl_prometida", idioma, v=f.n(d["cifra_prometida_auroc"])))
    fl = p["filas"]
    partes.append(t("decl_filas", idioma, u=f.miles(fl["usadas"]), f=f.miles(fl["sin_puntuacion"])))
    if fl.get("unidades"):
        partes.append(t("decl_unidades", idioma, k=f.miles(fl["unidades"])))
    return "; ".join(partes)


def _entrada(inf: dict[str, Any], resultado: Any, f: _F, idioma: str) -> str:
    en, de, rm = inf["entrada"], inf["desenlace"], inf["remuestreo"]
    kv: list[tuple[str, str]] = []
    nombre = en.get("nombre_del_fichero")
    if nombre:
        kv.append((t("kv_fichero", idioma), _e(nombre)))
    kv.append((t("kv_huella", idioma), f'<code>{_e(en["huella_csv"])}</code>'))
    kv.append((t("kv_lectura", idioma), _e(t("kv_lectura_valor", idioma, c=en["codificacion"],
                                             s=en["separador"], n=f.miles(en["filas_leidas"])))))
    kv.append((t("kv_desenlace", idioma),
               _e(t("kv_desenlace_valor", idioma, col=de["columna"], ev=de["evento"],
                    e=f.miles(de["eventos"]), ne=f.miles(de["no_eventos"])))))
    if de.get("variantes_unidas"):
        for v in de["variantes_unidas"]:
            lista = ", ".join(f'{x["texto"]} ({f.miles(x["filas"])})' for x in v["variantes"])
            kv.append(("", _e(t("kv_variantes", idioma, v=v["valor"], lista=lista))))
    if en["filas_sin_desenlace"] or en["filas_malformadas_fuera"]:
        kv.append(("", _e(t("kv_fuera", idioma, a=f.miles(en["filas_sin_desenlace"]),
                            b=f.miles(en["filas_malformadas_fuera"])))))
    al = inf.get("alcance")
    if al:
        kv.append((t("kv_horizonte", idioma), _e(al["horizonte_del_desenlace"])))
        kv.append((t("kv_momento", idioma), _e(al["momento_de_la_puntuacion"])))
    donde = inf.get("donde_se_calcula")
    if donde:
        kv.append((t("kv_donde", idioma), _e(_bi(donde["texto"], idioma))))
    for p in inf["puntuaciones"]:
        kv.append((f'{t("th_puntuacion", idioma)} · {p["columna"]}',
                   _e(_declaracion(p, f, idioma))))
    kv.append((t("kv_representa", idioma),
               _e(t("si" if inf["la_muestra_representa_a_la_poblacion"] else "no", idioma))))
    diseno = (t("diseno_paciente", idioma, col=rm["columna_de_paciente"])
              if rm.get("columna_de_paciente") else t("diseno_iid", idioma))
    kv.append((t("kv_remuestreo", idioma),
               _e(t("kv_remuestreo_valor", idioma, diseno=diseno, r=f.miles(rm["remuestras"]),
                    nivel=f.pct(rm["nivel"], 0), semilla=rm["semilla"]))))
    cols = [s["columna"] for s in inf["puntuaciones"][0].get("subgrupos") or []]
    kv.append((t("kv_subgrupos", idioma), _e(", ".join(cols) if cols else t("kv_ninguna", idioma))))
    v = inf["versiones"]
    kv.append((t("kv_versiones", idioma),
               _e(t("kv_versiones_valor", idioma, n=v["nucleo"], i=inf["version_del_informe"],
                    c=v["catalogo_de_metricas"][:12] + "…"))))
    filas = "".join(f"<tr><th>{_e(k)}</th><td>{vv}</td></tr>" for k, vv in kv)
    return (f'<section class="blk"><h3>{_e(t("s_entrada", idioma))}</h3>'
            f'<div class="tw"><table class="kv"><tbody>{filas}</tbody></table></div></section>')


def _recibo(resultado: Any, f: _F, idioma: str) -> str:
    r = resultado.recibo
    nivel = resultado.nivel_de_garantia
    kv = [(t("r_id", idioma), f'<code>{_e(r.get("receipt_id"))}</code>'),
          (t("r_fecha", idioma), _e(r.get("created_at") or "—"))]
    actor = (r.get("subject") or {}).get("actor")
    if isinstance(actor, dict):
        actor = _bi(actor, idioma)
    if actor:
        kv.append((t("r_quien", idioma), _e(actor)))
    kv.append((t("kv_huella", idioma), f'<code>{_e((r.get("dataset") or {}).get("digest"))}</code>'))
    huella_inf = ((r.get("output") or {}).get("output_digests") or {}).get("report")
    if huella_inf:
        kv.append((t("r_huella_informe", idioma), f'<code>{_e(huella_inf)}</code>'))
    sobre = resultado.sobre
    if sobre and sobre.get("signatures"):
        s = sobre["signatures"][0]
        kv.append((t("r_firma", idioma),
                   _e(t("r_clave", idioma, k=s.get("keyid"), s=str(s.get("sig", ""))[:16]))))
    else:
        kv.append((t("r_firma", idioma), _e(t("r_sin_firma", idioma))))
    sentido = t("garantia_a1_sentido" if nivel == "A1" else "garantia_a0_sentido", idioma)
    kv.append((t("r_garantia", idioma), _e(sentido)))
    ev = r.get("evidence") or {}
    if ev.get("attests"):
        kv.append((t("r_atestigua", idioma), _e(_bi(ev["attests"], idioma))))
    if ev.get("does_not_attest"):
        kv.append((t("r_no_atestigua", idioma), _e(_bi(ev["does_not_attest"], idioma))))
    filas = "".join(f"<tr><th>{_e(k)}</th><td>{v}</td></tr>" for k, v in kv)
    return (f'<section class="blk"><h3>{_e(t("s_recibo", idioma))}</h3>'
            f'<div class="tw"><table class="kv"><tbody>{filas}</tbody></table></div></section>')


def _no_es(inf: dict[str, Any], idioma: str) -> str:
    ps = "".join(f"<p>{_e(_bi(x, idioma))}</p>" for x in inf.get("lo_que_no_es") or [])
    return f'<section class="not"><h3>{_e(t("s_no_es", idioma))}</h3>{ps}</section>'


# ===========================================================================================
# TRIPOD+AI de validación externa
# ===========================================================================================

@dataclass(frozen=True)
class _Item:
    clave: str
    dominio: str
    rotulo: str
    estado: str                # MEDIDO, DECLARADO o FALTA (los tres, y no hay un cuarto)
    dato: str                  # el valor escrito, o el motivo si falta
    ruta: str                  # `informe#camino` del que sale (NO se pinta: son nombres de campo
                               # en castellano; las pruebas comprueban que existe)
    no_aplica: bool = False    # FALTA porque no aplica, no porque no conste


def items_tripod(resultado: Any, idioma: str) -> list[_Item]:
    """Los temas de TRIPOD+AI de validación externa que esta evaluación puede rellenar, y
    los que dice que no constan. Cada línea lleva el camino del que sale.

    Los rótulos y los motivos vienen del inventario de `expediente_clinico` cuando el tema es
    el mismo; los pocos temas propios de una validación externa (tamaño, puntuación evaluada,
    comparación…) se rotulan en el catálogo de este informe."""
    inf = resultado.informe
    f = _F(idioma)
    ps = inf["puntuaciones"]
    items: list[_Item] = []

    def rot(clave: str) -> str:
        return POR_CLAVE[clave].rotulo[idioma] if clave in POR_CLAVE else t(clave, idioma)

    def dom(clave: str, defecto: str) -> str:
        return POR_CLAVE[clave].dominio if clave in POR_CLAVE else defecto

    def add(clave: str, dominio: str, estado: str, dato: str, ruta: str, *,
            no_aplica: bool = False) -> None:
        items.append(_Item(clave, dom(clave, dominio), rot(clave), estado, dato, ruta, no_aplica))

    def falta(clave: str, dominio: str, motivo: str, ruta: str, *, no_aplica: bool = False) -> None:
        add(clave, dominio, FALTA, motivo, ruta, no_aplica=no_aplica)

    def motivo_inv(clave: str) -> str:
        return POR_CLAVE[clave].motivo_si_falta[idioma]

    en, de = inf["entrada"], inf["desenlace"]
    # ---- participantes ----
    if en.get("nombre_del_fichero"):
        add("fuente_de_los_datos", "participantes", DECLARADO,
            t("vi_fuente", idioma, f=en["nombre_del_fichero"], n=f.miles(en["filas_leidas"])),
            "informe#entrada.nombre_del_fichero")
    else:
        add("fuente_de_los_datos", "participantes", DECLARADO,
            t("vi_fuente_sin_nombre", idioma, n=f.miles(en["filas_leidas"])),
            "informe#entrada.filas_leidas")
    add("vx_desarrollo", "participantes", DECLARADO,
        "; ".join(t("vi_desarrollo", idioma, c=p["columna"],
                    v=t(_DEV[p["validacion_externa"]["casos_en_su_desarrollo"]], idioma),
                    ext=t("vi_desarrollo_declarado" if p["validacion_externa"]["es_validacion_externa"]
                          else "vi_desarrollo_no_externa", idioma)) for p in ps),
        "informe#puntuaciones[0].validacion_externa")
    falta("poblacion", "participantes", t("mo_no_recoge", idioma), "informe#entrada")
    falta("criterios_de_inclusion", "participantes", t("mo_no_recoge", idioma), "informe#entrada")
    add("vx_representatividad", "participantes", DECLARADO,
        t("si" if inf["la_muestra_representa_a_la_poblacion"] else "no", idioma),
        "informe#la_muestra_representa_a_la_poblacion")
    add("vx_tamano", "participantes", MEDIDO,
        t("vi_tamano", idioma, n=f.miles(en["filas_con_desenlace"]), e=f.miles(de["eventos"]),
          s=f.miles(en["filas_sin_desenlace"]), m=f.miles(en["filas_malformadas_fuera"])),
        "informe#entrada.filas_con_desenlace")
    add("huella_del_dataset", "participantes", MEDIDO, en["huella_csv"], "informe#entrada.huella_csv")
    # ---- predictores ----
    add("vx_puntuacion", "predictores", DECLARADO,
        "; ".join(t("vi_puntuacion", idioma, c=p["columna"], tipo=_nombre_tipo(p["declaracion"]["tipo"], idioma),
                    dir=t("decl_dir_mas_alto" if p["declaracion"]["direccion"] == "mas_alto_mas_riesgo"
                          else "decl_dir_mas_bajo", idioma),
                    orig=(t("decl_origen", idioma, o=p["declaracion"]["origen"]) if p["declaracion"].get("origen")
                          else t("decl_sin_origen", idioma))) for p in ps),
        "informe#puntuaciones[0].declaracion")
    falta("predictores", "predictores", t("mo_predictores", idioma), "informe#puntuaciones[0].declaracion")
    # ---- desenlace ----
    add("vx_desenlace", "desenlace", DECLARADO,
        t("vi_desenlace", idioma, c=de["columna"], e=de["evento"]), "informe#desenlace.evento")
    falta("definicion_del_desenlace", "desenlace", t("mo_no_recoge", idioma), "informe#desenlace")
    al = inf["alcance"]
    add("momento_del_desenlace", "desenlace", DECLARADO,
        t("vi_horizonte", idioma, h=al["horizonte_del_desenlace"], m=al["momento_de_la_puntuacion"]),
        "informe#alcance.horizonte_del_desenlace")
    add("prevalencia_observada", "desenlace", MEDIDO,
        "; ".join(t("vi_prev_obs", idioma, c=p["columna"], p=f.pct(p["prevalencia"])) for p in ps),
        "informe#puntuaciones[0].prevalencia")
    if inf["la_muestra_representa_a_la_poblacion"]:
        add("prevalencia_declarada", "desenlace", DECLARADO,
            t("vi_prev_decl", idioma, p=f.pct(ps[0]["prevalencia"])),
            "informe#la_muestra_representa_a_la_poblacion")
    else:
        falta("prevalencia_declarada", "desenlace", motivo_inv("prevalencia_declarada"),
              "informe#la_muestra_representa_a_la_poblacion")
    # ---- análisis ----
    add("metricas", "analisis", MEDIDO,
        "; ".join(t("vi_discr", idioma, c=p["columna"],
                    a=f.n(p["discriminacion"]["auroc"]["valor"]) if p["discriminacion"]["auroc"].get("valor") is not None else "—",
                    ia=_ic_corto(p["discriminacion"]["auroc"], f.n, idioma),
                    p=f.n(p["discriminacion"]["average_precision"]["valor"])
                    if p["discriminacion"]["average_precision"].get("valor") is not None else "—")
                  for p in ps),
        "informe#puntuaciones[0].discriminacion")
    aplican = [p for p in ps if p["calibracion"].get("aplica")]
    if aplican:
        add("calibracion", "analisis", MEDIDO,
            "; ".join(
                t("vi_calib", idioma, c=p["columna"],
                  e=f.n(p["calibracion"]["fiabilidad"]["ece"]) if p["calibracion"].get("fiabilidad", {}).get("ece") is not None else "—",
                  a=f.n(p["calibracion"]["intercepto"]["valor"], 2) if p["calibracion"]["intercepto"].get("valor") is not None else "—",
                  b=f.n(p["calibracion"]["pendiente"]["valor"], 2) if p["calibracion"]["pendiente"].get("valor") is not None else "—")
                if p["calibracion"].get("aplica") else t("vi_calib_na", idioma, c=p["columna"])
                for p in ps),
            f'informe#puntuaciones[{ps.index(aplican[0])}].calibracion')
    else:
        falta("calibracion", "analisis", _bi(ps[0]["calibracion"].get("motivo"), idioma),
              "informe#puntuaciones[0].calibracion", no_aplica=True)
    con_umbral = [p for p in ps if p["al_umbral"].get("declarado")]
    if con_umbral:
        add("umbrales", "analisis", MEDIDO,
            "; ".join(t("vi_umbral", idioma, c=p["columna"], u=_umbral_texto(p, f, idioma))
                      if p["al_umbral"].get("declarado") else t("vi_umbral_sin", idioma, c=p["columna"])
                      for p in ps),
            f'informe#puntuaciones[{ps.index(con_umbral[0])}].al_umbral')
    else:
        falta("umbrales", "analisis", motivo_inv("umbrales"), "informe#puntuaciones[0].al_umbral")
    dca = [p for p in ps if (p["curva_de_decision"] or {}).get("declarada")]
    if dca:
        def linea_dca(p: dict[str, Any]) -> str:
            c = p["curva_de_decision"]
            if not c.get("aplica"):
                return t("vi_dca_na", idioma, c=p["columna"])
            if not c.get("declarada"):
                return t("vi_dca_sin", idioma, c=p["columna"])
            return t("vi_dca", idioma, c=p["columna"], k=len(c["puntos"]),
                     s=sum(1 for q in c["puntos"] if q["supera_las_referencias"]))
        add("curva_de_decision", "analisis", MEDIDO, "; ".join(linea_dca(p) for p in ps),
            f'informe#puntuaciones[{ps.index(dca[0])}].curva_de_decision')
    elif any((p["curva_de_decision"] or {}).get("aplica") for p in ps):
        falta("curva_de_decision", "analisis", motivo_inv("curva_de_decision"),
              "informe#puntuaciones[0].curva_de_decision")
    else:
        falta("curva_de_decision", "analisis", _bi(ps[0]["curva_de_decision"].get("motivo"), idioma),
              "informe#puntuaciones[0].curva_de_decision", no_aplica=True)
    if ps[0].get("subgrupos"):
        add("subgrupos", "analisis", MEDIDO,
            "; ".join(t("vi_subgrupos", idioma, col=s["columna"], k=len(s["grupos"]),
                        i=sum(1 for g in s["grupos"] if not g.get("soporte_suficiente")))
                      for s in ps[0]["subgrupos"]),
            "informe#puntuaciones[0].subgrupos")
    else:
        falta("subgrupos", "analisis", motivo_inv("subgrupos"), "informe#puntuaciones[0].subgrupos")
    add("politica_de_faltantes", "analisis", MEDIDO,
        "; ".join(t("vi_faltantes", idioma, c=p["columna"], f=f.miles(p["filas"]["sin_puntuacion"]),
                    s=f.miles(en["filas_sin_desenlace"])) for p in ps),
        "informe#puntuaciones[0].filas")
    comp = inf.get("comparacion")
    if comp and comp.get("posible") and comp.get("diferencia") is not None:
        add("vx_comparacion", "analisis", MEDIDO,
            t("vi_comparacion", idioma, a=comp["columna_a"], b=comp["columna_b"],
              d=f.signed(comp["diferencia"]), n=f.miles(comp["filas_comunes"]),
              ic=_ic_corto({"ic": comp.get("ic"), "ic_no_disponible": comp.get("ic_no_disponible")},
                           f.signed, idioma)),
            "informe#comparacion.diferencia")
    elif comp:
        falta("vx_comparacion", "analisis", _bi(comp.get("motivo"), idioma) or t("mo_sin_comun", idioma),
              "informe#comparacion")
    else:
        falta("vx_comparacion", "analisis", t("mo_una_sola", idioma), "informe#puntuaciones")
    rm = inf["remuestreo"]
    diseno = (t("diseno_paciente", idioma, col=rm["columna_de_paciente"])
              if rm.get("columna_de_paciente") else t("diseno_iid", idioma))
    add("vx_incertidumbre", "analisis", MEDIDO,
        t("vi_incertidumbre", idioma, diseno=diseno, r=f.miles(rm["remuestras"]),
          nivel=f.pct(rm["nivel"], 0)), "informe#remuestreo.remuestras")
    falta("proceso_actual", "analisis", t("mo_proceso", idioma), "informe#puntuaciones[0].al_umbral")
    lne = (inf.get("lo_que_no_es") or [{}])[0]
    externa = next((x for x in inf.get("lo_que_no_es") or [] if x.get("id") == "no_es_validacion_externa"), None)
    add("alcance_de_validacion", "analisis", MEDIDO,
        _bi(lne, idioma) + (" " + _bi(externa, idioma) if externa else ""),
        "informe#lo_que_no_es[0]")
    add("vx_reproducibilidad", "analisis", MEDIDO,
        t("vi_versiones", idioma, n=inf["versiones"]["nucleo"],
          c=inf["versiones"]["catalogo_de_metricas"][:12], s=rm["semilla"]),
        "informe#versiones.nucleo")
    # ---- aplicabilidad ----
    falta("uso_previsto", "aplicabilidad", t("mo_no_recoge", idioma), "informe#entrada")
    for clave, textos in NO_DISPONIBLE:
        if clave is None:
            items.append(_Item("autor:" + textos["en"][0], "aplicabilidad", textos[idioma][0], FALTA,
                               t("mo_del_autor", idioma), "informe#entrada"))
    return items


def _tripod(resultado: Any, idioma: str) -> str:
    items = items_tripod(resultado, idioma)
    filas_por_dominio: list[str] = []
    for dominio in DOMINIOS:
        delante = [i for i in items if i.dominio == dominio]
        if not delante:
            continue
        filas_por_dominio.append(
            f'<tr class="dom"><th colspan="3">{_e(rotulo_de_dominio(dominio, idioma))}</th></tr>')
        for i in delante:
            etiqueta = t("tp_no_aplica" if i.no_aplica else _ETIQUETA_DE_ESTADO[i.estado], idioma)
            clase = "st-" + ("na" if i.no_aplica else i.estado)
            filas_por_dominio.append(
                f'<tr class="item"><td>{_e(i.rotulo)}</td>'
                f'<td><span class="st {clase}">{_e(etiqueta)}</span></td>'
                f'<td>{_e(i.dato)}</td></tr>')
    cab = "".join(f"<th>{_e(t(c, idioma))}</th>"
                  for c in ("tp_th_tema", "tp_th_estado", "tp_th_dato"))
    seccion = (f'<section class="blk tripod"><h3>{_e(t("s_tripod", idioma))}</h3>'
               f'<p class="small muted">{_e(t("tp_intro", idioma))}</p>'
               f'<div class="tw"><table class="tp"><thead><tr>{cab}</tr></thead>'
               f'<tbody>{"".join(filas_por_dominio)}</tbody></table></div>'
               f'</section>')
    # LA ÚLTIMA PUERTA (invariante 6 del 109): esta ficha no califica el riesgo de sesgo.
    texto = html.unescape(re.sub(r"<[^>]+>", " ", seccion))
    sin_veredicto_de_riesgo(texto, origen="informe_de_puntuacion.tripod")
    return seccion


# ===========================================================================================
# El documento
# ===========================================================================================

_CSS = """
:root{color-scheme:light;--fg:#17211f;--muted:#56635f;--line:#cfd8d4;--raised:#f4f7f6;
--accent:#0f6b5c;--accent-soft:#dcefe9;--s2:#7a4fc2;--warn:#8a5a00;--warn-soft:#fbefd3;--bg:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
.page{max-width:980px;margin:0 auto;padding:20px 16px 40px;display:flex;flex-direction:column;gap:20px}
h1,h2,h3{font-family:Georgia,"Times New Roman",serif;line-height:1.2;margin:0;font-weight:600}
h1{font-size:26px}h2{font-size:19px}h3{font-size:16px}
h4{margin:0;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
p{margin:0}.muted{color:var(--muted)}.small{font-size:12.5px}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.88em;word-break:break-all}
header.top{border-bottom:2px solid var(--accent);padding-bottom:12px;display:flex;flex-direction:column;gap:6px}
.k{font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--accent)}
.lang-chip{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
.pill{display:inline-block;align-self:flex-start;font-size:11.5px;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid var(--line);background:var(--raised);color:var(--fg)}
.pill.warn{background:var(--warn-soft);border-color:var(--warn);color:var(--warn)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
.tiles.one{grid-template-columns:minmax(220px,360px)}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:3px;min-width:0;break-inside:avoid}
.tile .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:21px;font-variant-numeric:tabular-nums}
.tile .ci{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--muted)}
.tile .q{font-size:12.5px;color:var(--muted)}
.tile.na .v{font-family:inherit;font-size:14px;color:var(--muted)}
.tile .v.txt{font-family:inherit;font-size:15px}
section.score{display:flex;flex-direction:column;gap:12px}
section.score>h2{border-left:4px solid var(--accent);padding-left:10px}
section.score.c2>h2{border-left-color:var(--s2)}
.blk{display:flex;flex-direction:column;gap:8px}
.note{font-size:13px;padding:8px 10px;border-radius:6px;background:var(--raised);border:1px solid var(--line)}
.note.warn{background:var(--warn-soft);border-color:var(--warn)}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 8px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:var(--raised)}
td.n{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
table.kv th{width:30%;background:var(--raised);font-weight:600;font-size:12px}
tr.dom th{background:var(--accent-soft);color:var(--accent);font-size:12px}
.st{display:inline-block;font-size:11px;font-weight:600;padding:0 7px;border-radius:999px;border:1px solid var(--line);white-space:nowrap}
.st-medido{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
.st-declarado{background:var(--raised)}
.st-falta,.st-na{background:#fff;color:var(--muted);border-style:dashed}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.chart{border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:6px;min-width:0;break-inside:avoid}
.chart svg{width:100%;height:auto;max-width:100%;display:block}
.ax{stroke:var(--line);stroke-width:1;fill:none}.gridl{stroke:var(--line);stroke-width:.6;stroke-dasharray:2 3;fill:none}
.diag{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4;fill:none}
.l1,.s1{stroke:var(--accent);stroke-width:2;fill:none}.l2,.s2{stroke:var(--s2);stroke-width:2;fill:none}
.f1{fill:var(--accent)}.f2{fill:var(--s2)}.all{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 3;fill:none}
.prom{fill:#fff;stroke:var(--fg);stroke-width:1.2}
.tl{fill:var(--muted);font-family:inherit;font-size:11px}
.tl.lbl{paint-order:stroke;stroke:#fff;stroke-width:3px;fill:var(--fg)}
.leg{display:flex;gap:4px 14px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.leg .sw{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:3px;background:var(--muted)}
.leg .sw.s1{background:var(--accent)}.leg .sw.s2{background:var(--s2)}
section.not{border-left:3px solid var(--warn);padding:6px 0 6px 14px;display:flex;flex-direction:column;gap:6px}
section.conclusiones{display:flex;flex-direction:column;gap:8px;font-size:14px}
section.conclusiones ul{margin:4px 0 0;padding-left:20px;display:flex;flex-direction:column;gap:6px}
.cs{border-left:4px solid var(--accent);padding-left:12px;display:flex;flex-direction:column;gap:4px}
.cs.c2{border-left-color:var(--s2)}
.adv,.alc{color:var(--muted);font-size:13px}
details.anexo{border:1px solid var(--line);border-radius:8px}
details.anexo>summary{cursor:pointer;padding:8px 12px;font-weight:600;background:var(--raised);border-radius:8px}
details.anexo[open]>summary{border-bottom:1px solid var(--line);border-radius:8px 8px 0 0}
.anexo-cuerpo{padding:12px;display:flex;flex-direction:column;gap:14px}
footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px}
@page{size:A4;margin:12mm}
@media print{
html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-size:10.5px}.page{max-width:none;padding:0;gap:12px}
h1{font-size:20px}h2{font-size:15px}h3{font-size:13px}
.tile .v{font-size:16px}table{font-size:9.5px}th,td{padding:3px 6px}
section.score,.blk,.not{break-inside:avoid}.tripod tr{break-inside:avoid}
section.conclusiones{font-size:11px;break-after:page}
details.anexo{break-inside:auto}details.anexo>summary{break-after:avoid}
.tw{overflow:visible}
}
@media print{details.anexo::details-content{content-visibility:visible;display:block;height:auto}}
"""


def _lista(items: Sequence[str]) -> str:
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items if x) + "</ul>"


def _et(clave: str, idioma: str) -> str:
    """El rótulo de una pregunta o de una línea, listo para ir delante de su texto: una pregunta
    ya acaba en «?»; lo demás lleva dos puntos («Al umbral de hoy: Con el umbral…»)."""
    r = t(clave, idioma)
    return r if r.endswith(("?", ":", ".")) else r + ":"


def _en_llano_de(p: dict[str, Any], f: _F, idioma: str) -> str:
    """Las cinco preguntas de UNA puntuación, en lenguaje llano y con las cifras del informe.
    Va en el anexo de la puntuación: la página de conclusiones lleva solo el veredicto."""
    hechos: list[str] = []
    a = p["discriminacion"]["auroc"]
    if a.get("valor") is None:
        hechos.append(f'<b>{_e(_et("q_ordena", idioma))}</b> '
                      f'{_e(t("c_ordena_na", idioma, motivo=_bi(a.get("no_definido"), idioma)))}')
    else:
        ic = a.get("ic")
        if ic is None:
            pill = t("ic_sin", idioma)
        elif ic["alto"] < 0.5:
            pill = t("pill_ic_bajo_azar", idioma)
        elif ic["bajo"] > 0.5:
            pill = t("pill_ic_sobre_azar", idioma)
        else:
            pill = t("pill_ic_con_azar", idioma)
        pill = pill[:1].upper() + pill[1:]
        hechos.append(f'<b>{_e(_et("q_ordena", idioma))}</b> ' + _e(t(
            "c_ordena", idioma, v=f.n(a["valor"]), ic=_ic(a, f.n, idioma),
            pp=f.pct(a["valor"], 0), pill=pill)))
        pr = p.get("cifra_prometida")
        if pr:
            hechos.append(f'<b>{_e(t("ref_etiqueta", idioma))}.</b> '
                          + _e(_referencia_linea(pr, a, f, idioma)) + " "
                          + f'<span class="adv">{_e(_bi(pr.get("advertencia"), idioma))}</span>')
    cal = p["calibracion"]
    if not cal.get("aplica"):
        hechos.append(f'<b>{_e(_et("q_riesgo", idioma))}</b> '
                      f'{_e(t("c_riesgo_na", idioma, motivo=_bi(cal.get("motivo"), idioma)))}')
    elif cal["en_grande"].get("valor") is None:
        hechos.append(f'<b>{_e(_et("q_riesgo", idioma))}</b> '
                      f'{_e(_bi(cal["en_grande"].get("no_definido"), idioma))}')
    else:
        g, ia, ib = cal["en_grande"], cal["intercepto"], cal["pendiente"]

        def num(m: dict[str, Any]) -> str:
            return f.n(m["valor"], 2) if m.get("valor") is not None else "—"

        def par(m: dict[str, Any]) -> str:
            return _ic(m, lambda x: f.n(x, 2), idioma) or "—"

        hechos.append(
            f'<b>{_e(_et("q_riesgo", idioma))}</b> ' + _e(t(
                "c_riesgo", idioma, v=f.signed(g["valor"] * 100, 1),
                ic=_ic(g, lambda x: f.signed(x * 100, 1), idioma), a=num(ia), ia=par(ia),
                b=num(ib), ib=par(ib))))
    u = p["al_umbral"]
    if not u.get("declarado"):
        hechos.append(f'<b>{_e(_et("q_umbral", idioma))}</b> '
                      f'{_e(t("c_umbral_sin", idioma, motivo=_bi(u.get("motivo"), idioma)))}')
    else:
        s_, e_ = u["sensibilidad"], u["especificidad"]
        if s_.get("valor") is None or e_.get("valor") is None:
            hechos.append(f'<b>{_e(_et("q_umbral", idioma))}</b> '
                          f'{_e(_bi((s_ if s_.get("valor") is None else e_).get("no_definido"), idioma))}')
        else:
            txt = t("c_umbral", idioma, u=_umbral_texto(p, f, idioma), k=f.miles(u["avisos"]),
                    s=f.pct(s_["valor"]), ics=_ic(s_, f.pct, idioma), s100=f.n(s_["valor"] * 100, 0),
                    e=f.pct(e_["valor"]), ice=_ic(e_, f.pct, idioma), e100=f.n(e_["valor"] * 100, 0))
            vpp, vpn = u["vpp"], u["vpn"]
            if vpp.get("valor") is not None and vpn.get("valor") is not None:
                txt += " " + t("c_umbral_vp", idioma, vpp=f.pct(vpp["valor"]),
                               icp=_ic(vpp, f.pct, idioma), p100=f.n(vpp["valor"] * 100, 0),
                               vpn=f.pct(vpn["valor"]), icn=_ic(vpn, f.pct, idioma),
                               n100=f.n(vpn["valor"] * 100, 0))
            else:
                txt += " " + t("c_umbral_sin_vp", idioma,
                               motivo=_bi(vpp.get("no_definido"), idioma) or "—")
            hechos.append(f'<b>{_e(_et("q_umbral", idioma))}</b> {_e(txt)}')
    medidos, insuf, total = _grupos_de(p)
    if not total:
        grupos = t("grupos_ninguno", idioma)
    else:
        partes = [t("grupos_medidos", idioma, k=len(medidos)) + "."]
        if len(medidos) >= 2:
            lo, hi = min(medidos, key=lambda x: x[1]), max(medidos, key=lambda x: x[1])
            partes.append(t("grupos_extremos", idioma, a=lo[0], va=f.n(lo[1]), b=hi[0],
                            vb=f.n(hi[1])))
        partes.append((t("grupos_insuf", idioma, k=insuf) if insuf else t("grupos_todos", idioma)) + ".")
        grupos = " ".join(partes)
    hechos.append(f'<b>{_e(_et("q_grupos", idioma))}</b> {_e(grupos)}')
    rot = _rotulos_unicos([_rotulo_de_alcance(cal if cal.get("aplica") else None, idioma),
                           _rotulo_de_alcance((u or {}).get("beneficio_neto"), idioma),
                           _rotulo_de_alcance(p.get("curva_de_decision"), idioma)])
    if rot:
        r = ((cal if cal.get("aplica") else None) or (u or {}).get("beneficio_neto")
             or p.get("curva_de_decision") or {}).get("rotulo_de_alcance") or {}
        hechos.append(f'<span class="alc">{_e(t("c_rotulo", idioma, rotulo=_bi(r.get("rotulo"), idioma), porque=_bi(r.get("porque"), idioma)))}</span>')
    return _lista(hechos)


def _frente_en_llano(inf: dict[str, Any], f: _F, idioma: str) -> str:
    """«Frente a la otra», con todas las palabras (anexo de la comparación)."""
    c = inf.get("comparacion")
    if not c:
        return ""
    if not c.get("posible") or c.get("diferencia") is None:
        return (f'<p><b>{_e(_et("q_frente", idioma))}</b> '
                f'{_e(_bi(c.get("motivo"), idioma) or "—")}</p>')
    a, b = c["columna_a"], c["columna_b"]
    est = {"a_mayor_que_b": "frente_a_mayor", "a_menor_que_b": "frente_b_mayor",
           "el_intervalo_incluye_el_cero": "frente_sin_dif"}.get(c.get("resultado"))
    txt = " ".join(x for x in (
        t(est, idioma, a=a, b=b) + "." if est else "",
        t("frente_dif_texto", idioma, d=f.signed(c["diferencia"]), ic=_ic(
            {"valor": c["diferencia"], "ic": c.get("ic"), "ic_no_disponible": c.get("ic_no_disponible")},
            f.signed, idioma)),
        t("frente_filas", idioma, n=f.miles(c["filas_comunes"]), e=f.miles(c["eventos_comunes"])),
        t("frente_directa", idioma)) if x)
    return f'<p><b>{_e(_et("q_frente", idioma))}</b> {_e(txt)}</p>'


def _conclusiones_de(p: dict[str, Any], f: _F, idioma: str) -> str:
    """Lo que la página de conclusiones dice de UNA puntuación: la frase-veredicto que compuso el
    núcleo, una línea de cifras clave y, si prometía una cifra, su referencia con la advertencia."""
    partes = []
    v = p.get("veredicto")
    if v:
        partes.append(f'<p class="veredicto"><b>{_e(_bi(v["texto"], idioma))}</b></p>')
    cifras: list[str] = []
    a = p["discriminacion"]["auroc"]
    if a.get("valor") is not None:
        cifras.append(t("c_k_auroc", idioma, v=f.n(a["valor"]), ic=_ic(a, f.n, idioma)))
    cal = p["calibracion"]
    g = cal.get("en_grande") if cal.get("aplica") else None
    if g and g.get("valor") is not None:
        cifras.append(t("c_k_riesgo", idioma, v=f.signed(g["valor"] * 100, 1),
                        ic=_ic(g, lambda x: f.signed(x * 100, 1), idioma)))
    u = p["al_umbral"]
    if u.get("declarado") and u["sensibilidad"].get("valor") is not None \
            and u["especificidad"].get("valor") is not None:
        txt = t("c_k_umbral", idioma, u=_umbral_texto(p, f, idioma),
                s=f.pct(u["sensibilidad"]["valor"]), e=f.pct(u["especificidad"]["valor"]))
        if u["vpp"].get("valor") is not None and u["vpn"].get("valor") is not None:
            txt += ", " + t("c_k_vp", idioma, p=f.pct(u["vpp"]["valor"]), n=f.pct(u["vpn"]["valor"]))
        cifras.append(txt)
    medidos, insuf, total = _grupos_de(p)
    if total:
        cifras.append(t("c_k_grupos", idioma, k=len(medidos), i=insuf))
    if cifras:
        partes.append(f'<p class="cifras"><b>{_e(t("c_cifras", idioma))}</b> '
                      f'{_e(" · ".join(cifras))}</p>')
    pr = p.get("cifra_prometida")
    if pr and pr.get("resultado") in ("dentro", "fuera") and a.get("valor") is not None:
        partes.append(f'<p class="small muted">{_e(t("c_ref_corta", idioma, r=f.n(pr["prometida"]), e=f.n(a["valor"]), pos=t("ref_dentro" if pr["resultado"] == "dentro" else "ref_fuera", idioma)))}</p>')
    elif pr:
        partes.append(f'<p class="small muted">{_e(_referencia_linea(pr, a, f, idioma))} '
                      f'{_e(_bi(pr.get("advertencia"), idioma))}</p>')
    return "".join(partes)


def _frente_compacto(inf: dict[str, Any], f: _F, idioma: str) -> str:
    c = inf.get("comparacion")
    if not c:
        return ""
    if not c.get("posible") or c.get("diferencia") is None:
        return (f'<p><b>{_e(_et("q_frente", idioma))}</b> '
                f'{_e(_bi(c.get("motivo"), idioma) or "—")}</p>')
    ic = _ic({"valor": c["diferencia"], "ic": c.get("ic"),
              "ic_no_disponible": c.get("ic_no_disponible")}, f.signed, idioma)
    return (f'<p><b>{_e(_et("q_frente", idioma))}</b> '
            + _e(t("c_frente_linea", idioma, a=c["columna_a"], b=c["columna_b"],
                   d=f.signed(c["diferencia"]), ic=ic, n=f.miles(c["filas_comunes"]),
                   e=f.miles(c["eventos_comunes"]), directa=t("frente_directa", idioma)))
            + '</p>')


def _limites_cortos(inf: dict[str, Any], idioma: str) -> list[str]:
    """Tres o cuatro viñetas CORTAS: cada una sale de un dato del informe (no de una lista
    escrita a mano), y su texto completo está en el anexo de límites."""
    ids = {x.get("id") for x in inf.get("lo_que_no_es") or []}
    ps = inf["puntuaciones"]
    out = [t("l_rep_si" if inf["la_muestra_representa_a_la_poblacion"] else "l_rep_no", idioma)]
    externa = next((x for x in inf.get("lo_que_no_es") or []
                    if x.get("id") == "no_es_validacion_externa"), None)
    if externa:
        out.append(_bi(externa, idioma))
    elif ps and all((p.get("validacion_externa") or {}).get("casos_en_su_desarrollo") == "no" for p in ps):
        out.append(t("l_dev_no", idioma))
    if "no_es_validacion_clinica" in ids:
        out.append(t("l_clinica", idioma))
    if "paradoja_del_tratamiento" in ids:
        out.append(t("l_paradoja", idioma))
    return out


def _conclusiones(resultado: Any, f: _F, idioma: str) -> str:
    """R6: la página que se lee primero y que cabe en UNA hoja A4: qué se evaluó, una
    frase-veredicto por puntuación, sus cifras clave y tres o cuatro límites. Las definiciones
    en llano, las advertencias completas y los intervalos van en los anexos."""
    inf = resultado.informe
    de = inf["desenlace"]
    total = inf["entrada"]["filas_con_desenlace"]
    al = inf.get("alcance") or {}
    partes = [f'<h2>{_e(t("s_conclusiones", idioma))}</h2>',
              f'<p>{_e(t("c_intro", idioma, col=de["columna"], ev=de["evento"], n=f.miles(total), e=f.miles(de["eventos"]), p=f.pct(de["eventos"] / total) if total else "—", hor=al.get("horizonte_del_desenlace", "—"), mom=al.get("momento_de_la_puntuacion", "—")))}</p>',
              _avisos(inf, idioma)]
    for k, p in enumerate(inf["puntuaciones"]):
        partes.append(f'<div class="cs c{k + 1}"><h3>{_e(p["columna"])}</h3>'
                      f'{_conclusiones_de(p, f, idioma)}</div>')
    partes.append(_frente_compacto(inf, f, idioma))
    limites = [_e(x) for x in _limites_cortos(inf, idioma)]
    firma = _e(t("c_lim_firma", idioma)) + (
        "" if resultado.nivel_de_garantia == "A1" else " " + _e(t("c_lim_sin_firma", idioma)))
    partes.append(f'<section class="not"><h3>{_e(t("c_lim", idioma))}</h3>{_lista(limites)}'
                  f'<p><b>{firma}</b></p></section>')
    partes.append(f'<p class="small muted">{_e(t("c_anexos", idioma))}</p>')
    return f'<section class="conclusiones">{"".join(x for x in partes if x)}</section>'


def _limites_completos(resultado: Any, idioma: str) -> str:
    """Los límites con todas las palabras (anexo): la nota de representatividad, la de validación
    externa de cada puntuación y todo lo que el informe dice que NO es."""
    inf = resultado.informe
    limites: list[str] = []
    rep_ = inf.get("representatividad")
    if rep_:
        limites.append(_e(_bi(rep_.get("nota"), idioma)))
        if rep_.get("porque"):
            limites.append(_e(_bi(rep_["porque"], idioma)))
    for p in inf["puntuaciones"]:
        if p.get("validacion_externa"):
            limites.append(_e(_bi(p["validacion_externa"]["nota"], idioma)))
    no_es = [_e(_bi(x, idioma)) for x in inf.get("lo_que_no_es") or []]
    return (f'<section class="not"><h3>{_e(t("c_lim", idioma))}</h3>{_lista(limites)}'
            f'<h4>{_e(t("s_no_es", idioma))}</h4>{_lista(no_es)}</section>')


def _anexo(clave: str, letra: str, cuerpo: str, idioma: str, **campos: Any) -> str:
    """Un anexo desplegable: `<details>` nativo, sin JavaScript. Cerrado en pantalla; al
    imprimir el CSS pide que se imprima abierto (`::details-content`)."""
    return (f'<details class="anexo"><summary>{_e(t(clave, idioma, l=letra, **campos))}'
            f'</summary><div class="anexo-cuerpo">{cuerpo}</div></details>')


def informe_html(resultado: Any, *, idioma: str) -> str:
    """El informe HTML (autocontenido) de un `ResultadoDeEvaluacion`: una página de
    conclusiones en lenguaje llano y, debajo, los detalles técnicos en anexos desplegables.

    Pura: no lee disco ni reloj. `idioma` es `"es"` o `"en"`; cualquier otro revienta (un
    informe en un idioma que no se pidió es peor que ningún informe)."""
    if idioma not in IDIOMAS:
        raise ValueError(f"idioma {idioma!r}: solo {', '.join(IDIOMAS)}")
    inf = resultado.informe
    f = _F(idioma)
    ps = inf["puntuaciones"]
    nombres = [p["columna"] for p in ps]
    h1 = (t("h1_una", idioma, a=nombres[0]) if len(nombres) == 1
          else t("h1_dos", idioma, a=nombres[0], b=nombres[1]))
    rm, de = inf["remuestreo"], inf["desenlace"]
    total = inf["entrada"]["filas_con_desenlace"]
    diseno = (t("diseno_paciente", idioma, col=rm["columna_de_paciente"])
              if rm.get("columna_de_paciente") else t("diseno_iid", idioma))
    resumen = t("resumen_muestra", idioma, n=f.miles(total), e=f.miles(de["eventos"]),
                p=f.pct(de["eventos"] / total) if total else "—", nivel=f.pct(rm["nivel"], 0),
                r=f.miles(rm["remuestras"]), diseno=diseno)
    nivel = resultado.nivel_de_garantia
    garantia = t("garantia_a1" if nivel == "A1" else "garantia_a0", idioma)
    cab = (f'<header class="top"><span class="k">{_e(t("etiqueta", idioma))}</span>'
           f'<h1>{_e(h1)}</h1><p class="muted">{_e(resumen)}</p>'
           f'<div class="lang-chip"><span class="pill">{_e(garantia)}</span>'
           f'<span class="small muted">{_e(t("solo_agregados", idioma))}'
           + (f' {_e(_bi(inf["donde_se_calcula"]["texto"], idioma))}'
              if inf.get("donde_se_calcula") else "")
           + '</span></div></header>')
    letras = iter("ABCDEFGH")
    anexos = []
    for k, p in enumerate(ps):
        cuerpo = (f'<section class="score c{k + 1}"><h2>{_e(p["columna"])}</h2>'
                  f'<p class="small muted">{_e(_declaracion(p, f, idioma))}</p>'
                  f'<div class="blk"><h4>{_e(t("a_en_llano", idioma))}</h4>'
                  f'{_en_llano_de(p, f, idioma)}</div>'
                  f'{_tarjetas(p, f, idioma)}{_tabla_umbral(p, f, idioma)}'
                  f'{_tabla_subgrupos(p, f, idioma)}</section>')
        anexos.append(_anexo("a_puntuacion", next(letras), cuerpo, idioma, c=p["columna"]))
    anexos.append(_anexo("a_graficas", next(letras), _graficas(inf, f, idioma), idioma))
    comp = _comparacion(inf, f, idioma)
    if comp:
        anexos.append(_anexo("a_comparacion", next(letras),
                             comp + _frente_en_llano(inf, f, idioma), idioma))
    anexos.append(_anexo("a_limites", next(letras), _limites_completos(resultado, idioma), idioma))
    anexos.append(_anexo("a_como", next(letras),
                         _entrada(inf, resultado, f, idioma) + _recibo(resultado, f, idioma), idioma))
    anexos.append(_anexo("a_tripod", next(letras), _tripod(resultado, idioma), idioma))
    secciones = [cab, _conclusiones(resultado, f, idioma), *anexos]
    return (f'<!doctype html>\n<html lang="{idioma}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_e(t("doc_titulo", idioma))}</title><style>{_CSS}</style></head>'
            f'<body><div class="page">{"".join(x for x in secciones if x)}</div></body></html>\n')
