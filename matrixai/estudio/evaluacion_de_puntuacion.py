# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""«¿Funciona aquí?» — 137-C1: evaluar una puntuación de riesgo YA REGISTRADA.

Un hospital, un banco o una aseguradora usan una puntuación hecha con otra población
(una escala publicada, el modelo de un proveedor) y casi nadie comprueba si funciona con
SUS casos. Esto toma un CSV con el desenlace observado y una o dos columnas de
puntuación, y dice —sin entrenar nada, sin que los datos salgan de la máquina— si ordena,
si acierta en la probabilidad (cuando lo es), qué pasa al umbral con el que se usa hoy,
con quién falla y si una mejora a la otra.

**ESTE MÓDULO COMPONE; NO ESCRIBE FÓRMULAS.** AUROC, AP, Brier, calibración, intervalos,
tabla al umbral, curva de decisión, segmentos y comparación emparejada ya viven en
`metricas`, `incertidumbre`, `calibracion`, `perfil_clinico`, `segmentos` y
`comparaciones`; el recibo, en `pipelines.receipt`. Dos sitios calculando el mismo AUROC
acaban divergiendo. Lo NUEVO de aquí es solo:

* **la entrada**: leer el CSV como lo escribe la gente (coma decimal, Windows-1252,
  `Sí`/`SI`/`sí`, `NA`, un índice de fila de R sin nombre…) y DECIR lo que se hizo;
* **las declaraciones** obligatorias por puntuación (tipo, dirección, umbral, origen,
  cifra prometida) y si la muestra representa a la población;
* **los avisos**: dirección dudosa (AUROC < 0,5), falta de puntuación concentrada en un
  desenlace, subgrupos sin evidencia, y lo que el informe NO es (reglas 17, 18 y 19);
* **el informe y su recibo** (HMAC con el mecanismo de `attest`, techo de garantía A1).

LO QUE SE DECIDIÓ AL COMPONER (para que no se «simplifique» sin saberlo):

* **Nada se deduce de los valores.** Un `0–1` no se toma por probabilidad ni un `0–100`
  por porcentaje: lo declara quien evalúa. Probabilidad fuera de rango = RECHAZO que dice
  cuántos y cuáles; no se recorta.
* **La dirección nunca se invierte en silencio.** «Más bajo = más riesgo» ordena con la
  puntuación negada (y el umbral negado), pero el número se publica tal y como sale, y si
  el AUROC queda por debajo de 0,5 se avisa de que la dirección puede estar al revés.
  Una probabilidad con «más bajo = más riesgo» no es la probabilidad del evento: se mide
  cómo ordena y su calibración queda «no aplica» con su motivo.
* **Calibración, Brier y curva de decisión SOLO con probabilidad.** Con una escala de
  puntos el motivo lo da el propio registro métrico (`brier_score` rechaza `scores`).
* **VPP y VPN solo si «la muestra representa a la población» es sí**, con la prevalencia
  de la muestra declarada como la local (la regla del 109-C2: nunca supuesta).
* **La representatividad afecta a más que VPP y VPN (R3).** Con «no», la calibración, el
  beneficio neto y la curva de decisión llevan el rótulo «describen la muestra aportada, no
  la población de uso» (`rotulo_de_alcance`); con «sí», el informe dice que lo declara quien
  ejecuta y MatrixAI no lo comprueba. Sin ajustes por muestreo.
* **Qué se predice y en qué plazo (R2)**: `horizonte_del_desenlace`, `momento_de_la_puntuacion`
  y, por puntuación, `casos_en_su_desarrollo` son OBLIGATORIAS y sin valor por omisión. Con
  `si` o `no_se_sabe` el informe dice que NO es una validación externa de esa puntuación.
* **Dónde se calcula (`donde_se_calcula`, obligatorio)**: `local` o `servidor_de_la_demo`. El
  informe lo escribe tal cual; «todo en local» sería falso en la demo.
* **La cifra prometida es una REFERENCIA PUBLICADA frente a una ESTIMACIÓN LOCAL (R4)**:
  se dan las dos cifras y dónde cae la una respecto al intervalo de la otra, SIEMPRE con la
  advertencia de que pueden no ser comparables. «Fuera» no es un veredicto.
* **Subgrupos por debajo de 30 filas o 10 eventos: «evidencia insuficiente», SIN número**
  (tampoco el de «el resto»: con dos grupos, el resto ES el otro grupo y lo revelaría).
  Una columna con más de `MAXIMO_DE_VALORES_POR_SUBGRUPO` valores se rechaza: parece un
  identificador, y un informe con un grupo por persona publicaría una cifra por persona.
  **Suprimir grupos pequeños ayuda, pero NO garantiza anonimización** (137, revisión R5): el
  informe lo dice así y no promete privacidad.
* **El informe no lleva filas**, solo agregados. Por la misma razón, un tramo de la curva
  de fiabilidad con menos de 30 filas no publica su cifra.
* **El intercepto y la pendiente de calibración comparten UN remuestreo**: la fórmula
  registrada ajusta `a` y `b` a la vez, así que su intervalo se saca de las mismas
  remuestras (dos pasadas harían el ajuste Newton dos veces por nada).

Y sigue siendo stdlib puro.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import platform
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from matrixai.estudio.calibracion import ajustar_recalibracion_logistica, curva_de_fiabilidad
from matrixai.estudio.comparaciones import comparar_candidatos
from matrixai.estudio.errores import ErrorDeEstudio
from matrixai.estudio.esquemas import ValorDeMetrica
from matrixai.estudio.evaluacion_de_puntuacion_textos import cifra, motivo
from matrixai.estudio.incertidumbre import Intervalo, _percentil, _recortar, intervalo, medir
from matrixai.estudio.metricas import (
    Indefinida,
    Muestra,
    calcular,
    digest_del_catalogo,
    direccion_de,
)
from matrixai.estudio.perfil_clinico import (
    BENEFICIO_NETO_NO_TRATAR,
    beneficio_neto,
    curva_de_decision,
    tabla_de_umbrales,
)
from matrixai.estudio.segmentos import (
    MINIMO_EVENTOS_POR_SEGMENTO,
    MINIMO_FILAS_POR_SEGMENTO,
    analizar_segmento,
)
from matrixai.estudio.textos import motivo as motivo_del_nucleo
from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.receipt import firmar_recibo, nivel_del_recibo

__all__ = [
    "AUSENTES", "CASOS_EN_SU_DESARROLLO", "DIRECCIONES", "DONDE_SE_CALCULA",
    "MAXIMO_DE_VALORES_POR_SUBGRUPO", "MINIMO_DE_REMUESTRAS",
    "TIPOS_DE_PUNTUACION", "VERSION_DEL_INFORME", "DeclaracionDePuntuacion",
    "EvaluacionRechazada", "ResultadoDeEvaluacion", "evaluar_puntuacion",
    "resumen_de_la_evaluacion",
]

VERSION_DEL_INFORME = "2.0.0"
ESQUEMA_DEL_INFORME = "matrixai.estudio.score_evaluation_report"

#: Lo que declara quien evalúa. Cerrado, como todo vocabulario de esta casa.
TIPOS_DE_PUNTUACION = ("probabilidad", "probabilidad_pct", "puntuacion")
DIRECCIONES = ("mas_alto_mas_riesgo", "mas_bajo_mas_riesgo")
#: R2: ¿estos casos participaron en el desarrollo de la puntuación? Sin valor por omisión.
CASOS_EN_SU_DESARROLLO = ("si", "no", "no_se_sabe")
#: Dónde corrió el cálculo. Sin valor por omisión: «todo en local» no es verdad en la demo,
#: donde lo calcula el servidor con el CSV que la persona envió.
DONDE_SE_CALCULA = ("local", "servidor_de_la_demo")
#: Un horizonte o un momento son una frase corta («durante el ingreso», «al ingreso»).
MAXIMO_DE_CARACTERES_DE_UN_TEXTO_DECLARADO = 200

#: Las marcas de «aquí no hay dato», comparadas sin mayúsculas. NO está `none`: puede ser
#: un desenlace legítimo («ninguno»), y confundirlo con un vacío descartaría filas.
AUSENTES = frozenset({"", "na", "n/a", "null", "nan", "-", "?"})

#: Más grupos que esto en una columna de subgrupo = parece un identificador.
MAXIMO_DE_VALORES_POR_SUBGRUPO = 30

#: Un intervalo de menos remuestras no describe nada.
MINIMO_DE_REMUESTRAS = 20

#: «Se concentra en un desenlace»: la tasa de falta difiere al menos 2 puntos entre
#: eventos y no eventos Y el test exacto de Fisher no lo atribuye al azar (p < 0,01).
DIFERENCIA_MINIMA_DE_FALTA = 0.02
P_MAXIMO_DE_FALTA_CONCENTRADA = 0.01

NIVEL = 0.95
ESTIMANDO = "fixed_model_on_population"
_CLASES = ("no_evento", "evento")
_POSITIVA = "evento"


class EvaluacionRechazada(ErrorDeEstudio):
    """Lo que no se puede evaluar tal como llega, con su motivo en los dos idiomas.

    Subclase de `ErrorDeEstudio` (quien capture los rechazos del paquete la captura), pero
    con el catálogo de textos de este corte, no el de `textos.py`."""

    def __init__(self, clave: str, **campos: object) -> None:
        textos = motivo(clave, **campos)
        ValueError.__init__(self, textos["en"])
        self.clave = clave
        self.campos = dict(campos)
        self.es = textos["es"]
        self.en = textos["en"]


# ---------------------------------------------------------------------------
# Las declaraciones
# ---------------------------------------------------------------------------

def _texto_obligatorio(valor: Any, campo: str, ejemplo: str) -> str:
    """R2: una declaración en texto, obligatoria: ni vacía ni de relleno ni kilométrica."""
    if (not isinstance(valor, str) or not valor.strip()
            or len(valor.strip()) > MAXIMO_DE_CARACTERES_DE_UN_TEXTO_DECLARADO):
        raise EvaluacionRechazada("texto_obligatorio_vacio", campo=campo, opciones=ejemplo,
                                  valor=repr(valor if not isinstance(valor, str) else valor[:60]))
    return " ".join(valor.split())


def _numero(valor: Any, campo: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
        raise EvaluacionRechazada("declaracion_no_numerica", campo=campo, valor=repr(valor))
    return float(valor)


@dataclass(frozen=True)
class DeclaracionDePuntuacion:
    """Lo que se declara de UNA puntuación. `tipo` y `direccion` no tienen valor por
    omisión: de ellos depende todo, y nada se deduce de los valores.

    * `tipo`: `probabilidad` (0–1), `probabilidad_pct` (0–100) o `puntuacion` (cualquier
      escala que ordena).
    * `direccion`: `mas_alto_mas_riesgo` o `mas_bajo_mas_riesgo`.
    * `umbral`: con el que se usa hoy, EN LA UNIDAD DE LA COLUMNA (23,5 en una columna de
      %). Sin umbral no hay tabla al umbral. Se avisa cuando la puntuación lo ALCANZA
      (`>=` con «más alto», `<=` con «más bajo»).
    * `origen`: texto libre (proveedor, artículo, versión).
    * `cifra_prometida`: el AUROC que promete su origen, si se conoce.
    * `casos_en_su_desarrollo` (R2, OBLIGATORIO, solo por nombre y sin valor por omisión):
      `si`, `no` o `no_se_sabe`. Con `si` o `no_se_sabe` el informe dice que esto NO es una
      validación externa de esa puntuación. Lo declara quien ejecuta; MatrixAI no lo comprueba.
    """

    columna: str
    tipo: str
    direccion: str
    umbral: float | None = None
    origen: str | None = None
    cifra_prometida: float | None = None
    casos_en_su_desarrollo: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.casos_en_su_desarrollo not in CASOS_EN_SU_DESARROLLO:
            raise EvaluacionRechazada("declaracion_fuera_del_vocabulario",
                                      campo="casos_en_su_desarrollo",
                                      opciones=list(CASOS_EN_SU_DESARROLLO),
                                      valor=repr(self.casos_en_su_desarrollo))
        if not isinstance(self.columna, str) or not self.columna.strip():
            raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="columna",
                                      opciones="un nombre de columna", valor=repr(self.columna))
        if self.tipo not in TIPOS_DE_PUNTUACION:
            raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="tipo",
                                      opciones=list(TIPOS_DE_PUNTUACION), valor=repr(self.tipo))
        if self.direccion not in DIRECCIONES:
            raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="direccion",
                                      opciones=list(DIRECCIONES), valor=repr(self.direccion))
        if self.umbral is not None:
            u = _numero(self.umbral, f"umbral de {self.columna}")
            tope = {"probabilidad": 1.0, "probabilidad_pct": 100.0}.get(self.tipo)
            if tope is not None and not (0.0 <= u <= tope):
                raise EvaluacionRechazada("declaracion_fuera_de_rango",
                                          campo=f"umbral de {self.columna}", minimo=0,
                                          maximo=int(tope), valor=repr(self.umbral))
        if self.cifra_prometida is not None:
            c = _numero(self.cifra_prometida, f"cifra prometida de {self.columna}")
            if not (0.0 <= c <= 1.0):
                raise EvaluacionRechazada("declaracion_fuera_de_rango",
                                          campo=f"cifra prometida (AUROC) de {self.columna}",
                                          minimo=0, maximo=1, valor=repr(self.cifra_prometida))
        if self.origen is not None and not isinstance(self.origen, str):
            raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="origen",
                                      opciones="un texto", valor=repr(self.origen))

    @property
    def es_probabilidad(self) -> bool:
        return self.tipo in ("probabilidad", "probabilidad_pct")

    @property
    def mas_alto(self) -> bool:
        return self.direccion == "mas_alto_mas_riesgo"

    @property
    def umbral_en_probabilidad(self) -> float | None:
        """El umbral en la escala con la que se mide: el % pasa a 0–1."""
        if self.umbral is None:
            return None
        return self.umbral / 100.0 if self.tipo == "probabilidad_pct" else float(self.umbral)

    def a_json(self) -> dict[str, Any]:
        return {"columna": self.columna, "tipo": self.tipo, "direccion": self.direccion,
                "umbral": self.umbral, "origen": self.origen,
                "cifra_prometida_auroc": self.cifra_prometida,
                "casos_en_su_desarrollo": self.casos_en_su_desarrollo}


@dataclass(frozen=True)
class ResultadoDeEvaluacion:
    """El informe (agregados, sin filas), el recibo y, si se dio clave, su sobre firmado.

    `nivel_de_garantia` es el que SOSTIENE el recibo, deducido por
    `pipelines.receipt.nivel_del_recibo`: A0 sin firmar, A1 firmado, y no pasa de ahí."""

    informe: dict[str, Any]
    recibo: dict[str, Any]
    sobre: dict[str, Any] | None
    nivel_de_garantia: str

    def a_json(self) -> dict[str, Any]:
        return {"informe": self.informe, "recibo": self.sobre if self.sobre else self.recibo,
                "nivel_de_garantia": self.nivel_de_garantia}


# ---------------------------------------------------------------------------
# Leer el CSV como lo escribe la gente
# ---------------------------------------------------------------------------

@dataclass
class _Tabla:
    cabecera: list[str]
    filas: list[list[str]]
    codificacion: str
    bom: bool
    separador: str
    malformadas: int
    indice_sin_nombre: bool
    huella: str


def _decodificar(contenido: bytes | str) -> tuple[str, str, bool, str]:
    """(texto, codificación, había BOM, huella). UTF-8 si es válido; si no, Windows-1252
    (Excel «CSV delimitado» en Windows), y se DICE. Otra cosa no se adivina."""
    if isinstance(contenido, str):
        bom = contenido.startswith("﻿")
        return (contenido[1:] if bom else contenido, "utf-8", bom,
                hashlib.sha256(contenido.encode("utf-8")).hexdigest())
    crudo = bytes(contenido)
    huella = hashlib.sha256(crudo).hexdigest()
    bom = crudo.startswith(b"\xef\xbb\xbf")
    sin_bom = crudo[3:] if bom else crudo
    try:
        return sin_bom.decode("utf-8"), "utf-8", bom, huella
    except UnicodeDecodeError:
        pass
    try:
        return sin_bom.decode("cp1252"), "windows-1252", bom, huella
    except UnicodeDecodeError as exc:
        raise EvaluacionRechazada("csv_ni_utf8_ni_windows1252", valor=exc.start) from None


def _separador(texto: str) -> str:
    """Coma, punto y coma, tabulador o barra: el que parte las primeras líneas SIEMPRE en el
    mismo número de campos (>1) y en más. Con coma decimal el `,` no es consistente y gana el
    `;`, que es lo que pasa con Excel europeo."""
    lineas = [ln for ln in texto.splitlines()[:200] if ln.strip()]
    if not lineas:
        return ","
    mejor, mejor_puntos = ",", (False, 0.0, 0)
    for d in (",", ";", "\t", "|"):
        try:
            cuentas = [len(next(csv.reader([ln], delimiter=d))) for ln in lineas]
        except csv.Error:
            continue
        cabecera = cuentas[0]
        coherencia = sum(1 for c in cuentas if c == cabecera) / len(cuentas)
        puntos = (coherencia == 1.0 and cabecera > 1, coherencia if cabecera > 1 else 0.0,
                  cabecera)
        if puntos > mejor_puntos:
            mejor, mejor_puntos = d, puntos
    return mejor


def _leer_csv(contenido: bytes | str) -> _Tabla:
    """Cabecera + filas de celdas recortadas.

    UNA FORMA REAL, medida con `write.csv` de R (support2 de hbiostat): la cabecera trae N
    nombres y CADA fila de datos N+1 campos, porque la primera columna es el índice de fila,
    sin nombre. Si TODAS las filas tienen un campo más que la cabecera, el primero se toma
    por ese índice (y el informe lo dice). Si solo ALGUNAS no cuadran, se excluyen y se
    cuentan: eso es una fila rota, no un índice."""
    texto, codificacion, bom, huella = _decodificar(contenido)
    if not texto.strip():
        raise EvaluacionRechazada("csv_vacio")
    sep = _separador(texto)
    try:
        registros = [[c.strip() for c in r]
                     for r in csv.reader(io.StringIO(texto, newline=""), delimiter=sep)
                     if r and any(c.strip() for c in r)]
    except csv.Error as exc:
        raise EvaluacionRechazada("csv_ilegible", valor=str(exc)) from None
    if not registros:
        raise EvaluacionRechazada("csv_vacio")
    cabecera, datos = registros[0], registros[1:]
    if len(set(cabecera)) != len(cabecera) or any(not c for c in cabecera):
        repetidas = sorted({c for c in cabecera if cabecera.count(c) > 1})
        raise EvaluacionRechazada("csv_cabecera_invalida",
                                  valor=repr(repetidas) if repetidas else "nombre vacío")
    n = len(cabecera)
    indice = bool(datos) and all(len(f) == n + 1 for f in datos)
    if indice:
        datos = [f[1:] for f in datos]
    buenas = [f for f in datos if len(f) == n]
    if not buenas:
        raise EvaluacionRechazada("csv_sin_filas", valor=n)
    return _Tabla(cabecera=cabecera, filas=buenas, codificacion=codificacion, bom=bom,
                  separador=sep, malformadas=len(datos) - len(buenas),
                  indice_sin_nombre=indice, huella=huella)


_NUM_ENTERO = re.compile(r"[+-]?\d+")
_NUM_PUNTO = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+(?:\.\d+)?[eE][+-]?\d+)")
_NUM_COMA = re.compile(r"[+-]?(?:\d+,\d+|,\d+)")


def _clave_de_texto(texto: str) -> str | None:
    """La clave con la que se UNEN variantes: sin mayúsculas, sin tildes, sin espacios de
    más; y un número es el mismo número escrito como `1`, `1.0` o `1,0`. `None` = ausente."""
    t = texto.strip()
    if t.casefold() in AUSENTES:
        return None
    if _NUM_ENTERO.fullmatch(t) or _NUM_PUNTO.fullmatch(t) or _NUM_COMA.fullmatch(t):
        return format(float(t.replace(",", ".")), "g")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.casefold().split())


def _etiqueta(variantes: Counter) -> str:
    return sorted(variantes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _unir(celdas: Sequence[str]) -> tuple[list[str | None], dict[str, Counter]]:
    """Claves por celda y, por clave, las variantes que se unieron (con sus filas)."""
    claves: list[str | None] = []
    variantes: dict[str, Counter] = {}
    for c in celdas:
        k = _clave_de_texto(c)
        claves.append(k)
        if k is not None:
            variantes.setdefault(k, Counter())[c.strip()] += 1
    return claves, variantes


def _variantes_unidas(variantes: dict[str, Counter]) -> list[dict[str, Any]]:
    return [{"valor": _etiqueta(v),
             "variantes": [{"texto": t, "filas": n}
                           for t, n in sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))]}
            for _, v in sorted(variantes.items()) if len(v) > 1]


def _leer_numerica(nombre: str, celdas: Sequence[str]
                   ) -> tuple[list[float | None], str, list[str]]:
    """(valores, estilo decimal, marcas de ausencia vistas). El estilo se DETECTA POR COLUMNA
    y se declara: `coma`, `punto` o `entero`. Mezclar coma y punto o cualquier otra cosa no
    numérica se rechaza diciendo cuántos y cuáles: no se adivina."""
    valores: list[float | None] = []
    coma = punto = 0
    malos: list[str] = []
    n_malos = 0
    ausentes: set[str] = set()
    for c in celdas:
        t = c.strip()
        if t.casefold() in AUSENTES:
            valores.append(None)
            ausentes.add(t if t else "(vacío)")
        elif _NUM_ENTERO.fullmatch(t):
            valores.append(float(t))
        elif _NUM_PUNTO.fullmatch(t):
            valores.append(float(t))
            punto += 1
        elif _NUM_COMA.fullmatch(t):
            valores.append(float(t.replace(",", ".")))
            coma += 1
        else:
            valores.append(None)
            n_malos += 1
            if len(malos) < 5:
                malos.append(t)
    if n_malos:
        raise EvaluacionRechazada("puntuacion_no_numerica", campo=nombre, valor=n_malos,
                                  opciones=repr(malos))
    if coma and punto:
        raise EvaluacionRechazada("puntuacion_decimal_mezclado", campo=nombre, valor=coma,
                                  opciones=punto)
    return valores, ("coma" if coma else "punto" if punto else "entero"), sorted(ausentes)


# ---------------------------------------------------------------------------
# Pequeñas piezas: JSON seguro, Fisher, métricas
# ---------------------------------------------------------------------------

def _json_seguro(valor: Any) -> Any:
    """El informe tiene que poder canonicalizarse y firmarse: nada de tuplas, NaN ni infinitos."""
    if isinstance(valor, dict):
        return {str(k): _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, float):
        if not math.isfinite(valor):
            raise ValueError("un número no finito no puede viajar en el informe")
        return valor
    return valor


#: Cuántos puntos de la curva ROC viajan en el informe: los extremos y, como mucho, 38 vértices
#: reales más. Los puntos no se inventan ni se interpolan: son VÉRTICES de la curva exacta.
PUNTOS_DE_LA_CURVA_ROC = 40


def _curva_roc(muestra: Muestra) -> list[dict[str, float]] | None:
    """La curva ROC de la muestra, adelgazada a `PUNTOS_DE_LA_CURVA_ROC` vértices REALES.

    Se recorren los valores distintos de la puntuación de mayor a menor (los empates se
    cierran juntos: un vértice por valor, no por fila), con (fpr, tpr) acumulados. Los
    extremos (0,0) y (1,1) siempre están. Si hay más vértices que el tope, se eligen los más
    cercanos a repartos iguales de la LONGITUD de la curva: un subconjunto de la curva exacta,
    nunca un punto interpolado. Solo agregados: dos números por punto, ninguna fila.
    `None` si falta algún desenlace (no hay curva), que es lo que dice también el AUROC."""
    puntuacion = muestra.puntuacion_del_positivo
    if puntuacion is None:
        return None
    ys = [y == _POSITIVA for y in muestra.y_true]
    positivos = sum(ys)
    negativos = len(ys) - positivos
    if not positivos or not negativos:
        return None
    cuenta: dict[float, list[int]] = {}
    for sc, y in zip(puntuacion, ys):
        c = cuenta.setdefault(sc, [0, 0])
        c[0 if y else 1] += 1
    vertices = [(0.0, 0.0)]
    tp = fp = 0
    for sc in sorted(cuenta, reverse=True):
        tp += cuenta[sc][0]
        fp += cuenta[sc][1]
        vertices.append((fp / negativos, tp / positivos))
    if len(vertices) > PUNTOS_DE_LA_CURVA_ROC:
        acumulada = [0.0]
        for (x0, y0), (x1, y1) in zip(vertices, vertices[1:]):
            acumulada.append(acumulada[-1] + abs(x1 - x0) + abs(y1 - y0))
        total = acumulada[-1]
        elegidos = {0, len(vertices) - 1}
        for j in range(1, PUNTOS_DE_LA_CURVA_ROC - 1):
            objetivo = total * j / (PUNTOS_DE_LA_CURVA_ROC - 1)
            elegidos.add(min(range(len(vertices)), key=lambda i: abs(acumulada[i] - objetivo)))
        vertices = [vertices[i] for i in sorted(elegidos)]
    return [{"fpr": x, "tpr": y} for x, y in vertices]


def _p_de_fisher(a: int, b: int, c: int, d: int) -> float:
    """Test exacto de Fisher de dos colas sobre [[a, b], [c, d]], con `lgamma` para que
    aguante decenas de miles de filas sin factoriales enormes."""
    n1, n2 = a + b, c + d
    k, total = a + c, a + b + c + d
    lg = math.lgamma

    def logp(x: int) -> float:
        return (lg(n1 + 1) - lg(x + 1) - lg(n1 - x + 1) + lg(n2 + 1) - lg(k - x + 1)
                - lg(n2 - k + x + 1) - (lg(total + 1) - lg(k + 1) - lg(total - k + 1)))

    observado = logp(a)
    p = sum(math.exp(lp) for x in range(max(0, k - n2), min(k, n1) + 1)
            if (lp := logp(x)) <= observado + 1e-7)
    return min(1.0, p)


def _metrica(vm: ValorDeMetrica) -> dict[str, Any]:
    """Un `ValorDeMetrica` del núcleo con su intervalo, en la forma del informe: valor o
    motivo, intervalo o motivo, nunca un cero inventado."""
    salida: dict[str, Any] = {"valor": vm.value, "no_definido": vm.undefined_reason,
                              "ic": None, "ic_no_disponible": None}
    u = vm.uncertainty
    if u and vm.value is not None:
        if u.get("ci_low") is not None:
            salida["ic"] = {"bajo": u["ci_low"], "alto": u["ci_high"], "nivel": u["level"],
                            "metodo": u["method"], "unidad_de_remuestreo": u["resampling_unit"],
                            "remuestras": u["n_resamples"],
                            "remuestras_validas": u["n_resamples_used"]}
        else:
            salida["ic_no_disponible"] = u.get("undefined_reason")
    return salida


def _simple(vm: ValorDeMetrica) -> dict[str, Any]:
    return {"valor": vm.value, "no_definido": vm.undefined_reason}


def _rotulo_de_alcance(ctx: "_Contexto") -> dict[str, Any] | None:
    """R3: con «la muestra NO representa a la población de uso», lo que depende de la
    prevalencia (calibración, beneficio neto, curva de decisión) se rotula como descripción de
    LA MUESTRA, con su porqué. Con «sí» no hay rótulo: la nota de representatividad dice que
    lo declara quien ejecuta."""
    if ctx.representa:
        return None
    return {"rotulo": motivo("rotulo_muestra_aportada"),
            "porque": motivo("porque_muestra_aportada")}


def _aviso(clave: str, **campos: Any) -> dict[str, Any]:
    textos = motivo(clave, **campos)
    datos = {k: (dict(v) if isinstance(v, dict) else v) for k, v in campos.items()}
    return {"id": clave, "es": textos["es"], "en": textos["en"], "datos": datos}


# ---------------------------------------------------------------------------
# El contexto de una evaluación y las muestras
# ---------------------------------------------------------------------------

@dataclass
class _Contexto:
    y: list[int | None]
    unidades: list[str] | None
    diseno: str
    remuestras: int
    semilla: int
    representa: bool

    @property
    def comun(self) -> dict[str, Any]:
        return {"diseno": self.diseno, "estimando": ESTIMANDO, "semilla": self.semilla,
                "remuestras": self.remuestras}


def _muestra(decl: DeclaracionDePuntuacion, crudos: Sequence[float], ys: Sequence[int],
             unidades: Sequence[str] | None) -> Muestra:
    """La muestra binaria de UNA puntuación, ya orientada.

    Probabilidad con «más alto = más riesgo» → `probabilidades` (calibración disponible).
    Todo lo demás → `puntuaciones`, con el signo cambiado si «más bajo = más riesgo»: ordena
    igual de bien, y una probabilidad al revés NO es la del evento, así que no se hace pasar
    por una."""
    y_true = [_POSITIVA if y else _CLASES[0] for y in ys]
    comunes = dict(classes=_CLASES, positive_label=_POSITIVA, unidades=unidades)
    if decl.es_probabilidad and decl.mas_alto:
        p = [v / 100.0 if decl.tipo == "probabilidad_pct" else v for v in crudos]
        return Muestra.binaria(y_true, probabilidades=p, **comunes)
    return Muestra.binaria(y_true, puntuaciones=[v if decl.mas_alto else -v for v in crudos],
                           **comunes)


def _umbral_de_trabajo(decl: DeclaracionDePuntuacion) -> float | None:
    u = decl.umbral_en_probabilidad
    if u is None:
        return None
    return u if decl.mas_alto else -u


# ---------------------------------------------------------------------------
# Las secciones del informe
# ---------------------------------------------------------------------------

def _calibracion(decl: DeclaracionDePuntuacion, muestra: Muestra, ctx: _Contexto
                 ) -> dict[str, Any]:
    if not decl.es_probabilidad:
        try:
            calcular("brier_score", muestra)
            razon = motivo_del_nucleo("no_son_probabilidades", campo="calibration")
        except ErrorDeEstudio as rechazo:  # el registro métrico lo dice, no esta función
            razon = rechazo.bilingue
        return {"aplica": False, "motivo": razon}
    if not decl.mas_alto:
        return {"aplica": False,
                "motivo": motivo("probabilidad_con_direccion_inversa", campo=decl.columna)}

    salida: dict[str, Any] = {"aplica": True, "motivo": None,
                              "metodo_de_recalibracion": "logistico_completo",
                              "rotulo_de_alcance": _rotulo_de_alcance(ctx)}
    salida["en_grande"] = _metrica(medir("calibration_in_the_large", muestra, **ctx.comun))
    salida["brier"] = _simple(calcular("brier_score", muestra))

    ajuste = ajustar_recalibracion_logistica(muestra)
    if ajuste.a is None:
        razon = ajuste.undefined_reason
        salida["intercepto"] = {"valor": None, "no_definido": razon, "ic": None,
                                "ic_no_disponible": None}
        salida["pendiente"] = dict(salida["intercepto"])
    else:
        # UN remuestreo para los dos: la fórmula ajusta `a` y `b` juntos. La pendiente se
        # recoge de las MISMAS remuestras válidas que el intercepto.
        pendientes: list[float] = []

        def formula_a(m: Muestra, _umbral: float | None) -> float | Indefinida:
            r = ajustar_recalibracion_logistica(m)
            if r.a is None:
                return Indefinida(r.undefined_reason)  # type: ignore[arg-type]
            pendientes.append(r.b)  # type: ignore[arg-type]
            return r.a

        base = dict(muestra=muestra, semilla=ctx.semilla, remuestras=ctx.remuestras,
                    estimando=ESTIMANDO, diseno=ctx.diseno, nivel=NIVEL)
        punto_a = ValorDeMetrica(metric_id="calibration_intercept", formula_version="1.0.0",
                                 value=ajuste.a, n_observations=muestra.n,
                                 n_units=muestra.n_unidades)
        ic_a = intervalo("calibration_intercept", valor=punto_a, formula=formula_a, **base)
        salida["intercepto"] = _metrica(ValorDeMetrica(
            metric_id="calibration_intercept", formula_version="1.0.0", value=ajuste.a,
            n_observations=muestra.n, n_units=muestra.n_unidades, uncertainty=ic_a.a_json()))
        ic_b: Intervalo
        comun_b = dict(metric_id="calibration_slope", design=ic_a.design, estimand=ic_a.estimand,
                       resampling_unit=ic_a.resampling_unit, seed=ic_a.seed,
                       n_resamples=ic_a.n_resamples, level=ic_a.level, method=ic_a.method,
                       n_resamples_used=ic_a.n_resamples_used,
                       n_resamples_degenerate=ic_a.n_resamples_degenerate)
        if ic_a.disponible and len(pendientes) == ic_a.n_resamples_used:
            ordenadas = sorted(pendientes)
            cola = (1.0 - NIVEL) / 2.0
            ic_b = Intervalo(**comun_b, ci_low=_percentil(ordenadas, cola),
                             ci_high=_percentil(ordenadas, 1.0 - cola))
        else:
            ic_b = Intervalo(**comun_b, undefined_reason=ic_a.undefined_reason
                             or motivo_del_nucleo("remuestras_degeneradas", valor=0,
                                                  opciones=ic_a.n_resamples))
        salida["pendiente"] = _metrica(ValorDeMetrica(
            metric_id="calibration_slope", formula_version="1.0.0", value=ajuste.b,
            n_observations=muestra.n, n_units=muestra.n_unidades, uncertainty=ic_b.a_json()))

    curva = curva_de_fiabilidad(muestra)
    bins = []
    for b in curva.bins:
        if b.n == 0:
            bins.append({"desde": b.limite_inferior, "hasta": b.limite_superior, "n": 0,
                         "eventos": None, "probabilidad_media": None,
                         "tasa_observada": None, "motivo": None})
        elif b.n < MINIMO_FILAS_POR_SEGMENTO:
            bins.append({"desde": b.limite_inferior, "hasta": b.limite_superior, "n": b.n,
                         "eventos": None, "probabilidad_media": None, "tasa_observada": None,
                         "motivo": motivo("fiabilidad_pocas_filas", valor=b.n,
                                          opciones=MINIMO_FILAS_POR_SEGMENTO)})
        else:
            bins.append({"desde": b.limite_inferior, "hasta": b.limite_superior, "n": b.n,
                         "eventos": b.eventos, "probabilidad_media": b.probabilidad_media_predicha,
                         "tasa_observada": b.tasa_observada, "motivo": None})
    salida["fiabilidad"] = {"metodo_de_bins": curva.metodo_de_bins,
                            "n_bins": curva.n_bins_solicitados, "ece": curva.ece,
                            "no_definido": curva.undefined_reason, "bins": bins}
    return salida


def _al_umbral(decl: DeclaracionDePuntuacion, muestra: Muestra, ctx: _Contexto,
               prevalencia: float | None) -> dict[str, Any]:
    if decl.umbral is None:
        return {"declarado": False, "motivo": motivo("sin_umbral_de_uso", campo=decl.columna)}
    trabajo = _umbral_de_trabajo(decl)
    tabla = tabla_de_umbrales(
        muestra, umbrales=[trabajo], prevalencia=prevalencia if ctx.representa else None,
        declarados_por="declared by whoever runs the evaluation: the sample represents the "
                       "population of use", **ctx.comun)
    fila = tabla.filas[0]
    salida: dict[str, Any] = {
        "declarado": True, "umbral": decl.umbral, "umbral_en_la_escala_medida": trabajo,
        "regla_de_aviso": "mayor_o_igual" if decl.mas_alto else "menor_o_igual",
        "avisos": fila.tp + fila.fp,
        "matriz": {"tp": fila.tp, "fp": fila.fp, "fn": fila.fn, "tn": fila.tn},
        "sensibilidad": _metrica(fila.sensibilidad),
        "especificidad": _metrica(fila.especificidad),
        "vpp": _metrica(fila.ppv), "vpn": _metrica(fila.npv),
        "prevalencia_de_la_muestra": tabla.prevalencia_observada,
        "prevalencia_declarada": tabla.prevalencia_declarada,
        "nota": (motivo("ic_con_la_prevalencia_fija", valor=cifra(prevalencia or 0.0,
                                                                  porcentaje=True))
                 if ctx.representa and prevalencia is not None else None),
    }
    if decl.es_probabilidad and decl.mas_alto:
        u = decl.umbral_en_probabilidad
        if u is not None and u < 1.0 and muestra.n >= 1:
            positivos = sum(1 for y in muestra.y_true if y == _POSITIVA)
            salida["beneficio_neto"] = {
                "modelo": beneficio_neto(tp=fila.tp, fp=fila.fp, n=muestra.n,
                                         umbral_de_probabilidad=u),
                "tratar_a_todos": beneficio_neto(tp=positivos, fp=muestra.n - positivos,
                                                 n=muestra.n, umbral_de_probabilidad=u),
                "tratar_a_nadie": BENEFICIO_NETO_NO_TRATAR,
                "rotulo_de_alcance": _rotulo_de_alcance(ctx)}
    return salida


def _curva_de_decision(decl: DeclaracionDePuntuacion, muestra: Muestra,
                       umbrales: Sequence[float] | None, ctx: _Contexto) -> dict[str, Any]:
    if not decl.es_probabilidad or not decl.mas_alto:
        razon = (motivo("probabilidad_con_direccion_inversa", campo=decl.columna)
                 if decl.es_probabilidad else
                 motivo_del_nucleo("dca_sin_probabilidad_calibrada", campo="curva_de_decision"))
        return {"aplica": False, "motivo": razon}
    if not umbrales:
        return {"aplica": True, "declarada": False,
                "motivo": motivo("umbrales_de_decision_no_declarados")}
    curva = curva_de_decision(muestra, umbrales_de_probabilidad=list(umbrales))
    return {"aplica": True, "declarada": True, "motivo": None,
            "rotulo_de_alcance": _rotulo_de_alcance(ctx),
            "prevalencia": curva.prevalencia_observada,
            "puntos": [{"umbral": p.umbral_de_probabilidad, "tp": p.tp, "fp": p.fp,
                        "beneficio_neto": p.beneficio_neto,
                        "tratar_a_todos": p.beneficio_neto_tratar_a_todos,
                        "tratar_a_nadie": p.beneficio_neto_no_tratar,
                        "supera_las_referencias": p.supera_las_referencias}
                       for p in curva.puntos],
            "umbrales_sin_ventaja": list(curva.umbrales_sin_ventaja)}


def _subgrupos(decl: DeclaracionDePuntuacion, muestra: Muestra, columnas: Sequence[str],
               celdas_por_columna: dict[str, list[str]], filas_usadas: list[int],
               ctx: _Contexto) -> list[dict[str, Any]]:
    salida = []
    for col in columnas:
        celdas = [celdas_por_columna[col][i] for i in filas_usadas]
        claves, variantes = _unir(celdas)
        etiquetas = {k: _etiqueta(v) for k, v in variantes.items()}
        grupos: dict[str | None, list[int]] = {}
        for pos, k in enumerate(claves):
            grupos.setdefault(k, []).append(pos)
        entradas = []
        for k in sorted(grupos, key=lambda x: (x is None, etiquetas.get(x, ""))):
            posiciones = grupos[k]
            etiqueta = etiquetas.get(k) if k is not None else None
            segmento = f"{col}={etiqueta if etiqueta is not None else '(sin dato)'}"
            analisis = analizar_segmento("auroc", muestra, posiciones, segmento_id=segmento,
                                         predefinido=True)
            entrada: dict[str, Any] = {"valor": etiqueta, "sin_dato": k is None,
                                       "n": analisis.n,
                                       "soporte_suficiente": analisis.soporte_suficiente}
            if not analisis.soporte_suficiente:
                # «Evidencia insuficiente», SIN número: ni eventos, ni AUROC, ni el del resto.
                entrada["motivo"] = analisis.motivo_de_cautela
            else:
                entrada["eventos"] = analisis.eventos
                # R5: el AUROC del grupo lleva su intervalo (bootstrap del propio grupo, con
                # el mismo diseño, semilla y remuestras que el global). Medido: ~0,6 s por
                # grupo de ~1.250 filas con 500 remuestras; el coste crece con filas × columnas
                # × puntuaciones, no con el número de grupos.
                entrada["auroc"] = _metrica(medir("auroc", _recortar(muestra, posiciones),
                                                  **ctx.comun))
                if decl.es_probabilidad and decl.mas_alto:
                    cal = analizar_segmento("calibration_in_the_large", muestra, posiciones,
                                            segmento_id=segmento, predefinido=True)
                    entrada["calibracion_en_grande"] = _simple(cal.metrica_segmento)
            entradas.append(entrada)
        salida.append({"columna": col, "variantes_unidas": _variantes_unidas(variantes),
                       "grupos": entradas})
    return salida


# ---------------------------------------------------------------------------
# La función pública
# ---------------------------------------------------------------------------

def evaluar_puntuacion(
        csv_contenido: bytes | str, *, columna_desenlace: str, valor_del_evento: str,
        puntuaciones: Sequence[DeclaracionDePuntuacion], muestra_representativa: bool,
        horizonte_del_desenlace: str, momento_de_la_puntuacion: str, donde_se_calcula: str,
        columnas_de_subgrupo: Sequence[str] = (), columna_de_paciente: str | None = None,
        umbrales_de_decision: Sequence[float] | None = None, remuestras: int = 1000,
        semilla: int = 137, nombre_del_fichero: str | None = None, actor: str | None = None,
        clave: bytes | None = None, id_de_clave: str = "evaluate-score",
        creado_en: str | None = None) -> ResultadoDeEvaluacion:
    """Evalúa una o dos puntuaciones ya registradas contra un desenlace binario observado.

    * `csv_contenido`: los bytes del fichero (UTF-8, o Windows-1252 si no lo es: se dice) o
      su texto. Esta función no toca el disco ni el reloj.
    * `valor_del_evento`: qué valor del desenlace es el evento. Lo declara quien evalúa.
    * `muestra_representativa`: «¿esta muestra representa a la población donde se usa?».
      Sin valor por omisión: solo con `True` se publican VPP y VPN.
    * `horizonte_del_desenlace` (R2, obligatorio): en qué plazo se observó el desenlace
      («durante el ingreso», «a 30 días»). `momento_de_la_puntuacion` (R2, obligatorio):
      cuándo se calculó la puntuación respecto al caso («al ingreso», «día 3»). Los dos
      van al informe y al recibo; sin ellos no se sabe QUÉ se predice ni CUÁNDO.
    * `donde_se_calcula`: `local` o `servidor_de_la_demo`, obligatorio. Lo dice el LLAMANTE
      (esta función no lo puede saber) y el informe lo escribe tal cual: «todo en local» sería
      falso en la demo, donde el servidor calcula con el CSV que la persona envió.
    * `columna_de_paciente`: si un paciente tiene varias filas, los intervalos se calculan
      remuestreando PACIENTES (`groups`), no filas.
    * `umbrales_de_decision`: probabilidades entre 0 y 1 (también para una columna en %)
      de la curva de decisión; sin ellos no se inventa un barrido.
    * `remuestras`: del bootstrap; queda escrito en el informe (la demo pasa 500).
    * `clave` + `creado_en`: firman el recibo (HMAC, mecanismo de `attest`); sin clave el
      recibo sale sin firmar (A0). Con clave no pasa de A1.

    Devuelve el informe (sin filas), el recibo y su nivel. Lo que no se puede evaluar tal
    como llega levanta `EvaluacionRechazada`, con su motivo en los dos idiomas."""
    # -- las declaraciones ---------------------------------------------------
    declaraciones = list(puntuaciones)
    if not 1 <= len(declaraciones) <= 2:
        raise EvaluacionRechazada("numero_de_puntuaciones", valor=len(declaraciones))
    if not all(isinstance(d, DeclaracionDePuntuacion) for d in declaraciones):
        raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="puntuaciones",
                                  opciones="DeclaracionDePuntuacion", valor="otra cosa")
    horizonte_del_desenlace = _texto_obligatorio(
        horizonte_del_desenlace, "horizonte_del_desenlace", "durante el ingreso")
    momento_de_la_puntuacion = _texto_obligatorio(
        momento_de_la_puntuacion, "momento_de_la_puntuacion", "al ingreso")
    if donde_se_calcula not in DONDE_SE_CALCULA:
        raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="donde_se_calcula",
                                  opciones=list(DONDE_SE_CALCULA), valor=repr(donde_se_calcula))
    if not isinstance(muestra_representativa, bool):
        raise EvaluacionRechazada("declaracion_no_booleana", campo="muestra_representativa",
                                  valor=repr(muestra_representativa))
    if type(remuestras) is not int or remuestras < MINIMO_DE_REMUESTRAS:
        raise EvaluacionRechazada("declaracion_fuera_de_rango", campo="remuestras",
                                  minimo=MINIMO_DE_REMUESTRAS, maximo="sin tope",
                                  valor=repr(remuestras))
    if type(semilla) is not int:
        raise EvaluacionRechazada("declaracion_fuera_del_vocabulario", campo="semilla",
                                  opciones="un entero", valor=repr(semilla))
    if umbrales_de_decision is not None:
        lista = list(umbrales_de_decision)
        ok = all(not isinstance(u, bool) and isinstance(u, (int, float)) and 0 <= u < 1
                 for u in lista) and len(set(lista)) == len(lista)
        if not ok:
            raise EvaluacionRechazada("umbrales_de_decision_invalidos", valor=repr(lista))
        umbrales_de_decision = [float(u) for u in lista]
    if clave is not None and not creado_en:
        raise EvaluacionRechazada("recibo_sin_fecha")

    # -- el CSV ---------------------------------------------------------------
    tabla = _leer_csv(csv_contenido)
    columnas = {nombre: j for j, nombre in enumerate(tabla.cabecera)}

    def celdas(nombre: str, papel: str) -> list[str]:
        if nombre not in columnas:
            raise EvaluacionRechazada("columna_inexistente", campo=papel, valor=repr(nombre),
                                      opciones=tabla.cabecera)
        j = columnas[nombre]
        return [f[j] for f in tabla.filas]

    vistas: dict[str, str] = {}
    papeles = [("desenlace", columna_desenlace)]
    papeles += [(f"puntuación {k + 1}", d.columna) for k, d in enumerate(declaraciones)]
    if columna_de_paciente:
        papeles.append(("identificador de paciente", columna_de_paciente))
    for papel, nombre in papeles:
        if nombre in vistas:
            raise EvaluacionRechazada("columna_repetida_en_el_encargo", valor=repr(nombre),
                                      campo=f"{vistas[nombre]} y {papel}")
        vistas[nombre] = papel
    sub_columnas = list(dict.fromkeys(columnas_de_subgrupo))
    celdas_sub = {c: celdas(c, "subgrupo") for c in sub_columnas}
    for c, valores_c in celdas_sub.items():
        distintos = len(_unir(valores_c)[1])
        if distintos > MAXIMO_DE_VALORES_POR_SUBGRUPO:
            raise EvaluacionRechazada("subgrupo_con_demasiados_valores", campo=c,
                                      valor=distintos, opciones=MAXIMO_DE_VALORES_POR_SUBGRUPO)

    # -- el desenlace ----------------------------------------------------------
    crudo_y = celdas(columna_desenlace, "desenlace")
    claves_y, variantes_y = _unir(crudo_y)
    if len(variantes_y) > 2:
        mas_vistos = [_etiqueta(v) for _, v in sorted(
            variantes_y.items(), key=lambda kv: -sum(kv[1].values()))][:10]
        raise EvaluacionRechazada("desenlace_con_mas_de_dos_valores", campo=columna_desenlace,
                                  valor=len(variantes_y), opciones=repr(mas_vistos))
    if not variantes_y:
        raise EvaluacionRechazada("desenlace_sin_filas", campo=columna_desenlace)
    clave_evento = _clave_de_texto(str(valor_del_evento))
    if clave_evento is None or clave_evento not in variantes_y:
        raise EvaluacionRechazada("desenlace_sin_el_evento", campo=columna_desenlace,
                                  valor=repr(valor_del_evento),
                                  opciones=repr([_etiqueta(v) for v in variantes_y.values()]))
    y: list[int | None] = [None if k is None else int(k == clave_evento) for k in claves_y]
    con_desenlace = [i for i, v in enumerate(y) if v is not None]
    eventos_total = sum(1 for i in con_desenlace if y[i])
    no_eventos_total = len(con_desenlace) - eventos_total

    # -- las puntuaciones -------------------------------------------------------
    leidas = []
    for k, d in enumerate(declaraciones):
        crudas = celdas(d.columna, f"puntuación {k + 1}")
        valores, estilo, ausentes = _leer_numerica(d.columna, crudas)
        tope = {"probabilidad": 1.0, "probabilidad_pct": 100.0}.get(d.tipo)
        if tope is not None:
            fuera = [crudas[i] for i, v in enumerate(valores) if v is not None
                     and not (0.0 <= v <= tope)]
            if fuera:
                raise EvaluacionRechazada("probabilidad_fuera_de_rango", campo=d.columna,
                                          minimo=0, maximo=int(tope), valor=len(fuera),
                                          opciones=repr(fuera[:5]))
        leidas.append((crudas, valores, estilo, ausentes))

    # -- el diseño de remuestreo -------------------------------------------------
    unidades_todas: list[str] | None = None
    if columna_de_paciente:
        ids = celdas(columna_de_paciente, "identificador de paciente")
        unidades_todas = [v if v.casefold() not in AUSENTES else f"__sin_identificador_{i}"
                          for i, v in enumerate(ids)]
    ctx = _Contexto(y=y, unidades=unidades_todas,
                    diseno="groups" if unidades_todas else "iid", remuestras=remuestras,
                    semilla=semilla, representa=muestra_representativa)

    avisos: list[dict[str, Any]] = []
    if tabla.malformadas:
        avisos.append(_aviso("filas_malformadas_fuera", valor=tabla.malformadas))

    # -- una por una ---------------------------------------------------------------
    informes: list[dict[str, Any]] = []
    muestras: list[Muestra] = []
    usadas: list[list[int]] = []
    for k, (d, (crudas, valores, estilo, ausentes)) in enumerate(zip(declaraciones, leidas)):
        idx = [i for i in con_desenlace if valores[i] is not None]
        if not idx:
            raise EvaluacionRechazada("ninguna_fila_utilizable", campo=d.columna)
        ys = [y[i] for i in idx]
        muestra = _muestra(d, [valores[i] for i in idx], ys,  # type: ignore[arg-type]
                           [unidades_todas[i] for i in idx] if unidades_todas else None)
        muestras.append(muestra)
        usadas.append(idx)
        eventos = sum(ys)  # type: ignore[arg-type]
        prevalencia = eventos / len(idx)

        # La falta de puntuación, por desenlace (regla 15).
        faltan_ev = sum(1 for i in con_desenlace if y[i] and valores[i] is None)
        faltan_no = sum(1 for i in con_desenlace if not y[i] and valores[i] is None)
        tasa_ev = faltan_ev / eventos_total if eventos_total else None
        tasa_no = faltan_no / no_eventos_total if no_eventos_total else None
        p_falta = None
        concentrada = False
        if (faltan_ev + faltan_no) and eventos_total and no_eventos_total:
            p_falta = _p_de_fisher(faltan_ev, eventos_total - faltan_ev, faltan_no,
                                   no_eventos_total - faltan_no)
            concentrada = (abs(tasa_ev - tasa_no) >= DIFERENCIA_MINIMA_DE_FALTA  # type: ignore[operator]
                           and p_falta < P_MAXIMO_DE_FALTA_CONCENTRADA)
        avisos_k: list[dict[str, Any]] = []
        if concentrada:
            avisos_k.append(_aviso("falta_concentrada_en_un_desenlace", campo=d.columna,
                                   valor=cifra(tasa_ev, porcentaje=True),  # type: ignore[arg-type]
                                   opciones=cifra(tasa_no, porcentaje=True),  # type: ignore[arg-type]
                                   minimo=cifra(p_falta, 4)))  # type: ignore[arg-type]

        kw = ctx.comun
        auroc = medir("auroc", muestra, **kw)
        if auroc.value is not None and auroc.value < 0.5:
            avisos_k.append(_aviso("direccion_posiblemente_mal_declarada", campo=d.columna,
                                   valor=cifra(auroc.value)))
        promesa = None
        if d.cifra_prometida is not None:
            m = _metrica(auroc)
            if m["ic"] is not None:
                dentro = m["ic"]["bajo"] <= d.cifra_prometida <= m["ic"]["alto"]
                # R4: referencia publicada FRENTE A estimación local. `resultado` dice solo
                # dónde cae la referencia respecto al intervalo local; no es un veredicto.
                promesa = {"metrica": "auroc", "prometida": d.cifra_prometida,
                           "referencia_publicada": d.cifra_prometida,
                           "estimacion_local": auroc.value,
                           "intervalo": [m["ic"]["bajo"], m["ic"]["alto"]],
                           "resultado": "dentro" if dentro else "fuera", "motivo": None,
                           "advertencia": motivo("referencia_publicada_advertencia")}
            else:
                promesa = {"metrica": "auroc", "prometida": d.cifra_prometida,
                           "referencia_publicada": d.cifra_prometida,
                           "estimacion_local": auroc.value,
                           "intervalo": None, "resultado": None,
                           "motivo": motivo("cifra_prometida_sin_intervalo", campo=d.columna),
                           "advertencia": motivo("referencia_publicada_advertencia")}

        subgrupos = _subgrupos(d, muestra, sub_columnas, celdas_sub, idx, ctx) if sub_columnas else []
        informes.append({
            "indice": k + 1, "columna": d.columna, "declaracion": d.a_json(),
            "lectura": {"decimal": estilo, "marcas_de_ausencia": ausentes},
            "filas": {"con_desenlace": len(con_desenlace), "usadas": len(idx),
                      "sin_puntuacion": len(con_desenlace) - len(idx),
                      "tasa_de_falta_en_eventos": tasa_ev,
                      "tasa_de_falta_en_no_eventos": tasa_no,
                      "p_exacto_de_fisher": p_falta, "falta_concentrada": concentrada,
                      "unidades": muestra.n_unidades},
            "eventos": eventos, "prevalencia": prevalencia,
            "discriminacion": {
                "auroc": _metrica(auroc),
                "average_precision": _metrica(medir("average_precision", muestra, **kw)),
                "average_precision_del_azar": prevalencia,
                "curva_roc": _curva_roc(muestra)},
            "cifra_prometida": promesa,
            "validacion_externa": {
                "casos_en_su_desarrollo": d.casos_en_su_desarrollo,
                "es_validacion_externa": d.casos_en_su_desarrollo == "no",
                "comprobada_por_matrixai": False,
                "nota": motivo({"si": "validacion_externa_si",
                                "no_se_sabe": "validacion_externa_no_se_sabe",
                                "no": "validacion_externa_no"}[d.casos_en_su_desarrollo],
                               campo=d.columna)},
            "calibracion": _calibracion(d, muestra, ctx),
            "al_umbral": _al_umbral(d, muestra, ctx, prevalencia),
            "curva_de_decision": _curva_de_decision(d, muestra, umbrales_de_decision, ctx),
            "subgrupos": subgrupos,
            "avisos": avisos_k,
        })

    # -- dos puntuaciones: SOLO sobre las filas que tienen las dos ----------------------
    comparacion = None
    if len(declaraciones) == 2:
        comunes = [i for i in con_desenlace
                   if leidas[0][1][i] is not None and leidas[1][1][i] is not None]
        if not comunes:
            comparacion = {"posible": False, "motivo": motivo("comparacion_sin_filas_en_comun")}
        else:
            ys_c = [y[i] for i in comunes]
            unidades_c = [unidades_todas[i] for i in comunes] if unidades_todas else None
            m_a = _muestra(declaraciones[0], [leidas[0][1][i] for i in comunes], ys_c, unidades_c)  # type: ignore[misc]
            m_b = _muestra(declaraciones[1], [leidas[1][1][i] for i in comunes], ys_c, unidades_c)  # type: ignore[misc]
            cmp = comparar_candidatos("auroc", m_a, m_b, **ctx.comun)
            ic = cmp.intervalo
            neutro = {"mejora": "a_mayor_que_b", "inferioridad": "a_menor_que_b",
                      "inconcluso": "el_intervalo_incluye_el_cero"}.get(cmp.veredicto,
                                                                          cmp.veredicto)
            comparacion = {
                "posible": True, "metrica": "auroc", "sentido": "a_menos_b",
                "columna_a": declaraciones[0].columna, "columna_b": declaraciones[1].columna,
                "filas_comunes": len(comunes), "eventos_comunes": sum(ys_c),  # type: ignore[arg-type]
                "auroc_a": _simple(calcular("auroc", m_a)),
                "auroc_b": _simple(calcular("auroc", m_b)),
                "diferencia": cmp.diferencia_puntual,
                "ic": ({"bajo": ic.ci_low, "alto": ic.ci_high, "nivel": ic.level,
                        "metodo": ic.method, "remuestras": ic.n_resamples,
                        "remuestras_validas": ic.n_resamples_used}
                       if ic is not None and ic.disponible else None),
                "ic_no_disponible": (ic.undefined_reason
                                     if ic is not None and not ic.disponible else None),
                "resultado": neutro, "veredicto_del_nucleo": cmp.veredicto,
                "motivo": cmp.undefined_reason}

    if sub_columnas:
        n_grupos = sum(len(s["grupos"]) for r in informes for s in r["subgrupos"])
        avisos.append(_aviso("multiplicidad_en_subgrupos", valor=n_grupos))

    # -- lo que NO es (reglas 17, 18 y 19) ---------------------------------------------
    no_es = [{"id": c, **motivo(c)} for c in ("no_es_validacion_clinica",)]
    no_externas = [d.columna for d in declaraciones if d.casos_en_su_desarrollo != "no"]
    if no_externas:
        no_es.append({"id": "no_es_validacion_externa",
                      **motivo("no_es_validacion_externa", campo=", ".join(no_externas))})
    if any(d.umbral is not None for d in declaraciones):
        no_es.append({"id": "paradoja_del_tratamiento", **motivo("paradoja_del_tratamiento")})
    no_es.append({"id": "no_es_anonimizacion", **motivo("no_es_anonimizacion")})
    no_es.append({"id": "depende_del_registro", **motivo("depende_del_registro")})

    import matrixai  # noqa: PLC0415

    etiqueta_evento = _etiqueta(variantes_y[clave_evento])
    otras = [v for k_, v in variantes_y.items() if k_ != clave_evento]
    informe: dict[str, Any] = {
        "esquema": ESQUEMA_DEL_INFORME, "version_del_informe": VERSION_DEL_INFORME,
        "entrada": {
            "huella_csv": f"sha256:{tabla.huella}", "nombre_del_fichero": nombre_del_fichero,
            "codificacion": tabla.codificacion, "bom": tabla.bom, "separador": tabla.separador,
            "filas_leidas": len(tabla.filas) + tabla.malformadas,
            "filas_malformadas_fuera": tabla.malformadas,
            "filas_con_desenlace": len(con_desenlace),
            "filas_sin_desenlace": len(tabla.filas) - len(con_desenlace),
            "indice_sin_nombre": tabla.indice_sin_nombre,
            "indice_sin_nombre_texto": (
                {"es": "Cada fila traía un campo más que la cabecera: el primero se tomó por "
                       "un índice de fila sin nombre (así lo escribe R con write.csv) y no se usa.",
                 "en": "Every row carried one field more than the header: the first was taken "
                       "as an unnamed row index (as R's write.csv writes it) and is not used."}
                if tabla.indice_sin_nombre else None),
            "columnas": list(tabla.cabecera)},
        "alcance": {"horizonte_del_desenlace": horizonte_del_desenlace,
                    "momento_de_la_puntuacion": momento_de_la_puntuacion},
        "donde_se_calcula": {
            "valor": donde_se_calcula,
            "texto": motivo("calculado_en_local" if donde_se_calcula == "local"
                            else "calculado_en_el_servidor_de_la_demo")},
        "desenlace": {
            "columna": columna_desenlace, "evento": etiqueta_evento,
            "valor_declarado": valor_del_evento, "eventos": eventos_total,
            "no_eventos": no_eventos_total,
            "valor_de_no_evento": _etiqueta(otras[0]) if otras else None,
            "variantes_unidas": _variantes_unidas(variantes_y)},
        "remuestreo": {"diseno": ctx.diseno, "columna_de_paciente": columna_de_paciente,
                       "estimando": ESTIMANDO, "semilla": semilla, "remuestras": remuestras,
                       "nivel": NIVEL},
        "la_muestra_representa_a_la_poblacion": muestra_representativa,
        "representatividad": {
            "declarada": muestra_representativa, "comprobada_por_matrixai": False,
            "nota": motivo("representatividad_no_comprobada_si" if muestra_representativa
                           else "representatividad_no_comprobada_no"),
            **({"rotulo": _rotulo_de_alcance(ctx)["rotulo"],
                "porque": _rotulo_de_alcance(ctx)["porque"]}
               if not muestra_representativa else {"rotulo": None, "porque": None})},
        "puntuaciones": informes, "comparacion": comparacion,
        "avisos": avisos, "lo_que_no_es": no_es,
        "versiones": {"nucleo": matrixai.__version__,
                      "catalogo_de_metricas": digest_del_catalogo()},
    }
    informe = _json_seguro(informe)
    recibo = _recibo(informe, declaraciones, leidas, tabla, columna_desenlace, valor_del_evento,
                     muestra_representativa, sub_columnas, columna_de_paciente,
                     umbrales_de_decision, nombre_del_fichero, actor, creado_en)
    sobre = firmar_recibo(recibo, clave=clave, key_id=id_de_clave) if clave is not None else None
    return ResultadoDeEvaluacion(informe=informe, recibo=recibo, sobre=sobre,
                                 nivel_de_garantia=nivel_del_recibo(sobre or recibo))


# ---------------------------------------------------------------------------
# El recibo
# ---------------------------------------------------------------------------

def _recibo(informe: dict[str, Any], declaraciones: Sequence[DeclaracionDePuntuacion],
            leidas: Sequence[tuple], tabla: _Tabla, columna_desenlace: str, valor_del_evento: str,
            representa: bool, subgrupos: Sequence[str], paciente: str | None,
            umbrales_de_decision: Sequence[float] | None, nombre: str | None,
            actor: str | None, creado_en: str | None) -> dict[str, Any]:
    """El recibo, con el esquema de `evaluation_attestation` de `pipelines.receipt` — el mismo
    mecanismo que `attest`, sin inventar otro. Aquí no hay un `.onnx`: el «modelo» es la
    columna de puntuación TAL COMO se registró, con la huella de sus celdas y el origen que
    se declaró, y dicho en el propio recibo (`provenance`). El nivel no se declara: lo
    deduce `nivel_del_recibo` de lo que lleva, y sin evidencia reproducible no pasa de A1."""
    huella = tabla.huella
    modelos, pasos, metricas = [], [], []
    for d, (crudas, _valores, estilo, _aus), res in zip(declaraciones, leidas,
                                                        informe["puntuaciones"]):
        digest = "sha256:" + hashlib.sha256("\n".join(crudas).encode("utf-8")).hexdigest()
        modelo_id = f"registered-score:{d.columna}"
        modelos.append({
            "model_id": modelo_id, "version": "external", "digest": digest,
            "kind": "registered_score",
            "provenance": "external: a score recorded by the requester's own system; "
                          "MatrixAI neither trained nor ran it",
            "origin_declared": d.origen, "type_declared": d.tipo,
            "direction_declared": d.direccion, "decimal_style_read": estilo})
        pasos.append({"id": f"evaluate_registered_score_{res['indice']}", "model": modelo_id,
                      "entry_hash": digest})
        cal = res["calibracion"]
        listado = [("auroc", res["discriminacion"]["auroc"]),
                   ("average_precision", res["discriminacion"]["average_precision"])]
        if cal.get("aplica"):
            listado += [("calibration_in_the_large", cal["en_grande"]),
                        ("calibration_intercept", cal["intercepto"]),
                        ("calibration_slope", cal["pendiente"]), ("brier_score", cal["brier"])]
        if res["al_umbral"].get("declarado"):
            for nombre_m, clave_m in (("sensitivity", "sensibilidad"),
                                      ("specificity", "especificidad"), ("ppv", "vpp"),
                                      ("npv", "vpn")):
                listado.append((nombre_m, res["al_umbral"][clave_m]))
        for nombre_m, m in listado:
            ic = m.get("ic")
            metricas.append({
                "name": nombre_m, "score": d.columna, "value": m["valor"], "split": "provided",
                "direction": direccion_de(nombre_m) if nombre_m in _NOMBRES_DEL_REGISTRO else None,
                "ci_low": ic["bajo"] if ic else None, "ci_high": ic["alto"] if ic else None,
                "undefined_reason": m["no_definido"], "dataset_sha256": huella})
    declaraciones_json = {
        "outcome_column": columna_desenlace, "event_value": valor_del_evento,
        "scores": [d.a_json() for d in declaraciones],
        "outcome_horizon": informe["alcance"]["horizonte_del_desenlace"],
        "score_timing": informe["alcance"]["momento_de_la_puntuacion"],
        "sample_represents_population": representa, "subgroup_columns": list(subgrupos),
        "patient_column": paciente, "decision_thresholds": (
            list(umbrales_de_decision) if umbrales_de_decision else None)}
    id_ = "scoreeval-" + hashlib.sha256(
        (huella + jcs_bytes(declaraciones_json).decode("utf-8")
         + str(informe["remuestreo"]["semilla"]) + str(informe["remuestreo"]["remuestras"])
         ).encode("utf-8")).hexdigest()[:16]
    return _json_seguro({
        "schema_version": "1.0", "receipt_id": id_, "event_type": "evaluation_attestation",
        # La hora la pone quien firma: aquí no se lee el reloj.
        "created_at": creado_en,
        "subject": {"purpose": motivo("recibo_proposito"),
                    "actor": actor or motivo("recibo_actor_no_consta")},
        "models": modelos,
        "dataset": {
            "name": nombre or "csv", "digest": f"sha256:{huella}",
            "rows_total": informe["entrada"]["filas_leidas"],
            "rows_with_outcome": informe["entrada"]["filas_con_desenlace"],
            "rows_measured": [r["filas"]["usadas"] for r in informe["puntuaciones"]],
            "target_column": columna_desenlace,
            "input_columns": [d.columna for d in declaraciones],
            "encoding_read": tabla.codificacion, "separator_read": tabla.separador,
            "unnamed_row_index": tabla.indice_sin_nombre},
        "declarations": declaraciones_json,
        "parameters": {"seed": informe["remuestreo"]["semilla"],
                       "resamples": informe["remuestreo"]["remuestras"],
                       "design": informe["remuestreo"]["diseno"],
                       "estimand": ESTIMANDO, "level": NIVEL},
        "metrics": metricas,
        "steps": pasos,
        "output": {"outcome": "completed",
                   "output_digests": {"report": hashlib.sha256(jcs_bytes(informe)).hexdigest()}},
        "environment": {"matrixai_core": informe["versiones"]["nucleo"],
                        "metric_catalog_digest": informe["versiones"]["catalogo_de_metricas"],
                        "report_version": VERSION_DEL_INFORME,
                        "computed_where": informe["donde_se_calcula"]["valor"],
                        "python": platform.python_version(), "platform": platform.platform()},
        # Techo declarado Y deducido: `nivel_del_recibo` mira solo `evidence` (sin ninguna
        # clave reproducible) y la firma, así que A1 es lo más que este recibo sostiene.
        "assurance_ceiling": "A1",
        "evidence": {"attests": motivo("recibo_atestigua"),
                     "does_not_attest": motivo("recibo_no_atestigua")},
    })


_NOMBRES_DEL_REGISTRO = frozenset({"auroc", "average_precision", "brier_score",
                                   "sensitivity", "specificity", "ppv", "npv"})


# ---------------------------------------------------------------------------
# Un resumen para la terminal
# ---------------------------------------------------------------------------

def resumen_de_la_evaluacion(informe: dict[str, Any], locale: str = "es") -> str:
    """Unas líneas para la terminal. El informe completo es el JSON; esto solo lo resume."""
    i = "es" if str(locale or "es").lower() == "es" else "en"

    def n(x: float | None, d: int = 3) -> str:
        t = "—" if x is None else f"{x:.{d}f}"
        return t.replace(".", ",") if i == "es" else t

    def con_ic(m: dict[str, Any]) -> str:
        if m["valor"] is None:
            return (m["no_definido"] or {}).get(i, "—")
        ic = m["ic"]
        return n(m["valor"]) + (f"  (IC 95 %: {n(ic['bajo'])} – {n(ic['alto'])})" if ic else "")

    lineas = []
    e = informe["entrada"]
    lineas.append(("Filas leídas: {} · con desenlace: {} · codificación {} · separador {!r}"
                   if i == "es" else
                   "Rows read: {} · with outcome: {} · encoding {} · separator {!r}"
                   ).format(e["filas_leidas"], e["filas_con_desenlace"], e["codificacion"],
                            e["separador"]))
    al = informe["alcance"]
    lineas.append(("Desenlace: {} · puntuación calculada: {}" if i == "es" else
                   "Outcome: {} · score calculated: {}").format(
                       al["horizonte_del_desenlace"], al["momento_de_la_puntuacion"]))
    lineas.append(informe["donde_se_calcula"]["texto"][i])
    if e["indice_sin_nombre_texto"]:
        lineas.append(e["indice_sin_nombre_texto"][i])
    for u in informe["desenlace"]["variantes_unidas"]:
        lineas.append(("Variantes unidas en «{}»: {}" if i == "es" else
                       "Variants merged into «{}»: {}").format(
                           u["valor"], ", ".join(f"{v['texto']} ({v['filas']})"
                                                 for v in u["variantes"])))
    for r in informe["puntuaciones"]:
        f = r["filas"]
        lineas.append(("\n[{}] {} — {} filas, {} eventos, {} sin puntuación" if i == "es" else
                       "\n[{}] {} — {} rows, {} events, {} without a score").format(
                           r["indice"], r["columna"], f["usadas"], r["eventos"],
                           f["sin_puntuacion"]))
        lineas.append("  " + r["validacion_externa"]["nota"][i])
        lineas.append("  AUROC: " + con_ic(r["discriminacion"]["auroc"]))
        lineas.append("  AP: " + con_ic(r["discriminacion"]["average_precision"]))
        cal = r["calibracion"]
        if cal["aplica"]:
            lineas.append(("  Calibración en grande: " if i == "es" else
                           "  Calibration-in-the-large: ") + con_ic(cal["en_grande"]))
            lineas.append(("  Intercepto: " if i == "es" else "  Intercept: ")
                          + con_ic(cal["intercepto"]))
            lineas.append(("  Pendiente: " if i == "es" else "  Slope: ")
                          + con_ic(cal["pendiente"]))
        else:
            lineas.append(("  Calibración: no aplica. " if i == "es" else
                           "  Calibration: not applicable. ") + cal["motivo"][i])
        u = r["al_umbral"]
        if u["declarado"]:
            lineas.append(("  Al umbral {}: sensibilidad ".format(u["umbral"]) if i == "es" else
                           "  At threshold {}: sensitivity ".format(u["umbral"]))
                          + con_ic(u["sensibilidad"]))
            lineas.append(("  Especificidad: " if i == "es" else "  Specificity: ")
                          + con_ic(u["especificidad"]))
            if u["vpp"]["valor"] is None:
                lineas.append("  PPV/NPV: " + (u["vpp"]["no_definido"] or {}).get(i, "—"))
        for a in r["avisos"]:
            lineas.append("  ! " + a[i])
    c = informe["comparacion"]
    if c:
        if c["posible"]:
            lineas.append(("\nA − B (AUROC) sobre {} filas comunes: {}".format(
                c["filas_comunes"], n(c["diferencia"])) if i == "es" else
                "\nA − B (AUROC) on {} common rows: {}".format(
                    c["filas_comunes"], n(c["diferencia"])))
                + (f"  (IC 95 %: {n(c['ic']['bajo'])} – {n(c['ic']['alto'])})"
                   if c["ic"] else ""))
        else:
            lineas.append("\n" + c["motivo"][i])
    for a in informe["avisos"]:
        lineas.append("! " + a[i])
    lineas.append("")
    for t in informe["lo_que_no_es"]:
        lineas.append("· " + t[i])
    return "\n".join(lineas)
