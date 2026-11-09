"""LA lista de textos que dicen «aquí no había dato». Módulo hoja (sin imports del paquete) para que
`dataset_analysis` y `preparacion` —que ya se importan en círculo por `fechas`— la compartan sin duplicarla.

Hallazgo 11 (2026-11-08): con «NA» escrito la columna numérica pasaba a categórica en la preparación mientras
el análisis la leía numérica; eran dos sitios decidiendo lo mismo con reglas distintas."""
from __future__ import annotations

from typing import Any, Callable, Iterable

#: La heurística GENERAL del CSV tecleado a mano (cuenta faltantes en cualquier columna y en el objetivo). Como siempre.
TEXTOS_DE_FALTANTE_GENERALES = frozenset({"", "na", "n/a", "null", "nan", "none", "-", "?"})

#: Los que, ADEMÁS, dicen «no hay dato» pero SOLO en una columna que sin ellos es numérica (o fecha): `#n/a` es el error de Excel;
#: `.`, el faltante de SAS/Stata. En una de texto (o en el objetivo) son un valor, como antes del hallazgo 11.
TEXTOS_SOLO_EN_NUMERICA = frozenset({"#n/a", "."})

#: Sin distinguir mayúsculas ni espacios. La lista de una columna que sin ellos es numérica o fecha (preparación y análisis).
TEXTOS_DE_FALTANTE = TEXTOS_DE_FALTANTE_GENERALES | TEXTOS_SOLO_EN_NUMERICA


def es_texto_de_faltante(valor: Any) -> bool:
    """¿Es una CADENA de `TEXTOS_DE_FALTANTE`? En la preparación solo se aplica a una columna que, sin esos
    textos, es numérica: en una de texto «NA» puede ser una categoría legítima (Norteamérica) y NO se toca."""
    if not isinstance(valor, str):
        return False
    texto = valor.strip().lower()
    # «nan» es el único de la lista que `float()` ya lee: llega al modelo como NaN con su indicador a 0, como siempre
    # (medido por `test_152_c1_una_sola_cadena`, 120 filas por ejemplo publicado). No cambia aquí.
    return texto in TEXTOS_DE_FALTANTE and texto != "nan"


def sin_dato_en_un_objetivo_numerico(valor: Any) -> bool:
    """¿Esta celda del OBJETIVO de una REGRESIÓN dice «aquí no había dato»? Entonces la fila es una fila SIN objetivo: se quita
    del estudio (y del CSV preparado) y se DECLARA, nunca se rellena ni se rechaza el CSV por ella (hallazgo 11, M3; decisión A
    de Roberto, 09-11, tarjeta `h11-objetivo-con-na-de-excel`). Vacía, `None`, NaN, o cualquier texto de LA lista
    (`TEXTOS_DE_FALTANTE`: «NA», «#N/A», «-», «.», « na »…, sin mayúsculas ni espacios), «nan» incluido: en una predictora
    «nan» llega al modelo como NaN, pero un objetivo sin número no tiene nada que medir.

    Solo para un objetivo NUMÉRICO: en una clasificación «NA» puede ser una clase (Norteamérica) y en un texto, un valor; ahí
    sigue mandando el criterio de siempre (`dataset_analysis._is_null`). Antes de esto, «NA» se quitaba (era de la lista
    general) y «#N/A» de Excel paraba el estudio entero con «no es un número»: eran dos reglas para la misma celda."""
    if valor is None:
        return True
    if isinstance(valor, float):
        return valor != valor  # NaN
    if not isinstance(valor, str):
        return False
    texto = valor.strip().lower()
    return texto == "" or texto in TEXTOS_DE_FALTANTE


def textos_leidos_como_faltante(distintos: Iterable[Any], el_resto_es: Callable[[list[Any]], bool]) -> frozenset[Any]:
    """LA REGLA, en un solo sitio (reparación del hallazgo 11): de los valores DISTINTOS de una columna (sin los vacíos), los
    que son texto de faltante —y se leerán como faltante—, SOLO si quitados queda algo y ese resto `el_resto_es` (numérico,
    fecha: lo decide quien llama con SU lectura del tipo). Si no, ninguno: la columna es de texto y «NA» es un valor.

    La usan el tipado y la preparación (`preparacion`), la detección de fechas (`fechas`) y el análisis (`dataset_analysis`):
    antes cada uno decidía con su regla, y el análisis llamaba identificador a una columna que el estudio usaba como número."""
    textos, resto = [], []
    for v in distintos:
        (textos if es_texto_de_faltante(v) else resto).append(v)
    if not textos or not resto or not el_resto_es(resto):
        return frozenset()
    return frozenset(textos)


def recuento_por_texto(valores: Iterable[Any], textos: Iterable[Any]) -> dict[str, int]:
    """`{texto: celdas}` de los `textos` leídos como faltante, sin repetir: «NA», « NA» y «NA » son UN texto, «NA»,
    con tres celdas (auditoría del hallazgo 11, M1: se declaraban `['NA', 'NA', 'NA']`). Y por su forma NORMALIZADA, la
    misma con la que `es_texto_de_faltante` los reconoce (recortada y en minúsculas): «NA» y «na» son UN texto con la
    cuenta de los dos (re-auditoría del hallazgo 11, obs.: se declaraban como dos). Se nombra con la forma más frecuente
    tal como venía (recortada; a igualdad, la primera por orden). Ordenado por ese nombre."""
    leidos = set(textos)
    formas: dict[str, dict[str, int]] = {}
    for v in valores:
        if v in leidos:
            t = str(v).strip()
            de_esta = formas.setdefault(t.lower(), {})
            de_esta[t] = de_esta.get(t, 0) + 1
    cuenta = {min(f, key=lambda t: (-f[t], t)): sum(f.values()) for f in formas.values()}
    return dict(sorted(cuenta.items()))
