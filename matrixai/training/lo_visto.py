# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""115-C1 — «valor nunca visto» y «fuera del rango visto», fila a fila.

**Qué pasa hoy.** `transformar_fila` ya sabe, columna a columna, cuándo una
categoría no apareció en el entrenamiento: la cambia por `__desconocida__` y
sigue. El modelo predice igual, y **nadie se entera** (112, H27). Y de un número
fuera de lo que se vio al entrenar no se dice nada, aunque la política guarda
los extremos desde el 2026-09-14 (`PropuestaDeColumna.minimo`/`maximo`).

**Qué hace este módulo.** Para una fila, qué columnas traen algo que el modelo
no vio, con el valor y lo que sí vio; y, cuando no hay nada, lo AFIRMA — pero
solo si pudo comprobarlo todo: una política antigua sin extremos no puede decir
«dentro de lo visto» de sus numéricas, y eso sale como «sin comprobar», no como
un sí.

**Una sola decisión de «desconocida».** Lo que es una categoría nueva lo decide
`transformar_fila`, y aquí se le PREGUNTA (se mira qué columnas devolvió como
`CATEGORIA_DESCONOCIDA`) en vez de repetir su criterio. Dos sitios decidiendo lo
mismo acaban divergiendo, y aquí la divergencia sería una marca que dice
«nuevo» sobre un valor que el modelo sí trató como conocido, o al revés.

**Estado.** Sin cablear: nace con la pasada de Fase 0 del 113 viva. «Usar», el
lote y el paquete lo llaman después.

**Lo que redacta el núcleo se traduce en el núcleo**: cada marca guarda el HECHO
(columna, tipo, valor, extremos) y compone su motivo en los dos idiomas, con las
cifras escritas a la manera de cada uno.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from matrixai.training.fechas import VARIABLES, variables_de_fecha
from matrixai.training.preparacion import (
    CATEGORIA_DESCONOCIDA,
    CATEGORIA_FALTANTE,
    PoliticaDePreparacion,
    nombre_de_indicador,
    transformar_fila,
)

__all__ = ["ComprobacionDeLoVisto", "Marca", "comprobar_lo_visto"]

#: Clave -> {idioma: plantilla}. Uno debajo del otro, como en los demás
#: catálogos del núcleo: una clave sin pareja salta a la vista.
MOTIVOS: dict[str, dict[str, str]] = {
    "categoria_nueva": {
        "es": "«{columna}»: el valor «{valor}» no apareció al entrenar; el modelo lo trata "
              "como desconocido",
        "en": "“{columna}”: the value “{valor}” did not appear in training; the model treats "
              "it as unknown",
    },
    "fuera_del_rango": {
        "es": "«{columna}»: {valor} queda fuera de lo que se vio al entrenar (de {minimo} a "
              "{maximo})",
        "en": "“{columna}”: {valor} is outside what training saw (from {minimo} to {maximo})",
    },
    "faltante_nunca_visto": {
        "es": "«{columna}»: viene vacío, y al entrenar nunca faltó; el modelo lo rellena, pero "
              "lo que prediga así no se ha medido",
        "en": "“{columna}”: it is empty, and it was never missing in training; the model fills "
              "it in, but what it predicts like this has not been measured",
    },
    # 142, I1 (auditoría a ciegas del 29-10): una fecha que no se lee con el formato del
    # estudio sale FALTANTE en sus variables (`transformar_fila`), así que el modelo la
    # predice como el hueco —idéntica al bit, medido con LightGBM y con la densa— y aquí no
    # se decía nada. Se marca SIEMPRE, haya faltado o no la fecha al entrenar: aunque el
    # modelo sepa qué hacer con un hueco, la persona escribió una fecha y no se leyó.
    "fecha_ilegible": {
        "es": "«{columna}»: «{valor}» no se pudo leer como fecha; el modelo la trata como si "
              "faltara",
        "en": "“{columna}”: “{valor}” could not be read as a date; the model treats it as if it "
              "were missing",
    },
    # 142, M1: el periodo visto son el mínimo y el máximo de `{columna}__dias_desde_1970`
    # que la política guarda desde el 114-C5 (sus derivadas); se escriben como fechas.
    "fecha_fuera_del_periodo": {
        "es": "«{columna}»: {valor} queda fuera del periodo que se vio al entrenar (de {minimo} "
              "a {maximo})",
        "en": "“{columna}”: {valor} is outside the period training saw (from {minimo} to "
              "{maximo})",
    },
}

#: Las marcas que llevan los extremos de lo visto, y sin ellos no dicen fuera de qué.
_CON_EXTREMOS = ("fuera_del_rango", "fecha_fuera_del_periodo")


def _cifra(valor: float, idioma: str) -> str:
    """Hasta seis cifras significativas, con la coma en castellano."""
    texto = f"{valor:.6g}"
    return texto.replace(".", ",") if idioma == "es" else texto


def _fecha_de_dias(dias: float) -> str:
    """Días desde 1970 (la referencia FIJA de `fechas.py`) escritos como fecha ISO, y con la
    hora solo si la trae: en días se guardan, pero «19523» no lo lee nadie. ISO en los dos
    idiomas, que no se confunde día con mes."""
    # Redondeado al minuto: la fracción del día viaja en coma flotante, y 10:00 no debe salir 09:59.
    momento = datetime(1970, 1, 1) + timedelta(minutes=round(dias * 1440))
    return momento.strftime("%Y-%m-%d %H:%M" if (momento.hour or momento.minute) else "%Y-%m-%d")


@dataclass(frozen=True)
class Marca:
    """Una columna de UNA fila con algo que el modelo no vio."""

    columna: str
    #: "categoria_nueva" | "fuera_del_rango" | "faltante_nunca_visto" | "fecha_ilegible"
    #: | "fecha_fuera_del_periodo". Con las dos de fecha, `valor` es el texto tal como
    #: llegó; con la del periodo, `minimo`/`maximo` son días desde 1970 (las unidades de
    #: la política), y el motivo los escribe como fechas.
    tipo: str
    valor: str | float | None
    minimo: float | None = None
    maximo: float | None = None

    def __post_init__(self) -> None:
        if self.tipo not in MOTIVOS:
            raise ValueError(f"tipo de marca {self.tipo!r} desconocido: {sorted(MOTIVOS)}")
        if self.tipo in _CON_EXTREMOS and (self.minimo is None or self.maximo is None):
            raise ValueError(f"una marca «{self.tipo}» sin los extremos no dice fuera de qué")

    def motivo(self) -> dict[str, str]:
        textos: dict[str, str] = {}
        for idioma, plantilla in MOTIVOS[self.tipo].items():
            if self.tipo == "fuera_del_rango":
                textos[idioma] = plantilla.format(
                    columna=self.columna, valor=_cifra(float(self.valor), idioma),
                    minimo=_cifra(self.minimo, idioma), maximo=_cifra(self.maximo, idioma))
            elif self.tipo == "fecha_fuera_del_periodo":
                textos[idioma] = plantilla.format(
                    columna=self.columna, valor=self.valor,
                    minimo=_fecha_de_dias(self.minimo), maximo=_fecha_de_dias(self.maximo))
            else:
                textos[idioma] = plantilla.format(columna=self.columna, valor=self.valor)
        return textos

    def a_json(self) -> dict[str, Any]:
        return {"columna": self.columna, "tipo": self.tipo, "valor": self.valor,
                "minimo": self.minimo, "maximo": self.maximo, "motivo": self.motivo()}


@dataclass(frozen=True)
class ComprobacionDeLoVisto:
    """Lo que se comprobó de una fila.

    `sin_rango`: las columnas que NO se pudieron comprobar porque la política
    no guarda sus extremos (las numéricas de las escritas antes del 2026-09-14, y
    una fecha cuyas derivadas no los traen). No son «dentro»: ese estudio no lo
    midió."""

    marcas: tuple[Marca, ...]
    sin_rango: tuple[str, ...] = ()

    @property
    def dentro_de_lo_visto(self) -> bool | None:
        """`True` solo si se comprobó TODO y no hay marcas; `False` con alguna
        marca; `None` si no hay marcas pero quedó algo sin comprobar."""
        if self.marcas:
            return False
        return None if self.sin_rango else True

    def a_json(self) -> dict[str, Any]:
        return {"dentro_de_lo_visto": self.dentro_de_lo_visto,
                "marcas": [m.a_json() for m in self.marcas],
                "sin_rango": list(self.sin_rango)}


def comprobar_lo_visto(fila: Mapping[str, Any],
                       politica: PoliticaDePreparacion) -> ComprobacionDeLoVisto:
    """Las marcas de `fila` (CRUDA, la misma que recibiría `transformar_fila`).

    - categórica: marcada si `transformar_fila` la devolvió como
      `CATEGORIA_DESCONOCIDA`, con el valor que traía;
    - numérica: marcada si su valor queda por debajo del `minimo` o por encima
      del `maximo` que vio el entrenamiento (los extremos mismos están dentro);
    - un faltante se marca SOLO si al entrenar esa columna nunca faltó
      (`faltante_nunca_visto`); si faltaba, el modelo ya lo aprendió;
    - fecha: marcada si no se puede leer con el formato del estudio
      (`fecha_ilegible`, la misma lectura que `transformar_fila`), o si cae fuera del
      periodo visto (`fecha_fuera_del_periodo`, los extremos de sus días desde 1970)."""
    transformada = transformar_fila(dict(fila), politica)
    marcas: list[Marca] = []
    sin_rango: list[str] = []
    for propuesta in politica.columnas:
        columna = propuesta.columna
        # UN FALTANTE donde el entrenamiento nunca tuvo ninguno SÍ es algo que el
        # modelo no vio (auditoría del 2026-09-22: con la columna vacía, «Probar una
        # fila» afirmaba «todos los valores están dentro de lo visto»). Donde sí
        # faltaba al entrenar, no se marca: eso el modelo lo aprendió.
        nunca_falto = propuesta.proporcion_faltante == 0
        if propuesta.tipo == "fecha":
            crudo = fila.get(columna)
            if _es_faltante_crudo(crudo):
                if nunca_falto:
                    marcas.append(Marca(columna=columna, tipo="faltante_nunca_visto", valor=None))
                continue
            # 142, I1: la MISMA lectura que `transformar_fila` (`variables_de_fecha` con el
            # formato de la propuesta), no otra: si allí sale `None`, el modelo recibió un
            # hueco, y aquí se dice. Un criterio propio podría marcar una fecha que el
            # modelo sí leyó (`2023-6-15` se lee con `%Y-%m-%d`, medido) o al revés.
            variables = variables_de_fecha(crudo, propuesta.formato_de_fecha)
            if variables is None:
                marcas.append(Marca(columna=columna, tipo="fecha_ilegible", valor=str(crudo)))
                continue
            # 142, M1: el periodo visto, con los extremos que la política YA guarda en la
            # derivada `__dias_desde_1970` (114-C5). Antes se mandaba a `sin_rango` con una
            # razón falsa («no guardó el rango»): ahora solo va ahí si de verdad no lo trae.
            dias = _derivada_de_dias(politica, propuesta)
            if dias is None or dias.minimo is None or dias.maximo is None:
                sin_rango.append(columna)
                continue
            valor_en_dias = variables[VARIABLES.index("dias_desde_1970")]
            if valor_en_dias < dias.minimo or valor_en_dias > dias.maximo:
                marcas.append(Marca(columna=columna, tipo="fecha_fuera_del_periodo",
                                    valor=str(crudo).strip(),
                                    minimo=dias.minimo, maximo=dias.maximo))
            continue
        if propuesta.tipo == "categorica":
            if transformada.get(columna) == CATEGORIA_DESCONOCIDA:
                marcas.append(Marca(columna=columna, tipo="categoria_nueva",
                                    valor=str(fila.get(columna))))
            elif transformada.get(columna) == CATEGORIA_FALTANTE and nunca_falto:
                marcas.append(Marca(columna=columna, tipo="faltante_nunca_visto", valor=None))
            continue
        if transformada.get(nombre_de_indicador(columna)) == 1.0:
            if nunca_falto:
                marcas.append(Marca(columna=columna, tipo="faltante_nunca_visto", valor=None))
            continue
        if propuesta.minimo is None or propuesta.maximo is None:
            sin_rango.append(columna)
            continue
        valor = float(transformada[columna])
        if valor < propuesta.minimo or valor > propuesta.maximo:
            marcas.append(Marca(columna=columna, tipo="fuera_del_rango", valor=valor,
                                minimo=propuesta.minimo, maximo=propuesta.maximo))
    return ComprobacionDeLoVisto(marcas=tuple(marcas), sin_rango=tuple(sin_rango))


def _derivada_de_dias(politica: PoliticaDePreparacion, propuesta: Any) -> Any:
    """La derivada `{columna}__dias_desde_1970` de una fecha, o `None` si la política no la
    trae. Por NOMBRE y no por posición: la receta decide el orden (`fechas.VARIABLES`)."""
    nombre = f"{propuesta.columna}__dias_desde_1970"
    return next((d for d in politica.derivadas_de(propuesta) if d.columna == nombre), None)


def _es_faltante_crudo(valor: Any) -> bool:
    return valor is None or (isinstance(valor, str) and valor.strip() == "")
