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
from typing import Any, Mapping

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
}


def _cifra(valor: float, idioma: str) -> str:
    """Hasta seis cifras significativas, con la coma en castellano."""
    texto = f"{valor:.6g}"
    return texto.replace(".", ",") if idioma == "es" else texto


@dataclass(frozen=True)
class Marca:
    """Una columna de UNA fila con algo que el modelo no vio."""

    columna: str
    tipo: str  # "categoria_nueva" | "fuera_del_rango" | "faltante_nunca_visto"
    valor: str | float | None
    minimo: float | None = None
    maximo: float | None = None

    def __post_init__(self) -> None:
        if self.tipo not in MOTIVOS:
            raise ValueError(f"tipo de marca {self.tipo!r} desconocido: {sorted(MOTIVOS)}")
        if self.tipo == "fuera_del_rango" and (self.minimo is None or self.maximo is None):
            raise ValueError("una marca «fuera del rango» sin los extremos no dice fuera de qué")

    def motivo(self) -> dict[str, str]:
        textos: dict[str, str] = {}
        for idioma, plantilla in MOTIVOS[self.tipo].items():
            if self.tipo == "fuera_del_rango":
                textos[idioma] = plantilla.format(
                    columna=self.columna, valor=_cifra(float(self.valor), idioma),
                    minimo=_cifra(self.minimo, idioma), maximo=_cifra(self.maximo, idioma))
            else:
                textos[idioma] = plantilla.format(columna=self.columna, valor=self.valor)
        return textos

    def a_json(self) -> dict[str, Any]:
        return {"columna": self.columna, "tipo": self.tipo, "valor": self.valor,
                "minimo": self.minimo, "maximo": self.maximo, "motivo": self.motivo()}


@dataclass(frozen=True)
class ComprobacionDeLoVisto:
    """Lo que se comprobó de una fila.

    `sin_rango`: las numéricas que NO se pudieron comprobar porque la política
    no guarda sus extremos (las escritas antes del 2026-09-14). No son «dentro»:
    ese estudio no lo midió."""

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
      (`faltante_nunca_visto`); si faltaba, el modelo ya lo aprendió."""
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
            # 114-C5: una fecha fuera del periodo visto no se marca todavía —sus
            # extremos son días desde 1970, y un motivo en esos números no se lee—,
            # así que se DICE que no se comprobó (`sin_rango`), nunca un «dentro»
            # inventado. Faltar donde nunca faltó sí se marca, como en las demás.
            if _es_faltante_crudo(fila.get(columna)):
                if nunca_falto:
                    marcas.append(Marca(columna=columna, tipo="faltante_nunca_visto", valor=None))
            else:
                sin_rango.append(columna)
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


def _es_faltante_crudo(valor: Any) -> bool:
    return valor is None or (isinstance(valor, str) and valor.strip() == "")
