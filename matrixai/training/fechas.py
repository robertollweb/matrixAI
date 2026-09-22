# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C5 — las FECHAS como variables.

**Qué pasa hoy, medido el 2026-09-21.** Una columna de fechas que entra como
predictor llega a `ajustar_preparacion` como texto, y la preparación la toma por
CATEGÓRICA (su detección es `isinstance(v, (int, float))`, por diseño). Cada
fecha distinta es una categoría, y una fecha que no estaba en el entrenamiento
—que en uso son TODAS: se predice lo que viene— sale `__desconocida__`. La
columna no aporta nada justo en las filas que importan, y nada lo dice.

**Qué hace este módulo.** Convierte una fecha en cuatro números que sí tienen
sentido para una fecha que nunca se vio:

- `anio`, `mes` (1–12) y `dia_de_la_semana` (0 = lunes … 6 = domingo, el
  `weekday()` de Python);
- `dias_desde_1970`: los días desde el 1970-01-01, con la fracción del día si
  la fecha trae hora. La referencia es FIJA: no depende de los datos, así que
  una fila nueva se convierte igual que las del entrenamiento sin guardar nada
  de ellas;
- y `hora` (0 a 24, con los minutos como fracción), SOLO si el formato trae
  hora. Añadida el 2026-09-22 antes de medir: dos de los siete conjuntos
  elegidos para C5 son horarios (tráfico, recogidas), y un árbol no saca el
  ciclo del día de la fracción de `dias_desde_1970`. Con un formato sin hora no
  se inventa una columna constante: la receta tiene cuatro variables.

**Qué NO hace.** No adivina formatos fila a fila: el formato se decide una vez,
sobre la columna entera, con la MISMA detección que el análisis del conjunto
(`dataset_analysis._detect_date_format`, reutilizada y no copiada), y viaja en
la receta. Una columna con formatos mezclados no se detecta y se queda como
estaba. Un valor que no se puede leer con el formato de su receta no se
rellena: sus cuatro variables salen como faltantes y la columna se DEVUELVE
para que el llamante lo diga.

**Estado.** Funciones puras, sin cablear: este fichero nace con la pasada de
Fase 0 del 113 viva, y el cableado (la preparación, «Usar», el paquete) se hace
después. Hasta entonces no lo importa nadie, a propósito.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from matrixai.training.dataset_analysis import _DATE_FORMATS, _detect_date_format

#: Los formatos que se pueden confundir entre sí: día/mes y mes/día. Si una columna
#: se lee ENTERA con los dos (todos los días ≤ 12), la detección no puede saber cuál
#: es, y el orden de `_DATE_FORMATS` elegiría por ella (auditoría del 2026-09-22:
#: una serie mensual de EE. UU., `MM/01/AAAA`, salía como `%d/%m/%Y`, con todo el año
#: comprimido en los días 1–12 de enero y sin un aviso).
_CONFUNDIBLES = ("%d/%m/%Y", "%m/%d/%Y")

#: La referencia de `dias_desde_1970`. Cambiarla cambia todos los números de
#: una receta ya guardada: por eso va escrita en la receta y se comprueba al
#: leerla.
REFERENCIA = "1970-01-01"
_REFERENCIA = datetime(1970, 1, 1)

#: Las variables que salen de cada fecha, en este orden. Mismo motivo que la
#: referencia: van en la receta y se comprueban al leerla.
VARIABLES = ("anio", "mes", "dia_de_la_semana", "dias_desde_1970")
#: Y con un formato que trae hora, una más.
VARIABLES_CON_HORA = VARIABLES + ("hora",)


def variables_del_formato(formato: str) -> tuple[str, ...]:
    return VARIABLES_CON_HORA if "%H" in formato else VARIABLES


class RecetaDeFechaIncompatible(ValueError):
    """Una receta guardada con otras variables, otra referencia u otro formato
    que este código no sabe leer. Se niega en vez de convertir a su manera: una
    fecha convertida con otra receta da otros números, y el modelo los
    recibiría sin que nada lo dijera."""


@dataclass(frozen=True)
class RecetaDeFecha:
    """Cómo se convierte UNA columna: su nombre y el formato detectado."""

    columna: str
    formato: str

    def __post_init__(self) -> None:
        if self.formato not in _DATE_FORMATS:
            raise RecetaDeFechaIncompatible(
                f"formato {self.formato!r} desconocido para {self.columna!r}: "
                f"los que se saben leer son {list(_DATE_FORMATS)}")

    @property
    def variables(self) -> tuple[str, ...]:
        return variables_del_formato(self.formato)

    def columnas_derivadas(self) -> tuple[str, ...]:
        return tuple(f"{self.columna}__{v}" for v in self.variables)

    def a_json(self) -> dict[str, Any]:
        return {"columna": self.columna, "formato": self.formato,
                "variables": list(self.variables), "referencia": REFERENCIA}

    @classmethod
    def desde_json(cls, datos: Mapping[str, Any]) -> "RecetaDeFecha":
        esperadas = list(variables_del_formato(str(datos.get("formato"))))
        if list(datos.get("variables") or []) != esperadas:
            raise RecetaDeFechaIncompatible(
                f"la receta de {datos.get('columna')!r} declara las variables "
                f"{datos.get('variables')!r} y este código produce {esperadas!r}")
        if datos.get("referencia") != REFERENCIA:
            raise RecetaDeFechaIncompatible(
                f"la receta de {datos.get('columna')!r} cuenta los días desde "
                f"{datos.get('referencia')!r} y este código, desde {REFERENCIA}")
        return cls(columna=str(datos["columna"]), formato=str(datos["formato"]))


def _es_faltante(valor: Any) -> bool:
    return valor is None or (isinstance(valor, str) and valor.strip() == "")


def detectar_fechas(filas: Sequence[Mapping[str, Any]],
                    columnas: Sequence[str]) -> tuple[RecetaDeFecha, ...]:
    """Las columnas de `columnas` que son fechas, con su formato.

    Una columna es fecha si TODOS sus valores no vacíos son texto y se leen con
    un mismo formato de `_DATE_FORMATS`. Se decide sobre las filas que se pasen,
    que el llamante elige: para no mirar el test, las de entrenamiento (el
    formato de una columna no depende del pliegue, pero la regla de la
    preparación es ajustar solo con lo que se entrena, y aquí no hay motivo
    para saltársela)."""
    recetas: list[RecetaDeFecha] = []
    for columna in columnas:
        formato = _formato_de(filas, columna)
        if formato is not None and not _es_ambigua(filas, columna, formato):
            recetas.append(RecetaDeFecha(columna=columna, formato=formato))
    return tuple(recetas)


def fechas_ambiguas(filas: Sequence[Mapping[str, Any]],
                    columnas: Sequence[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Las columnas que parecen fecha pero se leen igual de bien con día/mes que
    con mes/día, con los formatos posibles. `detectar_fechas` NO las convierte: quien
    llama tiene que decirlo (o preguntar), no elegir por orden."""
    salida = []
    for columna in columnas:
        formato = _formato_de(filas, columna)
        if formato is not None and _es_ambigua(filas, columna, formato):
            salida.append((columna, _CONFUNDIBLES))
    return tuple(salida)


def _textos_de(filas: Sequence[Mapping[str, Any]], columna: str) -> list[str] | None:
    no_vacios = [f.get(columna) for f in filas if not _es_faltante(f.get(columna))]
    if not no_vacios or not all(isinstance(v, str) for v in no_vacios):
        return None
    return [v.strip() for v in no_vacios]


def _formato_de(filas: Sequence[Mapping[str, Any]], columna: str) -> str | None:
    textos = _textos_de(filas, columna)
    return None if textos is None else _detect_date_format(textos)


def _es_ambigua(filas: Sequence[Mapping[str, Any]], columna: str, formato: str) -> bool:
    if formato not in _CONFUNDIBLES:
        return False
    otro = next(f for f in _CONFUNDIBLES if f != formato)
    textos = _textos_de(filas, columna) or []
    try:
        for t in textos:
            datetime.strptime(t, otro)
    except ValueError:
        return False
    return True


def variables_de_fecha(valor: Any, formato: str) -> tuple[float, ...] | None:
    """Las variables de UN valor (cuatro, o cinco con hora), o `None` si no se puede leer con
    `formato`. Un faltante también da `None`: el que distingue los dos casos es
    `expandir_fila`, que sabe cuál fue."""
    if _es_faltante(valor) or not isinstance(valor, str):
        return None
    try:
        momento = datetime.strptime(valor.strip(), formato)
    except ValueError:
        return None
    dias = (momento - _REFERENCIA).total_seconds() / 86400.0
    fecha = (float(momento.year), float(momento.month), float(momento.weekday()), dias)
    if "%H" not in formato:
        return fecha
    return fecha + (momento.hour + momento.minute / 60.0 + momento.second / 3600.0,)


def expandir_fila(fila: Mapping[str, Any],
                  recetas: Sequence[RecetaDeFecha]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """La fila con cada columna de fecha sustituida por sus cuatro variables.

    Devuelve también las columnas cuyo valor NO se pudo leer con el formato de
    su receta (no las vacías: una vacía es un faltante, y la preparación ya sabe
    tratarlo). Sus cuatro variables salen `None` —faltantes, nunca un cero ni
    una fecha inventada— y quien llama decide cómo decirlo.

    No toca la fila que recibe."""
    nueva = dict(fila)
    ilegibles: list[str] = []
    for receta in recetas:
        valor = nueva.pop(receta.columna, None)
        variables = variables_de_fecha(valor, receta.formato)
        if variables is None and not _es_faltante(valor):
            ilegibles.append(receta.columna)
        for nombre, v in zip(receta.columnas_derivadas(),
                             variables if variables is not None else (None,) * len(receta.variables)):
            nueva[nombre] = v
    return nueva, tuple(ilegibles)
