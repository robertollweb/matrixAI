# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140-C1 — el tiempo como lo trae la gente, y la ventana que se expande.

**Tres piezas, todas puras (Python sin dependencias: el núcleo tiene
`dependencies = []`).**

1. `convertir_tiempos`: la columna de tiempo llega como TEXTO del CSV
   (`2024-01-31`, `31/01/2024`), como `datetime`/`date` o como número. El texto
   se lee con el formato que detecta `fechas.py` (la misma detección del análisis
   del conjunto, no una copia). **Las fechas ambiguas (día/mes frente a
   mes/día) se RECHAZAN con su motivo, no se eligen por orden** (`fechas.py`
   lo exige a quien llama). Lo que no se puede leer —un vacío, un texto que no
   es fecha ni número, un `nan`— también se rechaza, con cuántos valores y
   ejemplos: una fila sin fecha no se puede colocar antes ni después de un corte.
2. `convertir_gap`: el hueco llega del JSON como `dict` (`{"magnitud": 7,
   "unidad": "days"}`), no como `Horizonte`. Se convierte; lo que no se entiende
   se rechaza con motivo, no con un `AttributeError`.
3. `elegir_ventana`: los PLIEGUES de la validación hacia delante (ventana que
   se expande) con hueco.

**La ventana, y las decisiones que no son obvias.**

- Las filas de desarrollo se ordenan por (tiempo, id) y se cortan en `K+1`
  bloques cronológicos de tamaño parecido. **El primer bloque solo entrena**
  (no tiene un pasado con el que validarse); los `K` siguientes se validan, cada
  uno con todo lo anterior. Es la forma de `TimeSeriesSplit(K)`, que se usa en
  las pruebas como ORÁCULO (no se importa aquí).
- **Filas con el MISMO instante nunca quedan a los dos lados de un corte**: los
  cortes caen entre instantes distintos. Con pocos instantes distintos se
  reducen los pliegues (o se rechaza).
- **El pliegue 0 es el MÁS RECIENTE** (el de más entrenamiento, validado con lo
  último del desarrollo). Toda la selección y el modelo entregado del estudio
  salen del pliegue (0, 0): si el 0 fuera el primero, serían los de menos datos.
- **`entrena` sale ORDENADO por tiempo** (y `valida` también): la cola de parada
  temprana del backend es «las últimas filas», y solo es hacia delante si lo está.
- **El hueco (`gap`) quita de `entrena` las filas cuyo tiempo esté a menos de
  `gap` del inicio de su `valida`**: la etiqueta de una fila del final del
  entrenamiento mira `gap` hacia delante y puede caer dentro de lo que se valida.
- **Sin semilla**: la ventana es determinista; barajar mezclaría pasado y futuro.
  Por eso `repeats` se fija a 1 (el llamante lo declara).
- **Guarda de clase (D2 del contrato 140)**: con objetivo de clasificación, cada
  bloque de validación —y cada entrenamiento— tiene que tener TODAS las clases.
  Si con `K` pliegues no, se baja `K` hasta 2 declarándolo; si ni con 2, se
  rechaza con los números (cuántas filas de la clase hay y cómo se reparten por
  bloque).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from matrixai.estudio import Horizonte
from matrixai.estudio.vocabulario import UNIDADES_DE_TIEMPO
from matrixai.training.diagnostico import Limite
from matrixai.training.dataset_analysis import _DATE_FORMATS
from matrixai.training.fechas import _CONFUNDIBLES, detectar_fechas, fechas_ambiguas
from matrixai.training.particion_por_diseno_comun import Rechazo, clave_de_clase
from matrixai.training.particion_por_diseno_textos import motivo

__all__ = [
    "MINIMO_BLOQUES_VALIDOS", "TiemposConvertidos", "VentanaExpansiva",
    "convertir_gap", "convertir_tiempos", "elegir_ventana", "segundos_de",
]

#: Con 2 pliegues hay 3 bloques: uno que solo entrena y dos que se validan.
MINIMO_PLIEGUES = 2
MINIMO_BLOQUES_VALIDOS = MINIMO_PLIEGUES + 1

_SEGUNDOS_POR_UNIDAD = {
    "seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0,
    "weeks": 604800.0, "months": 2629800.0, "years": 31557600.0,
}
_ORIGEN = datetime(1970, 1, 1)
_EJEMPLOS = 3


def segundos_de(horizonte: Horizonte) -> float:
    return horizonte.magnitud * _SEGUNDOS_POR_UNIDAD[horizonte.unidad]


# ---------------------------------------------------------------------------
# 1. El tiempo de cada fila
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TiemposConvertidos:
    """El tiempo de cada fila, en segundos, y de dónde salió."""

    valores: dict[str, float]
    #: El formato de fecha detectado en el texto (`%d/%m/%Y`…); `None` si el
    #: tiempo no venía como texto de fecha.
    formato: str | None
    #: `False` si es un número sin unidad (el hueco se aplica como segundos).
    es_fecha: bool


def _vacio(valor: Any) -> bool:
    return valor is None or (isinstance(valor, str) and valor.strip() == "")


def _ejemplos(valores: Sequence[Any]) -> str:
    vistos: list[str] = []
    for v in valores:
        texto = "(vacío)" if _vacio(v) else repr(v)
        if texto not in vistos:
            vistos.append(texto)
        if len(vistos) == _EJEMPLOS:
            break
    return ", ".join(vistos)


def _ilegible(campo: str, malos: Sequence[Any], total: int) -> Rechazo:
    return Rechazo("tiempo_ilegible", "time_column",
                   {"columna": campo, "valores_ilegibles": len(malos), "filas": total,
                    "ejemplos": _ejemplos(malos)},
                   campo=campo, valor=len(malos), ejemplos=_ejemplos(malos))


def _como_numero(valor: Any) -> float | None:
    if isinstance(valor, bool):
        return None
    try:
        numero = float(valor.strip() if isinstance(valor, str) else valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None


def _se_lee_como(texto: str, formato: str | None) -> bool:
    if formato is None:
        return _como_numero(texto) is not None
    try:
        datetime.strptime(texto.strip(), formato)
    except ValueError:
        return False
    return True


def convertir_tiempos(valores: Mapping[str, Any], campo: str) -> TiemposConvertidos:
    """`{id: valor del CSV}` → `{id: segundos}`. Lanza `Rechazo` si no se
    entiende, si falta algún valor o si las fechas son ambiguas."""
    malos = [v for v in valores.values() if _vacio(v)]
    if malos:
        raise _ilegible(campo, malos, len(valores))

    if all(isinstance(v, str) for v in valores.values()):
        filas = [{campo: v} for v in valores.values()]
        if fechas_ambiguas(filas, [campo]):
            raise Rechazo("fecha_ambigua", "time_column",
                          {"columna": campo, "formatos_posibles": list(_CONFUNDIBLES),
                           "ejemplos": _ejemplos(list(valores.values()))},
                          campo=campo, ejemplos=_ejemplos(list(valores.values())))
        recetas = detectar_fechas(filas, [campo])
        if recetas:
            formato = recetas[0].formato
            cache: dict[str, float] = {}
            salida: dict[str, float] = {}
            for obs, texto in valores.items():
                if texto not in cache:
                    cache[texto] = (datetime.strptime(texto.strip(), formato) - _ORIGEN).total_seconds()
                salida[obs] = cache[texto]
            return TiemposConvertidos(salida, formato, True)
        # Ni una fecha con un solo formato: ¿son todos números escritos como texto
        # (`2024`, `20240131`)? Un CSV lo trae así.
        numeros = {obs: _como_numero(t) for obs, t in valores.items()}
        malos = [t for obs, t in valores.items() if numeros[obs] is None]
        if malos:
            # Para decir CUÁNTOS no se entienden hay que saber qué se esperaba: la lectura (fecha con algún formato, o
            # número) que entiende más valores. Con 58 fechas y 2 «N/D», los ilegibles son 2, no 60.
            mejor = [t for t in valores.values() if not _se_lee_como(t, None)]
            for formato in _DATE_FORMATS:
                sin_leer = [t for t in valores.values() if not _se_lee_como(t, formato)]
                if len(sin_leer) < len(mejor):
                    mejor = sin_leer
            raise _ilegible(campo, mejor, len(valores))
        return TiemposConvertidos({o: float(n) for o, n in numeros.items()}, None, False)

    salida = {}
    hay_fecha = False
    malos = []
    for obs, valor in valores.items():
        if isinstance(valor, datetime):
            salida[obs] = valor.timestamp()
            hay_fecha = True
        elif isinstance(valor, date):
            salida[obs] = datetime(valor.year, valor.month, valor.day).timestamp()
            hay_fecha = True
        else:
            numero = _como_numero(valor)
            if numero is None:
                malos.append(valor)
            else:
                salida[obs] = numero
    if malos:
        raise _ilegible(campo, malos, len(valores))
    return TiemposConvertidos(salida, None, hay_fecha)


def convertir_gap(gap: Any) -> Horizonte | None:
    """El hueco como `Horizonte`: acepta `None`, un `Horizonte` o el `dict` del
    JSON. Cualquier otra cosa, o un `dict` que no se entiende, es un `Rechazo`."""
    if gap is None or isinstance(gap, Horizonte):
        return gap
    detalle = ""
    if isinstance(gap, Mapping):
        try:
            horizonte = Horizonte.desde_json(dict(gap), "gap")
            if math.isfinite(horizonte.magnitud):
                return horizonte
            detalle = "la magnitud no es finita"
        except (ValueError, TypeError, KeyError) as exc:
            detalle = str(exc)
    raise Rechazo("hueco_ilegible", "gap", {"valor": repr(gap), "detalle": detalle},
                  valor=repr(gap), opciones=", ".join(UNIDADES_DE_TIEMPO))


# ---------------------------------------------------------------------------
# 2. La ventana que se expande
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VentanaExpansiva:
    """`pliegues[p] = (entrena, valida)`; el pliegue 0 es el más reciente."""

    folds: int
    #: Los `K+1` bloques cronológicos de desarrollo (el primero solo entrena).
    bloques: tuple[tuple[str, ...], ...]
    pliegues: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    #: Cuántas filas quitó el hueco de `entrena`, por pliegue.
    quitadas_por_el_hueco: tuple[int, ...]

    @property
    def sin_validacion(self) -> tuple[str, ...]:
        """Las filas de desarrollo que no están en ninguna `valida`: el primer
        bloque. No tienen predicción fuera de pliegue."""
        return self.bloques[0]


class _NoCabe(Exception):
    def __init__(self, causa: str, **datos: Any) -> None:
        super().__init__(causa)
        self.causa = causa
        self.datos = datos


def _instantes(ids: Sequence[str], tiempo: Mapping[str, float]) -> list[list[str]]:
    """Las filas agrupadas por instante, en orden cronológico (y por id dentro)."""
    por_instante: dict[float, list[str]] = {}
    for obs in sorted(ids, key=lambda o: (tiempo[o], o)):
        por_instante.setdefault(tiempo[obs], []).append(obs)
    return [por_instante[t] for t in sorted(por_instante)]


def _inicios_de_bloque(tamanos: Sequence[int], n_bloques: int) -> list[int]:
    """En qué instante empieza cada bloque. Los cortes caen entre instantes
    distintos y lo más cerca posible del reparto en partes iguales."""
    total = sum(tamanos)
    prefijo = [0]
    for t in tamanos:
        prefijo.append(prefijo[-1] + t)
    inicios = [0]
    m = len(tamanos)
    for j in range(1, n_bloques):
        objetivo = j * total / n_bloques
        minimo, maximo = inicios[-1] + 1, m - (n_bloques - j)
        inicios.append(min(range(minimo, maximo + 1), key=lambda i: (abs(prefijo[i] - objetivo), i)))
    return inicios


def _clases_de(filas: Sequence[str], clase_por_id: Mapping[str, Any]) -> dict[Any, int]:
    cuenta: dict[Any, int] = {}
    for obs in filas:
        cuenta[clase_por_id[obs]] = cuenta.get(clase_por_id[obs], 0) + 1
    return cuenta


def _ventana(ids: Sequence[str], tiempo: Mapping[str, float], folds: int, hueco_s: float,
             clase_por_id: Mapping[str, Any] | None, clases: Sequence[Any]) -> VentanaExpansiva:
    instantes = _instantes(ids, tiempo)
    if len(instantes) < folds + 1:
        raise _NoCabe("instantes", instantes=len(instantes))
    inicios = _inicios_de_bloque([len(i) for i in instantes], folds + 1)
    limites_ = inicios + [len(instantes)]
    bloques = tuple(tuple(o for inst in instantes[limites_[j]:limites_[j + 1]] for o in inst)
                    for j in range(folds + 1))

    pliegues = []
    quitadas = []
    for p in range(folds):
        b = folds - p                       # pliegue 0 = el bloque más reciente
        valida = bloques[b]
        inicio = tiempo[valida[0]]
        pasado = [o for j in range(b) for o in bloques[j]]
        # El hueco: se quitan las filas a menos de `hueco` del inicio de la validación.
        entrena = tuple(o for o in pasado if inicio - tiempo[o] >= hueco_s)
        if not entrena:
            raise _NoCabe("hueco", pliegue=p)
        quitadas.append(len(pasado) - len(entrena))
        pliegues.append((entrena, valida))
    ventana = VentanaExpansiva(folds, bloques, tuple(pliegues), tuple(quitadas))

    if clase_por_id is not None:
        for entrena, valida in ventana.pliegues:
            for lado in (valida, entrena):
                if len(_clases_de(lado, clase_por_id)) < len(clases):
                    raise _NoCabe("clase", ventana=ventana)
    return ventana


def elegir_ventana(ids: Sequence[str], tiempo: Mapping[str, float], *, folds: int, gap: Horizonte | None,
                   clase_por_id: Mapping[str, Any] | None = None,
                   clases: Sequence[Any] | None = None,
                   campo_de_tiempo: str = "time_column"
                   ) -> tuple[VentanaExpansiva, tuple[Limite, ...]]:
    """La ventana con los pliegues pedidos o, si no cabe, con menos (hasta 2)
    declarándolo. Lanza `Rechazo` si ni con 2 cabe.

    `clase_por_id` solo se pasa en clasificación (y con más de una clase): con
    él, cada bloque de validación y cada entrenamiento tiene todas las clases
    de `clases` (por omisión, las de `ids`; el llamante pasa las de TODO el
    reparto para que una clase que solo está en la prueba también cuente).
    `campo_de_tiempo` es el nombre de la columna, para los mensajes."""
    hueco_s = segundos_de(gap) if gap is not None else 0.0
    if clase_por_id is None:
        clases = []
    elif clases is None:
        clases = sorted({clase_por_id[o] for o in ids}, key=str)
    else:
        clases = sorted(clases, key=str)
    pedidos = max(MINIMO_PLIEGUES, folds)
    ultima: _NoCabe | None = None
    for k in range(pedidos, MINIMO_PLIEGUES - 1, -1):
        try:
            ventana = _ventana(ids, tiempo, k, hueco_s, clase_por_id, clases)
        except _NoCabe as fallo:
            ultima = fallo      # lo que impidió k pliegues es lo que se declara si cabe k − 1
            continue
        if k == pedidos:
            return ventana, ()
        return ventana, _limite_de_reduccion(pedidos, k, ultima, clases, ids, clase_por_id)
    assert ultima is not None
    raise _rechazo(ultima, ids, tiempo, hueco_s, gap, clase_por_id, clases, campo_de_tiempo)


def _limite_de_reduccion(pedidos: int, usados: int, causa: _NoCabe | None, clases: Sequence[Any],
                         ids: Sequence[str], clase_por_id: Mapping[str, Any] | None) -> tuple[Limite, ...]:
    if causa is None:
        return ()
    base = {"pliegues_pedidos": pedidos, "pliegues_usados": usados}
    if causa.causa == "instantes":
        n = causa.datos["instantes"]
        return (Limite(clave="pliegues_reducidos_por_instantes", campo="time_column",
                       motivo=motivo("pliegues_reducidos_por_instantes", opciones=pedidos, valor=usados, n=n),
                       medida={**base, "instantes_distintos": n}),)
    if causa.causa == "hueco":
        return (Limite(clave="pliegues_reducidos_por_hueco", campo="gap",
                       motivo=motivo("pliegues_reducidos_por_hueco", opciones=pedidos, valor=usados),
                       medida=base),)
    assert causa.causa == "clase" and clase_por_id is not None
    conteo = _clases_de(ids, clase_por_id)
    rara = min(clases, key=lambda c: (conteo.get(c, 0), str(c)))
    return (Limite(clave="pliegues_reducidos_por_clase_en_bloques", campo="objetivo",
                   motivo=motivo("pliegues_reducidos_por_clase_en_bloques", opciones=pedidos, valor=usados,
                                 clase=clave_de_clase(rara)),
                   medida={**base, "clase_mas_rara": clave_de_clase(rara),
                           "filas_de_la_clase_mas_rara": conteo.get(rara, 0)}),)


def _rechazo(fallo: _NoCabe, ids, tiempo, hueco_s, gap, clase_por_id, clases, campo) -> Rechazo:
    if fallo.causa == "instantes":
        n = fallo.datos["instantes"]
        return Rechazo("instantes_insuficientes", "time_column",
                       {"columna": campo, "instantes_distintos": n, "minimo": MINIMO_BLOQUES_VALIDOS},
                       campo=campo, valor=n, minimo=MINIMO_BLOQUES_VALIDOS)
    if fallo.causa == "hueco":
        texto = f"{gap.magnitud:g} {gap.unidad}" if gap is not None else "0"
        return Rechazo("hueco_sin_filas_para_entrenar", "gap", {"hueco_segundos": hueco_s}, valor=texto)
    # clase: los números del último intento (2 pliegues = 3 bloques)
    ventana: VentanaExpansiva = fallo.datos["ventana"]
    conteo = _clases_de(ids, clase_por_id)
    por_bloque = [_clases_de(b, clase_por_id) for b in ventana.bloques]
    # la clase que más falla: la que menos filas tiene y no está en algún bloque validado
    faltan = [c for c in clases if any(c not in bloque for bloque in por_bloque[1:])
              or c not in por_bloque[0]]
    rara = min(faltan or clases, key=lambda c: (conteo.get(c, 0), str(c)))
    detalle = ", ".join(str(b.get(rara, 0)) for b in por_bloque)
    return Rechazo("bloques_sin_clase", "objetivo",
                   {"clase": clave_de_clase(rara), "filas_de_la_clase_en_desarrollo": conteo.get(rara, 0),
                    "pliegues": MINIMO_PLIEGUES, "filas_de_la_clase_por_bloque": [b.get(rara, 0) for b in por_bloque],
                    "filas_por_bloque": [len(b) for b in ventana.bloques]},
                   clase=clave_de_clase(rara), valor=conteo.get(rara, 0), opciones=MINIMO_PLIEGUES, detalle=detalle)
