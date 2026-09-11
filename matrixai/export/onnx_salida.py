# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Qué SIGNIFICA lo que devuelve un modelo ONNX — **en un solo sitio**.

Es el hermano de `matrixai.onnx_entrada`: aquél decide cómo se ALIMENTA un
modelo ajeno, y éste cómo se LEE lo que contesta.

POR QUÉ EXISTE (contrato 102-C0, medido el 2026-09-05). `matrixai attest`
tomaba **la primera salida** del grafo y la umbralizaba a 0,5. Los conversores
estándar —skl2onnx, onnxmltools— ponen `label` primero, así que:

* con un modelo **binario** cuadraba **por accidente**: la etiqueta ya es 0 o 1
  y umbralizarla a 0,5 devuelve la misma etiqueta;
* con un modelo de **tres clases** no. Reproducido con una regresión logística
  sobre iris: la exactitud real es **0,9733** y `attest` atestiguaba **0,6667**,
  porque la clase 2 pasaba por el umbral y salía 1. Un número falso, atado a
  los digests del modelo y de los datos, y **firmable**.

LAS TRES REGLAS QUE SE SIGUEN AQUÍ, y son del contrato:

1. **La salida se elige por lo que declara** —tarea, nombre, tipo y semántica
   del grafo—, nunca por su posición.
2. **`argmax` devuelve una etiqueta REAL, no un índice supuesto**: el índice
   solo es una etiqueta si alguien lo dice (las claves de un ZipMap, unas
   clases declaradas, o unas etiquetas observadas que resultan ser `0..K-1`) —
   y de dónde salió queda escrito en el recibo.
3. **Un escalar puede ser una regresión, una puntuación o una probabilidad**, y
   no se distingue por la forma. Si el grafo lo dice —lo produce un `Sigmoid`,
   un `Softmax`— se usa eso; si no lo dice nadie, **se pide el mapa de salida**
   en vez de adivinar. Adivinar aquí es lo que produjo el 0,6667.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "SalidaOnnxAmbigua",
    "SEMANTICAS",
    "SalidaElegida",
    "elegir_salida",
    "resolver_clases",
    "etiqueta_de",
    "valor_de",
    "misma_etiqueta",
]

#: Lo que puede significar una salida. La lista es CERRADA: una semántica que no
#: está no se interpreta a medias.
SEMANTICAS = ("etiqueta", "probabilidades", "probabilidad_positiva", "valor")

_TIPOS_DE_ETIQUETA = ("tensor(int64)", "tensor(int32)", "tensor(string)")
_TIPOS_FLOTANTES = ("tensor(float)", "tensor(double)")

#: Qué operador del grafo declara qué. Solo estos tres: los demás no dicen
#: nada sobre la escala de lo que sale, y suponerlo es el defecto que esto
#: viene a cerrar. `LogSoftmax` queda fuera A PROPÓSITO: su salida ordena igual
#: pero **no son probabilidades**, y llamarlas así sería afirmar de más.
_OPERADOR_DECLARA = {
    "Sigmoid": "probabilidad_positiva",
    "Softmax": "probabilidades",
    "ArgMax": "etiqueta",
}


class SalidaOnnxAmbigua(ValueError):
    """No se sabe qué significa lo que devuelve el modelo, y se dice qué falta."""


@dataclass(frozen=True)
class SalidaElegida:
    """La salida que se va a leer, y **por qué** ésa.

    El `motivo` no es decorativo: va al recibo. Quien lo lea tiene que poder
    saber si el número salió de la predicción del propio modelo o de un
    `argmax` que hizo MatrixAI, sin abrir el ONNX.
    """

    nombre: str
    indice: int
    semantica: str
    tipo_onnx: str
    motivo: str


def _ancho(meta: Any) -> int | None:
    """Cuántos valores devuelve por fila, o `None` si no se puede saber.

    El primer eje es el LOTE, igual que en las entradas. Una salida de rango 1
    (`["N"]`) es **un** valor por fila; `["N", 3]` son tres.
    """
    forma = list(getattr(meta, "shape", []) or [])
    if len(forma) < 2:
        return 1 if forma else None
    ultimo = forma[-1]
    return ultimo if isinstance(ultimo, int) else None


def _es_zipmap(meta: Any) -> bool:
    return str(getattr(meta, "type", "") or "").startswith("seq(map")


def _semantica_del_grafo(ruta_modelo: Any, nombre: str) -> str | None:
    """Lo que el GRAFO declara sobre esa salida, o `None` si no declara nada.

    Esto no es adivinar por la forma: es leer el operador que la produce. Un
    `Sigmoid` dice «probabilidad de la clase positiva» y un `Softmax` dice
    «probabilidades»; un `MatMul` no dice nada, y entonces no se supone nada.

    Si `onnx` no está instalado se devuelve `None` —no se puede leer el grafo—
    y el camino ambiguo pedirá el mapa de salida. Degradar avisando, no
    degradar en silencio.
    """
    if ruta_modelo is None:
        return None
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        return None
    try:
        modelo = onnx.load(str(ruta_modelo))
    except Exception:  # noqa: BLE001 — un grafo ilegible no es un fallo de esto
        return None
    for nodo in modelo.graph.node:
        if nombre in list(nodo.output):
            return _OPERADOR_DECLARA.get(nodo.op_type)
    return None


def _salidas(sesion: Any) -> list[tuple[int, Any]]:
    return list(enumerate(sesion.get_outputs()))


def _por_nombre(sesion: Any, nombre: str) -> tuple[int, Any]:
    for i, meta in _salidas(sesion):
        if meta.name == nombre:
            return i, meta
    disponibles = [m.name for _, m in _salidas(sesion)]
    raise SalidaOnnxAmbigua(
        f"el modelo no tiene una salida llamada {nombre!r}; declara {disponibles}")


def elegir_salida(
    sesion: Any,
    *,
    tarea: str,
    ruta_modelo: Any = None,
    nombre: str | None = None,
    semantica: str | None = None,
) -> SalidaElegida:
    """Qué salida se lee y qué significa.

    `tarea` es `"clasificacion"` o `"regresion"`. `nombre` y `semantica` son el
    **mapa de salida** que se pide cuando el modelo no declara lo bastante:
    quien los pasa está diciendo qué es qué, y eso queda en el recibo.
    """
    if semantica is not None and semantica not in SEMANTICAS:
        raise SalidaOnnxAmbigua(
            f"no sé qué es una salida {semantica!r}; las que sé leer son "
            f"{list(SEMANTICAS)}")

    if nombre is not None:
        indice, meta = _por_nombre(sesion, nombre)
        elegida_semantica = semantica or _deducir(meta, ruta_modelo, tarea)
        if elegida_semantica is None:
            raise SalidaOnnxAmbigua(
                f"la salida {nombre!r} es {meta.type} y ni el grafo ni nadie dice qué "
                f"significa: declara también qué es, con una de {list(SEMANTICAS)}")
        return SalidaElegida(
            nombre=meta.name, indice=indice, semantica=elegida_semantica,
            tipo_onnx=str(meta.type),
            motivo=(f"la salida y su significado los declaró quien atestigua"
                    if semantica else
                    f"la salida la declaró quien atestigua; el significado, el grafo"))

    if tarea == "regresion":
        return _elegir_para_regresion(sesion, ruta_modelo, semantica)
    return _elegir_para_clasificacion(sesion, ruta_modelo, semantica)


def _deducir(meta: Any, ruta_modelo: Any, tarea: str) -> str | None:
    """Qué significa esta salida, mirando tipo y grafo. `None` si no se sabe."""
    tipo = str(getattr(meta, "type", "") or "")
    if _es_zipmap(meta):
        return "probabilidades"
    if tipo in _TIPOS_DE_ETIQUETA:
        return "etiqueta"
    if tipo in _TIPOS_FLOTANTES:
        del_grafo = _semantica_del_grafo(ruta_modelo, meta.name)
        if del_grafo is not None:
            return del_grafo
        if tarea == "regresion":
            return "valor" if (_ancho(meta) or 1) == 1 else None
        ancho = _ancho(meta)
        return "probabilidades" if (ancho is not None and ancho > 1) else None
    return None


def _elegir_para_clasificacion(
        sesion: Any, ruta_modelo: Any, semantica: str | None) -> SalidaElegida:
    salidas = _salidas(sesion)
    etiquetas = [(i, m) for i, m in salidas
                 if str(getattr(m, "type", "")) in _TIPOS_DE_ETIQUETA]
    zipmaps = [(i, m) for i, m in salidas if _es_zipmap(m)]
    flotantes = [(i, m) for i, m in salidas
                 if str(getattr(m, "type", "")) in _TIPOS_FLOTANTES]

    # 1. LA PREDICCIÓN DEL PROPIO MODELO, si la declara. Es la que coincide con
    #    su inferencia nativa: si el modelo decide con otro umbral que 0,5, o
    #    con un orden de clases suyo, esa decisión ya está tomada aquí y no hay
    #    que reconstruirla —ni equivocarse reconstruyéndola—.
    #
    #    Auditoría externa 2026-09-09: con DOS salidas de tipo etiqueta, esto
    #    elegía `etiquetas[0]` sin comprobar que fuera la única — el ORDEN
    #    decidía, exactamente lo que la regla 1 del módulo (más abajo) dice
    #    que nunca decide. Mismo criterio que YA aplica
    #    `_elegir_para_regresion()` con varias salidas numéricas: si hay más
    #    de una candidata igual de válida, se pide el mapa en vez de adivinar.
    if etiquetas and semantica in (None, "etiqueta"):
        if len(etiquetas) == 1:
            i, meta = etiquetas[0]
            return SalidaElegida(
                nombre=meta.name, indice=i, semantica="etiqueta", tipo_onnx=str(meta.type),
                motivo=(f"el modelo declara la salida {meta.name!r} ({meta.type}) con su "
                        "propia predicción: se usa ésa, que es su inferencia nativa"))
        disponibles = [(m.name, str(m.type)) for _, m in etiquetas]
        raise SalidaOnnxAmbigua(
            f"el modelo declara {len(etiquetas)} salidas de tipo etiqueta "
            f"({disponibles}) y ninguna posición es más 'la predicción' que otra: "
            f"leer la primera sería adivinar por orden, no por declaración. "
            f"Declara el mapa de salida: qué salida se lee y qué significa "
            f"(una de {list(SEMANTICAS)})")

    # 2. Un ZipMap trae las clases DENTRO: cada fila es {clase: probabilidad}.
    #    Misma regla que el punto 1: más de un ZipMap es ambigüedad, no un
    #    orden a seguir.
    if zipmaps and semantica in (None, "probabilidades"):
        if len(zipmaps) == 1:
            i, meta = zipmaps[0]
            return SalidaElegida(
                nombre=meta.name, indice=i, semantica="probabilidades",
                tipo_onnx=str(meta.type),
                motivo=(f"la salida {meta.name!r} es un mapa por clase ({meta.type}): las "
                        "clases vienen dentro, no hay que suponerlas"))
        disponibles = [(m.name, str(m.type)) for _, m in zipmaps]
        raise SalidaOnnxAmbigua(
            f"el modelo declara {len(zipmaps)} salidas ZipMap ({disponibles}) y "
            f"ninguna posición es más 'la predicción' que otra: leer la primera "
            f"sería adivinar por orden, no por declaración. Declara el mapa de "
            f"salida: qué salida se lee y qué significa (una de {list(SEMANTICAS)})")

    # 3. Un tensor de flotantes: solo si el grafo o quien atestigua dicen qué es.
    for i, meta in flotantes:
        elegida = semantica or _deducir(meta, ruta_modelo, "clasificacion")
        if elegida is None:
            continue
        del_grafo = _semantica_del_grafo(ruta_modelo, meta.name)
        if semantica:
            motivo = "el significado de la salida lo declaró quien atestigua"
        elif del_grafo:
            motivo = (f"lo declara el grafo: la salida {meta.name!r} la produce un nodo "
                      f"que significa {elegida!r}")
        else:
            motivo = (f"la salida {meta.name!r} devuelve {_ancho(meta)} valores por fila: "
                      "se leen como probabilidades por clase")
        return SalidaElegida(nombre=meta.name, indice=i, semantica=elegida,
                             tipo_onnx=str(meta.type), motivo=motivo)

    disponibles = [(m.name, str(m.type)) for _, m in salidas]
    raise SalidaOnnxAmbigua(
        "no se puede saber qué de lo que devuelve el modelo es su predicción: sus "
        f"salidas son {disponibles} y ninguna lo declara. Un escalar puede ser una "
        "regresión, una puntuación o una probabilidad, y umbralizarlo a 0,5 sin "
        "saberlo es inventarse el resultado. Declara el mapa de salida: qué salida "
        f"se lee y qué significa (una de {list(SEMANTICAS)})")


def _elegir_para_regresion(
        sesion: Any, ruta_modelo: Any, semantica: str | None) -> SalidaElegida:
    flotantes = [(i, m) for i, m in _salidas(sesion)
                 if str(getattr(m, "type", "")) in _TIPOS_FLOTANTES]
    if not flotantes:
        disponibles = [(m.name, str(m.type)) for _, m in _salidas(sesion)]
        raise SalidaOnnxAmbigua(
            f"para medir una regresión hace falta una salida numérica y el modelo "
            f"declara {disponibles}")
    de_un_valor = [(i, m) for i, m in flotantes if (_ancho(m) or 1) == 1]
    if len(de_un_valor) == 1:
        i, meta = de_un_valor[0]
        return SalidaElegida(
            nombre=meta.name, indice=i, semantica=semantica or "valor",
            tipo_onnx=str(meta.type),
            motivo=(f"la salida {meta.name!r} es la única numérica de un valor por "
                    "fila: es el valor que se mide"))
    candidatas = [m.name for _, m in (de_un_valor or flotantes)]
    raise SalidaOnnxAmbigua(
        f"el modelo declara {len(candidatas)} salidas numéricas ({candidatas}) y no se "
        "sabe cuál es el valor predicho: declara el mapa de salida con el nombre de la "
        "que se mide")


def resolver_clases(
    elegida: SalidaElegida,
    *,
    clases_declaradas: list | None,
    etiquetas_observadas: list,
    ancho: int | None,
) -> tuple[list | None, str]:
    """Qué etiqueta hay detrás de cada posición, y **de dónde se sabe**.

    Devuelve `(clases, origen)`. El origen va al recibo: no es lo mismo que las
    clases las declare quien atestigua, que vengan dentro del modelo, o que se
    hayan deducido de la columna objetivo — y quien lea el número tiene derecho
    a saber cuál de las tres.

    Cuando no se puede saber, se levanta: **un índice no es una etiqueta**
    mientras nadie lo diga.
    """
    if elegida.semantica in ("etiqueta", "valor"):
        # La etiqueta la da el modelo; no hay posiciones que traducir.
        return (list(clases_declaradas) if clases_declaradas else None,
                "declaradas" if clases_declaradas else "las da el modelo")

    if clases_declaradas:
        # `probabilidad_positiva` es UNA columna que vale por DOS clases: la
        # positiva es el valor y la negativa es su complemento. Comparar contra
        # el ancho aquí dejaba sin salida el binario más común que existe —un
        # sigmoide `[N,1]` con etiquetas de texto—, y además de forma cruel: sin
        # `--classes` el error pedía «declara cuál es la negativa y cuál la
        # positiva», y al declararlas respondía «se declararon 2 clases y el
        # modelo devuelve 1 valores por fila». El mensaje pedía exactamente lo
        # que luego rechazaba. Encontrado por la auditoría propia del
        # 2026-09-11; lo que se exige aquí son DOS, ni una ni tres.
        if elegida.semantica == "probabilidad_positiva":
            if len(clases_declaradas) != 2:
                raise SalidaOnnxAmbigua(
                    f"la salida es la probabilidad de la clase positiva (una columna) y se "
                    f"declararon {len(clases_declaradas)} clases: hacen falta exactamente dos, "
                    f"la negativa primero y la positiva después")
            return list(clases_declaradas), "declaradas"
        if ancho is not None and len(clases_declaradas) != ancho:
            raise SalidaOnnxAmbigua(
                f"se declararon {len(clases_declaradas)} clases y el modelo devuelve "
                f"{ancho} valores por fila: no se recorta ni se rellena")
        return list(clases_declaradas), "declaradas"

    if elegida.semantica == "probabilidad_positiva":
        observadas = _distintas(etiquetas_observadas)
        if len(observadas) <= 2 and all(_como_numero(v) in (0.0, 1.0) for v in observadas):
            return [0, 1], "deducidas de la columna objetivo (binaria 0/1)"
        raise SalidaOnnxAmbigua(
            f"la salida es la probabilidad de la clase positiva y la columna objetivo "
            f"trae {observadas}: declara cuál es la clase negativa y cuál la positiva")

    observadas = _distintas(etiquetas_observadas)
    numeros = [_como_numero(v) for v in observadas]
    if ancho is not None and all(n is not None and float(n).is_integer() for n in numeros):
        enteros = sorted(int(n) for n in numeros)
        if enteros and enteros[0] >= 0 and enteros[-1] < ancho:
            # No es una suposición: se ha comprobado contra los datos y queda
            # escrito de dónde sale.
            return list(range(ancho)), "deducidas de la columna objetivo (0..K-1)"
    raise SalidaOnnxAmbigua(
        f"el modelo devuelve {ancho} valores por fila y la columna objetivo trae "
        f"{observadas}: no se sabe qué clase es cada posición. Declara el orden de las "
        "clases; un índice no es una etiqueta mientras nadie lo diga")


def _distintas(valores: list) -> list:
    vistas: list = []
    for v in valores:
        if v not in vistas:
            vistas.append(v)
        if len(vistas) > 12:
            break
    return vistas


def _como_numero(v: Any) -> float | None:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _fila(bruto: Any) -> Any:
    """La fila 0 de lo que devolvió el modelo, sea tensor, lista o ZipMap."""
    try:
        return bruto.tolist()[0]
    except AttributeError:
        return list(bruto)[0]


def etiqueta_de(bruto: Any, elegida: SalidaElegida, clases: list | None) -> Any:
    """La etiqueta que predice el modelo para esta fila.

    Nunca devuelve un índice: o la etiqueta que dio el modelo, o la clase que
    ocupa esa posición según `clases`.
    """
    valores = _fila(bruto)

    if elegida.semantica == "etiqueta":
        return _escalar(valores)

    if isinstance(valores, dict):
        # ZipMap: las claves SON las clases. No hay índice que traducir.
        return max(valores.items(), key=lambda kv: kv[1])[0]

    if elegida.semantica == "probabilidad_positiva":
        p = float(_escalar(valores))
        negativa, positiva = (clases or [0, 1])[0], (clases or [0, 1])[1]
        return positiva if p >= 0.5 else negativa

    if not isinstance(valores, list):
        raise SalidaOnnxAmbigua(
            f"la salida {elegida.nombre!r} devolvió un solo valor donde se esperaban "
            "probabilidades por clase")
    posicion = max(range(len(valores)), key=lambda i: valores[i])
    if clases is None:
        raise SalidaOnnxAmbigua(
            "no hay clases con las que traducir la posición del máximo a una etiqueta")
    if posicion >= len(clases):
        raise SalidaOnnxAmbigua(
            f"el modelo devolvió {len(valores)} valores y solo hay {len(clases)} clases")
    return clases[posicion]


def valor_de(bruto: Any) -> float:
    """El número que predice el modelo para esta fila."""
    return float(_escalar(_fila(bruto)))


def _escalar(valores: Any) -> Any:
    if isinstance(valores, list):
        if len(valores) != 1:
            raise SalidaOnnxAmbigua(
                f"se esperaba un valor por fila y llegaron {len(valores)}")
        return valores[0]
    return valores


def misma_etiqueta(predicha: Any, esperada: Any) -> bool:
    """¿Son la misma clase?

    Compara por número cuando las dos lo son —`1`, `1.0` y `"1"` son la misma
    clase— y por texto cuando no. Así funcionan las clases **no consecutivas**
    (`3`, `7`, `9`) y las de **texto** (`"setosa"`), que es lo que devuelven los
    conversores cuando el modelo original tenía etiquetas así.
    """
    a, b = _texto(predicha), _texto(esperada)
    na, nb = _como_numero(a), _como_numero(b)
    if na is not None and nb is not None:
        return na == nb
    return a.strip() == b.strip()


def _texto(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
