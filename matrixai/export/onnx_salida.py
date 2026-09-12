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
3. **La FORMA no declara nada**, ni la de un escalar ni la de varias columnas.
   Un escalar puede ser una regresión, una puntuación o una probabilidad; y
   `[N, k]` puede ser una distribución por clase, unos logits o las k salidas de
   un regresor. Si el grafo lo dice —lo produce un `Sigmoid`, un `Softmax`— se
   usa eso; si no lo dice nadie, **se pide el mapa de salida** en vez de
   adivinar. Adivinar aquí es lo que produjo el 0,6667.
4. **La salida elegida tiene que SERVIR para lo que se mide.** Una regresión se
   mide sobre un valor y una clasificación sobre una clase; leer la salida de
   etiquetas «como si fuera» el valor de una regresión daba un MAE entre códigos
   de categoría —0,0267 sobre iris, con `kind: "etiqueta"` en el recibo—, que es
   un número que no mide ninguna distancia. Se comprueba en `elegir_salida()`,
   que es por donde pasan las tres vías.
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
    "texto_de_etiqueta",
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


#: QUÉ SEMÁNTICAS PUEDE LEER CADA TAREA. Medido el 2026-09-12 sobre el caso que
#: encontró la auditoría propia del 09-11: `--metric mae --output-name label`
#: sobre el iris de tres clases devolvía **`mae: 0,0267`** con `kind: "etiqueta"`
#: en el recibo — un error medio entre CÓDIGOS de clase (0, 1, 2), que no mide
#: ninguna distancia porque entre «setosa» y «virginica» no hay distancia. El
#: criterio literal del corte dice que forzar una salida incorrecta TIENE que
#: fallar, y forzarla por nombre no fallaba: `elegir_salida()` comprobaba que la
#: salida existiera y que alguien dijera qué significa, pero no que eso que
#: significa sirviera para lo que se está midiendo.
_SEMANTICAS_DE_LA_TAREA = {
    "regresion": ("valor",),
    "clasificacion": ("etiqueta", "probabilidades", "probabilidad_positiva"),
}

#: Cómo se nombra cada tarea al explicar el rechazo, y POR QUÉ no encaja. El
#: mensaje tiene que decir qué pasó, no solo que no se puede.
_POR_QUE_NO_ENCAJA = {
    "regresion": ("lo que se mide es una regresión: un error medio entre etiquetas de "
                  "clase no mide ninguna distancia, porque entre dos categorías no la "
                  "hay"),
    "clasificacion": ("lo que se mide es una clasificación: un valor continuo no es "
                      "una clase, y redondearlo a una sería inventarse la predicción"),
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
        elegida = SalidaElegida(
            nombre=meta.name, indice=indice, semantica=elegida_semantica,
            tipo_onnx=str(meta.type),
            motivo=(f"la salida y su significado los declaró quien atestigua"
                    if semantica else
                    f"la salida la declaró quien atestigua; el significado, el grafo"))
    elif tarea == "regresion":
        elegida = _elegir_para_regresion(sesion, ruta_modelo, semantica)
    else:
        elegida = _elegir_para_clasificacion(sesion, ruta_modelo, semantica)

    # LA SALIDA TIENE QUE SERVIR PARA LO QUE SE MIDE, y esto se comprueba aquí
    # —en el ÚNICO sitio por el que pasan las tres vías— y no dentro de cada
    # una: la vía del `nombre` no lo comprobaba, y por ahí entraba el 0,0267.
    _exigir_que_sirva_para_la_tarea(elegida, tarea)
    return elegida


def _exigir_que_sirva_para_la_tarea(elegida: SalidaElegida, tarea: str) -> None:
    """Rechaza leer una salida que no puede sostener la medida que se pide.

    La tarea desconocida se trata como clasificación, igual que hace
    `elegir_salida()` al repartir: dos criterios distintos para el mismo `else`
    acabarían divergiendo.
    """
    cual = "regresion" if tarea == "regresion" else "clasificacion"
    permitidas = _SEMANTICAS_DE_LA_TAREA[cual]
    if elegida.semantica in permitidas:
        return
    raise SalidaOnnxAmbigua(
        f"la salida {elegida.nombre!r} significa {elegida.semantica!r} y "
        f"{_POR_QUE_NO_ENCAJA[cual]}. Para esta medida la salida tiene que "
        f"significar una de {list(permitidas)}: declara otra salida, o mide otra cosa")


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
        # LA ANCHURA NO DECLARA NADA, y esto era la última inferencia por forma
        # que quedaba viva (auditoría propia 2026-09-11; reproducido el 09-12).
        # Un `[N, 2]` flotante salido de un `MatMul` se leía como
        # `probabilidades` **solo porque devolvía dos columnas**: con un regresor
        # de dos salidas, `attest` medía su exactitud y respondía un número sin
        # quejarse, y el recibo declaraba `kind: "probabilidades"` sobre algo que
        # nadie había dicho que lo fuera. El argmax salía bien por invariancia
        # cuando de verdad eran logits, pero el recibo AFIRMABA de más — y con un
        # regresor no salía bien nada. Ahora: si ningún operador lo declara y
        # nadie lo dice, se pide el mapa de salida.
        return None
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
        # Solo hay DOS motivos posibles, y los dos son una declaración: o la
        # trae quien atestigua, o la trae el grafo. El tercero que había aquí
        # —«devuelve N valores por fila: se leen como probabilidades»— era la
        # inferencia por forma que `_deducir` acaba de dejar sin caso.
        del_grafo = _semantica_del_grafo(ruta_modelo, meta.name)
        motivo = ("el significado de la salida lo declaró quien atestigua" if semantica
                  else (f"lo declara el grafo: la salida {meta.name!r} la produce un nodo "
                        f"que significa {elegida!r}"))
        return SalidaElegida(nombre=meta.name, indice=i, semantica=elegida,
                             tipo_onnx=str(meta.type), motivo=motivo)

    disponibles = [(m.name, str(m.type)) for _, m in salidas]
    raise SalidaOnnxAmbigua(
        "no se puede saber qué de lo que devuelve el modelo es su predicción: sus "
        f"salidas son {disponibles} y ninguna lo declara. Un escalar puede ser una "
        "regresión, una puntuación o una probabilidad, y umbralizarlo a 0,5 sin "
        "saberlo es inventarse el resultado; varias columnas pueden ser "
        "probabilidades, logits o las salidas de un regresor, y la anchura no "
        "distingue entre las tres. Declara el mapa de salida: qué salida se lee y "
        f"qué significa (una de {list(SEMANTICAS)})")


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

    Compara por número cuando las dos están escritas en forma CANÓNICA de
    número —`1`, `1.0` y `"1"` son la misma clase— y por texto en cuanto una
    de las dos no lo está. Así funcionan las clases **no consecutivas**
    (`3`, `7`, `9`) y las de **texto** (`"setosa"`), que es lo que devuelven
    los conversores cuando el modelo original tenía etiquetas así.

    **Por qué la forma canónica y no «parsea como número»** (auditoría
    2026-09-11): con la regla anterior, `"01"` y `"1"` eran la misma clase, y
    también `"007"` y `"7"`. En un dataset de códigos de categoría donde el
    cero a la izquierda es significativo —los hay, y muchos— eso hacía que
    `attest` certificara **1,0 donde la exactitud real era 0,5**. Un número
    falso que además se puede firmar, que es exactamente el defecto que el
    corte 102-C0 existía para cerrar, reaparecido por otra puerta.

    Un `"01"` no lo produce ninguna salida numérica de un ONNX: es una forma
    TEXTUAL, y tratarla como número es adivinar que quien la escribió no
    quería decir lo que escribió. La canonicidad se comprueba volviendo a
    escribir el número y viendo si sale el mismo texto: `"1"`→`1`→`"1"` sí;
    `"01"`→`1`→`"1"` no.
    """
    a, b = texto_de_etiqueta(predicha).strip(), texto_de_etiqueta(esperada).strip()
    # Texto idéntico es la misma clase, sin más preguntas. Va ANTES de la vía
    # numérica por un caso real que la vía numérica falla: `"nan"` como
    # etiqueta (una categoría escrita así, o un ausente serializado) daba
    # `False` contra sí mismo, porque `float("nan") != float("nan")` por
    # norma IEEE. Dos filas con la MISMA etiqueta contadas como distintas
    # bajan la exactitud atestiguada sin que nadie lo note. Defecto anterior
    # a esta reparación; se cierra de paso.
    if a == b:
        return True
    if _es_numero_canonico(a) and _es_numero_canonico(b):
        return _como_numero(a) == _como_numero(b)
    return a == b


def _es_numero_canonico(texto: str) -> bool:
    """¿Este texto es la forma en que Python escribiría ese número?

    `"1"` y `"1.0"` sí (`str(1)` / `repr(1.0)`); `"01"`, `"007"`, `"1e0"` y
    `" 1"` no — parsean como número pero nadie los escribe así al convertir
    uno, así que son texto con significado propio."""
    if not texto:
        return False
    try:
        if str(int(texto)) == texto:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return repr(float(texto)) == texto
    except (TypeError, ValueError):
        return False


def texto_de_etiqueta(v: Any) -> str:
    """El TEXTO de una etiqueta, venga como venga del modelo o del CSV.

    Es público porque `attest` construye con él el vocabulario de clases que le
    pasa al registro métrico, y dos maneras de escribir la misma etiqueta —una
    aquí y otra allí— harían que la clase predicha y la esperada dejaran de
    casar sin que nadie lo notase. Los `bytes` salen de `tensor(string)`.
    """
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
