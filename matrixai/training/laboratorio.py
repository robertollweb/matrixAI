# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Contrato 133-C2 — el bloque `laboratorio`: lo MEDIDO de una red densa entrenada.

Aquí vive todo lo que no es torch: la petición (`CapturaSpec`), los topes, la
elección de la muestra de épocas, las filas de validación y la forma del bloque.
Lo comparten el entrenador de torch (`dense_torch_trainer`) y el de sin torch
(`dense_trainer`), para que los dos hablen el mismo idioma y los topes se escriban
UNA vez.

Regla: nada de lo que está aquí se inventa. Cada número del bloque sale del
entrenamiento que ocurrió; lo que se pidió y no se guardó va en `omitido`, con lo
pedido, lo aplicado y el motivo (en es y en). `pedido` y `aplicado` son DATOS (números
en la `unidad` que dice la misma entrada: `epocas`, `parametros`, `filas`); el texto para
personas va solo en `motivo`.

Sin `CapturaSpec` (el valor por omisión de todos los llamantes) no se construye
nada de esto y el entrenamiento es byte a byte el de antes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VERSION = 1

#: Definición de la norma de gradiente que se mide, dicha en el propio bloque para
#: que quien lo pinte no la tenga que adivinar.
DEFINICION_GRADIENTE = {
    "es": ("Norma L2 del gradiente del lote (de la matriz de pesos W y del sesgo b de cada "
           "capa), promediada sobre los lotes medidos de la época —hasta 8, repartidos: uno "
           "de cada `cada_k_lotes`, empezando por el primero—, medida tras la "
           "retropropagación y antes del decaimiento de pesos (weight_decay)."),
    "en": ("L2 norm of the batch gradient (of each layer's weight matrix W and bias b), "
           "averaged over the epoch's measured batches —up to 8, spread out: one in every "
           "`cada_k_lotes`, starting with the first—, measured after backpropagation and "
           "before weight decay."),
}

#: Qué topes rigen cada parte del bloque. `max_numeros` es el TOTAL del bloque y se
#: reparte en este orden: gradientes → pesos finales → filas → pesos por época (lo
#: que no cabe se recorta y va a `omitido`). Viaja en el bloque (`tope_por_parte`).
TOPES_POR_PARTE: dict[str, tuple[str, ...]] = {
    "grad_norm": ("max_epocas_grad", "max_lotes_grad_por_epoca", "max_numeros"),
    "pesos_finales": ("max_params_finales", "max_numeros"),
    "filas": ("max_filas", "max_numeros"),
    "pesos_por_epoca": ("max_epocas_muestra", "max_params_muestra", "max_numeros"),
    "total": ("max_numeros",),
}

#: Qué cuenta como «número» en los topes y en `numeros`: los VALORES medidos (pesos
#: y sesgos, normas de gradiente y los valores x e y de las filas). Las marcas
#: (índices de fila, números de época, motivos) no cuentan.
DEFINICION_NUMEROS = {
    "es": ("Valores medidos: pesos y sesgos (finales y por época), normas de gradiente y "
           "los valores de las filas (entrada y objetivo). Los índices y los números de "
           "época no cuentan."),
    "en": ("Measured values: weights and biases (final and per epoch), gradient norms and "
           "the rows' values (input and target). Row indices and epoch numbers do not "
           "count."),
}


@dataclass(frozen=True)
class CapturaSpec:
    """Lo que se le pide al entrenador que guarde de lo que aprende la red.

    Los topes son parte de la petición: la demo pasa los suyos (`captura_de_demo`)
    y lo que no quepa sale en `omitido`, nunca recortado en silencio. Cada parte
    tiene sus topes (`TOPES_POR_PARTE`) y además TODAS comparten `max_numeros`.

    - `gradientes`: norma del gradiente por capa y por época.
    - `pesos_finales`: los de la MEJOR época (los mismos de «Probar una fila»).
    - `max_epocas_muestra`: épocas con pesos (con 3 o más huecos, siempre la primera, la
      mejor y la última; con menos, la mejor y la última).
    - `max_filas`: filas de validación YA preparadas que acompañan al bloque.
    - `max_numeros`: tope del TOTAL de números del bloque (ver `DEFINICION_NUMEROS`):
      pesos finales + pesos por época + normas de gradiente + filas. Se reparte en
      el orden de `TOPES_POR_PARTE`.
    - `max_params_muestra`: por encima de esta red, no hay pesos por época.
    - `max_params_finales`: por encima de esta red, ni siquiera pesos finales.
    - `max_epocas_grad`: épocas con norma de gradiente guardada. Con más épocas, una
      muestra repartida que incluye la primera, la mejor y la última (las que caben si
      el tope es menor de 3), y se declara.
    - `max_lotes_grad_por_epoca`: lotes de cada época en los que se mide la norma del
      gradiente, repartidos: uno de cada k, k = ⌈lotes / tope⌉ (`lotes_cada`). Medido en
      el 133-C2: medirla en TODOS los lotes cuesta casi lo mismo por lote en cualquier red
      (35–90 µs, según la carga) y subía el entrenamiento un 10–11 % en redes de 50.000 y
      198.146 parámetros; así el coste por época queda acotado.
    - `columnas` / `etiquetas`: nombres para las filas; los pone quien conoce el
      modelo (el entrenador solo ve vectores).
    """

    gradientes: bool = True
    pesos_finales: bool = True
    max_epocas_muestra: int = 12
    max_filas: int = 20
    max_numeros: int = 200_000
    max_params_muestra: int = 50_000
    max_params_finales: int = 200_000
    max_epocas_grad: int = 200
    max_lotes_grad_por_epoca: int = 8
    columnas: tuple[str, ...] | None = None
    etiquetas: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for nombre in ("max_epocas_muestra", "max_filas", "max_numeros",
                       "max_params_muestra", "max_params_finales", "max_epocas_grad"):
            v = getattr(self, nombre)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"CapturaSpec.{nombre} debe ser un entero >= 0 (recibido {v!r})")
        v = self.max_lotes_grad_por_epoca
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(
                f"CapturaSpec.max_lotes_grad_por_epoca debe ser un entero >= 1 (recibido {v!r})")
        for nombre in ("gradientes", "pesos_finales"):
            if not isinstance(getattr(self, nombre), bool):
                raise ValueError(f"CapturaSpec.{nombre} debe ser booleano")
        # Una lista llega de JSON: se congela para que el dataclass siga siendo hashable.
        for nombre in ("columnas", "etiquetas"):
            v = getattr(self, nombre)
            if v is not None and not isinstance(v, tuple):
                object.__setattr__(self, nombre, tuple(str(x) for x in v))

    def con_nombres(self, columnas: Any, etiquetas: Any) -> "CapturaSpec":
        """Copia con los nombres puestos, respetando los que el llamante ya fijó."""
        from dataclasses import replace
        return replace(
            self,
            columnas=self.columnas if self.columnas is not None
            else (tuple(str(c) for c in columnas) if columnas else None),
            etiquetas=self.etiquetas if self.etiquetas is not None
            else (tuple(str(e) for e in etiquetas) if etiquetas else None),
        )

    def topes(self) -> dict[str, int]:
        return {
            "max_epocas_muestra": self.max_epocas_muestra,
            "max_filas": self.max_filas,
            "max_numeros": self.max_numeros,
            "max_params_muestra": self.max_params_muestra,
            "max_params_finales": self.max_params_finales,
            "max_epocas_grad": self.max_epocas_grad,
            "max_lotes_grad_por_epoca": self.max_lotes_grad_por_epoca,
        }


def captura_de_demo() -> CapturaSpec:
    """Los topes de la demo (plan 133-C2): <= 8 épocas muestreadas, <= 20.000
    números EN TOTAL (todo el bloque) y ningún peso por encima de 20.000 parámetros.
    Los gradientes, de las 50 épocas de la demo."""
    return CapturaSpec(max_epocas_muestra=8, max_numeros=20_000,
                       max_params_muestra=20_000, max_params_finales=20_000,
                       max_epocas_grad=50)


# ---------------------------------------------------------------------------
# Textos (es / en) — lo que redacta el core se traduce en el core.
# ---------------------------------------------------------------------------

#: Unidades de `pedido` y `aplicado` en `omitido` (claves estables, no texto).
UNIDADES_OMITIDO = ("epocas", "parametros", "filas")


def _omitido(que: str, pedido: int, aplicado: int, es: str, en: str, *,
             unidad: str, **datos: Any) -> dict[str, Any]:
    """Una entrada de `omitido`. `pedido` y `aplicado` son NÚMEROS en la `unidad` dada
    (`UNIDADES_OMITIDO`); `pedido` 0 = no se pidió. El texto va solo en `motivo` (es/en)."""
    assert unidad in UNIDADES_OMITIDO, unidad
    o = {"que": que, "pedido": int(pedido), "aplicado": int(aplicado), "unidad": unidad,
         "motivo": {"es": es, "en": en}}
    o.update(datos)
    return o


#: Un fallo de la captura se dice con un texto FIJO; el tipo de la excepción va
#: aparte, como dato (`error`), nunca pegado al texto que se pinta.
FALLO_BLOQUE = (
    "No se pudo armar el laboratorio de esta ejecución. El entrenamiento no cambió.",
    "This run's laboratory could not be built. Training was not affected.")
FALLO_GRADIENTE = (
    "La medida del gradiente falló durante el entrenamiento: esta ejecución no guarda "
    "gradientes. El entrenamiento siguió igual, sin ella.",
    "The gradient measurement failed during training: this run keeps no gradients. "
    "Training went on unchanged without it.")
FALLO_PESOS_POR_EPOCA = (
    "No se pudieron guardar los pesos de las épocas durante el entrenamiento: esta "
    "ejecución no guarda sus pesos por época. El entrenamiento siguió igual.",
    "The per-epoch weights could not be stored during training: this run does not keep "
    "its per-epoch weights. Training went on unchanged.")
DE_PARTIDA = (
    "Ninguna época mejoró los pesos con los que empezó este entrenamiento (los que ya "
    "tenía el modelo): los pesos finales son esos, los de partida (época 0).",
    "No epoch improved on the weights this training started from (the ones the model "
    "already had): the final weights are those starting weights (epoch 0).")
#: Re-auditoría del C2 del Studio (18-10, N-3): un entrenamiento que DIVERGE (pérdida de validación no finita en todas las
#: épocas) nunca mejora la de partida, y el entrenador se queda con los pesos de ANTES de entrenar; el bloque los daba como
#: «época 1 · la mejor». Son la época 0, y se dice por qué.
DE_PARTIDA_SIN_MEJORA = (
    "Ninguna época dio una pérdida de validación finita (el entrenamiento divergió): los pesos «finales» son los de "
    "antes de entrenar (época 0), no los de ninguna época entrenada.",
    "No epoch produced a finite validation loss (training diverged): the \"final\" weights are the ones from before "
    "training (epoch 0), not those of any trained epoch.")


def omitido_por_fallo(que: str, pedido: int, textos: tuple[str, str], exc: BaseException,
                      epoca: int | None = None, *, unidad: str = "epocas") -> dict[str, Any]:
    datos: dict[str, Any] = {"error": type(exc).__name__}
    if epoca is not None:
        datos["epoca"] = epoca
    return _omitido(que, pedido, 0, textos[0], textos[1], unidad=unidad, **datos)


def _numeros_de_capas(capas: list[dict[str, Any]]) -> int:
    n = 0
    for c in capas:
        W = c["W"]
        n += (len(W) * len(W[0]) if W else 0) + len(c["b"])
    return n


# ---------------------------------------------------------------------------
# El plan: qué cabe, antes de guardar nada.
# ---------------------------------------------------------------------------

def planear(spec: CapturaSpec, n_params: int, n_capas: int, epocas_pedidas: int,
            n_filas_validacion: int, ancho_fila: int,
            con_gradientes: bool = True) -> dict[str, Any]:
    """Decide qué se guarda y qué se omite, solo con tamaños.

    `max_numeros` es el TOTAL del bloque y se reparte en el orden de
    `TOPES_POR_PARTE`: gradientes (2 números por capa y época guardada) → pesos
    finales (`n_params`) → filas (`ancho_fila` = valores de entrada + objetivo por
    fila) → pesos por época (`n_params` cada una). Con las épocas PEDIDAS: las que
    corran de verdad son las mismas o menos, así que el bloque nunca pasa del tope.

    Devuelve `{pesos_finales, n_muestra, pedido_muestra, gradientes, n_epocas_grad, n_filas,
    omitido}`;
    `n_muestra` es el número MÁXIMO de épocas con pesos (primera + mejor + última
    incluidas) y `n_epocas_grad` el de épocas con gradiente. Función pura: lo mismo
    para los dos entrenadores (`con_gradientes=False` en el de sin torch, que no los
    mide)."""
    omitido: list[dict[str, Any]] = []
    restante = spec.max_numeros
    pedidas = max(1, int(epocas_pedidas))

    # 1. Gradientes: pocos números, y lo primero.
    gradientes = bool(spec.gradientes and con_gradientes)
    n_epocas_grad = 0
    if not spec.gradientes:
        omitido.append(_omitido(
            "grad_norm", 0, 0, "No se pidieron gradientes.", "Gradients were not requested.",
            unidad="epocas"))
    if gradientes:
        por_epoca = 2 * max(1, int(n_capas))
        n_epocas_grad = min(spec.max_epocas_grad, restante // por_epoca)
        if n_epocas_grad <= 0:
            gradientes = False
            n_epocas_grad = 0
            omitido.append(_omitido(
                "grad_norm", pedidas, 0,
                f"Los topes ({spec.max_epocas_grad} épocas, {spec.max_numeros} números en "
                f"total) no dejan guardar ninguna época de gradientes.",
                f"The caps ({spec.max_epocas_grad} epochs, {spec.max_numeros} numbers in "
                f"total) leave room for no epoch of gradients.", unidad="epocas"))
        else:
            restante -= min(pedidas, n_epocas_grad) * por_epoca

    # 2. Pesos finales.
    pesos_finales = spec.pesos_finales
    motivo_sin_finales: tuple[str, str] | None = None
    if not spec.pesos_finales:
        motivo_sin_finales = ("No se pidieron los pesos finales.",
                              "Final weights were not requested.")
        omitido.append(_omitido("pesos_finales", 0, 0, *motivo_sin_finales, unidad="parametros"))
    elif n_params > spec.max_params_finales:
        pesos_finales = False
        motivo_sin_finales = (
            f"La red tiene {n_params} parámetros, más que el tope de {spec.max_params_finales} "
            f"para guardar pesos: no se guarda ninguno de los pesos de esta red.",
            f"The network has {n_params} parameters, above the {spec.max_params_finales} cap for "
            f"storing weights: none of this network's weights are stored.")
        omitido.append(_omitido("pesos_finales", n_params, 0, *motivo_sin_finales,
                                unidad="parametros"))
    elif n_params > restante:
        pesos_finales = False
        motivo_sin_finales = (
            f"Los pesos finales ({n_params} números) no caben en el tope de {spec.max_numeros} "
            f"números del bloque.",
            f"The final weights ({n_params} numbers) do not fit the block's "
            f"{spec.max_numeros}-number cap.")
        omitido.append(_omitido("pesos_finales", n_params, 0, *motivo_sin_finales,
                                unidad="parametros"))
    else:
        restante -= n_params

    # 3. Filas de validación.
    ancho = max(1, int(ancho_fila))
    n_filas = min(spec.max_filas, n_filas_validacion)
    if spec.max_filas == 0:
        omitido.append(_omitido(
            "filas", 0, 0, "No se pidieron filas de validación.",
            "No validation rows were requested.", unidad="filas"))
    elif n_filas * ancho > restante:
        cabe = max(0, restante // ancho)
        omitido.append(_omitido(
            "filas", n_filas, cabe,
            f"Las filas ({ancho} números cada una) no caben en el tope de {spec.max_numeros} "
            f"números del bloque: se guardan {cabe}.",
            f"The rows ({ancho} numbers each) do not fit the block's {spec.max_numeros}-number "
            f"cap: {cabe} are stored.", unidad="filas"))
        n_filas = cabe
    elif n_filas < spec.max_filas:
        omitido.append(_omitido(
            "filas", spec.max_filas, n_filas,
            f"La validación solo tiene {n_filas_validacion} filas.",
            f"The validation set only has {n_filas_validacion} rows.", unidad="filas"))
    restante -= n_filas * ancho

    # 4. Pesos por época: con lo que sobra.
    n_muestra = 0
    pedido = min(spec.max_epocas_muestra, pedidas)
    if spec.max_epocas_muestra == 0:
        omitido.append(_omitido(
            "pesos_por_epoca", 0, 0,
            "No se pidieron pesos por época.", "Per-epoch weights were not requested.",
            unidad="epocas"))
    elif not pesos_finales:
        es, en = motivo_sin_finales or ("", "")
        omitido.append(_omitido(
            "pesos_por_epoca", pedido, 0,
            f"Sin pesos finales no se guardan los de cada época. {es}",
            f"Without final weights, no per-epoch weights are stored. {en}", unidad="epocas"))
    elif n_params > spec.max_params_muestra:
        omitido.append(_omitido(
            "pesos_por_epoca", pedido, 0,
            f"La red tiene {n_params} parámetros, más que el tope de {spec.max_params_muestra} "
            f"para guardar pesos por época: solo se guardan los finales (los de la mejor época).",
            f"The network has {n_params} parameters, above the {spec.max_params_muestra} cap for "
            f"per-epoch weights: only the final ones (the best epoch's) are stored.",
            unidad="epocas"))
    else:
        cabe = restante // n_params if n_params > 0 and restante > 0 else 0
        n_muestra = max(0, min(pedido, cabe))
        if n_muestra < pedido:
            omitido.append(_omitido(
                "pesos_por_epoca", pedido, n_muestra,
                f"El tope de {spec.max_numeros} números del bloque solo da para "
                f"{n_muestra} épocas con {n_params} parámetros cada una.",
                f"The block's {spec.max_numeros}-number cap only fits {n_muestra} epochs "
                f"of {n_params} parameters each.", unidad="epocas"))

    return {"pesos_finales": pesos_finales, "n_muestra": n_muestra,
            "pedido_muestra": pedido,
            "gradientes": gradientes, "n_epocas_grad": n_epocas_grad,
            "n_filas": n_filas, "omitido": omitido}


def lotes_cada(n_lotes: int, spec: CapturaSpec) -> int:
    """Cada cuántos lotes se mide la norma del gradiente: k = ⌈lotes / tope⌉, así que en
    una época se miden como mucho `max_lotes_grad_por_epoca` lotes (el 0, el k, el 2k…)."""
    return max(1, -(-int(n_lotes) // spec.max_lotes_grad_por_epoca))


def epocas_programadas(epocas_pedidas: int, n_muestra: int) -> frozenset[int]:
    """Épocas en las que se clona el estado MIENTRAS se entrena: `n_muestra`
    puntos escalonados entre la 1 y la última pedida (la primera siempre). La
    mejor y la última no se saben antes: se añaden al acabar, y `elegir_muestra`
    recorta de forma pareja hasta `n_muestra`. Con menos de 3 huecos no se
    programa nada (no caben la primera, la mejor y la última)."""
    if n_muestra < 3 or epocas_pedidas < 1:
        return frozenset()
    if epocas_pedidas == 1:
        return frozenset({1})
    return frozenset(
        1 + round(k * (epocas_pedidas - 1) / (n_muestra - 1)) for k in range(n_muestra))


def elegir_muestra(disponibles: dict[int, Any], primera: int, mejor: int, ultima: int,
                   n_muestra: int) -> list[tuple[int, list[str]]]:
    """De las épocas con estado guardado, las que van al bloque: SIEMPRE la mejor,
    la última y la primera (si caben en `n_muestra`; en ese orden de prioridad) y
    el resto repartido de forma pareja hasta llenar. Devuelve `[(época, [motivos])]`
    ordenado por época."""
    motivos: dict[int, list[str]] = {}
    for ep, nombre in ((mejor, "mejor"), (ultima, "ultima"), (primera, "primera")):
        if ep in disponibles:
            motivos.setdefault(ep, []).append(nombre)
    elegidas = list(motivos)[:max(0, n_muestra)]
    huecos = n_muestra - len(elegidas)
    resto = sorted(e for e in disponibles if e not in motivos)
    if huecos > 0 and resto:
        if huecos >= len(resto):
            extra = resto
        elif huecos == 1:
            extra = [resto[len(resto) // 2]]
        else:
            extra = sorted({resto[round(k * (len(resto) - 1) / (huecos - 1))]
                            for k in range(huecos)})
        for e in extra:
            motivos[e] = ["muestra"]
            elegidas.append(e)
    return [(e, motivos[e]) for e in sorted(elegidas)]


def muestrear_gradientes(hist: list[tuple[int, Any]], mejor: int, n: int
                         ) -> tuple[list[tuple[int, Any]], dict[str, Any] | None]:
    """Las épocas de gradiente que se guardan: todas si caben en `n`; si no, `n`
    repartidas (`elegir_muestra`) y el `omitido` que lo declara, cuyo texto dice cuáles
    de la primera, la mejor y la última entraron DE VERDAD (con `n` < 3 no caben las
    tres). `hist` = `[(época, valores)]` en orden."""
    if len(hist) <= n:
        return hist, None
    por_epoca = dict(hist)
    elegidas_m = elegir_muestra(por_epoca, hist[0][0], mejor, hist[-1][0], n)
    elegidas = [e for e, _m in elegidas_m]
    es_n = {"primera": "la primera", "mejor": "la mejor", "ultima": "la última"}
    en_n = {"primera": "the first", "mejor": "the best", "ultima": "the last"}
    orden = [k for k in ("primera", "mejor", "ultima")
             if any(k in m for _e, m in elegidas_m)]
    if not orden:
        ent_es, ent_en = "", ""
    else:
        def _y(nombres: list[str], y: str) -> str:
            return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + f" {y} " + nombres[-1]
        ent_es = f" (entre ellas, {_y([es_n[k] for k in orden], 'y')})"
        ent_en = f" (among them, {_y([en_n[k] for k in orden], 'and')})"
    return [(e, por_epoca[e]) for e in elegidas], _omitido(
        "grad_norm", len(hist), len(elegidas),
        f"Se midió el gradiente en {len(hist)} épocas; los topes dejan guardar {n}, "
        f"repartidas{ent_es}.",
        f"The gradient was measured in {len(hist)} epochs; the caps allow storing {n}, "
        f"spread out{ent_en}.", unidad="epocas")


# ---------------------------------------------------------------------------
# Filas de validación
# ---------------------------------------------------------------------------

def etiqueta_de(y: list[float], loss_fn: str, etiquetas: tuple[str, ...] | None) -> str | None:
    """La etiqueta (texto) de una fila, o None si no hay con qué nombrarla
    (regresión: el valor ya es `y`)."""
    if not etiquetas or loss_fn == "mse":
        return None
    if loss_fn == "cross_entropy" and len(y) == len(etiquetas):
        return etiquetas[max(range(len(y)), key=lambda i: y[i])]
    if loss_fn == "binary_cross_entropy" and len(etiquetas) >= 2 and len(y) == 1:
        return etiquetas[1] if y[0] >= 0.5 else etiquetas[0]
    return None


def filas_de_validacion(val_ex: list[tuple[list[float], list[float]]], n: int,
                        loss_fn: str, spec: CapturaSpec) -> dict[str, Any] | None:
    """~n filas de la validación ENTERA, repartidas a paso fijo (sin azar: la misma
    muestra en cada ejecución), YA preparadas —las que entraron a la red—, con su
    etiqueta. `de` = cuántas filas tiene esa validación."""
    if n <= 0 or not val_ex:
        return None
    total = len(val_ex)
    n = min(n, total)
    indices = sorted({(k * total) // n for k in range(n)})
    x = [[float(v) for v in val_ex[i][0]] for i in indices]
    y = [[float(v) for v in val_ex[i][1]] for i in indices]
    return {
        "preparadas": True,
        "columnas": list(spec.columnas) if spec.columnas else None,
        "etiquetas": list(spec.etiquetas) if spec.etiquetas else None,
        "indices": indices,
        "de": total,
        "x": x,
        "y": y,
        "etiqueta": [etiqueta_de(yy, loss_fn, spec.etiquetas) for yy in y],
    }


def ancho_de_fila(val_ex: list[tuple[list[float], list[float]]]) -> int:
    """Números de una fila (entrada + objetivo), para el presupuesto de `planear`."""
    return (len(val_ex[0][0]) + len(val_ex[0][1])) if val_ex else 0


# ---------------------------------------------------------------------------
# El bloque
# ---------------------------------------------------------------------------

def capas_de(network: Any, formas: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """La geometría de cada capa: `formas[i] = (salidas, entradas)`."""
    out = []
    for layer, (salidas, entradas) in zip(network.layers, formas):
        out.append({
            "indice": int(layer.index),
            "salidas": int(salidas),
            "entradas": int(entradas),
            "activacion": getattr(layer, "activation", None),
            "clave_W": f"{network.name}.W{layer.index}",
            "clave_b": f"{network.name}.b{layer.index}",
        })
    return out


def _reconciliar_muestra(omitido: list[dict[str, Any]], plan: dict[str, Any],
                         muestra: list[Any] | None, epocas_corridas: int | None) -> None:
    """`aplicado` = las épocas con pesos que SE GUARDARON (N-4): el plan solo sabe el máximo.
    Si ya hay un `omitido` de los pesos por época, se corrige su `aplicado`; si no hay y
    la muestra de {mejor, última} (menos de 3 huecos) se quedó corta porque son la misma
    época, se declara."""
    if plan.get("n_muestra", 0) <= 0 or muestra is None:
        return
    guardadas = len(muestra)
    existente = [o for o in omitido if o["que"] == "pesos_por_epoca"]
    if existente:
        for o in existente:
            if o["aplicado"] != 0 and o["aplicado"] != guardadas and not o.get("error"):
                o["aplicado"] = guardadas
        return
    n = plan["n_muestra"]
    if n < 3 and epocas_corridas is not None and guardadas < min(n, epocas_corridas):
        omitido.append(_omitido(
            "pesos_por_epoca", plan.get("pedido_muestra", n), guardadas,
            f"Con menos de 3 huecos para épocas solo se guardan la mejor y la última, y aquí "
            f"son la misma época: se guardó {guardadas}.",
            f"With fewer than 3 slots for epochs only the best and the last are stored, and "
            f"here they are the same epoch: {guardadas} stored.", unidad="epocas"))


def armar_bloque(*, entrenador: str, network: Any, plan: dict[str, Any],
                 finales: list[dict[str, Any]] | None, epoca_final: int | None,
                 muestra: list[tuple[int, list[str], list[dict[str, Any]]]] | None,
                 grad: dict[str, Any] | None, filas: dict[str, Any] | None,
                 spec: CapturaSpec, n_params: int, forma: list[dict[str, Any]],
                 extra_omitido: list[dict[str, Any]] | None = None,
                 parciales: list[int] | None = None,
                 epocas_corridas: int | None = None,
                 motivo_de_partida: tuple[str, str] = DE_PARTIDA) -> dict[str, Any]:
    """Ensambla el bloque `laboratorio`. `forma` ya trae la geometría de las capas.

    `epocas_corridas`: las épocas que corrieron de verdad (solo el entrenador de torch).
    El `aplicado` de los pesos por época es lo GUARDADO de verdad: con menos de 3 huecos
    no se programa ninguna época y la muestra es {mejor, última}, que puede ser una sola.

    `parciales`: épocas que acabaron A MEDIAS (por `plazo`); sus pesos llevan
    `parcial: True` (en la muestra y en los finales), igual que su gradiente."""
    omitido = list(plan["omitido"]) + list(extra_omitido or [])
    _reconciliar_muestra(omitido, plan, muestra, epocas_corridas)
    a_medias = set(parciales or [])
    pesos_finales = None
    if finales is not None:
        pesos_finales = {"epoca": epoca_final, "parcial": epoca_final in a_medias,
                         "capas": finales}
        if epoca_final == 0:
            pesos_finales["de_partida"] = True
            pesos_finales["motivo"] = {"es": motivo_de_partida[0], "en": motivo_de_partida[1]}
    bloque: dict[str, Any] = {
        "version": VERSION,
        "disponible": True,
        "entrenador": entrenador,
        "capas": [
            {"indice": f["indice"], "entradas": f["entradas"], "salidas": f["salidas"],
             "activacion": f["activacion"], "clave_W": f["clave_W"], "clave_b": f["clave_b"]}
            for f in forma],
        "parametros": n_params,
        "pesos_finales": pesos_finales,
        "pesos_por_epoca": (
            [{"epoca": e, "motivo": m, "parcial": e in a_medias, "capas": c}
             for e, m, c in muestra]
            if muestra else None),
        "grad_norm": grad,
        "filas": filas,
        "tope": spec.topes(),
        "tope_por_parte": {k: list(v) for k, v in TOPES_POR_PARTE.items()},
        "omitido": omitido,
    }
    por_parte = {
        "grad_norm": (sum(len(l["W"]) + len(l["b"]) for l in grad["por_capa"]) if grad else 0),
        "pesos_finales": _numeros_de_capas(finales) if finales is not None else 0,
        "filas": ((sum(len(r) for r in filas["x"]) + sum(len(r) for r in filas["y"]))
                  if filas else 0),
        "pesos_por_epoca": sum(_numeros_de_capas(c) for _e, _m, c in (muestra or [])),
    }
    bloque["numeros_por_parte"] = por_parte
    bloque["numeros"] = sum(por_parte.values())
    bloque["definicion_numeros"] = DEFINICION_NUMEROS
    return bloque


def bloque_no_disponible(motivo_es: str, motivo_en: str, entrenador: str | None = None,
                         error: str | None = None) -> dict[str, Any]:
    """El bloque de quien NO trae laboratorio (una red compuesta, un transformer…, o
    una captura que falló): lo dice, sin pesos. `error` = el tipo de la excepción,
    como dato aparte del texto."""
    bloque = {
        "version": VERSION,
        "disponible": False,
        "entrenador": entrenador,
        "motivo": {"es": motivo_es, "en": motivo_en},
        "pesos_finales": None,
        "pesos_por_epoca": None,
        "grad_norm": None,
        "filas": None,
        "omitido": [],
    }
    if error is not None:
        bloque["error"] = error
    return bloque


def bloque_por_fallo(exc: BaseException, entrenador: str | None) -> dict[str, Any]:
    """El bloque de una captura que falló al armarse: texto fijo y el tipo aparte."""
    return bloque_no_disponible(FALLO_BLOQUE[0], FALLO_BLOQUE[1], entrenador,
                                error=type(exc).__name__)


def resumen_del_bloque(bloque: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lo único del laboratorio que viaja en el estado público (~200-400 B)."""
    if not bloque:
        return None
    if not bloque.get("disponible"):
        return {"version": bloque.get("version", VERSION), "disponible": False,
                "entrenador": bloque.get("entrenador"), "motivo": bloque.get("motivo")}
    muestra = bloque.get("pesos_por_epoca") or []
    grad = bloque.get("grad_norm")
    filas = bloque.get("filas")
    return {
        "version": bloque["version"],
        "disponible": True,
        "entrenador": bloque["entrenador"],
        "capas": len(bloque["capas"]),
        "parametros": bloque["parametros"],
        "pesos_finales": bloque["pesos_finales"] is not None,
        "epocas_con_pesos": [m["epoca"] for m in muestra],
        "gradientes": bool(grad),
        "filas": len(filas["x"]) if filas else 0,
        "numeros": bloque["numeros"],
        "omitido": [o["que"] for o in bloque["omitido"]],
    }


def bloque_sin_torch(network: Any, parametros: dict[str, dict[str, Any]], epoca: int,
                     val_ex: list[tuple[list[float], list[float]]], loss_fn: str,
                     spec: CapturaSpec, epocas_pedidas: int) -> dict[str, Any]:
    """El bloque del entrenador SIN torch: solo los pesos finales y las filas.

    Se DECLARA lo que falta y por qué: ese entrenador (`DenseSupervisedTrainer`,
    SGD de lote 1) no guarda pesos por época ni mide el gradiente por capa de forma
    que se pueda comparar con el de torch, así que `pesos_por_epoca` y `grad_norm`
    van a `None` y en `omitido`."""
    capas = []
    for layer in network.layers:
        W = parametros[f"{network.name}.W{layer.index}"]["values"]
        b = parametros[f"{network.name}.b{layer.index}"]["values"]
        capas.append({"W": [[float(v) for v in fila] for fila in W],
                      "b": [float(v) for v in b]})
    n_params = _numeros_de_capas(capas)
    plan = planear(spec, n_params, len(capas), epocas_pedidas, len(val_ex),
                   ancho_de_fila(val_ex), con_gradientes=False)
    plan["n_muestra"] = 0
    if spec.max_epocas_muestra > 0 and plan["pesos_finales"]:
        # Lo que el plan diga de los pesos por época no es el motivo: este entrenador
        # no los guarda nunca.
        plan["omitido"] = [o for o in plan["omitido"] if o["que"] != "pesos_por_epoca"]
        plan["omitido"].append(_omitido(
            "pesos_por_epoca", min(spec.max_epocas_muestra, max(1, epocas_pedidas)), 0,
            "Esta ejecución se entrenó sin torch: ese entrenador no guarda los pesos de cada "
            "época, solo los finales.",
            "This run was trained without torch: that trainer does not keep each epoch's "
            "weights, only the final ones.", unidad="epocas"))
    if spec.gradientes:
        plan["omitido"].append(_omitido(
            "grad_norm", max(1, epocas_pedidas), 0,
            "Esta ejecución se entrenó sin torch: los gradientes por capa solo se miden con el "
            "entrenador de torch.",
            "This run was trained without torch: per-layer gradients are only measured by the "
            "torch trainer.", unidad="epocas"))
    finales = capas if plan["pesos_finales"] else None
    forma = capas_de(network, [(len(c["W"]), len(c["W"][0])) for c in capas])
    return armar_bloque(
        entrenador="stdlib", network=network, plan=plan, finales=finales, epoca_final=epoca,
        muestra=None, grad=None, filas=filas_de_validacion(val_ex, plan["n_filas"], loss_fn, spec),
        spec=spec, n_params=n_params, forma=forma)
