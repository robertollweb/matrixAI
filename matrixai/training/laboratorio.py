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
pedido, lo aplicado y el motivo (en es y en).

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


@dataclass(frozen=True)
class CapturaSpec:
    """Lo que se le pide al entrenador que guarde de lo que aprende la red.

    Los topes son parte de la petición: la demo pasa los suyos (`captura_de_demo`)
    y lo que no quepa sale en `omitido`, nunca recortado en silencio.

    - `gradientes`: norma del gradiente por capa y por época.
    - `pesos_finales`: los de la MEJOR época (los mismos de «Probar una fila»).
    - `max_epocas_muestra`: épocas con pesos (siempre la primera, la mejor y la última).
    - `max_filas`: filas de validación YA preparadas que acompañan al bloque.
    - `max_numeros`: tope de números de pesos (finales + muestra) por ejecución.
    - `max_params_muestra`: por encima de esta red, no hay pesos por época.
    - `max_params_finales`: por encima de esta red, ni siquiera pesos finales.
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
    max_params_finales: int = 500_000
    max_lotes_grad_por_epoca: int = 8
    columnas: tuple[str, ...] | None = None
    etiquetas: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for nombre in ("max_epocas_muestra", "max_filas", "max_numeros",
                       "max_params_muestra", "max_params_finales"):
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
            "max_lotes_grad_por_epoca": self.max_lotes_grad_por_epoca,
        }


def captura_de_demo() -> CapturaSpec:
    """Los topes de la demo (plan 133-C2): <= 8 épocas muestreadas, <= 20.000
    números y ningún peso por encima de 20.000 parámetros."""
    return CapturaSpec(max_epocas_muestra=8, max_numeros=20_000,
                       max_params_muestra=20_000, max_params_finales=20_000)


# ---------------------------------------------------------------------------
# Textos (es / en) — lo que redacta el core se traduce en el core.
# ---------------------------------------------------------------------------

def _omitido(que: str, pedido: Any, aplicado: Any, es: str, en: str) -> dict[str, Any]:
    return {"que": que, "pedido": pedido, "aplicado": aplicado,
            "motivo": {"es": es, "en": en}}


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
            n_filas_validacion: int, ancho_fila: int) -> dict[str, Any]:
    """Decide qué se guarda y qué se omite, solo con tamaños.

    Devuelve `{pesos_finales, n_muestra, gradientes, n_filas, omitido}`; `n_muestra`
    es el número MÁXIMO de épocas con pesos (primera + mejor + última incluidas).
    Función pura: lo mismo para los dos entrenadores, y es lo que prueba
    `test_tope_de_tamano_se_declara`."""
    omitido: list[dict[str, Any]] = []

    pesos_finales = spec.pesos_finales
    if pesos_finales and n_params > spec.max_params_finales:
        pesos_finales = False
        omitido.append(_omitido(
            "pesos_finales", f"{n_params} parámetros", 0,
            f"La red tiene {n_params} parámetros, más que el tope de {spec.max_params_finales} "
            f"para guardar pesos: solo se guardan las normas de gradiente.",
            f"The network has {n_params} parameters, above the {spec.max_params_finales} cap for "
            f"storing weights: only gradient norms are stored."))

    n_muestra = 0
    if spec.max_epocas_muestra > 0 and pesos_finales:
        if n_params > spec.max_params_muestra:
            omitido.append(_omitido(
                "pesos_por_epoca", f"{spec.max_epocas_muestra} épocas", 0,
                f"La red tiene {n_params} parámetros, más que el tope de {spec.max_params_muestra} "
                f"para guardar pesos por época: solo se guardan los finales (los de la mejor época).",
                f"The network has {n_params} parameters, above the {spec.max_params_muestra} cap for "
                f"per-epoch weights: only the final ones (the best epoch's) are stored."))
        else:
            # Presupuesto de números: lo que sobra tras los pesos finales.
            presupuesto = spec.max_numeros - n_params
            cabe = presupuesto // n_params if n_params > 0 and presupuesto > 0 else 0
            pedido = min(spec.max_epocas_muestra, max(1, epocas_pedidas))
            n_muestra = max(0, min(pedido, cabe))
            if n_muestra < pedido:
                omitido.append(_omitido(
                    "pesos_por_epoca", f"{pedido} épocas", n_muestra,
                    f"El tope de {spec.max_numeros} números por ejecución solo da para "
                    f"{n_muestra} épocas con {n_params} parámetros cada una.",
                    f"The {spec.max_numeros}-number cap per run only fits {n_muestra} epochs "
                    f"of {n_params} parameters each."))
    elif spec.max_epocas_muestra == 0 and pesos_finales:
        omitido.append(_omitido(
            "pesos_por_epoca", "0 épocas", 0,
            "No se pidieron pesos por época.", "Per-epoch weights were not requested."))

    n_filas = min(spec.max_filas, n_filas_validacion)
    if n_filas < spec.max_filas:
        omitido.append(_omitido(
            "filas", f"{spec.max_filas} filas", n_filas,
            f"La validación solo tiene {n_filas_validacion} filas.",
            f"The validation set only has {n_filas_validacion} rows."))

    return {"pesos_finales": pesos_finales, "n_muestra": n_muestra,
            "gradientes": spec.gradientes, "n_filas": n_filas, "omitido": omitido}


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
    if n_muestra == 1 or epocas_pedidas == 1:
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
    """~n filas de validación, repartidas a paso fijo (sin azar: la misma muestra
    en cada ejecución), YA preparadas —las que entraron a la red—, con su etiqueta."""
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


def armar_bloque(*, entrenador: str, network: Any, plan: dict[str, Any],
                 finales: list[dict[str, Any]] | None, epoca_final: int | None,
                 muestra: list[tuple[int, list[str], list[dict[str, Any]]]] | None,
                 grad: dict[str, Any] | None, filas: dict[str, Any] | None,
                 spec: CapturaSpec, n_params: int, forma: list[dict[str, Any]],
                 extra_omitido: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Ensambla el bloque `laboratorio`. `forma` ya trae la geometría de las capas."""
    omitido = list(plan["omitido"]) + list(extra_omitido or [])
    bloque: dict[str, Any] = {
        "version": VERSION,
        "disponible": True,
        "entrenador": entrenador,
        "capas": [
            {"indice": f["indice"], "entradas": f["entradas"], "salidas": f["salidas"],
             "activacion": f["activacion"], "clave_W": f["clave_W"], "clave_b": f["clave_b"]}
            for f in forma],
        "parametros": n_params,
        "pesos_finales": (
            {"epoca": epoca_final, "capas": finales} if finales is not None else None),
        "pesos_por_epoca": (
            [{"epoca": e, "motivo": m, "capas": c} for e, m, c in muestra]
            if muestra else None),
        "grad_norm": grad,
        "filas": filas,
        "tope": spec.topes(),
        "omitido": omitido,
    }
    n = 0
    if finales is not None:
        n += _numeros_de_capas(finales)
    for _e, _m, c in (muestra or []):
        n += _numeros_de_capas(c)
    if grad:
        n += sum(len(l["W"]) + len(l["b"]) for l in grad["por_capa"])
    if filas:
        n += sum(len(r) for r in filas["x"]) + sum(len(r) for r in filas["y"])
    bloque["numeros"] = n
    return bloque


def bloque_no_disponible(motivo_es: str, motivo_en: str, entrenador: str | None = None
                         ) -> dict[str, Any]:
    """El bloque de quien NO trae laboratorio (TabM, transformer…): lo dice, sin pesos."""
    return {
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


def resumen_del_bloque(bloque: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lo único del laboratorio que viaja en el estado público (~200-400 B)."""
    if not bloque:
        return None
    if not bloque.get("disponible"):
        return {"version": bloque.get("version", VERSION), "disponible": False,
                "motivo": bloque.get("motivo")}
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
    plan = planear(spec, n_params, len(capas), epocas_pedidas, len(val_ex), 0)
    plan["omitido"] = [o for o in plan["omitido"] if o["que"] != "pesos_por_epoca"]
    plan["omitido"].append(_omitido(
        "pesos_por_epoca", f"{spec.max_epocas_muestra} épocas", 0,
        "Esta ejecución se entrenó sin torch: ese entrenador no guarda los pesos de cada época, "
        "solo los finales.",
        "This run was trained without torch: that trainer does not keep each epoch's weights, "
        "only the final ones."))
    if spec.gradientes:
        plan["omitido"].append(_omitido(
            "grad_norm", "gradientes por capa y época", 0,
            "Esta ejecución se entrenó sin torch: los gradientes por capa solo se miden con el "
            "entrenador de torch.",
            "This run was trained without torch: per-layer gradients are only measured by the "
            "torch trainer."))
    finales = capas if plan["pesos_finales"] else None
    forma = capas_de(network, [(len(c["W"]), len(c["W"][0])) for c in capas])
    return armar_bloque(
        entrenador="stdlib", network=network, plan=plan, finales=finales, epoca_final=epoca,
        muestra=None, grad=None, filas=filas_de_validacion(val_ex, plan["n_filas"], loss_fn, spec),
        spec=spec, n_params=n_params, forma=forma)
