# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Atestiguar la evaluación de un modelo AJENO — contrato 87-C2.

QUÉ HACE. Se le da un modelo que MatrixAI no ha entrenado (un `.onnx`, venga de
donde venga), unos datos de evaluación y qué métrica se quiere; corre la
evaluación, y ata el número a los digests del modelo y de los datos en un
recibo verificable.

QUÉ ATESTIGUA, y esto va también en el propio recibo: **la evaluación, el
entorno y las entradas**. Es decir: «este fichero, sobre estos datos, dio este
número, aquí y ahora».

QUÉ **NO** ATESTIGUA:

* **No es prueba de entrenamiento.** Nada aquí dice de dónde salieron esos
  pesos ni con qué datos se hicieron. Un modelo copiado de otro sitio produce
  exactamente el mismo recibo que uno propio.
* **No dice que el modelo sea bueno.** Dice qué salió al medirlo así.
* **Y su techo de garantía es A1/A2**: se firma como cualquier recibo —hoy con
  HMAC, o sea consistencia y no autenticidad— y no puede llegar más arriba,
  porque los niveles altos hablan del entorno de ejecución y de pruebas
  criptográficas que este camino no da.

Decir las tres cosas es lo que separa un recibo de un adorno.
"""

from __future__ import annotations

from matrixai.onnx_entrada import (
    EntradaOnnxIncompatible, caracteristicas_declaradas, tensor_para)

import csv
import hashlib
import io
import platform
from pathlib import Path
from typing import Any

__all__ = ["AtestacionImposible", "METRICAS", "atestiguar"]

#: Las métricas que este camino sabe medir. La lista es CERRADA: pedir una que
#: no está se contesta con las que hay, en vez de devolver un cero que parece
#: un resultado.
METRICAS = ("accuracy", "mae")


class AtestacionImposible(RuntimeError):
    """No se puede atestiguar, y se dice por qué."""


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _leer_csv(texto: str) -> tuple[list[str], list[list[str]]]:
    filas = list(csv.reader(io.StringIO(texto)))
    if len(filas) < 2:
        raise AtestacionImposible(
            "el CSV no tiene cabecera y al menos una fila: no hay nada que medir")
    # UNA CABECERA VACÍA NO ES UNA CABECERA (2026-08-25). Lo cazó el agente que
    # construía la pantalla, sondeando el endpoint: un CSV cuya primera línea
    # está en blanco reventaba con `IndexError: list index out of range` al
    # buscar `cabecera[-1]`, y quien lo recibía se llevaba un
    # `error_inesperado` en vez de un motivo escrito. Un fallo que se puede
    # nombrar no se deja como traza.
    if not filas[0] or not any(str(c).strip() for c in filas[0]):
        raise AtestacionImposible(
            "la primera línea del CSV está vacía, así que no hay nombres de columna: "
            "sin cabecera no se sabe cuál es el objetivo ni cuáles las entradas")
    return filas[0], filas[1:]


#: El propósito que escribe el core cuando nadie declara uno. Cerrado a los dos
#: idiomas del producto: uno que no esté cae al castellano, que es el de casa.
_PROPOSITO_POR_DEFECTO = {
    "es": "medir {metrica} de un modelo externo",
    "en": "measure {metrica} of an external model",
}


def _proposito_por_defecto(metrica: str, locale: str) -> str:
    idioma = str(locale or "es").strip().lower()
    plantilla = _PROPOSITO_POR_DEFECTO.get(idioma, _PROPOSITO_POR_DEFECTO["es"])
    return plantilla.format(metrica=metrica)


def atestiguar(
    modelo: str | Path,
    datos: str | Path,
    *,
    metrica: str = "accuracy",
    columna: str | None = None,
    proposito: str = "",
    actor: str = "",
    locale: str = "es",
) -> dict[str, Any]:
    """Corre el modelo sobre los datos y devuelve el RECIBO (sin firmar).

    Firmarlo es un acto aparte, como en todo el 81: quien firma decide con qué
    clave, y un recibo sin firmar es A0 y lo dice.
    """
    if metrica not in METRICAS:
        raise AtestacionImposible(
            f"no sé medir {metrica!r}; este camino mide {list(METRICAS)}. Pedir una "
            f"que no está y devolver un cero sería dar un resultado inventado")
    ruta_modelo, ruta_datos = Path(modelo), Path(datos)
    for ruta, que in ((ruta_modelo, "el modelo"), (ruta_datos, "los datos")):
        if not ruta.is_file():
            raise AtestacionImposible(f"{que} no está en {str(ruta)!r}")

    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError as exc:
        raise AtestacionImposible(
            "onnxruntime no está instalado, así que no se puede ejecutar un modelo "
            "ONNX: `pip install \"matrixai-core[export]\"`") from exc

    texto = ruta_datos.read_text(encoding="utf-8")
    cabecera, filas = _leer_csv(texto)
    objetivo = columna or cabecera[-1]
    if objetivo not in cabecera:
        raise AtestacionImposible(
            f"la columna {objetivo!r} no está en el CSV (hay {cabecera})")
    indice = cabecera.index(objetivo)
    entradas_nombres = [c for i, c in enumerate(cabecera) if i != indice]

    sesion = onnxruntime.InferenceSession(str(ruta_modelo),
                                          providers=["CPUExecutionProvider"])
    metas = sesion.get_inputs()
    if len(metas) != 1:
        raise AtestacionImposible(
            f"el modelo declara {len(metas)} entradas y este camino sabe alimentar "
            "una: mapear varias por su cuenta sería adivinar cuál es cuál")
    meta = metas[0]
    # LA FORMA Y EL TIPO, DECIDIDOS EN UN SOLO SITIO (`matrixai.onnx_entrada`),
    # el mismo que usa el ejecutor del motor. La 2ª auditoría externa lo dijo
    # así: «la garantía de C1 no alcanza C2». Este camino tenía su propia copia,
    # y comparaba el número de columnas contra los ejes ENTEROS de la forma: en
    # `[1, "features"]` eso es el `1`, así que rechazaba modelos válidos
    # diciendo que esperaban una entrada.
    esperado_declarado, comprobacion_forma = caracteristicas_declaradas(meta)
    if esperado_declarado is not None and esperado_declarado != len(entradas_nombres):
        raise AtestacionImposible(
            f"el modelo espera {esperado_declarado} entradas y el CSV trae "
            f"{len(entradas_nombres)} "
            f"columnas además de {objetivo!r}: no se recorta ni se rellena")

    aciertos = 0
    error_absoluto = 0.0
    medidas = 0
    for fila in filas:
        if len(fila) != len(cabecera):
            continue
        try:
            vector = [float(v) for i, v in enumerate(fila) if i != indice]
        except ValueError:
            # Una fila que no es numérica no se convierte en un cero: se salta
            # y se cuenta, para que el recibo diga sobre cuántas se midió.
            continue
        # El TIPO que el modelo declara, no `float` siempre: un modelo de
        # `int64` alimentado con `3.7` lo trunca en silencio a `3`, y el acierto
        # que salga de ahí iría al recibo como si se hubiera medido sobre el dato
        # del CSV. Un dato que nadie escribió no se atestigua.
        try:
            tensor = tensor_para(meta, vector, "el modelo que se atestigua")
        except EntradaOnnxIncompatible as exc:
            raise AtestacionImposible(str(exc)) from None
        salida = sesion.run(None, {meta.name: tensor})[0]
        try:
            valores = salida.tolist()[0]
        except AttributeError:
            valores = list(salida)[0]
        esperado = fila[indice]
        medidas += 1
        if metrica == "accuracy":
            # UNA SOLA SALIDA NO ES UNA LISTA DE CLASES (auditoría externa del
            # 2026-08-25, hallazgo 1 — BLOQUEANTE, y reproducido con un
            # clasificador sigmoide: la exactitud correcta con umbral 0,5 era
            # **1,0** y esto atestiguaba **0,5**).
            #
            # `argmax` de un vector de UN elemento es siempre 0, así que el
            # modelo «predecía» siempre la clase 0. Es el mismo defecto que ya
            # se cerró en el evaluador del core, aquí un piso más allá: **el
            # número equivocado acaba atado a los digests y se puede FIRMAR**.
            #
            # Con una salida se usa el umbral, que es lo que significa un
            # sigmoide; con varias, el argmax, que es lo que significa un
            # softmax.
            if isinstance(valores, list) and len(valores) == 1:
                predicho = 1 if float(valores[0]) >= 0.5 else 0
            elif isinstance(valores, list):
                predicho = max(range(len(valores)), key=lambda i: valores[i])
            else:
                predicho = 1 if float(valores) >= 0.5 else 0
            try:
                aciertos += int(predicho == int(float(esperado)))
            except ValueError:
                # La etiqueta no es un número: se compara por posición solo si
                # el CSV declara las clases, y aquí no las declara.
                raise AtestacionImposible(
                    f"la columna {objetivo!r} trae {esperado!r}, que no es un índice de "
                    "clase; para etiquetas por su nombre hace falta declarar el orden "
                    "de las clases, y este camino todavía no lo pide") from None
        else:
            valor = valores[0] if isinstance(valores, list) else valores
            error_absoluto += abs(float(valor) - float(esperado))

    if medidas == 0:
        raise AtestacionImposible(
            "ninguna fila del CSV se pudo medir: no hay resultado que atestiguar")

    valor = (aciertos / medidas) if metrica == "accuracy" else (error_absoluto / medidas)
    digest_modelo, digest_datos = _sha256(ruta_modelo), _sha256(ruta_datos)
    from matrixai.pipelines.canonical import jcs_bytes

    recibo: dict[str, Any] = {
        "schema_version": "1.0",
        "receipt_id": "attest-" + hashlib.sha256(
            (digest_modelo + digest_datos + metrica).encode()).hexdigest()[:16],
        "event_type": "evaluation_attestation",
        # La hora la pone quien firma: aquí no se inventa un reloj, y un recibo
        # sin `created_at` no pasa §14.2 — se rellena al firmar.
        "created_at": None,
        # QUIÉN LO PIDE NO LO SABE ESTE MÓDULO (medido conduciendo la aplicación,
        # 2026-08-26). El respaldo era `"matrixai attest"` —el nombre del comando
        # del terminal—, así que un recibo emitido desde la PANTALLA del Studio
        # decía que lo había pedido el CLI. En un documento cuyo asunto entero es
        # la procedencia, eso no es un detalle: es el documento afirmando algo
        # que no pasó. La propia pantalla lo tenía escrito —«el suyo nombra el
        # terminal, que no es por donde ha entrado esto»—: avisaba del defecto en
        # vez de no tenerlo.
        #
        # Ahora quien llama lo dice (el CLI pone su nombre, el backend el suyo) y,
        # si nadie lo dice, el recibo declara que **no consta** en vez de
        # inventarse un canal.
        # EL PROPÓSITO DE RESPALDO, EN EL IDIOMA QUE SE PIDA. Estaba fijo en
        # castellano, así que un recibo emitido desde una pantalla en inglés
        # decía «Purpose: medir accuracy de un modelo externo» — medido el
        # 2026-08-26 al capturar las pantallas del manual. Lo que redacta el
        # core se traduce EN EL CORE, no al pintarlo.
        "subject": {"purpose": proposito or _proposito_por_defecto(metrica, locale),
                    "actor": actor or "no consta quién lo pidió"},
        "models": [{
            "model_id": ruta_modelo.name,
            "version": "external",
            "digest": f"sha256:{digest_modelo}",
            # DE DÓNDE SALE ESTE MODELO: de fuera. Que conste en el recibo y no
            # solo en la documentación.
            "provenance": "external: not trained by MatrixAI",
            # CÓMO SE ALIMENTÓ, DICHO. Si el modelo declara ejes dinámicos no se
            # puede contrastar el número de columnas contra la forma, y el recibo
            # lo dice —`"no declarada"`— en vez de callar: callar aquí es afirmar
            # que se comprobó.
            "input_spec": {
                "name": meta.name,
                "type": str(getattr(meta, "type", "") or ""),
                "shape": [d if isinstance(d, int) else str(d)
                          for d in (getattr(meta, "shape", []) or [])],
                "features_declared": esperado_declarado,
                "shape_check": comprobacion_forma,
            },
        }],
        "dataset": {
            "name": ruta_datos.name,
            "digest": f"sha256:{digest_datos}",
            "rows_total": len(filas),
            "rows_measured": medidas,
            "target_column": objetivo,
            "input_columns": entradas_nombres,
        },
        "metrics": [{
            "name": metrica,
            "value": valor,
            "split": "provided",
            "direction": "higher_is_better" if metrica == "accuracy" else "lower_is_better",
            "dataset_sha256": digest_datos,
        }],
        "steps": [{
            "id": "evaluate",
            "model": ruta_modelo.name,
            "entry_hash": f"sha256:{digest_modelo}",
        }],
        "output": {"outcome": "completed",
                   "output_digests": {"metrics": hashlib.sha256(
                       jcs_bytes({"name": metrica, "value": valor})).hexdigest()}},
        # EL ENTORNO, REGISTRADO Y NO SOLO PROMETIDO (auditoría externa del
        # 2026-08-25, hallazgo 8). `evidence.attests` decía «la evaluación, el
        # entorno y las entradas» y el recibo **no traía ni una línea de
        # entorno**: una afirmación mayor que la evidencia guardada, que es
        # exactamente lo que este producto dice no hacer.
        "environment": {
            "onnxruntime": getattr(onnxruntime, "__version__", None),
            "providers": list(sesion.get_providers() or []),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "evidence": {
            "attests": "the evaluation, the environment and the inputs",
            "does_not_attest": (
                "how this model was trained, nor with what data: a copied model "
                "produces the same receipt as an own one"),
        },
    }
    return recibo
