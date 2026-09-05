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

from matrixai.export.onnx_salida import (
    SalidaOnnxAmbigua, elegir_salida, etiqueta_de, misma_etiqueta, resolver_clases,
    valor_de)
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
    salida: str | None = None,
    semantica: str | None = None,
    clases: list | None = None,
) -> dict[str, Any]:
    """Corre el modelo sobre los datos y devuelve el RECIBO (sin firmar).

    Firmarlo es un acto aparte, como en todo el 81: quien firma decide con qué
    clave, y un recibo sin firmar es A0 y lo dice.

    `salida`, `semantica` y `clases` son el **mapa de salida** (102-C0): qué
    salida del grafo se lee, qué significa y en qué orden están las clases. No
    hacen falta cuando el modelo lo declara —una salida de etiquetas, un mapa
    por clase, un `Softmax`—; se piden cuando no, porque suponerlo es lo que
    hacía que un modelo de tres clases atestiguara 0,6667 donde la exactitud
    era 0,9733.
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

    # LAS FILAS QUE DE VERDAD SE PUEDEN MEDIR, decididas UNA VEZ y aquí. Antes
    # el filtro vivía dentro del bucle; al deducir las clases de la columna
    # objetivo hacía falta el mismo criterio, y tenerlo en dos sitios habría
    # hecho lo de siempre: tres filas de basura —que el recibo ya declara como
    # no medidas— habrían aportado sus etiquetas a la deducción de clases y la
    # habrían hecho imposible. Una fila que no es numérica no se convierte en un
    # cero: se salta, se cuenta, y tampoco cuenta para las clases.
    medibles: list[tuple[list, str]] = []
    for fila in filas:
        if len(fila) != len(cabecera):
            continue
        try:
            vector = [float(v) for i, v in enumerate(fila) if i != indice]
        except ValueError:
            continue
        medibles.append((vector, fila[indice]))

    if not medibles:
        raise AtestacionImposible(
            "ninguna fila del CSV se pudo medir: no hay resultado que atestiguar")

    # QUÉ SE LEE DE LO QUE DEVUELVE EL MODELO, Y POR QUÉ ÉSO (102-C0). Antes se
    # tomaba `run(...)[0]` —la PRIMERA salida— y se umbralizaba a 0,5. Los
    # conversores ponen `label` primero: en binaria cuadraba por accidente y con
    # tres clases no. La decisión vive en `onnx_salida`, el mismo sitio para
    # todos los caminos que lean un modelo ajeno.
    tarea = "regresion" if metrica == "mae" else "clasificacion"
    try:
        elegida = elegir_salida(sesion, tarea=tarea, ruta_modelo=ruta_modelo,
                                nombre=salida, semantica=semantica)
    except SalidaOnnxAmbigua as exc:
        raise AtestacionImposible(str(exc)) from None

    meta_salida = sesion.get_outputs()[elegida.indice]
    forma_salida = list(getattr(meta_salida, "shape", []) or [])
    ancho_salida = forma_salida[-1] if (len(forma_salida) > 1
                                        and isinstance(forma_salida[-1], int)) else None
    etiquetas_del_csv = [esperado for _, esperado in medibles]

    if str(elegida.tipo_onnx).startswith("seq(map"):
        # Las clases vienen DENTRO del modelo, una por clave: no hay posiciones
        # que traducir ni nada que suponer.
        clases_usadas, origen_de_clases = None, "las trae el modelo (mapa por clase)"
    else:
        try:
            clases_usadas, origen_de_clases = resolver_clases(
                elegida, clases_declaradas=clases,
                etiquetas_observadas=etiquetas_del_csv, ancho=ancho_salida)
        except SalidaOnnxAmbigua as exc:
            raise AtestacionImposible(str(exc)) from None

    aciertos = 0
    error_absoluto = 0.0
    medidas = 0
    for vector, esperado in medibles:
        # El TIPO que el modelo declara, no `float` siempre: un modelo de
        # `int64` alimentado con `3.7` lo trunca en silencio a `3`, y el acierto
        # que salga de ahí iría al recibo como si se hubiera medido sobre el dato
        # del CSV. Un dato que nadie escribió no se atestigua.
        try:
            tensor = tensor_para(meta, vector, "el modelo que se atestigua")
        except EntradaOnnxIncompatible as exc:
            raise AtestacionImposible(str(exc)) from None
        bruto = sesion.run(None, {meta.name: tensor})[elegida.indice]
        medidas += 1
        try:
            if metrica == "accuracy":
                # LA ETIQUETA, NO UN ÍNDICE (102-C0). O la que dio el modelo, o
                # la clase que ocupa la posición del máximo — y de dónde salen
                # esas clases va en el recibo.
                predicha = etiqueta_de(bruto, elegida, clases_usadas)
                aciertos += int(misma_etiqueta(predicha, esperado))
            else:
                error_absoluto += abs(valor_de(bruto) - float(esperado))
        except SalidaOnnxAmbigua as exc:
            raise AtestacionImposible(str(exc)) from None
        except ValueError:
            raise AtestacionImposible(
                f"la columna {objetivo!r} trae {esperado!r}, que no es un número, y "
                f"{metrica!r} se mide sobre números") from None

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
            # QUÉ SE LEYÓ DE LO QUE DEVOLVIÓ, Y CON QUÉ CLASES (102-C0). Quien
            # lea el recibo tiene que poder saber, sin abrir el ONNX, si el
            # número salió de la predicción del propio modelo o de un `argmax`
            # que hizo MatrixAI, y de dónde salieron las clases. Un recibo que
            # no lo dice deja el número sin apellido.
            "output_spec": {
                "name": elegida.nombre,
                "type": elegida.tipo_onnx,
                "kind": elegida.semantica,
                "chosen_because": elegida.motivo,
                "classes": list(clases_usadas) if clases_usadas is not None else None,
                "classes_source": origen_de_clases,
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
