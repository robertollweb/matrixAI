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

from matrixai.estudio.errores import ErrorDeEstudio
from matrixai.estudio.metricas import REGISTRO, Muestra, calcular, direccion_de
from matrixai.export.onnx_salida import (
    SalidaOnnxAmbigua, elegir_salida, etiqueta_de, misma_etiqueta, resolver_clases,
    texto_de_etiqueta, valor_de)
from matrixai.onnx_entrada import (
    EntradaOnnxIncompatible, caracteristicas_declaradas, tensor_para)

import csv
import hashlib
import io
import platform
from pathlib import Path
from typing import Any

__all__ = ["AtestacionImposible", "METRICAS", "atestiguar"]

#: LO QUE ESTE CAMINO PUEDE PONER SOBRE LA MESA. `attest` corre un modelo ajeno
#: fila a fila y lee UNA salida: de ahí sale la clase predicha (o el valor
#: predicho) y nada más. No hay probabilidades calibradas —una salida declarada
#: `probabilidades` se lee para sacar la clase del máximo, no se publica como
#: distribución—, ni puntuaciones, ni pesos de muestreo. Declararlo aquí es lo
#: que permite que la LISTA de métricas la decida el registro y no esta línea.
#:
#: Y NO APORTA LA CLASE POSITIVA, aunque `_muestra` tenga que poner una (desde el
#: 115-C2). Declara la última del vocabulario porque `Muestra` no se construye
#: binaria sin ella, y con las clases sacadas de lo observado esa es arbitraria.
#: Con `accuracy` y `macro_f1` no importa (está probado); con sensibilidad o VPP
#: el número cambiaba con el ORDEN de las filas (0,75 o 0,50 para los mismos
#: datos, medido), y el registro las abrió a las etiquetas en el 115-C2.
_LO_QUE_APORTA_ATESTIGUAR = ("y_true", "classes", "labels")


def _se_puede_atestiguar(registrada: Any) -> bool:
    """¿Puede este camino alimentar esa métrica del registro?

    Se le pregunta al REGISTRO por sus requisitos y por sus grupos alternativos
    —un AUROC no necesita `scores` NI `probabilities`, necesita una de las dos—
    en vez de repetir aquí qué necesita cada métrica. Repetirlo era lo que
    convertía esto en una lista escrita a mano.
    """
    return all(
        any(x in _LO_QUE_APORTA_ATESTIGUAR for x in registrada.grupo_de(requisito))
        for requisito in registrada.spec.requires)


#: Las métricas que este camino sabe medir. **Ya no se escriben aquí**: salen
#: del registro único de 105-C1, que es donde viven las fórmulas desde el
#: 2026-09-06. Antes eran `("accuracy", "mae")` a mano y este módulo calculaba
#: las dos con fórmula propia (`aciertos / medidas`, `error_absoluto / medidas`):
#: dos sitios midiendo lo mismo, que es exactamente lo que el invariante 1 del
#: 105 prohíbe, y el 100 §9 D15 declara «métricas abiertas» como HECHO. Medido
#: el 2026-09-12: abrirla añade `macro_f1`, `rmse` y `r2` sin escribir una sola
#: fórmula, y deja fuera —bien— las que piden probabilidades o puntuaciones, que
#: este camino no tiene.
METRICAS = tuple(mid for mid, reg in REGISTRO.items() if _se_puede_atestiguar(reg))


def _tarea_de(metrica: str) -> str:
    """`"regresion"` o `"clasificacion"`, según lo que declare el REGISTRO.

    Estaba escrito como `"regresion" if metrica == "mae" else "clasificacion"`:
    con la lista abierta, esa línea habría medido un `rmse` como si fuera una
    clasificación.
    """
    return "regresion" if "regression" in REGISTRO[metrica].tareas else "clasificacion"


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


def _vocabulario(clases_usadas: list | None, etiquetas: list) -> tuple[list[str], list[str]]:
    """Las clases COMO TEXTO y cada etiqueta traducida a la suya.

    El registro mide contra un vocabulario declarado: `y_true` y la clase
    predicha tienen que ser LA MISMA cadena para que cuenten como acierto. Aquí
    no lo son —el ONNX devuelve un `int64` y el CSV una cadena—, así que quien
    decide si dos etiquetas son la misma clase sigue siendo `misma_etiqueta()`,
    que es la regla de este paquete y ya sabe que `"01"` no es `"1"`. Lo que se
    traduce es la IDENTIDAD de la clase; la FÓRMULA la pone el registro.

    Una etiqueta que no está entre las declaradas se añade al vocabulario en vez
    de rechazarse: las clases declaradas dicen qué hay en cada POSICIÓN de la
    salida, no qué puede traer la columna objetivo, y una fila cuya clase el
    modelo no puede predecir es una fila que falla — que es lo que contaba antes
    y lo que sigue contando.
    """
    vocabulario = [texto_de_etiqueta(c) for c in (clases_usadas or [])]
    traducidas: list[str] = []
    for etiqueta in etiquetas:
        for clase in vocabulario:
            if misma_etiqueta(etiqueta, clase):
                traducidas.append(clase)
                break
        else:
            nueva = texto_de_etiqueta(etiqueta)
            vocabulario.append(nueva)
            traducidas.append(nueva)
    return vocabulario, traducidas


def _muestra(tarea: str, esperadas: list, predichas: list,
             clases_usadas: list | None) -> Muestra:
    """Lo medido, en el documento que el registro sabe leer.

    LA CLASE POSITIVA, y por qué se declara la última. `Muestra` no se construye
    binaria sin ella, y este camino no siempre sabe cuál es: cuando las clases
    vienen declaradas, el orden de este paquete es «la negativa primero y la
    positiva después» (`resolver_clases`), así que la última es la positiva; y
    cuando salen de lo observado no hay ninguna positiva que valga. No importa
    para lo que aquí se mide y está PROBADO que no importa: las dos métricas de
    clasificación que este camino puede alimentar —`accuracy` y `macro_f1`—
    salen de la matriz de confusión, que con la clase predicha declarada usa la
    regla `declared_label` y no mira la positiva. Por eso tampoco se publica en
    el recibo: un campo que no se ha medido no se afirma.
    """
    if tarea == "regresion":
        return Muestra.regresion(esperadas, predichas)
    vocabulario, traducidas = _vocabulario(clases_usadas, list(esperadas) + list(predichas))
    cuantas = len(esperadas)
    if len(vocabulario) < 2:
        raise AtestacionImposible(
            f"solo consta una clase ({vocabulario[0]!r} en la columna objetivo y en "
            f"todo lo que predijo el modelo): una clasificación se mide contra un "
            f"vocabulario de al menos dos clases, y con una sola el número no dice "
            f"contra qué se comparó. Declara las clases")
    if len(vocabulario) == 2:
        return Muestra(task="binary_classification", y_true=tuple(traducidas[:cuantas]),
                       classes=tuple(vocabulario), positive_label=vocabulario[-1],
                       predictions=tuple(traducidas[cuantas:]))
    return Muestra.multiclase(traducidas[:cuantas], classes=vocabulario,
                              predicciones=traducidas[cuantas:])


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
        # SE DICE QUÉ PASA, no solo que no se puede. Con la lista abierta hay dos
        # negativas distintas y confundirlas manda a buscar una errata donde no
        # la hay: una métrica que el registro NO conoce, y una que sí conoce pero
        # que este camino no puede alimentar —un AUROC necesita puntuaciones, y
        # de un modelo ajeno aquí solo se lee la clase predicha—.
        if metrica in REGISTRO:
            raise AtestacionImposible(
                f"{metrica!r} existe en el registro métrico, pero necesita "
                f"{list(REGISTRO[metrica].spec.requires)} y atestiguar un modelo ajeno "
                f"solo da la clase (o el valor) que predice. Este camino mide "
                f"{list(METRICAS)}")
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
    tarea = _tarea_de(metrica)
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

    # SE RECOGE LO PREDICHO; EL NÚMERO LO CALCULA EL REGISTRO. Aquí había un
    # `aciertos / medidas` y un `error_absoluto / medidas` propios: la segunda
    # copia de dos fórmulas que ya viven en `matrixai.estudio.metricas` desde el
    # 105-C1 (2026-09-06). El día que una de las dos cambiara —un peso, un
    # redondeo, un empate— nadie sabría cuál de los dos números firmó el recibo.
    predichas: list[Any] = []
    esperadas: list[Any] = []
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
        try:
            if tarea == "clasificacion":
                # LA ETIQUETA, NO UN ÍNDICE (102-C0). O la que dio el modelo, o
                # la clase que ocupa la posición del máximo — y de dónde salen
                # esas clases va en el recibo.
                predichas.append(etiqueta_de(bruto, elegida, clases_usadas))
                esperadas.append(esperado)
            else:
                predichas.append(valor_de(bruto))
                esperadas.append(float(esperado))
        except SalidaOnnxAmbigua as exc:
            raise AtestacionImposible(str(exc)) from None
        except ValueError:
            raise AtestacionImposible(
                f"la columna {objetivo!r} trae {esperado!r}, que no es un número, y "
                f"{metrica!r} se mide sobre números") from None

    medidas = len(predichas)
    if medidas == 0:
        raise AtestacionImposible(
            "ninguna fila del CSV se pudo medir: no hay resultado que atestiguar")

    # EL NÚMERO, CON LA FÓRMULA DEL REGISTRO ÚNICO. Lo que este módulo aporta es
    # QUÉ se midió —qué salida se leyó, qué clase es cada posición, qué fila se
    # pudo medir—; cuánto vale una exactitud lo dice un solo sitio.
    try:
        medido = calcular(metrica, _muestra(tarea, esperadas, predichas, clases_usadas))
    except ErrorDeEstudio as exc:
        raise AtestacionImposible(
            f"no se puede medir {metrica!r} sobre lo que dio el modelo: "
            f"{exc.motivo(locale)}") from None
    if medido.value is None:
        # Un valor ausente NO es un cero, y un recibo que firmara un cero aquí
        # afirmaría que se midió algo que no se pudo medir.
        razon = (medido.undefined_reason or {}).get(
            "es" if str(locale or "es").strip().lower() == "es" else "en", "")
        raise AtestacionImposible(
            f"{metrica!r} no se puede medir sobre estos datos: {razon}")
    valor = medido.value
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
            # LA DIRECCIÓN LA DECLARA EL REGISTRO. Estaba escrita como «higher
            # si es accuracy, lower en cualquier otro caso»: con la lista
            # abierta, ese `else` habría publicado un `r2` como si menos fuera
            # mejor. Una métrica con valor ideal en vez de dirección pondría
            # `null` aquí, que es la respuesta honesta y no una inventada.
            "direction": direccion_de(metrica),
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
