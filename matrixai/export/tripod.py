# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La ficha TRIPOD+AI de un paquete — contrato 85-C6.

QUÉ ES. TRIPOD+AI (BMJ, 2024) es la lista de comprobación —27 ítems— que se pide
para publicar un estudio que desarrolla o valida un modelo de predicción
clínica. Lo medido en los estudios bibliométricos posteriores es que **la
mayoría de los trabajos sigue sin cumplirla**: se rellena a mano, al final y de
memoria.

QUÉ HACE ESTO. Escribe lo que el run YA guardó —`reproduce.json`, el `.mxai`, el
`.mxtrain`, la receta y las métricas— en el formato que ese público tiene que
entregar. **No inventa una sola casilla**, y las que no puede rellenar las
enumera con su motivo.

LO QUE NO ES, y va escrito en la propia ficha: no es un producto sanitario, no
aprueba nada ante nadie, no sustituye al criterio de quien firma el estudio y no
convierte un modelo entrenado con datos sintéticos en uno validado clínicamente.

SOBRE LA NUMERACIÓN. Este informe va por TEMA y no por número de ítem. El PDF
oficial del checklist usa fuentes con subconjunto y su numeración no se pudo
extraer de forma fiable (ver `documentacion/85_C6_MAPA_TRIPOD.md`): **inventar
los números sería exactamente el defecto que este producto persigue**. Quien
rellene el checklist oficial encontrará aquí el contenido de cada tema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matrixai.export.expediente_clinico import (
    CALIBRACION_Y_UMBRAL,
    DOMINIOS,
    ENTRENAMIENTO_DE_LA_RED,
    ENTRENAMIENTO_DEL_ENTREGADO,
    FILAS_DEL_ESTUDIO,
    REPARTO_POR_DISENO,
    FUENTE_EN_ESTUDIO,
    FUERA_POR_EL_MODO,
    MODO_DEL_ESTUDIO,
    MOTOR_ENTREGADO,
    POR_CLAVE,
    ExpedienteClinico,
    linea_de_campo,
    rotulo_de_dominio,
    sin_veredicto_de_riesgo,
    texto_del_valor,
)

__all__ = ["FichaNoDisponible", "ficha_tripod"]


class FichaNoDisponible(RuntimeError):
    """No hay paquete del que sacar la ficha."""


#: Lo que este informe NO puede rellenar, y por qué.
#:
#: **LA LISTA YA NO ES CERRADA, Y ESA ES LA CORRECCIÓN DE 109-C3.** Cuando se
#: escribió (85-C6) el core no medía calibración ni subgrupos ni declaraba
#: política de ausentes, así que decir «no se calcula» era verdad. Desde
#: 105-C3/C4 y 109-C2 lo mide, y dejar el aviso puesto lo convertiría en una
#: frase falsa dentro de una ficha cuyo único valor es no decir ninguna: un
#: apaño que caduca sin avisar es peor que no ponerlo.
#:
#: Ahora cada línea lleva **la clave del inventario que la rellenaría**. Si ese
#: campo está en el paquete, la línea no se imprime — porque ya no falta. Las
#: que llevan `None` no las puede rellenar ningún run: son del autor del
#: estudio.
#:
#: Y los motivos de las que SÍ tienen clave se han reescrito: lo que pasa hoy
#: no es que el core no sepa medirlo, es que ESTE paquete no lo trae.
_NO_DISPONIBLE: tuple[tuple[str | None, dict[str, tuple[str, str]]], ...] = (
    ("politica_de_faltantes", {
        "es": ("Datos ausentes",
               "este paquete no declara ninguna política de valores ausentes, así "
               "que no se puede afirmar qué se hizo con ellos"),
        "en": ("Missing data",
               "this package declares no missing-value policy, so what was done "
               "with them cannot be stated")}),
    ("calibracion", {
        "es": ("Calibración",
               "este paquete no trae la curva de calibración, que el core sí "
               "sabe medir"),
        "en": ("Calibration",
               "this package ships no calibration curve, which the core does know "
               "how to measure")}),
    ("subgrupos", {
        "es": ("Equidad por subgrupos",
               "este paquete no trae ningún subgrupo medido; hace falta que el "
               "equipo los predefina"),
        "en": ("Subgroup fairness",
               "this package ships no measured subgroup; it needs a team to "
               "predefine them")}),
    (None, {"es": ("Financiación", "es del autor del estudio, no del run"),
            "en": ("Funding", "belongs to a study's author, not to a run")}),
    (None, {"es": ("Conflictos de interés", "ídem"),
            "en": ("Conflicts of interest", "idem")}),
    (None, {"es": ("Aprobación ética y registro del estudio", "ídem"),
            "en": ("Ethical approval and study registration", "idem")}),
    ("uso_previsto", {
        "es": ("Uso previsto y población destinataria",
               "lo escribe una persona; este informe lo pide, no lo inventa"),
        "en": ("Intended use and target population",
               "a person writes this; this report asks for it, it does not invent "
               "it")}),
)

#: Público (137-C2): el informe de «¿Funciona aquí?» toma de aquí los rótulos de lo que solo
#: declara el autor (financiación, conflictos, ética) en vez de repetirlos.
NO_DISPONIBLE = _NO_DISPONIBLE

_T = {
    "es": {
        "titulo": "Ficha TRIPOD+AI",
        # Lo compone el núcleo desde el manifiesto, y eso es cierto por los dos caminos: la orden
        # `matrixai report --tripod` y el expediente de un estudio en el Studio. Nombrar la orden
        # afirmaba algo que en el segundo camino no se ejecutó (auditoría de 138-C1..C3).
        "generado": "La compone el núcleo de MatrixAI a partir del manifiesto del "
                    "paquete (`reproduce.json`). Cada línea sale de lo que se guardó; "
                    "lo que no se sabe se dice.",
        "aviso_sintetico": "**AVISO: los datos de entrenamiento son SINTÉTICOS**, "
                           "generados por la receta que viaja en este paquete. Un "
                           "modelo entrenado así no está validado clínicamente.",
        "aviso_no_regulatorio": "Esto no es un producto sanitario ni una aprobación "
                                "regulatoria, y no sustituye al criterio de quien "
                                "firma el estudio.",
        "datos": "Datos", "fuente": "Fuente", "filas": "Filas que entrenaron",
        "huella_datos": "Huella del dataset (sha256)", "receta": "Receta de datos",
        "predictores": "Predictores (entradas)", "excluidas": "Columnas excluidas",
        "objetivo": "Variable de resultado", "particion": "Partición y semillas",
        "modelo": "Modelo", "arquitectura": "Arquitectura (`.mxai`)",
        "hiperparametros": "Hiperparámetros (`.mxtrain`)",
        "epocas": "Épocas declaradas / efectivas / ejecutadas",
        "entorno": "Entorno", "metricas": "Rendimiento",
        "medido_sobre": "medido sobre el dataset",
        "disponibilidad": "Disponibilidad del modelo",
        "reproducibilidad": "Reproducibilidad",
        "no_disponible": "Lo que esta ficha NO puede rellenar",
        "perfil": "Perfil clínico de la tarea",
        "perfil_intro": "Lo que este paquete sostiene de lo que un revisor "
                        "pregunta, y el campo del que sale cada línea. Se rellena "
                        "lo medido, se marca lo declarado y se enumera lo que "
                        "falta; **esta ficha no emite ningún juicio sobre el "
                        "riesgo de sesgo**, que lo pone quien revisa el estudio.",
        "perfil_avisos": "Avisos sobre el propio paquete",
        "sin_dato": "no disponible",
        "paquete_usable": "El paquete incluye `model.onnx`, `predict.py` e "
                          "`inference_spec.json`: predice con valores crudos sin "
                          "instalar MatrixAI.",
        "paquete_no_usable": "El paquete NO incluye {faltan}: no predice con valores "
                             "crudos sin instalar MatrixAI.",
        "reproducible_si": "El paquete se declara reproducible. Se comprueba con "
                           "`matrixai verify <paquete>`.",
        "reproducible_no": "El paquete se declara NO reproducible",
        "m_sin_modo": "el paquete no declara con qué modo se generó",
        "m_sin_tipos": "el run no registró los tipos de las columnas de entrada",
        "m_sin_metricas": "el paquete no publica ninguna métrica",
    },
    "en": {
        "titulo": "TRIPOD+AI record",
        "generado": "Composed by the MatrixAI core from the package manifest "
                    "(`reproduce.json`). Every line comes from what was recorded; what "
                    "is not known is said.",
        "aviso_sintetico": "**WARNING: the training data is SYNTHETIC**, generated by "
                           "the recipe shipped in this package. A model trained this "
                           "way is not clinically validated.",
        "aviso_no_regulatorio": "This is not a medical device nor a regulatory "
                                "approval, and it does not replace the judgement of "
                                "whoever signs the study.",
        "datos": "Data", "fuente": "Source", "filas": "Rows that trained",
        "huella_datos": "Dataset digest (sha256)", "receta": "Data recipe",
        "predictores": "Predictors (inputs)", "excluidas": "Excluded columns",
        "objetivo": "Outcome variable", "particion": "Split and seeds",
        "modelo": "Model", "arquitectura": "Architecture (`.mxai`)",
        "hiperparametros": "Hyperparameters (`.mxtrain`)",
        "epocas": "Epochs declared / effective / ran",
        "entorno": "Environment", "metricas": "Performance",
        "medido_sobre": "measured on dataset",
        "disponibilidad": "Model availability",
        "reproducibilidad": "Reproducibility",
        "no_disponible": "What this record CANNOT fill in",
        "perfil": "Clinical task profile",
        "perfil_intro": "What this package supports of what a reviewer asks, and "
                        "where every line comes from. Measured items are filled "
                        "in, declared items are marked, missing items are listed; "
                        "**this record passes no judgement on risk of bias**, "
                        "which is made by whoever reviews a study.",
        "perfil_avisos": "Warnings about this package itself",
        "sin_dato": "not available",
        "paquete_usable": "The package ships `model.onnx`, `predict.py` and "
                          "`inference_spec.json`: it predicts from raw values with no "
                          "MatrixAI installed.",
        "paquete_no_usable": "The package does NOT ship {faltan}: it does not predict from "
                             "raw values without MatrixAI installed.",
        "reproducible_si": "The package declares itself reproducible. Check it with "
                           "`matrixai verify <package>`.",
        "reproducible_no": "The package declares itself NOT reproducible",
        "m_sin_modo": "the package does not declare which generation mode produced it",
        "m_sin_tipos": "the run did not record the input field types",
        "m_sin_metricas": "the package publishes no metrics",
    },
}


def _t(locale: str) -> dict[str, str]:
    return _T.get(str(locale or "en").strip().lower(), _T["en"])


def _linea(rotulo: str, valor: Any, sin_dato: str, motivo: str = "",
           ruta: str = "") -> str:
    """Una línea de la ficha: su valor, o que no lo hay **y por qué**, y SU RUTA.

    La ruta (`fichero#campo`) cierra el criterio de terminado del 109-C3: cada
    frase de la ficha traza a un campo. Va también cuando el dato no está —
    sobre todo entonces—: saber QUÉ campo lo arreglaría es la mitad útil de un
    hueco, y una ficha de reporte cuyo lector no puede comprobar una línea vale
    lo mismo que una rellenada de memoria.
    """
    cola_ruta = f" · `{ruta}`" if ruta else ""
    if valor in (None, "", [], {}):
        cola = f" ({motivo})" if motivo else ""
        return f"- **{rotulo}**: _{sin_dato}_{cola}{cola_ruta}"
    if isinstance(valor, (list, tuple)):
        valor = ", ".join(str(v) for v in valor)
    return f"- **{rotulo}**: {valor}{cola_ruta}"


def _campos_del_modelo(bundle: Path) -> list[str]:
    """Los campos de entrada que declara el `.mxai` que viaja en el paquete.

    Con su rango cuando lo declara: `edad (18–100)` dice más que `edad`, y es lo
    que un revisor necesita para saber sobre qué población se entrenó.

    Si el modelo no está o no se puede leer, se devuelve vacío y quien llama
    dirá que no hay dato — nunca una lista a medias.
    """
    ruta = bundle / "model.mxai"
    if not ruta.is_file():
        return []
    try:
        from matrixai.parser.parser import parse_text  # noqa: PLC0415

        programa = parse_text(ruta.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    campos: list[str] = []
    for vector in getattr(programa, "vectors", []) or []:
        for nombre, tipo in (getattr(vector, "field_types", None) or {}).items():
            rango = getattr(tipo, "range", None)
            lo, hi = getattr(rango, "minimum", None), getattr(rango, "maximum", None)
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                campos.append(f"{nombre} ({lo:g}–{hi:g})")
            else:
                campos.append(str(nombre))
    return campos


def ficha_tripod(bundle_dir: str | Path, *, locale: str = "en") -> str:
    """La ficha, en Markdown, de un paquete ya exportado."""
    bundle = Path(bundle_dir)
    manifiesto_path = bundle / "reproduce.json"
    if not manifiesto_path.is_file():
        raise FichaNoDisponible(
            f"{bundle}: no reproduce.json — a TRIPOD+AI record is written from what "
            f"the run recorded, and this package records nothing")
    try:
        m = json.loads(manifiesto_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FichaNoDisponible(f"{manifiesto_path}: {exc}") from exc

    t = _t(locale)
    gen = m.get("generation") or {}
    art = m.get("artifacts") or {}
    env = m.get("environment") or {}
    receta = (bundle / "data_recipe.txt")
    receta_txt = receta.read_text(encoding="utf-8").strip() if receta.is_file() else ""

    filas = []
    filas.append(f"# {t['titulo']}")
    filas.append("")
    filas.append(f"> {t['generado']}")
    filas.append(">")
    filas.append(f"> {t['aviso_no_regulatorio']}")
    # EL AVISO DE SINTÉTICO VA ARRIBA, no en una nota al pie: si los datos son
    # sintéticos, es lo primero que tiene que saber quien lea.
    if receta_txt:
        filas.append(">")
        filas.append(f"> {t['aviso_sintetico']}")
    filas.append("")

    idioma = "es" if _t(locale) is _T["es"] else "en"
    expediente = ExpedienteClinico.desde_paquete(bundle)

    filas.append(f"## {t['datos']}")
    modo = gen.get("mode")
    filas.append(_linea(t["fuente"],
                        f"`{modo}`" if modo else None, t["sin_dato"],
                        FUENTE_EN_ESTUDIO.motivo_si_falta[idioma] if expediente.es_de_estudio
                        else t["m_sin_modo"], "reproduce.json#generation.mode"))
    if expediente.es_de_estudio:
        # UN ESTUDIO NO ENTRENA CON TODAS SUS FILAS: `rows` son las del conjunto (las de la
        # huella) y el reparto, lo que el estudio contó. Rótulos y rutas, los del inventario:
        # la ficha y PROBAST no pueden decir dos cosas de la misma cifra.
        for evidencia in (FILAS_DEL_ESTUDIO, *expediente.reparto_del_estudio()):
            campo = expediente.campo(evidencia.ruta)
            filas.append(_linea(evidencia.rotulo[idioma], campo.valor, t["sin_dato"],
                                evidencia.motivo_si_falta[idioma], evidencia.ruta))
    else:
        filas.append(_linea(t["filas"], (art.get("dataset") or {}).get("rows"),
                            t["sin_dato"], ruta="reproduce.json#artifacts.dataset.rows"))
    filas.append(_linea(t["huella_datos"], (art.get("dataset") or {}).get("sha256"),
                        t["sin_dato"],
                        ruta="reproduce.json#artifacts.dataset.sha256"))
    # LOS PREDICTORES, DEL MODELO SI EL MANIFIESTO NO LOS TRAE.
    #
    # La primera ficha de la galería decía «Predictores: no disponible» sobre un
    # `.mxai` que declara los cuatro con su rango. El dato estaba **dentro del
    # propio paquete** (`model.mxai`), y no llegaba porque la captura del run
    # —que es la que manda para `generation`— todavía no los registra.
    #
    # Leerlos del modelo empaquetado no es inventarlos: es citar el fichero que
    # viaja al lado, y por eso se dice de dónde salen.
    _del_manifiesto = list((gen.get("field_types") or {}).keys())
    _predictores = _del_manifiesto or _campos_del_modelo(bundle)
    filas.append(_linea(t["predictores"], _predictores or None,
                        t["sin_dato"], t["m_sin_tipos"],
                        ruta=("reproduce.json#generation.field_types"
                              if _del_manifiesto else "model.mxai#vectors")))
    filas.append(_linea(t["excluidas"], gen.get("excluded_identifiers"),
                        t["sin_dato"],
                        ruta="reproduce.json#generation.excluded_identifiers"))
    filas.append(_linea(t["particion"], json.dumps(gen.get("seeds"), sort_keys=True)
                        if gen.get("seeds") else None, t["sin_dato"],
                        ruta="reproduce.json#generation.seeds"))
    if receta_txt:
        filas.append("")
        filas.append(f"**{t['receta']}** · `data_recipe.txt#`:")
        filas.append("")
        filas.append("```")
        filas.append(receta_txt)
        filas.append("```")
    filas.append("")

    filas.append(f"## {t['modelo']}")
    filas.append(_linea(t["arquitectura"], (art.get("model") or {}).get("sha256"),
                        t["sin_dato"], ruta="reproduce.json#artifacts.model.sha256"))
    filas.append(_linea(t["hiperparametros"], (art.get("training") or {}).get("sha256"),
                        t["sin_dato"],
                        ruta="reproduce.json#artifacts.training.sha256"))
    # LAS TRES ÉPOCAS, Y NINGUNA COMO `None`. Imprimir «60 / None / None» pone
    # tres números en fila donde solo hay uno: un valor ausente no es un dato, y
    # aquí encima parece uno. Cada hueco se marca por su cuenta.
    _epocas = [gen.get(k) for k in ("epochs_declared", "epochs_effective", "epochs_ran")]
    if expediente.es_de_estudio and expediente.entrenamiento_de_la_red is not None:
        # Reparación de la re-auditoría del 139 (I1): un estudio con una red entregada con épocas fijas NO las registra en
        # `generation` (eso es del camino clásico) sino en `study.network_training`; la línea «Épocas» decía «no disponible»
        # de un estudio que guardó 20. Se escribe la del estudio, con el rótulo y la ruta del inventario (PROBAST).
        campo = expediente.campo(ENTRENAMIENTO_DE_LA_RED.ruta)
        filas.append(_linea(ENTRENAMIENTO_DE_LA_RED.rotulo[idioma],
                            texto_del_valor(ENTRENAMIENTO_DE_LA_RED, campo, idioma) if campo.hay_dato else None,
                            t["sin_dato"], ENTRENAMIENTO_DE_LA_RED.motivo_si_falta[idioma], ENTRENAMIENTO_DE_LA_RED.ruta))
    elif expediente.es_de_estudio and expediente.entrenamiento_del_entregado is not None:
        # Auditoría de Roberto del 139 (P3-5): una red que cayó al pliegue 0 no trae `network_training` (no se reentrenó con
        # épocas fijas), y la línea decía «no disponible» sin una palabra del plazo que pudo cortarla. Se escribe cómo terminó
        # el entrenamiento de la red ENTREGADA (`study.delivered_network_training`), con el rótulo y la ruta del inventario.
        campo = expediente.campo(ENTRENAMIENTO_DEL_ENTREGADO.ruta)
        filas.append(_linea(ENTRENAMIENTO_DEL_ENTREGADO.rotulo[idioma],
                            texto_del_valor(ENTRENAMIENTO_DEL_ENTREGADO, campo, idioma) if campo.hay_dato else None,
                            t["sin_dato"], ENTRENAMIENTO_DEL_ENTREGADO.motivo_si_falta[idioma],
                            ENTRENAMIENTO_DEL_ENTREGADO.ruta))
    else:
        filas.append(_linea(
            t["epocas"],
            " / ".join(str(v) if v is not None else f"_{t['sin_dato']}_" for v in _epocas)
            if any(v is not None for v in _epocas) else None, t["sin_dato"],
            ruta="reproduce.json#generation.epochs_declared+effective+ran"))
    _piezas = [
        f"matrixai {env['matrixai_version']}" if env.get("matrixai_version") else None,
        f"python {(env.get('python') or {}).get('version')}"
        if (env.get("python") or {}).get("version") else None,
        f"{gen['backend']}/{gen['device']}" if gen.get("backend") and gen.get("device")
        else (gen.get("backend") or gen.get("device")),
    ]
    filas.append(_linea(t["entorno"], " · ".join(p for p in _piezas if p) or None,
                        t["sin_dato"], ruta="reproduce.json#environment"))
    if expediente.es_de_estudio:
        # 138-C1: el modo con que corrió el estudio, y en el rápido quién no compitió por él.
        _lineas_del_modo = [MODO_DEL_ESTUDIO]
        if expediente.campo(MODO_DEL_ESTUDIO.ruta).valor == "rapido":
            _lineas_del_modo.append(FUERA_POR_EL_MODO)
        # 139-C2: qué motor es el modelo entregado y cómo se calibró y se eligió su umbral. Mismos rótulos y rutas que el
        # inventario (PROBAST): la ficha y los huecos no pueden decir dos cosas de lo mismo.
        _lineas_del_modo.extend((MOTOR_ENTREGADO, CALIBRACION_Y_UMBRAL))
        # 140-C2: cómo se repartieron las filas del examen (diseño, columnas, hueco, cobertura). Mismo rótulo y ruta que el
        # inventario (PROBAST). Solo si el estudio lo declaró.
        if expediente.diseno_del_reparto is not None:
            _lineas_del_modo.insert(0, REPARTO_POR_DISENO)
        for evidencia in _lineas_del_modo:
            campo = expediente.campo(evidencia.ruta)
            filas.append(_linea(evidencia.rotulo[idioma],
                                texto_del_valor(evidencia, campo, idioma) if campo.hay_dato else None,
                                t["sin_dato"], evidencia.motivo_si_falta[idioma], evidencia.ruta))
    filas.append("")

    filas.append(f"## {t['metricas']}")
    metricas = m.get("metrics") or []
    if metricas:
        # Dos cifras con el MISMO nombre (un estudio publica la estimación de la selección y la
        # de la prueba reservada) no se distinguen sin su partición: ahí, y solo ahí, se escribe.
        _repetidas = {n for n in (x.get("name") for x in metricas if isinstance(x, dict))
                      if sum(1 for y in metricas if isinstance(y, dict) and y.get("name") == n) > 1}
        # Y si ya hay que distinguirlas, TODAS llevan la suya: una sin etiqueta junto a otra con ella se lee mal.
        if _repetidas:
            _repetidas = {x.get("name") for x in metricas if isinstance(x, dict)}
        for metrica in metricas:
            if not isinstance(metrica, dict):
                continue
            sobre = metrica.get("dataset_sha256")
            cola = f" ({t['medido_sobre']} `{str(sobre)[:12]}…`)" if sobre else ""
            if metrica.get("name") in _repetidas and metrica.get("split"):
                cola += f" [{metrica['split']}]"
            filas.append(f"- **{metrica.get('name')}**: {metrica.get('value')}{cola}"
                         f" · `reproduce.json#metrics`")
    else:
        # Y esto es lo que más importa de la sección: un modelo sin métricas
        # publicadas no se presenta con una tabla vacía que parezca un formato.
        filas.append(f"- _{t['sin_dato']}_ ({t['m_sin_metricas']}) "
                     f"· `reproduce.json#metrics`")
    filas.append("")

    # ── EL PERFIL CLÍNICO (109-C3) ────────────────────────────────────────────
    #
    # Solo si el paquete lo trae. Un paquete que no es clínico no necesita
    # veinticuatro líneas diciendo que le falta todo lo clínico: el invariante 7
    # del 109 dice que las funciones clínicas se construyen cuando un piloto las
    # pide, y una sección entera de huecos sobre algo que nadie pidió es ruido,
    # no rigor.
    resueltos, avisos = expediente.resolver(idioma)
    por_clave = {evidencia.clave: campo for evidencia, campo in resueltos}
    if expediente.hay_perfil or expediente.tiene("team_declaration.json"):
        filas.append(f"## {t['perfil']}")
        filas.append("")
        filas.append(f"> {t['perfil_intro']}")
        # (Aquí iba una nota de que el corte se construyó «sin piloto»: es del PROYECTO, no de
        # este paquete, y no va en un documento que recibe quien usa el modelo.)
        filas.append("")
        for dominio in DOMINIOS:
            entradas = [(e, c) for e, c in resueltos if e.dominio == dominio]
            if not entradas:
                continue
            filas.append(f"### {rotulo_de_dominio(dominio, idioma)}")
            filas.append("")
            for evidencia, campo in entradas:
                filas.append(linea_de_campo(evidencia, campo, idioma))
            filas.append("")
        if avisos:
            filas.append(f"**{t['perfil_avisos']}**")
            filas.append("")
            for aviso in avisos:
                filas.append(f"- {aviso}")
            filas.append("")

    filas.append(f"## {t['disponibilidad']} · {t['reproducibilidad']}")
    # LO QUE EL PAQUETE LLEVA, mirado en el disco. La frase era fija, y un paquete de ESTUDIO (138-C2a) no lleva
    # `model.onnx` ni `inference_spec.json`: decía que predice sin instalar nada y no es verdad.
    _faltan = [f"`{n}`" for n in ("model.onnx", "predict.py", "inference_spec.json")
               if not (bundle / n).is_file()]
    if _faltan:
        filas.append("- " + t["paquete_no_usable"].format(faltan=", ".join(_faltan)))
    else:
        filas.append(f"- {t['paquete_usable']}")
    if m.get("reproducible"):
        filas.append(f"- {t['reproducible_si']} · `reproduce.json#reproducible`")
    else:
        # Un manifiesto de ESTUDIO (138-C2a) trae su motivo en los dos idiomas; el del
        # camino clásico, solo `reproducible_reason` (en inglés). Sin el bilingüe, sale el de siempre.
        _bilingue = m.get("reproducible_reason_i18n")
        motivo = ((_bilingue.get(idioma) if isinstance(_bilingue, dict) else None)
                  or m.get("reproducible_reason") or "")
        filas.append(f"- {t['reproducible_no']}: {motivo} "
                     f"· `reproduce.json#reproducible_reason`")
    filas.append("")

    filas.append(f"## {t['no_disponible']}")
    for clave, textos in _NO_DISPONIBLE:
        # Lo que el paquete SÍ trae ya no falta, y decir que falta sería la
        # media verdad tranquilizadora al revés: un hueco inventado.
        if clave is not None and por_clave[clave].hay_dato:
            continue
        rotulo, motivo = textos[idioma]
        if clave == "politica_de_faltantes":
            # Lo que registra la preparación del paquete, si lo trae: no es una política
            # declarada, pero tampoco es «no se puede afirmar qué se hizo con ellos».
            motivo = expediente.ausentes_segun_la_preparacion(idioma) or motivo
        cola = f" · `{POR_CLAVE[clave].ruta}`" if clave is not None else ""
        filas.append(f"- **{rotulo}**: {motivo}{cola}")
    filas.append("")
    # LA ÚLTIMA PUERTA (invariante 6 del 109). Una ficha de reporte no califica
    # el riesgo de sesgo ni en un sentido ni en el otro; si alguna redacción
    # llega a hacerlo, esto revienta en vez de publicarlo.
    return sin_veredicto_de_riesgo("\n".join(filas), origen="tripod.ficha_tripod")
