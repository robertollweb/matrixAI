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

__all__ = ["FichaNoDisponible", "ficha_tripod"]


class FichaNoDisponible(RuntimeError):
    """No hay paquete del que sacar la ficha."""


#: Lo que este informe NO puede rellenar, y por qué. La lista es CERRADA y se
#: imprime SIEMPRE: una ficha con huecos explicados es útil; una con huecos
#: mudos se lee como un descuido, y una rellenada a ojo es peor que las dos.
_NO_DISPONIBLE = {
    "es": [
        ("Datos ausentes", "el core no declara hoy ninguna política de valores "
                           "ausentes, así que no se puede afirmar qué se hizo con ellos"),
        ("Calibración", "no se calcula: el paquete publica exactitud y pérdidas, "
                        "no una curva de calibración"),
        ("Equidad por subgrupos", "no se calcula: haría falta declarar los "
                                  "subgrupos y medir en cada uno"),
        ("Financiación", "es del autor del estudio, no del run"),
        ("Conflictos de interés", "ídem"),
        ("Aprobación ética y registro del estudio", "ídem"),
        ("Uso previsto y población destinataria", "lo escribe una persona; este "
                                                  "informe lo pide, no lo inventa"),
    ],
    "en": [
        ("Missing data", "the core declares no missing-data policy today, so what "
                         "was done with them cannot be stated"),
        ("Calibration", "not computed: the package publishes accuracy and losses, "
                        "not a calibration curve"),
        ("Subgroup fairness", "not computed: it would need the subgroups declared "
                              "and measured one by one"),
        ("Funding", "belongs to the study's author, not to the run"),
        ("Conflicts of interest", "idem"),
        ("Ethical approval and study registration", "idem"),
        ("Intended use and target population", "a person writes this; this report "
                                               "asks for it, it does not invent it"),
    ],
}

_T = {
    "es": {
        "titulo": "Ficha TRIPOD+AI",
        "generado": "Generada por `matrixai report --tripod` desde el manifiesto del "
                    "paquete. Cada línea sale de lo que el run guardó; lo que no se "
                    "sabe se dice.",
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
        "sin_dato": "no disponible",
        "paquete_usable": "El paquete incluye `model.onnx`, `predict.py` e "
                          "`inference_spec.json`: predice con valores crudos sin "
                          "instalar MatrixAI.",
        "reproducible_si": "El paquete se declara reproducible. Se comprueba con "
                           "`matrixai verify <paquete>`.",
        "reproducible_no": "El paquete se declara NO reproducible",
        "m_sin_modo": "el paquete no declara con qué modo se generó",
        "m_sin_tipos": "el run no registró los tipos de las columnas de entrada",
        "m_sin_metricas": "el paquete no publica ninguna métrica",
    },
    "en": {
        "titulo": "TRIPOD+AI record",
        "generado": "Generated by `matrixai report --tripod` from the package "
                    "manifest. Every line comes from what the run recorded; what is "
                    "not known is said.",
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
        "sin_dato": "not available",
        "paquete_usable": "The package ships `model.onnx`, `predict.py` and "
                          "`inference_spec.json`: it predicts from raw values with no "
                          "MatrixAI installed.",
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


def _linea(rotulo: str, valor: Any, sin_dato: str, motivo: str = "") -> str:
    """Una línea de la ficha: su valor, o que no lo hay **y por qué**."""
    if valor in (None, "", [], {}):
        cola = f" ({motivo})" if motivo else ""
        return f"- **{rotulo}**: _{sin_dato}_{cola}"
    if isinstance(valor, (list, tuple)):
        valor = ", ".join(str(v) for v in valor)
    return f"- **{rotulo}**: {valor}"


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

    filas.append(f"## {t['datos']}")
    modo = gen.get("mode")
    filas.append(_linea(t["fuente"],
                        f"`{modo}`" if modo else None, t["sin_dato"],
                        t["m_sin_modo"]))
    filas.append(_linea(t["filas"], (art.get("dataset") or {}).get("rows"), t["sin_dato"]))
    filas.append(_linea(t["huella_datos"], (art.get("dataset") or {}).get("sha256"), t["sin_dato"]))
    # LOS PREDICTORES, DEL MODELO SI EL MANIFIESTO NO LOS TRAE.
    #
    # La primera ficha de la galería decía «Predictores: no disponible» sobre un
    # `.mxai` que declara los cuatro con su rango. El dato estaba **dentro del
    # propio paquete** (`model.mxai`), y no llegaba porque la captura del run
    # —que es la que manda para `generation`— todavía no los registra.
    #
    # Leerlos del modelo empaquetado no es inventarlos: es citar el fichero que
    # viaja al lado, y por eso se dice de dónde salen.
    _predictores = list((gen.get("field_types") or {}).keys()) or _campos_del_modelo(bundle)
    filas.append(_linea(t["predictores"], _predictores or None,
                        t["sin_dato"], t["m_sin_tipos"]))
    filas.append(_linea(t["excluidas"], gen.get("excluded_identifiers"), t["sin_dato"]))
    filas.append(_linea(t["particion"], json.dumps(gen.get("seeds"), sort_keys=True)
                        if gen.get("seeds") else None, t["sin_dato"]))
    if receta_txt:
        filas.append("")
        filas.append(f"**{t['receta']}** (`data_recipe.txt`):")
        filas.append("")
        filas.append("```")
        filas.append(receta_txt)
        filas.append("```")
    filas.append("")

    filas.append(f"## {t['modelo']}")
    filas.append(_linea(t["arquitectura"], (art.get("model") or {}).get("sha256"), t["sin_dato"]))
    filas.append(_linea(t["hiperparametros"], (art.get("training") or {}).get("sha256"), t["sin_dato"]))
    # LAS TRES ÉPOCAS, Y NINGUNA COMO `None`. Imprimir «60 / None / None» pone
    # tres números en fila donde solo hay uno: un valor ausente no es un dato, y
    # aquí encima parece uno. Cada hueco se marca por su cuenta.
    _epocas = [gen.get(k) for k in ("epochs_declared", "epochs_effective", "epochs_ran")]
    filas.append(_linea(
        t["epocas"],
        " / ".join(str(v) if v is not None else f"_{t['sin_dato']}_" for v in _epocas)
        if any(v is not None for v in _epocas) else None, t["sin_dato"]))
    _piezas = [
        f"matrixai {env['matrixai_version']}" if env.get("matrixai_version") else None,
        f"python {(env.get('python') or {}).get('version')}"
        if (env.get("python") or {}).get("version") else None,
        f"{gen['backend']}/{gen['device']}" if gen.get("backend") and gen.get("device")
        else (gen.get("backend") or gen.get("device")),
    ]
    filas.append(_linea(t["entorno"], " · ".join(p for p in _piezas if p) or None,
                        t["sin_dato"]))
    filas.append("")

    filas.append(f"## {t['metricas']}")
    metricas = m.get("metrics") or []
    if metricas:
        for metrica in metricas:
            if not isinstance(metrica, dict):
                continue
            sobre = metrica.get("dataset_sha256")
            cola = f" ({t['medido_sobre']} `{str(sobre)[:12]}…`)" if sobre else ""
            filas.append(f"- **{metrica.get('name')}**: {metrica.get('value')}{cola}")
    else:
        # Y esto es lo que más importa de la sección: un modelo sin métricas
        # publicadas no se presenta con una tabla vacía que parezca un formato.
        filas.append(f"- _{t['sin_dato']}_ ({t['m_sin_metricas']})")
    filas.append("")

    filas.append(f"## {t['disponibilidad']} · {t['reproducibilidad']}")
    filas.append(f"- {t['paquete_usable']}")
    if m.get("reproducible"):
        filas.append(f"- {t['reproducible_si']}")
    else:
        motivo = m.get("reproducible_reason") or ""
        filas.append(f"- {t['reproducible_no']}: {motivo}")
    filas.append("")

    filas.append(f"## {t['no_disponible']}")
    for rotulo, motivo in _NO_DISPONIBLE[("es" if _t(locale) is _T["es"] else "en")]:
        filas.append(f"- **{rotulo}**: {motivo}")
    filas.append("")
    return "\n".join(filas)
