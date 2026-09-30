# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.1c — ¿se puede exportar a ONNX el candidato de solo texto?

El diseño de C3.1 (contrato 107, pieza 3) lo dejó escrito ANTES de construir:
el TF-IDF se exporta con `skl2onnx` SOLO si una prueba de PARIDAD sobre los
textos reales de A, B y C da las mismas probabilidades que scikit-learn —el
tokenizador de ONNX no es el de scikit-learn, y ese es el riesgo—; si no hay
paridad, el candidato gana igual pero su paquete sale «sin exportación», apagado
y con su motivo.

Esto mide esa paridad. Por cada tarea: el vectorizador y el lineal DEL MOTOR
(`construir_vectorizador_tfidf`, `construir_lineal_de_texto`: los mismos que el
estudio), ajustados con la partición `train`, convertidos con `skl2onnx` y
ejecutados con onnxruntime sobre la partición `test`, frente a scikit-learn.
Y un CONTROL del instrumento: el vectorizador solo, sobre siete frases escritas
a mano, para ver QUÉ se rompe (acentos, puntuación) y que no es la sonda.

Local y sin red; B es clínica, pero aquí no sale de la máquina.

    python3 benchmarks/texto_107c30/paridad_onnx_c31c.py   # ~20 s
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
TAREAS = RAIZ / "tareas"
SALIDA = RAIZ / "paridad_onnx_c31c.json"
#: A (binaria, inglés), B (binaria, castellano, clínica) y C (regresión, castellano).
LETRAS = ("a", "b", "c")
#: Lo que el diseño llama «las mismas probabilidades»: la tolerancia de las
#: demás pruebas de paridad de exportación del repo (float32 frente a float64).
TOLERANCIA = 1e-4

FRASES_DE_CONTROL = [
    "The cat sat on the mat. The cat is happy!",
    "Hello world, hello again; world peace?",
    "La información pública del órgano de contratación",
    "Año 2025: servicio de limpieza (lote 1)",
    "don't stop-believing e-mail U.S.A. 3.5 kg",
    "mat cat cat sat world",
    "órgano información año servicio",
]


def _leer(letra: str) -> dict[str, list[dict[str, str]]]:
    csv.field_size_limit(10 ** 9)
    with open(TAREAS / f"tarea_{letra}.csv", encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    por_particion: dict[str, list[dict[str, str]]] = {}
    for fila in filas:
        por_particion.setdefault(fila["particion"], []).append(fila)
    return por_particion


def _es_regresion(objetivos: list[str]) -> bool:
    try:
        valores = [float(v) for v in objetivos]
    except ValueError:
        return False
    return len(set(valores)) > 20


def _convertir(modelo, *, clasificacion: bool):
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType
    opciones = {id(modelo.steps[-1][1]): {"zipmap": False}} if clasificacion else None
    return to_onnx(modelo, initial_types=[("texto", StringTensorType([None, 1]))],
                   options=opciones, target_opset=17)


def medir_tarea(letra: str) -> dict:
    import numpy as np
    import onnxruntime as ort
    from sklearn.pipeline import Pipeline

    from matrixai_engines.motores.texto_tfidf import construir_lineal_de_texto, construir_vectorizador_tfidf

    particiones = _leer(letra)
    train, test = particiones["train"], particiones["test"]
    objetivos = [f["target"] for f in train]
    regresion = _es_regresion(objetivos)
    tarea = "regression" if regresion else "clasificacion"
    y = np.array([float(v) for v in objetivos]) if regresion else np.array(objetivos)
    modelo = Pipeline([("tfidf", construir_vectorizador_tfidf()),
                       ("lineal", construir_lineal_de_texto(tarea, semilla=0))])
    modelo.fit([f["texto"] or "" for f in train], y)
    textos = [f["texto"] or "" for f in test]
    try:
        onx = _convertir(modelo, clasificacion=not regresion)
    except Exception as error:  # noqa: BLE001 — un fallo de conversión ES un resultado
        return {"tarea": tarea, "conversion": "falla", "error": f"{type(error).__name__}: {error}"[:400]}
    sesion = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    salida = sesion.run(None, {"texto": np.array(textos, dtype=object).reshape(-1, 1)})
    if regresion:
        a, b = modelo.predict(textos), salida[0].ravel()
        dif = np.abs(a - b)
        return {"tarea": tarea, "n_test": len(textos), "diferencia_maxima": float(dif.max()),
                "diferencia_relativa_maxima": float((dif / (np.abs(a) + 1e-9)).max()),
                "filas_fuera_de_tolerancia": int((dif > TOLERANCIA).sum()),
                "paridad": bool(dif.max() <= TOLERANCIA)}
    a, b = modelo.predict_proba(textos), salida[1]
    dif = np.abs(a - b).max(axis=1)
    return {"tarea": tarea, "clases": len(modelo.classes_), "n_test": len(textos),
            "diferencia_maxima": float(dif.max()),
            "filas_fuera_de_tolerancia": int((dif > TOLERANCIA).sum()),
            "filas_con_mas_de_0_01": int((dif > 0.01).sum()),
            "acuerdo_de_clase": float(np.mean(a.argmax(1) == b.argmax(1))),
            "paridad": bool(dif.max() <= TOLERANCIA)}


def control_del_instrumento() -> dict:
    """El vectorizador SOLO, sin el lineal, sobre frases escritas a mano: con y
    sin la opción `tokenexp` de skl2onnx (la que dice acercar el tokenizador)."""
    import numpy as np
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType
    from sklearn.feature_extraction.text import TfidfVectorizer

    textos = FRASES_DE_CONTROL * 3
    resultado = {}
    for nombre, opciones in (("por_omision", None), ("tokenexp_de_sklearn", {"tokenexp": r"\b\w\w+\b"})):
        vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(textos)
        onx = to_onnx(vectorizador, initial_types=[("t", StringTensorType([None, 1]))],
                      options={id(vectorizador): opciones} if opciones else None, target_opset=17)
        sesion = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        b = sesion.run(None, {"t": np.array(FRASES_DE_CONTROL, dtype=object).reshape(-1, 1)})[0]
        a = vectorizador.transform(FRASES_DE_CONTROL).toarray()
        resultado[nombre] = {frase: round(float(d), 4)
                             for frase, d in zip(FRASES_DE_CONTROL, np.abs(a - b).max(axis=1))}
    return resultado


def main() -> int:
    import onnxruntime
    import skl2onnx
    import sklearn

    tareas = {letra.upper(): medir_tarea(letra) for letra in LETRAS}
    hay_paridad = all(t.get("paridad") is True for t in tareas.values())
    resultado = {
        "corte": "107-C3.1c",
        "que_mide": "paridad de probabilidades (o valores) entre scikit-learn y su conversión ONNX "
                    "del candidato de solo texto, sobre la partición test de A, B y C",
        "tolerancia": TOLERANCIA,
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "skl2onnx": skl2onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "tareas": tareas,
        "control_del_instrumento": control_del_instrumento(),
        "hay_paridad": hay_paridad,
        "decision_del_diseno": ("se exporta a ONNX" if hay_paridad else
                                "SIN exportación ONNX: el candidato gana igual y su paquete sale "
                                "sin ONNX, apagado y con su motivo (contrato 107, C3.1, pieza 3)"),
    }
    SALIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: resultado[k] for k in ("tareas", "hay_paridad")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
