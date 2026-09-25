# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — mide, para cada tarea de `preparar_tareas.py`, tres condiciones
sobre las MISMAS particiones y compara (1) y (2) contra (0) con
`matrixai.estudio.comparaciones.comparar_candidatos`.

Condiciones (las (3) respondedor local y (4) Jev NO están aquí — (3) tiene su
propia sonda de COSTE en `sondear_respondedor.py`, sin condición completa; (4)
espera la decisión de Roberto sobre el intermediario, `[?]` del contrato):

  (0) sin el texto: solo las «otras columnas» de la tarea (LightGBM).
  (1) el embedding de C1 (`paraphrase-multilingual-MiniLM-L12-v2-onnx-int8`,
      decisión 3a) COMO COLUMNAS, junto a las «otras columnas» (LightGBM).
  (2) TF-IDF con un lineal (`scikit-learn`), solo el texto.

**Por qué LightGBM para (0) y (1) y no para (2)**: el encargo dice «el modelo
TABULAR de las tres condiciones: el mismo para todas» — (0) y (1) son las dos
condiciones TABULARES (columnas de números/categorías: las «otras columnas» y,
en (1), el embedding tratado como más columnas). (2) no es tabular en ese
sentido: es su propia condición, «TF-IDF con un lineal», textual y dispersa —
mezclarla con LightGBM sería inventar una condición que el pre-registro no
pide. Se usa LightGBM (`matrixai_engines.motores.arbol_lightgbm.
MotorArbolLightGBM`) porque es «el aprobado» (102-C2, decisión de Roberto del
2026-09-06) y el que ya usa el resto de esta casa para tabular — no hay
motivo para otro.

**Tarea B no tiene «otras columnas»** (el pre-registro lo declara a
propósito: mide la representación, no el estudio con columnas). Su condición
(0) no puede tener CERO columnas —LightGBM no ajusta sin ninguna—, así que se
le da una columna CONSTANTE (`_sin_columnas=0.0`): sin varianza, el árbol no
puede partir por ella y el modelo converge a predecir la prevalencia de
entrenamiento para todas las filas — el baseline «no sé nada» correcto y
honesto, con el MISMO camino de código que las demás tareas.

Escribe `resultado_c30.json` TRAS CADA TAREA (no al final): si algo tarda
demasiado o se para a media tabla, lo ya medido queda en disco.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, "/home/deployer/matrixAI")
sys.path.insert(0, "/home/deployer/matrixai-engines/src")

from matrixai.estudio import ProblemSpec  # noqa: E402
from matrixai.estudio.comparaciones import comparar_candidatos  # noqa: E402
from matrixai.estudio.metricas import Muestra, calcular  # noqa: E402
from matrixai_engines import Presupuesto  # noqa: E402
from matrixai_engines.motores.arbol_lightgbm import MotorArbolLightGBM  # noqa: E402
from matrixai_engines.particiones import Particion  # noqa: E402

SEMILLA = 107030
MARGEN_EQUIVALENCIA = 0.05  # 105-C5, el mismo que cita el contrato (línea ~163 del 105)
HILOS = 4
WALL_SECONDS_LGBM = 300.0
REMUESTRAS_COMPARACION = 1000
DISENO_COMPARACION = "iid"
ESTIMANDO_COMPARACION = "fixed_model_on_population"

PROVEEDOR_EMBEDDING_ID = "paraphrase-multilingual-MiniLM-L12-v2-onnx-int8"

TAREAS: dict[str, dict[str, Any]] = {
    "A": dict(fichero="tarea_a.csv", tipo="binary_classification",
              clases=("True", "Fake"), positive_label="Fake",
              otras_columnas=("topic", "source"), metrica="auroc", idioma="es"),
    "B": dict(fichero="tarea_b.csv", tipo="binary_classification",
              clases=("no_r52", "r52"), positive_label="r52",
              otras_columnas=(), metrica="auroc", idioma="es"),
    "C": dict(fichero="tarea_c.csv", tipo="regression",
              clases=None, positive_label=None,
              otras_columnas=("tipo_code", "procedimiento_code", "cpv_division", "organo"),
              metrica="rmse", idioma="es"),
    "D": dict(fichero="tarea_d.csv", tipo="binary_classification",
              clases=("vigente", "agotada"), positive_label="agotada",
              otras_columnas=("departamento", "rango", "ambito", "anio_publicacion"),
              metrica="auroc", idioma="es"),
}

SALIDA_POR_OMISION = RAIZ / "resultado_c30.json"


def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# lectura de las tareas ya preparadas
# ---------------------------------------------------------------------------

def leer_tarea(nombre: str) -> dict[str, list[dict[str, Any]]]:
    cfg = TAREAS[nombre]
    ruta = RAIZ / "tareas" / cfg["fichero"]
    if not ruta.is_file():
        raise SystemExit(f"falta {ruta}: corre preparar_tareas.py primero")
    por_particion: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    with ruta.open("r", encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            por_particion[fila["particion"]].append(fila)
    return por_particion


# ---------------------------------------------------------------------------
# condición (0)/(1): tabular con LightGBM
# ---------------------------------------------------------------------------

def _valor_objetivo(fila: dict[str, Any], tipo: str) -> Any:
    return float(fila["target"]) if tipo == "regression" else fila["target"]


def _particion_tabular(filas: list[dict[str, Any]], columnas: tuple[str, ...], *,
                       tipo: str, con_target: bool,
                       embeddings: dict[str, list[float]] | None = None) -> Particion:
    features = []
    for fila in filas:
        f = {c: fila[c] for c in columnas if not c.startswith("emb_")}
        if embeddings is not None:
            vec = embeddings[fila["row_id"]]
            for i, v in enumerate(vec):
                f[f"emb_{i}"] = float(v)
        features.append(f)
    target = [_valor_objetivo(fila, tipo) for fila in filas] if con_target else None
    return Particion(row_ids=[fila["row_id"] for fila in filas], features=features, target=target)


def _columnas_condicion0(cfg: dict[str, Any]) -> tuple[str, ...]:
    return cfg["otras_columnas"] if cfg["otras_columnas"] else ("_sin_columnas",)


def _inyectar_columna_constante(filas: list[dict[str, Any]]) -> None:
    for fila in filas:
        fila["_sin_columnas"] = 0.0


def _spec(nombre: str, cfg: dict[str, Any], predictores: tuple[str, ...], *, candidato: str) -> ProblemSpec:
    kwargs: dict[str, Any] = dict(
        problem_id=f"107c30-{nombre}", target="target", task=cfg["tipo"],
        observation_unit="fila", predictors=predictores,
    )
    if cfg["tipo"] != "regression":
        kwargs["classes"] = cfg["clases"]
        kwargs["positive_label"] = cfg["positive_label"]
    return ProblemSpec(**kwargs)


def _digest_particion(row_ids_train: list[str], row_ids_test: list[str]) -> str:
    """El digest de LA PARTICIÓN (qué filas caen en train+dev y cuáles en
    test) -- NO lleva el nombre de la condición. Las tres condiciones (0),
    (1) y (2) leen las MISMAS filas de `tareas/tarea_X.csv` (misma columna
    `particion`), así que su digest tiene que ser IDÉNTICO: es justo lo que
    `comparar_candidatos` comprueba con `protocolo_candidato`/
    `protocolo_baseline` -- si aquí se metiera el nombre de la condición,
    cada comparación saldría `incomparable` por protocolo distinto, aunque
    las filas fueran las mismas fila por fila. (Sabotaje cubierto por
    `tests/test_107_c30_arnes.py`: particiones distintas por condición.)"""
    return hashlib.sha256(
        json.dumps({"train": sorted(row_ids_train), "test": sorted(row_ids_test)},
                   sort_keys=True).encode()
    ).hexdigest()


def _ajustar_y_predecir_lightgbm(train: Particion, val: Particion, test: Particion, spec: ProblemSpec,
                                 *, candidate: str, split_plan_digest: str, semilla: int) -> tuple[Muestra, dict]:
    motor = MotorArbolLightGBM()
    presupuesto = Presupuesto(seed=semilla, wall_seconds=WALL_SECONDS_LGBM, hilos=HILOS)
    t0 = time.perf_counter()
    fit_result, fitted = motor.fit(train, val, spec, presupuesto, candidate=candidate,
                                   split_plan_digest=split_plan_digest)
    if fitted is None:
        raise RuntimeError(f"lightgbm no ajustó ({candidate}): {fit_result.state} {fit_result.reason}")
    t1 = time.perf_counter()
    if spec.task == "regression":
        predicciones = motor.predict(fitted, test)
        muestra = Muestra(task="regression", y_true=tuple(test.target), predictions=tuple(predicciones))
    else:
        scores = motor.decision_scores(fitted, test)
        muestra = Muestra(task=spec.task, y_true=tuple(test.target), classes=spec.classes,
                          positive_label=spec.positive_label, scores=tuple(scores))
    t2 = time.perf_counter()
    tiempos = {"ajuste_s": t1 - t0, "prediccion_s": t2 - t1,
              "n_train": train.n, "n_val": val.n, "n_test": test.n,
              "resources": fit_result.resources.a_json() if fit_result.resources else None}
    return muestra, tiempos


# ---------------------------------------------------------------------------
# condición (1): el embedding como columnas
# ---------------------------------------------------------------------------

def cargar_proveedor_embedding():
    from matrixai_engines.embeddings import ProveedorDeTexto
    return ProveedorDeTexto.cargar(PROVEEDOR_EMBEDDING_ID, hilos=HILOS)


def codificar_textos(proveedor, filas: list[dict[str, Any]], *, idioma: str, lote: int = 32) -> dict[str, list[float]]:
    textos = [fila["texto"] for fila in filas]
    vectores = proveedor.codificar(textos, idioma=idioma, lote=lote)
    salida = {}
    for fila, vec in zip(filas, vectores):
        salida[fila["row_id"]] = [float(v) for v in vec]
    return salida


# ---------------------------------------------------------------------------
# condición (2): TF-IDF + lineal
# ---------------------------------------------------------------------------

def _condicion2_tfidf_lineal(train: list[dict[str, Any]], val: list[dict[str, Any]],
                             test: list[dict[str, Any]], cfg: dict[str, Any],
                             *, semilla: int) -> tuple[Muestra, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge

    t0 = time.perf_counter()
    entrenamiento = train + val  # el lineal no necesita partición de validación propia
    vectorizador = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizador.fit_transform([f["texto"] for f in entrenamiento])
    X_test = vectorizador.transform([f["texto"] for f in test])
    t1 = time.perf_counter()

    if cfg["tipo"] == "regression":
        y_train = [float(f["target"]) for f in entrenamiento]
        modelo = Ridge(alpha=1.0, random_state=semilla)
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        muestra = Muestra(task="regression", y_true=tuple(float(f["target"]) for f in test),
                          predictions=tuple(float(p) for p in predicciones))
    else:
        y_train = [f["target"] for f in entrenamiento]
        modelo = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=semilla)
        modelo.fit(X_train, y_train)
        indice_positiva = list(modelo.classes_).index(cfg["positive_label"])
        proba = modelo.predict_proba(X_test)[:, indice_positiva]
        muestra = Muestra(task=cfg["tipo"], y_true=tuple(f["target"] for f in test),
                          classes=cfg["clases"], positive_label=cfg["positive_label"],
                          scores=tuple(float(p) for p in proba))
    t2 = time.perf_counter()
    tiempos = {"vectorizacion_s": t1 - t0, "ajuste_y_prediccion_s": t2 - t1,
              "n_train": len(entrenamiento), "n_test": len(test),
              "n_terminos_tfidf": X_train.shape[1]}
    return muestra, tiempos


# ---------------------------------------------------------------------------
# comparación frente a (0)
# ---------------------------------------------------------------------------

def _comparar(metric_id: str, candidato: Muestra, baseline: Muestra, *, semilla: int,
             protocolo_candidato: str, protocolo_baseline: str) -> dict[str, Any]:
    resultado = comparar_candidatos(
        metric_id, candidato, baseline, diseno=DISENO_COMPARACION, estimando=ESTIMANDO_COMPARACION,
        semilla=semilla, margen_equivalencia=MARGEN_EQUIVALENCIA, remuestras=REMUESTRAS_COMPARACION,
        protocolo_candidato=protocolo_candidato, protocolo_baseline=protocolo_baseline,
    )
    return resultado.a_json()


def _metrica_puntual(metric_id: str, muestra: Muestra) -> float | None:
    r = calcular(metric_id, muestra)
    return r.value


# ---------------------------------------------------------------------------
# una tarea entera
# ---------------------------------------------------------------------------

def evaluar_tarea(nombre: str, *, incluir_embedding: bool, semilla: int = SEMILLA) -> dict[str, Any]:
    cfg = TAREAS[nombre]
    particiones = leer_tarea(nombre)
    train, val, test = particiones["train"], particiones["dev"], particiones["test"]
    metric_id = cfg["metrica"]

    resultado: dict[str, Any] = {
        "tarea": nombre, "metrica": metric_id,
        "n_train": len(train), "n_dev": len(val), "n_test": len(test),
        "condiciones": {},
    }

    # El digest de LA PARTICIÓN, una sola vez para toda la tarea: (0), (1) y
    # (2) leen las MISMAS filas (misma columna `particion` del CSV de
    # `preparar_tareas.py`), así que su `split_plan_digest`/`protocolo_*`
    # tiene que ser IDÉNTICO -- si llevara el nombre de la condición,
    # `comparar_candidatos` marcaría CADA comparación `incomparable` por
    # protocolo distinto (ver `_digest_particion`), aunque las filas fueran
    # las mismas fila por fila. (Sabotaje cubierto en
    # `tests/test_107_c30_arnes.py`: particiones distintas por condición.)
    digest = _digest_particion([f["row_id"] for f in train + val], [f["row_id"] for f in test])

    # ---- condición (0): sin texto ----------------------------------------
    columnas0 = _columnas_condicion0(cfg)
    if columnas0 == ("_sin_columnas",):
        for filas in (train, val, test):
            _inyectar_columna_constante(filas)
    spec0 = _spec(nombre, cfg, columnas0, candidato="condicion_0_sin_texto")
    p_train0 = _particion_tabular(train, columnas0, tipo=cfg["tipo"], con_target=True)
    p_val0 = _particion_tabular(val, columnas0, tipo=cfg["tipo"], con_target=True)
    p_test0 = _particion_tabular(test, columnas0, tipo=cfg["tipo"], con_target=True)
    muestra0, tiempos0 = _ajustar_y_predecir_lightgbm(
        p_train0, p_val0, p_test0, spec0, candidate="condicion_0_sin_texto",
        split_plan_digest=digest, semilla=semilla)
    resultado["condiciones"]["0_sin_texto"] = {
        "columnas": list(columnas0), "metrica_puntual": _metrica_puntual(metric_id, muestra0),
        "tiempos": tiempos0, "split_plan_digest": digest,
    }

    # ---- condición (2): TF-IDF + lineal (no necesita el embedding) -------
    muestra2, tiempos2 = _condicion2_tfidf_lineal(train, val, test, cfg, semilla=semilla)
    resultado["condiciones"]["2_tfidf_lineal"] = {
        "metrica_puntual": _metrica_puntual(metric_id, muestra2), "tiempos": tiempos2,
        "split_plan_digest": digest,
    }
    resultado["comparaciones"] = {
        "2_vs_0": _comparar(metric_id, muestra2, muestra0, semilla=semilla,
                            protocolo_candidato=digest, protocolo_baseline=digest),
    }

    # ---- condición (1): embedding como columnas ---------------------------
    if incluir_embedding:
        proveedor = cargar_proveedor_embedding()
        t_emb0 = time.perf_counter()
        emb_train = codificar_textos(proveedor, train, idioma=cfg["idioma"])
        emb_val = codificar_textos(proveedor, val, idioma=cfg["idioma"])
        emb_test = codificar_textos(proveedor, test, idioma=cfg["idioma"])
        t_emb1 = time.perf_counter()
        dim = proveedor.dimension
        columnas1 = tuple(cfg["otras_columnas"]) + tuple(f"emb_{i}" for i in range(dim))
        spec1 = _spec(nombre, cfg, columnas1, candidato="condicion_1_embedding")
        p_train1 = _particion_tabular(train, columnas1, tipo=cfg["tipo"], con_target=True, embeddings=emb_train)
        p_val1 = _particion_tabular(val, columnas1, tipo=cfg["tipo"], con_target=True, embeddings=emb_val)
        p_test1 = _particion_tabular(test, columnas1, tipo=cfg["tipo"], con_target=True, embeddings=emb_test)
        muestra1, tiempos1 = _ajustar_y_predecir_lightgbm(
            p_train1, p_val1, p_test1, spec1, candidate="condicion_1_embedding",
            split_plan_digest=digest, semilla=semilla)
        tiempos1["codificacion_embedding_s"] = t_emb1 - t_emb0
        tiempos1["proveedor"] = proveedor.id
        tiempos1["dimension"] = dim
        resultado["condiciones"]["1_embedding"] = {
            "columnas": list(columnas1), "metrica_puntual": _metrica_puntual(metric_id, muestra1),
            "tiempos": tiempos1, "split_plan_digest": digest,
        }
        resultado["comparaciones"]["1_vs_0"] = _comparar(
            metric_id, muestra1, muestra0, semilla=semilla,
            protocolo_candidato=digest, protocolo_baseline=digest)
    else:
        resultado["condiciones"]["1_embedding"] = {"omitida": True, "motivo": "--sin-embedding"}
        resultado["comparaciones"]["1_vs_0"] = None

    return resultado


# ---------------------------------------------------------------------------
# --estimar: aritmética a partir de una calibración pequeña y REAL
# ---------------------------------------------------------------------------

def _calibrar_lightgbm(*, semilla: int) -> float:
    """Segundos por fila (train+test) de un LightGBM con 5 columnas
    numéricas, medido sobre datos sintéticos pequeños (400 filas) -- no es
    "suponer": es una medida real, solo que barata y no la tarea de verdad."""
    import random as _random
    rng = _random.Random(semilla)
    filas = [{"row_id": str(i),
             "c0": rng.random(), "c1": rng.random(), "c2": rng.random(),
             "c3": rng.random(), "c4": rng.random(),
             "target": "si" if rng.random() < 0.4 else "no"} for i in range(400)]
    columnas = ("c0", "c1", "c2", "c3", "c4")
    train_f, test_f = filas[:300], filas[300:]
    p_train = _particion_tabular(train_f, columnas, tipo="binary_classification", con_target=True)
    p_test = _particion_tabular(test_f, columnas, tipo="binary_classification", con_target=True)
    spec = ProblemSpec(problem_id="calibracion", target="target", task="binary_classification",
                       observation_unit="fila", predictors=columnas, classes=("no", "si"),
                       positive_label="si")
    t0 = time.perf_counter()
    _, tiempos = _ajustar_y_predecir_lightgbm(p_train, p_train, p_test, spec, candidate="calibracion",
                                              split_plan_digest="c" * 64, semilla=semilla)
    total = time.perf_counter() - t0
    return total / (len(train_f) + len(test_f))


def _calibrar_tfidf(*, semilla: int) -> float:
    """Segundos por fila de TF-IDF+lineal, sobre 400 textos cortos sintéticos."""
    import random as _random
    rng = _random.Random(semilla)
    palabras = ["contrato", "servicio", "suministro", "obra", "ayuntamiento", "licitacion",
               "plazo", "importe", "expediente", "adjudicacion"]
    filas = [{"texto": " ".join(rng.choices(palabras, k=12)),
             "target": "si" if rng.random() < 0.4 else "no"} for i in range(400)]
    train_f, test_f = filas[:300], filas[300:]
    cfg = {"tipo": "binary_classification", "clases": ("no", "si"), "positive_label": "si"}
    t0 = time.perf_counter()
    _, _ = _condicion2_tfidf_lineal(train_f, [], test_f, cfg, semilla=semilla)
    total = time.perf_counter() - t0
    return total / len(filas)


def _calibrar_embedding(*, idioma: str = "es") -> float:
    """Segundos por texto codificado, sobre 20 textos REALES de la tarea A
    (si están preparados) o sintéticos si no."""
    ruta_a = RAIZ / "tareas" / "tarea_a.csv"
    textos = []
    if ruta_a.is_file():
        with ruta_a.open(encoding="utf-8", newline="") as f:
            for i, fila in enumerate(csv.DictReader(f)):
                if i >= 20:
                    break
                textos.append(fila["texto"])
    if not textos:
        textos = ["Texto de calibración número %d, sin datos reales preparados todavía." % i
                  for i in range(20)]
    proveedor = cargar_proveedor_embedding()
    t0 = time.perf_counter()
    proveedor.codificar(textos, idioma=idioma, lote=32)
    total = time.perf_counter() - t0
    return total / len(textos)


def estimar(*, semilla: int = SEMILLA) -> dict[str, Any]:
    print("calibrando lightgbm...", file=sys.stderr)
    tasa_lgbm = _calibrar_lightgbm(semilla=semilla)
    print("calibrando tf-idf+lineal...", file=sys.stderr)
    tasa_tfidf = _calibrar_tfidf(semilla=semilla)
    print("calibrando el embedding (20 textos reales de A, si existen)...", file=sys.stderr)
    tasa_embedding = _calibrar_embedding()

    tabla: dict[str, Any] = {
        "calibracion": {
            "tasa_lightgbm_s_por_fila": tasa_lgbm,
            "tasa_tfidf_lineal_s_por_fila": tasa_tfidf,
            "tasa_embedding_s_por_texto": tasa_embedding,
            "medida_sobre": "400 filas sintéticas (lightgbm/tfidf) y 20 textos reales de la tarea A "
                            "(embedding, si `tareas/tarea_a.csv` ya existe) -- NO es una suposición: "
                            "es una medición real, pequeña y barata, para no tener que 'suponer' la "
                            "aritmética que pide --estimar.",
        },
        "tareas": {},
    }
    total_general = 0.0
    for nombre, cfg in TAREAS.items():
        ruta = RAIZ / "tareas" / cfg["fichero"]
        if not ruta.is_file():
            tabla["tareas"][nombre] = {"error": f"falta {ruta}: corre preparar_tareas.py primero"}
            continue
        particiones = leer_tarea(nombre)
        n_train, n_val, n_test = len(particiones["train"]), len(particiones["dev"]), len(particiones["test"])
        n_total = n_train + n_val + n_test

        t0 = n_total * tasa_lgbm  # condición 0
        t2 = n_total * tasa_tfidf  # condición 2 (vectoriza train+dev+test)
        t1_emb = n_total * tasa_embedding  # codificar TODAS las filas
        t1_lgbm = n_total * tasa_lgbm  # con más columnas (dim ~384): mismo orden de magnitud,
        # LightGBM parte por gain y el número de columnas afecta menos que el número de filas
        # (declarado como simplificación: no se recalibra por columna, HAY que decirlo).
        t1 = t1_emb + t1_lgbm
        # comparar_candidatos: remuestreo de 1000 iteraciones sobre ~n_test filas, en Python puro;
        # medido aparte (no en esta calibración) que cuesta del orden de baja resolución por
        # comparación -- se declara con una constante fija y conservadora, no una calibración:
        t_comparacion = 2 * 2.0  # 2 comparaciones (1 vs 0, 2 vs 0), ~2s cada una a esta escala

        total_tarea = t0 + t1 + t2 + t_comparacion
        total_general += total_tarea
        tabla["tareas"][nombre] = {
            "n_train": n_train, "n_dev": n_val, "n_test": n_test, "n_total": n_total,
            "estimado_s": {
                "condicion_0": round(t0, 2),
                "condicion_1_codificar_embedding": round(t1_emb, 2),
                "condicion_1_lightgbm": round(t1_lgbm, 2),
                "condicion_1_total": round(t1, 2),
                "condicion_2_tfidf_lineal": round(t2, 2),
                "comparaciones_bootstrap": round(t_comparacion, 2),
                "total_tarea": round(total_tarea, 2),
            },
            "aritmetica": f"({n_total} filas) x tasa -- condicion_0 y condicion_1_lightgbm usan "
                          f"tasa_lightgbm={tasa_lgbm:.6f}s/fila; condicion_1_codificar_embedding usa "
                          f"tasa_embedding={tasa_embedding:.6f}s/texto; condicion_2 usa "
                          f"tasa_tfidf_lineal={tasa_tfidf:.6f}s/fila; mas {t_comparacion:.1f}s fijos "
                          f"de bootstrap (2 comparaciones).",
        }
    tabla["total_general_s"] = round(total_general, 2)
    tabla["total_general_min"] = round(total_general / 60.0, 2)
    return tabla


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimar", action="store_true", help="duración estimada, con su aritmética, sin medir")
    ap.add_argument("--solo", choices=list(TAREAS), help="mide solo esta tarea")
    ap.add_argument("--sin-embedding", action="store_true", help="omite la condición (1), el embedding")
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_OMISION)
    ns = ap.parse_args()

    if ns.estimar:
        tabla = estimar()
        destino = ns.salida if ns.salida != SALIDA_POR_OMISION else RAIZ / "estimacion_c30.json"
        destino.write_text(json.dumps(tabla, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(tabla, indent=2, ensure_ascii=False))
        print(f"\nescrito en {destino}", file=sys.stderr)
        return 0

    tareas_a_medir = [ns.solo] if ns.solo else list(TAREAS)
    salida: dict[str, Any] = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0",
        "semilla": SEMILLA, "margen_equivalencia": MARGEN_EQUIVALENCIA,
        "diseno_comparacion": DISENO_COMPARACION, "estimando_comparacion": ESTIMANDO_COMPARACION,
        "remuestras_comparacion": REMUESTRAS_COMPARACION,
        "proveedor_embedding": PROVEEDOR_EMBEDDING_ID,
        "script_sha256": sha256_de(Path(__file__)),
        "regla_idioma_sha256": sha256_de(RAIZ / "regla_idioma.json"),
        "generado": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tareas": {},
    }
    if ns.salida.is_file():
        try:
            previo = json.loads(ns.salida.read_text(encoding="utf-8"))
            salida["tareas"].update(previo.get("tareas", {}))
        except (json.JSONDecodeError, OSError):
            pass

    for nombre in tareas_a_medir:
        print(f"=== midiendo tarea {nombre} ===", file=sys.stderr)
        t0 = time.perf_counter()
        resultado = evaluar_tarea(nombre, incluir_embedding=not ns.sin_embedding)
        resultado["duracion_total_s"] = time.perf_counter() - t0
        salida["tareas"][nombre] = resultado
        ns.salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(resultado, indent=2, ensure_ascii=False), file=sys.stderr)
        print(f"--- tarea {nombre}: {resultado['duracion_total_s']:.1f}s, escrito en {ns.salida} ---",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
