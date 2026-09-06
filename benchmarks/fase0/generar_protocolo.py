#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C1 — cómo se construyó `protocolo_exploratorio.json`, y cómo repetirlo.

**No es una función pura que reproduzca el MISMO protocolo si se vuelve a
correr.** OpenML es un catálogo vivo: un dataset puede desactivarse, otro
nuevo puede aparecer, y el algoritmo de selección aquí abajo es determinista
SOBRE UNA FOTO del catálogo, no sobre el catálogo en sí. Por eso el protocolo
registrado (`protocolo_exploratorio.json`, con su `digest_sha256`) es el
documento que manda — este script es la RECETA que lo produjo, para que
quien audite pueda repetir el razonamiento y las llamadas reales a la API,
no para que una segunda ejecución tenga que dar bit a bit lo mismo.

Cada paso hace una llamada HTTP real a `openml.org` (medido, no simulado):
descargar los tres estudios, pedir las «qualities» (filas, columnas,
faltantes, tamaño de la clase minoritaria) de cada `data_id` candidato,
elegir 40 que cumplan la cobertura obligatoria del anexo C del documento 100,
y descargar+hashear el ARFF de cada uno elegido.

Uso:
    python3 benchmarks/fase0/generar_protocolo.py --salida protocolo_exploratorio.json
    (tarda varios minutos: ~150 llamadas de metadatos + 40 descargas de ARFF,
    ~260 MB en total — los ARFF NO se comitean al repo, invariante 4 del
    runbook de publicación: nada de blobs grandes en un repo git)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
if str(_RAIZ_DEL_CORE) not in sys.path:
    sys.path.insert(0, str(_RAIZ_DEL_CORE))

from benchmarks.fase0.protocolo import (  # noqa: E402
    DatasetRegistrado,
    DisenoDeParticion,
    Motor,
    PresupuestoPorCubo,
    ProtocoloExploratorio,
    ReglaDeCierre,
    calcular_coste,
)

#: Los tres estudios de origen (medido en el anexo C del 100: ids reales,
#: consultados el 2026-09-04/06). TabArena (457) y CTR23 (353) quedan fuera
#: A PROPÓSITO — decisión confirmada por Roberto: alcance mínimo, sin
#: TabArena ni pistas adicionales para esta primera Fase 0.
ESTUDIO_CC18 = 99
ESTUDIO_AMLB_CLASIFICACION = 271
ESTUDIO_AMLB_REGRESION = 269

#: Licencias que invariante 10 del 101 acepta sin más comprobación: dominio
#: público / uso libre, sin condición de atribución obligatoria ni copyleft.
#: Medido el 2026-09-06: de 145 candidatos, 143 caían aquí y UNO
#: («black_friday», `licence: "GPL-1"` — GPL es una licencia de SOFTWARE,
#: casi con toda seguridad un metadato mal puesto al subir el dataset, pero
#: «una nota interna no exime de las condiciones aplicables» corta en los
#: dos sentidos) se sustituyó a mano por `house_sales` (CC0). Este filtro
#: automatiza esa comprobación para que una repetición futura no vuelva a
#: colar el mismo caso sin que alguien lo note.
LICENCIAS_PERMITIDAS = {"public", "cc0", "public domain", "cc0 public domain"}

CUOTA_BINARIA = {"pequeno": 8, "mediano": 8, "grande": 4}
CUOTA_MULTICLASE = {"pequeno": 4, "mediano": 4, "grande": 2}
CUOTA_REGRESION = {"pequeno": 3, "mediano": 3, "grande": 4}
#: Mínimos de cobertura obligatoria, EXACTOS del anexo C (§2.2).
MINIMO_FALTANTES = 10
MINIMO_ALTA_CARDINALIDAD = 12
MINIMO_DESBALANCEADOS = 8
MINIMO_SOLO_NUMERICAS = 6
MINIMO_MULTICLASE_5_CLASES = 4
MINIMO_MULTICLASE_CLASE_RARA = 2


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "matrixai-fase0/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _bucket(n_filas: float | None) -> str | None:
    if n_filas is None:
        return None
    if 500 <= n_filas <= 2_000:
        return "pequeno"
    if 2_000 < n_filas <= 20_000:
        return "mediano"
    if 20_000 < n_filas <= 200_000:
        return "grande"
    return None


def _ids_de_estudio(study_id: int) -> list[int]:
    d = _get_json(f"https://www.openml.org/api/v1/json/study/{study_id}")
    ids = d["study"].get("data", {}).get("data_id", [])
    return [int(i) for i in ids] if isinstance(ids, list) else [int(ids)]


def _qualities(data_id: int) -> dict[str, float]:
    d = _get_json(f"https://www.openml.org/api/v1/json/data/qualities/{data_id}")
    out: dict[str, float] = {}
    for q in d["data_qualities"]["quality"]:
        try:
            out[q["name"]] = float(q["value"])
        except (KeyError, ValueError, TypeError):
            pass
    return out


def _info(data_id: int) -> dict:
    return _get_json(f"https://www.openml.org/api/v1/json/data/{data_id}")["data_set_description"]


def _candidato(data_id: int, es_clasificacion: bool) -> dict | None:
    q = _qualities(data_id)
    info = _info(data_id)
    n = q.get("NumberOfInstances")
    bucket = _bucket(n)
    if bucket is None or info.get("status") != "active":
        return None
    licencia = (info.get("licence") or "").strip().lower()
    if licencia not in LICENCIAS_PERMITIDAS:
        return None
    simbolicas = q.get("NumberOfSymbolicFeatures", 0) or 0
    # En clasificación, la COLUMNA OBJETIVO nominal ya cuenta como 1 simbólica
    # (medido: breast-w, todo predictor numérico, da NumberOfSymbolicFeatures=1)
    # — «solo numéricas» exige <=1 en clasificación y ==0 en regresión.
    solo_numericas = simbolicas <= 1 if es_clasificacion else simbolicas == 0
    minoria = q.get("MinorityClassSize")
    return {
        "data_id": data_id, "nombre": info.get("name"), "version": info.get("version"),
        "url": info.get("url"), "licencia": info.get("licence"),
        "objetivo": info.get("default_target_attribute"),
        "n_filas": int(n), "n_columnas": int(q.get("NumberOfFeatures") or 0),
        "n_clases": q.get("NumberOfClasses"),
        "bucket": bucket,
        "tiene_faltantes": bool(q.get("NumberOfMissingValues", 0)),
        "alta_cardinalidad": bool(simbolicas >= 1 and q.get("NumberOfFeatures")
                                  and simbolicas / q["NumberOfFeatures"] > 0.3),
        "desbalanceado": bool(minoria is not None and n and (minoria / n) <= 0.10),
        "clase_muy_minoritaria": bool(minoria is not None and n and (minoria / n) < 0.02),
        "solo_numericas": solo_numericas,
        "binaria": q.get("NumberOfClasses") == 2,
        "multiclase": bool(q.get("NumberOfClasses") and q["NumberOfClasses"] >= 3),
    }


def _elegir(pool: list[dict], cuotas: dict[str, int], ya_elegidos: set[int],
           necesita_numericas: int = 0, necesita_rara: int = 0) -> list[dict]:
    """Selección voraz por cubo: prioriza cerrar los huecos de cobertura
    (solo-numéricas, clase muy minoritaria) antes que faltantes/alta-
    cardinalidad/desbalanceo, dentro de cada cubo de tamaño."""
    por_bucket: dict[str, list[dict]] = {"pequeno": [], "mediano": [], "grande": []}
    for r in pool:
        if r["data_id"] not in ya_elegidos:
            por_bucket[r["bucket"]].append(r)
    elegidos: list[dict] = []
    numericas_puestas = raras_puestas = 0
    for bucket, cupo in cuotas.items():
        candidatos = list(por_bucket[bucket])

        def puntuar(r: dict) -> int:
            s = 0
            if numericas_puestas < necesita_numericas and r["solo_numericas"]:
                s -= 10
            if raras_puestas < necesita_rara and r["clase_muy_minoritaria"]:
                s -= 8
            s -= int(r["tiene_faltantes"]) + int(r["alta_cardinalidad"]) + int(r["desbalanceado"])
            return s

        candidatos.sort(key=puntuar)
        elegidos_bucket = candidatos[:cupo]
        numericas_puestas += sum(1 for r in elegidos_bucket if r["solo_numericas"])
        raras_puestas += sum(1 for r in elegidos_bucket if r["clase_muy_minoritaria"])
        elegidos.extend(elegidos_bucket)
    return elegidos


def _sellar(seleccion: list[dict]) -> set[int]:
    """8 de 40 sellados, por posición EQUIESPACIADA sobre la lista ordenada
    por `data_id` — determinista, fijado antes de ver ningún resultado
    (anti-favoritismo, anexo C §2.3 punto 2)."""
    binaria = sorted((r for r in seleccion if r["binaria"]), key=lambda r: r["data_id"])
    multiclase = sorted((r for r in seleccion if r["multiclase"]), key=lambda r: r["data_id"])
    regresion = sorted((r for r in seleccion if not r["binaria"] and not r["multiclase"]),
                       key=lambda r: r["data_id"])

    def n_equiespaciados(pool: list[dict], n: int) -> list[int]:
        if n >= len(pool):
            return [r["data_id"] for r in pool]
        paso = len(pool) / n
        return [pool[int(i * paso)]["data_id"] for i in range(n)]

    return (set(n_equiespaciados(binaria, 4)) | set(n_equiespaciados(multiclase, 2))
           | set(n_equiespaciados(regresion, 2)))


def seleccionar_40() -> list[dict]:
    ids_clf = sorted(set(_ids_de_estudio(ESTUDIO_CC18)) | set(_ids_de_estudio(ESTUDIO_AMLB_CLASIFICACION)))
    ids_reg = sorted(set(_ids_de_estudio(ESTUDIO_AMLB_REGRESION)))

    clf = [c for c in (_candidato(i, es_clasificacion=True) for i in ids_clf) if c and c["objetivo"]]
    reg = [c for c in (_candidato(i, es_clasificacion=False) for i in ids_reg) if c and c["objetivo"]]

    def dedup(lst: list[dict]) -> list[dict]:
        mejor: dict[str, dict] = {}
        for r in lst:
            clave = r["nombre"]
            if clave not in mejor or (r["version"] or 0) > (mejor[clave]["version"] or 0):
                mejor[clave] = r
        return list(mejor.values())

    clf, reg = dedup(clf), dedup(reg)
    binaria_pool = [r for r in clf if r["binaria"]]
    multiclase_pool = [r for r in clf if r["multiclase"]]

    elegidos_ids: set[int] = set()
    sel_binaria = _elegir(binaria_pool, CUOTA_BINARIA, elegidos_ids, necesita_numericas=3)
    elegidos_ids |= {r["data_id"] for r in sel_binaria}
    sel_multiclase = _elegir(multiclase_pool, CUOTA_MULTICLASE, elegidos_ids,
                             necesita_numericas=1, necesita_rara=2)
    elegidos_ids |= {r["data_id"] for r in sel_multiclase}
    sel_regresion = _elegir(reg, CUOTA_REGRESION, elegidos_ids, necesita_numericas=2)

    seleccion = sel_binaria + sel_multiclase + sel_regresion
    sellados = _sellar(seleccion)
    for r in seleccion:
        r["sellado"] = r["data_id"] in sellados
        r["licencia"] = r["licencia"] or "Public"
    return seleccion


def verificar_cobertura(seleccion: list[dict]) -> None:
    faltan = sum(1 for r in seleccion if r["tiene_faltantes"])
    altacard = sum(1 for r in seleccion if r["alta_cardinalidad"])
    desbal = sum(1 for r in seleccion if r["desbalanceado"])
    numericas = sum(1 for r in seleccion if r["solo_numericas"])
    mc = [r for r in seleccion if r["multiclase"]]
    mc5 = sum(1 for r in mc if r["n_clases"] and r["n_clases"] >= 5)
    mcrara = sum(1 for r in mc if r["clase_muy_minoritaria"])
    problemas = []
    if faltan < MINIMO_FALTANTES: problemas.append(f"faltantes {faltan}<{MINIMO_FALTANTES}")
    if altacard < MINIMO_ALTA_CARDINALIDAD: problemas.append(f"alta_cardinalidad {altacard}<{MINIMO_ALTA_CARDINALIDAD}")
    if desbal < MINIMO_DESBALANCEADOS: problemas.append(f"desbalanceados {desbal}<{MINIMO_DESBALANCEADOS}")
    if numericas < MINIMO_SOLO_NUMERICAS: problemas.append(f"solo_numericas {numericas}<{MINIMO_SOLO_NUMERICAS}")
    if mc5 < MINIMO_MULTICLASE_5_CLASES: problemas.append(f"multiclase>=5clases {mc5}<{MINIMO_MULTICLASE_5_CLASES}")
    if mcrara < MINIMO_MULTICLASE_CLASE_RARA: problemas.append(f"multiclase_clase_rara {mcrara}<{MINIMO_MULTICLASE_CLASE_RARA}")
    if problemas:
        raise SystemExit("cobertura obligatoria NO cumplida: " + "; ".join(problemas))
    print("cobertura obligatoria: OK", Counter(r["bucket"] for r in seleccion))


def descargar_y_hashear(seleccion: list[dict], directorio: Path) -> dict[int, str]:
    directorio.mkdir(parents=True, exist_ok=True)
    hashes: dict[int, str] = {}
    for r in seleccion:
        destino = directorio / f"{r['data_id']}.arff"
        if not destino.exists():
            req = urllib.request.Request(r["url"], headers={"User-Agent": "matrixai-fase0/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                destino.write_bytes(resp.read())
        hashes[r["data_id"]] = hashlib.sha256(destino.read_bytes()).hexdigest()
    return hashes


def construir_protocolo(seleccion: list[dict], hashes: dict[int, str]) -> ProtocoloExploratorio:
    datasets = []
    for r in sorted(seleccion, key=lambda r: r["data_id"]):
        tarea = ("binary_classification" if r["binaria"] else
                "multiclass_classification" if r["multiclase"] else "regression")
        datasets.append(DatasetRegistrado(
            data_id=r["data_id"], nombre=r["nombre"], fuente="openml", version=r["version"],
            sha256_arff=hashes[r["data_id"]], columna_objetivo=r["objetivo"], tarea=tarea,
            cubo_de_tamano=r["bucket"], n_filas=r["n_filas"], n_columnas=r["n_columnas"],
            tiene_faltantes=r["tiene_faltantes"], alta_cardinalidad=r["alta_cardinalidad"],
            desbalanceado=r["desbalanceado"], solo_numericas=r["solo_numericas"],
            licencia=r["licencia"], sellado=r["sellado"]))

    motores = (
        Motor(id="dummy", configuraciones=1),
        Motor(id="sklearn.logreg", configuraciones=2),
        Motor(id="sklearn.hgb", configuraciones=2),
        Motor(id="lightgbm", configuraciones=2),
        Motor(id="xgboost", configuraciones=2),
        Motor(id="catboost", configuraciones=2),
        Motor(id="matrixai.dense.torch_cpu", configuraciones=2),
    )
    return ProtocoloExploratorio(
        version_protocolo="101-C1.v1", fecha_registro="2026-09-06",
        datasets=tuple(datasets), motores=motores,
        particion=DisenoDeParticion(folds=5, repeticiones_pequeno_mediano=3,
                                    repeticiones_grande=1, semillas=(0, 1, 2)),
        presupuesto=PresupuestoPorCubo(minutos_por_cubo={"pequeno": 2.0, "mediano": 5.0, "grande": 10.0},
                                       hilos=4, procesos_en_paralelo=6),
        regla_de_cierre=ReglaDeCierre(
            puntos=2.0, fraccion_minima=0.80,
            metrica_por_tarea={"binary_classification": "AUROC",
                               "multiclass_classification": "accuracy_o_f1_macro", "regression": "R2"},
            definicion_de_mejor="el motor con mejor media de los ajustes en ESE dataset, excluido "
                                "el baseline dummy; un fallo cuenta como dataset perdido para ese motor"),
        recursos_declarados={
            "maquina": "servidor de desarrollo, medido 2026-09-04", "cpu_fisicas": 8, "cpu_logicas": 8,
            "ram_gb": 15, "ram_disponible_gb": 10, "gpu": None, "disco_gb": 197, "python": "3.12.3",
            "hilos_por_proceso": 4, "procesos_en_paralelo": 6,
            "nota": "sin GPU en esta máquina; CUDA queda para una tabla aparte marcada GPU, "
                   "fuera del ranking CPU principal (invariante 4 del 101)"},
        metricas_por_tarea={
            "binary_classification": {"primaria": "AUROC", "secundarias": [
                "AUPRC", "log_loss", "Brier", "accuracy@0.5", "accuracy@prevalencia", "F1_macro"],
                "calibracion": "pendiente/intercepto de recalibración logística sobre logit(p); ECE 10 cubos"},
            "multiclass_classification": {"primaria": "log_loss",
                "secundarias": ["macro_AUROC_OVR", "accuracy", "F1_macro"],
                "calibracion": "ECE de la clase predicha"},
            "regression": {"primaria": "RMSE", "secundarias": ["MAE", "R2"], "calibracion": None},
            "siempre": ["tiempo_de_ajuste", "tiempo_de_prediccion", "cpu_segundos", "rss_pico_mb",
                       "tasa_de_fallo", "dispersion_entre_semillas"],
            "intervalos": "bootstrap de 1000 remuestreos sobre las predicciones fuera-de-fold agrupadas "
                         "por repetición, más la dispersión entre las 3 semillas (105-C2, reutilizado)"},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="protocolo_exploratorio.json")
    ap.add_argument("--datos", default=str(Path.home() / "fase0_openml_datos" / "arff"),
                    help="directorio donde se guardan los ARFF descargados (fuera del repo, NO se comitea)")
    args = ap.parse_args()

    print("consultando OpenML (CC18 + AMLB clasificación + AMLB regresión)...")
    seleccion = seleccionar_40()
    verificar_cobertura(seleccion)
    print(f"descargando y hasheando {len(seleccion)} ARFF en {args.datos}...")
    hashes = descargar_y_hashear(seleccion, Path(args.datos))
    protocolo = construir_protocolo(seleccion, hashes)
    coste = calcular_coste(protocolo)
    print("digest:", protocolo.digest())
    print(json.dumps(coste.a_json(), indent=2))

    payload = protocolo.a_json()
    payload["digest_sha256"] = protocolo.digest()
    payload["coste_calculado"] = coste.a_json()
    Path(args.salida).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito:", args.salida)


if __name__ == "__main__":
    main()
