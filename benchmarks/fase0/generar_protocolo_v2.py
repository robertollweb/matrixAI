#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""113-C0 — cómo se construyó `protocolo_exploratorio_v2.json`, y cómo repetirlo.

**No re-implementa `generar_protocolo.py`: lo REUTILIZA.** Las funciones que
consultan OpenML, que descargan y hashean un ARFF, y que miden cardinalidad
sobre la cabecera son las MISMAS que produjeron la v1 — importadas, no
copiadas. Dos sitios declarando lo mismo acaban divergiendo, y aquí el riesgo
es peor que de costumbre: un segundo cálculo de `desbalanceado` o de
`max_cardinalidad_nominal` que discrepe del de la v1 dejaría dos protocolos
que no se pueden comparar.

**Qué re-firma este guion, y qué NO toca** (decidido por Roberto el
2026-09-21, `TASKS.md` «Re-firma del protocolo de Fase 0»; contrato 113-C0):

- **P1 (a)**: en MULTICLASE, `desbalanceado` pasa a exigir que la cuota de la
  clase minoritaria sea `<= 0.2/k` (k = número de clases) — la misma fórmula
  que en binaria (0.2/2 = 10 %), sin un segundo umbral elegido mirando los
  datos. Se recalcula MIDIENDO sobre el ARFF ya descargado de cada uno de los
  10 multiclase de la v1 (ninguno de los 3 que entran es multiclase), pasando
  por el LECTOR ÚNICO de este paquete (`lector_arff.cargar` — ver
  `test_101_c3_un_solo_lector_de_arff.py`: nada aquí llama a `scipy.io.arff`
  por su cuenta). En binaria y regresión, `desbalanceado` NO se toca: sigue
  siendo el que la v1 ya tenía, byte a byte. **Y P1 se aplica también a los
  multiclase SELLADOS** (corrección del supervisor, 2026-09-21): la decisión
  de Roberto es UNA sola fórmula («marca solo yeast»), y «los sellados no se
  re-sellan» se refiere a su SELECCIÓN —la regla equiespaciada que se aplicó
  antes de ver resultados—, no a un campo derivado cuando cambia la regla que
  lo deriva. El agente siguió la frase del encargo («ni se tocan»), que era
  del supervisor y exageraba la nota original. `letter` pasa de `True` (regla
  vieja, cuota 3,67 % ≤ 10 %) a `False` (cuota·k = 0,954 > 0,2). Ni la pasada
  ni el veredicto leen `desbalanceado`: solo el catálogo y la cobertura.
- **P2 (a)**: sustituye 3 datasets, uno por celda (tarea, cubo de tamaño), sin
  tocar ningún `sellado`. Entran `dresses-sales` (23381, binaria/pequeño),
  `kick` (41162, binaria/grande) y `SAT11-HAND-runtime-regression` (41980,
  regresión/mediano); salen `breast-w` (15), `adult` (1590) y `elevators`
  (216) — los tres NO sellados, uno por cada celda de los que entran. Los
  otros 37 datasets de la v1 quedan intactos.
- **P3**: una sola versión nueva (`113-C0.v2`), que junta esta re-firma con la
  receta de la densa del contrato 113-C1 (invariante 1 de ese contrato: «la
  receta se fija ANTES de medir y va en el protocolo v2»), declarada dentro
  de la entrada de `matrixai.dense.torch_cpu` en `motores` (campo `receta`,
  aditivo — ver `Motor.receta` en `protocolo.py`).

**Lo que este guion NO recalcula**: `particion`, `presupuesto`,
`regla_de_cierre`, `recursos_declarados` y `metricas_por_tarea` se copian de
la v1 sin tocar — nada de eso está en la re-firma decidida el 2026-09-21, y
"P3: una sola versión nueva" no es "una versión distinta en todo".

Uso:
    python3 benchmarks/fase0/generar_protocolo_v2.py \
        --salida benchmarks/fase0/protocolo_exploratorio_v2.json \
        --seleccion-salida /home/deployer/fase0_openml_datos/seleccion_v2.json
    (descarga y hashea 3 ARFF nuevos — el resto ya está en `--datos`, del
    generador de la v1 — y hace unas pocas llamadas de metadatos a OpenML)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_RAIZ_DEL_CORE = Path(__file__).resolve().parents[2]
if str(_RAIZ_DEL_CORE) not in sys.path:
    sys.path.insert(0, str(_RAIZ_DEL_CORE))

from benchmarks.fase0 import generar_protocolo as g1  # noqa: E402
from benchmarks.fase0.lector_arff import cargar as _cargar_arff  # noqa: E402
from benchmarks.fase0.protocolo import (  # noqa: E402
    DatasetRegistrado,
    Motor,
    ProtocoloExploratorio,
    UMBRAL_ALTA_CARDINALIDAD,
    calcular_coste,
)

RUTA_V1 = Path(__file__).resolve().parent / "protocolo_exploratorio.json"

#: P2(a) — decidido por Roberto el 2026-09-21, `TASKS.md` «Re-firma del
#: protocolo de Fase 0»: uno por celda (tarea, cubo de tamaño), sin tocar
#: ningún `sellado`. Los tres que salen NO estaban sellados en la v1.
DATA_IDS_SALIENTES = {15: "breast-w", 1590: "adult", 216: "elevators"}
DATA_IDS_ENTRANTES = {
    23381: ("dresses-sales", True),   # (nombre, es_clasificacion)
    41162: ("kick", True),
    41980: ("SAT11-HAND-runtime-regression", False),
}

#: P1(a): en MULTICLASE, desbalanceado si `cuota_minoritaria * k <= 0.2`
#: — equivalente a `cuota_minoritaria <= 0.2 / k`. La misma fórmula de la
#: binaria (0.2/2 = 0.10), sin un segundo umbral elegido mirando los datos.
UMBRAL_P1_DESBALANCE = 0.2

#: La receta de la densa, del contrato 113-C1, fijada ANTES de medir —
#: invariante 1 de ese contrato. `learning_rate` es el valor por omisión de
#: Adam (Kingma y Ba, 2015; el mismo que `torch.optim.Adam`): no se elige
#: mirando datos. `epochs` y `batch_size` se declaran SIN CAMBIOS respecto a
#: lo que el núcleo ya genera (`dense_generator.py`), a propósito: así el
#: efecto medido en C4/C5 es el del optimizador y la parada temprana, no el
#: de entrenar más o con otro lote.
RECETA_DENSA_113_C1: dict[str, Any] = {
    "optimizador": "adam",
    "learning_rate": 0.001,
    "early_stop": {"patience": 10, "metric": "validation_loss"},
    "epochs": 50,
    "batch_size": 8,
}


def _cuota_minoritaria_y_k(ruta_arff: Path, columna_objetivo: str) -> tuple[float, int]:
    """(cuota_de_la_clase_minoritaria, k) MEDIDOS sobre el ARFF ya descargado.

    Pasa por el LECTOR ÚNICO de `benchmarks/fase0` (`lector_arff.cargar`) en
    vez de llamar a `scipy.io.arff` por su cuenta: tres copias sueltas ya
    divergieron una vez (el `"?"` como faltante, el objetivo declarado, las
    normalizaciones), y `test_101_c3_un_solo_lector_de_arff.py` existe
    justamente para que no vuelva a pasar. `cargar()` ya deja el `"?"` como
    `None` — la misma condición que la sonda de deployer-02
    (`scratchpad/refirma/medir_desbalance.py`) aplicaba a mano.
    """
    leido = _cargar_arff(ruta_arff, objetivo_declarado=columna_objetivo)
    conteo = Counter(fila[leido.objetivo] for fila in leido.filas
                     if fila.get(leido.objetivo) is not None)
    n = sum(conteo.values())
    k = len(conteo)
    minoria = min(conteo.values())
    return minoria / n, k


def re_firmar_desbalanceado_multiclase(datasets_v1: list[dict], directorio: Path) -> dict[int, bool]:
    """`{data_id: desbalanceado}` recalculado con P1(a) para TODOS los
    `multiclass_classification` de la v1 — MEDIDO, sellados incluidos: quien
    llama decide si lo aplica a un sellado o no (`construir_datasets_v2` no
    lo aplica; ver su comentario). Separar «medir» de «aplicar» es a
    propósito: así esta función sigue siendo honesta («esto es lo que P1
    daría aquí») aunque el sellado no se toque.

    La binaria y la regresión no cambian (P1 no las toca), y ninguno de los 3
    datasets que entran en P2 es multiclase."""
    salida: dict[int, bool] = {}
    for d in datasets_v1:
        if d["tarea"] != "multiclass_classification":
            continue
        ruta = directorio / f"{d['data_id']}.arff"
        cuota, k = _cuota_minoritaria_y_k(ruta, d["columna_objetivo"])
        salida[d["data_id"]] = (cuota * k) <= UMBRAL_P1_DESBALANCE
    return salida


def construir_entrantes(directorio: Path) -> list[dict]:
    """Los 3 candidatos que entran (P2a), con TODOS los campos calculados
    igual que en la v1: `_candidato` (metadatos), descarga+sha256+md5,
    `medir_cardinalidad` (cabecera del ARFF)."""
    filas = []
    for data_id, (nombre_esperado, es_clasificacion) in DATA_IDS_ENTRANTES.items():
        c = g1._candidato(data_id, es_clasificacion=es_clasificacion)
        if c is None or not c["objetivo"]:
            raise SystemExit(f"{data_id} ({nombre_esperado}) ya no pasa los filtros del generador "
                             "(activo/licencia/bucket) — la re-firma no puede continuar a ciegas")
        if c["nombre"] != nombre_esperado:
            raise SystemExit(f"{data_id} se esperaba {nombre_esperado!r} y OpenML devuelve "
                             f"{c['nombre']!r} — no se re-firma con un nombre que no cuadra")
        # `_candidato` guarda `info.get("version")` tal cual, y la API de
        # OpenML lo da como CADENA («'1'», medido 2026-09-21) mientras los 40
        # de la v1 lo tienen como ENTERO en el fichero registrado — casteado
        # explícito para que el tipo no dependa de un capricho de la API.
        c["version"] = int(c["version"])
        c["licencia"] = c["licencia"] or "Public"
        filas.append(c)
    return filas


def descargar_hashear_y_verificar_md5(seleccion: list[dict], directorio: Path) -> dict[int, str]:
    """Como `g1.descargar_y_hashear`, y ADEMÁS contrasta el md5 declarado por
    la API de OpenML (`GET data/{data_id}`) contra el md5 del fichero bajado
    — el criterio del corte: "verificados contra el `md5_checksum` de OpenML
    y su `sha256`"."""
    import urllib.request

    directorio.mkdir(parents=True, exist_ok=True)
    hashes: dict[int, str] = {}
    for r in seleccion:
        destino = directorio / f"{r['data_id']}.arff"
        if not destino.exists():
            req = urllib.request.Request(r["url"], headers={"User-Agent": "matrixai-fase0/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                destino.write_bytes(resp.read())
        crudo = destino.read_bytes()
        info = g1._info(r["data_id"])
        md5_openml = info.get("md5_checksum")
        md5_local = hashlib.md5(crudo).hexdigest()
        if not md5_openml or md5_local != md5_openml:
            raise SystemExit(
                f"{r['data_id']} ({r['nombre']}): md5 NO coincide — OpenML dice "
                f"{md5_openml!r}, el fichero bajado da {md5_local!r}. No se registra "
                "un ARFF sin verificar")
        hashes[r["data_id"]] = hashlib.sha256(crudo).hexdigest()
    return hashes


def construir_datasets_v2(payload_v1: dict, directorio: Path) -> list[DatasetRegistrado]:
    """Los 40 de la v2: 37 de la v1 sin tocar (salvo `desbalanceado` en los
    multiclase, P1) + 3 nuevos (P2a)."""
    datasets_v1 = payload_v1["datasets"]
    reafirmado_desbalance = re_firmar_desbalanceado_multiclase(datasets_v1, directorio)

    conservados = []
    for d in datasets_v1:
        if d["data_id"] in DATA_IDS_SALIENTES:
            continue
        fila = dict(d)
        # P1 a TODOS los multiclase, sellados incluidos (ver la cabecera):
        # una sola fórmula. De un sellado solo cambia este campo derivado;
        # todo lo demás queda byte a byte como en la v1, y hay prueba de ello.
        if d["data_id"] in reafirmado_desbalance:
            fila["desbalanceado"] = reafirmado_desbalance[d["data_id"]]
        conservados.append(DatasetRegistrado.desde_json(fila))

    entrantes_crudos = construir_entrantes(directorio)
    hashes = descargar_hashear_y_verificar_md5(entrantes_crudos, directorio)
    g1.medir_cardinalidad(entrantes_crudos, directorio)

    entrantes = []
    for r in entrantes_crudos:
        tarea = ("binary_classification" if r["binaria"] else
                "multiclass_classification" if r["multiclase"] else "regression")
        entrantes.append(DatasetRegistrado(
            data_id=r["data_id"], nombre=r["nombre"], fuente="openml", version=r["version"],
            sha256_arff=hashes[r["data_id"]], columna_objetivo=r["objetivo"], tarea=tarea,
            cubo_de_tamano=r["bucket"], n_filas=r["n_filas"], n_columnas=r["n_columnas"],
            tiene_faltantes=r["tiene_faltantes"],
            max_cardinalidad_nominal=r["max_cardinalidad_nominal"],
            alta_cardinalidad=r["max_cardinalidad_nominal"] >= UMBRAL_ALTA_CARDINALIDAD,
            desbalanceado=r["desbalanceado"], solo_numericas=r["solo_numericas"],
            licencia=r["licencia"], sellado=False,
            file_id=g1.file_id_desde_url(r["url"])))

    todos = conservados + entrantes
    _exigir_celdas(todos)
    return sorted(todos, key=lambda d: d.data_id)


def _exigir_celdas(datasets: list[DatasetRegistrado]) -> None:
    """Cada dataset entrante cae EXACTAMENTE en la celda (tarea, cubo) del que
    sale — el criterio que la decisión de Roberto puso por escrito: "conserva
    40 = 20/10/10 y 15/15/10 por tamaño"."""
    from collections import Counter as _C
    por_celda = _C((d.tarea, d.cubo_de_tamano) for d in datasets)
    esperado = {
        ("binary_classification", "pequeno"): 8, ("binary_classification", "mediano"): 8,
        ("binary_classification", "grande"): 4,
        ("multiclass_classification", "pequeno"): 4, ("multiclass_classification", "mediano"): 4,
        ("multiclass_classification", "grande"): 2,
        ("regression", "pequeno"): 3, ("regression", "mediano"): 3, ("regression", "grande"): 4,
    }
    for celda, cupo in esperado.items():
        if por_celda.get(celda, 0) != cupo:
            raise SystemExit(f"celda {celda} tiene {por_celda.get(celda, 0)}, se esperaban {cupo} "
                             "— la sustitución P2 se ha salido de su celda")
    if len(datasets) != 40:
        raise SystemExit(f"la v2 tiene {len(datasets)} datasets, se esperaban 40")


def construir_motores_v2(payload_v1: dict) -> tuple[Motor, ...]:
    """Los mismos 7 motores de la v1, y la receta de la densa (113-C1) SOLO
    en `matrixai.dense.torch_cpu` — aditivo: los otros seis quedan bit a bit
    como estaban."""
    motores = []
    for m in payload_v1["motores"]:
        if m["id"] == "matrixai.dense.torch_cpu":
            motores.append(Motor(id=m["id"], configuraciones=m["configuraciones"],
                                 receta=RECETA_DENSA_113_C1))
        else:
            motores.append(Motor(**m))
    return tuple(motores)


def construir_protocolo_v2(payload_v1: dict, directorio: Path, *,
                           version_protocolo: str, fecha_registro: str) -> ProtocoloExploratorio:
    datasets = tuple(construir_datasets_v2(payload_v1, directorio))
    motores = construir_motores_v2(payload_v1)
    # particion / presupuesto / regla_de_cierre / recursos_declarados /
    # metricas_por_tarea: SIN TOCAR — P3 dice «una sola versión nueva», no
    # «una versión distinta en todo». Se leen desde el mismo payload de la v1
    # con el MISMO código que ya sabe leerlo (`ProtocoloExploratorio.
    # desde_json`), una sola vez, no una segunda lectura a mano.
    v1 = ProtocoloExploratorio.desde_json(payload_v1)
    return ProtocoloExploratorio(
        version_protocolo=version_protocolo, fecha_registro=fecha_registro,
        datasets=datasets, motores=motores,
        particion=v1.particion, presupuesto=v1.presupuesto,
        regla_de_cierre=v1.regla_de_cierre,
        recursos_declarados=dict(v1.recursos_declarados),
        metricas_por_tarea=dict(v1.metricas_por_tarea),
    )


def escribir_seleccion_v2(datasets: tuple[DatasetRegistrado, ...],
                          ruta_seleccion_v1: Path, ruta_salida: Path) -> None:
    """`seleccion_v2.json`, AL LADO de `seleccion_40_final.json` — sin tocarlo.
    Mismo esquema (uno por dataset, con lo que se sabía antes de sellar):
    para los 37 conservados se copia la fila de la v1 tal cual (con
    `desbalanceado` re-firmado si es multiclase); para los 3 entrantes se
    compone con las mismas qualities que ya se pidieron para construirlos."""
    por_id = {d.data_id: d for d in datasets}
    filas_v1 = {}
    if ruta_seleccion_v1.exists():
        filas_v1 = {int(f["data_id"]): f for f in json.loads(ruta_seleccion_v1.read_text(encoding="utf-8"))}

    salida = []
    for data_id, ds in sorted(por_id.items()):
        if data_id in filas_v1 and data_id not in DATA_IDS_ENTRANTES:
            fila = dict(filas_v1[data_id])
            fila["desbalanceado"] = ds.desbalanceado
            salida.append(fila)
            continue
        q = g1._qualities(data_id)
        info = g1._info(data_id)
        minoria, n = q.get("MinorityClassSize"), q.get("NumberOfInstances")
        # Misma definición que `_candidato` (generar_protocolo.py): «clase muy
        # minoritaria» es cuota < 2 %, sin depender de P1 ni de la tarea — un
        # dato disponible en las mismas qualities que ya se piden aquí, no un
        # `None` de relleno.
        clase_muy_minoritaria = bool(minoria is not None and n and (minoria / n) < 0.02)
        salida.append({
            "data_id": data_id, "name": ds.nombre, "version": str(ds.version),
            "n_rows": q.get("NumberOfInstances"), "n_feats": q.get("NumberOfFeatures"),
            "n_classes": q.get("NumberOfClasses"),
            "missing_values": q.get("NumberOfMissingValues"),
            "symbolic_feats": q.get("NumberOfSymbolicFeatures"),
            "numeric_feats": q.get("NumberOfNumericFeatures"),
            "minority_class_size": q.get("MinorityClassSize"),
            "majority_class_size": q.get("MajorityClassSize"),
            "target": ds.columna_objetivo, "licence": ds.licencia,
            "in_clf_study": ds.tarea != "regression", "in_reg_study": ds.tarea == "regression",
            "url": info.get("url"), "status": info.get("status"),
            "bucket": ds.cubo_de_tamano, "tiene_faltantes": ds.tiene_faltantes,
            "alta_cardinalidad": ds.alta_cardinalidad, "desbalanceado": ds.desbalanceado,
            "solo_numericas": ds.solo_numericas,
            "binaria": ds.tarea == "binary_classification",
            "multiclase": ds.tarea == "multiclass_classification",
            "clase_muy_minoritaria": clase_muy_minoritaria, "sellado": ds.sellado,
        })
    ruta_salida.write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocolo-v1", default=str(RUTA_V1))
    ap.add_argument("--salida", default=str(Path(__file__).resolve().parent / "protocolo_exploratorio_v2.json"))
    ap.add_argument("--datos", default=str(Path.home() / "fase0_openml_datos" / "arff"))
    ap.add_argument("--seleccion-v1", default=str(Path.home() / "fase0_openml_datos" / "seleccion_40_final.json"))
    ap.add_argument("--seleccion-salida", default=str(Path.home() / "fase0_openml_datos" / "seleccion_v2.json"))
    ap.add_argument("--version-protocolo", default="113-C0.v2")
    ap.add_argument("--fecha-registro", default="2026-09-21")
    args = ap.parse_args()

    payload_v1 = json.loads(Path(args.protocolo_v1).read_text(encoding="utf-8"))
    directorio = Path(args.datos)

    print("re-firmando P1 (multiclase) + P2 (3 sustituciones) + receta de la densa (113-C1)...")
    protocolo = construir_protocolo_v2(payload_v1, directorio,
                                       version_protocolo=args.version_protocolo,
                                       fecha_registro=args.fecha_registro)
    coste = calcular_coste(protocolo)
    print("digest v2:", protocolo.digest())
    print(json.dumps(coste.a_json(), indent=2))

    payload = protocolo.a_json()
    payload["digest_sha256"] = protocolo.digest()
    payload["coste_calculado"] = coste.a_json()
    Path(args.salida).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("escrito:", args.salida)

    escribir_seleccion_v2(protocolo.datasets, Path(args.seleccion_v1), Path(args.seleccion_salida))
    print("escrito:", args.seleccion_salida)


if __name__ == "__main__":
    main()
