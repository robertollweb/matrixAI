# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C5 — ¿convertir las fechas en variables MEJORA? Medido con el protocolo sellado.

Para cada conjunto del protocolo y cada reparto (temporal y aleatorio): la misma
preparación del núcleo y el mismo motor, con la fecha como texto («sin», lo de
hoy) y con la fecha convertida (`matrixai.training.fechas`, «con»). Se predice el
MISMO test y se comparan con el veredicto emparejado del 105-C5.

**Por qué un lector de ARFF propio.** El de Fase 0 (`benchmarks/fase0/
lector_arff.py`) EXCLUYE los atributos `STRING` —con razón allí: `scipy` no los
lee y suelen ser texto que identifica la fila—, y en los cinco conjuntos de esta
medición la fecha es `STRING`. Leídos por ese camino, la fecha desaparecería
antes de medir (comprobado el 2026-09-22 leyendo sus cabeceras).

**Pesado**: se lanza con la máquina quieta, nunca con otra medición viva.

    cd /home/deployer/matrixAI && python3 benchmarks/fechas_114c5/medir_114c5.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from generar_protocolo import huella  # noqa: E402

RUTA_PROTOCOLO = AQUI / "protocolo_114c5.json"
RUTA_SALIDA = AQUI / "resultado_114c5.json"
DATOS = Path.home() / "fechas_114c5_datos"
URL_ARFF = "https://www.openml.org/api/v1/json/data/{id}"


def cargar_protocolo() -> dict:
    protocolo = json.loads(RUTA_PROTOCOLO.read_text(encoding="utf-8"))
    if protocolo.get("huella") != huella(protocolo):
        raise SystemExit("el protocolo no cuadra con su huella: se tocó después de sellarlo. No se mide.")
    return protocolo


# -- los datos --------------------------------------------------------------

def descargar(data_id: int) -> Path:
    DATOS.mkdir(parents=True, exist_ok=True)
    destino = DATOS / f"{data_id}.arff"
    if not destino.exists():
        with urllib.request.urlopen(URL_ARFF.format(id=data_id), timeout=60) as r:
            url = json.load(r)["data_set_description"]["url"]
        with urllib.request.urlopen(url, timeout=300) as r:
            destino.write_bytes(r.read())
    return destino


def leer_arff(ruta: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Filas y tipos. Numéricos a `float`; `?` y vacío a `None`; el resto, texto
    tal cual (sin las comillas del ARFF). Conserva los `STRING`."""
    tipos: dict[str, str] = {}
    lineas = ruta.read_text(encoding="utf-8", errors="replace").splitlines()
    inicio = 0
    for i, linea in enumerate(lineas):
        l = linea.strip()
        if l.lower().startswith("@attribute"):
            resto = l[len("@attribute"):].strip()
            if resto.startswith(("'", '"')):
                cierre = resto.index(resto[0], 1)
                nombre, tipo = resto[1:cierre], resto[cierre + 1:].strip()
            else:
                nombre, tipo = resto.split(None, 1)
            tipos[nombre] = "numerico" if tipo.lower().split()[0] in (
                "real", "integer", "numeric") else "texto"
        elif l.lower() == "@data":
            inicio = i + 1
            break
    nombres = list(tipos)
    filas = []
    lector = csv.reader(io.StringIO("\n".join(lineas[inicio:])), quotechar="'",
                        skipinitialspace=True)
    for j, valores in enumerate(lector):
        if not valores or valores[0].startswith("%"):
            continue
        fila: dict[str, Any] = {"row_id": str(j)}
        for nombre, valor in zip(nombres, valores):
            valor = valor.strip()
            if valor in ("?", ""):
                fila[nombre] = None
            elif tipos[nombre] == "numerico":
                fila[nombre] = float(valor)
            else:
                fila[nombre] = valor
        filas.append(fila)
    return filas, tipos


def preparar_conjunto(conjunto: dict, filas: list[dict[str, Any]]) -> tuple[list[dict], str, tuple]:
    """Aplica el objetivo derivado y las exclusiones del protocolo."""
    objetivo = conjunto["objetivo"]
    derivado = conjunto.get("objetivo_derivado")
    if derivado:
        assert derivado["regla"] == "home_score > away_score", derivado
        for f in filas:
            f[objetivo] = (derivado["positiva"] if f["home_score"] > f["away_score"]
                           else derivado["negativa"])
    fuera = set(conjunto["excluidas"]) | {"row_id", objetivo}
    for columna in conjunto["excluidas"]:
        if columna not in filas[0]:
            raise SystemExit(f"{conjunto['nombre']}: la columna excluida {columna!r} no está")
    filas = [f for f in filas if f.get(objetivo) is not None and f.get(conjunto["fecha"])]
    predictores = tuple(c for c in filas[0] if c not in fuera)
    return filas, objetivo, predictores


def repartir(filas: list[dict], conjunto: dict, reparto: str, semilla: int):
    corte_pedido = int(len(filas) * 0.8)
    if reparto == "temporal":
        def momento(f):
            return datetime.strptime(f[conjunto["fecha"]], conjunto["formato"])
        orden = sorted(filas, key=lambda f: (momento(f), f["row_id"]))
        # Sin partir una MISMA fecha entre los dos lados (auditoría del 2026-09-22:
        # 86 filas de Avocado con la fecha del corte quedaban a cada lado). El corte
        # se lleva al primer cambio de fecha a partir del 80 %: «entrenar en el
        # pasado y probar en el futuro» no admite un día que esté en los dos.
        corte = corte_pedido
        while 0 < corte < len(orden) and momento(orden[corte]) == momento(orden[corte - 1]):
            corte += 1
        return orden[:corte], orden[corte:]
    orden = list(filas)
    random.Random(semilla).shuffle(orden)
    return orden[:corte_pedido], orden[corte_pedido:]


# -- una variante -----------------------------------------------------------

def medir_variante(variante: str, train, test, conjunto: dict, objetivo: str,
                   predictores: tuple, semilla: int, nombre_motor: str):
    from matrixai.estudio import ProblemSpec
    from matrixai.estudio.metricas import Muestra
    from matrixai.training.fechas import detectar_fechas, expandir_fila
    from matrixai.training.preparacion import (ajustar_preparacion, tipar_columnas_numericas,
                                              transformar_fila)
    from matrixai_engines.particiones import Particion, Presupuesto
    from matrixai_engines.registro_de_motores import motor_para

    train = [dict(f) for f in train]
    test = [dict(f) for f in test]
    columnas = predictores
    recetas = ()
    if variante == "con":
        recetas = detectar_fechas(train, (conjunto["fecha"],))
        if [r.formato for r in recetas] != [conjunto["formato"]]:
            raise SystemExit(f"{conjunto['nombre']}: se detectó {recetas!r} y el protocolo "
                             f"registra {conjunto['formato']!r}")
        train = [expandir_fila(f, recetas)[0] for f in train]
        test = [expandir_fila(f, recetas)[0] for f in test]
        columnas = tuple(c for c in predictores if c != conjunto["fecha"]) + \
            recetas[0].columnas_derivadas()
    tipar_columnas_numericas(train + test, columnas)
    motor = motor_para(nombre_motor)
    capacidades = motor.capabilities()
    politica = ajustar_preparacion(train, objetivo=objetivo, columnas=columnas,
                                   admite_categoricas=capacidades.admite_categoricas,
                                   admite_faltantes=capacidades.admite_faltantes)

    def transformadas(filas):
        salida = []
        for f in filas:
            t = transformar_fila(f, politica)
            t["__row_id__"], t[objetivo] = f["row_id"], f[objetivo]
            salida.append(t)
        return salida

    binaria = conjunto["tarea"] == "binary_classification"
    derivado = conjunto.get("objetivo_derivado") or {}
    spec = ProblemSpec(
        problem_id=f"114c5-{conjunto['data_id']}", target=objetivo, task=conjunto["tarea"],
        observation_unit="fila",
        classes=((derivado["negativa"], derivado["positiva"]) if binaria else None),
        positive_label=(derivado["positiva"] if binaria else None),
        predictors=politica.columnas_de_salida())
    tr = Particion.desde_filas(transformadas(train), row_id_field="__row_id__", target_field=objetivo)
    te_filas = transformadas(test)
    te = Particion.desde_filas(te_filas, row_id_field="__row_id__", target_field=objetivo)
    inicio = time.perf_counter()
    resultado, ajustado = motor.fit(tr, None, spec, Presupuesto(wall_seconds=900.0, seed=semilla),
                                    candidate=variante, split_plan_digest=hashlib.sha256(
                                        f"{conjunto['data_id']}".encode()).hexdigest())
    if ajustado is None:
        raise SystemExit(f"{conjunto['nombre']} ({variante}): el motor no ajustó: {resultado}")
    segundos = time.perf_counter() - inicio
    y = tuple(f[objetivo] for f in te_filas)
    if binaria:
        muestra = Muestra(task=conjunto["tarea"], y_true=y, classes=ajustado.classes,
                          positive_label=ajustado.positive_label,
                          probabilities=tuple(motor.predict_proba(ajustado, te)))
    else:
        muestra = Muestra(task=conjunto["tarea"], y_true=tuple(float(v) for v in y),
                          predictions=tuple(float(v) for v in motor.predict(ajustado, te)))
    return muestra, {"segundos_de_ajuste": round(segundos, 2), "n_columnas": len(columnas),
                     "recetas": [r.a_json() for r in recetas]}


def comparar(conjunto: dict, reparto: str, protocolo: dict, sin, con) -> dict:
    from matrixai.estudio.comparaciones import comparar_candidatos
    from matrixai.estudio.metricas import calcular

    metrica = protocolo["metrica"][conjunto["tarea"]]
    valor_sin = calcular(metrica, sin).value
    valor_con = calcular(metrica, con).value
    margen = (0.01 * valor_sin if conjunto["tarea"] == "regression"
              else protocolo["margen_de_equivalencia"]["binary_classification"])
    veredicto = comparar_candidatos(
        metrica, con, sin, diseno=protocolo["comparacion"]["diseno"][reparto],
        estimando=protocolo["comparacion"]["estimando"], semilla=protocolo["semilla"],
        margen_equivalencia=margen, remuestras=protocolo["comparacion"]["remuestras"])
    return {"metrica": metrica, "sin": valor_sin, "con": valor_con, "margen": margen,
            "comparacion": veredicto.a_json()}


def _commit(raiz: Path) -> dict:
    def git(*a):
        return subprocess.run(("git", "-C", str(raiz), *a), capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "arbol_sucio": bool(git("status", "--porcelain"))}


def main() -> None:
    protocolo = cargar_protocolo()
    salida: dict[str, Any] = {
        "protocolo": protocolo["version"], "huella_del_protocolo": protocolo["huella"],
        "procedencia": {"matrixAI": _commit(AQUI.parents[1]),
                        "matrixai-engines": _commit(Path("/home/deployer/matrixai-engines"))},
        "creado": datetime.utcnow().isoformat(timespec="seconds") + "Z", "conjuntos": []}
    for conjunto in protocolo["conjuntos"]:
        ruta = descargar(conjunto["data_id"])
        filas, _ = leer_arff(ruta)
        filas, objetivo, predictores = preparar_conjunto(conjunto, filas)
        registro = {"data_id": conjunto["data_id"], "nombre": conjunto["nombre"],
                    "sha256_arff": hashlib.sha256(ruta.read_bytes()).hexdigest(),
                    "n_filas": len(filas), "predictores": list(predictores), "repartos": {}}
        for reparto in ("temporal", "aleatorio"):
            train, test = repartir(filas, conjunto, reparto, protocolo["semilla"])
            sin, info_sin = medir_variante("sin", train, test, conjunto, objetivo, predictores,
                                           protocolo["semilla"], protocolo["motor"])
            con, info_con = medir_variante("con", train, test, conjunto, objetivo, predictores,
                                           protocolo["semilla"], protocolo["motor"])
            registro["repartos"][reparto] = {"n_train": len(train), "n_test": len(test),
                                             "sin": info_sin, "con": info_con,
                                             **comparar(conjunto, reparto, protocolo, sin, con)}
            print(conjunto["nombre"], reparto, registro["repartos"][reparto]["comparacion"]["veredicto"],
                  flush=True)
        salida["conjuntos"].append(registro)
    temporales = [c["repartos"]["temporal"]["comparacion"]["veredicto"] for c in salida["conjuntos"]]
    salida["criterio"] = {
        "texto": protocolo["criterio"], "mejoras": temporales.count("mejora"),
        "inferioridades": temporales.count("inferioridad"),
        "cumple": temporales.count("mejora") >= 3 and "inferioridad" not in temporales}
    RUTA_SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(salida["criterio"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
