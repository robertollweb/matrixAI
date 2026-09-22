# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""114-C5 — genera y SELLA el protocolo de la medición de las fechas.

Todo lo que decide el veredicto se escribe aquí ANTES de medir: los conjuntos,
qué columnas salen y por qué, el objetivo, los dos repartos, el motor, la
métrica, el margen y el criterio. La huella (sha256 del JSON canónico sin el
campo `huella`) se comprueba al arrancar la medición: si alguien toca el
protocolo después, la medición se niega.

    python3 benchmarks/fechas_114c5/generar_protocolo.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUTA = Path(__file__).resolve().parent / "protocolo_114c5.json"

CONJUNTOS = [
    {"data_id": 43820, "nombre": "Avocado-Prices", "tarea": "regression",
     "objetivo": "AveragePrice", "fecha": "Date", "formato": "%Y-%m-%d",
     "licencia": "ODbL (metadatos de OpenML)",
     "excluidas": {"Unnamed:_0": "índice que sigue el orden de las semanas dentro de cada región"}},
    {"data_id": 43771, "nombre": "International-football-results", "tarea": "binary_classification",
     "objetivo": "gana_el_local", "fecha": "date", "formato": "%Y-%m-%d", "licencia": "CC0",
     "objetivo_derivado": {"regla": "home_score > away_score", "positiva": "si", "negativa": "no",
                           "nota": "el empate cuenta como «no gana»"},
     "excluidas": {"home_score": "ES el desenlace", "away_score": "ES el desenlace"}},
    {"data_id": 42998, "nombre": "metro-interstate-traffic", "tarea": "regression",
     "objetivo": "traffic_volume", "fecha": "date_time", "formato": "%Y-%m-%d %H:%M:%S",
     "licencia": "CC BY 4.0 (UCI)", "excluidas": {}},
    {"data_id": 43746, "nombre": "NYC-Uber-Pickups-with-Weather-and-Holidays", "tarea": "regression",
     "objetivo": "pickups", "fecha": "pickup_dt", "formato": "%Y-%m-%d %H:%M:%S",
     "licencia": "CC0", "excluidas": {}},
    {"data_id": 43805, "nombre": "Daily-Electricity-Price-and-Demand-Data", "tarea": "regression",
     "objetivo": "demand", "fecha": "date", "formato": "%Y-%m-%d", "licencia": "ODbL / DbCL",
     "excluidas": {
         "demand_pos_RRP": "con demand_neg_RRP suma EXACTAMENTE el objetivo (medido: 7,3e-11)",
         "demand_neg_RRP": "con demand_pos_RRP suma EXACTAMENTE el objetivo",
         "RRP": "precio del mismo día, se fija a la vez que la demanda",
         "RRP_positive": "precio del mismo día", "RRP_negative": "precio del mismo día",
         "frac_at_neg_RRP": "precio del mismo día"}},
]

PROTOCOLO = {
    "version": "114-C5.v1",
    "que_se_compara": {
        "sin": "la preparación de hoy: la fecha entra como texto (categórica)",
        "con": "la fecha sustituida por sus variables (`matrixai.training.fechas`); "
               "recetas detectadas SOLO con las filas de entrenamiento"},
    "conjuntos": CONJUNTOS,
    "repartos": {
        "temporal": "ordenar por la fecha; el 80 % más antiguo entrena y el resto prueba, sin partir "
                    "una misma fecha entre los dos: el corte se lleva al primer cambio de fecha "
                    "a partir del 80 % (corregido antes de medir, auditoría del 2026-09-22)",
        "aleatorio": "barajar con la semilla; 80 % entrena, 20 % prueba"},
    "reparto_del_veredicto": "temporal",
    "motor": "lightgbm",
    "metrica": {"regression": "rmse", "binary_classification": "auroc"},
    "margen_de_equivalencia": {"regression": "1 % del RMSE de «sin» en ese conjunto y reparto",
                               "binary_classification": 0.01},
    "comparacion": {"funcion": "matrixai.estudio.comparaciones.comparar_candidatos",
                    "diseno": {"temporal": "temporal", "aleatorio": "iid"},
                    "estimando": "fixed_model_on_population", "remuestras": 1000},
    "criterio": "«mejora» en al menos 3 de los 5 en el reparto temporal, y NINGUNA «inferioridad»",
    "semilla": 0,
    "descartados": {
        "Melbourne-Housing-Snapshot": "CC BY-NC-SA: no comercial",
        "house_sales": "su fecha (20141013T000000) no la detecta el núcleo: hueco de detección, aparte",
        "Superstore-Sales-Dataset": "declara GPL 2, una licencia de software",
        "Inside Airbnb / versiones con la fecha partida": "no traen la fecha en texto"},
}


def huella(protocolo: dict) -> str:
    sin = {k: v for k, v in protocolo.items() if k != "huella"}
    return hashlib.sha256(json.dumps(sin, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


if __name__ == "__main__":
    sellado = dict(PROTOCOLO, huella=huella(PROTOCOLO))
    RUTA.write_text(json.dumps(sellado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(RUTA, sellado["huella"])
