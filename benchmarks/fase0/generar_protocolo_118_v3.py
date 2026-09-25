#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""118-C0 — el protocolo v3 de la red densa, registrado ANTES de construir ninguna palanca.

Contrato 118 (`documentacion/118_LA_RED_DENSA_COMPITE_CONTRACT.md`), aprobado por Roberto el
2026-09-25. Este guion escribe `protocolo_118_v3.json` y su digest; se commitea ANTES de tocar
el núcleo para C1, y la prueba `tests/test_118_c0_protocolo_v3.py` ata que:

- todo lo que la v2 fija (conjuntos, sellado, partición, semillas, presupuesto, recursos,
  regla de cierre, métricas) se COPIA de `protocolo_exploratorio_v2.json` byte a byte: la v3
  no puede mover el listón ni el reparto para que la densa salga mejor;
- cada palanca queda DEFINIDA aquí con sus parámetros, antes de existir su código; C4 y C5
  se registran con una enmienda propia antes de medirlas (su diseño depende de lo que
  salga de C1–C3), y lo dice el propio JSON.

**Por qué cada palanca se mide SOLA sobre la receta de la v2** y no acumulando: si se
acumulan, no se sabe cuál subió. Después se combinan las que ayudaron (regla abajo) y la
combinación se confirma UNA vez en los sellados.

    python3 benchmarks/fase0/generar_protocolo_118_v3.py            # escribe el JSON
    python3 benchmarks/fase0/generar_protocolo_118_v3.py --comprobar # falla si difiere
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RUTA_V2 = AQUI / "protocolo_exploratorio_v2.json"
RUTA_V3 = AQUI / "protocolo_118_v3.json"

#: Lo que la v3 copia de la v2 sin tocar.
CAMPOS_COPIADOS_DE_LA_V2 = ("datasets", "particion", "presupuesto", "recursos_declarados",
                            "regla_de_cierre", "metricas_por_tarea")

PALANCAS = [
    {
        "id": "118-C1.cuantiles",
        "que": "las entradas NUMÉRICAS de la densa pasan por una transformación por cuantiles "
               "hacia la normal en vez del escalado mín-máx por el rango de train",
        "parametros": {
            "transformacion": "sklearn.preprocessing.QuantileTransformer",
            "output_distribution": "normal",
            "n_quantiles": "min(1000, filas de train)",
            "subsample": "todas las filas de train",
            "ajustada_sobre": "train (nunca validation ni test)",
            "fuera_de_rango": "el propio transformador lo lleva al extremo de su rango (sin extrapolar)",
            "categoricas": "sin cambio (one-hot como en la v2)",
            "faltantes": "sin cambio (la preparación de la v2 los resuelve antes)",
        },
        "conjuntos": "todos los NO sellados",
        "motivo": "análisis del 24-09: colas largas aplastadas por mín-máx y extrapolación sin "
                  "freno (house_prices, R² −129 y −302 en dos pliegues de la amplia)",
    },
    {
        "id": "118-C2.objetivo",
        "que": "en REGRESIÓN, el objetivo se estandariza con la media y la desviación de train "
               "(y, si es positivo y sesgado, antes se aplica log1p); las predicciones se "
               "deshacen igual antes de medir",
        "parametros": {
            "estandarizar": "(y − media_train) / desviacion_train",
            "log1p_si": "mínimo de train > 0 y asimetría (skewness) de train > 1",
            "ajustado_sobre": "train",
            "clasificacion": "sin cambio",
        },
        "conjuntos": "los NO sellados de REGRESIÓN (en los demás la palanca no cambia nada)",
        "motivo": "análisis del 24-09: las cinco regresiones de objetivo con cola larga pierden "
                  "entre 5,6 y 16,5 puntos",
    },
    {
        "id": "118-C3.receta",
        "que": "la receta de entrenamiento de la densa",
        "parametros": {
            "lote": "min(256, filas de train)",
            "optimizador": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "programa_de_tasa": "coseno sobre las épocas máximas",
            "epocas_maximas": 200,
            "parada_temprana": {"paciencia": 20, "metrica": "validation_loss"},
            "plazo": "el mismo del motor (0,75 del presupuesto de pared): nunca más reloj",
        },
        "conjuntos": "todos los NO sellados",
        "motivo": "análisis del 24-09: lote de 8 y 50 épocas; el plazo corta antes de converger "
                  "en los grandes (letter −13,4, connect-4 −4,1) y micro-mass pierde 27 sin "
                  "regularización",
    },
    {
        "id": "118-C4.embeddings",
        "que": "embeddings para las categóricas de muchos valores",
        "parametros": None,
        "registro_pendiente": "se define en una ENMIENDA de este protocolo antes de medirla",
    },
    {
        "id": "118-C5.semillas",
        "que": "ensamblado de 3–5 semillas de la densa",
        "parametros": None,
        "registro_pendiente": "se define en una ENMIENDA de este protocolo antes de medirla",
    },
]

VEREDICTO = {
    "por_palanca": {
        "campo": "los motores de la v2 con sus resultados de la pasada v2 (mismas particiones), y "
                 "en el sitio de la densa v2, la densa con la palanca",
        "cumplidos": "cuántos conjuntos no sellados cumplen la regla de cierre (a 2 puntos del "
                     "mejor) con la palanca, frente a la densa v2 en los MISMOS conjuntos",
        "por_conjunto": "diferencia emparejada (palanca − densa v2) por repetición y pliegue, "
                        "intervalo 95 % por bootstrap de pliegues (1.000 remuestras, semilla 0): "
                        "«mejora» si excluye el cero por arriba, «inferioridad» si por abajo",
        "la_palanca_ayuda_si": "sube los cumplidos Y sus «mejora» superan a sus «inferioridad»",
    },
    "combinacion": "las palancas que ayudaron, juntas, sobre los no sellados; si la combinada "
                   "cumple en menos conjuntos que la mejor palanca sola, se queda la mejor sola",
    "confirmacion": "la receta elegida, UNA vez, en los SELLADOS; es la cifra que se publica",
    "liston": "a 2 puntos del mejor en ≥ 80 % de los conjuntos (el de la v2, sin tocar)",
}


def componer() -> dict:
    v2 = json.loads(RUTA_V2.read_text(encoding="utf-8"))
    protocolo = {
        "version_protocolo": "118-C0.v3",
        "fecha_registro": "2026-09-25",
        "base": {"protocolo": "113-C0.v2", "digest_sha256": v2["digest_sha256"]},
        "motor": "matrixai.dense.torch_cpu",
        "receta_base": next(m["receta"] for m in v2["motores"] if m["id"] == "matrixai.dense.torch_cpu"),
        "palancas": PALANCAS,
        "veredicto": VEREDICTO,
        "coste_estimado": {
            "densa_v2_en_no_sellados_h_de_intentos": 7.54,
            "procesos_en_paralelo": v2["presupuesto"]["procesos_en_paralelo"],
            "h_reales_por_pasada_de_palanca": 3.8,
            "nota": "medido sobre la pasada v2 (387 intentos de la densa en los no sellados); "
                    "C2 solo corre las regresiones y C3 debería acortar (lote grande). "
                    "Cada pasada, a la COLA NOCTURNA.",
        },
    }
    for campo in CAMPOS_COPIADOS_DE_LA_V2:
        protocolo[campo] = v2[campo]
    canonico = json.dumps(protocolo, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    protocolo["digest_sha256"] = hashlib.sha256(canonico.encode("utf-8")).hexdigest()
    return protocolo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comprobar", action="store_true")
    a = ap.parse_args(argv)
    nuevo = json.dumps(componer(), ensure_ascii=False, indent=2) + "\n"
    if a.comprobar:
        if not RUTA_V3.is_file() or RUTA_V3.read_text(encoding="utf-8") != nuevo:
            print("protocolo_118_v3.json NO coincide con lo que compone este guion", file=sys.stderr)
            return 1
        print("protocolo_118_v3.json: coincide")
        return 0
    RUTA_V3.write_text(nuevo, encoding="utf-8")
    print(RUTA_V3, json.loads(nuevo)["digest_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
