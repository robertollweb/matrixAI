#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Escribe `protocolo_sino_b.json`: la REGLA de la palanca de las sí/no de la densa, registrada
ANTES de medir (decisión de Roberto del 2026-10-04 11:47, «opción B»).

La palanca (`matrixai_engines.motores.densa.PALANCA_SINO_CATEGORICA`) genera `categorical`
cada columna de entrada sí/no que hoy el núcleo tipa `boolean`, para que un hueco que el
entrenamiento de la parte no tuvo no pierda la parte. Cambia lo que aprende la red en esas
columnas, así que entra en la densa de por omisión SOLO si no empeora frente a la de hoy, con
esta regla. El digest se calcula como en `generar_protocolo_118_v3.py`: sha256 del JSON
canónico (claves ordenadas, sin espacios) sin el propio campo.

    python3 benchmarks/fase0/generar_protocolo_sino_b.py            # escribe
    python3 benchmarks/fase0/generar_protocolo_sino_b.py --comprobar # el escrito es el de aquí
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
RUTA = _AQUI / "protocolo_sino_b.json"
RUTA_V2 = _AQUI / "protocolo_exploratorio_v2.json"


def componer() -> dict:
    v2 = json.loads(RUTA_V2.read_text(encoding="utf-8"))
    protocolo = {
        "version_protocolo": "sino-b.v1",
        "fecha_registro": "2026-10-04",
        "registrado_por": "deployer-d4",
        "decision": ("Roberto, 2026-10-04 11:47: OPCIÓN B (TASKS.md, «LA MISMA PARTE PERDIDA, EN "
                     "UNA COLUMNA BOOLEANA»): la red anterior trata como categórica de dos "
                     "valores cada sí/no de entrada; versión nueva del motor, medida ANTES con la "
                     "regla escrita antes de medir; entra solo si no empeora"),
        "estado": ("REGISTRADA antes de medir nada de calidad de la palanca. Lo único medido "
                   "antes: la sonda de UNA parte de sick con un hueco inyectado, que demuestra "
                   "el defecto (la de hoy pierde la parte, la B la predice), no compara calidad"),
        "base": {"protocolo": v2.get("version_protocolo"), "digest_sha256": v2["digest_sha256"],
                 "que_se_hereda": ("la partición, las semillas, las repeticiones, el tope de "
                                   "pared por cubo y la métrica de cierre por tarea de la v2, "
                                   "tal cual: las dos variantes corren los MISMOS pliegues")},
        "motor": "matrixai.dense.torch_cpu",
        "variantes": {
            "hoy": "MotorDensaPropia() — palanca=None, la densa de hoy byte a byte",
            "B": "MotorDensaPropia(palanca='04-10.sino_categorica')",
        },
        "conjuntos": ["sick", "house_prices_nominal"],
        "criterio_de_los_conjuntos": (
            "los NO sellados de la Fase 0 con al menos un predictor sí/no TAL COMO LE LLEGA a la "
            "densa (texto cuyos valores el núcleo tipa `boolean`, tras `tipar_columnas_"
            "numericas`). Medido antes de registrar, con el lector del arnés sobre los 40 ARFF: "
            "sick (20 columnas), house_prices_nominal (1: CentralAir). En los demás la palanca "
            "no cambia nada (lo sostiene `tests/test_sino_categorica.py` en motores)"),
        "excluidos": {
            "kr-vs-kp": ("SELLADO (34 columnas sí/no): el examen final de la cartera no se usa "
                         "para decidir nada; si la palanca entra, NO se mide ahí"),
        },
        "regla": {
            "por_conjunto": (
                "diferencia emparejada (B − hoy) de la métrica de cierre de la v2 (AUROC en "
                "binaria, R² en regresión) por repetición y pliegue; intervalo al 95 % por "
                "bootstrap de PLIEGUES, 1.000 remuestras, semilla 0 (`pasada_114c6_ensamblado."
                "diferencias_emparejadas` + `protocolo._intervalo_pareado`, los de 118 y C6): "
                "«inferioridad» si el intervalo queda entero por debajo de 0, «mejora» si entero "
                "por encima"),
            "completitud": ("B empeora también si un intento que la de hoy completa no lo "
                            "completa B (falla o agota el tope)"),
            "entra_si": ("NINGÚN conjunto da «inferioridad» Y B completa todo lo que completa "
                         "la de hoy"),
            "si_entra": ("la palanca pasa a ser la densa de por omisión, con versión nueva "
                         "(1.2.0), auditoría Opus y ff sin push"),
            "si_no_entra": ("se queda la de hoy y la parte con el hueco se sigue perdiendo; se "
                            "le lleva a Roberto con los números (las opciones A y D están "
                            "descritas en TASKS.md)"),
        },
        "coste": {
            "estimado_s": 3533,
            "como_se_estima": ("la pasada v2 de la densa en estos dos conjuntos tardó 314 s "
                               "(sick, 15 intentos) + 864 s (house_prices_nominal, 15) = 1.178 s "
                               "por variante; × 2 variantes = 2.356 s; × 1,5 de margen"),
            "cota_peor_caso_s": 12600,
            "como_se_acota": ("30 intentos × 300 s de tope (sick, cubo mediano) + 30 × 120 s "
                              "(house_prices_nominal, cubo pequeño)"),
            "recursos": "cola nocturna: techo 4G sin swap, 5 CPU; tope de la cola 14.000 s",
        },
    }
    canonico = json.dumps(protocolo, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    protocolo["digest_sha256"] = hashlib.sha256(canonico.encode("utf-8")).hexdigest()
    return protocolo


def main() -> int:
    texto = json.dumps(componer(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if "--comprobar" in sys.argv:
        if RUTA.read_text(encoding="utf-8") != texto:
            print(f"{RUTA} NO es el que compone este generador", file=sys.stderr)
            return 1
        print(f"{RUTA} cuadra con su generador")
        return 0
    RUTA.write_text(texto, encoding="utf-8")
    print(RUTA, json.loads(texto)["digest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
