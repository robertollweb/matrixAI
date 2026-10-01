# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119, tras C4 — cuántos conjuntos cumple CADA motor cuando el motor nuevo
(`matrixai.dense.tabm_cpu`) está en el campo. Solo CUENTA: no mide nada.

El 33/40 de lightgbm y de sklearn.hgb (pasada v2, 113-C5) se contó SIN el motor
nuevo. «Cumplir» es quedar a ≤2 puntos del MEJOR del campo, y con el motor nuevo
dentro «el mejor» sube en los conjuntos donde gana: los demás cumplen menos.
Poner «38/40 frente a 33/40» mezcla dos campos distintos; esto los pone en el
mismo.

Cómo: los resultados ya sellados de C3 (32 no sellados) y C4 (8 sellados) del
motor nuevo, la tabla de la v2 CONGELADA para los demás, y las MISMAS funciones
de 119-C3/C4 (`campo_de_la_comparacion`, `fallos_del_motor`,
`registros_de_los_fallos`, `_cumplidos_de`; la regla de cierre del protocolo).
Los intentos esperados de cada conjunto son los que el motor nuevo completó:
C3 y C4 no tuvieron ni un fallo, así que coinciden con el plan.

Control: el motor nuevo tiene que salir con 38, el X/40 que compuso C4; si no,
el guion PARA.

    python3 benchmarks/fase0/recuento_119_con_el_motor_nuevo_en_el_campo.py   # segundos
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

import pasada_119_c3 as p  # noqa: E402

SALIDA = _AQUI / "recuento_119_con_el_motor_nuevo_en_el_campo.json"


def main() -> int:
    c3 = json.loads((_AQUI / "resultado_pasada_119_c3.json").read_text(encoding="utf-8"))
    c4 = json.loads((_AQUI / "resultado_pasada_119_c4.json").read_text(encoding="utf-8"))
    nuevo = c3["resultados"] + c4["resultados"]
    p.c6.preparar_protocolo_v2()
    protocolo = p.c3.protocolo_registrado()
    todos = p.c5.datasets_de_la_pasada(protocolo)
    nombres = [d.nombre for d in todos]
    metrica = p.c5.metrica_de_cierre_por_dataset(protocolo, todos)
    v2 = p._payload_v2()["resultados"]

    esperadas: dict[str, set] = defaultdict(set)
    for r in nuevo:
        esperadas[r["dataset"]].add((r["repeticion"], r["pliegue"]))
    if set(esperadas) != set(nombres):
        raise SystemExit(f"los conjuntos del motor nuevo no son los 40 del protocolo: "
                         f"{sorted(set(nombres) ^ set(esperadas))}")

    campo = p.p118.campo_de_la_comparacion(v2, nuevo, set(nombres))
    tabla = {}
    for motor in sorted({r["motor"] for r in campo}):
        fallos = []
        for n in nombres:
            de_aqui = [r for r in campo if r["dataset"] == n and r["motor"] == motor]
            fallos += p.registros_de_los_fallos(
                n, motor, p.fallos_del_motor(de_aqui, motor, metrica[n], sorted(esperadas[n])))
        cumplidos = p._cumplidos_de(campo + fallos, protocolo.regla_de_cierre, motor=motor,
                                    metrica_por_dataset=metrica, nombres_de_los_conjuntos=nombres,
                                    con_detalle=True)
        tabla[motor] = {"cumplidos": cumplidos["cumplidos"], "de": cumplidos["datasets"],
                        "mejor_en": sum(1 for e in cumplidos["detalle"] if e.get("mejor") == motor)}

    if tabla[p.NOMBRE_MOTOR_NUEVO]["cumplidos"] != c4["veredicto_x_de_40"]["x_de_40"]:
        raise SystemExit(f"CONTROL: el motor nuevo sale con {tabla[p.NOMBRE_MOTOR_NUEVO]['cumplidos']} "
                         f"y C4 compuso {c4['veredicto_x_de_40']['x_de_40']}: el recuento no es el de C4")
    salida = {
        "que_es": "cumplidos de CADA motor sobre los 40 con el motor nuevo en el campo (en el sitio de "
                  "la densa v2); mismas funciones que 119-C3/C4; solo cuenta, no mide",
        "fuentes": {"c3": c3["digest_resultados_crudos"], "c4": c4["digest_resultados_crudos"],
                    "v2": "pasada_v2_113_resultado.json (congelada)"},
        "control": f"el motor nuevo da {tabla[p.NOMBRE_MOTOR_NUEVO]['cumplidos']}, el X/40 de C4",
        "tabla": tabla,
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(tabla, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
