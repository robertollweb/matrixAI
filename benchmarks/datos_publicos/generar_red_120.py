# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""120-WEB — LA RED NUEVA EN EL STUDIO: los datos públicos de 120-C3′ y 120-C3‴, GENERADOS DESDE
LOS RESULTADOS de esas medidas.

Mismo patrón que `generar_red_119.py`: **ninguna cifra se escribe a mano**. Este guion lee los dos
resultados (`resultado_120_c3p.json`, el paquete con GPU; `resultado_120_c3s2.json`, el paquete
sin GPU con la guarda de memoria) y compone `red_120_en_el_studio.json`, que la web (portada y
«Cómo medimos») enseña tal cual. Una prueba (`tests/test_120_web_datos_publicos.py`) lo regenera y
lo compara BYTE A BYTE con el publicado.

    python3 benchmarks/datos_publicos/generar_red_120.py            # comprueba el publicado
    python3 benchmarks/datos_publicos/generar_red_120.py --escribir # lo regenera

**Qué se COPIA**: el veredicto de cada medida (cuántos suben, bajan, dejan de completar…), la
diferencia de cada conjunto que sube (la cifra del campeón en el TEST, en puntos, tal cual la
escribió `veredicto_120_c3p.py`), los campeones de antes y de después, los conjuntos donde la guarda
dejó fuera a la red con sus dos cifras (memoria estimada y disponible, del propio estudio), la
memoria del contenedor, el entorno de la medida y la fecha de inicio. **Qué se COMPRUEBA, y PARA si
no cuadra**: que cada veredicto diga `mejora`, que sus «suben» sean exactamente los conjuntos de clase
`sube` de sus comparaciones, y que ninguno baje. Nunca se publica un número calculado aquí.

Salida determinista (claves ordenadas, sin ninguna fecha de «ahora»).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FASE0 = RAIZ / "benchmarks" / "fase0"
SALIDA = AQUI / "red_120_en_el_studio.json"
MOTOR = "matrixai.dense.tabm_cpu"

#: Las dos medidas que se publican: el paquete con GPU (la red, un motor más; la anterior sigue) y el
#: paquete sin GPU (la red en el sitio de la anterior, con la guarda de memoria).
MEDIDAS = {
    "paquete_gpu": {"resultado": "resultado_120_c3p.json", "corte": "C3′", "enmienda": 7},
    "paquete_cpu": {"resultado": "resultado_120_c3s2.json", "corte": "C3‴", "enmienda": 9},
}


class NoCuadra(Exception):
    pass


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _medida(nombre: str, spec: dict[str, Any]) -> dict[str, Any]:
    ruta = FASE0 / spec["resultado"]
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    veredicto = datos["veredicto"]
    comparaciones = datos["comparaciones"]
    if veredicto.get("mejora") is not True:
        raise NoCuadra(f"{nombre}: el veredicto no dice «mejora»")
    suben = sorted(k for k, c in comparaciones.items() if c["clase"] == "sube")
    if len(suben) != veredicto["suben"]:
        raise NoCuadra(f"{nombre}: el veredicto dice {veredicto['suben']} suben y las comparaciones {len(suben)}")
    if veredicto["bajan"] != 0 or veredicto["dejan_de_completar"] != 0:
        raise NoCuadra(f"{nombre}: algo baja o deja de completar")
    fuera_por_memoria = []
    for conjunto, registro in sorted(datos["conjuntos"].items()):
        motivo = (registro.get("motores_fuera_por_memoria") or {}).get(MOTOR)
        if motivo:
            fuera_por_memoria.append({"conjunto": conjunto, "estimada_gb": motivo["estimada_gb"],
                                      "disponible_gb": motivo["disponible_gb"]})
    procedencia = datos["procedencia"]
    return {
        "corte": spec["corte"],
        "enmienda": spec["enmienda"],
        "veredicto": {k: veredicto[k] for k in ("suben", "bajan", "dejan_de_completar", "iguales",
                                                 "ninguno_completa", "tabm_campeon_en", "conjuntos")},
        "suben": [{"conjunto": k, "diferencia_en_puntos": comparaciones[k]["diferencia"],
                   "campeon_antes": comparaciones[k]["campeon_r1"],
                   "campeon": comparaciones[k]["campeon_c3p"]} for k in suben],
        "fuera_por_memoria": fuera_por_memoria,
        "condiciones": {"memoria_del_contenedor": procedencia.get("memoria_del_contenedor"),
                        "entorno_extra": procedencia.get("entorno_extra") or {},
                        "inicio": procedencia["inicio"]},
        "fuente": {"fichero": f"benchmarks/fase0/{spec['resultado']}", "sha256": _sha256(ruta)},
    }


def componer() -> dict[str, Any]:
    return {"motor": MOTOR, **{nombre: _medida(nombre, spec) for nombre, spec in MEDIDAS.items()}}


def serializar(datos: dict[str, Any]) -> str:
    return json.dumps(datos, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escribir", action="store_true")
    args = parser.parse_args(argv)
    try:
        texto = serializar(componer())
    except NoCuadra as exc:
        print(f"NO CUADRA: {exc}", file=sys.stderr)
        return 2
    if args.escribir:
        SALIDA.write_text(texto, encoding="utf-8")
        print(f"escrito {SALIDA}")
        return 0
    if not SALIDA.exists() or SALIDA.read_text(encoding="utf-8") != texto:
        print(f"{SALIDA.name} no es el que se genera hoy: regenerar con --escribir", file=sys.stderr)
        return 1
    print("el publicado es el que se genera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
