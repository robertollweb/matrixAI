# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""122-WEB — TabM ELEGIBLE EN EL MODO EXPERTO: los datos públicos de 122-C0, GENERADOS DESDE EL RESULTADO.

Mismo patrón que `generar_red_120.py`: **ninguna cifra se escribe a mano**. Lee
`benchmarks/contrato122/resultado_122_c0.json` y compone `tabm_122_c0.json`, que la web enseña
tal cual (vía `frontend/scripts/copiar-datos-de-tabm-122.mjs`).

    python3 benchmarks/datos_publicos/generar_tabm_122_c0.py            # comprueba el publicado
    python3 benchmarks/datos_publicos/generar_tabm_122_c0.py --escribir # lo regenera

**Qué se COPIA**: el veredicto, cuántos conjuntos se midieron y cuántos cumplen, la regla (puntos y
fracción mínima), los conjuntos donde NO cumple con su distancia en puntos y quién era el mejor, y los
pliegues de TabM cortados por el plazo. **Qué se CUENTA aquí** (no se escribe): los pliegues de TabM
(registros `motor == "tabm"`) y cuántos terminaron `completed` y cuántos pararon por plazo. **Qué se
COMPRUEBA, y PARA si no cuadra**: que `cumplidos` sea el número de conjuntos con `cumple` y `de` el de
conjuntos del detalle; que los cortados por plazo del veredicto sean los registros con `parado_por_plazo`.
Salida determinista (claves ordenadas, sin fecha de «ahora»).
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
FUENTE = RAIZ / "benchmarks" / "contrato122" / "resultado_122_c0.json"
SALIDA = AQUI / "tabm_122_c0.json"


class NoCuadra(Exception):
    pass


def componer() -> dict[str, Any]:
    datos = json.loads(FUENTE.read_text(encoding="utf-8"))
    v = datos["veredicto"]
    detalle = v["detalle"]
    if v["de"] != len(detalle):
        raise NoCuadra(f"`de` dice {v['de']} y el detalle trae {len(detalle)} conjuntos")
    fuera = [d for d in detalle if not d["cumple"]]
    if v["cumplidos"] != len(detalle) - len(fuera):
        raise NoCuadra("`cumplidos` no es el número de conjuntos con `cumple`")
    if v["tabm_completa_todos"] is not True or v["conjuntos_incompletos_de_tabm"]:
        raise NoCuadra("TabM no completó todos los conjuntos")
    tabm = [r for r in datos["resultados"] if r["motor"] == "tabm"]
    if any(r["estado"] != "completed" for r in tabm):
        raise NoCuadra("algún pliegue de TabM no está `completed`")
    cortados = sorted((r["dataset"], r["pliegue"]) for r in tabm if r.get("parado_por_plazo"))
    declarados = sorted((c["dataset"], c["pliegue"]) for c in v["tabm_con_el_plazo_cortado"])
    if cortados != declarados:
        raise NoCuadra(f"cortados por plazo: el veredicto dice {declarados} y los registros {cortados}")
    conjuntos = sorted(d["dataset"] for d in detalle)
    return {
        "corte": datos["corte"],
        "cumplidos": v["cumplidos"],
        "conjuntos": conjuntos,
        "de": v["de"],
        "fuera_de_la_regla": [
            {"conjunto": d["dataset"], "distancia_en_puntos": d["distancia_en_puntos"], "mejor": d["mejor"]}
            for d in sorted(fuera, key=lambda d: d["dataset"])],
        "fuente": {"fichero": "benchmarks/contrato122/resultado_122_c0.json",
                   "sha256": hashlib.sha256(FUENTE.read_bytes()).hexdigest()},
        "pliegues_de_tabm": len(tabm),
        "pliegues_cortados_por_el_plazo": [{"conjunto": c, "pliegue": p} for c, p in cortados],
        "regla": {"fraccion_minima": v["regla"]["fraccion_minima"], "puntos": v["regla"]["puntos"]},
        "tabm_completa_todos": v["tabm_completa_todos"],
        "veredicto": v["veredicto"],
    }


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
