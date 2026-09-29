# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 (4) — LAS PASADAS DE JEV Y SU COSTE, REGISTRADOS DE LO QUE PASÓ (2026-09-29).

**Por qué existe.** `resultado_c30_jev_<T>.json` guarda los segundos que ESA ejecución pasó
preguntando. La tarea C necesitó cuatro: la primera la paró un corte de red (`TimeoutError`, 212 s),
la segunda el saldo de la cuenta (HTTP 402, 3.053 s), la tercera el tope de 3 h de la cola, y la
cuarta terminó el resto desde la caché en 1.141 s. El resultado dice 1.122 s de preguntas, que es solo
la última: publicarlo como «lo que tardó Jev» sería declarar lo que NO pasó. Aquí se registra cada
pasada con su línea de origen, copiada tal cual, y el generador de los datos públicos
(`benchmarks/datos_publicos/generar_texto_107.py`) suma desde aquí.

**De dónde sale cada cifra.** Las pasadas de la cola, de `~/cola-nocturna/resumen.txt` (la línea que
escribe `cola-nocturna.sh` al terminar cada trabajo: nombre, `rc` y segundos de pared). La última,
lanzada a mano con el mismo código (`~/.agentes/107c30-jev/resto_de_c.sh`), de su registro. El coste,
de `GET https://openrouter.ai/api/v1/key` (`usage`, en dólares) con la clave de `jev.env`, que NO se
escribe en ninguna parte: esa clave solo se ha usado para esta medida (A, C y unas pocas llamadas de
prueba con un texto inventado, ~0,00001 $ cada una), así que su uso ES el coste de la medida.

    python3 benchmarks/texto_107c30/registrar_pasadas_jev.py   # escribe pasadas_c30_jev.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "pasadas_c30_jev.json"
RESUMEN_DE_LA_COLA = Path("/home/deployer/cola-nocturna/resumen.txt")
REGISTRO_DEL_RESTO = Path("/home/deployer/.agentes/107c30-jev/resto_de_c.log")

#: Qué trabajo de la cola midió qué tarea (los nombres con que se encolaron).
TAREA_DE_LA_PASADA = {"107c30-jev-a": "A", "107c30-jev-c": "C", "107c30-jev-c2": "C", "107c30-jev-c3": "C"}
#: Lo que `cola-nocturna.sh` quiere decir con cada `rc` (su propia línea lo explica igual).
DESENLACE = {0: "completada", 1: "error", 124: "tope_de_tiempo", 137: "techo_de_memoria", 143: "techo_de_memoria"}

_LINEA_COLA = re.compile(r"^(?P<fin>\S+) (?P<nombre>107c30-jev-[a-z0-9]+) rc=(?P<rc>\d+) (?P<s>\d+)s\b")
_LINEA_RESTO = re.compile(r"^--- tarea (?P<tarea>[A-Z]): (?P<s>[0-9.]+)s, escrito en ")
_REGISTRO_EN_LA_LINEA = re.compile(r" · registro (?P<ruta>\S+)")
#: LA CAUSA de una pasada que acabó en error, de la última línea de excepción de su registro: una página
#: que dijera solo «error» haría pensar que falló Jev, y las dos de C fueron la red y el saldo de la cuenta.
_CAUSAS = ((re.compile(r"^TimeoutError\b"), "corte_de_red"),
           (re.compile(r"ErrorJevFatal: HTTP 402\b"), "sin_saldo_en_la_cuenta"))


def _causa_del_error(linea_de_la_cola: str) -> str | None:
    m = _REGISTRO_EN_LA_LINEA.search(linea_de_la_cola)
    if not m or not Path(m["ruta"]).is_file():
        return None
    for linea in reversed(Path(m["ruta"]).read_text(encoding="utf-8", errors="replace").splitlines()):
        for patron, causa in _CAUSAS:
            if patron.search(linea.strip()):
                return causa
    return "otra"


def _pasadas() -> list[dict]:
    pasadas = []
    for linea in RESUMEN_DE_LA_COLA.read_text(encoding="utf-8").splitlines():
        m = _LINEA_COLA.match(linea)
        if not m or m["nombre"] not in TAREA_DE_LA_PASADA:
            continue
        rc = int(m["rc"])
        pasadas.append({"nombre": m["nombre"], "tarea": TAREA_DE_LA_PASADA[m["nombre"]], "fin": m["fin"],
                        "rc": rc, "desenlace": DESENLACE.get(rc, "otro"), "segundos_de_pared": int(m["s"]),
                        "origen": "cola-nocturna/resumen.txt", "linea_de_origen": linea.split(" · registro")[0]})
        if rc not in (0, 124):
            pasadas[-1]["causa"] = _causa_del_error(linea)
    for linea in REGISTRO_DEL_RESTO.read_text(encoding="utf-8").splitlines():
        m = _LINEA_RESTO.match(linea)
        if m:
            pasadas.append({"nombre": "resto-a-mano", "tarea": m["tarea"], "fin": None, "rc": 0,
                            "desenlace": "completada", "segundos_de_pared": float(m["s"]),
                            "origen": "~/.agentes/107c30-jev/resto_de_c.log",
                            "linea_de_origen": linea.split(", escrito en ")[0]})
    return pasadas


def _uso_de_la_clave_usd() -> float:
    sys.path.insert(0, str(AQUI))
    from jev import respondedor_jev as rj  # noqa: PLC0415

    clave = rj.cargar_clave_api()
    peticion = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                      headers={"Authorization": f"Bearer {clave}"})
    with urllib.request.urlopen(peticion, timeout=30) as r:
        return float(json.loads(r.read().decode())["data"]["usage"])


def main() -> None:
    pasadas = _pasadas()
    por_tarea: dict[str, dict] = {}
    for p in pasadas:
        t = por_tarea.setdefault(p["tarea"], {"pasadas": [], "segundos_de_pared_total": 0.0})
        t["pasadas"].append(p)
        t["segundos_de_pared_total"] += p["segundos_de_pared"]
    if not any(p["desenlace"] == "completada" for t in por_tarea.values() for p in t["pasadas"]):
        raise SystemExit("ninguna pasada completada: no hay nada que registrar")
    datos = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0, condición (4) -- Jev",
        "por_tarea": por_tarea,
        "coste": {"uso_de_la_clave_usd": _uso_de_la_clave_usd(),
                  "cubre": "tareas A y C y unas pocas llamadas de prueba con un texto inventado (~0,00001 $ "
                           "cada una); la clave no se usó para nada más",
                  "fuente": "GET https://openrouter.ai/api/v1/key (data.usage)"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"escrito {SALIDA}: " + ", ".join(f"{t} {len(v['pasadas'])} pasada(s), {v['segundos_de_pared_total']:.0f} s"
                                            for t, v in sorted(por_tarea.items()))
          + f"; coste {datos['coste']['uso_de_la_clave_usd']:.4f} $")


if __name__ == "__main__":
    main()
