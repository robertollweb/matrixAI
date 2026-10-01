# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-WEB (campo) — LA TABLA DE LOS 40 CONJUNTOS EN EL CAMPO DE LA RED NUEVA, GENERADA DESDE
LOS REGISTROS SELLADOS. Ninguna cifra se escribe a mano.

«Cómo medimos» enseñaba arriba la medición v1 del 16-09 (101-C5) y la red nueva (TabM + PLR,
`matrixai.dense.tabm_cpu`) solo al final. Decidido: la página pasa al campo de la red nueva, que
es EXACTAMENTE con el que se contó el 38/40: los motores de la v2 CONGELADA (113, 22-09) con sus
resultados, y en el sitio de la densa v2, la red nueva medida en 119-C3 y 119-C4.

    python3 benchmarks/datos_publicos/generar_campo_119.py            # comprueba el publicado
    python3 benchmarks/datos_publicos/generar_campo_119.py --escribir # lo regenera

**No duplica la regla.** Importa `pasada_119_c3` (campo, fallos y `_cumplidos_de` con detalle: lo
mismo que `recuento_119_con_el_motor_nuevo_en_el_campo.py`), `generar_red_119` (verificación de
sellos) y `generar` (conjuntos, regla y procedencia de la v2, y la columna de la red anterior).

**Qué se verifica, y PARA si no cuadra:** los dos sellos de C3 y de C4; que la v2 en disco es la
que C4 usó (sha256) y la que publica `fase0_publico.json`; que los totales por motor son IGUALES
a los del recuento ya escrito (red nueva 38 y mejor en 23, lightgbm 26, catboost 25, sklearn.hgb
25, xgboost 24, lineal 13) y el de la red nueva al X/40 de C4; que la red anterior da el 16 de C4;
que todos los motores coinciden en quién es «el mejor» de cada conjunto; y que la distancia
recalculada desde las medias es la que escribió la regla.

**Procedencia, tal cual:** la v2 NO es anclable entera (110 de sus 3.479 intentos se midieron con
el árbol de matrixAI sucio al relanzar tras la caída en `kick`); C3 y C4 sí llevan commits y
sellos. Viaja dentro del JSON, por bloque.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FASE0 = RAIZ / "benchmarks" / "fase0"
for ruta in (str(AQUI), str(RAIZ), str(FASE0)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

import generar as g_v2  # noqa: E402 -- conjuntos, regla y procedencia de la v2
import generar_red_119 as g_red  # noqa: E402 -- sellos y lectura

DatosQueNoCuadran = g_red.DatosQueNoCuadran
MOTOR_NUEVO = g_red.MOTOR_NUEVO
RED_ANTERIOR = "matrixai.dense.torch_cpu"
BASELINE = "baseline"

RUTA_C3 = g_red.RUTA_C3
RUTA_C4 = g_red.RUTA_C4
RUTA_RECUENTO = g_red.RUTA_RECUENTO
RUTA_V2 = g_red.RUTA_V2
RUTA_PUBLICADO = AQUI / "campo_119_publico.json"

_exigir = g_red._exigir


def _media(valores: list[float]) -> float:
    return sum(valores) / len(valores)


def componer() -> dict[str, Any]:
    c3, c4, recuento = g_red._cargar(RUTA_C3), g_red._cargar(RUTA_C4), g_red._cargar(RUTA_RECUENTO)
    g_red._verificar_sellos("resultado_pasada_119_c3.json", c3)
    g_red._verificar_sellos("resultado_pasada_119_c4.json", c4)
    _exigir(recuento["fuentes"]["c3"] == c3["digest_resultados_crudos"]
            and recuento["fuentes"]["c4"] == c4["digest_resultados_crudos"],
            "el recuento cita para C3 o C4 un sello que no es el de los resultados en disco")
    v = c4["veredicto_x_de_40"]
    _exigir(v is not None and c3["tipo_de_ejecucion"] == "pasada" and not c3["parcial"]
            and c4["tipo_de_ejecucion"] == "pasada" and not c4["parcial"],
            "C3 o C4 no son una pasada real completa")
    _exigir(c3["protocolo_119_v4_digest_sha256"] == c4["protocolo_119_v4_digest_sha256"],
            "C3 y C4 no se midieron con el mismo protocolo")
    sha_v2 = hashlib.sha256(RUTA_V2.read_bytes()).hexdigest()
    _exigir(c4["procedencia"]["datos_de_entrada"]["pasada_v2_113_resultado"]["sha256"] == sha_v2,
            "la v2 en disco no es la que C4 registró por sha256")

    # --- la v2 congelada, con el MISMO `componer` que la página ya usa --------------------
    v2 = g_red._cargar(RUTA_V2)
    protocolo_v2 = g_red._cargar(g_v2.RUTA_PROTOCOLO_ULTIMA_MEDICION)
    bloque_v2 = g_v2.componer(v2, protocolo_v2, RUTA_V2.name)
    publicado_v2 = g_red._cargar(g_v2.RUTA_PUBLICADO)["ultima_medicion"]
    _exigir(publicado_v2["fuente"]["digest_resultados_crudos"] == v2["digest_resultados_crudos"],
            "fase0_publico.json no publica la v2 que C4 usó")

    # --- el campo y la regla, con las funciones de 119 (importadas, no copiadas) ----------
    import pasada_119_c3 as p  # noqa: E402 -- pesado: después de los sellos

    nuevo = c3["resultados"] + c4["resultados"]
    p.c6.preparar_protocolo_v2()
    protocolo = p.c3.protocolo_registrado()
    todos = p.c5.datasets_de_la_pasada(protocolo)
    nombres = [d.nombre for d in todos]
    metrica = p.c5.metrica_de_cierre_por_dataset(protocolo, todos)
    resultados_v2 = p._payload_v2()["resultados"]
    esperadas: dict[str, set] = collections.defaultdict(set)
    for r in nuevo:
        esperadas[r["dataset"]].add((r["repeticion"], r["pliegue"]))
    _exigir(set(esperadas) == set(nombres), "los conjuntos del motor nuevo no son los 40 del protocolo")
    _exigir(set(nombres) == {c["nombre"] for c in bloque_v2["conjuntos"]},
            "los conjuntos del protocolo 119 no son los de la v2 publicada")

    campo = p.p118.campo_de_la_comparacion(resultados_v2, nuevo, set(nombres))
    motores = sorted({r["motor"] for r in campo})
    _exigir(RED_ANTERIOR not in motores and MOTOR_NUEVO in motores,
            "en el campo la red anterior tiene que estar sustituida por la nueva")

    # valores medidos por (conjunto, motor), de los registros crudos del campo
    valores: dict[tuple, list[float]] = collections.defaultdict(list)
    fallos_n: collections.Counter = collections.Counter()
    for r in campo:
        clave = (r["dataset"], r["motor"])
        if r["estado"] != g_v2.ESTADO_QUE_CUENTA:
            fallos_n[clave] += 1
            continue
        x = r.get(metrica[r["dataset"]])
        if x is not None:
            valores[clave].append(float(x))

    detalle: dict[tuple, dict] = {}
    totales: dict[str, dict] = {}
    for motor in motores:
        fallos = []
        for n in nombres:
            de_aqui = [r for r in campo if r["dataset"] == n and r["motor"] == motor]
            fallos += p.registros_de_los_fallos(
                n, motor, p.fallos_del_motor(de_aqui, motor, metrica[n], sorted(esperadas[n])))
        c = p._cumplidos_de(campo + fallos, protocolo.regla_de_cierre, motor=motor,
                            metrica_por_dataset=metrica, nombres_de_los_conjuntos=nombres,
                            con_detalle=True)
        for e in c["detalle"]:
            detalle[(e["dataset"], motor)] = e
        totales[motor] = {"cumplidos": c["cumplidos"], "de": c["datasets"], "fraccion": c["fraccion"],
                          "mejor_en": sum(1 for e in c["detalle"] if e.get("mejor") == motor)}

    # --- CONTROL: iguales al recuento ya escrito y al X/40 de C4 -------------------------
    esperado = {m: {"cumplidos": t["cumplidos"], "de": t["de"], "mejor_en": t["mejor_en"]}
                for m, t in recuento["tabla"].items()}
    _exigir({m: {k: t[k] for k in ("cumplidos", "de", "mejor_en")} for m, t in totales.items()} == esperado,
            f"CONTROL: los totales por motor no son los del recuento: {totales} frente a {esperado}")
    _exigir(totales[MOTOR_NUEVO]["cumplidos"] == v["x_de_40"],
            "CONTROL: la red nueva no da el X/40 de C4")
    cumplidos_v2 = {m: x["cumplidos"] for m, x in bloque_v2["veredicto"].items()}
    _exigir(cumplidos_v2[RED_ANTERIOR] == v["densa_v2_en_los_mismos_conjuntos"]["cumplidos"],
            "CONTROL: la red anterior en la v2 no da el número que C4 escribió")

    conjuntos = []
    por_nombre_v2 = {c["nombre"]: c for c in bloque_v2["conjuntos"]}
    for n in sorted(nombres, key=str.lower):
        mejores = {detalle[(n, m)]["mejor"] for m in motores if m != BASELINE}
        _exigir(len(mejores) == 1, f"{n}: los motores no coinciden en quién es el mejor: {mejores}")
        mejor = mejores.pop()
        celdas: dict[str, Any] = {}
        for m in motores:
            vs = valores.get((n, m), [])
            celda: dict[str, Any] = {"media": _media(vs) if vs else None, "n_medidas": len(vs),
                                     "n_fallos": fallos_n[(n, m)], "compite": m != BASELINE}
            if m != BASELINE:
                d = detalle[(n, m)]
                celda.update({"cumple": d["cumple"], "perdido_por_fallo": d["perdido_por_fallo"],
                              "distancia_en_puntos": d["distancia_en_puntos"],
                              "es_el_mejor": d["mejor"] == m})
                if d["distancia_en_puntos"] is not None and celda["media"] is not None:
                    recal = (_media(valores[(n, mejor)]) - celda["media"]) * 100.0
                    _exigir(abs(recal - d["distancia_en_puntos"]) <= g_v2.TOLERANCIA_EN_PUNTOS,
                            f"{n}/{m}: la distancia recalculada desde las medias es {recal} y la "
                            f"regla escribió {d['distancia_en_puntos']}")
            celdas[m] = celda
        ra = bloque_v2["resultados"][n][RED_ANTERIOR]
        meta = por_nombre_v2[n]
        conjuntos.append({
            **meta, "mejor": mejor, "celdas": celdas,
            "red_anterior": {k: ra.get(k) for k in ("media", "n_medidas", "n_fallos", "cumple",
                                                     "perdido_por_fallo", "distancia_en_puntos",
                                                     "es_el_mejor")},
        })

    # --- procedencia, tal cual ----------------------------------------------------------
    pv2 = bloque_v2["procedencia"]
    sucios = sum(x["n_intentos"] for x in pv2["por_procedencia"] if x["repositorios_sucios"])

    def _de(c: dict, ruta: Path) -> dict:
        repos = c["procedencia"]["repositorios"]
        return {"artefacto": ruta.name, "corte": c["corte"], "creado": c["creado"],
                "medido": c["procedencia"]["medido"], "n_intentos": c["n_intentos"],
                "digest_resultados_crudos": c["digest_resultados_crudos"],
                "digest_solo_de_resultados": c["digest_solo_de_resultados"],
                "commits": {r: x["commit"] for r, x in repos.items()},
                "arbol_sucio": any(x["arbol_sucio"] for x in repos.values())}

    return {
        "formato": "119-CAMPO.v1",
        "motor_nuevo": MOTOR_NUEVO,
        "red_anterior": RED_ANTERIOR,
        "motores": motores,
        "regla": bloque_v2["regla"],
        "conjuntos": conjuntos,
        "totales": {m: dict(sorted(t.items())) for m, t in sorted(totales.items())},
        "sin_la_red_nueva": {
            "que_es": "la v2 congelada (113, 22-09): los motores que había entonces, la red "
                      "anterior incluida y SIN la red nueva en el campo",
            "por_motor": {m: {k: x[k] for k in ("cumplidos", "datasets", "fraccion",
                                                "aciertos_por_ser_el_mejor")}
                          for m, x in sorted(bloque_v2["veredicto"].items())},
        },
        "control": {
            "totales_iguales_al_recuento": RUTA_RECUENTO.name,
            "red_nueva_es_el_x_de_40_de_c4": v["x_de_40"],
            "red_anterior_es_el_de_c4": v["densa_v2_en_los_mismos_conjuntos"]["cumplidos"],
        },
        "procedencia": {
            "v2": {**bloque_v2["fuente"], "anclable_entera": pv2["anclable_entera"],
                   "intentos_con_arbol_sucio": sucios,
                   "por_procedencia": pv2["por_procedencia"]},
            "c3": _de(c3, RUTA_C3),
            "c4": _de(c4, RUTA_C4),
            "sellos_verificados": True,
        },
        "lo_que_no_dicen": bloque_v2["lo_que_no_dicen"],
    }


def serializar(datos: dict[str, Any]) -> str:
    """UNA forma de escribirlo, para que la comparación byte a byte signifique algo."""
    return json.dumps(datos, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def generar() -> str:
    return serializar(componer())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escribir", action="store_true", help="regenera el JSON publicado")
    args = parser.parse_args(argv)
    texto = generar()
    if args.escribir:
        RUTA_PUBLICADO.write_text(texto, encoding="utf-8")
        print(f"escrito {RUTA_PUBLICADO} ({len(texto.encode('utf-8'))} bytes)")
        return 0
    publicado = RUTA_PUBLICADO.read_text(encoding="utf-8") if RUTA_PUBLICADO.exists() else None
    if publicado != texto:
        print("el JSON publicado NO es el que sale de los registros: regenerarlo con --escribir "
              "(y mirar por qué cambió)", file=sys.stderr)
        return 1
    print("el JSON publicado es el que sale de los registros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
