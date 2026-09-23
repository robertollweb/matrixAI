# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""117-C1 — LOS DATOS PÚBLICOS DE FASE 0, GENERADOS DESDE EL ARTEFACTO SELLADO.

La página «Cómo medimos» (117-C2) y las fichas de «Ejemplos medidos» (117-C3) enseñan
cifras de la Fase 0. **Ninguna se escribe a mano** (invariante 1 del 117): salen de aquí,
que las lee del artefacto de la pasada y de su protocolo, y una prueba
(`tests/test_117_c1_datos_publicos.py`) regenera el JSON y lo compara byte a byte con el
publicado. Es la regla de la tabla de terceros: se mide, no se escribe.

**Qué pasadas, DECIDIDO por Roberto el 2026-09-23 (117-C5).** El JSON publicado lleva DOS
bloques, cada uno compuesto con el MISMO `componer()` porque los dos artefactos tienen la
misma forma (medido):

- **`principal`**: la pasada AMPLIA (`pasada_amplia_101_c5_resultado.json`, protocolo v1
  `protocolo_exploratorio.json`, digest `3646af61...`). Es la que SOSTIENE la cartera —las
  dos entradas de `matrixai_engines.cartera.CARTERA_APROBADA` sellan este mismo
  `evidencia_digest`— y es anclable entera (los dos repos limpios al medir). La página y las
  fichas de «ejemplos medidos» se enseñan desde aquí.
- **`ultima_medicion`**: la v2 del 113 (`pasada_v2_113_resultado.json`, protocolo v2
  `protocolo_exploratorio_v2.json`, digest `600ebdaa...`), la que decidió el 2026-09-22 que
  catboost sale de la cartera y la única medida con la receta nueva de la red densa. Su
  procedencia NO es anclable entera —110 de sus 3.479 intentos se midieron con el árbol de
  matrixAI sucio al relanzar tras la caída en `kick`—, y eso viaja DENTRO de su propio bloque
  (`ultima_medicion.procedencia`), no en una nota aparte. Se enseña APARTE, como «última
  medición, no anclable entera»: no sustituye al bloque principal ni se mezcla con él.

Hasta el 2026-09-23 este fichero componía TODO desde la v2 porque era «la vigente». Se
corrigió el mismo día: la cartera nunca dejó de citar la amplia para lo que APRUEBA, y
enseñar al público una cifra que la propia cartera no usa para aprobar habría sido dos
sitios declarando cosas distintas. Cuando una pasada nueva se selle y pase a sostener la
cartera, lo que cambia aquí son las rutas de `principal` (117-C5).

**Qué NO calcula.** El veredicto, las distancias al mejor, si cada motor cumple en cada
conjunto: todo eso lo escribió la pasada y aquí se COPIA. Lo único que se calcula es la
MEDIA de cada motor en cada conjunto —la pasada no la guarda en su detalle—, y para que no
sea un segundo sitio con la cuenta, se COMPRUEBA contra la pasada: la distancia al mejor
que sale de estas medias tiene que ser la que la pasada escribió, o no se genera nada. Esto
se aplica a los DOS bloques por igual.

    python3 benchmarks/datos_publicos/generar.py            # comprueba el publicado
    python3 benchmarks/datos_publicos/generar.py --escribir # lo regenera
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
FASE0 = AQUI.parent / "fase0"

#: EL BLOQUE PRINCIPAL: la pasada que sostiene la cartera (117-C5). `matrixai_engines.
#: cartera.CARTERA_APROBADA` sella `evidencia_digest` sobre ESTE artefacto.
RUTA_ARTEFACTO_PRINCIPAL = FASE0 / "pasada_amplia_101_c5_resultado.json"
RUTA_PROTOCOLO_PRINCIPAL = FASE0 / "protocolo_exploratorio.json"

#: EL BLOQUE APARTE: la última medición, no anclable entera. Nunca sustituye al principal.
RUTA_ARTEFACTO_ULTIMA_MEDICION = FASE0 / "pasada_v2_113_resultado.json"
RUTA_PROTOCOLO_ULTIMA_MEDICION = FASE0 / "protocolo_exploratorio_v2.json"

RUTA_PUBLICADO = AQUI / "fase0_publico.json"

#: Qué «medida» es una media aceptable: la de los intentos COMPLETADOS con valor. Un fallo
#: no es un cero (cuenta como conjunto perdido en la regla, y eso ya lo dice `cumple`).
ESTADO_QUE_CUENTA = "completed"

#: Cuánto puede separarse la distancia recalculada de la que escribió la pasada. Es solo
#: redondeo de coma flotante: la media es la misma cuenta.
TOLERANCIA_EN_PUNTOS = 1e-9

#: LO QUE ESTAS CIFRAS NO DICEN (invariante 4 del 117), con el mismo peso que lo que dicen.
#: Viaja en el JSON para que la página y las fichas no puedan enseñar una sin la otra.
LO_QUE_NO_DICEN = (
    {"es": "Los intervalos miden cuánto cambia el resultado al REENTRENAR sobre otros "
           "repartos del mismo conjunto, no cómo le iría al motor con otra muestra de datos.",
     "en": "The intervals measure how much the result changes when RETRAINING on other "
           "splits of the same dataset, not how the engine would do on another data sample."},
    {"es": "«2 puntos» no pesa lo mismo en todas las métricas: 2 puntos de R² no son 2 de AUROC.",
     "en": "“2 points” does not weigh the same in every metric: 2 points of R² are not 2 of AUROC."},
    {"es": "Cada motor corre con sus valores por omisión, sin buscar hiperparámetros: con "
           "búsqueda, el orden podría cambiar.",
     "en": "Each engine runs with its default values, with no hyperparameter search: with a "
           "search, the order could change."},
    {"es": "Un conjunto de diferencia en el recuento no distingue a dos motores.",
     "en": "A one-dataset difference in the count does not tell two engines apart."},
)


class DatosQueNoCuadran(RuntimeError):
    """Lo recalculado aquí no coincide con lo que escribió la pasada: no se publica."""


def _media(valores: list[float]) -> float:
    return sum(valores) / len(valores)


def componer(artefacto: dict[str, Any], protocolo: dict[str, Any], nombre_artefacto: str) -> dict[str, Any]:
    if artefacto.get("parcial") is not False:
        raise DatosQueNoCuadran("la pasada no está terminada (`parcial` no es false)")
    veredicto = artefacto["alcance_y_veredicto"]
    metrica_por_conjunto = veredicto["metrica_de_cierre_por_dataset"]
    por_motor = veredicto["por_motor"]

    # -- los valores de cada (conjunto, motor), de los registros crudos ------------------
    valores: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    fallos: collections.Counter = collections.Counter()
    for r in artefacto["resultados"]:
        clave = (r["dataset"], r["motor"])
        if r["estado"] != ESTADO_QUE_CUENTA:
            fallos[clave] += 1
            continue
        valor = r.get(metrica_por_conjunto[r["dataset"]])
        if valor is not None:
            valores[clave].append(float(valor))

    conjuntos_del_protocolo = sorted(protocolo["datasets"], key=lambda d: d["nombre"])
    motores = sorted({r["motor"] for r in artefacto["resultados"]})
    detalle = {(f["dataset"], m): f for m, v in por_motor.items() for f in v["detalle"]}

    resultados: dict[str, dict[str, Any]] = {}
    for d in conjuntos_del_protocolo:
        nombre = d["nombre"]
        fila: dict[str, Any] = {}
        for m in motores:
            vs = valores.get((nombre, m), [])
            celda: dict[str, Any] = {
                "media": _media(vs) if vs else None,
                "n_medidas": len(vs),
                "n_fallos": fallos[(nombre, m)],
                # El baseline no compite por el primer puesto: la regla lo excluye.
                "compite": m in por_motor,
            }
            f = detalle.get((nombre, m))
            if f is not None:
                celda.update({"cumple": f["cumple"], "perdido_por_fallo": f["perdido_por_fallo"],
                              "distancia_en_puntos": f["distancia_en_puntos"],
                              "es_el_mejor": f["mejor"] == m})
            fila[m] = celda
        # LA COMPROBACIÓN que impide que esto sea un segundo sitio con la cuenta.
        for m in por_motor:
            celda = fila[m]
            if celda.get("distancia_en_puntos") is None or celda["media"] is None:
                continue
            mejor = detalle[(nombre, m)]["mejor"]
            recalculada = (fila[mejor]["media"] - celda["media"]) * 100.0
            if abs(recalculada - celda["distancia_en_puntos"]) > TOLERANCIA_EN_PUNTOS:
                raise DatosQueNoCuadran(
                    f"{nombre}/{m}: la distancia recalculada desde las medias es {recalculada} "
                    f"y la pasada escribió {celda['distancia_en_puntos']}")
        resultados[nombre] = fila

    # -- la procedencia, entera -----------------------------------------------------------
    n_por_procedencia = collections.Counter(r.get("procedencia_id") for r in artefacto["resultados"])
    procedencias = []
    for pid, p in sorted(artefacto["procedencias"].items(), key=lambda kv: kv[1]["medido"]):
        sucios = sorted(nombre for nombre, repo in p["repositorios"].items() if repo["arbol_sucio"])
        procedencias.append({"id": pid, "medido": p["medido"], "n_intentos": n_por_procedencia[pid],
                             "repositorios_sucios": sucios,
                             "commits": {n: repo["commit"] for n, repo in sorted(p["repositorios"].items())}})
    n_sin = n_por_procedencia.get(None, 0)
    anclable_entera = n_sin == 0 and all(not p["repositorios_sucios"] for p in procedencias)

    return {
        "formato": "117-C1.v1",
        "fuente": {
            "artefacto": nombre_artefacto,
            "digest_resultados_crudos": artefacto["digest_resultados_crudos"],
            "creado": artefacto["creado"],
            "n_intentos": artefacto["n_intentos"],
            "horas_de_reloj": round(artefacto["total_wall_s"] / 3600.0, 2),
            "protocolo": {"version": protocolo["version_protocolo"],
                          "digest_sha256": protocolo["digest_sha256"],
                          "fecha_registro": protocolo["fecha_registro"]},
        },
        "procedencia": {"anclable_entera": anclable_entera, "intentos_sin_procedencia": n_sin,
                        "por_procedencia": procedencias},
        "regla": protocolo["regla_de_cierre"],
        "conjuntos": [{
            "nombre": d["nombre"], "data_id": d["data_id"],
            "enlace": f"https://www.openml.org/d/{d['data_id']}",
            "tarea": d["tarea"], "cubo": d["cubo_de_tamano"], "n_filas": d["n_filas"],
            "n_columnas": d["n_columnas"], "sellado": d["sellado"],
            "metrica": metrica_por_conjunto[d["nombre"]],
            # «Public» en OpenML es VISIBILIDAD, no una licencia (117-C0): se nombra así.
            "visibilidad_en_openml": d["licencia"],
        } for d in conjuntos_del_protocolo],
        "motores": motores,
        "resultados": resultados,
        "veredicto": {m: {k: v[k] for k in ("cumplidos", "datasets", "fraccion", "cumple_la_regla",
                                           "aciertos_por_ser_el_mejor",
                                           "datasets_que_le_faltan_para_cumplir",
                                           "datasets_que_puede_perder_sin_incumplir")}
                      for m, v in sorted(por_motor.items())},
        "lo_que_no_dicen": list(LO_QUE_NO_DICEN),
    }


def serializar(datos: dict[str, Any]) -> str:
    """UNA forma de escribirlo, para que la comparación byte a byte signifique algo."""
    return json.dumps(datos, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def generar() -> str:
    principal = componer(
        json.loads(RUTA_ARTEFACTO_PRINCIPAL.read_text(encoding="utf-8")),
        json.loads(RUTA_PROTOCOLO_PRINCIPAL.read_text(encoding="utf-8")),
        RUTA_ARTEFACTO_PRINCIPAL.name)
    ultima_medicion = componer(
        json.loads(RUTA_ARTEFACTO_ULTIMA_MEDICION.read_text(encoding="utf-8")),
        json.loads(RUTA_PROTOCOLO_ULTIMA_MEDICION.read_text(encoding="utf-8")),
        RUTA_ARTEFACTO_ULTIMA_MEDICION.name)
    # DOS bloques, la MISMA forma (117-C1.v1 cada uno): «principal» es lo que sostiene la
    # cartera (evidencia_digest de matrixai_engines.cartera) y se enseña como tal; «ultima_
    # medicion» es aparte, y su propio `procedencia.anclable_entera` dice que no lo es entera.
    return serializar({
        "formato": "117-C1.v2",
        "principal": principal,
        "ultima_medicion": ultima_medicion,
    })


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
        print("el JSON publicado NO es el que sale del artefacto: regenerarlo con --escribir "
              "(y mirar por qué cambió)", file=sys.stderr)
        return 1
    print("el JSON publicado es el que sale del artefacto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
