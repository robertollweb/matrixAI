# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-WEB — LOS DATOS PÚBLICOS DE LA RED NUEVA (TabM + embeddings PLR), GENERADOS DESDE LOS
REGISTROS SELLADOS del contrato 119.

Mismo patrón que `generar.py` (117-C1) y `generar_texto_107.py`: **ninguna cifra se escribe a
mano**. Este guion lee tres JSON de `benchmarks/fase0/` y compone `red_119_publico.json`, que la
sección nueva de «Cómo medimos» enseña tal cual. Una prueba
(`tests/test_119_web_datos_publicos.py`) lo regenera y lo compara BYTE A BYTE con el publicado.

    python3 benchmarks/datos_publicos/generar_red_119.py            # comprueba el publicado
    python3 benchmarks/datos_publicos/generar_red_119.py --escribir # lo regenera

Las tres fuentes:

- `resultado_pasada_119_c3.json`: los 32 conjuntos NO sellados (31 cumplen).
- `resultado_pasada_119_c4.json`: los 8 SELLADOS, medidos UNA vez; lleva `veredicto_x_de_40`
  (el X/40 que se publica, con el reparto 31+7 y la densa de antes en los mismos 40: 16/40).
- `recuento_119_con_el_motor_nuevo_en_el_campo.json`: los cumplidos de CADA motor con el motor
  nuevo en el campo (mismas funciones que C3/C4; solo cuenta, no mide).

**Qué se verifica, y PARA si no cuadra** (no se publica nada a medias):
los DOS sellos de cada resultado (`digest_resultados_crudos` y `digest_solo_de_resultados`, con
`matrixai.estudio.validacion.digest_canonico`, como `pasada_119_c4.verificar_los_sellos_de_c3`);
que los sellos que el recuento cita como fuentes son los de C3 y C4; que su control (38) es el
X/40 de C4; que 31 + 7 = X; que el detalle por conjunto suma lo que el reparto dice; y que el
fichero de la v2 que C4 usó (por sha256) es el que está en el disco.

**Qué se COPIA y qué se CALCULA.** Se COPIA, sin recalcular: el X/40, el reparto, la densa de
antes, el detalle por conjunto (cumple, mejor motor, distancia), la tabla recontada, lo declarado
(Allstate, la parada temprana, las otras redes, los sellados) y la procedencia. La única
operación propia es una SUMA de comprobación (31 + 7) que se compara con el X/40 que C4 ya
escribió: nunca se publica un número calculado aquí.

Salida determinista (claves ordenadas, sin ninguna fecha de «ahora»): las fechas son las que cada
fuente ya traía.
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
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from matrixai.estudio.validacion import digest_canonico  # noqa: E402 -- el mismo canonicalizador

FASE0 = RAIZ / "benchmarks" / "fase0"
RUTA_C3 = FASE0 / "resultado_pasada_119_c3.json"
RUTA_C4 = FASE0 / "resultado_pasada_119_c4.json"
RUTA_RECUENTO = FASE0 / "recuento_119_con_el_motor_nuevo_en_el_campo.json"
RUTA_V2 = FASE0 / "pasada_v2_113_resultado.json"
RUTA_PUBLICADO = AQUI / "red_119_publico.json"
#: La cuenta de ANTES (la v2 sin la red nueva en el campo): se COPIA de lo que la página ya
#: publica de esa misma pasada, tras comprobar que es la misma (por su sello).
RUTA_FASE0_PUBLICO = AQUI / "fase0_publico.json"

MOTOR_NUEVO = "matrixai.dense.tabm_cpu"


class DatosQueNoCuadran(RuntimeError):
    """Lo leído no cuadra con su sello o con otra fuente: no se publica nada."""


def _cargar(ruta: Path) -> dict[str, Any]:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _verificar_sellos(nombre: str, payload: dict[str, Any]) -> None:
    """Recalcula los DOS sellos (misma forma que `pasada_119_c4.verificar_los_sellos_de_c3`)."""
    sin_el_campo = {k: v for k, v in payload.items() if k != "digest_resultados_crudos"}
    problemas = []
    if digest_canonico(sin_el_campo) != payload.get("digest_resultados_crudos"):
        problemas.append("digest_resultados_crudos NO cuadra")
    if digest_canonico(payload.get("resultados")) != payload.get("digest_solo_de_resultados"):
        problemas.append("digest_solo_de_resultados NO cuadra")
    if problemas:
        raise DatosQueNoCuadran(f"{nombre}: {'; '.join(problemas)} -- el fichero se ha tocado "
                                "después de sellarlo")


def _exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise DatosQueNoCuadran(mensaje)


def _detalle_por_conjunto(bloque: str, nuevo: list[dict], antes: list[dict],
                          metricas: dict[str, str]) -> list[dict[str, Any]]:
    """Une, por nombre, el detalle del motor nuevo y el de la red de antes (ambos COPIADOS)."""
    por_nombre_antes = {d["dataset"]: d for d in antes}
    _exigir(len(por_nombre_antes) == len(antes) and len(nuevo) == len(antes)
            and {d["dataset"] for d in nuevo} == set(por_nombre_antes),
            f"{bloque}: el detalle del motor nuevo y el de la red de antes no son los mismos conjuntos")
    filas = []
    for d in sorted(nuevo, key=lambda x: x["dataset"]):
        a = por_nombre_antes[d["dataset"]]
        filas.append({
            "conjunto": d["dataset"],
            "bloque": bloque,
            "metrica": metricas.get(d["dataset"]),
            "cumple": d["cumple"],
            "perdido_por_fallo": d["perdido_por_fallo"],
            "mejor": d["mejor"],
            "distancia_en_puntos": d["distancia_en_puntos"],
            "red_de_antes": {
                "cumple": a["cumple"],
                "perdido_por_fallo": a["perdido_por_fallo"],
                "mejor": a["mejor"],
                "distancia_en_puntos": a["distancia_en_puntos"],
            },
        })
    return filas


def componer() -> dict[str, Any]:
    c3, c4, recuento = _cargar(RUTA_C3), _cargar(RUTA_C4), _cargar(RUTA_RECUENTO)
    _verificar_sellos("resultado_pasada_119_c3.json", c3)
    _verificar_sellos("resultado_pasada_119_c4.json", c4)

    # Lo que el recuento cita como fuentes tiene que ser lo que hay en disco.
    _exigir(recuento["fuentes"]["c3"] == c3["digest_resultados_crudos"],
            "el recuento cita para C3 un sello que no es el de resultado_pasada_119_c3.json")
    _exigir(recuento["fuentes"]["c4"] == c4["digest_resultados_crudos"],
            "el recuento cita para C4 un sello que no es el de resultado_pasada_119_c4.json")

    v = c4["veredicto_x_de_40"]
    _exigir(v is not None, "C4 no trae `veredicto_x_de_40`: no es una pasada real completa")
    _exigir(c3["tipo_de_ejecucion"] == "pasada" and not c3["parcial"]
            and c4["tipo_de_ejecucion"] == "pasada" and not c4["parcial"],
            "C3 o C4 no son una pasada real completa")
    _exigir(c3["protocolo_119_v4_digest_sha256"] == c4["protocolo_119_v4_digest_sha256"],
            "C3 y C4 no se midieron con el mismo protocolo")
    _exigir(v["sellos_de_c3_verificados"]["cuadra"], "C4 dice que los sellos de C3 no cuadran")

    c3_ok = v["reparto"]["c3_no_sellados"]
    c4_ok = v["reparto"]["c4_sellados"]
    antes = v["densa_v2_en_los_mismos_conjuntos"]
    _exigir(c3_ok["cumplidos"] + c4_ok["cumplidos"] == v["x_de_40"],
            "31 + 7 no suman el X/40 que C4 escribió")
    _exigir(c3_ok["datasets"] + c4_ok["datasets"] == v["n_conjuntos"],
            "32 + 8 no suman los conjuntos que C4 escribió")
    _exigir(c3_ok["cumplidos"] == sum(1 for d in c3_ok["detalle"] if d["cumple"])
            and c4_ok["cumplidos"] == sum(1 for d in c4_ok["detalle"] if d["cumple"]),
            "el detalle por conjunto no suma los cumplidos del reparto")
    a3, a4 = antes["reparto"]["c3_no_sellados"], antes["reparto"]["c4_sellados"]
    _exigir(a3["cumplidos"] + a4["cumplidos"] == antes["cumplidos"],
            "el reparto de la red de antes no suma su total")

    tabla = recuento["tabla"]
    _exigir(tabla[MOTOR_NUEVO]["cumplidos"] == v["x_de_40"] and recuento["control"].find(str(v["x_de_40"])) >= 0,
            "el control del recuento (el motor nuevo) no da el X/40 de C4")
    _exigir(all(t["de"] == v["n_conjuntos"] for t in tabla.values()),
            "el recuento no es sobre los mismos conjuntos")

    # El fichero de la v2 que C4 usó (sha256 registrado en su procedencia) es el que hay.
    de_entrada = c4["procedencia"]["datos_de_entrada"]["pasada_v2_113_resultado"]
    sha_v2 = hashlib.sha256(RUTA_V2.read_bytes()).hexdigest()
    _exigir(de_entrada["sha256"] == sha_v2, "la v2 en disco no es la que C4 registró por sha256")
    v2 = _cargar(RUTA_V2)
    antes_v2 = _cargar(RUTA_FASE0_PUBLICO)["ultima_medicion"]
    _exigir(antes_v2["fuente"]["digest_resultados_crudos"] == v2["digest_resultados_crudos"],
            "fase0_publico.json no publica la v2 que C4 usó: la cuenta «sin la red nueva» sería de otra pasada")
    sin_la_red_nueva = {m: {"cumplidos": x["cumplidos"], "de": x["datasets"]}
                        for m, x in sorted(antes_v2["veredicto"].items())}
    _exigir(sin_la_red_nueva["matrixai.dense.torch_cpu"]["cumplidos"] == antes["cumplidos"],
            "la v2 publicada da a la red de antes otro número que el que C4 escribió")

    metricas = {**c3["metrica_de_cierre_por_dataset"], **c4["metrica_de_cierre_por_dataset"]}
    detalle = (_detalle_por_conjunto("no_sellado", c3_ok["detalle"], a3["detalle"], metricas)
               + _detalle_por_conjunto("sellado", c4_ok["detalle"], a4["detalle"], metricas))
    detalle.sort(key=lambda f: f["conjunto"].lower())

    allstate = c3["allstate_medido"]
    _exigir(allstate["medido"] and allstate["todos_por_debajo_del_minimo"],
            "Allstate ya no entrena menos épocas que el mínimo de la regla: lo declarado cambió")
    parada = c4["parada_temprana_declarada"]

    return {
        "formato": "119-WEB.v1",
        "motor": c3["motor"],
        "titular": {
            "x_de_40": v["x_de_40"],
            "n_conjuntos": v["n_conjuntos"],
            "fraccion": v["fraccion"],
            "red_de_antes_x_de_40": antes["cumplidos"],
            "red_de_antes_fraccion": antes["fraccion"],
        },
        "reparto": {
            "no_sellados": {k: c3_ok[k] for k in ("cumplidos", "datasets", "fraccion")},
            "sellados": {k: c4_ok[k] for k in ("cumplidos", "datasets", "fraccion")},
            "red_de_antes_no_sellados": {k: a3[k] for k in ("cumplidos", "datasets", "fraccion")},
            "red_de_antes_sellados": {k: a4[k] for k in ("cumplidos", "datasets", "fraccion")},
        },
        "decision_d2": {
            "decision": v["decision_segun_d2"]["decision"],
            "umbral_de_la_cartera": v["decision_segun_d2"]["umbral_de_la_cartera"]["valor"],
            "fuente": v["decision_segun_d2"]["fuente"],
        },
        "detalle_por_conjunto": detalle,
        "recuento_con_el_motor_nuevo_en_el_campo": {
            "que_es": recuento["que_es"],
            "tabla": {m: dict(sorted(t.items())) for m, t in sorted(tabla.items())},
            "sin_la_red_nueva": {
                "que_es": "la v2 congelada (113, 22-09): los motores que había entonces, SIN el motor "
                          "nuevo en el campo; en el sitio de la red de antes, la red de antes",
                "fuente": "pasada_v2_113_resultado.json",
                "digest_resultados_crudos": v2["digest_resultados_crudos"],
                "por_motor": sin_la_red_nueva,
            },
        },
        "declarado": {
            "allstate": {
                "conjunto": "Allstate_Claims_Severity",
                "epocas_ejecutadas_min": allstate["epocas_min"],
                "epocas_ejecutadas_max": allstate["epocas_max"],
                "minimo_de_la_regla": allstate["minimo_de_la_regla"],
                "n_pliegues": len(allstate["detalle"]),
                "todos_parados_por_el_plazo": all(d["parado_por_plazo"] for d in allstate["detalle"]),
                "original_es": c3["allstate_declarado"],
            },
            "parada_temprana": {
                "metrica": parada["metrica_que_declaran_los_intentos"],
                "por_tarea": parada["por_tarea"],
                "clasificacion_no_es_como_la_fuente": parada["en_clasificacion_no_es_como_la_fuente"],
                "conjuntos_de_clasificacion_en_los_sellados":
                    parada["conjuntos_de_clasificacion_en_esta_ejecucion"],
                "conjuntos_de_clasificacion_en_los_no_sellados":
                    c3["parada_temprana_declarada"]["conjuntos_de_clasificacion_en_esta_ejecucion"],
            },
            "otras_redes": {
                "fuente": recuento["fuentes"]["v2"],
                "sha256": de_entrada["sha256"],
                "creado": v2["creado"],
                "se_remidieron": False,
            },
            "sellados": {
                "medidos_una_sola_vez": True,
                "n_intentos": c4["n_intentos"],
                "n_intentos_reintentados": c4["n_intentos_reintentados"],
                "que_es_x_de_40": c4["que_es_veredicto_x_de_40"],
            },
        },
        "procedencia": {
            "c3": {
                "artefacto": RUTA_C3.name, "corte": c3["corte"], "creado": c3["creado"],
                "medido": c3["procedencia"]["medido"],
                "digest_resultados_crudos": c3["digest_resultados_crudos"],
                "digest_solo_de_resultados": c3["digest_solo_de_resultados"],
                "n_intentos": c3["n_intentos"], "total_wall_s": c3["total_wall_s"],
                "commits": {r: x["commit"] for r, x in c3["procedencia"]["repositorios"].items()},
                "arbol_sucio": any(x["arbol_sucio"] for x in c3["procedencia"]["repositorios"].values()),
            },
            "c4": {
                "artefacto": RUTA_C4.name, "corte": c4["corte"], "creado": c4["creado"],
                "medido": c4["procedencia"]["medido"],
                "digest_resultados_crudos": c4["digest_resultados_crudos"],
                "digest_solo_de_resultados": c4["digest_solo_de_resultados"],
                "n_intentos": c4["n_intentos"], "total_wall_s": c4["total_wall_s"],
                "commits": {r: x["commit"] for r, x in c4["procedencia"]["repositorios"].items()},
                "arbol_sucio": any(x["arbol_sucio"] for x in c4["procedencia"]["repositorios"].values()),
            },
            "protocolo_digest_sha256": c4["protocolo_119_v4_digest_sha256"],
            "sellos_verificados": True,
        },
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
        print("el JSON publicado NO es el que sale de los registros: regenerarlo con "
              "--escribir (y mirar por qué cambió)", file=sys.stderr)
        return 1
    print("el JSON publicado es el que sale de los registros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
