#!/usr/bin/env python3
"""131 · C1 — el CONTROL del instrumento con semilla. No mide nada: lee lo que dejó la cola y comprueba.

    python3 control_131_c1.py control_131_c1_s0.json control_131_c1_s1.json [--c5 c5_129_diez.json]

(i)  SEMILLA DE SIEMPRE (sin `--semilla`): campeón, `media_de_la_seleccion` entera y test BIT A BIT lo de `c5_129_diez.json`,
     y el mismo `split_plan_digest` que el C5 (mismo reparto) en `<salida>_estados/<conjunto>.json` contra `c5_129_diez_estados/`.
(ii) SEMILLA 1: cada estudio TERMINA (completed); el test tiene el MISMO tamaño (`n_observations`) y la misma proporción de clases
     que el de la semilla 0 —la proporción se comprueba sobre el reparto del núcleo (`proponer_particion`, el que usa el estudio)
     recalculado fuera: iguales conteos por clase, otras filas— y el `split_plan_digest` es OTRO; los pliegues de la búsqueda
     también cambian (la `media_de_la_seleccion` difiere de la de la semilla 0). La especificidad medida × los negativos del
     test tiene que ser un entero (coherencia con ese conteo).
Qué NO comprueba (dicho): la proporción de clases del test del ESTUDIO no sale del estado; se comprueba la función del reparto y
la coherencia de la especificidad con ese conteo, no las filas.
Salida: 0 si todo cuadra, 1 si no.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent


def _por_ruta(nombre, fichero):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / fichero)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LECTOR = _por_ruta("lector_131_c3_c1", "lector_131_c3.py")
R0 = LECTOR.R0
V3 = LECTOR.V3


def reparto_del_test(csv_text, objetivo, semilla, test_fraction=0.2):
    """El test (ids) y el conteo por clase del reparto del núcleo (`proponer_particion`, iid estratificado por el objetivo),
    como lo pide el estudio (`seed`, `test_fraction=0.2`). Los ids son la POSICIÓN de la fila, como los sintéticos del estudio."""
    from matrixai.training.particion_por_diseno import proponer_particion
    filas = [dict(f, _id=str(i)) for i, f in enumerate(csv.DictReader(io.StringIO(csv_text)))]
    prop = proponer_particion(filas, plan_id="control_131_c1", observation_id_field="_id", split_type="iid", seed=semilla,
                              test_fraction=test_fraction, folds=5, repeats=1, objetivo=objetivo)
    prueba = {o for o, rol in prop.plan.assignments.items() if rol == "test"}
    clase_de = {f["_id"]: f[objetivo] for f in filas}
    conteo = {}
    for o in prueba:
        conteo[clase_de[o]] = conteo.get(clase_de[o], 0) + 1
    return prueba, dict(sorted(conteo.items()))


def _digest(estado):
    return ((estado.get("seleccion") or {}).get("evaluacion_final") or {}).get("split_plan_digest")


def _n_obs(estado):
    ms = (((estado.get("seleccion") or {}).get("evaluacion_final") or {}).get("metrics")) or []
    return {m.get("n_observations") for m in ms if isinstance(m, dict)}


def comprobar(s0, s1, c5, estados0, estados1, estados_c5, csv_de):
    """Devuelve (ok, líneas). `csv_de(nombre) -> (texto, objetivo, clase_positiva)`; los `estados*` son {nombre: estado}."""
    ok, lin = True, []

    def marca(cond, texto):
        nonlocal ok
        ok &= bool(cond)
        lin.append(("OK     " if cond else "FALLA  ") + texto)

    for n in sorted(s0):
        a, b, ref = s0[n], s1.get(n), c5.get(n)
        marca(a.get("semilla") is None, f"{n}: la medida (i) no lleva semilla (la de siempre): {a.get('semilla')!r}")
        marca(b is not None and b.get("semilla") == 1, f"{n}: la medida (ii) lleva la semilla 1 en el registro: {b and b.get('semilla')!r}")
        if ref is None:
            marca(False, f"{n}: no está en el C5")
            continue
        iguales, dif = LECTOR.bit_a_bit(a, ref)
        marca(a.get("estado") == "completed" and iguales,
              f"{n}: (i) semilla de siempre, BIT A BIT el C5 (campeón, media de la selección, test)" + ("" if iguales else f" — difiere: {dif}"))
        d_c5, d0 = _digest(estados_c5.get(n, {})), _digest(estados0.get(n, {}))
        marca(d0 is not None and d0 == d_c5, f"{n}: (i) mismo reparto que el C5 (split_plan_digest {str(d0)[:12]} = {str(d_c5)[:12]})")
        if b is None:
            marca(False, f"{n}: falta la medida con la semilla 1")
            continue
        marca(b.get("estado") == "completed", f"{n}: (ii) con la semilla 1 el estudio TERMINA ({b.get('estado')})")
        if b.get("estado") != "completed":
            continue
        d1 = _digest(estados1.get(n, {}))
        marca(d1 is not None and d1 != d0, f"{n}: (ii) otro reparto (split_plan_digest {str(d1)[:12]} ≠ {str(d0)[:12]})")
        marca(b.get("media_de_la_seleccion") != a.get("media_de_la_seleccion"),
              f"{n}: (ii) otros pliegues de la búsqueda (la media de la selección difiere de la de la semilla 0)")
        n0, n1 = _n_obs(estados0.get(n, {})), _n_obs(estados1.get(n, {}))
        marca(len(n0) == 1 and n0 == n1, f"{n}: (ii) test del MISMO tamaño en el estudio (n_observations {sorted(n0)} = {sorted(n1)})")
        texto, objetivo, positiva = csv_de(n)
        p0, c0 = reparto_del_test(texto, objetivo, 0)
        p1, c1 = reparto_del_test(texto, objetivo, 1)
        marca(c0 == c1 and len(p0) == len(p1) and p0 != p1,
              f"{n}: (ii) reparto del núcleo: test de {len(p0)} = {len(p1)} filas, mismos conteos por clase {c0} = {c1}, otras filas")
        marca(n1 == {len(p1)}, f"{n}: (ii) el n_observations del estudio ({sorted(n1)}) es el tamaño del test del reparto ({len(p1)})")
        negativos = sum(v for k, v in c1.items() if str(k) != str(positiva))
        esp = (b.get("test") or {}).get("metricas", {}).get("specificity")
        if esp is not None and negativos:
            tn = esp * negativos
            marca(abs(tn - round(tn)) < 1e-6, f"{n}: (ii) especificidad × negativos del test = {tn:.4f} (entero: coherente con el conteo)")
    return ok, lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s0"); ap.add_argument("s1")
    ap.add_argument("--c5", default=str(AQUI / "c5_129_diez.json"))
    a = ap.parse_args()
    cargar = lambda p: json.loads(Path(p).read_text())
    d0, d1, c5 = cargar(a.s0), cargar(a.s1), cargar(a.c5)
    estados = lambda p, nombres: {n: cargar(Path(p).with_name(Path(p).stem + "_estados") / f"{n}.json") for n in nombres}
    gen = R0.cargar_generador()
    banco = {b["nombre"]: b for b in json.loads(R0.BORRADOR.read_text())["conjuntos"]["banco"]}

    def csv_de(n):
        c = banco[n]
        texto = R0.csv_de(gen, c["data_id"])[0].decode("utf-8")
        return texto, c["objetivo"], c.get("clase_positiva")

    ok, lin = comprobar(d0["conjuntos"], d1["conjuntos"], c5["conjuntos"], estados(a.s0, d0["conjuntos"]),
                        estados(a.s1, d1["conjuntos"]), estados(a.c5, d0["conjuntos"]), csv_de)
    print("\n".join(lin)); print("CONTROL C1:", "CUADRA" if ok else "NO CUADRA")
    sys.exit(0 if ok else 1)



if __name__ == "__main__":
    main()
