#!/usr/bin/env python3
"""CONTRATO 81-C6 — riesgo de reingreso a 30 días: la demostración.

    DEMOSTRACIÓN. Datos sintéticos. NO validado clínicamente y NO apto
    para decisiones asistenciales: es un sistema de apoyo, y no sustituye
    al profesional.

Sin dependencias externas. Sin claves. Sin red. El dataset se genera aquí
mismo con una semilla, y la regla que produce la etiqueta está publicada.

Lo que enseña, en este orden:

 1. El dataset sintético y su regla;
 2. El pipeline, caso a caso;
 3. Calibración (§16.4), y lo que NO dice;
 4. A QUIÉN deja sin respuesta la abstención;
 5. El recibo, y lo que NO lleva;
 6. Tocar el submodelo invalida la verificación;
 7. El coste (§16.4).

*(Esta lista decía SEIS pasos cuando el programa ya imprimía SIETE, y en
otro orden: al insertar el de la cobertura por clase se quedó atrás. Es
exactamente el defecto que la 3ª pasada de auditoría encontró en el caso
de routing —una «salida esperada» que no es la que sale— cometido aquí
mismo por quien lo estaba arreglando allí. Se corrige LEYENDO la salida
del programa, no escribiéndola a mano otra vez.)*
"""
from __future__ import annotations

import json
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent))

from matrixai.reference.readmission import (  # noqa: E402
    ADVERTENCIA, analisis_de_confianza, cobertura_por_clase,
)
from matrixai.reference.readmission_datos import (  # noqa: E402
    REGLA_DE_LA_ETIQUETA, columnas, generar_dataset, vector_de,
)
from matrixai.reference.readmission_pipeline import (  # noqa: E402
    clasificador_del_registry, evaluar_caso,
)
from matrixai.registry.model_registry import ModelRegistry  # noqa: E402

MOMENTO = "2026-08-20T10:00:00Z"

#: Los casos de la demostración. Los tres primeros son de la distribución
#: del dataset; los otros existen para enseñar que el caso **se abstiene**
#: en vez de inventar.
CASOS = [
    ("alto riesgo",
     {"edad": 88, "ingresos_previos_12m": 6, "dias_de_estancia": 30,
      "servicio": "cardiologia", "alta_voluntaria": 1},
     "disnea de reposo y edema en miembros inferiores"),
    ("bajo riesgo",
     {"edad": 34, "ingresos_previos_12m": 0, "dias_de_estancia": 2,
      "servicio": "cirugia", "alta_voluntaria": 0},
     "dolor controlado, sin fiebre"),
    ("sin nota clínica",
     {"edad": 71, "ingresos_previos_12m": 2, "dias_de_estancia": 9,
      "servicio": "neumologia", "alta_voluntaria": 0},
     ""),
    ("falta una variable obligatoria",
     {"edad": 79, "servicio": "medicina_interna", "alta_voluntaria": 0},
     "edema y ajuste de diuretico"),
    ("servicio que no existe",
     {"edad": 79, "ingresos_previos_12m": 3, "dias_de_estancia": 11,
      "servicio": "urgencias", "alta_voluntaria": 0},
     "disnea"),
    ("dato posterior a la decisión",
     {"edad": 79, "ingresos_previos_12m": 3, "dias_de_estancia": 11,
      "servicio": "cardiologia", "alta_voluntaria": 0,
      "analitica_at": "2026-08-21T09:00:00Z"},
     "disnea y edema"),
]


def _separador(titulo: str) -> None:
    print()
    print(f"── {titulo} " + "─" * max(0, 72 - len(titulo)))


def main() -> int:
    print("MatrixAI — 81-C6: riesgo de reingreso a 30 días")
    print("=" * 78)
    # LA ADVERTENCIA, ARRIBA. No en un anexo.
    print(ADVERTENCIA["es"])
    print("=" * 78)

    registry = ModelRegistry(AQUI / "registry")
    try:
        clasificar, digest, version = clasificador_del_registry(
            registry, "readmission_head", "v1")
    except Exception as exc:  # noqa: BLE001
        print(f"\nNo se pudo cargar el clasificador: {exc}")
        # El comando EXACTO, no «ver el README»: un mensaje que manda a
        # otro sitio a buscar la respuesta hace trabajar de más a quien ya
        # está atascado. La receta entera está en el README, en «Cómo
        # reconstruirlo desde cero».
        print("\nEntrena el caso primero:")
        print("  python3 -m matrixai train examples/readmission/readmission.mxai \\")
        print("    --training examples/readmission/readmission.mxtrain \\")
        print("    --output examples/readmission/run")
        print("\ny después publícalo en el registry del ejemplo — los cuatro")
        print("pasos están en examples/readmission/README.md.")
        return 3

    _separador("Paso 1 — El dataset sintético y su regla")
    csv_texto = generar_dataset(400, 42)
    print(f"  Filas: {len(csv_texto.strip().splitlines()) - 1}")
    print(f"  Columnas: {len(columnas())}  (9 tabulares + 11 de la nota)")
    print(f"  Regla de la etiqueta: {REGLA_DE_LA_ETIQUETA}")
    print("  Determinista: el mismo CSV byte a byte con la misma semilla")

    _separador("Paso 2 — El pipeline, caso a caso")
    latencias = []
    for nombre, registro, nota in CASOS:
        empieza = time.perf_counter()
        r = evaluar_caso(registro, nota, momento_de_decision=MOMENTO,
                         clasificador=clasificar, clasificador_digest=digest,
                         clasificador_version=version)
        latencias.append(time.perf_counter() - empieza)
        if r["outcome"] == "prediction":
            detalle = f"{r['class']}  (confianza {r['confidence']:.3f})"
        elif r["outcome"] == "abstained":
            detalle = f"SE ABSTIENE  ·  {r['policy']['rule_id']}"
        else:
            detalle = f"no ejecutó  ·  {r.get('reason', '')[:50]}"
        print(f"  {nombre:32} → {detalle}")

    _separador("Paso 3 — Calibración (§16.4), y lo que NO dice")
    import csv as _csv
    import io as _io
    filas = list(_csv.DictReader(_io.StringIO(csv_texto)))
    cols = columnas()
    predicciones = []
    for f in filas:
        p = clasificar([float(f[c]) for c in cols])
        clase = max(p, key=p.get)
        predicciones.append((clase, p[clase], f["etiqueta"]))
    calibracion = analisis_de_confianza(predicciones)
    print(f"  ECE: {calibracion['ece']:.4f} sobre {calibracion['n']} casos")
    for b in calibracion["bins"]:
        print(f"    {b['from']:.1f}-{b['to']:.1f}  n={b['n']:4d}  "
              f"acierto={b['accuracy']:.3f}  brecha={b['gap']:+.3f}")
    print(f"  Alcance: {calibracion['scope']}")
    print(f"  {calibracion['caveat']}")

    _separador("Paso 4 — A QUIÉN deja sin respuesta la abstención")
    cobertura = cobertura_por_clase(predicciones)
    print(f"  Se abstiene en el {cobertura['abstention_rate']:.1%} del dataset")
    print(f"  Cuando responde, acierta el {cobertura['accuracy_when_answering']:.1%}")
    print("  …pero ese número se consigue callándose, y no por igual:")
    for clase, d in sorted(cobertura["by_class"].items()):
        print(f"    {clase:14} n={d['n']:4d}  se abstiene {d['abstention_rate']:6.1%}  "
              f"cubre {d['coverage']:6.1%}")
    print(f"  Peor cubierta: {cobertura['worst_covered_class']} "
          f"(diferencia de {cobertura['spread']:.1%} entre clases)")
    print("  LIMITACIÓN CONOCIDA: el sistema casi no se pronuncia sobre los")
    print("  pacientes que SÍ reingresan, que es para lo que existiría.")

    _separador("Paso 5 — El recibo, y lo que NO lleva")
    caso = CASOS[0]
    r = evaluar_caso(caso[1], "NHC-4491203 " + caso[2], momento_de_decision=MOMENTO,
                     clasificador=clasificar, clasificador_digest=digest,
                         clasificador_version=version)
    recibo = r["receipt"]
    print(f"  Pasos: {[p['id'] for p in recibo['steps']]}")
    for m in recibo["models"]:
        print(f"    {m['model_id']:14} {m['version']:32} {m['digest'][:22]}…")
    print(f"  Momento de corte: {recibo['input']['data_cutoff']}")
    print(f"  Huella del texto: {recibo['input']['text_digest'][:26]}…")
    serializado = json.dumps(recibo, ensure_ascii=False, default=str)
    print(f"  ¿Aparece el identificador del paciente en el recibo?  "
          f"{'SÍ — FALLO' if 'NHC-4491203' in serializado else 'NO'}")

    _separador("Paso 6 — Tocar el submodelo invalida la verificación")
    taller = Path(tempfile.mkdtemp())
    try:
        shutil.copytree(AQUI / "registry", taller / "registry")
        params = taller / "registry/entries/readmission_head/v1/params.json"
        # Mismos valores, otros bytes: el hash cubre el fichero.
        params.write_text(json.dumps(json.loads(params.read_text())))
        try:
            clasificador_del_registry(ModelRegistry(taller / "registry"),
                                      "readmission_head", "v1")
            print("  ERROR: la manipulación NO se detectó")
        except Exception as exc:  # noqa: BLE001
            print(f"  Detectada — {str(exc)[:100]}…")
    finally:
        shutil.rmtree(taller, ignore_errors=True)

    _separador("Paso 7 — El coste (§16.4)")
    latencias.sort()
    pico_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  Latencia por caso: mediana {latencias[len(latencias)//2]*1000:.2f} ms · "
          f"máximo {latencias[-1]*1000:.2f} ms")
    print(f"  Memoria pico del proceso: {pico_kib/1024:.1f} MiB")
    print("  Ejecución local, sin red y sin GPU.")

    print()
    print("=" * 78)
    print(ADVERTENCIA["es"])
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
