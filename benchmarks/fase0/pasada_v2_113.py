# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""113-C4 — la pasada de Fase 0 con el protocolo v2 (re-firma + densa con Adam).

Es la MISMA pasada que 101-C5 (`pasada_amplia_101_c5.py`), con tres diferencias
y ninguna más:

1. Lee `protocolo_exploratorio_v2.json` (113-C0), no la v1: 40 conjuntos tras la
   re-firma (P1 y P2 decididos por Roberto el 2026-09-21).
2. Escribe en `pasada_v2_113_resultado.json`, no en el artefacto de 101-C5, que
   queda como evidencia de lo que se midió entonces (invariante 7 del 113).
3. ANTES de medir, exige que la receta que EJECUTA el motor denso sea la que el
   protocolo v2 registró (invariante 1: «la receta se fija ANTES de medir»). Si
   no coincide, no se mide nada: medir otra receta llamándola la registrada es
   exactamente la deriva de protocolo que este proyecto ya pagó una vez.

La guarda de reutilización de 101-C5 NO pasa (medido el 2026-09-21: desde su
ancla cambiaron los cuatro árboles y el camino de medición), así que esta
pasada es COMPLETA; por eso escribe en un artefacto nuevo y no reaprovecha el
viejo.

    cd /home/deployer/matrixAI && setsid nohup python3 \\
      benchmarks/fase0/pasada_v2_113.py >> /tmp/c113.log 2>&1 < /dev/null & disown
"""
from __future__ import annotations

import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

import pasada_amplia_101_c5 as c5  # noqa: E402
import pasada_exploratoria_101_c3 as c3  # noqa: E402

RUTA_DEL_PROTOCOLO_V2 = _AQUI / "protocolo_exploratorio_v2.json"
RUTA_DE_SALIDA_V2 = _AQUI / "pasada_v2_113_resultado.json"


def exigir_la_receta_registrada() -> dict:
    """La receta que la densa EJECUTARÍA tiene que ser la del protocolo v2.

    No compara constantes: genera un `training_text` real con el núcleo, le
    aplica la receta del motor (`densa.aplicar_receta`, lo mismo que hace
    `_ajustar`) y lo lee con el analizador del núcleo. Así compara lo que
    entrenaría, que es lo que el protocolo registró."""
    import csv
    import io
    import json
    import random

    from matrixai.playground_api import generate_project_from_dataset
    from matrixai.training.parser import parse_training_text
    from matrixai_engines.motores import densa

    protocolo = json.loads(RUTA_DEL_PROTOCOLO_V2.read_text(encoding="utf-8"))
    entradas = [m for m in protocolo["motores"] if m["id"] == "matrixai.dense.torch_cpu"]
    if len(entradas) != 1 or "receta" not in entradas[0]:
        raise SystemExit("el protocolo v2 no registra la receta de la densa: no se mide")
    registrada = entradas[0]["receta"]

    rng = random.Random(0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["x", "y"])
    for _ in range(60):
        x = rng.uniform(0, 10)
        w.writerow([round(x, 3), "si" if x > 5 else "no"])
    texto = generate_project_from_dataset(buf.getvalue(), "y", locale="es",
                                          column_type_overrides={"y": "categorical"})["training_text"]
    leido = parse_training_text(densa.aplicar_receta(texto))
    ejecutada = {
        "optimizador": leido.optimizer.type,
        "learning_rate": leido.optimizer.learning_rate,
        "early_stop": {"patience": leido.run.early_stop_patience,
                       "metric": leido.run.early_stop_metric},
        # Las épocas las fija el motor con `epochs_override`, no el texto.
        "epochs": densa.EPOCAS_POR_DEFECTO,
        "batch_size": leido.dataset.batch.size,
    }
    if ejecutada != registrada:
        raise SystemExit(f"la receta que se ejecutaría NO es la registrada en el protocolo v2: "
                         f"registrada={registrada} ejecutada={ejecutada}. No se mide.")
    return {"registrada": registrada, "ejecutada": ejecutada}


def main(argv=None) -> None:
    c3.RUTA_DEL_PROTOCOLO = RUTA_DEL_PROTOCOLO_V2
    comprobacion = exigir_la_receta_registrada()
    print(f"receta registrada: {comprobacion['registrada']}", flush=True)
    print(f"receta ejecutada:  {comprobacion['ejecutada']}", flush=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--salida" not in argv:
        argv += ["--salida", str(RUTA_DE_SALIDA_V2)]
    c5.main(argv)


if __name__ == "__main__":
    main()
