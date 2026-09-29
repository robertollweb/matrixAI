# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — cuántas filas de la tarea A llevan el marcador `NUMBER`, por versión del
corpus y partición (y en `train` también por `target`).

**Por qué existe.** En la versión 1 del corpus FakeNewsCorpusSpanish (`train.xlsx` y
`development.xlsx`, columna `version_origen="v1"`) los números vienen enmascarados con la
palabra `NUMBER`; en la v2 (`test.xlsx`, `version_origen="v2"`) no. Las 5 preguntas de la
tarea A (`medir_c30_respondedor.py`, `medir_c30_jev.py`) salen del chi² sobre ese CSV, y
dos de ellas («number de», «mil number»...) citan el propio marcador -- este guion es lo
que mide, en vez de suponer, cuántas filas lo llevan y dónde, para que
`benchmarks/datos_publicos/generar_texto_107.py` pueda enseñarlo sin escribir la cifra a
mano.

**Qué NO hace.** No decide nada del muestreo ni de las preguntas: eso ya está hecho y
sellado en `resultado_c30_respondedor_a.json` / `resultado_c30_jev_A.json`. Esto es un
recuento aparte, sobre el mismo CSV, con su propia procedencia (sha256 del CSV y del propio
guion) para que se pueda comprobar sin volver a correr nada.

    python3 benchmarks/texto_107c30/medir_marcador_number.py   # escribe marcador_number_A.json

Lee `tareas/tarea_a.csv`, el enlace SIN VERSIONAR a los datos locales (no se commitea, ver
la cabecera del propio directorio): por eso el generador público (`generar_texto_107.py`) y
su prueba NUNCA leen el CSV, solo `marcador_number_A.json`, que sí es versionado.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RUTA_CSV = AQUI / "tareas" / "tarea_a.csv"
RUTA_SALIDA = AQUI / "marcador_number_A.json"

#: LITERAL y sensible a mayúsculas -- a propósito: el marcador es SIEMPRE `NUMBER` en
#: mayúsculas en el corpus, y una coincidencia en minúsculas sería una palabra de verdad.
REGEX_NUMBER = r"\bNUMBER\b"
_PATRON_NUMBER = re.compile(REGEX_NUMBER)  # sin re.IGNORECASE

#: Las noticias completas de FakeNewsCorpusSpanish caben de sobra en el límite por omisión
#: de `csv` (131072 bytes/campo), pero se sube igual: un campo de origen distinto que no
#: quepa no debe tumbar el recuento con un `_csv.Error` silencioso de interpretación.
csv.field_size_limit(10_000_000)

COLUMNAS_ESPERADAS = {"row_id", "particion", "target", "texto", "topic", "source", "version_origen"}


def _sha256_de(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def medir() -> dict:
    with RUTA_CSV.open("r", encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    if not filas:
        raise ValueError(f"{RUTA_CSV} está vacío")
    columnas = set(filas[0])
    if columnas != COLUMNAS_ESPERADAS:
        raise ValueError(f"{RUTA_CSV} no tiene las columnas esperadas: "
                          f"faltan {COLUMNAS_ESPERADAS - columnas}, sobran {columnas - COLUMNAS_ESPERADAS}")

    por_version_y_particion: dict[str, dict[str, dict[str, int]]] = {}
    train_por_target: dict[str, dict[str, int]] = {}
    n_con_number_total = 0
    for fila in filas:
        tiene_number = bool(_PATRON_NUMBER.search(fila["texto"]))
        n_con_number_total += int(tiene_number)

        bloque = (por_version_y_particion
                  .setdefault(fila["version_origen"], {})
                  .setdefault(fila["particion"], {"n_filas": 0, "n_con_number": 0}))
        bloque["n_filas"] += 1
        bloque["n_con_number"] += int(tiene_number)

        if fila["particion"] == "train":
            b = train_por_target.setdefault(fila["target"], {"n_filas": 0, "n_con_number": 0})
            b["n_filas"] += 1
            b["n_con_number"] += int(tiene_number)

    return {
        "tarea": "A",
        "csv": {"ruta_relativa_al_directorio": "tareas/tarea_a.csv", "sha256": _sha256_de(RUTA_CSV),
                "n_filas": len(filas)},
        "regex": REGEX_NUMBER,
        "sensible_a_mayusculas": True,
        "n_filas_con_number_total": n_con_number_total,
        "por_version_y_particion": por_version_y_particion,
        "train_por_target": train_por_target,
        "script_sha256": _sha256_de(Path(__file__)),
        "generado": datetime.now(timezone.utc).astimezone().isoformat(),
    }


def main() -> int:
    datos = medir()
    texto = json.dumps(datos, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    RUTA_SALIDA.write_text(texto, encoding="utf-8")
    print(f"escrito {RUTA_SALIDA} ({len(texto.encode('utf-8'))} bytes)")
    print(f"n_filas_con_number_total={datos['n_filas_con_number_total']} de {datos['csv']['n_filas']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
