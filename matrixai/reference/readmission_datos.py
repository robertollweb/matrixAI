"""CONTRATO 81-C6 — el dataset SINTÉTICO del caso, y por qué lo es.

**Aquí no hay ni un dato de nadie.** El §16.2 exige datos sintéticos o
debidamente desidentificados, y se elige lo primero por la razón obvia:
un dataset desidentificado sigue siendo de personas, y este caso es una
demostración que va a viajar en un repositorio público.

Las filas se generan de una **regla declarada** más ruido, con semilla:
así el dataset se puede regenerar byte a byte —que es lo que pide el
paquete reproducible del 82— y cualquiera puede leer de qué está hecha la
etiqueta en vez de tener que confiar en ella.

**Y las columnas del CSV son EXACTAMENTE el vector de la fusión.** No es
comodidad: si el entrenamiento viera unas columnas y el servicio armara
otras, el modelo predeciría bien en la suite y mal en el producto. Es el
invariante que este repositorio ya tiene escrito con todas las letras
(coherencia train/serve), y aquí se cumple porque las dos mitades llaman a
la MISMA función de codificación.
"""

from __future__ import annotations

import csv
import io
import random
from typing import Any

from matrixai.reference.readmission import (
    CATEGORIAS,
    codificar_texto,
    codificar_tabular,
    fusionar,
    preparar_texto,
)
from matrixai.reference.readmission_pipeline import TALLA_TABULAR, VOCABULARIO

__all__ = ["ETIQUETAS", "REGLA_DE_LA_ETIQUETA", "columnas", "generar_dataset",
           "vector_de"]

ETIQUETAS = ("no_reingreso", "reingreso")

#: DE QUÉ ESTÁ HECHA LA ETIQUETA, en una frase que se puede auditar. Un
#: dataset sintético cuya regla no se publica es tan opaco como uno real,
#: y encima falsamente tranquilizador.
REGLA_DE_LA_ETIQUETA = (
    "riesgo = 0.35·edad_norm + 0.30·ingresos_previos_norm + 0.20·estancia_norm "
    "+ 0.10·(disnea|edema|oxigeno en la nota) + 0.05·soledad; "
    "etiqueta = reingreso si riesgo + ruido(±0.05) > 0.5"
)

#: Frases del dominio, con las palabras del vocabulario declarado. Cortas
#: y sin nada que se parezca a una nota real de nadie.
_FRASES_RIESGO = (
    "disnea de reposo y edema en miembros inferiores",
    "precisa oxigeno domiciliario, refiere disnea",
    "edema y ajuste de diuretico, vive en soledad",
    "confusion y caida reciente, disnea al esfuerzo",
)
_FRASES_LLANAS = (
    "dolor controlado, sin fiebre",
    "buena evolucion, sin disnea ni edema",
    "control de insulina estable, sin dolor",
    "afebril, sin dolor ni confusion",
)


def columnas() -> list[str]:
    """Los nombres del CSV: las 9 tabulares y las del vocabulario.

    Se deducen de la MISMA codificación que usa el pipeline, llamándola:
    escribirlas a mano aquí sería un segundo sitio declarando la forma del
    vector, y el día que cambie una, el modelo entrenaría con otra.
    """
    ejemplo = codificar_tabular({
        "edad": 0, "ingresos_previos_12m": 0, "dias_de_estancia": 0,
        "servicio": CATEGORIAS["servicio"][0], "alta_voluntaria": 0,
    })
    return list(ejemplo["values"]) + [f"nota_{p}" for p in VOCABULARIO]


def vector_de(registro: dict[str, Any], nota: str) -> list[float]:
    """El vector de un caso — la MISMA función que usa el pipeline."""
    tabular = codificar_tabular(registro)
    texto = codificar_texto(preparar_texto(nota), VOCABULARIO)
    return fusionar(tabular, texto,
                    tallas_esperadas=(TALLA_TABULAR, len(VOCABULARIO)))["vector"]


def _riesgo(registro: dict[str, Any], nota: str) -> float:
    edad = min(max(float(registro["edad"]), 0.0), 120.0) / 120.0
    ingresos = min(max(float(registro["ingresos_previos_12m"]), 0.0), 20.0) / 20.0
    estancia = min(max(float(registro["dias_de_estancia"]), 0.0), 60.0) / 60.0
    agudo = 1.0 if any(p in nota for p in ("disnea", "edema", "oxigeno")) else 0.0
    soledad = 1.0 if "soledad" in nota else 0.0
    return 0.35 * edad + 0.30 * ingresos + 0.20 * estancia + 0.10 * agudo + 0.05 * soledad


def generar_dataset(filas: int = 400, semilla: int = 42) -> str:
    """El CSV del caso, determinista para una semilla dada.

    Determinista importa dos veces: el paquete del 82 promete que se puede
    **regenerar el mismo dataset**, y sin eso `R1` no puede comparar nada.
    """
    if type(filas) is not int or filas < 1:
        raise ValueError(f"filas debe ser un entero positivo, y es {filas!r}")
    rnd = random.Random(semilla)
    buffer = io.StringIO()
    escritor = csv.writer(buffer, lineterminator="\n")
    escritor.writerow([*columnas(), "etiqueta"])

    for _ in range(filas):
        alta_riesgo = rnd.random() < 0.5
        registro = {
            "edad": rnd.randint(70, 95) if alta_riesgo else rnd.randint(25, 65),
            "ingresos_previos_12m": rnd.randint(2, 8) if alta_riesgo else rnd.randint(0, 2),
            "dias_de_estancia": rnd.randint(10, 40) if alta_riesgo else rnd.randint(1, 8),
            "servicio": rnd.choice(CATEGORIAS["servicio"]),
            "alta_voluntaria": rnd.choice([0, 1]),
        }
        nota = rnd.choice(_FRASES_RIESGO if alta_riesgo else _FRASES_LLANAS)
        riesgo = _riesgo(registro, nota) + rnd.uniform(-0.05, 0.05)
        etiqueta = ETIQUETAS[1] if riesgo > 0.5 else ETIQUETAS[0]
        escritor.writerow([*(f"{v:.6f}" for v in vector_de(registro, nota)), etiqueta])

    return buffer.getvalue()
