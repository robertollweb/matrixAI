#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 111-C1 — la guarda de arranque de la acción.

**EL FALLO QUE ESTO EXISTE PARA QUE NO VUELVA A PASAR.** `action.yml`
instala `matrixai-core` de PyPI y luego ejecuta
`python -m matrixai.ci.verify_action`. Ese módulo **no viaja en 1.7.0 ni
en 1.7.1** —comprobado el 2026-09-15 con `git cat-file -e` sobre las dos
etiquetas, y 1.7.1 era la última publicada—, así que la acción, hoy, en
la máquina de cualquiera que la use, moría con

    /usr/bin/python: No module named matrixai.ci

y ya está: ni qué falta, ni por qué, ni qué hay que poner para
arreglarlo. En el árbol de desarrollo funciona SOLO porque `matrixai`
está instalado en modo editable, que es la trampa de siempre —lo que se
prueba aquí no es lo que corre allí—.

**Por qué es un fichero aparte y no una función del módulo.** Lo que hay
que detectar es justo que el módulo NO está; una guarda que viviera
dentro de él tampoco se podría importar. Este fichero viaja con la
acción (se lee por `$GITHUB_ACTION_PATH`, que es el sitio donde GitHub
deja la propia acción), no con el paquete de PyPI, así que está siempre.

**Lo que NO hace: decidir por la cadena de versión.** Lo que importa es
si el módulo se puede importar y trae lo que se le va a llamar, no qué
diga un número —el árbol editable de esta máquina declara `0.1.0` y sí
lo tiene, y una comparación de versiones lo habría rechazado—. La
versión se MIDE y se enseña para diagnosticar; no decide.

**Y falla EN ROJO, nunca en verde.** Es el mismo motivo por el que existe
el corte entero: un tick verde cuando la verificación no llegó ni a
empezar convierte «no se midió» en «está bien» a la vista de un equipo.
"""

from __future__ import annotations

import os
import sys

#: Lo que `action.yml` ejecuta. Si esto cambia allí y no aquí, la guarda
#: comprueba una cosa y la acción usa otra: hay una prueba que lo ata.
MODULO = "matrixai.ci.verify_action"

#: Lo que se le llama al módulo. Que el paquete exista no basta: una
#: versión que trajera `matrixai/ci/` con otra forma rompería igual.
ATRIBUTO = "main"

#: La primera versión de `matrixai-core` que trae este módulo.
#: MEDIDO el 2026-09-15: `matrixai/ci/` NO está en v1.7.0 ni en v1.7.1. Se
#: iba a publicar como 1.7.2; el 2026-09-17 Roberto decidió **1.8.0**, porque
#: esa release trae módulos públicos nuevos y SemVer la hace menor. La guarda
#: sigue haciendo falta después de publicarla: quien fije una versión anterior
#: (o un espejo de PyPI atrasado) recibe el motivo en vez de un error de Python.
VERSION_MINIMA = "1.8.0"

#: Dónde preguntar la versión, en orden. `pip install matrixai-core` deja
#: la distribución `matrixai-core`; el árbol editable de desarrollo deja
#: `matrixai`. Se prueban las dos y, si no hay ninguna, se dice «unknown»
#: en vez de inventarse una.
DISTRIBUCIONES = ("matrixai-core", "matrixai")


def version_instalada() -> str:
    """La versión de `matrixai-core` que hay, o `"unknown"`.

    Un valor ausente NO es un cero ni una versión antigua: es «no se ha
    podido saber», y eso se escribe tal cual en el mensaje.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8, fuera de soporte
        return "unknown"
    for nombre in DISTRIBUCIONES:
        try:
            return f"{version(nombre)} (as `{nombre}`)"
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - un metadato roto no es la noticia
            continue
    return "unknown"


def problema_de_importacion() -> str | None:
    """`None` si el módulo está y sirve; si no, por qué no, en una línea."""
    try:
        import importlib

        modulo = importlib.import_module(MODULO)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo aquí es EL fallo
        return f"{type(exc).__name__}: {exc}"
    if not callable(getattr(modulo, ATRIBUTO, None)):
        return (f"the module was imported but has no callable {ATRIBUTO}(): "
                f"this matrixai-core ships a different {MODULO}")
    return None


def diagnostico(problema: str | None, version: str) -> tuple[int, list[str]]:
    """El veredicto de la guarda. Función PURA: se le dan las dos medidas.

    Devuelve el código de salida y las líneas del mensaje. Separado de
    quien mide para que la suite pueda probar el mensaje entero sin tener
    que desinstalar nada.
    """
    if problema is None:
        return 0, [f"matrixai-core {version} provides {MODULO}."]
    return 1, [
        f"This action runs `python -m {MODULO}`, and the matrixai-core "
        f"installed on this runner does NOT provide it.",
        f"Installed matrixai-core: {version}. Import failed with: {problema}.",
        f"It needs matrixai-core >= {VERSION_MINIMA}, the first release that "
        f"ships `matrixai/ci/`; 1.7.1 and earlier do NOT.",
        f"Fix it by leaving `matrixai-version` empty (it takes the latest "
        f"release) or setting it to '>={VERSION_MINIMA}', or by installing "
        f"matrixai-core from source in a step before this action.",
        "The job is failing HERE, on purpose. The alternative is a green tick "
        "for a verification that never ran, and in CI nobody opens the log of "
        "a green job.",
    ]


def main(argv: list[str] | None = None) -> int:
    """Mide, escribe el motivo donde se ve, y devuelve el código."""
    version = version_instalada()
    problema = problema_de_importacion()
    codigo, lineas = diagnostico(problema, version)

    if codigo == 0:
        print(lineas[0])
        return 0

    titular = "matrixai-core does not ship this action's module"
    # Una anotación de GitHub es UNA línea: los saltos van como `%0A`, que
    # es como las pinta la página del trabajo. Y además se escribe el texto
    # entero en el registro y en el resumen, porque una anotación se corta.
    cuerpo = "%0A".join(lineas)
    print(f"::error title=matrixai verify: {titular}::{cuerpo}")
    print(f"\n{titular}\n")
    for linea in lineas:
        print(f"  {linea}")
    resumen = os.environ.get("GITHUB_STEP_SUMMARY")
    if resumen:
        try:
            with open(resumen, "a", encoding="utf-8") as fh:
                fh.write("## matrixai verify — NOT VERIFIED\n\n"
                         f"**{titular}**\n\n"
                         + "\n".join(f"* {l}" for l in lineas) + "\n\n")
        except OSError:
            pass
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
