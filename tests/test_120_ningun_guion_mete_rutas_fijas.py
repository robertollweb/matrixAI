# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""NINGÚN GUION DEL NÚCLEO METE EN `sys.path` LOS ÁRBOLES PRINCIPALES CON RUTA FIJA (02-10).

La re-auditoría de 120-C1 lo midió: `benchmarks/texto_107c30/medir_c30*.py` hacían
`sys.path.insert(0, "/home/deployer/matrixAI")` (y motores) al IMPORTARSE, y cuatro pruebas del
núcleo (y una de motores) los importan al RECOGER. Desde un worktree, todo proceso hijo de la
pasada —y el `__path__` del paquete de espacio de nombres `benchmarks`— cargaba el código del
árbol principal, no el que se probaba: un verde que no mide lo que dice. Sobre el árbol principal
no se ve (las dos rutas son la misma), y por eso hace falta esta prueba: busca la forma, no el
efecto.

SUS LÍMITES, DICHOS (re-auditoría final de 120-C1): caza la forma clásica en UNA línea
(`sys.path.insert/append(…"/home/deployer/…")`). NO ve `sys.path[0:0] =`, `site.addsitedir`, una
variable o una tupla con la ruta recorrida en un bucle (el caso de `test_c101_c5`, arreglado a mano
el 02-10), `os.environ["PYTHONPATH"]`, ni nada fuera de `benchmarks/`, `tests/` y `scripts/`. Ampliar
la búsqueda a «fichero con `sys.path` y una ruta literal al principal» da falsos positivos
(`medir_114c5.py` la usa para registrar un commit). Es una red, no una garantía: un rojo de
`test_m3_…hijo_importa` en motores sigue siendo la señal de verdad.
"""
from __future__ import annotations

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_FIJA = re.compile(r"""sys\.path\.(insert|append)\([^)]*["']/home/deployer/""")


def test_ningun_guion_de_benchmarks_ni_de_pruebas_mete_una_ruta_fija_en_sys_path():
    culpables = []
    for carpeta in ("benchmarks", "tests", "scripts"):
        for f in (RAIZ / carpeta).rglob("*.py"):
            if f.resolve() == Path(__file__).resolve():
                continue  # su docstring cita la línea vieja
            for n, linea in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if RUTA_FIJA.search(linea):
                    culpables.append(f"{f.relative_to(RAIZ)}:{n}: {linea.strip()}")
    assert not culpables, "rutas fijas en sys.path (usar rutas relativas a __file__):\n" + "\n".join(culpables)


def test_importar_medir_c30_no_mete_el_arbol_principal_si_ya_esta_el_propio():
    """El efecto, en este árbol: tras importar el medidor como lo hacen las pruebas, `sys.path`
    no gana ninguna ruta que no sea de ESTE núcleo o de su motores hermano."""
    import importlib.util
    import sys
    antes = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            "_medir_c30_guardia", RAIZ / "benchmarks" / "texto_107c30" / "medir_c30.py")
        modulo = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(RAIZ / "benchmarks" / "texto_107c30"))
        spec.loader.exec_module(modulo)
        nuevas = [p for p in sys.path if p not in antes]
        propias = {str(RAIZ), str(RAIZ.parent / "matrixai-engines" / "src"),
                   str(RAIZ / "benchmarks" / "texto_107c30")}
        assert set(nuevas) <= propias, f"rutas ajenas en sys.path: {set(nuevas) - propias}"
    finally:
        sys.path[:] = antes
