"""LA RUEDA LLEVA LOS DATOS QUE EL PAQUETE LEE (2026-09-17).

Construyendo la 1.8.0 en un contenedor limpio —`python -m build`, instalar la
rueda FUERA del árbol, importar— faltaba `matrixai/text/embeddings/
catalogo_medido.json`: `package-data` solo incluía `templates/**`. Desde el árbol
editable de desarrollo todo pasaba, porque el fichero está al lado del código.
Quien instalara desde PyPI —y la imagen del Studio, que instala el núcleo con un
`pip install` normal— se quedaba sin el catálogo que usan los proveedores de
texto de `matrixai-engines`.

Esta prueba no construye nada: compara cada fichero que NO es Python bajo
`matrixai/`, de los que git sigue, con los patrones de `package-data`. Un dato
nuevo sin su patrón la pone en rojo aquí, no en casa de quien lo instala.
"""
from __future__ import annotations

import fnmatch
import subprocess
import tomllib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def _patrones() -> list[str]:
    datos = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))
    return datos["tool"]["setuptools"]["package-data"]["matrixai"]


def _ficheros_no_python_seguidos_por_git() -> list[str]:
    salida = subprocess.run(["git", "ls-files", "matrixai"], cwd=RAIZ,
                            capture_output=True, text=True, check=True).stdout
    return [f[len("matrixai/"):] for f in salida.splitlines()
            if f and not f.endswith(".py")]


def test_cada_fichero_de_datos_del_paquete_casa_con_package_data():
    patrones = _patrones()
    fuera = [f for f in _ficheros_no_python_seguidos_por_git()
             if not any(fnmatch.fnmatch(f, p) for p in patrones)]
    assert not fuera, (
        "ficheros del paquete que NO viajarían en la rueda (añade su patrón a "
        f"[tool.setuptools.package-data] en pyproject.toml): {fuera}")


def test_y_el_caso_que_lo_destapo_esta_cubierto():
    """La otra mitad: una lista vacía de ficheros también dejaría pasar la de
    arriba."""
    ficheros = _ficheros_no_python_seguidos_por_git()
    assert "text/embeddings/catalogo_medido.json" in ficheros
    assert any(fnmatch.fnmatch("text/embeddings/catalogo_medido.json", p) for p in _patrones())
