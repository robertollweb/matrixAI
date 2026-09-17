"""LA RUEDA LLEVA LOS DATOS QUE EL PAQUETE LEE (2026-09-17).

Construyendo la 1.8.0 en un contenedor limpio —`python -m build`, instalar la
rueda FUERA del árbol, importar— faltaba `matrixai/text/embeddings/
catalogo_medido.json`: `package-data` solo incluía `templates/**`. Desde el árbol
editable de desarrollo todo pasaba, porque el fichero está al lado del código.
Quien instalara desde PyPI —y la imagen del Studio, que instala el núcleo con un
`pip install` normal— se quedaba sin el catálogo que usan los proveedores de
texto de `matrixai-engines`.

Esta prueba no construye nada: expande los patrones de `package-data` con
`Path.glob` —como setuptools, donde `*` NO cruza `/`— y exige que cada fichero no
Python que git sigue bajo `matrixai/` salga en esa expansión. Un dato nuevo sin su
patrón la pone en rojo aquí, no en casa de quien lo instala.

**Y NO CON `fnmatch`**, que es como se escribió primero: ahí `*` sí cruza `/`, así
que `text/embeddings/*.json` daba por cubierto un `text/embeddings/extra/nuevo.json`
que la rueda real NO lleva, y al revés marcaba fuera un hijo directo de
`templates/` que la rueda SÍ lleva. Lo midió deployer-02 el 2026-09-17
construyendo ruedas reales de los dos casos.
"""
from __future__ import annotations

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


def _lo_que_empaqueta_package_data() -> set[str]:
    paquete = RAIZ / "matrixai"
    return {p.relative_to(paquete).as_posix()
            for patron in _patrones() for p in paquete.glob(patron) if p.is_file()}


def test_cada_fichero_de_datos_del_paquete_casa_con_package_data():
    empaquetados = _lo_que_empaqueta_package_data()
    fuera = [f for f in _ficheros_no_python_seguidos_por_git() if f not in empaquetados]
    assert not fuera, (
        "ficheros del paquete que NO viajarían en la rueda (añade su patrón a "
        f"[tool.setuptools.package-data] en pyproject.toml): {fuera}")


def test_y_el_caso_que_lo_destapo_esta_cubierto():
    """La otra mitad: una lista vacía de ficheros también dejaría pasar la de
    arriba."""
    ficheros = _ficheros_no_python_seguidos_por_git()
    assert "text/embeddings/catalogo_medido.json" in ficheros
    assert "text/embeddings/catalogo_medido.json" in _lo_que_empaqueta_package_data()
