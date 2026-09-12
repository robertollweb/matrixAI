# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — CUATRO de los CUARENTA datasets del protocolo no se podían leer.

El 10 % del protocolo, y nadie lo sabía: la pasada exploratoria del 101-C3
solo usa doce, y ninguno de los cuatro está entre ellos. Salió el 2026-09-12 al
calibrar los tiempos de C5, que es exactamente para lo que servía calibrar.

Cuando C5 llegue a ellos fallarán, y la regla de cierre cuenta un fallo como
**dataset perdido para ese motor**: cuatro datasets perdidos para los siete
motores a la vez, por un problema de FORMATO que no tiene nada que ver con la
calidad de ningún modelo.

Las cuatro causas, medidas:
  · `pendigits` escribe `, 8` para cuadrar las columnas a la vista, y declara
    `{0,…,9}` sin espacios.
  · `diamonds` y `house_prices_nominal` declaran `{Fair, Good, 'Very Good'}`
    —con comillas y espacios— y sus datos vienen sin ellas.
  · `us_crime` tiene una columna `STRING`, que `scipy` no soporta.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.fase0.lector_arff import cargar, normalizar_texto_arff

DATOS = Path("/home/deployer/fase0_openml_datos/arff")
PROTOCOLO = json.loads(
    (Path(__file__).resolve().parent.parent / "benchmarks" / "fase0"
     / "protocolo_exploratorio.json").read_text(encoding="utf-8"))
POR_NOMBRE = {e["nombre"]: e for e in PROTOCOLO["datasets"]}

pytestmark = pytest.mark.skipif(not DATOS.is_dir(),
                                reason="los ARFF del protocolo no están descargados")


def _cargar(nombre):
    e = POR_NOMBRE[nombre]
    return cargar(DATOS / f"{e['data_id']}.arff", e.get("columna_objetivo"))


@pytest.mark.parametrize("nombre,filas", [
    ("pendigits", 10992), ("diamonds", 53940),
    ("house_prices_nominal", 1460), ("us_crime", 1994)])
def test_los_cuatro_que_no_se_podian_leer_YA_se_leen(nombre, filas):
    r = _cargar(nombre)
    assert len(r.filas) == filas
    assert r.normalizaciones, f"{nombre} entró sin normalizar: ¿ya no hacía falta?"


def test_los_CUARENTA_del_protocolo_se_leen():
    """El número que de verdad importa para C5. Si baja, hay un dataset que la
    validación amplia va a perder entero."""
    fallan = []
    for e in PROTOCOLO["datasets"]:
        f = DATOS / f"{e['data_id']}.arff"
        if not f.exists():
            continue
        try:
            cargar(f, e.get("columna_objetivo"))
        except Exception as error:  # noqa: BLE001
            fallan.append(f"{e['nombre']}: {type(error).__name__}")
    assert fallan == [], f"datasets ilegibles: {fallan}"


def test_los_36_LIMPIOS_no_se_tocan():
    """La otra mitad, y la que impide que esto se vuelva un lavado general: un
    fichero que entra tal cual NO se normaliza, y su informe lo dice saliendo
    vacío. Normalizar siempre haría invisible la diferencia entre un fichero
    limpio y uno que hubo que arreglar."""
    tocados = []
    for e in PROTOCOLO["datasets"]:
        f = DATOS / f"{e['data_id']}.arff"
        if not f.exists():
            continue
        if cargar(f, e.get("columna_objetivo")).normalizaciones:
            tocados.append(e["nombre"])
    assert sorted(tocados) == ["diamonds", "house_prices_nominal", "pendigits", "us_crime"]


def test_la_columna_STRING_se_EXCLUYE_y_se_dice_cual():
    """No se convierte en categórica en silencio: en `us_crime` es el nombre
    del municipio, un IDENTIFICADOR de la fila. Metérselo al modelo como
    categórica sería darle una columna que identifica cada caso — una fuga en
    potencia, y además inútil."""
    r = _cargar("us_crime")
    assert r.columnas_excluidas == ["communityname"]
    assert "communityname" not in r.filas[0]


def test_EL_FICHERO_DEL_DISCO_NO_SE_TOCA():
    """Lo que hace que el sha256 del protocolo siga valiendo. Normalizar el
    fichero en disco para poder leerlo rompería justo la huella que lo hace
    útil como evidencia."""
    import hashlib
    e = POR_NOMBRE["pendigits"]
    ruta = DATOS / f"{e['data_id']}.arff"
    antes = hashlib.sha256(ruta.read_bytes()).hexdigest()
    _cargar("pendigits")
    assert hashlib.sha256(ruta.read_bytes()).hexdigest() == antes
    assert antes == e["sha256_arff"], "el fichero ya no es el que el protocolo registró"


def test_la_normalizacion_no_INVENTA_valores():
    """Recorta espacios y comillas; no traduce, ni rellena, ni renombra."""
    crudo = ("@relation r\n@attribute a {x, 'y z'}\n@attribute b REAL\n"
             "@data\n x ,1.5\n'y z', 2.0\n")
    texto, notas, excluidas = normalizar_texto_arff(crudo)
    assert "{x,y z}" in texto
    assert "x,1.5" in texto and "y z,2.0" in texto
    assert excluidas == []
    assert notas


def test_una_columna_STRING_sale_del_TEXTO_normalizado_entera():
    crudo = ("@relation r\n@attribute id STRING\n@attribute v REAL\n"
             "@data\nLakewood,0.5\nTukwila,0.7\n")
    texto, _, excluidas = normalizar_texto_arff(crudo)
    assert excluidas == ["id"]
    assert "Lakewood" not in texto
    assert "0.5" in texto
