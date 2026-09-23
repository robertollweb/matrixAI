# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""117-C1 — los datos públicos de Fase 0 salen del artefacto, nunca de una mano.

`benchmarks/datos_publicos/fase0_publico.json` es lo que leerán la página «Cómo medimos»
y las fichas de «Ejemplos medidos». Si alguien retoca una cifra, o una pasada cambia y
nadie lo regenera, esto falla: la cifra publicada tiene que ser la que produce el
generador desde el artefacto sellado.

**REESCRITO el 2026-09-23 (117-C5)**: el JSON publicado pasó de un único bloque compuesto
desde la v2 a DOS bloques — `principal` (la pasada AMPLIA, la que sostiene la cartera) y
`ultima_medicion` (la v2, aparte, no anclable entera). Cada aserto que antes leía
`publicado[...]` ahora lee `publicado["principal"][...]` o recorre los dos bloques, según
si el invariante aplica a uno o a los dos.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "datos_publicos_generar", RAIZ / "benchmarks" / "datos_publicos" / "generar.py")
generar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generar)


def _entradas_principal():
    return (json.loads(generar.RUTA_ARTEFACTO_PRINCIPAL.read_text(encoding="utf-8")),
            json.loads(generar.RUTA_PROTOCOLO_PRINCIPAL.read_text(encoding="utf-8")))


def _entradas_ultima_medicion():
    return (json.loads(generar.RUTA_ARTEFACTO_ULTIMA_MEDICION.read_text(encoding="utf-8")),
            json.loads(generar.RUTA_PROTOCOLO_ULTIMA_MEDICION.read_text(encoding="utf-8")))


def _publicado():
    return json.loads(generar.RUTA_PUBLICADO.read_text(encoding="utf-8"))


def test_lo_publicado_es_BYTE_A_BYTE_lo_que_sale_del_artefacto():
    assert generar.RUTA_PUBLICADO.read_text(encoding="utf-8") == generar.generar(), (
        "fase0_publico.json no es lo que sale de los artefactos: "
        "python3 benchmarks/datos_publicos/generar.py --escribir, y mirar por qué cambió")


def test_el_bloque_principal_es_la_AMPLIA_y_es_anclable_entera():
    """117-C5: la página y las fichas se enseñan desde el bloque que SOSTIENE la cartera.
    Un sabotaje que apunte `principal` a la v2 (no anclable entera) tiene que ponerse rojo
    aquí, no solo en la prueba de C5 que compara contra `cartera.py`."""
    publicado = _publicado()
    principal = publicado["principal"]
    assert principal["fuente"]["artefacto"] == "pasada_amplia_101_c5_resultado.json"
    assert principal["fuente"]["digest_resultados_crudos"] == (
        "3646af61b155c1321ad3fd71f25e9300ed9797b9b5dec0f37ed835cd835ecf48")
    assert principal["procedencia"]["anclable_entera"] is True


def test_el_bloque_ultima_medicion_es_la_v2_y_NO_es_anclable_entera():
    publicado = _publicado()
    ultima = publicado["ultima_medicion"]
    assert ultima["fuente"]["artefacto"] == "pasada_v2_113_resultado.json"
    assert ultima["fuente"]["digest_resultados_crudos"] == (
        "600ebdaad2ca206eac148e25c38b57add8ad2d64652fafca3d8226758010c0d0")
    assert ultima["procedencia"]["anclable_entera"] is False
    assert ultima["procedencia"]["por_procedencia"]  # 117-C1: no está vacía


def test_el_veredicto_publicado_es_el_de_cada_pasada_tambien_el_de_nuestra_red():
    """Invariante 5: nuestra propia red, tal cual. Se compara con CADA artefacto, no con un
    número escrito aquí, para que esta prueba no caduque con la próxima pasada — y para los
    DOS bloques por separado, que cada uno cita su propio artefacto."""
    publicado = _publicado()
    for clave, (artefacto, _) in (("principal", _entradas_principal()),
                                  ("ultima_medicion", _entradas_ultima_medicion())):
        bloque = publicado[clave]
        for motor, v in artefacto["alcance_y_veredicto"]["por_motor"].items():
            assert bloque["veredicto"][motor]["cumplidos"] == v["cumplidos"], (clave, motor)
            assert bloque["veredicto"][motor]["cumple_la_regla"] == v["cumple_la_regla"], (clave, motor)
        assert "matrixai.dense.torch_cpu" in bloque["veredicto"], clave


def test_la_procedencia_de_la_ultima_medicion_viaja_DENTRO_de_su_bloque():
    """La v2 tiene 110 intentos medidos con el árbol sucio. Si eso se quedara en una nota
    aparte, la página podría enseñar las cifras sin la advertencia. Y tiene que quedarse
    DENTRO de `ultima_medicion`, no filtrarse al bloque `principal`."""
    publicado = _publicado()
    ultima = publicado["ultima_medicion"]
    procedencia = ultima["procedencia"]
    assert procedencia["anclable_entera"] is False
    assert sum(p["n_intentos"] for p in procedencia["por_procedencia"]) == ultima["fuente"]["n_intentos"]
    assert any(p["repositorios_sucios"] for p in procedencia["por_procedencia"])
    # El principal no hereda ni el aviso ni el conteo de intentos sucios de la ultima.
    assert publicado["principal"]["procedencia"]["anclable_entera"] is True
    assert all(not p["repositorios_sucios"] for p in publicado["principal"]["procedencia"]["por_procedencia"])


def test_los_sellados_salen_marcados_y_la_visibilidad_no_se_llama_licencia():
    publicado = _publicado()
    for clave, (_, protocolo) in (("principal", _entradas_principal()),
                                  ("ultima_medicion", _entradas_ultima_medicion())):
        bloque = publicado[clave]
        assert sum(c["sellado"] for c in bloque["conjuntos"]) == sum(
            d["sellado"] for d in protocolo["datasets"]), clave
        assert all("licencia" not in c for c in bloque["conjuntos"]), clave
        assert len(bloque["lo_que_no_dicen"]) == len(generar.LO_QUE_NO_DICEN) >= 4, clave
        assert all(set(x) == {"es", "en"} for x in bloque["lo_que_no_dicen"]), clave


def test_una_cifra_retocada_en_el_ARTEFACTO_PRINCIPAL_cambia_lo_publicado():
    """La otra mitad: probar el artefacto no es probar el código que lo produce. Se compone
    uno NUEVO con el mismo código, desde un artefacto con una métrica cambiada — sobre el
    bloque PRINCIPAL, que es el que la página enseña."""
    artefacto, protocolo = _entradas_principal()
    original = generar.serializar(generar.componer(artefacto, protocolo, "x.json"))
    tocado = copy.deepcopy(artefacto)
    r = next(x for x in tocado["resultados"] if x["motor"] == "baseline" and x["estado"] == "completed")
    r[r["metrica_de_cierre"]] = (r[r["metrica_de_cierre"]] or 0.0) + 0.123
    assert generar.serializar(generar.componer(tocado, protocolo, "x.json")) != original


def test_una_cifra_retocada_en_el_ARTEFACTO_DE_LA_ULTIMA_MEDICION_cambia_lo_publicado():
    """El mismo sabotaje, sobre el segundo artefacto: los dos bloques usan el mismo
    `componer()`, y los dos tienen que reaccionar a un dato tocado."""
    artefacto, protocolo = _entradas_ultima_medicion()
    original = generar.serializar(generar.componer(artefacto, protocolo, "x.json"))
    tocado = copy.deepcopy(artefacto)
    r = next(x for x in tocado["resultados"] if x["motor"] == "baseline" and x["estado"] == "completed")
    r[r["metrica_de_cierre"]] = (r[r["metrica_de_cierre"]] or 0.0) + 0.123
    assert generar.serializar(generar.componer(tocado, protocolo, "x.json")) != original


def test_si_las_medias_no_reproducen_las_distancias_de_la_pasada_NO_se_genera():
    """La media es lo único que se calcula aquí; si divergiera de la cuenta de la pasada,
    se publicaría un segundo número. Una métrica de un motor que COMPITE, cambiada en el
    artefacto sin que su distancia cambie, tiene que parar el generador — comprobado con
    el artefacto PRINCIPAL, que es el que la página enseña."""
    artefacto, protocolo = _entradas_principal()
    tocado = copy.deepcopy(artefacto)
    r = next(x for x in tocado["resultados"] if x["motor"] == "lightgbm" and x["estado"] == "completed")
    r[r["metrica_de_cierre"]] += 0.05
    with pytest.raises(generar.DatosQueNoCuadran):
        generar.componer(tocado, protocolo, "x.json")


def test_una_pasada_a_medias_no_se_publica():
    artefacto, protocolo = _entradas_principal()
    tocado = dict(artefacto, parcial=True)
    with pytest.raises(generar.DatosQueNoCuadran):
        generar.componer(tocado, protocolo, "x.json")
