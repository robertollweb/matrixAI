# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""117-C1 — los datos públicos de Fase 0 salen del artefacto, nunca de una mano.

`benchmarks/datos_publicos/fase0_publico.json` es lo que leerán la página «Cómo medimos»
y las fichas de «Ejemplos medidos». Si alguien retoca una cifra, o la pasada cambia y
nadie lo regenera, esto falla: la cifra publicada tiene que ser la que produce el
generador desde el artefacto sellado.
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


def _entradas():
    return (json.loads(generar.RUTA_ARTEFACTO.read_text(encoding="utf-8")),
            json.loads(generar.RUTA_PROTOCOLO.read_text(encoding="utf-8")))


def test_lo_publicado_es_BYTE_A_BYTE_lo_que_sale_del_artefacto():
    assert generar.RUTA_PUBLICADO.read_text(encoding="utf-8") == generar.generar(), (
        "fase0_publico.json no es lo que sale del artefacto: "
        "python3 benchmarks/datos_publicos/generar.py --escribir, y mirar por qué cambió")


def test_el_veredicto_publicado_es_el_de_la_pasada_tambien_el_de_nuestra_red():
    """Invariante 5: nuestra propia red, tal cual. Se compara con el artefacto, no con un
    número escrito aquí, para que esta prueba no caduque con la próxima pasada."""
    artefacto, _ = _entradas()
    publicado = json.loads(generar.RUTA_PUBLICADO.read_text(encoding="utf-8"))
    for motor, v in artefacto["alcance_y_veredicto"]["por_motor"].items():
        assert publicado["veredicto"][motor]["cumplidos"] == v["cumplidos"], motor
        assert publicado["veredicto"][motor]["cumple_la_regla"] == v["cumple_la_regla"], motor
    assert "matrixai.dense.torch_cpu" in publicado["veredicto"]


def test_la_procedencia_viaja_DENTRO_y_dice_que_no_es_anclable_entera():
    """La v2 tiene 110 intentos medidos con el árbol sucio. Si eso se quedara en una nota
    aparte, la página podría enseñar las cifras sin la advertencia."""
    publicado = json.loads(generar.RUTA_PUBLICADO.read_text(encoding="utf-8"))
    procedencia = publicado["procedencia"]
    assert procedencia["anclable_entera"] is False
    assert sum(p["n_intentos"] for p in procedencia["por_procedencia"]) == publicado["fuente"]["n_intentos"]
    assert any(p["repositorios_sucios"] for p in procedencia["por_procedencia"])


def test_los_sellados_salen_marcados_y_la_visibilidad_no_se_llama_licencia():
    publicado = json.loads(generar.RUTA_PUBLICADO.read_text(encoding="utf-8"))
    _, protocolo = _entradas()
    assert sum(c["sellado"] for c in publicado["conjuntos"]) == sum(d["sellado"] for d in protocolo["datasets"])
    assert all("licencia" not in c for c in publicado["conjuntos"])
    assert len(publicado["lo_que_no_dicen"]) == len(generar.LO_QUE_NO_DICEN) >= 4
    assert all(set(x) == {"es", "en"} for x in publicado["lo_que_no_dicen"])


def test_una_cifra_retocada_en_el_ARTEFACTO_cambia_lo_publicado():
    """La otra mitad: probar el artefacto no es probar el código que lo produce. Se compone
    uno NUEVO con el mismo código, desde un artefacto con una métrica cambiada."""
    artefacto, protocolo = _entradas()
    original = generar.serializar(generar.componer(artefacto, protocolo))
    tocado = copy.deepcopy(artefacto)
    r = next(x for x in tocado["resultados"] if x["motor"] == "baseline" and x["estado"] == "completed")
    r[r["metrica_de_cierre"]] = (r[r["metrica_de_cierre"]] or 0.0) + 0.123
    assert generar.serializar(generar.componer(tocado, protocolo)) != original


def test_si_las_medias_no_reproducen_las_distancias_de_la_pasada_NO_se_genera():
    """La media es lo único que se calcula aquí; si divergiera de la cuenta de la pasada,
    se publicaría un segundo número. Una métrica de un motor que COMPITE, cambiada en el
    artefacto sin que su distancia cambie, tiene que parar el generador."""
    artefacto, protocolo = _entradas()
    tocado = copy.deepcopy(artefacto)
    r = next(x for x in tocado["resultados"] if x["motor"] == "lightgbm" and x["estado"] == "completed")
    r[r["metrica_de_cierre"]] += 0.05
    with pytest.raises(generar.DatosQueNoCuadran):
        generar.componer(tocado, protocolo)


def test_una_pasada_a_medias_no_se_publica():
    artefacto, protocolo = _entradas()
    tocado = dict(artefacto, parcial=True)
    with pytest.raises(generar.DatosQueNoCuadran):
        generar.componer(tocado, protocolo)
