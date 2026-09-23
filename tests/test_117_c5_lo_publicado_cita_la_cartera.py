# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""117-C5 — QUE NO CADUQUE: lo publicado cita el artefacto que sostiene la cartera.

**DECIDIDO por Roberto el 2026-09-23 (noche)**: la página «Cómo medimos» y las fichas de
«ejemplos medidos» enseñan la pasada que SOSTIENE la cartera de motores vigente —hoy la
AMPLIA (`pasada_amplia_101_c5_resultado.json`, digest `3646af61...`)—, no la última medida.
Cuando una pasada nueva se selle y pase a sostener la cartera, lo que tiene que cambiar es
`benchmarks/datos_publicos/generar.py` (sus rutas `RUTA_ARTEFACTO_PRINCIPAL` /
`RUTA_PROTOCOLO_PRINCIPAL`) y regenerarse `fase0_publico.json` y sus copias — y ESTE fichero
es el que lo vigila: falla si el bloque `principal` de lo publicado (el original en
`matrixAI`, y cada COPIA que vive en `matrixaistudio`) cita un `digest_resultados_crudos`
que no es el `evidencia_digest` que `matrixai_engines.cartera.CARTERA_APROBADA` sella para
sus motores aprobados.

**Por qué «el digest que aprueba la cartera» y no «el digest de la pasada amplia» escrito
aquí a mano.** Escribir el digest de la amplia en este fichero sería un TERCER sitio
declarando lo mismo que `generar.py` y que `cartera.py` — y el día que la cartera se mueva a
una pasada nueva, este fichero seguiría comparando contra la vieja sin que nada lo avisara.
Se lee de `CARTERA_APROBADA` en vivo, igual que hace `test_c102_c1_evidencia_de_la_cartera.
py` para lo que la cartera APRUEBA.

**Qué NO vigila.** El bloque `ultima_medicion` de `fase0_publico.json` NO tiene que citar el
digest de la cartera — al contrario, `test_117_c1_datos_publicos.py` exige que sea el de la
v2 y que NO sea anclable entera. Confundir los dos bloques sería el defecto que 117-C5 vino
a evitar, no algo que este fichero deba tolerar en ningún sentido.

QUÉ LO PONE ROJO: que el bloque `principal` (en el original o en cualquier copia) cite un
digest que la cartera no usa para aprobar — por ejemplo, si `generar.py` apuntara
`RUTA_ARTEFACTO_PRINCIPAL` a la v2 (sabotaje comprobado, ver 117-C5 en la bitácora/informe).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from matrixai_engines.cartera import CARTERA_APROBADA

RAIZ = Path(__file__).resolve().parents[1]  # .../matrixAI
CASA = RAIZ.parent
STUDIO = CASA / "matrixaistudio"

#: El original y cada COPIA publicada del JSON de 117-C1 (117-C3, 117-C2 lo leen de aquí).
_COPIAS_DE_FASE0_PUBLICO = (
    RAIZ / "benchmarks" / "datos_publicos" / "fase0_publico.json",
    STUDIO / "frontend" / "src" / "datos" / "fase0_publico.json",
    STUDIO / "studio-backend" / "matrixai_studio" / "ejemplos_medidos" / "fase0_publico.json",
)

#: El artefacto CRUDO descargable de la web (no el JSON compuesto de C1: el que C1 LEE).
#: Mismo invariante con otra forma de fichero -- ver `frontend/scripts/copiar-datos-de-
#: fase0.mjs`, que es la única vía por la que este fichero entra en `matrixaistudio`.
_ARTEFACTO_CRUDO_DESCARGABLE = (
    STUDIO / "frontend" / "public" / "fase0" / "pasada_amplia_101_c5_resultado.json")


def _digest_que_aprueba_la_cartera() -> str:
    """El digest que las entradas APROBADAS sellan como evidencia. Si la cartera está
    vacía (decisión pendiente) no hay nada que 117-C5 pueda comparar -- se salta, no se
    inventa un digest. Si dos entradas aprobadas citaran digests distintos,
    `test_c102_c1_evidencia_de_la_cartera.py` (que exige que las citen todas MISMA pasada)
    ya lo habría puesto rojo antes que este fichero; aquí se re-comprueba por las dos
    direcciones, no se asume."""
    if not CARTERA_APROBADA:
        pytest.skip("la cartera esta vacia (decision pendiente): nada que 117-C5 pueda "
                    "comparar hasta que Roberto vuelva a sellar")
    digests = {e.evidencia_digest for e in CARTERA_APROBADA}
    assert len(digests) == 1, (
        f"las entradas aprobadas de la cartera citan digests DISTINTOS {digests}: 117-C5 no "
        "puede decidir cual es «el» artefacto que la sostiene hasta que eso se resuelva "
        "(ver test_c102_c1_evidencia_de_la_cartera.py, que exige que las tres/las que haya "
        "citen la MISMA pasada)")
    return next(iter(digests))


def test_el_digest_que_aprueba_la_cartera_es_el_de_la_pasada_AMPLIA():
    """Control del propio instrumento (medir, no suponer): a fecha de este corte la cartera
    tiene que estar citando la AMPLIA (`3646af61...`), que es justo la premisa de la
    decisión de Roberto del 2026-09-23. Si esto cambiara sin que el resto del fichero se
    tocara, sería la señal de que 117-C5 necesita revisarse, no solo regenerarse."""
    assert _digest_que_aprueba_la_cartera() == (
        "3646af61b155c1321ad3fd71f25e9300ed9797b9b5dec0f37ed835cd835ecf48")


def test_el_bloque_PRINCIPAL_del_JSON_ORIGINAL_cita_el_digest_de_la_cartera():
    datos = json.loads(
        (RAIZ / "benchmarks" / "datos_publicos" / "fase0_publico.json").read_text(encoding="utf-8"))
    assert datos["principal"]["fuente"]["digest_resultados_crudos"] == _digest_que_aprueba_la_cartera(), (
        "el bloque `principal` de benchmarks/datos_publicos/fase0_publico.json cita un "
        "artefacto que la cartera NO usa para aprobar -- regenerar con "
        "`python3 benchmarks/datos_publicos/generar.py --escribir` tras corregir "
        "`RUTA_ARTEFACTO_PRINCIPAL`/`RUTA_PROTOCOLO_PRINCIPAL`")


@pytest.mark.parametrize("ruta", _COPIAS_DE_FASE0_PUBLICO, ids=lambda r: str(r.relative_to(CASA)))
def test_el_bloque_PRINCIPAL_de_CADA_COPIA_publicada_cita_el_digest_de_la_cartera(ruta):
    if not ruta.exists():
        pytest.fail(f"{ruta} no existe: 117-C2/C3 tenían que haberla copiado desde el "
                    "original de matrixAI (ver `copiar-datos-de-fase0.mjs` / "
                    "`generar_ejemplos_medidos.py`)")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert "principal" in datos, (
        f"{ruta} no tiene bloque `principal`: es una copia de antes de 117-C5 (un único "
        "bloque plano) -- regenerarla con el script correspondiente")
    assert datos["principal"]["fuente"]["digest_resultados_crudos"] == _digest_que_aprueba_la_cartera(), (
        f"{ruta}: el bloque `principal` cita un artefacto que la cartera NO usa para "
        "aprobar -- esta copia está desactualizada respecto al original de matrixAI")


def test_el_artefacto_crudo_descargable_de_la_web_es_el_que_aprueba_la_cartera():
    """La descarga de `frontend/public/fase0/` no es el JSON compuesto de C1: es el
    artefacto CRUDO que C1 lee. Mismo invariante, otra forma de fichero -- comprobado
    contra su propio `digest_resultados_crudos`, calculado por la propia pasada al
    escribirse (no por este test)."""
    if not _ARTEFACTO_CRUDO_DESCARGABLE.exists():
        pytest.fail(f"{_ARTEFACTO_CRUDO_DESCARGABLE} no existe: el descargable «principal» "
                    "de 117-C2 falta -- correr `node scripts/copiar-datos-de-fase0.mjs` en "
                    "matrixaistudio/frontend")
    crudo = json.loads(_ARTEFACTO_CRUDO_DESCARGABLE.read_text(encoding="utf-8"))
    assert crudo["digest_resultados_crudos"] == _digest_que_aprueba_la_cartera(), (
        f"{_ARTEFACTO_CRUDO_DESCARGABLE}: el artefacto crudo descargable NO es el que la "
        "cartera usa para aprobar")


def test_el_bloque_ULTIMA_MEDICION_no_tiene_que_citar_el_digest_de_la_cartera():
    """La otra mitad, para que quede escrita y no se confunda con la de arriba: el bloque
    `ultima_medicion` cita OTRO digest a propósito (la v2, no anclable entera) -- este test
    documenta que NO es un defecto, y que confundir los dos bloques es justo lo que 117-C5
    existe para impedir."""
    datos = json.loads(
        (RAIZ / "benchmarks" / "datos_publicos" / "fase0_publico.json").read_text(encoding="utf-8"))
    assert datos["ultima_medicion"]["fuente"]["digest_resultados_crudos"] != _digest_que_aprueba_la_cartera()
    assert datos["ultima_medicion"]["procedencia"]["anclable_entera"] is False
