# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El `file_id` de OpenML: lo que hace falta para que el protocolo registrado
sirva para VOLVER A BAJAR los datos, no solo para comprobarlos después.

**EL HALLAZGO, Y LO QUE DE ÉL RESULTÓ SER FALSO (medido el 2026-09-14).**

Anotado: «`file_id` no está ni en `protocolo_exploratorio.json` ni entre las
claves de `seleccion_40_final.json` — solo viaja dentro de la cadena `url`; si
OpenML cambia una URL, quien reproduzca la Fase 0 no puede saber qué fichero
exacto se usó».

Medido, en las dos direcciones:

* **Cierto, y peor de lo dicho, en el protocolo**: `protocolo_exploratorio.json`
  no trae `file_id` **ni `url` ni ninguna cadena `http`** — cero ocurrencias de
  las tres. Y `seleccion_40_final.json`/`hashes.json`, de donde se decía que el
  `file_id` «se puede parsear», **no están en el repositorio**: viven en
  `/home/deployer/fase0_openml_datos/`. Quien clone el repo no tiene ninguna
  URL. De los 40, **31 tienen `file_id` distinto de su `data_id`**: suponer que
  coinciden acierta en 9.
* **Falso, en la conclusión**: con `data_id` SÍ se llega hoy al fichero exacto.
  Consultados los 40 contra `GET /api/v1/json/data/{data_id}` el 2026-09-14:
  **40/40** devuelven el mismo `file_id` que la `url` guardada, la misma
  `version`, el mismo nombre y `status: active`; y su `md5_checksum` coincide
  con el md5 de los 40 ARFF locales, que a su vez cuadran con los 40
  `sha256_arff` sellados. «No se puede volver a bajar los datos» es falso hoy.

Lo que queda en pie, que no es poco: la reproducción depende de una
**indirección viva**. Con `file_id` el ARFF se pide por URL directa; sin él
hay que preguntarle al catálogo, y que el mapa `data_id -> file_id` siga
siendo el mismo dentro de cinco años no lo garantiza nadie. El `sha256_arff`
detecta que dejó de serlo — detectarlo no es repararlo.

**POR QUÉ AQUÍ NO SE RE-FIRMA NADA.** Meter el campo cambia el
`digest_sha256` del protocolo, y ese digest está citado dentro de evidencia ya
commiteada (`pasada_exploratoria_101_c3_conforme_20260914.json` cita
`eb54f42166835ad1…`). Re-firmar es escribir un protocolo NUEVO: es decisión de
Roberto, y hay otras correcciones esperando la misma re-firma. Estas pruebas
cubren **el código que lo haría**, y la que exige el campo en el fichero
registrado ESTUVO en `skip` hasta la re-firma del 2026-09-14; ya corre.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
_FASE0 = _RAIZ / "benchmarks" / "fase0"
if str(_FASE0) not in sys.path:
    sys.path.insert(0, str(_FASE0))

from benchmarks.fase0.file_id_para_la_re_firma import (  # noqa: E402
    FileIdNoDeducible, file_id_desde_url, file_ids_desde_seleccion,
    protocolo_con_file_id, url_de_descarga)
from benchmarks.fase0.protocolo import ProtocoloError, ProtocoloExploratorio  # noqa: E402

RUTA_PROTOCOLO = _FASE0 / "protocolo_exploratorio.json"
RUTA_SELECCION = Path("/home/deployer/fase0_openml_datos/seleccion_40_final.json")

con_la_seleccion_local = pytest.mark.skipif(
    not RUTA_SELECCION.exists(),
    reason=f"{RUTA_SELECCION} no está en esta máquina (no va en el repositorio)")


def _payload() -> dict:
    return json.loads(RUTA_PROTOCOLO.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# La función que compone el campo
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url, esperado", [
    ("https://openml.org/data/v1/download/1595261/adult.arff", 1595261),
    ("https://www.openml.org/data/v1/download/52350/breast-w.arff", 52350),
    ("https://api.openml.org/data/v1/download/3/kr-vs-kp.arff", 3),
    ("http://openml.org/data/v1/download/22044770/okcupid-stem.arff", 22044770),
])
def test_el_file_id_va_DENTRO_de_la_url_de_descarga(url, esperado):
    """El número del medio de la URL es el `file_id`, no el `data_id`:
    `adult` es `data_id` 1590 y `file_id` 1595261."""
    assert file_id_desde_url(url) == esperado


@pytest.mark.parametrize("url", [
    "https://openml.org/api/v1/json/data/1590",       # metadatos, no descarga
    "https://ejemplo.invalido/data/v1/download/7/x.arff",
    "openml.org/data/v1/download/7/x.arff",
    "",
    None,
    1595261,
])
def test_una_url_que_NO_reconozco_levanta_en_vez_de_callar(url):
    """Estricta a propósito. Devolver `None` ante una URL rara convertiría «no
    entiendo este dato» en «este dataset no tiene fichero», y ese `None` se
    colaría dentro del sello con pinta de medida."""
    with pytest.raises(FileIdNoDeducible):
        file_id_desde_url(url)


def test_la_url_se_recompone_desde_el_file_id_y_vuelve_a_dar_el_mismo():
    """Las dos mitades: extraer y componer. Componer es lo que hace útil el
    campo — con él, bajar el ARFF exacto no necesita preguntarle nada a la API
    de OpenML."""
    url = url_de_descarga(1595261, "adult")
    assert url == "https://openml.org/data/v1/download/1595261/adult.arff"
    assert file_id_desde_url(url) == 1595261


@pytest.mark.parametrize("file_id", [0, -3, None, "1595261", True])
def test_componer_una_url_con_un_file_id_que_no_lo_es_levanta(file_id):
    with pytest.raises(FileIdNoDeducible):
        url_de_descarga(file_id, "adult")


# --------------------------------------------------------------------------
# El hueco, medido sobre los 40 de verdad
# --------------------------------------------------------------------------

def test_el_protocolo_registrado_TRAE_el_file_id_y_SIGUE_sin_url():
    """CERRADO EL 2026-09-14 con la re-firma, y esta prueba es la de antes con
    el signo cambiado.

    Decía que en el fichero sellado no había `file_id` **ni ninguna `url`** —
    era el hallazgo, y era más gordo de lo anotado: quien clonara el
    repositorio no tenía de dónde sacarlo. Ahora los 40 lo traen.

    **Y sigue sin `url`, a propósito**: `url_de_descarga(file_id, nombre)` la
    compone, y guardar las dos sería la misma decisión escrita en dos sitios.
    Por eso la segunda mitad del aserto NO se invierte: se conserva tal cual."""
    crudo = RUTA_PROTOCOLO.read_text(encoding="utf-8")
    assert "file_id" in crudo
    assert "http" not in crudo, "el protocolo no guarda urls: se componen del file_id"
    claves = {k for d in _payload()["datasets"] for k in d}
    assert "file_id" in claves and "url" not in claves


@con_la_seleccion_local
def test_los_CUARENTA_recuperan_su_file_id_de_la_seleccion_local():
    """40/40 se recuperan parseando la `url`... en esta máquina. El fichero no
    va en el repositorio, así que esto mide que la función sirve, no que el
    hueco esté cerrado."""
    file_ids = file_ids_desde_seleccion(RUTA_SELECCION)
    data_ids = [d["data_id"] for d in _payload()["datasets"]]
    assert len(data_ids) == 40
    assert set(file_ids) == set(data_ids)
    assert all(isinstance(v, int) and v > 0 for v in file_ids.values())


@con_la_seleccion_local
def test_el_file_id_NO_se_puede_deducir_del_data_id():
    """31 de 40 son distintos. Si alguien «ahorra» el campo poniendo
    `file_id = data_id`, se lleva 31 URLs que no existen — y las 9 que sí,
    reforzando la idea equivocada."""
    file_ids = file_ids_desde_seleccion(RUTA_SELECCION)
    distintos = [i for i, f in file_ids.items() if i != f]
    assert len(distintos) == 31, (
        f"medido el 2026-09-14: 31 de 40. Ahora salen {len(distintos)}: "
        "o cambió la selección, o cambió OpenML")


# --------------------------------------------------------------------------
# El payload re-firmado: que el campo entre, y que entre DENTRO del sello
# --------------------------------------------------------------------------

def test_protocolo_con_file_id_NO_muta_lo_que_recibe():
    """Devuelve un objeto nuevo. Una función que edita su argumento no deja
    comparar el antes con el después, que es justo lo que hay que mirar antes
    de re-firmar."""
    payload = _payload()
    antes = json.dumps(payload, sort_keys=True)
    nuevo = protocolo_con_file_id(payload, {d["data_id"]: 7 for d in payload["datasets"]})
    assert json.dumps(payload, sort_keys=True) == antes
    assert all(d["file_id"] == 7 for d in nuevo["datasets"])


def test_si_falta_el_file_id_de_UNO_no_se_compone_nada():
    """Media limpieza es peor que ninguna: un protocolo con 39 de 40 promete
    que los 40 se bajan por URL directa y cumple con 39."""
    payload = _payload()
    file_ids = {d["data_id"]: 7 for d in payload["datasets"]}
    file_ids.pop(next(iter(file_ids)))
    with pytest.raises(FileIdNoDeducible):
        protocolo_con_file_id(payload, file_ids)


def test_el_campo_ENTRA_EN_EL_SELLO_las_dos_mitades():
    """Las dos mitades, porque cada una sola se pasa con un fallo distinto.

    * Sin `file_id`, el digest tiene que ser EXACTAMENTE el de hoy: si el
      cambio moviera el digest del protocolo ya registrado, habría roto una
      evidencia commiteada sin que nadie lo pidiera.
    * Con `file_id`, el digest tiene que CAMBIAR: si no cambiara, el campo
      estaría fuera del sello y se podría reeditar en silencio — el mismo
      defecto que el `alcance` antes del 2026-09-12.
    """
    payload = _payload()
    digest_de_hoy = ProtocoloExploratorio.desde_json(payload).digest()
    assert digest_de_hoy == payload["digest_sha256"], (
        "el protocolo registrado ya no cuadra con su propio digest_sha256")

    con_campo = protocolo_con_file_id(
        payload, {d["data_id"]: 1000 + i for i, d in enumerate(payload["datasets"])})
    digest_nuevo = ProtocoloExploratorio.desde_json(con_campo).digest()
    assert digest_nuevo != digest_de_hoy, (
        "añadir file_id NO mueve el digest: el campo quedaría FUERA del sello")


def test_un_protocolo_con_el_file_id_A_MEDIAS_se_RECHAZA():
    """Todos o ninguno. La tolerancia de `desde_json` existe para poder cargar
    el protocolo de antes de la re-firma, no para permitir un catálogo con el
    campo en parte de las filas."""
    payload = protocolo_con_file_id(
        _payload(), {d["data_id"]: 1000 + i
                     for i, d in enumerate(_payload()["datasets"])})
    payload["datasets"][3] = {k: v for k, v in payload["datasets"][3].items()
                              if k != "file_id"}
    with pytest.raises(ProtocoloError):
        ProtocoloExploratorio.desde_json(payload)


@pytest.mark.parametrize("valor", [0, -1, "no-soy-un-numero", 3.5])
def test_un_file_id_que_no_es_un_entero_positivo_se_RECHAZA(valor):
    payload = protocolo_con_file_id(
        _payload(), {d["data_id"]: 1000 + i
                     for i, d in enumerate(_payload()["datasets"])})
    payload["datasets"][0]["file_id"] = valor
    with pytest.raises((ProtocoloError, ValueError, TypeError)):
        ProtocoloExploratorio.desde_json(payload)


def test_un_file_id_que_llega_como_CADENA_de_digitos_se_convierte():
    """Y este NO se rechaza, a propósito: **el aserto que lo exigía estaba mal
    él**, no el producto. La API de OpenML devuelve `file_id` como cadena
    (`"1595261"`), así que quien componga el campo desde la API trae cadenas;
    `desde_json` las convierte con `int()`, que además revienta si no son
    dígitos (el caso de arriba). Rechazarlas obligaría a que cada llamante
    convirtiera por su cuenta — y el que se olvidara metería una cadena dentro
    del sello, que canonicaliza distinto que el número."""
    payload = protocolo_con_file_id(
        _payload(), {d["data_id"]: 1000 + i
                     for i, d in enumerate(_payload()["datasets"])})
    payload["datasets"][0]["file_id"] = "1595261"
    cargado = ProtocoloExploratorio.desde_json(payload)
    assert cargado.datasets[0].file_id == 1595261
    assert cargado.a_json()["datasets"][0]["file_id"] == 1595261


# --------------------------------------------------------------------------
# La prueba que EXIGE el campo, y el aviso de que hoy no aplica
# --------------------------------------------------------------------------

# EL GUARDIÁN DEL `skip`, BORRADO EL 2026-09-14 — y funcionó.
#
# Vivía aquí `test_HOY_el_protocolo_registrado_sigue_SIN_file_id_y_por_eso_hay
# _un_skip`: existía para ponerse ROJA el día de la re-firma y decir qué había
# que hacer, porque **un `skip` que caduca en silencio es peor que no ponerlo**
# — su motivo sigue sonando razonable mucho después de dejar de ser cierto.
#
# Se re-firmó, se puso roja, y su mensaje llevaba los tres pasos: quitar el
# `skip` de abajo, borrarla, y actualizar `DIGEST_REGISTRADO_ANTES_DE_MEDIR`.
# Los tres hechos. Queda su epitafio para que se vea que el mecanismo sirvió,
# y para el siguiente que necesite aparcar algo con fecha de caducidad.

def test_el_protocolo_registrado_trae_el_file_id_de_los_40():
    """Lo que un protocolo reproducible tiene que decir: con qué FICHERO
    exacto se midió, sin depender de que un catálogo vivo siga contestando lo
    mismo."""
    datasets = _payload()["datasets"]
    assert len(datasets) == 40
    for entrada in datasets:
        file_id = entrada.get("file_id")
        assert isinstance(file_id, int) and file_id > 0, entrada["nombre"]
        assert url_de_descarga(file_id, entrada["nombre"]).endswith(
            f"/{file_id}/{entrada['nombre']}.arff")


# --------------------------------------------------------------------------
# El generador: que un protocolo hecho HOY sí lo lleve
# --------------------------------------------------------------------------

def test_el_generador_ya_CABLEA_el_file_id_que_la_API_le_daba():
    """El hueco estaba en el cableado, no en el API. `_candidato` se traía
    `info["url"]` —`descargar_y_hashear` descarga POR ESA URL— y
    `construir_protocolo` se quedaba solo con el `data_id`. El dato llegaba y
    el llamante no lo usaba, que es la forma catorce veces repetida de este
    defecto. Un protocolo generado hoy sí lo lleva; el registrado el
    2026-09-06 no, y añadírselo es re-firmarlo."""
    from benchmarks.fase0.generar_protocolo import construir_protocolo

    seleccion = [dict(data_id=1590, nombre="adult", version=2, objetivo="class",
                      url="https://openml.org/data/v1/download/1595261/adult.arff",
                      n_filas=48842, n_columnas=15, bucket="mediano",
                      tiene_faltantes=True, max_cardinalidad_nominal=42,
                      desbalanceado=False, solo_numericas=False, licencia="Public",
                      sellado=False, binaria=True, multiclase=False)]
    protocolo = construir_protocolo(seleccion, {1590: "b" * 64})
    assert protocolo.datasets[0].file_id == 1595261
    assert protocolo.a_json()["datasets"][0]["file_id"] == 1595261
