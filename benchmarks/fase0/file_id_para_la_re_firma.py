# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El `file_id` de OpenML que el protocolo registrado NO trae, y el guion que
lo añadiría cuando Roberto autorice la re-firma.

**EL HUECO, MEDIDO el 2026-09-14 y NO el que estaba anotado.** La anotación
decía que el `file_id` «solo viaja dentro de la cadena `url`» de
`seleccion_40_final.json`. Medido:

* `protocolo_exploratorio.json` —lo ÚNICO sellado y commiteado— no trae
  `file_id` **ni `url` ni nada que empiece por `http`**: cero ocurrencias de
  las tres. Trae `data_id`, `version` y `sha256_arff`.
* `seleccion_40_final.json` y `hashes.json` **no están en el repositorio**:
  viven en `/home/deployer/fase0_openml_datos/`, en esta máquina. Quien
  reproduzca desde el repo no los tiene, así que la `url` de la que se decía
  «se puede parsear» tampoco está a su alcance.
* De los 40, **31 tienen `file_id` distinto de `data_id`** y solo 9 coinciden
  (los ids bajos, por casualidad histórica). O sea: inventarse
  `file_id = data_id` fallaría en 31 de 40.

**Y POR QUÉ EL HALLAZGO ESTABA EXAGERADO, también medido.** Con `data_id` SÍ
se llega hoy al fichero exacto: `GET /api/v1/json/data/{data_id}` devuelve
`file_id`, `url`, `version` y `md5_checksum`. Consultados los 40 el
2026-09-14: **40/40** devuelven el MISMO `file_id` que la `url` guardada, la
misma `version`, el mismo nombre y `status: active`, y el `md5_checksum` que
declaran coincide con el md5 de los 40 ARFF descargados —que a su vez cuadran
con los 40 `sha256_arff` sellados—. Así que «no se puede volver a bajar los
datos» es FALSO hoy: se puede, con una llamada de metadatos.

Lo que el `file_id` añade no es «poder conseguir el fichero», es **quitar una
indirección viva del camino**: con él, el ARFF se pide por URL directa sin
preguntarle nada al catálogo; sin él, la reproducción depende de que el mapa
`data_id -> file_id` que sirve OpenML hoy sea el mismo dentro de cinco años.
El `sha256_arff` detecta que dejó de serlo, pero detectarlo no es repararlo.

**POR QUÉ ESTE FICHERO NO RE-FIRMA NADA.** Añadir `file_id` al protocolo
cambia su `digest_sha256`, y ese digest está citado dentro de evidencia ya
commiteada (`pasada_exploratoria_101_c3_conforme_20260914.json` cita
`eb54f42166835ad1…`). Re-firmar es escribir un protocolo NUEVO, no reparar
uno; es decisión de Roberto y hay otras dos correcciones esperando la misma
re-firma. Aquí queda lo que hace falta para que re-firmar sea **ejecutar un
guion**:

    python3 benchmarks/fase0/file_id_para_la_re_firma.py            # simula
    python3 benchmarks/fase0/file_id_para_la_re_firma.py --escribir # re-firma

En modo simulación no escribe nada y dice qué `file_id` pondría, de dónde lo
sacó y **qué digest nuevo saldría**. Solo con `--escribir` toca el fichero.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

_AQUI = Path(__file__).resolve().parent
if str(_AQUI) not in sys.path:
    sys.path.insert(0, str(_AQUI))

from protocolo import ProtocoloExploratorio  # noqa: E402

#: La forma de la URL de descarga de OpenML. Las 40 de la selección tienen
#: EXACTAMENTE esta forma (medido: una sola forma tras normalizar los dígitos),
#: y el número del medio es el `file_id`, no el `data_id` — confundirlos
#: acertaría en 9 de 40.
_URL_DE_DESCARGA = re.compile(
    r"^https?://(?:www\.|api\.)?openml\.org/data/v1/download/(?P<file_id>\d+)(?:/|$)")

RUTA_PROTOCOLO = _AQUI / "protocolo_exploratorio.json"
RUTA_SELECCION = Path("/home/deployer/fase0_openml_datos/seleccion_40_final.json")


class FileIdNoDeducible(ValueError):
    """No se pudo averiguar el `file_id` de un dataset.

    Es un error y no un `None` a propósito: un protocolo con el campo a medias
    es peor que uno sin el campo, porque el que lo lea creerá que puede bajar
    los 40 por URL directa y podrá bajar 38. Media limpieza es peor que
    ninguna.
    """


def file_id_desde_url(url: str) -> int:
    """El `file_id` que lleva dentro una URL de descarga de OpenML.

    Estricta a propósito: cualquier otra forma levanta en vez de devolver
    `None`. Una URL que no reconozco no es «un dataset sin file_id», es un
    dato que no entiendo, y tragármelo en silencio es justo cómo se cuela un
    campo falso dentro de un sello.
    """
    if not isinstance(url, str):
        raise FileIdNoDeducible(f"la url no es una cadena: {url!r}")
    casa = _URL_DE_DESCARGA.match(url.strip())
    if casa is None:
        raise FileIdNoDeducible(
            f"la url no tiene la forma de descarga de OpenML: {url!r}")
    return int(casa.group("file_id"))


def url_de_descarga(file_id: int, nombre: str) -> str:
    """La URL directa al ARFF exacto, compuesta a partir del `file_id`.

    Es la mitad que justifica el campo: con esto, reproducir la Fase 0 no
    necesita preguntarle a la API de OpenML qué fichero correspondía.
    """
    if not isinstance(file_id, int) or isinstance(file_id, bool) or file_id <= 0:
        raise FileIdNoDeducible(f"file_id no es un entero positivo: {file_id!r}")
    if not nombre:
        raise FileIdNoDeducible("un ARFF sin nombre no compone una url")
    return f"https://openml.org/data/v1/download/{file_id}/{nombre}.arff"


def file_ids_desde_seleccion(ruta: str | Path = RUTA_SELECCION) -> dict[int, int]:
    """`data_id -> file_id` leyendo la selección local, parseando su `url`.

    Fuente barata y sin red, pero **no está en el repositorio**: si no existe,
    el que reproduzca tiene que ir a OpenML (`file_ids_desde_openml`).
    """
    payload = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {int(r["data_id"]): file_id_desde_url(r["url"]) for r in payload}


def file_ids_desde_openml(data_ids: Iterable[int], *,
                          espera_s: float = 0.4) -> dict[int, int]:
    """`data_id -> file_id` preguntándole a OpenML. Solo METADATOS: un JSON de
    un par de KB por dataset, nunca el ARFF."""
    import time

    salida: dict[int, int] = {}
    for data_id in data_ids:
        peticion = urllib.request.Request(
            f"https://www.openml.org/api/v1/json/data/{int(data_id)}",
            headers={"User-Agent": "matrixai-fase0/1.0"})
        with urllib.request.urlopen(peticion, timeout=30) as respuesta:
            info = json.loads(respuesta.read())["data_set_description"]
        salida[int(data_id)] = int(info["file_id"])
        time.sleep(espera_s)
    return salida


def protocolo_con_file_id(payload: Mapping[str, Any],
                          file_ids: Mapping[int, int]) -> dict[str, Any]:
    """El protocolo con `file_id` en CADA dataset — un objeto NUEVO.

    No muta lo que recibe: el que llama tiene que poder comparar el antes y el
    después, y una función que edita su argumento no deja comparar nada.

    Levanta si falta el `file_id` de alguno: ver `FileIdNoDeducible`.
    """
    datasets = payload.get("datasets")
    if not datasets:
        raise FileIdNoDeducible("el payload no trae datasets")
    nuevos = []
    for entrada in datasets:
        data_id = int(entrada["data_id"])
        if data_id not in file_ids:
            raise FileIdNoDeducible(
                f"no hay file_id para data_id={data_id} ({entrada.get('nombre')}): "
                "un protocolo con el campo a medias es peor que sin el campo")
        file_id = int(file_ids[data_id])
        if file_id <= 0:
            raise FileIdNoDeducible(f"file_id no positivo para data_id={data_id}: {file_id!r}")
        nuevos.append({**entrada, "file_id": file_id})
    return {**payload, "datasets": nuevos}


def _digest_del_payload(payload: Mapping[str, Any]) -> str:
    """El digest que tendría ese payload, calculado **con el mismo código que
    corre de verdad** (`ProtocoloExploratorio`), no re-implementando la
    canonicalización aquí: probar el artefacto no es probar el código que lo
    produce, y dos canonicalizadores acaban divergiendo."""
    return ProtocoloExploratorio.desde_json(dict(payload)).digest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocolo", default=str(RUTA_PROTOCOLO))
    ap.add_argument("--seleccion", default=str(RUTA_SELECCION),
                    help="selección local de la que sacar el file_id parseando su url")
    ap.add_argument("--desde-openml", action="store_true",
                    help="pedir los file_id a la API de OpenML (solo metadatos)")
    ap.add_argument("--escribir", action="store_true",
                    help="RE-FIRMA: sobreescribe el protocolo. Sin esto, solo simula.")
    args = ap.parse_args(argv)

    ruta = Path(args.protocolo)
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    data_ids = [int(d["data_id"]) for d in payload["datasets"]]

    fuentes: dict[str, dict[int, int]] = {}
    if Path(args.seleccion).exists():
        fuentes["seleccion local"] = file_ids_desde_seleccion(args.seleccion)
    if args.desde_openml or not fuentes:
        fuentes["API de OpenML"] = file_ids_desde_openml(data_ids)

    # Si hay dos fuentes se CONTRASTAN, no se coge una y se supone la otra.
    nombres = list(fuentes)
    if len(nombres) > 1:
        a, b = fuentes[nombres[0]], fuentes[nombres[1]]
        discrepan = [i for i in data_ids if a.get(i) != b.get(i)]
        print(f"contraste {nombres[0]} vs {nombres[1]}: "
              f"{len(data_ids) - len(discrepan)}/{len(data_ids)} coinciden")
        if discrepan:
            print("  DISCREPAN:", [(i, a.get(i), b.get(i)) for i in discrepan])
            print("  no se re-firma con dos fuentes que se contradicen")
            return 2
    file_ids = fuentes[nombres[0]]

    nuevo = protocolo_con_file_id(payload, file_ids)
    digest_viejo = _digest_del_payload(payload)
    digest_nuevo = _digest_del_payload(nuevo)
    iguales = sum(1 for i in data_ids if file_ids[i] == i)
    print(f"file_id para {len(data_ids)} datasets, fuente: {nombres[0]}")
    print(f"  coinciden con su data_id: {iguales}; distintos: {len(data_ids) - iguales}")
    print(f"  digest actual: {digest_viejo}")
    print(f"  digest tras la re-firma: {digest_nuevo}")
    if digest_nuevo == digest_viejo:
        print("  EL DIGEST NO SE MUEVE: el campo estaría FUERA del sello. "
              "Revisar `DatasetRegistrado.a_json` antes de seguir.")
        return 3
    if not args.escribir:
        print("simulación: no se ha escrito nada. `--escribir` re-firma de verdad.")
        return 0
    nuevo = {**nuevo, "digest_sha256": digest_nuevo}
    ruta.write_text(json.dumps(nuevo, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"RE-FIRMADO. Falta cambiar a mano, y con su commit explicando por qué:")
    print(f"  tests/test_c101_c1_protocolo_fase0.py :: DIGEST_REGISTRADO_ANTES_DE_MEDIR")
    print(f"  -> {digest_nuevo}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
