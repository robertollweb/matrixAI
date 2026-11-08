# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El ORIGEN DEL FICHERO en el recibo y en el manifiesto de reproducción — contrato 150-C4.

POR QUÉ EXISTE. Un Excel entra al producto convertido a un CSV canónico (`matrixai.tablas`, 150-C1), y un CSV
`;` con decimal español entra con la coma pasada a punto (150-C2). En los dos casos **lo que se hashea y lo que
el modelo vio es el CSV derivado, no el fichero que la persona subió**. Sin este bloque, el recibo enseñaba la
huella de un CSV que nadie ha visto y no decía de qué libro salió, con qué hoja ni qué se hizo por el camino.

QUÉ ES `origen_del_fichero`. El `origen` que declara la conversión (`TablaConvertida.origen`: formato, nombre,
sha256 del ORIGINAL, hoja, rango, filas excluidas, conversor y declaraciones), tal cual, más —si la entrada
canónica del estudio convirtió algo al entrar— sus `declaraciones_de_entrada`. Para un CSV que solo se
convirtió al entrar (decimal español) se compone un origen con lo que el estudio sabe: la huella del texto tal
como llegó, su tamaño y lo que se hizo con él.

TRES REGLAS (las que el contrato fija y este módulo SOSTIENE, cada una con su prueba):

1. **El hash que se verifica sigue siendo el del CSV derivado.** `artifacts.dataset.sha256` es el del CSV que
   entró al entrenamiento; la huella del original es INFORMATIVA y no entra en ninguna comprobación (R1–R3 del
   verificador): no se puede prometer reproducir desde el Excel con otra versión de la biblioteca. El bloque
   del manifiesto lo dice él mismo (`informativo`, `nota`), y la línea del recibo también.
2. **Lo que no hay no se inventa.** Un CSV sin conversión ni declaración NO lleva `origen_del_fichero`: ni
   vacío, ni con la huella de sí mismo. La clave se OMITE.
3. **Forma cerrada.** Una captura del run que lo lleva es de la versión `1.3`; un núcleo que no la conoce la
   rechaza en vez de publicarlo como «parámetro de generación» (que es lo que hace con una clave que no tiene
   dada de alta en `_RUN_PROVENANCE_CORE_KEYS`).

Y TRES MÁS, de la reparación tras la auditoría (150-C4, I1–I3 y M1):

4. **El origen está ATADO al CSV — con lo que de verdad se comprueba, y nada más.** La conversión declara la huella
   del CSV que devuelve (`sha256_del_csv`); el estudio la compara con el texto que recibe y RECHAZA un origen cuya
   huella no es la de ese CSV (`origen_del_fichero_no_es_de_este_csv`): sin eso, cualquiera podía adjuntar a su CSV el
   origen de un libro ajeno y la ficha lo afirmaba. Donde el CSV puede llegar preparado (`train-start`), no se
   rechaza: se graba `vinculo` y la ficha lo dice. **Pero la huella la declara el propio CLIENTE** (el servidor no
   guarda lo que devolvió la conversión ni la firma), así que `vinculo: "comprobado"` significa SOLO «el dataset
   coincide con la huella que el origen declara para su CSV», y la ficha lo dice con esas palabras: nunca «es el CSV
   que devolvió la conversión» (decisión del supervisor, remate de 150-C4). Un `vinculo: "comprobado"` sin
   `sha256_del_csv` es imposible y el validador lo rechaza.
5. **Lo que la ficha cita no puede reescribirla.** `formato` es una lista cerrada; `nombre` y `hoja` van en código
   en línea con un delimitador más largo que cualquier racha de comillas invertidas que lleven (una hoja
   «Datos `2024`» sale entera), y se rechazan los caracteres de control y de dirección (`matrixai.nombre_citable`:
   C0, C1, U+2028/2029 y U+200E, U+200F, U+061C, U+202A–U+202E, U+2066–U+2069), que reordenan lo que se lee sin
   cambiar lo que se guarda. La conversión de un libro QUITA esos mismos caracteres (y lo declara) con la misma regla.
6. **Lo que dice el servidor no lo dice el cliente.** `informativo`, `nota` y `vinculo` los escribe el núcleo: lo
   que traiga el cliente con esos nombres se descarta al entrar y, en el manifiesto, lo del núcleo va DESPUÉS.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from matrixai import nombre_citable
from matrixai.export.reproduce import (
    ReproduceManifestError,
    _require_full_sha256,
    _require_text,
    canonical_json,
)

__all__ = [
    "CLAVE", "VERSION_DE_LA_CAPTURA_CON_ORIGEN", "TOPE_DE_BYTES_DEL_ORIGEN", "FORMATOS", "VINCULOS",
    "OrigenDeOtroCsv", "normalizar_origen_del_fichero", "numero_de_declaraciones", "origen_de_un_csv",
    "es_origen_de_excel", "entrada_canonica_de",
    "origen_para_el_estudio", "atar_al_csv", "huella_del_texto", "bloque_del_manifiesto", "texto_del_origen",
    "codigo_en_linea", "ROTULO", "frases_del_origen", "frases_bilingues_del_origen",
]

#: El nombre de la clave: en la captura del run, en el sobre del estudio y en `artifacts.dataset`.
CLAVE = "origen_del_fichero"

#: La versión de la captura del run (`run_provenance.schema_version`) que puede llevarlo. Una captura que no lo lleva
#: sigue siendo `1.2`, con los mismos bytes de siempre: subir la versión de TODAS las capturas habría cambiado el
#: digest de cada paquete que no tiene nada que ver con un Excel.
VERSION_DE_LA_CAPTURA_CON_ORIGEN = "1.3"

#: Lo máximo que pesa el bloque (JSON canónico). Una hoja con miles de columnas renombradas declara miles de
#: cosas; sin tope, el recibo y la captura de un modelo guardado crecerían sin límite por lo que traiga un fichero.
TOPE_DE_BYTES_DEL_ORIGEN = 1_048_576

#: La versión del esquema de `origen` que ESTE núcleo sabe leer (`matrixai.tablas.ESQUEMA_DE_ORIGEN`). No se importa de
#: allí: `matrixai.tablas` no tiene por qué cargarse para escribir un recibo.
_ESQUEMA_QUE_SE_SABE_LEER = 1

_FORMATOS_DE_EXCEL = ("xlsx", "xls", "xlsm")

#: Los formatos que un origen puede declarar: los que de verdad produce `matrixai.tablas.convertir`
#: (`Libro._clasificar`: «xls» por firma OLE, «xlsx» o «xlsm» —con macros— por el ZIP) y el «csv» que compone
#: `origen_de_un_csv` (150-C2). Lista CERRADA (I3): antes era texto libre y la ficha imprimía, en mayúsculas,
#: lo que mandase el cliente («**VERIFICADO**»).
FORMATOS = (*_FORMATOS_DE_EXCEL, "csv")

#: Lo que `vinculo` puede decir (I1): «comprobado» = el dataset coincide, byte a byte, con la huella que el origen declara
#: para su CSV (`sha256_del_csv`, que la declara el cliente: el servidor no guarda lo que devolvió la conversión);
#: «no_comprobado» = no coincide o no se pudo saber (el CSV pudo prepararse después).
VINCULOS = ("comprobado", "no_comprobado")

#: Las claves que escribe el NÚCLEO, nunca el cliente (M1): lo que llegue con estos nombres se descarta al entrar.
_CLAVES_DEL_NUCLEO = ("informativo", "nota", "vinculo")

#: Los caracteres que un nombre o una hoja citados no pueden llevar viven en `matrixai.nombre_citable` (UNA regla, que
#: también aplica la conversión de un libro al componer el origen). Un U+202E en el nombre de una hoja hace que la ficha
#: SE LEA distinta de lo que guarda («xslx.exe» como «exe.xlsx»).
_CONTROL = nombre_citable.CONTROL
_BIDI = nombre_citable.DIRECCION

#: Cómo se llama la línea en cada idioma (la ficha y el expediente la rotulan igual: los dos leen de aquí).
ROTULO = {"es": "Origen", "en": "Origin"}


def _lista_de_declaraciones(valor: Any, que: str) -> list[dict[str, Any]]:
    if not isinstance(valor, list):
        raise ReproduceManifestError(f"{que} must be a list, got {valor!r}")
    for i, d in enumerate(valor):
        if not isinstance(d, dict):
            raise ReproduceManifestError(f"{que}[{i}] must be an object, got {d!r}")
        _require_text(d.get("tipo"), f"{que}[{i}].tipo")
        cuantas = d.get("cuantas")
        if cuantas is not None and (type(cuantas) is not int or cuantas < 0):
            raise ReproduceManifestError(f"{que}[{i}].cuantas must be a non-negative integer or null, got {cuantas!r}")
    return valor


def normalizar_origen_del_fichero(crudo: Any, *, campo: str = CLAVE) -> dict[str, Any]:
    """Valida el origen y lo devuelve (copia). Lo imposible se corta aquí, con su motivo.

    Se exige lo que hace falta para que la línea del recibo diga la verdad: `formato` y la huella COMPLETA del
    original (`sha256`, 64 hex: la corta `data_…` no es una prueba). Todo lo demás es opcional —un valor ausente
    es una respuesta y no se rellena—, pero lo que venga tiene que tener su forma. Las claves que este módulo no
    conoce (`conversor`, `rango`, `filas_excluidas`…) se CONSERVAN: son del esquema de la conversión, no de aquí.
    """
    if not isinstance(crudo, Mapping):
        raise ReproduceManifestError(f"{campo} must be an object, got {crudo!r}")
    origen = dict(crudo)

    esquema = origen.get("esquema")
    if esquema is not None:
        # «No se interpreta a medias»: un esquema más nuevo que el que este núcleo conoce puede querer decir otra cosa
        # con los mismos nombres.
        if type(esquema) is not int or esquema < 1 or esquema > _ESQUEMA_QUE_SE_SABE_LEER:
            raise ReproduceManifestError(
                f"{campo}.esquema must be an integer between 1 and {_ESQUEMA_QUE_SE_SABE_LEER} (the versions of the origin "
                f"this core can read), got {esquema!r}: an origin it does not know how to read is not interpreted half-way")

    _require_text(origen.get("formato"), f"{campo}.formato")
    if origen["formato"] not in FORMATOS:
        raise ReproduceManifestError(
            f"{campo}.formato must be one of {', '.join(FORMATOS)} (the formats the conversion produces), "
            f"got {origen['formato']!r}")
    _require_full_sha256(origen.get("sha256"), f"{campo}.sha256")
    if origen.get("sha256_del_csv") is not None:
        _require_full_sha256(origen["sha256_del_csv"], f"{campo}.sha256_del_csv")
    if origen.get("vinculo") is not None and origen["vinculo"] not in VINCULOS:
        raise ReproduceManifestError(
            f"{campo}.vinculo must be one of {', '.join(VINCULOS)}, got {origen['vinculo']!r}")
    if origen.get("vinculo") == "comprobado" and not origen.get("sha256_del_csv"):
        # Remate de 150-C4 (menor 3): «comprobado» es «coincide con la huella que el origen declara para su CSV»; sin esa
        # huella no hay nada con lo que coincidir, y un sobre o un paquete editado a mano lo diría igualmente.
        raise ReproduceManifestError(
            f"{campo}.vinculo is 'comprobado' but {campo}.sha256_del_csv is missing: a link cannot be checked against a "
            f"digest the origin does not declare")
    if origen.get("huella_de") is not None and origen["huella_de"] != "texto_recibido":
        raise ReproduceManifestError(f"{campo}.huella_de must be 'texto_recibido' or absent, got {origen['huella_de']!r}")

    for clave in ("nombre", "hoja"):
        valor = origen.get(clave)
        if valor is None:
            continue
        _require_text(valor, f"{campo}.{clave}")
        if len(valor) > nombre_citable.TOPE_DE_CARACTERES or _CONTROL.search(valor):
            raise ReproduceManifestError(
                f"{campo}.{clave} must be a text of at most 255 characters with no control characters, got {valor!r}")
        if _BIDI.search(valor):
            raise ReproduceManifestError(
                f"{campo}.{clave} carries a bidirectional control character (U+200E, U+200F, U+061C, U+202A–U+202E, "
                f"U+2066–U+2069), which makes the receipt read differently from what it records, got {valor!r}")
    octetos = origen.get("bytes")
    if octetos is not None and (type(octetos) is not int or octetos < 0):
        raise ReproduceManifestError(f"{campo}.bytes must be a non-negative integer or null, got {octetos!r}")

    for clave in ("declaraciones", "declaraciones_de_entrada"):
        if clave in origen and origen[clave] is not None:
            _lista_de_declaraciones(origen[clave], f"{campo}.{clave}")
        elif clave in origen:
            del origen[clave]

    try:
        pesa = len(canonical_json(origen).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ReproduceManifestError(f"{campo} must be JSON-serialisable: {exc}") from exc
    if pesa > TOPE_DE_BYTES_DEL_ORIGEN:
        raise ReproduceManifestError(
            f"{campo} is {pesa} bytes, over the {TOPE_DE_BYTES_DEL_ORIGEN} that a receipt carries: a file that "
            f"declares that much needs its own report, not a line in the receipt")
    return origen


def numero_de_declaraciones(origen: Mapping[str, Any]) -> int:
    """Cuántas cosas declaró el origen: las de la conversión del libro y las de la entrada canónica."""
    return sum(len(origen.get(c) or []) for c in ("declaraciones", "declaraciones_de_entrada"))


def origen_de_un_csv(texto_recibido: str, declaraciones: Sequence[Mapping[str, Any]], *,
                     nombre: str | None = None) -> dict[str, Any]:
    """El origen de un CSV que la entrada canónica CONVIRTIÓ o sobre el que declaró algo (150-C2).

    `texto_recibido` es el texto TAL COMO LLEGÓ al estudio, antes de normalizarlo: es lo único del original que el
    estudio ve (si el navegador decodificó o quitó un BOM, ese cambio ya estaba hecho), y `huella_de` lo dice.
    """
    octetos = texto_recibido.encode("utf-8", "surrogatepass")
    origen: dict[str, Any] = {
        "esquema": _ESQUEMA_QUE_SE_SABE_LEER,
        "formato": "csv",
        "sha256": hashlib.sha256(octetos).hexdigest(),
        "bytes": len(octetos),
        "huella_de": "texto_recibido",
        "declaraciones": [dict(d) for d in declaraciones],
    }
    if isinstance(nombre, str) and nombre.strip():
        origen["nombre"] = nombre.strip()[:255]
    return normalizar_origen_del_fichero(origen)


def origen_para_el_estudio(origen_recibido: Any, *, texto_recibido: str,
                           declaraciones_de_entrada: Sequence[Mapping[str, Any]] | None,
                           nombre: str | None = None) -> dict[str, Any] | None:
    """El `origen_del_fichero` de UN estudio, o `None` si no hay nada que decir.

    * Con `origen_recibido` (lo que devolvió la conversión de un libro): ese origen, validado, más las
      `declaraciones_de_entrada` si la entrada canónica también convirtió algo.
    * Sin él y con `declaraciones_de_entrada` (un CSV `;` con decimal español): el origen de ese CSV.
    * Sin ninguno de los dos: `None`. **No se inventa un origen para un CSV que no ha pasado por nada.**
    """
    entrada = [dict(d) for d in (declaraciones_de_entrada or [])]
    if origen_recibido is not None:
        # I1: el origen tiene que ser de ESTE CSV (el texto tal como llegó, que es lo que devolvió la conversión).
        origen = atar_al_csv(normalizar_origen_del_fichero(origen_recibido), texto_recibido, exigir=True)
        if entrada:
            origen["declaraciones_de_entrada"] = _lista_de_declaraciones(entrada, "declaraciones_de_entrada")
            return normalizar_origen_del_fichero(origen)
        return origen
    if entrada:
        return origen_de_un_csv(texto_recibido, entrada, nombre=nombre)
    return None


def es_origen_de_excel(origen: Any) -> bool:
    """¿Este origen (sin validar aún) dice que el CSV salió de la CONVERSIÓN de un libro de Excel?"""
    return isinstance(origen, Mapping) and origen.get("formato") in _FORMATOS_DE_EXCEL


def entrada_canonica_de(csv_text: str, origen_recibido: Any) -> tuple[str, list[dict]]:
    """La entrada canónica del estudio (150-C2) para ESTE CSV: `(texto, declaraciones_de_entrada)`.

    Un CSV que viene de `convertir` (origen de formato xlsx, xls o xlsm) **ya es canónico**: coma, punto decimal, sin BOM
    propio, escrito por nuestro escritor. Ese CSV pasa TAL CUAL y sin declaraciones, y el estudio guarda exactamente lo que
    devolvió la conversión (`dataset.sha256` = `sha256_del_csv`). Todo lo demás (un CSV `;` de verdad, con o sin origen propio)
    se canoniza como en el C2.

    Medido con dos libros (remate de 150-C4): volver a canonizarlo era PEOR que no hacerlo. Un Excel de una columna con la
    cabecera `peso;altura` salía con la cabecera reescrita a `peso,altura` sobre filas de una columna (HTTP 500), y una
    cabecera que empezaba por U+FEFF perdía el carácter, de modo que `dataset.sha256` dejaba de ser `sha256_del_csv`. Pero
    saltarse la canonización AQUÍ no basta: el análisis del núcleo (`analyze_dataset_csv`) y el entrenamiento canonizan por su
    cuenta. Por eso la conversión (`matrixai.tablas`) escribe un CSV que es un PUNTO FIJO de la entrada canónica (`peso_altura`,
    sin U+FEFF; una prueba lo recorre en todos los libros de los fixtures) y esta función solo lo deja pasar sin tocarlo.

    `normalize_csv_text_declarando` se importa aquí, no arriba: este módulo no necesita el entrenamiento para escribir un
    recibo."""
    if es_origen_de_excel(origen_recibido):
        return csv_text, []
    from matrixai.training.data import normalize_csv_text_declarando  # noqa: PLC0415
    return normalize_csv_text_declarando(csv_text)


class OrigenDeOtroCsv(ReproduceManifestError):
    """El origen no es del CSV que llega (I1). Lleva su `clave` y su `motivo` en es/en: lo redacta el núcleo."""

    clave = "origen_del_fichero_no_es_de_este_csv"

    def __init__(self, declarada: str | None, recibida: str):
        self.declarada, self.recibida = declarada, recibida
        if declarada is None:
            self.motivo = {
                "es": ("el origen del fichero no declara la huella del CSV que devolvió la conversión "
                       "(`sha256_del_csv`), así que no se puede comprobar que sea de este CSV"),
                "en": ("the file origin does not declare the digest of the CSV the conversion returned "
                       "(`sha256_del_csv`), so it cannot be checked to belong to this CSV")}
        else:
            self.motivo = {
                "es": (f"el origen del fichero es de otro CSV: la conversión devolvió uno con huella "
                       f"{declarada[:12]}… y el que llega tiene {recibida[:12]}…"),
                "en": (f"the file origin belongs to another CSV: the conversion returned one with digest "
                       f"{declarada[:12]}… and the one received has {recibida[:12]}…")}
        super().__init__(self.motivo["en"])


def huella_del_texto(texto: str) -> str:
    """El sha256 de un texto como UTF-8: el mismo que `TablaConvertida.csv_bytes` cuando el texto es su `csv_text`."""
    return hashlib.sha256(texto.encode("utf-8", "surrogatepass")).hexdigest()


def atar_al_csv(origen: Mapping[str, Any], texto_csv: str, *, exigir: bool) -> dict[str, Any]:
    """El origen (ya validado) con su `vinculo` al CSV que de verdad llega, escrito por el NÚCLEO (I1).

    Lo que traiga el cliente como `informativo`, `nota` o `vinculo` se descarta (M1): eso lo dice el núcleo.
    * `exigir=True` (el estudio, que recibe el CSV tal cual lo devolvió la conversión): si `sha256_del_csv` falta o
      no es la huella de `texto_csv`, lanza `OrigenDeOtroCsv`; si cuadra, `vinculo = "comprobado"`.
    * `exigir=False` (`train-start`, donde el CSV puede llegar preparado): nunca rechaza; `vinculo` dice si cuadró.
    """
    limpio = {k: v for k, v in origen.items() if k not in _CLAVES_DEL_NUCLEO}
    declarada = limpio.get("sha256_del_csv")
    recibida = huella_del_texto(texto_csv)
    if declarada == recibida:
        limpio["vinculo"] = "comprobado"
    elif exigir:
        raise OrigenDeOtroCsv(declarada, recibida)
    else:
        limpio["vinculo"] = "no_comprobado"
    return limpio


#: Lo que se dice junto al origen en el manifiesto: la huella del original NO es lo que se verifica.
_NOTA = {
    "es": ("Informativo: lo que se verifica es la huella del CSV derivado (`artifacts.dataset.sha256`), que es lo que "
           "vio el modelo. La huella del original no entra en ninguna comprobación (R1–R3): no se puede prometer "
           "reproducir desde el fichero original con otra versión de la biblioteca que lo convierte."),
    "en": ("Informative: what is verified is the digest of the derived CSV (`artifacts.dataset.sha256`), which is what "
           "the model saw. The digest of the original file takes part in no check (R1–R3): reproducing from the "
           "original file cannot be promised with another version of the library that converts it."),
}


def bloque_del_manifiesto(origen: Mapping[str, Any]) -> dict[str, Any]:
    """Lo que `artifacts.dataset.origen_del_fichero` publica: el origen más su carácter informativo, dicho.

    M1: `informativo` y `nota` van DESPUÉS del origen a propósito: si el origen trajera las suyas (un
    `informativo: false`, una nota que dice «esta huella SÍ se verifica»), siempre pierden."""
    return {**origen, "informativo": True, "nota": dict(_NOTA)}


def _pl(n: int, idioma: str) -> str:
    if idioma == "es":
        return f"{n} declaración" if n == 1 else f"{n} declaraciones"
    return f"{n} declaration" if n == 1 else f"{n} declarations"


def codigo_en_linea(texto: str) -> str:
    """`texto` como código en línea de Markdown (CommonMark), sin que nada de dentro se interprete (I3).

    El delimitador es una racha de comillas invertidas MÁS LARGA que la más larga que lleve el texto, y si el
    texto empieza o acaba con una (o empieza y acaba con un espacio), se rodea de un espacio, que CommonMark quita.
    Dentro de un código en línea no hay enlaces, ni énfasis, ni HTML: `javascript:`, `<img onerror>` o `<script>`
    salen como texto. Una hoja «Datos `2024`» sale «`` Datos `2024` ``», que se lee «Datos `2024`»."""
    racha = max((len(m) for m in re.findall(r"`+", texto)), default=0)
    delimitador = "`" * (racha + 1)
    relleno = " " if (texto.startswith("`") or texto.endswith("`")
                      or (texto.startswith(" ") and texto.endswith(" ") and texto.strip())) else ""
    return f"{delimitador}{relleno}{texto}{relleno}{delimitador}"


#: Lo que dice la ficha del VÍNCULO entre el origen y este dataset (I1), en cada idioma.
_VINCULO = {
    # Lo que se COMPRUEBA y nada más: la huella del origen la declara el cliente, y el servidor no guarda lo que devolvió
    # la conversión. Ninguna frase dice «comprobado» de que el dataset ES el CSV de la conversión.
    "comprobado": {
        "es": ("el dataset coincide con la huella que el origen declara para su CSV; el servidor no guarda lo que "
               "devolvió la conversión"),
        "en": ("the dataset matches the digest the origin declares for its CSV; the server does not keep what the "
               "conversion returned")},
    "no_comprobado": {
        "es": ("sin comprobar: el dataset no coincide, byte a byte, con la huella que el origen declara para su CSV; "
               "pudo prepararse después"),
        "en": ("not checked: the dataset does not match, byte for byte, the digest the origin declares for its CSV; "
               "it may have been prepared afterwards")},
    "no_consta": {
        "es": "sin comprobar: el origen no declara la huella de su CSV, así que no consta que este dataset salga de ese fichero",
        "en": ("not checked: the origin declares no digest for its CSV, so nothing records that this dataset came "
               "from that file")},
}


def frases_bilingues_del_origen(origen: Mapping[str, Any]) -> list[dict[str, str]]:
    """Las frases de lo que se hizo al convertir (filas de total quitadas, nombres cambiados…), TAL COMO las redactó el núcleo
    en `declaraciones` y `declaraciones_de_entrada` (`texto.es` / `texto.en`), una `{es, en}` por declaración. Una declaración
    sin su frase se omite, no se inventa; si falta un idioma, se usa el otro (nunca un hueco)."""
    frases: list[dict[str, str]] = []
    for clave in ("declaraciones", "declaraciones_de_entrada"):
        for d in origen.get(clave) or []:
            texto = d.get("texto") if isinstance(d, Mapping) else None
            if not isinstance(texto, Mapping):
                continue
            es, en = texto.get("es"), texto.get("en")
            es = es.strip() if isinstance(es, str) else ""
            en = en.strip() if isinstance(en, str) else ""
            if es or en:
                frases.append({"es": es or en, "en": en or es})
    return frases


def frases_del_origen(origen: Mapping[str, Any], idioma: str) -> list[str]:
    """Esas frases en UN idioma."""
    return [f["es" if idioma == "es" else "en"] for f in frases_bilingues_del_origen(origen)]


def texto_del_origen(origen: Mapping[str, Any], idioma: str, *, plano: bool = False) -> str:
    """«Excel `ventas.xlsx`, hoja `Datos`, huella del original `<sha256>` (vínculo …), 3 declaraciones» (+ lo que NO es).

    Una sola redacción para la ficha TRIPOD+AI y para el expediente: dos sitios declarando lo mismo acaban
    divergiendo. Lo que el origen no sabe (nombre, hoja) se omite, no se rellena. `nombre` y `hoja` van en
    `codigo_en_linea` (I3). Para un CSV cuya huella es la del TEXTO RECIBIDO (I2) no se llama «del original»: es
    la del texto ya decodificado como UTF-8 (sin BOM, si el navegador lo quitó), que puede no ser la del fichero."""
    es = idioma == "es"
    # `plano` (150-C5): para una pantalla o un HTML, donde no hay Markdown que interprete las comillas invertidas: el nombre y la
    # hoja van entre comillas del idioma. La ficha y el expediente (Markdown) siguen con `codigo_en_linea`.
    cod = (lambda x: f"«{x}»" if es else f"“{x}”") if plano else codigo_en_linea
    lengua = "es" if es else "en"
    formato = str(origen.get("formato") or "")
    etiqueta = "Excel" if formato in _FORMATOS_DE_EXCEL else formato.upper()
    piezas = [etiqueta + (f" {cod(origen['nombre'])}" if origen.get("nombre") else "")]
    if origen.get("hoja"):
        piezas.append(("hoja " if es else "sheet ") + cod(origen["hoja"]))
    huella = cod(str(origen.get("sha256")))
    if origen.get("huella_de") == "texto_recibido":
        piezas.append((f"huella del texto recibido (decodificado como UTF-8; puede no coincidir con la del fichero) "
                       f"{huella}") if es else
                      (f"digest of the received text (decoded as UTF-8; may not match the file's) {huella}"))
    else:
        vinculo = origen.get("vinculo")
        if vinculo == "no_comprobado" and not origen.get("sha256_del_csv"):
            vinculo = "no_consta"
        dice = _VINCULO.get(vinculo if vinculo in VINCULOS else "no_consta", _VINCULO["no_consta"])[lengua]
        piezas.append((f"huella del original {huella} ({dice})" if es
                       else f"digest of the original {huella} ({dice})"))
    piezas.append(_pl(numero_de_declaraciones(origen), "es" if es else "en"))
    recibido = origen.get("huella_de") == "texto_recibido"
    cola = (" (informativo: lo que se verifica es la huella del dataset derivado, la de arriba; la "
            + ("del texto recibido" if recibido else "del original") + " no entra en la verificación)" if es else
            " (informative: what is verified is the digest of the derived dataset, the one above; the digest of "
            + ("the received text" if recibido else "the original") + " takes no part in verification)")
    return ", ".join(piezas) + cola
