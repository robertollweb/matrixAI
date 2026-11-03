"""La TRAZA FIRMADA del servicio exportado — contrato 153-C1 (Ola 3, idea 19).

El servicio que sale de «Servir el modelo» escribe, hora a hora, un bloque con SOLO recuentos
(nada de filas ni de valores sueltos; DR1), firmado con una clave PROPIA del servicio y encadenado
con el anterior. Este módulo es el FORMATO y su COMPROBADOR: lo usan el Studio (C4), `/verificar`
(C6, en JS con los mismos vectores) y `matrixai trace verify`. **El escritor no vive aquí**: el
servicio instala el núcleo con que se entrenó el modelo, así que su escritor va en la plantilla
`servidor.py` (C3).

Formato `matrixai.traza/v1`: JSONL, un bloque por línea. Referencia completa y vectores en
`tests/fixtures/traza/` (`vectores_traza_v1.json`, `comprobar_traza.mjs`).

**Un solo canonicalizador**: `matrixai.pipelines.canonical.jcs_bytes` (JCS, RFC 8785). Lo que se
firma es `PREFIJO_FIRMA + jcs_bytes(bloque sin "firma")` (separación de dominio, como el 145-D8: una
firma de traza no vale como firma de recibo ni al revés). `prev` es el SHA-256 de `jcs_bytes` del
bloque anterior CON su firma; el génesis lleva 64 ceros. La firma y la huella son las de
`matrixai.pipelines.firma_asimetrica` (145-C1).

**Tres estados, y solo tres** (145-D11): `valida` («firma válida»), `no_coincide` («la firma no
coincide») y `no_se_pudo_comprobar`. Nunca «verificado». Se devuelve el PRIMER fallo, en este orden:

1. lectura y FORMA de cada línea                  -> `no_se_pudo_comprobar` (`formato` | `version`)
2. TODAS las firmas, bloque a bloque              -> `no_coincide` (`clave` | `firma`)
3. coherencia (sumas, k)                          -> `no_se_pudo_comprobar` (`formato`)
4. cadena, de un bloque al siguiente              -> `no_coincide` (`cadena`)
5. ancla (la huella que quien comprueba esperaba) -> `no_coincide` (`ancla`)

«La firma no coincide» es una propiedad de UN bloque (su contenido no es lo que firmó su clave);
«cadena rota» es una propiedad de PARES (todas las firmas buenas, pero no encajan). Las firmas se
miran todas antes que la cadena, y las sumas después de las firmas: un bloque editado debe decir
«la firma no coincide», no «formato».

**Lo que una firma válida NO demuestra** (§4 del contrato; `LO_QUE_NO_DEMUESTRA`): que el servicio
sea el tuyo si no cotejaste la huella, que las horas sean ciertas (las pone el reloj del servicio),
que no falten los últimos bloques (truncar la cola NO se detecta), ni que el modelo acierte más o
menos. No hay revocación.

`cryptography` es opcional (extra `[firma]`): sin ella el estado es `no_se_pudo_comprobar` con el
motivo `falta_cryptography`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Any, Iterable

from matrixai.pipelines import firma_asimetrica as _asim
from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.firma_asimetrica import (
    NO_COINCIDE, NO_SE_PUDO_COMPROBAR, VALIDA, texto_del_estado,
)

__all__ = [
    "FORMATO", "GENESIS", "K_MINIMO_POR_DEFECTO", "LO_QUE_NO_DEMUESTRA", "MOTIVOS", "PREFIJO_FIRMA",
    "SALIDAS", "TrazaIlegible", "comprobar_fichero", "comprobar_traza", "describir", "firmar_bloque",
    "hash_de_bloque", "leer_bloques", "lineas_de_texto", "mensaje_firmado",
]

FORMATO = "matrixai.traza/v1"
#: Separación de dominio de la firma (145-D8): el mensaje firmado empieza por esto.
PREFIJO_FIRMA = b"matrixai.traza.firma/v1\n"
GENESIS = "0" * 64
#: DS2/DS9: el umbral por omisión del escritor; cada bloque graba el SUYO en `k_minimo`.
K_MINIMO_POR_DEFECTO = 20
MAX_ENTERO = 2 ** 53 - 1  # lo que JavaScript representa exacto: Python y JS no pueden discrepar

#: Códigos de salida de `matrixai trace verify`: los mismos que `receipt verify` y `verify` (82-C2).
SALIDAS = {VALIDA: 0, NO_COINCIDE: 2, NO_SE_PUDO_COMPROBAR: 3}

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_FECHA = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_CAMPOS = {"formato", "seq", "prev", "servicio", "ventana", "n", "k_minimo", "incompleto", "reinicio",
           "salida", "columnas", "firma"}
_PROFUNDIDAD_MAXIMA = 64  # el JS tiene el mismo tope: una anidación absurda es «formato» en los dos


def _t(es: str, en: str) -> dict[str, str]:
    return {"es": es, "en": en}


#: Qué dice cada motivo (el core redacta, el core traduce). `detalle` del resultado dice el DÓNDE.
MOTIVOS: dict[str, dict[str, str]] = {
    "firma": _t("el contenido de un bloque no es lo que firmó su clave (editado o firma estropeada)",
                "a block's content is not what its key signed (edited, or the signature is damaged)"),
    "clave": _t("la huella de un bloque no es la de la clave pública que lleva",
                "a block's key fingerprint is not the one of the public key it carries"),
    "cadena": _t("las firmas son buenas pero los bloques no encajan entre sí (falta, sobra o está "
                 "cambiado de sitio alguno)",
                 "the signatures are good but the blocks do not fit each other (one is missing, "
                 "extra or out of place)"),
    "ancla": _t("la huella de la clave no es la que se esperaba (otro servicio u otra clave)",
                "the key fingerprint is not the expected one (another service or another key)"),
    "formato": _t("una línea no se pudo leer como un bloque de matrixai.traza/v1",
                  "a line could not be read as a matrixai.traza/v1 block"),
    "version": _t("el fichero es de una versión del formato que este programa no sabe leer",
                  "the file is from a format version this program cannot read"),
    "falta_cryptography": _t("falta cryptography, que la firma Ed25519 necesita "
                             "(pip install 'matrixai[firma]')",
                             "cryptography is missing and Ed25519 needs it (pip install 'matrixai[firma]')"),
}

#: §4 del contrato, texto fijo (es/en) para Seguimiento, /verificar y el README. Sin «garantiza»
#: ni «verificado»: lo barre una prueba.
LO_QUE_NO_DEMUESTRA: dict[str, list[str]] = {
    "es": [
        "Que el servicio sea el tuyo, si no cotejaste la huella de la clave por otro canal.",
        "Que las horas sean ciertas: las pone el reloj del servicio.",
        "Que no falten los últimos bloques: si alguien borra el final del fichero, sigue saliendo "
        "«firma válida». Compara el último bloque con el que anuncia el servicio.",
        "Que el modelo acierte más o menos: la traza mide entradas y salidas, no aciertos.",
        "No hay revocación de claves.",
    ],
    "en": [
        "That the service is yours, if you did not compare the key fingerprint through another channel.",
        "That the times are true: the service's own clock sets them.",
        "That no trailing blocks are missing: if someone deletes the end of the file it still reads "
        "\"valid signature\". Compare the last block with the one the service announces.",
        "That the model is right more or less often: the trace measures inputs and outputs, not hits.",
        "There is no key revocation.",
    ],
}


class TrazaIlegible(Exception):
    """Una línea no se pudo leer: `motivo` (`formato` | `version`), `bloque` (índice de línea) y `detalle`."""

    def __init__(self, motivo: str, bloque: int | None, detalle: dict[str, str]):
        super().__init__(detalle["es"])
        self.motivo, self.bloque, self.detalle = motivo, bloque, detalle


class _Formato(Exception):
    def __init__(self, motivo: str, detalle: dict[str, str]):
        self.motivo, self.detalle = motivo, detalle


def _mal(es: str, en: str, motivo: str = "formato") -> _Formato:
    return _Formato(motivo, _t(es, en))


# ───────────────────────────── lectura estricta ─────────────────────────────

def _sin_duplicados(pares):
    d: dict[str, Any] = {}
    for k, v in pares:
        if k in d:
            raise _mal(f"clave duplicada: {k!r}", f"duplicate key: {k!r}")
        d[k] = v
    return d


def _no_float(texto):
    raise _mal(f"número no entero: {texto}", f"not an integer: {texto}")


def _no_constante(texto):
    raise _mal(f"constante no permitida: {texto}", f"constant not allowed: {texto}")


def _entero_json(texto):
    # `-0` es un entero para Python y no para el JS que mira el texto: un solo criterio
    if texto.startswith("-"):
        raise _mal(f"entero negativo: {texto}", f"negative integer: {texto}")
    return int(texto)


def _recorrer(x: Any, profundidad: int = 0) -> None:
    """Cadenas que se pueden escribir en UTF-8 (una `\\ud800` suelta no: no hay bytes que firmar) y
    anidación acotada."""
    if profundidad > _PROFUNDIDAD_MAXIMA:
        raise _mal("anidación excesiva", "nesting too deep")
    if isinstance(x, str):
        try:
            x.encode("utf-8")
        except UnicodeEncodeError:
            raise _mal("una cadena lleva un sustituto UTF-16 suelto", "a string carries a lone UTF-16 surrogate") from None
    elif isinstance(x, list):
        for v in x:
            _recorrer(v, profundidad + 1)
    elif isinstance(x, dict):
        for k, v in x.items():
            _recorrer(k, profundidad + 1)
            _recorrer(v, profundidad + 1)


def _entero(x: Any, nombre: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > MAX_ENTERO:
        raise _mal(f"{nombre} no es un entero 0..2^53-1", f"{nombre} is not an integer 0..2^53-1")
    return x


def _lista_de_enteros(x: Any, largo: int, nombre: str) -> int:
    if not isinstance(x, list) or len(x) != largo:
        raise _mal(f"{nombre} no es una lista de {largo} enteros", f"{nombre} is not a list of {largo} integers")
    for v in x:
        _entero(v, nombre)
    return sum(x)


def _histograma(h: Any, nombre: str) -> tuple[int, int]:
    """`{"bins": [10 enteros], "fuera": entero}` -> (suma de bins, fuera)."""
    if not isinstance(h, dict) or set(h) != {"bins", "fuera"}:
        raise _mal(f"{nombre}: campos incorrectos", f"{nombre}: wrong fields")
    return _lista_de_enteros(h["bins"], 10, f"{nombre}.bins"), _entero(h["fuera"], f"{nombre}.fuera")


def _estructura(b: Any, coherencia: bool = False) -> None:
    """Con `coherencia=False` solo la FORMA (lo mínimo para poder comprobar la firma); con `True`
    también las sumas y el umbral. Las sumas se miran DESPUÉS de las firmas."""
    if not isinstance(b, dict):
        raise _mal("el bloque no es un objeto", "the block is not an object")
    if b.get("formato") != FORMATO:
        raise _mal(f"formato desconocido: {b.get('formato')!r}", f"unknown format: {b.get('formato')!r}", "version")
    if set(b) != _CAMPOS:
        raise _mal(f"campos distintos de los de v1: sobran {sorted(set(b) - _CAMPOS)}, faltan {sorted(_CAMPOS - set(b))}",
                   f"fields differ from v1: extra {sorted(set(b) - _CAMPOS)}, missing {sorted(_CAMPOS - set(b))}")
    _entero(b["seq"], "seq")
    n = _entero(b["n"], "n")
    k = _entero(b["k_minimo"], "k_minimo")
    if k < 1:
        raise _mal("k_minimo debe ser >= 1", "k_minimo must be >= 1")
    if not isinstance(b["prev"], str) or not _HEX64.match(b["prev"]):
        raise _mal("prev no es SHA-256 en hex minúscula", "prev is not a lowercase hex SHA-256")
    s = b["servicio"]
    if not isinstance(s, dict) or set(s) != {"id", "huella_clave", "clave_publica", "huella_modelo"}:
        raise _mal("servicio: campos incorrectos", "servicio: wrong fields")
    if not isinstance(s["id"], str) or not (1 <= len(s["id"]) <= 128):  # puntos de código, como el JS
        raise _mal("servicio.id: de 1 a 128 caracteres", "servicio.id: 1 to 128 characters")
    for campo in ("huella_clave", "clave_publica", "huella_modelo"):
        if not isinstance(s[campo], str) or not _HEX64.match(s[campo]):
            raise _mal(f"servicio.{campo} no es hex de 64", f"servicio.{campo} is not 64 hex characters")
    v = b["ventana"]
    if not isinstance(v, dict) or set(v) != {"desde", "hasta"}:
        raise _mal("ventana: campos incorrectos", "ventana: wrong fields")
    for campo in ("desde", "hasta"):
        if not isinstance(v[campo], str) or not _FECHA.match(v[campo]):
            raise _mal(f"ventana.{campo} no es AAAA-MM-DDThh:mm:ssZ", f"ventana.{campo} is not YYYY-MM-DDThh:mm:ssZ")
    if v["desde"] > v["hasta"]:  # la HORA no se comprueba (DS11); solo que la ventana no esté al revés
        raise _mal("ventana.desde > ventana.hasta", "ventana.desde > ventana.hasta")
    if not isinstance(b["incompleto"], bool):
        raise _mal("incompleto no es booleano", "incompleto is not a boolean")
    if b["reinicio"] not in (None, "sin_previo"):
        raise _mal("reinicio: solo null o \"sin_previo\"", "reinicio: only null or \"sin_previo\"")
    f = b["firma"]
    if not isinstance(f, dict) or set(f) != {"alg", "valor"} or f["alg"] != "ed25519" or not isinstance(f["valor"], str):
        raise _mal("firma: se espera {alg:\"ed25519\", valor:<base64>}", "firma: expected {alg:\"ed25519\", valor:<base64>}")
    if b["incompleto"]:
        if b["salida"] is not None or b["columnas"] is not None:
            raise _mal("incompleto: salida y columnas deben ser null (solo n)", "incompleto: salida and columnas must be null (only n)")
        if coherencia and n >= k:
            raise _mal("incompleto con n >= k_minimo: se habría cerrado entero", "incompleto with n >= k_minimo: it would have closed in full")
        return
    if coherencia and n < k:
        raise _mal("un bloque completo con n < k_minimo (DS2: no se cierra)", "a complete block with n < k_minimo (DS2: it does not close)")
    sal = b["salida"]
    if not isinstance(sal, dict) or set(sal) != {"probabilidad", "etiquetas", "valor"}:
        raise _mal("salida: campos incorrectos", "salida: wrong fields")
    if sal["probabilidad"] is not None and _lista_de_enteros(sal["probabilidad"], 10, "salida.probabilidad") != n and coherencia:
        raise _mal("salida.probabilidad no suma n", "salida.probabilidad does not add up to n")
    if sal["etiquetas"] is not None:
        e = sal["etiquetas"]
        if not isinstance(e, dict):
            raise _mal("salida.etiquetas no es un objeto", "salida.etiquetas is not an object")
        for c in e.values():
            _entero(c, "etiquetas")
        if coherencia and sum(e.values()) != n:
            raise _mal("salida.etiquetas no suma n", "salida.etiquetas does not add up to n")
    if sal["valor"] is not None:
        suma, fuera = _histograma(sal["valor"], "salida.valor")
        if sal["probabilidad"] is not None or sal["etiquetas"] is not None:
            raise _mal("salida.valor (regresión) con probabilidad o etiquetas: deben ser null",
                       "salida.valor (regression) with probabilidad or etiquetas: they must be null")
        if coherencia and (suma != n or fuera > suma):
            raise _mal("salida.valor: sus casillas no suman n o fuera > casillas", "salida.valor: its bins do not add up to n or fuera > bins")
    cols = b["columnas"]
    if not isinstance(cols, dict):
        raise _mal("columnas no es un objeto", "columnas is not an object")
    for nombre, c in cols.items():
        if not isinstance(c, dict) or c.get("tipo") not in ("numerica", "categorica"):
            raise _mal(f"columna {nombre!r}: tipo", f"column {nombre!r}: type")
        if c["tipo"] == "numerica":
            if set(c) != {"tipo", "bins", "faltantes", "fuera"}:
                raise _mal(f"columna {nombre!r}: campos", f"column {nombre!r}: fields")
            suma = _lista_de_enteros(c["bins"], 10, "bins")
            if (suma + _entero(c["faltantes"], "faltantes") != n or _entero(c["fuera"], "fuera") > suma) and coherencia:
                raise _mal(f"columna {nombre!r}: bins + faltantes != n, o fuera > bins",
                           f"column {nombre!r}: bins + faltantes != n, or fuera > bins")
        else:
            if set(c) != {"tipo", "nuevas", "faltantes"}:
                raise _mal(f"columna {nombre!r}: campos", f"column {nombre!r}: fields")
            if _entero(c["nuevas"], "nuevas") + _entero(c["faltantes"], "faltantes") > n and coherencia:
                raise _mal(f"columna {nombre!r}: nuevas + faltantes > n", f"column {nombre!r}: nuevas + faltantes > n")


# ───────────────────────────── firma y cadena ─────────────────────────────

def mensaje_firmado(bloque: dict) -> bytes:
    """Lo que se firma: el prefijo de dominio + el JCS del bloque SIN `firma`."""
    return PREFIJO_FIRMA + jcs_bytes({k: v for k, v in bloque.items() if k != "firma"})


def hash_de_bloque(bloque: dict) -> str:
    """El `prev` del bloque siguiente: SHA-256 (hex) del JCS del bloque COMPLETO, con su firma."""
    return hashlib.sha256(jcs_bytes(bloque)).hexdigest()


def firmar_bloque(clave_privada: Any, bloque: dict) -> dict:
    """El bloque con su `firma` puesta (la que traiga se sustituye). Lo usan las pruebas y quien escriba
    trazas desde Python; la plantilla del servicio tiene su propio escritor."""
    sin = {k: v for k, v in bloque.items() if k != "firma"}
    return {**sin, "firma": {"alg": _asim.ALGORITMO, "valor": _asim.firmar(clave_privada, mensaje_firmado(sin))}}


def _base64_canonico(texto: str) -> bool:
    """Una firma Ed25519 es 64 bytes = 88 caracteres base64 con SU forma única (dos textos para los mismos
    bytes serían una maleabilidad que Python y JS podrían tratar distinto)."""
    try:
        crudo = base64.b64decode(texto, validate=True)
    except (ValueError, binascii.Error):
        return False
    return len(crudo) == 64 and base64.b64encode(crudo).decode("ascii") == texto


def _res(estado: str, motivo: str | None = None, bloque: int | None = None,
         detalle: dict[str, str] | None = None, **extra: Any) -> dict[str, Any]:
    r = {"estado": estado, "motivo": motivo, "bloque": bloque, "detalle": detalle}
    r.update(extra)
    return r


# ───────────────────────────── lectura de ficheros ─────────────────────────────

def lineas_de_texto(texto: str) -> list[str]:
    """Las líneas (bloques) de un fichero ya decodificado: se parte SOLO por `\\n` (no por `\\u2028` ni
    otros separadores de `splitlines`) y la última línea vacía, la del `\\n` final, no cuenta. Una línea
    en blanco en medio sí cuenta: no es un bloque."""
    partes = texto.split("\n")
    if partes and partes[-1] == "":
        partes.pop()
    return partes


def leer_bloques(lineas: Iterable[str]) -> list[dict]:
    """Los bloques de las líneas, leídos con el criterio estricto del formato (sin claves duplicadas, solo
    enteros, forma de v1) y SIN comprobar firmas. `TrazaIlegible` si una línea no se puede leer."""
    bloques = []
    for i, linea in enumerate(lineas):
        try:
            b = json.loads(linea, object_pairs_hook=_sin_duplicados, parse_float=_no_float,
                           parse_constant=_no_constante, parse_int=_entero_json)
            _recorrer(b)
            _estructura(b)
        except _Formato as e:
            raise TrazaIlegible(e.motivo, i, e.detalle) from None
        except RecursionError:
            raise TrazaIlegible("formato", i, _t("anidación excesiva", "nesting too deep")) from None
        except (ValueError, TypeError) as e:  # JSON mal formado (incluye un entero de miles de cifras)
            raise TrazaIlegible("formato", i, _t(f"no es JSON: {e}", f"not JSON: {e}")) from None
        bloques.append(b)
    return bloques


# ───────────────────────────── el comprobador ─────────────────────────────

def comprobar_traza(lineas: Iterable[str], huella_anclada: str | None = None,
                    esperar_genesis: bool = False) -> dict[str, Any]:
    """Comprueba una traza (una línea de texto por bloque) y dice QUÉ pasa, sin lanzar por culpa del
    contenido. `huella_anclada`: la huella de clave (64 hex) que quien comprueba esperaba, o `None`.
    `esperar_genesis`: exigir que el fichero empiece en el génesis (primera importación).

    Devuelve `{estado, motivo, bloque, detalle, ...}`: `bloque` es el índice de la línea (desde 0) del primer
    fallo; `detalle` es `{"es", "en"}`. Si `estado == "valida"` lleva además `bloques, empieza_en, ultimo_seq,
    ultimo_hash, huellas, reinicios, anclada`. **Que sea válida NO dice que no falten los últimos bloques** (§4):
    `ultimo_seq` y `ultimo_hash` están para cotejarlos con lo que el servicio anuncia o con una importación
    anterior.

    `ValueError` solo si `huella_anclada` no es una huella (64 hex minúscula): eso es un error de quien llama.
    """
    if huella_anclada is not None and not (isinstance(huella_anclada, str) and _HEX64.match(huella_anclada)):
        raise ValueError("la huella anclada es SHA-256 en hexadecimal minúscula (64 caracteres)")
    lineas = list(lineas)
    try:
        bloques = leer_bloques(lineas)
    except TrazaIlegible as e:
        return _res(NO_SE_PUDO_COMPROBAR, e.motivo, e.bloque, e.detalle)
    if not bloques:
        return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _t("fichero sin bloques", "file without blocks"))
    if not _asim.disponible():
        return _res(NO_SE_PUDO_COMPROBAR, "falta_cryptography", None, MOTIVOS["falta_cryptography"])
    for i, b in enumerate(bloques):
        s = b["servicio"]
        crudo = bytes.fromhex(s["clave_publica"])
        if hashlib.sha256(crudo).hexdigest() != s["huella_clave"]:
            return _res(NO_COINCIDE, "clave", i, _t("la huella no es la de la clave pública que lleva el bloque",
                                                    "the fingerprint is not the one of the public key the block carries"))
        try:
            buena = _base64_canonico(b["firma"]["valor"]) and _asim.comprobar(crudo, mensaje_firmado(b), b["firma"]["valor"])
        except _asim.ClaveInvalida:  # 32 bytes que no son un punto de la curva: no firmó nada
            buena = False
        if not buena:
            return _res(NO_COINCIDE, "firma", i, _t("la firma no coincide con el contenido del bloque",
                                                    "the signature does not match the block's content"))
    for i, b in enumerate(bloques):
        try:
            _estructura(b, coherencia=True)
        except _Formato as e:
            return _res(NO_SE_PUDO_COMPROBAR, e.motivo, i, e.detalle)
    reinicios: list[int] = []
    for i, b in enumerate(bloques):
        if i == 0 and esperar_genesis and not (b["seq"] == 0 and b["prev"] == GENESIS):
            return _res(NO_COINCIDE, "cadena", 0, _t("se esperaba el bloque génesis (seq 0, prev de ceros) y el fichero empieza más tarde",
                                                     "the genesis block (seq 0, prev of zeros) was expected and the file starts later"))
        if b["reinicio"] == "sin_previo":
            if not (b["seq"] == 0 and b["prev"] == GENESIS):
                return _res(NO_COINCIDE, "cadena", i, _t("un reinicio sin_previo debe tener seq 0 y prev de ceros",
                                                         "a sin_previo restart must have seq 0 and a prev of zeros"))
            reinicios.append(i)
        elif i > 0:
            a = bloques[i - 1]
            if b["seq"] != a["seq"] + 1:
                return _res(NO_COINCIDE, "cadena", i, _t(f"seq {b['seq']} no sigue a {a['seq']}", f"seq {b['seq']} does not follow {a['seq']}"))
            if b["prev"] != hash_de_bloque(a):
                return _res(NO_COINCIDE, "cadena", i, _t("prev no es el SHA-256 del bloque anterior", "prev is not the SHA-256 of the previous block"))
            if b["servicio"]["huella_clave"] != a["servicio"]["huella_clave"]:
                return _res(NO_COINCIDE, "cadena", i, _t("la clave cambia dentro de la cadena sin un reinicio sin_previo",
                                                         "the key changes inside the chain without a sin_previo restart"))
    huellas: list[str] = []
    for b in bloques:
        h = b["servicio"]["huella_clave"]
        if h not in huellas:
            huellas.append(h)
    if huella_anclada is not None:
        for i, b in enumerate(bloques):
            if b["servicio"]["huella_clave"] != huella_anclada:
                return _res(NO_COINCIDE, "ancla", i, _t("la huella de este bloque no es la anclada", "this block's fingerprint is not the anchored one"))
    return _res(VALIDA, None, None, None, bloques=len(bloques), empieza_en=bloques[0]["seq"],
                ultimo_seq=bloques[-1]["seq"], ultimo_hash=hash_de_bloque(bloques[-1]),
                huellas=huellas, reinicios=reinicios, anclada=huella_anclada is not None)


def comprobar_fichero(contenido: bytes | str, huella_anclada: str | None = None,
                      esperar_genesis: bool = False) -> dict[str, Any]:
    """Como `comprobar_traza`, sobre el contenido de un fichero `.jsonl` (bytes UTF-8 o texto)."""
    if isinstance(contenido, (bytes, bytearray)):
        try:
            contenido = bytes(contenido).decode("utf-8")
        except UnicodeDecodeError:
            return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _t("el fichero no es UTF-8", "the file is not UTF-8"))
    return comprobar_traza(lineas_de_texto(contenido), huella_anclada, esperar_genesis)


# ───────────────────────────── lo que se le dice a quien lee ─────────────────────────────

def describir(resultado: dict[str, Any], idioma: str = "es") -> list[str]:
    """Las líneas de pantalla para un resultado, en `es` o `en`, con el vocabulario cerrado de 145-D11. Si es
    válida, lleva el último bloque y lo que NO demuestra (que falten los últimos bloques, la primera cosa)."""
    idioma = "es" if idioma == "es" else "en"
    es = idioma == "es"
    estado = resultado["estado"]
    lineas = [(("Traza: " if es else "Trace: ") + texto_del_estado(estado, idioma))]
    if estado != VALIDA:
        motivo = resultado.get("motivo")
        if motivo in MOTIVOS:
            lineas.append(MOTIVOS[motivo][idioma])
        if resultado.get("bloque") is not None:
            lineas.append((f"en la línea {resultado['bloque'] + 1} del fichero" if es
                           else f"at line {resultado['bloque'] + 1} of the file"))
        if resultado.get("detalle"):
            lineas.append(resultado["detalle"][idioma])
        return lineas
    lineas.append((f"{resultado['bloques']} bloques, del seq {resultado['empieza_en']} al {resultado['ultimo_seq']}" if es
                   else f"{resultado['bloques']} blocks, from seq {resultado['empieza_en']} to {resultado['ultimo_seq']}"))
    lineas.append(("Huella de la clave: " if es else "Key fingerprint: ") + ", ".join(resultado["huellas"]))
    lineas.append((f"Último bloque visto: seq {resultado['ultimo_seq']}, hash {resultado['ultimo_hash']}" if es
                   else f"Last block seen: seq {resultado['ultimo_seq']}, hash {resultado['ultimo_hash']}"))
    if not resultado.get("anclada"):
        lineas.append(("Sin huella anclada: solo se comprobó que el fichero es coherente consigo mismo; la clave es la que el propio fichero dice."
                       if es else
                       "No anchored fingerprint: it only shows the file is consistent with itself; the key is whatever the file says."))
    if resultado["empieza_en"] > 0:
        lineas.append(("El fichero no empieza en el génesis: los bloques anteriores no están aquí." if es
                       else "The file does not start at the genesis block: the earlier blocks are not here."))
    if resultado["reinicios"]:
        lineas.append((f"Cadena nueva (reinicio sin_previo) en las líneas: {[i + 1 for i in resultado['reinicios']]}" if es
                       else f"New chain (sin_previo restart) at lines: {[i + 1 for i in resultado['reinicios']]}"))
    if len(resultado["huellas"]) > 1:
        lineas.append(("Hay más de una clave en el fichero." if es else "There is more than one key in the file."))
    lineas.append(("Lo que esto NO demuestra:" if es else "What this does NOT show:"))
    lineas.extend("- " + frase for frase in LO_QUE_NO_DEMUESTRA[idioma])
    return lineas
