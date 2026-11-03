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

0. TAMAÑO: 1 MiB por línea, 64 MiB el fichero    -> `no_se_pudo_comprobar` (`formato`), antes de leer JSON
1. lectura y FORMA de cada línea                  -> `no_se_pudo_comprobar` (`formato` | `version`)
2. TODAS las firmas, bloque a bloque              -> `no_coincide` (`clave` | `firma`); una clave de orden
   pequeño o no canónica es `clave` antes de verificar (con ella, cualquier contenido «estaría firmado»)
3. coherencia (sumas, k)                          -> `no_se_pudo_comprobar` (`formato`)
4. cadena, de un bloque al siguiente, y entre    -> `no_coincide` (`cadena`)
   cadenas: un bloque repetido, o una cadena nueva que empieza antes de que acabe la anterior de la misma clave
5. ancla (la huella que quien comprueba esperaba) -> `no_coincide` (`ancla`)

«La firma no coincide» es una propiedad de UN bloque (su contenido no es lo que firmó su clave);
«cadena rota» es una propiedad de PARES (todas las firmas buenas, pero no encajan). Las firmas se
miran todas antes que la cadena, y las sumas después de las firmas: un bloque editado debe decir
«la firma no coincide», no «formato».

**Lo que una firma válida NO demuestra** (§4 del contrato; `LO_QUE_NO_DEMUESTRA`): que el servicio
sea el tuyo si no cotejaste la huella, que las horas sean ciertas (las pone el reloj del servicio),
que no falten los últimos bloques (truncar la cola NO se detecta), que no falten bloques antes de un
reinicio `sin_previo` (borrar o reordenar se detecta DENTRO de una cadena; entre cadenas, solo un bloque
repetido o una cadena de la misma clave fuera de orden), ni que el modelo acierte más o menos. No hay
revocación.

`cryptography` es opcional (extra `[firma]`): sin ella el estado es `no_se_pudo_comprobar` con el
motivo `falta_cryptography`.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import json
import re
from typing import Any, Iterable, Iterator

from matrixai.pipelines import firma_asimetrica as _asim
from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.firma_asimetrica import (
    NO_COINCIDE, NO_SE_PUDO_COMPROBAR, VALIDA, texto_del_estado,
)

__all__ = [
    "AVISO_COLA", "AVISO_REINICIO", "FORMATO", "GENESIS", "K_MINIMO_POR_DEFECTO", "LO_QUE_NO_DEMUESTRA", "MOTIVOS", "PREFIJO_FIRMA",
    "SALIDAS", "TOPE_FICHERO", "TOPE_LINEA", "TrazaIlegible", "comprobar_fichero", "comprobar_traza", "describir", "firmar_bloque",
    "hash_de_bloque", "leer_bloques", "lineas_de_texto", "mensaje_firmado",
]

FORMATO = "matrixai.traza/v1"
#: Separación de dominio de la firma (145-D8): el mensaje firmado empieza por esto.
PREFIJO_FIRMA = b"matrixai.traza.firma/v1\n"
GENESIS = "0" * 64
#: DS2/DS9: el umbral por omisión del escritor; cada bloque graba el SUYO en `k_minimo`.
K_MINIMO_POR_DEFECTO = 20
MAX_ENTERO = 2 ** 53 - 1  # lo que JavaScript representa exacto: Python y JS no pueden discrepar
#: Topes de tamaño (153-C1, I2), los MISMOS en el JS, comprobados ANTES de leer el JSON. MEDIDO (rep-c1, `~/.agentes/153/rep-c1/`):
#: un bloque pesa 1,2 KB con 6 columnas y 4,7 KB con 42; un AÑO de bloques horarios con 42 columnas, en un solo fichero, 41,5 MB
#: (el §7 del contrato estimaba ~33). La línea de 1 MiB es ~200 bloques de 42 columnas; el fichero de 64 MiB, 1,6 veces ese año (y
#: el servicio escribe un fichero por día, DR5); y 74 MB, que antes costaban 1,1 GB y 43 s, son «formato» en 0,4 s. En Node, una
#: línea de 100 MB era «sin memoria» a los 16 s; ahora, «formato» en 0,1 s. Lo que aquí decía «el peor fichero de 64 MiB, 193 MB» era
#: FALSO (re-auditoría): 64 MiB de líneas diminutas (`ab\n`, `\n`) gastaban 535–1.284 MB en Python y 502–1.013 MB en Node, por partir
#: el fichero entero. Remate (`~/.agentes/153/remate-c1/`, RSS máximo bajo `MemoryMax=3G`, sin swap): ya no se crea la lista; esas
#: líneas diminutas, 85 MB en Python (64 de ellos, el propio fichero) y «formato» en 0,1 s; el PEOR caso es ahora 64 MiB de bloques
#: VÁLIDOS mínimos (100.029 incompletos de 672 bytes): 141 MB y 50 s en Python, 308 MB y 36 s en Node (antes, 222 y 369 MB). El año
#: de 42 columnas, 74 MB en Python y 188 MB en Node. No hace falta tope de número de líneas: una línea que no es un bloque para la
#: lectura ahí, y más de ~100.000 bloques no caben en 64 MiB.
TOPE_LINEA = 1 << 20      # bytes UTF-8 de una línea, sin su `\n`
TOPE_FICHERO = 64 << 20   # bytes del fichero; con líneas sueltas, la suma de (bytes de cada línea + 1)

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
    "clave": _t("la clave pública de un bloque no vale: su huella no es la suya, o no es una clave Ed25519 utilizable "
                "(no canónica, fuera de la curva o de orden pequeño, que «firmaría» cualquier contenido)",
                "a block's public key is not usable: its fingerprint is not its own, or it is not a usable Ed25519 key "
                "(non-canonical, off the curve or of small order, which would \"sign\" any content)"),
    "cadena": _t("las firmas son buenas pero los bloques no encajan entre sí (dentro de una cadena falta, sobra o está "
                 "cambiado de sitio alguno; o un bloque está repetido; o una cadena empieza antes de que acabe la anterior de "
                 "la misma clave)",
                 "the signatures are good but the blocks do not fit each other (inside a chain one is missing, extra or "
                 "out of place; or a block is repeated; or a chain starts before the previous one of the same key ends)"),
    "ancla": _t("la huella de la clave no es la que se esperaba (otro servicio u otra clave)",
                "the key fingerprint is not the expected one (another service or another key)"),
    "formato": _t("el fichero o una de sus líneas no se pudo leer como matrixai.traza/v1 (o pasa del tope de tamaño)",
                  "the file or one of its lines could not be read as matrixai.traza/v1 (or it is over the size limit)"),
    "version": _t("el fichero es de una versión del formato que este programa no sabe leer",
                  "the file is from a format version this program cannot read"),
    "falta_cryptography": _t("falta cryptography, que la firma Ed25519 necesita "
                             "(pip install 'matrixai[firma]')",
                             "cryptography is missing and Ed25519 needs it (pip install 'matrixai[firma]')"),
}

#: El aviso de la COLA (§4): lo dicen `describir`, la CLI (también con `--json`) y el Studio. El cotejo con lo que anuncia el
#: servicio es una OBLIGACIÓN de C3: `/health` y `/modelo` dan `ultimo_seq` y `ultimo_hash` (anotado en el contrato).
AVISO_COLA: dict[str, str] = {
    "es": "Que no falten los últimos bloques: si alguien borra el final del fichero, sigue saliendo «firma válida». "
          "Compara el último bloque (seq y hash) con el que anuncia el servicio en su /health.",
    "en": "That no trailing blocks are missing: if someone deletes the end of the file it still reads \"valid signature\". "
          "Compare the last block (seq and hash) with the one the service announces on its /health.",
}
#: Lo que NO demuestra en un REINICIO (B1 de la auditoría de C1): `sin_previo` dice «no hay bloque anterior», así que nada enlaza
#: la cadena nueva con lo que hubo antes.
AVISO_REINICIO: dict[str, str] = {
    "es": "Que no falten bloques antes de un reinicio: una cadena nueva («sin_previo») no enlaza con lo anterior, así que pueden "
          "faltar los últimos bloques de la cadena de antes, o una cadena entera. Borrar o reordenar bloques se detecta DENTRO "
          "de una cadena; entre cadenas, solo un bloque repetido o una cadena de la misma clave que empieza antes de que acabe "
          "la anterior.",
    "en": "That no blocks are missing before a restart: a new chain (\"sin_previo\") does not link to what came before, so the "
          "last blocks of the previous chain, or a whole chain, may be missing. Deleting or reordering blocks is detected INSIDE "
          "a chain; across chains, only a repeated block or a chain of the same key that starts before the previous one ends.",
}

#: §4 del contrato, texto fijo (es/en) para Seguimiento, /verificar y el README. Sin «garantiza»
#: ni «verificado»: lo barre una prueba.
LO_QUE_NO_DEMUESTRA: dict[str, list[str]] = {
    "es": [
        "Que el servicio sea el tuyo, si no cotejaste la huella de la clave por otro canal.",
        "Que las horas sean ciertas: las pone el reloj del servicio.",
        AVISO_COLA["es"],
        AVISO_REINICIO["es"],
        "Que el modelo acierte más o menos: la traza mide entradas y salidas, no aciertos.",
        "No hay revocación de claves.",
    ],
    "en": [
        "That the service is yours, if you did not compare the key fingerprint through another channel.",
        "That the times are true: the service's own clock sets them.",
        AVISO_COLA["en"],
        AVISO_REINICIO["en"],
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


# ───────────────────────────── claves y firmas Ed25519 que no valen (I1) ─────────────────────────────
#
# OpenSSL (y por tanto `cryptography` y `node:crypto`) acepta claves públicas de ORDEN PEQUEÑO y codificaciones no canónicas: con
# la clave identidad (`01 00…`) y la firma R = identidad, S = 0, la ecuación de Ed25519 se cumple para CUALQUIER mensaje. Medido
# por la auditoría de C1: 10 combinaciones de clave y firma así daban «firma válida». Aquí se decide ANTES de verificar, con
# aritmética propia (la misma en el JS), para no depender de lo que acepte cada biblioteca ni cada navegador (C6, WebCrypto).

_P = 2 ** 255 - 19
_D = (-121665 * pow(121666, -1, _P)) % _P
_L = 2 ** 252 + 27742317777372353535851937790883648493  # el orden del grupo: S < L (RFC 8032 §5.1.7)
_Y8 = 0x7A03AC9277FDC74EC6CC392CFA53202A0F67100D760B3CBA4FD84D3D706A17C7
#: Las coordenadas y de los 8 puntos de orden pequeño (orden 1, 2, 4 y 8). Con los dos signos de x salen los 8 puntos; los de
#: x = 0 (y = ±1) con el bit de signo puesto son además codificaciones no canónicas. Todos se rechazan.
_Y_DE_ORDEN_PEQUENO = frozenset({0, 1, _P - 1, _Y8, _P - _Y8})


def _y_de(crudo32: bytes) -> int:
    return int.from_bytes(crudo32, "little") & ((1 << 255) - 1)


def _clave_publica_utilizable(crudo32: bytes) -> bool:
    """¿Son estos 32 bytes una clave Ed25519 que solo firma quien tiene su privada? `y` canónico (< p), que no sea de un punto de
    orden pequeño, y que exista el punto (x² = (y²−1)/(d·y²+1) es un cuadrado módulo p)."""
    y = _y_de(crudo32)
    if y >= _P or y in _Y_DE_ORDEN_PEQUENO:
        return False
    x2 = (y * y - 1) * pow(_D * y * y + 1, -1, _P) % _P
    return x2 == 0 or pow(x2, (_P - 1) // 2, _P) == 1  # x2 == 0 solo con y = ±1, ya fuera


def _firma_bien_formada(firma64: bytes) -> bool:
    """R con su y canónico y no de orden pequeño; S < L. Lo que no cumpla no es una firma, la acepte o no la biblioteca."""
    y_r = _y_de(firma64[:32])
    return y_r < _P and y_r not in _Y_DE_ORDEN_PEQUENO and int.from_bytes(firma64[32:], "little") < _L


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


def _bytes_utf8(linea: str) -> int:
    """Lo que pesa la línea en UTF-8. Un sustituto suelto cuenta 3 bytes, como en el JS (luego el lector lo rechaza)."""
    if len(linea) > TOPE_LINEA:  # los puntos de código nunca son más que los bytes: no hace falta codificar 100 MB
        return len(linea)
    return len(linea.encode("utf-8", "surrogatepass"))


def _tramos(contenido: bytes | str, salto: bytes | str) -> Iterator[tuple[int, int]]:
    """`(inicio, fin)` de cada línea del contenido, SIN crear la lista de líneas (I2 de la re-auditoría: `split` de 64 MiB de
    líneas diminutas eran 22 millones de objetos, 535–1.284 MB). Las mismas líneas que `split` quitando la última vacía."""
    n, i = len(contenido), 0
    while i < n:
        j = contenido.find(salto, i)
        if j < 0:
            yield i, n
            return
        yield i, j
        i = j + 1


def _linea_grande(i: int) -> dict[str, Any]:
    return _res(NO_SE_PUDO_COMPROBAR, "formato", i, _t(f"la línea pesa más de {TOPE_LINEA >> 20} MiB, el tope de un bloque",
                                                       f"the line is over {TOPE_LINEA >> 20} MiB, the limit for one block"))


def _fuera_de_tope(tamanos: Iterable[int]) -> dict[str, Any] | None:
    """El tope de línea, en orden, y luego el del fichero: ANTES de leer ningún JSON (I2). Igual en el JS. `tamanos`: lo que
    pesa cada línea en bytes UTF-8 (un iterable: no hace falta tener las líneas a la vez)."""
    total = 0
    for i, n in enumerate(tamanos):
        if n > TOPE_LINEA:
            return _linea_grande(i)
        total += n + 1
    if total > TOPE_FICHERO:
        return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _TEXTO_FICHERO_GRANDE)
    return None


def _fuera_de_tope_en_bytes(contenido: bytes) -> dict[str, Any] | None:
    """`_fuera_de_tope` sobre las líneas de unos bytes, sin mirarlas una a una: una línea pasa del tope si y solo si hay
    `TOPE_LINEA + 1` bytes seguidos sin `\n`, así que se salta de ventana en ventana al ÚLTIMO salto de cada una (64 pasos
    para 64 MiB, tengan las líneas el tamaño que tengan). El índice de la línea son los saltos de antes. Mismo veredicto."""
    n, pos = len(contenido), 0
    while n - pos > TOPE_LINEA:
        j = contenido.rfind(b"\n", pos, pos + TOPE_LINEA + 1)
        if j < 0:  # TOPE_LINEA + 1 bytes sin salto desde el principio de una línea
            return _linea_grande(contenido.count(b"\n", 0, pos))
        pos = j + 1
    total = n + (1 if n and contenido[n - 1] != 0x0A else 0)  # cada línea, más su `\n` (la última puede no llevarlo)
    if total > TOPE_FICHERO:
        return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _TEXTO_FICHERO_GRANDE)
    return None


_TEXTO_FICHERO_GRANDE = _t(f"el fichero pesa más de {TOPE_FICHERO >> 20} MiB, el tope de una traza (el servicio escribe un "
                           "fichero por día: compruébalos por separado)",
                           f"the file is over {TOPE_FICHERO >> 20} MiB, the limit for one trace (the service writes one file "
                           "per day: check them one by one)")


def _leer_linea(linea: str, i: int) -> dict:
    """Un bloque leído con el criterio estricto del formato y su FORMA comprobada; `TrazaIlegible` si no."""
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
    return b


def leer_bloques(lineas: Iterable[str]) -> list[dict]:
    """Los bloques de las líneas, leídos con el criterio estricto del formato (sin claves duplicadas, solo
    enteros, forma de v1) y SIN comprobar firmas. `TrazaIlegible` si una línea no se puede leer o pasa del tope."""
    lineas = list(lineas)
    fuera = _fuera_de_tope(_bytes_utf8(linea) for linea in lineas)
    if fuera is not None:
        raise TrazaIlegible(fuera["motivo"], fuera["bloque"], fuera["detalle"])
    return [_leer_linea(linea, i) for i, linea in enumerate(lineas)]


# ───────────────────────────── el comprobador ─────────────────────────────

def _fallo_de_firma(b: dict, utilizables: dict[str, bool] | None = None) -> tuple[str, dict[str, str]] | None:
    """`None` si la firma del bloque es buena; si no, `(motivo, detalle)` con motivo `clave` o `firma`. `utilizables`: lo ya
    sabido de cada clave pública (todos los bloques de una cadena llevan la misma)."""
    s = b["servicio"]
    crudo = bytes.fromhex(s["clave_publica"])
    if hashlib.sha256(crudo).hexdigest() != s["huella_clave"]:
        return "clave", _t("la huella no es la de la clave pública que lleva el bloque",
                           "the fingerprint is not the one of the public key the block carries")
    if utilizables is None:
        utilizables = {}
    if s["clave_publica"] not in utilizables:
        utilizables[s["clave_publica"]] = _clave_publica_utilizable(crudo)
    if not utilizables[s["clave_publica"]]:
        return "clave", _t("la clave pública no es una clave Ed25519 utilizable (no canónica, fuera de la curva o de orden "
                           "pequeño): con ella cualquier contenido parecería firmado",
                           "the public key is not a usable Ed25519 key (non-canonical, off the curve or of small order): with "
                           "it any content would look signed")
    valor = b["firma"]["valor"]
    buena = False
    if _base64_canonico(valor) and _firma_bien_formada(base64.b64decode(valor)):
        try:
            buena = _asim.comprobar(crudo, mensaje_firmado(b), valor)
        except _asim.ClaveInvalida:  # 32 bytes que la biblioteca no acepta: no firmó nada
            buena = False
    if not buena:
        return "firma", _t("la firma no coincide con el contenido del bloque", "the signature does not match the block's content")
    return None


def comprobar_traza(lineas: Iterable[str], huella_anclada: str | None = None,
                    esperar_genesis: bool = False) -> dict[str, Any]:
    """Comprueba una traza (una línea de texto por bloque) y dice QUÉ pasa, sin lanzar por culpa del
    contenido. `huella_anclada`: la huella de clave (64 hex) que quien comprueba esperaba, o `None`.
    `esperar_genesis`: exigir que la primera línea sea el génesis (seq 0, prev de ceros y SIN reinicio: primera importación).

    Devuelve `{estado, motivo, bloque, detalle, ...}`: `bloque` es el índice de la línea (desde 0) del primer
    fallo; `detalle` es `{"es", "en"}`. Si `estado == "valida"` lleva además `bloques, empieza_en, ultimo_seq,
    ultimo_hash, huellas, reinicios, cadenas, anclada` (`cadenas`: `{inicio, fin, seq_inicial, seq_final, huella}` por
    cadena, con los índices de línea desde 0). **Que sea válida NO dice que no falten los últimos bloques ni los de antes
    de un reinicio** (§4): `ultimo_seq` y `ultimo_hash` están para cotejarlos con lo que el servicio anuncia o con una
    importación anterior.

    Cada bloque se lee y se comprueba UNA vez y de él se guarda solo lo que la cadena necesita (no el bloque entero:
    I2, la memoria). El orden en que se informa del primer fallo es el de arriba, como si se hicieran pasadas.

    `ValueError` solo si `huella_anclada` no es una huella (64 hex minúscula): eso es un error de quien llama.
    """
    _huella_valida(huella_anclada)
    lineas = lineas if isinstance(lineas, (list, tuple)) else list(lineas)
    return _comprobar(lineas, _fuera_de_tope(_bytes_utf8(linea) for linea in lineas), huella_anclada, esperar_genesis)


def _huella_valida(huella_anclada: Any) -> None:
    if huella_anclada is not None and not (isinstance(huella_anclada, str) and _HEX64.match(huella_anclada)):
        raise ValueError("la huella anclada es SHA-256 en hexadecimal minúscula (64 caracteres)")


def _comprobar(lineas: Iterable[str], fuera: dict[str, Any] | None, huella_anclada: str | None,
               esperar_genesis: bool) -> dict[str, Any]:
    """El comprobador sobre líneas ya medidas (`fuera`: el resultado de los topes). `lineas` puede ser un generador: con un
    fichero, cada línea se decodifica AL LEERLA y se suelta (I2). Del primer fallo de firma y del primero de coherencia se
    guarda solo ese: tras uno de firma ya solo puede cambiar el veredicto un `formato` más abajo (no se verifica ni se guarda
    nada más), y tras uno de coherencia, una firma rota más abajo (no se guarda la cadena)."""
    if fuera is not None:
        return fuera
    con_firma = _asim.disponible()
    utilizables: dict[str, bool] = {}  # la comprobación de la clave, una vez por clave y no por bloque
    huellas_vistas: dict[str, str] = {}  # una sola copia de cada huella en los registros
    # (seq, prev, hash, huella, reinicio, desde, hasta): solo lo que la cadena necesita
    registros: list[tuple] = []
    primera_firma = primera_coherencia = None  # (índice, motivo, detalle)
    vistas = 0
    for i, linea in enumerate(lineas):
        try:
            b = _leer_linea(linea, i)
        except TrazaIlegible as e:
            return _res(NO_SE_PUDO_COMPROBAR, e.motivo, e.bloque, e.detalle)
        vistas += 1
        if primera_firma is not None or not con_firma:
            continue
        fallo = _fallo_de_firma(b, utilizables)
        if fallo is not None:
            primera_firma = (i, *fallo)
            registros = []
            continue
        if primera_coherencia is not None:
            continue
        try:
            _estructura(b, coherencia=True)
        except _Formato as e:
            primera_coherencia = (i, e.motivo, e.detalle)
            registros = []
            continue
        huella = huellas_vistas.setdefault(b["servicio"]["huella_clave"], b["servicio"]["huella_clave"])
        registros.append((b["seq"], b["prev"], hash_de_bloque(b), huella, b["reinicio"], b["ventana"]["desde"], b["ventana"]["hasta"]))
    if not vistas:
        return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _t("fichero sin bloques", "file without blocks"))
    if not con_firma:
        return _res(NO_SE_PUDO_COMPROBAR, "falta_cryptography", None, MOTIVOS["falta_cryptography"])
    if primera_firma is not None:
        return _res(NO_COINCIDE, primera_firma[1], primera_firma[0], primera_firma[2])
    if primera_coherencia is not None:
        return _res(NO_SE_PUDO_COMPROBAR, primera_coherencia[1], primera_coherencia[0], primera_coherencia[2])
    fallo = _fallo_de_cadena(registros, esperar_genesis)
    if fallo is not None:
        return fallo
    huellas: list[str] = []
    for r in registros:
        if r[3] not in huellas:
            huellas.append(r[3])
    if huella_anclada is not None:
        for i, r in enumerate(registros):
            if r[3] != huella_anclada:
                return _res(NO_COINCIDE, "ancla", i, _t("la huella de este bloque no es la anclada", "this block's fingerprint is not the anchored one"))
    reinicios = [i for i, r in enumerate(registros) if r[4] == "sin_previo"]
    inicios = sorted({0, *reinicios})
    cadenas = [{"inicio": a, "fin": z, "seq_inicial": registros[a][0], "seq_final": registros[z][0], "huella": registros[a][3]}
               for a, z in zip(inicios, [x - 1 for x in inicios[1:]] + [len(registros) - 1])]
    return _res(VALIDA, None, None, None, bloques=len(registros), empieza_en=registros[0][0],
                ultimo_seq=registros[-1][0], ultimo_hash=registros[-1][2],
                huellas=huellas, reinicios=reinicios, cadenas=cadenas, anclada=huella_anclada is not None)


def _fallo_de_cadena(registros: list[tuple], esperar_genesis: bool) -> dict[str, Any] | None:
    """La cadena, de un bloque al siguiente, y entre cadenas (B1): un bloque repetido, o una cadena nueva de una clave que
    empieza ANTES de que acabe la anterior de esa misma clave, es `cadena`. La hora no se comprueba (DS11): solo el ORDEN de
    las ventanas firmadas de una misma clave."""
    vistos: set[str] = set()
    ultimo_hasta: dict[str, str] = {}  # huella -> `hasta` del último bloque visto de esa clave
    for i, (seq, prev, h, huella, reinicio, desde, hasta) in enumerate(registros):
        # remate-c1 (menor 2): un fichero que EMPIEZA en un reinicio también tiene seq 0 y prev de ceros, pero delante pudo estar la
        # cadena desde el génesis entera: esperar el génesis es exigir además que no sea un reinicio (el escritor de C3 escribe el
        # génesis sin `reinicio`; `sin_previo` es solo cuando HABÍA traza y su última línea no se puede leer).
        if i == 0 and esperar_genesis and not (seq == 0 and prev == GENESIS and reinicio is None):
            return _res(NO_COINCIDE, "cadena", 0, _t(
                "se esperaba el bloque génesis (seq 0, prev de ceros, sin reinicio) y el fichero empieza más tarde o en un reinicio sin_previo",
                "the genesis block (seq 0, prev of zeros, no restart) was expected and the file starts later or at a sin_previo restart"))
        if h in vistos:
            return _res(NO_COINCIDE, "cadena", i, _t("este bloque ya estaba antes en el fichero (repetido)",
                                                     "this block was already earlier in the file (repeated)"))
        vistos.add(h)
        if reinicio == "sin_previo":
            if not (seq == 0 and prev == GENESIS):
                return _res(NO_COINCIDE, "cadena", i, _t("un reinicio sin_previo debe tener seq 0 y prev de ceros",
                                                         "a sin_previo restart must have seq 0 and a prev of zeros"))
            if huella in ultimo_hasta and desde < ultimo_hasta[huella]:
                return _res(NO_COINCIDE, "cadena", i, _t(
                    f"esta cadena nueva empieza ({desde}) antes de que acabe la anterior de la misma clave ({ultimo_hasta[huella]}): "
                    "cadenas cambiadas de orden o repetidas",
                    f"this new chain starts ({desde}) before the previous one of the same key ends ({ultimo_hasta[huella]}): "
                    "chains out of order or repeated"))
        elif (seq == 0) != (prev == GENESIS):
            return _res(NO_COINCIDE, "cadena", i, _t(
                "bloque incoherente: el génesis (seq 0) lleva prev de ceros y ningún otro lo lleva",
                "incoherent block: the genesis block (seq 0) has a prev of zeros and no other block has one"))
        elif i > 0:
            a = registros[i - 1]
            if seq != a[0] + 1:
                return _res(NO_COINCIDE, "cadena", i, _t(f"seq {seq} no sigue a {a[0]}", f"seq {seq} does not follow {a[0]}"))
            if prev != a[2]:
                return _res(NO_COINCIDE, "cadena", i, _t("prev no es el SHA-256 del bloque anterior", "prev is not the SHA-256 of the previous block"))
            if huella != a[3]:
                return _res(NO_COINCIDE, "cadena", i, _t("la clave cambia dentro de la cadena sin un reinicio sin_previo",
                                                         "the key changes inside the chain without a sin_previo restart"))
        ultimo_hasta[huella] = hasta
    return None


def comprobar_fichero(contenido: bytes | str, huella_anclada: str | None = None,
                      esperar_genesis: bool = False) -> dict[str, Any]:
    """Como `comprobar_traza`, sobre el contenido de un fichero `.jsonl` (bytes UTF-8 o texto). Con bytes, el tope del
    fichero se mira ANTES de decodificar, un BOM o un UTF-8 inválido son `formato` (como en el JS, `TextDecoder` estricto), y
    cada línea se decodifica al leerla, no el fichero entero."""
    _huella_valida(huella_anclada)
    if not isinstance(contenido, (bytes, bytearray, memoryview)):  # texto: las mismas líneas que `lineas_de_texto`, sin la lista
        texto = contenido
        return _comprobar((texto[a:z] for a, z in _tramos(texto, "\n")),
                          _fuera_de_tope(_bytes_utf8(texto[a:z]) for a, z in _tramos(texto, "\n")), huella_anclada, esperar_genesis)
    if len(contenido) > TOPE_FICHERO:
        return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _TEXTO_FICHERO_GRANDE)
    contenido = bytes(contenido)
    if not _es_utf8(contenido):
        return _res(NO_SE_PUDO_COMPROBAR, "formato", None, _t("el fichero no es UTF-8", "the file is not UTF-8"))
    return _comprobar((contenido[a:z].decode("utf-8") for a, z in _tramos(contenido, b"\n")),
                      _fuera_de_tope_en_bytes(contenido), huella_anclada, esperar_genesis)


def _es_utf8(contenido: bytes) -> bool:
    """UTF-8 estricto (sin sobrelargos ni sustitutos) sin crear el texto entero: a trozos de 1 MiB."""
    dec = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        for i in range(0, len(contenido), 1 << 20):
            dec.decode(contenido[i:i + (1 << 20)])
        dec.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


# ───────────────────────────── lo que se le dice a quien lee ─────────────────────────────

def describir(resultado: dict[str, Any], idioma: str = "es") -> list[str]:
    """Las líneas de pantalla para un resultado, en `es` o `en`, con el vocabulario cerrado de 145-D11. Si es
    válida, lleva el último bloque, las cadenas si hay reinicios, y lo que NO demuestra (§4: que falten los últimos
    bloques o los de antes de un reinicio, entre otras)."""
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
    nb, nc = resultado["bloques"], len(resultado["cadenas"])
    bloques = (f"{nb} bloque{'' if nb == 1 else 's'}" if es else f"{nb} block{'' if nb == 1 else 's'}")
    cadenas = (f"{nc} cadena{'' if nc == 1 else 's'}" if es else f"{nc} chain{'' if nc == 1 else 's'}")
    if resultado["reinicios"]:  # «del seq 0 al 0» con tres cadenas no diría nada: las cadenas se listan abajo
        lineas.append((f"{bloques} en {cadenas}" if es else f"{bloques} in {cadenas}"))
    else:
        lineas.append((f"{bloques}, del seq {resultado['empieza_en']} al {resultado['ultimo_seq']}" if es
                       else f"{bloques}, from seq {resultado['empieza_en']} to {resultado['ultimo_seq']}"))
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
        donde = ", ".join(str(i + 1) for i in resultado["reinicios"])
        una = len(resultado["reinicios"]) == 1
        lineas.append((f"{cadenas} (reinicio sin_previo en {'la línea' if una else 'las líneas'} {donde}):" if es else
                       f"{cadenas} (sin_previo restart at {'line' if una else 'lines'} {donde}):"))
        for c in resultado["cadenas"]:
            lineas.append((f"- líneas {c['inicio'] + 1}–{c['fin'] + 1}: seq {c['seq_inicial']}–{c['seq_final']}, clave {c['huella'][:16]}…"
                           if es else
                           f"- lines {c['inicio'] + 1}–{c['fin'] + 1}: seq {c['seq_inicial']}–{c['seq_final']}, key {c['huella'][:16]}…"))
        lineas.append(("En cada reinicio pueden faltar los bloques de antes: una cadena nueva no enlaza con la anterior." if es
                       else "At each restart the earlier blocks may be missing: a new chain does not link to the previous one."))
    if len(resultado["huellas"]) > 1:
        lineas.append(("Hay más de una clave en el fichero." if es else "There is more than one key in the file."))
    lineas.append(("Lo que esto NO demuestra:" if es else "What this does NOT show:"))
    lineas.extend("- " + frase for frase in LO_QUE_NO_DEMUESTRA[idioma])
    return lineas
