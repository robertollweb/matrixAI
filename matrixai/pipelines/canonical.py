"""Canonicalización JCS (RFC 8785) — contrato 81-C1.

Sin una forma canónica, «firmar» no significa nada: el mismo contenido
serializado de dos maneras da dos firmas, y una firma buena parece mala.

**Esto NO sustituye a `canonical_json` del contrato 82.** Aquél usa
`ensure_ascii=True` y ordena por code points de Python; JCS exige UTF-8
real y orden por unidades UTF-16. Y no se puede cambiar el otro: sus
digests viven dentro de manifiestos ya emitidos, así que tocarlo
invalidaría paquetes que hoy verifican. Dos usos distintos que conviven,
cada uno documentado.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["jcs_canonical", "jcs_bytes"]

#: Escapes que el RFC 8785 exige en forma corta. El resto de controles van
#: como `\uXXXX` en minúsculas.
_CORTOS = {
    '"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f",
    "\n": "\\n", "\r": "\\r", "\t": "\\t",
}


def _cadena(valor: str) -> str:
    piezas = ['"']
    for caracter in valor:
        corto = _CORTOS.get(caracter)
        if corto is not None:
            piezas.append(corto)
        elif caracter < "\x20":
            piezas.append(f"\\u{ord(caracter):04x}")
        else:
            # UTF-8 tal cual: escapar aquí daría otro digest que ningún
            # verificador que siga el RFC reproduciría.
            piezas.append(caracter)
    piezas.append('"')
    return "".join(piezas)


def _numero(valor: float | int) -> str:
    """Un número **como lo escribe ECMAScript**, que es lo que exige JCS.

    `json.dumps(1.0)` da `1.0` y ECMAScript da `1`: dos digests distintos
    para el mismo número. Y `NaN`/`Infinity` no existen en JSON — se
    cortan aquí en vez de firmar algo que nadie más puede leer.

    **AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: esto usaba las reglas
    de Python y NO las de ECMAScript.** Medido: `1e-6` salía `1e-6` y
    ECMAScript escribe `0.000001`; `1.2345678901234568e+20` salía
    `123456789012345683968` y ECMAScript escribe `123456789012345680000`.
    Una firma hecha sobre esos bytes **no es verificable por nadie más**,
    que es exactamente lo único que JCS existe para garantizar — y era la
    razón entera de la decisión §0.1.

    Se implementa la regla del §7.1.12.1 de ECMAScript, que es la que el
    RFC 8785 cita: con `k` dígitos significativos y el punto decimal en la
    posición `n`,

    * `k <= n <= 21` → los dígitos y `n - k` ceros;
    * `0 < n <= 21`  → los dígitos con el punto tras `n` de ellos;
    * `-6 < n <= 0`  → `0.`, luego `-n` ceros, luego los dígitos;
    * en otro caso   → exponencial con exponente `n - 1`.

    Los umbrales son lo que Python hace distinto: cambia a exponencial en
    `1e-5` y ECMAScript aguanta hasta `1e-7`.
    """
    if isinstance(valor, bool):  # `bool` es `int` en Python
        return "true" if valor else "false"
    if isinstance(valor, int):
        return str(valor)
    if math.isnan(valor) or math.isinf(valor):
        raise ValueError(
            f"{valor!r} no es un número JSON: NaN e Infinity no existen en JSON "
            "y firmarlos produciría bytes que ningún otro verificador puede leer")
    if valor == 0.0:
        # `-0.0` se escribe `0` en ECMAScript: `repr` daría `-0.0` y sería
        # otro digest para el mismo número.
        return "0"

    negativo = valor < 0
    # `repr` da los dígitos significativos MÁS CORTOS que round-trip, que es
    # justo lo que pide ECMAScript; lo que hay que rehacer es la FORMA.
    mantisa, _, exponente = f"{abs(valor):.17e}".partition("e")
    corto = repr(abs(valor))
    digitos = corto.replace(".", "").replace("-", "").lstrip("0")
    if "e" in corto or "E" in corto:
        base, _, exp = corto.lower().partition("e")
        digitos = base.replace(".", "").rstrip("0") or "0"
        n = int(exp) + (len(base.split(".")[0].lstrip("-")))
    else:
        entera, _, decimal = corto.partition(".")
        entera = entera.lstrip("-")
        if entera == "0":
            sin_ceros = decimal.lstrip("0")
            n = -(len(decimal) - len(sin_ceros))
            digitos = sin_ceros.rstrip("0") or "0"
        else:
            n = len(entera)
            digitos = (entera + decimal).rstrip("0") or "0"
    digitos = digitos.rstrip("0") or "0"
    k = len(digitos)

    if k <= n <= 21:
        texto = digitos + "0" * (n - k)
    elif 0 < n <= 21:
        texto = digitos[:n] + "." + digitos[n:]
    elif -6 < n <= 0:
        texto = "0." + "0" * (-n) + digitos
    else:
        e = n - 1
        cuerpo = digitos if k == 1 else digitos[0] + "." + digitos[1:]
        texto = f"{cuerpo}e{'+' if e >= 0 else '-'}{abs(e)}"
    return ("-" + texto) if negativo else texto


def _clave_utf16(clave: str) -> list[int]:
    """El orden del RFC: por unidades de código UTF-16, no por code point.

    No es lo mismo: un emoji (U+1F600) se codifica como el par subrogado
    0xD83D 0xDE00, así que ordena ANTES que U+E000 en UTF-16 y DESPUÉS por
    code point. Un verificador que ordene distinto rechaza firmas buenas.
    """
    return [unidad for unidad in clave.encode("utf-16-be")]


def _valor(dato: Any) -> str:
    if dato is None:
        return "null"
    if isinstance(dato, bool):
        return "true" if dato else "false"
    if isinstance(dato, (int, float)):
        return _numero(dato)
    if isinstance(dato, str):
        return _cadena(dato)
    if isinstance(dato, (list, tuple)):
        return "[" + ",".join(_valor(x) for x in dato) + "]"
    if isinstance(dato, dict):
        claves = sorted(dato.keys(), key=_clave_utf16)
        return "{" + ",".join(f"{_cadena(str(k))}:{_valor(dato[k])}" for k in claves) + "}"
    raise ValueError(
        f"{type(dato).__name__} no es un tipo JSON: canonicalizarlo exigiría "
        "inventar una representación, y una firma sobre algo inventado no "
        "prueba nada")


def jcs_canonical(payload: Any) -> str:
    """La forma canónica JCS del objeto, como texto."""
    return _valor(payload)


def jcs_bytes(payload: Any) -> bytes:
    """Los BYTES que se firman. Es lo que hay que guardar y verificar:
    reparsear y volver a serializar antes de comprobar una firma es donde
    se cuelan las diferencias (P23-R-0016)."""
    return jcs_canonical(payload).encode("utf-8")
