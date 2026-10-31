"""Firma ASIMÉTRICA Ed25519 — contrato 145-C1 (Ola 2, idea 30).

Las primitivas, en un solo sitio: la usan el recibo (`receipt.py`) y, en los
cortes siguientes, la atestación y el Informe Vivo. Dos sitios con su propia
idea de «qué es una huella» o «cómo se carga una clave» acabarían divergiendo.

**Lo que una firma válida de aquí demuestra, y lo que no** (§2 del contrato):
que los bytes firmados los produjo quien tiene la clave PRIVADA de esa huella
y que nadie los ha tocado después. NO demuestra quién es esa persona, ni que el
modelo sea correcto, ni la hora (la pone el firmante), y NO hay revocación.

**La huella de la clave pública — formato estable** (no cambia sin subir de
versión el contrato):

    huella = SHA-256 de los 32 BYTES CRUDOS de la clave pública Ed25519
             (RFC 8032, 4.º punto), en hexadecimal minúscula, 64 caracteres.

Los bytes crudos y no el PEM/SPKI: el mismo par de claves da la misma huella
da igual cómo se haya guardado o transportado, y se calcula igual en
`crypto.subtle` (`exportKey('raw')` + `digest('SHA-256')`), que es lo que hará
la página de verificación (C4).

**Tres estados, y solo tres** (D11): `valida`, `no_coincide`,
`no_se_pudo_comprobar`. «No se pudo comprobar» no es una acusación ni un
aprobado. Las palabras que se enseñan a quien lo lea están en `VOCABULARIO`
(es/en) y nunca dicen «verificado», «auténtico» ni «certificado» a secas.

**`cryptography` es opcional** (extra `[firma]`). Sin ella, firmar da un error
claro (`FirmaNoDisponible`) y comprobar dice «no se pudo comprobar: falta
cryptography». Nada del resto del paquete la necesita.

**La clave privada nunca sale de aquí por su cuenta**: ninguna función la
imprime, la registra ni la mete en un mensaje de error. Quien la guarda
(`generar_par`) la devuelve como bytes PEM y es quien decide dónde escribirla.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Any

__all__ = [
    "ALGORITMO", "ClaveInvalida", "ESTADOS", "FirmaNoDisponible", "NO_COINCIDE",
    "NO_SE_PUDO_COMPROBAR", "VALIDA", "VOCABULARIO", "cargar_clave_privada",
    "cargar_clave_publica", "clave_publica_pem", "comprobar", "disponible",
    "firmar", "generar_par", "huella_de_clave_publica", "texto_del_estado",
]

#: Cómo se llama el algoritmo DENTRO del sobre (`"alg"`), para poder migrar a
#: otro sin adivinar qué firma es cuál.
ALGORITMO = "ed25519"

VALIDA = "valida"
NO_COINCIDE = "no_coincide"
NO_SE_PUDO_COMPROBAR = "no_se_pudo_comprobar"
ESTADOS = (VALIDA, NO_COINCIDE, NO_SE_PUDO_COMPROBAR)

#: D11. Lo que se le dice a quien lee. NUNCA «verificado», «auténtico» ni
#: «certificado» a secas: lo barre una prueba (es/en, estas cadenas incluidas).
VOCABULARIO: dict[str, dict[str, str]] = {
    "es": {
        VALIDA: "firma válida",
        NO_COINCIDE: "la firma no coincide",
        NO_SE_PUDO_COMPROBAR: "no se pudo comprobar",
    },
    "en": {
        VALIDA: "valid signature",
        NO_COINCIDE: "the signature does not match",
        NO_SE_PUDO_COMPROBAR: "could not be checked",
    },
}


def texto_del_estado(estado: str, idioma: str = "es") -> str:
    """La frase cerrada de D11 para un estado, en el idioma pedido."""
    return VOCABULARIO.get(idioma, VOCABULARIO["en"])[estado]


class FirmaNoDisponible(RuntimeError):
    """Falta `cryptography`: no se puede firmar ni comprobar con Ed25519."""


class ClaveInvalida(ValueError):
    """La clave no se pudo leer. El mensaje NUNCA lleva el material de la clave."""


_FALTA = ("falta cryptography, que la firma Ed25519 necesita: instálala con "
          "`pip install 'matrixai[firma]'` (o `pip install cryptography`)")


def _modulos():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - lo cubre la prueba con el módulo bloqueado
        raise FirmaNoDisponible(_FALTA) from exc
    return InvalidSignature, serialization, ed25519


def disponible() -> bool:
    """¿Está instalada `cryptography`?"""
    try:
        _modulos()
    except FirmaNoDisponible:
        return False
    return True


def _crudo(publica: Any) -> bytes:
    _, serialization, _ = _modulos()
    return publica.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def huella_de_clave_publica(clave: Any) -> str:
    """La huella (formato del docstring del módulo) de una clave pública o privada.

    Acepta lo mismo que `cargar_clave_publica`, incluida una privada (se mira
    SOLO su pública).
    """
    return hashlib.sha256(_crudo(cargar_clave_publica(clave))).hexdigest()


def generar_par() -> tuple[bytes, str]:
    """Un par nuevo: `(privada_pem, publica_pem)`.

    La privada va en PKCS#8 SIN cifrar (bytes): quien la recibe decide dónde
    escribirla (en `/config` con 0600, C2). La pública, en SPKI.
    """
    _, serialization, ed25519 = _modulos()
    privada = ed25519.Ed25519PrivateKey.generate()
    pem_privada = privada.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())
    return pem_privada, clave_publica_pem(privada.public_key())


def clave_publica_pem(clave: Any) -> str:
    """La pública en PEM/SPKI (la que se publica y se cotea por otro canal)."""
    _, serialization, _ = _modulos()
    return cargar_clave_publica(clave).public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")


_HEX64 = re.compile(r"\A[0-9a-fA-F]{64}\Z")


def cargar_clave_privada(origen: Any):
    """Una `Ed25519PrivateKey` desde un objeto ya cargado, PEM (PKCS#8) o 32 bytes crudos."""
    _, serialization, ed25519 = _modulos()
    if isinstance(origen, ed25519.Ed25519PrivateKey):
        return origen
    datos = origen.encode("utf-8") if isinstance(origen, str) else origen
    if not isinstance(datos, (bytes, bytearray)):
        raise ClaveInvalida("la clave privada no es ni un objeto de clave, ni un PEM, ni 32 bytes")
    datos = bytes(datos)
    try:
        if len(datos) == 32:
            return ed25519.Ed25519PrivateKey.from_private_bytes(datos)
        clave = serialization.load_pem_private_key(datos.strip(), password=None)
    except (ValueError, TypeError) as exc:
        # NO se reenvía `exc`: aunque hoy su texto no lleve la clave, un mensaje
        # de error es lo primero que acaba en un log.
        raise ClaveInvalida("no se pudo leer la clave privada (se espera un PEM PKCS#8 "
                            "de Ed25519 sin cifrar)") from None
    if not isinstance(clave, ed25519.Ed25519PrivateKey):
        raise ClaveInvalida("la clave privada no es de Ed25519")
    return clave


def cargar_clave_publica(origen: Any):
    """Una `Ed25519PublicKey` desde: un objeto de clave (pública o privada), un
    PEM público (SPKI) o privado (se toma su pública), los 32 bytes crudos, o
    su texto en hexadecimal (64 caracteres) o base64."""
    _, serialization, ed25519 = _modulos()
    if isinstance(origen, ed25519.Ed25519PublicKey):
        return origen
    if isinstance(origen, ed25519.Ed25519PrivateKey):
        return origen.public_key()
    if isinstance(origen, str):
        origen = origen.encode("utf-8")
    if not isinstance(origen, (bytes, bytearray)):
        raise ClaveInvalida("la clave pública no es ni un objeto de clave, ni un PEM, ni 32 bytes")
    datos = bytes(origen)
    if len(datos) == 32:
        try:
            return ed25519.Ed25519PublicKey.from_public_bytes(datos)
        except ValueError:
            raise ClaveInvalida("los 32 bytes no son una clave pública de Ed25519") from None
    texto = datos.strip()
    try:
        if texto.startswith(b"-----BEGIN"):
            if b"PRIVATE KEY" in texto.split(b"\n", 1)[0]:
                return cargar_clave_privada(texto).public_key()
            clave = serialization.load_pem_public_key(texto)
            if not isinstance(clave, ed25519.Ed25519PublicKey):
                raise ClaveInvalida("la clave pública no es de Ed25519")
            return clave
        cadena = texto.decode("ascii")
        if _HEX64.match(cadena):
            return ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(cadena))
        crudo = base64.b64decode(cadena, validate=True)
        if len(crudo) == 32:
            return ed25519.Ed25519PublicKey.from_public_bytes(crudo)
    except ClaveInvalida:
        raise
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error):
        pass
    raise ClaveInvalida("no se pudo leer la clave pública (se espera un PEM, o sus 32 bytes "
                        "en hexadecimal o base64)")


def firmar(clave_privada: Any, mensaje: bytes) -> str:
    """La firma Ed25519 de `mensaje`, en base64 estándar (64 bytes → 88 caracteres)."""
    return base64.b64encode(cargar_clave_privada(clave_privada).sign(mensaje)).decode("ascii")


def comprobar(clave_publica: Any, mensaje: bytes, firma_b64: Any) -> bool:
    """¿Esa firma es de esa clave sobre ese mensaje? Una firma mal formada es `False`."""
    invalida, _, _ = _modulos()
    if not isinstance(firma_b64, str):
        return False
    try:
        firma = base64.b64decode(firma_b64, validate=True)
    except (ValueError, binascii.Error):
        return False
    if len(firma) != 64:
        return False
    try:
        cargar_clave_publica(clave_publica).verify(firma, mensaje)
    except invalida:
        return False
    return True
