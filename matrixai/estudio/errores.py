# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los rechazos de este paquete — 104-C0.

**Excepción y no un campo `valido: false`**, por la misma razón que el 82-C1:
un valor AUSENTE es una respuesta legítima (un modelo sin receta, una métrica
indefinida, un registro sin `y_true`) y se publica con su motivo; un valor
IMPOSIBLE —una clase positiva que no está entre las clases, una partición que
mete al mismo paciente en desarrollo y en prueba— no es una respuesta: es un
fallo de cableado de quien construye el documento, y publicarlo como «no
válido» lo dejaría salir por la puerta dentro de un campo que ya nadie mira.

Cada excepción lleva su motivo **en los dos idiomas** (`.es` / `.en`) y la clave
del catálogo (`.clave`), para que quien la enseñe no tenga que reconocer la
frase con una expresión regular. `str(exc)` da el inglés, que es lo que se lee
en un traceback y lo que hace el resto del repositorio.
"""

from __future__ import annotations

from matrixai.estudio.textos import motivo as _motivo

__all__ = [
    "ErrorDeEstudio",
    "EsquemaInvalido",
    "FugaDeTest",
    "MigracionImposible",
    "ProtocoloRoto",
]


class ErrorDeEstudio(ValueError):
    """Raíz de todo lo que este paquete rechaza, con motivo escrito y bilingüe."""

    def __init__(self, clave: str, **campos: object) -> None:
        textos = _motivo(clave, **campos)
        super().__init__(textos["en"])
        self.clave = clave
        self.campos = dict(campos)
        self.es = textos["es"]
        self.en = textos["en"]

    def motivo(self, locale: str = "es") -> str:
        """El motivo en el idioma que se pida; el que no se sepa, en inglés."""
        idioma = str(locale or "en").strip().lower()
        return self.es if idioma == "es" else self.en

    @property
    def bilingue(self) -> dict[str, str]:
        """Las dos redacciones, para guardarlas en un registro o un recibo."""
        return {"es": self.es, "en": self.en}


class EsquemaInvalido(ErrorDeEstudio):
    """El documento no puede existir con esa forma."""


class ProtocoloRoto(ErrorDeEstudio):
    """La secuencia del protocolo no se respeta (evaluar sin congelar, etc.)."""


class FugaDeTest(ProtocoloRoto):
    """Alguien ha intentado LEER la prueba reservada donde no puede.

    Es una subclase de `ProtocoloRoto` a propósito: quien quiera cazar
    cualquier ruptura del protocolo la caza, y quien quiera contar
    específicamente las fugas también.
    """


class MigracionImposible(ErrorDeEstudio):
    """El documento viene en una versión que este core no sabe leer.

    Ojo a la asimetría, que es la política entera: una versión ANTERIOR
    conocida se lee y se marca; una desconocida —o una del futuro— se rechaza.
    Interpretar a medias un documento que no se entiende es peor que no leerlo.
    """
