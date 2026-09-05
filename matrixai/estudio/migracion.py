# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Versiones de esquema y política de migración — 104-C0.

LA POLÍTICA, en una frase: **un documento de una versión anterior conocida se
lee y se MARCA; uno de una versión desconocida se rechaza**. No se borra
ninguno de los dos.

Las dos mitades importan y son asimétricas a propósito:

* Leer y marcar. Un proyecto guardado hace tres meses tiene que abrirse. Y tiene
  que verse que se abrió migrado: la marca (`schema_migration`) dice de qué
  versión venía y por qué pasos se subió, para que nadie atribuya a un documento
  antiguo campos que rellenó la migración. *Una nota vieja miente igual que un
  dato falso.*
* Rechazar lo desconocido. Un documento en `9.9` —o en una versión que este core
  no tiene— no se interpreta «lo que se pueda»: eso es adivinar qué significa un
  campo nuevo, que es justo lo que el `schema_version` existe para evitar. Es la
  misma decisión que toma `reproduce.py` con las versiones de `run_provenance`.

HOY NO HAY NINGUNA MIGRACIÓN REGISTRADA, y se dice aquí en vez de dejarlo
adivinar: `1.0` es la primera versión de estos esquemas, así que el catálogo por
omisión está vacío. El mecanismo existe desde el primer día porque añadirlo
después obliga a decidir qué hacer con los documentos ya escritos, y para
entonces ya hay proyectos guardados.

El digest de un documento NO incluye la marca de migración: la marca cuenta cómo
se leyó, no lo que dice. Si entrara, el mismo plan de partición tendría un
digest distinto según se hubiera cargado de un fichero antiguo o de uno nuevo, y
los vínculos artefacto-evaluación dejarían de casar por un motivo que no tiene
nada que ver con su contenido.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from matrixai.estudio.errores import EsquemaInvalido, MigracionImposible
from matrixai.estudio.validacion import exigir_mapa, exigir_texto

__all__ = [
    "CLAVE_DE_MARCA", "ESTUDIO_SCHEMA_VERSION", "MIGRACIONES",
    "RegistroDeMigraciones",
]

#: La versión de TODOS los esquemas de este paquete. Una sola para el conjunto:
#: se publican y se consumen juntos, y versionarlos por separado obligaría a
#: cada consumidor a llevar una matriz de compatibilidad.
ESTUDIO_SCHEMA_VERSION = "1.0"

#: Dónde vive la marca dentro del documento.
CLAVE_DE_MARCA = "schema_migration"

Paso = Callable[[dict[str, Any]], dict[str, Any]]


class RegistroDeMigraciones:
    """Los caminos de una versión a la siguiente, por esquema.

    Es una CLASE y no un diccionario de módulo para que las pruebas —y quien
    quiera experimentar— puedan tener el suyo sin tocar el del producto. Un
    registro global mutable convierte cualquier prueba en un efecto colateral
    para la siguiente.
    """

    def __init__(self, version_actual: str = ESTUDIO_SCHEMA_VERSION) -> None:
        self.version_actual = version_actual
        self._pasos: dict[tuple[str, str], tuple[str, Paso]] = {}

    # -- registro --------------------------------------------------------
    def registrar(self, esquema: str, desde: str, hasta: str, funcion: Paso) -> None:
        """Un paso `desde` -> `hasta` para un esquema.

        Registrar dos veces el mismo origen se rechaza: dos caminos desde la
        misma versión es un fallo de cableado, y elegir uno «por orden de
        registro» haría que el resultado dependiera del orden de los imports.
        """
        exigir_texto(esquema, "esquema")
        exigir_texto(desde, "desde")
        exigir_texto(hasta, "hasta")
        if desde == hasta:
            raise EsquemaInvalido("sin_camino_de_migracion", campo=esquema,
                                  valor=desde, opciones=hasta)
        if (esquema, desde) in self._pasos:
            raise EsquemaInvalido("migracion_ya_registrada", campo=esquema,
                                  valor=desde)
        self._pasos[(esquema, desde)] = (hasta, funcion)

    # -- consulta --------------------------------------------------------
    def versiones_legibles(self, esquema: str) -> tuple[str, ...]:
        """Las versiones de las que se puede llegar a la actual, la actual
        incluida. Es lo que se escribe en el motivo cuando se rechaza una: un
        «versión inválida» sin la lista obliga a leerse el código."""
        legibles = {self.version_actual}
        for (esq, desde), (hasta, _) in self._pasos.items():
            if esq != esquema:
                continue
            visto = {desde}
            actual = hasta
            while actual != self.version_actual and (esq, actual) in self._pasos:
                if actual in visto:
                    break
                visto.add(actual)
                actual = self._pasos[(esq, actual)][0]
            if actual == self.version_actual:
                legibles.add(desde)
        return tuple(sorted(legibles))

    # -- lectura ---------------------------------------------------------
    def preparar(self, esquema: str, payload: Mapping[str, Any]
                 ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Devuelve el documento en la versión actual y su marca (o `None`).

        `None` en la marca significa **el documento ya venía en la versión
        actual**, no «no se sabe»: es lo único que puede significar, porque un
        documento sin `schema_version` se rechaza antes de llegar aquí.
        """
        documento = exigir_mapa(payload, esquema)
        version = documento.get("schema_version")
        if not isinstance(version, str) or not version.strip():
            raise EsquemaInvalido("falta_campo", campo=f"{esquema}.schema_version")

        if version == self.version_actual:
            marca = documento.get(CLAVE_DE_MARCA)
            return dict(documento), (dict(marca) if isinstance(marca, Mapping) else None)

        cuerpo = dict(documento)
        cadena: list[str] = []
        actual = version
        visitadas = {version}
        while actual != self.version_actual:
            paso = self._pasos.get((esquema, actual))
            if paso is None:
                raise MigracionImposible(
                    "version_no_legible", campo=esquema, valor=version,
                    opciones=list(self.versiones_legibles(esquema)))
            siguiente, funcion = paso
            cuerpo = dict(funcion(dict(cuerpo)))
            cuerpo["schema_version"] = siguiente
            cadena.append(f"{actual}->{siguiente}")
            if siguiente in visitadas:  # pragma: no cover - un ciclo es un fallo de registro
                raise MigracionImposible(
                    "sin_camino_de_migracion", campo=esquema, valor=version,
                    opciones=self.version_actual)
            visitadas.add(siguiente)
            actual = siguiente

        marca = {"desde": version, "hasta": self.version_actual, "pasos": cadena}
        cuerpo[CLAVE_DE_MARCA] = marca
        return cuerpo, marca


#: El registro del producto. Vacío a día de hoy —`1.0` es la primera versión— y
#: eso es un hecho, no un olvido: cuando exista `1.1`, aquí se registra el paso
#: `1.0 -> 1.1` y los proyectos antiguos siguen abriéndose.
MIGRACIONES = RegistroDeMigraciones()
