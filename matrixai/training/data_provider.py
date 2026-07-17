# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — interfaz `DataProvider` +
registro. Un proveedor sabe convertir SU fuente (local determinista o una
API pública concreta) en un CSV canónico compatible con C1
(`analyze_dataset_csv`) — nunca decide esquema/target, eso sigue siendo
C1/C2/el usuario (invariante 8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class DataProviderError(Exception):
    """Cualquier fallo de un `DataProvider` — mensaje siempre accionable
    (invariante 7: fallo externo limpio, cero estado a medias)."""


@dataclass(frozen=True)
class LicenseInfo:
    name: str
    url: str
    summary: str
    requires_attribution: bool
    commercial_use_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "summary": self.summary,
            "requires_attribution": self.requires_attribution,
            "commercial_use_allowed": self.commercial_use_allowed,
        }


@dataclass(frozen=True)
class ProviderMetadata:
    provider_id: str
    display_name: str
    description: str
    requires_network: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "description": self.description,
            "requires_network": self.requires_network,
        }


@dataclass(frozen=True)
class DownloadEstimate:
    estimated_rows: int | None
    estimated_bytes: int | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_rows": self.estimated_rows,
            "estimated_bytes": self.estimated_bytes,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DownloadResult:
    csv_text: str
    rows: int
    columns: list[str]
    source_url: str | None
    fetched_at: str
    license_info: LicenseInfo
    provenance_extra: dict[str, Any]


@runtime_checkable
class DataProvider(Protocol):
    provider_id: str

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Errores accionables (lista vacía si `config` es válida) — nunca
        toca la red."""
        ...

    def check_availability(self) -> bool:
        """¿Responde la fuente externa ahora mismo? `True` para un
        proveedor sin red (`synthetic_local`)."""
        ...

    def get_metadata(self) -> ProviderMetadata: ...

    def get_license_info(self) -> LicenseInfo: ...

    def estimate_download(self, config: dict[str, Any]) -> DownloadEstimate:
        """Estimación SIN descargar de verdad — puede tocar la red para
        una petición ligera (p.ej. HEAD/metadata), nunca el cuerpo
        completo."""
        ...

    def download(self, config: dict[str, Any], *, license_accepted: bool) -> DownloadResult:
        """Lanza `DataProviderError` si `license_accepted` es falso —
        DEBE ser la primera comprobación, antes de cualquier petición de
        red (invariante 2, ver `require_license_accepted`)."""
        ...


def require_license_accepted(license_accepted: bool, provider_id: str) -> None:
    """Invariante 2 (bloqueante): sin aceptación explícita registrada NO
    se emite ninguna petición de red. Cada `download()` de cada proveedor
    llama a esto como PRIMERA línea, antes de tocar `secure_fetch` —
    centralizado aquí para que ningún proveedor pueda "olvidarlo" con un
    mensaje distinto o, peor, sin lanzar nada."""
    if not license_accepted:
        raise DataProviderError(
            f"El proveedor {provider_id!r} exige aceptar su licencia antes de "
            "descargar — pásalo con license_accepted=True tras mostrar "
            "get_license_info() al usuario y registrar su aceptación."
        )


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, DataProvider] = {}

    def register(self, provider: DataProvider) -> None:
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> DataProvider:
        try:
            return self._providers[provider_id]
        except KeyError:
            raise DataProviderError(
                f"Proveedor de datos {provider_id!r} desconocido. "
                f"Proveedores registrados: {sorted(self._providers)}."
            ) from None

    def list_providers(self) -> list[ProviderMetadata]:
        return [p.get_metadata() for p in self._providers.values()]


_default_registry: ProviderRegistry | None = None


def get_default_registry() -> ProviderRegistry:
    """Singleton perezoso — los proveedores concretos se importan aquí
    dentro (no arriba del módulo) para que este fichero, la interfaz, no
    dependa de ninguna implementación concreta."""
    global _default_registry
    if _default_registry is None:
        from matrixai.training.provider_open_meteo import OpenMeteoProvider
        from matrixai.training.provider_stooq import StooqProvider
        from matrixai.training.provider_synthetic_local import SyntheticLocalProvider

        registry = ProviderRegistry()
        registry.register(SyntheticLocalProvider())
        registry.register(OpenMeteoProvider())
        registry.register(StooqProvider())
        _default_registry = registry
    return _default_registry
