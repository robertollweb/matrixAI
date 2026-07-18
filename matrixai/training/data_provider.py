# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — interfaz `DataProvider` +
registro + registro auditable de aceptación de licencia. Un proveedor
sabe convertir SU fuente (local determinista o una API pública concreta)
en un CSV canónico compatible con C1 (`analyze_dataset_csv`) — nunca
decide esquema/target, eso sigue siendo C1/C2/el usuario (invariante 8).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
    summary_i18n: dict[str, str] | None = None

    def localized_summary(self, locale: str | None = None) -> str:
        normalized = str(locale or "es").strip().lower()
        if normalized != "es" and self.summary_i18n:
            translated = self.summary_i18n.get(normalized)
            if isinstance(translated, str) and translated.strip():
                return translated
        return self.summary

    def to_dict(self, *, locale: str | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "summary": self.localized_summary(locale),
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
    notes_i18n: dict[str, str] | None = None

    def localized_notes(self, locale: str | None = None) -> str:
        normalized = str(locale or "es").strip().lower()
        if normalized != "es" and self.notes_i18n:
            translated = self.notes_i18n.get(normalized)
            if isinstance(translated, str) and translated.strip():
                return translated
        return self.notes

    def to_dict(self, *, locale: str | None = None) -> dict[str, Any]:
        return {
            "estimated_rows": self.estimated_rows,
            "estimated_bytes": self.estimated_bytes,
            "notes": self.localized_notes(locale),
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


@dataclass(frozen=True)
class LicenseAcceptance:
    """Recibo AUDITABLE de una aceptación de licencia — no un booleano
    efímero (auditoría 2026-07-17 [MEDIA]). `license_digest` fija los
    términos EXACTOS aceptados: si el proveedor cambia su `LicenseInfo`
    después, el digest deja de coincidir y el recibo deja de ser válido
    (`require_valid_acceptance`) — "la versión actual de la licencia",
    no solo "alguna vez aceptaste algo de este proveedor"."""

    acceptance_id: str
    provider_id: str
    license_name: str
    license_url: str
    license_digest: str
    accepted_at: str
    actor: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_id": self.acceptance_id,
            "provider_id": self.provider_id,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "license_digest": self.license_digest,
            "accepted_at": self.accepted_at,
            "actor": self.actor,
        }


def _license_digest(info: LicenseInfo) -> str:
    raw = json.dumps(info.to_dict(), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LicenseAcceptanceStore:
    """Registro auditable de aceptaciones de licencia (invariante 2):
    quién aceptó qué licencia, de qué proveedor, cuándo, y con qué
    versión EXACTA de los términos — nunca un booleano efímero sin
    rastro. `download()` de cada proveedor exige un `LicenseAcceptance`
    válido (mismo proveedor, digest vigente) antes de tocar la red.

    v1: registro EN MEMORIA (vive mientras el proceso del backend esté
    arriba) — basta para bloquear descargas sin aceptación dentro de una
    sesión/proceso, que es lo que exige la invariante 2 tal como está
    redactada. Persistir entre reinicios (fichero/SQLite) es una
    extensión de infraestructura, documentada como pendiente, no un
    cambio de este contrato."""

    def __init__(self) -> None:
        self._acceptances: dict[str, LicenseAcceptance] = {}

    def record(self, provider: "DataProvider", *, actor: str) -> LicenseAcceptance:
        if not actor or not actor.strip():
            raise DataProviderError("actor es obligatorio para registrar una aceptación de licencia.")
        info = provider.get_license_info()
        acceptance = LicenseAcceptance(
            acceptance_id=str(uuid.uuid4()),
            provider_id=provider.provider_id,
            license_name=info.name,
            license_url=info.url,
            license_digest=_license_digest(info),
            accepted_at=_utcnow_iso(),
            actor=actor.strip(),
        )
        self._acceptances[acceptance.acceptance_id] = acceptance
        return acceptance

    def get(self, acceptance_id: str) -> LicenseAcceptance | None:
        return self._acceptances.get(acceptance_id)

    def verify(self, acceptance_id: str, provider: "DataProvider") -> LicenseAcceptance:
        acceptance = self._acceptances.get(acceptance_id)
        if acceptance is None:
            raise DataProviderError(
                f"acceptance_id {acceptance_id!r} no existe — acepta la licencia de "
                f"{provider.provider_id!r} antes de descargar."
            )
        require_valid_acceptance(acceptance, provider)
        return acceptance


def require_valid_acceptance(acceptance: "LicenseAcceptance | None", provider: "DataProvider") -> None:
    """Invariante 2 (bloqueante): sin un recibo de aceptación VÁLIDO
    (mismo proveedor, digest de licencia vigente) NO se emite ninguna
    petición de red. Cada `download()` de cada proveedor llama a esto
    como PRIMERA línea, antes de tocar `secure_fetch` — centralizado
    aquí para que ningún proveedor pueda "olvidarlo" ni aceptar un
    recibo ajeno o con términos caducados."""
    if acceptance is None:
        raise DataProviderError(
            f"El proveedor {provider.provider_id!r} exige un recibo de aceptación de "
            "licencia (LicenseAcceptance) antes de descargar — regístralo con "
            "LicenseAcceptanceStore.record() tras mostrar get_license_info() al usuario."
        )
    if acceptance.provider_id != provider.provider_id:
        raise DataProviderError(
            f"El recibo de aceptación es de {acceptance.provider_id!r}, no de "
            f"{provider.provider_id!r}."
        )
    current_digest = _license_digest(provider.get_license_info())
    if acceptance.license_digest != current_digest:
        raise DataProviderError(
            f"Los términos de licencia de {provider.provider_id!r} cambiaron desde que "
            "se aceptó este recibo — vuelve a aceptar antes de descargar."
        )


@runtime_checkable
class DataProvider(Protocol):
    provider_id: str

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Errores accionables (lista vacía si `config` es válida) — nunca
        toca la red."""
        ...

    def check_availability(self) -> bool:
        """¿Responde la fuente externa ahora mismo CON el CSV que
        promete, no solo con "algún" 200? `True` para un proveedor sin
        red (`synthetic_local`)."""
        ...

    def get_metadata(self) -> ProviderMetadata: ...

    def get_license_info(self) -> LicenseInfo: ...

    def estimate_download(self, config: dict[str, Any]) -> DownloadEstimate:
        """Estimación SIN descargar de verdad — puede tocar la red para
        una petición ligera (p.ej. HEAD/metadata), nunca el cuerpo
        completo."""
        ...

    def download(self, config: dict[str, Any], *, license_acceptance: "LicenseAcceptance | None") -> DownloadResult:
        """Lanza `DataProviderError` si `license_acceptance` no es un
        recibo válido para ESTE proveedor — DEBE ser la primera
        comprobación, antes de cualquier petición de red (invariante 2,
        ver `require_valid_acceptance`)."""
        ...


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
_default_acceptance_store: LicenseAcceptanceStore | None = None


def get_default_registry() -> ProviderRegistry:
    """Singleton perezoso — los proveedores concretos se importan aquí
    dentro (no arriba del módulo) para que este fichero, la interfaz, no
    dependa de ninguna implementación concreta.

    Auditoría 2026-07-17 [ALTA + recomendación]: `stooq` se retira del
    registro — verificado con `curl` real que `stooq.com`/`stooq.pl`
    gatean hoy TODA petición sin motor JS tras un reto de
    prueba-de-trabajo, así que un proveedor "registrado y disponible"
    nunca puede completar una descarga real; mantenerlo registrado sería
    ofrecer una funcionalidad rota. Sustituido por `ecb_fx` (Banco
    Central Europeo, tipos de cambio diarios — API pública real, HTTPS,
    CSV directo, sin JS, uso comercial permitido con atribución;
    verificado con `curl` real durante este mismo corte). La clase
    `StooqProvider` se conserva en el módulo (no se borra el código,
    documentado como "no registrada" en su propio docstring) por si
    Roberto decide reactivarla cuando/si Stooq cambie su política."""
    global _default_registry
    if _default_registry is None:
        from matrixai.training.provider_ecb_fx import EcbFxProvider
        from matrixai.training.provider_open_meteo import OpenMeteoProvider
        from matrixai.training.provider_synthetic_local import SyntheticLocalProvider

        registry = ProviderRegistry()
        registry.register(SyntheticLocalProvider())
        registry.register(OpenMeteoProvider())
        registry.register(EcbFxProvider())
        _default_registry = registry
    return _default_registry


def get_default_acceptance_store() -> LicenseAcceptanceStore:
    global _default_acceptance_store
    if _default_acceptance_store is None:
        _default_acceptance_store = LicenseAcceptanceStore()
    return _default_acceptance_store
