# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — registro de proveedores +
registro auditable de aceptación de licencia (invariante 2).

Reauditoría 2026-07-17 (ronda 2) [MEDIA]: el gate original solo recibía
un booleano efímero, sin rastro de QUÉ licencia se aceptó, cuándo, ni
quién — `LicenseAcceptance`/`LicenseAcceptanceStore` sustituyen ese
booleano por un recibo persistente (v1: en memoria) verificable contra
el proveedor y la versión VIGENTE de sus términos."""
from __future__ import annotations

import pytest

from matrixai.training.data_provider import (
    DataProvider,
    DataProviderError,
    DownloadEstimate,
    LicenseAcceptance,
    LicenseAcceptanceStore,
    LicenseInfo,
    ProviderMetadata,
    ProviderRegistry,
    get_default_acceptance_store,
    get_default_registry,
    require_valid_acceptance,
)


class _StubProvider:
    provider_id = "stub"

    def validate_config(self, config):
        return []

    def check_availability(self) -> bool:
        return True

    def get_metadata(self) -> ProviderMetadata:
        return ProviderMetadata("stub", "Stub", "desc", requires_network=False)

    def get_license_info(self) -> LicenseInfo:
        return LicenseInfo("Stub", "", "", False, True)

    def estimate_download(self, config):
        return None

    def download(self, config, *, license_acceptance):
        require_valid_acceptance(license_acceptance, self)
        return None


class TestProviderRegistry:
    def test_register_and_get(self):
        registry = ProviderRegistry()
        registry.register(_StubProvider())
        assert registry.get("stub").provider_id == "stub"

    def test_unknown_provider_raises_actionable_error(self):
        registry = ProviderRegistry()
        registry.register(_StubProvider())
        with pytest.raises(DataProviderError, match="desconocido"):
            registry.get("no_existe")

    def test_list_providers_returns_metadata(self):
        registry = ProviderRegistry()
        registry.register(_StubProvider())
        metas = registry.list_providers()
        assert len(metas) == 1
        assert metas[0].provider_id == "stub"


class TestLocalizedProviderText:
    def test_license_info_localizes_without_changing_the_canonical_dict(self):
        info = LicenseInfo(
            "Licencia", "https://example.test", "Resumen español", False, True,
            summary_i18n={"en": "English summary"},
        )
        assert info.to_dict()["summary"] == "Resumen español"
        assert info.to_dict(locale="es")["summary"] == "Resumen español"
        assert info.to_dict(locale="en")["summary"] == "English summary"
        assert "summary_i18n" not in info.to_dict(locale="en")

    def test_download_estimate_localizes_dynamic_notes(self):
        estimate = DownloadEstimate(
            10, 100, "Diez días", notes_i18n={"en": "Ten days"},
        )
        assert estimate.to_dict()["notes"] == "Diez días"
        assert estimate.to_dict(locale="en")["notes"] == "Ten days"


class TestDefaultRegistry:
    def test_default_registry_has_the_three_v1_providers(self):
        registry = get_default_registry()
        ids = {m.provider_id for m in registry.list_providers()}
        assert ids == {"synthetic_local", "open_meteo", "ecb_fx"}

    def test_stooq_is_not_registered(self):
        """Auditoría 2026-07-17 (ronda 2) [ALTA]: stooq.com/stooq.pl
        gatean hoy toda petición de servidor tras un reto anti-bot JS
        (verificado con curl real) — un proveedor "registrado" que nunca
        puede completar una descarga real es peor que no ofrecerlo."""
        registry = get_default_registry()
        with pytest.raises(DataProviderError, match="desconocido"):
            registry.get("stooq")

    def test_default_registry_is_a_singleton(self):
        assert get_default_registry() is get_default_registry()

    def test_each_provider_satisfies_the_protocol_shape(self):
        for meta in get_default_registry().list_providers():
            provider = get_default_registry().get(meta.provider_id)
            assert isinstance(provider, DataProvider)
            license_info = provider.get_license_info()
            assert isinstance(license_info, LicenseInfo)
            assert license_info.name


class TestLicenseAcceptanceStore:
    def test_record_and_verify_round_trip(self):
        store = LicenseAcceptanceStore()
        provider = _StubProvider()
        acceptance = store.record(provider, actor="roberto")
        assert isinstance(acceptance, LicenseAcceptance)
        assert acceptance.provider_id == "stub"
        assert acceptance.actor == "roberto"
        verified = store.verify(acceptance.acceptance_id, provider)
        assert verified == acceptance

    def test_record_requires_a_non_empty_actor(self):
        store = LicenseAcceptanceStore()
        with pytest.raises(DataProviderError, match="actor"):
            store.record(_StubProvider(), actor="")

    def test_verify_unknown_acceptance_id_raises(self):
        store = LicenseAcceptanceStore()
        with pytest.raises(DataProviderError, match="no existe"):
            store.verify("no-existe", _StubProvider())

    def test_verify_rejects_acceptance_from_a_different_provider(self):
        store = LicenseAcceptanceStore()

        class _OtherProvider(_StubProvider):
            provider_id = "other"

        acceptance = store.record(_StubProvider(), actor="roberto")
        with pytest.raises(DataProviderError, match="no de 'other'"):
            store.verify(acceptance.acceptance_id, _OtherProvider())

    def test_verify_rejects_a_stale_license_digest(self):
        """Si el proveedor cambia sus términos DESPUÉS de que alguien
        aceptó, el recibo antiguo deja de ser válido — "la versión
        actual de la licencia", no solo "alguna vez aceptaste algo"."""
        store = LicenseAcceptanceStore()

        class _MutableLicenseProvider(_StubProvider):
            def __init__(self):
                self.summary = "v1"

            def get_license_info(self):
                return LicenseInfo("Stub", "", self.summary, False, True)

        provider = _MutableLicenseProvider()
        acceptance = store.record(provider, actor="roberto")
        provider.summary = "v2 — términos cambiados"
        with pytest.raises(DataProviderError, match="cambiaron"):
            store.verify(acceptance.acceptance_id, provider)


class TestRequireValidAcceptance:
    def test_none_acceptance_raises(self):
        with pytest.raises(DataProviderError, match="exige un recibo"):
            require_valid_acceptance(None, _StubProvider())

    def test_valid_acceptance_passes(self):
        store = LicenseAcceptanceStore()
        provider = _StubProvider()
        acceptance = store.record(provider, actor="roberto")
        require_valid_acceptance(acceptance, provider)  # no debe lanzar

    def test_stub_provider_download_rejects_without_acceptance(self):
        provider = _StubProvider()
        with pytest.raises(DataProviderError):
            provider.download({}, license_acceptance=None)


class TestDefaultAcceptanceStore:
    def test_is_a_singleton(self):
        assert get_default_acceptance_store() is get_default_acceptance_store()
