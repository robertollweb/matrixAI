# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""BIBLIOTECA_PROYECTOS_INTELIGENTES C5 — registro de proveedores +
gate de licencia (invariante 2)."""
from __future__ import annotations

import pytest

from matrixai.training.data_provider import (
    DataProvider,
    DataProviderError,
    LicenseInfo,
    ProviderMetadata,
    ProviderRegistry,
    get_default_registry,
    require_license_accepted,
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

    def download(self, config, *, license_accepted: bool):
        require_license_accepted(license_accepted, self.provider_id)
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


class TestDefaultRegistry:
    def test_default_registry_has_the_three_v1_providers(self):
        registry = get_default_registry()
        ids = {m.provider_id for m in registry.list_providers()}
        assert ids == {"synthetic_local", "open_meteo", "stooq"}

    def test_default_registry_is_a_singleton(self):
        assert get_default_registry() is get_default_registry()

    def test_each_provider_satisfies_the_protocol_shape(self):
        for meta in get_default_registry().list_providers():
            provider = get_default_registry().get(meta.provider_id)
            assert isinstance(provider, DataProvider)
            license_info = provider.get_license_info()
            assert isinstance(license_info, LicenseInfo)
            assert license_info.name


class TestLicenseGate:
    def test_require_license_accepted_raises_when_false(self):
        with pytest.raises(DataProviderError, match="exige aceptar su licencia"):
            require_license_accepted(False, "stub")

    def test_require_license_accepted_passes_when_true(self):
        require_license_accepted(True, "stub")  # no debe lanzar

    def test_stub_provider_download_rejects_without_acceptance(self):
        provider = _StubProvider()
        with pytest.raises(DataProviderError):
            provider.download({}, license_accepted=False)
