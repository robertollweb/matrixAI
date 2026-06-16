"""P20 C3 — Capability registry y resolución de scopes por capacidad."""
import pytest

from matrixai.actions import (
    CAPABILITIES,
    CapabilityRegistry,
    HIGH_RISK_CAPABILITIES,
    MUTATING_CAPABILITIES,
    REQUIRED_SCOPE_FIELDS,
    ScopeValidationResult,
    registry,
)


class TestCapabilityRegistryListing:
    def test_capability_registry_lists_all_supported_capabilities(self):
        caps = registry.list_capabilities()
        expected = {
            "http_get", "http_post", "notification", "email_send",
            "database_read", "database_write", "filesystem_read",
            "filesystem_write", "subprocess_spawn",
        }
        assert set(caps) == expected

    def test_capability_registry_returns_sorted_list(self):
        caps = registry.list_capabilities()
        assert caps == sorted(caps)

    def test_capability_registry_rejects_unknown_capability(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            registry.risk_level("fly_to_moon")

    def test_registry_rejects_unknown_in_required_scope_fields(self):
        with pytest.raises(ValueError, match="Unknown capability"):
            registry.required_scope_fields("telepathy")


class TestCapabilityRegistryRiskLevels:
    def test_capability_registry_classifies_risk_levels(self):
        assert registry.risk_level("http_get") == "low"
        assert registry.risk_level("email_send") == "medium"
        assert registry.risk_level("database_write") == "high"

    def test_registry_risk_level_low_for_http_get(self):
        assert registry.risk_level("http_get") == "low"

    def test_registry_risk_level_low_for_notification(self):
        assert registry.risk_level("notification") == "low"

    def test_registry_risk_level_medium_for_email_send(self):
        assert registry.risk_level("email_send") == "medium"

    def test_registry_risk_level_medium_for_http_post(self):
        assert registry.risk_level("http_post") == "medium"

    def test_registry_risk_level_high_for_database_write(self):
        assert registry.risk_level("database_write") == "high"

    def test_registry_risk_level_high_for_filesystem_write(self):
        assert registry.risk_level("filesystem_write") == "high"

    def test_registry_risk_level_high_for_subprocess_spawn(self):
        assert registry.risk_level("subprocess_spawn") == "high"

    def test_capability_high_risk_forces_sandbox_required(self):
        for cap in HIGH_RISK_CAPABILITIES:
            assert registry.is_high_risk(cap), f"{cap} should be high-risk"
        assert not registry.is_high_risk("http_get")
        assert not registry.is_high_risk("email_send")


class TestRequiredScopeFields:
    def test_capability_registry_validates_required_scope_fields(self):
        # email_send requires both fields
        fields = registry.required_scope_fields("email_send")
        assert "allowed_recipients" in fields
        assert "allowed_domains" in fields

    def test_registry_required_scope_email_send_includes_recipients_and_domains(self):
        fields = registry.required_scope_fields("email_send")
        assert set(fields) >= {"allowed_recipients", "allowed_domains"}

    def test_registry_required_scope_http_get_includes_allowed_urls(self):
        fields = registry.required_scope_fields("http_get")
        assert "allowed_urls" in fields

    def test_registry_required_scope_filesystem_write_includes_allowed_paths(self):
        fields = registry.required_scope_fields("filesystem_write")
        assert "allowed_paths" in fields

    def test_registry_required_scope_database_write_includes_tables_and_operations(self):
        fields = registry.required_scope_fields("database_write")
        assert "allowed_tables" in fields
        assert "allowed_operations" in fields


class TestScopeValidation:
    def test_registry_validate_scope_accepts_complete_scope(self):
        scope = {"allowed_recipients": ["ops@example.com"], "allowed_domains": ["example.com"]}
        result = registry.validate_scope("email_send", scope)
        assert isinstance(result, ScopeValidationResult)
        assert result.ok is True
        assert result.errors == []

    def test_registry_validate_scope_rejects_missing_required_field(self):
        scope = {"allowed_recipients": ["ops@example.com"]}  # missing allowed_domains
        result = registry.validate_scope("email_send", scope)
        assert result.ok is False
        assert any("allowed_domains" in e for e in result.errors)

    def test_registry_validate_scope_rejects_unknown_capability(self):
        result = registry.validate_scope("quantum_tunnel", {"target": "mars"})
        assert result.ok is False
        assert any("Unknown" in e for e in result.errors)

    def test_registry_validate_scope_effective_scope_populated_on_success(self):
        scope = {"allowed_urls": ["https://api.example.com/data"], "timeout_seconds": 5}
        result = registry.validate_scope("http_get", scope)
        assert result.ok
        assert result.effective_scope == scope

    def test_registry_resolve_scope_returns_effective_scope(self):
        scope = {"allowed_paths": ["/var/log/matrixai/"], "max_file_size_bytes": 1024}
        effective = registry.resolve_scope("filesystem_read", scope)
        assert effective == scope

    def test_registry_resolve_scope_raises_on_invalid_scope(self):
        with pytest.raises(ValueError, match="Invalid scope"):
            registry.resolve_scope("email_send", {"only_one_field": "x"})


class TestRollbackRequirement:
    def test_registry_requires_rollback_for_email_send(self):
        assert registry.requires_rollback("email_send") is True

    def test_registry_requires_rollback_for_http_post(self):
        assert registry.requires_rollback("http_post") is True

    def test_registry_requires_rollback_for_database_write(self):
        assert registry.requires_rollback("database_write") is True

    def test_registry_does_not_require_rollback_for_http_get(self):
        assert registry.requires_rollback("http_get") is False

    def test_registry_does_not_require_rollback_for_database_read(self):
        assert registry.requires_rollback("database_read") is False
