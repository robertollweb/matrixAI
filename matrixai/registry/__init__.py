# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from matrixai.registry.entry_hash import compute_entry_hash, sha256_bytes, sha256_str
from matrixai.registry.layout import ENTRY_FILES, RegistryLayout
from matrixai.registry.model_registry import (
    DuplicateEntryError,
    EntryNotFoundError,
    ModelRegistry,
    ModelRegistryError,
    VerificationError,
)
from matrixai.registry.composite_hash import compute_composite_model_hash, verify_composite_model
from matrixai.registry.resolver import CompositeModelResolver, ImportResolutionError
from matrixai.registry.schema import (
    MATRIXAI_REGISTRY_SCHEMA_VERSION,
    MATRIXAI_REGISTRY_VERSION,
    RegistryEntry,
    RegistryEntryError,
)
from matrixai.registry.signing import (
    build_signature_record,
    get_signing_key,
    sign_entry_hash,
    signing_key_fingerprint,
    verify_entry_signature,
)

__all__ = [
    "CompositeModelResolver",
    "DuplicateEntryError",
    "compute_composite_model_hash",
    "verify_composite_model",
    "ENTRY_FILES",
    "ImportResolutionError",
    "EntryNotFoundError",
    "MATRIXAI_REGISTRY_VERSION",
    "MATRIXAI_REGISTRY_SCHEMA_VERSION",
    "ModelRegistry",
    "ModelRegistryError",
    "RegistryEntry",
    "RegistryEntryError",
    "RegistryLayout",
    "VerificationError",
    "build_signature_record",
    "compute_entry_hash",
    "get_signing_key",
    "sha256_bytes",
    "sha256_str",
    "sign_entry_hash",
    "signing_key_fingerprint",
    "verify_entry_signature",
]
