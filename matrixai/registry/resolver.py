# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""CompositeModelResolver — resolves ImportSpec against the local registry."""
from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from matrixai.registry.model_registry import EntryNotFoundError, ModelRegistry
from matrixai.registry.schema import RegistryEntry

if TYPE_CHECKING:
    from matrixai.ir.schema import ImportSpec, MatrixAIProgram

_PINNED_RE = re.compile(r"^v\d+(\.\d+)*$")


class ImportResolutionError(Exception):
    pass


class CompositeModelResolver:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._cache: dict[tuple[str, str], RegistryEntry] = {}

    def resolve(
        self,
        program: "MatrixAIProgram",
        *,
        allow_mutable_tags: bool = False,
    ) -> "list[ImportSpec]":
        from matrixai.ir.schema import ImportSpec

        hydrated: list[ImportSpec] = []
        for imp in program.imports:
            entry = self._fetch(imp, allow_mutable_tags=allow_mutable_tags)
            hydrated.append(
                ImportSpec(
                    alias=imp.alias,
                    registry_name=imp.registry_name,
                    version=imp.version,
                    mode=imp.mode,
                    resolved_entry_hash=entry.entry_hash,
                    resolved_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return hydrated

    def _fetch(
        self,
        imp: "ImportSpec",
        *,
        allow_mutable_tags: bool,
    ) -> RegistryEntry:
        cache_key = (imp.registry_name, imp.version)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not _PINNED_RE.match(imp.version):
            if not allow_mutable_tags:
                raise ImportResolutionError(
                    f"Import {imp.alias!r} uses mutable tag {imp.version!r}. "
                    f"Pass allow_mutable_tags=True to allow mutable tag resolution."
                )
            warnings.warn(
                f"Import {imp.alias!r} uses mutable tag {imp.version!r}; "
                f"resolved entry may change if the tag is re-pointed.",
                UserWarning,
                stacklevel=4,
            )

        try:
            entry = self.registry.get(imp.registry_name, imp.version)
        except EntryNotFoundError as exc:
            raise ImportResolutionError(
                f"Cannot resolve import {imp.alias!r}: "
                f"{imp.registry_name}@{imp.version} not found in registry"
            ) from exc

        self._cache[cache_key] = entry
        return entry
