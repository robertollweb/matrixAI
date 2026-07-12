# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from pathlib import Path

_REGISTRY_JSON = "registry.json"
_ENTRIES_DIR = "entries"
_TAGS_DIR = "tags"
_LOCAL_KEY_FILE = ".registry_signing_key"

# Files present under each entries/<name>/<version>/ directory.
ENTRY_FILES: dict[str, str] = {
    "manifest":           "manifest.json",
    "model":              "model.mxai",
    "params":             "params.json",
    "weights_mxw":        "weights.mxw",
    "training_trace":     "training_trace.json",
    "evaluation_report":  "evaluation_report.json",
    "signature":          "signature.json",
}


class RegistryLayout:
    """Computes canonical filesystem paths for a registry rooted at ``root``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        return self.root / _REGISTRY_JSON

    @property
    def entries_dir(self) -> Path:
        return self.root / _ENTRIES_DIR

    @property
    def tags_dir(self) -> Path:
        return self.root / _TAGS_DIR

    @property
    def local_signing_key_path(self) -> Path:
        return self.root / _LOCAL_KEY_FILE

    def entry_dir(self, name: str, version: str) -> Path:
        return self.root / _ENTRIES_DIR / name / version

    def entry_file(self, name: str, version: str, file_key: str) -> Path:
        if file_key not in ENTRY_FILES:
            raise KeyError(f"Unknown entry file key {file_key!r}. Valid: {list(ENTRY_FILES)}")
        return self.entry_dir(name, version) / ENTRY_FILES[file_key]

    def tag_path(self, name: str, tag: str) -> Path:
        return self.root / _TAGS_DIR / name / tag

    def required_entry_paths(self, name: str, version: str) -> dict[str, Path]:
        return {k: self.entry_file(name, version, k) for k in ENTRY_FILES}
