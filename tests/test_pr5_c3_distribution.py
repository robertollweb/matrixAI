# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from matrixai.scaffolding import scaffold_project


class TestPR5C3DistributionScaffolding(unittest.TestCase):
    def test_scaffold_accepts_absolute_project_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "absolute-model"

            created_path = scaffold_project(str(project_path), "classification")

            self.assertEqual(created_path, project_path)
            self.assertTrue((project_path / "absolute-model.mxai").exists())
            self.assertTrue((project_path / "absolute-model.mxtrain").exists())
            self.assertIn(
                "# absolute-model",
                (project_path / "README.md").read_text(encoding="utf-8"),
            )

    def test_scaffold_accepts_nested_relative_project_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            created_path = scaffold_project(
                "nested/relative-model",
                "classification",
                output_dir=Path(temp_dir),
            )

            self.assertEqual(created_path, Path(temp_dir) / "nested" / "relative-model")
            self.assertTrue((created_path / "relative-model.mxai").exists())
            self.assertTrue((created_path / "relative-model.mxtrain").exists())


if __name__ == "__main__":
    unittest.main()