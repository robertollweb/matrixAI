"""C1 — Stable public API of the playground engine.

Verifies that `matrixai.playground_api` exposes the documented surface, that every
symbol is callable, and that each public alias is the *same object* as the private
function it re-exports (so the re-export adds no behavioural drift) — the vast
majority live in `matrixai.playground`, but BIBLIOTECA_PROYECTOS_INTELIGENTES C1/C2
added a few from their own modules (`dataset_analysis.py`/`dataset_project.py`):
cleaner factoring for genuinely new capabilities, not everything needs to live in
the (already large) `playground.py`.
"""

from __future__ import annotations

import unittest

import matrixai.playground as pg
import matrixai.training.dataset_analysis as _dataset_analysis
import matrixai.training.dataset_project as _dataset_project
import matrixai.training.data_provider as _data_provider
from matrixai import playground_api as api


# public alias -> private name in playground.py (None = same name, already public)
_ALIAS_MAP = {
    "analyze_playground_request": "analyze_playground_request",
    "serve": "serve",
    "handler_class": "_handler_class",
    "visual_model": "_visual_model",
    "PROJECT_ROOT": "PROJECT_ROOT",
    "training_jobs": "_training_jobs",
    "generate_training_from_mxai": "_generate_training_from_mxai",
    "generate_synthetic_dataset": "_generate_synthetic_dataset",
    "suggest_field_ranges": "_suggest_field_ranges",
    "validate_training_csv": "_validate_training_csv",
    "csv_template": "_csv_template",
    "run_playground_training": "_run_playground_training",
    "submit_training_job": "_submit_training_job",
    "get_job_status": "_get_job_status",
    "cancel_job": "_cancel_job",
    "playground_run_with_params": "_playground_run_with_params",
    "refine_prompt": "_refine_prompt",
    "coerce_field_ranges": "_coerce_field_ranges",
    "coerce_field_types": "_coerce_field_types",
    "coerce_field_categories": "_coerce_field_categories",
    "coerce_field_identifiers": "_coerce_field_identifiers",
    "normalize_csv_with_ranges": "_normalize_csv_with_ranges",
    "build_pipeline_stages": "_build_pipeline_stages",
    "build_artifacts": "_build_artifacts",
    "dense_llm_schema": "_dense_llm_schema",
    "llm_field_ranges": "_llm_field_ranges",
    "resolve_llm_config_path": "_resolve_llm_config_path",
    "detect_llm_mode": "_detect_llm_mode",
    "safe_float": "_safe_float",
}

# BIBLIOTECA C1/C2: public alias -> (module, name) for re-exports that come from
# OUTSIDE playground.py. Kept separate from _ALIAS_MAP (which assumes the source
# is always `pg`) rather than overloading that dict's value type.
_EXTERNAL_ALIAS_MAP = {
    "analyze_dataset_csv": (_dataset_analysis, "analyze_dataset_csv"),
    "DatasetAnalysisError": (_dataset_analysis, "DatasetAnalysisError"),
    "generate_project_from_dataset": (_dataset_project, "generate_project_from_dataset"),
    "generate_temporal_project_from_dataset": (_dataset_project, "generate_temporal_project_from_dataset"),
    "DatasetProjectError": (_dataset_project, "DatasetProjectError"),
    "get_default_registry": (_data_provider, "get_default_registry"),
    "DataProviderError": (_data_provider, "DataProviderError"),
}


class TestPlaygroundPublicAPI(unittest.TestCase):
    def test_all_lists_every_alias(self) -> None:
        self.assertEqual(set(api.__all__), set(_ALIAS_MAP) | set(_EXTERNAL_ALIAS_MAP))

    # Shared state/constants that are intentionally not callable.
    _NON_CALLABLE = {"PROJECT_ROOT", "training_jobs"}

    def test_every_symbol_present_and_callable(self) -> None:
        for name in api.__all__:
            obj = getattr(api, name, None)
            self.assertIsNotNone(obj, f"{name} missing from playground_api")
            if name not in self._NON_CALLABLE:
                self.assertTrue(callable(obj), f"{name} is not callable")

    def test_aliases_are_the_same_object_as_core(self) -> None:
        for public_name, private_name in _ALIAS_MAP.items():
            self.assertIs(
                getattr(api, public_name),
                getattr(pg, private_name),
                f"{public_name} must be the same object as playground.{private_name}",
            )
        for public_name, (module, private_name) in _EXTERNAL_ALIAS_MAP.items():
            self.assertIs(
                getattr(api, public_name),
                getattr(module, private_name),
                f"{public_name} must be the same object as {module.__name__}.{private_name}",
            )

    def test_safe_float_behaviour_through_public_api(self) -> None:
        # Sanity: a re-exported helper actually works when called via the public API.
        self.assertEqual(api.safe_float("1.5"), 1.5)
        self.assertEqual(api.safe_float("nope", default=0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
