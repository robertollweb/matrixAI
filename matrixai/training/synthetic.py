# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

import random
import warnings
from typing import Any

from matrixai.ir.schema import MatrixAIProgram, VectorSpec
from matrixai.training.data import InMemoryDataAdapter
from matrixai.training.spec import TrainingSpec
from matrixai.types import TypeSpec


class SyntheticDataGenerator:
    def __init__(
        self,
        program: MatrixAIProgram,
        training: TrainingSpec,
        seed: int,
        rows: int,
        mode: str = "random",
        field_ranges: dict[str, tuple[float, float]] | None = None,
        domain_scale: bool = False,
        field_types: dict[str, str] | None = None,
        one_hot_groups: dict[str, list[str]] | None = None,
        domain_rules: Any = None,
    ):
        self.program = program
        self.training = training
        self.seed = seed
        self.rows = rows
        self.mode = mode
        self.field_ranges: dict[str, tuple[float, float]] = field_ranges or {}
        # M5: emit ranged columns in domain scale (salary 35000, age 72) instead
        # of normalized [0,1]. Internal logic (coherent runtime run, rebalance)
        # always works on normalized values; only the OUTPUT rows are converted.
        # The training boundary must normalize back with the SAME ranges.
        self.domain_scale = domain_scale
        # S2: declared type per column ("number" | "integer" | "boolean").
        # boolean quantizes the normalized sample to {0,1} BEFORE the coherent
        # runtime sees it; integer only changes the domain-scale rounding.
        self.field_types: dict[str, str] = field_types or {}
        # S2-C2: one-hot groups {original_col: [expanded member columns]}. Each
        # row gets exactly one member = 1 and the rest 0 (true one-hot, not the
        # ordinal encoding a single Scalar would give). Members live in the
        # expanded VECTOR/columns; the runtime and rebalance see them as 0/1.
        self.one_hot_groups: dict[str, list[str]] = one_hot_groups or {}
        self._one_hot_members: set[str] = {
            col for members in self.one_hot_groups.values() for col in members
        }

        # M8 v2: LLM-proposed domain rules (already normalized to [0,1]). When set,
        # multiclass labels come from the deterministic evaluator (plausible signal)
        # instead of the untrained runtime (toy). None → unchanged coherent behaviour.
        self.domain_rules = domain_rules
        self.domain_rules_used: int = 0
        # True when domain-rule labelling collapsed to a single class on the sampled
        # rows (bad thresholds / missing ranges / unhit branches) → the signal is
        # useless and the caller should fall back to coherent honestly.
        self.domain_rules_degenerate: bool = False
        # Declared classes that never appear in the generated data (≥2 still present,
        # so not degenerate). The caller warns: the model can't learn absent classes
        # and macro F1 will be low. Populated after generate() for classification.
        self.missing_labels: list[str] = []

        self.rng = random.Random(seed)
        # Rows in coherent mode that fell back to random (runtime failure or no valid label).
        # Populated after generate(); check this to verify true coherence.
        self.coherent_fallback_count: int = 0

    def generate(self) -> InMemoryDataAdapter:
        input_vector_name = self.training.dataset.input.vector
        input_columns = self.training.dataset.input.columns
        target_name = self.training.dataset.target.name

        vector_spec: VectorSpec | None = None
        for v in self.program.vectors:
            if v.name == input_vector_name:
                vector_spec = v
                break

        if not vector_spec:
            raise ValueError(f"Vector '{input_vector_name}' not found in program.")

        target_type = self.training.dataset.target.type
        is_regression = target_type.name in {"Scalar", "Integer"}
        is_probability = target_type.name.lower() == "probability"
        target_labels = target_type.parameters.get("labels") or target_type.parameters.get("args", [])
        if not is_regression and not is_probability and not target_labels:
            raise ValueError("Target must be a Label[...] type with labels defined.")

        regression_range = self._regression_range(target_type) if is_regression else (-1.0, 1.0)

        self.coherent_fallback_count = 0
        # Pre-initialize network parameters once so all coherent rows share the same
        # mapping: consistent but non-trivial feature→label correlations for NETWORK models.
        _coherent_params = None
        if self.mode == "coherent":
            try:
                from matrixai.parameters.store import build_initial_parameter_set  # noqa: PLC0415
                _coherent_params = build_initial_parameter_set(self.program).runtime_parameters()
            except Exception:  # noqa: BLE001
                _coherent_params = None

        generated_rows: list[dict[str, Any]] = []
        for _ in range(self.rows):
            row_dict: dict[str, Any] = {}
            for col in input_columns:
                col_type = vector_spec.field_types.get(col)
                if not col_type:
                    row_dict[col] = round(self.rng.uniform(0.0, 1.0), 4)
                else:
                    row_dict[col] = self._sample_type(col_type, col)
                # S2: boolean columns live in {0,1} also in normalized space,
                # so the coherent runtime and training see true flag semantics.
                if self.field_types.get(col) == "boolean":
                    try:
                        row_dict[col] = 1.0 if float(row_dict[col]) >= 0.5 else 0.0
                    except (TypeError, ValueError):
                        pass

            # S2-C2: overwrite one-hot group members with a single active value
            # (done before the coherent runtime so it sees valid one-hot input).
            for members in self.one_hot_groups.values():
                present = [m for m in members if m in row_dict]
                if not present:
                    continue
                active = self.rng.choice(present)
                for m in present:
                    row_dict[m] = 1 if m == active else 0

            if self.mode == "random":
                if is_regression:
                    lo, hi = regression_range
                    row_dict[target_name] = round(self.rng.uniform(lo, hi), 4)
                elif is_probability:
                    row_dict[target_name] = float(self.rng.randint(0, 1))
                else:
                    row_dict[target_name] = self.rng.choice(target_labels)

            elif self.mode == "coherent":
                if is_probability:
                    # NETWORK models don't produce meaningful outputs without trained params;
                    # generate balanced 0.0/1.0 labels for BCE training.
                    row_dict[target_name] = float(self.rng.randint(0, 1))

                elif is_regression:
                    fell_back = False
                    try:
                        from matrixai.runtime.runtime import MatrixAIRuntime
                        result = MatrixAIRuntime().run(self.program, {input_vector_name: row_dict})
                        target_val = result["state"].get(target_name)
                        if isinstance(target_val, (int, float)):
                            row_dict[target_name] = round(float(target_val), 6)
                        else:
                            lo, hi = regression_range
                            row_dict[target_name] = round(self.rng.uniform(lo, hi), 4)
                            fell_back = True
                    except Exception:
                        lo, hi = regression_range
                        row_dict[target_name] = round(self.rng.uniform(lo, hi), 4)
                        fell_back = True
                    if fell_back:
                        self.coherent_fallback_count += 1
                elif self.domain_rules is not None:
                    # M8 v2: deterministic domain-rule labelling (plausible signal).
                    # Rules are pre-normalized to [0,1]; row_dict is still normalized
                    # here (domain-scale conversion happens later).
                    row_dict[target_name] = self.domain_rules.label_for(row_dict)
                    self.domain_rules_used += 1
                else:
                    from matrixai.runtime.runtime import MatrixAIRuntime
                    fell_back = False
                    try:
                        runtime = MatrixAIRuntime()
                        result = runtime.run(self.program, {input_vector_name: row_dict},
                                             parameters=_coherent_params)

                        target_val = result["state"].get(target_name)
                        if isinstance(target_val, dict) and "label" in target_val:
                            label = target_val["label"]
                        elif isinstance(target_val, str):
                            label = target_val
                        elif isinstance(target_val, dict) and "probabilities" in target_val:
                            label = max(target_val["probabilities"].items(), key=lambda x: x[1])[0]
                        elif isinstance(target_val, (list, tuple)) and len(target_val) == len(target_labels):
                            # ProbabilityMap output from NETWORK models: argmax gives predicted class
                            idx = int(max(range(len(target_val)), key=lambda i: float(target_val[i])))
                            label = target_labels[idx]
                        else:
                            activated_labels = [
                                act["call"] for act in result["actions"]
                                if act.get("activated") and act.get("call") in target_labels
                            ]
                            if activated_labels:
                                label = activated_labels[0]
                            else:
                                label = self.rng.choice(target_labels)
                                fell_back = True

                        if label not in target_labels:
                            label = self.rng.choice(target_labels)
                            fell_back = True

                        row_dict[target_name] = label
                    except Exception:
                        row_dict[target_name] = self.rng.choice(target_labels)
                        fell_back = True

                    if fell_back:
                        self.coherent_fallback_count += 1
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            generated_rows.append(row_dict)

        if self.mode == "coherent" and self.coherent_fallback_count > 0:
            fallback_label = "values" if is_regression else "labels"
            warnings.warn(
                f"SyntheticDataGenerator: {self.coherent_fallback_count}/{self.rows} rows fell back to "
                f"random {fallback_label} in coherent mode (runtime did not produce a valid output). "
                f"The dataset is marked as mode='coherent' but those rows are effectively random.",
                UserWarning,
                stacklevel=2,
            )

        # Rebalance classification labels when the distribution is too skewed.
        # Threshold: any class with < (rows / n_classes) * 0.4 rows triggers rebalance.
        # Rebalancing sorts rows by mean feature score and assigns labels in equal chunks —
        # this creates a genuine monotonic correlation without running the runtime again.
        # M8 v2: never rebalance domain-rule labels — the class skew (e.g. few
        # CRÍTICO) is the intended domain distribution; rebalancing would destroy it.
        if (not is_regression and not is_probability and target_labels
                and len(target_labels) >= 2 and not self.domain_rules_used):
            from collections import Counter  # noqa: PLC0415
            counts = Counter(r[target_name] for r in generated_rows)
            min_expected = max(1, self.rows // len(target_labels))
            if any(counts.get(lbl, 0) < max(1, int(min_expected * 0.4)) for lbl in target_labels):
                generated_rows = self._rebalance_labels(
                    generated_rows, target_name, target_labels, input_columns,
                )

        # M8 v2: flag degenerate domain-rule labelling (≈1 class) so the caller can
        # fall back to coherent instead of shipping a single-class (useless) dataset.
        if self.domain_rules_used:
            distinct = {r[target_name] for r in generated_rows}
            self.domain_rules_degenerate = len(distinct) < 2

        # M8 v2: declared classes absent from the data (≥2 present → not degenerate).
        # The model can't learn classes it never sees → low macro F1; the caller warns.
        if not is_regression and not is_probability and target_labels:
            present = {r[target_name] for r in generated_rows}
            self.missing_labels = [lbl for lbl in target_labels if lbl not in present]

        # M5: convert ranged input columns to domain scale for human-readable
        # output. Done AFTER labels and rebalancing so all internal logic ran
        # on normalized values. Targets are never converted.
        if self.domain_scale and (self.field_ranges or self.field_types):
            generated_rows = [
                {
                    col: (self._to_domain(val, col) if col != target_name else val)
                    for col, val in row.items()
                }
                for row in generated_rows
            ]

        adapter = InMemoryDataAdapter(
            rows=generated_rows,
            input_vector=input_vector_name,
            input_columns=input_columns,
            target=target_name,
            labels=[] if (is_regression or is_probability) else target_labels,
            source=f"synthetic-{self.seed}-{self.mode}",
        )
        return adapter

    def _to_domain(self, value: Any, col: str) -> float | int:
        """Map a normalized [0,1] sample back to its domain range, with
        human-friendly rounding (salary 35000, not 35000.1234).

        S2: the declared type overrides the span heuristic — boolean emits
        0/1 ints regardless of range, integer always rounds to int."""
        col_type = self.field_types.get(col)
        if col_type == "boolean":
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return value
        if col not in self.field_ranges:
            return value
        lo, hi = self.field_ranges[col]
        try:
            raw = lo + float(value) * (hi - lo)
        except (TypeError, ValueError):
            return value
        if col_type == "integer":
            return int(round(raw))
        span = hi - lo
        if span >= 50 and float(lo).is_integer() and float(hi).is_integer():
            return int(round(raw))
        if span > 1:
            return round(raw, 2)
        return round(raw, 4)

    def _rebalance_labels(
        self,
        rows: list[dict[str, Any]],
        target_name: str,
        target_labels: list[str],
        input_columns: list[str],
    ) -> list[dict[str, Any]]:
        """Sort rows by mean feature score and assign labels in equal-sized chunks.

        This guarantees balanced classes and a genuine monotonic feature→label correlation:
        rows with the lowest mean feature values get label[0], rows with the highest get label[-1].
        Works without running the runtime again — no fallback risk.
        """
        n = len(rows)
        n_classes = len(target_labels)

        def _score(row: dict[str, Any]) -> float:
            vals = [row.get(c, 0.0) for c in input_columns if isinstance(row.get(c), (int, float))]
            return sum(vals) / len(vals) if vals else 0.0

        sorted_rows = sorted(rows, key=_score)
        chunk = n / n_classes
        rebalanced = []
        for i, row in enumerate(sorted_rows):
            label_idx = min(int(i / chunk), n_classes - 1)
            rebalanced.append({**row, target_name: target_labels[label_idx]})
        return rebalanced

    def _regression_range(self, type_spec: TypeSpec) -> tuple[float, float]:
        if type_spec.range is not None:
            lo = type_spec.range.minimum if type_spec.range.minimum is not None else -1.0
            hi = type_spec.range.maximum if type_spec.range.maximum is not None else 1.0
            return float(lo), float(hi)
        return -1.0, 1.0

    def _sample_type(self, type_spec: TypeSpec, field_name: str = "") -> float:
        # LLM-provided field_ranges: normalise to [0, 1] so values align with the
        # inference slider range (sliders always send 0–1 regardless of domain scale).
        if field_name and field_name in self.field_ranges:
            lo, hi = self.field_ranges[field_name]
            raw = self.rng.uniform(lo, hi)
            return round((raw - lo) / (hi - lo) if hi != lo else 0.5, 4)

        # Integer fields (e.g. an EMBEDDING source `cat: Integer[0, vocab-1]`) must
        # sample whole indices, not floats — the embedding lookup uses the raw index
        # and the value is NOT normalized. randint is inclusive on both ends.
        if type_spec.name == "Integer" and type_spec.range is not None:
            lo = int(type_spec.range.minimum) if type_spec.range.minimum is not None else 0
            hi = int(type_spec.range.maximum) if type_spec.range.maximum is not None else 1
            if hi < lo:
                lo, hi = hi, lo
            return float(self.rng.randint(lo, hi))

        # Explicit range in the DSL (e.g. FLOAT [10, 20]): honour it as-is.
        # The user or the type system defined this range intentionally.
        if type_spec.range is not None:
            min_val = type_spec.range.minimum if type_spec.range.minimum is not None else 0.0
            max_val = type_spec.range.maximum if type_spec.range.maximum is not None else 1.0
            return round(self.rng.uniform(float(min_val), float(max_val)), 4)

        # Default: [0, 1] matches the inference slider range for untyped Scalar fields
        return round(self.rng.uniform(0.0, 1.0), 4)
