# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""P22 C7 — ContinualVersioner: promote an approved candidate to the P21 registry."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from matrixai import __version__ as _MATRIXAI_VERSION
from matrixai.ir.continual import ContinualPolicySpec
from matrixai.parameters.store import ParameterSet
from matrixai.registry.entry_hash import compute_entry_hash, sha256_str
from matrixai.registry.model_registry import ModelRegistry
from matrixai.registry.schema import RegistryEntry


# ── error ──────────────────────────────────────────────────────────────────────

class ContinualVersioningError(Exception):
    pass


def _validate_pending_approval_tokens(
    pending: Any,
    signing_key: str | None,
    *,
    signature_required: bool = False,
) -> None:
    from matrixai.continual.approval import _make_approval_token, _make_decision_token

    def _key_for(token: str, kind: str) -> str | None:
        if token.startswith("hmac-sha256:"):
            if not signing_key:
                raise ContinualVersioningError(
                    f"Cannot promote: {kind} is HMAC-signed but no signing key was provided."
                )
            return signing_key
        if token.startswith("sha256:"):
            if signature_required:
                raise ContinualVersioningError(
                    f"Cannot promote: {kind} is unsigned (sha256:) but the policy "
                    "declares SIGNATURE_REQUIRED true. "
                    "Provide --signing-key to generate HMAC-signed tokens."
                )
            return None
        raise ContinualVersioningError(f"Cannot promote: unsupported {kind} prefix.")

    expected_approval = _make_approval_token(
        pending.policy_hash,
        pending.candidate_parameter_set_id,
        pending.created_at,
        _key_for(pending.approval_token, "approval_token"),
        expires_at=pending.expires_at,
    )
    if expected_approval != pending.approval_token:
        raise ContinualVersioningError("Cannot promote: PendingApproval approval_token mismatch.")

    expected_decision = _make_decision_token(
        pending.approval_token,
        pending.status,
        pending.decided_by,
        pending.decided_at,
        _key_for(pending.decision_token, "decision_token"),
    )
    if expected_decision != pending.decision_token:
        raise ContinualVersioningError("Cannot promote: PendingApproval decision_token mismatch.")


# ── result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContinualVersioningResult:
    registry_name: str
    previous_version: str
    new_version: str
    entry_hash: str
    pushed_at: str
    continual_update_id: str
    parent_parameter_set_id: str
    candidate_parameter_set_id: str


# ── versioner ─────────────────────────────────────────────────────────────────

class ContinualVersioner:
    """Promotes an approved candidate ParameterSet into the P21 ModelRegistry.

    When ``promote()`` is called:

    1. Validates that ``approval_report.passed`` is ``True``.
    2. Determines the next minor subversion (e.g. ``v1.0`` → ``v1.1``).
    3. Builds a :class:`~matrixai.registry.schema.RegistryEntry` whose
       ``model_hash`` and ``parameter_schema_hash`` are carried unchanged from
       the candidate (same architecture, new weights).
    4. Writes the entry + the serialised ``ApprovalGateReport`` into the
       registry entry directory.
    5. Stores the candidate :class:`~matrixai.parameters.store.ParameterSet`
       as ``params.json`` in the entry directory.
    6. Updates the ``"current"`` tag to point to the new version.
    """

    def __init__(
        self,
        policy: ContinualPolicySpec,
        registry: ModelRegistry,
        approval_report: Any,          # ApprovalGateReport — avoid circular import
        candidate_parameter_set: ParameterSet,
        *,
        continual_update_id: str | None = None,
        now: datetime | None = None,
        approval_signing_key: str | None = None,
    ) -> None:
        self._policy = policy
        self._registry = registry
        self._report = approval_report
        self._candidate = candidate_parameter_set
        self._update_id = continual_update_id or _generate_update_id(policy, now)
        self._now = now or datetime.now(tz=timezone.utc)
        self._approval_signing_key = approval_signing_key

    # ── public API ─────────────────────────────────────────────────────────────

    def promote(self, *, human_approved: bool = False) -> ContinualVersioningResult:
        """Execute the promotion and return the versioning result.

        When the approval report has status ``"pending_human"`` (i.e. HUMAN_APPROVAL
        is ``true`` in the policy), the caller must explicitly pass
        ``human_approved=True`` to confirm that the human approved the pending
        token.  This prevents automatic promotion when a human review is required.
        """
        if not self._report.passed:
            raise ContinualVersioningError(
                f"Cannot promote: ApprovalGateReport status is {self._report.status!r}; "
                "approval gate must pass before promoting."
            )

        if self._report.status == "pending_human" and not human_approved:
            raise ContinualVersioningError(
                "Cannot promote: HUMAN_APPROVAL is required but has not been confirmed. "
                "Pass human_approved=True once the pending approval token has been approved."
            )

        if self._report.status == "pending_human" and human_approved:
            pending = self._report.pending_approval
            if pending is None:
                raise ContinualVersioningError(
                    "Cannot promote: ApprovalGateReport is pending_human but has no PendingApproval."
                )

            if pending.policy_hash != self._report.policy_hash:
                raise ContinualVersioningError("Cannot promote: PendingApproval policy_hash mismatch.")
            if pending.candidate_parameter_set_id != self._report.candidate_parameter_set_id:
                raise ContinualVersioningError("Cannot promote: PendingApproval candidate_parameter_set_id mismatch.")

            if pending.expires_at is not None:
                from datetime import datetime, timezone
                try:
                    expiry = datetime.fromisoformat(pending.expires_at)
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=timezone.utc)
                    if self._now >= expiry:
                        raise ContinualVersioningError(
                            f"Cannot promote: PendingApproval token has expired at "
                            f"{pending.expires_at!r}. A new approval must be requested."
                        )
                except ValueError:
                    raise ContinualVersioningError(
                        f"Cannot promote: PendingApproval expires_at {pending.expires_at!r} "
                        "is not a valid ISO-8601 datetime. Reject to fail closed."
                    ) from None

            if pending.status != "approved":
                raise ContinualVersioningError(
                    f"Cannot promote: PendingApproval status is {pending.status!r}; "
                    "an approved human decision is required."
                )
            if not pending.decided_at or not pending.decided_by or not pending.decision_token:
                raise ContinualVersioningError(
                    "Cannot promote: approved PendingApproval must include decided_at, "
                    "decided_by and decision_token."
                )
            _validate_pending_approval_tokens(
                pending,
                self._approval_signing_key,
                signature_required=self._policy.audit.signature_required,
            )

        # Cryptographic binding: ensure the report belongs to this candidate and policy
        if self._report.candidate_parameter_set_id != self._candidate.parameter_set_id:
            raise ContinualVersioningError(
                f"Approval report candidate_parameter_set_id "
                f"({self._report.candidate_parameter_set_id!r}) does not match "
                f"the candidate ParameterSet ({self._candidate.parameter_set_id!r}). "
                "The report must have been generated for this exact candidate."
            )
        if self._report.policy_hash != self._policy.policy_hash:
            raise ContinualVersioningError(
                f"Approval report policy_hash ({self._report.policy_hash!r}) does not match "
                f"the current policy ({self._policy.policy_hash!r}). "
                "The report must have been generated under the current policy."
            )

        registry_name = self._policy.registry_name
        if not registry_name:
            raise ContinualVersioningError(
                "Cannot promote: REGISTRY_NAME is not declared in the policy. "
                "Add 'REGISTRY_NAME <name>' to the CONTINUAL_POLICY block."
            )

        base_version = self._policy.base_version or "v1.0"
        new_version = self._next_version(registry_name, base_version)

        entry = self._build_entry(registry_name, new_version)
        self._registry.push(entry)
        self._store_artifacts(registry_name, new_version, entry)
        self._registry.tag(registry_name, new_version, "current")

        parent_ps_id = self._candidate.metrics.get(
            "parent_parameter_set_id",
            self._report.baseline_parameter_set_id,
        )
        return ContinualVersioningResult(
            registry_name=registry_name,
            previous_version=base_version,
            new_version=new_version,
            entry_hash=entry.entry_hash,
            pushed_at=self._now.isoformat(),
            continual_update_id=self._update_id,
            parent_parameter_set_id=parent_ps_id,
            candidate_parameter_set_id=self._candidate.parameter_set_id,
        )

    # ── version bumping ────────────────────────────────────────────────────────

    def _next_version(self, registry_name: str, base_version: str) -> str:
        major = _major(base_version)
        max_minor = _minor(base_version)   # start from base minor, not 0

        try:
            existing = self._registry.list({"name": registry_name})
        except Exception:  # noqa: BLE001
            existing = []

        for entry in existing:
            if _major(entry.version) == major:
                max_minor = max(max_minor, _minor(entry.version))

        return f"v{major}.{max_minor + 1}"

    # ── entry construction ─────────────────────────────────────────────────────

    def _build_entry(self, name: str, version: str) -> RegistryEntry:
        report_json = json.dumps(_report_to_dict(self._report), sort_keys=True, separators=(",", ":"))
        eval_report_hash = sha256_str(report_json)

        dataset_fp = self._candidate.metrics.get("dataset_fingerprint", "")
        parent_ps_id = self._candidate.metrics.get(
            "parent_parameter_set_id",
            self._report.baseline_parameter_set_id,
        )

        metrics = {
            "parent_parameter_set_id": parent_ps_id,
            "continual_update_id": self._update_id,
            "validation_loss": self._report.candidate_metrics.loss,
            "accuracy": self._report.candidate_metrics.accuracy,
            "macro_f1": self._report.candidate_metrics.macro_f1,
            "holdout_samples": self._report.holdout_samples,
            "promoted_at": self._now.isoformat(),
            "policy_hash": self._policy.policy_hash,
        }

        entry_hash = compute_entry_hash(
            name=name,
            version=version,
            model_hash=self._candidate.model_hash,
            parameter_schema_hash=self._candidate.parameter_schema_hash,
            parameter_set_id=self._candidate.parameter_set_id,
            training_trace_hash="",
            evaluation_report_hash=eval_report_hash,
            matrixai_version=_MATRIXAI_VERSION,
        )
        return RegistryEntry(
            name=name,
            version=version,
            entry_hash=entry_hash,
            model_hash=self._candidate.model_hash,
            parameter_schema_hash=self._candidate.parameter_schema_hash,
            parameter_set_id=self._candidate.parameter_set_id,
            input_type={},
            output_type={},
            metrics=metrics,
            matrixai_version=_MATRIXAI_VERSION,
            created_at=self._now.isoformat(),
            training_dataset_fingerprint=dataset_fp,
            interpretability_level="full",
            training_trace_hash="",
            evaluation_report_hash=eval_report_hash,
        )

    def _store_artifacts(self, name: str, version: str, entry: RegistryEntry) -> None:
        from matrixai.parameters.store import write_parameter_set

        entry_dir = self._registry.layout.entry_dir(name, version)

        # Parameter values
        write_parameter_set(entry_dir / "params.json", self._candidate)

        # Approval gate report (the "evaluation report" for this entry)
        report_dict = _report_to_dict(self._report)
        (entry_dir / "approval_gate_report.json").write_text(
            json.dumps(report_dict, sort_keys=True, indent=2),
            encoding="utf-8",
        )


# ── helpers ────────────────────────────────────────────────────────────────────

def _major(version: str) -> int:
    """Return major component of a 'vMAJOR.MINOR' string."""
    parts = version.lstrip("v").split(".")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return 1


def _minor(version: str) -> int:
    """Return minor component of a 'vMAJOR.MINOR' string."""
    parts = version.lstrip("v").split(".")
    try:
        return int(parts[1])
    except (ValueError, IndexError):
        return 0


def _generate_update_id(policy: ContinualPolicySpec, now: datetime | None = None) -> str:
    ts = (now or datetime.now(tz=timezone.utc)).strftime("%Y%m%d%H%M%S")
    phash = policy.policy_hash[7:15] if policy.policy_hash.startswith("sha256:") else policy.policy_hash[:8]
    return f"cu-{ts}-{phash}"


def _report_to_dict(report: Any) -> dict[str, Any]:
    """Serialize an ApprovalGateReport (or any dataclass) to a JSON-safe dict."""
    try:
        return dataclasses.asdict(report)
    except TypeError:
        return {"raw": str(report)}
