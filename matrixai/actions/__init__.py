# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from matrixai.actions.schema import (
    ActionContractSpec,
    RollbackSpec,
    RateLimitSpec,
    SandboxLimitsSpec,
    CAPABILITIES,
    HIGH_RISK_CAPABILITIES,
    MUTATING_CAPABILITIES,
)
from matrixai.actions.parser import MxactParseError, parse_mxact
from matrixai.actions.registry import (
    CapabilityRegistry,
    REQUIRED_SCOPE_FIELDS,
    ScopeValidationResult,
    registry,
)
from matrixai.actions.dryrun import (
    DryRunReport,
    DryRunSimulator,
    RateTracker,
)
from matrixai.actions.executor import (
    ActionExecutor,
    ActionExecutorError,
    ActionResult,
    ExecutionContext,
)
from matrixai.actions.sandbox import (
    SandboxedActionExecutor,
    SandboxedExecutorError,
    SandboxParams,
    SandboxResult,
)
from matrixai.actions.trace import (
    ActionTrace,
    build_action_trace,
    sign_action_trace,
    verify_action_trace,
)
from matrixai.actions.rollback import (
    RollbackManager,
    RollbackResult,
    RollbackError,
)
from matrixai.actions.approval import (
    ApprovalStore,
    ApprovalError,
    HumanApprovalGate,
    PendingExecution,
)
from matrixai.actions.contract import (
    ActionContractValidationResult,
    canonical_dict,
    check_signing_key_available,
    compute_action_contract_hash,
    require_signing_key,
    validate_action_contract,
)

__all__ = [
    "ActionContractSpec",
    "ActionContractValidationResult",
    "MxactParseError",
    "RollbackSpec",
    "RateLimitSpec",
    "SandboxLimitsSpec",
    "CAPABILITIES",
    "HIGH_RISK_CAPABILITIES",
    "MUTATING_CAPABILITIES",
    "canonical_dict",
    "check_signing_key_available",
    "compute_action_contract_hash",
    "parse_mxact",
    "require_signing_key",
    "validate_action_contract",
    "CapabilityRegistry",
    "REQUIRED_SCOPE_FIELDS",
    "ScopeValidationResult",
    "registry",
    "DryRunReport",
    "DryRunSimulator",
    "RateTracker",
    "ActionExecutor",
    "ActionExecutorError",
    "ActionResult",
    "ExecutionContext",
    "SandboxedActionExecutor",
    "SandboxedExecutorError",
    "SandboxParams",
    "SandboxResult",
    "ActionTrace",
    "build_action_trace",
    "sign_action_trace",
    "verify_action_trace",
    "RollbackManager",
    "RollbackResult",
    "RollbackError",
    "ApprovalStore",
    "ApprovalError",
    "HumanApprovalGate",
    "PendingExecution",
]
