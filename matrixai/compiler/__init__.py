# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from matrixai.compiler.backend_contract import BackendContractAnalyzer, BackendContractReport, torch_backend_metadata
from matrixai.compiler.differentiable_python import DifferentiablePythonCompiler
from matrixai.compiler.python_backend import PythonBackendCompiler


def __getattr__(name: str):
	if name in {"TorchForwardError", "TorchForwardRunner"}:
		from matrixai.compiler.torch_forward import TorchForwardError, TorchForwardRunner

		return {"TorchForwardError": TorchForwardError, "TorchForwardRunner": TorchForwardRunner}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
	"BackendContractAnalyzer",
	"BackendContractReport",
	"DifferentiablePythonCompiler",
	"PythonBackendCompiler",
	"TorchForwardError",
	"TorchForwardRunner",
	"torch_backend_metadata",
]
