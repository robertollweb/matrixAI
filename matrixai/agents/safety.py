# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from matrixai.ir import MatrixAIProgram
from matrixai.sandbox import SandboxPolicy


class SafetyAgent:
    def review(self, program: MatrixAIProgram) -> list[str]:
        return SandboxPolicy.mvp_simulate_only().review(program).messages()