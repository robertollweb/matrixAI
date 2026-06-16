# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

from __future__ import annotations

from typing import Any

_MSGS: dict[str, dict[str, str]] = {
    "no_actions": {
        "es": "No se activo ninguna accion discreta. El grafo fue evaluado en modo simulado.",
        "en": "No discrete action was triggered. The graph was evaluated in simulation mode.",
    },
    "activated": {
        "es": "activo",
        "en": "activated",
    },
    "not_activated": {
        "es": "no activo",
        "en": "not activated",
    },
    "action_status": {
        "es": "La accion {name} quedo {status} porque {source}={value:.4f} y el umbral era {threshold:.4f}.",
        "en": "Action {name} was {status} because {source}={value:.4f} and the threshold was {threshold:.4f}.",
    },
    "simulated_call": {
        "es": "La llamada {call} se ejecuto solo como simulacion.",
        "en": "Call {call} was executed as simulation only.",
    },
    "nodes_evaluated": {
        "es": "Nodos evaluados: {nodes}.",
        "en": "Nodes evaluated: {nodes}.",
    },
}


class AuditorAgent:
    def explain(self, result: dict[str, Any], locale: str = "es") -> str:
        loc = locale if locale in ("es", "en") else "es"
        actions = result.get("actions", [])
        trace = result.get("trace", [])

        if not actions:
            return _MSGS["no_actions"][loc]

        lines: list[str] = []
        for action in actions:
            status = _MSGS["activated"][loc] if action["activated"] else _MSGS["not_activated"][loc]
            lines.append(
                _MSGS["action_status"][loc].format(
                    name=action["name"],
                    status=status,
                    source=action["source"],
                    value=action["value"],
                    threshold=action["threshold"],
                )
            )
            if action["activated"]:
                lines.append(_MSGS["simulated_call"][loc].format(call=action["call"]))

        if trace:
            evaluated = ", ".join(item["node"] for item in trace)
            lines.append(_MSGS["nodes_evaluated"][loc].format(nodes=evaluated))

        return "\n".join(lines)
