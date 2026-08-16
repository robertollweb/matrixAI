# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""A made-up function call must be reported, not silently evaluated to 0.0.

Decided by Roberto (2026-08-16) after this was measured: `mifuncion(x)`
passed the parser, the type checker and the verifier with zero findings,
and then `CallNode.eval` returned `0.0` — a constant dressed up as a
computation. The model trained, predicted and exported, and the number
meant nothing.

WARNING and not an error, on purpose: a name can still resolve at run time
from the evaluation environment, so refusing it would break `.mxai` files
that work today.
"""
from __future__ import annotations

from pathlib import Path

from matrixai.parser.parser import parse_text
from matrixai.types import check_program_types

AVISO = "no known implementation"


def _programa(expr: str) -> str:
    return f"""PROJECT P

VECTOR E[1]
  x: Scalar
END

FUNCTION F
  y = {expr}
END

GRAPH
  E -> F
END
"""


def _avisos(expr: str) -> list[str]:
    r = check_program_types(parse_text(_programa(expr)))
    return [w for w in r.warnings if AVISO in w]


def test_una_funcion_inventada_avisa_con_su_nombre():
    avisos = _avisos("mifuncion_inventada(x)")
    assert len(avisos) == 1
    # El NOMBRE y la CONSECUENCIA: sin los dos, el aviso no lleva a nada.
    assert "mifuncion_inventada" in avisos[0]
    assert "0.0" in avisos[0]


def test_no_es_un_error_el_modelo_sigue_siendo_valido():
    """Romper un `.mxai` que hoy compila no era la decisión."""
    r = check_program_types(parse_text(_programa("mifuncion_inventada(x)")))
    assert r.errors == []
    assert r.ok is True


def test_las_funciones_CON_semantica_no_avisan():
    """ARREGLAR UN SESGO PUEDE CREAR EL CONTRARIO — se prueban los dos lados.

    Un aviso que acusa a una función legítima enseña a ignorar los avisos,
    y entonces el de verdad tampoco se lee.
    """
    for expr in (
        "sigmoid(x)",        # builtin de `CallNode.eval`
        "clip(x, 0, 1)",     # builtin con varios argumentos
        "confidence(x)",     # firma nativa, se resuelve del entorno
        "softmax(x)",        # firma nativa
        "x * 2",             # sin llamada ninguna
    ):
        assert _avisos(expr) == [], f"acusó a {expr!r}, que sí tiene semántica"


def test_NINGUN_ejemplo_del_repositorio_dispara_el_aviso():
    """El criterio duro: si un `.mxai` legítimo lo dispara, está mal EL AVISO.

    Se recorren los ejemplos que se publican con el core. Si alguien añade
    uno que use una función sin semántica, esto se pone rojo y hay que
    mirarlo — que es exactamente para lo que sirve.
    """
    raiz = Path(__file__).resolve().parent.parent / "examples"
    analizados = 0
    culpables: list[str] = []
    for fichero in sorted(raiz.rglob("*.mxai")):
        try:
            r = check_program_types(parse_text(fichero.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — un ejemplo que no parsea no es cosa de esta prueba
            continue
        analizados += 1
        culpables += [f"{fichero.name}: {w}" for w in r.warnings if AVISO in w]
    # Que se hayan analizado de verdad: con 0 ficheros esto pasaría vacío.
    assert analizados >= 20, f"solo se analizaron {analizados} ejemplos"
    assert culpables == []
