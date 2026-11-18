# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""172-C1-bis — el guion del contenedor del 170 comprueba el EBM COMPLETO, no el ligero.

Desde el 172-C1-bis `motor_para("interpret.ebm")` devuelve el perfil LIGERO (el que usa el estudio); el 170 midio el completo (valores por
omision de interpret-core). Si `correr_c170_en_contenedor.sh` volviera a pedirlo por nombre, repetir el 170 mediria otro perfil sin decirlo.
"""
from __future__ import annotations

import re
from pathlib import Path

GUION = Path(__file__).resolve().parents[1] / "benchmarks" / "c170" / "correr_c170_en_contenedor.sh"


def _sin_comentarios(texto: str) -> str:
    return "\n".join(l for l in texto.splitlines() if not l.lstrip().startswith("#"))


def test_el_guion_del_170_no_pide_el_ebm_por_su_nombre_y_pide_el_completo_explicito():
    codigo = _sin_comentarios(GUION.read_text(encoding="utf-8"))
    assert not re.search(r"motor_para\(\s*[\"']interpret\.ebm[\"']", codigo), "pide el EBM por nombre: es el perfil LIGERO, no el que midio el 170"
    assert re.search(r"^\s*m\s*=\s*MotorEBM\(\)\s*$", codigo, re.M), "tiene que construir el perfil completo: MotorEBM() sin perfil"
    assert "from matrixai_engines.motores.interpret_ebm import MotorEBM" in codigo
