# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C2 — el contenedor de la pasada NO sobrevive a su tope.

El 2026-09-26 la pasada de TabICL pasó su tope de 7.200 s y siguió midiendo: `timeout` mató al
bash del trabajo, pero el `trap` del envoltorio no saltó (bash lo aplaza mientras `docker run`
está en primer plano) y python3, PID 1 del contenedor, ignoraba el SIGTERM. Murió solo al
borrarse el worktree bajo sus pies. Medido con una réplica de la misma cadena: con el patrón viejo
quedaba 1 contenedor vivo tras volver `timeout`; con `--init` y `& wait`, 0.

Estas pruebas atan las dos piezas, porque cada una sola no basta y las dos parecen adorno.
"""
from __future__ import annotations

import re
from pathlib import Path

GUION = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0" / "correr_116c2_en_contenedor.sh"


def _ultima_orden_docker_run() -> str:
    texto = GUION.read_text(encoding="utf-8")
    inicio = texto.rindex("docker run --rm")
    return texto[inicio:]


def test_la_pasada_corre_con_init_para_que_el_sigterm_llegue_a_python():
    assert "--init" in _ultima_orden_docker_run().split("\n", 1)[0]


def test_docker_run_va_en_segundo_plano_y_se_espera_con_wait():
    """Con `docker run` en primer plano el trap no corre hasta que el contenedor acaba."""
    cola = _ultima_orden_docker_run()
    assert re.search(r'pasada_116c2_tabicl\.py "\$\{ARGS\[@\]\}" &\s*\nwait \$!\s*$', cola), cola[-200:]


def test_el_trap_quita_el_contenedor_por_su_nombre_tambien_con_term():
    texto = GUION.read_text(encoding="utf-8")
    assert re.search(r"""trap 'docker rm -f "\$NOMBRE_CONTENEDOR"[^']*' EXIT INT TERM""", texto)
    assert '--name "$NOMBRE_CONTENEDOR"' in _ultima_orden_docker_run()
