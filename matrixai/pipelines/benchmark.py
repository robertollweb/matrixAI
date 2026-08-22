"""El benchmark base del runtime — contrato 81-C3 §13.4/§13.5.

> «Exista un benchmark reproducible contra el que comparar cortes
> posteriores.»

Para qué sirve de verdad: el motor se mete **entre** los nodos —verifica,
resuelve por digest, evalúa políticas, monta entradas, digesta salidas y
escribe traza—, y todo eso cuesta. Sin un número, la siguiente vez que
alguien añada una comprobación nadie sabrá si costó 2 % o 200 %.

**Lo que mide es el motor, no los modelos.** El nodo de referencia no
hace nada a propósito: si ejecutara algo, el número diría más del modelo
que del runtime, y comparar dos cortes dejaría de significar nada.

**Y lo que este módulo NO hace es aprobar o suspender.** No hay umbral de
tiempo: un umbral se pone en rojo el día que la máquina está cargada, y
una prueba que falla por el vecino acaba desactivada. Aquí se **mide y se
declara**; lo que sí se comprueba en la suite es que el resultado sea
**reproducible** —el mismo digest de salida en todas las pasadas—, que es
el criterio que el contrato escribió.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

from matrixai.pipelines.engine import ejecutar_pipeline

__all__ = ["pipeline_de_referencia", "correr_benchmark"]

#: El registry del pipeline de referencia. Un digest fijo, para que la
#: resolución cueste lo mismo en todas las pasadas.
_REGISTRY = {"referencia": "sha256:" + "b" * 64}


def pipeline_de_referencia(nodos: int = 20) -> dict[str, Any]:
    """Una cadena de `nodos` pasos, cada uno dependiendo del anterior.

    En cadena y no en abanico a propósito: es el caso que obliga a
    **transportar** de un nodo al siguiente, que es donde el motor hace
    su trabajo. Un abanico mediría sobre todo la ordenación.
    """
    return {
        "pipeline_id": "benchmark",
        "version": "1.0",
        "timeout_s": 600,
        "nodes": [
            {"id": f"n{i}", "model": "referencia",
             "entry_hash": _REGISTRY["referencia"], "kind": "nada",
             **({"depends_on": [f"n{i - 1}"]} if i else {})}
            for i in range(nodos)
        ],
    }


def _nodo_que_no_hace_nada(*, entradas: dict[str, Any], nodo: dict[str, Any],
                           contexto: dict[str, Any]) -> str:
    """Devuelve algo estable y barato: lo que se mide es el motor."""
    return nodo["id"]


def correr_benchmark(nodos: int = 20, pasadas: int = 5) -> dict[str, Any]:
    """Corre el pipeline de referencia y devuelve los números MEDIDOS.

    Devuelve también el digest de salida de cada pasada: si dos pasadas
    dieran digests distintos, el tiempo daría igual — el motor no sería
    reproducible, que es lo que el corte promete.
    """
    tiempos: list[float] = []
    digests: list[str] = []
    pipeline = pipeline_de_referencia(nodos)

    for _ in range(pasadas):
        empieza = time.perf_counter()
        salida = ejecutar_pipeline(
            pipeline, registry=_REGISTRY,
            ejecutores={"nada": _nodo_que_no_hace_nada})
        tiempos.append(time.perf_counter() - empieza)
        if salida["report"]["status"] != "ok":
            # Un benchmark sobre una ejecución que falló mide el camino de
            # error, no el del trabajo. Se corta y se dice.
            raise RuntimeError(
                f"el pipeline de referencia no terminó bien: "
                f"{salida['report']['status']} — {salida['report']['reason']}")
        digests.append(salida["report"]["outputs"][f"n{nodos - 1}"])

    return {
        "nodes": nodos,
        "runs": pasadas,
        "seconds_median": statistics.median(tiempos),
        "seconds_min": min(tiempos),
        "seconds_max": max(tiempos),
        "ms_per_node_median": (statistics.median(tiempos) / nodos) * 1000,
        # La comprobación que de verdad importa: mismo grafo, mismo
        # resultado. Un motor rápido que devuelve otra cosa cada vez no
        # sirve para sostener un recibo.
        "output_digest": digests[0],
        "reproducible": len(set(digests)) == 1,
    }


if __name__ == "__main__":  # pragma: no cover — se corre a mano
    import json

    print(json.dumps(correr_benchmark(), indent=2))
