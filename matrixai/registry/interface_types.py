"""Los tipos de INTERFAZ de un modelo, deducidos de su propio `.mxai`.

Para qué: `check_composite_program_types` compara la salida de una pieza
con la entrada de la siguiente, y hace `if not dst_in: continue`. Con los
manifiestos que escribía `registry push` —`input_type: {}`,
`output_type: {}`— esa comprobación **no podía fallar nunca**: medido el
2026-08-19, una conexión imposible daba `Typecheck OK`, `rc=0`. Un
compositor que enseñe «tipos compatibles» apoyándose en eso estaría
afirmando por omisión.

Qué NO hace: inventar. Si la interfaz del programa no se puede determinar
sin ambigüedad —no hay `GRAPH`, entra por varios sitios, sale por varios,
o el extremo no es de una forma que se sepa describir— devuelve `None`, y
entonces la entrada se publica sin tipos y quien la lea lo verá dicho
(«Publicada sin declarar tipos»). Un tipo inventado es peor que ninguno:
daría por buena una conexión que no lo es.

Los tipos **no entran en `entry_hash`** (`compute_entry_hash` los excluye
a propósito), así que anotarlos no toca la cadena de integridad de P21.
"""

from typing import Any

__all__ = ["deducir_tipos_de_interfaz"]


def _fuentes_y_sumideros(edges: list[tuple[str, str]]) -> tuple[set[str], set[str]]:
    origenes = {a for a, _ in edges}
    destinos = {b for _, b in edges}
    return origenes - destinos, destinos - origenes


def _tipo_de_entrada(program: Any, nodo: str) -> dict | None:
    for vector in getattr(program, "vectors", []) or []:
        if vector.name == nodo:
            # `size` es imprescindible: sin la talla no se puede comprobar
            # que una conexión encaje, solo que el `kind` coincide.
            talla = getattr(vector, "size", None)
            if type(talla) is not int:
                return None
            return {"name": vector.name, "kind": "VECTOR", "size": talla}
    for seq in getattr(program, "sequences", []) or []:
        if seq.name == nodo:
            return {"name": seq.name, "kind": "SEQUENCE"}
    return None


def _tipo_de_salida(program: Any, nodo: str) -> dict | None:
    for red in getattr(program, "networks", []) or []:
        if red.name == nodo:
            salida = getattr(red, "output", None)
            tipo = getattr(red, "output_type_str", None)
            if not salida or not tipo:
                return None
            return {"name": salida, "kind": tipo}
    return None


def deducir_tipos_de_interfaz(program: Any) -> tuple[dict | None, dict | None]:
    """`(input_type, output_type)` del programa, o `None` donde no consta.

    La entrada es el único nodo por el que ENTRA el grafo (sin aristas
    que lleguen a él) y la salida el único por el que SALE. Con más de
    uno la interfaz es ambigua y no se declara: elegir «el primero»
    publicaría un tipo que puede no ser el que conecta.
    """
    grafo = getattr(program, "graph", None)
    edges = list(getattr(grafo, "edges", []) or []) if grafo is not None else []
    if not edges:
        return None, None

    fuentes, sumideros = _fuentes_y_sumideros(edges)
    entrada = _tipo_de_entrada(program, next(iter(fuentes))) if len(fuentes) == 1 else None
    salida = _tipo_de_salida(program, next(iter(sumideros))) if len(sumideros) == 1 else None
    return entrada, salida
