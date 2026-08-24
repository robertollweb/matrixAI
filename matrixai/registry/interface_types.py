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


def _describir_salida(nombre: str, anotacion: str) -> dict | None:
    """La anotación del DSL, dicha en el MISMO vocabulario que la entrada.

    POR QUÉ EXISTE (2026-08-23). Los dos extremos se describían en
    alfabetos distintos: la entrada normalizada —`{kind: "VECTOR", size:
    n}`, sacada del nodo por el que entra el grafo— y la salida en CRUDO,
    la cadena tal cual la escribió el `.mxai` (`"Score"`, `"Vector[1]"`,
    `"Tensor[3]"`). `_composite_types_compatible` compara `kind` con
    `kind`, así que **`"Vector[1]"` no casaba ni con un `VECTOR`**, y
    ninguna pareja publicada por `registry push` podía encajar jamás:
    medido sobre `examples/text-routing`, que enruta con 100 % de acierto
    y su propio verificador rechazaba.

    Esto NO afloja la comprobación —un `Score` sigue sin entrar en un
    `VECTOR[1]`, y las tallas se siguen comparando—: solo hace que los
    dos lados hablen el mismo idioma, que es la condición para que
    comparar signifique algo.

    Y no inventa: una anotación que no se sabe leer devuelve `None`, y
    entonces la entrada se publica **sin tipos** y quien la lea lo verá
    dicho, que es lo que ya hacía este módulo con lo ambiguo.
    """
    from ..types import parse_type_spec

    try:
        spec = parse_type_spec(anotacion)
    except ValueError:
        return None
    if spec.name == "Vector":
        dim = spec.parameters.get("dim")
        tipo: dict = {"name": nombre, "kind": "VECTOR"}
        if isinstance(dim, int):
            tipo["size"] = dim
        return tipo
    if spec.name == "Tensor":
        forma = spec.parameters.get("shape")
        tipo = {"name": nombre, "kind": "Tensor"}
        if isinstance(forma, list) and forma:
            tipo["shape"] = list(forma)
        return tipo
    # El resto se queda con su nombre CANÓNICO —`Score`, `Probability`,
    # `ProbabilityMap`—: sin los corchetes, que son las etiquetas y no el
    # tipo, y por eso hacían que dos salidas del mismo tipo con distintas
    # etiquetas se leyeran como tipos distintos.
    return {"name": nombre, "kind": spec.name}


def _tipo_de_salida(program: Any, nodo: str) -> dict | None:
    for red in getattr(program, "networks", []) or []:
        if red.name == nodo:
            salida = getattr(red, "output", None)
            tipo = getattr(red, "output_type_str", None)
            if not salida or not tipo:
                return None
            return _describir_salida(salida, tipo)
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
