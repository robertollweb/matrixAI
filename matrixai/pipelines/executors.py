"""Ejecutores de nodo de verdad — contrato 81-C3.

El motor (`engine.py`) sabe orquestar y no sabe ejecutar nada: recibe un
`kind → función` y llama. Aquí está la función que ejecuta **un modelo
real del registry de P21**, que es lo que convierte el 81 de una caja de
reglas correctas en algo que corre.

Tres reglas, y ninguna es paranoia de más:

* **El digest se comprueba OTRA VEZ al ejecutar.** La verificación previa
  mira el registry en el segundo cero; entre ese momento y el nodo pueden
  pasar minutos, y el registry es un directorio en disco. Un recibo que
  se apoya en una comprobación de hace diez minutos afirma sobre algo que
  ya no miró.
* **Una entrada manipulada NO ejecuta.** `verify` de P21 existe justo para
  eso, y saltárselo aquí dejaría que un modelo tocado produjera una
  salida con recibo.
* **Los tipos se comprueban si están declarados, y si no, se DICE.** Una
  entrada publicada sin tipos no se puede contrastar; decir «comprobado»
  igualmente sería afirmar por omisión, que es el defecto que este mismo
  repositorio arregló en `check_composite_program_types`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

__all__ = ["EntradaAmbigua", "ModeloAlterado", "ModeloCambiado", "SinRecetaDeEnlace",
           "TipoDeEntradaIncompatible", "ejecutor_de_modelos", "mapa_del_registry"]


class ModeloCambiado(RuntimeError):
    """Lo que hay en el registry ya no es lo que el pipeline declaró."""


class ModeloAlterado(RuntimeError):
    """La entrada del registry no pasa su propia verificación de integridad."""


class EntradaAmbigua(ValueError):
    """Hay más de una fuente para la entrada del nodo, y no se elige una."""


class TipoDeEntradaIncompatible(ValueError):
    """Lo que llega no encaja con lo que el modelo declara esperar."""


class SinRecetaDeEnlace(ValueError):
    """Un nodo recibe la salida de otro y no dice cómo se enlaza.

    **No se mapea automáticamente.** La salida de un modelo es su estado
    entero; adivinar qué parte alimenta el vector del siguiente sería
    inventar una receta que nadie escribió, y el resultado saldría
    plausible y equivocado — con recibo.
    """


def _partir(referencia: str) -> tuple[str, str]:
    """`nombre@version`. La versión es OBLIGATORIA.

    Un nombre a secas obligaría a elegir «la última», y elegir por
    nosotros es exactamente lo que el contrato prohíbe: la traza diría
    «corrió con lo que hubiera».
    """
    if "@" not in referencia:
        raise ModeloCambiado(
            f"{referencia!r} no dice versión: un nombre puede apuntar hoy a una "
            "cosa y mañana a otra, y la traza no se podría comprobar después")
    nombre, _, version = referencia.partition("@")
    if not nombre.strip() or not version.strip():
        raise ModeloCambiado(f"{referencia!r} no es una referencia de modelo")
    return nombre.strip(), version.strip()


def mapa_del_registry(registry: Any) -> dict[str, str]:
    """El `nombre@version → entry_hash` que el motor necesita, MEDIDO.

    Se construye leyendo el registry, no escribiéndolo a mano: un mapa
    copiado a mano es un segundo sitio declarando lo mismo, y acabaría
    divergiendo del registry que dice describir.
    """
    return {f"{e.name}@{e.version}": e.entry_hash for e in registry.list()}


def _valor_del_estado(producido: Any, nombre: str, node_id: str, campo: str) -> Any:
    """El valor que la receta pide, sacado del estado de quien produjo.

    Un nombre que no está NO se rellena con cero: el nodo correría con un
    hueco. Y una lista de más de un elemento no se recorta al primero —
    eso sería elegir por el modelo.
    """
    estado = producido.get("state") if isinstance(producido, dict) else None
    if not isinstance(estado, dict) or nombre not in estado:
        disponibles = sorted(estado)[:8] if isinstance(estado, dict) else []
        raise SinRecetaDeEnlace(
            f"el nodo {node_id!r} enlaza {campo!r} con {nombre!r} y quien le "
            f"precede no ha producido nada con ese nombre (hay: {disponibles})")
    valor = estado[nombre]
    if isinstance(valor, list):
        if len(valor) != 1:
            raise SinRecetaDeEnlace(
                f"el nodo {node_id!r} enlaza {campo!r} con {nombre!r}, que tiene "
                f"{len(valor)} valores y el campo admite uno: quedarse con el "
                "primero sería elegir por el modelo")
        return valor[0]
    return valor


def _entrada_enlazada(nodo: dict[str, Any], entradas: dict[str, Any]) -> Any:
    """La entrada compuesta con la receta que el nodo DECLARA.

    `input_map: {"Signal": {"routing_signal": "routing_signal"}}` se lee
    «el campo `routing_signal` del vector `Signal` toma lo que mi
    predecesor dejó en `routing_signal`».
    """
    receta = nodo.get("input_map")
    node_id = str(nodo.get("id") or "?")
    if not isinstance(receta, dict) or not receta:
        raise SinRecetaDeEnlace(
            f"el nodo {node_id!r} recibe la salida de {sorted(entradas)} y no "
            "declara `input_map`: la salida de un modelo es su estado entero, y "
            "adivinar qué parte alimenta su vector sería inventar una receta que "
            "nadie escribió")
    if len(entradas) != 1:
        raise EntradaAmbigua(
            f"el nodo {node_id!r} recibe {sorted(entradas)} y la receta de enlace "
            "solo sabe leer de uno: hay que decir de cuál")
    producido = next(iter(entradas.values()))

    construida: dict[str, Any] = {}
    for vector, campos in receta.items():
        if not isinstance(campos, dict) or not campos:
            raise SinRecetaDeEnlace(
                f"el nodo {node_id!r} declara el vector {vector!r} sin campos: un "
                "vector vacío no es una entrada")
        construida[str(vector)] = {
            str(campo): _valor_del_estado(producido, str(origen), node_id, str(campo))
            for campo, origen in campos.items()
        }
    return construida


def _entrada_del_nodo(nodo: dict[str, Any], entradas: dict[str, Any]) -> Any:
    """De dónde sale el dato: de la declaración o de quien le precede.

    Si hay las dos cosas, o hay dos predecesores, **no se elige**: un nodo
    que corre con una de dos entradas posibles da un resultado que nadie
    puede explicar después.
    """
    declarada = nodo.get("input")
    if declarada is not None and entradas:
        raise EntradaAmbigua(
            f"el nodo {nodo.get('id')!r} declara su propia entrada Y recibe la de "
            f"{sorted(entradas)}: elegir una haría que la traza describiera algo "
            "distinto de lo que corrió")
    if declarada is not None:
        return declarada
    if not entradas:
        raise EntradaAmbigua(
            f"el nodo {nodo.get('id')!r} no declara entrada ni depende de nadie: "
            "no hay con qué ejecutarlo")
    return _entrada_enlazada(nodo, entradas)


def _comprobar_tipo(entrada: Any, input_type: Any, node_id: str) -> str:
    """Devuelve qué se comprobó — o por qué no se pudo."""
    if not isinstance(input_type, dict) or not input_type:
        # Sin tipos declarados no hay nada que contrastar. Se DICE, en vez
        # de dejar que un «ok» se lea como «encaja».
        return "sin tipos declarados en el registry: la entrada no se ha contrastado"
    nombre = input_type.get("name")
    if not isinstance(entrada, dict):
        raise TipoDeEntradaIncompatible(
            f"el nodo {node_id!r} espera un objeto con el vector {nombre!r} y "
            f"recibe {type(entrada).__name__}")
    if nombre and nombre not in entrada:
        raise TipoDeEntradaIncompatible(
            f"el nodo {node_id!r} espera el vector {nombre!r} y lo que llega trae "
            f"{sorted(entrada)}")
    talla = input_type.get("size")
    campos = entrada.get(nombre) if nombre else None
    if type(talla) is int and isinstance(campos, dict) and len(campos) != talla:
        # Las DOS tallas en el mensaje: «no encaja» a secas no deja ver si
        # falta un campo o si es otro vector entero.
        raise TipoDeEntradaIncompatible(
            f"el nodo {node_id!r} espera {talla} campos en {nombre!r} y recibe "
            f"{len(campos)}")
    return f"entrada contrastada contra {nombre!r}"


def ejecutor_de_modelos(
    registry: Any, *, comprobar_integridad: bool = True
) -> Callable[..., Any]:
    """Un ejecutor que corre modelos REALES del registry de P21."""
    from matrixai.parser import parse_text
    from matrixai.runtime import MatrixAIRuntime

    def ejecutar(*, entradas: dict[str, Any], nodo: dict[str, Any],
                 contexto: dict[str, Any]) -> Any:
        node_id = str(nodo.get("id") or "?")
        nombre, version = _partir(str(nodo.get("model") or ""))
        entrada_reg = registry.get(nombre, version)

        declarado = nodo.get("entry_hash")
        if entrada_reg.entry_hash != declarado:
            raise ModeloCambiado(
                f"el nodo {node_id!r} declaró {declarado} para {nombre}@{version} y "
                f"ahora el registry tiene {entrada_reg.entry_hash}: seguir produciría "
                "una traza que describe otro modelo")

        if comprobar_integridad:
            # Medido: `verify` de P21 LEVANTA `VerificationError` cuando el
            # artefacto no cuadra; solo devuelve `False` en otros casos. Con
            # un `if not verify(...)` a secas el motivo que sale es el de
            # P21 y este `ModeloAlterado` no se alcanzaría nunca — un
            # camino que se lee en el código y no existe al ejecutar.
            from matrixai.registry.model_registry import VerificationError
            try:
                intacto = registry.verify(nombre, version)
            except VerificationError as exc:
                raise ModeloAlterado(
                    f"{nombre}@{version} no pasa su verificación de integridad "
                    f"({exc}): un modelo tocado no ejecuta, porque su salida "
                    "saldría con recibo") from exc
            if not intacto:
                raise ModeloAlterado(
                    f"{nombre}@{version} no pasa su propia verificación de "
                    "integridad: un modelo tocado no ejecuta, porque su salida "
                    "saldría con recibo")

        directorio = Path(registry.layout.entry_dir(nombre, version))
        texto = (directorio / "model.mxai").read_text(encoding="utf-8")
        parametros: dict[str, Any] = {}
        fichero_params = directorio / "params.json"
        if fichero_params.exists():
            from matrixai.parameters import ParameterSet
            parametros = ParameterSet.from_dict(
                json.loads(fichero_params.read_text(encoding="utf-8"))
            ).runtime_parameters()

        dato = _entrada_del_nodo(nodo, entradas)
        comprobacion = _comprobar_tipo(dato, entrada_reg.input_type, node_id)

        traza = contexto.get("traza")
        if traza is not None:
            # Lo comprobado —y lo NO comprobado— viaja en la traza como una
            # política más: es una decisión que se tomó al ejecutar, y sin
            # dejarla escrita el recibo no podría distinguir «encajaba» de
            # «no se pudo mirar».
            traza.anotar_politica(node_id, {
                "decision": "allow", "rule_id": "input_type",
                "explain": comprobacion,
                "policy": "registry_input_type", "policy_version": "1.0",
            })

        return MatrixAIRuntime().run(parse_text(texto), dato, parameters=parametros)

    return ejecutar
