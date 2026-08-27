"""El runtime: resolver por digest y respetar el orden — contrato 81-C3.

Tres reglas, y cada una evita una forma distinta de acabar con una traza
que no describe lo que pasó:

* **Se resuelve por DIGEST, no por nombre.** Un nombre puede apuntar hoy
  a una cosa y mañana a otra; aceptar «el que haya» convertiría la traza
  en una promesa que después nadie puede comprobar.
* **Se verifica ANTES de ejecutar.** Empezar un pipeline roto deja
  trabajo a medias y efectos ya ejecutados que no se pueden deshacer.
* **Se dicen TODOS los problemas**, no el primero: arreglar uno y volver
  a chocar con el siguiente es hacer trabajar a alguien de más.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CicloEnElPipeline", "DigestNoCoincide", "EntradaQueFalta",
           "SalidaAlterada", "entradas_para", "ordenar_por_dependencias",
           "resolver_por_digest", "verificar_antes_de_ejecutar"]


class DigestNoCoincide(ValueError):
    """El modelo que hay no es el que el pipeline declara."""


class CicloEnElPipeline(ValueError):
    """El grafo no se puede recorrer: hay un ciclo o falta un nodo."""


class EntradaQueFalta(ValueError):
    """Un nodo esperaba la salida de otro y no está."""


class SalidaAlterada(ValueError):
    """Lo que se iba a transportar no es lo que su productor declaró."""


#: Los `kind` cuyos modelos NO viven en el registry (87-C1). Un ONNX ajeno es
#: un FICHERO: no hay entrada de registry contra la que contrastar su digest, y
#: exigirla haría que un pipeline honesto no arrancara nunca.
#:
#: La lista es CERRADA a propósito: lo que no está aquí se resuelve por el
#: registry, como siempre. Y lo que está **sigue teniendo que declarar su
#: digest** — quien lo comprueba es su ejecutor, al abrir el fichero y otra vez
#: al ejecutarlo.
KINDS_FUERA_DEL_REGISTRY = ("onnx",)


def _resolver_si_toca(nodo: dict, registry: dict[str, str]) -> None:
    """Resuelve el modelo del nodo contra el registry — salvo que su `kind` no
    viva ahí (87-C1), y entonces se exige el digest igual.

    Saltarse la comprobación entera para esos nodos habría dejado entrar un
    pipeline que dice «corre este onnx» sin decir CUÁL, que es exactamente lo
    que `resolver_por_digest` existe para impedir.
    """
    if str(nodo.get("kind") or "") in KINDS_FUERA_DEL_REGISTRY:
        if not str(nodo.get("entry_hash") or "").strip():
            raise DigestNoCoincide(
                f"el paso que usa {str(nodo.get('model') or '')!r} no declara su "
                "digest, y sin él la traza diría que corrió «un modelo», no CUÁL")
        return
    resolver_por_digest(str(nodo.get("model") or ""), nodo.get("entry_hash"), registry)


def resolver_por_digest(modelo: str, digest: str | None, registry: dict[str, str]) -> str:
    """Devuelve el digest resuelto, o falla.

    **Sin digest declarado NO se resuelve**: es la diferencia entre «este
    pipeline corrió con ESTE modelo» y «corrió con lo que hubiera bajo
    ese nombre».
    """
    if not digest:
        raise DigestNoCoincide(
            f"el paso que usa {modelo!r} no declara su digest, y resolver por "
            "nombre haría que la traza no pudiera comprobarse después: un "
            "nombre puede apuntar hoy a una cosa y mañana a otra")
    disponible = registry.get(modelo)
    if disponible is None:
        raise DigestNoCoincide(
            f"{modelo!r} no está en este registry, así que no hay con qué "
            "contrastar el digest que el pipeline declara")
    if disponible != digest:
        raise DigestNoCoincide(
            f"{modelo!r} declara {digest} y aquí hay {disponible}: seguir "
            "produciría una traza que describe otro modelo")
    return disponible


def ordenar_por_dependencias(grafo: dict[str, list[str]]) -> list[str]:
    """Orden topológico ESTABLE del grafo.

    Estable a propósito: dos ejecuciones del mismo pipeline tienen que dar
    el mismo orden, o dos trazas del mismo grafo no se podrían comparar.
    """
    pendientes = list(grafo)          # el orden de declaración, para estabilidad
    resueltos: list[str] = []
    visitando: set[str] = set()
    hecho: set[str] = set()

    def visitar(nodo: str, camino: list[str]) -> None:
        if nodo in hecho:
            return
        if nodo in visitando:
            ciclo = " → ".join([*camino, nodo])
            raise CicloEnElPipeline(
                f"hay un ciclo en el pipeline ({ciclo}): empezarlo dejaría "
                "trabajo a medias y efectos que no se pueden deshacer")
        if nodo not in grafo:
            raise CicloEnElPipeline(
                f"un paso depende de {nodo!r}, que no existe en este pipeline")
        visitando.add(nodo)
        for dependencia in grafo[nodo]:
            visitar(dependencia, [*camino, nodo])
        visitando.discard(nodo)
        hecho.add(nodo)
        resueltos.append(nodo)

    for nodo in pendientes:
        visitar(nodo, [])
    return resueltos


def verificar_antes_de_ejecutar(pipeline: Any, registry: dict[str, str]) -> dict[str, Any]:
    """Comprueba el pipeline entero y devuelve **todos** sus problemas."""
    problemas: list[str] = []
    if not isinstance(pipeline, dict):
        return {"ok": False, "problems": ["un pipeline es un objeto"], "order": []}

    nodos = pipeline.get("nodes")
    if not isinstance(nodos, list) or not nodos:
        return {"ok": False, "problems": ["un pipeline sin pasos no ejecuta nada"],
                "order": []}

    # AUDITORÍA 1ª pasada [CRÍTICO]: sin `pipeline_id`, la raíz de la traza
    # caía a un `"run"` inventado — y **todas** las ejecuciones anónimas
    # compartían raíz, con lo que dos trazas distintas no se podían
    # distinguir. El contrato pide «generar un identificador raíz» y «atar
    # los nodos a su raíz»: una raíz fabricada e igual para todos no es un
    # identificador, es un nombre.
    if not str(pipeline.get("pipeline_id") or "").strip():
        problemas.append(
            "el pipeline no declara `pipeline_id`, y la raíz de la traza no "
            "se inventa: dos ejecuciones sin id compartirían raíz y sus "
            "trazas no se podrían distinguir")

    limite = pipeline.get("timeout_s")
    if type(limite) is not int or limite <= 0:
        # Ni cero ni negativo: uno mata antes de empezar y el otro no
        # limita nada, y las dos cosas mienten sobre lo que promete el
        # campo.
        problemas.append(
            f"timeout_s debe ser un entero de segundos mayor que 0, y es {limite!r}")

    grafo: dict[str, list[str]] = {}
    vistos: set[str] = set()
    #: Los que un id repetido deja fuera del grafo. No entran en él, pero
    #: sus problemas se cuentan igual: ver abajo.
    duplicados: list[dict[str, Any]] = []
    for nodo in nodos:
        if not isinstance(nodo, dict) or not nodo.get("id"):
            problemas.append("hay un paso sin id")
            continue
        node_id = str(nodo["id"])
        if node_id in vistos:
            # AUDITORÍA 1ª pasada (2026-08-20) [BLOQUEANTE]: dos pasos con
            # el mismo id corrían con `status: ok` y **uno de ellos
            # desaparecía** — el grafo es un diccionario y el segundo
            # pisaba al primero. Un nodo declarado que nunca corrió y nadie
            # lo dijo es justo lo que este runtime existe para no permitir,
            # y además «el paso n0» no identificaría a ninguno de los dos.
            # Es la misma regla que el verificador de recibos ya aplica a
            # los campos críticos ambiguos.
            problemas.append(
                f"hay dos pasos con el id {node_id!r}: «el paso {node_id}» no "
                "identificaría a ninguno de los dos, y uno se perdería sin "
                "que nadie lo dijera")
            # PERO NO SE SALTA LO SUYO.
            #
            # Aquí había un `continue`, y con él **el id repetido tapaba
            # todo lo que ese nodo habría dicho**: su modelo fuera del
            # registry, su dependencia inexistente. Medido (sonda
            # adversaria del C3, 2026-08-20): los mismos dos nodos daban
            # TRES problemas con ids distintos y solo DOS con el id
            # repetido.
            #
            # Es justo lo que la promesa de «todos los problemas» existe
            # para evitar: arreglabas el id, volvías a lanzar, y
            # aparecían dos problemas nuevos que ya estaban ahí. En el
            # grafo no entra —pisaría al primero, que es el motivo
            # original del `continue`—, pero se le mira igual.
            duplicados.append(nodo)
            continue
        vistos.add(node_id)
        grafo[node_id] = [str(d) for d in (nodo.get("depends_on") or [])]
        try:
            _resolver_si_toca(nodo, registry)
        except DigestNoCoincide as exc:
            problemas.append(str(exc))

    # Lo de los nodos tapados por un id repetido, dicho también.
    for nodo in duplicados:
        try:
            _resolver_si_toca(nodo, registry)
        except DigestNoCoincide as exc:
            problemas.append(str(exc))
        for dependencia in (nodo.get("depends_on") or []):
            if str(dependencia) not in vistos:
                problemas.append(
                    f"un paso depende de {str(dependencia)!r}, que no existe "
                    "en este pipeline")

    # AUDITORÍA 1ª pasada [BLOQUEANTE]: una política rota solo se
    # descubría **al llegar a su nodo**, con lo que los nodos anteriores ya
    # habían corrido —y sus efectos externos con ellos—. La verificación
    # previa existe precisamente para que «empezar un pipeline roto» no
    # deje trabajo a medias: si valida el grafo y no las políticas, valida
    # media cosa.
    from matrixai.pipelines.policy import PoliticaInvalida, evaluar_politica

    for nodo in nodos:
        if not isinstance(nodo, dict):
            continue
        politica = nodo.get("policy") if nodo.get("policy") is not None else pipeline.get("policy")
        if politica is None:
            continue
        try:
            # Con un contexto VACÍO: aquí no se decide nada, solo se
            # comprueba que la política se puede leer. Una regla que no
            # dispare por falta de datos no es un problema del pipeline.
            evaluar_politica(politica, {})
        except PoliticaInvalida as exc:
            problemas.append(
                f"la política de {str(nodo.get('id') or '?')!r} no se puede "
                f"leer: {exc}")

    orden: list[str] = []
    try:
        orden = ordenar_por_dependencias(grafo)
    except CicloEnElPipeline as exc:
        problemas.append(str(exc))

    return {"ok": not problemas, "problems": problemas, "order": orden}


def entradas_para(
    nodo: str,
    depende_de: list[str],
    salidas: dict[str, dict[str, Any]],
    *,
    comprobar_digests: bool = False,
) -> dict[str, Any]:
    """Lo que este nodo recibe, tomado de lo que produjeron los suyos.

    **Lo que falta NO se rellena.** Ni con `None` ni con vacío: el nodo
    correría con un hueco y devolvería algo plausible y equivocado, que
    es el fallo que después nadie sabe explicar. Se para y se dice quién
    esperaba qué.

    **Haber corrido no es haber producido**: una salida guardada como
    `None` es un hueco, y pasarla adelante lo esconde.
    """
    import hashlib

    entradas: dict[str, Any] = {}
    for dependencia in depende_de:
        producido = salidas.get(dependencia)
        if not isinstance(producido, dict) or producido.get("value") is None:
            raise EntradaQueFalta(
                f"el nodo {nodo!r} espera la salida de {dependencia!r} y no la "
                "hay: o no ha corrido todavía, o corrió sin producir nada. No "
                "se rellena con un hueco, porque el nodo devolvería algo "
                "plausible y equivocado")
        valor = producido["value"]

        digest = producido.get("digest")
        if comprobar_digests and digest:
            # Solo si el productor lo DECLARÓ: quien no declaró digest no
            # es quien miente, y exigirlo pararía pipelines honestos.
            crudo = valor if isinstance(valor, bytes) else str(valor).encode("utf-8")
            real = "sha256:" + hashlib.sha256(crudo).hexdigest()
            if real != digest:
                raise SalidaAlterada(
                    f"la salida de {dependencia!r} no es la que declaró "
                    f"({digest} frente a {real}): no se entrega a {nodo!r}, "
                    "porque un nodo que recibe algo distinto de lo que su "
                    "predecesor produjo da un resultado que no describe nada")
        entradas[dependencia] = valor
    return entradas
