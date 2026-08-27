"""El motor que ejecuta un pipeline de extremo a extremo — contrato 81-C3.

Hasta aquí el 81 tenía siete piezas correctas y **nada que las uniera**:
la resolución por digest, el orden estable, la verificación previa, el
transporte, la traza, las políticas y el recibo vivían cada una por su
lado. Este módulo es el cableado — que es donde este proyecto lleva
catorce veces encontrando el hueco.

Las reglas que fija, y ninguna es de comodidad:

* **Se verifica ANTES y no se empieza si algo falla.** Un pipeline roto
  que arranca deja trabajo a medias y efectos ya ejecutados que no se
  pueden deshacer. Se levantan TODOS los problemas juntos.
* **Un `kind` sin ejecutor NO corre.** Ni se salta ni cae en un ejecutor
  por defecto: un nodo que devuelve algo plausible sin haber ejecutado
  nada es exactamente el fallo que después nadie sabe explicar.
* **`abstain` no autoriza.** «Esta política no opina» no es un permiso; si
  se declaró una política y no dice `allow`, el nodo no corre.
* **`dry-run` no ejecuta NADA**, y por eso **no emite recibo**: simular
  no es ejecutar, y un recibo de una simulación se leería como prueba de
  que algo pasó.
* **Lo que no llegó a arrancar se DICE.** Un pipeline que para a la
  mitad y solo enseña lo que corrió se lee como una ejecución completa.
* **El límite de tiempo se comprueba ENTRE nodos, y así se declara.** Este
  runtime no puede interrumpir a un nodo por dentro; prometerlo sería
  peor que no ofrecerlo. A quien pueda respetarlo se le pasa el tiempo
  que queda, y el informe dice cuál de las dos cosas pasó.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Callable

from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.policy import evaluar_politica
from matrixai.pipelines.runtime import entradas_para, verificar_antes_de_ejecutar
from matrixai.pipelines.trace import ReintentoNoSeguro, TrazaDeEjecucion

__all__ = [
    "Cancelacion",
    "EjecutorDesconocido",
    "PipelineInvalido",
    "ReciboImposibleDeEmitir",
    "digest_de",
    "ejecutar_pipeline",
    "emitir_recibo",
]

#: Los estados finales de una ejecución. Cada uno dice algo distinto y
#: por eso no se colapsan: `failed` es «un nodo se rompió», `denied` es
#: «una política dijo que no» —que no es un fallo del pipeline—,
#: `timed_out` y `cancelled` son «no se llegó al final», y hay que poder
#: distinguirlos sin leer la traza entera.
ESTADOS = ("ok", "failed", "denied", "timed_out", "cancelled")


class PipelineInvalido(ValueError):
    """El pipeline no pasó la verificación previa: no ha ejecutado nada.

    Lleva la lista ENTERA de problemas: arreglar uno y volver a chocar
    con el siguiente es hacer trabajar a alguien de más.
    """

    def __init__(self, problemas: list[str]) -> None:
        self.problemas = list(problemas)
        super().__init__(
            "el pipeline no se ejecuta porque no pasa la verificación previa: "
            + "; ".join(self.problemas))


class EjecutorDesconocido(RuntimeError):
    """Nadie sabe ejecutar este nodo, y no se inventa quién."""


class ReciboImposibleDeEmitir(ValueError):
    """Lo que hay no sostiene un recibo, y firmarlo lo haría afirmar de más."""


class Cancelacion:
    """Un testigo que alguien de fuera puede activar durante la ejecución.

    No mata nada por dentro —este runtime no puede—: impide que arranque
    lo que todavía no ha arrancado, que es justo lo que pide el criterio
    «una cancelación detiene las operaciones pendientes».
    """

    def __init__(self) -> None:
        self._motivo: str | None = None

    def cancelar(self, motivo: str = "cancelado por quien lo lanzó") -> None:
        if self._motivo is None:
            self._motivo = str(motivo or "cancelado sin motivo declarado")

    @property
    def activa(self) -> bool:
        return self._motivo is not None

    @property
    def motivo(self) -> str | None:
        return self._motivo


def digest_de(valor: Any) -> str:
    """El sha256 de una salida, con la MISMA regla que el transporte.

    Si aquí se calculara de otra forma que en `entradas_para`, todo
    digest declarado por el motor fallaría al comprobarse — y el fallo
    parecería una manipulación.
    """
    crudo = valor if isinstance(valor, bytes) else str(valor).encode("utf-8")
    return "sha256:" + hashlib.sha256(crudo).hexdigest()


def _huella_de_entradas(
    node_id: str, nodo: dict[str, Any], salidas: dict[str, dict[str, Any]]
) -> str | None:
    """La huella de lo que entra, hecha con los digests YA declarados.

    Un nodo sin dependencias no tiene entrada que describir, y devolver
    el digest del vacío diría que sí la tuvo.
    """
    dependencias = sorted(str(d) for d in (nodo.get("depends_on") or []))
    if not dependencias:
        return None
    partes = [f"{d}={(salidas.get(d) or {}).get('digest')}" for d in dependencias]
    return digest_de("|".join(partes))


def _politica_del_nodo(nodo: dict[str, Any], pipeline: dict[str, Any]) -> Any:
    """La del nodo manda sobre la del pipeline; si no hay ninguna, no hay."""
    if nodo.get("policy") is not None:
        return nodo["policy"]
    return pipeline.get("policy")


def ejecutar_pipeline(
    pipeline: Any,
    *,
    registry: dict[str, str],
    ejecutores: dict[str, Callable[..., Any]] | None = None,
    contexto: dict[str, Any] | None = None,
    dry_run: bool = False,
    cancelacion: Cancelacion | None = None,
    reloj: Callable[[], float] = time.monotonic,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline entero y devuelve informe y traza.

    `ejecutores` es un diccionario `kind → función`. La función recibe
    `entradas` (lo que produjeron sus dependencias), `nodo` (su
    declaración) y `contexto` —donde viaja `remaining_s`, para quien
    sepa respetarlo—, y devuelve lo que produce. Lo que devuelva se
    digesta y se transporta; **`None` es un hueco**, no un valor, y el
    consumidor lo rechazará.
    """
    verificacion = verificar_antes_de_ejecutar(pipeline, registry)
    if not verificacion["ok"]:
        raise PipelineInvalido(verificacion["problems"])

    pipeline = dict(pipeline)
    ejecutores = dict(ejecutores or {})
    contexto_base = dict(contexto or {})
    orden: list[str] = list(verificacion["order"])
    por_id = {str(n["id"]): dict(n) for n in pipeline["nodes"]}

    # AUDITORÍA EXTERNA (2026-08-20) [ALTO]: la raíz era el `pipeline_id`,
    # así que **dos ejecuciones del mismo pipeline compartían raíz** y sus
    # trazas no se podían distinguir. §13.2 pide «generar un identificador
    # raíz de EJECUCIÓN», y el del pipeline identifica el grafo, no la vez
    # que se corrió.
    #
    # `run_id` se puede inyectar —lo necesitan las pruebas y quien quiera
    # atar la traza a un identificador suyo—; si no viene, se genera uno.
    raiz = str(run_id or f"{pipeline.get('pipeline_id')}#{uuid.uuid4().hex[:16]}")
    traza = TrazaDeEjecucion(raiz, dry_run=dry_run)
    limite_s = float(pipeline["timeout_s"])
    empezado = reloj()

    salidas: dict[str, dict[str, Any]] = {}
    estado = "ok"
    motivo: str | None = None
    nodo_culpable: str | None = None
    arrancados: list[str] = []

    for node_id in orden:
        nodo = por_id[node_id]

        # ENTRE nodos, no dentro: lo que este runtime puede parar es lo
        # que todavía no ha arrancado, y decirlo así es la diferencia
        # entre un límite y una promesa.
        if cancelacion is not None and cancelacion.activa:
            estado, motivo = "cancelled", cancelacion.motivo
            break
        transcurrido = reloj() - empezado
        if transcurrido >= limite_s:
            estado = "timed_out"
            motivo = (f"el pipeline lleva {transcurrido:.3f}s y su límite son "
                      f"{limite_s:.3f}s: {node_id!r} no arranca")
            nodo_culpable = node_id
            break

        politica = _politica_del_nodo(nodo, pipeline)
        resultado_politica: dict[str, Any] | None = None
        if politica is not None:
            contexto_politica = dict(contexto_base)
            contexto_politica.update(nodo.get("policy_context") or {})
            resultado_politica = evaluar_politica(politica, contexto_politica)

        traza.abrir_nodo(
            node_id,
            modelo=str(nodo.get("model") or ""),
            # El digest que el nodo DECLARÓ y que su ejecutor comprueba antes de
            # correr. Sin él, el paso de un modelo ajeno se leería como «un
            # fichero que se llamaba ajeno.onnx».
            model_digest=str(nodo.get("entry_hash") or "") or None,
            input_digest=None,
            efectos=bool(nodo.get("effects")),
            idempotente=bool(nodo.get("idempotent")),
        )
        arrancados.append(node_id)
        if resultado_politica is not None:
            traza.anotar_politica(node_id, resultado_politica)
            if resultado_politica["decision"] != "allow":
                # `abstain` tampoco autoriza: si se declaró una política y
                # no dijo que sí, nadie ha dicho que sí.
                estado = "denied"
                motivo = resultado_politica["explain"]
                nodo_culpable = node_id
                traza.cerrar_nodo(node_id, ok=False, error=motivo)
                break

        if dry_run:
            # NO se llama al ejecutor. Ni con un ejecutor de mentira: el
            # criterio del contrato es que dry-run no ejecute modelos, y
            # llamarlo «para ver si está» ya lo ejecutaría.
            #
            # Y tampoco se transporta: nadie ha producido nada, así que
            # no hay entrada que montar. Lo que una simulación NO ha
            # comprobado se dice en el informe (`not_checked`), porque un
            # «dry-run OK» a secas se lee como que el cableado de datos
            # también está bien.
            traza.cerrar_nodo(node_id, ok=True)
            continue

        try:
            entradas = entradas_para(
                node_id,
                [str(d) for d in (nodo.get("depends_on") or [])],
                salidas,
                comprobar_digests=True,
            )
        except Exception as exc:  # EntradaQueFalta y SalidaAlterada
            estado, motivo, nodo_culpable = "failed", str(exc), node_id
            traza.cerrar_nodo(node_id, ok=False, error=motivo)
            break

        # La huella de la entrada se compone de los digests que declararon
        # los productores, NO de reserializar los valores: la traza no
        # toca el dato (§13.3), y volver a serializarlo aquí daría una
        # huella distinta de la que el productor firmó.
        traza.anotar_entrada(node_id, _huella_de_entradas(node_id, nodo, salidas))

        kind = str(nodo.get("kind") or "")
        ejecutor = ejecutores.get(kind)
        if ejecutor is None:
            estado = "failed"
            motivo = (f"el nodo {node_id!r} es de tipo {kind!r} y no hay ejecutor "
                      f"registrado para él (hay: {sorted(ejecutores) or 'ninguno'}). "
                      "No se ejecuta con otro ni se salta: devolvería algo "
                      "plausible sin haber corrido nada")
            nodo_culpable = node_id
            traza.cerrar_nodo(node_id, ok=False, error=motivo)
            break

        contexto_nodo = dict(contexto_base)
        contexto_nodo["remaining_s"] = max(0.0, limite_s - (reloj() - empezado))
        contexto_nodo["traza"] = traza
        contexto_nodo["node_id"] = node_id

        intentos_extra = int(nodo.get("retries") or 0)
        producido: Any = None
        error_ultimo: str | None = None
        while True:
            try:
                producido = ejecutor(entradas=entradas, nodo=nodo, contexto=contexto_nodo)
                error_ultimo = None
                break
            except Exception as exc:
                error_ultimo = f"{type(exc).__name__}: {exc}"
                if intentos_extra <= 0:
                    break
                try:
                    traza.reintentar(node_id)
                except ReintentoNoSeguro as negativa:
                    # Se NIEGA el reintento y se dice por qué: callarlo
                    # dejaría un nodo que parece no reintentable por
                    # descuido en vez de por seguridad.
                    error_ultimo = f"{error_ultimo} · reintento no realizado: {negativa}"
                    break
                intentos_extra -= 1

        if error_ultimo is not None:
            estado, motivo, nodo_culpable = "failed", error_ultimo, node_id
            traza.cerrar_nodo(node_id, ok=False, error=error_ultimo)
            break

        if producido is None:
            # Haber corrido no es haber producido. Se corta AQUÍ, con el
            # nodo que no produjo, en vez de dos nodos más allá con un
            # «falta la salida de X» que ya no dice quién falló.
            estado = "failed"
            motivo = (f"el nodo {node_id!r} terminó sin producir nada: una salida "
                      "vacía no se transporta, porque el siguiente correría con "
                      "un hueco")
            nodo_culpable = node_id
            traza.cerrar_nodo(node_id, ok=False, error=motivo)
            break

        huella = digest_de(producido)
        salidas[node_id] = {"value": producido, "digest": huella}
        traza.cerrar_nodo(node_id, ok=True, output_digest=huella)

    # Lo que NUNCA arrancó, dicho: sin esto, una traza cortada a la mitad
    # se lee igual que una ejecución entera.
    sin_arrancar = [n for n in orden if n not in arrancados]

    informe: dict[str, Any] = {
        "status": estado,
        "reason": motivo,
        "node": nodo_culpable,
        "order": orden,
        "executed": list(arrancados),
        "not_started": sin_arrancar,
        "dry_run": bool(dry_run),
        "elapsed_s": round(reloj() - empezado, 6),
        "timeout_s": limite_s,
        # El alcance del límite, escrito: comprobarlo entre nodos no es lo
        # mismo que poder cortar uno por dentro, y quien lea el informe
        # tiene que saber cuál de las dos cosas tiene.
        "timeout_scope": "between-nodes; a running node is not interrupted",
        # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: el contrato dice
        # «impedir que un nodo no autorizado ejecute un efecto externo» y
        # este motor **no lo impide**: llama a una función de Python, y esa
        # función puede escribir un fichero sin pedir permiso a nadie.
        # Medido: un nodo con `effects: false` creó un fichero y terminó
        # con `status: ok`.
        #
        # Lo que sí hace, y es lo que se declara: la traza **se niega a
        # registrar** un efecto no declarado, y en `dry-run` no llama al
        # ejecutor ni una vez. Impedirlo de verdad exige aislamiento — el
        # sandbox del C5—, y prometerlo aquí sería exactamente la mitad
        # tranquilizadora que este contrato persigue. Misma línea que
        # §13.2 bis con los reintentos: el runtime no puede garantizarlo
        # por su cuenta, así que **no lo finge**.
        "effects_enforcement": (
            "declared-only; this runtime records and refuses to log undeclared "
            "effects, but it does NOT prevent an executor from touching the "
            "world. Real enforcement needs the C5 sandbox"),
        "outputs": {n: s["digest"] for n, s in salidas.items()},
    }
    if dry_run:
        # Declarar lo que PASÓ, no lo que se pidió: una simulación
        # comprueba el grafo, la resolución por digest y las políticas, y
        # NO comprueba nada de lo que solo existe al ejecutar.
        informe["checked"] = ["graph", "digest_resolution", "policies"]
        informe["not_checked"] = ["node_execution", "output_transport", "output_digests"]
    return {"report": informe, "trace": traza.a_dict(), "traza": traza,
            "values": salidas}


def emitir_recibo(
    resultado: dict[str, Any],
    *,
    pipeline: dict[str, Any],
    receipt_id: str,
    purpose: str = "",
    actor: str = "",
) -> dict[str, Any]:
    """Construye el recibo §14.2 a partir de lo que la traza VIO.

    Dos negativas, y las dos por lo mismo —un recibo afirma, y afirmar
    de más es el fallo caro—:

    * **una simulación no produce recibo**: no ejecutó nada, y el papel
      se leería como que sí;
    * **una traza incompleta tampoco**: una ejecución cortada no puede
      pasar por una que terminó bien.
    """
    traza = resultado.get("trace") or {}
    informe = resultado.get("report") or {}
    if traza.get("dry_run"):
        raise ReciboImposibleDeEmitir(
            "un dry-run no emite recibo: no ejecutó nada, y el recibo se "
            "leería como prueba de que algo pasó")
    if not traza.get("complete"):
        raise ReciboImposibleDeEmitir(
            "la traza tiene nodos sin cerrar: una ejecución cortada no puede "
            "pasar por una que terminó bien")

    pasos = []
    for nodo in traza.get("nodes") or []:
        pasos.append({
            "id": nodo["node_id"],
            "model": nodo["model"],
            "model_digest": nodo.get("model_digest"),
            "status": nodo["status"],
            "started_at": nodo["started_at"],
            "ended_at": nodo["ended_at"],
            "input_digest": nodo["input_digest"],
            "output_digest": nodo["output_digest"],
            "attempts": nodo["attempts"],
            "error": nodo["error"],
        })

    politicas = [p for nodo in (traza.get("nodes") or []) for p in nodo["policies"]]

    # El camino ejecutado, nodo a nodo y en orden: qué modelo se usó en
    # cada uno y qué decidió su política. Es lo que permite reconstruir la
    # ejecución sin tener la traza delante.
    modelos_declarados = {str(n.get("id")): n for n in (pipeline.get("nodes") or [])
                          if isinstance(n, dict)}
    camino = [
        {
            "node_id": n["node_id"],
            "node_digest": n.get("input_digest"),
            # `model_ref` ata el modelo A ESTE nodo, que es lo que
            # `models` por sí sola no puede decir.
            "model_ref": {
                "model": n["model"],
                "digest": (modelos_declarados.get(n["node_id"]) or {}).get("entry_hash"),
            },
            "output_digest": n.get("output_digest"),
            "status": n["status"],
            # Qué se decidió aquí y por qué. Sin regla, `null`: no se
            # inventa una decisión donde no hubo política.
            "decision": ([p.get("explain") for p in n["policies"]] or [None])[0],
        }
        for n in (traza.get("nodes") or [])
    ]

    # AUDITORÍA 2ª pasada (2026-08-20) [CRÍTICO]: el recibo emitido aquí
    # se dejaba fuera **cuatro de las doce secciones** que el §14.2
    # declara — `created_at`, `models`, `input` y `attestation`—, y la que
    # más duele es `models`: §16.6 pide que «el recibo identifique todas
    # las versiones de modelos» y no las identificaba ninguna.
    #
    # El aviso estaba a la vista y no lo vi: el caso del C6 **las añadía a
    # mano** después de llamar aquí. Un llamante que completa lo que el
    # emisor debía dar es la señal de que el emisor da de menos — y el
    # siguiente llamante no lo compensará.
    modelos = [
        {"model_id": str(n.get("model") or ""),
         "version": str(n.get("version") or "declared-in-pipeline"),
         "digest": n.get("entry_hash")}
        for n in (pipeline.get("nodes") or []) if isinstance(n, dict)
    ]
    # Las entradas del pipeline son los nodos SIN dependencias: por ahí
    # entra el dato. Lo que viaja es su huella, nunca el valor — es la
    # misma regla de §13.3 que gobierna la traza.
    entradas = [n["node_id"] for n in (traza.get("nodes") or [])
                if not n.get("input_digest")]

    return {
        "schema_version": "1.0",
        "receipt_id": receipt_id,
        "event_type": "pipeline_execution",
        # Un recibo sin fecha no se puede ordenar ni caducar. La de la
        # traza, no una nueva: la del recibo sería la de emitirlo, que
        # puede ser mucho después de lo que describe.
        "created_at": (traza.get("nodes") or [{}])[0].get("started_at"),
        "subject": {"purpose": purpose, "actor": actor,
                    "environment": traza.get("runtime")},
        # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: faltaban
        # `pipeline_digest` y `executed_path_digest`, que P23-R-0015
        # declara **DEBE**. Sin el segundo, dos ejecuciones que recorren
        # ramas distintas del mismo grafo producen recibos idénticos y un
        # tercero no puede reconstruir qué pasó — que es exactamente el
        # motivo por el que §14.2 bis existe.
        "pipeline": {
            "pipeline_id": pipeline.get("pipeline_id"),
            "pipeline_version": pipeline.get("version"),
            # El grafo COMPLETO, tal como se declaró.
            "pipeline_digest": digest_de(jcs_bytes(pipeline).decode("utf-8")),
            # Y el subgrafo que DE VERDAD se recorrió, con el modelo de
            # cada nodo atado a él: sin esa atadura la lista de modelos no
            # dice cuál corrió dónde.
            "executed_path_digest": digest_de(jcs_bytes(camino).decode("utf-8")),
            "nodes": camino,
        },
        "models": modelos,
        "input": {
            "entry_nodes": entradas,
            # El motor NO ve el dato del caso: solo lo que produjo cada
            # nodo. Inventar aquí un `payload_digest` sería firmar una
            # huella de algo que no se ha visto; quien tenga el dato lo
            # añade (el caso del 81-C6 lo hace).
            "payload_digest": None,
            "raw_data_included": False,
        },
        # `attestation` NO va en el payload a propósito: el firmante, la
        # clave y la firma viven en el SOBRE DSSE (`firmar_recibo`), y
        # repetirlos aquí serían dos sitios declarando lo mismo — con el
        # agravante de que el de dentro va firmado y podría contradecir al
        # de fuera.
        "steps": pasos,
        "checks": {"policy_results": politicas},
        "output": {"outcome": informe.get("status"),
                   "output_digests": informe.get("outputs") or {},
                   # Lo que no arrancó viaja DENTRO del recibo: si se
                   # quedara solo en el informe, el recibo de un pipeline
                   # cortado no se distinguiría del de uno entero.
                   "not_started": informe.get("not_started") or []},
        "evidence": {"trace_root": traza.get("root_id"),
                     "trace_complete": traza.get("complete"),
                     "runtime": traza.get("runtime")},
    }
