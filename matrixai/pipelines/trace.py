"""La traza unificada de una ejecución — contrato 81-C3.

Lo que el runtime registra es lo que después sostiene el recibo: una
traza incompleta produce un recibo que afirma más de lo que vio.

Dos reglas que no son de registro sino de seguridad, y por eso viven
aquí y no en la pantalla:

* **Un nodo no autorizado no ejecuta un efecto externo.** Y en `dry-run`
  no lo ejecuta ninguno: simular con efectos reales no es simular.
* **Un reintento no repite un efecto no idempotente.** El contrato
  (§13.2 bis) reconoce que el runtime *no puede garantizarlo
  unilateralmente*, así que no lo finge: se niega, y quien sepa que su
  operación es segura lo declara. **Por defecto NO es idempotente** —dar
  por seguro lo que nadie ha declarado duplicaría cobros, correos y
  escrituras.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

__all__ = ["EfectoNoAutorizado", "ReintentoNoSeguro", "TrazaDeEjecucion"]


class EfectoNoAutorizado(RuntimeError):
    """Un nodo intentó tocar el mundo sin permiso para hacerlo."""


class ReintentoNoSeguro(RuntimeError):
    """Reintentar esto podría duplicar un efecto que no se puede deshacer."""


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class TrazaDeEjecucion:
    """Registro de una ejecución, con su raíz y sus nodos."""

    def __init__(self, root_id: str, *, runtime: str = "matrixai", dry_run: bool = False) -> None:
        if not str(root_id or "").strip():
            raise ValueError("una traza sin raíz no relaciona nada")
        self.root_id = root_id.strip()
        self.runtime = runtime
        self.dry_run = bool(dry_run)
        self._nodos: dict[str, dict[str, Any]] = {}
        self._orden: list[str] = []

    def abrir_nodo(self, node_id: str, *, modelo: str, input_digest: str | None = None,
                   efectos: bool = False, idempotente: bool = False,
                   model_digest: str | None = None) -> None:
        if node_id in self._nodos:
            raise ValueError(
                f"el nodo {node_id!r} ya está en esta traza: dos nodos con el "
                "mismo id harían ambigua cualquier afirmación sobre él")
        self._nodos[node_id] = {
            "root_id": self.root_id,      # cada nodo atado a su raíz
            "node_id": node_id,
            "model": modelo,
            # QUÉ MODELO ERA, no solo cómo se llamaba (87-C1). Para una entrada
            # del registry el nombre+versión se resuelve a un digest; para un
            # FICHERO ajeno el nombre no dice nada — mañana ese `ajeno.onnx`
            # puede ser otro. Sin esto, el recibo de un modelo de fuera
            # describiría «un fichero que se llamaba así».
            "model_digest": model_digest,
            "runtime": self.runtime,
            "started_at": _ahora(),
            "ended_at": None,
            "status": "open",             # y así se queda si nadie lo cierra
            "error": None,
            "input_digest": input_digest,
            "output_digest": None,
            "policies": [],
            "effects": [],
            "may_have_effects": bool(efectos),
            "idempotent": bool(idempotente),
            "attempts": 1,
        }
        self._orden.append(node_id)

    def _nodo(self, node_id: str) -> dict[str, Any]:
        nodo = self._nodos.get(node_id)
        if nodo is None:
            raise ValueError(f"el nodo {node_id!r} no está en esta traza")
        return nodo

    def anotar_entrada(self, node_id: str, input_digest: str | None) -> None:
        """La huella de lo que el nodo recibió, sabida DESPUÉS de abrirlo.

        Se sabe al montar las entradas, que es después de abrir el nodo:
        un nodo cuyas entradas no se pueden montar tiene que aparecer en
        la traza igual, o el fallo no quedaría atado a nadie.
        """
        if input_digest is not None:
            self._nodo(node_id)["input_digest"] = input_digest

    def anotar_politica(self, node_id: str, resultado: dict[str, Any]) -> None:
        """La política evaluada, CON la regla que decidió.

        Se guarda el resultado entero del evaluador: «passed» no es
        evidencia auditable, y el `explain` que trae dentro sí.
        """
        self._nodo(node_id)["policies"].append(dict(resultado))

    def registrar_efecto(self, node_id: str, descripcion: str) -> None:
        nodo = self._nodo(node_id)
        if self.dry_run:
            raise EfectoNoAutorizado(
                f"el nodo {node_id!r} intentó un efecto externo en dry-run: "
                "simular con efectos reales no es simular")
        if not nodo["may_have_effects"]:
            raise EfectoNoAutorizado(
                f"el nodo {node_id!r} no está autorizado a ejecutar efectos "
                "externos, y ha intentado uno")
        nodo["effects"].append(descripcion)

    def cerrar_nodo(self, node_id: str, *, ok: bool, error: str | None = None,
                    output_digest: str | None = None) -> None:
        nodo = self._nodo(node_id)
        nodo["ended_at"] = _ahora()
        nodo["status"] = "ok" if ok else "failed"
        nodo["error"] = None if ok else (error or "sin motivo declarado")
        if output_digest:
            nodo["output_digest"] = output_digest

    def reintentar(self, node_id: str) -> None:
        """Vuelve a abrir un nodo fallido, si es seguro repetirlo.

        §13.2 bis: el runtime no puede garantizar por su cuenta que un
        reintento no duplique un efecto no idempotente, así que **no lo
        intenta**. Quien sepa que su operación es repetible lo declara.
        """
        nodo = self._nodo(node_id)
        if nodo["may_have_effects"] and not nodo["idempotent"]:
            raise ReintentoNoSeguro(
                f"el nodo {node_id!r} puede tener efectos externos y no se ha "
                "declarado idempotente: reintentarlo podría duplicar algo que "
                "no se puede deshacer")
        if not nodo["idempotent"]:
            raise ReintentoNoSeguro(
                f"el nodo {node_id!r} no se ha declarado idempotente; por "
                "defecto no se reintenta, porque dar por seguro lo que nadie "
                "ha declarado duplicaría cobros, correos y escrituras")
        nodo["attempts"] += 1
        nodo["status"] = "open"
        nodo["ended_at"] = None
        nodo["error"] = None

    def a_dict(self) -> dict[str, Any]:
        nodos = [self._nodos[n] for n in self._orden]
        return {
            "root_id": self.root_id,
            "runtime": self.runtime,
            "dry_run": self.dry_run,
            "nodes": nodos,
            # Una ejecución con nodos abiertos NO está completa: darla por
            # terminada haría que el recibo dijera que fue bien.
            "complete": all(n["status"] in ("ok", "failed") for n in nodos),
        }
