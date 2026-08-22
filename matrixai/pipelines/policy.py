"""Políticas como esquema JSON versionado — contrato 81-C1.

Decisión de Roberto (2026-08-20): **no se inventa un lenguaje todavía**.
Un DSL propio sería el tercero de la casa después de `.mxai` y
`.mxtrain`, y este esquema es mucho más barato. Si se demuestra
insuficiente, entonces se abre un contrato para el DSL — y no antes.

Las cinco reglas que el contrato fija, y por qué cada una:

* **Determinista**: sin red, sin reloj, sin aleatoriedad, sin efectos
  externos. Una política que consulta algo de fuera da veredictos
  distintos para el mismo pipeline, y entonces el recibo no prueba nada.
* **Precedencia explícita**: manda la PRIMERA regla que casa. Sin un
  orden declarado, dos reglas que se contradicen dan un resultado que
  depende de cómo esté escrito el diccionario.
* **Fallo cerrado**: lo que no se entiende NIEGA. Saltarse una regla rota
  deja pasar justo lo que esa regla existía para parar.
* **`abstain` es un resultado**, no un hueco: «esta política no opina» es
  distinto de permitir y de negar, y sin él una política que no aplica
  tendría que mentir en una dirección.
* **El resultado NOMBRA la regla**: «passed» no es evidencia auditable;
  «`require_data_cutoff@1.2.0 → allow (R3)`» sí.

Los operandos son EXPLÍCITOS: `{"var": "cutoff"}` lee del contexto y
cualquier otra cosa es un literal. Sin esa distinción, `{"eq": ["cutoff",
"2026-01-01"]}` buscaría un dato llamado `"2026-01-01"` y la regla no
dispararía nunca — y una política que parece bien escrita y no dispara
jamás es peor que una que falla, porque nadie va a mirarla.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PoliticaInvalida", "evaluar_politica", "ESQUEMAS_SOPORTADOS"]

#: Versiones del esquema que este evaluador sabe leer. Una futura NO se
#: interpreta a medias: se corta, como hace la captura del contrato 82.
ESQUEMAS_SOPORTADOS = ("1.0",)

_DECISIONES = ("allow", "deny", "abstain")

#: Centinela para «este dato no viene». No se usa `None` porque `null` ES
#: un valor JSON legítimo: confundirlos haría que comparar contra un hueco
#: diera `True`.
_AUSENTE = object()


class PoliticaInvalida(ValueError):
    """La política no se puede evaluar tal como llega.

    Es un error y no un `deny` silencioso: una política mal escrita es un
    fallo de quien la escribió, y tragárselo la convertiría en una que
    niega todo sin que nadie sepa por qué.
    """


def _dato(contexto: dict[str, Any], operando: Any) -> Any:
    """Resuelve un operando: `{"var": "x"}` es el contexto, lo demás literal.

    EXPLÍCITO a propósito. La primera versión trataba toda cadena como
    nombre del contexto, así que `{"eq": ["cutoff", "2026-01-01"]}`
    buscaba una clave llamada `"2026-01-01"` y la regla no casaba nunca —
    una política que parece escrita bien y no dispara jamás es peor que
    una que falla, porque nadie va a mirarla.
    """
    if isinstance(operando, dict) and set(operando) == {"var"}:
        nombre = operando["var"]
        if not isinstance(nombre, str):
            raise PoliticaInvalida(f"var debe nombrar un dato, no {nombre!r}")
        return contexto.get(nombre, _AUSENTE)
    return operando


def _predicado(cuando: Any, contexto: dict[str, Any]) -> bool:
    """¿Casa esta condición? Levanta si no la entiende — fallo cerrado."""
    if not isinstance(cuando, dict) or len(cuando) != 1:
        raise PoliticaInvalida(f"condición no reconocida: {cuando!r}")
    (operador, argumentos), = cuando.items()

    if operador in ("all", "any"):
        if not isinstance(argumentos, list):
            raise PoliticaInvalida(f"{operador} necesita una lista")
        resultados = [_predicado(x, contexto) for x in argumentos]
        return all(resultados) if operador == "all" else any(resultados)
    if operador == "not":
        return not _predicado(argumentos, contexto)

    if not isinstance(argumentos, list) or len(argumentos) != 2:
        raise PoliticaInvalida(f"{operador} necesita exactamente dos operandos")
    izq, der = _dato(contexto, argumentos[0]), _dato(contexto, argumentos[1])

    # Un dato que NO VIENE no participa en ninguna comparación: rellenarlo
    # con `None` o con cero haría que una regla opinara sobre algo que no
    # ha visto.
    if izq is _AUSENTE or der is _AUSENTE:
        return False

    if operador == "eq":
        return izq == der
    if operador == "ne":
        return izq != der
    if operador in ("gt", "ge", "lt", "le"):
        if not isinstance(izq, (int, float)) or not isinstance(der, (int, float)):
            return False
        return {"gt": izq > der, "ge": izq >= der,
                "lt": izq < der, "le": izq <= der}[operador]
    if operador == "in":
        return isinstance(der, (list, tuple)) and izq in der
    raise PoliticaInvalida(f"operador desconocido: {operador!r}")


def evaluar_politica(politica: Any, contexto: dict[str, Any]) -> dict[str, Any]:
    """Evalúa la política contra el contexto y explica QUÉ decidió."""
    if not isinstance(politica, dict):
        raise PoliticaInvalida("una política es un objeto")
    version_esquema = politica.get("schema_version")
    if version_esquema not in ESQUEMAS_SOPORTADOS:
        raise PoliticaInvalida(
            f"schema_version {version_esquema!r} no soportada; este evaluador lee "
            f"{list(ESQUEMAS_SOPORTADOS)}. Una versión que no se conoce no se "
            "interpreta a medias.")

    nombre = str(politica.get("name") or "?")
    version = str(politica.get("version") or "?")
    por_defecto = politica.get("default", "deny")
    if por_defecto not in _DECISIONES:
        raise PoliticaInvalida(f"default debe ser uno de {list(_DECISIONES)}")

    reglas = politica.get("rules")
    if not isinstance(reglas, list):
        raise PoliticaInvalida("rules debe ser una lista, aunque esté vacía")

    # LA PRIMERA que casa manda: precedencia explícita, no la «más
    # específica» ni la última leída.
    for regla in reglas:
        if not isinstance(regla, dict):
            raise PoliticaInvalida(f"regla no reconocida: {regla!r}")
        rid = str(regla.get("id") or "?")
        decision = regla.get("then")
        if decision not in _DECISIONES:
            raise PoliticaInvalida(
                f"la regla {rid} decide {decision!r}, y solo valen {list(_DECISIONES)}")
        try:
            casa = _predicado(regla.get("when"), contexto)
        except PoliticaInvalida as exc:
            # FALLO CERRADO, y con el motivo dentro: la regla que no se
            # entiende NIEGA, en vez de saltarse y dejar pasar justo lo
            # que existía para parar.
            return {
                "decision": "deny", "rule_id": rid,
                "explain": f"{nombre}@{version} → deny ({rid}: no se pudo evaluar — {exc})",
                "policy": nombre, "policy_version": version,
            }
        if casa:
            return {
                "decision": decision, "rule_id": rid,
                "explain": f"{nombre}@{version} → {decision} ({rid})",
                "policy": nombre, "policy_version": version,
            }

    return {
        "decision": por_defecto, "rule_id": None,
        "explain": f"{nombre}@{version} → {por_defecto} (default: ninguna regla casó)",
        "policy": nombre, "policy_version": version,
    }
