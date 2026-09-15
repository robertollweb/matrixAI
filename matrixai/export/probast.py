# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""109-C3 — los huecos PROBAST+AI de un paquete: qué sostiene y qué no.

QUÉ ES PROBAST+AI. PROBAST es la herramienta con la que un revisor juzga el
riesgo de sesgo y la aplicabilidad de un estudio de modelo de predicción, y
PROBAST+AI es su versión para modelos con aprendizaje automático, con listas
distintas para DESARROLLO y para EVALUACIÓN de un modelo.

QUÉ HACE ESTO, Y SOBRE TODO QUÉ NO HACE. Rellena lo que el core ya midió, marca
lo que declaró una persona y **enumera lo que falta, con el campo del expediente
que lo sostendría**. No puntúa. No cierra ningún dominio. **No califica el
riesgo de sesgo, ni en un sentido ni en el otro** (invariante 6 del contrato
109): esa palabra la pone quien revisa el estudio, leyéndolo, y un programa que
la imprimiese estaría firmando por él. Que una casilla salga MEDIDA no califica
nada — dice que el dato existe y dónde está.

**SOBRE EL TEXTO DE LAS PREGUNTAS SEÑAL, Y ES LO MÁS IMPORTANTE DE ESTE
MÓDULO.** Este informe **NO reproduce el enunciado de las preguntas señal de
PROBAST+AI**, y no es un descuido: el enunciado oficial no se ha podido
verificar contra la fuente en este entorno, y escribirlo de memoria sería una
cita fabricada — el mismo defecto que el 109 ya retiró de su propia fila de C2
y que `tripod.py` evita con la numeración de TRIPOD+AI. Aquí se hace lo mismo
que allí: **se va por DOMINIO**, se dice por qué, y se deja la puerta abierta a
que el enunciado entre por donde tiene que entrar.

Y esa puerta es `probast_instrument.json`: si el equipo —que sí tiene el
instrumento— lo deja en el paquete, este informe responde **pregunta a
pregunta**, con la evidencia que el paquete sostiene en el dominio de cada una.
Sin él, enumera por dominio, que es lo que se puede hacer sin inventar.

EL CONTRATO 109 DECLARA 16 preguntas señal para desarrollo y 18 para
evaluación. Ese número se usa aquí **como comprobación, no como afirmación**:
si llega un instrumento, se compara con lo que traiga y se dice si no cuadra.
Sin instrumento, se cita de dónde sale el número y que no se ha comprobado.

C3 ESTÁ MARCADO «SOLO SI EL PILOTO LO NECESITA», Y NO HAY PILOTO. Roberto abrió
el 109 el 2026-09-14 sin disparador, a sabiendas: el invariante 7 sigue diciendo
que la vertical la elige la tarea del socio y este corte se está construyendo
sabiendo que lo contradice. Va escrito aquí y en la propia ficha, como en el
107.

STDLIB PURO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matrixai.export.expediente_clinico import (
    DECLARADO,
    DOMINIOS,
    ESTADOS,
    FALTA,
    MEDIDO,
    Campo,
    Evidencia,
    ExpedienteClinico,
    ExpedienteNoDisponible,
    linea_de_campo,
    rotulo_de_dominio,
    rotulo_de_estado,
    sin_veredicto_de_riesgo,
)

#: Se re-exporta a propósito: quien llame aquí tiene que poder distinguir «este
#: paquete no existe» de «este paquete no trae nada», y es el mismo error que
#: levanta la ficha TRIPOD+AI.
__all__ = [
    "CUENTAS_SEGUN_EL_CONTRATO",
    "INSTRUMENTO",
    "ExpedienteNoDisponible",
    "MODOS",
    "PreguntaSenal",
    "huecos_probast",
    "mapa_probast",
]

#: El fichero, opcional, con el instrumento oficial que el equipo sí tiene.
INSTRUMENTO = "probast_instrument.json"

#: Las dos listas de PROBAST+AI. Tokens en inglés, como el resto de los
#: vocabularios que viajan en JSON en este paquete.
MODOS = ("development", "evaluation")

#: Lo que declara el contrato 109. **No comprobado contra el instrumento
#: oficial en este entorno**: se usa para contrastar el que llegue, y la ficha
#: dice las dos cosas.
CUENTAS_SEGUN_EL_CONTRATO: dict[str, int] = {"development": 16, "evaluation": 18}


@dataclass(frozen=True)
class PreguntaSenal:
    """Una pregunta del instrumento y lo que el paquete sostiene para ella.

    `estado` es uno de los tres de siempre y no hay un cuarto: `medido`,
    `declarado`, `falta`. No hay puntuación ni veredicto.
    """

    pregunta_id: str
    dominio: str
    texto: str | None
    estado: str
    apoyos: tuple[Campo, ...]

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS:
            raise ValueError(f"estado desconocido: {self.estado!r}")

    def a_json(self) -> dict[str, Any]:
        return {"pregunta_id": self.pregunta_id, "dominio": self.dominio,
                "texto": self.texto, "estado": self.estado,
                "apoyos": [c.a_json() for c in self.apoyos]}


_T: dict[str, dict[str, str]] = {
    "es": {
        "titulo": "Huecos PROBAST+AI",
        "cita": "PROBAST+AI (BMJ, 2025) es la herramienta con la que un revisor "
                "juzga el riesgo de sesgo y la aplicabilidad de un modelo de "
                "predicción. Esto es una **guía de reporte, no una certificación**.",
        "no_juzga": "**Este informe no puntúa ni cierra ningún dominio, y no "
                    "califica el riesgo de sesgo.** Rellena lo que el core midió, "
                    "marca lo que declaró una persona y enumera lo que falta con el "
                    "campo que lo sostendría. El juicio lo emite quien revisa el "
                    "estudio; que una casilla salga MEDIDA solo dice que el dato "
                    "existe y dónde está.",
        "sin_enunciados": "**Los enunciados oficiales de las preguntas señal no "
                          "viajan en este informe.** No se han podido verificar "
                          "contra la fuente en este entorno, y escribirlos de "
                          "memoria sería una cita fabricada. Este informe va por "
                          "DOMINIO. Si el equipo deja el instrumento oficial en "
                          "`probast_instrument.json`, se responde pregunta a "
                          "pregunta.",
        "cuentas": "El contrato 109 declara 16 preguntas señal para desarrollo y 18 "
                   "para evaluación; **no está comprobado aquí** contra el "
                   "instrumento oficial.",
        "cuentas_ok": "El instrumento trae {n} preguntas para «{modo}», que es lo "
                      "que declara el contrato 109.",
        "cuentas_no": "**El instrumento trae {n} preguntas para «{modo}» y el "
                      "contrato 109 declara {esperadas}.** Manda el instrumento; se "
                      "escribe la diferencia en vez de callarla.",
        "sin_piloto": "**C3 está marcado «solo si el piloto lo necesita», y no hay "
                      "piloto.** El 109 se abrió el 2026-09-14 sin disparador, a "
                      "sabiendas: el invariante 7 dice que la vertical la elige la "
                      "tarea del socio, y esto se construyó sabiendo que lo "
                      "contradice.",
        "dominio": "Dominio", "estado": "Estado", "dato": "Dato", "campo": "Campo",
        "por_dominio": "Lo que este paquete sostiene, por dominio",
        "preguntas": "Las preguntas del instrumento",
        "resumen": "Resumen",
        "aviso_dominios": "La asignación de cada línea a su dominio es de este "
                          "informe, no del instrumento: sirve para encontrar junto "
                          "lo que hace falta y no sustituye a leerlo.",
        "modo": "Lista",
        "modo_development": "desarrollo de un modelo",
        "modo_evaluation": "evaluación de un modelo",
        "sin_instrumento": "el paquete no lo trae",
        "instrumento_roto": "está en el paquete, pero no se puede leer",
        "no_declara": "el instrumento no declara",
        "avisos": "Avisos sobre el propio paquete",
    },
    "en": {
        "titulo": "PROBAST+AI gaps",
        "cita": "PROBAST+AI (BMJ, 2025) is the tool a reviewer uses to judge risk of "
                "bias and applicability of a prediction model. This is a **reporting "
                "aid, not a certification**.",
        "no_juzga": "**This report does not score, does not close any domain, and "
                    "does not rate risk of bias.** It fills in what the core "
                    "measured, marks what a person declared, and lists what is "
                    "missing along with the field that would support it. The "
                    "judgement is made by whoever reviews the study; a MEASURED box "
                    "only says the datum exists and where it is.",
        "sin_enunciados": "**The official wording of the signalling questions is not "
                          "in this report.** It could not be verified against the "
                          "source in this environment, and writing it from memory "
                          "would be a fabricated citation. This report goes by "
                          "DOMAIN. If the team drops the official instrument into "
                          "`probast_instrument.json`, each question is answered one "
                          "by one.",
        "cuentas": "Contract 109 states 16 signalling questions for development and "
                   "18 for evaluation; **that is not verified here** against the "
                   "official instrument.",
        "cuentas_ok": "The instrument ships {n} questions for «{modo}», which is what "
                      "contract 109 states.",
        "cuentas_no": "**The instrument ships {n} questions for «{modo}» and contract "
                      "109 states {esperadas}.** The instrument wins; the difference "
                      "is written rather than swallowed.",
        "sin_piloto": "**C3 is marked «only if the pilot needs it», and there is no "
                      "pilot.** Contract 109 was opened on 2026-09-14 with no "
                      "trigger, knowingly: invariant 7 says the vertical is chosen by "
                      "the partner's task, and this was built knowing it contradicts "
                      "that.",
        "dominio": "Domain", "estado": "State", "dato": "Value", "campo": "Field",
        "por_dominio": "What this package supports, by domain",
        "preguntas": "The instrument's questions",
        "resumen": "Summary",
        "aviso_dominios": "Assigning each line to a domain is this report's doing, "
                          "not the instrument's: it puts together what is needed and "
                          "does not replace reading it.",
        "modo": "List",
        "modo_development": "model development",
        "modo_evaluation": "model evaluation",
        "sin_instrumento": "the package does not ship it",
        "instrumento_roto": "it is in the package, but it cannot be read",
        "no_declara": "the instrument does not declare",
        "avisos": "Warnings about the package itself",
    },
}


def _t(locale: str) -> dict[str, str]:
    return _T["es"] if str(locale or "en").strip().lower() == "es" else _T["en"]


def _idioma(locale: str) -> str:
    return "es" if str(locale or "en").strip().lower() == "es" else "en"


def _leer_instrumento(bundle: Path) -> tuple[dict[str, Any] | None, str]:
    """El instrumento del equipo, si viaja en el paquete.

    Devuelve `(instrumento, motivo)`. Un fichero ilegible **no es un fichero
    ausente**: se dice que está y que está roto, porque decir «no lo traes»
    sobre algo que sí trajiste es media verdad tranquilizadora.
    """
    ruta = bundle / INSTRUMENTO
    if not ruta.is_file():
        return None, "ausente"
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"roto: {exc}"
    if not isinstance(datos, dict):
        return None, "roto: el instrumento no es un objeto JSON"
    return datos, ""


def _estado_del_dominio(apoyos: tuple[Campo, ...]) -> str:
    """El estado de una pregunta a partir de lo que sostiene su dominio.

    `MEDIDO` solo si el core midió algo; `DECLARADO` si lo único que hay lo
    escribió una persona; `FALTA` si no hay nada. Nunca un cuarto valor, y sobre
    todo nunca un juicio: esto dice qué clase de evidencia hay, no si basta.
    """
    estados = {c.estado for c in apoyos}
    if MEDIDO in estados:
        return MEDIDO
    if DECLARADO in estados:
        return DECLARADO
    return FALTA


def _analizar(bundle_dir: str | Path, idioma: str) -> dict[str, Any]:
    """Una sola lectura del paquete, para el mapa y para la ficha.

    Que las dos salidas salgan de aquí es lo que impide que el JSON diga una
    cosa y el Markdown otra — el fallo que este repositorio ya se ha comido
    catorce veces por el lado del cableado.
    """
    expediente = ExpedienteClinico.desde_paquete(bundle_dir)
    pares, avisos = expediente.resolver(idioma)

    por_dominio: dict[str, list[tuple[Evidencia, Campo]]] = {d: [] for d in DOMINIOS}
    for evidencia, campo in pares:
        por_dominio[evidencia.dominio].append((evidencia, campo))

    instrumento, motivo = _leer_instrumento(expediente.bundle)
    preguntas: list[PreguntaSenal] = []
    modo: str | None = None
    desajuste: str | None = None
    if instrumento is not None:
        modo = instrumento.get("modo") or instrumento.get("mode")
        crudas = instrumento.get("preguntas") or instrumento.get("questions") or []
        esperadas = CUENTAS_SEGUN_EL_CONTRATO.get(str(modo))
        if esperadas is not None and len(crudas) != esperadas:
            desajuste = f"{len(crudas)} != {esperadas}"
        for i, cruda in enumerate(crudas):
            if not isinstance(cruda, dict):
                continue
            dominio = str(cruda.get("dominio") or cruda.get("domain") or "")
            conocido = dominio in DOMINIOS
            apoyos = tuple(c for _e, c in por_dominio.get(dominio, [])) if conocido else ()
            preguntas.append(PreguntaSenal(
                pregunta_id=str(cruda.get("id") or f"#{i + 1}"),
                dominio=dominio if conocido else "",
                texto=cruda.get("texto") or cruda.get("text"),
                estado=_estado_del_dominio(apoyos) if conocido else FALTA,
                apoyos=apoyos))
    return {"expediente": expediente, "por_dominio": por_dominio,
            "avisos": avisos, "instrumento": instrumento, "motivo": motivo,
            "modo": modo, "desajuste": desajuste, "preguntas": preguntas}


def mapa_probast(bundle_dir: str | Path, *, locale: str = "en") -> dict[str, Any]:
    """La forma legible por máquina: cada línea con su estado y su campo.

    `juicio_de_riesgo` va explícito y va a `None` **siempre**. No es un campo
    que algún día se rellene: está para que quien consuma este JSON buscando un
    veredicto lo encuentre vacío y lea por qué, en vez de deducir que se olvidó.
    """
    idioma = _idioma(locale)
    a = _analizar(bundle_dir, idioma)
    return {
        "schema": "matrixai.export.probast",
        "paquete": str(a["expediente"].bundle),
        "instrumento": {
            "presente": a["instrumento"] is not None,
            "motivo": a["motivo"], "modo": a["modo"],
            "desajuste_de_cuenta": a["desajuste"],
            "cuentas_segun_el_contrato": dict(CUENTAS_SEGUN_EL_CONTRATO),
            "cuentas_comprobadas_contra_el_instrumento_oficial": False,
        },
        "por_dominio": {
            dominio: [{"clave": e.clave, "rotulo": e.rotulo[idioma], **c.a_json()}
                      for e, c in entradas]
            for dominio, entradas in a["por_dominio"].items()},
        "preguntas": [p.a_json() for p in a["preguntas"]],
        "avisos": list(a["avisos"]),
        "juicio_de_riesgo": None,
    }


def huecos_probast(bundle_dir: str | Path, *, locale: str = "en") -> str:
    """La ficha en Markdown: por dominio, y pregunta a pregunta si hay instrumento."""
    t = _t(locale)
    idioma = _idioma(locale)
    a = _analizar(bundle_dir, idioma)

    filas: list[str] = [f"# {t['titulo']}", ""]
    filas.append(f"> {t['cita']}")
    filas.append(">")
    filas.append(f"> {t['no_juzga']}")
    filas.append(">")
    filas.append(f"> {t['sin_piloto']}")
    filas.append("")

    if a["instrumento"] is None:
        filas.append(t["sin_enunciados"])
        filas.append("")
        detalle = (t["instrumento_roto"] if str(a["motivo"]).startswith("roto")
                   else t["sin_instrumento"])
        filas.append(f"- **{t['modo']}**: _{rotulo_de_estado(FALTA, idioma)}_ — "
                     f"{detalle} · `{INSTRUMENTO}#modo`")
        filas.append(f"- {t['cuentas']}")
        filas.append("")
    else:
        modo = str(a["modo"])
        n = len(a["preguntas"])
        esperadas = CUENTAS_SEGUN_EL_CONTRATO.get(modo)
        rotulo_modo = t.get(f"modo_{modo}", modo)
        filas.append(f"- **{t['modo']}**: {rotulo_modo} · `{INSTRUMENTO}#modo`")
        if esperadas is None:
            filas.append(f"- {t['cuentas']} · `{INSTRUMENTO}#modo`")
        elif a["desajuste"]:
            filas.append("- " + t["cuentas_no"].format(
                n=n, modo=rotulo_modo, esperadas=esperadas)
                + f" · `{INSTRUMENTO}#preguntas`")
        else:
            filas.append("- " + t["cuentas_ok"].format(n=n, modo=rotulo_modo)
                         + f" · `{INSTRUMENTO}#preguntas`")
        filas.append("")
        filas.append(f"## {t['preguntas']}")
        filas.append("")
        filas.append(f"| # | {t['dominio']} | {t['estado']} | {t['campo']} |")
        filas.append("|---|---|---|---|")
        for pregunta in a["preguntas"]:
            campos = ", ".join(f"`{c.ruta}`" for c in pregunta.apoyos) or "—"
            filas.append(f"| {pregunta.pregunta_id} | {pregunta.dominio or '—'} | "
                         f"{rotulo_de_estado(pregunta.estado, idioma)} | {campos} |")
        filas.append("")

    filas.append(f"## {t['por_dominio']}")
    filas.append("")
    filas.append(f"_{t['aviso_dominios']}_")
    filas.append("")
    cuenta = {MEDIDO: 0, DECLARADO: 0, FALTA: 0}
    for dominio in DOMINIOS:
        entradas = a["por_dominio"].get(dominio) or []
        if not entradas:
            continue
        filas.append(f"### {rotulo_de_dominio(dominio, idioma)}")
        filas.append("")
        for evidencia, campo in entradas:
            cuenta[campo.estado] += 1
            filas.append(linea_de_campo(evidencia, campo, idioma))
        filas.append("")

    if a["avisos"]:
        filas.append(f"## {t['avisos']}")
        filas.append("")
        for aviso in a["avisos"]:
            filas.append(f"- {aviso}")
        filas.append("")

    filas.append(f"## {t['resumen']}")
    filas.append("")
    for estado in (MEDIDO, DECLARADO, FALTA):
        filas.append(f"- {rotulo_de_estado(estado, idioma)}: {cuenta[estado]}")
    filas.append("")
    return sin_veredicto_de_riesgo("\n".join(filas), origen="probast.huecos_probast")
