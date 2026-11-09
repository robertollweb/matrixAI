# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C7 — LAS MONOTONÍAS DECLARADAS AL PLANTEAR, DENTRO DEL RECIBO FIRMADO: qué entra, qué no, y cómo se compone.

POR QUÉ AQUÍ. El recibo `study_execution` se firma al terminar el estudio y lo calculado DESPUÉS no puede entrar en un recibo ya
firmado (D2 del contrato 149): por eso las monotonías que la persona declaró antes de ver resultados se comprueban DENTRO del estudio y
su resultado viaja en el cuerpo, bajo la sección raíz opcional `comportamiento`. Quien sabe qué es un agregado y qué es una fila es el
núcleo, que fabricó esos documentos; por eso la composición vive aquí y el Studio solo calcula y entrega (la misma partición que el
expediente, `banco_en_el_paquete`).

LA SECCIÓN SE COMPONE DESDE UNA LISTA BLANCA, escrita clave a clave (`_monotonia` de `banco_en_el_paquete`: la MISMA poda que ya viaja en
el paquete exportado, no una segunda copia): ni una fila de desarrollo, ni un identificador de fila, ni un valor por fila. Solo
agregados (tasas, medianas, un retroceso máximo), los TRAMOS de la rejilla donde hubo retroceso y la curva de la fila TÍPICA de train
(medianas y modas, sin datos de nadie: el contraejemplo sintético de D5). `exigir_sin_filas` es la segunda cerradura: se niega a devolver
una sección con una clave de fila real.

EL NIVEL NO SE TOCA. `alcance_del_recibo` / `nivel_del_recibo` leen `evidence`, no esta sección: tener `comportamiento` no sube el nivel
(A1 sigue siendo A1) ni lo baja. Esta sección dice qué se comprobó, no cuánto se puede confiar en el recibo.

SIN DECLARACIONES, NADA. Si nadie declaró una dirección, `seccion_del_recibo` devuelve `None` y el cuerpo del recibo es BYTE A BYTE el de
antes de este corte (I12): la clave está AUSENTE, no vale `null`.

LO QUE NO SE PUDO COMPROBAR SE ESCRIBE: una columna declarada que no es numérica, una que superó el tope de tiempo o un cálculo que
falló figuran con su estado y su motivo es/en (un valor ausente no es un cero ni un «sin contraejemplos»).

STDLIB PURO (más los módulos hermanos del 149).
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from matrixai.estudio.banco_en_el_paquete import _bilingue, _monotonia, _num, exigir_sin_filas
from matrixai.estudio.comportamiento_textos import motivo

__all__ = ["ESTADOS", "NOMBRE_DE_LA_SECCION", "VERSION_DE_LA_SECCION", "seccion_del_recibo", "motivo_del_tope", "motivo_de_un_fallo"]

#: El nombre de la sección raíz del cuerpo del recibo.
NOMBRE_DE_LA_SECCION = "comportamiento"
VERSION_DE_LA_SECCION = 1

#: Lo que puede pasar con cada columna declarada. Cerrada: una cuarta no entra sin pasar por aquí.
ESTADOS = ("comprobada", "no_aplica", "no_se_pudo_comprobar")


def motivo_del_tope(segundos: float) -> dict[str, str]:
    """El motivo bilingüe de una columna que no llegó a comprobarse porque el cálculo agotó su tope de tiempo."""
    return motivo("tope_de_tiempo_del_recibo", segundos=f"{segundos:g}")


def motivo_de_un_fallo() -> dict[str, str]:
    """El motivo bilingüe de una columna que no se comprobó porque el cálculo falló (sin jerga ni trazas: va a un recibo)."""
    return motivo("fallo_en_el_recibo")


def _declaracion(d: Mapping[str, Any]) -> dict[str, Any]:
    """La declaración sin su `id` (interno del Studio) y con la marca antes/después que puso el SERVIDOR."""
    return {"columna": str(d.get("columna")), "direccion": str(d.get("direccion")),
            "momento": str(d.get("momento")), "declarada_en": str(d.get("declarada_en"))}


def _tolerancia(resultado: Mapping[str, Any] | None, comprobadas: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """La tolerancia (ε, q) con que se evaluó: la de la primera columna comprobada (todas llevan la misma) o, si ninguna se comprobó, la de
    partida de la tarea. `None` si no hay ninguna: un valor ausente no es un cero."""
    if comprobadas:
        t = comprobadas[0].get("tolerancia") or {}
        return {"relativa": t.get("relativa"), "fraccion_minima": t.get("fraccion_minima"), "por_omision": t.get("por_omision")}
    partida = (resultado or {}).get("tolerancia_de_partida")
    if isinstance(partida, Mapping):
        return {"relativa": _num(partida.get("epsilon_relativo")), "fraccion_minima": _num(partida.get("q_relativo")),
                "por_omision": True}
    return None


def seccion_del_recibo(*, declaraciones: Sequence[Mapping[str, Any]], resultado: Mapping[str, Any] | None,
                       sin_comprobar: Mapping[str, Mapping[str, str]] | None = None, protocolo: Mapping[str, Any],
                       motivo_general: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """La sección `comportamiento` del cuerpo del recibo, o `None` si nadie declaró nada (entonces el cuerpo es el de siempre).

    `declaraciones`: las VIGENTES, `{columna, direccion, momento, declarada_en}`, en el orden en que se hicieron (la marca la puso el
    servidor). `resultado`: el documento del banco (`comportamiento.json`: B y C) o `None` si no hubo cálculo; solo se lee su parte de
    monotonía, no_aplica, muestra, parámetros, escala y versión. `sin_comprobar`: `{columna: motivo}` de las que el cálculo no alcanzó
    (tope de tiempo, fallo). `protocolo`: lo que el Studio fijó para el cálculo, que se conoce aunque no haya resultado. `motivo_general`:
    por qué ninguna columna se comprobó, si es el caso (sin muestra, sin banco…).

    Levanta `matrixai.estudio.banco_en_el_paquete.BancoConFilas` si lo compuesto llevase una clave de fila real."""
    if not declaraciones:
        return None
    resultado = resultado if isinstance(resultado, Mapping) else None
    sin_comprobar = sin_comprobar or {}
    monotonias = {(m.get("declaracion") or {}).get("columna"): m for m in (resultado or {}).get("monotonia") or [] if isinstance(m, Mapping)}
    no_aplica = {n.get("columna"): n for n in (resultado or {}).get("no_aplica") or []
                 if isinstance(n, Mapping) and n.get("prueba") == "monotonia"}

    columnas: list[dict[str, Any]] = []
    comprobadas: list[Mapping[str, Any]] = []
    for d in declaraciones:
        c = str(d.get("columna"))
        if c in monotonias:
            doc = _monotonia(monotonias[c], relativa=True)
            veredicto = doc.get("veredicto") or {}
            if veredicto.get("clave") == "no_se_pudo_comprobar":
                # Remate del 149-C7 (menor a): una entrada «comprobada» con veredicto «no se pudo comprobar» se contradice (binaria,
                # columna que oscila): el estado dice lo mismo que el veredicto, con el motivo del banco o el genérico.
                # (se conserva lo medido —filas, curva, tolerancia—: es lo que explica por qué no se pudo)
                columnas.append({"estado": "no_se_pudo_comprobar", **doc,
                                 "motivo": (_bilingue(veredicto.get("motivo")) or _bilingue(veredicto.get("texto"))
                                            or motivo_de_un_fallo())})
                continue
            columnas.append({"estado": "comprobada", **doc})
            comprobadas.append(doc)
        elif c in no_aplica:
            columnas.append({"estado": "no_aplica", "declaracion": _declaracion(d),
                             "tipo": str(no_aplica[c].get("tipo")), "motivo": _bilingue(no_aplica[c].get("motivo"))})
        else:
            columnas.append({"estado": "no_se_pudo_comprobar", "declaracion": _declaracion(d),
                             "motivo": _bilingue(sin_comprobar.get(c)) or _bilingue(motivo_general) or motivo_de_un_fallo()})

    muestra = (resultado or {}).get("muestra") if isinstance((resultado or {}).get("muestra"), Mapping) else {}
    parametros = (resultado or {}).get("parametros") if isinstance((resultado or {}).get("parametros"), Mapping) else {}
    escala = (resultado or {}).get("escala") if isinstance((resultado or {}).get("escala"), Mapping) else {}
    seccion = {
        "version": VERSION_DE_LA_SECCION,
        "protocolo": {
            # Lo que el Studio FIJÓ para este cálculo (se conoce aunque no haya resultado), y lo que el cálculo midió de verdad.
            "medido_sobre": "muestra_de_desarrollo",
            "semilla": _num(protocolo.get("semilla")),
            "filas_por_columna": _num(protocolo.get("filas_por_columna")),
            "puntos_de_la_rejilla": _num(protocolo.get("puntos_de_la_rejilla")),
            "tope_de_tiempo_segundos": _num(protocolo.get("tope_de_tiempo_segundos")),
            "filas_de_la_muestra": _num(muestra.get("filas")),
            "filas_de_desarrollo": _num(muestra.get("filas_de_desarrollo")),
            "tarea": (resultado or {}).get("tarea") if isinstance((resultado or {}).get("tarea"), str) else None,
            "motor": (resultado or {}).get("motor") if isinstance((resultado or {}).get("motor"), str) else None,
            "version_del_banco": (resultado or {}).get("version_del_banco") if isinstance((resultado or {}).get("version_del_banco"), str) else None,
            "tolerancia": _tolerancia(resultado, comprobadas),
            "escala": {"tarea": escala.get("tarea"), "valor": _num(escala.get("valor")),
                       "clave_de_la_escala": escala.get("clave_de_la_escala") if isinstance(escala.get("clave_de_la_escala"), str) else None,
                       "descripcion": _bilingue(escala.get("descripcion"))} if escala else None,
            "parametros_de_los_documentos": {"filas_de_c": _num(parametros.get("filas_de_c"))} if parametros else None,
        },
        "columnas": columnas,
        # La frase fija de límites (I4), también a nivel de sección: el recibo se lee sin abrir ninguna columna.
        "limites": _limites(comprobadas, protocolo, muestra),
    }
    exigir_sin_filas(seccion)
    return copy.deepcopy(seccion)


def _limites(comprobadas: Sequence[Mapping[str, Any]], protocolo: Mapping[str, Any], muestra: Mapping[str, Any]) -> dict[str, str]:
    """La frase de límites tal como la escribió el banco (la de una columna comprobada); sin ninguna, la compuesta con el protocolo."""
    for c in comprobadas:
        propia = _bilingue((c.get("veredicto") or {}).get("limites"))
        if propia is not None:
            return propia
    return motivo("limites_del_banco", puntos=int(protocolo.get("puntos_de_la_rejilla") or 0),
                  filas=int(min(protocolo.get("filas_por_columna") or 0, muestra.get("filas") or protocolo.get("filas_por_columna") or 0)),
                  semilla=int(protocolo.get("semilla") or 0))
