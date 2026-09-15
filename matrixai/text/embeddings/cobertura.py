# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — qué idiomas cubre un proveedor, según lo MEDIDO.

Invariante 7 del 107: «Idioma cubierto declarado por proveedor; un texto en
otro idioma es una limitación del diagnóstico (103-C2, "evidencia
insuficiente"), no un silencio».

Aquí vive la mitad de C1 de esa frase: el dato y el veredicto. La otra mitad
—que el diagnóstico lo enseñe— es 103-C2 y C3, y no se hace aquí.

Lo importante de este módulo es lo que NO hace: no lee la ficha del autor. Un
proveedor puede decir que cubre 101 idiomas; lo que se responde aquí sale de
haberlo medido sobre un corpus real, y si no se ha medido, la respuesta es
«no medido», que es una respuesta y no un hueco.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTEFACTO = Path(__file__).resolve().parent / "catalogo_medido.json"

#: Umbrales sobre la AUC de coherencia por documento (`medicion.
#: coherencia_por_documento`): ¿el modelo pone más cerca dos frases del mismo
#: artículo que dos de artículos distintos, DENTRO de ese idioma?
#:
#: Elegidos ANTES de mirar el español y con el inglés nativo de referencia:
#: los modelos entrenados en inglés dan ahí 0,90-0,98 sobre su propio idioma,
#: y el azar es 0,50. El corte es 0,90 para «cubierto» —lo que un modelo
#: consigue en el idioma para el que lo hicieron— y 0,70 para separar «le
#: funciona a medias» de «no le funciona». No están puestos para que a
#: ningún proveedor le salga un veredicto concreto; si alguien los mueve,
#: que sea con un motivo escrito, no para aprobar a alguien.
UMBRAL_CUBIERTO = 0.90
UMBRAL_LIMITADO = 0.70

CUBIERTO = "cubierto"
LIMITADO = "limitado"
NO_CUBIERTO = "no_cubierto"
NO_MEDIDO = "no_medido"


@dataclass(frozen=True)
class Veredicto:
    """Qué se puede decir de este proveedor en este idioma, y con qué número.

    `frase` está redactada para que quien la lea sepa qué hacer, no para
    tranquilizarle: media verdad tranquilizadora es peor que callarse.
    """

    proveedor: str
    idioma: str
    estado: str
    auc: float | None
    tokens_por_palabra: float | None
    frase: str

    def es_utilizable(self) -> bool:
        return self.estado == CUBIERTO

    def to_dict(self) -> dict[str, Any]:
        return {
            "proveedor": self.proveedor,
            "idioma": self.idioma,
            "estado": self.estado,
            "auc_coherencia_por_documento": self.auc,
            "tokens_por_palabra": self.tokens_por_palabra,
            "frase": self.frase,
        }


def estado_por_auc(auc: float) -> str:
    if auc >= UMBRAL_CUBIERTO:
        return CUBIERTO
    if auc >= UMBRAL_LIMITADO:
        return LIMITADO
    return NO_CUBIERTO


def _cargar(ruta: Path | None = None) -> dict:
    r = ruta or ARTEFACTO
    if not r.is_file():
        return {"proveedores": {}}
    return json.loads(r.read_text(encoding="utf-8"))


def veredicto(proveedor_id: str, idioma: str, *, artefacto: Path | None = None) -> Veredicto:
    datos = _cargar(artefacto)
    fila = (datos.get("proveedores") or {}).get(proveedor_id)
    if fila is None:
        return Veredicto(
            proveedor_id,
            idioma,
            NO_MEDIDO,
            None,
            None,
            f"{proveedor_id} no está en el catálogo medido: no se puede decir nada de {idioma}.",
        )
    m = fila.get("medicion") or {}
    coherencia = (m.get("coherencia_por_documento") or {}).get(idioma)
    if coherencia is None:
        motivo = m.get("no_medido_porque") or fila.get("candidato", {}).get("no_descargado_porque") or ""
        return Veredicto(
            proveedor_id,
            idioma,
            NO_MEDIDO,
            None,
            None,
            f"{proveedor_id} no se ha medido en {idioma}"
            + (f": {motivo}" if motivo else ".")
            + " Lo que diga su ficha no es una medida.",
        )
    auc = float(coherencia["auc"])
    tpp = ((m.get("tokenizacion") or {}).get(idioma) or {}).get("tokens_por_palabra")
    estado = estado_por_auc(auc)
    frases = {
        CUBIERTO: (
            f"{proveedor_id} distingue temas en {idioma} igual de bien que en el idioma para el "
            f"que se entrenó (AUC {auc:.3f} sobre 0,5 de azar)."
        ),
        LIMITADO: (
            f"{proveedor_id} en {idioma} funciona a medias: AUC {auc:.3f}, por debajo del "
            f"{UMBRAL_CUBIERTO:.2f} que alcanza en su propio idioma. Sus vectores llevan señal, "
            f"pero menos; lo que se decida con ellos en {idioma} vale menos que en inglés."
        ),
        NO_CUBIERTO: (
            f"{proveedor_id} NO cubre {idioma}: AUC {auc:.3f}, cerca del 0,5 del azar. Dará un "
            f"vector para cualquier texto en {idioma}, y ese vector no significa gran cosa."
        ),
    }
    return Veredicto(proveedor_id, idioma, estado, auc, tpp, frases[estado])


def exigir_cubierto(proveedor_id: str, idioma: str, *, artefacto: Path | None = None) -> Veredicto:
    """Para quien no quiera seguir adelante con un idioma que no se cubre.

    Existe para que 103-C2/C3 tengan de dónde tirar: el silencio no es opción,
    o se declara la limitación o se para.
    """
    from matrixai.text.embeddings.proveedor import IdiomaNoCubierto

    v = veredicto(proveedor_id, idioma, artefacto=artefacto)
    if not v.es_utilizable():
        raise IdiomaNoCubierto(v.frase)
    return v


def tabla(*, artefacto: Path | None = None) -> list[dict[str, Any]]:
    """El catálogo entero en filas, para enseñarlo o escribirlo."""
    datos = _cargar(artefacto)
    filas = []
    for cid, fila in (datos.get("proveedores") or {}).items():
        m = fila.get("medicion") or {}
        filas.append(
            {
                "id": cid,
                "familia": fila.get("familia"),
                "licencia_spdx": (fila.get("licencia") or {}).get("spdx"),
                "pesos_bytes": fila.get("pesos_bytes"),
                "dimension": m.get("dimension"),
                "segundos_por_1000_es": ((m.get("tiempo") or {}).get("es") or {}).get("segundos_por_1000_textos"),
                "segundos_por_1000_en": ((m.get("tiempo") or {}).get("en") or {}).get("segundos_por_1000_textos"),
                "es": veredicto(cid, "es", artefacto=artefacto).estado,
                "en": veredicto(cid, "en", artefacto=artefacto).estado,
                "descargado": bool(fila.get("descargado")),
            }
        )
    return filas
