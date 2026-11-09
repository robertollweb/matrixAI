# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""149-C6 — EL BANCO DE COMPORTAMIENTO EN EL PAQUETE EXPORTADO: qué viaja, qué no, y cómo se escribe en el expediente.

POR QUÉ AQUÍ. Los resultados del banco (`comportamiento.json`, `comportamiento_falta.json`) los guarda el Studio junto al estudio y
llevan filas de desarrollo: los identificadores de las filas medidas (`filas.ids`, `contraejemplos[].fila`, `peores[].fila`), los
valores reales de una fila antes y después de moverla (`mayores_pendientes`) y un valor por fila (`retrocesos`, `efectos`). Nada de
eso puede salir de la máquina (D5 del contrato 149). Quien sabe qué es una fila y qué es un agregado es el núcleo, que fabricó esos
documentos; por eso la poda vive aquí y el Studio solo llama.

LA REGLA ES UNA LISTA BLANCA. `banco_para_el_paquete` COMPONE el documento desde cero con claves escritas una a una; no copia
diccionarios ajenos ni quita lo prohibido de una copia (una clave nueva que mañana añada el banco no viajaría sola, ni viajaría una
fila por descuido). Y `exigir_sin_filas` es la segunda cerradura: recorre un documento ya compuesto y se niega si encuentra una clave de
fila real, para quien escriba el fichero (`construir_paquete`).

LO QUE VIAJA. Por columna: la sensibilidad agregada (tasas y percentiles de pendiente, SIN las filas de mayor pendiente), la
monotonía declarada con su veredicto, su lectura y sus límites pegados (I4), los TRAMOS de rejilla donde hubo retroceso (qué dos puntos
de la rejilla y cuánto, sin la fila ni sus valores) y la curva de la FILA TÍPICA (medianas y modas de train, sin datos de nadie: el
contraejemplo sintético de D5), y «si falta una columna» con la caída media por columna. Lo que NO se pudo calcular se escribe con su
motivo en `no_calculado`.

SIN BANCO, NADA. Si no hay ningún resultado del banco, `banco_para_el_paquete` devuelve `None` y el paquete no lleva el fichero: el
expediente es byte a byte el de antes de este corte.

STDLIB PURO. Las monotonías declaradas NO entran en el recibo firmado (eso es el C7); esto es solo el expediente.
"""
from __future__ import annotations

from typing import Any, Mapping

from matrixai.estudio.comportamiento_textos import IDIOMAS

__all__ = [
    "BancoConFilas", "NOMBRE_DEL_FICHERO", "VERSION_DEL_BANCO_EN_EL_PAQUETE", "banco_para_el_paquete", "exigir_sin_filas",
    "partes_sin_calcular", "texto_del_banco",
]

#: El fichero dentro del paquete. Nombre fijo.
NOMBRE_DEL_FICHERO = "banco_de_comportamiento.json"
VERSION_DEL_BANCO_EN_EL_PAQUETE = 1

#: Las claves que, en cualquier sitio del documento, delatan una fila de desarrollo real (o un valor por fila). Cerrada y escrita.
CLAVES_DE_FILA = frozenset({
    "ids", "id", "fila", "filas_ids", "mayores_pendientes", "valor_antes", "valor_despues", "salida_antes", "salida_despues",
    "retrocesos", "efectos", "amplitudes", "peores", "para_recalcular", "muestra", "row_id", "unit_id", "y_true",
})


class BancoConFilas(ValueError):
    """El documento que iba a viajar en el paquete lleva una clave de fila real. El paquete no se escribe."""


def exigir_sin_filas(documento: Any, *, ruta: str = "$") -> None:
    """Recorre `documento` y levanta `BancoConFilas` si alguna clave, a cualquier profundidad, es de `CLAVES_DE_FILA`."""
    if isinstance(documento, Mapping):
        for clave, valor in documento.items():
            if clave in CLAVES_DE_FILA:
                raise BancoConFilas(f"{ruta}.{clave}: una clave de fila de desarrollo no viaja en el paquete (D5)")
            exigir_sin_filas(valor, ruta=f"{ruta}.{clave}")
    elif isinstance(documento, (list, tuple)):
        for i, valor in enumerate(documento):
            exigir_sin_filas(valor, ruta=f"{ruta}[{i}]")


def _num(valor: Any) -> Any:
    return valor if isinstance(valor, (int, float)) and not isinstance(valor, bool) else None


def _bilingue(valor: Any) -> dict[str, str] | None:
    if isinstance(valor, Mapping) and all(isinstance(valor.get(i), str) for i in IDIOMAS):
        return {i: valor[i] for i in IDIOMAS}
    return None


def _sensibilidad(doc: Mapping[str, Any]) -> dict[str, Any]:
    pruebas = []
    for p in doc.get("pruebas") or []:
        pendiente = p.get("pendiente") if isinstance(p.get("pendiente"), Mapping) else None
        pruebas.append({
            "fraccion": _num(p.get("fraccion")), "signo": p.get("signo") if isinstance(p.get("signo"), str) else None,
            "delta": _num(p.get("delta")), "filas_comprobadas": _num(p.get("filas_comprobadas")),
            "cambian_de_decision": _num(p.get("cambian_de_decision")),
            "fraccion_que_cambia_de_decision": _num(p.get("fraccion_que_cambia_de_decision")),
            "pendiente": None if pendiente is None else {k: _num(pendiente.get(k)) for k in ("mediana", "p90", "maximo")},
        })
    rango = doc.get("rango_visto") if isinstance(doc.get("rango_visto"), Mapping) else {}
    return {"columna": str(doc.get("columna")),
            "rango_visto": {"minimo": _num(rango.get("minimo")), "maximo": _num(rango.get("maximo"))},
            "filas_pedidas": _num((doc.get("filas") or {}).get("pedidas")), "pruebas": pruebas}


def _extremos_de_la_rejilla(documento: Mapping[str, Any], curva: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """(mínimo, máximo) de la rejilla donde se comprobó, o `None` si no se sabe. Se usa SOLO para pasar posiciones a relativas."""
    rejilla = documento.get("rejilla") if isinstance(documento.get("rejilla"), Mapping) else {}
    valores = rejilla.get("valores") or (curva or {}).get("x") or []
    finitos = [float(v) for v in valores if _num(v) is not None and v == v and abs(v) != float("inf")]
    if len(finitos) < 2 or max(finitos) <= min(finitos):
        return None
    return min(finitos), max(finitos)


def _relativa(x: Any, extremos: tuple[float, float] | None) -> float | None:
    """La posición `x` como fracción 0…1 de la rejilla (`None` si no se puede decir): sin el valor real ni los extremos reales."""
    if extremos is None or _num(x) is None:
        return None
    return round((float(x) - extremos[0]) / (extremos[1] - extremos[0]), 6)


def _tramos(documento: Mapping[str, Any], extremos: tuple[float, float] | None = None, relativa: bool = False) -> list[dict[str, Any]]:
    """Dónde de la rejilla hubo retroceso y cuánto: SIN la fila (`fila`) ni las salidas de esa fila (`y_a`, `y_b`)."""
    veredicto = documento.get("veredicto") if isinstance(documento.get("veredicto"), Mapping) else {}
    vistos = documento.get("contraejemplos") or (documento.get("filas_que_retroceden") or {}).get("filas") or []
    if veredicto.get("clave") not in ("con_contraejemplos", "sin_contraejemplos"):
        return []
    if relativa:
        # Recibo FIRMADO (decisión de Roberto, 149-C7): posiciones relativas 0…1 de la rejilla, ningún valor real de x.
        return [{"posicion_a": _relativa(c.get("x_a"), extremos), "posicion_b": _relativa(c.get("x_b"), extremos),
                 "retroceso": _num(c.get("retroceso"))} for c in vistos if isinstance(c, Mapping)]
    return [{"x_a": _num(c.get("x_a")), "x_b": _num(c.get("x_b")), "retroceso": _num(c.get("retroceso"))}
            for c in vistos if isinstance(c, Mapping)]


def _monotonia(item: Mapping[str, Any], relativa: bool = False) -> dict[str, Any]:
    """`relativa=True` (el recibo firmado): la rejilla y los tramos en posiciones 0…1 de la rejilla comprobada, SIN el mínimo ni el
    máximo reales ni ningún valor real de la columna. El paquete exportado usa `relativa=False` (sigue con su `rango_visto`)."""
    doc = item.get("documento") if isinstance(item.get("documento"), Mapping) else {}
    declaracion = item.get("declaracion") if isinstance(item.get("declaracion"), Mapping) else {}
    veredicto = doc.get("veredicto") if isinstance(doc.get("veredicto"), Mapping) else {}
    filas = doc.get("filas") if isinstance(doc.get("filas"), Mapping) else {}
    tolerancia = doc.get("tolerancia") if isinstance(doc.get("tolerancia"), Mapping) else {}
    minima = tolerancia.get("fraccion_minima") if isinstance(tolerancia.get("fraccion_minima"), Mapping) else {}
    tipica = item.get("fila_tipica") if isinstance(item.get("fila_tipica"), Mapping) else None
    curva = tipica.get("curva") if tipica and isinstance(tipica.get("curva"), Mapping) else None
    extremos = _extremos_de_la_rejilla(doc, curva)
    return {
        # La declaración sin su `id` (un identificador interno del Studio) y con la marca antes/después que puso el servidor.
        "declaracion": {"columna": str(declaracion.get("columna")), "direccion": str(declaracion.get("direccion")),
                        "momento": str(declaracion.get("momento")), "declarada_en": str(declaracion.get("declarada_en"))},
        "veredicto": {"clave": veredicto.get("clave"), "texto": _bilingue(veredicto.get("texto")),
                      "lectura": _bilingue(veredicto.get("lectura")), "motivo": _bilingue(veredicto.get("motivo")),
                      "limites": _bilingue(veredicto.get("limites"))},
        "efecto_total": _num(doc.get("efecto_total")), "amplitud_tipica": _num(doc.get("amplitud_tipica")),
        "fraccion_con_violacion": _num(doc.get("fraccion_con_violacion")), "retroceso_maximo": _num(doc.get("retroceso_maximo")),
        "filas_con_violacion": _num(doc.get("filas_con_violacion")),
        "filas": {"pedidas": _num(filas.get("pedidas")), "comprobadas": _num(filas.get("comprobadas")),
                  "sin_prediccion": _num(filas.get("sin_prediccion"))},
        "tolerancia": {"relativa": _num(tolerancia.get("relativa")), "absoluta": _num(tolerancia.get("absoluta")),
                       "fraccion_minima": _num(minima.get("relativa")), "por_omision": tolerancia.get("por_omision") is True},
        "rejilla_puntos": _num((doc.get("rejilla") or {}).get("puntos")),
        "tramos_con_retroceso": _tramos(doc, extremos, relativa),
        "fila_tipica": None if curva is None else {
            "medido_sobre": "fila_tipica", "de_donde": "valor_de_referencia_de_train",
            **({"posiciones_relativas": [_relativa(v, extremos) for v in curva.get("x") or []]} if relativa
               else {"x": [_num(v) for v in curva.get("x") or []]}),
            "y": [_num(v) for v in curva.get("y") or []]},
    }


def _falta(documento: Mapping[str, Any]) -> dict[str, Any]:
    columnas = []
    for c in documento.get("columnas") or []:
        if not isinstance(c, Mapping):
            continue
        columnas.append({"columna": str(c.get("columna")), "caida_media": _num(c.get("caida_media")),
                         "desviacion": _num(c.get("desviacion")), "pliegues_con_caida": _num(c.get("pliegues_con_caida")),
                         "nunca_falto_al_entrenar": c.get("nunca_falto_al_entrenar") if isinstance(c.get("nunca_falto_al_entrenar"), bool) else None,
                         "motivo": _bilingue(c.get("motivo"))})
    return {"metrica": documento.get("metrica") if isinstance(documento.get("metrica"), str) else None,
            "medido_sobre": documento.get("medido_sobre") if isinstance(documento.get("medido_sobre"), str) else None,
            "pliegues_medidos": _num(documento.get("pliegues_medidos")), "filas_medidas": _num(documento.get("filas_medidas")),
            "calculado_en": documento.get("calculado_en") if isinstance(documento.get("calculado_en"), str) else None,
            "columnas": columnas}


_SIN_CALCULAR = {
    "monotonia": {"es": "ninguna monotonía declarada: no hay veredicto por columna",
                  "en": "no monotonicity declared: there is no per-column verdict"},
    "falta": {"es": "«si falta una columna» no se ha calculado para este modelo",
              "en": "“if a column is missing” has not been computed for this model"},
    "sensibilidad": {"es": "la sensibilidad no se ha calculado para este modelo",
                     "en": "sensitivity has not been computed for this model"},
}


def banco_para_el_paquete(*, resultado: Mapping[str, Any] | None, falta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """El documento que viaja en el paquete, o `None` si no hay ningún resultado del banco (entonces el paquete no lleva el fichero).

    `resultado`: `comportamiento.json` (B y C) ya juzgado como de ESTE modelo; `falta`: `comportamiento_falta.json`, ídem. Quien llama
    decide si un resultado es de este modelo (el digest vive en el Studio); aquí solo se poda. Determinista: sin fecha de hoy."""
    if resultado is None and falta is None:
        return None
    base = resultado if resultado is not None else falta
    modelo = base.get("modelo") if isinstance(base.get("modelo"), Mapping) else {}
    muestra = (resultado or {}).get("muestra") if isinstance((resultado or {}).get("muestra"), Mapping) else {}
    sensibilidad = [_sensibilidad(d) for d in (resultado or {}).get("sensibilidad") or [] if isinstance(d, Mapping) and d.get("disponible") is not False]
    monotonia = [_monotonia(m) for m in (resultado or {}).get("monotonia") or [] if isinstance(m, Mapping)]
    no_calculado: list[dict[str, Any]] = []
    for d in (resultado or {}).get("sensibilidad") or []:
        if isinstance(d, Mapping) and d.get("disponible") is False:
            no_calculado.append({"parte": "sensibilidad", "columna": str(d.get("columna")), "motivo": _bilingue(d.get("motivo"))})
    for n in (resultado or {}).get("no_aplica") or []:
        if isinstance(n, Mapping):
            no_calculado.append({"parte": str(n.get("prueba")), "columna": str(n.get("columna")), "motivo": _bilingue(n.get("motivo"))})
    if resultado is None:
        no_calculado.append({"parte": "sensibilidad", "columna": None, "motivo": dict(_SIN_CALCULAR["sensibilidad"])})
    if resultado is not None and not monotonia:
        no_calculado.append({"parte": "monotonia", "columna": None, "motivo": dict(_SIN_CALCULAR["monotonia"])})
    if falta is None:
        no_calculado.append({"parte": "falta", "columna": None, "motivo": dict(_SIN_CALCULAR["falta"])})
    documento = {
        "version": VERSION_DEL_BANCO_EN_EL_PAQUETE, "version_del_banco": base.get("version_del_banco"),
        # Declarado en el propio fichero y comprobable: ninguna fila de desarrollo, ni sus identificadores.
        "sin_filas_de_desarrollo": True,
        "modelo": {"digest": modelo.get("digest"), "motor": modelo.get("motor")},
        "tarea": base.get("tarea"),
        "medido_sobre": None if resultado is None else {
            "clave": resultado.get("medido_sobre"), "filas": _num(muestra.get("filas")), "semilla": _num(muestra.get("semilla"))},
        "calculado_en": {"comportamiento": None if resultado is None else resultado.get("calculado_en"),
                         "falta": None if falta is None else falta.get("calculado_en")},
        "sensibilidad": sensibilidad, "monotonia": monotonia,
        "falta": None if falta is None else _falta(falta),
        "no_calculado": no_calculado,
    }
    exigir_sin_filas(documento)
    return documento


def partes_sin_calcular(banco: Mapping[str, Any], idioma: str) -> list[str]:
    """Lo que el banco de ESTE paquete no calculó, una frase por parte y con su motivo, en `idioma`. Vacío si no falta nada."""
    idioma = "es" if idioma == "es" else "en"
    nombre = {"sensibilidad": {"es": "sensibilidad", "en": "sensitivity"},
              "monotonia": {"es": "monotonía", "en": "monotonicity"},
              "falta": {"es": "si falta una columna", "en": "if a column is missing"}}
    salida: list[str] = []
    for item in banco.get("no_calculado") or []:
        if not isinstance(item, Mapping):
            continue
        motivo = _bilingue(item.get("motivo"))
        parte = nombre.get(item.get("parte"), {}).get(idioma, str(item.get("parte")))
        columna = f" · `{item['columna']}`" if item.get("columna") else ""
        salida.append(f"{parte}{columna}: {motivo[idioma] if motivo else '—'}")
    return salida


_T = {
    "es": {
        "base": "banco de pruebas sobre {filas} filas de desarrollo (semilla {semilla}); solo viajan agregados y curvas de una fila "
                "típica, ninguna fila de desarrollo ni sus identificadores",
        "sens": "sensibilidad a cambios pequeños en {n} columnas",
        "sin_sens": "sensibilidad no calculada",
        "mono": "{n} monotonías declaradas por una persona ({antes} antes de ver resultados, {despues} después): {con} con contraejemplos, "
                "{sin} sin contraejemplos en esta rejilla, {no} no se pudo comprobar",
        "sin_mono": "ninguna monotonía declarada",
        "falta": "«si falta una columna» en {n} columnas (mayor caída media: {peor}, {cifra})",
        "falta_sin": "«si falta una columna» sin cifra en ninguna columna",
        "sin_falta": "«si falta una columna» no calculado",
        "ilegible": "no se puede leer",
    },
    "en": {
        "base": "test bench over {filas} development rows (seed {semilla}); only aggregates and curves of a typical row travel, "
                "no development row and no row identifiers",
        "sens": "sensitivity to small changes in {n} columns",
        "sin_sens": "sensitivity not computed",
        "mono": "{n} monotonicities declared by a person ({antes} before seeing results, {despues} after): {con} with counterexamples, "
                "{sin} no counterexamples on this grid, {no} could not be checked",
        "sin_mono": "no monotonicity declared",
        "falta": "“if a column is missing” on {n} columns (largest mean drop: {peor}, {cifra})",
        "falta_sin": "“if a column is missing” gave no figure for any column",
        "sin_falta": "“if a column is missing” not computed",
        "ilegible": "cannot be read",
    },
}


def _cifra(valor: float, idioma: str) -> str:
    texto = f"{valor:.3f}"
    return texto.replace(".", ",") if idioma == "es" else texto


def texto_del_banco(banco: Mapping[str, Any], idioma: str) -> str:
    """El resumen de una línea del banco que viaja en el paquete, en `idioma`, compuesto SOLO desde el documento. Termina con la frase
    fija de límites (I4) cuando hay alguna monotonía, tal como la escribió el banco; sin veredicto de riesgo y sin «aprobado»."""
    idioma = "es" if idioma == "es" else "en"
    t = _T[idioma]
    medido = banco.get("medido_sobre") if isinstance(banco.get("medido_sobre"), Mapping) else None
    piezas: list[str] = []
    if medido is not None:
        piezas.append(t["base"].format(filas=medido.get("filas"), semilla=medido.get("semilla")))
    sens = banco.get("sensibilidad") or []
    piezas.append(t["sens"].format(n=len(sens)) if sens else t["sin_sens"])
    mono = banco.get("monotonia") or []
    limites = None
    if mono:
        claves = [(m.get("veredicto") or {}).get("clave") for m in mono]
        momentos = [(m.get("declaracion") or {}).get("momento") for m in mono]
        piezas.append(t["mono"].format(
            n=len(mono), antes=momentos.count("antes"), despues=momentos.count("despues"),
            con=claves.count("con_contraejemplos"), sin=claves.count("sin_contraejemplos"), no=claves.count("no_se_pudo_comprobar")))
        limites = _bilingue((mono[0].get("veredicto") or {}).get("limites"))
    else:
        piezas.append(t["sin_mono"])
    falta = banco.get("falta") if isinstance(banco.get("falta"), Mapping) else None
    if falta is None:
        piezas.append(t["sin_falta"])
    else:
        con_cifra = [c for c in falta.get("columnas") or [] if isinstance(c, Mapping) and _num(c.get("caida_media")) is not None]
        if con_cifra:
            peor = max(con_cifra, key=lambda c: (c["caida_media"], str(c.get("columna"))))
            piezas.append(t["falta"].format(n=len(con_cifra), peor=f"`{peor['columna']}`", cifra=_cifra(peor["caida_media"], idioma)))
        else:
            piezas.append(t["falta_sin"])
    texto = "; ".join(piezas) + "."
    if limites is not None:
        texto += " " + limites[idioma]
    return texto
