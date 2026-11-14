# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""165-C3 — EL RECIBO DE «FILAS A REVISAR» (`anomaly_review`) Y LAS CIFRAS CON ETIQUETAS APORTADAS.

Esto SOLO describe datos YA calculados: el detector vive en `matrixai_engines.anomalias` (sklearn, numpy) y aquí no se importa
nada de fuera de la biblioteca estándar (lo vigila `test_c104_c0_esquemas`). Dos cosas:

1. `recibo_de_filas_a_revisar(...)`: el recibo (tipo `anomaly_review`, secciones en `pipelines.receipt._SECCIONES_POR_TIPO`, que
   es donde las exige también el verificador) con sus textos es/en: lo que NO afirma y los LÍMITES medidos por el banco del C2.
2. `cifras_con_etiquetas(...)`: AUROC, AP y precisión@k con intervalo (el bootstrap del 104-C2, `incertidumbre.intervalo`; no se
   escribe otro), y SOLO cuando la persona aporta una columna «¿era anómala?». Sin etiquetas esas cifras NO EXISTEN: ni la clave
   `metrics` aparece en el recibo, ni un cero, ni un `null` que se pinte como cero (invariante 1 del contrato 165).

Una puntuación es un ORDEN, no una probabilidad; una fila a revisar no es un error ni un fraude.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from matrixai.estudio.incertidumbre import intervalo
from matrixai.estudio.metricas import Indefinida, Muestra, calcular
from matrixai.estudio.esquemas import ValorDeMetrica
from matrixai.pipelines.canonical import jcs_bytes
from matrixai.pipelines.receipt import firmar_recibo

__all__ = ["CIFRAS_CON_ETIQUETAS", "CLAVES_DE_MOTIVO", "LIMITES_MEDIDOS", "NO_AFIRMA", "TEXTO_DE_LIMITES", "cifras_con_etiquetas",
           "digest_de_la_lista", "recibo_de_filas_a_revisar", "firmar_recibo_de_filas_a_revisar", "texto_de_motivo"]

#: Las cifras que solo existen con etiquetas. El verificador las prohíbe sin ellas.
CIFRAS_CON_ETIQUETAS = ("auroc", "average_precision", "precision_at_k")

#: LO MEDIDO por el banco del 165-C2 (regla sellada `bab0a2c7…`; `benchmarks/c165/`): de las n filas SEMBRADAS por MatrixAI, qué
#: parte pone el detector vigente en las 2n primeras de la lista, con tasas de siembra del 1 % y del 5 %. El valor extremo, de los 6
#: conjuntos del banco; la combinación imposible, de los 4 que tienen una pareja de columnas correlacionadas; la categoría rara, de UNO.
#: NO es acierto sobre las anomalías de nadie: es lo que el detector apenas ve.
#:
#: `categoria_rara` está EN REVISIÓN (auditoría 165, I4): su 7 % / 37 % salió de una siembra (`valor_nuevo_…`) que cae al FINAL del
#: orden alfabético —el extremo del eje ordinal—, y con un nombre de mitad de alfabeto la cifra baja. Hasta que el C2b
#: (`benchmarks/c165/REGLA_C2B.md`) mida con nombres neutros, NO se enseña ninguna cifra de la categoría rara (`None`), y el texto lo dice.
LIMITES_MEDIDOS = {
    "extremo": {"1 %": 0.16, "5 %": 0.55},
    "categoria_rara": None,
    "combinacion_imposible": {"1 %": 0.01, "5 %": 0.13},
}


def _pc(x: float) -> str:
    return f"{round(100 * x)} %"


TEXTO_DE_LIMITES = {
    "es": ("Medido por MatrixAI sobre anomalías SEMBRADAS por nosotros, no sobre las tuyas: si en un conjunto hay n filas "
           f"con un valor extremo, el detector pone en las 2n primeras de la lista el {_pc(LIMITES_MEDIDOS['extremo']['1 %'])} "
           f"de ellas cuando son el 1 % de las filas y el {_pc(LIMITES_MEDIDOS['extremo']['5 %'])} cuando son el 5 %; y con una "
           "combinación imposible de valores que por separado son normales, "
           f"el {_pc(LIMITES_MEDIDOS['combinacion_imposible']['1 %'])} y el {_pc(LIMITES_MEDIDOS['combinacion_imposible']['5 %'])}: "
           "eso es lo que el detector apenas ve. Lo que recupera con una categoría rara está EN REVISIÓN y no se da ninguna cifra: "
           "con columnas de categorías la lista depende de cómo se llamen (se codifican por orden alfabético, y renombrar una "
           "categoría puede cambiar la lista). No dice nada de lo que hay en tu fichero. Una lista que no destaca una fila "
           "no prueba que esa fila esté bien."),
    "en": ("Measured by MatrixAI on anomalies SEEDED by us, not on yours: if a dataset has n rows with an extreme value, the "
           f"detector puts {_pc(LIMITES_MEDIDOS['extremo']['1 %'])} of them in the first 2n of the list when they are 1 % of the "
           f"rows and {_pc(LIMITES_MEDIDOS['extremo']['5 %'])} when they are 5 %; and with an impossible combination of values that "
           f"are each normal on their own, {_pc(LIMITES_MEDIDOS['combinacion_imposible']['1 %'])} and "
           f"{_pc(LIMITES_MEDIDOS['combinacion_imposible']['5 %'])}: that is what the detector barely sees. What it recovers with a "
           "rare category is UNDER REVIEW and no figure is given: with category columns the list depends on what the categories "
           "are called (they are encoded in alphabetical order, and renaming a category can change the list). It says nothing "
           "about what is in your file. A list that does not highlight a row does not prove the row is fine."),
}

NO_AFIRMA = {
    "es": [
        "Sin etiquetas no se calcula ni se afirma exactitud, sensibilidad, AUROC ni «porcentaje de anomalías detectadas».",
        "La puntuación es un ORDEN (rango), no una probabilidad ni un veredicto: «rara» significa rara respecto a este fichero.",
        "Una fila a revisar no es un error ni un fraude: no se borra ni se corrige, se señala.",
        "El número de filas realmente anómalas se desconoce; no se estima ninguna proporción.",
        "Las columnas a la cola describen qué valores son extremos o raros, no la razón interna del detector.",
        "La estabilidad mide solo cuánto cambia la lista con la semilla, no si la lista es correcta.",
    ],
    "en": [
        "Without labels no accuracy, sensitivity, AUROC or «percentage of anomalies found» is computed or claimed.",
        "The score is an ORDER (rank), not a probability or a verdict: «unusual» means unusual with respect to this file.",
        "A row to review is not an error or a fraud: it is not deleted or corrected, it is pointed out.",
        "The number of truly anomalous rows is unknown; no proportion is estimated.",
        "The tail columns describe which values are extreme or rare, not the detector's internal reason.",
        "Stability only measures how much the list changes with the seed, not whether the list is right.",
    ],
}

_ALCANCE_DE_LAS_ETIQUETAS = {
    "es": ("Las cifras con etiquetas valen solo para las filas que la persona etiquetó, tal como las etiquetó. Si revisó "
           "sobre todo las primeras de la lista, son optimistas y no describen al resto del fichero."),
    "en": ("The figures with labels hold only for the rows the person labelled, as labelled. If they mostly reviewed the "
           "top of the list, they are optimistic and do not describe the rest of the file."),
}

_ATESTIGUA = {
    "es": "Qué datos (huella), qué columnas, qué detector y semilla, y la lista devuelta (huella), tal como se calcularon.",
    "en": "Which data (fingerprint), which columns, which detector and seed, and the returned list (fingerprint), as computed.",
}
_NO_ATESTIGUA = {
    "es": "Que las filas señaladas sean anómalas, ni que las no señaladas estén bien: ver `no_claims` y `limits`.",
    "en": "That the flagged rows are anomalous, nor that the unflagged ones are fine: see `no_claims` and `limits`.",
}
_PROPOSITO = {
    "es": "Lista ordenada de filas a revisar de un CSV sin columna objetivo",
    "en": "Ordered list of rows to review from a CSV without a target column",
}


# ---------------------------------------------------------------------------
# Los motivos de exclusión y los avisos del detector, por CLAVE (auditoría 165: salían en castellano y sin clave)
# ---------------------------------------------------------------------------
#
# Motores (`matrixai_engines.anomalias`) DECIDE y emite `{clave, parametros}`; el texto lo redacta el núcleo, en es y en (lo que
# redacta el núcleo se traduce en el núcleo), y la pantalla enseña el de su idioma. Una clave que no esté aquí es un error de
# cableado y `texto_de_motivo` falla: no se inventa un texto.

def _n(x: Any, idioma: str) -> str:
    """Un entero legible: «1.000» en castellano, «1,000» en inglés."""
    if isinstance(x, bool) or not isinstance(x, int):
        return str(x)
    t = f"{x:,}"
    return t.replace(",", ".") if idioma == "es" else t


def _cols(columnas: Sequence[str]) -> str:
    return ", ".join(f"«{c}»" for c in columnas)


def _textos(textos: Mapping[str, int], idioma: str) -> str:
    return ", ".join(f"«{t}» ({_n(int(n), idioma)})" for t, n in textos.items())


def _t_vacia(p, i):
    return "Vacía: ninguna fila tiene dato." if i == "es" else "Empty: no row has a value."


def _t_casi_vacia(p, i):
    if i == "es":
        return f"Casi vacía: {_n(p['sin_dato'], i)} de {_n(p['filas'], i)} filas sin dato."
    return f"Almost empty: {_n(p['sin_dato'], i)} of {_n(p['filas'], i)} rows have no value."


def _t_constante(p, i):
    return ("Constante: un solo valor, no distingue unas filas de otras." if i == "es"
            else "Constant: a single value, it does not tell rows apart.")


def _t_categorica_casi_unica(p, i):
    if i == "es":
        return (f"Identificador o casi única: {_n(p['distintos'], i)} valores distintos en {_n(p['presentes'], i)} filas. "
                "Dominaría el aislamiento («rara» sería «única»).")
    return (f"Identifier or nearly unique: {_n(p['distintos'], i)} distinct values in {_n(p['presentes'], i)} rows. "
            "It would dominate the isolation («unusual» would just mean «unique»).")


def _t_identificador_numerico(p, i):
    d, n = _n(p["distintos"], i), _n(p["presentes"], i)
    if p.get("por") == "nombre":
        return (f"Identificador: lo dice su nombre, y sus {n} enteros son casi todos distintos ({d})." if i == "es" else
                f"Identifier: its name says so, and its {n} integers are nearly all distinct ({d}).")
    return (f"Identificador: {d} enteros distintos en {n} filas que forman un tramo casi seguido (un contador o un número de "
            "registro)." if i == "es" else
            f"Identifier: {d} distinct integers in {n} rows forming an almost unbroken run (a counter or a record number).")


def _t_numeros_con_formato(p, i):
    if i == "es":
        return (f"{_n(p['celdas'], i)} de sus {_n(p['presentes'], i)} celdas tienen forma de número con separadores que no se "
                f"leen como número (p. ej. «{p['ejemplo']}»): no se mira ni como número ni como categoría. Si son miles, quita el "
                "separador; si es la coma decimal, guarda el CSV separado por «;».")
    return (f"{_n(p['celdas'], i)} of its {_n(p['presentes'], i)} cells look like numbers with separators that are not read as "
            f"numbers (e.g. «{p['ejemplo']}»): it is used neither as a number nor as a category. If they are thousands, remove "
            "the separator; if it is a decimal comma, save the CSV separated by «;».")


def _t_numerica_con_textos(p, i):
    if i == "es":
        return (f"Mezcla números con {_n(p['textos'], i)} celdas de texto (de {_n(p['presentes'], i)}; p. ej. «{p['ejemplo']}»): "
                "no se mira ni como número ni como categoría. Si ese texto significa «sin dato», deja la celda vacía.")
    return (f"Mixes numbers with {_n(p['textos'], i)} text cells (out of {_n(p['presentes'], i)}; e.g. «{p['ejemplo']}»): it is "
            "used neither as a number nor as a category. If that text means «no value», leave the cell empty.")


def _t_numero_ambiguo(p, i):
    if p.get("motivo_de_la_lectura") == "mezcla_con_punto":
        return (f"Mezcla decimales con coma y con punto (p. ej. «{p['ejemplo']}»): no se sabe cuál manda y no se mira."
                if i == "es" else
                f"Mixes decimal commas and decimal points (e.g. «{p['ejemplo']}»): there is no telling which one is meant, "
                "so it is not used.")
    if i == "es":
        return (f"{_n(p['celdas'], i)} celdas como «{p['ejemplo']}» pueden ser un decimal o un número con miles: no se mira, para "
                "no ordenar las filas por una lectura dudosa. Si son miles, quita el punto.")
    return (f"{_n(p['celdas'], i)} cells like «{p['ejemplo']}» could be a decimal or a number with thousands: it is not used, so "
            "that rows are not ranked on a doubtful reading. If they are thousands, remove the dot.")


def _t_columna_de_etiquetas(p, i):
    return ("La columna de etiquetas que aportaste: no se mira para puntuar (sería hacer trampa)." if i == "es"
            else "The labels column you provided: it is not used for scoring (that would be cheating).")


def _t_pocas_filas(p, i):
    if i == "es":
        return f"Solo {_n(p['filas'], i)} filas: con menos de {_n(p['minimo'], i)} el detector no es fiable."
    return f"Only {_n(p['filas'], i)} rows: with fewer than {_n(p['minimo'], i)} the detector is not reliable."


def _t_cambia_con_la_semilla(p, i):
    return ("Esta lista cambia con la semilla: no te fíes del orden fino." if i == "es"
            else "This list changes with the seed: do not trust its fine order.")


def _t_filas_sin_dato(p, i):
    if i == "es":
        return f"{_n(p['filas'], i)} filas sin ningún dato en las columnas que se miran no se puntuaron."
    return f"{_n(p['filas'], i)} rows with no value in any of the columns used were not scored."


def _t_numerica_casi_unica(p, i):
    if i == "es":
        return (f"«{p['columna']}» tiene {_n(p['distintos'], i)} valores enteros distintos en {_n(p['presentes'], i)} filas. Si es "
                "un identificador (un número de historia, de cliente, de pedido), quítala de las columnas que se miran: si no, "
                "sus valores extremos saldrán como motivo.")
    return (f"«{p['columna']}» has {_n(p['distintos'], i)} distinct integer values in {_n(p['presentes'], i)} rows. If it is an "
            "identifier (a record, customer or order number), remove it from the columns used: otherwise its extreme values "
            "will show up as reasons.")


def _t_textos_leidos_como_faltante(p, i):
    if i == "es":
        return (f"En «{p['columna']}», {_textos(p['textos'], i)} se leyeron como «sin dato», con la misma lista que el resto del "
                "Studio; la columna sigue siendo numérica.")
    return (f"In «{p['columna']}», {_textos(p['textos'], i)} were read as «no value», with the same list as the rest of the "
            "Studio; the column is still numeric.")


def _t_categorias_por_nombre(p, i):
    if i == "es":
        return (f"Con columnas de categorías ({_cols(p['columnas'])}) la lista depende de cómo se llamen: se codifican por orden "
                "alfabético y renombrar una categoría puede cambiar la lista. Está en revisión.")
    return (f"With category columns ({_cols(p['columnas'])}) the list depends on what the categories are called: they are encoded "
            "in alphabetical order and renaming a category can change the list. This is under review.")


def _t_categorias_con_vacias(p, i):
    if i == "es":
        return (f"En {_cols(p['columnas'])} hay celdas vacías: se codifican en un extremo del eje y pueden subir esas filas en la "
                "lista (cada fila lo dice en su motivo). Está en revisión.")
    return (f"{_cols(p['columnas'])} have empty cells: they are encoded at one end of the axis and can push those rows up the "
            "list (each row says so in its reason). This is under review.")


_TEXTOS_DE_MOTIVO = {
    # por qué una columna NO se mira
    "vacia": _t_vacia, "casi_vacia": _t_casi_vacia, "constante": _t_constante,
    "categorica_casi_unica": _t_categorica_casi_unica, "identificador_numerico": _t_identificador_numerico,
    "numeros_con_formato": _t_numeros_con_formato, "numerica_con_textos": _t_numerica_con_textos,
    "numero_ambiguo": _t_numero_ambiguo, "columna_de_etiquetas": _t_columna_de_etiquetas,
    # avisos del detector
    "pocas_filas": _t_pocas_filas, "cambia_con_la_semilla": _t_cambia_con_la_semilla, "filas_sin_dato": _t_filas_sin_dato,
    "numerica_casi_unica": _t_numerica_casi_unica, "textos_leidos_como_faltante": _t_textos_leidos_como_faltante,
    "categorias_por_nombre": _t_categorias_por_nombre, "categorias_con_vacias": _t_categorias_con_vacias,
}

#: Las claves que existen (la pantalla y las pruebas las leen de aquí, no de una copia).
CLAVES_DE_MOTIVO = tuple(_TEXTOS_DE_MOTIVO)


def texto_de_motivo(clave: str, parametros: Mapping[str, Any] | None = None) -> dict[str, str]:
    """`{es, en}` del motivo o aviso `clave` con sus `parametros`. Una clave desconocida es un error de cableado: `KeyError`."""
    f = _TEXTOS_DE_MOTIVO[clave]
    p = dict(parametros or {})
    return {"es": f(p, "es"), "en": f(p, "en")}


def digest_de_la_lista(filas_a_revisar: Sequence[Mapping[str, Any]]) -> str:
    """La huella (sha256 de la forma canónica) de la lista tal como se devuelve: el recibo la ata, y tocar una cifra la rompe."""
    return hashlib.sha256(jcs_bytes(list(filas_a_revisar))).hexdigest()


# ---------------------------------------------------------------------------
# Las cifras con etiquetas aportadas
# ---------------------------------------------------------------------------

def _precision_en_k(k: int):
    def formula(muestra: Muestra, _umbral: float | None):
        n = len(muestra.y_true)
        if n < 1:
            return Indefinida({"es": "sin filas", "en": "no rows"})
        orden = sorted(range(n), key=lambda i: (-muestra.scores[i], i))[: min(k, n)]
        return sum(1 for i in orden if muestra.y_true[i] == "1") / len(orden)
    return formula


def _motivo_sin_dos_clases(positivas: int, n: int) -> dict[str, str]:
    if positivas == 0:
        return {"es": f"no se calcula la precisión en las k primeras: ninguna de las {n} filas etiquetadas es anómala",
                "en": f"precision at k is not computed: none of the {n} labelled rows is anomalous"}
    return {"es": f"no se calcula la precisión en las k primeras: las {n} filas etiquetadas son anómalas (no hay ninguna normal)",
            "en": f"precision at k is not computed: all {n} labelled rows are anomalous (there is no normal one)"}


def cifras_con_etiquetas(puntuaciones: Sequence[float], anomala: Sequence[bool], *, k: int | None = None,
                         semilla: int = 0, remuestras: int = 1000) -> dict[str, Any]:
    """AUROC, AP y precisión@k con su intervalo, sobre las filas ETIQUETADAS (`anomala[i]` True/False). Mayor puntuación = más rara.

    Si falta una clase (todas anómalas o ninguna) cada cifra sale `value: None` con su motivo — nunca un cero. `k` por omisión es
    el número de filas marcadas anómalas (tantas como hay). El intervalo es el bootstrap por filas del núcleo (104-C2), 95 %."""
    if len(puntuaciones) != len(anomala):
        raise ValueError("puntuaciones y etiquetas tienen que alinearse fila a fila")
    n = len(puntuaciones)
    positivas = sum(1 for a in anomala if a)
    if n < 2:
        raise ValueError("hacen falta al menos 2 filas etiquetadas")
    kk = k if k is not None else max(1, positivas)
    kk = max(1, min(int(kk), n))
    muestra = Muestra(task="binary_classification", y_true=tuple("1" if a else "0" for a in anomala), classes=("0", "1"),
                      positive_label="1", scores=tuple(float(p) for p in puntuaciones), score_rule="puntuacion_del_detector")
    comun = dict(diseno="iid", estimando="fixed_model_on_population", semilla=semilla, remuestras=remuestras)
    salida: list[dict[str, Any]] = []

    def _anota(nombre: str, vm: ValorDeMetrica, ic) -> None:
        d: dict[str, Any] = {"name": nombre, "value": vm.value, "undefined_reason": vm.undefined_reason,
                             "ci_low": None, "ci_high": None, "ci_unavailable": None, "level": None, "method": None,
                             "resamples": remuestras}
        if vm.value is not None:
            if ic.ci_low is not None:
                d.update(ci_low=ic.ci_low, ci_high=ic.ci_high, level=ic.level, method=ic.method)
            else:
                d["ci_unavailable"] = ic.undefined_reason
        salida.append(d)

    for nombre, metric_id in (("auroc", "auroc"), ("average_precision", "average_precision")):
        vm = calcular(metric_id, muestra)
        _anota(nombre, vm, intervalo(metric_id, muestra, valor=vm, estratificar=False, **comun))
    formula = _precision_en_k(kk)
    p = formula(muestra, None)
    if positivas == 0 or positivas == n:
        # Sin las dos clases no hay nada que medir: «0 de k» no es una precisión, es la ausencia de anómalas (o de normales). Con
        # SU motivo, no el del AUROC (auditoría 165, menor 2: la pantalla lo enseñaba bajo «Precisión en las k primeras»).
        vm = ValorDeMetrica(metric_id="precision_at_k", formula_version="1.0.0", value=None,
                            undefined_reason=_motivo_sin_dos_clases(positivas, n), n_observations=n)
        _anota("precision_at_k", vm, None)
    elif isinstance(p, Indefinida):  # no ocurre con n >= 2, pero un valor ausente no es un cero
        vm = ValorDeMetrica(metric_id="precision_at_k", formula_version="1.0.0", value=None, undefined_reason=p.motivo,
                            n_observations=n)
        _anota("precision_at_k", vm, None)
    else:
        vm = ValorDeMetrica(metric_id="precision_at_k", formula_version="1.0.0", value=float(p), n_observations=n)
        _anota("precision_at_k", vm, intervalo("precision_at_k", muestra, valor=vm, formula=formula, estratificar=False, **comun))
    return {"rows_labelled": n, "rows_anomalous": positivas, "k": kk, "metrics": salida}


# ---------------------------------------------------------------------------
# El recibo
# ---------------------------------------------------------------------------

def recibo_de_filas_a_revisar(resultado: Mapping[str, Any], *, huella_del_csv: str, creado_en: str | None, nombre: str | None = None,
                              actor: str | None = None, version_de_motores: str | None = None,
                              etiquetas: Mapping[str, Any] | None = None,
                              filas_devueltas: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """El recibo de lo que `matrixai_engines.anomalias.puntuar_filas` devolvió (`resultado`), sin recalcular nada.

    `etiquetas` es lo que devuelve `cifras_con_etiquetas` MÁS `{"columna": …, "valor_anomalo": …, "filas_sin_etiqueta": n}`; si es
    `None` el recibo declara `labels.source = "none"` y NO lleva `metrics`. `filas_devueltas` es la lista tal como se entrega
    (por omisión, la del resultado)."""
    lista = list(filas_devueltas if filas_devueltas is not None else resultado["filas_a_revisar"])
    d, det, est = resultado["datos"], resultado["detector"], resultado["estabilidad"]
    digest_lista = digest_de_la_lista(lista)
    cuerpo_etiquetas: dict[str, Any] = {"source": "none", "note": {
        "es": "Sin etiquetas: ninguna cifra de acierto.", "en": "No labels: no accuracy figure."}}
    metricas = None
    if etiquetas is not None:
        cuerpo_etiquetas = {"source": "provided", "column": etiquetas["columna"], "anomalous_value": etiquetas["valor_anomalo"],
                            "rows_labelled": etiquetas["rows_labelled"], "rows_anomalous": etiquetas["rows_anomalous"],
                            "rows_unlabelled": etiquetas.get("filas_sin_etiqueta", 0), "k": etiquetas["k"],
                            "scope": dict(_ALCANCE_DE_LAS_ETIQUETAS)}
        metricas = list(etiquetas["metrics"])
    id_ = "rowreview-" + hashlib.sha256((huella_del_csv + digest_lista + str(det["semilla"])).encode("utf-8")).hexdigest()[:16]
    recibo: dict[str, Any] = {
        "schema_version": "1.0", "receipt_id": id_, "event_type": "anomaly_review",
        "created_at": creado_en,
        "subject": {"purpose": dict(_PROPOSITO), "actor": actor or "—"},
        "dataset": {"name": nombre or "csv", "digest": f"sha256:{huella_del_csv}", "rows_total": d["filas"],
                    "rows_scored": d["filas_puntuadas"], "rows_ignored": len(d["filas_ignoradas"])},
        "detector": {"name": det["nombre"], "seed": det["semilla"], "n_estimators": det["n_estimadores"],
                     "contamination": None, "encoding": det["codificacion"], "engine_version": version_de_motores},
        "columns": {"used": [dict(c) for c in resultado["columnas_usadas"]],
                    "excluded": [dict(c) for c in resultado["columnas_excluidas"]]},
        "stability": {"seeds": list(est["semillas"]), "k": est["k"], "mean_overlap": est["solape_medio"],
                      "min_overlap": est["solape_minimo"], "reading": est["lectura"]},
        "limits": dict(TEXTO_DE_LIMITES),
        "no_claims": {"es": list(NO_AFIRMA["es"]), "en": list(NO_AFIRMA["en"])},
        "labels": cuerpo_etiquetas,
        "steps": [{"id": "score_rows", "detector": det["nombre"], "seed": det["semilla"],
                   "entry_hash": f"sha256:{huella_del_csv}"}],   # el verificador exige `sha256:` + 64 hex (auditoría 165, B1)
        "output": {"outcome": "completed", "rows_returned": len(lista), "output_digests": {"list": digest_lista}},
        "assurance_ceiling": "A1",
        "evidence": {"attests": dict(_ATESTIGUA), "does_not_attest": dict(_NO_ATESTIGUA)},
    }
    if metricas is not None:
        recibo["metrics"] = metricas
    return recibo


def firmar_recibo_de_filas_a_revisar(recibo: dict[str, Any], *, clave: bytes | None = None, key_id: str | None = None) -> dict[str, Any]:
    """El sobre firmado (el mecanismo de siempre); no firma lo que `problemas_de_esquema` rechaza."""
    return firmar_recibo(recibo, clave=clave, key_id=key_id)
