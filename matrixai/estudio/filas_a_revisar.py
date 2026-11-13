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

__all__ = ["CIFRAS_CON_ETIQUETAS", "LIMITES_MEDIDOS", "NO_AFIRMA", "TEXTO_DE_LIMITES", "cifras_con_etiquetas",
           "digest_de_la_lista", "recibo_de_filas_a_revisar", "firmar_recibo_de_filas_a_revisar"]

#: Las cifras que solo existen con etiquetas. El verificador las prohíbe sin ellas.
CIFRAS_CON_ETIQUETAS = ("auroc", "average_precision", "precision_at_k")

#: LO MEDIDO por el banco del 165-C2 (regla sellada `bab0a2c7…`; `benchmarks/c165/`): de las n filas SEMBRADAS por MatrixAI, qué
#: parte pone el detector vigente en las 2n primeras de la lista. Medias de 6 conjuntos del Studio (la categoría rara, de UNO),
#: con tasas de siembra del 1 % y del 5 %. NO es acierto sobre las anomalías de nadie: es lo que el detector apenas ve.
LIMITES_MEDIDOS = {
    "extremo": {"1 %": 0.16, "5 %": 0.55},
    "categoria_rara": {"1 %": 0.07, "5 %": 0.37},
    "combinacion_imposible": {"1 %": 0.01, "5 %": 0.13},
}


def _pc(x: float) -> str:
    return f"{round(100 * x)} %"


TEXTO_DE_LIMITES = {
    "es": ("Medido por MatrixAI sobre anomalías SEMBRADAS por nosotros, no sobre las tuyas: si en un conjunto hay n filas "
           f"con un valor extremo, el detector pone en las 2n primeras de la lista el {_pc(LIMITES_MEDIDOS['extremo']['1 %'])} "
           f"de ellas cuando son el 1 % de las filas y el {_pc(LIMITES_MEDIDOS['extremo']['5 %'])} cuando son el 5 %; con una "
           f"categoría rara, el {_pc(LIMITES_MEDIDOS['categoria_rara']['1 %'])} y el {_pc(LIMITES_MEDIDOS['categoria_rara']['5 %'])} "
           "(un solo conjunto con categóricas); y con una combinación imposible de valores que por separado son normales, "
           f"el {_pc(LIMITES_MEDIDOS['combinacion_imposible']['1 %'])} y el {_pc(LIMITES_MEDIDOS['combinacion_imposible']['5 %'])}: "
           "eso es lo que el detector apenas ve. No dice nada de lo que hay en tu fichero. Una lista que no destaca una fila "
           "no prueba que esa fila esté bien."),
    "en": ("Measured by MatrixAI on anomalies SEEDED by us, not on yours: if a dataset has n rows with an extreme value, the "
           f"detector puts {_pc(LIMITES_MEDIDOS['extremo']['1 %'])} of them in the first 2n of the list when they are 1 % of the "
           f"rows and {_pc(LIMITES_MEDIDOS['extremo']['5 %'])} when they are 5 %; with a rare category, "
           f"{_pc(LIMITES_MEDIDOS['categoria_rara']['1 %'])} and {_pc(LIMITES_MEDIDOS['categoria_rara']['5 %'])} (a single dataset "
           "with categorical columns); and with an impossible combination of values that are each normal on their own, "
           f"{_pc(LIMITES_MEDIDOS['combinacion_imposible']['1 %'])} and {_pc(LIMITES_MEDIDOS['combinacion_imposible']['5 %'])}: "
           "that is what the detector barely sees. It says nothing about what is in your file. A list that does not "
           "highlight a row does not prove the row is fine."),
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
        # Sin las dos clases no hay nada que medir: «0 de k» no es una precisión, es la ausencia de anómalas (o de normales). Mismo
        # motivo que el del AUROC, no un cero.
        vm = ValorDeMetrica(metric_id="precision_at_k", formula_version="1.0.0", value=None,
                            undefined_reason=salida[0]["undefined_reason"], n_observations=n)
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
        "steps": [{"id": "score_rows", "detector": det["nombre"], "seed": det["semilla"], "entry_hash": huella_del_csv}],
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
