# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b — un intervalo de Wilson cuenta FILAS: con varias filas del mismo sujeto, se CORRIGE por sujeto (Kish).

**El problema** (auditoría I4 del 140b). La cobertura conforme medida en la prueba (`conforme.medir_cobertura`), el acierto
de cada banda (`bandas.medir_bandas`), las cifras del simulador de automatización y la sensibilidad de la matriz de confusión
llevan un intervalo de Wilson del 95 % sobre las FILAS. Eso supone filas independientes. Con un reparto por sujeto
(`groups`, `groups_and_time`) las filas de la prueba son de unos pocos sujetos, cada uno con su propia tasa, y el «95 %»
cubre de verdad bastante menos: medido por la auditoría con 13 sujetos de 40 filas y una desviación del logit entre sujetos
de 0,5 o 1, 0,84 y 0,61.

**La corrección** (decisión de Roberto del 01-11, tarjeta `140b-wilson-por-sujetos`, opción A): el EFECTO DE DISEÑO de
Kish. Del 0/1 de cada fila (cubierta o no, acierto o no) y del sujeto de cada fila se estima la correlación intraclase (ICC)
por ANOVA de un factor —truncada a 0 si sale negativa—, y con ella

    deff = 1 + (m0 − 1) · ICC,     n_eff = n / deff,

donde `m0` es el tamaño medio de los sujetos corregido por su desigualdad. El intervalo es el de Wilson con `n_eff` en lugar
de `n` (la misma proporción observada). Con un sujeto por fila, `m0 = 1` y no cambia nada.

**Lo que se dice al lado: la cobertura REAL de ese intervalo, MEDIDA.** La corrección es aproximada (el ICC se estima de los
mismos pocos sujetos), así que, como el intervalo del estricto (`COBERTURA_MEDIDA_DEL_INTERVALO`), el resultado lleva la
cobertura que se midió por simulación para su número de sujetos (`COBERTURA_MEDIDA_POR_SUJETOS`, UNA tabla, con su
procedencia) y la marca de «aproximado» con menos de `SUJETOS_PARA_QUE_NO_SEA_APROXIMADO` sujetos.

**Cuándo NO cambia nada, byte a byte.** Sin `units` (`iid`, `temporal`) o con un sujeto por fila (`unidades >= filas`),
`correccion_por_sujetos` devuelve `None`, el intervalo es el Wilson de siempre y el JSON de quien lo usa no gana ni una clave.

**Los casos raros, declarados.** ICC negativo → 0 (deff = 1). Todas las filas 0, o todas 1: el ANOVA no tiene varianza con
la que estimar el ICC; se toma 0 (no hay indicio de que los sujetos difieran), y la cobertura medida, que se midió con esta
MISMA regla, dice cuánto acierta así. Un solo sujeto con varias filas: el ICC no se puede estimar y las filas valen como UNA
(`n_eff = 1`, el límite de deff = m); la cobertura no está medida y se dice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

from matrixai.training.diagnostico import intervalo_wilson

__all__ = ["AVISO_ESTUDIO_ANTERIOR_A_LA_CORRECCION", "AVISO_SIN_RECUENTO_DE_SUJETOS", "COBERTURA_MEDIDA_POR_SUJETOS", "CorreccionPorSujetos",
           "NIVEL_NOMINAL", "PROCEDENCIA_DE_LA_COBERTURA_POR_SUJETOS", "SUJETOS_PARA_QUE_NO_SEA_APROXIMADO",
           "cobertura_medida_por_sujetos", "correccion_de_recuentos", "correccion_por_sujetos", "intervalo_recortado",
           "n_efectivo_de_recuentos", "texto_de_un_conjunto", "unidades_distintas"]

#: El nivel nominal del intervalo (el `_Z_95` de `intervalo_wilson`).
NIVEL_NOMINAL = 0.95

# ---------------------------------------------------------------------------
# La cobertura MEDIDA del intervalo corregido
# ---------------------------------------------------------------------------

#: LA TABLA, UNA SOLA VEZ: (sujetos → cobertura real del intervalo de Wilson corregido por Kish, del 95 % nominal), medida
#: por simulación con ESTE código (`correccion_de_recuentos`), en `~/.agentes/140b/reparacion2/medir_kish.py` (su salida,
#: `medir_kish.json`). Cada cifra es la MEDIA de las 24 celdas con efecto del sujeto (4 formas de filas por sujeto: 5, 20 y
#: 40 fijas y 1–39 variables × 3 tasas: 0,6, 0,9 y 0,97 × 2 desviaciones del logit entre sujetos: 0,5 y 1), R = 2.000
#: repeticiones por celda, redondeada a tres decimales (semilla 20261101, 93 s). Una prueba la compara con ese fichero. NO se edita a mano ni se
#: interpola: si se mide otra rejilla, se vuelve a medir y se pega la nueva con su procedencia.
COBERTURA_MEDIDA_POR_SUJETOS: tuple[tuple[int, float], ...] = (
    (2, 0.893),
    (3, 0.906),
    (4, 0.917),
    (5, 0.914),
    (6, 0.920),
    (8, 0.920),
    (10, 0.927),
    (13, 0.930),
    (15, 0.931),
    (20, 0.935),
    (30, 0.937),
    (50, 0.939),
    (100, 0.941),
)

#: De dónde sale esa tabla: datos SINTÉTICOS (sujetos con su propia tasa), no el estudio de nadie.
PROCEDENCIA_DE_LA_COBERTURA_POR_SUJETOS = {
    "medida_por": "140b, segunda reparación (medir_kish.py)",
    "fecha": "2026-11-01",
    "datos": "sintéticos: n sujetos, cada uno con su tasa (logit normal), y sus filas 0/1 independientes dentro del sujeto",
    "rejilla": "filas por sujeto 5, 20, 40 y 1-39; tasa 0,6, 0,9 y 0,97; desviación del logit entre sujetos 0,5 y 1",
    "repeticiones": 2000,
    "cobertura": "media de las 24 celdas con efecto del sujeto, del intervalo de Wilson con n efectivo de Kish",
}

#: Por debajo de este número de sujetos el intervalo corregido se marca «aproximado». Medido (`medir_kish.json`): desde 10
#: sujetos la PEOR de las 24 celdas con efecto cubre 0,877 o más y la media 0,927 o más; con 8, la peor 0,874; con 5, 0,848;
#: con 3, 0,817. (El Wilson por filas, en esas mismas celdas, cubre de media 0,81–0,83 con cualquier número de sujetos.)
SUJETOS_PARA_QUE_NO_SEA_APROXIMADO = 10


def cobertura_medida_por_sujetos(sujetos: int) -> tuple[float | None, int | None]:
    """`(cobertura medida, sujetos del punto medido)`. Para un número que NO está en la rejilla, el punto medido inmediatamente
    menor o igual, nunca interpolado hacia arriba (la cobertura sube con los sujetos: así no se promete más de lo medido). Con
    menos sujetos que el primer punto, `(None, None)`: «no medido» no es un cero."""
    punto = None
    for medido, cobertura in COBERTURA_MEDIDA_POR_SUJETOS:
        if medido <= sujetos:
            punto = (cobertura, medido)
    return punto if punto is not None else (None, None)


# ---------------------------------------------------------------------------
# Las cuentas
# ---------------------------------------------------------------------------

def unidades_distintas(unidades: Sequence[Any] | None, indices: Sequence[int] | None = None) -> int | None:
    """Cuántos sujetos distintos hay entre las filas (`indices`, o todas). `None` si la muestra no trae `units`."""
    if unidades is None:
        return None
    if indices is None:
        return len(set(unidades))
    return len({unidades[i] for i in indices})


def intervalo_recortado(eventos: float, n: float) -> tuple[float, float] | None:
    """El intervalo de Wilson del 95 % recortado a [0, 1] (el recorte de `medir_bandas` y `medir_cobertura`), o `None`."""
    w = intervalo_wilson(eventos, n)
    return None if w is None else (max(0.0, w[0] - w[1]), min(1.0, w[0] + w[1]))


def _icc_y_m0(exitos: Sequence[int], filas: Sequence[int]) -> tuple[float | None, float | None]:
    """El ICC del 0/1 por ANOVA de un factor (truncado a [0, 1]) y `m0`, de los recuentos POR SUJETO (`exitos[i]` de
    `filas[i]`). Las sumas son `math.fsum` de términos que solo dependen de cada sujeto: el resultado NO depende del orden de
    los sujetos (el Informe Vivo tabula estas cifras y una prueba exige que casen con las del Studio al último bit).
    `(None, None)` con un solo sujeto."""
    k = len(filas)
    if k < 2:
        return None, None
    total = sum(filas)
    exitos_totales = sum(exitos)
    m0 = (total - sum(m * m for m in filas) / total) / (k - 1)
    if total == k:
        return 0.0, m0
    # Σ m_i (p_i − p̄)² = Σ e_i²/m_i − E²/N;   Σ m_i p_i (1 − p_i) = Σ e_i (m_i − e_i)/m_i
    entre = math.fsum(e * e / m for e, m in zip(exitos, filas)) - exitos_totales * exitos_totales / total
    dentro = math.fsum(e * (m - e) / m for e, m in zip(exitos, filas))
    msb = max(0.0, entre) / (k - 1)
    msw = dentro / (total - k)
    den = msb + (m0 - 1) * msw
    if den <= 0:
        # Sin varianza (todas 0 o todas 1): no hay con qué estimar el ICC; 0, declarado arriba.
        return 0.0, m0
    return min(1.0, max(0.0, (msb - msw) / den)), m0


def _num(x: float, decimales: int, es: bool) -> str:
    texto = f"{x:.{decimales}f}"
    return texto.replace(".", ",") if es else texto


def _de_cada_100(cobertura: float) -> int:
    """La cifra «de cada 100»: el entero por debajo de lo medido (88,7 → 88), para no redondear hacia arriba una promesa."""
    return int(round(cobertura * 1000)) // 10


@dataclass(frozen=True)
class CorreccionPorSujetos:
    """Un intervalo de Wilson corregido por sujeto (Kish), con lo que se sabe de cuánto acierta."""

    filas: int
    exitos: int
    unidades: int
    #: La correlación intraclase del 0/1 (ANOVA, truncada a [0, 1]); `None` con un solo sujeto.
    icc: float | None
    m0: float | None
    efecto_de_diseno: float
    #: `filas / efecto_de_diseno`, a tres decimales (lo que viaja: el Informe Vivo lo tabula y rehace el intervalo con él).
    n_efectivo: float
    #: El intervalo corregido, recortado a [0, 1].
    intervalo_95: tuple[float, float] | None
    #: El de siempre, por filas, para quien quiera ver cuánto cambia.
    intervalo_por_filas: tuple[float, float] | None
    cobertura_medida: float | None
    sujetos_del_punto_medido: int | None
    aproximado: bool
    texto: dict[str, str] = field(compare=False)
    texto_aproximado: dict[str, str] | None = field(compare=False, default=None)

    def a_json(self) -> dict[str, Any]:
        return {"metodo": "wilson_con_efecto_de_diseno_de_kish", "filas": self.filas, "unidades": self.unidades,
                "icc": self.icc, "efecto_de_diseno": self.efecto_de_diseno, "n_efectivo": self.n_efectivo,
                "intervalo_95": list(self.intervalo_95) if self.intervalo_95 else None,
                "intervalo_por_filas": list(self.intervalo_por_filas) if self.intervalo_por_filas else None,
                "cobertura_medida": self.cobertura_medida, "sujetos_del_punto_medido": self.sujetos_del_punto_medido,
                "aproximado": self.aproximado, "texto": dict(self.texto),
                "texto_aproximado": None if self.texto_aproximado is None else dict(self.texto_aproximado)}


def _frases(filas: int, unidades: int, icc: float | None, deff: float, n_eff: float,
            cobertura: float | None, medido: int | None) -> tuple[dict[str, str], dict[str, str] | None]:
    nominal = round(100 * NIVEL_NOMINAL)
    textos: dict[str, str] = {}
    for idioma in ("es", "en"):
        es = idioma == "es"
        if icc is None:
            base = (f"intervalo del {nominal} % corregido por sujeto: las {filas} filas son todas de un solo sujeto y valen "
                    "como una (no se puede medir cuánto se parecen entre sí)" if es else
                    f"{nominal}% interval corrected by subject: the {filas} rows all come from a single subject and count as "
                    "one (how alike they are cannot be measured)")
        else:
            base = (f"intervalo del {nominal} % corregido por sujeto: las {filas} filas son de {unidades} sujetos y, por lo "
                    f"que se parecen las de un mismo sujeto (correlación intraclase {_num(icc, 2, es)}), valen como unas "
                    f"{_num(n_eff, 0, es)} independientes (efecto de diseño de Kish: {_num(deff, 2, es)})" if es else
                    f"{nominal}% interval corrected by subject: the {filas} rows come from {unidades} subjects and, because "
                    f"rows of the same subject are alike (intraclass correlation {_num(icc, 2, es)}), they count as about "
                    f"{_num(n_eff, 0, es)} independent ones (Kish design effect: {_num(deff, 2, es)})")
        if cobertura is None or medido is None:
            primero = COBERTURA_MEDIDA_POR_SUJETOS[0][0] if COBERTURA_MEDIDA_POR_SUJETOS else None
            cola = (f"; con {unidades} sujeto(s) no se ha medido cuánto acierta un intervalo así (se midió desde {primero} "
                    "sujetos): no hay cifra que dar" if es else
                    f"; with {unidades} subject(s) it has not been measured how often an interval like this is right (it "
                    f"was measured from {primero} subjects): there is no figure to give")
        else:
            c = _de_cada_100(cobertura)
            if medido == unidades:
                cola = (f"; con {unidades} sujetos, un intervalo así acierta unas {c} de cada 100 veces, no {nominal} "
                        "(medido por simulación)" if es else
                        f"; with {unidades} subjects, an interval like this is right about {c} times in 100, not {nominal} "
                        "(measured by simulation)")
            elif unidades > COBERTURA_MEDIDA_POR_SUJETOS[-1][0]:
                cola = (f"; con {unidades} sujetos, un intervalo así acierta unas {c} de cada 100 veces, no {nominal} (lo "
                        f"medido llega hasta {medido} sujetos; medido por simulación)" if es else
                        f"; with {unidades} subjects, an interval like this is right about {c} times in 100, not {nominal} "
                        f"(what was measured goes up to {medido} subjects; measured by simulation)")
            else:
                cola = (f"; con {unidades} sujetos, un intervalo así acierta unas {c} de cada 100 veces o algo más, no "
                        f"{nominal} (se midió con {medido} sujetos, el punto medido más cercano por debajo; medido por "
                        "simulación)" if es else
                        f"; with {unidades} subjects, an interval like this is right about {c} times in 100 or a little "
                        f"more, not {nominal} (it was measured with {medido} subjects, the nearest measured point below; "
                        "measured by simulation)")
        textos[idioma] = base + cola
    aproximado = _frase_de_aproximado(unidades)
    return textos, aproximado


def _frase_de_aproximado(unidades: int) -> dict[str, str] | None:
    """La marca, o `None`. El umbral se decide AQUÍ, una vez: el booleano y la marca no pueden discrepar en la frontera."""
    if unidades >= SUJETOS_PARA_QUE_NO_SEA_APROXIMADO:
        return None
    nominal = round(100 * NIVEL_NOMINAL)
    return {"es": f"intervalo aproximado: con menos de {SUJETOS_PARA_QUE_NO_SEA_APROXIMADO} sujetos la corrección por "
                  f"sujeto se queda corta de su nivel del {nominal} %",
            "en": f"approximate interval: with fewer than {SUJETOS_PARA_QUE_NO_SEA_APROXIMADO} subjects the correction by "
                  f"subject falls short of its {nominal}% level"}


def _kish(exitos: Sequence[int], filas: Sequence[int]):
    """`(exitos, filas, icc, m0, deff, n_eff, total, unidades)` sin los sujetos sin filas, o `None` (sin filas, o un sujeto
    por fila). LA cuenta: `correccion_de_recuentos` y `n_efectivo_de_recuentos` (lo que tabula el Informe Vivo) la usan."""
    if len(exitos) != len(filas):
        raise ValueError("exitos y filas tienen que ser de los mismos sujetos")
    pares = [(int(e), int(m)) for e, m in zip(exitos, filas) if m]     # un sujeto sin filas aquí no cuenta
    exitos, filas = [e for e, _ in pares], [m for _, m in pares]
    total, unidades = sum(filas), len(filas)
    if total <= 0 or unidades >= total:
        return None
    icc, m0 = _icc_y_m0(exitos, filas)
    deff = float(total) if icc is None else 1.0 + (m0 - 1.0) * icc
    n_eff = max(1.0, round(total / deff, 3))
    return exitos, filas, icc, m0, deff, n_eff, total, unidades


def n_efectivo_de_recuentos(exitos: Sequence[int], filas: Sequence[int]) -> float | None:
    """Solo `n_efectivo` de `correccion_de_recuentos` (la misma cuenta, sin redactar frases): `None` si no hay que corregir."""
    k = _kish(exitos, filas)
    return None if k is None else k[5]


def correccion_de_recuentos(exitos: Sequence[int], filas: Sequence[int]) -> CorreccionPorSujetos | None:
    """La corrección de los recuentos POR SUJETO (`exitos[i]` de `filas[i]`, un elemento por sujeto con alguna fila). `None` si
    no hay filas o si cada sujeto tiene UNA fila (nada que corregir: el Wilson de siempre)."""
    cuenta = _kish(exitos, filas)
    if cuenta is None:
        return None
    exitos, filas, icc, m0, deff, n_eff, total, unidades = cuenta
    k = sum(exitos)
    intervalo = intervalo_recortado(k / total * n_eff, n_eff)
    cobertura, medido = (None, None) if icc is None else cobertura_medida_por_sujetos(unidades)
    textos, aproximado = _frases(total, unidades, icc, deff, n_eff, cobertura, medido)
    return CorreccionPorSujetos(
        filas=total, exitos=k, unidades=unidades, icc=icc, m0=m0, efecto_de_diseno=deff, n_efectivo=n_eff,
        intervalo_95=intervalo, intervalo_por_filas=intervalo_recortado(k, total), cobertura_medida=cobertura,
        sujetos_del_punto_medido=medido, aproximado=aproximado is not None, texto=textos, texto_aproximado=aproximado)


def texto_de_un_conjunto(filas: int, unidades: int) -> dict[str, Any] | None:
    """Lo que se dice UNA vez de un conjunto de cifras corregidas por sujeto (el simulador de automatización: cada cifra lleva
    su propio `n_efectivo`): cuántas filas y sujetos, cuánto acierta un intervalo así con ESOS sujetos y la marca de
    aproximado. `None` con un sujeto por fila."""
    if unidades <= 0 or unidades >= filas:
        return None
    nominal = round(100 * NIVEL_NOMINAL)
    cobertura, medido = cobertura_medida_por_sujetos(unidades)
    textos = {}
    for idioma in ("es", "en"):
        es = idioma == "es"
        base = (f"intervalos del {nominal} % corregidos por sujeto (efecto de diseño de Kish): las {filas} filas son de "
                f"{unidades} sujetos, y cada cifra cuenta lo que se parecen las filas de un mismo sujeto" if es else
                f"{nominal}% intervals corrected by subject (Kish design effect): the {filas} rows come from {unidades} "
                "subjects, and each figure accounts for how alike rows of the same subject are")
        if cobertura is None:
            cola = (f"; con {unidades} sujeto(s) no se ha medido cuánto aciertan: no hay cifra que dar" if es else
                    f"; with {unidades} subject(s) how often they are right has not been measured: there is no figure")
        else:
            c = _de_cada_100(cobertura)
            cola = (f"; con {unidades} sujetos, un intervalo así acierta unas {c} de cada 100 veces, no {nominal}, y menos "
                    "en una cifra de menos sujetos (medido por simulación)" if es else
                    f"; with {unidades} subjects, an interval like this is right about {c} times in 100, not {nominal}, and "
                    "less in a figure with fewer subjects (measured by simulation)")
        textos[idioma] = base + cola
    aproximado = _frase_de_aproximado(unidades)
    return {"filas": filas, "unidades": unidades, "cobertura_medida": cobertura, "sujetos_del_punto_medido": medido,
            "aproximado": aproximado is not None, "texto": textos, "texto_aproximado": aproximado}


def correccion_por_sujetos(exitos: Sequence[bool | int], unidades: Sequence[Hashable] | None,
                           indices: Sequence[int] | None = None) -> CorreccionPorSujetos | None:
    """La corrección de un 0/1 por fila (`exitos`) con el sujeto de cada fila (`unidades`), sobre `indices` (o todas las
    filas). `None` sin `units` o con un sujeto por fila: el intervalo de siempre, sin una clave más."""
    if unidades is None:
        return None
    if indices is None:
        indices = range(len(exitos))
    por_sujeto: dict[Hashable, list[int]] = {}
    for i in indices:
        cuenta = por_sujeto.setdefault(unidades[i], [0, 0])
        cuenta[0] += 1 if exitos[i] else 0
        cuenta[1] += 1
    return correccion_de_recuentos([c[0] for c in por_sujeto.values()], [c[1] for c in por_sujeto.values()])


#: La advertencia para quien enseña un intervalo y NO tiene el sujeto de cada fila (el simulador con unas puntuaciones
#: guardadas ANTES de esta corrección, que no llevan el sujeto): se sabe que el reparto fue por sujeto, no cuántos sujetos hay
#: en cada cifra, y no se puede corregir.
AVISO_SIN_RECUENTO_DE_SUJETOS: dict[str, str] = {
    "es": "intervalos de Wilson por FILAS sin corregir: este estudio se repartió por sujeto y lo guardado no dice de qué sujeto "
          "es cada fila, así que pueden contar como independientes filas que no lo son. Si hay varias filas por sujeto son más "
          "estrechos de lo que deberían; vuelve a preparar el simulador para corregirlos",
    "en": "uncorrected Wilson intervals by ROWS: this study was split by subject and what was saved does not say which subject "
          "each row belongs to, so they may count as independent rows that are not. With several rows per subject they are "
          "narrower than they should be; set up the simulator again to correct them",
}

#: La advertencia para una cifra de un estudio repartido por sujeto y ANTERIOR a esta corrección (lo guardado no dice si hay
#: sujetos repetidos en ella ni de qué sujeto es cada fila): su intervalo es el de Wilson por filas y no se puede corregir.
AVISO_ESTUDIO_ANTERIOR_A_LA_CORRECCION: dict[str, str] = {
    "es": "intervalo de Wilson por FILAS sin corregir: este estudio se repartió por sujeto y es anterior a la corrección por "
          "sujeto, así que no se sabe cuántos sujetos hay en esta cifra. Si hay varias filas por sujeto es más estrecho de lo "
          "que debería; vuelve a lanzar el estudio para corregirlo",
    "en": "uncorrected Wilson interval by ROWS: this study was split by subject and predates the correction by subject, so it "
          "is not known how many subjects this figure has. With several rows per subject it is narrower than it should be; run "
          "the study again to correct it",
}
