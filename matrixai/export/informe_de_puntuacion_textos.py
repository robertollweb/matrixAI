# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Los rótulos del informe HTML de «¿Funciona aquí?», EN LOS DOS IDIOMAS — 137-C2.

Mismo patrón que `evaluacion_de_puntuacion_textos.py`: catálogo propio, dos redacciones por
clave, y una prueba que comprueba que las dos piden LOS MISMOS huecos.

**Lo que ya redactó el núcleo NO se copia aquí.** Los avisos, los motivos de «no aplica» o
«no se publica», las frases de «lo que este informe no es» y las del recibo viajan en el
informe ya traducidas a los dos idiomas (`{"es": …, "en": …}`); el informe HTML las toma tal
cual. Aquí solo están los rótulos del propio documento: títulos, cabeceras de tabla, el
significado de cada medida y lo que quiere decir cada nivel de garantía.

Sin adjetivos: ningún rótulo dice «bueno», «malo» ni «aceptable». Los textos que acompañan
una cifra dicen QUÉ es, no si basta.
"""

from __future__ import annotations

__all__ = ["IDIOMAS", "TEXTOS", "huecos_de", "t"]

IDIOMAS = ("es", "en")

TEXTOS: dict[str, dict[str, str]] = {
    # ---- el documento --------------------------------------------------------------
    "doc_titulo": {"es": "¿Funciona aquí? · Evaluación local de una puntuación",
                   "en": "Does it work here? · Local evaluation of a score"},
    "etiqueta": {"es": "¿Funciona aquí? · informe", "en": "Does it work here? · report"},
    "h1_una": {"es": "¿Funciona aquí «{a}»?", "en": "Does “{a}” work here?"},
    "h1_dos": {"es": "¿Funcionan aquí «{a}» y «{b}»?", "en": "Do “{a}” and “{b}” work here?"},
    "resumen_muestra": {
        "es": "{n} casos con desenlace · {e} eventos ({p}) · intervalos al {nivel} con {r} "
              "remuestras, {diseno}",
        "en": "{n} cases with an outcome · {e} events ({p}) · {nivel} intervals from {r} "
              "resamples, {diseno}"},
    "diseno_iid": {"es": "estratificadas por desenlace", "en": "stratified by outcome"},
    "diseno_paciente": {"es": "por paciente (columna «{col}»)",
                        "en": "by patient (column “{col}”)"},
    "solo_agregados": {
        "es": "Este informe no lleva filas de datos, solo agregados.",
        "en": "This report carries no data rows, only aggregates."},
    "garantia_a0": {"es": "Garantía A0 · sin firmar", "en": "Assurance A0 · unsigned"},
    "garantia_a1": {"es": "Garantía A1 · firmado", "en": "Assurance A1 · signed"},
    "garantia_a0_sentido": {
        "es": "A0: el recibo no lleva firma. Recoge lo que se midió y con qué entradas, pero "
              "nada impide que alguien lo reescriba sin que se note.",
        "en": "A0: the receipt carries no signature. It records what was measured and from "
              "which inputs, but nothing prevents someone from rewriting it unnoticed."},
    "garantia_a1_sentido": {
        "es": "A1: el recibo está firmado con la clave de esta instalación. Quien tenga esa "
              "clave comprueba que ni las entradas declaradas ni las cifras cambiaron desde que "
              "se firmó. No pasa de ahí: no valora la puntuación ni cómo se registró el "
              "desenlace.",
        "en": "A1: the receipt is signed with this installation's key. Whoever holds that key "
              "can check that neither the declared inputs nor the figures changed since it was "
              "signed. It goes no further: it does not assess the score or how the outcome "
              "was recorded."},
    # ---- tarjetas ------------------------------------------------------------------
    "q_ordena": {"es": "¿Ordena bien?", "en": "Does it rank well?"},
    "q_riesgo": {"es": "¿Acierta en el riesgo?", "en": "Is the risk right?"},
    "q_umbral": {"es": "Al umbral de hoy", "en": "At today's threshold"},
    "q_grupos": {"es": "¿Con quién falla?", "en": "Who does it fail for?"},
    "q_frente": {"es": "Frente a la otra", "en": "Against the other score"},
    "na": {"es": "no aplica", "en": "not applicable"},
    "pill_ic_sobre_azar": {"es": "el intervalo no incluye 0,5", "en": "the interval excludes 0.5"},
    "pill_ic_con_azar": {"es": "el intervalo incluye 0,5 (el azar)",
                         "en": "the interval includes 0.5 (chance)"},
    "pill_ic_bajo_azar": {"es": "por debajo de 0,5: ¿dirección al revés?",
                          "en": "below 0.5: is the direction reversed?"},
    "pill_interc_excluye": {"es": "intercepto: el intervalo no incluye 0",
                            "en": "intercept: the interval excludes 0"},
    "pill_interc_incluye": {"es": "intercepto: el intervalo incluye 0",
                            "en": "intercept: the interval includes 0"},
    "pill_avisos": {"es": "{k} avisos", "en": "{k} alerts"},
    "pill_sin_umbral": {"es": "sin umbral", "en": "no threshold"},
    "pill_sin_grupos": {"es": "sin subgrupos", "en": "no subgroups"},
    "auroc_rotulo": {"es": "AUROC", "en": "AUROC"},
    "ic_linea": {"es": "IC {nivel}: {a} a {b}", "en": "{nivel} CI: {a} to {b}"},
    "ic_sin": {"es": "sin intervalo", "en": "no interval"},
    "ap_linea": {"es": "AP {v} (el azar daría {z})", "en": "AP {v} (chance would give {z})"},
    "ref_linea": {
        "es": "Referencia publicada {r} · estimación local {e}: la referencia cae {pos} del "
              "intervalo local",
        "en": "Published reference {r} · local estimate {e}: the reference falls {pos} the "
              "local interval"},
    "ref_dentro": {"es": "dentro", "en": "inside"},
    "ref_fuera": {"es": "fuera", "en": "outside"},
    "ref_etiqueta": {"es": "Referencia publicada frente a estimación local",
                     "en": "Published reference against local estimate"},
    "citl_linea": {
        "es": "riesgo medio prometido − observado · {ic}",
        "en": "mean risk promised − observed · {ic}"},
    "citl_pp": {"es": "{v} puntos porcentuales", "en": "{v} percentage points"},
    "calib_detalle": {
        "es": "Intercepto {a} ({ia}) · pendiente {b} ({ib}) · Brier {br} · ECE {ece}",
        "en": "Intercept {a} ({ia}) · slope {b} ({ib}) · Brier {br} · ECE {ece}"},
    "umbral_titulo": {"es": "Al umbral de hoy ({u})", "en": "At today's threshold ({u})"},
    "umbral_sens_linea": {"es": "sensibilidad · {ic}", "en": "sensitivity · {ic}"},
    "umbral_esp_linea": {"es": "Especificidad {v} ({ic})", "en": "Specificity {v} ({ic})"},
    "grupos_medidos": {"es": "Grupos medidos: {k}", "en": "Groups measured: {k}"},
    "grupos_insuf": {"es": "Grupos con evidencia insuficiente: {k}",
                     "en": "Groups with insufficient evidence: {k}"},
    "grupos_todos": {"es": "Todos los grupos con evidencia suficiente",
                     "en": "Every group has sufficient evidence"},
    "grupos_extremos": {
        "es": "AUROC más bajo: {a} ({va}) · más alto: {b} ({vb}). Cada grupo lleva su "
              "intervalo en el anexo: los intervalos de grupos pequeños son anchos.",
        "en": "Lowest AUROC: {a} ({va}) · highest: {b} ({vb}). Each group carries its "
              "interval in the annex: intervals of small groups are wide."},
    "grupos_ninguno": {"es": "No se declaró ninguna columna de subgrupos.",
                       "en": "No subgroup column was declared."},
    "frente_titulo": {"es": "«{a}» frente a «{b}»", "en": "“{a}” against “{b}”"},
    "frente_a_mayor": {"es": "AUROC de «{a}» mayor que el de «{b}»",
                       "en": "AUROC of “{a}” higher than that of “{b}”"},
    "frente_b_mayor": {"es": "AUROC de «{a}» menor que el de «{b}»",
                       "en": "AUROC of “{a}” lower than that of “{b}”"},
    "frente_sin_dif": {"es": "el intervalo de la diferencia incluye 0",
                       "en": "the interval of the difference includes 0"},
    "frente_dif_linea": {"es": "diferencia de AUROC · {ic}", "en": "AUROC difference · {ic}"},
    "frente_filas": {"es": "Sobre las {n} filas que tienen las dos ({e} eventos).",
                     "en": "Over the {n} rows that have both ({e} events)."},
    "frente_directa": {
        "es": "Es la comparación más directamente interpretable: las dos puntuaciones se "
              "miden sobre los mismos casos.",
        "en": "This is the most directly interpretable comparison: both scores are measured "
              "over the same cases."},
    "frente_valores": {"es": "AUROC «{a}» {va} · AUROC «{b}» {vb}",
                       "en": "AUROC “{a}” {va} · AUROC “{b}” {vb}"},
    # ---- R6: la página de conclusiones y los anexos --------------------------------
    "s_conclusiones": {"es": "Conclusiones", "en": "Conclusions"},
    "c_intro": {
        "es": "Qué se evaluó: desenlace «{col}» (evento = «{ev}»), {n} casos con {e} eventos "
              "({p}). Horizonte del desenlace: {hor}. Momento de la puntuación: {mom}.",
        "en": "What was evaluated: outcome “{col}” (event = “{ev}”), {n} cases with {e} events "
              "({p}). Outcome horizon: {hor}. Score timing: {mom}."},
    "c_cifras": {"es": "Cifras clave:", "en": "Key figures:"},
    "c_k_auroc": {"es": "AUROC {v} ({ic})", "en": "AUROC {v} ({ic})"},
    "c_k_riesgo": {"es": "riesgo que dice − observado: {v} puntos porcentuales ({ic})",
                   "en": "risk it states − observed: {v} percentage points ({ic})"},
    "c_k_umbral": {"es": "umbral {u}: sensibilidad {s}, especificidad {e}",
                   "en": "threshold {u}: sensitivity {s}, specificity {e}"},
    "c_k_vp": {"es": "VPP {p}, VPN {n}", "en": "PPV {p}, NPV {n}"},
    "c_k_grupos": {"es": "subgrupos: {k} medidos, {i} sin evidencia suficiente",
                   "en": "subgroups: {k} measured, {i} without sufficient evidence"},
    "c_ref_corta": {
        "es": "Referencia publicada {r} frente a estimación local {e}: la referencia cae {pos} "
              "del intervalo local. Pueden no ser comparables (población, desenlace, "
              "seguimiento e incertidumbre); caer fuera del intervalo no demuestra que la "
              "cifra original esté mal.",
        "en": "Published reference {r} against local estimate {e}: the reference falls {pos} "
              "the local interval. They may not be comparable (population, outcome, follow-up "
              "and uncertainty); falling outside the interval does not show that the "
              "original figure is wrong."},
    "c_frente_linea": {
        "es": "«{a}» frente a «{b}»: diferencia de AUROC {d} ({ic}) sobre {n} casos con "
              "las dos ({e} eventos). {directa}",
        "en": "“{a}” against “{b}”: AUROC difference {d} ({ic}) over {n} cases with both "
              "({e} events). {directa}"},
    "frente_dif_texto": {"es": "Diferencia de AUROC: {d} ({ic}).",
                         "en": "AUROC difference: {d} ({ic})."},
    "l_rep_no": {
        "es": "La muestra se declaró NO representativa (MatrixAI no lo comprueba): la "
              "calibración, el beneficio neto y la curva de decisión describen la muestra "
              "aportada, no la población de uso; VPP y VPN no se publican.",
        "en": "The sample was declared NOT representative (MatrixAI does not check it): "
              "calibration, net benefit and the decision curve describe the supplied sample, "
              "not the population of use; PPV and NPV are not published."},
    "l_rep_si": {
        "es": "Que la muestra representa a la población de uso lo declara quien ejecuta la "
              "evaluación, y MatrixAI no lo comprueba.",
        "en": "That the sample represents the population of use is declared by whoever runs "
              "the evaluation, and MatrixAI does not check it."},
    "l_dev_no": {
        "es": "Se declaró que los casos no participaron en el desarrollo de la puntuación "
              "(MatrixAI no lo comprueba).",
        "en": "It was declared that the cases did not take part in the development of the "
              "score (MatrixAI does not check it)."},
    "l_clinica": {
        "es": "No es una validación para uso clínico ni una evaluación regulatoria: es "
              "retrospectiva y depende de cómo se registró el desenlace.",
        "en": "It is not a validation for clinical use nor a regulatory assessment: it is "
              "retrospective and depends on how the outcome was recorded."},
    "l_paradoja": {
        "es": "Si la puntuación ya se usaba para decidir, parecerá peor de lo que es "
              "(paradoja del tratamiento).",
        "en": "If the score was already used to decide, it will look worse than it is "
              "(treatment paradox)."},
    "a_en_llano": {"es": "En lenguaje llano", "en": "In plain language"},
    "a_limites": {"es": "Anexo {l} · Límites completos", "en": "Annex {l} · Full limits"},
    "c_score": {"es": "«{c}»: {decl}.", "en": "“{c}”: {decl}."},
    "c_ordena": {
        "es": "AUROC {v} ({ic}). Dicho en llano: si se toman al azar un caso con evento y uno "
              "sin evento, la puntuación señala más riesgo en el caso con evento en el {pp} "
              "de las parejas. {pill}.",
        "en": "AUROC {v} ({ic}). In plain terms: if one case with the event and one without "
              "are picked at random, the score points to more risk in the case with the event "
              "in {pp} of the pairs. {pill}."},
    "c_ordena_na": {"es": "{motivo}", "en": "{motivo}"},
    "c_ref": {
        "es": "Referencia publicada: {r}. Estimación local: {e} ({ic}). La referencia cae "
              "{pos} del intervalo local.",
        "en": "Published reference: {r}. Local estimate: {e} ({ic}). The reference falls "
              "{pos} the local interval."},
    "c_riesgo": {
        "es": "Riesgo medio que dice la puntuación menos riesgo observado: {v} puntos "
              "porcentuales ({ic}); un valor positivo es que la puntuación dice más riesgo "
              "del que hubo. Intercepto {a} ({ia}), pendiente {b} ({ib}).",
        "en": "Mean risk the score states minus observed risk: {v} percentage points ({ic}); "
              "a positive value means the score states more risk than there was. Intercept "
              "{a} ({ia}), slope {b} ({ib})."},
    "c_riesgo_na": {"es": "No aplica. {motivo}", "en": "Not applicable. {motivo}"},
    "c_umbral": {
        "es": "Con el umbral de uso ({u}) hubo {k} casos con aviso. Sensibilidad {s} ({ics}): "
              "de cada 100 eventos, {s100} tenían aviso. Especificidad {e} ({ice}): de cada "
              "100 casos sin evento, {e100} no tenían aviso.",
        "en": "At the threshold of use ({u}) there were {k} cases with an alert. Sensitivity "
              "{s} ({ics}): out of every 100 events, {s100} had an alert. Specificity {e} "
              "({ice}): out of every 100 cases without an event, {e100} had no alert."},
    "c_umbral_vp": {
        "es": "VPP {vpp} ({icp}): de cada 100 avisos, {p100} acabaron en evento. VPN {vpn} "
              "({icn}): de cada 100 casos sin aviso, {n100} no tuvieron evento.",
        "en": "PPV {vpp} ({icp}): out of every 100 alerts, {p100} ended in an event. NPV {vpn} "
              "({icn}): out of every 100 cases with no alert, {n100} had no event."},
    "c_umbral_sin_vp": {"es": "VPP y VPN: {motivo}", "en": "PPV and NPV: {motivo}"},
    "c_umbral_sin": {"es": "{motivo}", "en": "{motivo}"},
    "c_grupos": {"es": "{medidos} {insuf}", "en": "{medidos} {insuf}"},
    "c_rotulo": {
        "es": "Calibración, beneficio neto y curva de decisión {rotulo}. {porque}",
        "en": "Calibration, net benefit and the decision curve {rotulo}. {porque}"},
    "c_lim": {"es": "Límites de este informe", "en": "Limits of this report"},
    "c_lim_firma": {
        "es": "La firma acredita integridad comprobable con la clave; el valor científico "
              "depende del diseño y de los datos.",
        "en": "The signature certifies integrity that can be checked with the key; the "
              "scientific value depends on the design and on the data."},
    "c_lim_sin_firma": {
        "es": "Este informe no está firmado (garantía A0).",
        "en": "This report is not signed (assurance A0)."},
    "c_anexos": {
        "es": "Más límites, definiciones en llano y detalles técnicos: en los anexos de abajo.",
        "en": "More limits, plain-language definitions and technical details: in the annexes "
              "below."},
    "a_puntuacion": {"es": "Anexo {l} · Detalle de «{c}»", "en": "Annex {l} · Detail of “{c}”"},
    "a_graficas": {"es": "Anexo {l} · Gráficas", "en": "Annex {l} · Charts"},
    "a_comparacion": {"es": "Anexo {l} · Una frente a otra", "en": "Annex {l} · One against the other"},
    "a_como": {"es": "Anexo {l} · Cómo se hizo y recibo", "en": "Annex {l} · How it was done and receipt"},
    "a_tripod": {"es": "Anexo {l} · Ficha TRIPOD+AI de validación externa",
                 "en": "Annex {l} · TRIPOD+AI external-validation record"},
    # ---- secciones -----------------------------------------------------------------
    "s_avisos": {"es": "Avisos", "en": "Warnings"},
    "s_graficas": {"es": "Gráficas", "en": "Charts"},
    "s_umbral": {"es": "Al umbral de uso: {u}", "en": "At the threshold of use: {u}"},
    "s_subgrupos": {"es": "Por subgrupos", "en": "By subgroup"},
    "s_comparacion": {"es": "Una frente a otra", "en": "One against the other"},
    "s_no_es": {"es": "Lo que este informe no es", "en": "What this report is not"},
    "s_entrada": {"es": "Cómo se hizo", "en": "How it was done"},
    "s_recibo": {"es": "Recibo", "en": "Receipt"},
    "s_tripod": {"es": "Ficha TRIPOD+AI de validación externa",
                 "en": "TRIPOD+AI external-validation record"},
    # ---- tabla al umbral -----------------------------------------------------------
    "th_medida": {"es": "Medida", "en": "Measure"},
    "th_valor": {"es": "Valor", "en": "Value"},
    "th_ic": {"es": "Intervalo al 95 %", "en": "95 % interval"},
    "th_sentido": {"es": "Qué quiere decir", "en": "What it means"},
    "m_sens": {"es": "Sensibilidad", "en": "Sensitivity"},
    "m_esp": {"es": "Especificidad", "en": "Specificity"},
    "m_vpp": {"es": "VPP", "en": "PPV"},
    "m_vpn": {"es": "VPN", "en": "NPV"},
    "m_bn": {"es": "Beneficio neto", "en": "Net benefit"},
    "d_sens": {"es": "De cada 100 eventos, cuántos tenían aviso.",
               "en": "Out of every 100 events, how many had an alert."},
    "d_esp": {"es": "De cada 100 casos sin evento, cuántos no tenían aviso.",
              "en": "Out of every 100 cases without an event, how many had no alert."},
    "d_vpp": {"es": "De cada 100 avisos, cuántos acabaron en evento.",
              "en": "Out of every 100 alerts, how many ended in an event."},
    "d_vpn": {"es": "De cada 100 casos sin aviso, cuántos no tuvieron evento.",
              "en": "Out of every 100 cases with no alert, how many had no event."},
    "d_bn": {
        "es": "Frente a avisar a todos: {todos}; a nadie: {nadie}. Con probabilidades, más "
              "alto es más beneficio.",
        "en": "Against alerting everyone: {todos}; no one: {nadie}. Higher means more benefit."},
    "matriz": {
        "es": "Matriz de confusión: {tp} con aviso y evento, {fp} con aviso y sin evento, "
              "{fn} sin aviso y con evento, {tn} sin aviso y sin evento. "
              "Regla del aviso: puntuación {regla} el umbral.",
        "en": "Confusion matrix: {tp} with alert and event, {fp} with alert and no event, "
              "{fn} with no alert and an event, {tn} with no alert and no event. "
              "Alert rule: score {regla} the threshold."},
    "regla_mayor": {"es": "mayor o igual que", "en": "at or above"},
    "regla_menor": {"es": "menor o igual que", "en": "at or below"},
    "sin_valor": {"es": "—", "en": "—"},
    # ---- subgrupos -----------------------------------------------------------------
    "th_grupo": {"es": "Grupo", "en": "Group"},
    "th_casos": {"es": "Casos", "en": "Cases"},
    "th_eventos": {"es": "Eventos", "en": "Events"},
    "th_auroc": {"es": "AUROC (IC 95 %)", "en": "AUROC (95 % CI)"},
    "th_citl": {"es": "Riesgo prometido − observado", "en": "Risk promised − observed"},
    "evidencia_insuficiente": {
        "es": "Evidencia insuficiente (menos de {f} filas o de {e} casos en alguno de los dos "
              "desenlaces): sin número.",
        "en": "Insufficient evidence (fewer than {f} rows, or fewer than {e} cases in either "
              "outcome): no number."},
    "sin_dato_grupo": {"es": "(sin dato)", "en": "(no data)"},
    "subgrupos_pie": {
        "es": "Un grupo sin evidencia suficiente no publica ni su tamaño. Suprimir grupos "
              "pequeños ayuda, pero no garantiza anonimización.",
        "en": "A group without sufficient evidence does not even publish its size. Suppressing "
              "small groups helps, but does not guarantee anonymisation."},
    # ---- gráficas ------------------------------------------------------------------
    "g_auroc": {"es": "Cómo ordena: AUROC con su intervalo",
                "en": "How it ranks: AUROC with its interval"},
    "g_auroc_aria": {"es": "AUROC de cada puntuación con su intervalo, frente a 0,5",
                     "en": "AUROC of each score with its interval, against 0.5"},
    "g_auroc_azar": {"es": "0,5 = el azar", "en": "0.5 = chance"},
    "g_auroc_prometida": {"es": "◆ cifra prometida", "en": "◆ promised figure"},
    "g_roc": {"es": "Curva ROC y umbral de uso", "en": "ROC curve and threshold of use"},
    "g_roc_aria": {"es": "Sensibilidad frente a falsas alarmas al umbral de uso",
                   "en": "Sensitivity against false alarms at the threshold of use"},
    "g_roc_x": {"es": "Falsas alarmas (1 − especificidad)", "en": "False alarms (1 − specificity)"},
    "g_roc_y": {"es": "Casos detectados (sensibilidad)", "en": "Cases detected (sensitivity)"},
    "g_roc_pie": {
        "es": "La línea es la curva ROC (unos 40 puntos de la curva exacta, sin interpolar). Cada "
              "punto grande es el umbral de uso, con el intervalo de cada eje. La diagonal es "
              "el azar.",
        "en": "The line is the ROC curve (about 40 points of the exact curve, not interpolated). "
              "Each large dot is the threshold of use, with the interval on each axis. The "
              "diagonal is chance."},
    "g_fiab": {"es": "Cuánto acierta en el riesgo: curva de fiabilidad",
               "en": "How right the risk is: reliability curve"},
    "g_fiab_aria": {"es": "Riesgo observado frente a riesgo que dice la puntuación",
                    "en": "Observed risk against the risk the score states"},
    "g_fiab_x": {"es": "Riesgo que dice la puntuación", "en": "Risk the score states"},
    "g_fiab_y": {"es": "Lo que pasó de verdad", "en": "What actually happened"},
    "g_fiab_pie": {
        "es": "Tamaño del punto = casos del tramo. La diagonal es la fiabilidad perfecta. "
              "Tramos sin cifra (pocas filas, evidencia insuficiente): {k}.",
        "en": "Dot size = cases in the bin. The diagonal is perfect reliability. "
              "Bins without a figure (few rows, insufficient evidence): {k}."},
    "g_fiab_tip": {"es": "{n} casos: la puntuación dice {p}, pasó en {o}",
                   "en": "{n} cases: the score states {p}, it happened in {o}"},
    "g_fiab_pie_sin": {
        "es": "Tamaño del punto = casos del tramo. La diagonal es la fiabilidad perfecta.",
        "en": "Dot size = cases in the bin. The diagonal is perfect reliability."},
    "g_dca": {"es": "Utilidad: curva de decisión", "en": "Usefulness: decision curve"},
    "g_dca_aria": {"es": "Beneficio neto frente al umbral de probabilidad",
                   "en": "Net benefit against the probability threshold"},
    "g_dca_x": {"es": "Umbral de probabilidad", "en": "Probability threshold"},
    "g_dca_y": {"es": "Beneficio neto", "en": "Net benefit"},
    "g_dca_todos": {"es": "avisar a todos", "en": "alert everyone"},
    "g_dca_nadie": {"es": "avisar a nadie (0)", "en": "alert no one (0)"},
    "g_dca_pie": {
        "es": "La curva de la puntuación por encima de las dos referencias dice que, a ese "
              "umbral, avisar con ella da más beneficio que avisar a todos o a nadie.",
        "en": "A score curve above both references means that, at that threshold, alerting "
              "with it gives more benefit than alerting everyone or no one."},
    "g_no_aplica": {"es": "«{col}»: {motivo}", "en": "“{col}”: {motivo}"},
    # ---- cómo se hizo --------------------------------------------------------------
    "th_puntuacion": {"es": "Puntuación", "en": "Score"},
    "th_declarado": {"es": "Declarado", "en": "Declared"},
    "decl_tipo_probabilidad": {"es": "probabilidad (0 a 1)", "en": "probability (0 to 1)"},
    "decl_tipo_probabilidad_pct": {"es": "probabilidad en % (0 a 100)",
                                   "en": "probability in % (0 to 100)"},
    "decl_tipo_puntuacion": {"es": "escala de puntos", "en": "points scale"},
    "decl_dir_mas_alto": {"es": "más alto = más riesgo", "en": "higher = more risk"},
    "decl_dir_mas_bajo": {"es": "más bajo = más riesgo", "en": "lower = more risk"},
    "decl_umbral": {"es": "umbral de uso {u}", "en": "threshold of use {u}"},
    "decl_sin_umbral": {"es": "sin umbral de uso", "en": "no threshold of use"},
    "decl_origen": {"es": "origen: {o}", "en": "source: {o}"},
    "decl_sin_origen": {"es": "origen: no consta", "en": "source: not reported"},
    "decl_desarrollo": {"es": "casos en su desarrollo: {v}",
                        "en": "cases in its development: {v}"},
    "dev_si": {"es": "sí", "en": "yes"},
    "dev_no": {"es": "no", "en": "no"},
    "dev_no_se_sabe": {"es": "no se sabe", "en": "unknown"},
    "kv_horizonte": {"es": "Horizonte del desenlace", "en": "Outcome horizon"},
    "kv_momento": {"es": "Momento de la puntuación", "en": "Score timing"},
    "kv_donde": {"es": "Dónde se calculó", "en": "Where it was calculated"},
    "decl_prometida": {"es": "cifra prometida (AUROC) {v}", "en": "promised figure (AUROC) {v}"},
    "decl_filas": {
        "es": "{u} filas usadas · {f} sin puntuación (quedan fuera)",
        "en": "{u} rows used · {f} without a score (left out)"},
    "decl_unidades": {"es": "{k} pacientes", "en": "{k} patients"},
    "kv_fichero": {"es": "Fichero", "en": "File"},
    "kv_huella": {"es": "Huella del CSV (sha256)", "en": "CSV digest (sha256)"},
    "kv_lectura": {"es": "Lectura del CSV", "en": "CSV reading"},
    "kv_lectura_valor": {"es": "codificación {c} · separador {s} · {n} filas leídas",
                         "en": "encoding {c} · separator {s} · {n} rows read"},
    "kv_desenlace": {"es": "Desenlace", "en": "Outcome"},
    "kv_desenlace_valor": {
        "es": "columna «{col}», evento = «{ev}» ({e} eventos, {ne} sin evento)",
        "en": "column “{col}”, event = “{ev}” ({e} events, {ne} without)"},
    "kv_fuera": {
        "es": "{a} filas sin desenlace y {b} filas con otro número de columnas quedan fuera",
        "en": "{a} rows without an outcome and {b} rows with a different number of columns "
              "are left out"},
    "kv_representa": {"es": "¿La muestra representa a la población de uso?",
                      "en": "Does the sample represent the population of use?"},
    "si": {"es": "sí", "en": "yes"},
    "no": {"es": "no", "en": "no"},
    "kv_remuestreo": {"es": "Remuestreo", "en": "Resampling"},
    "kv_remuestreo_valor": {
        "es": "{r} remuestras (bootstrap), {diseno}; nivel {nivel}, semilla {semilla}",
        "en": "{r} resamples (bootstrap), {diseno}; level {nivel}, seed {semilla}"},
    "kv_subgrupos": {"es": "Columnas de subgrupo", "en": "Subgroup columns"},
    "kv_ninguna": {"es": "ninguna", "en": "none"},
    "kv_versiones": {"es": "Versiones", "en": "Versions"},
    "kv_versiones_valor": {
        "es": "núcleo MatrixAI {n} · informe {i} · catálogo de métricas {c}",
        "en": "MatrixAI core {n} · report {i} · metrics catalogue {c}"},
    "kv_variantes": {"es": "Escrituras unidas en «{v}»: {lista}",
                     "en": "Spellings merged into “{v}”: {lista}"},
    "kv_indice": {"es": "Índice de fila sin nombre", "en": "Unnamed row index"},
    # ---- recibo --------------------------------------------------------------------
    "r_id": {"es": "Identificador del recibo", "en": "Receipt identifier"},
    "r_fecha": {"es": "Emitido", "en": "Issued"},
    "r_quien": {"es": "Lo pidió", "en": "Requested by"},
    "r_huella_informe": {"es": "Huella de las cifras (sha256)", "en": "Digest of the figures (sha256)"},
    "r_firma": {"es": "Firma (HMAC)", "en": "Signature (HMAC)"},
    "r_sin_firma": {"es": "sin firmar", "en": "unsigned"},
    "r_clave": {"es": "clave {k}, firma {s}…", "en": "key {k}, signature {s}…"},
    "r_atestigua": {"es": "Qué atestigua", "en": "What it attests"},
    "r_no_atestigua": {"es": "Qué no atestigua", "en": "What it does not attest"},
    "r_garantia": {"es": "Nivel de garantía", "en": "Assurance level"},
    # ---- TRIPOD+AI -----------------------------------------------------------------
    "tp_intro": {
        "es": "TRIPOD+AI (BMJ, 2024) es la lista que se pide para publicar la validación de un "
              "modelo de predicción. Aquí se rellena, tema a tema, lo que esta evaluación midió "
              "o recibió declarado; lo demás dice «no consta». Los temas van agrupados y sin "
              "numerar: la numeración del listado oficial no se reproduce, y quien rellene el "
              "formulario oficial encontrará aquí el contenido de cada tema. No es una "
              "declaración de cumplimiento ni sustituye al criterio de quien firma el estudio.",
        "en": "TRIPOD+AI (BMJ, 2024) is the checklist requested to publish the validation of a "
              "prediction model. Here, topic by topic, what this evaluation measured or "
              "received as a declaration is filled in; everything else says “not reported”. "
              "Topics are grouped and unnumbered: the numbering of the official checklist is "
              "not reproduced, and whoever fills in the official form will find here the "
              "content of each topic. This is not a statement of compliance and does not "
              "replace the judgement of whoever signs the study."},
    "tp_th_tema": {"es": "Tema", "en": "Topic"},
    "tp_th_estado": {"es": "Estado", "en": "Status"},
    "tp_th_dato": {"es": "Dato", "en": "Entry"},
    "tp_medido": {"es": "Medido", "en": "Measured"},
    "tp_declarado": {"es": "Declarado por quien evalúa", "en": "Declared by whoever evaluates"},
    "tp_falta": {"es": "No consta", "en": "Not reported"},
    "tp_no_aplica": {"es": "No aplica", "en": "Not applicable"},
    # ítems propios (los que el inventario del paquete no tiene)
    "vx_representatividad": {"es": "¿La muestra representa a la población de uso?",
                             "en": "Does the sample represent the population of use?"},
    "vx_tamano": {"es": "Tamaño de la muestra y eventos", "en": "Sample size and events"},
    "vx_puntuacion": {"es": "Puntuación evaluada (modelo o escala)",
                      "en": "Score evaluated (model or scale)"},
    "vx_desenlace": {"es": "Variable de resultado observada", "en": "Observed outcome variable"},
    "vx_comparacion": {"es": "Comparación entre las dos puntuaciones",
                       "en": "Comparison between the two scores"},
    "vx_incertidumbre": {"es": "Incertidumbre (remuestreo)", "en": "Uncertainty (resampling)"},
    "vx_reproducibilidad": {"es": "Versiones y semilla", "en": "Versions and seed"},
    # motivos propios de «no consta»
    "mo_no_recoge": {"es": "esta evaluación no recoge este dato", "en": "this evaluation does not collect this"},
    "mo_predictores": {
        "es": "la puntuación llega ya calculada: esta evaluación no ve sus variables de entrada",
        "en": "the score arrives already computed: this evaluation does not see its input "
              "variables"},
    "mo_del_autor": {"es": "lo declara quien firma el estudio, no esta evaluación",
                     "en": "whoever signs the study declares this, not this evaluation"},
    "mo_proceso": {
        "es": "no se recogió cómo se decide hoy sin la puntuación; solo consta el umbral de "
              "uso, si se declaró",
        "en": "how decisions are made today without the score was not collected; only the "
              "threshold of use is on record, if one was declared"},
    "mo_una_sola": {"es": "se evaluó una sola puntuación", "en": "only one score was evaluated"},
    "mo_sin_comun": {"es": "las dos puntuaciones no tienen filas en común con desenlace",
                     "en": "the two scores share no rows with an outcome"},
    "mo_sin_umbral": {"es": "no se declaró umbral de uso en ninguna puntuación",
                      "en": "no threshold of use was declared for any score"},
    # valores de ítems
    "vi_fuente": {"es": "CSV aportado por quien evalúa: {f} ({n} filas leídas)",
                  "en": "CSV supplied by whoever evaluates: {f} ({n} rows read)"},
    "vi_fuente_sin_nombre": {"es": "CSV aportado por quien evalúa, sin nombre de fichero ({n} filas leídas)",
                             "en": "CSV supplied by whoever evaluates, no file name ({n} rows read)"},
    "vi_tamano": {
        "es": "{n} filas con desenlace, {e} eventos; {s} filas sin desenlace y {m} con otro "
              "número de columnas quedan fuera",
        "en": "{n} rows with an outcome, {e} events; {s} rows without an outcome and {m} with "
              "a different number of columns are left out"},
    "vi_puntuacion": {"es": "«{c}»: {tipo}, {dir}; {orig}", "en": "“{c}”: {tipo}, {dir}; {orig}"},
    "vx_desarrollo": {"es": "¿Participaron estos casos en el desarrollo de la puntuación?",
                      "en": "Did these cases take part in the development of the score?"},
    "vi_desarrollo": {"es": "«{c}»: {v}; {ext}", "en": "“{c}”: {v}; {ext}"},
    "vi_desarrollo_no_externa": {"es": "no es una validación externa",
                                 "en": "this is not an external validation"},
    "vi_desarrollo_declarado": {"es": "declarado, MatrixAI no lo comprueba",
                                "en": "declared, MatrixAI does not check it"},
    "vi_horizonte": {"es": "horizonte del desenlace: {h}; puntuación calculada: {m}",
                     "en": "outcome horizon: {h}; score calculated: {m}"},
    "vi_desenlace": {"es": "columna «{c}», evento = «{e}»", "en": "column “{c}”, event = “{e}”"},
    "vi_prev_obs": {"es": "«{c}»: {p}", "en": "“{c}”: {p}"},
    "vi_prev_decl": {
        "es": "la de esta muestra ({p}), declarada por quien evalúa como la de la población de uso",
        "en": "this sample's ({p}), declared by whoever evaluates as the population of use's"},
    "vi_discr": {"es": "«{c}»: AUROC {a} ({ia}) · AP {p}", "en": "“{c}”: AUROC {a} ({ia}) · AP {p}"},
    "vi_calib": {"es": "«{c}»: ECE {e} · intercepto {a} · pendiente {b}",
                 "en": "“{c}”: ECE {e} · intercept {a} · slope {b}"},
    "vi_calib_na": {"es": "«{c}»: no aplica", "en": "“{c}”: not applicable"},
    "vi_umbral": {"es": "«{c}»: umbral de uso {u}, con su tabla de sensibilidad y especificidad",
                  "en": "“{c}”: threshold of use {u}, with its sensitivity and specificity table"},
    "vi_umbral_sin": {"es": "«{c}»: sin umbral", "en": "“{c}”: no threshold"},
    "vi_dca": {"es": "«{c}»: {k} umbrales de probabilidad; supera a las dos referencias en {s}",
               "en": "“{c}”: {k} probability thresholds; beats both references at {s}"},
    "vi_dca_na": {"es": "«{c}»: no aplica", "en": "“{c}”: not applicable"},
    "vi_dca_sin": {"es": "«{c}»: no se declaró el rango de umbrales",
                   "en": "“{c}”: no threshold range was declared"},
    "vi_subgrupos": {"es": "{col}: {k} grupos ({i} con evidencia insuficiente)",
                     "en": "{col}: {k} groups ({i} with insufficient evidence)"},
    "vi_faltantes": {
        "es": "«{c}»: {f} filas sin puntuación, fuera del análisis (casos completos); "
              "{s} filas sin desenlace, fuera",
        "en": "“{c}”: {f} rows without a score, left out of the analysis (complete cases); "
              "{s} rows without an outcome, left out"},
    "vi_comparacion": {"es": "AUROC «{a}» − «{b}» = {d} ({ic}), sobre {n} filas comunes",
                       "en": "AUROC “{a}” − “{b}” = {d} ({ic}), over {n} shared rows"},
    "vi_incertidumbre": {
        "es": "{r} remuestras (bootstrap), {diseno}; intervalo al {nivel}",
        "en": "{r} resamples (bootstrap), {diseno}; {nivel} interval"},
    "vi_versiones": {"es": "núcleo {n}, catálogo de métricas {c}…, semilla {s}",
                     "en": "core {n}, metrics catalogue {c}…, seed {s}"},
}


def huecos_de(plantilla: str) -> set[str]:
    """Los `{huecos}` de una plantilla (para comprobar que las dos redacciones piden los
    mismos campos: un hueco que existe en castellano y no en inglés solo revienta cuando
    alguien pide inglés)."""
    import string  # noqa: PLC0415

    return {campo for _, campo, _, _ in string.Formatter().parse(plantilla) if campo}


def t(clave: str, idioma: str, **campos: object) -> str:
    """El rótulo de `clave` en `idioma`, con sus huecos. Una clave inexistente REVIENTA: un
    rótulo suelto escrito en el módulo dejaría el informe a medias traducido."""
    if clave not in TEXTOS:
        raise KeyError(f"{clave!r} no está en el catálogo del informe de «¿Funciona aquí?»")
    if idioma not in IDIOMAS:
        raise ValueError(f"idioma {idioma!r}: solo {', '.join(IDIOMAS)}")
    return TEXTOS[clave][idioma].format(**campos)
