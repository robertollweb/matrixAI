# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""109-C2 — el perfil de la tarea, clínico: umbrales con IC, PPV/NPV **a
prevalencia**, calibración, curva de decisión (DCA) con sus dos referencias y
subgrupos.

Criterio de terminado, literal: «tabla de umbrales con IC, calibración, DCA con
sus referencias y subgrupos; prueba numérica a mano de DCA (beneficio neto =
TP/n − FP/n · pt/(1−pt)); invertir la probabilidad intercambia sens/esp y la
prueba lo caza».

LOS NÚMEROS DE LA DCA ESTÁN ESCRITOS AQUÍ, A MANO, sobre una cohorte inventada
de diez filas. No se comparan contra otra ejecución del mismo código —eso solo
demostraría que el código es consistente consigo mismo—: se comparan contra la
aritmética hecha con lápiz, que está en el docstring de cada prueba.

Y LA INVERSIÓN, que es el control conocido-bueno de todo lo demás: MEDIDO,
son dos hechos distintos, y cada uno vale en un umbral que hay que decir.
Cambiar `p` por `1−p` dejando la misma clase positiva y el MISMO umbral `t` da
`sens'(t) = 1 − sens(1−t)`: complementa las métricas del umbral REFLEJADO, que
solo coinciden con las del mismo umbral en `t = 0,5`. El intercambio exacto
aparece cuando la inversión se escribe entera —`1−p` es la probabilidad de la
OTRA clase, y el umbral se refleja a `1−t`—: entonces sí, sens ↔ esp. Las dos
están medidas abajo con sus números **en 0,5 y en 0,15**: 0,5 es el punto fijo
del reflejo, el único umbral donde reflejar y no reflejar dan lo mismo, así que
comprobarlas solo ahí no comprobaba nada del reflejo.
"""
from __future__ import annotations

import dataclasses
import random
import unittest

from matrixai.estudio import EsquemaInvalido
from matrixai.estudio.calibracion import ajustar_recalibracion_logistica, curva_de_fiabilidad
from matrixai.estudio.incertidumbre import intervalo
from matrixai.estudio.metricas import EntradaNoMedible, Muestra, calcular, matriz_de_confusion
from matrixai.estudio.perfil_clinico import (
    _T,
    ALCANCES_DE_VALIDACION,
    BENEFICIO_NETO_NO_TRATAR,
    NPV_A_PREVALENCIA,
    PPV_A_PREVALENCIA,
    CurvaDeDecision,
    PerfilClinico,
    alcance_de_validacion,
    beneficio_neto,
    curva_de_decision,
    ficha_del_perfil,
    tabla_de_umbrales,
)
from matrixai.estudio.segmentos import analizar_segmento
from matrixai.estudio.textos import IDIOMAS, huecos_de
from matrixai.estudio.vocabulario import ETIQUETAS_DE_EVIDENCIA, TIPOS_DE_PARTICION

# ---------------------------------------------------------------------------
# La cohorte de diez filas sobre la que se hacen las cuentas a mano.
#
#   fila   1     2     3     4  |  5     6     7     8     9    10
#   y     si    si    si    si  | no    no    no    no    no    no
#   p    0.90  0.75  0.60  0.35 | 0.55  0.45  0.30  0.20  0.10  0.05
#
# n = 10, positivos = 4, negativos = 6, prevalencia observada = 0,4.
# Ningún `p` vale exactamente 0,5 ni 0,2... salvo la fila 8, que vale 0,20
# justo para que el umbral 0,20 caiga sobre ella y el `>=` de la matriz de
# confusión (105-C1) quede probado donde importa.
# ---------------------------------------------------------------------------
_Y = ("si", "si", "si", "si", "no", "no", "no", "no", "no", "no")
_P = (0.90, 0.75, 0.60, 0.35, 0.55, 0.45, 0.30, 0.20, 0.10, 0.05)


def _a_mano(y=_Y, p=_P, positiva: str = "si") -> Muestra:
    return Muestra.binaria(y, classes=("no", "si"), positive_label=positiva,
                           probabilidades=p if positiva == "si" else [1 - x for x in p])


def _cohorte_clinica_sintetica(n: int = 400, *, seed: int = 20260825
                               ) -> tuple[Muestra, list[dict]]:
    """La forma del caso clínico SINTÉTICO de `/casos`: reingreso hospitalario,
    con la receta que viaja en ese paquete —`alto: edad > 75 OR
    reingreso_previo > 0.5`, si no `bajo`— y un modelo que la aprende a medias.

    DÓNDE ESTÁ LA CITA, porque no está en este repositorio y una auditoría del
    2026-09-15 la dio por fabricada tras buscarla en `examples/readmission/`,
    que es OTRO caso (81-C6, clases `reingreso`/`no_reingreso`). `/casos` es la
    página de casos de la web pública, en el repositorio del Studio:
    `matrixaistudio/frontend/public/casos/clinico/receta.txt` dice exactamente
    `alto: edad > 75 OR reingreso_previo > 0.5` / `DEFAULT: bajo`, y es también
    el `data_recipe.txt` de su `paquete.zip`. Semilla 20260825, 400 filas y
    las cuatro columnas con sus rangos (edad 18–100, dias_ingresado 0–60,
    num_diagnosticos 0–15, reingreso_previo 0–1) son las de ese caso.
    Comprobado el 2026-09-16.

    LO QUE NO ES IGUAL, dicho para que nadie lo lea de más: las filas de aquí
    se sortean con `random.Random`, no con el generador del core, así que NO
    son las de `/casos` (prevalencia 0,6925 aquí; 0,6575 allí, 263 de 400,
    medido regenerando ese dataset con el generador del core, cuya huella
    coincide con la publicada, `1cefbf74…`). Y el «modelo» es una logística
    con ruido escrita abajo, no el que se entrenó en `/casos`.

    **Datos sintéticos: aquí no hay precisión clínica que atribuir a nadie.**
    Lo que se comprueba con esto es el CÁLCULO del perfil, no un modelo.
    """
    rng = random.Random(seed)
    filas = []
    ys, ps = [], []
    for _ in range(n):
        edad = rng.uniform(18, 100)
        dias = rng.uniform(0, 60)
        diagnosticos = rng.randint(0, 15)
        reingreso = rng.uniform(0, 1)
        y = "alto" if (edad > 75 or reingreso > 0.5) else "bajo"
        # Un "modelo" que ve las dos señales de la receta con ruido: ni
        # perfecto ni inútil, que es donde una curva de decisión dice algo.
        z = 0.06 * (edad - 70) + 3.0 * (reingreso - 0.5) + rng.gauss(0, 0.9)
        p = 1.0 / (1.0 + 2.718281828459045 ** (-z))
        filas.append({"edad": edad, "dias_ingresado": dias,
                      "num_diagnosticos": diagnosticos, "reingreso_previo": reingreso})
        ys.append(y)
        ps.append(p)
    muestra = Muestra.binaria(ys, classes=("bajo", "alto"), positive_label="alto",
                              probabilidades=ps)
    return muestra, filas


class BeneficioNetoAManoTest(unittest.TestCase):
    """`beneficio neto = TP/n − FP/n · pt/(1−pt)`, con lápiz."""

    def test_pt_0_20_da_0_30(self):
        """pt=0,20 → se trata a quien tenga p >= 0,20: las cuatro filas `si`
        (0,90 0,75 0,60 0,35) y cuatro `no` (0,55 0,45 0,30 0,20). TP=4, FP=4.
        w = 0,20/0,80 = 0,25.
        NB = 4/10 − (4/10)·0,25 = 0,4 − 0,1 = **0,30**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=4, fp=4, n=10, umbral_de_probabilidad=0.20), 0.30, places=12)

    def test_pt_0_50_da_0_20(self):
        """pt=0,50 → p >= 0,50: 0,90 0,75 0,60 (`si`) y 0,55 (`no`). TP=3, FP=1.
        w = 0,50/0,50 = 1. NB = 3/10 − (1/10)·1 = **0,20**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=3, fp=1, n=10, umbral_de_probabilidad=0.50), 0.20, places=12)

    def test_pt_0_80_da_0_10(self):
        """pt=0,80 → solo 0,90. TP=1, FP=0. w = 0,80/0,20 = 4.
        NB = 1/10 − 0·4 = **0,10**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=1, fp=0, n=10, umbral_de_probabilidad=0.80), 0.10, places=12)

    def test_pt_0_10_da_31_novenos_de_noventa(self):
        """pt=0,10 → todos menos 0,05. TP=4, FP=5. w = 0,10/0,90 = 1/9.
        NB = 2/5 − (1/2)·(1/9) = 36/90 − 5/90 = **31/90 = 0,344444…**."""
        self.assertAlmostEqual(
            beneficio_neto(tp=4, fp=5, n=10, umbral_de_probabilidad=0.10),
            31 / 90, places=12)

    def test_el_CODIGO_da_lo_MISMO_que_el_lapiz(self):
        """La otra mitad: que la curva completa, contando ella sola TP y FP
        sobre la cohorte, reproduzca los cuatro números de arriba."""
        curva = curva_de_decision(_a_mano(),
                                  umbrales_de_probabilidad=(0.10, 0.20, 0.50, 0.80))
        esperados = {0.10: (4, 5, 31 / 90), 0.20: (4, 4, 0.30),
                     0.50: (3, 1, 0.20), 0.80: (1, 0, 0.10)}
        self.assertEqual(len(curva.puntos), 4)
        for punto in curva.puntos:
            tp, fp, nb = esperados[punto.umbral_de_probabilidad]
            with self.subTest(pt=punto.umbral_de_probabilidad):
                self.assertEqual((punto.tp, punto.fp), (tp, fp))
                self.assertAlmostEqual(punto.beneficio_neto, nb, places=12)

    def test_pt_igual_a_uno_no_es_un_beneficio_enorme_sino_ninguno(self):
        """`pt/(1−pt)` divide por cero. Se rechaza; no se devuelve un número."""
        with self.assertRaises(EntradaNoMedible) as e:
            beneficio_neto(tp=1, fp=0, n=10, umbral_de_probabilidad=1.0)
        self.assertEqual(e.exception.clave, "umbral_de_probabilidad_uno")

    def test_pt_cero_es_legitimo_y_no_penaliza_los_falsos_positivos(self):
        """w = 0/(1−0) = 0: a pt=0 tratar no cuesta nada, y NB = TP/n."""
        self.assertAlmostEqual(
            beneficio_neto(tp=4, fp=6, n=10, umbral_de_probabilidad=0.0), 0.4, places=12)


class LasDosReferenciasTest(unittest.TestCase):
    """«Tratar a todos» y «a nadie», que es lo que convierte un número suelto
    en una decisión."""

    def test_a_nadie_vale_cero_por_definicion_y_no_se_mide(self):
        curva = curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.10, 0.50))
        for punto in curva.puntos:
            self.assertEqual(punto.beneficio_neto_no_tratar, 0.0)
        self.assertEqual(BENEFICIO_NETO_NO_TRATAR, 0.0)

    def test_tratar_a_todos_a_mano(self):
        """Tratar a todos: TP = los 4 positivos, FP = los 6 negativos.
        pt=0,20 → 4/10 − (6/10)·0,25 = 0,4 − 0,15 = **0,25**.
        pt=0,50 → 4/10 − (6/10)·1    = 0,4 − 0,60 = **−0,20**.
        pt=0,10 → 2/5 − (3/5)·(1/9)  = 6/15 − 1/15 = **1/3**."""
        curva = curva_de_decision(_a_mano(),
                                  umbrales_de_probabilidad=(0.10, 0.20, 0.50))
        esperados = {0.10: 1 / 3, 0.20: 0.25, 0.50: -0.20}
        for punto in curva.puntos:
            with self.subTest(pt=punto.umbral_de_probabilidad):
                self.assertAlmostEqual(punto.beneficio_neto_tratar_a_todos,
                                       esperados[punto.umbral_de_probabilidad],
                                       places=12)

    def test_este_modelo_supera_a_las_dos_donde_se_esperaba(self):
        curva = curva_de_decision(_a_mano(),
                                  umbrales_de_probabilidad=(0.10, 0.20, 0.50, 0.80))
        self.assertTrue(all(p.supera_las_referencias for p in curva.puntos))
        self.assertEqual(curva.umbrales_sin_ventaja, ())
        self.assertTrue(curva.supera_en_algun_umbral)

    def test_empatar_con_tratar_a_todos_NO_es_superarlo(self):
        """pt=0,05: el modelo trata a los diez (todos tienen p >= 0,05), así
        que TP=4 y FP=6 son los MISMOS que los de «tratar a todos» y los dos
        beneficios netos valen 0,4 − 0,6·(1/19) = **0,368421…**. Aportar lo
        mismo que tratar a todo el mundo no es aportar nada."""
        curva = curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.05,))
        punto = curva.puntos[0]
        self.assertEqual((punto.tp, punto.fp), (4, 6))
        self.assertAlmostEqual(punto.beneficio_neto, 0.4 - 0.6 / 19, places=12)
        self.assertAlmostEqual(punto.beneficio_neto_tratar_a_todos, 0.4 - 0.6 / 19,
                               places=12)
        self.assertFalse(punto.supera_las_referencias)
        self.assertEqual(curva.umbrales_sin_ventaja, (0.05,))
        self.assertFalse(curva.supera_en_algun_umbral)

    def test_un_modelo_al_reves_no_supera_a_tratar_a_todos_y_la_curva_lo_dice(self):
        """La probabilidad invertida, a pt=0,20: se trata a quien tenga
        1−p >= 0,20, o sea p <= 0,80 → tres `si` (0,75 0,60 0,35) y los seis
        `no`. TP=3, FP=6. NB = 3/10 − (6/10)·0,25 = 0,30 − 0,15 = **0,15**,
        por debajo del 0,25 de tratar a todos."""
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                    probabilidades=[1 - p for p in _P])
        curva = curva_de_decision(invertida, umbrales_de_probabilidad=(0.20,))
        punto = curva.puntos[0]
        self.assertEqual((punto.tp, punto.fp), (3, 6))
        self.assertAlmostEqual(punto.beneficio_neto, 0.15, places=12)
        self.assertAlmostEqual(punto.beneficio_neto_tratar_a_todos, 0.25, places=12)
        self.assertGreater(punto.beneficio_neto, punto.beneficio_neto_no_tratar)
        self.assertFalse(punto.supera_las_referencias)


class InversionDeLaProbabilidadTest(unittest.TestCase):
    """El control conocido-bueno. Si esto no casa, ningún número de arriba
    vale nada."""

    def test_en_0_5_invertir_p_con_el_MISMO_umbral_COMPLEMENTA_sens_y_esp(self):
        """Original en t=0,5: TP=3 (0,90 0,75 0,60), FN=1 (0,35), TN=5, FP=1
        (0,55) → sens = 3/4 = **0,75**, esp = 5/6 = **0,8333…**.
        Invertida (p → 1−p, misma clase positiva, mismo t=0,5): se predice `si`
        cuando 1−p >= 0,5, o sea p <= 0,5 → 0,35 (`si`) y 0,45 0,30 0,20 0,10
        0,05 (`no`) → TP=1, FN=3, FP=5, TN=1 → sens = 1/4 = **0,25** = 1−0,75 y
        esp = 1/6 = **0,1666…** = 1−0,8333.
        Eso es el COMPLEMENTO, no el intercambio: medido, no supuesto. Y SOLO
        EN 0,5: lo que complementa es el umbral reflejado, 1−t, que aquí es el
        mismo. Fuera de 0,5, ver la prueba siguiente."""
        directa = _a_mano()
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                    probabilidades=[1 - p for p in _P])
        sens = calcular("sensitivity", directa, umbral=0.5).value
        esp = calcular("specificity", directa, umbral=0.5).value
        self.assertAlmostEqual(sens, 0.75, places=12)
        self.assertAlmostEqual(esp, 5 / 6, places=12)

        sens_inv = calcular("sensitivity", invertida, umbral=0.5).value
        esp_inv = calcular("specificity", invertida, umbral=0.5).value
        self.assertAlmostEqual(sens_inv, 0.25, places=12)
        self.assertAlmostEqual(esp_inv, 1 / 6, places=12)
        self.assertAlmostEqual(sens_inv, 1 - sens, places=12)
        self.assertAlmostEqual(esp_inv, 1 - esp, places=12)
        # Y NO es un intercambio: decirlo así sería heredar una frase sin medirla.
        self.assertNotAlmostEqual(sens_inv, esp, places=6)
        self.assertNotAlmostEqual(esp_inv, sens, places=6)

    def test_fuera_de_0_5_el_MISMO_umbral_NO_complementa_y_el_REFLEJADO_si(self):
        """t=0,15, que no cae sobre ningún `p` ni sobre ningún `1−p` (por eso
        0,15: en 0,40 la fila 3 cae justo en 1−t=0,60 y el empate rompe la
        igualdad por una fila).

        Directa en t=0,15: los cuatro `si` tienen p >= 0,15 → TP=4, FN=0 →
        sens = **1,0**; de los `no`, 0,55 0,45 0,30 0,20 son FP y 0,10 0,05 TN
        → esp = 2/6 = **0,3333…**.
        Directa en 1−t=0,85: solo 0,90 → TP=1, FN=3 → sens = **0,25**; ningún
        `no` llega → TN=6 → esp = **1,0**.
        Invertida (p → 1−p, misma clase `si`, MISMO t=0,15): `si` cuando
        1−p >= 0,15 ⟺ p <= 0,85 → 0,75 0,60 0,35 son TP y 0,90 FN → sens' =
        **0,75**; los seis `no` tienen p <= 0,85 → FP=6, TN=0 → esp' = **0,0**.

        Complemento en el MISMO umbral: 1−1,0 = 0,0 y 1−0,333 = 0,667. NO
        casa. Complemento en el REFLEJADO: 1−0,25 = 0,75 y 1−1,0 = 0,0. Casa."""
        directa = _a_mano()
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                    probabilidades=[1 - p for p in _P])
        sens_t = calcular("sensitivity", directa, umbral=0.15).value
        esp_t = calcular("specificity", directa, umbral=0.15).value
        sens_reflejado = calcular("sensitivity", directa, umbral=0.85).value
        esp_reflejado = calcular("specificity", directa, umbral=0.85).value
        sens_inv = calcular("sensitivity", invertida, umbral=0.15).value
        esp_inv = calcular("specificity", invertida, umbral=0.15).value
        self.assertAlmostEqual(sens_t, 1.0, places=12)
        self.assertAlmostEqual(esp_t, 1 / 3, places=12)
        self.assertAlmostEqual(sens_reflejado, 0.25, places=12)
        self.assertAlmostEqual(esp_reflejado, 1.0, places=12)
        self.assertAlmostEqual(sens_inv, 0.75, places=12)
        self.assertAlmostEqual(esp_inv, 0.0, places=12)
        # lo que es verdad: el complemento del umbral reflejado
        self.assertAlmostEqual(sens_inv, 1 - sens_reflejado, places=12)
        self.assertAlmostEqual(esp_inv, 1 - esp_reflejado, places=12)
        # lo que el docstring decía y es falso fuera de 0,5
        self.assertNotAlmostEqual(sens_inv, 1 - sens_t, places=6)
        self.assertNotAlmostEqual(esp_inv, 1 - esp_t, places=6)

    def test_invertir_la_probabilidad_ENTERA_INTERCAMBIA_sens_y_esp(self):
        """La inversión escrita entera: `1−p` es la probabilidad de la OTRA
        clase, así que la clase positiva pasa a ser `no` y el umbral se refleja
        a 1−t. Es el MISMO clasificador visto desde el otro lado, y por eso la
        sensibilidad de uno es la especificidad del otro.

        Con t=0,5: la invertida predice `no` cuando 1−p >= 0,5 ⟺ p <= 0,5 →
        0,35 (`si`) y los cinco `no` por debajo de 0,5. TP(no)=5, FN(no)=1
        (0,55), TN(no)=3 (0,90 0,75 0,60), FP(no)=1 (0,35) →
        sens_inv = 5/6 = esp_directa **0,8333…** y esp_inv = 3/4 =
        sens_directa **0,75**. INTERCAMBIO exacto."""
        directa = _a_mano()
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                                    probabilidades=[1 - p for p in _P])
        sens = calcular("sensitivity", directa, umbral=0.5).value
        esp = calcular("specificity", directa, umbral=0.5).value
        sens_inv = calcular("sensitivity", invertida, umbral=0.5).value
        esp_inv = calcular("specificity", invertida, umbral=0.5).value
        self.assertAlmostEqual(sens_inv, esp, places=12)
        self.assertAlmostEqual(esp_inv, sens, places=12)
        self.assertAlmostEqual(sens_inv, 5 / 6, places=12)
        self.assertAlmostEqual(esp_inv, 0.75, places=12)

    def test_fuera_de_0_5_el_intercambio_EXIGE_reflejar_el_umbral(self):
        """La prueba de arriba usa t=0,5 = 1−t, así que no distingue reflejar
        de no reflejar. Aquí, t=0,15: directa sens = 1,0 y esp = 1/3 (ver la
        prueba del complemento).

        Inversión entera REFLEJANDO a 0,85: predice `no` cuando 1−p >= 0,85
        ⟺ p <= 0,15 → 0,10 y 0,05 → TP(no)=2, FN(no)=4 → sens_inv = 2/6 =
        **0,3333…** = esp directa; ningún `si` tiene p <= 0,15 → TN(no)=4 →
        esp_inv = **1,0** = sens directa. INTERCAMBIO.

        Inversión entera SIN reflejar (0,15): predice `no` cuando p <= 0,85 →
        los seis `no` → sens_inv = **1,0**; y 0,75 0,60 0,35 → FP(no)=3, 0,90
        → TN(no)=1 → esp_inv = **0,25**. NO es el intercambio (pediría 1/3 y
        1,0)."""
        directa = _a_mano()
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                                    probabilidades=[1 - p for p in _P])
        sens = calcular("sensitivity", directa, umbral=0.15).value
        esp = calcular("specificity", directa, umbral=0.15).value
        reflejada = (calcular("sensitivity", invertida, umbral=0.85).value,
                     calcular("specificity", invertida, umbral=0.85).value)
        sin_reflejar = (calcular("sensitivity", invertida, umbral=0.15).value,
                        calcular("specificity", invertida, umbral=0.15).value)
        self.assertAlmostEqual(reflejada[0], 1 / 3, places=12)
        self.assertAlmostEqual(reflejada[1], 1.0, places=12)
        self.assertAlmostEqual(reflejada[0], esp, places=12)
        self.assertAlmostEqual(reflejada[1], sens, places=12)
        self.assertAlmostEqual(sin_reflejar[0], 1.0, places=12)
        self.assertAlmostEqual(sin_reflejar[1], 0.25, places=12)
        self.assertNotAlmostEqual(sin_reflejar[0], esp, places=6)

    def test_el_intercambio_exacto_se_ROMPE_con_un_empate_en_el_umbral(self):
        """Por qué la cohorte de arriba no tiene ningún `p` igual a 0,5, y no
        es casualidad: la matriz de confusión decide con `>=`, así que la fila
        que cae JUSTO en el umbral se cuenta positiva en los dos lados de la
        inversión y el intercambio deja de ser exacto por esa fila.

        Cohorte igual pero con la fila 4 en 0,50 en vez de 0,35: la directa da
        sens = 4/4 = 1,0 y esp = 5/6; la invertida da sens = 5/6 (coincide) y
        esp = 3/4, que NO es 1,0. Está escrito para que nadie 'simplifique' la
        cohorte metiendo un 0,5."""
        p_con_empate = (0.90, 0.75, 0.60, 0.50, 0.55, 0.45, 0.30, 0.20, 0.10, 0.05)
        directa = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                  probabilidades=p_con_empate)
        invertida = Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                                    probabilidades=[1 - p for p in p_con_empate])
        self.assertAlmostEqual(calcular("sensitivity", directa, umbral=0.5).value, 1.0)
        self.assertAlmostEqual(calcular("specificity", directa, umbral=0.5).value, 5 / 6)
        self.assertAlmostEqual(calcular("sensitivity", invertida, umbral=0.5).value, 5 / 6)
        self.assertAlmostEqual(calcular("specificity", invertida, umbral=0.5).value, 0.75)

    def test_la_TABLA_del_perfil_tambien_intercambia_al_invertir(self):
        """No basta con que lo hagan las fórmulas de C1: lo que publica este
        corte es la tabla, y es la tabla la que tiene que cazarlo."""
        comun = dict(umbrales=(0.5,), diseno="iid",
                     estimando="fixed_model_on_population", semilla=1, remuestras=50)
        directa = tabla_de_umbrales(_a_mano(), **comun).filas[0]
        invertida = tabla_de_umbrales(
            Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                            probabilidades=[1 - p for p in _P]), **comun).filas[0]
        self.assertAlmostEqual(invertida.sensibilidad.value, directa.especificidad.value,
                               places=12)
        self.assertAlmostEqual(invertida.especificidad.value, directa.sensibilidad.value,
                               places=12)
        self.assertEqual((invertida.tp, invertida.fn), (directa.tn, directa.fp))

    def test_la_TABLA_intercambia_fuera_de_0_5_con_el_umbral_reflejado(self):
        """La prueba de arriba es en 0,5, el punto fijo del reflejo: una tabla
        que calculase sens/esp en `1−umbral` en vez de en `umbral` la pasaría
        entera. Aquí los números van a mano (ver
        `test_fuera_de_0_5_el_intercambio_EXIGE_reflejar_el_umbral`): directa
        en 0,15 → sens 1,0 y esp 1/3; invertida entera en 0,85 → sens 1/3 y
        esp 1,0."""
        comun = dict(diseno="iid", estimando="fixed_model_on_population", semilla=1,
                     remuestras=50)
        directa = tabla_de_umbrales(_a_mano(), umbrales=(0.15,), **comun).filas[0]
        invertida = tabla_de_umbrales(
            Muestra.binaria(_Y, classes=("no", "si"), positive_label="no",
                            probabilidades=[1 - p for p in _P]),
            umbrales=(0.85,), **comun).filas[0]
        self.assertAlmostEqual(directa.sensibilidad.value, 1.0, places=12)
        self.assertAlmostEqual(directa.especificidad.value, 1 / 3, places=12)
        self.assertAlmostEqual(invertida.sensibilidad.value, 1 / 3, places=12)
        self.assertAlmostEqual(invertida.especificidad.value, 1.0, places=12)
        self.assertEqual((directa.tp, directa.fp, directa.tn, directa.fn), (4, 4, 2, 0))
        self.assertEqual((invertida.tp, invertida.fn), (directa.tn, directa.fp))


class PrevalenciaTest(unittest.TestCase):
    """Lo que más importa de este corte: PPV y NPV dependen de la prevalencia
    y sens/esp no."""

    def test_sin_prevalencia_declarada_NO_se_publican_ppv_ni_npv(self):
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  remuestras=50)
        fila = tabla.filas[0]
        self.assertFalse(tabla.publica_ppv_y_npv)
        self.assertIsNone(tabla.prevalencia_declarada)
        for valor in (fila.ppv, fila.npv):
            with self.subTest(metric_id=valor.metric_id):
                self.assertIsNone(valor.value)
                self.assertIsNotNone(valor.undefined_reason)
                self.assertEqual(set(valor.undefined_reason), set(IDIOMAS))

    def test_y_NO_se_callan_las_que_no_dependen_de_ella(self):
        """La otra mitad: un aserto negativo lo pasaría un perfil que no
        publicara nada. Sensibilidad y especificidad SÍ salen, con IC, porque
        no dependen de la prevalencia."""
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  remuestras=200)
        fila = tabla.filas[0]
        self.assertAlmostEqual(fila.sensibilidad.value, 0.75, places=12)
        self.assertAlmostEqual(fila.especificidad.value, 5 / 6, places=12)
        for valor in (fila.sensibilidad, fila.especificidad):
            with self.subTest(metric_id=valor.metric_id):
                self.assertIsNotNone(valor.uncertainty)
                self.assertIsNotNone(valor.uncertainty["ci_low"])
                self.assertLessEqual(valor.uncertainty["ci_low"], valor.value)
                self.assertGreaterEqual(valor.uncertainty["ci_high"], valor.value)

    def test_el_ppv_a_prevalencia_2_por_ciento_a_mano(self):
        """sens = 3/4, esp = 5/6 en t=0,5. A una prevalencia declarada del 2 %:

          PPV = sens·pr / (sens·pr + (1−esp)·(1−pr))
              = 0,75·0,02 / (0,75·0,02 + (1/6)·0,98)
              = 0,015 / 0,178333… = **0,0841121…**

          NPV = esp·(1−pr) / (esp·(1−pr) + (1−sens)·pr)
              = (5/6)·0,98 / ((5/6)·0,98 + 0,25·0,02)
              = 0,816666… / 0,821666… = **0,9939148…**

        Y el PPV OBSERVADO en esta muestra vale 3/4 = 0,75. Nueve veces el
        transportado, con la misma aritmética impecable: eso es lo que se
        estaría publicando si la prevalencia no viajara con el número."""
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  prevalencia=0.02, remuestras=50)
        fila = tabla.filas[0]
        self.assertAlmostEqual(fila.ppv.value, 0.015 / (0.015 + (1 / 6) * 0.98),
                               places=12)
        self.assertAlmostEqual(fila.ppv.value, 0.08411214953271028, places=12)
        self.assertAlmostEqual(fila.npv.value, 0.9939148073022313, places=12)
        self.assertAlmostEqual(calcular("ppv", _a_mano(), umbral=0.5).value, 0.75,
                               places=12)
        self.assertEqual(fila.ppv.metric_id, PPV_A_PREVALENCIA)
        self.assertEqual(fila.npv.metric_id, NPV_A_PREVALENCIA)

    def test_la_prevalencia_de_la_muestra_viaja_al_lado_de_la_declarada(self):
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  prevalencia=0.02, remuestras=50)
        self.assertAlmostEqual(tabla.prevalencia_declarada, 0.02)
        self.assertAlmostEqual(tabla.prevalencia_observada, 0.4)

    def test_el_transportado_trae_su_intervalo(self):
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=3,
                                  prevalencia=0.02, remuestras=300)
        ppv = tabla.filas[0].ppv
        self.assertIsNotNone(ppv.uncertainty)
        self.assertEqual(ppv.uncertainty["metric_id"], PPV_A_PREVALENCIA)
        self.assertIsNotNone(ppv.uncertainty["ci_low"])
        self.assertLessEqual(ppv.uncertainty["ci_low"], ppv.value)
        self.assertGreaterEqual(ppv.uncertainty["ci_high"], ppv.value)

    def test_a_prevalencia_igual_a_la_de_la_muestra_el_transporte_da_el_observado(self):
        """Control del instrumento: con pr = la prevalencia de la muestra, la
        fórmula de Bayes tiene que devolver el PPV observado (0,75) y el NPV
        observado (5/6). Si no lo hiciera, el transporte estaría mal escrito y
        los números de arriba tampoco valdrían."""
        tabla = tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  prevalencia=0.4, remuestras=50)
        fila = tabla.filas[0]
        self.assertAlmostEqual(fila.ppv.value,
                               calcular("ppv", _a_mano(), umbral=0.5).value, places=12)
        self.assertAlmostEqual(fila.npv.value,
                               calcular("npv", _a_mano(), umbral=0.5).value, places=12)

    def test_una_prevalencia_fuera_de_cero_uno_se_rechaza(self):
        with self.assertRaises(EsquemaInvalido):
            tabla_de_umbrales(_a_mano(), umbrales=(0.5,), diseno="iid",
                              estimando="fixed_model_on_population", semilla=1,
                              prevalencia=1.5, remuestras=10)


class UmbralesYCableadoTest(unittest.TestCase):
    def test_los_umbrales_los_declara_quien_decide_y_no_hay_omision(self):
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(_a_mano(), umbrales_de_probabilidad=())
        self.assertEqual(e.exception.clave, "umbrales_no_declarados")
        with self.assertRaises(EntradaNoMedible):
            tabla_de_umbrales(_a_mano(), umbrales=(), diseno="iid",
                              estimando="fixed_model_on_population", semilla=1)

    def test_un_umbral_repetido_no_son_dos_medidas(self):
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.2, 0.2))
        self.assertEqual(e.exception.clave, "umbral_de_probabilidad_repetido")

    def test_una_muestra_con_la_clase_YA_decidida_se_rechaza(self):
        """Sobre etiquetas fijas la curva saldría igual en todos los umbrales:
        una curva plana creíble y falsa. Es cableado, así que excepción."""
        con_etiquetas = Muestra.binaria(
            _Y, classes=("no", "si"), positive_label="si", probabilidades=_P,
            predicciones=["si" if p >= 0.5 else "no" for p in _P])
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(con_etiquetas, umbrales_de_probabilidad=(0.2, 0.5))
        self.assertEqual(e.exception.clave, "dca_con_etiquetas_declaradas")

    def test_la_TABLA_de_umbrales_tambien_rechaza_la_clase_YA_decidida(self):
        """La otra guarda con la MISMA clave, la de `tabla_de_umbrales`. La
        prueba de arriba solo llama a la curva y no podía cubrirla: medido el
        2026-09-16, quitar esta guarda dejaba todas las pruebas verdes y la
        tabla publicaba en 0,1 0,2 0,5 0,8 y 0,9 la MISMA fila (3/1/5/1, sens
        0,75, esp 0,833), porque `matriz_de_confusion` da prioridad a
        `predictions` sobre el umbral. Plana, creíble y falsa, en la mitad del
        corte que no es la curva.

        El control de que la guarda protege algo: la misma muestra SIN la clase
        decidida da filas distintas en 0,2 y en 0,8."""
        con_etiquetas = Muestra.binaria(
            _Y, classes=("no", "si"), positive_label="si", probabilidades=_P,
            predicciones=["si" if p >= 0.5 else "no" for p in _P])
        comun = dict(umbrales=(0.2, 0.8), diseno="iid",
                     estimando="fixed_model_on_population", semilla=1, remuestras=20)
        with self.assertRaises(EntradaNoMedible) as e:
            tabla_de_umbrales(con_etiquetas, **comun)
        self.assertEqual(e.exception.clave, "dca_con_etiquetas_declaradas")

        filas = tabla_de_umbrales(_a_mano(), **comun).filas
        self.assertNotEqual((filas[0].tp, filas[0].fp), (filas[1].tp, filas[1].fp))

    def test_una_muestra_de_puntuaciones_crudas_no_tiene_curva_de_decision(self):
        cruda = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                puntuaciones=[p * 10 - 5 for p in _P])
        with self.assertRaises(EntradaNoMedible) as e:
            curva_de_decision(cruda, umbrales_de_probabilidad=(0.2,))
        self.assertEqual(e.exception.clave, "dca_sin_probabilidad_calibrada")

    def test_con_probabilidad_Y_puntuacion_cruda_la_curva_corta_sobre_la_PROBABILIDAD(self):
        """Lo que se protege no ha cambiado: un `pt` de probabilidad no se compara
        NUNCA contra un logit. Hasta el 2026-09-17 se protegía rechazando la
        muestra, porque `puntuacion_del_positivo` daba prioridad a `scores`
        (medido el 2026-09-14). Desde que la probabilidad manda en la fuente
        (`test_c105_c1_una_muestra_una_escala.py`), la muestra con las dos cosas
        da EXACTAMENTE la curva de la de solo probabilidades."""
        crudas = [p * 10 - 5 for p in _P]
        ambigua = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                  probabilidades=_P, puntuaciones=crudas)
        solo = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                               probabilidades=_P)
        self.assertEqual(ambigua.escala_de_decision, "calibrated_probability")
        umbrales = (0.2, 0.5, 0.58)
        self.assertEqual(curva_de_decision(ambigua, umbrales_de_probabilidad=umbrales).puntos,
                         curva_de_decision(solo, umbrales_de_probabilidad=umbrales).puntos)

    def test_con_probabilidad_Y_puntuacion_cruda_la_tabla_exige_umbral_de_PROBABILIDAD(self):
        """La tabla corta en la escala declarada: con probabilidades, un umbral
        fuera de [0,1] se rechaza aunque la muestra traiga también `scores`
        (antes se aceptaba y cortaba sobre el logit), y uno válido da la misma
        fila que la muestra de solo probabilidades."""
        comun = dict(diseno="iid", estimando="fixed_model_on_population", semilla=1,
                     remuestras=20)
        ambigua = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                  probabilidades=_P, puntuaciones=[p * 10 - 5 for p in _P])
        solo = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                               probabilidades=_P)
        with self.assertRaises(EsquemaInvalido):
            tabla_de_umbrales(ambigua, umbrales=(3.0,), **comun)
        a = tabla_de_umbrales(ambigua, umbrales=(0.58,), **comun).filas[0]
        b = tabla_de_umbrales(solo, umbrales=(0.58,), **comun).filas[0]
        self.assertEqual((a.tp, a.fp, a.tn, a.fn), (b.tp, b.fp, b.tn, b.fn))

    def test_la_tabla_sobre_puntuaciones_crudas_admite_umbrales_fuera_de_0_1(self):
        """La otra mitad: con puntuaciones crudas el umbral NO vive en [0,1], y
        exigirlo ahí rechazaría un umbral legítimo. Sin probabilidades, el corte
        es sobre `scores`, que es lo que mira la matriz de confusión."""
        cruda = Muestra.binaria(_Y, classes=("no", "si"), positive_label="si",
                                puntuaciones=[p * 10 - 5 for p in _P])
        tabla = tabla_de_umbrales(cruda, umbrales=(0.0,), diseno="iid",
                                  estimando="fixed_model_on_population", semilla=1,
                                  remuestras=20)
        self.assertAlmostEqual(tabla.filas[0].sensibilidad.value, 0.75, places=12)
        with self.assertRaises(EsquemaInvalido):
            tabla_de_umbrales(_a_mano(), umbrales=(3.0,), diseno="iid",
                              estimando="fixed_model_on_population", semilla=1,
                              remuestras=20)

    def test_cero_filas_no_dan_una_curva_en_cero(self):
        vacia = Muestra(task="binary_classification", y_true=(), classes=("no", "si"),
                        positive_label="si", probabilities=())
        curva = curva_de_decision(vacia, umbrales_de_probabilidad=(0.2,))
        self.assertEqual(curva.puntos, ())
        self.assertIsNotNone(curva.undefined_reason)
        self.assertIsNone(curva.prevalencia_observada)

    def test_un_intervalo_derivado_sin_su_punto_se_rechaza(self):
        """`formula=` sin `valor=` mandaría a `calcular()` a buscar en el
        catálogo un `metric_id` que por definición no está."""
        with self.assertRaises(EntradaNoMedible) as e:
            intervalo(PPV_A_PREVALENCIA, _a_mano(), diseno="iid",
                      estimando="fixed_model_on_population", semilla=1,
                      remuestras=10, formula=lambda m, u: 0.5)
        self.assertEqual(e.exception.clave, "intervalo_derivado_sin_punto")


class AlcanceDeValidacionTest(unittest.TestCase):
    def test_lo_normal_es_interna_unicamente(self):
        self.assertEqual(alcance_de_validacion(evidencia="independent_test",
                                               diseno="iid"), "internal_only")
        self.assertEqual(alcance_de_validacion(evidencia="development_estimate",
                                               diseno="iid"), "internal_only")
        self.assertEqual(alcance_de_validacion(evidencia="not_evaluated",
                                               diseno="temporal"), "internal_only")

    def test_temporal_y_externa_solo_cuando_existe_la_cohorte(self):
        self.assertEqual(alcance_de_validacion(evidencia="independent_test",
                                               diseno="temporal"), "temporal")
        self.assertEqual(alcance_de_validacion(evidencia="external_validation",
                                               diseno="iid"), "external")

    def test_todos_los_alcances_estan_en_el_vocabulario(self):
        for evidencia in ("independent_test", "external_validation",
                          "development_estimate", "repeated_test_use",
                          "test_used_for_development", "not_evaluated"):
            for diseno in ("iid", "temporal", "groups", "groups_and_time"):
                with self.subTest(evidencia=evidencia, diseno=diseno):
                    self.assertIn(alcance_de_validacion(evidencia=evidencia,
                                                        diseno=diseno),
                                  ALCANCES_DE_VALIDACION)


def _perfil(**cambios):
    muestra = _a_mano()
    base = dict(
        perfil_id="perfil-a-mano", evidencia="independent_test", diseno="iid",
        datos_sinteticos=True,
        tabla=tabla_de_umbrales(muestra, umbrales=(0.5,), diseno="iid",
                                estimando="fixed_model_on_population", semilla=1,
                                remuestras=50),
        curva=curva_de_decision(muestra, umbrales_de_probabilidad=(0.2, 0.5)))
    base.update(cambios)
    return PerfilClinico(**base)


class SegmentosPredefinidosTest(unittest.TestCase):
    """«Sin declaración no hay subgrupos confirmatorios» (decisión 5 del 109),
    hecho cumplir en la construcción y no en una nota."""

    def _segmento(self, predefinido: bool):
        return analizar_segmento("sensitivity", _a_mano(), [0, 1, 4, 5],
                                 segmento_id="mayores_de_75", predefinido=predefinido,
                                 umbral=0.5)

    def test_sin_equipo_un_segmento_no_puede_viajar_como_predefinido(self):
        with self.assertRaises(EsquemaInvalido) as e:
            _perfil(segmentos=(self._segmento(True),))
        self.assertEqual(e.exception.clave, "segmento_confirmatorio_sin_equipo")

    def test_sin_equipo_los_exploratorios_SI_viajan_marcados(self):
        """La otra mitad: prohibirlo todo dejaría el perfil mudo. Lo que no se
        puede es presentarlo como predefinido."""
        perfil = _perfil(segmentos=(self._segmento(False),))
        self.assertEqual(perfil.segmentos_exploratorios, ("mayores_de_75",))
        self.assertIsNone(perfil.segmentos_predefinidos_por)

    def test_con_equipo_declarado_si_se_puede(self):
        perfil = _perfil(segmentos=(self._segmento(True),),
                         segmentos_predefinidos_por="comité de calidad del hospital X")
        self.assertEqual(perfil.segmentos_exploratorios, ())

    def test_la_regla_vale_TAMBIEN_para_las_curvas_por_segmento(self):
        curva = curva_de_decision(_a_mano(), umbrales_de_probabilidad=(0.2,),
                                  segmento_id="mayores_de_75", predefinido=True)
        with self.assertRaises(EsquemaInvalido) as e:
            _perfil(curvas_por_segmento=(curva,))
        self.assertEqual(e.exception.clave, "segmento_confirmatorio_sin_equipo")

    def test_una_curva_de_segmento_sin_id_no_es_un_segmento(self):
        with self.assertRaises(EsquemaInvalido):
            CurvaDeDecision(puntos=(), n=0, segmento_id="x", predefinido=None,
                            undefined_reason={"es": "a", "en": "b"})


class TablaYCurvaDeLaMismaMuestraTest(unittest.TestCase):
    """La ficha escribe dos veces la prevalencia «observada en esta muestra»,
    una de la tabla y otra de la curva. Medido el 2026-09-16: con tabla y curva
    de cohortes distintas el perfil se sellaba y la ficha decía 0.4000 arriba y
    0.5000 encima de la curva, las dos «de esta muestra»."""

    def test_una_curva_de_OTRO_n_no_hace_perfil_con_esta_tabla(self):
        otra = curva_de_decision(_recorte(_a_mano(), [0, 1, 4, 5]),
                                 umbrales_de_probabilidad=(0.2, 0.5))
        with self.assertRaises(EsquemaInvalido) as e:
            _perfil(curva=otra)
        self.assertEqual(e.exception.clave, "filas_desalineadas")

    def test_el_mismo_n_con_OTRA_prevalencia_tampoco(self):
        """Diez filas también, pero la fila 5 pasa a `si`: prevalencia 0,5 y no
        0,4. El `n` solo no lo habría cazado."""
        y_otra = tuple("si" if i == 4 else y for i, y in enumerate(_Y))
        otra = curva_de_decision(_a_mano(y=y_otra), umbrales_de_probabilidad=(0.2, 0.5))
        self.assertEqual(otra.n, 10)
        self.assertAlmostEqual(otra.prevalencia_observada, 0.5)
        with self.assertRaises(EsquemaInvalido) as e:
            _perfil(curva=otra)
        self.assertEqual(e.exception.clave, "fuera_de_rango")

    def test_la_misma_muestra_en_DOS_objetos_si_hace_perfil(self):
        """La otra mitad: lo que se compara es lo que el documento afirma de la
        muestra, no la identidad del objeto — el productor del Studio construye
        la curva sobre una copia reexpandida de la misma muestra."""
        perfil = _perfil(curva=curva_de_decision(_a_mano(),
                                                 umbrales_de_probabilidad=(0.2, 0.5)))
        self.assertEqual(perfil.tabla.prevalencia_observada,
                         perfil.curva.prevalencia_observada)


class MapasLibresDelPerfilTest(unittest.TestCase):
    """`recalibracion` y `politica_de_faltantes` no tienen esquema propio —sus
    productores les dan formas distintas— pero tienen que ser un mapa, no venir
    vacíos, y salir en la ficha."""

    def test_un_objeto_que_no_es_un_mapa_se_rechaza_AL_CONSTRUIR(self):
        """Medido el 2026-09-16: pasar el `RecalibracionLogistica` en vez de su
        `a_json()` se aceptaba, y reventaba después con un `TypeError` dentro
        de `a_json()`, lejos de quien lo pasó."""
        recal = ajustar_recalibracion_logistica(_a_mano())
        for campo in ("recalibracion", "politica_de_faltantes"):
            for valor in (recal, ["edad", "mediana"], "mediana"):
                with self.subTest(campo=campo, valor=type(valor).__name__):
                    with self.assertRaises(EsquemaInvalido) as e:
                        _perfil(**{campo: valor})
                    self.assertEqual(e.exception.clave, "no_es_mapa")

    def test_un_mapa_VACIO_no_es_ni_una_politica_ni_su_ausencia(self):
        """`{}` salía `null` en el JSON y una sección muda en la ficha: dos
        documentos diciendo cosas distintas del mismo campo. La ausencia se
        dice con `None`."""
        for campo in ("recalibracion", "politica_de_faltantes"):
            with self.subTest(campo=campo):
                with self.assertRaises(EsquemaInvalido) as e:
                    _perfil(**{campo: {}})
                self.assertEqual(e.exception.clave, "falta_campo")

    def test_la_recalibracion_sale_en_la_ficha_dentro_de_calibracion(self):
        """Se sellaba en el JSON y la ficha no la nombraba nunca."""
        recal = ajustar_recalibracion_logistica(_a_mano()).a_json()
        perfil = _perfil(recalibracion=recal)
        self.assertEqual(perfil.a_json()["recalibracion"], recal)
        for locale, calibracion, dca, rotulo in (
                ("es", "## Calibración", "## Beneficio neto",
                 "Recalibración que consta en el perfil"),
                ("en", "## Calibration", "## Net benefit",
                 "Recalibration on record in this profile")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(perfil, locale=locale)
                seccion = ficha[ficha.index(calibracion):ficha.index(dca)]
                self.assertIn(rotulo, seccion)
                self.assertIn("- **metodo**: logistico_completo", seccion)
                self.assertIn(f"- **a**: {recal['a']!r}", seccion)
                self.assertIn(f"- **b**: {recal['b']!r}", seccion)
                self.assertIn("- **convergio**: true", seccion)
                self.assertIn("- **undefined_reason**: null", seccion)

    def test_sin_recalibracion_la_ficha_dice_que_NO_CONSTA_y_con_ella_no(self):
        """«No consta», no «no se hizo»: `None` no dice cuál de las dos."""
        con = _perfil(recalibracion=ajustar_recalibracion_logistica(_a_mano()).a_json())
        for locale, frase in (("es", "No consta ninguna recalibración"),
                              ("en", "No recalibration is on record")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))
                self.assertNotIn(frase, ficha_del_perfil(con, locale=locale))

    def test_la_politica_con_la_forma_del_productor_del_Studio_sale_en_JSON(self):
        """La forma que de verdad llega (`perfil_clinico_del_estudio.py`): una
        lista de objetos. Con el `repr` de Python salía `'columna'` y `False`
        en un documento; en JSON, como la escribe el expediente del 109-C3."""
        politica = {"fuente": "medida", "filas_de_train_efectivas": 320,
                    "columnas": [{"columna": "edad", "proporcion_faltante": 0.1,
                                  "admite_nativo": False}]}
        ficha = ficha_del_perfil(_perfil(politica_de_faltantes=politica), locale="en")
        self.assertIn('- **columnas**: [{"admite_nativo": false, "columna": "edad", '
                      '"proporcion_faltante": 0.1}]', ficha)
        self.assertIn("- **filas_de_train_efectivas**: 320", ficha)
        self.assertIn("- **fuente**: medida", ficha)
        self.assertNotIn("'columna'", ficha)
        self.assertNotIn("False", ficha)

    def test_un_motivo_bilingue_dentro_del_mapa_sale_en_el_idioma_de_la_ficha(self):
        """Una recalibración que no se pudo ajustar (una sola clase) trae su
        `undefined_reason` en los dos idiomas: la ficha inglesa no puede llevar
        la redacción castellana."""
        una_clase = Muestra.binaria(("si", "si", "si"), classes=("no", "si"),
                                    positive_label="si", probabilidades=(0.9, 0.8, 0.7))
        recal = ajustar_recalibracion_logistica(una_clase).a_json()
        self.assertIsNone(recal["a"])
        perfil = _perfil(recalibracion=recal)
        for locale, otro in (("es", "en"), ("en", "es")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(perfil, locale=locale)
                self.assertIn(f"- **undefined_reason**: {recal['undefined_reason'][locale]}",
                              ficha)
                self.assertNotIn(recal["undefined_reason"][otro], ficha)


class FichaDelPerfilTest(unittest.TestCase):
    """Lo que lee una persona. El alcance va ARRIBA, y lo que no se publica se
    dice con su motivo."""

    def _indice(self, ficha: str, trozo: str) -> int:
        self.assertIn(trozo, ficha, f"no está en la ficha: {trozo!r}")
        return ficha.index(trozo)

    def test_validacion_interna_unicamente_va_ANTES_que_el_primer_numero(self):
        for locale, aviso, umbrales in (("es", "VALIDACIÓN INTERNA ÚNICAMENTE", "## Umbrales"),
                                        ("en", "INTERNAL VALIDATION ONLY", "## Thresholds")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(_perfil(), locale=locale)
                sitio = self._indice(ficha, aviso)
                self.assertLess(sitio, self._indice(ficha, umbrales))
                self.assertLess(sitio, self._indice(ficha, "0.75"))
                # y no en una nota al pie: dentro de las cinco primeras líneas
                lineas = [l for l in ficha.splitlines() if l.strip()]
                self.assertIn(aviso, "\n".join(lineas[:5]))

    def test_una_validacion_externa_NO_dice_interna_unicamente(self):
        """La otra mitad: si la frase saliera siempre, no estaría diciendo nada."""
        ficha = ficha_del_perfil(_perfil(evidencia="external_validation"), locale="es")
        self.assertNotIn("VALIDACIÓN INTERNA ÚNICAMENTE", ficha)
        self.assertIn("VALIDACIÓN EXTERNA", ficha)

    #: Lo que la ficha escribe arriba para cada alcance, y la frase que NIEGA
    #: una separación temporal. Medidas sobre la redacción, en los dos idiomas.
    _MARCA_DE_ALCANCE = {
        "es": {"internal_only": "VALIDACIÓN INTERNA ÚNICAMENTE",
               "temporal": "VALIDACIÓN TEMPORAL", "external": "VALIDACIÓN EXTERNA"},
        "en": {"internal_only": "INTERNAL VALIDATION ONLY",
               "temporal": "TEMPORAL VALIDATION", "external": "EXTERNAL VALIDATION"},
    }
    _NIEGA_SEPARACION_TEMPORAL = {"es": "ni separación temporal", "en": "no temporal split"}
    _AFIRMA_SEPARACION_TEMPORAL = {"es": "SÍ separa por tiempo", "en": "DOES separate by time"}

    def test_el_alcance_no_niega_una_separacion_temporal_que_existe_en_las_24(self):
        """Las 24 combinaciones de evidencia × diseño, y no una muestra: el
        defecto vivía justo en las seis que nadie miró (auditoría del
        2026-09-15). Con evidencia débil (`test_used_for_development`,
        `development_estimate`, `not_evaluated`) y diseño `temporal` o
        `groups_and_time` el alcance es `internal_only` —bien—, y la ficha
        escribía «no hay cohorte externa NI SEPARACIÓN TEMPORAL»: falso, la
        partición sí separa por tiempo.

        Para cada una, en los dos idiomas: la marca del alcance es la suya; con
        un diseño que separa por tiempo la ficha no niega esa separación, y si
        el alcance es interno lo DICE (un aserto negativo lo pasa una ficha en
        blanco); y sin separación temporal la negación sigue escrita, porque ahí
        es verdad."""
        base = _perfil()
        con_tiempo_e_interna = 0
        combinaciones = 0
        for evidencia in ETIQUETAS_DE_EVIDENCIA:
            for diseno in TIPOS_DE_PARTICION:
                perfil = dataclasses.replace(base, evidencia=evidencia, diseno=diseno)
                alcance = alcance_de_validacion(evidencia=evidencia, diseno=diseno)
                con_tiempo = diseno in ("temporal", "groups_and_time")
                combinaciones += 1
                if con_tiempo and alcance == "internal_only":
                    con_tiempo_e_interna += 1
                for locale in ("es", "en"):
                    with self.subTest(evidencia=evidencia, diseno=diseno, locale=locale):
                        ficha = ficha_del_perfil(perfil, locale=locale)
                        cabecera = next(l for l in ficha.splitlines() if l.startswith("> "))
                        self.assertIn(self._MARCA_DE_ALCANCE[locale][alcance], cabecera)
                        niega = self._NIEGA_SEPARACION_TEMPORAL[locale]
                        afirma = self._AFIRMA_SEPARACION_TEMPORAL[locale]
                        if con_tiempo:
                            self.assertNotIn(niega, ficha)
                            if alcance == "internal_only":
                                self.assertIn(afirma, cabecera)
                        elif alcance == "internal_only":
                            self.assertIn(niega, cabecera)
                            self.assertNotIn(afirma, ficha)
        # el recorrido es el entero, y las seis que fallaban están dentro
        self.assertEqual(combinaciones, 24)
        self.assertEqual(con_tiempo_e_interna, 6)

    def test_los_datos_sinteticos_se_avisan_y_no_se_les_atribuye_precision(self):
        self.assertIn("SINTÉTICOS", ficha_del_perfil(_perfil(), locale="es"))
        self.assertIn("SYNTHETIC", ficha_del_perfil(_perfil(), locale="en"))
        limpia = ficha_del_perfil(_perfil(datos_sinteticos=False), locale="es")
        self.assertNotIn("SINTÉTICOS", limpia)

    def test_sin_prevalencia_la_ficha_DICE_que_calla_ppv_y_npv(self):
        for locale, frase in (("es", "no se publican PPV"), ("en", "PPV and NPV are not")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(_perfil(), locale=locale)
                self.assertIn(frase, ficha)

    def test_la_casilla_vacia_no_queda_MUDA_y_el_motivo_va_debajo(self):
        """La casilla dice «no publicado» y el motivo entero, con las métricas
        que lo comparten, va una sola vez debajo de la tabla."""
        for locale, marca, motivo_ in (("es", "no publicado", "nadie ha declarado la prevalencia"),
                                       ("en", "not published", "nobody declared target")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(_perfil(), locale=locale)
                self.assertIn(f"| {marca} |", ficha)
                self.assertIn(motivo_, ficha)
                self.assertIn(f"**{PPV_A_PREVALENCIA}, {NPV_A_PREVALENCIA}**", ficha)
                # una sola vez, no una por umbral
                self.assertEqual(ficha.count(motivo_), 1)

    def test_con_prevalencia_no_hay_nota_de_ausencia(self):
        """La otra mitad: la nota no se imprime siempre."""
        perfil = _perfil(tabla=tabla_de_umbrales(
            _a_mano(), umbrales=(0.5,), diseno="iid",
            estimando="fixed_model_on_population", semilla=1, prevalencia=0.02,
            remuestras=50))
        self.assertNotIn("no publicado", ficha_del_perfil(perfil, locale="es"))
        self.assertNotIn("not published", ficha_del_perfil(perfil, locale="en"))

    def test_con_prevalencia_la_ficha_la_ENSEÑA_al_lado_del_numero(self):
        muestra = _a_mano()
        perfil = _perfil(tabla=tabla_de_umbrales(
            muestra, umbrales=(0.5,), diseno="iid",
            estimando="fixed_model_on_population", semilla=1, prevalencia=0.02,
            remuestras=50, declarados_por="el equipo de medicina interna"))
        ficha = ficha_del_perfil(perfil, locale="es")
        self.assertIn("0.0200", ficha)          # la declarada
        self.assertIn("0.4000", ficha)          # la de la muestra, al lado
        self.assertIn("0.0841", ficha)          # el PPV transportado
        self.assertIn("el equipo de medicina interna", ficha)
        self.assertIn("Bayes", ficha)           # el supuesto, escrito

    def test_cuando_el_modelo_no_supera_a_las_dos_la_ficha_lo_dice(self):
        perfil = _perfil(curva=curva_de_decision(_a_mano(),
                                                 umbrales_de_probabilidad=(0.05,)))
        self.assertIn("no supera a las dos referencias en ningún",
                      ficha_del_perfil(perfil, locale="es"))
        self.assertIn("beats both references at no declared",
                      ficha_del_perfil(perfil, locale="en"))

    def test_y_cuando_SI_supera_no_lo_dice(self):
        ficha = ficha_del_perfil(_perfil(), locale="es")
        self.assertNotIn("no supera a las dos referencias en ningún", ficha)

    def test_la_curva_de_CADA_SUBGRUPO_sale_en_la_ficha_con_sus_numeros(self):
        """Medido el 2026-09-16: vaciar el bucle que escribe `curvas_por_
        segmento` borraba todas las curvas por subgrupo del documento y las 67
        pruebas seguían verdes.

        El subgrupo son las filas 1, 2, 5 y 6: `si` 0,90 0,75 y `no` 0,55 0,45.
        n=4, dos positivos → prevalencia **0,5** (la de la muestra entera es
        0,4, así que la del subgrupo no se puede confundir con ella).
        pt=0,20 → los cuatro tienen p >= 0,20: TP=2, FP=2.
          NB = 2/4 − (2/4)·0,25 = **0,375**; tratar a todos, lo mismo → no supera.
        pt=0,50 → 0,90 0,75 (`si`) y 0,55 (`no`): TP=2, FP=1.
          NB = 2/4 − (1/4)·1 = **0,25**; tratar a todos = 2/4 − 2/4 = **0** → supera.
        Ninguna de esas filas coincide con las de la muestra entera (0,30/0,25
        y 0,20/−0,20)."""
        indices = [0, 1, 4, 5]
        curva = curva_de_decision(_recorte(_a_mano(), indices),
                                  umbrales_de_probabilidad=(0.2, 0.5),
                                  segmento_id="mayores_de_75", predefinido=False)
        perfil = _perfil(curvas_por_segmento=(curva,))
        for locale, dca, subgrupos, marca, si, sin_ventaja in (
                ("es", "## Beneficio neto", "## Subgrupos", "exploratorio", "sí",
                 "NO supera a las dos referencias en estos umbrales**: 0.200"),
                ("en", "## Net benefit", "## Subgroups", "exploratory", "yes",
                 "does NOT beat both references at these thresholds**: 0.200")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(perfil, locale=locale)
                inicio = self._indice(ficha, f"### mayores_de_75 ({marca})")
                # dentro de la sección de la curva de decisión, no en otra
                self.assertLess(self._indice(ficha, dca), inicio)
                seccion = ficha[inicio:self._indice(ficha, subgrupos)]
                self.assertIn("| 0.200 | 0.3750 | 0.3750 | 0.0000 | no |", seccion)
                self.assertIn(f"| 0.500 | 0.2500 | 0.0000 | 0.0000 | {si} |", seccion)
                self.assertIn(sin_ventaja, seccion)

    def test_la_curva_de_cada_subgrupo_dice_SU_prevalencia(self):
        """El beneficio neto depende de la prevalencia, y la de un subgrupo no
        es la de la muestra: la curva la guardaba y la ficha no la escribía, así
        que se leía a la de arriba. Subgrupo de la prueba anterior:
        prevalencia 0,5 con n=4; la muestra entera, 0,4 con n=10."""
        curva = curva_de_decision(_recorte(_a_mano(), [0, 1, 4, 5]),
                                  umbrales_de_probabilidad=(0.2, 0.5),
                                  segmento_id="mayores_de_75", predefinido=False)
        perfil = _perfil(curvas_por_segmento=(curva,))
        for locale, subgrupos, frase in (
                ("es", "## Subgrupos", "SOBRE ESTE SUBGRUPO, a su prevalencia observada"),
                ("en", "## Subgroups", "ON THIS SUBGROUP, at its observed prevalence")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(perfil, locale=locale)
                inicio = self._indice(ficha, "### mayores_de_75")
                seccion = ficha[inicio:self._indice(ficha, subgrupos)]
                self.assertIn(f"{frase}: 0.5000 (n=4).", seccion)
                self.assertNotIn("0.4000", seccion)

    def test_la_ficha_dice_que_el_BENEFICIO_NETO_tambien_es_de_esta_muestra(self):
        """`TP/n` y `FP/n` son proporciones de la muestra: el beneficio neto
        depende de la prevalencia igual que el PPV. Lo que NO se puede hacer es
        transportarlo con una fórmula, y la ficha lo dice cuando hay una
        prevalencia declarada distinta."""
        for locale, frase in (("es", "SOBRE ESTA MUESTRA"), ("en", "ON THIS SAMPLE")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))
        # la prevalencia observada, al lado
        self.assertIn("0.4000", ficha_del_perfil(_perfil(), locale="es"))

        con_prevalencia = _perfil(tabla=tabla_de_umbrales(
            _a_mano(), umbrales=(0.5,), diseno="iid",
            estimando="fixed_model_on_population", semilla=1, prevalencia=0.02,
            remuestras=50))
        for locale, frase in (("es", "no está transportado a la prevalencia"),
                              ("en", "it is not transported to declared")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(con_prevalencia, locale=locale))

    def test_y_sin_prevalencia_declarada_no_habla_de_transportar(self):
        """La otra mitad: el aviso del transporte solo aparece cuando hay algo
        a lo que no se transporta."""
        self.assertNotIn("no está transportado",
                         ficha_del_perfil(_perfil(), locale="es"))

    def test_sin_calibracion_medida_la_ficha_no_la_da_por_buena(self):
        for locale, frase in (("es", "No se ha medido la calibración"),
                              ("en", "Calibration was not measured")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))

    def test_con_calibracion_medida_sale_el_ece(self):
        perfil = _perfil(calibracion=curva_de_fiabilidad(_a_mano(), n_bins=4))
        self.assertIn("ECE", ficha_del_perfil(perfil, locale="es"))

    def test_sin_politica_de_faltantes_se_dice_que_no_consta(self):
        for locale, frase in (("es", "No se declara ninguna política"),
                              ("en", "No missing-data policy")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))

    def test_con_politica_de_faltantes_se_escribe(self):
        perfil = _perfil(politica_de_faltantes={"edad": "mediana de train + indicador"})
        self.assertIn("mediana de train + indicador",
                      ficha_del_perfil(perfil, locale="es"))

    def test_un_segmento_sin_cifra_no_sale_con_un_guion_mudo(self):
        """Un segmento de solo negativos no tiene sensibilidad: la ficha
        escribe POR QUÉ, no un guion."""
        solo_negativos = [i for i, y in enumerate(_Y) if y == "no"]
        segmento = analizar_segmento("sensitivity", _a_mano(), solo_negativos,
                                     segmento_id="solo_negativos", predefinido=False,
                                     umbral=0.5)
        self.assertIsNone(segmento.metrica_segmento.value)
        ficha = ficha_del_perfil(_perfil(segmentos=(segmento,)), locale="es")
        self.assertIn("solo_negativos", ficha)
        self.assertIn(segmento.metrica_segmento.undefined_reason["es"], ficha)

    def test_sin_equipo_la_ficha_dice_que_nadie_predefinio_subgrupos(self):
        for locale, frase in (("es", "Nadie ha predefinido subgrupos"),
                              ("en", "Nobody predefined any subgroup")):
            with self.subTest(locale=locale):
                self.assertIn(frase, ficha_del_perfil(_perfil(), locale=locale))

    def test_la_ficha_INGLESA_no_lleva_castellano_colado(self):
        """Un barrido con palabras FUNCIONALES sobre la ficha entera: no se
        pueden evitar escribiendo en castellano, y cazan lo que una lista de
        palabras escogida a mano no caza — por ejemplo un «IC 95 %» que se
        quedó sin traducir dentro de una casilla."""
        import re
        segmento = analizar_segmento("sensitivity", _a_mano(), [0, 1, 4, 5],
                                     segmento_id="segment_a", predefinido=False,
                                     umbral=0.5)
        perfil = _perfil(segmentos=(segmento,),
                         curva=curva_de_decision(_a_mano(),
                                                 umbrales_de_probabilidad=(0.05, 0.5)),
                         calibracion=curva_de_fiabilidad(_a_mano(), n_bins=4))
        ficha = ficha_del_perfil(perfil, locale="en")
        palabras = set(re.findall(r"[a-záéíóúñ]+", ficha.lower()))
        self.assertEqual(palabras.intersection(TextosDeLaFichaTest.FUNCIONALES), set())

    def test_y_la_ficha_CASTELLANA_no_esta_en_ingles(self):
        """La otra mitad: un barrido que solo mira la inglesa lo pasaría una
        ficha que saliera siempre en inglés."""
        ficha = ficha_del_perfil(_perfil(), locale="es")
        for marca in ("Thresholds", "not published", "INTERNAL VALIDATION"):
            self.assertNotIn(marca, ficha)

    def test_un_idioma_que_no_se_sabe_sale_en_ingles(self):
        self.assertEqual(ficha_del_perfil(_perfil(), locale="fr"),
                         ficha_del_perfil(_perfil(), locale="en"))


class TextosDeLaFichaTest(unittest.TestCase):
    """Lo que redacta el core se traduce en el core: las dos redacciones, con
    las mismas claves y sin castellano colado en la inglesa."""

    FUNCIONALES = ("el", "la", "los", "las", "del", "con", "que", "desde", "para",
                   "por", "una", "uno", "sin", "pero", "como", "cuando", "más",
                   "porque", "aunque", "sobre", "entre", "hay", "tiene", "puede")

    def test_los_dos_idiomas_tienen_las_mismas_claves(self):
        self.assertEqual(set(_T), set(IDIOMAS))
        self.assertEqual(set(_T["es"]), set(_T["en"]))

    def test_ninguna_redaccion_esta_vacia(self):
        for idioma in IDIOMAS:
            for clave, texto in _T[idioma].items():
                with self.subTest(idioma=idioma, clave=clave):
                    self.assertTrue(texto.strip())

    def test_los_huecos_son_los_mismos_en_los_dos_idiomas(self):
        for clave in _T["es"]:
            with self.subTest(clave=clave):
                self.assertEqual(huecos_de(_T["es"][clave]), huecos_de(_T["en"][clave]))

    def test_el_ingles_no_lleva_castellano_dentro(self):
        import re
        for clave, texto in _T["en"].items():
            palabras = set(re.findall(r"[a-záéíóúñ]+", texto.lower()))
            with self.subTest(clave=clave):
                self.assertEqual(palabras.intersection(self.FUNCIONALES), set())

    #: Las tres casillas que SÍ son iguales en los dos idiomas, y por qué:
    #: `col_recuentos` es «TP/FP/TN/FN» (siglas, no palabras), `col_pt` es el
    #: símbolo `pt` de la literatura de DCA, y «no» se escribe igual en
    #: castellano y en inglés. La lista es CERRADA: añadir una casilla aquí
    #: para que pase la prueba es justo lo que la prueba existe para impedir.
    IGUALES_A_PROPOSITO = ("col_recuentos", "col_pt", "no")

    def test_las_dos_redacciones_son_DISTINTAS(self):
        """Copiar el castellano en la casilla inglesa pasaría las tres pruebas
        de arriba en cuanto no llevara ninguna palabra funcional."""
        iguales = [c for c in _T["es"] if _T["es"][c] == _T["en"][c]]
        self.assertEqual(sorted(iguales), sorted(self.IGUALES_A_PROPOSITO))


class CasoClinicoSinteticoTest(unittest.TestCase):
    """El perfil entero sobre el caso clínico SINTÉTICO de `/casos` (reingreso
    hospitalario). **Datos sintéticos: se comprueba el cálculo, nunca la
    precisión clínica.**"""

    @classmethod
    def setUpClass(cls):
        cls.muestra, cls.filas = _cohorte_clinica_sintetica()

    def test_el_perfil_se_construye_entero_y_sella(self):
        umbrales_del_equipo = (0.3, 0.5, 0.7)
        tabla = tabla_de_umbrales(
            self.muestra, umbrales=umbrales_del_equipo, diseno="iid",
            estimando="fixed_model_on_population", semilla=20260914, remuestras=200)
        curva = curva_de_decision(self.muestra,
                                  umbrales_de_probabilidad=umbrales_del_equipo)
        mayores = [i for i, f in enumerate(self.filas) if f["edad"] > 75]
        perfil = PerfilClinico(
            perfil_id="reingreso-sintetico", evidencia="independent_test",
            diseno="iid", datos_sinteticos=True, tabla=tabla, curva=curva,
            calibracion=curva_de_fiabilidad(self.muestra, n_bins=10),
            segmentos=(analizar_segmento("sensitivity", self.muestra, mayores,
                                         segmento_id="edad_mayor_de_75",
                                         predefinido=False, umbral=0.5),),
            curvas_por_segmento=(curva_de_decision(
                _recorte(self.muestra, mayores),
                umbrales_de_probabilidad=umbrales_del_equipo,
                segmento_id="edad_mayor_de_75", predefinido=False),))
        self.assertEqual(perfil.alcance, "internal_only")
        self.assertEqual(len(perfil.tabla.filas), 3)
        self.assertEqual(len(perfil.curva.puntos), 3)
        self.assertEqual(perfil.segmentos_exploratorios, ("edad_mayor_de_75",))
        self.assertEqual(len(perfil.digest()), 64)
        # el documento se puede serializar entero
        cuerpo = perfil.a_json()
        self.assertEqual(cuerpo["schema"], PerfilClinico.ESQUEMA)
        self.assertEqual(cuerpo["alcance"], "internal_only")
        self.assertTrue(cuerpo["datos_sinteticos"])
        # y la ficha sale en los dos idiomas, con el aviso arriba
        for locale, aviso in (("es", "SINTÉTICOS"), ("en", "SYNTHETIC")):
            with self.subTest(locale=locale):
                ficha = ficha_del_perfil(perfil, locale=locale)
                self.assertIn(aviso, ficha)
                self.assertLess(ficha.index(aviso), len(ficha) // 3)

    def test_la_prevalencia_de_esta_cohorte_NO_es_la_de_un_hospital(self):
        """La receta del caso (`edad > 75 OR reingreso_previo > 0.5`) produce
        una prevalencia de ~65 % en `/casos` (0,6575) y de 0,6925 en la cohorte
        de esta prueba, que sortea sus propias filas (ver
        `_cohorte_clinica_sintetica`). Un PPV calculado ahí y enseñado a un equipo
        cuya población reingresa al 10 % sería un número que miente sin
        equivocarse en ninguna cuenta — y por eso, sin prevalencia declarada,
        no se publica."""
        tabla = tabla_de_umbrales(self.muestra, umbrales=(0.5,), diseno="iid",
                                  estimando="fixed_model_on_population",
                                  semilla=1, remuestras=100)
        self.assertGreater(tabla.prevalencia_observada, 0.55)
        self.assertIsNone(tabla.filas[0].ppv.value)

        # Medido sobre esta cohorte en t=0,5: prevalencia 0,6925, sens 0,6209,
        # esp 0,9594; PPV observado **0,9718** y NPV observado **0,5291**. A una
        # prevalencia declarada del 2 %, el mismo modelo da PPV **0,2377** y NPV
        # **0,9920**: el PPV cae a la CUARTA parte y el NPV sube. Los dos lados,
        # porque corregir un sesgo puede crear el contrario.
        al_dos = tabla_de_umbrales(self.muestra, umbrales=(0.5,), diseno="iid",
                                   estimando="fixed_model_on_population", semilla=1,
                                   prevalencia=0.02, remuestras=100)
        ppv_observado = calcular("ppv", self.muestra, umbral=0.5).value
        npv_observado = calcular("npv", self.muestra, umbral=0.5).value
        self.assertAlmostEqual(ppv_observado, 0.971751, places=5)
        self.assertAlmostEqual(al_dos.filas[0].ppv.value, 0.237652, places=5)
        self.assertAlmostEqual(npv_observado, 0.529148, places=5)
        self.assertAlmostEqual(al_dos.filas[0].npv.value, 0.992001, places=5)
        self.assertLess(al_dos.filas[0].ppv.value, ppv_observado / 3)
        self.assertGreater(al_dos.filas[0].npv.value, npv_observado)


def _recorte(muestra: Muestra, indices) -> Muestra:
    """Un recorte VALIDADO (no el atajo interno de 105-C2): esta muestra se
    construye una vez, no mil, y pasar por `__post_init__` la comprueba."""
    return Muestra.binaria(
        [muestra.y_true[i] for i in indices], classes=list(muestra.classes),
        positive_label=muestra.positive_label,
        probabilidades=[muestra.probabilidad_del_positivo[i] for i in indices])


class MatrizYCurvaCuadranTest(unittest.TestCase):
    """La curva cuenta TP y FP con la misma matriz de 105-C1: si un día dejaran
    de cuadrar, este corte estaría publicando recuentos propios."""

    def test_los_recuentos_de_la_curva_son_los_de_la_matriz(self):
        muestra = _a_mano()
        for pt in (0.1, 0.2, 0.5, 0.8):
            matriz = matriz_de_confusion(muestra, umbral=pt)
            punto = curva_de_decision(muestra, umbrales_de_probabilidad=(pt,)).puntos[0]
            with self.subTest(pt=pt):
                self.assertEqual((punto.tp, punto.fp), (matriz.tp, matriz.fp))


if __name__ == "__main__":
    unittest.main()
