# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b — tercera reparación (tercera auditoría, `~/.agentes/140b/reauditoria3/INFORME.md`, K1–K5 y la nota), en el núcleo.

- K1: el efecto de diseño de Kish con el tamaño PONDERADO `Σm²/N` (no el `m0` del ANOVA): con visitas desiguales la cobertura
  vuelve a la que dice la frase; con un sujeto que tiene más de la mitad de las filas, la marca de aproximado y una frase que
  no promete la cifra.
- K2: todas 0 o todas 1 → lo prudente (`n_eff = N²/Σm²`) y el intervalo vuelve a cubrir.
- K3: «de cada 100» redondea HACIA ABAJO (0,927 → 92).
- K4: la procedencia de las dos tablas está en el repositorio (`benchmarks/cobertura_140b/`).
- La nota: cada marca de «aproximado» nombra SU intervalo.

Cada prueba con nombre de hallazgo salió ROJA con el código anterior (revirtiendo solo su arreglo).
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pytest

from matrixai.estudio import intervalo_por_filas as ipf
from matrixai.estudio.intervalo_por_filas import (
    FRACCION_DOMINANTE, correccion_de_recuentos, cobertura_medida_por_sujetos, texto_de_las_unidades,
    texto_de_un_conjunto)

RAIZ = Path(__file__).resolve().parents[1]


def _cobertura(rng, k: int, filas_de, tasa: float, sd: float, repeticiones: int) -> tuple[float, int, int]:
    """(cobertura del corregido, repeticiones sin varianza, cuántas de ellas cubre), con EL código del núcleo."""
    base = math.log(tasa / (1 - tasa))
    verdad = float(np.mean(1 / (1 + np.exp(-(base + sd * rng.standard_normal(400_000))))))
    dentro = sin_var = sin_var_dentro = 0
    for _ in range(repeticiones):
        p = 1 / (1 + np.exp(-(base + sd * rng.standard_normal(k))))
        m = filas_de(rng, k)
        e = rng.binomial(m, p)
        c = correccion_de_recuentos(e.tolist(), m.tolist())
        cubre = c.intervalo_95[0] <= verdad <= c.intervalo_95[1]
        dentro += cubre
        if c.sin_varianza:
            sin_var += 1
            sin_var_dentro += cubre
    return dentro / repeticiones, sin_var, sin_var_dentro


def _lognormal(rng, k):
    s2 = math.log(1 + 1.5 ** 2)
    return np.maximum(1, np.round(rng.lognormal(math.log(10) - s2 / 2, math.sqrt(s2), k))).astype(int)


# ---------------------------------------------------------------------------------------------------------------------------
# K1 · el tamaño ponderado
# ---------------------------------------------------------------------------------------------------------------------------

def test_K1_con_visitas_desiguales_la_cobertura_es_la_que_dice_la_frase():
    """La celda del auditor que más caía: 15 sujetos con tamaños lognormales de CV 1,5. Con `m0` cubría 0,851 diciendo 93; con
    `Σm²/N`, 0,915 (R = 2.000). Aquí R = 600 con semilla fija: el umbral queda a > 2 errores estándar de los dos."""
    rng = np.random.default_rng(11)
    cobertura, _, _ = _cobertura(rng, 15, _lognormal, 0.9, 1.0, 600)
    assert cobertura >= 0.885, cobertura
    medida, _ = cobertura_medida_por_sujetos(15)
    assert cobertura >= medida - 0.03, (cobertura, medida)


def test_K1_con_tamanos_iguales_no_cambia_nada():
    """Con tamaños iguales `Σm²/N = m0 = m`: el mismo efecto de diseño que antes."""
    c = correccion_de_recuentos([38, 30, 40, 22, 35, 39, 28, 40, 33, 37, 25, 36, 39], [40] * 13)
    assert c.m_ponderado == pytest.approx(40) and c.m0 == pytest.approx(40)
    assert c.efecto_de_diseno == pytest.approx(1 + 39 * c.icc)


def test_K1_un_sujeto_con_mas_de_la_mitad_de_las_filas_lleva_la_marca_y_no_promete_la_cifra():
    """La celda del auditor (91 filas de un sujeto y 1 de cada uno de los demás) cubre 0,734 con 10 sujetos: la frase no
    puede decir «92 de cada 100». Y la frontera es «más de la mitad»: con el 50 % justo, ni marca ni frase distinta."""
    assert FRACCION_DOMINANTE == 0.5
    dominante = correccion_de_recuentos([80] + [1] * 14, [91] + [1] * 14)       # 15 sujetos, el mayor con el 86,7 %
    assert dominante.unidades == 15 and dominante.fraccion_del_mayor == pytest.approx(91 / 105)
    assert dominante.aproximado is True
    assert "un solo sujeto tiene el 87 % de las filas" in dominante.texto_aproximado["es"]
    assert "a single subject has 87% of the rows" in dominante.texto_aproximado["en"]
    assert "acierta bastante menos" in dominante.texto["es"] and "quite a lot less often" in dominante.texto["en"]
    assert not re.search(r"acierta unas \d+ de cada 100 veces, no 95", dominante.texto["es"])
    # el 50 % justo: no es «más de la mitad»
    mitad = correccion_de_recuentos([9, 1, 2, 1, 3, 0, 2, 1, 4, 2, 1, 0], [20, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1])
    assert mitad.fraccion_del_mayor == pytest.approx(0.5) and mitad.aproximado is False and mitad.texto_aproximado is None
    assert "acierta unas 92 de cada 100 veces o algo más, no 95" in mitad.texto["es"]


def test_K1_el_conjunto_del_simulador_tambien_ve_al_sujeto_dominante():
    unidades = ["A"] * 60 + [f"S{i}" for i in range(20) for _ in range(2)]        # 21 sujetos, A con el 60 %
    c = texto_de_las_unidades(unidades)
    assert c["unidades"] == 21 and c["filas"] == 100 and c["aproximado"] is True
    assert "un solo sujeto tiene el 60 % de las filas" in c["texto_aproximado"]["es"]
    assert "acierta bastante menos" in c["texto"]["es"]
    repartidas = texto_de_las_unidades([f"S{i}" for i in range(20) for _ in range(5)])
    assert repartidas == texto_de_un_conjunto(100, 20) and repartidas["aproximado"] is False


# ---------------------------------------------------------------------------------------------------------------------------
# K2 · sin varianza
# ---------------------------------------------------------------------------------------------------------------------------

def test_K2_cuando_todas_las_filas_dan_lo_mismo_el_intervalo_cubre():
    """Una celda del auditor: 4 × 10 filas, tasa 0,97, sd 1 (con 5 × 40, 200/200 daba [0,981; 1] con la verdad en 0,954). Con
    ICC = 0 esos intervalos no cubrían NUNCA. Ahora se cuentan esas repeticiones y cuántas cubre."""
    rng = np.random.default_rng(5)
    _, sin_var, cubiertas = _cobertura(rng, 4, lambda rng, k: np.full(k, 10), 0.97, 1.0, 400)
    assert sin_var >= 20, sin_var                                   # el caso se da de verdad
    assert cubiertas / sin_var >= 0.9, (cubiertas, sin_var)
    # y el caso exacto del auditor (verdad 0,954, medida por él con 2·10⁶ extracciones): antes [0,981; 1], fuera
    bajo, alto = correccion_de_recuentos([40] * 5, [40] * 5).intervalo_95
    assert bajo <= 0.954 <= alto, (bajo, alto)


def test_K2_con_tamanos_desiguales_el_n_efectivo_es_el_numero_efectivo_de_sujetos():
    filas = [30, 5, 5, 10]
    c = correccion_de_recuentos(filas, filas)                       # todas 1
    assert c.sin_varianza and c.n_efectivo == round(sum(filas) ** 2 / sum(m * m for m in filas), 3)


# ---------------------------------------------------------------------------------------------------------------------------
# K3 · «de cada 100» hacia abajo
# ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cobertura,esperado", [(0.927, 92), (0.9275, 92), (0.929, 92), (0.930, 93), (0.8999, 90),
                                                (0.891, 89)])
def test_K3_de_cada_100_es_el_entero_por_debajo(cobertura, esperado):
    assert ipf._de_cada_100(cobertura) == esperado


def test_K3_la_frase_de_un_punto_que_redondeado_subiria_dice_el_de_abajo(monkeypatch):
    """Con un punto en 0,927 (el de 10 sujetos de la tabla anterior) la frase dice 92, no 93."""
    monkeypatch.setattr(ipf, "COBERTURA_MEDIDA_POR_SUJETOS", ((2, 0.870), (10, 0.927)))
    c = correccion_de_recuentos([3, 4, 5, 6, 7, 8, 9, 2, 1, 5], [10] * 10)
    assert c.cobertura_medida == 0.927
    assert "acierta unas 92 de cada 100 veces, no 95" in c.texto["es"] and "93 de cada 100" not in c.texto["es"]
    assert "right about 92 times in 100" in c.texto["en"]


# ---------------------------------------------------------------------------------------------------------------------------
# K4 · la procedencia, en el repositorio
# ---------------------------------------------------------------------------------------------------------------------------

def test_K4_la_procedencia_de_las_dos_tablas_esta_versionada_y_el_c0_es_el_sellado():
    kish = RAIZ / "benchmarks" / "cobertura_140b" / "kish"
    c0 = RAIZ / "benchmarks" / "cobertura_140b" / "c0"
    for f in (kish / "medir_kish.py", kish / "medir_kish.json", c0 / "PROTOCOLO.md", c0 / "PROTOCOLO.sha256",
              c0 / "salidas" / "e3.json", c0 / "e3_cobertura.py", c0 / "lib140b.py", c0 / "simcob.py"):
        assert f.exists(), f
    # el protocolo es el sellado, y los guiones de E3 son los que él sella
    sello = (c0 / "PROTOCOLO.sha256").read_text().split()[0]
    assert hashlib.sha256((c0 / "PROTOCOLO.md").read_bytes()).hexdigest() == sello
    protocolo = (c0 / "PROTOCOLO.md").read_text()
    for guion in ("lib140b.py", "simcob.py", "e3_cobertura.py"):
        assert f"{hashlib.sha256((c0 / guion).read_bytes()).hexdigest()}  {guion}" in protocolo, guion
    # el guion de Kish lleva su duración estimada y su techo escritos
    texto = (kish / "medir_kish.py").read_text()
    assert "Duración estimada y techo" in texto and "timeout 900" in texto
    assert ipf.PROCEDENCIA_DE_LA_COBERTURA_POR_SUJETOS["medida_por"].endswith("(benchmarks/cobertura_140b/kish/medir_kish.py)")


# ---------------------------------------------------------------------------------------------------------------------------
# La nota · cada marca nombra su intervalo
# ---------------------------------------------------------------------------------------------------------------------------

def test_cada_marca_de_aproximado_nombra_su_intervalo():
    from matrixai.training.particion_por_diseno_grupos_y_tiempo import _frase_de_aproximado as marca_del_estricto
    estricto = marca_del_estricto(16)
    kish = correccion_de_recuentos([3, 4, 5, 6, 7, 8, 9, 2, 1], [10] * 9).texto_aproximado
    assert estricto["es"].startswith("intervalo aproximado (el de la nota, por remuestreo de sujetos): con menos de 50")
    assert estricto["en"].startswith("approximate interval (the score's, by resampling subjects): with fewer than 50")
    assert kish["es"].startswith("intervalo aproximado (el de Wilson corregido por sujeto): con menos de 10")
    assert kish["en"].startswith("approximate interval (the Wilson one corrected by subject): with fewer than 10")
