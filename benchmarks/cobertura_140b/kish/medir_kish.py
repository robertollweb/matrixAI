# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b, tercera reparación (K1): la cobertura REAL del intervalo de Wilson corregido por Kish, medida con EL CÓDIGO del
núcleo (`intervalo_por_filas.correccion_de_recuentos`), y la del Wilson por filas al lado. Produce `medir_kish.json`, del
que sale `COBERTURA_MEDIDA_POR_SUJETOS` (una prueba los compara número a número).

**Duración estimada y techo, escritos ANTES de medir** (regla del 25-09): la rejilla anterior (468 celdas × 2.000) tardó
93 s; esta tiene 585 celdas de tamaños realistas + 124 de concentración, con tamaños más grandes: **estimado 3–5 min**,
con TECHO de 900 s (`timeout 900`), un proceso, `nice`, y la carga ≤ 6 antes de lanzarlo.

**Modelo**: k sujetos, cada uno con su tasa (logit normal: `logit(tasa) + sd·z`), sus filas 0/1 independientes dentro del
sujeto, y los tamaños independientes de la tasa. La verdad es la proporción por filas esperada, `E[expit]` (10⁶ extracciones).

**La rejilla, con los tamaños que trae la gente** (K1: la anterior tenía 3 de sus 4 formas de tamaño FIJO, donde `m0` y
`Σm²/N` coinciden, y la media escondía la peor celda):
- sujetos: 2, 3, 4, 5, 6, 8, 10, 13, 15, 20, 30, 50, 100;
- filas por sujeto: 5, 20 y 40 fijas, uniforme 1–39, y lognormal con media 10 y CV 1,5 (mínimo 1);
- tasa 0,6, 0,9 y 0,97; desviación del logit entre sujetos 0 (control), 0,5 y 1; R = 2.000 por celda.
- APARTE, la CONCENTRACIÓN: un sujeto con el 30–90 % de las filas (el resto, uniforme 1–39), con 3, 5, 10, 30 y 100
  sujetos, tasa 0,6 y 0,9, sd 0,5 y 1; y las dos celdas exactas del auditor (91 filas + 1 por sujeto, 10 y 30 sujetos).

**Lo que se registra además**: en cada celda, las repeticiones SIN varianza (todas 0 o todas 1) y cuántas de ellas cubre
(K2: con la regla anterior, ninguna).

Uso, desde la raíz del núcleo: `timeout 900 nice python3 benchmarks/cobertura_140b/kish/medir_kish.py [R]`.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
import matrixai  # noqa: E402
from matrixai.estudio.intervalo_por_filas import correccion_de_recuentos, intervalo_recortado  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEMILLA = 20261102
SUJETOS = [2, 3, 4, 5, 6, 8, 10, 13, 15, 20, 30, 50, 100]
_S2 = math.log(1 + 1.5 ** 2)
FORMAS = {
    "5": lambda rng, k: np.full(k, 5),
    "20": lambda rng, k: np.full(k, 20),
    "40": lambda rng, k: np.full(k, 40),
    "1-39": lambda rng, k: rng.integers(1, 40, k),
    "lognormal_cv1.5": lambda rng, k: np.maximum(1, np.round(rng.lognormal(math.log(10) - _S2 / 2, math.sqrt(_S2), k))).astype(int),
}
TASAS = [0.6, 0.9, 0.97]
SDS = [0.0, 0.5, 1.0]


def dominante(fraccion):
    def filas(rng, k):
        resto = rng.integers(1, 40, k - 1)
        grande = max(1, int(round(fraccion / (1 - fraccion) * resto.sum())))
        return np.concatenate([[grande], resto])
    return filas


def auditor(rng, k):
    return np.array([91] + [1] * (k - 1))


def verdad_de(rng, base, sd):
    return float(np.mean(1 / (1 + np.exp(-(base + sd * rng.standard_normal(1_000_000)))))) if sd > 0 else 1 / (1 + math.exp(-base))


def celda(rng, k, forma, filas_de, p0, sd):
    base = math.log(p0 / (1 - p0))
    verdad = verdad_de(rng, base, sd)
    dentro_k = dentro_f = sin_var = sin_var_dentro = 0
    fraccion = 0.0
    for _ in range(R):
        p = 1 / (1 + np.exp(-(base + sd * rng.standard_normal(k))))
        m = filas_de(rng, k)
        e = rng.binomial(m, p)
        c = correccion_de_recuentos(e.tolist(), m.tolist())
        iv_f = intervalo_recortado(int(e.sum()), int(m.sum()))
        iv_k = c.intervalo_95 if c is not None else iv_f
        dentro = iv_k[0] <= verdad <= iv_k[1]
        dentro_k += dentro
        dentro_f += iv_f[0] <= verdad <= iv_f[1]
        fraccion += m.max() / m.sum()
        if c is not None and c.sin_varianza:
            sin_var += 1
            sin_var_dentro += dentro
    return {"sujetos": k, "forma": forma, "tasa": p0, "sd_logit": sd, "kish": dentro_k / R, "por_filas": dentro_f / R,
            "fraccion_del_mayor_media": round(fraccion / R, 3), "sin_varianza": sin_var,
            "sin_varianza_cubiertas": sin_var_dentro}


def main():
    rng = np.random.default_rng(SEMILLA)
    t0 = time.time()
    celdas = []
    for k in SUJETOS:
        for forma, filas_de in FORMAS.items():
            for p0 in TASAS:
                for sd in SDS:
                    celdas.append(celda(rng, k, forma, filas_de, p0, sd))
        print(f"{k} sujetos: {time.time() - t0:.0f} s", flush=True)
    concentracion = []
    for k in (3, 5, 10, 30, 100):
        for fr in (0.3, 0.4, 0.5, 0.6, 0.75, 0.9):
            for p0 in (0.6, 0.9):
                for sd in (0.5, 1.0):
                    concentracion.append({**celda(rng, k, f"dominante_{fr}", dominante(fr), p0, sd), "fraccion": fr})
    for k in (10, 30):
        concentracion.append(celda(rng, k, "auditor_91_y_1", auditor, 0.9, 1.0))
    print(f"concentración: {time.time() - t0:.0f} s", flush=True)

    tabla = {}
    for k in SUJETOS:
        con = [c for c in celdas if c["sujetos"] == k and c["sd_logit"] > 0]
        por_forma = {f: round(float(np.mean([c["kish"] for c in con if c["forma"] == f])), 3) for f in FORMAS}
        kish = sorted(c["kish"] for c in con)
        tabla[k] = {"kish_peor_forma_media": min(por_forma.values()),
                    "kish_por_forma_media": por_forma,
                    "kish_media_con_efecto": round(float(np.mean(kish)), 3),
                    "kish_p10_con_efecto": round(float(np.percentile(kish, 10)), 3),
                    "kish_min_con_efecto": round(kish[0], 3),
                    "filas_media_con_efecto": round(float(np.mean([c["por_filas"] for c in con])), 3),
                    "kish_media_sin_efecto": round(float(np.mean([c["kish"] for c in celdas
                                                                  if c["sujetos"] == k and c["sd_logit"] == 0])), 3)}
        print(k, json.dumps(tabla[k], ensure_ascii=False), flush=True)
    salida = Path(__file__).with_name("medir_kish.json")
    salida.write_text(json.dumps({
        "nucleo": str(Path(matrixai.__file__).resolve().relative_to(RAIZ)), "R": R, "semilla": SEMILLA,
        "resumen_de_la_tabla": "kish_peor_forma_media: por número de sujetos, la PEOR de las 5 formas de tamaños, cada una en "
                               "media de sus 6 celdas con efecto del sujeto (3 tasas × sd 0,5 y 1)",
        "celdas": celdas, "concentracion": concentracion, "tabla": tabla,
        "segundos": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
