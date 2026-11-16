# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C5 — pruebas del analizador sobre datos sintéticos (python3 pruebas_analizador.py; sin pytest)."""
import analiza_kit_acierto as a

P = a.importar_protocolo()
claves = [(r, p) for r in range(3) for p in range(5)]
rival = {k: 0.80 + 0.01 * (i % 3) for i, k in enumerate(claves)}


def ver(delta, n=15, **kw):
    t = {k: v + delta(i) for i, (k, v) in enumerate(rival.items())}
    t = dict(list(t.items())[:n])
    return a.veredicto(P, t, rival, n_esperado=15, digest_ok=True, motivo_digest=None, **kw)


def main():
    # la regla: el hash coincide, y los números citados están en REGLA.md
    assert a.exigir_la_regla() == a.REGLA_SHA256
    regla = (a.AQUI / "REGLA.md").read_text(encoding="utf-8")
    assert "±0,005" in regla and a.MARGEN_DE_EQUIVALENCIA == 0.005 and "1.000 remuestras" in regla and a.REMUESTRAS_DEL_BOOTSTRAP == 1000
    g = ver(lambda i: 0.03 + 0.001 * (i % 2)); assert g["veredicto"] == "gana", g
    p = ver(lambda i: -0.03 - 0.001 * (i % 2)); assert p["veredicto"] == "pierde", p
    e = ver(lambda i: 0.001 * (-1) ** i); assert (e["veredicto"], e["subclase"]) == ("empate", "equivalente"), e
    # equivalente aunque EXCLUYA el cero: +0,002 constante cabe en ±0,005 -> empate, no gana
    e2 = ver(lambda i: 0.002 + 0.0001 * (i % 2)); assert (e2["veredicto"], e2["subclase"]) == ("empate", "equivalente"), e2
    n = ver(lambda i: 0.05 * (-1) ** i); assert (n["veredicto"], n["subclase"]) == ("empate", "no_concluyente"), n
    # no comparables: faltan pliegues de TabICL / del rival / digest distinto / menos de 2
    assert ver(lambda i: 0.03, n=14)["veredicto"] == "no_comparable"
    r_corto = dict(list(rival.items())[:14])
    assert a.veredicto(P, dict(rival), r_corto, n_esperado=15, digest_ok=True, motivo_digest=None)["veredicto"] == "no_comparable"
    assert a.veredicto(P, dict(rival), rival, n_esperado=15, digest_ok=False, motivo_digest="x")["veredicto"] == "no_comparable"
    assert a.veredicto(P, {(0, 0): 1.0}, {(0, 0): 0.0}, n_esperado=1, digest_ok=True, motivo_digest=None)["veredicto"] == "no_comparable"
    # signo: diferencia = TabICL - rival
    assert g["diferencia_media"] > 0 and p["diferencia_media"] < 0
    print("pruebas del analizador: OK")


if __name__ == "__main__":
    main()
