#!/usr/bin/env python3
"""120 · el VEREDICTO de una pasada de C2 contra R1, y cuándo PARAR (regla 9). Un solo sitio: lo usa el guion de la
medida (`referencia_120_r0.py --politica … --contra R1`) para cortar, y quien lea el resultado para decidir.

    python3 veredicto_120.py referencia_120_r1.json referencia_120_c2_l2_0.json   # el veredicto por tamaño

Lo que fija (enmienda 3 del protocolo 120, registrada ANTES de medir):

- **La cifra de un conjunto** es la del CAMPEÓN en `seleccion.media_de_la_seleccion` (la media de los pliegues de la
  validación cruzada sobre desarrollo, «la que decide»), en puntos ×100: AUROC en binarias, accuracy en multiclase y,
  en regresión, **R² de la validación cruzada = 1 − (RMSE del campeón / RMSE de la línea base)²**, con las dos medias
  del propio estudio. El estudio no da R² por intento y el único R² que da es el del TEST (`evaluacion_final`), que
  aquí no se usa nunca. La línea base no depende de la política de los árboles, así que esa R² es una función
  monótona del RMSE del campeón con la misma escala en R1 y en C2.
- **El control de cada conjunto**: la media de la línea base es la MISMA en R1 y en C2 (mismos pliegues, misma
  semilla). Si no lo es, el conjunto es INCOMPARABLE y la pasada para: el instrumento no está midiendo lo mismo.
- **Completar** es terminar (`completed`) con campeón. Un conjunto que en R1 no completaba y con C2 sí cuenta como
  uno que SUBE; al revés, «deja de completar».
- **El tamaño** de un conjunto es el que DECLARA el propio estudio (`politica_por_tamano.tamano` de los intentos de
  árboles), no uno calculado aquí.
- **La regla 4 por tamaño** (enmienda 1): en ese tamaño, más suben (≥ 1) que bajan (≥ 1), ninguno baja ≥ 2 y nadie
  deja de completar. |diferencia| < 1 no cuenta.
- **La regla 9, operativa**: la pasada de un perfil de L2 se CORTA en el primer conjunto que baja ≥ 2 o deja de
  completar, sea del tamaño que sea (la letra de la enmienda 1: «ese perfil se corta»). Lo no medido no cuenta
  como pérdida, pero tampoco como mejora: en los tamaños que no llegaron a decidirse se queda la configuración de
  HOY (regla 4c). Es la lectura conservadora: corta antes y nunca cambia nada sin medirlo.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

UMBRAL = 1.0            # |diferencia| < 1 punto no cuenta
BAJA_QUE_CORTA = 2.0    # una bajada ≥ 2 puntos corta la pasada y es un «no» en su tamaño
TAMANOS = ("pequeno", "mediano", "grande")
_METRICAS_DE_CIERRE = {"auroc", "accuracy", "rmse"}

#: Los valores de MATRIXAI_POLITICA_DE_ARBOLES que se miden: su L2 y los tamaños donde APLICA la política. `l2_*` es
#: C2 (los tres tamaños, enmienda 1); `c2b_*` es C2b (enmienda 5, Roberto 02-10: en «pequeño», el motor de HOY).
L2_DE_LA_POLITICA = {"l2_0": 0.0, "l2_1": 1.0, "c2b_l2_0": 0.0, "c2b_l2_1": 1.0}
TAMANOS_DE_LA_POLITICA = {"l2_0": TAMANOS, "l2_1": TAMANOS,
                          "c2b_l2_0": ("mediano", "grande"), "c2b_l2_1": ("mediano", "grande")}


class Incomparable(Exception):
    """R1 y C2 no midieron lo mismo en un conjunto (otra línea base, otra métrica): la pasada para."""


def _medias(rec: dict) -> tuple[str | None, dict[str, float]]:
    m = rec.get("media_de_la_seleccion") or {}
    valores = {x["motor"]: float(x["valor"]) for x in m.get("motores", [])
               if x.get("compite") and x.get("valor") is not None}
    return m.get("metrica"), valores


def completa(rec: dict | None) -> bool:
    if not rec or rec.get("estado") != "completed" or not rec.get("campeon"):
        return False
    _, valores = _medias(rec)
    return rec["campeon"] in valores


def puntos(rec: dict) -> float | None:
    """La cifra del campeón en puntos ×100 (ver la cabecera); None si el conjunto no completa."""
    if not completa(rec):
        return None
    metrica, valores = _medias(rec)
    if metrica not in _METRICAS_DE_CIERRE:
        raise Incomparable(f"{rec.get('nombre')}: métrica {metrica!r} sin regla de cierre registrada")
    v = valores[rec["campeon"]]
    if metrica in ("auroc", "accuracy"):
        return 100.0 * v
    base = valores.get("baseline")
    if not base:
        raise Incomparable(f"{rec.get('nombre')}: regresión sin RMSE de la línea base")
    return 100.0 * (1.0 - (v / base) ** 2)


def linea_base(rec: dict | None) -> tuple[str | None, float | None]:
    if not rec:
        return None, None
    metrica, valores = _medias(rec)
    return metrica, valores.get("baseline")


def tamano_declarado(rec: dict) -> str | None:
    """El tamaño que declararon los intentos de árboles (`perfiles_declarados` lo escribe el guion); si los
    pliegues no coinciden, el de la mayoría, y `tamano_unanime` lo dice."""
    tams = [p["tamano"] for p in rec.get("perfiles_declarados") or [] if p.get("tamano") in TAMANOS]
    return Counter(tams).most_common(1)[0][0] if tams else None


def comparar(r1: dict | None, c2: dict) -> dict:
    """Un conjunto: R1 frente a C2. `clase` ∈ sube | baja | baja_2 | igual | deja_de_completar |
    empieza_a_completar | ninguno_completa. Lanza `Incomparable` si la línea base difiere."""
    nombre = c2.get("nombre")
    c1, cc = completa(r1), completa(c2)
    if c1 and cc:
        m1, b1 = linea_base(r1)
        m2, b2 = linea_base(c2)
        if m1 != m2:
            raise Incomparable(f"{nombre}: métrica {m1!r} en R1 y {m2!r} en C2")
        if b1 is None or b2 is None or abs(b1 - b2) > 1e-9 * max(1.0, abs(b1)):
            raise Incomparable(f"{nombre}: la línea base no es la misma (R1 {b1!r}, C2 {b2!r}): otros pliegues")
        p1, p2 = puntos(r1), puntos(c2)
        d = p2 - p1
        clase = ("baja_2" if d <= -BAJA_QUE_CORTA else "baja" if d <= -UMBRAL else
                 "sube" if d >= UMBRAL else "igual")
        return {"nombre": nombre, "clase": clase, "r1": p1, "c2": p2, "diferencia": d,
                "tamano": tamano_declarado(c2)}
    clase = ("deja_de_completar" if c1 else "empieza_a_completar" if cc else "ninguno_completa")
    return {"nombre": nombre, "clase": clase, "r1": puntos(r1) if c1 else None,
            "c2": puntos(c2) if cc else None, "diferencia": None, "tamano": tamano_declarado(c2)}


def paridad_fuera_de_la_politica(comparacion: dict, politica: str | None) -> str | None:
    """C2b (enmienda 5): un conjunto de un tamaño donde la política NO aplica entrena con el motor de HOY, así que su
    cifra tiene que ser la de R1 EXACTA (o los dos sin completar). Si no, el instrumento no mide lo que dice: PARO.
    `None` si cuadra o si la política aplica en ese tamaño (o no hay política)."""
    if politica is None or comparacion.get("tamano") in TAMANOS_DE_LA_POLITICA[politica]:
        return None
    if comparacion["clase"] == "ninguno_completa":
        return None
    d = comparacion.get("diferencia")
    if comparacion["clase"] == "igual" and d is not None and abs(d) < 1e-9:
        return None
    return (f"{comparacion['nombre']} ({comparacion.get('tamano')}, fuera de la política) no es IDÉNTICO a R1: "
            f"{comparacion['clase']} {d!r}")


def corta(comparacion: dict, politica: str | None = None) -> str | None:
    """La regla 9: el motivo para cortar la pasada tras este conjunto, o None. Con C2b, un conjunto de un tamaño
    donde la política no aplica nunca corta: es el control de paridad (`paridad_fuera_de_la_politica`)."""
    if politica is not None and comparacion.get("tamano") not in TAMANOS_DE_LA_POLITICA[politica]:
        return None
    if comparacion["clase"] == "baja_2":
        return f"{comparacion['nombre']} baja {comparacion['diferencia']:.2f} puntos (≥ {BAJA_QUE_CORTA:g})"
    if comparacion["clase"] == "deja_de_completar":
        return f"{comparacion['nombre']} deja de completar"
    return None


def veredicto(comparaciones: list[dict]) -> dict:
    """La regla 4 por tamaño. Un tamaño sin ningún conjunto que decida se queda con la configuración de hoy."""
    por = {t: [c for c in comparaciones if c["tamano"] == t] for t in TAMANOS}
    sin_tamano = [c["nombre"] for c in comparaciones if c["tamano"] not in TAMANOS]
    out = {}
    for t, cs in por.items():
        cl = Counter(c["clase"] for c in cs)
        suben = cl["sube"] + cl["empieza_a_completar"]
        bajan = cl["baja"] + cl["baja_2"]
        mejora = bool(cs) and suben > bajan and not cl["baja_2"] and not cl["deja_de_completar"]
        out[t] = {"conjuntos": [c["nombre"] for c in cs], "suben": suben, "bajan": bajan,
                  "bajan_2": cl["baja_2"], "dejan_de_completar": cl["deja_de_completar"],
                  "iguales": cl["igual"], "ninguno_completa": cl["ninguno_completa"],
                  "mejora": mejora, "se_queda": "el perfil" if mejora else "la configuración de hoy"}
    return {"por_tamano": out, "sin_tamano_declarado": sin_tamano}


def _main(argv: list[str]) -> int:
    r1 = json.loads(Path(argv[1]).read_text())["conjuntos"]
    c2 = json.loads(Path(argv[2]).read_text())
    comps = []
    for n, rec in c2["conjuntos"].items():
        comps.append(comparar(r1.get(n), rec))
        c = comps[-1]
        dif = "" if c["diferencia"] is None else f"{c['diferencia']:+.2f}"
        print(f"{n:34} {str(c['tamano']):8} {c['clase']:20} {dif}")
    print(json.dumps(veredicto(comps), ensure_ascii=False, indent=1))
    if c2.get("corte"):
        print("CORTADA:", c2["corte"])
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
