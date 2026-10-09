#!/usr/bin/env python3
"""128 · C4 — el resultado de la integración (los 13 con 6 GB) contra la referencia de la v2.12, con la regla del §2 del
contrato escrita ANTES de ver el resultado (09-10, mientras la cola lo mide):

    python3 comparar_c4_128.py referencia_128_rcpu_v212.json c4_128_rcpu.json --exacto   # sin TabM: bit a bit
    python3 comparar_c4_128.py referencia_128_pcpu_v212.json c4_128_pcpu.json            # el producto, con su suelo

- **`--exacto`** (sin TabM ni densa, determinista): los que terminaban en la referencia tienen que salir IGUALES al
  último decimal —mismo campeón, misma media de la selección de cada motor que compite, misma cifra del test— (el C1 es
  solo memoria: paridad bit a bit). Los que el C2 toca (más de 20 M de celdas: solo Allstate) no terminaban, así que no
  tienen con qué compararse: cuentan como «empieza a completar».
- **Sin `--exacto`** (el producto con TabM, que para por reloj): la regla del 120 que Roberto aprobó para el 128 (E2) —
  ninguno baja más que el SUELO DE RUIDO (el mayor de las dos pasadas, medido repitiendo los 4 de control), ninguno deja
  de terminar—, y un cambio de campeón se ENSEÑA para explicarlo con lo medido (la regla de Roberto: «cambia de campeón
  sin explicación medida» también descarta): el guion no lo puede explicar, así que lo marca «revisar».
- En las dos: APSFailure y Allstate tienen que TERMINAR CON CAMPEÓN (el objetivo del contrato).

La cifra del test, la de `veredicto_120_c3p.cifra_de_test` (campeón, test independiente, métrica de cierre ×100); las
medias, las de `veredicto_120._medias`. Un solo sitio para cada cosa: se importan por ruta.

Código de salida: 0 CUMPLE, 1 NO CUMPLE, 2 revisar (solo cambios de campeón por explicar), 3 incomparable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
GRANDES = ("APSFailure", "Allstate_Claims_Severity")
EXACTO = 1e-12


def _cargar_por_ruta(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v3p = _cargar_por_ruta("veredicto_120_c3p", AQUI / "veredicto_120_c3p.py")
v120 = _cargar_por_ruta("veredicto_120", AQUI / "veredicto_120.py")
Incomparable = v3p.Incomparable


def suelo_de(doc: dict) -> float | None:
    s = doc.get("suelo_de_ruido")
    return float(s) if isinstance(s, (int, float)) else None


def comparar_uno(nombre: str, ref: dict | None, nuevo: dict | None, *, exacto: bool, suelo: float) -> dict:
    """Un conjunto. `clase` ∈ igual | dentro_del_suelo | sube | baja | deja_de_completar | empieza_a_completar |
    ninguno_completa | falta (no se midió en el nuevo); con `exacto`, `igual` exige además las medias de todos los motores
    y el campeón idénticos (si no, `distinto`)."""
    c_ref, c_nuevo = v3p.completa(ref), v3p.completa(nuevo)
    base = {"nombre": nombre, "campeon_ref": (ref or {}).get("campeon"), "campeon_nuevo": (nuevo or {}).get("campeon"),
            "memoria_pico_ref_mb": (ref or {}).get("memoria_pico_mb"),
            "memoria_pico_nuevo_mb": (nuevo or {}).get("memoria_pico_mb"),
            "estado_ref": (ref or {}).get("estado"), "estado_nuevo": (nuevo or {}).get("estado")}
    if nuevo is None:
        return {**base, "clase": "falta"}
    if not (c_ref and c_nuevo):
        clase = "deja_de_completar" if c_ref else "empieza_a_completar" if c_nuevo else "ninguno_completa"
        return {**base, "clase": clase, "test_ref": v3p.cifra_de_test(ref) if c_ref else None,
                "test_nuevo": v3p.cifra_de_test(nuevo) if c_nuevo else None}
    if ref.get("tarea") != nuevo.get("tarea"):
        raise Incomparable(f"{nombre}: tarea {ref.get('tarea')!r} en la referencia y {nuevo.get('tarea')!r} en el nuevo")
    t_ref, t_nuevo = v3p.cifra_de_test(ref), v3p.cifra_de_test(nuevo)
    d = t_nuevo - t_ref
    m_ref, m_nuevo = v120._medias(ref)[1], v120._medias(nuevo)[1]
    motores = sorted(set(m_ref) | set(m_nuevo))
    dif_medias = {m: (None if m not in m_ref or m not in m_nuevo else m_nuevo[m] - m_ref[m]) for m in motores}
    medias_iguales = all(v is not None and abs(v) <= EXACTO for v in dif_medias.values())
    mismo_campeon = ref.get("campeon") == nuevo.get("campeon")
    if exacto:
        clase = "igual" if (mismo_campeon and medias_iguales and abs(d) <= EXACTO) else "distinto"
    elif abs(d) <= EXACTO:
        clase = "igual"
    elif abs(d) <= suelo:
        clase = "dentro_del_suelo"
    else:
        clase = "sube" if d > 0 else "baja"
    return {**base, "clase": clase, "test_ref": t_ref, "test_nuevo": t_nuevo, "diferencia": d,
            "mismo_campeon": mismo_campeon, "medias_iguales": medias_iguales, "diferencia_de_medias": dif_medias}


def veredicto(filas: list[dict], *, exacto: bool) -> dict:
    malas = {"distinto", "baja", "deja_de_completar", "falta"}
    no_cumple = [f["nombre"] for f in filas if f["clase"] in malas]
    grandes_sin_terminar = [f["nombre"] for f in filas if f["nombre"] in GRANDES and f["clase"] not in
                            ("empieza_a_completar", "igual", "dentro_del_suelo", "sube")]
    por_revisar = [f["nombre"] for f in filas
                   if not exacto and f.get("mismo_campeon") is False and f["clase"] not in malas]
    if no_cumple or grandes_sin_terminar:
        resultado, codigo = "NO CUMPLE", 1
    elif por_revisar:
        resultado, codigo = "REVISAR", 2
    else:
        resultado, codigo = "CUMPLE", 0
    return {"resultado": resultado, "codigo": codigo, "empeoran": no_cumple, "grandes_sin_terminar": grandes_sin_terminar,
            "cambios_de_campeon_por_explicar": por_revisar,
            "empiezan_a_completar": [f["nombre"] for f in filas if f["clase"] == "empieza_a_completar"],
            "suben": [f["nombre"] for f in filas if f["clase"] == "sube"]}


def comparar(ref_doc: dict, nuevo_doc: dict, *, exacto: bool) -> dict:
    s_ref, s_nuevo = suelo_de(ref_doc), suelo_de(nuevo_doc)
    if not exacto and (s_ref is None or s_nuevo is None):
        raise Incomparable(f"sin suelo de ruido en las dos pasadas (referencia {s_ref}, nuevo {s_nuevo}): sin él, la "
                           "regla del producto no se puede aplicar")
    suelo = 0.0 if exacto else max(s_ref, s_nuevo)
    ref, nuevo = ref_doc.get("conjuntos") or {}, nuevo_doc.get("conjuntos") or {}
    filas = [comparar_uno(n, ref.get(n), nuevo.get(n), exacto=exacto, suelo=suelo) for n in ref]
    extra = sorted(set(nuevo) - set(ref))
    return {"exacto": exacto, "suelo": suelo, "suelo_ref": s_ref, "suelo_nuevo": s_nuevo, "filas": filas,
            "conjuntos_de_mas_en_el_nuevo": extra, **veredicto(filas, exacto=exacto)}


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("referencia")
    ap.add_argument("nuevo")
    ap.add_argument("--exacto", action="store_true")
    ap.add_argument("--json", default=None, help="escribe la comparación entera aquí")
    a = ap.parse_args(argv)
    try:
        r = comparar(json.loads(Path(a.referencia).read_text()), json.loads(Path(a.nuevo).read_text()), exacto=a.exacto)
    except Incomparable as e:
        print(f"INCOMPARABLE: {e}")
        return 3
    print(f"{'bit a bit' if r['exacto'] else 'suelo ' + _fmt(r['suelo'], 2) + ' puntos'}  ·  {a.nuevo} contra {a.referencia}")
    for f in r["filas"]:
        print(f"  {f['nombre'][:34]:34} {f['clase']:20} campeón {str(f['campeon_ref'])[-14:]:>14} → "
              f"{str(f['campeon_nuevo'])[-14:]:14} test {_fmt(f.get('test_ref'))} → {_fmt(f.get('test_nuevo'))}  "
              f"memoria {_fmt(f['memoria_pico_ref_mb'], 0)} → {_fmt(f['memoria_pico_nuevo_mb'], 0)} MB")
    print(f"VEREDICTO: {r['resultado']}  · empeoran {r['empeoran']} · grandes sin terminar {r['grandes_sin_terminar']} · "
          f"campeón por explicar {r['cambios_de_campeon_por_explicar']} · empiezan a completar {r['empiezan_a_completar']}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1))
    return r["codigo"]


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
