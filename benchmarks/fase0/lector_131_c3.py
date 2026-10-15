#!/usr/bin/env python3
"""131 · C3 — el LECTOR de la regla del C3 (contrato 131, entrada «LA REGLA DEL C3 CON LA B, ESCRITA ANTES DE MEDIR»).
No mide nada: lee JSON y aplica la regla tal como está escrita.

    python3 lector_131_c3.py DIR [--c5-diez c5_129_diez.json] [--c5-grandes c5_129_grandes.json] [--json salida.json]

DIR trae lo que dejó la cola (nombres que fija `encolar_c3_131.sh`):
    c3_131_e_pub_s{0..4}.json     grupo E, brazo PUBLICADO (v2.13), semilla N    (dresses-sales, pc1, us_crime)
    c3_131_e_c2_s{0..4}.json      grupo E, brazo C2 del 131, semilla N
    c3_131_s_c2_diez.json         grupo S, brazo C2, la semilla de siempre (sin --semilla): los siete medianos del S
    c3_131_s_c2_grandes.json      grupo S, brazo C2: KDDCup09_appetency, APSFailure, Allstate_Claims_Severity
y el C5 del 129 (`c5_129_diez.json`, `c5_129_grandes.json`) es «lo que da el publicado» para el grupo S.

LA REGLA (literal; el umbral, 1,0 sobre el SE corregido, no es de aquí)
- Cifra = la del TEST del campeón ×100 (`veredicto_120_c3p.cifra_de_test`: AUROC en binaria, exactitud en multiclase, R² en
  regresión; mayor es mejor).
- GRUPO E (dresses-sales, pc1, us_crime), 5 semillas (0–4) en cada brazo. d_s = C2 − publicado, pareada por semilla.
  EMPEORA si la media de d baja MÁS de 2 puntos Y MÁS de dos errores típicos de d (desviación de las 5 ÷ √5).
  Desviación = MUESTRAL (ddof = 1); la regla dice «desviación de las 5» sin más: si con ddof = 0 el veredicto de un conjunto
  fuera otro, se DICE («sensible a ddof»), no se esconde.
- GRUPO S (los otros diez), semilla de siempre, solo el brazo C2: BIT A BIT frente al C5 (campeón, `media_de_la_seleccion`
  entera y el test entero).
- ENTRA si: ninguno de E empeora; ninguno de los 13 deja de terminar; S sale bit a bit; y el control de árboles cuadra
  (los 4 ejemplos a 4 decimales —`referencia_120_r0.CONTROL`— sobre el brazo C2 con la semilla 0: dresses-sales, pc1 y
  us_crime del grupo E y climate del S).
- SE DECLARA, sin decidir: cuántas veces cambia el campeón entre semillas en cada brazo (= semillas 1–4 cuyo campeón difiere
  del de la semilla 0; la robustez a la partición del 129) y la media de d con su error típico.

Un incumplimiento ya es NO ENTRA aunque falten cosas; faltando datos y sin incumplimiento: INCOMPLETO, nunca ENTRA.
Salida: 0 ENTRA, 1 NO ENTRA, 2 INCOMPLETO.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent


def _por_ruta(nombre, fichero):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / fichero)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V3 = _por_ruta("veredicto_120_c3p_l131", "veredicto_120_c3p.py")
R0 = _por_ruta("referencia_120_r0_l131", "referencia_120_r0.py")

GRUPO_E = ["dresses-sales", "pc1", "us_crime"]
SEMILLAS = (0, 1, 2, 3, 4)
GRUPO_S = ["climate-model-simulation-crashes", "micro-mass", "mfeat-factors", "wilt", "pendigits", "house_16H",
           "Internet-Advertisements", "KDDCup09_appetency", "APSFailure", "Allstate_Claims_Severity"]
UMBRAL_PUNTOS = 2.0
UMBRAL_ERRORES_TIPICOS = 2.0


# ------------------------------------------------------------------------------------------------ lectura
def _conj(ruta):
    p = Path(ruta)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def leer(directorio, c5_diez, c5_grandes):
    """(e, s, c5, procedencias): e[brazo][semilla] = {nombre: registro}; s = {nombre: registro}."""
    d = Path(directorio)
    e = {"pub": {}, "c2": {}}
    prov = {}
    for brazo in e:
        for sem in SEMILLAS:
            doc = _conj(d / f"c3_131_e_{brazo}_s{sem}.json")
            if doc is not None:
                e[brazo][sem] = doc.get("conjuntos") or {}
                prov[f"e_{brazo}_s{sem}"] = doc.get("procedencia") or {}
    s, c5 = {}, {}
    for f in ("c3_131_s_c2_diez.json", "c3_131_s_c2_grandes.json"):
        doc = _conj(d / f)
        if doc is not None:
            s.update(doc.get("conjuntos") or {})
            prov[f[:-5]] = doc.get("procedencia") or {}
    for f in (c5_diez, c5_grandes):
        doc = _conj(f)
        if doc is not None:
            c5.update(doc.get("conjuntos") or {})
    return e, s, c5, prov


# ------------------------------------------------------------------------------------------------ piezas
def cifra(rec):
    """La cifra del test ×100, o None si no completa. Un registro roto (`Incomparable`) cuenta como instrumento roto."""
    if not rec:
        return None
    return V3.cifra_de_test(rec)


def regla_e(ds, *, ddof=1):
    """(media, error típico, empeora) de las diferencias pareadas. Empeora = media < −2 Y −media > 2·error típico."""
    n = len(ds)
    media = statistics.fmean(ds)
    et = statistics.stdev(ds) / math.sqrt(n) if ddof == 1 else statistics.pstdev(ds) / math.sqrt(n)
    return media, et, bool(media < -UMBRAL_PUNTOS and -media > UMBRAL_ERRORES_TIPICOS * et)


def cambios_de_campeon(por_semilla):
    """Cuántas de las semillas 1–4 tienen un campeón distinto del de la semilla 0 (None si falta la 0)."""
    if 0 not in por_semilla:
        return None
    return sum(1 for s, c in por_semilla.items() if s != 0 and c != por_semilla[0])


def bit_a_bit(rec, ref):
    """(¿idénticos?, qué difiere): campeón, `media_de_la_seleccion` entera y el test entero."""
    dif = [k for k in ("campeon", "media_de_la_seleccion", "test") if rec.get(k) != ref.get(k)]
    return not dif, dif


def _completa(rec):
    return bool(rec) and V3.completa(rec)


# ------------------------------------------------------------------------------------------------ la regla
def aplicar(e, s, c5, prov=None):
    """Aplica la regla. Devuelve un dict con `veredicto` (ENTRA | NO ENTRA | INCOMPLETO), `incumple`, `falta`, las filas
    del grupo E y del S, el control de árboles y lo que se declara."""
    incumple, falta, avisos = [], [], []
    prov = prov or {}

    # --- la procedencia de lo medido: la semilla que dice cada fichero es la que toca
    for clave, p in prov.items():
        if clave.startswith("e_"):
            sem = int(clave.rsplit("_s", 1)[1])
            if (p.get("semilla") or 0) != sem or p.get("modo") != "pcpu":
                falta.append(f"{clave}: procedencia con semilla {p.get('semilla')!r} / modo {p.get('modo')!r} "
                             f"(se esperaba la {sem} y pcpu): no es esta medida")
        elif (p.get("semilla") or 0) != 0 or p.get("modo") != "pcpu":
            falta.append(f"{clave}: el grupo S va con la semilla de siempre y pcpu, y trae "
                         f"{p.get('semilla')!r} / {p.get('modo')!r}")

    # --- grupo E
    filas_e = {}
    for n in GRUPO_E:
        fila = {"d": {}, "pub": {}, "c2": {}, "campeon": {"pub": {}, "c2": {}}}
        for sem in SEMILLAS:
            rp, rc = e["pub"].get(sem, {}).get(n), e["c2"].get(sem, {}).get(n)
            if rc is not None and not _completa(rc):
                incumple.append(f"{n} semilla {sem}: el brazo C2 DEJA DE TERMINAR ({rc.get('estado')})")
            if rc is None or rp is None:
                falta.append(f"{n} semilla {sem}: falta {'el C2' if rc is None else 'el publicado'}")
                continue
            if not _completa(rp):
                falta.append(f"{n} semilla {sem}: el publicado no completa ({rp.get('estado')}): sin pareja")
                continue
            if not _completa(rc):
                continue
            try:
                p, c = cifra(rp), cifra(rc)
            except V3.Incomparable as exc:
                falta.append(f"{n} semilla {sem}: registro roto: {exc}")
                continue
            fila["pub"][sem], fila["c2"][sem], fila["d"][sem] = p, c, c - p
            fila["campeon"]["pub"][sem], fila["campeon"]["c2"][sem] = rp.get("campeon"), rc.get("campeon")
        fila["cambios_de_campeon"] = {b: cambios_de_campeon(fila["campeon"][b]) for b in ("pub", "c2")}
        if len(fila["d"]) == len(SEMILLAS):
            ds = [fila["d"][s_] for s_ in SEMILLAS]
            media, et, emp = regla_e(ds)
            _, _, emp0 = regla_e(ds, ddof=0)
            fila.update(media_d=media, error_tipico_d=et, empeora=emp, empeora_ddof0=emp0)
            if emp:
                incumple.append(f"{n}: EMPEORA (media de d {media:+.3f} puntos, error típico {et:.3f})")
            if emp != emp0:
                avisos.append(f"{n}: SENSIBLE a ddof (con ddof=1 empeora={emp}, con ddof=0 empeora={emp0})")
        else:
            fila.update(media_d=None, error_tipico_d=None, empeora=None, empeora_ddof0=None)
        filas_e[n] = fila

    # --- grupo S
    filas_s = {}
    for n in GRUPO_S:
        rec, ref = s.get(n), c5.get(n)
        if ref is None:
            falta.append(f"{n}: falta el C5 de referencia")
            continue
        if rec is None:
            falta.append(f"{n}: no medido")
            continue
        if not _completa(rec):
            incumple.append(f"{n}: el brazo C2 DEJA DE TERMINAR ({rec.get('estado')})")
            filas_s[n] = {"bit_a_bit": False, "difiere": ["no completa"]}
            continue
        ok, dif = bit_a_bit(rec, ref)
        filas_s[n] = {"bit_a_bit": ok, "difiere": dif, "campeon_c2": rec.get("campeon"), "campeon_c5": ref.get("campeon")}
        if not ok:
            incumple.append(f"{n}: NO es bit a bit el C5 (difiere: {', '.join(dif)})")

    # --- el control de árboles: los 4 ejemplos, brazo C2, semilla 0
    cuatro = {n: (e["c2"].get(0, {}).get(n) if n in GRUPO_E else s.get(n)) for n in R0.CONTROL}
    if all(cuatro.values()):
        ok, lineas = R0.controlar(cuatro)
        control = {"cuadra": ok, "lineas": lineas}
        if not ok:
            incumple.append("el control de árboles NO cuadra (los 4 ejemplos a 4 decimales)")
    else:
        control = {"cuadra": None, "lineas": [f"faltan: {[n for n, r in cuatro.items() if not r]}"]}
        falta.append("el control de árboles: faltan registros de los 4 ejemplos")

    # --- el publicado de hoy contra el C5 (informativo: dice si «el publicado» es el C5 de verdad)
    pub0 = {}
    for n in GRUPO_E:
        rp, ref = e["pub"].get(0, {}).get(n), c5.get(n)
        if rp is not None and ref is not None and _completa(rp):
            pub0[n] = bit_a_bit(rp, ref)[0]
    if pub0 and not all(pub0.values()):
        avisos.append(f"el publicado (v2.13) con la semilla 0 NO es bit a bit el C5 en {[n for n, v in pub0.items() if not v]}: "
                      f"el grupo S se compara con el C5, no con el publicado de hoy")

    veredicto = "NO ENTRA" if incumple else ("INCOMPLETO" if falta else "ENTRA")
    return {"veredicto": veredicto, "incumple": incumple, "falta": falta, "avisos": avisos, "grupo_e": filas_e,
            "grupo_s": filas_s, "control_de_arboles": control, "publicado_semilla0_contra_c5": pub0}


def informe(r):
    out = ["GRUPO E (C2 − publicado, puntos de test, por semilla 0–4)"]
    for n, f in r["grupo_e"].items():
        ds = " ".join(f"{f['d'][s]:+.2f}" if s in f["d"] else "  ·  " for s in SEMILLAS)
        resumen = ("media %+.3f ± %.3f  %s" % (f["media_d"], f["error_tipico_d"], "EMPEORA" if f["empeora"] else "no empeora")
                   if f["media_d"] is not None else "INCOMPLETO")
        out.append(f"  {n:14s} d = [{ds}]  {resumen}  | cambios de campeón (semillas 1–4 vs 0): "
                   f"publicado {f['cambios_de_campeon']['pub']}, C2 {f['cambios_de_campeon']['c2']}")
    out.append("GRUPO S (C2 contra el C5, bit a bit)")
    for n, f in r["grupo_s"].items():
        out.append(f"  {n:34s} {'BIT A BIT' if f['bit_a_bit'] else 'DIFIERE: ' + ', '.join(f['difiere'])}")
    out.append("CONTROL DE ÁRBOLES: " + {True: "CUADRA", False: "NO CUADRA", None: "sin datos"}[r["control_de_arboles"]["cuadra"]])
    out += [f"AVISO: {a}" for a in r["avisos"]]
    out += [f"INCUMPLE: {m}" for m in r["incumple"]] + [f"FALTA: {m}" for m in r["falta"]]
    out.append("C3: " + r["veredicto"])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directorio")
    ap.add_argument("--c5-diez", default=str(AQUI / "c5_129_diez.json"))
    ap.add_argument("--c5-grandes", default=str(AQUI / "c5_129_grandes.json"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    e, s, c5, prov = leer(a.directorio, a.c5_diez, a.c5_grandes)
    r = aplicar(e, s, c5, prov)
    print(informe(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit({"ENTRA": 0, "NO ENTRA": 1, "INCOMPLETO": 2}[r["veredicto"]])


if __name__ == "__main__":
    main()
