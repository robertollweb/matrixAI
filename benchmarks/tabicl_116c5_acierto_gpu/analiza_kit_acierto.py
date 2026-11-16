# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""116-C5 — compara el acierto de TabICL en GPU (el JSON que devuelve Roberto) con lo YA medido en CPU, pliegue a pliegue.

    python3 analiza_kit_acierto.py resultado_116c5_acierto_gpu.json [--salida-md analisis.md] [--salida-json analisis.json]

Se corre AQUÍ (servidor), no en el portátil. Lee la REGLA de `REGLA.md` (escrita el 2026-11-16 antes de que existiera
ningún dato) y SE NIEGA a correr si su sha256 no es el que cita este fichero: cambiar la regla exige cambiar también
`REGLA_SHA256` aquí, y eso queda en git.

Rivales (sin remedir; mismas particiones, demostrado por digest):
  - `lightgbm`, de `benchmarks/fase0/pasada_v2_113_resultado.json` (113-C4);
  - la red TabM (`matrixai.dense.tabm_cpu`), de `benchmarks/fase0/resultado_pasada_119_c3.json` (119-C3).
Control informativo (sin veredicto): TabICL en CPU de `benchmarks/fase0/resultado_116c2_tabicl.json` (116-C2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FASE0 = AQUI.parent / "fase0"
NUCLEO = AQUI.parent.parent
MOTORES = NUCLEO.parent / "matrixai-engines" / "src"
for _ruta in (str(FASE0), str(NUCLEO), str(MOTORES)):
    if _ruta not in sys.path:
        sys.path.insert(0, _ruta)

#: La huella de REGLA.md. Si el fichero cambia, esto cambia a la vez (y se ve en git); si no, el analizador no corre.
REGLA_SHA256 = "a7fa88a988f7cb08e01f5509167f338b4ca55bab17425f342125f7d55713f259"
#: Los números de la regla (REGLA.md, «Cómo se compara»).
MARGEN_DE_EQUIVALENCIA = 0.005
SEMILLA_DEL_BOOTSTRAP = 0
REMUESTRAS_DEL_BOOTSTRAP = 1000
PROTOCOLO_ESPERADO = ("113-C0.v2", "1b4e902e")
METRICAS_MAS_ES_MEJOR = {"auroc": True, "accuracy": True, "r2": True}

RUTA_V2 = FASE0 / "pasada_v2_113_resultado.json"
RUTA_TABM = FASE0 / "resultado_pasada_119_c3.json"
RUTA_CPU = FASE0 / "resultado_116c2_tabicl.json"
MOTOR_TABM = "matrixai.dense.tabm_cpu"
MOTOR_LGBM = "lightgbm"
MOTOR_TABICL = "tabicl.v2"


def sha256_de(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def exigir_la_regla() -> str:
    regla = AQUI / "REGLA.md"
    if not regla.exists():
        raise SystemExit("falta REGLA.md: sin la regla escrita ANTES no se analiza")
    calculado = sha256_de(regla)
    if calculado != REGLA_SHA256:
        raise SystemExit(f"REGLA.md ha cambiado (sha256 {calculado}, el analizador cita {REGLA_SHA256}). La regla se "
                         f"escribe antes de los datos: si se cambia a propósito, hay que cambiar AQUÍ la huella y dejarlo "
                         f"en git, no en silencio.")
    return calculado


def importar_protocolo():
    import protocolo as p  # noqa: PLC0415
    return p


# ---------------------------------------------------------------------------
# lectura por pliegue
# ---------------------------------------------------------------------------

def por_pliegue(registros: list[dict], *, dataset: str, motor: str, metric_id: str, estados_validos) -> dict[tuple, float]:
    return {(r["repeticion"], r["pliegue"]): r[metric_id] for r in registros
            if r.get("dataset") == dataset and r.get("motor") == motor and r.get("estado") in estados_validos
            and r.get(metric_id) is not None}


def media(mapa: dict) -> float | None:
    return sum(mapa.values()) / len(mapa) if mapa else None


# ---------------------------------------------------------------------------
# el veredicto de UN conjunto frente a UN rival (REGLA.md, punto 4 y 5)
# ---------------------------------------------------------------------------

def veredicto(protocolo, tabicl_pp: dict, rival_pp: dict, *, n_esperado: int, digest_ok: bool | None,
              motivo_digest: str | None) -> dict:
    if not digest_ok:
        return {"veredicto": "no_comparable", "motivo": motivo_digest or "el digest de los pliegues no coincide"}
    if len(tabicl_pp) != n_esperado:
        return {"veredicto": "no_comparable",
                "motivo": f"TabICL GPU completó {len(tabicl_pp)} de {n_esperado} pliegues del conjunto"}
    faltan_del_rival = sorted(set(tabicl_pp) - set(rival_pp))
    if faltan_del_rival:
        return {"veredicto": "no_comparable", "motivo": f"al rival le faltan {len(faltan_del_rival)} pliegues: {faltan_del_rival}"}
    comunes = sorted(set(tabicl_pp) & set(rival_pp))
    if len(comunes) < 2:
        return {"veredicto": "no_comparable", "motivo": f"menos de 2 pliegues comunes ({len(comunes)})"}
    d = [tabicl_pp[k] - rival_pp[k] for k in comunes]
    bajo, alto = protocolo._intervalo_pareado(d, semilla=SEMILLA_DEL_BOOTSTRAP, remuestras=REMUESTRAS_DEL_BOOTSTRAP)
    cabe = bajo >= -MARGEN_DE_EQUIVALENCIA and alto <= MARGEN_DE_EQUIVALENCIA
    if bajo > 0 and not cabe:
        v, sub = "gana", None
    elif alto < 0 and not cabe:
        v, sub = "pierde", None
    else:
        v, sub = "empate", ("equivalente" if cabe else "no_concluyente")
    return {"veredicto": v, "subclase": sub, "n_pliegues": len(comunes), "media_tabicl": media({k: tabicl_pp[k] for k in comunes}),
            "media_rival": media({k: rival_pp[k] for k in comunes}), "diferencia_media": sum(d) / len(d),
            "intervalo": [bajo, alto], "cabe_en_el_margen": cabe}


def analizar(kit: dict, *, ruta_v2: Path = RUTA_V2, ruta_tabm: Path = RUTA_TABM, ruta_cpu: Path | None = RUTA_CPU) -> dict:
    protocolo = importar_protocolo()
    estados = protocolo.ESTADOS_QUE_CUENTAN_COMO_MEDIDA
    if (kit["protocolo_version"], kit["protocolo_digest_sha256"][:8]) != PROTOCOLO_ESPERADO:
        raise SystemExit(f"el JSON del kit es del protocolo {kit['protocolo_version']} {kit['protocolo_digest_sha256'][:8]}, "
                         f"no del {PROTOCOLO_ESPERADO}")
    if kit.get("regla_sha256") not in (None, REGLA_SHA256):
        raise SystemExit("el kit se midió con otra REGLA.md distinta de la que cita este analizador")
    if any(r.get("sellado") for r in kit["resultados"]):
        raise SystemExit("el JSON del kit contiene un conjunto SELLADO: no se analiza")
    v2 = json.loads(ruta_v2.read_text(encoding="utf-8"))
    tabm = json.loads(ruta_tabm.read_text(encoding="utf-8"))
    cpu = json.loads(ruta_cpu.read_text(encoding="utf-8")) if ruta_cpu and ruta_cpu.exists() else None
    for nombre, rival in (("v2", v2), ("119-C3", tabm)):
        if rival.get("protocolo_digest_sha256", "")[:8] not in ("", PROTOCOLO_ESPERADO[1]) and nombre == "119-C3":
            raise SystemExit(f"{nombre} es de otro protocolo")

    conjuntos = {}
    for dataset in kit["conjuntos_pedidos"]:
        metric_id = kit["metrica_de_cierre_por_dataset"][dataset]
        if metric_id not in METRICAS_MAS_ES_MEJOR:
            raise SystemExit(f"{dataset}: métrica {metric_id} sin dirección declarada en la regla")
        regimen = kit["regimen_por_dataset"][dataset]
        n_esperado = regimen["n_pliegues"]
        tabicl_pp = por_pliegue(kit["resultados"], dataset=dataset, motor=MOTOR_TABICL, metric_id=metric_id, estados_validos=estados)
        if kit.get("es_subconjunto_de_prueba"):  # una prueba corre solo algunos pliegues: lo esperado son los intentados
            n_esperado = len({(r["repeticion"], r["pliegue"]) for r in kit["resultados"] if r.get("dataset") == dataset})
        fallos = [{"repeticion": r["repeticion"], "pliegue": r["pliegue"], "estado": r["estado"], "motivo": r.get("motivo")}
                  for r in kit["resultados"] if r.get("dataset") == dataset and r.get("estado") not in estados]
        digest_kit = kit["pliegues_digest_por_dataset"].get(dataset)
        digest_119 = (tabm.get("pliegues_digest_por_dataset") or {}).get(dataset)
        plan_v2 = (v2["particion_por_dataset"].get(dataset) or {}).get("plan_digest")
        plan_kit = kit["plan_digest_por_dataset"].get(dataset)
        comprobaciones = {"pliegues_digest_kit": digest_kit, "pliegues_digest_119": digest_119,
                          "plan_digest_kit": plan_kit, "plan_digest_v2": plan_v2}
        ok_tabm = digest_kit is not None and digest_kit == digest_119
        ok_v2 = plan_kit is not None and plan_kit == plan_v2
        lgbm_pp = por_pliegue(v2["resultados"], dataset=dataset, motor=MOTOR_LGBM, metric_id=metric_id, estados_validos=estados)
        tabm_pp = por_pliegue(tabm["resultados"], dataset=dataset, motor=MOTOR_TABM, metric_id=metric_id, estados_validos=estados)
        control = None
        if cpu is not None:
            cpu_pp = por_pliegue(cpu["resultados"], dataset=dataset, motor=MOTOR_TABICL, metric_id=metric_id, estados_validos=estados)
            comunes = sorted(set(cpu_pp) & set(tabicl_pp))
            if comunes:
                control = {"n_pliegues": len(comunes), "media_gpu_menos_cpu": sum(tabicl_pp[k] - cpu_pp[k] for k in comunes) / len(comunes),
                           "max_abs_diferencia": max(abs(tabicl_pp[k] - cpu_pp[k]) for k in comunes)}
        conjuntos[dataset] = {
            "metrica": metric_id, "cubo": regimen.get("cubo"), "n_pliegues_esperados": n_esperado,
            "n_pliegues_tabicl_completados": len(tabicl_pp), "fallos_de_tabicl": fallos,
            "digests": comprobaciones, "media_tabicl_gpu": media(tabicl_pp),
            "frente_a_tabm": veredicto(protocolo, tabicl_pp, tabm_pp, n_esperado=n_esperado, digest_ok=ok_tabm,
                                       motivo_digest="los pliegues del kit no coinciden con los de 119-C3 (digest)"),
            "frente_a_lightgbm": veredicto(protocolo, tabicl_pp, lgbm_pp, n_esperado=n_esperado, digest_ok=ok_v2,
                                           motivo_digest="el plan del kit no coincide con el de la v2 (digest)"),
            "control_tabicl_gpu_frente_a_cpu": control,
        }
    return {"regla_sha256": REGLA_SHA256, "margen_de_equivalencia": MARGEN_DE_EQUIVALENCIA, "kit_maquina": kit.get("maquina"),
            "es_subconjunto_de_prueba": kit.get("es_subconjunto_de_prueba"), "es_sonda_cpu": kit.get("es_sonda_cpu"),
            "conjuntos": conjuntos, "recuento": recuento(conjuntos)}


def recuento(conjuntos: dict) -> dict:
    salida = {}
    for rival in ("frente_a_tabm", "frente_a_lightgbm"):
        c = {"gana": 0, "pierde": 0, "empate_equivalente": 0, "empate_no_concluyente": 0, "no_comparable": 0}
        for datos in conjuntos.values():
            v = datos[rival]
            if v["veredicto"] == "empate":
                c["empate_" + v["subclase"]] += 1
            else:
                c[v["veredicto"]] += 1
        salida[rival] = c
    return salida


# ---------------------------------------------------------------------------
# informe en texto
# ---------------------------------------------------------------------------

def _celda(v: dict) -> str:
    if v["veredicto"] == "no_comparable":
        return "NO COMPARABLE"
    lo, hi = v["intervalo"]
    etiqueta = ("GANA" if v["veredicto"] == "gana" else "PIERDE" if v["veredicto"] == "pierde"
                else f"empate ({v['subclase'].replace('_', ' ')})")
    return f"{v['diferencia_media']:+.4f} [{lo:+.4f}, {hi:+.4f}] {etiqueta}"


def a_markdown(a: dict) -> str:
    L = [f"# Kit 116-C5: TabICL (GPU) frente a TabM y a LightGBM", "",
         f"Regla: `REGLA.md`, sha256 `{a['regla_sha256']}`. Margen de equivalencia ±{a['margen_de_equivalencia']}. "
         f"Diferencia = TabICL GPU − rival (métrica cruda, más es mejor), media de pliegues emparejados con su intervalo "
         f"bootstrap al 95 %.", ""]
    if a.get("es_sonda_cpu") or a.get("es_subconjunto_de_prueba"):
        L += ["**ATENCIÓN: esto es una SONDA/PRUEBA (CPU o subconjunto); no es la medición y no se lee como resultado.**", ""]
    L += ["| conjunto | métrica | pliegues | TabICL GPU | vs TabM | vs LightGBM | control GPU−CPU |", "|---|---|---|---|---|---|---|"]
    for nombre, d in a["conjuntos"].items():
        c = d["control_tabicl_gpu_frente_a_cpu"]
        ctrl = f"{c['media_gpu_menos_cpu']:+.4f} (máx abs Δ {c['max_abs_diferencia']:.4f}, n={c['n_pliegues']})" if c else "—"
        media_t = f"{d['media_tabicl_gpu']:.4f}" if d["media_tabicl_gpu"] is not None else "—"
        L.append(f"| {nombre} | {d['metrica']} | {d['n_pliegues_tabicl_completados']}/{d['n_pliegues_esperados']} | {media_t} | "
                 f"{_celda(d['frente_a_tabm'])} | {_celda(d['frente_a_lightgbm'])} | {ctrl} |")
    L += ["", "## TabICL frente a TabM y frente a LightGBM: gana / pierde / empate dentro del margen", "",
          "| rival | gana | pierde | empate equivalente | empate no concluyente | no comparable |", "|---|---|---|---|---|---|"]
    for rival, etiqueta in (("frente_a_tabm", "TabM (119-C3)"), ("frente_a_lightgbm", "LightGBM (113-C4)")):
        c = a["recuento"][rival]
        L.append(f"| {etiqueta} | {c['gana']} | {c['pierde']} | {c['empate_equivalente']} | {c['empate_no_concluyente']} | {c['no_comparable']} |")
    nc = [(n, r, d[r]["motivo"]) for n, d in a["conjuntos"].items() for r in ("frente_a_tabm", "frente_a_lightgbm")
          if d[r]["veredicto"] == "no_comparable"]
    if nc:
        L += ["", "## No comparables (con su motivo)", ""] + [f"- {n} ({r.replace('frente_a_', '')}): {m}" for n, r, m in nc]
    fallos = [(n, f) for n, d in a["conjuntos"].items() for f in d["fallos_de_tabicl"]]
    if fallos:
        L += ["", "## Pliegues de TabICL que no completaron", ""] + [
            f"- {n} rep={f['repeticion']} pliegue={f['pliegue']}: {f['estado']} {f['motivo'] or ''}" for n, f in fallos]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("resultado", help="resultado_116c5_acierto_gpu.json devuelto por el kit")
    ap.add_argument("--salida-md", default=None)
    ap.add_argument("--salida-json", default=None)
    ap.add_argument("--v2", default=str(RUTA_V2))
    ap.add_argument("--tabm", default=str(RUTA_TABM))
    ap.add_argument("--cpu", default=str(RUTA_CPU))
    a = ap.parse_args(argv)
    exigir_la_regla()
    kit = json.loads(Path(a.resultado).read_text(encoding="utf-8"))
    analisis = analizar(kit, ruta_v2=Path(a.v2), ruta_tabm=Path(a.tabm), ruta_cpu=Path(a.cpu))
    texto = a_markdown(analisis)
    print(texto)
    if a.salida_md:
        Path(a.salida_md).write_text(texto, encoding="utf-8")
    if a.salida_json:
        Path(a.salida_json).write_text(json.dumps(analisis, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
