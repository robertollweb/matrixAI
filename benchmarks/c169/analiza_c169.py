#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169 — EL ANALIZADOR: ¿la rejilla de 8 configuraciones de HGB mejora al perfil de hoy en los 6 conjuntos NUEVOS? La regla es `REGLA.md` (§2–§3).

    python3 benchmarks/c169/analiza_c169.py <resultado.json> [--json <salida.json>]

Veredictos: CONFIRMADA / NO CUMPLE / NO CONCLUYENTE (N < 3 no saturados) / SIN VEREDICTO (la medida no es completa o los sellos no cuadran) /
SONDA (una sonda no decide nada). Sale con 0 CONFIRMADA, 1 NO CUMPLE, 2 SIN VEREDICTO, 3 NO CONCLUYENTE y 4 SONDA (rep1: una sonda ya no sale con
el mismo 0 que una CONFIRMADA).

LOS SELLOS (rep1, I1 de la aud1). El orden es regla -> analizador -> sello del código:
  1. `REGLA_SHA256_SELLADA` (aquí abajo) es el sha256 de `REGLA.md`. Como está DENTRO de este fichero, el sha del analizador ya lo cubre.
  2. `SELLO_DEL_CODIGO.json` guarda el sha256 de este analizador, del medidor y de las dependencias que el veredicto importa
     (`codigo_de_ahora()`). Lo escribe `sellar_codigo_c169.py` (el supervisor, tras la última auditoría). NO se incluye a sí mismo.
  Sin ese sello, con algún valor nulo o con un fichero que ya no cuadra, el medidor no mide (salvo `--sonda`) y este analizador no da veredicto.

LO DECLARADO NO SE CREE (rep1, I2 de la aud1): la métrica de cierre por tarea, el presupuesto por cubo, la semilla 0, los 4 hilos, los digests
de la salida y `n_test` se COMPRUEBAN en cada registro contra lo que fija REGLA.md §1 (`problemas_de_lo_declarado`).

NO reescribe nada del 160 ni del 163-C2: importa `analiza_c160` (R, IC95 por bootstrap de pliegues, clasificar), `analiza_c163_c2` (la lectura de un
registro, el coste, las comprobaciones de brazo) y `protocolo._intervalo_pareado` (el MISMO bootstrap —1.000 remuestras, semilla 0, 95 %, percentiles—
para el IC de D).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parents[1]
for _ruta in (_AQUI, _RAIZ / "benchmarks" / "c163", _RAIZ / "benchmarks" / "c160", _RAIZ / "benchmarks" / "fase0", _RAIZ,
              _RAIZ.parent / "matrixai-engines" / "src"):
    if str(_ruta) not in sys.path:
        sys.path.insert(0, str(_ruta))

import analiza_c160 as a160  # noqa: E402
import analiza_c163_c2 as an163  # noqa: E402
import conjuntos_c169 as cj  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# LOS UMBRALES DE LA REGLA (REGLA.md §2–§3). Un número por línea; las pruebas los fijan contra REGLA.md.
# ----------------------------------------------------------------------------------------------------------------------
UMBRAL_DE_SATURACION = 0.98          # saturado: media de la métrica de cierre de hgb_hoy sobre sus 5 pliegues >= 0,98
N_MINIMO_NO_SATURADOS = 3            # (0) potencia mínima
A_REDUCCION_MEDIA_MIN = 0.03         # (a) media de R sobre los no saturados >= +3 %, sin redondear
C_EMPEORA_MAS_DE = 0.03              # (c) no saturado: R < -3 % y IC95 de R < 0
C_SATURADO_EMPEORA_D = -0.01         # (c) saturado: D < -0,01 y IC95 de D < 0 (fijado DESPUÉS de ver `sick`)
D_COSTE_MAXIMO = 8.0                 # (d) Σ tiempo_de_ajuste rejilla <= 8 x Σ hgb_hoy, razón de sumas sobre los 6
PLIEGUES = [0, 1, 2, 3, 4]
REPETICION = 0
N_CONJUNTOS = 6

# LO QUE REGLA.md §1 FIJA DEL PROTOCOLO, para comprobar lo que cada registro DECLARA (no creérselo). `tests/test_c169_regla.py` los fija contra el
# texto de REGLA.md y contra el protocolo v2 (de donde los saca el medidor): si divergen, una prueba se pone roja.
METRICA_POR_TAREA = {"binary_classification": "auroc", "multiclass_classification": "accuracy", "regression": "r2"}
PRESUPUESTO_POR_CUBO_S = {"pequeno": 120.0, "mediano": 300.0}
HILOS_POR_INTENTO = 4
SEMILLA = 0                          # la semilla de la repetición 0 (`particion.semillas[0]` del v2)
FRACCION_DE_PRUEBA = 0.2             # la prueba fija de la Fase 0 (`particiones_base`: test_fraction=0.2); n_test = 0,2 x n_filas, ±1 por redondeo
TOLERANCIA_N_TEST = 1

# LO QUE SE DICE DE LOS DATOS CON EL VEREDICTO (aud1 M-4 y la observación de BookID). No cambia la regla: la hace legible.
PREDICTORES_A_DECIR = {
    "analcatdata_authorship": ("BookID", "lleva `BookID` como predictor: OpenML NO lo marca como `ignore_attribute`, así que la regla (§1) "
                                         "manda incluirlo; es un identificador del libro, y el conjunto está cerca del umbral de saturación (0,98)"),
}
#: Los avisos de lectura que ya se sabe por qué salen (`data_id -> trozos del texto`). Cualquier OTRO aviso se marca NUEVO en el informe.
AVISOS_DE_LECTURA_ESPERADOS = {
    44960: ("sobran 1",),            # el fichero trae `cooling_load` (ignore_attribute); se quita DESPUÉS de leer, al aplicar REGLA.md §1
    44962: ("FALTAN 2",),            # `month` y `day` son atributos STRING: el lector único los excluye (y lo declara) antes de contar
}

REFERENCIA, REJILLA = "hgb_hoy", "hgb_rejilla"
BRAZOS = (REFERENCIA, REJILLA)
ESTADOS_DE_MEDIDA = protocolo_mod.ESTADOS_QUE_CUENTAN_COMO_MEDIDA

RUTA_DE_LA_REGLA = _AQUI / "REGLA.md"
RUTA_DEL_ANALIZADOR = Path(__file__).resolve()
RUTA_DEL_MEDIDOR = _AQUI / "medir_c169.py"
#: El sello del CÓDIGO (rep1, I1). No está en `codigo_de_ahora()`: un sello no puede contener su propio sha256.
RUTA_DEL_SELLO_DEL_CODIGO = _AQUI / "SELLO_DEL_CODIGO.json"

#: sha256 de `REGLA.md` SELLADA. La pone el supervisor al sellar. `None` = sin sellar: ninguna medida real vale.
REGLA_SHA256_SELLADA: str | None = "eb65a7e92013fc3992072fdd66cd908f0ccf8844c26f12c6828229e964e3390e"  # sellada por el supervisor el 15-11


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def dependencias_de_ahora() -> dict[str, str]:
    """sha256 de lo que el veredicto IMPORTA y no copia (las del 163-C2, que ya cubren el 160, la Fase 0 y el núcleo) MÁS el 163-C2 y el catálogo."""
    d = dict(an163.dependencias_de_ahora())
    d["benchmarks/c163/analiza_c163_c2.py"] = _sha256(Path(an163.__file__).resolve())
    d["benchmarks/c169/conjuntos_c169.py"] = _sha256(Path(cj.__file__).resolve())
    d["benchmarks/c169/conjuntos_c169.json"] = _sha256(cj.RUTA_DEL_CATALOGO)
    return d


def codigo_de_ahora() -> dict[str, str]:
    """Lo que cubre el SELLO DEL CÓDIGO: este analizador (que lleva dentro el sha de la regla), el medidor y las dependencias del veredicto.
    NO el propio `SELLO_DEL_CODIGO.json` (circular) ni las pruebas."""
    d = {"benchmarks/c169/analiza_c169.py": _sha256(RUTA_DEL_ANALIZADOR), "benchmarks/c169/medir_c169.py": _sha256(RUTA_DEL_MEDIDOR)}
    d.update(dependencias_de_ahora())
    return d


def leer_sello_del_codigo(ruta: Path | None = None) -> dict | None:
    """El contenido de `SELLO_DEL_CODIGO.json`, o None si no existe o no se puede leer."""
    ruta = Path(ruta) if ruta is not None else RUTA_DEL_SELLO_DEL_CODIGO
    try:
        sello = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return sello if isinstance(sello, dict) else None


def problemas_del_sello_del_codigo(ruta: Path | None = None) -> list[str]:
    """Vacía SOLO si el sello existe, no tiene nulos, cubre EXACTAMENTE `codigo_de_ahora()`, cada sha cuadra con el fichero de ahora y se selló
    sobre la regla sellada de este analizador."""
    ruta = Path(ruta) if ruta is not None else RUTA_DEL_SELLO_DEL_CODIGO
    sello = leer_sello_del_codigo(ruta)
    if sello is None:
        return [f"falta el sello del código o no se puede leer ({ruta.name}): el supervisor lo pone con `sellar_codigo_c169.py`"]
    p: list[str] = []
    sha = sello.get("sha256")
    if not isinstance(sha, dict) or not sha or any(v is None for v in sha.values()):
        return ["el sello del código está SIN SELLAR (valores nulos): ninguna medida vale todavía"]
    ahora = codigo_de_ahora()
    if set(sha) != set(ahora):
        p.append(f"el sello del código no cubre exactamente lo que importa el veredicto: faltan {sorted(set(ahora) - set(sha))}, "
                 f"sobran {sorted(set(sha) - set(ahora))}")
    for etiqueta in sorted(set(sha) & set(ahora)):
        if sha[etiqueta] != ahora[etiqueta]:
            p.append(f"`{etiqueta}` ({ahora[etiqueta][:12]}) no es el sellado ({str(sha[etiqueta])[:12]}): cambió DESPUÉS de sellar el código")
    if sello.get("regla_sha256_sellada") != REGLA_SHA256_SELLADA:
        p.append(f"el sello del código se hizo sobre otra regla ({str(sello.get('regla_sha256_sellada'))[:12]}) que la sellada en este "
                 f"analizador ({str(REGLA_SHA256_SELLADA)[:12]})")
    return p


def sellos_de_ahora() -> dict:
    sello = leer_sello_del_codigo()
    return {"regla_sha256": _sha256(RUTA_DE_LA_REGLA) if RUTA_DE_LA_REGLA.exists() else None,
            "analizador_sha256": _sha256(RUTA_DEL_ANALIZADOR), "dependencias_sha256": dependencias_de_ahora(),
            "codigo_sha256": codigo_de_ahora(), "sello_del_codigo": (sello or {}).get("sha256")}


# ----------------------------------------------------------------------------------------------------------------------
# 1. UN CONJUNTO
# ----------------------------------------------------------------------------------------------------------------------
def evaluar_conjunto(nombre: str, tarea: str, metric_id: str, registros: dict[int, dict[str, dict]], esperados: list[int]) -> dict:
    """UN conjunto: por pliegue la métrica de los dos brazos, R y su IC, D y su IC, saturado (por la REFERENCIA sola), coste.
    `registros[pliegue][brazo]`."""
    ev = an163.evaluar_conjunto(nombre, tarea, metric_id, "hgb", registros, esperados, saturados=())
    ev["saturado"] = False
    if not ev["completo"]:
        return ev
    pl = ev["pliegues"]
    hoy = [ev["por_pliegue"][f][REFERENCIA] for f in pl]
    rej = [ev["por_pliegue"][f][REJILLA] for f in pl]
    media_hoy = statistics.fmean(hoy)
    ev["saturado"] = bool(media_hoy >= UMBRAL_DE_SATURACION)       # SOLO la referencia: la rejilla no decide quién es saturado
    diferencias = [b - a for a, b in zip(hoy, rej)]
    ev["d"] = statistics.fmean(diferencias)
    ic_d = protocolo_mod._intervalo_pareado(diferencias)
    ev["ic_d"] = list(ic_d) if ic_d else None
    if ev["saturado"]:
        # Para un saturado la medida es D, no R: R/«mejora»/«empeora» de R no cuentan.
        ev["mejora_clara"] = False
        ev["empeora_de_verdad"] = bool(ic_d is not None and ev["d"] < C_SATURADO_EMPEORA_D and ic_d[1] < 0.0)
    return ev


# ----------------------------------------------------------------------------------------------------------------------
# 2. LA REGLA
# ----------------------------------------------------------------------------------------------------------------------
def aplicar_la_regla(evaluados: list[dict]) -> dict:
    """(0) y (a)–(d) sobre conjuntos YA evaluados y completos. `veredicto`: CONFIRMADA / NO CUMPLE / NO CONCLUYENTE."""
    no_sat = [e for e in evaluados if not e["saturado"]]
    n = len(no_sat)
    cero = n >= N_MINIMO_NO_SATURADOS
    r_ns = [e["r"] for e in no_sat]
    a_media = statistics.fmean(r_ns) if r_ns else None
    a_ok = a_media is not None and a_media >= A_REDUCCION_MEDIA_MIN
    b_min = math.ceil(n / 2) if n else 0
    mejoras = [e["nombre"] for e in no_sat if e["mejora_clara"]]
    b_ok = n > 0 and len(mejoras) >= b_min
    empeoran = [e["nombre"] for e in evaluados if e["empeora_de_verdad"]]
    c_ok = not empeoran
    suma = {b: sum(e["coste_s"][b] for e in evaluados) for b in BRAZOS}
    ratios = [e["coste_ratio"] for e in evaluados if e["coste_ratio"] is not None]
    d_ratio = suma[REJILLA] / suma[REFERENCIA] if suma[REFERENCIA] > 0 else None
    d_ok = d_ratio is not None and d_ratio <= D_COSTE_MAXIMO
    if not cero:
        veredicto = "NO CONCLUYENTE"
    elif a_ok and b_ok and c_ok and d_ok:
        veredicto = "CONFIRMADA"
    else:
        veredicto = "NO CUMPLE"
    return {"veredicto": veredicto, "n_no_saturados": n, "saturados": [e["nombre"] for e in evaluados if e["saturado"]],
            "no_saturados": [e["nombre"] for e in no_sat],
            "cero": {"n": n, "umbral": N_MINIMO_NO_SATURADOS, "cumple": cero},
            "a": {"media_r_no_saturados": a_media, "umbral": A_REDUCCION_MEDIA_MIN, "cumple": a_ok},
            "b": {"mejoras_claras": mejoras, "n": len(mejoras), "umbral": b_min, "cumple": b_ok},
            "c": {"empeoran": empeoran, "cumple": c_ok,
                  "d_de_los_saturados": {e["nombre"]: e["d"] for e in evaluados if e["saturado"]}},
            "d": {"razon_de_sumas": d_ratio, "mediana_por_conjunto": statistics.median(ratios) if ratios else None,
                  "maxima_por_conjunto": max(ratios) if ratios else None, "umbral": D_COSTE_MAXIMO, "cumple": d_ok}}


# ----------------------------------------------------------------------------------------------------------------------
# 3. COMPLETITUD, SELLOS Y LA MEDIDA ENTERA
# ----------------------------------------------------------------------------------------------------------------------
def problemas_de_completitud(payload: dict, es_sonda: bool) -> list[str]:
    p: list[str] = []
    conjuntos = payload.get("conjuntos", [])
    nombres = [c.get("nombre") for c in conjuntos]
    for n in sorted({n for n in nombres if nombres.count(n) > 1}):
        p.append(f"conjunto DUPLICADO en la salida: {n}")
    if not es_sonda:
        esperados = {d.nombre for d in cj.conjuntos()}
        if set(nombres) != esperados:
            p.append(f"los conjuntos no son los {N_CONJUNTOS} de la regla: faltan {sorted(esperados - set(nombres))}, "
                     f"sobran {sorted(set(nombres) - esperados)}")
        for c in conjuntos:
            if list(c.get("pliegues_esperados") or []) != PLIEGUES:
                p.append(f"{c.get('nombre')}: pliegues declarados {c.get('pliegues_esperados')} != {PLIEGUES}")
    ids_registrados = {d.nombre: d.data_id for d in cj.conjuntos()}
    for c in conjuntos:
        if c.get("nombre") in ids_registrados and c.get("data_id") != ids_registrados[c["nombre"]]:
            p.append(f"{c.get('nombre')}: data_id {c.get('data_id')} != el registrado {ids_registrados[c['nombre']]}")
        if c.get("nombre") in ids_registrados:
            esperado_ign = cj.ignorados_registrados()[ids_registrados[c["nombre"]]]
            if c.get("columnas_ignoradas_por_openml") != esperado_ign:
                p.append(f"{c.get('nombre')}: declara haber excluido {c.get('columnas_ignoradas_por_openml')} y los `ignore_attribute` de "
                         f"OpenML son {esperado_ign}")
        if c.get("sellado"):
            p.append(f"{c.get('nombre')} está marcado como SELLADO")
    esperado = {(c.get("nombre"), b, int(f)) for c in conjuntos for b in BRAZOS for f in (c.get("pliegues_esperados") or [])}
    sha_ok = {d.nombre: d.sha256_arff for d in cj.conjuntos()}
    vistos: dict[tuple, int] = {}
    for r in payload.get("resultados", []):
        k = (r.get("dataset"), r.get("motor"), int(r.get("pliegue", -1)))
        vistos[k] = vistos.get(k, 0) + 1
        if r.get("motor") not in BRAZOS:
            p.append(f"el intento {k} es de un brazo que no es de esta medida ({r.get('motor')!r})")
            continue
        if r.get("repeticion") != REPETICION:
            p.append(f"un registro de la repetición {r.get('repeticion')} ({k}): la regla es sobre la 0")
        if r.get("estado") not in ESTADOS_DE_MEDIDA:
            p.append(f"el intento {k} no cuenta como medida (estado {r.get('estado')!r}): falta ese pliegue")
            continue
        metrica, coste = _numero(r, an163.metrica_del_registro), _numero(r, an163.coste_del_registro)
        if metrica is None:
            p.append(f"el intento {k} no trae su métrica de cierre ({r.get('metrica_de_cierre')!r})")
        elif not math.isfinite(metrica):
            p.append(f"el intento {k}: su métrica de cierre no es un número finito ({metrica!r})")
        if coste is None:
            p.append(f"el intento {k} no trae `tiempo_de_ajuste` (el coste de (d))")
        elif not math.isfinite(coste) or coste < 0:
            p.append(f"el intento {k}: `tiempo_de_ajuste` no es un número finito y >= 0 ({coste!r})")
        if r.get("dataset") in sha_ok and r.get("datos_sha256") != sha_ok[r["dataset"]]:
            p.append(f"el intento {k}: sha256 de los datos {str(r.get('datos_sha256'))[:12]} != el registrado "
                     f"{sha_ok[r['dataset']][:12]}")
        p += an163.problemas_de_la_declaracion(r)
    faltan_i, sobran_i = sorted(esperado - set(vistos)), sorted(set(vistos) - esperado)
    dobles = sorted(k for k, n in vistos.items() if n > 1)
    if faltan_i:
        p.append(f"faltan {len(faltan_i)} intentos (p. ej. {faltan_i[:3]})")
    if sobran_i:
        p.append(f"{len(sobran_i)} registros fuera de lo declarado (p. ej. {sobran_i[:3]})")
    if dobles:
        p.append(f"{len(dobles)} intentos registrados más de una vez (p. ej. {dobles[:3]})")
    return p


def _numero(r: dict, lector) -> float | None:
    try:
        return lector(r)
    except (TypeError, ValueError):
        return float("nan")                  # un valor que no es un número cuenta como no finito, no como ausente


def problemas_de_lo_declarado(payload: dict, es_sonda: bool) -> list[str]:
    """I2 de la aud1: lo que DECLARAN cada conjunto y cada registro se compara con lo que fija REGLA.md §1 y el catálogo; no se cree.
    Por registro que cuenta como medida: métrica de cierre de SU tarea, tarea y cubo del catálogo, presupuesto de SU cubo (salvo en una sonda,
    que mide con el suyo), semilla 0, 4 hilos GRABADOS, `entorno_digest` y `motor_digest` = los de `digest_de_la_cache` de la salida y `n_test`
    = el de su conjunto (que a su vez es 0,2 x n_filas)."""
    p: list[str] = []
    catalogo = {d.nombre: d for d in cj.conjuntos()}
    cache = payload.get("digest_de_la_cache") or {}
    entorno, motores_d = cache.get("entorno"), cache.get("motores") or {}
    if not entorno:
        p.append("la salida no declara `digest_de_la_cache.entorno`: no hay con qué comparar el código de cada intento")
    for b in BRAZOS:
        if not motores_d.get(b):
            p.append(f"la salida no declara `digest_de_la_cache.motores.{b}`")
    n_test_de: dict[str, object] = {}
    for c in payload.get("conjuntos", []):
        d = catalogo.get(c.get("nombre"))
        if d is None:
            continue                                         # un conjunto que no es de los 6 ya lo dice la completitud
        esperado = {"tarea": d.tarea, "cubo": d.cubo_de_tamano, "metrica_de_cierre": METRICA_POR_TAREA[d.tarea], "n_filas": d.n_filas}
        for campo, valor in esperado.items():
            if c.get(campo) != valor:
                p.append(f"{d.nombre}: declara {campo}={c.get(campo)!r} y la regla/el catálogo dicen {valor!r}")
        nt = c.get("n_test")
        if isinstance(nt, bool) or not isinstance(nt, int) or abs(nt - FRACCION_DE_PRUEBA * d.n_filas) > TOLERANCIA_N_TEST:
            p.append(f"{d.nombre}: n_test={nt!r} no es la prueba fija de la Fase 0 ({FRACCION_DE_PRUEBA:g} x {d.n_filas} filas)")
        n_test_de[d.nombre] = nt
    agrupados: dict[str, list] = {}

    def mal(motivo: str, k) -> None:
        agrupados.setdefault(motivo, []).append(k)

    for r in payload.get("resultados", []):
        if r.get("motor") not in BRAZOS or r.get("estado") not in ESTADOS_DE_MEDIDA:
            continue                                         # ya lo dice la completitud
        d = catalogo.get(r.get("dataset"))
        if d is None:
            continue
        k = (r.get("dataset"), r.get("motor"), r.get("pliegue"))
        if r.get("metrica_de_cierre") != METRICA_POR_TAREA[d.tarea]:
            mal(f"métrica de cierre distinta de la de su tarea ({d.tarea} -> {METRICA_POR_TAREA[d.tarea]})", k)
        if r.get("tarea") != d.tarea or r.get("cubo") != d.cubo_de_tamano or r.get("data_id") != d.data_id:
            mal("tarea, cubo o data_id distintos de los del catálogo", k)
        if not es_sonda and r.get("presupuesto_wall_s") != PRESUPUESTO_POR_CUBO_S.get(d.cubo_de_tamano):
            mal(f"presupuesto de pared distinto del de su cubo ({PRESUPUESTO_POR_CUBO_S})", k)
        if isinstance(r.get("semilla"), bool) or r.get("semilla") != SEMILLA:
            mal(f"semilla distinta de {SEMILLA}", k)
        if isinstance(r.get("hilos"), bool) or r.get("hilos") != HILOS_POR_INTENTO:
            mal(f"no graba {HILOS_POR_INTENTO} hilos (`hilos`)", k)
        if r.get("entorno_digest") != entorno:
            mal("`entorno_digest` distinto del de la salida (otro código: ¿caché de otra medida?)", k)
        if r.get("motor_digest") != motores_d.get(r.get("motor")):
            mal("`motor_digest` distinto del de su brazo en la salida", k)
        if r.get("n_test") != n_test_de.get(d.nombre):
            mal("`n_test` distinto del de su conjunto", k)
    for motivo, ks in agrupados.items():
        p.append(f"{len(ks)} intento(s) con {motivo} (p. ej. {ks[:3]})")
    return p


def problemas_de_sellos(payload: dict) -> list[str]:
    p: list[str] = []
    sellos = sellos_de_ahora()
    if REGLA_SHA256_SELLADA is None:
        p.append("la regla NO está sellada (`REGLA_SHA256_SELLADA` es None): ninguna medida vale todavía")
    elif sellos["regla_sha256"] != REGLA_SHA256_SELLADA:
        p.append(f"REGLA.md ({str(sellos['regla_sha256'])[:12]}) no es la sellada ({REGLA_SHA256_SELLADA[:12]})")
    al_medir = payload.get("dependencias_sha256_al_medir")
    if not isinstance(al_medir, dict):
        p.append("la salida no declara `dependencias_sha256_al_medir`")
    else:
        for etiqueta, propio in sellos["dependencias_sha256"].items():
            if al_medir.get(etiqueta) != propio:
                p.append(f"`{etiqueta}` al medir ({str(al_medir.get(etiqueta))[:12]}) != el que importa este analizador ({propio[:12]})")
    for clave_, propio in (("regla_sha256_al_medir", sellos["regla_sha256"]), ("analizador_sha256_al_medir", sellos["analizador_sha256"])):
        valor = payload.get(clave_)
        if valor is None:
            p.append(f"la salida no declara `{clave_}`")
        elif valor != propio:
            p.append(f"`{clave_}` de la salida ({valor[:12]}) != el fichero de ahora ({str(propio)[:12]}): cambió DESPUÉS de medir")
    if payload.get("regla_sha256_sellada") != REGLA_SHA256_SELLADA:
        p.append("la salida declara otro sello de regla que el de este analizador")
    p += problemas_del_sello_del_codigo()
    if payload.get("sello_del_codigo_al_medir") != sellos["sello_del_codigo"]:
        p.append("la salida no declara el sello del código con que se midió, o se midió con otro sello del código "
                 "(`sello_del_codigo_al_medir`) que el de ahora")
    return p


def analizar(payload: dict, *, comprobar_sellos: bool = True) -> dict:
    """El veredicto de una salida de `medir_c169.py`."""
    problemas: list[str] = []
    es_sonda = bool(payload.get("es_sonda"))
    if payload.get("parcial"):
        problemas.append("la salida es PARCIAL (la medida no terminó)")
    problemas += problemas_de_completitud(payload, es_sonda)
    problemas += problemas_de_lo_declarado(payload, es_sonda)
    rs = payload.get("resultados", [])
    for brazo in BRAZOS:
        for campo in ("entorno_digest", "motor_digest"):
            if len({r.get(campo) for r in rs if r.get("motor") == brazo}) > 1:
                problemas.append(f"los intentos del brazo {brazo} se midieron con CÓDIGO DISTINTO ({campo})")
    if len({r.get("entorno_digest") for r in rs}) > 1:
        problemas.append("los brazos se midieron con CÓDIGO DISTINTO (entorno_digest)")
    if comprobar_sellos:
        problemas += problemas_de_sellos(payload)
    por_clave = {(r["dataset"], r["motor"], int(r["pliegue"])): r for r in rs
                 if r.get("estado") in ESTADOS_DE_MEDIDA and r.get("motor") in BRAZOS}
    evaluados = []
    for c in payload.get("conjuntos", []):
        esperados = [int(f) for f in c.get("pliegues_esperados") or []]
        registros: dict[int, dict[str, dict]] = {}
        for f in esperados:
            for b in BRAZOS:
                r = por_clave.get((c["nombre"], b, f))
                if r is not None:
                    registros.setdefault(f, {})[b] = r
        e = evaluar_conjunto(c["nombre"], c["tarea"], c["metrica_de_cierre"], registros, esperados)
        if not e["completo"]:
            problemas.append(f"{c['nombre']}: faltan pliegues {e['pliegues_que_faltan']}")
        evaluados.append(e)
    regla = aplicar_la_regla(evaluados) if evaluados and all(e["completo"] for e in evaluados) else None
    if es_sonda:
        veredicto = "SONDA"
    elif problemas or regla is None:
        veredicto = "SIN VEREDICTO"
    else:
        veredicto = regla["veredicto"]
    return {"veredicto": veredicto, "problemas": problemas, "es_sonda": es_sonda, "sellos_de_ahora": sellos_de_ahora(),
            "regla": regla, "conjuntos": evaluados, "lo_que_se_dice_de_los_datos": lo_que_se_dice_de_los_datos(payload),
            "nota_del_umbral": "El umbral de -0,01 de los saturados se fijó DESPUÉS de ver `sick` (163-C2), donde habría quedado dentro "
                               "(-0,42 puntos). Con 6 conjuntos pequeños la potencia es baja: un NO CUMPLE no demuestra que la rejilla sea "
                               "mala y una CONFIRMADA no la integra (decisión de Roberto)."}


def lo_que_se_dice_de_los_datos(payload: dict) -> list[str]:
    """Lo que el veredicto DICE de los datos medidos, sacado de lo que la salida declara (aud1 M-4): columnas que el lector único excluyó,
    `ignore_attribute` quitados, predictores que conviene nombrar (BookID) y los avisos de lectura, separando los ya explicados de los NUEVOS."""
    L: list[str] = []
    for c in payload.get("conjuntos", []):
        nombre = c.get("nombre")
        lectura = c.get("lectura_declarada") or {}
        excluidas = lectura.get("columnas_excluidas") or []
        if excluidas:
            motivos = sorted({str(m) for m in (lectura.get("motivos_de_exclusion") or {}).values()})
            L.append(f"{nombre} se midió SIN {excluidas}: las excluyó el lector único de la Fase 0, igual en los dos brazos"
                     + (f" ({'; '.join(motivos)})" if motivos else ""))
        ignoradas = c.get("columnas_ignoradas_por_openml") or []
        if ignoradas:
            L.append(f"{nombre} se midió SIN {ignoradas}: OpenML las marca `ignore_attribute` (REGLA.md §1)")
        if nombre in PREDICTORES_A_DECIR:
            columna, frase = PREDICTORES_A_DECIR[nombre]
            predictores = c.get("predictores")
            if predictores is None:
                L.append(f"{nombre}: la salida no declara sus predictores; según el catálogo {frase}")
            elif columna in predictores:
                L.append(f"{nombre} {frase}")
            else:
                L.append(f"{nombre}: `{columna}` NO está entre sus predictores declarados (se esperaba: {frase})")
        esperados = AVISOS_DE_LECTURA_ESPERADOS.get(c.get("data_id"), ())
        for aviso in lectura.get("avisos") or []:
            ya = any(t in str(aviso) for t in esperados)
            L.append(f"{nombre}: aviso de lectura {'ya explicado' if ya else 'NUEVO'}: {aviso}")
    return L


def _f(x, nd: int = 3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def informe_en_texto(r: dict) -> str:
    L = [f"=== 169: rejilla de HGB en 6 conjuntos nuevos — {r['veredicto']} ==="]
    for p in r["problemas"]:
        L.append(f"  PROBLEMA: {p}")
    g = r.get("regla")
    if g:
        L.append(f"  (0) no saturados: {g['cero']['n']} (>= {g['cero']['umbral']}) -> {'ok' if g['cero']['cumple'] else 'NO'}  saturados: {g['saturados']}")
        L.append(f"  (a) media de R sobre los no saturados: {_f(g['a']['media_r_no_saturados'], 4)} (>= {g['a']['umbral']}) -> {'ok' if g['a']['cumple'] else 'NO'}")
        L.append(f"  (b) mejoras claras: {g['b']['n']} (>= {g['b']['umbral']}) {g['b']['mejoras_claras']} -> {'ok' if g['b']['cumple'] else 'NO'}")
        L.append(f"  (c) empeoran de verdad: {g['c']['empeoran']} -> {'ok' if g['c']['cumple'] else 'NO'}")
        L.append(f"  (d) razón de sumas de ajuste: {_f(g['d']['razon_de_sumas'], 2)} (<= {g['d']['umbral']}); mediana {_f(g['d']['mediana_por_conjunto'], 2)}, "
                 f"máxima {_f(g['d']['maxima_por_conjunto'], 2)} -> {'ok' if g['d']['cumple'] else 'NO'}")
    L.append(f"  {'conjunto':34s} sat      R        IC95 de R            D        IC95 de D           mejora empeora")
    for e in r["conjuntos"]:
        if "d" not in e:
            L.append(f"  {e['nombre']:34s} (sin medida completa)")
            continue
        ic, icd = e["ic"], e.get("ic_d")
        L.append(f"  {e['nombre']:34s} {'S' if e['saturado'] else ' '}   {e['r']:+8.4f}  [{_f(ic[0]) if ic else '—'}, {_f(ic[1]) if ic else '—'}]  "
                 f"{e['d']:+8.4f}  [{_f(icd[0], 4) if icd else '—'}, {_f(icd[1], 4) if icd else '—'}]  "
                 f"{'sí' if e['mejora_clara'] else '  '}    {'SÍ' if e['empeora_de_verdad'] else '  '}")
    for frase in r.get("lo_que_se_dice_de_los_datos") or []:
        L.append(f"  DATOS: {frase}")
    L.append(f"  NOTA: {r['nota_del_umbral']}")
    return "\n".join(L)


CODIGOS_DE_SALIDA = {"CONFIRMADA": 0, "NO CUMPLE": 1, "SIN VEREDICTO": 2, "NO CONCLUYENTE": 3, "SONDA": 4}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("resultado")
    p.add_argument("--json", default=None, help="escribe el veredicto completo en este JSON")
    a = p.parse_args(argv)
    payload = json.loads(Path(a.resultado).resolve().read_text(encoding="utf-8"))
    r = analizar(payload)
    print(informe_en_texto(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return CODIGOS_DE_SALIDA.get(r["veredicto"], CODIGOS_DE_SALIDA["SIN VEREDICTO"])


if __name__ == "__main__":
    sys.exit(main())
