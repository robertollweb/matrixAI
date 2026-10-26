#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""139b · ¿ENTRA EL MOTOR LINEAL (`sklearn.lineal`) COMO COMPETIDOR DEL ESTUDIO? La medición del protocolo
`documentacion/139b_MEDICION_DEL_LINEAL_PROTOCOLO.md` (escrito y fijado ANTES de medir; léelo antes de tocar nada).

El INSTRUMENTO es el propio estudio del producto, en el mismo proceso (como `medir_114c4_cierre.py`): `iniciar_estudio` /
`estudio_status`. Por cada conjunto de la rejilla corre DOS estudios con la misma semilla, los mismos pliegues y el
mismo test:

    A · «hoy»          los motores del estudio de esta instalación (los árboles, la referencia, la densa y TabM si hay torch)
    B · «hoy + lineal» lo mismo y `sklearn.lineal` como un competidor MÁS (peso de búsqueda 1,0, como lightgbm y hgb)

y compara la cifra del campeón en el TEST (`seleccion.evaluacion_final`), con la regla de la enmienda 7 del 120 (la que
decidió si TabM entraba): `veredicto_120_c3p.comparar` / `corta`, sin tocarlas. El lineal se añade ENVOLVIENDO
`estudio_job._motores_del_estudio` en este proceso; no se cambia ni una línea del Studio ni de los motores.

NO reutiliza `referencia_120_r0.py` entero (ese arranca contenedores Docker con una imagen del Studio y el Studio publicado no
trae el lineal): importa de él lo que no depende de Docker (el CSV de la Fase 0, el cuerpo del estudio, el registro de un
estudio, la lectura de los intentos), por RUTA y sin tocar `sys.path`.

    python3 benchmarks/fase0/medicion_139b_lineal.py --estimar [--condicion con_torch]
    python3 benchmarks/fase0/medicion_139b_lineal.py --humo --salida <ruta fuera del repo>   # 1 conjunto, sin redes, ~1 min
    python3 benchmarks/fase0/medicion_139b_lineal.py --condicion sin_torch --salida benchmarks/fase0/resultado_139b_lineal_sin_torch.json   # cola
    python3 benchmarks/fase0/medicion_139b_lineal.py --condicion con_torch --salida benchmarks/fase0/resultado_139b_lineal_con_torch.json   # cola

Punto de control: la salida (os.replace tras cada estudio). Se reutilizan SOLO registros `completed` medidos con el mismo
guion, la misma semilla y el mismo conjunto de motores (CLAUDE.md, «ANTES DE RELANZAR UNA PASADA»: un registro fallido NO
se reutiliza jamás; y si se toca el digest de algo que este guion no cubre, `--forzar`).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
NUCLEO = AQUI.parents[1]

#: El motor que se mide y el peso con que reparte el presupuesto (el de los ligeros: medido, ~5 s por intento en frío igual que
#: lightgbm y hgb en la pasada v2 del 113; `estudio_job._PESO_DE_BUSQUEDA`).
LINEAL = "sklearn.lineal"
PESO_DEL_LINEAL = 1.0
OWNER = "medicion-139b"
TOPE_ESTUDIO_S = 3600          # por estudio; pasado esto se registra «plazo»
#: LAS DOS CONDICIONES (protocolo 139b §2): el Studio SIN torch (los árboles y la referencia: el paquete CPU descargable, o
#: `MATRIXAI_ESTUDIO_SIN_TABM=1` + `MATRIXAI_ESTUDIO_SIN_DENSA=1`, que es la política de producción 120-C6) y el Studio CON torch
#: (además la densa y TabM: el paquete con GPU). SIN torch va primero: es la más favorable al lineal y la más barata; si el lineal
#: no entra ahí, no hay nada que confirmar con torch.
CONDICIONES = ("sin_torch", "con_torch")
#: Los conjuntos de cada condición, EN ESTE ORDEN (baratos y favorables primero; el más pesado, el último: si el tope de la cola
#: lo corta se pierde solo él). Fijadas en el protocolo §3 ANTES de medir. NINGÚN sellado. «favorable» = en la pasada v2 del 113
#: el lineal quedó por delante del mejor árbol o a menos de 3 puntos; «control» = a más de 3 puntos (el lineal debe PERDER: si
#: en uno de ellos el estudio lo elige, es la maldición del ganador —un candidato más con ruido— y no mérito del lineal).
GRUPO = {
    "diabetes": "favorable", "balance-scale": "favorable", "climate-model-simulation-crashes": "favorable",
    "dresses-sales": "favorable", "wilt": "favorable", "yeast": "favorable", "house_prices_nominal": "favorable",
    "ozone-level-8hr": "favorable", "PhishingWebsites": "favorable", "us_crime": "favorable",
    "mfeat-factors": "favorable", "optdigits": "favorable", "Internet-Advertisements": "favorable",
    "kc2": "control", "Moneyball": "control", "pc1": "control", "jm1": "control", "wine_quality": "control",
    "pc4": "control", "sick": "control",
}
REJILLA_DE = {
    "sin_torch": ("diabetes", "balance-scale", "climate-model-simulation-crashes", "dresses-sales", "kc2", "wilt",
                  "Moneyball", "yeast", "pc1", "jm1", "wine_quality", "pc4", "sick", "house_prices_nominal",
                  "ozone-level-8hr", "PhishingWebsites", "us_crime", "mfeat-factors", "optdigits",
                  "Internet-Advertisements"),
    "con_torch": ("diabetes", "balance-scale", "climate-model-simulation-crashes", "dresses-sales", "yeast", "wilt",
                  "pc1", "house_prices_nominal", "ozone-level-8hr", "us_crime"),
}
#: Los conjuntos en los que se repite A para fijar el SUELO DE RUIDO antes de medir (los más baratos).
REPETICION_DEL_SUELO = ("diabetes", "climate-model-simulation-crashes", "dresses-sales")
SELLADOS = {3, 6, 46, 201, 1050, 1462, 40900, 42225}      # data_id de los 8 sellados (CLAUDE.md / 101)

_modulos: dict = {}


def _cargar(nombre: str, ruta: Path):
    """Un módulo por RUTA, sin tocar `sys.path` (mismo criterio que `referencia_120_r0.py`)."""
    if nombre not in _modulos:
        spec = importlib.util.spec_from_file_location(nombre, ruta)
        m = importlib.util.module_from_spec(spec)
        sys.modules[nombre] = m          # lo necesitan los dataclass/pickle del módulo
        spec.loader.exec_module(m)
        _modulos[nombre] = m
    return _modulos[nombre]


def r0():
    return _cargar("referencia_120_r0", AQUI / "referencia_120_r0.py")


def v3():
    return _cargar("veredicto_120_c3p", AQUI / "veredicto_120_c3p.py")


def ahora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ el instrumento
class _Lineal:
    """El interruptor del lineal. `activo` lo decide el guion por estudio; el envoltorio se instala UNA vez."""
    activo = False
    instalado = False


def instalar_el_lineal() -> None:
    """Envuelve `estudio_job._motores_del_estudio` para que, con `_Lineal.activo`, el lineal compita como un candidato MÁS
    (al final de los permitidos: ante un empate exacto gana el de antes). Con `activo = False` no cambia nada: el peso se
    quita del diccionario y la función es la de siempre."""
    from matrixai_studio import estudio_job as ej               # noqa: PLC0415
    from matrixai_engines.registro_de_motores import motor_para  # noqa: PLC0415
    if _Lineal.instalado:
        return
    original = ej._motores_del_estudio

    def con_el_lineal(texto_declarado, *, red_de_matrixai=False):
        permitidos, motores = original(texto_declarado, red_de_matrixai=red_de_matrixai)
        if not _Lineal.activo or LINEAL in permitidos:
            return permitidos, motores
        return permitidos + (LINEAL,), {**motores, LINEAL: motor_para(LINEAL)}

    ej._motores_del_estudio = con_el_lineal
    _Lineal.instalado = True


#: LOS HIJOS. Cada intento y el reajuste final del campeón corren en un proceso `multiprocessing.spawn`, que RE-IMPORTA
#: `estudio_job` desde cero: sin el peso del lineal en SU `_PESO_DE_BUSQUEDA`, el reparto del reajuste final levanta
#: `KeyError: 'sklearn.lineal'` (medido en el humo del 26-10: B «completaba» sin campeón). Un hijo spawn ejecuta también el
#: nivel superior de este guion (como `__mp_main__`), así que la variable de entorno que fija el padre (heredada al lanzar el
#: hijo) basta para que cada hijo ponga el peso igual que el padre. Las funciones de abajo NO hacen nada en el padre salvo
#: cuando él mismo las llama.
_ENV_LINEAL = "MATRIXAI_139B_LINEAL"


def _peso_del_lineal_en_este_proceso(activo: bool) -> None:
    from matrixai_studio import estudio_job as ej               # noqa: PLC0415
    if activo:
        ej._PESO_DE_BUSQUEDA[LINEAL] = PESO_DEL_LINEAL
    else:
        ej._PESO_DE_BUSQUEDA.pop(LINEAL, None)


def poner_el_lineal(activo: bool) -> None:
    instalar_el_lineal()
    _Lineal.activo = activo
    _peso_del_lineal_en_este_proceso(activo)
    if activo:
        os.environ[_ENV_LINEAL] = "1"
    else:
        os.environ.pop(_ENV_LINEAL, None)


if os.environ.get(_ENV_LINEAL) == "1":          # un hijo spawn de un estudio B (ver arriba)
    _peso_del_lineal_en_este_proceso(True)


def estudiar(nombre: str, c: dict, gen, variante: str, semilla: int, estados: Path) -> dict:
    """Un estudio real en este proceso; el registro es el de `referencia_120_r0.registrar` más la variante."""
    from matrixai_studio.estudio_job import estudio_status, iniciar_estudio  # noqa: PLC0415
    R = r0()
    R.SEMILLA = semilla if semilla != 0 else None      # semilla 0 = el cuerpo de siempre, sin la clave `seed`
    csv_bytes, cols, excl, nfilas = R.csv_de(gen, c["data_id"])
    cuerpo = R.cuerpo_del_estudio(c, cols, csv_bytes.decode("utf-8"), R.SEMILLA)
    ini = ahora()
    la = open("/proc/loadavg").read().split()[:3]
    t0 = time.time()
    poner_el_lineal(variante == "B")
    r = iniciar_estudio(cuerpo, owner=OWNER)
    est_ini = (r.get("estudio") or {}) if isinstance(r, dict) else {}
    job = est_ini.get("job_id")
    if not r.get("ok") or not job:
        rec = R.registrar(nombre, c, cuerpo, {"estado": "rechazado"}, time.time() - t0, 200, r, la, ini)
        rec.update(variante=variante, csv={"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes)})
        return rec
    est: dict = {}
    while True:
        est = estudio_status(job, OWNER)
        if est.get("estado") in ("completed", "completed_budget_limited", "failed", "cancelled"):
            break
        if time.time() - t0 > TOPE_ESTUDIO_S:
            try:
                from matrixai_studio.estudio_job import cancelar_estudio  # noqa: PLC0415
                cancelar_estudio(job, OWNER)
            except Exception:                                              # noqa: BLE001
                pass
            est = dict(est, estado="plazo")
            break
        time.sleep(2.0)
    estados.mkdir(parents=True, exist_ok=True)
    (estados / f"{nombre}__{variante}.json").write_text(json.dumps(est, ensure_ascii=False, indent=1, default=str))
    rec = R.registrar(nombre, c, cuerpo, est, time.time() - t0, 200, None, la, ini)
    rec.update(job_id=job, variante=variante, csv={"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes),
                                                   "excluidas_por_string": excl})
    return rec


# ------------------------------------------------------------------------------------------------ el veredicto
def umbrales(suelo: float) -> tuple[float, float]:
    V = v3()
    return max(V.UMBRAL, suelo), max(V.BAJA_QUE_CORTA, suelo)


def comparar(a: dict | None, b: dict, suelo: float) -> dict:
    """Un conjunto: A (hoy) frente a B (hoy + lineal), con la regla de la enmienda 7. Añade la lectura del lineal: su
    posición en la media de selección y si es un EMPATE (a menos de un error típico corregido del campeón de A) y más barato."""
    V = v3()
    comp = V.comparar(a, b, suelo)
    comp["grupo"] = GRUPO.get(comp["nombre"])
    comp["lineal_campeon"] = b.get("campeon") == LINEAL
    comp["lectura_del_lineal"] = lectura_del_lineal(a, b)
    return comp


def _por_pliegue(rec: dict, motor: str) -> dict:
    return {(x["repeticion"], x["pliegue"]): float(x["valor"]) for x in rec.get("intentos") or []
            if x.get("motor") == motor and x.get("fase") == "seleccion" and x.get("estado") == "completed"
            and x.get("valor") is not None}


def _segundos(rec: dict, motor: str) -> float | None:
    s = [float(x["wall_seconds"]) for x in rec.get("intentos") or []
         if x.get("motor") == motor and x.get("wall_seconds") is not None]
    return round(sum(s), 2) if s else None


def error_tipico_corregido(diffs: list[float], k: int) -> float | None:
    """Error típico de la media de las diferencias PAREADAS por pliegue, con la corrección de Nadeau-Bengio (el factor del 131:
    sqrt(1/K + n_test/n_train), con n_test/n_train = 1/(K-1); 1,50 con K = 5). Informativo: NO decide nada del veredicto."""
    if len(diffs) < 2 or k < 2:
        return None
    return statistics.stdev(diffs) * math.sqrt(1.0 / len(diffs) + 1.0 / (k - 1))


def lectura_del_lineal(a: dict, b: dict) -> dict:
    """De B (que tiene a todos): dónde queda el lineal, y frente al campeón de A, pliegue a pliegue."""
    medias = {x["motor"]: x["valor"] for x in ((b.get("media_de_la_seleccion") or {}).get("motores") or [])
              if x.get("compite") and x.get("valor") is not None}
    metrica = (b.get("media_de_la_seleccion") or {}).get("metrica")
    out = {"metrica": metrica, "media_lineal": medias.get(LINEAL), "medias_de_b": medias,
           "segundos_lineal": _segundos(b, LINEAL), "campeon_a": a.get("campeon") if a else None}
    campeon = a.get("campeon") if a else None
    if LINEAL not in medias or not campeon or campeon not in medias:
        out["empate"] = None
        return out
    menor_es_mejor = metrica in {"rmse", "mae", "log_loss", "brier"}
    lin, cam = _por_pliegue(b, LINEAL), _por_pliegue(b, campeon)
    comunes = sorted(set(lin) & set(cam))
    signo = -1.0 if menor_es_mejor else 1.0
    diffs = [signo * (lin[p] - cam[p]) for p in comunes]          # > 0: el lineal va por delante
    k = len({p[1] for p in comunes}) or len(comunes)
    se = error_tipico_corregido(diffs, k)
    media = statistics.fmean(diffs) if diffs else None
    barato = (out["segundos_lineal"] is not None and _segundos(b, campeon) is not None
              and out["segundos_lineal"] < _segundos(b, campeon))
    out.update(n_pliegues=len(comunes), diferencia_media_con_el_campeon_de_a=media, error_tipico_corregido=se,
               segundos_campeon_de_a=_segundos(b, campeon), lineal_mas_barato=barato,
               empate=(media is not None and se is not None and abs(media) <= se),
               lineal_por_delante=(media is not None and media > 0),
               lineal_claramente_por_delante=(media is not None and se is not None and media > se))
    out["empate_y_mas_barato"] = bool(out["empate"] and barato)
    return out


def veredicto(comps: list[dict]) -> dict:
    """La regla 4 de la enmienda 7 (`veredicto_120_c3p.veredicto`) y, al lado, los números del lineal."""
    V = v3()
    base = V.veredicto(comps)
    base.pop("tabm_campeon_en", None)
    base["lineal_campeon_en"] = [c["nombre"] for c in comps if c.get("lineal_campeon")]
    base["lineal_empata_y_es_mas_barato_en"] = [c["nombre"] for c in comps
                                                if (c.get("lectura_del_lineal") or {}).get("empate_y_mas_barato")]
    base["lineal_campeon_en_control"] = [c["nombre"] for c in comps if c.get("lineal_campeon") and c.get("grupo") == "control"]
    base["lineal_por_delante_en_la_seleccion"] = [c["nombre"] for c in comps
                                                  if (c.get("lectura_del_lineal") or {}).get("lineal_por_delante")]
    base["lineal_claramente_por_delante_en_la_seleccion"] = [
        c["nombre"] for c in comps if (c.get("lectura_del_lineal") or {}).get("lineal_claramente_por_delante")]
    base["lineal_empata_en"] = [c["nombre"] for c in comps if (c.get("lectura_del_lineal") or {}).get("empate")]
    pa = [c["pared_c3p_s"] - c["pared_r1_s"] for c in comps if c.get("pared_c3p_s") is not None and c.get("pared_r1_s") is not None]
    base["segundos_de_pared_que_anade_por_estudio"] = {"media": round(statistics.fmean(pa), 1) if pa else None,
                                                       "maximo": round(max(pa), 1) if pa else None}
    # PROPUESTA (la decide Roberto, protocolo §4.2): la regla de TabM, más «al menos DOS suben».
    base["entra_por_la_regla_del_proyecto"] = base["mejora"]
    base["entra_con_la_propuesta_de_dos"] = bool(base["mejora"] and base["suben"] >= 2)
    return base


def sin_control(a: dict | None, b: dict) -> str | None:
    """El instrumento mide lo mismo en A y en B: la línea base (que no depende de ningún motor) tiene que dar LA MISMA media
    en los dos (mismos pliegues, misma semilla) y el lineal tiene que haber competido en B y NO en A. Si no, no es un resultado."""
    def medias(r):
        return {x["motor"]: x["valor"] for x in ((r.get("media_de_la_seleccion") or {}).get("motores") or [])
                if x.get("compite") and x.get("valor") is not None}
    V = v3()
    if not (V.completa(a) and V.completa(b)):
        return None
    ma, mb = medias(a), medias(b)
    n = b.get("nombre")
    if LINEAL in ma:
        return f"{n}: el lineal compite en A («hoy»): no es la condición registrada"
    if LINEAL not in mb:
        motivo = ((b.get("motores_que_no_puntuaron") or {}) or {}).get(LINEAL)
        return f"{n}: el lineal NO puntuó en B ({str(motivo)[:160]}): no se está midiendo B"
    if ma.get("baseline") is None or mb.get("baseline") is None or abs(ma["baseline"] - mb["baseline"]) > 1e-9:
        return f"{n}: la línea base difiere entre A ({ma.get('baseline')!r}) y B ({mb.get('baseline')!r}): otros pliegues"
    return None


# ------------------------------------------------------------------------------------------------ la pasada
def declarados() -> dict[str, dict]:
    """Los conjuntos de la Fase 0 con su tarea, objetivo y clase positiva (los mismos que usan R0 y todos los kits)."""
    out: dict[str, dict] = {}
    for f in r0().FASE0_DECLARADOS:
        for d in json.loads(Path(f).read_text())["datasets_declarados"]:
            out[d["nombre"]] = d
    return out


def comprobar_rejilla(rejilla, decl) -> None:
    for n in rejilla:
        d = decl.get(n)
        if d is None:
            raise SystemExit(f"PARO: {n} no es un conjunto de la Fase 0")
        if d.get("sellado") or d["data_id"] in SELLADOS:
            raise SystemExit(f"PARO: {n} está SELLADO: no entra en una medición de elección de motor")


def _commit(ruta: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ruta), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:                                                               # noqa: BLE001
        return None


def _sha_del_protocolo() -> str | None:
    """`documentacion/` está en el .gitignore del núcleo: en el worktree de la cola NO está. Se busca en el árbol principal
    (`~/matrixAI`), donde vive; sin él, `None` (no se inventa)."""
    for raiz in (NUCLEO, Path.home() / "matrixAI"):
        f = raiz / "documentacion/139b_MEDICION_DEL_LINEAL_PROTOCOLO.md"
        if f.exists():
            return sha(f)
    return None


def procedencia(args, rejilla) -> dict:
    import matrixai_studio, matrixai_engines, matrixai                             # noqa: PLC0415,E401
    raiz = lambda m: Path(m.__file__).resolve().parents[1]                         # noqa: E731
    return {"corte": "139b", "inicio": ahora(), "sha256_guion": sha(__file__), "semilla": args.semilla,
            "condicion": args.condicion, "rejilla": list(rejilla), "grupo": {n: GRUPO[n] for n in rejilla}, "peso_del_lineal": PESO_DEL_LINEAL,
            "protocolo": "documentacion/139b_MEDICION_DEL_LINEAL_PROTOCOLO.md",
            "protocolo_sha256": _sha_del_protocolo(),
            "modulos": {"matrixai": str(Path(matrixai.__file__).parent), "matrixai_engines": str(Path(matrixai_engines.__file__).parent),
                        "matrixai_studio": str(Path(matrixai_studio.__file__).parent)},
            "commits": {"nucleo": _commit(NUCLEO), "motores": _commit(raiz(matrixai_engines).parent),
                        "studio": _commit(raiz(matrixai_studio).parent)},
            "entorno": {k: v for k, v in os.environ.items() if k.startswith("MATRIXAI_")},
            "loadavg_al_empezar": open("/proc/loadavg").read().split()[:3], "humo": bool(args.humo)}


def guardar(salida: Path, datos: dict) -> None:
    tmp = salida.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, salida)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salida", type=Path, default=AQUI / "resultado_139b_lineal.json")
    ap.add_argument("--condicion", choices=CONDICIONES, default="sin_torch",
                    help="sin_torch: el Studio sin las redes (el paquete CPU); con_torch: con la densa y TabM")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--solo", help="NOMBRE[,NOMBRE]: (diagnóstico) solo esos de la rejilla")
    ap.add_argument("--estimar", action="store_true", help="imprime la rejilla y la estimación y sale; no mide")
    ap.add_argument("--humo", action="store_true", help="UN conjunto (diabetes), sin TabM ni la densa, ~1,5 min: solo comprueba que arranca")
    ap.add_argument("--sin-suelo", action="store_true", help="(diagnóstico) no repite A para el suelo de ruido")
    ap.add_argument("--cortar", action="store_true", help="aplica la regla 9 de C3′ (para en el primer conjunto que baja >= 2 o deja de completar); por omisión se mide todo")
    ap.add_argument("--forzar", action="store_true", help="ignora el punto de control")
    args = ap.parse_args(argv)
    REJILLA = REJILLA_DE[args.condicion]
    rejilla = REJILLA
    if args.solo:
        pedidos = [x.strip() for x in args.solo.split(",")]
        rejilla = tuple(n for n in REJILLA if n in pedidos)
        if len(rejilla) != len(pedidos):
            raise SystemExit(f"PARO: no están en la rejilla: {sorted(set(pedidos) - set(REJILLA))}")
    decl = declarados()
    comprobar_rejilla(rejilla, decl)
    if args.estimar:
        print(f"condición {args.condicion}; rejilla ({len(rejilla)}): {', '.join(rejilla)}\nsuelo de ruido: A repetido en {', '.join(REPETICION_DEL_SUELO)}\n"
              f"semilla {args.semilla}; estimación y techo: ver el protocolo §5")
        return 0
    if args.humo:
        # El humo solo comprueba el cableado (A/B, el lineal, el registro, el veredicto), en ~1 min: un conjunto, sin las redes.
        args.condicion = "sin_torch"
        rejilla = ("diabetes",)
        args.sin_suelo = True
    if args.condicion == "sin_torch":
        # El paquete CPU: ni la densa ni TabM (los interruptores del propio producto; rige la política de producción 120-C6).
        os.environ["MATRIXAI_ESTUDIO_SIN_TABM"] = "1"
        os.environ["MATRIXAI_ESTUDIO_SIN_DENSA"] = "1"
    if not args.salida.parent.exists():
        raise SystemExit(f"PARO: no existe {args.salida.parent}")
    estados = args.salida.with_name(args.salida.stem + "_estados")
    # Los estudios guardados del instrumento (checkpoints, modelos), FUERA de ~/.config/matrixai (no se ensucia el almacén del
    # Studio) y FUERA del árbol (la cola copia todo lo que un trabajo escribe en su worktree): un directorio temporal que se
    # borra al salir. Lo que importa de cada estudio ya va en `<salida>_estados/` y en el JSON.
    import atexit, shutil, tempfile                                                # noqa: PLC0415,E401
    almacen = tempfile.mkdtemp(prefix="139b_almacen_")
    atexit.register(shutil.rmtree, almacen, ignore_errors=True)
    os.environ["MATRIXAI_ESTUDIOS_DIR"] = almacen
    R = r0()
    gen = R.cargar_generador()
    pro = procedencia(args, rejilla)
    datos = {"procedencia": pro, "conjuntos": {}, "suelo_de_ruido": None, "comparaciones": [], "corte": None}
    previo: dict = {}
    if args.salida.exists() and not args.forzar:
        viejo = json.loads(args.salida.read_text())
        pv = viejo.get("procedencia", {})
        if (pv.get("sha256_guion") == pro["sha256_guion"] and pv.get("semilla") == pro["semilla"] and pv.get("condicion") == pro["condicion"]
                and pv.get("peso_del_lineal") == pro["peso_del_lineal"] and pv.get("commits") == pro["commits"]
                and not viejo.get("corte")):
            previo = viejo
    # Reutilizables: SOLO `completed` (un registro fallido no se reutiliza nunca) y de la misma variante.
    def reusable(nombre, variante):
        r = ((previo.get("conjuntos") or {}).get(nombre) or {}).get(variante)
        return r if r and r.get("estado") == "completed" and r.get("campeon") else None

    def medir(nombre, variante):
        r = reusable(nombre, variante)
        if r is not None:
            print(f"[{nombre} {variante}] reutilizado", flush=True)
            return r
        t = time.time()
        r = estudiar(nombre, decl[nombre], gen, variante, args.semilla, estados)
        print(f"[{ahora()}] {nombre} {variante}: {r.get('estado')} campeón={r.get('campeon')} {time.time() - t:.0f} s", flush=True)
        return r

    V = v3()
    # 1) EL SUELO DE RUIDO: A repetido en los tres más baratos (la mayor |diferencia| de la cifra de test; 0 si idénticas).
    if not args.sin_suelo:
        repetidas = {}
        for n in REPETICION_DEL_SUELO:
            if n not in rejilla:
                continue
            a1 = medir(n, "A")
            datos["conjuntos"].setdefault(n, {})["A"] = a1
            a2 = reusable(n, "A_repetida") or None
            if a2 is None:
                t = time.time()
                a2 = estudiar(n, decl[n], gen, "A", args.semilla, estados.with_name(estados.name + "_repeticion"))
                a2["variante"] = "A_repetida"
                print(f"[{ahora()}] {n} A_repetida: {a2.get('estado')} campeón={a2.get('campeon')} {time.time() - t:.0f} s", flush=True)
            datos["conjuntos"][n]["A_repetida"] = a2
            repetidas[n] = (a1, a2)
            guardar(args.salida, datos)
        try:
            datos["suelo_de_ruido"] = V.suelo_de_ruido({n: p[0] for n, p in repetidas.items()},
                                                       {n: p[1] for n, p in repetidas.items()}) if repetidas else 0.0
        except V.Incomparable as e:
            # Un campeón que cambia al repetir A, o que completa en una y no en la otra: el ruido NO es un número. No se
            # para (de noche): se registra, se usa el listón de siempre y la lectura queda MARCADA.
            datos["suelo_de_ruido"] = 0.0
            datos["suelo_de_ruido_incomparable"] = str(e)
        print(f"suelo de ruido: {datos['suelo_de_ruido']}", flush=True)
    else:
        datos["suelo_de_ruido"] = 0.0
    guardar(args.salida, datos)
    suelo = datos["suelo_de_ruido"]
    # 2) LA REJILLA: A y B de cada conjunto, la comparación y la regla 9 tras cada uno.
    comps: list[dict] = []
    for i, n in enumerate(rejilla):
        a = datos["conjuntos"].get(n, {}).get("A") or medir(n, "A")
        datos["conjuntos"].setdefault(n, {})["A"] = a
        guardar(args.salida, datos)
        b = medir(n, "B")
        datos["conjuntos"][n]["B"] = b
        parada = None
        try:
            parada = sin_control(a, b)
            if parada is None:
                comps.append(comparar(a, b, suelo))
        except V.Incomparable as e:
            parada = f"{n}: {e}"
        datos["comparaciones"] = comps
        if parada is not None:
            # Un conjunto en el que el instrumento no midió lo registrado NO cuenta (ni sube ni baja) y queda DICHO; la pasada
            # sigue (de noche no hay nadie que la reanude), salvo que ya sean tres: entonces es el instrumento, no el conjunto.
            datos.setdefault("excluidos_por_el_instrumento", []).append({"nombre": n, "motivo": parada})
            print("EXCLUIDO (el instrumento no midió lo registrado):", parada, flush=True)
            guardar(args.salida, datos)
            if len(datos["excluidos_por_el_instrumento"]) >= 3:
                datos["corte"] = {"tipo": "instrumento", "motivo": "tres conjuntos excluidos por el instrumento", "tras": n,
                                  "sin_medir": list(rejilla[i + 1:])}
                guardar(args.salida, datos)
                return 3
            continue
        faltan = len(rejilla) - i - 1
        motivo = V.corta(comps, faltan) if comps else None
        guardar(args.salida, datos)
        if motivo and "regla_9_habria_cortado" not in datos:
            # Por omisión NO se corta (de noche el tiempo está pagado y Roberto pidió el RECUENTO por conjunto): se anota dónde
            # habría cortado la regla 9 y se mide todo. Con `--cortar`, es la de C3′: se para ahí.
            datos["regla_9_habria_cortado"] = {"motivo": motivo, "tras": n, "faltaban": faltan}
            print("REGLA 9 — cortaría aquí:", motivo, flush=True)
        if motivo and args.cortar:
            datos["corte"] = {"tipo": "regla_9", "motivo": motivo, "tras": n, "sin_medir": list(rejilla[i + 1:])}
            guardar(args.salida, datos)
            break
    datos["veredicto"] = veredicto(comps) if comps else None
    datos["fin"] = ahora()
    guardar(args.salida, datos)
    for c in comps:
        dif = "" if c["diferencia"] is None else f"{c['diferencia']:+.2f}"
        print(f"{c['nombre']:34} {c['clase']:20} {dif:>7}  {c['campeon_r1']} -> {c['campeon_c3p']}", flush=True)
    print(json.dumps(datos["veredicto"], ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
