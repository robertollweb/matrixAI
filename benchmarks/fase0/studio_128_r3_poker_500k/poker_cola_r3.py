"""128-R3, regla (2): poker-hand-500k en el SERVIDOR (cola nocturna: 2 CPU, 6 GB, 5 pliegues), un ESTUDIO ENTERO por el camino
real (`iniciar_estudio` en este proceso, el hilo del estudio de verdad, un proceso por intento y otro para el reajuste final),
con el código del 128-R3. Escrita ANTES de medir (la regla del contrato, 2026-10-14 07:50). ESTIMADA ~2.650 s (los 2.494 s del
128-R2 + ~120 s del reajuste de la densa con su techo), NO medida; tope de la cola 5.400 s.

Los criterios, comprobados al final, punto por punto (la regla (2) del 128-R3, `documentacion/128_CONJUNTOS_GRANDES_CONTRACT.md`):
  (a)  `completed`, con campeón y cifra de test;
  (b)  el reajuste de la densa con `wall_seconds` = el de la búsqueda (175 s; plazo 131,25 s): en el sobre (D,
       `entrenamiento_de_los_reajustes`) Y en el proceso `reajuste-final` (espía de solo lectura, `espia_cola/`);
  (c)  ninguna red ENTREGADA con `plazo_antes_de_la_primera_epoca_completa`; si el campeón es una red, `epocas_completas ≥ 1`
       y su reajuste en (D) tampoco lo dice. Las DEMÁS redes reajustadas con 0 épocas (con (E) no se entregan) se declaran
       en `informacion`, sin decidir (m4 de la re-auditoría del 128-R3);
  (c′) el reajuste de la densa, gane o no, NO por debajo de la referencia en las filas de selección del pliegue 0
       (`recomputo_de_seleccion.evaluaciones_de_origen`) y a ≤ 0,05 de su intento p0-r0 de la búsqueda;
  (d)  el test del campeón a ≤ 0,05 de su media de selección (`media_de_la_seleccion`) y ≥ la referencia en test (la
       referencia en test no va en el sobre: la mide el espía en el reajuste, sobre las MISMAS filas de test).
Y por dónde corrió: `__file__` de los tres paquetes en este proceso y en cada hijo (espía), dentro de los worktrees de la cola.

Y CON QUÉ CÓDIGO (m3 de la re-auditoría del 128-R3): el Studio fijado tiene que llevar el 128-R3 entero (`_FUNCIONES_DEL_128_R3`,
con `hasattr`, al arrancar), y este guion y el espía se FIJAN al encolar: `encolar.sh` los copia a `fijado/<sha256>/` y pasa sus
sha256 en `R3_SHA256_GUION` y `R3_SHA256_ESPIA`; al arrancar se comparan con lo que corre de verdad (este fichero y el
`sitecustomize` cargado) y, si no coinciden o faltan, NO se mide nada. Los sha256 quedan en `veredicto.json`.

El SABOTAJE OBLIGATORIO (regla (4)): el reajuste vuelto a la reserva fija pone roja esta regla en (b) y (c′). Lo comprueba
`--comprobar <estado.json> <espia.jsonl>` sobre cualquier estado ya escrito, sin correr nada (así se ve que los criterios tienen
dientes con el estado de un estudio con el defecto: el de la cola del 128-R2).

Qué escribe (en `./resultados-128-r3/`, bajo el directorio de la orden: la cola lo copia a
`~/cola-nocturna/resultados/<nombre>/`): `poker-hand-500k.estado.json` (el estado ENTERO), `espia.jsonl`, `veredicto.json` y
`veredicto.txt`. Y todo por la salida estándar (el registro de la cola).

Uso (lo pone `~/.agentes/128-r3/encolar.sh`): taskset -c 6,7 python3 poker_cola_r3.py   [--humo N] [--humo-grande]
  --humo N       SOLO para probar el guion: las N primeras filas (no mide nada de la regla);
  --humo-grande  con --humo, baja el umbral de «grande» para que el camino de (A) también se pruebe con pocas filas.
  --autoprueba   sin correr nada, sobre estados y ficheros fabricados: (c) (M3 de la auditoría: una red campeona sin
                 declaración da ROJO; m4: una red reajustada con 0 épocas que no se entrega NO pone rojo, la entregada sí),
                 la fijación del guion y del espía y la guarda del código del 128-R3 (m3), cada una en los dos sentidos.
Guarda `if __name__ == "__main__"`: cada intento es un `multiprocessing.spawn` que re-importa este guion."""
import gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

MIO = Path("/home/deployer/.agentes/128-r3")
CSV_GZ = MIO / "datos" / "poker-hand-500k.csv.gz"
SHA256_GZ = "1a2316ecf90c8c12dbd6054287aa56465e53497074fd0ebb6905f7af9ba7797b"      # el del manifiesto del kit 6
TOPE_S = 5000.0        # el estudio entero; la cola pone el suyo por encima (5400)
DENSA = "matrixai.dense.torch_cpu"
TABM = "matrixai.dense.tabm_cpu"
#: Las redes del estudio (`_MOTOR_DENSO` y `_MOTOR_TABM` de `estudio_job`): una red entregada DECLARA su entrenamiento.
REDES = (DENSA, TABM)
REF = "baseline"
TOLERANCIA = 0.05      # la del contrato, (c′) y (d)
#: m3 de la re-auditoría del 128-R3: lo que tiene que haber en el `estudio_job` fijado. Una función por pieza: (A)
#: `_reparto_del_reajuste_final` (también en `58b0f3a`), (E) `_reajuste_sin_ninguna_epoca` (desde `fde5979`) y m2
#: `_la_del_nucleo_entre_los_que_quedan` (solo en el commit final del corte). La guarda de verdad es la de antecesor de
#: `encolar.sh`; esta, al ARRANCAR, comprueba el código que se importó.
_FUNCIONES_DEL_128_R3 = ("_reparto_del_reajuste_final", "_reajuste_sin_ninguna_epoca", "_la_del_nucleo_entre_los_que_quedan")


def _lo_que_falta_del_128_r3(modulo) -> list[str]:
    """Las funciones del 128-R3 que NO tiene `modulo` (`[]`: es el código del corte entero)."""
    return [f for f in _FUNCIONES_DEL_128_R3 if not hasattr(modulo, f)]


def _sha256(ruta) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _espia_cargado() -> str | None:
    """El fichero del espía cargado en ESTE proceso: el `sitecustomize` que lleva los parches del espía (otro
    `sitecustomize`, el del sistema, no cuenta). `None` si no está."""
    modulo = sys.modules.get("sitecustomize")
    parches = getattr(modulo, "_PARCHES", None)
    if not isinstance(parches, dict) or "matrixai_engines.motor" not in parches:
        return None
    return getattr(modulo, "__file__", None)


def _codigo_de_la_medida(entorno, guion, espia) -> dict:
    """m3: ¿lo que corre es lo que se fijó al encolar? `guion`, este fichero; `espia`, el `sitecustomize` del espía cargado
    (`_espia_cargado`); `entorno`, donde `encolar.sh` dejó `R3_SHA256_GUION` y `R3_SHA256_ESPIA`. Devuelve lo que va a
    `veredicto.json`: rutas, sha256 de lo que corre y de lo fijado, y `motivos` (vacío = fijado). Sin las dos variables
    no está fijado: no se lanzó desde `encolar.sh`."""
    fijado_guion, fijado_espia = entorno.get("R3_SHA256_GUION"), entorno.get("R3_SHA256_ESPIA")
    corre_guion = _sha256(guion)
    corre_espia = _sha256(espia) if espia else None
    motivos = []
    if not fijado_guion or not fijado_espia:
        motivos.append("faltan R3_SHA256_GUION o R3_SHA256_ESPIA: no se lanzó desde encolar.sh, el guion y el espía no "
                       "están fijados")
    if fijado_guion and corre_guion != fijado_guion:
        motivos.append(f"el guion que corre ({guion}, sha256 {corre_guion}) no es el fijado ({fijado_guion})")
    if espia is None:
        motivos.append("el espía no está cargado en este proceso (ningún `sitecustomize` con sus parches)")
    elif fijado_espia and corre_espia != fijado_espia:
        motivos.append(f"el espía que corre ({espia}, sha256 {corre_espia}) no es el fijado ({fijado_espia})")
    return {"guion": {"ruta": str(guion), "sha256": corre_guion, "sha256_fijado": fijado_guion},
            "espia": {"ruta": None if espia is None else str(espia), "sha256": corre_espia, "sha256_fijado": fijado_espia},
            "motivos": motivos, "fijado": not motivos}


def _commit_de(ruta) -> str | None:
    """El commit del worktree en que está `ruta` (lo que de verdad se midió), o `None` si git no lo dice."""
    try:
        return subprocess.run(["git", "-C", str(Path(ruta).resolve().parent), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 -- es una declaración: si no se puede, se dice None
        return None


def _entregado_con_una_epoca(campeon, declarado) -> bool:
    """(c), el entregado. 128-R3 (reparación de su auditoría, M3): si el campeón es una RED, su declaración
    (`entrenamiento_del_entregado`) tiene que estar y decir `epocas_completas ≥ 1`. Una declaración AUSENTE no es «entrenó
    bien» (un valor ausente no es un cero), y unas `epocas_completas` ausentes tampoco. Un árbol no declara nada: pasa."""
    if campeon not in REDES:
        return True
    completas = None if declarado is None else declarado.get("epocas_completas")
    return isinstance(completas, int) and not isinstance(completas, bool) and completas >= 1


def _entregado_sin_plazo_antes_de_la_primera_epoca(campeon, declarado) -> bool:
    """(c), el entregado: sin `plazo_antes_de_la_primera_epoca_completa`. Con una red, también exige la declaración (M3)."""
    if campeon in REDES and declarado is None:
        return False
    return (declarado or {}).get("plazo_antes_de_la_primera_epoca_completa") is not True


def _cero_epocas(completas) -> bool:
    """`epocas_completas == 0` de verdad (un booleano no es un número de épocas: `False == 0` en Python)."""
    return isinstance(completas, int) and not isinstance(completas, bool) and completas == 0


def _valor(evaluacion, metrica):
    return next((m.get("value") for m in (evaluacion or {}).get("metrics") or [] if m.get("metric_id") == metrica), None)


def comprobar(est: dict, espia: list[dict], *, raiz_de_la_cola: str | None, techo_esperado: float | None = 175.0) -> dict:
    """La regla (2), punto por punto, sobre un estado ENTERO y lo que vio el espía. Sin correr nada. `techo_esperado`, el
    de poker-hand-500k (175 s); `None` solo en el humo, donde el techo es el que se declare (pocas filas)."""
    sel = est.get("seleccion") or {}
    metrica = (sel.get("media_de_la_seleccion") or {}).get("metrica") or "accuracy"
    campeon = sel.get("candidate_engine")
    candidato = (sel.get("decision") or {}).get("chosen_candidate")
    test = _valor(sel.get("evaluacion_final"), metrica)
    declarada = est.get("presupuesto_segun_el_tamano") or {}
    techo = declarada.get("segundos_por_intento")
    busqueda = sel.get("reparto_de_la_busqueda_segundos") or {}
    redes = {e.get("motor"): e for e in (sel.get("entrenamiento_de_los_reajustes") or [])}
    densa = redes.get(DENSA)
    origen = {p["candidato"]: _valor(p["evaluacion"], metrica)
              for p in ((sel.get("recomputo_de_seleccion") or {}).get("evaluaciones_de_origen") or [])}
    p0 = next((_valor({"metrics": [f.get("metrica_de_seleccion") or {}]}, metrica) for f in est.get("leaderboard") or []
               if f.get("engine") == DENSA and str(f.get("candidate", "")).endswith("-p0-r0")), None)
    medias = {m["motor"]: m.get("valor") for m in (sel.get("media_de_la_seleccion") or {}).get("motores") or []}
    del_reajuste = [r for r in espia if r.get("que") == "fit" and r.get("proceso") == "reajuste-final"]
    densa_en_el_hijo = [r for r in del_reajuste if r.get("candidato") == f"{DENSA}-seleccion"]
    test_espia = next((r for r in espia if r.get("que") == "test" and r.get("proceso") == "reajuste-final"), None)
    ref_test = (test_espia or {}).get("muestra_test_baseline")
    from matrixai_engines.motores import densa as motor_denso
    fraccion = motor_denso.FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR
    puntos = {
        "a_completed": est.get("estado") == "completed",
        "a_hay_campeon": campeon is not None,
        "a_hay_cifra_de_test": test is not None,
        "b_grande_con_techo": techo is not None and (techo_esperado is None or abs(techo - techo_esperado) < 1e-6),
        "b_techo_de_la_busqueda_de_la_densa_es_el_declarado": techo is not None and busqueda.get(DENSA) == techo,
        "b_sobre_la_densa_se_reajusto_con_ese_techo": bool(densa) and techo is not None
                                                     and densa.get("techo_del_reajuste_segundos") == techo,
        "b_sobre_plazo_de_la_densa": bool(densa) and techo is not None and (
            (densa.get("entrenamiento") or {}).get("plazo_de_entrenamiento_segundos") is not None
            and abs(densa["entrenamiento"]["plazo_de_entrenamiento_segundos"] - techo * fraccion) < 1e-6),
        "b_hijo_la_densa_recibio_ese_techo": bool(densa_en_el_hijo) and techo is not None
                                              and all(r.get("wall_seconds") == techo for r in densa_en_el_hijo),
        # (c), m4 de la re-auditoría del 128-R3: decide la red ENTREGADA. Una red entregada declara su entrenamiento
        # (`entrenamiento_del_entregado`; un árbol, `None`): ausente con una red, ROJO
        "c_si_el_campeon_es_una_red_tiene_una_epoca": _entregado_con_una_epoca(
            campeon, sel.get("entrenamiento_del_entregado")),
        "c_entregado_sin_plazo_antes_de_la_primera_epoca": _entregado_sin_plazo_antes_de_la_primera_epoca(
            campeon, sel.get("entrenamiento_del_entregado")),
        # y su reajuste en (D) dice lo mismo: la red entregada está en `entrenamiento_de_los_reajustes`, sin la bandera
        "c_el_reajuste_del_entregado_sin_plazo_antes_de_la_primera_epoca": campeon not in REDES or (
            campeon in redes and _entregado_sin_plazo_antes_de_la_primera_epoca(
                campeon, redes[campeon].get("entrenamiento"))),
        "c2_densa_no_por_debajo_de_la_referencia": (origen.get(f"{DENSA}-seleccion") is not None
                                                    and origen.get(f"{REF}-seleccion") is not None
                                                    and origen[f"{DENSA}-seleccion"] >= origen[f"{REF}-seleccion"]),
        "c2_densa_cerca_de_su_p0": (origen.get(f"{DENSA}-seleccion") is not None and p0 is not None
                                    and abs(origen[f"{DENSA}-seleccion"] - p0) <= TOLERANCIA),
        "d_test_cerca_de_su_media": (test is not None and medias.get(campeon) is not None
                                     and abs(test - medias[campeon]) <= TOLERANCIA),
        "d_test_no_por_debajo_de_la_referencia": test is not None and ref_test is not None and test >= ref_test,
    }
    if raiz_de_la_cola is not None:
        rutas = [p for r in espia for p in (r.get("modulos") or {}).values() if p]
        puntos["camino_hijos_dentro_de_la_cola"] = bool(rutas) and all(p.startswith(raiz_de_la_cola) for p in rutas)
    # m4: las DEMÁS redes reajustadas con 0 épocas (con (E) no se entregan): se DECLARAN, no deciden
    informacion = {"c_redes_reajustadas_sin_una_epoca_que_no_se_entregan": sorted(
        m for m, e in redes.items() if m != campeon and (
            (e.get("entrenamiento") or {}).get("plazo_antes_de_la_primera_epoca_completa") is True
            or _cero_epocas((e.get("entrenamiento") or {}).get("epocas_completas"))))}
    return {"cumple_la_regla_2": all(puntos.values()), "puntos": puntos, "informacion": informacion,
            "metrica": metrica, "campeon": campeon,
            "candidato": candidato, "test": test, "referencia_en_test": ref_test, "techo_por_intento": techo,
            "densa_reajuste": densa, "densa_en_el_hijo": densa_en_el_hijo, "p0_de_la_densa": p0,
            "seleccion_p0": origen, "medias": medias,
            "motores_que_no_puntuaron": sel.get("motores_que_no_puntuaron"),
            "entrenamiento_del_entregado": sel.get("entrenamiento_del_entregado"),
            "presupuesto_total_segundos": est.get("presupuesto_total_segundos"),
            "estado": est.get("estado"), "error": est.get("error"), "error_motivo": est.get("error_motivo")}


def _texto(veredicto: dict, pared: float | None) -> str:
    lineas = [f"REGLA (2) DEL 128-R3: {'CUMPLE' if veredicto['cumple_la_regla_2'] else 'NO CUMPLE'}"]
    lineas += [f"  {'ok ' if v else 'MAL'} {k}" for k, v in veredicto["puntos"].items()]
    lineas += [f"  info {k}: {v}" for k, v in (veredicto.get("informacion") or {}).items()]
    if veredicto.get("codigo_de_la_medida"):
        c = veredicto["codigo_de_la_medida"]
        lineas += [f"  código: guion {c['guion']['sha256']} | espía {c['espia']['sha256']} | fijado {c['fijado']}"]
    lineas += [f"  campeón: {veredicto['campeon']} | test {veredicto['test']} | referencia en test {veredicto['referencia_en_test']}"
               f" | techo {veredicto['techo_por_intento']} | pared {pared if pared is None else round(pared)} s",
               f"  densa: selección p0 {veredicto['seleccion_p0'].get(DENSA + '-seleccion')} | su p0-r0 {veredicto['p0_de_la_densa']}"
               f" | referencia {veredicto['seleccion_p0'].get(REF + '-seleccion')}",
               f"  reajuste de la densa: {json.dumps(veredicto['densa_reajuste'], ensure_ascii=False)}"]
    return "\n".join(lineas) + "\n"


def _estado_que_cumple() -> tuple[dict, list[dict]]:
    """Un estado ENTERO fabricado (con lo que vio el espía) que CUMPLE la regla (2): poker-hand-500k con la densa reajustada
    con su techo (175 s, 3 épocas), un árbol de campeón y las cifras dentro de sus tolerancias. Base de los casos de m4."""
    from matrixai_engines.motores import densa as motor_denso
    techo = 175.0
    def ev(v):
        return {"metrics": [{"metric_id": "accuracy", "value": v}]}
    red = {"plazo_de_entrenamiento_segundos": techo * motor_denso.FRACCION_DEL_PRESUPUESTO_PARA_ENTRENAR,
           "epocas_completas": 3, "plazo_antes_de_la_primera_epoca_completa": False}
    est = {"estado": "completed", "presupuesto_segun_el_tamano": {"segundos_por_intento": techo},
           "leaderboard": [{"engine": DENSA, "candidate": f"{DENSA}-p0-r0",
                            "metrica_de_seleccion": {"metric_id": "accuracy", "value": 0.60}}],
           "seleccion": {
               "candidate_engine": "lightgbm", "decision": {"chosen_candidate": "lightgbm-seleccion"},
               "evaluacion_final": ev(0.62), "reparto_de_la_busqueda_segundos": {DENSA: techo, "lightgbm": techo},
               "entrenamiento_de_los_reajustes": [{"motor": DENSA, "techo_del_reajuste_segundos": techo,
                                                   "entrenamiento": red}],
               "recomputo_de_seleccion": {"evaluaciones_de_origen": [
                   {"candidato": f"{DENSA}-seleccion", "evaluacion": ev(0.58)},
                   {"candidato": f"{REF}-seleccion", "evaluacion": ev(0.50)}]},
               "media_de_la_seleccion": {"metrica": "accuracy", "motores": [
                   {"motor": "lightgbm", "valor": 0.63}, {"motor": DENSA, "valor": 0.60}, {"motor": TABM, "valor": 0.61}]},
               "entrenamiento_del_entregado": None}}
    espia = [{"que": "fit", "proceso": "reajuste-final", "candidato": f"{DENSA}-seleccion", "wall_seconds": techo},
             {"que": "test", "proceso": "reajuste-final", "muestra_test_baseline": 0.50}]
    return est, espia


def autoprueba_c() -> int:
    """Sin correr nada, cada caso en los dos sentidos. Devuelve el número de casos que NO dan lo esperado (0 = bien).

    * (c), M3 de la auditoría: una red campeona sin declaración, o con 0 épocas, o con `epocas_completas` ausente, da ROJO;
      con 1 época, verde; un árbol sin declaración, verde.
    * (c), m4 de la re-auditoría: sobre un estado ENTERO que cumple, una red reajustada con 0 épocas que NO se entrega (TabM,
      o la densa) no pone rojo y se declara en `informacion`; la entregada con 0 épocas, o con su reajuste en (D) con la
      bandera o sin declarar, sí pone rojo.
    * m3: la fijación del guion y del espía (`_codigo_de_la_medida`) y la guarda del código (`_lo_que_falta_del_128_r3`)."""
    import copy
    import types
    fallos = 0

    def caso(nombre, dio, esperado):
        nonlocal fallos
        bien = dio == esperado
        fallos += not bien
        print(f"  {'ok ' if bien else 'MAL'} {nombre}: dio {dio}, se esperaba {esperado}")

    # (c), M3: los dos puntos del entregado
    def estado(campeon, declarado):
        return {"estado": "completed", "seleccion": {"candidate_engine": campeon, "entrenamiento_del_entregado": declarado}}
    una = {"epocas_completas": 1, "plazo_antes_de_la_primera_epoca_completa": False}
    cero = {"epocas_completas": 0, "plazo_antes_de_la_primera_epoca_completa": True}
    for nombre, campeon, declarado, esperado in [
            ("densa sin declaración", DENSA, None, (False, False)),
            ("TabM sin declaración", TABM, None, (False, False)),
            ("densa con 0 épocas", DENSA, cero, (False, False)),
            ("densa sin epocas_completas", DENSA, {"plazo_antes_de_la_primera_epoca_completa": False}, (False, True)),
            ("densa con epocas_completas=True (no es un número)", DENSA, {**una, "epocas_completas": True}, (False, True)),
            ("densa con 1 época", DENSA, una, (True, True)),
            ("TabM con 1 época", TABM, una, (True, True)),
            ("árbol sin declaración", "lightgbm", None, (True, True))]:
        puntos = comprobar(estado(campeon, declarado), [], raiz_de_la_cola=None)["puntos"]
        caso(f"(c) {nombre}", (puntos["c_si_el_campeon_es_una_red_tiene_una_epoca"],
                               puntos["c_entregado_sin_plazo_antes_de_la_primera_epoca"]), esperado)

    # (c), m4: sobre un estado ENTERO que cumple
    base, espia = _estado_que_cumple()

    def variante(cambiar):
        est = copy.deepcopy(base)
        cambiar(est["seleccion"])
        v = comprobar(est, espia, raiz_de_la_cola=None)
        return v["cumple_la_regla_2"], v["informacion"]["c_redes_reajustadas_sin_una_epoca_que_no_se_entregan"]
    def nada(sel):
        pass
    def tabm_cero_sin_entregar(sel):
        sel["entrenamiento_de_los_reajustes"].append({"motor": TABM, "techo_del_reajuste_segundos": 300.0,
                                                      "entrenamiento": dict(cero)})
    def densa_cero_sin_entregar(sel):
        sel["entrenamiento_de_los_reajustes"][0]["entrenamiento"].update(cero)
    def densa_cero_entregada(sel):
        densa_cero_sin_entregar(sel)
        sel.update(candidate_engine=DENSA, entrenamiento_del_entregado=dict(cero),
                   decision={"chosen_candidate": f"{DENSA}-seleccion"})
    def tabm_entregada(sel):
        sel["entrenamiento_de_los_reajustes"].append({"motor": TABM, "techo_del_reajuste_segundos": 300.0,
                                                      "entrenamiento": dict(una)})
        sel.update(candidate_engine=TABM, entrenamiento_del_entregado=dict(una),
                   decision={"chosen_candidate": f"{TABM}-seleccion"})
    def tabm_entregada_con_su_reajuste_sin_epoca(sel):
        tabm_entregada(sel)
        sel["entrenamiento_de_los_reajustes"][-1]["entrenamiento"] = dict(cero)
    def tabm_entregada_sin_su_reajuste(sel):
        tabm_entregada(sel)
        sel["entrenamiento_de_los_reajustes"].pop()
    for nombre, cambiar, esperado in [
            ("control: el estado entero CUMPLE", nada, (True, [])),
            ("TabM reajustada con 0 épocas que NO se entrega: verde, y se declara", tabm_cero_sin_entregar, (True, [TABM])),
            ("la densa reajustada con 0 épocas que NO se entrega: verde, y se declara", densa_cero_sin_entregar,
             (True, [DENSA])),
            ("la densa ENTREGADA con 0 épocas: ROJO", densa_cero_entregada, (False, [])),
            ("control: TabM entregada con 1 época en los dos sitios: verde", tabm_entregada, (True, [])),
            ("TabM entregada, pero su reajuste en (D) con la bandera: ROJO", tabm_entregada_con_su_reajuste_sin_epoca,
             (False, [])),
            ("TabM entregada sin su reajuste en (D): ROJO", tabm_entregada_sin_su_reajuste, (False, []))]:
        caso(f"(c) m4 {nombre}", variante(cambiar), esperado)

    # m3: la fijación del guion y del espía, con ficheros fabricados
    with tempfile.TemporaryDirectory() as d:
        guion, espia_f = Path(d) / "guion.py", Path(d) / "sitecustomize.py"
        guion.write_text("# el guion\n", encoding="utf-8")
        espia_f.write_text("# el espía\n", encoding="utf-8")
        bien = {"R3_SHA256_GUION": _sha256(guion), "R3_SHA256_ESPIA": _sha256(espia_f)}
        for nombre, entorno, espia_cargado, esperado in [
                ("lo fijado es lo que corre: fijado", bien, espia_f, (True, 0)),
                ("el guion cambió tras encolar: NO", {**bien, "R3_SHA256_GUION": "0" * 64}, espia_f, (False, 1)),
                ("el espía cambió tras encolar: NO", {**bien, "R3_SHA256_ESPIA": "0" * 64}, espia_f, (False, 1)),
                ("sin las variables (no se lanzó desde encolar.sh): NO", {}, espia_f, (False, 1)),
                ("el espía no está cargado: NO", bien, None, (False, 1))]:
            c = _codigo_de_la_medida(entorno, guion, espia_cargado)
            caso(f"m3 fijación: {nombre}", (c["fijado"], len(c["motivos"])), esperado)
        c = _codigo_de_la_medida(bien, guion, espia_f)
        caso("m3 fijación: los sha256 que van al veredicto", (c["guion"]["sha256"], c["espia"]["sha256"]),
             (bien["R3_SHA256_GUION"], bien["R3_SHA256_ESPIA"]))
    # m3: la guarda del código del 128-R3 (los commits del corte, por las funciones que tienen)
    todas = dict.fromkeys(_FUNCIONES_DEL_128_R3, lambda *a, **k: None)
    for nombre, quitar, esperado in [
            ("el commit final del 128-R3: no falta nada", (), []),
            ("como 58b0f3a, sin (E) ni m2", ("_reajuste_sin_ninguna_epoca", "_la_del_nucleo_entre_los_que_quedan"),
             ["_reajuste_sin_ninguna_epoca", "_la_del_nucleo_entre_los_que_quedan"]),
            ("como fde5979, sin m2", ("_la_del_nucleo_entre_los_que_quedan",), ["_la_del_nucleo_entre_los_que_quedan"]),
            ("como bf0178a, sin nada", _FUNCIONES_DEL_128_R3, list(_FUNCIONES_DEL_128_R3))]:
        modulo = types.SimpleNamespace(**{k: v for k, v in todas.items() if k not in quitar})
        caso(f"m3 código: {nombre}", _lo_que_falta_del_128_r3(modulo), esperado)
    print(f"AUTOPRUEBA: {'TODO BIEN' if not fallos else f'{fallos} MAL'}")
    return fallos


def main():
    if "--autoprueba" in sys.argv:
        sys.exit(1 if autoprueba_c() else 0)
    if "--comprobar" in sys.argv:
        i = sys.argv.index("--comprobar")
        est = json.loads(Path(sys.argv[i + 1]).read_text(encoding="utf-8"))
        espia = ([json.loads(l) for l in Path(sys.argv[i + 2]).read_text(encoding="utf-8").splitlines() if l.strip()]
                 if len(sys.argv) > i + 2 else [])
        veredicto = comprobar(est, espia, raiz_de_la_cola=None)
        print(_texto(veredicto, None), end="")
        return
    humo = int(sys.argv[sys.argv.index("--humo") + 1]) if "--humo" in sys.argv else None
    # m3: ANTES DE NADA, ¿corre lo que se fijó al encolar? Si no, no se mide (un humo, a mano, sí sigue: no decide nada).
    codigo = _codigo_de_la_medida(os.environ, Path(__file__).resolve(), _espia_cargado())
    print("código de la medida:", json.dumps(codigo, ensure_ascii=False), flush=True)
    if not codigo["fijado"] and not humo:
        print("NO SE MIDE: " + "; ".join(codigo["motivos"]), file=sys.stderr, flush=True)
        sys.exit(3)
    salida = Path("resultados-128-r3-humo" if humo else "resultados-128-r3")
    salida.mkdir(exist_ok=True)
    estudios = MIO / ("cola-estudios-humo" if humo else "cola-estudios")
    modelos = MIO / ("cola-modelos-humo" if humo else "cola-modelos")
    estudios.mkdir(parents=True, exist_ok=True)
    modelos.mkdir(parents=True, exist_ok=True)
    # NUNCA en `~/.config/matrixai/studio-estudios`: las dos variables (nombres comprobados en `estudio_job` y `uso`/`endpoints`).
    os.environ["MATRIXAI_ESTUDIOS_DIR"] = str(estudios)
    os.environ["MATRIXAI_MODELS_DIR"] = str(modelos)
    espia_log = Path(os.environ.get("ESPIA_R3_COLA_LOG", ""))
    assert str(espia_log).startswith(str(MIO)), "el espía de solo lectura no está activo (ESPIA_R3_COLA_LOG)"
    if espia_log.exists():      # lo de una pasada anterior (un humo) no puede leerse como de esta: se aparta, no se borra
        espia_log.rename(espia_log.with_name(f"{espia_log.name}.anterior-{time.strftime('%Y%m%d-%H%M%S')}"))
    assert hashlib.sha256(CSV_GZ.read_bytes()).hexdigest() == SHA256_GZ, "el CSV no es el del manifiesto del kit 6"
    from matrixai_studio import estudio_job
    import matrixai, matrixai_engines
    esperado = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p.endswith("studio-backend")]
    raiz_de_la_cola = esperado[0].removesuffix("matrixaistudio/studio-backend") if esperado else None
    print("estudio_job:", estudio_job.__file__, "| núcleo:", matrixai.__file__, "| motores:", matrixai_engines.__file__,
          "| afinidad", sorted(os.sched_getaffinity(0)), "| PYTHONPATH studio:", esperado, flush=True)
    assert esperado and estudio_job.__file__.startswith(esperado[0]), "el estudio NO sale del worktree fijado por la cola"
    assert all(m.__file__.startswith(raiz_de_la_cola) for m in (matrixai, matrixai_engines)), "núcleo o motores de otro sitio"
    falta = _lo_que_falta_del_128_r3(estudio_job)
    assert not falta, f"este código no es el del 128-R3 entero: le falta {falta}"
    if humo and "--humo-grande" in sys.argv:
        # SOLO humo: el camino de (A) con pocas filas (umbral bajado, y un coeficiente que da ~30 s por intento)
        estudio_job._UMBRAL_DE_CELDAS_GRANDE = 1000
        estudio_job._SEGUNDOS_POR_CELDA_GRANDE = 30.0 / (humo * 10 * 10)
    print("presupuesto por tamaño:", estudio_job._UMBRAL_DE_CELDAS_GRANDE, estudio_job._SEGUNDOS_POR_CELDA_GRANDE,
          "| factor de clases (multiclase, 10):", estudio_job._factor_de_clases("multiclass_classification", 10), flush=True)

    texto = gzip.decompress(CSV_GZ.read_bytes()).decode("utf-8")
    if humo:
        texto = "\n".join(texto.split("\n")[:humo + 1]) + "\n"
    cabecera = texto.split("\n", 1)[0].split(",")
    cuerpo = {"csv_text": texto, "objetivo": "Class", "tarea": "multiclass_classification",
              "entradas": [c for c in cabecera if c != "Class"], "unidad_de_observacion": "una fila"}
    owner = "128-r3"
    t0 = time.time()
    est = None
    try:
        r = estudio_job.iniciar_estudio(cuerpo, owner=owner, directorio_de_estudios=estudios)
        job_id = (r.get("estudio") or {}).get("job_id")
        print("iniciar:", {k: v for k, v in (r.get("estudio") or {}).items() if k != "motivo"}, flush=True)
        assert job_id, r
        ultimo = None
        while True:
            est = estudio_job.estudio_status(job_id, owner)
            visto = (est.get("estado"), est.get("n_intentos_completados"))
            if visto != ultimo:
                print(f"[{time.time() - t0:7.0f}s] {visto[0]} intentos completados {visto[1]}/{est.get('n_intentos_total')}",
                      flush=True)
                ultimo = visto
            if est.get("estado") in ("completed", "failed", "cancelled"):
                break
            if time.time() - t0 > TOPE_S:
                print("TOPE del guion: se cancela", flush=True)
                estudio_job.cancelar_estudio(job_id, owner=owner)
                time.sleep(20)
                est = estudio_job.estudio_status(job_id, owner)
                break
            time.sleep(5)
        pared = time.time() - t0
    finally:
        shutil.rmtree(estudios, ignore_errors=True)
        shutil.rmtree(modelos, ignore_errors=True)

    (salida / "poker-hand-500k.estado.json").write_text(json.dumps(est, ensure_ascii=False, indent=1, default=str))
    espia = [json.loads(l) for l in espia_log.read_text(encoding="utf-8").splitlines() if l.strip()] if espia_log.exists() else []
    shutil.copy(espia_log, salida / "espia.jsonl") if espia_log.exists() else None
    veredicto = comprobar(json.loads(json.dumps(est, default=str)), espia, raiz_de_la_cola=raiz_de_la_cola,
                          techo_esperado=None if humo else 175.0)
    veredicto.update({"pared_s": round(pared, 1), "segundos_reales": est.get("segundos_reales"),
                      "humo": humo, "loadavg_final": open("/proc/loadavg").read().split()[:3],
                      "codigo_de_la_medida": {**codigo, "commits": {
                          "matrixaistudio": _commit_de(estudio_job.__file__), "matrixAI": _commit_de(matrixai.__file__),
                          "matrixai-engines": _commit_de(matrixai_engines.__file__)}}})
    (salida / "veredicto.json").write_text(json.dumps(veredicto, ensure_ascii=False, indent=1, default=str))
    (salida / "veredicto.txt").write_text(_texto(veredicto, pared))
    print(_texto(veredicto, pared), flush=True)


if __name__ == "__main__":
    main()
