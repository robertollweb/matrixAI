#!/usr/bin/env python3
"""128 · C5 — el VEREDICTO de «la GPU en el paquete GPU»: el brazo B (TabM con `MATRIXAI_TABM_DISPOSITIVO=auto` y
`--gpus all`) contra el brazo A (`cpu`), medidos en el MISMO PC de Roberto. Un solo sitio: lo usa el medidor del kit
(`kit_128_c5/medir_c5.py`) para cortar, y quien lea el resultado.

    python3 veredicto_128_c5.py resultado_128_c5.json [--suelo X]

LA REGLA (C5′, la de Roberto del 09-10 13:40, «banda de 1 punto + tiempo»; contrato 128, «REGLA NUEVA DE LA GPU»).
Sustituye a la E2 («ninguno baja más que el suelo de ruido y alguno mejora»), que con un suelo de 0 contaba como baja
hasta −0,08 puntos. Por conjunto, d = cifra del test de B (GPU) − la de A (CPU), en puntos; suelo = el MAYOR de los dos
brazos; `m1 = max(1, suelo)` y `m2 = max(2, suelo)`:

- **igual**: |d| < m1 · **sube**: d ≥ m1 · **baja tolerada**: −m2 < d ≤ −m1 (no bloquea por sí sola; cuenta en contra)
  · **empeora**: d ≤ −m2, o deja de terminar (bloquea siempre).
- **mejora de tiempo**: el estudio de B tarda la MITAD o menos que el de A (pared) y el conjunto queda «igual».
- **ENTRA si** ninguno empeora y (suben + mejoras de tiempo) SUPERAN a las bajas toleradas (la regla 4 del 120, con el
  tiempo como mejora). Si no, el dispositivo se queda en `cpu` y el código no se retira (la señal lo permite).
- **El corte** (regla 9): en el primer conjunto que empeora o deja de terminar, o cuando las bajas toleradas ya no se
  pueden compensar ni con TODOS los que faltan.
- Una caída a CPU DECLARADA no cuenta como GPU: ni su subida ni su mejora de tiempo.

- **La cifra de un conjunto** es la del campeón en el TEST (`seleccion.evaluacion_final`), en puntos ×100: la misma que
  C3′ (`veredicto_120_c3p.cifra_de_test`, importada, no copiada). Los dos brazos tienen los mismos candidatos, así que
  no hay «el máximo sobre más candidatos» que suba por construcción; y es lo que recibe quien usa el modelo.
- **El suelo de ruido** es el de CADA brazo: los 4 conjuntos de control se miden dos veces en cada uno, y la mayor
  |diferencia| de la cifra entre las dos (0 si son idénticas) es su suelo. Se usa el MAYOR de los dos: la cifra de
  TabM depende del reloj (plazo de pared), así que el suelo de la CPU tampoco es cero, y en la GPU sumará el orden
  de las sumas de cuBLAS. Un campeón que cambia al repetir NO es un error (es ruido, y se declara); uno que completa
  en una repetición y no en la otra SÍ: el instrumento no es repetible, `Incomparable`.
- «Deja de completar» y «empieza a completar» cuentan aparte (la primera siempre impide entrar; la segunda cuenta como
  mejora).
- **Presencia**: el brazo B tiene que haber entrenado TabM en la GPU en algún conjunto (el dispositivo lo declara el
  motor) y el A en ninguno; si no, no se midió lo registrado y NO entra, diga lo que diga la tabla. Y (128, I1 de su
  auditoría) donde TabM es el CAMPEÓN del brazo B, su modelo ENTREGADO —el del reajuste final, el que recibe quien usa
  el Studio— tiene que haberse entrenado en la GPU o declarar su caída a la CPU: una búsqueda en GPU con un entregado
  de CPU callado (el B1) no es «la GPU».
- **Una caída a la CPU DECLARADA en el brazo B no es la GPU** (128, N3 de su 2.ª auditoría): si el modelo ENTREGADO de un
  conjunto declara `cayo_a_cpu`, ese conjunto se entrenó en la CPU: su mejora NO cuenta como mejora de la GPU (se dice
  «cayó a CPU: no cuenta como GPU»), y todas esas caídas se LISTAN en `veredicto.caidas_a_cpu_en_b`.
- **Lo que la comparación INCLUYE y no está medido** (128, N4): desde la reparación del B1 la densa ve CUDA en el
  paquete GPU, y eso cuesta RAM de host y tiempo. En el kit lo paga el brazo B y no el A: una diferencia de memoria,
  un OOM o más pared del B puede venir de ahí y no de TabM. No se pudo medir sin una GPU (`COSTE_DE_VER_CUDA`); el JSON
  recoge la RAM pico de cada brazo (`veredicto.ram_pico_mb_por_brazo`) para verlo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

EXACTO = 1e-9
#: C5′: la banda de «igual»/«sube» (puntos) y la bajada que ya es «empeora»; el suelo de ruido las SUBE si es mayor.
BANDA_MINIMA = 1.0
EMPEORA_MINIMA = 2.0
#: C5′: B tarda esta fracción de la pared de A o menos → mejora de tiempo (si el resultado queda «igual»).
FRACCION_DE_TIEMPO = 0.5
AQUI = Path(__file__).resolve().parent
#: aviso (NO regla): con un suelo así de alto la regla de arriba no distingue casi nada.
SUELO_QUE_AVISA = 2.0


def _cargar(nombre: str):
    """El veredicto de C3′, por ruta (sin tocar `sys.path`). Está al lado (repo) o al lado del kit."""
    for carpeta in (AQUI, AQUI.parent):
        ruta = carpeta / f"{nombre}.py"
        if ruta.is_file():
            spec = importlib.util.spec_from_file_location(nombre, ruta)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(f"{nombre}.py no está junto a veredicto_128_c5.py")


_v3 = _cargar("veredicto_120_c3p")
Incomparable = _v3.Incomparable
completa = _v3.completa
cifra_de_test = _v3.cifra_de_test
MOTOR_TABM = _v3.MOTOR_TABM


def suelo_de_ruido(primera: dict[str, dict], segunda: dict[str, dict]) -> float:
    """La mayor |diferencia| de la cifra de test entre las dos pasadas de los mismos conjuntos; 0.0 si son idénticas.
    A diferencia de C3′, un campeón que cambia al repetir es ruido (se declara en `campeones_que_cambian`), no un
    error. `Incomparable` si los conjuntos no son los mismos o uno completa en una pasada y no en la otra."""
    if set(primera) != set(segunda) or not primera:
        raise Incomparable(f"el control de repetición no tiene los mismos conjuntos: {sorted(primera)} / {sorted(segunda)}")
    peor = 0.0
    for n in sorted(primera):
        a, b = primera[n], segunda[n]
        if completa(a) != completa(b):
            raise Incomparable(f"{n}: completa en una repetición y no en la otra ({a.get('estado')} / {b.get('estado')})")
        if not completa(a):
            continue
        d = abs(cifra_de_test(a) - cifra_de_test(b))
        peor = max(peor, d if d >= EXACTO else 0.0)
    return peor


def campeones_que_cambian(primera: dict[str, dict], segunda: dict[str, dict]) -> list[str]:
    return sorted(n for n in primera if completa(primera[n]) and completa(segunda.get(n))
                  and primera[n].get("campeon") != segunda[n].get("campeon"))


#: VRAM de más (MB) sobre la de antes del estudio que se toma como «se usó la GPU» en el brazo B cuando el estado del
#: estudio no enseña el dispositivo de los intentos (solo el del modelo entregado). Prueba indirecta: por eso solo vale
#: en B, y solo para decir «sí se usó» (nunca para acusar al brazo A).
VRAM_QUE_DELATA_LA_GPU_MB = 200.0


def entreno_tabm_en_gpu(rec: dict | None, *, con_vram: bool = False) -> bool:
    """¿Entrenó TabM en la GPU en este conjunto? Lo que el motor DECLARÓ: el dispositivo del modelo entregado o, en
    cualquier intento que el estado enseñe, un `device`/`dispositivo` = `cuda` (`dispositivos_vistos`). Con
    `con_vram`, también la VRAM de más medida durante el estudio (`recursos.vram_extra_mb`)."""
    if not rec:
        return False
    entregado = rec.get("entrenamiento_del_entregado") or {}
    if entregado.get("dispositivo") == "cuda":
        return True
    if int((rec.get("dispositivos_vistos") or {}).get("cuda", 0)) > 0:
        return True
    extra = (rec.get("recursos") or {}).get("vram_extra_mb")
    return bool(con_vram and isinstance(extra, (int, float)) and extra >= VRAM_QUE_DELATA_LA_GPU_MB)


def entregado_de_tabm_en_cpu_sin_decirlo(rec: dict | None) -> bool:
    """128 (I1): ¿el campeón es TabM y su modelo ENTREGADO NO consta en la GPU NI declara su caída a la CPU? Es lo que
    pasaba con el B1 (la búsqueda en la GPU, el reajuste final en la CPU sin decirlo). Con el campeón en otro motor no
    hay entregado de TabM que mirar: `False`."""
    if not rec or not completa(rec) or rec.get("campeon") != MOTOR_TABM:
        return False
    entregado = rec.get("entrenamiento_del_entregado") or {}
    return not (entregado.get("dispositivo") == "cuda" or entregado.get("cayo_a_cpu") is True)


def cayo_a_cpu_el_entregado(rec: dict | None) -> bool:
    """128 (N3): ¿el modelo ENTREGADO de este conjunto declara que se entrenó en la CPU por una caída?"""
    return bool(rec and (rec.get("entrenamiento_del_entregado") or {}).get("cayo_a_cpu") is True)


def caidas_a_cpu(conjuntos: dict[str, dict]) -> list[dict]:
    """128 (N3): las caídas a CPU DECLARADAS por el modelo entregado de cada conjunto, con su motivo."""
    return [{"nombre": n, "campeon": r.get("campeon"),
             "por_que": (r.get("entrenamiento_del_entregado") or {}).get("por_que_cayo")
             or (r.get("entrenamiento_del_entregado") or {}).get("caida_a_cpu")}
            for n, r in sorted(conjuntos.items()) if cayo_a_cpu_el_entregado(r)]


#: 128 (N4): lo que la comparación INCLUYE y no se ha medido. NO es una regla: se dice junto al veredicto.
COSTE_DE_VER_CUDA = ("el brazo B lleva de más el coste de que la densa vea CUDA en el paquete GPU (RAM de host y "
                     "tiempo; NO medido): una diferencia de memoria o un OOM del B puede venir de ahí y no de TabM")


def presencia(conjuntos_a: dict[str, dict], conjuntos_b: dict[str, dict]) -> str | None:
    """El motivo para PARAR (o para no entrar) o None."""
    en_a = sorted(n for n, r in conjuntos_a.items() if completa(r) and entreno_tabm_en_gpu(r))
    if en_a:
        return f"el brazo A (CPU) entrenó TabM en la GPU en {en_a}: no es la referencia registrada"
    callados = sorted(n for n, r in conjuntos_b.items() if entregado_de_tabm_en_cpu_sin_decirlo(r))
    if callados:
        return (f"el brazo B entregó TabM entrenado en la CPU sin declarar la caída en {callados}: lo que se compara "
                "no es la GPU (el modelo entregado no la usó)")
    if not any(completa(r) and entreno_tabm_en_gpu(r, con_vram=True) for r in conjuntos_b.values()):
        return "el brazo B no entrenó TabM en la GPU en ningún conjunto: no se midió la GPU"
    return None


def clase_por_diferencia(d: float, suelo: float) -> str:
    """C5′: `sube` | `igual` | `baja_tolerada` | `empeora` según d = B − A (puntos) y el suelo. Los bordes son
    del lado que dice el contrato: sube con d ≥ m1; baja tolerada con d ≤ −m1 (y > −m2); empeora con d ≤ −m2."""
    m1, m2 = max(BANDA_MINIMA, suelo), max(EMPEORA_MINIMA, suelo)
    if d <= -m2 + EXACTO:
        return "empeora"
    if d <= -m1 + EXACTO:
        return "baja_tolerada"
    if d >= m1 - EXACTO:
        return "sube"
    return "igual"


def mejora_de_tiempo(pared_a, pared_b) -> bool:
    """C5′: ¿el estudio de B tardó la mitad o menos que el de A? Sin las dos paredes medidas, no."""
    return (isinstance(pared_a, (int, float)) and isinstance(pared_b, (int, float)) and pared_a > 0
            and pared_b <= FRACCION_DE_TIEMPO * pared_a + EXACTO)


def comparar(a: dict | None, b: dict, suelo: float) -> dict:
    """Un conjunto: brazo A (CPU) frente a B (GPU). `clase` ∈ sube | igual | mejora_de_tiempo | baja_tolerada |
    empeora | deja_de_completar | empieza_a_completar | ninguno_completa. `mejora_de_tiempo` es un «igual» cuyo
    estudio de B tardó la mitad o menos."""
    nombre = b.get("nombre") or (a or {}).get("nombre")
    ca, cb = completa(a), completa(b)
    base = {"nombre": nombre, "campeon_a": (a or {}).get("campeon"), "campeon_b": b.get("campeon"),
            "pared_a_s": (a or {}).get("pared_estudio_s"), "pared_b_s": b.get("pared_estudio_s"),
            "tabm_en_gpu_en_b": entreno_tabm_en_gpu(b, con_vram=True),
            "cayo_a_cpu_en_b": cayo_a_cpu_el_entregado(b)}
    if ca and cb:
        if a.get("tarea") != b.get("tarea"):
            raise Incomparable(f"{nombre}: tarea {a.get('tarea')!r} en A y {b.get('tarea')!r} en B")
        xa, xb = cifra_de_test(a), cifra_de_test(b)
        d = xb - xa
        clase = clase_por_diferencia(d, suelo)
        if clase == "igual" and mejora_de_tiempo(base["pared_a_s"], base["pared_b_s"]):
            clase = "mejora_de_tiempo"
        return {**base, "clase": clase, "a": xa, "b": xb, "diferencia": d}
    clase = "deja_de_completar" if ca else "empieza_a_completar" if cb else "ninguno_completa"
    return {**base, "clase": clase, "a": cifra_de_test(a) if ca else None,
            "b": cifra_de_test(b) if cb else None, "diferencia": None}


def _es_mejora(c: dict) -> bool:
    """C5′: sube, mejora de tiempo o empieza a completar; la de un entregado que cayó a la CPU NO cuenta (N3)."""
    return c["clase"] in ("sube", "mejora_de_tiempo", "empieza_a_completar") and not c.get("cayo_a_cpu_en_b")


def corta(comparaciones: list[dict], esperados: int | None = None) -> str | None:
    """El corte (regla 9) tras el último conjunto medido del brazo B: el último empeora o deja de completar, o las
    bajas toleradas ya no se pueden compensar ni con TODOS los que faltan (`esperados` = cuántos tiene la pasada;
    sin él solo se mira lo primero)."""
    ult = comparaciones[-1]
    if ult["clase"] == "empeora":
        return f"{ult['nombre']} empeora {ult['diferencia']:.2f} puntos (la banda de empeorar)"
    if ult["clase"] == "deja_de_completar":
        return f"{ult['nombre']} deja de completar"
    if esperados is not None:
        bajas = sum(1 for c in comparaciones if c["clase"] == "baja_tolerada")
        mejoras = sum(1 for c in comparaciones if _es_mejora(c))
        faltan = max(0, esperados - len(comparaciones))
        if bajas and bajas >= mejoras + faltan:
            return (f"{bajas} bajas toleradas y {mejoras} mejoras: ni con los {faltan} conjuntos que faltan se "
                    "compensan")
    return None


def veredicto(comparaciones: list[dict], suelo: float, *, esperados: int, presencia_motivo: str | None = None) -> dict:
    """La regla C5′. `esperados` = cuántos conjuntos debía haber (13): con menos (corte o pasada parcial), NO entra."""
    cl = Counter(c["clase"] for c in comparaciones)
    candidatas = [c for c in comparaciones if c["clase"] in ("sube", "mejora_de_tiempo", "empieza_a_completar")]
    # N3: la mejora de un conjunto cuyo entregado cayó a la CPU no es de la GPU
    suben = sum(1 for c in candidatas if c["clase"] != "mejora_de_tiempo" and _es_mejora(c))
    tiempo = sum(1 for c in candidatas if c["clase"] == "mejora_de_tiempo" and _es_mejora(c))
    mejoras_caidas = [c["nombre"] for c in candidatas if c.get("cayo_a_cpu_en_b")]
    bajas = cl["baja_tolerada"]
    motivos = []
    if presencia_motivo:
        motivos.append(presencia_motivo)
    if cl["empeora"]:
        motivos.append(f"empeoran (bajan {max(EMPEORA_MINIMA, suelo):g} puntos o más): "
                       f"{[c['nombre'] for c in comparaciones if c['clase'] == 'empeora']}")
    if cl["deja_de_completar"]:
        motivos.append(f"dejan de completar: {[c['nombre'] for c in comparaciones if c['clase'] == 'deja_de_completar']}")
    if suben + tiempo <= bajas:
        detalle = (f" (solo {mejoras_caidas}, que cayó a CPU: no cuenta como GPU)" if mejoras_caidas else "")
        motivos.append(f"las mejoras ({suben} suben + {tiempo} de tiempo) no superan a las bajas toleradas "
                       f"({bajas}: {[c['nombre'] for c in comparaciones if c['clase'] == 'baja_tolerada']})"
                       + detalle)
    if len(comparaciones) < esperados:
        motivos.append(f"solo {len(comparaciones)} de {esperados} conjuntos medidos")
    return {"regla": "C5′ (banda de 1 punto + tiempo): ninguno empeora ni deja de terminar y las mejoras "
                     "(suben + mejoras de tiempo) superan a las bajas toleradas",
            "suelo_de_ruido": suelo, "suelo_alto": suelo > SUELO_QUE_AVISA,
            "banda": {"igual_o_sube_desde": max(BANDA_MINIMA, suelo), "empeora_desde": max(EMPEORA_MINIMA, suelo),
                      "mejora_de_tiempo_si_pared_b_le_a_x": FRACCION_DE_TIEMPO},
            "conjuntos": [c["nombre"] for c in comparaciones], "esperados": esperados,
            "suben": suben, "mejoras_de_tiempo": tiempo, "bajas_toleradas": bajas,
            "mejoras_que_cayeron_a_cpu": mejoras_caidas,
            "coste_no_medido_en_b": COSTE_DE_VER_CUDA, "empeoran": cl["empeora"],
            "dejan_de_completar": cl["deja_de_completar"],
            "iguales": cl["igual"], "ninguno_completa": cl["ninguno_completa"],
            "tabm_campeon_en_b": [c["nombre"] for c in comparaciones if c["campeon_b"] == MOTOR_TABM],
            "entra": not motivos, "motivos_para_no_entrar": motivos}


def evaluar(doc: dict, suelo_dado: float | None = None) -> dict:
    """Todo el veredicto de un `resultado_128_c5.json`: el suelo (el mayor de los dos brazos, o `suelo_dado`), las
    comparaciones y la regla."""
    a, b = doc["brazos"]["A"], doc["brazos"]["B"]
    suelos = {}
    for k, brazo in (("A", a), ("B", b)):
        if brazo.get("repeticion_de_control"):
            suelos[k] = suelo_de_ruido({n: brazo["conjuntos"][n] for n in brazo["repeticion_de_control"]},
                                       brazo["repeticion_de_control"])
    if suelo_dado is not None:
        suelo = float(suelo_dado)
    elif len(suelos) == 2:
        suelo = max(suelos.values())
    else:
        raise Incomparable("falta la repetición de control de algún brazo: no hay suelo de ruido (usa --suelo con el "
                           "del resultado principal si es una pasada parcial)")
    comps = []
    for n, rec in b["conjuntos"].items():
        comps.append(comparar(a["conjuntos"].get(n), rec, suelo))
    esperados = int(doc.get("esperados") or len(a["conjuntos"]))
    v = veredicto(comps, suelo, esperados=esperados,
                  presencia_motivo=presencia(a["conjuntos"], b["conjuntos"]))
    v["suelos_por_brazo"] = suelos
    v["caidas_a_cpu_en_b"] = caidas_a_cpu(b["conjuntos"])           # con campeón y motivo (N3)
    v["ram_pico_mb_por_brazo"] = {                                  # N4: lo que el medidor midió de cada brazo
        k: max([x for x in ((r.get("recursos") or {}).get("ram_pico_mb") for r in brazo["conjuntos"].values())
                if isinstance(x, (int, float))], default=None) for k, brazo in (("A", a), ("B", b))}
    v["campeones_que_cambian_al_repetir"] = {
        k: campeones_que_cambian({n: brazo["conjuntos"][n] for n in brazo["repeticion_de_control"]},
                                 brazo["repeticion_de_control"])
        for k, brazo in (("A", a), ("B", b)) if brazo.get("repeticion_de_control")}
    return {"comparaciones": comps, "veredicto": v}


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 2
    doc = json.loads(Path(argv[1]).read_text())
    suelo = float(argv[argv.index("--suelo") + 1]) if "--suelo" in argv else None
    try:
        r = evaluar(doc, suelo)
    except Incomparable as e:
        print("INCOMPARABLE:", e); return 3
    for c in r["comparaciones"]:
        dif = "" if c["diferencia"] is None else f"{c['diferencia']:+.2f}"
        marca = "  [cayó a CPU: no cuenta como GPU]" if c.get("cayo_a_cpu_en_b") else ""
        print(f"{c['nombre']:34} {c['clase']:20} {dif:>7}  {c['campeon_a']} -> {c['campeon_b']}{marca}")
    print(json.dumps(r["veredicto"], ensure_ascii=False, indent=1))
    for k in r["veredicto"]["caidas_a_cpu_en_b"]:
        print(f"CAYÓ A CPU en el brazo B: {k['nombre']} (campeón {k['campeon']}, {k['por_que']}): no cuenta como GPU")
    print("NO MEDIDO, incluido en la comparación:", COSTE_DE_VER_CUDA)
    if (doc["brazos"]["B"] or {}).get("corte"):
        print("CORTADA:", doc["brazos"]["B"]["corte"])
    print("LA GPU ENTRA" if r["veredicto"]["entra"] else "LA GPU NO ENTRA (el dispositivo se queda en cpu)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
