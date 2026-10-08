#!/usr/bin/env python3
"""128 · C5 — el VEREDICTO de «la GPU en el paquete GPU»: el brazo B (TabM con `MATRIXAI_TABM_DISPOSITIVO=auto` y
`--gpus all`) contra el brazo A (`cpu`), medidos en el MISMO PC de Roberto. Un solo sitio: lo usa el medidor del kit
(`kit_128_c5/medir_c5.py`) para cortar, y quien lea el resultado.

    python3 veredicto_128_c5.py resultado_128_c5.json [--suelo X]

LA REGLA, la de Roberto (contrato 128 §2: «no podemos empeorar lo conseguido, es solo si mejoramos») y la E2 del
08-10 («la del 120»): **la GPU entra si NINGÚN conjunto baja más que el suelo de ruido medido repitiendo, NINGUNO deja
de terminar y ALGUNO mejora.** Si no, el dispositivo se queda en `cpu` y el código no se retira (la señal lo permite).

- **La cifra de un conjunto** es la del campeón en el TEST (`seleccion.evaluacion_final`), en puntos ×100: la misma que
  C3′ (`veredicto_120_c3p.cifra_de_test`, importada, no copiada). Los dos brazos tienen los mismos candidatos, así que
  no hay «el máximo sobre más candidatos» que suba por construcción; y es lo que recibe quien usa el modelo.
- **El suelo de ruido** es el de CADA brazo: los 4 conjuntos de control se miden dos veces en cada uno, y la mayor
  |diferencia| de la cifra entre las dos (0 si son idénticas) es su suelo. Se usa el MAYOR de los dos: la cifra de
  TabM depende del reloj (plazo de pared), así que el suelo de la CPU tampoco es cero, y en la GPU sumará el orden
  de las sumas de cuBLAS. Un campeón que cambia al repetir NO es un error (es ruido, y se declara); uno que completa
  en una repetición y no en la otra SÍ: el instrumento no es repetible, `Incomparable`.
- **Baja** = B − A < −suelo; **sube** = B − A > suelo; **igual**, dentro del suelo. «Deja de completar» y «empieza a
  completar» cuentan aparte (la primera siempre impide entrar; la segunda cuenta como mejora).
- **Presencia**: el brazo B tiene que haber entrenado TabM en la GPU en algún conjunto (el dispositivo lo declara el
  motor) y el A en ninguno; si no, no se midió lo registrado y NO entra, diga lo que diga la tabla. Y (128, I1 de su
  auditoría) donde TabM es el CAMPEÓN del brazo B, su modelo ENTREGADO —el del reajuste final, el que recibe quien usa
  el Studio— tiene que haberse entrenado en la GPU o declarar su caída a la CPU: una búsqueda en GPU con un entregado
  de CPU callado (el B1) no es «la GPU».
- **El corte** (como la regla 9 del 120): la pasada del brazo B se corta en el primer conjunto que baja más que el
  suelo o deja de completar. Lo no medido no cuenta como pérdida, pero sin los 13 el veredicto no entra.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

EXACTO = 1e-9
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


def comparar(a: dict | None, b: dict, suelo: float) -> dict:
    """Un conjunto: brazo A (CPU) frente a B (GPU). `clase` ∈ sube | baja | igual | deja_de_completar |
    empieza_a_completar | ninguno_completa."""
    nombre = b.get("nombre") or (a or {}).get("nombre")
    ca, cb = completa(a), completa(b)
    base = {"nombre": nombre, "campeon_a": (a or {}).get("campeon"), "campeon_b": b.get("campeon"),
            "pared_a_s": (a or {}).get("pared_estudio_s"), "pared_b_s": b.get("pared_estudio_s"),
            "tabm_en_gpu_en_b": entreno_tabm_en_gpu(b, con_vram=True)}
    if ca and cb:
        if a.get("tarea") != b.get("tarea"):
            raise Incomparable(f"{nombre}: tarea {a.get('tarea')!r} en A y {b.get('tarea')!r} en B")
        xa, xb = cifra_de_test(a), cifra_de_test(b)
        d = xb - xa
        clase = "baja" if d < -suelo - EXACTO else "sube" if d > suelo + EXACTO else "igual"
        return {**base, "clase": clase, "a": xa, "b": xb, "diferencia": d}
    clase = "deja_de_completar" if ca else "empieza_a_completar" if cb else "ninguno_completa"
    return {**base, "clase": clase, "a": cifra_de_test(a) if ca else None,
            "b": cifra_de_test(b) if cb else None, "diferencia": None}


def corta(comparaciones: list[dict]) -> str | None:
    """El corte tras el último conjunto medido del brazo B."""
    ult = comparaciones[-1]
    if ult["clase"] == "baja":
        return f"{ult['nombre']} baja {ult['diferencia']:.2f} puntos (más que el suelo de ruido)"
    if ult["clase"] == "deja_de_completar":
        return f"{ult['nombre']} deja de completar"
    return None


def veredicto(comparaciones: list[dict], suelo: float, *, esperados: int, presencia_motivo: str | None = None) -> dict:
    """La regla E2. `esperados` = cuántos conjuntos debía haber (13): con menos (corte o pasada parcial), NO entra."""
    cl = Counter(c["clase"] for c in comparaciones)
    suben = cl["sube"] + cl["empieza_a_completar"]
    motivos = []
    if presencia_motivo:
        motivos.append(presencia_motivo)
    if cl["baja"]:
        motivos.append(f"bajan más que el suelo de ruido ({suelo:g} puntos): "
                       f"{[c['nombre'] for c in comparaciones if c['clase'] == 'baja']}")
    if cl["deja_de_completar"]:
        motivos.append(f"dejan de completar: {[c['nombre'] for c in comparaciones if c['clase'] == 'deja_de_completar']}")
    if not suben:
        motivos.append("ninguno mejora")
    if len(comparaciones) < esperados:
        motivos.append(f"solo {len(comparaciones)} de {esperados} conjuntos medidos")
    return {"regla": "E2: ninguno baja más que el suelo de ruido, ninguno deja de terminar, alguno mejora",
            "suelo_de_ruido": suelo, "suelo_alto": suelo > SUELO_QUE_AVISA,
            "conjuntos": [c["nombre"] for c in comparaciones], "esperados": esperados,
            "suben": suben, "bajan": cl["baja"], "dejan_de_completar": cl["deja_de_completar"],
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
        print(f"{c['nombre']:34} {c['clase']:20} {dif:>7}  {c['campeon_a']} -> {c['campeon_b']}")
    print(json.dumps(r["veredicto"], ensure_ascii=False, indent=1))
    if (doc["brazos"]["B"] or {}).get("corte"):
        print("CORTADA:", doc["brazos"]["B"]["corte"])
    print("LA GPU ENTRA" if r["veredicto"]["entra"] else "LA GPU NO ENTRA (el dispositivo se queda en cpu)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
