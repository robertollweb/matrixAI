#!/usr/bin/env python3
"""128-C5 — el MEDIDOR del kit: corre DENTRO de un contenedor `python:3.12-slim` (solo biblioteca estándar) y habla con
el Studio por HTTP. Nada de `/proc`, señales de Linux ni `docker` aquí: el instrumento del 120 (`referencia_120_r0.py`)
se importa por ruta y se REUSA entero (la forma de pedir, de leer el estado y de registrar un estudio es la suya).

Lo que lanza el medidor a mano (normalmente lo hace `lanzar.ps1` / `lanzar.sh`):

    python3 medir_c5.py --brazo A --studio http://c5-128-studio-A:8765 --salida /salida --datos /work/kit/datos
    python3 medir_c5.py --brazo B --studio http://c5-128-studio-B:8765 ...      # exige el brazo A terminado
    python3 medir_c5.py --componer --salida /salida                              # junta A + B -> resultado_128_c5.json

Un brazo = los 13 conjuntos de referencia, uno tras otro, y después la REPETICIÓN de los 4 de control (el suelo de
ruido de ese brazo). Se corta y se relanza donde estaba: tras cada conjunto se escribe `brazo_<X>.json` (os.replace) y
al relanzar se reusan SOLO los `completed` con la misma procedencia. El brazo B se corta, como la regla 9 del 120, en
el primer conjunto que baja más que el suelo de ruido o deja de completar (`veredicto_128_c5.corta`).

Códigos de salida: 0 bien · 2 los datos no cuadran con el manifiesto · 3 el brazo A no está terminado · 4 el Studio ya
no contesta (el lanzador lo rearranca y vuelve a llamar) · 5 (`--prueba`) la GPU NO se usó · 6 incomparable, o lo que se
mide no es lo registrado (el brazo A entrenó TabM en la GPU, o el B entregó TabM de CPU sin decirlo: I1 del 128).

RELANZAR NO BORRA NADA (128, I2 de su auditoría). Un brazo ya TERMINADO con un corte por la regla 9 es su resultado: la
misma orden no lo repite (sale con 0 y el lanzador compone). Si hay que medirlo otra vez (`--forzar`), o el punto de
control no vale (otra imagen, otro ajuste, un corte del instrumento), el `brazo_<X>.json` de antes se GUARDA al lado
con su hora (`brazo_<X>.anterior-AAAAMMDD-HHMMSS.json`) antes de empezar: nunca se sobrescribe una pasada.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
TOPE_ESTUDIO_S = 3600           # por conjunto; pasado esto se cancela y se registra «plazo» (como el 120)
CONJUNTO_DE_PRUEBA = "dresses-sales"


def cargar(nombre: str):
    """Un módulo de al lado (kit) o del directorio de arriba (repo `benchmarks/fase0/`), por ruta."""
    for carpeta in (AQUI, AQUI.parent):
        ruta = carpeta / f"{nombre}.py"
        if ruta.is_file():
            spec = importlib.util.spec_from_file_location(nombre, ruta)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(f"{nombre}.py no está ni en {AQUI} ni en {AQUI.parent}")


R0 = cargar("referencia_120_r0")
V = cargar("veredicto_128_c5")


def sha(ruta) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def ahora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ los datos del kit
def leer_manifiesto(datos: Path) -> dict:
    return json.loads((Path(datos) / "manifiesto.json").read_text())


def verificar_datos(datos: Path) -> list[str]:
    """Los CSV del kit son los del manifiesto, byte a byte (sha256 del .gz y del CSV descomprimido). La lista de
    problemas: vacía = bien."""
    problemas = []
    for c in leer_manifiesto(datos)["conjuntos"]:
        ruta = Path(datos) / c["fichero"]
        if not ruta.is_file():
            problemas.append(f"{c['nombre']}: falta {c['fichero']}"); continue
        crudo = ruta.read_bytes()
        if hashlib.sha256(crudo).hexdigest() != c["sha256_gz"]:
            problemas.append(f"{c['nombre']}: el .gz no es el del manifiesto"); continue
        try:
            contenido = gzip.decompress(crudo)
        except OSError as e:
            problemas.append(f"{c['nombre']}: no se descomprime ({e})"); continue
        if hashlib.sha256(contenido).hexdigest() != c["sha256"] or len(contenido) != c["bytes"]:
            problemas.append(f"{c['nombre']}: el CSV no es el del manifiesto")
    return problemas


def csv_de(datos: Path, c: dict) -> tuple[bytes, list[str]]:
    contenido = gzip.decompress((Path(datos) / c["fichero"]).read_bytes())
    columnas = next(csv.reader(io.StringIO(contenido.decode("utf-8"), newline="")))
    return contenido, columnas


# ------------------------------------------------------------------ lo que se muestrea DENTRO del Studio
def leer_muestras(ruta: Path | None) -> list[dict]:
    """Las muestras de `muestreador.py` (una línea JSON cada ~2 s: t, ram_mb, vram_mb, gpu_util). Tolerante: una
    línea a medias, o ningún fichero, no es un error."""
    if not ruta or not Path(ruta).is_file():
        return []
    out = []
    for linea in Path(ruta).read_text(errors="replace").splitlines():
        try:
            m = json.loads(linea)
            if isinstance(m, dict) and isinstance(m.get("t"), (int, float)):
                out.append(m)
        except ValueError:
            continue
    return out


def picos(muestras: list[dict], t0: float, t1: float) -> dict:
    """Los picos de RAM del contenedor y de VRAM/uso de la GPU entre t0 y t1, y la VRAM de ANTES (la última muestra
    anterior a t0: lo que ya ocupaban otros procesos)."""
    dentro = [m for m in muestras if t0 <= m["t"] <= t1]
    antes = [m for m in muestras if m["t"] < t0]

    def maximo(k, lote):
        v = [m[k] for m in lote if isinstance(m.get(k), (int, float))]
        return round(max(v), 1) if v else None
    vram_antes = None
    for m in reversed(antes):
        if isinstance(m.get("vram_mb"), (int, float)):
            vram_antes = m["vram_mb"]; break
    vram_pico = maximo("vram_mb", dentro)
    return {"ram_pico_mb": maximo("ram_mb", dentro), "vram_pico_mb": vram_pico, "vram_antes_mb": vram_antes,
            "vram_extra_mb": (round(vram_pico - vram_antes, 1) if vram_pico is not None and vram_antes is not None
                              else None),
            "gpu_uso_pico_pct": maximo("gpu_util", dentro), "muestras": len(dentro)}


def dispositivos_vistos(o, cuenta: Counter | None = None) -> dict:
    """Cuántas veces el estado del estudio dice `device`/`dispositivo` = cpu | cuda, y cuántas `caida_a_cpu`,
    donde estén. Lo que el estado no enseñe no se inventa: sin nada, cuenta vacía."""
    cuenta = Counter() if cuenta is None else cuenta
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("device", "dispositivo") and v in ("cpu", "cuda"):
                cuenta[v] += 1
            elif k == "caida_a_cpu" and v:
                cuenta["caidas_a_cpu"] += 1
            else:
                dispositivos_vistos(v, cuenta)
    elif isinstance(o, list):
        for v in o:
            dispositivos_vistos(v, cuenta)
    return dict(cuenta)


# ------------------------------------------------------------------ un estudio
def esperar_al_studio(url: str, plazo_s: float = 300.0) -> dict:
    t0 = time.time()
    ultimo = None
    while time.time() - t0 < plazo_s:
        try:
            return R0.pedir(url, "/api/studio/status", plazo=10)
        except Exception as e:                       # noqa: BLE001 — arrancando
            ultimo = repr(e)
            time.sleep(3)
    raise TimeoutError(f"el Studio no contestó en {plazo_s:.0f} s: {ultimo}")


def estudiar(url: str, c: dict, datos: Path, estados: Path, muestreo: Path | None, *, tope_s: float = TOPE_ESTUDIO_S,
             pausa_s: float = 5.0) -> dict:
    contenido, columnas = csv_de(datos, c)
    cuerpo = {"csv_text": contenido.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in columnas if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    ini, t0 = ahora(), time.time()
    http, r = R0.pedir_o_rechazo(url, "/api/studio/estudio/iniciar", cuerpo)
    job = (r.get("job_id") or (r.get("estudio") or {}).get("job_id")) if isinstance(r, dict) else None
    if http != 200 or not job:
        rec = R0.registrar(c["nombre"], c, cuerpo, {"estado": "rechazado"}, time.time() - t0, http, r, [], ini)
        t1 = time.time()
    else:
        est: dict = {}
        while True:
            try:
                est = R0.pedir(url, f"/api/studio/estudio/estado/{job}", plazo=60)
            except urllib.error.HTTPError as e:
                est = {"estado": "sondeo_fallido", "error": repr(e)}
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                raise                               # el Studio ya no contesta: lo trata quien llama
            if est.get("estado") in ("completed", "failed", "cancelled", "sondeo_fallido"):
                break
            if time.time() - t0 > tope_s:
                try:
                    R0.pedir(url, f"/api/studio/estudio/cancelar/{job}", {}, 30)
                except Exception:                    # noqa: BLE001, S110
                    pass
                est = dict(est, estado="plazo"); break
            time.sleep(pausa_s)
        t1 = time.time()
        estados.mkdir(parents=True, exist_ok=True)
        (estados / f"{c['nombre']}.json").write_text(json.dumps(est, ensure_ascii=False, indent=1))
        rec = R0.registrar(c["nombre"], c, cuerpo, est, t1 - t0, 200, None, [], ini)
        rec["job_id"] = job
        rec["entrenamiento_del_entregado"] = (est.get("seleccion") or {}).get("entrenamiento_del_entregado")
        rec["dispositivos_vistos"] = dispositivos_vistos(est)
    rec["csv"] = {"filas": c["filas"], "columnas": c["columnas"], "bytes": c["bytes"], "sha256": c["sha256"]}
    rec["recursos"] = picos(leer_muestras(muestreo), t0, t1)
    rec["inicio_epoch"], rec["fin_epoch"] = round(t0, 1), round(t1, 1)
    return rec


# ------------------------------------------------------------------ un brazo
def guardar(ruta: Path, datos: dict) -> None:
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1))
    os.replace(tmp, ruta)


def apartar(ruta: Path) -> Path:
    """128 (I2): el `brazo_<X>.json` que NO se va a continuar se guarda al lado con su hora, antes de empezar otro.
    Devuelve dónde quedó. Nunca pisa uno ya apartado (mismo segundo: `-2`, `-3`…)."""
    # El nombre se compone entero (no con `with_suffix`: «brazo_A.anterior-…» ya tiene un punto y `with_suffix` lo
    # tomaba por la extensión y devolvía «brazo_A.json» otra vez: un bucle sin fin, cazado por su prueba).
    hora = time.strftime("%Y%m%d-%H%M%S")
    destino, n = ruta.with_name(f"{ruta.stem}.anterior-{hora}.json"), 1
    while destino.exists():
        n += 1
        destino = ruta.with_name(f"{ruta.stem}.anterior-{hora}-{n}.json")
    os.replace(ruta, destino)
    return destino


class Corte(Exception):
    """El brazo se para: el motivo y el código de salida."""

    def __init__(self, motivo: str, codigo: int):
        super().__init__(motivo)
        self.motivo, self.codigo = motivo, codigo


def medir_brazo(a: argparse.Namespace, estudiar_fn=estudiar) -> int:
    salida, datos = Path(a.salida), Path(a.datos)
    problemas = verificar_datos(datos)
    if problemas:
        print("PARO: los datos del kit no son los del manifiesto:\n  " + "\n  ".join(problemas), flush=True)
        return 2
    manifiesto = leer_manifiesto(datos)
    conjuntos = sorted(manifiesto["conjuntos"], key=lambda x: x["orden"])
    control = [c for c in conjuntos if c["es_control"]]
    solo = [x for x in a.solo.split(",") if x] if a.solo else None
    if a.prueba:
        solo = [CONJUNTO_DE_PRUEBA]
    if solo:
        desconocidos = sorted(set(solo) - {c["nombre"] for c in conjuntos})
        if desconocidos:
            print(f"PARO: conjuntos que no existen: {desconocidos}", flush=True); return 2
        conjuntos = [c for c in conjuntos if c["nombre"] in solo]
    ruta_brazo = salida / f"{'prueba_' if a.prueba else ''}brazo_{a.brazo}.json"
    estados = salida / "estados" / f"{'prueba_' if a.prueba else ''}{a.brazo}"
    muestreo = salida / f"muestras_{a.brazo}.jsonl"
    crudo_entorno = os.environ.get(a.entorno_de_env, "") if a.entorno_de_env else (a.entorno or "")
    entorno = json.loads(crudo_entorno) if crudo_entorno else {}
    huella = {"imagen_id": a.imagen_id, "sha256_medir_c5": sha(__file__), "sha256_veredicto": sha(V.__file__),
              "sha256_manifiesto": sha(datos / "manifiesto.json"), "memoria": a.memoria, "cpus": a.cpus,
              "entorno": entorno, "brazo": a.brazo}

    brazo_a = None
    if a.brazo == "B" and not a.sin_comparar:
        ruta_a = salida / f"{'prueba_' if a.prueba else ''}brazo_A.json"
        if not ruta_a.is_file() or not json.loads(ruta_a.read_text()).get("terminado"):
            print("PARO: el brazo B se compara con el A y el A no está terminado.", flush=True); return 3
        brazo_a = json.loads(ruta_a.read_text())

    previo: dict = {}
    anterior = json.loads(ruta_brazo.read_text()) if ruta_brazo.is_file() else None
    if anterior is not None and not a.forzar and anterior.get("huella") == huella:
        corte = anterior.get("corte") or {}
        if anterior.get("terminado") and corte.get("tipo") == "regla_9":
            # 128 (I2): la pasada que bajó ES el resultado. Relanzar la misma orden no la tira para medir otra vez
            # (eso permitiría repetir hasta que entrase): se deja como está y el lanzador compone con ella.
            print(f"El brazo {a.brazo} ya TERMINÓ con un corte por la regla 9 ({corte.get('motivo')}): es su resultado "
                  f"y no se repite. Para medirlo otra vez: --forzar (la pasada de ahora se guarda con su hora).",
                  flush=True)
            return 0
        if not corte or corte.get("tipo") == "servidor":
            previo = anterior
    if anterior is not None and not previo:
        print(f"La pasada anterior del brazo {a.brazo} no se continúa: se guarda en {apartar(ruta_brazo).name}",
              flush=True)
    doc = {"brazo": a.brazo, "huella": huella, "procedencia": {
               "studio": a.studio, "imagen": a.imagen, "imagen_id": a.imagen_id, "memoria": a.memoria, "cpus": a.cpus,
               "entorno": entorno, "inicio": ahora(), "prueba": bool(a.prueba), "solo": solo,
               "tope_por_conjunto_s": TOPE_ESTUDIO_S, "sha256_medir_c5": huella["sha256_medir_c5"],
               "sha256_veredicto_128_c5": huella["sha256_veredicto"],
               "sha256_referencia_120_r0": sha(R0.__file__)},
           "conjuntos": dict(previo.get("conjuntos") or {}),
           "repeticion_de_control": dict(previo.get("repeticion_de_control") or {}),
           "suelo_de_ruido": previo.get("suelo_de_ruido"), "comparaciones": {}, "corte": None, "terminado": False}
    try:
        doc["procedencia"]["status"] = esperar_al_studio(a.studio)
    except TimeoutError as e:
        print("PARO:", e, flush=True); return 4
    guardar(ruta_brazo, doc)

    def correr(c: dict, destino: dict, reusar: bool = True) -> dict:
        n = c["nombre"]
        if reusar and destino.get(n, {}).get("estado") == "completed":
            print(f"[{n}] reusado (completed, misma procedencia)", flush=True)
            return destino[n]
        print(f"[{ahora()}] brazo {a.brazo} · {n} ...", flush=True)
        try:
            rec = estudiar_fn(a.studio, c, datos, estados if destino is doc["conjuntos"] else estados / "repeticion",
                              muestreo if muestreo.is_file() else None)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            raise Corte(f"el Studio ya no contesta durante {n}: {e!r}"[:300], 4)
        destino[n] = rec
        guardar(ruta_brazo, doc)
        print(f"   {rec.get('estado')} {rec.get('pared_estudio_s')} s campeón={rec.get('campeon')} "
              f"VRAM extra={rec.get('recursos', {}).get('vram_extra_mb')} MB", flush=True)
        return rec

    def vigilar_lo_registrado(n: str) -> None:
        """128 (I1): lo que se mide tiene que ser lo registrado, conjunto a conjunto, y se PARA en cuanto no lo es (no
        tras horas): el brazo A no puede entrenar TabM en la GPU, y en el B, si TabM es el campeón, su modelo
        ENTREGADO tiene que constar en la GPU o declarar su caída (el B1: búsqueda en GPU, entregado de CPU callado)."""
        rec = doc["conjuntos"].get(n)
        if a.brazo == "A" and rec and V.completa(rec) and V.entreno_tabm_en_gpu(rec):
            motivo = f"{n}: el brazo A (CPU) entrenó TabM en la GPU: no es la referencia registrada"
        elif a.brazo == "B" and V.entregado_de_tabm_en_cpu_sin_decirlo(rec):
            e = (rec or {}).get("entrenamiento_del_entregado") or {}
            motivo = (f"{n}: TabM es el campeón y su modelo ENTREGADO se entrenó en '{e.get('dispositivo')}' sin "
                      "declarar la caída a la CPU: lo que se mide no es la GPU (B1 del 128)")
        else:
            return
        doc["corte"] = {"tipo": "instrumento", "motivo": motivo, "tras": n}
        raise Corte(motivo, 6)

    suelo_a = (brazo_a or {}).get("suelo_de_ruido")
    comps: list[dict] = []
    codigo = 0
    try:
        con_control = not a.prueba and all(c["nombre"] in {x["nombre"] for x in conjuntos} for c in control)
        primeros = [c for c in conjuntos if c["es_control"]] if con_control else []
        resto = [c for c in conjuntos if c not in primeros]
        for c in primeros:
            correr(c, doc["conjuntos"])
            vigilar_lo_registrado(c["nombre"])
        if con_control:
            for c in control:
                correr(c, doc["repeticion_de_control"], reusar=True)
            try:
                doc["suelo_de_ruido"] = V.suelo_de_ruido({c["nombre"]: doc["conjuntos"][c["nombre"]] for c in control},
                                                         doc["repeticion_de_control"])
            except V.Incomparable as e:
                doc["corte"] = {"tipo": "instrumento", "motivo": f"el control no se repite: {e}"}
                raise Corte(str(e), 6)
            guardar(ruta_brazo, doc)
            print(f"SUELO DE RUIDO del brazo {a.brazo} (los 4 de control, dos veces): {doc['suelo_de_ruido']:g} puntos",
                  flush=True)
        if a.suelo is not None:
            suelo = a.suelo
        elif brazo_a is not None and doc["suelo_de_ruido"] is not None:
            suelo = max(doc["suelo_de_ruido"], suelo_a or 0.0)
        else:
            suelo = doc["suelo_de_ruido"]
        doc["suelo_usado"] = suelo
        cola = list(primeros) + resto
        for i, c in enumerate(cola):
            if c not in primeros:
                correr(c, doc["conjuntos"])
                vigilar_lo_registrado(c["nombre"])
            if brazo_a is None or suelo is None:
                continue
            try:
                comp = V.comparar(brazo_a["conjuntos"].get(c["nombre"]), doc["conjuntos"][c["nombre"]], suelo)
            except V.Incomparable as e:
                doc["corte"] = {"tipo": "instrumento", "motivo": str(e), "tras": c["nombre"]}
                raise Corte(str(e), 6)
            comps.append(comp)
            doc["comparaciones"][c["nombre"]] = comp
            guardar(ruta_brazo, doc)
            dif = "" if comp["diferencia"] is None else f" {comp['diferencia']:+.2f} puntos"
            print(f"   contra el brazo A: {comp['clase']}{dif} ({comp['campeon_a']} -> {comp['campeon_b']}; "
                  f"{comp['pared_a_s']} -> {comp['pared_b_s']} s)", flush=True)
            porque = V.corta(comps)
            if porque and not a.sin_cortar:
                doc["corte"] = {"tipo": "regla_9", "motivo": porque, "tras": c["nombre"],
                                "sin_medir": [x["nombre"] for x in cola[i + 1:]]}
                print("CORTE:", porque, "-- sin medir:", doc["corte"]["sin_medir"], flush=True)
                break
    except Corte as e:
        doc["procedencia"]["fin"] = ahora()
        if not doc.get("corte"):
            doc["corte"] = {"tipo": "servidor" if e.codigo == 4 else "instrumento", "motivo": e.motivo}
        # un fallo del servidor NO es un resultado: no se marca terminado y el lanzador reintenta
        guardar(ruta_brazo, doc)
        print("PARO:", e.motivo, flush=True)
        return e.codigo
    doc["procedencia"]["fin"] = ahora()
    doc["terminado"] = True
    guardar(ruta_brazo, doc)
    if a.prueba and a.brazo == "B":
        rec = doc["conjuntos"].get(CONJUNTO_DE_PRUEBA, {})
        usada = V.entreno_tabm_en_gpu(rec, con_vram=True)
        print("PRUEBA:", "la GPU SE USO" if usada else "la GPU NO SE USO", "-- dispositivo del modelo:",
              (rec.get("entrenamiento_del_entregado") or {}).get("dispositivo"), "-- VRAM extra:",
              rec.get("recursos", {}).get("vram_extra_mb"), "MB", flush=True)
        if not usada:
            return 5
    return codigo


# ------------------------------------------------------------------ juntar los dos brazos
def componer(a: argparse.Namespace) -> int:
    salida = Path(a.salida)
    brazos = {}
    for k in ("A", "B"):
        ruta = salida / f"brazo_{k}.json"
        if not ruta.is_file():
            print(f"PARO: falta {ruta.name}", flush=True); return 3
        brazos[k] = json.loads(ruta.read_text())
    maquina = {}
    for fichero in sorted((salida / "maquina").glob("*.txt")) if (salida / "maquina").is_dir() else []:
        maquina[fichero.name] = fichero.read_text(errors="replace").strip()
    doc = {"procedencia": {"kit": "128-C5", "compuesto": ahora(), "maquina": maquina,
                           "sha256_veredicto_128_c5": sha(V.__file__), "sha256_medir_c5": sha(__file__)},
           "esperados": len(leer_manifiesto(Path(a.datos))["conjuntos"]) if a.datos else 13, "brazos": brazos}
    try:
        doc.update(V.evaluar(doc, a.suelo))
    except V.Incomparable as e:
        doc["veredicto"] = {"entra": False, "motivos_para_no_entrar": [f"incomparable: {e}"]}
    guardar(salida / "resultado_128_c5.json", doc)
    v = doc["veredicto"]
    print(json.dumps(v, ensure_ascii=False, indent=1))
    print("LA GPU ENTRA" if v.get("entra") else "LA GPU NO ENTRA (el dispositivo se queda en cpu)")
    print("Manda salida/resultado_128_c5.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brazo", choices=("A", "B"))
    ap.add_argument("--studio", help="URL del Studio de ese brazo (p. ej. http://c5-128-studio-A:8765)")
    ap.add_argument("--salida", default="/salida")
    ap.add_argument("--datos", default=str(AQUI / "datos"))
    ap.add_argument("--imagen", default=None)
    ap.add_argument("--imagen-id", dest="imagen_id", default=None)
    ap.add_argument("--memoria", default="6g")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--entorno", default=None, help="JSON con las variables de más que lleva el Studio de este brazo")
    ap.add_argument("--entorno-de-env", dest="entorno_de_env", default=None,
                    help="como --entorno, pero el JSON viene de esta VARIABLE DE ENTORNO (PowerShell 5.1 pierde las comillas)")
    ap.add_argument("--solo", default="", help="solo estos conjuntos (coma); la pasada es PARCIAL y sin control")
    ap.add_argument("--prueba", action="store_true", help="un conjunto pequeño por brazo, para comprobar la GPU")
    ap.add_argument("--forzar", action="store_true", help="ignora el punto de control")
    ap.add_argument("--suelo", type=float, default=None, help="suelo de ruido dado (pasadas parciales)")
    ap.add_argument("--sin-comparar", action="store_true", dest="sin_comparar")
    ap.add_argument("--sin-cortar", action="store_true", dest="sin_cortar")
    ap.add_argument("--componer", action="store_true")
    ap.add_argument("--verificar-datos", action="store_true", dest="verificar")
    a = ap.parse_args(argv)
    if a.verificar:
        problemas = verificar_datos(Path(a.datos))
        print("DATOS OK" if not problemas else "DATOS MAL:\n  " + "\n  ".join(problemas))
        return 0 if not problemas else 2
    if a.componer:
        return componer(a)
    if not a.brazo or not a.studio:
        ap.error("--brazo y --studio son obligatorios (o --componer / --verificar-datos)")
    return medir_brazo(a)


if __name__ == "__main__":
    sys.exit(main())
