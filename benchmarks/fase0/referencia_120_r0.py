#!/usr/bin/env python3
"""120 · R0 — la medida de REFERENCIA: el Studio publicado hoy (matrixai-studio:v2.7.2, CPU) sobre los 13 conjuntos
no sellados de 119-C5a, con el PROPIO estudio como instrumento. La cifra es `seleccion.media_de_la_seleccion`
(media de la métrica de selección en los pliegues de la validación cruzada); nunca el test; ninguna métrica se calcula aquí.

    python3 referencia_120_r0.py --solo-humo      # los 4 ejemplos + CONTROL contra lo medido el 01-10 (cierre v2.7.2)
    python3 referencia_120_r0.py                  # los 13 (el control va primero y, si no cuadra, PARA)
    python3 referencia_120_r0.py --forzar         # ignora el punto de control
    python3 referencia_120_r0.py --solo NOMBRE[,NOMBRE]   # (diagnóstico) solo esos; no hace el control si no están los 4

Punto de control: r0.json (reescrito con os.replace tras cada conjunto). Se reusan SOLO registros `completed` con la
misma imagen y el mismo guion. Solo LEE los repos (ni escribe, ni deja __pycache__: dont_write_bytecode).
"""
import argparse, atexit, re, urllib.error, csv, hashlib, importlib.util, io, json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HOME = Path.home()
AQUI = Path(__file__).resolve().parent
IMAGEN = "matrixai-studio:v2.7.2"   # R0; otra con --imagen (R1: el código con D9 y C1)
MEMORIA = "6g"                       # el techo de R0; otro con --memoria
BORRADOR = AQUI / "protocolo_120.json"   # el protocolo REGISTRADO (mismo directorio)
SONDA = AQUI / "resultado_sonda_119_c5a.json"
NUCLEO = AQUI.parents[1]              # benchmarks/fase0 -> el repo del núcleo; sus hermanos al lado
GENERADOR = NUCLEO.parent / "matrixaistudio/studio-backend/scripts/generar_ejemplos_medidos.py"
EJEMPLOS = NUCLEO.parent / "matrixaistudio/studio-backend/matrixai_studio/ejemplos_medidos"
SALIDA = AQUI / "referencia_120_r0.json"
ESTADOS = AQUI / "referencia_120_r0_estados"
TOPE_ESTUDIO_S = 3600          # por conjunto (sondeo); pasado esto, se registra «plazo» y se cancela
MENOR_ES_MEJOR = {"rmse", "mae", "log_loss", "brier"}
# Lo medido ESTA NOCHE contra la MISMA imagen (cierre-v272/ejemplos_v272.log), a 4 decimales.
CONTROL = {
    "dresses-sales": {"lightgbm": 0.5549, "sklearn.hgb": 0.6290},
    "climate-model-simulation-crashes": {"lightgbm": 0.9387, "sklearn.hgb": 0.9444},
    "us_crime": {"lightgbm": 0.1396, "sklearn.hgb": 0.1402},
    "pc1": {"lightgbm": 0.8841, "sklearn.hgb": 0.8777},
}
_contenedor = None
_RX_CAND = re.compile(r"-p(\d+)-r(\d+)$")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ datos: el MISMO conversor que los ejemplos del Studio
def cargar_generador():
    spec = importlib.util.spec_from_file_location("generar_ejemplos_medidos", GENERADOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def csv_de(gen, data_id):
    """Idéntico a `_dataset_medido` de generar_ejemplos_medidos.py: leer_arff + csv.writer(lineterminator='\\n')."""
    columnas, excluidas, filas = gen.leer_arff(gen.ARFF_DIR / f"{data_id}.arff")
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columnas)
    w.writerows(filas)
    return buf.getvalue().encode("utf-8"), columnas, excluidas, len(filas)


def comprobar_bytes(gen, conjuntos):
    """Para los 4 ejemplos: mi CSV == el del Studio, byte a byte. Si no, PARA."""
    res = {}
    for c in conjuntos:
        f = EJEMPLOS / f"{c['nombre']}.csv"
        if not f.is_file():
            continue
        mio = csv_de(gen, c["data_id"])[0]
        ok = mio == f.read_bytes()
        res[c["nombre"]] = {"byte_a_byte": ok, "sha256": hashlib.sha256(mio).hexdigest(), "bytes": len(mio)}
        if not ok:
            raise SystemExit(f"PARO: el CSV de {c['nombre']} NO es byte a byte el del Studio: {res[c['nombre']]}")
    return res


# ------------------------------------------------------------------ contenedor
def borrar_contenedor():
    global _contenedor
    if _contenedor:
        subprocess.run(["docker", "rm", "-f", _contenedor], capture_output=True)
        _contenedor = None


def estado_del_contenedor(nombre):
    """`OOMKilled`, código de salida y estado del contenedor (02-10: para saber POR QUÉ murió)."""
    r = subprocess.run(["docker", "inspect", nombre, "--format",
                        "{{.State.OOMKilled}}|{{.State.ExitCode}}|{{.State.Status}}|{{.State.Error}}"],
                       capture_output=True, text=True)
    if r.returncode:
        return {"inspeccion": "falló", "stderr": r.stderr.strip()[:300]}
    oom, codigo, estado, error = (r.stdout.strip().split("|") + ["", "", "", ""])[:4]
    return {"oom_killed": oom == "true", "codigo_de_salida": codigo, "estado": estado, "error": error}


def arrancar_contenedor():
    global _contenedor
    s = socket.socket(); s.bind(("127.0.0.1", 0)); puerto = s.getsockname()[1]; s.close()
    nombre = f"r0-120-{os.getpid()}-{int(time.time())}"
    _contenedor = nombre                       # antes de lanzarlo: si el run muere a medias, el atexit lo borra
    atexit.register(borrar_contenedor)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: (borrar_contenedor(), os._exit(143)))
    # SIN --rm (02-10): en R0 el servidor murió con APSFailure y `--rm` no dejó rastro de por qué; ahora
    # se lee `OOMKilled` antes de borrarlo (`estado_del_contenedor`).
    r = subprocess.run(["docker", "run", "-d", "--init", "--name", nombre, "-p", f"127.0.0.1:{puerto}:8765",
                        f"--memory={MEMORIA}", f"--memory-swap={MEMORIA}", "--cpus=2", "-e",
                        "MATRIXAI_LICENSE_ENABLED=false", IMAGEN],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("docker run falló: " + r.stderr)
    base = f"http://127.0.0.1:{puerto}"
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            pedir(base, "/api/studio/status", plazo=5)
            return base, nombre
        except Exception:
            time.sleep(2)
    raise SystemExit("el servidor no contestó en 120 s")


def pedir(base, ruta, cuerpo=None, plazo=120):
    datos = None if cuerpo is None else json.dumps(cuerpo).encode()
    req = urllib.request.Request(base + ruta, data=datos, method="POST" if cuerpo is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=plazo) as r:
        return json.loads(r.read().decode())


def pedir_o_rechazo(base, ruta, cuerpo, plazo=300):
    """Como `pedir`, pero un HTTP 4xx/5xx es un RESULTADO (lo que ve quien usa el Studio): se devuelve tal cual."""
    try:
        return 200, pedir(base, ruta, cuerpo, plazo)
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(txt)
        except Exception:
            return e.code, {"texto": txt[:4000]}


def dentro(nombre, codigo):
    r = subprocess.run(["docker", "exec", nombre, "python3", "-c", codigo], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


# ------------------------------------------------------------------ extracción tolerante del estado
def buscar_entradas(o, ruta=""):
    """Lista de dicts con wall_seconds (los intentos del leaderboard), donde estén. Devuelve (ruta, lista)."""
    if isinstance(o, list) and o and all(isinstance(x, dict) for x in o) and any("wall_seconds" in x or
            "wall_seconds" in (x.get("recursos") or {}) for x in o):
        return [(ruta, o)]
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            out += buscar_entradas(v, f"{ruta}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            out += buscar_entradas(v, f"{ruta}[{i}]")
    return out


def intentos_de(est):
    sel = est.get("seleccion") or {}
    for ruta, lista in buscar_entradas(est):
        filas = []
        for x in lista:
            rec = x.get("recursos") or {}
            m = _RX_CAND.search(str(x.get("candidate") or ""))
            ms = x.get("metrica_de_seleccion") or {}
            filas.append({"motor": x.get("engine"), "candidato": x.get("candidate"),
                          "pliegue": int(m.group(1)) if m else None, "repeticion": int(m.group(2)) if m else None,
                          "fase": x.get("fase"), "estado": x.get("estado"), "motivo": x.get("motivo"),
                          "wall_seconds": x.get("wall_seconds", rec.get("wall_seconds")),
                          "metrica": ms.get("metric_id"), "valor": ms.get("value"),
                          "valor_indefinido": ms.get("undefined_reason")})
        return ruta, filas
    return None, []


def registrar(nombre, c, cuerpo_ok, est, pared, http, rechazo, loadavg, ini):
    sel = est.get("seleccion") or {}
    media = sel.get("media_de_la_seleccion")
    ruta, filas = intentos_de(est)
    rec = {"nombre": nombre, "data_id": c["data_id"], "tarea": c["tarea"], "objetivo": c["objetivo"],
           "clase_positiva_enviada": cuerpo_ok.get("clase_positiva"), "http": http, "rechazo": rechazo,
           "estado": est.get("estado"), "campeon": sel.get("candidate_engine"), "media_de_la_seleccion": media,
           "motores_que_no_puntuaron": sel.get("motores_que_no_puntuaron") or est.get("motores_que_no_puntuaron"),
           "intentos": filas, "intentos_ruta_en_el_estado": ruta, "pared_estudio_s": round(pared, 1),
           "loadavg_al_empezar": loadavg, "inicio": ini, "fin": ahora()}
    return rec


def compiten(rec):
    m = rec.get("media_de_la_seleccion") or {}
    return {x["motor"]: x["valor"] for x in m.get("motores", []) if x.get("compite") and x.get("valor") is not None}


def estudiar(base, nombre, c, gen):
    csv_bytes, cols, excl, nfilas = csv_de(gen, c["data_id"])
    texto = csv_bytes.decode("utf-8")
    cuerpo = {"csv_text": texto, "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    ini = ahora(); la = open("/proc/loadavg").read().split()[:3]; t0 = time.time()
    http, r = pedir_o_rechazo(base, "/api/studio/estudio/iniciar", cuerpo)
    job = r.get("job_id") or (r.get("estudio") or {}).get("job_id") if isinstance(r, dict) else None
    if http != 200 or not job:
        rec = registrar(nombre, c, cuerpo, {"estado": "rechazado"}, time.time() - t0, http, r, la, ini)
        rec["csv"] = {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes), "excluidas_por_string": excl}
        return rec
    est = {}
    while True:
        try:
            est = pedir(base, f"/api/studio/estudio/estado/{job}", plazo=60)
        except Exception as e:
            est = {"estado": "sondeo_fallido", "error": repr(e)}
        if est.get("estado") in ("completed", "failed", "cancelled", "sondeo_fallido"):
            break
        if time.time() - t0 > TOPE_ESTUDIO_S:
            try: pedir(base, f"/api/studio/estudio/cancelar/{job}", {}, 30)
            except Exception: pass
            est = dict(est, estado="plazo"); break
        time.sleep(5)
    (ESTADOS / f"{nombre}.json").write_text(json.dumps(est, ensure_ascii=False, indent=1))
    rec = registrar(nombre, c, cuerpo, est, time.time() - t0, 200, None, la, ini)
    rec["job_id"] = job
    rec["csv"] = {"filas": nfilas, "columnas": len(cols), "bytes": len(csv_bytes), "excluidas_por_string": excl}
    return rec


def controlar(recs):
    """Los 4 ejemplos reproducen lo medido esta noche, a 4 decimales. Devuelve (ok, líneas)."""
    ok, lin = True, []
    for n, esperado in CONTROL.items():
        r = recs.get(n)
        if not r or r.get("estado") != "completed":
            ok = False; lin.append(f"{n}: NO completó ({r and r.get('estado')})"); continue
        got = compiten(r)
        for m, v in esperado.items():
            g = got.get(m)
            cuadra = g is not None and round(g, 4) == v
            ok &= cuadra
            lin.append(f"{n:34} {m:12} esperado {v:.4f} medido {('%.4f' % g) if g is not None else None} {'OK' if cuadra else 'NO CUADRA'}")
    return ok, lin


def guardar(datos):
    tmp = SALIDA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1))
    os.replace(tmp, SALIDA)


def main():
    global IMAGEN, SALIDA, ESTADOS, MEMORIA
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-humo", action="store_true"); ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--solo", default="")
    ap.add_argument("--imagen", default=IMAGEN)
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--memoria", default=MEMORIA)
    a = ap.parse_args()
    IMAGEN, MEMORIA = a.imagen, a.memoria
    SALIDA = Path(a.salida).resolve()
    ESTADOS = SALIDA.with_name(SALIDA.stem + "_estados")
    ESTADOS.mkdir(exist_ok=True)
    banco = json.loads(BORRADOR.read_text())["conjuntos"]["banco"]
    banco = sorted(banco, key=lambda x: x["orden"])
    declarados = {d["nombre"]: d for d in json.loads(SONDA.read_text())["datasets_declarados"]}
    for b in banco:                                             # lo declarado por 119-C5a manda sobre el borrador si difieren
        d = declarados[b["nombre"]]
        for k in ("data_id", "tarea", "objetivo", "clase_positiva"):
            if str(d[k]) != str(b[k]):
                raise SystemExit(f"PARO: {b['nombre']}.{k}: borrador {b[k]!r} != 119-C5a {d[k]!r}")
        if d.get("sellado"):
            raise SystemExit(f"PARO: {b['nombre']} está sellado")
    if a.solo_humo:
        banco = [b for b in banco if b["nombre"] in CONTROL]
    elif a.solo:
        banco = [b for b in banco if b["nombre"] in a.solo.split(",")]
    gen = cargar_generador()
    bytes_ok = comprobar_bytes(gen, [b for b in banco if b["nombre"] in CONTROL])
    print("CSV byte a byte contra los del Studio:", {k: v["byte_a_byte"] for k, v in bytes_ok.items()}, flush=True)

    imagen = subprocess.run(["docker", "image", "inspect", IMAGEN, "--format", "{{.Id}}"], capture_output=True, text=True).stdout.strip()
    guion = sha(__file__)
    previo = {}
    if SALIDA.exists() and not a.forzar:
        p = json.loads(SALIDA.read_text())
        if p.get("procedencia", {}).get("imagen_id") == imagen and p.get("procedencia", {}).get("sha256_guion") == guion:
            previo = {k: v for k, v in p.get("conjuntos", {}).items() if v.get("estado") == "completed"}
    datos = {"procedencia": {"imagen": IMAGEN, "imagen_id": imagen, "sha256_guion": guion,
                             "sha256_generador_de_csv": sha(GENERADOR),
                             "sha256_borrador_protocolo": sha(BORRADOR), "inicio": ahora(), "csv_byte_a_byte": bytes_ok,
                             "instrumento": "estudio del Studio por HTTP; folds/repeats por omisión; cifra = seleccion.media_de_la_seleccion",
                             "nota_multiclase": "clase_positiva solo se envía en binarias",
                             "memoria_del_contenedor": MEMORIA},
             "conjuntos": dict(previo), "control": None}
    base, nombre_c = arrancar_contenedor()
    srv = {"base": base, "nombre": nombre_c}
    datos["procedencia"]["contenedor"] = nombre_c
    datos["procedencia"]["dentro"] = dentro(nombre_c,
        "import matrixai, matrixai_studio.estudio_job as j; print('matrixai', matrixai.__version__, j.__file__)")
    print(datos["procedencia"]["dentro"], flush=True)
    try:
        datos["procedencia"]["status"] = pedir(base, "/api/studio/status")
    except Exception as e:
        datos["procedencia"]["status"] = repr(e)
    guardar(datos)

    def correr(b):
        n = b["nombre"]
        if n in previo:
            print(f"[{n}] reusado (completed, misma imagen y guion)", flush=True); return
        print(f"[{ahora()}] {n} (loadavg {open('/proc/loadavg').read().split()[0]}) ...", flush=True)
        try:
            rec = estudiar(srv["base"], n, b, gen)
        except (urllib.error.URLError, ConnectionError, OSError) as e:   # el servidor ya no contesta
            rec = {"nombre": n, "estado": "servidor_muerto", "error": repr(e)[:300], "http": None,
                   "pared_estudio_s": None, "campeon": None}
        if rec.get("estado") in ("sondeo_fallido", "servidor_muerto"):
            # SE DICE POR QUÉ y el siguiente conjunto arranca con un servidor NUEVO (02-10: en R0 una muerte
            # con APSFailure dejó Allstate sin medir y la pasada en rc=1).
            rec["contenedor"] = estado_del_contenedor(srv["nombre"])
            borrar_contenedor()
            srv["base"], srv["nombre"] = arrancar_contenedor()
        datos["conjuntos"][n] = rec
        guardar(datos)
        print(f"   {rec['estado']} http={rec['http']} {rec['pared_estudio_s']} s campeón={rec['campeon']} "
              f"{compiten(rec)}", flush=True)

    pend_control = [b for b in banco if b["nombre"] in CONTROL]
    for b in pend_control:
        correr(b)
    if all(n in datos["conjuntos"] for n in CONTROL if any(b["nombre"] == n for b in banco)) and len(pend_control) == 4:
        ok, lin = controlar(datos["conjuntos"])
        datos["control"] = {"cuadra": ok, "lineas": lin}
        guardar(datos)
        print("\n".join(lin)); print("CONTROL:", "CUADRA" if ok else "NO CUADRA — PARO", flush=True)
        if not ok:
            datos["procedencia"]["fin"] = ahora(); guardar(datos); sys.exit(2)
    if not a.solo_humo:
        for b in banco:
            if b["nombre"] not in CONTROL:
                correr(b)
    datos["procedencia"]["fin"] = ahora()
    guardar(datos)
    borrar_contenedor()


if __name__ == "__main__":
    main()
