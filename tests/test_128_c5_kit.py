"""128 · C5 — el KIT del PC de Roberto (`benchmarks/fase0/kit_128_c5/`): los datos, el medidor contra un Studio SIMULADO
(un servidor HTTP local con estados fabricados: aquí no hay GPU), el punto de control, el corte y los lanzadores."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"
KIT = _FASE0 / "kit_128_c5"


def _cargar(ruta: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    antes = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = antes
    return mod


M = _cargar(KIT / "medir_c5.py", "medir_c5_en_pruebas")
V = M.V
CONTROL = ["dresses-sales", "climate-model-simulation-crashes", "us_crime", "pc1"]
pytestmark = pytest.mark.skipif(not (KIT / "datos" / "manifiesto.json").is_file(), reason="el kit no tiene sus datos generados")


# ------------------------------------------------------------------------------------------------ los datos
def test_los_13_conjuntos_del_kit_son_los_del_instrumento_del_120():
    manifiesto = M.leer_manifiesto(KIT / "datos")
    banco = json.loads((_FASE0 / "protocolo_120.json").read_text())["conjuntos"]["banco"]
    esperado = [(b["orden"], b["nombre"], b["data_id"], b["tarea"], b["objetivo"], b["clase_positiva"])
                for b in sorted(banco, key=lambda x: x["orden"])]
    obtenido = [(c["orden"], c["nombre"], c["data_id"], c["tarea"], c["objetivo"], c["clase_positiva"])
                for c in sorted(manifiesto["conjuntos"], key=lambda x: x["orden"])]
    assert len(obtenido) == 13 and obtenido == esperado
    assert [c["nombre"] for c in manifiesto["conjuntos"] if c["es_control"]] == CONTROL


def test_los_datos_del_kit_cuadran_con_su_manifiesto_byte_a_byte():
    assert M.verificar_datos(KIT / "datos") == []


def test_un_byte_cambiado_en_un_CSV_del_kit_se_detecta(tmp_path):
    datos = tmp_path / "datos"
    shutil.copytree(KIT / "datos", datos)
    ruta = datos / "pc1.csv.gz"
    contenido = bytearray(gzip.decompress(ruta.read_bytes()))
    contenido[-5] ^= 1
    ruta.write_bytes(gzip.compress(bytes(contenido), mtime=0))
    problemas = M.verificar_datos(datos)
    assert len(problemas) == 1 and "pc1" in problemas[0]
    (datos / "wilt.csv.gz").unlink()
    assert any("falta" in p for p in M.verificar_datos(datos))


def test_los_CSV_de_control_son_los_del_Studio_byte_a_byte():
    ejemplos = _FASE0.parents[1].parent / "matrixaistudio" / "studio-backend" / "matrixai_studio" / "ejemplos_medidos"
    if not ejemplos.is_dir():
        pytest.skip("no está el Studio al lado")
    for c in M.leer_manifiesto(KIT / "datos")["conjuntos"]:
        if c["es_control"]:
            assert M.csv_de(KIT / "datos", c)[0] == (ejemplos / f"{c['nombre']}.csv").read_bytes(), c["nombre"]


# ------------------------------------------------------------------------------------------------ un Studio simulado
class StudioSimulado:
    """Un servidor HTTP con los tres endpoints que usa el medidor. La cifra de test de cada estudio sale de `cifras[
    (brazo, nombre)]` (por omisión 80,0; una lista = una por vuelta, para la repetición del control)."""

    def __init__(self, brazo, cifras=None, gpu=False, caer_en=None, campeon="lightgbm", entregado=None):
        self.brazo, self.cifras, self.gpu, self.caer_en = brazo, cifras or {}, gpu, caer_en
        # 128 (I1): el campeón y lo que declara su modelo ENTREGADO, si una prueba quiere otros
        self.campeon, self.entregado = campeon, entregado
        self.iniciados: list[str] = []
        self.trabajos: dict[str, str] = {}
        self.vueltas: dict[str, int] = {}
        sim = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, cuerpo, codigo=200):
                crudo = json.dumps(cuerpo).encode()
                self.send_response(codigo); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(crudo))); self.end_headers(); self.wfile.write(crudo)

            def do_GET(self):
                if self.path == "/api/studio/status":
                    return self._json({"ok": True, "gpu": sim.gpu})
                if self.path.startswith("/api/studio/estudio/estado/"):
                    return self._json(sim.estado(self.path.rsplit("/", 1)[1]))
                self._json({}, 404)

            def do_POST(self):
                largo = int(self.headers.get("Content-Length") or 0)
                cuerpo = json.loads(self.rfile.read(largo) or b"{}")
                if self.path == "/api/studio/estudio/iniciar":
                    nombre = sim.nombre_de(cuerpo)
                    if sim.caer_en == nombre:
                        self.connection.close()
                        return
                    sim.iniciados.append(nombre)
                    sim.vueltas[nombre] = sim.vueltas.get(nombre, 0) + 1
                    job = f"j{len(sim.iniciados)}"
                    sim.trabajos[job] = nombre
                    return self._json({"job_id": job})
                self._json({}, 404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def nombre_de(self, cuerpo):
        manifiesto = M.leer_manifiesto(KIT / "datos")
        cabecera = cuerpo["csv_text"].split("\n", 1)[0]
        n = len(cuerpo["csv_text"])
        for c in manifiesto["conjuntos"]:
            if c["bytes"] == len(cuerpo["csv_text"].encode()) and c["objetivo"] in cabecera:
                return c["nombre"]
        raise AssertionError(f"conjunto desconocido ({n} caracteres)")

    def estado(self, job):
        nombre = self.trabajos[job]
        manifiesto = {c["nombre"]: c for c in M.leer_manifiesto(KIT / "datos")["conjuntos"]}
        c = manifiesto[nombre]
        valor = self.cifras.get((self.brazo, nombre), 80.0)
        if isinstance(valor, list):
            valor = valor[min(self.vueltas[nombre], len(valor)) - 1]
        if valor is None:
            return {"estado": "failed", "seleccion": None}
        metrica = {"binary_classification": "auroc", "multiclass_classification": "accuracy", "regression": "r2"}[c["tarea"]]
        dispositivo = "cuda" if self.gpu else "cpu"
        return {"estado": "completed",
                "seleccion": {"candidate_engine": self.campeon, "media_de_la_seleccion": {"motores": []},
                              "evaluacion_final": {"evaluated_role": "test", "evidence": "independent_test",
                                                   "metrics": [{"metric_id": metrica, "value": valor / 100.0}]},
                              "entrenamiento_del_entregado": (
                                  self.entregado if self.entregado is not None else
                                  {"dispositivo": dispositivo if self.gpu else None, "parado_por_plazo": False})},
                "leaderboard": [{"engine": "matrixai.dense.tabm_cpu", "candidate": "x-p0-r0", "wall_seconds": 5.0,
                                 "estado": "completed", "hiperparametros": {"device": dispositivo}}]}

    def cerrar(self):
        self.httpd.shutdown(); self.httpd.server_close()


def _args(salida, brazo, studio, *extra):
    return ["--brazo", brazo, "--studio", studio.url, "--salida", str(salida), "--datos", str(KIT / "datos"),
            "--imagen", "img:c5", "--imagen-id", "sha256:abc", *extra]


@pytest.fixture()
def estudios():
    abiertos = []

    def crear(*a, **k):
        s = StudioSimulado(*a, **k); abiertos.append(s); return s
    yield crear
    for s in abiertos:
        s.cerrar()


def _brazos(tmp_path, estudios, cifras_b=None, gpu_b=True):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    b = estudios("B", cifras_b, gpu=gpu_b)
    return a, b


# ------------------------------------------------------------------------------------------------ el flujo entero
def test_los_dos_brazos_los_13_y_el_control_repetido_componen_un_json_con_veredicto(tmp_path, estudios):
    a, b = _brazos(tmp_path, estudios, {("B", "micro-mass"): 83.0})
    assert M.main(_args(tmp_path, "B", b)) == 0
    assert len(a.iniciados) == 13 + 4 and len(b.iniciados) == 13 + 4          # los 13 y los 4 de control repetidos
    assert M.main(["--componer", "--salida", str(tmp_path), "--datos", str(KIT / "datos")]) == 0
    r = json.loads((tmp_path / "resultado_128_c5.json").read_text())
    assert set(r["brazos"]) == {"A", "B"} and len(r["brazos"]["B"]["conjuntos"]) == 13
    assert r["brazos"]["A"]["suelo_de_ruido"] == 0.0 and r["brazos"]["B"]["terminado"] is True
    assert r["brazos"]["B"]["procedencia"]["imagen_id"] == "sha256:abc"
    assert r["veredicto"]["entra"] is True and r["veredicto"]["suben"] == 1
    assert r["procedencia"]["sha256_veredicto_128_c5"] == M.sha(V.__file__)


def test_el_resultado_lleva_los_dispositivos_vistos_y_los_picos_de_recursos(tmp_path, estudios):
    muestras = tmp_path / "muestras_B.jsonl"
    a = estudios("A")
    M.main(_args(tmp_path, "A", a))
    b = estudios("B", gpu=True)
    muestras.write_text("\n".join(json.dumps({"t": t, "ram_mb": 100 + t % 7, "vram_mb": 500.0 if t < 10**10 else 0, "gpu_util": 40.0})
                                  for t in (1.0, 2.0)) + "\nlinea rota\n")
    assert M.main(_args(tmp_path, "B", b, "--solo", "dresses-sales", "--sin-comparar")) == 0
    rec = json.loads((tmp_path / "brazo_B.json").read_text())["conjuntos"]["dresses-sales"]
    assert rec["dispositivos_vistos"] == {"cuda": 2} and rec["entrenamiento_del_entregado"]["dispositivo"] == "cuda"
    assert rec["recursos"]["muestras"] == 0          # las muestras de arriba son de antes del estudio: no cuentan dentro


def test_los_picos_se_toman_entre_el_inicio_y_el_fin_del_estudio():
    muestras = [{"t": 1, "ram_mb": 10, "vram_mb": 1000.0, "gpu_util": 0}, {"t": 5, "ram_mb": 300, "vram_mb": 4000.0, "gpu_util": 90},
                {"t": 6, "ram_mb": 200, "vram_mb": 3500.0, "gpu_util": 70}, {"t": 99, "ram_mb": 9999, "vram_mb": 9999.0, "gpu_util": 1}]
    p = M.picos(muestras, 4.0, 7.0)
    assert (p["ram_pico_mb"], p["vram_pico_mb"], p["vram_antes_mb"], p["vram_extra_mb"], p["gpu_uso_pico_pct"]) == (300, 4000.0, 1000.0, 3000.0, 90)
    assert M.picos([], 0, 1)["vram_extra_mb"] is None


# ------------------------------------------------------------------------------------------------ cortar y seguir
def test_relanzar_la_misma_orden_reusa_lo_medido_y_no_vuelve_a_iniciar_estudios(tmp_path, estudios):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    n = len(a.iniciados)
    assert M.main(_args(tmp_path, "A", a)) == 0
    assert len(a.iniciados) == n


def test_un_estudio_que_no_completo_se_REINTENTA_al_relanzar(tmp_path, estudios):
    a = estudios("A", {("A", "pendigits"): None})
    M.main(_args(tmp_path, "A", a, "--solo", "pendigits"))
    assert json.loads((tmp_path / "brazo_A.json").read_text())["conjuntos"]["pendigits"]["estado"] == "failed"
    a.cifras = {}
    M.main(_args(tmp_path, "A", a, "--solo", "pendigits"))
    assert a.iniciados.count("pendigits") == 2
    assert json.loads((tmp_path / "brazo_A.json").read_text())["conjuntos"]["pendigits"]["estado"] == "completed"


def test_otra_imagen_o_otro_ajuste_NO_reusa_el_punto_de_control(tmp_path, estudios):
    a = estudios("A")
    M.main(_args(tmp_path, "A", a, "--solo", "wilt"))
    M.main(_args(tmp_path, "A", a, "--solo", "wilt", "--memoria", "24g"))
    assert a.iniciados.count("wilt") == 2


def test_si_el_Studio_deja_de_contestar_sale_con_4_no_se_marca_terminado_y_al_relanzar_sigue(tmp_path, estudios):
    a = estudios("A", caer_en="pendigits")
    assert M.main(_args(tmp_path, "A", a, "--solo", "wilt,pendigits,house_16H")) == 4
    brazo = json.loads((tmp_path / "brazo_A.json").read_text())
    assert brazo["terminado"] is False and brazo["corte"]["tipo"] == "servidor" and "wilt" in brazo["conjuntos"]
    a.caer_en = None
    assert M.main(_args(tmp_path, "A", a, "--solo", "wilt,pendigits,house_16H")) == 0
    assert a.iniciados.count("wilt") == 1          # lo ya medido se reusó
    assert json.loads((tmp_path / "brazo_A.json").read_text())["terminado"] is True


def test_el_brazo_B_exige_el_A_terminado(tmp_path, estudios):
    assert M.main(_args(tmp_path, "B", estudios("B", gpu=True))) == 3


# ------------------------------------------------------------------------------------------------ el corte
def test_el_brazo_B_se_CORTA_en_el_primer_conjunto_que_baja_mas_que_el_suelo(tmp_path, estudios):
    a, b = _brazos(tmp_path, estudios, {("B", "wilt"): 70.0})
    assert M.main(_args(tmp_path, "B", b)) == 0
    brazo = json.loads((tmp_path / "brazo_B.json").read_text())
    assert brazo["corte"]["tipo"] == "regla_9" and brazo["corte"]["tras"] == "wilt"
    assert "wilt" in brazo["conjuntos"] and "pendigits" not in brazo["conjuntos"] and brazo["corte"]["sin_medir"][0] == "pendigits"
    assert "pendigits" not in b.iniciados
    M.main(["--componer", "--salida", str(tmp_path), "--datos", str(KIT / "datos")])
    assert json.loads((tmp_path / "resultado_128_c5.json").read_text())["veredicto"]["entra"] is False


def test_con_sin_cortar_mide_todo_aunque_baje(tmp_path, estudios):
    a, b = _brazos(tmp_path, estudios, {("B", "wilt"): 70.0})
    M.main(_args(tmp_path, "B", b, "--sin-cortar"))
    assert "Allstate_Claims_Severity" in json.loads((tmp_path / "brazo_B.json").read_text())["conjuntos"]


def test_el_suelo_del_control_repetido_se_usa_para_no_cortar_por_ruido(tmp_path, estudios):
    """El control se repite con otra cifra (suelo 1,5): una bajada de 1,0 en `wilt` es ruido y no corta."""
    a, b = _brazos(tmp_path, estudios, {("B", "pc1"): [80.0, 81.5], ("B", "wilt"): 79.0, ("B", "micro-mass"): 90.0})
    assert M.main(_args(tmp_path, "B", b)) == 0
    brazo = json.loads((tmp_path / "brazo_B.json").read_text())
    assert brazo["suelo_de_ruido"] == pytest.approx(1.5) and brazo["corte"] is None and brazo["terminado"]


# ------------------------------------------------------------------------------------------------ la prueba corta
def test_la_PRUEBA_en_B_sin_GPU_usada_sale_con_5(tmp_path, estudios, capsys):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a, "--prueba")) == 0
    assert a.iniciados == ["dresses-sales"]
    b = estudios("B", gpu=False)
    assert M.main(_args(tmp_path, "B", b, "--prueba")) == 5
    assert "NO SE USO" in capsys.readouterr().out


def test_la_PRUEBA_en_B_con_la_GPU_usada_sale_con_0(tmp_path, estudios, capsys):
    a = estudios("A")
    M.main(_args(tmp_path, "A", a, "--prueba"))
    b = estudios("B", gpu=True)
    assert M.main(_args(tmp_path, "B", b, "--prueba")) == 0
    assert "la GPU SE USO" in capsys.readouterr().out


def test_datos_manipulados_paran_antes_de_medir(tmp_path, estudios, monkeypatch):
    monkeypatch.setattr(M, "verificar_datos", lambda datos: ["pc1: el CSV no es el del manifiesto"])
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 2 and a.iniciados == []


# ------------------------------------------------------------------------------------------------ el empaquetado y los lanzadores
def test_el_empaquetado_trae_las_piezas_del_repo_identicas_y_el_medidor_corre_desde_el(tmp_path):
    emp = _cargar(_FASE0 / "empaquetar_kit_128_c5.py", "empaquetar_en_pruebas")
    destino = tmp_path / "kit"
    emp.armar(destino)
    for nombre in emp.PIEZAS_DEL_REPO:
        assert (destino / nombre).read_bytes() == (_FASE0 / nombre).read_bytes()
    sumas = dict(reversed(l.split("  ", 1)) for l in (destino / "SHA256SUMS").read_text().splitlines())
    assert all(hashlib.sha256((destino / f).read_bytes()).hexdigest() == h for f, h in sumas.items())
    salida = subprocess.run([sys.executable, "-I", str(destino / "medir_c5.py"), "--verificar-datos", "--datos", str(destino / "datos")],
                            capture_output=True, text=True, cwd=tmp_path)
    assert salida.returncode == 0 and "DATOS OK" in salida.stdout, salida.stderr[-1500:]


def test_los_lanzadores_existen_y_el_ps1_es_ASCII_puro():
    for f in ("lanzar.sh", "lanzar.ps1", "lanzar.bat", "LEEME.md", "medir_c5.py", "muestreador.py"):
        assert (KIT / f).is_file(), f
    for f in ("lanzar.ps1", "lanzar.bat"):
        (KIT / f).read_bytes().decode("ascii")                  # PowerShell 5.1 lee un .ps1 sin BOM como ANSI
    assert subprocess.run(["bash", "-n", str(KIT / "lanzar.sh")]).returncode == 0


@pytest.mark.parametrize("fichero", ["lanzar.sh", "lanzar.ps1"])
def test_los_dos_lanzadores_hacen_lo_mismo_B_con_GPU_y_auto_A_sin_GPU_y_cpu(fichero):
    t = (KIT / fichero).read_text()
    assert "MATRIXAI_TABM_DISPOSITIVO" in t and "c5-128-studio-" in t and "--gpus" in t and "all" in t
    assert "auto" in t and "cpu" in t
    assert "MATRIXAI_LICENSE_ENABLED=false" in t and "--memory-swap" in t and "--init" in t
    # SOLO el brazo B lleva la GPU: la reserva de GPU del Studio va dentro de la rama del B
    if fichero == "lanzar.sh":
        assert '[ "$b" = B ] && { gpu=(--gpus all); disp=auto; }' in t
    else:
        assert 'if ($b -eq "B") { $gpu = @("--gpus", "all"); $disp = "auto" } else { $gpu = @(); $disp = "cpu" }' in t
    assert "/work/kit" in t and "kit:ro" in t


def test_el_kit_monta_en_work_kit_porque_el_instrumento_del_120_exige_dos_niveles_por_encima():
    """`referencia_120_r0.py` hace `AQUI.parents[1]` al importarse: montado en `/kit` revienta con IndexError (medido)."""
    assert "parents[1]" in (_FASE0 / "referencia_120_r0.py").read_text()
    assert "/kit/" not in (KIT / "lanzar.sh").read_text().replace("/work/kit/", "")


# ------------------------------------------------------------------------------------------------ 128 integración: la capa GPU se construye en el PC
_DOCKERFILE_GPU = _FASE0.parents[2] / "matrixaistudio" / "bundle-gpu" / "Dockerfile.gpu"


def _docker_falso(tmp_path):
    """Un `docker` que solo anota las órdenes: `image inspect` dice que no existe lo que aún no se cargó/construyó."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    estado = tmp_path / "estado"
    estado.mkdir()
    (bin_ / "docker").write_text(textwrap.dedent('''\
        #!/usr/bin/env bash
        echo "$@" >> "$LOG_DOCKER"
        nombre() { echo "$1" | tr '/:' '__'; }
        if [ "$1" = image ] && [ "$2" = inspect ]; then
          [ -f "$ESTADO_DOCKER/$(nombre "$3")" ] || exit 1
          [ "$4" = "--format" ] && echo sha256:abc
          exit 0
        fi
        case "$1" in
          load) touch "$ESTADO_DOCKER/matrixai-studio_128"; echo "Loaded image: matrixai-studio:128";;
          build) cp "$4/Dockerfile" "$ESTADO_DOCKER/Dockerfile.visto"; touch "$ESTADO_DOCKER/$(nombre "$3")";;
        esac
        exit 0
        '''))
    (bin_ / "docker").chmod(0o755)
    return bin_, estado, tmp_path / "docker.log"


def _kit_empaquetado(tmp_path, con_tar=True):
    emp = _cargar(_FASE0 / "empaquetar_kit_128_c5.py", "empaquetar_para_la_capa")
    tar = tmp_path / "base.tar"
    tar.write_bytes(b"tar-de-mentira")
    kit = tmp_path / "kit"
    emp.armar(kit, tar_base=tar if con_tar else None)
    return kit


def _lanzar(kit, tmp_path, *args, antes=None):
    import os
    bin_, estado, log = _docker_falso(tmp_path) if not (tmp_path / "bin").exists() else (
        tmp_path / "bin", tmp_path / "estado", tmp_path / "docker.log")
    if antes:
        antes(estado)
    entorno = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "LOG_DOCKER": str(log), "ESTADO_DOCKER": str(estado)}
    r = subprocess.run(["bash", str(kit / "lanzar.sh"), "--prueba", "--brazos", "B", "--salida", str(tmp_path / "salida"), *args],
                       capture_output=True, text=True, env=entorno, timeout=120)
    return r, (log.read_text().splitlines() if log.exists() else []), estado


def test_el_kit_CONSTRUYE_la_capa_GPU_en_el_PC_cargando_el_tar_y_con_el_Dockerfile_de_la_integracion(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    assert (kit / "imagen" / "Dockerfile.gpu").read_bytes() == _DOCKERFILE_GPU.read_bytes()      # copiado tal cual
    assert (kit / "imagen" / "matrixai-studio-128.tar").read_bytes() == b"tar-de-mentira"
    r, ordenes, estado = _lanzar(kit, tmp_path)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    carga = [i for i, o in enumerate(ordenes) if o.startswith("load -i ") and o.endswith("matrixai-studio-128.tar")]
    construye = [i for i, o in enumerate(ordenes) if o.startswith("build -t matrixai-studio-gpu-128:local ")]
    assert len(carga) == 1 and len(construye) == 1 and carga[0] < construye[0]
    visto = (estado / "Dockerfile.visto").read_text()
    assert "FROM matrixai-studio:128" in visto and "FROM matrixai-studio:v2" not in visto
    assert "torch==2.6.0+cu124" in visto                                    # la capa del paquete GPU, no otra
    # y los dos brazos corren CONTRA ESA IMAGEN, no contra una que haya que pasar
    corridas = [o for o in ordenes if o.startswith("run ")]
    assert corridas and all("matrixai-studio-gpu-128:local" in o for o in corridas if "c5-128-red" not in o or "--entrypoint python3" in o)
    assert any("--gpus all" in o and "matrixai-studio-gpu-128:local" in o for o in corridas)


def test_la_segunda_vez_no_vuelve_a_cargar_ni_a_construir(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _lanzar(kit, tmp_path)
    r, ordenes, _ = _lanzar(kit, tmp_path)               # el estado del docker falso persiste en tmp_path
    # `_lanzar` reescribe el docker falso pero conserva `estado`; el registro se acumula
    assert r.returncode == 0
    assert sum(o.startswith("load -i ") for o in ordenes) == 1 and sum(o.startswith("build -t ") for o in ordenes) == 1


def test_sin_el_tar_el_kit_PARA_y_lo_dice_en_vez_de_medir_con_otra_imagen(tmp_path):
    kit = _kit_empaquetado(tmp_path, con_tar=False)
    r, ordenes, _ = _lanzar(kit, tmp_path)
    assert r.returncode == 2 and "matrixai-studio-128.tar" in r.stdout
    assert not any(o.startswith("build ") for o in ordenes)


def test_con_imagen_explicita_no_carga_ni_construye(tmp_path):
    kit = _kit_empaquetado(tmp_path, con_tar=False)
    r, ordenes, _ = _lanzar(kit, tmp_path, "--imagen", "otra_gpu", antes=lambda estado: (estado / "otra_gpu").touch())
    assert r.returncode == 0 and not any(o.startswith(("load ", "build ")) for o in ordenes)


@pytest.mark.parametrize("fichero", ["lanzar.sh", "lanzar.ps1"])
def test_los_dos_lanzadores_cargan_el_tar_y_construyen_la_capa_sin_pedir_imagen(fichero):
    t = (KIT / fichero).read_text()
    assert "matrixai-studio-128.tar" in t and "Dockerfile.gpu" in t and "matrixai-studio:128" in t
    assert "matrixai-studio-gpu-128:local" in t and "docker build" in t and "docker load" in t
    assert "build_capa.log" in t


def test_el_leeme_ya_no_pide_imagen_y_cuenta_la_capa():
    t = (KIT / "LEEME.md").read_text()
    assert "<imagen>" not in t.replace("`-Imagen <imagen>` (`--imagen`)", "")
    assert "matrixai-studio-128.tar" in t and "3 GB" in t and "no pisa tu instalación normal" in t


# ------------------------------------------------------------------------------------------------ 128 (reparación, B2): lanzar.ps1 EJECUTADO
# La auditoría del 128 midió `lanzar.ps1` con PowerShell 7.4 en Docker y un `docker` falso: no compilaba (`(a; b)`), y
# arreglado eso, `$rc = Medidor $b` se tragaba la salida del medidor y el kit paraba tras el brazo A; y en PowerShell 5.1
# el JSON de `C5_ENTORNO` perdía las comillas. Aquí se EJECUTA de principio a fin, los dos brazos, con pwsh (en el
# host si lo hay; si no, en el contenedor `mcr.microsoft.com/powershell` con techo, si la imagen ya está en este
# Docker: la prueba no descarga nada) y con el paso de argumentos «Legacy» (el de Windows PowerShell 5.1). Un Windows
# de verdad NO se prueba aquí.
_IMAGEN_PWSH = "mcr.microsoft.com/powershell:lts-ubuntu-22.04"

_DOCKER_FALSO_PS = textwrap.dedent('''\
    #!/usr/bin/env bash
    # docker FALSO para lanzar.ps1: anota cada orden (un argumento por línea) y el C5_ENTORNO que ve, y simula lo justo.
    { echo "=== $1"; for a in "$@"; do printf 'ARG %s\\n' "$a"; done; printf 'ENV_C5 %s\\n' "${C5_ENTORNO-<sin>}"; } >> "$LOG_DOCKER"
    # lo que vería el contenedor en C5_ENTORNO, como lo resuelve docker: «-e C5_ENTORNO=valor» o «-e C5_ENTORNO» (del entorno)
    prev=""; for a in "$@"; do
      if [ "$prev" = "-e" ]; then
        case "$a" in C5_ENTORNO) printf 'EFECTIVO_C5 %s\\n' "${C5_ENTORNO-<sin>}" >> "$LOG_DOCKER";;
                     C5_ENTORNO=*) printf 'EFECTIVO_C5 %s\\n' "${a#C5_ENTORNO=}" >> "$LOG_DOCKER";; esac
      fi; prev="$a"; done
    case "$1" in
      image) for a in "$@"; do [ "$a" = "{{.Id}}" ] && echo "sha256:falso"; done; exit 0;;
      version|info) echo "docker falso $1"; exit 0;;
      run)
        medidor=0; brazo=""; prev=""
        for a in "$@"; do
          case "$a" in
            --verificar-datos) echo "DATOS OK"; exit 0;;
            --componer) echo "COMPUESTO (falso)"; echo "componer" >> "$FASES"; exit 0;;
            /work/kit/medir_c5.py) medidor=1;;
          esac
          [ "$prev" = "--brazo" ] && brazo="$a"
          prev="$a"
        done
        if [ "$medidor" = 1 ]; then
          echo "medidor $brazo" >> "$FASES"
          echo "[12:00:00] brazo $brazo · dresses-sales ..."
          echo "   completed 115 s campeón=lightgbm VRAM extra=None MB"
          rc="RC_$brazo"; exit "${!rc:-0}"
        fi
        for a in "$@"; do [ "$a" = "-d" ] && { echo "idfalso"; exit 0; }; done
        echo "torch 2.6.0 cuda 12.4 GPU: falsa"; exit 0;;
    esac
    exit 0
    ''')


def _como_correr_pwsh():
    """`("host", None)`, `("docker", None)` o `(None, motivo)`."""
    if shutil.which("pwsh"):
        return "host", None
    if not shutil.which("docker"):
        return None, "ni pwsh ni docker en esta máquina"
    if subprocess.run(["docker", "image", "inspect", _IMAGEN_PWSH], capture_output=True).returncode != 0:
        return None, f"sin pwsh y sin la imagen {_IMAGEN_PWSH} en este Docker (no se descarga en una prueba)"
    return "docker", None


def _pwsh(tmp_path, orden: str, entorno: dict[str, str] | None = None, timeout=240):
    """Corre `orden` en pwsh con el kit en /kit (solo lectura), el docker falso primero en el PATH y /out escribible."""
    import os
    import uuid
    modo, motivo = _como_correr_pwsh()
    if modo is None:
        pytest.skip(motivo)
    falso = tmp_path / "fakebin"
    falso.mkdir(exist_ok=True)
    (falso / "docker").write_text(_DOCKER_FALSO_PS)
    (falso / "docker").chmod(0o755)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    out.chmod(0o777)
    entorno = {"LOG_DOCKER": "/out/docker.log", "FASES": "/out/fases.log", **(entorno or {})}
    if modo == "host":
        rutas = {"/kit": str(KIT), "/out": str(out), "/fakebin": str(falso)}
        for viejo, nuevo in rutas.items():
            orden = orden.replace(viejo, nuevo)
            entorno = {k: v.replace(viejo, nuevo) for k, v in entorno.items()}
        r = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", orden],
                           env={**os.environ, **entorno}, capture_output=True, text=True, timeout=timeout)
    else:
        nombre = f"rep128-pwsh-{uuid.uuid4().hex[:10]}"
        args = ["docker", "run", "--rm", "--name", nombre, "--network", "none", "--memory=1g", "--memory-swap=1g",
                "--cpus=1", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                "-v", f"{KIT}:/kit:ro", "-v", f"{falso}:/fakebin:ro", "-v", f"{out}:/out"]
        for k, v in entorno.items():
            args += ["-e", f"{k}={v}"]
        try:
            r = subprocess.run([*args, _IMAGEN_PWSH, "pwsh", "-NoProfile", "-NonInteractive", "-Command", orden],
                               capture_output=True, text=True, timeout=timeout)
        finally:
            subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)
    log = (out / "docker.log").read_text() if (out / "docker.log").exists() else ""
    fases = (out / "fases.log").read_text().split() if (out / "fases.log").exists() else []
    return r, log, fases


def _ordenes(log: str) -> list[dict]:
    """El registro del docker falso, por orden: sus argumentos y el C5_ENTORNO que vio."""
    ordenes = []
    for bloque in log.split("=== ")[1:]:
        lineas = bloque.splitlines()
        ordenes.append({"args": [l[4:] for l in lineas if l.startswith("ARG ")],
                        "c5": next((l[7:] for l in lineas if l.startswith("ENV_C5 ")), None),
                        "efectivo": next((l[12:] for l in lineas if l.startswith("EFECTIVO_C5 ")), None)})
    return ordenes


_LANZAR = ("$PSNativeCommandArgumentPassing = '{modo}'; $env:PATH = '/fakebin:' + $env:PATH; "
           "& /kit/lanzar.ps1 -Brazos AB -Salida /out/salida; exit $LASTEXITCODE")


def test_lanzar_ps1_COMPILA_en_PowerShell(tmp_path):
    """El guion entero se analiza antes de ejecutarse: un error de sintaxis (la `(a; b)` de la línea 89) lo mataba
    antes de hacer nada."""
    r, _, _ = _pwsh(tmp_path, "$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
                              "'/kit/lanzar.ps1', [ref]$null, [ref]$e); "
                              "$e | ForEach-Object { 'ERROR ' + $_.Extent.StartLineNumber + ': ' + $_.Message }; "
                              "'ERRORES=' + $e.Count")
    assert r.returncode == 0, r.stderr[-1500:]
    assert "ERRORES=0" in r.stdout, r.stdout[-2000:]


@pytest.mark.parametrize("modo", ["Standard", "Legacy"])
def test_lanzar_ps1_de_principio_a_fin_corre_los_DOS_brazos_ENSENA_el_medidor_y_compone(tmp_path, modo):
    """`Legacy` es como Windows PowerShell 5.1 pasa los argumentos a un programa externo (sin escapar comillas)."""
    r, log, fases = _pwsh(tmp_path, _LANZAR.format(modo=modo))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    assert fases == ["medidor", "A", "medidor", "B", "componer"], fases          # el B arranca tras el A, y se compone
    # lo que escribe el medidor llega a la PANTALLA (antes se lo tragaba `$rc = Medidor $b`)
    assert "brazo A · dresses-sales" in r.stdout and "brazo B · dresses-sales" in r.stdout, r.stdout[-2000:]
    medidores = [o for o in _ordenes(log) if "/work/kit/medir_c5.py" in o["args"] and "--brazo" in o["args"]]
    assert [o["args"][o["args"].index("--brazo") + 1] for o in medidores] == ["A", "B"]
    for o in medidores:
        brazo = o["args"][o["args"].index("--brazo") + 1]
        # lo que vería el medidor en su contenedor se lee como JSON (con «Legacy» —5.1— y el JSON en la línea de
        # órdenes, llegaba `{MATRIXAI_TABM_DISPOSITIVO:cpu}`: sin comillas, y el medidor no lo podía leer)
        assert json.loads(o["efectivo"]) == ({"MATRIXAI_TABM_DISPOSITIVO": "cpu"} if brazo == "A" else
                                             {"MATRIXAI_TABM_DISPOSITIVO": "auto", "gpus": "all"}), o["efectivo"]
        # y por eso el JSON NO va en la línea de órdenes: `-e C5_ENTORNO` y el valor, del entorno del lanzador
        assert "C5_ENTORNO" in o["args"] and not any(a.startswith("C5_ENTORNO=") for a in o["args"]), o["args"]
        assert o["args"][o["args"].index("--entorno-de-env") + 1] == "C5_ENTORNO"
    # el Studio del B con la GPU y `auto`; el del A sin GPU y `cpu`
    studios = [o["args"] for o in _ordenes(log) if o["args"][:2] == ["run", "-d"]]
    assert any("--gpus" in a and "MATRIXAI_TABM_DISPOSITIVO=auto" in a for a in studios)
    assert any("--gpus" not in a and "MATRIXAI_TABM_DISPOSITIVO=cpu" in a for a in studios)


def test_lanzar_ps1_si_el_brazo_A_falla_PARA_con_su_codigo_y_no_arranca_el_B(tmp_path):
    r, _, fases = _pwsh(tmp_path, _LANZAR.format(modo="Standard"), {"RC_A": "3"})
    assert r.returncode == 3, r.stdout[-1500:]
    assert fases == ["medidor", "A"], fases


def test_lanzar_ps1_si_el_Studio_del_brazo_deja_de_contestar_lo_rearranca_hasta_tres_veces(tmp_path):
    r, _, fases = _pwsh(tmp_path, _LANZAR.format(modo="Standard"), {"RC_A": "4"})
    assert r.returncode == 4, r.stdout[-1500:]
    assert fases == ["medidor", "A"] * 3, fases


def test_lanzar_bat_ya_no_pide_imagen_y_devuelve_el_codigo_del_ps1():
    t = (KIT / "lanzar.bat").read_text()
    assert "-Imagen <imagen>" not in t
    assert "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0lanzar.ps1\" %*" in t
    assert t.index("set RC=%ERRORLEVEL%") < t.index("pause") < t.index("exit /b %RC%")


# ------------------------------------------------------------------------------------------------ 128 (reparación, I1): el ENTREGADO
def test_I1_si_TabM_gana_en_B_y_su_ENTREGADO_se_entreno_en_CPU_sin_decirlo_el_brazo_B_PARA_en_ese_conjunto(tmp_path, estudios):
    """El B1 visto desde el kit: los intentos de la búsqueda en la GPU (`dispositivos_vistos` con `cuda`) y el modelo
    ENTREGADO en la CPU sin caída declarada. Antes la «presencia» lo daba por bueno (cualquier `cuda` del estado) y
    Roberto habría medido ~4 h de otra cosa; ahora para en el PRIMER conjunto donde pasa, con su motivo."""
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    b = estudios("B", gpu=True, campeon=V.MOTOR_TABM,
                 entregado={"dispositivo": "cpu", "cayo_a_cpu": False, "parado_por_plazo": False})
    assert M.main(_args(tmp_path, "B", b)) == 6
    brazo = json.loads((tmp_path / "brazo_B.json").read_text())
    assert brazo["corte"]["tipo"] == "instrumento" and "ENTREGADO" in brazo["corte"]["motivo"], brazo["corte"]
    assert brazo["terminado"] is False and len(b.iniciados) == 1                   # paró en el primero, no tras horas
    assert V.presencia(json.loads((tmp_path / "brazo_A.json").read_text())["conjuntos"], brazo["conjuntos"])


@pytest.mark.parametrize("entregado", [{"dispositivo": "cuda", "cayo_a_cpu": False, "parado_por_plazo": False},
                                       {"dispositivo": "cpu", "cayo_a_cpu": True, "parado_por_plazo": False}])
def test_I1_con_el_entregado_en_la_GPU_o_su_caida_DECLARADA_el_brazo_B_sigue(tmp_path, estudios, entregado):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    b = estudios("B", gpu=True, campeon=V.MOTOR_TABM, entregado=entregado)
    assert M.main(_args(tmp_path, "B", b)) == 0
    assert json.loads((tmp_path / "brazo_B.json").read_text())["terminado"] is True


def test_I1_si_el_brazo_A_entrena_TabM_en_la_GPU_tambien_PARA(tmp_path, estudios):
    a = estudios("A", gpu=True)
    assert M.main(_args(tmp_path, "A", a)) == 6
    assert "brazo A (CPU) entrenó TabM en la GPU" in json.loads((tmp_path / "brazo_A.json").read_text())["corte"]["motivo"]


# ------------------------------------------------------------------------------------------------ 128 (reparación, I2): relanzar no borra
def test_I2_relanzar_tras_un_corte_por_la_regla_9_NO_tira_la_pasada_que_bajo(tmp_path, estudios):
    """Antes, la misma orden tras el corte sobrescribía `brazo_B.json` con los conjuntos vacíos y medía otra vez: se
    perdía la pasada que bajó y se podía repetir hasta que entrase."""
    _a, b = _brazos(tmp_path, estudios, {("B", "wilt"): 70.0})
    assert M.main(_args(tmp_path, "B", b)) == 0
    antes, n = (tmp_path / "brazo_B.json").read_bytes(), len(b.iniciados)
    assert json.loads(antes)["corte"]["tipo"] == "regla_9"
    b.cifras = {}                                     # «esta vez entraría»: relanzar no puede repetir hasta que entre
    assert M.main(_args(tmp_path, "B", b)) == 0
    assert len(b.iniciados) == n and (tmp_path / "brazo_B.json").read_bytes() == antes
    assert M.main(["--componer", "--salida", str(tmp_path), "--datos", str(KIT / "datos")]) == 0
    assert json.loads((tmp_path / "resultado_128_c5.json").read_text())["veredicto"]["entra"] is False


def test_I2_con_forzar_se_mide_otra_vez_y_la_pasada_anterior_se_GUARDA_con_su_hora(tmp_path, estudios):
    _a, b = _brazos(tmp_path, estudios, {("B", "wilt"): 70.0})
    M.main(_args(tmp_path, "B", b))
    antes = (tmp_path / "brazo_B.json").read_bytes()
    b.cifras = {}
    assert M.main(_args(tmp_path, "B", b, "--forzar")) == 0
    apartadas = sorted(tmp_path.glob("brazo_B.anterior-*.json"))
    assert len(apartadas) == 1 and apartadas[0].read_bytes() == antes             # la que bajó, entera
    assert json.loads((tmp_path / "brazo_B.json").read_text())["corte"] is None


def test_I2_un_punto_de_control_que_no_vale_tampoco_se_pisa(tmp_path, estudios):
    a = estudios("A")
    M.main(_args(tmp_path, "A", a, "--solo", "wilt"))
    antes = (tmp_path / "brazo_A.json").read_bytes()
    M.main(_args(tmp_path, "A", a, "--solo", "wilt", "--memoria", "24g"))           # otra procedencia: no se continúa
    M.main(_args(tmp_path, "A", a, "--solo", "wilt", "--memoria", "12g"))           # y otra, en el mismo segundo
    apartadas = sorted(tmp_path.glob("brazo_A.anterior-*.json"))
    assert len(apartadas) == 2 and antes in [x.read_bytes() for x in apartadas]


# ------------------------------------------------------------------------------------------------ 128 (reparación): el suelo del corte
def test_el_brazo_B_corta_con_el_MAYOR_de_los_dos_suelos_y_no_solo_con_el_suyo(tmp_path, estudios):
    """«El suelo es el MAYOR de los dos brazos» (el veredicto; y el medidor, al cortar el B): el A repite su control
    con 1,5 puntos de diferencia y el B idéntico (suelo 0); una bajada de 1,0 en `wilt` es ruido del A y NO corta.
    Sabotaje VERDE de la auditoría (`medidor_suelo_solo_de_b`): cortar con el suelo del B solo."""
    a = estudios("A", {("A", "pc1"): [80.0, 81.5]})
    assert M.main(_args(tmp_path, "A", a)) == 0
    assert json.loads((tmp_path / "brazo_A.json").read_text())["suelo_de_ruido"] == pytest.approx(1.5)
    b = estudios("B", {("B", "wilt"): 79.0, ("B", "micro-mass"): 90.0}, gpu=True)
    assert M.main(_args(tmp_path, "B", b)) == 0
    brazo = json.loads((tmp_path / "brazo_B.json").read_text())
    assert brazo["suelo_de_ruido"] == 0.0 and brazo["suelo_usado"] == pytest.approx(1.5)
    assert brazo["corte"] is None and brazo["terminado"] is True
    assert brazo["comparaciones"]["wilt"]["clase"] == "igual" and "Allstate_Claims_Severity" in brazo["conjuntos"]
