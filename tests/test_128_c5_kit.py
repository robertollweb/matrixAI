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
def test_los_dos_lanzadores_hacen_lo_mismo_los_DOS_brazos_con_GPU_A_cpu_y_B_auto(fichero):
    """128, I2 de la auditoría final: antes SOLO el brazo B llevaba `--gpus all`, y la comparación mezclaba con TabM en
    la GPU lo que cuesta que la densa vea CUDA. Ahora los dos ven la GPU y solo cambia la señal de TabM."""
    t = (KIT / fichero).read_text()
    assert "MATRIXAI_TABM_DISPOSITIVO" in t and "c5-128-studio-" in t and "--gpus" in t and "all" in t
    assert "auto" in t and "cpu" in t
    assert "MATRIXAI_LICENSE_ENABLED=false" in t and "--memory-swap" in t and "--init" in t
    if fichero == "lanzar.sh":
        assert 'local gpu=(--gpus all); local disp=cpu' in t and '[ "$b" = B ] && disp=auto' in t
        assert 'ent=\'{"MATRIXAI_TABM_DISPOSITIVO":"cpu","gpus":"all"}\'' in t
    else:
        assert '$gpu = @("--gpus", "all")' in t and 'if ($b -eq "B") { $disp = "auto" } else { $disp = "cpu" }' in t
        assert "$gpu = @()" not in t
        assert """$ent = '{"MATRIXAI_TABM_DISPOSITIVO":"cpu","gpus":"all"}'""" in t
    assert "/work/kit" in t and "kit:ro" in t


def test_el_kit_monta_en_work_kit_porque_el_instrumento_del_120_exige_dos_niveles_por_encima():
    """`referencia_120_r0.py` hace `AQUI.parents[1]` al importarse: montado en `/kit` revienta con IndexError (medido)."""
    assert "parents[1]" in (_FASE0 / "referencia_120_r0.py").read_text()
    assert "/kit/" not in (KIT / "lanzar.sh").read_text().replace("/work/kit/", "")


# ------------------------------------------------------------------------------------------------ 128 integración: la capa GPU se construye en el PC
_DOCKERFILE_GPU = _FASE0.parents[2] / "matrixaistudio" / "bundle-gpu" / "Dockerfile.gpu"


def _docker_falso(tmp_path):
    """Un `docker` que solo anota las órdenes: `image inspect` dice que no existe lo que aún no se cargó/construyó.

    128 (I3 de la auditoría final): cada imagen guarda su ID (`load` le pone el de `ID_DEL_TAR`, el del tar que se
    cargaría de verdad) y la capa, la base sobre la que se construyó (su `--label matrixai.kit.base=…`), que es lo que el
    lanzador compara ahora. Una imagen «hecha» a mano (un fichero vacío) tiene el ID `sha256:abc` y ninguna etiqueta."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    estado = tmp_path / "estado"
    estado.mkdir()
    (bin_ / "docker").write_text(textwrap.dedent('''\
        #!/usr/bin/env bash
        echo "$@" >> "$LOG_DOCKER"
        nombre() { echo "$1" | tr '/:' '__'; }
        if [ "$1" = image ] && [ "$2" = inspect ]; then
          f="$ESTADO_DOCKER/$(nombre "$3")"
          [ -f "$f" ] || exit 1
          if [ "$4" = "--format" ]; then
            case "$5" in
              *Labels*) if [ -f "$f.base" ]; then printf '{"matrixai.kit.base":"%s"' "$(cat "$f.base")"
                          [ -f "$f.dockerfile" ] && printf ',"matrixai.kit.dockerfile":"%s"' "$(cat "$f.dockerfile")"
                          printf '}\\n'; else echo null; fi;;
              *) id="$(cat "$f")"; echo "${id:-sha256:abc}";;
            esac
          fi
          exit 0
        fi
        case "$1" in
          load) printf '%s' "${ID_DEL_TAR:-sha256:abc}" > "$ESTADO_DOCKER/matrixai-studio_128"; echo "Loaded image: matrixai-studio:128";;
          build)
            etiqueta=""; base=""; dsha=""; dir=""; prev=""
            for a in "$@"; do
              [ "$prev" = "-t" ] && etiqueta="$a"
              if [ "$prev" = "--label" ]; then
                case "$a" in matrixai.kit.base=*) base="${a#matrixai.kit.base=}";; matrixai.kit.dockerfile=*) dsha="${a#matrixai.kit.dockerfile=}";; esac
              fi
              prev="$a"; dir="$a"
            done
            cp "$dir/Dockerfile" "$ESTADO_DOCKER/Dockerfile.visto"
            printf 'sha256:capa-sobre-%s' "$base" > "$ESTADO_DOCKER/$(nombre "$etiqueta")"
            [ -n "$base" ] && printf '%s' "$base" > "$ESTADO_DOCKER/$(nombre "$etiqueta").base"
            [ -n "$dsha" ] && printf '%s' "$dsha" > "$ESTADO_DOCKER/$(nombre "$etiqueta").dockerfile";;
          rmi) for a in "${@:2}"; do [ "$a" = -f ] || rm -f "$ESTADO_DOCKER/$(nombre "$a")" "$ESTADO_DOCKER/$(nombre "$a").base" "$ESTADO_DOCKER/$(nombre "$a").dockerfile"; done;;
        esac
        exit 0
        '''))
    (bin_ / "docker").chmod(0o755)
    return bin_, estado, tmp_path / "docker.log"


def _tar_oci_falso(ruta: Path, etiqueta="matrixai-studio:128", semilla="base-128") -> tuple[str, str]:
    """Un `docker save` mínimo y COHERENTE (formato OCI, como el de este servidor): `index.json` → un índice interno →
    un manifiesto → la configuración, cada blob con el sha256 de su contenido, y `manifest.json` con la etiqueta.
    Devuelve (el digest del índice: el `.Id` con containerd; el de la configuración: el `.Id` con el almacén clásico)."""
    import io
    import tarfile
    def blob(d):
        crudo = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(crudo).hexdigest(), crudo
    hc, config = blob({"semilla": semilla, "config": {"Labels": None}})
    hm, manifiesto = blob({"schemaVersion": 2, "config": {"digest": f"sha256:{hc}"}, "layers": []})
    hi, indice = blob({"schemaVersion": 2, "manifests": [{"digest": f"sha256:{hm}"}]})
    ficheros = {"oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
                "index.json": json.dumps({"schemaVersion": 2, "manifests": [{
                    "mediaType": "application/vnd.oci.image.index.v1+json", "digest": f"sha256:{hi}",
                    "annotations": {"io.containerd.image.name": f"docker.io/library/{etiqueta}",
                                    "org.opencontainers.image.ref.name": etiqueta.split(":", 1)[1]}}]}).encode(),
                "manifest.json": json.dumps([{"Config": f"blobs/sha256/{hc}", "RepoTags": [etiqueta],
                                              "Layers": []}]).encode(),
                f"blobs/sha256/{hc}": config, f"blobs/sha256/{hm}": manifiesto, f"blobs/sha256/{hi}": indice}
    with tarfile.open(ruta, "w") as t:
        for nombre, crudo in ficheros.items():
            info = tarfile.TarInfo(nombre)
            info.size = len(crudo)
            t.addfile(info, io.BytesIO(crudo))
    return f"sha256:{hi}", f"sha256:{hc}"


def _kit_empaquetado(tmp_path, con_tar=True):
    emp = _cargar(_FASE0 / "empaquetar_kit_128_c5.py", "empaquetar_para_la_capa")
    tar = tmp_path / "base.tar"
    _tar_oci_falso(tar)
    kit = tmp_path / "kit"
    emp.armar(kit, tar_base=tar if con_tar else None)
    return kit


def _ids_del_kit(kit):
    ruta = kit / "imagen" / "matrixai-studio-128.ids"
    return ruta.read_text().split() if ruta.exists() else []


def _lanzar(kit, tmp_path, *args, antes=None, id_del_tar=None):
    import os
    bin_, estado, log = _docker_falso(tmp_path) if not (tmp_path / "bin").exists() else (
        tmp_path / "bin", tmp_path / "estado", tmp_path / "docker.log")
    if antes:
        antes(estado)
    # lo que daría `docker load` con el tar del kit: su ID (el de containerd, el primero que apunta el empaquetador)
    id_del_tar = id_del_tar or (_ids_del_kit(kit) or ["sha256:abc"])[0]
    entorno = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "LOG_DOCKER": str(log), "ESTADO_DOCKER": str(estado),
               "ID_DEL_TAR": id_del_tar}
    r = subprocess.run(["bash", str(kit / "lanzar.sh"), "--prueba", "--brazos", "B", "--salida", str(tmp_path / "salida"), *args],
                       capture_output=True, text=True, env=entorno, timeout=120)
    return r, (log.read_text().splitlines() if log.exists() else []), estado


def test_el_kit_CONSTRUYE_la_capa_GPU_en_el_PC_cargando_el_tar_y_con_el_Dockerfile_de_la_integracion(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    assert (kit / "imagen" / "Dockerfile.gpu").read_bytes() == _DOCKERFILE_GPU.read_bytes()      # copiado tal cual
    assert (kit / "imagen" / "matrixai-studio-128.tar").read_bytes() == (tmp_path / "base.tar").read_bytes()
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
    # 128 (I3): la base y la capa con IDENTIDAD. La base de este Docker da `FALSO_ID_BASE` (tras un `load`,
    # `FALSO_ID_TRAS_CARGAR`); la capa dice sobre qué base se construyó (`FALSO_BASE_DE_LA_CAPA`; tras un `build`, su
    # `--label`). Otra imagen da `sha256:falso`. El estado vive en `ESTADO_FALSO` (/out).
    case "$1" in
      image)
        img="$3"; fmt=""; [ "$4" = "--format" ] && fmt="$5"
        case "$fmt" in
          *Labels*) b="${FALSO_BASE_DE_LA_CAPA-}"; d="${FALSO_DOCKERFILE_DE_LA_CAPA-}"
                    [ -f "$ESTADO_FALSO/capa_construida" ] && b="$(cat "$ESTADO_FALSO/capa_construida")"
                    [ -f "$ESTADO_FALSO/capa_construida_dockerfile" ] && d="$(cat "$ESTADO_FALSO/capa_construida_dockerfile")"
                    if [ -n "$b" ]; then printf '{"matrixai.kit.base":"%s"' "$b"
                      [ -n "$d" ] && printf ',"matrixai.kit.dockerfile":"%s"' "$d"
                      printf '}\\n'; else echo null; fi;;
          *.Id*) if [ "$img" = "matrixai-studio:128" ]; then
                   id="${FALSO_ID_BASE-}"; [ -f "$ESTADO_FALSO/base_cargada" ] && id="${FALSO_ID_TRAS_CARGAR-$FALSO_ID_BASE}"
                   echo "$id"
                 else echo "sha256:falso"; fi;;
        esac
        exit 0;;
      load) touch "$ESTADO_FALSO/base_cargada"; echo "Loaded image: matrixai-studio:128"; exit 0;;
      build) for a in "$@"; do case "$a" in
               matrixai.kit.base=*) printf '%s' "${a#matrixai.kit.base=}" > "$ESTADO_FALSO/capa_construida";;
               matrixai.kit.dockerfile=*) printf '%s' "${a#matrixai.kit.dockerfile=}" > "$ESTADO_FALSO/capa_construida_dockerfile";;
             esac; done
             echo "capa construida (falso)"; exit 0;;
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
    """Corre `orden` en pwsh con el kit EMPAQUETADO en /kit (solo lectura), el docker falso primero en el PATH y /out
    escribible. 128 (I3 de la auditoría final): el kit empaquetado, no la carpeta del repo, porque el lanzador exige la
    identidad de su imagen (`imagen/matrixai-studio-128.ids`, la apunta el empaquetador). Por omisión la base y la capa
    de este «Docker» son las del kit (no se carga ni se construye nada); `entorno` lo cambia."""
    import os
    import uuid
    modo, motivo = _como_correr_pwsh()
    if modo is None:
        pytest.skip(motivo)
    kit = _kit_empaquetado(tmp_path) if not (tmp_path / "kit").exists() else tmp_path / "kit"
    id_del_kit = _ids_del_kit(kit)[0]
    entorno = {"FALSO_ID_BASE": id_del_kit, "FALSO_BASE_DE_LA_CAPA": id_del_kit, "ESTADO_FALSO": "/out",
               "FALSO_DOCKERFILE_DE_LA_CAPA": hashlib.sha256((kit / "imagen" / "Dockerfile.gpu").read_bytes()).hexdigest(),
               **(entorno or {})}
    falso = tmp_path / "fakebin"
    falso.mkdir(exist_ok=True)
    (falso / "docker").write_text(_DOCKER_FALSO_PS)
    (falso / "docker").chmod(0o755)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    out.chmod(0o777)
    entorno = {"LOG_DOCKER": "/out/docker.log", "FASES": "/out/fases.log", **(entorno or {})}
    if modo == "host":
        rutas = {"/kit": str(kit), "/out": str(out), "/fakebin": str(falso)}
        for viejo, nuevo in rutas.items():
            orden = orden.replace(viejo, nuevo)
            entorno = {k: v.replace(viejo, nuevo) for k, v in entorno.items()}
        r = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", orden],
                           env={**os.environ, **entorno}, capture_output=True, text=True, timeout=timeout)
    else:
        nombre = f"rep128-pwsh-{uuid.uuid4().hex[:10]}"
        args = ["docker", "run", "--rm", "--name", nombre, "--network", "none", "--memory=1g", "--memory-swap=1g",
                "--cpus=1", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                "-v", f"{kit}:/kit:ro", "-v", f"{falso}:/fakebin:ro", "-v", f"{out}:/out"]
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
        assert json.loads(o["efectivo"]) == ({"MATRIXAI_TABM_DISPOSITIVO": "cpu", "gpus": "all"} if brazo == "A" else
                                             {"MATRIXAI_TABM_DISPOSITIVO": "auto", "gpus": "all"}), o["efectivo"]
        # y por eso el JSON NO va en la línea de órdenes: `-e C5_ENTORNO` y el valor, del entorno del lanzador
        assert "C5_ENTORNO" in o["args"] and not any(a.startswith("C5_ENTORNO=") for a in o["args"]), o["args"]
        assert o["args"][o["args"].index("--entorno-de-env") + 1] == "C5_ENTORNO"
    # los DOS Studios con la GPU (I2 de la auditoría final): el del B con `auto`, el del A con `cpu`
    studios = [o["args"] for o in _ordenes(log) if o["args"][:2] == ["run", "-d"]]
    assert len(studios) == 2 and all("--gpus" in a and a[a.index("--gpus") + 1] == "all" for a in studios), studios
    assert any("MATRIXAI_TABM_DISPOSITIVO=auto" in a and "c5-128-studio-B" in a for a in studios)
    assert any("MATRIXAI_TABM_DISPOSITIVO=cpu" in a and "c5-128-studio-A" in a for a in studios)
    # y con la base y la capa del kit ya en este «Docker», ni se carga ni se construye nada
    assert not any(o["args"][:1] in (["load"], ["build"]) for o in _ordenes(log))


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


# --- 128 (2.ª auditoría, N1): «-Solo a,b» y «-Brazos A,B» TECLEADOS en la consola de PowerShell -------------------------
# Ahí PowerShell los lee como una LISTA: con `[string]$Solo` se juntaba con un espacio («a b») y el medidor paraba con
# «conjuntos que no existen»; con `-Brazos A,B` se medían los dos brazos pero NO se componía el resultado.
_CONSOLA = ("$PSNativeCommandArgumentPassing = '{modo}'; $env:PATH = '/fakebin:' + $env:PATH; "
            "& /kit/lanzar.ps1 {parametros} -Salida /out/salida; exit $LASTEXITCODE")


@pytest.mark.parametrize("modo", ["Standard", "Legacy"])
def test_N1_Solo_con_coma_tecleado_en_la_consola_llega_al_medidor_con_comas_y_Brazos_A_coma_B_compone(tmp_path, modo):
    r, log, fases = _pwsh(tmp_path, _CONSOLA.format(modo=modo, parametros="-Solo dresses-sales,pc1 -Brazos A,B"))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    medidores = [o["args"] for o in _ordenes(log) if "--brazo" in o["args"]]
    assert [m[m.index("--brazo") + 1] for m in medidores] == ["A", "B"], medidores
    for m in medidores:
        assert m[m.index("--solo") + 1] == "dresses-sales,pc1", m               # NO «dresses-sales pc1»
    assert fases[-1] == "componer", fases                                        # «A,B» es «AB»: compone


def test_N1_Solo_entre_comillas_o_por_lanzar_bat_File_y_sin_Solo_siguen_igual(tmp_path):
    r, log, _ = _pwsh(tmp_path, _CONSOLA.format(modo="Standard", parametros="-Solo 'dresses-sales,pc1' -Brazos AB"))
    assert r.returncode == 0, r.stdout[-1500:]
    m = [o["args"] for o in _ordenes(log) if "--brazo" in o["args"]][0]
    assert m[m.index("--solo") + 1] == "dresses-sales,pc1"
    (tmp_path / "sin").mkdir()
    r, log, _ = _pwsh(tmp_path / "sin", _CONSOLA.format(modo="Standard", parametros=""))
    assert r.returncode == 0, r.stdout[-1500:]
    assert all("--solo" not in o["args"] for o in _ordenes(log) if "--brazo" in o["args"])


def test_N5_un_ENTREGADO_sin_dispositivo_declarado_el_B1_real_no_dice_se_entreno_en_None(tmp_path, estudios):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    b = estudios("B", gpu=True, campeon=V.MOTOR_TABM, entregado={"dispositivo": None})
    assert M.main(_args(tmp_path, "B", b)) == 6
    motivo = json.loads((tmp_path / "brazo_B.json").read_text())["corte"]["motivo"]
    assert "'None'" not in motivo and "sin dispositivo declarado" in motivo, motivo
    # y con un dispositivo declarado, sigue diciéndolo tal cual
    b2 = estudios("B", gpu=True, campeon=V.MOTOR_TABM, entregado={"dispositivo": "cpu", "cayo_a_cpu": False})
    (tmp_path / "brazo_B.json").unlink()
    assert M.main(_args(tmp_path, "B", b2)) == 6
    assert "se entrenó en 'cpu'" in json.loads((tmp_path / "brazo_B.json").read_text())["corte"]["motivo"]


# --- 128-C5′: `--sin-cortar` / `-SinCortar` llega al medidor desde los dos lanzadores -----------------------------------
def test_C5p_lanzar_sh_pasa_sin_cortar_al_medidor_solo_si_se_pide(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    r, ordenes, _ = _lanzar(kit, tmp_path, "--sin-cortar")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    medidores = [o for o in ordenes if "medir_c5.py --brazo" in o]
    assert medidores and all(" --sin-cortar" in o for o in medidores), medidores
    (tmp_path / "sin").mkdir()
    r, ordenes, _ = _lanzar(_kit_empaquetado(tmp_path / "sin"), tmp_path / "sin")
    assert r.returncode == 0
    assert not any("--sin-cortar" in o for o in ordenes)


def test_C5p_lanzar_ps1_pasa_sin_cortar_al_medidor_solo_si_se_pide(tmp_path):
    r, log, _ = _pwsh(tmp_path, _CONSOLA.format(modo="Standard", parametros="-SinCortar -Brazos AB"))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    medidores = [o["args"] for o in _ordenes(log) if "--brazo" in o["args"]]
    assert medidores and all("--sin-cortar" in m for m in medidores), medidores
    (tmp_path / "sin").mkdir()
    r, log, _ = _pwsh(tmp_path / "sin", _CONSOLA.format(modo="Standard", parametros=""))
    assert r.returncode == 0
    assert not any("--sin-cortar" in o["args"] for o in _ordenes(log) if "--brazo" in o["args"])


# ------------------------------------------------------------------------------------------------ 128-C5′ (revisión del 10-10)
def test_H18_los_dispositivos_de_TabM_son_solo_los_de_las_entradas_del_motor_TabM():
    est = {"leaderboard": [
        {"engine": V.MOTOR_TABM, "candidate": "t-p0-r0", "wall_seconds": 5.0, "config_efectiva": {
            "hiperparametros": {"device": "cuda"}, "entrenamiento_efectivo": {"dispositivo": "cuda"}}},
        {"engine": V.MOTOR_TABM, "candidate": "t-p1-r0", "wall_seconds": 5.0,
         "config_efectiva": {"hiperparametros": {"device": "cpu", "caida_a_cpu": "cuda_no_disponible"}}},
        {"engine": "matrixai.dense.torch_cpu", "candidate": "d-p0-r0", "wall_seconds": 3.0,
         "config_efectiva": {"hiperparametros": {"device": "cuda"}}},          # otro componente: NO cuenta
        {"engine": "lightgbm", "candidate": "l-p0-r0", "wall_seconds": 1.0}]}
    assert M.dispositivos_de_tabm(est) == {"cuda": 2, "cpu": 1, "caidas_a_cpu": 1}
    assert M.dispositivos_vistos(est)["cuda"] == 3                                 # lo de todo el estado, que NO prueba
    assert M.dispositivos_de_tabm({"leaderboard": [{"engine": "lightgbm", "wall_seconds": 1.0}]}) == {}
    assert M.dispositivos_de_tabm({}) == {}


def test_H18_el_registro_de_un_conjunto_lleva_los_dispositivos_de_TabM(tmp_path, estudios):
    a = estudios("A")
    M.main(_args(tmp_path, "A", a))
    b = estudios("B", gpu=True)
    assert M.main(_args(tmp_path, "B", b, "--solo", "dresses-sales", "--sin-comparar")) == 0
    rec = json.loads((tmp_path / "brazo_B.json").read_text())["conjuntos"]["dresses-sales"]
    assert rec["dispositivos_de_tabm"] == {"cuda": 1} and V.entreno_tabm_en_gpu(rec) is True


def test_H18_la_PRUEBA_con_solo_VRAM_de_mas_y_sin_declaracion_de_dispositivo_NO_da_la_GPU_por_usada(
        tmp_path, estudios, capsys, monkeypatch):
    a = estudios("A")
    M.main(_args(tmp_path, "A", a, "--prueba"))
    b = estudios("B", gpu=False)                                    # el Studio no declara cuda en ningún sitio
    monkeypatch.setattr(M, "picos", lambda m, t0, t1: {"ram_pico_mb": 100.0, "vram_pico_mb": 3000.0,
                                                       "vram_antes_mb": 0.0, "vram_extra_mb": 3000.0,
                                                       "gpu_uso_pico_pct": 50.0, "muestras": 3})
    assert M.main(_args(tmp_path, "B", b, "--prueba")) == 5
    sal = capsys.readouterr().out
    assert "NO SE USO" in sal and "VRAM extra (auxiliar, no prueba): 3000.0" in sal


def _argumentos_de_B(tmp_path, b, **cambios):
    base = {"--imagen-id": "sha256:abc", "--memoria": "6g", "--cpus": "2"}
    base.update(cambios)
    return ["--brazo", "B", "--studio", b.url, "--salida", str(tmp_path), "--datos", str(KIT / "datos"),
            "--imagen", "img:c5"] + [x for kv in base.items() for x in kv]


@pytest.mark.parametrize("cambio", [{"--imagen-id": "sha256:otra"}, {"--memoria": "8g"}, {"--cpus": "4"}])
def test_H16c_el_brazo_B_PARA_antes_de_medir_si_su_huella_no_es_compatible_con_la_del_A(tmp_path, estudios, cambio, capsys):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    b = estudios("B", gpu=True)
    assert M.main(_argumentos_de_B(tmp_path, b, **cambio)) == 6
    sal = capsys.readouterr().out
    assert "no son comparables" in sal and next(iter(cambio)).lstrip("-").replace("-", "_") in sal
    assert b.iniciados == [] and not (tmp_path / "brazo_B.json").exists()          # no se midió NI UN conjunto


def test_H16c_si_el_codigo_del_veredicto_del_A_no_es_el_de_ahora_el_B_PARA(tmp_path, estudios, capsys):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a)) == 0
    ruta = tmp_path / "brazo_A.json"
    brazo_a = json.loads(ruta.read_text())
    brazo_a["huella"]["sha256_veredicto"] = "0" * 64                                # el A se midió con otro veredicto
    ruta.write_text(json.dumps(brazo_a))
    b = estudios("B", gpu=True)
    assert M.main(_args(tmp_path, "B", b)) == 6
    assert "sha256_veredicto" in capsys.readouterr().out and b.iniciados == []


def test_H16c_con_la_misma_huella_el_B_mide_y_el_entorno_distinto_del_brazo_no_cuenta(tmp_path, estudios):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a, "--entorno", json.dumps({"MATRIXAI_TABM_DISPOSITIVO": "cpu"}))) == 0
    b = estudios("B", gpu=True)
    assert M.main(_args(tmp_path, "B", b, "--entorno", json.dumps({"MATRIXAI_TABM_DISPOSITIVO": "auto", "gpus": "all"}))) == 0


def test_H16a_un_brazo_A_sin_ese_conjunto_PARA_al_B_como_instrumento_no_como_regla_9(tmp_path, estudios):
    a = estudios("A")
    assert M.main(_args(tmp_path, "A", a, "--solo", "wilt")) == 0                   # el A terminó, pero SOLO con wilt
    b = estudios("B", gpu=True)
    assert M.main(_args(tmp_path, "B", b, "--solo", "pc1,wilt", "--suelo", "0")) == 6
    brazo = json.loads((tmp_path / "brazo_B.json").read_text())
    assert brazo["corte"]["tipo"] == "instrumento" and "no tiene ese conjunto" in brazo["corte"]["motivo"]
    assert brazo["terminado"] is False and brazo["comparaciones"]["pc1"]["clase"] == "sin_referencia"


def test_R97_componer_enseña_el_instrumento_y_el_producto_por_separado(tmp_path, estudios, capsys):
    a, b = _brazos(tmp_path, estudios, {("B", "micro-mass"): 83.0})
    assert M.main(_args(tmp_path, "B", b)) == 0
    capsys.readouterr()
    assert M.main(["--componer", "--salida", str(tmp_path), "--datos", str(KIT / "datos")]) == 0
    sal = capsys.readouterr().out
    v = json.loads((tmp_path / "resultado_128_c5.json").read_text())["veredicto"]
    assert (v["instrumento_midio_bien"], v["la_regla_se_cumple"], v["producto_entra"]) == (True, True, True)
    assert "INSTRUMENTO: midió bien" in sal and "LA GPU ENTRA" in sal


def test_R97_componer_con_dos_brazos_incomparables_dice_que_el_instrumento_no_midio_bien(tmp_path, estudios, capsys):
    a, b = _brazos(tmp_path, estudios, {("B", "micro-mass"): 83.0})
    assert M.main(_args(tmp_path, "B", b)) == 0
    ruta = tmp_path / "brazo_B.json"
    brazo_b = json.loads(ruta.read_text())
    brazo_b["huella"]["imagen_id"] = "sha256:otra"
    ruta.write_text(json.dumps(brazo_b))
    capsys.readouterr()
    assert M.main(["--componer", "--salida", str(tmp_path), "--datos", str(KIT / "datos")]) == 0
    sal = capsys.readouterr().out
    v = json.loads((tmp_path / "resultado_128_c5.json").read_text())["veredicto"]
    assert v["instrumento_midio_bien"] is False and v["la_regla_se_cumple"] is None and v["producto_entra"] is False
    assert any("imagen_id" in m for m in v["motivos_del_instrumento"])
    assert "INSTRUMENTO: NO midió bien" in sal and "LA GPU NO ENTRA" in sal


def test_H16b_el_BANCO_del_veredicto_es_el_manifiesto_del_kit():
    manifiesto = M.leer_manifiesto(KIT / "datos")["conjuntos"]
    assert list(V.BANCO) == [c["nombre"] for c in sorted(manifiesto, key=lambda x: x["orden"])]


# ------------------------------------------------------------------------------------------------ 128-R4 en GPU: el paso OPCIONAL «TabM por fases»
# `lanzar.* --fases` / `-Fases`, DESPUÉS de los dos brazos: la sonda `sonda_128_c0.py` (que viaja en el kit) corre DENTRO de la imagen GPU,
# con `--gpus all` y el techo de memoria, sobre los CSV del kit. Aquí, con un docker FALSO; la GPU de verdad no se puede comprobar.
CONJUNTOS_DE_FASES = "APSFailure,KDDCup09_appetency,Allstate_Claims_Severity,house_16H,mfeat-factors"


def _lanzar_libre(kit, tmp_path, *args):
    """`lanzar.sh` con un docker falso y SIN forzar `--prueba --brazos B`. Devuelve (resultado, órdenes de docker, una por línea)."""
    import os
    bin_, estado, log = _docker_falso(tmp_path) if not (tmp_path / "bin").exists() else (
        tmp_path / "bin", tmp_path / "estado", tmp_path / "docker.log")
    entorno = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "LOG_DOCKER": str(log), "ESTADO_DOCKER": str(estado)}
    r = subprocess.run(["bash", str(kit / "lanzar.sh"), "--salida", str(tmp_path / "salida"), *args],
                       capture_output=True, text=True, env=entorno, timeout=120)
    return r, (log.read_text().splitlines() if log.exists() else [])


def _la_de_fases(ordenes):
    return [o for o in ordenes if "sonda_128_c0.py" in o]


def test_fases_sh_sin_el_flag_NO_hay_paso_de_fases_y_el_flujo_es_el_de_siempre(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert not _la_de_fases(ordenes) and any("--componer" in o for o in ordenes)
    assert "Paso opcional" not in r.stdout and not (tmp_path / "salida" / "tabm_por_fases_gpu.json").exists()


def _con_imagen_hecha(tmp_path, nombre="otra_gpu"):
    (tmp_path / "estado").mkdir(exist_ok=True)
    (tmp_path / "estado" / nombre).touch()


def test_fases_sh_tras_los_dos_brazos_y_tras_componer_corre_la_sonda_en_la_imagen_GPU_con_los_CSV_del_kit(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu", "--fases")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-800:]
    f = _la_de_fases(ordenes)
    assert len(f) == 1, f
    o = f[0]
    assert o.startswith("run ") and "--gpus all" in o and "--memory=6g" in o and "--memory-swap=6g" in o and "--cpus=2" in o
    assert "--name c5-128-fases" in o and "--init" in o and "--entrypoint python3" in o and "otra_gpu " in o
    assert "-v " + str(kit) + ":/work/kit:ro" in o and ":/salida" in o
    assert "/work/kit/sonda_128_c0.py --tabm-por-fases --gpu --dentro-del-contenedor --datos-del-kit /work/kit/datos" in o
    assert "--salida /salida/tabm_por_fases_gpu.json" in o and "--temporales /salida/fases_tmp" in o
    assert "--plazo-amplio 1800" in o and "--solo " + CONJUNTOS_DE_FASES in o and "--memoria 6g" in o
    # DESPUÉS de los dos brazos y de componer
    indice = {k: next(i for i, x in enumerate(ordenes) if k in x) for k in ("--brazo A", "--brazo B", "--componer")}
    assert indice["--brazo A"] < indice["--brazo B"] < indice["--componer"] < ordenes.index(o)
    # y se comprueba que Docker ve la GPU (el paso 0 también vale para este paso)
    assert any("--gpus all --entrypoint python3 otra_gpu -c import torch" in x for x in ordenes)


def test_fases_sh_con_prueba_usa_un_conjunto_pequeno_y_plazo_corto_y_con_opciones_las_respeta(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu", "--brazos", "B", "--prueba", "--fases")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-800:]
    (o,) = _la_de_fases(ordenes)
    assert "--solo dresses-sales" in o and "--plazo-amplio 120" in o
    assert not any("--componer" in x for x in ordenes)                       # una prueba no compone
    (tmp_path / "b").mkdir(); _docker_falso(tmp_path / "b"); _con_imagen_hecha(tmp_path / "b")
    r, ordenes = _lanzar_libre(_kit_empaquetado(tmp_path / "b"), tmp_path / "b", "--imagen", "otra_gpu", "--solo-fases",
                               "--conjuntos-fases", "house_16H, mfeat-factors", "--plazo-fases", "600", "--memoria", "24g")
    (o,) = _la_de_fases(ordenes)
    assert "--solo house_16H,mfeat-factors" in o and "--plazo-amplio 600" in o and "--memory=24g" in o and "--memoria 24g" in o


def test_fases_sh_solo_fases_no_mide_los_brazos_ni_compone_pero_si_comprueba_la_GPU(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu", "--solo-fases")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-800:]
    assert len(_la_de_fases(ordenes)) == 1
    assert not any("--brazo" in x or "--componer" in x or x.startswith("run -d") for x in ordenes)
    assert any("--gpus all --entrypoint python3 otra_gpu -c import torch" in x for x in ordenes)


def test_fases_sh_con_un_solo_brazo_no_corre_el_paso_porque_va_despues_de_los_dos(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu", "--brazos", "A", "--fases")
    assert r.returncode == 0 and not _la_de_fases(ordenes), ordenes


def test_fases_sh_si_la_GPU_no_se_ve_para_con_3_y_no_arranca_la_sonda(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    bin_, estado, log = _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    falso = (bin_ / "docker").read_text().replace('case "$1" in\n  load)', 'case "$*" in *"import torch"*) exit 1;; esac\ncase "$1" in\n  load)', 1)
    assert falso != (bin_ / "docker").read_text()
    (bin_ / "docker").write_text(falso)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu", "--solo-fases")
    assert r.returncode == 3 and "no ve la GPU" in r.stdout and not _la_de_fases(ordenes)


def test_fases_sh_el_fallo_de_la_sonda_se_devuelve_y_lo_ya_compuesto_queda_escrito(tmp_path, monkeypatch):
    kit = _kit_empaquetado(tmp_path)
    bin_, estado, log = _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    falso = (bin_ / "docker").read_text().replace('case "$1" in\n  load)', 'case "$*" in *sonda_128_c0*) exit 5;; esac\ncase "$1" in\n  load)', 1)
    assert falso != (bin_ / "docker").read_text()
    (bin_ / "docker").write_text(falso)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu", "--fases")
    assert r.returncode == 5 and "El paso de fases termino con codigo 5" in r.stdout
    assert any("--componer" in x for x in ordenes) and "Hecho. Manda:" in r.stdout          # el resultado principal ya estaba compuesto


def test_fases_los_dos_lanzadores_declaran_las_mismas_opciones_y_el_mismo_paso():
    sh, ps = (KIT / "lanzar.sh").read_text(), (KIT / "lanzar.ps1").read_text()
    for t in (sh, ps):
        assert "sonda_128_c0.py" in t and "--tabm-por-fases" in t and "--gpu" in t and "--dentro-del-contenedor" in t
        assert "--datos-del-kit" in t and "/work/kit/datos" in t and "tabm_por_fases_gpu.json" in t and "c5-128-fases" in t
        assert "--gpus" in t and "--tope-estudio" in t and CONJUNTOS_DE_FASES in t
    assert "--fases" in sh and "--solo-fases" in sh and "--conjuntos-fases" in sh and "--plazo-fases" in sh
    assert "$Fases" in ps and "$SoloFases" in ps and "$ConjuntosFases" in ps and "$PlazoFases" in ps
    (KIT / "lanzar.ps1").read_bytes().decode("ascii")


def test_fases_ps1_trata_ConjuntosFases_como_una_lista_igual_que_Solo_y_la_une_con_comas():
    """N1 (2.ª auditoría): en la consola de PowerShell `-ConjuntosFases a,b` llega como LISTA. Con `[string]` se juntaría con un espacio."""
    ps = (KIT / "lanzar.ps1").read_text()
    assert "[string[]]$ConjuntosFases" in ps
    assert '[string]$ConjuntosFases = (($ConjuntosFases | Where-Object { $_ -ne "" }) -join ",")' in ps
    assert '"--solo", $conj' in ps and "@args_" in ps           # un solo texto «a,b» en un array de argumentos, sin comillas dobles dentro


def test_fases_el_leeme_cuenta_el_paso_con_su_orden_exacta_y_dice_que_la_estimacion_no_esta_medida():
    t = (KIT / "LEEME.md").read_text()
    assert ".\\lanzar.ps1 -Fases" in t and "./lanzar.sh --fases" in t and "-SoloFases" in t and "--solo-fases" in t
    for c in CONJUNTOS_DE_FASES.split(","):
        assert c in t
    i = t.index("4. Opcional")
    paso = t[i:t.index("## Qué me mandas")]
    assert "Opcional (estimado 20 min – 1,5 h, no medido)" in paso and "Cálculo de la estimación (no medida;" in paso   # dicha como estimada, dos veces
    assert "minutos por conjunto" in paso and "tabm_por_fases_gpu.json" in paso
    assert "--gpus all" in paso and "6 GB" in paso and "1.800" in paso
    # M10 de la auditoría final: el paso usa la memoria de `-Memoria` (la orden de Roberto lleva 16g), no «6 GB» fijos
    assert "la misma memoria que los brazos (`-Memoria` / `--memoria`" in paso and "y 6 GB:" not in paso


@pytest.mark.parametrize("modo", ["Standard", "Legacy"])
def test_fases_ps1_corre_el_paso_tras_componer_con_la_lista_tecleada_en_la_consola(tmp_path, modo):
    r, log, fases = _pwsh(tmp_path, _CONSOLA.format(modo=modo, parametros="-Fases -ConjuntosFases house_16H,mfeat-factors -PlazoFases 600"))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    assert fases == ["medidor", "A", "medidor", "B", "componer"], fases
    ordenes = [o["args"] for o in _ordenes(log)]
    (sonda,) = [a for a in ordenes if "/work/kit/sonda_128_c0.py" in a]
    assert "--gpus" in sonda and sonda[sonda.index("--gpus") + 1] == "all" and "--memory=6g" in sonda
    assert sonda[sonda.index("--solo") + 1] == "house_16H,mfeat-factors"            # NO «house_16H mfeat-factors»
    assert sonda[sonda.index("--plazo-amplio") + 1] == "600" and sonda[sonda.index("--salida") + 1] == "/salida/tabm_por_fases_gpu.json"
    assert "--dentro-del-contenedor" in sonda and "--tabm-por-fases" in sonda
    assert ordenes.index(sonda) > max(i for i, a in enumerate(ordenes) if "--componer" in a)


def test_fases_ps1_sin_lista_usa_los_cinco_de_siempre_y_SoloFases_no_mide_brazos(tmp_path):
    r, log, fases = _pwsh(tmp_path, _CONSOLA.format(modo="Standard", parametros="-SoloFases"))
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    assert fases == [], fases                                                       # ni medidor ni componer
    (sonda,) = [o["args"] for o in _ordenes(log) if "/work/kit/sonda_128_c0.py" in o["args"]]
    assert sonda[sonda.index("--solo") + 1] == CONJUNTOS_DE_FASES and sonda[sonda.index("--plazo-amplio") + 1] == "1800"


def test_fases_ps1_sin_Fases_no_hay_paso_y_con_un_solo_brazo_tampoco(tmp_path):
    r, log, _ = _pwsh(tmp_path, _CONSOLA.format(modo="Standard", parametros="-Brazos AB"))
    assert r.returncode == 0 and not any("/work/kit/sonda_128_c0.py" in o["args"] for o in _ordenes(log))
    (tmp_path / "a").mkdir()
    r, log, _ = _pwsh(tmp_path / "a", _CONSOLA.format(modo="Standard", parametros="-Brazos A -Fases"))
    assert r.returncode == 0 and not any("/work/kit/sonda_128_c0.py" in o["args"] for o in _ordenes(log))


# --- el empaquetado: la sonda viaja en el kit, idéntica, con su SHA256SUMS, y arranca desde allí
def test_fases_el_empaquetado_mete_la_sonda_identica_con_su_suma_y_la_sonda_arranca_desde_el_kit(tmp_path):
    emp = _cargar(_FASE0 / "empaquetar_kit_128_c5.py", "empaquetar_la_sonda")
    assert "sonda_128_c0.py" in emp.PIEZAS_DEL_REPO
    destino = tmp_path / "kit"
    escritos = emp.armar(destino)
    assert (destino / "sonda_128_c0.py").read_bytes() == (_FASE0 / "sonda_128_c0.py").read_bytes()
    sumas = dict(reversed(l.split("  ", 1)) for l in (destino / "SHA256SUMS").read_text().splitlines())
    assert sumas["sonda_128_c0.py"] == hashlib.sha256((_FASE0 / "sonda_128_c0.py").read_bytes()).hexdigest()
    assert destino / "sonda_128_c0.py" in escritos and "referencia_120_r0.py" in sumas              # y la pieza que la sonda importa por ruta
    # sin el repo ni sus hermanos: la sonda del kit se carga (importa `referencia_120_r0.py` de su carpeta) y reconoce sus opciones
    r = subprocess.run([sys.executable, "-I", str(destino / "sonda_128_c0.py"), "--help"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "--datos-del-kit" in r.stdout and "--dentro-del-contenedor" in r.stdout and "--gpu" in r.stdout
    assert not list(destino.rglob("__pycache__"))


# ------------------------------------------------------------------------------------------------ 128, auditoría final: I3 y el brazo A
# I3: los dos kits del 128 usan las mismas etiquetas (`matrixai-studio:128`, `matrixai-studio-gpu-128:local`) y el lanzador
# solo cargaba y construía si la capa NO existía: el kit 2, seguido al pie de su LEEME (sin `-Reconstruir`), medía en
# silencio la imagen del kit 1 —sin el arreglo del dropout ni FP32— y decía «la GPU SE USÓ». Ahora el empaquetador apunta
# el ID de la imagen DEL TAR (`imagen/matrixai-studio-128.ids`) y el lanzador lo compara con la base de este Docker y con
# la base sobre la que se construyó la capa; si no es la del kit, recarga y reconstruye él solo, o para diciendo por qué.
def test_I3_el_empaquetado_apunta_los_IDs_de_la_imagen_LEIDOS_DEL_TAR(tmp_path):
    emp = _cargar(_FASE0 / "empaquetar_kit_128_c5.py", "empaquetar_ids")
    tar = tmp_path / "base.tar"
    del_indice, de_la_config = _tar_oci_falso(tar)
    kit = tmp_path / "kit"
    emp.armar(kit, tar_base=tar)
    assert _ids_del_kit(kit) == [del_indice, de_la_config]           # containerd (este servidor) y el almacén clásico
    sumas = dict(reversed(l.split("  ", 1)) for l in (kit / "SHA256SUMS").read_text().splitlines())
    assert "imagen/matrixai-studio-128.ids" in sumas                  # viaja en el kit con su suma
    # otro tar (otra imagen con la MISMA etiqueta) da otros IDs: es lo que distingue el kit 2 del 1
    otro = tmp_path / "otro.tar"
    assert set(_tar_oci_falso(otro, semilla="kit-1")).isdisjoint({del_indice, de_la_config})


def test_I3_un_tar_sin_la_etiqueta_o_con_un_blob_que_no_es_lo_que_dice_PARA_al_empaquetar(tmp_path):
    import tarfile
    emp = _cargar(_FASE0 / "empaquetar_kit_128_c5.py", "empaquetar_ids_malos")
    sin_etiqueta = tmp_path / "otra.tar"
    _tar_oci_falso(sin_etiqueta, etiqueta="matrixai-studio:otra")
    with pytest.raises(SystemExit, match="no trae la etiqueta matrixai-studio:128"):
        emp.armar(tmp_path / "k1", tar_base=sin_etiqueta)
    bueno = tmp_path / "bueno.tar"
    _, config = _tar_oci_falso(bueno)
    roto = tmp_path / "roto.tar"
    with tarfile.open(bueno) as t, tarfile.open(roto, "w") as r:
        for m in t.getmembers():
            datos = t.extractfile(m).read()
            if m.name == f"blobs/sha256/{config.split(':')[1]}":
                datos = datos.replace(b"base-128", b"base-129")         # mismo tamaño, otro contenido
            r.addfile(m, __import__("io").BytesIO(datos))
    with pytest.raises(SystemExit, match="no es lo que dice su nombre"):
        emp.armar(tmp_path / "k2", tar_base=roto)
    with pytest.raises(SystemExit, match="no es un `docker save`"):
        (tmp_path / "nada.tar").write_bytes(b"")
        with tarfile.open(tmp_path / "nada.tar", "w"):
            pass
        emp.armar(tmp_path / "k3", tar_base=tmp_path / "nada.tar")


_SHA_DEL_DOCKERFILE_GPU = hashlib.sha256(_DOCKERFILE_GPU.read_bytes()).hexdigest()


def _con_imagenes_de_otro_kit(estado, base="sha256:kit-1", capa_sobre="sha256:kit-1", dockerfile=_SHA_DEL_DOCKERFILE_GPU):
    """El Docker del portátil de Roberto tras el kit 1: las MISMAS etiquetas, otra imagen. `dockerfile`: el sha256 del
    Dockerfile.gpu con que se construyó la capa (128, m7); `None`: una capa de antes de esa etiqueta."""
    (estado / "matrixai-studio_128").write_text(base)
    (estado / "matrixai-studio-gpu-128_local").write_text("sha256:capa-vieja")
    if capa_sobre is not None:
        (estado / "matrixai-studio-gpu-128_local.base").write_text(capa_sobre)
        if dockerfile is not None:
            (estado / "matrixai-studio-gpu-128_local.dockerfile").write_text(dockerfile)


def test_I3_sh_con_la_base_de_OTRO_kit_y_su_capa_recarga_el_tar_y_rehace_la_capa_sin_Reconstruir(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    r, ordenes, estado = _lanzar(kit, tmp_path, antes=_con_imagenes_de_otro_kit)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert "NO es la de este kit (sha256:kit-1)" in r.stdout, r.stdout[-1500:]
    assert sum(o.startswith("load -i ") for o in ordenes) == 1
    (construye,) = [o for o in ordenes if o.startswith("build ")]
    assert construye.startswith("build -t matrixai-studio-gpu-128:local --label matrixai.kit.base=" + _ids_del_kit(kit)[0])
    # y los brazos miden contra la capa NUEVA
    assert (estado / "matrixai-studio-gpu-128_local.base").read_text() == _ids_del_kit(kit)[0]
    assert ordenes.index(construye) < next(i for i, o in enumerate(ordenes) if "medir_c5.py --brazo" in o)


def test_I3_sh_con_la_base_del_kit_y_una_capa_sobre_otra_base_solo_rehace_la_capa(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    id_del_kit = _ids_del_kit(kit)[0]
    r, ordenes, _ = _lanzar(kit, tmp_path, antes=lambda e: _con_imagenes_de_otro_kit(e, base=id_del_kit))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert not any(o.startswith("load ") for o in ordenes)
    assert sum(o.startswith(f"build -t matrixai-studio-gpu-128:local --label matrixai.kit.base={id_del_kit} ")
               for o in ordenes) == 1
    assert "no se construyo sobre la base de este kit" in r.stdout
    # una capa construida a mano sin la etiqueta (la de un kit anterior al arreglo) también se rehace
    (tmp_path / "b").mkdir()
    kit_b = _kit_empaquetado(tmp_path / "b")
    r, ordenes, _ = _lanzar(kit_b, tmp_path / "b",
                            antes=lambda e: _con_imagenes_de_otro_kit(e, base=_ids_del_kit(kit_b)[0], capa_sobre=None))
    assert r.returncode == 0 and sum(o.startswith("build ") for o in ordenes) == 1


def test_I3_sh_con_la_base_y_la_capa_del_kit_no_carga_ni_construye(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    id_del_kit = _ids_del_kit(kit)[0]
    r, ordenes, _ = _lanzar(kit, tmp_path, antes=lambda e: _con_imagenes_de_otro_kit(e, base=id_del_kit,
                                                                                    capa_sobre=id_del_kit))
    assert r.returncode == 0, r.stdout[-1500:]
    assert not any(o.startswith(("load ", "build ")) for o in ordenes)
    # la base con el ID del ALMACÉN CLÁSICO (el de la configuración) también es la del kit
    (tmp_path / "c").mkdir()
    kit_c = _kit_empaquetado(tmp_path / "c")
    clasico = _ids_del_kit(kit_c)[1]
    r, ordenes, _ = _lanzar(kit_c, tmp_path / "c", antes=lambda e: _con_imagenes_de_otro_kit(e, base=clasico,
                                                                                            capa_sobre=clasico))
    assert r.returncode == 0 and not any(o.startswith(("load ", "build ")) for o in ordenes)


@pytest.mark.parametrize("dockerfile_de_la_capa", [None, "0" * 64], ids=["capa_de_antes_sin_la_etiqueta", "otro_Dockerfile"])
def test_m7_sh_con_la_base_del_kit_pero_otro_Dockerfile_gpu_rehace_la_capa(tmp_path, dockerfile_de_la_capa):
    """128, m7 de la re-auditoría final: la etiqueta de la base no cubría el Dockerfile.gpu. La capa sobre LA base del
    kit, pero construida con otro Dockerfile (o sin esa etiqueta), se rehace; y la etiqueta nueva lleva el sha256 del
    Dockerfile.gpu de ESTE kit."""
    kit = _kit_empaquetado(tmp_path)
    id_del_kit = _ids_del_kit(kit)[0]
    r, ordenes, estado = _lanzar(kit, tmp_path, antes=lambda e: _con_imagenes_de_otro_kit(
        e, base=id_del_kit, capa_sobre=id_del_kit, dockerfile=dockerfile_de_la_capa))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert not any(o.startswith("load ") for o in ordenes)
    (construye,) = [o for o in ordenes if o.startswith("build ")]
    assert f"--label matrixai.kit.dockerfile={_SHA_DEL_DOCKERFILE_GPU} " in construye + " ", construye
    assert (estado / "matrixai-studio-gpu-128_local.dockerfile").read_text() == _SHA_DEL_DOCKERFILE_GPU
    assert "con su Dockerfile.gpu: la rehago" in r.stdout


def test_I3_sh_si_el_tar_cargado_no_es_el_que_apunta_el_kit_PARA_sin_construir_ni_medir(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    r, ordenes, _ = _lanzar(kit, tmp_path, id_del_tar="sha256:otro-tar")
    assert r.returncode == 2 and "el tar no es el de este kit" in r.stdout, r.stdout[-1500:]
    assert sum(o.startswith("load ") for o in ordenes) == 1
    assert not any(o.startswith("build ") or "medir_c5.py --brazo" in o for o in ordenes)


def test_I3_sh_un_kit_sin_su_identidad_PARA_en_vez_de_medir_lo_que_haya(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    (kit / "imagen" / "matrixai-studio-128.ids").unlink()
    r, ordenes, _ = _lanzar(kit, tmp_path, antes=lambda e: _con_imagenes_de_otro_kit(e))
    assert r.returncode == 2 and "matrixai-studio-128.ids" in r.stdout, r.stdout[-1500:]
    assert not any(o.startswith(("load ", "build ")) or "medir_c5.py --brazo" in o for o in ordenes)


def test_I2_sh_los_DOS_brazos_con_la_GPU_A_cpu_B_auto_y_el_medidor_lo_apunta(tmp_path):
    kit = _kit_empaquetado(tmp_path)
    _docker_falso(tmp_path); _con_imagen_hecha(tmp_path)
    r, ordenes = _lanzar_libre(kit, tmp_path, "--imagen", "otra_gpu")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    studios = [o for o in ordenes if o.startswith("run -d")]
    assert len(studios) == 2 and all("--gpus all" in o for o in studios), studios
    assert any("c5-128-studio-A" in o and "MATRIXAI_TABM_DISPOSITIVO=cpu" in o for o in studios)
    assert any("c5-128-studio-B" in o and "MATRIXAI_TABM_DISPOSITIVO=auto" in o for o in studios)
    (medidor_a,) = [o for o in ordenes if "medir_c5.py --brazo A" in o]
    assert '--entorno {"MATRIXAI_TABM_DISPOSITIVO":"cpu","gpus":"all"}' in medidor_a
    # y el paso 0 (Docker ve la GPU) también con un solo brazo A: el A ya la necesita
    (tmp_path / "a").mkdir(); _docker_falso(tmp_path / "a"); _con_imagen_hecha(tmp_path / "a")
    r, ordenes = _lanzar_libre(_kit_empaquetado(tmp_path / "a"), tmp_path / "a", "--imagen", "otra_gpu", "--brazos", "A")
    assert r.returncode == 0 and any("--gpus all --entrypoint python3 otra_gpu -c import torch" in o for o in ordenes)


def test_I2_el_veredicto_admite_los_dos_brazos_con_la_GPU_a_la_vista():
    """El brazo A con `gpus: all` en su entorno no es «otra cosa» para la huella: lo único que distingue a los dos brazos
    sigue siendo la señal (y que vean la GPU, que ahora es igual)."""
    base = {"imagen_id": "sha256:x", "sha256_medir_c5": "a", "sha256_veredicto": "b", "sha256_manifiesto": "c",
            "memoria": "6g", "cpus": "2"}
    a = {**base, "entorno": {"MATRIXAI_TABM_DISPOSITIVO": "cpu", "gpus": "all"}, "brazo": "A"}
    b = {**base, "entorno": {"MATRIXAI_TABM_DISPOSITIVO": "auto", "gpus": "all"}, "brazo": "B"}
    assert V.incompatibilidades_de_huellas(a, b) == []


@pytest.mark.parametrize("modo", ["Standard", "Legacy"])
def test_I3_ps1_con_la_base_de_OTRO_kit_recarga_y_rehace_la_capa_y_mide_contra_ella(tmp_path, modo):
    kit = _kit_empaquetado(tmp_path)
    id_del_kit = _ids_del_kit(kit)[0]
    r, log, fases = _pwsh(tmp_path, _LANZAR.format(modo=modo), {
        "FALSO_ID_BASE": "sha256:kit-1", "FALSO_ID_TRAS_CARGAR": id_del_kit, "FALSO_BASE_DE_LA_CAPA": "sha256:kit-1"})
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    assert "NO es la de este kit (sha256:kit-1)" in r.stdout, r.stdout[-2000:]
    ordenes = [o["args"] for o in _ordenes(log)]
    assert sum(a[:1] == ["load"] for a in ordenes) == 1
    (construye,) = [a for a in ordenes if a[:1] == ["build"]]
    # la etiqueta llega ENTERA también con el paso de argumentos de 5.1 («Legacy»)
    assert construye[construye.index("--label") + 1] == f"matrixai.kit.base={id_del_kit}", construye
    assert fases == ["medidor", "A", "medidor", "B", "componer"], fases
    assert ordenes.index(construye) < next(i for i, a in enumerate(ordenes) if "--brazo" in a)


def test_I3_ps1_con_la_base_del_kit_y_la_capa_sobre_otra_solo_rehace_la_capa(tmp_path):
    r, log, fases = _pwsh(tmp_path, _LANZAR.format(modo="Standard"), {"FALSO_BASE_DE_LA_CAPA": "sha256:kit-1"})
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    ordenes = [o["args"] for o in _ordenes(log)]
    assert not any(a[:1] == ["load"] for a in ordenes) and sum(a[:1] == ["build"] for a in ordenes) == 1
    assert "no se construyo sobre la base de este kit" in r.stdout and fases[-1] == "componer"


def test_m7_ps1_con_la_base_del_kit_pero_otro_Dockerfile_gpu_rehace_la_capa(tmp_path):
    r, log, fases = _pwsh(tmp_path, _LANZAR.format(modo="Standard"), {"FALSO_DOCKERFILE_DE_LA_CAPA": "0" * 64})
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]
    ordenes = [o["args"] for o in _ordenes(log)]
    (construye,) = [a for a in ordenes if a[:1] == ["build"]]
    sha = hashlib.sha256((tmp_path / "kit" / "imagen" / "Dockerfile.gpu").read_bytes()).hexdigest()
    assert f"matrixai.kit.dockerfile={sha}" in construye, construye
    assert not any(a[:1] == ["load"] for a in ordenes) and "con su Dockerfile.gpu: la rehago" in r.stdout


def test_I3_ps1_si_el_tar_cargado_no_es_el_del_kit_PARA_sin_construir_ni_medir(tmp_path):
    r, log, fases = _pwsh(tmp_path, _LANZAR.format(modo="Standard"), {
        "FALSO_ID_BASE": "sha256:kit-1", "FALSO_ID_TRAS_CARGAR": "sha256:otro-tar"})
    assert r.returncode == 2 and "el tar no es el de este kit" in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
    ordenes = [o["args"] for o in _ordenes(log)]
    assert sum(a[:1] == ["load"] for a in ordenes) == 1 and not any(a[:1] == ["build"] for a in ordenes)
    assert fases == []
