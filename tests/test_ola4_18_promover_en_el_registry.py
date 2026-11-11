"""Ola 4, idea 18 — ``ModelRegistry.promote``: mover ``current`` es un acto humano con constancia.

Contrato 83 §5.2: «promover es un acto humano y queda registrado con quién y cuándo». ``tag`` solo
mueve una etiqueta y no deja rastro de quién; ``promote`` exige el quién, comprueba la entrada,
mueve ``current`` y escribe una línea firmada en ``promotions.jsonl``.
"""
from __future__ import annotations

import json

import pytest

from matrixai.registry import EntryNotFoundError, ModelRegistry, ModelRegistryError
from tests.test_p21_cut3_model_registry import _make_entry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    r = ModelRegistry(tmp_path / "reg")
    r.push(_make_entry(version="v1"))
    r.push(_make_entry(version="v2", parameter_set_id="ps_002"))
    return r


def test_promover_mueve_current_y_devuelve_la_constancia(reg):
    assert reg.current_version("text_encoder") is None
    rec = reg.promote("text_encoder", "v1", by="Ana")
    assert reg.current_version("text_encoder") == "v1"
    assert (rec["version"], rec["previous_version"], rec["promoted_by"]) == ("v1", None, "Ana")
    assert rec["promoted_at"]
    # la etiqueta resuelve como cualquier otra
    assert reg.get("text_encoder", "current").version == "v1"


def test_promover_recuerda_la_vigente_anterior(reg):
    reg.promote("text_encoder", "v1", by="Ana")
    rec = reg.promote("text_encoder", "v2", by="Luis", note="mejor recall")
    assert rec["previous_version"] == "v1"
    assert reg.current_version("text_encoder") == "v2"
    assert rec["note"] == "mejor recall"


def test_sin_quien_no_se_promueve_y_current_no_se_mueve(reg):
    for vacio in ("", "   "):
        with pytest.raises(ModelRegistryError, match="who is promoting"):
            reg.promote("text_encoder", "v1", by=vacio)
    assert reg.current_version("text_encoder") is None
    assert not (reg.path / "promotions.jsonl").exists()


def test_una_version_que_no_existe_o_una_etiqueta_no_se_promueven(reg):
    reg.promote("text_encoder", "v1", by="Ana")
    with pytest.raises(EntryNotFoundError):
        reg.promote("text_encoder", "v9", by="Ana")
    with pytest.raises(EntryNotFoundError, match="not a tag"):
        reg.promote("text_encoder", "current", by="Ana")
    assert reg.current_version("text_encoder") == "v1"


def test_una_entrada_manipulada_no_se_promueve(reg):
    mf = reg.path / "entries" / "text_encoder" / "v2" / "manifest.json"
    datos = json.loads(mf.read_text())
    datos["entry_hash"] = "sha256:" + "0" * 64
    mf.write_text(json.dumps(datos))
    with pytest.raises(ModelRegistryError):
        reg.promote("text_encoder", "v2", by="Ana")
    assert reg.current_version("text_encoder") is None
    assert reg.promotions() == []


def test_la_constancia_va_firmada_y_se_comprueba_al_leerla(reg):
    reg.promote("text_encoder", "v1", by="Ana")
    reg.promote("text_encoder", "v2", by="Luis")
    lista = reg.promotions("text_encoder")
    assert [p["version"] for p in lista] == ["v1", "v2"]
    assert all(p["signature_ok"] is True for p in lista)
    assert reg.promotions("otro") == []


def test_cambiar_el_quien_en_el_fichero_rompe_la_firma(reg):
    reg.promote("text_encoder", "v1", by="Ana")
    f = reg.path / "promotions.jsonl"
    f.write_text(f.read_text().replace("Ana", "Mallory"))
    assert reg.promotions()[0]["promoted_by"] == "Mallory"
    assert reg.promotions()[0]["signature_ok"] is False


def test_si_no_se_puede_escribir_la_constancia_current_vuelve_a_como_estaba(reg):
    reg.promote("text_encoder", "v1", by="Ana")
    # un DIRECTORIO donde debería estar el fichero: abrirlo para añadir falla
    (reg.path / "promotions.jsonl").unlink()
    (reg.path / "promotions.jsonl").mkdir()
    with pytest.raises(ModelRegistryError, match="left as it was"):
        reg.promote("text_encoder", "v2", by="Luis")
    assert reg.current_version("text_encoder") == "v1"


# ── Reparación de la auditoría del 11-11 (I4, I5, M5) ──────────────────────────


def _cadena_rota(proms):
    """Cuántas constancias dicen sustituir a una versión que NO era la de la constancia anterior."""
    return sum(1 for a, b in zip(proms, proms[1:]) if b["previous_version"] != a["version"])


@pytest.fixture
def reg4(tmp_path, monkeypatch):
    monkeypatch.delenv("MATRIXAI_REGISTRY_SIGNING_KEY", raising=False)
    r = ModelRegistry(tmp_path / "reg")
    for i in range(1, 5):
        r.push(_make_entry(version=f"v{i}", parameter_set_id=f"ps_{i:03d}"))
    return r


def test_varios_hilos_promoviendo_y_leyendo_no_dan_lecturas_rotas_ni_constancias_incoherentes(reg4):
    """I4: `tag()` escribía con `write_text` (no atómico) y `promote` leía-movía-anotaba sin candado. Medido por la auditoría:
    717 lecturas rotas de 3.189 y 87 de 241 constancias con un `previous_version` falso. Los errores de los hilos se RECOGEN y se
    afirman aquí, en el principal (una aserción dentro de un hilo no suspende la prueba)."""
    import threading

    reg4.promote("text_encoder", "v1", by="base")
    errores_lector: list[str] = []
    errores_prom: list[str] = []
    lecturas = [0]
    parar = threading.Event()

    def lector():
        while not parar.is_set():
            try:
                reg4.get("text_encoder", "current")
                reg4.current_version("text_encoder")
                lecturas[0] += 1
            except Exception as exc:  # noqa: BLE001
                errores_lector.append(f"{type(exc).__name__}: {exc}")

    def promotor(i):
        for k in range(40):
            try:
                reg4.promote("text_encoder", f"v{(i + k) % 4 + 1}", by=f"h{i}")
            except Exception as exc:  # noqa: BLE001
                errores_prom.append(f"{type(exc).__name__}: {exc}")

    lectores = [threading.Thread(target=lector) for _ in range(4)]
    promotores = [threading.Thread(target=promotor, args=(i,)) for i in range(4)]
    for t in lectores + promotores:
        t.start()
    for t in promotores:
        t.join()
    parar.set()
    for t in lectores:
        t.join()
    assert lecturas[0] > 0
    assert errores_lector == []
    assert errores_prom == []
    proms = reg4.promotions("text_encoder")
    assert len(proms) == 1 + 4 * 40
    assert _cadena_rota(proms) == 0
    assert reg4.current_version("text_encoder") == proms[-1]["version"]


_PROMOTOR_DE_OTRO_PROCESO = """
import sys
from matrixai.registry import ModelRegistry
r = ModelRegistry(sys.argv[1])
for k in range(30):
    r.promote("text_encoder", f"v{(int(sys.argv[2]) + k) % 4 + 1}", by="p" + sys.argv[2])
"""


def test_dos_procesos_promoviendo_a_la_vez_dejan_la_cadena_coherente(reg4):
    """I4, la mitad entre PROCESOS (dos Studios o un CI con el mismo registro): el candado de hilo no los ve; el de fichero sí."""
    import os
    import subprocess
    import sys

    reg4.promote("text_encoder", "v1", by="base")
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entorno = {**os.environ, "PYTHONPATH": raiz + os.pathsep + os.environ.get("PYTHONPATH", "")}
    procesos = [subprocess.Popen([sys.executable, "-c", _PROMOTOR_DE_OTRO_PROCESO, str(reg4.path), str(i)],
                                 env=entorno, stderr=subprocess.PIPE, text=True) for i in range(3)]
    salidas = [(p.wait(timeout=120), p.stderr.read()) for p in procesos]
    assert [rc for rc, _ in salidas] == [0, 0, 0], salidas
    proms = reg4.promotions("text_encoder")
    assert len(proms) == 1 + 3 * 30
    assert _cadena_rota(proms) == 0


def test_una_etiqueta_current_ilegible_no_se_lee_como_sin_vigente(reg):
    """I4: `current_version` se tragaba el JSON roto y devolvía `None`, que `promote` grababa como «no había vigente»."""
    reg.promote("text_encoder", "v1", by="Ana")
    reg.layout.tag_path("text_encoder", "current").write_text('{"name": "text_encoder", "vers')
    with pytest.raises(ModelRegistryError, match="unreadable"):
        reg.current_version("text_encoder")
    with pytest.raises(ModelRegistryError):
        reg.promote("text_encoder", "v2", by="Luis")
    assert [p["version"] for p in reg.promotions()] == ["v1"]


def test_la_escritura_de_la_etiqueta_es_atomica_y_su_temporal_no_va_entre_las_etiquetas(reg, monkeypatch):
    """La etiqueta se escribe en un temporal y se mueve con `os.replace` (atómico). El temporal va en la RAÍZ del registro: quien
    enumera `tags/<nombre>/` (el core en `server.py`, el Studio) lee cada fichero de ahí y lo tomaría por una etiqueta."""
    import os

    from matrixai.registry import model_registry as mr

    movidos: list[tuple[str, str]] = []
    real = os.replace

    def espia(src, dst):
        movidos.append((str(src), str(dst)))
        return real(src, dst)

    monkeypatch.setattr(mr.os, "replace", espia)
    reg.promote("text_encoder", "v1", by="Ana")
    reg.promote("text_encoder", "v2", by="Ana")
    destino = str(reg.layout.tag_path("text_encoder", "current"))
    a_la_etiqueta = [src for src, dst in movidos if dst == destino]
    assert len(a_la_etiqueta) == 2
    assert all(os.path.dirname(src) == str(reg.path) for src in a_la_etiqueta)
    assert sorted(p.name for p in reg.layout.tag_path("text_encoder", "current").parent.iterdir()) == ["current"]
    assert not list(reg.path.glob(".*.tmp"))


def test_con_clave_activa_una_linea_sin_firma_es_false_no_none(reg):
    """I5: con una ruta de registro siempre hay clave y `promote` siempre firma; una línea sin firma es una manipulación."""
    reg.promote("text_encoder", "v1", by="Ana")
    f = reg.path / "promotions.jsonl"
    d = json.loads(f.read_text())
    d.pop("signature")
    d.pop("record_hash")
    d["promoted_by"] = "Mallory"
    f.write_text(json.dumps(d) + "\n")
    assert reg.promotions()[0]["signature_ok"] is False
    d["signature"] = ""
    f.write_text(json.dumps(d) + "\n")
    assert reg.promotions()[0]["signature_ok"] is False


def test_si_no_se_puede_escribir_la_constancia_y_no_habia_vigente_la_etiqueta_desaparece(reg):
    """M5: revertir cuando NO había vigente previa es BORRAR `current`, no dejarla apuntando a la que no se pudo anotar."""
    (reg.path / "promotions.jsonl").mkdir(parents=True)
    with pytest.raises(ModelRegistryError, match="left as it was"):
        reg.promote("text_encoder", "v1", by="Ana")
    assert reg.current_version("text_encoder") is None
    assert not reg.layout.tag_path("text_encoder", "current").exists()


# ── 2.ª reparación (re-auditoría del 11-11): N-M1 (el hash cubre `previous_version`), N-M3 («ya era» dentro del candado) ──

@pytest.mark.parametrize("campo, valor", [("previous_version", "v9"), ("previous_version", None), ("version", "v9"),
                                          ("promoted_by", "Mallory"), ("promoted_at", "2020-01-01T00:00:00+00:00"),
                                          ("note", "otra"), ("entry_hash", "sha256:" + "0" * 64)])
def test_el_hash_de_la_constancia_cubre_cada_campo_tambien_previous_version(reg, campo, valor):
    """N-M1: ninguna prueba sostenía que el hash cubriera `previous_version` (sabotaje verde: 14/14). Manipular la anterior es
    reescribir la historia de qué se sustituyó: la firma tiene que dejar de cuadrar, con el resto de la línea intacto."""
    reg.promote("text_encoder", "v1", by="Ana")
    reg.promote("text_encoder", "v2", by="Luis")
    f = reg.path / "promotions.jsonl"
    lineas = f.read_text().splitlines()
    d = json.loads(lineas[1])
    assert d[campo] != valor
    d[campo] = valor
    f.write_text(lineas[0] + "\n" + json.dumps(d) + "\n")
    assert [p["signature_ok"] for p in reg.promotions()] == [True, False]


def test_promover_la_que_ya_es_vigente_se_decide_dentro_del_candado_una_sola_constancia_con_hilos(reg4):
    """N-M3: «ya era la vigente» se miraba FUERA del candado (en el backend): 8 hilos promoviendo `v2` a la vez dejaban 2
    constancias, la segunda con `previous_version: v2`. Con `ya_vigente_sin_cambios` lo decide `promote` con el candado tomado:
    una sola constancia (de `v1` a `v2`) y las demás contestan `already`."""
    import threading

    reg4.promote("text_encoder", "v1", by="base")
    barrera = threading.Barrier(8)
    respuestas: list[dict] = []
    errores: list[str] = []

    def promotor(i):
        barrera.wait()
        try:
            respuestas.append(reg4.promote("text_encoder", "v2", by=f"h{i}", ya_vigente_sin_cambios=True))
        except Exception as exc:  # noqa: BLE001 — se recoge y se afirma en el principal
            errores.append(f"{type(exc).__name__}: {exc}")

    hilos = [threading.Thread(target=promotor, args=(i,)) for i in range(8)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    assert errores == []
    assert sorted(r["already"] for r in respuestas) == [False] + [True] * 7
    proms = reg4.promotions("text_encoder")
    assert [(p["version"], p["previous_version"]) for p in proms] == [("v1", None), ("v2", "v1")]
    assert all("already" not in p for p in proms)   # la clave no se escribe en la línea
    # sin la opción, re-promover la vigente SÍ anota (lo de siempre, para quien lo pida)
    reg4.promote("text_encoder", "v2", by="otra vez")
    assert [p["previous_version"] for p in reg4.promotions("text_encoder")][-1] == "v2"
