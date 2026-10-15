"""131 · C1 — `--semilla N` de `referencia_120_r0.py` (contrato 131, entrada «EL C1, SU REGLA COMPLETA»).

Nada arranca un contenedor ni un estudio: el servidor es un doble que anota el cuerpo que recibe. Cada prueba lleva el
nombre de lo que protege (un sabotaje de ese punto tiene que ponerla en rojo).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"


def _cargar(nombre: str, fichero: str):
    spec = importlib.util.spec_from_file_location(nombre, _FASE0 / fichero)
    mod = importlib.util.module_from_spec(spec)
    antes = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = antes
    return mod


G = _cargar("referencia_120_r0_c1_131", "referencia_120_r0.py")
CONJ = {"data_id": 1, "tarea": "binary_classification", "objetivo": "y", "clase_positiva": "1"}
COLS = ["a", "b", "y"]
TEXTO = "a,b,y\n1,2,1\n3,4,0\n"


def _cuerpo_de_hoy(c, cols, texto):
    """El cuerpo EXACTO que armaba `estudiar` antes del 131 (copiado tal cual): el patrón-oro."""
    cuerpo = {"csv_text": texto, "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    return cuerpo


@pytest.fixture
def servidor(monkeypatch, tmp_path):
    """`estudiar` entero contra un servidor de mentira: anota el cuerpo y entrega un estado completado."""
    recibidos = []

    def pedir_o_rechazo(base, ruta, cuerpo, plazo=300):
        recibidos.append(cuerpo)
        return 200, {"job_id": "j1"}

    monkeypatch.setattr(G, "csv_de", lambda gen, data_id: (TEXTO.encode(), COLS, [], 2))
    monkeypatch.setattr(G, "pedir_o_rechazo", pedir_o_rechazo)
    monkeypatch.setattr(G, "pedir", lambda base, ruta, cuerpo=None, plazo=120: {"estado": "completed"})
    monkeypatch.setattr(G, "ESTADOS", tmp_path)
    monkeypatch.setattr(G, "SEMILLA", None)
    return recibidos


def test_sin_la_opcion_el_cuerpo_es_byte_a_byte_el_de_hoy(servidor):
    G.estudiar("http://x", "pc1", CONJ, None)
    assert json.dumps(servidor[0]).encode() == json.dumps(_cuerpo_de_hoy(CONJ, COLS, TEXTO)).encode()
    assert "seed" not in servidor[0]


def test_sin_la_opcion_el_cuerpo_de_regresion_tampoco_lleva_seed():
    c = {"data_id": 2, "tarea": "regression", "objetivo": "y"}
    assert G.cuerpo_del_estudio(c, COLS, TEXTO) == _cuerpo_de_hoy(c, COLS, TEXTO)
    assert json.dumps(G.cuerpo_del_estudio(c, COLS, TEXTO)) == json.dumps(_cuerpo_de_hoy(c, COLS, TEXTO))


def test_con_la_opcion_el_cuerpo_lleva_la_semilla_y_nada_mas_cambia(servidor, monkeypatch):
    monkeypatch.setattr(G, "SEMILLA", 3)
    G.estudiar("http://x", "pc1", CONJ, None)
    esperado = dict(_cuerpo_de_hoy(CONJ, COLS, TEXTO), seed=3)
    assert servidor[0] == esperado and servidor[0]["seed"] == 3


def test_la_semilla_cero_se_manda_aunque_sea_la_de_siempre(servidor, monkeypatch):
    """`--semilla 0` NO es «sin la opción»: manda `seed: 0` (el mismo reparto, otro cuerpo)."""
    monkeypatch.setattr(G, "SEMILLA", 0)
    G.estudiar("http://x", "pc1", CONJ, None)
    assert servidor[0]["seed"] == 0


@pytest.mark.parametrize("semilla", [None, 0, 4])
def test_la_semilla_va_en_el_registro(servidor, monkeypatch, semilla):
    monkeypatch.setattr(G, "SEMILLA", semilla)
    rec = G.estudiar("http://x", "pc1", CONJ, None)
    assert "semilla" in rec and rec["semilla"] == semilla


def test_la_semilla_va_tambien_en_el_registro_de_un_rechazo(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "csv_de", lambda gen, d: (TEXTO.encode(), COLS, [], 2))
    monkeypatch.setattr(G, "pedir_o_rechazo", lambda *a, **k: (422, {"texto": "no"}))
    monkeypatch.setattr(G, "SEMILLA", 2)
    assert G.estudiar("http://x", "pc1", CONJ, None)["semilla"] == 2


# ---------------------------------------------------------------- el punto de control
def _doc(semilla, *, registro_semilla="igual", estado="completed", corte=None):
    return {"procedencia": {"imagen_id": "I", "sha256_guion": "G", "politica": None, "modo": "pcpu", "semilla": semilla},
            "corte": corte,
            "conjuntos": {"pc1": {"estado": estado, "semilla": semilla if registro_semilla == "igual" else registro_semilla}}}


def test_el_punto_de_control_reutiliza_con_la_misma_semilla():
    assert set(G.reutilizables(_doc(2), "I", "G", None, "pcpu", 2)) == {"pc1"}
    assert set(G.reutilizables(_doc(None), "I", "G", None, "pcpu", None)) == {"pc1"}


def test_el_punto_de_control_no_reutiliza_un_registro_de_otra_semilla():
    assert G.reutilizables(_doc(1), "I", "G", None, "pcpu", 2) == {}
    assert G.reutilizables(_doc(0), "I", "G", None, "pcpu", None) == {}     # --semilla 0 y «sin opción» no se mezclan
    assert G.reutilizables(_doc(None), "I", "G", None, "pcpu", 0) == {}


def test_el_punto_de_control_no_reutiliza_un_registro_cuya_semilla_no_es_la_de_la_procedencia():
    """La procedencia dice 2 pero ese registro lo midió otra semilla (un fichero editado a mano o mezclado)."""
    assert G.reutilizables(_doc(2, registro_semilla=1), "I", "G", None, "pcpu", 2) == {}
    assert G.reutilizables(_doc(2, registro_semilla=None), "I", "G", None, "pcpu", 2) == {}


def test_el_punto_de_control_no_reutiliza_si_la_procedencia_es_de_otra_semilla_aunque_el_registro_coincida():
    """Un fichero a medias entre dos semillas: la procedencia dice 1 y el registro dice 2. Pidiendo 2, no se reusa."""
    assert G.reutilizables(_doc(1, registro_semilla=2), "I", "G", None, "pcpu", 2) == {}


def test_un_punto_de_control_de_antes_del_131_no_se_reutiliza_con_semilla():
    antes = _doc(None)
    del antes["procedencia"]["semilla"]
    del antes["conjuntos"]["pc1"]["semilla"]
    assert G.reutilizables(antes, "I", "G", None, "pcpu", 3) == {}
    assert set(G.reutilizables(antes, "I", "G", None, "pcpu", None)) == {"pc1"}     # y sin opción, como siempre


def test_lo_demas_de_la_condicion_de_reuso_sigue_igual():
    assert G.reutilizables(_doc(1), "OTRA", "G", None, "pcpu", 1) == {}
    assert G.reutilizables(_doc(1), "I", "OTRO", None, "pcpu", 1) == {}
    assert G.reutilizables(_doc(1), "I", "G", "l2_0", "pcpu", 1) == {}
    assert G.reutilizables(_doc(1), "I", "G", None, "rcpu", 1) == {}
    assert G.reutilizables(_doc(1, estado="failed"), "I", "G", None, "pcpu", 1) == {}
    assert G.reutilizables(_doc(1, corte={"tipo": "regla_9"}), "I", "G", None, "pcpu", 1) == {}


def test_una_salida_con_medidas_de_otra_semilla_no_se_pisa_en_silencio():
    assert G.semilla_ajena(_doc(1), 1) is None and G.semilla_ajena(_doc(None), None) is None
    assert "semilla" in G.semilla_ajena(_doc(1), 2) and G.semilla_ajena(_doc(None), 1)
    assert G.semilla_ajena(_doc(0), None)


def test_main_lee_la_semilla_de_la_linea_de_ordenes_y_la_pasa_a_la_condicion_de_reuso():
    """Cableado: `--semilla` llega a SEMILLA, a la procedencia y a `reutilizables`/`semilla_ajena` (el hueco de siempre
    está en el cableado, no en la función)."""
    fuente = (_FASE0 / "referencia_120_r0.py").read_text()
    assert 'ap.add_argument("--semilla", type=int' in fuente and "SEMILLA = a.semilla" in fuente
    assert '"modo": MODO, "semilla": SEMILLA' in fuente
    assert "reutilizables(p, imagen, guion, POLITICA, MODO, SEMILLA)" in fuente
    assert "semilla_ajena(p, SEMILLA)" in fuente
    assert "cuerpo_del_estudio(c, cols, texto, SEMILLA)" in fuente


def test_con_otra_semilla_el_control_de_arboles_no_para_y_con_la_cero_si(monkeypatch, tmp_path):
    """Lo que hace `main` con la semilla: ≠ 0 → el control se calcula pero no para (motivo escrito); 0 o nada → como siempre."""
    visto = {}

    def falso_arrancar(politica=None):
        raise SystemExit("PARO de la prueba: no se arranca ningún contenedor")

    monkeypatch.setattr(G, "arrancar_contenedor", falso_arrancar)
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "sha256:x", "returncode": 0})())
    monkeypatch.setattr(G, "cargar_generador", lambda: None)
    monkeypatch.setattr(G, "sha", lambda p: "x")        # el generador y el ejemplo viven en los repos hermanos
    monkeypatch.setattr(G, "comprobar_bytes", lambda g, c: {})
    for semilla, esperado in ((None, None), (0, None), (1, "semilla 1"), (4, "semilla 4")):
        argv = ["x", "--solo-humo", "--pcpu", "--salida", str(tmp_path / f"s{semilla}.json")]
        if semilla is not None:
            argv += ["--semilla", str(semilla)]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            G.main()
        visto[semilla] = G.SIN_CONTROL_DE_ARBOLES
        assert G.SEMILLA == semilla
    assert visto[None] is None and visto[0] is None
    assert visto[1].startswith("semilla 1") and visto[4].startswith("semilla 4")
