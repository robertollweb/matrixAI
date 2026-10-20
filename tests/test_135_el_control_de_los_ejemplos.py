"""135 §2.3 (punto 11) — la guarda de los controles de las medidas (`comprobar_bytes` de `referencia_120_r0.py`, que usa
también `generar_kit_128_c5.py`) y la prueba del kit 128 preguntan lo MISMO, con UNA función (`comparar_con_el_ejemplo`):
¿es el ejemplo del Studio, byte a byte, el CSV de la Fase 0 sin exactamente las columnas que su catálogo declara quitadas y
que la Fase 0 tenía (`columnas_quitadas` con `en_fase0`)? Cualquier otra diferencia PARA.

Primero con datos pequeños con las formas de los de verdad (una coma y unas comillas dentro de un valor, valores vacíos,
tildes), que corren en cualquier clon; después con los CSV REALES del Studio de al lado y los ARFF de la Fase 0, que se
saltan si no están. Cada prueba de «PARA» es un sabotaje de un solo punto: tiene que dar rojo si la guarda deja de mirarlo.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import types
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


R0 = _cargar("referencia_120_r0_135_control", "referencia_120_r0.py")
GK = _cargar("generar_kit_128_c5_135_control", "generar_kit_128_c5.py")


def _csv(filas) -> bytes:
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\n").writerows(filas)
    return buf.getvalue().encode("utf-8")


def _filas(contenido: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(contenido.decode("utf-8"), newline="")))


def _sin(filas, columnas):
    fuera = {filas[0].index(c) for c in columnas}
    return [[v for i, v in enumerate(f) if i not in fuera] for f in filas]


# ------------------------------------------------------------------------------------------------ datos pequeños
CABECERA = ["Study", "Run", "a", "texto", "y"]
FILAS = [CABECERA,
         ["1", "1", "0.5", "Señal, con coma", ""],
         ["1", "2", "", 'una "cita"', "1"],
         ["2", "1", "0.25", "año", "0"]]
FASE0 = _csv(FILAS)
STUDIO = _csv(_sin(FILAS, ["Study", "Run"]))


def _ficha(**cambios):
    quitadas = [{"columna": "Study", "por": "ignore_attribute", "en_fase0": True},
                {"columna": "Run", "por": "ignore_attribute", "en_fase0": True},
                {"columna": "communityname", "por": "string", "en_fase0": False}]   # STRING: no está en la Fase 0
    return {"nombre": "x", "columnas_quitadas": cambios.get("quitadas", quitadas)}


def test_el_ejemplo_es_el_CSV_de_la_Fase_0_sin_lo_que_el_catalogo_declara_y_dice_que_lo_desconto():
    assert b'"Se\xc3\xb1al, con coma"' in FASE0 and b'"una ""cita"""' in FASE0      # las formas que se quieren probar
    r = R0.comparar_con_el_ejemplo(FASE0, STUDIO, _ficha())
    assert r == {"es_el_mismo": True, "columnas_descontadas": ["Study", "Run"], "motivo": None,
                 "sha256_fase0": hashlib.sha256(FASE0).hexdigest(), "bytes_fase0": len(FASE0),
                 "sha256_studio": hashlib.sha256(STUDIO).hexdigest(), "bytes_studio": len(STUDIO)}


def test_lo_descontado_va_en_el_orden_de_la_cabecera_de_la_Fase_0_no_en_el_del_catalogo():
    ficha = _ficha(quitadas=[{"columna": "Run", "en_fase0": True}, {"columna": "Study", "en_fase0": True}])
    assert R0.comparar_con_el_ejemplo(FASE0, STUDIO, ficha)["columnas_descontadas"] == ["Study", "Run"]


def _con(filas_studio):
    return _csv(filas_studio)


SABOTAJES_DEL_STUDIO = {
    "una_celda_cambiada": lambda f: [f[0], f[1], [f[2][0], f[2][1], "0"], f[3]],
    "una_fila_de_menos": lambda f: f[:-1],
    "una_fila_de_mas": lambda f: f + [f[-1]],
    "una_columna_quitada_que_el_catalogo_no_declara": lambda f: [[r[1], r[2]] for r in f],
    "dos_columnas_en_otro_orden": lambda f: [[r[1], r[0], r[2]] for r in f],
    "una_columna_declarada_que_el_Studio_no_quito": lambda f: [[a, *r] for a, r in zip(["Run", "1", "2", "1"], f)],
}


@pytest.mark.parametrize("sabotaje", sorted(SABOTAJES_DEL_STUDIO))
def test_cualquier_otra_diferencia_del_Studio_no_es_el_mismo(sabotaje):
    studio = _con(SABOTAJES_DEL_STUDIO[sabotaje](_filas(STUDIO)))
    assert studio != STUDIO
    r = R0.comparar_con_el_ejemplo(FASE0, studio, _ficha())
    assert r["es_el_mismo"] is False and r["motivo"] and r["columnas_descontadas"] == [], r


def test_otro_fin_de_linea_no_es_el_mismo():
    r = R0.comparar_con_el_ejemplo(FASE0, STUDIO.replace(b"\n", b"\r\n"), _ficha())
    assert r["es_el_mismo"] is False and "otros bytes" in r["motivo"]


# R-3 (re-auditoría del 20-10): dos detalles del «byte a byte» que ninguna prueba fijaba. Sin nada que descontar se comparan los bytes
# TAL CUAL (un CRLF o un salto de línea de más o de menos NO es el mismo CSV), y al descontar también (el salto final cuenta).
_FINES = {"CRLF": lambda b: b.replace(b"\n", b"\r\n"), "salto_final_de_mas": lambda b: b + b"\n",
          "sin_salto_final": lambda b: b[:-1]}


@pytest.mark.parametrize("fin", sorted(_FINES))
def test_R3_sin_nada_que_descontar_un_fin_de_linea_distinto_no_es_el_mismo_CSV(fin):
    for ficha in (None, _ficha(quitadas=[])):
        otro = _FINES[fin](FASE0)
        assert otro != FASE0
        r = R0.comparar_con_el_ejemplo(FASE0, otro, ficha)
        assert r["es_el_mismo"] is False and r["motivo"], (fin, r)


@pytest.mark.parametrize("fin", sorted(_FINES))
def test_R3_al_descontar_el_salto_de_linea_del_final_cuenta(fin):
    otro = _FINES[fin](STUDIO)
    assert otro != STUDIO
    r = R0.comparar_con_el_ejemplo(FASE0, otro, _ficha())
    assert r["es_el_mismo"] is False and r["motivo"] and r["columnas_descontadas"] == [], (fin, r)


SABOTAJES_DE_LA_FICHA = {
    "una_declarada_con_en_fase0_falso": [{"columna": "Study", "en_fase0": False}, {"columna": "Run", "en_fase0": True}],
    "una_sin_declarar": [{"columna": "Study", "en_fase0": True}],
    "en_fase0_que_no_es_verdadero_sino_texto": [{"columna": "Study", "en_fase0": "true"}, {"columna": "Run", "en_fase0": True}],
    "una_que_la_Fase_0_no_tenia": [{"columna": "Study", "en_fase0": True}, {"columna": "Run", "en_fase0": True},
                                   {"columna": "fold", "en_fase0": True}],
    "la_misma_dos_veces": [{"columna": "Study", "en_fase0": True}, {"columna": "Run", "en_fase0": True},
                           {"columna": "Run", "en_fase0": True}],
}


@pytest.mark.parametrize("sabotaje", sorted(SABOTAJES_DE_LA_FICHA))
def test_una_ficha_que_no_dice_exactamente_lo_quitado_no_es_el_mismo(sabotaje):
    r = R0.comparar_con_el_ejemplo(FASE0, STUDIO, _ficha(quitadas=SABOTAJES_DE_LA_FICHA[sabotaje]))
    assert r["es_el_mismo"] is False and r["motivo"] and r["columnas_descontadas"] == [], r


@pytest.mark.parametrize("ficha", [None, {"nombre": "x"}, {"nombre": "x", "columnas_quitadas": []},
                                   {"nombre": "x", "columnas_quitadas": [{"columna": "communityname", "en_fase0": False}]}])
def test_sin_nada_que_descontar_tiene_que_ser_el_mismo_CSV_como_antes_del_135(ficha):
    """Un Studio de antes del 135 (ficha sin `columnas_quitadas`), sin catálogo, o con solo STRING quitadas."""
    assert R0.comparar_con_el_ejemplo(FASE0, FASE0, ficha)["es_el_mismo"] is True
    assert R0.comparar_con_el_ejemplo(FASE0, FASE0, ficha)["columnas_descontadas"] == []
    r = R0.comparar_con_el_ejemplo(FASE0, STUDIO, ficha)
    assert r["es_el_mismo"] is False and "le faltan ['Study', 'Run']" in r["motivo"]


def test_sin_nada_que_descontar_se_comparan_los_bytes_tal_cual_aunque_no_se_sepan_reescribir():
    rara = b'"Study","Run","a","texto","y"\n"1","1","0.5","x",""\n'          # comillas que csv.writer no pondría
    assert R0.comparar_con_el_ejemplo(rara, rara, None)["es_el_mismo"] is True


def test_si_el_CSV_de_la_Fase_0_no_sale_igual_al_reescribirlo_no_se_le_quitan_columnas():
    rara = b'"Study","Run","a","texto","y"\n"1","1","0.5","x",""\n'
    studio = _csv(_sin(_filas(rara), ["Study", "Run"]))
    r = R0.comparar_con_el_ejemplo(rara, studio, _ficha())
    assert r["es_el_mismo"] is False and "reescribirlo" in r["motivo"]


def test_la_ficha_se_lee_del_catalogo_del_Studio_y_sin_catalogo_no_hay_ficha(tmp_path):
    assert R0.ficha_del_ejemplo(tmp_path, "x") is None
    (tmp_path / "catalogo.json").write_text(json.dumps({"conjuntos": [{"nombre": "otro"}, _ficha()]}))
    assert R0.ficha_del_ejemplo(tmp_path, "x") == _ficha()
    assert R0.ficha_del_ejemplo(tmp_path, "no-esta") is None


# ------------------------------------------------------------------------------------------------ la guarda, con un doble del generador
def _gen_doble(filas):
    return types.SimpleNamespace(ARFF_DIR=Path("/no-se-lee"),
                                 leer_arff=lambda ruta: (list(filas[0]), [], [list(f) for f in filas[1:]]))


def _studio_doble(tmp_path, contenido, ficha):
    (tmp_path / "x.csv").write_bytes(contenido)
    (tmp_path / "catalogo.json").write_text(json.dumps({"conjuntos": [ficha]}))
    return tmp_path


def test_la_guarda_no_para_y_escribe_lo_que_comparo_y_lo_que_desconto(monkeypatch, tmp_path):
    monkeypatch.setattr(R0, "EJEMPLOS", _studio_doble(tmp_path, STUDIO, _ficha()))
    res = R0.comprobar_bytes(_gen_doble(FILAS), [{"nombre": "x", "data_id": 1}])
    assert res == {"x": {"byte_a_byte": True, "sha256": hashlib.sha256(FASE0).hexdigest(), "bytes": len(FASE0),
                         "columnas_descontadas": ["Study", "Run"], "csv_del_studio": str(tmp_path / "x.csv"),
                         "sha256_del_studio": hashlib.sha256(STUDIO).hexdigest(), "bytes_del_studio": len(STUDIO)}}


@pytest.mark.parametrize("sabotaje", sorted(SABOTAJES_DEL_STUDIO))
def test_la_guarda_PARA_ante_cualquier_otra_diferencia_del_Studio(monkeypatch, tmp_path, sabotaje):
    monkeypatch.setattr(R0, "EJEMPLOS", _studio_doble(tmp_path, _con(SABOTAJES_DEL_STUDIO[sabotaje](_filas(STUDIO))), _ficha()))
    with pytest.raises(SystemExit, match=r"^PARO: el CSV de x del Studio NO es"):
        R0.comprobar_bytes(_gen_doble(FILAS), [{"nombre": "x", "data_id": 1}])


@pytest.mark.parametrize("sabotaje", sorted(SABOTAJES_DE_LA_FICHA))
def test_la_guarda_PARA_si_el_catalogo_no_dice_exactamente_lo_quitado(monkeypatch, tmp_path, sabotaje):
    ficha = _ficha(quitadas=SABOTAJES_DE_LA_FICHA[sabotaje])
    monkeypatch.setattr(R0, "EJEMPLOS", _studio_doble(tmp_path, STUDIO, ficha))
    with pytest.raises(SystemExit, match=r"^PARO: el CSV de x del Studio NO es"):
        R0.comprobar_bytes(_gen_doble(FILAS), [{"nombre": "x", "data_id": 1}])


def test_R4_la_guarda_PARA_si_falta_el_CSV_de_un_ejemplo_del_control(monkeypatch, tmp_path):
    """Un ejemplo de `CONTROL` que el Studio no trae NO se salta en silencio: el kit habría escrito «control: [los 4]» sin comparar uno."""
    monkeypatch.setattr(R0, "EJEMPLOS", tmp_path)                       # ni CSV ni catálogo
    monkeypatch.setattr(R0, "CONTROL", {**R0.CONTROL, "x": {}})
    with pytest.raises(SystemExit, match=r"^PARO: .*falta el CSV del ejemplo del control x"):
        R0.comprobar_bytes(_gen_doble(FILAS), [{"nombre": "x", "data_id": 1}])


def test_R4_un_conjunto_que_no_es_del_control_y_no_esta_en_el_Studio_se_sigue_saltando(monkeypatch, tmp_path):
    monkeypatch.setattr(R0, "EJEMPLOS", tmp_path)
    assert R0.comprobar_bytes(_gen_doble(FILAS), [{"nombre": "x", "data_id": 1}]) == {}


def test_la_guarda_sin_catalogo_exige_el_mismo_CSV(monkeypatch, tmp_path):
    (tmp_path / "x.csv").write_bytes(STUDIO)
    monkeypatch.setattr(R0, "EJEMPLOS", tmp_path)
    with pytest.raises(SystemExit, match="le faltan"):
        R0.comprobar_bytes(_gen_doble(FILAS), [{"nombre": "x", "data_id": 1}])


# ------------------------------------------------------------------------------------------------ el manifiesto del kit 128
def test_sin_nada_descontado_el_manifiesto_del_kit_no_cambia_ni_un_byte():
    """Su sha256 entra en la `huella` de los brazos del kit: con un Studio de antes del 135 tiene que salir el de git."""
    ruta = _FASE0 / "kit_128_c5" / "datos" / "manifiesto.json"
    hoy = json.loads(ruta.read_text())
    control = {n: {"columnas_descontadas": []} for n in hoy["control"]}
    compuesto = GK.componer_manifiesto(control, set(hoy["control"]), hoy["conjuntos"])
    assert json.dumps(compuesto, ensure_ascii=False, indent=1) + "\n" == ruta.read_text()


def test_con_algo_descontado_el_manifiesto_ya_no_dice_byte_a_byte_y_dice_que_se_desconto():
    control = {"pc1": {"columnas_descontadas": []}, "climate": {"columnas_descontadas": ["Study", "Run"]}}
    m = GK.componer_manifiesto(control, {"pc1", "climate"}, [])
    assert "control_byte_a_byte_con_los_del_studio" not in m
    assert m["control_contra_los_del_studio"]["columnas_descontadas"] == {"climate": ["Study", "Run"], "pc1": []}


def test_el_generador_del_kit_escribe_en_el_manifiesto_lo_que_devolvio_la_guarda(monkeypatch, tmp_path):
    """El cableado: `main` pasa al manifiesto lo que devuelve `comprobar_bytes`, no una constante."""
    banco = {"conjuntos": {"banco": [{"orden": 1, "nombre": "x", "cubo": "pequeno", "data_id": 1,
                                      "tarea": "regression", "objetivo": "y", "clase_positiva": None}]}}
    (tmp_path / "borrador.json").write_text(json.dumps(banco))
    (tmp_path / "sonda.json").write_text(json.dumps({"datasets_declarados": [
        {"nombre": "x", "data_id": 1, "tarea": "regression", "objetivo": "y", "clase_positiva": None}]}))
    r0 = types.SimpleNamespace(BORRADOR=tmp_path / "borrador.json", SONDA=tmp_path / "sonda.json", CONTROL={"x": {}},
                               cargar_generador=lambda: None, csv_de=lambda gen, d: (FASE0, CABECERA, [], 3),
                               comprobar_bytes=lambda gen, cs: {"x": {"columnas_descontadas": ["Study", "Run"]}})
    monkeypatch.setattr(GK, "_cargar_r0", lambda: r0)
    GK.main(["--destino", str(tmp_path / "datos")])
    m = json.loads((tmp_path / "datos" / "manifiesto.json").read_text())
    assert m["control_contra_los_del_studio"]["columnas_descontadas"] == {"x": ["Study", "Run"]}


# ------------------------------------------------------------------------------------------------ con los datos de verdad
_EJEMPLOS = R0.EJEMPLOS


def _necesita_los_datos_reales():
    if not (_EJEMPLOS / "catalogo.json").is_file() or not R0.GENERADOR.is_file():
        pytest.skip("no está el Studio al lado")


def _generador_real(monkeypatch):
    _necesita_los_datos_reales()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)        # no dejar __pycache__ en el Studio de al lado
    gen = R0.cargar_generador()
    if not all((gen.ARFF_DIR / f"{b['data_id']}.arff").is_file() for b in _control_del_banco()):
        pytest.skip(f"no están los ARFF de la Fase 0 ({gen.ARFF_DIR})")
    return gen


def _control_del_banco():
    banco = json.loads(R0.BORRADOR.read_text())["conjuntos"]["banco"]
    return [b for b in banco if b["nombre"] in R0.CONTROL]


def test_la_guarda_real_no_para_con_el_Studio_de_al_lado_y_descuenta_lo_que_su_catalogo_declara(monkeypatch):
    gen = _generador_real(monkeypatch)
    res = R0.comprobar_bytes(gen, _control_del_banco())
    assert sorted(res) == sorted(R0.CONTROL)
    for n, r in res.items():
        ficha = R0.ficha_del_ejemplo(_EJEMPLOS, n) or {}
        declaradas = [q["columna"] for q in ficha.get("columnas_quitadas", []) if q.get("en_fase0") is True]
        assert r["byte_a_byte"] is True and sorted(r["columnas_descontadas"]) == sorted(declaradas), n
        assert r["sha256_del_studio"] == hashlib.sha256((_EJEMPLOS / f"{n}.csv").read_bytes()).hexdigest()


def _copia_del_studio(tmp_path):
    copia = tmp_path / "ejemplos_medidos"
    copia.mkdir()
    for f in [_EJEMPLOS / "catalogo.json", *(_EJEMPLOS / f"{n}.csv" for n in R0.CONTROL)]:
        shutil.copy2(f, copia / f.name)
    return copia


CLIMATE = "climate-model-simulation-crashes"


def _reescribir(ruta, cambio):
    ruta.write_bytes(_csv(cambio(_filas(ruta.read_bytes()))))


def _celda(f):
    f[5][3] = f[5][3] + "1"
    return f


SABOTAJES_REALES = {
    "una_celda_cambiada": lambda copia: _reescribir(copia / f"{CLIMATE}.csv", _celda),
    "una_fila_de_menos": lambda copia: _reescribir(copia / f"{CLIMATE}.csv", lambda f: f[:-1]),
    "una_columna_quitada_que_el_catalogo_no_declara": lambda copia: _reescribir(
        copia / f"{CLIMATE}.csv", lambda f: _sin(f, ["vconst_corr"])),
    "dos_columnas_en_otro_orden": lambda copia: _reescribir(
        copia / f"{CLIMATE}.csv", lambda f: [[r[1], r[0], *r[2:]] for r in f]),
}


def _catalogo(copia, cambio):
    ruta = copia / "catalogo.json"
    cat = json.loads(ruta.read_text())
    cambio({d["nombre"]: d for d in cat["conjuntos"]})
    ruta.write_text(json.dumps(cat, ensure_ascii=False))


def _una_de_UCI_con_en_fase0_falso(fichas):
    q = [q for q in fichas[CLIMATE].get("columnas_quitadas", []) if q.get("en_fase0") is True]
    if not q:
        pytest.skip("el Studio de al lado es de antes del 135: su climate no quita columnas de la Fase 0")
    q[0]["en_fase0"] = False


def _la_string_de_us_crime_como_si_la_Fase_0_la_tuviera(fichas):
    q = [q for q in fichas["us_crime"].get("columnas_quitadas", []) if q.get("en_fase0") is False]
    if not q:
        pytest.skip("el Studio de al lado es de antes del 135: su us_crime no declara columnas quitadas")
    q[0]["en_fase0"] = True


SABOTAJES_REALES["una_declarada_con_en_fase0_falso"] = lambda copia: _catalogo(copia, _una_de_UCI_con_en_fase0_falso)
SABOTAJES_REALES["una_string_declarada_como_de_la_Fase_0"] = lambda copia: _catalogo(
    copia, _la_string_de_us_crime_como_si_la_Fase_0_la_tuviera)


@pytest.mark.parametrize("sabotaje", sorted(SABOTAJES_REALES))
def test_la_guarda_real_PARA_con_una_copia_saboteada_del_Studio(monkeypatch, tmp_path, sabotaje):
    gen = _generador_real(monkeypatch)
    copia = _copia_del_studio(tmp_path)
    monkeypatch.setattr(R0, "EJEMPLOS", copia)
    R0.comprobar_bytes(gen, _control_del_banco())                   # la copia sin tocar: no para
    SABOTAJES_REALES[sabotaje](copia)
    with pytest.raises(SystemExit, match=r"^PARO: el CSV de (climate-model-simulation-crashes|us_crime) del Studio NO es"):
        R0.comprobar_bytes(gen, _control_del_banco())


def test_R5_el_generador_del_kit_dice_que_regenerarlo_con_un_Studio_que_quita_columnas_cambia_su_identidad():
    """R-5: la identidad del kit (su manifiesto, que entra en la huella del brazo) depende del Studio con que se genera; dicho donde se genera."""
    doc = GK.__doc__
    assert "CAMBIA SU IDENTIDAD" in doc and "control_contra_los_del_studio" in doc
    assert "huella" in doc and "INCOMPARABLES (PARO 6)" in doc
