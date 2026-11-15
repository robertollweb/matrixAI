# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""169-C1 — el registro de los 6 conjuntos de CONFIRMACION_C2.md y la negativa a medir sin sello. Sin red, sin medir, sin llamar a `main()`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parents[1]
for _p in (_RAIZ / "benchmarks" / "c169", _RAIZ / "benchmarks" / "fase0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analiza_c169 as an  # noqa: E402
import conjuntos_c169 as cj  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

V2 = _RAIZ / "benchmarks" / "fase0" / "protocolo_exploratorio_v2.json"
#: El digest del protocolo v2 FIRMADO, el que citan 119-C3 y lo que le sigue (`resultado_pasada_119_c3.json`). Si cambia, se movió lo ya medido.
DIGEST_V2 = "1b4e902e61e06edcfda8e9f4ccef1232a233d46718089c5f0a2f814feb9dc878"
SHA256_FICHERO_V2 = "40ef7614d12bc5f727779d878ad430db9eb340b79d40894e4f174125d1f85b22"
TABLA = _RAIZ / "benchmarks" / "c163" / "CONFIRMACION_C2.md"


def _tabla():
    out = {}
    for linea in TABLA.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in linea.strip().strip("|").split("|")]
        if len(c) == 9 and c[1].isdigit() and len(c[8]) == 64:
            out[int(c[1])] = (c[2].split(" (")[0], c[8], int(c[4].split(" × ")[0].replace(".", "")))
    return out


def test_el_catalogo_trae_exactamente_los_6_de_la_tabla_con_su_sha256_y_sus_filas():
    t = _tabla()
    assert set(t) == set(cj.DATA_IDS) and len(t) == 6
    for d in cj.conjuntos():
        nombre, sha, filas = t[d.data_id]
        assert (d.nombre, d.sha256_arff, d.n_filas) == (nombre, sha, filas)
        assert d.sellado is False and d.file_id is not None


def test_el_protocolo_v2_firmado_NO_cambia_ni_su_fichero_ni_su_digest_ni_sus_40():
    assert hashlib.sha256(V2.read_bytes()).hexdigest() == SHA256_FICHERO_V2
    base = protocolo_mod.ProtocoloExploratorio.cargar(V2)
    assert base.digest() == DIGEST_V2 and len(base.datasets) == 40
    registrado = json.loads((_RAIZ / "benchmarks" / "fase0" / "resultado_pasada_119_c3.json").read_text(encoding="utf-8"))
    assert registrado["protocolo_digest_sha256"] == DIGEST_V2          # el de lo YA medido es éste


def test_ninguno_de_los_6_esta_en_el_v2_y_ampliarlo_en_memoria_no_toca_la_base():
    base = protocolo_mod.ProtocoloExploratorio.cargar(V2)
    assert not ({d.data_id for d in base.datasets} & set(cj.DATA_IDS))
    amp = cj.protocolo_ampliado(base)
    assert len(amp.datasets) == 46 and len(base.datasets) == 40 and base.digest() == DIGEST_V2
    assert amp.digest() != base.digest()                               # es OTRO objeto: nunca se escribe en el v2
    # lo no tocado: mismo presupuesto por cubo (120 s / 300 s), partición y regla de cierre
    assert amp.presupuesto == base.presupuesto and amp.particion == base.particion and amp.regla_de_cierre == base.regla_de_cierre
    # los 40 de antes, fila a fila iguales
    assert [d.a_json() for d in amp.datasets[:40]] == [d.a_json() for d in base.datasets]


def test_ampliar_un_protocolo_que_ya_trae_uno_de_los_6_se_niega():
    base = protocolo_mod.ProtocoloExploratorio.cargar(V2)
    ya = protocolo_mod.ProtocoloExploratorio.cargar(V2)
    import dataclasses
    con_uno = dataclasses.replace(ya, datasets=tuple(ya.datasets) + (cj.conjuntos()[0],))
    with pytest.raises(SystemExit):
        cj.protocolo_ampliado(con_uno)
    del base


def test_la_descarga_con_sha256_que_no_cuadra_FALLA_CERRADO_y_no_escribe_nada(tmp_path):
    d = cj.conjuntos()[2]
    with pytest.raises(SystemExit, match="No se escribe nada"):
        cj.asegurar_arff(d, tmp_path, descargar=lambda url: b"esto no es el ARFF")
    assert list(tmp_path.iterdir()) == []


def test_la_descarga_correcta_se_escribe_atomica_y_se_pide_por_la_url_directa_del_file_id(tmp_path):
    d = cj.conjuntos()[2]
    contenido = b"@relation x\n"
    d2 = type(d)(**{**d.__dict__, "sha256_arff": hashlib.sha256(contenido).hexdigest()})
    vistas = []
    ruta = cj.asegurar_arff(d2, tmp_path, descargar=lambda u: vistas.append(u) or contenido)
    assert ruta.read_bytes() == contenido and vistas == [f"https://openml.org/data/v1/download/{d.file_id}/{d.nombre}.arff"]
    assert [p.name for p in tmp_path.iterdir()] == [f"{d.data_id}.arff"]


def test_un_fichero_ya_presente_con_otro_sha256_se_niega_y_no_se_pisa(tmp_path):
    d = cj.conjuntos()[0]
    (tmp_path / f"{d.data_id}.arff").write_bytes(b"otro contenido")
    with pytest.raises(SystemExit, match="No se mide ni se pisa"):
        cj.asegurar_arff(d, tmp_path, descargar=lambda u: pytest.fail("no debe descargar"))
    assert (tmp_path / f"{d.data_id}.arff").read_bytes() == b"otro contenido"


def test_los_6_ARFF_de_este_equipo_cuadran_con_su_sha256_registrado():
    import pasada_exploratoria_101_c3 as c3
    faltan = [d.nombre for d in cj.conjuntos() if not (c3.ARFF_DIR / f"{d.data_id}.arff").exists()]
    if faltan:
        pytest.skip(f"no bajados todavía: {faltan} (`medir_c169` los baja y verifica)")
    for d in cj.conjuntos():
        assert cj.sha256_de(c3.ARFF_DIR / f"{d.data_id}.arff") == d.sha256_arff


# ---------------------------------------------------------------- la negativa a medir sin sello
def test_negativa_el_medidor_se_niega_a_medir_sin_sello(monkeypatch):
    import medir_c169 as m
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    with pytest.raises(SystemExit, match="NO está sellada"):
        m.comprobar_el_sello(False, "ab" * 32)


def test_negativa_el_medidor_se_niega_a_medir_si_REGLA_md_ya_no_es_la_sellada(monkeypatch):
    import medir_c169 as m
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "ab" * 32)
    with pytest.raises(SystemExit, match="no es la sellada"):
        m.comprobar_el_sello(False, "cd" * 32)


def test_el_medidor_con_la_regla_sellada_y_cuadrando_pasa_y_una_sonda_se_salta_el_sello(monkeypatch):
    import medir_c169 as m
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", "ab" * 32)
    m.comprobar_el_sello(False, "ab" * 32)
    monkeypatch.setattr(an, "REGLA_SHA256_SELLADA", None)
    m.comprobar_el_sello(True, None)


def test_el_medidor_solo_mide_los_dos_brazos_de_hgb_y_sus_piezas_son_las_del_163():
    import medir_c169 as m
    import medir_c163_c2 as m163
    brazos = m.motores()
    assert list(brazos) == ["hgb_hoy", "hgb_rejilla"]
    assert brazos["hgb_hoy"].rejilla_por_validacion is False and brazos["hgb_rejilla"].rejilla_por_validacion is True
    assert m.m163 is m163 and m.an163.problemas_de_la_declaracion is an.an163.problemas_de_la_declaracion


# ---------------------------------------------------------------- los `ignore_attribute` de OpenML no entran como predictores
def test_ignore_attribute_el_catalogo_fija_cooling_load_para_energy_efficiency_y_nada_para_los_otros_5():
    ign = cj.ignorados_registrados()
    assert ign[44960] == ["cooling_load"] and all(v == [] for k, v in ign.items() if k != 44960)
    assert "openml.org/api/v1/json/data" in json.loads(cj.RUTA_DEL_CATALOGO.read_text(encoding="utf-8"))["ignore_attribute"]["origen"]


def test_ignore_attribute_quitar_ignorados_quita_la_columna_de_todas_las_filas_y_la_declara():
    filas = [{"row_id": "a", "x": 1.0, "cooling_load": 5.0, "y": 2.0}, {"row_id": "b", "x": 2.0, "cooling_load": 6.0, "y": 3.0}]
    assert cj.quitar_ignorados(filas, "y", ["cooling_load"]) == ["cooling_load"]
    assert all("cooling_load" not in f for f in filas) and all(f["x"] and f["y"] for f in filas)


def test_ignore_attribute_una_exclusion_que_no_se_puede_aplicar_PARA():
    with pytest.raises(SystemExit, match="no está en el fichero"):
        cj.quitar_ignorados([{"row_id": "a", "y": 1.0}], "y", ["cooling_load"])
    with pytest.raises(SystemExit, match="es el objetivo"):
        cj.quitar_ignorados([{"row_id": "a", "y": 1.0}], "y", ["y"])


def test_ignore_attribute_el_camino_real_de_energy_efficiency_no_trae_cooling_load_y_lo_declara(monkeypatch):
    import pasada_exploratoria_101_c3 as c3
    if not (c3.ARFF_DIR / "44960.arff").exists():
        pytest.skip("ARFF no bajado")
    import pasada_amplia_101_c5 as c5
    import medir_c169  # noqa: F401  (importarlo no cambia nada: el cambio de `c5.cargar_arff` lo hace main)
    # el catálogo de lectura con SOLO los 6 (monkeypatch: no se deja nada global)
    monkeypatch.setattr(c3, "_CATALOGO_POR_DATA_ID", {d.data_id: d.a_json() for d in cj.conjuntos()})
    monkeypatch.setattr(c3, "_RUTA_DEL_CATALOGO", c3.RUTA_DEL_PROTOCOLO)
    filas, objetivo = cj.cargar_arff_sin_ignorados(44960)
    assert objetivo == "heating_load" and "cooling_load" not in filas[0] and "heating_load" in filas[0]
    assert c3.LECTURA_DECLARADA["44960"]["columnas_ignoradas_por_openml"] == ["cooling_load"]
    filas2, _ = cj.cargar_arff_sin_ignorados(1464)
    assert c3.LECTURA_DECLARADA["1464"]["columnas_ignoradas_por_openml"] == [] and len(filas2[0]) == 5 + 1
    del c5


def test_ignore_attribute_main_de_medir_pone_la_lectura_sin_ignorados_en_el_lugar_de_la_que_usa_c5():
    """`particiones_base` de c5 llama a SU `cargar_arff`: si `medir_c169.main` no lo sustituye, la exclusión no llega a los intentos."""
    import inspect
    import medir_c169
    lineas = [l.strip() for l in inspect.getsource(medir_c169.main).splitlines()]
    sustitucion = [i for i, l in enumerate(lineas) if l.startswith("c5.cargar_arff = cj.cargar_arff_sin_ignorados")]   # una línea comentada no cuenta
    uso = [i for i, l in enumerate(lineas) if "c5.particiones_base(" in l and not l.startswith("#")]
    assert len(sustitucion) == 1 and len(uso) == 1 and sustitucion[0] < uso[0]
