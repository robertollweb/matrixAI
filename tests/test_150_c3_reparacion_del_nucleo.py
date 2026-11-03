# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C3, reparación de su auditoría — lo que toca al núcleo de `matrixai.tablas`.

* I1: la guarda de memoria mira también el techo PROPIO del proceso (`RLIMIT_AS`), no solo la memoria del equipo: el
  Studio convierte en un proceso con techo y un libro sano que no cabía empezaba a leerse y acababa en un
  `MemoryError` dicho como «fichero dañado».
* M4: `parametro_invalido` en inglés dice su parámetro en inglés.
* M5: sin extensión, los motivos de «esto no es un Excel» no citan una extensión (ni «(sin extensión)» en inglés).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from matrixai.tablas import TablaRechazada, convertir, inspeccionar
from matrixai.tablas import topes
from matrixai.tablas.errores import rechazar

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS y /proc: Linux")


def _en_un_proceso_con_techo(margen_mb: int, codigo: str) -> dict:
    """Corre `codigo` en un proceso cuyo `RLIMIT_AS` blando es su `VmSize` de ahora + `margen_mb`."""
    guion = textwrap.dedent(f"""
        import json, resource, sys
        sys.path[:0] = {sys.path!r}
        from matrixai.tablas import topes
        from matrixai.tablas.errores import TablaRechazada
        usado = topes.memoria_virtual_del_proceso()
        resource.setrlimit(resource.RLIMIT_AS, (usado + ({margen_mb} << 20), resource.getrlimit(resource.RLIMIT_AS)[1]))
        try:
{textwrap.indent(codigo, " " * 12)}
            print(json.dumps({{"ok": True}}))
        except TablaRechazada as e:
            print(json.dumps({{"ok": False, **e.como_dict()}}))
        """)
    out = subprocess.run([sys.executable, "-I", "-c", guion], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_I1_la_guarda_rechaza_lo_que_no_cabe_en_el_techo_del_proceso_con_sus_cifras():
    # 300 MB estimados para los dos procesos: a éste le toca un tercio (100 MB) y su techo deja 60 MB libres.
    r = _en_un_proceso_con_techo(60, "topes.comprobar_memoria(300_000_000, 'H')")
    assert r["ok"] is False and r["codigo"] == "memoria_insuficiente", r
    assert set(r["detalle"]) >= {"hoja", "necesario", "disponible", "techo"}
    assert r["detalle"]["necesario"] == "100.0 MB", r
    assert "techo" in r["motivo_es"] and "limit" in r["motivo_en"]
    assert "No indica que el fichero esté dañado" in r["motivo_es"]
    assert "does not mean the file is damaged" in r["motivo_en"]


def test_I1_la_guarda_deja_pasar_lo_que_si_cabe_en_el_techo_del_proceso():
    """La parte de ESTE proceso, no la suma: 300 MB estimados (100 propios) caben bajo un techo con 150 MB libres. Sin la
    fracción, se rechazaba lo que se convierte (medido: 200.000 x 10 con 300 MiB de techo)."""
    assert _en_un_proceso_con_techo(150, "topes.comprobar_memoria(300_000_000, 'H')") == {"ok": True}


def test_I1_el_hueco_se_mide_desde_antes_de_leer_y_no_cuenta_dos_veces_lo_ya_gastado():
    """Con `usado_al_empezar`, lo que la lectura ya ocupó entre una comprobación y otra no se resta otra vez: 80 MB
    reservados después de medir no cambian la respuesta."""
    codigo = ("al_empezar = topes.memoria_virtual_del_proceso()\n"
              "lastre = bytearray(80 << 20)\n"
              "topes.comprobar_memoria(300_000_000, 'H', usado_al_empezar=al_empezar)")
    assert _en_un_proceso_con_techo(150, codigo) == {"ok": True}
    sin = "lastre = bytearray(80 << 20)\ntopes.comprobar_memoria(300_000_000, 'H')"
    assert _en_un_proceso_con_techo(150, sin)["codigo"] == "memoria_insuficiente"


def test_I1_sin_techo_propio_la_guarda_no_opina_del_techo(monkeypatch):
    import resource
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        pytest.skip("este proceso ya tiene un RLIMIT_AS")
    assert topes.hueco_bajo_el_techo_del_proceso() is None


def test_I1_la_guarda_apagada_tampoco_mira_el_techo():
    codigo = ("import os\nos.environ['MATRIXAI_GUARDIA_MEMORIA'] = '0'\n"
              "topes.comprobar_memoria(300_000_000, 'H')")
    assert _en_un_proceso_con_techo(60, codigo) == {"ok": True}


def test_M4_parametro_invalido_en_ingles_dice_el_parametro_en_ingles():
    for kw, en in [({"fila_de_cabecera": 0}, "fila_de_cabecera must be an integer ≥ 1"),
                   ({"fila_de_cabecera": 1, "filas_de_cabecera": 3}, "filas_de_cabecera must be 1 or 2"),
                   ({"fila_de_cabecera": 1, "filas_a_excluir": "x"}, "must be a list of integer row numbers"),
                   ({"fila_de_cabecera": 1, "filas_a_excluir": [0]}, "filas_a_excluir must be a list of row numbers")]:
        with pytest.raises(TablaRechazada) as e:
            convertir(b"PK", "x.xlsx", **kw)
        assert e.value.codigo == "parametro_invalido" and en in e.value.motivo_en, (kw, e.value.motivo_en)
        assert "debe" not in e.value.motivo_en and "debe" in e.value.motivo_es


def test_M4_el_sufijo_en_solo_cambia_el_ingles():
    e = rechazar("parametro_invalido", parametro="uno", parametro_en="one")
    assert e.motivo_es == "Parámetro no válido: uno." and e.motivo_en == "Invalid parameter: one."
    # sin sufijo, el mismo valor en los dos (como antes)
    assert rechazar("parametro_invalido", parametro="p").motivo_en == "Invalid parameter: p."


@pytest.mark.parametrize("datos, codigo", [(b"a;b\n1;2\n3;4\n", "texto_no_excel"), (b"%PDF-1.4\n%...", "pdf_no_excel"),
                                           (b'<?xml version="1.0"?><raiz><a>1</a></raiz>', "xml_no_excel")])
@pytest.mark.parametrize("nombre", ["", "sin_punto"])
def test_M5_sin_extension_no_se_cita_ninguna_extension(datos, codigo, nombre):
    with pytest.raises(TablaRechazada) as e:
        inspeccionar(datos, nombre)
    assert e.value.codigo == codigo, e.value.codigo
    assert "(sin extensión)" not in e.value.motivo_es + e.value.motivo_en and "lleve la extensión" not in e.value.motivo_es
    assert "even though" not in e.value.motivo_en and "sin" not in e.value.motivo_en.split()
    with pytest.raises(TablaRechazada) as e:                      # con extensión, se sigue citando
        inspeccionar(datos, "libro.xlsx")
    assert ".xlsx" in e.value.motivo_es and ".xlsx" in e.value.motivo_en


def test_I1_leer_pasa_a_cada_guarda_la_memoria_de_ANTES_de_empezar(monkeypatch):
    """`api.leer` comprueba la memoria dos veces (con la cota provisional y con la de la geometría); entre una y otra la
    lectura ya ha ocupado memoria, que es parte de lo estimado. Las dos tienen que recibir el MISMO `usado_al_empezar`,
    medido antes de empezar: sin él, el hueco de la segunda se mide con lo ya gastado dentro y se rechaza un libro que
    cabe (medido en el Studio: 30.000 x 20 rechazado a 110 y 150 MiB, que convierte)."""
    pytest.importorskip("python_calamine")
    from matrixai.tablas import api as api_tablas
    import importlib.util
    import pathlib
    import matrixai
    generar = pathlib.Path(matrixai.__file__).resolve().parent.parent / "tests" / "fixtures" / "tablas" / "generar.py"
    spec = importlib.util.spec_from_file_location("generar_c150_rep", generar)
    G = importlib.util.module_from_spec(spec)
    sys.modules["generar_c150_rep"] = G
    spec.loader.exec_module(G)
    vistos: list = []
    real = api_tablas.comprobar_memoria

    def espia(necesario, hoja, *, usado_al_empezar=None):
        vistos.append(usado_al_empezar)
        return real(necesario, hoja, usado_al_empezar=usado_al_empezar)

    monkeypatch.setattr(api_tablas, "comprobar_memoria", espia)
    monkeypatch.setattr(api_tablas, "memoria_virtual_del_proceso", lambda: 123_456_789)
    convertir(G.xlsx([G.Hoja("H", [["a", "b"], [1, 2]])]), "x.xlsx", fila_de_cabecera=1)
    assert len(vistos) >= 2 and set(vistos) == {123_456_789}, vistos
