# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C8 — abrir un Parquet como tabla (`matrixai.tablas.parquet_lector`): el núcleo.

Una prueba con nombre por cada DECLARACIÓN y por cada GUARDA (cada una tiene su sabotaje en `~/.agentes/150/c8/`):

* tipos: `int64` con nulos sin perder precisión, `NaN`/infinito → vacío CONTADO, zona horaria → UTC declarado, `ns`/`us`/`ms` →
  segundos declarado (con cuántos valores perdieron fracción), listas/estructuras/mapas/binarios EXCLUIDOS por nombre y tipo;
* guardas con los METADATOS y ANTES de leer un dato (bomba, filas, columnas, memoria, techo del proceso, pie imposible, cortado,
  cifrado), todas con su motivo es/en y ninguna con una excepción de pyarrow;
* sin `pyarrow` el módulo se importa, dice «instala pyarrow» y Excel sigue funcionando (invariante 7);
* determinismo (mismos bytes), el origen y un paquete con origen Parquet que verifica.

`MATRIXAI_EXIGIR_PARQUET=1` (lo ponen los guiones de medición de C8): si falta `pyarrow` el entorno FALLA en vez de saltar en
silencio. Es aparte de `MATRIXAI_EXIGIR_TABLAS` (calamine) a propósito: exigir calamine no puede romper un entorno sin pyarrow.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys

import pytest

from matrixai import limits
from matrixai.tablas import CODIGOS, TablaRechazada, Topes, convertir, detectar_formato, inspeccionar
from matrixai.tablas import parquet_lector as pl

RAIZ = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "tablas"

HAY_PYARROW = importlib.util.find_spec("pyarrow") is not None
requiere_pyarrow = pytest.mark.skipif(
    not HAY_PYARROW, reason="falta pyarrow (extra `parquet`): pip install 'matrixai-core[parquet]'")

_spec = importlib.util.spec_from_file_location("generar_parquet_150", FIXTURES / "generar_parquet.py")
GP = importlib.util.module_from_spec(_spec)
sys.modules["generar_parquet_150"] = GP
_spec.loader.exec_module(GP)


def leer(nombre: str) -> bytes:
    return (FIXTURES / nombre).read_bytes()


def filas_de(t) -> list[list[str]]:
    return list(csv.reader(io.StringIO(t.csv_text)))


def columna(t, nombre: str) -> list[str]:
    cab, *resto = filas_de(t)
    i = cab.index(nombre)
    return [f[i] for f in resto]


def decl(t, tipo, columna=None) -> list[dict]:
    return [d for d in t.origen["declaraciones"] if d["tipo"] == tipo and (columna is None or d["columna"] == columna)]


def rechazo(f, *a, **k) -> TablaRechazada:
    with pytest.raises(TablaRechazada) as e:
        f(*a, **k)
    return e.value


def escribir_parquet(tabla, **kw) -> bytes:
    import pyarrow.parquet as pq
    buf = io.BytesIO()
    pq.write_table(tabla, buf, **kw)
    return buf.getvalue()


# ── El entorno ──────────────────────────────────────────────────────────────────────────
def test_el_entorno_de_medicion_tiene_pyarrow():
    """Que las pruebas con pyarrow no se salten TODAS en silencio: donde se mide, falla si falta."""
    if os.environ.get("MATRIXAI_EXIGIR_PARQUET") == "1":
        assert HAY_PYARROW, ("MATRIXAI_EXIGIR_PARQUET=1 y pyarrow no está: las pruebas de Parquet se habrían saltado "
                             "en silencio")
    elif not HAY_PYARROW:
        pytest.skip("sin MATRIXAI_EXIGIR_PARQUET=1: este entorno no mide el extra `parquet`")


def test_exigir_pyarrow_es_aparte_de_exigir_calamine():
    """`MATRIXAI_EXIGIR_TABLAS` (calamine) no obliga a tener pyarrow: son dos extras y dos variables."""
    fuente = pathlib.Path(__file__).read_text(encoding="utf-8")
    assert 'os.environ.get("MATRIXAI_EXIGIR_PARQUET")' in fuente
    assert "EXIGIR_TABLAS" not in fuente.replace("`MATRIXAI_EXIGIR_TABLAS`", "").replace("EXIGIR_TABLAS` (calamine)", "") \
        .split("def test_exigir_pyarrow_es_aparte")[0]


# ── Invariante 7: sin pyarrow el módulo se importa y lo dice ────────────────────────────
def test_sin_pyarrow_el_modulo_se_importa_y_dice_instala_pyarrow_en_los_dos_idiomas():
    """`python3 -s -E` (sin `site-packages` del usuario ni PYTHONPATH): `matrixai.tablas` importa, NO importa pyarrow, y un
    Parquet válido-de-forma dice `falta_pyarrow` con «pyarrow» y cómo instalarlo; un fichero cortado se rechaza igual (no
    necesita el lector). Esto es lo que ve quien no tiene el extra."""
    ejemplo = FIXTURES / "parquet_tipos.parquet"
    codigo = f"""
import sys
sys.path[:0] = [{str(RAIZ)!r}]
import matrixai.tablas as t
assert 'pyarrow' not in sys.modules, 'importó pyarrow al importar el módulo'
datos = open({str(ejemplo)!r}, 'rb').read()
r = []
for d in (datos, datos[:-20]):
    try:
        t.convertir(d, 'a.parquet')
    except t.TablaRechazada as e:
        r.append((e.codigo, e.motivo_es, e.motivo_en))
try:
    t.inspeccionar(datos, 'a.parquet')
except t.TablaRechazada as e:
    r.append((e.codigo, '', ''))
assert 'pyarrow' not in sys.modules, 'importó pyarrow sin tenerlo'
print(r[0][0], r[1][0], r[2][0]); print(r[0][1]); print(r[0][2])
"""
    out = subprocess.run([sys.executable, "-s", "-E", "-c", codigo], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    primera, es, en = out.stdout.strip().splitlines()
    assert primera == "falta_pyarrow parquet_danado falta_pyarrow"
    assert "pyarrow" in es and "pip install" in es and "Excel" in es
    assert "pyarrow" in en and "pip install" in en and "Excel" in en


def test_importar_el_modulo_no_importa_pyarrow_aunque_este_instalado():
    """Con pyarrow DISPONIBLE (PYTHONPATH a su site), `import matrixai.tablas` y abrir un Excel que no existe no lo cargan:
    solo un Parquet lo importa (+66 MB de RSS)."""
    codigo = (f"import sys; sys.path[:0] = [{str(RAIZ)!r}]\nimport matrixai.tablas as t\n"
              "try:\n    t.convertir(b'a,b\\n1,2\\n', 'x.xlsx', fila_de_cabecera=1)\nexcept t.TablaRechazada as e:\n    print(e.codigo)\n"
              "print('pyarrow' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split() == ["texto_no_excel", "False"]


def test_sin_pyarrow_un_excel_se_abre_igual(monkeypatch):
    """«Excel sigue funcionando»: con `pyarrow` fuera de juego, un libro de Excel se convierte como siempre."""
    if importlib.util.find_spec("python_calamine") is None:
        pytest.skip("falta python-calamine (extra `tablas`)")
    monkeypatch.setattr(pl, "pyarrow_disponible", lambda: False)
    t = convertir((FIXTURES / "oficina_con_valor.xlsx").read_bytes(), "o.xlsx", hoja="Datos", fila_de_cabecera=3, filas_a_excluir=[10])
    assert t.origen["formato"] == "xlsx"
    assert rechazo(convertir, leer("parquet_tipos.parquet"), "p.parquet").codigo == "falta_pyarrow"


@requiere_pyarrow
def test_pyarrow_instalado_que_no_carga_no_dice_instala_pyarrow(monkeypatch):
    """Si `find_spec` lo encuentra pero el import falla (sin memoria, instalación rota), «instala pyarrow» es falso."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    monkeypatch.setattr(pl, "pyarrow_disponible", lambda: True)
    e = rechazo(convertir, leer("parquet_grupos.parquet"), "g.parquet")
    assert e.codigo == "pyarrow_no_carga" and "ModuleNotFoundError" in e.motivo_es
    assert "instalado" in e.motivo_es and "installed" in e.motivo_en and "instala pyarrow" not in e.motivo_es


def test_falta_pyarrow_tiene_su_motivo_en_los_dos_idiomas():
    es, en = CODIGOS["falta_pyarrow"]
    assert "pyarrow" in es and "pyarrow" in en and "pip install" in es and "pip install" in en
    assert "matrixai-core[parquet]" in es and "matrixai-core[parquet]" in en


# ── Criterio 1: formato por FIRMA ───────────────────────────────────────────────────────
def test_un_parquet_se_reconoce_por_su_firma_aunque_se_llame_csv_o_xlsx():
    assert detectar_formato(leer("parquet_tipos.parquet")[:4096]) == "parquet"
    assert pl.es_parquet(leer("parquet_tipos.parquet"))
    assert not pl.es_parquet(b"PARENT_ID,x\n1,2\n"), "un CSV que empieza por «PARE» NO es un Parquet cifrado"
    assert not pl.es_parquet(b"a,b\n1,2\n")


@requiere_pyarrow
@pytest.mark.parametrize("nombre", ["datos.csv", "datos.xlsx", "datos", "DATOS.PARQUET", ""])
def test_la_extension_no_manda_un_parquet_se_abre_con_cualquier_nombre(nombre):
    t = convertir(leer("parquet_grupos.parquet"), nombre)
    assert t.origen["formato"] == "parquet" and len(filas_de(t)) == 26


# ── Qué hace con cada tipo ──────────────────────────────────────────────────────────────
@requiere_pyarrow
def test_int64_con_nulos_no_pierde_precision_por_encima_de_2_a_la_53():
    """El fichero que haría `pandas`: `int64` con nulos → `float64` → 9007199254740993 sale 9007199254740992. Aquí cada valor
    es un `int` de Python: sale tal cual y el nulo es VACÍO (jamás 0 ni «nan»)."""
    t = convertir(leer("parquet_enteros.parquet"), "e.parquet")
    assert columna(t, "x") == ["9007199254740993", "", "9223372036854775807", "-9223372036854775808", "7", ""]
    assert not decl(t, "entero_no_exacto"), "un entero de Parquet es EXACTO: no hay nada que declarar"
    assert "0" not in columna(t, "x")


@requiere_pyarrow
def test_nan_e_infinito_salen_vacios_y_contados_y_un_nulo_no_se_cuenta():
    t = convertir(leer("parquet_tipos.parquet"), "t.parquet")
    nan = columna(t, "flotante_nan")
    assert nan == ["1.5", "", "", "1.5", "", "", "1.5", "", "", "1.5", "", ""]
    d, = decl(t, "no_finito", "flotante_nan")
    assert d["cuantas"] == 4, "los NaN son 4; los 4 nulos NO se cuentan (un valor ausente no es un NaN)"
    inf = columna(t, "flotante_inf")
    assert inf[1] == inf[2] == "" and decl(t, "no_finito", "flotante_inf")[0]["cuantas"] == 6
    celdas = {c.lower() for fila in filas_de(t)[1:] for c in fila}
    assert not celdas & {"nan", "inf", "-inf", "infinity", "-infinity"}, "ni «nan» ni «inf» llegan al CSV"
    assert "NaN" in d["texto"]["es"] and "NaN" in d["texto"]["en"]


@requiere_pyarrow
def test_la_zona_horaria_se_escribe_en_utc_y_se_declara_con_su_zona_y_cuantas():
    t = convertir(leer("parquet_tipos.parquet"), "t.parquet")
    # 2024-03-31 00:00+01:00 es 2024-03-30 23:00 UTC; el cambio de hora (+02:00 desde las 02:00 locales) no desordena los UTC
    madrid = columna(t, "ts_madrid_ns")
    assert madrid[0] == "2024-03-30 23:00:00" and madrid[2] == "2024-03-31 01:00:00" and madrid[3] == "2024-03-31 02:00:00"
    d, = decl(t, "parquet_zona_horaria_a_utc", "ts_madrid_ns")
    assert d["detalle"]["zona"] == "Europe/Madrid" and d["cuantas"] == 12
    assert "UTC" in d["texto"]["es"] and "Europe/Madrid" in d["texto"]["es"]
    assert "UTC" in d["texto"]["en"] and "time zone" in d["texto"]["en"]
    assert decl(t, "parquet_zona_horaria_a_utc", "ts_utc_us")[0]["detalle"]["zona"] == "UTC"
    # una columna SIN zona no declara zona
    assert not decl(t, "parquet_zona_horaria_a_utc", "ts_naive_ms")
    assert all("+" not in v and "Z" not in v for v in madrid), "sin indicar la zona: el analizador no la entiende"


@requiere_pyarrow
def test_la_zona_horaria_nula_no_se_cuenta_como_instante():
    import pyarrow as pa
    datos = escribir_parquet(pa.table({"t": pa.array([1_700_000_000_000_000, None, 1_700_000_001_000_000], pa.timestamp("us", tz="Europe/Madrid"))}))
    t = convertir(datos, "z.parquet")
    assert decl(t, "parquet_zona_horaria_a_utc", "t")[0]["cuantas"] == 2
    assert columna(t, "t") == ["2023-11-14 22:13:20", "", "2023-11-14 22:13:21"]


@requiere_pyarrow
def test_ns_a_segundos_se_declara_con_cuantos_valores_perdieron_su_fraccion():
    t = convertir(leer("parquet_tipos.parquet"), "t.parquet")
    con, = decl(t, "parquet_unidad_a_segundos", "ts_naive_ns_con_fraccion")
    assert con["detalle"]["unidad"] == "ns" and con["cuantas"] == 12
    assert "12 valores perdieron su fracción" in con["texto"]["es"] and "nanosegundos" in con["texto"]["es"]
    assert "12 values lost their fraction" in con["texto"]["en"] and "nanoseconds" in con["texto"]["en"]
    assert columna(t, "ts_naive_ns_con_fraccion")[1] == "2024-01-02 00:00:00", "se TRUNCA (no se redondea al segundo siguiente)"
    # sin fracción: se declara la unidad igualmente (cuantas = 0) y el texto lo dice
    sin, = decl(t, "parquet_unidad_a_segundos", "ts_madrid_ns")
    assert sin["cuantas"] == 0 and "ningún valor perdió" in sin["texto"]["es"] and "no value lost" in sin["texto"]["en"]


@requiere_pyarrow
def test_la_fraccion_se_trunca_tambien_en_fechas_anteriores_a_1970():
    """`divmod` y no `int()`: -0,5 s es el segundo -1 con resto, no el 0 (que sería redondear hacia cero)."""
    import pyarrow as pa
    datos = escribir_parquet(pa.table({"t": pa.array([-1_500_000_000, 1_500_000_000], pa.timestamp("ns"))}))
    t = convertir(datos, "n.parquet")
    assert columna(t, "t") == ["1969-12-31 23:59:58", "1970-01-01 00:00:01"]
    assert decl(t, "parquet_unidad_a_segundos", "t")[0]["cuantas"] == 2
    # y .9999 s NO sube al segundo siguiente (truncar no es redondear)
    datos = escribir_parquet(pa.table({"t": pa.array([1_999_999_999, 2_000_000_000, 59_999_999_999], pa.timestamp("ns"))}))
    assert columna(convertir(datos, "n.parquet"), "t") == ["1970-01-01 00:00:01", "1970-01-01 00:00:02", "1970-01-01 00:00:59"]


@requiere_pyarrow
def test_fechas_horas_decimales_booleanos_y_textos_se_escriben_como_los_entiende_el_analizador():
    t = convertir(leer("parquet_tipos.parquet"), "t.parquet")
    assert columna(t, "dia")[:2] == ["2024-01-01", "2024-01-02"]
    assert columna(t, "dia_ms")[:2] == ["2024-02-01", "2024-02-02"]
    assert columna(t, "hora_us")[:2] == ["08:30:00", "08:30:01"]
    assert columna(t, "importe")[:2] == ["1234.50", "1235.50"], "decimal exacto, con su escala"
    assert columna(t, "cat")[:4] == ["a", "b", "", "a"], "un diccionario se descodifica; su nulo es vacío"
    assert columna(t, "flag")[:3] == ["true", "false", ""]
    assert columna(t, "uint8")[:2] == ["200", "201"]
    assert columna(t, "texto")[:4] == ["año", "ñandú, con coma", 'con "comillas"', "salto\nde línea"]
    assert columna(t, "simple")[:4] == ["0.1", "0.25", "3", ""], "float32: la representación más corta de ESE float32"
    assert columna(t, "grande")[0] == "4611686018427387904"


@requiere_pyarrow
def test_una_fecha_fuera_de_los_anos_1_a_9999_queda_vacia_y_se_declara():
    import pyarrow as pa
    datos = escribir_parquet(pa.table({"d": pa.array([0, 19_000, 2_000_000_000], pa.date32()),
                                       "t": pa.array([10**12, 0, 2 * 10**17], pa.timestamp("ms"))}))
    t = convertir(datos, "f.parquet")
    assert columna(t, "d") == ["1970-01-01", "2022-01-08", ""]
    assert decl(t, "parquet_fecha_fuera_de_rango", "d")[0]["cuantas"] == 1
    assert columna(t, "t")[2] == "" and decl(t, "parquet_fecha_fuera_de_rango", "t")[0]["cuantas"] == 1
    assert "1 a 9999" in decl(t, "parquet_fecha_fuera_de_rango", "d")[0]["texto"]["es"]


# ── Lo que un CSV no puede guardar ──────────────────────────────────────────────────────
@requiere_pyarrow
@pytest.mark.parametrize("nombre,categoria,tipo_arrow", [
    ("lista", "lista", "list<element: int64>"),
    ("estructura", "estructura", "struct<a: int64>"),
    ("mapa", "mapa", "map<string, int64 ('mapa')>"),
    ("binario", "binario", "binary"),
])
def test_listas_estructuras_mapas_y_binarios_se_excluyen_declarados_por_nombre_y_tipo(nombre, categoria, tipo_arrow):
    d = leer("parquet_anidados.parquet")
    t = convertir(d, "a.parquet")
    cab = filas_de(t)[0]
    assert cab == ["id", "valor"], "el CSV lleva SOLO lo que se puede guardar, en su orden"
    e, = decl(t, "parquet_columna_excluida", nombre)
    assert e["detalle"]["categoria"] == categoria and e["detalle"]["tipo_arrow"] == tipo_arrow
    assert nombre in e["texto"]["es"] and tipo_arrow in e["texto"]["es"]
    assert nombre in e["texto"]["en"] and tipo_arrow in e["texto"]["en"]
    assert e["texto"]["es"] != e["texto"]["en"]
    assert t.origen["columnas"] == 2 and t.origen["columnas_del_fichero"] == 6


@requiere_pyarrow
def test_la_lista_de_columnas_excluidas_se_ve_en_inspeccionar_ANTES_de_convertir_y_sin_leer_un_dato(monkeypatch):
    """Que la persona vea qué no entra ANTES de seguir: `inspeccionar` lo da, leyendo solo el pie (nunca un lote)."""
    with monkeypatch.context() as m:
        _prohibir_leer(m)
        v = inspeccionar(leer("parquet_anidados.parquet"), "a.parquet")
    assert v["formato"] == "parquet" and v["filas"] == 12 and v["hojas"] == [] and v["hoja_propuesta"] is None
    assert [c["nombre"] for c in v["columnas_excluidas"]] == ["lista", "estructura", "mapa", "binario"]
    assert {c["nombre"]: c["categoria"] for c in v["columnas_excluidas"]} == {
        "lista": "lista", "estructura": "estructura", "mapa": "mapa", "binario": "binario"}
    assert [c["nombre"] for c in v["columnas"] if c["se_convierte"]] == ["id", "valor"]
    avisos = [a for a in v["avisos"] if a["tipo"] == "parquet_columna_excluida"]
    assert [a["columna"] for a in avisos] == ["lista", "estructura", "mapa", "binario"]
    assert all(a["texto"]["es"] and a["texto"]["en"] for a in avisos)
    # y lo mismo que luego declara la conversión (misma redacción)
    t = convertir(leer("parquet_anidados.parquet"), "a.parquet")
    assert [a["texto"] for a in avisos] == [decl(t, "parquet_columna_excluida", a["columna"])[0]["texto"] for a in avisos]


@requiere_pyarrow
def test_inspeccionar_avisa_de_la_zona_horaria_y_de_la_unidad_antes_de_convertir():
    v = inspeccionar(leer("parquet_tipos.parquet"), "t.parquet")
    previstas = {(a["columna"], a["detalle"]["que"]) for a in v["avisos"] if a["tipo"] == "parquet_conversion_prevista"}
    assert ("ts_madrid_ns", "zona_horaria_a_utc") in previstas and ("ts_madrid_ns", "unidad_a_segundos") in previstas
    assert ("ts_naive_ns_con_fraccion", "unidad_a_segundos") in previstas
    assert ("dia", "unidad_a_segundos") not in previstas
    assert v["sha256"] == hashlib.sha256(leer("parquet_tipos.parquet")).hexdigest()
    assert v["bytes_descomprimidos"] > 0 and v["conversor"]["versiones"]["pyarrow"]


@requiere_pyarrow
def test_un_parquet_con_todas_las_columnas_imposibles_no_se_convierte_y_dice_por_que():
    for f in (inspeccionar, convertir):
        e = rechazo(f, leer("parquet_solo_anidados.parquet"), "s.parquet")
        assert e.codigo == "parquet_sin_columnas_convertibles"
        assert "2 columnas" in e.motivo_es and "2 columns" in e.motivo_en


# ── Nombres ─────────────────────────────────────────────────────────────────────────────
@requiere_pyarrow
def test_nombres_vacio_repetido_y_con_espacios_se_arreglan_y_se_declaran():
    t = convertir(leer("parquet_nombres.parquet"), "n.parquet")
    assert filas_de(t)[0] == ["Edad", "Edad_2", "columna_3", "Peso", "altura"]
    assert decl(t, "parquet_nombre_repetido", "Edad_2")[0]["detalle"]["original"] == "Edad"
    assert decl(t, "parquet_nombre_vacio", "columna_3")[0]["detalle"]["posicion"] == 3
    assert decl(t, "nombre_normalizado", "Peso") and decl(t, "nombre_normalizado", "altura")
    assert "n.º 3" in decl(t, "parquet_nombre_vacio")[0]["texto"]["es"] and "no. 3" in decl(t, "parquet_nombre_vacio")[0]["texto"]["en"]


@requiere_pyarrow
def test_una_sola_columna_con_punto_y_coma_en_el_nombre_no_parece_un_csv_punto_y_coma():
    """Como en Excel (remate de 150-C4): con UNA columna, `a;b` en la cabecera haría que el CSV se leyera separado por `;`."""
    import pyarrow as pa
    t = convertir(escribir_parquet(pa.table({"peso;altura": pa.array([1, 2])})), "u.parquet")
    assert filas_de(t)[0] == ["peso_altura"] and decl(t, "punto_y_coma_en_el_nombre", "peso_altura")


# ── Lotes, determinismo, origen ─────────────────────────────────────────────────────────
@requiere_pyarrow
def test_la_lectura_por_lotes_cruza_los_bordes_de_lote_y_de_grupo_sin_cambiar_el_csv(monkeypatch):
    base = convertir(leer("parquet_grupos.parquet"), "g.parquet")
    assert len(filas_de(base)) == 26
    vistos = []
    real = pl._filas_de_los_lotes
    monkeypatch.setattr(pl, "TAMANO_DE_LOTE", 7)
    monkeypatch.setattr(pl, "_filas_de_los_lotes", lambda *a, **k: (vistos.append(1), real(*a, **k))[1])
    chico = convertir(leer("parquet_grupos.parquet"), "g.parquet")
    assert chico.csv_bytes == base.csv_bytes and vistos
    assert [f[0] for f in filas_de(chico)[1:]] == [str(i) for i in range(25)]


@requiere_pyarrow
def test_mismo_fichero_mismos_bytes_y_el_hash_del_csv_es_el_del_origen():
    d = leer("parquet_tipos.parquet")
    a, b = convertir(d, "t.parquet"), convertir(d, "t.parquet")
    assert a.csv_bytes == b.csv_bytes and a.origen == b.origen
    assert a.origen["sha256_del_csv"] == hashlib.sha256(a.csv_bytes).hexdigest()
    assert a.csv_bytes.endswith(b"\n") and b"\r" not in a.csv_bytes and not a.csv_bytes.startswith(b"\xef\xbb\xbf")


@requiere_pyarrow
def test_el_origen_de_un_parquet_dice_lo_que_sabe_y_omite_lo_que_no():
    d = leer("parquet_tipos.parquet")
    t = convertir(d, "ruta/t.parquet")
    o = t.origen
    assert o["formato"] == "parquet" and o["nombre"] == "t.parquet" and o["bytes"] == len(d)
    assert o["sha256"] == hashlib.sha256(d).hexdigest()
    assert o["filas"] == 12 and o["columnas"] == 19 and o["filas_excluidas"] == []
    assert "hoja" not in o and "fila_de_cabecera" not in o and "rango" not in o, "un Parquet no tiene hoja ni cabecera que inventar"
    assert o["conversor"]["nombre"] == "matrixai.tablas" and o["conversor"]["versiones"]["pyarrow"]
    from matrixai.export.origen_del_fichero import normalizar_origen_del_fichero
    assert normalizar_origen_del_fichero(o)["formato"] == "parquet"


@requiere_pyarrow
def test_el_csv_de_un_parquet_pasa_por_el_producto_sin_tocarlo():
    from matrixai.training.dataset_analysis import analyze_dataset_csv
    t = convertir(leer("parquet_tipos.parquet"), "t.parquet")
    r = analyze_dataset_csv(t.csv_text)
    tipos = {c: r["columns"][c]["type"] for c in r["column_order"]}
    assert tipos["dia"] == "date" and tipos["importe"] == "number" and tipos["cat"] == "categorical"
    assert tipos["flag"] == "boolean" and tipos["entero_con_nulos"] == "integer"
    assert tipos["ts_utc_us"] == "date", "con segundos y sin zona el analizador reconoce el instante"
    from matrixai.export.origen_del_fichero import entrada_canonica_de
    texto, declaraciones = entrada_canonica_de(t.csv_text, t.origen)
    assert texto == t.csv_text and declaraciones == [], "el CSV de una conversión de Parquet ya es canónico: pasa tal cual"


@requiere_pyarrow
def test_un_paquete_con_origen_parquet_verifica_igual_que_sin_origen():
    """Invariante 11 con el origen NUEVO: un paquete que lleva `origen_del_fichero` de un Parquet verifica como el que no lo lleva."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_150_c4_origen_en_el_recibo import _informe, _paquete_regenerable
    origen = convertir(leer("parquet_tipos.parquet"), "t.parquet").origen
    sin = _informe(_paquete_regenerable())
    con = _informe(_paquete_regenerable(origen=origen))
    assert con == sin and con["etapas"]["manifest"][0] == "PASS"
    from matrixai.export.origen_del_fichero import texto_del_origen
    assert "PARQUET" in texto_del_origen(origen, "es") and "PARQUET" in texto_del_origen(origen, "en")
    assert "hoja" not in texto_del_origen(origen, "es").lower().split("huella")[0]


def test_la_ficha_de_puntuacion_dice_fichero_parquet_y_no_csv_a_secas():
    from matrixai.export.informe_de_puntuacion_textos import t
    assert "Parquet" in t("vi_fuente_parquet", "es", f="a.parquet", n=3) and "Parquet" in t("vi_fuente_parquet", "en", f="a.parquet", n=3)
    assert "Excel" not in t("vi_fuente_parquet", "es", f="a.parquet", n=3)


# ── Lo que un Parquet no tiene de Excel se RECHAZA, no se ignora ────────────────────────
@requiere_pyarrow
@pytest.mark.parametrize("kw", [{"hoja": "Datos"}, {"fila_de_cabecera": 3}, {"filas_de_cabecera": 2}, {"filas_a_excluir": [4]},
                                {"rellenar_combinadas": True}, {"partir_columna_unica": ","},
                                {"componer_cabecera_de_dos_niveles": True}])
def test_lo_de_excel_que_se_pide_a_un_parquet_se_rechaza_en_vez_de_ignorarse(kw):
    """Ignorar `filas_a_excluir` dejaría en el CSV una fila que la persona pidió quitar y le diría que no."""
    e = rechazo(convertir, leer("parquet_grupos.parquet"), "g.parquet", **kw)
    assert e.codigo == "parametro_invalido" and "Parquet" in e.motivo_es and "Parquet" in e.motivo_en


@requiere_pyarrow
def test_fila_de_cabecera_1_o_ausente_valen_para_un_parquet_y_para_un_excel_sigue_siendo_obligatoria():
    assert convertir(leer("parquet_grupos.parquet"), "g.parquet", fila_de_cabecera=1).csv_bytes == \
        convertir(leer("parquet_grupos.parquet"), "g.parquet").csv_bytes
    e = rechazo(convertir, b"PK\x03\x04" + b"\0" * 30, "x.xlsx")
    assert e.codigo == "parametro_invalido", "sin `fila_de_cabecera` un libro de Excel sigue rechazándose"


# ── Las guardas, ANTES de leer un dato ──────────────────────────────────────────────────
def _prohibir_leer(monkeypatch):
    import pyarrow.parquet as pq

    def no(*a, **k):
        raise AssertionError("se leyeron datos antes de pasar la guarda")
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", no)
    monkeypatch.setattr(pq.ParquetFile, "read", no, raising=False)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", no, raising=False)


@requiere_pyarrow
def test_la_bomba_se_rechaza_con_los_metadatos_sin_leer_un_dato(monkeypatch):
    _prohibir_leer(monkeypatch)
    e = rechazo(convertir, leer("parquet_tipos.parquet"), "t.parquet", topes=Topes(max_descomprimido=1000))
    assert e.codigo == "bomba" and e.detalle["bytes_descomprimidos"] > 1000
    assert "descomprimidos" in e.motivo_es and "uncompressed" in e.motivo_en and "Parquet" in e.motivo_es
    assert rechazo(inspeccionar, leer("parquet_tipos.parquet"), "t.parquet", topes=Topes(max_descomprimido=1000)).codigo == "bomba"


@requiere_pyarrow
def test_el_tope_de_filas_se_mira_en_el_pie_antes_de_leer(monkeypatch):
    _prohibir_leer(monkeypatch)
    e = rechazo(convertir, leer("parquet_tipos.parquet"), "t.parquet", topes=Topes(max_filas_parquet=5))
    assert e.codigo == "demasiadas_filas" and e.detalle["tope"] == 5 and "12 filas" in e.motivo_es and "12 rows" in e.motivo_en


@requiere_pyarrow
def test_el_tope_de_columnas_se_mira_en_el_pie_antes_de_leer(monkeypatch):
    _prohibir_leer(monkeypatch)
    e = rechazo(convertir, leer("parquet_tipos.parquet"), "t.parquet", topes=Topes(max_columnas=3))
    assert e.codigo == "demasiadas_columnas" and "19 columnas" in e.motivo_es and "19 columns" in e.motivo_en


@requiere_pyarrow
def test_la_guarda_de_memoria_es_la_que_ya_existe_y_rechaza_antes_de_leer(monkeypatch):
    """Invariante 6: se le pregunta a `limits.memoria_insuficiente` (la de los CSV), no a una segunda."""
    _prohibir_leer(monkeypatch)
    llamadas = []
    real = limits.memoria_insuficiente
    monkeypatch.setattr(limits, "_memoria_disponible_del_sistema", lambda: 10)
    monkeypatch.setattr(limits, "memoria_insuficiente", lambda n: llamadas.append(n) or real(n))
    e = rechazo(convertir, leer("parquet_tipos.parquet"), "t.parquet")
    assert e.codigo == "memoria_insuficiente" and llamadas
    assert "Parquet" in e.motivo_es and "Parquet" in e.motivo_en and e.motivo_es != e.motivo_en
    assert "{" not in e.motivo_es + e.motivo_en


@requiere_pyarrow
def test_la_guarda_de_memoria_se_puede_apagar_como_la_de_los_csv(monkeypatch):
    monkeypatch.setattr(limits, "_memoria_disponible_del_sistema", lambda: 1)
    monkeypatch.setenv("MATRIXAI_GUARDIA_MEMORIA", "0")
    assert convertir(leer("parquet_grupos.parquet"), "g.parquet").origen["filas"] == 25


@requiere_pyarrow
def test_la_memoria_necesaria_es_la_mayor_de_la_cota_por_celda_y_la_cota_por_byte():
    """Una columna de textos con diccionario declara muy pocos bytes descomprimidos y su CSV es 16 veces mayor (x67 medido):
    por eso hay dos cotas y vale la mayor."""
    f = pl._Fichero(leer("parquet_tipos.parquet"), "t.parquet", Topes())
    por_celda = f.filas * 19 * pl.BYTES_POR_CELDA_DE_PARQUET
    por_byte = f.descomprimido * pl.FACTOR_RAM_POR_BYTE_DESCOMPRIMIDO_DE_PARQUET
    assert f.memoria_necesaria(19) == max(por_celda, por_byte)
    assert por_celda != por_byte


@requiere_pyarrow
def test_el_techo_propio_del_proceso_se_mira_antes_de_leer(monkeypatch):
    """Con un techo (`RLIMIT_AS`, el del proceso de conversión del Studio) al que no le cabe, rechaza con las dos cifras."""
    _prohibir_leer(monkeypatch)
    monkeypatch.setattr(pl, "hueco_bajo_el_techo_del_proceso", lambda *a: (2 << 30, 1000))
    monkeypatch.setattr(limits, "_memoria_disponible_del_sistema", lambda: 1 << 40)
    e = rechazo(convertir, leer("parquet_tipos.parquet"), "t.parquet")
    assert e.codigo == "memoria_insuficiente" and "techo" in e.motivo_es and "limit" in e.motivo_en


def test_un_parquet_cortado_se_rechaza_sin_necesitar_pyarrow_y_nunca_con_una_excepcion_de_biblioteca(monkeypatch):
    def sin_pyarrow():
        raise AssertionError("un fichero cortado se rechaza por su marca final, sin abrir pyarrow")
    monkeypatch.setattr(pl, "_importar_pyarrow", sin_pyarrow)
    d = leer("parquet_tipos.parquet")
    for corte in (5, 12, 100, len(d) // 2, len(d) - 9, len(d) - 5, len(d) - 1):
        for f in (convertir, inspeccionar):
            e = rechazo(f, d[:corte], "c.parquet")
            assert e.codigo == "parquet_danado", (corte, e.codigo)
            assert "Traceback" not in e.motivo_es
    e = rechazo(convertir, d[:-1], "c.parquet")
    assert "PAR1" in e.detalle["tipo"] and "cortado" in e.detalle["tipo"]
    assert "dañado" in e.motivo_es and "damaged" in e.motivo_en


@requiere_pyarrow
def test_un_pie_que_declara_un_tamano_imposible_se_rechaza_antes_de_abrir(monkeypatch):
    import pyarrow.parquet as pq

    def no(*a, **k):
        raise AssertionError("se abrió el fichero con un pie imposible")
    monkeypatch.setattr(pq, "ParquetFile", no)
    d = bytearray(leer("parquet_tipos.parquet"))
    d[-8:-4] = (len(d) * 4).to_bytes(4, "little")
    e = rechazo(convertir, bytes(d), "p.parquet")
    assert e.codigo == "parquet_danado" and "pie" in e.detalle["tipo"]


@requiere_pyarrow
def test_basura_con_marcas_y_pie_validos_se_rechaza_con_su_motivo_y_sin_excepcion_de_pyarrow():
    for basura in (b"PAR1" + b"\x00" * 64 + (40).to_bytes(4, "little") + b"PAR1",
                   b"PAR1" + bytes(range(256)) * 4 + (200).to_bytes(4, "little") + b"PAR1"):
        for f in (convertir, inspeccionar):
            e = rechazo(f, basura, "b.parquet")
            assert e.codigo == "parquet_danado" and e.motivo_es and e.motivo_en
            assert "ArrowInvalid" not in e.motivo_es + e.motivo_en or "(" in e.motivo_es


def test_un_parquet_con_el_pie_cifrado_se_rechaza_con_su_motivo():
    cifrado = b"PARE" + b"\x01" * 40 + (30).to_bytes(4, "little") + b"PARE"
    e = rechazo(convertir, cifrado, "c.parquet")
    assert e.codigo == "parquet_cifrado" and "cifrado" in e.motivo_es and "encrypted" in e.motivo_en
    assert rechazo(inspeccionar, cifrado, "c.parquet").codigo == "parquet_cifrado"


@requiere_pyarrow
def test_un_fallo_de_pyarrow_que_habla_de_cifrado_sale_como_parquet_cifrado():
    """Un Parquet con las COLUMNAS cifradas (pie en claro) abre el pie y falla al leer: pyarrow lo dice con «decrypt»."""
    import pyarrow as pa
    f = pl._Fichero(leer("parquet_grupos.parquet"), "g.parquet", Topes())
    assert f._rechazo_de(OSError("Cannot decrypt ColumnMetadata")).codigo == "parquet_cifrado"
    assert f._rechazo_de(pa.ArrowNotImplementedError("Unsupported codec LZ4_HADOOP")).codigo == "parquet_no_legible"
    assert f._rechazo_de(pa.ArrowInvalid("x")).codigo == "parquet_danado"


@requiere_pyarrow
def test_el_csv_de_una_conversion_de_parquet_no_se_vuelve_a_canonizar_y_el_de_un_csv_punto_y_coma_si():
    """Lo MISMO que Excel: el CSV que escribe la conversión ya es canónico; un CSV `;` de verdad sí se canoniza."""
    from matrixai.export.origen_del_fichero import entrada_canonica_de
    texto = "a;b\n1,5;2\n"
    origen_parquet = convertir(leer("parquet_grupos.parquet"), "g.parquet").origen
    assert entrada_canonica_de(texto, origen_parquet) == (texto, [])
    canonico, declaraciones = entrada_canonica_de(texto, None)
    assert canonico != texto and declaraciones


@requiere_pyarrow
def test_un_parquet_sin_filas_o_sin_columnas_se_rechaza_como_vacio():
    import pyarrow as pa
    sin_filas = escribir_parquet(pa.table({"a": pa.array([], pa.int64())}))
    for f in (convertir, inspeccionar):
        e = rechazo(f, sin_filas, "v.parquet")
        assert e.codigo == "parquet_vacio" and "filas" in e.motivo_es and "rows" in e.motivo_en


@requiere_pyarrow
def test_un_fallo_de_pyarrow_al_leer_un_lote_sale_como_rechazo_y_no_como_excepcion(monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    def roto(*a, **k):
        raise pa.ArrowInvalid("Corrupt snappy compressed data")
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", roto)
    e = rechazo(convertir, leer("parquet_grupos.parquet"), "g.parquet")
    assert e.codigo == "parquet_danado" and "ArrowInvalid" in e.motivo_es


@requiere_pyarrow
def test_sin_memoria_en_pyarrow_sale_como_lector_sin_memoria_sin_acusar_al_fichero(monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    def sin(*a, **k):
        raise pa.ArrowMemoryError("realloc of size 123456789012 failed")
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", sin)
    e = rechazo(convertir, leer("parquet_grupos.parquet"), "g.parquet")
    assert e.codigo == "lector_sin_memoria"
    assert "demasiado grande" in e.motivo_es and "too large" in e.motivo_en


@requiere_pyarrow
def test_si_el_pie_declara_mas_filas_de_las_que_hay_se_rechaza_en_vez_de_dar_un_csv_corto(monkeypatch):
    """«Declarar lo que PASÓ»: un CSV con menos filas de las que dice el pie sería una pérdida callada."""
    import pyarrow.parquet as pq
    real = pq.ParquetFile.iter_batches

    def corta(self, *a, **k):
        for i, lote in enumerate(real(self, *a, **k)):
            if i < 1:
                yield lote
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", corta)
    monkeypatch.setattr(pl, "TAMANO_DE_LOTE", 7)
    e = rechazo(convertir, leer("parquet_grupos.parquet"), "g.parquet")
    assert e.codigo == "parquet_danado" and "25" in e.detalle["tipo"]


# ── Fixtures ────────────────────────────────────────────────────────────────────────────
def test_los_fixtures_de_parquet_son_pequenos_y_no_hay_de_mas():
    esperados = {"parquet_tipos.parquet", "parquet_anidados.parquet", "parquet_nombres.parquet", "parquet_enteros.parquet",
                 "parquet_grupos.parquet", "parquet_solo_anidados.parquet"}
    assert {p.name for p in FIXTURES.glob("*.parquet")} == esperados
    for p in FIXTURES.glob("*.parquet"):
        assert p.stat().st_size < 10_000, p.name
        assert p.read_bytes()[:4] == b"PAR1" and p.read_bytes()[-4:] == b"PAR1"


@requiere_pyarrow
def test_los_fixtures_commiteados_son_los_que_produce_el_guion():
    """Con la versión de pyarrow del guion: byte a byte. Con otra, el pie (`created_by`) cambia: se compara lo que el
    PRODUCTO lee de cada uno (el CSV y sus declaraciones)."""
    import pyarrow
    regenerados = GP.todos()
    assert set(regenerados) == {p.name for p in FIXTURES.glob("*.parquet")}
    if pyarrow.__version__ == GP.VERSION_DE_PYARROW:
        for nombre, b in regenerados.items():
            assert leer(nombre) == b, f"{nombre} no coincide con generar_parquet.py"
    else:
        for nombre, b in regenerados.items():
            if nombre == "parquet_solo_anidados.parquet":
                continue
            assert convertir(leer(nombre), nombre).csv_bytes == convertir(b, nombre).csv_bytes, nombre
    assert GP.todos() == GP.todos(), "el guion es determinista"


# ── Los motivos y las declaraciones nuevas, en los dos idiomas y con singular y plural ───────
_DETALLE = dict(descomprimido="1 GB", original="1 KB", tope="1000", hoja="", necesario="1 GB", disponible="1 MB",
                techo="1 GB", tipo="X", segundos=120, bytes_descomprimidos=1)


def test_toda_variante_de_parquet_tiene_motivo_en_es_y_en_sin_llaves_sueltas():
    from matrixai.tablas.errores import VARIANTES, rechazar
    visto = 0
    for (codigo, variante) in [k for k in VARIANTES if "parquet" in k[1]]:
        for n in (1, 2):
            e = rechazar(codigo, variante=variante, filas=n, columnas=n, **_DETALLE)
            assert e.motivo_es and e.motivo_en and e.motivo_es != e.motivo_en, (codigo, variante)
            assert "{" not in e.motivo_es + e.motivo_en, (codigo, variante)
            assert "Parquet" in e.motivo_es and "Parquet" in e.motivo_en, (codigo, variante)
            assert " 1 filas" not in e.motivo_es and " 1 rows" not in e.motivo_en and " 1 columnas" not in e.motivo_es
        visto += 1
    assert visto >= 6


def test_los_codigos_nuevos_de_parquet_estan_en_el_catalogo_y_el_viejo_ya_no():
    for codigo in ("falta_pyarrow", "pyarrow_no_carga", "parquet_danado", "parquet_cifrado", "parquet_no_legible", "parquet_vacio",
                   "parquet_sin_columnas_convertibles"):
        assert codigo in CODIGOS, codigo
    assert "parquet_no_soportado" not in CODIGOS, "«todavía no se abre» es falso desde el 150-C8"
    for es, en in CODIGOS.values():
        assert "todavía no se abre" not in es and "cannot be opened yet" not in en


def test_las_declaraciones_nuevas_de_parquet_no_dicen_1_valores_ni_2_valor():
    import re
    from matrixai.tablas.declaraciones import TIPOS, declarar
    for tipo in [t for t in TIPOS if t.startswith("parquet_")]:
        for n in (1, 2, 5):
            d = declarar(tipo, columna="c", cuantas=n, zona="UTC", unidad="ns", tipo_arrow="list<element: int64>",
                         categoria="lista", posicion=3, original="o", que="zona_horaria_a_utc")
            for idioma in ("es", "en"):
                texto = d["texto"][idioma]
                assert "{" not in texto and texto, (tipo, idioma)
                if n == 1:
                    assert not re.search(r"(?<![\d.,])1 (valores|instantes|fechas|values|instants|dates)\b", texto), texto
                if n > 1:
                    assert not re.search(r"(?<![\d.,])[2-9] (valor|instante|fecha|value|instant|date)\b", texto), texto
