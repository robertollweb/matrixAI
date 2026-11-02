"""150-C2 — el decimal español (`12,5`, `1.234,56`) en la ENTRADA de un CSV con `;`.

Una sola regla (`matrixai.training.decimal_es`), llamada desde `normalize_csv_delimiter`, o sea desde todo lo que entra por
`normalize_csv_text`: el análisis, el entrenamiento, la verificación, el estudio y el lote. Aquí el núcleo; la entrada
canónica del estudio (que leía el CSV crudo) se prueba en `studio-backend/tests/test_150_c2_entrada_canonica.py`.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest. Las pruebas que explican «por qué NO hace lo obvio» llevan su nombre:
`test_una_columna_mixta_NO_se_convierte…`, `test_1_234_sin_coma_es_ambiguo…`, `test_un_codigo_con_ceros…`, `test_un_porcentaje…`,
`test_un_csv_con_comas_NO_se_toca…`.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from matrixai.training.data import (normalize_csv_delimiter, normalize_csv_text, normalize_csv_text_declarando,
                                    strip_csv_bom)
from matrixai.training.dataset_analysis import analyze_dataset_csv

AQUI = Path(__file__).resolve().parent
NUCLEO = AQUI.parent


def _norm(texto: str) -> tuple[str, list[dict]]:
    return normalize_csv_text_declarando(texto)


def _columna(texto_normalizado: str, nombre: str) -> list[str]:
    filas = list(csv.reader(io.StringIO(texto_normalizado)))
    i = filas[0].index(nombre)
    return [f[i] for f in filas[1:]]


def _tipos(texto: str) -> dict[str, str]:
    return {c: i["type"] for c, i in analyze_dataset_csv(texto)["columns"].items()}


def _declaradas(declaraciones: list[dict], tipo: str) -> dict[str, dict]:
    return {d["columna"]: d for d in declaraciones if d["tipo"] == tipo}


# El código de ANTES de C2, copiado tal cual: es contra lo que se comparan los bytes. Si esta copia se «arregla», la
# prueba deja de decir lo que dice.
def _normalize_csv_delimiter_antes(csv_text: str) -> str:
    if not csv_text.strip():
        return csv_text
    header_line = csv_text.splitlines()[0]
    if "," in header_line or ";" not in header_line:
        return csv_text
    reader = csv.reader(io.StringIO(csv_text), delimiter=";")
    out = io.StringIO()
    writer = csv.writer(out, delimiter=",", lineterminator="\n")
    writer.writerows(reader)
    return out.getvalue()


def _normalize_csv_text_antes(csv_text: str) -> str:
    return _normalize_csv_delimiter_antes(csv_text.removeprefix("\ufeff"))


# ── 1. La regla, con las formas que trae la gente ───────────────────────────────────────────────────────────────────

def test_el_ejemplo_del_contrato_precio_con_coma_es_un_numero():
    """`fecha;precio;unidades` con `12,5`: medido antes de C2 → `precio: identifier`/`categorical`."""
    t = "fecha;precio;unidades\n2024-01-01;12,5;3\n2024-01-02;9,75;4\n2024-01-03;20;5\n"
    antes = _tipos(_normalize_csv_text_antes(t))
    assert antes["precio"] != "number", "la prueba ya no mide el defecto: el decimal con coma ya era número"
    salida, decl = _norm(t)
    assert salida == "fecha,precio,unidades\n2024-01-01,12.5,3\n2024-01-02,9.75,4\n2024-01-03,20,5\n"
    assert _tipos(t) == {"fecha": "date", "precio": "number", "unidades": "integer"}
    assert list(_declaradas(decl, "decimal_coma_convertido")) == ["precio"]
    assert _declaradas(decl, "decimal_coma_convertido")["precio"]["cuantas"] == 2  # el `20` no cambia: no se cuenta


def test_miles_con_punto_y_negativos_y_signo():
    t = "importe;n\n1.234,56;1\n-3,25;2\n+0,5;3\n12.345.678,9;4\n0,05;5\n-1.000,00;6\n"
    salida, decl = _norm(t)
    assert _columna(salida, "importe") == ["1234.56", "-3.25", "+0.5", "12345678.9", "0.05", "-1000.00"]
    assert _declaradas(decl, "decimal_coma_convertido")["importe"]["cuantas"] == 6


def test_huecos_marcas_de_ausencia_y_espacios_no_impiden_ni_se_convierten():
    """Las celdas vacías y las marcas de ausencia del análisis (`NA`, `-`…) no cuentan; un hueco NO es un cero."""
    t = "a;b\n1,5;x\n;y\n NA ;z\n2,25 ;w\n-;v\n"
    salida, decl = _norm(t)
    assert _columna(salida, "a") == ["1.5", "", " NA ", "2.25 ", "-"]  # lo que no se convierte queda como estaba
    assert _declaradas(decl, "decimal_coma_convertido")["a"]["cuantas"] == 2
    assert _tipos(t)["a"] == "number"


def test_una_columna_mixta_NO_se_convierte_y_se_dice():
    """POR QUÉ NO convierte lo que sí parece decimal: «todo o nada». Una celda que no encaja (`abc`) deja la columna ENTERA
    como estaba: convertir solo las que parecen números cambiaría el valor de una columna de texto sin que lo viera nadie."""
    t = "a;b\n1,5;1\nabc;2\n2,5;3\n"
    salida, decl = _norm(t)
    assert _columna(salida, "a") == ["1,5", "abc", "2,5"]
    assert not _declaradas(decl, "decimal_coma_convertido")
    d = _declaradas(decl, "decimal_coma_no_convertido")["a"]
    assert d["cuantas"] == 2 and d["detalle"]["motivo"] == "no_numerica"
    assert d["detalle"]["ejemplo"] == {"valor": "abc", "fila": 2}
    assert "«abc»" in d["texto"]["es"] and "«abc»" in d["texto"]["en"]


def test_decimales_con_coma_y_con_punto_en_la_misma_columna_no_se_convierten():
    t = "a;b\n1,5;1\n2.5;2\n"
    salida, decl = _norm(t)
    assert _columna(salida, "a") == ["1,5", "2.5"]
    assert _declaradas(decl, "decimal_coma_no_convertido")["a"]["detalle"]["motivo"] == "mezcla_con_punto"


def test_1_234_sin_coma_es_ambiguo_y_no_se_toca_con_aviso():
    """`1.234`: ¿1,234 o 1234? Sin coma no hay forma de saberlo. Se deja y se avisa (contrato: «con aviso»)."""
    solo = "a;b\n1.234;1\n2.500;2\n3;3\n"
    salida, decl = _norm(solo)
    assert salida == "a,b\n1.234,1\n2.500,2\n3,3\n"
    d = _declaradas(decl, "decimal_coma_no_convertido")["a"]
    assert d["detalle"]["motivo"] == "ambigua" and d["cuantas"] == 2
    # y MEZCLADA con un decimal con coma, la columna tampoco se convierte (no se adivina cuál de las dos lecturas manda)
    mezcla = "a;b\n1.234,5;1\n2.500;2\n"
    salida, decl = _norm(mezcla)
    assert _columna(salida, "a") == ["1.234,5", "2.500"]
    assert _declaradas(decl, "decimal_coma_no_convertido")["a"]["detalle"]["motivo"] == "ambigua"


def test_un_decimal_con_punto_de_siempre_no_se_toca_ni_se_avisa():
    """`3.14159` y `0.5` son decimales con punto sin duda: nada que convertir y nada que decir (no se inventan avisos)."""
    t = "a;b\n3.14159;1\n0.5;2\n12.25;3\n"
    salida, decl = _norm(t)
    assert salida == "a,b\n3.14159,1\n0.5,2\n12.25,3\n" and decl == []


def test_un_codigo_con_ceros_a_la_izquierda_que_parece_decimal_NO_se_convierte():
    """«01,02» es un código (capítulo, versión), no 1,02: convertirlo lo destruiría. Tampoco `007`."""
    t = "codigo;n;zip\n01,02;1;007\n03,04;2;012\n"
    salida, decl = _norm(t)
    assert salida == 'codigo,n,zip\n"01,02",1,007\n"03,04",2,012\n'
    assert decl == []  # ni siquiera «parece decimal»: es un código, no hay nada que declarar
    # y un código mezclado con decimales de verdad deja la columna como estaba, y se dice
    t2 = "v;n\n0,5;1\n01,02;2\n"
    salida, decl = _norm(t2)
    assert _columna(salida, "v") == ["0,5", "01,02"]
    assert _declaradas(decl, "decimal_coma_no_convertido")["v"]["detalle"]["ejemplo"]["valor"] == "01,02"


def test_un_porcentaje_con_coma_NO_se_convierte_y_se_dice():
    """«12,5 %» no es un número: quitar el `%` lo convertiría en 12,5 o en 0,125 y eso es calcular. Se deja y se dice."""
    t = "tasa;n\n12,5 %;1\n7,0%;2\n;3\n"
    salida, decl = _norm(t)
    assert _columna(salida, "tasa") == ["12,5 %", "7,0%", ""]
    d = _declaradas(decl, "decimal_coma_no_convertido")["tasa"]
    assert d["detalle"]["motivo"] == "porcentaje" and d["cuantas"] == 2
    assert "%" in d["texto"]["es"] and "%" in d["texto"]["en"]


def test_las_fechas_dd_mm_aaaa_y_las_cabeceras_con_tildes_pasan_intactas():
    t = "fecha;año;descripción;importe\n05/01/2024;2024;café, con leche;1,5\n06/01/2024;2024;té;2,5\n"
    salida, decl = _norm(t)
    filas = list(csv.reader(io.StringIO(salida)))
    assert filas[0] == ["fecha", "año", "descripción", "importe"]
    assert filas[1] == ["05/01/2024", "2024", "café, con leche", "1.5"]
    assert _tipos(t)["fecha"] == "date" and _tipos(t)["importe"] == "number"
    assert list(_declaradas(decl, "decimal_coma_convertido")) == ["importe"]  # `año` (enteros) y la fecha no se tocan


def test_la_cabecera_nunca_se_convierte_aunque_parezca_un_numero():
    t = "1;2\n3,5;4\n4,5;5\n"
    salida, _ = _norm(t)
    assert salida == "1,2\n3.5,4\n4.5,5\n"  # la cabecera sigue siendo `1,2`; la columna 1 sí se convierte
    # y una cabecera con comas no es un CSV `;` por la señal conservadora de siempre: no se toca NADA
    t2 = "1,5;2,5\n3,5;4,5\n"
    assert normalize_csv_text(t2) == t2


def test_un_csv_con_comas_NO_se_toca_en_ingles_1_234_son_miles():
    """Solo un CSV que ya es `;`. Con `,` como separador, `"1,234"` y `"12,5"` entrecomillados quedan como estaban."""
    t = 'a,b\n"1,234",2\n"12,5",3\n'
    assert normalize_csv_text(t) == t
    assert normalize_csv_text_declarando(t) == (t, [])
    # ni siquiera con un `;` dentro de un valor
    t2 = 'a,b\n"12,5; algo",3\n'
    assert normalize_csv_text(t2) == t2
    # ni con una cabecera que lleva `,` Y `;` (la señal conservadora de siempre: con una coma en la cabecera, no es un `;`)
    t3 = "a,b;c\n1,5;2,5\n"
    assert normalize_csv_text(t3) == t3 and normalize_csv_text_declarando(t3) == (t3, [])


def test_un_valor_multilinea_entrecomillado_sobrevive_con_el_decimal():
    t = 'nota;peso\n"linea1\nlinea2";1,5\nb;2,5\n'
    salida, _ = _norm(t)
    assert list(csv.reader(io.StringIO(salida))) == [["nota", "peso"], ["linea1\nlinea2", "1.5"], ["b", "2.5"]]


def test_filas_cortas_y_largas_no_rompen_ni_inventan():
    t = "a;b;c\n1,5;2\n2,5;3;4;5\n"
    salida, _ = _norm(t)
    assert salida == "a,b,c\n1.5,2\n2.5,3,4,5\n"


def test_bom_y_decimal_a_la_vez():
    t = "\ufefffecha;valor\n2024-01-01;5,5\n"
    salida, decl = _norm(t)
    assert salida == "fecha,valor\n2024-01-01,5.5\n"
    assert _declaradas(decl, "decimal_coma_convertido")["valor"]["cuantas"] == 1


def test_columna_cuyo_unico_decimal_lleva_espacios_conserva_los_espacios():
    salida, _ = _norm("a;b\n 1,5 ;x\n2;y\n")
    assert _columna(salida, "a") == [" 1.5 ", "2"]


# ── 2. Idempotencia y mismos bytes ──────────────────────────────────────────────────────────────────────────────────

_FORMAS = [
    "fecha;precio;unidades\n2024-01-01;12,5;3\n2024-01-02;1.234,56;4\n",
    "a;b\n1,5;x\nabc;y\n",
    "a;b\n1.234;1\n2.500;2\n",
    "tasa;n\n12,5 %;1\n",
    "\ufeffa;b\n1,5;2\n",
    "a;b\n1;2\n3;4\n",
    "a,b\n1,2\n",
    'a,b\n"12,5",2\n',
    "a;b\n",
    "a;b",
    "",
    "   \n",
    'nota;peso\n"l1\nl2";1,5\n',
    "a;b;c\n;;\n1,5;;\n",
]


@pytest.mark.parametrize("texto", _FORMAS)
def test_normalizar_dos_veces_es_normalizar_una(texto):
    una = normalize_csv_text(texto)
    assert normalize_csv_text(una) == una
    # y la segunda vez no tiene nada que declarar
    assert normalize_csv_text_declarando(una)[1] == []


_SIN_NADA_QUE_CONVERTIR = [
    "a;b\n1,5;x\nabc;y\n",           # columna mixta
    "a;b\n1.234;1\n2.500;2\n",       # ambigua
    "tasa;n\n12,5 %;1\n",             # porcentaje
    "a;b\n1;2\n3;4\n",               # sin decimales
    "a;b\n",                          # solo cabecera
    "a;b",                             # sin salto final
    "a,b\n1,2\n",                     # separado por comas
    "a,b\n1,2",                        # separado por comas y sin salto final: ni se le añade
    'a,b\n"12,5",2\n',                # coma con comillas en un CSV de comas
    "",
    "   \n",
    "codigo;n\n01,02;1\n",            # código
]


@pytest.mark.parametrize("texto", _SIN_NADA_QUE_CONVERTIR)
def test_un_csv_sin_columnas_que_cumplan_sale_con_los_mismos_bytes_que_antes(texto):
    """Invariante 8: lo que no necesita la regla sale como salía (la función de ANTES, copiada arriba)."""
    assert normalize_csv_text(texto) == _normalize_csv_text_antes(texto)
    assert not _declaradas(normalize_csv_text_declarando(texto)[1], "decimal_coma_convertido")


def _csvs_de_los_tres_repos() -> list[Path]:
    raices = [NUCLEO, NUCLEO.parent / "matrixai-engines", NUCLEO.parent / "matrixaistudio"]
    hallados: list[Path] = []
    for raiz in raices:
        if not raiz.is_dir():
            continue
        for ruta in raiz.rglob("*.csv"):
            partes = set(ruta.parts)
            if partes & {"node_modules", ".git", "__pycache__", ".venv"}:
                continue
            hallados.append(ruta)
    return sorted(hallados)


def test_todos_los_csv_de_los_tres_repos_y_del_catalogo_salen_con_los_mismos_bytes():
    """Criterio 2 de C2: recorre TODO `*.csv` de los tres árboles (pruebas, ejemplos, `ejemplos_medidos` del catálogo,
    plantillas) y compara con la función de ANTES. Uno que cambie solo puede ser un `;` con una columna que cumple la
    regla, y entonces tiene que declararlo. Los que no cambian (todos hoy) son bytes idénticos y el mismo hash."""
    rutas = _csvs_de_los_tres_repos()
    assert len(rutas) >= 50, f"se esperaban decenas de CSV en los tres repos y salieron {len(rutas)}: ¿faltan los hermanos?"
    cambiados = []
    for ruta in rutas:
        try:
            texto = ruta.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        nuevo, declaraciones = normalize_csv_text_declarando(texto)
        antes = _normalize_csv_text_antes(texto)
        if nuevo != antes:
            cambiados.append((ruta, declaraciones))
            assert declaraciones, f"{ruta}: cambió y no lo declara"
        else:
            assert hashlib.sha256(nuevo.encode()).hexdigest() == hashlib.sha256(antes.encode()).hexdigest()
    # Los ÚNICOS que cambian hoy son cuatro fixtures `;` con decimal coma del «¿Funciona aquí?» del Studio
    # (`lectura_compartida`): son justo el caso que la regla existe para convertir (`0,3` es 0,3 y su propio lector, el de
    # TS y el del núcleo, ya lo lee así). Si aparece otro, o falta uno, esta lista se mira a mano.
    assert sorted(r.name for r, _ in cambiados) == [
        "ausentes.csv", "cabecera_vacia.csv", "etiquetas_numericas.csv", "filas_solo_separadores.csv"], \
        [str(r) for r, _ in cambiados]
    for ruta, declaraciones in cambiados:
        assert ruta.parent.name == "lectura_compartida"
        assert ruta.read_text(encoding="utf-8").splitlines()[0].count(";") >= 1  # solo un CSV `;` puede cambiar
        assert any(d["tipo"] == "decimal_coma_convertido" for d in declaraciones)


def test_el_csv_que_genera_el_producto_pasa_por_la_normalizacion_con_los_mismos_bytes():
    """El CSV de R1 (verify.py:671 lo re-normaliza) viene del generador del producto: coma y punto. Nunca cambia."""
    from matrixai.playground_api import generate_synthetic_dataset
    mxai = ("PROJECT P\n\nVECTOR V[2]\n  peso_kg: Scalar\n  edad: Scalar\nEND\n\nNETWORK R\n  INPUT V\n"
            "  LAYER Dense units=4 activation=relu\n  LAYER Dense units=1 activation=linear\n  OUTPUT y: Scalar\nEND\n\n"
            "GRAPH\n  V -> R\nEND\n")
    mxtrain = ("MODEL model.mxai\n\nDATASET D\n  SOURCE csv(\"d.csv\")\n  INPUT V FROM COLUMNS [peso_kg, edad]\n"
               "  TARGET y: Scalar\n  SPLIT train=0.8 validation=0.2 seed=1\n  BATCH size=8\nEND\n\n"
               "LOSS L\n  TYPE mse\n  PREDICTION R\n  TARGET y\nEND\n\nOPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.01\n"
               "  UPDATE R.*\nEND\n\nRUN\n  EPOCHS 2\nEND\n")
    g = generate_synthetic_dataset(mxai, mxtrain, 30, 5, "coherent", False,
                                   field_ranges_override={"peso_kg": (40.0, 120.0), "edad": (18.0, 90.0)})
    assert g.get("ok"), g
    assert normalize_csv_text(g["csv_text"]) == g["csv_text"]


def test_la_optimizacion_de_las_columnas_muertas_decide_y_cuenta_igual_que_la_regla_ingenua():
    """`ClasificadorDeColumnas.ver` deja de contar una columna en cuanto trae una celda no numérica (solo sigue mirando las que
    tienen forma de decimal con coma, para poder decirlo). Eso es una optimización (51.000 × 100: de 22 s a 4): aquí se mide que
    NO cambia nada contra la regla escrita a pelo, celda a celda, sobre tablas al azar con las formas de la calle."""
    import random
    from matrixai.training.dataset_analysis import _NULL_TOKENS
    from matrixai.training.decimal_es import ClasificadorDeColumnas, _CLASES, _clase
    rng = random.Random(2026)
    celdas = ["12,5", "1.234,56", "-3,25", "0,5", "7", "-2", "0", "", "  ", " 4,5 ", "NA", "n/a", "-", "?", "abc", "3.5", "1.234", "12.345.678",
              "01,02", "007", "12,5 %", "7,0%", "1,2,3", "2024-01-01", "05/01/2024", "café", "1e3", "٣", "+1,5"]
    for _ in range(300):
        n_col = rng.randint(1, 4)
        filas = [[rng.choice(celdas) for _ in range(n_col)] for _ in range(rng.randint(1, 12))]
        cl = ClasificadorDeColumnas([f"c{j}" for j in range(n_col)])
        for f in filas:
            cl.ver(f)
        esperado_convertir = set()
        esperado_cuantas = {}
        for j in range(n_col):
            clases = []
            for f in filas:
                nucleo = f[j].strip()
                if not nucleo or nucleo.lower() in _NULL_TOKENS:
                    continue
                clases.append(_CLASES[_clase(nucleo)])
            if "D" in clases and all(c in ("D", "E") for c in clases):
                esperado_convertir.add(j)
            esperado_cuantas[j] = clases.count("D") + clases.count("PCT")
        assert set(cl.a_convertir()) == esperado_convertir, filas
        declaradas = {d["columna"]: d for d in cl.declaraciones()}
        for j in range(n_col):
            d = declaradas.get(f"c{j}")
            if j in esperado_convertir:
                assert d["tipo"] == "decimal_coma_convertido" and d["cuantas"] == esperado_cuantas[j], (filas, j)
            elif d is not None and d["detalle"]["motivo"] in ("no_numerica", "mezcla_con_punto"):
                assert d["cuantas"] == esperado_cuantas[j], (filas, j, d)


# ── 3. El resultado llega al análisis, al entrenamiento y a lo que la pantalla enseña ──────────────────────────────

def test_el_analisis_declara_lo_que_convirtio_y_un_csv_normal_no_gana_ninguna_clave():
    t = "fecha;precio;unidades\n2024-01-01;12,5;3\n2024-01-02;9,75;4\n"
    r = analyze_dataset_csv(t)
    assert r["columns"]["precio"]["type"] == "number"
    d = r["declaraciones_de_entrada"]
    assert [x["columna"] for x in d] == ["precio"] and set(d[0]) == {"tipo", "columna", "cuantas", "detalle", "texto"}
    json.dumps(r)  # viaja por HTTP: tiene que ser JSON
    normal = analyze_dataset_csv("fecha,precio,unidades\n2024-01-01,12.5,3\n2024-01-02,9.75,4\n")
    assert "declaraciones_de_entrada" not in normal


def test_el_entrenamiento_de_siempre_entrena_un_csv_punto_y_coma_con_decimal_coma():
    """Antes de C2 `_submit_training_job` lo RECHAZABA («field peso_kg must be numeric, got '64,3'»): medido con el código de
    HEAD. Aquí entra por la MISMA puerta que el Studio, y el recibo guarda el hash del CSV tal como llegó (`;`, coma)."""
    from matrixai import playground as pg
    import sys
    sys.path.insert(0, str(AQUI / "fixtures" / "decimal_es"))
    import generar_paquete_antes as g
    import random
    rng = random.Random(3)
    filas = ["peso_kg;edad;riesgo"]
    for _ in range(40):
        p = round(rng.uniform(50, 110), 1)
        filas.append(f"{str(p).replace('.', ',')};{rng.randint(20, 80)};{'alto' if p > 85 else 'bajo'}")
    csv_es = "\n".join(filas) + "\n"
    with mock.patch.dict(os.environ, {"MATRIXAI_TRAIN_BACKEND": "stdlib"}):
        r = pg._submit_training_job(g.MXAI, g.MXTRAIN, csv_es, None,
                                    field_ranges={"peso_kg": (40.0, 120.0), "edad": (18.0, 90.0)}, seed=7)
        assert r.get("ok"), r
        job = pg._training_jobs[r["job_id"]]
        tope = time.monotonic() + 120
        while job["status"] == "running" and time.monotonic() < tope:
            time.sleep(0.05)
    assert job["status"] == "done", (job["status"], job.get("error"))
    cap = job["run_provenance"]
    assert cap["dataset_sha256_raw"] == hashlib.sha256(csv_es.encode()).hexdigest()  # el ORIGINAL, `;` y coma
    assert cap["dataset_sha256_prepared"] != cap["dataset_sha256_raw"]


def test_el_texto_sale_en_los_dos_idiomas_y_ninguno_dice_1_celdas():
    casos = [
        "a;b\n1,5;1\n",                      # convertido, singular
        "a;b\n1,5;1\n2,5;2\n",               # convertido, plural
        "a;b\n1,5;1\nabc;2\n",               # no numérica
        "a;b\n1,5;1\n2.5;2\n",               # mezcla con punto
        "a;b\n1.234;1\n",                    # ambigua, singular
        "a;b\n1.234;1\n2.345;2\n",           # ambigua, plural
        "a;b\n12,5 %;1\n",                   # porcentaje, singular
        "a;b\n12,5 %;1\n3,5 %;2\n",          # porcentaje, plural
    ]
    for t in casos:
        _, decl = _norm(t)
        assert decl, t
        for d in decl:
            for idioma in ("es", "en"):
                texto = d["texto"][idioma]
                assert texto.strip() and "{n}" not in texto
                assert " 1 celdas" not in texto and " 1 cells" not in texto and " 1 porcentajes" not in texto \
                    and " 1 percentages" not in texto, texto


# ── 4. Compatibilidad hacia atrás: un paquete de ANTES se verifica igual ───────────────────────────────────────────

def test_un_paquete_de_antes_se_verifica_igual():
    """Criterio 4: el paquete lo generó el código de HEAD ANTES de C2 (ver `fixtures/decimal_es/generar_paquete_antes.py`)
    desde un CSV `;` de Excel, y `informe_antes.json` es lo que ese mismo código dijo al verificarlo. Con el código de
    AHORA cada etapa da el mismo estado, y ninguna da FAIL. Medido: el entrenamiento rechazaba un `;` con decimal coma, así
    que no existe ningún paquete anterior con uno convertible; el CSV del paquete usa `.` y una coma que no es decimal."""
    from matrixai.export.verify import verify_package
    antes = json.loads((AQUI / "fixtures" / "decimal_es" / "informe_antes.json").read_text(encoding="utf-8"))
    paquete = Path(tempfile.mkdtemp(prefix="c150c2-")) / "p"
    shutil.copytree(AQUI / "fixtures" / "decimal_es" / "paquete_antes", paquete)
    for locale in ("es", "en"):
        inf = verify_package(paquete, run_training=True, locale=locale)
        ahora = {n: e.get("status") for n, e in inf["stages"].items()}
        assert ahora == antes["etapas"], (ahora, antes["etapas"])
        assert "FAIL" not in ahora.values()


def test_el_csv_de_excel_del_paquete_de_antes_no_cambia_de_hash_con_el_codigo_nuevo():
    """El `dataset_sha256_prepared` que el paquete de antes guardó es el del CSV normalizado con el código de antes;
    con el de ahora el mismo CSV `;` sale con los mismos bytes (su única coma está en una columna de texto)."""
    import sys
    sys.path.insert(0, str(AQUI / "fixtures" / "decimal_es"))
    import generar_paquete_antes as g
    t = g.csv_de_excel()
    assert normalize_csv_text(t) == _normalize_csv_text_antes(t)
    assert normalize_csv_text_declarando(t)[1] == []


# ── 5. Forma y tamaño reales ─────────────────────────────────────────────────────────────────────────────────────────

def _grande(filas: int, columnas: int) -> str:
    import random
    rng = random.Random(11)
    cab = ";".join(f"c{j}" for j in range(columnas))
    lineas = [cab]
    for i in range(filas):
        lineas.append(";".join((f"{rng.uniform(-100, 1e6):,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")
                                if j % 3 == 0 else str(rng.randint(0, 999)) if j % 3 == 1 else f"t{rng.randint(0, 9)}")
                               for j in range(columnas)))
    return "\n".join(lineas) + "\n"


def test_forma_y_tamano_reales_20000x30_con_su_tope():
    t = _grande(20_000, 30)
    inicio = time.perf_counter()
    salida, decl = normalize_csv_text_declarando(t)
    t1 = time.perf_counter() - inicio
    antes = time.perf_counter()
    _normalize_csv_text_antes(t)
    t0 = time.perf_counter() - antes
    assert len(_declaradas(decl, "decimal_coma_convertido")) == 10, [d["columna"] for d in decl]
    assert t1 < max(6 * t0, 8.0), f"la regla cuesta demasiado: {t1:.1f} s contra {t0:.1f} s de antes"
    # lo convertido se re-lee como número
    fila = next(csv.reader(io.StringIO(salida.split("\n", 2)[1])))
    float(fila[0])
