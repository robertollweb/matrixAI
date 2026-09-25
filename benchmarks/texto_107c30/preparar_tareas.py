# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — construye las cuatro tareas del pre-registro como tablas.

Lee el borrador de pre-registro (2026-09-22) y el bloque «Completado el
2026-09-24 (sin medir modelos)» de
`documentacion/107_TEXTO_CON_PREENTRENADOS_CONTRACT.md`, y hace lo que ese
bloque dejaba pendiente de verdad: construir las tablas, no solo contarlas.

Las cuatro tareas:

  A. FakeNewsCorpusSpanish — binaria por `Category` (verdadera/falsa).
     v1 (`train.xlsx` + `development.xlsx`) trae `Category` como texto
     ("Fake"/"True"); v2 (`test.xlsx`) trae columnas en MAYÚSCULAS
     (`CATEGORY`, `TOPICS`...) y `CATEGORY` es un booleano DE EXCEL de
     verdad (tipo de celda "b"), no la cadena de v1. Es la trampa escrita
     en el contrato — quien concatene sin normalizar el tipo primero no
     completa el estudio o lo completa mal (el mismo patrón que el
     BLOQUEANTE de 0/1 y Sí/No del 23-09, `99dd97e`).
  B. CodiEsp — binaria por el código CIE-10 `r52` (regla ya fijada y medida
     el 24-09: prevalencia train más cercana a 30%, mínimo 10 positivos en
     test). Sin otras columnas: mide la representación, no el estudio con
     columnas (así lo declara el propio pre-registro).
  C. PLACSP — regresión sobre `log10(importe_sin_iva)`, solo filas en
     castellano según `regla_idioma.json` (reescrita en este corte: la
     lista exacta que usó la sesión del 24-09 se perdió, ver ese fichero).
  D. BOE «Legislación consolidada» — binaria por `vigencia_agotada`,
     también filtrada por la misma regla de idioma.

Cada tarea se escribe en `tareas/tarea_<letra>.csv` con una columna
`particion` (train/dev/test). Para A y B la partición es la NATURAL del
corpus (ya viene partida en train/dev/test de origen). Para C y D no hay
partición de origen, así que se fija aquí con una semilla declarada
(`SEMILLA_PARTICION`) y se guarda en el propio CSV — `medir_c30.py` no
vuelve a barajar nada, lee la columna.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent
DATOS = Path("/home/deployer/datos_107c30")
SALIDA_TAREAS = RAIZ / "tareas"
REGLA_IDIOMA = RAIZ / "regla_idioma.json"
RESUMEN = RAIZ / "resumen_tareas.json"

SEMILLA_PARTICION = 107030  # 107-C3.0, declarada aquí, no oculta en el código
REPARTO_PARTICION = (0.70, 0.15, 0.15)  # train / dev / test, para C y D

#: 105-C2, reutilizado (línea ~130 de 105_EVALUACION_PARA_DECIDIR_CONTRACT.md):
#: mínimo de casos positivos en test para que un segmento cuente.
MINIMO_EVENTOS_POR_SEGMENTO = 10


# ---------------------------------------------------------------------------
# utilidades comunes
# ---------------------------------------------------------------------------

def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _escribir_csv(ruta: Path, filas: list[dict[str, Any]], columnas: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for fila in filas:
            w.writerow(fila)


def _particion_fija(row_ids: list[str], *, semilla: int) -> dict[str, str]:
    """Train/dev/test reproducible: MISMA función para C y D, una sola vez."""
    ids = sorted(row_ids)  # orden determinista ANTES de barajar
    rng = random.Random(semilla)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * REPARTO_PARTICION[0])
    n_dev = int(n * REPARTO_PARTICION[1])
    asignacion = {}
    for i, rid in enumerate(ids):
        if i < n_train:
            asignacion[rid] = "train"
        elif i < n_train + n_dev:
            asignacion[rid] = "dev"
        else:
            asignacion[rid] = "test"
    return asignacion


# ---------------------------------------------------------------------------
# lector de .xlsx mínimo (sin depender de que openpyxl siga instalado): usa
# openpyxl si está, y si no, falla con un motivo claro. openpyxl NO viene con
# matrixAI/matrixai-engines (dependencies=[] es invariante del núcleo); aquí
# es un benchmark fuera de esos paquetes y se declara la dependencia extra.
# ---------------------------------------------------------------------------

def _leer_xlsx(ruta: Path) -> list[dict[str, Any]]:
    """Cada fila como dict {columna: (valor, tipo_de_celda)}.

    `tipo_de_celda` es el que da openpyxl ('s' texto, 'b' booleano, 'n'
    número...) — es lo que hace falta para detectar el booleano de Excel de
    v2 SIN adivinar por el valor.
    """
    try:
        import openpyxl
    except ImportError as e:
        raise SystemExit(
            "falta 'openpyxl' para leer los .xlsx de FakeNewsCorpusSpanish "
            "(pip install --user openpyxl)"
        ) from e
    wb = openpyxl.load_workbook(str(ruta), read_only=True, data_only=True)
    ws = wb.active
    filas_it = ws.iter_rows()
    cabecera = [c.value for c in next(filas_it)]
    filas = []
    for r in filas_it:
        if all(c.value is None for c in r):
            continue
        fila = {}
        for nombre, celda in zip(cabecera, r):
            fila[nombre] = (celda.value, celda.data_type)
        filas.append(fila)
    wb.close()
    return filas


# ---------------------------------------------------------------------------
# A · FakeNewsCorpusSpanish
# ---------------------------------------------------------------------------

def _normalizar_category(valor: Any, tipo: str, *, origen: str) -> str:
    """La trampa escrita en el contrato: v2 trae un BOOLEANO de Excel, no la
    cadena "Fake"/"True" de v1. Sin normalizar esto antes de concatenar, un
    consumidor que compare `== "Fake"` se come en silencio TODO v2 (un
    booleano `False` nunca es igual a la cadena "Fake"), o al revés si
    alguien lo convierte con `str()` a "True"/"False" en vez de "Fake"/"True"
    -- que es justo el patrón del BLOQUEANTE de 0/1 y Sí/No del 23-09.

    Se normaliza a las MISMAS dos cadenas que usa v1 ("Fake"/"True"), que son
    las que the propio README del corpus usa. El booleano de Excel `True`
    corresponde a "True" (noticia real) y `False` a "Fake": comprobado leyendo
    las tres primeras filas de test.xlsx contra la fuente citada en la
    cabecera (El Economista/El País -> True; una nota sin fuente verificable
    de "El matinal" sobre una ley que no existe -> False).
    """
    if tipo == "b":
        if not isinstance(valor, bool):
            raise ValueError(f"{origen}: celda declarada booleana sin valor bool: {valor!r}")
        return "True" if valor else "Fake"
    if tipo == "s":
        texto = str(valor).strip()
        if texto not in ("Fake", "True"):
            raise ValueError(f"{origen}: Category de texto fuera de {{Fake,True}}: {texto!r}")
        return texto
    raise ValueError(f"{origen}: tipo de celda inesperado para Category/CATEGORY: {tipo!r}")


def tarea_a() -> dict[str, Any]:
    base = DATOS / "FakeNewsCorpusSpanish"
    ficheros = {
        "train": base / "train.xlsx",
        "development": base / "development.xlsx",
        "test": base / "test.xlsx",
    }
    for r in ficheros.values():
        if not r.is_file():
            raise SystemExit(f"falta {r}")

    filas_salida: list[dict[str, Any]] = []
    contador = 0

    # v1: train.xlsx (particion=train) + development.xlsx (particion=dev)
    for particion, clave in (("train", "train"), ("dev", "development")):
        for fila in _leer_xlsx(ficheros[clave]):
            valor_cat, tipo_cat = fila["Category"]
            categoria = _normalizar_category(valor_cat, tipo_cat, origen=f"v1/{clave}")
            contador += 1
            filas_salida.append({
                "row_id": f"a-v1-{clave}-{contador}",
                "particion": particion,
                "target": categoria,
                "texto": f"{fila['Headline'][0]}\n\n{fila['Text'][0]}",
                "topic": fila["Topic"][0],
                "source": fila["Source"][0],
                "version_origen": "v1",
            })

    # v2: test.xlsx (particion=test), columnas en MAYÚSCULAS, CATEGORY booleano
    for fila in _leer_xlsx(ficheros["test"]):
        valor_cat, tipo_cat = fila["CATEGORY"]
        categoria = _normalizar_category(valor_cat, tipo_cat, origen="v2/test")
        contador += 1
        headline = fila["HEADLINE"][0] or ""
        texto_cuerpo = fila["TEXT"][0] or ""
        filas_salida.append({
            "row_id": f"a-v2-test-{contador}",
            "particion": "test",
            "target": categoria,
            "texto": f"{headline}\n\n{texto_cuerpo}",
            "topic": fila["TOPICS"][0],
            "source": fila["SOURCE"][0],
            "version_origen": "v2",
        })

    columnas = ["row_id", "particion", "target", "texto", "topic", "source", "version_origen"]
    _escribir_csv(SALIDA_TAREAS / "tarea_a.csv", filas_salida, columnas)

    n_por_particion = {p: sum(1 for f in filas_salida if f["particion"] == p) for p in ("train", "dev", "test")}
    n_por_target = {}
    for p in ("train", "dev", "test"):
        n_por_target[p] = {
            t: sum(1 for f in filas_salida if f["particion"] == p and f["target"] == t)
            for t in ("Fake", "True")
        }
    return {
        "tarea": "A",
        "nombre": "FakeNewsCorpusSpanish",
        "n_filas": len(filas_salida),
        "n_por_particion": n_por_particion,
        "n_por_particion_y_target": n_por_target,
        "objetivo": {
            "columna": "target",
            "tarea_del_estudio": "binary_classification",
            "clases": ["True", "Fake"],
            "positive_label": "Fake",
            "nota": "positive_label='Fake' -- se enmarca como DETECTAR noticias falsas, que es la "
                    "lectura natural del corpus; True/Fake son las DOS cadenas que usa el propio "
                    "corpus (v1 nativamente, v2 tras normalizar su booleano de Excel).",
        },
        "columna_texto": "texto (Headline + Text; HEADLINE + TEXT en v2)",
        "otras_columnas": ["topic", "source"],
        "procedencia": [
            {"fichero": str(ficheros[k]), "sha256": sha256_de(ficheros[k])} for k in ("train", "development", "test")
        ],
        "particion": "NATURAL del corpus: train.xlsx=train, development.xlsx=dev, test.xlsx=test",
        "trampa_normalizada": "CATEGORY de v2 es booleano de Excel (tipo de celda 'b'); se normaliza a "
                               "las cadenas 'True'/'Fake' de v1 en _normalizar_category(), NUNCA con "
                               "str(valor_booleano) (que daría 'True'/'False', rompiendo el vocabulario).",
    }


# ---------------------------------------------------------------------------
# B · CodiEsp — binaria por r52
# ---------------------------------------------------------------------------

CODIGO_CODIESP = "r52"  # fijado el 24-09 (prevalencia train más cercana a 30%, ver contrato)


def _leer_d_tsv(ruta: Path) -> dict[str, set[str]]:
    """articleID -> conjunto de códigos de diagnóstico (columna D)."""
    salida: dict[str, set[str]] = {}
    with ruta.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.rstrip("\n")
            if not linea:
                continue
            partes = linea.split("\t")
            if len(partes) < 2:
                continue
            articulo, codigo = partes[0], partes[1]
            salida.setdefault(articulo, set()).add(codigo.strip().lower())
    return salida


def tarea_b() -> dict[str, Any]:
    base = DATOS / "codiesp" / "final_dataset_v4_to_publish"
    splits = {
        "train": (base / "train" / "trainD.tsv", base / "train" / "text_files"),
        "dev": (base / "dev" / "devD.tsv", base / "dev" / "text_files"),
        "test": (base / "test" / "testD.tsv", base / "test" / "text_files"),
    }
    for tsv, txt_dir in splits.values():
        if not tsv.is_file() or not txt_dir.is_dir():
            raise SystemExit(f"falta {tsv} o {txt_dir}")

    filas_salida: list[dict[str, Any]] = []
    resumen_particion: dict[str, dict[str, int]] = {}
    for particion, (tsv, txt_dir) in splits.items():
        por_caso = _leer_d_tsv(tsv)
        casos = sorted(p.stem for p in txt_dir.glob("*.txt"))
        n_pos = 0
        for caso in casos:
            codigos = por_caso.get(caso, set())
            positivo = CODIGO_CODIESP in codigos
            n_pos += int(positivo)
            texto = (txt_dir / f"{caso}.txt").read_text(encoding="utf-8")
            filas_salida.append({
                "row_id": f"b-{particion}-{caso}",
                "particion": particion,
                "target": "r52" if positivo else "no_r52",
                "texto": texto,
            })
        resumen_particion[particion] = {"n": len(casos), "n_r52": n_pos}

    columnas = ["row_id", "particion", "target", "texto"]
    _escribir_csv(SALIDA_TAREAS / "tarea_b.csv", filas_salida, columnas)

    n_test_pos = resumen_particion["test"]["n_r52"]
    if n_test_pos < MINIMO_EVENTOS_POR_SEGMENTO:
        raise SystemExit(
            f"tarea B: solo {n_test_pos} positivos de '{CODIGO_CODIESP}' en test, por debajo del "
            f"mínimo declarado ({MINIMO_EVENTOS_POR_SEGMENTO}, 105-C2)"
        )

    ficheros_origen = [base / s / f"{s}D.tsv" for s in ("train", "dev", "test")]
    return {
        "tarea": "B",
        "nombre": "CodiEsp",
        "n_filas": len(filas_salida),
        "n_por_particion": {p: v["n"] for p, v in resumen_particion.items()},
        "n_por_particion_y_target": {
            p: {"r52": v["n_r52"], "no_r52": v["n"] - v["n_r52"]} for p, v in resumen_particion.items()
        },
        "objetivo": {
            "columna": "target",
            "tarea_del_estudio": "binary_classification",
            "clases": ["no_r52", "r52"],
            "positive_label": "r52",
            "nota": f"código CIE-10 '{CODIGO_CODIESP}' (\"Dolor, no clasificado en otra parte\"), "
                    "fijado el 24-09: prevalencia train más cercana a 30% entre los códigos de "
                    "diagnóstico, desempate alfabético, con >=10 positivos en test.",
        },
        "columna_texto": "texto (el caso clínico completo)",
        "otras_columnas": [],
        "otras_columnas_nota": "el pre-registro declara esta tarea SIN otras columnas a propósito "
                                "(\"es texto solo: mide la representación, no el estudio con "
                                "columnas\"); la condición (0) de medir_c30.py degenera aquí en un "
                                "baseline SIN predictores (ver ese script).",
        "procedencia": [{"fichero": str(f), "sha256": sha256_de(f)} for f in ficheros_origen],
        "particion": "NATURAL del corpus CodiEsp (500 train / 250 dev / 250 test, de origen)",
    }


# ---------------------------------------------------------------------------
# regla de idioma — compartida por C y D
# ---------------------------------------------------------------------------

_RE_PALABRA = re.compile(r"\b[^\W\d_]+\b", re.UNICODE)


def _sin_diacriticos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def cargar_regla_idioma(ruta: Path = REGLA_IDIOMA) -> dict[str, Any]:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _marcadores_normalizados(regla: dict[str, Any]) -> dict[str, set[str]]:
    salida = {}
    for codigo, bloque in regla["idiomas"].items():
        salida[codigo] = {_sin_diacriticos(m.lower()) for m in bloque["marcadores"]}
    return salida


def detectar_idioma(texto: str, regla: dict[str, Any], *,
                    _marcadores_cache: dict[str, set[str]] | None = None) -> str:
    """'es' por omisión; el código de idioma si >=umbral marcadores exclusivos
    y distintos de ESE idioma aparecen como palabra completa en el texto."""
    marcadores = _marcadores_cache if _marcadores_cache is not None else _marcadores_normalizados(regla)
    normalizado = _sin_diacriticos((texto or "").lower())
    palabras = set(_RE_PALABRA.findall(normalizado))
    umbral = regla["umbral_marcadores_distintos"]
    cuentas = {}
    for codigo, marcs in marcadores.items():
        n = len(palabras & marcs)
        if n >= umbral:
            cuentas[codigo] = n
    if not cuentas:
        return "es"
    maximo = max(cuentas.values())
    ganadores = sorted(c for c, n in cuentas.items() if n == maximo)
    return ganadores[0]


# ---------------------------------------------------------------------------
# C · PLACSP — regresión sobre log10(importe_sin_iva)
# ---------------------------------------------------------------------------

def tarea_c(regla: dict[str, Any]) -> dict[str, Any]:
    fichero = DATOS / "placsp" / "placsp_202508_dedup.jsonl"
    if not fichero.is_file():
        raise SystemExit(f"falta {fichero}")

    marcadores = _marcadores_normalizados(regla)
    total = 0
    n_castellano = 0
    n_otros: dict[str, int] = {}
    filas_candidatas: list[dict[str, Any]] = []
    for linea in fichero.open("r", encoding="utf-8"):
        linea = linea.strip()
        if not linea:
            continue
        total += 1
        registro = json.loads(linea)
        objeto = registro.get("objeto") or ""
        idioma = detectar_idioma(objeto, regla, _marcadores_cache=marcadores)
        if idioma != "es":
            n_otros[idioma] = n_otros.get(idioma, 0) + 1
            continue
        n_castellano += 1
        try:
            importe = float(registro.get("importe_sin_iva") or 0)
        except (TypeError, ValueError):
            importe = 0.0
        if importe <= 0:
            continue
        cpv = registro.get("cpv") or []
        cpv_division = str(cpv[0])[:2] if cpv else "NA"
        filas_candidatas.append({
            "row_id": registro["id"],
            "target": math.log10(importe),
            "texto": objeto,
            "tipo_code": registro.get("tipo_code") or "NA",
            "procedimiento_code": registro.get("procedimiento_code") or "NA",
            "cpv_division": cpv_division,
            "organo": registro.get("organo") or "NA",
        })

    asignacion = _particion_fija([f["row_id"] for f in filas_candidatas], semilla=SEMILLA_PARTICION)
    for f in filas_candidatas:
        f["particion"] = asignacion[f["row_id"]]

    columnas = ["row_id", "particion", "target", "texto", "tipo_code", "procedimiento_code",
                "cpv_division", "organo"]
    _escribir_csv(SALIDA_TAREAS / "tarea_c.csv", filas_candidatas, columnas)

    n_por_particion = {p: sum(1 for f in filas_candidatas if f["particion"] == p) for p in ("train", "dev", "test")}

    return {
        "tarea": "C",
        "nombre": "PLACSP (agosto 2025)",
        "n_expedientes_totales_del_mes": total,
        "n_castellano_regla_actual": n_castellano,
        "n_otros_idiomas_regla_actual": n_otros,
        "comparacion_con_24_09": {
            "castellano_24_09": 31162,
            "total_24_09": 31210,
            "castellano_ahora": n_castellano,
            "total_ahora": total,
            "nota": "la lista de marcadores del 24-09 se perdió (scratchpad/107c30 vivía fuera del "
                    "repo, en /tmp); esta es una reconstrucción DESDE CERO con el mismo principio, "
                    "no la misma lista literal -- una diferencia en el recuento es ESPERABLE y no "
                    "se ha retocado la lista para intentar igualarla.",
        },
        "n_filas_con_importe_positivo": len(filas_candidatas),
        "n_por_particion": n_por_particion,
        "objetivo": {
            "columna": "target",
            "tarea_del_estudio": "regression",
            "transformacion": "log10(importe_sin_iva)",
            "nota": "solo filas con importe_sin_iva > 0 (igual que el 30.591/31.162 del 24-09).",
        },
        "columna_texto": "texto (objeto del contrato)",
        "otras_columnas": ["tipo_code", "procedimiento_code", "cpv_division", "organo"],
        "procedencia": [{
            "fichero": str(fichero), "sha256": sha256_de(fichero),
            "fichero_origen_zip": str(DATOS / "placsp" / "licitacionesPerfilesContratanteCompleto3_202508.zip"),
            "sha256_zip": sha256_de(DATOS / "placsp" / "licitacionesPerfilesContratanteCompleto3_202508.zip"),
        }],
        "particion": f"FIJADA aquí (no hay partición de origen): semilla {SEMILLA_PARTICION}, "
                     f"reparto {REPARTO_PARTICION}, guardada en la columna 'particion' del CSV.",
    }


# ---------------------------------------------------------------------------
# D · BOE — binaria por vigencia_agotada
# ---------------------------------------------------------------------------

def tarea_d(regla: dict[str, Any]) -> dict[str, Any]:
    fichero = DATOS / "boe" / "legislacion_consolidada_muestra5000.json"
    if not fichero.is_file():
        raise SystemExit(f"falta {fichero}")

    datos = json.loads(fichero.read_text(encoding="utf-8"))
    registros = datos["data"]
    marcadores = _marcadores_normalizados(regla)

    total = len(registros)
    n_castellano = 0
    n_otros: dict[str, int] = {}
    filas_candidatas: list[dict[str, Any]] = []
    for r in registros:
        titulo = r.get("titulo") or ""
        idioma = detectar_idioma(titulo, regla, _marcadores_cache=marcadores)
        if idioma != "es":
            n_otros[idioma] = n_otros.get(idioma, 0) + 1
            continue
        n_castellano += 1
        vigencia = (r.get("vigencia_agotada") or "").strip().upper()
        if vigencia not in ("S", "N"):
            continue
        fecha_pub = r.get("fecha_publicacion") or ""
        anio = fecha_pub[:4] if len(fecha_pub) >= 4 and fecha_pub[:4].isdigit() else "NA"
        filas_candidatas.append({
            "row_id": r["identificador"],
            "target": "agotada" if vigencia == "S" else "vigente",
            "texto": titulo,
            "departamento": (r.get("departamento") or {}).get("texto") or "NA",
            "rango": (r.get("rango") or {}).get("texto") or "NA",
            "ambito": (r.get("ambito") or {}).get("texto") or "NA",
            "anio_publicacion": anio,
        })

    asignacion = _particion_fija([f["row_id"] for f in filas_candidatas], semilla=SEMILLA_PARTICION)
    for f in filas_candidatas:
        f["particion"] = asignacion[f["row_id"]]

    columnas = ["row_id", "particion", "target", "texto", "departamento", "rango", "ambito",
                "anio_publicacion"]
    _escribir_csv(SALIDA_TAREAS / "tarea_d.csv", filas_candidatas, columnas)

    n_por_particion = {p: sum(1 for f in filas_candidatas if f["particion"] == p) for p in ("train", "dev", "test")}
    n_pos_total = sum(1 for f in filas_candidatas if f["target"] == "agotada")

    return {
        "tarea": "D",
        "nombre": "BOE - Legislación consolidada (muestra 5000)",
        "n_registros_muestra": total,
        "n_castellano_regla_actual": n_castellano,
        "n_otros_idiomas_regla_actual": n_otros,
        "comparacion_con_24_09": {
            "castellano_24_09": 5000, "total_24_09": 5000,
            "castellano_ahora": n_castellano, "total_ahora": total,
            "nota": "misma advertencia que en C: regla reconstruida desde cero, no la lista literal "
                    "del 24-09.",
        },
        "n_filas_con_vigencia_valida": len(filas_candidatas),
        "n_positivos_agotada": n_pos_total,
        "n_por_particion": n_por_particion,
        "objetivo": {
            "columna": "target", "tarea_del_estudio": "binary_classification",
            "clases": ["vigente", "agotada"], "positive_label": "agotada",
        },
        "columna_texto": "texto (titulo)",
        "otras_columnas": ["departamento", "rango", "ambito", "anio_publicacion"],
        "procedencia": [{"fichero": str(fichero), "sha256": sha256_de(fichero)}],
        "particion": f"FIJADA aquí (no hay partición de origen): semilla {SEMILLA_PARTICION}, "
                     f"reparto {REPARTO_PARTICION}, guardada en la columna 'particion' del CSV.",
        "declarado_en_el_contrato": "cuarta tarea NO forzada: 'una candidata razonable... la muestra "
                                     "es una llamada suelta a la API, no un mes reproducible como en "
                                     "C'. Se construye igualmente aquí porque el encargo pide "
                                     "'cuatro tareas', con esa reserva heredada y repetida.",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    if not REGLA_IDIOMA.is_file():
        raise SystemExit(f"falta {REGLA_IDIOMA} (escríbela antes de contar)")
    regla = cargar_regla_idioma()

    resumen = {
        "contrato": "107_TEXTO_CON_PREENTRENADOS_CONTRACT.md, C3.0",
        "regla_idioma_fichero": str(REGLA_IDIOMA),
        "regla_idioma_sha256": sha256_de(REGLA_IDIOMA),
        "semilla_particion_c_d": SEMILLA_PARTICION,
        "reparto_particion_c_d": list(REPARTO_PARTICION),
        "tareas": {},
    }

    for letra, funcion, args in (
        ("A", tarea_a, ()),
        ("B", tarea_b, ()),
        ("C", tarea_c, (regla,)),
        ("D", tarea_d, (regla,)),
    ):
        print(f"=== tarea {letra} ===", file=sys.stderr)
        info = funcion(*args)
        resumen["tareas"][letra] = info
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str), file=sys.stderr)

    RESUMEN.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nresumen escrito en {RESUMEN}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
