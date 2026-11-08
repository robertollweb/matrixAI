# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C5 remate (I2 de la auditoría) — «¿Funciona aquí?» con un libro de Excel: el INFORME dice de dónde salió el CSV.

Antes el Anexo D decía «Fichero casos.xlsx · Huella del CSV · Lectura del CSV: utf-8 · separador `,`» y el PROBAST «CSV aportado
por quien evalúa: casos.xlsx»: ni Excel, ni la hoja, ni la fila TOTAL que la persona quitó (invariante 3: nada se quita sin quedar
declarado; declarar lo que PASÓ). Cada prueba lleva el nombre de lo que sostiene.
"""
from __future__ import annotations

import hashlib
import random

import pytest

from matrixai.estudio.evaluacion_de_puntuacion import (
    DeclaracionDePuntuacion, EvaluacionRechazada, evaluar_puntuacion,
)
from matrixai.export.informe_de_puntuacion import informe_html, items_tripod

R2 = dict(horizonte_del_desenlace="durante el ingreso", momento_de_la_puntuacion="al ingreso", donde_se_calcula="local")


def _csv() -> str:
    r = random.Random(3)
    filas = ["p,y"]
    for _ in range(300):
        y = 1 if r.random() < 0.3 else 0
        filas.append(f"{min(0.99, max(0.01, r.gauss(0.6 if y else 0.3, 0.2))):.3f},{y}")
    return "\n".join(filas) + "\n"


CSV = _csv()
HUELLA_CSV = hashlib.sha256(CSV.encode("utf-8")).hexdigest()
ORIGEN = {
    "esquema": 1, "formato": "xlsx", "nombre": "casos.xlsx", "sha256": "a" * 64, "bytes": 4096, "hoja": "Pedidos",
    "fila_de_cabecera": 3, "filas_de_cabecera": 1, "rango": "A1:B310", "sha256_del_csv": HUELLA_CSV,
    "filas_excluidas": [{"fila": 124, "texto": "TOTAL"}],
    "declaraciones": [
        {"tipo": "filas_excluidas", "columna": None, "cuantas": 1, "detalle": {},
         "texto": {"es": "Se quitó la fila 124 («TOTAL») a petición de quien subió el libro.",
                   "en": "Row 124 (“TOTAL”) was removed at the request of whoever uploaded the workbook."}},
    ],
    # lo que el CLIENTE no puede decir: lo escribe el núcleo
    "vinculo": "comprobado", "informativo": False, "nota": "esta huella SÍ se verifica",
}


def _evaluar(csv=CSV, **kw):
    return evaluar_puntuacion(
        csv.encode("utf-8"), columna_desenlace="y", valor_del_evento="1", remuestras=40, muestra_representativa=True,
        puntuaciones=[DeclaracionDePuntuacion("p", "probabilidad", "mas_alto_mas_riesgo", casos_en_su_desarrollo="no")],
        nombre_del_fichero="casos.xlsx", **R2, **kw)


def test_el_informe_de_un_libro_dice_que_vino_de_Excel_la_hoja_y_lo_que_se_quito_en_los_dos_idiomas():
    r = _evaluar(origen_del_fichero=ORIGEN)
    for idioma, excel, hoja, frase in (("es", "Excel", "Pedidos", "Se quitó la fila 124"), ("en", "Excel", "Pedidos", "Row 124")):
        html = informe_html(r, idioma=idioma)
        assert excel in html and hoja in html and frase in html, idioma
        assert "Origen del fichero" in html if idioma == "es" else "File origin" in html
    texto = r.informe["entrada"]["origen_del_fichero_texto"]
    assert "Pedidos" in texto["linea"]["es"] and "sheet" in texto["linea"]["en"]
    assert texto["frases"] == [{"es": ORIGEN["declaraciones"][0]["texto"]["es"], "en": ORIGEN["declaraciones"][0]["texto"]["en"]}]


def test_el_PROBAST_no_llama_CSV_al_libro_y_cita_el_origen():
    r = _evaluar(origen_del_fichero=ORIGEN)
    fuente = next(i for i in items_tripod(r, "es") if i.clave == "fuente_de_los_datos")
    assert "Libro de Excel" in fuente.dato and "Pedidos" in fuente.dato and "Se quitó la fila 124" in fuente.dato
    assert not fuente.dato.startswith("CSV aportado")
    fuente_en = next(i for i in items_tripod(r, "en") if i.clave == "fuente_de_los_datos")
    assert "Excel workbook" in fuente_en.dato and "Row 124" in fuente_en.dato


def test_el_vinculo_lo_escribe_el_nucleo_no_el_cliente():
    ok = _evaluar(origen_del_fichero=ORIGEN).informe["entrada"]["origen_del_fichero"]
    assert ok["vinculo"] == "comprobado" and "informativo" not in ok and "nota" not in ok
    otro = _evaluar(origen_del_fichero={**ORIGEN, "sha256_del_csv": "f" * 64}).informe["entrada"]["origen_del_fichero"]
    assert otro["vinculo"] == "no_comprobado"  # el cliente decía «comprobado» y la huella no es la de este CSV


def test_sin_libro_el_informe_no_inventa_origen_y_es_el_de_siempre():
    r = _evaluar()
    assert "origen_del_fichero" not in r.informe["entrada"] and "origen_del_fichero_texto" not in r.informe["entrada"]
    html = informe_html(r, idioma="es")
    assert "Origen del fichero" not in html
    fuente = next(i for i in items_tripod(r, "es") if i.clave == "fuente_de_los_datos")
    assert fuente.dato.startswith("CSV aportado por quien evalúa: casos.xlsx")


@pytest.mark.parametrize("malo", [{"formato": "pdf"}, {"sha256": "corta"}, {"esquema": 99}])
def test_un_origen_mal_formado_se_rechaza_con_su_motivo(malo):
    with pytest.raises(EvaluacionRechazada) as e:
        _evaluar(origen_del_fichero={**ORIGEN, **malo})
    assert e.value.es and e.value.en
