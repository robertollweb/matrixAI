# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El análisis de un CSV juzga cada VALOR DISTINTO una vez, no cada celda (2026-09-27).

Medido con 51.000 filas × 100 columnas: `analyze_dataset_csv` tardaba 12,8 s con flotantes
de dos decimales y 14,5 s con cuatro (el peor caso: casi todo valor distinto), y ahora 3,4
y 8,2 s. Casi todo se iba en preguntas hechas celda a celda —«¿es ausente?», «¿parsea
como número?», «¿tiene un cero delante?», «¿es fecha?», «¿es de sí/no?»— cuya respuesta es
la misma sobre los valores distintos. El resultado es IDÉNTICO: comparado contra el
módulo de antes en 1.500 CSV al azar con las formas difíciles × 4 declaraciones de
ausencia, 0 diferencias (y el banco de esa comparación, saboteado tres veces, las veía).

Estas pruebas atan el COSTE por recuento, no por reloj —el reloj de esta máquina
compartida no es un instrumento—, y lo que el cambio podía romper: que lo que depende de
las CELDAS (cuántas faltan) se siga contando por celdas."""
from __future__ import annotations

import pytest

from matrixai.training import dataset_analysis
from matrixai.training.dataset_analysis import _is_boolean_column, analyze_dataset_csv

FILAS = 3000


def _csv_repetido() -> str:
    """3.000 filas y MUY pocos valores distintos por columna: donde juzgar por celda y
    juzgar por valor se separan por tres órdenes de magnitud."""
    lineas = ["id,edad,zona,activo,nota"]
    for i in range(FILAS):
        lineas.append(",".join([str(i + 1), str(20 + i % 5), ("norte", "sur", "NA")[i % 3],
                                ("sí", "no")[i % 2], ("", "3.5", "4.25", " na ")[i % 4]]))
    return "\n".join(lineas) + "\n"


def test_cada_valor_distinto_se_pregunta_si_es_ausente_UNA_vez(monkeypatch):
    real = dataset_analysis._is_null
    preguntas: list[object] = []

    def contado(valor, tokens_de_ausencia=None):
        preguntas.append(valor)
        return real(valor, tokens_de_ausencia)

    monkeypatch.setattr(dataset_analysis, "_is_null", contado)
    analyze_dataset_csv(_csv_repetido())
    # id: 3.000 distintos; edad 5; zona 3; activo 2; nota 4. Por celda serían 15.000.
    distintos_por_columna = FILAS + 5 + 3 + 2 + 4
    assert len(preguntas) == distintos_por_columna, len(preguntas)


def test_el_tipo_numerico_se_decide_con_los_valores_distintos(monkeypatch):
    real = dataset_analysis._numeric_kind
    tamanos: list[int] = []

    def contado(valores):
        tamanos.append(len(valores))
        return real(valores)

    monkeypatch.setattr(dataset_analysis, "_numeric_kind", contado)
    analisis = analyze_dataset_csv(_csv_repetido())
    assert analisis["columns"]["edad"]["type"] == "integer"
    assert analisis["columns"]["nota"]["type"] == "number"
    # Una llamada por columna que llega hasta ahí (`activo` es de sí/no y sale antes), con
    # SUS valores distintos no ausentes: `zona` 2, `nota` 2, `edad` 5 e `id` 3.000. Por
    # celda serían 3.000, 3.000, 1.500 y 2.000.
    assert sorted(tamanos) == [2, 2, 5, FILAS], tamanos


def test_la_columna_de_si_no_se_para_en_el_primer_valor_que_no_lo_es():
    """El segundo valor NO es una cadena: si la función siguiera recorriendo tras un «3,5»
    que ya no es de sí/no, reventaría en él. Parar en el primero es lo que la hace barata
    en toda columna numérica."""
    assert _is_boolean_column(["3.5", None]) is False  # type: ignore[list-item]
    assert _is_boolean_column(["sí", "no", "tal vez", None]) is False  # type: ignore[list-item]
    assert _is_boolean_column(["Sí", " no ", "SÍ"]) is True
    # «Sí» y «SI» son DOS palabras en minúsculas («sí», «si»): con «no», tres.
    assert _is_boolean_column(["Sí", " no ", "SI"]) is False


def test_las_ausencias_se_CUENTAN_por_celda_aunque_se_juzguen_por_valor():
    """Lo que el cambio podía romper: «NA» se pregunta una vez, pero falta en MIL celdas."""
    analisis = analyze_dataset_csv(_csv_repetido())
    nota = analisis["columns"]["nota"]
    # «» y « na » son cada uno 1 de cada 4 filas: 1.500 ausentes de 3.000.
    assert (nota["null_count"], nota["null_ratio"]) == (1500, 0.5), nota
    zona = analisis["columns"]["zona"]
    assert (zona["null_count"], zona["cardinality"]) == (1000, 2), zona


@pytest.mark.parametrize("tokens,esperado", [(None, 1500), ({"?"}, 0), ({""}, 750)])
def test_la_ausencia_declarada_tambien_se_cuenta_por_celda(tokens, esperado):
    nota = analyze_dataset_csv(_csv_repetido(), tokens_de_ausencia=tokens)["columns"]["nota"]
    assert nota["null_count"] == esperado, (tokens, nota)


def test_las_filas_duplicadas_se_cuentan_como_antes():
    texto = "a,b\n1,x\n1,x\n2,y\n1,x\n2,y\n3,z\n"
    assert analyze_dataset_csv(texto)["duplicate_rows"] == 3
