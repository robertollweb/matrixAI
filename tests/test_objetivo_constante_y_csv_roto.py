# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Dos hallazgos MEDIDOS conduciendo el producto el 2026-08-13.

1. **Un objetivo CONSTANTE entrenaba hasta pérdida 0,000 y todo el
   producto lo daba por perfecto.** CSV de 300 filas con `y = 7` siempre,
   elegido como objetivo: `Pérdida 0.000`, `Error de validación 0.000`,
   raíl «Entrenamiento completado», auditoría «6 pasan» y despliegue
   disponible. El core ya rechazaba la columna constante como FEATURE
   («Ninguna columna es utilizable como feature: [...] tienen un único
   valor») y ya rechazaba el objetivo constante en CLASIFICACIÓN («no hay
   nada que clasificar»); la puerta abierta era la de REGRESIÓN.

2. **Un CSV que no es un CSV entraba sin decir palabra.** Fichero con una
   comilla sin cerrar, una fila de 2 campos, bytes binarios y una fila de
   5: `analyze_dataset_csv` contestaba `ok` con «2 filas · 3 columnas» y
   un esquema donde `b` y `c` salían constantes «al 50 % de vacías» — no
   estaban vacías, es que sus filas nunca se leyeron.

Las dos mitades de cada caso: que se rechaza lo que hay que rechazar, y
que NO se rechaza lo que es legítimo (un CSV con comillas de verdad,
celdas vacías de verdad, `;` de Excel, líneas en blanco al final).
"""
from __future__ import annotations

import pytest

from matrixai.training.dataset_analysis import (
    DatasetAnalysisError,
    analyze_dataset_csv,
    constant_target_error,
    structural_damage_error,
)


# El fichero EXACTO del hallazgo 2 (auditoría 2026-08-13).
_CSV_ROTO = (
    'a,b,c\n'
    '1,2,3\n'
    '"sin cerrar,5,6\n'
    '7,8\n'
    '\x00\x01\x02binario,9,10\n'
    '11,12,13,14,15\n'
)


def _csv_objetivo_constante(filas: int = 300) -> str:
    lineas = ["a,b,y"]
    for i in range(filas):
        lineas.append(f"{i},{i * 2},7")
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------------------
# 1 — El objetivo que no varía
# ---------------------------------------------------------------------------

class TestObjetivoConstante:
    def test_objetivo_de_un_solo_valor_se_rechaza_con_su_valor(self):
        """El caso conducido: 300 filas, `y = 7` siempre."""
        motivo = constant_target_error(_csv_objetivo_constante(), "y")
        assert motivo is not None
        # El mensaje tiene que decir QUÉ columna y QUÉ valor: sin eso hay
        # que ir a abrir el CSV para saber de qué habla.
        assert "'y'" in motivo
        assert "'7'" in motivo
        assert "un único valor" in motivo

    def test_un_objetivo_que_varia_no_se_toca(self):
        assert constant_target_error(_csv_objetivo_constante(), "a") is None

    def test_basta_una_fila_distinta_para_que_valga(self):
        """El corte es «más de un valor distinto», no «muchos»: dos filas
        distintas ya son un problema de aprendizaje, por pobre que sea."""
        csv_text = "x,y\n1,7\n2,7\n3,7\n4,8\n"
        assert constant_target_error(csv_text, "y") is None

    def test_los_nulos_no_cuentan_como_un_segundo_valor(self):
        """Una celda vacía en el objetivo NO es un valor: al preparar el
        dataset esa fila se descarta. Si contara, un CSV con `y = 7` y
        cuatro huecos pasaría como si el objetivo variara."""
        csv_text = "x,y\n1,7\n2,\n3,7\n4,NA\n5,7\n"
        motivo = constant_target_error(csv_text, "y")
        assert motivo is not None and "'7'" in motivo

    def test_un_objetivo_de_texto_constante_tambien(self):
        csv_text = "x,resultado\n1,lluvia\n2,lluvia\n3,lluvia\n"
        motivo = constant_target_error(csv_text, "resultado")
        assert motivo is not None and "'lluvia'" in motivo

    def test_columna_que_no_existe_no_es_cosa_de_esta_regla(self):
        """Aguas abajo ya hay un mensaje para eso. Dos sitios diciendo lo
        mismo acaban divergiendo."""
        assert constant_target_error(_csv_objetivo_constante(), "no_existe") is None

    def test_columna_entera_vacia_no_es_cosa_de_esta_regla(self):
        """Cero valores distintos NO es «un único valor»: el mensaje de
        `dataset_project` («no es un target válido», tipo unknown) es más
        preciso y es el que tiene que salir."""
        csv_text = "x,y\n1,\n2,\n3,\n"
        assert constant_target_error(csv_text, "y") is None

    def test_sobre_un_fichero_roto_no_se_diagnostica_el_objetivo(self):
        """En `_CSV_ROTO` la columna `c` PARECE constante ('3') porque las
        filas que la contradicen nunca se leyeron. Diagnosticar el objetivo
        ahí mandaría a alguien a cambiar de columna cuando lo que tiene que
        arreglar es el fichero — y taparía el mensaje que sí sirve."""
        assert constant_target_error(_CSV_ROTO, "c") is None
        with pytest.raises(DatasetAnalysisError):
            analyze_dataset_csv(_CSV_ROTO)

    def test_ni_aunque_el_objetivo_sea_una_columna_que_SÍ_llegó(self):
        """`a` es constante y está en la primera posición, así que en las
        filas cortas su valor sí se lee. Aun así no se diagnostica: el
        fichero entero está roto, y ése es el motivo que hay que dar."""
        csv_text = "a,b,c\n7,1,2\n7,3\n7,4,5\n"
        assert constant_target_error(csv_text, "a") is None

    def test_csv_vacio_o_sin_columna_pedida_no_revienta(self):
        assert constant_target_error("", "y") is None
        assert constant_target_error("   ", "y") is None
        assert constant_target_error("a,b\n1,2\n", "") is None

    def test_el_delimitador_de_excel_tambien_se_ve(self):
        """El CSV de Excel europeo va con `;`. Sin normalizar, la columna
        `y` ni se encontraría y el objetivo constante pasaría."""
        csv_text = "a;b;y\n1;2;7\n3;4;7\n5;6;7\n"
        motivo = constant_target_error(csv_text, "y")
        assert motivo is not None and "'7'" in motivo


# ---------------------------------------------------------------------------
# 2 — El fichero que no es una tabla
# ---------------------------------------------------------------------------

class TestCsvRoto:
    def test_el_fichero_del_hallazgo_se_rechaza(self):
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(_CSV_ROTO)
        assert "no se puede leer como una tabla" in str(exc.value)

    def test_el_fichero_del_hallazgo_nombra_sus_averias(self):
        """No basta con rechazar: lo que salía antes era «2 filas · 3
        columnas» y un esquema. El mensaje tiene que decir QUÉ pasa, que es
        lo que evita ir a buscarlo a ciegas."""
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(_CSV_ROTO)
        mensaje = str(exc.value)
        assert "menos campos que la cabecera" in mensaje
        assert "comilla sin cerrar" in mensaje
        assert "bytes de control" in mensaje

    def test_una_fila_con_campos_de_menos_se_dice_por_su_nombre(self):
        csv_text = "a,b,c\n1,2,3\n4,5\n6,7,8\n"
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(csv_text)
        mensaje = str(exc.value)
        assert "menos campos que la cabecera" in mensaje
        # La fila, con su línea física: es lo que hay que ir a mirar.
        assert "línea 3" in mensaje
        assert "'c'" in mensaje

    def test_una_fila_con_campos_de_mas_se_dice_por_su_nombre(self):
        csv_text = "a,b,c\n1,2,3\n4,5,6,7,8\n"
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(csv_text)
        mensaje = str(exc.value)
        assert "5 campos y la cabecera 3" in mensaje
        assert "'7'" in mensaje and "'8'" in mensaje

    def test_bytes_de_control_en_una_celda(self):
        csv_text = "a,b\n1,dos\n3,cua\x00tro\n"
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(csv_text)
        assert "bytes de control" in str(exc.value)

    def test_una_comilla_sin_cerrar_se_traga_el_fichero_y_se_dice(self):
        """Sin filas cortas de por medio ni bytes raros: el único síntoma es
        que una celda se ha tragado el salto que separaba dos filas."""
        csv_text = 'a,b\n1,"sin cerrar\n2,dos\n3,tres\n'
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(csv_text)
        assert "comilla sin cerrar" in str(exc.value)

    def test_muchos_danos_se_cuentan_en_vez_de_listarse_enteros(self):
        lineas = ["a,b,c"] + [f"{i},{i}" for i in range(20)]
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv("\n".join(lineas) + "\n")
        # 20 filas rotas: 3 con nombre y apellidos y el resto CONTADO. El
        # número exacto va en el aserto a propósito — «y muchos más» cuando
        # se sabe cuántos sería redondear a peor.
        assert "y 17 problema(s) más" in str(exc.value)


class TestLaSegundaPuerta:
    """`structural_damage_error` es el mismo veredicto para quien recibe un
    CSV sin pasar por `analyze_dataset_csv` — `prepare_dataset_from_
    provenance` (reentrenar con datos nuevos) tragaba el fichero roto y
    preparaba UNA fila de las cinco, sin decirlo."""

    def test_da_el_mismo_motivo_que_el_analisis(self):
        motivo = structural_damage_error(_CSV_ROTO)
        assert motivo is not None
        with pytest.raises(DatasetAnalysisError) as exc:
            analyze_dataset_csv(_CSV_ROTO)
        # El MISMO texto: dos sitios diciendo lo mismo con palabras
        # distintas es tener uno de los dos mal.
        assert motivo == str(exc.value)

    def test_un_csv_sano_pasa(self):
        assert structural_damage_error("a,b\n1,2\n3,4\n") is None

    def test_no_lanza_con_un_csv_vacio_o_sin_cabecera(self):
        """Esos los dice `analyze_dataset_csv` con su propio mensaje; aquí
        se calla para no tener dos versiones del mismo error."""
        assert structural_damage_error("") is None
        assert structural_damage_error("   ") is None


class TestNoSeRechazaLoQueSiEsUnCsv:
    """La otra mitad: un rechazo que se lleva por delante ficheros buenos
    es peor que el defecto que arregla."""

    def test_comillas_de_verdad_con_comas_dentro(self):
        csv_text = (
            'ciudad,nota,valor\n'
            '"Santander, Cantabria",buena,1\n'
            '"Bilbao, Bizkaia",mala,2\n'
            '"Gijón, Asturias",buena,3\n'
        )
        r = analyze_dataset_csv(csv_text)
        assert r["rows_total"] == 3
        assert r["column_order"] == ["ciudad", "nota", "valor"]

    def test_texto_entrecomillado_con_comas_Y_SALTOS_dentro(self):
        """El falso positivo que cazó la suite entera.

        La primera versión de la regla rechazaba cualquier celda que
        CONTUVIERA un salto de línea, y eso puso rojo
        `test_comas_y_saltos_entre_comillas` (test_repreparacion_c62_c3):
        un CSV real trae texto entrecomillado con comas y saltos dentro, y
        está soportado a propósito. El criterio bueno es que la celda ACABE
        en salto — eso solo pasa cuando el fichero se termina dentro de una
        comilla sin cerrar.
        """
        filas = "".join(
            f'"nota {i}, con coma\ny salto",{i / 10:.1f},{i % 2}\n' for i in range(12)
        )
        r = analyze_dataset_csv("nota,valor,target\n" + filas)
        assert r["rows_total"] == 12

    def test_una_celda_multilinea_en_la_ULTIMA_fila_tampoco_es_un_daño(self):
        assert analyze_dataset_csv('a,b\n1,"linea1\nlinea2"\n')["rows_total"] == 1

    def test_comillas_dobles_escapadas_dentro_de_una_celda(self):
        csv_text = 'a,b\n1,"dice ""hola"""\n2,"dice ""adiós"""\n'
        assert analyze_dataset_csv(csv_text)["rows_total"] == 2

    def test_celdas_vacias_de_verdad_no_son_campos_que_falten(self):
        """Una celda vacía (`1,,3`) es un dato ausente y ya tiene su
        recuento de nulos; una fila corta (`1,2`) es otra cosa."""
        csv_text = "a,b,c\n1,,3\n4,5,6\n7,8,\n"
        r = analyze_dataset_csv(csv_text)
        assert r["rows_total"] == 3
        assert r["columns"]["b"]["null_count"] == 1
        assert r["columns"]["c"]["null_count"] == 1

    def test_lineas_en_blanco_al_final(self):
        csv_text = "a,b\n1,2\n3,4\n\n\n"
        assert analyze_dataset_csv(csv_text)["rows_total"] == 2

    def test_excel_europeo_con_bom_y_punto_y_coma(self):
        csv_text = "﻿a;b;y\n1;2;3\n4;5;6\n"
        r = analyze_dataset_csv(csv_text)
        assert r["column_order"] == ["a", "b", "y"]
        assert r["rows_total"] == 2

    def test_un_tabulador_dentro_de_una_celda_es_texto_raro_no_binario(self):
        csv_text = "a,b\n1,con\ttabulador\n2,sin\n"
        assert analyze_dataset_csv(csv_text)["rows_total"] == 2
