# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""La columna de TEXTO LIBRE de un CSV: se dice, y se dice bien.

Medido el 2026-08-14: un CSV de 120 reseñas («comentario,valoracion») salía
del análisis con `comentario: identifier` —porque cada fila es distinta— y
de ahí quedaba EXCLUIDA de ser feature. Con dos columnas, el proyecto moría
con «todas son el target, identificadores, fechas o columnas vacías»: al
usuario se le decía que su columna de reseñas era un identificador. Con más
columnas, el modelo se generaba ignorando justo la columna que le importaba,
y la exclusión solo constaba en `provenance` con el motivo equivocado.

Lo que se arregla es lo que se DICE, no el enrutado: el camino desde datos
sigue sin construir modelos de texto (eso lo hace el generador de
transformers desde un prompt con `campo: Text`), pero deja de llamar
identificador a una reseña y dice por dónde sí se puede.

Y el caso que hoy funciona —un identificador de verdad— no se toca: un DNI,
un UUID o un id de pedido también tienen `unique_ratio == 1.0` y DEBEN
seguir siendo identificadores excluidos, sin aviso ninguno.
"""
from __future__ import annotations

import random
from unittest.mock import patch

import pytest

import matrixai.playground as pg
from matrixai.training.dataset_analysis import analyze_dataset_csv
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
)

_FRASES = [
    "el servicio fue excelente y el trato muy cercano",
    "el pedido llego roto y nadie responde a los correos",
    "todo perfecto, repetire sin dudarlo la verdad",
    "no funciona como dice la descripcion, decepcionado",
]
_FRASES_EN = [
    "the delivery was late and the box was damaged",
    "excellent product, i would buy it again for sure",
    "customer support never replied to any of my emails",
    "exactly what i was looking for, very happy with it",
]


def _csv_texto(n: int = 120, columnas_extra: bool = False) -> str:
    rnd = random.Random(3)
    cab = "comentario,importe,valoracion" if columnas_extra else "comentario,valoracion"
    filas = [cab]
    for i in range(n):
        pos = i % 2 == 0
        frase = f"{_FRASES[i % len(_FRASES)]} (caso {i})"
        extra = f",{rnd.uniform(5, 900):.2f}" if columnas_extra else ""
        filas.append(f'"{frase}"{extra},{"positivo" if pos else "negativo"}')
    return "\n".join(filas) + "\n"


def _columna(valores: list[str], cabecera: str) -> str:
    filas = [f"{cabecera},importe,entregado"]
    rnd = random.Random(5)
    for i, v in enumerate(valores):
        filas.append(f"{v},{rnd.uniform(5, 900):.2f},{'si' if i % 3 else 'no'}")
    return "\n".join(filas) + "\n"


def _avisos(res) -> list[str]:
    return [w for s in res.get("pipeline_stages") or [] for w in (s.get("warnings") or [])]


def _avisos_de_texto(res) -> list[str]:
    return [w for w in _avisos(res) if "TEXTO" in w or "TEXT " in w or "TEXT w" in w]


# ---------------------------------------------------------------------------
# El detector, con los dos lados delante
# ---------------------------------------------------------------------------

class TestDetectorDeTextoLibre:
    @pytest.mark.parametrize("frases", [_FRASES, _FRASES_EN])
    def test_una_columna_de_frases_se_reconoce_como_texto(self, frases):
        valores = [f'"{frases[i % len(frases)]} (caso {i})"' for i in range(120)]
        a = analyze_dataset_csv(_columna(valores, "comentario"))
        assert a["columns"]["comentario"]["looks_like_free_text"] is True

    def test_la_evidencia_viaja_con_sus_numeros(self):
        """Un umbral sin la medida al lado no se puede discutir."""
        a = analyze_dataset_csv(_csv_texto())
        ev = a["columns"]["comentario"]["free_text_evidence"]
        assert ev["median_words"] >= 4
        assert ev["median_chars"] >= 25
        assert ev["distinct_word_ratio"] <= 0.5
        assert ev["values_sampled"] == 120

    @pytest.mark.parametrize("nombre,valores", [
        ("pedido_id", [f"PED-2026-{i:05d}" for i in range(120)]),
        ("uuid", ["%08x-%04x-%04x-%04x-%012x" % (i, i, i, i, i * 7) for i in range(120)]),
        ("dni", [f"{10000000 + i * 7}{'TRWAGMYFPDXBNJZSQVHLCKE'[(10000000 + i * 7) % 23]}"
                 for i in range(120)]),
        ("email", [f"cliente{i}@empresa{i % 7}.com" for i in range(120)]),
        ("url", [f"https://empresa.com/productos/categoria/{i}/detalle?ref={i * 3}"
                 for i in range(120)]),
        ("sku", [f"SKU-{i:06d}-XZ-{i * 3:05d}-REV{i % 9}" for i in range(120)]),
        # Un nombre completo es casi-único como un id, y NO es texto libre.
        ("nombre", [f"{n} {a} {b}" for n, a, b in zip(
            ["Roberto", "Ana", "Luis", "Marta", "Carmen", "Javier"] * 20,
            ["Llamosas", "Garcia", "Martin", "Ruiz", "Saiz", "Perez",
             "Diaz", "Gomez", "Lopez", "Blanco"] * 12,
            [f"Conde{i}" for i in range(120)])]),
    ])
    def test_un_identificador_de_verdad_no_se_marca(self, nombre, valores):
        """`unique_ratio == 1.0` NO significa texto: es el caso que hoy
        funciona bien y el que no se puede estropear."""
        a = analyze_dataset_csv(_columna(valores, nombre))
        col = a["columns"][nombre]
        assert col["type"] == "identifier"
        assert "looks_like_free_text" not in col

    def test_pocas_filas_no_bastan_para_afirmarlo(self):
        valores = [f'"{_FRASES[i % len(_FRASES)]} (caso {i})"' for i in range(6)]
        a = analyze_dataset_csv(_columna(valores, "comentario"))
        assert "looks_like_free_text" not in a["columns"]["comentario"]

    def test_el_tipo_NO_cambia(self):
        """A propósito: el tipo viaja por la API de análisis a tres
        interfaces con listas cerradas —una es la clásica, que no se toca—.
        La marca es aditiva; quien no la conoce sigue viendo lo de siempre."""
        a = analyze_dataset_csv(_csv_texto())
        assert a["columns"]["comentario"]["type"] == "identifier"


# ---------------------------------------------------------------------------
# Lo que se le dice a quien subió el CSV
# ---------------------------------------------------------------------------

class TestProyectoDesdeDatos:
    def test_con_otras_columnas_el_modelo_sale_igual_pero_con_aviso(self):
        r = generate_project_from_dataset(_csv_texto(columnas_extra=True),
                                          target_column="valoracion")
        assert r["ok"] is True
        # el modelo es el mismo de antes: la columna de texto sigue fuera
        assert "comentario" not in r["mxai"]
        assert "importe" in r["mxai"]
        avisos = [w for w in _avisos(r) if "comentario" in w]
        assert avisos, _avisos(r)
        assert "Text" in avisos[0]  # por dónde SÍ se puede

    def test_el_motivo_estructurado_deja_de_mentir(self):
        r = generate_project_from_dataset(_csv_texto(columnas_extra=True),
                                          target_column="valoracion")
        motivos = r["provenance"]["excluded_column_reasons"]
        assert motivos["comentario"]["looks_like_free_text"] is True

    def test_sin_ninguna_otra_columna_el_error_dice_la_verdad(self):
        """Antes: «todas son el target, identificadores, fechas o columnas
        vacías», que manda a buscar lo que no existe."""
        with pytest.raises(DatasetProjectError) as exc:
            generate_project_from_dataset(_csv_texto(), target_column="valoracion")
        mensaje = str(exc.value)
        assert "comentario" in mensaje
        assert "TEXTO" in mensaje
        assert "comentario: Text" in mensaje  # la salida, no solo el problema
        assert "identificadores, fechas" not in mensaje

    def test_un_identificador_de_verdad_no_gana_ningun_aviso(self):
        r = generate_project_from_dataset(
            _columna([f"PED-2026-{i:05d}" for i in range(120)], "pedido_id"),
            target_column="entregado")
        assert r["ok"] is True
        motivos = r["provenance"]["excluded_column_reasons"]
        assert "looks_like_free_text" not in motivos["pedido_id"]
        assert not [w for w in _avisos(r) if "pedido_id" in w]

    def test_el_aviso_se_traduce_en_el_core(self):
        es = generate_project_from_dataset(_csv_texto(columnas_extra=True),
                                           target_column="valoracion", locale="es")
        en = generate_project_from_dataset(_csv_texto(columnas_extra=True),
                                           target_column="valoracion", locale="en")
        a_es = [w for w in _avisos(es) if "comentario" in w][0]
        a_en = [w for w in _avisos(en) if "comentario" in w][0]
        assert "TEXTO escrito por una persona" in a_es
        assert "TEXT written by a person" in a_en
        assert not any(f" {p} " in a_en for p in ("el", "la", "del", "con", "que", "para"))

    def test_el_idioma_llega_al_generador_no_solo_a_este_aviso(self):
        """El hueco estaba en el CABLEADO: esta rama pedía el análisis SIN
        `locale`, así que un CSV subido con la aplicación en inglés recibía
        los avisos del pipeline en español."""
        visto: dict = {}
        real = pg.analyze_playground_request

        def espia(payload):
            visto.update(payload)
            return real(payload)

        with patch("matrixai.playground.analyze_playground_request", espia):
            generate_project_from_dataset(_csv_texto(columnas_extra=True),
                                          target_column="valoracion", locale="en")
        assert visto.get("locale") == "en"
