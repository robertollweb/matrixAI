# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.1b — la declaración «texto, en <idioma>» en la confirmación.

El contrato 107 (sección «C3.1 — DISEÑO») cierra C3.0 con una medida
concreta: un candidato de SOLO texto (TF-IDF + lineal, `MotorTextoTfidf`,
107-C3.1a) compite en el estudio con los motores de tabla. Antes de que
pueda competir, alguien tiene que poder DECLARAR qué columna es texto y en
qué idioma -- eso es este corte, en el núcleo.

Hasta hoy `_sin_texto_libre` (114-C0) solo sabía EXCLUIR una columna de texto
libre con su aviso. Lo que cambia aquí: `confirmar_desde_csv` gana
`columnas_de_texto={"<campo>": "es"|"en"}`; una columna DECLARADA deja de
excluirse -- sale de los predictores de TABLA igual que antes, pero ahora
queda registrada en `Confirmacion.texto` (para que el candidato de texto la
use), no perdida como una `Exclusion` más. Sin declarar, nada cambia: es
justo lo que prueba `test_114_c0_texto_libre_fuera_del_estudio.py`, que este
fichero no repite.
"""
from __future__ import annotations

import csv
import io
import random

import pytest

from matrixai.training.objetivo import (
    Aviso,
    Bloqueo,
    TextoDeclarado,
    confirmar_desde_csv,
)

_POS = ["excelente producto, volvería a comprar sin duda", "muy buena calidad y atención"]
_NEG = ["pésimo servicio, no lo recomiendo nunca", "terrible experiencia, decepcionante"]


def _csv(n=120, seed=0):
    """`comentario` (texto libre de verdad), `importe` (numérico, NO texto
    libre -- sirve para probar el aviso de "declarada pero no lo parece") y
    `edad` (numérico, queda como predictor de tabla en TODOS los casos, para
    que declarar `importe` como texto no deje el estudio sin entradas)."""
    rng = random.Random(seed)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["row_id", "comentario", "importe", "edad", "y"])
    w.writeheader()
    for i in range(n):
        pos = i % 2 == 0
        frase = f"{rng.choice(_POS if pos else _NEG)} (caso {i})"
        # tildes, mayúsculas, eñes y algún texto vacío -- formas reales, no cómodas.
        if i % 31 == 0:
            frase = ""
        elif i % 9 == 0:
            frase = frase.upper()
        w.writerow({"row_id": str(i), "comentario": frase, "importe": round(rng.uniform(5, 900), 2),
                    "edad": rng.randint(20, 80), "y": "si" if pos else "no"})
    return buf.getvalue()


def _confirmar(**kw):
    kw.setdefault("clase_positiva", "si")
    kw.setdefault("unidad_de_observacion", "una reseña")
    return confirmar_desde_csv(_csv(), objetivo="y", **kw)


# ---------------------------------------------------------------------------
# En POSITIVO: declarada, sale de los predictores de tabla y aparece en `texto`
# ---------------------------------------------------------------------------

class TestColumnaDeclarada:
    def test_sale_de_los_predictores_y_entra_en_texto(self):
        c = _confirmar(columnas_de_texto={"comentario": "es"})
        assert c.confirmado, (c.bloqueos, c.preguntas)
        # Los motores de TABLA siguen sin verla -- quedan `importe` y `edad`.
        assert c.problema.predictors == ("importe", "edad")
        assert c.texto == (TextoDeclarado(campo="comentario", idioma="es"),)
        # Y NO es una Exclusion más: declarar no es lo mismo que excluir.
        assert c.excluidas == ()
        assert c.a_json()["texto"] == [{"campo": "comentario", "idioma": "es"}]

    def test_en_ingles_tambien(self):
        c = _confirmar(columnas_de_texto={"comentario": "en"})
        assert c.confirmado
        assert c.texto == (TextoDeclarado(campo="comentario", idioma="en"),)

    def test_funciona_igual_si_comentario_no_viene_en_las_entradas_declaradas(self):
        """La pantalla puede reenviar `entradas` sin la columna de texto (ni
        siquiera la conoce como predictor) -- declarada, tiene que seguir
        entrando al candidato de texto igual, mismo criterio que 114-C0 con
        la exclusión no declarada."""
        c = _confirmar(entradas=["importe"], columnas_de_texto={"comentario": "es"})
        assert c.confirmado
        assert c.problema.predictors == ("importe",)
        assert c.texto == (TextoDeclarado(campo="comentario", idioma="es"),)

    def test_declararla_tambien_si_esta_en_las_entradas_declaradas(self):
        c = _confirmar(entradas=["comentario", "importe"], columnas_de_texto={"comentario": "es"})
        assert c.confirmado
        assert c.problema.predictors == ("importe",)
        assert c.texto == (TextoDeclarado(campo="comentario", idioma="es"),)


# ---------------------------------------------------------------------------
# En NEGATIVO: sin declarar, lo de hoy (114-C0), sin tocar
# ---------------------------------------------------------------------------

class TestSinDeclararNoCambiaNada:
    def test_columnas_de_texto_none_es_lo_de_siempre(self):
        con_none = _confirmar(columnas_de_texto=None)
        sin_parametro = _confirmar()
        assert con_none.problema.predictors == sin_parametro.problema.predictors
        assert [e.clave for e in con_none.excluidas] == [e.clave for e in sin_parametro.excluidas]
        assert con_none.texto == () and sin_parametro.texto == ()

    def test_columnas_de_texto_vacio_es_lo_de_siempre(self):
        c = _confirmar(columnas_de_texto={})
        assert c.texto == ()
        assert [e.campo for e in c.excluidas] == ["comentario"]

    def test_sin_declarar_la_columna_se_excluye_como_texto_libre(self):
        c = _confirmar()
        assert c.confirmado
        assert c.problema.predictors == ("importe", "edad")
        assert [e.campo for e in c.excluidas] == ["comentario"]
        assert c.excluidas[0].clave == "texto_libre_excluido"
        assert c.texto == ()


# ---------------------------------------------------------------------------
# Errores: inexistente, objetivo, idioma raro, dos columnas -- todos Bloqueo
# ---------------------------------------------------------------------------

class TestErroresDeLaDeclaracion:
    def test_columna_inexistente(self):
        c = _confirmar(columnas_de_texto={"no_existe": "es"})
        assert not c.confirmado
        assert [b.clave for b in c.bloqueos] == ["columna_de_texto_inexistente"]
        m = c.bloqueos[0].motivo
        assert "'no_existe'" in m["es"] and "'no_existe'" in m["en"]
        assert c.texto == ()

    def test_la_columna_de_texto_es_el_objetivo(self):
        c = _confirmar(columnas_de_texto={"y": "es"})
        assert not c.confirmado
        assert [b.clave for b in c.bloqueos] == ["columna_de_texto_es_el_objetivo"]
        assert "'y'" in c.bloqueos[0].motivo["es"]
        assert c.texto == ()

    @pytest.mark.parametrize("idioma", ["fr", "de", "ES", "castellano", ""])
    def test_idioma_no_admitido(self, idioma):
        c = _confirmar(columnas_de_texto={"comentario": idioma})
        assert not c.confirmado
        assert [b.clave for b in c.bloqueos] == ["idioma_de_texto_no_admitido"]
        m = c.bloqueos[0].motivo
        assert "'comentario'" in m["es"]
        assert "es" in m["es"] and "en" in m["es"]  # los dos admitidos, nombrados
        assert c.texto == ()

    def test_dos_columnas_declaradas_es_limitacion_no_fallo_raro(self):
        c = _confirmar(columnas_de_texto={"comentario": "es", "importe": "es"})
        assert not c.confirmado
        assert [b.clave for b in c.bloqueos] == ["columnas_de_texto_multiples"]
        m = c.bloqueos[0].motivo
        # Dicho como LIMITACIÓN de esta versión, no como un error genérico.
        assert "limitaci" in m["es"].lower()
        assert "limitation" in m["en"].lower()
        assert c.texto == ()

    def test_las_claves_son_del_catalogo_bilingue_del_103(self):
        """Un `Bloqueo` con una clave que no está en `MOTIVOS` levantaría
        `KeyError` al componerse -- si esto no lanza, la clave existe y
        compone en los dos idiomas (comprobado también por
        `test_c103_c1_objetivo.py::TestElCatalogoHablaLosDosIdiomas`, que
        recorre el catálogo entero)."""
        from matrixai.training.objetivo_textos import motivo
        for clave, campos in [
            ("columna_de_texto_inexistente", {"campo": "'x'", "opciones": "a, b"}),
            ("columna_de_texto_es_el_objetivo", {"campo": "'x'"}),
            ("idioma_de_texto_no_admitido", {"campo": "'x'", "valor": "'fr'", "opciones": "es, en"}),
            ("columnas_de_texto_multiples", {"opciones": "'a', 'b'"}),
            ("columna_de_texto_declarada_no_parece_libre", {"campo": "'x'"}),
        ]:
            m = motivo(clave, **campos)
            assert m["es"] and m["en"] and "{" not in m["es"] and "{" not in m["en"]


# ---------------------------------------------------------------------------
# Aviso: declarar una columna que el detector no ve como texto libre
# ---------------------------------------------------------------------------

class TestAvisoCuandoNoPareceTextoLibre:
    def test_declarar_una_columna_numerica_se_admite_con_aviso(self):
        """«La persona sabe de sus datos»: se admite igual, pero se dice."""
        c = _confirmar(columnas_de_texto={"importe": "es"})
        assert c.confirmado, (c.bloqueos, c.preguntas)
        assert c.texto == (TextoDeclarado(campo="importe", idioma="es"),)
        assert [a.clave for a in c.avisos] == ["columna_de_texto_declarada_no_parece_libre"]
        assert "'importe'" in c.avisos[0].motivo["es"]
        assert "'importe'" in c.avisos[0].motivo["en"]
        # `importe` ya no es un predictor de tabla (fue declarada texto), y
        # `comentario` -- SIN declarar en este caso -- se excluye como texto
        # libre de siempre: solo `edad` queda como entrada de tabla.
        assert c.problema.predictors == ("edad",)
        assert [e.campo for e in c.excluidas] == ["comentario"]

    def test_un_aviso_no_bloquea_ni_es_pregunta(self):
        c = _confirmar(columnas_de_texto={"importe": "es"})
        assert isinstance(c.avisos[0], Aviso)
        assert c.bloqueos == () and c.preguntas == ()
        # `importe` no cuenta como Exclusion (motivo distinto: uno se admite
        # con aviso, el otro se descarta) -- la única Exclusion es la de
        # `comentario`, el texto libre sin declarar.
        assert [e.campo for e in c.excluidas] == ["comentario"]
        assert "importe" not in [e.campo for e in c.excluidas]

    def test_declarar_texto_de_verdad_no_da_aviso(self):
        c = _confirmar(columnas_de_texto={"comentario": "es"})
        assert c.avisos == ()


# ---------------------------------------------------------------------------
# El documento entero (`a_json`) lleva las claves nuevas
# ---------------------------------------------------------------------------

def test_a_json_lleva_texto_y_avisos_siempre():
    for c in (_confirmar(), _confirmar(columnas_de_texto={"comentario": "es"}),
             _confirmar(columnas_de_texto={"importe": "es"})):
        j = c.a_json()
        assert "texto" in j and "avisos" in j
        assert isinstance(j["texto"], list) and isinstance(j["avisos"], list)
