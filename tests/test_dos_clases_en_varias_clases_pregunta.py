"""DOS CLASES CON «VARIAS CLASES»: LA CONFIRMACIÓN PREGUNTA (decisión de Roberto, 06-10).

Medido el 05-10: con `multiclass_classification` y un objetivo de dos clases, la
confirmación lo aceptaba callada y el estudio salía sin clase positiva, sin umbral y
sin AUROC. Ahora pregunta —nunca bloquea— y la respuesta es cambiar la tarea a binaria,
tras lo cual `clase_positiva` se pide por su camino de siempre.
"""
from __future__ import annotations

import pytest

from matrixai.training.objetivo import confirmar_desde_csv
from matrixai.training.objetivo_textos import huecos_de, MOTIVOS

CLAVE = "dos_clases_en_varias"


def _csv(valores: list[str], n: int = 60) -> str:
    return "\n".join(["x,y"] + [f"{i},{valores[i % len(valores)]}" for i in range(n)]) + "\n"


def _conf(csv: str, tarea: str | None, **kw):
    return confirmar_desde_csv(csv, objetivo="y", tarea=tarea,
                               unidad_de_observacion="una fila", **kw)


def _claves(conf) -> list[str]:
    return [p.clave for p in conf.preguntas]


def test_dos_clases_con_varias_clases_pregunta_y_no_confirma():
    conf = _conf(_csv(["si", "no"]), "multiclass_classification")
    assert CLAVE in _claves(conf)
    assert not conf.confirmado
    assert not conf.bloqueos          # NUNCA un error: es una pregunta
    p = next(p for p in conf.preguntas if p.clave == CLAVE)
    assert p.campo == "y"
    assert set(p.opciones) == {"binary_classification", "multiclass_classification"}


def test_la_pregunta_sale_en_los_dos_idiomas_con_el_objetivo_y_las_clases():
    p = next(p for p in _conf(_csv(["si", "no"]), "multiclass_classification").preguntas
             if p.clave == CLAVE)
    assert "DOS clases" in p.motivo["es"] and "Clasificación binaria" in p.motivo["es"]
    assert "TWO classes" in p.motivo["en"] and "Binary classification" in p.motivo["en"]
    for idioma in ("es", "en"):
        assert "'y'" in p.motivo[idioma] and "no, si" in p.motivo[idioma]
        assert "{" not in p.motivo[idioma]
        assert "AUROC" in p.motivo[idioma]       # lo que se pierde, dicho


def test_contestar_binaria_pide_la_clase_positiva_por_el_camino_de_siempre():
    conf = _conf(_csv(["si", "no"]), "binary_classification")
    assert CLAVE not in _claves(conf)
    assert "clase_positiva" in _claves(conf)
    conf = _conf(_csv(["si", "no"]), "binary_classification", clase_positiva="si")
    assert conf.confirmado and not conf.preguntas


def test_tres_clases_con_varias_clases_sigue_como_hoy():
    conf = _conf(_csv(["a", "b", "c"]), "multiclass_classification")
    assert CLAVE not in _claves(conf)
    assert conf.confirmado


def test_una_binaria_deducida_sigue_como_hoy():
    conf = _conf(_csv(["si", "no"]), None)
    assert CLAVE not in _claves(conf)
    assert _claves(conf) == ["clase_positiva"]


def test_dos_clases_mas_un_NA_que_no_cuenta_como_clase_tambien_pregunta():
    # tres valores distintos en la columna, pero «NA» es ausencia: son DOS clases.
    conf = _conf(_csv(["si", "no", "NA"]), "multiclass_classification")
    assert CLAVE in _claves(conf)


def test_un_nivel_legitimo_llamado_none_cuenta_como_clase():
    # con «None» declarado como nivel (no ausente) hay TRES clases: no se pregunta.
    conf = _conf(_csv(["si", "no", "None"]), "multiclass_classification",
                 tokens_de_ausencia={"NA"})
    assert CLAVE not in _claves(conf)


def test_con_un_analisis_hecho_sin_los_tokens_de_ausencia_se_cuentan_los_valores_leidos():
    """El backend pasa `analisis` ya compuesto (sin `tokens_de_ausencia`): su cardinalidad
    cuenta «sin_dato» como clase (3). Las clases son las del objetivo YA leído (2): se pregunta."""
    from matrixai.training.dataset_analysis import analyze_dataset_csv

    csv = _csv(["si", "no", "sin_dato"])
    analisis = analyze_dataset_csv(csv)
    assert analisis["columns"]["y"]["cardinality"] == 3
    conf = _conf(csv, "multiclass_classification", analisis=analisis, tokens_de_ausencia={"sin_dato"})
    assert CLAVE in _claves(conf)


def test_regresion_no_pregunta_nada_de_esto():
    conf = _conf(_csv([str(i * 1.5) for i in range(40)]), "regression")
    assert CLAVE not in _claves(conf)


def test_el_catalogo_tiene_las_dos_redacciones_con_los_mismos_huecos():
    t = MOTIVOS[CLAVE]
    assert set(t) == {"es", "en"} and huecos_de(t["es"]) == huecos_de(t["en"])


@pytest.mark.parametrize("tarea", ["multiclass_classification"])
def test_la_pregunta_viaja_en_el_json(tarea):
    d = _conf(_csv(["si", "no"]), tarea).a_json()
    assert CLAVE in [p["clave"] for p in d["preguntas"]]
