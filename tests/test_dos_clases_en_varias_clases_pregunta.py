"""DOS CLASES CON «VARIAS CLASES»: LA CONFIRMACIÓN PREGUNTA (decisión de Roberto, 06-10).

Con `multiclass_classification` y un objetivo de dos clases, la confirmación levantaba
`EsquemaInvalido` («una clasificación multiclase tiene tres clases o más; llegaron 2»), y la
pantalla lo enseñaba como un error rojo SOLO EN CASTELLANO (medido por la auditoría sobre
`fcce68f`; la nota de partida decía que el estudio salía callado sin clase positiva, y no se
reproduce: `ProblemSpec` nunca lo admitió). Ahora pregunta —nunca bloquea—, en los dos idiomas, y
la ÚNICA respuesta es cambiar la tarea a binaria, tras lo cual `clase_positiva` se pide por su
camino de siempre. Auditoría, 1.ª pasada: el texto ya no promete un desenlace que no existe ni
ofrece «multiclase» como respuesta (I-1), nombra las clases como las trae el CSV (M-2) y lo que se
tomó como dato ausente (M-1).
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
    # «multiclase» NO es una respuesta: contestarla devolvía la misma pregunta para siempre (I-1).
    assert p.opciones == ("binary_classification",)


def test_la_pregunta_sale_en_los_dos_idiomas_con_el_objetivo_y_las_clases():
    p = next(p for p in _conf(_csv(["si", "no"]), "multiclass_classification").preguntas
             if p.clave == CLAVE)
    assert "DOS clases" in p.motivo["es"] and "Clasificación binaria" in p.motivo["es"]
    assert "TWO classes" in p.motivo["en"] and "Binary classification" in p.motivo["en"]
    for idioma in ("es", "en"):
        assert "'y'" in p.motivo[idioma] and "(no, si)" in p.motivo[idioma]
        assert "{" not in p.motivo[idioma]
        assert "AUROC" in p.motivo[idioma]       # lo que da la binaria, dicho
    # Lo que PASA: multiclase es para tres o más. Y nada del desenlace inventado (I-1).
    assert "tres clases o más" in p.motivo["es"] and "three classes or more" in p.motivo["en"]
    assert "saldría" not in p.motivo["es"] and "come out" not in p.motivo["en"]


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


# ── auditoría, 1.ª pasada (I-1, M-1…M-4) ─────────────────────────────────────────────────────────

def _pregunta(conf):
    return next(p for p in conf.preguntas if p.clave == CLAVE)


@pytest.mark.parametrize("valores,vistas", [
    (["0", "1"], "(0, 1)"),                   # la forma más corriente (M-3, S14)
    (["Sí", "No"], "(No, Sí)"),
    (["-1", "1"], "(1, -1)"),
    (["Minor", "Major"], "(Major, Minor)"),
])
def test_las_clases_se_nombran_como_las_trae_el_csv(valores, vistas):
    """M-2: «class_0, class_1» o «major, minor» no es lo que ve quien mira su CSV. En el orden de las
    etiquetas del modelo, el mismo de la pregunta de la clase positiva."""
    p = _pregunta(_conf(_csv(valores), "multiclass_classification"))
    assert vistas in p.motivo["es"] and vistas in p.motivo["en"], p.motivo["es"]
    assert "class_" not in p.motivo["es"]


def test_un_objetivo_0_1_con_binaria_y_la_clase_como_se_escribe_confirma():
    conf = _conf(_csv(["0", "1"]), "binary_classification", clase_positiva="1")
    assert conf.confirmado, [(p.clave, p.motivo["es"]) for p in conf.preguntas]


@pytest.mark.parametrize("valores,nombrados", [
    (["Minor", "Major", "None"], "«None»"),
    (["si", "no", "NA"], "«NA»"),
    (["si", "no", "NA", "?"], "«NA», «?»"),
])
def test_lo_que_se_lee_como_ausente_se_nombra(valores, nombrados):
    """M-1: quien ve tres valores en su CSV tiene que saber por qué se le dice «DOS»."""
    p = _pregunta(_conf(_csv(valores), "multiclass_classification"))
    assert nombrados in p.motivo["es"] and nombrados in p.motivo["en"], p.motivo["es"]
    assert "dato ausente" in p.motivo["es"] and "missing" in p.motivo["en"]


def test_una_celda_vacia_no_se_nombra_como_ausente():
    p = _pregunta(_conf(_csv(["si", "no", ""]), "multiclass_classification"))
    assert "dato ausente" not in p.motivo["es"] and "«»" not in p.motivo["es"]


def test_la_pregunta_no_pisa_la_columna_de_texto_declarada():
    """La 1.ª versión de esta reparación guardaba el motivo en una variable `texto` que la función ya usaba
    para la columna de texto declarada: `a_json()` reventaba con `AttributeError` (un 500)."""
    conf = _conf(_csv(["si", "no"]), "multiclass_classification")
    assert conf.texto == () or all(hasattr(t, "a_json") for t in conf.texto)
    assert CLAVE in [p["clave"] for p in conf.a_json()["preguntas"]]


@pytest.mark.parametrize("valores,kw,bloqueo", [
    (["###", "si"], {}, "objetivo_no_predecible"),
    (["si", "no"], {"clases": ["si"]}, "objetivo_con_una_sola_clase"),
])
def test_sin_etiquetas_que_contar_no_se_pregunta_ni_revienta(valores, kw, bloqueo):
    """M-3 (S12): sin `etiquetas is not None`, `len(None)` era un `TypeError` (un 500 por el endpoint)."""
    conf = _conf(_csv(valores), "multiclass_classification", **kw)
    assert CLAVE not in _claves(conf)
    assert bloqueo in [b.clave for b in conf.bloqueos]


def test_con_dos_clases_declaradas_tambien_se_pregunta():
    """M-4: con `clases` declaradas, cuentan las que el modelo va a emitir; antes salía el error viejo del
    esquema, solo en castellano."""
    conf = _conf(_csv(["a", "b", "c"]), "multiclass_classification", clases=["a", "b"])
    assert CLAVE in _claves(conf)


def test_con_una_clase_positiva_mandada_por_api_tambien_se_pregunta():
    """M-3 (S15): multiclase + `clase_positiva` no se salta la pregunta (acabaría en el error del esquema)."""
    conf = _conf(_csv(["si", "no"]), "multiclass_classification", clase_positiva="si")
    assert CLAVE in _claves(conf) and not conf.confirmado
