# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""138-C5 (reparación de su auditoría, 26-10) — lo que se INTERPOLA en un motivo del diagnóstico va en su idioma.

El defecto: `identificador_probable` metía `evidencia = "tipo de columna" | "nombre"` en las dos redacciones, y el motivo inglés
decía «'NHC' looks like an identifier (tipo de columna)…»; `proxy_por_nmi` hacía lo mismo con «NMI aritmética». Lo vio el
auditor gratuito de la web (C5), que enseña el motivo inglés tal cual. Lo que redacta el núcleo se traduce en el núcleo:
`motivo()` acepta un campo `{"es": …, "en": …}` y cada redacción toma el suyo.

Dos guardas: los dos casos, medidos por su texto; y un BARRIDO del fuente (no una lista de casos escogida a mano): ningún
`motivo(...)` de `diagnostico.py` recibe una cadena con letras escrita en el código —sola, en una condicional o a través de una
variable local—, porque una cadena así entra igual en los dos idiomas.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from matrixai.training import diagnostico_textos
from matrixai.training.diagnostico import identificador_probable, proxy_por_nmi
from matrixai.training.diagnostico_textos import motivo

_FUENTE = Path(diagnostico_textos.__file__).with_name("diagnostico.py")

#: Palabras FUNCIONALES del castellano (no se pueden evitar escribiendo en castellano) y las dos que causaron el defecto.
_CASTELLANO = re.compile(
    r"\b(el|la|los|las|del|de|con|que|desde|para|por|una|un|tipo|columna|nombre|aritmética|método)\b", re.IGNORECASE)


def test_identificador_por_tipo_en_ingles_no_lleva_castellano():
    m = identificador_probable("NHC", tipo_columna="identifier").motivo
    assert "(column type)" in m["en"], m["en"]
    assert "(tipo de columna)" in m["es"], m["es"]
    assert not _CASTELLANO.search(m["en"].replace("'NHC'", "")), m["en"]


def test_identificador_por_nombre_en_ingles_no_lleva_castellano():
    s = identificador_probable("cliente_id")
    assert "(name)" in s.motivo["en"], s.motivo["en"]
    assert "(nombre)" in s.motivo["es"], s.motivo["es"]
    assert not _CASTELLANO.search(s.motivo["en"].replace("'cliente_id'", "")), s.motivo["en"]
    # la medida conserva su CLAVE de siempre: quien la lea no cambia
    assert s.medida["evidencia"] == "nombre"
    assert identificador_probable("x", tipo_columna="identifier").medida["evidencia"] == "tipo de columna"


def test_proxy_por_nmi_en_ingles_dice_el_metodo_en_ingles():
    categoria = ["a"] * 40 + ["b"] * 40
    objetivo = ["si"] * 40 + ["no"] * 40
    s = proxy_por_nmi(categoria, objetivo, columna="servicio", soporte_minimo=5)
    assert s is not None
    assert "method arithmetic NMI" in s.motivo["en"], s.motivo["en"]
    assert "método NMI aritmética" in s.motivo["es"], s.motivo["es"]
    assert "aritmética" not in s.motivo["en"]


def test_motivo_toma_de_cada_campo_bilingue_el_de_su_idioma_y_deja_los_demas_tal_cual():
    m = motivo("identificador_probable", campo="'col'", valor={"es": "ES", "en": "EN"})
    assert "(ES)" in m["es"] and "(EN)" in m["en"]
    # un diccionario que NO es exactamente {es, en} no se toma por bilingüe
    m = motivo("identificador_probable", campo="'col'", valor="suelto")
    assert "(suelto)" in m["es"] and "(suelto)" in m["en"]


def _cadenas_con_letras(nodo: ast.AST, nombres: dict[str, list[ast.AST]], hondo: int = 0) -> list[str]:
    """Las cadenas literales con letras que pueden LLEGAR como valor de una expresión: solas, en una condicional, a través de
    una variable (local o del módulo) o de lo que se indexa. No cuentan las que están dentro de un `{"es": …, "en": …}`, las
    de un índice (`medida["n"]`) ni los argumentos de una llamada (`_pct(x, "es")`): esas no son el valor que se interpola."""
    if hondo > 5:
        return []
    if isinstance(nodo, ast.Dict) and {getattr(k, "value", None) for k in nodo.keys} == {"es", "en"}:
        return []
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return [nodo.value] if re.search(r"[^\W\d_]", nodo.value) else []
    if isinstance(nodo, ast.Name):
        return [c for v in nombres.get(nodo.id, []) for c in _cadenas_con_letras(v, nombres, hondo + 1)]
    if isinstance(nodo, ast.Subscript):
        contenedor = nodo.value
        if isinstance(contenedor, ast.Name):
            # un diccionario del módulo indexado: lo que llega es uno de sus VALORES
            for v in nombres.get(contenedor.id, []):
                if isinstance(v, ast.Dict):
                    return [c for x in v.values for c in _cadenas_con_letras(x, nombres, hondo + 1)]
        return []
    if isinstance(nodo, ast.Call):
        return []
    return [c for hijo in ast.iter_child_nodes(nodo) for c in _cadenas_con_letras(hijo, nombres, hondo + 1)]


def _asignaciones(nodo: ast.AST, *, solo_arriba: bool = False) -> dict[str, list[ast.AST]]:
    asignadas: dict[str, list[ast.AST]] = {}
    for n in (nodo.body if solo_arriba else ast.walk(nodo)):  # type: ignore[attr-defined]
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            for destino in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(destino, ast.Name):
                    asignadas.setdefault(destino.id, []).append(n.value)
    return asignadas


def _llamadas_a_motivo() -> list[tuple[str, int, str, ast.AST, dict[str, list[ast.AST]]]]:
    arbol = ast.parse(_FUENTE.read_text(encoding="utf-8"))
    del_modulo = _asignaciones(arbol, solo_arriba=True)
    llamadas = []
    for funcion in ast.walk(arbol):
        if not isinstance(funcion, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nombres = {**del_modulo, **_asignaciones(funcion)}
        for n in ast.walk(funcion):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "motivo":
                for kw in n.keywords:
                    llamadas.append((funcion.name, n.lineno, kw.arg or "**", kw.value, nombres))
    return llamadas


def test_el_barrido_ve_las_llamadas():
    # sin esto, un barrido que no encuentra ninguna llamada sale verde sin mirar nada
    assert len({(f, l) for f, l, *_ in _llamadas_a_motivo()}) >= 10


def test_ningun_motivo_del_diagnostico_recibe_una_cadena_con_letras_escrita_en_el_codigo():
    culpables = []
    for funcion, linea, campo, valor, locales in _llamadas_a_motivo():
        for cadena in _cadenas_con_letras(valor, locales):
            culpables.append(f"{funcion}:{linea} {campo}={cadena!r}")
    assert not culpables, ("estas cadenas entran IGUAL en el motivo castellano y en el inglés: van como "
                           f"{{'es': …, 'en': …}}: {culpables}")


@pytest.mark.parametrize("clave", sorted(diagnostico_textos.MOTIVOS))
def test_las_plantillas_inglesas_no_llevan_castellano(clave):
    en = diagnostico_textos.MOTIVOS[clave]["en"]
    sin_huecos = re.sub(r"\{\w+\}", "", en)
    # «after_outcome» es el valor declarado, no una palabra; lo demás tiene que estar en inglés
    hallado = re.compile(r"\b(el|la|los|las|del|de|con|que|desde|para|por|una|un|y|o)\b").findall(sin_huecos)
    assert not hallado, (clave, hallado)
