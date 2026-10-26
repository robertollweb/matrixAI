"""138, reparación 2 — la ficha TRIPOD y los huecos PROBAST no llevan referencias INTERNAS del proyecto.

Son documentos que recibe un usuario (el Studio los sirve tal cual en el expediente de cada estudio, y
`matrixai export` los deja junto al modelo). Llevaban «(105-C3)», «(105-C4)», «Perfil clínico de la tarea
(109-C2)», «El contrato 109 declara 16 preguntas señal…», «es anterior a 109-C3» o «invariante 4 del
contrato 109»: nombres de cortes y contratos de cómo se construyó, que no le dicen nada a quien lo lee.

Barrido, no lista de frases: se COMPONEN los documentos (es/en; estudio y camino clásico, con y sin perfil,
con instrumento que cuadra y que no) y se busca cualquier patrón de referencia interna. También en el mapa
JSON, claves incluidas.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matrixai.export.manifiesto_de_estudio import escribir_manifiesto_de_estudio  # noqa: E402
from matrixai.export.probast import huecos_probast, mapa_probast  # noqa: E402
from matrixai.export.tripod import ficha_tripod  # noqa: E402
from test_c109_c3_expediente import DECLARACION, paquete_clinico  # noqa: E402

#: Lo que delata una nota del PROYECTO dentro de un documento del usuario.
_FUERA = r"(?<![A-Za-zÀ-ÿ0-9])"   # sin \b: «cuentas_segun_el_contrato» lleva «_» delante, que \b toma por letra
_DENTRO = r"(?![A-Za-zÀ-ÿ0-9])"
PATRONES = (
    re.compile(r"\b\d{2,3}-C\d", re.IGNORECASE),                     # 105-C3, 109-C2, 138-C2a…
    re.compile(_FUERA + r"contrat[oa]s?" + _DENTRO, re.IGNORECASE),  # «el contrato 109», y la clave del mapa
    re.compile(_FUERA + r"contracts?" + _DENTRO, re.IGNORECASE),
    re.compile(_FUERA + r"invariantes?" + _DENTRO, re.IGNORECASE),
    re.compile(_FUERA + r"invariants?" + _DENTRO, re.IGNORECASE),
    re.compile(_FUERA + r"pilot[oa]?s?" + _DENTRO, re.IGNORECASE),
)

HUELLA = "46a6b5fb901ade9afb61dcd93451e05ddbf2d09f2b7d7643a32111922d5e1b9e"
ESTUDIO = dict(
    filas=200, huella_del_dataset=HUELLA,
    tipos_de_campo={"x1": "numerica", "x2": "numerica", "categoria": "categorica"},
    columnas_excluidas=["row_id"], semilla_del_reparto=7,
    metricas=[{"name": "auroc", "value": 0.75, "split": "test", "n_observations": 40}],
    filas_de_desarrollo=160, filas_de_prueba_reservada=40, filas_que_entrenaron_el_modelo=68,
    modo="rapido", motores_fuera_por_el_modo=["matrixai.dense.torch_cpu"])


def _instrumento(n: int) -> dict:
    return {"version": "instrumento de prueba", "modo": "development",
            "preguntas": [{"id": f"{i + 1}.1", "dominio": "analisis", "texto": f"enunciado {i + 1}"}
                          for i in range(n)]}


def _paquetes(raiz: Path) -> dict[str, Path]:
    p = {}
    p["estudio"] = raiz / "estudio"
    p["estudio"].mkdir()
    escribir_manifiesto_de_estudio(p["estudio"], **ESTUDIO)
    p["estudio_completo"] = raiz / "estudio_completo"
    p["estudio_completo"].mkdir()
    escribir_manifiesto_de_estudio(p["estudio_completo"], filas=50, modo="completo")
    p["clasico_con_perfil"] = paquete_clinico(raiz / "c1", declaracion=DECLARACION)
    p["clasico_sin_perfil"] = paquete_clinico(raiz / "c2", con_perfil=False)
    p["clasico_instrumento_que_cuadra"] = paquete_clinico(raiz / "c3", instrumento=_instrumento(16))
    p["clasico_instrumento_que_no"] = paquete_clinico(raiz / "c4", instrumento=_instrumento(12))
    return p


def _referencias(texto: str) -> list[str]:
    halladas = []
    for patron in PATRONES:
        for m in patron.finditer(texto):
            halladas.append(texto[max(0, m.start() - 50):m.end() + 30].replace("\n", " "))
    return halladas


@pytest.mark.parametrize("locale", ["es", "en"])
def test_ni_la_ficha_ni_los_huecos_ni_el_mapa_llevan_referencias_internas(tmp_path, locale):
    documentos = 0
    for nombre, paquete in _paquetes(tmp_path).items():
        for quien, texto in (("ficha", ficha_tripod(paquete, locale=locale)),
                             ("huecos", huecos_probast(paquete, locale=locale)),
                             ("mapa", json.dumps(mapa_probast(paquete, locale=locale), ensure_ascii=False))):
            assert texto.strip(), (nombre, quien)
            assert _referencias(texto) == [], (nombre, quien, locale)
            documentos += 1
    assert documentos == 18


def test_el_barrido_tiene_dientes():
    for interna in ("medir (105-C3)", "Perfil clínico de la tarea (109-C2)", "El contrato 109 declara",
                    "Contract 109 states", "invariante 4", "invariant 4 of", "anterior a 109-C3",
                    "sin piloto", "until a pilot", '"cuentas_segun_el_contrato"', "138-C2a"):
        assert _referencias(interna), interna
    for limpia in ("Calibración", "the package predates", "contracción", "pilotaje"):
        # «contracción» y «pilotaje» no son referencias internas: los patrones van por palabra entera.
        assert not _referencias(limpia), limpia


def test_las_cuentas_esperadas_siguen_dichas_sin_nombrar_el_contrato(tmp_path):
    """Lo que la frase del contrato 109 sostenía se conserva: 16/18 y que no está comprobado."""
    paquetes = _paquetes(tmp_path)
    es = huecos_probast(paquetes["clasico_sin_perfil"], locale="es")
    en = huecos_probast(paquetes["clasico_sin_perfil"], locale="en")
    assert "16 preguntas señal para desarrollo y 18 para evaluación" in es and "no está comprobad" in es
    assert "16 signalling questions for development and 18 for evaluation" in en and "not verified" in en
    no_es = huecos_probast(paquetes["clasico_instrumento_que_no"], locale="es")
    assert "trae 12 preguntas" in no_es and "16" in no_es
    mapa = mapa_probast(paquetes["clasico_sin_perfil"], locale="en")
    assert mapa["instrumento"]["cuentas_esperadas"] == {"development": 16, "evaluation": 18}
    assert mapa["instrumento"]["cuentas_comprobadas_contra_el_instrumento_oficial"] is False


# ---------------------------------------------------------------------------
# Y los AUSENTES de un estudio: el paquete lleva `politica_preparacion.json`, y la ficha
# decía «este paquete no declara ninguna política de valores ausentes, así que no se puede
# afirmar qué se hizo con ellos» (re-auditoría de 138, 26-10). Se compone la política con el
# MISMO código que la escribe (`ajustar_preparacion`), no a mano.
# ---------------------------------------------------------------------------

def _paquete_de_estudio_con_preparacion(destino: Path, *, admite_faltantes: bool) -> Path:
    from matrixai.training.preparacion import ajustar_preparacion, tipar_columnas_numericas  # noqa: PLC0415
    destino.mkdir()
    escribir_manifiesto_de_estudio(destino, **ESTUDIO)
    # 50 filas: «presión» falta en 4 (8 %), «categoría» en 5 (10 %), «edad» en ninguna.
    filas = [{"edad": str(20 + i), "presión": "" if i % 12 == 0 and i else str(100 + i),
              "categoría": "" if i % 10 == 3 else ("A" if i % 2 else "B"), "y": str(i % 2)}
             for i in range(50)]
    tipar_columnas_numericas(filas, ["edad", "presión", "categoría"])   # como el estudio, con el texto del CSV
    politica = ajustar_preparacion(filas, objetivo="y", columnas=["edad", "presión", "categoría"],
                                   admite_categoricas=True, admite_faltantes=admite_faltantes)
    (destino / "politica_preparacion.json").write_text(
        json.dumps(politica.a_json(), ensure_ascii=False), encoding="utf-8")
    return destino


@pytest.mark.parametrize("admite", [True, False])
def test_la_ficha_de_un_estudio_dice_lo_que_su_preparacion_hace_con_los_ausentes(tmp_path, admite):
    paquete = _paquete_de_estudio_con_preparacion(tmp_path / "p", admite_faltantes=admite)
    politica = json.loads((paquete / "politica_preparacion.json").read_text(encoding="utf-8"))
    presion = next(c for c in politica["columnas"] if c["columna"] == "presión")
    assert presion["proporcion_faltante"] == pytest.approx(0.08)   # 4 de 50, medido por el escritor real
    mediana = presion["mediana"]
    for locale in ("es", "en"):
        ficha = ficha_tripod(paquete, locale=locale)
        huecos = huecos_probast(paquete, locale=locale)
        for texto in (ficha, huecos):
            assert "no se puede afirmar qué se hizo con ellos" not in texto
            assert "lo que se hizo con ellos no consta" not in texto
            assert "cannot be stated" not in texto and "is not on record" not in texto
            assert "`politica_preparacion.json#columnas`" in texto
        linea = next(l for l in ficha.splitlines() if l.startswith("- **Datos ausentes**" if locale == "es"
                                                                    else "- **Missing data**"))
        if locale == "es":
            assert "ajustada en 50 filas de entrenamiento" in linea
            assert "presión: 8 % ausente; " + ("se deja ausente para el motor, que lo admite" if admite
                                               else f"se rellena con la mediana de entrenamiento ({mediana:g})") in linea
            assert "categoría: 10 % ausente; un ausente es una categoría propia" in linea
        else:
            assert "fitted on 50 training rows" in linea
            assert "presión: 8 % missing; " + ("it is left missing for the engine, which accepts it" if admite
                                               else f"it is filled with the training median ({mediana:g})") in linea
            assert "categoría: 10 % missing; a missing value is a category of its own" in linea
        # «edad» no tenía ausentes: no se nombra como si los tuviera.
        assert "edad:" not in linea
    # Sigue siendo FALTA en PROBAST: nadie DECLARÓ una política; lo que cambia es el motivo.
    por_clave = {c["clave"]: c for dom in mapa_probast(paquete, locale="es")["por_dominio"].values() for c in dom}
    assert por_clave["politica_de_faltantes"]["estado"] == "falta"


def test_sin_preparacion_en_el_paquete_sigue_el_motivo_de_siempre(tmp_path):
    paquetes = _paquetes(tmp_path)
    ficha = ficha_tripod(paquetes["estudio"], locale="es")
    assert ("**Datos ausentes**: este paquete no declara ninguna política de valores ausentes, así que no "
            "se puede afirmar qué se hizo con ellos") in ficha
