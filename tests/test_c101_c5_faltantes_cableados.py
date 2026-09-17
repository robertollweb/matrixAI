# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — un hueco en el CSV mataba el proyecto entero.

EL BLOQUEANTE, reproducido con 12 filas sintéticas y dos huecos en una sola
columna numérica: `generate_project_from_dataset` moría con

    El CSV preparado no pasa la validación del modelo que acaba de generarse.
    ['DATASET row 4 field edad is empty', 'DATASET row 7 field edad is empty']

porque `_prepare_training_csv` escribía la celda vacía tal cual. Y con una
CATEGÓRICA el mismo hueco salía por dos sitios distintos: la rama de embedding
escribía un índice inexistente («») —mismo rechazo— y la de one-hot ponía la
fila ENTERA a cero, en silencio, indistinguible de una fila rota.

`preparacion.py` (103-C3) ya tenía la política del núcleo para las dos cosas
—mediana + indicador `{columna}__faltante`, y `__faltante__` como tercer
estado de una categórica— y este camino no la usaba: hueco de CABLEADO.

Se verifica por el camino REAL (el producto: `generate_project_from_dataset`,
`_validate_training_csv` y `prepare_dataset_from_provenance`) y por LOS DOS
LADOS — una columna SIN faltantes no puede ganar indicador ni cambiar de
forma, o el arreglo habría creado el sesgo contrario.
"""
from __future__ import annotations

import csv
import io

import pytest

from matrixai.playground import _validate_training_csv
from matrixai.training.categorical import embedding_source_columns
from matrixai.training.dataset_project import (
    DatasetProjectError,
    _prepare_v1,
    PREPARATION_SPEC_VERSION,
    _sha256_text,
    generate_project_from_dataset,
    prepare_dataset_from_provenance,
)
from matrixai.training.dense_generator import _ONEHOT_MAX
from matrixai.training.preparacion import (
    CATEGORIA_DESCONOCIDA,
    CATEGORIA_FALTANTE,
    ajustar_preparacion,
    nombre_de_indicador,
    transformar_fila,
)

# Alturas sin hueco. No son consecutivas a propósito: una columna de enteros
# correlativos la clasifica C1 como `identifier` y desaparece del modelo.
_ALTURAS = ["150.5", "163.2", "171.9", "158.4", "180.1", "149.7",
            "167.3", "175.8", "152.6", "169.0", "177.4", "161.1"]
# Mediana 26.5, MEDIA 42.0: el aserto que mira el valor escrito distingue las
# dos políticas, que es justo lo que hay que poder distinguir.
_EDADES = ["20", "21", "", "23", "24", "", "26", "27", "28", "29", "30", "180"]


def _csv(edades: list[str] | None = None, targets: list[str] | None = None) -> str:
    edades = edades if edades is not None else _EDADES
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["edad", "altura", "target"])
    w.writeheader()
    for i in range(12):
        w.writerow({
            "edad": edades[i],
            "altura": _ALTURAS[i],
            "target": (targets[i] if targets is not None else ["si", "no"][i % 2]),
        })
    return out.getvalue()


def _csv_categorico(n_valores: int, huecos: tuple[int, ...] = (2, 5),
                    filas: int = 14, hueco_texto: str = "") -> str:
    """`cat` con `n_valores` categorías reales y huecos en las filas dadas.

    Con `n_valores > _ONEHOT_MAX` el modelo la consume como ÍNDICE de
    embedding; por debajo, como one-hot. Las dos ramas escribían el hueco de
    forma distinta y las dos estaban mal.
    """
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["cat", "altura", "target"])
    w.writeheader()
    for i in range(filas):
        w.writerow({
            "cat": hueco_texto if i in huecos else f"v{i % n_valores}",
            "altura": _ALTURAS[i % len(_ALTURAS)],
            "target": ["si", "no"][i % 2],
        })
    return out.getvalue()


def _cabecera(res: dict) -> list[str]:
    return res["csv_text"].splitlines()[0].split(",")


def _filas(res: dict) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(res["csv_text"])))


def _valida_contra_su_modelo(res: dict) -> None:
    v = _validate_training_csv(
        res["mxai"], res["training_text"], res["csv_text"],
        field_ranges=res.get("field_ranges"),
    )
    assert v.get("ok"), v.get("errors") or v.get("error")


# ---------------------------------------------------------------------------
# EL BLOQUEANTE
# ---------------------------------------------------------------------------

class TestHuecoNumerico:
    def test_un_hueco_numerico_ya_no_mata_el_proyecto(self):
        """ANTES: DatasetProjectError 'DATASET row 4 field edad is empty'."""
        res = generate_project_from_dataset(_csv(), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)

    def test_el_hueco_se_rellena_con_la_MEDIANA_no_con_la_media(self):
        """La media de las edades presentes es 42.0 y la mediana 26.5 — un
        solo valor extremo (180) las separa, que es el motivo por el que el
        núcleo eligió mediana."""
        res = generate_project_from_dataset(_csv(), target_column="target")
        filas = _filas(res)
        # Mitad positiva: la celda que estaba vacía ahora vale la mediana...
        assert filas[2]["edad"] == "26.5"
        assert filas[5]["edad"] == "26.5"
        # ...y mitad negativa: NO vale la media, ni cero, ni sigue vacía.
        assert filas[2]["edad"] not in ("", "0", "42", "42.0")

    def test_una_celda_que_tenia_valor_no_cambia_ni_un_byte(self):
        res = generate_project_from_dataset(_csv(), target_column="target")
        filas = _filas(res)
        assert [f["edad"] for f in filas if f["edad"] != "26.5"] == [
            "20", "21", "23", "24", "26", "27", "28", "29", "30", "180"]

    def test_el_indicador_viaja_al_CSV_y_al_MODELO(self):
        """Sin indicador la imputación miente en silencio: el modelo no puede
        distinguir «valía 26.5» de «no había valor»."""
        res = generate_project_from_dataset(_csv(), target_column="target")
        assert "edad_faltante" in _cabecera(res)
        # Y en el modelo generado, no solo en el CSV: si no lo declara el
        # VECTOR, el CSV trae una columna que el modelo no conoce.
        assert "edad_faltante" in res["mxai"]

    def test_el_indicador_marca_las_filas_que_se_rellenaron_y_solo_esas(self):
        res = generate_project_from_dataset(_csv(), target_column="target")
        filas = _filas(res)
        marcadas = [i for i, f in enumerate(filas) if f["edad_faltante"] == "1"]
        # Mitad positiva: las dos filas con hueco...
        assert marcadas == [2, 5]
        # ...y mitad negativa: el resto vale "0", no vacío (un aserto que solo
        # mirase "no es 1" lo pasaría una columna entera en blanco).
        assert {f["edad_faltante"] for i, f in enumerate(filas) if i not in (2, 5)} == {"0"}


class TestHuecoCategorico:
    def test_un_hueco_en_una_categorica_de_EMBEDDING_ya_no_mata_el_proyecto(self):
        """La rama de embedding escribía «» cuando el valor no estaba en el
        vocabulario, y el modelo rechazaba su propio CSV. Es el caso real de
        `KDDCup09_appetency`, donde el faltante llega como '?'."""
        res = generate_project_from_dataset(
            _csv_categorico(_ONEHOT_MAX + 3, filas=20, hueco_texto="?"),
            target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)
        # Que esta columna va de verdad por la rama de EMBEDDING lo dice el
        # modelo generado, no el conteo de valores de esta prueba.
        assert embedding_source_columns(res["mxai"]) == {"cat"}
        filas = _filas(res)
        vocab = res["provenance"]["preparation_spec"]["category_vocabularies"]["cat"]
        # `__faltante__` está, detrás de los valores reales. Ya no es el
        # ÚLTIMO porque este modelo va por embedding y lleva además el código
        # reservado de «categoría nunca vista»
        # (`_reservar_codigo_de_desconocida`), que es otra cosa: un hueco no es
        # un valor nuevo. Lo que esta prueba defiende —que el hueco escribe el
        # índice de SU categoría y no una celda vacía— se comprueba ahora
        # preguntando por la posición en vez de darla por hecha.
        assert CATEGORIA_FALTANTE in vocab
        assert vocab[-1] == CATEGORIA_DESCONOCIDA
        # El índice escrito es el de `__faltante__`, no una celda vacía.
        assert filas[2]["cat"] == str(vocab.index(CATEGORIA_FALTANTE))
        assert filas[0]["cat"] == "0"

    def test_un_hueco_en_una_categorica_ONE_HOT_deja_de_ser_una_fila_a_cero(self):
        """Antes la fila con hueco salía con TODAS las columnas del grupo a 0
        — media verdad: indistinguible de una fila rota. Ahora tiene su propia
        columna."""
        res = generate_project_from_dataset(_csv_categorico(3), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)
        filas = _filas(res)
        grupo = [c for c in _cabecera(res) if c.startswith("cat__")]
        assert "cat__faltante" in grupo
        # Mitad positiva: la fila con hueco marca la columna de faltante...
        assert filas[2]["cat__faltante"] == "1"
        # ...y no se colapsa con ninguna categoría real (mitad negativa).
        assert {filas[2][c] for c in grupo if c != "cat__faltante"} == {"0"}
        # Y una fila con valor sigue marcando el suyo y no el de faltante.
        assert filas[0]["cat__faltante"] == "0"
        assert sum(int(filas[0][c]) for c in grupo) == 1

    def test_una_categorica_constante_con_huecos_se_sigue_excluyendo(self):
        """El faltante NO rescata una columna que se iba a caer. Una categórica
        de un solo valor es `constant` y se excluye antes de llegar a la
        política; si `__faltante__` la subiera a dos valores entraría en el
        modelo y cambiaría la forma de datasets que hoy funcionan.

        (Aquí hubo una guarda explícita en `dataset_project.py`; su sabotaje
        salió VERDE porque el mecanismo real es la exclusión por constante y
        aquella línea no podía ejecutarse. Se quitó la línea y se quedó esta
        prueba, que mira el mecanismo que de verdad decide.)"""
        res = generate_project_from_dataset(
            _csv_categorico(1, huecos=(2, 5)), target_column="target")
        assert res["ok"]
        assert res["provenance"]["excluded_column_reasons"]["cat"]["reason"] == "constant_feature"
        assert "cat" not in res["provenance"]["preparation_spec"]["category_vocabularies"]
        assert not [c for c in _cabecera(res) if c.startswith("cat")]
        # Y no se declara un faltante de una columna que no está en el modelo.
        assert res["provenance"]["missing_values"]["missing_category"] == {}


# ---------------------------------------------------------------------------
# EL OTRO LADO: sin faltantes, nada cambia
# ---------------------------------------------------------------------------

class TestSinFaltantesNadaCambia:
    def test_una_numerica_sin_huecos_no_gana_indicador(self):
        sin_huecos = ["20", "21", "22.5", "23", "24", "25.5",
                      "26", "27", "28", "29", "30", "180"]
        res = generate_project_from_dataset(_csv(edades=sin_huecos), target_column="target")
        assert res["ok"]
        # Mitad negativa: ninguna columna de indicador...
        assert not [c for c in _cabecera(res) if "faltante" in c]
        assert "faltante" not in res["mxai"]
        # ...y mitad positiva: la columna sí está, con sus valores intactos.
        assert "edad" in _cabecera(res)
        assert [f["edad"] for f in _filas(res)] == sin_huecos

    def test_una_categorica_sin_huecos_no_gana_la_categoria_faltante(self):
        res = generate_project_from_dataset(
            _csv_categorico(3, huecos=()), target_column="target")
        vocab = res["provenance"]["preparation_spec"]["category_vocabularies"]["cat"]
        assert CATEGORIA_FALTANTE not in vocab
        assert "cat__faltante" not in _cabecera(res)

    def test_un_dataset_limpio_produce_el_MISMO_CSV_que_sin_politica(self):
        """El barrido de md5 en una prueba: preparar con la política activa y
        sin ella tiene que dar el mismo hash byte a byte cuando no hay huecos
        — es lo que garantiza que ninguna receta guardada cambie."""
        from matrixai.training.dataset_project import _prepare_training_csv
        limpio = ["20", "21", "22.5", "23", "24", "25.5",
                  "26", "27", "28", "29", "30", "180"]
        rows = list(csv.DictReader(io.StringIO(_csv(edades=limpio))))
        args = (rows, ["edad", "altura"],
                {"edad": {"type": "number"}, "altura": {"type": "number"}},
                {"edad": "edad", "altura": "altura"}, "target", "classification",
                {"si": "si", "no": "no"}, "predicted_class", {})
        con = _prepare_training_csv(*args, missing_policy=None)
        assert _sha256_text(con.text) == _sha256_text(
            _prepare_training_csv(*args).text)


# ---------------------------------------------------------------------------
# EL OBJETIVO NUNCA SE IMPUTA
# ---------------------------------------------------------------------------

class TestElObjetivoNoSeImputa:
    def test_una_fila_sin_objetivo_se_descarta_y_no_se_rellena(self):
        targets = ["si", "no", "si", "", "si", "no", "si", "no", "si", "no", "si", "no"]
        res = generate_project_from_dataset(_csv(targets=targets), target_column="target")
        assert res["provenance"]["rows_dropped_null_target"] == 1
        assert len(_filas(res)) == 11
        assert "" not in [f["predicted_class"] for f in _filas(res)]

    def test_el_objetivo_viaja_bajo_una_clave_que_ninguna_columna_puede_tener(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. Las features llegan a
        `ajustar_preparacion` con su nombre SAFE y el objetivo con el CRUDO: un
        objetivo llamado `edad` junto a una feature `edad!` —que sanea a
        `edad`— compartirían clave, la feature pisaría al objetivo y la mediana
        se ajustaría sobre OTRAS filas.

        Aquí el objetivo `edad` se vacía en las dos últimas filas. Con el
        centinela, la mediana sale de las 8 presentes de las filas 0-9 (45);
        sin él, `con_objetivo` serían las filas con la FEATURE presente y
        saldría 55."""
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["edad!", "altura", "edad"])
        w.writeheader()
        valores = ["", "", "10", "20", "30", "40", "50", "60", "70", "80", "90", "1000"]
        for i in range(12):
            w.writerow({"edad!": valores[i], "altura": _ALTURAS[i],
                        "edad": "" if i >= 10 else ["si", "no"][i % 2]})
        res = generate_project_from_dataset(out.getvalue(), target_column="edad")
        politica = res["provenance"]["preparation_spec"]["missing_policy"]
        assert politica["columnas"][0]["mediana"] == 45.0
        assert politica["columnas"][0]["mediana"] != 55.0
        assert _filas(res)[0]["edad"] == "45"

    def test_la_mediana_se_ajusta_solo_sobre_las_filas_que_se_escriben(self):
        """Una fila sin objetivo no entrena, así que su valor no puede mover la
        mediana con la que se rellenan las que sí entrenan."""
        edades = ["20", "21", "", "23", "24", "", "26", "27", "28", "29", "30", "180"]
        # Se quitan del train las dos edades más altas: la mediana baja.
        targets = ["si", "no", "si", "no", "si", "no", "si", "no", "si", "no", "", ""]
        res = generate_project_from_dataset(
            _csv(edades=edades, targets=targets), target_column="target")
        # Presentes con objetivo: 20,21,23,24,26,27,28,29 -> mediana 25.0
        assert [f["edad"] for f in _filas(res)][2] == "25"


class TestSoloCuentanLasFilasQueSeEscriben:
    """LAS LÍNEAS QUE NO HACEN LO OBVIO, con su nombre. Un hueco en una fila
    que se va a DESCARTAR (sin objetivo) no puede decidir la forma del modelo:
    añadiría una columna que después nadie marca nunca — una constante a 0 que
    encima afirma que este dataset tiene faltantes donde no los tiene."""

    def test_un_hueco_solo_en_una_fila_sin_objetivo_no_crea_indicador(self):
        edades = ["20", "21", "22.5", "23", "24", "25.5",
                  "26", "27", "28", "29", "30", ""]
        targets = ["si", "no"] * 5 + ["si", ""]
        res = generate_project_from_dataset(
            _csv(edades=edades, targets=targets), target_column="target")
        assert res["provenance"]["rows_dropped_null_target"] == 1
        assert not [c for c in _cabecera(res) if "faltante" in c]
        assert res["provenance"]["preparation_spec"]["missing_policy"] is None

    def test_un_hueco_categorico_solo_en_una_fila_sin_objetivo_no_crea_categoria(self):
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["cat", "altura", "target"])
        w.writeheader()
        for i in range(12):
            w.writerow({"cat": "" if i == 11 else f"v{i % 3}", "altura": _ALTURAS[i],
                        "target": "" if i == 11 else ["si", "no"][i % 2]})
        res = generate_project_from_dataset(out.getvalue(), target_column="target")
        vocab = res["provenance"]["preparation_spec"]["category_vocabularies"]["cat"]
        assert CATEGORIA_FALTANTE not in vocab
        assert "cat__faltante" not in _cabecera(res)
        # Mitad positiva: la columna sí está, con sus tres valores.
        assert sorted(vocab) == ["v0", "v1", "v2"]


# ---------------------------------------------------------------------------
# LA RECETA: congelada, y la de ayer sigue valiendo
# ---------------------------------------------------------------------------

class TestRecetaYReproducibilidad:
    def test_la_receta_congela_la_politica_con_su_mediana(self):
        res = generate_project_from_dataset(_csv(), target_column="target")
        politica = res["provenance"]["preparation_spec"]["missing_policy"]
        assert politica is not None
        assert [c["columna"] for c in politica["columnas"]] == ["edad"]
        assert politica["columnas"][0]["mediana"] == 26.5
        assert politica["columnas"][0]["tipo"] == "numerica"

    def test_repreparar_reproduce_el_mismo_CSV_byte_a_byte(self):
        raw = _csv()
        res = generate_project_from_dataset(raw, target_column="target")
        re_prep = prepare_dataset_from_provenance(raw, res["provenance"])
        assert re_prep.csv_text == res["csv_text"]
        assert _sha256_text(re_prep.csv_text) == res["provenance"]["prepared_csv_sha256"]

    def test_repreparar_una_categorica_con_huecos_reproduce_su_CSV(self):
        raw = _csv_categorico(3, hueco_texto="?")
        res = generate_project_from_dataset(raw, target_column="target")
        re_prep = prepare_dataset_from_provenance(raw, res["provenance"])
        assert re_prep.csv_text == res["csv_text"]
        assert re_prep.compatibility.prepared_matches is True

    def test_la_mediana_NO_se_recalcula_al_repreparar_otro_CSV(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. Si `_prepare_v1`
        reajustara la mediana sobre el CSV nuevo, el dataset dejaría de ser el
        que entrenó el modelo en cuanto cambiara una fila."""
        raw = _csv()
        res = generate_project_from_dataset(raw, target_column="target")
        otras = ["1", "1", "", "1", "1", "", "1", "1", "1", "1", "1", "1"]
        re_prep = prepare_dataset_from_provenance(_csv(edades=otras), res["provenance"])
        filas = list(csv.DictReader(io.StringIO(re_prep.csv_text)))
        # Mediana CONGELADA (26.5), no la del CSV nuevo (que sería 1.0).
        assert filas[2]["edad"] == "26.5"
        assert filas[2]["edad_faltante"] == "1"

    def test_una_receta_ANTERIOR_al_arreglo_sigue_reproduciendo_su_CSV(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. Un modelo guardado
        antes de esto tiene una categórica con huecos cuyo vocabulario NO trae
        `__faltante__`: su CSV puso la fila a cero. Re-prepararlo tiene que
        repetir ESE criterio, no el de hoy, o el modelo guardado pierde su
        dataset."""
        raw = _csv_categorico(3, hueco_texto="?")
        res = generate_project_from_dataset(raw, target_column="target")
        antigua = dict(res["provenance"])
        spec = dict(antigua["preparation_spec"])
        vocab = dict(spec["category_vocabularies"])
        vocab["cat"] = [v for v in vocab["cat"] if v != CATEGORIA_FALTANTE]
        spec["category_vocabularies"] = vocab
        spec.pop("missing_policy")  # tal y como la escribía el código anterior
        antigua["preparation_spec"] = spec
        # El hash preparado de aquel modelo es el de AQUEL CSV, no el de hoy:
        # se recalcula con el preparador aplicando la receta vieja, que es lo
        # único que había entonces. Lo que esta prueba mira de verdad es la
        # FORMA del CSV producido, justo debajo.
        rows = list(csv.DictReader(io.StringIO(raw)))
        antigua["prepared_csv_sha256"] = _sha256_text(_prepare_v1(rows, spec).text)
        re_prep = prepare_dataset_from_provenance(raw, antigua)
        assert re_prep.compatibility.prepared_matches is True
        filas = list(csv.DictReader(io.StringIO(re_prep.csv_text)))
        assert "cat__faltante" not in filas[0]
        assert {filas[2][c] for c in filas[2] if c.startswith("cat__")} == {"0"}
        # Y la mitad positiva: con el vocabulario de hoy SÍ la tendría.
        assert "cat__faltante" in _filas(res)[0]

    def test_una_receta_con_la_politica_corrupta_da_un_error_accionable(self):
        """Una receta editada a mano o truncada al guardar no puede salir como
        un `KeyError` dentro del núcleo: en el producto eso es un error interno
        sin nada que hacer con él."""
        raw = _csv()
        res = generate_project_from_dataset(raw, target_column="target")
        rota = dict(res["provenance"])
        rota["preparation_spec"] = dict(rota["preparation_spec"], missing_policy={"vaya": 1})
        with pytest.raises(DatasetProjectError, match="missing_policy"):
            prepare_dataset_from_provenance(raw, rota)

    def test_la_version_de_receta_NO_sube(self):
        """Subirla marcaría como incompatibles TODAS las recetas guardadas, y
        no hace falta: una receta sin `missing_policy` ni `__faltante__` en su
        vocabulario reproduce su CSV con el preparador de hoy."""
        assert PREPARATION_SPEC_VERSION == 1


# ---------------------------------------------------------------------------
# LO QUE LA IMPUTACIÓN AFIRMA, DECLARADO
# ---------------------------------------------------------------------------

class TestProcedenciaDeclaraLoImputado:
    def test_la_procedencia_dice_QUE_se_relleno_y_CON_QUE(self):
        res = generate_project_from_dataset(_csv(), target_column="target")
        bloque = res["provenance"]["missing_values"]
        assert bloque["imputed_numeric"] == {
            "edad": {"cells": 2, "imputed_with": 26.5, "strategy": "median",
                     "indicator_column": "edad_faltante"},
        }
        assert "impute_missing_numeric_median" in res["provenance"]["operations"]
        assert "add_missing_value_indicator" in res["provenance"]["operations"]

    def test_la_procedencia_dice_cuantas_celdas_categoricas_faltaban(self):
        res = generate_project_from_dataset(
            _csv_categorico(3, hueco_texto="?"), target_column="target")
        bloque = res["provenance"]["missing_values"]
        assert bloque["missing_category"] == {
            "cat": {"cells": 2, "category": CATEGORIA_FALTANTE},
        }
        assert "encode_missing_category" in res["provenance"]["operations"]

    def test_sin_faltantes_la_procedencia_lo_dice_en_vez_de_callar(self):
        limpio = ["20", "21", "22.5", "23", "24", "25.5",
                  "26", "27", "28", "29", "30", "180"]
        bloque = generate_project_from_dataset(
            _csv(edades=limpio), target_column="target")["provenance"]["missing_values"]
        assert bloque == {"imputed_numeric": {}, "missing_category": {}, "limits": []}

    def test_los_limites_del_nucleo_viajan_a_la_procedencia(self):
        """Una columna con más de la mitad de huecos levanta un `Limite` en
        `preparacion.py`; si no viaja, la imputación se hace sin decir que se
        apoya en muy poco."""
        casi_vacia = ["20", "", "", "", "", "", "", "", "", "", "", "180"]
        res = generate_project_from_dataset(_csv(edades=casi_vacia), target_column="target")
        claves = {l["clave"] for l in res["provenance"]["missing_values"]["limits"]}
        assert "faltantes_por_encima_del_umbral" in claves
        assert "pocas_filas_para_preparacion" in claves

    def test_los_limites_viajan_UNA_vez_y_la_receta_OMITE_la_clave(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. La receta guarda la
        política SIN sus `limites`: no hacen falta para reproducir y repetirlos
        costaba 88 KB en `KDDCup09_appetency` (medido; con la política
        duplicada además, la procedencia pasaba de 175 KB a 512 KB). La clave
        se OMITE en vez de escribirse vacía, porque `limites: []` afirmaría
        que esta preparación no encontró nada que declarar."""
        casi_vacia = ["20", "", "", "", "", "", "", "", "", "", "", "180"]
        raw = _csv(edades=casi_vacia)
        res = generate_project_from_dataset(raw, target_column="target")
        spec = res["provenance"]["preparation_spec"]["missing_policy"]
        # Mitad negativa: ni la clave, ni una lista vacía que mienta.
        assert "limites" not in spec
        # Mitad positiva: los límites están, una vez, donde se declaran.
        assert len(res["provenance"]["missing_values"]["limits"]) >= 2
        # Y sin ellos la receta sigue reproduciendo el CSV byte a byte.
        assert prepare_dataset_from_provenance(raw, res["provenance"]).csv_text \
            == res["csv_text"]


# ---------------------------------------------------------------------------
# LO QUE NO SE IMPUTA, Y LO QUE SE RECHAZA
# ---------------------------------------------------------------------------

class TestLoQueNoSeImputa:
    def test_un_valor_presente_pero_ilegible_no_se_rellena_ni_se_marca(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. Un valor que no parsea
        es una respuesta equivocada, no un hueco: imputarlo y marcar el
        indicador sería declarar «no había valor» cuando lo había. Llega por
        re-preparar otro CSV con la misma receta."""
        raw = _csv()
        res = generate_project_from_dataset(raw, target_column="target")
        con_basura = ["20", "21", "", "basura", "24", "", "26", "27", "28", "29", "30", "180"]
        re_prep = prepare_dataset_from_provenance(_csv(edades=con_basura), res["provenance"])
        filas = list(csv.DictReader(io.StringIO(re_prep.csv_text)))
        assert filas[3]["edad"] == "basura"        # viaja tal cual...
        assert filas[3]["edad_faltante"] == "0"    # ...y NO se declara ausente
        assert filas[2]["edad"] == "26.5"          # el hueco de verdad sí

    def test_un_indicador_que_chocaria_con_una_columna_real_es_un_error(self):
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["edad", "edad_faltante", "target"])
        w.writeheader()
        for i in range(12):
            w.writerow({"edad": _EDADES[i], "edad_faltante": _ALTURAS[i],
                        "target": ["si", "no"][i % 2]})
        with pytest.raises(DatasetProjectError, match="__faltante"):
            generate_project_from_dataset(out.getvalue(), target_column="target")

    def test_una_categorica_con_el_centinela_literal_y_huecos_es_un_error(self):
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["cat", "altura", "target"])
        w.writeheader()
        valores = [CATEGORIA_FALTANTE, "a", "", "b", "a", "", CATEGORIA_FALTANTE,
                   "b", "a", "b", "a", "b"]
        for i in range(12):
            w.writerow({"cat": valores[i], "altura": _ALTURAS[i],
                        "target": ["si", "no"][i % 2]})
        with pytest.raises(DatasetProjectError, match="reservado"):
            generate_project_from_dataset(out.getvalue(), target_column="target")


class TestBooleanaConHuecosPasaACategorica:
    """DECISIÓN DE ROBERTO 2026-09-16 — «tratarla como categórica sola»: una
    BOOLEANA con huecos pasa a `categorical` SOLA en cuanto tiene un hueco que
    se va a escribir de verdad, con el hueco como su propia categoría
    (`__faltante__`) — no se inventa ningún valor. Reemplaza a
    `TestUnaBooleanaConHuecosSigueSinPolitica`: aquella medía la DEUDA (se
    rechazaba sin más); esta mide el cierre, sobre la misma columna `flag` y
    el mismo CSV, y añade la equivalencia con el override manual, la
    procedencia, los dos casos que NO cambian y la receta re-aplicada.
    """

    def _csv_booleano(self, huecos: tuple[int, ...] = (2, 5),
                       targets: list[str] | None = None) -> str:
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["flag", "altura", "target"])
        w.writeheader()
        for i in range(12):
            w.writerow({"flag": "" if i in huecos else ["si", "no"][i % 2],
                        "altura": _ALTURAS[i],
                        "target": (targets[i] if targets is not None
                                   else ["si", "no"][i % 2])})
        return out.getvalue()

    def test_sin_override_se_convierte_sola_y_trae_su_indicador(self):
        """(a) Sin override: ok, valida contra su propio modelo, y
        `flag__faltante` sale 1 justo en las filas con hueco (2 y 5)."""
        res = generate_project_from_dataset(self._csv_booleano(), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)
        assert "flag__faltante" in _cabecera(res)
        filas = _filas(res)
        for i, fila in enumerate(filas):
            assert fila["flag__faltante"] == ("1" if i in (2, 5) else "0")

    def test_es_exactamente_igual_al_override_manual(self):
        """(b) EQUIVALENCIA: mismo `csv_text`, mismo modelo (`mxai`/
        `training_text`) y misma `preparation_spec` que declararla a mano
        `column_type_overrides={"flag": "categorical"}`."""
        raw = self._csv_booleano()
        auto = generate_project_from_dataset(raw, target_column="target")
        manual = generate_project_from_dataset(
            raw, target_column="target",
            column_type_overrides={"flag": "categorical"})
        assert auto["csv_text"] == manual["csv_text"]
        assert auto["mxai"] == manual["mxai"]
        assert auto["training_text"] == manual["training_text"]
        assert (auto["provenance"]["preparation_spec"]
                == manual["provenance"]["preparation_spec"])

    def test_procedencia_declara_lo_que_paso_no_lo_que_se_pidio(self):
        """(c) `column_type_overrides` sigue siendo el del usuario —vacío,
        porque no pidió nada— y `operations` gana la marca del cambio
        AUTOMÁTICO; el override manual no la lleva, porque a él nadie se lo
        cambió: ya lo pidió él."""
        raw = self._csv_booleano()
        auto = generate_project_from_dataset(raw, target_column="target")
        assert auto["provenance"]["column_type_overrides"] == {}
        assert "boolean_with_missing_as_categorical:flag" in auto["provenance"]["operations"]

        manual = generate_project_from_dataset(
            raw, target_column="target",
            column_type_overrides={"flag": "categorical"})
        assert "boolean_with_missing_as_categorical:flag" not in manual["provenance"]["operations"]

    def test_una_booleana_sin_huecos_no_cambia(self):
        """(d) Sin huecos: sigue `boolean` en la receta y sin la operación —
        un dataset limpio no gana ni una columna ni un cambio de tipo."""
        res = generate_project_from_dataset(
            self._csv_booleano(huecos=()), target_column="target")
        assert res["ok"]
        assert res["provenance"]["preparation_spec"]["column_types"]["flag"] == "boolean"
        assert not any(op.startswith("boolean_with_missing_as_categorical")
                       for op in res["provenance"]["operations"])

    def test_huecos_solo_en_filas_sin_objetivo_no_cambia(self):
        """(e) Los huecos de `flag` (filas 2 y 5) caen exactamente en las
        filas SIN objetivo: esas filas no se escriben, así que el CSV
        preparado no tiene ningún hueco que convertir — sigue `boolean`."""
        objetivos = ["si", "no", "", "si", "no", "", "si", "no", "si", "no", "si", "no"]
        res = generate_project_from_dataset(
            self._csv_booleano(huecos=(2, 5), targets=objetivos), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)
        assert res["provenance"]["preparation_spec"]["column_types"]["flag"] == "boolean"
        assert not any(op.startswith("boolean_with_missing_as_categorical")
                       for op in res["provenance"]["operations"])

    def test_con_marca_de_ausencia_DECLARADA_cuenta_esa_marca(self):
        """El camino de la Fase 0 declara su marca de ausencia
        (`tokens_de_ausencia`), y con ella ausente es SOLO lo declarado. La
        marca va a propósito fuera de la heurística (`?` está dentro): si la
        conversión mirase la heurística en vez de la declaración, `SIN_DATO`
        sería un valor, la columna seguiría booleana y el verificador la
        rechazaría."""
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["flag", "altura", "target"])
        w.writeheader()
        for i in range(12):
            w.writerow({"flag": "SIN_DATO" if i in (2, 5) else ["si", "no"][i % 2],
                        "altura": _ALTURAS[i], "target": ["si", "no"][i % 2]})
        res = generate_project_from_dataset(out.getvalue(), target_column="target",
                                            tokens_de_ausencia={"SIN_DATO"})
        assert res["ok"]
        _valida_contra_su_modelo(res)
        assert "boolean_with_missing_as_categorical:flag" in res["provenance"]["operations"]
        for i, fila in enumerate(_filas(res)):
            assert fila["flag__faltante"] == ("1" if i in (2, 5) else "0")

    def test_override_explicito_a_boolean_con_huecos_se_sigue_rechazando(self):
        """(f) Invariante 8, el usuario manda incluso para pedir el tipo que
        se rechaza: declararla `column_type_overrides={"flag": "boolean"}` a
        mano la deja FUERA de esta conversión a propósito, no por descuido —
        sigue rechazándose exactamente igual que antes de este corte."""
        with pytest.raises(DatasetProjectError, match="flag is empty"):
            generate_project_from_dataset(
                self._csv_booleano(), target_column="target",
                column_type_overrides={"flag": "boolean"})

    def test_la_receta_automatica_re_aplicada_sigue_tratando_flag_como_categorica(self):
        """(g) La RECETA que salió del camino automático, re-aplicada
        (`prepare_dataset_from_provenance` -> `_prepare_v1`, la misma función
        pública que usan el reentrenamiento y la predicción) sobre un CSV
        NUEVO donde `flag` YA NO tiene huecos: sigue tratándola como
        categórica —one-hot con `flag__faltante` en el vocabulario, congelado
        en la receta— y lo preparado sigue validando contra el modelo. La
        receta es la fuente de verdad, no una relectura del CSV nuevo."""
        raw = self._csv_booleano()
        res = generate_project_from_dataset(raw, target_column="target")
        raw_nuevo = self._csv_booleano(huecos=())
        re_prep = prepare_dataset_from_provenance(raw_nuevo, res["provenance"])
        cabecera_nueva = re_prep.csv_text.splitlines()[0].split(",")
        assert "flag__faltante" in cabecera_nueva
        v = _validate_training_csv(
            res["mxai"], res["training_text"], re_prep.csv_text,
            field_ranges=res.get("field_ranges"),
        )
        assert v.get("ok"), v.get("errors") or v.get("error")


# ---------------------------------------------------------------------------
# UN SOLO SITIO DECLARA EL SUFIJO
# ---------------------------------------------------------------------------

class TestElSufijoSeDeclaraUnaVez:
    def test_columnas_de_salida_y_transformar_fila_usan_el_mismo_nombre(self):
        politica = ajustar_preparacion(
            [{"x": 1.0, "y": 1}, {"x": None, "y": 0}, {"x": 3.0, "y": 1}],
            objetivo="y", columnas=["x"], admite_categoricas=False,
            admite_faltantes=False)
        emitidas = set(transformar_fila({"x": None}, politica))
        assert emitidas == set(politica.columnas_de_salida())
        assert nombre_de_indicador("x") in emitidas
