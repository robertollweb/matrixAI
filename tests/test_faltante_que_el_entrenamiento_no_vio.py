# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""04-10 — un HUECO que el entrenamiento no tuvo perdía la parte entera.

EL DEFECTO, visto por Roberto en `dresses-sales` (la red anterior fallaba una parte
del estudio) y medido con la partición REAL del estudio (semilla 0, 5 partes): las
2 filas de 500 con `V8` vacía caen las dos en la parte MEDIDA del pliegue 1, ninguna
en su entrenamiento ni en su cola. El estudio ajusta la preparación en cada parte con
sus filas de entrenamiento y se la aplica a todas, así que `transformar_fila` escribe
`__faltante__` en esas dos filas; la red anterior genera su proyecto con
entrenamiento+cola, que no lo traen, y al preparar la parte medida
`prepare_dataset_from_provenance` lo rechazaba con «La columna 'V8' trae valores que
el modelo no conoce: ['__faltante__']». La parte se perdía entera por dos celdas.

Es el mismo defecto que `test_c101_c5_categoria_no_vista.py` arregló para
`__desconocida__` (la otra marca de `transformar_fila`), y se arregla igual: para el
modelo, un hueco que su entrenamiento no tuvo es una categoría que no vio.

  · one-hot → el grupo entero a 0 («ninguna de las conocidas»), no la referencia;
  · embedding → el código reservado de lo no visto (en embedding no hay «todo a 0»);
    sin código reservado (una receta anterior), se rechaza diciendo QUÉ falta;
  · y se DECLARA aparte, con su recuento: no es lo mismo para quien lo lee.

Y EL MISMO HUECO, CRUDO (auditoría del 04-10): una celda VACÍA en esa columna, por el
camino directo (reimportar un CSV en la clásica, sin `transformar_fila` delante), se
escribía en embedding como un índice `""` que el modelo rechazaba después en
`validate-csv` («field cat is empty»), y en one-hot salía a 0 sin decirlo. En una receta
posterior a la política de faltantes (trae `missing_policy`) va ahora por el mismo
camino que la marca; una receta anterior se reproduce con el criterio de entonces.

Lo que no cambia: si el entrenamiento SÍ tuvo huecos, `__faltante__` es una categoría
suya y se escribe con su propio índice, sin aviso; y un valor nuevo del usuario sigue
abortando aunque llegue junto a un hueco.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from matrixai.playground import _validate_training_csv
from matrixai.training.categorical import embedding_source_columns
from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
    prepare_dataset_from_provenance,
)
from matrixai.training.dense_generator import _ONEHOT_MAX
from matrixai.training.particion_por_diseno import proponer_particion
from matrixai.training.preparacion import (
    CATEGORIA_DESCONOCIDA,
    CATEGORIA_FALTANTE,
    ajustar_preparacion,
    tipar_columnas_numericas,
    transformar_fila,
)

_ALTURAS = ["150.5", "163.2", "171.9", "158.4", "180.1", "149.7",
            "167.3", "175.8", "152.6", "169.0", "177.4", "161.1"]
_N_EMBEDDING = _ONEHOT_MAX + 3   # va por embedding
_N_ONEHOT = 3                    # va por one-hot


def _csv(n_valores: int, filas: int = 26, extra: dict[int, str] | None = None) -> str:
    """`cat` con `n_valores` categorías; `extra` pone en la fila i otro valor (una marca
    del núcleo o un valor nuevo) en vez de su categoría."""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["cat", "altura", "target"])
    w.writeheader()
    for i in range(filas):
        w.writerow({"cat": (extra or {}).get(i, f"v{i % n_valores}"),
                    "altura": _ALTURAS[i % len(_ALTURAS)], "target": ["si", "no"][i % 2]})
    return out.getvalue()


def _filas(texto: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(texto)))


def _vocab(res: dict, col: str = "cat") -> list[str]:
    return res["provenance"]["preparation_spec"]["category_vocabularies"][col]


def _avisos_de_hueco(rep) -> list[str]:
    return [w for w in rep.compatibility.warnings if "un hueco" in w]


# --- La forma de datos REAL: la parte de dresses-sales que perdía la red anterior ---------

_DRESSES = (Path(__file__).resolve().parents[2] / "matrixaistudio" / "studio-backend"
            / "matrixai_studio" / "ejemplos_medidos" / "dresses-sales.csv")


@pytest.fixture(scope="module")
def parte_de_dresses():
    """La parte 1 (pliegue de índice 1) de un estudio por omisión, como la arma el
    Studio: `proponer_particion` (iid, semilla 0, 20 % de test, 5 partes, estratificada
    por `Class`, id de fila = posición), `tipar_columnas_numericas` sobre los predictores
    (como `estudio_job`), entrenamiento 85 % / cola 15 % posicional, la
    preparación AJUSTADA con el entrenamiento y con las capacidades de la red anterior
    (categóricas sí, faltantes no), y el proyecto generado con entrenamiento+cola, que
    es lo que hace la densa. Sin entrenar nada: solo el núcleo."""
    if not _DRESSES.is_file():
        pytest.fail(f"no está {_DRESSES} (matrixaistudio, al lado de este repo): sin él no "
                    "se puede comprobar la forma de datos real que perdía la parte")
    filas = list(csv.DictReader(_DRESSES.open(encoding="utf-8")))
    for posicion, fila in enumerate(filas):
        fila["__row_id__"] = str(posicion)
    propuesta = proponer_particion(
        filas, plan_id="dresses", observation_id_field="__row_id__", split_type="iid",
        seed=0, test_fraction=0.2, folds=5, repeats=1, unit_id_field=None,
        time_column=None, gap=None, objetivo="Class")
    predictores = [c for c in filas[0] if c not in ("Class", "__row_id__")]
    tipar_columnas_numericas(filas, predictores)
    por_id = {f["__row_id__"]: f for f in filas}
    pliegue = propuesta.pliegues.pliegue_de(repeticion=0, pliegue=1)
    entrena = list(pliegue.entrena)
    corte = int(len(entrena) * 0.85)
    politica = ajustar_preparacion(
        [por_id[i] for i in entrena[:corte]], objetivo="Class", columnas=predictores,
        admite_categoricas=True, admite_faltantes=False, con_fechas=True)
    columnas = politica.columnas_de_salida()

    def escribir(ids: list[str]) -> tuple[str, list[dict]]:
        transformadas = [transformar_fila(por_id[i], politica) for i in ids]
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow([*columnas, "Class"])
        for i, t in zip(ids, transformadas):
            w.writerow(["" if t.get(c) is None else t[c] for c in columnas] + [por_id[i]["Class"]])
        return out.getvalue(), transformadas

    combinado, de_entrenamiento = escribir(entrena)
    medido, de_la_medida = escribir(list(pliegue.valida))
    res = generate_project_from_dataset(
        combinado, "Class", locale="es", column_type_overrides={"Class": "categorical"},
        tokens_de_ausencia={""})
    return {"res": res, "medido": medido, "de_entrenamiento": de_entrenamiento,
            "de_la_medida": de_la_medida, "ids_medidos": list(pliegue.valida)}


class TestLaParteQuePerdiaLaRedAnterior:
    def test_la_premisa_los_dos_huecos_de_V8_solo_estan_en_la_parte_medida(self, parte_de_dresses):
        """Control del instrumento: si la partición o la preparación cambian y los huecos
        ya no caen así, esta clase no mediría nada y tiene que decirlo."""
        p = parte_de_dresses
        assert sum(t["V8"] == CATEGORIA_FALTANTE for t in p["de_entrenamiento"]) == 0
        assert sum(t["V8"] == CATEGORIA_FALTANTE for t in p["de_la_medida"]) == 2
        assert embedding_source_columns(p["res"]["mxai"]) >= {"v8"}
        assert CATEGORIA_FALTANTE not in _vocab(p["res"], "V8")
        assert _vocab(p["res"], "V8")[-1] == CATEGORIA_DESCONOCIDA

    def test_la_parte_YA_NO_se_pierde(self, parte_de_dresses):
        """EL DEFECTO. Antes: DatasetProjectError «La columna 'V8' trae valores que el
        modelo no conoce: ['__faltante__']»."""
        p = parte_de_dresses
        rep = prepare_dataset_from_provenance(p["medido"], p["res"]["provenance"])
        assert rep.compatibility.ok

    def test_los_dos_huecos_van_al_codigo_reservado_y_las_demas_filas_no_se_mueven(self, parte_de_dresses):
        p = parte_de_dresses
        vocab = _vocab(p["res"], "V8")
        rep = prepare_dataset_from_provenance(p["medido"], p["res"]["provenance"])
        escritas = _filas(rep.csv_text)
        huecos = [i for i, t in enumerate(p["de_la_medida"]) if t["V8"] == CATEGORIA_FALTANTE]
        assert [escritas[i]["v8"] for i in huecos] == [str(vocab.index(CATEGORIA_DESCONOCIDA))] * 2
        # mitad negativa: una fila con una manga conocida conserva SU índice
        conocida = next(i for i, t in enumerate(p["de_la_medida"]) if t["V8"] in vocab
                        and t["V8"] != CATEGORIA_DESCONOCIDA)
        assert escritas[conocida]["v8"] == str(vocab.index(p["de_la_medida"][conocida]["V8"]))
        assert [(i, k) for i, f in enumerate(escritas) for k, v in f.items() if not (v or "").strip()] == []

    def test_se_declara_con_su_recuento_y_aparte_de_las_categorias_no_vistas(self, parte_de_dresses):
        """En esa misma parte, V8 trae además 2 filas con una manga que el entrenamiento
        no vio (`Petal`, `threequater`): son otras dos, y cada cosa se cuenta en su aviso."""
        p = parte_de_dresses
        rep = prepare_dataset_from_provenance(p["medido"], p["res"]["provenance"])
        de_v8 = [w for w in _avisos_de_hueco(rep) if "'V8'" in w]
        assert len(de_v8) == 1 and "2 filas" in de_v8[0], rep.compatibility.warnings
        assert "código reservado" in de_v8[0]
        no_vistas = [w for w in rep.compatibility.warnings if "nunca vio" in w and "'V8'" in w]
        assert len(no_vistas) == 1 and "2 filas" in no_vistas[0]


# --- Las dos ramas, con datos fabricados -------------------------------------------------

class TestEmbedding:
    def test_un_hueco_que_el_entrenamiento_no_tuvo_YA_NO_pierde_la_parte(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        assert embedding_source_columns(res["mxai"]) == {"cat"}
        assert CATEGORIA_FALTANTE not in _vocab(res)
        rep = prepare_dataset_from_provenance(
            _csv(_N_EMBEDDING, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
        assert rep.compatibility.ok

    def test_se_escribe_el_codigo_reservado_y_la_fila_de_al_lado_no_se_mueve(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        vocab = _vocab(res)
        rep = prepare_dataset_from_provenance(
            _csv(_N_EMBEDDING, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
        filas = _filas(rep.csv_text)
        assert filas[4]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        assert filas[5]["cat"] == str(vocab.index("v5"))

    def test_sin_codigo_reservado_lo_rechaza_diciendo_QUE_falta(self):
        """Una receta anterior a `_reservar_codigo_de_desconocida`: no hay dónde escribir
        el hueco. Se sigue abortando, pero no como «un valor que el modelo no conoce»."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        prov = res["provenance"]
        spec = prov["preparation_spec"]
        spec["category_vocabularies"]["cat"] = [
            v for v in spec["category_vocabularies"]["cat"] if v != CATEGORIA_DESCONOCIDA]
        with pytest.raises(DatasetProjectError) as exc:
            prepare_dataset_from_provenance(_csv(_N_EMBEDDING, extra={4: CATEGORIA_FALTANTE}), prov)
        assert "huecos (1 filas)" in str(exc.value)
        assert "EMBEDDING" in str(exc.value)
        assert "no conoce" not in str(exc.value)

    def test_se_declara_y_dice_como_se_codifico(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        rep = prepare_dataset_from_provenance(
            _csv(_N_EMBEDDING, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "'cat'" in avisos[0] and "1 filas" in avisos[0], avisos
        assert "código reservado" in avisos[0]


class TestOneHot:
    def test_el_grupo_entero_a_cero_y_el_de_al_lado_marca_el_suyo(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        assert embedding_source_columns(res["mxai"]) == set()
        rep = prepare_dataset_from_provenance(
            _csv(_N_ONEHOT, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
        assert rep.compatibility.ok
        filas = _filas(rep.csv_text)
        grupo = [c for c in filas[0] if c.startswith("cat__")]
        assert len(grupo) == _N_ONEHOT
        assert {filas[4][c] for c in grupo} == {"0"}
        assert sum(float(filas[5][c]) for c in grupo) == 1.0

    def test_NO_se_usa_la_categoria_de_referencia(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        rep = prepare_dataset_from_provenance(
            _csv(_N_ONEHOT, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
        assert _filas(rep.csv_text)[4]["cat__v0"] == "0"

    def test_se_declara_como_ninguna_de_las_conocidas_con_el_recuento_REAL(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        rep = prepare_dataset_from_provenance(
            _csv(_N_ONEHOT, extra={i: CATEGORIA_FALTANTE for i in (2, 4, 6)}), res["provenance"])
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "3 filas" in avisos[0], avisos
        assert "ninguna de las conocidas" in avisos[0]


class TestLoQueNoCambia:
    def test_si_el_entrenamiento_SI_tuvo_huecos_es_una_categoria_suya_y_no_se_avisa(self):
        """El sesgo contrario: con `__faltante__` en el vocabulario (el entrenamiento tuvo
        huecos), el hueco tiene su propio índice y no es «lo no visto»."""
        res = generate_project_from_dataset(
            _csv(_N_EMBEDDING, extra={1: CATEGORIA_FALTANTE}), target_column="target")
        vocab = _vocab(res)
        assert CATEGORIA_FALTANTE in vocab
        rep = prepare_dataset_from_provenance(
            _csv(_N_EMBEDDING, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
        assert _filas(rep.csv_text)[4]["cat"] == str(vocab.index(CATEGORIA_FALTANTE))
        assert _avisos_de_hueco(rep) == []

    def test_sin_huecos_no_se_avisa_de_nada(self):
        crudo = _csv(_N_ONEHOT)
        res = generate_project_from_dataset(crudo, target_column="target")
        assert _avisos_de_hueco(prepare_dataset_from_provenance(crudo, res["provenance"])) == []

    def test_un_valor_NUEVO_del_usuario_sigue_abortando_aunque_llegue_junto_a_un_hueco(self):
        """El hueco es una marca del núcleo; `v99` es un dato nuevo y el vocabulario no se
        amplía en silencio. El error nombra SOLO el valor nuevo."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        with pytest.raises(DatasetProjectError) as exc:
            prepare_dataset_from_provenance(
                _csv(_N_EMBEDDING, extra={4: CATEGORIA_FALTANTE, 5: "v99"}), res["provenance"])
        assert "v99" in str(exc.value)
        assert CATEGORIA_FALTANTE not in str(exc.value)

    def test_un_hueco_y_una_categoria_no_vista_se_cuentan_cada_uno_en_su_aviso(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        rep = prepare_dataset_from_provenance(
            _csv(_N_ONEHOT, extra={2: CATEGORIA_DESCONOCIDA, 4: CATEGORIA_FALTANTE,
                                   6: CATEGORIA_FALTANTE}), res["provenance"])
        no_vistas = [w for w in rep.compatibility.warnings if "nunca vio" in w]
        assert len(no_vistas) == 1 and "1 filas" in no_vistas[0], rep.compatibility.warnings
        huecos = _avisos_de_hueco(rep)
        assert len(huecos) == 1 and "2 filas" in huecos[0], rep.compatibility.warnings


# --- El mismo hueco, CRUDO: el camino directo (auditoría del 04-10) ----------------------

def _valida_contra_su_modelo(res: dict, csv_text: str) -> dict:
    return _validate_training_csv(res["mxai"], res["training_text"], csv_text,
                                  field_ranges=res.get("field_ranges"))


class TestElHuecoCrudoEnUnaRecetaNueva:
    def test_la_receta_es_posterior_a_la_politica_de_faltantes(self):
        """Control: lo que distingue una receta nueva es la CLAVE `missing_policy`."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        assert "missing_policy" in res["provenance"]["preparation_spec"]

    def test_en_embedding_va_al_codigo_reservado_y_el_modelo_lo_acepta(self):
        """EL DEFECTO I1: el índice `""` pasaba la re-preparación sin aviso y el modelo
        rechazaba luego el conjunto entero en `validate-csv`."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        vocab = _vocab(res)
        rep = prepare_dataset_from_provenance(_csv(_N_EMBEDDING, extra={4: ""}), res["provenance"])
        filas = _filas(rep.csv_text)
        assert filas[4]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        assert filas[5]["cat"] == str(vocab.index("v5"))
        v = _valida_contra_su_modelo(res, rep.csv_text)
        assert v.get("ok"), v.get("errors") or v.get("error")

    def test_en_one_hot_el_grupo_a_cero_pero_ahora_DICHO(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        rep = prepare_dataset_from_provenance(_csv(_N_ONEHOT, extra={4: ""}), res["provenance"])
        filas = _filas(rep.csv_text)
        assert {filas[4][c] for c in filas[0] if c.startswith("cat__")} == {"0"}
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "1 filas" in avisos[0], rep.compatibility.warnings

    def test_la_marca_y_la_celda_vacia_juntas_se_cuentan_y_se_escriben_igual(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        vocab = _vocab(res)
        rep = prepare_dataset_from_provenance(
            _csv(_N_EMBEDDING, extra={4: "", 6: CATEGORIA_FALTANTE}), res["provenance"])
        filas = _filas(rep.csv_text)
        assert filas[4]["cat"] == filas[6]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "2 filas" in avisos[0], rep.compatibility.warnings
        assert _valida_contra_su_modelo(res, rep.csv_text).get("ok")

    def test_la_marca_con_espacios_alrededor_tambien_cuenta(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        rep = prepare_dataset_from_provenance(
            _csv(_N_ONEHOT, extra={4: f"  {CATEGORIA_FALTANTE} "}), res["provenance"])
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "1 filas" in avisos[0], rep.compatibility.warnings

    def test_una_fila_SIN_OBJETIVO_no_se_cuenta_porque_no_se_escribe(self):
        """Se descarta (nunca se inventa un objetivo): contarla afirmaría que se codificó."""
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        crudo = _csv(_N_ONEHOT, extra={4: CATEGORIA_FALTANTE, 6: CATEGORIA_FALTANTE,
                                       8: CATEGORIA_DESCONOCIDA, 10: CATEGORIA_DESCONOCIDA})
        lineas = crudo.splitlines()
        for i in (7, 11):   # las filas 6 y 10 (la línea 0 es la cabecera), sin objetivo
            lineas[i] = lineas[i].rsplit(",", 1)[0] + ","
        rep = prepare_dataset_from_provenance("\n".join(lineas) + "\n", res["provenance"])
        assert len(_filas(rep.csv_text)) == 24
        huecos = _avisos_de_hueco(rep)
        assert len(huecos) == 1 and "1 filas" in huecos[0], rep.compatibility.warnings
        no_vistas = [w for w in rep.compatibility.warnings if "nunca vio" in w]
        assert len(no_vistas) == 1 and "1 filas" in no_vistas[0], rep.compatibility.warnings


class TestUnaRecetaANTERIORSeReproduceComoEntonces:
    """Sin la clave `missing_policy` la receta es anterior a la política de faltantes y su
    criterio era otro: re-preparar tiene que seguir dando lo mismo que entonces."""

    def _sin_la_clave(self, res: dict) -> dict:
        prov = res["provenance"]
        del prov["preparation_spec"]["missing_policy"]
        return prov

    def test_en_one_hot_el_hueco_crudo_sale_a_cero_y_SIN_aviso_como_entonces(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        rep = prepare_dataset_from_provenance(_csv(_N_ONEHOT, extra={4: ""}), self._sin_la_clave(res))
        filas = _filas(rep.csv_text)
        assert {filas[4][c] for c in filas[0] if c.startswith("cat__")} == {"0"}
        assert _avisos_de_hueco(rep) == []

    def test_en_embedding_el_hueco_crudo_NO_se_reubica(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        rep = prepare_dataset_from_provenance(_csv(_N_EMBEDDING, extra={4: ""}), self._sin_la_clave(res))
        assert _filas(rep.csv_text)[4]["cat"] == ""
        assert _avisos_de_hueco(rep) == []

    @pytest.mark.parametrize("n", [_N_ONEHOT, _N_EMBEDDING], ids=["one-hot", "embedding"])
    def test_el_CSV_ENTERO_sale_byte_a_byte_como_entonces(self, n):
        """Lo que toca a la clásica y a los modelos ya guardados: no solo la celda, el CSV
        entero. Lo esperado sale del CSV que escribió la GENERACIÓN (`res["csv_text"]`, que no
        pasa por `prepare_dataset_from_provenance` ni por la receta sin la clave) con la celda
        del hueco escrita a mano según el criterio de entonces: a 0 en one-hot, vacía en
        embedding. Medido además contra el núcleo de main (791501e): el mismo CSV byte a byte."""
        res = generate_project_from_dataset(_csv(n), target_column="target")
        prov = self._sin_la_clave(res)
        filas = list(csv.reader(io.StringIO(res["csv_text"])))
        cabecera = filas[0]
        for j, nombre in enumerate(cabecera):
            if nombre == "cat" or nombre.startswith("cat__"):
                filas[1 + 4][j] = "" if nombre == "cat" else "0"
        out = io.StringIO()
        csv.writer(out).writerows(filas)
        rep = prepare_dataset_from_provenance(_csv(n, extra={4: ""}), prov)
        assert rep.csv_text == out.getvalue()

    @pytest.mark.parametrize("n", [_N_ONEHOT, _N_EMBEDDING], ids=["one-hot", "embedding"])
    def test_la_MARCA_literal_no_depende_de_la_clave_y_se_rescata_con_su_aviso(self, n):
        """La marca la escribe el núcleo (`transformar_fila`), no quien hizo la receta: su
        rescate no depende de `missing_policy`. Antes abortaba como «no conoce» en las dos."""
        res = generate_project_from_dataset(_csv(n), target_column="target")
        vocab = _vocab(res)
        rep = prepare_dataset_from_provenance(
            _csv(n, extra={4: CATEGORIA_FALTANTE}), self._sin_la_clave(res))
        filas = _filas(rep.csv_text)
        if n == _N_EMBEDDING:
            assert filas[4]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        else:
            assert {filas[4][c] for c in filas[0] if c.startswith("cat__")} == {"0"}
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "1 filas" in avisos[0], rep.compatibility.warnings


class TestLasMarcasEnFilasSinObjetivo:
    """Re-auditoría del 04-10 (I-1): contar solo las filas que se escriben hizo que una marca
    que SOLO estaba en filas sin objetivo volviera a ser «un valor que el modelo no conoce» y
    se perdiera el conjunto entero — con `__desconocida__` eso funcionaba antes."""

    @pytest.mark.parametrize("marca", [CATEGORIA_DESCONOCIDA, CATEGORIA_FALTANTE])
    @pytest.mark.parametrize("n", [_N_ONEHOT, _N_EMBEDDING], ids=["one-hot", "embedding"])
    def test_una_marca_solo_en_filas_descartadas_no_aborta_ni_se_cuenta(self, n, marca):
        res = generate_project_from_dataset(_csv(n), target_column="target")
        lineas = _csv(n, extra={4: marca, 7: marca}).splitlines()
        for i in (5, 8):    # las filas 4 y 7 (la línea 0 es la cabecera), sin objetivo
            lineas[i] = lineas[i].rsplit(",", 1)[0] + ","
        rep = prepare_dataset_from_provenance("\n".join(lineas) + "\n", res["provenance"])
        assert rep.compatibility.ok
        assert len(_filas(rep.csv_text)) == 24
        assert [w for w in rep.compatibility.warnings if "nunca vio" in w or "un hueco" in w] == []

    def test_un_valor_NUEVO_de_verdad_sigue_abortando_aunque_solo_este_en_esas_filas(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        lineas = _csv(_N_ONEHOT, extra={4: "v99"}).splitlines()
        lineas[5] = lineas[5].rsplit(",", 1)[0] + ","
        with pytest.raises(DatasetProjectError) as exc:
            prepare_dataset_from_provenance("\n".join(lineas) + "\n", res["provenance"])
        assert "v99" in str(exc.value)


class TestLosTokensDeAusenciaReales:
    """Re-auditoría (I-3): el hueco crudo no es solo `""`. Un `NA` o un `?` (lo que traen los
    ARFF) tienen que ir por el mismo camino, o el aviso diría «código reservado» mientras el
    CSV lleva `""`."""

    @pytest.mark.parametrize("token", ["NA", "?"])
    def test_con_la_heuristica_un_token_de_ausencia_va_al_codigo_reservado(self, token):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        # control: sin tokens declarados la receta no trae la clave y manda la heurística
        assert "tokens_de_ausencia" not in res["provenance"]["preparation_spec"]
        vocab = _vocab(res)
        rep = prepare_dataset_from_provenance(_csv(_N_EMBEDDING, extra={4: token}), res["provenance"])
        assert _filas(rep.csv_text)[4]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        avisos = _avisos_de_hueco(rep)
        assert len(avisos) == 1 and "1 filas" in avisos[0], rep.compatibility.warnings
        assert _valida_contra_su_modelo(res, rep.csv_text).get("ok")

    def test_con_tokens_DECLARADOS_cuenta_el_declarado(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target",
                                            tokens_de_ausencia={"?"})
        vocab = _vocab(res)
        rep = prepare_dataset_from_provenance(_csv(_N_EMBEDDING, extra={4: "?"}), res["provenance"])
        assert _filas(rep.csv_text)[4]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        assert len(_avisos_de_hueco(rep)) == 1


def test_en_one_hot_un_hueco_va_al_grupo_a_cero_aunque_el_vocabulario_traiga_desconocida():
    """La elección de la rama one-hot, con su nombre (re-auditoría, M-6): si el entrenamiento
    trajo la marca `__desconocida__` como dato (la cola de la densa), su columna existe, pero un
    hueco NO va ahí: va a «ninguna de las conocidas», que es lo que dice su aviso. Solo en
    embedding, donde no hay «todo a 0», se usa el código reservado."""
    res = generate_project_from_dataset(_csv(_N_ONEHOT, extra={1: CATEGORIA_DESCONOCIDA}),
                                        target_column="target")
    assert embedding_source_columns(res["mxai"]) == set()
    assert CATEGORIA_DESCONOCIDA in _vocab(res)
    rep = prepare_dataset_from_provenance(_csv(_N_ONEHOT, extra={4: CATEGORIA_FALTANTE}), res["provenance"])
    filas = _filas(rep.csv_text)
    assert {filas[4][c] for c in filas[0] if c.startswith("cat__")} == {"0"}
    assert "ninguna de las conocidas" in _avisos_de_hueco(rep)[0]


# --- Tercera auditoría (04-10): lo que f94efa1 cambia de verdad, y la casi-marca -----------

@pytest.mark.parametrize("marca", [CATEGORIA_DESCONOCIDA, CATEGORIA_FALTANTE])
def test_embedding_SIN_codigo_reservado_y_la_marca_solo_en_filas_descartadas_ya_no_aborta(marca):
    """Lo único que f94efa1 cambia frente a main: en un modelo embedding SIN código reservado
    (una receta anterior a `_reservar_codigo_de_desconocida`), una marca que solo está en filas
    sin objetivo abortaba («no reservó» o «no conoce»), aunque esas filas no se escriben. Ahora
    pasa, y el CSV es el mismo que el de quitar esas filas a mano."""
    res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
    prov = res["provenance"]
    spec = prov["preparation_spec"]
    spec["category_vocabularies"]["cat"] = [
        v for v in spec["category_vocabularies"]["cat"] if v != CATEGORIA_DESCONOCIDA]
    lineas = _csv(_N_EMBEDDING, extra={4: marca, 7: marca}).splitlines()
    for i in (5, 8):    # las filas 4 y 7, sin objetivo
        lineas[i] = lineas[i].rsplit(",", 1)[0] + ","
    rep = prepare_dataset_from_provenance("\n".join(lineas) + "\n", prov)
    a_mano = [l for k, l in enumerate(_csv(_N_EMBEDDING).splitlines()) if k not in (5, 8)]
    esperado = prepare_dataset_from_provenance("\n".join(a_mano) + "\n", prov)
    assert rep.csv_text == esperado.csv_text
    assert [w for w in rep.compatibility.warnings if "nunca vio" in w or "un hueco" in w] == []


@pytest.mark.parametrize("n", [_N_ONEHOT, _N_EMBEDDING], ids=["one-hot", "embedding"])
@pytest.mark.parametrize("casi", ["__FALTANTE__", "__Desconocida__"])
def test_una_CASI_marca_es_un_valor_nuevo_y_sigue_abortando(n, casi):
    """Las marcas se comparan EXACTAS: `__FALTANTE__` no la escribe el núcleo, es un dato del
    usuario que el modelo no conoce. Un filtro relajado (sin mayúsculas) la haría pasar en
    silencio como hueco."""
    res = generate_project_from_dataset(_csv(n), target_column="target")
    with pytest.raises(DatasetProjectError) as exc:
        prepare_dataset_from_provenance(_csv(n, extra={4: casi}), res["provenance"])
    assert casi in str(exc.value) and "no conoce" in str(exc.value)
