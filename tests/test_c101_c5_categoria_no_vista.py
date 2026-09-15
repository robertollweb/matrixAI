# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — una categoría que el entrenamiento no vio perdía el dataset ENTERO.

EL DEFECTO, medido el 2026-09-14 sobre los ARFF reales del protocolo de Fase 0
reproduciendo el camino del motor denso (generar con train+validation,
re-preparar el test con su procedencia):

  · `Moneyball` (columna `Team`, 39 categorías, ~788 filas de train):
    **11 de los 15 pliegues** caídos, por entre 1 y 3 filas de 246.
  · `splice` (`attribute_1`, con una letra de ambigüedad rarísima):
    **15 de 15**.

`preparacion.transformar_fila` (103-C3) marca con `CATEGORIA_DESCONOCIDA` un
valor PRESENTE que el train no vio — el tercer estado que el núcleo declara no
colapsar. Ese centinela llegaba a `prepare_dataset_from_provenance` y el
vocabulario congelado no lo traía, así que se rechazaba como «un valor que el
modelo no conoce»: el núcleo sin reconocer su propia marca.

Y los 4 pliegues de `Moneyball` que pasaban lo hacían **por casualidad**: si la
validación de ese pliegue traía el centinela, entraba al vocabulario como una
categoría más. El mismo dato daba un resultado u otro según qué filas cayeran
en validación, que es peor que fallar siempre.

LA POLÍTICA ELEGIDA, y por qué no es la misma en las dos ramas:

  · **one-hot → el grupo entero a 0** («ninguna de las conocidas»). NO se le
    reserva una columna: una columna que vale 0 en TODAS las filas de train no
    recibe gradiente, así que su peso se queda en la inicialización y al
    predecir metería un peso SIN ENTRENAR donde el todo-a-cero no mete nada. Y
    no se usa la categoría de REFERENCIA, que sería afirmar que esa fila es de
    la clase más frecuente: inventar el dato que falta.
  · **embedding → un código reservado** `__desconocida__` en el vocabulario,
    desde que se genera el proyecto. Ahí no hay «todo a cero»: el CSV lleva el
    ÍNDICE del valor, y un valor sin índice se escribía `""` y el modelo
    rechazaba su propia fila.

Y en los dos casos **se declara**: el informe de compatibilidad dice en qué
columna y en cuántas filas pasó. Un dato que el modelo no puede situar y que
nadie cuenta es media verdad tranquilizadora.

LO QUE NO CAMBIA, y es la mitad que evita el sesgo contrario: un valor NUEVO
del usuario sigue abortando igual que siempre (el vocabulario no se amplía en
silencio, invariante del contrato 62), y un modelo one-hot no gana ni una
columna — su CSV preparado sale byte a byte igual que antes de esto.
"""
from __future__ import annotations

import csv
import io

import pytest

from matrixai.playground import _validate_training_csv
from matrixai.training.categorical import embedding_source_columns
from matrixai.training.dataset_project import (
    DatasetProjectError,
    _sha256_text,
    generate_project_from_dataset,
    prepare_dataset_from_provenance,
)
from matrixai.training.dense_generator import _ONEHOT_MAX
from matrixai.training.preparacion import CATEGORIA_DESCONOCIDA

_ALTURAS = ["150.5", "163.2", "171.9", "158.4", "180.1", "149.7",
            "167.3", "175.8", "152.6", "169.0", "177.4", "161.1"]


def _csv(n_valores: int, filas: int = 26, valor_extra: str | None = None,
         fila_extra: int | None = None) -> str:
    """`cat` con `n_valores` categorías. Si se da `valor_extra`, la fila
    `fila_extra` lo lleva en vez de su categoría — así se fabrica el CSV de
    predicción con una categoría que el modelo no vio."""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["cat", "altura", "target"])
    w.writeheader()
    for i in range(filas):
        valor = f"v{i % n_valores}"
        if valor_extra is not None and i == fila_extra:
            valor = valor_extra
        w.writerow({"cat": valor, "altura": _ALTURAS[i % len(_ALTURAS)],
                    "target": ["si", "no"][i % 2]})
    return out.getvalue()


def _cabecera(texto: str) -> list[str]:
    return texto.splitlines()[0].split(",")


def _filas(texto: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(texto)))


def _vocab(res: dict) -> list[str]:
    return res["provenance"]["preparation_spec"]["category_vocabularies"]["cat"]


def _valida_contra_su_modelo(res: dict) -> None:
    v = _validate_training_csv(
        res["mxai"], res["training_text"], res["csv_text"],
        field_ranges=res.get("field_ranges"),
    )
    assert v.get("ok"), v.get("errors") or v.get("error")


_N_EMBEDDING = _ONEHOT_MAX + 3   # va por embedding
_N_ONEHOT = 3                    # va por one-hot


class TestEmbeddingCodigoReservado:
    """La rama que perdía `Moneyball` y `splice`."""

    def test_el_vocabulario_reserva_el_codigo(self):
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        assert embedding_source_columns(res["mxai"]) == {"cat"}
        assert _vocab(res)[-1] == CATEGORIA_DESCONOCIDA

    def test_el_modelo_generado_acepta_su_propio_CSV_con_el_codigo_dentro(self):
        """Reservar un código que el modelo luego no admite sería cambiar el
        fallo de sitio, no arreglarlo."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        assert res["ok"]
        _valida_contra_su_modelo(res)

    def test_una_categoria_no_vista_YA_NO_pierde_el_dataset(self):
        """EL DEFECTO. Antes: DatasetProjectError «trae valores que el modelo
        no conoce: ['__desconocida__']», y con él los 11 pliegues de
        `Moneyball`."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        crudo = _csv(_N_EMBEDDING, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        assert rep.compatibility.ok

    def test_el_indice_escrito_es_el_del_codigo_reservado(self):
        """Mitad de contenido: que no aborte no basta, hay que mirar QUÉ
        escribe. Una celda vacía también pasaría el aserto de arriba y el
        modelo rechazaría la fila después."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        vocab = _vocab(res)
        crudo = _csv(_N_EMBEDDING, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        filas = _filas(rep.csv_text)
        assert filas[4]["cat"] == str(vocab.index(CATEGORIA_DESCONOCIDA))
        # ...y la fila de al lado, que sí tenía una categoría conocida, no se
        # movió: el arreglo no puede mandar todo al código reservado.
        assert filas[5]["cat"] == str(vocab.index("v5"))

    def test_el_CSV_de_prediccion_no_deja_ninguna_celda_vacia(self):
        """Lo que reventaba de verdad: el índice inexistente se escribía `""`
        y el modelo rechazaba su propia fila."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        crudo = _csv(_N_EMBEDDING, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        assert _cabecera(rep.csv_text) == _cabecera(res["csv_text"])
        vacias = [(i, k) for i, f in enumerate(_filas(rep.csv_text))
                  for k, v in f.items() if v is None or v.strip() == ""]
        assert vacias == []

    def test_el_codigo_reservado_va_SIEMPRE_al_final(self):
        """Los 4 pliegues «buenos» de `Moneyball` metían el centinela en el
        vocabulario POR CASUALIDAD, en la posición en la que apareciera en el
        CSV — así que el índice del código cambiaba de un pliegue a otro. Aquí
        el CSV de ajuste ya lo trae en mitad de los datos y tiene que salir una
        sola vez, y la última."""
        crudo = _csv(_N_EMBEDDING, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=1)
        res = generate_project_from_dataset(crudo, target_column="target")
        vocab = _vocab(res)
        assert vocab.count(CATEGORIA_DESCONOCIDA) == 1
        assert vocab[-1] == CATEGORIA_DESCONOCIDA

    def test_un_modelo_SIN_el_codigo_reservado_lo_DICE(self):
        """Un modelo guardado ANTES de esto va por embedding y su vocabulario
        no reserva nada: la fila no se puede representar. Sigue abortando —no
        se escribe una celda vacía en silencio— pero diciendo qué falta, no
        llamándolo «un valor que el modelo no conoce»."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        prov = res["provenance"]
        spec = prov["preparation_spec"]
        spec["category_vocabularies"]["cat"] = [
            v for v in spec["category_vocabularies"]["cat"]
            if v != CATEGORIA_DESCONOCIDA]
        crudo = _csv(_N_EMBEDDING, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        with pytest.raises(DatasetProjectError) as exc:
            prepare_dataset_from_provenance(crudo, prov)
        assert "EMBEDDING" in str(exc.value)
        assert "1 filas" in str(exc.value)


class TestOneHotNingunaDeLasConocidas:
    """La otra rama, y la línea que NO hace lo obvio."""

    def test_una_categoria_no_vista_YA_NO_pierde_el_dataset(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        assert embedding_source_columns(res["mxai"]) == set()
        crudo = _csv(_N_ONEHOT, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        assert rep.compatibility.ok

    def test_el_grupo_entero_sale_A_CERO_y_el_de_al_lado_no(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        crudo = _csv(_N_ONEHOT, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        filas = _filas(rep.csv_text)
        grupo = [c for c in _cabecera(rep.csv_text) if c.startswith("cat__")]
        assert len(grupo) == _N_ONEHOT
        # Mitad positiva: la fila con la categoría no vista, a cero entera...
        assert {filas[4][c] for c in grupo} == {"0"}
        # ...y mitad negativa: una fila normal sigue marcando SU columna, o un
        # CSV con el grupo entero en blanco pasaría igual.
        assert sum(float(filas[5][c]) for c in grupo) == 1.0

    def test_NO_se_usa_la_categoria_de_REFERENCIA(self):
        """«Ninguna de las conocidas» no es «la más frecuente». Rellenarla con
        la referencia sería inventar el dato que falta — la misma regla que
        impide imputar el objetivo."""
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        crudo = _csv(_N_ONEHOT, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        filas = _filas(rep.csv_text)
        assert filas[4]["cat__v0"] == "0"

    def test_un_modelo_one_hot_NO_gana_una_columna_reservada(self):
        """LA LÍNEA QUE NO HACE LO OBVIO, con su nombre. Reservarle también
        aquí una columna sería simétrico y estaría MAL: valdría 0 en todas las
        filas de train, no recibiría gradiente, y al predecir metería un peso
        sin entrenar donde el todo-a-cero no mete nada. Además cambiaría el CSV
        preparado de TODOS los datasets con categóricas, moviendo números ya
        medidos, a cambio de nada."""
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        assert CATEGORIA_DESCONOCIDA not in _vocab(res)
        assert [c for c in _cabecera(res["csv_text"]) if c.startswith("cat__")] == [
            "cat__v0", "cat__v1", "cat__v2"]

    def test_el_enrutado_a_embedding_NO_se_mueve_por_el_codigo(self):
        """La trampa de tocar esto: el generador manda el prompt ENTERO al
        camino composite en cuanto UNA categórica pasa de `_ONEHOT_MAX`. Con el
        código reservado contado, una columna de exactamente `_ONEHOT_MAX`
        valores pasaría a 13 y el modelo cambiaría de forma sin que nadie lo
        pidiera. El código se añade SOLO si alguna ya pasa el umbral sin
        contarlo."""
        res = generate_project_from_dataset(_csv(_ONEHOT_MAX), target_column="target")
        assert embedding_source_columns(res["mxai"]) == set()
        assert CATEGORIA_DESCONOCIDA not in _vocab(res)


class TestLoQueNoCambia:
    def test_un_valor_NUEVO_del_usuario_sigue_abortando(self):
        """La invariante del contrato 62: el vocabulario no se amplía en
        silencio. `__desconocida__` es la marca del núcleo; `v99` es un dato
        nuevo del usuario y NO es lo mismo."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        crudo = _csv(_N_EMBEDDING, valor_extra="v99", fila_extra=4)
        with pytest.raises(DatasetProjectError) as exc:
            prepare_dataset_from_provenance(crudo, res["provenance"])
        assert "no conoce" in str(exc.value) and "v99" in str(exc.value)

    def test_lo_mismo_en_one_hot(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        crudo = _csv(_N_ONEHOT, valor_extra="v99", fila_extra=4)
        with pytest.raises(DatasetProjectError) as exc:
            prepare_dataset_from_provenance(crudo, res["provenance"])
        assert "v99" in str(exc.value)

    def test_repreparar_el_MISMO_crudo_sigue_saliendo_byte_a_byte(self):
        for n in (_N_ONEHOT, _N_EMBEDDING):
            crudo = _csv(n)
            res = generate_project_from_dataset(crudo, target_column="target")
            rep = prepare_dataset_from_provenance(crudo, res["provenance"])
            assert rep.csv_text == res["csv_text"], n
            assert _sha256_text(rep.csv_text) == res["provenance"]["prepared_csv_sha256"]


class TestSeDeclara:
    """Un dato que el modelo no puede situar y que nadie cuenta es media
    verdad tranquilizadora."""

    def test_el_informe_dice_la_columna_y_CUANTAS_filas(self):
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        crudo = _csv(_N_ONEHOT, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        avisos = [w for w in rep.compatibility.warnings if "nunca vio" in w]
        assert len(avisos) == 1, rep.compatibility.warnings
        assert "'cat'" in avisos[0]
        assert "1 filas" in avisos[0]

    def test_el_recuento_es_el_REAL_no_un_booleano_disfrazado(self):
        """Con tres filas afectadas el aviso tiene que decir tres."""
        res = generate_project_from_dataset(_csv(_N_ONEHOT), target_column="target")
        crudo = _csv(_N_ONEHOT)
        filas = crudo.splitlines()
        for i in (2, 4, 6):
            partes = filas[i].split(",")
            partes[0] = CATEGORIA_DESCONOCIDA
            filas[i] = ",".join(partes)
        rep = prepare_dataset_from_provenance("\n".join(filas) + "\n", res["provenance"])
        avisos = [w for w in rep.compatibility.warnings if "nunca vio" in w]
        assert len(avisos) == 1 and "3 filas" in avisos[0], avisos

    def test_sin_ninguna_categoria_no_vista_NO_se_avisa_de_nada(self):
        """El sesgo contrario: un aviso que sale siempre no informa."""
        crudo = _csv(_N_ONEHOT)
        res = generate_project_from_dataset(crudo, target_column="target")
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        assert [w for w in rep.compatibility.warnings if "nunca vio" in w] == []

    def test_en_EMBEDDING_tambien_se_declara(self):
        """La trampa de este arreglo: en embedding el código SÍ está en el
        vocabulario, así que el centinela no aparece como «valor nuevo» y las
        filas que el modelo no puede situar se colarían sin que nadie las
        contara. Salvar el dataset y no decir cuántas filas se salvaron a
        ciegas sería media verdad tranquilizadora."""
        res = generate_project_from_dataset(_csv(_N_EMBEDDING), target_column="target")
        crudo = _csv(_N_EMBEDDING, valor_extra=CATEGORIA_DESCONOCIDA, fila_extra=4)
        rep = prepare_dataset_from_provenance(crudo, res["provenance"])
        avisos = [w for w in rep.compatibility.warnings if "nunca vio" in w]
        assert len(avisos) == 1, rep.compatibility.warnings
        assert "1 filas" in avisos[0]
        # Y dice CÓMO se codificó, que en embedding no es lo mismo que en
        # one-hot: el vector de ese código no se entrenó con ningún ejemplo.
        assert "no se entrenó" in avisos[0]
