# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C5 — un NIVEL DECLARADO no es un dato ausente.

EL DEFECTO, medido el 2026-09-15 sobre `house_prices_nominal` (OpenML 42563),
el único dataset de los 40 del protocolo al que le pasa y los 15 únicos fallos
de toda la pasada registrada:

  1. Su cabecera ARFF declara `@ATTRIBUTE MasVnrType {BrkCmn, BrkFace, None,
     Stone}`. `None` es un NIVEL —«sin revestimiento de mampostería»— y sale en
     **864 de 1.460 filas, el 59,18 %**. Los ausentes de verdad son 8 y llegan
     como `?`.
  2. `ajustar_preparacion(admite_faltantes=False)` —lo que pide el motor denso—
     rellena esos 8 con la cadena reservada `__faltante__`.
  3. El motor denso escribe un CSV con esas filas y llama a
     `generate_project_from_dataset`, que lo relee.
  4. `_is_null('None')` daba `True` por la heurística `_NULL_TOKENS`: la
     columna salía con **864 ausentes y cardinalidad 4** (el nivel `None`
     desaparecía) Y con el centinela dentro.
  5. La guarda del centinela levantaba `DatasetProjectError`. **Los 15 pliegues
     perdidos en 0,9 s de un presupuesto de 120.**

LA GUARDA NO ERA EL DEFECTO: hizo su trabajo. La ambigüedad existía de verdad,
creada por el paso 4. Taparla habría convertido un fallo ruidoso en un
resultado equivocado callado, con el 59 % de una columna imputado.

LA HEURÍSTICA TAMPOCO ES EL DEFECTO, y por eso no se borra: en un CSV tecleado
a mano «none», «NA» o «-» casi siempre SÍ significan «no había dato». Lo que
faltaba era la forma de decir «aquí los niveles están declarados», y eso es
`tokens_de_ausencia` (ver `dataset_analysis._is_null`): quien produce el CSV y
sabe cómo marca la ausencia lo declara, y entonces la heurística no se aplica.

SE PRUEBAN LOS DOS LADOS, porque arreglar un sesgo puede crear el contrario:
que un nivel declarado deje de leerse como nulo, Y que sin declaración la
heurística siga contando exactamente lo que contaba. Y se prueba que la guarda
del centinela SIGUE protegiendo cuando la ambigüedad es real.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from matrixai.training.dataset_analysis import analyze_dataset_csv
from matrixai.training.dataset_project import (
    DatasetProjectError,
    _prepare_v1,
    generate_project_from_dataset,
)
from matrixai.training.preparacion import CATEGORIA_FALTANTE

#: El ARFF real. Vive fuera del repo (lo descarga el protocolo de Fase 0), así
#: que la prueba que lo usa se salta si no está — pero las de forma sintética,
#: que son las que llevan los asertos finos, corren SIEMPRE.
ARFF_REAL = Path("/home/deployer/fase0_openml_datos/arff/42563.arff")

_N = 24
_SUPERFICIE = [str(80 + i * 7) for i in range(_N)]
_PRECIO = [str(100_000 + i * 3_300) for i in range(_N)]


def _csv(revestimiento: list[str]) -> str:
    """Un CSV con la MISMA forma que el que pierde el motor denso: una
    categórica cuyo nivel mayoritario se llama `None`, y el centinela que la
    preparación ya escribió en los huecos de verdad."""
    assert len(revestimiento) == _N
    buffer = io.StringIO()
    escritor = csv.writer(buffer)
    escritor.writerow(["revestimiento", "superficie", "precio"])
    for rev, sup, pre in zip(revestimiento, _SUPERFICIE, _PRECIO):
        escritor.writerow([rev, sup, pre])
    return buffer.getvalue()


#: Lo que el motor denso le entrega al núcleo: 14 filas del nivel legítimo
#: `None`, dos niveles más, y 2 filas que la preparación marcó `__faltante__`.
#: NINGUNA celda vacía — el motor escribe la ausencia así y solo así.
_COMO_LO_ESCRIBE_EL_MOTOR = (
    ["None"] * 14 + ["Ladrillo"] * 5 + ["Piedra"] * 3 + [CATEGORIA_FALTANTE] * 2
)


class TestElNivelDeclaradoCuentaComoCategoria:
    """Con la declaración, `None` es la categoría que es."""

    def test_sin_declarar_el_nivel_None_se_contaba_como_ausente(self):
        """EL DEFECTO, tal cual. No es una prueba del arreglo: es la medida de
        lo que la heurística hace, y está aquí para que el arreglo se compare
        contra un número y no contra un recuerdo."""
        info = analyze_dataset_csv(_csv(_COMO_LO_ESCRIBE_EL_MOTOR))["columns"]["revestimiento"]
        assert info["null_count"] == 14
        assert info["cardinality"] == 3  # `None` NO está: desapareció como nivel

    def test_declarando_la_marca_de_ausencia_el_nivel_sobrevive(self):
        info = analyze_dataset_csv(
            _csv(_COMO_LO_ESCRIBE_EL_MOTOR), tokens_de_ausencia={""},
        )["columns"]["revestimiento"]
        assert info["null_count"] == 0
        assert info["cardinality"] == 4  # None, Ladrillo, Piedra, __faltante__

    def test_el_proyecto_se_genera_y_las_filas_de_None_cuentan_como_su_categoria(self):
        """La prueba de producto: no basta con que no levante, hay que ver las
        14 filas escritas en la columna que les toca. Un aserto de «no lanzó»
        lo pasaría un CSV preparado con la columna entera a cero."""
        resultado = generate_project_from_dataset(
            _csv(_COMO_LO_ESCRIBE_EL_MOTOR), "precio", locale="es",
            tokens_de_ausencia={""},
        )
        receta = resultado["provenance"]["preparation_spec"]
        assert receta["category_vocabularies"]["revestimiento"] == [
            "None", "Ladrillo", "Piedra", CATEGORIA_FALTANTE]

        filas = list(csv.DictReader(io.StringIO(resultado["csv_text"])))
        assert len(filas) == _N
        def a_uno(columna: str) -> int:
            return sum(1 for f in filas if f[columna] not in ("0", "0.0", ""))
        assert a_uno("revestimiento__none") == 14
        assert a_uno("revestimiento__ladrillo") == 5
        assert a_uno("revestimiento__piedra") == 3
        assert a_uno("revestimiento__faltante") == 2

    def test_sin_declarar_el_mismo_CSV_sigue_levantando(self):
        """El camino por defecto NO cambia, ni siquiera para este CSV: sin
        declaración la ambigüedad es real —nadie ha dicho que `None` sea un
        nivel— y callarla sería inventarse la respuesta."""
        with pytest.raises(DatasetProjectError) as exc:
            generate_project_from_dataset(
                _csv(_COMO_LO_ESCRIBE_EL_MOTOR), "precio", locale="es")
        assert CATEGORIA_FALTANTE in str(exc.value)


class TestLaHeuristicaSigueIntactaDondeSIDebeAplicarse:
    """El CSV tecleado a mano, que es para el que se escribió `_NULL_TOKENS`."""

    #: Sin esquema, nadie ha declarado nada: los 10 marcadores de ausencia
    #: habituales significan lo que siempre significaron.
    _A_MANO = (["azul"] * 8 + ["rojo"] * 6 + ["none"] * 4
               + ["NA"] * 3 + ["-"] * 2 + ["?"] * 1)

    def test_un_CSV_sin_declaracion_cuenta_sus_nulos_como_siempre(self):
        info = analyze_dataset_csv(_csv(self._A_MANO))["columns"]["revestimiento"]
        assert info["null_count"] == 10   # none(4) + NA(3) + -(2) + ?(1)
        assert info["cardinality"] == 2   # azul, rojo

    def test_y_la_heuristica_no_distingue_mayusculas(self):
        mezclado = (["azul"] * 14 + ["None"] * 4 + ["n/a"] * 3
                    + ["NULL"] * 2 + ["NaN"] * 1)
        info = analyze_dataset_csv(_csv(mezclado))["columns"]["revestimiento"]
        assert info["null_count"] == 10
        assert info["cardinality"] == 1

    def test_una_declaracion_es_LITERAL_y_no_baja_a_minusculas(self):
        """Una declaración no es otra heurística. Si el origen dice que su
        marca es `?`, entonces `NA` es un valor — y `None` también."""
        rev = ["None"] * 12 + ["NA"] * 5 + ["Piedra"] * 5 + ["?"] * 2
        declarado = analyze_dataset_csv(
            _csv(rev), tokens_de_ausencia={"?"})["columns"]["revestimiento"]
        assert declarado["null_count"] == 2
        assert declarado["cardinality"] == 3   # None, NA, Piedra: los tres valen

        heuristico = analyze_dataset_csv(_csv(rev))["columns"]["revestimiento"]
        assert heuristico["null_count"] == 19  # None(12) + NA(5) + ?(2)
        assert heuristico["cardinality"] == 1


class TestLaGuardaDelCentinelaSigueProtegiendo:
    """El arreglo no puede volver falso el aviso que sí hacía falta."""

    #: Ambigüedad de VERDAD: celdas ausentes por la marca DECLARADA (vacías) y
    #: además un `__faltante__` literal. No se pueden distinguir, y eso hay que
    #: decirlo aunque la declaración esté puesta.
    _AMBIGUO = (["None"] * 12 + ["Ladrillo"] * 5 + ["Piedra"] * 3
                + [CATEGORIA_FALTANTE] * 2 + [""] * 2)

    def test_con_declaracion_y_ambiguedad_real_sigue_levantando(self):
        with pytest.raises(DatasetProjectError) as exc:
            generate_project_from_dataset(
                _csv(self._AMBIGUO), "precio", locale="es", tokens_de_ausencia={""})
        assert CATEGORIA_FALTANTE in str(exc.value)

    def test_sin_declaracion_y_ambiguedad_real_sigue_levantando(self):
        with pytest.raises(DatasetProjectError) as exc:
            generate_project_from_dataset(_csv(self._AMBIGUO), "precio", locale="es")
        assert CATEGORIA_FALTANTE in str(exc.value)


class TestLaRecetaCongelaComoSeLeyoLaAusencia:
    """Si la receta no lo declarara, re-preparar volvería a la heurística y el
    CSV re-preparado dejaría de ser el que entrenó el modelo."""

    def test_la_receta_guarda_la_declaracion_y_re_preparar_reproduce_el_CSV(self):
        texto = _csv(_COMO_LO_ESCRIBE_EL_MOTOR)
        resultado = generate_project_from_dataset(
            texto, "precio", locale="es", tokens_de_ausencia={""})
        receta = resultado["provenance"]["preparation_spec"]
        assert receta["tokens_de_ausencia"] == [""]

        filas = list(csv.DictReader(io.StringIO(texto)))
        assert _prepare_v1(filas, receta).text == resultado["csv_text"]

    def test_una_receta_SIN_la_clave_re_prepara_con_la_heuristica_de_entonces(self):
        """AUSENTE no es VACÍA: un modelo anterior a este arreglo se re-prepara
        con el criterio con el que nació, no con el de hoy."""
        texto = _csv(_COMO_LO_ESCRIBE_EL_MOTOR)
        resultado = generate_project_from_dataset(
            texto, "precio", locale="es", tokens_de_ausencia={""})
        receta = dict(resultado["provenance"]["preparation_spec"])
        assert receta.pop("tokens_de_ausencia") == [""]

        filas = list(csv.DictReader(io.StringIO(texto)))
        con_heuristica = _prepare_v1(filas, receta).text
        assert con_heuristica != resultado["csv_text"]
        # Y la diferencia es exactamente la que el defecto producía: las 14
        # filas del nivel `None` se escriben como el centinela.
        columnas = list(csv.DictReader(io.StringIO(con_heuristica)))
        a_uno = sum(1 for f in columnas if f["revestimiento__faltante"] not in ("0", "0.0"))
        assert a_uno == 16   # las 2 de verdad + las 14 que sí tenían dato

    def test_una_receta_que_declara_el_conjunto_VACIO_no_es_lo_mismo_que_no_declarar(self):
        """«Aquí no falta nada» es una respuesta; «nadie lo miró» no lo es."""
        texto = _csv(["None"] * 14 + ["Ladrillo"] * 6 + ["Piedra"] * 4)
        declarado = analyze_dataset_csv(texto, tokens_de_ausencia=set())
        assert declarado["columns"]["revestimiento"]["null_count"] == 0
        assert declarado["columns"]["revestimiento"]["cardinality"] == 3
        heuristico = analyze_dataset_csv(texto)
        assert heuristico["columns"]["revestimiento"]["null_count"] == 14


class TestElOBJETIVOTambienPuedeTenerUnNivelLlamadoNone:
    """SEGUNDO HUECO DE CABLEADO, encontrado midiendo el arreglo con su control
    el 2026-09-15. `analyze_dataset_csv` ya recibía la declaración, pero
    `objetivo.confirmar_desde_csv` volvía a medir los valores CRUDOS del
    objetivo sobre las filas (`_distinct_non_null` y `no_entrenable_en_train`)
    con la heurística: un objetivo binario `None`/`Piedra` salía con el bloqueo
    `objetivo_con_una_sola_clase` y, antes de eso, `generate_project_from_
    dataset` levantaba «menos de 2 valores distintos».

    Se prueba CONTRA UN CONTROL conocido-bueno —el mismo CSV con los niveles
    `Ladrillo`/`Piedra`— porque sin él «sale un bloqueo» no distingue un defecto
    de una confirmación incompleta: el control también deja preguntas abiertas.
    """

    _RESPUESTAS = dict(unidad_de_observacion="una vivienda", clase_positiva="Piedra",
                       momento_de_prediccion="al tasar", uso_previsto="demo")

    @staticmethod
    def _csv_con_objetivo(niveles: list[str]) -> str:
        buffer = io.StringIO()
        escritor = csv.writer(buffer)
        escritor.writerow(["superficie", "revestimiento"])
        for i, nivel in enumerate(niveles):
            escritor.writerow([str(80 + i * 3), nivel])
        return buffer.getvalue()

    _CONTROL = ["Ladrillo"] * 15 + ["Piedra"] * 15
    _CASO = ["None"] * 15 + ["Piedra"] * 15

    def _bloqueos(self, niveles: list[str], tokens):
        resultado = generate_project_from_dataset(
            self._csv_con_objetivo(niveles), "revestimiento", locale="es",
            tokens_de_ausencia=tokens, **self._RESPUESTAS)
        return [b["clave"] for b in (resultado["confirmacion"]["bloqueos"] or [])]

    def test_el_control_no_bloquea(self):
        assert self._bloqueos(self._CONTROL, {""}) == []
        assert self._bloqueos(self._CONTROL, None) == []

    def test_declarado_el_objetivo_con_nivel_None_se_comporta_COMO_EL_CONTROL(self):
        assert self._bloqueos(self._CASO, {""}) == self._bloqueos(self._CONTROL, {""})

    def test_sin_declarar_el_mismo_objetivo_sigue_sin_poder_generarse(self):
        """El camino por defecto no cambia: sin declaración, `None` es ausente
        y un objetivo con un solo valor no se puede clasificar."""
        with pytest.raises(DatasetProjectError, match="menos de 2 valores distintos"):
            generate_project_from_dataset(
                self._csv_con_objetivo(self._CASO), "revestimiento", locale="es",
                **self._RESPUESTAS)

    def test_no_entrenable_en_train_usa_el_criterio_declarado(self):
        """La función suelta, por su nombre: es la que produjo el bloqueo y
        vive fuera del flujo, así que una prueba del flujo sola la dejaría
        cubierta solo de rebote.

        Los valores van INTERCALADOS a propósito. Con los 12 `None` primero, el
        train de ratio 0,5 no tendría bajo la heurística NINGÚN valor no nulo, y
        eso la función lo distingue de «una sola clase» y devuelve `None` — el
        primer aserto que escribí daba rojo por eso, y el fallo estaba en el
        ASERTO, no en el producto. Intercalados, el train ve las dos cosas y la
        diferencia entre los dos criterios queda aislada."""
        from matrixai.training.objetivo import no_entrenable_en_train

        valores = ["None", "Piedra"] * 12
        # Heurística: los `None` no cuentan, queda una sola clase -> BLOQUEO.
        bloqueo = no_entrenable_en_train(valores, objetivo="y", ratio=0.5)
        assert bloqueo is not None
        assert bloqueo.clave == "objetivo_con_una_sola_clase"
        # Declarado: son dos clases, y no hay nada que bloquear.
        assert no_entrenable_en_train(valores, objetivo="y", ratio=0.5,
                                      tokens_de_ausencia={""}) is None


@pytest.mark.skipif(not ARFF_REAL.exists(), reason="el ARFF del protocolo no está en esta máquina")
class TestElFicheroQueDeVerdadSePerdia:
    """`house_prices_nominal` por el camino REAL del motor denso. Probar la
    función no es probar el producto: aquí entra el ARFF tal cual, lo lee el
    lector del protocolo, lo prepara el núcleo con `admite_faltantes=False` y
    lo escribe el propio `_escribir_csv` del motor."""

    @staticmethod
    def _csv_como_lo_escribe_el_motor() -> tuple[str, str]:
        import sys
        for ruta in ("/home/deployer/matrixAI/benchmarks/fase0",
                     "/home/deployer/matrixai-engines/src"):
            if ruta not in sys.path:
                sys.path.insert(0, ruta)
        import lector_arff
        from matrixai_engines.motores.densa import _columnas, _escribir_csv
        from matrixai.training.preparacion import (ajustar_preparacion,
                                                   tipar_columnas_numericas,
                                                   transformar_fila)

        leido = lector_arff.cargar(ARFF_REAL, objetivo_declarado="SalePrice",
                                   n_columnas_declaradas=80)
        objetivo = leido.objetivo
        filas = [f for f in leido.filas if f[objetivo] is not None]
        predictores = tuple(sorted(k for k in filas[0] if k not in ("row_id", objetivo)))
        tipar_columnas_numericas(filas, predictores)
        politica = ajustar_preparacion(filas, objetivo=objetivo, columnas=predictores,
                                       admite_categoricas=True, admite_faltantes=False)
        preparadas = []
        for fila in filas:
            t = transformar_fila(fila, politica)
            t["row_id"] = fila["row_id"]
            t[objetivo] = fila[objetivo]
            preparadas.append(t)
        columnas = _columnas([k for k in preparadas[0] if k not in ("row_id", objetivo)])
        return _escribir_csv(columnas, objetivo, preparadas,
                             [f[objetivo] for f in preparadas]), objetivo

    def test_el_ARFF_declara_None_como_nivel_y_son_864_filas(self):
        texto = ARFF_REAL.read_text(encoding="utf-8", errors="replace")
        assert "@ATTRIBUTE MasVnrType {BrkCmn, BrkFace, None, Stone}" in texto
        csv_texto, _ = self._csv_como_lo_escribe_el_motor()
        filas = list(csv.DictReader(io.StringIO(csv_texto)))
        assert sum(1 for f in filas if f["MasVnrType"] == "None") == 864
        assert sum(1 for f in filas if f["MasVnrType"] == CATEGORIA_FALTANTE) == 8

    def test_sin_declarar_se_pierde_el_dataset_entero(self):
        csv_texto, objetivo = self._csv_como_lo_escribe_el_motor()
        with pytest.raises(DatasetProjectError) as exc:
            generate_project_from_dataset(csv_texto, objetivo, locale="es")
        assert "MasVnrType" in str(exc.value)

    def test_declarando_la_marca_del_motor_las_864_filas_son_su_categoria(self):
        from matrixai_engines.motores.densa import TOKENS_DE_AUSENCIA_DEL_CSV

        csv_texto, objetivo = self._csv_como_lo_escribe_el_motor()
        resultado = generate_project_from_dataset(
            csv_texto, objetivo, locale="es",
            tokens_de_ausencia=set(TOKENS_DE_AUSENCIA_DEL_CSV))

        esquema = resultado["provenance"]["schema_final"]["MasVnrType"]
        assert esquema["null_count"] == 0
        assert esquema["cardinality"] == 5   # los 4 niveles + el centinela

        vocabulario = resultado["provenance"]["preparation_spec"][
            "category_vocabularies"]["MasVnrType"]
        assert "None" in vocabulario
        indice = {v: i for i, v in enumerate(vocabulario)}
        safe = resultado["provenance"]["preparation_spec"]["feature_name_map"]["MasVnrType"]
        filas = list(csv.DictReader(io.StringIO(resultado["csv_text"])))
        escritas = sum(1 for f in filas if f[safe] == str(indice["None"]))
        assert escritas == 864
        assert sum(1 for f in filas
                   if f[safe] == str(indice[CATEGORIA_FALTANTE])) == 8
