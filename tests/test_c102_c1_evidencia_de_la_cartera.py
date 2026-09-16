# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""102-C1 — LO QUE LA CARTERA AFIRMA SE RE-DERIVA, no se cree.

HALLAZGO GRAVE de la auditoría externa del 2026-09-12, y la razón de que este
fichero exista: la entrada de cartera de `matrixai_engines` afirmaba «lightgbm
10/12 = 0,833 >= 0,80, CUMPLE» sobre la pasada real, y **ningún test de ningún
repositorio leía ese JSON**. Lo único que se comprobaba era que la CADENA
«10/12» apareciera en el texto de la evidencia. El auditor hundió el `auroc`
de lightgbm en dos datasets del JSON —la regla pasaba a 8/12 = 0,667, NO
CUMPLE— y las dos suites siguieron VERDES: la promoción de un motor a
«soportado» descansaba sobre una frase, no sobre una medición.

**RE-SELLADA EL 2026-09-16, SOBRE 101-C5 (LA PASADA AMPLIA, 40 DATASETS).**
Este fichero se reescribió CONTRA LA EVIDENCIA NUEVA. Y una trampa concreta
merece quedar escrita porque estaba puesta y medida, dos veces:

1. La versión del 2026-09-14 tenía un ayudante que emparejaba dos pasadas que
   la entrada de entonces ya no citaba — dejarlo habría puesto el fichero
   VERDE midiendo otra cosa, que es peor que dejarlo rojo. Se retiró junto con
   la afirmación que re-derivaba.

2. **ESTA VEZ la trampa estaba en la MÉTRICA, no en el fichero.** El JSON de
   101-C5 guarda cada medida DOS VECES por registro —`r["auroc"]` (o
   `r["accuracy"]`, `r["r2"]`…) en el primer nivel, y otra copia dentro de
   `r["metricas"]`— y `aplicar_regla_de_cierre` lee SOLO la del primer nivel.
   Sabotear `r["metricas"][métrica]` no mueve el veredicto: hay que tocar
   `r[métrica]`. Y la métrica NO es «auroc» para todos los cuarenta: sale de
   `metrica_de_cierre_por_dataset`, con `accuracy` en multiclase y `r2` en
   regresión. Un ayudante que leyera siempre `r["auroc"]` —como hacía la
   versión anterior de este fichero, escrita cuando los doce datasets eran
   todos binarios— habría medido **mal en silencio** el primer puesto
   discutido de `catboost` en `okcupid-stem` (que se cierra con `accuracy`, y
   donde `auroc` vale `None`): con `auroc` fijo, ese dataset sale con CERO
   pliegues comparables y el aserto `0 > 0` es `False`, así que la dispersión
   dejaría de discutir un dataset que sí discute. Medido antes de escribir
   este fichero, no supuesto.

POR QUÉ ESTE TEST VIVE EN `matrixAI` Y NO EN `matrixai-engines`, que es donde
está la afirmación:

1. Aquí viven las DOS cosas que hay que volver a juntar: el JSON de la pasada
   (`benchmarks/fase0/pasada_amplia_101_c5_resultado.json`) y la REGLA que lo
   juzga (`benchmarks/fase0/protocolo.py`, registrada con hash por 101-C1
   antes de medir). Un test que re-deriva un número se pone donde viven el
   número y su fórmula, no donde vive la frase que los cita.

2. `benchmarks/` NO se publica en PyPI (invariante 1 del 102). Un test dentro
   de `matrixai-engines` —que es un paquete que se instala solo— solo podría
   alcanzar ese JSON adivinando una ruta a un repositorio hermano, y tendría
   que saltarse cuando no la encuentra. Un salto silencioso es exactamente el
   banco de pruebas sin dientes que este hallazgo denuncia: volvería a estar
   verde sin haber medido nada.

3. La dirección contraria sí funciona y está medida: `matrixai_engines` es
   importable desde la suite de `matrixAI`, así que el test puede leer LA
   AFIRMACIÓN en su fuente (`cartera.CARTERA_APROBADA`) en vez de copiarla.

**RE-DERIVAR SIEMPRE DESDE `art["resultados"]` CON `veredicto_con_su_alcance`,
NUNCA LEYENDO `art["alcance_y_veredicto"]["por_motor"]`.** Ese bloque viene
precalculado por el mismo guion que se está auditando: comparar la ficha
contra él no prueba nada sobre los resultados crudos. Es literalmente el
defecto que este fichero existe para cerrar.

QUÉ LO PONE ROJO, que es lo que define si tiene dientes:
  * editar el JSON de la pasada (los `auroc`/`accuracy`/`r2` de PRIMER NIVEL,
    un `estado`, un dataset) para CUALQUIERA de los tres motores aprobados;
  * editar la regla de cierre registrada, o su aplicación;
  * editar el texto de la evidencia de cualquiera de las tres entradas para
    que diga otro número;
  * promover o despromover un motor sin que la medición lo acompañe;
  * rellenar un campo del `alcance` con algo que la pasada no dice — o
    vaciarlo cuando la pasada sí lo dice.

Las cifras NO están escritas en este fichero a propósito: se leen del texto de
la cartera con expresiones regulares y se comparan contra lo re-derivado. Un
número a mano aquí sería un tercer sitio declarando lo mismo, y la próxima vez
divergiría este.

**QUÉ SE RETIRA Y POR QUÉ, con nombre.** Dos tests de la versión anterior
comprobaban exactamente el mismo invariante que
`test_esta_en_la_cartera_EXACTAMENTE_quien_cumple_la_regla` de aquí abajo, y
la evidencia nueva les quitó el suelo textual en el que se apoyaban:

  * `test_el_UNICO_QUE_CUMPLE_es_una_medida_y_no_una_forma_de_hablar` —
    re-derivaba la frase «Es el UNICO que cumple», y esa frase ya no está en
    ninguna entrada (medido con grep antes de escribir esto: cero apariciones
    de «UNICO» en las tres evidencias). Con tres motores aprobados «el único»
    ni siquiera podría ser cierto. Lo que vigilaba —que la cartera no tenga ni
    uno de más ni uno de menos— es EXACTAMENTE lo que mide
    `test_esta_en_la_cartera_EXACTAMENTE_quien_cumple_la_regla`, así que
    escribirlo aparte sería el mismo aserto dos veces.
  * `test_los_otros_motores_NO_cumplen_la_regla_y_NO_estan_en_la_cartera` —
    extraía de la evidencia, con regex, la fracción exacta de cada motor NO
    aprobado (`xgboost 30/40 = 0,750`, etc.). Medido: ninguna de las tres
    entradas nuevas cita esos números — cada una habla de sí misma y, como
    mucho, del otro motor SELLADO con el que se solapan los intervalos de
    confianza (lightgbm cita a catboost «0,825», no a xgboost). La prosa que
    este test re-derivaba no existe ya, y su mitad estructural (quién no está)
    la sigue cubriendo el mismo test de conjuntos.

Las dos retiradas quedan documentadas aquí, no borradas en silencio: si algún
día una entrada vuelve a nombrar la fracción exacta de un rival en su texto,
lo natural es traer de vuelta la comprobación textual — pero apoyada en la
frase que exista entonces, no en esta.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from benchmarks.fase0.protocolo import (ProtocoloExploratorio, aplicar_regla_de_cierre,
                                        veredicto_con_su_alcance)
from benchmarks.fase0 import pasada_amplia_101_c5 as _c5
from matrixai.estudio.validacion import digest_canonico
from matrixai_engines.cartera import CARTERA_APROBADA

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"

#: Cómo se llaman en el texto de la cartera los motores que en el JSON tienen
#: otro id. Escrito aquí y no adivinado: «no adivinar, medir».
_ID_EN_EL_JSON = {"catboost": "catboost", "lightgbm": "lightgbm",
                  "sklearn.hgb": "sklearn.hgb", "xgboost": "xgboost",
                  "sklearn.lineal": "sklearn.lineal",
                  "densa": "matrixai.dense.torch_cpu", "baseline": "baseline"}

#: El que la regla de cierre EXCLUYE del cálculo de «el mejor»: no compite.
_QUE_NO_COMPITE = "baseline"

#: El nombre de biblioteca de `MotorSoportado.biblioteca` no siempre coincide
#: con la clave que la PROCEDENCIA del JSON usa para la misma biblioteca
#: (`sklearn` en la cartera, `scikit-learn` en `versiones_de_bibliotecas`).
#: Escrito aquí, medido contra el JSON, no adivinado.
_NOMBRE_EN_LA_PROCEDENCIA = {"lightgbm": "lightgbm", "catboost": "catboost",
                             "sklearn": "scikit-learn"}


def _buscar(patron: str, texto: str, que: str) -> re.Match:
    encontrado = re.search(patron, texto)
    if encontrado is None:
        raise AssertionError(
            f"la evidencia de la cartera ya no dice {que}: no casa {patron!r}.\n"
            "Si el texto se ha reescrito, este test tiene que reescribirse CON él "
            "-- lo que no puede pasar es que la afirmación quede sin re-derivar.\n"
            f"Texto actual:\n{texto}")
    return encontrado


def _como_numero(escrito: str) -> float:
    """`"0,833"` -> `0.833`. Se compara EL NÚMERO, no la cadena: «cuando
    Roberto dice "me sale distinto", comparar los números» — y al revés, una
    comparación de cadenas fallaría por un `0,80` frente a un `0.8` sin que
    nada estuviera mal."""
    return float(escrito.replace(",", "."))


def _como_entero_con_puntos(escrito: str) -> int:
    """`"3.479"` -> `3479`. La pasada AMPLIA escribe sus recuentos grandes con
    el punto de los miles, y `_como_numero` lo confundiría con un decimal."""
    return int(escrito.replace(".", ""))


def _decimales_de(escrito: str) -> int:
    return len(escrito.split(",")[1]) if "," in escrito else 0


class _ContraLaPasadaQueLaEntradaCITA(unittest.TestCase):
    """Carga LAS TRES entradas de cartera y **el JSON que ellas mismas
    nombran**.

    El nombre del fichero se lee del texto con una expresión regular en vez de
    escribirse aquí: así, el día que las entradas cambien de pasada, este
    fichero mide la NUEVA sin que nadie tenga que acordarse. Y se comprueba
    que las TRES citan el MISMO fichero — si un día una entrada citara una
    pasada distinta de las otras dos sin que nadie lo notara, sería
    exactamente la clase de divergencia silenciosa que este módulo existe
    para impedir.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not CARTERA_APROBADA:
            raise AssertionError(
                "la cartera esta vacia: no hay ninguna entrada que re-derivar. Si "
                "es una decision pendiente, este fichero no tiene nada que hacer "
                "hasta que Roberto vuelva a sellar")
        cls.protocolo = ProtocoloExploratorio.cargar(str(_FASE0 / "protocolo_exploratorio.json"))
        # El nombre puede venir con una ruta delante (`benchmarks/fase0/...`):
        # se captura solo el fichero, con el prefijo de directorio opcional y
        # sin capturar.
        _PATRON_DEL_FICHERO = r"`(?:[\w./]+/)?(pasada_\S+?\.json)`"
        cls.nombre_del_json = _buscar(
            _PATRON_DEL_FICHERO, CARTERA_APROBADA[0].evidencia,
            "de que fichero de evidencia habla").group(1)
        for entrada in CARTERA_APROBADA:
            citado = _buscar(_PATRON_DEL_FICHERO, entrada.evidencia,
                             f"de que fichero de evidencia habla {entrada.motor}").group(1)
            if citado != cls.nombre_del_json:
                raise AssertionError(
                    f"{entrada.motor!r} cita {citado!r} y otra entrada de la misma "
                    f"cartera cita {cls.nombre_del_json!r}: las tres tienen que "
                    "sostenerse en LA MISMA pasada, o dos sitios declarando lo "
                    "mismo acaban divergiendo")
        cls.payload = json.loads((_FASE0 / cls.nombre_del_json).read_text(encoding="utf-8"))
        cls.resultados = cls.payload["resultados"]
        cls.metrica_por_dataset = cls.payload["alcance_y_veredicto"]["metrica_de_cierre_por_dataset"]
        cls.datasets_declarados = [d.nombre for d in cls.protocolo.datasets]
        cls.motores_declarados = _c5.nombres_de_los_motores_de_la_pasada()

    def _regla_sobre(self, motor_en_el_texto: str, resultados=None) -> dict:
        """**METRICA POR DATASET SIEMPRE**: con las tres tareas mezcladas (una
        sola métrica para los 40 haría desaparecer del denominador a
        multiclase y regresión sin dejar rastro — documentado en
        `aplicar_regla_de_cierre`)."""
        return aplicar_regla_de_cierre(
            resultados if resultados is not None else self.resultados,
            self.protocolo.regla_de_cierre, motor=_ID_EN_EL_JSON[motor_en_el_texto],
            metrica_por_dataset=self.metrica_por_dataset)

    def _veredicto_sobre(self, motor_en_el_texto: str, resultados=None) -> dict:
        return veredicto_con_su_alcance(
            self.protocolo, resultados if resultados is not None else self.resultados,
            motor=_ID_EN_EL_JSON[motor_en_el_texto],
            motores_declarados=self.motores_declarados,
            datasets_declarados=self.datasets_declarados,
            metrica_por_dataset=self.metrica_por_dataset)

    def _metrica_por_pliegue(self, motor_id_en_json: str, dataset: str) -> dict:
        """El valor de cada (repetición, pliegue), CON LA MÉTRICA DE ESE
        DATASET — nunca `auroc` fijo. Emparejar por pliegue es lo que
        convierte «gana por 0,007 de media» en «pierde 9 de 15», y hacerlo con
        la métrica que NO es mide silenciosamente mal en cuanto el dataset es
        de multiclase o de regresión (ver el docstring del módulo)."""
        metrica = self.metrica_por_dataset.get(dataset, "auroc")
        return {(r["repeticion"], r["pliegue"]): r[metrica] for r in self.resultados
                if r["motor"] == motor_id_en_json and r["dataset"] == dataset
                and r.get("repeticion") is not None and r.get("pliegue") is not None
                and r.get(metrica) is not None}

    def _afirma_la_misma_fraccion(self, escrito: str, medido: float, de_quien: str) -> None:
        """La fracción escrita tiene que ser la medida **a la precisión con la
        que está escrita**. Comparar a la precisión escrita: «cuando Roberto
        dice "me sale distinto", comparar los números»."""
        self.assertAlmostEqual(
            _como_numero(escrito), medido, places=_decimales_de(escrito),
            msg=f"{de_quien}: la cartera escribe {escrito!r} y la regla re-aplicada da "
                f"{medido!r}")


class LaEntradaDeCarteraEsLaQueSeMIDIOTest(_ContraLaPasadaQueLaEntradaCITA):
    """Las TRES entradas, cada una contra la pasada real y la regla
    registrada."""

    # -- la mitad que promueve, para CADA UNA de las tres --------------
    def test_el_numero_que_afirma_la_cartera_SALE_de_aplicar_la_regla(self):
        """EL TEST DEL HALLAZGO, por TRIPLICADO. Vuelve a aplicar la regla
        pre-registrada a la pasada real y exige que salga exactamente lo que
        CADA entrada afirma.

        Con el sabotaje del auditor —hundir la métrica de cierre de un motor
        sellado en un par de datasets— la re-derivación de ESE motor baja y su
        subTest se pone rojo diciendo los dos números; los otros dos siguen
        verdes, que es justo la resolución que hace falta para saber a QUIÉN
        se le ha tocado el JSON."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                m = _buscar(
                    rf"{re.escape(entrada.motor)} (\d+)/(\d+) = (\d,\d+) >= (\d,\d+), CUMPLE",
                    entrada.evidencia, "el número, la fracción y el listón")
                cumplidos, total, fraccion, liston = (
                    int(m.group(1)), int(m.group(2)), m.group(3), m.group(4))
                r = self._regla_sobre(entrada.motor)
                self.assertEqual(
                    (r["cumplidos"], r["datasets"]), (cumplidos, total),
                    f"{entrada.motor}: la cartera afirma {cumplidos}/{total} y la regla "
                    f"aplicada a {self.nombre_del_json} da {r['cumplidos']}/{r['datasets']}")
                self._afirma_la_misma_fraccion(fraccion, r["fraccion"], entrada.motor)
                self.assertAlmostEqual(_como_numero(liston), r["fraccion_minima"], places=6)
                self.assertTrue(r["cumple_la_regla"],
                                f"{entrada.motor}: la cartera dice CUMPLE y la regla "
                                "re-aplicada dice que no")

    def test_la_regla_citada_es_la_REGISTRADA_CON_HASH_no_otra(self):
        """«a menos de 2 puntos del mejor en >=80 % de los datasets». Si
        alguien relaja el protocolo para que el número salga, el texto y el
        protocolo dejan de coincidir. La frase viene de `_EVIDENCIA_COMUN` y
        es literalmente la misma en las tres entradas — se comprueba una vez
        contra el protocolo y luego que las TRES la citen."""
        m = _buscar(r"a menos de (\d+) puntos del mejor en >=(\d+) % de los datasets",
                    CARTERA_APROBADA[0].evidencia, "la regla con sus dos parámetros")
        regla = self.protocolo.regla_de_cierre
        self.assertEqual(float(m.group(1)), regla.puntos)
        self.assertEqual(float(m.group(2)) / 100.0, regla.fraccion_minima)
        self.assertIn("fallo", regla.definicion_de_mejor.lower(),
                      "la entrada afirma «contando un fallo como dataset perdido»: "
                      "eso tiene que seguir estando en la definición registrada")
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertRegex(
                    entrada.evidencia,
                    r"a menos de \d+ puntos del mejor en >=\d+ % de los datasets")

    def test_los_intentos_y_los_QUINCE_fallos_estan_en_el_JSON(self):
        """El recuento sale del texto, no de un número escrito aquí a mano —
        y la afirmación de HOY es distinta de la de 101-C3: aquella pasada
        tenía CERO fallos, ésta tiene 15, y la evidencia dice de quién son.
        Sin la segunda mitad, un sabotaje que trasladara un fallo a uno de los
        TRES motores aprobados pasaría desapercibido."""
        m = _buscar(r"([\d.]+) intentos, ([\d.]+) completados y (\d+) fallidos",
                    CARTERA_APROBADA[0].evidencia, "los intentos, los completados y los fallidos")
        total = _como_entero_con_puntos(m.group(1))
        completados = _como_entero_con_puntos(m.group(2))
        fallidos = int(m.group(3))
        self.assertEqual(len(self.resultados), total)
        no_completados = [r for r in self.resultados if r.get("estado") != "completed"]
        self.assertEqual(sum(1 for r in self.resultados if r.get("estado") == "completed"),
                         completados)
        self.assertEqual(len(no_completados), fallidos)

        aprobados_ids = {_ID_EN_EL_JSON[e.motor] for e in CARTERA_APROBADA}
        for r in no_completados:
            self.assertEqual(r["motor"], "matrixai.dense.torch_cpu",
                             "la evidencia dice que los 15 fallos son todos del motor denso "
                             "propio: un fallo en otro motor lo desmentiria")
            self.assertNotIn(r["motor"], aprobados_ids,
                             "un fallo en uno de los TRES motores aprobados, y la evidencia "
                             "sigue diciendo «de ninguno de los aprobados»")

        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertIn(m.group(0), entrada.evidencia)

    def test_la_version_de_la_BIBLIOTECA_que_la_cartera_aprueba_es_la_MEDIDA(self):
        """La cartera aprueba el par motor + biblioteca (ver `MotorSoportado`).

        **LA FRASE «medida con X Y.Y.Y» YA NO ESTÁ en ninguna de las tres
        evidencias** (medido con `grep` antes de escribir este test, no
        supuesto): la prosa nueva no repite la versión de cada biblioteca.
        Lo que SÍ se puede re-derivar es contra la PROCEDENCIA ANCLADA que el
        propio JSON registra — es de ahí de donde salió esa versión, y es
        justamente lo que hace a esta pasada «la primera con procedencia
        anclable», como dice la propia evidencia."""
        medidas = self.payload["procedencia"]["versiones_de_bibliotecas"]
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertIsNotNone(entrada.biblioteca)
                self.assertIsNotNone(entrada.versiones_de_la_biblioteca)
                clave = _NOMBRE_EN_LA_PROCEDENCIA[entrada.biblioteca]
                medida = medidas.get(clave)
                self.assertIsNotNone(medida, f"la procedencia no registra version de {clave!r}")
                self.assertEqual(
                    entrada.versiones_de_la_biblioteca.minima, medida,
                    f"{entrada.motor}: la cartera aprueba desde "
                    f"{entrada.versiones_de_la_biblioteca.minima!r} y la procedencia que "
                    f"sostiene la evidencia midio {medida!r}")

    # -- la otra mitad: quién no está, y por qué, SIN nombrar sus números --
    def test_esta_en_la_cartera_EXACTAMENTE_quien_cumple_la_regla(self):
        """El invariante entero, sin citar ningún número: quien la regla
        aprueba está, y quien no, no. Es lo que impide una promoción a mano —
        y con tres motores dentro, es la ÚNICA forma que queda de comprobar
        «ni uno de más, ni uno de menos» (ver el docstring del módulo: esto
        sustituye a dos tests retirados de la versión anterior)."""
        aprobados_por_la_regla = {
            nombre for nombre, id_json in _ID_EN_EL_JSON.items()
            if id_json != _ID_EN_EL_JSON[_QUE_NO_COMPITE]
            and self._regla_sobre(nombre)["cumple_la_regla"]}
        en_la_cartera = {e.motor for e in CARTERA_APROBADA}
        self.assertEqual({_ID_EN_EL_JSON[n] for n in aprobados_por_la_regla}, en_la_cartera)

    # -- el JSON no se toca ---------------------------------------------
    def test_el_JSON_de_la_evidencia_sigue_siendo_EL_MISMO_que_se_midio(self):
        """El digest que el propio fichero se calculó al escribirse. Coge
        cualquier edición, también las que no mueven la regla — un `wall_s`,
        un `procedencia_id`, una fila añadida."""
        payload = dict(self.payload)
        guardado = payload.pop("digest_resultados_crudos")
        self.assertEqual(digest_canonico(payload), guardado,
                         f"{self.nombre_del_json} ha cambiado desde que se midió: "
                         "su propio digest ya no cuadra")

    def test_el_digest_que_las_TRES_ENTRADAS_SELLAN_es_el_de_ESE_JSON(self):
        """`evidencia` NOMBRA el artefacto; `evidencia_digest` lo IDENTIFICA —
        y las tres entradas sellan el MISMO digest, porque las tres se
        sostienen en la misma pasada."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertEqual(entrada.evidencia_digest,
                                 self.payload["digest_resultados_crudos"],
                                 f"{entrada.motor} sella un digest que no es el de "
                                 f"{self.nombre_del_json}")
                abreviado = _buscar(
                    r"digest de sus resultados crudos `([0-9a-f]+)\.\.\.`",
                    entrada.evidencia, f"el digest abreviado de la pasada en {entrada.motor}"
                    ).group(1)
                self.assertTrue(entrada.evidencia_digest.startswith(abreviado),
                                f"{entrada.motor}: el texto abrevia {abreviado!r} y el digest "
                                f"sellado es {entrada.evidencia_digest!r}")

    def test_el_ALCANCE_declara_el_recorte_que_la_pasada_tuvo_de_verdad(self):
        """Los tres recortes, re-derivados del JSON en vez de creídos: cuántos
        motores corrieron, cuántos datasets y cuántas configuraciones. **Y
        AHORA SÍ se sigue algo de multiclase y de regresión** — a diferencia
        de la pasada de 12 (solo binaria), la AMPLIA las midió enteras, así
        que `tareas_sin_medir` y `cubos_sin_medir` tienen que venir VACÍOS. Un
        alcance que siguiera diciendo «falta multiclase» sobre una pasada que
        sí la midió sería la media verdad en la otra dirección: alarmar con un
        recorte que ya no existe."""
        motores_en_los_registros = {r["motor"] for r in self.resultados}
        datasets = {r["dataset"] for r in self.resultados}
        configuraciones_que_corrieron = {(r["motor"], r["configuracion"])
                                         for r in self.resultados}
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                alcance = entrada.alcance
                self.assertEqual(len(alcance.motores_que_corrieron), len(motores_en_los_registros),
                                 "el alcance dice que corrieron unos motores y el JSON trae otros")
                self.assertEqual(len(alcance.motores_que_no_compitieron), 0,
                                 "esta pasada corrio LOS SIETE: no falta ninguno")

                self.assertEqual(alcance.datasets_que_corrieron, len(datasets))
                self.assertEqual(alcance.datasets_del_protocolo, len(self.protocolo.datasets))

                self.assertEqual(alcance.configuraciones_que_corrieron,
                                 len(configuraciones_que_corrieron))
                self.assertEqual(alcance.configuraciones_del_protocolo,
                                 sum(m.configuraciones for m in self.protocolo.motores))
                self.assertLess(alcance.configuraciones_que_corrieron,
                                alcance.configuraciones_del_protocolo,
                                "si algun dia corren TODAS, este aserto avisa de que el texto "
                                "se quedo viejo")

                self.assertEqual(alcance.tareas_sin_medir, (),
                                 "la pasada AMPLIA midio multiclase y regresion enteras: un "
                                 "alcance que siga diciendolas «sin medir» miente por omision "
                                 "en la direccion alarmista")
                self.assertEqual(alcance.cubos_sin_medir, (),
                                 "los tres cubos, incluido el grande, corrieron en esta pasada")

    def test_el_MARGEN_y_los_PRIMEROS_PUESTOS_del_alcance_son_los_medidos(self):
        """Los dos números que hacen legible el veredicto, y los dos se pueden
        inventar a mano en la entrada sin que nada más se entere: cuántos
        datasets puede perder antes de bajarse del listón, y cuántos de sus
        aciertos son «es el mejor de los que corrieron» (distancia 0,0000)."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                r = self._regla_sobre(entrada.motor)
                alcance = entrada.alcance
                self.assertEqual(alcance.datasets_cumplidos, r["cumplidos"])
                self.assertEqual(alcance.datasets_evaluados, r["datasets"])
                self.assertEqual(alcance.margen_en_datasets,
                                 r["datasets_que_puede_perder_sin_incumplir"])
                self.assertEqual(alcance.aciertos_por_ser_el_mejor,
                                 r["aciertos_por_ser_el_mejor"])
                self.assertEqual(sum(1 for d in r["detalle"] if d["distancia_en_puntos"] == 0.0),
                                 alcance.aciertos_por_ser_el_mejor)


class LaEvidenciaNoDiceDeMASDeLoQueSeMIDIOTest(_ContraLaPasadaQueLaEntradaCITA):
    """HALLAZGO de la auditoría del 2026-09-12, y sigue siendo el mismo aunque
    la afirmación haya cambiado dos veces desde entonces: la entrada de
    entonces decía «es el unico de los cuatro que reprodujo 177/177 intentos
    entre las dos pasadas», y medido era falso en las dos mitades. Media
    verdad tranquilizadora es peor que callarse, sobre todo dentro de la
    evidencia que sostiene una promoción.

    **CADA UNA DE LAS TRES ENTRADAS DE HOY se obliga a declarar lo que la
    debilita**, y las tres lo hacen con SU PROPIA prosa, no con una plantilla
    compartida: `lightgbm` y `sklearn.hgb` narran dataset por dataset,
    `catboost` —que tiene CUATRO primeros puestos discutidos y TRES cumplidos
    dentro del ruido— los agrupa en una sola frase cada bloque, y
    `sklearn.hgb` —que no tiene NINGÚN primer puesto discutido— lo declara con
    esas mismas letras en vez de callarlo. Por eso las pruebas de este bloque
    van una por motor cuando la prosa lo exige, en vez de un único patrón
    compartido: forzar un patrón común habría sido medir la forma, no el
    contenido.
    """

    _MARCA_DE_QUE_NO_ES_UN_DOMINIO_LIMPIO = {
        "lightgbm": "NO SON SIETE DOMINIOS",
        "sklearn.hgb": "NINGUNO lo discute la dispersion",
        "catboost": "NO SON TRECE DOMINIOS",
    }

    def _detalle_de(self, motor_en_el_texto: str) -> list[dict]:
        return self._regla_sobre(motor_en_el_texto)["detalle"]

    def test_los_primeros_puestos_QUE_LA_DISPERSION_DISCUTE_son_los_medidos(self):
        """De los datasets que cada motor gana a distancia 0,0000, los
        discutidos son aquellos en los que **pierde la mayoría de los
        pliegues emparejados** contra el segundo — re-derivado con la métrica
        de CADA dataset, no con `auroc` fijo (ver el docstring del módulo:
        `okcupid-stem`, uno de los cuatro de `catboost`, se cierra con
        `accuracy`).

        Las dos direcciones importan y por eso es una igualdad de conjuntos:
        si saliera uno de más, la entrada estaría escondiendo; si de menos,
        estaría alarmando con algo que la pasada no dice."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                mi_id = _ID_EN_EL_JSON[entrada.motor]
                discutidos_medidos = set()
                for d in self._detalle_de(entrada.motor):
                    if d["distancia_en_puntos"] != 0.0 or d["segundo"] is None:
                        continue
                    mio = self._metrica_por_pliegue(mi_id, d["dataset"])
                    suyo = self._metrica_por_pliegue(d["segundo"], d["dataset"])
                    comunes = set(mio) & set(suyo)
                    perdidos = sum(1 for k in comunes if mio[k] < suyo[k])
                    if comunes and perdidos * 2 > len(comunes):
                        discutidos_medidos.add(d["dataset"])

                declarados = entrada.alcance.aciertos_que_la_dispersion_discute
                self.assertIsNotNone(
                    declarados, f"{entrada.motor}: `aciertos_que_la_dispersion_discute` a "
                    "`None` dice «no se midió», y está medido")
                self.assertEqual(set(declarados), discutidos_medidos, entrada.motor)

    def test_la_prosa_declara_QUE_NO_es_un_dominio_limpio(self):
        """LA MITAD QUE COSTÓ EL HALLAZGO LA VEZ ANTERIOR: que el campo esté
        bien no basta si la prosa que lee una persona no lo matiza. Cada
        entrada lo dice a su manera — comprobado con la marca literal de SU
        propio texto, no con una frase compartida que ninguna de las tres
        repite ya."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                marca = self._MARCA_DE_QUE_NO_ES_UN_DOMINIO_LIMPIO[entrada.motor]
                self.assertIn(marca, entrada.evidencia,
                             f"{entrada.motor}: la entrada ya no matiza sus primeros "
                             "puestos -- si el texto ha cambiado de forma, este test se "
                             "reescribe CON el, no se borra")

    def test_y_el_TEXTO_da_los_numeros_CON_LA_FORMA_de_cada_entrada(self):
        """La mitad que costó el hallazgo la vez anterior, aplicada a las TRES
        formas de prosa que hay hoy: un aserto negativo lo pasaría un texto
        vacío; esto exige lo que SÍ tiene que estar, con el número que le
        corresponde a cada motor."""
        detalle_de = {e.motor: {d["dataset"]: d for d in self._detalle_de(e.motor)}
                     for e in CARTERA_APROBADA}

        # -- lightgbm: una frase por dataset, con pliegues y rival nombrado --
        entrada = next(e for e in CARTERA_APROBADA if e.motor == "lightgbm")
        m = _buscar(r"([\w.\-]+), lo gana por (\d,\d+) puntos perdiendo (\d+) de los (\d+) "
                    r"pliegues emparejados contra ([\w.]+)",
                    entrada.evidencia, "cuanto gana lightgbm y cuantos pliegues pierde")
        dataset, ventaja, perdidos, emparejados, rival = (
            m.group(1), _como_numero(m.group(2)), int(m.group(3)), int(m.group(4)),
            m.group(5).rstrip("."))
        self.assertIn(dataset, entrada.alcance.aciertos_que_la_dispersion_discute)
        d = detalle_de["lightgbm"][dataset]
        self.assertEqual(d["segundo"], rival)
        self.assertAlmostEqual(ventaja,
                               round(d["ventaja_sobre_el_segundo_en_puntos"],
                                     _decimales_de(m.group(2))), places=6)
        mio = self._metrica_por_pliegue(_ID_EN_EL_JSON["lightgbm"], dataset)
        suyo = self._metrica_por_pliegue(rival, dataset)
        comunes = set(mio) & set(suyo)
        self.assertEqual(len(comunes), emparejados)
        self.assertEqual(sum(1 for k in comunes if mio[k] < suyo[k]), perdidos)

        # -- sklearn.hgb: CERO que contar, declarado con todas las letras --
        entrada = next(e for e in CARTERA_APROBADA if e.motor == "sklearn.hgb")
        self.assertEqual(entrada.alcance.aciertos_que_la_dispersion_discute, (),
                         "medido y vacio, no None: sus diez primeros puestos se "
                         "comprobaron pliegue a pliegue y ninguno se discute")
        self.assertIn("NINGUNO lo discute la dispersion", entrada.evidencia)

        # -- catboost: CUATRO en la misma frase, solo la ventaja de cada uno.
        # "gana por" solo se escribe delante del primero de la lista -- "en
        # wilt (gana por 0,007), sick (0,014), okcupid-stem (0,022) y pc3
        # (0,309)" -- así que el prefijo es opcional en los tres siguientes.
        entrada = next(e for e in CARTERA_APROBADA if e.motor == "catboost")
        vistos = re.findall(r"([\w.\-]+) \((?:gana por )?(\d,\d+)\)", entrada.evidencia)
        self.assertTrue(vistos, "no se encontro ningun 'nombre (gana por X)' en catboost")
        self.assertEqual({v[0] for v in vistos},
                         set(entrada.alcance.aciertos_que_la_dispersion_discute))
        for dataset, ventaja in vistos:
            with self.subTest(dataset=dataset):
                d = detalle_de["catboost"][dataset]
                self.assertAlmostEqual(
                    _como_numero(ventaja),
                    round(d["ventaja_sobre_el_segundo_en_puntos"], _decimales_de(ventaja)),
                    places=6)

    def test_los_cumplidos_DENTRO_DEL_RUIDO_estan_declarados_y_son_los_medidos(self):
        """La otra mitad de «gana por nada»: no ganar por poco, sino CUMPLIR
        por poco. El resumen del propio JSON no lo enseña — su contador solo
        mira a los que NO cumplen —, así que esto se re-deriva de los
        intervalos de cada dataset cumplido."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                dentro_del_ruido = {
                    d["dataset"] for d in self._detalle_de(entrada.motor)
                    if d["cumple"] and (d.get("intervalo_de_la_distancia") or {}).get("cruza_el_liston")}
                declarados = entrada.alcance.cumplidos_cuyo_intervalo_cruza_el_liston
                self.assertIsNotNone(declarados, f"{entrada.motor}: `None` dice «no medido», "
                                     "y los intervalos estan en la pasada")
                self.assertEqual(set(declarados), dentro_del_ruido, entrada.motor)
                self.assertTrue(dentro_del_ruido, f"{entrada.motor}: sin ninguno dentro del "
                                "ruido este test estaria verde por vacio")

    def test_y_el_TEXTO_da_LA_DISTANCIA_Y_EL_INTERVALO_de_cada_uno(self):
        """Los tres números por dataset — distancia, bajo y alto del intervalo
        al 95 % — re-derivados con la forma de prosa de cada entrada."""
        detalle_de = {e.motor: {d["dataset"]: d for d in self._detalle_de(e.motor)}
                     for e in CARTERA_APROBADA}

        patrones_de_uno_solo = {
            "lightgbm": r"([\w.\-]+), cumple DENTRO DEL RUIDO \((\d,\d+) puntos contra un "
                       r"liston de \d,\d+, intervalo \[(\d,\d+) \.\. (\d,\d+)\], que lo cruza\)",
            "sklearn.hgb": r"([\w.\-]+), cumple dentro del ruido \((\d,\d+) puntos, intervalo "
                          r"\[(\d,\d+) \.\. (\d,\d+)\]\)",
        }
        for motor, patron in patrones_de_uno_solo.items():
            with self.subTest(motor=motor):
                entrada = next(e for e in CARTERA_APROBADA if e.motor == motor)
                m = _buscar(patron, entrada.evidencia, f"el dataset dentro del ruido de {motor}")
                dataset, distancia, bajo, alto = m.group(1), m.group(2), m.group(3), m.group(4)
                self.assertIn(dataset, entrada.alcance.cumplidos_cuyo_intervalo_cruza_el_liston)
                d = detalle_de[motor][dataset]
                intervalo = d["intervalo_de_la_distancia"]
                self.assertAlmostEqual(_como_numero(distancia),
                                       round(d["distancia_en_puntos"], _decimales_de(distancia)),
                                       places=6)
                self.assertAlmostEqual(_como_numero(bajo),
                                       round(intervalo["bajo"], _decimales_de(bajo)), places=6)
                self.assertAlmostEqual(_como_numero(alto),
                                       round(intervalo["alto"], _decimales_de(alto)), places=6)
                self.assertLess(_como_numero(bajo), self.protocolo.regla_de_cierre.puntos)
                self.assertGreater(_como_numero(alto), self.protocolo.regla_de_cierre.puntos)

        # -- catboost: TRES, en una sola frase --
        entrada = next(e for e in CARTERA_APROBADA if e.motor == "catboost")
        vistos = re.findall(
            r"([\w.\-]+) \((\d,\d+)(?: puntos)?, (?:intervalo )?\[(\d,\d+) \.\. (\d,\d+)\]\)",
            entrada.evidencia)
        self.assertTrue(vistos, "no se encontraron los tres cumplidos dentro del ruido de catboost")
        self.assertEqual({v[0] for v in vistos},
                         set(entrada.alcance.cumplidos_cuyo_intervalo_cruza_el_liston))
        for dataset, distancia, bajo, alto in vistos:
            with self.subTest(dataset=dataset):
                d = detalle_de["catboost"][dataset]
                intervalo = d["intervalo_de_la_distancia"]
                self.assertAlmostEqual(_como_numero(distancia),
                                       round(d["distancia_en_puntos"], _decimales_de(distancia)),
                                       places=6)
                self.assertAlmostEqual(_como_numero(bajo),
                                       round(intervalo["bajo"], _decimales_de(bajo)), places=6)
                self.assertAlmostEqual(_como_numero(alto),
                                       round(intervalo["alto"], _decimales_de(alto)), places=6)

    def test_quien_CAMBIA_DE_LADO_DEL_LISTON_segun_la_semilla_esta_medido(self):
        """Re-aplicar la regla registrada a cada semilla por separado (una
        sola, no combinaciones): los que cumplen con una y no con otra son los
        declarados. Con los 40 datasets solo `xgboost` cambia — medido, y es
        distinto de lo que declaraba la entrada de 09-14, donde eran TRES los
        que cambiaban. El motor sellado no puede ser uno de ellos: la cartera
        se niega a construir esa entrada."""
        semillas = sorted({r["repeticion"] for r in self.resultados
                           if r.get("repeticion") is not None})
        self.assertGreater(len(semillas), 1, "sin varias semillas no hay nada que comparar")

        cambian = set()
        for nombre in _ID_EN_EL_JSON:
            if nombre == _QUE_NO_COMPITE:
                continue
            lados = {self._regla_sobre(nombre, [r for r in self.resultados
                                                if r["repeticion"] == s])["cumple_la_regla"]
                     for s in semillas}
            if len(lados) > 1:
                cambian.add(_ID_EN_EL_JSON[nombre])

        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                declarados = entrada.alcance.motores_que_cambian_de_lado_del_liston_segun_la_semilla
                self.assertIsNotNone(declarados, f"{entrada.motor}: `None` dice «no medido»")
                self.assertEqual(set(declarados), cambian)
                self.assertNotIn(_ID_EN_EL_JSON[entrada.motor], cambian,
                                 f"{entrada.motor} cambia de lado del liston segun la semilla: "
                                 "la cartera no deberia haber podido construir esta entrada")

    def test_cumplidos_con_menos_semillas_es_la_lectura_con_SOLO_la_semilla_0(self):
        """LA TRAMPA DE ESTA PASADA, medida y no supuesta: los 10 datasets del
        cubo grande corrieron con UNA sola repetición (la 0), así que filtrar
        por la semilla 1 o la 2 solas BORRA el cubo grande y el denominador
        baja de 40 a 30. Solo la semilla 0 sola sigue cubriendo los 40 con
        MENOS repeticiones que las tres juntas, y es exactamente el número que
        las tres entradas publican en `cumplidos_con_menos_semillas` (35 para
        las tres, medido)."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                sub = [r for r in self.resultados if r.get("repeticion") == 0]
                r = self._regla_sobre(entrada.motor, sub)
                self.assertEqual(r["datasets"], 40,
                                 "la semilla 0 sola tiene que seguir cubriendo los 40: es "
                                 "la unica repeticion que tiene el cubo grande")
                self.assertEqual(entrada.alcance.cumplidos_con_menos_semillas, r["cumplidos"],
                                 f"{entrada.motor}: con la semilla 0 sola la regla da "
                                 f"{r['cumplidos']} y la entrada declara "
                                 f"{entrada.alcance.cumplidos_con_menos_semillas}")
                self.assertGreaterEqual(
                    r["cumplidos"], entrada.alcance.datasets_cumplidos,
                    f"{entrada.motor}: el campo solo se rellena cuando la lectura con "
                    "menos semillas es MAS generosa que la publicada -- si fuera al "
                    "reves el guardia de `AlcanceDeLaMedicion` lo habria rechazado")

    def test_cumplidos_por_semilla_se_queda_en_None_porque_el_cubo_grande_NO_TIENE_tres(self):
        """REESCRITO ENTERO — 2026-09-16. Sustituye a
        `test_las_lecturas_por_semilla_VIAJAN_en_el_json` y
        `test_la_entrada_REAL_lo_deja_en_None_y_publica_las_TRES_lecturas` de
        `test_c102_c1_alcance_de_la_cartera.py` (ese fichero, del otro repo,
        cubría UNA sola entrada del 09-14 con `(12, 12, 11)` publicado) y al
        `test_el_recuento_DE_CADA_SEMILLA_es_el_medido` que tenía este
        fichero, que comparaba `cumplidos_por_semilla` contra una tupla
        «(X, Y y Z de W)» que la prosa de HOY ya no escribe — medido: `grep
        semilla` no encuentra nada en las tres evidencias.

        **LA RAZÓN NO ES QUE NADIE LO MIDIERA**: es que con el cubo grande a
        una sola repetición, leer «solo la semilla 1» o «solo la 2» borra el
        cubo grande entero y el denominador baja de 40 a 30. Un `25/30` y un
        `34/40` no se pueden meter en la misma tupla sin mentir por el
        denominador, así que el campo se queda en `None` en las TRES entradas
        — medido aquí, no supuesto: filtrar por la semilla 1 o la 2 sola dejan
        caer datasets del cubo grande."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertIsNone(
                    entrada.alcance.cumplidos_por_semilla,
                    f"{entrada.motor}: el cubo grande solo corrio con una repeticion, asi "
                    "que las lecturas por semilla no comparten denominador y no se pueden "
                    "publicar como una tupla de tres sin mentir")

        for semilla in (1, 2):
            with self.subTest(semilla=semilla):
                sub = [r for r in self.resultados if r.get("repeticion") == semilla]
                datasets_en_esa_lectura = {r["dataset"] for r in sub}
                self.assertLess(
                    len(datasets_en_esa_lectura), len(self.datasets_declarados),
                    f"la semilla {semilla} sola SI cubre los 40: entonces "
                    "`cumplidos_por_semilla` deberia poder rellenarse, y este test esta "
                    "midiendo un hecho que ya no es cierto de la pasada")

    def test_la_entrada_declara_lo_que_la_DEBILITA_y_no_solo_lo_que_la_sostiene(self):
        """La mitad que caza el esconder: los datasets que aparecen en
        `aciertos_que_la_dispersion_discute` y en
        `cumplidos_cuyo_intervalo_cruza_el_liston` tienen que estar NOMBRADOS
        en la prosa, no solo en el campo estructurado — si no, quien lea el
        texto no se entera de cuál es."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertTrue(entrada.alcance.cumplidos_cuyo_intervalo_cruza_el_liston,
                                f"{entrada.motor}: los tres motores tienen al menos un "
                                "cumplido dentro del ruido -- si saliera vacio, este test "
                                "esta midiendo la pasada equivocada")
                for nombre in sorted(entrada.alcance.cumplidos_cuyo_intervalo_cruza_el_liston):
                    self.assertIn(nombre, entrada.evidencia,
                                 f"{entrada.motor}: {nombre!r} esta en el alcance y no en "
                                 "la prosa")
                for nombre in sorted(entrada.alcance.aciertos_que_la_dispersion_discute or ()):
                    self.assertIn(nombre, entrada.evidencia,
                                 f"{entrada.motor}: {nombre!r} esta en el alcance y no en "
                                 "la prosa")


if __name__ == "__main__":
    unittest.main()


class LaCarteraMIRA_LA_PROCEDENCIA_DeSuEvidenciaTest(_ContraLaPasadaQueLaEntradaCITA):
    """UN SELLO SOBRE EVIDENCIA IRREPRODUCIBLE — cerrado el 2026-09-16.

    Hasta hoy la entrada citaba un digest y **nadie comprobaba que ese artefacto
    fuera anclable**. Y no es teórico: de las cinco mediciones de Fase 0, CUATRO
    dicen `anclable: false`, y una de ellas por un motivo serio —se midió con el
    propio guion de la pasada modificado y sin commitear, así que el commit que
    declara **no identifica el código que produjo esos números**—. Un sello
    sobre eso es exactamente lo que el sello existe para impedir.

    **Por qué la comprobación vive AQUÍ y no en `__post_init__`.**
    `matrixai_engines` no alcanza `benchmarks/fase0/`: allí un campo obligatorio
    solo garantizaría que alguien escribió un `True`, y un `True` escrito a mano
    sobre una pasada sucia se lee igual que uno medido — sería peor que no tener
    campo. Este fichero SÍ abre el JSON, así que aquí el campo es obligatorio y
    además se contrasta. **Obligatorio donde se puede medir, declarado donde se
    lee.**
    """

    def test_toda_entrada_DECLARA_si_su_evidencia_es_anclable(self):
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertIsNotNone(
                    entrada.evidencia_anclable,
                    f"la entrada de {entrada.motor!r} no dice si su evidencia se "
                    "puede volver a atar a un commit. `None` aquí no es «da "
                    "igual»: es que nadie lo miró.")

    def test_lo_que_DECLARA_es_lo_que_el_artefacto_dice_de_si_mismo(self):
        """La mitad que impide que el campo sea una firma en blanco."""
        medido = self.payload["procedencia"]["anclable"]
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertEqual(
                    entrada.evidencia_anclable, medido,
                    f"{entrada.motor!r} declara `evidencia_anclable="
                    f"{entrada.evidencia_anclable}` y el artefacto que cita dice "
                    f"{medido}. Un campo que no coincide con lo medido no es una "
                    "declaración: es una afirmación sin respaldo dentro del sello.")

    def test_una_entrada_APROBADA_se_sostiene_en_evidencia_ANCLABLE(self):
        """El invariante entero, y el que de verdad protege a quien lo lea."""
        for entrada in CARTERA_APROBADA:
            with self.subTest(motor=entrada.motor):
                self.assertTrue(
                    entrada.evidencia_anclable,
                    f"{entrada.motor!r} lleva el sello de «soportado» sobre una "
                    "medición que NO se puede reproducir. Si de verdad hay que "
                    "aprobarlo, lo que se mueve es la medición —repetirla con el "
                    "árbol limpio—, no el sello.")

    def test_y_el_artefacto_NO_trae_avisos_de_procedencia(self):
        """`anclable` es la conclusión; `avisos` es el detalle que la sostiene.
        Si algún día salieran incoherentes, el que manda es el detalle."""
        avisos = self.payload["procedencia"]["avisos"]
        self.assertEqual(
            avisos, [],
            f"el artefacto que la cartera sella trae avisos de procedencia: "
            f"{avisos}")
        self.assertEqual(self.payload["procedencia"]["anclable"], not avisos)
