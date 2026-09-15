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

**EL 2026-09-14 LA CARTERA CAMBIÓ ENTERA, y este fichero con ella.** No porque
nadie tocara la regla: los 720 intentos que sostenían a lightgbm eran de CUATRO
motores de los SIETE que el protocolo registra, y los tres ausentes
—`sklearn.hgb`, `xgboost`, `catboost`— son los rivales DIRECTOS de un GBM.
Corrida la pasada CONFORME (los siete, 1.260 intentos, cero fallos), la misma
regla da `catboost` 12/12 y `lightgbm` 9/12: uno entra y el otro sale, en la
misma decisión de Roberto. Lo que cambió no fue el motor, fue el alcance.

Este fichero se reescribió CONTRA LA EVIDENCIA NUEVA, y una trampa concreta
merece quedar escrita porque estaba puesta y medida: la clase de abajo tenía un
ayudante, `_reproducidos_entre_las_dos_pasadas()`, que emparejaba las pasadas
del 07-09 y del 13-09 — ninguna de las dos es ya la que la entrada cita.
Dejarlo apuntando ahí habría puesto el fichero VERDE **midiendo otra cosa**,
que es peor que dejarlo rojo. Se retiró junto con la afirmación que re-derivaba
(la entrada de hoy no dice nada de reproducibilidad entre pasadas), y en su
lugar se re-derivan las afirmaciones que la entrada SÍ hace: ver la segunda
clase.

POR QUÉ ESTE TEST VIVE EN `matrixAI` Y NO EN `matrixai-engines`, que es donde
está la afirmación:

1. Aquí viven las DOS cosas que hay que volver a juntar: el JSON de la pasada
   (`benchmarks/fase0/pasada_exploratoria_101_c3_conforme_20260914.json`) y la
   REGLA que lo juzga (`benchmarks/fase0/protocolo.py`, registrada con hash
   por 101-C1 antes de medir). Un test que re-deriva un número se pone donde
   viven el número y su fórmula, no donde vive la frase que los cita.

2. `benchmarks/` NO se publica en PyPI (invariante 1 del 102). Un test dentro
   de `matrixai-engines` —que es un paquete que se instala solo— solo podría
   alcanzar ese JSON adivinando una ruta a un repositorio hermano, y tendría
   que saltarse cuando no la encuentra. Un salto silencioso es exactamente el
   banco de pruebas sin dientes que este hallazgo denuncia: volvería a estar
   verde sin haber medido nada.

3. La dirección contraria sí funciona y está medida: `matrixai_engines` es
   importable desde la suite de `matrixAI`, así que el test puede leer LA
   AFIRMACIÓN en su fuente (`cartera.CARTERA_APROBADA[0].evidencia`) en vez de
   copiarla.

QUÉ LO PONE ROJO, que es lo que define si tiene dientes:
  * editar el JSON de la pasada (los `auroc`, un `estado`, un dataset);
  * editar la regla de cierre registrada, o su aplicación;
  * editar el texto de la evidencia de la cartera para que diga otro número;
  * promover o despromover un motor sin que la medición lo acompañe;
  * rellenar un campo del `alcance` con algo que la pasada no dice — o
    vaciarlo cuando la pasada sí lo dice.

Las cifras NO están escritas en este fichero a propósito: se leen del texto de
la cartera con expresiones regulares y se comparan contra lo re-derivado. Un
número a mano aquí sería un tercer sitio declarando lo mismo, y la próxima vez
divergiría este.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from benchmarks.fase0.protocolo import ProtocoloExploratorio, aplicar_regla_de_cierre
from matrixai.estudio.validacion import digest_canonico
from matrixai_engines.cartera import CARTERA_APROBADA

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"

#: QUIÉN LLEVA EL SELLO HOY. Escrito una vez y usado en las expresiones
#: regulares, para que el día que la cartera vuelva a moverse este fichero
#: falle por UN sitio y no por seis.
_MOTOR_DE_LA_CARTERA = "catboost"

#: Cómo se llaman en el texto de la cartera los motores que en el JSON tienen
#: otro id. Escrito aquí y no adivinado: «no adivinar, medir».
_ID_EN_EL_JSON = {"catboost": "catboost", "lightgbm": "lightgbm",
                  "sklearn.hgb": "sklearn.hgb", "xgboost": "xgboost",
                  "sklearn.lineal": "sklearn.lineal",
                  "densa": "matrixai.dense.torch_cpu", "baseline": "baseline"}

#: El que la regla de cierre EXCLUYE del cálculo de «el mejor»: no compite.
_QUE_NO_COMPITE = "baseline"


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


def _decimales_de(escrito: str) -> int:
    return len(escrito.split(",")[1]) if "," in escrito else 0


class _ContraLaPasadaQueLaEntradaCITA(unittest.TestCase):
    """Carga la entrada de cartera y **el JSON que ella misma nombra**.

    El nombre del fichero se lee del texto con una expresión regular en vez de
    escribirse aquí: así, el día que la entrada cambie de pasada, este fichero
    mide la NUEVA sin que nadie tenga que acordarse — que es exactamente lo que
    no pasó el 2026-09-14 con `_reproducidos_entre_las_dos_pasadas()`, que se
    quedó apuntando a dos pasadas que ya no eran las de la entrada.
    """

    @classmethod
    def setUpClass(cls) -> None:
        entradas = [e for e in CARTERA_APROBADA if e.motor == _MOTOR_DE_LA_CARTERA]
        if not entradas:
            raise AssertionError(
                f"la cartera ya no sella a {_MOTOR_DE_LA_CARTERA!r} (tiene "
                f"{[e.motor for e in CARTERA_APROBADA]}). Si la promoción se ha "
                "movido, este fichero se reescribe CON ella: lo que no puede pasar "
                "es que se quede verde midiendo a quien ya no lleva el sello")
        cls.entrada = entradas[0]
        cls.alcance = cls.entrada.alcance
        cls.protocolo = ProtocoloExploratorio.cargar(str(_FASE0 / "protocolo_exploratorio.json"))
        cls.nombre_del_json = _buscar(r"`(pasada_exploratoria_\S+?\.json)`", cls.entrada.evidencia,
                                      "de qué fichero de evidencia habla").group(1)
        cls.payload = json.loads((_FASE0 / cls.nombre_del_json).read_text(encoding="utf-8"))
        cls.resultados = cls.payload["resultados"]

    def _regla_sobre(self, motor_en_el_texto: str, resultados=None) -> dict:
        return aplicar_regla_de_cierre(resultados if resultados is not None else self.resultados,
                                       self.protocolo.regla_de_cierre,
                                       motor=_ID_EN_EL_JSON[motor_en_el_texto])

    def _afirma_la_misma_fraccion(self, escrito: str, medido: float, de_quien: str) -> None:
        """La fracción escrita tiene que ser la medida **a la precisión con la
        que está escrita**.

        ESTO ERA UN TRUNCAMIENTO HASTA EL 2026-09-14, y el aserto era el que
        estaba mal, no el producto: comparaba contra `int(fraccion * 1000) /
        1000` porque los números que había entonces (0,8333 -> «0,833»;
        0,5833 -> «0,583») salen iguales truncados que redondeados. Con
        `xgboost` en la competición aparece 8/12 = 0,6666..., que se escribe
        «0,667» redondeando y «0,666» truncando — y el texto redondea, que es lo
        correcto. El aserto viejo lo declaraba rojo.

        Comparar a la precisión escrita vale para los dos casos y sigue teniendo
        dientes de sobra: la diferencia entre 8/12 y 10/12 es de 0,166, más de
        cien veces la tolerancia."""
        self.assertAlmostEqual(
            _como_numero(escrito), medido, places=_decimales_de(escrito),
            msg=f"{de_quien}: la cartera escribe {escrito!r} y la regla re-aplicada da "
                f"{medido!r}")


class LaEntradaDeCarteraEsLaQueSeMIDIOTest(_ContraLaPasadaQueLaEntradaCITA):
    """La entrada de catboost, contra la pasada real y la regla registrada."""

    # -- la mitad que promueve ------------------------------------------
    def test_el_numero_que_afirma_la_cartera_SALE_de_aplicar_la_regla(self):
        """EL TEST DEL HALLAZGO. Vuelve a aplicar la regla pre-registrada a la
        pasada real y exige que salga exactamente lo que la entrada afirma.

        Con el sabotaje del auditor —hundir el `auroc` del motor sellado en dos
        datasets— la re-derivación baja y este aserto se pone rojo diciendo los
        dos números."""
        m = _buscar(rf"{re.escape(_MOTOR_DE_LA_CARTERA)} (\d+)/(\d+) = (\d,\d+) >= (\d,\d+), CUMPLE",
                    self.entrada.evidencia, "el número, la fracción y el listón")
        cumplidos, total, fraccion, listón = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)

        r = self._regla_sobre(_MOTOR_DE_LA_CARTERA)
        self.assertEqual((r["cumplidos"], r["datasets"]), (cumplidos, total),
                         f"la cartera afirma {cumplidos}/{total} y la regla aplicada a "
                         f"{self.nombre_del_json} da {r['cumplidos']}/{r['datasets']}")
        self._afirma_la_misma_fraccion(fraccion, r["fraccion"], _MOTOR_DE_LA_CARTERA)
        self.assertAlmostEqual(_como_numero(listón), r["fraccion_minima"], places=6)
        self.assertTrue(r["cumple_la_regla"],
                        "la cartera dice CUMPLE y la regla re-aplicada dice que no")

    def test_la_regla_citada_es_la_REGISTRADA_CON_HASH_no_otra(self):
        """«a menos de 2 puntos del mejor en >=80 % de los datasets, contando
        un fallo como dataset perdido». Si alguien relaja el protocolo para que
        el número salga, el texto y el protocolo dejan de coincidir."""
        m = _buscar(r"a menos de (\d+) puntos del mejor en >=(\d+) % de los datasets",
                    self.entrada.evidencia, "la regla con sus dos parámetros")
        regla = self.protocolo.regla_de_cierre
        self.assertEqual(float(m.group(1)), regla.puntos)
        self.assertEqual(float(m.group(2)) / 100.0, regla.fraccion_minima)
        self.assertIn("fallo", regla.definicion_de_mejor.lower(),
                      "la entrada afirma «contando un fallo como dataset perdido»: "
                      "eso tiene que seguir estando en la definición registrada")

    def test_los_intentos_sin_un_solo_fallo_estan_en_el_JSON(self):
        """El recuento sale del texto, no del nombre de este método: el nombre
        traía «720» y se quedó viejo el día que la pasada pasó a 1.260."""
        m = _buscar(r"(\d+) de (\d+) intentos completados, CERO fallos",
                    self.entrada.evidencia, "los intentos y los fallos")
        completados, total = int(m.group(1)), int(m.group(2))
        self.assertEqual(len(self.resultados), total)
        self.assertEqual(sum(1 for r in self.resultados if r.get("estado") == "completed"),
                         completados)
        self.assertEqual([r for r in self.resultados if r.get("estado") != "completed"], [],
                         "«CERO fallos» y el JSON trae intentos no completados")

    def test_la_version_de_la_BIBLIOTECA_que_la_cartera_aprueba_es_la_medida(self):
        """La cartera aprueba el par motor + biblioteca (ver `MotorSoportado`).
        La evidencia dice con qué versión se midió, y el rango aprobado tiene
        que empezar ahí: aprobar desde una versión que nadie midió sería otra
        vez afirmar lo que no se ha comprobado."""
        m = _buscar(rf"medida con {re.escape(_MOTOR_DE_LA_CARTERA)} (\d+\.\d+\.\d+)",
                    self.entrada.evidencia, "con qué versión de la biblioteca se midió")
        self.assertIsNotNone(self.entrada.versiones_de_la_biblioteca)
        self.assertEqual(self.entrada.versiones_de_la_biblioteca.minima, m.group(1))
        self.assertEqual(self.entrada.biblioteca, _MOTOR_DE_LA_CARTERA)

    # -- la otra mitad: los que NO están, y por qué --------------------
    def test_los_otros_motores_NO_cumplen_la_regla_y_NO_estan_en_la_cartera(self):
        """Sin esta mitad, el test de arriba lo pasaría una regla que devuelve
        «cumple» siempre. Y los números de los rivales están en la misma
        entrada de cartera, así que se re-derivan igual.

        **LOS CINCO, incluido `lightgbm`**, que hasta el 2026-09-14 estaba en
        el otro lado de este mismo fichero: aquí es donde se mide que salió, y
        que salió porque la regla —sin tocarla— dice que no cumple."""
        for nombre in ("lightgbm", "sklearn.hgb", "xgboost", "sklearn.lineal", "densa"):
            with self.subTest(motor=nombre):
                m = _buscar(rf"{re.escape(nombre)} (\d+)/(\d+) = (\d,\d+)",
                            self.entrada.evidencia, f"el número de {nombre}")
                r = self._regla_sobre(nombre)
                self.assertEqual((r["cumplidos"], r["datasets"]),
                                 (int(m.group(1)), int(m.group(2))))
                self._afirma_la_misma_fraccion(m.group(3), r["fraccion"], nombre)
                self.assertFalse(r["cumple_la_regla"])
                self.assertNotIn(_ID_EN_EL_JSON[nombre], [e.motor for e in CARTERA_APROBADA])

    def test_esta_en_la_cartera_EXACTAMENTE_quien_cumple_la_regla(self):
        """El invariante entero, sin citar ningún número: quien la regla
        aprueba está, y quien no, no. Es lo que impide una promoción a mano."""
        aprobados_por_la_regla = {
            nombre for nombre, id_json in _ID_EN_EL_JSON.items()
            if id_json != _ID_EN_EL_JSON[_QUE_NO_COMPITE]
            and aplicar_regla_de_cierre(self.resultados, self.protocolo.regla_de_cierre,
                                        motor=id_json)["cumple_la_regla"]}
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

    def test_el_digest_que_la_entrada_SELLA_es_el_de_ESE_JSON(self):
        """`evidencia` NOMBRA el artefacto; `evidencia_digest` lo IDENTIFICA —
        campo obligatorio desde el 2026-09-14— y **solo aquí se pueden juntar
        los dos**: `matrixai-engines` no alcanza `benchmarks/`, así que en su
        suite el digest es una cadena de 64 caracteres bien formada y nada más.

        Sin esto, reescribir la pasada conservando el nombre dejaría la entrada
        citándola igual con los números de debajo cambiados, que es justo lo que
        el campo vino a impedir."""
        self.assertEqual(self.entrada.evidencia_digest,
                         self.payload["digest_resultados_crudos"],
                         f"la entrada sella un digest que no es el de {self.nombre_del_json}")
        # Y el prefijo abreviado del texto tiene que ser prefijo del de verdad:
        # media verdad en un digest se lee igual de bien que uno entero.
        abreviado = _buscar(r"digest de sus resultados crudos `([0-9a-f]+)\.\.\.`",
                            self.entrada.evidencia, "el digest abreviado de la pasada").group(1)
        self.assertTrue(self.entrada.evidencia_digest.startswith(abreviado),
                        f"el texto abrevia {abreviado!r} y el digest sellado es "
                        f"{self.entrada.evidencia_digest!r}")

    def test_el_ALCANCE_declara_el_recorte_que_la_pasada_tuvo_de_verdad(self):
        """Los tres recortes, re-derivados del JSON en vez de creídos: cuántos
        motores corrieron, cuántos datasets y **cuántas configuraciones** —la
        tercera dimensión, que faltaba hasta el 2026-09-14 y es la que dice si
        el rival corrió en su mejor versión o en la de fábrica—.

        Los motores y los datasets salen de los propios registros. El número de
        configuraciones que el PROTOCOLO registra sale del protocolo; el que
        corrió, de las configuraciones distintas que aparecen en los registros,
        una por motor."""
        motores_en_los_registros = {r["motor"] for r in self.resultados}
        self.assertEqual(len(self.alcance.motores_que_corrieron), len(motores_en_los_registros),
                         "el alcance dice que corrieron unos motores y el JSON trae otros")
        self.assertEqual(len(self.alcance.motores_que_no_compitieron), 0,
                         "esta pasada es la CONFORME: no falta ninguno, y el alcance lo "
                         "declara con una lista vacía — que afirma, no calla")

        datasets = {r["dataset"] for r in self.resultados}
        self.assertEqual(self.alcance.datasets_que_corrieron, len(datasets))
        self.assertEqual(self.alcance.datasets_del_protocolo, len(self.protocolo.datasets))

        configuraciones_que_corrieron = {(r["motor"], r["configuracion"])
                                         for r in self.resultados}
        self.assertEqual(self.alcance.configuraciones_que_corrieron,
                         len(configuraciones_que_corrieron),
                         "el alcance declara un número de configuraciones que no es el "
                         "que hay en los registros")
        self.assertEqual(self.alcance.configuraciones_del_protocolo,
                         sum(m.configuraciones for m in self.protocolo.motores),
                         "el alcance declara un total de configuraciones que no es el que "
                         "el protocolo registra: sin ese denominador, «7 configuraciones» "
                         "se lee como si fueran todas")
        self.assertLess(self.alcance.configuraciones_que_corrieron,
                        self.alcance.configuraciones_del_protocolo,
                        "si algún día corren TODAS, este aserto avisa de que el texto "
                        "—que hoy explica por qué falta la mitad— se quedó viejo")

    def test_el_MARGEN_y_los_PRIMEROS_PUESTOS_del_alcance_son_los_medidos(self):
        """Los dos números que hacen legible el veredicto, y los dos se pueden
        inventar a mano en la entrada sin que nada más se entere: cuántos
        datasets puede perder antes de bajarse del listón, y cuántos de sus
        aciertos son «es el mejor de los que corrieron» (distancia 0,0000)."""
        r = self._regla_sobre(_MOTOR_DE_LA_CARTERA)
        self.assertEqual(self.alcance.datasets_cumplidos, r["cumplidos"])
        self.assertEqual(self.alcance.datasets_evaluados, r["datasets"])
        self.assertEqual(self.alcance.margen_en_datasets,
                         r["datasets_que_puede_perder_sin_incumplir"])
        self.assertEqual(self.alcance.aciertos_por_ser_el_mejor,
                         r["aciertos_por_ser_el_mejor"])
        # Y por el otro lado: «el mejor» es distancia 0,0000 en el detalle.
        self.assertEqual(sum(1 for d in r["detalle"] if d["distancia_en_puntos"] == 0.0),
                         self.alcance.aciertos_por_ser_el_mejor)


class LaEvidenciaNoDiceDeMASDeLoQueSeMIDIOTest(_ContraLaPasadaQueLaEntradaCITA):
    """HALLAZGO de la auditoría del 2026-09-12, y sigue siendo el mismo aunque
    la afirmación haya cambiado: la entrada de entonces decía «es el unico de
    los cuatro que reprodujo 177/177 intentos entre las dos pasadas», y medido
    era falso en las dos mitades —fueron 177 de 180, y el baseline reprodujo
    180/180, estrictamente más—. Media verdad tranquilizadora es peor que
    callarse, sobre todo dentro de la evidencia que sostiene una promoción.

    **AQUELLA FRASE YA NO ESTÁ: la entrada de hoy no habla de reproducibilidad
    entre pasadas.** Lo que esta clase re-derivaba —`_reproducidos_entre_las_
    dos_pasadas()`, que emparejaba el 07-09 con el 13-09— se retiró con ella: si
    se hubiera quedado, este fichero estaría verde comparando dos pasadas que la
    cartera ya no cita, que es la forma más cara de no medir nada.

    LO QUE SE RE-DERIVA AHORA es la afirmación equivalente de la entrada de hoy,
    y es más grande: «EL 12/12 NO ES TAN SOLIDO COMO SE LEE CON MEDIAS». La
    entrada se obliga a declarar tres cosas que la debilitan —dos primeros
    puestos que la dispersión discute, un dataset que cumple dentro del ruido, y
    tres rivales que cambian de lado del listón según la semilla— y las pone en
    campos comprobables de `alcance`. Comprobables quiere decir esto: aquí se
    vuelven a medir contra el JSON, con la regla registrada, y tienen que salir
    EXACTAMENTE lo declarado. Ni de menos —sería esconder— ni de más —sería
    alarmar con algo que no se midió—.
    """

    def _detalle_del_motor_sellado(self) -> list[dict]:
        return self._regla_sobre(_MOTOR_DE_LA_CARTERA)["detalle"]

    def _auroc_por_pliegue(self, motor: str, dataset: str) -> dict:
        """El AUROC de cada (repetición, pliegue). Emparejar por pliegue es lo
        que convierte «gana por 0,007 de media» en «pierde 9 de 15»."""
        return {(r["repeticion"], r["pliegue"]): r["auroc"] for r in self.resultados
                if r["motor"] == motor and r["dataset"] == dataset
                and r.get("auroc") is not None}

    def test_el_UNICO_QUE_CUMPLE_es_una_medida_y_no_una_forma_de_hablar(self):
        """La entrada dice «Es el UNICO que cumple». Un superlativo es justo lo
        que se coló la vez anterior («el unico de los cuatro»), así que se
        vuelve a contar: de los que COMPITEN —el baseline no, la regla lo
        excluye del cálculo de «el mejor»— cumple exactamente uno, y es el
        sellado."""
        self.assertIn("UNICO que cumple", self.entrada.evidencia,
                      "si la entrada ha dejado de afirmar el superlativo, este test se "
                      "reescribe con ella en vez de seguir midiendo una frase que ya no "
                      "está")
        cumplen = {nombre for nombre in _ID_EN_EL_JSON
                   if nombre != _QUE_NO_COMPITE
                   and self._regla_sobre(nombre)["cumple_la_regla"]}
        self.assertEqual(cumplen, {_MOTOR_DE_LA_CARTERA})

    def test_los_primeros_puestos_QUE_LA_DISPERSION_DISCUTE_son_los_medidos(self):
        """«De sus CINCO aciertos por ser el mejor, DOS los decide una
        diferencia que la dispersión se come.» Se re-deriva el criterio entero:
        de los datasets que gana a distancia 0,0000, los discutidos son aquellos
        en los que **pierde la mayoría de los pliegues emparejados** contra el
        segundo. Tienen que salir exactamente los declarados.

        Las dos direcciones importan y por eso es una igualdad de conjuntos: si
        saliera uno de más, la entrada estaría escondiendo; si de menos, estaría
        alarmando con algo que la pasada no dice."""
        discutidos_medidos = set()
        for d in self._detalle_del_motor_sellado():
            if d["distancia_en_puntos"] != 0.0 or d["segundo"] is None:
                continue
            mio = self._auroc_por_pliegue(_ID_EN_EL_JSON[_MOTOR_DE_LA_CARTERA], d["dataset"])
            suyo = self._auroc_por_pliegue(d["segundo"], d["dataset"])
            comunes = set(mio) & set(suyo)
            perdidos = sum(1 for k in comunes if mio[k] < suyo[k])
            if perdidos * 2 > len(comunes):
                discutidos_medidos.add(d["dataset"])

        declarados = self.alcance.aciertos_que_la_dispersion_discute
        self.assertIsNotNone(declarados,
                             "`aciertos_que_la_dispersion_discute` a `None` dice «no se "
                             "midió», y está medido: sale de la propia pasada")
        self.assertEqual(set(declarados), discutidos_medidos)

    def test_y_el_TEXTO_dice_con_cuanto_gana_y_cuantos_pliegues_pierde(self):
        """LA MITAD QUE COSTÓ EL HALLAZGO LA VEZ ANTERIOR: que el campo esté
        bien no basta si la prosa que lee una persona sigue diciendo «12/12» a
        secas. La entrada nombra los dos datasets con su ventaja y sus pliegues
        perdidos, y esos números también se re-derivan.

        Un aserto negativo —«que no diga que es sólido»— lo pasaría un texto
        vacío; esto exige lo que SÍ tiene que estar."""
        for dataset, patron in (
                ("wilt", r"en wilt gana por (\d,\d+) puntos y pierde (\d+) de los (\d+) "
                         r"pliegues emparejados"),
                ("sick", r"en sick gana por (\d,\d+) y pierde otros (\d+) de (\d+)")):
            with self.subTest(dataset=dataset):
                m = _buscar(patron, self.entrada.evidencia,
                            f"cuánto gana en {dataset} y cuántos pliegues pierde")
                ventaja, perdidos, emparejados = (_como_numero(m.group(1)),
                                                  int(m.group(2)), int(m.group(3)))
                d = next(x for x in self._detalle_del_motor_sellado()
                         if x["dataset"] == dataset)
                medida = d["ventaja_sobre_el_segundo_en_puntos"]
                self.assertAlmostEqual(
                    ventaja, round(medida, _decimales_de(m.group(1))), places=6,
                    msg=f"la entrada dice que en {dataset} gana por {ventaja} y se "
                        f"midió {medida}")
                mio = self._auroc_por_pliegue(_ID_EN_EL_JSON[_MOTOR_DE_LA_CARTERA], dataset)
                suyo = self._auroc_por_pliegue(d["segundo"], dataset)
                comunes = set(mio) & set(suyo)
                self.assertEqual(len(comunes), emparejados)
                self.assertEqual(sum(1 for k in comunes if mio[k] < suyo[k]), perdidos)

    def test_el_dataset_que_cumple_DENTRO_DEL_RUIDO_esta_declarado_y_es_el_medido(self):
        """El resumen del propio JSON NO lo enseña: su contador
        `intervalos_de_los_que_NO_cumplen_que_cruzan_el_liston` mira solo a los
        que NO cumplen, y vale 0. Quien leyera el resumen tendría un 12/12 sin
        enterarse de que uno de los doce está dentro del ruido, y por eso la
        entrada lo declara aparte. Aquí se re-deriva de los intervalos."""
        dentro_del_ruido = {d["dataset"] for d in self._detalle_del_motor_sellado()
                            if d["cumple"]
                            and (d.get("intervalo_de_la_distancia") or {}).get("cruza_el_liston")}
        declarados = self.alcance.cumplidos_cuyo_intervalo_cruza_el_liston
        self.assertIsNotNone(declarados,
                             "`cumplidos_cuyo_intervalo_cruza_el_liston` a `None` dice "
                             "«no se midió», y los intervalos están en la pasada")
        self.assertEqual(set(declarados), dentro_del_ruido)

        # Y la prosa, con su distancia y su intervalo — los tres números.
        m = _buscar(r"([\w.\-]+), cumple DENTRO DEL RUIDO: (\d,\d+) puntos contra un liston "
                    r"de (\d,\d+), con intervalo al \d+ % de \[(\d,\d+) \.\. (\d,\d+)\]",
                    self.entrada.evidencia, "el dataset que cumple dentro del ruido")
        self.assertIn(m.group(1), dentro_del_ruido,
                      "la prosa nombra un dataset que la medición no pone dentro del ruido")
        d = next(x for x in self._detalle_del_motor_sellado() if x["dataset"] == m.group(1))
        intervalo = d["intervalo_de_la_distancia"]
        self.assertAlmostEqual(_como_numero(m.group(2)),
                               round(d["distancia_en_puntos"], _decimales_de(m.group(2))),
                               places=6)
        self.assertAlmostEqual(_como_numero(m.group(3)), self.protocolo.regla_de_cierre.puntos,
                               places=6)
        self.assertAlmostEqual(_como_numero(m.group(4)),
                               round(intervalo["bajo"], _decimales_de(m.group(4))), places=6)
        self.assertAlmostEqual(_como_numero(m.group(5)),
                               round(intervalo["alto"], _decimales_de(m.group(5))), places=6)
        # El intervalo tiene que cruzar el listón de verdad, no solo estar escrito:
        # sin esto, unos números copiados a mano que no lo crucen pasarían.
        self.assertLess(_como_numero(m.group(4)), self.protocolo.regla_de_cierre.puntos)
        self.assertGreater(_como_numero(m.group(5)), self.protocolo.regla_de_cierre.puntos)

    def test_quien_CAMBIA_DE_LADO_DEL_LISTON_segun_la_semilla_esta_medido(self):
        """Re-aplicar la regla registrada a cada semilla por separado. Los que
        cumplen con una y no con otra son los declarados — y el motor sellado no
        puede ser uno de ellos: la cartera se niega a construir esa entrada, así
        que si apareciera aquí sería que el guardia no funciona."""
        semillas = sorted({r["repeticion"] for r in self.resultados})
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

        declarados = self.alcance.motores_que_cambian_de_lado_del_liston_segun_la_semilla
        self.assertIsNotNone(declarados, "`motores_que_cambian_de_lado...` a `None` dice «no "
                                         "se midió», y se mide re-aplicando la regla")
        self.assertEqual(set(declarados), cambian)
        self.assertNotIn(_ID_EN_EL_JSON[_MOTOR_DE_LA_CARTERA], cambian,
                         "el motor sellado cambia de lado del listón según la semilla: la "
                         "cartera no debería haber podido construir esta entrada")

    def test_el_recuento_DE_CADA_SEMILLA_es_el_medido(self):
        """`(12, 12, 11)` se lee entero: el techo, el suelo, y dónde cae el
        publicado. Es una comprobación de SENSIBILIDAD y no un segundo
        veredicto —cada semilla son 5 pliegues en vez de 15, o sea menos datos—,
        pero es la que sostiene las dos lecturas sin que el campo tenga que
        elegir cuál contar."""
        semillas = sorted({r["repeticion"] for r in self.resultados})
        medidos = tuple(self._regla_sobre(_MOTOR_DE_LA_CARTERA,
                                          [r for r in self.resultados if r["repeticion"] == s]
                                          )["cumplidos"] for s in semillas)
        self.assertIsNotNone(self.alcance.cumplidos_por_semilla,
                             "`cumplidos_por_semilla` a `None` dice «no se midió»; la "
                             "entrada de hoy lo declara y la pasada lo permite medir")
        self.assertEqual(tuple(self.alcance.cumplidos_por_semilla), medidos)

        # Y el texto, que cita los mismos tres números.
        m = _buscar(rf"{re.escape(_MOTOR_DE_LA_CARTERA)} cumple con las tres "
                    r"\((\d+), (\d+) y (\d+) de (\d+)\)",
                    self.entrada.evidencia, "el recuento de cada semilla")
        self.assertEqual(tuple(int(m.group(i)) for i in (1, 2, 3)), medidos)
        self.assertEqual(int(m.group(4)), self.alcance.datasets_evaluados)

    def test_cumplidos_con_menos_semillas_SOLO_se_rellena_si_lo_publicado_es_el_SUELO(self):
        """LA LÍNEA QUE EXPLICA POR QUÉ NO HACE LO OBVIO, con su prueba.

        `cumplidos_con_menos_semillas` está a `None` en esta entrada **a
        propósito**, y su comentario en el core dice por qué: la pantalla
        escribe debajo de ese número «que es la lectura más estricta», y aquí
        leer con menos semillas BAJA el recuento (12 con las tres, 11 con la
        semilla 2 sola), así que rellenarlo pondría esa frase debajo del techo.
        Para la entrada anterior era al revés y por eso el campo existe.

        Sin este test, el siguiente que pase rellena el campo «por completitud»
        —el guardia del core solo mira que quede entre el publicado y el total,
        y un 12 cabe— y el producto escribe una media verdad tranquilizadora.
        Aquí se mide la dirección de verdad, contra la pasada."""
        semillas = sorted({r["repeticion"] for r in self.resultados})
        por_semilla = [self._regla_sobre(_MOTOR_DE_LA_CARTERA,
                                         [r for r in self.resultados if r["repeticion"] == s]
                                         )["cumplidos"] for s in semillas]
        publicado = self.alcance.datasets_cumplidos
        lo_publicado_es_el_suelo = publicado <= min(por_semilla)

        if lo_publicado_es_el_suelo:
            self.assertIsNotNone(
                self.alcance.cumplidos_con_menos_semillas,
                "leer con menos semillas solo puede subir el recuento, así que hay una "
                "lectura menos estricta que enseñar y el campo se queda callado")
        else:
            self.assertIsNone(
                self.alcance.cumplidos_con_menos_semillas,
                f"lo publicado ({publicado}) NO es el suelo (con menos semillas baja a "
                f"{min(por_semilla)}), así que este campo tiene que quedarse en `None`: "
                "rellenarlo haría que el producto escribiera «la lectura más estricta» "
                "debajo del TECHO")

    def test_la_entrada_declara_lo_que_la_DEBILITA_y_no_solo_lo_que_la_sostiene(self):
        """La mitad que caza el esconder, y va aparte de las igualdades de
        arriba a propósito: aquellas comparan conjuntos, y un conjunto vacío
        declarado contra un conjunto vacío medido las pasaría las dos. Esto
        exige que, **habiendo algo que declarar, esté declarado** — que es la
        forma que tiene «declarar lo que PASÓ» de poder fallar."""
        hay_algo_que_decir = {
            "aciertos_que_la_dispersion_discute": bool(
                self.alcance.aciertos_que_la_dispersion_discute),
            "cumplidos_cuyo_intervalo_cruza_el_liston": bool(
                self.alcance.cumplidos_cuyo_intervalo_cruza_el_liston),
            "motores_que_cambian_de_lado_del_liston_segun_la_semilla": bool(
                self.alcance.motores_que_cambian_de_lado_del_liston_segun_la_semilla)}
        self.assertTrue(any(hay_algo_que_decir.values()),
                        "los tres campos de debilidad han salido vacíos: o la pasada ha "
                        "cambiado, o este test dejó de mirar donde tenía que mirar")
        # Y el texto no puede vender solidez sin matizarla: la entrada dice con
        # todas las letras que el número no se lee con medias.
        self.assertIn("NO ES TAN SOLIDO", self.entrada.evidencia)
        for nombre in sorted(self.alcance.aciertos_que_la_dispersion_discute or ()):
            self.assertIn(nombre, self.entrada.evidencia,
                          f"{nombre!r} está en el alcance y no en la prosa: quien lee el "
                          "texto no se entera de cuál es")
        for nombre in sorted(self.alcance.cumplidos_cuyo_intervalo_cruza_el_liston or ()):
            self.assertIn(nombre, self.entrada.evidencia)


if __name__ == "__main__":
    unittest.main()
