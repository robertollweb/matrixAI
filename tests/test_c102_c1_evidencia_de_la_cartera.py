# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""102-C1 — LO QUE LA CARTERA AFIRMA SE RE-DERIVA, no se cree.

HALLAZGO GRAVE de la auditoría externa del 2026-09-12, y la razón de que este
fichero exista: la entrada de cartera de `matrixai_engines` afirma «lightgbm
10/12 = 0,833 >= 0,80, CUMPLE» sobre la pasada real, y **ningún test de ningún
repositorio leía ese JSON**. Lo único que se comprobaba era que la CADENA
«10/12» apareciera en el texto de la evidencia. El auditor hundió el `auroc`
de lightgbm en dos datasets del JSON —la regla pasaba a 8/12 = 0,667, NO
CUMPLE— y las dos suites siguieron VERDES: la promoción de un motor a
«soportado» descansaba sobre una frase, no sobre una medición.

POR QUÉ ESTE TEST VIVE EN `matrixAI` Y NO EN `matrixai-engines`, que es donde
está la afirmación:

1. Aquí viven las DOS cosas que hay que volver a juntar: el JSON de la pasada
   (`benchmarks/fase0/pasada_exploratoria_101_c3_remedida_20260913.json`) y la
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
  * promover o despromover un motor sin que la medición lo acompañe.

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

#: Cómo se llaman en el texto de la cartera los motores que en el JSON tienen
#: otro id. Escrito aquí y no adivinado: «no adivinar, medir».
_ID_EN_EL_JSON = {"lightgbm": "lightgbm", "sklearn.lineal": "sklearn.lineal",
                  "densa": "matrixai.dense.torch_cpu", "baseline": "baseline"}


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


class LaEntradaDeCarteraEsLaQueSeMIDIOTest(unittest.TestCase):
    """La entrada de lightgbm, contra la pasada real y la regla registrada."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.entrada = next(e for e in CARTERA_APROBADA if e.motor == "lightgbm")
        cls.protocolo = ProtocoloExploratorio.cargar(str(_FASE0 / "protocolo_exploratorio.json"))
        cls.nombre_del_json = _buscar(r"`(pasada_exploratoria_\S+?\.json)`", cls.entrada.evidencia,
                                      "de qué fichero de evidencia habla").group(1)
        cls.payload = json.loads((_FASE0 / cls.nombre_del_json).read_text(encoding="utf-8"))
        cls.resultados = cls.payload["resultados"]

    def _regla_sobre(self, motor_en_el_texto: str) -> dict:
        return aplicar_regla_de_cierre(self.resultados, self.protocolo.regla_de_cierre,
                                       motor=_ID_EN_EL_JSON[motor_en_el_texto])

    # -- la mitad que promueve ------------------------------------------
    def test_el_numero_que_afirma_la_cartera_SALE_de_aplicar_la_regla(self):
        """EL TEST DEL HALLAZGO. Vuelve a aplicar la regla pre-registrada a la
        pasada real y exige que salga exactamente lo que la entrada afirma.

        Con el sabotaje del auditor —hundir el `auroc` de lightgbm en dos
        datasets— la re-derivación da 8/12 y este aserto se pone rojo diciendo
        los dos números."""
        m = _buscar(r"lightgbm (\d+)/(\d+) = (\d,\d+) >= (\d,\d+), CUMPLE",
                    self.entrada.evidencia, "el número, la fracción y el listón")
        cumplidos, total, fraccion, listón = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)

        r = self._regla_sobre("lightgbm")
        self.assertEqual((r["cumplidos"], r["datasets"]), (cumplidos, total),
                         f"la cartera afirma {cumplidos}/{total} y la regla aplicada a "
                         f"{self.nombre_del_json} da {r['cumplidos']}/{r['datasets']}")
        # Truncado a tres decimales, que es como está escrito (0,8333... -> 0,833).
        self.assertAlmostEqual(_como_numero(fraccion), int(r["fraccion"] * 1000) / 1000.0,
                               places=6)
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

    def test_los_720_intentos_sin_un_solo_fallo_estan_en_el_JSON(self):
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
        La evidencia dice con qué versión de lightgbm se midió, y el rango
        aprobado tiene que empezar ahí: aprobar desde una versión que nadie
        midió sería otra vez afirmar lo que no se ha comprobado."""
        m = _buscar(r"medida con lightgbm (\d+\.\d+\.\d+)", self.entrada.evidencia,
                    "con qué versión de la biblioteca se midió")
        self.assertIsNotNone(self.entrada.versiones_de_la_biblioteca)
        self.assertEqual(self.entrada.versiones_de_la_biblioteca.minima, m.group(1))
        self.assertEqual(self.entrada.biblioteca, "lightgbm")

    # -- la otra mitad: los que NO están, y por qué --------------------
    def test_los_otros_motores_NO_cumplen_la_regla_y_NO_estan_en_la_cartera(self):
        """Sin esta mitad, el test de arriba lo pasaría una regla que devuelve
        «cumple» siempre. Y los números de los rivales están en la misma
        entrada de cartera, así que se re-derivan igual."""
        for nombre, patron in (("sklearn.lineal", r"sklearn\.lineal (\d+)/(\d+) = (\d,\d+)"),
                               ("densa", r"densa (\d+)/(\d+) = (\d,\d+)")):
            with self.subTest(motor=nombre):
                m = _buscar(patron, self.entrada.evidencia, f"el número de {nombre}")
                r = self._regla_sobre(nombre)
                self.assertEqual((r["cumplidos"], r["datasets"]),
                                 (int(m.group(1)), int(m.group(2))))
                self.assertAlmostEqual(_como_numero(m.group(3)),
                                       int(r["fraccion"] * 1000) / 1000.0, places=6)
                self.assertFalse(r["cumple_la_regla"])
                self.assertNotIn(_ID_EN_EL_JSON[nombre], [e.motor for e in CARTERA_APROBADA])

    def test_esta_en_la_cartera_EXACTAMENTE_quien_cumple_la_regla(self):
        """El invariante entero, sin citar ningún número: quien la regla
        aprueba está, y quien no, no. Es lo que impide una promoción a mano."""
        aprobados_por_la_regla = {
            nombre for nombre, id_json in _ID_EN_EL_JSON.items()
            if id_json != "baseline"
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


class LaEvidenciaNoDiceDeMASDeLoQueSeMIDIOTest(unittest.TestCase):
    """HALLAZGO de la misma auditoría: la entrada afirmaba «es el unico de los
    cuatro que reprodujo 177/177 intentos entre las dos pasadas». Medido, es
    falso en las dos mitades: fueron 177 de 180 (no 177 de 177), y el baseline
    reprodujo 180/180 — estrictamente más. Media verdad tranquilizadora es
    peor que callarse, sobre todo dentro de la evidencia que sostiene una
    promoción.

    El texto corregido afirma menos y lo afirma entero: el mejor DE LOS TRES
    QUE COMPITEN, con los tres números, y con el baseline declarado aparte y
    con su motivo. Esto lo re-deriva."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.entrada = next(e for e in CARTERA_APROBADA if e.motor == "lightgbm")
        cls.identicos = _reproducidos_entre_las_dos_pasadas()

    def test_los_numeros_de_reproducibilidad_son_los_medidos(self):
        for nombre, patron in (
                ("lightgbm", r"(\d+)/(\d+) intentos identicos"),
                ("sklearn.lineal", r"frente a (\d+)/(\d+) de\s+sklearn\.lineal"),
                ("densa", r"(\d+)/(\d+) de la densa"),
                ("baseline", r"El baseline reprodujo (\d+)/(\d+)")):
            with self.subTest(motor=nombre):
                m = _buscar(patron, self.entrada.evidencia, f"la reproducibilidad de {nombre}")
                self.assertEqual(self.identicos[_ID_EN_EL_JSON[nombre]],
                                 (int(m.group(1)), int(m.group(2))))

    def test_el_baseline_reprodujo_MAS_y_la_entrada_no_lo_esconde(self):
        """La mitad que costó el hallazgo: el baseline reprodujo estrictamente
        más que lightgbm, así que «el único de los cuatro» era falso. Si algún
        día lightgbm reprodujera más que el baseline, este test avisa de que el
        texto —que hoy explica por qué el baseline va aparte— se quedó viejo."""
        self.assertGreater(self.identicos["baseline"][0], self.identicos["lightgbm"][0])
        self.assertNotIn("unico de los cuatro", self.entrada.evidencia)
        self.assertIn("baseline", self.entrada.evidencia,
                      "declarar lo que PASÓ: el baseline reprodujo más y eso se dice, "
                      "con el motivo por el que aun así no compite")

    def test_lightgbm_reprodujo_mas_que_los_OTROS_DOS_que_compiten(self):
        """Lo que sí es cierto, y es lo que la entrada afirma ahora."""
        for otro in ("sklearn.lineal", "matrixai.dense.torch_cpu"):
            self.assertGreater(self.identicos["lightgbm"][0], self.identicos[otro][0], otro)


def _reproducidos_entre_las_dos_pasadas() -> dict[str, tuple[int, int]]:
    """Intentos con el MISMO `auroc` entre la pasada del 07-09 y la
    re-medición del 09-12, emparejados por (dataset, motor, pliegue,
    repetición). Devuelve `{motor: (identicos, comunes)}`."""
    def indexar(ruta: Path) -> dict:
        crudo = json.loads(ruta.read_text(encoding="utf-8"))
        return {(r["dataset"], r["motor"], r["pliegue"], r["repeticion"]): r
                for r in crudo["resultados"]}

    vieja = indexar(_FASE0 / "pasada_exploratoria_101_c3_resultado.json")
    nueva = indexar(_FASE0 / "pasada_exploratoria_101_c3_remedida_20260913.json")
    cuenta: dict[str, list[int]] = {}
    for clave in set(vieja) & set(nueva):
        a, b = vieja[clave], nueva[clave]
        par = cuenta.setdefault(clave[1], [0, 0])
        par[1] += 1
        if (a.get("estado") == b.get("estado") == "completed"
                and a.get("auroc") is not None and a.get("auroc") == b.get("auroc")):
            par[0] += 1
    return {motor: (identicos, comunes) for motor, (identicos, comunes) in cuenta.items()}


if __name__ == "__main__":
    unittest.main()
