# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Una columna DERIVADA de otra en el proveedor sintético.

Por qué existe (2026-08-11): las plantillas sintéticas declaraban
columnas INDEPENDIENTES, incluida la que hace de objetivo. Un objetivo
independiente de sus entradas es ruido: no hay nada que aprender, y la
propia ficha de esas plantillas tiene que admitir que «un modelo
entrenado con ellas no dice nada sobre el mundo».

Roberto pidió la plantilla de Kelvin —`K = °C + 273,15`—, que es el caso
contrario: una relación exacta, donde se VE que el entrenamiento la
encuentra. Sin columnas derivadas no se puede publicar.

Se prueba la relación FILA A FILA, no un promedio: un generador que
acertara de media y fallara en cada fila pasaría cualquier comprobación
agregada y arruinaría el ejemplo.
"""
from __future__ import annotations

import csv
import io
import unittest

from matrixai.training.data_provider import (
    DataProviderError,
    LicenseAcceptance,
    LicenseAcceptanceStore,
)
from matrixai.training.provider_synthetic_local import SyntheticLocalProvider


def _aceptacion(p: SyntheticLocalProvider) -> LicenseAcceptance:
    """El recibo de licencia como lo emite el registro, no a mano: es un
    recibo auditable con digest de los términos, y uno inventado no
    valdría (`require_valid_acceptance` lo rechaza)."""
    return LicenseAcceptanceStore().record(p, actor="test")


KELVIN = {
    "seed": 20260811,
    "rows": 50,
    "columns": [
        {"name": "centigrados", "type": "number", "range": [-40.0, 60.0]},
        {"name": "kelvin", "type": "linear", "from": "centigrados", "scale": 1.0, "offset": 273.15},
    ],
}


class ColumnaDerivada(unittest.TestCase):
    def setUp(self) -> None:
        self.p = SyntheticLocalProvider()

    def _filas(self, config: dict) -> list[dict[str, str]]:
        r = self.p.download(config, license_acceptance=_aceptacion(self.p))
        return list(csv.DictReader(io.StringIO(r.csv_text)))

    def test_la_relacion_se_cumple_en_TODAS_las_filas(self) -> None:
        filas = self._filas(KELVIN)
        self.assertEqual(len(filas), 50)
        for f in filas:
            self.assertAlmostEqual(
                float(f["kelvin"]), float(f["centigrados"]) + 273.15, places=3,
                msg=f"fila {f}: la derivada no sale de su origen",
            )

    def test_y_las_dos_columnas_VARIAN(self) -> None:
        # Un generador que devolviera siempre el mismo número cumpliría
        # la relación y no serviría para entrenar nada.
        filas = self._filas(KELVIN)
        self.assertGreater(len({f["centigrados"] for f in filas}), 40)

    def test_sigue_siendo_determinista_por_seed(self) -> None:
        self.assertEqual(self._filas(KELVIN), self._filas(KELVIN))

    def test_el_ruido_separa_la_derivada_de_su_recta_pero_no_la_desliga(self) -> None:
        con_ruido = dict(KELVIN)
        con_ruido["columns"] = [
            KELVIN["columns"][0],
            {**KELVIN["columns"][1], "noise": 2.0},
        ]
        filas = self._filas(con_ruido)
        errores = [abs(float(f["kelvin"]) - float(f["centigrados"]) - 273.15) for f in filas]
        self.assertGreater(max(errores), 0.0, "con ruido>0 alguna fila tiene que separarse")
        # Pero sigue siendo la MISMA recta: si el ruido arrastrara el
        # valor, el ejemplo enseñaría una relación que no está.
        self.assertLess(sum(errores) / len(errores), 6.0)

    def test_sin_ruido_no_se_toca_el_valor(self) -> None:
        filas = self._filas(KELVIN)
        self.assertTrue(all(
            abs(float(f["kelvin"]) - float(f["centigrados"]) - 273.15) < 1e-9 for f in filas
        ))


class ConfiguracionesQueSeRECHAZAN(unittest.TestCase):
    """Lo que NO se acepta, que es donde vive la seguridad de esto."""

    def setUp(self) -> None:
        self.p = SyntheticLocalProvider()

    def _errores(self, columnas: list[dict]) -> list[str]:
        return self.p.validate_config({"seed": 1, "rows": 5, "columns": columnas})

    def test_un_origen_declarado_DESPUES_no_vale(self) -> None:
        # Es lo que impide los ciclos por construcción.
        errs = self._errores([
            {"name": "kelvin", "type": "linear", "from": "centigrados", "scale": 1.0, "offset": 273.15},
            {"name": "centigrados", "type": "number", "range": [-40.0, 60.0]},
        ])
        self.assertTrue(any("ANTES" in e for e in errs), errs)

    def test_un_origen_que_no_existe_no_vale(self) -> None:
        errs = self._errores([
            {"name": "a", "type": "number", "range": [0.0, 1.0]},
            {"name": "b", "type": "linear", "from": "fantasma", "scale": 1.0, "offset": 0.0},
        ])
        self.assertTrue(errs)

    def test_apuntarse_a_si_misma_no_vale(self) -> None:
        errs = self._errores([
            {"name": "a", "type": "linear", "from": "a", "scale": 1.0, "offset": 0.0},
        ])
        self.assertTrue(errs)

    def test_faltan_escala_o_desplazamiento(self) -> None:
        errs = self._errores([
            {"name": "a", "type": "number", "range": [0.0, 1.0]},
            {"name": "b", "type": "linear", "from": "a"},
        ])
        self.assertTrue(any("scale" in e for e in errs), errs)
        self.assertTrue(any("offset" in e for e in errs), errs)

    def test_un_ruido_negativo_no_vale(self) -> None:
        errs = self._errores([
            {"name": "a", "type": "number", "range": [0.0, 1.0]},
            {"name": "b", "type": "linear", "from": "a", "scale": 1.0, "offset": 0.0, "noise": -1},
        ])
        self.assertTrue(any("noise" in e for e in errs), errs)

    def test_una_config_invalida_NO_genera_csv(self) -> None:
        with self.assertRaises(DataProviderError):
            self.p.download(
                {"seed": 1, "rows": 5, "columns": [
                    {"name": "b", "type": "linear", "from": "no_existe", "scale": 1.0, "offset": 0.0},
                ]},
                license_acceptance=_aceptacion(self.p),
            )

    def test_lo_de_siempre_sigue_funcionando(self) -> None:
        # Las columnas independientes no cambian: este tipo se AÑADE.
        errs = self._errores([
            {"name": "a", "type": "number", "range": [0.0, 1.0]},
            {"name": "b", "type": "categorical", "categories": ["x", "y"]},
        ])
        self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()


class EtiquetaDerivada(unittest.TestCase):
    """`threshold`: la etiqueta sale de los datos, no de un sorteo aparte.

    Sin esto, una plantilla de CLASIFICACIÓN sortea su columna objetivo
    por separado y no guarda ninguna relación con las entradas: se
    publica «predice si esta máquina va a fallar» y el modelo no puede
    acertar más que echándolo a suertes. Se comprueba la relación, no la
    forma: que las filas de una clase tengan de verdad valores distintos
    de las de otra.
    """

    MAQUINAS = {
        "seed": 7,
        "rows": 300,
        "columns": [
            {"name": "horas_uso", "type": "number", "range": [0.0, 50000.0]},
            {"name": "desgaste", "type": "linear", "from": "horas_uso",
             "scale": 0.00002, "offset": 0.0, "noise": 0.12},
            {"name": "estado", "type": "threshold", "from": "desgaste",
             "cuts": [0.45, 0.8], "labels": ["OK", "DESGASTE", "FALLO"]},
        ],
    }

    def setUp(self) -> None:
        self.p = SyntheticLocalProvider()

    def _filas(self, config: dict) -> list[dict[str, str]]:
        r = self.p.download(config, license_acceptance=_aceptacion(self.p))
        return list(csv.DictReader(io.StringIO(r.csv_text)))

    def test_salen_las_TRES_clases(self) -> None:
        clases = {f["estado"] for f in self._filas(self.MAQUINAS)}
        self.assertEqual(clases, {"OK", "DESGASTE", "FALLO"})

    def test_y_la_clase_DEPENDE_de_la_entrada(self) -> None:
        filas = self._filas(self.MAQUINAS)
        def media(clase: str) -> float:
            v = [float(f["horas_uso"]) for f in filas if f["estado"] == clase]
            return sum(v) / len(v)
        # Lo que hace que haya algo que aprender: más horas de uso, peor
        # estado. Si esto deja de cumplirse, la plantilla vuelve a ser
        # ruido con una etiqueta encima.
        self.assertLess(media("OK"), media("DESGASTE"))
        self.assertLess(media("DESGASTE"), media("FALLO"))

    def test_los_cortes_desordenados_se_RECHAZAN(self) -> None:
        # Un corte fuera de orden deja un tramo vacío: la plantilla
        # declararía una clase que el dataset no contiene.
        errs = self.p.validate_config({"seed": 1, "rows": 5, "columns": [
            {"name": "x", "type": "number", "range": [0.0, 1.0]},
            {"name": "y", "type": "threshold", "from": "x", "cuts": [0.8, 0.2], "labels": ["a", "b", "c"]},
        ]})
        self.assertTrue(any("creciente" in e for e in errs), errs)

    def test_las_etiquetas_tienen_que_ser_UNA_MAS_que_los_cortes(self) -> None:
        errs = self.p.validate_config({"seed": 1, "rows": 5, "columns": [
            {"name": "x", "type": "number", "range": [0.0, 1.0]},
            {"name": "y", "type": "threshold", "from": "x", "cuts": [0.5], "labels": ["a", "b", "c"]},
        ]})
        self.assertTrue(any("una más que cortes" in e for e in errs), errs)

    def test_etiquetas_repetidas_se_rechazan(self) -> None:
        errs = self.p.validate_config({"seed": 1, "rows": 5, "columns": [
            {"name": "x", "type": "number", "range": [0.0, 1.0]},
            {"name": "y", "type": "threshold", "from": "x", "cuts": [0.5], "labels": ["a", "a"]},
        ]})
        self.assertTrue(errs)

    def test_un_origen_posterior_se_rechaza(self) -> None:
        errs = self.p.validate_config({"seed": 1, "rows": 5, "columns": [
            {"name": "y", "type": "threshold", "from": "x", "cuts": [0.5], "labels": ["a", "b"]},
            {"name": "x", "type": "number", "range": [0.0, 1.0]},
        ]})
        self.assertTrue(any("ANTES" in e for e in errs), errs)

    def test_el_reparto_no_deja_ninguna_clase_VACIA(self) -> None:
        # Una clase declarada que no aparece nunca la rechaza el propio
        # verificador del core, con un error que no señalaría aquí.
        filas = self._filas(self.MAQUINAS)
        for clase in ("OK", "DESGASTE", "FALLO"):
            self.assertGreater(sum(1 for f in filas if f["estado"] == clase), 10, clase)


class UnaSerieConMEMORIA(unittest.TestCase):
    """`seasonal`: una columna que recuerda por dónde iba.

    Hallazgo de la auditoría externa de la demo (2026-08-12): «Consumo
    eléctrico de mañana» y «Serie temporal sintética» prometían una
    relación temporal y daban **R² NEGATIVO** —peor que contestar siempre
    la media—. Los retardos y el desplazamiento del objetivo estaban bien
    puestos; lo que faltaba era que hubiera algo que aprender: cada día
    se muestreaba independiente del anterior, así que mirar hoy no decía
    nada de mañana.

    Medido después del arreglo, entrenando 50 épocas: R² 0,689 y 0,916.
    """

    ONDA = {
        "seed": 20260812,
        "rows": 400,
        "columns": [
            {"name": "medida", "type": "seasonal",
             "period": 60, "amplitude": 8.0, "offset": 10.0, "noise": 0.8},
        ],
    }

    def setUp(self) -> None:
        self.p = SyntheticLocalProvider()

    def _valores(self, config: dict) -> list[float]:
        r = self.p.download(config, license_acceptance=_aceptacion(self.p))
        return [float(f["medida"]) for f in csv.DictReader(io.StringIO(r.csv_text))]

    def test_un_valor_se_parece_al_SIGUIENTE(self) -> None:
        """Lo que hace que haya algo que predecir.

        Se compara con el salto que daría una columna INDEPENDIENTE del
        mismo recorrido: si la onda no fuera más suave que el sorteo, no
        habría memoria que aprovechar.
        """
        v = self._valores(self.ONDA)
        salto_onda = sum(abs(v[i + 1] - v[i]) for i in range(len(v) - 1)) / (len(v) - 1)
        sorteados = self._valores({**self.ONDA, "columns": [
            {"name": "medida", "type": "number", "range": [2.0, 18.0]}]})
        salto_azar = sum(abs(sorteados[i + 1] - sorteados[i]) for i in range(len(sorteados) - 1)) / (len(sorteados) - 1)
        self.assertLess(salto_onda, salto_azar / 3, f"onda {salto_onda:.2f} vs azar {salto_azar:.2f}")

    def test_recorre_TODO_su_rango(self) -> None:
        # Una onda que apenas se mueve sería tan inútil como el ruido.
        v = self._valores(self.ONDA)
        self.assertGreater(max(v) - min(v), 12)

    def test_sigue_siendo_determinista(self) -> None:
        self.assertEqual(self._valores(self.ONDA), self._valores(self.ONDA))

    def test_una_amplitud_de_CERO_se_rechaza(self) -> None:
        # Daría una columna constante: ninguna señal, y la plantilla
        # prometería una serie.
        errs = self.p.validate_config({**self.ONDA, "columns": [
            {"name": "m", "type": "seasonal", "period": 60, "amplitude": 0, "offset": 10}]})
        self.assertTrue(any("constante" in e for e in errs), errs)

    def test_un_periodo_no_positivo_se_rechaza(self) -> None:
        for periodo in (0, -5):
            errs = self.p.validate_config({**self.ONDA, "columns": [
                {"name": "m", "type": "seasonal", "period": periodo, "amplitude": 3, "offset": 1}]})
            self.assertTrue(any("period" in e for e in errs), (periodo, errs))
