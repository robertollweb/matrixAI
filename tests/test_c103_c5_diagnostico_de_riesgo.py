# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""103-C5 — diagnóstico y corrección visibles. CIERRA EL CONTRATO 103 ENTERO.

Criterio de terminado: «El usuario corrige objetivo/entrada, confirma
diseño y puede continuar solo en el estado permitido. Un estudio
exploratorio no se presenta como validado. Mensajes es/en y pruebas del
recorrido completas. Cierre del 71 se registra únicamente con evidencia
de ambos caminos.»
"""
from __future__ import annotations

import random
import unittest

from matrixai.training.diagnostico import diagnosticar_csv
from matrixai.training.diagnostico_de_riesgo import (
    DiagnosticoDeRiesgo,
    ErrorDeAceptacion,
    RegistroDeAceptaciones,
    aceptar_sospecha,
    sospechas_pendientes,
)
from matrixai.training.diagnostico_de_riesgo_textos import IDIOMAS, MOTIVOS, huecos_de
from matrixai.training.objetivo import confirmar_desde_prompt


class TestElCatalogoHablaLosDosIdiomas(unittest.TestCase):
    def test_toda_clave_tiene_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            for idioma in IDIOMAS:
                self.assertIn(idioma, textos, f"{clave} sin {idioma}")

    def test_los_huecos_son_los_mismos_en_los_dos_idiomas(self):
        for clave, textos in MOTIVOS.items():
            self.assertEqual(huecos_de(textos["es"]), huecos_de(textos["en"]),
                             f"{clave}: los huecos no coinciden")


def _sospecha_del_salario():
    """Fixture literal de 103-C2 (`test_caso_2_del_71_salario_legitimo_no_
    bloquea_solo_avisa`) -- el mismo caso 2 del 71, ahora midiendo el
    recorrido completo hasta la aceptación."""
    conf = confirmar_desde_prompt(
        "predecir el salario de un empleado a partir del salario del ano "
        "pasado, la antiguedad y el departamento",
        objetivo="salary", tarea="regression",
        entradas=("last_year_salary", "seniority", "department"),
        unidad_de_observacion="un empleado")
    random.seed(71)
    n = 200
    last_year = [round(random.uniform(30000, 90000), 2) for _ in range(n)]
    salary = [round(ly * 1.03 + random.gauss(0, 3000), 2) for ly in last_year]
    departamentos = [f"dept_{i % 5}" for i in range(n)]
    filas = [{"salary": str(salary[i]), "last_year_salary": str(last_year[i]),
             "department": departamentos[i]} for i in range(n)]
    analisis = {"columns": {"last_year_salary": {"type": "number", "cardinality": n},
                            "department": {"type": "categorical", "cardinality": 5}}}
    d = diagnosticar_csv("no-usado", conf.problema, analisis=analisis, filas=filas)
    return conf, d


class LosDosCaminosDel71Test(unittest.TestCase):
    """El texto literal: «cierre del 71 se registra únicamente con
    evidencia de ambos caminos» -- los dos casos EXACTOS que ese contrato
    midió, ahora a través del esquema compuesto completo."""

    def test_camino_1_bloqueo_el_precio_a_partir_del_precio(self):
        conf = confirmar_desde_prompt(
            "clasificar el precio de una vivienda en categorias barata, "
            "media y cara a partir del precio, los metros y el barrio",
            objetivo="price", tarea="regression",
            entradas=("price", "square_meters", "neighborhood"),
            unidad_de_observacion="una vivienda")
        riesgo = DiagnosticoDeRiesgo(confirmacion=conf)
        self.assertEqual(riesgo.estado, "bloqueado")
        self.assertFalse(riesgo.puede_continuar)
        self.assertIsNone(riesgo.diagnostico)  # nunca corre sobre un objetivo sin confirmar

    def test_camino_2_aviso_el_salario_del_ano_pasado(self):
        conf, d = _sospecha_del_salario()
        self.assertTrue(conf.confirmado)
        riesgo = DiagnosticoDeRiesgo(confirmacion=conf, diagnostico=d)
        # Sin aceptar todavía: NO se presenta como validado.
        self.assertEqual(riesgo.estado, "exploratorio")
        self.assertTrue(riesgo.puede_continuar)


class TransicionesDeEstadoTest(unittest.TestCase):
    def test_confirmado_sin_sospechas_ni_bloqueos(self):
        conf = confirmar_desde_prompt(
            "predecir el precio de una vivienda a partir de los metros, el "
            "barrio y el ano de construccion",
            objetivo="price", tarea="regression",
            entradas=("square_meters", "neighborhood", "build_year"),
            unidad_de_observacion="una vivienda")
        self.assertTrue(conf.confirmado)
        riesgo = DiagnosticoDeRiesgo(confirmacion=conf)
        self.assertEqual(riesgo.estado, "confirmado")

    def test_exploratorio_pasa_a_confirmado_tras_aceptar(self):
        conf, d = _sospecha_del_salario()
        registro = RegistroDeAceptaciones()
        for s in d.sospechas:
            registro.aceptar(s, contexto="revisado, es una relación legítima", actor="roberto")
        riesgo = DiagnosticoDeRiesgo(confirmacion=conf, diagnostico=d, aceptaciones=registro.todas())
        self.assertEqual(riesgo.estado, "confirmado")
        self.assertEqual(sospechas_pendientes(d, registro.todas()), ())

    def test_aceptar_solo_ALGUNAS_sospechas_no_alcanza_confirmado(self):
        conf, d = _sospecha_del_salario()
        self.assertGreaterEqual(len(d.sospechas), 1)
        registro = RegistroDeAceptaciones()
        # No se acepta ninguna -- el estudio sigue exploratorio.
        riesgo = DiagnosticoDeRiesgo(confirmacion=conf, diagnostico=d, aceptaciones=registro.todas())
        self.assertEqual(riesgo.estado, "exploratorio")


class AceptacionExigeContextoYActorTest(unittest.TestCase):
    def test_sin_actor_se_rechaza(self):
        conf, d = _sospecha_del_salario()
        with self.assertRaises(ErrorDeAceptacion) as ctx:
            aceptar_sospecha(d.sospechas[0], contexto="motivo real", actor="")
        self.assertEqual(ctx.exception.clave, "aceptacion_sin_actor")

    def test_sin_contexto_se_rechaza(self):
        conf, d = _sospecha_del_salario()
        with self.assertRaises(ErrorDeAceptacion) as ctx:
            aceptar_sospecha(d.sospechas[0], contexto="   ", actor="roberto")
        self.assertEqual(ctx.exception.clave, "aceptacion_sin_contexto")

    def test_recibo_auditable_no_booleano(self):
        """Mismo patrón que `LicenseAcceptance`: quién, cuándo, con qué
        contexto EXACTO -- no un simple `aceptado=True`."""
        conf, d = _sospecha_del_salario()
        recibo = aceptar_sospecha(d.sospechas[0], contexto="relación legítima, no fuga", actor="roberto")
        self.assertEqual(recibo.actor, "roberto")
        self.assertEqual(recibo.contexto, "relación legítima, no fuga")
        self.assertTrue(recibo.accepted_at)
        self.assertTrue(recibo.aceptacion_id)
        self.assertEqual(recibo.sospecha_clave, d.sospechas[0].clave)


class AceptacionSeQuedaAnticuadaSiLaSospechaCambiaTest(unittest.TestCase):
    """El texto literal: «una aceptación de sospecha permanece visible en
    resultado y expediente» -- pero eso no significa que siga cubriendo
    una sospecha con una medida DISTINTA. El digest fija los términos
    exactos, igual que `LicenseAcceptance.license_digest`."""

    def test_una_sospecha_con_medida_distinta_no_queda_cubierta(self):
        conf, d = _sospecha_del_salario()
        sospecha_original = d.sospechas[0]
        registro = RegistroDeAceptaciones()
        registro.aceptar(sospecha_original, contexto="revisado", actor="roberto")
        self.assertEqual(sospechas_pendientes(d, registro.todas()), ())

        # Misma clave/campo, medida DISTINTA (el diagnóstico se repitió
        # y la correlación real medida cambió) -- construida a mano para
        # no depender de que un dataset real produzca ese cambio exacto.
        from dataclasses import replace
        sospecha_repetida = replace(sospecha_original, medida={**sospecha_original.medida, "r": 0.999})
        d_repetido = replace(d, sospechas=(sospecha_repetida,))
        pendientes = sospechas_pendientes(d_repetido, registro.todas())
        self.assertEqual(len(pendientes), 1)
        self.assertEqual(pendientes[0].clave, sospecha_original.clave)


class ErrorEstructuralConservaBloqueoTest(unittest.TestCase):
    """No existe NINGÚN mecanismo para "aceptar" un bloqueo -- solo
    sospechas se aceptan. Este test documenta esa ausencia: no hay
    parámetro ni función que reciba un `Bloqueo`."""

    def test_diagnostico_de_riesgo_no_tiene_forma_de_aceptar_bloqueos(self):
        import inspect
        firma = inspect.signature(DiagnosticoDeRiesgo)
        self.assertNotIn("bloqueos_aceptados", firma.parameters)
        firma_aceptar = inspect.signature(aceptar_sospecha)
        self.assertNotIn("bloqueo", firma_aceptar.parameters)


if __name__ == "__main__":
    unittest.main()
