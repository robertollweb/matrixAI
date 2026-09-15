# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""109-C3 — EL EMPAQUETADOR ESCRIBE EL EXPEDIENTE CLÍNICO (el hueco de cableado).

EL HUECO, tal cual lo dejó 109-C3. `matrixai/export/expediente_clinico.py` ya
sabe LEER `clinical_profile.json` y `team_declaration.json`, y con ellos
`tripod.py`/`probast.py` componen la ficha y los huecos — está hecho, probado
y commiteado. Pero `matrixai.export.bundle.EdgeBundler.bundle()` (el
empaquetador de VERDAD, el que llaman `matrixai/cli.py` y el backend del
Studio) no tenía ni un parámetro para ninguno de los dos: el lector estaba
probado sobre un fichero que ningún paquete real llegaba a traer nunca. Este
fichero prueba el CABLEADO que cierra ese hueco: `EdgeBundler.bundle()` /
`create_edge_bundle()` aceptan ahora `clinical_profile=` (un `PerfilClinico`
de 109-C2, o su `a_json()`) y `team_declaration=` (lo que declaró una
persona), y escriben los dos ficheros SOLO cuando hay algo que escribir.

NO FABRICAR, LAS DOS DIRECCIONES.
* Sin perfil clínico, `clinical_profile.json` no viaja — ni vacío ni a medias.
* Sin declaración del equipo, `team_declaration.json` no viaja.
* Lo que SÍ viaja, viaja TAL CUAL: el mismo `PerfilClinico.a_json()` que
  produciría 109-C2, sin recomponerlo aquí — dos sitios recomponiendo el
  mismo sobre acaban divergiendo, que es la nota que más veces se repite en
  este repositorio.

LA PRUEBA QUE DE VERDAD VALE, al final del fichero: construir un paquete real
(un `.mxai` real de los ejemplos, un `PerfilClinico` real del mismo código de
109-C2) y comprobar que `matrixai report --probast` sobre ESE paquete —el
CLI, en su propio proceso, no la función— deja de decir «falta» en las líneas
que ahora sí tiene. Probar la función no es probar el producto.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_BASE = Path(__file__).parent.parent
_FALL_RISK_MXAI = _BASE / "examples" / "fall-risk.mxai"


def _onnx_available() -> bool:
    from importlib import util
    return util.find_spec("onnx") is not None


def _ort_available() -> bool:
    from importlib import util
    return util.find_spec("onnxruntime") is not None


_ONNX_LISTO = _onnx_available() and _ort_available()


# ---------------------------------------------------------------------------
# Un perfil clínico DE VERDAD, del mismo código que corre en producción
# (109-C2). Escribir el JSON a mano probaría el artefacto y no el código que
# lo produce — la misma trampa que documenta test_c109_c3_expediente.py.
# ---------------------------------------------------------------------------

def _muestra(n: int = 400, semilla: int = 20260915):
    from matrixai.estudio.metricas import Muestra
    rnd = random.Random(semilla)
    y, probs = [], []
    for _ in range(n):
        real = rnd.random() < 0.22
        p = min(0.999, max(0.001, rnd.gauss(0.72 if real else 0.22, 0.18)))
        y.append("caida" if real else "sin_caida")
        probs.append((1.0 - p, p))
    return Muestra(task="binary_classification", y_true=tuple(y),
                   classes=("sin_caida", "caida"), positive_label="caida",
                   probabilities=tuple(probs))


def _perfil_de_verdad():
    from matrixai.estudio.calibracion import curva_de_fiabilidad
    from matrixai.estudio.perfil_clinico import (
        PerfilClinico, curva_de_decision, tabla_de_umbrales,
    )
    from matrixai.estudio.segmentos import analizar_segmento

    muestra = _muestra()
    umbrales = (0.3, 0.5, 0.8)
    tabla = tabla_de_umbrales(
        muestra, umbrales=umbrales, diseno="iid",
        estimando="fixed_model_on_population", semilla=11, prevalencia=0.12,
        declarados_por="Comité de caídas, Hospital X", remuestras=40)
    segmento = analizar_segmento(
        "sensitivity", muestra, tuple(range(180)),
        segmento_id="mayores_de_80", predefinido=True)
    return PerfilClinico(
        perfil_id="perfil-c109-c3-bundle", evidencia="independent_test",
        diseno="iid", datos_sinteticos=True, tabla=tabla,
        curva=curva_de_decision(muestra, umbrales_de_probabilidad=umbrales),
        calibracion=curva_de_fiabilidad(muestra, n_bins=5),
        segmentos=(segmento,), segmentos_predefinidos_por="Comité de caídas",
        politica_de_faltantes={"estrategia": "descartar_fila",
                               "declarada_por": "Comité de caídas"})


#: Lo que declararía un equipo. Nada de umbrales aquí: los umbrales y quién
#: los declaró viven SOLO en el perfil (109-C2) — ver
#: test_c109_c3_expediente.py::test_los_umbrales_los_manda_el_PERFIL_y_nadie_mas.
_DECLARACION = {
    "uso_previsto": "cribado de riesgo de caída en planta de geriatría",
    "poblacion": "adultos >= 65 años ingresados en planta de geriatría del "
                 "Hospital X",
    "criterios_de_inclusion": ["edad >= 65", "ingreso en planta de geriatría"],
    "definicion_del_desenlace": "caída registrada durante el ingreso",
    "momento_del_desenlace": "en cualquier momento del ingreso",
    "proceso_actual": "escala de Downton aplicada a mano por enfermería",
    "declarado_por": "Comité de caídas, Hospital X",
}


# ---------------------------------------------------------------------------
# Un paquete DE VERDAD: el mismo `create_edge_bundle` que llaman `cli.py` y
# el backend del Studio, sobre un `.mxai` real (fall-risk.mxai).
# ---------------------------------------------------------------------------

def _programa_y_parametros():
    from matrixai.parameters import build_initial_parameter_set
    from matrixai.parser import parse_file
    prog = parse_file(_FALL_RISK_MXAI)
    return prog, build_initial_parameter_set(prog)


def _construir_paquete(destino: Path, *, con_expediente: bool):
    from matrixai.export import create_edge_bundle
    from matrixai.parameters import write_parameter_set

    prog, ps = _programa_y_parametros()
    params_path = destino.parent / f"{destino.name}.params.json"
    write_parameter_set(str(params_path), ps)
    kwargs = {}
    if con_expediente:
        kwargs["clinical_profile"] = _perfil_de_verdad()
        kwargs["team_declaration"] = dict(_DECLARACION)
    resultado = create_edge_bundle(
        prog, ps, mxai_path=str(_FALL_RISK_MXAI), params_path=str(params_path),
        outdir=str(destino), validate=True, **kwargs)
    return resultado


def _correr_report(bundle_dir: Path, *, locale: str = "es"
                   ) -> subprocess.CompletedProcess:
    """El CLI de VERDAD, en su propio proceso — no la función interna."""
    return subprocess.run(
        [sys.executable, "-m", "matrixai.cli", "report", "--probast",
         str(bundle_dir), "--locale", locale],
        capture_output=True, text=True, timeout=180, cwd=str(_BASE))


def _cuenta_del_resumen(stdout: str) -> dict[str, int]:
    """Las tres líneas finales «- <rótulo>: N» del resumen de `probast.py`.

    Son las ÚNICAS líneas que empiezan por «- » y terminan en un entero
    pelado: las líneas de un campo siempre terminan en `` `ruta` ``.
    """
    cuenta: dict[str, int] = {}
    for linea in stdout.splitlines():
        if not linea.startswith("- ") or ":" not in linea:
            continue
        rotulo, _, numero = linea[2:].rpartition(":")
        numero = numero.strip()
        if numero.isdigit():
            cuenta[rotulo.strip()] = int(numero)
    return cuenta


def _linea_de(stdout: str, rotulo: str) -> str:
    for linea in stdout.splitlines():
        if f"**{rotulo}**" in linea:
            return linea
    raise AssertionError(f"no aparece ninguna línea para «{rotulo}» en:\n{stdout}")


# ---------------------------------------------------------------------------
# Unitarias: las dos funciones que deciden qué se escribe, sin bundle
# completo alrededor (rápidas, y con el nombre exacto de cada invariante).
# ---------------------------------------------------------------------------

class NoFabricarTest(unittest.TestCase):
    def test_sin_perfil_no_hay_bloque(self):
        from matrixai.export.bundle import build_clinical_profile_block
        self.assertIsNone(build_clinical_profile_block(None))

    def test_perfil_con_esquema_equivocado_revienta(self):
        from matrixai.export.bundle import EdgeBundleError, build_clinical_profile_block
        with self.assertRaises(EdgeBundleError):
            build_clinical_profile_block({"schema": "otra.cosa", "x": 1})

    def test_perfil_que_no_es_ni_mapa_ni_objeto_revienta(self):
        from matrixai.export.bundle import EdgeBundleError, build_clinical_profile_block
        with self.assertRaises(EdgeBundleError):
            build_clinical_profile_block("no soy un perfil")

    def test_perfil_de_verdad_pasa_tal_cual(self):
        from matrixai.export.bundle import build_clinical_profile_block
        perfil = _perfil_de_verdad()
        self.assertEqual(build_clinical_profile_block(perfil), perfil.a_json())
        # Y el mapa ya producido (el camino que usaría quien solo tiene el
        # JSON, no el objeto) da EXACTAMENTE lo mismo.
        self.assertEqual(build_clinical_profile_block(perfil.a_json()),
                         perfil.a_json())

    def test_sin_declaracion_no_hay_bloque(self):
        from matrixai.export.bundle import build_team_declaration_block
        self.assertIsNone(build_team_declaration_block(None))
        self.assertIsNone(build_team_declaration_block({}))

    def test_declaracion_que_no_es_mapa_revienta(self):
        from matrixai.export.bundle import EdgeBundleError, build_team_declaration_block
        with self.assertRaises(EdgeBundleError):
            build_team_declaration_block(["uso_previsto", "x"])

    def test_declaracion_de_verdad_pasa_tal_cual(self):
        from matrixai.export.bundle import build_team_declaration_block
        self.assertEqual(build_team_declaration_block(_DECLARACION), _DECLARACION)


# ---------------------------------------------------------------------------
# Con un paquete de verdad alrededor.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_ONNX_LISTO, "onnx/onnxruntime no instalados")
class ElEmpaquetadorEscribeElExpedienteTest(unittest.TestCase):

    def test_sin_perfil_ni_declaracion_NINGUNO_de_los_dos_ficheros_viaja(self):
        with TemporaryDirectory() as d:
            bundle_dir = Path(d) / "bundle"
            resultado = _construir_paquete(bundle_dir, con_expediente=False)
            self.assertNotIn("clinical_profile.json", resultado.files)
            self.assertNotIn("team_declaration.json", resultado.files)
            self.assertFalse((bundle_dir / "clinical_profile.json").exists())
            self.assertFalse((bundle_dir / "team_declaration.json").exists())

    def test_con_perfil_los_DOS_ficheros_viajan_TAL_CUAL(self):
        with TemporaryDirectory() as d:
            bundle_dir = Path(d) / "bundle"
            perfil = _perfil_de_verdad()
            from matrixai.export import create_edge_bundle
            from matrixai.parameters import write_parameter_set
            prog, ps = _programa_y_parametros()
            params_path = Path(d) / "params.json"
            write_parameter_set(str(params_path), ps)
            resultado = create_edge_bundle(
                prog, ps, mxai_path=str(_FALL_RISK_MXAI),
                params_path=str(params_path), outdir=str(bundle_dir),
                validate=True, clinical_profile=perfil,
                team_declaration=_DECLARACION)

            self.assertIn("clinical_profile.json", resultado.files)
            self.assertIn("team_declaration.json", resultado.files)
            en_disco = json.loads(
                (bundle_dir / "clinical_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(en_disco, perfil.a_json())
            declarado_en_disco = json.loads(
                (bundle_dir / "team_declaration.json").read_text(encoding="utf-8"))
            self.assertEqual(declarado_en_disco, _DECLARACION)

    def test_report_probast_DEJA_DE_DECIR_FALTA_cuando_hay_expediente(self):
        """LA PRUEBA DE PUNTA A PUNTA: el CLI de verdad, sobre un paquete
        real con un perfil real, deja de decir «falta» en las líneas que
        109-C3 dijo que este cableado tenía que cerrar."""
        with TemporaryDirectory() as d:
            base = Path(d)
            sin = base / "sin_expediente"
            con = base / "con_expediente"
            _construir_paquete(sin, con_expediente=False)
            _construir_paquete(con, con_expediente=True)
            r_sin = _correr_report(sin)
            r_con = _correr_report(con)

        self.assertEqual(r_sin.returncode, 0, r_sin.stderr)
        self.assertEqual(r_con.returncode, 0, r_con.stderr)

        cuenta_sin = _cuenta_del_resumen(r_sin.stdout)
        cuenta_con = _cuenta_del_resumen(r_con.stdout)
        self.assertIn("falta", cuenta_sin)
        self.assertIn("falta", cuenta_con)
        # El recuento GLOBAL de huecos baja: no es que una línea cambie de
        # sitio, es que el paquete sostiene de verdad más de lo que sostenía.
        self.assertLess(cuenta_con["falta"], cuenta_sin["falta"],
                        f"sin={r_sin.stdout}\ncon={r_con.stdout}")

        # Y campo a campo: cada uno de estos, HOY (sin expediente) decía
        # «falta», y con el expediente puesto deja de decirlo.
        rotulos_que_tienen_que_dejar_de_faltar = (
            "Uso previsto",
            "Población destinataria",
            "Criterios de inclusión",
            "Definición del desenlace",
            "Momento del desenlace",
            "Comparación con el proceso actual",
            "¿Datos sintéticos?",
            "Prevalencia declarada (población destino)",
            "Prevalencia observada en la muestra",
            "Umbrales medidos",
            "Umbrales declarados por",
            "Calibración (ECE)",
            "Curva de decisión (beneficio neto)",
            "Subgrupos medidos",
            "Subgrupos predefinidos por",
            "Política de datos ausentes",
            "Alcance de la validación",
        )
        for rotulo in rotulos_que_tienen_que_dejar_de_faltar:
            with self.subTest(rotulo=rotulo):
                linea_antes = _linea_de(r_sin.stdout, rotulo)
                self.assertIn("_falta_", linea_antes, linea_antes)
                linea_despues = _linea_de(r_con.stdout, rotulo)
                self.assertNotIn("_falta_", linea_despues, linea_despues)

        # Y lo que 109-C3 dijo EXPLÍCITAMENTE que este cableado NO cierra
        # (105-C5, un corte distinto) sigue faltando en los dos: si esto
        # empezara a decir otra cosa, sería una afirmación fabricada, no un
        # arreglo.
        for r in (r_sin, r_con):
            linea = _linea_de(r.stdout, "Comparación con el baseline")
            self.assertIn("_falta_", linea, linea)


if __name__ == "__main__":
    unittest.main()
