#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""150-C7 — corre las pruebas de los Excel REALES EXIGIÉNDOLAS: una saltada es un ROJO (contrato 150, §6.2).

    PYTHONPATH=<rueda de python-calamine> python3 scripts/exigir_excel_reales.py

Pone MATRIXAI_EXIGIR_TABLAS=1 y MATRIXAI_EXIGIR_REALES=1 (faltar el lector o un libro FALLA) y además cuenta las
saltadas del informe: si hay alguna, sale con código 1 aunque pytest diga verde. Los xfail (hallazgos abiertos con su
reproducción) no son saltadas y se listan.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

RAIZ = pathlib.Path(__file__).resolve().parent.parent
env = dict(os.environ, MATRIXAI_EXIGIR_TABLAS="1", MATRIXAI_EXIGIR_REALES="1")
with tempfile.TemporaryDirectory() as t:
    xml = pathlib.Path(t) / "r.xml"
    rc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_150_c7_excel_reales.py", "-q", "-rsx", "-p", "no:cacheprovider",
                         f"--junitxml={xml}"], cwd=RAIZ, env=env).returncode
    raiz = ET.parse(xml).getroot()
    suites = raiz.findall("testsuite") if raiz.tag == "testsuites" else [raiz]
    saltadas = sum(int(s.get("skipped", 0)) for s in suites)
    casos = sum(int(s.get("tests", 0)) for s in suites)
print(f"pruebas: {casos}  saltadas (incluye xfail): {saltadas}")
saltadas_de_verdad = [c.get("name") for s in suites for c in s.iter("testcase")
                      if (sk := c.find("skipped")) is not None and sk.get("type") != "pytest.xfail"]
if saltadas_de_verdad:
    print("SALTADAS (rojo):", *saltadas_de_verdad, sep="\n  ")
    sys.exit(1)
sys.exit(rc)
