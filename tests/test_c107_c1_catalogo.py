# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — el catálogo medido de proveedores de embeddings.

La regla de esta casa: **la tabla se mide, no se escribe**. `TERCEROS.md` ya
se gobierna así y ha cazado dos fallos en un día. Esta tabla igual, y lo que
la sostiene es esto:

  · los digests y tamaños del artefacto se RECALCULAN aquí con el mismo
    código que los escribió, sobre los ficheros que hay en la caché;
  · los tiempos no se pueden reverificar (dependen de la máquina), así que lo
    que se exige es su COHERENCIA INTERNA: un número tecleado a mano rompe la
    aritmética que lo une con los demás;
  · cada fila lleva el método con el que se midió, el entorno, y —cuando no se
    pudo medir— el motivo. Un hueco sin motivo es un fallo, no un hueco.

Estas pruebas usan las MISMAS funciones que `scripts/medir_catalogo_embeddings.py`,
para que no haya dos formas de decir lo mismo.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from matrixai.text.embeddings import medicion
from matrixai.text.embeddings.catalogo import CANDIDATOS, Candidato, por_id
from matrixai.text.embeddings.descarga import DescargaError, raiz_cache, verificar_fichero

ARTEFACTO = Path(__file__).resolve().parent.parent / "matrixai" / "text" / "embeddings" / "catalogo_medido.json"


def _artefacto() -> dict:
    return json.loads(ARTEFACTO.read_text(encoding="utf-8"))


class TestElArtefactoExisteYCubreLaLista(unittest.TestCase):
    def test_hay_artefacto(self):
        self.assertTrue(ARTEFACTO.is_file(), f"falta {ARTEFACTO}; genéralo con scripts/medir_catalogo_embeddings.py")

    def test_todo_candidato_tiene_su_fila(self):
        datos = _artefacto()
        faltan = [c.id for c in CANDIDATOS if c.id not in datos["proveedores"]]
        self.assertEqual(faltan, [], f"candidatos sin fila en la tabla: {faltan}")

    def test_no_sobra_ninguna_fila(self):
        datos = _artefacto()
        conocidos = {c.id for c in CANDIDATOS}
        sobran = [k for k in datos["proveedores"] if k not in conocidos]
        self.assertEqual(sobran, [], f"filas de candidatos que ya no existen: {sobran}")

    def test_las_DOS_familias_estan_representadas(self):
        # El corte pide dos familias, no una con adornos.
        familias = {c.familia for c in CANDIDATOS}
        self.assertEqual(familias, {"estatica", "onnx_frase"})
        for familia in familias:
            self.assertGreaterEqual(len([c for c in CANDIDATOS if c.familia == familia]), 2, familia)


class TestElNucleoSigueSiendoLigero(unittest.TestCase):
    """102, invariante 1: `matrixai-core` se instala con `dependencies = []`.

    Por eso `numpy` y `onnxruntime` se importan DENTRO de las funciones que los
    necesitan y no arriba del módulo, y por eso el catálogo se puede leer en
    una máquina que no los tenga. Esa decisión está escrita en tres docstrings;
    sin esta prueba no la sostiene nada, y el siguiente que pase sube los
    imports al principio «para que se vea mejor».
    """

    def test_el_catalogo_se_lee_sin_numpy_ni_onnxruntime(self):
        import subprocess
        import sys

        guion = (
            "import sys\n"
            "class Veto:\n"
            "    def find_module(self, nombre, ruta=None):\n"
            "        return self\n"
            "    def find_spec(self, nombre, ruta=None, destino=None):\n"
            "        if nombre.split('.')[0] in ('numpy', 'onnxruntime'):\n"
            "            raise ImportError('vetado a proposito: ' + nombre)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Veto())\n"
            "from matrixai.text.embeddings import catalogo, descarga, medicion, cobertura, proveedor\n"
            "assert len(catalogo.CANDIDATOS) >= 4\n"
            "assert cobertura.veredicto(catalogo.CANDIDATOS[0].id, 'es').estado\n"
            "assert medicion.spdx_de('mit', repo='x') == 'MIT'\n"
            "assert descarga.host_permitido('huggingface.co')\n"
            "assert proveedor.elegir_longitud([], tope=99)[0] == 99\n"
            "try:\n"
            "    import numpy\n"
            "except ImportError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('el veto no funcionó: numpy se importó igual')\n"
            "print('ok')\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", guion],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(r.returncode, 0, f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("ok", r.stdout)


class TestLaLicenciaNoEsUnAdorno(unittest.TestCase):
    def test_cada_fila_trae_spdx_con_su_fuente_y_su_fecha(self):
        for cid, fila in _artefacto()["proveedores"].items():
            lic = fila.get("licencia")
            self.assertIsNotNone(lic, cid)
            for clave in ("spdx", "fuente", "leida_el", "tal_como_lo_declara_el_origen", "compatible"):
                self.assertIn(clave, lic, f"{cid}: la licencia no trae {clave}")
            self.assertIn(lic["spdx"], set(medicion.LICENCIAS_SPDX.values()), cid)
            self.assertTrue(lic["fuente"].startswith("https://"), cid)
            self.assertRegex(lic["leida_el"], r"^\d{4}-\d{2}-\d{2}T", cid)

    def test_una_licencia_incompatible_NO_esta_descargada(self):
        for cid, fila in _artefacto()["proveedores"].items():
            if not fila["licencia"]["compatible"]:
                self.assertFalse(fila.get("descargado"), f"{cid}: descargado con licencia incompatible")
                self.assertTrue(fila["licencia"]["motivo_si_no"], f"{cid}: incompatible sin motivo")

    def test_lo_descargado_lleva_su_recibo_de_aceptacion(self):
        for cid, fila in _artefacto()["proveedores"].items():
            if fila.get("descargado"):
                recibo = fila.get("aceptacion_de_licencia")
                self.assertIsNotNone(recibo, f"{cid}: descargado sin recibo de aceptación")
                self.assertEqual(recibo["spdx"], fila["licencia"]["spdx"], cid)

    def test_una_licencia_sin_declarar_no_se_supone(self):
        with self.assertRaises(medicion.MedicionError) as ctx:
            medicion.spdx_de(None, repo="alguien/algo")
        self.assertIn("no declara licencia", str(ctx.exception))

    def test_una_licencia_desconocida_para_en_vez_de_adivinar(self):
        with self.assertRaises(medicion.MedicionError) as ctx:
            medicion.spdx_de("licencia-rarisima-2.0", repo="alguien/algo")
        self.assertIn("sin equivalencia SPDX", str(ctx.exception))

    def test_la_traduccion_a_spdx_respeta_las_mayusculas_del_estandar(self):
        self.assertEqual(medicion.spdx_de("apache-2.0", repo="x"), "Apache-2.0")
        self.assertEqual(medicion.spdx_de("mit", repo="x"), "MIT")

    def test_hay_licencias_declaradas_incompatibles_y_dicen_por_que(self):
        self.assertTrue(medicion.INCOMPATIBLES)
        for spdx, motivo in medicion.INCOMPATIBLES.items():
            self.assertIn(spdx, set(medicion.LICENCIAS_SPDX.values()), spdx)
            self.assertTrue(motivo.strip(), spdx)


class TestLosNumerosSeMiden(unittest.TestCase):
    """Lo que hace que esta tabla caduque ruidosa."""

    def test_los_digests_del_artefacto_se_RECALCULAN_sobre_los_ficheros(self):
        datos = _artefacto()
        comprobados = 0
        for cid, fila in datos["proveedores"].items():
            if not fila.get("descargado"):
                continue
            d = raiz_cache() / cid
            for f in medicion.ficheros_fijados(fila):
                try:
                    verificar_fichero(d / f.ruta, f)
                except DescargaError as e:
                    self.fail(f"{cid}/{f.ruta}: {e}")
                comprobados += 1
        if comprobados == 0:
            self.skipTest("no hay proveedores descargados en esta máquina; nada que recalcular")

    def test_el_tamano_declarado_es_la_suma_de_sus_ficheros(self):
        for cid, fila in _artefacto()["proveedores"].items():
            if "ficheros" not in fila:
                continue
            suma = sum(f["tamano_bytes"] for f in fila["ficheros"])
            self.assertEqual(suma, fila["descarga_total_bytes"], cid)
            del_modelo = [f for f in fila["ficheros"] if f["ruta"] == por_id(cid).modelo]
            self.assertEqual(len(del_modelo), 1, cid)
            self.assertEqual(del_modelo[0]["tamano_bytes"], fila["pesos_bytes"], cid)

    def test_los_tiempos_cuadran_entre_si(self):
        """Un tiempo tecleado a mano rompe la aritmética que lo ata al resto."""
        vistos = 0
        for cid, fila in _artefacto()["proveedores"].items():
            m = fila.get("medicion") or {}
            for idioma, t in (m.get("tiempo") or {}).items():
                vistos += 1
                self.assertLessEqual(t["segundos_min"], t["segundos_mediana"], f"{cid}/{idioma}")
                self.assertLessEqual(t["segundos_mediana"], t["segundos_max"], f"{cid}/{idioma}")
                # La mediana se guarda redondeada a 4 decimales, así que los
                # derivados solo pueden cuadrar DENTRO de ese redondeo: la
                # tolerancia se calcula de ahí, no se afloja a ojo. Con ella,
                # un número tecleado a mano sigue rompiendo la aritmética.
                mediana = t["segundos_mediana"]
                error_rel = 0.00005 / mediana
                esperado = mediana * 1000.0 / t["n_textos"]
                self.assertAlmostEqual(t["segundos_por_1000_textos"], esperado, places=3, msg=f"{cid}/{idioma}")
                por_segundo = t["n_textos"] / mediana
                self.assertLessEqual(
                    abs(t["textos_por_segundo"] - por_segundo),
                    por_segundo * error_rel + 0.05,
                    f"{cid}/{idioma}: textos_por_segundo {t['textos_por_segundo']} no sale de la mediana {mediana}",
                )
                self.assertGreaterEqual(t["repeticiones"], 3, f"{cid}/{idioma}: muy pocas repeticiones")
        if vistos == 0:
            self.skipTest("ninguna fila trae tiempos todavía")

    def test_el_corte_pide_MIL_textos_y_se_miden_mil(self):
        vistos = 0
        for cid, fila in _artefacto()["proveedores"].items():
            for idioma, t in ((fila.get("medicion") or {}).get("tiempo") or {}).items():
                vistos += 1
                self.assertEqual(t["n_textos"], 1000, f"{cid}/{idioma}")
        if vistos == 0:
            self.skipTest("ninguna fila trae tiempos todavía")

    def test_el_instrumento_esta_comprobado_antes_de_dar_el_numero(self):
        """Un cronómetro que midiese el arranque en vez del trabajo daría una
        tabla preciosa y falsa. El control mide con varios tamaños y exige que
        el segundos-por-1.000 se mantenga."""
        vistos = 0
        for cid, fila in _artefacto()["proveedores"].items():
            control = (fila.get("medicion") or {}).get("control_del_instrumento")
            if control is None:
                continue
            vistos += 1
            self.assertTrue(control["lineal"], f"{cid}: el cronómetro no escala con el trabajo: {control}")
            self.assertGreaterEqual(len(control["tamanos"]), 3, cid)
        if vistos == 0:
            self.skipTest("ninguna fila trae medición todavía")

    def test_el_control_CAZA_un_instrumento_que_mide_el_arranque(self):
        """Y comprobar el control, no solo tenerlo.

        Un `control_del_instrumento` que aprobara siempre sería peor que no
        tenerlo: su presencia hace pensar que alguien mira. Aquí se le pone
        delante un instrumento que tarda casi lo mismo con 250 textos que con
        1.000 —exactamente lo que mediría un cronómetro puesto alrededor del
        arranque— y se exige que lo suspenda.
        """
        import time

        class ArranqueCaro:
            def codificar(self, textos, lote=32):
                time.sleep(0.05 + 0.000004 * len(textos))
                return None

        r = medicion.control_de_linealidad(ArranqueCaro(), ["x"] * 1000)
        self.assertFalse(r["lineal"], f"el control aprobó un instrumento no lineal: {r}")
        self.assertGreater(r["dispersion_relativa"], 0.35)

    def test_y_aprueba_uno_que_SI_escala_con_el_trabajo(self):
        """La otra mitad: un control que suspendiera a todo el mundo tampoco
        vale. Un instrumento de coste proporcional tiene que pasarlo."""
        import time

        class CosteProporcional:
            def codificar(self, textos, lote=32):
                time.sleep(0.0004 * len(textos))
                return None

        r = medicion.control_de_linealidad(CosteProporcional(), ["x"] * 1000)
        self.assertTrue(r["lineal"], f"el control suspendió a un instrumento lineal: {r}")

    def test_cada_medicion_dice_con_que_maquina_Y_CON_QUE_CARGA_se_hizo(self):
        """El entorno va por FILA, no una vez para todo el artefacto: cada
        proveedor se mide en un momento distinto y la máquina no está igual de
        tranquila. Un entorno global le pondría a los primeros la carga del
        último — declarar lo que se pidió en vez de lo que pasó."""
        vistos = 0
        for cid, fila in _artefacto()["proveedores"].items():
            m = fila.get("medicion") or {}
            if not m.get("tiempo"):
                continue
            vistos += 1
            entorno = m.get("entorno")
            self.assertIsNotNone(entorno, f"{cid}: hay tiempos y no se dice en qué máquina")
            for clave in ("cpu", "nucleos", "onnxruntime", "numpy", "python", "medido_el", "carga_al_empezar"):
                self.assertIn(clave, entorno, f"{cid}: el entorno no trae {clave}")
            self.assertIn("carga_al_terminar", entorno, f"{cid}: no se dice cómo acabó la máquina")
        if vistos == 0:
            self.skipTest("todavía no hay tiempos")

    def test_una_medicion_hecha_con_dependencia_de_fuera_lo_DICE(self):
        """Un número que solo se obtiene instalando algo que el núcleo no
        lleva no es el mismo número, y el catálogo tiene que decirlo."""
        for cid, fila in _artefacto()["proveedores"].items():
            m = fila.get("medicion") or {}
            if not m.get("tiempo"):
                continue
            self.assertIn("medido_con_tokenizador_de_fuera_del_nucleo", m, cid)
            if m["medido_con_tokenizador_de_fuera_del_nucleo"]:
                self.assertIn("tokenizers", m["que_significa_eso"], cid)


class TestElCriterioDeTerminadoDelCorte(unittest.TestCase):
    """«Tabla medida; **uno de cada familia funcionando con onnxruntime sobre
    1.000 textos en es y en**; decisión de cuáles entran como `- [?]`.»

    Literal, y por familia: una tabla con las dos familias listadas pero con
    una sola ejecutada no cumple el corte, y desde fuera las dos cosas se
    parecen mucho.
    """

    def _medidos_de(self, familia: str) -> list[str]:
        salida = []
        for cid, fila in _artefacto()["proveedores"].items():
            if fila.get("familia") != familia:
                continue
            t = (fila.get("medicion") or {}).get("tiempo") or {}
            if t.get("es", {}).get("n_textos") == 1000 and t.get("en", {}).get("n_textos") == 1000:
                salida.append(cid)
        return salida

    def test_hay_al_menos_un_ESTATICO_corrido_sobre_1000_textos_en_es_y_en(self):
        self.assertTrue(self._medidos_de("estatica"), "ningún proveedor estático ejecutado")

    def test_hay_al_menos_un_ONNX_DE_FRASE_corrido_sobre_1000_textos_en_es_y_en(self):
        self.assertTrue(self._medidos_de("onnx_frase"), "ningún transformer de frase ejecutado")

    def test_lo_ejecutado_lo_ejecuto_onnxruntime(self):
        # El corte dice «con onnxruntime», no «con lo que sea». Cada fila
        # medida guarda la versión del runtime con el que corrió.
        for familia in ("estatica", "onnx_frase"):
            for cid in self._medidos_de(familia):
                m = _artefacto()["proveedores"][cid]["medicion"]
                self.assertIn("onnxruntime", m["entorno"], cid)
                self.assertRegex(m["entorno"]["onnxruntime"], r"^\d+\.\d+", cid)
                self.assertIn(m["describe"]["forma_del_grafo"], ("bolsa", "secuencia"), cid)


class TestLoQueNoSePudoMedirLoDice(unittest.TestCase):
    def test_todo_candidato_no_descargado_tiene_motivo(self):
        for c in CANDIDATOS:
            fila = _artefacto()["proveedores"].get(c.id, {})
            if fila.get("descargado"):
                continue
            motivo = c.no_descargado_porque or (fila.get("medicion") or {}).get("no_medido_porque")
            self.assertTrue(motivo, f"{c.id}: ni descargado ni con motivo de por qué no")

    def test_un_proveedor_sin_medir_lo_dice_en_vez_de_dejar_el_hueco(self):
        for cid, fila in _artefacto()["proveedores"].items():
            m = fila.get("medicion")
            if m is None:
                self.assertTrue(
                    por_id(cid).no_descargado_porque,
                    f"{cid}: sin medición y sin explicar por qué",
                )
            elif "no_medido_porque" in m:
                self.assertTrue(m["no_medido_porque"].strip(), cid)


class TestLaCoberturaDeIdiomaSeMideNoSeCita(unittest.TestCase):
    def test_lo_que_dice_el_autor_va_marcado_como_CITA_con_su_url(self):
        for c in CANDIDATOS:
            self.assertTrue(c.idiomas_segun_su_autor.strip(), c.id)
            self.assertTrue(c.fuente_de_esa_cita.startswith("https://"), c.id)

    def test_toda_fila_medida_trae_es_Y_en(self):
        vistos = 0
        for cid, fila in _artefacto()["proveedores"].items():
            m = fila.get("medicion") or {}
            if "tokenizacion" not in m:
                continue
            vistos += 1
            for bloque in ("tokenizacion", "coherencia_por_documento", "tiempo"):
                self.assertEqual(sorted(m[bloque]), ["en", "es"], f"{cid}/{bloque}")
        if vistos == 0:
            self.skipTest("ninguna fila medida todavía")

    def test_cada_medicion_de_idioma_dice_COMO_se_midio(self):
        vistos = 0
        for cid, fila in _artefacto()["proveedores"].items():
            m = fila.get("medicion") or {}
            if "metodo" not in m:
                continue
            vistos += 1
            self.assertIn("idiomas", m["metodo"], cid)
            self.assertIn("tiempo", m["metodo"], cid)
            self.assertGreater(len(m["metodo"]["idiomas"]), 80, f"{cid}: el método no explica nada")
        if vistos == 0:
            self.skipTest("ninguna fila medida todavía")

    def test_el_corpus_de_medida_esta_fijado_por_su_digest(self):
        datos = _artefacto()
        if "corpus_de_medida" not in datos:
            self.skipTest("todavía no se ha medido nada")
        c = datos["corpus_de_medida"]
        self.assertRegex(c["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(c["url"].startswith("https://"))
        self.assertTrue(c["licencia"])
        self.assertEqual(c["n_textos_por_idioma"], 1000)


class TestLasDecisionesDelCatalogoSonExplicitas(unittest.TestCase):
    def test_toda_revision_es_un_commit_completo(self):
        for c in CANDIDATOS:
            self.assertRegex(c.revision, r"^[0-9a-f]{40}$", c.id)

    def test_todo_candidato_dice_por_que_esta_en_la_lista(self):
        for c in CANDIDATOS:
            self.assertGreater(len(c.motivo), 40, f"{c.id}: sin motivo de por qué es candidato")

    def test_el_modelo_esta_entre_los_ficheros_que_se_bajan(self):
        for c in CANDIDATOS:
            self.assertIn(c.modelo, c.ficheros, c.id)

    def test_una_revision_de_rama_no_se_admite(self):
        with self.assertRaises(ValueError):
            Candidato(
                id="x",
                familia="estatica",
                repo="a/b",
                revision="main",
                ficheros=("m.onnx",),
                modelo="m.onnx",
                tokenizador="wordpiece",
                tokens_especiales=False,
                idiomas_segun_su_autor="x",
                fuente_de_esa_cita="https://x",
                motivo="y" * 50,
            )

    def test_la_longitud_efectiva_de_la_libreria_va_con_su_fuente(self):
        """Cuando un candidato declara una longitud que no está en ningún
        fichero del paquete, tiene que decir de dónde sale."""
        for c in CANDIDATOS:
            if c.longitud_efectiva_de_su_libreria is not None:
                self.assertTrue(
                    c.fuente_de_la_longitud_efectiva.strip(),
                    f"{c.id}: declara longitud efectiva sin decir de dónde sale",
                )


if __name__ == "__main__":
    unittest.main()
