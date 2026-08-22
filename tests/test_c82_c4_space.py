"""CONTRATO 82-C4 — la plantilla del Space, que es DEL USUARIO.

Corrección de Roberto (2026-08-19): *«El Space no es de MatrixAI, es de
cada usuario: su modelo, sus datos, su Space.»* Con eso el motivo para
posponerlo desaparece — no hay nada que sostener.

Este fichero cubre la **pieza 1**, la que no consume nada de nadie: la
plantilla viaja DENTRO del paquete. Publicar el Space es otra cosa y es
una casilla al publicar, nunca un efecto automático: crear repos en la
cuenta de alguien porque sí es justo lo que este proyecto evita en todo
lo demás.
"""

import unittest

from matrixai.export.space import space_app_py, space_readme_md


class LaPlantillaViajaEnElPaqueteTest(unittest.TestCase):
    def test_el_readme_lleva_el_front_matter_que_HF_entiende(self):
        texto = space_readme_md("mi_modelo")
        # Sin front-matter, Hugging Face no sabe que es un Space y lo
        # trata como un repo de ficheros: la plantilla no serviría.
        self.assertTrue(texto.startswith("---\n"))
        for clave in ("title:", "sdk:", "app_file:"):
            self.assertIn(clave, texto)
        self.assertIn("mi_modelo", texto)

    def test_la_app_es_python_valido(self):
        """Una plantilla que no compila no es una plantilla: es un fichero
        que alguien va a pegar en su Space para que falle."""
        compile(space_app_py(), "app.py", "exec")

    def test_la_app_usa_lo_que_el_paquete_YA_trae(self):
        codigo = space_app_py()
        # `predict.py` e `inference_spec.json` viajan en el bundle desde el
        # contrato EXPORT: la plantilla los usa en vez de reimplementar la
        # inferencia, que acabaría divergiendo.
        self.assertIn("predict", codigo)
        self.assertIn("inference_spec.json", codigo)

    def test_declara_sus_LIMITES_y_remite_al_paquete(self):
        """Un Space gratuito es CPU con tope de tiempo. Cuando no quepa,
        **lo dice y remite al paquete descargable** —el camino sin
        límites—, en vez de quedarse colgado o enseñar medio resultado."""
        codigo = space_app_py()
        texto = space_readme_md("m")
        junto = codigo + texto
        self.assertIn("CPU", junto)
        # Y nombra la salida: descargar el paquete y verificar en local.
        self.assertIn("matrixai verify", junto)

    def test_no_crea_nada_en_la_cuenta_de_nadie(self):
        """La plantilla es TEXTO. Si tuviera una llamada a `create_repo`,
        estaría creando recursos por su cuenta, que es lo que el contrato
        prohíbe explícitamente."""
        codigo = space_app_py()
        for prohibido in ("create_repo", "HfApi(", "upload_file", "login("):
            self.assertNotIn(prohibido, codigo)


if __name__ == "__main__":
    unittest.main()


class LaPlantillaLLEGAAlPaqueteTest(unittest.TestCase):
    """Que exista la función no sirve de nada si el ZIP no la lleva. Este
    proyecto lleva dieciséis huecos de cableado documentados."""

    def test_el_bundler_escribe_space_app_py_y_su_readme(self):
        import inspect
        from matrixai.export import bundle
        fuente = inspect.getsource(bundle)
        # Se comprueba sobre el código del bundler y no empaquetando un
        # ONNX entero a propósito: el bundle completo necesita
        # onnxruntime, y una prueba que se salta cuando falta no prueba
        # nada.
        self.assertIn("space_app_py()", fuente)
        self.assertIn("space_readme_md(", fuente)
        self.assertIn('"app.py"', fuente)

    def test_el_nombre_del_space_sale_del_modelo_no_de_una_constante(self):
        """Un Space llamado «matrixai-model» para todos los modelos del
        mundo no dice de cuál es."""
        import inspect
        from matrixai.export import bundle
        self.assertIn('getattr(program, "name"', inspect.getsource(bundle))


class LaPlantillaValeEnLosDosSitiosTest(unittest.TestCase):
    """Dentro del paquete vive en `space/` y los artefactos están arriba;
    publicada como Space, HF exige `app.py` en la RAÍZ y los artefactos
    quedan a su lado. **El mismo fichero para los dos casos**: dos
    versiones de esto acabarían divergiendo, y la que se ve en el Space no
    sería la que se probó."""

    def _ejecutar_en(self, carpeta_app, carpeta_artefactos):
        """Ejecuta el `_junto_a_mi` de la plantilla en un disco de mentira."""
        import tempfile
        from pathlib import Path
        raiz = Path(tempfile.mkdtemp())
        (raiz / carpeta_app).mkdir(parents=True, exist_ok=True)
        (raiz / carpeta_artefactos).mkdir(parents=True, exist_ok=True)
        (raiz / carpeta_artefactos / "predict.py").write_text("x = 1\n")
        (raiz / carpeta_artefactos / "inference_spec.json").write_text('{"inputs": []}')
        app = raiz / carpeta_app / "app.py"
        app.write_text(space_app_py(), encoding="utf-8")

        # Se ejecuta SOLO la parte de localización, sin gradio: importar la
        # app entera exigiría la dependencia y la prueba se saltaría — y una
        # prueba que se salta no prueba nada.
        codigo = space_app_py().split("import gradio")[0]
        entorno: dict = {"__file__": str(app)}
        exec(compile(codigo, str(app), "exec"), entorno)  # noqa: S102
        resto = space_app_py()
        inicio = resto.index("AQUI = ")
        fin = resto.index("# Un Space gratuito")
        exec(compile(resto[inicio:fin], str(app), "exec"), entorno)  # noqa: S102
        return entorno

    def test_dentro_del_paquete_encuentra_los_artefactos_arriba(self):
        e = self._ejecutar_en("space", ".")
        self.assertTrue(str(e["PREDICT"]).endswith("predict.py"))
        self.assertTrue(e["PREDICT"].is_file(), "no encontró predict.py un nivel arriba")

    def test_publicada_como_Space_los_encuentra_a_su_lado(self):
        e = self._ejecutar_en(".", ".")
        self.assertTrue(e["PREDICT"].is_file(), "no encontró predict.py a su lado")


class ElSpaceLLEVA_LoQueNecesitaParaARRANCARTest(unittest.TestCase):
    """Roberto, 2026-08-20: «lo he probado y no funciona».

    Midiéndolo salió que el Space subía con el `requirements.txt` del
    PAQUETE —`numpy` y `onnxruntime`, que es lo que hace falta para
    PREDECIR— y que le faltaban las dos cosas que solo el Space usa:

    * **`gradio`**, que `app.py` importa en su primera línea;
    * **`matrixai`**, que `app.py` ejecuta (`python -m matrixai verify`)
      para el bloque «Is this package intact?» — que es literalmente el
      criterio de cierre del 82-C4.

    Un botón que siempre contesta «no puedo comprobarlo» es peor que no
    tenerlo, porque se lee como que **el paquete** es el que falla.
    """

    def test_lleva_gradio_que_es_lo_que_app_py_importa(self):
        from matrixai.export.space import space_app_py, space_requirements_txt

        # El otro lado primero: si `app.py` no lo importara, pedirlo
        # sobraría — y esta prueba pasaría por el motivo equivocado.
        self.assertIn("import gradio", space_app_py())
        self.assertIn("gradio==", space_requirements_txt("1.5.0"))

    def test_lleva_matrixai_que_es_lo_que_app_py_EJECUTA(self):
        from matrixai.export.space import space_app_py, space_requirements_txt

        self.assertIn('"-m", "matrixai", "verify"', space_app_py())
        self.assertIn("matrixai==1.5.0", space_requirements_txt("1.5.0"))

    def test_sin_version_conocida_se_pide_SIN_fijar_en_vez_de_inventarla(self):
        from matrixai.export.space import space_requirements_txt

        requisitos = space_requirements_txt(None)
        self.assertIn("matrixai\n", requisitos)
        self.assertNotIn("matrixai==", requisitos)

    def test_el_front_matter_FIJA_la_version_del_sdk(self):
        """Sin `sdk_version`, Hugging Face elige la que quiera el día que
        se cree el Space: el mismo paquete publicado con seis meses de
        diferencia arrancaría sobre dos Gradio distintos, y el que falle
        lo haría sin que nadie hubiera tocado nada."""
        from matrixai.export.space import SDK_VERSION, space_readme_md

        front = space_readme_md("MiModelo").split("---")[1]
        self.assertIn(f"sdk_version: {SDK_VERSION}", front)

    def test_y_la_version_del_sdk_es_LA_MISMA_en_los_dos_sitios(self):
        """Dos sitios declarando lo mismo acaban divergiendo: el
        front-matter y los requisitos tienen que fijar la MISMA."""
        from matrixai.export.space import SDK_VERSION, space_readme_md, space_requirements_txt

        self.assertIn(f"sdk_version: {SDK_VERSION}", space_readme_md("M"))
        self.assertIn(f"gradio=={SDK_VERSION}", space_requirements_txt(None))

    def test_el_paquete_NO_arrastra_gradio_para_predecir_en_tu_maquina(self):
        """El otro lado, que es el que se rompe al apretar: los requisitos
        del Space NO son los del paquete. Quien se descargue el ZIP para
        predecir en su máquina no tiene por qué instalar gradio."""
        from matrixai.export.bundle import _REQUIREMENTS

        self.assertNotIn("gradio", _REQUIREMENTS)
        self.assertIn("onnxruntime", _REQUIREMENTS)
