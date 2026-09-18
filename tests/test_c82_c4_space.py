"""CONTRATO 82-C4 — la plantilla del Space, que es DEL USUARIO.

Corrección de Roberto (2026-08-19): *«El Space no es de MatrixAI, es de
cada usuario: su modelo, sus datos, su Space.»* Con eso el motivo para
posponerlo desaparece — no hay nada que sostener.

REVISIÓN 2026-09-16 — Space ESTÁTICO con ONNX Runtime Web, no Gradio.
Decisión de Roberto por lo medido el 2026-08-21: un Space de Gradio
devuelve 402 a cualquier cuenta sin HF PRO (público o privado), así que
ningún usuario gratuito podía llegar nunca a ver la demo; uno estático se
crea igual con una cuenta gratuita. Este fichero cubre la **pieza 1**, la
que no consume nada de nadie: la plantilla viaja DENTRO del paquete.
Publicar el Space es otra cosa y es una casilla al publicar, nunca un
efecto automático: crear repos en la cuenta de alguien porque sí es
justo lo que este proyecto evita en todo lo demás.
"""

import unittest

from matrixai.export.space import space_index_html, space_readme_md
from matrixai.export.wasm_exporter import ORT_WEB_MIN_VERSION, _build_predict_js


class LaPlantillaViajaEnElPaqueteTest(unittest.TestCase):
    def test_el_readme_lleva_el_front_matter_que_HF_entiende(self):
        texto = space_readme_md("mi_modelo")
        # Sin front-matter, Hugging Face no sabe que es un Space y lo
        # trata como un repo de ficheros: la plantilla no serviría.
        self.assertTrue(texto.startswith("---\n"))
        for clave in ("title:", "sdk:", "app_file:"):
            self.assertIn(clave, texto)
        self.assertIn("mi_modelo", texto)

    def test_el_front_matter_declara_sdk_static_y_app_file_index(self):
        """Confirmado localmente contra el `huggingface_hub` instalado:
        `constants.SPACES_SDK_TYPES` incluye `"static"`, y
        `SpaceCardData.app_file` es "Path to your main application file
        (... or static html code)"."""
        front = space_readme_md("M").split("---")[1]
        self.assertIn("sdk: static", front)
        self.assertIn("app_file: index.html", front)
        # Y NO gradio: ese SDK es justo lo que se ha dejado por el 402.
        self.assertNotIn("gradio", front.lower())

    def test_sin_sdk_version_que_ya_no_significa_nada(self):
        """`sdk_version` solo aplica a Gradio/Streamlit (documentado así en
        `huggingface_hub.repocard_data.SpaceCardData`): un Space estático
        no arranca ningún runtime, así que fijar una versión que nadie usa
        sería inventar precisión donde no la hay."""
        self.assertNotIn("sdk_version", space_readme_md("m"))

    def test_la_pagina_es_html_valido_minimo(self):
        """Una plantilla que no es HTML reconocible no es una plantilla: es
        un fichero que alguien va a publicar en su Space para que falle."""
        pagina = space_index_html("m")
        self.assertTrue(pagina.strip().startswith("<!doctype html>"))
        self.assertIn("<html", pagina)
        self.assertIn("</html>", pagina)

    def test_la_pagina_usa_lo_que_el_paquete_YA_trae(self):
        pagina = space_index_html("m")
        # `predict.js` (ORT Web) e `inference_spec.json` viajan en el
        # bundle desde el contrato EXPORT: la plantilla los usa en vez de
        # reimplementar la inferencia, que acabaría divergiendo.
        self.assertIn("predict.js", pagina)
        self.assertIn("inference_spec.json", pagina)
        self.assertIn("example_input.json", pagina)

    def test_no_crea_nada_en_la_cuenta_de_nadie(self):
        """La plantilla es TEXTO. Si tuviera una llamada a un cliente de
        Hugging Face, estaría creando recursos por su cuenta, que es lo
        que el contrato prohíbe explícitamente."""
        pagina = space_index_html("m")
        for prohibido in ("create_repo", "HfApi(", "upload_file", "huggingface_hub"):
            self.assertNotIn(prohibido, pagina)


if __name__ == "__main__":
    unittest.main()


class LaPlantillaLLEGAAlPaqueteTest(unittest.TestCase):
    """Que exista la función no sirve de nada si el ZIP no la lleva. Este
    proyecto lleva dieciséis huecos de cableado documentados."""

    def test_el_bundler_escribe_space_index_html_y_su_readme(self):
        import inspect
        from matrixai.export import bundle
        fuente = inspect.getsource(bundle)
        # Se comprueba sobre el código del bundler y no empaquetando un
        # ONNX entero a propósito: el bundle completo necesita
        # onnxruntime, y una prueba que se salta cuando falta no prueba
        # nada.
        self.assertIn("space_index_html(", fuente)
        self.assertIn("space_readme_md(", fuente)
        self.assertIn('"index.html"', fuente)

    def test_el_nombre_del_space_sale_del_modelo_no_de_una_constante(self):
        """Un Space llamado «matrixai-model» para todos los modelos del
        mundo no dice de cuál es."""
        import inspect
        from matrixai.export import bundle
        self.assertIn('getattr(program, "name"', inspect.getsource(bundle))

    def test_el_predict_js_del_space_es_EL_QUE_GENERA_wasm_exporter(self):
        """No una copia que puede divergir: el bundler tiene que llamar al
        MISMO generador que usa el bundle WASM, no reimplementar la parte
        que habla con ONNX Runtime por segunda vez."""
        import inspect
        from matrixai.export import bundle
        fuente = inspect.getsource(bundle)
        self.assertIn("_build_predict_js(", fuente)
        self.assertIn(
            "from matrixai.export.wasm_exporter import _build_predict_js",
            fuente,
        )
        # Y NO una escritura literal de otro contenido bajo ese nombre —
        # si el bundler escribiera su propio texto a `predict.js` en vez
        # de llamar al generador, esta prueba pasaría por el motivo
        # equivocado.
        i_predict_js = fuente.index('"predict.js"')
        alrededor = fuente[i_predict_js - 40:i_predict_js + 200]
        self.assertIn("_build_predict_js(program, export_result)", alrededor)


class ElSpaceNoDuplicaElModeloEnElZipTest(unittest.TestCase):
    """Diseño (82-C4 revisión 2026-09-16): `model.onnx` YA está en la raíz
    del paquete (para predecir en local); duplicarlo dentro de `space/`
    doblaría el tamaño del ZIP descargable sin necesidad — al publicar,
    `publish_hf._upload_space` aplana el paquete entero (menos `space/`) y
    ENCIMA el contenido de `space/`, así que `model.onnx` acaba junto a
    `index.html`/`predict.js` en el Space publicado sin haber viajado dos
    veces en el ZIP."""

    def test_el_bundler_NO_escribe_model_onnx_dentro_de_space(self):
        import inspect
        from matrixai.export import bundle
        fuente = inspect.getsource(bundle)
        self.assertNotIn('_space / "model.onnx"', fuente)

    def test_predict_js_pide_el_modelo_con_ruta_relativa_al_Space_aplanado(self):
        """`_build_predict_js` (reusado, sin tocar) pide `./model.onnx`:
        una ruta relativa a donde SE SIRVA `predict.js`. Al publicar,
        `_upload_space` deja `model.onnx` justo ahí (ver test en
        studio-backend); en el ZIP sin publicar `space/index.html` no lo
        encontrará junto a él, y la página lo dice en vez de fallar en
        silencio (ver `LaPaginaDeclaraLoQueHaceYLoQueNoTest`)."""
        # La cadena buscada es literal en el generador, independiente del
        # modelo — no hace falta construir un programa real para leerla.
        self.assertIn("'./model.onnx'", _fuente_predict_js())


def _fuente_predict_js() -> str:
    import inspect
    from matrixai.export import wasm_exporter
    return inspect.getsource(wasm_exporter._build_predict_js)


class LaPaginaDeclaraLoQueHaceYLoQueNoTest(unittest.TestCase):
    """82-C4 revisión 2026-09-16 [reemplaza a H9 de la auditoría del
    2026-08-20]: un Space ESTÁTICO no tiene Python ni proceso, así que YA
    NO puede ejecutar `matrixai verify` dentro. Callarlo sería media
    verdad tranquilizadora; la página lo DICE y da el comando en local —
    que es donde `verify` siempre pudo demostrar algo de verdad."""

    def _pagina(self):
        return space_index_html("m")

    def test_dice_que_corre_en_el_navegador_y_no_manda_los_datos(self):
        pagina = self._pagina()
        self.assertIn("in your browser", pagina)
        self.assertIn("never", pagina.lower())

    def test_dice_QUE_NO_verifica_y_por_que(self):
        pagina = self._pagina()
        self.assertIn("does NOT", pagina)
        self.assertIn("does not verify", pagina.lower())

    def test_da_el_comando_LOCAL_para_verificar_de_verdad(self):
        pagina = self._pagina()
        self.assertIn("matrixai verify .", pagina)
        self.assertIn("matrixai verify . --retrain", pagina)

    def test_NO_finge_ejecutar_verify_dentro_del_Space(self):
        """El defecto que reemplaza esta prueba: la plantilla anterior
        ejecutaba `python -m matrixai verify` vía `subprocess` DENTRO del
        Space. Un Space estático no tiene proceso — si esta cadena
        reapareciera, la página fingiría un botón que no puede
        funcionar."""
        pagina = self._pagina()
        self.assertNotIn("subprocess", pagina)
        self.assertNotIn('"-m", "matrixai", "verify"', pagina)

    def test_no_declara_limites_de_CPU_de_un_Space_que_ya_no_ejecuta_nada(self):
        """La plantilla de Gradio hablaba de "free CPU Space with a time
        limit" porque el servidor de HF ejecutaba Python. Un Space
        estático no ejecuta nada en el servidor — repetir esa frase sería
        describir un límite que ya no existe."""
        junto = space_readme_md("m") + self._pagina()
        self.assertNotIn("CPU Space", junto)
        self.assertNotIn("time limit", junto)


class LaVersionDeOrtWebEsLaMismaQueEnElBundleWasmTest(unittest.TestCase):
    """Dos sitios fijando la versión de ONNX Runtime Web acabarían
    divergiendo: la que carga la página del Space tiene que ser la MISMA
    que la que espera el `predict.js` que la propia página incluye."""

    def test_el_cdn_de_la_pagina_usa_ORT_WEB_MIN_VERSION(self):
        pagina = space_index_html("m")
        self.assertIn(f"onnxruntime-web@{ORT_WEB_MIN_VERSION}", pagina)

    def test_y_es_la_MISMA_constante_que_importa_wasm_exporter(self):
        import inspect
        from matrixai.export import space
        fuente = inspect.getsource(space)
        self.assertIn(
            "from matrixai.export.wasm_exporter import ORT_WEB_MIN_VERSION",
            fuente,
        )
        # Y NO una segunda constante local con el mismo valor a mano —
        # eso es justo lo que hace que dos sitios acaben divergiendo.
        self.assertNotIn('ORT_WEB_MIN_VERSION = "', fuente)


class ElNombreDelModeloSeEscapaEnLaPaginaTest(unittest.TestCase):
    """El nombre del modelo lo escribe el USUARIO y va dentro del HTML de un
    Space que puede ser público. `space_index_html` lo pasa por
    `html.escape` — y hasta el 2026-09-18 ninguna prueba lo sostenía: quitar
    esa línea dejaba las 46 pruebas de este fichero en verde (sabotaje de la
    auditoría). Una línea que hace algo no obvio necesita su prueba."""

    def test_un_nombre_con_marcado_no_llega_crudo_a_la_pagina(self):
        malicioso = 'modelo</title><script>alert("x")</script>'
        html = space_index_html(malicioso)
        self.assertNotIn("<script>alert(", html)
        self.assertNotIn("</title><script>", html)
        self.assertIn("&lt;script&gt;alert(", html)

    def test_y_un_nombre_normal_sale_tal_cual(self):
        self.assertIn("<h1>riesgo de caída</h1>", space_index_html("riesgo de caída"))


class ElScriptDeLaPaginaEsJavaScriptValidoTest(unittest.TestCase):
    """La plantilla de `index.html` es una cadena NORMAL de Python: una barra
    invertida seguida de `n` dentro de ella llega a la página como un salto de
    línea REAL, y dentro de un literal de JavaScript o de un comentario `//` es
    un error de sintaxis que mata el script entero — la página se queda en
    «Loading…» sin decir por qué. Pasó DOS veces seguidas el 2026-09-18 al
    arreglar el codificador, y ninguna prueba de este fichero lo habría visto:
    todas comparan texto. Esta pasa el script generado por `node --check`."""

    def _script_de_la_pagina(self) -> str:
        html = space_index_html("modelo")
        inicio = html.index('"use strict"')
        return html[inicio:html.index("</script>", inicio)]

    def test_el_script_generado_pasa_node_check(self):
        import shutil
        import subprocess
        import tempfile
        node = shutil.which("node")
        if node is None:
            self.skipTest("sin `node` en esta máquina: no se puede comprobar la sintaxis del script")
        with tempfile.TemporaryDirectory() as tmp:
            ruta = f"{tmp}/pagina.js"
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(self._script_de_la_pagina())
            r = subprocess.run([node, "--check", ruta], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, f"el script de la página no es JavaScript válido:\n{r.stderr}")

    def test_y_el_FUENTE_de_la_plantilla_no_lleva_ninguna_barra_invertida(self):
        # La causa, además del síntoma. Se lee el FUENTE de `space.py`, no el
        # HTML generado: en el generado Python ya ha convertido el escape y la
        # barra no está (la primera versión de esta prueba miraba ahí y un
        # sabotaje con una barra invertida y una `n` la dejó VERDE).
        from pathlib import Path
        import matrixai.export.space as space
        fuente = Path(space.__file__).read_text(encoding="utf-8")
        inicio = fuente.index('_INDEX_HTML_TEMPLATE = """') + len('_INDEX_HTML_TEMPLATE = """')
        plantilla = fuente[inicio:fuente.index('"""', inicio)]
        self.assertNotIn(chr(92), plantilla)
