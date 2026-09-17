# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C2 — la declaración del componente de terceros, y su validador.

Lo que se fija aquí:

1. **Ni un número tecleado.** Cada dato de la declaración se contrasta contra
   `catalogo_medido.json` leído aparte: si alguien escribe a mano una
   dimensión, un digest o una AUC, esto se pone rojo. Es la misma regla que
   gobierna `TERCEROS.md` y la tabla de 107-C1 — *la tabla se mide, no se
   escribe*, justo para que caduque RUIDOSA.
2. **Las TRES cosas del invariante 6**, reescrito el 2026-09-14 porque se midió
   falso: el digest de los pesos, el del tokenizer **y el truncado**. Y la
   tercera con su procedencia comprobada contra el paquete, no solo copiada.
3. **Lo opaco se dice, y lo que no se sabe también** (invariante 4): `opaco:
   true` con su frase, y `lo_que_no_se_sabe` con los hechos que lo sostienen.
4. **Licencia por código Y por checkpoint** (invariante 5): la revisión es el
   commit entero, la licencia viene del origen en ESA revisión, y sin
   aceptación no hay declaración.
5. **El validador es fail-closed**: quitar cualquier campo obligatorio lo caza.
"""
from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

from matrixai.export import terceros
from matrixai.text.embeddings import cobertura
from matrixai.text.embeddings.catalogo import por_id
from matrixai.text.embeddings.declaracion import (
    ARTEFACTO,
    FRASE_DE_OPACIDAD,
    DeclaracionImposible,
    componente_de_terceros,
)

#: El estático inglés. Se usa porque es el que tiene las dos mitades medidas:
#: cubre inglés y NO cubre español, así que sirve para comprobar que un idioma
#: no cubierto se declara en vez de callarse (invariante 7). Un aserto negativo
#: lo pasa cualquier cosa; este no.
ESTATICO = "potion-base-8M"

#: Y el transformer inglés, que es el caso contrario en la longitud: su
#: truncado SÍ lo declara el paquete (tres veces, y distinto cada vez).
ONNX = "all-MiniLM-L6-v2-onnx"

#: La longitud fijada de `potion-base-8M`, tal como la fija quien lo ejecuta
#: (`matrixai_engines.embeddings.LONGITUDES_FIJADAS`). Se escribe aquí porque
#: es lo ÚNICO que no sale del catálogo: es el dato que el paquete del
#: proveedor no declara en ningún fichero.
LONGITUD_DEL_ESTATICO = {
    "tokens": 512,
    "origen": "libreria_del_proveedor",
    "fuente": "valor por omisión de `max_length` en `model2vec.StaticModel.encode` "
              "(model2vec 0.9.0), leído con `inspect.signature` el 2026-09-14",
    "por_que": "no está en ningún fichero del paquete: ni en config.json (que dice "
               "seq_length: 1000000), ni en tokenizer.json, ni en modules.json",
    "fijada_en": "matrixai_engines.embeddings.proveedor_de_texto.LONGITUDES_FIJADAS",
}

LONGITUD_DEL_ONNX = {
    "tokens": 256,
    "origen": "paquete",
    "fuente": "sentence_bert_config.json:max_seq_length del paquete, en su revisión",
    "por_que": "es la que usa sentence-transformers al cargar el modelo; el paquete "
               "declara otras dos y no coinciden",
    "fijada_en": "tests/test_c107_c2_declaracion.py (fijación de prueba)",
}


def _artefacto() -> dict:
    return json.loads(ARTEFACTO.read_text(encoding="utf-8"))


def _fila(proveedor_id: str = ESTATICO) -> dict:
    return _artefacto()["proveedores"][proveedor_id]


def _declaracion(proveedor_id: str = ESTATICO, **cambios):
    argumentos = {
        "longitud": LONGITUD_DEL_ESTATICO if proveedor_id == ESTATICO else LONGITUD_DEL_ONNX,
        "ejecutado_por": "matrixai-engines",
    }
    argumentos.update(cambios)
    return componente_de_terceros(proveedor_id, **argumentos)


def _artefacto_retocado(tmp: Path, retoque) -> Path:
    """Una copia del catálogo con un cambio, para probar lo que pasa cuando el
    artefacto dice otra cosa. Nunca se toca el artefacto de verdad."""
    datos = _artefacto()
    retoque(datos)
    ruta = tmp / "catalogo_retocado.json"
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    return ruta


class TestNingunNumeroEstaTecleado(unittest.TestCase):
    """Todo dato medible de la declaración se contrasta contra el artefacto."""

    def test_los_digests_y_los_tamanos_son_los_del_catalogo(self):
        c = _declaracion()
        fila = _fila()
        por_ruta = {f["ruta"]: f for f in fila["ficheros"]}
        self.assertEqual(c["digest_de_los_pesos"], por_ruta[por_id(ESTATICO).modelo])
        self.assertEqual(c["digest_del_tokenizer"], por_ruta["tokenizer.json"])
        self.assertEqual(c["ficheros_fijados"], fila["ficheros"])

    def test_la_dimension_y_la_composicion_salen_de_la_medicion(self):
        c = _declaracion()
        m = _fila()["medicion"]
        self.assertEqual(c["dimension"], m["dimension"])
        self.assertEqual(c["pooling"], m["describe"]["pooling"])
        self.assertEqual(c["normaliza"], m["describe"]["normaliza"])
        self.assertEqual(c["compuesto_segun"], m["describe"]["leido_de"])

    def test_la_licencia_entera_sale_del_catalogo_con_su_fuente_y_su_fecha(self):
        c = _declaracion()
        self.assertEqual(c["licencia"], _fila()["licencia"])
        self.assertEqual(c["aceptacion_de_licencia"], _fila()["aceptacion_de_licencia"])
        # Y la fuente es la URL del ORIGEN en la revisión exacta, no la del repo.
        self.assertIn(c["revision"], c["licencia"]["fuente"])

    def test_las_aucs_de_idioma_son_las_medidas(self):
        c = _declaracion()
        medido = _fila()["medicion"]["coherencia_por_documento"]
        for idioma, fila in medido.items():
            with self.subTest(idioma=idioma):
                self.assertEqual(
                    c["idiomas_medidos"][idioma]["auc_coherencia_por_documento"],
                    fila["auc"])

    def test_el_artefacto_manda_sobre_la_declaracion(self):
        """Si el catálogo dice otra dimensión, la declaración dice esa otra.

        Es la prueba de que no hay una copia escondida: sin esto, un fichero de
        constantes podría estar diciendo 256 pase lo que pase.
        """
        import tempfile
        tmp = Path(tempfile.mkdtemp())

        def retocar(datos):
            datos["proveedores"][ESTATICO]["medicion"]["dimension"] = 1234

        ruta = _artefacto_retocado(tmp, retocar)
        c = _declaracion(artefacto=ruta)
        self.assertEqual(c["dimension"], 1234)


class TestLasTresCosasDelInvariante6(unittest.TestCase):
    """Con los dos digests el vector TODAVÍA no está prometido: hace falta el
    truncado, y con su procedencia. Medido: coseno 0,658 con el corte contra
    1,000 sin él, sobre un texto de 8.008 tokens."""

    def test_estan_las_tres(self):
        tres = terceros.las_tres_cosas(_declaracion())
        self.assertEqual(sorted(tres), ["digest_de_los_pesos", "digest_del_tokenizer",
                                        "longitud_maxima_tokens"])
        for clave, valor in tres.items():
            with self.subTest(clave=clave):
                self.assertTrue(valor, clave)

    def test_sin_alguno_de_los_campos_de_la_longitud_no_hay_declaracion(self):
        for campo in ("tokens", "origen", "fuente", "por_que", "fijada_en"):
            with self.subTest(campo=campo):
                longitud = dict(LONGITUD_DEL_ESTATICO)
                longitud.pop(campo)
                with self.assertRaises(DeclaracionImposible):
                    _declaracion(longitud=longitud)

    def test_una_longitud_que_no_es_la_que_se_midio_se_rechaza(self):
        """Lo medido describe el vector de SU truncado. Atribuir una AUC medida
        con 512 tokens a un proveedor que trunca a 256 es declarar un número
        que nadie ha medido."""
        longitud = dict(LONGITUD_DEL_ESTATICO, tokens=256)
        with self.assertRaises(DeclaracionImposible) as e:
            _declaracion(longitud=longitud)
        self.assertIn("512", str(e.exception))

    def test_decir_que_la_longitud_viene_del_paquete_cuando_no_viene_se_rechaza(self):
        """`potion-base-8M` no declara 512 en ningún fichero: los dos que tiene
        dicen 1.000.000, o sea «no truncar». Llamar «paquete» a ese origen
        borraría justo el hallazgo que reescribió el invariante 6."""
        longitud = dict(LONGITUD_DEL_ESTATICO, origen="paquete")
        with self.assertRaises(DeclaracionImposible):
            _declaracion(longitud=longitud)

    def test_decir_que_viene_de_una_libreria_cuando_lo_declara_el_paquete_se_rechaza(self):
        """El lado contrario, que es el que se olvida: `all-MiniLM-L6-v2`
        declara sus 256 en `sentence_bert_config.json`. Arreglar un sesgo puede
        crear el contrario, así que se prueban los dos lados."""
        longitud = dict(LONGITUD_DEL_ONNX, origen="libreria_del_proveedor")
        with self.assertRaises(DeclaracionImposible):
            _declaracion(ONNX, longitud=longitud)

    def test_el_onnx_con_su_origen_correcto_si_se_declara(self):
        c = _declaracion(ONNX)
        self.assertEqual(c["longitud_maxima"]["tokens"], 256)
        self.assertEqual(c["longitud_maxima"]["origen"], "paquete")

    def test_un_origen_que_no_esta_en_el_vocabulario_se_rechaza(self):
        longitud = dict(LONGITUD_DEL_ESTATICO, origen="me_lo_han_dicho")
        with self.assertRaises(DeclaracionImposible):
            _declaracion(longitud=longitud)

    def test_la_longitud_viaja_con_contra_que_se_comprobo(self):
        c = _declaracion()
        contra = c["longitud_maxima"]["comprobada_contra"]
        self.assertTrue(contra, "sin testigos, `origen` sería prosa")
        for linea in contra:
            with self.subTest(linea=linea):
                self.assertIn("catalogo_medido.json", linea)

    def test_el_paquete_ensena_TODAS_las_longitudes_que_declara(self):
        """Hallazgo 3 de 107-C1: `all-MiniLM-L6-v2` declara TRES longitudes
        distintas en tres ficheros suyos. Elegir en silencio baja el coseno a
        0,999 en los textos largos — lo bastante poco como para no verlo."""
        c = _declaracion(ONNX)
        declaradas = c["longitud_maxima"]["que_declara_el_paquete"]
        self.assertGreaterEqual(len(declaradas), 3)
        self.assertEqual(
            sorted(e["valor"] for e in declaradas),
            sorted(e["valor"] for e in
                   _fila(ONNX)["medicion"]["describe"]["max_tokens_declarados_por_el_paquete"]))


class TestLoOpacoSeDiceYLoQueNoSeSabeTambien(unittest.TestCase):
    """Invariante 4: «un embedding preentrenado es OPACO, y el expediente lo
    escribe así». Declarar lo que se conoce Y lo que no."""

    def test_opaco_es_true_y_lleva_su_frase(self):
        c = _declaracion()
        self.assertIs(c["opaco"], True)
        self.assertEqual(c["opacidad"], FRASE_DE_OPACIDAD)
        self.assertIn("no auditados", c["opacidad"])

    def test_la_frase_de_opacidad_es_LA_MISMA_que_dice_el_proveedor(self):
        """Dos sitios declarando lo mismo acaban divergiendo. El proveedor del
        núcleo escribe esta frase en su `describe()`, y lo que quedó grabado en
        el catálogo al medirlo es esa salida suya: si alguien cambia una de las
        dos, esto se pone rojo."""
        self.assertEqual(_fila()["medicion"]["describe"]["opacidad"], FRASE_DE_OPACIDAD)

    def test_hay_lo_que_no_se_sabe_y_no_es_una_frase_de_relleno(self):
        c = _declaracion()
        dicho = c["lo_que_no_se_sabe"]
        self.assertGreaterEqual(len(dicho), 4)
        junto = " ".join(dicho)
        # Los cuatro huecos que 107-C1 dejó medidos y escritos.
        self.assertIn("no se ha auditado", junto)
        self.assertIn("107-C0", junto)
        self.assertIn("no se extrapolan", junto)
        self.assertIn("«no medido» no es «no cubierto»", junto)

    def test_un_idioma_medido_y_no_cubierto_se_declara_con_su_numero(self):
        """`potion-base-8M` da AUC 0,63 en español sobre 0,5 de azar: da un
        vector para cualquier frase y ese vector no significa gran cosa. Es
        exactamente el caso que el invariante 7 existe para no dejar en
        silencio."""
        c = _declaracion()
        es = c["idiomas_medidos"]["es"]
        self.assertEqual(es["estado"], cobertura.NO_CUBIERTO)
        self.assertEqual(es["auc_coherencia_por_documento"],
                         _fila()["medicion"]["coherencia_por_documento"]["es"]["auc"])
        self.assertIn("NO cubre", es["frase"])
        self.assertIn("NO cubiertos por este proveedor: es",
                      " ".join(c["lo_que_no_se_sabe"]))

    def test_un_idioma_que_nadie_midio_responde_no_medido_y_no_un_hueco(self):
        """«No medido» no es «no cubierto»: son dos hechos distintos. Del
        francés no se puede decir nada; del español de este proveedor sí."""
        c = _declaracion(idiomas=("fr",))
        self.assertEqual(c["idiomas_medidos"]["fr"]["estado"], cobertura.NO_MEDIDO)
        self.assertIsNone(c["idiomas_medidos"]["fr"]["auc_coherencia_por_documento"])
        self.assertNotEqual(c["idiomas_medidos"]["fr"]["estado"],
                            c["idiomas_medidos"]["es"]["estado"])

    def test_la_cita_del_autor_va_marcada_como_cita(self):
        """`potion-base-8M` dice «English» y el multilingüe dice «101 idiomas»:
        eso no es una medida, y el componente no puede enseñarlo como si lo
        fuera."""
        c = _declaracion()
        segun_el_autor = c["idiomas_segun_su_autor"]
        self.assertEqual(segun_el_autor["cita"], por_id(ESTATICO).idiomas_segun_su_autor)
        self.assertIn("no una medida", segun_el_autor["advertencia"])

    def test_cuando_no_se_pasan_dependencias_se_dice(self):
        sin = _declaracion()
        self.assertNotIn("dependencias", sin)
        self.assertIn("no dice con qué versiones", " ".join(sin["lo_que_no_se_sabe"]))
        con = _declaracion(dependencias={"onnxruntime": {"version": "1.26.0",
                                                         "licencia": "MIT"}})
        self.assertEqual(con["dependencias"]["onnxruntime"]["version"], "1.26.0")
        self.assertNotIn("no dice con qué versiones", " ".join(con["lo_que_no_se_sabe"]))

    def test_los_pesos_no_van_en_la_imagen_y_lo_dice(self):
        self.assertIs(_declaracion()["pesos_en_la_imagen"], False)

    def test_datos_de_ajuste_ninguno_mientras_no_se_ajuste_nada(self):
        self.assertEqual(_declaracion()["datos_de_ajuste"], "ninguno")


class TestLicenciaPorCodigoYPorCheckpoint(unittest.TestCase):
    """Invariante 5: licencia por código Y por checkpoint concreto; descargar y
    aceptar no elimina restricciones."""

    def test_la_revision_es_el_commit_entero_y_no_una_rama(self):
        revision = _declaracion()["revision"]
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", revision), revision)

    def test_una_licencia_incompatible_no_se_declara_por_bueno_que_sea(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())

        def retocar(datos):
            lic = datos["proveedores"][ESTATICO]["licencia"]
            lic["spdx"] = "CC-BY-NC-4.0"
            lic["compatible"] = False
            lic["motivo_si_no"] = "prohíbe el uso comercial"

        with self.assertRaises(DeclaracionImposible) as e:
            _declaracion(artefacto=_artefacto_retocado(tmp, retocar))
        self.assertIn("uso comercial", str(e.exception))

    def test_sin_aceptacion_de_licencia_no_hay_declaracion(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())

        def retocar(datos):
            datos["proveedores"][ESTATICO].pop("aceptacion_de_licencia", None)

        with self.assertRaises(DeclaracionImposible) as e:
            _declaracion(artefacto=_artefacto_retocado(tmp, retocar))
        self.assertIn("aceptad", str(e.exception).lower())


class TestUnProveedorQueNuncaSeEjecutoNoSeDeclara(unittest.TestCase):
    """Fijado y con licencia leída NO es lo mismo que medido. Un paquete no
    puede declarar como el componente que produce sus vectores a algo de lo que
    no se conoce ni la dimensión."""

    def test_un_multilingue_nunca_ejecutado_no_se_puede_declarar_y_dice_por_que(self):
        """Hasta el 2026-09-17 el ejemplo era `potion-multilingual-128M`, y
        asertaba su motivo («537 MB»). Ese día se midió: ya no es «nunca
        ejecutado», y su declaración se rechaza por OTRA razón (la longitud que
        se fija no es la medida), que es la prueba de abajo. La intención se
        conserva con el que sigue sin ejecutarse."""
        longitud = dict(LONGITUD_DEL_ESTATICO, fijada_en="da igual")
        with self.assertRaises(DeclaracionImposible) as e:
            componente_de_terceros("multilingual-e5-small-onnx-int8", longitud=longitud,
                                   ejecutado_por="matrixai-engines")
        self.assertIn("nunca se ha ejecutado", str(e.exception))
        self.assertIn("135 MB", str(e.exception))

    def test_y_potion_multilingual_medido_TAMPOCO_se_declara_con_otra_longitud(self):
        """Medido no es declarable a cualquier longitud: lo medido describe el
        vector a la longitud con que se midió."""
        longitud = dict(LONGITUD_DEL_ESTATICO, fijada_en="da igual")
        with self.assertRaises(DeclaracionImposible) as e:
            componente_de_terceros("potion-multilingual-128M", longitud=longitud,
                                   ejecutado_por="matrixai-engines")
        self.assertNotIn("nunca se ha ejecutado", str(e.exception))
        self.assertIn("se midió con", str(e.exception))

    def test_un_proveedor_que_no_esta_en_el_catalogo_tampoco(self):
        with self.assertRaises(DeclaracionImposible):
            componente_de_terceros("un-modelo-cualquiera",
                                   longitud=LONGITUD_DEL_ESTATICO,
                                   ejecutado_por="matrixai-engines")

    def test_sin_ejecutado_por_no_hay_declaracion(self):
        """Unos pesos ajenos los corre alguien concreto. Un valor por omisión
        aquí sería mentira el día que lo ejecute otro."""
        with self.assertRaises(DeclaracionImposible):
            componente_de_terceros(ESTATICO, longitud=LONGITUD_DEL_ESTATICO,
                                   ejecutado_por="  ")

    def test_sin_catalogo_medido_no_se_inventa_nada(self):
        import tempfile
        with self.assertRaises(DeclaracionImposible):
            _declaracion(artefacto=Path(tempfile.mkdtemp()) / "no_existe.json")


class TestElValidadorEsFailClosed(unittest.TestCase):
    """Lo que `verify`, el manifiesto y el BOM usan para decir «está entero»."""

    def test_una_declaracion_de_verdad_pasa_entera(self):
        self.assertEqual(terceros.validar(_declaracion()), [])

    def test_quitar_CUALQUIER_campo_obligatorio_lo_caza(self):
        base = _declaracion()
        obligatorios = [
            "componente", "id", "familia", "repo", "revision", "ejecutado_por",
            "licencia", "aceptacion_de_licencia", "digest_de_los_pesos",
            "digest_del_tokenizer", "longitud_maxima", "dimension", "pooling",
            "normaliza", "idiomas_medidos", "opaco", "opacidad", "datos_de_ajuste",
            "pesos_en_la_imagen",
        ]
        for campo in obligatorios:
            with self.subTest(campo=campo):
                roto = copy.deepcopy(base)
                roto.pop(campo)
                self.assertIn(campo, " ".join(terceros.validar(roto)))

    def test_quitar_un_campo_DENTRO_de_la_licencia_o_del_digest_tambien(self):
        base = _declaracion()
        for camino in (("licencia", "spdx"), ("licencia", "fuente"),
                       ("licencia", "leida_el"),
                       ("aceptacion_de_licencia", "aceptada_el"),
                       ("digest_de_los_pesos", "digest"),
                       ("digest_del_tokenizer", "digest"),
                       ("digest_del_tokenizer", "algoritmo"),
                       ("longitud_maxima", "tokens"), ("longitud_maxima", "fuente"),
                       ("longitud_maxima", "fijada_en")):
            with self.subTest(camino=camino):
                roto = copy.deepcopy(base)
                roto[camino[0]].pop(camino[1])
                self.assertIn(".".join(camino), terceros.validar(roto))

    def test_datos_de_ajuste_solo_admite_ninguno(self):
        """Cualquier otro valor describe un modelo AJUSTADO, que ya no es el
        del catálogo: sus digests, su licencia y su cobertura son otros."""
        roto = dict(_declaracion(), datos_de_ajuste="un corpus del hospital")
        self.assertIn("datos_de_ajuste", terceros.validar(roto))

    def test_opaco_false_no_cuela_y_la_frase_vacia_tampoco(self):
        self.assertIn("opaco", terceros.validar(dict(_declaracion(), opaco=False)))
        self.assertIn("opacidad", terceros.validar(dict(_declaracion(), opacidad="  ")))

    def test_pesos_en_la_imagen_true_contradice_al_paquete_entero(self):
        self.assertIn("pesos_en_la_imagen",
                      terceros.validar(dict(_declaracion(), pesos_en_la_imagen=True)))

    def test_una_licencia_incompatible_no_pasa_el_validador(self):
        roto = copy.deepcopy(_declaracion())
        roto["licencia"]["compatible"] = False
        self.assertIn("licencia.compatible", terceros.validar(roto))

    def test_aceptar_OTRA_licencia_no_es_aceptar_esta(self):
        roto = copy.deepcopy(_declaracion())
        roto["aceptacion_de_licencia"]["spdx"] = "Apache-2.0"
        self.assertIn("aceptacion_de_licencia.spdx", terceros.validar(roto))

    def test_una_revision_que_no_es_un_commit_entero_no_cuela(self):
        for revision in ("main", "v1.0", "bf8b0566", "z" * 40):
            with self.subTest(revision=revision):
                self.assertIn("revision",
                              terceros.validar(dict(_declaracion(), revision=revision)))

    def test_una_dimension_booleana_no_es_una_dimension(self):
        """`True` es un `int` para Python y se habría publicado como «1»."""
        self.assertIn("dimension", terceros.validar(dict(_declaracion(), dimension=True)))

    def test_un_algoritmo_de_digest_desconocido_no_cuela(self):
        roto = copy.deepcopy(_declaracion())
        roto["digest_del_tokenizer"]["algoritmo"] = "crc32"
        self.assertIn("digest_del_tokenizer.algoritmo", terceros.validar(roto))

    def test_lo_que_no_es_un_diccionario_se_rechaza_entero(self):
        for basura in (None, [], "embedding_provider", 0):
            with self.subTest(basura=basura):
                self.assertEqual(terceros.validar(basura), ["embedding_provider"])


class TestComponerLaDeclaracionNoLE_PIDE_DEPENDENCIAS_AL_NUCLEO(unittest.TestCase):
    """102, invariante 1: `matrixai-core` se instala con `dependencies = []`.

    Esta pieza compone el componente de terceros LEYENDO el catálogo, sin
    ejecutar nada, y por eso tiene que funcionar en una máquina sin `numpy` ni
    `onnxruntime`. Lo hace pasando por `matrixai.export.terceros` para
    comprobarse, y ese camino arrastra el paquete `matrixai.export` entero: si
    alguien sube ahí un `import numpy` al principio de un módulo, esto se pone
    rojo en vez de descubrirse en la máquina de un cliente.
    """

    def test_se_compone_sin_numpy_ni_onnxruntime(self):
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
            "from matrixai.text.embeddings.declaracion import componente_de_terceros\n"
            f"c = componente_de_terceros({ESTATICO!r}, longitud={LONGITUD_DEL_ESTATICO!r},\n"
            "                            ejecutado_por='matrixai-engines')\n"
            "assert c['opaco'] is True and c['dimension'] > 0\n"
            "try:\n"
            "    import numpy\n"
            "except ImportError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('el veto no funcionó: numpy se importó igual')\n"
            "print('ok')\n"
        )
        r = subprocess.run([sys.executable, "-c", guion],
                           cwd=str(Path(__file__).resolve().parent.parent),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("ok", r.stdout)


class TestLaDeclaracionDelNucleoYLaDeEnginesNoDivergen(unittest.TestCase):
    """`matrixai-engines` compone la suya para el proveedor que habla español
    (107-C1, D1). Son dos composiciones del MISMO componente, y dos sitios
    declarando lo mismo acaban divergiendo: lo que impide que pase es que el
    validador de aquí es el que gobierna a los dos.

    Esto se lee del FUENTE de engines, no se ejecuta: ejecutarlo exige
    `tokenizers`, `onnxruntime` y 118 MB de pesos descargados.
    """

    FUENTE = (Path(__file__).resolve().parents[2] / "matrixai-engines" / "src"
              / "matrixai_engines" / "embeddings" / "proveedor_de_texto.py")

    @unittest.skipUnless(FUENTE.is_file(), f"no está {FUENTE}")
    def test_engines_declara_todas_las_claves_que_el_validador_exige(self):
        fuente = self.FUENTE.read_text(encoding="utf-8")
        cuerpo = fuente.split("def declaracion(", 1)[1]
        claves = set(re.findall(r'^\s{12}"([a-z_]+)":', cuerpo, flags=re.M))
        # Las que el validador exige en el primer nivel del componente.
        exigidas = {c for c in terceros.validar({}) if "." not in c}
        exigidas.discard("embedding_provider")
        faltan = sorted(exigidas - claves)
        self.assertEqual(faltan, [], f"engines no declara: {faltan}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
