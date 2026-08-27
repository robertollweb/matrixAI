"""CONTRATO 85-C2b — los motivos de `verify`, en el idioma que se pida.

Medido el 2026-08-24: `matrixai/export/verify.py` redactaba TODOS sus
motivos solo en inglés, y esos motivos ya se pintan en la interfaz por
fases. Con el idioma en español salía media pantalla en inglés. *Lo que
redacta el core se traduce en el core, no al pintarlo.*

Lo que este fichero mide, y por qué así:

* **El barrido va sobre paquetes DE VERDAD**, no sobre el diccionario:
  un diccionario perfecto y un `verify_package` que sigue escribiendo
  literales darían el mismo verde. Los paquetes de aquí se construyen con
  el escritor de manifiestos del producto y dejan las cuatro etapas en
  estados distintos —`PASS`, `FAIL`, `NOT_RUN`, `INCOMPARABLE`—.
* **Y TAMBIÉN sobre el catálogo entero**, porque ningún abanico de
  paquetes toca las 50 y pico frases: una que ninguna sonda alcance se
  quedaría sin traducir y sin que nadie lo viera hasta tenerla delante.
  Las dos cosas: la primera prueba que el cableado llega, la segunda que
  no se ha quedado ninguna atrás.
* **Con palabras FUNCIONALES**, no con una lista escogida a mano: `the`,
  `of`, `was`, `not`, `from`… no se pueden evitar escribiendo en inglés,
  ni `el`, `la`, `del`, `que`, `desde`… escribiendo en castellano. Una
  lista de palabras de contenido se esquiva sin querer y no barre.

Lo que NO se traduce se comprueba igual de fuerte: los cuatro `status` y
los nombres de etapa son VALORES —quien encadena `verify && desplegar`
depende de ellos—, y lo que un motivo interpola (una ruta, una huella, un
número) es un dato y no cambia de idioma.
"""

import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from matrixai.export.reproduce import (
    REPRODUCE_MANIFEST_FILENAME,
    añadir_inventario_de_ficheros,
    manifest_digest,
    write_reproduce_manifest,
)
from matrixai.export.verify import ESTADOS, verify_package
from matrixai.export.verify_textos import IDIOMAS, _MOTIVOS

# --------------------------------------------------------------------------
# El barrido
# --------------------------------------------------------------------------

#: Inglés funcional. Escribir dos frases en inglés sin ninguna de éstas no
#: se puede: son los artículos, las preposiciones, los auxiliares y los
#: demostrativos. Fuera queda lo que también es castellano (`a`, `no`,
#: `he`, `me`) o lo que se solapa por accidente (`son`, `solo`).
_FUNCIONALES_INGLESAS = (
    "the", "of", "and", "or", "is", "are", "was", "were", "be", "been", "not",
    "with", "from", "does", "do", "did", "this", "that", "these", "those",
    "it", "its", "they", "their", "there", "has", "have", "had", "but", "so",
    "than", "then", "which", "what", "when", "where", "how", "can", "cannot",
    "could", "would", "should", "will", "must", "into", "only", "also", "any",
    "all", "some", "each", "every", "more", "most", "other", "same", "such",
    "because", "about", "against", "between", "before", "after", "both",
    "too", "very", "just", "in", "on", "to", "for", "by", "at", "as", "if",
    "nor", "neither", "whether", "your", "our", "my",
)

#: Castellano funcional, con el mismo criterio y las mismas exclusiones.
_FUNCIONALES_ESPANOLAS = (
    "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al", "de",
    "en", "con", "que", "por", "para", "se", "su", "sus", "lo", "ni", "pero",
    "porque", "cuando", "donde", "más", "menos", "este", "esta", "esto",
    "estos", "estas", "ese", "esa", "eso", "esos", "sin", "sobre", "entre",
    "hasta", "desde", "antes", "después", "aunque", "ya", "es", "era", "fue",
    "ha", "han", "hay", "puede", "pueden", "tiene", "tienen", "dice", "todo",
    "toda", "todos", "todas", "otro", "otra", "otros", "otras", "mismo",
    "misma", "qué", "cuál", "quién", "nada", "algo", "cada", "así", "ningún",
    "ninguna", "ninguno", "cualquier", "según", "tampoco", "también", "muy",
)

_INTRUSAS = {"es": _FUNCIONALES_INGLESAS, "en": _FUNCIONALES_ESPANOLAS}

#: Las claves de una etapa que llevan PROSA. Son las que traduce el core.
_CLAVES_DE_PROSA = ("reason", "note", "problem")

#: Y las que llevan DATOS: huellas, rutas, números, nombres de artefacto y
#: de campo. Se declaran para que una clave nueva rompa esta prueba en vez
#: de colarse sin que nadie decida de cuál de los dos montones es —un
#: barrido que solo mira lo que ya conocía afirma por omisión sobre lo
#: demás.
_CLAVES_DE_DATO = (
    "status", "artifacts", "artifact", "path", "expected", "found", "field",
    "missing", "checked", "files_checked", "uncovered_files", "rows",
    "sha256", "csv_text", "best_epoch", "metrics", "metric", "published",
    "obtained", "difference", "tolerance", "applied_from_capture",
    "not_applied_from_capture", "not_applied",
)


def _sin_citas(texto: str) -> str:
    """El texto quitándole lo que viene CITADO de fuera.

    Entre «comillas» (español) o "comillas" (inglés) viaja lo que este
    verificador no escribió: el mensaje de una excepción de la biblioteca
    estándar, o el motivo que el propio paquete redactó en su
    `reproduce.json`. Eso es un dato, igual que una ruta o una huella, y
    no cambia de idioma — por eso el barrido lo salta.
    """
    return re.sub(r'"[^"]*"', ' … ', re.sub("«[^»]*»", " … ", texto))


def _intrusas(texto: str, idioma: str) -> list[str]:
    """Las palabras funcionales del OTRO idioma que hay en `texto`."""
    limpio = _sin_citas(texto)
    return sorted({p for p in _INTRUSAS[idioma]
                   if re.search(rf"\b{re.escape(p)}\b", limpio, re.IGNORECASE)})


def _prosa(nodo, ruta: str = ""):
    """Recorre el informe y devuelve `(dónde, texto)` de cada frase."""
    if isinstance(nodo, dict):
        for clave, valor in nodo.items():
            if clave in _CLAVES_DE_PROSA and isinstance(valor, str):
                yield f"{ruta}.{clave}", valor
            else:
                yield from _prosa(valor, f"{ruta}.{clave}")
    elif isinstance(nodo, list):
        for i, valor in enumerate(nodo):
            yield from _prosa(valor, f"{ruta}[{i}]")


# --------------------------------------------------------------------------
# Paquetes de verdad
# --------------------------------------------------------------------------

_RECETA = "genera 200 filas\n"
_CSV = "a,b\n1,2\n3,4\n"


def _paquete() -> Path:
    """Un paquete honesto en disco, con el `reproduce.json` del producto."""
    d = Path(tempfile.mkdtemp())
    (d / "model.mxai").write_text("NETWORK N\n  DENSE 4\n")
    (d / "training.mxtrain").write_text(
        Path("examples/celsius_to_kelvin.mxtrain").read_text(encoding="utf-8"))
    (d / "recipe.txt").write_text(_RECETA)
    sha = hashlib.sha256(_CSV.encode()).hexdigest()
    train = (d / "training.mxtrain").read_text(encoding="utf-8")
    write_reproduce_manifest(
        d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
        dataset_sha256=sha, dataset_rows=2, generation={"seeds": {"dataset": 42}},
        weights_source="trained",
        run_provenance={
            "schema_version": "1.2",
            "mxai_sha256": hashlib.sha256((d / "model.mxai").read_bytes()).hexdigest(),
            "mxtrain_sha256": hashlib.sha256(train.encode()).hexdigest(),
            "mxtrain_text": train,
            "recipe_sha256": hashlib.sha256(_RECETA.encode()).hexdigest(),
            "recipe_text": _RECETA, "seeds": {"dataset": 42},
            "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
            "dataset_rows": 2, "dataset_rows_used": 2,
            "epochs_effective": 10000, "epochs_ran": 10000, "warm_start": False,
            "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
        })
    añadir_inventario_de_ficheros(d)
    return d


def _tocar(d: Path, cambio) -> Path:
    """Cambia el manifiesto y le REHACE su `manifest_sha256`.

    Como lo haría quien fabrica el paquete: si no se rehace, todo cae por
    el mismo sitio (`manifest` FAIL) y las demás ramas no se tocan nunca.
    """
    m = json.loads((d / REPRODUCE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    cambio(m)
    m["manifest_sha256"] = manifest_digest(m)
    (d / REPRODUCE_MANIFEST_FILENAME).write_text(json.dumps(m), encoding="utf-8")
    return d


def _paquete_regenerable() -> Path:
    """Uno que R1 SÍ puede rehacer, para llegar a `training` y a `R3`.

    Sin él las cuatro etapas se quedan en `NOT_RUN`/`INCOMPARABLE` y la
    nota del entrenamiento —que también es prosa— no se barre nunca.
    """
    from matrixai.export.reproduce import _epochs_from_training
    from matrixai.playground import _generate_synthetic_dataset
    from matrixai.training.dataset_project import generate_project_from_dataset

    filas = "".join(f"{c},{c + 273.15}\n" for c in range(30))
    proj = generate_project_from_dataset(
        "centigrados,prediccionKelvin\n" + filas, "prediccionKelvin",
        column_type_overrides={"centigrados": "number"},
        column_range_overrides={"centigrados": (0.0, 29.0)})
    gen = _generate_synthetic_dataset(
        proj["mxai"], proj["training_text"], 30, 7, "coherent",
        field_ranges_override=proj.get("field_ranges"))
    sha = hashlib.sha256(gen["csv_text"].encode("utf-8")).hexdigest()
    epocas = _epochs_from_training(proj["training_text"])

    d = Path(tempfile.mkdtemp())
    (d / "model.mxai").write_text(proj["mxai"], encoding="utf-8")
    (d / "training.mxtrain").write_text(proj["training_text"], encoding="utf-8")
    (d / "recipe.txt").write_text("regenerable\n", encoding="utf-8")
    write_reproduce_manifest(
        d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
        dataset_sha256=sha, dataset_rows=30, generation={"seeds": {"dataset": 7}},
        weights_source="trained",
        run_provenance={
            "schema_version": "1.2",
            "mxai_sha256": hashlib.sha256(proj["mxai"].encode()).hexdigest(),
            "mxtrain_sha256": hashlib.sha256(proj["training_text"].encode()).hexdigest(),
            "mxtrain_text": proj["training_text"],
            "recipe_sha256": hashlib.sha256(b"regenerable\n").hexdigest(),
            "recipe_text": "regenerable\n", "seeds": {"dataset": 7},
            "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
            "dataset_rows": 30, "dataset_rows_used": 30,
            "epochs_effective": epocas, "epochs_ran": epocas, "warm_start": False,
            "mode": "coherent", "field_ranges": proj.get("field_ranges") or {},
            "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
        })
    añadir_inventario_de_ficheros(d)
    return d


def _casos() -> list[tuple[str, Path, dict]]:
    """Los paquetes del barrido: `(etiqueta, directorio, extras)`.

    Se construyen UNA vez y se verifican en los dos idiomas: el mismo
    paquete tiene que dar el mismo veredicto y otra prosa.
    """
    con_digest_roto = _paquete()
    (con_digest_roto / "recipe.txt").write_text(_RECETA + "y otra línea\n")

    con_fichero_de_mas = _paquete()
    (con_fichero_de_mas / "sobra.txt").write_text("nadie me ha mirado\n")

    enlazado = _paquete()
    os.symlink(enlazado / "model.mxai", enlazado / "enlace.mxai")
    _tocar(enlazado, lambda m: m["artifacts"].__setitem__("model", {
        "path": "enlace.mxai",
        "sha256": hashlib.sha256((enlazado / "model.mxai").read_bytes()).hexdigest()}))

    return [
        # Cuatro etapas y tres estados distintos: PASS, INCOMPARABLE, NOT_RUN.
        ("paquete honesto, sin reentrenar", _paquete(), {}),
        # Y aquí el cuarto: FAIL, con su `problem` por artefacto.
        ("una línea cambiada en la receta", con_digest_roto, {}),
        ("sin reproduce.json", Path(tempfile.mkdtemp()), {}),
        ("un fichero que el manifiesto no cubre", con_fichero_de_mas, {}),
        ("una schema_version que este verificador no lee",
         _tocar(_paquete(), lambda m: m.__setitem__("schema_version", "999.0")), {}),
        ("un artefacto sin sha256 y otro que no es ni objeto ni null",
         _tocar(_paquete(), lambda m: (
             m["artifacts"].__setitem__("training", {"path": "training.mxtrain"}),
             m["artifacts"].__setitem__("recipe", 7))), {}),
        ("una ruta que se sale del paquete",
         _tocar(_paquete(), lambda m: m["artifacts"].__setitem__(
             "model", {"path": "../fuera.txt", "sha256": "a" * 64})), {}),
        ("una ruta absoluta",
         _tocar(_paquete(), lambda m: m["artifacts"].__setitem__(
             "model", {"path": "/etc/hostname", "sha256": "a" * 64})), {}),
        ("un enlace simbólico interno", enlazado, {}),
        # El paquete cita lo que él mismo escribió, y eso NO se traduce.
        ("el paquete declara que R1 no es posible",
         _tocar(_paquete(), lambda m: m["verifiable"].__setitem__("r1", {
             "possible": False, "missing": ["recipe"],
             "reason": "Cannot regenerate the dataset: the recipe does not travel."})),
         {}),
        # Y el caro: R1 rehace el dataset, `training` entrena y deja su nota.
        ("regenerable y reentrenado", _paquete_regenerable(), {"run_training": True}),
    ]


class ElInformeEnteroHablaUnSoloIdiomaTest(unittest.TestCase):
    """El barrido, sobre paquetes de verdad y en los dos sentidos.

    *Arreglar un sesgo puede crear el contrario*: se mide el español SIN
    inglés y el inglés SIN español, porque traducirlo todo al castellano y
    dejar la mitad inglesa rota sería el mismo defecto con otra ropa.
    """

    @classmethod
    def setUpClass(cls):
        cls.casos = _casos()

    def test_en_espanol_no_queda_ni_una_palabra_funcional_inglesa(self):
        for etiqueta, paquete, extras in self.casos:
            informe = verify_package(paquete, locale="es", **extras)
            for donde, texto in _prosa(informe["stages"], "stages"):
                with self.subTest(caso=etiqueta, donde=donde):
                    self.assertEqual(
                        [], _intrusas(texto, "es"),
                        f"queda inglés en la salida en español: {texto!r}")

    def test_en_ingles_no_queda_ni_una_palabra_funcional_espanola(self):
        for etiqueta, paquete, extras in self.casos:
            informe = verify_package(paquete, locale="en", **extras)
            for donde, texto in _prosa(informe["stages"], "stages"):
                with self.subTest(caso=etiqueta, donde=donde):
                    self.assertEqual(
                        [], _intrusas(texto, "en"),
                        f"queda español en la salida en inglés: {texto!r}")

    def test_y_el_barrido_LLEGA_a_prosa_de_verdad(self):
        """Un aserto negativo lo pasa un informe vacío.

        Sin esto, borrar todos los motivos dejaría los dos barridos de
        arriba en verde: *un aserto negativo lo pasa un render en blanco*.
        """
        frases = [t for _, paquete, extras in self.casos
                  for _, t in _prosa(verify_package(paquete, locale="es", **extras),
                                     "informe")]
        self.assertGreaterEqual(len(frases), 30, "el barrido casi no mira nada")
        # Y que las CUATRO etapas hayan pasado por los cuatro estados: un
        # abanico que solo llega a `INCOMPARABLE` no barre `PASS` ni `FAIL`.
        vistos = {e["status"]
                  for _, paquete, extras in self.casos
                  for e in verify_package(paquete, locale="es", **extras)["stages"].values()}
        self.assertEqual(set(ESTADOS), vistos)

    def test_una_clave_nueva_del_informe_no_se_cuela_sin_clasificar(self):
        """Prosa o dato: alguien tiene que decirlo.

        Si mañana una etapa saca un `explanation`, el barrido de arriba lo
        saltaría en silencio y este fichero seguiría verde diciendo que no
        queda inglés. Afirmar por omisión, otra vez.
        """
        claves = set()
        for _, paquete, extras in self.casos:
            for etapa in verify_package(paquete, locale="es", **extras)["stages"].values():
                # La etapa y las fichas de artefacto roto: los dos sitios
                # donde `verify` pone algo escrito. Más adentro ya son
                # datos medidos (`metrics` lleva por clave el NOMBRE de
                # cada métrica, que es del modelo y no de este fichero).
                claves.update(etapa)
                for ficha in etapa.get("artifacts") or []:
                    claves.update(ficha)
        self.assertEqual(
            set(), claves - set(_CLAVES_DE_PROSA) - set(_CLAVES_DE_DATO),
            "clave del informe sin clasificar como prosa o como dato")


class ElCatalogoEnteroEstaTraducidoTest(unittest.TestCase):
    """Ningún abanico de paquetes toca las 50 y pico frases.

    Una que ninguna sonda alcance —la del enlace irresoluble, la del
    generador que no devuelve CSV— se quedaría sin traducir hasta que
    alguien se la encontrase en producción, que es donde peor se ve.
    """

    #: Con qué se rellenan los huecos: un dato, que no es de ningún idioma.
    _DATOS = {"fichero": "reproduce.json", "error": "OSError: nope",
              "version": "'999.0'", "conocidos": "['1.0']", "cita": "quoted",
              "faltan": "recipe, seed", "porque": "…", "claves": "warm_start",
              "alcances": "lo_que_sea", "declarado": "a" * 16, "actual": "b" * 16}

    def test_ni_una_frase_del_catalogo_esta_en_el_idioma_equivocado(self):
        for idioma in IDIOMAS:
            for clave, plantilla in _MOTIVOS[idioma].items():
                with self.subTest(idioma=idioma, clave=clave):
                    texto = plantilla.format(**self._DATOS)
                    self.assertEqual([], _intrusas(texto, idioma),
                                     f"{clave} [{idioma}]: {texto!r}")

    def test_los_dos_idiomas_dicen_LAS_MISMAS_cosas(self):
        """Una clave que falte en uno sale en producción en el otro."""
        self.assertEqual(set(_MOTIVOS["es"]), set(_MOTIVOS["en"]))

    def test_no_sobra_ninguna_frase_ni_falta_ninguna(self):
        """El catálogo y `verify.py` se miran de frente.

        Una entrada que ya nadie usa se queda vieja sin que se note, y una
        frase que `verify.py` escriba a mano no la ve ningún barrido de
        catálogo. *Dos sitios declarando lo mismo acaban divergiendo.*
        """
        import ast

        fuente = Path("matrixai/export/verify.py").read_text(encoding="utf-8")
        usadas: set[str] = set()
        for nodo in ast.walk(ast.parse(fuente)):
            # Con `ast` y no con una regex a propósito: una de las claves
            # se elige con un condicional DENTRO de la llamada
            # (`"p_sin_sha256" if … else "p_sin_ruta_ni_sha256"`) y una
            # regex de la primera comilla se dejaba la otra fuera —
            # midiendo, se veía como «frase que ya nadie escribe».
            if not (isinstance(nodo, ast.Call)
                    and isinstance(nodo.func, ast.Name)
                    and nodo.func.id == "motivo" and nodo.args):
                continue
            primero = nodo.args[0]
            ramas = ([primero.body, primero.orelse] if isinstance(primero, ast.IfExp)
                     else [primero])
            for rama in ramas:
                self.assertIsInstance(rama, ast.Constant,
                                      "una clave que no es literal no se puede medir")
                usadas.add(rama.value)
        self.assertEqual(set(), usadas - set(_MOTIVOS["es"]),
                         "verify.py pide claves que el catálogo no tiene")
        self.assertEqual(set(), set(_MOTIVOS["es"]) - usadas,
                         "el catálogo tiene frases que ya nadie escribe")


class LoQueNoEsProsaNoCambiaTest(unittest.TestCase):
    """Los `status` y los nombres de etapa son VALORES, no prosa.

    Quien encadena `verify && desplegar` los lee por programa: traducirlos
    convertiría un cambio de idioma en un cambio de contrato.
    """

    @classmethod
    def setUpClass(cls):
        cls.casos = _casos()

    def test_el_veredicto_es_EL_MISMO_en_los_dos_idiomas(self):
        for etiqueta, paquete, extras in self.casos:
            with self.subTest(caso=etiqueta):
                es = verify_package(paquete, locale="es", **extras)
                en = verify_package(paquete, locale="en", **extras)
                self.assertEqual(list(es["stages"]), list(en["stages"]))
                self.assertEqual(
                    {n: e["status"] for n, e in es["stages"].items()},
                    {n: e["status"] for n, e in en["stages"].items()})
                self.assertEqual(es["exit_code"], en["exit_code"])
                self.assertEqual(es["ok"], en["ok"])
                self.assertEqual(es["fully_checked"], en["fully_checked"])
                self.assertEqual(es["unchecked_stages"], en["unchecked_stages"])

    def test_los_nombres_de_etapa_no_se_traducen(self):
        informe = verify_package(_paquete(), locale="es")
        self.assertEqual(set(informe["stages"]), {"manifest", "R1", "training", "R3"})

    def test_un_dato_interpolado_es_EL_MISMO_en_los_dos(self):
        """Una ruta, una huella o un número no cambian de idioma."""
        d = _paquete()
        (d / "recipe.txt").write_text(_RECETA + "y otra línea\n")
        roto = {loc: verify_package(d, locale=loc)["stages"]["manifest"]["artifacts"][0]
                for loc in IDIOMAS}
        self.assertEqual(roto["es"]["artifact"], roto["en"]["artifact"])
        self.assertEqual(roto["es"]["path"], roto["en"]["path"])
        self.assertEqual(roto["es"]["expected"], roto["en"]["expected"])
        self.assertEqual(roto["es"]["found"], roto["en"]["found"])
        # Y el `problem`, que SÍ es prosa, no dice lo mismo en los dos.
        self.assertNotEqual(roto["es"]["problem"], roto["en"]["problem"])

    def test_la_version_desconocida_viaja_igual_dentro_del_motivo(self):
        """El motivo se traduce; el `999.0` que nombra, no."""
        d = _tocar(_paquete(), lambda m: m.__setitem__("schema_version", "999.0"))
        for locale in IDIOMAS:
            with self.subTest(locale=locale):
                etapa = verify_package(d, locale=locale)["stages"]["manifest"]
                self.assertIn("999.0", etapa["reason"])
                self.assertIn("1.0", etapa["reason"])

    def test_el_nombre_de_la_opcion_del_CLI_tampoco_se_traduce(self):
        """`--retrain` es lo que hay que teclear: traducirlo mandaría a
        escribir una opción que no existe."""
        for locale in IDIOMAS:
            with self.subTest(locale=locale):
                etapas = verify_package(_paquete(), locale=locale)["stages"]
                self.assertIn("--retrain", etapas["training"]["reason"])
                self.assertIn("--retrain", etapas["R3"]["reason"])

    def test_lo_que_el_PAQUETE_redacto_se_cita_sin_traducir(self):
        """No se le pone en la boca al paquete algo que no dijo: su motivo
        es un dato suyo, como su huella."""
        suyo = "Cannot regenerate the dataset: the recipe does not travel."
        d = _tocar(_paquete(), lambda m: m["verifiable"].__setitem__("r1", {
            "possible": False, "missing": ["recipe"], "reason": suyo}))
        for locale in IDIOMAS:
            with self.subTest(locale=locale):
                etapa = verify_package(d, locale=locale)["stages"]["R1"]
                self.assertEqual(etapa["status"], "INCOMPARABLE")
                self.assertIn(suyo, etapa["reason"])


class UnMotivoAUSENTE_SigueAusenteTest(unittest.TestCase):
    """*Un motivo ausente no es un motivo vacío.*

    Un `manifest: PASS` no trae motivo, y no puede empezar a traer una
    cadena vacía ni un «sin motivo»: la interfaz que la pintara estaría
    diciendo algo sobre el paquete que nadie ha comprobado.
    """

    def test_un_PASS_no_estrena_motivo_por_traducirlo(self):
        for locale in IDIOMAS:
            with self.subTest(locale=locale):
                etapa = verify_package(_paquete(), locale=locale)["stages"]["manifest"]
                self.assertEqual(etapa["status"], "PASS", etapa.get("reason"))
                self.assertNotIn("reason", etapa)

    def test_y_lo_que_NO_es_PASS_sigue_trayéndolo_en_los_dos_idiomas(self):
        for locale in IDIOMAS:
            informe = verify_package(_paquete(), locale=locale)
            for nombre, etapa in informe["stages"].items():
                if etapa["status"] == "PASS":
                    continue
                with self.subTest(locale=locale, etapa=nombre):
                    self.assertTrue(etapa.get("reason", "").strip(),
                                    f"{nombre} sin motivo en {locale}")


class ElIdiomaSePideYSeRESPETATest(unittest.TestCase):
    def test_por_defecto_es_espanol_como_el_resto_del_core(self):
        """El patrón de la casa (`playground.py`): `locale="es"`."""
        etapa = verify_package(_paquete())["stages"]["training"]
        self.assertEqual([], _intrusas(etapa["reason"], "es"))

    def test_un_idioma_que_no_existe_cae_al_espanol_y_no_revienta(self):
        """Fallo cerrado hacia el idioma de la casa, sin excepción: un
        informe que revienta por un `locale` raro no dice nada del
        paquete."""
        for raro in (None, "", "  ", "klingon", "ES", " En "):
            with self.subTest(locale=raro):
                etapa = verify_package(_paquete(), locale=raro)["stages"]["training"]
                self.assertTrue(etapa["reason"])

    def test_ES_y_En_con_mayusculas_son_el_mismo_idioma(self):
        self.assertEqual(
            verify_package(_paquete(), locale="ES")["stages"]["training"]["reason"],
            verify_package(_paquete(), locale="es")["stages"]["training"]["reason"])
        self.assertEqual(
            verify_package(_paquete(), locale=" En ")["stages"]["training"]["reason"],
            verify_package(_paquete(), locale="en")["stages"]["training"]["reason"])


class ElCLI_PuedePedirElIdiomaTest(unittest.TestCase):
    """`matrixai verify --locale es|en`.

    El defecto del CLI es `en` y NO el `es` del core, a propósito: su
    ayuda, sus errores y la ficha del Space que lo ejecuta están en
    inglés, y un informe en español debajo de una ayuda en inglés es
    media herramienta traducida.
    """

    def _correr(self, *argv: str) -> tuple[int, str]:
        # `main()` lee de `sys.argv` (no acepta lista): se sustituye, que
        # es la misma puerta por la que entra quien lo teclea.
        import contextlib
        import io
        import sys as _sys

        from matrixai.cli import main
        salida = io.StringIO()
        viejo = _sys.argv
        _sys.argv = ["matrixai", "verify", *argv]
        try:
            with contextlib.redirect_stdout(salida):
                codigo = main()
        finally:
            _sys.argv = viejo
        return codigo, salida.getvalue()

    def test_el_defecto_del_CLI_sigue_siendo_ingles(self):
        codigo, texto = self._correr(str(_paquete()))
        self.assertEqual(codigo, 3)
        self.assertIn("--retrain", texto)
        self.assertEqual([], _intrusas(texto.replace("NOT_RUN", ""), "en"))

    def test_y_se_le_puede_pedir_espanol(self):
        codigo, texto = self._correr(str(_paquete()), "--locale", "es")
        self.assertEqual(codigo, 3)
        self.assertIn("--retrain", texto)
        self.assertEqual([], _intrusas(texto, "es"))

    def test_los_status_y_las_etapas_salen_igual_en_los_dos(self):
        """Lo que un guion lee del texto no cambia con el idioma."""
        _, en = self._correr(str(_paquete()), "--locale", "en")
        _, es = self._correr(str(_paquete()), "--locale", "es")
        for marca in ("manifest", "R1", "training", "R3", "PASS", "NOT_RUN",
                      "INCOMPARABLE"):
            with self.subTest(marca=marca):
                self.assertIn(marca, en)
                self.assertIn(marca, es)

    def test_y_el_codigo_de_salida_tampoco(self):
        d = _paquete()
        (d / "recipe.txt").write_text(_RECETA + "y otra línea\n")
        self.assertEqual(self._correr(str(d), "--locale", "en")[0],
                         self._correr(str(d), "--locale", "es")[0])

    def test_el_json_tambien_respeta_el_idioma(self):
        _, texto = self._correr(str(_paquete()), "--locale", "es", "--json")
        informe = json.loads(texto)
        for donde, frase in _prosa(informe["stages"], "stages"):
            with self.subTest(donde=donde):
                self.assertEqual([], _intrusas(frase, "es"))


if __name__ == "__main__":
    unittest.main()
