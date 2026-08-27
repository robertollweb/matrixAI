"""87-C2 — ATESTIGUAR LA EVALUACIÓN DE UN MODELO AJENO.

Se le da un modelo que MatrixAI **no ha entrenado**, unos datos y una métrica;
corre la evaluación y ata el número a los digests del modelo y de los datos en
un recibo verificable.

Lo que hace que esto valga algo es lo que NO promete, y por eso está en el
recibo y no solo en la documentación:

* **No es prueba de entrenamiento.** Un modelo copiado de otro sitio produce el
  mismo recibo que uno propio, y el recibo lo dice con esas palabras.
* **No dice que el modelo sea bueno**: dice qué salió al medirlo así.
* **Techo A1/A2**: se firma como cualquier recibo —hoy HMAC, consistencia y no
  autenticidad—, y sin firma es **A0**, que también se dice.
"""
from __future__ import annotations

import sys
import unittest
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory

RAIZ_DEL_REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).parent))

from matrixai.export.attest import METRICAS, AtestacionImposible, atestiguar  # noqa: E402
from matrixai.pipelines.receipt import (  # noqa: E402
    firmar_recibo,
    nivel_del_recibo,
    problemas_de_esquema,
    verificar_recibo,
)
from test_c87_c1_ejecutor_onnx import _modelo_onnx  # noqa: E402

_HAS = util.find_spec("onnxruntime") is not None and util.find_spec("onnx") is not None


def _datos(d: Path, filas: int = 10, sucias: int = 0) -> Path:
    ruta = d / "eval.csv"
    lineas = ["a,b,c,y"]
    for i in range(filas):
        a, b = i / filas, (filas - i) / filas
        lineas.append(f"{a},{b},0.5,{1 if b > a else 0}")
    for _ in range(sucias):
        lineas.append("no,es,un,numero")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class AtestiguaLoQueMideTest(unittest.TestCase):
    def test_mide_y_ata_el_numero_a_los_dos_digests(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, digest = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d))
        self.assertEqual(recibo["metrics"][0]["name"], "accuracy")
        self.assertEqual(recibo["models"][0]["digest"], f"sha256:{digest}")
        # Y la métrica dice SOBRE QUÉ datos se midió.
        self.assertEqual(recibo["metrics"][0]["dataset_sha256"],
                         recibo["dataset"]["digest"].removeprefix("sha256:"))

    def test_dice_cuantas_filas_MIDIO_y_cuantas_habia(self):
        """Una fila que no es numérica no se convierte en un cero: se salta, y
        el recibo dice sobre cuántas se midió de verdad."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d, filas=10, sucias=3))
        self.assertEqual(recibo["dataset"]["rows_total"], 13)
        self.assertEqual(recibo["dataset"]["rows_measured"], 10)

    def test_el_recibo_declara_QUE_NO_demuestra(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d))
        self.assertIn("not trained by MatrixAI", recibo["models"][0]["provenance"])
        self.assertIn("copied model", recibo["evidence"]["does_not_attest"])

    def test_es_un_recibo_de_VERDAD_y_pasa_su_esquema(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d))
            recibo["created_at"] = "2026-08-25T10:00:00Z"
        self.assertEqual(problemas_de_esquema(recibo), [])

    def test_firmado_verifica_y_sube_de_A0(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d))
            recibo["created_at"] = "2026-08-25T10:00:00Z"
            sobre = firmar_recibo(recibo, clave=b"k", key_id="k1")
        self.assertTrue(verificar_recibo(sobre, clave=b"k")["ok"])
        self.assertNotEqual(nivel_del_recibo(sobre), "A0")

    def test_el_mismo_modelo_y_los_mismos_datos_dan_el_MISMO_id(self):
        """Sin eso, dos atestaciones de lo mismo no se podrían comparar."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            datos = _datos(d)
            self.assertEqual(atestiguar(modelo, datos)["receipt_id"],
                             atestiguar(modelo, datos)["receipt_id"])


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class LoQueNoSePuedeMedirNoSeInventaTest(unittest.TestCase):
    def test_una_metrica_que_no_sabe_medir_se_contesta_con_las_que_hay(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, _datos(d), metrica="f1_macro")
        self.assertIn("accuracy", str(e.exception))
        self.assertIn("mae", str(e.exception))

    def test_un_CSV_con_otras_columnas_no_se_recorta_ni_se_rellena(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            ruta = d / "otro.csv"
            ruta.write_text("a,b,y\n0.1,0.2,1\n", encoding="utf-8")
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, ruta)
        self.assertIn("no se recorta ni se rellena", str(e.exception))

    def test_una_columna_objetivo_que_no_existe_se_dice(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, _datos(d), columna="no_existe")
        self.assertIn("no_existe", str(e.exception))

    def test_una_CABECERA_VACIA_se_dice_en_vez_de_reventar(self):
        """Lo cazó el agente que construía la pantalla, sondeando el endpoint:
        un CSV cuya primera línea está en blanco daba `IndexError: list index
        out of range` y quien lo recibía se llevaba un «error inesperado» en
        vez de un motivo. Un fallo que se puede nombrar no se deja como traza.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            for contenido in ("\n0.1,0.2,0.3,1\n", ",,,\n0.1,0.2,0.3,1\n"):
                ruta = d / "sin_cabecera.csv"
                ruta.write_text(contenido, encoding="utf-8")
                with self.subTest(contenido=contenido):
                    with self.assertRaises(AtestacionImposible) as e:
                        atestiguar(modelo, ruta)
                    self.assertIn("primera línea", str(e.exception))

    def test_un_CSV_sin_filas_no_produce_un_cero(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            ruta = d / "vacio.csv"
            ruta.write_text("a,b,c,y\n", encoding="utf-8")
            with self.assertRaises(AtestacionImposible):
                atestiguar(modelo, ruta)

    def test_las_metricas_que_sabe_medir_estan_declaradas(self):
        self.assertEqual(METRICAS, ("accuracy", "mae"))


if __name__ == "__main__":
    unittest.main()


# ── LO QUE ENCONTRÓ LA AUDITORÍA EXTERNA (2026-08-25) ───────────────────────

def _sigmoide(d: Path) -> Path:
    """Un clasificador de UNA salida: `sigmoid(2a + 2b - 2)`.

    Es el caso del hallazgo 1: con una sola posición, `argmax` es siempre 0.
    """
    import onnx
    from onnx import TensorProto, helper

    W = helper.make_tensor("W", TensorProto.FLOAT, [2, 1], [2.0, 2.0])
    B = helper.make_tensor("B", TensorProto.FLOAT, [1], [-2.0])
    g = helper.make_graph(
        [helper.make_node("MatMul", ["entrada", "W"], ["z0"]),
         helper.make_node("Add", ["z0", "B"], ["z1"]),
         helper.make_node("Sigmoid", ["z1"], ["salida"])],
        "sig",
        [helper.make_tensor_value_info("entrada", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("salida", TensorProto.FLOAT, [1, 1])], [W, B])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    ruta = d / "sigmoide.onnx"
    onnx.save(m, str(ruta))
    return ruta


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class UnaSalidaNoEsUnaListaDeClasesTest(unittest.TestCase):
    """HALLAZGO 1 DE LA AUDITORÍA EXTERNA (BLOQUEANTE) — y el peor de todos,
    porque **el número equivocado se ata a los digests y se puede FIRMAR**.

    Reproducido: con un sigmoide de una salida, la exactitud correcta con umbral
    0,5 era **1,0** y `attest` atestiguaba **0,5**, porque `argmax` de un vector
    de un elemento es siempre 0 y el modelo «predecía» siempre la clase 0.
    """

    def _datos(self, d: Path) -> Path:
        ruta = d / "eval.csv"
        filas = ["a,b,y", "0.1,0.1,0", "0.2,0.0,0", "0.9,0.9,1",
                 "1.0,0.8,1", "0.0,0.3,0", "0.95,1.0,1"]
        ruta.write_text("\n".join(filas) + "\n", encoding="utf-8")
        return ruta

    def test_un_sigmoide_se_mide_con_UMBRAL_no_con_argmax(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            recibo = atestiguar(_sigmoide(d), self._datos(d))
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)

    def test_y_un_softmax_de_dos_clases_sigue_midiendose_con_argmax(self):
        """El arreglo no puede consistir en cambiar el criterio para todos."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d))
        self.assertEqual(recibo["metrics"][0]["name"], "accuracy")
        self.assertGreaterEqual(recibo["metrics"][0]["value"], 0.0)


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class ElEntornoSeREGISTRATest(unittest.TestCase):
    """HALLAZGO 8: `evidence.attests` decía «la evaluación, el ENTORNO y las
    entradas» y el recibo no traía ni una línea de entorno. Una afirmación
    mayor que la evidencia guardada."""

    def test_el_recibo_trae_el_entorno_que_dice_atestiguar(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, _ = _modelo_onnx(d)
            recibo = atestiguar(modelo, _datos(d))
        entorno = recibo["environment"]
        self.assertTrue(entorno["onnxruntime"])
        self.assertTrue(entorno["providers"])
        self.assertTrue(entorno["python"])
        self.assertIn("environment", recibo["evidence"]["attests"])


def _modelo_con(tipo_entrada: str, forma: list, ruta: Path) -> None:
    """Un modelo lineal de 3 entradas, con el tipo y la forma que se pidan."""
    import onnx
    from onnx import TensorProto, helper

    tipo = {"float": TensorProto.FLOAT, "int64": TensorProto.INT64}[tipo_entrada]
    W = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [1.0, 0.0, 0.0])
    nodos = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
    if tipo == TensorProto.INT64:
        nodos = [helper.make_node("Cast", ["X"], ["Xf"], to=TensorProto.FLOAT),
                 helper.make_node("MatMul", ["Xf", "W"], ["Y"])]
    g = helper.make_graph(
        nodos, "g",
        [helper.make_tensor_value_info("X", tipo, forma)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 1])], [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.save(m, str(ruta))


class AttestAlimentaCOMO_EL_EJECUTORTest(unittest.TestCase):
    """`attest` alimenta el modelo con el mismo código que el ejecutor del motor.

    2ª auditoría externa (2026-08-25), hallazgo 1 residual: **«la garantía de C1
    no alcanza C2»**. C1 ya comprobaba tipos y ejes dinámicos; C2 tenía su propia
    copia y hacía justo lo que C1 había dejado de hacer.
    """

    def test_un_eje_DINAMICO_ya_no_se_confunde_con_una_entrada(self):
        # `[1, "features"]`: el `1` es el LOTE. El camino viejo se quedaba con
        # los ejes enteros —solo el `1`—, lo comparaba con las 3 columnas del
        # CSV y RECHAZABA un modelo perfectamente válido.
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = d / "dinamico.onnx"
            _modelo_con("float", [1, "features"], modelo)
            datos = d / "d.csv"
            datos.write_text("a,b,c,y\n1,2,3,1\n0,5,6,0\n", encoding="utf-8")
            recibo = atestiguar(modelo, datos, columna="y", metrica="accuracy")
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        # Y no se afirma por omisión: con ejes dinámicos no se pudo contrastar,
        # y el recibo lo DICE en vez de callar.
        spec = recibo["models"][0]["input_spec"]
        self.assertEqual(spec["shape_check"], "no declarada")
        self.assertIsNone(spec["features_declared"])

    def test_un_modelo_de_ENTEROS_no_traga_un_3_7_en_silencio(self):
        # ONNX Runtime acepta el float, lo trunca a 3 y devuelve un número: el
        # acierto que saliera de ahí iría al recibo, atado a los digests y
        # firmable, como si se hubiera medido sobre el dato del CSV.
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = d / "entero.onnx"
            _modelo_con("int64", ["N", 3], modelo)
            datos = d / "d.csv"
            datos.write_text("a,b,c,y\n3.7,2,3,1\n0,5,6,0\n", encoding="utf-8")
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, datos, columna="y", metrica="accuracy")
        self.assertIn("truncaría en silencio", str(e.exception))

    def test_y_el_recibo_dice_CON_QUE_se_alimento(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = d / "entero.onnx"
            _modelo_con("int64", ["N", 3], modelo)
            datos = d / "d.csv"
            datos.write_text("a,b,c,y\n1,2,3,1\n0,5,6,0\n", encoding="utf-8")
            recibo = atestiguar(modelo, datos, columna="y", metrica="accuracy")
        spec = recibo["models"][0]["input_spec"]
        self.assertEqual(spec["type"], "tensor(int64)")
        self.assertEqual(spec["features_declared"], 3)
        self.assertEqual(spec["shape_check"], "declarada")


class QuienLoPideNoLoSABE_ESTE_MODULOTest(unittest.TestCase):
    """Medido conduciendo la aplicación (2026-08-26).

    El respaldo de `actor` era `"matrixai attest"` —el nombre del comando del
    TERMINAL—, así que un recibo emitido desde la PANTALLA del Studio decía que
    lo había pedido el CLI. En un documento cuyo asunto entero es la
    procedencia, eso es el documento afirmando algo que no pasó.

    Lo peor: la pantalla YA lo tenía escrito —«el suyo nombra el terminal, que
    no es por donde ha entrado esto»—. Advertir de un defecto no es no tenerlo.
    """

    def _recibo(self, **kw):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _sigmoide(d)
            datos = d / "d.csv"
            datos.write_text("a,b,y\n0.1,0.1,0\n0.9,0.9,1\n", encoding="utf-8")
            return atestiguar(modelo, datos, columna="y", **kw)

    def test_sin_actor_el_recibo_dice_que_NO_CONSTA(self):
        recibo = self._recibo()
        self.assertEqual(recibo["subject"]["actor"], "no consta quién lo pidió")

    def test_y_NO_nombra_el_terminal(self):
        # Un canal inventado se lee como un hecho: quien verifique el recibo no
        # tiene forma de saber que ese nombre lo puso un respaldo.
        self.assertNotIn("matrixai attest", self._recibo()["subject"]["actor"])

    def test_quien_llama_SI_puede_decirlo(self):
        self.assertEqual(self._recibo(actor="MatrixAI Studio (pantalla)")["subject"]["actor"],
                         "MatrixAI Studio (pantalla)")

    def test_el_CLI_pone_SU_nombre_porque_ahi_es_verdad(self):
        fuente = (RAIZ_DEL_REPO / "matrixai" / "cli.py").read_text(encoding="utf-8")
        self.assertIn('actor=args.actor or "matrixai attest"', fuente)


class ElPROPOSITO_DE_RESPALDO_HablaSuIdiomaTest(unittest.TestCase):
    """Medido el 2026-08-26 capturando las pantallas del manual.

    El propósito que escribe el core cuando nadie declara uno estaba fijo en
    castellano, así que un recibo emitido desde una pantalla en inglés decía
    «Purpose: medir accuracy de un modelo externo» debajo de un «LEVEL A0»
    traducido. *Lo que redacta el core se traduce EN EL CORE.*
    """

    def _proposito(self, **kw) -> str:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            datos = d / "d.csv"
            datos.write_text("a,b,y\n0.1,0.1,0\n0.9,0.9,1\n", encoding="utf-8")
            return atestiguar(_sigmoide(d), datos, columna="y", **kw)["subject"]["purpose"]

    def test_en_castellano_por_defecto(self):
        self.assertEqual(self._proposito(), "medir accuracy de un modelo externo")

    def test_y_en_ingles_cuando_se_pide(self):
        self.assertEqual(self._proposito(locale="en"),
                         "measure accuracy of an external model")

    def test_un_idioma_que_no_hay_cae_al_de_casa(self):
        # Ni se inventa una traducción ni se deja la clave cruda.
        self.assertEqual(self._proposito(locale="fr"),
                         "medir accuracy de un modelo externo")

    def test_lo_que_DECLARE_quien_lo_pide_manda_en_los_dos(self):
        for idioma in ("es", "en"):
            with self.subTest(idioma=idioma):
                self.assertEqual(
                    self._proposito(proposito="auditoría interna", locale=idioma),
                    "auditoría interna")

    def test_la_metrica_va_dentro_de_la_frase(self):
        self.assertIn("mae", self._proposito(metrica="mae", locale="en"))
