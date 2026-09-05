"""102-C0 — `attest` ATESTIGUA LO QUE EL MODELO DICE, NO SU PRIMERA SALIDA.

EL DEFECTO, medido el 2026-09-05 sobre el core 1.7.0 publicado. `atestiguar`
tomaba `sesion.run(None, ...)[0]` —la **primera** salida del grafo— y la
umbralizaba a 0,5. Los conversores estándar (skl2onnx, onnxmltools) ponen
`label` primero, así que:

* con un modelo **binario** cuadraba **por accidente**: la etiqueta ya es 0 o 1
  y umbralizarla a 0,5 devuelve la misma etiqueta;
* con **tres clases** no. Con la regresión logística de `tests/data/`, la
  exactitud real es **0,9733** y `attest` atestiguaba **0,6667**: la clase 2
  pasaba el umbral y salía 1.

Y lo que lo hace grave no es el número: es que ese número queda **atado a los
digests del modelo y de los datos** y **se puede firmar**. Un recibo cuyo
asunto entero es la procedencia afirmando algo que no pasó.

LO QUE SE COMPRUEBA AQUÍ, que es el criterio del corte: binaria, multiclase,
regresión, clases de texto, clases no consecutivas y orden invertido **coinciden
con la inferencia nativa**; el caso documentado se reproduce con su dataset y su
modelo exactos; forzar una salida incorrecta falla; y el recibo dice qué salida
se leyó y con qué clases.
"""
from __future__ import annotations

import hashlib
import unittest
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.export.attest import AtestacionImposible, atestiguar

_HAS = util.find_spec("onnxruntime") is not None and util.find_spec("onnx") is not None
DATOS = Path(__file__).resolve().parent / "data"

#: El caso documentado en `documentacion/100_ANALISIS_CAMBIO_DE_PARADIGMA.md`
#: §0.11. El número va atado al **digest del fichero**, no a «un iris»: si
#: alguien regenera el modelo, esta prueba lo dice en vez de exigirle 0,9733 a
#: una regresión distinta. El fixture se generó una vez con scikit-learn
#: (`LogisticRegression(max_iter=1000)` sobre `load_iris`) y se versiona porque
#: la suite del core no depende de scikit-learn.
IRIS_ONNX = DATOS / "iris_3clases.onnx"
IRIS_CSV = DATOS / "iris.csv"
IRIS_DIGEST = "7c11e9fb9e968c4a8cc358b5aad8631d2ed1ece7432b8f743400c6147a0cf25d"
EXACTITUD_DOCUMENTADA = 0.9733333333333334
LO_QUE_ATESTIGUABA_ANTES = 0.6666666666666666


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _guardar(destino: Path, grafo, opsets=None) -> Path:
    import onnx
    from onnx import helper

    modelo = helper.make_model(
        grafo, opset_imports=opsets or [helper.make_opsetid("", 13)])
    modelo.ir_version = 9
    onnx.save(modelo, str(destino))
    return destino


def _clasificador(destino: Path, *, clases: list, con_etiqueta: bool = True,
                  tipo_etiqueta: str = "int64", etiqueta_primero: bool = True) -> Path:
    """Un clasificador de K clases: `Softmax(X @ W)`, con `label` delante.

    Es la forma que producen los conversores: la etiqueta primero y las
    probabilidades después. K se deduce de `clases`.
    """
    from onnx import TensorProto, helper

    k = len(clases)
    # Una diagonal: la clase i gana cuando la columna i es la mayor. Así el
    # resultado esperado se puede escribir a mano en el CSV.
    pesos = [1.0 if i == j else 0.0 for i in range(k) for j in range(k)]
    W = helper.make_tensor("W", TensorProto.FLOAT, [k, k], pesos)
    nodos = [helper.make_node("MatMul", ["X", "W"], ["z"]),
             helper.make_node("Softmax", ["z"], ["probabilities"], axis=1)]
    salidas = []
    if con_etiqueta:
        etiquetas = helper.make_tensor(
            "etiquetas",
            TensorProto.STRING if tipo_etiqueta == "string" else TensorProto.INT64,
            [k],
            [str(c).encode() for c in clases] if tipo_etiqueta == "string"
            else [int(c) for c in clases])
        nodos += [
            helper.make_node("ArgMax", ["probabilities"], ["pos"], axis=1, keepdims=0),
            helper.make_node("Gather", ["etiquetas", "pos"], ["label"], axis=0)]
        tipo = TensorProto.STRING if tipo_etiqueta == "string" else TensorProto.INT64
        salidas.append(helper.make_tensor_value_info("label", tipo, ["N"]))
        iniciales = [W, etiquetas]
    else:
        iniciales = [W]
    probabilidades = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, ["N", k])
    salidas = salidas + [probabilidades] if etiqueta_primero else [probabilidades] + salidas
    g = helper.make_graph(nodos, "clf",
                          [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", k])],
                          salidas, iniciales)
    return _guardar(destino, g)


def _zipmap(destino: Path, clases: list) -> Path:
    """Un clasificador cuya salida es un MAPA por clase: `{clase: probabilidad}`.

    Es lo que produce skl2onnx por defecto. Las clases vienen DENTRO, así que no
    hay ninguna posición que traducir — y por eso este caso caza el defecto de
    confundir un índice con una etiqueta.
    """
    from onnx import TensorProto, helper

    k = len(clases)
    pesos = [1.0 if i == j else 0.0 for i in range(k) for j in range(k)]
    W = helper.make_tensor("W", TensorProto.FLOAT, [k, k], pesos)
    mapa = helper.make_map_type_proto(
        TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
    g = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["z"]),
         helper.make_node("Softmax", ["z"], ["p"], axis=1),
         helper.make_node("ZipMap", ["p"], ["probabilities"], domain="ai.onnx.ml",
                          classlabels_int64s=[int(c) for c in clases])],
        "zip",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", k])],
        [helper.make_value_info("probabilities", helper.make_sequence_type_proto(mapa))],
        [W])
    return _guardar(destino, g, [helper.make_opsetid("", 13),
                                 helper.make_opsetid("ai.onnx.ml", 2)])


def _regresor(destino: Path) -> Path:
    """Un regresor: un solo número por fila, y nada que diga que es otra cosa."""
    from onnx import TensorProto, helper

    W = helper.make_tensor("W", TensorProto.FLOAT, [2, 1], [1.0, 1.0])
    g = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])], "reg",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 1])], [W])
    return _guardar(destino, g)


def _csv(destino: Path, cabecera: str, filas: list[str]) -> Path:
    destino.write_text(cabecera + "\n" + "\n".join(filas) + "\n", encoding="utf-8")
    return destino


def _exactitud_nativa(modelo: Path, csv: Path) -> float:
    """Lo que da el modelo ejecutado a pelo, sin pasar por MatrixAI.

    Es la referencia: `attest` tiene que coincidir con ESTO. Comparar contra un
    número escrito a mano solo probaría que el número está escrito a mano.
    """
    import csv as _csv

    import numpy as np
    import onnxruntime

    sesion = onnxruntime.InferenceSession(str(modelo), providers=["CPUExecutionProvider"])
    entrada = sesion.get_inputs()[0].name
    salidas = sesion.get_outputs()
    # LA REFERENCIA BUSCA LA ETIQUETA, ESTÉ DONDE ESTÉ. Si mirara siempre la
    # posición 0 daría por buena justo la costumbre que este corte cierra.
    cual = next((i for i, o in enumerate(salidas)
                 if o.type in ("tensor(int64)", "tensor(string)")), None)
    if cual is None:
        cual = next((i for i, o in enumerate(salidas) if o.type.startswith("seq(map")), 0)
    filas = list(_csv.reader(csv.read_text(encoding="utf-8").splitlines()))[1:]
    aciertos = 0
    for fila in filas:
        vector = np.array([[float(v) for v in fila[:-1]]], dtype=np.float32)
        bruto = sesion.run(None, {entrada: vector})
        primera = bruto[cual]
        if salidas[cual].type.startswith("seq(map"):
            predicha = max(primera[0].items(), key=lambda kv: kv[1])[0]
        elif salidas[cual].type in ("tensor(int64)", "tensor(string)"):
            predicha = primera[0]
        else:  # solo probabilidades: la posición del máximo ES la clase 0..K-1
            predicha = int(np.argmax(primera[0]))
        if isinstance(predicha, bytes):
            predicha = predicha.decode()
        aciertos += int(str(predicha) == str(fila[-1]))
    return aciertos / len(filas)


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class ElCasoDocumentadoTest(unittest.TestCase):
    """El de `100 §0.11`, con su dataset y su modelo EXACTOS."""

    def test_el_fixture_es_el_documentado_y_no_otro_iris(self):
        """El número va atado al digest del fichero.

        Si alguien regenera el modelo, esta prueba lo dice en vez de exigirle
        0,9733 a una regresión que no es la medida. *No exigir el número
        documentado a cualquier iris reentrenado* es del propio contrato.
        """
        self.assertTrue(IRIS_ONNX.is_file(), "falta el fixture del caso documentado")
        self.assertEqual(
            _sha256(IRIS_ONNX), IRIS_DIGEST,
            "el modelo del caso documentado ha cambiado: la exactitud de abajo es la "
            "de ESTE fichero, no la de cualquier iris. Vuelve a medirla antes de "
            "tocar el número")

    def test_atestigua_la_exactitud_REAL_y_no_la_de_la_primera_salida(self):
        recibo = atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase")
        self.assertEqual(recibo["metrics"][0]["value"], EXACTITUD_DOCUMENTADA)
        # Y el defecto no vuelve por otro camino.
        self.assertNotEqual(recibo["metrics"][0]["value"], LO_QUE_ATESTIGUABA_ANTES)

    def test_coincide_con_la_inferencia_nativa(self):
        recibo = atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase")
        self.assertEqual(recibo["metrics"][0]["value"],
                         _exactitud_nativa(IRIS_ONNX, IRIS_CSV))

    def test_el_recibo_dice_QUE_SALIDA_leyo_y_con_que_clases(self):
        recibo = atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase")
        spec = recibo["models"][0]["output_spec"]
        self.assertEqual(spec["name"], "label")
        self.assertEqual(spec["kind"], "etiqueta")
        self.assertEqual(spec["type"], "tensor(int64)")
        self.assertIn("inferencia nativa", spec["chosen_because"])
        self.assertEqual(spec["classes_source"], "las da el modelo")


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class CoincideConLaInferenciaNativaTest(unittest.TestCase):
    """Binaria, multiclase, texto, no consecutivas, orden invertido y regresión."""

    def test_binaria_con_etiqueta_y_probabilidades(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "bin.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y",
                       ["3,0,0", "0,3,1", "2,1,0", "1,2,1", "5,0,0"])
            recibo = atestiguar(modelo, csv)
            self.assertEqual(recibo["metrics"][0]["value"], _exactitud_nativa(modelo, csv))
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)

    def test_multiclase_sin_etiqueta_deduce_las_clases_de_la_columna_y_lo_DICE(self):
        """Sin salida de etiquetas hay que traducir una posición a una clase.

        Con la columna objetivo trayendo `0..K-1` la deducción es comprobable, y
        el recibo dice que se dedujo — no la presenta como declarada.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1, 2], con_etiqueta=False)
            csv = _csv(d / "d.csv", "a,b,c,y",
                       ["3,0,0,0", "0,3,0,1", "0,0,3,2", "5,1,1,0"])
            recibo = atestiguar(modelo, csv)
            self.assertEqual(recibo["metrics"][0]["value"], _exactitud_nativa(modelo, csv))
        spec = recibo["models"][0]["output_spec"]
        self.assertEqual(spec["kind"], "probabilidades")
        self.assertEqual(spec["classes"], [0, 1, 2])
        self.assertIn("deducidas", spec["classes_source"])

    def test_la_salida_buena_NO_siempre_es_la_PRIMERA(self):
        """Un modelo que declara `probabilities` antes que `label`.

        Es el defecto original por su otra cara: no era que la primera salida
        estuviera mal interpretada, es que **la posición no significa nada**.
        Aquí leer `run(...)[0]` devuelve un vector de probabilidades donde se
        esperaba una etiqueta, y con clases `[7, 9]` ni siquiera el índice
        coincidiría por casualidad con la clase.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "inv.onnx", clases=[7, 9],
                                   etiqueta_primero=False)
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,7", "0,3,9", "5,1,7"])
            import onnxruntime
            orden = [o.name for o in onnxruntime.InferenceSession(
                str(modelo), providers=["CPUExecutionProvider"]).get_outputs()]
            self.assertEqual(orden, ["probabilities", "label"])
            recibo = atestiguar(modelo, csv)
            self.assertEqual(recibo["metrics"][0]["value"], _exactitud_nativa(modelo, csv))
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        self.assertEqual(recibo["models"][0]["output_spec"]["name"], "label")

    def test_clases_de_TEXTO(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "t.onnx", clases=["setosa", "virginica"],
                                   tipo_etiqueta="string")
            csv = _csv(d / "d.csv", "a,b,clase",
                       ["3,0,setosa", "0,3,virginica", "4,1,setosa"])
            recibo = atestiguar(modelo, csv, columna="clase")
            self.assertEqual(recibo["metrics"][0]["value"], _exactitud_nativa(modelo, csv))
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)

    def test_clases_NO_CONSECUTIVAS_por_un_mapa_del_propio_modelo(self):
        """Un ZipMap con clases `[7, 9, 3]`: la posición 1 es la clase **9**.

        Es el caso que un `argmax` por índice contesta mal siempre: devolvería
        `1`, que aquí no es ninguna de las tres clases.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _zipmap(d / "z.onnx", clases=[7, 9, 3])
            csv = _csv(d / "d.csv", "a,b,c,y",
                       ["3,0,0,7", "0,3,0,9", "0,0,3,3", "1,5,1,9"])
            recibo = atestiguar(modelo, csv)
            self.assertEqual(recibo["metrics"][0]["value"], _exactitud_nativa(modelo, csv))
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        spec = recibo["models"][0]["output_spec"]
        self.assertIn("mapa por clase", spec["classes_source"])

    def test_clases_no_consecutivas_SIN_declarar_se_niega_en_vez_de_adivinar(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[7, 9, 3], con_etiqueta=False)
            csv = _csv(d / "d.csv", "a,b,c,y", ["3,0,0,7", "0,3,0,9", "0,0,3,3"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv)
        self.assertIn("un índice no es una etiqueta", str(e.exception))

    def test_y_declarando_el_ORDEN_INVERTIDO_sale_lo_que_sale_de_verdad(self):
        """Las mismas probabilidades con las clases al revés dan otra cosa.

        Si el orden no se usara, los dos casos darían lo mismo — y uno de los
        dos estaría mal sin que nada lo dijera.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1], con_etiqueta=False)
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0", "0,3,1", "4,0,0"])
            derecho = atestiguar(modelo, csv, clases=[0, 1])["metrics"][0]["value"]
            invertido = atestiguar(modelo, csv, clases=[1, 0])["metrics"][0]["value"]
        self.assertEqual(derecho, 1.0)
        self.assertEqual(invertido, 0.0)

    def test_regresion(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,2", "2,2,4", "0,1,2"])
            recibo = atestiguar(modelo, csv, metrica="mae")
        # Predice a+b: 2, 4, 1 frente a 2, 4, 2 → error medio 1/3.
        self.assertAlmostEqual(recibo["metrics"][0]["value"], 1 / 3)
        self.assertEqual(recibo["models"][0]["output_spec"]["kind"], "valor")


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class LoQueNoSE_SABE_SE_PIDE_NoSeAdivinaTest(unittest.TestCase):
    """El corazón del corte: **un escalar no se umbraliza por su forma.**"""

    def test_un_escalar_sin_semantica_pide_el_mapa_de_salida(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,1", "0,0,0"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv, metrica="accuracy")
        mensaje = str(e.exception)
        self.assertIn("puede ser una regresión, una puntuación o una probabilidad",
                      mensaje)
        self.assertIn("Declara el mapa de salida", mensaje)

    def test_pero_un_SIGMOIDE_lo_declara_el_grafo_y_no_hace_falta_pedirlo(self):
        """No es adivinar por la forma: es leer el operador que la produce."""
        from onnx import TensorProto, helper
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            W = helper.make_tensor("W", TensorProto.FLOAT, [2, 1], [2.0, 2.0])
            B = helper.make_tensor("B", TensorProto.FLOAT, [1], [-2.0])
            g = helper.make_graph(
                [helper.make_node("MatMul", ["X", "W"], ["z0"]),
                 helper.make_node("Add", ["z0", "B"], ["z1"]),
                 helper.make_node("Sigmoid", ["z1"], ["salida"])], "sig",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 2])],
                [helper.make_tensor_value_info("salida", TensorProto.FLOAT, ["N", 1])],
                [W, B])
            modelo = _guardar(d / "s.onnx", g)
            csv = _csv(d / "d.csv", "a,b,y", ["0.1,0.1,0", "0.9,0.9,1", "1,0.8,1"])
            recibo = atestiguar(modelo, csv)
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        spec = recibo["models"][0]["output_spec"]
        self.assertEqual(spec["kind"], "probabilidad_positiva")
        self.assertIn("lo declara el grafo", spec["chosen_because"])

    def test_forzar_una_salida_que_no_existe_dice_las_que_hay(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0", "0,3,1"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv, salida="no_existe")
        self.assertIn("label", str(e.exception))
        self.assertIn("probabilities", str(e.exception))

    def test_forzar_una_semantica_INCORRECTA_falla_en_vez_de_dar_un_numero(self):
        """Leer un vector de tres probabilidades «como si fuera una etiqueta»
        no puede devolver un número: tiene que decir que no."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1, 2], con_etiqueta=False)
            csv = _csv(d / "d.csv", "a,b,c,y", ["3,0,0,0", "0,3,0,1"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv, salida="probabilities", semantica="etiqueta")
        self.assertIn("se esperaba un valor por fila", str(e.exception))

    def test_una_semantica_que_no_existe_se_contesta_con_las_que_hay(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv, semantica="lo_que_sea")
        self.assertIn("etiqueta", str(e.exception))

    def test_declarar_MENOS_clases_de_las_que_devuelve_no_se_rellena(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1, 2], con_etiqueta=False)
            csv = _csv(d / "d.csv", "a,b,c,y", ["3,0,0,0", "0,3,0,1"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv, clases=[0, 1])
        self.assertIn("no se recorta ni se rellena", str(e.exception))


if __name__ == "__main__":
    unittest.main()
