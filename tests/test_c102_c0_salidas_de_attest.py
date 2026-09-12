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

from matrixai.export.attest import METRICAS, AtestacionImposible, atestiguar
from matrixai.export.onnx_salida import (
    SalidaOnnxAmbigua, elegir_salida, misma_etiqueta, resolver_clases)

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


def _clasificador_con_dos_etiquetas(destino: Path, *, clases: list) -> Path:
    """Auditoría externa 2026-09-09: DOS salidas de tipo etiqueta
    (`tensor(int64)`), `label_a` y `label_b`, ninguna más "la predicción"
    que la otra -- el caso exacto que reproducía `SalidaOnnxAmbigua`
    eligiendo la primera por orden en vez de rechazar."""
    from onnx import TensorProto, helper

    k = len(clases)
    pesos = [1.0 if i == j else 0.0 for i in range(k) for j in range(k)]
    W = helper.make_tensor("W", TensorProto.FLOAT, [k, k], pesos)
    etiquetas = helper.make_tensor("etiquetas", TensorProto.INT64, [k],
                                   [int(c) for c in clases])
    nodos = [
        helper.make_node("MatMul", ["X", "W"], ["z"]),
        helper.make_node("Softmax", ["z"], ["probabilities"], axis=1),
        helper.make_node("ArgMax", ["probabilities"], ["pos"], axis=1, keepdims=0),
        helper.make_node("Gather", ["etiquetas", "pos"], ["label_a"], axis=0),
        helper.make_node("Gather", ["etiquetas", "pos"], ["label_b"], axis=0),
    ]
    g = helper.make_graph(
        nodos, "clf_dos_etiquetas",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", k])],
        [helper.make_tensor_value_info("label_a", TensorProto.INT64, ["N"]),
         helper.make_tensor_value_info("label_b", TensorProto.INT64, ["N"])],
        [W, etiquetas])
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


def _dos_columnas_sin_declarar(destino: Path) -> Path:
    """`X @ W` con DOS columnas de salida y **ningún operador que diga qué son**.

    Es el caso crudo que la auditoría del 2026-09-11 dejó sin test: sin
    `Softmax`, sin `Sigmoid` y sin `ArgMax`, esas dos columnas pueden ser
    probabilidades, logits o las dos salidas de un regresor — y la anchura no
    distingue entre las tres.
    """
    from onnx import TensorProto, helper

    W = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
    g = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])], "crudo",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 2])], [W])
    return _guardar(destino, g)


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

    def test_dos_salidas_de_tipo_etiqueta_piden_el_mapa_en_vez_de_elegir_la_primera(self):
        """Auditoría externa 2026-09-09: con `label_a`/`label_b` (las dos
        `tensor(int64)`), la posición NO decide -- se rechaza en vez de
        leer `label_a` por venir primero en el grafo."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador_con_dos_etiquetas(d / "m.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0", "0,3,1"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv)
        mensaje = str(e.exception)
        self.assertIn("label_a", mensaje)
        self.assertIn("label_b", mensaje)
        self.assertIn("Declara el mapa de salida", mensaje)

    def test_pero_declarando_CUAL_de_las_dos_etiquetas_si_funciona(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador_con_dos_etiquetas(d / "m.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0", "0,3,1"])
            recibo = atestiguar(modelo, csv, salida="label_b", semantica="etiqueta")
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        self.assertEqual(recibo["models"][0]["output_spec"]["name"], "label_b")

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


# ── LO QUE ENCONTRÓ LA AUDITORÍA PROPIA DEL 2026-09-11, reparado el 09-12 ───

@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class ForzarUnaSalidaIncorrectaFALLATest(unittest.TestCase):
    """El criterio literal del corte: **«Forzar salida incorrecta falla»**.

    Y no fallaba por la puerta del NOMBRE. Reproducido el 2026-09-12 sobre el
    iris de tres clases, antes de tocar nada::

        atestiguar(iris, iris_csv, metrica="mae", salida="label")
        → 0.02666666666666667,  output_spec.kind = "etiqueta"

    Un MAE entre los códigos 0, 1 y 2 de tres especies de lirio. No mide
    ninguna distancia —entre «setosa» y «virginica» no hay distancia— y sin
    embargo sale un número pequeño, de los que tranquilizan, atado a los dos
    digests y **firmable**. `elegir_salida()` comprobaba que la salida
    existiera y que alguien dijera qué significa, pero no que eso que significa
    sirviera para lo que se está midiendo.

    LAS DOS MITADES. Si solo se probara que falla, lo pasaría una versión que
    falle siempre: por eso las dos salidas CORRECTAS están aquí abajo con su
    número.
    """

    def test_un_MAE_sobre_la_salida_de_ETIQUETAS_falla(self):
        with self.assertRaises(AtestacionImposible) as e:
            atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase", metrica="mae",
                       salida="label")
        mensaje = str(e.exception)
        self.assertIn("etiqueta", mensaje)
        self.assertIn("regresión", mensaje)
        self.assertIn("valor", mensaje)

    def test_y_ya_no_devuelve_aquel_0_0267(self):
        """El número medido antes de la reparación, escrito para que se vea que
        lo que se cierra es ESTO y no una posibilidad teórica."""
        try:
            recibo = atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase", metrica="mae",
                                salida="label")
        except AtestacionImposible:
            return
        self.fail(f"volvió a medir un MAE entre etiquetas de clase: "
                  f"{recibo['metrics'][0]['value']!r}")

    def test_pero_el_MAE_sobre_la_salida_NUMERICA_sigue_saliendo(self):
        """La otra mitad: forzar la salida BUENA no puede fallar."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,2", "2,2,4", "0,1,2"])
            recibo = atestiguar(modelo, csv, metrica="mae", salida="Y")
        self.assertAlmostEqual(recibo["metrics"][0]["value"], 1 / 3)
        self.assertEqual(recibo["models"][0]["output_spec"]["kind"], "valor")

    def test_y_la_EXACTITUD_sobre_esa_misma_salida_de_etiquetas_tambien(self):
        """La otra mitad, por el otro lado: la salida `label` es la correcta
        para una clasificación, y forzarla por nombre sigue midiendo."""
        recibo = atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase", metrica="accuracy",
                            salida="label")
        self.assertEqual(recibo["metrics"][0]["value"], EXACTITUD_DOCUMENTADA)

    def test_leer_un_VALOR_como_si_fuera_una_clase_falla_igual(self):
        """El mismo defecto por el lado contrario: una clasificación no se mide
        sobre una salida que significa un valor continuo."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,1", "0,0,0"])
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv, metrica="accuracy", semantica="valor")
        self.assertIn("no es una clase", str(e.exception))


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class LaANCHURA_NoDeclaraNadaTest(unittest.TestCase):
    """La última inferencia por FORMA que quedaba viva, y el corte la prohíbe.

    Medido el 2026-09-12 con un `MatMul` de dos columnas y nada más::

        atestiguar(crudo, csv, metrica="accuracy")
        → 1.0,  output_spec.kind = "probabilidades",
          chosen_because = "devuelve 2 valores por fila: se leen como
                            probabilidades por clase"

    El `argmax` acierta por invariancia CUANDO de verdad son logits, pero el
    recibo afirma «probabilidades» sobre algo que nadie ha declarado, y con un
    **regresor de dos salidas** no acierta nada: devuelve un número y se queda
    tan ancho.
    """

    def _csv(self, d: Path) -> Path:
        return _csv(d / "d.csv", "a,b,y", ["3,0,0", "0,3,1", "5,1,0"])

    def test_dos_columnas_que_nadie_declara_piden_el_mapa_de_salida(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _dos_columnas_sin_declarar(d / "crudo.onnx")
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, self._csv(d), metrica="accuracy")
        mensaje = str(e.exception)
        self.assertIn("la anchura no distingue", mensaje)
        self.assertIn("Declara el mapa de salida", mensaje)

    def test_pero_DECLARANDO_que_son_probabilidades_se_mide_igual_que_antes(self):
        """La otra mitad. El número no cambia: lo que cambia es que ahora lo
        sostiene una declaración y el recibo dice de quién es."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _dos_columnas_sin_declarar(d / "crudo.onnx")
            recibo = atestiguar(modelo, self._csv(d), metrica="accuracy",
                                semantica="probabilidades")
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        spec = recibo["models"][0]["output_spec"]
        self.assertEqual(spec["kind"], "probabilidades")
        self.assertIn("quien atestigua", spec["chosen_because"])

    def test_y_un_SOFTMAX_lo_sigue_declarando_el_grafo_sin_pedir_nada(self):
        """La otra mitad, y la que impide que el arreglo sea «fallar siempre»:
        cuando el grafo SÍ lo declara, no hay que declarar nada."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1, 2], con_etiqueta=False)
            csv = _csv(d / "d.csv", "a,b,c,y", ["3,0,0,0", "0,3,0,1", "0,0,3,2"])
            recibo = atestiguar(modelo, csv)
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)
        spec = recibo["models"][0]["output_spec"]
        self.assertEqual(spec["kind"], "probabilidades")
        self.assertIn("lo declara el grafo", spec["chosen_because"])


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class ElNUMERO_LoCalculaElRegistroUnicoTest(unittest.TestCase):
    """`attest` ya no tiene fórmula propia — invariante 1 del 105.

    Tenía dos: `aciertos / medidas` y `error_absoluto / medidas`, escritas aquí
    mientras las mismas dos fórmulas vivían en `matrixai.estudio.metricas` desde
    el 2026-09-06. Dos sitios midiendo lo mismo acaban divergiendo, y el día que
    divergieran nadie sabría cuál de los dos números firmó el recibo.

    CÓMO SE PRUEBA QUE DE VERDAD LO CALCULA EL REGISTRO, y no que los dos
    coincidan por ahora: se sustituye la FÓRMULA del registro por una que
    devuelve un número imposible y se comprueba que el recibo lo trae. Con una
    fórmula propia, el recibo seguiría trayendo la exactitud de siempre.
    """

    def _iris(self, metrica: str = "accuracy", **kw) -> dict:
        return atestiguar(IRIS_ONNX, IRIS_CSV, columna="clase", metrica=metrica, **kw)

    def test_la_EXACTITUD_sale_de_la_formula_del_registro(self):
        from dataclasses import replace
        from unittest import mock

        from matrixai.estudio import metricas

        falsa = replace(metricas.REGISTRO["accuracy"], formula=lambda m, u: 0.123456)
        with mock.patch.dict(metricas.REGISTRO, {"accuracy": falsa}):
            recibo = self._iris()
        self.assertEqual(recibo["metrics"][0]["value"], 0.123456)
        # Y sin el parche vuelve a ser la de verdad: el parche no deja rastro.
        self.assertEqual(self._iris()["metrics"][0]["value"], EXACTITUD_DOCUMENTADA)

    def test_y_el_MAE_tambien(self):
        from dataclasses import replace
        from unittest import mock

        from matrixai.estudio import metricas

        falsa = replace(metricas.REGISTRO["mae"], formula=lambda m, u: 42.0)
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,2", "2,2,4", "0,1,2"])
            with mock.patch.dict(metricas.REGISTRO, {"mae": falsa}):
                con_parche = atestiguar(modelo, csv, metrica="mae")
            sin_parche = atestiguar(modelo, csv, metrica="mae")
        self.assertEqual(con_parche["metrics"][0]["value"], 42.0)
        self.assertAlmostEqual(sin_parche["metrics"][0]["value"], 1 / 3)

    def test_las_metricas_las_decide_el_REGISTRO_no_una_linea_a_mano(self):
        from matrixai.estudio.metricas import REGISTRO

        for metrica in METRICAS:
            self.assertIn(metrica, REGISTRO)
        # Las dos históricas siguen; y las que el registro ya tenía y este
        # camino puede alimentar entran solas, sin escribir una fórmula.
        for metrica in ("accuracy", "mae", "macro_f1", "rmse", "r2"):
            self.assertIn(metrica, METRICAS)
        # Y NO entran las que piden algo que atestiguar un modelo ajeno no da:
        # de un ONNX de fuera aquí se lee la clase predicha, no una puntuación
        # ni una distribución. Decir que se saben medir sería prometer de más.
        for metrica in ("auroc", "log_loss", "brier_score", "average_precision"):
            self.assertNotIn(metrica, METRICAS)

    def test_una_metrica_que_el_registro_TIENE_pero_que_aqui_no_se_puede_alimentar(self):
        """Son dos negativas distintas y confundirlas manda a buscar una errata
        donde no la hay."""
        with self.assertRaises(AtestacionImposible) as e:
            self._iris(metrica="auroc")
        mensaje = str(e.exception)
        self.assertIn("existe en el registro", mensaje)
        self.assertIn("scores", mensaje)

    def test_y_una_que_el_registro_NO_tiene_se_contesta_con_las_que_hay(self):
        with self.assertRaises(AtestacionImposible) as e:
            self._iris(metrica="f1_macro")
        self.assertIn("no sé medir", str(e.exception))

    def test_la_DIRECCION_la_declara_el_registro(self):
        """Estaba escrita como «higher si es accuracy, lower en cualquier otro
        caso». Con la lista abierta, ese `else` publicaría un `r2` —donde más
        es mejor— como si menos fuera mejor."""
        from matrixai.estudio.metricas import direccion_de

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,2", "2,2,4", "0,1,3"])
            r2 = atestiguar(modelo, csv, metrica="r2")["metrics"][0]
            mae = atestiguar(modelo, csv, metrica="mae")["metrics"][0]
        # El R² NO es simétrico, así que este número también dice que lo real y
        # lo predicho no se han pasado del revés al construir la muestra:
        # SS_res = 4 y SS_tot = 2 sobre la columna objetivo dan -1,0; al revés
        # darían +0,143. Un MAE no lo habría notado.
        self.assertAlmostEqual(r2["value"], -1.0)
        self.assertEqual(r2["direction"], "higher_is_better")
        self.assertEqual(mae["direction"], "lower_is_better")
        self.assertEqual(r2["direction"], direccion_de("r2"))
        self.assertEqual(self._iris()["metrics"][0]["direction"], "higher_is_better")

    def test_las_metricas_NUEVAS_se_atestiguan_de_verdad(self):
        """Un RMSE y un macro-F1 escritos a mano, no recalculados con el mismo
        registro que se está probando."""
        import math

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _regresor(d / "r.onnx")
            # Predice a+b: 2, 4, 1 frente a 2, 4, 2 → errores 0, 0, 1.
            csv = _csv(d / "d.csv", "a,b,y", ["1,1,2", "2,2,4", "0,1,2"])
            rmse = atestiguar(modelo, csv, metrica="rmse")["metrics"][0]["value"]

            clf = _clasificador(d / "m.onnx", clases=[0, 1])
            # Dos aciertos y un fallo: la clase 1 predicha como 0.
            datos = _csv(d / "c.csv", "a,b,y", ["3,0,0", "0,3,1", "3,0,1"])
            macro = atestiguar(clf, datos, metrica="macro_f1")["metrics"][0]["value"]
            exactitud = atestiguar(clf, datos)["metrics"][0]["value"]
        self.assertAlmostEqual(rmse, math.sqrt(1 / 3))
        # F1 de la clase 0: P=1/2, R=1 → 0,666667. F1 de la clase 1: P=1, R=1/2
        # → 0,666667. Media macro: 0,666667 (redondeo a 6 decimales del
        # evaluador histórico, que el registro conserva a propósito).
        self.assertEqual(macro, 0.666667)
        self.assertAlmostEqual(exactitud, 2 / 3)


@unittest.skipUnless(_HAS, "onnx + onnxruntime required")
class UnVOCABULARIO_DeUnaSolaClaseNoMideNadaTest(unittest.TestCase):
    """Lo que cambió al adoptar el documento del registro, dicho y probado.

    El registro mide una clasificación contra un vocabulario DECLARADO de al
    menos dos clases. Cuando las clases no se declaran y todo lo que se ve —la
    columna objetivo y todo lo que predijo el modelo— es la misma clase, antes
    salía un 1,0 que no decía contra qué se había comparado. Ahora se dice qué
    falta y cómo darlo, que es lo que se puede arreglar.
    """

    def _todo_una_clase(self, d: Path) -> tuple[Path, Path]:
        return (_clasificador(d / "m.onnx", clases=[0, 1]),
                _csv(d / "d.csv", "a,b,y", ["3,0,0", "5,1,0", "9,2,0"]))

    def test_sin_declarar_las_clases_se_dice_lo_que_falta(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, csv = self._todo_una_clase(d)
            with self.assertRaises(AtestacionImposible) as e:
                atestiguar(modelo, csv)
        mensaje = str(e.exception)
        self.assertIn("solo consta una clase", mensaje)
        self.assertIn("Declara las clases", mensaje)

    def test_y_declarandolas_se_mide(self):
        """La otra mitad: el 1,0 legítimo —«no falló ni una de estas tres»—
        sigue estando, en cuanto se dice contra qué vocabulario se mide."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo, csv = self._todo_una_clase(d)
            recibo = atestiguar(modelo, csv, clases=[0, 1])
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)

    def test_una_etiqueta_ESCRITA_DE_OTRA_FORMA_sigue_siendo_la_misma_clase(self):
        """QUIÉN decide que dos etiquetas son la misma clase no cambió con la
        migración al registro, y esto lo mide.

        El registro cuenta un acierto cuando `y_true` y la clase predicha son
        **la misma cadena**; aquí no lo son —el ONNX devuelve un `int64` 1 y la
        columna objetivo dice `1.0`—, así que quien las junta sigue siendo
        `misma_etiqueta()`, la regla de este paquete, que ya sabe que `"1"` y
        `"1.0"` son la misma clase y que `"01"` no lo es. Sin esta prueba, un
        vocabulario construido con una comparación de texto cruda pasaba la
        suite entera en verde (medido el 2026-09-12, sabotaje 8) y dejaba la
        exactitud en 0,0 para un modelo que acierta todo.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0.0", "0,3,1.0"])
            recibo = atestiguar(modelo, csv)
        self.assertEqual(recibo["metrics"][0]["value"], 1.0)

    def test_cual_de_las_dos_clases_sea_la_positiva_NO_cambia_la_exactitud(self):
        """`Muestra` no se construye binaria sin clase positiva y este camino no
        siempre sabe cuál es, así que declara la última. Esto mide que da igual
        —accuracy sale de la matriz de confusión, que con la clase predicha
        declarada usa `declared_label` y no mira la positiva— en vez de
        suponerlo."""
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            modelo = _clasificador(d / "m.onnx", clases=[0, 1])
            csv = _csv(d / "d.csv", "a,b,y", ["3,0,0", "0,3,1", "3,0,1"])
            derecho = atestiguar(modelo, csv, clases=[0, 1])["metrics"][0]["value"]
            invertido = atestiguar(modelo, csv, clases=[1, 0])["metrics"][0]["value"]
        self.assertEqual(derecho, invertido)
        self.assertAlmostEqual(derecho, 2 / 3)


class SigmoidDeUnaColumnaConClasesDeTextoTest(unittest.TestCase):
    """102-C0, auditoría propia 2026-09-11 — el binario más común del mundo
    no se podía atestiguar de NINGUNA manera.

    Un sigmoide `[N,1]` (Keras/torch de toda la vida) con etiquetas de texto:
    sin `--classes`, `attest` pedía «declara cuál es la clase negativa y cuál
    la positiva»; al declararlas, respondía «se declararon 2 clases y el
    modelo devuelve 1 valores por fila». El mensaje pedía exactamente lo que
    luego rechazaba, porque la comprobación `len(clases) == ancho` corría
    antes de la rama `probabilidad_positiva` — y una probabilidad positiva es
    UNA columna que vale por DOS clases.
    """

    def _sigmoide(self):
        import numpy as np
        from onnx import TensorProto, helper
        grafo = helper.make_graph(
            [helper.make_node("MatMul", ["X", "W"], ["z"]),
             helper.make_node("Sigmoid", ["z"], ["p"])],
            "sig",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("p", TensorProto.FLOAT, [None, 1])],
            [helper.make_tensor("W", TensorProto.FLOAT, [2, 1],
                                np.array([[0.5], [0.5]], dtype=np.float32).tobytes(), raw=True)])
        return helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 13)])

    def _elegida(self):
        import tempfile
        import onnx
        import onnxruntime as ort
        modelo = self._sigmoide()
        ruta = f"{tempfile.mkdtemp()}/m.onnx"
        onnx.save(modelo, ruta)
        sesion = ort.InferenceSession(modelo.SerializeToString())
        return elegir_salida(sesion, tarea="clasificacion", ruta_modelo=ruta)

    def test_declarar_dos_clases_de_texto_YA_funciona(self):
        elegida = self._elegida()
        self.assertEqual(elegida.semantica, "probabilidad_positiva")
        clases, origen = resolver_clases(
            elegida, clases_declaradas=["no", "yes"],
            etiquetas_observadas=["no", "yes"], ancho=1)
        self.assertEqual(clases, ["no", "yes"])
        self.assertEqual(origen, "declaradas")

    def test_pero_TRES_clases_siguen_sin_valer(self):
        """La otra mitad: sin ella, lo de arriba lo pasaría un backend que
        acepta cualquier lista. Una probabilidad positiva son exactamente dos
        clases, ni una ni tres."""
        with self.assertRaises(SalidaOnnxAmbigua):
            resolver_clases(self._elegida(), clases_declaradas=["a", "b", "c"],
                            etiquetas_observadas=["no", "yes"], ancho=1)

    def test_y_sin_declararlas_sigue_pidiendolas_con_su_motivo(self):
        """Deducir `no`/`yes` de la columna objetivo sería adivinar cuál es la
        positiva: eso sigue rechazado, que es lo correcto."""
        with self.assertRaises(SalidaOnnxAmbigua):
            resolver_clases(self._elegida(), clases_declaradas=None,
                            etiquetas_observadas=["no", "yes"], ancho=1)


class CodigosConCeroALaIzquierdaTest(unittest.TestCase):
    """102-C0, auditoría 2026-09-11 — el NUANCED que se dejó sin reparar
    producía una exactitud FALSA, no solo un matiz.

    `misma_etiqueta` comparaba por número en cuanto las dos etiquetas
    parseaban como tal, así que `"01"` y `"1"` eran la misma clase. En un
    dataset de códigos de categoría donde el cero a la izquierda es
    significativo —los hay, y muchos— eso hacía que `attest` certificara
    **1,0 donde la exactitud real era 0,5**. Un número falso que además se
    puede firmar: exactamente el defecto que este corte existía para cerrar,
    reaparecido por otra puerta.

    La regla nueva compara por número solo cuando las dos están escritas en
    forma CANÓNICA. Un `"01"` no lo produce ninguna salida numérica de un
    ONNX: es una forma textual, y tratarla como número es adivinar que quien
    la escribió no quería decir lo que escribió.
    """

    def test_cero_a_la_izquierda_es_OTRA_clase(self):
        self.assertFalse(misma_etiqueta("01", "1"))
        self.assertFalse(misma_etiqueta("007", "7"))
        self.assertFalse(misma_etiqueta("1e0", "1"))

    def test_pero_las_formas_canonicas_siguen_siendo_la_misma(self):
        """Sin esta mitad, lo de arriba lo pasaría una versión que compara
        solo texto — y entonces el int64 `1` que devuelve un ONNX dejaría de
        casar con el `"1"` del CSV, que es el caso normal."""
        self.assertTrue(misma_etiqueta(1, "1"))
        self.assertTrue(misma_etiqueta("1", "1.0"))
        self.assertTrue(misma_etiqueta(1.0, 1))
        self.assertTrue(misma_etiqueta(7, "7"))
        self.assertTrue(misma_etiqueta("setosa", "setosa"))

    def test_una_etiqueta_igual_a_si_misma_SIEMPRE_casa(self):
        """Defecto anterior, cerrado de paso: `"nan"` como etiqueta daba
        `False` contra sí mismo porque `float("nan") != float("nan")` por
        norma IEEE. Dos filas con la MISMA etiqueta contadas como distintas
        bajan la exactitud atestiguada sin que nadie lo note."""
        for etiqueta in ("nan", "inf", "-inf", "01", "setosa", "3"):
            self.assertTrue(misma_etiqueta(etiqueta, etiqueta), etiqueta)

    def test_el_efecto_sobre_la_exactitud_atestiguada(self):
        """El caso del auditor, reducido a su esencia: 4 filas de las que el
        modelo acierta 2. Con la regla vieja salían 4 de 4."""
        predichas = ["01", "01", "1", "1"]
        esperadas = ["01", "1", "01", "1"]
        aciertos = sum(1 for p, e in zip(predichas, esperadas) if misma_etiqueta(p, e))
        self.assertEqual(aciertos, 2, "se están contando como iguales '01' y '1'")
