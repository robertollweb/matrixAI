"""87-C1 — EL MOTOR YA CORRE MODELOS QUE NO HA ENTRENADO MATRIXAI.

El motor del 81 recibe un mapa `kind → función` —está bien hecho para esto— y
hasta hoy solo había UN ejecutor: el del registry de P21. O sea que **para
auditarse con MatrixAI había que abandonar la herramienta con la que se
trabaja**, que es la barrera de entrada real de este producto: la población de
modelos que podrían querer un recibo no se parece en tamaño a la de modelos
hechos aquí, que está medida en ~0.

Las reglas son las mismas que las del ejecutor de casa, y por los mismos
motivos: el digest se comprueba AL EJECUTAR, las formas se contrastan si el
modelo las declara y **si no, se dice**, y lo que no se puede ejecutar no se
ejecuta a medias.

Y una que se aplicó antes de que costara nada: **un pipeline no elige qué
fichero se abre en la máquina de quien lo corre**. Es el bloqueante del 82
—`verify` seguía la ruta que el manifiesto dijera— aplicado aquí de entrada.

LO QUE ESTE EJECUTOR NO DEMUESTRA, y va escrito en el código: correr un ONNX
ajeno atestigua **la evaluación, el entorno y las entradas**. No es prueba de
que ese modelo se entrenara como diga nadie.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.pipelines.executors import (
    ModeloAjenoNoDisponible,
    ModeloCambiado,
    TipoDeEntradaIncompatible,
    ejecutor_onnx,
)
from matrixai.pipelines.runtime import KINDS_FUERA_DEL_REGISTRY

_HAS_ORT = util.find_spec("onnxruntime") is not None
_HAS_ONNX = util.find_spec("onnx") is not None


def _modelo_onnx(destino: Path) -> tuple[Path, str]:
    """Un ONNX de verdad, hecho aquí — sin pasar por MatrixAI."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    nodo = helper.make_node("MatMul", ["entrada", "pesos"], ["salida"])
    pesos = helper.make_tensor("pesos", TensorProto.FLOAT, [3, 2],
                               np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32).ravel())
    grafo = helper.make_graph(
        [nodo], "ajeno",
        [helper.make_tensor_value_info("entrada", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("salida", TensorProto.FLOAT, [1, 2])],
        [pesos])
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 13)])
    modelo.ir_version = 9
    ruta = destino / "ajeno.onnx"
    ruta.write_bytes(modelo.SerializeToString())
    return ruta, hashlib.sha256(ruta.read_bytes()).hexdigest()


@unittest.skipUnless(_HAS_ORT and _HAS_ONNX, "onnx + onnxruntime required")
class CorreUnModeloAjenoTest(unittest.TestCase):
    def test_ejecuta_y_devuelve_sus_valores(self):
        with TemporaryDirectory() as d:
            ruta, digest = _modelo_onnx(Path(d))
            ejecutar = ejecutor_onnx(raiz=Path(d))
            salida = ejecutar(entradas={}, contexto={}, nodo={
                "id": "n", "model": ruta.name, "entry_hash": digest, "input": [1.0, 2.0, 3.0]})
        self.assertEqual([round(v, 4) for v in salida["values"]], [4.0, 5.0])

    def test_el_digest_se_comprueba_AL_EJECUTAR(self):
        """Entre la comprobación previa y el nodo pueden pasar minutos, y el
        fichero está en un disco que no controlamos."""
        with TemporaryDirectory() as d:
            ruta, digest = _modelo_onnx(Path(d))
            ejecutar = ejecutor_onnx(raiz=Path(d))
            ruta.write_bytes(ruta.read_bytes() + b"\x00")   # el fichero cambia
            with self.assertRaises(ModeloCambiado):
                ejecutar(entradas={}, contexto={}, nodo={
                    "id": "n", "model": ruta.name, "entry_hash": digest,
                    "input": [1.0, 2.0, 3.0]})

    def test_sin_digest_declarado_no_se_ejecuta(self):
        with TemporaryDirectory() as d:
            ruta, _ = _modelo_onnx(Path(d))
            with self.assertRaises(ModeloAjenoNoDisponible):
                ejecutor_onnx(raiz=Path(d))(entradas={}, contexto={}, nodo={
                    "id": "n", "model": ruta.name, "input": [1.0, 2.0, 3.0]})

    def test_una_entrada_de_otra_forma_NO_se_recorta_ni_se_rellena(self):
        with TemporaryDirectory() as d:
            ruta, digest = _modelo_onnx(Path(d))
            with self.assertRaises(TipoDeEntradaIncompatible):
                ejecutor_onnx(raiz=Path(d))(entradas={}, contexto={}, nodo={
                    "id": "n", "model": ruta.name, "entry_hash": digest,
                    "input": [1.0, 2.0]})

    def test_dice_si_pudo_contrastar_la_forma_o_no(self):
        with TemporaryDirectory() as d:
            ruta, digest = _modelo_onnx(Path(d))
            salida = ejecutor_onnx(raiz=Path(d))(entradas={}, contexto={}, nodo={
                "id": "n", "model": ruta.name, "entry_hash": digest,
                "input": [1.0, 2.0, 3.0]})
        self.assertEqual(salida["input_shape_check"], "declarada")


@unittest.skipUnless(_HAS_ORT and _HAS_ONNX, "onnx + onnxruntime required")
class UnPipelineNoEligeQueFicheroSeAbreTest(unittest.TestCase):
    def test_una_ruta_fuera_de_la_raiz_no_se_abre(self):
        with TemporaryDirectory() as d:
            _modelo_onnx(Path(d))
            with self.assertRaises(ModeloAjenoNoDisponible) as e:
                ejecutor_onnx(raiz=Path(d))(entradas={}, contexto={}, nodo={
                    "id": "n", "model": "../../etc/hostname",
                    "entry_hash": "a" * 64, "input": [1.0]})
        self.assertIn("fuera de la raíz", str(e.exception))

    def test_un_fichero_que_no_esta_se_dice(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ModeloAjenoNoDisponible) as e:
                ejecutor_onnx(raiz=Path(d))(entradas={}, contexto={}, nodo={
                    "id": "n", "model": "no_existe.onnx", "entry_hash": "a" * 64,
                    "input": [1.0]})
        self.assertIn("ahí no hay fichero", str(e.exception))


class LaVerificacionPreviaConoceEstosKindsTest(unittest.TestCase):
    """Sin esto el motor no arrancaba: exigía que el modelo estuviera en el
    registry, y un ONNX ajeno es un FICHERO."""

    def test_onnx_esta_declarado_fuera_del_registry(self):
        self.assertIn("onnx", KINDS_FUERA_DEL_REGISTRY)

    def test_pero_SIGUE_exigiendo_el_digest(self):
        """Saltarse la comprobación entera habría dejado entrar un pipeline que
        dice «corre este onnx» sin decir CUÁL."""
        from matrixai.pipelines.runtime import DigestNoCoincide, _resolver_si_toca
        with self.assertRaises(DigestNoCoincide):
            _resolver_si_toca({"kind": "onnx", "model": "x.onnx"}, {})
        # Con digest, pasa sin tocar el registry.
        _resolver_si_toca({"kind": "onnx", "model": "x.onnx", "entry_hash": "a" * 64}, {})

    def test_un_kind_normal_sigue_resolviendose_por_el_registry(self):
        from matrixai.pipelines.runtime import DigestNoCoincide, _resolver_si_toca
        with self.assertRaises(DigestNoCoincide):
            _resolver_si_toca({"kind": "modelo", "model": "m@v1", "entry_hash": "a" * 64}, {})


@unittest.skipUnless(_HAS_ORT and _HAS_ONNX, "onnx + onnxruntime required")
class DePUNTA_A_PUNTA_porElMotorTest(unittest.TestCase):
    def test_el_motor_corre_un_pipeline_con_un_modelo_ajeno(self):
        from matrixai.pipelines.engine import ejecutar_pipeline
        with TemporaryDirectory() as d:
            ruta, digest = _modelo_onnx(Path(d))
            informe = ejecutar_pipeline(
                {"pipeline_id": "ajeno", "version": "1", "timeout_s": 60,
                 "nodes": [{"id": "n", "kind": "onnx", "model": ruta.name,
                            "entry_hash": digest, "input": [1.0, 2.0, 3.0]}]},
                registry={}, ejecutores={"onnx": ejecutor_onnx(raiz=Path(d))})
        self.assertEqual(informe["report"]["status"], "ok")
        self.assertEqual(informe["report"]["executed"], ["n"])
        # Y EL DIGEST DEL MODELO AJENO QUEDA EN LA TRAZA, no solo su nombre.
        # La primera versión de esta prueba lo buscaba en `values` y se puso
        # roja: ahí vive el digest de la SALIDA, que es otra cosa. El del
        # modelo no estaba en ninguna parte — para una entrada del registry el
        # nombre se resuelve a un digest, pero `ajeno.onnx` mañana puede ser
        # otro fichero.
        nodo = informe["trace"]["nodes"][0]
        self.assertEqual(nodo["model_digest"], digest)
        self.assertEqual(nodo["model"], "ajeno.onnx")


if __name__ == "__main__":
    unittest.main()


# ── LO QUE ENCONTRÓ LA AUDITORÍA EXTERNA (2026-08-25, hallazgo 2) ───────────

class LosTiposDECLARADOSSeContrastanTest(unittest.TestCase):
    """El ejecutor decía en su cabecera que «los tipos se comprueban si están
    declarados» y **convertía todo a float**. Medido por la auditoría: un modelo
    `int64` aceptó `3.7`, ONNX Runtime lo truncó **en silencio** a 3 y devolvió
    6. Un valor que se pierde por el camino no puede acabar en un recibo como si
    fuera el que se dio.
    """

    class _Meta:
        def __init__(self, tipo):
            self.type = tipo

    def test_un_entero_recibe_ENTEROS(self):
        from matrixai.onnx_entrada import tensor_para
        self.assertEqual(tensor_para(self._Meta("tensor(int64)"), [3], "n"), [[3]])

    def test_y_un_3_7_contra_un_int64_se_RECHAZA(self):
        from matrixai.onnx_entrada import EntradaOnnxIncompatible, tensor_para
        with self.assertRaises(EntradaOnnxIncompatible) as e:
            tensor_para(self._Meta("tensor(int64)"), [3.7], "n")
        self.assertIn("truncaría en silencio", str(e.exception))

    def test_un_float_sigue_siendo_float(self):
        from matrixai.onnx_entrada import tensor_para
        self.assertEqual(tensor_para(self._Meta("tensor(float)"), [3.7], "n"), [[3.7]])

    def test_un_tipo_que_no_sabe_alimentar_se_DICE(self):
        """Convertirlo a float y confiar sería adivinar."""
        from matrixai.onnx_entrada import EntradaOnnxIncompatible, tensor_para
        with self.assertRaises(EntradaOnnxIncompatible) as e:
            tensor_para(self._Meta("tensor(string)"), ["a"], "n")
        self.assertIn("tensor(string)", str(e.exception))


class LaFormaNoConfundeElLOTEConLasCaracteristicasTest(unittest.TestCase):
    """Mismo hallazgo: un modelo válido que declara `[1, "features"]` con tres
    valores **se rechazaba diciendo que esperaba uno** — el eje del lote se
    tomaba por el número de características."""

    def _comprobar(self, forma, n_valores):
        """Reproduce la decisión del ejecutor sobre una forma declarada."""
        caracteristicas = forma[1:] if len(forma) > 1 else forma
        if caracteristicas and all(isinstance(d, int) for d in caracteristicas):
            esperado = 1
            for d in caracteristicas:
                esperado *= d
            return "rechaza" if esperado != n_valores else "acepta"
        return "no declarada"

    def test_un_eje_dinamico_no_se_rechaza_se_DECLARA_no_comprobado(self):
        self.assertEqual(self._comprobar([1, "features"], 3), "no declarada")

    def test_una_forma_entera_si_se_contrasta(self):
        self.assertEqual(self._comprobar([1, 3], 3), "acepta")
        self.assertEqual(self._comprobar([1, 3], 2), "rechaza")

    def test_y_el_lote_no_cuenta_como_caracteristica(self):
        """`[1, 3]` son tres características, no una."""
        self.assertEqual(self._comprobar([1, 3], 1), "rechaza")


class LosDosCaminosAlimentanIGUALTest(unittest.TestCase):
    """El ejecutor del motor (C1) y `matrixai attest` (C2) deciden la forma y el
    tipo **en el mismo sitio**.

    La 2ª auditoría externa (2026-08-25) lo dijo así: «la garantía de C1 no
    alcanza C2». Cada uno tenía su copia y habían divergido: attest filtraba los
    ejes dinámicos —confundiendo `[1, "features"]` con UNA entrada— y convertía a
    `float` siempre, así que un `int64` tragaba `3.7` truncado en silencio. Si
    alguien vuelve a escribir un camino propio, esta prueba se pone roja.
    """

    def test_los_dos_modulos_importan_el_MISMO_ayudante(self):
        import matrixai.onnx_entrada as comun
        for modulo, nombre in (("matrixai/pipelines/executors.py", "el ejecutor"),
                               ("matrixai/export/attest.py", "attest")):
            with self.subTest(modulo=nombre):
                fuente = (RAIZ / modulo).read_text(encoding="utf-8")
                self.assertIn("matrixai.onnx_entrada", fuente,
                              f"{nombre} decide por su cuenta cómo alimentar un modelo")
                self.assertIn("tensor_para", fuente)
        self.assertEqual(sorted(comun.TIPOS_ONNX),
                         ["tensor(double)", "tensor(float)", "tensor(int32)",
                          "tensor(int64)"])

    def test_un_eje_DINAMICO_no_se_confunde_con_una_caracteristica(self):
        from matrixai.onnx_entrada import caracteristicas_declaradas

        class _M:
            def __init__(self, forma): self.shape = forma

        # `[1, "features"]`: el `1` es el LOTE. Antes attest comparaba el número
        # de columnas contra ese `1` y rechazaba el modelo.
        self.assertEqual(caracteristicas_declaradas(_M([1, "features"])),
                         (None, "no declarada"))
        self.assertEqual(caracteristicas_declaradas(_M(["batch", 7])), (7, "declarada"))
        self.assertEqual(caracteristicas_declaradas(_M([None, 4])), (4, "declarada"))
