# SPDX-License-Identifier: AGPL-3.0-only
"""PESOS_GRANDES C4 — formato binario `.mxw` para pesos entrenados.

Ver 48_PESOS_GRANDES_CONTRATO.md. `write_mxw`/`read_mxw` deben dar round-trip
exacto (mismos tensores), detectar tamper (hash de contenido), escribir
atómicamente (nunca un `.mxw` a medias), y `build_parameter_template_for_state`
debe dar el mismo resultado materializado que el camino json de siempre.
"""
from __future__ import annotations

import tempfile
import unittest
from importlib import util
from pathlib import Path

_HAS_TORCH = util.find_spec("torch") is not None


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RoundTripTest(unittest.TestCase):
    def _state(self):
        import torch
        return {
            "Net.W1": torch.randn(8, 2),
            "Net.b1": torch.randn(8),
            "Net.W2": torch.randn(2, 8),
            "Net.b2": torch.randn(2),
        }

    def test_write_read_gives_identical_tensors(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw
        state = self._state()
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        header = write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        self.assertEqual(header["model_hash"], "mh")
        self.assertEqual(len(header["tensors"]), 4)

        loaded = read_mxw(path)
        self.assertEqual(set(loaded), set(state))
        for k, v in state.items():
            self.assertTrue(torch.allclose(loaded[k], v))
            self.assertEqual(list(loaded[k].shape), list(v.shape))
            self.assertEqual(loaded[k].dtype, torch.float32)

    def test_non_float32_input_is_coerced(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw
        state = {"W": torch.randn(4, 4).double()}  # float64
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        loaded = read_mxw(path)
        self.assertEqual(loaded["W"].dtype, torch.float32)
        self.assertTrue(torch.allclose(loaded["W"], state["W"].float()))

    def test_mmap_reader_matches_without_full_body_copy(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw_mmap

        state = self._state()
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        loaded = read_mxw_mmap(path)
        self.assertEqual(set(loaded), set(state))
        for key, expected in state.items():
            self.assertTrue(torch.equal(loaded[key], expected.float()))

    def test_writer_requires_model_and_schema_bindings(self) -> None:
        from matrixai.parameters.binary_store import write_mxw, MxwError

        tmp = Path(tempfile.mkdtemp())
        with self.assertRaisesRegex(MxwError, "model_hash"):
            write_mxw(
                tmp / "no-model.mxw", self._state(),
                model_hash="", parameter_schema_hash="sh",
            )
        with self.assertRaisesRegex(MxwError, "parameter_schema_hash"):
            write_mxw(
                tmp / "no-schema.mxw", self._state(),
                model_hash="mh", parameter_schema_hash="",
            )

    def test_header_only_read_does_not_need_the_body(self) -> None:
        from matrixai.parameters.binary_store import write_mxw, read_mxw_header
        state = self._state()
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        header = write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        header2 = read_mxw_header(path)
        self.assertEqual(header, header2)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TamperDetectionTest(unittest.TestCase):
    def test_flipped_byte_in_body_is_detected(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw, MxwError
        state = {"W": torch.randn(4, 4)}
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF  # flip a byte inside the body
        path.write_bytes(bytes(data))
        with self.assertRaises(MxwError):
            read_mxw(path)

    def test_verify_false_skips_hash_check(self) -> None:
        """Escotilla explícita, no un default silencioso — para diagnóstico."""
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw
        state = {"W": torch.randn(4, 4)}
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        read_mxw(path, verify=False)  # no debe lanzar

    def test_truncated_file_raises_clean_error(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw, MxwError
        state = {"W": torch.randn(4, 4)}
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])  # corta el fichero a la mitad
        with self.assertRaises(MxwError):
            read_mxw(path)

    def test_wrong_magic_raises_clean_error(self) -> None:
        from matrixai.parameters.binary_store import read_mxw_header, MxwError
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "bad.mxw"
        path.write_bytes(b"NOTX" + b"\x00" * 20)
        with self.assertRaises(MxwError):
            read_mxw_header(path)

    def test_missing_file_raises_clean_error(self) -> None:
        from matrixai.parameters.binary_store import read_mxw_header, MxwError
        with self.assertRaises(MxwError):
            read_mxw_header("/nonexistent/path/x.mxw")

    @staticmethod
    def _rewrite_header(path: Path, mutate) -> None:
        """Reescribe SOLO la cabecera del .mxw (deja el cuerpo, y por tanto su
        `content_hash`, intacto) aplicando `mutate(header_dict)`. Sirve para
        forjar cabeceras incoherentes que superan el hash de contenido."""
        import json
        import struct
        raw = path.read_bytes()
        (hlen,) = struct.unpack("<Q", raw[4:12])
        header = json.loads(raw[12:12 + hlen].decode("utf-8"))
        body = raw[12 + hlen:]
        mutate(header)
        new_hb = json.dumps(header).encode("utf-8")
        path.write_bytes(b"MXW1" + struct.pack("<Q", len(new_hb)) + new_hb + body)

    def test_inconsistent_shape_vs_nbytes_raises_mxw_error(self) -> None:
        """Reauditoría Opus (BAJA): una cabecera cuyo `shape` no cuadra con
        `nbytes` (aunque el `content_hash` del cuerpo sea correcto) debe fallar
        con `MxwError` explícito, no con un `ValueError` de `reshape` que se
        escaparía del `except MxwError` de los callers."""
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw, MxwError
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, {"W": torch.randn(2, 2)}, model_hash="mh", parameter_schema_hash="sh")
        # 16 bytes reales (2x2 float32); declaramos shape [3] (12 bytes) -> incoherente
        self._rewrite_header(path, lambda h: h["tensors"][0].__setitem__("shape", [3]))
        with self.assertRaises(MxwError):
            read_mxw(path, verify=False)  # verify=False: aislamos el chequeo de cabecera

    def test_nbytes_not_multiple_of_four_raises_mxw_error(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw, MxwError
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, {"W": torch.randn(2, 2)}, model_hash="mh", parameter_schema_hash="sh")
        self._rewrite_header(path, lambda h: h["tensors"][0].__setitem__("nbytes", 6))
        with self.assertRaises(MxwError):
            read_mxw(path, verify=False)

    def test_offset_out_of_body_raises_mxw_error(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw, MxwError
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, {"W": torch.randn(2, 2)}, model_hash="mh", parameter_schema_hash="sh")
        self._rewrite_header(path, lambda h: h["tensors"][0].__setitem__("offset", 9999))
        with self.assertRaises(MxwError):
            read_mxw(path, verify=False)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class AtomicWriteTest(unittest.TestCase):
    def test_no_tmp_files_left_after_success(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw
        state = {"W": torch.randn(4, 4)}
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash="mh", parameter_schema_hash="sh")
        leftovers = list(tmp.glob("*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertTrue(path.exists())

    def test_rewrite_replaces_atomically(self) -> None:
        import torch
        from matrixai.parameters.binary_store import write_mxw, read_mxw
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, {"W": torch.zeros(2, 2)}, model_hash="mh", parameter_schema_hash="sh")
        write_mxw(path, {"W": torch.ones(2, 2)}, model_hash="mh2", parameter_schema_hash="sh2")
        loaded = read_mxw(path)
        self.assertTrue(torch.allclose(loaded["W"], torch.ones(2, 2)))
        header = __import__(
            "matrixai.parameters.binary_store", fromlist=["read_mxw_header"]
        ).read_mxw_header(path)
        self.assertEqual(header["model_hash"], "mh2")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TemplateHelperParityTest(unittest.TestCase):
    """`build_parameter_template_for_state` + `materialize_parameter_set` sobre
    tensores leídos de un `.mxw` da EXACTAMENTE los mismos valores que
    materializar directamente desde el state_dict en memoria (el camino json
    de C2/C3) — el binario no es una vía con drift."""

    MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""

    def test_binary_roundtrip_materializes_identically_to_direct_json(self) -> None:
        import torch
        from matrixai.parser import parse_text
        from matrixai.forward.dense_torch import (
            build_parameter_template_for_state, materialize_parameter_set,
        )
        from matrixai.parameters.binary_store import write_mxw, read_mxw

        program = parse_text(self.MXAI)
        net, template = build_parameter_template_for_state(program)
        self.assertIs(net, program.networks[0])
        self.assertTrue(template.model_hash)
        self.assertTrue(template.parameter_schema_hash)

        state = {
            "Net.W1": torch.randn(8, 2), "Net.b1": torch.randn(8),
            "Net.W2": torch.randn(2, 8), "Net.b2": torch.randn(2),
        }
        direct = materialize_parameter_set(net, state, template)

        tmp = Path(tempfile.mkdtemp())
        path = tmp / "m.mxw"
        write_mxw(path, state, model_hash=template.model_hash,
                  parameter_schema_hash=template.parameter_schema_hash)
        via_binary = materialize_parameter_set(net, read_mxw(path), template)

        for key in state:
            self.assertEqual(
                direct.parameters[key]["values"], via_binary.parameters[key]["values"]
            )
        self.assertEqual(direct.model_hash, via_binary.model_hash)


if __name__ == "__main__":
    unittest.main()
