# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""TRANSFORMER_BLOQUE C1 — regresiones de la auditoría de Roberto (2026-07-10).

5 hallazgos (4 ALTA + 1 MEDIA), todos con repro exacto de la auditoría:
  ALTA-1: Reshape podía cambiar L/DIM silenciosamente antes del bloque.
  ALTA-2: Reshape permitía saltarse el POOL obligatorio.
  ALTA-3: el backend fallaba abierto (supported=True) y generaba manifests
          inválidos (vocab=0 filtrado, segunda llamada sin sequence_map).
  ALTA-4: colisión de program_hash por no serializar TransformerBlockSpec.position.
  MEDIA:  se aceptaba EMBEDDING declarado después del BLOCK TRANSFORMER.
"""
from __future__ import annotations

from matrixai.compiler.backend_contract import BackendContractAnalyzer
from matrixai.parameters.store import program_hash
from matrixai.parser.parser import parse_text
from matrixai.types import check_composite_network_types


def _sequences(prog):
    return {s.name: s for s in prog.sequences}


class TestAltaUnoReshapeAntesDelBloque:
    """Reshape entre la EMBEDDING y el BLOCK TRANSFORMER no debe poder cambiar
    silenciosamente L/DIM (violaría la herencia obligatoria, decisión 3)."""

    def test_reshape_before_block_rejected(self):
        src = """
PROJECT F1
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  LAYER Reshape target=[32,256]
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "Reshape" in msg and "sequence stream" in msg

    def test_layernorm_before_block_still_allowed(self):
        """Regresión de la propia corrección: LayerNorm/Dropout/Activation SÍ
        pueden preceder al bloque (no cambian shape) — no deben empezar a fallar."""
        src = """
PROJECT F1b
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  LAYER LayerNorm
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert res.ok, res.errors


class TestAltaDosPoolObligatorioNoEludible:
    """Ninguna capa (Reshape incluido) puede sacar el stream de secuencia sin
    pasar por un POOL mean|cls explícito (decisión 4)."""

    def test_reshape_after_block_cannot_skip_pool(self):
        src = """
PROJECT F2
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  LAYER Reshape target=[8192]
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "Reshape" in msg and "POOL" in msg

    def test_dense_after_block_without_pool_still_rejected(self):
        """La ruta original (Dense directo tras el bloque) sigue rechazada."""
        src = """
PROJECT F2b
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok


class TestAltaTresBackendFallaCerrado:
    """El backend debe marcar supported=False/differentiable=False y no exponer
    trainable_parameters ni un manifest con vocab=0 mientras C2 no exista."""

    _SRC = """
PROJECT F3
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  BLOCK enc TRANSFORMER
    LAYERS 4
    HEADS 4
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END

GRAPH
  Texto -> N
END
"""

    def test_report_fails_closed(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        assert report.ok is False
        node = next(n for n in report.nodes if n.node == "N")
        assert node.supported is False
        assert node.differentiable is False
        assert "TRANSFORMER_BLOQUE C2" in node.reason

    def test_no_trainable_parameters_exposed(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        assert report.trainable_parameters == []

    def test_layer_manifest_does_not_leak_vocab_zero(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        # Ni un shape=[0, dim] en ningún sitio del manifest de capas.
        for entry in report.layer_manifest:
            for param in entry.get("parameters", []):
                shape = param.get("shape")
                if shape:
                    assert shape[0] != 0, f"vocab=0 sentinel leaked into manifest: {entry}"

    def test_layer_manifest_reports_pending_transformer_block(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        block_entries = [e for e in report.layer_manifest if e.get("block_name") == "enc"]
        assert len(block_entries) == 1
        assert block_entries[0]["differentiable"] is False

    def test_second_typecheck_call_site_receives_sequence_map(self):
        """Antes del fix, la construcción de layer_manifest usaba un
        check_composite_network_types SIN sequence_map, así que el INPUT
        SEQUENCE nunca resolvía y el resultado quedaba con errores espurios."""
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        node = next(n for n in report.nodes if n.node == "N")
        # El único motivo de "no soportado" debe ser el bloque transformer
        # pendiente — NUNCA "INPUT no es un VECTOR o SEQUENCE declarado".
        assert "not a declared" not in node.reason


class TestAltaCuatroColisionHash:
    """Dos redes válidas y semánticamente distintas (orden relativo del bloque
    frente a un LAYER top-level) deben producir program_hash DIFERENTES."""

    @staticmethod
    def _network(order: str) -> str:
        if order == "layernorm_first":
            body = "LAYER LayerNorm\n  BLOCK enc TRANSFORMER\n    LAYERS 1\n  END"
        else:
            body = "BLOCK enc TRANSFORMER\n    LAYERS 1\n  END\n  LAYER LayerNorm"
        return f"""
PROJECT F4
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  {body}
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""

    def test_moving_block_relative_to_layer_changes_hash(self):
        prog_a = parse_text(self._network("layernorm_first"))
        prog_b = parse_text(self._network("transformer_first"))
        assert program_hash(prog_a) != program_hash(prog_b)

    def test_both_orderings_typecheck_ok(self):
        # Ambas son arquitecturas válidas (no es un error de gramática, es
        # una diferencia estructural real que el hash debe distinguir).
        for order in ("layernorm_first", "transformer_first"):
            prog = parse_text(self._network(order))
            res = check_composite_network_types(
                prog.networks[0], {}, _sequences(prog)
            )
            assert res.ok, (order, res.errors)

    def test_transformer_block_position_is_serialized(self):
        prog = parse_text(self._network("layernorm_first"))
        data = prog.to_dict()
        tb_dict = data["networks"][0]["transformer_blocks"][0]
        assert "position" in tb_dict
        assert tb_dict["position"] == 1  # 1 top_layer (LayerNorm) precede it


class TestMediaEmbeddingDebePrecederAlBloque:
    def test_embedding_after_block_rejected(self):
        src = """
PROJECT F5
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  EMBEDDING tok FROM Texto DIM 128
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "BEFORE the block" in msg

    def test_embedding_before_block_still_ok(self):
        """No regresión: el orden correcto (ya cubierto en C1) sigue pasando."""
        src = """
PROJECT F5b
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  BLOCK enc TRANSFORMER
    LAYERS 1
  END
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert res.ok, res.errors
