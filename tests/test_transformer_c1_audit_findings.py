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
        # C2 + re-auditoría ronda 3: `supported` conserva el significado
        # histórico de ejecución (False hasta C4) y `lowering_supported` expone
        # aparte la matemática C2. report.ok falla cerrado y ningún consumidor
        # externo ve un "supported=True" engañoso.
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        assert report.ok is False
        node = next(n for n in report.nodes if n.node == "N")
        assert node.supported is False             # ejecución cerrada hasta C4
        assert node.lowering_supported is True     # matemática/lowering C2
        assert node.lowering_ok is True
        assert node.differentiable is True
        assert "TRANSFORMER_BLOQUE C4" in node.reason

    def test_trainable_parameters_now_exposed_by_c2(self):
        # (Antes de C2 este test afirmaba lista vacía; C2 entrega el manifest,
        # así que los parámetros del bloque se REPORTAN — lowering auditable.)
        from matrixai.parameters.network_params import transformer_block_param_count
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        block_params = [p for p in report.trainable_parameters if ".enc." in p.path]
        total = sum(
            (p.shape[0] * p.shape[1] if len(p.shape) == 2 else p.shape[0])
            for p in block_params
        )
        assert total == transformer_block_param_count(4, 128, 4 * 128)

    def test_layer_manifest_does_not_leak_vocab_zero(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        # Ni un shape=[0, dim] en ningún sitio del manifest de capas.
        for entry in report.layer_manifest:
            for param in entry.get("parameters", []):
                shape = param.get("shape")
                if shape:
                    assert shape[0] != 0, f"vocab=0 sentinel leaked into manifest: {entry}"

    def test_layer_manifest_reports_real_transformer_block(self):
        # (Antes de C2 la entrada era un placeholder "pending"; ahora es la
        # entrada real con el lowering completo.)
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        block_entries = [e for e in report.layer_manifest if e.get("block_name") == "enc"]
        assert len(block_entries) == 1
        entry = block_entries[0]
        assert entry["differentiable"] is True
        assert entry["layers"] == 4 and entry["heads"] == 4
        assert len(entry["parameters"]) == 4 * 12  # 12 params por encoder layer

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


# ---------------------------------------------------------------------------
# Auditoría 2ª ronda (2026-07-10): la máquina de fases explícita
# (ids → embedded → transformed → pooled) cierra las rutas equivalentes que
# el gating por rank dejaba abiertas, y el backend falla cerrado para
# CUALQUIER composite con INPUT SEQUENCE hasta que C2 implemente su forward.
# ---------------------------------------------------------------------------

class TestRonda2ReshapeFingiendoStreamEmbebido:
    """[ALTA] Reshape [L]→[16,4] sobre los ids crudos fabricaba un rank-2 que
    el bloque aceptaba como stream embebido (sin EMBEDDING, DIM falso, sin POOL)."""

    def test_reshape_on_raw_ids_rejected(self):
        src = """
PROJECT R2A
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  LAYER Reshape target=[16,4]
  BLOCK enc TRANSFORMER
    LAYERS 1
    HEADS 4
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "raw token ids" in msg and "EMBEDDING" in msg

    def test_any_layer_on_raw_ids_rejected(self):
        """Ninguna capa puede tocar los ids crudos — LayerNorm tampoco.
        (Una red solo-Dense ni siquiera es composite: parsea como dense_network
        y el checker denso ya rechaza el INPUT no-VECTOR por su propia vía.)"""
        src = """
PROJECT R2Ab
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  LAYER LayerNorm
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok
        assert any("raw token ids" in e for e in res.errors)


class TestRonda2BloqueTrasPool:
    """[ALTA] POOL antes del bloque + Reshape que reconstruye rank-2: el bloque
    aceptaba [64,2] como stream, con DIM cambiado de 128 a 2 y sin POOL posterior."""

    def test_pool_then_reshape_then_block_rejected(self):
        src = """
PROJECT R2B
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  POOL mean
  LAYER Reshape target=[64,2]
  BLOCK enc TRANSFORMER
    LAYERS 1
    HEADS 2
  END
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert not res.ok
        msg = "; ".join(res.errors)
        assert "between the EMBEDDING and the POOL" in msg

    def test_block_expects_exact_embedded_shape(self):
        """Defensa adicional: el bloque exige EXACTAMENTE [L, embedding.dim] —
        un CONCAT que pisara el stream en fase embedded también fallaría."""
        src = """
PROJECT R2Bb
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 128
  CONCAT [tok] -> pisado
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
        assert any("expects the embedded stream" in e for e in res.errors)


class TestRonda2SequenceInputFallaCerradoEnBackend:
    """[ALTA] Un composite con INPUT SEQUENCE pero SIN transformer typechequea
    bien (shapes coherentes) pero su forward stdlib no existe hasta C2 — el
    backend debe marcarlo no soportado en vez de dejar construir el ParameterSet
    y reventar dentro de composite_forward."""

    _SRC = """
PROJECT R2C
SEQUENCE Texto
  length = 64
  vocab_size = 30000
END
NETWORK N
  INPUT Texto
  EMBEDDING tok FROM Texto DIM 2
  POOL mean
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END

GRAPH
  Texto -> N
END
"""

    def test_backend_fails_closed_without_transformer(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        assert report.ok is False
        node = next(n for n in report.nodes if n.node == "N")
        assert node.supported is False
        assert node.differentiable is False
        assert "SEQUENCE input" in node.reason
        assert report.trainable_parameters == []

    def test_layer_manifest_reports_pending_sequence_input(self):
        prog = parse_text(self._SRC)
        report = BackendContractAnalyzer().analyze(prog)
        pending = [e for e in report.layer_manifest if e.get("layer_type") == "SequenceInput"]
        assert len(pending) == 1
        assert pending[0]["differentiable"] is False

    def test_parameter_manifest_raises_for_sequence_input(self):
        from matrixai.parameters.network_params import composite_network_parameter_manifest
        import pytest as _pytest
        prog = parse_text(self._SRC)
        net = prog.networks[0]
        res = check_composite_network_types(net, {}, _sequences(prog))
        assert res.ok  # el typecheck en sí es correcto — el bloqueo es del backend
        with _pytest.raises(NotImplementedError, match="SEQUENCE input"):
            composite_network_parameter_manifest(net.name, net, res)

    def test_tabular_composite_backend_still_supported(self):
        """No regresión: el composite tabular clásico sigue soportado."""
        src = """
PROJECT R2Cb
VECTOR Product[2]
  category_id: Integer[0, 100]
  price: Scalar
END
NETWORK Net
  INPUT Product
  EMBEDDING cat FROM category_id VOCAB 100 DIM 8
  CONCAT [cat, price] -> features
  LAYER Dense units=2 activation=softmax
  OUTPUT label: ProbabilityMap[a, b]
END

GRAPH
  Product -> Net
END
"""
        prog = parse_text(src)
        report = BackendContractAnalyzer().analyze(prog)
        assert report.ok is True
        node = next(n for n in report.nodes if n.node == "Net")
        assert node.supported is True
        assert len(report.trainable_parameters) > 0


class TestRonda2FaseNoInferidaDelRank:
    def test_final_pool_check_uses_phase_not_rank(self):
        """Reshape que aplana tras el bloque: UN error de gating (no dos) — la
        comprobación final consulta la fase y no re-dispara sobre el rank."""
        src = """
PROJECT R2D
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
        gating = [e for e in res.errors if "Reshape" in e]
        missing_pool = [e for e in res.errors if "missing POOL" in e]
        assert len(gating) == 1
        assert missing_pool == []  # fase "invalid" suprime la cascada

    def test_happy_path_still_ok(self):
        """La ruta canónica del contrato sigue pasando tras la máquina de fases."""
        src = """
PROJECT R2E
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
  LAYER Dense units=64 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT clase: ProbabilityMap[a,b]
END
"""
        prog = parse_text(src)
        res = check_composite_network_types(prog.networks[0], {}, _sequences(prog))
        assert res.ok, res.errors
        assert res.input_is_sequence is True
