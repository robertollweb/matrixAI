"""P21 C2 — Parser IMPORT: sintaxis, validaciones, IR, integración."""
from __future__ import annotations

import pytest

from matrixai.ir.schema import ImportSpec, MatrixAIProgram
from matrixai.parser.parser import MatrixAIParseError, parse_text

# ── helpers ───────────────────────────────────────────────────────────────────

_MINIMAL_MXAI = """\
PROJECT Test

VECTOR Input[1]
  x
END

GRAPH
  Input -> Output
END
"""

_IMPORT_FROZEN = "IMPORT TextEncoder FROM registry text_encoder@v1 FROZEN"
_IMPORT_TRAINABLE = "IMPORT Sentiment FROM registry sentiment_classifier@v1 TRAINABLE"


def _with_imports(*imports: str, body: str = _MINIMAL_MXAI) -> str:
    """Prepend IMPORT lines to a minimal .mxai body (after PROJECT)."""
    header = "PROJECT Test\n"
    rest = "\n".join(body.splitlines()[1:])  # drop original PROJECT line
    return header + "\n".join(imports) + "\n" + rest


# ── ImportSpec dataclass ──────────────────────────────────────────────────────

def test_import_spec_is_frozen():
    imp = ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="FROZEN")
    with pytest.raises(Exception):  # FrozenInstanceError
        imp.alias = "Changed"  # type: ignore[misc]


def test_import_spec_resolved_fields_default_empty():
    imp = ImportSpec(alias="Enc", registry_name="enc", version="v1", mode="TRAINABLE")
    assert imp.resolved_entry_hash == ""
    assert imp.resolved_at == ""


def test_import_spec_accepts_explicit_resolved_fields():
    imp = ImportSpec(
        alias="Enc", registry_name="enc", version="v1", mode="FROZEN",
        resolved_entry_hash="sha256:" + "a" * 64,
        resolved_at="2026-06-01T00:00:00+00:00",
    )
    assert imp.resolved_entry_hash.startswith("sha256:")


# ── program.imports default ───────────────────────────────────────────────────

def test_program_imports_field_empty_by_default():
    prog = parse_text(_MINIMAL_MXAI)
    assert prog.imports == []


# ── valid IMPORT parsing ──────────────────────────────────────────────────────

def test_parser_accepts_import_frozen():
    src = _with_imports(_IMPORT_FROZEN)
    prog = parse_text(src)
    assert len(prog.imports) == 1
    imp = prog.imports[0]
    assert imp.alias == "TextEncoder"
    assert imp.registry_name == "text_encoder"
    assert imp.version == "v1"
    assert imp.mode == "FROZEN"


def test_parser_accepts_import_trainable():
    src = _with_imports(_IMPORT_TRAINABLE)
    prog = parse_text(src)
    assert prog.imports[0].mode == "TRAINABLE"


def test_parser_accepts_multiple_imports():
    src = _with_imports(_IMPORT_FROZEN, _IMPORT_TRAINABLE)
    prog = parse_text(src)
    assert len(prog.imports) == 2
    aliases = {imp.alias for imp in prog.imports}
    assert aliases == {"TextEncoder", "Sentiment"}


def test_parser_accepts_tag_version():
    src = _with_imports("IMPORT Enc FROM registry text_encoder@latest FROZEN")
    prog = parse_text(src)
    assert prog.imports[0].version == "latest"


def test_parser_accepts_dotted_version():
    src = _with_imports("IMPORT Enc FROM registry text_encoder@v1.2 FROZEN")
    prog = parse_text(src)
    assert prog.imports[0].version == "v1.2"


def test_parser_accepts_prod_tag():
    src = _with_imports("IMPORT Enc FROM registry text_encoder@prod FROZEN")
    prog = parse_text(src)
    assert prog.imports[0].version == "prod"


# ── resolve fields default to empty (C4 fills them) ──────────────────────────

def test_parsed_import_resolved_fields_are_empty():
    src = _with_imports(_IMPORT_FROZEN)
    imp = parse_text(src).imports[0]
    assert imp.resolved_entry_hash == ""
    assert imp.resolved_at == ""


# ── syntax errors ─────────────────────────────────────────────────────────────

def test_parser_rejects_import_without_version():
    with pytest.raises(MatrixAIParseError):
        parse_text(_with_imports("IMPORT Enc FROM registry text_encoder FROZEN"))


def test_parser_rejects_import_invalid_mode():
    with pytest.raises(MatrixAIParseError):
        parse_text(_with_imports("IMPORT Enc FROM registry text_encoder@v1 STATIC"))


def test_parser_rejects_import_missing_from():
    with pytest.raises(MatrixAIParseError):
        parse_text(_with_imports("IMPORT Enc registry text_encoder@v1 FROZEN"))


def test_parser_rejects_import_missing_registry_keyword():
    with pytest.raises(MatrixAIParseError):
        parse_text(_with_imports("IMPORT Enc FROM text_encoder@v1 FROZEN"))


def test_parser_rejects_import_uppercase_registry_name():
    """Registry names must start lowercase per contract naming convention."""
    with pytest.raises(MatrixAIParseError):
        parse_text(_with_imports("IMPORT Enc FROM registry TextEncoder@v1 FROZEN"))


# ── duplicate alias and collision validation ──────────────────────────────────

def test_parser_rejects_duplicate_alias():
    src = _with_imports(
        "IMPORT Enc FROM registry text_encoder@v1 FROZEN",
        "IMPORT Enc FROM registry other_encoder@v1 TRAINABLE",
    )
    with pytest.raises(MatrixAIParseError, match="more than once"):
        parse_text(src)


def test_parser_rejects_alias_colliding_with_vector():
    # The minimal body has a VECTOR named "Input"
    src = _with_imports("IMPORT Input FROM registry text_encoder@v1 FROZEN")
    with pytest.raises(MatrixAIParseError, match="collides"):
        parse_text(src)


def test_parser_rejects_alias_colliding_with_network():
    src = """\
PROJECT Test

IMPORT MyNet FROM registry text_encoder@v1 FROZEN

VECTOR Input[1]
  x
END

NETWORK MyNet
  INPUT Input
  LAYER Dense units=4 activation=relu
  OUTPUT y: Probability
END

GRAPH
  Input -> MyNet
END
"""
    with pytest.raises(MatrixAIParseError, match="collides"):
        parse_text(src)


# ── to_dict includes imports ──────────────────────────────────────────────────

def test_program_to_dict_includes_imports():
    src = _with_imports(_IMPORT_FROZEN)
    prog = parse_text(src)
    d = prog.to_dict()
    assert "imports" in d
    assert len(d["imports"]) == 1
    imp_dict = d["imports"][0]
    assert imp_dict["alias"] == "TextEncoder"
    assert imp_dict["mode"] == "FROZEN"


def test_program_to_dict_omits_imports_when_empty():
    prog = parse_text(_MINIMAL_MXAI)
    d = prog.to_dict()
    assert "imports" not in d


# ── existing programs unaffected ──────────────────────────────────────────────

def test_existing_program_without_imports_unchanged():
    """Parsing a P1-P20 program must produce imports=[] — no regressions."""
    prog = parse_text(_MINIMAL_MXAI)
    assert isinstance(prog, MatrixAIProgram)
    assert prog.imports == []
    assert prog.project == "Test"
