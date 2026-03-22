"""
tests/test_typesimplify.py — Integration tests for TypeAwareSimplifier.

Each test builds a FakeFunction, inserts prototype entries into PROTOTYPES,
runs TypeAwareSimplifier.run(), and checks func.hir contents.

XRAM local variable lookup (get_locals) naturally returns [] under the mocked
IDA modules, so _augment_with_local_vars is a no-op in most tests.
"""

import pytest
from unittest.mock import patch

from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param
from pseudo8051.locals import LocalVar

from .helpers import make_single_block_func


# ── Fixture: clean PROTOTYPES between tests ───────────────────────────────────

@pytest.fixture(autouse=True)
def clean_prototypes():
    """Remove any entries added to PROTOTYPES during a test."""
    keys_before = set(PROTOTYPES.keys())
    yield
    for k in list(PROTOTYPES.keys()):
        if k not in keys_before:
            del PROTOTYPES[k]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _texts(func):
    """Extract .text strings from func.hir (Statement nodes only)."""
    from pseudo8051.ir.hir import Statement
    return [n.text for n in func.hir if isinstance(n, Statement)]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSingleRegParam:
    def test_basic_accum_relay_with_param(self):
        """A = R7; XRAM[X] = A; with proto count→R7 → XRAM[X] = count;"""
        PROTOTYPES["f"] = FuncProto(
            return_type="void",
            params=[Param("count", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("f", ["A = R7;", "XRAM[X] = A;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert texts == ["XRAM[X] = count;"]

    def test_no_proto_structural_patterns_run(self):
        """Without a prototype, structural patterns (AccumRelayPattern) still fire."""
        func = make_single_block_func("unknown_fn", ["A = R7;", "XRAM[X] = A;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        # AccumRelayPattern collapses A-relay even without reg_map
        assert "XRAM[X] = R7;" in texts
        assert "A = R7;" not in texts
        assert "XRAM[X] = A;" not in texts

    def test_return_statement_preserved(self):
        """return; statement passes through unchanged."""
        PROTOTYPES["g"] = FuncProto(
            return_type="void",
            params=[Param("val", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("g", ["A = R7;", "XRAM[X] = A;", "return;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "return;" in texts


class TestUsercallParamSubst:
    def test_three_relay_pairs(self):
        """Three A-relay pairs with params in R7, R5, R3 → all substituted."""
        PROTOTYPES["h"] = FuncProto(
            return_type="void",
            params=[
                Param("H", "uint8_t", ("R7",)),
                Param("M", "uint8_t", ("R5",)),
                Param("L", "uint8_t", ("R3",)),
            ],
        )
        func = make_single_block_func("h", [
            "A = R7;", "XRAM[X1] = A;",
            "A = R5;", "XRAM[X2] = A;",
            "A = R3;", "XRAM[X3] = A;",
        ])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "XRAM[X1] = H;" in texts
        assert "XRAM[X2] = M;" in texts
        assert "XRAM[X3] = L;" in texts

    def test_pair_param(self):
        """16-bit param in R6R7: pair substituted in expression."""
        PROTOTYPES["p"] = FuncProto(
            return_type="void",
            params=[Param("val", "uint16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("p", ["XRAM[X] = R6R7;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "XRAM[X] = val;" in texts


class TestRetvalRenaming:
    def test_u32_retval(self):
        """R4R5R6R7 = callee(R6R7); → uint32_t retval1 = callee(arg);"""
        PROTOTYPES["callee"] = FuncProto(
            return_type="uint32_t",
            return_regs=("R4", "R5", "R6", "R7"),
            params=[Param("arg", "uint16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("caller", ["R4R5R6R7 = callee(R6R7);"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert len(texts) == 1
        assert texts[0] == "uint32_t retval1 = callee(arg);"

    def test_u8_retval(self):
        """A = getter(); with caller return → uint8_t retval1 = getter();"""
        PROTOTYPES["getter"] = FuncProto(
            return_type="uint8_t",
            return_regs=("A",),
            params=[],
        )
        func = make_single_block_func("user", ["A = getter();"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert texts[0] == "uint8_t retval1 = getter();"


class TestXramLocalDecl:
    def test_local_declaration_prepended(self):
        """Declared XRAM local → declaration comment prepended to hir."""
        PROTOTYPES["fn"] = FuncProto(return_type="void", params=[])
        func = make_single_block_func("fn", ["XRAM[EXT_DC8A] = R7;"])

        local = LocalVar(name="var1", type="uint16_t", addr=0xdc8a)

        with patch("pseudo8051.locals.get_locals", return_value=[local]), \
             patch("pseudo8051.constants.resolve_ext_addr",
                   side_effect=lambda a: f"EXT_{a:04X}"):
            TypeAwareSimplifier().run(func)

        from pseudo8051.ir.hir import Statement
        decl_stmts = [n for n in func.hir if isinstance(n, Statement)
                      and "uint16_t var1;" in n.text]
        assert len(decl_stmts) == 1
        decl = decl_stmts[0].text
        assert decl.startswith("uint16_t var1;")
        assert "EXT_DC8A" in decl
        assert "0xdc8a" in decl

    def test_local_decl_at_start(self):
        """Local variable declaration is the first node in hir."""
        PROTOTYPES["fn2"] = FuncProto(return_type="void", params=[])
        func = make_single_block_func("fn2", ["return;"])

        local = LocalVar(name="cnt", type="uint8_t", addr=0xdc00)

        with patch("pseudo8051.locals.get_locals", return_value=[local]), \
             patch("pseudo8051.constants.resolve_ext_addr",
                   return_value="EXT_DC00"):
            TypeAwareSimplifier().run(func)

        from pseudo8051.ir.hir import Statement
        assert isinstance(func.hir[0], Statement)
        assert "uint8_t cnt;" in func.hir[0].text
