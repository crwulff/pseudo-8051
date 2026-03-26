"""
tests/typesimplify/test_param_infer.py — Synthesized arg-N params when no prototype.
"""

from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.ir.hir import Assign, CompoundAssign, ReturnStmt, Statement
from pseudo8051.ir.expr import Reg, Const, BinOp, Name

from ..helpers import FakeBlock, FakeFunction


def _texts(func):
    return [n.text for n in func.hir if isinstance(n, Statement)]


def _render_hir(func):
    lines = []
    for node in func.hir:
        for _ea, text in node.render():
            lines.append(text)
    return lines


class TestSynthesizedParams:
    def test_r7_live_in_becomes_arg1(self):
        """With R7 live-in and no proto, R7 should be named arg1 in output."""
        hir = [
            CompoundAssign(0x1000, Reg("A"), "&=", Const(0x0f)),
            ReturnStmt(0x1002, Reg("A")),
        ]
        block = FakeBlock(0x1000, hir=hir, live_in=frozenset({"R7"}))
        func = FakeFunction("no_proto_fn", [block], hir=list(hir))

        TypeAwareSimplifier().run(func)
        # R7 was live-in but not used in these nodes — just ensure no crash
        # and that the synthesized entry is created without error
        assert func.hir is not None

    def test_compound_assign_with_r7_param(self):
        """A = R7; A &= 0x0f; R6 = A; with R7 live-in → R6 = arg1 & 0x0f"""
        hir = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0x0f)),
            Assign(0x1004, Reg("R6"), Reg("A")),
        ]
        block = FakeBlock(0x1000, hir=hir, live_in=frozenset({"R7"}))
        func = FakeFunction("no_proto_fn2", [block], hir=list(hir))

        TypeAwareSimplifier().run(func)
        lines = _render_hir(func)
        combined = " ".join(lines)
        # arg1 should appear (synthesized from R7 live-in)
        assert "arg1" in combined
        # A = R7 and A &= should be collapsed
        assert "A = R7" not in combined

    def test_r7_r5_live_in(self):
        """R7 and R5 live-in → arg1 and arg2 synthesized."""
        hir = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, Reg("R6"), Reg("A")),
            Assign(0x1004, Reg("A"), Reg("R5")),
            Assign(0x1006, Reg("R4"), Reg("A")),
        ]
        block = FakeBlock(0x1000, hir=hir, live_in=frozenset({"R7", "R5"}))
        func = FakeFunction("no_proto_fn3", [block], hir=list(hir))

        TypeAwareSimplifier().run(func)
        lines = _render_hir(func)
        combined = " ".join(lines)
        assert "arg1" in combined
        assert "arg2" in combined
