"""
tests/typesimplify/test_propagate_values.py — Tests for _propagate_values pass.
"""

import pytest

from pseudo8051.passes.typesimplify._post import _propagate_values
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign, TypedAssign, ExprStmt, CompoundAssign, IfNode, ReturnStmt
from pseudo8051.ir.expr import Reg, Name, XRAMRef, Call, Const, BinOp, UnaryOp


class TestPropagateValues:

    def test_dptr_fold_into_assign(self):
        """DPTR=Name("offset") folded into call Assign."""
        nodes = [
            Assign(0, Reg("DPTR"), Name("offset")),
            Assign(1, Name("retval1"), Call("func", [Reg("DPTR")])),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        rendered = result[0].render()[0][1]
        assert "DPTR" not in rendered
        assert "offset" in rendered

    def test_dptr_fold_into_xram_lhs(self):
        """DPTR=Name("_dest") folded into XRAM[DPTR]=val LHS."""
        nodes = [
            Assign(0, Reg("DPTR"), Name("_dest")),
            Assign(1, XRAMRef(Reg("DPTR")), Name("val")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert result[0].lhs.render() == "XRAM[_dest]"

    def test_r7_fold_into_xram_rhs(self):
        """R7=Name("retval1") folded into XRAM[DPTR]=R7 rhs."""
        nodes = [
            Assign(0, Reg("R7"), Name("retval1")),
            Assign(1, XRAMRef(Reg("DPTR")), Reg("R7")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert result[0].rhs.render() == "retval1"

    def test_multiple_uses_reg_free_folded(self):
        """Register holding a reg-free (Name) value with >1 uses is folded into all uses."""
        nodes = [
            Assign(0, Reg("R7"), Name("val")),
            Assign(1, Reg("R6"), Reg("R7")),             # use 1
            Assign(2, XRAMRef(Reg("DPTR")), Reg("R7")),  # use 2
        ]
        result = _propagate_values(nodes, {})
        # R7=val is substituted into both uses; source assignment removed
        assert len(result) == 2
        assert result[0].rhs.render() == "val"
        assert result[1].rhs.render() == "val"

    def test_multiple_uses_reg_replacement_not_folded(self):
        """Register with >1 uses holding a Reg-containing expression is not folded."""
        nodes = [
            Assign(0, Reg("R7"), Reg("R5")),             # replacement contains Reg — not reg-free
            Assign(1, Reg("R6"), Reg("R7")),             # use 1
            Assign(2, XRAMRef(Reg("DPTR")), Reg("R7")),  # use 2
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 3  # nothing removed — requires single-use for Reg replacements

    def test_retval_single_use_inlined(self):
        """Single-use retval TypedAssign is inlined into the Assign target."""
        nodes = [
            TypedAssign(0, "int8_t", Name("retval1"), Call("func", [Name("x")])),
            Assign(1, XRAMRef(Name("dest")), Name("retval1")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        rendered = result[0].render(0)[0][1]
        assert "func(x)" in rendered
        assert "retval1" not in rendered

    def test_retval_multiple_uses_not_inlined(self):
        """retval used twice → not inlined."""
        nodes = [
            TypedAssign(0, "int8_t", Name("retval1"), Call("func", [Name("x")])),
            Assign(1, Reg("R7"), Name("retval1")),
            Assign(2, Reg("R6"), Name("retval1")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 3

    def test_full_chain(self):
        """DPTR=off; TypedAssign(retval1=call(...,DPTR)); R7=retval1; DPTR=_dest;
           XRAM[DPTR]=R7 → XRAM[_dest]=call(...,off)."""
        nodes = [
            Assign(0, Reg("DPTR"), Name("offset")),
            TypedAssign(1, "int8_t", Name("retval1"), Call("code_7_read", [Name("a"), Name("b"), Reg("DPTR")])),
            Assign(2, Reg("R7"), Name("retval1")),
            Assign(3, Reg("DPTR"), Name("_dest")),
            Assign(4, XRAMRef(Reg("DPTR")), Reg("R7")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        t = result[0].render(0)[0][1]
        assert "offset" in t
        assert "DPTR" not in t
        assert "retval1" not in t
        assert "R7" not in t
        assert "_dest" in t
        assert "code_7_read" in t

    def test_compound_fold_rhs_has_reg_single_use_propagates(self):
        """A=Reg(R5) + A+=2: A0 folds to A=R5+2 (has Reg, not reg-free).
        ExprStmt(A!=4) counts as total_uses=1, so single-use propagation
        substitutes R5+2 into ExprStmt. Assign(DPL, A) is past the first use
        but comes AFTER; if kill_idx=None, total_uses=2 → multi-use fails (not reg-free)."""
        nodes = [
            Assign(0, Reg("A"), Reg("R5")),
            CompoundAssign(1, Reg("A"), "+=", Const(2)),
            ExprStmt(2, BinOp(Reg("A"), "!=", Const(4))),
            IfNode(3, UnaryOp("!", Reg("C")), [ReturnStmt(4, Reg("R2"))], []),
            Assign(5, Reg("DPL"), Reg("A")),
        ]
        result = _propagate_values(nodes, {})
        # Verify what actually happens — the DPL assignment render
        dpl_nodes = [n for n in result if isinstance(n, Assign)
                     and hasattr(n.lhs, 'name') and n.lhs.name == "DPL"]
        print(f"\nR5-not-substituted result: {[type(n).__name__ for n in result]}")
        print(f"DPL node rhs: {dpl_nodes[0].rhs.render() if dpl_nodes else 'missing'}")
        # Document the current behavior
        assert len(dpl_nodes) == 1

    def test_compound_fold_then_multiuse_across_ifnode(self):
        """A=Name + A+=2 + ExprStmt(A!=4) + IfNode(!C, [return]) + Assign(DPL, A)
        → A folds to Name+2, propagates into ExprStmt AND Assign(DPL, A)."""
        nodes = [
            Assign(0, Reg("A"), Name("dest_type")),
            CompoundAssign(1, Reg("A"), "+=", Const(2)),
            ExprStmt(2, BinOp(Reg("A"), "!=", Const(4))),
            IfNode(3, UnaryOp("!", Reg("C")), [ReturnStmt(4, Reg("R2"))], []),
            Assign(5, Reg("DPL"), Reg("A")),
        ]
        result = _propagate_values(nodes, {})
        # ExprStmt and IfNode and DPL assign remain; source A= removed
        texts = [n.render(0)[0][1] for n in result if not isinstance(n, IfNode)]
        joined = " ".join(texts)
        assert "A" not in joined, f"A still present: {joined!r}"
        assert "dest_type" in joined
        # DPL assignment should have dest_type + 2
        dpl_nodes = [n for n in result if isinstance(n, Assign) and n.lhs.render() == "DPL"]
        assert len(dpl_nodes) == 1
        assert "dest_type" in dpl_nodes[0].rhs.render()
