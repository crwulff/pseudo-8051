"""
tests/typesimplify/test_propagate_values.py — Tests for _propagate_values pass.
"""

import pytest

from pseudo8051.passes.typesimplify._post import _propagate_values
from pseudo8051.passes.typesimplify._propagate import _subst_from_reg_exprs
from pseudo8051.passes.patterns._utils import VarInfo, _canonicalize_expr
from pseudo8051.ir.hir import Assign, TypedAssign, ExprStmt, CompoundAssign, IfNode, ReturnStmt, SwitchNode
from pseudo8051.ir.hir._base import NodeAnnotation
from pseudo8051.ir.expr import Reg, Name, XRAMRef, Call, Const, BinOp, UnaryOp


class TestAdditiveFold:
    """Verify _canonicalize_expr collapses additive chains."""

    def test_dec_folds(self):
        """(R6 + 0x61) - 1 → R6 + 0x60"""
        e = BinOp(BinOp(Reg("R6"), '+', Const(0x61)), '-', Const(1))
        result = _canonicalize_expr(e, {}, [], {})
        assert result.render() == "R6 + 0x60", result.render()

    def test_add_const_folds(self):
        """(R6 + 0x60) + 0xfe → R6 + 0x5e  (with 8-bit wrap)"""
        e = BinOp(BinOp(Reg("R6"), '+', Const(0x60)), '+', Const(0xfe))
        result = _canonicalize_expr(e, {}, [], {})
        assert result.render() == "R6 + 0x5e", result.render()

    def test_chain_folds(self):
        """((R6 + 0x61) - 1) + 0xfe → R6 + 0x5e in one pass"""
        e = BinOp(BinOp(BinOp(Reg("R6"), '+', Const(0x61)), '-', Const(1)), '+', Const(0xfe))
        result = _canonicalize_expr(e, {}, [], {})
        assert result.render() == "R6 + 0x5e", result.render()

    def test_cancel_to_zero(self):
        """(R6 + 5) - 5 → R6"""
        e = BinOp(BinOp(Reg("R6"), '+', Const(5)), '-', Const(5))
        result = _canonicalize_expr(e, {}, [], {})
        assert result.render() == "R6", result.render()

    def test_post_inc_const_folds(self):
        """Const(0xe17c)++ → Const(0xe17c)  (post-increment: use original value)"""
        e = UnaryOp('++', Const(0xe17c), post=True)
        result = _canonicalize_expr(e, {}, [], {})
        assert isinstance(result, Const)
        assert result.value == 0xe17c

    def test_pre_inc_const_folds(self):
        """++Const(0x28) → Const(0x29)  (pre-increment: use incremented value)"""
        e = UnaryOp('++', Const(0x28), post=False)
        result = _canonicalize_expr(e, {}, [], {})
        assert isinstance(result, Const)
        assert result.value == 0x29

    def test_post_dec_const_folds(self):
        """Const(0x27)-- → Const(0x27)  (post-decrement: use original value)"""
        e = UnaryOp('--', Const(0x27), post=True)
        result = _canonicalize_expr(e, {}, [], {})
        assert isinstance(result, Const)
        assert result.value == 0x27

    def test_pre_dec_const_folds(self):
        """--Const(0x28) → Const(0x27)  (pre-decrement: use decremented value)"""
        e = UnaryOp('--', Const(0x28), post=False)
        result = _canonicalize_expr(e, {}, [], {})
        assert isinstance(result, Const)
        assert result.value == 0x27

    def test_post_inc_const_with_alias_drops_alias(self):
        """Const(0xe179, alias='EXT_E179')++ → Const(0xe179) (alias removed)"""
        e = UnaryOp('++', Const(0xe179, alias='EXT_E179'), post=True)
        result = _canonicalize_expr(e, {}, [], {})
        assert isinstance(result, Const)
        assert result.value == 0xe179
        assert result.alias is None


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


    def test_dptr_chain_pre_incr_propagation(self):
        """DPTR=K; XRAM[EXT]=v0; XRAM[++DPTR]=v1; XRAM[++DPTR]=v2; XRAM[++DPTR]=v3
        → all XRAM addresses resolved sequentially without gaps.

        Regression test for the DPTR chain propagation fix:
        after substituting DPTR=K into XRAM[++DPTR] → XRAM[K+1], a synthetic
        Assign(DPTR, K+1) must be injected so subsequent XRAM[++DPTR] nodes
        can continue propagating rather than falling back to stale annotations.
        """
        from pseudo8051.ir.hir import BreakStmt
        K = 0xe179
        nodes = [
            Assign(0x100, Reg("DPTR"), Const(K)),
            Assign(0x102, XRAMRef(Const(K)), Const(0x2d)),       # XRAM[EXT_E179] = 0x2d
            Assign(0x104, XRAMRef(UnaryOp('++', Reg("DPTR"), post=False)), Const(0x33)),  # XRAM[++DPTR]
            Assign(0x106, XRAMRef(UnaryOp('++', Reg("DPTR"), post=False)), Const(0x29)),  # XRAM[++DPTR]
            Assign(0x108, XRAMRef(UnaryOp('++', Reg("DPTR"), post=False)), Const(0x1e)),  # XRAM[++DPTR]
            BreakStmt(0x10a),
        ]
        result = _propagate_values(nodes, {})

        # DPTR assignment should be gone; all XRAM refs should be constants
        texts = [n.render(0)[0][1] for n in result]
        assert not any("DPTR" in t for t in texts), \
            f"DPTR still present in output: {texts}"
        # Sequential addresses: 0xe179, 0xe17a, 0xe17b, 0xe17c
        assert any(f"XRAM[0x{K:x}]" in t for t in texts), \
            f"XRAM[0xe179] not found: {texts}"
        assert any(f"XRAM[0x{K+1:x}]" in t for t in texts), \
            f"XRAM[0xe17a] not found: {texts}"
        assert any(f"XRAM[0x{K+2:x}]" in t for t in texts), \
            f"XRAM[0xe17b] not found: {texts}"
        assert any(f"XRAM[0x{K+3:x}]" in t for t in texts), \
            f"XRAM[0xe17c] not found: {texts}"


    def test_dptr_chain_xram_post_incr_then_pre_incr(self):
        """DPTR=K; XRAM[DPTR++]=v0; XRAM[DPTR++]=v1; XRAM[DPTR]=v2
        → all XRAM addresses resolved sequentially without gaps.

        Regression for the post-increment XRAM form: the RMW collapser folds
        'XRAM[DPTR]=A; DPTR++;' → 'XRAM[DPTR++]=A'.  Without the fix,
        _xram_pre_incr_delta only detected XRAM[++DPTR] (pre-increment) and
        skipped XRAM[DPTR++] (post-increment), so no synthetic DPTR=K+1 was
        injected, causing downstream XRAM[++DPTR] nodes to use stale annotations.
        """
        from pseudo8051.ir.hir import BreakStmt
        K = 0xdc68
        nodes = [
            Assign(0x100, Reg("DPTR"), Const(K)),
            Assign(0x102, XRAMRef(UnaryOp('++', Reg("DPTR"), post=True)), Const(0xff)),  # XRAM[DPTR++]
            Assign(0x104, XRAMRef(UnaryOp('++', Reg("DPTR"), post=True)), Const(0x22)),  # XRAM[DPTR++]
            Assign(0x106, XRAMRef(Reg("DPTR")), Const(0x9f)),                             # XRAM[DPTR]
            BreakStmt(0x108),
        ]
        result = _propagate_values(nodes, {})

        texts = [n.render(0)[0][1] for n in result]
        assert not any("DPTR" in t for t in texts), \
            f"DPTR still present in output: {texts}"
        assert any(f"XRAM[0x{K:x}]"   in t for t in texts), f"XRAM[0xdc68] not found: {texts}"
        assert any(f"XRAM[0x{K+1:x}]" in t for t in texts), f"XRAM[0xdc69] not found: {texts}"
        assert any(f"XRAM[0x{K+2:x}]" in t for t in texts), f"XRAM[0xdc6a] not found: {texts}"

    def test_dptr_chain_post_incr_then_pre_incr(self):
        """DPTR=K; XRAM[K]=v0; DPTR++; XRAM[K+1]=v1; XRAM[++DPTR]=v2; XRAM[++DPTR]=v3
        → all XRAM addresses resolved sequentially without gaps.

        Regression for case-159 pattern: the AEE1 callee inlines as a post-increment
        DPTR++ (ExprStmt) between two constant-address XRAM writes.  Without the fix,
        DPTR=K is substituted into DPTR++ → ExprStmt(Const(K)) (no-op), no synthetic
        is injected, and subsequent XRAM[++DPTR] nodes use a stale annotation that is
        off by one.
        """
        from pseudo8051.ir.hir import BreakStmt
        K = 0xe179
        nodes = [
            Assign(0x100, Reg("DPTR"), Const(K)),
            Assign(0x102, XRAMRef(Const(K)),     Const(0x21)),   # XRAM[EXT_E179] = 0x21
            ExprStmt(0x104, UnaryOp('++', Reg("DPTR"), post=True)),  # DPTR++
            Assign(0x106, XRAMRef(Const(K + 1)), Const(0x1d)),   # XRAM[0xe17a] = 0x1d
            Assign(0x108, XRAMRef(UnaryOp('++', Reg("DPTR"), post=False)), Const(0x26)),  # XRAM[++DPTR]
            Assign(0x10a, XRAMRef(UnaryOp('++', Reg("DPTR"), post=False)), Const(0x22)),  # XRAM[++DPTR]
            Assign(0x10c, XRAMRef(UnaryOp('++', Reg("DPTR"), post=False)), Const(0xff)),  # XRAM[++DPTR]
            BreakStmt(0x10e),
        ]
        result = _propagate_values(nodes, {})

        texts = [n.render(0)[0][1] for n in result]
        assert not any("DPTR" in t for t in texts), \
            f"DPTR still present in output: {texts}"
        assert any(f"XRAM[0x{K:x}]"   in t for t in texts), f"XRAM[0xe179] not found: {texts}"
        assert any(f"XRAM[0x{K+1:x}]" in t for t in texts), f"XRAM[0xe17a] not found: {texts}"
        assert any(f"XRAM[0x{K+2:x}]" in t for t in texts), f"XRAM[0xe17b] not found: {texts}"
        assert any(f"XRAM[0x{K+3:x}]" in t for t in texts), f"XRAM[0xe17c] not found: {texts}"
        assert any(f"XRAM[0x{K+4:x}]" in t for t in texts), f"XRAM[0xe17d] not found: {texts}"


class TestNameLhsPropagation:
    """Tests for Name-lhs (TypedAssign) single-use propagation into call args."""

    def test_name_assign_folded_into_call(self):
        """TypedAssign(Name('arg1'), Name('xarg1')) folded into cmp32(arg1, 0)."""
        from pseudo8051.ir.hir import TypedAssign
        nodes = [
            TypedAssign(0, "uint8_t", Name("arg1"), Name("xarg1")),
            ExprStmt(1, Call("cmp32", [Name("arg1"), Const(0)])),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        rendered = result[0].render()[0][1]
        assert "arg1 =" not in rendered
        assert "xarg1" in rendered

    def test_name_assign_folded_into_return(self):
        """TypedAssign(Name('n'), Name('val')) folded into ReturnStmt."""
        from pseudo8051.ir.hir import TypedAssign
        nodes = [
            TypedAssign(0, "uint8_t", Name("n"), Name("val")),
            ReturnStmt(1, Name("n")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert "val" in result[0].value.render()

    def test_name_assign_not_folded_when_blocked(self):
        """Intermediate write to xarg1 blocks folding of arg1=xarg1."""
        from pseudo8051.ir.hir import TypedAssign
        clobber = Assign(1, Name("xarg1"), Const(99))
        nodes = [
            TypedAssign(0, "uint8_t", Name("arg1"), Name("xarg1")),
            clobber,
            ExprStmt(2, Call("cmp32", [Name("arg1"), Const(0)])),
        ]
        result = _propagate_values(nodes, {})
        # xarg1 is written between arg1= and cmp32 → blocked
        assert len(result) == 3

    def test_name_assign_not_folded_when_two_uses(self):
        """arg1 used twice → not folded (single-use only for Name-lhs)."""
        from pseudo8051.ir.hir import TypedAssign
        nodes = [
            TypedAssign(0, "uint8_t", Name("arg1"), Name("xarg1")),
            ExprStmt(1, Call("foo", [Name("arg1"), Const(0)])),
            ExprStmt(2, Call("bar", [Name("arg1")])),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 3   # kept unchanged

    def test_name_assign_with_intermediary_c_assign(self):
        """arg1=xarg1; C=0; cmp32(arg1, 0) — C=0 doesn't block folding."""
        from pseudo8051.ir.hir import TypedAssign
        nodes = [
            TypedAssign(0, "uint8_t", Name("arg1"), Name("xarg1")),
            Assign(1, Reg("C"), Const(0)),
            ExprStmt(2, Call("cmp32", [Name("arg1"), Const(0)])),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 2
        rendered = result[1].render()[0][1]
        assert "xarg1" in rendered


class TestSubstFromRegExprs:
    """Tests for _subst_from_reg_exprs: annotation-driven expression substitution."""

    def _make_ann(self, reg_exprs: dict) -> NodeAnnotation:
        ann = NodeAnnotation()
        ann.reg_exprs = reg_exprs
        return ann

    def test_switch_subject_simplified_via_reg_exprs(self):
        """switch(A/3) with reg_exprs[A]=arg1*3 → switch(arg1)."""
        sw = SwitchNode(0, BinOp(Reg("A"), '/', Const(3)),
                        {0: [], 1: [], 2: []})
        sw.ann = self._make_ann({"A": BinOp(Name("arg1"), '*', Const(3))})
        result, changed = _subst_from_reg_exprs([sw])
        assert changed
        assert len(result) == 1
        assert result[0].subject.render() == "arg1"

    def test_no_ann_node_skipped(self):
        """Node with ann=None is left unchanged."""
        sw = SwitchNode(0, BinOp(Reg("A"), '/', Const(3)),
                        {0: [], 1: []})
        sw.ann = None
        result, changed = _subst_from_reg_exprs([sw])
        assert not changed
        assert result[0].subject.render() == "A / 3"

    def test_reg_non_typegroup_expr_substituted(self):
        """reg_exprs[A]=R6+1 where R6 is not a TypeGroup member → safe to substitute."""
        sw = SwitchNode(0, BinOp(Reg("A"), '/', Const(3)),
                        {0: [], 1: []})
        sw.ann = self._make_ann({"A": BinOp(Reg("R6"), '+', Const(1))})
        result, changed = _subst_from_reg_exprs([sw])
        assert changed
        assert result[0].subject.render() == "(R6 + 1) / 3"

    def test_typegroup_reg_expr_not_substituted(self):
        """reg_exprs[A]=R7+1 where R7 IS a TypeGroup member → not substituted."""
        from pseudo8051.passes.patterns._utils import TypeGroup
        sw = SwitchNode(0, BinOp(Reg("A"), '/', Const(3)),
                        {0: [], 1: []})
        ann = self._make_ann({"A": BinOp(Reg("R7"), '+', Const(1))})
        ann.reg_groups = [TypeGroup("arg1", "int8_t", ("R7",))]
        sw.ann = ann
        result, changed = _subst_from_reg_exprs([sw])
        assert not changed
        assert result[0].subject.render() == "A / 3"

    def test_assign_rhs_simplified_via_reg_exprs(self):
        """Assign(lhs, A*2) with reg_exprs[A]=Name("arg1") → Assign(lhs, arg1*2)."""
        node = Assign(0, Reg("R6"), BinOp(Reg("A"), '*', Const(2)))
        node.ann = self._make_ann({"A": Name("arg1")})
        result, changed = _subst_from_reg_exprs([node])
        assert changed
        assert result[0].rhs.render() == "arg1 * 2"

    def test_binop_multi_reg_lhs_blocks_downstream_subst(self):
        """Multi-reg Assign with BinOp RHS adds to brace_written; stale annotation
        for those regs is blocked in a downstream node."""
        from pseudo8051.ir.expr import RegGroup
        # Node 0: R6R7 = osdAddr * 9  (computed, brace=False)
        mul_result = Assign(0x100, RegGroup(("R6", "R7")),
                            BinOp(Name("osdAddr"), "*", Const(9)))
        mul_result.ann = None
        # Node 1: A += R6  — stale ann says R6 = Name("staleVal")
        addc = CompoundAssign(0x101, Reg("A"), "+=", Reg("R6"))
        addc.ann = self._make_ann({"R6": Name("staleVal")})
        result, changed = _subst_from_reg_exprs([mul_result, addc])
        # R6 must NOT be substituted with staleVal
        assert result[1].rhs == Reg("R6"), (
            f"Expected R6 unchanged, got {result[1].rhs.render()!r}"
        )

    def test_simple_multi_reg_lhs_clears_brace_written(self):
        """Multi-reg Assign with simple RHS (Name) clears brace_written for those
        regs; a downstream annotation for them is now valid and should be applied."""
        from pseudo8051.ir.expr import RegGroup
        # Node 0: R6R7 = osdAddr * 9 — adds R6,R7 to brace_written
        mul_result = Assign(0x100, RegGroup(("R6", "R7")),
                            BinOp(Name("osdAddr"), "*", Const(9)))
        mul_result.ann = None
        # Node 1: R6R7 = xarg2  (simple load) — clears R6,R7 from brace_written
        load = Assign(0x102, RegGroup(("R6", "R7")), Name("xarg2"))
        load.ann = None
        # Node 2: A += R6  — annotation now fresh: R6 = Name("xarg2_hi")
        addc = CompoundAssign(0x103, Reg("A"), "+=", Reg("R6"))
        addc.ann = self._make_ann({"R6": Name("xarg2_hi")})
        result, changed = _subst_from_reg_exprs([mul_result, load, addc])
        assert changed
        assert result[2].rhs == Name("xarg2_hi"), (
            f"Expected xarg2_hi substituted, got {result[2].rhs.render()!r}"
        )
