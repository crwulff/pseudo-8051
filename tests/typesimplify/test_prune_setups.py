"""
Tests for _fold_and_prune_setups and _collect_hir_name_refs fixes.

Bug 1: CompoundAssign LHS not counted as a read by _collect_hir_name_refs,
       causing the preceding setup Assign to be incorrectly pruned.

Bug 2a: _fold_and_prune_setups recurse into IfNode branches without outer
        context, pruning assigns whose values are used AFTER the enclosing
        IfNode in the merge block.

Bug 2b: RegGroup(('R2','R3')) in call args must add 'R2' and 'R3' to refs
        (via individual names in Regs.reg_set()) so that a branch
        Assign(R3, ...) is preserved when the merge block uses RegGroup.

Bug 2c: _outer_refs must NOT be consulted when pruning DPTR++ nodes; a DPTR++
        value never flows across a control-flow merge, and passing outer refs
        incorrectly keeps orphaned DPTR++ nodes alive (leading to "EXT_E25B++").
"""
from pseudo8051.passes.typesimplify._post import (
    _fold_and_prune_setups,
    _collect_hir_name_refs,
    _is_dptr_inc_node,
)
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign, CompoundAssign, ExprStmt, IfNode
from pseudo8051.ir.expr import Reg, Regs, RegGroup, Name, Const, Call, UnaryOp


EA = 0  # dummy ea


class TestCollectHirNameRefsCompoundAssign:
    """Bug 1 — _collect_hir_name_refs must count CompoundAssign LHS as a read."""

    def test_compound_assign_lhs_counted(self):
        """CompoundAssign(A, '-=', Const(5)) must add 'A' to refs."""
        node = CompoundAssign(EA, Reg("A"), "-=", Const(5))
        refs = _collect_hir_name_refs([node])
        assert "A" in refs

    def test_compound_assign_rhs_still_counted(self):
        """CompoundAssign rhs register is also counted."""
        node = CompoundAssign(EA, Reg("A"), "-=", Reg("R1"))
        refs = _collect_hir_name_refs([node])
        assert "A" in refs
        assert "R1" in refs


class TestPruneSetupsCompoundAssignRead:
    """Bug 1 — setup Assign must NOT be pruned when downstream CompoundAssign reads its LHS."""

    def test_prune_setup_respects_compound_assign_read(self):
        """
        [Assign(A, Name('x')), CompoundAssign(A, '-=', Const(5))]
        → A is read by CompoundAssign; first node must survive.
        """
        nodes = [
            Assign(EA, Reg("A"), Name("x")),
            CompoundAssign(EA, Reg("A"), "-=", Const(5)),
        ]
        result = _fold_and_prune_setups(nodes, {})
        assert len(result) == 2
        assert isinstance(result[0], Assign)
        assert result[0].lhs.name == "A"

    def test_truly_dead_setup_still_pruned(self):
        """
        [Assign(A, Name('x')), ExprStmt(Call('f', [Reg('R0')]))]
        → A is not used downstream; Assign(A, ...) is pruned.
        """
        nodes = [
            Assign(EA, Reg("A"), Name("x")),
            ExprStmt(EA, Call("f", [Reg("R0")])),
        ]
        result = _fold_and_prune_setups(nodes, {})
        assert len(result) == 1
        assert isinstance(result[0], ExprStmt)


class TestPruneSetupsIfNodeScoping:
    """Bug 2 — assigns inside IfNode branches must not be pruned if used after the IfNode."""

    def _make_ifnode_with_else_assign(self, reg: str, val_name: str) -> IfNode:
        """IfNode with empty then-branch and else-branch [Assign(Reg(reg), Name(val_name))]."""
        return IfNode(
            EA,
            Reg("C"),
            [],
            [Assign(EA, Reg(reg), Name(val_name))],
        )

    def test_branch_assign_kept_when_ref_after_ifnode(self):
        """
        [IfNode(else=[Assign(R3, Name('font_base_hi'))]),
         ExprStmt(Call('set_osd_addr', [Reg('R3')]))]
        → R3 is read by the call after the IfNode; inner Assign must survive.
        """
        nodes = [
            self._make_ifnode_with_else_assign("R3", "font_base_hi"),
            ExprStmt(EA, Call("set_osd_addr", [Reg("R3")])),
        ]
        result = _fold_and_prune_setups(nodes, {})
        assert len(result) == 2
        ifnode = result[0]
        assert isinstance(ifnode, IfNode)
        assert len(ifnode.else_nodes) == 1
        inner = ifnode.else_nodes[0]
        assert isinstance(inner, Assign)
        assert inner.lhs.name == "R3"

    def test_branch_assign_pruned_when_truly_dead(self):
        """
        [IfNode(else=[Assign(R3, Name('font_base_hi'))]),
         ExprStmt(Call('f', []))]
        → R3 is NOT read after the IfNode; inner Assign should be pruned.
        """
        nodes = [
            self._make_ifnode_with_else_assign("R3", "font_base_hi"),
            ExprStmt(EA, Call("f", [])),
        ]
        result = _fold_and_prune_setups(nodes, {})
        assert len(result) == 2
        ifnode = result[0]
        assert isinstance(ifnode, IfNode)
        assert len(ifnode.else_nodes) == 0

    def test_branch_assign_kept_via_reggroup(self):
        """
        Bug 2b: merge call uses RegGroup(('R2','R3')); individual names 'R2' and
        'R3' must appear in refs so that R3=font_base_hi is preserved.
        """
        nodes = [
            self._make_ifnode_with_else_assign("R3", "font_base_hi"),
            ExprStmt(EA, Call("set_osd_addr", [Const(8), Name("_byte_sel"),
                                               RegGroup(("R2", "R3"))])),
        ]
        result = _fold_and_prune_setups(nodes, {})
        ifnode = result[0]
        assert isinstance(ifnode, IfNode)
        assert len(ifnode.else_nodes) == 1, "R3=font_base_hi must not be pruned"

    def test_collect_refs_includes_reggroup_components(self):
        """
        _collect_hir_name_refs must add each component name when a RegGroupExpr
        appears in a read position (individual regs, not the concatenated pair name).
        """
        node = ExprStmt(EA, Call("f", [RegGroup(("R2", "R3"))]))
        refs = _collect_hir_name_refs([node])
        assert "R2" in refs
        assert "R3" in refs
        assert "R2R3" not in refs


class TestPruneDptrInc:
    """Bug 2c — DPTR++ in a branch must be pruned on local refs only, never outer."""

    def _dptr_inc_node(self):
        return ExprStmt(EA, UnaryOp("++", Reg("DPTR")))

    def test_dptr_inc_pruned_locally_even_with_outer_dptr_ref(self):
        """
        IfNode(then=[DPTR++]) where merge block uses DPTR.
        The DPTR++ has no LOCAL downstream DPTR use → must be pruned.
        (Old outer-refs propagation incorrectly kept it alive.)
        """
        merge_uses_dptr = ExprStmt(EA, Call("f", [Reg("DPTR")]))
        nodes = [
            IfNode(EA, Reg("C"),
                   [self._dptr_inc_node()],  # then: DPTR++ with no local follow-up
                   []),
            merge_uses_dptr,
        ]
        result = _fold_and_prune_setups(nodes, {})
        assert isinstance(result[0], IfNode)
        assert len(result[0].then_nodes) == 0, "DPTR++ must be pruned (no local downstream)"

    def test_dptr_inc_kept_when_locally_needed(self):
        """DPTR++ with a downstream XRAM[DPTR] in the same branch must survive."""
        from pseudo8051.ir.expr import XRAMRef
        nodes = [
            IfNode(EA, Reg("C"),
                   [self._dptr_inc_node(),
                    Assign(EA, Reg("A"), XRAMRef(Reg("DPTR")))],
                   []),
        ]
        result = _fold_and_prune_setups(nodes, {})
        assert isinstance(result[0], IfNode)
        assert len(result[0].then_nodes) == 2, "DPTR++ must survive (A=XRAM[DPTR] follows)"


class TestConsolidateXramSingleLoad:
    """Regression: same register re-used with different XRAM locals in different scopes."""

    def test_single_load_scope_isolation(self):
        """
        IfNode(else=[R5=Name("font_base_hi"), R3=Reg("R5")]),
        R5=Name("_byte_sel")
        After _consolidate_xram_local_loads, R3 in the else-branch must
        be Name("font_base_hi"), not Name("_byte_sel").
        """
        from pseudo8051.passes.typesimplify._post import _consolidate_xram_local_loads
        from pseudo8051.ir.hir import Assign, IfNode

        reg_map = {
            "_font_sym": VarInfo("font_base_hi", "uint8_t", (), xram_sym="_font_sym"),
            "_bsel_sym": VarInfo("_byte_sel",    "uint8_t", (), xram_sym="_bsel_sym"),
        }
        nodes = [
            IfNode(EA, Reg("C"), [],
                   [Assign(EA, Reg("R5"), Name("font_base_hi")),
                    Assign(EA, Reg("R3"), Reg("R5"))]),
            Assign(EA, Reg("R5"), Name("_byte_sel")),
        ]
        result = _consolidate_xram_local_loads(nodes, dict(reg_map))
        ifnode = result[0]
        r3_assign = ifnode.else_nodes[1]
        assert isinstance(r3_assign.rhs, Name), (
            f"R3 rhs must be Name, not {type(r3_assign.rhs)}")
        assert r3_assign.rhs.name == "font_base_hi", (
            f"Expected 'font_base_hi' but got '{r3_assign.rhs.name}'")

    def test_single_load_subst_stops_at_reg_redef(self):
        """
        Regression: if the same register (A) is redefined after the single-load,
        _subst_reg_in_scope must NOT propagate past that redefinition.

        nodes = [A = Name("_arg1"),        ← single-load, triggers subst
                 A = XRAM[EXT_E25C],       ← redefines A — subst must stop here
                 Assign(Reg("dest"), Reg("A"))]  ← A should stay as Reg("A")
        """
        from pseudo8051.passes.typesimplify._post import _consolidate_xram_local_loads
        from pseudo8051.ir.hir import Assign
        from pseudo8051.ir.expr import XRAMRef, Reg as Reg_, Name as Name_

        reg_map = {
            "_arg1_sym": VarInfo("_arg1", "uint8_t", (), xram_sym="_arg1_sym"),
        }
        nodes = [
            Assign(EA, Reg("A"), Name("_arg1")),         # single-load → triggers subst
            Assign(EA, Reg("A"), XRAMRef(Reg("DPTR"))),  # redefines A — stop here
            Assign(EA, Reg("R1"), Reg("A")),              # A here is the NEW value
        ]
        result = _consolidate_xram_local_loads(nodes, dict(reg_map))
        # The third node must still have Reg("A") on the rhs, not Name("_arg1")
        r1_assign = result[2]
        assert isinstance(r1_assign, Assign)
        assert isinstance(r1_assign.rhs, Regs) and r1_assign.rhs.is_single, (
            f"R1 rhs must stay Reg('A'), not {type(r1_assign.rhs)}")
        assert r1_assign.rhs.name == "A"
