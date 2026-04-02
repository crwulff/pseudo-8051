"""
tests/typesimplify/test_fold_return.py — _fold_return_chains helper.
"""

import pytest
from pseudo8051.ir.hir import Assign, ReturnStmt, IfNode, SwitchNode
from pseudo8051.ir.expr import Reg, Const, BinOp, XRAMRef, Name
from pseudo8051.passes.typesimplify._post import _fold_return_chains


class TestFoldReturnChains:
    def test_basic_fold(self):
        """Assign(R7, expr); ReturnStmt(R7) → ReturnStmt(expr)"""
        expr = BinOp(XRAMRef(Name("SYM")), "&", Const(1))
        nodes = [
            Assign(0x1000, Reg("R7"), expr),
            ReturnStmt(0x1002, Reg("R7")),
        ]
        result = _fold_return_chains(nodes, ("R7",))
        assert len(result) == 1
        assert isinstance(result[0], ReturnStmt)
        assert result[0].value is expr

    def test_none_value_folded(self):
        """Assign(R7, expr); ReturnStmt(None) → ReturnStmt(expr)"""
        expr = Const(42)
        nodes = [
            Assign(0x1000, Reg("R7"), expr),
            ReturnStmt(0x1002),
        ]
        result = _fold_return_chains(nodes, ("R7",))
        assert len(result) == 1
        assert isinstance(result[0], ReturnStmt)
        assert result[0].value is expr

    def test_no_match_different_reg(self):
        """Assign(R6, expr); ReturnStmt(R7) — different reg, no fold."""
        nodes = [
            Assign(0x1000, Reg("R6"), Const(1)),
            ReturnStmt(0x1002, Reg("R7")),
        ]
        result = _fold_return_chains(nodes, ("R7",))
        assert len(result) == 2

    def test_no_match_non_adjacent(self):
        """Assign(R7, expr); stmt; ReturnStmt(R7) — not adjacent, no fold."""
        nodes = [
            Assign(0x1000, Reg("R7"), Const(1)),
            Assign(0x1002, XRAMRef(Name("x")), Reg("R7")),
            ReturnStmt(0x1004, Reg("R7")),
        ]
        result = _fold_return_chains(nodes, ("R7",))
        assert len(result) == 3

    def test_inside_if_then(self):
        """Fold inside if-then body."""
        expr = Const(5)
        inner = [
            Assign(0x2000, Reg("R7"), expr),
            ReturnStmt(0x2002, Reg("R7")),
        ]
        nodes = [
            IfNode(0x1000, BinOp(Reg("R5"), "!=", Const(0)), inner, []),
        ]
        result = _fold_return_chains(nodes, ("R7",))
        assert len(result) == 1
        assert isinstance(result[0], IfNode)
        then = result[0].then_nodes
        assert len(then) == 1
        assert isinstance(then[0], ReturnStmt)
        assert then[0].value is expr

    def test_inside_switch_case(self):
        """Fold inside a switch case body."""
        expr = Const(7)
        case_body = [
            Assign(0x2000, Reg("R7"), expr),
            ReturnStmt(0x2002, Reg("R7")),
        ]
        sw = SwitchNode(0x1000, Reg("R5"),
                        [([1], case_body)],
                        default_label=None, default_body=None)
        result = _fold_return_chains([sw], ("R7",))
        assert len(result) == 1
        assert isinstance(result[0], SwitchNode)
        _vals, body = result[0].cases[0]
        assert len(body) == 1
        assert isinstance(body[0], ReturnStmt)
        assert body[0].value is expr

    def test_inside_switch_default(self):
        """Fold inside switch default body."""
        expr = Const(99)
        default_body = [
            Assign(0x2000, Reg("R7"), expr),
            ReturnStmt(0x2002, Reg("R7")),
        ]
        sw = SwitchNode(0x1000, Reg("R5"),
                        [],
                        default_label=None, default_body=default_body)
        result = _fold_return_chains([sw], ("R7",))
        assert len(result) == 1
        sw_out = result[0]
        assert isinstance(sw_out, SwitchNode)
        body = sw_out.default_body
        assert len(body) == 1
        assert isinstance(body[0], ReturnStmt)
        assert body[0].value is expr

    def test_ea_from_assign(self):
        """Folded ReturnStmt uses ea from the Assign node."""
        nodes = [
            Assign(0xABCD, Reg("R7"), Const(3)),
            ReturnStmt(0xABCF, Reg("R7")),
        ]
        result = _fold_return_chains(nodes, ("R7",))
        assert result[0].ea == 0xABCD


class TestFoldReturnChainsMultiReg:
    """Tests for multi-register (16-bit) return folding."""

    # ── RegGroup fold ─────────────────────────────────────────────────────────

    def test_reggroup_fold(self):
        """Assign(RegGroup(R2,R1), expr); ReturnStmt(None) → ReturnStmt(expr)."""
        from pseudo8051.ir.expr import RegGroup
        expr = Name("some_ptr")
        nodes = [
            Assign(0x1000, RegGroup(("R2", "R1")), expr),
            ReturnStmt(0x1002),
        ]
        result = _fold_return_chains(nodes, ("R2", "R1"))
        assert len(result) == 1
        assert isinstance(result[0], ReturnStmt)
        assert result[0].value is expr

    def test_reggroup_fold_ea_from_assign(self):
        """Folded ReturnStmt uses ea from the Assign node."""
        from pseudo8051.ir.expr import RegGroup
        nodes = [
            Assign(0xDEAD, RegGroup(("R2", "R1")), Name("ptr")),
            ReturnStmt(0xDEAF),
        ]
        result = _fold_return_chains(nodes, ("R2", "R1"))
        assert result[0].ea == 0xDEAD

    # ── No single-reg fold for multi-reg returns ──────────────────────────────

    def test_no_single_reg_fold_for_multi_ret(self):
        """R2=expr; ReturnStmt(None) must NOT fold for a 2-register return."""
        nodes = [
            Assign(0x1000, Reg("R2"), Reg("R2")),   # self-assignment artifact
            ReturnStmt(0x1002),
        ]
        result = _fold_return_chains(nodes, ("R2", "R1"))
        # Should NOT get ReturnStmt(Reg("R2"))
        for node in result:
            if isinstance(node, ReturnStmt) and node.value is not None:
                assert node.value != Reg("R2"), "single-reg incorrectly folded"

    # ── Canonical pair expression from reg_map ────────────────────────────────

    def test_canon_from_reg_map_no_individual_assigns(self):
        """ReturnStmt(None) with no preceding ret_reg assigns → Name('dest') from reg_map."""
        from pseudo8051.passes.patterns._utils import VarInfo
        reg_map = {
            "R2R1": VarInfo("dest", "uint16_t", ("R2", "R1"), is_param=True),
            "R2":   VarInfo("dest", "uint16_t", ("R2", "R1"), is_param=True),
            "R1":   VarInfo("dest", "uint16_t", ("R2", "R1"), is_param=True),
        }
        nodes = [ReturnStmt(0x1000)]
        result = _fold_return_chains(nodes, ("R2", "R1"), reg_map)
        assert len(result) == 1
        ret = result[0]
        assert isinstance(ret, ReturnStmt)
        assert ret.value is not None
        assert ret.value.render() == "dest"

    def test_self_assign_stripped_then_canon(self):
        """R2=R2 (self-assign) before ReturnStmt(None) → stripped, return dest."""
        from pseudo8051.passes.patterns._utils import VarInfo
        reg_map = {
            "R2R1": VarInfo("dest", "uint16_t", ("R2", "R1"), is_param=True),
            "R2":   VarInfo("dest", "uint16_t", ("R2", "R1"), is_param=True),
            "R1":   VarInfo("dest", "uint16_t", ("R2", "R1"), is_param=True),
        }
        nodes = [
            Assign(0x1000, Reg("R2"), Reg("R2")),   # self-assignment artifact
            ReturnStmt(0x1002),
        ]
        result = _fold_return_chains(nodes, ("R2", "R1"), reg_map)
        assert len(result) == 1   # self-assign stripped
        ret = result[0]
        assert isinstance(ret, ReturnStmt)
        assert ret.value.render() == "dest"

    # ── All ret_regs assigned consecutively ───────────────────────────────────

    def test_all_regs_byte_field_collapse(self):
        """R2=Name('dest.hi'); R1=Name('dest.lo'); ReturnStmt → ReturnStmt(Name('dest'))."""
        nodes = [
            Assign(0x1000, Reg("R2"), Name("dest.hi")),
            Assign(0x1002, Reg("R1"), Name("dest.lo")),
            ReturnStmt(0x1004),
        ]
        result = _fold_return_chains(nodes, ("R2", "R1"))
        assert len(result) == 1
        ret = result[0]
        assert isinstance(ret, ReturnStmt)
        assert ret.value.render() == "dest"

    def test_all_regs_fall_back_to_reg_map(self):
        """R2=Const(1); R1=Const(2); ReturnStmt → fallback to reg_map pair name."""
        from pseudo8051.passes.patterns._utils import VarInfo
        reg_map = {"R2R1": VarInfo("retval", "uint16_t", ("R2", "R1"))}
        nodes = [
            Assign(0x1000, Reg("R2"), Const(1)),
            Assign(0x1002, Reg("R1"), Const(2)),
            ReturnStmt(0x1004),
        ]
        result = _fold_return_chains(nodes, ("R2", "R1"), reg_map)
        assert len(result) == 1
        ret = result[0]
        assert isinstance(ret, ReturnStmt)
        assert ret.value is not None   # some expression from pair/reg_map

    def test_all_regs_inside_if(self):
        """Multi-reg fold recurses into IfNode bodies."""
        inner = [
            Assign(0x2000, Reg("R2"), Name("dest.hi")),
            Assign(0x2002, Reg("R1"), Name("dest.lo")),
            ReturnStmt(0x2004),
        ]
        nodes = [IfNode(0x1000, BinOp(Reg("R5"), "!=", Const(0)), inner, [])]
        result = _fold_return_chains(nodes, ("R2", "R1"))
        assert len(result) == 1
        then = result[0].then_nodes
        assert len(then) == 1
        assert isinstance(then[0], ReturnStmt)
        assert then[0].value.render() == "dest"
