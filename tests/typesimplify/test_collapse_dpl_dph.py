from pseudo8051.passes.typesimplify import _collapse_dpl_dph
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign, ExprStmt
from pseudo8051.ir.expr import Reg, Const


class TestCollapseDplDph:
    """Unit tests for _collapse_dpl_dph."""

    def test_basic(self):
        """DPH=R4; DPL=R0; → DPTR = R4R0;"""
        nodes = [
            Assign(0, Reg("DPH"), Reg("R4")),
            Assign(1, Reg("DPL"), Reg("R0")),
        ]
        result = _collapse_dpl_dph(nodes, {})
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].lhs.render() == "DPTR"
        assert result[0].rhs.render() == "R4R0"

    def test_with_varname(self):
        """DPH=R4; DPL=R0; with reg_map R4R0→_dest → DPTR = _dest;"""
        reg_map = {"R4R0": VarInfo("_dest", "uint16_t", ("R4", "R0"))}
        nodes = [
            Assign(0, Reg("DPH"), Reg("R4")),
            Assign(1, Reg("DPL"), Reg("R0")),
        ]
        result = _collapse_dpl_dph(nodes, reg_map)
        assert len(result) == 1
        assert result[0].rhs.render() == "_dest"

    def test_reverse_order(self):
        """DPL=R0; DPH=R4; → DPTR = R4R0; (DPH hi, DPL lo regardless of order)"""
        nodes = [
            Assign(0, Reg("DPL"), Reg("R0")),
            Assign(1, Reg("DPH"), Reg("R4")),
        ]
        result = _collapse_dpl_dph(nodes, {})
        assert len(result) == 1
        assert result[0].rhs.render() == "R4R0"

    def test_skips_setup_assigns(self):
        """DPH=R4; R5=1; DPL=R0; → [R5=1, DPTR=R4R0] (setup assign skippable)"""
        nodes = [
            Assign(0, Reg("DPH"), Reg("R4")),
            Assign(1, Reg("R5"), Const(1)),
            Assign(2, Reg("DPL"), Reg("R0")),
        ]
        result = _collapse_dpl_dph(nodes, {})
        assert len(result) == 2
        assert result[0].lhs.render() == "DPTR"
        assert result[1].lhs.render() == "R5"

    def test_blocked_by_non_setup(self):
        """DPH=R4; ExprStmt(DPTR); DPL=R0; → no collapse."""
        nodes = [
            Assign(0, Reg("DPH"), Reg("R4")),
            ExprStmt(1, Reg("DPTR")),
            Assign(2, Reg("DPL"), Reg("R0")),
        ]
        result = _collapse_dpl_dph(nodes, {})
        assert len(result) == 3
