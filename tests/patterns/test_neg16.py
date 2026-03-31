from pseudo8051.ir.hir import Assign, CompoundAssign
from pseudo8051.ir.expr import Reg, Const, BinOp
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.neg16 import Neg16Pattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestNeg16Pattern:
    def _pat(self):
        return Neg16Pattern()

    def _neg_nodes(self, r_lo="R7", r_hi="R6"):
        return [
            Assign(0,  Reg("C"), Const(0)),
            Assign(2,  Reg("A"), Const(0)),
            CompoundAssign(4,  Reg("A"), "-=", BinOp(Reg(r_lo), "+", Reg("C"))),
            Assign(6,  Reg(r_lo), Reg("A")),
            Assign(8,  Reg("A"), Const(0)),
            CompoundAssign(10, Reg("A"), "-=", BinOp(Reg(r_hi), "+", Reg("C"))),
            Assign(12, Reg(r_hi), Reg("A")),
        ]

    def test_basic_neg16(self):
        """7-node SUBB negation collapses to 'x = -x;'."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = self._neg_nodes()
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 7
        assert isinstance(replacement[0], Assign)
        assert replacement[0].render()[0][1] == "x = -x;"

    def test_no_fire_wrong_var_order(self):
        """lo and hi from different VarInfo → no match."""
        vlo = VarInfo("a", "uint8_t", ("R7",))
        vhi = VarInfo("b", "uint8_t", ("R6",))
        reg_map = {"R7": vlo, "R6": vhi}
        nodes = self._neg_nodes()
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_too_few_nodes(self):
        """Fewer than 7 nodes → no match."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = self._neg_nodes()[:6]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_mismatched_store(self):
        """Store register doesn't match SUBB source → no match."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = [
            Assign(0,  Reg("C"), Const(0)),
            Assign(2,  Reg("A"), Const(0)),
            CompoundAssign(4,  Reg("A"), "-=", BinOp(Reg("R7"), "+", Reg("C"))),
            Assign(6,  Reg("R6"), Reg("A")),   # should be R7
            Assign(8,  Reg("A"), Const(0)),
            CompoundAssign(10, Reg("A"), "-=", BinOp(Reg("R6"), "+", Reg("C"))),
            Assign(12, Reg("R6"), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None
