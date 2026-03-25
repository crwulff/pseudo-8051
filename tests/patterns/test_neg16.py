from pseudo8051.ir.hir import Statement
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.neg16 import Neg16Pattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestNeg16Pattern:
    def _pat(self):
        return Neg16Pattern()

    def _neg_nodes(self, r_lo="R7", r_hi="R6"):
        return [
            Statement(0,  "C = 0;"),
            Statement(2,  "A = 0;"),
            Statement(4,  f"A -= {r_lo} + C;"),
            Statement(6,  f"{r_lo} = A;"),
            Statement(8,  "A = 0;"),
            Statement(10, f"A -= {r_hi} + C;"),
            Statement(12, f"{r_hi} = A;"),
        ]

    def test_basic_neg16(self):
        """7-statement SUBB negation collapses to 'x = -x;'."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = self._neg_nodes()
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 7
        assert replacement[0].text == "x = -x;"

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
            Statement(0,  "C = 0;"),
            Statement(2,  "A = 0;"),
            Statement(4,  "A -= R7 + C;"),
            Statement(6,  "R6 = A;"),   # should be R7
            Statement(8,  "A = 0;"),
            Statement(10, "A -= R6 + C;"),
            Statement(12, "R6 = A;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None
