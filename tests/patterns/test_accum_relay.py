from pseudo8051.ir.hir import Assign
from pseudo8051.ir.expr import Reg, Name, XRAMRef
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.accum_relay import AccumRelayPattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestAccumRelayPattern:
    def _pat(self):
        return AccumRelayPattern()

    def test_basic_no_subst(self):
        """A = R7; XRAM[sym] = A; → XRAM[sym] = R7; (no reg_map)"""
        nodes = [
            Assign(0, Reg("A"), Reg("R7")),
            Assign(2, XRAMRef(Name("sym")), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 2
        assert len(replacement) == 1
        assert replacement[0].render(0)[0][1] == "XRAM[sym] = R7;"

    def test_with_param_substitution(self):
        """A = R7; XRAM[sym] = A; with R7=H param → XRAM[sym] = H;"""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        nodes = [
            Assign(0, Reg("A"), Reg("R7")),
            Assign(2, XRAMRef(Name("sym")), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "XRAM[sym] = H;"

    def test_no_fire_second_not_a_read(self):
        """Second statement doesn't read A → pattern must not fire."""
        nodes = [
            Assign(0, Reg("A"), Reg("R7")),
            Assign(2, XRAMRef(Name("sym")), Reg("R6")),
        ]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_trivial_a_eq_a(self):
        """A = A; ... → skip (rhs is A)."""
        nodes = [
            Assign(0, Reg("A"), Reg("A")),
            Assign(2, XRAMRef(Name("sym")), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_target_is_a(self):
        """A = R7; A = A; → skip (lhs of second is A)."""
        nodes = [
            Assign(0, Reg("A"), Reg("R7")),
            Assign(2, Reg("A"), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_only_one_node(self):
        """Only one node → no pair to collapse."""
        nodes = [Assign(0, Reg("A"), Reg("R7"))]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_ea_preserved(self):
        """Output Assign carries the ea of the first node."""
        nodes = [
            Assign(0x1234, Reg("A"), Reg("R6")),
            Assign(0x1236, Reg("R7"), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is not None
        assert result[0][0].ea == 0x1234

    def test_with_pair_substitution(self):
        """Target uses a reg pair that gets substituted downstream."""
        vinfo = VarInfo("myvar", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        # A = R5; R6 = A; → R6 = R5;  (LHS substitution happens downstream)
        nodes = [
            Assign(0, Reg("A"), Reg("R5")),
            Assign(2, Reg("R6"), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "R6 = R5;"
