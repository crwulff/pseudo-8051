from pseudo8051.ir.hir import Statement
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.accum_relay import AccumRelayPattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestAccumRelayPattern:
    def _pat(self):
        return AccumRelayPattern()

    def test_basic_no_subst(self):
        """A = R7; XRAM[sym] = A; → XRAM[sym] = R7; (no reg_map)"""
        nodes = [Statement(0, "A = R7;"), Statement(2, "XRAM[sym] = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 2
        assert len(replacement) == 1
        assert replacement[0].text == "XRAM[sym] = R7;"

    def test_with_param_substitution(self):
        """A = R7; XRAM[sym] = A; with R7=H param → XRAM[sym] = H;"""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        nodes = [Statement(0, "A = R7;"), Statement(2, "XRAM[sym] = A;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].text == "XRAM[sym] = H;"

    def test_no_fire_second_not_a_read(self):
        """Second statement doesn't read A → pattern must not fire."""
        nodes = [Statement(0, "A = R7;"), Statement(2, "XRAM[sym] = R6;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_trivial_a_eq_a(self):
        """A = A; ... → skip (expr == 'A')."""
        nodes = [Statement(0, "A = A;"), Statement(2, "XRAM[sym] = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_target_is_a(self):
        """A = R7; A = A; → skip (target is 'A')."""
        nodes = [Statement(0, "A = R7;"), Statement(2, "A = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_only_one_node(self):
        """Only one node → no pair to collapse."""
        nodes = [Statement(0, "A = R7;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_ea_preserved(self):
        """Output Statement carries the ea of the first node."""
        nodes = [Statement(0x1234, "A = R6;"), Statement(0x1236, "R7 = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is not None
        assert result[0][0].ea == 0x1234

    def test_with_pair_substitution(self):
        """Target uses a reg pair that gets substituted."""
        vinfo = VarInfo("myvar", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        # A = R5; R6R7 = A; → R6R7 = R5; → myvar = R5;
        nodes = [Statement(0, "A = R5;"), Statement(2, "R6R7 = A;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].text == "myvar = R5;"
