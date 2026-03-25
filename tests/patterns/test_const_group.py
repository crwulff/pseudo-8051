from pseudo8051.ir.hir import Statement
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.const_group import ConstGroupPattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestConstGroupPattern:
    def _pat(self):
        return ConstGroupPattern()

    def _u32_reg_map(self, name="dividend"):
        vinfo = VarInfo(name, "uint32_t", ("R4", "R5", "R6", "R7"))
        return {
            "R4R5R6R7": vinfo, "R4": vinfo, "R5": vinfo,
            "R6": vinfo, "R7": vinfo,
        }

    def test_u32_const_group(self):
        """5 stmts loading 0x00005dc0 into R4R5R6R7."""
        reg_map = self._u32_reg_map()
        nodes = [
            Statement(0, "A = 0x00;"),
            Statement(2, "R4 = A;"),
            Statement(4, "R5 = A;"),
            Statement(6, "R6 = 0x5d;"),
            Statement(8, "R7 = 0xc0;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 5
        assert len(replacement) == 1
        assert replacement[0].text == "uint32_t dividend = 0x00005dc0;"

    def test_u32_fold_into_call(self):
        """Const group immediately followed by return with the pair → fold."""
        reg_map = self._u32_reg_map()
        nodes = [
            Statement(0, "A = 0x00;"),
            Statement(2, "R4 = A;"),
            Statement(4, "R5 = A;"),
            Statement(6, "R6 = 0x5d;"),
            Statement(8, "R7 = 0xc0;"),
            Statement(10, "return func(R4R5R6R7);"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 6   # consumed 5 group stmts + 1 return
        assert replacement[0].text == "return func(0x00005dc0);"

    def test_u16_const_group(self):
        """2-byte constant into R6R7."""
        vinfo = VarInfo("val", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [
            Statement(0, "R6 = 0x12;"),
            Statement(2, "R7 = 0x34;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].text == "uint16_t val = 0x1234;"

    def test_no_fire_incomplete_group(self):
        """Only R6 loaded, R7 missing → no match."""
        vinfo = VarInfo("val", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [Statement(0, "R6 = 0x12;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_single_reg_var(self):
        """Single-byte VarInfo (regs < 2) → pattern skips it."""
        vinfo = VarInfo("x", "uint8_t", ("R7",))
        reg_map = {"R7": vinfo}
        nodes = [Statement(0, "R7 = 0x42;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None
